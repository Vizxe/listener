"""OpenVINO GenAI backend -- runs Whisper on the Intel Arc GPU.

This is the fast path on this machine. CTranslate2 (faster-whisper) has no
Intel GPU backend, but OpenVINO does, and the Arc B580 reports FP16, INT8 and
GPU_HW_MATMUL support.

Two things it gives us that the CPU backend does not:
  * ~25x realtime on large-v3-turbo instead of a fraction of realtime
  * `hotwords`, which biases vocabulary on EVERY internal 30s window rather
    than only the first one the way Whisper's `initial_prompt` does. On a
    90-minute lecture that matters.

One thing it does not give us: per-word confidence. OpenVINO returns a single
sequence-level score, so `conf` is omitted from words and
`meta.has_word_confidence` is set to false. The viewer degrades gracefully.
"""
from __future__ import annotations

import time
import wave
from pathlib import Path

_PIPE_CACHE = {}


def _read_wav16k(path: Path):
    """Read the preprocessed WAV as mono float32 in [-1, 1].

    preprocess.py always emits 16 kHz mono pcm_s16le, so the stdlib is enough.
    librosa would also work but costs ~10s of import time and drags in numba
    and scipy for a job that is two lines of numpy.
    """
    import numpy as np

    with wave.open(str(path), "rb") as wf:
        channels, width, rate, frames = (
            wf.getnchannels(), wf.getsampwidth(), wf.getframerate(), wf.getnframes())
        raw = wf.readframes(frames)

    if width != 2:
        raise ValueError(
            f"{path.name}: expected 16-bit PCM, got {width * 8}-bit. "
            f"Let preprocess.py produce the file the ASR reads.")

    audio = np.frombuffer(raw, dtype="<i2").astype("float32") / 32768.0
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    if rate != 16000:
        raise ValueError(
            f"{path.name}: expected 16 kHz, got {rate} Hz. "
            f"Check preprocess.sample_rate in config.yaml.")
    return audio, rate


def _resolve_model(model: str, cfg: dict, logger) -> str:
    """Accept either a local directory or a HuggingFace repo id."""
    p = Path(model)
    if not p.is_absolute():
        candidate = Path(cfg["_root"]) / model
        if candidate.exists():
            return str(candidate)
    if p.exists():
        return str(p)

    from huggingface_hub import snapshot_download
    logger.info("fetching %s from HuggingFace (first run only)", model)
    return snapshot_download(model)


def _load_pipe(cfg: dict, logger):
    """Compiling the model takes ~20s, so keep it for the whole batch."""
    import openvino_genai as og

    a = cfg["asr"]
    o = a.get("openvino", {})
    device = o.get("device", "GPU")
    model = a["model"]
    key = (model, device)
    if key in _PIPE_CACHE:
        return _PIPE_CACHE[key]

    path = _resolve_model(model, cfg, logger)

    kwargs = {}
    if bool(a.get("word_timestamps", True)):
        # Must be set on the constructor: it decomposes the cross-attention
        # SDPA layers that the timestamps are derived from.
        kwargs["word_timestamps"] = True

    cache_dir = o.get("cache_dir")
    if cache_dir:
        cache = Path(cfg["_root"]) / cache_dir
        cache.mkdir(parents=True, exist_ok=True)
        kwargs["CACHE_DIR"] = str(cache)   # skips recompilation next run

    logger.info("compiling %s on %s ...", model, device)
    t0 = time.time()
    try:
        pipe = og.ASRPipeline(path, device, **kwargs)
    except AttributeError:
        pipe = og.WhisperPipeline(path, device, **kwargs)
    except Exception as exc:
        fallback = o.get("fallback_device")
        if not fallback:
            raise
        logger.warning("%s failed on %s (%s); falling back to %s",
                       model, device, type(exc).__name__, fallback)
        pipe = og.ASRPipeline(path, fallback, **kwargs)
        device = fallback

    logger.info("pipeline ready in %.1fs (device=%s)", time.time() - t0, device)
    _PIPE_CACHE[key] = (pipe, device)
    return _PIPE_CACHE[key]


def transcribe(wav: Path, cfg: dict, logger, initial_prompt: str = "") -> dict:
    a = cfg["asr"]
    o = a.get("openvino", {})
    pipe, device = _load_pipe(cfg, logger)

    audio, sr = _read_wav16k(wav)
    total = len(audio) / sr
    logger.info("audio %.1fs loaded at %d Hz", total, sr)

    if initial_prompt:
        logger.info("biasing prompt (%d chars): %s", len(initial_prompt), initial_prompt)
    else:
        logger.warning("no biasing prompt -- add terms to course/notation.md, "
                       "this is the highest-leverage ASR knob you have")

    gen = {
        "return_timestamps": True,
        "word_timestamps": bool(a.get("word_timestamps", True)),
        "task": "transcribe",
    }
    lang = a.get("language")
    if lang:
        gen["language"] = lang if lang.startswith("<|") else f"<|{lang}|>"
    if initial_prompt:
        gen["initial_prompt"] = initial_prompt
        # hotwords applies to every window, not just the first.
        if o.get("use_hotwords", True):
            gen["hotwords"] = initial_prompt

    t0 = time.time()
    res = pipe.generate(audio, **gen)
    wall = time.time() - t0
    logger.info("decoded in %s (%.2fx realtime)", _hms(wall),
                total / wall if wall > 0 else 0)

    segments = []
    for c in (res.chunks[0] if res.chunks else []):
        segments.append({
            "start": round(float(c.start_ts), 3),
            "end": round(float(c.end_ts), 3),
            "text": c.text.strip(),
        })

    # OpenVINO reports end_ts = -1 for an open-ended trailing chunk. Left
    # alone that lands a negative span in transcript.json, which would quietly
    # poison Stage 3 chapter bounds and any duration arithmetic downstream.
    repaired = 0
    for i, seg in enumerate(segments):
        if seg["end"] > seg["start"]:
            continue
        if i + 1 < len(segments):
            fallback = segments[i + 1]["start"]
        else:
            fallback = total
        seg["end"] = round(max(float(fallback), seg["start"]), 3)
        repaired += 1
    if repaired:
        logger.info("repaired %d segment end time(s) reported as open-ended", repaired)

    words = []
    for w in (res.words[0] if getattr(res, "words", None) else []):
        token = w.text.strip()
        if not token:
            continue
        # No `conf` key: OpenVINO has no per-word probability to report, and a
        # made-up number here would be worse than an absent one.
        words.append({
            "w": token,
            "start": round(float(w.start_ts), 3),
            "end": round(float(w.end_ts), 3),
        })

    if not words and segments:
        logger.warning("no word timestamps returned -- the viewer will fall back "
                       "to segment-level seeking for this lecture")

    # Timeline invariants (monotonicity, coverage) are enforced by Stage 1 for
    # every backend, so they are not re-checked here.
    langs = getattr(res, "languages", None)

    return {
        "words": words,
        "segments": segments,
        "meta": {
            "backend": "openvino",
            "model": a["model"],
            "device": device,
            "language": (langs[0] if langs else a.get("language")),
            "has_word_confidence": False,
            # Chunk times come from timestamp tokens and drift from the DTW
            # word times; Stage 1 re-cuts segments from the words.
            "segment_times_reliable": False,
            "used_hotwords": "hotwords" in gen,
            "asr_duration": round(total, 3),
        },
    }


def _hms(s: float) -> str:
    s = int(round(s))
    return f"{s // 3600:d}:{(s % 3600) // 60:02d}:{s % 60:02d}"
