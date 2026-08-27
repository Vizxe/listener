"""faster-whisper backend.

CTranslate2 (what faster-whisper runs on) targets CPU and NVIDIA CUDA only --
there is no Intel Arc / SYCL backend, so on this machine `device` must stay
"cpu". That is slow but correct, and it gives native word-level timestamps
without a separate forced-alignment pass.

Speed knobs, in order of effect:
  asr.model                 large-v3 -> large-v3-turbo is ~5x faster
  asr.faster_whisper.compute_type   int8 is the fast one on CPU
  asr.beam_size             1 (greedy) is noticeably faster than 5
"""
from __future__ import annotations

import time
from pathlib import Path

_MODEL_CACHE = {}


def _load_model(cfg: dict, logger):
    """Load once per process; Stage 1 transcribes many lectures per run."""
    from faster_whisper import WhisperModel

    a = cfg["asr"]
    f = a.get("faster_whisper", {})
    key = (a["model"], f.get("device", "cpu"), f.get("compute_type", "int8"))
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]

    download_root = f.get("download_root")
    if download_root:
        download_root = str(Path(cfg["_root"]) / download_root)

    logger.info("loading %s (device=%s, compute_type=%s) -- first run downloads the model",
                key[0], key[1], key[2])
    t0 = time.time()
    model = WhisperModel(
        a["model"],
        device=f.get("device", "cpu"),
        compute_type=f.get("compute_type", "int8"),
        cpu_threads=int(f.get("cpu_threads", 8)),
        num_workers=int(f.get("num_workers", 1)),
        download_root=download_root,
    )
    logger.info("model ready in %.1fs", time.time() - t0)
    _MODEL_CACHE[key] = model
    return model


def transcribe(wav: Path, cfg: dict, logger, initial_prompt: str = "") -> dict:
    a = cfg["asr"]
    model = _load_model(cfg, logger)

    if initial_prompt:
        logger.info("initial_prompt (%d chars): %s", len(initial_prompt), initial_prompt)
    else:
        logger.warning("no initial_prompt -- add terms to course/notation.md, "
                       "this is the highest-leverage ASR knob you have")

    segments_iter, info = model.transcribe(
        str(wav),
        language=a.get("language") or None,
        beam_size=int(a.get("beam_size", 5)),
        vad_filter=bool(a.get("vad_filter", True)),
        word_timestamps=bool(a.get("word_timestamps", True)),
        initial_prompt=initial_prompt or None,
    )

    total = float(getattr(info, "duration", 0.0) or 0.0)
    logger.info("audio %.1fs, detected language %s (p=%.2f)",
                total, info.language, getattr(info, "language_probability", 0.0))

    words, segments = [], []
    t0 = time.time()
    last_log = 0.0

    # transcribe() returns a generator -- work happens as we iterate.
    for seg in segments_iter:
        segments.append({
            "start": round(float(seg.start), 3),
            "end": round(float(seg.end), 3),
            "text": seg.text.strip(),
        })
        for w in (seg.words or []):
            token = w.word.strip()
            if not token:
                continue
            words.append({
                "w": token,
                "start": round(float(w.start), 3),
                "end": round(float(w.end), 3),
                "conf": round(float(w.probability), 4),
            })

        # Progress matters here: a CPU run on a full lecture takes a while.
        if total and seg.end - last_log >= 300:
            last_log = seg.end
            elapsed = time.time() - t0
            speed = seg.end / elapsed if elapsed > 0 else 0
            remain = (total - seg.end) / speed if speed > 0 else 0
            logger.info("  %5.1f%%  audio %s / %s   %.2fx realtime   eta %s",
                        100 * seg.end / total,
                        _hms(seg.end), _hms(total), speed, _hms(remain))

    logger.info("decoded %d segments / %d words in %s",
                len(segments), len(words), _hms(time.time() - t0))

    return {
        "words": words,
        "segments": segments,
        "meta": {
            "backend": "faster_whisper",
            # Segments and words come from one decoding pass, so they agree.
            "segment_times_reliable": True,
            "model": a["model"],
            "device": a.get("faster_whisper", {}).get("device", "cpu"),
            "compute_type": a.get("faster_whisper", {}).get("compute_type", "int8"),
            "beam_size": int(a.get("beam_size", 5)),
            "vad_filter": bool(a.get("vad_filter", True)),
            "language": info.language,
            "language_probability": round(float(getattr(info, "language_probability", 0.0)), 4),
            "asr_duration": round(total, 3),
        },
    }


def _hms(s: float) -> str:
    s = int(round(s))
    return f"{s // 3600:d}:{(s % 3600) // 60:02d}:{s % 60:02d}"
