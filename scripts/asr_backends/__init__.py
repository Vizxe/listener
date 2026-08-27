"""ASR backend registry.

Every backend exposes the same call:

    transcribe(wav: Path, cfg: dict, logger, initial_prompt: str) -> dict
        {"words":    [{"w","start","end"[,"conf"]}, ...],
         "segments": [{"start","end","text"}, ...],
         "meta":     {...backend-specific...}}

Stage 1 owns the file format; backends only produce these two lists. Swapping
the engine must never change transcript.json's shape.

`conf` is optional: OpenVINO has no per-word probability to report. Backends
that cannot supply it omit the key and set meta.has_word_confidence = False
rather than inventing a number.
"""
from __future__ import annotations

AVAILABLE = {
    # GPU (Intel Arc via OpenVINO). Fast. No per-word confidence.
    "openvino": "openvino_gpu",
    # CPU (CTranslate2). Slow, but reports per-word confidence.
    "faster_whisper": "faster_whisper_cpu",
}


def get_backend(name: str):
    if name not in AVAILABLE:
        raise ValueError(
            f"unknown asr.backend {name!r}. Available: {sorted(AVAILABLE)}")
    import importlib
    return importlib.import_module(f"asr_backends.{AVAILABLE[name]}")
