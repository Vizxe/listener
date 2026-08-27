"""Audio preprocessing: anything -> 16 kHz mono WAV, denoised and levelled.

Lecture hall recordings are noisy and the levels drift as the lecturer walks
around. Cleaning this up measurably improves the transcript, so it happens
before every ASR run.

Standalone use:
    python scripts/preprocess.py audio/algebra-07.mp3 --out /tmp/out.wav
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402


class PreprocessError(RuntimeError):
    pass


def _tool(name: str) -> str:
    exe = shutil.which(name)
    if not exe:
        raise PreprocessError(
            f"{name} not found on PATH. Install ffmpeg and reopen the shell.")
    return exe


def probe_duration(path: Path) -> float:
    """Duration in seconds, via ffprobe."""
    out = subprocess.run(
        [_tool("ffprobe"), "-v", "error", "-show_entries", "format=duration",
         "-of", "json", str(path)],
        capture_output=True, text=True, check=False)
    if out.returncode != 0:
        raise PreprocessError(f"ffprobe failed on {path.name}: {out.stderr.strip()[:300]}")
    try:
        return float(json.loads(out.stdout)["format"]["duration"])
    except (KeyError, ValueError, json.JSONDecodeError) as exc:
        raise PreprocessError(f"could not read duration of {path.name}: {exc}") from exc


def preprocess(src: Path, dst: Path, cfg: dict, logger=None, force: bool = False) -> Path:
    """Convert `src` to a 16 kHz mono WAV at `dst`. Returns `dst`.

    Skips the conversion when `dst` already exists and is newer than `src`,
    so re-running Stage 1 does not redo ffmpeg work.
    """
    pcfg = cfg.get("preprocess", {})
    sr = int(pcfg.get("sample_rate", 16000))
    dst.parent.mkdir(parents=True, exist_ok=True)

    if (not force and dst.exists() and dst.stat().st_size > 0
            and dst.stat().st_mtime >= src.stat().st_mtime):
        if logger:
            logger.info("preprocess: reusing %s", dst.name)
        return dst

    if not pcfg.get("enabled", True):
        filters = f"aresample={sr}"
    else:
        chain = str(pcfg.get("filters", "")).strip().rstrip(",")
        filters = f"{chain},aresample={sr}" if chain else f"aresample={sr}"

    cmd = [_tool("ffmpeg"), "-y", "-hide_banner", "-loglevel", "error",
           "-i", str(src),
           "-vn", "-sn", "-dn",
           "-ac", "1",
           "-af", filters,
           "-ar", str(sr),
           "-c:a", "pcm_s16le",
           str(dst)]

    if logger:
        logger.info("preprocess: %s -> %s", src.name, dst.name)
        logger.info("preprocess: filters = %s", filters)

    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0 or not dst.exists() or dst.stat().st_size == 0:
        raise PreprocessError(
            f"ffmpeg failed on {src.name}: {proc.stderr.strip()[:500]}")
    return dst


def viewer_audio(wav: Path, original: Path, cfg: dict, logger=None,
                 force: bool = False):
    """Produce the audio file the viewer plays. Returns the path, or None.

    The viewer must NOT play the original lecture MP3. Long VBR MP3s carry
    only a 100-entry Xing seek table, so a browser seeking into one lands at a
    byte offset it estimated from average bitrate. Measured on lecture 11,
    Chrome's seeks were out by 1.85s on average and 5.76s at worst -- the audio
    plays from somewhere other than the `currentTime` it reports, and since the
    highlight follows the reported clock, everything after a seek is wrong.

    Opus in Ogg carries granule positions, so seeking is sample-accurate. The
    same measurement puts it at 0.02s mean error, i.e. exact.
    """
    vcfg = cfg.get("preprocess", {}).get("viewer_audio", {})
    if not vcfg.get("enabled", True):
        return None

    ext = str(vcfg.get("ext", "opus"))
    dst = wav.parent / f"audio.{ext}"
    src = wav if vcfg.get("source", "preprocessed") == "preprocessed" else original

    if (not force and dst.exists() and dst.stat().st_size > 0
            and dst.stat().st_mtime >= src.stat().st_mtime):
        if logger:
            logger.info("viewer audio: reusing %s", dst.name)
        return dst

    cmd = [_tool("ffmpeg"), "-y", "-hide_banner", "-loglevel", "error",
           "-i", str(src), "-vn", "-ac", "1",
           "-c:a", str(vcfg.get("codec", "libopus")),
           "-b:a", str(vcfg.get("bitrate", "32k"))]
    cmd += [str(a) for a in vcfg.get("extra_args", ["-vbr", "on",
                                                   "-application", "voip"])]
    cmd += [str(dst)]

    if logger:
        logger.info("viewer audio: encoding %s (seek-accurate)", dst.name)
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0 or not dst.exists() or dst.stat().st_size == 0:
        # Not fatal: the viewer falls back to the original file, which plays
        # fine and only seeks badly.
        if logger:
            logger.warning("viewer audio encode failed, the viewer will fall back "
                           "to the original file (seeking will be imprecise): %s",
                           proc.stderr.strip()[:300])
        return None
    if logger:
        logger.info("viewer audio: %s (%.1f MB)", dst.name, dst.stat().st_size / 1e6)
    return dst


def main() -> int:
    ap = argparse.ArgumentParser(description="Convert audio to 16 kHz mono WAV for ASR.")
    ap.add_argument("src", help="input audio file")
    ap.add_argument("--out", required=True, help="output .wav path")
    ap.add_argument("--config", default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    cfg = common.load_config(args.config)
    logger = common.setup_logging("preprocess", cfg)
    src = Path(args.src).resolve()
    if not src.exists():
        logger.error("no such file: %s", src)
        return 1
    dst = preprocess(src, Path(args.out).resolve(), cfg, logger, args.force)
    logger.info("wrote %s (%.1f MB, %s)", dst, dst.stat().st_size / 1e6,
                common.hhmmss(probe_duration(dst)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
