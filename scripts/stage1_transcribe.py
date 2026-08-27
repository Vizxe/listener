"""Stage 1 -- transcription.  PHASE A: unload LM Studio before running this.

    python scripts/stage1_transcribe.py --lecture algebra-07
    python scripts/stage1_transcribe.py --all
    python scripts/stage1_transcribe.py --all --force

Reads:  audio/<file>, course/notation.md, course/glossary.json
Writes: data/<lecture_id>/transcript.json  (+ refreshes lecture.json for the viewer)

One lecture failing never stops the batch; failures are listed at the end.
"""
from __future__ import annotations

import argparse
import re
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
import merge  # noqa: E402
import preprocess as pp  # noqa: E402
from asr_backends import get_backend  # noqa: E402


def rebuild_segments_from_words(words, segments, cfg, logger):
    """Re-cut segments so their times come from the word stream.

    OpenVINO produces segments and words by two different mechanisms: segment
    bounds come from Whisper's timestamp tokens, word bounds from
    cross-attention DTW. The word times are accurate -- cutting the audio at
    words[i].start reliably starts on that word -- but the segment times are
    not. On lecture 11, segment 0 claimed 0:30 for speech that begins at 0:41.

    That mismatch is what desyncs the viewer: a row's gutter timestamp and the
    words displayed in that row describe different moments, so clicking the
    gutter seeks somewhere the row is not. Re-cutting on sentence endings and
    speech pauses makes every row self-consistent -- its time, its text and its
    words all come from the same source.
    """
    if not words:
        return segments, {"resegmented": False}

    s = cfg["asr"].get("segmentation", {})
    max_s = float(s.get("max_seconds", 18.0))
    min_s = float(s.get("min_seconds", 2.0))
    gap_s = float(s.get("gap_seconds", 0.8))

    out, cur = [], []
    for idx, w in enumerate(words):
        cur.append(w)
        nxt = words[idx + 1] if idx + 1 < len(words) else None
        span = w["end"] - cur[0]["start"]
        ends_sentence = w["w"][-1:] in ".?!"
        gap = (nxt["start"] - w["end"]) if nxt else 0.0

        if (nxt is None
                or span >= max_s
                or (ends_sentence and span >= min_s)
                or (gap >= gap_s and span >= min_s)):
            out.append({
                "start": round(cur[0]["start"], 3),
                "end": round(w["end"], 3),
                "text": " ".join(x["w"] for x in cur),
            })
            cur = []

    if cur:
        out.append({
            "start": round(cur[0]["start"], 3),
            "end": round(cur[-1]["end"], 3),
            "text": " ".join(x["w"] for x in cur),
        })

    before = len(segments)
    logger.info("re-cut %d backend segment(s) into %d word-aligned segment(s)",
                before, len(out))
    segments[:] = out
    return segments, {"resegmented": True, "segments_before": before}


def strip_prompt_echo(words, segments, prompt, logger):
    """Drop stretches where the recogniser transcribed our own biasing prompt.

    Whisper fills unclear audio with whatever it was primed with. Lecture
    recordings open with room noise and chatter, so the first window is a prime
    candidate -- on lecture 11 it emitted 135 words of the vocabulary list
    before any speech started. `hotwords` raises the odds further, because it
    re-primes every window rather than only the first.

    Detection is exact rather than fuzzy: real speech does not reproduce fifty
    consecutive characters of the prompt, so a long common substring is
    conclusive and cannot fire on a lecturer who merely says "epsilon, delta".
    """
    if not prompt or not segments:
        return {"echo_segments": 0, "echo_words": 0}

    import difflib

    def norm(s):
        return re.sub(r"[^a-z0-9 ]+", " ", s.lower())

    def collapse(s):
        return re.sub(r"\s+", " ", s).strip()

    p = collapse(norm(prompt))
    spans, kept = [], []

    for seg in segments:
        s = collapse(norm(seg["text"]))
        if len(s) < 40:
            kept.append(seg)
            continue
        m = difflib.SequenceMatcher(None, s, p, autojunk=False)
        block = m.find_longest_match(0, len(s), 0, len(p))
        if block.size >= 50:
            spans.append((seg["start"], seg["end"]))
            logger.warning("dropping prompt echo at %s (%d chars matched the "
                           "biasing prompt)", common.hhmmss(seg["start"]), block.size)
        else:
            kept.append(seg)

    if not spans:
        return {"echo_segments": 0, "echo_words": 0}

    def inside(t):
        return any(a - 0.01 <= t <= b + 0.01 for a, b in spans)

    before = len(words)
    words[:] = [w for w in words if not inside(w["start"])]
    removed_words = before - len(words)

    removed_segs = len(segments) - len(kept)
    segments[:] = kept

    logger.warning("removed %d echoed segment(s) / %d word(s) -- the audio there "
                   "was too unclear to transcribe", removed_segs, removed_words)
    return {"echo_segments": removed_segs, "echo_words": removed_words}


def enforce_time_invariants(words, segments, duration, logger):
    """Guarantee the timeline properties every later stage assumes.

    Whisper decodes long audio in windows, and at a window seam the first word
    of the new window can start a fraction of a second before the last word of
    the old one ended. Left alone that makes `words` not sorted by time, which
    quietly breaks anything doing a binary search or a time-sliced window --
    the viewer's word lookup, and Stage 2's 90-second slicing.

    The fix nudges a start forward to keep the sequence non-decreasing. The
    shifts are sub-second and well under seek resolution.
    """
    stats = {"monotonicity_repairs": 0, "span_repairs": 0}

    prev = 0.0
    for w in words:
        if w["start"] < prev:
            w["start"] = prev
            stats["monotonicity_repairs"] += 1
        if w["end"] < w["start"]:
            w["end"] = w["start"]
            stats["span_repairs"] += 1
        prev = w["start"]

    for seg in segments:
        if seg["end"] < seg["start"]:
            seg["end"] = seg["start"]
            stats["span_repairs"] += 1

    if stats["monotonicity_repairs"]:
        logger.info("timeline: nudged %d word start(s) to keep the transcript "
                    "time-ordered (window-seam overlaps)",
                    stats["monotonicity_repairs"])
    if stats["span_repairs"]:
        logger.info("timeline: fixed %d negative-length span(s)", stats["span_repairs"])

    if words and duration > 0:
        coverage = words[-1]["end"] / duration
        stats["coverage"] = round(coverage, 4)
        if coverage < 0.9:
            logger.warning("transcript covers only %.0f%% of the audio timeline "
                           "-- possible truncation, check this lecture",
                           coverage * 100)
    return stats


def transcribe_one(audio: Path, course_id: str, lecture_id: str, cfg: dict, logger,
                   force: bool = False) -> Path:
    """Run Stage 1 for a single lecture. Returns the transcript.json path."""
    out_dir = common.lecture_dir(cfg, course_id, lecture_id)
    out_path = out_dir / "transcript.json"

    wav = out_dir / "audio16k.wav"
    wav = pp.preprocess(audio, wav, cfg, logger, force=force)
    duration = pp.probe_duration(wav)
    pp.viewer_audio(wav, audio, cfg, logger, force=force)
    logger.info("%s: %s of audio", lecture_id, common.hhmmss(duration))

    backend = get_backend(cfg["asr"]["backend"])
    initial_prompt = common.build_initial_prompt(cfg, course_id)

    t0 = time.time()
    result = backend.transcribe(wav, cfg, logger, initial_prompt)
    wall = time.time() - t0

    echo = strip_prompt_echo(
        result["words"], result["segments"], initial_prompt, logger)

    # Backends declare whether their segment times can be trusted. When they
    # cannot, re-cut from the word stream so the viewer's row times, row text
    # and word highlighting all agree. "always"/"never" override the backend.
    mode = str(cfg["asr"].get("segmentation", {}).get("rebuild_from_words", "auto")).lower()
    trusted = bool(result.get("meta", {}).get("segment_times_reliable", True))
    if mode == "always" or (mode == "auto" and not trusted):
        _, reseg = rebuild_segments_from_words(
            result["words"], result["segments"], cfg, logger)
    else:
        reseg = {"resegmented": False}

    timeline = enforce_time_invariants(
        result["words"], result["segments"], duration, logger)
    timeline.update(echo)
    timeline.update(reseg)

    transcript = {
        "lecture_id": lecture_id,
        "course_id": course_id,
        "duration": round(duration, 3),
        "words": result["words"],
        "segments": result["segments"],
        "meta": {
            **result.get("meta", {}),
            "timeline": timeline,
            "audio_file": audio.name,
            "initial_prompt": initial_prompt,
            "preprocess_filters": cfg.get("preprocess", {}).get("filters", ""),
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "wall_seconds": round(wall, 1),
            "realtime_factor": round(duration / wall, 2) if wall > 0 else None,
        },
    }
    common.atomic_write_json(out_path, transcript)

    logger.info("%s: wrote %s  (%d words, %d segments, %.2fx realtime)",
                lecture_id, out_path.name, len(result["words"]),
                len(result["segments"]), duration / wall if wall > 0 else 0)

    if not cfg.get("preprocess", {}).get("keep_wav", True):
        wav.unlink(missing_ok=True)

    merge.merge_lecture(cfg, course_id, lecture_id, logger)
    return out_path


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Stage 1: transcribe lecture audio to word-level JSON.")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--lecture", help="lecture id (audio filename stem, slugified)")
    g.add_argument("--all", action="store_true", help="every file in audio/")
    ap.add_argument("--course", required=True)
    ap.add_argument("--force", action="store_true", help="redo completed lectures")
    ap.add_argument("--limit", type=int, default=None, help="stop after N lectures")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    cfg = common.load_config(args.config)
    logger = common.setup_logging("stage1", cfg)
    report = common.BatchReport("Stage 1 (transcribe)")

    common.ensure_course(cfg, args.course)
    files = common.audio_files(cfg, args.course)
    if not files:
        logger.error("no audio found in %s", common.course_audio(cfg, args.course))
        logger.error("drop lecture files there and re-run.")
        return 1

    if args.lecture:
        audio = common.resolve_audio(cfg, args.course, args.lecture)
        if audio is None:
            logger.error("no audio matches lecture id %r", args.lecture)
            logger.error("available: %s",
                         ", ".join(common.lecture_id_for(p) for p in files))
            return 1
        targets = [audio]
    else:
        targets = files[: args.limit] if args.limit else files

    logger.info("backend=%s model=%s -- %d lecture(s) queued",
                cfg["asr"]["backend"], cfg["asr"]["model"], len(targets))

    for i, audio in enumerate(targets, 1):
        lecture_id = common.lecture_id_for(audio)
        out = common.lecture_dir(cfg, args.course, lecture_id) / "transcript.json"

        if common.stage_is_done(out, args.force):
            logger.info("[%d/%d] %s -- already done, skipping", i, len(targets), lecture_id)
            report.record_skip(lecture_id)
            continue

        logger.info("")
        logger.info("[%d/%d] %s  <- %s", i, len(targets), lecture_id, audio.name)
        try:
            transcribe_one(audio, args.course, lecture_id, cfg, logger, args.force)
            report.record_ok(lecture_id)
        except KeyboardInterrupt:
            logger.warning("interrupted by user")
            report.record_fail(lecture_id, "interrupted")
            break
        except Exception as exc:
            logger.error("%s failed: %s", lecture_id, exc)
            logger.debug(traceback.format_exc())
            report.record_fail(lecture_id, f"{type(exc).__name__}: {exc}")

    merge.write_lecture_list(cfg, args.course, logger)
    return report.print_summary(logger)


if __name__ == "__main__":
    raise SystemExit(main())
