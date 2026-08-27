"""Stage 2N -- notes pass. Transcript to readable student notes.

    python scripts/stage2_notes.py --lecture 20260213-lecture11
    python scripts/stage2_notes.py --lecture <id> --window 560-1100   # iterate
    python scripts/stage2_notes.py --all

Reads:  data/<id>/transcript.json, data/<id>/math.json (optional),
        course/notation.md, course/glossary.json
Writes: data/<id>/notes.json  (+ refreshes lecture.json)

Design notes
------------
This is a summarising pass, not an extraction pass, so the windows are much
larger than Stage 2's: you cannot write a coherent note about a proof while
looking through a 90-second slot. Six minutes with a minute of overlap keeps
whole arguments intact.

Timestamps come from an `anchor_quote` located back in the word stream, the
same way Stage 2 derives spans. The model is never asked for a time.

Where Stage 2 has already converted a formula, that LaTeX is handed to this
pass verbatim so the notes inherit maths that has already been validated
rather than re-deriving it from the same garbled speech.
"""
from __future__ import annotations

import argparse
import difflib
import re
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
import merge  # noqa: E402
from llm import LLM  # noqa: E402
from stage2_math import build_windows, locate, course_context, window_text  # noqa: E402

KINDS = ["definition", "theorem", "proof", "example", "method", "remark", "admin"]

SCHEMA = {
    "type": "object",
    "properties": {
        "blocks": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "heading": {"type": "string"},
                    "kind": {"type": "string", "enum": KINDS},
                    "anchor_quote": {"type": "string"},
                    "body": {"type": "string"},
                    "key_points": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["heading", "kind", "anchor_quote", "body"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["blocks"],
    "additionalProperties": False,
}

SYSTEM = """\
You are a strong student taking notes in a university mathematics lecture.

You get a verbatim speech-recognition transcript of one stretch of the lecture,
and the LaTeX formulas already extracted from that same stretch.

Write the notes you would want to read a week before the exam: organised, much
shorter than the transcript, mathematics exact, waffle gone. You are writing
for a reader who has the audio available but does not want to listen to it
again.

Rules:
- Reply with JSON only.
- Split the stretch into 1 to 4 blocks, each covering one idea. Do not produce
  one block per sentence.
- heading: a short noun phrase naming the idea. Not a whole sentence.
- kind: one of definition, theorem, proof, example, method, remark, admin.
- anchor_quote: 5 to 12 words copied EXACTLY from the transcript, taken from
  where this block's material begins. This is what links the note to the
  audio, so copy it character for character and never invent or paraphrase it.
- body: the note itself, in light markdown:
    blank line between paragraphs
    "- " at the start of a line for a bullet, one bullet per line, never
      several bullets run together on the same line
    **bold** for emphasis
    $...$ for inline mathematics, $$...$$ for displayed mathematics
  Prefer the supplied LaTeX verbatim over re-deriving a formula yourself.
  Write in your own words -- explain the idea, do not transcribe the talking.
- key_points: 1 to 3 one-line takeaways. Omit for admin or chatter.

- Skip administrative talk, jokes and digressions that carry no mathematics.
  If the whole stretch is non-mathematical, return {"blocks": []}.
- Never invent mathematics that is not in the transcript. Where the transcript
  is garbled and you cannot tell what was meant, say so in one short phrase
  instead of guessing.
- Because you can see the whole lecture, you will recognise ideas that recur.
  Write about an idea where it is DEVELOPED in your window, not every time it
  is mentioned. Check the list of notes already written: if a heading there
  already covers what you are about to write, either go deeper on what is new
  in your window or leave it out. Do not restate the lecture's overall arc in
  every window, and make headings specific enough to tell two windows apart --
  "Bounding the remainder" beats a second "Proof one via Picard iteration".
- You may be given the FULL lecture transcript as background. It is there so
  you understand the argument your window sits inside -- what has been set up,
  and what it is building towards. It is NOT your assignment. Write only about
  what is spoken inside your window, and take every anchor_quote from the
  window text, never from the background.
- The notation and glossary sections are reference material. Never quote them
  in anchor_quote and never write a note about their examples. In particular,
  never produce a "notation", "conventions" or "definitions used" block that
  restates them: they are given to you so you can apply them silently, not so
  you can summarise them back. Only write about what the lecturer said.
"""


def math_for_window(items, t0: float, t1: float, limit: int = 40) -> str:
    """The already-validated LaTeX covering this stretch."""
    inside = [m for m in items if m["t_end"] >= t0 and m["t_start"] <= t1]
    if not inside:
        return ""
    lines = []
    for m in inside[:limit]:
        flag = "  (uncertain)" if m.get("ambiguous") or m.get("confidence", 1) < 0.6 else ""
        lines.append(f"- {m['latex']}{flag}")
    return ("## LaTeX already extracted from this stretch\n"
            "Reuse these verbatim where they apply.\n\n" + "\n".join(lines))


def full_transcript_block(words, marker_s: float = 120.0, max_chars: int = 120000):
    """The entire lecture as background, with coarse time markers.

    Returned identically for every window of a lecture, and placed first in the
    prompt, so the long prefix stays byte-identical between calls and the
    server's KV cache can be reused rather than re-read ten times.

    Markers every couple of minutes let the model locate its own window inside
    the whole, which is the point of handing it the lecture at all.
    """
    if not words:
        return "", False
    out, next_mark = [], 0.0
    for w in words:
        if w["start"] >= next_mark:
            out.append(f"[{common.hhmmss(w['start'])}]")
            while next_mark <= w["start"]:
                next_mark += marker_s
        out.append(w["w"])
    text = " ".join(out)
    if len(text) <= max_chars:
        return text, False
    # A very long lecture would blow the window; keep the head and tail so the
    # arc survives, and say so rather than silently dropping the middle.
    half = max_chars // 2
    gap = "\n\n[... middle of the lecture omitted for length ...]\n\n"
    return text[:half] + gap + text[-half:], True


def notes_so_far(blocks, max_chars: int = 30000) -> str:
    """Every note already written for this lecture, in full.

    The heading-only outline is enough to avoid repeating yourself, but not to
    continue an argument -- for that the model needs to see what it actually
    said. Trimmed from the front when it grows too long, because the recent
    notes matter more than the opening ones.
    """
    if not blocks:
        return ""
    rendered = []
    for b in blocks:
        part = ["### [" + common.hhmmss(b["t_start"]) + "] " + b["heading"], b["body"]]
        for k in (b.get("key_points") or []):
            part.append("- " + k)
        rendered.append("\n".join(part))

    out, total = [], 0
    for chunk in reversed(rendered):
        if total + len(chunk) > max_chars:
            break
        out.append(chunk)
        total += len(chunk)
    out.reverse()
    dropped = len(rendered) - len(out)
    head = ("## The notes you have written for this lecture so far" +
            (" (earliest " + str(dropped) + " omitted for length)" if dropped else "") +
            "\n" + "Continue from these. Do not write any of them again." + "\n" + "\n")
    return head + ("\n" + "\n").join(out)


def outline_so_far(blocks, limit: int = 12) -> str:
    if not blocks:
        return ""
    lines = [f"- {b['heading']}" for b in blocks[-limit:]]
    return ("## Notes already written earlier in this lecture\n"
            "Continue from these; do not repeat them.\n\n" + "\n".join(lines))


def build_prompt(win, ctx: str, maths: str, outline: str, background: str = "",
                 bg_label: str = "FULL LECTURE TRANSCRIPT") -> str:
    parts = []

    # Background first and unchanging, so the shared prefix is as long as
    # possible across the windows of one lecture and the server can reuse its
    # KV cache instead of re-reading the transcript ten times.
    if background:
        parts.append("=== " + bg_label + " -- BACKGROUND ONLY ===")
        parts.append("Background so you can see how your window fits the argument "
                     "around it. Do NOT write notes about any of it. Only the "
                     "window at the end of this prompt is yours.")
        parts.append(background)
        parts.append("=== END BACKGROUND ===")

    if ctx or outline:
        parts.append("=== REFERENCE ONLY -- do not write notes about this section ===")
        if ctx:
            parts.append(ctx)
        if outline:
            parts.append(outline)
        parts.append("=== END REFERENCE ===")
    if maths:
        parts.append(maths)

    parts.append(
        f"=== YOUR WINDOW: {common.hhmmss(win['t0'])} to {common.hhmmss(win['t1'])} "
        f"-- write notes on THIS and nothing else ===\n\n" + window_text(win))
    parts.append(
        f"Write the notes for the window above ({common.hhmmss(win['t0'])} to "
        f"{common.hhmmss(win['t1'])}). Use the background to understand it, but "
        f"every block you produce must describe material spoken inside the "
        f"window, and every anchor_quote must be copied from the window text.")
    return "\n\n".join(parts)


def clean_blocks(raw, win, logger, label, min_match: float):
    out, dropped = [], 0
    if not isinstance(raw, dict):
        return out, 1
    for b in raw.get("blocks", []) or []:
        if not isinstance(b, dict):
            dropped += 1
            continue
        heading = str(b.get("heading", "")).strip()
        body = str(b.get("body", "")).strip()
        quote = str(b.get("anchor_quote", "")).strip()
        if not heading or not body:
            dropped += 1
            continue

        kind = str(b.get("kind", "remark")).strip().lower()
        if kind not in KINDS:
            kind = "remark"

        located = locate(quote, win["words"]) if quote else None
        if located is None:
            t_start, ratio = win["t0"], 0.0
        else:
            t_start, _end, ratio = located

        anchored = ratio >= min_match
        if not anchored:
            logger.warning("%s: anchor quote for %r matched only %.0f%% of the "
                           "transcript; pinning the note to the window start",
                           label, heading[:40], ratio * 100)
            t_start = win["t0"]

        points = [str(p).strip() for p in (b.get("key_points") or []) if str(p).strip()]

        out.append({
            "heading": heading,
            "kind": kind,
            "t_start": round(float(t_start), 3),
            "t_end": round(float(win["t1"]), 3),   # refined once neighbours are known
            "body": body,
            "key_points": points[:3],
            "anchor_quote": quote,
            "anchored": anchored,
            "match_ratio": round(float(ratio), 3),
        })
    return out, dropped


def dedupe(blocks, tolerance: float = 90.0, body_similarity: float = 0.72):
    """Collapse repeats.

    Two passes. Near-in-time blocks about the same thing come from the window
    overlap. Far-apart blocks with the same heading are a different problem:
    with the whole lecture in view the model re-summarises the arc, so an
    identical heading anywhere in the lecture is suspect -- but only merged
    when the bodies agree too, because a lecture really can prove one theorem
    twice and both proofs deserve a note.
    """
    kept = []
    for b in sorted(blocks, key=lambda x: x["t_start"]):
        clash = None
        for k in kept:
            gap = abs(k["t_start"] - b["t_start"])
            near = gap <= tolerance
            a = re.sub(r"\W+", "", k["heading"].lower())
            c = re.sub(r"\W+", "", b["heading"].lower())
            ratio = difflib.SequenceMatcher(None, a, c).ratio()
            # Blocks anchored within a window-overlap of each other are the
            # same idea arriving twice, so near-synonym headings count there
            # even though they would not further apart. Measured on lecture 11
            # this merges "integrating factor trick" with "integrating factor
            # construction" 43s later, and leaves genuinely distinct
            # neighbours like "case epsilon = 0" / "case delta = 0" alone.
            if gap <= 45.0 and ratio > 0.6:
                clash = k
                break
            same_heading = a == c or ratio > 0.85
            if not same_heading:
                continue
            if near:
                clash = k
                break
            body_ratio = difflib.SequenceMatcher(
                None, k["body"][:600].lower(), b["body"][:600].lower()).ratio()
            if body_ratio >= body_similarity:
                clash = k
                break
        if clash is None:
            kept.append(b)
        elif len(b["body"]) > len(clash["body"]):
            kept[kept.index(clash)] = b
    kept.sort(key=lambda x: x["t_start"])

    for i, b in enumerate(kept):
        if i + 1 < len(kept):
            b["t_end"] = round(max(kept[i + 1]["t_start"], b["t_start"]), 3)
        b["id"] = f"n_{i + 1:03d}"
    return kept


def process_lecture(cfg, course_id, lecture_id, logger, force=False, window_range=None):
    d = common.lecture_dir(cfg, course_id, lecture_id, create=False)
    tpath = d / "transcript.json"
    if not tpath.exists():
        raise FileNotFoundError(f"{lecture_id}: no transcript.json -- run Stage 1 first")

    out_path = d / "notes.json"
    if common.stage_is_done(out_path, force) and not window_range:
        logger.info("%s -- notes.json exists, skipping (use --force)", lecture_id)
        return None

    transcript = common.read_json(tpath)
    words = transcript.get("words", [])
    if not words:
        raise ValueError(f"{lecture_id}: transcript has no words")

    mpath = d / "math.json"
    math_items = common.read_json(mpath).get("items", []) if mpath.exists() else []
    if math_items:
        logger.info("%s: reusing %d validated formula(s) from Stage 2",
                    lecture_id, len(math_items))
    else:
        logger.info("%s: no math.json -- notes will render maths the model writes itself",
                    lecture_id)

    n = cfg.get("notes", {})
    windows = build_windows(words, float(n.get("window_s", 360)),
                            float(n.get("overlap_s", 60)))
    if window_range:
        lo, hi = window_range
        windows = [w for w in windows if w["t1"] >= lo and w["t0"] <= hi]
    logger.info("%s: %d window(s) of %.0fs", lecture_id, len(windows),
                float(n.get("window_s", 360)))

    llm = LLM(cfg, logger)
    ctx = course_context(cfg, course_id)
    min_match = float(n.get("min_anchor_match", 0.35))
    max_tokens = int(n.get("max_tokens", 6000))

    mode = str(n.get("context_mode", "neighbours")).lower()
    background, truncated = "", False
    if mode == "full":
        background, truncated = full_transcript_block(
            words,
            marker_s=float(n.get("background_marker_s", 120)),
            max_chars=int(n.get("max_context_chars", 120000)))
        logger.info("%s: background = whole transcript, %d chars (~%d tokens)%s",
                    lecture_id, len(background), len(background) // 4,
                    " [middle omitted]" if truncated else "")
    elif mode == "neighbours":
        logger.info("%s: background = the window either side", lecture_id)
    elif mode == "preceding":
        logger.info("%s: background = everything before each window, plus the "
                    "notes written for it", lecture_id)

    blocks, failed = [], 0
    for i, win in enumerate(windows, 1):
        label = f"{lecture_id} [{common.hhmmss(win['t0'])}]"
        bg = background
        outline = outline_so_far(blocks)
        bg_label = "FULL LECTURE TRANSCRIPT"

        if mode == "preceding":
            # Only what has already happened. There is no later material in
            # reach, so the model cannot write about a proof it has not got to.
            bg, _ = full_transcript_block(
                [w for w in words if w["start"] < win["t0"]],
                marker_s=float(n.get("background_marker_s", 120)),
                max_chars=int(n.get("max_context_chars", 120000)))
            bg_label = "THE LECTURE SO FAR"
            outline = notes_so_far(blocks, int(n.get("max_notes_context_chars", 30000)))

        if mode == "neighbours":
            # Just the minutes either side. Enough to see what the lecturer was
            # setting up and where it lands, without putting the whole lecture
            # in reach for the model to wander into.
            pad = float(n.get("neighbour_seconds", 360))
            bg, _ = full_transcript_block(
                [w for w in words
                 if win["t0"] - pad <= w["start"] <= win["t1"] + pad],
                marker_s=float(n.get("background_marker_s", 120)),
                max_chars=int(n.get("max_context_chars", 120000)))

        prompt = build_prompt(win, ctx,
                              math_for_window(math_items, win["t0"], win["t1"]),
                              outline, bg, bg_label)
        data = llm.json_call(SYSTEM, prompt, SCHEMA, "lecture_notes",
                             max_tokens=max_tokens, retries=1, label=label)
        if data is None:
            failed += 1
            continue
        got, dropped = clean_blocks(data, win, logger, label, min_match)
        blocks.extend(got)
        logger.info("  [%d/%d] %s  %d block(s)%s", i, len(windows),
                    common.hhmmss(win["t0"]), len(got),
                    f", {dropped} malformed" if dropped else "")

    blocks = dedupe(blocks)
    words_written = sum(len(b["body"].split()) for b in blocks)

    payload = {
        "lecture_id": lecture_id,
        "course_id": course_id,
        "blocks": blocks,
        "meta": {
            "model": cfg["llm"]["model"],
            "window_s": n.get("window_s", 360),
            "overlap_s": n.get("overlap_s", 60),
            "windows": len(windows),
            "windows_failed": failed,
            "used_math_items": len(math_items),
            "context_mode": mode,
            "background_chars": len(background),
            "background_truncated": truncated,
            "unanchored_blocks": sum(1 for b in blocks if not b["anchored"]),
            "note_words": words_written,
            "transcript_words": len(words),
            "compression": (round(len(words) / words_written, 1) if words_written else None),
            "llm": dict(llm.stats),
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        },
    }

    if window_range:
        logger.info("window range given -- not writing notes.json")
        return payload

    common.atomic_write_json(out_path, payload)
    logger.info("%s: wrote notes.json (%d blocks, %d words, %.1fx shorter than the "
                "transcript, %d window(s) failed)",
                lecture_id, len(blocks), words_written,
                payload["meta"]["compression"] or 0, failed)
    llm.log_stats()
    merge.merge_lecture(cfg, course_id, lecture_id, logger)
    return payload


def preview(payload, logger):
    print()
    for b in payload["blocks"]:
        print("=" * 96)
        flag = "" if b["anchored"] else "   [UNANCHORED]"
        print(f"[{common.hhmmss(b['t_start'])}] {b['kind'].upper()}  {b['heading']}{flag}")
        print()
        for line in b["body"].splitlines():
            print("   " + line)
        if b["key_points"]:
            print()
            for p in b["key_points"]:
                print("   * " + p)
        print()
    print("=" * 96)
    print(f"{len(payload['blocks'])} block(s), {payload['meta']['note_words']} words")


def main() -> int:
    ap = argparse.ArgumentParser(description="Stage 2N: write student notes from the transcript.")
    ap.add_argument("--course", required=True)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--lecture")
    g.add_argument("--all", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--window", help="only this time range, e.g. 600-1200 (prints, writes nothing)")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    cfg = common.load_config(args.config)
    logger = common.setup_logging("stage2-notes", cfg)
    report = common.BatchReport("Stage 2N (notes)")

    window_range = None
    if args.window:
        try:
            lo, hi = args.window.split("-")
            window_range = (float(lo), float(hi))
        except ValueError:
            logger.error("--window wants START-END in seconds, e.g. 600-1200")
            return 1

    ids = [args.lecture] if args.lecture else common.lecture_ids(cfg, args.course)
    if not ids:
        logger.error("nothing to do for course %s -- run Stage 1 first", args.course)
        return 1

    for lecture_id in ids:
        try:
            result = process_lecture(cfg, args.course, lecture_id, logger, args.force, window_range)
            if result is None:
                report.record_skip(lecture_id)
            else:
                report.record_ok(lecture_id)
                if window_range:
                    preview(result, logger)
        except KeyboardInterrupt:
            logger.warning("interrupted")
            report.record_fail(lecture_id, "interrupted")
            break
        except Exception as exc:
            logger.error("%s failed: %s", lecture_id, exc)
            logger.debug(traceback.format_exc())
            report.record_fail(lecture_id, f"{type(exc).__name__}: {exc}")

    return report.print_summary(logger)


if __name__ == "__main__":
    raise SystemExit(main())
