"""Stage 2C -- course notes. One model, one call, the whole lecture.

    python scripts/stage2_course_notes.py --course MAT267 --lecture 20260213-lecture11
    python scripts/stage2_course_notes.py --course MAT267 --all

Reads:  data/<course>/<id>/transcript.json, notes.json
        (+ math.json and structure.json when they exist)
        courses/<course>/notation.md, glossary.json
Writes: data/<course>/<id>/course_notes.json  (+ refreshes lecture.json)

Design notes
------------
Stage 2N writes good notes about six minutes of lecture at a time, and that
window is the ceiling on what it can do: it cannot open with what the lecture
turned out to be about, cannot fold the proof at 00:41 into the theorem at
00:12, and cannot drop a definition that the lecturer later replaced. Every
one of those needs the whole lecture in view at once.

So this stage does the opposite of every other LLM pass in the project. No
windows, no sweep, no reconciliation: the entire transcript and the entire set
of Stage 2N notes go into a single prompt, and one much stronger model --
reached through OpenRouter, the only stage here that leaves the machine --
writes the lecture up as one document.

The Stage 2N notes are passed in as well as the transcript, and are not
redundant: they carry the timestamps. Each section names the note ids it draws
on, which is how a written-up section that merges four minutes with a callback
twenty minutes later still gets an honest time to seek to. Where a section
names no usable note, its `anchor_quote` is located in the word stream the
same way every other stage derives a span, and where that fails too the
section is pinned after its predecessor and flagged rather than given a
timestamp nobody can defend.
"""
from __future__ import annotations

import argparse
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
import merge  # noqa: E402
from llm import LLM, LLMError  # noqa: E402
from stage2_math import locate, course_context  # noqa: E402
from stage2_notes import KINDS, full_transcript_block  # noqa: E402

SECTION_KINDS = ["overview"] + KINDS + ["summary"]

# Every property is listed in `required`. OpenAI-style strict structured
# outputs -- which is what OpenRouter forwards -- rejects a schema whose
# `required` is a subset of its properties, so optional fields are spelled as
# "always present, allowed to be empty" instead.
SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "summary": {"type": "string"},
        "prerequisites": {"type": "array", "items": {"type": "string"}},
        "sections": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "heading": {"type": "string"},
                    "kind": {"type": "string", "enum": SECTION_KINDS},
                    "covers": {"type": "array", "items": {"type": "string"}},
                    "anchor_quote": {"type": "string"},
                    "body": {"type": "string"},
                    "key_points": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["heading", "kind", "covers", "anchor_quote",
                             "body", "key_points"],
                "additionalProperties": False,
            },
        },
        "takeaways": {"type": "array", "items": {"type": "string"}},
        "open_questions": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["title", "summary", "prerequisites", "sections",
                 "takeaways", "open_questions"],
    "additionalProperties": False,
}

SYSTEM = """\
You are writing up one university mathematics lecture as course notes.

You are given, for a single lecture: the verbatim speech-recognition
transcript of the whole thing, the rough notes taken live during it, the
formulas already extracted from it, and the course's own notation.

The rough notes were written six minutes at a time by someone who could not
see the rest of the lecture. Your advantage is that you can. Use it. Produce
the write-up a good student makes afterwards, with the whole hour in front of
them: one document that has a beginning, states each idea once in the right
place, and can be read on its own by someone who was not there.

Rules:
- Reply with JSON only.
- title: what this lecture is about. A short noun phrase, not a sentence, and
  not "Lecture 11".
- summary: 2 to 4 sentences. What the lecture set out to do and what it
  established. This is the first thing the reader sees.
- prerequisites: the things a reader must already know to follow this. 0 to 6
  short items, named, not explained.

- sections: 4 to 14 of them, in READING order -- the order that makes the
  write-up coherent, which is usually but not always the order things were
  said. This is the heart of the document.
    heading: a short noun phrase naming the idea.
    kind: one of overview, definition, theorem, proof, example, method,
      remark, admin, summary.
    covers: the ids of the rough notes this section draws on, copied from the
      list you are given, e.g. ["n_004", "n_005"]. This is how the section is
      linked back to the audio, so get it right, list every note you actually
      used, and never invent an id. Use [] only for something you wrote from
      the transcript alone.
    anchor_quote: 5 to 12 words copied EXACTLY from the transcript, from where
      this section's material begins. A fallback for when `covers` is empty,
      so copy it character for character and never paraphrase it.
    body: the note itself, in light markdown:
        blank line between paragraphs
        "- " at the start of a line for a bullet, one bullet per line, never
          several bullets run together on the same line
        **bold** for emphasis
        $...$ for inline mathematics, $$...$$ for displayed mathematics
      Prefer the supplied LaTeX verbatim over re-deriving a formula yourself.
    key_points: 0 to 3 one-line takeaways for this section alone.

- takeaways: 3 to 6 one-line statements. What the reader should still know a
  month from now.
- open_questions: anything the lecturer explicitly left open, deferred to next
  time, or set as an exercise. [] if there was none. Do not invent these.

- Merge, do not concatenate. Where the rough notes cover one idea three times
  because it spans three windows, write it once. Where a proof is given twenty
  minutes after the theorem it proves, put them together and say when each was
  spoken. Where the lecturer said something and then corrected it, write the
  corrected version and say it was a correction.
- Where the rough notes and the transcript disagree, the transcript is what
  was actually said. The rough notes are a reading of it, and they can be
  wrong.
- Cut administrative talk, jokes and digressions unless they carry
  mathematics. An "admin" section is worth writing only for something like a
  changed deadline or an examinable/non-examinable ruling.
- Never invent mathematics that is not in the transcript. Where the transcript
  is garbled and you cannot tell what was meant, say so in one short phrase
  instead of guessing. Do not fill gaps with textbook material the lecturer
  did not cover.
- The notation and glossary sections are reference material. Apply them
  silently. Never quote them in anchor_quote and never write a section that
  just restates them.
"""


# --------------------------------------------------------------- prompt ----

def notes_block(blocks, limit_chars: int = 200000) -> str:
    """Every Stage 2N note, with its id and time, in full.

    The ids are the point: they are what the model hands back in `covers`, and
    therefore the only way a merged section gets a timestamp anyone can check.
    """
    if not blocks:
        return ""
    parts, total = [], 0
    for b in blocks:
        chunk = ["### " + b["id"] + "  [" + common.hhmmss(b["t_start"]) + "-"
                 + common.hhmmss(b["t_end"]) + "]  " + b["kind"] + ": " + b["heading"],
                 b["body"]]
        for k in (b.get("key_points") or []):
            chunk.append("- " + k)
        text = "\n".join(chunk)
        if total + len(text) > limit_chars:
            break
        parts.append(text)
        total += len(text)
    head = ("## The rough notes taken during this lecture\n"
            "Written six minutes at a time, so they repeat themselves and they "
            "stop and start. Cite the ids in `covers`.\n")
    return head + "\n" + "\n\n".join(parts)


def math_block(items, limit: int = 200) -> str:
    if not items:
        return ""
    lines = []
    for m in items[:limit]:
        flag = "  (uncertain)" if m.get("ambiguous") or m.get("confidence", 1) < 0.6 else ""
        lines.append("- [" + common.hhmmss(m["t_start"]) + "] " + m["latex"] + flag)
    return ("## LaTeX already extracted and checked for this lecture\n"
            "Reuse these verbatim where they apply.\n\n" + "\n".join(lines))


def chapters_block(chapters) -> str:
    if not chapters:
        return ""
    lines = []
    for ch in chapters:
        lines.append("- [" + common.hhmmss(ch.get("start", 0)) + "] "
                     + str(ch.get("title", "")) + " (" + str(ch.get("type", "")) + ")")
    return ("## How the lecture was divided into chapters\n"
            "One reading of the lecture's shape. You are free to disagree with "
            "it.\n\n" + "\n".join(lines))


def build_prompt(lecture_id, transcript, notes, maths, chapters, ctx, ids) -> str:
    parts = [
        "You are writing up lecture `" + lecture_id + "`.",
    ]
    if ctx:
        parts.append("=== REFERENCE ONLY -- do not write a section about this ===\n\n"
                     + ctx + "\n\n=== END REFERENCE ===")
    if chapters:
        parts.append(chapters)
    if maths:
        parts.append(maths)
    if notes:
        parts.append(notes)
    parts.append("## The full transcript of the lecture\n\n" + transcript)
    parts.append(
        "Write the course notes for this lecture. The note ids you may cite in "
        "`covers` are exactly: " + ", ".join(ids) + ". Every anchor_quote must "
        "be copied from the transcript above.")
    return "\n\n".join(parts)


# ---------------------------------------------------------------- clean ----

def clean_sections(raw, notes, words, logger, lecture_id, min_match: float,
                   duration: float):
    """Turn the model's sections into timed, id'd blocks.

    Time comes from the notes a section says it covers. That is a claim the
    model makes about its own work, so it is checked -- an id that is not in
    the lecture is dropped and counted, not quietly accepted.
    """
    by_id = {b["id"]: b for b in notes}
    out, dropped, bogus_ids = [], 0, 0
    # The end of the section immediately before, not the furthest point
    # reached so far: a merged section legitimately runs to the end of the
    # lecture, and letting that drag the fallback with it would strand the
    # next unanchored section in the closing minutes.
    prev_end = 0.0

    for s in (raw.get("sections") or []):
        if not isinstance(s, dict):
            dropped += 1
            continue
        heading = str(s.get("heading", "")).strip()
        body = str(s.get("body", "")).strip()
        if not heading or not body:
            dropped += 1
            continue

        kind = str(s.get("kind", "remark")).strip().lower()
        if kind not in SECTION_KINDS:
            kind = "remark"

        covers, unknown = [], []
        for cid in (s.get("covers") or []):
            cid = str(cid).strip()
            (covers if cid in by_id else unknown).append(cid)
        if unknown:
            bogus_ids += len(unknown)
            logger.warning("%s: section %r cites %d note id(s) this lecture does "
                           "not have (%s); ignoring them", lecture_id, heading[:40],
                           len(unknown), ", ".join(unknown[:4]))

        quote = str(s.get("anchor_quote", "")).strip()
        ratio, source = 0.0, "none"

        if covers:
            spans = [by_id[c] for c in covers]
            t_start = min(b["t_start"] for b in spans)
            t_end = max(b["t_end"] for b in spans)
            ratio, source = 1.0, "notes"
        else:
            # No usable note, so fall back to the transcript the same way
            # every other stage does.
            located = locate(quote, words) if quote else None
            if located and located[2] >= min_match:
                t_start, t_end, ratio = located[0], located[1], located[2]
                source = "quote"
            else:
                if located:
                    ratio = located[2]
                logger.warning("%s: section %r cites no note and its anchor quote "
                               "matched only %.0f%% of the transcript; pinning it "
                               "after the previous section",
                               lecture_id, heading[:40], ratio * 100)
                t_start = prev_end
                t_end = min(prev_end + 60.0, duration or prev_end + 60.0)

        points = [str(p).strip() for p in (s.get("key_points") or []) if str(p).strip()]

        out.append({
            "id": "c_%03d" % (len(out) + 1),
            "order": len(out),
            "heading": heading,
            "kind": kind,
            "t_start": round(float(t_start), 3),
            "t_end": round(float(max(t_end, t_start)), 3),
            "body": body,
            "key_points": points[:3],
            "covers": covers,
            "anchor_quote": quote,
            "anchored": source != "none",
            "time_source": source,
            "match_ratio": round(float(ratio), 3),
        })
        prev_end = float(max(t_end, t_start))

    return out, dropped, bogus_ids


def strings(raw, key, limit: int = 12):
    return [str(x).strip() for x in (raw.get(key) or []) if str(x).strip()][:limit]


# -------------------------------------------------------------- pipeline ----

def process_lecture(cfg, course_id, lecture_id, logger, force=False):
    d = common.lecture_dir(cfg, course_id, lecture_id, create=False)

    tpath = d / "transcript.json"
    if not tpath.exists():
        raise FileNotFoundError(f"{lecture_id}: no transcript.json -- run Stage 1 first")
    npath = d / "notes.json"
    if not npath.exists():
        raise FileNotFoundError(
            f"{lecture_id}: no notes.json -- this stage writes up the Stage 2N "
            f"notes, so run stage2_notes.py first")

    out_path = d / "course_notes.json"
    if common.stage_is_done(out_path, force):
        logger.info("%s -- course_notes.json exists, skipping (use --force)", lecture_id)
        return None

    transcript = common.read_json(tpath)
    words = transcript.get("words", [])
    if not words:
        raise ValueError(f"{lecture_id}: transcript has no words")
    duration = float(transcript.get("duration") or (words[-1]["end"] if words else 0))

    notes = common.read_json(npath).get("blocks", [])
    if not notes:
        raise ValueError(f"{lecture_id}: notes.json has no blocks -- rerun Stage 2N")

    mpath = d / "math.json"
    math_items = common.read_json(mpath).get("items", []) if mpath.exists() else []
    spath = d / "structure.json"
    chapters = common.read_json(spath).get("chapters", []) if spath.exists() else []

    cn = cfg.get("course_notes", {})
    section = str(cn.get("llm_section", "openrouter"))
    max_tokens = int(cn.get("max_tokens", 16000))
    min_match = float(cn.get("min_anchor_match", 0.35))

    body, truncated = full_transcript_block(
        words,
        marker_s=float(cn.get("transcript_marker_s", 60)),
        max_chars=int(cn.get("max_transcript_chars", 400000)))
    if truncated:
        logger.warning("%s: transcript is longer than course_notes."
                       "max_transcript_chars; the middle was dropped", lecture_id)

    prompt = build_prompt(lecture_id, body, notes_block(notes),
                          math_block(math_items), chapters_block(chapters),
                          course_context(cfg, course_id),
                          [b["id"] for b in notes])

    llm = LLM(cfg, logger, section=section)
    logger.info("%s: one call to %s via `%s` -- %d note(s), %d formula(s), "
                "%d chapter(s), %d prompt chars (~%d tokens)",
                lecture_id, llm.model, section, len(notes), len(math_items),
                len(chapters), len(prompt), len(prompt) // 4)

    data = llm.json_call(SYSTEM, prompt, SCHEMA, "course_notes",
                         max_tokens=max_tokens, retries=1, label=lecture_id)
    if data is None:
        raise RuntimeError(f"{lecture_id}: the model returned nothing usable twice")

    sections, dropped, bogus = clean_sections(
        data, notes, words, logger, lecture_id, min_match, duration)
    if not sections:
        raise RuntimeError(f"{lecture_id}: the model returned no usable sections")

    words_written = sum(len(s["body"].split()) for s in sections)
    note_words = sum(len(b["body"].split()) for b in notes)

    payload = {
        "lecture_id": lecture_id,
        "course_id": course_id,
        "title": str(data.get("title", "")).strip() or lecture_id,
        "summary": str(data.get("summary", "")).strip(),
        "prerequisites": strings(data, "prerequisites", 8),
        "sections": sections,
        "takeaways": strings(data, "takeaways", 8),
        "open_questions": strings(data, "open_questions", 8),
        "meta": {
            "model": llm.model,
            "llm_section": section,
            "source_notes": len(notes),
            "used_math_items": len(math_items),
            "used_chapters": len(chapters),
            "prompt_chars": len(prompt),
            "transcript_truncated": truncated,
            "malformed_sections": dropped,
            "unknown_note_ids": bogus,
            "unanchored_sections": sum(1 for s in sections if not s["anchored"]),
            "sections_from_quote": sum(1 for s in sections
                                       if s["time_source"] == "quote"),
            "note_words": words_written,
            "rough_note_words": note_words,
            "transcript_words": len(words),
            "compression": (round(len(words) / words_written, 1)
                            if words_written else None),
            "llm": dict(llm.stats),
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        },
    }

    common.atomic_write_json(out_path, payload)
    logger.info("%s: wrote course_notes.json (%d section(s), %d words, %.1fx "
                "shorter than the transcript, %d unanchored)",
                lecture_id, len(sections), words_written,
                payload["meta"]["compression"] or 0,
                payload["meta"]["unanchored_sections"])
    llm.log_stats()
    merge.merge_lecture(cfg, course_id, lecture_id, logger)
    return payload


def preview(payload):
    print()
    print("=" * 96)
    print(payload["title"])
    print("=" * 96)
    print(payload["summary"])
    if payload["prerequisites"]:
        print("\nAssumes: " + ", ".join(payload["prerequisites"]))
    for s in payload["sections"]:
        print()
        print("-" * 96)
        flag = "" if s["anchored"] else "   [UNANCHORED]"
        print("[%s] %s  %s%s" % (common.hhmmss(s["t_start"]), s["kind"].upper(),
                                 s["heading"], flag))
        print()
        for line in s["body"].splitlines():
            print("   " + line)
        for p in s["key_points"]:
            print("   * " + p)
    if payload["takeaways"]:
        print("\n" + "=" * 96)
        print("Takeaways")
        for t in payload["takeaways"]:
            print("  * " + t)
    if payload["open_questions"]:
        print("\nLeft open")
        for q in payload["open_questions"]:
            print("  ? " + q)
    print()


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Stage 2C: write up a whole lecture as course notes, in one call.")
    ap.add_argument("--course", required=True)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--lecture")
    g.add_argument("--all", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--print", dest="show", action="store_true",
                    help="print the document after writing it")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    cfg = common.load_config(args.config)
    logger = common.setup_logging("stage2-course-notes", cfg)
    report = common.BatchReport("Stage 2C (course notes)")

    ids = [args.lecture] if args.lecture else common.lecture_ids(cfg, args.course)
    if not ids:
        logger.error("nothing to do for course %s -- run Stage 1 first", args.course)
        return 1

    for lecture_id in ids:
        try:
            result = process_lecture(cfg, args.course, lecture_id, logger, args.force)
            if result is None:
                report.record_skip(lecture_id)
            else:
                report.record_ok(lecture_id)
                if args.show:
                    preview(result)
        except LLMError as exc:
            # A misconfigured endpoint fails identically for every lecture, so
            # there is nothing to gain from working through the rest of them.
            logger.error("%s", exc)
            report.record_fail(lecture_id, str(exc))
            break
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
