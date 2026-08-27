"""Stage 3 -- structure pass. Groups the lecture into chapters.

    python scripts/stage3_structure.py --lecture 20260213-lecture11
    python scripts/stage3_structure.py --all

Reads:  data/<id>/notes.json (preferred) or transcript.json, course/glossary.json
Writes: data/<id>/structure.json  (+ refreshes lecture.json)

Design notes
------------
The brief asked for one pass over the full transcript, reconciled into a single
output. That is what this does -- but it reads the *notes* rather than the raw
transcript, because Stage 2N has already compressed the lecture 3.4x and given
every block a heading, a kind and a time span. A 48-minute lecture becomes ~34
headings, which fits in one prompt comfortably, so there is no chunking to
reconcile and no risk of two halves disagreeing about where a topic starts.

It also keeps the chapter strip, the concept index and search consistent with
the notes the reader actually sees. Deriving chapters independently from the
transcript would let them drift apart.

The model never emits a timestamp. It groups note blocks by index, and the
chapter's span is taken from the first and last block it names.

Falls back to transcript segments when a lecture has no notes yet.
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
from llm import LLM  # noqa: E402

TYPES = ["admin", "motivation", "definition", "theorem", "proof", "example", "aside"]

SCHEMA = {
    "type": "object",
    "properties": {
        "chapters": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "type": {"type": "string", "enum": TYPES},
                    "first_block": {"type": "integer"},
                    "last_block": {"type": "integer"},
                    "summary": {"type": "string"},
                    "concepts": {"type": "array", "items": {"type": "string"}},
                    "assumes": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["title", "type", "first_block", "last_block",
                             "summary", "concepts", "assumes"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["chapters"],
    "additionalProperties": False,
}

SYSTEM = """\
You organise a university mathematics lecture into chapters.

You are given the lecture's notes as a numbered list of blocks, in order. Group
consecutive blocks into chapters: a chapter is one coherent stretch of the
lecture, typically three to eight blocks. Every block must belong to exactly
one chapter, chapters must not overlap, and together they must cover every
block from the first to the last with no gaps.

For each chapter:
- title: a short noun phrase naming the topic. Not a sentence.
- type: one of admin, motivation, definition, theorem, proof, example, aside.
  Pick what the chapter mostly is.
- first_block / last_block: the numbers of the first and last block in it.
- summary: two or three sentences. This is read on its own in a list of search
  results, by someone who has not seen the lecture, so it must stand alone.
  Say what was covered and what the result was -- not "this chapter discusses".
- concepts: the mathematical ideas this chapter is ABOUT. Lowercase noun
  phrases, singular, no articles: "uniform continuity", not "Uniform
  Continuities" or "the uniform continuity". Three to six per chapter. Only
  real mathematical concepts -- not "the lecturer", not "the proof".
- assumes: concepts the chapter USES but does not itself define or prove.
  These build a prerequisite graph, so they matter. Think about what a reader
  would already need to know to follow the chapter, including results proved
  in an earlier chapter of this same lecture: a chapter that proves a theorem
  a second way assumes that theorem's statement; a chapter applying an
  inequality assumes the inequality. Use the same lowercase singular form as
  concepts. Only the very first chapter is likely to have an empty list, so
  if you are about to return an empty list for a later chapter, look again.
  Never repeat a concept in both concepts and assumes for the same chapter.

Reply with JSON only.
"""


def blocks_from_notes(notes):
    return [{
        "t_start": b["t_start"], "t_end": b["t_end"],
        "heading": b["heading"], "kind": b["kind"],
        "gist": " ".join((b.get("key_points") or [])) or b["body"][:220],
    } for b in notes]


def blocks_from_segments(segments, group: int = 12):
    """Fallback when a lecture has no notes: bundle transcript segments."""
    out = []
    for i in range(0, len(segments), group):
        chunk = segments[i:i + group]
        text = " ".join(s["text"] for s in chunk)
        out.append({
            "t_start": chunk[0]["start"], "t_end": chunk[-1]["end"],
            "heading": text[:70], "kind": "transcript", "gist": text[:260],
        })
    return out


def render_blocks(blocks) -> str:
    lines = []
    for i, b in enumerate(blocks):
        lines.append(f"[{i}] {common.hhmmss(b['t_start'])} ({b['kind']}) "
                     f"{b['heading']}\n     {b['gist'][:220]}")
    return "\n".join(lines)


def clean_chapters(raw, blocks, aliases, logger, lecture_id):
    n = len(blocks)
    out = []
    for c in (raw or {}).get("chapters", []) or []:
        if not isinstance(c, dict):
            continue
        try:
            lo = int(c["first_block"])
            hi = int(c["last_block"])
        except (KeyError, TypeError, ValueError):
            continue
        lo, hi = max(0, min(lo, n - 1)), max(0, min(hi, n - 1))
        if hi < lo:
            lo, hi = hi, lo

        title = str(c.get("title", "")).strip()
        summary = str(c.get("summary", "")).strip()
        if not title:
            continue

        ctype = str(c.get("type", "aside")).strip().lower()
        if ctype not in TYPES:
            ctype = "aside"

        def norm_list(key):
            seen, vals = set(), []
            for x in (c.get(key) or []):
                v = common.normalize_concept(x, aliases)
                if v and v not in seen:
                    seen.add(v)
                    vals.append(v)
            return vals

        concepts = norm_list("concepts")
        assumes = [a for a in norm_list("assumes") if a not in concepts]

        out.append({
            "title": title,
            "start": round(float(blocks[lo]["t_start"]), 3),
            "end": round(float(blocks[hi]["t_end"]), 3),
            "type": ctype,
            "summary": summary,
            "concepts": concepts,
            "assumes": assumes,
            "_lo": lo, "_hi": hi,
        })

    out.sort(key=lambda x: (x["start"], x["_lo"]))

    # The brief wants one reconciled output: no overlaps, no gaps. The model
    # is asked for a clean partition but does not always deliver one, so the
    # boundaries are repaired here rather than trusted.
    overlaps, gaps = 0, 0
    for i, ch in enumerate(out):
        if i + 1 < len(out):
            nxt = out[i + 1]["start"]
            if ch["end"] > nxt:
                ch["end"] = nxt
                overlaps += 1
            elif ch["end"] < nxt:
                # The strip runs the width of the lecture, so a hole in it
                # would be a hole in the UI. Whatever was said in the gap
                # belongs to the chapter it followed.
                ch["end"] = nxt
                gaps += 1
        if ch["end"] < ch["start"]:
            ch["end"] = ch["start"]
    repaired = overlaps + gaps
    if out:
        out[0]["start"] = min(out[0]["start"], float(blocks[0]["t_start"]))
        out[-1]["end"] = max(out[-1]["end"], float(blocks[-1]["t_end"]))
    if repaired:
        logger.info("%s: closed %d gap(s) and trimmed %d overlap(s) so the "
                    "chapters partition the lecture", lecture_id, gaps, overlaps)

    covered = set()
    for ch in out:
        covered.update(range(ch["_lo"], ch["_hi"] + 1))
        del ch["_lo"], ch["_hi"]
    missing = n - len(covered)
    if missing:
        logger.warning("%s: %d note block(s) fell outside every chapter",
                       lecture_id, missing)

    for i, ch in enumerate(out, 1):
        ch["id"] = f"ch_{i:02d}"
    return out


def process_lecture(cfg, course_id, lecture_id, logger, force=False):
    d = common.lecture_dir(cfg, course_id, lecture_id, create=False)
    out_path = d / "structure.json"
    if common.stage_is_done(out_path, force):
        logger.info("%s -- structure.json exists, skipping (use --force)", lecture_id)
        return None

    npath, tpath = d / "notes.json", d / "transcript.json"
    if npath.exists():
        blocks = blocks_from_notes(common.read_json(npath).get("blocks", []))
        source = "notes"
    elif tpath.exists():
        blocks = blocks_from_segments(common.read_json(tpath).get("segments", []))
        source = "transcript segments"
    else:
        raise FileNotFoundError(f"{lecture_id}: nothing to structure -- run Stage 1 first")

    if not blocks:
        raise ValueError(f"{lecture_id}: no content to structure")
    logger.info("%s: %d block(s) from %s", lecture_id, len(blocks), source)

    aliases = common.glossary_aliases(cfg, course_id)
    llm = LLM(cfg, logger)

    # Left to itself the model swings between 2 and 8 chapters on the same
    # lecture. A chapter strip needs a predictable granularity, so the target
    # is computed from the running time and stated explicitly.
    scfg = cfg.get("structure", {})
    minutes = max(1.0, float(blocks[-1]["t_end"]) / 60.0)
    per = float(scfg.get("minutes_per_chapter", 7))
    target = int(max(int(scfg.get("min_chapters", 4)),
                     min(int(scfg.get("max_chapters", 12)), round(minutes / per))))

    prompt = (f"Lecture: {lecture_id}\n"
              f"Duration: {common.hhmmss(blocks[-1]['t_end'])}\n\n"
              f"Notes blocks, in order:\n\n{render_blocks(blocks)}\n\n"
              f"Group all {len(blocks)} blocks (0 to {len(blocks) - 1}) into chapters.\n\n"
              f"You have {len(blocks)} blocks and each chapter should hold "
              f"3 to 8 of them, so produce roughly {max(3, len(blocks) // 8)} to "
              f"{max(4, len(blocks) // 3)} chapters -- about {target} is right for a "
              f"lecture this length. Returning two or three chapters for "
              f"{len(blocks)} blocks means each one spans a third of the lecture, "
              f"which is too coarse to navigate or to search. Split a long "
              f"argument into its stages -- statement, first proof, second proof, "
              f"consequences -- rather than keeping it as one block.")

    data = llm.json_call(SYSTEM, prompt, SCHEMA, "lecture_structure",
                         max_tokens=int(cfg.get("structure", {}).get("max_tokens", 5000)),
                         retries=1, label=lecture_id)
    if data is None:
        raise RuntimeError("the model gave unusable output twice")

    chapters = clean_chapters(data, blocks, aliases, logger, lecture_id)
    if not chapters:
        raise RuntimeError("no usable chapters came back")

    all_concepts = sorted({c for ch in chapters for c in ch["concepts"]})
    payload = {
        "lecture_id": lecture_id,
        "course_id": course_id,
        "chapters": chapters,
        "meta": {
            "model": cfg["llm"]["model"],
            "source": source,
            "blocks": len(blocks),
            "chapters": len(chapters),
            "distinct_concepts": len(all_concepts),
            "llm": dict(llm.stats),
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        },
    }
    common.atomic_write_json(out_path, payload)
    logger.info("%s: wrote structure.json (%d chapters, %d distinct concepts)",
                lecture_id, len(chapters), len(all_concepts))
    merge.merge_lecture(cfg, course_id, lecture_id, logger)
    return payload


def main() -> int:
    ap = argparse.ArgumentParser(description="Stage 3: group a lecture into chapters.")
    ap.add_argument("--course", required=True)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--lecture")
    g.add_argument("--all", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    cfg = common.load_config(args.config)
    logger = common.setup_logging("stage3", cfg)
    report = common.BatchReport("Stage 3 (structure)")

    ids = [args.lecture] if args.lecture else common.lecture_ids(cfg, args.course)
    if not ids:
        logger.error("nothing to do for course %s -- run Stage 1 first", args.course)
        return 1

    for lecture_id in ids:
        try:
            if process_lecture(cfg, args.course, lecture_id, logger, args.force) is None:
                report.record_skip(lecture_id)
            else:
                report.record_ok(lecture_id)
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
