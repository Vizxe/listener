"""Stage 2 -- math pass. Speech to LaTeX.  PHASE B: Whisper unloaded.

    python scripts/stage2_math.py --lecture 20260213-lecture11
    python scripts/stage2_math.py --lecture <id> --window 600-900   # prompt iteration
    python scripts/stage2_math.py --all

Reads:  data/<id>/transcript.json, course/notation.md, course/glossary.json
Writes: data/<id>/math.json  (+ refreshes lecture.json)

Design notes
------------
The transcript is immutable. Math is a parallel layer that references time
spans; nothing here ever rewrites transcript.json.

The model is never asked for timestamps. It is asked for `source_text` -- the
exact words it converted -- and Stage 2 locates that text back in the word
stream to derive the span. An LLM asked to copy a phrase is far more reliable
than an LLM asked to keep a clock, and a wrong quote is detectable where a
wrong timestamp is not.

Windows stay small (~90s) even though the context budget allows more, because
a 9B degrades on long extraction. The spare context goes on history instead:
recently extracted LaTeX is fed back so notation stays consistent across
window boundaries.
"""
from __future__ import annotations

import argparse
import difflib
import json
import re
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
import merge  # noqa: E402
from llm import LLM  # noqa: E402

KINDS = ["definition", "theorem", "expression", "step", "example"]

SCHEMA = {
    "type": "object",
    "properties": {
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "latex": {"type": "string"},
                    "kind": {"type": "string", "enum": KINDS},
                    "confidence": {"type": "number"},
                    "ambiguous": {"type": "boolean"},
                    "source_text": {"type": "string"},
                    "note": {"type": "string"},
                },
                "required": ["latex", "kind", "confidence", "ambiguous", "source_text"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["items"],
    "additionalProperties": False,
}

SYSTEM = """\
You convert spoken university mathematics into LaTeX.

You are given a verbatim speech-recognition transcript of a short stretch of a
lecture. It contains recognition errors and disfluencies. Transcribe the
mathematics that was spoken; do not correct, complete, or improve the maths.

Rules:
- Reply with JSON only.
- Extract each distinct mathematical statement. Ignore pure prose, admin talk
  and asides that contain no mathematics.
- latex: LaTeX for that statement, with no surrounding $ or \\[ delimiters.
- source_text: the exact words from the transcript that you converted, copied
  verbatim from the input. Never paraphrase it and never invent words: this is
  what locates the statement in time.
- kind: one of definition, theorem, expression, step, example.
- confidence: 0 to 1, how sure you are that the LaTeX matches what was said.
- ambiguous: true when the spoken words have more than one valid reading.
  "x squared plus one over x" can be x^2 + 1/x or (x^2+1)/x -- say it is
  ambiguous rather than silently picking one. Prefer the reading given in the
  course notation when one applies.
- note: optional. At most one short clause, and only when something genuinely
  needs flagging. Never explain your reasoning, never justify a choice, and
  never mention the reference section. Leave it out if in doubt.
- If the window contains no mathematics, return {"items": []}.

A Greek letter spoken by name is a LaTeX command, never a word: "tau" is
\\tau, "epsilon" is \\epsilon, "delta" is \\delta, "sigma" is \\sigma. Writing
g(tau) instead of g(\\tau) is wrong.

Emit one item per statement, not one per fragment. If a phrase yields a
complete statement, emit only that statement -- never also emit the pieces
inside it. For "y of t is at most 1", emit y(t) \\leq 1 and nothing else; do
not additionally emit y(t).

Never use \\text{...} as a placeholder for mathematics you could not make out.
If you cannot render a statement, leave it out entirely rather than emitting a
description of it.

Extract ONLY from the transcript window. The notation, glossary and
already-extracted sections are reference material telling you which
conventions to follow -- they are not lecture content. Never quote them in
source_text, and never emit an item for an example that appears in them. Every
source_text you return must be words the lecturer actually said in the window.
"""


# ------------------------------------------------------------------ windows --

def build_windows(words, window_s: float, overlap_s: float):
    """Overlapping time windows covering the whole transcript."""
    if not words:
        return []
    stride = max(1.0, window_s - overlap_s)
    end_t = words[-1]["end"]
    out, t = [], max(0.0, words[0]["start"])
    while t < end_t:
        lo, hi = t, t + window_s
        chunk = [w for w in words if w["start"] >= lo and w["start"] < hi]
        if chunk:
            out.append({"t0": lo, "t1": min(hi, end_t), "words": chunk})
        t += stride
    return out


def _norm_tokens(text: str):
    return [t for t in re.sub(r"[^a-z0-9\s]+", " ", text.lower()).split() if t]


def locate(source_text: str, words):
    """Find `source_text` inside the window's words. Returns (start, end, ratio).

    Uses the longest matching token run rather than an exact search, because
    the model normalises punctuation and casing when it quotes.
    """
    src = _norm_tokens(source_text)
    if not src or not words:
        return None
    hay = [_norm_tokens(w["w"]) for w in words]
    flat, owner = [], []
    for i, toks in enumerate(hay):
        for tok in toks:
            flat.append(tok)
            owner.append(i)
    if not flat:
        return None

    sm = difflib.SequenceMatcher(None, flat, src, autojunk=False)
    blocks = [b for b in sm.get_matching_blocks() if b.size > 0]
    if not blocks:
        return None

    first, last = blocks[0].a, blocks[-1].a + blocks[-1].size - 1
    matched = sum(b.size for b in blocks)
    ratio = matched / len(src)
    w0, w1 = owner[first], owner[min(last, len(owner) - 1)]
    return words[w0]["start"], words[w1]["end"], ratio


# ------------------------------------------------------------------ prompts --

def course_context(cfg: dict, course_id: str) -> str:
    course = common.course_notes_dir(cfg, course_id)
    parts = []
    notation = course / "notation.md"
    if notation.exists():
        parts.append("## Course notation and conventions\n\n"
                     + notation.read_text(encoding="utf-8").strip())
    gl = course / "glossary.json"
    if gl.exists():
        try:
            g = common.read_json(gl)
            aliases = g.get("aliases", {})
            corrections = g.get("corrections", [])
            if aliases:
                lines = [f"- {k} -> {v}" for k, v in list(aliases.items())[:60]]
                parts.append("## Canonical names\n\n" + "\n".join(lines))
            if corrections:
                lines = []
                for c in corrections[-40:]:
                    if isinstance(c, dict) and c.get("wrong") and c.get("right"):
                        lines.append(f"- {c['wrong']} -> {c['right']}")
                if lines:
                    parts.append("## Corrections you have been given before\n\n"
                                 + "\n".join(lines))
        except json.JSONDecodeError:
            pass
    return "\n\n".join(parts)


def window_text(win) -> str:
    return " ".join(w["w"] for w in win["words"])


def history_block(items, upto_t: float, history_s: float, limit: int = 40) -> str:
    recent = [it for it in items if it["t_end"] >= upto_t - history_s]
    if not recent:
        return ""
    lines = [f"- {it['latex']}" for it in recent[-limit:]]
    return ("## LaTeX already extracted from the preceding minutes\n"
            "Use the same notation and symbol conventions as these.\n\n"
            + "\n".join(lines))


def build_user_prompt(cfg, win, ctx: str, history: str) -> str:
    # The reference material is fenced off explicitly. Without this the model
    # happily extracts the worked examples out of notation.md, and quotes the
    # history list back as though the lecturer had said it.
    parts = []
    if ctx or history:
        parts.append("=== REFERENCE ONLY -- do not extract anything from this "
                     "section ===")
        if ctx:
            parts.append(ctx)
        if history:
            parts.append(history)
        parts.append("=== END REFERENCE ===")

    parts.append(
        f"=== TRANSCRIPT WINDOW ({common.hhmmss(win['t0'])} to "
        f"{common.hhmmss(win['t1'])}) -- extract only from here ===\n\n"
        + window_text(win))
    parts.append("Extract the mathematics spoken in the transcript window above. "
                 "Every source_text must be copied from that window.")
    return "\n\n".join(parts)


# ----------------------------------------------------------------- validate --

def clean_items(raw, win, logger, label, min_match=0.35):
    """Validate model output and attach time spans located from source_text."""
    out, dropped = [], 0
    if not isinstance(raw, dict):
        return out, 1
    for it in raw.get("items", []) or []:
        if not isinstance(it, dict):
            dropped += 1
            continue
        latex = str(it.get("latex", "")).strip()
        source = str(it.get("source_text", "")).strip()
        if not latex or not source:
            dropped += 1
            continue

        latex = re.sub(r"^\$+|\$+$", "", latex).strip()
        latex = re.sub(r"^\\\[|\\\]$", "", latex).strip()
        latex = fix_greek(latex)
        if not latex:
            dropped += 1
            continue

        kind = str(it.get("kind", "expression")).strip().lower()
        if kind not in KINDS:
            kind = "expression"

        try:
            conf = float(it.get("confidence", 0.5))
        except (TypeError, ValueError):
            conf = 0.5
        conf = min(1.0, max(0.0, conf))

        located = locate(source, win["words"])
        if located is None:
            t_start, t_end, ratio = win["t0"], win["t1"], 0.0
        else:
            t_start, t_end, ratio = located

        # A quote that is not in the window did not come from the audio -- it
        # is the reference material being read back, or an invention. Either
        # way it has no honest timestamp, so it is dropped rather than pinned
        # to an arbitrary span. Logged, never silent.
        if ratio < min_match:
            logger.warning("%s: dropping %r -- its source_text matched only "
                           "%.0f%% of the window (%r)",
                           label, latex[:40], ratio * 100, source[:60])
            dropped += 1
            continue

        if ratio < 0.6:
            logger.warning("%s: source_text only %.0f%% matched the transcript "
                           "(%r)", label, ratio * 100, source[:60])
            conf = min(conf, 0.4)

        ambiguous = bool(it.get("ambiguous", False))
        note = str(it.get("note", "")).strip()

        # A description of maths is not maths. Surface it in the viewer with a
        # warning rather than letting it pass as a confident conversion.
        if looks_like_prose(latex):
            ambiguous = True
            conf = min(conf, 0.4)
            note = (note + "; " if note else "") + "LaTeX contains prose, not a full conversion"

        out.append({
            "latex": latex,
            "t_start": round(float(t_start), 3),
            "t_end": round(float(max(t_end, t_start)), 3),
            "kind": kind,
            "confidence": round(conf, 3),
            "ambiguous": ambiguous,
            "source_text": source,
            "match_ratio": round(float(ratio), 3),
        })
        if note:
            out[-1]["note"] = note
    return out, dropped


_PROSE_OK = {"text", "mathrm", "operatorname", "cdot", "leq", "geq", "int", "sum",
             "frac", "sqrt", "exp", "log", "sin", "cos", "tan", "lim", "infty",
             "left", "right", "quad", "qquad", "times", "partial", "nabla"}


_GREEK = ["varepsilon", "vartheta", "varphi", "alpha", "beta", "gamma", "delta",
          "epsilon", "zeta", "theta", "iota", "kappa", "lambda", "sigma", "tau",
          "upsilon", "omega", "omicron", "eta", "mu", "nu", "xi", "pi", "rho",
          "phi", "chi", "psi"]
# Longest first so "varepsilon" is not eaten by "eta"; \b stops "beta" matching
# inside "\beta" or "theta".
_GREEK_RE = re.compile(r"(?<!\\)(?<![A-Za-z])(" + "|".join(_GREEK) + r")(?![A-Za-z])")


def fix_greek(latex: str) -> str:
    """Turn a Greek letter written as a bare word into its LaTeX command.

    The model mostly gets this right once told, but "g(tau)" still slips
    through occasionally and renders as upright text. It is a mechanical
    substitution, so do it mechanically instead of hoping. Anything already
    escaped, or inside \\text{...}, is left alone.
    """
    parts = re.split(
        r"((?:\\text|\\operatorname|\\mathrm|\\mathbf|\\mathit|\\mathsf|\\mathtt"
        r"|\\textrm|\\textbf|\\label|\\ref)\{[^}]*\})", latex)
    for i in range(0, len(parts), 2):          # odd indices are the protected spans
        parts[i] = _GREEK_RE.sub(lambda m: "\\" + m.group(1), parts[i])
    return "".join(parts)


_PLACEHOLDER_WORDS = {"something", "someone", "unknown", "unclear", "inaudible",
                      "unspecified", "involving", "etcetera", "blah", "whatever",
                      "expression", "somethings", "unintelligible"}


def looks_like_prose(latex: str) -> bool:
    """True when the LaTeX is really a description of maths the model could not
    hear, e.g. "Y < something involving Y".

    Two different tells, because the model hedges in two different ways:

      * bare English sitting loose in the maths -- LaTeX command names and
        \\text{} content are stripped first, so anything left is suspect;
      * a placeholder word anywhere at all, including inside \\text{}. That
        matters because \\text{constant} is a perfectly good label while
        \\text{something involving } is an admission of defeat, and stripping
        \\text{} wholesale would let the second one through.
    """
    if any(w in _PLACEHOLDER_WORDS for w in re.findall(r"[A-Za-z]+", latex.lower())):
        return True

    stripped = re.sub(r"\\text\{[^}]*\}", " ", latex)
    stripped = re.sub(r"\\[a-zA-Z]+", " ", stripped)
    return len(re.findall(r"[A-Za-z]{3,}", stripped)) >= 2


def dedupe(items, tolerance: float = 8.0):
    """Windows overlap, so the same statement gets extracted twice. Keep the
    higher-confidence copy of any near-duplicate."""
    # First collapse fragments: several items quoting the identical phrase are
    # the model emitting a statement and the pieces inside it. The longest
    # LaTeX is the complete one -- and it is often the lower-confidence entry,
    # so confidence is the wrong tie-break here.
    by_source = {}
    for it in items:
        key = " ".join(_norm_tokens(it["source_text"]))
        if not key:
            by_source[id(it)] = it
            continue
        prev = by_source.get(key)
        if prev is None or len(it["latex"]) > len(prev["latex"]):
            by_source[key] = it
    items = list(by_source.values())

    kept = []
    for it in sorted(items, key=lambda x: (x["t_start"], -x["confidence"])):
        clash = None
        for k in kept:
            if abs(k["t_start"] - it["t_start"]) > tolerance:
                continue
            a = re.sub(r"\s+", "", k["latex"])
            b = re.sub(r"\s+", "", it["latex"])
            if a == b or difflib.SequenceMatcher(None, a, b).ratio() > 0.92:
                clash = k
                break
        if clash is None:
            kept.append(it)
        elif it["confidence"] > clash["confidence"]:
            kept[kept.index(clash)] = it
    kept.sort(key=lambda x: x["t_start"])
    for i, it in enumerate(kept, 1):
        it["id"] = f"m_{i:03d}"
    return kept


# --------------------------------------------------------------------- run --

def process_lecture(cfg, course_id, lecture_id, logger, force=False, window_range=None):
    d = common.lecture_dir(cfg, course_id, lecture_id, create=False)
    tpath = d / "transcript.json"
    if not tpath.exists():
        raise FileNotFoundError(f"{lecture_id}: no transcript.json -- run Stage 1 first")

    out_path = d / "math.json"
    if common.stage_is_done(out_path, force) and not window_range:
        logger.info("%s -- math.json exists, skipping (use --force)", lecture_id)
        return None

    transcript = common.read_json(tpath)
    words = transcript.get("words", [])
    if not words:
        raise ValueError(f"{lecture_id}: transcript has no words")

    m = cfg["math"]
    windows = build_windows(words, float(m["window_s"]), float(m["overlap_s"]))
    if window_range:
        lo, hi = window_range
        windows = [w for w in windows if w["t1"] >= lo and w["t0"] <= hi]
    logger.info("%s: %d window(s) of %.0fs (overlap %.0fs)",
                lecture_id, len(windows), m["window_s"], m["overlap_s"])

    llm = LLM(cfg, logger)
    ctx = course_context(cfg, course_id)
    history_s = float(m.get("history_s", 300))
    retries = int(m.get("max_retries", 1))
    min_match = float(m.get("min_source_match", 0.35))

    items, failed = [], 0
    for i, win in enumerate(windows, 1):
        label = f"{lecture_id} [{common.hhmmss(win['t0'])}]"
        prompt = build_user_prompt(cfg, win, ctx, history_block(items, win["t0"], history_s))

        data = llm.json_call(SYSTEM, prompt, SCHEMA, "math_items",
                             retries=retries, label=label)
        if data is None:
            failed += 1
            continue

        got, dropped = clean_items(data, win, logger, label, min_match)
        items.extend(got)
        logger.info("  [%d/%d] %s  %d item(s)%s",
                    i, len(windows), common.hhmmss(win["t0"]), len(got),
                    f", {dropped} malformed" if dropped else "")

    items = dedupe(items)
    ambiguous = sum(1 for it in items if it["ambiguous"])
    low = sum(1 for it in items if it["confidence"] < 0.6)

    payload = {
        "lecture_id": lecture_id,
        "course_id": course_id,
        "items": items,
        "meta": {
            "model": cfg["llm"]["model"],
            "base_url": cfg["llm"]["base_url"],
            "window_s": m["window_s"],
            "overlap_s": m["overlap_s"],
            "history_s": history_s,
            "windows": len(windows),
            "windows_failed": failed,
            "ambiguous_items": ambiguous,
            "low_confidence_items": low,
            "llm": dict(llm.stats),
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        },
    }

    if window_range:
        logger.info("window range given -- not writing math.json")
        return payload

    common.atomic_write_json(out_path, payload)
    logger.info("%s: wrote math.json (%d items, %d ambiguous, %d low-confidence, "
                "%d window(s) failed)", lecture_id, len(items), ambiguous, low, failed)
    llm.log_stats()
    merge.merge_lecture(cfg, course_id, lecture_id, logger)
    return payload


def main() -> int:
    ap = argparse.ArgumentParser(description="Stage 2: convert spoken maths to LaTeX.")
    ap.add_argument("--course", required=True)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--lecture")
    g.add_argument("--all", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--window", help="only this time range, e.g. 600-900 (for prompt "
                                     "iteration; does not write math.json)")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    cfg = common.load_config(args.config)
    logger = common.setup_logging("stage2", cfg)
    report = common.BatchReport("Stage 2 (math)")

    window_range = None
    if args.window:
        try:
            lo, hi = args.window.split("-")
            window_range = (float(lo), float(hi))
        except ValueError:
            logger.error("--window wants START-END in seconds, e.g. 600-900")
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


def preview(payload, logger):
    """Side-by-side LaTeX and source_text, for judging prompt quality."""
    print()
    print("=" * 96)
    for it in payload["items"]:
        flags = []
        if it["ambiguous"]:
            flags.append("AMBIGUOUS")
        if it["confidence"] < 0.6:
            flags.append("low-conf")
        if it.get("match_ratio", 1) < 0.8:
            flags.append(f"match {it['match_ratio']:.0%}")
        head = f"[{common.hhmmss(it['t_start'])}] {it['kind']:<10} conf {it['confidence']:.2f}"
        if flags:
            head += "  <" + ", ".join(flags) + ">"
        print(head)
        print(f"   LaTeX : {it['latex']}")
        print(f"   said  : {it['source_text'][:150]}")
        if it.get("note"):
            print(f"   note  : {it['note']}")
        print("-" * 96)
    print(f"{len(payload['items'])} item(s)")


if __name__ == "__main__":
    raise SystemExit(main())
