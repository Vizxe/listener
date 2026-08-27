"""Stage 4 -- merge. Builds the single file the viewer reads.

Merges whatever stage output exists for a lecture into data/<id>/lecture.json.
Right now that is Stage 1 only; Stages 2 and 3 slot in as they land, and the
viewer's data contract does not change when they do.

    python scripts/merge.py --lecture algebra-07
    python scripts/merge.py --all

Also writes lecture.js: the same payload wrapped in an assignment. Browsers
refuse fetch() against file: URLs, so opening viewer.html straight off disk
needs a script tag rather than a fetch. lecture.json stays the canonical,
greppable artifact; lecture.js is a derived convenience.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

STAGE_FILES = {
    "transcript": "transcript.json",
    "math": "math.json",
    "notes": "notes.json",
    "course_notes": "course_notes.json",
    "structure": "structure.json",
}


def merge_lecture(cfg: dict, course_id: str, lecture_id: str, logger=None) -> Path:
    d = common.lecture_dir(cfg, course_id, lecture_id, create=False)
    tpath = d / STAGE_FILES["transcript"]
    if not tpath.exists():
        raise FileNotFoundError(lecture_id + ": no transcript.json -- run Stage 1 first")

    transcript = common.read_json(tpath)

    # Prefer the seek-accurate derived file. Browsers seeking a long VBR MP3
    # land seconds away from the time they report, which desyncs the viewer
    # after every click; Ogg Opus seeks exactly. Falls back to the original.
    audio_src = None
    for cand in sorted(d.glob("audio.*")):
        if cand.suffix.lower() not in (".wav",):
            audio_src = f"../data/{course_id}/{lecture_id}/{cand.name}"
            break

    lecture = {
        "lecture_id": lecture_id,
        "course_id": course_id,
        "duration": transcript.get("duration"),
        "audio_file": transcript.get("meta", {}).get("audio_file"),
        "audio_src": audio_src,
        "words": transcript.get("words", []),
        "segments": transcript.get("segments", []),
        "math": [],
        "notes": [],
        # A document rather than a list: title, summary, sections, takeaways.
        # The viewer renders it as a whole, so it is carried across whole.
        "course_notes": None,
        "chapters": [],
        "stages": {"transcript": True, "math": False, "notes": False,
                   "course_notes": False, "structure": False},
        "meta": {
            "transcript": transcript.get("meta", {}),
            "merged_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        },
    }

    mpath = d / STAGE_FILES["math"]
    if mpath.exists():
        lecture["math"] = common.read_json(mpath).get("items", [])
        lecture["stages"]["math"] = True

    npath = d / STAGE_FILES["notes"]
    if npath.exists():
        lecture["notes"] = common.read_json(npath).get("blocks", [])
        lecture["stages"]["notes"] = True

    cpath = d / STAGE_FILES["course_notes"]
    if cpath.exists():
        doc = common.read_json(cpath)
        doc.pop("lecture_id", None)
        doc.pop("course_id", None)
        lecture["course_notes"] = doc
        lecture["stages"]["course_notes"] = True

    spath = d / STAGE_FILES["structure"]
    if spath.exists():
        lecture["chapters"] = common.read_json(spath).get("chapters", [])
        lecture["stages"]["structure"] = True

    out = d / "lecture.json"
    common.atomic_write_json(out, lecture)

    payload = json.dumps(lecture, ensure_ascii=False, separators=(",", ":"))
    (d / "lecture.js").write_text(
        "window.__LECTURE__ = " + payload + ";\n", encoding="utf-8")

    if logger:
        logger.info("%s: merged -> lecture.json (%s) "
                    "[math=%s notes=%s course_notes=%s structure=%s]",
                    lecture_id, _size(out), lecture["stages"]["math"],
                    lecture["stages"]["notes"], lecture["stages"]["course_notes"],
                    lecture["stages"]["structure"])
    return out


def write_lecture_list(cfg: dict, course_id: str, logger=None) -> Path:
    """An index of one course's lectures, for the viewer's picker."""
    data = common.project_path(cfg, "data") / course_id
    entries = []
    dirs = sorted(p for p in data.iterdir() if p.is_dir()) if data.exists() else []
    for d in dirs:
        lj = d / "lecture.json"
        if not lj.exists():
            continue
        try:
            rec = common.read_json(lj)
        except json.JSONDecodeError:
            continue
        entries.append({
            "lecture_id": rec.get("lecture_id", d.name),
            "course_id": course_id,
            "duration": rec.get("duration"),
            "audio_file": rec.get("audio_file"),
            "stages": rec.get("stages", {}),
        })

    data.mkdir(parents=True, exist_ok=True)
    out = data / "lectures.json"
    payload = {"course_id": course_id, "lectures": entries}
    common.atomic_write_json(out, payload)
    (data / "lectures.js").write_text(
        "window.__LECTURES__ = " + json.dumps(payload, ensure_ascii=False) + ";\n",
        encoding="utf-8")
    if logger:
        logger.info("%s: lecture index -> %d lecture(s)", course_id, len(entries))
    return out


def _size(p: Path) -> str:
    n = p.stat().st_size
    return ("%.1f MB" % (n / 1e6)) if n >= 1e6 else ("%.0f kB" % (n / 1e3))


def main() -> int:
    ap = argparse.ArgumentParser(description="Stage 4: merge stage outputs for the viewer.")
    ap.add_argument("--course", required=True)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--lecture")
    g.add_argument("--all", action="store_true")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    cfg = common.load_config(args.config)
    logger = common.setup_logging("merge", cfg)
    report = common.BatchReport("Stage 4 (merge)")

    ids = [args.lecture] if args.lecture else common.lecture_ids(cfg, args.course)
    if not ids:
        logger.error("nothing to merge for course %s", args.course)
        return 1

    for lecture_id in ids:
        try:
            merge_lecture(cfg, args.course, lecture_id, logger)
            report.record_ok(lecture_id)
        except Exception as exc:
            logger.error("%s failed: %s", lecture_id, exc)
            report.record_fail(lecture_id, "%s: %s" % (type(exc).__name__, exc))

    write_lecture_list(cfg, args.course, logger)
    return report.print_summary(logger)


if __name__ == "__main__":
    raise SystemExit(main())
