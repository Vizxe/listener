"""One-off migration: move flat lectures into a course.

    python scripts/migrate_to_courses.py --course MAT267 --title "Analysis II"
    python scripts/migrate_to_courses.py --course MAT267 --dry-run

Before, everything sat at the top level: audio/<file>, data/<lecture>/,
course/notation.md, index/. Afterwards each of those lives under a course id,
so lectures, notation and the concept index stay separated per course.

Nothing is deleted -- files are moved, and the run stops before touching
anything if the destination already holds data.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="Move flat-layout data into a course.")
    ap.add_argument("--course", required=True, help="course id, e.g. MAT267")
    ap.add_argument("--title", default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    cfg = common.load_config(args.config)
    logger = common.setup_logging("migrate", cfg)
    cid = args.course

    if not common.valid_course_id(cid):
        logger.error("bad course id %r", cid)
        return 1

    root = common.ROOT
    audio_root = common.project_path(cfg, "audio")
    data_root = common.project_path(cfg, "data")
    index_root = common.project_path(cfg, "index")
    notes_root = common.project_path(cfg, "courses")

    moves = []

    # Loose audio files at the top of audio/
    for p in sorted(audio_root.glob("*")) if audio_root.exists() else []:
        if p.is_file() and p.suffix.lower() in common.AUDIO_EXTS:
            moves.append((p, audio_root / cid / p.name))

    # Lecture directories sitting directly under data/
    for p in sorted(data_root.glob("*")) if data_root.exists() else []:
        if p.is_dir() and (p / "transcript.json").exists():
            moves.append((p, data_root / cid / p.name))
    for name in ("lectures.json", "lectures.js"):
        old = data_root / name
        if old.exists():
            moves.append((old, None))          # regenerated, just remove

    # The old single course/ directory
    old_notes = root / "course"
    if old_notes.is_dir():
        for p in sorted(old_notes.glob("*")):
            if p.is_file():
                moves.append((p, notes_root / cid / p.name))

    # Flat index files
    for name in ("concepts.json", "chunks.jsonl"):
        old = index_root / name
        if old.exists():
            moves.append((old, index_root / cid / name))
    old_vec = index_root / "vectors"
    if old_vec.is_dir() and any(old_vec.iterdir()):
        moves.append((old_vec, index_root / cid / "vectors"))
    old_bm = index_root / "bm25"
    if old_bm.is_dir() and not any(old_bm.iterdir()):
        moves.append((old_bm, None))

    if not moves:
        logger.info("nothing to migrate -- the layout already looks course-based")
        common.ensure_course(cfg, cid, args.title)
        return 0

    logger.info("planned moves:")
    for src, dst in moves:
        logger.info("  %-52s -> %s", src.relative_to(root),
                    dst.relative_to(root) if dst else "(remove)")

    clashes = [d for _, d in moves if d and d.exists()]
    if clashes:
        for c in clashes:
            logger.error("destination already exists: %s", c.relative_to(root))
        logger.error("refusing to overwrite -- move or remove these first")
        return 1

    if args.dry_run:
        logger.info("dry run: nothing changed")
        return 0

    common.ensure_course(cfg, cid, args.title)

    for src, dst in moves:
        if dst is None:
            if src.is_dir():
                shutil.rmtree(src, ignore_errors=True)
            else:
                src.unlink(missing_ok=True)
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))
    logger.info("moved %d item(s)", len(moves))

    if old_notes.is_dir() and not any(old_notes.iterdir()):
        old_notes.rmdir()
        logger.info("removed the empty course/ directory")

    logger.info("")
    logger.info("Now re-merge and rebuild the index:")
    logger.info("  python scripts/merge.py --course %s --all", cid)
    logger.info("  python scripts/stage6_index.py --course %s", cid)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
