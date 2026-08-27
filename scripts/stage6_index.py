"""Stage 6 -- build the library indexes.

    python scripts/stage6_index.py              # everything
    python scripts/stage6_index.py --concepts   # concept index only (no LLM)
    python scripts/stage6_index.py --no-vectors # skip embeddings

Reads:  data/*/lecture.json, course/glossary.json
Writes: index/concepts.json, index/chunks.jsonl, index/vectors/

The concept index needs nothing but the files on disk. Only the embedding step
talks to a model, so `--concepts` works with everything unloaded.
"""
from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
import library  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="Stage 6: build the concept and search indexes.")
    ap.add_argument("--course", help="one course; omit to rebuild every course")
    ap.add_argument("--concepts", action="store_true", help="concept index only")
    ap.add_argument("--no-vectors", action="store_true", help="skip the embedding pass")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    cfg = common.load_config(args.config)
    logger = common.setup_logging("stage6", cfg)
    failures = []

    if args.course:
        course_ids = [args.course]
    else:
        course_ids = [c["id"] for c in common.load_courses(cfg)]
    if not course_ids:
        logger.error("no courses registered -- create one first")
        return 1

    for course_id in course_ids:
        lectures = library.load_lectures(cfg, course_id)
        if not lectures:
            logger.warning("%s: no merged lectures, skipping", course_id)
            continue
        logger.info("%s: %d lecture(s)", course_id, len(lectures))

        try:
            library.build_concept_index(cfg, course_id, logger)
        except Exception as exc:
            logger.error("%s concept index failed: %s", course_id, exc)
            logger.debug(traceback.format_exc())
            failures.append(f"{course_id}/concepts")

        if args.concepts:
            continue

        chunks = []
        try:
            chunks = library.build_chunks(cfg, course_id, logger)
        except Exception as exc:
            logger.error("%s chunking failed: %s", course_id, exc)
            logger.debug(traceback.format_exc())
            failures.append(f"{course_id}/chunks")

        if chunks and not args.no_vectors:
            try:
                library.build_vectors(cfg, course_id, chunks, logger)
            except Exception as exc:
                # Lexical search still works without these, so this is a
                # degraded index rather than a failed one.
                logger.error("%s embeddings failed (%s) -- search will be "
                             "lexical-only until this is rerun", course_id, exc)
                failures.append(f"{course_id}/vectors")

    logger.info("")
    logger.info("=" * 62)
    if failures:
        logger.error("Stage 6 finished with problems: %s", ", ".join(failures))
    else:
        logger.info("Stage 6 complete")
    logger.info("=" * 62)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
