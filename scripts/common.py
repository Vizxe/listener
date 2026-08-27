"""Shared plumbing: config, paths, logging, atomic writes, batch reporting.

Every stage reads files and writes files. Nothing here holds another stage's
output in memory, and nothing here imports a model.
"""
from __future__ import annotations

import json
import logging
import os
import re
import sys
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent

AUDIO_EXTS = {".mp3", ".m4a", ".wav", ".flac", ".opus", ".ogg", ".aac", ".wma",
              ".mp4", ".m4b", ".mkv", ".webm"}


# ---------------------------------------------------------------- config ----

def load_config(path=None) -> dict:
    path = Path(path) if path else ROOT / "config.yaml"
    with open(path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    cfg["_root"] = ROOT
    return cfg


def project_path(cfg: dict, key: str) -> Path:
    """Resolve a `paths:` entry to an absolute path."""
    return ROOT / cfg["paths"][key]


def lecture_dir(cfg: dict, course_id: str, lecture_id: str, create: bool = True) -> Path:
    d = project_path(cfg, "data") / course_id / lecture_id
    if create:
        d.mkdir(parents=True, exist_ok=True)
    return d


# ------------------------------------------------------------------- io -----

def read_json(path):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def atomic_write_json(path, obj, indent: int = 2) -> None:
    """Write via temp file + replace, so a crash never leaves a half-written
    file that the next run would treat as completed output."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(obj, fh, ensure_ascii=False, indent=indent)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def stage_is_done(path, force: bool) -> bool:
    """True when `path` already holds usable output and --force was not given."""
    path = Path(path)
    if force or not path.exists() or path.stat().st_size == 0:
        return False
    try:
        read_json(path)
        return True
    except (json.JSONDecodeError, UnicodeDecodeError):
        return False  # corrupt output is not completed output


# -------------------------------------------------------------- logging -----

def setup_logging(stage: str, cfg: dict = None) -> logging.Logger:
    logs = (project_path(cfg, "logs") if cfg else ROOT / "logs")
    logs.mkdir(parents=True, exist_ok=True)
    logfile = logs / f"{stage}-{datetime.now():%Y%m%d}.log"

    logger = logging.getLogger(stage)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    con = logging.StreamHandler(sys.stdout)
    con.setFormatter(logging.Formatter("%(asctime)s  %(levelname)-7s %(message)s", "%H:%M:%S"))
    logger.addHandler(con)

    fh = logging.FileHandler(logfile, encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s  %(levelname)-7s %(message)s", "%Y-%m-%d %H:%M:%S"))
    logger.addHandler(fh)

    logger.info("log file: %s", logfile)
    return logger


# ------------------------------------------------------------- lectures -----

def audio_files(cfg: dict, course_id: str):
    d = project_path(cfg, "audio") / course_id
    if not d.exists():
        return []
    return sorted(p for p in d.iterdir()
                  if p.is_file() and p.suffix.lower() in AUDIO_EXTS)


def lecture_ids(cfg: dict, course_id: str):
    """Lectures that have at least been transcribed."""
    d = project_path(cfg, "data") / course_id
    if not d.exists():
        return []
    return sorted(p.name for p in d.iterdir()
                  if p.is_dir() and (p / "transcript.json").exists())


def lecture_id_for(path: Path) -> str:
    """Filename stem, slugified. `Algebra 07.mp3` -> `algebra-07`."""
    slug = re.sub(r"[^a-z0-9]+", "-", path.stem.lower()).strip("-")
    return slug or "lecture"


def resolve_audio(cfg: dict, course_id: str, lecture_id: str):
    for p in audio_files(cfg, course_id):
        if lecture_id_for(p) == lecture_id:
            return p
    return None


# -------------------------------------------------------- prompt biasing ----

def _vocabulary_from_notation(text: str):
    """Pull bullet items out of the `## Vocabulary` section only."""
    m = re.search(r"^##\s+Vocabulary\s*$(.*?)(?=^##\s|\Z)",
                  text, re.MULTILINE | re.DOTALL)
    if not m:
        return []
    terms = []
    for line in m.group(1).splitlines():
        line = line.strip()
        if line.startswith(("-", "*")):
            for part in line.lstrip("-* ").split(","):
                part = part.strip()
                if part and not part.startswith("("):
                    terms.append(part)
    return terms


def build_initial_prompt(cfg: dict, course_id: str) -> str:
    """notation.md vocabulary + glossary terms -> Whisper `initial_prompt`.

    Whisper honours only ~224 tokens of prompt, so this is deliberately a flat
    comma list of headwords rather than prose -- more terms per token.
    """
    course = course_notes_dir(cfg, course_id)
    terms = []

    notation = course / "notation.md"
    if notation.exists():
        terms += _vocabulary_from_notation(notation.read_text(encoding="utf-8"))

    glossary = course / "glossary.json"
    if glossary.exists():
        try:
            g = read_json(glossary)
            terms += [str(t) for t in g.get("terms", [])]
            terms += [str(k) for k in g.get("aliases", {}).keys()]
        except json.JSONDecodeError:
            pass

    seen, uniq = set(), []
    for t in terms:
        k = t.lower()
        if k not in seen:
            seen.add(k)
            uniq.append(t)

    if not uniq:
        return ""

    limit = int(cfg["asr"].get("initial_prompt_max_chars", 850))
    out = "This is a university mathematics lecture. Terms used: "
    for i, t in enumerate(uniq):
        add = t if i == 0 else f", {t}"
        if len(out) + len(add) + 1 > limit:
            break
        out += add
    return out + "."


# ------------------------------------------------------------ reporting -----

def hhmmss(seconds: float) -> str:
    s = int(round(seconds))
    return f"{s // 3600:d}:{(s % 3600) // 60:02d}:{s % 60:02d}"


@dataclass
class BatchReport:
    """Collects per-lecture outcomes so one bad lecture never kills a batch."""
    stage: str
    ok: list = field(default_factory=list)
    skipped: list = field(default_factory=list)
    failed: list = field(default_factory=list)

    def record_ok(self, lecture_id):
        self.ok.append(lecture_id)

    def record_skip(self, lecture_id):
        self.skipped.append(lecture_id)

    def record_fail(self, lecture_id, reason):
        self.failed.append((lecture_id, reason))

    def print_summary(self, logger) -> int:
        logger.info("")
        logger.info("=" * 62)
        logger.info("%s summary: %d ok, %d skipped, %d failed",
                    self.stage, len(self.ok), len(self.skipped), len(self.failed))
        for lid in self.skipped:
            logger.info("  skip   %s (output exists; use --force to redo)", lid)
        for lid in self.ok:
            logger.info("  ok     %s", lid)
        for lid, reason in self.failed:
            logger.error("  FAIL   %s: %s", lid, reason)
        logger.info("=" * 62)
        return 1 if self.failed else 0


# ------------------------------------------------------- concept naming ----

# Plurals that the naive "drop the s" rule gets wrong. Maths is full of them.
_IRREGULAR = {
    "axes": "axis", "bases": "basis", "matrices": "matrix", "indices": "index",
    "vertices": "vertex", "analyses": "analysis", "hypotheses": "hypothesis",
    "theses": "thesis", "radii": "radius", "foci": "focus", "loci": "locus",
    "maxima": "maximum", "minima": "minimum", "criteria": "criterion",
    "polyhedra": "polyhedron", "simplices": "simplex", "vertices ": "vertex",
}

# Words that simply end in s. Stripping it produces nonsense.
_INVARIANT = {
    "series", "species", "calculus", "basis", "axis", "analysis", "hypothesis",
    "class", "mass", "gauss", "cross", "modulus", "radius", "locus", "focus",
    "lens", "means", "continuous", "homogeneous", "simultaneous",
}


def _singular(word: str) -> str:
    w = word.lower()
    if w in _IRREGULAR:
        return _IRREGULAR[w]
    if w in _INVARIANT or len(w) <= 3:
        return w
    if w.endswith(("ss", "us", "is", "ous")):
        return w
    if w.endswith("ies") and len(w) > 4:
        return w[:-3] + "y"
    if w.endswith("es") and w.endswith(("ches", "shes", "xes", "zes", "sses")):
        return w[:-2]
    if w.endswith("s"):
        return w[:-1]
    return w


def normalize_concept(text: str, aliases: dict = None) -> str:
    """Canonical form of a concept name, so it joins across lectures.

    Lowercase, no articles, singular head. The brief's requirement: if the
    lecturer says "unif. cts" and "uniformly continuous" they must collapse to
    one concept, and the alias map in course/glossary.json is what makes that
    possible -- rules alone will never catch it.
    """
    if not text:
        return ""

    # Try the alias map on the raw text before punctuation is stripped: the
    # abbreviations worth aliasing ("unif. cts") are mostly punctuation.
    raw = re.sub(r"\s+", " ", str(text).lower()).strip()
    if aliases and raw in aliases:
        raw = re.sub(r"\s+", " ", aliases[raw].lower()).strip()

    s = re.sub(r"[^a-z0-9\s'-]+", " ", raw)
    s = re.sub(r"\s+", " ", s).strip()
    if not s:
        return ""

    if aliases:
        direct = aliases.get(s)
        if direct:
            s = re.sub(r"\s+", " ", str(direct).lower()).strip()

    words = [w for w in s.split(" ") if w not in ("a", "an", "the", "of", "'s")]
    if not words:
        return ""

    # Only the head noun gets singularised: "uniform continuity" must not
    # become "uniform continuit", and "sets of measure zero" keeps its tail.
    words[-1] = _singular(words[-1])
    out = " ".join(words).strip()

    if aliases:
        out = re.sub(r"\s+", " ", str(aliases.get(out, out)).lower()).strip()
    return out


def glossary_aliases(cfg: dict, course_id: str) -> dict:
    """The alias map for one course, lowercased on both sides."""
    path = course_notes_dir(cfg, course_id) / "glossary.json"
    if not path.exists():
        return {}
    try:
        g = read_json(path)
    except json.JSONDecodeError:
        return {}
    return {str(k).lower().strip(): str(v).lower().strip()
            for k, v in (g.get("aliases") or {}).items()}


# --------------------------------------------------------------- courses ----

COURSE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


def valid_course_id(cid) -> bool:
    return bool(cid) and bool(COURSE_ID.match(str(cid))) and ".." not in str(cid)


def courses_file(cfg: dict) -> Path:
    return ROOT / "courses.json"


def load_courses(cfg: dict) -> list:
    """The course registry. Missing or broken file yields an empty list."""
    p = courses_file(cfg)
    if not p.exists():
        return []
    try:
        data = read_json(p)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return []
    out = []
    for c in (data.get("courses") or []):
        if isinstance(c, dict) and valid_course_id(c.get("id")):
            out.append(c)
    return sorted(out, key=lambda c: str(c.get("id", "")).lower())


def save_courses(cfg: dict, courses: list) -> Path:
    p = courses_file(cfg)
    atomic_write_json(p, {"courses": courses})
    return p


def get_course(cfg: dict, course_id: str):
    for c in load_courses(cfg):
        if c["id"] == course_id:
            return c
    return None


def ensure_course(cfg: dict, course_id: str, title: str = None) -> dict:
    """Register a course and create its directories. Idempotent."""
    if not valid_course_id(course_id):
        raise ValueError(f"bad course id {course_id!r}: letters, digits, . _ - only")
    courses = load_courses(cfg)
    existing = next((c for c in courses if c["id"] == course_id), None)
    if existing is None:
        existing = {
            "id": course_id,
            "title": (title or course_id).strip()[:200] or course_id,
            "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        courses.append(existing)
        save_courses(cfg, courses)
    elif title and title.strip() and existing.get("title") != title.strip():
        existing["title"] = title.strip()[:200]
        save_courses(cfg, courses)

    course_audio(cfg, course_id).mkdir(parents=True, exist_ok=True)
    course_data(cfg, course_id).mkdir(parents=True, exist_ok=True)
    course_index(cfg, course_id).mkdir(parents=True, exist_ok=True)
    notes_dir = course_notes_dir(cfg, course_id)
    notes_dir.mkdir(parents=True, exist_ok=True)

    notation = notes_dir / "notation.md"
    if not notation.exists():
        notation.write_text(DEFAULT_NOTATION.format(course=course_id), encoding="utf-8")
    glossary = notes_dir / "glossary.json"
    if not glossary.exists():
        atomic_write_json(glossary, {
            "_comment": f"Accumulated corrections for {course_id}. "
                        f"The pipeline appends here; hand edits are safe.",
            "terms": [], "aliases": {}, "corrections": [],
        })
    return existing


def course_audio(cfg: dict, course_id: str) -> Path:
    return project_path(cfg, "audio") / course_id


def course_data(cfg: dict, course_id: str) -> Path:
    return project_path(cfg, "data") / course_id


def course_index(cfg: dict, course_id: str) -> Path:
    return project_path(cfg, "index") / course_id


def course_notes_dir(cfg: dict, course_id: str) -> Path:
    return project_path(cfg, "courses") / course_id


DEFAULT_NOTATION = """\
# {course} — notation and conventions

Edit this from the course page, or straight on disk. Two jobs:

1. **Stage 1 vocabulary biasing.** The list below is fed to the recogniser.
   Whisper honours only ~224 tokens of prompt, so keep it tight and put the
   most-misheard terms first. This is the highest-leverage knob in the
   pipeline.
2. **Stage 2/3 context.** The whole file goes to the model on every prompt.

> **Only list terms this course actually uses.** Priming a term that never
> occurs makes the recogniser reach for it: with "Sylow" in this list, a
> lecture that said "zero is equal to one" came back as "Sylow is equal to 1".

## Vocabulary

Terms the recogniser gets wrong. Most-misheard first.

- (add the names, theorems and symbols this course keeps mangling)

## Conventions

How this lecturer speaks, so the maths pass can disambiguate.

- "eff of ex" -> `f(x)`
- "a sub n" -> `a_n`

## Do not correct

Things that look like errors but are not.

- (add entries here as you find them)
"""
