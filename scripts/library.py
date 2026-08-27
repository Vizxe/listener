"""Stage 6 -- the library. Concept index and hybrid search.

Two features, deliberately separate:

  build_concept_index()  purely deterministic. Every concept mapped to its
                         occurrences across lectures, first definition-type
                         occurrence marked canonical. No LLM at query time,
                         and none at build time either.

  build_search_index()   chunks + BM25 + embeddings, combined at query time
                         with reciprocal rank fusion.

On chunk granularity: the brief asks for one chunk per chapter rather than
fixed token windows, and that is the default. But Stage 3 sometimes returns a
chapter spanning twenty minutes, which is far too much text for one embedding
to represent. Those get split at *note block* boundaries -- still semantic
units with their own headings, never arbitrary token windows -- and each piece
keeps its parent chapter's title, type and concepts.
"""
from __future__ import annotations

import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path

import common


# ------------------------------------------------------------- gathering ---

def load_lectures(cfg: dict, course_id: str):
    """Every merged lecture in one course."""
    data = common.project_path(cfg, "data") / course_id
    out = []
    if not data.exists():
        return out
    for d in sorted(p for p in data.iterdir() if p.is_dir()):
        lj = d / "lecture.json"
        if not lj.exists():
            continue
        try:
            out.append(common.read_json(lj))
        except json.JSONDecodeError:
            continue
    return out


def words_between(lecture, t0: float, t1: float) -> str:
    return " ".join(w["w"] for w in lecture.get("words", [])
                    if t0 <= w["start"] < t1)


def notes_between(lecture, t0: float, t1: float):
    return [n for n in lecture.get("notes", [])
            if n["t_start"] < t1 and n["t_end"] > t0]


def course_sections_between(lecture, t0: float, t1: float):
    """Stage 2C sections whose span overlaps [t0, t1).

    Stage 2C writes in reading order rather than clock order, so a section
    that merges a theorem with the proof given twenty minutes later genuinely
    spans both, and lands in both chunks. That is the right answer for search:
    either chunk is a truthful place to find it.
    """
    doc = lecture.get("course_notes") or {}
    out = []
    for sec in doc.get("sections", []) or []:
        a = sec.get("t_start")
        if a is None:
            continue
        b = sec.get("t_end")
        if b is None:
            b = a
        if a < t1 and b > t0:
            out.append(sec)
    return out


def course_section_text(sections) -> str:
    parts = []
    for sec in sections:
        bits = [sec.get("heading", ""), sec.get("body", "")]
        bits.extend(sec.get("key_points") or [])
        parts.append(". ".join(x for x in bits if x))
    return "\n".join(parts)


# -------------------------------------------------------- concept index ----

def build_concept_index(cfg: dict, course_id: str, logger=None) -> Path:
    """concept -> every place it comes up, with the canonical definition first.

    Scoped to one course: MAT267's "convergence" and another course's are not
    the same entry, and joining them would make the index useless.
    """
    aliases = common.glossary_aliases(cfg, course_id)
    lectures = load_lectures(cfg, course_id)
    concepts = {}

    for lec in lectures:
        lid = lec["lecture_id"]
        for ch in lec.get("chapters", []):
            for raw in ch.get("concepts", []):
                name = common.normalize_concept(raw, aliases)
                if not name:
                    continue
                rec = concepts.setdefault(name, {"concept": name, "occurrences": [],
                                                 "assumed_by": [], "canonical": None})
                rec["occurrences"].append({
                    "lecture_id": lid,
                    "chapter_id": ch.get("id"),
                    "title": ch.get("title"),
                    "type": ch.get("type"),
                    "t": round(float(ch.get("start", 0)), 3),
                    "summary": ch.get("summary", ""),
                    "link": f"viewer.html?course={course_id}&lecture={lid}"
                            f"&t={int(float(ch.get('start', 0)))}",
                })
            for raw in ch.get("assumes", []):
                name = common.normalize_concept(raw, aliases)
                if not name:
                    continue
                rec = concepts.setdefault(name, {"concept": name, "occurrences": [],
                                                 "assumed_by": [], "canonical": None})
                rec["assumed_by"].append({
                    "lecture_id": lid,
                    "chapter_id": ch.get("id"),
                    "title": ch.get("title"),
                    "t": round(float(ch.get("start", 0)), 3),
                })

    # The canonical occurrence is where the concept is actually defined. Fall
    # back to the earliest mention when nothing defines it.
    order = {"definition": 0, "theorem": 1, "motivation": 2, "example": 3,
             "proof": 4, "aside": 5, "admin": 6}
    for rec in concepts.values():
        rec["occurrences"].sort(key=lambda o: (o["lecture_id"], o["t"]))
        if rec["occurrences"]:
            best = min(rec["occurrences"],
                       key=lambda o: (order.get(o.get("type"), 9),
                                      o["lecture_id"], o["t"]))
            rec["canonical"] = {"lecture_id": best["lecture_id"],
                                "chapter_id": best["chapter_id"],
                                "t": best["t"], "link": best["link"]}
        rec["count"] = len(rec["occurrences"])

    payload = {
        "course_id": course_id,
        "concepts": [concepts[k] for k in sorted(concepts)],
        "meta": {
            "course_id": course_id,
            "lectures": len(lectures),
            "distinct_concepts": len(concepts),
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        },
    }
    out = common.course_index(cfg, course_id) / "concepts.json"
    common.atomic_write_json(out, payload)
    if logger:
        logger.info("%s concept index: %d concept(s) across %d lecture(s)",
                    course_id, len(concepts), len(lectures))
    return out


# --------------------------------------------------------------- chunks ----

def build_chunks(cfg: dict, course_id: str, logger=None):
    """One chunk per chapter, splitting over-long chapters at note boundaries.

    Plus one whole-lecture chunk for every lecture Stage 2C has written up,
    carrying the parts of that write-up which belong to no single chapter:
    the title, the summary, what it assumes, what it established, what it left
    open. Those are the only text in the project that describes a lecture as a
    whole, so without a chunk of their own the query "which lecture proved
    completeness" has nothing to match.
    """
    scfg = cfg.get("search", {})
    max_s = float(scfg.get("max_chunk_seconds", 420))
    lectures = load_lectures(cfg, course_id)
    chunks = []

    for lec in lectures:
        lid = lec["lecture_id"]

        # Built before the chapter check below, so a lecture written up by
        # Stage 2C is searchable even where Stage 3 has not run on it yet.
        overview = overview_chunk(lec, course_id)
        if overview:
            chunks.append(overview)

        chapters = lec.get("chapters", [])
        if not chapters:
            if logger:
                logger.warning("%s has no chapters -- run Stage 3; skipping it", lid)
            continue

        for ch in chapters:
            spans = [(ch["start"], ch["end"])]
            if ch["end"] - ch["start"] > max_s:
                blocks = notes_between(lec, ch["start"], ch["end"])
                if len(blocks) >= 2:
                    parts = max(2, int(math.ceil((ch["end"] - ch["start"]) / max_s)))
                    per = max(1, len(blocks) // parts)
                    spans = []
                    for i in range(0, len(blocks), per):
                        grp = blocks[i:i + per]
                        spans.append((grp[0]["t_start"], grp[-1]["t_end"]))
                    spans[0] = (ch["start"], spans[0][1])
                    spans[-1] = (spans[-1][0], ch["end"])
                    if logger:
                        logger.info("%s %s: %.0f min chapter split into %d chunks "
                                    "at note boundaries", lid, ch.get("id"),
                                    (ch["end"] - ch["start"]) / 60, len(spans))

            for k, (t0, t1) in enumerate(spans):
                blocks = notes_between(lec, t0, t1)
                course_text = course_section_text(
                    course_sections_between(lec, t0, t1))
                note_text = "\n".join(
                    b["heading"] + ". " + b["body"] for b in blocks)
                body = words_between(lec, t0, t1)
                title = ch.get("title", "")
                if len(spans) > 1 and blocks:
                    title = f"{ch.get('title', '')} - {blocks[0]['heading']}"

                chunks.append({
                    "id": f"{lid}:{ch.get('id')}:{k}",
                    "lecture_id": lid,
                    "course_id": course_id,
                    "chapter_id": ch.get("id"),
                    "title": title,
                    "type": ch.get("type"),
                    "t_start": round(float(t0), 3),
                    "t_end": round(float(t1), 3),
                    "summary": ch.get("summary", ""),
                    "concepts": ch.get("concepts", []),
                    "link": f"viewer.html?course={course_id}&lecture={lid}"
                            f"&t={int(float(t0))}",
                    # Embedded and searched: the summary carries the gist,
                    # the course notes carry the version of the idea that was
                    # written knowing how the lecture ended, the rough notes
                    # carry the maths, and the transcript carries the exact
                    # words the lecturer used.
                    "text": "\n".join(x for x in (ch.get("summary", ""),
                                                  course_text, note_text,
                                                  body) if x).strip(),
                })

    out = common.course_index(cfg, course_id) / "chunks.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        for c in chunks:
            fh.write(json.dumps(c, ensure_ascii=False) + "\n")
    if logger:
        logger.info("%s chunks: %d from %d lecture(s)", course_id, len(chunks), len(lectures))
    return chunks


def overview_chunk(lec, course_id: str):
    """The whole-lecture part of a Stage 2C write-up, as one chunk.

    Deliberately not the section bodies -- those are already in the chapter
    chunks that cover their time, and repeating them here would let one
    lecture answer a query twice.
    """
    doc = lec.get("course_notes") or {}
    if not doc:
        return None
    lid = lec["lecture_id"]
    title = str(doc.get("title") or "").strip()
    parts = [title, str(doc.get("summary") or "").strip()]
    if doc.get("prerequisites"):
        parts.append("Assumes: " + "; ".join(doc["prerequisites"]))
    if doc.get("takeaways"):
        parts.append("Takeaways: " + "; ".join(doc["takeaways"]))
    if doc.get("open_questions"):
        parts.append("Left open: " + "; ".join(doc["open_questions"]))
    # Headings only, so a query naming an idea can still find the lecture as a
    # whole even when that idea's own chunk ranks below it.
    heads = [s.get("heading", "") for s in (doc.get("sections") or [])]
    if any(heads):
        parts.append("Covers: " + "; ".join(h for h in heads if h))
    text = "\n".join(x for x in parts if x).strip()
    if not text:
        return None

    return {
        "id": f"{lid}:course-notes:0",
        "lecture_id": lid,
        "course_id": course_id,
        "chapter_id": "course-notes",
        "title": title or lid,
        "type": "overview",
        "t_start": 0.0,
        "t_end": round(float(lec.get("duration") or 0.0), 3),
        "summary": str(doc.get("summary") or "").strip(),
        "concepts": [],
        "link": f"viewer.html?course={course_id}&lecture={lid}&t=0&pane=course",
        "text": text,
    }


def load_chunks(cfg: dict, course_id: str):
    path = common.course_index(cfg, course_id) / "chunks.jsonl"
    if not path.exists():
        return []
    out = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


# ------------------------------------------------------------ embeddings ---

def embed_texts(cfg: dict, texts, logger=None):
    """Embeddings via the OpenAI-compatible endpoint in config."""
    from openai import OpenAI

    e = cfg["embeddings"]
    client = OpenAI(base_url=e["base_url"], api_key=e.get("api_key", "not-needed"),
                    timeout=float(e.get("timeout_s", 120)))
    batch = int(e.get("batch_size", 16))
    vecs = []
    for i in range(0, len(texts), batch):
        part = [t[:8000] for t in texts[i:i + batch]]
        resp = client.embeddings.create(model=e["model"], input=part)
        vecs.extend([d.embedding for d in resp.data])
        if logger and len(texts) > batch:
            logger.info("  embedded %d/%d", min(i + batch, len(texts)), len(texts))
    return vecs


def build_vectors(cfg: dict, course_id: str, chunks, logger=None):
    import numpy as np

    if not chunks:
        return None
    vecs = embed_texts(cfg, [c["text"] for c in chunks], logger)
    arr = np.asarray(vecs, dtype="float32")
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    arr = arr / norms                      # cosine becomes a dot product
    d = common.course_index(cfg, course_id) / "vectors"
    d.mkdir(parents=True, exist_ok=True)
    np.save(d / "embeddings.npy", arr)
    common.atomic_write_json(d / "ids.json",
                             {"ids": [c["id"] for c in chunks],
                              "model": cfg["embeddings"]["model"],
                              "dim": int(arr.shape[1])})
    if logger:
        logger.info("%s vectors: %d x %d", course_id, arr.shape[0], arr.shape[1])
    return arr


# ---------------------------------------------------------------- search ---

_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str):
    return _TOKEN.findall(str(text).lower())


class Searcher:
    """Hybrid search. Lexical and semantic, fused with reciprocal rank.

    Vector-only search is not an option here: mathematical names are precise,
    and a query for "Sylow" has to hit the lecture that says Sylow rather than
    the one that is merely nearby in embedding space.
    """

    def __init__(self, cfg: dict, course_id: str, logger=None):
        self.cfg = cfg
        self.course_id = course_id
        self.logger = logger
        self.chunks = load_chunks(cfg, course_id)
        self.by_id = {c["id"]: c for c in self.chunks}
        self.bm25 = None
        self.vectors = None
        self.vec_ids = []

        if self.chunks:
            try:
                from rank_bm25 import BM25Okapi
                self.bm25 = BM25Okapi([tokenize(c["title"] + " " + c["text"])
                                       for c in self.chunks])
            except ImportError:
                if logger:
                    logger.warning("rank_bm25 not installed -- lexical search off")

        vdir = common.course_index(cfg, course_id) / "vectors"
        if (vdir / "embeddings.npy").exists() and (vdir / "ids.json").exists():
            try:
                import numpy as np
                self.vectors = np.load(vdir / "embeddings.npy")
                self.vec_ids = common.read_json(vdir / "ids.json")["ids"]
            except Exception as exc:
                if logger:
                    logger.warning("could not load vectors: %s", exc)

    @property
    def ready(self):
        return bool(self.chunks)

    def _lexical(self, query, k):
        if not self.bm25:
            return []
        scores = self.bm25.get_scores(tokenize(query))
        order = sorted(range(len(scores)), key=lambda i: -scores[i])[:k]
        return [(self.chunks[i]["id"], float(scores[i])) for i in order
                if scores[i] > 0]

    def _semantic(self, query, k):
        if self.vectors is None or not self.vec_ids:
            return []
        try:
            import numpy as np
            qv = embed_texts(self.cfg, [query])[0]
            q = np.asarray(qv, dtype="float32")
            n = np.linalg.norm(q) or 1.0
            sims = self.vectors @ (q / n)
            order = np.argsort(-sims)[:k]
            return [(self.vec_ids[i], float(sims[i])) for i in order]
        except Exception as exc:
            if self.logger:
                self.logger.warning("semantic search unavailable: %s", exc)
            return []

    def search(self, query: str, k: int = 10):
        if not query.strip() or not self.chunks:
            return []
        cfg_s = self.cfg.get("search", {})
        rrf_k = float(cfg_s.get("rrf_k", 60))
        pool = max(k * 3, 20)

        lex = self._lexical(query, pool)
        sem = self._semantic(query, pool)

        fused = {}
        for rank, (cid, score) in enumerate(lex):
            fused.setdefault(cid, {"rrf": 0.0, "lex": None, "sem": None})
            fused[cid]["rrf"] += 1.0 / (rrf_k + rank + 1)
            fused[cid]["lex"] = round(score, 4)
        for rank, (cid, score) in enumerate(sem):
            fused.setdefault(cid, {"rrf": 0.0, "lex": None, "sem": None})
            fused[cid]["rrf"] += 1.0 / (rrf_k + rank + 1)
            fused[cid]["sem"] = round(score, 4)

        ranked = sorted(fused.items(), key=lambda kv: -kv[1]["rrf"])[:k]
        out = []
        for cid, s in ranked:
            c = self.by_id.get(cid)
            if not c:
                continue
            out.append({
                "id": cid,
                "lecture_id": c["lecture_id"],
                "course_id": c.get("course_id", self.course_id),
                "chapter_id": c["chapter_id"],
                "title": c["title"],
                "type": c["type"],
                "t_start": c["t_start"],
                "summary": c["summary"],
                "concepts": c["concepts"],
                "link": c["link"],
                "score": round(s["rrf"], 6),
                "lexical": s["lex"],
                "semantic": s["sem"],
                "matched": [m for m in ("lexical" if s["lex"] is not None else None,
                                        "semantic" if s["sem"] is not None else None) if m],
            })
        return out
