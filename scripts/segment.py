"""Topic segmentation of a transcript, for Stage 2N's windows.

Stage 2N's default is equal six-minute windows. That number is a compromise
and it shows: a definition that takes ninety seconds gets the same slot as a
proof that takes twelve minutes, boundaries land mid-sentence, and the overlap
exists only to catch the ideas the boundaries cut in half.

This module cuts the lecture where it actually changes subject instead.

    windows = segment.topic_windows(cfg, words, logger, lecture_id)

Why not reuse Stage 3's chapters: Stage 3 reads `notes.json`. It cannot tell
Stage 2N where to cut, because it does not exist until Stage 2N has run. The
boundaries therefore have to come from the transcript itself, before any model
has read it.

How it works -- TextTiling, with embeddings standing in for the original's
word-overlap score:

  1. Cut the word stream into small equal blocks (30s by default). These are
     measurement granularity, not windows.
  2. Embed each block, through the same local endpoint Stage 6 uses.
  3. At every block boundary, compare the mean embedding of the blocks before
     it with the mean of the blocks after. Where the lecturer is developing
     one idea the two agree; where the subject changes they do not.
  4. Score each valley in that curve by its *depth* -- how far it falls from
     the peaks either side -- rather than by its absolute value. A lecture
     that stays on one topic has a flat curve and few deep valleys, which is
     the correct answer for it.
  5. Keep the valleys deeper than `mean + threshold * stdev`, subject to the
     length limits, and snap each to the nearest pause in the audio, because
     lecturers stop talking when they change subject.

No LLM is involved, and nothing here invents a timestamp: every boundary is
the start time of a real word.

Degrades rather than fails. If the embedding endpoint is down, this returns
None and Stage 2N falls back to fixed windows with a warning.
"""
from __future__ import annotations

import statistics


def _blocks(words, block_s: float):
    """The word stream cut into contiguous equal-length blocks."""
    out, cur = [], []
    if not words:
        return out
    edge = words[0]["start"] + block_s
    for w in words:
        if w["start"] >= edge and cur:
            out.append(cur)
            cur = []
            while w["start"] >= edge:
                edge += block_s
        cur.append(w)
    if cur:
        out.append(cur)
    return out


def _similarity_curve(vecs, k: int):
    """Cosine similarity across each block boundary, k blocks either side.

    Returns (gap_index, similarity) pairs, where gap_index g sits between
    block g-1 and block g.
    """
    import numpy as np

    arr = np.asarray(vecs, dtype="float32")
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    arr = arr / norms

    nb = len(arr)
    out = []
    for g in range(k, nb - k + 1):
        left = arr[g - k:g].mean(axis=0)
        right = arr[g:g + k].mean(axis=0)
        ln = float(np.linalg.norm(left)) or 1.0
        rn = float(np.linalg.norm(right)) or 1.0
        out.append((g, float(left @ right) / (ln * rn)))
    return out


def _valleys(curve):
    """Local minima of the similarity curve, scored by depth.

    Depth, not absolute similarity: how far the curve falls from the peaks
    either side of it. Two lectures can sit at completely different similarity
    baselines -- a dense proof is more self-similar throughout than a
    rambling revision session -- and depth is what survives that difference.
    """
    sims = [s for _g, s in curve]
    out = []
    for i in range(1, len(sims) - 1):
        if not (sims[i] <= sims[i - 1] and sims[i] <= sims[i + 1]):
            continue
        j = i
        while j > 0 and sims[j - 1] >= sims[j]:
            j -= 1
        left_peak = sims[j]
        j = i
        while j < len(sims) - 1 and sims[j + 1] >= sims[j]:
            j += 1
        right_peak = sims[j]
        depth = (left_peak - sims[i]) + (right_peak - sims[i])
        out.append({"gap": curve[i][0], "depth": depth, "sim": sims[i]})
    return out


def _snap_to_pause(words, t: float, half: float):
    """Move a boundary to the biggest silence near it.

    A boundary derived from block statistics lands wherever a block happened
    to start, which is usually mid-sentence. Lecturers pause when they change
    subject, so the largest gap nearby is a better cut than the arithmetic
    one -- and it keeps the note's first words from being the tail of the
    previous thought.
    """
    best_t, best_gap = t, -1.0
    for i in range(1, len(words)):
        s = words[i]["start"]
        if s < t - half:
            continue
        if s > t + half:
            break
        gap = s - words[i - 1]["end"]
        if gap > best_gap:
            best_gap, best_t = gap, s
    return best_t if best_gap > 0 else t


def _enforce_min(cands, t0: float, t1: float, min_s: float):
    """Greedy by depth: take the strongest boundary, then anything far enough
    from every boundary already taken."""
    kept = []
    for c in sorted(cands, key=lambda c: -c["depth"]):
        t = c["t"]
        if t - t0 < min_s or t1 - t < min_s:
            continue
        if any(abs(t - k["t"]) < min_s for k in kept):
            continue
        kept.append(c)
    kept.sort(key=lambda c: c["t"])
    return kept


def _enforce_max(kept, rejected, words, t0: float, t1: float,
                 min_s: float, max_s: float, snap_half: float):
    """Split any stretch still longer than max_s.

    A topic really can run twenty minutes, and "one note per topic" would then
    hand the model twenty minutes to summarise in one block -- which is the
    failure the six-minute window was avoiding in the first place. So the
    length cap stays. It is applied by cutting at the best *rejected* boundary
    inside the stretch, falling back to the longest pause near the midpoint
    when the curve offers nothing.
    """
    edges = [t0] + [c["t"] for c in kept] + [t1]
    guard = 12
    while guard > 0:
        guard -= 1
        widest_i, widest = -1, max_s
        for i in range(len(edges) - 1):
            span = edges[i + 1] - edges[i]
            if span > widest:
                widest, widest_i = span, i
        if widest_i < 0:
            break

        a, b = edges[widest_i], edges[widest_i + 1]
        inside = [c for c in rejected
                  if a + min_s <= c["t"] <= b - min_s
                  and all(abs(c["t"] - e) >= min_s for e in edges)]
        if inside:
            cut = max(inside, key=lambda c: c["depth"])["t"]
            rejected = [c for c in rejected if c["t"] != cut]
        else:
            cut = _snap_to_pause(words, (a + b) / 2.0, snap_half)
            if cut <= a + 1 or cut >= b - 1:
                cut = (a + b) / 2.0
        edges.insert(widest_i + 1, cut)
        edges.sort()
    return edges


def topic_windows(cfg: dict, words, logger, label: str = ""):
    """Windows cut at topic changes, or None if that was not possible."""
    n = cfg.get("notes", {})
    t = n.get("topic", {}) or {}
    block_s = float(t.get("block_s", 30))
    k = int(t.get("context_blocks", 4))
    threshold = float(t.get("threshold", 0.3))
    min_s = float(t.get("min_seconds", 120))
    max_s = float(t.get("max_seconds", 600))
    snap_half = float(t.get("snap_window_s", 20))

    blocks = _blocks(words, block_s)
    # The curve needs k blocks of context either side of a gap, plus a block
    # either side of a valley to have peaks to measure its depth against.
    if len(blocks) < 2 * k + 3:
        logger.info("%s: %.0f min is too short to segment by topic (%d blocks "
                    "of %.0fs); using fixed windows", label,
                    (words[-1]["end"] - words[0]["start"]) / 60 if words else 0,
                    len(blocks), block_s)
        return None

    try:
        import library
        vecs = library.embed_texts(
            cfg, [" ".join(w["w"] for w in b) for b in blocks])
    except Exception as exc:
        logger.warning("%s: could not embed the transcript for topic "
                       "segmentation (%s: %s)", label, type(exc).__name__,
                       str(exc)[:160])
        return None

    if len(vecs) != len(blocks):
        logger.warning("%s: embedding returned %d vector(s) for %d block(s)",
                       label, len(vecs), len(blocks))
        return None

    curve = _similarity_curve(vecs, k)
    valleys = _valleys(curve)
    if not valleys:
        logger.info("%s: the lecture never changes subject sharply enough to "
                    "cut on; using fixed windows", label)
        return None

    depths = [v["depth"] for v in valleys]
    cutoff = statistics.fmean(depths)
    if len(depths) > 1:
        cutoff += threshold * statistics.stdev(depths)

    t0 = words[0]["start"]
    t1 = words[-1]["end"]
    for v in valleys:
        v["t"] = _snap_to_pause(words, blocks[v["gap"]][0]["start"], snap_half)

    strong = [v for v in valleys if v["depth"] >= cutoff]
    kept = _enforce_min(strong, t0, t1, min_s)
    weak = [v for v in valleys if v not in kept]
    edges = _enforce_max(kept, weak, words, t0, t1, min_s, max_s, snap_half)

    # No overlap, unlike fixed windows. Overlap is there to catch an idea a
    # boundary cut in half, and these boundaries are chosen not to do that.
    windows = []
    for i in range(len(edges) - 1):
        a, b = edges[i], edges[i + 1]
        chunk = [w for w in words if a <= w["start"] < b]
        if chunk:
            windows.append({"t0": a, "t1": b, "words": chunk})
    if not windows:
        return None

    spans = sorted((w["t1"] - w["t0"]) for w in windows)
    logger.info("%s: topic segmentation -- %d block(s) of %.0fs, %d valley(s), "
                "%d over the depth cutoff, %d window(s) of %.1f-%.1f min "
                "(median %.1f)", label, len(blocks), block_s, len(valleys),
                len(strong), len(windows), spans[0] / 60, spans[-1] / 60,
                spans[len(spans) // 2] / 60)
    return windows
