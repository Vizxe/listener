# Lecture Processing Pipeline

Turns a folder of math lecture recordings into a synced viewer and, later, a
searchable cross-lecture library.

**Status: all stages built, driven from the browser.** Upload a lecture, pick
how to process it, watch it run, read the notes, search the course.

```bash
.venv/Scripts/python.exe scripts/server.py
```

Then open the URL it prints. It binds `0.0.0.0`, so the same URL works from a
phone or laptop over Tailscale -- the startup banner lists your Tailscale
address directly.

Everything is **per course**. MAT267's lectures, notation and concept index are
separate from any other course's, and nothing is shared between them.

---

## The GPU situation, resolved

The brief specced WhisperX with the faster-whisper `large-v3` backend at
float16 on the GPU. That specific stack cannot run here: this machine has an
**Intel Arc B580 (12 GB)**, and faster-whisper runs on CTranslate2, which
targets **CPU and NVIDIA CUDA only**.

**But the GPU is not lost.** OpenVINO runs Whisper on the Arc perfectly well.
OpenVINO reports the card as a discrete GPU with 11.6 GiB usable and
`FP16 / INT8 / GPU_HW_MATMUL` support, and OpenVINO GenAI's ASR pipeline
supports the word-level timestamps this project depends on.

Measured on this machine against your real 48-minute lecture 11, on the Arc:

| Model | Wall clock | Speed | Words |
|---|---|---|---|
| `whisper-large-v3-turbo` fp16 | 1 m 27 s | **33x** realtime | 5681 |
| `whisper-large-v3` fp16 | 5 m 29 s | **8.8x** realtime | 5730 |

A 40-lecture course of this length is roughly **50 minutes on turbo** or
**3.5 hours on large-v3**. Neither is a bottleneck, which is why the default is
the accurate one rather than the fast one -- see
[Why large-v3 and not turbo](#why-large-v3-and-not-turbo).

One knock-on effect worth naming: the original reason for the strict Phase A /
Phase B split was Whisper and the LLM competing for VRAM. They still both want
the Arc now, so keeping them apart still makes sense -- but your tighter
overall budget is **16 GB of system RAM**.

---

## Layout

```
audio/<course>/         raw lecture files (uploads land here)
config.yaml             every tunable lives here
courses.json            the course registry
courses/<course>/
  notation.md           vocabulary + conventions, editable from the site
  glossary.json         accumulated corrections
index/<course>/
  concepts.json         concept -> every occurrence, canonical marked
  chunks.jsonl          retrieval chunks (one per chapter, long ones split)
  vectors/              embeddings + ids
data/<course>/<lecture_id>/
  transcript.json       Stage 1 output
  math.json             Stage 2 output  (formulas)
  notes.json            Stage 2N output (student notes)
  course_notes.json     Stage 2C output (the lecture written up whole)
  structure.json        Stage 3 output  (chapters)
  lecture.json          merged; the only file the viewer reads
  lecture.js            same payload, script-tag wrapped (see note below)
  audio16k.wav          preprocessed audio for the ASR (cached, regenerable)
  audio.opus            seek-accurate audio for the viewer (see below)
  lectures.json/.js     index of available lectures
viewer/                 static site: open viewer.html directly
scripts/                the pipeline
models/ov-cache/        compiled OpenVINO kernels (cuts startup 23s -> 2s)
logs/                   per-stage run logs
start.cmd               double-click launcher; holds your key, gitignored
```

---

## Setup

```bash
python -m venv .venv
```

```bash
.venv/Scripts/python.exe -m pip install -r requirements.txt
```

ffmpeg must be on PATH (it already is here). Models download from HuggingFace
on first use.

### start.cmd

Double-click `start.cmd` in the project root to bring the site up: it exports
your OpenRouter key, starts the server on the venv's Python, and opens the
browser with the access key already in the URL. Anything you pass it is
forwarded on, so `start.cmd --port 9000` works.

**It is gitignored, because the key lives in it** -- which also means a fresh
clone will not have one. Recreate it with the two lines that matter:

```bat
set "OPENROUTER_API_KEY=sk-or-v1-..."
"%~dp0.venv\Scripts\python.exe" "%~dp0scripts\server.py" --open %*
```

Leaving the placeholder in place is handled rather than half-broken: the
launcher clears the variable instead of exporting the placeholder, so Stage 2C
says plainly that no key is set rather than sending it to OpenRouter and
getting a 401 back. Every other stage is local and unaffected.

The key reaches the pipeline by inheritance -- the server exports nothing
itself, and `jobs.py` spawns each stage with the server's own environment.

---

## Running Stage 1

Upload from the site, or drop files in `audio/<course>/`. The lecture id is the
filename stem, slugified:
`Algebra 07.mp3` becomes `algebra-07`.

```bash
.venv/Scripts/python.exe scripts/stage1_transcribe.py --course MAT267 --lecture algebra-07
```

```bash
.venv/Scripts/python.exe scripts/stage1_transcribe.py --course MAT267 --all
```

Completed lectures are skipped unless you pass `--force`. A lecture that fails
is logged and the batch continues; failures are listed in the summary and the
exit code is 1.

The viewer still opens straight off disk for reading, but the site is the
normal way in now.

Deep links work: `viewer.html?course=MAT267&lecture=algebra-07&t=1342` loads that
lecture and seeks to 22:22.

---

## Choosing a backend

`asr.backend` in `config.yaml`.

**`openvino` (default) -- Intel Arc GPU.**
Fast, and `asr.model` takes an OpenVINO IR model, either a HuggingFace repo id
or a local directory:

- `OpenVINO/whisper-large-v3-fp16-ov` -- **the default**, ~9x realtime
- `OpenVINO/whisper-large-v3-turbo-fp16-ov` -- ~32x realtime, more math errors
- `OpenVINO/whisper-large-v3-turbo-int8-ov` -- fastest, slight further cost

### Why large-v3 and not turbo

Both were run over your 48-minute lecture 11. turbo took 87 s, large-v3 took
329 s. Both are fine; neither is a bottleneck. But turbo makes the kind of
mistake this project cannot absorb, because the errors land on the symbols:

| Audio | turbo | large-v3 |
|---|---|---|
| "delta and epsilon to be zero" | "delta and **hexagon** to be zero" | correct |
| "the **n** plus one term" | "the **m** plus one term" | correct |
| "divided by n plus one **factorial**" | "divided by n plus one **by w**" | correct |
| "**m prime of s**" | "m' ... in s" | correct |
| "integrating **functions**" | "integrating **fun**" | correct |

Stage 2 has to turn this speech into LaTeX. "hexagon" instead of "epsilon"
is not a typo it can recover from. The 4 minutes per lecture is worth it --
across 40 lectures that is about 3.5 hours total.

This backend also uses **`hotwords`**, which biases vocabulary on *every*
internal 30-second window. Whisper's native `initial_prompt` only conditions
the first one, so on a 90-minute lecture hotwords is a real upgrade over what
the original plan could have done.

**`faster_whisper` -- CPU only.**
Roughly 0.2-0.4x realtime on this Ryzen, so hours per lecture. Kept for one
reason: **it is the only backend that reports per-word confidence.**

### The confidence trade-off

OpenVINO returns a single sequence-level score, not per-word probabilities.
So on the GPU backend, words carry no `conf` field and
`meta.has_word_confidence` is `false`.

That costs you the viewer's low-confidence marker -- words the recogniser was
unsure about get a dotted amber underline, which in testing correctly flagged
the one genuine transcription error in a clip. If you want that signal on a
particular lecture, re-run it with `backend: faster_whisper` and wait.

Backends never invent a confidence number to fill the gap. The viewer checks
the field's *type*, so a missing `conf` renders normally rather than marking
every word as suspect.

---

## The two phases

**Phase A -- transcription.** `stage1_transcribe.py`. Nothing talks to
LM Studio.

**Phase B -- LLM passes.** Stages 2, 2N, 3 and 6. Point `llm.base_url` at
anything that serves an OpenAI-compatible API; the code does not care whether
that is LM Studio or a cloud endpoint. Currently configured for LM Studio at
`http://localhost:1234/v1` with `qwen/qwen3.5-9b`, and
`text-embedding-nomic-embed-text-v1.5` for embeddings -- both confirmed loaded
on your server.

**Phase B2 -- the one that leaves the machine.** Stage 2C only, configured
separately under `openrouter:`. Opt-in per run and absent from the default
pipeline, so nothing calls out unless you tick it. Everything downstream of it
-- the index, the embeddings, the search -- stays local.

---

## Why there is a `lecture.js` next to `lecture.json`

You asked for a viewer you can open off disk with no server. Browsers block
`fetch()` against `file://` URLs, so the viewer cannot read `lecture.json`
directly. It loads `lecture.js` -- the identical payload wrapped as
`window.__LECTURE__ = {...}` -- through a `<script>` tag, which `file://` does
allow. Verified with a real headless Chrome load of a `file://` URL.

`lecture.json` remains the canonical, greppable artifact. `lecture.js` is
generated alongside it by `merge.py` and is gitignored.

**If you ever serve this over HTTP**, use a server that implements Range
requests. `python -m http.server` does not, which makes `seekable.end(0)` zero
and silently breaks every seek in the viewer.

---

## Vocabulary biasing

`course/notation.md`'s `## Vocabulary` section is flattened into the ASR
prompt, together with any `terms` and `aliases` keys in
`course/glossary.json`. Whisper honours only ~224 tokens of prompt, so the list
is truncated to `asr.initial_prompt_max_chars` -- put the most-misheard terms
first.

This is the highest-leverage knob in the pipeline, and lecture 11 shows why.
With a generic placeholder word list, the recogniser rendered **Grönwall's
lemma** -- the entire subject of the lecture -- as *"wrong was lemma"* and
*"your own walls, lemma"*. Adding `Gronwall` to the vocabulary fixed every
occurrence.

**A wrong entry actively creates errors.** The same placeholder list contained
`Sylow`, which this course never mentions. large-v3 duly turned *"**zero** is
equal to one"* into *"**Sylow** is equal to 1"*. Removing it fixed that too.
Only list terms the course actually uses.

The exact prompt used is recorded in each `transcript.json` under
`meta.initial_prompt`, so you can tell which run used which vocabulary.

### Segment times are re-cut from the words

OpenVINO produces two timelines by two different mechanisms: segment bounds
come from Whisper's timestamp tokens, word bounds from cross-attention DTW.
**The word times are accurate; the segment times are not.** Cutting the audio
at any `words[i].start` reliably starts on that word, but on lecture 11
segment 0 claimed 0:30 for speech that actually begins at 0:41, and 223 of 373
rows displayed words that did not match the row's own text.

That is what desyncs the viewer -- a row's timestamp gutter and the words shown
in that row describe different moments, so clicking the gutter seeks somewhere
the row is not.

Stage 1 therefore re-cuts segments from the word stream, breaking on sentence
endings and speech pauses (`asr.segmentation` in `config.yaml`). Every row's
time, text and words then come from one source. After the change: 409 rows,
zero mismatches, worst gutter-to-first-word offset 0.00s.

Backends declare `meta.segment_times_reliable`; faster-whisper sets it true
because its segments and words come from a single decoding pass, so its
segments are left alone. Set `segmentation.rebuild_from_words` to `always` or
`never` to override.

One consequence worth knowing: during a genuine pause the highlight stays on
the last spoken word until the next one begins. Lecture 11 has 70 pauses over
two seconds and a longest of 7.4s, so on a deep link into a silence the
highlighted word can sit a few seconds behind the playhead. That is deliberate
-- it shows you where you are rather than blanking out.

### Why the viewer does not play your MP3

The viewer plays `data/<id>/audio.opus`, generated during preprocessing, not
the original file in `audio/`. This is not a preference -- the original cannot
be seeked accurately.

Your lecture MP3 is **VBR** (frames ranging from 32 to 256 kbps). A VBR MP3
carries a Xing header whose seek table is **100 entries** for the whole file.
Across 48 minutes that is one entry per 29 seconds, at one byte of resolution,
so a browser seeking into it interpolates from average bitrate and lands
somewhere near, but not at, the requested time. It then reports the time you
asked for. The word highlight follows that reported clock, so after any seek
the highlight and the audio disagree -- and stay disagreeing until the next
seek. Playing straight through from zero is unaffected, which is exactly the
symptom: normal listening is fine, clicking is not.

Measured in Chrome by capturing the audio actually playing and cross-
correlating it against the known waveform, with a WAV file as the
exact-seeking control:

| Format | Mean seek error | Worst |
|---|---|---|
| Original VBR MP3 | **1.85 s** | **5.76 s** |
| Ogg Opus | **0.02 s** | **0.04 s** |

Ogg carries granule positions, so seeking is sample-accurate. Opus at 32 kbps
mono is also 9.4 MB against the MP3's 43 MB.

It is encoded from the preprocessed WAV, so it shares the transcript's exact
timeline and gets the denoising and level-matching as a bonus. Set
`preprocess.viewer_audio.source: original` if you would rather hear the raw
recording, at the cost of a ~30 ms offset against the transcript.

If the encode ever fails, the viewer falls back to the original file and says
so in a banner rather than pretending seeking works.

### Prompt echo

Whisper fills unclear audio with whatever it was primed with. Lecture 11 opens
with room noise, and the recogniser emitted 136 words of the vocabulary list
before any speech began. `hotwords` makes this likelier, since it re-primes
every window rather than only the first.

Stage 1 detects and removes these stretches by looking for a long exact
substring shared with the prompt -- real speech never reproduces fifty
consecutive characters of it, so it cannot misfire on a lecturer who simply
says "epsilon, delta". Removals are logged and counted in
`meta.timeline.echo_words`.

---

## Demo lectures in `audio/`

Two files are present so you can open the viewer before you have real
transcripts:

- `selftest-demo.wav` -- a tone with a hand-written transcript (has `conf`
  data, so it shows the low-confidence markers).
- `ttscheck-demo.wav` -- synthesised speech with known content, used to verify
  the ASR path end to end.

Delete both once you have real lectures:

```bash
rm audio/selftest-demo.wav audio/ttscheck-demo.wav && rm -rf data/selftest-demo data/ttscheck-demo
```

---

## On a phone

The site is laid out for a phone as well as a desktop, which matters because
the whole point of binding `0.0.0.0` is reading it away from the machine.

**The viewer shows one pane at a time.** Two panes side by side works at
1280px; on a 375px screen it gives each about 330px of height, which is not
enough for either. A tab bar under the chapter strip switches between
Transcript, Notes, Course and Formulas, and the chosen one gets the full
height. It also folds in the right pane's own tab control, so there is one
place to switch rather than two. On a wide screen the bar is hidden and both
panes are visible as before.

**The chapter strip names itself underneath.** Labels drop off the strip below
520px, where a seven-minute chapter of a fifty-minute lecture gets about 50px
-- room for five characters. The original fallback was the `title` tooltip,
which is exactly the thing a touch screen does not have, so on a phone the
chapter titles were unreachable rather than merely abbreviated. A caption bar
under the strip now names the chapter under the playhead -- coloured dot, start
time, full title -- and follows playback. It appears at 820px, the phone
breakpoint, not at 520px: in between, the labels survive but ellipsise down to
a word or two, and there is still no tooltip to recover the rest from. Tapping
a segment seeks into it, so tapping along the strip reads it out chapter by
chapter.

Other phone-specific details: tap targets are 44px under `pointer: coarse`,
and the search box is 16px because anything smaller makes iOS zoom the page
when you focus it.

One layout trap worth recording. The course page scrolled sideways on a phone,
and the cause was the transcription-model `<select>`: a select is sized by its
longest option, and grid and flex children default to `min-width: auto`, so it
refused to shrink and dragged the whole column past the viewport. `min-width: 0`
on the column is the fix, and it is worth reaching for whenever a page mostly
fits but scrolls sideways anyway.

---

## Viewer controls

| Action | Control |
|---|---|
| Play / pause | `space` |
| Seek ±5s | `←` `→` |
| Seek ±30s | `shift` + `←` `→` |
| Jump to a word | click it |
| Jump to a segment | click its timestamp |
| Resume auto-scroll | "Jump to current" button (appears after you scroll manually) |
| Share a moment | "Copy link at time" |
| Jump to a note or formula | click it |
| Jump to a chapter | click the strip at the top |
| See a chapter's title on a phone | the caption under the strip; it follows playback |
| Switch panes | NOTES / COURSE / FORMULAS toggle |
| Show only unsure conversions | "flagged only" checkbox (Formulas view) |
| Search every lecture | `/` or the "Search & index" button (needs the backend) |
| Close the library | `esc` |
| Correct a formula | "fix" on its card (needs the backend) |
| Switch pane on a phone | Transcript / Notes / Course / Formulas tabs |

Hover any word for its timestamp, and its confidence when the backend
reports one.

---

## Using the site

**Home page** lists your courses. Each card shows how many lectures and
concepts it holds, and flags anything sitting unprocessed.

**A course page** does the rest:

- **Upload** by dropping an audio *or video* file. Video is accepted and its
  audio track is taken automatically -- an mp4 lecture recording works fine.
  Upload goes to a `.part` file and is renamed only on success, so an
  interrupted transfer never looks like a lecture waiting to be processed.
- **Choose how to process it**: which transcription model, which steps to run
  (transcribe / formulas / notes / chapters / search index), and whether to
  redo steps that already have output. Processing starts automatically once an
  upload finishes.
- **Watch it run.** Progress, the current step and a live log. The log follows
  new output while you are at the bottom and holds still the moment you scroll
  up to read something, the way a terminal does. Jobs run **one at a time** --
  there is a single GPU, and two transcriptions racing would be slower than
  doing them in turn.
- **Edit `notation.md` for that course** in the browser. It is the highest
  leverage thing you can change, and it now lives per course so MAT267's
  vocabulary never bleeds into another subject.

Each stage runs as a **subprocess of its own CLI**, not inside the server.
Whisper and the LLM give their memory back cleanly when a process exits, a
crash in a native runtime cannot take the site down with it, and what the
browser triggers is exactly the command you would otherwise have typed.

Per-run options never touch `config.yaml`: they are written to a scratch config
the subprocess reads, so two jobs cannot fight over the file and a crash cannot
leave the project configured differently from how you left it.

---

## Reaching it from a phone

`server.host` is `0.0.0.0`, so the site answers on every interface including
Tailscale. The startup banner prints a ready-to-open URL for each address it
finds and labels the Tailscale one.

**That also means it is not just on your machine any more**, and this server
accepts uploads and starts jobs. So when the bind address is not loopback an
**access key** is required by default: it is generated at startup, carried in
the URL, and stored in a cookie on first visit so a phone stays signed in.

```yaml
server:
  host: "0.0.0.0"
  auth: auto     # auto | off | <your own fixed key>
```

`auto` means "require a key unless bound to loopback". `off` disables it, which
is only sensible on `127.0.0.1`. Set a fixed string if you would rather have a
stable URL you can bookmark. There is no user accounts system here -- the key
is the whole of the security model, so treat it as a password and do not port
forward this to the open internet.

---

## Courses

```bash
.venv/Scripts/python.exe scripts/migrate_to_courses.py --course MAT267 --title "Analysis II"
```

That was a one-off to move the original flat layout into a course; it stops
before touching anything if a destination already exists, and `--dry-run`
prints the plan. New courses are made from the site.

Every stage takes `--course`:

```bash
.venv/Scripts/python.exe scripts/stage1_transcribe.py --course MAT267 --all
.venv/Scripts/python.exe scripts/stage6_index.py --course MAT267
```

Search and the concept index are scoped to a course, deliberately. Two subjects
both saying "convergence" do not mean the same thing, and merging them would
make the index worse rather than richer.

---

## The backend

```bash
.venv/Scripts/python.exe scripts/server.py --open
```

The viewer used to be a static file. It still opens off disk and everything
you read -- transcript, notes, formulas, chapter strip, deep links -- works
there. What needs the backend is the library and the correction editor, and
the "Search & index" button only appears when a backend answers.

`scripts/server.py` is stdlib only. It binds `127.0.0.1` and has no
authentication, because it has no network exposure; do not change the bind
address without adding some. It refuses path traversal, caps request bodies,
and serves **Range requests** -- without those `seekable.end(0)` is 0 and every
seek in the viewer fails silently.

| Endpoint | |
|---|---|
| `GET /api/health` | what the index holds, whether lexical and semantic are live |
| `GET /api/search?q=&k=` | hybrid search |
| `GET /api/concepts` | the concept index |
| `POST /api/corrections` | append a correction to `course/glossary.json` |
| `GET /api/reindex` | reload the index after rebuilding it |

### Corrections

The "fix" button on a formula card opens an editor with a live KaTeX preview.
Saving posts to the backend, which merges into `course/glossary.json` --
read-modify-write under a lock, so your hand edits survive. Re-editing the same
card replaces its correction instead of piling up duplicates.

Corrections feed forward: `course_context()` puts them in every later Stage 2
and Stage 2N prompt, so a fix made in lecture 11 conditions lecture 12.

---

## Stage 3 -- chapters

```bash
.venv/Scripts/python.exe scripts/stage3_structure.py --course MAT267 --lecture 20260213-lecture11
```

One pass, one output, as the brief asked -- but it reads the **notes**, not the
raw transcript. Stage 2N has already compressed the lecture 3.4x and given
every block a heading and a span, so a 48-minute lecture becomes ~34 headings
that fit in a single prompt. Nothing to chunk, nothing to reconcile, and the
chapter strip, concept index and search all stay consistent with the notes you
actually read.

The model groups blocks by index and never emits a timestamp; spans come from
the blocks it names. Boundaries are then repaired into a true partition --
overlaps trimmed, gaps closed -- because a strip with a hole in it is a bug.

**A caveat worth knowing.** The 9B is stubborn about granularity: told to
produce ~7 chapters for this lecture it returns 3, one of them 23 minutes long.
The chapters are semantically right, just coarse. Rather than fight it, the
search indexer splits over-long chapters at note-block boundaries, so
retrieval chunks stay ~4-9 minutes while the strip keeps the model's own
sense of structure.

---

## Stage 6 -- the library

```bash
.venv/Scripts/python.exe scripts/stage6_index.py --course MAT267   # everything
.venv/Scripts/python.exe scripts/stage6_index.py --course MAT267 --concepts   # no model needed
```

**Concept index.** Purely deterministic, exactly as specified -- no LLM at
query time and none at build time either. Every concept maps to its
occurrences across lectures, with the first `definition`-type occurrence
marked canonical, rendered as an A-Z index where every entry is a deep link.
Concepts are normalised (lowercase, singular head, no articles) and passed
through the alias map in `course/glossary.json`, which is what lets
"unif. cts" and "uniformly continuous" collapse into one entry.

It also carries `assumed_by`, the prerequisite graph. On lecture 11 that
immediately showed something useful: *fundamental theorem of calculus* is
assumed but never defined -- a gap for another lecture to fill.

**Hybrid search.** BM25 over the chunks, embeddings from the same
OpenAI-compatible endpoint, combined with reciprocal rank fusion. Never
vector-only: results carry a `lexical` / `semantic` badge, and the difference
is visible. Searching "Sylow", which this course never mentions, returns
*semantic matches only* at half the score of a real hit -- exactly why
mathematical names need a lexical leg.

Chunk granularity is one chapter, not fixed token windows. Chapters longer
than `search.max_chunk_seconds` are split at note-block boundaries -- still
semantic units with their own headings -- and each piece keeps its parent
chapter's title, type and concepts.

A chunk's text is the chapter summary, then the Stage 2C sections overlapping
it, then the Stage 2N notes, then the transcript. Both note passes are indexed
because they fail differently: the rough notes carry the maths, and the
write-up carries the wording of someone who already knew how the lecture
ended. Stage 2C also gets **one extra chunk per lecture**, `<id>:course-notes:0`,
holding the parts of the write-up that belong to no single chapter -- title,
summary, prerequisites, takeaways, open questions, section headings. That is
the only text in the project describing a lecture *as a whole*, so without it
"which lecture proved completeness" has nothing to match. Its deep link
carries `&pane=course` so the hit opens on the pane you matched. This chunk is
built before the chapter check, so a lecture Stage 3 has not reached yet is
still searchable.

Nothing about this leaves the machine: the write-up is produced by a cloud
model, but it is indexed and searched by the same local BM25 and the same
LM Studio embeddings as everything else.

If embeddings fail the build does not: it logs, keeps the lexical index, and
search degrades to BM25 rather than breaking.

---

## Stage 2N -- the notes pass

This is what the right-hand pane shows: student-style notes written from the
transcript, anchored to the audio.

```bash
.venv/Scripts/python.exe scripts/stage2_notes.py --course MAT267 --lecture 20260213-lecture11
```

Iterate on the prompt against one stretch without rewriting `notes.json`:

```bash
.venv/Scripts/python.exe scripts/stage2_notes.py --course MAT267 --lecture 20260213-lecture11 --window 480-1020
```

Windows are **six minutes** here against Stage 2's ninety seconds. Summarising
needs context where extraction needs precision -- you cannot write a coherent
note about a proof while looking through a ninety-second slot.

Each block carries a `heading`, a `kind` (definition / theorem / proof /
example / method / remark / admin), a body in light markdown with inline
`$maths$`, and one to three `key_points`.

**Timestamps come from an `anchor_quote`**, not from the model: it copies 5-12
words verbatim from where the block's material starts, and Stage 2N locates
that back in the word stream. Same reasoning as the math pass -- a model asked
to copy a phrase is reliable, a model asked to keep a clock is not, and a bad
quote is detectable. Notes whose quote cannot be found get pinned to their
window and marked "approx. time" in the viewer rather than silently claiming a
position.

Where `math.json` exists its validated LaTeX is handed to this pass, so the
notes inherit formulas that have already been checked rather than re-deriving
them from the same garbled speech.

Measured on lecture 11: 10 windows, **31 blocks, 1506 words -- 3.7x shorter
than the transcript**, every block anchored, 0 failed windows, ~4.5 minutes.

### How much the model should see

`notes.context_mode` decides what sits in the prompt alongside the window:

| mode | what it adds | tokens / lecture | dup headings | anchored | compression |
|---|---|---|---|---|---|
| `window` | nothing | 21k | 1 | 34/34 | 3.4x |
| `neighbours` | ~6 min either side | 49k | 1 | 27/27 | **3.5x** |
| `preceding` | everything **before** it, plus the notes written for it | 80k | **0** | **31/31** | 3.0x |
| `full` | the entire lecture | 98k | 4 | 28/29 | 3.0x |

**`preceding` is the default.** Measured on lecture 11 it is the only mode with
no repeated headings at all, everything anchored, and headings that read as one
argument rather than a list -- existence and uniqueness, Euler method, fixed
point, Gronwall's statement, the first proof by iteration, the remainder bound,
the connection to Picard, *then* proof two and the integrating factor,
continuous dependence, the two cases, uniqueness as a corollary.

**Handing it the whole lecture (`full`) is worse, not better.** Two failures:

- It re-summarises the lecture's arc in every window rather than covering what
  is new there, which is where the duplicate headings come from.
- It writes about material it can see but was not assigned. On lecture 11 it
  produced *"Proof Two via Integrating Factor"* anchored at **05:42**, when the
  second proof begins at 24:10 -- it had found a plausible-sounding quote
  inside its own window while describing something eighteen minutes later.

That second point matters beyond this setting: the anchor-quote mechanism
catches a quote that **is not in the window**, which is fabrication. It cannot
catch a real quote attached to the wrong material. Provenance checking is not
comprehension checking.

`preceding` avoids it structurally rather than by instruction -- there is no
later material in the prompt to wander into. Under `preceding` all the
second-proof notes land at 23:59, 25:41 and 26:24, each at match 1.00.

The trade is length: `preceding` keeps more detail (3.0x compression against
`neighbours`' 3.5x) and costs 63% more tokens, about five minutes a lecture
against three and a half. If you prefer terser notes, `neighbours` is the one
to switch to.

None of this is a context-*window* limit -- 98k fits comfortably and every run
completed cleanly. It is a 9B's effective attention span. On a larger model the
balance would likely tip the other way.

### Rendering

The viewer renders the notes with a deliberately small markdown subset --
paragraphs, bullets, `**bold**`, `$inline$` and `$$display$$` maths. It is
about 70 lines of JS rather than a markdown library, and it builds text nodes
rather than assigning `innerHTML`, so note content cannot inject markup.

Two quirks of real model output it handles: maths nested inside bold
(`**$y$**` is common, and naive handling shows raw dollar signs), and bullets
written run-together on one line instead of one per line.

---

## Stage 2C -- course notes

The **Course** tab beside the notes. Same lecture, written up once by a
stronger model that was shown the whole thing.

```bash
set OPENROUTER_API_KEY=sk-or-...
.venv/Scripts/python.exe scripts/stage2_course_notes.py --course MAT267 --lecture 20260213-lecture11
.venv/Scripts/python.exe scripts/stage2_course_notes.py --course MAT267 --lecture 20260213-lecture11 --print
```

### Why a second notes pass at all

Stage 2N's six-minute window is not a tuning choice you can raise your way out
of; it is the ceiling on what that pass can ever do. Inside one window the
model cannot open with what the lecture turned out to be about, cannot fold
the proof at 00:41 into the theorem at 00:12, and cannot drop a definition the
lecturer replaced ten minutes later. Every one of those needs the whole
lecture at once, and "the whole lecture at once" is the one thing a window
sweep structurally cannot offer.

So this stage does the opposite of every other LLM pass here. **No windows, no
sweep, no reconciliation.** The entire transcript, the entire set of Stage 2N
notes, the validated LaTeX from Stage 2 and the Stage 3 chapters go into a
single prompt, and one call comes back with a document: a title, a summary,
what it assumes, sections in *reading* order, what to remember, and what was
left open.

### Where the timestamps come from

The Stage 2N notes are handed over as well as the transcript, and they are not
redundant -- they carry the times. Each note goes in with its id, and each
section names the ids it drew on in `covers`. A section merging four minutes
of setup with a callback twenty minutes later gets an honest span from the
union of the notes it cites.

That is a claim the model makes about its own work, so it is checked. An id
this lecture does not have is dropped and counted (`meta.unknown_note_ids`),
never quietly accepted. A section citing no usable note falls back to locating
its `anchor_quote` in the word stream, the same way Stages 2 and 2N derive
every span. When that fails too the section is pinned after the one before it
and flagged `anchored: false`, which the viewer shows as an amber
**approx. time** -- deliberately the previous section's end rather than the
furthest point reached so far, because a legitimately wide merged section
would otherwise strand the next one in the closing minutes.

### The endpoint

This is the **only stage that leaves the machine**, so it is the only one that
is opt-in. It is in the web UI's stage list but *not* in `DEFAULT_STAGES`: a
default run stays local and free. Configure it under `openrouter:` in
`config.yaml`, and leave `api_key` empty -- `config.yaml` is in git, and the
key is read from `OPENROUTER_API_KEY` instead. A missing key fails at once
with the variable named, and stops the batch rather than repeating the same
failure per lecture.

Point `course_notes.llm_section` at `llm` instead and the whole stage runs on
LM Studio, at some cost in quality: the value here is a model that can hold
ninety minutes in its head.

### What the viewer does with it

Sections are in reading order, not clock order, so the pane has nothing sorted
to binary-search -- `findSection()` scans and picks the **tightest** span
containing the playhead, which lets a short section win over the long merged
one it sits inside. Otherwise it is the notes pane: same markdown subset, same
KaTeX, same click-to-seek, same current-item highlight. A section reaching
past its start time shows a quiet `to 12:04`, because the click drops you at
the start and the reader should know how far it runs.

## Stage 2 -- the math pass

```bash
.venv/Scripts/python.exe scripts/stage2_math.py --course MAT267 --lecture 20260213-lecture11
```

Iterate on the prompt against one stretch of a lecture without rewriting
`math.json`:

```bash
.venv/Scripts/python.exe scripts/stage2_math.py --course MAT267 --lecture 20260213-lecture11 --window 560-760
```

`--window START-END` prints every item side by side with the words that
produced it, and deliberately writes nothing.

### The model is never asked for timestamps

It is asked for `source_text` -- the exact words it converted -- and Stage 2
locates that quote back in the word stream to derive `t_start` / `t_end`.
An LLM asked to copy a phrase is far more reliable than one asked to keep a
clock, and a bad quote is *detectable* where a bad timestamp is not: each item
carries a `match_ratio`, and anything under 0.5 gets its confidence capped and
a warning logged.

On lecture 11, 152 of 158 quotes matched 80% or more of the transcript.

### What runs on top of the model output

- **Fragment collapsing.** The model tends to emit both a statement and the
  pieces inside it -- `y(t)` *and* `y(t) \leq 1` from one phrase. Items quoting
  the same words collapse to the longest LaTeX. Note that this is deliberately
  *not* decided by confidence: the fragment usually scores higher than the
  complete statement.
- **Overlap dedup.** Windows overlap by 20s, so statements near a seam get
  extracted twice; near-identical LaTeX within 8s collapses.
- **Prose detection.** "Y(t) < \text{something involving } Y" is a description
  of maths, not maths. Those get `ambiguous: true`, confidence capped at 0.4
  and a note, so they surface in the viewer with a warning rather than passing
  as a confident conversion. `\text{constant}` is left alone -- a label is fine,
  a description is not.

### Measured on lecture 11

41 windows, 158 items, 12 ambiguous, **0 retries and 0 failed windows**.
62k prompt + 19k completion tokens in 447s (~7.5 minutes).

Two things make it that reliable: LM Studio's `json_schema` structured output
constrains the shape at generation time, and `enable_thinking: false` stops
qwen3.5 burning the token budget narrating before it answers.

### LaTeX escapes that JSON eats

Worth knowing about because it fails silently. JSON treats `` and `` as
escapes for backspace and form feed. A model that writes `"oldsymbol"`
instead of `"\boldsymbol"` therefore hands the parser a control character,
and the command name loses its first letter: `oldsymbol{\epsilon}` arrives
as `<BS>oldsymbol{\epsilon}`. It renders as an error in the viewer and is
invisible in a diff, because writing the JSON back out re-escapes it.

`scripts/llm.py` repairs this on every parsed reply and logs how many it
fixed. Neither character can legitimately appear in lecture notes, so putting
the backslash back is safe. It caught one occurrence in lecture 11.

### If you swap in a cloud endpoint

Only `llm.base_url` and `llm.model` need to change. Two local quirks are
handled defensively in `scripts/llm.py` and are harmless elsewhere: LM Studio
rejects `response_format: json_object` (it wants `json_schema`), and with a
reasoning model the reply can arrive in `reasoning_content` while `content` is
empty -- the client reads whichever is populated.

One local quirk is *not* harmless elsewhere, which is why the `openrouter:`
section exists as a worked example. LM Studio's `enable_thinking` is a
chat-template argument; a cloud endpoint rejects the request outright when it
arrives. `send_thinking_flag: false` stops sending it. Alongside it,
`structured_outputs: false` stops sending the JSON schema for a model that
does not implement structured outputs (the reply is still parsed out of
whatever comes back), `api_key_env` names an environment variable to read the
key from, and `headers.referer` / `headers.title` become `HTTP-Referer` and
`X-Title`. Every one of these defaults to the existing local behaviour, so the
`llm:` section is unchanged.

---

## Still open

Nothing in the brief is unbuilt, but three things are worth your attention:

- **`course/notation.md` is still my starter list.** It is the highest-leverage
  knob in the pipeline and it should be yours.
- **Stage 2's `kind` collapses** -- 88 of 92 formulas come back as
  `expression`. Stage 3 and the concept index lean on chapter types rather
  than formula types, so this matters less than it did, but it is still wrong.
- **Only one real lecture exists.** Every `--all` path works but has never run
  over a batch.
- **Stage 2C has been run against a stub, not a real model.** Every path
  around the call is exercised -- prompt assembly, the `covers` mapping, both
  fallbacks, the index, the pane -- but the prompt itself has not been tuned
  against real output the way Stage 2N's was, and the section count and
  granularity are guesses until it has.
- **The access key is the entire security model.** Fine behind Tailscale; do
  not expose this to the open internet.

---

## Vendored dependencies

`viewer/vendor/katex/` holds KaTeX 0.18.4 (0.6 MB: the minified CSS and JS plus
20 woff2 fonts). It is committed rather than fetched from a CDN so the viewer
works offline and off disk, with no build step. Only woff2 fonts are kept --
the woff and ttf fallbacks exist for browsers that predate this project by a
decade.
