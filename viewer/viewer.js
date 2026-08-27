/* Lecture viewer -- Stage 1 (transcript + click-to-seek).
 *
 * Opens straight off disk. Browsers block fetch() against file: URLs, so the
 * lecture payload arrives as a <script> that assigns window.__LECTURE__.
 *
 * Deep links:  viewer.html?lecture=algebra-07&t=1342
 */
"use strict";

(function () {

  var els = {
    picker:     document.getElementById("lecture-picker"),
    meta:       document.getElementById("lecture-meta"),
    status:     document.getElementById("status"),
    transcript: document.getElementById("transcript"),
    wordCount:  document.getElementById("word-count"),
    jump:       document.getElementById("jump-to-current"),
    audio:      document.getElementById("audio"),
    copyLink:   document.getElementById("copy-link"),
    paneRight:  document.getElementById("pane-right"),
    math:       document.getElementById("math"),
    notes:      document.getElementById("notes"),
    course:     document.getElementById("course"),
    rightCount: document.getElementById("right-count"),
    tabNotes:   document.getElementById("tab-notes"),
    tabCourse:  document.getElementById("tab-course"),
    tabMath:    document.getElementById("tab-math"),
    flaggedWrap: document.getElementById("flagged-wrap"),
    onlyFlagged: document.getElementById("only-flagged"),
    strip:       document.getElementById("chapter-strip"),
    openLibrary: document.getElementById("open-library"),
    library:     document.getElementById("library"),
    libQ:        document.getElementById("lib-q"),
    libBody:     document.getElementById("lib-body"),
    libNote:     document.getElementById("lib-note"),
    libClose:    document.getElementById("lib-close"),
    libTabSearch: document.getElementById("lib-tab-search"),
    libTabIndex: document.getElementById("lib-tab-index"),
    mtabs:       document.getElementById("mobile-tabs")
  };

  var lecture = null;
  var wordSpans = [];       // index-aligned with lecture.words
  var segSpans = [];        // index-aligned with lecture.segments
  var wordStarts = [];      // plain number array -> cheap binary search
  var curWord = -1;
  var curSeg = -1;
  var mathCards = [];       // index-aligned with lecture.math
  var mathStarts = [];
  var curMath = -1;
  var noteEls = [];         // index-aligned with lecture.notes
  var noteStarts = [];
  var curNote = -1;
  // Course-note sections are in reading order, not clock order, so they get
  // no sorted starts array to binary-search: see findSection().
  var sectionEls = [];      // index-aligned with lecture.course_notes.sections
  var sections = [];
  var curSection = -1;
  var rightTab = "notes";
  var courseId = null;
  var stripSegs = [];
  var playhead = null;
  var curChapter = -1;
  var autoScroll = true;

  // ------------------------------------------------------------- helpers --

  function qs(name) {
    var m = new RegExp("[?&]" + name + "=([^&#]*)").exec(location.search);
    return m ? decodeURIComponent(m[1].replace(/\+/g, " ")) : null;
  }

  // The id lands in a <script src>, so it must not be able to walk the tree.
  function safeId(id) {
    return (id && /^[A-Za-z0-9][A-Za-z0-9._-]*$/.test(id) && id.indexOf("..") === -1)
      ? id : null;
  }

  function hhmmss(t) {
    t = Math.max(0, Math.floor(t || 0));
    var h = Math.floor(t / 3600), m = Math.floor((t % 3600) / 60), s = t % 60;
    var mm = (m < 10 && h > 0 ? "0" : "") + m;
    return (h > 0 ? h + ":" : "") + mm + ":" + (s < 10 ? "0" : "") + s;
  }

  function setStatus(msg, isError) {
    if (!msg) { els.status.hidden = true; return; }
    els.status.hidden = false;
    els.status.textContent = msg;
    els.status.className = "status" + (isError ? " error" : "");
  }

  function loadScript(src, onload, onerror) {
    var s = document.createElement("script");
    s.src = src;
    s.onload = onload;
    s.onerror = onerror;
    document.head.appendChild(s);
  }

  // ---------------------------------------------------------- transcript --

  function render() {
    var frag = document.createDocumentFragment();
    var words = lecture.words || [];
    var segs = lecture.segments || [];

    wordSpans = new Array(words.length);
    segSpans = new Array(segs.length);
    wordStarts = new Array(words.length);
    for (var i = 0; i < words.length; i++) wordStarts[i] = words[i].start;

    // Walk words and segments together; both are time-ordered.
    var wi = 0;
    for (var si = 0; si < segs.length; si++) {
      var seg = segs[si];

      var row = document.createElement("div");
      row.className = "seg";

      var time = document.createElement("span");
      time.className = "seg-time";
      time.textContent = hhmmss(seg.start);
      time.dataset.t = seg.start;
      row.appendChild(time);

      var text = document.createElement("span");
      text.className = "seg-text";

      var placed = 0;
      var limit = (si + 1 < segs.length) ? segs[si + 1].start : Infinity;
      while (wi < words.length && words[wi].start < limit) {
        var w = words[wi];
        var span = document.createElement("span");
        // conf is optional -- the OpenVINO backend cannot report it. Test the
        // type, not just presence: null < 0.5 is true in JS, which would flag
        // every single word as low confidence.
        var hasConf = typeof w.conf === "number";
        span.className = "w" + (hasConf && w.conf < 0.5 ? " lowconf" : "");
        span.textContent = w.w;
        span.dataset.i = wi;
        span.title = hasConf
          ? hhmmss(w.start) + "  conf " + w.conf.toFixed(2)
          : hhmmss(w.start);
        text.appendChild(span);
        text.appendChild(document.createTextNode(" "));
        wordSpans[wi] = span;
        wi++; placed++;
      }

      // No word timestamps for this stretch -- still clickable by segment.
      if (placed === 0) text.textContent = seg.text || "";

      row.appendChild(text);
      segSpans[si] = row;
      frag.appendChild(row);
    }

    els.transcript.innerHTML = "";
    els.transcript.appendChild(frag);

    els.wordCount.textContent =
      words.length.toLocaleString() + " words, " +
      segs.length.toLocaleString() + " segments";
  }

  // Last word whose start <= t.
  function findWord(t) {
    var lo = 0, hi = wordStarts.length - 1, best = -1;
    while (lo <= hi) {
      var mid = (lo + hi) >> 1;
      if (wordStarts[mid] <= t) { best = mid; lo = mid + 1; } else { hi = mid - 1; }
    }
    return best;
  }

  function segOfWord(i) {
    if (i < 0) return -1;
    var segs = lecture.segments || [];
    var t = lecture.words[i].start;
    var lo = 0, hi = segs.length - 1, best = -1;
    while (lo <= hi) {
      var mid = (lo + hi) >> 1;
      if (segs[mid].start <= t) { best = mid; lo = mid + 1; } else { hi = mid - 1; }
    }
    return best;
  }

  // ---------------------------------------------------------- notes pane --

  // A deliberately small markdown subset: paragraphs, "- " bullets, **bold**,
  // $inline$ and $$display$$ maths. Everything is built as text nodes, never
  // innerHTML, so note content cannot inject markup.

  function renderTex(tex, el, display) {
    try {
      if (!window.katex) throw new Error("KaTeX not loaded");
      katex.render(tex, el, { throwOnError: true, displayMode: !!display });
    } catch (err) {
      el.className = "note-badmath";
      el.textContent = tex;
      el.title = "KaTeX could not render this: " + (err && err.message ? err.message : err);
    }
  }

  // A fresh regex per call: this function recurses (bold can contain maths),
  // and a shared /g regex would carry lastIndex across the recursion.
  function inlineInto(parent, text) {
    var re = /\$\$([\s\S]+?)\$\$|\$([^$\n]+?)\$|\*\*([\s\S]+?)\*\*/g;
    var last = 0, m;
    while ((m = re.exec(text)) !== null) {
      if (m.index > last) {
        parent.appendChild(document.createTextNode(text.slice(last, m.index)));
      }
      if (m[1] !== undefined) {
        var d = document.createElement("div");
        d.className = "note-display";
        renderTex(m[1].trim(), d, true);
        parent.appendChild(d);
      } else if (m[2] !== undefined) {
        var s = document.createElement("span");
        renderTex(m[2].trim(), s, false);
        parent.appendChild(s);
      } else {
        // "**$y$**" is common. Recurse so the maths inside bold still renders
        // instead of showing raw dollar signs.
        var b = document.createElement("strong");
        inlineInto(b, m[3]);
        parent.appendChild(b);
      }
      last = re.lastIndex;
    }
    if (last < text.length) {
      parent.appendChild(document.createTextNode(text.slice(last)));
    }
  }

  // The model sometimes writes bullets inline -- "- first. - second." -- all on
  // one line. Split those out, but only outside maths, where a bare "-" is
  // usually a minus sign rather than a bullet.
  function splitInlineBullets(text) {
    var masked = text.replace(/\$\$[\s\S]+?\$\$|\$[^$\n]+?\$/g,
                              function (m) { return m.replace(/./g, " "); });
    var cuts = [];
    var re = /(^|\s)[-*]\s+/g, m;
    while ((m = re.exec(masked)) !== null) {
      cuts.push([m.index + m[1].length, re.lastIndex]);
    }
    if (cuts.length < 2) return null;
    var items = [];
    for (var i = 0; i < cuts.length; i++) {
      var from = cuts[i][1];
      var to = (i + 1 < cuts.length) ? cuts[i + 1][0] : text.length;
      var piece = text.slice(from, to).trim();
      if (piece) items.push(piece);
    }
    return items.length >= 2 ? items : null;
  }

  function markdownLite(container, body) {
    var chunks = String(body).replace(/\r/g, "").split(/\n\s*\n/);
    chunks.forEach(function (chunk) {
      chunk = chunk.replace(/^\n+|\n+$/g, "");
      if (!chunk) return;
      var lines = chunk.split("\n");
      var bullets = lines.filter(function (l) { return /^\s*[-*]\s+/.test(l); });

      if (bullets.length && bullets.length === lines.length) {
        var ul = document.createElement("ul");
        lines.forEach(function (l) {
          var li = document.createElement("li");
          inlineInto(li, l.replace(/^\s*[-*]\s+/, ""));
          ul.appendChild(li);
        });
        container.appendChild(ul);
        return;
      }

      var whole = chunk.trim();
      var only = /^\$\$([\s\S]+)\$\$$/.exec(whole);
      if (only) {
        var d = document.createElement("div");
        d.className = "note-display";
        renderTex(only[1].trim(), d, true);
        container.appendChild(d);
        return;
      }

      var joined = lines.join(" ");
      var inline = splitInlineBullets(joined);
      if (inline) {
        var ul2 = document.createElement("ul");
        inline.forEach(function (item) {
          var li = document.createElement("li");
          inlineInto(li, item);
          ul2.appendChild(li);
        });
        container.appendChild(ul2);
        return;
      }

      var p = document.createElement("p");
      inlineInto(p, joined);
      container.appendChild(p);
    });
  }

  function renderNotes() {
    var notes = (lecture.notes || []).slice().sort(function (a, b) {
      return a.t_start - b.t_start;
    });
    lecture.notes = notes;
    noteEls = new Array(notes.length);
    noteStarts = notes.map(function (n) { return n.t_start; });
    els.notes.innerHTML = "";

    if (!notes.length) {
      var empty = document.createElement("div");
      empty.className = "pane-empty";
      empty.appendChild(document.createTextNode("No notes for this lecture yet. Run:"));
      empty.appendChild(document.createElement("br"));
      var code = document.createElement("code");
      code.textContent = "python scripts/stage2_notes.py --lecture " + lecture.lecture_id;
      empty.appendChild(code);
      els.notes.appendChild(empty);
      return;
    }

    var frag = document.createDocumentFragment();
    notes.forEach(function (n, i) {
      var art = document.createElement("article");
      art.className = "note";
      art.dataset.i = i;

      var head = document.createElement("div");
      head.className = "note-head";

      var time = document.createElement("span");
      time.className = "note-time";
      time.textContent = hhmmss(n.t_start);
      head.appendChild(time);

      var chip = document.createElement("span");
      chip.className = "chip " + (n.kind || "remark");
      chip.textContent = n.kind || "remark";
      head.appendChild(chip);

      var h = document.createElement("span");
      h.className = "note-heading";
      h.textContent = n.heading;
      head.appendChild(h);

      if (n.anchored === false) {
        var warn = document.createElement("span");
        warn.className = "note-unanchored";
        warn.textContent = "approx. time";
        warn.title = "The model's anchor quote was not found in the transcript, "
                   + "so this note is pinned to the start of its window.";
        head.appendChild(warn);
      }
      art.appendChild(head);

      var body = document.createElement("div");
      body.className = "note-body";
      markdownLite(body, n.body);
      art.appendChild(body);

      if (n.key_points && n.key_points.length) {
        var box = document.createElement("div");
        box.className = "note-points";
        var ul = document.createElement("ul");
        n.key_points.forEach(function (p) {
          var li = document.createElement("li");
          inlineInto(li, p);
          ul.appendChild(li);
        });
        box.appendChild(ul);
        art.appendChild(box);
      }

      noteEls[i] = art;
      frag.appendChild(art);
    });
    els.notes.appendChild(frag);
  }

  function findNote(t) {
    var lo = 0, hi = noteStarts.length - 1, best = -1;
    while (lo <= hi) {
      var mid = (lo + hi) >> 1;
      if (noteStarts[mid] <= t) { best = mid; lo = mid + 1; } else { hi = mid - 1; }
    }
    return best;
  }

  els.notes.addEventListener("click", function (ev) {
    var art = ev.target.closest ? ev.target.closest(".note") : null;
    if (!art) return;
    var n = lecture.notes[+art.dataset.i];
    if (n) seek(n.t_start);
  });

  // --------------------------------------------------- course notes pane --

  // Stage 2C's write-up of the whole lecture: one document with a title, a
  // summary, sections in READING order, and what to remember. Same markdown
  // subset and the same click-to-seek as the notes pane; what differs is that
  // a section can span two distant stretches of the lecture, because it was
  // written by something that had already heard both.

  function listInto(parent, items, cls) {
    if (!items || !items.length) return;
    var ul = document.createElement("ul");
    if (cls) ul.className = cls;
    items.forEach(function (item) {
      var li = document.createElement("li");
      inlineInto(li, String(item));
      ul.appendChild(li);
    });
    parent.appendChild(ul);
  }

  function courseAside(title, items) {
    if (!items || !items.length) return null;
    var box = document.createElement("section");
    box.className = "course-aside";
    var h = document.createElement("h3");
    h.textContent = title;
    box.appendChild(h);
    listInto(box, items);
    return box;
  }

  function renderCourseNotes() {
    var doc = lecture.course_notes || null;
    sections = (doc && doc.sections) || [];
    sectionEls = new Array(sections.length);
    curSection = -1;
    els.course.innerHTML = "";

    if (!doc || !sections.length) {
      var empty = document.createElement("div");
      empty.className = "pane-empty";
      empty.appendChild(document.createTextNode(
        "No course notes for this lecture yet. They are written from the "
        + "whole transcript and the notes beside them, by the model in "
        + "config.yaml under "));
      var cfg = document.createElement("code");
      cfg.textContent = "course_notes.llm_section";
      empty.appendChild(cfg);
      empty.appendChild(document.createTextNode(". Run:"));
      empty.appendChild(document.createElement("br"));
      var code = document.createElement("code");
      code.textContent = "python scripts/stage2_course_notes.py --course "
        + (courseId || "COURSE") + " --lecture " + lecture.lecture_id;
      empty.appendChild(code);
      els.course.appendChild(empty);
      return;
    }

    var frag = document.createDocumentFragment();

    var head = document.createElement("header");
    head.className = "course-head";
    if (doc.title) {
      var h1 = document.createElement("h2");
      h1.className = "course-title";
      inlineInto(h1, doc.title);
      head.appendChild(h1);
    }
    if (doc.summary) {
      var sum = document.createElement("div");
      sum.className = "course-summary";
      markdownLite(sum, doc.summary);
      head.appendChild(sum);
    }
    if (doc.prerequisites && doc.prerequisites.length) {
      var pre = document.createElement("p");
      pre.className = "course-prereq";
      var lbl = document.createElement("span");
      lbl.className = "course-prereq-label";
      lbl.textContent = "Assumes";
      pre.appendChild(lbl);
      inlineInto(pre, doc.prerequisites.join(" · "));
      head.appendChild(pre);
    }
    if (head.childNodes.length) frag.appendChild(head);

    sections.forEach(function (s, i) {
      var art = document.createElement("article");
      art.className = "note section";
      art.dataset.i = i;

      var h = document.createElement("div");
      h.className = "note-head";

      var time = document.createElement("span");
      time.className = "note-time";
      time.textContent = hhmmss(s.t_start);
      h.appendChild(time);

      var chip = document.createElement("span");
      chip.className = "chip " + (s.kind || "remark");
      chip.textContent = s.kind || "remark";
      h.appendChild(chip);

      var title = document.createElement("span");
      title.className = "note-heading";
      title.textContent = s.heading;
      h.appendChild(title);

      // A section built from several rough notes usually spans more of the
      // lecture than its start time suggests, and the reader is about to be
      // dropped at that start time, so say how far it reaches.
      if (s.t_end > s.t_start + 1) {
        var span = document.createElement("span");
        span.className = "section-span";
        span.textContent = "to " + hhmmss(s.t_end);
        span.title = (s.covers && s.covers.length)
          ? "Written from " + s.covers.length + " of the notes beside this pane"
          : "Located in the transcript by quote";
        h.appendChild(span);
      }

      if (s.anchored === false) {
        var warn = document.createElement("span");
        warn.className = "note-unanchored";
        warn.textContent = "approx. time";
        warn.title = "This section cited no note and its anchor quote was not "
                   + "found in the transcript, so it is pinned after the "
                   + "section before it.";
        h.appendChild(warn);
      }
      art.appendChild(h);

      var body = document.createElement("div");
      body.className = "note-body";
      markdownLite(body, s.body);
      art.appendChild(body);

      if (s.key_points && s.key_points.length) {
        var box = document.createElement("div");
        box.className = "note-points";
        listInto(box, s.key_points);
        art.appendChild(box);
      }

      sectionEls[i] = art;
      frag.appendChild(art);
    });

    var take = courseAside("What to remember", doc.takeaways);
    if (take) frag.appendChild(take);
    var open = courseAside("Left open", doc.open_questions);
    if (open) { open.classList.add("course-open"); frag.appendChild(open); }

    if (doc.meta && doc.meta.model) {
      var by = document.createElement("p");
      by.className = "course-by";
      by.textContent = "Written up by " + doc.meta.model;
      frag.appendChild(by);
    }

    els.course.appendChild(frag);
  }

  // Reading order is not clock order, so there is nothing sorted to bisect.
  // A linear scan over a dozen or so sections is cheaper than maintaining a
  // second ordering, and picking the tightest containing span means a short
  // section wins over the long one it sits inside.
  function findSection(t) {
    var best = -1, bestSpan = Infinity;
    for (var i = 0; i < sections.length; i++) {
      var s = sections[i];
      var end = (s.t_end > s.t_start) ? s.t_end : s.t_start;
      if (s.t_start <= t && t < end) {
        var span = end - s.t_start;
        if (span < bestSpan) { best = i; bestSpan = span; }
      }
    }
    return best;
  }

  els.course.addEventListener("click", function (ev) {
    var art = ev.target.closest ? ev.target.closest(".section") : null;
    if (!art) return;
    var s = sections[+art.dataset.i];
    if (s) seek(s.t_start);
  });

  // ------------------------------------------------------------- tabs -----

  function setTab(which) {
    if (which !== "notes" && which !== "course" && which !== "math") which = "notes";
    rightTab = which;
    els.notes.hidden = which !== "notes";
    els.course.hidden = which !== "course";
    els.math.hidden = which !== "math";
    // The flagged-only filter belongs to the formula pane alone.
    els.flaggedWrap.hidden = which !== "math";
    els.tabNotes.classList.toggle("is-active", which === "notes");
    els.tabCourse.classList.toggle("is-active", which === "course");
    els.tabMath.classList.toggle("is-active", which === "math");
    updateRightCount();
  }

  function updateRightCount() {
    if (!lecture) return;
    if (rightTab === "notes") {
      var n = (lecture.notes || []).length;
      els.rightCount.textContent = n ? n + " notes" : "";
    } else if (rightTab === "course") {
      els.rightCount.textContent = sections.length
        ? sections.length + " sections" : "";
    } else {
      var items = lecture.math || [];
      var flagged = items.filter(function (m) {
        return m.ambiguous || (m.confidence !== undefined && m.confidence < 0.6);
      }).length;
      els.rightCount.textContent = items.length + " items"
        + (flagged ? ", " + flagged + " flagged" : "");
    }
  }

  els.tabNotes.addEventListener("click", function () { setTab("notes"); });
  els.tabCourse.addEventListener("click", function () { setTab("course"); });
  els.tabMath.addEventListener("click", function () { setTab("math"); });

  // ------------------------------------------------------- phone panes ----

  // Narrow screens show one pane at a time. The body class does the hiding in
  // CSS, so on a wide screen these are inert and both panes stay visible.
  function setMobilePane(which) {
    document.body.classList.toggle("m-transcript", which === "transcript");
    document.body.classList.toggle("m-right", which !== "transcript");
    if (which !== "transcript") setTab(which);
    [].forEach.call(els.mtabs.querySelectorAll(".mtab"), function (b) {
      b.classList.toggle("is-active", b.dataset.pane === which);
    });
    // Re-anchor after a switch: the pane was display:none, so the browser
    // could not have scrolled it while it was hidden.
    if (autoScroll) {
      if (which === "transcript" && curWord >= 0 && wordSpans[curWord]) {
        wordSpans[curWord].scrollIntoView({ block: "center" });
      } else if (which === "notes" && curNote >= 0 && noteEls[curNote]) {
        noteEls[curNote].scrollIntoView({ block: "nearest" });
      } else if (which === "course" && curSection >= 0 && sectionEls[curSection]) {
        sectionEls[curSection].scrollIntoView({ block: "nearest" });
      } else if (which === "math" && curMath >= 0 && mathCards[curMath]) {
        mathCards[curMath].scrollIntoView({ block: "nearest" });
      }
    }
  }

  els.mtabs.addEventListener("click", function (ev) {
    var b = ev.target.closest ? ev.target.closest(".mtab") : null;
    if (b) setMobilePane(b.dataset.pane);
  });

  // ----------------------------------------------------------- math pane --

  function renderMath() {
    var items = (lecture.math || []).slice().sort(function (a, b) {
      return a.t_start - b.t_start;
    });
    lecture.math = items;

    mathCards = new Array(items.length);
    mathStarts = items.map(function (m) { return m.t_start; });
    els.math.innerHTML = "";

    if (!items.length) {
      var none = document.createElement("div");
      none.className = "pane-empty";
      none.textContent = "No formulas extracted for this lecture yet.";
      els.math.appendChild(none);
      return;
    }

    var flagged = 0;
    var frag = document.createDocumentFragment();

    items.forEach(function (m, i) {
      var card = document.createElement("div");
      card.className = "card";
      card.dataset.i = i;
      var isFlagged = m.ambiguous || (m.confidence !== undefined && m.confidence < 0.6);
      if (isFlagged) { flagged++; card.dataset.flagged = "1"; }

      var head = document.createElement("div");
      head.className = "card-head";

      var time = document.createElement("span");
      time.className = "card-time";
      time.textContent = hhmmss(m.t_start);
      head.appendChild(time);

      var chip = document.createElement("span");
      chip.className = "chip " + (m.kind || "expression");
      chip.textContent = m.kind || "expression";
      head.appendChild(chip);

      if (m.confidence !== undefined) {
        var conf = document.createElement("span");
        conf.className = "card-time";
        conf.textContent = "conf " + Number(m.confidence).toFixed(2);
        head.appendChild(conf);
      }

      if (isFlagged) {
        var badge = document.createElement("span");
        badge.className = "badge";
        badge.textContent = m.ambiguous ? "ambiguous" : "low confidence";
        badge.title = m.note || "The model was unsure about this conversion.";
        head.appendChild(badge);
      }

      if (api.available) {
        var fix = document.createElement("button");
        fix.className = "fixbtn";
        fix.textContent = "fix";
        fix.title = "Correct this conversion and save it to course/glossary.json";
        fix.addEventListener("click", function (ev) {
          ev.stopPropagation();
          openFixer(card, m);
        });
        head.appendChild(fix);
      }
      card.appendChild(head);

      // Rendered LaTeX, falling back to the raw source when KaTeX cannot
      // parse it -- seeing the broken LaTeX is far more useful than an
      // error message or an empty card.
      var tex = document.createElement("div");
      tex.className = "card-latex";
      try {
        if (!window.katex) throw new Error("KaTeX not loaded");
        katex.render(m.latex, tex, { throwOnError: true, displayMode: false });
      } catch (err) {
        tex.className = "card-latex raw";
        tex.textContent = m.latex;
        tex.title = "KaTeX could not render this: " + (err && err.message ? err.message : err);
      }
      card.appendChild(tex);

      if (m.source_text) {
        var said = document.createElement("div");
        said.className = "card-said";
        said.textContent = m.source_text;
        card.appendChild(said);
      }

      if (m.note) {
        var note = document.createElement("div");
        note.className = "card-note";
        note.textContent = m.note;
        card.appendChild(note);
      }

      mathCards[i] = card;
      frag.appendChild(card);
    });

    els.math.appendChild(frag);
    applyMathFilter();
  }

  function applyMathFilter() {
    var only = els.onlyFlagged && els.onlyFlagged.checked;
    mathCards.forEach(function (card) {
      if (!card) return;
      card.classList.toggle("hidden", !!only && card.dataset.flagged !== "1");
    });
  }

  if (els.onlyFlagged) els.onlyFlagged.addEventListener("change", applyMathFilter);

  els.math.addEventListener("click", function (ev) {
    var card = ev.target.closest ? ev.target.closest(".card") : null;
    if (!card) return;
    var m = lecture.math[+card.dataset.i];
    if (m) seek(m.t_start);
  });

  // Latest item whose span has started; -1 when the playhead is past its end.
  function findMath(t) {
    var lo = 0, hi = mathStarts.length - 1, best = -1;
    while (lo <= hi) {
      var mid = (lo + hi) >> 1;
      if (mathStarts[mid] <= t) { best = mid; lo = mid + 1; } else { hi = mid - 1; }
    }
    if (best < 0) return -1;
    var item = lecture.math[best];
    return (t <= item.t_end + 0.5) ? best : -1;
  }

  // ------------------------------------------------------------ scrolling --

  function scrollToSpan(span) {
    if (!span) return;
    var box = els.transcript.getBoundingClientRect();
    var r = span.getBoundingClientRect();
    if (r.top >= box.top + 60 && r.bottom <= box.bottom - 120) return; // visible

    var target = els.transcript.scrollTop + (r.top - box.top) - box.height * 0.35;

    // A deep link can jump thousands of words. Animating that is both slow and
    // dizzying, so only short follow-along moves get the smooth treatment.
    // "instant" is required rather than "auto" here, because the CSS sets
    // scroll-behavior: smooth and "auto" would defer to it.
    var far = Math.abs(target - els.transcript.scrollTop) > box.height * 2;
    els.transcript.scrollTo({ top: target, behavior: far ? "instant" : "smooth" });
  }

  // Pausing auto-scroll keys off actual user gestures, not scroll events.
  // A scroll event cannot tell you who caused it, and our own smooth scrolls
  // emit ~100 of them over more than a second -- any timeout guard around that
  // is a race. A wheel tick or a drag on the scrollbar is unambiguous.
  function userTookOver() {
    if (!autoScroll) return;
    autoScroll = false;
    els.jump.hidden = false;
  }

  els.transcript.addEventListener("wheel", userTookOver, { passive: true });
  els.transcript.addEventListener("touchmove", userTookOver, { passive: true });

  els.transcript.addEventListener("mousedown", function (ev) {
    // Clicks past the content box are on the scrollbar itself.
    if (ev.offsetX > els.transcript.clientWidth) userTookOver();
  });

  els.transcript.addEventListener("keydown", function (ev) {
    if (["PageUp", "PageDown", "Home", "End", "ArrowUp", "ArrowDown"].indexOf(ev.key) >= 0) {
      userTookOver();
    }
  });

  els.jump.addEventListener("click", function () {
    autoScroll = true;
    els.jump.hidden = true;
    if (curWord >= 0) scrollToSpan(wordSpans[curWord]);
  });

  // --------------------------------------------------------------- tick ---

  function tick() {
    if (lecture && wordStarts.length) {
      var t = els.audio.currentTime;
      var i = findWord(t);

      if (i !== curWord) {
        if (curWord >= 0 && wordSpans[curWord]) wordSpans[curWord].classList.remove("current");
        if (i >= 0 && wordSpans[i]) wordSpans[i].classList.add("current");
        curWord = i;

        var si = segOfWord(i);
        if (si !== curSeg) {
          if (curSeg >= 0 && segSpans[curSeg]) segSpans[curSeg].classList.remove("active");
          if (si >= 0 && segSpans[si]) segSpans[si].classList.add("active");
          curSeg = si;
        }
        if (autoScroll && i >= 0) scrollToSpan(wordSpans[i]);
      }

      if (mathStarts.length) {
        var mi = findMath(t);
        if (mi !== curMath) {
          if (curMath >= 0 && mathCards[curMath]) mathCards[curMath].classList.remove("current");
          if (mi >= 0 && mathCards[mi]) {
            mathCards[mi].classList.add("current");
            if (autoScroll && rightTab === "math") {
              mathCards[mi].scrollIntoView({ block: "nearest" });
            }
          }
          curMath = mi;
        }
      }

      if (stripSegs.length) {
        var dur = lecture.duration || 0;
        if (playhead && dur > 0) {
          playhead.style.left = (100 * Math.min(1, t / dur)) + "%";
        }
        var ci = findChapter(t);
        if (ci !== curChapter) {
          if (curChapter >= 0 && stripSegs[curChapter]) stripSegs[curChapter].classList.remove("current");
          if (ci >= 0 && stripSegs[ci]) stripSegs[ci].classList.add("current");
          curChapter = ci;
        }
      }

      if (noteStarts.length) {
        var ni = findNote(t);
        if (ni !== curNote) {
          if (curNote >= 0 && noteEls[curNote]) noteEls[curNote].classList.remove("current");
          if (ni >= 0 && noteEls[ni]) {
            noteEls[ni].classList.add("current");
            if (autoScroll && rightTab === "notes") {
              noteEls[ni].scrollIntoView({ block: "nearest" });
            }
          }
          curNote = ni;
        }
      }

      if (sections.length) {
        var si2 = findSection(t);
        if (si2 !== curSection) {
          if (curSection >= 0 && sectionEls[curSection]) {
            sectionEls[curSection].classList.remove("current");
          }
          if (si2 >= 0 && sectionEls[si2]) {
            sectionEls[si2].classList.add("current");
            if (autoScroll && rightTab === "course") {
              sectionEls[si2].scrollIntoView({ block: "nearest" });
            }
          }
          curSection = si2;
        }
      }
    }
    requestAnimationFrame(tick);
  }

  // -------------------------------------------------------- interaction ---

  function seek(t) {
    t = Math.max(0, t);
    if (isFinite(els.audio.duration) && els.audio.duration > 0) {
      t = Math.min(t, els.audio.duration - 0.05);
    }
    els.audio.currentTime = t;
  }

  els.transcript.addEventListener("click", function (ev) {
    var target = ev.target;
    if (target.classList.contains("seg-time")) {
      seek(parseFloat(target.dataset.t));
      autoScroll = true; els.jump.hidden = true;
      return;
    }
    if (target.classList.contains("w")) {
      var i = parseInt(target.dataset.i, 10);
      if (!isNaN(i) && lecture.words[i]) {
        seek(lecture.words[i].start);
        autoScroll = true; els.jump.hidden = true;
      }
    }
  });

  document.addEventListener("keydown", function (ev) {
    var tag = (ev.target && ev.target.tagName) || "";
    var typing = (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT");

    if (ev.key === "Escape" && !els.library.hidden) {
      ev.preventDefault();
      closeLibrary();
      return;
    }
    if (ev.key === "/" && !typing && api.available && els.library.hidden) {
      ev.preventDefault();
      openLibrary("search");
      return;
    }
    if (typing) return;
    if (ev.ctrlKey || ev.altKey || ev.metaKey) return;

    if (ev.code === "Space") {
      ev.preventDefault();
      if (els.audio.paused) { els.audio.play(); } else { els.audio.pause(); }
    } else if (ev.code === "ArrowLeft") {
      ev.preventDefault();
      seek(els.audio.currentTime - (ev.shiftKey ? 30 : 5));
    } else if (ev.code === "ArrowRight") {
      ev.preventDefault();
      seek(els.audio.currentTime + (ev.shiftKey ? 30 : 5));
    }
  });

  function flash(btn, msg) {
    var old = btn.textContent;
    btn.textContent = msg;
    setTimeout(function () { btn.textContent = old; }, 1400);
  }

  els.copyLink.addEventListener("click", function () {
    if (!lecture) return;
    var t = Math.floor(els.audio.currentTime || 0);
    var base = location.href.split("?")[0].split("#")[0];
    var url = base + "?course=" + encodeURIComponent(courseId)
      + "&lecture=" + encodeURIComponent(lecture.lecture_id) + "&t=" + t;
    if (navigator.clipboard && navigator.clipboard.writeText) {
      navigator.clipboard.writeText(url).then(
        function () { flash(els.copyLink, "Copied " + hhmmss(t)); },
        function () { window.prompt("Copy this deep link:", url); }
      );
    } else {
      window.prompt("Copy this deep link:", url);
    }
  });

  els.picker.addEventListener("change", function () {
    location.search = "?course=" + encodeURIComponent(courseId)
      + "&lecture=" + encodeURIComponent(els.picker.value);
  });

  // ------------------------------------------------------ chapter strip ---

  function renderStrip() {
    var chapters = (lecture.chapters || []).slice().sort(function (a, b) {
      return a.start - b.start;
    });
    lecture.chapters = chapters;
    stripSegs = [];
    els.strip.innerHTML = "";

    var total = lecture.duration || (chapters.length ? chapters[chapters.length - 1].end : 0);
    if (!chapters.length || !total) {
      els.strip.hidden = true;
      return;
    }
    els.strip.hidden = false;

    chapters.forEach(function (ch, i) {
      var seg = document.createElement("div");
      seg.className = "strip-seg " + (ch.type || "aside");
      // Proportional width, so the strip is a map of the lecture's shape.
      seg.style.flex = Math.max(0.001, (ch.end - ch.start) / total) + " 0 0";
      seg.dataset.i = i;
      seg.title = hhmmss(ch.start) + " - " + hhmmss(ch.end) + "  ("
                + (ch.type || "aside") + ")\n" + ch.title
                + (ch.summary ? "\n\n" + ch.summary : "");

      var label = document.createElement("span");
      label.className = "strip-label";
      label.textContent = ch.title;
      seg.appendChild(label);

      stripSegs.push(seg);
      els.strip.appendChild(seg);
    });

    playhead = document.createElement("div");
    playhead.className = "strip-playhead";
    els.strip.appendChild(playhead);
  }

  els.strip.addEventListener("click", function (ev) {
    var seg = ev.target.closest ? ev.target.closest(".strip-seg") : null;
    if (!seg) return;
    var ch = lecture.chapters[+seg.dataset.i];
    if (ch) seek(ch.start);
  });

  function findChapter(t) {
    var chapters = lecture.chapters || [];
    for (var i = chapters.length - 1; i >= 0; i--) {
      if (chapters[i].start <= t) return i;
    }
    return -1;
  }

  // ------------------------------------------------------------ backend ---

  var api = { available: false, info: null };

  function apiGet(path) {
    return fetch(path, { headers: { "Accept": "application/json" } })
      .then(function (r) {
        if (!r.ok) return r.json().then(function (e) { throw new Error(e.error || r.status); });
        return r.json();
      });
  }

  function apiPost(path, body) {
    return fetch(path, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body)
    }).then(function (r) {
      return r.json().then(function (j) {
        if (!r.ok) throw new Error(j.error || ("HTTP " + r.status));
        return j;
      });
    });
  }

  function detectBackend() {
    // Opened straight off disk there is no backend, and that is a supported
    // way to use this -- the reading features all work, the library does not.
    if (location.protocol === "file:") return Promise.resolve(false);
    return apiGet("/api/health").then(function (info) {
      api.available = true;
      api.info = info;
      els.openLibrary.hidden = false;
      return true;
    }).catch(function () { return false; });
  }

  // ------------------------------------------------------------ library ---

  var libTab = "search";
  var searchTimer = null;

  function openLibrary(tab) {
    els.library.hidden = false;
    setLibTab(tab || libTab);
    els.libQ.focus();
    els.libQ.select();
  }

  function closeLibrary() { els.library.hidden = true; }

  function setLibTab(tab) {
    // The one box means two things, so carrying text across the switch makes
    // the other tab look empty rather than filtered.
    if (tab !== libTab) els.libQ.value = "";
    libTab = tab;
    els.libTabSearch.classList.toggle("is-active", tab === "search");
    els.libTabIndex.classList.toggle("is-active", tab === "index");
    els.libQ.placeholder = tab === "search"
      ? "Search across every lecture…"
      : "Filter concepts…";
    if (tab === "index") { renderIndex(); } else { runSearch(); }
  }

  els.openLibrary.addEventListener("click", function () { openLibrary("search"); });
  els.libClose.addEventListener("click", closeLibrary);
  els.libTabSearch.addEventListener("click", function () { setLibTab("search"); });
  els.libTabIndex.addEventListener("click", function () { setLibTab("index"); });

  els.libQ.addEventListener("input", function () {
    clearTimeout(searchTimer);
    searchTimer = setTimeout(function () {
      if (libTab === "index") { renderIndex(); } else { runSearch(); }
    }, 220);
  });

  function libEmpty(msg) {
    els.libBody.innerHTML = "";
    var d = document.createElement("div");
    d.className = "pane-empty";
    d.textContent = msg;
    els.libBody.appendChild(d);
  }

  function goTo(lectureId, t) {
    if (lectureId === lecture.lecture_id) {
      seek(t);
      closeLibrary();
      autoScroll = true;
      els.jump.hidden = true;
    } else {
      location.search = "?course=" + encodeURIComponent(courseId)
        + "&lecture=" + encodeURIComponent(lectureId) + "&t=" + Math.floor(t);
    }
  }

  function runSearch() {
    var q = els.libQ.value.trim();
    if (!q) { return libEmpty("Type to search every lecture's chapters."); }
    els.libNote.textContent = "searching…";
    apiGet("/api/search?k=20&q=" + encodeURIComponent(q)).then(function (data) {
      var res = data.results || [];
      els.libNote.textContent = res.length + " result" + (res.length === 1 ? "" : "s");
      els.libBody.innerHTML = "";
      if (!res.length) { return libEmpty("Nothing matched “" + q + "”."); }
      var frag = document.createDocumentFragment();
      res.forEach(function (r) {
        var hit = document.createElement("div");
        hit.className = "hit";
        hit.addEventListener("click", function () { goTo(r.lecture_id, r.t_start); });

        var head = document.createElement("div");
        head.className = "hit-head";

        var t = document.createElement("span");
        t.className = "card-time";
        t.textContent = r.lecture_id + " · " + hhmmss(r.t_start);
        head.appendChild(t);

        var chip = document.createElement("span");
        chip.className = "chip " + (r.type || "aside");
        chip.textContent = r.type || "aside";
        head.appendChild(chip);

        var title = document.createElement("span");
        title.className = "hit-title";
        title.textContent = r.title;
        head.appendChild(title);

        // Say *why* something matched. A semantic-only hit on a precise
        // mathematical name usually means the term is not in the corpus.
        var why = document.createElement("span");
        why.className = "hit-whys";
        (r.matched || []).forEach(function (m) {
          var w = document.createElement("span");
          w.className = "hit-why " + m;
          w.textContent = m;
          why.appendChild(w);
        });
        head.appendChild(why);
        hit.appendChild(head);

        var sum = document.createElement("div");
        sum.className = "hit-sum";
        sum.textContent = r.summary || "";
        hit.appendChild(sum);
        frag.appendChild(hit);
      });
      els.libBody.appendChild(frag);
    }).catch(function (err) {
      els.libNote.textContent = "";
      libEmpty("Search failed: " + err.message);
    });
  }

  var conceptCache = null;

  function renderIndex() {
    var filter = els.libQ.value.trim().toLowerCase();
    var draw = function (data) {
      var list = (data.concepts || []).filter(function (c) {
        return !filter || c.concept.indexOf(filter) >= 0;
      });
      els.libNote.textContent = list.length + " concept" + (list.length === 1 ? "" : "s");
      els.libBody.innerHTML = "";
      if (!list.length) { return libEmpty("No concepts match."); }

      var frag = document.createDocumentFragment();
      var letter = null;
      list.forEach(function (c) {
        var initial = (c.concept[0] || "?").toUpperCase();
        if (initial !== letter) {
          letter = initial;
          var h = document.createElement("div");
          h.className = "idx-letter";
          h.textContent = letter;
          frag.appendChild(h);
        }

        var row = document.createElement("div");
        row.className = "idx-row";

        var name = document.createElement("span");
        name.className = "idx-name";
        name.textContent = c.concept;
        row.appendChild(name);

        var links = document.createElement("span");
        links.className = "idx-links";
        (c.occurrences || []).forEach(function (o) {
          var a = document.createElement("span");
          var isCanon = c.canonical && c.canonical.lecture_id === o.lecture_id
                        && Math.abs(c.canonical.t - o.t) < 0.01;
          a.className = "idx-link" + (isCanon ? " canonical" : "");
          a.textContent = o.lecture_id + " " + hhmmss(o.t);
          a.title = (isCanon ? "Where it is introduced\n" : "") + (o.title || "");
          a.addEventListener("click", function (ev) {
            ev.stopPropagation();
            goTo(o.lecture_id, o.t);
          });
          links.appendChild(a);
        });
        row.appendChild(links);

        if ((c.assumed_by || []).length) {
          var as = document.createElement("span");
          as.className = "idx-assumed";
          as.textContent = "assumed by " + c.assumed_by.length;
          as.title = c.assumed_by.map(function (a) {
            return a.lecture_id + " " + hhmmss(a.t) + "  " + (a.title || "");
          }).join("\n");
          row.appendChild(as);
        }
        frag.appendChild(row);
      });
      els.libBody.appendChild(frag);
    };

    if (conceptCache) return draw(conceptCache);
    els.libNote.textContent = "loading…";
    apiGet("/api/concepts?course=" + encodeURIComponent(courseId)).then(function (d) {
      conceptCache = d;
      draw(d);
    }).catch(function (err) {
      els.libNote.textContent = "";
      libEmpty("Concept index unavailable: " + err.message);
    });
  }

  // ------------------------------------------------- correction editor ----

  function openFixer(card, item) {
    if (card.querySelector(".fixform")) return;
    var form = document.createElement("div");
    form.className = "fixform";
    form.addEventListener("click", function (ev) { ev.stopPropagation(); });

    var lab = document.createElement("label");
    lab.textContent = "Corrected LaTeX";
    form.appendChild(lab);

    var ta = document.createElement("textarea");
    ta.value = item.latex || "";
    ta.spellcheck = false;
    form.appendChild(ta);

    var preview = document.createElement("div");
    preview.className = "fixpreview";
    form.appendChild(preview);

    var row = document.createElement("div");
    row.className = "fixrow";
    var save = document.createElement("button");
    save.className = "btn";
    save.textContent = "Save to glossary";
    var cancel = document.createElement("button");
    cancel.className = "btn";
    cancel.textContent = "Cancel";
    var msg = document.createElement("span");
    msg.className = "fixmsg";
    row.appendChild(save);
    row.appendChild(cancel);
    row.appendChild(msg);
    form.appendChild(row);

    function repaint() {
      preview.innerHTML = "";
      var v = ta.value.trim();
      if (!v) return;
      renderTex(v, preview, false);
    }
    ta.addEventListener("input", repaint);
    repaint();

    cancel.addEventListener("click", function () { form.remove(); });

    save.addEventListener("click", function () {
      var right = ta.value.trim();
      if (!right) { msg.className = "fixmsg err"; msg.textContent = "Enter the correct LaTeX."; return; }
      save.disabled = true;
      msg.className = "fixmsg";
      msg.textContent = "saving…";
      apiPost("/api/corrections", {
        course_id: courseId,
        lecture_id: lecture.lecture_id,
        item_id: item.id,
        kind: "latex",
        wrong: item.latex || "",
        right: right,
        source_text: item.source_text || ""
      }).then(function () {
        msg.className = "fixmsg ok";
        msg.textContent = "saved — it will feed into later lectures";
        item.latex = right;
        card.classList.add("corrected");
        setTimeout(function () { form.remove(); renderMath(); applyMathFilter(); }, 900);
      }).catch(function (err) {
        save.disabled = false;
        msg.className = "fixmsg err";
        msg.textContent = "failed: " + err.message;
      });
    });

    card.appendChild(form);
    ta.focus();
  }

  // --------------------------------------------------------------- boot ---

  function mountLecture(seekTo) {
    lecture = window.__LECTURE__;
    render();
    renderMath();
    renderNotes();
    renderCourseNotes();
    renderStrip();
    // Default to whichever pane actually has something in it, unless the link
    // that got us here asked for one -- a search hit on the whole-lecture
    // write-up carries ?pane=course, and landing on the rough notes instead
    // would drop the reader somewhere they cannot see what they matched.
    var start = (lecture.notes && lecture.notes.length) || !(lecture.math || []).length
                ? "notes" : "math";
    var wantPane = qs("pane");
    if (wantPane === "course" && sections.length) start = "course";
    else if (wantPane === "notes" || wantPane === "math") start = wantPane;
    setTab(start);
    setMobilePane(start);

    // audio_src is the seek-accurate derived file (Ogg Opus). The original
    // lecture MP3 is only a fallback: browsers seeking a long VBR MP3 land
    // seconds from the time they report, which desyncs everything after a
    // click. See README, "Why the viewer does not play your MP3".
    if (lecture.audio_src) {
      els.audio.src = lecture.audio_src.split("/").map(encodeURIComponent).join("/");
    } else if (lecture.audio_file) {
      els.audio.src = "../audio/" + encodeURIComponent(lecture.audio_file);
      setStatus("Playing the original file directly -- seeking will be imprecise. "
                + "Re-run Stage 1 to generate the seek-accurate audio.", false);
    } else {
      setStatus("This lecture has no audio recorded; playback is unavailable.", true);
    }

    els.meta.textContent =
      courseId + " · " + lecture.lecture_id + "  " + hhmmss(lecture.duration) +
      (lecture.stages && lecture.stages.math ? "" : "  (transcript only)");

    document.title = lecture.lecture_id + " -- Lecture viewer";

    if (seekTo !== null && seekTo > 0) {
      var go = function () { seek(seekTo); };
      if (els.audio.readyState >= 1) { go(); }
      else { els.audio.addEventListener("loadedmetadata", go, { once: true }); }
    }
    requestAnimationFrame(tick);
  }

  function fillPicker(selected) {
    var list = (window.__LECTURES__ && window.__LECTURES__.lectures) || [];
    els.picker.innerHTML = "";
    if (!list.length) {
      var none = document.createElement("option");
      none.textContent = "(no lectures yet)";
      els.picker.appendChild(none);
      els.picker.disabled = true;
      return list;
    }
    list.forEach(function (rec) {
      var opt = document.createElement("option");
      opt.value = rec.lecture_id;
      opt.textContent = rec.lecture_id + "  (" + hhmmss(rec.duration) + ")";
      if (rec.lecture_id === selected) opt.selected = true;
      els.picker.appendChild(opt);
    });
    return list;
  }

  function boot() {
    courseId = safeId(qs("course"));
    if (!courseId) {
      setStatus("No course given. Open a course from the home page.", true);
      return;
    }
    var wanted = safeId(qs("lecture"));
    var t = parseFloat(qs("t"));
    if (isNaN(t)) t = null;

    loadScript("../data/" + encodeURIComponent(courseId) + "/lectures.js", function () {
      var list = fillPicker(wanted);

      var id = wanted;
      if (!id && list.length) id = list[0].lecture_id;   // default to the first
      if (!id) {
        setStatus("No lectures found. Run Stage 1 first, then reload.", true);
        return;
      }
      fillPicker(id);

      loadScript("../data/" + encodeURIComponent(courseId) + "/"
                 + encodeURIComponent(id) + "/lecture.js",
        function () {
          if (!window.__LECTURE__) {
            setStatus("data/" + courseId + "/" + id + "/lecture.js was empty.", true);
            return;
          }
          setStatus(null);
          detectBackend().then(function () {
            mountLecture(t);
            // ?q=... opens the library on that search, so a search itself is
            // shareable, the same way a timestamp is.
            var wantQ = qs("q");
            if (wantQ && api.available) {
              els.libQ.value = wantQ;
              openLibrary("search");
              runSearch();
            }
          });
        },
        function () {
          setStatus("Could not load this lecture. Has it been processed yet?", true);
        });
    }, function () {
      setStatus("No lectures for course " + courseId + " yet. Upload one from the home page.", true);
    });
  }

  boot();
})();
