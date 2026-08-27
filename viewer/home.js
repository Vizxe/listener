/* Home page: courses, upload, processing options, notation editing.
 *
 * Talks to scripts/server.py. Everything here needs the backend, so unlike
 * the viewer there is no offline mode to fall back to.
 */
"use strict";

(function () {

  var $ = function (id) { return document.getElementById(id); };
  var course = null;          // currently open course id
  var models = null;
  var pollTimer = null;

  // The key rides in the URL on first visit; the server also sets a cookie,
  // but keeping it on links means a shared URL still works.
  function keyParam() {
    var m = /[?&]key=([^&#]*)/.exec(location.search);
    return m ? "key=" + m[1] : "";
  }

  function url(path) {
    var k = keyParam();
    if (!k) return path;
    return path + (path.indexOf("?") >= 0 ? "&" : "?") + k;
  }

  function api(path, opts) {
    return fetch(url(path), opts).then(function (r) {
      return r.json().catch(function () { return {}; }).then(function (j) {
        if (!r.ok) throw new Error(j.error || ("HTTP " + r.status));
        return j;
      });
    });
  }

  function say(el, text, cls) {
    el.textContent = text;
    el.className = "msg" + (cls ? " " + cls : "");
  }

  function hhmmss(t) {
    t = Math.max(0, Math.floor(t || 0));
    var h = Math.floor(t / 3600), m = Math.floor((t % 3600) / 60), s = t % 60;
    return (h > 0 ? h + ":" + (m < 10 ? "0" : "") : "") + m + ":" + (s < 10 ? "0" : "") + s;
  }

  // ------------------------------------------------------------ courses ---

  function showCourses() {
    course = null;
    $("view-course").hidden = true;
    $("view-courses").hidden = false;
    $("crumb").textContent = "";
    history.replaceState(null, "", url(location.pathname));
    api("/api/courses").then(function (d) {
      var grid = $("course-grid");
      grid.innerHTML = "";
      if (!d.courses.length) {
        var e = document.createElement("div");
        e.className = "empty";
        e.textContent = "No courses yet. Create one to get started.";
        grid.appendChild(e);
        return;
      }
      d.courses.forEach(function (c) {
        var card = document.createElement("div");
        card.className = "course-card";
        card.addEventListener("click", function () { openCourse(c.id); });

        var h = document.createElement("h3");
        h.textContent = c.id;
        card.appendChild(h);

        var sub = document.createElement("div");
        sub.className = "sub";
        sub.textContent = c.title && c.title !== c.id ? c.title : " ";
        card.appendChild(sub);

        var stats = document.createElement("div");
        stats.className = "stats";
        stats.appendChild(spanText(c.lectures + " lecture" + (c.lectures === 1 ? "" : "s")));
        stats.appendChild(spanText(c.concepts + " concepts"));
        if (c.unprocessed && c.unprocessed.length) {
          var p = document.createElement("span");
          p.className = "pill";
          p.textContent = c.unprocessed.length + " unprocessed";
          stats.appendChild(p);
        }
        card.appendChild(stats);
        grid.appendChild(card);
      });
    }).catch(function (e) {
      $("course-grid").innerHTML = "";
      var d = document.createElement("div");
      d.className = "empty";
      d.textContent = "Could not reach the backend: " + e.message;
      $("course-grid").appendChild(d);
    });
  }

  function spanText(t) {
    var s = document.createElement("span");
    s.textContent = t;
    return s;
  }

  function openCourse(id) {
    course = id;
    $("view-courses").hidden = true;
    $("view-course").hidden = false;
    $("course-title").textContent = id;
    $("crumb").textContent = id;
    $("open-viewer").href = url("viewer.html?course=" + encodeURIComponent(id));
    history.replaceState(null, "", url(location.pathname + "?course=" + encodeURIComponent(id)));
    loadModels();
    refreshLectures();
    loadNotation();
    poll();
  }

  $("back").addEventListener("click", showCourses);

  // ------------------------------------------------------------- options ---

  function loadModels() {
    if (models) return renderOptions();
    api("/api/models").then(function (d) { models = d; renderOptions(); });
  }

  function renderOptions() {
    var sel = $("opt-model");
    if (!sel.options.length) {
      models.choices.forEach(function (c) {
        var o = document.createElement("option");
        o.value = c.id;
        o.textContent = c.label;
        if (c.id === models.current) o.selected = true;
        sel.appendChild(o);
      });
    }
    var box = $("opt-stages");
    if (!box.children.length) {
      models.stages.forEach(function (s) {
        var lab = document.createElement("label");
        var cb = document.createElement("input");
        cb.type = "checkbox";
        cb.value = s.id;
        cb.checked = models.default_stages.indexOf(s.id) >= 0;
        lab.appendChild(cb);
        lab.appendChild(document.createTextNode(s.label));
        box.appendChild(lab);
      });
    }
  }

  function chosenStages() {
    return [].slice.call($("opt-stages").querySelectorAll("input:checked"))
             .map(function (c) { return c.value; });
  }

  function chosenOptions() {
    return { model: $("opt-model").value, force: $("opt-force").checked };
  }

  // ------------------------------------------------------------ lectures ---

  function refreshLectures() {
    if (!course) return;
    api("/api/lectures?course=" + encodeURIComponent(course)).then(function (d) {
      var list = $("lecture-list");
      list.innerHTML = "";

      d.unprocessed.forEach(function (u) {
        list.appendChild(lectureItem({
          lecture_id: u.lecture_id, pending: true,
          note: u.file + " · " + (u.size / 1e6).toFixed(0) + " MB, not processed yet"
        }));
      });

      d.lectures.forEach(function (l) {
        list.appendChild(lectureItem({
          lecture_id: l.lecture_id,
          note: hhmmss(l.duration || 0),
          stages: l.stages || {}
        }));
      });

      if (!list.children.length) {
        var e = document.createElement("div");
        e.className = "empty";
        e.textContent = "No lectures yet. Upload one on the left.";
        list.appendChild(e);
      }
    });
  }

  function lectureItem(o) {
    var it = document.createElement("div");
    it.className = "item";

    var head = document.createElement("div");
    head.className = "item-head";

    var t = document.createElement("span");
    t.className = "item-title";
    t.textContent = o.lecture_id;
    head.appendChild(t);

    var n = document.createElement("span");
    n.className = "meta";
    n.textContent = o.note || "";
    head.appendChild(n);

    var actions = document.createElement("span");
    actions.className = "item-actions";

    var run = document.createElement("button");
    run.className = "btn";
    run.textContent = o.pending ? "Process" : "Re-run";
    run.addEventListener("click", function () { startJob(o.lecture_id); });
    actions.appendChild(run);

    if (!o.pending) {
      var open = document.createElement("a");
      open.className = "btn";
      open.textContent = "Open";
      open.href = url("viewer.html?course=" + encodeURIComponent(course)
                      + "&lecture=" + encodeURIComponent(o.lecture_id));
      actions.appendChild(open);
    }
    head.appendChild(actions);
    it.appendChild(head);

    if (o.stages) {
      var dots = document.createElement("div");
      dots.className = "stagedots";
      [["transcript", "transcript"], ["notes", "notes"],
       ["course_notes", "course notes"], ["math", "formulas"],
       ["structure", "chapters"]].forEach(function (p) {
        var d = document.createElement("span");
        d.className = "dot" + (o.stages[p[0]] ? " on" : "");
        d.textContent = p[1];
        dots.appendChild(d);
      });
      it.appendChild(dots);
    }
    return it;
  }

  // ---------------------------------------------------------------- jobs ---

  function startJob(lectureId) {
    var stages = chosenStages();
    if (!stages.length) {
      say($("up-msg"), "Pick at least one step to run.", "err");
      return;
    }
    api("/api/process", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        course: course, lecture: lectureId || null,
        stages: stages, options: chosenOptions()
      })
    }).then(function () {
      say($("up-msg"), "Started. Progress appears under Recent jobs.", "ok");
      poll();
    }).catch(function (e) { say($("up-msg"), e.message, "err"); });
  }

  // Job elements are reused between polls. Rebuilding them wholesale is what
  // made the log jump: replacing the node resets scrollTop to 0, so anyone
  // scrolled down to read output got yanked back to the top every 1.5s.
  var jobEls = {};

  function poll() {
    clearTimeout(pollTimer);
    api("/api/jobs").then(function (d) {
      var list = $("job-list");
      var jobs = (d.jobs || []).filter(function (j) { return j.course_id === course; })
                               .slice(0, 5);

      var empty = list.querySelector(".empty");
      if (!jobs.length) {
        if (!empty) {
          list.innerHTML = "";
          jobEls = {};
          var e = document.createElement("div");
          e.className = "empty";
          e.textContent = "Nothing has been processed yet in this course.";
          list.appendChild(e);
        }
      } else if (empty) {
        empty.remove();
      }

      var seen = {};
      var active = false;
      jobs.forEach(function (j, i) {
        seen[j.id] = true;
        if (j.status === "running" || j.status === "queued") active = true;
        var el = jobEls[j.id];
        if (!el) {
          el = buildJob(j);
          jobEls[j.id] = el;
        }
        updateJob(el, j);
        // Keep DOM order matching the API order without touching the nodes
        // themselves, so nothing inside them is re-created.
        var at = list.children[i];
        if (at !== el.root) list.insertBefore(el.root, at || null);
      });

      Object.keys(jobEls).forEach(function (id) {
        if (!seen[id]) {
          jobEls[id].root.remove();
          delete jobEls[id];
        }
      });

      $("busy").hidden = !active;
      pollTimer = setTimeout(poll, active ? 1500 : 8000);
      if (!active) refreshLecturesDebounced();
    }).catch(function () {
      pollTimer = setTimeout(poll, 8000);
    });
  }

  function buildJob(j) {
    var root = document.createElement("div");
    root.className = "item job";

    var head = document.createElement("div");
    head.className = "item-head";

    var title = document.createElement("span");
    title.className = "item-title";
    title.textContent = j.lecture_id || "all lectures";
    head.appendChild(title);

    var status = document.createElement("span");
    head.appendChild(status);

    var pct = document.createElement("span");
    pct.className = "meta";
    head.appendChild(pct);

    var actions = document.createElement("span");
    actions.className = "item-actions";
    var cancel = document.createElement("button");
    cancel.className = "btn";
    cancel.textContent = "Cancel";
    cancel.addEventListener("click", function () {
      api("/api/jobs/cancel", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ id: j.id })
      }).then(poll);
    });
    actions.appendChild(cancel);
    head.appendChild(actions);
    root.appendChild(head);

    var bar = document.createElement("div");
    bar.className = "progress";
    var fill = document.createElement("div");
    fill.className = "bar-fill";
    bar.appendChild(fill);
    root.appendChild(bar);

    var log = document.createElement("div");
    log.className = "log";
    // Console behaviour: follow the newest output, but stop following the
    // moment the reader scrolls up, and resume when they return to the bottom.
    log.dataset.stick = "1";
    log.addEventListener("scroll", function () {
      if (log.__busy) return;          // our own update, not the reader
      var atBottom = log.scrollHeight - log.scrollTop - log.clientHeight < 28;
      log.dataset.stick = atBottom ? "1" : "0";
    });
    root.appendChild(log);

    return { root: root, status: status, pct: pct, fill: fill, log: log,
             cancel: cancel, title: title };
  }

  function updateJob(el, j) {
    el.title.textContent = j.lecture_id || "all lectures";

    var label = j.status + (j.current ? " \u00b7 " + j.current : "");
    if (el.status.textContent !== label) {
      el.status.textContent = label;
      el.status.className = "status-tag " + j.status;
    }

    var pct = Math.round(j.progress * 100) + "%";
    if (el.pct.textContent !== pct) el.pct.textContent = pct;
    el.fill.style.width = (j.progress * 100) + "%";

    var running = (j.status === "running" || j.status === "queued");
    el.cancel.hidden = !running;

    // Log output only ever grows, so append the new tail rather than
    // replacing the content. Replacing it destroys the text node, which
    // resets scrollTop to 0 AND fires a scroll event -- and that event,
    // arriving while the box is momentarily at the top, flips the follow
    // flag off. Appending leaves the scroll position alone.
    var text = (j.log || []).join(String.fromCharCode(10));
    var prev = el.log.__text || "";
    if (text !== prev) {
      el.log.__busy = true;
      if (prev && text.indexOf(prev) === 0) {
        el.log.appendChild(document.createTextNode(text.slice(prev.length)));
      } else {
        el.log.textContent = text;
      }
      el.log.__text = text;
      if (el.log.dataset.stick !== "0") el.log.scrollTop = el.log.scrollHeight;
      // Clear on a timer, not requestAnimationFrame: rAF never fires while
      // the tab is in the background, which would leave this stuck on and
      // silently disable the reader's scroll handling for the whole session.
      setTimeout(function () { el.log.__busy = false; }, 0);
    }
  }

  var lastRefresh = 0;
  function refreshLecturesDebounced() {
    var now = Date.now();
    if (now - lastRefresh > 6000) { lastRefresh = now; refreshLectures(); }
  }

  // -------------------------------------------------------------- upload ---

  var drop = $("drop");
  $("pick").addEventListener("click", function () { $("file").click(); });
  $("file").addEventListener("change", function () {
    if (this.files && this.files[0]) upload(this.files[0]);
  });

  ["dragenter", "dragover"].forEach(function (ev) {
    drop.addEventListener(ev, function (e) {
      e.preventDefault(); e.stopPropagation();
      drop.classList.add("over");
    });
  });
  ["dragleave", "drop"].forEach(function (ev) {
    drop.addEventListener(ev, function (e) {
      e.preventDefault(); e.stopPropagation();
      drop.classList.remove("over");
    });
  });
  drop.addEventListener("drop", function (e) {
    var f = e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0];
    if (f) upload(f);
  });

  function upload(file) {
    if (!course) return;
    var msg = $("up-msg"), bar = $("up-bar");
    $("up-progress").hidden = false;
    bar.style.width = "0%";
    say(msg, "Uploading " + file.name + "…");

    // XHR rather than fetch: a lecture can be a gigabyte and upload progress
    // is the whole point of showing a bar.
    var form = new FormData();
    form.append("file", file, file.name);
    var xhr = new XMLHttpRequest();
    xhr.open("POST", url("/api/upload?course=" + encodeURIComponent(course)));
    xhr.upload.onprogress = function (e) {
      if (e.lengthComputable) bar.style.width = (100 * e.loaded / e.total) + "%";
    };
    xhr.onload = function () {
      var d = {};
      try { d = JSON.parse(xhr.responseText); } catch (err) {}
      if (xhr.status >= 200 && xhr.status < 300) {
        bar.style.width = "100%";
        say(msg, "Uploaded " + d.file + ". Starting processing…", "ok");
        refreshLectures();
        startJob(d.lecture_id);
      } else {
        say(msg, "Upload failed: " + (d.error || xhr.status), "err");
      }
      setTimeout(function () { $("up-progress").hidden = true; }, 1200);
    };
    xhr.onerror = function () {
      say(msg, "Upload failed: the connection dropped.", "err");
      $("up-progress").hidden = true;
    };
    xhr.send(form);
  }

  // ------------------------------------------------------------ notation ---

  function loadNotation() {
    api("/api/notation?course=" + encodeURIComponent(course)).then(function (d) {
      $("notation").value = d.text || "";
      say($("notation-msg"), "");
    });
  }

  $("save-notation").addEventListener("click", function () {
    api("/api/notation", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ course: course, text: $("notation").value })
    }).then(function () {
      say($("notation-msg"), "Saved. It applies to the next run.", "ok");
    }).catch(function (e) { say($("notation-msg"), e.message, "err"); });
  });

  // -------------------------------------------------------- new course ----

  $("new-course").addEventListener("click", function () {
    $("modal").hidden = false;
    $("nc-id").value = "";
    $("nc-title").value = "";
    say($("nc-msg"), "");
    $("nc-id").focus();
  });
  $("nc-cancel").addEventListener("click", function () { $("modal").hidden = true; });
  $("modal").addEventListener("click", function (e) {
    if (e.target === $("modal")) $("modal").hidden = true;
  });

  $("nc-create").addEventListener("click", function () {
    var id = $("nc-id").value.trim();
    if (!id) { say($("nc-msg"), "A course code is required.", "err"); return; }
    api("/api/courses", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ id: id, title: $("nc-title").value.trim() })
    }).then(function (d) {
      $("modal").hidden = true;
      openCourse(d.course.id);
    }).catch(function (e) { say($("nc-msg"), e.message, "err"); });
  });

  document.addEventListener("keydown", function (e) {
    if (e.key === "Escape" && !$("modal").hidden) $("modal").hidden = true;
  });

  // ---------------------------------------------------------------- boot ---

  var wanted = /[?&]course=([^&#]*)/.exec(location.search);
  if (wanted) { openCourse(decodeURIComponent(wanted[1])); } else { showCourses(); }
})();
