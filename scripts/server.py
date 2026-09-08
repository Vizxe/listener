"""The site: home page, viewer, upload, processing and the library API.

    python scripts/server.py                 # host/port from config.yaml
    python scripts/server.py --host 127.0.0.1 --no-auth

Binding
-------
`server.host` defaults to 0.0.0.0 so the site is reachable from a phone or
laptop over Tailscale. That also binds every other interface the machine has,
and this server accepts uploads and starts jobs -- so when the bind address is
not loopback an **access key** is required by default. The key is printed as a
ready-to-open URL at startup; visiting it once sets a cookie and the device
stays signed in.

Set `server.auth: off` to disable that. Only do it on 127.0.0.1.
"""
from __future__ import annotations

import argparse
import io
import json
import mimetypes
import os
import re
import secrets
import shutil
import sys
import threading
import urllib.parse
import webbrowser
from datetime import datetime, timezone
from http import cookies as http_cookies
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
import jobs as jobs_mod  # noqa: E402
import library  # noqa: E402
import merge  # noqa: E402

MAX_JSON_BODY = 512 * 1024
SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
COOKIE = "lecture_key"

mimetypes.add_type("application/javascript", ".js")
mimetypes.add_type("audio/ogg", ".opus")
mimetypes.add_type("font/woff2", ".woff2")


def is_loopback(host: str) -> bool:
    return host in ("127.0.0.1", "::1", "localhost")


def local_addresses():
    """Addresses this machine answers on, so the printed URL is one you can
    actually type into a phone. Tailscale hands out 100.x.y.z."""
    import socket
    found = []
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            ip = info[4][0]
            if ip not in found and not ip.startswith("127."):
                found.append(ip)
    except OSError:
        pass
    tailscale = [ip for ip in found if ip.startswith("100.")]
    return tailscale + [ip for ip in found if ip not in tailscale]


class State:
    def __init__(self, cfg, logger, key):
        self.cfg = cfg
        self.logger = logger
        self.key = key
        self.lock = threading.Lock()
        self._searchers = {}
        self.runner = jobs_mod.Runner(cfg, logger)
        self.runner.on_finish = self._job_finished

    def searcher(self, course_id, rebuild=False):
        with self.lock:
            if rebuild or course_id not in self._searchers:
                self._searchers[course_id] = library.Searcher(
                    self.cfg, course_id, self.logger)
            return self._searchers[course_id]

    def invalidate(self, course_id=None):
        with self.lock:
            if course_id:
                self._searchers.pop(course_id, None)
            else:
                self._searchers.clear()

    def _job_finished(self, job):
        # A finished job usually rewrote the index the searcher is holding.
        self.invalidate(job.course_id)


class Handler(BaseHTTPRequestHandler):
    server_version = "lecture-site"
    state: State = None
    root: Path = None

    # ------------------------------------------------------------ plumbing --

    def log_message(self, fmt, *args):
        if self.state and self.state.logger:
            self.state.logger.debug("%s %s", self.address_string(), fmt % args)

    def _send_json(self, obj, code=200, extra_headers=None):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for k, v in (extra_headers or {}):
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def _error(self, code, message):
        self._send_json({"error": message}, code)

    def _query(self):
        return urllib.parse.parse_qs(self.path.split("?", 1)[1] if "?" in self.path else "")

    def _param(self, qs, name, default=None):
        v = (qs.get(name) or [default])[0]
        return v.strip() if isinstance(v, str) else v

    # ------------------------------------------------------------- access ---

    def _authorised(self, qs):
        if not self.state.key:
            return True
        if self._param(qs, "key") == self.state.key:
            return True
        raw = self.headers.get("Cookie")
        if raw:
            c = http_cookies.SimpleCookie()
            try:
                c.load(raw)
            except http_cookies.CookieError:
                return False
            if COOKIE in c and c[COOKIE].value == self.state.key:
                return True
        return self.headers.get("X-Access-Key") == self.state.key

    def _deny(self):
        body = (b"<!doctype html><meta charset=utf-8>"
                b"<style>body{font:16px system-ui;background:#14161a;color:#dfe3ea;"
                b"padding:40px;line-height:1.6}code{background:#1b1e24;padding:2px 6px;"
                b"border-radius:4px}</style>"
                b"<h2>Access key required</h2><p>This server is reachable from the "
                b"network, so it needs the key printed in its console.</p>"
                b"<p>Open the URL it printed, of the form "
                b"<code>http://&lt;host&gt;:&lt;port&gt;/?key=...</code> &mdash; "
                b"once is enough, it is remembered on this device.</p>")
        self.send_response(401)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    # --------------------------------------------------------------- GET ----

    def do_GET(self):
        qs = self._query()
        if not self._authorised(qs):
            return self._deny()

        # Arriving with ?key=... sets the cookie so the phone stays signed in.
        setcookie = None
        if self.state.key and self._param(qs, "key") == self.state.key:
            setcookie = [("Set-Cookie",
                          f"{COOKIE}={self.state.key}; Path=/; Max-Age=31536000; SameSite=Lax")]

        route = self.path.split("?", 1)[0]
        if route.startswith("/api/"):
            return self._api_get(route, qs, setcookie)

        # Serve the home page from /viewer/ rather than /, so that every
        # relative link and stylesheet in it resolves against the directory
        # the files actually live in.
        if route in ("", "/"):
            query = self.path.split("?", 1)[1] if "?" in self.path else ""
            dest = "/viewer/index.html" + (("?" + query) if query else "")
            self.send_response(302)
            self.send_header("Location", dest)
            for k, v in (setcookie or []):
                self.send_header(k, v)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return

        return self._serve_file(setcookie)

    def _api_get(self, route, qs, setcookie=None):
        cfg = self.state.cfg
        course = self._param(qs, "course")

        if route == "/api/health":
            return self._send_json({
                "ok": True, "backend": True,
                "courses": len(common.load_courses(cfg)),
                "llm_model": cfg["llm"]["model"],
                "busy": self.state.runner.busy(),
            }, extra_headers=setcookie)

        if route == "/api/courses":
            out = []
            for c in common.load_courses(cfg):
                cid = c["id"]
                lectures = common.lecture_ids(cfg, cid)
                pending = [p.name for p in common.audio_files(cfg, cid)
                           if common.lecture_id_for(p) not in lectures]
                idx = common.course_index(cfg, cid) / "concepts.json"
                concepts = 0
                if idx.exists():
                    try:
                        concepts = len(common.read_json(idx).get("concepts", []))
                    except json.JSONDecodeError:
                        pass
                out.append({**c, "lectures": len(lectures),
                            "unprocessed": pending, "concepts": concepts})
            return self._send_json({"courses": out}, extra_headers=setcookie)

        if route == "/api/lectures":
            if not common.valid_course_id(course):
                return self._error(400, "bad or missing course")
            data = []
            for lid in common.lecture_ids(cfg, course):
                lj = common.lecture_dir(cfg, course, lid, create=False) / "lecture.json"
                try:
                    rec = common.read_json(lj)
                except (OSError, json.JSONDecodeError):
                    continue
                data.append({"lecture_id": lid, "duration": rec.get("duration"),
                             "stages": rec.get("stages", {}),
                             "audio_file": rec.get("audio_file")})
            pending = [{"file": p.name, "lecture_id": common.lecture_id_for(p),
                        "size": p.stat().st_size}
                       for p in common.audio_files(cfg, course)
                       if common.lecture_id_for(p) not in common.lecture_ids(cfg, course)]
            return self._send_json({"course_id": course, "lectures": data,
                                    "unprocessed": pending}, extra_headers=setcookie)

        if route == "/api/notation":
            if not common.valid_course_id(course):
                return self._error(400, "bad or missing course")
            path = common.course_notes_dir(cfg, course) / "notation.md"
            text = path.read_text(encoding="utf-8") if path.exists() else ""
            return self._send_json({"course_id": course, "text": text},
                                   extra_headers=setcookie)

        if route == "/api/concepts":
            if not common.valid_course_id(course):
                return self._error(400, "bad or missing course")
            path = common.course_index(cfg, course) / "concepts.json"
            if not path.exists():
                return self._error(404, "no concept index for this course yet")
            return self._send_json(common.read_json(path), extra_headers=setcookie)

        if route == "/api/search":
            q = self._param(qs, "q", "")
            if not common.valid_course_id(course):
                return self._error(400, "bad or missing course")
            if not q:
                return self._error(400, "missing q")
            try:
                k = max(1, min(50, int(self._param(qs, "k", "10"))))
            except (TypeError, ValueError):
                k = 10
            s = self.state.searcher(course)
            if not s.ready:
                return self._error(404, "no search index for this course yet")
            try:
                return self._send_json({"query": q, "course_id": course,
                                        "results": s.search(q, k)},
                                       extra_headers=setcookie)
            except Exception as exc:
                self.state.logger.error("search failed: %s", exc)
                return self._error(500, f"search failed: {exc}")

        if route == "/api/jobs":
            return self._send_json({"jobs": self.state.runner.list()},
                                   extra_headers=setcookie)

        if route == "/api/models":
            a = cfg["asr"]
            return self._send_json({
                "backend": a.get("backend"),
                "current": a.get("model"),
                "choices": [
                    {"id": "OpenVINO/whisper-large-v3-fp16-ov",
                     "label": "large-v3 — most accurate (~9x realtime)"},
                    {"id": "OpenVINO/whisper-large-v3-turbo-fp16-ov",
                     "label": "large-v3-turbo — ~4x faster, more symbol errors"},
                    {"id": "OpenVINO/whisper-medium-fp16-ov",
                     "label": "medium — fastest, noticeably weaker"},
                ],
                "stages": [{"id": k, "label": v[0]} for k, v in jobs_mod.STAGES.items()],
                "default_stages": jobs_mod.DEFAULT_STAGES,
                "windowing": {
                    "current": str(cfg.get("notes", {}).get("windowing", "fixed")),
                    "choices": [
                        {"id": "fixed",
                         "label": "Every %d min — simple, offline"
                                  % round(float(cfg.get("notes", {})
                                                .get("window_s", 360)) / 60)},
                        {"id": "topic",
                         "label": "By topic — cut where the subject changes"},
                    ],
                },
            }, extra_headers=setcookie)

        return self._error(404, "unknown endpoint")

    # -------------------------------------------------------------- files ---

    def _resolve(self, url_path: str):
        path = urllib.parse.unquote(url_path.split("?", 1)[0].split("#", 1)[0])
        candidate = (self.root / path.lstrip("/")).resolve()
        try:
            candidate.relative_to(self.root)
        except ValueError:
            return None
        if candidate.is_dir():
            candidate = candidate / "index.html"
        return candidate

    def _serve_file(self, setcookie=None):
        target = self._resolve(self.path)
        if target is None:
            return self._error(403, "forbidden")
        if not target.is_file():
            return self._error(404, "not found")

        ctype = mimetypes.guess_type(str(target))[0] or "application/octet-stream"
        size = target.stat().st_size
        rng = self.headers.get("Range")
        cache = ("public, max-age=31536000, immutable"
                 if "/vendor/" in self.path.replace("\\", "/") else "no-store")

        if rng:
            m = re.match(r"bytes=(\d*)-(\d*)", rng.strip())
            if m and (m.group(1) or m.group(2)):
                if m.group(1):
                    start = int(m.group(1))
                    end = int(m.group(2)) if m.group(2) else size - 1
                else:
                    start = max(0, size - int(m.group(2)))
                    end = size - 1
                end = min(end, size - 1)
                if start > end or start >= size:
                    self.send_response(416)
                    self.send_header("Content-Range", f"bytes */{size}")
                    self.end_headers()
                    return
                length = end - start + 1
                self.send_response(206)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
                self.send_header("Content-Length", str(length))
                self.send_header("Accept-Ranges", "bytes")
                self.send_header("Cache-Control", cache)
                for k, v in (setcookie or []):
                    self.send_header(k, v)
                self.end_headers()
                with open(target, "rb") as fh:
                    fh.seek(start)
                    left = length
                    while left > 0:
                        buf = fh.read(min(65536, left))
                        if not buf:
                            break
                        self.wfile.write(buf)
                        left -= len(buf)
                return

        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(size))
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Cache-Control", cache)
        for k, v in (setcookie or []):
            self.send_header(k, v)
        self.end_headers()
        with open(target, "rb") as fh:
            while True:
                buf = fh.read(65536)
                if not buf:
                    break
                self.wfile.write(buf)

    # -------------------------------------------------------------- POST ----

    def do_POST(self):
        qs = self._query()
        if not self._authorised(qs):
            return self._error(401, "access key required")
        route = self.path.split("?", 1)[0]

        if route == "/api/upload":
            return self._upload(qs)

        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            return self._error(400, "bad Content-Length")
        if length <= 0 or length > MAX_JSON_BODY:
            return self._error(413, "body missing or too large")
        try:
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            return self._error(400, f"bad JSON: {exc}")
        if not isinstance(payload, dict):
            return self._error(400, "expected an object")

        if route == "/api/courses":
            return self._create_course(payload)
        if route == "/api/notation":
            return self._save_notation(payload)
        if route == "/api/process":
            return self._start_job(payload)
        if route == "/api/jobs/cancel":
            ok = self.state.runner.cancel(str(payload.get("id", "")))
            return self._send_json({"ok": ok})
        if route == "/api/corrections":
            return self._post_correction(payload)
        return self._error(404, "unknown endpoint")

    # ------------------------------------------------------------ actions ---

    def _create_course(self, payload):
        cid = str(payload.get("id", "")).strip()
        if not common.valid_course_id(cid):
            return self._error(400, "course id: letters, digits, dot, dash, "
                                    "underscore; no spaces")
        try:
            c = common.ensure_course(self.state.cfg, cid,
                                     str(payload.get("title", "")).strip() or None)
        except ValueError as exc:
            return self._error(400, str(exc))
        self.state.logger.info("course ready: %s", cid)
        return self._send_json({"ok": True, "course": c})

    def _save_notation(self, payload):
        cid = str(payload.get("course", "")).strip()
        if not common.valid_course_id(cid):
            return self._error(400, "bad course")
        text = payload.get("text")
        if not isinstance(text, str):
            return self._error(400, "text must be a string")
        if len(text) > 200_000:
            return self._error(413, "notation file too large")
        common.ensure_course(self.state.cfg, cid)
        path = common.course_notes_dir(self.state.cfg, cid) / "notation.md"
        tmp = path.with_suffix(".md.tmp")
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, path)
        self.state.logger.info("%s: notation.md saved (%d chars)", cid, len(text))
        return self._send_json({"ok": True, "bytes": len(text.encode("utf-8"))})

    def _start_job(self, payload):
        cid = str(payload.get("course", "")).strip()
        if not common.valid_course_id(cid):
            return self._error(400, "bad course")
        lecture = str(payload.get("lecture", "")).strip() or None
        if lecture and not SAFE_ID.match(lecture):
            return self._error(400, "bad lecture id")
        stages = payload.get("stages") or jobs_mod.DEFAULT_STAGES
        if not isinstance(stages, list):
            return self._error(400, "stages must be a list")
        options = payload.get("options") or {}
        if not isinstance(options, dict):
            return self._error(400, "options must be an object")
        try:
            job = self.state.runner.submit(cid, lecture, stages, options)
        except ValueError as exc:
            return self._error(400, str(exc))
        return self._send_json({"ok": True, "job": job.snapshot()})

    def _post_correction(self, payload):
        cid = str(payload.get("course_id", "")).strip()
        if not common.valid_course_id(cid):
            return self._error(400, "bad course_id")
        right = str(payload.get("right", "")).strip()
        if not right:
            return self._error(400, "'right' is required")
        wrong = str(payload.get("wrong", "")).strip()
        if len(wrong) > 2000 or len(right) > 2000:
            return self._error(400, "correction too long")

        entry = {
            "wrong": wrong, "right": right,
            "kind": str(payload.get("kind", "latex")).strip()[:40] or "latex",
            "lecture_id": str(payload.get("lecture_id", "")).strip()[:100] or None,
            "item_id": str(payload.get("item_id", "")).strip()[:64] or None,
            "source_text": str(payload.get("source_text", "")).strip()[:1000] or None,
            "note": str(payload.get("note", "")).strip()[:500] or None,
            "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }

        common.ensure_course(self.state.cfg, cid)
        path = common.course_notes_dir(self.state.cfg, cid) / "glossary.json"
        with self.state.lock:
            try:
                data = common.read_json(path) if path.exists() else {}
            except json.JSONDecodeError:
                return self._error(500, "glossary.json is not valid JSON; fix it by hand")
            if not isinstance(data, dict):
                data = {}
            data.setdefault("terms", [])
            data.setdefault("aliases", {})
            corrections = data.setdefault("corrections", [])
            replaced = False
            for i, c in enumerate(corrections):
                if (isinstance(c, dict) and c.get("item_id")
                        and c.get("item_id") == entry["item_id"]
                        and c.get("lecture_id") == entry["lecture_id"]):
                    corrections[i] = entry
                    replaced = True
                    break
            if not replaced:
                corrections.append(entry)
            if entry["kind"] == "alias" and wrong:
                data["aliases"][wrong.lower()] = right.lower()
            common.atomic_write_json(path, data)

        self.state.logger.info("%s correction saved: %r -> %r", cid,
                               wrong[:40], right[:40])
        return self._send_json({"ok": True, "replaced": replaced,
                                "total": len(corrections)})

    # ------------------------------------------------------------- upload ---

    def _upload(self, qs):
        """Streamed multipart upload straight to audio/<course>/.

        Written to a .part file and renamed on success, so an interrupted
        upload never looks like a lecture waiting to be processed.
        """
        cfg = self.state.cfg
        ucfg = cfg.get("server", {}).get("uploads", {})
        max_bytes = int(float(ucfg.get("max_mb", 2048)) * 1024 * 1024)
        allowed = {e.lower() for e in ucfg.get("allowed_ext", [])} or common.AUDIO_EXTS

        course = self._param(qs, "course")
        if not common.valid_course_id(course):
            return self._error(400, "bad or missing course")

        ctype = self.headers.get("Content-Type", "")
        m = re.search(r'boundary="?([^";]+)"?', ctype)
        if "multipart/form-data" not in ctype or not m:
            return self._error(400, "expected multipart/form-data")
        boundary = ("--" + m.group(1)).encode()

        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            return self._error(400, "bad Content-Length")
        if length <= 0:
            return self._error(400, "empty upload")
        if length > max_bytes + 8192:
            return self._error(413, f"file exceeds {ucfg.get('max_mb', 2048)} MB")

        common.ensure_course(cfg, course)
        dest_dir = common.course_audio(cfg, course)
        dest_dir.mkdir(parents=True, exist_ok=True)

        reader = _MultipartReader(self.rfile, boundary, length)
        try:
            filename, tmp_path, written = reader.first_file(dest_dir, max_bytes)
        except _UploadError as exc:
            return self._error(400, str(exc))
        except Exception as exc:
            self.state.logger.error("upload failed: %s", exc)
            return self._error(500, f"upload failed: {exc}")

        if not filename:
            return self._error(400, "no file part found")

        ext = Path(filename).suffix.lower()
        if ext not in allowed:
            tmp_path.unlink(missing_ok=True)
            return self._error(400, f"{ext or 'that file type'} is not accepted")

        safe = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(filename).name).strip("._") or "lecture"
        final = dest_dir / safe
        n = 1
        while final.exists():
            final = dest_dir / f"{Path(safe).stem}-{n}{Path(safe).suffix}"
            n += 1
        os.replace(tmp_path, final)

        lecture_id = common.lecture_id_for(final)
        self.state.logger.info("%s: uploaded %s (%.1f MB) -> lecture %s",
                               course, final.name, written / 1e6, lecture_id)
        return self._send_json({"ok": True, "file": final.name,
                                "lecture_id": lecture_id, "bytes": written})


class _UploadError(Exception):
    pass


class _MultipartReader:
    """Just enough multipart to pull the first file part out, streaming.

    cgi.FieldStorage is gone in 3.13 and buffers everything anyway; a lecture
    can be a gigabyte, so it goes to disk as it arrives.
    """

    def __init__(self, stream, boundary, length):
        self.stream = stream
        self.boundary = boundary
        self.remaining = length

    def _readline(self):
        if self.remaining <= 0:
            return b""
        line = self.stream.readline(min(65536, self.remaining + 2))
        self.remaining -= len(line)
        return line

    def first_file(self, dest_dir, max_bytes):
        filename = None
        # Walk headers until the part that declares a filename.
        while True:
            line = self._readline()
            if not line:
                raise _UploadError("malformed upload: no file part")
            if not line.startswith(self.boundary):
                continue
            headers = {}
            while True:
                h = self._readline()
                if not h or h in (b"\r\n", b"\n"):
                    break
                k, _, v = h.decode("utf-8", "replace").partition(":")
                headers[k.strip().lower()] = v.strip()
            disp = headers.get("content-disposition", "")
            fm = re.search(r'filename="([^"]*)"', disp)
            if fm and fm.group(1):
                filename = fm.group(1)
                break

        tmp = dest_dir / f".upload-{secrets.token_hex(6)}.part"
        written = 0
        tail = b""
        try:
            with open(tmp, "wb") as out:
                while True:
                    chunk = self.stream.read(min(262144, max(0, self.remaining)))
                    if not chunk:
                        break
                    self.remaining -= len(chunk)
                    buf = tail + chunk
                    idx = buf.find(self.boundary)
                    if idx >= 0:
                        body = buf[:idx]
                        if body.endswith(b"\r\n"):
                            body = body[:-2]
                        out.write(body)
                        written += len(body)
                        break
                    # Hold back enough to catch a boundary split across reads.
                    keep = len(self.boundary) + 4
                    if len(buf) > keep:
                        out.write(buf[:-keep])
                        written += len(buf) - keep
                        tail = buf[-keep:]
                    else:
                        tail = buf
                    if written > max_bytes:
                        raise _UploadError("file too large")
                    if self.remaining <= 0:
                        out.write(tail.split(self.boundary)[0].rstrip(b"\r\n"))
                        written += len(tail)
                        break
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise
        return filename, tmp, written


def resolve_key(cfg, host, cli_no_auth, cli_key):
    mode = str(cfg.get("server", {}).get("auth", "auto")).strip()
    if cli_no_auth:
        return None
    if cli_key:
        return cli_key
    if mode.lower() == "off":
        return None
    if mode.lower() == "auto":
        return None if is_loopback(host) else secrets.token_urlsafe(12)
    return mode


def main() -> int:
    ap = argparse.ArgumentParser(description="Lecture site: viewer, uploads and library.")
    ap.add_argument("--host", default=None)
    ap.add_argument("--port", type=int, default=None)
    ap.add_argument("--key", default=None, help="use this access key")
    ap.add_argument("--no-auth", action="store_true", help="disable the access key")
    ap.add_argument("--open", action="store_true")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    cfg = common.load_config(args.config)
    scfg = cfg.get("server", {})
    host = args.host or scfg.get("host", "127.0.0.1")
    port = int(args.port or scfg.get("port", 8000))
    logger = common.setup_logging("server", cfg)

    key = resolve_key(cfg, host, args.no_auth, args.key)
    Handler.state = State(cfg, logger, key)
    Handler.root = Path(cfg["_root"]).resolve()

    courses = common.load_courses(cfg)
    logger.info("%d course(s): %s", len(courses),
                ", ".join(c["id"] for c in courses) or "none yet")

    shown = "127.0.0.1" if host in ("0.0.0.0", "::") else host
    url = f"http://{shown}:{port}/"
    httpd = ThreadingHTTPServer((host, port), Handler)

    logger.info("serving %s on %s:%d", Handler.root, host, port)
    if key:
        logger.info("")
        if is_loopback(host):
            logger.info("  access key required (you asked for one)")
        else:
            logger.info("  access key required -- bound to %s, which is more "
                        "than loopback", host)
        logger.info("  open:  %s?key=%s", url, key)
        if not is_loopback(host):
            for ip in local_addresses():
                tag = "  (Tailscale)" if ip.startswith("100.") else ""
                logger.info("         http://%s:%d/?key=%s%s", ip, port, key, tag)
        logger.info("")
    elif not is_loopback(host):
        logger.warning("bound to %s with NO access key -- anyone who can reach "
                       "this host can upload and start jobs", host)
    else:
        logger.info("open %s", url)

    if args.open:
        threading.Timer(0.6, lambda: webbrowser.open(
            url + (f"?key={key}" if key else ""))).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        logger.info("shutting down")
    finally:
        httpd.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
