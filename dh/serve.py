"""Local service of the "Run jobs" page. It listens on 127.0.0.1 only and blocks every outside connection of its
own process (workers/no_network.py); the models are read from the local Hugging Face cache.

    python -m dh.serve            # http://127.0.0.1:8765 (the Vite dev server proxies /api and /thumbs)
    python -m dh.serve 9000       # another port

GET  /api/health                             {"ok": true}
GET  /api/meter                              the processor's power in the last minute (dh/meter.py; Windows only)
GET  /api/jobs/status[?warm=embed,photo,llm] which models are loaded (warm starts loading them)
GET  /api/embed/search?q=...&k=8             the assumptions closest in meaning to a sentence
GET  /api/photo/sample?n=12&seed=0           a handful of archive photos to start from
GET  /api/photo/similar?signature=...        archive photos most like one archive photo
POST /api/photo/similar                      the same for an uploaded image (body: the raw image file)
GET  /api/render?cx=&cy=&span=&w=&h=&iters=  a view of the Mandelbrot set as PNG; figures in X-Render-* headers
GET  /api/caption/status[?warm=1]            is the photo-description model loaded
POST /api/caption                            body: the raw image file; returns one English sentence
POST /api/llm/generate                       body {"prompt": "..."}; the answer comes back while it is written,
                                             one JSON object per line
GET  /thumbs/<signature>.jpg                 a downloaded thumbnail (memobase/download_thumbs.py)
"""
from __future__ import annotations

import csv
import json
import re
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
THUMBS = ROOT / "data" / "memobase" / "thumbs"
MANIFEST = ROOT / "data" / "memobase" / "manifest.csv"

_meta: dict[str, dict] | None = None
_caption_lock = threading.Lock()          # one description at a time (the model is shared)
_caption_state = {"loaded": False, "loading": False, "error": None, "load_s": None}
MAX_IMAGE_BYTES = 15_000_000
IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp", "image/gif", "image/bmp", "image/tiff"}
FORMATS = "this format is not supported; use JPEG, PNG, WebP, GIF, BMP or TIFF"
RETRY_AFTER_S = 30                         # a model that failed to load is tried again after this long, not at once
BAD_NUMBER = "a number in the address could not be read"


def caption_evidence() -> dict:
    from workers import no_network
    from workers.jobs import CAPTION_ID, CAPTION_REVISION, THREADS
    return {"model": CAPTION_ID, "revision": CAPTION_REVISION[:12], "torch_threads": THREADS,
            "guard_installed": no_network.installed(), "library_offline_flags": no_network.library_offline_flags(),
            "blocked_attempts": len(no_network.attempts)}


def load_caption_model() -> None:
    """Load Florence-2-base once (about 20 s); safe to call from several threads."""
    with _caption_lock:
        if _caption_state["loaded"]:
            return
        _caption_state.update(loading=True, error=None)
        t0 = time.perf_counter()
        try:
            from workers.jobs import load_captioner
            load_captioner()                       # switches the network guard on first (jobs.ensure_offline)
            _caption_state.update(loaded=True, load_s=round(time.perf_counter() - t0, 1))
        except Exception as e:  # noqa: BLE001 - reported to the page
            import traceback
            traceback.print_exc()
            _caption_state["error"] = ("the description model is not on this computer" if isinstance(e, OSError)
                                       else f"the description model could not be loaded ({type(e).__name__})")
            _caption_state["failed_at"] = time.monotonic()
        finally:
            _caption_state["loading"] = False


def caption_status(warm: bool) -> dict:
    rested = time.monotonic() - _caption_state.get("failed_at", -1e9) > RETRY_AFTER_S
    if warm and not _caption_state["loaded"] and not _caption_state["loading"] and rested:
        _caption_state["loading"] = True
        threading.Thread(target=load_caption_model, daemon=True).start()
    out = {k: v for k, v in _caption_state.items() if k != "failed_at"}
    if _caption_state["loaded"]:
        out["offline"] = caption_evidence()
    return out


def describe(data: bytes, filename: str) -> tuple[int, dict]:
    """Describe one uploaded image with the caption job's function. The archive record is looked up when the
    file name is a Memobase signature (for the credit line); any other image works the same way."""
    from dh.tryjobs import BadImage, open_image
    from workers import no_network
    try:
        img, w, h = open_image(data, 2048)
    except BadImage as e:
        return e.code, {"error": str(e)}
    load_caption_model()
    if not _caption_state["loaded"]:
        return 500, {"error": _caption_state["error"] or "the description model could not be loaded"}
    from workers.jobs import caption_images
    from dh import meter
    with _caption_lock:
        before = len(no_network.attempts)
        with meter.Span() as span:
            text = caption_images([img])[0]
        secs = span.seconds
        blocked = len(no_network.attempts) - before
    out = {"description": text, "seconds": round(secs, 2), "width": w, "height": h, "offline": caption_evidence(),
           "blocked_during": blocked, "work": span.figures()}
    stem = Path(filename).stem
    rec = manifest().get(stem)
    if rec:
        out["archive"] = {"title": rec["title"], "date": rec["date"], "institution": rec["institution"],
                          "licence": rec["licence"], "object_url": rec["object_url"],
                          "creator": rec["creator"] or rec["exif_copyright"]}
    return 200, out


def manifest() -> dict[str, dict]:
    global _meta
    if _meta is None:
        with MANIFEST.open(encoding="utf-8") as f:
            _meta = {r["signature"]: r for r in csv.DictReader(f)}
    return _meta


class Handler(BaseHTTPRequestHandler):
    timeout = 30                           # seconds a connection may stall while its request is being read

    def _send(self, code: int, body: bytes, ctype: str, headers: dict | None = None) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        fresh = ctype.startswith("application/json") or bool(headers)
        self.send_header("Cache-Control", "no-store" if fresh else "max-age=3600")
        for k, v in (headers or {}).items():
            self.send_header(k, str(v))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code: int, obj: dict) -> None:
        self._send(code, json.dumps(obj, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")

    def _stream(self, items) -> None:
        """Send dicts as they come, one JSON object per line. If the reader goes away the source is closed,
        which stops the work behind it."""
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()
        def line(item: dict) -> None:
            self.wfile.write((json.dumps(item, ensure_ascii=False) + "\n").encode("utf-8"))
            self.wfile.flush()
        try:
            for item in items:
                line(item)
        except OSError:                                # the page closed the connection: nothing left to tell
            pass
        except Exception:  # noqa: BLE001 - the answer has begun, so the error goes into it as its last line
            import traceback
            traceback.print_exc()
            try:
                line({"error": "the service could not finish this answer"})
            except OSError:
                pass
        finally:
            items.close()

    def log_message(self, fmt, *args):  # quiet: the dashboard polls several times a second
        pass

    def _guarded(self, handler) -> None:
        try:
            handler()
        except Exception:  # noqa: BLE001 - the page gets a plain answer, the details stay in this output
            import traceback
            traceback.print_exc()
            try:
                self._json(500, {"error": "the service could not answer this request"})
            except Exception:  # noqa: BLE001 - the connection is already gone
                pass

    def do_GET(self):  # noqa: N802 - http.server API
        self._guarded(self._get)

    def do_POST(self):  # noqa: N802
        self._guarded(self._post)

    def _get(self):
        u = urlparse(self.path)
        if u.path == "/api/health":
            return self._json(200, {"ok": True, "running": False})
        if u.path == "/api/meter":
            from dh import meter
            return self._json(200, meter.now())
        if u.path == "/api/caption/status":
            warm = (parse_qs(u.query).get("warm") or ["0"])[0] == "1"
            return self._json(200, caption_status(warm))
        if u.path.startswith(("/api/jobs/", "/api/embed/", "/api/photo/", "/api/render")):
            from dh import tryjobs
            qs = parse_qs(u.query)

            def arg(name: str, default: str) -> str:
                return (qs.get(name) or [default])[0]
            try:
                if u.path == "/api/embed/search":
                    args = (arg("q", ""), int(arg("k", "8")))
                elif u.path == "/api/photo/sample":
                    args = (int(arg("n", "12")), int(arg("seed", "0")))
                elif u.path == "/api/photo/similar":
                    args = (arg("signature", ""), int(arg("k", "12")))
                elif u.path == "/api/render":
                    args = (float(arg("cx", "-0.6")), float(arg("cy", "0")), float(arg("span", "3.2")),
                            int(arg("w", "640")), int(arg("h", "400")), int(arg("iters", "200")))
                else:
                    args = ()
            except ValueError:
                return self._json(400, {"error": BAD_NUMBER})
            if u.path == "/api/jobs/status":
                return self._json(200, tryjobs.status([w for w in arg("warm", "").split(",") if w]))
            if u.path == "/api/embed/search":
                return self._json(*tryjobs.embed_search(*args))
            if u.path == "/api/photo/sample":
                return self._json(*tryjobs.photo_sample(*args))
            if u.path == "/api/photo/similar":
                return self._json(*tryjobs.photo_similar_signature(*args))
            if u.path == "/api/render":
                png, info = tryjobs.render(*args)
                return self._send(200, png, "image/png", {
                    "X-Render-Seconds": info["seconds"], "X-Render-Pixels": info["pixels"],
                    "X-Render-Iters": info["iters"], "X-Render-Tiles": info["tiles"],
                    "X-Render-Width": info["width"], "X-Render-Height": info["height"],
                    "X-Render-Joules": "" if info["joules"] is None else info["joules"],
                    "X-Render-Watts": "" if info["watts"] is None else info["watts"],
                    "X-Render-Flops": "" if info["flops"] is None else int(info["flops"])})
        m = re.fullmatch(r"/thumbs/([A-Za-z0-9_.\-]+\.jpg)", u.path)
        if m and ".." not in m.group(1):
            f = THUMBS / m.group(1)
            if f.exists():
                return self._send(200, f.read_bytes(), "image/jpeg")
        return self._json(404, {"error": "not found"})

    def _post(self):
        u = urlparse(self.path)
        try:
            size = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            size = -1
        if size < 0:
            return self._json(400, {"error": "the request has no readable length"})
        if u.path == "/api/caption":
            ctype = (self.headers.get("Content-Type") or "").split(";")[0].strip().lower()
            if ctype not in IMAGE_TYPES:
                return self._json(415, {"error": FORMATS})
            if not 0 < size <= MAX_IMAGE_BYTES:
                return self._json(413, {"error": f"the image must be between 1 byte and {MAX_IMAGE_BYTES // 1_000_000} MB"})
            data = self.rfile.read(size)
            name = (parse_qs(u.query).get("name") or [""])[0][:200]
            code, obj = describe(data, name)
            return self._json(code, obj)
        if u.path == "/api/photo/similar":
            ctype = (self.headers.get("Content-Type") or "").split(";")[0].strip().lower()
            if ctype not in IMAGE_TYPES:
                return self._json(415, {"error": FORMATS})
            if not 0 < size <= MAX_IMAGE_BYTES:
                return self._json(413, {"error": f"the image must be between 1 byte and {MAX_IMAGE_BYTES // 1_000_000} MB"})
            from dh import tryjobs
            k = (parse_qs(u.query).get("k") or ["12"])[0]
            return self._json(*tryjobs.photo_similar_upload(self.rfile.read(size), int(k) if re.fullmatch(r"[0-9]{1,3}", k) else 12))
        length = min(size, 32_000)
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
        except ValueError:                             # not JSON, or not UTF-8
            return self._json(400, {"error": "body must be JSON"})
        if u.path == "/api/llm/generate":
            if not isinstance(body, dict) or not isinstance(body.get("prompt"), str):
                return self._json(400, {"error": "send the request as {\"prompt\": \"...\"}"})
            from dh import tryjobs
            if not tryjobs.clean_prompt(body["prompt"]):
                return self._json(400, {"error": "the request is empty"})
            return self._stream(tryjobs.llm_stream(body["prompt"]))
        if not isinstance(body, dict):
            return self._json(400, {"error": "body must be a JSON object"})
        return self._json(404, {"error": "not found"})


def main() -> None:
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
    from workers.jobs import ensure_offline
    ensure_offline()                       # outside connections are refused from the start; models load from disk
    print("reading the photo index…", flush=True)
    t0 = time.perf_counter()
    from memobase.search import load_index
    load_index()
    manifest()
    print(f"ready in {time.perf_counter() - t0:.1f} s: http://127.0.0.1:{port}/api/health", flush=True)
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()


if __name__ == "__main__":
    main()
