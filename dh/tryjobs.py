"""Live, one-off use of three job types from the dashboard's "Run jobs" page. Every answer is computed in
this process with the job's own model or arithmetic (workers/jobs.py), with the network guard switched on.

embed   A typed sentence is embedded with all-MiniLM-L6-v2 (the model of the text-embedding job) and compared
        with the entries of the assumptions table (web/public/data/assumptions.json), which are embedded once
        with the same model. The page shows the entries closest in meaning.
photo   A photo is compared with the vectors the photo-indexing job wrote (data/memobase/index_clip_b32.npz).
        An archive photo is looked up by its signature (its stored vector is used); an uploaded photo is
        embedded first with the job's CLIP ViT-B/32 model. Only photos whose archive record carries no extra
        conditions of use are shown and compared (data/memobase/README.md asks for that when photos are shown).
render  A view of the Mandelbrot set is rendered with the arithmetic of the tile job (workers.jobs.render_view)
        and returned as a PNG.

llm     A request is answered by a small open language model (workers.jobs.write_text), piece by piece. This
        one is not a job type of the heat schedule: it has no batch job there. dh/measure.py times it on the
        page's example requests.

Every answer of a model carries its operations, counted while it is computed (dh/flops.py).

    python -m dh.tryjobs --demo      # saved examples for a copy of the dashboard without the local service
                                     # (the photos of the saved descriptions are described again, with figures)
    python -m dh.tryjobs --describe  # only that last part: the saved descriptions again, with figures
    python -m dh.tryjobs --recount   # keep the saved times and energies, count the operations of the saved answers again
"""
from __future__ import annotations

import csv
import json
import queue
import re
import shutil
import statistics
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from dh import flops, meter  # noqa: E402
ASSUMPTIONS = ROOT / "web" / "public" / "data" / "assumptions.json"
THUMBS = ROOT / "data" / "memobase" / "thumbs"
TAGS = ROOT / "data" / "memobase" / "tags.csv"
PEOPLE_TAGS = {"Porträt", "Gruppe", "Fest"}     # tags of memobase/search.py tag_all that mostly mean people
DEMO_OUT = ROOT / "web" / "public" / "data"
DESCRIBE_DEMO = DEMO_OUT / "describe_demo.json"
LLM_EXAMPLES = ROOT / "web" / "src" / "llm_examples.json"      # the example requests of the page (one source)
SIG_RE = re.compile(r"^[A-Za-z0-9_.\-]{1,120}$")
EMBED_MODEL = "all-MiniLM-L6-v2"
TILE_PIXELS = 320 * 40            # one tile of the tile job: render_chunk(size=320) draws strips of 320 x 40
TILE_ITERS = 200
MAX_W, MAX_H, MAX_ITERS = 960, 640, 400
STRIP_ROWS = 40                   # the live view is rendered in strips of this height, several at a time
RENDER_THREADS = 4
MAX_PIXELS = 40_000_000           # an uploaded image may not decode to more than this
MIN_SIDE = 8

_locks = {"embed": threading.Lock(), "photo": threading.Lock(), "render": threading.Lock()}
_decode = threading.BoundedSemaphore(2)   # uploads being decoded at the same moment (each may need a lot of memory)
RETRY_AFTER_S = 30                # a model that failed to load is tried again after this long, not at once
_load_lock = threading.Lock()     # models load one after the other, so the timing of one is not slowed by the other
_state = {k: {"loaded": False, "loading": False, "error": None, "load_s": None} for k in ("embed", "photo", "llm")}
_embed: dict = {}
_photo: dict = {}
_llm: dict = {}
_llm_gate = threading.Lock()      # one answer at a time: the model is shared
_llm_now: dict = {"stop": None}   # the stop switch of the answer being written; a new request flips it
_llm_swap = threading.Lock()      # so that two requests arriving together cannot both miss the answer to stop
_render_gate = threading.Lock()   # one picture at a time, so that its energy reading is its own
MAX_PROMPT = 2000                 # characters of one request to the language model
LLM_NAMES = {"embed": "the text model", "photo": "the photo model or the photo index", "llm": "the language model"}


class BadImage(Exception):
    """An upload that cannot be used; `code` is the HTTP status and the message is safe to show."""

    def __init__(self, code: int, message: str):
        super().__init__(message)
        self.code = code


def open_image(data: bytes, max_side: int):
    """Decode an uploaded image safely. Returns (RGB image no larger than max_side, original width, height)."""
    from PIL import Image, UnidentifiedImageError
    try:
        with _decode, Image.open(BytesIO(data)) as im:
            w, h = im.size
            if w * h > MAX_PIXELS:
                raise BadImage(413, f"the image has too many pixels (at most {MAX_PIXELS // 1_000_000} million)")
            if min(w, h) < MIN_SIDE:
                raise BadImage(400, f"the image is too small (at least {MIN_SIDE} pixels on each side)")
            im.draft("RGB", (max_side, max_side))          # lets a JPEG decode at reduced size; no effect otherwise
            im.load()
            if im.mode in ("L", "RGB"):
                im.thumbnail((max_side, max_side))          # shrink before the RGB copy is made, so that copy is small
            img = im.convert("RGB")
    except BadImage:
        raise
    except Image.DecompressionBombError:
        raise BadImage(413, f"the image has too many pixels (at most {MAX_PIXELS // 1_000_000} million)") from None
    except (UnidentifiedImageError, OSError, ValueError, SyntaxError, EOFError) as e:
        raise BadImage(400, f"not a readable image ({type(e).__name__})") from None
    img.thumbnail((max_side, max_side))
    return img, w, h


def _evidence() -> dict:
    from workers import no_network
    return {"guard_installed": no_network.installed(), "blocked_attempts": len(no_network.attempts)}


def _plain(e: Exception, what: str) -> str:
    """A message for the page without file paths; the full error goes to the service's own output."""
    traceback.print_exception(type(e), e, e.__traceback__, file=sys.stderr)
    if isinstance(e, (FileNotFoundError, OSError)):
        return f"{what} is not on this computer"
    return f"{what} could not be loaded ({type(e).__name__})"


# ------------------------------------------------------------------ loading
def _load_embed() -> None:
    from workers.jobs import ensure_offline, load_embedder
    ensure_offline()
    model = load_embedder()
    flops.watch(model)
    entries = json.loads(ASSUMPTIONS.read_text(encoding="utf-8"))
    texts = [" ".join(str(x) for x in (e.get("label"), e.get("note"), e.get("source")) if x) for e in entries]

    def encode():
        return model.encode(texts, batch_size=32, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
    vec = encode()                                             # the first call warms up
    times = []
    for _ in range(3):
        t0 = time.perf_counter()
        vec = encode()
        times.append(time.perf_counter() - t0)
    secs = statistics.median(times)
    _embed.update(model=model, entries=entries, vectors=vec.astype(np.float32),
                  corpus={"items": len(texts), "seconds": round(secs, 3), "per_second": round(len(texts) / secs, 1),
                          "dims": int(vec.shape[1])})


def _load_index() -> None:
    """The photo index and the archive records: enough to compare archive photos with each other."""
    if "vectors" in _photo:
        return
    from memobase.search import load_index, manifest
    vec, sigs = load_index()
    vec = vec / np.maximum(np.linalg.norm(vec, axis=1, keepdims=True), 1e-9)      # stored as float16: renormalise
    meta = manifest()
    free = np.array([i for i, s in enumerate(sigs) if meta[s]["extra_conditions"] == "no"], dtype=np.int64)
    # The handful of photos a visitor starts from prefers objects and places over people, as data/memobase/README.md
    # asks for the pitch. The tags come from a general model, so this is a preference and not a guarantee.
    start = free
    try:
        with TAGS.open(encoding="utf-8", newline="") as f:
            tags = {r["signature"]: {r["tag1"], r["tag2"]} for r in csv.DictReader(f)}
        calm = [int(i) for i in free if not (tags.get(sigs[int(i)], set()) & PEOPLE_TAGS)]
        if len(calm) >= 48:
            start = np.array(calm, dtype=np.int64)
    except (OSError, KeyError):
        pass
    _photo.update(vectors=vec, sigs=sigs, where={s: i for i, s in enumerate(sigs)}, meta=meta, free=free,
                  free_vectors=vec[free], start=start)


def _load_photo() -> None:
    from PIL import Image

    from workers.jobs import ensure_offline, load_clip
    ensure_offline()
    _load_index()
    model = load_clip()
    flops.watch(model)
    model.encode([Image.new("RGB", (224, 224))], batch_size=1, convert_to_numpy=True, show_progress_bar=False)   # warm-up
    _photo["model"] = model


def _load(kind: str) -> None:
    st = _state[kind]
    if st["loaded"]:                  # the usual case: no waiting behind another model that is still loading
        return
    with _load_lock:
        if st["loaded"]:
            return
        st.update(loading=True, error=None)
        t0 = time.perf_counter()
        try:
            {"embed": _load_embed, "photo": _load_photo, "llm": _load_llm}[kind]()
            st.update(loaded=True, load_s=round(time.perf_counter() - t0, 1))
        except Exception as e:  # noqa: BLE001 - reported to the page in plain words
            st["error"] = _plain(e, LLM_NAMES[kind])
            st["failed_at"] = time.monotonic()
        finally:
            st["loading"] = False


def status(warm: list[str]) -> dict:
    """State of the two models; `warm` names the ones to start loading in the background."""
    todo = []
    for kind in warm:
        st = _state.get(kind)
        if st and not st["loaded"] and not st["loading"] and time.monotonic() - st.get("failed_at", -1e9) > RETRY_AFTER_S:
            st["loading"] = True
            todo.append(kind)
    if todo:
        threading.Thread(target=lambda: [_load(k) for k in todo], daemon=True).start()
    out = {k: {a: b for a, b in v.items() if a != "failed_at"} for k, v in _state.items()}
    if _state["embed"]["loaded"]:
        out["embed"]["corpus"] = _embed["corpus"]
    if _state["llm"]["loaded"]:
        out["llm"].update(_llm["facts"])
    try:
        _load_index()
        out["photo"]["photos"] = len(_photo["sigs"])
        out["photo"]["photos_shown"] = int(len(_photo["free"]))
    except Exception as e:  # noqa: BLE001
        out["photo"]["index_error"] = _plain(e, "the photo index")
    out["render"] = {"loaded": True, "max_width": MAX_W, "max_height": MAX_H, "max_iters": MAX_ITERS}
    out["offline"] = _evidence()
    return out


# ------------------------------------------------------------------ text generation (small language model)
def _load_llm() -> None:
    """Load the model, write a few words to warm up, then time a short answer: the speed the page shows as
    'measured when the model loaded'."""
    from workers import jobs
    flops.watch(jobs.load_writer()[1])
    jobs.write_text("Say hello.", max_new_tokens=8)
    timed = jobs.write_text("Count from one to twenty in words.", max_new_tokens=40)
    _llm["weights"] = meter.writer_weights(jobs.load_writer()[1])
    _llm["facts"] = {"model": jobs.WRITER_ID.split("/")[-1], "parameters": jobs.WRITER_PARAMS,
                     "licence": jobs.WRITER_LICENCE, "threads": jobs.THREADS,
                     "max_tokens": jobs.WRITER_MAX_TOKENS, "measured_tokens_per_s": timed["tokens_per_s"]}


def clean_prompt(prompt: str) -> str:
    """The request as the model gets it: printable characters and line breaks, no more than MAX_PROMPT."""
    kept = "".join(ch if ch.isprintable() or ch == "\n" else " " for ch in str(prompt))
    return "\n".join(" ".join(line.split()) for line in kept.split("\n")).strip()[:MAX_PROMPT]


def llm_work(span, out: dict, count=None) -> dict:
    """Energy of one answer and its operations: counted while it was written (dh/flops.py, weight and attention
    products). Without a count, the products with the weights by formula (meter.writer_flops), which the weight
    part of the count equals exactly."""
    if count is not None and count.total:
        return span.figures(count.total)
    body, head = _llm["weights"]
    return span.figures(meter.writer_flops(body, head, int(out["prompt_tokens"]), int(out.get("produced") or out["tokens"])))


def llm_stream(prompt: str, max_tokens: int | None = None):
    """Write the answer to one request and hand it over piece by piece: dicts {"text": piece}, then one last
    dict with "done" and the figures, or {"error": ...}. One answer is written at a time; a new request stops
    the one before it, and so does closing this generator (the reader went away)."""
    from workers import jobs
    prompt = clean_prompt(prompt)
    if not prompt:
        yield {"error": "the request is empty"}
        return
    _load("llm")
    if not _state["llm"]["loaded"]:
        yield {"error": _state["llm"]["error"] or "the language model could not be loaded"}
        return
    limit = max(1, min(jobs.WRITER_MAX_TOKENS, int(max_tokens or jobs.WRITER_MAX_TOKENS)))
    stop = threading.Event()
    with _llm_swap:
        before, _llm_now["stop"] = _llm_now["stop"], stop
    if before is not None:
        before.set()
    with _llm_gate:
        if stop.is_set():                              # a newer request arrived while this one waited
            yield {"error": "replaced by a newer request"}
            return
        pieces: queue.Queue = queue.Queue()

        def work():
            try:
                with meter.Span() as span, flops.Count() as count:     # the count belongs to this thread
                    out = jobs.write_text(prompt, max_new_tokens=limit, on_text=lambda s: pieces.put({"text": s}), stop=stop.is_set)
                out.pop("text", None)
                pieces.put({"done": True, **out, "model": _llm["facts"]["model"], "offline": _evidence(),
                            "work": llm_work(span, out, count)})
            except Exception as e:  # noqa: BLE001 - the page gets a plain sentence, the details stay here
                traceback.print_exception(type(e), e, e.__traceback__, file=sys.stderr)
                pieces.put({"error": f"the language model could not answer ({type(e).__name__})"})
            finally:
                pieces.put(None)
        worker = threading.Thread(target=work, daemon=True)
        worker.start()
        try:
            while True:
                item = pieces.get()
                if item is None:
                    break
                yield item
        finally:
            stop.set()
            worker.join()
            with _llm_swap:
                if _llm_now["stop"] is stop:
                    _llm_now["stop"] = None


# ------------------------------------------------------------------ text embeddings
def embed_search(query: str, k: int = 8) -> tuple[int, dict]:
    query = " ".join("".join(ch if ch.isprintable() else " " for ch in query).split())[:300]
    if not query:
        return 200, {"query": "", "results": []}
    _load("embed")
    if not _state["embed"]["loaded"]:
        return 500, {"error": _state["embed"]["error"] or "the text model could not be loaded"}
    with _locks["embed"]:
        tokens = int(_embed["model"].tokenize([query])["input_ids"].shape[1])
        with meter.Span() as span, flops.Count() as count:
            q = _embed["model"].encode([query], convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)[0]
        secs = span.seconds
    scores = _embed["vectors"] @ q.astype(np.float32)
    out = []
    for j in np.argsort(-scores)[:max(1, min(24, k))]:
        e = _embed["entries"][int(j)]
        out.append({"id": e.get("id"), "label": e.get("label"), "group": e.get("group"), "value": e.get("value"),
                    "unit": e.get("unit"), "status": e.get("status"), "note": e.get("note"),
                    "score": round(float(scores[j]), 4)})
    return 200, {"query": query, "seconds": round(secs, 4), "results": out, "corpus": _embed["corpus"],
                 "model": EMBED_MODEL, "offline": _evidence(), "tokens": tokens,
                 "work": span.figures(count.total or meter.embed_flops(tokens))}


# ------------------------------------------------------------------ photo indexing
def _hit(sig: str, score: float | None) -> dict:
    from memobase.search import credit
    r = _photo["meta"][sig]
    h = {"signature": sig, "record_id": r["record_id"], "title": r["title"], "date": r["date"],
         "creator": r["creator"] or r["exif_copyright"], "institution": r["institution"], "licence": r["licence"],
         "licence_url": r["licence_url"], "object_url": r["object_url"], "extra_conditions": r["extra_conditions"],
         "thumb": f"thumbs/{sig}.jpg", "credit": credit(r)}
    if score is not None:
        h["score"] = round(float(score), 4)
    return h


def _nearest(v: np.ndarray, k: int, skip: str | None = None) -> tuple[list[dict], float]:
    """The photos without extra conditions of use whose vectors are closest to v."""
    t0 = time.perf_counter()
    scores = _photo["free_vectors"] @ v
    order = np.argsort(-scores)
    secs = time.perf_counter() - t0
    out = []
    for j in order:
        s = _photo["sigs"][int(_photo["free"][int(j)])]
        if s == skip:
            continue
        out.append(_hit(s, scores[j]))
        if len(out) >= max(1, min(24, k)):
            break
    return out, secs


def photo_sample(n: int = 12, seed: int = 0) -> tuple[int, dict]:
    """A fixed-seed handful of archive photos to start from: records without extra conditions of use, and among
    them the ones not tagged as portraits, groups or festivities."""
    _load_index()
    rng = np.random.default_rng(int(seed) % 100000)
    free, start = _photo["free"], _photo["start"]
    pick = [_photo["sigs"][int(start[int(i)])] for i in rng.permutation(len(start))[:max(1, min(24, n))]]
    return 200, {"photos": len(_photo["sigs"]), "photos_shown": int(len(free)), "results": [_hit(s, None) for s in pick]}


def photo_similar_signature(sig: str, k: int = 12) -> tuple[int, dict]:
    _load_index()
    if not SIG_RE.match(sig) or sig not in _photo["where"] or _photo["meta"][sig]["extra_conditions"] != "no":
        return 404, {"error": "no archive photo with this signature"}
    hits, secs = _nearest(_photo["vectors"][_photo["where"][sig]], k, skip=sig)
    # No model runs for an archive photo: its vector was computed by the photo-indexing job. What is done here is
    # the comparison alone, far too brief for an energy reading; "vector_flops" is what the vector itself took.
    return 200, {"query": _hit(sig, None), "rank_seconds": round(secs, 6), "photos": len(_photo["sigs"]),
                 "photos_shown": int(len(_photo["free"])), "results": hits,
                 "compare_flops": meter.rank_flops(len(_photo["free"]), int(_photo["vectors"].shape[1])),
                 "vector_flops": meter.photo_flops()}


def photo_similar_upload(data: bytes, k: int = 12) -> tuple[int, dict]:
    try:
        img, w, h = open_image(data, 1024)
    except BadImage as e:
        return e.code, {"error": str(e)}
    _load("photo")
    if not _state["photo"]["loaded"]:
        return 500, {"error": _state["photo"]["error"] or "the photo model could not be loaded"}
    with _locks["photo"]:
        with meter.Span() as span, flops.Count() as count:
            v = _photo["model"].encode([img], batch_size=1, convert_to_numpy=True, show_progress_bar=False,
                                       normalize_embeddings=True)[0].astype(np.float32)
        secs = span.seconds
    hits, rank = _nearest(v, k)
    one = count.total or meter.photo_flops()            # counted on this photo; else the measured file's figure
    total = None if not one else one + meter.rank_flops(len(_photo["free"]), int(v.shape[0]))
    return 200, {"encode_seconds": round(secs, 4), "rank_seconds": round(rank, 4), "width": w, "height": h,
                 "photos": len(_photo["sigs"]), "photos_shown": int(len(_photo["free"])), "results": hits,
                 "offline": _evidence(), "work": span.figures(total)}


# ------------------------------------------------------------------ tile rendering
def _palette(iters: int) -> np.ndarray:
    """Ink only: points far from the set are dark, points near its edge light, the set itself darkest."""
    t = (np.arange(iters + 1) / iters) ** 0.42
    lo, hi = np.array([23, 40, 50.0]), np.array([233, 240, 243.0])
    lut = (lo[None, :] + (hi - lo)[None, :] * t[:, None]).round().astype(np.uint8)
    lut[iters] = (5, 13, 17)
    return lut


def _finite(x: float, default: float) -> float:
    x = float(x)
    return x if np.isfinite(x) else default


_last_render: dict = {}            # energy and operations of the picture just rendered (read under the render lock's caller)


def escape_counts(cx: float, cy: float, span: float, w: int, h: int, iters: int) -> tuple[np.ndarray, float]:
    """The view as escape counts, rendered in strips side by side. Every strip gets its rows of the one row
    grid of the whole picture, so the strips together are exactly the picture rendered in one piece."""
    from workers.jobs import render_view
    half_h = span * h / w / 2
    x0, x1 = cx - span / 2, cx + span / 2
    ys = np.linspace(cy - half_h, cy + half_h, h)
    rows = [(a, min(h, a + STRIP_ROWS)) for a in range(0, h, STRIP_ROWS)]
    with _locks["render"]:
        # numpy frees the interpreter lock inside its loops, so threads are enough here
        with meter.Span() as span, ThreadPoolExecutor(max_workers=RENDER_THREADS) as pool:
            parts = list(pool.map(lambda r: render_view(x0, x1, 0.0, 0.0, w, r[1] - r[0], iters, ys=ys[r[0]:r[1]]), rows))
    esc = np.vstack(parts)
    _last_render["work"] = span.figures(meter.render_flops(w * h, iters, int(esc.sum(dtype=np.int64))))
    return esc, span.seconds


def render(cx: float, cy: float, span: float, w: int, h: int, iters: int) -> tuple[bytes, dict]:
    """One view centred on (cx, cy), `span` wide in the complex plane, w x h pixels. Returns the PNG and facts."""
    from PIL import Image
    w, h = max(32, min(MAX_W, int(w))), max(32, min(MAX_H, int(h)))
    iters = max(20, min(MAX_ITERS, int(iters)))
    span = max(1e-12, min(6.0, _finite(span, 3.2)))
    cx, cy = max(-3.0, min(3.0, _finite(cx, -0.6))), max(-3.0, min(3.0, _finite(cy, 0.0)))
    with _render_gate:                                  # so the figures read back belong to this picture
        esc, secs = escape_counts(cx, cy, span, w, h, iters)
        work = dict(_last_render.get("work") or {})
    buf = BytesIO()
    Image.fromarray(_palette(iters)[esc]).save(buf, format="PNG", compress_level=1)
    return buf.getvalue(), {"seconds": round(secs, 4), "width": w, "height": h, "iters": iters, "pixels": w * h,
                            "tiles": round(w * h * iters / (TILE_PIXELS * TILE_ITERS), 2),
                            "cx": cx, "cy": cy, "span": span, "joules": work.get("joules"), "watts": work.get("watts"),
                            "flops": work.get("flops")}


# ------------------------------------------------------------------ saved examples
DEMO_TEXT = ["what does electricity cost", "how big is the water tank", "how long may a job wait",
             "price of a used server"]
DEMO_VIEWS = [("The whole set", -0.6, 0.0, 3.2), ("Seahorse valley", -0.745, 0.11, 0.02),
              ("A spiral at the edge", -0.7435, 0.1314, 0.0012)]


def write_demo() -> dict:
    """Results of a few fixed inputs, for a copy of the dashboard that cannot reach the local service. The
    thumbnails of the photos named in it are copied next to it (records without extra conditions of use)."""
    text = {q: embed_search(q, 5)[1] for q in DEMO_TEXT}
    for v in text.values():
        v.pop("offline", None)
    _load_index()
    starts = [h["signature"] for h in photo_sample(3, seed=7)[1]["results"]]
    photos = []
    thumbs = DEMO_OUT / "thumbs"
    if thumbs.exists():
        shutil.rmtree(thumbs)
    thumbs.mkdir(parents=True)
    (thumbs / ".gitkeep").write_text("", encoding="utf-8")
    for s in starts:
        r = photo_similar_signature(s, 6)[1]
        photos.append({"query": r["query"], "results": r["results"], "rank_seconds": r["rank_seconds"],
                       "compare_flops": r["compare_flops"], "vector_flops": r["vector_flops"]})
        for h in [r["query"], *r["results"]]:
            shutil.copyfile(THUMBS / f"{h['signature']}.jpg", thumbs / f"{h['signature']}.jpg")
    tiles = []
    (DEMO_OUT / "tiles").mkdir(parents=True, exist_ok=True)
    for i, (name, cx, cy, span) in enumerate(DEMO_VIEWS, 1):
        png, info = render(cx, cy, span, 720, 450, 300)
        (DEMO_OUT / "tiles" / f"view{i}.png").write_bytes(png)
        tiles.append({"name": name, "file": f"data/tiles/view{i}.png", **info})
    demo = {"text": text, "photos": photos, "photo_count": len(_photo["sigs"]), "photos_shown": int(len(_photo["free"])),
            "tiles": tiles}
    # the language model's answers to the page's example requests (skipped when the model is not on this machine)
    _load("llm")
    if _state["llm"]["loaded"]:
        from workers import jobs
        answers = []
        for x in json.loads(LLM_EXAMPLES.read_text(encoding="utf-8")):
            with meter.Span() as span, flops.Count() as count:
                out = jobs.write_text(x["prompt"])
            answers.append({"label": x["label"], "prompt": x["prompt"], **out, "work": llm_work(span, out, count)})
        demo["llm"] = {**_llm["facts"], "answers": answers}
    (DEMO_OUT / "tryjobs_demo.json").write_text(json.dumps(demo, ensure_ascii=False), encoding="utf-8")
    return demo


def write_describe_demo() -> dict | None:
    """Describe the photos of the saved descriptions again on this computer and record what each sentence took:
    time, energy and operations (web/public/data/describe_demo.json). The photos stay the same; a sentence
    changes only if this computer writes a different one. None when the description model or one of the
    thumbnails is not on this computer (the file is then left as it is)."""
    from PIL import Image

    from workers import jobs
    rows = json.loads(DESCRIBE_DEMO.read_text(encoding="utf-8"))
    files = [THUMBS / f"{r['signature']}.jpg" for r in rows]
    if not rows or not all(f.exists() for f in files):
        return None
    try:
        flops.watch(jobs.load_captioner()[1])
    except Exception as e:  # noqa: BLE001 - the model is not here: keep the recording
        print(_plain(e, "the description model"), file=sys.stderr)
        return None

    def photo(f):
        with Image.open(f) as im:
            return im.convert("RGB")
    jobs.caption_images([photo(files[0])])                     # the first call warms up; it is not recorded
    changed = 0
    for r, f in zip(rows, files):
        img = photo(f)
        with meter.Span() as span, flops.Count() as count:
            text = jobs.caption_images([img])[0]
        changed += text != r.get("description")
        r["description"] = text
        r["work"] = span.figures(count.total or None)
    DESCRIBE_DEMO.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    return {"photos": len(rows), "changed": changed}


def recount_demo() -> dict:
    """Renew in the saved answers what does not depend on the state of the computer, and keep what was recorded.
    Operation counts, token counts and the order of the closest entries depend only on the input, the model and
    the data, so they are worked out again here; the recorded times and energies stay as they are (a saved photo
    comparison that has no time yet gets one). An answer of the language model is recounted only if this computer
    writes the same text again (the count belongs to that text)."""
    demo = json.loads((DEMO_OUT / "tryjobs_demo.json").read_text(encoding="utf-8"))
    done = {"text": 0, "photos": 0, "llm": 0, "llm_different": 0}
    for q, v in demo.get("text", {}).items():
        new = embed_search(q, 5)[1]
        if v.get("work") and (new.get("work") or {}).get("flops"):
            v["work"]["flops"], v["tokens"], v["results"] = new["work"]["flops"], new["tokens"], new["results"]
            done["text"] += 1
    for p in demo.get("photos", []):
        code, r = photo_similar_signature(p["query"]["signature"], len(p["results"]))
        if code == 200:
            p.setdefault("rank_seconds", r["rank_seconds"])
            p.update(compare_flops=r["compare_flops"], vector_flops=r["vector_flops"])
            done["photos"] += 1
    answers = (demo.get("llm") or {}).get("answers") or []
    if answers:
        _load("llm")
    if answers and _state["llm"]["loaded"]:
        from workers import jobs
        for a in answers:
            with flops.Count() as count:
                out = jobs.write_text(a["prompt"])
            if out["text"] == a["text"] and a.get("work") and count.total:
                a["work"]["flops"] = float(count.total)
                done["llm"] += 1
            else:
                done["llm_different"] += 1
    (DEMO_OUT / "tryjobs_demo.json").write_text(json.dumps(demo, ensure_ascii=False), encoding="utf-8")
    return done


if __name__ == "__main__":
    if "--recount" in sys.argv:
        d = recount_demo()
        print(f"saved examples recounted: {d['text']} searches, {d['photos']} photo comparisons, {d['llm']} answers"
              + (f"; {d['llm_different']} answers left as they were (this computer writes a different text)" if d["llm_different"] else ""))
    if "--demo" in sys.argv:
        d = write_demo()
        print(f"saved examples: {len(d['text'])} searches, {len(d['photos'])} photo comparisons, {len(d['tiles'])} views "
              f"-> {DEMO_OUT.relative_to(ROOT)}")
    if "--demo" in sys.argv or "--describe" in sys.argv:
        s = write_describe_demo()
        print("saved descriptions: left as they are (the description model or the thumbnails are not on this computer)"
              if s is None else f"saved descriptions: {s['photos']} photos described again, {s['changed']} sentences changed")
