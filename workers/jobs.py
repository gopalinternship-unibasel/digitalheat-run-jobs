"""The three real job types. Each processes one chunk of independent items and returns its output.

embed   sentence embeddings with all-MiniLM-L6-v2 (sentence-transformers, CPU). One chunk =
        CHUNK["embed"] passages of 120 words taken from this project's own documents (document indexing).
render  Mandelbrot tiles with numpy (a stand-in for rendering work; needs no model and no network).
image   photo indexing for archive search: CLIP ViT-B/32 image embeddings (sentence-transformers, CPU)
        of open-licence Memobase photos (data/memobase, see memobase/fetch_manifest.py). One chunk =
        CHUNK["image"] photos read from the frozen list data/memobase/ids.txt. The vectors make the
        photos searchable by text in German, French, Italian and English (memobase/search.py).
caption photo descriptions: one English sentence per photo from Florence-2-base (florence-community
        conversion, MIT licence), computed on this machine only. The model is read from the local Hugging
        Face cache at a pinned revision, Hugging Face offline mode is on, and workers/no_network.py blocks
        every outside connection of the process, so a description can never come from a cloud service.
        Same frozen photo list as "image"; one chunk = CHUNK["caption"] photos.

Determinism: outputs are compared bit for bit after pause and resume, so the thread count is fixed
(DH_THREADS, default: all logical cores) and computation stays in float32; only the finished vectors
are stored as float16. Each chunk always has the same items in the same order (indices wrap around).

In the full dashboard the same functions are timed in batches and run by a worker that the heat scheduler pauses
and resumes; those parts are not in this project, which answers single requests (dh/tryjobs.py).
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
# Items per chunk (target: well under half a second). One caption takes longer than that on this CPU, so the
# caption job checks for a pause after every photo; that is its finest possible step.
CHUNK = {"embed": 32, "render": 4, "image": 8, "caption": 1}
UNIT = {"embed": "passages", "render": "tiles", "image": "photos", "caption": "photos"}
LABEL = {"embed": "Text embeddings (all-MiniLM-L6-v2)", "render": "Tile rendering (numpy)",
         "image": "Photo indexing (CLIP ViT-B/32, Memobase photos)",
         "caption": "Photo descriptions (Florence-2-base, computed locally, Memobase photos)"}
PASSAGE_WORDS = 120
# Torch threads for the jobs (set the environment variable DH_THREADS to change it). Outputs differ bit for bit
# between thread counts, so runs that are compared must use the same value.
THREADS = int(os.environ.get("DH_THREADS", "4"))
IMAGE_DIR = ROOT / "data" / "memobase" / "thumbs"
IMAGE_LIST = ROOT / "data" / "memobase" / "ids.txt"
EMBED_ID = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_REVISION = "c9745ed1d9f207416be6d2e6f8de32d1f16199bf"   # pinned version
CLIP_ID = "sentence-transformers/clip-ViT-B-32"
CLIP_REVISION = "327ab6726d33c0e22f920c83f2ff9e4bd38ca37f"   # pinned version
CAPTION_ID = "florence-community/Florence-2-base"           # 0.23 B parameters, MIT licence, 463 MB weights
CAPTION_REVISION = "00921df66db728a9ceb750f5eca43e5c203a2051"   # pinned version
CAPTION_TASK = "<CAPTION>"            # one sentence; "<DETAILED_CAPTION>" and "<MORE_DETAILED_CAPTION>" also work
CAPTION_MAX_TOKENS = 256
_model = None
_clip = None
_captioner = None
_writer = None
# The small language model of the "Text generation" panel (dashboard, Run jobs). It is not one of the job types of
# the heat schedule: it has no batch job there (dh/measure.py times it on the page's example requests). Model and
# version are fixed.
WRITER_ID = "Qwen/Qwen3-0.6B"                 # 0.6 B parameters, Apache-2.0 licence, 1.5 GB weights
WRITER_REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"   # pinned version
WRITER_PARAMS, WRITER_LICENCE = "0.6 billion", "Apache-2.0"
WRITER_SAMPLING = dict(do_sample=True, temperature=0.7, top_p=0.8, top_k=20)   # the model card's advice, thinking off
WRITER_SYSTEM = ("Answer briefly and plainly, in plain text without Markdown. Use at most four sentences unless you are "
                 "asked for a list or a poem.")
WRITER_MAX_TOKENS = 160
WRITER_SEED = 1                    # fixed: the same request gets the same answer at the same thread count
_passages: list[str] | None = None
_image_ids: list[str] | None = None


def passages() -> list[str]:
    """Passages of 120 words, cut from the project's Markdown documents (deterministic)."""
    global _passages
    if _passages is None:
        words: list[str] = []
        for name in ("SOURCES.md", "FRAMEWORK.md", "CASE_BRIEF.md"):
            words += (ROOT / name).read_text(encoding="utf-8").split()
        step = PASSAGE_WORDS // 2                       # overlapping windows, as in document indexing
        _passages = [" ".join(words[i:i + PASSAGE_WORDS]) for i in range(0, len(words) - PASSAGE_WORDS, step)]
    return _passages


def load_embedder():
    global _model
    if _model is None:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")    # the model is cached; never touch the network
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        preload_pyarrow()
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(EMBED_ID, device="cpu", revision=EMBED_REVISION)
    return _model


def embed_chunk(index: int) -> np.ndarray:
    """Embed chunk number `index` (CHUNK['embed'] passages). Returns float16 vectors."""
    ps = passages()
    n = CHUNK["embed"]
    batch = [ps[(index * n + k) % len(ps)] for k in range(n)]
    vec = load_embedder().encode(batch, batch_size=n, convert_to_numpy=True, show_progress_bar=False,
                                 normalize_embeddings=True)
    return vec.astype(np.float16)


def render_chunk(index: int, size: int = 320, iters: int = 200) -> np.ndarray:
    """Render chunk number `index` (CHUNK['render'] tiles of size x size/8 pixels)."""
    out = []
    for k in range(CHUNK["render"]):
        i = index * CHUNK["render"] + k
        y0 = -1.5 + 3.0 * (i % 400) / 400
        y = np.linspace(y0, y0 + 3.0 / 400, size // 8)
        x = np.linspace(-2.0, 1.0, size)
        c = x[None, :] + 1j * y[:, None]
        z = np.zeros_like(c)
        esc = np.zeros(c.shape, dtype=np.int16)
        for _ in range(iters):
            m = np.abs(z) <= 2
            z[m] = z[m] ** 2 + c[m]
            esc[m] += 1
        out.append(esc)
    return np.stack(out)


def render_view(x0: float, x1: float, y0: float, y1: float, width: int, height: int, iters: int = 200,
                ys: np.ndarray | None = None) -> np.ndarray:
    """Escape counts for one free rectangle of the Mandelbrot set, with the arithmetic of render_chunk (which
    renders fixed strips of it). Used by the dashboard's live tile view. Returns int16, shape (height, width).
    `ys` gives the row coordinates directly (then y0 and y1 are not used): a picture cut into strips passes each
    strip its rows of the whole picture, so the strips match the picture rendered in one piece bit for bit."""
    x = np.linspace(x0, x1, width)
    y = np.linspace(y0, y1, height) if ys is None else np.asarray(ys, dtype=float)
    c = x[None, :] + 1j * y[:, None]
    z = np.zeros_like(c)
    esc = np.zeros(c.shape, dtype=np.int16)
    for _ in range(iters):
        m = np.abs(z) <= 2
        z[m] = z[m] ** 2 + c[m]
        esc[m] += 1
    return esc


def image_ids() -> list[str]:
    """The frozen work list (signatures in manifest order), written by memobase/download_thumbs.py."""
    global _image_ids
    if _image_ids is None:
        _image_ids = IMAGE_LIST.read_text(encoding="utf-8").split()
    return _image_ids


def n_items(kind: str) -> int | None:
    """Distinct items behind a job type (None when the items are generated)."""
    return {"image": len(image_ids()), "caption": len(image_ids()), "embed": len(passages())}.get(kind)


def preload_pyarrow() -> None:
    """Load pyarrow before sentence-transformers. On one Windows setup a worker started by another Python
    process crashed inside pyarrow's native library when sentence-transformers pulled it in late, through pandas;
    loading pyarrow first avoids it. pyarrow is optional: nothing happens when it is not installed."""
    try:
        import pyarrow  # noqa: F401
    except Exception:  # noqa: BLE001 - pyarrow is optional for these jobs
        pass


def load_clip():
    global _clip
    if _clip is None:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")    # the model is cached; never touch the network
        preload_pyarrow()
        import torch
        torch.set_num_threads(THREADS)
        from sentence_transformers import SentenceTransformer
        _clip = SentenceTransformer(CLIP_ID, device="cpu", revision=CLIP_REVISION)
        _clip.eval()
    return _clip


def image_chunk(index: int) -> np.ndarray:
    """Embed photos CHUNK['image']*index .. +CHUNK['image']-1 of the work list (wrapping).
    Returns float16 vectors, shape (CHUNK['image'], 512), L2-normalised."""
    from PIL import Image
    ids = image_ids()
    n = CHUNK["image"]
    imgs = []
    for k in range(n):
        with Image.open(IMAGE_DIR / f"{ids[(index * n + k) % len(ids)]}.jpg") as im:
            imgs.append(im.convert("RGB"))       # most archive photos are greyscale (mode L); CLIP needs RGB
    vec = load_clip().encode(imgs, batch_size=n, convert_to_numpy=True, show_progress_bar=False,
                             normalize_embeddings=True)
    return vec.astype(np.float16)


def caption_snapshot() -> Path:
    """Local folder of the pinned Florence-2-base snapshot in the Hugging Face cache (no network)."""
    from huggingface_hub import constants
    snap = (Path(constants.HF_HUB_CACHE) / ("models--" + CAPTION_ID.replace("/", "--")) / "snapshots"
            / CAPTION_REVISION)
    if not (snap / "model.safetensors").exists():
        raise FileNotFoundError(
            f"{CAPTION_ID} @ {CAPTION_REVISION[:12]} is not in the local cache ({snap}). Download it once with "
            f"huggingface_hub.snapshot_download('{CAPTION_ID}', revision='{CAPTION_REVISION}'); "
            "the caption job itself never downloads anything.")
    return snap


def ensure_offline() -> None:
    """Switch on the network guard (idempotent) and refuse to go on unless it is in place and the Hugging Face
    libraries that are already imported really run in offline mode."""
    from workers import no_network
    no_network.install()
    if not no_network.installed():
        raise RuntimeError("the network guard is not installed; captions must be computed offline")
    off = no_network.library_offline_flags()
    if any(v is False for v in off.values()):
        raise RuntimeError(f"Hugging Face offline mode is not in effect ({off}); captions must be computed offline")


def load_captioner():
    """Florence-2-base and its processor, loaded from the local snapshot folder with every outside network
    connection blocked (workers/no_network.py). Computation in float32 on the CPU."""
    global _captioner
    ensure_offline()
    if _captioner is None:
        preload_pyarrow()
        import torch
        torch.set_num_threads(THREADS)
        from transformers import AutoProcessor, Florence2ForConditionalGeneration
        ensure_offline()                              # transformers is imported now: check its offline flag too
        snap = caption_snapshot()
        # use_fast=False pins the image processor the snapshot was saved with (transformers warns that its
        # default changes to the fast one, which gives slightly different pixels and so different captions)
        processor = AutoProcessor.from_pretrained(snap, local_files_only=True, use_fast=False)
        model = Florence2ForConditionalGeneration.from_pretrained(snap, local_files_only=True, dtype=torch.float32)
        model.eval()
        _captioner = (processor, model)
    return _captioner


class Interrupted(Exception):
    """Raised inside a chunk when the scheduler asks the job to pause or stop before the chunk is finished.
    The worker discards the unfinished chunk and computes it again from scratch after the resume, so the
    output is the same as without the interruption; at most one unfinished photo's work is lost."""


# Blocks of Florence-2 before which the interrupt check runs. The image encoder (DaViT) takes about 70 to 75 %
# of a caption on the team laptop, so checking only
# between generated words left pauses waiting up to 4.6 s in a replayed day of the full dashboard. Checking before
# every vision block, text-encoder layer and decoder layer leaves at most about 0.41 s between two checks.
_CHECK_BEFORE = ("Florence2VisionConvEmbed", "Florence2VisionSpatialBlock", "Florence2VisionChannelBlock",
                 "BartEncoderLayer", "BartDecoderLayer")


def _install_interrupt_hooks(model, interrupt) -> list:
    """Forward pre-hooks that raise Interrupted before the next block when interrupt() returns True
    (control file read at most every 50 ms). Returns the hook handles; remove them after the call."""
    import time as _time
    last = [0.0]

    def check(_module, _args):
        now = _time.perf_counter()
        if now - last[0] >= 0.05:
            last[0] = now
            if interrupt():
                raise Interrupted()

    return [m.register_forward_pre_hook(check) for m in model.modules() if type(m).__name__ in _CHECK_BEFORE]


def caption_images(imgs: list, task: str = CAPTION_TASK, interrupt=None) -> list[str]:
    """Describe each PIL image (RGB) in English. Beam search (3 beams, from the model's generation config),
    no sampling, so the same photo always gets the same description at the same thread count.
    interrupt: optional callable; when it returns True during decoding, Interrupted is raised."""
    import torch
    ensure_offline()
    processor, model = load_captioner()
    inputs = processor(text=[task] * len(imgs), images=imgs, return_tensors="pt")
    if interrupt is not None and interrupt():
        raise Interrupted()
    hooks = _install_interrupt_hooks(model, interrupt) if interrupt is not None else []
    try:
        with torch.inference_mode():
            ids = model.generate(**inputs, max_new_tokens=CAPTION_MAX_TOKENS, num_beams=3, do_sample=False)
    finally:
        for h in hooks:
            h.remove()
    texts = processor.batch_decode(ids, skip_special_tokens=False)
    out = []
    for text, im in zip(texts, imgs):
        parsed = processor.post_process_generation(text, task=task, image_size=im.size)
        out.append(str(parsed[task]).strip())
    return out


def caption_chunk(index: int, interrupt=None) -> np.ndarray:
    """Describe photos CHUNK['caption']*index .. +CHUNK['caption']-1 of the work list (wrapping).
    Returns a numpy unicode array of the descriptions (saved as .npy without pickle).
    interrupt: see caption_images; the worker passes "has the scheduler said pause or stop?"."""
    from PIL import Image
    ids = image_ids()
    n = CHUNK["caption"]
    imgs = []
    for k in range(n):
        with Image.open(IMAGE_DIR / f"{ids[(index * n + k) % len(ids)]}.jpg") as im:
            imgs.append(im.convert("RGB"))
    return np.array(caption_images(imgs, interrupt=interrupt), dtype=str)


def load_writer():
    """The small language model and its tokenizer, loaded from the local snapshot folder with every outside
    network connection blocked (workers/no_network.py). Computation in float32 on the CPU."""
    global _writer
    ensure_offline()
    if _writer is None:
        import torch
        torch.set_num_threads(THREADS)
        from huggingface_hub import snapshot_download
        from transformers import AutoModelForCausalLM, AutoTokenizer
        ensure_offline()                              # transformers is imported now: check its offline flag too
        snap = snapshot_download(WRITER_ID, revision=WRITER_REVISION, local_files_only=True)
        tok = AutoTokenizer.from_pretrained(snap, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(snap, local_files_only=True, dtype=torch.float32)
        model.eval()
        _writer = (tok, model)
    return _writer


def next_piece(state: dict, whole: str) -> str:
    """The part of the answer that is new, given all of it decoded so far. Blanks before the answer are left out;
    half a character at the end (an unfinished UTF-8 sequence) waits for the rest. state keeps "lead" and "sent"."""
    if state.get("lead") is None:
        if not whole.strip():
            return ""
        state["lead"] = len(whole) - len(whole.lstrip())
    whole = whole[state["lead"]:]
    sent = state.get("sent", "")
    if whole.endswith("\ufffd") or len(whole) <= len(sent):
        return ""
    state["sent"] = whole
    return whole[len(sent):]


def write_text(prompt: str, max_new_tokens: int = WRITER_MAX_TOKENS, seed: int = WRITER_SEED, on_text=None, stop=None) -> dict:
    """Answer one request with the small language model. on_text(piece) is called with each new piece of the
    answer as soon as it is written; stop() returning True ends the answer after the current token.
    Returns the answer and how long it took: text, tokens (of the answer itself, without the end marker),
    prompt_tokens, first_token_s, writing_s (from the first to the last token), tokens_per_s (the tokens after
    the first, divided by writing_s), seconds (the whole call), stopped, hit_limit. Sampling with a fixed seed,
    so the same request gives the same answer."""
    import time as _time

    import torch
    from transformers import StoppingCriteria, StoppingCriteriaList
    from transformers.generation.streamers import BaseStreamer
    ensure_offline()
    tok, model = load_writer()
    msgs = [{"role": "system", "content": WRITER_SYSTEM}, {"role": "user", "content": prompt}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    enc = tok(text, return_tensors="pt")
    state = {"ids": [], "sent": "", "lead": None, "times": [], "prompt": True, "stopped": False}

    class Pieces(BaseStreamer):
        def put(self, value):
            if state["prompt"]:                       # the first call hands over the request itself
                state["prompt"] = False
                return
            state["times"].append(_time.perf_counter())
            state["ids"].extend(value.flatten().tolist())
            piece = next_piece(state, tok.decode(state["ids"], skip_special_tokens=True))
            if piece and on_text is not None:
                on_text(piece)

        def end(self):
            pass

    class Halt(StoppingCriteria):
        def __call__(self, input_ids, scores, **kwargs):
            if stop is not None and stop():
                state["stopped"] = True
                return True
            return False

    torch.manual_seed(seed)
    t0 = _time.perf_counter()
    with torch.inference_mode():
        model.generate(**enc, max_new_tokens=max(1, int(max_new_tokens)), streamer=Pieces(),
                       stopping_criteria=StoppingCriteriaList([Halt()]), **WRITER_SAMPLING)
    secs = _time.perf_counter() - t0
    special = set(tok.all_special_ids)
    ended = bool(state["ids"]) and state["ids"][-1] in special
    own = [t for i, t in zip(state["ids"], state["times"]) if i not in special]     # when each token of the answer came
    n = len(own)
    writing = own[-1] - own[0] if n > 1 else 0.0
    return {"text": tok.decode(state["ids"], skip_special_tokens=True).strip(), "tokens": n,
            "prompt_tokens": int(enc["input_ids"].shape[1]), "first_token_s": round(own[0] - t0, 3) if n else None,
            "writing_s": round(writing, 2), "tokens_per_s": round((n - 1) / writing, 1) if writing > 0 else None,
            "seconds": round(secs, 2), "stopped": state["stopped"], "produced": len(state["ids"]),
            "hit_limit": len(state["ids"]) >= int(max_new_tokens) and not ended and not state["stopped"]}


INTERRUPTIBLE = {"caption"}     # job types whose chunks can be abandoned midway (see Interrupted)


PROCESS = {"embed": embed_chunk, "render": render_chunk, "image": image_chunk, "caption": caption_chunk}


def warm_up(kind: str) -> None:
    if kind == "embed":
        load_embedder()
    if kind == "image":
        load_clip()
    if kind == "caption":
        load_captioner()
    PROCESS[kind](0)
