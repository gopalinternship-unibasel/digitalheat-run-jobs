"""Search the Memobase photos indexed by the image job, by text in German, French, Italian or English.

The photo vectors are the output of the photo-indexing job (workers/jobs.py "image"); this project ships them
ready-made in data/memobase/index_clip_b32.npz, and only load_index, manifest and credit are used here.
Queries are encoded with sentence-transformers/clip-ViT-B-32-multilingual-v1, a text encoder trained to
map 50+ languages into the same vector space as the CLIP ViT-B/32 image encoder. Similarity is the cosine
of the two vectors. Zero-shot tags work the same way, with a fixed label list.

    python memobase/search.py --build                    # collect the job output into the index file
    python memobase/search.py "Schnee in den Bergen"     # top 12 photos with their credit lines
    python memobase/search.py "gare" --k 5
    python memobase/search.py --tags                     # automatic tags for every photo -> tags.csv
    python memobase/search.py --export-demo              # results for the demo queries -> search_demo.json
    python memobase/search.py --serve                    # JSON search API on http://127.0.0.1:8765

Automatic tags and rankings come from a general model and are often wrong for archive photos
(local names, dates, rare objects); they are a search aid, not catalogue data.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
DATA = ROOT / "data" / "memobase"
INDEX = DATA / "index_clip_b32.npz"
JOB_OUT = ROOT / "runs" / "live" / "image_live" / "out"
TEXT_ID = "sentence-transformers/clip-ViT-B-32-multilingual-v1"
TEXT_REVISION = "58edf8cada9e398793dca955574a48cbb7f18be2"   # pinned version

DEMO_QUERIES = ["Schnee in den Bergen", "Bahnhof mit Zug", "Kühe auf der Weide", "Brücke über einen Fluss",
                "Fabrikhalle mit Arbeitern", "Umzug durch die Altstadt", "lac et montagnes", "église de village",
                "ponte sul fiume", "people skiing"]
TAGS = {  # label shown -> text the model compares with (German; the encoder is multilingual)
    "Berge": "ein Foto von Bergen", "See": "ein Foto von einem See", "Schnee": "ein Foto mit Schnee",
    "Wald": "ein Foto von einem Wald", "Dorf": "ein Foto von einem Dorf", "Stadt": "ein Foto von einer Stadtstrasse",
    "Kirche": "ein Foto von einer Kirche", "Brücke": "ein Foto von einer Brücke",
    "Eisenbahn": "ein Foto von einem Zug oder Bahnhof", "Auto": "ein Foto von einem Auto",
    "Schiff": "ein Foto von einem Schiff", "Flugzeug": "ein Foto von einem Flugzeug",
    "Bauernhof": "ein Foto von einem Bauernhof", "Tiere": "ein Foto von Kühen oder Pferden",
    "Fabrik": "ein Foto von einer Fabrik oder Werkstatt", "Fest": "ein Foto von einem Fest oder Umzug",
    "Gruppe": "ein Gruppenfoto von Menschen", "Porträt": "ein Porträt einer Person",
    "Innenraum": "ein Foto von einem Innenraum", "Dokument": "ein Foto von einem Dokument oder Plan",
}
_text_model = None


def manifest() -> dict[str, dict]:
    with (DATA / "manifest.csv").open(encoding="utf-8") as f:
        return {r["signature"]: r for r in csv.DictReader(f)}


def build_index(out_dir: Path = JOB_OUT) -> None:
    """Collect the job's chunk files into one index (photo i of chunk c is item c*CHUNK+i of ids.txt)."""
    from workers.jobs import CHUNK, CLIP_ID, CLIP_REVISION, THREADS, image_ids
    ids = image_ids()
    n = CHUNK["image"]
    files = sorted(out_dir.glob("chunk_*.npy"))
    files = [f for f in files if ".tmp" not in f.name]
    if len(files) * n < len(ids):
        raise SystemExit(f"only {len(files)} chunks in {out_dir}; the batch run that writes them is part of the full dashboard")
    vec = np.zeros((len(ids), 512), dtype=np.float16)
    seen = np.zeros(len(ids), dtype=bool)
    for f in files:
        c = int(f.stem.split("_")[1])
        arr = np.load(f)
        for k in range(n):
            j = (c * n + k) % len(ids)
            if not seen[j]:
                vec[j], seen[j] = arr[k], True
    if not seen.all():
        raise SystemExit(f"{(~seen).sum()} photos have no vector")
    np.savez_compressed(INDEX, vectors=vec, signatures=np.array(ids),
                        meta=np.array(json.dumps({"model": CLIP_ID, "revision": CLIP_REVISION, "threads": THREADS,
                                                  "source": str(out_dir.relative_to(ROOT)), "chunk": n})))
    print(f"index: {len(ids)} photos x 512 -> {INDEX.relative_to(ROOT)}")


def load_index() -> tuple[np.ndarray, list[str]]:
    z = np.load(INDEX)
    return z["vectors"].astype(np.float32), [str(s) for s in z["signatures"]]


def text_model():
    global _text_model
    if _text_model is None:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        from workers.jobs import preload_pyarrow
        preload_pyarrow()
        from huggingface_hub import snapshot_download
        from sentence_transformers import SentenceTransformer
        # Load from the local snapshot folder: with a repo name, transformers 4.57.3 asks the Hub whether
        # the tokenizer is a Mistral one and fails in offline mode.
        path = snapshot_download(TEXT_ID, revision=TEXT_REVISION, local_files_only=True)
        _text_model = SentenceTransformer(path, device="cpu")
    return _text_model


def encode(texts: list[str]) -> np.ndarray:
    return text_model().encode(texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)


def credit(r: dict) -> str:
    who = r["creator"] or r["exif_copyright"] or "unknown creator"
    return (f"{r['title']} ({r['date'] or 'undated'}), {who}; {r['institution']}; via Memobase / Memoriav; "
            f"{r['licence']}; {r['object_url']}")


def search(query: str, k: int = 12, only_without_conditions: bool = False) -> list[dict]:
    vec, sigs = load_index()
    meta = manifest()
    q = encode([query])[0]
    scores = vec @ q
    out = []
    for j in np.argsort(-scores):
        r = meta[sigs[j]]
        if only_without_conditions and r["extra_conditions"] == "yes":
            continue
        out.append({"score": round(float(scores[j]), 4), "signature": sigs[j], "record_id": r["record_id"],
                    "title": r["title"], "date": r["date"], "creator": r["creator"] or r["exif_copyright"],
                    "institution": r["institution"], "licence": r["licence"], "licence_url": r["licence_url"],
                    "object_url": r["object_url"], "extra_conditions": r["extra_conditions"],
                    "thumb": f"thumbs/{sigs[j]}.jpg", "credit": credit(r)})
        if len(out) >= k:
            break
    return out


def tag_all(top: int = 3) -> None:
    vec, sigs = load_index()
    labels = list(TAGS)
    lab = encode([TAGS[t] for t in labels])
    sim = vec @ lab.T
    with (DATA / "tags.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["signature"] + [f"tag{i + 1}" for i in range(top)] + [f"score{i + 1}" for i in range(top)])
        for j, s in enumerate(sigs):
            best = np.argsort(-sim[j])[:top]
            w.writerow([s] + [labels[b] for b in best] + [f"{sim[j, b]:.4f}" for b in best])
    counts = {}
    for j in range(len(sigs)):
        t = labels[int(np.argmax(sim[j]))]
        counts[t] = counts.get(t, 0) + 1
    print(f"tags for {len(sigs)} photos -> {(DATA / 'tags.csv').relative_to(ROOT)}; top tag counts:",
          dict(sorted(counts.items(), key=lambda x: -x[1])))


def export_demo(k: int = 8) -> None:
    res = {q: search(q, k) for q in DEMO_QUERIES}
    (DATA / "search_demo.json").write_text(json.dumps({
        "note": "Top results of the demo queries over the photos indexed by the scheduled image job. "
                "Rankings come from a general model and can be wrong.",
        "text_model": TEXT_ID, "results": res}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{len(res)} queries -> {(DATA / 'search_demo.json').relative_to(ROOT)}")


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802 - http.server API
        u = urlparse(self.path)
        if u.path == "/search":
            qs = parse_qs(u.query)
            q = (qs.get("q") or [""])[0]
            k = int((qs.get("k") or ["12"])[0])
            body = json.dumps({"query": q, "results": search(q, k) if q else []}, ensure_ascii=False).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
        elif u.path.startswith("/thumbs/") and u.path.endswith(".jpg") and "/" not in u.path[8:]:
            f = DATA / "thumbs" / u.path[8:]
            if not f.exists():
                self.send_error(404)
                return
            body = f.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
        else:
            self.send_error(404)
            return
        self.send_header("Access-Control-Allow-Origin", "*")    # the dashboard runs on another local port
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("query", nargs="?")
    ap.add_argument("--k", type=int, default=12)
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--tags", action="store_true")
    ap.add_argument("--export-demo", action="store_true")
    ap.add_argument("--serve", action="store_true")
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()
    if args.build:
        build_index()
    if args.tags:
        tag_all()
    if args.export_demo:
        export_demo()
    if args.query:
        for i, r in enumerate(search(args.query, args.k), 1):
            print(f"{i:2d}. {r['score']:.3f}  {r['credit']}")
    if args.serve:
        text_model()
        load_index()
        print(f"search API on http://127.0.0.1:{args.port}/search?q=...  (thumbnails under /thumbs/)")
        ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
