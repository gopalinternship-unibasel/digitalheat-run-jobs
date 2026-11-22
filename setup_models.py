"""Download the four models once into the local Hugging Face cache. The service itself never downloads anything:
it blocks outside connections and reads the models from this cache.

    python setup_models.py

About 2.7 GB in total. Run it again at any time: a model that is already in the cache is left as it is.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from huggingface_hub import constants, snapshot_download  # noqa: E402

from workers.jobs import (  # noqa: E402
    CAPTION_ID, CAPTION_REVISION, CLIP_ID, CLIP_REVISION, EMBED_ID, EMBED_REVISION, WRITER_ID, WRITER_REVISION,
)

MODELS = [
    ("text embeddings", EMBED_ID, EMBED_REVISION),
    ("photo vectors", CLIP_ID, CLIP_REVISION),
    ("photo descriptions", CAPTION_ID, CAPTION_REVISION),
    ("text generation", WRITER_ID, WRITER_REVISION),
]
# weights in other formats than the one used here are left out
SKIP = ["*.onnx", "onnx/*", "openvino/*", "*.h5", "*.ot", "*.msgpack", "tf_model*", "rust_model*", "flax_model*", "*.bin"]


def cached(repo: str, revision: str) -> Path | None:
    """The folder of this exact version in the cache, if its weights and configuration are there."""
    snap = Path(constants.HF_HUB_CACHE) / ("models--" + repo.replace("/", "--")) / "snapshots" / revision
    if snap.is_dir() and any(snap.rglob("model.safetensors")) and any(snap.rglob("config.json")):
        return snap
    return None


def main() -> int:
    failed = 0
    for what, repo, revision in MODELS:
        t0 = time.perf_counter()
        path = cached(repo, revision)
        if path is not None:
            print(f"{what}: {repo} is already in the cache", flush=True)
            continue
        try:
            path = Path(snapshot_download(repo, revision=revision, ignore_patterns=SKIP, max_workers=2))
        except OSError as e:
            failed += 1
            print(f"{what}: {repo} could not be stored ({type(e).__name__}: {e}).", flush=True)
            if getattr(e, "winerror", None) == 1314:
                print("  Windows refused to create a link in the model cache. Turn on Developer Mode (Settings > System > "
                      "For developers) or run this script once from an administrator terminal, then run it again.", flush=True)
            continue
        size = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
        print(f"{what}: {repo} ready ({size / 1e6:.0f} MB, {time.perf_counter() - t0:.0f} s)", flush=True)
    if failed:
        print(f"{failed} of {len(MODELS)} models are missing; their panels will say so until this script has run through.")
        return 1
    print("All four models are in the local cache. Start the service with: python -m dh.serve")
    return 0


if __name__ == "__main__":
    sys.exit(main())
