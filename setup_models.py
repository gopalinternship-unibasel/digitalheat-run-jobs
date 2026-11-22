"""Download the four models once into the local Hugging Face cache. The service itself never downloads anything:
it blocks outside connections and reads the models from this cache.

    python setup_models.py

About 2.7 GB in total. Run it again at any time: files that are already there are not fetched again.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from huggingface_hub import snapshot_download  # noqa: E402

from workers.jobs import CAPTION_ID, CAPTION_REVISION, CLIP_ID, CLIP_REVISION, WRITER_ID, WRITER_REVISION  # noqa: E402

MODELS = [
    ("text embeddings", "sentence-transformers/all-MiniLM-L6-v2", None),
    ("photo vectors", CLIP_ID, CLIP_REVISION),
    ("photo descriptions", CAPTION_ID, CAPTION_REVISION),
    ("text generation", WRITER_ID, WRITER_REVISION),
]
# weights in other formats than the one used here are left out
SKIP = ["*.onnx", "onnx/*", "openvino/*", "*.h5", "*.ot", "*.msgpack", "tf_model*", "rust_model*", "flax_model*", "*.bin"]


def main() -> None:
    for what, repo, revision in MODELS:
        t0 = time.perf_counter()
        path = Path(snapshot_download(repo, revision=revision, ignore_patterns=SKIP))
        size = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
        print(f"{what}: {repo} ready ({size / 1e6:.0f} MB, {time.perf_counter() - t0:.0f} s)", flush=True)
    print("All four models are in the local cache. Start the service with: python -m dh.serve")


if __name__ == "__main__":
    main()
