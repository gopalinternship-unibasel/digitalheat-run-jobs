"""Download the 640x480 thumbnails of the photos in data/memobase/manifest.csv and freeze the work list.

Every file is decoded with Pillow before it is kept: a failed resize on media.memobase.ch comes back as
HTTP 200 with an error text labelled image/jpeg, so the Content-Type header cannot be trusted.
Files that already exist and decode are not fetched again.

Writes data/memobase/thumbs/<signature>.jpg, data/memobase/thumbs_index.csv (size, pixels, mode, sha256
per file) and data/memobase/ids.txt: the frozen, ordered work list of the image job (manifest order,
only photos whose thumbnail decoded). The job reads ids.txt, never a directory listing.

The thumbnails are working copies for the demo. Do not put them in the submission or a public
repository: ship manifest.csv and this script instead (see data/memobase/README.md).

    python memobase/download_thumbs.py
"""
from __future__ import annotations

import csv
import hashlib
import io
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "memobase"
THUMBS = DATA / "thumbs"
UA = {"User-Agent": "DigitalHeat hackathon demo (sprintd Basel 2026; small one-off fetch)"}
WORKERS = 4                       # parallel connections; no rate limit is documented, stay modest


def decode_ok(data: bytes) -> tuple[bool, str, tuple[int, int]]:
    try:
        with Image.open(io.BytesIO(data)) as im:
            im.load()
            return True, im.mode, im.size
    except Exception as e:  # noqa: BLE001 - any decode failure means the file is unusable
        return False, f"{type(e).__name__}: {e}"[:120], (0, 0)


def fetch(row: dict) -> dict:
    sig, url = row["signature"], row["thumb_url"]
    path = THUMBS / f"{sig}.jpg"
    if path.exists():
        data = path.read_bytes()
        ok, mode, size = decode_ok(data)
        if ok:
            return {"signature": sig, "ok": True, "bytes": len(data), "mode": mode, "w": size[0], "h": size[1],
                    "sha256": hashlib.sha256(data).hexdigest(), "note": "cached"}
    last = ""
    for attempt in range(3):
        try:
            r = requests.get(url, headers=UA, timeout=60)
            if r.status_code == 200:
                ok, mode, size = decode_ok(r.content)
                if ok:
                    tmp = path.with_suffix(".part")
                    tmp.write_bytes(r.content)
                    tmp.replace(path)
                    return {"signature": sig, "ok": True, "bytes": len(r.content), "mode": mode, "w": size[0],
                            "h": size[1], "sha256": hashlib.sha256(r.content).hexdigest(), "note": ""}
                last = f"not an image ({mode}); first bytes {r.content[:40]!r}"
                break                                    # an error page will not get better on retry
            last = f"HTTP {r.status_code}"
        except requests.RequestException as e:
            last = f"{type(e).__name__}"
        time.sleep(1 + attempt)
    return {"signature": sig, "ok": False, "bytes": 0, "mode": "", "w": 0, "h": 0, "sha256": "", "note": last}


def main() -> None:
    THUMBS.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader((DATA / "manifest.csv").open(encoding="utf-8")))
    t0 = time.perf_counter()
    with ThreadPoolExecutor(WORKERS) as ex:
        results = list(ex.map(fetch, rows))      # map keeps manifest order
    dt = time.perf_counter() - t0
    with (DATA / "thumbs_index.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(results)
    good = [r["signature"] for r in results if r["ok"]]
    bad = [r for r in results if not r["ok"]]
    (DATA / "ids.txt").write_text("\n".join(good) + "\n", encoding="utf-8")
    modes = {}
    for r in results:
        if r["ok"]:
            modes[r["mode"]] = modes.get(r["mode"], 0) + 1
    print(f"{len(good)} of {len(rows)} thumbnails usable in {dt:.1f} s ({WORKERS} connections); "
          f"{sum(r['bytes'] for r in results) / 1e6:.1f} MB; colour modes {modes}")
    for r in bad[:10]:
        print("  failed:", r["signature"], r["note"])
    if bad:
        print(f"  {len(bad)} failed in total (listed in thumbs_index.csv, left out of ids.txt)")


if __name__ == "__main__":
    sys.exit(main())
