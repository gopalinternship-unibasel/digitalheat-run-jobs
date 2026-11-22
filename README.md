# Run jobs: five kinds of computing work, answered live

This is the "Run jobs" page of the DigitalHeat dashboard (team Cordon Vert, sprintd Basel GreenTech) as a project of
its own. The case: servers in a house do paid computing work and their waste heat warms the house. The page lets you
run that kind of work yourself, on your own computer, and shows what each answer cost.

| Panel | What you do | What runs |
|---|---|---|
| Text embeddings | Type a sentence; the entries of an assumptions table closest in meaning follow your typing | all-MiniLM-L6-v2 |
| Photo indexing | Pick an archive photo or give your own; the closest archive photos appear | CLIP ViT-B/32 |
| Tile rendering | Drag and zoom a view of the Mandelbrot set | plain arithmetic (numpy) |
| Photo descriptions | Give a photo; one English sentence describes it | Florence-2-base |
| Text generation | Write a request; the answer appears while it is being written | Qwen3-0.6B |

Above the panels a strip shows the processor's power over the last minute, and under each answer you see its
processor energy and the arithmetic operations (FLOPs) it needed. The energy reading works on Windows only (it uses
the Windows energy meter interface); on other systems the strip is hidden and the answers show time and operations.

Everything is computed on your computer. The service listens on 127.0.0.1 only and blocks outside connections of its
own process; the models are read from the local Hugging Face cache.

## What you need

- Python 3.10 or newer (built with 3.13) and Node.js 20.19 or newer, or 22.12 or newer (built with 24).
- About 4 GB of free disk space for the models and about 3 GB of free memory.
- No graphics card: everything runs on the processor.

## Set it up (once)

On macOS and Linux the command is often `python3`; a virtual environment gives you `python` and `pip`
(`python3 -m venv .venv`, then activate it).

```bash
pip install -r requirements.txt
python setup_models.py                 # downloads the four models (about 2.7 GB) into the Hugging Face cache
python memobase/download_thumbs.py     # downloads the archive thumbnails (about 120 MB) from Memobase
cd web
npm install
```

The thumbnails are not in this repository on purpose (see "Photos" below). Without them the photo panel still
compares photos, but shows no pictures. The download script rewrites `data/memobase/ids.txt` with the photos it
fetched; the service does not read that file, so a changed `ids.txt` in `git status` is harmless.

## Run it

Two terminals:

```bash
python -m dh.serve          # the local service, http://127.0.0.1:8765
```

```bash
cd web
npm run dev                 # the page, http://127.0.0.1:5420/run
```

Open `http://127.0.0.1:5420/run`. Each model loads the first time its panel is opened (a few seconds each).

If the service is not running, the page still works: every panel then shows saved answers
(`web/public/data/tryjobs_demo.json`, `describe_demo.json`). `python -m dh.tryjobs --demo`, run after the
thumbnails are downloaded, rewrites `tryjobs_demo.json` and the three tile pictures with your own computer's figures;
`describe_demo.json` is a fixed recording.

## Check it

```bash
python -m pytest -q         # needs pytest; tests that need a model or the thumbnails skip themselves when those are missing
cd web && npm run build     # type check and production build
```

## Put it into another app

- **The page** is one React component, `RunView` in `web/src/views/Run.tsx` (the photo-description panel is
  `web/src/views/Describe.tsx`). It needs `web/src/data.ts`, `format.ts`, `styles.css`, `components/ui.tsx`,
  `components/DataTable.tsx`, `llm_examples.json` and the three style sheets next to it, plus TanStack Router (the
  panel is chosen with `?job=embed|image|render|caption|llm`), TanStack Query and TanStack Table.
- **The service** is plain Python (`http.server`, no web framework): `dh/serve.py` holds the routes and is documented
  at its top; `dh/tryjobs.py` holds the live use of the models; `dh/meter.py` the energy and FLOP figures;
  `workers/jobs.py` the models and the tile arithmetic.
- **The connection:** the page calls relative addresses under `/api` and `/thumbs`. Send those to the service, as
  `web/vite.config.ts` does for the dev server.
- **Threads:** the models compute on 4 threads; set the environment variable `DH_THREADS` to change that.
- **Left over from the full dashboard:** `workers/jobs.py` and `memobase/search.py` are copied whole, so they also
  hold batch functions (`embed_chunk`, `image_chunk`, `caption_chunk`) and a text search that need files or a model
  that are not here. The page and the service do not use them.

## How the figures are made

- **Energy** is the processor's own energy counter read before and after a request (Windows EMI, `bench/emi.py`). It
  covers the processor only, not screen or fans, and includes every program that is running at that moment.
- **FLOPs** are calculated, not read from hardware, with a multiply-add counted as 2. Tiles are counted exactly from
  the picture. Text embeddings and photos use counts taken on real inputs (`measurements/flops_per_unit.json`). The
  language model is counted as 2 per weight of the matrix products each token passes through. Photo descriptions are
  not counted.
- **Saved answers** in `web/public/data` were recorded on the team laptop; their times and energies are that
  computer's, not yours.

## Photos

The photo panel uses 1,904 photos from Swiss archives, found through [Memobase](https://memobase.ch) (Memoriav). This
repository contains their records (`data/memobase/manifest.csv`: title, date, creator, institution, licence and
link; it lists 1,956 records, of which 1,904 are indexed), the vectors the photo-indexing job computed for them
(`index_clip_b32.npz`) and automatic tags (`tags.csv`), but not the photos. `memobase/download_thumbs.py` fetches the thumbnails from Memobase onto your computer.

The page only ever shows the 668 photos whose record sets no extra conditions of use, each whole, with creator,
institution, licence and a link to its record. Each photo stays under its own licence (Public Domain Mark,
CC BY or CC BY-SA, as its row in the manifest says).

## Models

| Model | Used for | Source |
|---|---|---|
| all-MiniLM-L6-v2 | text embeddings | huggingface.co/sentence-transformers/all-MiniLM-L6-v2 |
| CLIP ViT-B/32 | photo vectors | huggingface.co/sentence-transformers/clip-ViT-B-32 |
| Florence-2-base | photo descriptions | huggingface.co/florence-community/Florence-2-base (MIT) |
| Qwen3-0.6B | text generation | huggingface.co/Qwen/Qwen3-0.6B (Apache-2.0) |

Each model stays under its own licence; see its page. The small models here can be wrong, and the page says so.

## What was checked, and what was not

Checked on Windows 11: the tests, the build, and all five panels in the browser with and without the service.
Not checked: macOS and Linux, a computer without the models already downloaded, and installing from
`requirements.txt` into an empty Python environment.
