"""Measure the job types that had no batch figures: photo descriptions and text generation.

    python -m dh.measure                      # both
    python -m dh.measure caption --items 24   # photo descriptions on the first photos of the work list
    python -m dh.measure llm                  # text generation on the page's example requests
    python -m dh.measure llm --saved          # no new run: total the saved answers of web/public/data/tryjobs_demo.json
    python -m dh.measure --apply              # write the files again from the last run, without measuring

Each item is run for real, one after another, with the outside network blocked. For every item the script takes
the time, the energy from this computer's meter (dh/meter.py; the whole processor or computer, with every program
that is running) and the operations counted while the model computes (dh/flops.py). It also notes how busy the
other programs kept the processor, because their energy is inside the reading.

It writes:
    measurements/jobs_measured.json        every item and the totals of the run
    measurements/flops_per_unit.json       operations per photo and per written token, next to the earlier counts
    web/public/data/summary.json           the figures the page shows in each panel's fact box
    web/public/data/assumptions.json       the entry "Photo descriptions: speed and energy" of the assumptions table
    web/public/data/tryjobs_demo.json      after a new text-generation run: the saved answers, so that they and
                                           the fact box above them come from the same run

For text generation the unit is one written token (the end marker included), and reading the request is part of
the work: a short answer to a long request costs more per token than a long answer to a short one.

Photo descriptions need the thumbnails (python memobase/download_thumbs.py) and the description model; text
generation needs the language model. What is missing is skipped and said so.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from dh import flops, meter  # noqa: E402

OUT = ROOT / "measurements" / "jobs_measured.json"
DATA = ROOT / "web" / "public" / "data"
SUMMARY = DATA / "summary.json"
ASSUMPTIONS = DATA / "assumptions.json"
SAVED = DATA / "tryjobs_demo.json"
LLM_EXAMPLES = ROOT / "web" / "src" / "llm_examples.json"
QUIET_CORES = 0.25                 # other programs using less than a quarter of one core: a quiet machine


class OtherLoad:
    """How many processor cores the other programs kept busy during a block, on average (None if unknown)."""

    cores: float | None = None

    def __enter__(self):
        try:
            import psutil
            self._ps, self._me = psutil, psutil.Process()
            self._a = self._read()
        except Exception:  # noqa: BLE001 - without psutil the run is simply not labelled
            self._ps = None
        self._t0 = time.perf_counter()
        return self

    def _read(self) -> tuple[float, float]:
        t, own = self._ps.cpu_times(), self._me.cpu_times()
        return t.user + t.system, own.user + own.system        # busy time as the operating system reports it

    def __exit__(self, *exc):
        wall = time.perf_counter() - self._t0
        if self._ps is not None and wall > 0:
            b = self._read()
            self.cores = round(max(0.0, ((b[0] - self._a[0]) - (b[1] - self._a[1])) / wall), 2)
        return False


def totals(rows: list[dict], load) -> dict:
    """Totals of one run. Each row has units, seconds, joules and flops. Energy stays None when this computer has
    no meter or one reading failed; operations per unit are taken over the rows that have a count."""
    units = sum(r["units"] for r in rows)
    secs = sum(r["seconds"] for r in rows)
    known = bool(rows) and all(r["joules"] is not None for r in rows)
    joules = sum(r["joules"] for r in rows) if known else None
    counted = [r for r in rows if r.get("flops") and r["units"]]
    counted_units = sum(r["units"] for r in counted)
    cores = getattr(load, "cores", None)
    return {
        "runs": 1, "items": len(rows), "units": units, "seconds": round(secs, 3),
        "per_s": round(units / secs, 4) if secs > 0 else None,
        "joules": None if joules is None else round(joules, 3),
        "j_per_unit": None if joules is None or not units else round(joules / units, 4),
        "watts": None if joules is None or secs <= 0 else round(joules / secs, 2),
        "flops_per_unit": round(sum(r["flops"] for r in counted) / counted_units) if counted_units else None,
        "flops_per_unit_min": round(min(r["flops"] / r["units"] for r in counted)) if counted else None,
        "flops_per_unit_max": round(max(r["flops"] / r["units"] for r in counted)) if counted else None,
        "other_load_cores": cores, "quiet": cores is not None and cores < QUIET_CORES,
        "meter_scope": meter.scope(), "threads": meter.machine()["threads"],
    }


def quality(t: dict, what: str) -> str:
    """One sentence for the page about how good this measurement is. It names the average power of the run,
    because the energy per item depends on the power the processor works at: figures taken at different powers
    do not compare one to one."""
    if t["j_per_unit"] is None:
        return f"{what}; this computer gives no energy reading"
    at = f"{what}, at {t['watts']:.0f} W of {'computer' if t.get('meter_scope') == 'computer' else 'processor'} power on average"
    if t["quiet"]:
        return f"{at}; quiet machine"
    if t["other_load_cores"] is None:
        return f"{at}; whether other programs were running is not known"
    return f"{at}; other programs were running, so the energy per item is an upper value for that power"


def run_in_words(key: str, items: int) -> str:
    """What a run was, for the sentence about its quality."""
    return (f"{items} photos in one run, described one after another" if key == "caption"
            else f"{items} example requests, answered once each")


def measure_caption(items: int = 24) -> dict | None:
    """Describe the first `items` photos of the work list, one photo at a time as the batch job does."""
    from PIL import Image

    from workers import jobs
    try:
        ids = jobs.image_ids()[:items + 1]
        files = [jobs.IMAGE_DIR / f"{s}.jpg" for s in ids]
        if len(files) < 2 or not all(f.exists() for f in files):
            print("photo descriptions: skipped, the thumbnails are not on this computer", file=sys.stderr)
            return None
        model = jobs.load_captioner()[1]
    except Exception as e:  # noqa: BLE001 - a missing model or list: skip this job
        print(f"photo descriptions: skipped ({type(e).__name__})", file=sys.stderr)
        return None
    flops.watch(model)

    def photo(f):
        with Image.open(f) as im:
            return im.convert("RGB")
    jobs.caption_images([photo(files[-1])])                    # warm-up on a photo outside the measured ones
    rows = []
    with OtherLoad() as load:
        for sig, f in zip(ids[:-1], files[:-1]):
            img = photo(f)
            with meter.Span() as span, flops.Count() as count:
                text = jobs.caption_images([img])[0]
            rows.append({"signature": sig, "units": 1, "words": len(text.split()), **span.figures(count.total or None),
                         "weight_flops": count.weights, "attention_flops": count.attention})
    t = totals(rows, load)
    t.update(unit="photo", model=jobs.CAPTION_ID, revision=jobs.CAPTION_REVISION[:12],
             attention_counted=flops.attention_is_counted(model), rows=rows, what=run_in_words("caption", len(rows)))
    t["quality"] = quality(t, t["what"])
    return t


def _llm_totals(rows: list[dict], load, **facts) -> dict:
    t = totals(rows, load)
    writing = sum(r["writing_s"] for r in rows)
    t.update(unit="token", answers=len(rows), rows=rows, what=run_in_words("llm", len(rows)),
             tokens_per_s_writing=round(sum(r["tokens"] - 1 for r in rows) / writing, 2) if writing > 0 else None, **facts)
    t["quality"] = quality(t, t["what"])
    return t


def measure_llm() -> dict | None:
    """Answer the page's example requests once each; the unit is one written token (the end marker included)."""
    from workers import jobs
    try:
        model = jobs.load_writer()[1]
    except Exception as e:  # noqa: BLE001
        print(f"text generation: skipped ({type(e).__name__})", file=sys.stderr)
        return None
    flops.watch(model)
    jobs.write_text("Say hello.", max_new_tokens=8)             # warm-up
    asks = json.loads(LLM_EXAMPLES.read_text(encoding="utf-8"))
    rows, answers = [], []
    with OtherLoad() as load:
        for x in asks:
            with meter.Span() as span, flops.Count() as count:
                out = jobs.write_text(x["prompt"])
            work = span.figures(count.total or None)
            answers.append({"label": x["label"], "prompt": x["prompt"], **out, "work": work})
            rows.append({"label": x["label"], "units": int(out["produced"]), "tokens": int(out["tokens"]),
                         "prompt_tokens": int(out["prompt_tokens"]), "writing_s": out["writing_s"],
                         "first_token_s": out["first_token_s"], **work,
                         "weight_flops": count.weights, "attention_flops": count.attention})
    t = _llm_totals(rows, load, model=jobs.WRITER_ID, revision=jobs.WRITER_REVISION[:12],
                    attention_counted=flops.attention_is_counted(model))
    t["saved_answers"] = answers                                # written into the saved answers, not into the run file
    return t


def llm_from_saved() -> dict | None:
    """The same totals from the answers already saved for the page (recorded by python -m dh.tryjobs --demo), so
    that the fact box and the saved answers below it are one and the same run. Nothing is computed."""
    answers = (_read(SAVED, {}).get("llm") or {}).get("answers") or []
    rows = [{"label": a["label"], "units": int(a.get("produced") or a["tokens"]), "tokens": int(a["tokens"]),
             "prompt_tokens": int(a["prompt_tokens"]), "writing_s": a["writing_s"], "first_token_s": a.get("first_token_s"),
             **{k: (a.get("work") or {}).get(k) for k in ("seconds", "joules", "watts", "flops")}}
            for a in answers if (a.get("work") or {}).get("seconds")]
    if not rows:
        print("text generation: the saved answers carry no figures", file=sys.stderr)
        return None
    t = _llm_totals(rows, OtherLoad(), source="the saved answers of web/public/data/tryjobs_demo.json")
    t["meter_scope"] = (answers[0].get("work") or {}).get("scope") or t["meter_scope"]
    t["quality"] = quality(t, t["what"])
    return t


def _read(path: Path, default):
    """A JSON file, or `default` when it does not exist. A file that exists but cannot be read stops the run
    before anything is written, so that a damaged file is never replaced by a thinner one."""
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return default
    try:
        return json.loads(text)
    except ValueError as e:
        raise SystemExit(f"{path.name} is not readable JSON ({e}); nothing was written") from None


def _write(path: Path, data, indent: int | None = 1) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=indent), encoding="utf-8")
    os.replace(tmp, path)


def per_unit_flops(counts: dict, key: str) -> float | None:
    """Operations per item of a job type from the counts file, attention included where the file has it."""
    e = counts.get(key) or {}
    return e.get("flops_per_unit_with_attention") or e.get("flops_per_unit")


def caption_assumption(cap: dict) -> dict:
    """The row of the assumptions table about photo descriptions, in the wording of its neighbours."""
    value = f"{1 / cap['per_s']:.1f} seconds per photo" if cap.get("per_s") else "not timed"
    if cap.get("j_per_unit") is not None:
        value += f", {cap['j_per_unit']:.1f} J per photo"
    return {"label": "Photo descriptions: speed and energy", "value": value, "unit": "", "status": "measured",
            "source": f"1 timed run of {cap['items']} photos (python -m dh.measure), "
                      f"{'whole-computer' if cap.get('meter_scope') == 'computer' else 'processor package'} energy",
            "note": cap["quality"][0].upper() + cap["quality"][1:] + "."}


def write(results: dict) -> None:
    """Put a run into the files named at the top. Job types that were skipped keep what they had. Every file is
    read first; if one of them is damaged nothing is written."""
    measured, counts, summary = _read(OUT, {}), _read(meter.FLOPS_FILE, {}), _read(SUMMARY, {})
    table = _read(ASSUMPTIONS, None)
    cap, llm = results.get("caption"), results.get("llm")
    saved = _read(SAVED, None) if llm and llm.get("saved_answers") else None

    measured["convention"] = ("one run per job type on this computer; energy from the computer's own meter, gross "
                              "(every program that was running is inside it); operations counted while computing, "
                              "1 multiply-add = 2 FLOPs (dh/flops.py); text generation per written token, reading "
                              "the request included")
    for key, r in (("caption", cap), ("llm", llm)):
        if r:
            measured[key] = {k: v for k, v in r.items() if k != "saved_answers"}
            if r["flops_per_unit"]:
                n = r["units"]
                split = all(x.get("weight_flops") is not None for x in r["rows"])
                counts[key] = {
                    "unit": r["unit"], "precision": "FP32", "flops_per_unit": r["flops_per_unit"],
                    "flops_per_unit_min": r["flops_per_unit_min"], "flops_per_unit_max": r["flops_per_unit_max"], "units": n,
                    **({"weight_flops_per_unit": round(sum(x["weight_flops"] for x in r["rows"]) / n),
                        "attention_flops_per_unit": round(sum(x["attention_flops"] for x in r["rows"]) / n),
                        "attention_in_count": bool(r.get("attention_counted"))} if split else {}),
                    "counted_by": "dh/flops.py while the model computed (weight and attention products), python -m dh.measure",
                }

    work = summary.setdefault("workloads", {})
    jobs_ = work.setdefault("jobs", [])
    for j in jobs_:                                             # operations per item for every job type
        f = per_unit_flops(counts, j["key"])
        if f:
            j["gflop_per_unit"] = round(f / 1e9, 6)
    if cap:
        row = next((j for j in jobs_ if j["key"] == "caption"), None)
        if row is None:
            row = {"key": "caption", "label": "Photo descriptions", "unit": "photos"}
            jobs_.append(row)
        row.update(per_s=cap["per_s"], j_per_unit=cap["j_per_unit"], quality=cap["quality"])
        if isinstance(table, list):
            entry = next((e for e in table if e.get("id") == "measured_caption"), None)
            if entry is not None:
                entry.update(caption_assumption(cap))
    if llm:
        work["llm"] = {"key": "llm", "label": "Text generation", "unit": "tokens", "answers": llm["answers"],
                       "tokens": llm["units"], "per_s": llm["per_s"], "tokens_per_s_writing": llm["tokens_per_s_writing"],
                       "j_per_unit": llm["j_per_unit"], "quality": llm["quality"],
                       "gflop_per_unit": None if not llm["flops_per_unit"] else round(llm["flops_per_unit"] / 1e9, 6)}
        if saved is not None and isinstance(saved.get("llm"), dict):
            saved["llm"]["answers"] = llm["saved_answers"]

    _write(OUT, measured)
    _write(meter.FLOPS_FILE, counts)
    _write(SUMMARY, summary)
    if cap and isinstance(table, list):
        _write(ASSUMPTIONS, table, indent=None)
    if saved is not None:
        _write(SAVED, saved, indent=None)


def main(argv: list[str]) -> None:
    items = int(argv[argv.index("--items") + 1]) if "--items" in argv else 24
    names = [a for a in argv if a in ("caption", "llm")] or ["caption", "llm"]
    results = {}
    if "--apply" in argv:                                       # the last run again, with today's wording
        last = _read(OUT, {})
        for key in names:
            if key in last:
                what = run_in_words(key, last[key]["items"])
                results[key] = {**last[key], "what": what, "quality": quality(last[key], what)}
    else:
        if "caption" in names or ("llm" in names and "--saved" not in argv):
            from workers.jobs import ensure_offline
            ensure_offline()
        if "caption" in names:
            results["caption"] = measure_caption(items)
        if "llm" in names:
            results["llm"] = llm_from_saved() if "--saved" in argv else measure_llm()
    write(results)
    for key, r in results.items():
        if r:
            j = "no energy reading" if r["j_per_unit"] is None else f"{r['j_per_unit']:.2f} J"
            f = "no operation count" if not r["flops_per_unit"] else f"{r['flops_per_unit'] / 1e9:.2f} GFLOP"
            print(f"{key}: {r['units']} {r['unit']}s in {r['seconds']:.1f} s, {j} and {f} per {r['unit']}. {r['quality']}.")


if __name__ == "__main__":
    main(sys.argv[1:])
