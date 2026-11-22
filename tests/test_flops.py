"""The operations of an answer are counted while it is computed (dh/flops.py). The count must equal what
PyTorch's own FLOP counter gives where that counter can see the work, and known figures where it cannot."""
import threading

import pytest

torch = pytest.importorskip("torch")
from dh import flops, meter  # noqa: E402


def _tiny():
    return torch.nn.Sequential(
        torch.nn.Conv2d(3, 8, kernel_size=4, stride=4, bias=False),          # 16 x 16 -> 4 x 4
        torch.nn.Conv2d(8, 8, kernel_size=3, padding=1, groups=8),           # one kernel per channel
        torch.nn.Flatten(2), )


def test_weight_products_equal_pytorchs_own_counter():
    """Linear layers (with and without bias, on rows and on stacks of rows) and convolutions (also grouped)."""
    from torch.utils.flop_counter import FlopCounterMode
    conv, lin, head = _tiny(), torch.nn.Linear(16, 32), torch.nn.Linear(32, 5, bias=False)
    for m in (conv, lin, head):
        flops.watch(m)
    x = torch.randn(2, 3, 16, 16)

    def run():
        return head(lin(conv(x)))                       # (2, 8, 16) -> (2, 8, 32) -> (2, 8, 5)
    with torch.inference_mode():
        with FlopCounterMode(display=False) as theirs:
            run()
        with flops.Count() as ours:
            run()
    by_hand = 2 * (8 * 3 * 4 * 4) * 2 * 16 + 2 * (8 * 1 * 3 * 3) * 2 * 16 + 2 * (16 * 32) * 16 + 2 * (32 * 5) * 16
    assert ours.weights == by_hand == theirs.get_total_flops()
    assert ours.attention == 0 and ours.total == by_hand


def test_attention_products_are_counted_from_the_sizes():
    """Four heads, 10 queries, 12 keys of 16 numbers: the same as multiplying the matrices one after the other,
    which PyTorch's counter can see (it cannot see inside the fused attention function)."""
    from torch.utils.flop_counter import FlopCounterMode
    flops.watch(torch.nn.Linear(1, 1))                  # switches the attention count on
    q, k, v = torch.randn(2, 4, 10, 16), torch.randn(2, 4, 12, 16), torch.randn(2, 4, 12, 16)
    with flops.Count() as ours:
        out = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    with FlopCounterMode(display=False) as theirs:
        torch.softmax(q @ k.transpose(-1, -2) / 4.0, dim=-1) @ v
    assert ours.attention == 2 * 8 * 10 * 12 * (16 + 16) == theirs.get_total_flops()
    assert ours.weights == 0 and out.shape == (2, 4, 10, 16)
    again = torch.nn.functional.scaled_dot_product_attention(q, k, v)       # counting does not change the result
    assert torch.equal(out, again)


def test_nothing_is_counted_outside_a_count_and_watching_twice_counts_once():
    lin = torch.nn.Linear(4, 4)
    assert flops.watch(lin) == 1 and flops.watch(lin) == 1
    lin(torch.randn(3, 4))                              # no count open: nothing happens
    with flops.Count() as c:
        lin(torch.randn(3, 4))
    assert c.weights == 2 * 16 * 3


def test_a_count_belongs_to_its_thread_and_an_inner_count_adds_to_the_outer_one():
    lin = torch.nn.Linear(4, 4)
    flops.watch(lin)
    other = {}

    def elsewhere():
        lin(torch.randn(50, 4))                         # another answer, computed at the same moment
        with flops.Count() as c:
            lin(torch.randn(7, 4))
        other["flops"] = c.weights
    with flops.Count() as outer:
        t = threading.Thread(target=elsewhere)
        t.start()
        t.join()
        with flops.Count() as inner:
            lin(torch.randn(2, 4))
        lin(torch.randn(1, 4))
    assert other["flops"] == 2 * 16 * 7
    assert inner.weights == 2 * 16 * 2 and outer.weights == 2 * 16 * 3


def test_measured_totals_and_the_sentence_about_their_quality():
    from dh import measure

    class Load:
        cores = 0.1
    rows = [{"units": 1, "seconds": 2.0, "joules": 40.0, "flops": 4e11}, {"units": 1, "seconds": 3.0, "joules": 70.0, "flops": 6e11}]
    t = measure.totals(rows, Load())
    assert t["per_s"] == 0.4 and t["j_per_unit"] == 55.0 and t["watts"] == 22.0 and t["flops_per_unit"] == 5e11
    assert t["flops_per_unit_min"] == 4e11 and t["flops_per_unit_max"] == 6e11 and t["quiet"] is True
    assert measure.quality(t, "2 photos") == "2 photos, at 22 W of processor power on average; quiet machine"
    Load.cores = 1.5
    busy = measure.totals(rows, Load())
    assert busy["quiet"] is False and "upper value" in measure.quality(busy, "2 photos")
    Load.cores = None
    assert "not known" in measure.quality(measure.totals(rows, Load()), "2 photos")
    rows[0]["flops"] = None                             # one item without a count: the mean is over the counted one
    assert measure.totals(rows, Load())["flops_per_unit"] == 6e11
    rows[1]["joules"] = None                            # a computer without a meter
    none = measure.totals(rows, Load())
    assert none["j_per_unit"] is None and none["watts"] is None and "no energy reading" in measure.quality(none, "2 photos")


def test_a_measured_run_fills_the_files_and_a_damaged_file_stops_it(monkeypatch):
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as folder:
        _measured_run_in(Path(folder), monkeypatch)


def _measured_run_in(tmp_path, monkeypatch):
    from dh import measure
    counts, summary, table = tmp_path / "flops.json", tmp_path / "summary.json", tmp_path / "assumptions.json"
    counts.write_text('{"image": {"flops_per_unit": 8725463040.0, "flops_per_unit_with_attention": 8817623040.0}}', encoding="utf-8")
    summary.write_text('{"workloads": {"jobs": [{"key": "image", "per_s": 23.7}, {"key": "caption", "per_s": 0.17, "j_per_unit": null}]}}', encoding="utf-8")
    table.write_text('[{"id": "measured_caption", "label": "Photo descriptions: speed", "value": "5.9 seconds per photo", "note": "old"}]', encoding="utf-8")
    monkeypatch.setattr(measure, "OUT", tmp_path / "run.json")
    monkeypatch.setattr(measure.meter, "FLOPS_FILE", counts)
    monkeypatch.setattr(measure, "SUMMARY", summary)
    monkeypatch.setattr(measure, "ASSUMPTIONS", table)

    class Load:
        cores = 0.0
    rows = [{"units": 1, "seconds": 4.0, "joules": 80.0, "flops": 4.7e11, "weight_flops": 4.5e11, "attention_flops": 2e10}]
    cap = measure.totals(rows, Load())
    cap.update(unit="photo", rows=rows, attention_counted=True, what="1 photos in one run")
    cap["quality"] = measure.quality(cap, cap["what"])
    measure.write({"caption": cap, "llm": None})
    import json
    jobs = {j["key"]: j for j in json.loads(summary.read_text(encoding="utf-8"))["workloads"]["jobs"]}
    assert jobs["caption"]["j_per_unit"] == 80.0 and jobs["caption"]["per_s"] == 0.25 and jobs["caption"]["gflop_per_unit"] == 470.0
    assert jobs["image"]["gflop_per_unit"] == 8.817623 and jobs["image"]["per_s"] == 23.7           # the others keep theirs
    kept = json.loads(counts.read_text(encoding="utf-8"))
    assert kept["image"]["flops_per_unit"] == 8725463040.0 and kept["caption"]["flops_per_unit"] == 4.7e11
    entry = json.loads(table.read_text(encoding="utf-8"))[0]
    assert entry["value"] == "4.0 seconds per photo, 80.0 J per photo" and "quiet machine" in entry["note"]
    counts.write_text('{"image": {"flops_per_unit": 1,}}', encoding="utf-8")       # a hand edit gone wrong
    before = summary.read_text(encoding="utf-8")
    with pytest.raises(SystemExit):
        measure.write({"caption": cap})
    assert summary.read_text(encoding="utf-8") == before and counts.read_text(encoding="utf-8").endswith(",}}")


def test_odd_calls_are_counted_too():
    """A layer called by keyword, a convolution on one item without the leading size, and keys that are shared by
    several queries' items."""
    lin, conv = torch.nn.Linear(4, 6), torch.nn.Conv1d(4, 8, 3)
    flops.watch(lin)
    flops.watch(conv)
    with flops.Count() as c:
        lin(input=torch.randn(5, 4))
    assert c.weights == 2 * 24 * 5
    with flops.Count() as c:
        conv(torch.randn(4, 100))                       # one item: (channels, positions) -> (8, 98)
    assert c.weights == 2 * (8 * 4 * 3) * 98
    assert flops.attention_flops((1, 2, 5, 8), (3, 2, 7, 8), (3, 2, 7, 8)) == 2 * 6 * 5 * 7 * 16


# ------------------------------------------------------------------ with the real models (skipped when not here)
def test_one_photo_through_the_photo_model_is_the_measured_count():
    """8,817,623,040 FLOPs: the figure of measurements/flops_per_unit.json, reproduced to the last digit."""
    from PIL import Image

    from workers import jobs
    try:
        jobs.ensure_offline()
        model = jobs.load_clip()
    except Exception:  # noqa: BLE001
        pytest.skip("the photo model is not on this machine")
    flops.watch(model)
    with flops.Count() as c:
        model.encode([Image.new("RGB", (640, 480), (90, 110, 130))], batch_size=1, convert_to_numpy=True, show_progress_bar=False)
    image = meter.counts()["image"]
    assert c.weights == image["flops_per_unit"] and c.attention == image["attention_flops_per_unit_analytic"]
    assert c.total == meter.photo_flops() == 8817623040


def test_a_sentence_through_the_text_model_follows_its_layer_sizes():
    from workers import jobs
    try:
        jobs.ensure_offline()
        model = jobs.load_embedder()
    except Exception:  # noqa: BLE001
        pytest.skip("the text model is not on this machine")
    flops.watch(model)
    text = "what does electricity cost"
    tokens = int(model.tokenize([text])["input_ids"].shape[1])
    with flops.Count() as c:
        model.encode([text], convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
    layers = 6 * (4 * 384 * 384 + 2 * 384 * 1536)       # six layers: four attention matrices and two of the feed-forward step
    assert c.weights == 2 * layers * tokens + 2 * 384 * 384          # and once the pooling layer the library also runs
    assert c.attention == 6 * 4 * 384 * tokens * tokens
    assert abs(c.weights / meter.embed_flops(tokens) - 1) < 0.01     # the earlier per-token figure, within 1 %


def test_the_language_models_count_equals_the_formula_for_its_weights():
    from workers import jobs
    try:
        model = jobs.load_writer()[1]
    except Exception:  # noqa: BLE001
        pytest.skip("the language model is not on this machine")
    flops.watch(model)
    with flops.Count() as c:
        out = jobs.write_text("Count from one to five.", max_new_tokens=12)
    body, head = meter.writer_weights(model)
    p, w = out["prompt_tokens"], out["produced"]
    assert c.weights == meter.writer_flops(body, head, p, w)
    cfg = model.config                                   # each new token asks all tokens before it, in every layer and head
    asked = p * p + sum(p + k for k in range(1, w))
    assert c.attention == cfg.num_hidden_layers * 2 * cfg.num_attention_heads * 2 * cfg.head_dim * asked
    assert flops.attention_is_counted(model)
