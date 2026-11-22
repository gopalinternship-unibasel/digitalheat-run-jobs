"""The live figures of the Run jobs page: operation counts are checked against a direct count, the meter against
its own rules. No model is needed here."""
import numpy as np

from dh import meter
from workers.jobs import render_view


def test_tile_operations_equal_a_direct_count():
    """Count the steps of a small picture one by one, the way render_view works, and compare."""
    w, h, iters = 24, 16, 30
    esc = render_view(-2.0, 1.0, -1.0, 1.0, w, h, iters)
    x, y = np.linspace(-2.0, 1.0, w), np.linspace(-1.0, 1.0, h)
    flops = 0
    for yy in y:
        for xx in x:
            z, c = 0j, complex(xx, yy)
            for _ in range(iters):
                flops += 4                      # |z| is taken for every pixel in every step
                if abs(z) <= 2:
                    z = z * z + c
                    flops += 8
    assert meter.render_flops(w * h, iters, int(esc.sum())) == flops


def test_counts_per_unit_come_from_the_measured_file():
    c = meter.counts()
    assert meter.photo_flops() == c["image"]["flops_per_unit_with_attention"]
    per_token = c["embed"]["flops_per_unit"] / c["embed"]["padded_tokens_per_chunk_mean"]
    assert abs(meter.embed_flops(10) - 10 * per_token) < 1 and 15e6 < per_token < 30e6
    assert meter.rank_flops(668, 512) == 2 * 668 * 512
    # 5 tokens of request, 3 written: 5 + 2 tokens pass the layers, the output layer is used 3 times
    assert meter.writer_flops(100, 10, 5, 3) == 2 * 100 * 7 + 2 * 10 * 3


def test_a_span_gives_time_and_never_a_negative_energy():
    with meter.Span() as s:
        sum(i * i for i in range(200000))
    f = s.figures(flops=123.4)
    assert f["seconds"] > 0 and f["flops"] == 123.0
    assert (f["joules"] is None and f["watts"] is None) or (f["joules"] >= 0 and f["watts"] > 0)


def test_the_power_line_has_the_promised_shape():
    import time
    meter.start()
    time.sleep(1.3)
    now = meter.now()
    assert set(now) >= {"available", "watts", "series", "step_s"}
    if now["available"]:
        assert now["watts"] > 0 and 1 <= len(now["series"]) <= meter.KEEP and all(v >= 0 for v in now["series"])
    else:
        assert now["watts"] is None and now["series"] == []
