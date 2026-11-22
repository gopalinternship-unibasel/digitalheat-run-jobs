"""The live job endpoints of the dashboard use the jobs' own arithmetic and models."""
import numpy as np
from pathlib import Path

import pytest

from workers.jobs import CHUNK, render_chunk, render_view


def test_render_view_is_the_tile_arithmetic():
    """A view cut to one tile of the tile job gives exactly that tile."""
    index = 3
    tiles = render_chunk(index)
    for k in range(CHUNK["render"]):
        i = index * CHUNK["render"] + k
        y0 = -1.5 + 3.0 * (i % 400) / 400
        view = render_view(-2.0, 1.0, y0, y0 + 3.0 / 400, 320, 40, 200)
        assert view.shape == tiles[k].shape and np.array_equal(view, tiles[k])


def test_render_png_and_limits():
    from dh import tryjobs
    png, info = tryjobs.render(-0.6, 0.0, 3.2, 5000, 5000, 9999)
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    assert info["width"] == tryjobs.MAX_W and info["height"] == tryjobs.MAX_H and info["iters"] == tryjobs.MAX_ITERS
    small, facts = tryjobs.render(-0.6, 0.0, 3.2, 320, 40, 200)
    assert facts["tiles"] == 1.0 and facts["pixels"] == 320 * 40


def test_similar_photos_by_signature():
    from dh import tryjobs
    try:
        code, start = tryjobs.photo_sample(4, seed=1)
    except FileNotFoundError:
        pytest.skip("the photo index is not on this machine")
    assert code == 200 and len(start["results"]) == 4
    sig = start["results"][0]["signature"]
    code, out = tryjobs.photo_similar_signature(sig, 6)
    assert code == 200 and len(out["results"]) == 6
    assert all(h["signature"] != sig for h in out["results"])
    scores = [h["score"] for h in out["results"]]
    assert scores == sorted(scores, reverse=True) and scores[0] <= 1.0001
    assert tryjobs.photo_similar_signature("../etc/passwd", 3)[0] == 404


def test_embed_search_finds_the_electricity_price():
    from dh import tryjobs
    try:
        code, out = tryjobs.embed_search("what does electricity cost per kilowatt-hour", 5)
    except Exception as e:  # the model is not in the local cache
        pytest.skip(f"text model not available: {e}")
    if code != 200:
        pytest.skip(out.get("error", "text model not available"))
    assert out["corpus"]["items"] >= 50 and out["seconds"] > 0
    assert any("lectricity" in (h["label"] or "") for h in out["results"][:3])
    assert tryjobs.embed_search("   ", 5)[1]["results"] == []


def test_strips_equal_one_piece_at_deep_zoom():
    """The live view is rendered in strips; they must give the picture rendered in one piece."""
    from dh import tryjobs
    for cx, cy, span in ((-0.6, 0.0, 3.2), (-0.745, 0.11, 0.02), (-0.7435, 0.1314, 0.0012), (-0.743643887, 0.131825904, 2e-7)):
        w, h, iters = 240, 150, 120
        strips, _ = tryjobs.escape_counts(cx, cy, span, w, h, iters)
        half = span * h / w / 2
        whole = render_view(cx - span / 2, cx + span / 2, cy - half, cy + half, w, h, iters)
        assert np.array_equal(strips, whole), (cx, cy, span)


def test_uploads_that_cannot_be_used_get_a_plain_refusal():
    from io import BytesIO

    from PIL import Image

    from dh import tryjobs

    def png(w, h, mode="RGB"):
        b = BytesIO()
        Image.new(mode, (w, h)).save(b, format="PNG")
        return b.getvalue()
    for data, code in ((png(1, 1), 400), (png(50, 1), 400), (b"this is not an image", 400), (b"", 400), (png(9000, 9000, "L"), 413)):
        with pytest.raises(tryjobs.BadImage) as e:
            tryjobs.open_image(data, 1024)
        assert e.value.code == code and ":\\" not in str(e.value) and "/" not in str(e.value)
    img, w, h = tryjobs.open_image(png(3000, 2000), 1024)
    assert (w, h) == (3000, 2000) and max(img.size) == 1024


def test_only_photos_without_extra_conditions_are_shown():
    from dh import tryjobs
    try:
        code, start = tryjobs.photo_sample(12, seed=5)
    except FileNotFoundError:
        pytest.skip("the photo index is not on this machine")
    assert all(h["extra_conditions"] == "no" for h in start["results"])
    code, out = tryjobs.photo_similar_signature(start["results"][0]["signature"], 12)
    assert code == 200 and all(h["extra_conditions"] == "no" for h in out["results"])
    assert out["photos_shown"] < out["photos"]
    blocked = next(s for s in tryjobs._photo["sigs"] if tryjobs._photo["meta"][s]["extra_conditions"] == "yes")
    assert tryjobs.photo_similar_signature(blocked, 3)[0] == 404


def test_render_refuses_nothing_but_stays_finite():
    from dh import tryjobs
    png, info = tryjobs.render(float("nan"), float("inf"), float("nan"), 64, 40, 50)
    assert png[:4] == b"\x89PNG" and info["cx"] == -0.6 and info["cy"] == 0.0 and info["span"] == 3.2


def test_service_answers_odd_requests_plainly():
    """The HTTP handler itself: wrong numbers, wrong lengths and wrong file types get an answer at once."""
    import http.client
    import socket
    import threading
    from http.server import ThreadingHTTPServer

    from dh import serve
    srv = ThreadingHTTPServer(("127.0.0.1", 0), serve.Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    port = srv.server_address[1]

    def get(path):
        c = http.client.HTTPConnection("127.0.0.1", port, timeout=30)
        c.request("GET", path)
        r = c.getresponse()
        body = r.read()
        c.close()
        return r.status, body
    try:
        assert get("/api/health")[0] == 200
        assert get("/api/embed/search?q=tank&k=%C2%B2")[0] == 400
        assert get("/api/photo/sample?n=1e3")[0] == 400
        assert get("/api/render?cx=abc")[0] == 400
        assert get("/api/embed/search?q=%00%01")[1] == b'{"query": "", "results": []}'
        status, body = get("/api/render?cx=nan&cy=0&span=0&w=99999&h=1&iters=-5")
        assert status == 200 and body[:4] == b"\x89PNG"
        for length in ("-1", "abc"):          # no body follows: the answer must come without waiting for one
            s = socket.create_connection(("127.0.0.1", port), timeout=10)
            s.sendall(f"POST /api/nothing-here HTTP/1.0\r\nContent-Length: {length}\r\n\r\n".encode())
            assert s.recv(64).split()[1] == b"400", length
            s.close()
        c = http.client.HTTPConnection("127.0.0.1", port, timeout=30)
        c.request("POST", "/api/photo/similar", body=b"hello", headers={"Content-Type": "text/plain"})
        r = c.getresponse()
        assert r.status == 415 and b"JPEG" in r.read()
        c.close()
    finally:
        srv.shutdown()
        srv.server_close()


def test_an_uploaded_archive_photo_finds_itself():
    """The upload route end to end: a thumbnail of an archive photo must come back as its own closest match."""
    from dh import tryjobs
    try:
        code, start = tryjobs.photo_sample(1, seed=11)
    except FileNotFoundError:
        pytest.skip("the photo index is not on this machine")
    sig = start["results"][0]["signature"]
    thumb = Path(tryjobs.__file__).resolve().parent.parent / "data" / "memobase" / "thumbs" / f"{sig}.jpg"
    if not thumb.exists():
        pytest.skip("the thumbnails are not on this machine")
    code, out = tryjobs.photo_similar_upload(thumb.read_bytes(), 3)
    if code == 500:
        pytest.skip("the photo model is not on this machine")
    assert code == 200 and out["results"][0]["signature"] == sig and out["results"][0]["score"] > 0.97
    assert all(h["extra_conditions"] == "no" for h in out["results"])


def _writer_or_skip():
    from workers import jobs
    try:
        jobs.load_writer()
    except Exception:  # noqa: BLE001 - any reason the model cannot be loaded here
        pytest.skip("the language model is not on this machine")
    return jobs


def test_the_language_model_repeats_its_answer_and_stops_when_told():
    jobs = _writer_or_skip()
    pieces = []
    a = jobs.write_text("Name three colours.", max_new_tokens=16, on_text=pieces.append)
    b = jobs.write_text("Name three colours.", max_new_tokens=16)
    assert a["text"] and a["text"] == b["text"] and "".join(pieces).strip() == a["text"]
    assert 0 < a["tokens"] <= 16 and a["first_token_s"] > 0 and not a["stopped"]
    if a["tokens"] > 1:                                   # the figures the page shows must divide
        assert abs((a["tokens"] - 1) / a["writing_s"] - a["tokens_per_s"]) < 0.5 * a["tokens_per_s"] * 0.2 + 0.2
    got = []
    c = jobs.write_text("Count from one to fifty in words.", max_new_tokens=60, on_text=got.append, stop=lambda: len(got) >= 3)
    assert c["stopped"] and c["tokens"] < 20 and not c["hit_limit"]


def test_the_answer_route_sends_pieces_and_refuses_an_empty_request():
    import http.client
    import json
    import threading
    from http.server import ThreadingHTTPServer

    from dh import serve, tryjobs
    _writer_or_skip()
    assert tryjobs.clean_prompt("  a\x00b \n\n c  ") == "a b\n\nc" and len(tryjobs.clean_prompt("x" * 5000)) == tryjobs.MAX_PROMPT
    srv = ThreadingHTTPServer(("127.0.0.1", 0), serve.Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()

    def post(body: bytes):
        c = http.client.HTTPConnection("127.0.0.1", srv.server_address[1], timeout=300)
        c.request("POST", "/api/llm/generate", body=body, headers={"Content-Type": "application/json"})
        r = c.getresponse()
        data = r.read().decode("utf-8")
        c.close()
        return r.status, r.getheader("Content-Type"), data
    try:
        assert post(b'{"prompt": "   "}')[0] == 400
        assert post(b'[1, 2]')[0] == 400
        assert post(b'{"prompt": 7}')[0] == 400
        status, ctype, data = post(json.dumps({"prompt": "Name three colours."}).encode())
        lines = [json.loads(x) for x in data.splitlines() if x.strip()]
        assert status == 200 and ctype.startswith("application/x-ndjson")
        assert lines[-1].get("done") is True and lines[-1]["tokens"] > 0 and lines[-1]["tokens_per_s"] > 0
        assert all(set(x) == {"text"} for x in lines[:-1]) and "".join(x["text"] for x in lines[:-1]).strip()
    finally:
        srv.shutdown()
        srv.server_close()


def test_pieces_of_an_answer_add_up_also_after_leading_blanks():
    """What the page gets piece by piece must be the answer itself, whatever blanks or half characters come first."""
    from workers.jobs import next_piece
    for steps in (["\n", "\n\n", "\n\nHello", "\n\nHello world", "\n\nHello world, how"],
                  [" A", " A heat", " A heat pump"],
                  ["Gr", "Gr\ufffd", "Grüezi", "Grüezi mit"],
                  ["", "  ", "  1. One\n", "  1. One\n2. Two"]):
        state, got = {}, ""
        for whole in steps:
            got += next_piece(state, whole)
        assert got == steps[-1].lstrip(), (steps, got)
