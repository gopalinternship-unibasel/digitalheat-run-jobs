"""Live figures for the dashboard's "Run jobs" page: the processor's power while the laptop works, the energy of
one request, and how many arithmetic operations a request takes.

Energy comes from the computer's own meter, whichever of these it has:
- Windows: the processor's energy counter through EMI (bench/emi.py): the processor package only;
- Linux: the same counter through /sys/class/powercap (RAPL), where the system lets a normal user read it;
- macOS: the power of the whole computer as its battery controller reports it (ioreg), summed up over time.
In every case the reading includes every program that is running, not this service alone. A request is metered by
reading before and after it.

Operations (FLOPs; one multiply-add counts as 2, as in measurements/count_flops.py) are calculated, not read from
the hardware:
- tiles: exact, from the escape counts of the picture (every pixel takes |z| in every step: 4; every pixel still
  inside does z = z*z + c: 8);
- text embeddings and photos: the counts of measurements/flops_per_unit.json (matrix products counted on the real
  inputs of the measured runs), per token and per photo;
- the language model: 2 per weight of the matrix products each token passes through, from the loaded model itself.
The step that compares tokens with each other (attention) is not in these counts, except where the measured file
adds it. Photo descriptions are not counted.
"""
from __future__ import annotations

import json
import os
import plistlib
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FLOPS_FILE = ROOT / "measurements" / "flops_per_unit.json"
STEP_S = 0.5                       # the power line: one point every half second
KEEP = 120                         # points kept: the last minute

_meter = None
_error: str | None = None
_series: deque = deque(maxlen=KEEP)
_thread: threading.Thread | None = None
_guard = threading.Lock()


class RaplMeter:
    """Linux: the processor package's cumulative energy counter. read() gives (microjoules, nanoseconds)."""
    scope, source = "processor", "the processor's energy counter (Linux RAPL)"

    def __init__(self, base: Path = Path("/sys/class/powercap/intel-rapl:0")):
        self._file = base / "energy_uj"
        self._range = int((base / "max_energy_range_uj").read_text())
        self.read()

    def read(self) -> tuple[int, int]:
        return int(self._file.read_text()), time.monotonic_ns()

    def joules(self, a, b) -> float:
        d = b[0] - a[0]
        return (d + self._range if d < 0 else d) / 1e6      # the counter starts again at zero when it is full

    @staticmethod
    def seconds(a, b) -> float:
        return (b[1] - a[1]) / 1e9


def mac_watts(plist: bytes) -> float | None:
    """Power of the whole computer in watts from `ioreg -r -n AppleSmartBattery -a`, or None when it gives none.
    First choice is the controller's own figure for the system (PowerTelemetryData.SystemLoad, in milliwatts);
    otherwise, on battery only, voltage times current. Values outside 0.3 to 400 W are not believed."""
    try:
        rows = plistlib.loads(plist)
        d = rows[0] if isinstance(rows, list) and rows else {}
    except Exception:  # noqa: BLE001 - no battery controller, or an answer that is not a property list
        return None

    def ok(w):
        return w if isinstance(w, (int, float)) and 0.3 <= w <= 400 else None
    load = (d.get("PowerTelemetryData") or {}).get("SystemLoad")
    if isinstance(load, int) and ok(load / 1000) is not None:
        return load / 1000
    volts, amps = d.get("Voltage"), d.get("InstantAmperage", d.get("Amperage"))
    if d.get("ExternalConnected") is False and isinstance(volts, int) and isinstance(amps, int):
        if amps >= 2**63:                                  # a negative number stored as an unsigned one
            amps -= 2**64
        return ok(volts * abs(amps) / 1e6)
    return None


def _ioreg() -> bytes:
    return subprocess.run(["/usr/sbin/ioreg", "-r", "-n", "AppleSmartBattery", "-a"], capture_output=True, timeout=5).stdout


class MacMeter:
    """macOS: the computer's power now, asked once a second and summed up. read() gives (joules, seconds)."""
    scope, source = "computer", "the computer's power as its battery controller reports it (macOS)"

    def __init__(self, reader=_ioreg, step: float = 1.0):
        w = mac_watts(reader())
        if w is None:
            raise RuntimeError("no power figure from the battery controller")
        self._reader, self._step = reader, step
        self._lock = threading.Lock()
        self._joules, self._w, self._t = 0.0, w, time.monotonic()
        threading.Thread(target=self._follow, daemon=True).start()

    def _follow(self) -> None:
        while True:
            time.sleep(self._step)
            try:
                w = mac_watts(self._reader())
            except Exception:  # noqa: BLE001 - keep the last value
                w = None
            with self._lock:
                now = time.monotonic()
                self._joules += self._w * (now - self._t)
                self._t = now
                if w is not None:
                    self._w = w

    def read(self) -> tuple[float, float]:
        with self._lock:
            now = time.monotonic()
            return self._joules + self._w * (now - self._t), now

    @staticmethod
    def joules(a, b) -> float:
        return b[0] - a[0]

    @staticmethod
    def seconds(a, b) -> float:
        return b[1] - a[1]


def _get():
    """This computer's meter, or None when it has none that a normal user may read (the reason is kept in _error)."""
    global _meter, _error
    with _guard:
        if _meter is None and _error is None:
            try:
                if os.environ.get("DH_METER") == "off":
                    raise RuntimeError("switched off")
                if sys.platform == "win32":
                    from bench.emi import PackageMeter
                    _meter = PackageMeter()
                elif sys.platform == "darwin":
                    _meter = MacMeter()
                else:
                    _meter = RaplMeter()
            except Exception:  # noqa: BLE001 - any reason: the page then says that there is no power reading
                _error = {"win32": "Windows gives no reading of the processor's energy counter on this computer",
                          "darwin": "this Mac's battery controller gives no power figure (a desktop Mac has none)",
                          }.get(sys.platform, "the processor's energy counter cannot be read here (on Linux it is often limited to the administrator)")
        return _meter


def scope() -> str:
    """What the reading covers: "processor" (Windows, Linux) or "computer" (macOS)."""
    return getattr(_meter, "scope", "processor")


def _sample(m) -> None:
    a = m.read()
    while True:
        time.sleep(STEP_S)
        b = m.read()
        s = m.seconds(a, b)
        if s > 0:
            _series.append(round(m.joules(a, b) / s, 2))
        a = b


def start() -> None:
    """Begin drawing the power line (once); does nothing where there is no meter."""
    global _thread
    m = _get()
    with _guard:
        if m is not None and _thread is None:
            _thread = threading.Thread(target=_sample, args=(m,), daemon=True)
            _thread.start()


def machine() -> dict:
    """Cores of this computer's processor and the threads the jobs compute on (workers.jobs.THREADS)."""
    import os

    import psutil
    return {"cores": psutil.cpu_count(logical=False), "logical": psutil.cpu_count(),
            "threads": int(os.environ.get("DH_THREADS", "4"))}


def now() -> dict:
    """The power line of the last minute and its latest value (mean of the last second)."""
    start()
    pts = list(_series)
    if _meter is None or not pts:
        return {"available": _meter is not None, "error": _error, "watts": None, "series": [], "step_s": STEP_S,
                "machine": machine(), "scope": scope()}
    return {"available": True, "error": None, "watts": round(sum(pts[-2:]) / len(pts[-2:]), 1), "low": min(pts), "high": max(pts),
            "series": pts, "step_s": STEP_S, "machine": machine(), "scope": scope(),
            "source": getattr(_meter, "source", "the processor's energy counter (Windows)")}


class Span:
    """Meter one piece of work:  with Span() as s: ...  then s.figures()."""

    def __enter__(self):
        self._m = _get()
        self._a = self._m.read() if self._m is not None else None
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.seconds = time.perf_counter() - self._t0
        self._b = self._m.read() if self._m is not None else None
        return False

    def figures(self, flops: float | None = None) -> dict:
        """seconds, joules and watts of the span (joules and watts None without a meter), and the operations."""
        out = {"seconds": round(self.seconds, 4), "joules": None, "watts": None, "flops": None if flops is None else float(round(flops)),
               "scope": getattr(self._m, "scope", "processor")}
        if self._m is not None and self._a is not None and self._b is not None:
            j, s = self._m.joules(self._a, self._b), self._m.seconds(self._a, self._b)
            if s > 0 and j >= 0:
                out.update(joules=round(j, 3), watts=round(j / s, 1))
        return out


# ------------------------------------------------------------------ operations
_counts: dict | None = None


def counts() -> dict:
    """The measured operation counts per unit (measurements/flops_per_unit.json); {} when the file is missing."""
    global _counts
    if _counts is None:
        try:
            _counts = json.loads(FLOPS_FILE.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            _counts = {}
    return _counts


def render_flops(pixels: int, iters: int, inside_steps: int) -> float:
    """Exact for workers.jobs.render_view: |z| of every pixel in every step (4), z = z*z + c for the rest (8)."""
    return 4.0 * pixels * iters + 8.0 * inside_steps


def embed_flops(tokens: int) -> float | None:
    """The matrix products of all-MiniLM-L6-v2 for a text of this many tokens, from the measured count per token."""
    e = counts().get("embed") or {}
    try:
        return tokens * e["flops_per_unit"] / e["padded_tokens_per_chunk_mean"]
    except (KeyError, ZeroDivisionError, TypeError):
        return None


def photo_flops() -> float | None:
    """One photo through the CLIP ViT-B/32 image encoder, as counted (attention products included)."""
    e = counts().get("image") or {}
    return e.get("flops_per_unit_with_attention") or e.get("flops_per_unit")


def rank_flops(vectors: int, dims: int) -> float:
    """Comparing one vector with `vectors` others: a product and a sum per number."""
    return 2.0 * vectors * dims


def writer_weights(model) -> tuple[int, int]:
    """(weights in the matrix products of the model's layers, weights of its output layer over the vocabulary)."""
    head = model.get_output_embeddings().weight
    table = model.get_input_embeddings().weight
    body = sum(p.numel() for p in model.parameters() if p.ndim == 2 and p is not head and p is not table)
    return int(body), int(head.numel())


def writer_flops(body: int, head: int, prompt_tokens: int, written: int) -> float:
    """Every token of the request and every written token but the last passes the layers once (2 per weight); the
    output layer is used once per written token. `written` counts the end marker too."""
    return 2.0 * body * (prompt_tokens + max(0, written - 1)) + 2.0 * head * written
