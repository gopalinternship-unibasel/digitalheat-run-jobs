"""Count the arithmetic of one answer while a model computes it.

    flops.watch(model)                 # once, after the model is loaded
    with flops.Count() as c:
        model(...)
    c.weights, c.attention, c.total    # FLOPs of this piece of work

What is counted (one multiply-add = 2 FLOPs, the convention of measurements/flops_per_unit.json):
- weights: every product with a weight matrix or a convolution kernel (torch.nn.Linear, Conv1d, Conv2d), from the
  sizes of the data that really passed through, so longer texts, more drafts and bigger pictures count for more;
- attention: the two products inside every attention step (queries with keys, then the result with the values),
  from the sizes handed to torch.nn.functional.scaled_dot_product_attention. The whole products are counted, also
  where a mask afterwards hides a part of the result (a language model reading its request), as PyTorch's own
  counter does when it sees the two products one after the other.
Not counted: additions of biases, normalisation, activation functions, softmax and table look-ups, which are small
next to the products. The count is made of sizes only: it reads no hardware counter and costs a few microseconds
per layer.

The count belongs to the thread that opened it, so answers computed side by side do not mix.
"""
from __future__ import annotations

import math
import threading

_local = threading.local()
_setup = threading.Lock()
_patched = False
_FLAG = "_dh_flops_watched"


class Count:
    """The FLOPs of the work done in this thread between entering and leaving the block."""

    def __enter__(self):
        self.weights = 0
        self.attention = 0
        self._outer = getattr(_local, "count", None)
        _local.count = self
        return self

    def __exit__(self, *exc):
        _local.count = self._outer
        if self._outer is not None:                    # a count inside a count is part of the outer one
            self._outer.weights += self.weights
            self._outer.attention += self.attention
        return False

    @property
    def total(self) -> int:
        return self.weights + self.attention


def linear_flops(weights: int, in_features: int, numbers_in: int) -> int:
    """A weight matrix applied to numbers_in / in_features vectors: 2 per weight and vector."""
    return 2 * weights * (numbers_in // in_features)


def conv_flops(weights: int, batch: int, positions: int) -> int:
    """A convolution kernel applied at every output position: 2 per weight, position and picture."""
    return 2 * weights * batch * positions


def attention_flops(query_shape, key_shape, value_shape) -> int:
    """Queries (..., q, d) with keys (..., k, d), then the weights (..., q, k) with values (..., k, v):
    2 * q * k * (d + v) for every head of every item (the largest of the three leading sizes, as they broadcast)."""
    lead = max(math.prod(query_shape[:-2]), math.prod(key_shape[:-2]), math.prod(value_shape[:-2]))
    return 2 * lead * query_shape[-2] * key_shape[-2] * (query_shape[-1] + value_shape[-1])


def _linear_hook(module, args, kwargs, output):
    c = getattr(_local, "count", None)
    if c is not None:
        x = args[0] if args else kwargs.get("input")
        if x is not None:
            c.weights += linear_flops(module.weight.numel(), module.in_features, x.numel())


def _conv_hook(module, args, kwargs, output):
    c = getattr(_local, "count", None)
    if c is not None:
        batched = output.ndim == module.weight.ndim     # (items, channels, positions...); without items: one item
        c.weights += conv_flops(module.weight.numel(), output.shape[0] if batched else 1,
                                math.prod(output.shape[2:] if batched else output.shape[1:]))


def _watch_attention() -> None:
    """Wrap torch's attention function once, so that every call adds its two products to the open count.
    Called with the set-up lock held."""
    global _patched
    if _patched:
        return
    import torch.nn.functional as F
    real = F.scaled_dot_product_attention

    def counted(*args, **kwargs):
        c = getattr(_local, "count", None)
        if c is not None:
            q = args[0] if len(args) > 0 else kwargs["query"]
            k = args[1] if len(args) > 1 else kwargs["key"]
            v = args[2] if len(args) > 2 else kwargs["value"]
            c.attention += attention_flops(q.shape, k.shape, v.shape)
        return real(*args, **kwargs)

    counted.__wrapped__ = real
    F.scaled_dot_product_attention = counted
    _patched = True


def watch(model) -> int:
    """Put the counting hooks on every linear and convolution layer of a loaded model (safe to call again, also
    from several threads). Returns how many layers are watched now."""
    import torch
    n = 0
    with _setup:
        _watch_attention()
        for m in model.modules():
            if isinstance(m, torch.nn.Linear):
                hook = _linear_hook
            elif isinstance(m, (torch.nn.Conv1d, torch.nn.Conv2d)):
                hook = _conv_hook
            else:
                continue
            n += 1
            if not getattr(m, _FLAG, False):
                m.register_forward_hook(hook, with_kwargs=True)
                setattr(m, _FLAG, True)
    return n


def attention_is_counted(model) -> bool:
    """True when every part of the model that names its way of computing attention uses torch's attention
    function ("sdpa"), which the count sees. Any other way (the libraries' "eager" mode multiplies the matrices
    itself) is not seen, and the attention part of a count is then too low."""
    seen = {getattr(getattr(m, "config", None), "_attn_implementation", None) for m in model.modules()}
    return seen <= {None, "sdpa"}
