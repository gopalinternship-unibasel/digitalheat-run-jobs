"""Block network access to other machines from this Python process; only the local machine stays reachable.

Imported by the caption job before the model is loaded, so the photo descriptions are computed on this machine:
the model and processor are read from the local Hugging Face cache, Hugging Face offline mode is switched on,
and any attempt by Python code in this process to resolve or reach an outside host raises NetworkBlocked
instead of silently downloading or uploading something.

Scope (what this does and does not cover): it replaces the Python-level socket functions of this process
(connect, connect_ex, sendto, sendmsg on internet sockets; getaddrinfo, gethostbyname, gethostbyname_ex,
gethostbyaddr). Everything in Python that opens a network connection (urllib, requests, httpx, huggingface_hub)
goes through them. It does not cover native libraries that open sockets on their own, or child processes; the
caption job uses neither for networking (torch and transformers compute only). Loopback addresses (127.0.0.0/8,
::1) and the name "localhost" stay allowed, because the dashboard's local API uses them.
"""
from __future__ import annotations

import ipaddress
import os
import socket
import sys

_LOCAL_NAMES = {"localhost"}


class NetworkBlocked(RuntimeError):
    pass


def _is_local(host) -> bool:
    """True only for loopback IP literals (127.0.0.0/8, ::1), the unspecified address and the name localhost.
    Host names are never trusted by their spelling (e.g. '127.example.org' is an outside name)."""
    if host is None:
        return True
    h = host.decode() if isinstance(host, (bytes, bytearray)) else str(host)
    h = h.strip().rstrip(".").lower()
    if h in _LOCAL_NAMES:
        return True
    try:
        ip = ipaddress.ip_address(h.split("%", 1)[0].strip("[]"))
    except ValueError:
        return False                                    # a name other than localhost: outside
    return ip.is_loopback or ip.is_unspecified


def _host_of(address):
    return address[0] if isinstance(address, tuple) and address else address


_real = {
    "connect": socket.socket.connect, "connect_ex": socket.socket.connect_ex,
    "sendto": socket.socket.sendto, "sendmsg": getattr(socket.socket, "sendmsg", None),
    "getaddrinfo": socket.getaddrinfo, "gethostbyname": socket.gethostbyname,
    "gethostbyname_ex": socket.gethostbyname_ex, "gethostbyaddr": socket.gethostbyaddr,
}
_installed = False
attempts: list[str] = []          # every blocked attempt, for the probe, the worker's evidence file and the tests


def _block(what: str, host) -> None:
    attempts.append(f"{what} {host}")
    raise NetworkBlocked(f"network access blocked ({what} {host}); the caption job runs offline")


def _inet(sock) -> bool:
    return sock.family in (socket.AF_INET, socket.AF_INET6)


def _guard_connect(self, address):
    if _inet(self) and not _is_local(_host_of(address)):
        _block("connect", _host_of(address))
    return _real["connect"](self, address)


def _guard_connect_ex(self, address):
    if _inet(self) and not _is_local(_host_of(address)):
        _block("connect_ex", _host_of(address))
    return _real["connect_ex"](self, address)


def _guard_sendto(self, data, *args):
    address = args[-1] if args else None
    if _inet(self) and address is not None and not _is_local(_host_of(address)):
        _block("sendto", _host_of(address))
    return _real["sendto"](self, data, *args)


def _guard_sendmsg(self, buffers, ancdata=(), flags=0, address=None):
    if _inet(self) and address is not None and not _is_local(_host_of(address)):
        _block("sendmsg", _host_of(address))
    if address is None:
        return _real["sendmsg"](self, buffers, ancdata, flags)
    return _real["sendmsg"](self, buffers, ancdata, flags, address)


def _guard_getaddrinfo(host, *args, **kwargs):
    if not _is_local(host):
        _block("resolve", host)
    return _real["getaddrinfo"](host, *args, **kwargs)


def _guard_gethostbyname(host):
    if not _is_local(host):
        _block("resolve", host)
    return _real["gethostbyname"](host)


def _guard_gethostbyname_ex(host):
    if not _is_local(host):
        _block("resolve", host)
    return _real["gethostbyname_ex"](host)


def _guard_gethostbyaddr(host):
    if not _is_local(host):
        _block("reverse-resolve", host)
    return _real["gethostbyaddr"](host)


_GUARDS = {"connect": _guard_connect, "connect_ex": _guard_connect_ex, "sendto": _guard_sendto,
           "sendmsg": _guard_sendmsg, "getaddrinfo": _guard_getaddrinfo, "gethostbyname": _guard_gethostbyname,
           "gethostbyname_ex": _guard_gethostbyname_ex, "gethostbyaddr": _guard_gethostbyaddr}
_SOCKET_METHODS = ("connect", "connect_ex", "sendto", "sendmsg")


def _set_library_offline_flags() -> None:
    """huggingface_hub and transformers read HF_HUB_OFFLINE once, when they are imported. If either was imported
    before install(), switch its stored flag on as well."""
    hub = sys.modules.get("huggingface_hub.constants")
    if hub is not None:
        hub.HF_HUB_OFFLINE = True
    tf_hub = sys.modules.get("transformers.utils.hub")
    if tf_hub is not None and hasattr(tf_hub, "_is_offline_mode"):
        tf_hub._is_offline_mode = True


def install() -> None:
    """Switch the Hugging Face libraries to offline mode and block outside connections. Idempotent."""
    global _installed
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    _set_library_offline_flags()
    for name in _SOCKET_METHODS:
        if _real[name] is not None:
            setattr(socket.socket, name, _GUARDS[name])
    for name in ("getaddrinfo", "gethostbyname", "gethostbyname_ex", "gethostbyaddr"):
        setattr(socket, name, _GUARDS[name])
    _installed = True


def installed() -> bool:
    """True when every guarded socket function is really the guard right now (not just once installed)."""
    if not _installed:
        return False
    ok = all(getattr(socket.socket, n) is _GUARDS[n] for n in _SOCKET_METHODS if _real[n] is not None)
    return ok and all(getattr(socket, n) is _GUARDS[n]
                      for n in ("getaddrinfo", "gethostbyname", "gethostbyname_ex", "gethostbyaddr"))


def library_offline_flags() -> dict:
    """The offline flags the Hugging Face libraries actually use (None when the library is not imported)."""
    hub = sys.modules.get("huggingface_hub.constants")
    tf_hub = sys.modules.get("transformers.utils.hub")
    return {"huggingface_hub": None if hub is None else bool(hub.HF_HUB_OFFLINE),
            "transformers": None if tf_hub is None else bool(tf_hub.is_offline_mode())}


def uninstall() -> None:
    """Restore the real socket functions (for tests only; the offline settings stay as they are)."""
    global _installed
    for name in _SOCKET_METHODS:
        if _real[name] is not None:
            setattr(socket.socket, name, _real[name])
    for name in ("getaddrinfo", "gethostbyname", "gethostbyname_ex", "gethostbyaddr"):
        setattr(socket, name, _real[name])
    _installed = False
