"""CPU package energy on Windows through the Energy Meter Interface (EMI).

EMI exposes the processor's RAPL counters as cumulative energy (picowatt-hours) with a timestamp
(100 ns ticks). It needs no admin rights and no driver. Energy over an interval is the difference
of two readings. Only the processor package is covered: screen, fans and power-supply losses are not.

    from bench.emi import PackageMeter
    m = PackageMeter(); a = m.read(); ...; b = m.read(); joules = m.joules(a, b)
"""
from __future__ import annotations

import ctypes
import ctypes.wintypes as wt
import struct
import uuid

GUID_EMI = "45BD8344-7ED6-49cf-A440-C276C933B053"
IOCTL_EMI_GET_VERSION = 0x224000
IOCTL_EMI_GET_METADATA_SIZE = 0x224004
IOCTL_EMI_GET_METADATA = 0x224008
IOCTL_EMI_GET_MEASUREMENT = 0x22400C
PACKAGE_CHANNEL = "RAPL_Package0_PKG"


class _GUID(ctypes.Structure):
    _fields_ = [("Data1", wt.DWORD), ("Data2", wt.WORD), ("Data3", wt.WORD), ("Data4", ctypes.c_ubyte * 8)]


class _IFDATA(ctypes.Structure):
    _fields_ = [("cbSize", wt.DWORD), ("InterfaceClassGuid", _GUID), ("Flags", wt.DWORD), ("Reserved", ctypes.c_void_p)]


class EmiUnavailable(RuntimeError):
    pass


class PackageMeter:
    def __init__(self, channel: str = PACKAGE_CHANNEL):
        try:
            self._setupapi = ctypes.WinDLL("setupapi", use_last_error=True)
            self._k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        except (OSError, AttributeError) as e:
            raise EmiUnavailable("EMI is a Windows interface") from e
        s, k = self._setupapi, self._k32
        s.SetupDiGetClassDevsW.restype = ctypes.c_void_p
        s.SetupDiGetClassDevsW.argtypes = [ctypes.POINTER(_GUID), wt.LPCWSTR, wt.HWND, wt.DWORD]
        s.SetupDiEnumDeviceInterfaces.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(_GUID), wt.DWORD,
                                                  ctypes.POINTER(_IFDATA)]
        s.SetupDiGetDeviceInterfaceDetailW.argtypes = [ctypes.c_void_p, ctypes.POINTER(_IFDATA), ctypes.c_void_p,
                                                       wt.DWORD, ctypes.POINTER(wt.DWORD), ctypes.c_void_p]
        k.CreateFileW.restype = wt.HANDLE
        k.CreateFileW.argtypes = [wt.LPCWSTR, wt.DWORD, wt.DWORD, ctypes.c_void_p, wt.DWORD, wt.DWORD, wt.HANDLE]
        k.DeviceIoControl.argtypes = [wt.HANDLE, wt.DWORD, ctypes.c_void_p, wt.DWORD, ctypes.c_void_p, wt.DWORD,
                                      ctypes.POINTER(wt.DWORD), ctypes.c_void_p]
        self.channel = channel
        self._handle, self._index, self._count = self._open(channel)

    def _ioctl(self, handle, code, size):
        buf = ctypes.create_string_buffer(size)
        got = wt.DWORD(0)
        if not self._k32.DeviceIoControl(handle, code, None, 0, buf, size, ctypes.byref(got), None):
            raise EmiUnavailable(f"EMI ioctl 0x{code:X} failed (error {ctypes.get_last_error()})")
        return buf.raw[: got.value]

    def _open(self, channel):
        u = uuid.UUID(GUID_EMI)
        g = _GUID()
        g.Data1, g.Data2, g.Data3 = u.fields[0], u.fields[1], u.fields[2]
        for i, b in enumerate(u.bytes[8:]):
            g.Data4[i] = b
        hdev = self._setupapi.SetupDiGetClassDevsW(ctypes.byref(g), None, None, 0x2 | 0x10)
        i = 0
        while True:
            ifd = _IFDATA()
            ifd.cbSize = ctypes.sizeof(_IFDATA)
            if not self._setupapi.SetupDiEnumDeviceInterfaces(hdev, None, ctypes.byref(g), i, ctypes.byref(ifd)):
                break
            i += 1
            need = wt.DWORD(0)
            self._setupapi.SetupDiGetDeviceInterfaceDetailW(hdev, ctypes.byref(ifd), None, 0, ctypes.byref(need), None)
            buf = ctypes.create_string_buffer(need.value)
            struct.pack_into("I", buf, 0, 8 if ctypes.sizeof(ctypes.c_void_p) == 8 else 6)
            if not self._setupapi.SetupDiGetDeviceInterfaceDetailW(hdev, ctypes.byref(ifd), buf, need, None, None):
                continue
            path = ctypes.wstring_at(ctypes.addressof(buf) + 4)
            handle = self._k32.CreateFileW(path, 0x80000000, 0x3, None, 3, 0, None)
            if handle in (None, wt.HANDLE(-1).value):
                continue
            if struct.unpack("<H", self._ioctl(handle, IOCTL_EMI_GET_VERSION, 2))[0] < 2:
                continue
            size = struct.unpack("<I", self._ioctl(handle, IOCTL_EMI_GET_METADATA_SIZE, 4))[0]
            meta = self._ioctl(handle, IOCTL_EMI_GET_METADATA, size)
            count = struct.unpack_from("<H", meta, 66)[0]
            off, names = 68, []
            for _ in range(count):
                _unit, namesize = struct.unpack_from("<iH", meta, off)
                names.append(meta[off + 6: off + 6 + namesize].decode("utf-16-le").rstrip("\x00"))
                off += 6 + namesize
            if channel in names:
                return handle, names.index(channel), count
        raise EmiUnavailable(f"no EMI channel named {channel}")

    def read(self) -> tuple[int, int]:
        """(cumulative energy in picowatt-hours, timestamp in 100 ns ticks) of the package channel."""
        raw = self._ioctl(self._handle, IOCTL_EMI_GET_MEASUREMENT, 16 * self._count)
        return struct.unpack_from("<QQ", raw, 16 * self._index)

    @staticmethod
    def joules(a: tuple[int, int], b: tuple[int, int]) -> float:
        return (b[0] - a[0]) * 1e-12 * 3600.0

    @staticmethod
    def seconds(a: tuple[int, int], b: tuple[int, int]) -> float:
        return (b[1] - a[1]) * 1e-7

    @classmethod
    def watts(cls, a, b) -> float:
        s = cls.seconds(a, b)
        return cls.joules(a, b) / s if s > 0 else float("nan")
