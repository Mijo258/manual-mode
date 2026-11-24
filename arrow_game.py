"""ASCII arrow arena driven by the project's non-blocking localhost UDP bus."""

import argparse
from collections import deque
import ctypes
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import contracts
import ipc_bus


_RESET = "\x1b[0m"


def _windows_console():
    kernel32 = ctypes.windll.kernel32
    kernel32.GetStdHandle.argtypes = [ctypes.c_uint32]
    kernel32.GetStdHandle.restype = ctypes.c_void_p
    handle = kernel32.GetStdHandle(-11)
    if handle in (None, 0, ctypes.c_void_p(-1).value):
        raise OSError("No Windows console output handle.")
    return kernel32, handle


def enable_windows_ansi() -> bool:
    """Enable VT100 processing on a real console; redirected output is harmless."""
    try:
        if not sys.stdout.isatty():
            return False
        if os.name != "nt":
            return True  # VT100 is already available on ordinary Unix terminals.
        kernel32, handle = _windows_console()
        kernel32.GetConsoleMode.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32)]
        kernel32.GetConsoleMode.restype = ctypes.c_int
        kernel32.SetConsoleMode.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
        kernel32.SetConsoleMode.restype = ctypes.c_int
        mode = ctypes.c_uint32()
        if not kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
            return False
        return bool(kernel32.SetConsoleMode(handle, mode.value | 0x0004))
    except Exception:
        return False


def _ascii_text(value) -> str:
    text = str(value).encode("ascii", "replace").decode("ascii")
    return "".join(char if 32 <= ord(char) <= 126 else "?" for char in text)


class ArrowArena:
    def __init__(self, width: int = 29, height: int = 13, stale_timeout_sec: float = 1.5):
        if not isinstance(width, int) or not isinstance(height, int) or width < 1 or height < 1:
            raise ValueError("Arena width and height must be positive integers.")
        self.width, self.height = width, height
        self.stale_timeout_sec = float(stale_timeout_sec)
        if not math.isfinite(self.stale_timeout_sec) or self.stale_timeout_sec < 0:
            raise ValueError("Stale timeout must be nonnegative and finite.")
        self.x, self.y = width // 2, height // 2
        self.facing = "^"
        self.trail: deque[tuple[int, int]] = deque(maxlen=5)
        self.mode = "NONE"
        self.command = "STOP"
        self.confidence = 0.0
        self.fps = 0.0
        self.active_alarms: list[str] = []
        self.link_alive = False
        self.last_packet_sec: float | None = None
        self.packet_count = 0
        # A 79-column frame avoids autowrap in a standard 80-column console.
        self._screen_width = max(79, width + 2)

    def apply_packet(self, packet: contracts.ControlPacket, now_sec: float | None = None) -> None:
        self.mode = str(packet.mode).strip().upper()
        command = str(packet.command).strip().upper()
        self.command = command if command in {item.value for item in contracts.DirectionCommand} else "STOP"
        self.confidence = float(packet.confidence)
        self.fps = float(packet.fps)
        self.active_alarms = list(packet.active_alarms)
        self.link_alive = True
        self.last_packet_sec = time.monotonic() if now_sec is None else float(now_sec)
        self.packet_count += 1

    def step(self, now_sec: float | None = None) -> None:
        now = time.monotonic() if now_sec is None else float(now_sec)
        if self.last_packet_sec is None or now - self.last_packet_sec > self.stale_timeout_sec:
            self.link_alive = False
            self.command = "STOP"
            return
        movement = {"UP": (0, -1, "^"), "DOWN": (0, 1, "v"),
                    "LEFT": (-1, 0, "<"), "RIGHT": (1, 0, ">")}.get(self.command)
        if movement is None:
            return
        dx, dy, self.facing = movement
        old = self.x, self.y
        self.x = max(0, min(self.width - 1, self.x + dx))
        self.y = max(0, min(self.height - 1, self.y + dy))
        if old != (self.x, self.y):
            self.trail.append(old)

    def reset_position(self) -> None:
        self.x, self.y = self.width // 2, self.height // 2
        self.trail.clear()

    def render(self, use_ansi: bool = True) -> str:
        """Return one fixed-size ASCII screen, with optional ASCII ANSI escapes."""
        screen_width = self._screen_width

        def line(text: str, color: str = "") -> str:
            text = _ascii_text(text)[:screen_width]
            padding = " " * (screen_width - len(text))
            return color + text + _RESET + padding if use_ansi and color else text + padding

        telemetry = (f"MODE: {_ascii_text(self.mode)[:4]} | CMD: {self.command} | POS: ({self.x},{self.y})"
                     f" | CONF: {self.confidence:.2f} | FPS: {self.fps:04.1f}"
                     f" | LINK: {'LIVE' if self.link_alive else 'WAITING'}")
        lines = [line("=== AUTONOMOUS VEHICLE - LANDMARK CONTROL VERIFIER ===".center(screen_width)),
                 line(telemetry)]
        if self.active_alarms:
            banner = "[!!! SAFETY ALARM: " + " | ".join(map(_ascii_text, self.active_alarms)) + " !!!]"
            lines.append(line(banner, "\x1b[1;31m"))
        else:
            lines.append(line("[DMS SAFETY: OK - DRIVER ALERT]", "\x1b[32m"))
        lines.append(line(""))
        indent = " " * ((screen_width - self.width - 2) // 2)
        border = indent + "+" + "-" * self.width + "+"
        lines.append(line(border))
        trail = set(self.trail)
        arrow_color = "\x1b[1;31m" if self.active_alarms else "\x1b[1;36m"
        for y in range(self.height):
            cells = ["." if (x, y) in trail else " " for x in range(self.width)]
            if y == self.y:
                cells[self.x] = self.facing
            plain = indent + "|" + "".join(cells) + "|"
            if use_ansi and y == self.y:
                position = len(indent) + 1 + self.x
                colored = plain[:position] + arrow_color + plain[position] + _RESET + plain[position + 1:]
                lines.append(colored + " " * (screen_width - len(plain)))
            else:
                lines.append(line(plain))
        lines.append(line(border))
        lines.append(line(""))
        lines.append(line("Keys: [R] Reset Center | [Q/Esc] Quit   (Listening on UDP)"))
        return "\n".join(lines)


def launch_arrow_terminal(host: str | None = None, port: int | None = None,
                          python_exe: str | None = None) -> subprocess.Popen:
    script_path = Path(__file__).resolve()
    args = [sys.executable if python_exe is None else python_exe, str(script_path)]
    if host is not None:
        args.extend(("--host", host))
    if port is not None:
        args.extend(("--port", str(port)))
    kwargs = {"cwd": str(script_path.parent)}
    if os.name == "nt":
        kwargs["creationflags"] = subprocess.CREATE_NEW_CONSOLE
    return subprocess.Popen(args, **kwargs)


class _Coord(ctypes.Structure):
    _fields_ = [("x", ctypes.c_short), ("y", ctypes.c_short)]


class _SmallRect(ctypes.Structure):
    _fields_ = [(name, ctypes.c_short) for name in ("left", "top", "right", "bottom")]


class _BufferInfo(ctypes.Structure):
    _fields_ = [("size", _Coord), ("cursor", _Coord), ("attributes", ctypes.c_ushort),
                ("window", _SmallRect), ("maximum_size", _Coord)]


class _CursorInfo(ctypes.Structure):
    _fields_ = [("size", ctypes.c_uint32), ("visible", ctypes.c_int)]


def _native_cursor(console, visible: bool) -> None:
    kernel32, handle = console
    kernel32.GetConsoleCursorInfo.argtypes = [ctypes.c_void_p, ctypes.POINTER(_CursorInfo)]
    kernel32.SetConsoleCursorInfo.argtypes = [ctypes.c_void_p, ctypes.POINTER(_CursorInfo)]
    info = _CursorInfo()
    if kernel32.GetConsoleCursorInfo(handle, ctypes.byref(info)):
        info.visible = int(visible)
        kernel32.SetConsoleCursorInfo(handle, ctypes.byref(info))


def _native_home(console) -> None:
    kernel32, handle = console
    kernel32.SetConsoleCursorPosition.argtypes = [ctypes.c_void_p, _Coord]
    if not kernel32.SetConsoleCursorPosition(handle, _Coord(0, 0)):
        raise OSError("Unable to position the console cursor.")


def _native_setup():
    """Uncolored native-console redraw fallback when VT100 enabling is unavailable."""
    console = kernel32, handle = _windows_console()
    kernel32.GetConsoleScreenBufferInfo.argtypes = [ctypes.c_void_p, ctypes.POINTER(_BufferInfo)]
    info = _BufferInfo()
    if not kernel32.GetConsoleScreenBufferInfo(handle, ctypes.byref(info)):
        raise OSError("Unable to read the console buffer.")
    written = ctypes.c_uint32()
    count = info.size.x * info.size.y
    kernel32.FillConsoleOutputCharacterW.argtypes = [ctypes.c_void_p, ctypes.c_wchar, ctypes.c_uint32,
                                                   _Coord, ctypes.POINTER(ctypes.c_uint32)]
    kernel32.FillConsoleOutputAttribute.argtypes = [ctypes.c_void_p, ctypes.c_ushort, ctypes.c_uint32,
                                                  _Coord, ctypes.POINTER(ctypes.c_uint32)]
    kernel32.FillConsoleOutputCharacterW(handle, " ", count, _Coord(0, 0), ctypes.byref(written))
    kernel32.FillConsoleOutputAttribute(handle, info.attributes, count, _Coord(0, 0), ctypes.byref(written))
    _native_home(console)
    _native_cursor(console, False)
    return console


def _read_key() -> str | None:
    """Read only if a key is ready; ignore Windows extended-key scan codes."""
    if not sys.stdin.isatty():
        return None
    if os.name == "nt":
        import msvcrt

        if not msvcrt.kbhit():
            return None
        key = msvcrt.getwch()
        if key in ("\x00", "\xe0"):
            if msvcrt.kbhit():
                msvcrt.getwch()
            return None
        return key
    import select

    return sys.stdin.read(1) if select.select([sys.stdin], [], [], 0)[0] else None


def run_game(host: str | None = None, port: int | None = None, tick_hz: float = 20.0,
             move_hz: float = 7.0, *, width: int = 29, height: int = 13) -> None:
    if not 0.0 < tick_hz < float("inf") or not 0.0 < move_hz <= tick_hz:
        raise ValueError("Rates must be finite and satisfy 0 < move_hz <= tick_hz.")
    if not sys.stdout.isatty():
        raise RuntimeError("The arrow game needs an interactive terminal; use launch_arrow_terminal().")
    arena = ArrowArena(width, height)
    subscriber = ipc_bus.CommandSubscriber(host=host, port=port)
    ansi_ok = False
    native_console = None
    try:
        ansi_ok = enable_windows_ansi()
        if ansi_ok:
            sys.stdout.write("\x1b[2J\x1b[H\x1b[?25l")
            sys.stdout.flush()
        elif os.name == "nt":
            native_console = _native_setup()
        else:
            raise RuntimeError("The terminal does not support cursor positioning.")
        tick_interval, move_interval = 1.0 / tick_hz, 1.0 / move_hz
        next_tick = next_move = time.monotonic()
        while True:
            now = time.monotonic()
            packet = subscriber.poll_latest()
            if packet is not None:
                arena.apply_packet(packet, now_sec=now)
            if now >= next_move:
                arena.step(now)
                next_move += move_interval
                if next_move <= now:
                    next_move = now + move_interval
            key = _read_key()
            if key in ("q", "Q", "\x1b"):
                break
            if key in ("r", "R"):
                arena.reset_position()
            if native_console is not None:
                _native_home(native_console)
            sys.stdout.write(("\x1b[H" if ansi_ok else "") + arena.render(use_ansi=ansi_ok))
            sys.stdout.flush()
            next_tick = max(next_tick + tick_interval, time.monotonic())
            time.sleep(max(0.0, next_tick - time.monotonic()))
    except KeyboardInterrupt:
        pass
    finally:
        try:
            if ansi_ok:
                sys.stdout.write("\x1b[?25h\x1b[0m\n")
                sys.stdout.flush()
            elif native_console is not None:
                _native_cursor(native_console, True)
                sys.stdout.write("\n")
                sys.stdout.flush()
        finally:
            subscriber.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--tick-hz", type=float, default=20.0)
    parser.add_argument("--move-hz", type=float, default=7.0)
    parser.add_argument("--width", type=int, default=29)
    parser.add_argument("--height", type=int, default=13)
    args = parser.parse_args()
    run_game(args.host, args.port, args.tick_hz, args.move_hz, width=args.width, height=args.height)


if __name__ == "__main__":
    main()
