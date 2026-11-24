"""Headless arrow-game verification; --demo opens the optional manual UDP demo."""

import argparse
from contextlib import ExitStack
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from types import SimpleNamespace
from unittest.mock import Mock, patch

import arrow_game
import contracts
from ipc_bus import CommandPublisher, CommandSubscriber


def _packet(command: str, alarms: list[str] | None = None, mode: str = "HAND") -> contracts.ControlPacket:
    return contracts.ControlPacket(0.0, mode, command, 0.93, [] if alarms is None else alarms, 30.0)


def test_movement_clamping_trail_reset() -> None:
    arena = arrow_game.ArrowArena()
    center = arena.width // 2, arena.height // 2
    assert (arena.x, arena.y) == center and arena.facing == "^"
    x, y = center
    for command, expected in (("UP", (x, y - 1, "^")), ("RIGHT", (x + 1, y - 1, ">")),
                              ("DOWN", (x + 1, y, "v")), ("LEFT", (x, y, "<")),
                              ("STOP", (x, y, "<"))):
        arena.apply_packet(_packet(command), now_sec=100.0)
        arena.step(now_sec=100.1)
        assert (arena.x, arena.y, arena.facing) == expected
    assert len(arena.trail) == 4 and arena.trail[0] == center and arena.packet_count == 5
    for command, edge in (("UP", (x, 0)), ("RIGHT", (arena.width - 1, 0)),
                          ("DOWN", (arena.width - 1, arena.height - 1)), ("LEFT", (0, arena.height - 1))):
        arena.apply_packet(_packet(command), now_sec=100.0)
        for _ in range(50):
            arena.step(now_sec=100.1)
        assert (arena.x, arena.y) == edge and len(arena.trail) <= 5
        trail = list(arena.trail)
        arena.step(now_sec=100.1)
        assert list(arena.trail) == trail, "Boundary clamping created a phantom trail cell."
    arena.reset_position()
    assert (arena.x, arena.y) == center and not arena.trail
    arena.apply_packet(_packet("unknown"), now_sec=101.0)
    arena.step(now_sec=101.1)
    assert arena.command == "STOP" and (arena.x, arena.y) == center
    alarms = ["YAWN"]
    arena.apply_packet(_packet("up", alarms, mode="face"), now_sec=102.0)
    alarms.append("SLEEP")
    assert arena.mode == "FACE" and arena.command == "UP" and arena.active_alarms == ["YAWN"]
    print("Test 1 PASSED: movement, facing, all boundaries, bounded trail and reset.")


def test_stale_watchdog() -> None:
    arena = arrow_game.ArrowArena()
    arena.step(now_sec=0.0)
    assert arena.link_alive is False and arena.command == "STOP"
    arena.apply_packet(_packet("RIGHT"), now_sec=10.0)
    initial = arena.x, arena.y
    arena.step(now_sec=10.2)
    assert (arena.x, arena.y) == (initial[0] + 1, initial[1]) and arena.link_alive is True
    before_stale = arena.x, arena.y
    arena.step(now_sec=12.0)
    assert arena.link_alive is False and arena.command == "STOP"
    assert (arena.x, arena.y) == before_stale and arena.facing == ">"
    arena.apply_packet(_packet("UP"), now_sec=12.1)
    arena.step(now_sec=12.2)
    assert arena.link_alive is True and (arena.x, arena.y) == (before_stale[0], before_stale[1] - 1)
    boundary = arrow_game.ArrowArena()
    boundary.apply_packet(_packet("RIGHT"), now_sec=10.0)
    boundary.step(now_sec=11.5)
    assert boundary.link_alive is True
    boundary.step(now_sec=11.5001)
    assert boundary.link_alive is False and boundary.command == "STOP"
    print("Test 2 PASSED: stale link stops motion and a new packet restores the link.")


def test_ascii_rendering() -> None:
    arena = arrow_game.ArrowArena()
    ansi_pattern = re.compile(r"\x1b\[[0-9;]*m")
    for colored in (False, True):
        rendered = arena.render(use_ansi=colored)
        assert "DMS SAFETY: OK" in rendered and "LINK: WAITING" in rendered
        rendered.encode("ascii")
        visible = ansi_pattern.sub("", rendered)
        assert len({len(line) for line in visible.splitlines()}) == 1
        if not colored:
            assert "\x1b" not in rendered
        arena.apply_packet(_packet("RIGHT", ["SLEEP", "DISTRACTION"]), now_sec=10.0)
        rendered = arena.render(use_ansi=colored)
        assert "SAFETY ALARM" in rendered and "SLEEP" in rendered and "DISTRACTION" in rendered
        rendered.encode("ascii")
        assert len({len(line) for line in ansi_pattern.sub("", rendered).splitlines()}) == 1
        if colored:
            assert "\x1b[1;31m" in rendered
        arena.active_alarms = []
        arena.link_alive = False
    arena.apply_packet(_packet("STOP", ["SLEEP\x1b[2J", "DISTRACTION\u2605"], mode="F\u00c4CE"), now_sec=10.0)
    rendered = arena.render(use_ansi=False)
    rendered.encode("ascii")
    assert "\x1b" not in rendered, "Packet text injected terminal control characters."

    # Redirected/broken stdout must not enable or raise from console APIs.
    with patch.object(arrow_game.sys, "stdout", SimpleNamespace(isatty=lambda: False)):
        assert arrow_game.enable_windows_ansi() is False
    with patch.object(arrow_game.sys, "stdout", SimpleNamespace(isatty=Mock(side_effect=OSError("redirected")))):
        assert arrow_game.enable_windows_ansi() is False
    kernel = SimpleNamespace(GetStdHandle=Mock(return_value=0x12345678),
                             GetConsoleMode=Mock(), SetConsoleMode=Mock(return_value=1))

    def get_mode(handle, output):
        output._obj.value = 0x0001
        return 1

    kernel.GetConsoleMode.side_effect = get_mode
    with patch.object(arrow_game, "os", SimpleNamespace(name="nt")), \
            patch.object(arrow_game.ctypes, "windll", SimpleNamespace(kernel32=kernel), create=True), \
            patch.object(arrow_game.sys, "stdout", SimpleNamespace(isatty=lambda: True)):
        assert arrow_game.enable_windows_ansi() is True
    kernel.SetConsoleMode.assert_called_once_with(0x12345678, 0x0005)
    print("Test 3 PASSED: ASCII-only padded screens, alarm banners and safe ANSI enabling.")


def _receive(subscriber: CommandSubscriber) -> contracts.ControlPacket:
    deadline = time.monotonic() + 0.5
    while time.monotonic() < deadline:
        packet = subscriber.poll_latest()
        if packet is not None:
            return packet
        time.sleep(0.001)
    raise AssertionError("No loopback UDP packet arrived within 0.5 seconds.")


def test_udp_and_launcher() -> None:
    arena = arrow_game.ArrowArena()
    x, y = arena.x, arena.y
    with CommandSubscriber("127.0.0.1", 0) as subscriber:
        with CommandPublisher("127.0.0.1", subscriber.bound_port) as publisher:
            for packet, position in ((_packet("UP"), (x, y - 1)),
                                      (_packet("RIGHT"), (x + 1, y - 1)),
                                      (_packet("RIGHT", ["YAWN"], "EYE"), (x + 2, y - 1))):
                assert publisher.publish(packet)
                received = _receive(subscriber)
                assert received == packet
                arena.apply_packet(received, now_sec=100.0)
                arena.step(now_sec=100.1)
                assert (arena.x, arena.y) == position
                assert arena.mode == packet.mode and arena.command == packet.command
                assert arena.active_alarms == packet.active_alarms
                assert arena.confidence == packet.confidence and arena.fps == packet.fps
            assert arena.active_alarms == ["YAWN"] and arena.packet_count == 3

    # Popen is mocked: validate launcher arguments without ever creating a window.
    child = object()
    with patch("arrow_game.subprocess.Popen", return_value=child) as launch:
        assert arrow_game.launch_arrow_terminal(host="127.0.0.1", port=5056) is child
    args, kwargs = launch.call_args
    script = Path(arrow_game.__file__).resolve()
    assert args[0] == [sys.executable, str(script), "--host", "127.0.0.1", "--port", "5056"]
    assert kwargs["cwd"] == str(script.parent)
    if os.name == "nt":
        assert kwargs["creationflags"] == subprocess.CREATE_NEW_CONSOLE
    else:
        assert "creationflags" not in kwargs
    print("Test 4 PASSED: UDP-to-arena state/position and launcher arguments without a window.")


def _terminate_game(game: subprocess.Popen) -> None:
    if game.poll() is not None:
        return
    game.terminate()
    try:
        game.wait(timeout=2.0)
    except subprocess.TimeoutExpired:
        game.kill()
        game.wait(timeout=1.0)


def run_demo() -> None:
    if os.name != "nt" or not sys.stdin.isatty() or not sys.stdout.isatty():
        raise RuntimeError("The keyboard demo requires an interactive Windows terminal.")
    game = None
    command = "STOP"
    active = []
    controls = {"w": "UP", "a": "LEFT", "s": "DOWN", "d": "RIGHT", " ": "STOP"}
    alarm_keys = {"1": "SLEEP", "2": "YAWN", "3": "DISTRACTION"}
    print("Manual UDP demo: W/A/S/D move, Space STOP, 1/2/3 toggle alarms, Q quit.")
    with CommandPublisher() as publisher:
        try:
            game = arrow_game.launch_arrow_terminal()
            while game.poll() is None:
                key = arrow_game._read_key()
                if key is not None:
                    key = key.lower()
                if key in ("q", "\x1b"):
                    break
                changed = False
                if key in controls:
                    command, changed = controls[key], True
                elif key in alarm_keys:
                    alarm = alarm_keys[key]
                    active.remove(alarm) if alarm in active else active.append(alarm)
                    changed = True
                packet = contracts.ControlPacket(time.monotonic(), contracts.ControlMode.NONE.value,
                                                 command, 1.0, list(active), 20.0)
                publisher.publish(packet, force=changed)
                time.sleep(0.05)
        except KeyboardInterrupt:
            pass
        finally:
            publisher.publish(contracts.ControlPacket(time.monotonic(), "NONE", "STOP", 1.0, [], 20.0), force=True)
            if game is not None:
                _terminate_game(game)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo", action="store_true", help="Open the interactive Windows terminal demo.")
    args = parser.parse_args()
    if args.demo:
        run_demo()
        return
    with ExitStack() as guards:
        guards.enter_context(patch("arrow_game.subprocess.Popen", side_effect=AssertionError("Automated console launch.")))
        guards.enter_context(patch("builtins.input", side_effect=AssertionError("Automated keyboard wait.")))
        guards.enter_context(patch("arrow_game._read_key", side_effect=AssertionError("Automated keyboard polling.")))
        test_movement_clamping_trail_reset()
        test_stale_watchdog()
        test_ascii_rendering()
        test_udp_and_launcher()
    print("PHASE 4 VERIFIED OK")


if __name__ == "__main__":
    main()
