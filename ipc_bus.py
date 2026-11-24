"""Non-blocking UDP command transport, with change detection and queue draining."""

import socket
import time
from typing import Any

import contracts


class CommandPublisher:
    def __init__(self, host: str | None = None, port: int | None = None,
                 max_hz: float | None = None):
        cfg = contracts.load_config() if None in (host, port, max_hz) else {}
        self.host = host if host is not None else cfg.get("ipc", {}).get("host", "127.0.0.1")
        self.port = int(port if port is not None else cfg.get("ipc", {}).get("port", 5055))
        self.max_hz = float(max_hz if max_hz is not None
                            else cfg.get("control", {}).get("command_send_hz", 20))
        if not 0 <= self.port <= 65535 or not 0.0 < self.max_hz < float("inf"):
            raise ValueError("UDP port must be 0-65535 and max_hz must be positive and finite.")
        self._socket: socket.socket | None = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._socket.setblocking(False)
        self._last_signature: tuple | None = None
        self._last_sent = float("-inf")

    def publish(self, packet: contracts.ControlPacket, force: bool = False) -> bool:
        """Send changed commands immediately; throttle only unchanged heartbeats.

        Transient send failures and publishing after close return False. A failed
        send never advances the rate limiter or the last successfully sent state.
        """
        sock = self._socket
        if sock is None:
            return False
        signature = (packet.mode, packet.command, tuple(packet.active_alarms))
        now = time.monotonic()
        if not force and signature == self._last_signature and now - self._last_sent < 1.0 / self.max_hz:
            return False
        payload = packet.to_json().encode("utf-8")
        try:
            if sock.sendto(payload, (self.host, self.port)) != len(payload):
                return False
        except OSError:
            return False
        self._last_sent = now
        self._last_signature = signature
        return True

    def close(self) -> None:
        sock, self._socket = self._socket, None
        if sock is not None:
            sock.close()

    def __enter__(self) -> "CommandPublisher":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


def _valid_packet(packet: contracts.ControlPacket) -> bool:
    # Dataclass annotations alone do not validate decoded JSON field types.
    numbers: tuple[Any, ...] = (packet.timestamp, packet.confidence, packet.fps)
    return (
        all(isinstance(value, (int, float)) and not isinstance(value, bool)
            and float("-inf") < value < float("inf") for value in numbers)
        and isinstance(packet.mode, str)
        and isinstance(packet.command, str)
        and isinstance(packet.active_alarms, list)
        and all(isinstance(alarm, str) for alarm in packet.active_alarms)
    )


class CommandSubscriber:
    def __init__(self, host: str | None = None, port: int | None = None):
        cfg = contracts.load_config()["ipc"] if host is None or port is None else {}
        self.host = host if host is not None else cfg.get("host", "127.0.0.1")
        port = int(port if port is not None else cfg.get("port", 5055))
        if not 0 <= port <= 65535:
            raise ValueError("UDP port must be 0-65535.")
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind((self.host, port))
            sock.setblocking(False)
        except OSError:
            sock.close()
            raise
        self._socket: socket.socket | None = sock
        self._bound_port = sock.getsockname()[1]

    @property
    def bound_port(self) -> int:
        return self._bound_port

    def poll_latest(self) -> contracts.ControlPacket | None:
        """Drain queued datagrams, ignoring bad payloads and keeping the last valid one."""
        sock = self._socket
        if sock is None:
            return None
        latest = None
        while True:
            try:
                payload, _ = sock.recvfrom(4096)
            except BlockingIOError:
                break
            except OSError:
                break  # Includes Windows UDP connection-reset errors and concurrent close.
            try:
                packet = contracts.ControlPacket.from_json(payload.decode("utf-8"))
                if _valid_packet(packet):
                    latest = packet
            except (ValueError, TypeError, OverflowError, RecursionError):
                continue
        return latest

    def close(self) -> None:
        sock, self._socket = self._socket, None
        if sock is not None:
            sock.close()

    def __enter__(self) -> "CommandSubscriber":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
