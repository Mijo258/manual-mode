"""Run Phase 1 verification without GUI; use --live for the optional webcam view."""

import argparse
from dataclasses import replace
import multiprocessing
import select
import socket
import threading
import time
from unittest.mock import Mock, patch

import cv2
import numpy as np

from camera_hub import CameraHub
from contracts import ControlPacket, ControlMode, DirectionCommand
from ipc_bus import CommandPublisher, CommandSubscriber


def test_ipc_bus() -> None:
    with CommandSubscriber("127.0.0.1", 0) as subscriber:
        with CommandPublisher("127.0.0.1", subscriber.bound_port, 20) as publisher:
            assert subscriber.poll_latest() is None
            assert subscriber._socket.getblocking() is False
            assert publisher._socket.getblocking() is False
            packet = ControlPacket(1.25, ControlMode.NONE.value, DirectionCommand.STOP.value,
                                   1.0, [], 30.0)
            with patch("ipc_bus.time.monotonic", return_value=100.0) as clock:
                assert publisher.publish(packet)
                assert not publisher.publish(replace(packet, timestamp=2.0, confidence=0.8))
                packet = replace(packet, mode=ControlMode.FACE.value)
                assert publisher.publish(packet), "Mode changes must bypass the throttle."
                packet = replace(packet, command=DirectionCommand.UP.value)
                assert publisher.publish(packet), "Command changes must bypass the throttle."
                packet = replace(packet, active_alarms=["SLEEP"])
                assert publisher.publish(packet), "Alarm changes must bypass the throttle."
                assert not publisher.publish(replace(packet, active_alarms=["SLEEP"]))
                assert publisher.publish(packet, force=True)
                clock.return_value = 100.06
                packet = replace(packet, timestamp=8.75, confidence=0.92, fps=27.5)
                assert publisher.publish(packet), "Unchanged heartbeats resume after the interval."
                clock.return_value = 100.061
                assert not publisher.publish(packet)
                failed_packet = replace(packet, command=DirectionCommand.RIGHT.value)
                with patch.object(publisher, "_socket", Mock()) as failing_socket:
                    failing_socket.sendto.side_effect = BlockingIOError("UDP temporarily busy")
                    assert not publisher.publish(failed_packet)
                assert publisher.publish(failed_packet), "A failed send must not consume a change."
                packet = failed_packet

            # Corruption after valid packets must not erase the newest valid packet.
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as raw_sender:
                target = ("127.0.0.1", subscriber.bound_port)
                for payload in (b"{", b"\xff", b"[]", b'{}',
                                replace(packet, active_alarms="bad").to_json().encode("utf-8"),
                                replace(packet, fps="bad").to_json().encode("utf-8")):
                    raw_sender.sendto(payload, target)
            assert select.select([subscriber._socket], [], [], 0.5)[0], "No loopback UDP arrived."
            # All local sendto calls have completed; allow final datagrams to enter the queue.
            time.sleep(0.02)
            latest = subscriber.poll_latest()
            assert latest == packet, f"Queue did not drain to the exact newest packet: {latest!r}"
            assert subscriber.poll_latest() is None, "Stale packets remained after draining."
            publisher.close()
            publisher.close()
            assert not publisher.publish(packet)
        subscriber.close()
        subscriber.close()
        assert subscriber.poll_latest() is None
    print("Test A PASSED: UDP throttling, immediate changes, corruption handling and latest-only drain.")


class FakeCapture:
    """Numbered VGA frames; consumer acknowledgements pace the next frame."""

    def __init__(self, *args, opened: bool = True):
        self.opened = opened
        self.release_count = 0
        self.properties = {}
        self.frame_count = 0
        self.transient_failures = 2
        self.permit = threading.Event()
        self.permit.set()

    def isOpened(self) -> bool:
        return self.opened

    def set(self, prop, value) -> bool:
        self.properties[prop] = value
        if prop == cv2.CAP_PROP_BUFFERSIZE:
            raise NotImplementedError("Fake driver does not support buffer-size hints.")
        return True

    def read(self):
        if not self.opened:
            return False, None
        if self.transient_failures:
            self.transient_failures -= 1
            return False, None
        if not self.permit.wait(0.05) or not self.opened:
            return False, None
        self.permit.clear()
        self.frame_count += 1
        frame = np.full((480, 640, 3), self.frame_count, dtype=np.uint8)
        frame[0, 0, 1], frame[0, -1, 1] = 11, 99
        return True, frame

    def release(self) -> None:
        self.release_count += 1
        self.opened = False
        self.permit.set()


def test_camera_hub_mock() -> None:
    fake = FakeCapture()
    factory_calls = []

    def factory(*args):
        factory_calls.append(args)
        return fake

    hub = CameraHub({"camera": {"flip_horizontal": True}}, capture_factory=factory)
    assert hub.get_latest(timeout=0) is None
    target_frames = 20
    finished_frame = threading.Barrier(2)
    mutated = threading.Event()
    records = [[], []]
    first_copies = [None, None]
    errors = []

    def consume(index: int) -> None:
        last_id = -1
        try:
            for iteration in range(target_frames):
                packet = hub.get_latest(last_id, timeout=1.0)
                assert packet is not None, "Mock consumer timed out."
                assert packet.frame_id == iteration + 1
                assert packet.frame.dtype == np.uint8 and packet.frame.shape == (480, 640, 3)
                if iteration == 0:
                    first_copies[index] = packet.frame
                if index == 0:
                    assert packet.frame[0, 0, 1] == 99, "Configured horizontal flip was not applied."
                    packet.frame[:] = 0
                    mutated.set()
                else:
                    assert mutated.wait(1.0), "Mutating consumer did not acknowledge this frame."
                    assert packet.frame[1, 1, 0] == packet.frame_id, "Consumer copies share mutations."
                    assert packet.frame[0, 0, 1] == 99
                    mutated.clear()
                records[index].append((packet.frame_id, packet.timestamp_ms))
                last_id = packet.frame_id
                finished_frame.wait(timeout=1.0)
                if index == 1 and iteration + 1 < target_frames:
                    fake.permit.set()
        except BaseException as exc:
            errors.append(exc)
            finished_frame.abort()
            mutated.set()

    ticks = iter(range(1_000_000_000, 2_000_000_000, 100_000))
    with patch("camera_hub.time.monotonic_ns", side_effect=lambda: next(ticks)):
        hub.start()
        capture_thread = hub._thread
        hub.start()
        assert len(factory_calls) == 1, "Repeated start opened a second camera."
        assert factory_calls[0] == (0, cv2.CAP_DSHOW)
        consumers = [threading.Thread(target=consume, args=(i,), daemon=True) for i in range(2)]
        try:
            for thread in consumers:
                thread.start()
            deadline = time.monotonic() + 4.0
            for thread in consumers:
                thread.join(max(0.0, deadline - time.monotonic()))
            assert all(not thread.is_alive() for thread in consumers), "Mock consumers did not finish."
            if errors:
                raise errors[0]
            assert records[0] == records[1] and len(records[0]) >= 15
            assert all(a[1] < b[1] for a, b in zip(records[0], records[0][1:]))
            assert records[0][0][1] == 1000 and records[0][-1][1] == 1019
            assert not np.shares_memory(first_copies[0], first_copies[1])
            assert hub.actual_fps > 0.0
            assert fake.properties[cv2.CAP_PROP_FRAME_WIDTH] == 640
            assert fake.properties[cv2.CAP_PROP_FRAME_HEIGHT] == 480
            assert fake.properties[cv2.CAP_PROP_FPS] == 30
            assert fake.properties[cv2.CAP_PROP_BUFFERSIZE] == 1
            assert hub.get_latest(target_frames, timeout=0.02) is None

            waiter_ready = threading.Event()
            waiter_result = []

            def wait_for_shutdown():
                waiter_ready.set()
                waiter_result.append(hub.get_latest(target_frames, timeout=5.0))

            waiter = threading.Thread(target=wait_for_shutdown, daemon=True)
            waiter.start()
            assert waiter_ready.wait(0.5)
            hub.stop()
            waiter.join(0.5)
            assert not waiter.is_alive() and waiter_result == [None]
        finally:
            hub.stop()
            finished_frame.abort()
    assert not capture_thread.is_alive(), "stop() left the capture thread alive."
    assert not hub.is_running() and fake.release_count == 1
    hub.stop()
    assert fake.release_count == 1, "Repeated stop released the capture twice."
    assert hub.get_latest(target_frames, timeout=0) is None
    assert hub.get_latest(target_frames - 1, timeout=0).frame_id == target_frames

    # Exercise a failed explicit backend followed by a working default backend.
    captures = [FakeCapture(opened=False), FakeCapture()]
    fallback_calls = []

    def fallback_factory(*args):
        fallback_calls.append(args)
        return captures[len(fallback_calls) - 1]

    with CameraHub({"backend": "MSMF", "flip_horizontal": False}, fallback_factory) as fallback:
        packet = fallback.get_latest(timeout=1.0)
        assert packet is not None and packet.frame[0, 0, 1] == 11
        fallback_thread = fallback._thread
    assert fallback_calls == [(0, cv2.CAP_MSMF), (0,)]
    assert not fallback_thread.is_alive() and all(cap.release_count == 1 for cap in captures)
    unopened = [FakeCapture(opened=False), FakeCapture(opened=False)]
    attempts = iter(unopened)
    failed_hub = CameraHub({}, lambda *args: next(attempts))
    try:
        failed_hub.start()
    except RuntimeError as exc:
        assert "webcam 0" in str(exc)
    else:
        raise AssertionError("An unopened webcam must raise a clear RuntimeError.")
    failed_hub.stop()
    assert all(cap.release_count == 1 for cap in unopened)
    print("Test B PASSED: 20 shared frame IDs, isolated copies, strict timestamps and clean shutdown.")


def _webcam_probe_worker(connection, deadline: float) -> None:
    """Use a process so the parent can bound native camera open/read/release calls."""
    hub = CameraHub()
    observations = [[], []]
    errors = []
    result = (False, "Webcam did not produce frames before the deadline.")

    def consume(index: int) -> None:
        last_id = -1
        try:
            while len(observations[index]) < 5 and time.monotonic() < deadline:
                packet = hub.get_latest(last_id, timeout=min(0.15, max(0.0, deadline - time.monotonic())))
                if packet is None:
                    continue
                observations[index].append((packet.frame_id, packet.timestamp_ms, packet.frame.shape))
                last_id = packet.frame_id
        except Exception as exc:
            errors.append(str(exc))

    try:
        hub.start()
        consumers = [threading.Thread(target=consume, args=(i,), daemon=True) for i in range(2)]
        for thread in consumers:
            thread.start()
        for thread in consumers:
            thread.join(max(0.0, deadline - time.monotonic()))
        count = sum(map(len, observations))
        if errors:
            result = (False, "; ".join(errors))
        elif count == 10:
            height, width = observations[0][0][2][:2]
            result = (True, f"{width} x {height}; actual_fps={hub.actual_fps:.1f}; 10 frames across 2 consumers")
        else:
            reason = hub.last_error or f"Only {count}/10 consumer frames arrived before the deadline."
            result = (False, reason)
    except Exception as exc:
        result = (False, str(exc))
    finally:
        hub.stop()
        try:
            connection.send(result)
        finally:
            connection.close()


def test_real_webcam_probe() -> None:
    started = time.monotonic()
    hard_deadline = started + 3.0
    context = multiprocessing.get_context("spawn")
    receiver, sender = context.Pipe(duplex=False)
    process = context.Process(target=_webcam_probe_worker,
                              args=(sender, hard_deadline - 0.3), daemon=True)
    result = (False, "3-second probe deadline reached (camera open/read/cleanup may be unavailable).")
    try:
        process.start()
        sender.close()
        if receiver.poll(max(0.0, hard_deadline - time.monotonic() - 0.2)):
            try:
                result = receiver.recv()
            except EOFError:
                result = (False, "Webcam probe process exited without a result.")
    except (OSError, RuntimeError) as exc:
        result = (False, f"Unable to start webcam probe: {exc}")
    finally:
        receiver.close()
        sender.close()
        if process.pid is not None:
            process.join(max(0.0, min(0.05, hard_deadline - time.monotonic() - 0.1)))
            if process.is_alive():
                process.terminate()
                process.join(max(0.0, hard_deadline - time.monotonic()))
            if process.is_alive():
                process.kill()
            else:
                process.close()
    if result[0]:
        print(f"Test C PASSED: WEBCAM {result[1]}")
    else:
        print(f"WEBCAM SKIPPED: {result[1]}")


def run_live() -> None:
    with CameraHub() as hub:
        last_id = -1
        try:
            while hub.is_running():
                packet = hub.get_latest(last_id, timeout=0.1)
                if packet is not None:
                    last_id = packet.frame_id
                    cv2.putText(packet.frame, f"FPS: {hub.actual_fps:.1f}  frame_id: {last_id}",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.imshow("CameraHub live - q to quit", packet.frame)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
        finally:
            cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="Show the real webcam until q/Esc is pressed.")
    args = parser.parse_args()
    if args.live:
        run_live()
    else:
        test_ipc_bus()
        test_camera_hub_mock()
        test_real_webcam_probe()
        print("PHASE 1 VERIFIED OK")


if __name__ == "__main__":
    main()
