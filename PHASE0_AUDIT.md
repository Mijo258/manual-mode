# Phase 0 Workspace Audit

Audit date: 2026-11-22. Root: `C:/Users/Lenovo/manual-mode`.
Target supplied by the user: Windows 11, Intel Core i5 (11th generation), 8 GB RAM, one built-in webcam. Hardware was not benchmarked.

## 1. Environment summary

| Component | Observed value |
| --- | --- |
| Python | 3.11.9; MSC v.1938; 64-bit AMD64 |
| Interpreter used by `python` | `manual_venv/Scripts/python.exe` |
| OpenCV (`cv2.__version__`) | 5.1.0 |
| MediaPipe (`mediapipe.__version__`) | 1.2.3 |
| dlib (`dlib.__version__`) | 19.24.9 |
| MediaPipe inference API | Tasks: `mediapipe.tasks.python.vision.FaceLandmarker` and `HandLandmarker` |

Versions were read together in one command using `python -B` and actual package imports. Legacy `mp.solutions` appears in hand documentation/adapters and optional face-mesh connection-table lookups, not as an inference pipeline.

Dependency text files describe different environments and pins; they are not evidence of installed versions. The road traffic-light scripts require the Haar cascade API and their documentation specifies separate OpenCV 4 environments; their compatibility with this active OpenCV 5.1.0 interpreter was not tested. No installation, environment-file change, inference, camera access or audio playback was performed.

## 2. Exact file inventory

All 45 existing project files are listed, including files hidden by `.gitignore`. Cache, Git metadata and virtual-environment contents are excluded (`__pycache__`, `.git`, `venv`/`.venv`, `manual_venv`, `venv_facemesh`, `traffic_venv`, `position_venv`).

Project subdirectories: `face-landmark-project/`, `road-detection/`, `road-detection/traffic-light-detection/`, `road-detection/traffic-light-detection/by-color/`, `road-detection/traffic-light-detection/by-position/`, `road-detection/traffic-sign-detection/`.

Ownership means the file contains the actual capture/display call; standalone harnesses count. Imported processing classes alone do not own these resources.

| File Path | Purpose | Libraries Used | Owns VideoCapture? (Yes/No) | Owns imshow? (Yes/No) |
| --- | --- | --- | --- | --- |
| `.gitignore` | Git exclusion rules | None (text) | No | No |
| `detect_facemesh.py` | 478-point face/iris mesh; --mesh-only selects 468 points | cv2, NumPy, MediaPipe Tasks; stdlib | Yes | Yes |
| `eye_requirements.txt` | Eye tracker dependency declarations | None (text) | No | No |
| `eye_tracker.py` | 68-point dlib eyelid EAR/drowsiness detector and webcam demo | cv2, dlib, NumPy, SciPy; stdlib | Yes | Yes |
| `face_landmarker.task` | MediaPipe face/iris model bundle (478 points) | Consumed by MediaPipe Tasks | No | No |
| `face-landmark-project/detect_5point.py` | Five-point dlib face landmarks (eye corners and nose) | cv2, dlib, NumPy, os | Yes | Yes |
| `face-landmark-project/detect_68point.py` | 68-point dlib face landmarks; optional recording/JSONL | cv2, dlib, NumPy; stdlib | Yes | Yes |
| `face-landmark-project/detect_facemesh.py` | Face mesh demo; byte-identical to root copy | cv2, NumPy, MediaPipe Tasks; stdlib | Yes | Yes |
| `face-landmark-project/face_landmarker.task` | SafetyDaemon default face/iris model; identical to root bundle | Consumed by MediaPipe Tasks | No | No |
| `face-landmark-project/LANDMARKS_JSON` | Existing landmark JSON Lines output | None (data) | No | No |
| `face-landmark-project/lm.jsonl` | Existing landmark JSON Lines output | None (data) | No | No |
| `face-landmark-project/out.mp4` | Existing video asset; provenance not established | None (data) | No | No |
| `face-landmark-project/output.mp4` | Existing video asset; provenance not established | None (data) | No | No |
| `face-landmark-project/README.md` | Face landmark model, usage and topology documentation | None (Markdown) | No | No |
| `face-landmark-project/requirements.txt` | dlib face demo dependency declarations | None (text) | No | No |
| `face-landmark-project/requirements-facemesh.txt` | Face mesh dependency declarations | None (text) | No | No |
| `face-landmark-project/shape_predictor_5_face_landmarks.dat` | Default five-point face predictor for detect_5point.py | Consumed by dlib | No | No |
| `face-landmark-project/shape_predictor_68_face_landmarks.dat` | Default 68-point face predictor for detect_68point.py | Consumed by dlib | No | No |
| `face-landmark-project/shape_predictor_68_face_landmarks.dat.bz2` | Compressed 68-point predictor archive; not loadable as .dat | None (archive) | No | No |
| `face-landmark-project/test1.mp4` | Existing video asset; provenance not established | None (data) | No | No |
| `face-landmark-project/test2.mp4` | Existing video asset; provenance not established | None (data) | No | No |
| `hand_landmarker.task` | MediaPipe hand model bundle (21 points per hand) | Consumed by MediaPipe Tasks | No | No |
| `hand_requirements.txt` | Hand tracker dependency declarations | None (text) | No | No |
| `hand_tracker.py` | Reusable 21-point hand detector, angle geometry and webcam demo | cv2, NumPy, MediaPipe Tasks; stdlib | Yes | Yes |
| `master_controller.py` | Threaded camera owner, safety processing, FPS display and alarms | cv2, NumPy, safety_daemon; winsound, stdlib | Yes | Yes |
| `requirements.txt` | dlib face demo dependency declarations | None (text) | No | No |
| `requirements-facemesh.txt` | Master/MediaPipe Tasks dependency declarations | None (text) | No | No |
| `road-detection/traffic-light-detection/by-color/EXECUTION_GUIDE.md` | Color-based traffic light setup and usage guide | None (Markdown) | No | No |
| `road-detection/traffic-light-detection/by-color/requirements.txt` | Color traffic light dependency declarations | None (text) | No | No |
| `road-detection/traffic-light-detection/by-color/test_traffic_light_detector.py` | Color/ROI/rendering regression tests; not executed | cv2, NumPy, traffic_light_detector; unittest | No | No |
| `road-detection/traffic-light-detection/by-color/traffic_light_cascade.LICENSE.txt` | License for the color detector cascade asset | None (text) | No | No |
| `road-detection/traffic-light-detection/by-color/traffic_light_cascade.xml` | Pretrained traffic light cascade asset | Consumed by cv2.CascadeClassifier | No | No |
| `road-detection/traffic-light-detection/by-color/traffic_light_detector.py` | Haar housing detection and HSV lamp classification; webcam demo | cv2, NumPy; stdlib | Yes | Yes |
| `road-detection/traffic-light-detection/by-position/README.md` | Position-based traffic light setup and usage guide | None (Markdown) | No | No |
| `road-detection/traffic-light-detection/by-position/requirements.txt` | Position traffic light dependency declarations | None (text) | No | No |
| `road-detection/traffic-light-detection/by-position/traffic_light_cascade.xml` | Pretrained traffic light cascade asset | Consumed by cv2.CascadeClassifier | No | No |
| `road-detection/traffic-light-detection/by-position/traffic_light_position.py` | Haar housing detection and brightness/vertical-position classification | cv2, NumPy; stdlib | Yes | Yes |
| `road-detection/traffic-sign-detection/traffic_sign_arrows.py` | Arrow/sign template classification; delegates I/O to circles.run | cv2, NumPy, traffic_sign_circles; argparse | No | No |
| `road-detection/traffic-sign-detection/traffic_sign_circles.py` | Color/Hough circle and STOP detection; image/video/camera runner | cv2, NumPy; argparse, os, sys | Yes | Yes |
| `safety_daemon.py` | Frame-based 478-point EAR/MAR/iris safety processing | cv2, NumPy, MediaPipe Tasks (dynamic); stdlib | No | No |
| `selftest.py` | Hand topology/geometry/extraction checks and real-graph check; not executed | NumPy, hand_tracker; math, typing | No | No |
| `shape_predictor_5_face_landmarks.dat` | Root five-point face predictor; identical to nested copy | Consumed by dlib | No | No |
| `shape_predictor_68_face_landmarks.dat` | Eye tracker default 68-point predictor; identical to nested copy | Consumed by dlib | No | No |
| `shape_predictor_68_face_landmarks.dat.bz2` | Compressed 68-point predictor archive; identical to nested archive | None (archive) | No | No |
| `test_safety_daemon.py` | Safety geometry/alarms/controller tests with mocked I/O; not executed | NumPy, master_controller, safety_daemon; unittest, stdlib | No | No |
| `PHASE0_AUDIT.md` | New Phase 0 audit report | None (Markdown) | No | No |
| `project_config.json` | New Phase 0 configuration defaults and asset paths | None (JSON) | No | No |
| `contracts.py` | New enums, status/packet dataclasses and JSON configuration loader | dataclasses, enum, json, pathlib only | No | No |

`traffic_sign_arrows.py` delegates camera/display operations to `traffic_sign_circles.run`; its command-line video mode therefore uses the same resource owner. The root and nested face-mesh scripts and corresponding model/predictor copies have matching SHA-256 hashes.

## 3. Discovered asset paths

Paths below are relative to the project root.

| Exact relative path | Role |
| --- | --- |
| `face_landmarker.task` | Root `detect_facemesh.py` default; 478 points, or first 468 with `--mesh-only` |
| `face-landmark-project/face_landmarker.task` | `safety_daemon.py` and nested face-mesh demo default |
| `hand_landmarker.task` | `hand_tracker.py` default; 21 points per hand |
| `shape_predictor_5_face_landmarks.dat` | Existing root five-point face predictor; selected for config |
| `face-landmark-project/shape_predictor_5_face_landmarks.dat` | Nested five-point face demo default |
| `shape_predictor_68_face_landmarks.dat` | `eye_tracker.py` default; selected for config |
| `face-landmark-project/shape_predictor_68_face_landmarks.dat` | Nested 68-point face demo default |
| `shape_predictor_68_face_landmarks.dat.bz2` | Root compressed predictor archive |
| `face-landmark-project/shape_predictor_68_face_landmarks.dat.bz2` | Nested compressed predictor archive |
| `road-detection/traffic-light-detection/by-color/traffic_light_cascade.xml` | Color traffic-light model |
| `road-detection/traffic-light-detection/by-position/traffic_light_cascade.xml` | Position traffic-light model |

No dedicated five-point eye script/model exists. `eye_tracker.py` uses the 68-point face predictor and eyelid indices 36-41/42-47. The five-point face predictor is a different topology (four eye corners plus nose) and cannot supply six-point eyelid EAR.

No project `.wav` or `.mp3` files exist. All three alarm-sound config paths are therefore `null`. Existing alarms are generated with `winsound.Beep`, with a terminal-bell fallback. The face/hand Tasks paths are documented here; the requested config schema contains only dlib and audio asset keys.

## 4. Camera, loop and display call map

Line numbers refer to the inspected existing source. Face/68-point camera helpers select `cv2.CAP_DSHOW` on Windows and otherwise `cv2.CAP_ANY`. All selectable webcam indices default to 0.

| Source | Camera opening call | Existing loop | Display call |
| --- | --- | --- | --- |
| `master_controller.py` | L92: `cv2.VideoCapture(src)` | L151 capture and L254 main: `while True` | L286: `cv2.imshow("Master AV Controller", frame)` |
| `safety_daemon.py` | None | None | None (draws on supplied frame) |
| `detect_facemesh.py` | L375-376: `cv2.VideoCapture(index, backend)` / `cv2.VideoCapture(index)` | L512: `while True` | L558: `cv2.imshow(window, frame)` |
| `face-landmark-project/detect_facemesh.py` | L375-376: same helper calls | L512: `while True` | L558: `cv2.imshow(window, frame)` |
| `face-landmark-project/detect_5point.py` | L23: `cv2.VideoCapture(0)` | L31: `while True` | L66: `cv2.imshow("5-Point Facial Landmark Detection", frame)` |
| `face-landmark-project/detect_68point.py` | L351-352: `cv2.VideoCapture(index, backend)` / `cv2.VideoCapture(index)` | L423: `while True` | L469: `cv2.imshow(window, frame)` |
| `hand_tracker.py` | L1021: `cv2.VideoCapture(args.camera)` | L1068: `while not quit_requested` | L1124: `cv2.imshow("Hand Tracker - 21 landmarks (q to quit)", frame)` |
| `eye_tracker.py` | L537: `cv2.VideoCapture(args.camera)` | L560: `while True` | L646: `cv2.imshow("DMS - Eye Landmark & Drowsiness Tracker", annotated)` |
| `road-detection/traffic-light-detection/by-color/traffic_light_detector.py` | L731: `cv2.VideoCapture(0)` | L743: `while True` | L770: `cv2.imshow(window_name, annotated)` |
| `road-detection/traffic-light-detection/by-position/traffic_light_position.py` | L281: `cv2.VideoCapture(0)` | L291: `while True` | L317: `cv2.imshow("Traffic Light Position Detection - press q to quit", annotated)` |
| `road-detection/traffic-sign-detection/traffic_sign_circles.py` | L308: `cv2.VideoCapture(int(src) if src.isdigit() else src)` | L312: `while True` | L324 video / L339 image: `cv2.imshow(..., res)` |

## 5. Brief technical summary

- `master_controller.py` owns one capture per streamer, reads continuously on a background thread, protects the latest frame with a lock and gives consumers copies with frame IDs to avoid processing the same frame twice.
- Its main loop invokes `SafetyDaemon.process_frame`, routes alarms to the notifier, overlays smoothed FPS and displays one window; q/Esc and error paths clean up resources. Face/hand/eye mode routing and network command transmission are not implemented.
- `AlarmNotifier` starts a daemon audio thread and enforces a 2.0-second cooldown. Two `winsound.Beep` calls block that thread for 250 ms each; capture/main processing continues. `--test-alarm` invokes sound synchronously. No `playsound` or file-audio calls were found.
- `safety_daemon.py` loads the nested face Tasks bundle in VIDEO mode, tracks one face with detection/presence/tracking confidences of 0.5 and enforces increasing millisecond timestamps; it requires 478 landmarks and owns neither capture nor display.
- EAR averages six-point eye ratios using indices (33,160,158,133,153,144) and (362,385,387,263,373,380); MAR is distance(13,14)/distance(78,308). Both use pixel coordinates. Exact existing thresholds 0.20 and 0.50 are preserved in config.
- Gaze projects iris centers 468/473 onto each eye-corner axis. Both eyes must be below 0.25 or above 0.75 on the same side; those boundaries are preserved. The daemon currently exposes a Boolean `gaze_off_center`, not the scalar `gaze_ratio` required by the new contract.
- Alarms activate after 20 sleepy, 15 yawning or 30 distracted consecutive processed frames; missing/invalid observations reset counters. No duration-in-seconds settings exist. Config keeps the requested 1.5/1.5/2.0-second defaults; these are future settings, not a conversion of frame limits, whose elapsed duration varies with processing FPS.
- Standalone demos independently open webcam 0 and own GUI loops; launching them together can compete for the single webcam. Only face-mesh/68-point helpers explicitly choose DSHOW. The master does not configure 640x480/30 FPS or mirror frames; the new camera defaults are not yet wired into existing code.
- Existing alarm labels are `SLEEPING`, `YAWNING`, `DISTRACTED`; contract examples use `SLEEP`, `YAWN`, `DISTRACTION`. This needs an explicit mapping during later integration. The separate dlib eye tracker uses EAR 0.25/20 frames; config follows the safety daemon's 0.20 threshold.
- Default model paths are based on each script's directory; no hardcoded machine-absolute model/audio path was found in runtime code. Root/nested duplicate assets are present. `hand_tracker.py` uses Tasks despite legacy-shaped adapters; its accepted `model_complexity`/`smooth_landmarks` options have no effect under Tasks.

## 6. Phase 0 interface scope

`contracts.py` defines the requested control enums, `SafetyStatus` and `ControlPacket` fields, packet JSON round-trip methods and `load_config`. Relative config paths resolve from that module's directory. It has only standard-library imports and performs no camera, model or audio initialization.

`project_config.json` retains the exact safety thresholds and existing 2.0-second alarm cooldown, uses the requested Windows/VGA camera, loopback IPC and control defaults, and references existing root predictors. Existing scripts do not yet read this configuration or use the contracts. Only the three requested files are created; existing scripts/assets are left untouched.
