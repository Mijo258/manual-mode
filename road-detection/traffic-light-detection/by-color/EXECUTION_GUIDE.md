# Traffic-light detector execution guide

Use Python 3.11, a webcam, and a desktop session for OpenCV's preview window.
Run the commands below from the `by-color` directory. The only third-party
dependencies are pinned in `requirements.txt`:

```text
opencv-python==4.14.0.94
numpy==2.4.6
```

The Windows `opencv-python==5.0.0.93` package lacks
`cv2.CascadeClassifier`. OpenCV 4.14 retains the Haar cascade API. Pinning the
version prevents pip from selecting the incompatible OpenCV 5 package again.
NumPy 2.4.6 requires Python 3.11 or newer and satisfies this OpenCV wheel's
NumPy requirement. The existing `traffic_venv` uses Python 3.11.9.

## 1. Use the supplied pretrained cascade

`traffic_light_cascade.xml` is included beside the detector. It is an unmodified
copy of `haar_xml_07_19.xml` from
[MuhanadNabeel/TrafficLight-Detection](https://github.com/MuhanadNabeel/TrafficLight-Detection),
whose README and C++ example publish and use this model for traffic-light detection.
The original `mvirgo/Traffic-Light-Classifier` download URL returned HTTP 404.

Model provenance:

- Source revision: [`0ab7461664457d53636ccc3e2e3767705a6860d0`](https://github.com/MuhanadNabeel/TrafficLight-Detection/tree/0ab7461664457d53636ccc3e2e3767705a6860d0).
- Original file: [`haar_xml_07_19.xml`](https://github.com/MuhanadNabeel/TrafficLight-Detection/blob/0ab7461664457d53636ccc3e2e3767705a6860d0/haar_xml_07_19.xml).
- Format: OpenCV legacy Haar XML, 32 by 48 pixel training window, 26 stages,
  and 625 weak classifiers.
- Size: 266,916 bytes.
- SHA-256: `09b2fa233114c28c0e00289c472184f21e66d8fa9cc2ba73383d4cecaecf635a`.
- Upstream license: GNU LGPL v3, copied verbatim to
  [traffic_light_cascade.LICENSE.txt](traffic_light_cascade.LICENSE.txt).

The legacy XML element is named `myfacedetector` in the upstream file. That
internal name is preserved; the model's traffic-light purpose is documented by
the upstream project and its `objectDetection.cpp` example. No weights, XML
elements, or thresholds have been changed.

The XML structure and repository blob hash were checked without running the
detector. OpenCV loading and webcam testing are manual commands below. No camera
accuracy or FPS measurement has been performed.

To restore the same model manually, run this command in Git Bash:

```bash
curl --fail --location --output traffic_light_cascade.xml "https://raw.githubusercontent.com/MuhanadNabeel/TrafficLight-Detection/0ab7461664457d53636ccc3e2e3767705a6860d0/haar_xml_07_19.xml"
```

`--fail` returns an error for HTTP failures instead of saving a `404: Not Found`
response as the model. Startup validates the XML before passing it to OpenCV,
then checks that OpenCV loaded a nonempty classifier. The demo does not download
or train models automatically.

Expected files:

```text
by-color/
    traffic_light_detector.py
    traffic_light_cascade.xml
    traffic_light_cascade.LICENSE.txt
    test_traffic_light_detector.py
    requirements.txt
    EXECUTION_GUIDE.md
```

The default XML path is resolved relative to the Python file, so it also works
when the module is imported from another working directory. A missing file raises
a descriptive `FileNotFoundError`; an invalid or empty cascade raises `ValueError`.
The webcam demo prints the error and exits with a nonzero status.

## 2. Create and activate a virtual environment

If `traffic_venv` already exists, activate it and skip the creation command.
To create a new environment with the installed Python 3.11 interpreter:

```text
py -3.11 -m venv traffic_venv
```

Activate it in **Windows Git Bash** (the terminal showing `MINGW64`):

```bash
source traffic_venv/Scripts/activate
```

Or activate it in **Windows PowerShell**:

```powershell
.\traffic_venv\Scripts\Activate.ps1
```

If activation is unavailable, use the environment's interpreter directly. These
commands work from the project directory in Git Bash and PowerShell:

```text
./traffic_venv/Scripts/python.exe -m pip install --only-binary=:all: -r requirements.txt
./traffic_venv/Scripts/python.exe traffic_light_detector.py
```

The direct-interpreter commands replace steps 3 and 4 when activation is blocked.

## 3. Install the lightweight dependencies

With the virtual environment activated:

```text
python -m pip install --only-binary=:all: -r requirements.txt
```

The exact pins replace an incompatible OpenCV version automatically. The
`--only-binary=:all:` option uses prebuilt wheels and prevents source builds.

## 4. Run the webcam demo

Before opening the camera, verify that the supplied cascade loads:

```text
python -c "from traffic_light_detector import TrafficLightDetector; TrafficLightDetector(); print('Cascade loaded successfully')"
```

After that check succeeds:

```text
python traffic_light_detector.py
```

The demo opens webcam index `0` and uses one fixed sequence:

1. Find the traffic-light housing with Haar and draw a thin neutral outer box.
2. Examine only that housing crop to determine which lamp is illuminated.
3. Draw one colored inner box around the active lamp and display exactly one
   message: `RED is on`, `YELLOW is on`, or `GREEN is on`.

If the evidence is dim or ambiguous, the housing retains its outer box without
an inner box or an on-state message. Duplicate housing detections are suppressed.
Colors outside a detected housing are never classified. If Haar finds no housing,
there are no color results. Present the complete traffic-light unit to the camera.

When the initial grayscale Haar scan finds no housing, one additional scan uses
histogram-normalized `max(B, G, R)` intensity. Standard grayscale weights red less
than green; the retry gives equally strong red, yellow, and green lamps the same
intensity. It uses the same cascade and neighbor threshold. It does not classify
colors outside housing boxes. The second scan costs extra time on missed frames;
the preview FPS includes that work. No previous box or state is reused after a miss.

The only additional preview text is the small FPS display. There are no candidate
counters, pixel-count labels, or separate color modes. Run the script without the
previous mode or sensitivity command-line arguments. The direct command is:

```bash
./traffic_venv/Scripts/python.exe traffic_light_detector.py
```

Focus the preview window and press **q** to quit. Camera and window resources are
also released if an exception occurs or you interrupt the demo with Ctrl+C.
If the camera cannot open, check OS camera permissions and close other applications
using it. If a camera frame cannot be read, the demo exits with a clear error.

### Capture an actual camera failure

While the preview has focus, press **s** with the misclassified lamp on. This
saves an untouched camera frame as PNG and a matching JSON report in `diagnostics/`.
The terminal prints the saved report path; the preview gains no extra labels.
Nothing is saved until you press the key. You can choose another output directory:

```bash
./traffic_venv/Scripts/python.exe traffic_light_detector.py --snapshot-dir diagnostics
```

The report includes housing coordinates, selected states, per-color hue ranges,
pixel counts, brightness, saturation, and the current thresholds. A `housing_count`
of zero identifies a Haar localization failure. A housing with `UNKNOWN` or the
wrong color identifies a classification problem. The raw PNG preserves the camera
evidence needed to investigate that problem, without overlay colors contaminating
the pixels. Capture one example with each lamp on under the same lighting to
check camera behavior; synthetic tests alone cannot establish camera precision.

After finishing, leave an activated environment with:

```powershell
deactivate
```

## Repair a missing `cv2.CascadeClassifier`

The `opencv-python==5.0.0.93` Windows wheel imports as `cv2` but does not provide
`CascadeClassifier`. Reinstalling unpinned requirements selects that same wheel.
The corrected requirements select `opencv-python==4.14.0.94`, whose Python API
includes `CascadeClassifier` and `detectMultiScale`.

Run these commands manually in your current **Git Bash** terminal. They target
`traffic_venv` explicitly, so shell activation cannot select the wrong interpreter:

```bash
./traffic_venv/Scripts/python.exe -m pip install --only-binary=:all: -r requirements.txt
./traffic_venv/Scripts/python.exe -c "import cv2, numpy; print('OpenCV:', cv2.__version__); print('NumPy:', numpy.__version__); print('Loaded from:', cv2.__file__); print('Haar API:', cv2.CascadeClassifier)"
```

Expect OpenCV `4.14.x`, NumPy `2.4.6`, and a printed `CascadeClassifier` class.
There is no need for another uninstall/reinstall cycle of the same OpenCV 5 wheel.

The supplied cascade is ready for the manual loading check. If the XML is missing
or contains `404: Not Found`, restore it from the pinned source in step 1. Then
check it and open the camera:

```bash
./traffic_venv/Scripts/python.exe -c "from traffic_light_detector import TrafficLightDetector; TrafficLightDetector(); print('Cascade loaded successfully')"
./traffic_venv/Scripts/python.exe traffic_light_detector.py
```

If an API is still missing with OpenCV 4, startup reports the loaded version,
Python executable, and `cv2` location. A location pointing to a project-local
`cv2.py` or `cv2` directory outside `site-packages` indicates import shadowing;
rename that local module before retrying. Keep only one OpenCV distribution
installed because the standard, contrib, and headless packages share `cv2`.

All diagnostics, installation commands, and camera tests in this guide are for
manual execution. The detector never invokes pip or modifies the environment.

## Classification behavior

The supported colors are **red, yellow, and green**. The inclusive OpenCV uint8
hue ranges are:

| State | Hue range (0–179) |
| --- | --- |
| RED | 0–9 or 165–179 |
| YELLOW | 10–38, including warm amber |
| GREEN | 39–95 |

Inside each housing, color evidence requires saturation of at least 8 and at
least 6 levels of separation between the largest and smallest BGR channels.
The support brightness floor is 24. These masks preserve a dim colored rim even
when the central lamp pixels are much brighter. White and gray have no hue votes.

Warm amber belongs to yellow. Red still uses both ends of the hue scale; the
green and yellow hue boundaries are unchanged by the red-detection repair.

1. Separate connected regions **for each color**, so a small red fragment cannot
   merge into a larger dim green lens. Prefer saturated evidence (saturation at
   least 64), falling back to weak tints when necessary. Count pixels within 85%
   of that color region's peak brightness. A yellow core cannot raise the cutoff
   for a red rim because those color regions are measured separately.
2. Fill only holes enclosed by each region's outline. This associates a bright
   white or yellow-looking core with its surrounding red rim. The activation
   brightness is the greater of the reliable color peak and the enclosed area's
   90th-percentile brightness. A single white reflection cannot activate a large
   dark lens, while a small saturated colored fragment can still activate it.
3. Require activation brightness of at least **60**, replacing the former 90
   floor that excluded underexposed red lamps. Dim color evidence alone below
   that threshold does not count as an illuminated lamp. A dim rim can qualify
   when it encloses a sufficiently bright core.
4. Treat a differently colored core as part of a saturated enclosing rim when
   the core is substantially brighter or clipped. It must occupy at least 10%
   of the enclosing area and actually lie inside its outline. Mere bounding-box
   overlap cannot change a lamp's color. This handles red lamps whose cores look
   yellow even below the fixed clipping threshold after camera compression.
5. Compare lamp strengths using
   `activation_brightness * min(1, median_color_saturation / 64)`.
   Pale reflections receive less weight. Among regions within 85% of the strongest
   signal, select the one with the greatest color pixel count. Disconnected
   specks do not pool their votes. Tied counts between colors return `UNKNOWN`.

The selected region, including an enclosed core, supplies the inner box. Small
padding is clipped to the housing. There is one active-lamp box and one message
per housing. Neither vertical nor horizontal lamp order determines color.

Original crop pixels are never resized, blurred, or eroded before hue matching.
The default `min_pixels=1` preserves tiny visible fragments; it also accepts a
qualifying isolated noise pixel. Increase `min_pixels` for cameras with sensor
specks. If an entire lamp is clipped to neutral white, the image contains no hue
evidence to recover; reduce camera exposure and capture another frame.

For calibration, pass `ColorThresholds` to the detector constructor. `min_value`
sets the lamp activation floor; `min_support_value` sets the lower color-evidence
floor, capped at `min_value`. `peak_fraction` controls brightness comparisons
within a color region and between lamps. `min_pixels` sets minimum evidence size.
Set `contrast_fallback=False` in the detector constructor to use one Haar scan.
Choose calibration changes from captured camera examples. HSV and brightness
alone cannot always separate an illuminated source from a similar reflection;
neither camera accuracy nor edge-device FPS has been measured automatically.

## Manual synthetic checks

The checks cover all three active colors, underexposed red, dim rims around bright
cores, compressed yellow-looking red cores, warm amber, pale reflections, isolated
glare on dark lenses, and tiny colored fragments. They also check that disconnected
specks cannot pool their votes, bounding-box overlap cannot substitute for actual
enclosure, and the Haar retry gives equal intensity to all three colors. Existing
checks retain housing-only analysis, crop coordinates, duplicate suppression, and
two boxes with one message. These synthetic checks never open a camera:

```bash
./traffic_venv/Scripts/python.exe -m unittest -v test_traffic_light_detector
```

These checks are provided for manual execution; they were not run automatically.

## Importing into a future ROS 2 node

Camera access and GUI display run only under the `__main__` entry point. Create
one detector during node initialization, then call `process_frame` on each uint8
BGR image obtained from the node's image subscription (for example, a `bgr8` image
converted through `cv_bridge`):

```python
from traffic_light_detector import BGRImage, ColorThresholds, TrafficLightDetector

detector = TrafficLightDetector(
    cascade_path="/path/to/traffic_light_cascade.xml",
    thresholds=ColorThresholds(min_pixels=1),
)


def annotate_camera_frame(bgr_frame: BGRImage) -> BGRImage:
    return detector.process_frame(bgr_frame)
```

`process_frame` returns an annotated copy and leaves the input unchanged. For
machine-readable outputs, `detect(frame)` returns one `LightDetection` per housing:
`box` is the outer housing box, `lamp_box` is the active-lamp box or `None`, and
`state` is the color or `UNKNOWN`. Both boxes use full-frame coordinates.
`detect_lights(frame)` returns the housing boxes alone.

`find_active_light(crop, thresholds=detector.thresholds)` returns a state and a
lamp box relative to that crop. `classify_by_color(crop, thresholds=detector.thresholds)`
returns only the state. No ROS dependencies are needed for this module.
