# Traffic light classification by spatial position

This webcam demo uses a Haar cascade to locate traffic lights, then classifies
each crop by the vertical position of its brightest region. The top third maps
to **RED**, the middle to **YELLOW**, and the bottom to **GREEN**. Crops without a
sufficiently bright, distinct region return **UNKNOWN**.

Use Python **3.10 through 3.13**, a connected webcam, and a desktop session that
can display OpenCV windows. The requirements pin **opencv-python 4.12.0.88** and
**NumPy 2.2.6**. These versions support your Python 3.11 installation and retain
the Haar cascade API used by this script. Run the commands yourself; setup and
execution are not performed automatically.

## Repair your existing Git Bash environment

Your `MINGW64` prompt is **Git Bash on Windows**. `Remove-Item` is a PowerShell
command and is not needed in Git Bash. The installed OpenCV 5.0.0.93 package lacks
`cv2.CascadeClassifier`, and the previous download URL saved `404: Not Found`
instead of a cascade. Both issues must be fixed.

Run these commands in Git Bash. Keep your existing `position_venv` environment;
there is no need to recreate it. The install command replaces the incompatible
versions using the updated requirements. The download replaces the invalid XML.

```bash
cd ~/landmark-project/road-detection/traffic-light-detection/by-position
source position_venv/Scripts/activate
python -m pip install --upgrade --force-reinstall --only-binary=:all: -r requirements.txt
curl -fL -o traffic_light_cascade.xml https://raw.githubusercontent.com/eyeamskd/traffic-light-detection/master/cascade7.xml
```

Check the installation and cascade before opening the webcam:

```bash
python -c "import cv2; from traffic_light_position import TrafficLightPositionDetector; TrafficLightPositionDetector(); print('OpenCV:', cv2.__version__); print('Cascade loaded successfully')"
python traffic_light_position.py
```

The check should print `OpenCV: 4.12.0` and `Cascade loaded successfully`. Only run
the webcam command after the check succeeds. Focus its window and press **q** to
quit. If a command fails, resolve that error before continuing.

## New setup: Git Bash on Windows

Use this section only if you have not created an environment yet:

```bash
cd ~/landmark-project/road-detection/traffic-light-detection/by-position
python -m venv position_venv
source position_venv/Scripts/activate
python -m pip install --only-binary=:all: -r requirements.txt
curl -fL -o traffic_light_cascade.xml https://raw.githubusercontent.com/eyeamskd/traffic-light-detection/master/cascade7.xml
python traffic_light_position.py
```

## New setup: Windows PowerShell

Open PowerShell in this project directory:

```powershell
cd C:\Users\Lenovo\landmark-project\road-detection\traffic-light-detection\by-position
python -m venv position_venv
.\position_venv\Scripts\Activate.ps1
python -m pip install --only-binary=:all: -r requirements.txt
```

If activation is blocked by the script execution policy, allow scripts for the
current PowerShell session only, then activate again:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\position_venv\Scripts\Activate.ps1
```

Use `curl.exe` explicitly in PowerShell to avoid its older `curl` alias:

```powershell
curl.exe -fL -o traffic_light_cascade.xml https://raw.githubusercontent.com/eyeamskd/traffic-light-detection/master/cascade7.xml
python traffic_light_position.py
```

## New setup: macOS / Linux

Open a terminal in the directory containing these files, then run:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --only-binary=:all: -r requirements.txt
curl -fL -o traffic_light_cascade.xml https://raw.githubusercontent.com/eyeamskd/traffic-light-detection/master/cascade7.xml
python traffic_light_position.py
```

## Cascade source

The replacement is [`cascade7.xml` from eyeamskd/traffic-light-detection](https://github.com/eyeamskd/traffic-light-detection/blob/master/cascade7.xml).
Its download was checked: HTTP 200, 88,991 bytes, an `opencv_storage` XML root,
HAAR features, 20 stages, and a 17 x 32 detection window. These checks establish
the file's availability and structure, not accuracy on your camera. The original
`mvirgo/Traffic-Light-Classifier` URL produced the local 14-byte `404: Not Found`
file and should no longer be used. `curl -fL` reports HTTP errors and follows
redirects.

## Operation

The script loads `traffic_light_cascade.xml` from the script's directory and
opens the default webcam (`cv2.VideoCapture(0)`). Each detected light gets a
bounding box and a label such as `Pos: GREEN`; smoothed real-time FPS appears
at the top left. Focus the OpenCV window and press **q** to exit. **Ctrl+C** in
the terminal also exits. Camera and window resources are released on exit or
camera failure.

A missing cascade raises `FileNotFoundError` with the download command. Invalid
XML or a model OpenCV cannot load raises `ValueError` with download instructions.
An OpenCV installation without `CascadeClassifier` raises `RuntimeError` with
the imported module's version, location, and the dependency repair command. A
camera that cannot open or deliver frames raises a descriptive `RuntimeError`.

## How the classifier works

1. Convert the frame to grayscale and locate lights with `detectMultiScale`.
2. Extract HSV Value (brightness) from each BGR crop and apply a 5 x 5 Gaussian
   blur. Hue is not used.
3. Reject dark or low-contrast crops. Apply a binary threshold at the greater of
   180 and 85% of the blurred peak brightness.
4. Separate connected bright regions, discard regions smaller than three pixels
   or 0.5% of the crop, and select the region with the highest mean brightness.
   Larger area breaks ties.
5. Compute that region's `(x, y)` centroid with `cv2.moments`. Compare `y` with
   `height / 3` and `2 * height / 3` to choose the state.

The constructor exposes `brightness_threshold`, `relative_threshold`,
`minimum_contrast`, and `minimum_blob_fraction` for tuning. The default minimum
contrast requires the blurred peak to exceed the crop median by at least 20
brightness levels. `process_frame` returns an annotated copy of its input.

## Assumptions and limits

Each detection must enclose the **entire upright three-lamp housing**, with red
above yellow above green. Check the supplied cascade's boxes on your camera:
boxes around individual lamps or partial housings do not support this spatial
mapping. Horizontal lights, arrows, rotated housings, and multiple illuminated
lamps require additional logic.

Position can remain useful when a lamp's color washes out, provided its bright
region remains distinct. Strong glare or reflections can still select the wrong
region. Uniformly overexposed crops return `UNKNOWN`; dim or very small lamps may
also return `UNKNOWN`. Detection quality depends on the supplied cascade and
scene conditions. This demo does not establish autonomous-driving reliability.
