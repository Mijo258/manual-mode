"""
traffic_sign_circles.py
=======================
Detect CIRCULAR traffic signs (blue "mandatory" signs and red signs / red-ring
prohibition signs) in an image, a video file or a webcam stream.

STOP signs (circular "STOP" disc with white letters, and the classic octagon) are
detected here too: they come back with kind="stop".

Methods
-------
  color : HSV colour segmentation (blue / red) + contour circularity test
  hough : Hough circle transform, each circle validated by its inner colour
  both  : union of the two with duplicates removed   (default)

Install:   pip install opencv-python numpy
Examples:
  python traffic_sign_circles.py --source signs.png --out result.png
  python traffic_sign_circles.py --source 0 --show              # webcam
  python traffic_sign_circles.py --source drive.mp4 --show --method color
"""
import argparse
import os
import sys

import cv2
import numpy as np

# --------------------------------------------------------------------------
# HSV ranges (OpenCV: H 0-179, S 0-255, V 0-255)
# --------------------------------------------------------------------------
COLOR_RANGES = {
    "blue": [((100, 110, 50), (130, 255, 255))],
    "red": [((0, 100, 50), (10, 255, 255)), ((170, 100, 50), (179, 255, 255))],
}
IMAGE_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff")


# --------------------------------------------------------------------------
# I/O helpers
# --------------------------------------------------------------------------
def load_bgr(path):
    """Read an image; transparent pixels (PNG alpha) are composited on white."""
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"cannot read image: {path}")
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 4:
        alpha = img[:, :, 3:4].astype(np.float32) / 255.0
        rgb = img[:, :, :3].astype(np.float32)
        img = (rgb * alpha + 255.0 * (1.0 - alpha)).astype(np.uint8)
    return img


def color_mask(hsv, name):
    m = np.zeros(hsv.shape[:2], np.uint8)
    for lo, hi in COLOR_RANGES[name]:
        m |= cv2.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8))
    return m


# --------------------------------------------------------------------------
# Detection
# --------------------------------------------------------------------------
def detect_by_color(img, min_radius=12, circ_thr=0.78, fill_thr=0.80):
    """Blue / red blobs that look like discs (works for filled and ringed signs)."""
    hsv = cv2.cvtColor(cv2.GaussianBlur(img, (5, 5), 0), cv2.COLOR_BGR2HSV)
    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    dets = []
    for name in COLOR_RANGES:
        m = color_mask(hsv, name)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k_close)   # fill watermark gaps
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k_open)     # remove specks
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            area = cv2.contourArea(c)
            per = cv2.arcLength(c, True)
            if per == 0 or area < 1:
                continue
            (cx, cy), r = cv2.minEnclosingCircle(c)
            if r < min_radius:
                continue
            circularity = 4.0 * np.pi * area / (per * per)
            fill = area / (np.pi * r * r)
            if circularity < circ_thr or fill < fill_thr:
                continue
            dets.append(dict(center=(cx, cy), radius=r, color=name, method="color",
                             score=float(min(circularity, fill))))
    return dets


def detect_by_hough(img, min_radius=12, max_radius=0, min_color_frac=0.25):
    """Hough circles, kept only if they sit on a real colour boundary:
    just INSIDE the circle edge it is blue/red, just OUTSIDE it is not."""
    h, w = img.shape[:2]
    if max_radius <= 0:
        max_radius = min(h, w) // 2
    gray = cv2.medianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 5)
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1.2,
                               minDist=max(2 * min_radius, 20),
                               param1=120, param2=70,
                               minRadius=min_radius, maxRadius=max_radius)
    dets = []
    if circles is None:
        return dets
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    masks = {n: color_mask(hsv, n) for n in COLOR_RANGES}

    def frac(mask, cx, cy, rad, thick, filled=False):
        ring = np.zeros((h, w), np.uint8)
        cv2.circle(ring, (int(cx), int(cy)), int(max(rad, 1)), 255, -1 if filled else int(max(thick, 2)))
        tot = cv2.countNonZero(ring)
        return cv2.countNonZero(cv2.bitwise_and(mask, ring)) / tot if tot else 0.0

    for cx, cy, r in circles[0]:
        thick = 0.06 * r
        best, best_score = None, 0.0
        for name, m in masks.items():
            inner = frac(m, cx, cy, 0.88 * r, thick)
            outer = frac(m, cx, cy, 1.12 * r, thick)
            disc = frac(m, cx, cy, r, 0, filled=True)
            if inner >= 0.6 and outer <= 0.3 and disc >= min_color_frac and inner - outer > best_score:
                best, best_score = name, inner - outer
        if best:
            dets.append(dict(center=(float(cx), float(cy)), radius=float(r),
                             color=best, method="hough", score=float(best_score)))
    return dets


def _merge(primary, extra):
    out = list(primary)
    for b in extra:
        dup = False
        for a in out:
            d = np.hypot(a["center"][0] - b["center"][0], a["center"][1] - b["center"][1])
            if d < 0.6 * min(a["radius"], b["radius"]):
                dup = True
                break
        if not dup:
            out.append(b)
    return out


def reading_order(dets):
    """Sort detections row by row (top->bottom), left->right inside a row."""
    if not dets:
        return dets
    row_h = 1.6 * float(np.median([d["radius"] for d in dets]))
    return sorted(dets, key=lambda d: (round(d["center"][1] / row_h), d["center"][0]))


def detect_circular_signs(img, method="both", min_radius=12, max_side=1000, stop=True):
    """Return a list of dicts: center, radius, color, method, bbox, kind  (original pixel units).
    kind is "stop" for STOP signs (circular or octagonal), otherwise "circle"."""
    h, w = img.shape[:2]
    scale = min(1.0, max_side / float(max(h, w)))
    small = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else img

    dets = []
    if method in ("color", "both"):
        dets = detect_by_color(small, min_radius)
    if method in ("hough", "both"):
        dets = _merge(dets, detect_by_hough(small, min_radius))

    for d in dets:                                   # back to original resolution
        d["center"] = (d["center"][0] / scale, d["center"][1] / scale)
        d["radius"] = d["radius"] / scale
        cx, cy = d["center"]
        r = d["radius"]
        d["bbox"] = (int(cx - r), int(cy - r), int(2 * r), int(2 * r))
        d["kind"], d["shape"], d["polygon"] = "circle", "circle", None

    if stop:
        for d in dets:                                   # red disc + row of white letters = STOP
            if d["color"] == "red" and stop_text_score(img, d["center"][0], d["center"][1], d["radius"])[0]:
                d["kind"] = "stop"
        for o in detect_octagon_stop(img):               # classic 8-sided STOP sign
            if not any(np.hypot(o["center"][0] - d["center"][0], o["center"][1] - d["center"][1]) < 0.5 * d["radius"]
                       for d in dets):
                dets.append(dict(center=o["center"], radius=o["radius"], color="red", method="octagon",
                                 score=1.0, bbox=o["bbox"], kind="stop", shape="octagon", polygon=o["polygon"]))
            else:                                        # an octagon also seen as a disc: keep disc, mark shape
                for d in dets:
                    if np.hypot(o["center"][0] - d["center"][0], o["center"][1] - d["center"][1]) < 0.5 * d["radius"]:
                        d.update(kind="stop", shape="octagon", polygon=o["polygon"])
    return reading_order(dets)


# --------------------------------------------------------------------------
# STOP sign detection (word-STOP disc + classic octagon)
# --------------------------------------------------------------------------
def _octagon_vertices(hull, per):
    """Try several tolerances; return the polygon if one of them gives 8 corners."""
    scale = np.sqrt(cv2.contourArea(hull))
    for eps in (0.01, 0.015, 0.02, 0.025, 0.03):
        poly = cv2.approxPolyDP(hull, eps * per, True)
        if len(poly) != 8:
            continue
        # a real octagon has straight sides: every hull point must lie ON the polygon.
        # (a circle simplified to 8 corners bulges ~7% of its radius away from it)
        dev = max(abs(cv2.pointPolygonTest(poly, (float(p[0][0]), float(p[0][1])), True))
                  for p in hull)
        if dev / scale < 0.02:
            return poly
    return None



def stop_text_score(img, cx, cy, r):
    """Does the inside of this red disc contain the word STOP (a row of letters)?
    Returns (is_stop, number_of_letter_blobs)."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    disc = np.zeros(gray.shape, np.uint8)
    cv2.circle(disc, (int(cx), int(cy)), int(0.80 * r), 255, -1)       # excludes the white ring
    vals = gray[disc > 0].reshape(-1, 1)
    if vals.size < 100:
        return False, 0
    thr, _ = cv2.threshold(vals, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    white = ((gray > thr) & (disc > 0)).astype(np.uint8)
    white = cv2.morphologyEx(white, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, st, cen = cv2.connectedComponentsWithStats(white)
    letters = [st[k] for k in range(1, n)
               if st[k, cv2.CC_STAT_AREA] >= 0.004 * np.pi * r * r
               and 0.10 * 2 * r <= st[k, cv2.CC_STAT_HEIGHT] <= 0.60 * 2 * r]
    if not 3 <= len(letters) <= 6:
        return False, len(letters)
    x0 = min(l[cv2.CC_STAT_LEFT] for l in letters)
    x1 = max(l[cv2.CC_STAT_LEFT] + l[cv2.CC_STAT_WIDTH] for l in letters)
    y0 = min(l[cv2.CC_STAT_TOP] for l in letters)
    y1 = max(l[cv2.CC_STAT_TOP] + l[cv2.CC_STAT_HEIGHT] for l in letters)
    w, h = x1 - x0, y1 - y0
    heights = [l[cv2.CC_STAT_HEIGHT] for l in letters]
    centred = np.hypot((x0 + x1) / 2.0 - cx, (y0 + y1) / 2.0 - cy) < 0.25 * r
    row_like = 1.8 <= w / float(h) <= 6.0 and w >= 0.9 * r
    similar = max(heights) / float(min(heights)) < 1.6
    return bool(centred and row_like and similar), len(letters)



def detect_octagon_stop(img, min_area=400, min_white=0.04):
    """Classic 8-sided STOP sign."""
    hsv = cv2.cvtColor(cv2.GaussianBlur(img, (5, 5), 0), cv2.COLOR_BGR2HSV)
    red = color_mask(hsv, "red")
    red = cv2.morphologyEx(red, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    red = cv2.morphologyEx(red, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    cnts, _ = cv2.findContours(red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    white = cv2.inRange(hsv, (0, 0, 150), (179, 80, 255))   # low saturation, bright
    dets = []
    for c in cnts:
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        x, y, w, h = cv2.boundingRect(c)
        if not 0.75 <= w / float(h) <= 1.33:
            continue
        hull = cv2.convexHull(c)
        per = cv2.arcLength(hull, True)
        poly = _octagon_vertices(hull, per)
        if poly is None:
            continue
        fill = cv2.contourArea(hull) / float(w * h)
        if not 0.70 <= fill <= 0.92:                      # circle = 0.785, square = 1.0
            continue
        # white content (letters / border) inside the shape
        inner = np.zeros(red.shape, np.uint8)
        cv2.drawContours(inner, [hull], -1, 255, -1)
        inner = cv2.erode(inner, np.ones((3, 3), np.uint8), iterations=max(1, int(min(w, h) * 0.03)))
        total = cv2.countNonZero(inner)
        white_frac = cv2.countNonZero(cv2.bitwise_and(white, inner)) / float(max(total, 1))
        if white_frac < min_white:
            continue
        dets.append(dict(label="STOP", shape="octagon", bbox=(x, y, w, h), polygon=poly.reshape(-1, 2),
                         center=(x + w / 2.0, y + h / 2.0), radius=max(w, h) / 2.0,
                         white_frac=float(white_frac)))
    return dets



# --------------------------------------------------------------------------
# Drawing + main
# --------------------------------------------------------------------------
def draw_detections(img, dets):
    out = img.copy()
    th = max(2, int(max(img.shape[:2]) / 500))
    for i, d in enumerate(dets, 1):
        cx, cy = int(d["center"][0]), int(d["center"][1])
        r = int(d["radius"])
        if d.get("polygon") is not None:
            cv2.polylines(out, [d["polygon"].astype(np.int32)], True, (0, 200, 0), th)
        else:
            cv2.circle(out, (cx, cy), r, (0, 200, 0), th)
        cv2.circle(out, (cx, cy), max(2, th), (0, 0, 255), -1)
        txt = f"#{i} STOP" if d.get("kind") == "stop" else f"#{i} {d['color']}"
        cv2.putText(out, txt, (max(cx - r, 0), max(cy - r - 8, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, max(0.6, r / 90.0), (0, 0, 0), th + 2, cv2.LINE_AA)
        cv2.putText(out, txt, (max(cx - r, 0), max(cy - r - 8, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, max(0.6, r / 90.0), (0, 220, 0), th, cv2.LINE_AA)
    return out


def run(source, process, out=None, show=False):
    """process(frame) -> (annotated_frame, info_string). Works for image / video / webcam."""
    src = str(source)
    if src.isdigit() or not src.lower().endswith(IMAGE_EXT):
        cap = cv2.VideoCapture(int(src) if src.isdigit() else src)
        if not cap.isOpened():
            sys.exit(f"cannot open video source: {source}")
        writer = None
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            res, info = process(frame)
            if out and writer is None:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(out, fourcc, cap.get(cv2.CAP_PROP_FPS) or 25,
                                         (res.shape[1], res.shape[0]))
            if writer:
                writer.write(res)
            if show:
                cv2.imshow("result  (q = quit)", res)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
    else:
        img = load_bgr(src)
        res, info = process(img)
        print(info)
        out = out or os.path.splitext(src)[0] + "_result.png"
        cv2.imwrite(out, res)
        print(f"saved -> {out}")
        if show:
            cv2.imshow("result", res)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


def main():
    ap = argparse.ArgumentParser(description="Circular traffic-sign detection")
    ap.add_argument("--source", required=True, help="image path, video path or camera index (0)")
    ap.add_argument("--method", default="both", choices=["color", "hough", "both"])
    ap.add_argument("--min-radius", type=int, default=12)
    ap.add_argument("--out", default=None)
    ap.add_argument("--show", action="store_true")
    a = ap.parse_args()

    def process(frame):
        dets = detect_circular_signs(frame, a.method, a.min_radius)
        lines = [f"{len(dets)} sign(s) found ({sum(d.get('kind') == 'stop' for d in dets)} STOP)"]
        for i, d in enumerate(dets, 1):
            lines.append(f"  #{i}: {('STOP' if d.get('kind') == 'stop' else d['color']):<4} center=({d['center'][0]:.0f},{d['center'][1]:.0f}) "
                         f"radius={d['radius']:.0f} via {d['method']}")
        return draw_detections(frame, dets), "\n".join(lines)

    run(a.source, process, a.out, a.show)


if __name__ == "__main__":
    main()
