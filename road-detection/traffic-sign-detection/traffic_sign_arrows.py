"""
traffic_sign_arrows.py
======================
Recognise circular direction signs by finding the WHITE ARROW inside the sign and
analysing its shape and position.

Pipeline
  1. find circular signs            (traffic_sign_circles.py, must be in the same folder)
  2. inside each disc, isolate the white symbol with an adaptive (Otsu) threshold
  3. count white parts and measure their position relative to the sign centre
       - centred ROW of 3-6 white letters on a red disc -> STOP (circular, like your sign)
       - ring of >=2 curved arrows around the centre   -> ROUNDABOUT
       - 2 separate arrows                              -> classify each one
       - 1 wide bar on a red sign                       -> NO ENTRY
       - 1 arrow                                        -> shape matching against arrow templates
  4. shape matching = silhouette IoU against synthetic templates (tolerates +-10 deg tilt)
  5. STOP detection itself (circular + octagon) lives in traffic_sign_circles.py

Install:   pip install opencv-python numpy
Examples:
  python traffic_sign_arrows.py --source signs.png --out result.png
  python traffic_sign_arrows.py --source 0 --show
"""
import argparse

import cv2
import numpy as np

from traffic_sign_circles import detect_circular_signs, load_bgr, run

# --------------------------------------------------------------------------
# Arrow templates (drawn on a 100x100 canvas, then normalised to 64x64)
# --------------------------------------------------------------------------
N = 64
MEANING = {
    "UP": "GO STRAIGHT",
    "LEFT": "TURN LEFT",
    "RIGHT": "TURN RIGHT",
    "DOWN": "ARROW DOWN",
    "UP_LEFT": "ARROW UP-LEFT",
    "UP_RIGHT": "ARROW UP-RIGHT",
    "DOWN_LEFT": "KEEP LEFT",
    "DOWN_RIGHT": "KEEP RIGHT",
    "TURN_LEFT": "LEFT TURN AHEAD",
    "TURN_RIGHT": "RIGHT TURN AHEAD",
    "FORK_LEFT": "STRAIGHT OR LEFT",
    "FORK_RIGHT": "STRAIGHT OR RIGHT",
}


def normalise(mask):
    """Crop to bounding box and resize to NxN binary (anisotropic: shape only, not size)."""
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    crop = mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1].astype(np.uint8) * 255
    crop = cv2.resize(crop, (N, N), interpolation=cv2.INTER_AREA)
    return crop > 127


def _rotate(mask, angle_cw):
    """Rotate a boolean mask clockwise about its centre on a padded canvas."""
    h, w = mask.shape
    pad = int(np.hypot(h, w))
    canvas = np.zeros((pad, pad), np.uint8)
    y0, x0 = (pad - h) // 2, (pad - w) // 2
    canvas[y0:y0 + h, x0:x0 + w] = mask.astype(np.uint8) * 255
    m = cv2.getRotationMatrix2D((pad / 2, pad / 2), -angle_cw, 1.0)   # + angle = CCW in OpenCV
    return cv2.warpAffine(canvas, m, (pad, pad), flags=cv2.INTER_LINEAR) > 127


def _arrow_up():
    c = np.zeros((100, 100), np.uint8)
    cv2.fillPoly(c, [np.array([(50, 2), (16, 40), (84, 40)], np.int32)], 255)
    cv2.rectangle(c, (40, 40), (60, 98), 255, -1)
    return c > 0


def _turn_left():
    c = np.zeros((100, 100), np.uint8)
    cv2.fillPoly(c, [np.array([(0, 29), (34, 0), (34, 59)], np.int32)], 255)
    cv2.rectangle(c, (34, 20), (100, 39), 255, -1)
    cv2.rectangle(c, (71, 20), (100, 100), 255, -1)
    return c > 0


def _fork_left():
    c = np.zeros((100, 100), np.uint8)
    cv2.fillPoly(c, [np.array([(73, 0), (46, 23), (100, 23)], np.int32)], 255)       # head up
    cv2.rectangle(c, (60, 23), (84, 100), 255, -1)                                    # stem
    cv2.fillPoly(c, [np.array([(0, 51), (29, 30), (29, 71)], np.int32)], 255)        # head left
    cv2.rectangle(c, (29, 44), (60, 58), 255, -1)                                     # branch
    return c > 0


def build_templates():
    t = {}
    up = _arrow_up()
    names = ["UP", "UP_RIGHT", "RIGHT", "DOWN_RIGHT", "DOWN", "DOWN_LEFT", "LEFT", "UP_LEFT"]
    for k, name in enumerate(names):
        t[name] = normalise(_rotate(up, 45 * k))
    tl, fl = _turn_left(), _fork_left()
    t["TURN_LEFT"], t["TURN_RIGHT"] = normalise(tl), normalise(tl[:, ::-1])
    t["FORK_LEFT"], t["FORK_RIGHT"] = normalise(fl), normalise(fl[:, ::-1])
    return t


TEMPLATES = build_templates()


def match_shape(mask, min_iou=0.5):
    """Best template for a boolean mask -> (name, iou). Tries small tilts."""
    best, best_iou = None, 0.0
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None, 0.0
    crop = mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    for tilt in (0, -6, 6, -12, 12):
        probe = normalise(_rotate(crop, tilt) if tilt else crop)
        if probe is None:
            continue
        for name, tpl in TEMPLATES.items():
            inter = np.logical_and(probe, tpl).sum()
            union = np.logical_or(probe, tpl).sum()
            iou = inter / float(union) if union else 0.0
            if iou > best_iou:
                best, best_iou = name, iou
    return (best if best_iou >= min_iou else None), best_iou


# --------------------------------------------------------------------------
# White-symbol extraction and classification
# --------------------------------------------------------------------------
def extract_white(gray, cx, cy, r):
    """Boolean mask of the bright symbol inside the sign + the disc mask."""
    h, w = gray.shape
    disc = np.zeros((h, w), np.uint8)
    cv2.circle(disc, (int(cx), int(cy)), int(0.88 * r), 255, -1)    # skip the sign's border
    vals = gray[disc > 0].reshape(-1, 1)
    thr, _ = cv2.threshold(vals, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bright, dim = vals[vals > thr], vals[vals <= thr]
    if bright.size == 0 or dim.size == 0 or float(bright.mean()) - float(dim.mean()) < 60:
        return np.zeros(gray.shape, np.uint8)            # no real white symbol (plain coloured disc)
    white = ((gray > thr) & (disc > 0)).astype(np.uint8)
    white = cv2.morphologyEx(white, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return white


def split_components(white, r):
    n, lab, st, cen = cv2.connectedComponentsWithStats(white)
    min_area = 0.01 * np.pi * r * r
    comps = []
    for k in range(1, n):
        if st[k, cv2.CC_STAT_AREA] >= min_area:
            comps.append(dict(mask=lab == k, bbox=tuple(int(v) for v in st[k, :4]),
                              centroid=tuple(cen[k]), area=int(st[k, cv2.CC_STAT_AREA])))
    return comps


def is_ring(comps, cx, cy, r):
    """Roundabout: pieces spread on a ring around the centre, nothing in the middle."""
    if len(comps) < 2:
        return False
    pts = np.vstack([np.column_stack(np.nonzero(c["mask"])[::-1]) for c in comps]).astype(np.float32)
    d = np.hypot(pts[:, 0] - cx, pts[:, 1] - cy) / r
    centre_off = np.hypot(pts[:, 0].mean() - cx, pts[:, 1].mean() - cy) / r
    return 0.35 < d.mean() < 0.8 and d.std() < 0.15 and centre_off < 0.15


def classify_sign(img, det):
    cx, cy = det["center"]
    r = det["radius"]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    white = extract_white(gray, cx, cy, r)
    comps = split_components(white, r)
    info = dict(label="UNKNOWN", meaning="unknown symbol", parts=len(comps), offset=None, iou=0.0)
    if not comps:
        return info

    # position of the white symbol relative to the sign centre (-1..1, y down)
    ys, xs = np.nonzero(white)
    info["offset"] = (round(float((xs.mean() - cx) / r), 2), round(float((ys.mean() - cy) / r), 2))

    if is_ring(comps, cx, cy, r):
        info.update(label="ROUNDABOUT", meaning="ROUNDABOUT")
        return info

    if len(comps) == 1:
        c = comps[0]
        bw, bh = c["bbox"][2], c["bbox"][3]
        if det["color"] == "red" and bw > 2.2 * bh:
            info.update(label="NO_ENTRY", meaning="NO ENTRY")
            return info
        name, iou = match_shape(c["mask"])
        if name:
            info.update(label=name, meaning=MEANING[name], iou=round(float(iou), 2))
        return info

    # several separate arrows: classify each, then name the combination
    names = []
    for c in comps:
        name, _ = match_shape(c["mask"])
        names.append(name or "?")
    combo = "+".join(sorted(names))
    if sorted(names) == ["DOWN_LEFT", "DOWN_RIGHT"]:
        info.update(label=combo, meaning="KEEP LEFT OR RIGHT (pass either side)")
    else:
        info.update(label=combo, meaning="MULTIPLE ARROWS: " + combo)
    return info


def classify_all(img, method="both"):
    """Circular signs (+ STOP, circular or octagon) -> list of dicts with label/meaning."""
    dets = detect_circular_signs(img, method)          # STOP signs are flagged there (kind="stop")
    for d in dets:
        if d.get("kind") == "stop":
            d.update(label="STOP", meaning="STOP" + (" (octagon)" if d.get("shape") == "octagon" else ""),
                     parts=0, offset=(0.0, 0.0), iou=0.0)
        else:
            d.update(classify_sign(img, d))
    return dets


# --------------------------------------------------------------------------
# Drawing + main
# --------------------------------------------------------------------------
def draw_results(img, dets):
    out = img.copy()
    th = max(2, int(max(img.shape[:2]) / 500))
    for d in dets:
        cx, cy, r = int(d["center"][0]), int(d["center"][1]), int(d["radius"])
        ok = d["label"] != "UNKNOWN"
        if d.get("polygon") is not None:
            cv2.polylines(out, [d["polygon"].astype(np.int32)], True, (0, 200, 0), th + 1)
        else:
            cv2.circle(out, (cx, cy), r, (0, 200, 0) if ok else (0, 0, 255), th + 1)
        text = d["label"]
        scale = max(0.5, r / 200.0)
        (tw, tht), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, th)
        org = (int(cx - tw / 2), int(cy + r + tht + 10))
        cv2.putText(out, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), th + 3, cv2.LINE_AA)
        cv2.putText(out, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 200, 0) if ok else (0, 0, 255),
                    th, cv2.LINE_AA)
    return out


def main():
    ap = argparse.ArgumentParser(description="Direction-sign recognition from white arrows")
    ap.add_argument("--source", required=True, help="image, video or camera index")
    ap.add_argument("--method", default="both", choices=["color", "hough", "both"])
    ap.add_argument("--out", default=None)
    ap.add_argument("--show", action="store_true")
    a = ap.parse_args()

    def process(frame):
        dets = classify_all(frame, a.method)
        lines = [f"{len(dets)} sign(s)"]
        for i, d in enumerate(dets, 1):
            lines.append(f"  #{i}: {d['label']:<12} {d['meaning']:<38} parts={d['parts']} "
                         f"arrow_offset={d['offset']} iou={d['iou']}")
        return draw_results(frame, dets), "\n".join(lines)

    run(a.source, process, a.out, a.show)


if __name__ == "__main__":
    main()
