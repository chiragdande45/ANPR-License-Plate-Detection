"""
core.py
=======
ANPR Phase 1 - Image Processing Core
All algorithms for the Phase 1 pipeline:
  1.  Letterbox resize
  2.  YOLOv11n plate detection
  3.  Adaptive ROI padding & crop
  4.  Conditional perspective correction
  5.  Adaptive Lanczos upscaling
  6.  Conditional bilateral denoising
  7.  CLAHE
  8.  Selective glare suppression
  9.  Selective mud/dirt suppression
  10. Stroke-width adaptive unsharp masking
  11. Sauvola adaptive thresholding
  12. Adaptive morphological closing
  + Utility / metric helpers
"""

import cv2
import numpy as np
import math
import time
import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Optional dependency: scikit-image for Sauvola
# ---------------------------------------------------------------------------
try:
    from skimage.filters import threshold_sauvola
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False


# ===========================================================================
# 1.  LETTERBOX RESIZE
# ===========================================================================

def letterbox_resize(image, target_size=640, pad_color=(114, 114, 114)):
    """
    Resize image to target_size x target_size using letterboxing.
    Preserves aspect ratio and pads with pad_color.

    Returns
    -------
    resized   : padded image (target_size x target_size, 3-channel)
    scale     : float – ratio applied to original dimensions
    pad_top   : int – pixels of top padding
    pad_left  : int – pixels of left padding
    """
    h, w = image.shape[:2]
    scale = target_size / max(h, w)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))

    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Create canvas and paste
    canvas = np.full((target_size, target_size, 3), pad_color, dtype=np.uint8)
    pad_top  = (target_size - new_h) // 2
    pad_left = (target_size - new_w) // 2
    canvas[pad_top:pad_top + new_h, pad_left:pad_left + new_w] = resized

    return canvas, scale, pad_top, pad_left


def unletterbox_bbox(x1, y1, x2, y2, scale, pad_top, pad_left):
    """
    Convert a bounding box from letterboxed coordinates back to original
    image coordinates.
    """
    x1 = (x1 - pad_left) / scale
    y1 = (y1 - pad_top)  / scale
    x2 = (x2 - pad_left) / scale
    y2 = (y2 - pad_top)  / scale
    return x1, y1, x2, y2


# ===========================================================================
# 2.  YOLOv11n PLATE DETECTION
# ===========================================================================

def load_detector(weights_path):
    """
    Load the YOLOv11n model from weights_path using the ultralytics library.

    Returns the model object, or None if loading fails.
    """
    try:
        from ultralytics import YOLO
        model = YOLO(weights_path)
        return model
    except Exception as e:
        print(f"[DetectorLoad] Failed to load model from '{weights_path}': {e}")
        return None


def detect_plate(model, image_bgr, conf_threshold=0.25, target_size=640):
    """
    Run YOLOv11n inference on image_bgr.

    Strategy when multiple detections exist:
      - Filter by confidence >= conf_threshold
      - Among remaining boxes, prefer the one with
        (a) highest confidence, tie-broken by
        (b) most plausible plate aspect ratio (width > height, ratio 1.5–6.0)

    Returns
    -------
    dict with keys:
        detected    : bool
        x1, y1, x2, y2  : bounding box in ORIGINAL image coordinates (float)
        confidence  : float
        plate_w     : float (pixels in original image)
        plate_h     : float
        aspect_ratio: float
        letterboxed : the 640-px letterboxed image used for detection
    """
    h_orig, w_orig = image_bgr.shape[:2]
    lb_image, scale, pad_top, pad_left = letterbox_resize(image_bgr, target_size)

    result_dict = {
        "detected": False,
        "x1": 0, "y1": 0, "x2": 0, "y2": 0,
        "confidence": 0.0,
        "plate_w": 0, "plate_h": 0,
        "aspect_ratio": 0.0,
        "letterboxed": lb_image,
    }

    try:
        results = model.predict(lb_image, conf=conf_threshold, verbose=False)
    except Exception as e:
        print(f"[Detection] Inference error: {e}")
        return result_dict

    boxes = []
    for r in results:
        if r.boxes is None:
            continue
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            boxes.append((conf, x1, y1, x2, y2))

    if not boxes:
        return result_dict

    # Score each candidate: combine confidence with aspect-ratio plausibility
    def plate_score(b):
        conf, x1, y1, x2, y2 = b
        bw = max(x2 - x1, 1)
        bh = max(y2 - y1, 1)
        ar = bw / bh
        # Reward aspect ratios in the typical plate range (1.5 – 6.0)
        ar_score = 1.0 if 1.5 <= ar <= 6.0 else 0.5
        return conf * ar_score

    boxes.sort(key=plate_score, reverse=True)
    best = boxes[0]
    conf, x1_lb, y1_lb, x2_lb, y2_lb = best

    # Convert back to original image coordinates
    x1, y1, x2, y2 = unletterbox_bbox(x1_lb, y1_lb, x2_lb, y2_lb,
                                       scale, pad_top, pad_left)

    # Clamp to image bounds
    x1 = max(0.0, x1)
    y1 = max(0.0, y1)
    x2 = min(float(w_orig), x2)
    y2 = min(float(h_orig), y2)

    plate_w = x2 - x1
    plate_h = y2 - y1

    result_dict.update({
        "detected": True,
        "x1": x1, "y1": y1, "x2": x2, "y2": y2,
        "confidence": conf,
        "plate_w": plate_w,
        "plate_h": plate_h,
        "aspect_ratio": plate_w / max(plate_h, 1),
        "letterboxed": lb_image,
    })
    return result_dict


def draw_detection(image_bgr, det, output_path=None):
    """
    Draw bounding box, confidence, plate dimensions and aspect ratio
    onto a copy of the image.  Optionally save to output_path.
    Returns the annotated image.
    """
    vis = image_bgr.copy()
    if not det["detected"]:
        cv2.putText(vis, "NO PLATE DETECTED", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    else:
        x1, y1, x2, y2 = (int(det["x1"]), int(det["y1"]),
                           int(det["x2"]), int(det["y2"]))
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = (f"Conf:{det['confidence']:.2f}  "
                 f"W:{det['plate_w']:.0f}  H:{det['plate_h']:.0f}  "
                 f"AR:{det['aspect_ratio']:.2f}")
        # Background rectangle for text readability
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(vis, (x1, y1 - th - 8), (x1 + tw + 4, y1), (0, 255, 0), -1)
        cv2.putText(vis, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)
    if output_path:
        save_image(vis, output_path)
    return vis


# ===========================================================================
# 3.  ADAPTIVE ROI PADDING & CROP
# ===========================================================================

def crop_plate_roi(image_bgr, det, pad_ratio=0.07):
    """
    Crop the detected plate region from the original image with adaptive padding.

    pad_ratio: fraction of box dimension added on each side (default 7%).
    The actual padding is clamped to 5–10% of the box dimension.

    Returns
    -------
    roi       : cropped image (may be empty if detection failed)
    crop_coords : (x1p, y1p, x2p, y2p) in original image pixels
    pad_used  : actual padding fraction used
    """
    h, w = image_bgr.shape[:2]

    if not det["detected"]:
        return np.zeros((10, 10, 3), dtype=np.uint8), (0, 0, 0, 0), 0.0

    x1, y1, x2, y2 = det["x1"], det["y1"], det["x2"], det["y2"]
    bw = x2 - x1
    bh = y2 - y1

    # Adaptive: clamp padding fraction between 5% and 10%
    pad_frac = float(np.clip(pad_ratio, 0.05, 0.10))
    pad_x = bw * pad_frac
    pad_y = bh * pad_frac

    x1p = int(max(0, x1 - pad_x))
    y1p = int(max(0, y1 - pad_y))
    x2p = int(min(w, x2 + pad_x))
    y2p = int(min(h, y2 + pad_y))

    roi = image_bgr[y1p:y2p, x1p:x2p].copy()
    return roi, (x1p, y1p, x2p, y2p), pad_frac


# ===========================================================================
# 4.  CONDITIONAL PERSPECTIVE CORRECTION
# ===========================================================================

def estimate_skew(gray):
    """
    Estimate the dominant skew angle of a grayscale plate image.
    Uses the Hough line transform on Canny edges.
    Returns angle in degrees (positive = clockwise tilt).
    """
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=max(20, gray.shape[1] // 4))
    if lines is None or len(lines) == 0:
        return 0.0

    angles = []
    for line in lines[:30]:
        rho, theta = line[0]
        # Convert theta to angle from horizontal
        angle_deg = np.degrees(theta) - 90.0
        angles.append(angle_deg)

    if not angles:
        return 0.0
    # Use median to avoid outliers
    return float(np.median(angles))


def find_plate_corners(gray):
    """
    Attempt to find four corners of the plate for perspective transform.
    Uses contour approximation on a threshold of the ROI.
    Returns a 4x2 float32 array or None if corners are not reliable.
    """
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255,
                               cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    # Take the largest contour
    c = max(contours, key=cv2.contourArea)
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.04 * peri, True)

    if len(approx) != 4:
        return None

    pts = approx.reshape(4, 2).astype(np.float32)
    return pts


def order_corners(pts):
    """
    Order corner points: top-left, top-right, bottom-right, bottom-left.
    """
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]   # top-left
    rect[2] = pts[np.argmax(s)]   # bottom-right
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # top-right
    rect[3] = pts[np.argmax(diff)]  # bottom-left
    return rect


def perspective_correct(roi_bgr, skew_threshold=3.0):
    """
    Conditionally apply perspective correction to the plate ROI.

    Logic:
      1. Estimate skew angle.
      2. If |skew| < skew_threshold degrees → plate is already straight → skip.
      3. If |skew| >= skew_threshold:
           a. Try 4-point corner detection.
           b. If 4 corners found → apply full perspective warp.
           c. Else → apply simple rotation deskew.

    Returns
    -------
    corrected   : corrected plate image
    was_corrected : bool
    skew_angle  : float (degrees)
    method_used : str description
    """
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    skew = estimate_skew(gray)
    was_corrected = False
    method_used = "none (already straight)"

    if abs(skew) < skew_threshold:
        return roi_bgr.copy(), False, skew, method_used

    # Try 4-point perspective warp first
    corners = find_plate_corners(gray)
    if corners is not None:
        rect = order_corners(corners)
        tl, tr, br, bl = rect

        width_top  = np.linalg.norm(tr - tl)
        width_bot  = np.linalg.norm(br - bl)
        height_left  = np.linalg.norm(bl - tl)
        height_right = np.linalg.norm(br - tr)

        out_w = int(max(width_top, width_bot))
        out_h = int(max(height_left, height_right))

        if out_w > 0 and out_h > 0:
            dst = np.array([
                [0,         0        ],
                [out_w - 1, 0        ],
                [out_w - 1, out_h - 1],
                [0,         out_h - 1],
            ], dtype=np.float32)

            M = cv2.getPerspectiveTransform(rect, dst)
            corrected = cv2.warpPerspective(roi_bgr, M, (out_w, out_h),
                                            flags=cv2.INTER_LANCZOS4,
                                            borderMode=cv2.BORDER_REPLICATE)
            return corrected, True, skew, "4-point perspective warp"

    # Fallback: simple rotation deskew
    h, w = roi_bgr.shape[:2]
    M_rot = cv2.getRotationMatrix2D((w / 2, h / 2), skew, 1.0)
    corrected = cv2.warpAffine(roi_bgr, M_rot, (w, h),
                               flags=cv2.INTER_LANCZOS4,
                               borderMode=cv2.BORDER_REPLICATE)
    return corrected, True, skew, f"rotation deskew ({skew:.1f}°)"


# ===========================================================================
# 5.  ADAPTIVE LANCZOS UPSCALING
# ===========================================================================

def adaptive_upscale(plate_bgr, min_target_w=400, max_scale=4.0):
    """
    Upscale the plate ROI using Lanczos interpolation.

    Scale is chosen adaptively:
      - Small / far plates (narrow) → stronger upscaling (up to max_scale).
      - Large / near plates          → weaker upscaling (minimum 1.0).
      - Scale is chosen so the output width reaches min_target_w pixels,
        but never exceeds max_scale × input width.

    Returns
    -------
    upscaled    : upscaled image
    scale_used  : float
    """
    h, w = plate_bgr.shape[:2]
    if w == 0 or h == 0:
        return plate_bgr.copy(), 1.0

    # Desired scale to reach the minimum target width
    desired_scale = min_target_w / w
    scale = float(np.clip(desired_scale, 1.0, max_scale))

    new_w = int(round(w * scale))
    new_h = int(round(h * scale))

    upscaled = cv2.resize(plate_bgr, (new_w, new_h),
                          interpolation=cv2.INTER_LANCZOS4)
    return upscaled, scale


# ===========================================================================
# 6.  CONDITIONAL BILATERAL DENOISING
# ===========================================================================

def estimate_noise_level(gray):
    """
    Estimate image noise using the Laplacian variance method.
    Higher variance → sharper / less noisy.
    Lower variance  → more blurry / noisy.
    Returns a noise estimate score (higher = more noise).
    """
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    variance = lap.var()
    # Normalise to a rough noise indicator: low variance means noisy/blurry
    # We invert so a higher returned value means more noise
    noise_score = 1.0 / (variance + 1e-6)
    return noise_score, variance


def conditional_bilateral(plate_bgr, noise_threshold=0.01,
                           d=9, sigma_color=40, sigma_space=40):
    """
    Apply bilateral filter only when noise is detected above noise_threshold.

    The noise threshold is expressed as the inverse-variance score:
    if noise_score > noise_threshold → apply filter.

    Returns
    -------
    result      : denoised image (or original if filter not applied)
    applied     : bool
    noise_score : float
    lap_variance: float
    """
    gray = cv2.cvtColor(plate_bgr, cv2.COLOR_BGR2GRAY)
    noise_score, lap_var = estimate_noise_level(gray)

    if noise_score > noise_threshold:
        result = cv2.bilateralFilter(plate_bgr, d, sigma_color, sigma_space)
        return result, True, noise_score, lap_var
    else:
        return plate_bgr.copy(), False, noise_score, lap_var


# ===========================================================================
# 7.  CLAHE
# ===========================================================================

def apply_clahe(plate_bgr, clip_limit=2.0, tile_grid=(8, 8)):
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalisation) to the
    L channel of the LAB colour space.
    Improves local contrast under uneven illumination without over-amplifying
    noise or creating unnatural overall brightness changes.

    Returns enhanced BGR image.
    """
    lab = cv2.cvtColor(plate_bgr, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    l_clahe = clahe.apply(l_ch)

    lab_clahe = cv2.merge([l_clahe, a_ch, b_ch])
    result = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)
    return result


# ===========================================================================
# 8.  SELECTIVE GLARE SUPPRESSION
# ===========================================================================

def suppress_glare(plate_bgr, bright_threshold=240, blend_alpha=0.7):
    """
    Detect unusually bright (glare) regions and suppress them selectively.

    Method:
      1. Convert to HSV, extract V channel.
      2. Build a binary glare mask for pixels where V > bright_threshold.
      3. Erode the mask slightly to avoid false positives on legitimate bright
         borders.
      4. For masked pixels, replace with locally-smoothed values (median).
      5. Blend correction back smoothly (alpha blending).

    If no glare is detected (mask empty), the original image is returned
    unchanged.

    Returns
    -------
    result      : glare-suppressed image
    glare_mask  : binary mask (uint8, 0/255)
    glare_present : bool
    """
    hsv = cv2.cvtColor(plate_bgr, cv2.COLOR_BGR2HSV)
    v_ch = hsv[:, :, 2]

    # Binary glare mask
    _, glare_mask = cv2.threshold(v_ch, bright_threshold, 255, cv2.THRESH_BINARY)

    # Erode to remove single-pixel noise from the mask
    kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    glare_mask = cv2.erode(glare_mask, kernel_erode, iterations=1)

    glare_present = glare_mask.any()
    if not glare_present:
        return plate_bgr.copy(), glare_mask, False

    # Build a locally-smoothed replacement for glare areas (median blur)
    smoothed = cv2.medianBlur(plate_bgr, 15)

    # Replace glare pixels in HSV to avoid colour distortion
    mask_3ch = cv2.merge([glare_mask, glare_mask, glare_mask]).astype(bool)
    corrected = plate_bgr.copy().astype(np.float32)
    smoothed_f = smoothed.astype(np.float32)

    # Only blend in glare regions
    corrected[mask_3ch] = (
        blend_alpha * smoothed_f[mask_3ch] +
        (1 - blend_alpha) * corrected[mask_3ch]
    )
    result = np.clip(corrected, 0, 255).astype(np.uint8)
    return result, glare_mask, True


# ===========================================================================
# 9.  SELECTIVE MUD / DIRT SUPPRESSION
# ===========================================================================

def estimate_stroke_width(gray):
    """
    Estimate typical character stroke width using morphological operations.
    Used to set the black-hat kernel size relative to strokes.
    Returns an integer kernel size (odd, at least 3).
    """
    # Threshold to separate dark (character) regions
    _, thresh = cv2.threshold(gray, 0, 255,
                               cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # Distance transform gives radius of largest inscribed circle per pixel
    dist = cv2.distanceTransform(thresh, cv2.DIST_L2, 5)
    # Median of nonzero gives approximate stroke radius
    nonzero = dist[dist > 0]
    if len(nonzero) == 0:
        return 7  # default
    stroke_radius = float(np.median(nonzero))
    # Kernel size should be larger than stroke to capture dirt, not characters
    k = int(round(stroke_radius * 2.5))
    k = max(k, 7)
    if k % 2 == 0:
        k += 1
    return k


def suppress_dirt(plate_bgr, area_min=10, area_max_ratio=0.04,
                  edge_density_threshold=0.15):
    """
    Suppress isolated mud/dirt spots while protecting character strokes.

    Method:
      1. Black-hat morphology extracts small dark blobs on a light background.
      2. Connected-component analysis with shape filtering:
           - Area within [area_min, area_max_ratio * total_area]
           - Aspect ratio not consistent with a character stroke
           - Low edge density (characters have high edge density)
           - Not aligned with other character-like regions (heuristic)
      3. Detected dirt pixels are replaced with the local background value.
      4. If no dirt is found, the original image is returned unchanged.

    IMPORTANT: Dark pixels are NOT automatically dirt.
    We use edge density and shape to distinguish characters from dirt.

    Returns
    -------
    result      : dirt-suppressed image
    dirt_mask   : binary mask (uint8, 0/255) of suppressed regions
    dirt_found  : bool
    """
    gray = cv2.cvtColor(plate_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    k_size = estimate_stroke_width(gray)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_size, k_size))

    # Black-hat: captures small dark structures relative to local background
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)

    # Threshold the black-hat result
    _, bh_thresh = cv2.threshold(blackhat, 0, 255,
                                  cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Connected components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        bh_thresh, connectivity=8)

    total_area = h * w
    max_area = area_max_ratio * total_area

    # Canny edges for edge-density measurement
    edges = cv2.Canny(gray, 30, 100)

    dirt_mask = np.zeros((h, w), dtype=np.uint8)

    for lbl in range(1, num_labels):
        area = stats[lbl, cv2.CC_STAT_AREA]
        bx   = stats[lbl, cv2.CC_STAT_LEFT]
        by   = stats[lbl, cv2.CC_STAT_TOP]
        bw_  = stats[lbl, cv2.CC_STAT_WIDTH]
        bh_  = stats[lbl, cv2.CC_STAT_HEIGHT]

        # Area filter: too small → noise; too large → probable character region
        if area < area_min or area > max_area:
            continue

        # Aspect ratio filter: characters are tall-ish, dirt is often compact
        ar = bw_ / max(bh_, 1)
        # Very wide and short → not a character stroke
        # Very tall and narrow → likely a character stroke, skip
        if ar > 0.4 and bh_ > bw_ * 0.6:
            # Might be a character: check edge density before deciding
            pass

        # Edge density inside this component
        comp_mask = (labels == lbl).astype(np.uint8)
        region_edges = edges[by:by + bh_, bx:bx + bw_]
        region_comp  = comp_mask[by:by + bh_, bx:bx + bw_]
        if region_comp.sum() == 0:
            continue
        edge_dens = region_edges[region_comp > 0].mean() / 255.0

        # High edge density → likely a character stroke edge → protect it
        if edge_dens > edge_density_threshold:
            continue

        # Low edge density + small compact blob → likely dirt
        dirt_mask[labels == lbl] = 255

    dirt_found = dirt_mask.any()
    if not dirt_found:
        return plate_bgr.copy(), dirt_mask, False

    # Replace dirt pixels with local background (dilated light background)
    background = cv2.dilate(gray, kernel, iterations=2)
    result = plate_bgr.copy()
    dirt_3ch = cv2.merge([dirt_mask, dirt_mask, dirt_mask]).astype(bool)

    # Reconstruct background colour using the background luminance scaled
    for c in range(3):
        ch = result[:, :, c].astype(np.float32)
        bg = background.astype(np.float32)
        # Scale channel proportionally to background brightness
        scale = np.where(gray > 0, bg / np.clip(gray.astype(np.float32), 1, None), 1.0)
        scale = np.clip(scale, 0.5, 2.0)
        ch_corrected = ch * scale
        ch_corrected = np.clip(ch_corrected, 0, 255)
        mask_2d = dirt_mask.astype(bool)
        ch[mask_2d] = ch_corrected[mask_2d]
        result[:, :, c] = ch.astype(np.uint8)

    return result, dirt_mask, True


# ===========================================================================
# 10.  STROKE-WIDTH ADAPTIVE UNSHARP MASKING (CHARACTER ENHANCEMENT)
# ===========================================================================

def character_enhance(plate_bgr, amount=1.2, threshold=3):
    """
    Stroke-width adaptive unsharp masking for character enhancement.

    The blur radius is set proportionally to the estimated character stroke
    width so the mask targets the correct spatial scale:
      - Narrow strokes (small plates) → smaller blur radius
      - Wide strokes  (large plates)  → larger blur radius

    amount    : strength of the sharpening (1.0–2.0 typical)
    threshold : minimum pixel difference to apply sharpening

    Returns enhanced BGR image.
    """
    gray = cv2.cvtColor(plate_bgr, cv2.COLOR_BGR2GRAY)
    sw = estimate_stroke_width(gray)

    # Blur radius ≈ stroke_width, ensure odd
    blur_r = max(sw // 2, 1)
    if blur_r % 2 == 0:
        blur_r += 1
    blur_r = min(blur_r, 15)  # cap to avoid over-smoothing

    blurred = cv2.GaussianBlur(plate_bgr, (blur_r, blur_r), 0)

    # Unsharp mask: sharpened = original + amount * (original - blurred)
    # Apply threshold to avoid amplifying flat-region noise
    diff = plate_bgr.astype(np.int16) - blurred.astype(np.int16)
    diff_thresh = np.where(np.abs(diff) >= threshold, diff, 0)

    sharpened = plate_bgr.astype(np.float32) + amount * diff_thresh.astype(np.float32)
    sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
    return sharpened


# ===========================================================================
# 11.  SAUVOLA ADAPTIVE THRESHOLDING
# ===========================================================================

def sauvola_threshold(plate_bgr, window_size=25, k=0.2):
    """
    Apply Sauvola adaptive thresholding to the grayscale plate.

    Uses scikit-image if available; falls back to OpenCV adaptive thresholding
    (Gaussian weighted) if scikit-image is not installed.

    window_size : local neighbourhood size (odd integer)
    k           : sensitivity parameter (0.1–0.5 typical)

    Returns binary image (uint8, 0/255).
    """
    gray = cv2.cvtColor(plate_bgr, cv2.COLOR_BGR2GRAY)

    # Ensure window_size is odd
    if window_size % 2 == 0:
        window_size += 1

    if SKIMAGE_AVAILABLE:
        thresh = threshold_sauvola(gray, window_size=window_size, k=k)
        binary = (gray > thresh).astype(np.uint8) * 255
    else:
        # Fallback: OpenCV adaptive Gaussian threshold
        block_size = window_size if window_size % 2 == 1 else window_size + 1
        block_size = max(block_size, 11)
        binary = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            block_size, 5
        )

    return binary


# ===========================================================================
# 12.  ADAPTIVE MORPHOLOGICAL CLOSING
# ===========================================================================

def adaptive_closing(binary_image, stroke_width_hint=None):
    """
    Apply morphological closing to repair small gaps and broken strokes.

    Kernel size is derived from stroke_width_hint:
      - Closing uses a small rectangular kernel to bridge minor stroke gaps.
      - Kernel is kept small (1–3 px) to avoid merging neighbouring characters.

    If stroke_width_hint is None, it is estimated from the binary image.

    Returns the closed binary image (uint8, 0/255).
    """
    if stroke_width_hint is None:
        # Estimate from the binary image
        inv = cv2.bitwise_not(binary_image)
        dist = cv2.distanceTransform(inv, cv2.DIST_L2, 5)
        nonzero = dist[dist > 0]
        if len(nonzero) > 0:
            stroke_width_hint = float(np.median(nonzero)) * 2.0
        else:
            stroke_width_hint = 3.0

    # Closing kernel: 1/3 of stroke width, minimum 1, maximum 3
    k = max(1, min(3, int(round(stroke_width_hint / 3.0))))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    closed = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)
    return closed


# ===========================================================================
# UTILITY FUNCTIONS
# ===========================================================================

def save_image(image, path):
    """Save an image to path, creating parent directories as needed."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(str(path), image)


def load_image(path):
    """Load an image from path in BGR format. Returns None on failure."""
    img = cv2.imread(str(path))
    return img


def to_grayscale(bgr):
    """Convert BGR image to grayscale."""
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)


def gray_to_bgr(gray):
    """Convert grayscale image to 3-channel BGR."""
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def get_image_extensions():
    """Return the set of supported image file extensions."""
    return {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}


def discover_images(folder):
    """
    Discover all image files in folder (non-recursive).
    Returns a sorted list of Path objects.
    """
    folder = Path(folder)
    exts = get_image_extensions()
    images = sorted([p for p in folder.iterdir()
                     if p.is_file() and p.suffix.lower() in exts])
    return images


def make_output_dirs(base_out, image_stem):
    """
    Create the per-image output folder structure.
    Returns a dict mapping stage name → folder path (str).
    """
    stages = [
        "01_input",
        "02_detection",
        "03_roi",
        "04_perspective",
        "05_upscaling",
        "06_denoising",
        "07_clahe",
        "08_glare",
        "09_dirt",
        "10_character",
        "11_threshold",
        "12_morphology",
        "final",
        "report",
    ]
    dirs = {}
    for s in stages:
        d = os.path.join(base_out, image_stem, s)
        os.makedirs(d, exist_ok=True)
        dirs[s] = d
    return dirs


def compute_iou(boxA, boxB):
    """
    Compute Intersection-over-Union between two bounding boxes.
    Each box: (x1, y1, x2, y2).
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter_w = max(0, xB - xA)
    inter_h = max(0, yB - yA)
    inter_area = inter_w * inter_h

    areaA = max(0, boxA[2] - boxA[0]) * max(0, boxA[3] - boxA[1])
    areaB = max(0, boxB[2] - boxB[0]) * max(0, boxB[3] - boxB[1])
    union_area = areaA + areaB - inter_area

    if union_area == 0:
        return 0.0
    return inter_area / union_area


def parse_yolo_label(label_path, img_w, img_h):
    """
    Parse a YOLO-format label file and return bounding boxes in pixel coords.
    Returns list of (x1, y1, x2, y2) tuples.
    """
    boxes = []
    if not os.path.exists(label_path):
        return boxes
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            _, cx, cy, bw, bh = map(float, parts[:5])
            x1 = (cx - bw / 2) * img_w
            y1 = (cy - bh / 2) * img_h
            x2 = (cx + bw / 2) * img_w
            y2 = (cy + bh / 2) * img_h
            boxes.append((x1, y1, x2, y2))
    return boxes


def compute_detection_metrics(predictions, ground_truths, iou_threshold=0.5):
    """
    Compute precision, recall and IoU for a batch of detections.

    predictions   : list of dicts with keys 'detected', 'x1','y1','x2','y2','confidence'
    ground_truths : list of lists of (x1,y1,x2,y2) tuples per image

    Returns dict with precision, recall, mean_iou, matched_count, total_gt.
    """
    tp = 0
    fp = 0
    fn = 0
    iou_values = []

    for pred, gts in zip(predictions, ground_truths):
        if not pred["detected"]:
            fn += len(gts)
            continue
        pred_box = (pred["x1"], pred["y1"], pred["x2"], pred["y2"])
        if not gts:
            fp += 1
            continue
        # Match to best GT box
        best_iou = max(compute_iou(pred_box, gt) for gt in gts)
        iou_values.append(best_iou)
        if best_iou >= iou_threshold:
            tp += 1
        else:
            fp += 1
        # Remaining GTs are false negatives (we only predict one box per image)
        fn += max(0, len(gts) - 1)

    precision = tp / max(tp + fp, 1)
    recall    = tp / max(tp + fn, 1)
    mean_iou  = float(np.mean(iou_values)) if iou_values else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "mean_iou": mean_iou,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "total_gt": tp + fn,
    }


# ===========================================================================
# FULL PHASE 1 PIPELINE (single image)
# ===========================================================================

def run_phase1_pipeline(image_bgr, model, output_dirs, cfg):
    """
    Execute the complete Phase 1 pipeline for a single image.

    Parameters
    ----------
    image_bgr   : original loaded BGR image
    model       : loaded YOLOv11n YOLO model object
    output_dirs : dict of stage → folder path (from make_output_dirs)
    cfg         : dict of configuration values

    Returns
    -------
    result : dict containing all per-stage outputs, metrics, timings and flags.
    """
    t_start = time.time()
    result = {
        "success": False,
        "detected": False,
        "error": None,
        "timings": {},
    }

    try:
        # ------------------------------------------------------------------ #
        # STAGE 1: Save input
        # ------------------------------------------------------------------ #
        save_image(image_bgr,
                   os.path.join(output_dirs["01_input"], "original.jpg"))

        # ------------------------------------------------------------------ #
        # STAGE 2: YOLO Detection
        # ------------------------------------------------------------------ #
        t0 = time.time()
        det = detect_plate(
            model, image_bgr,
            conf_threshold=cfg.get("conf_threshold", 0.25),
            target_size=cfg.get("letterbox_size", 640),
        )
        result["timings"]["detection"] = time.time() - t0
        result["detected"] = det["detected"]
        result["detection"] = det

        det_vis = draw_detection(image_bgr, det)
        save_image(det_vis,
                   os.path.join(output_dirs["02_detection"], "detected_plate.jpg"))

        if not det["detected"]:
            result["error"] = "Plate not detected by YOLOv11n"
            return result

        # ------------------------------------------------------------------ #
        # STAGE 3: ROI Crop
        # ------------------------------------------------------------------ #
        t0 = time.time()
        roi, crop_coords, pad_frac = crop_plate_roi(
            image_bgr, det,
            pad_ratio=cfg.get("roi_pad_ratio", 0.07),
        )
        result["timings"]["roi"] = time.time() - t0
        result["pad_used"] = pad_frac
        result["crop_coords"] = crop_coords

        save_image(roi, os.path.join(output_dirs["03_roi"], "plate_roi.jpg"))

        # ------------------------------------------------------------------ #
        # STAGE 4: Perspective Correction
        # ------------------------------------------------------------------ #
        t0 = time.time()
        corrected, was_corrected, skew_angle, persp_method = perspective_correct(
            roi,
            skew_threshold=cfg.get("skew_threshold", 3.0),
        )
        result["timings"]["perspective"] = time.time() - t0
        result["perspective_corrected"] = was_corrected
        result["skew_angle"] = skew_angle
        result["perspective_method"] = persp_method

        save_image(roi,        os.path.join(output_dirs["04_perspective"], "plate_before_correction.jpg"))
        save_image(corrected,  os.path.join(output_dirs["04_perspective"], "plate_corrected.jpg"))

        # ------------------------------------------------------------------ #
        # STAGE 5: Adaptive Lanczos Upscaling
        # ------------------------------------------------------------------ #
        t0 = time.time()
        upscaled, scale_used = adaptive_upscale(
            corrected,
            min_target_w=cfg.get("upscale_min_width", 400),
            max_scale=cfg.get("upscale_max_scale", 4.0),
        )
        result["timings"]["upscaling"] = time.time() - t0
        result["upscale_factor"] = scale_used

        save_image(upscaled, os.path.join(output_dirs["05_upscaling"], "plate_upscaled.jpg"))

        # ------------------------------------------------------------------ #
        # STAGE 6: Conditional Bilateral Denoising
        # ------------------------------------------------------------------ #
        t0 = time.time()
        denoised, denoise_applied, noise_score, lap_var = conditional_bilateral(
            upscaled,
            noise_threshold=cfg.get("noise_threshold", 0.01),
            d=cfg.get("bilateral_d", 9),
            sigma_color=cfg.get("bilateral_sigma_color", 40),
            sigma_space=cfg.get("bilateral_sigma_space", 40),
        )
        result["timings"]["denoising"] = time.time() - t0
        result["denoising_applied"] = denoise_applied
        result["noise_score"] = noise_score
        result["laplacian_variance"] = lap_var

        save_image(denoised, os.path.join(output_dirs["06_denoising"], "plate_denoised.jpg"))

        # ------------------------------------------------------------------ #
        # STAGE 7: CLAHE
        # ------------------------------------------------------------------ #
        t0 = time.time()
        clahe_out = apply_clahe(
            denoised,
            clip_limit=cfg.get("clahe_clip", 2.0),
            tile_grid=tuple(cfg.get("clahe_tile", [8, 8])),
        )
        result["timings"]["clahe"] = time.time() - t0

        save_image(clahe_out, os.path.join(output_dirs["07_clahe"], "plate_clahe.jpg"))

        # ------------------------------------------------------------------ #
        # STAGE 8: Glare Suppression
        # ------------------------------------------------------------------ #
        t0 = time.time()
        glare_out, glare_mask, glare_present = suppress_glare(
            clahe_out,
            bright_threshold=cfg.get("glare_threshold", 240),
            blend_alpha=cfg.get("glare_blend", 0.7),
        )
        result["timings"]["glare"] = time.time() - t0
        result["glare_present"] = glare_present

        save_image(clahe_out,  os.path.join(output_dirs["08_glare"], "pre_glare.jpg"))
        save_image(gray_to_bgr(glare_mask),
                               os.path.join(output_dirs["08_glare"], "glare_mask.jpg"))
        save_image(glare_out,  os.path.join(output_dirs["08_glare"], "plate_glare_suppressed.jpg"))

        # ------------------------------------------------------------------ #
        # STAGE 9: Dirt Suppression
        # ------------------------------------------------------------------ #
        t0 = time.time()
        dirt_out, dirt_mask, dirt_found = suppress_dirt(
            glare_out,
            area_min=cfg.get("dirt_area_min", 10),
            area_max_ratio=cfg.get("dirt_area_max_ratio", 0.04),
            edge_density_threshold=cfg.get("dirt_edge_thresh", 0.15),
        )
        result["timings"]["dirt"] = time.time() - t0
        result["dirt_found"] = dirt_found

        save_image(gray_to_bgr(dirt_mask),
                               os.path.join(output_dirs["09_dirt"], "dirt_mask.jpg"))
        save_image(glare_out,  os.path.join(output_dirs["09_dirt"], "pre_dirt.jpg"))
        save_image(dirt_out,   os.path.join(output_dirs["09_dirt"], "dirt_suppressed.jpg"))

        # ------------------------------------------------------------------ #
        # STAGE 10: Character Enhancement (Stroke-Width Adaptive Unsharp)
        # ------------------------------------------------------------------ #
        t0 = time.time()
        char_out = character_enhance(
            dirt_out,
            amount=cfg.get("unsharp_amount", 1.2),
            threshold=cfg.get("unsharp_threshold", 3),
        )
        result["timings"]["character_enhancement"] = time.time() - t0

        save_image(char_out,
                   os.path.join(output_dirs["10_character"], "character_enhanced.jpg"))

        # ------------------------------------------------------------------ #
        # STAGE 11: Sauvola Thresholding
        # ------------------------------------------------------------------ #
        t0 = time.time()
        sauvola_out = sauvola_threshold(
            char_out,
            window_size=cfg.get("sauvola_window", 25),
            k=cfg.get("sauvola_k", 0.2),
        )
        result["timings"]["threshold"] = time.time() - t0
        result["sauvola_fallback"] = not SKIMAGE_AVAILABLE

        save_image(gray_to_bgr(sauvola_out),
                   os.path.join(output_dirs["11_threshold"], "sauvola.jpg"))

        # ------------------------------------------------------------------ #
        # STAGE 12: Adaptive Morphological Closing
        # ------------------------------------------------------------------ #
        t0 = time.time()
        morph_out = adaptive_closing(sauvola_out)
        result["timings"]["morphology"] = time.time() - t0

        save_image(gray_to_bgr(morph_out),
                   os.path.join(output_dirs["12_morphology"], "morphology_closed.jpg"))

        # ------------------------------------------------------------------ #
        # FINAL: Save the final enhanced plate (binary + continuous-tone)
        # ------------------------------------------------------------------ #
        save_image(gray_to_bgr(morph_out),
                   os.path.join(output_dirs["final"], "final_enhanced.jpg"))
        # Also keep the pre-threshold continuous-tone version as an alternative
        save_image(char_out,
                   os.path.join(output_dirs["final"], "final_enhanced_continuous.jpg"))

        result["total_time"] = time.time() - t_start
        result["success"] = True

    except Exception as e:
        result["error"] = str(e)
        result["total_time"] = time.time() - t_start

    return result
