"""
bbox_detector.py
────────────────
Weakly-supervised bounding box detection for LUMEN.

Pipeline:
  1. Extract GradCAM++ maps from MaxViT stages 2 and 3
  2. Fuse and sharpen the maps
  3. Detect lung region from the CT image itself (image processing)
  4. Mask the activation map — zero out anything outside the lungs
  5. Threshold + connected components → tight bounding boxes

Step 4 is the key addition: boxes that fall outside the lung fields
(abdomen, liver, ribs edge) are discarded before they are returned.
"""

import cv2
import numpy as np
import torch
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# ─── Per-class colours (RGB) ──────────────────────────────────────────────────
CLASS_COLORS_RGB = {
    'ILD':          (52,  152, 219),
    'Lung Cancer':  (231, 76,  60),
    'Normal':       (46,  204, 113),
    'Tuberculosis': (230, 126, 34),
    'COVID':        (155, 89,  182),
    'Pneumonia':    (241, 196, 15),
}

LABEL_KEYS = ['ILD', 'Lung Cancer', 'Normal', 'Tuberculosis', 'COVID', 'Pneumonia']

DISPLAY_NAMES = {
    'ILD':          'ILD/Fibrosis',
    'Lung Cancer':  'Lung Cancer/Nodules',
    'Normal':       'Normal',
    'Tuberculosis': 'Tuberculosis',
    'COVID':        'COVID-19',
    'Pneumonia':    'Pneumonia',
}

THRESHOLDS = {
    'ILD':          0.65,
    'Lung Cancer':  0.28,
    'Normal':       0.81,
    'Tuberculosis': 0.88,
    'COVID':        0.50,
    'Pneumonia':    0.68,
}

# ─── Tuning constants ─────────────────────────────────────────────────────────
CAM_THRESHOLD       = 0.50   # activation cutoff
MIN_AREA_FRAC       = 0.008  # min box area fraction
MAX_AREA_FRAC       = 0.55   # max box area fraction — rejects full-image blobs
MAX_BOXES           = 3      # max boxes per class
LUNG_OVERLAP_MIN    = 0.25   # box must overlap lung mask by at least 25%


# ─── Lung region detection ────────────────────────────────────────────────────

def _get_lung_mask(image_rgb_uint8):
    """
    Segment the lung fields from a CT image using image processing only.

    Handles both:
    - Normal CT: lungs are dark (air-filled)
    - Diseased CT (ILD, Pneumonia, COVID): lungs are partially bright
      due to ground-glass opacity, consolidation, honeycombing

    Strategy: instead of thresholding for dark lung regions, we use
    the body boundary to define a broad ROI, then exclude clearly
    non-lung structures (spine, liver, heart centre). This is more
    robust across different pathologies and windowing.

    Returns all-ones mask (no filtering) if segmentation is uncertain.
    """
    h, w = image_rgb_uint8.shape[:2]
    gray = cv2.cvtColor(image_rgb_uint8, cv2.COLOR_RGB2GRAY)

    # ── Step 1: find the body outline (patient boundary) ─────────────────────
    # Threshold to separate patient from black scanner background
    _, body = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)

    # Flood-fill from corners to remove scanner background noise
    flood = body.copy()
    flood_mask = np.zeros((h + 2, w + 2), np.uint8)
    for corner in [(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)]:
        cv2.floodFill(flood, flood_mask, (corner[1], corner[0]), 0)

    # Clean up body mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    body_mask = cv2.morphologyEx(flood, cv2.MORPH_CLOSE, kernel)

    # If body detection failed, return permissive mask
    if body_mask.sum() < 0.05 * h * w * 255:
        return np.ones((h, w), dtype=np.uint8) * 255

    # ── Step 2: find the bounding box of the body ─────────────────────────────
    body_coords = cv2.findNonZero(body_mask)
    if body_coords is None:
        return np.ones((h, w), dtype=np.uint8) * 255

    bx, by, bw, bh = cv2.boundingRect(body_coords)

    # ── Step 3: define lung ROI as central portion of the body ───────────────
    # Lungs occupy roughly the middle 80% width and middle 70% height
    # of the body bounding box on axial CT slices.
    # This works even when lungs are bright (ILD) because we're using
    # the body boundary rather than lung darkness.
    margin_x = int(bw * 0.08)
    margin_y_top = int(bh * 0.05)
    margin_y_bot = int(bh * 0.15)   # exclude liver/abdomen at bottom

    roi_x1 = bx + margin_x
    roi_y1 = by + margin_y_top
    roi_x2 = bx + bw - margin_x
    roi_y2 = by + bh - margin_y_bot

    # Clamp to image bounds
    roi_x1 = max(0, roi_x1)
    roi_y1 = max(0, roi_y1)
    roi_x2 = min(w, roi_x2)
    roi_y2 = min(h, roi_y2)

    lung_mask = np.zeros((h, w), dtype=np.uint8)
    lung_mask[roi_y1:roi_y2, roi_x1:roi_x2] = 255

    # ── Step 4: exclude the mediastinum (central chest structures) ────────────
    # Heart and great vessels occupy the central ~25% width of the chest.
    # Exclude this narrow vertical strip to avoid heart activations.
    centre_x     = (roi_x1 + roi_x2) // 2
    mediastinum_w = int((roi_x2 - roi_x1) * 0.18)
    med_x1 = centre_x - mediastinum_w
    med_x2 = centre_x + mediastinum_w
    lung_mask[roi_y1:roi_y2, med_x1:med_x2] = 0

    # ── Step 5: mask out scanner corners using the body mask ─────────────────
    # CT images are circular — corners are scanner air, not patient.
    # AND the body_mask itself into the lung_mask to remove any
    # activations that fall outside the actual patient body.
    lung_mask = cv2.bitwise_and(lung_mask, body_mask)

    # ── Step 6: dilate to avoid clipping boxes at lung edges ─────────────────
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))
    lung_mask = cv2.dilate(lung_mask, dilate_kernel)

    # Re-apply body mask after dilation so we don't dilate back into corners
    lung_mask = cv2.bitwise_and(lung_mask, body_mask)

    # Clamp to ROI vertical bounds
    lung_mask[:roi_y1, :] = 0
    lung_mask[roi_y2:, :] = 0

    return lung_mask


def _box_lung_overlap(x1, y1, x2, y2, lung_mask):
    """
    Return the fraction of the bounding box area that falls inside
    the lung mask.  Range [0, 1].
    """
    h, w = lung_mask.shape
    # Clamp box to image bounds
    bx1 = max(0, x1)
    by1 = max(0, y1)
    bx2 = min(w, x2)
    by2 = min(h, y2)

    if bx2 <= bx1 or by2 <= by1:
        return 0.0

    roi        = lung_mask[by1:by2, bx1:bx2]
    box_area   = (bx2 - bx1) * (by2 - by1)
    lung_pixels = (roi > 0).sum()

    return lung_pixels / box_area if box_area > 0 else 0.0


# ─── CAM helpers ──────────────────────────────────────────────────────────────

def _get_stage_layer(model, stage_idx):
    stage  = model.backbone.stages[stage_idx]
    blocks = list(stage.children())
    return blocks[-1] if blocks else stage


def _cam_for_class(model, tensor, class_idx, target_layers):
    cam       = GradCAMPlusPlus(model=model, target_layers=target_layers)
    grayscale = cam(
        input_tensor=tensor,
        targets=[ClassifierOutputTarget(class_idx)]
    )[0]
    return grayscale.astype(np.float32)


def _fuse_maps(map2, map3, w2=0.35, w3=0.65):
    if map2.shape != map3.shape:
        map2 = cv2.resize(map2, (map3.shape[1], map3.shape[0]))
    fused = w2 * map2 + w3 * map3

    mn, mx = fused.min(), fused.max()
    if mx - mn > 1e-6:
        fused = (fused - mn) / (mx - mn)

    # Sharpen — suppress background noise
    fused = np.power(fused, 1.5)

    mn, mx = fused.min(), fused.max()
    if mx - mn > 1e-6:
        fused = (fused - mn) / (mx - mn)

    return fused


# ─── Box extraction ───────────────────────────────────────────────────────────

def _map_to_boxes(fused_map, orig_size, lung_mask,
                  threshold=CAM_THRESHOLD,
                  min_area_frac=MIN_AREA_FRAC,
                  max_area_frac=MAX_AREA_FRAC,
                  max_boxes=MAX_BOXES,
                  lung_overlap_min=LUNG_OVERLAP_MIN):
    """
    Convert a fused activation map to bounding boxes, filtered by
    the lung mask so only boxes inside the lung fields are kept.
    """
    orig_w, orig_h = orig_size
    map_h, map_w   = fused_map.shape
    total_px       = map_h * map_w

    # ── Apply lung mask to the CAM map before thresholding ────────────────────
    # Resize lung mask to match CAM resolution
    lung_resized = cv2.resize(lung_mask, (map_w, map_h),
                              interpolation=cv2.INTER_NEAREST)
    lung_float   = (lung_resized > 0).astype(np.float32)

    # Zero out activations outside the lung region
    masked_map = fused_map * lung_float

    # ── Threshold → binary ────────────────────────────────────────────────────
    binary = (masked_map >= threshold).astype(np.uint8) * 255

    # ── Morphological cleanup ─────────────────────────────────────────────────
    kernel      = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    binary      = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary      = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_open)

    # ── Connected components ──────────────────────────────────────────────────
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary)

    min_px = min_area_frac * total_px
    max_px = max_area_frac * total_px
    boxes  = []

    for comp in range(1, num_labels):
        area = stats[comp, cv2.CC_STAT_AREA]
        if area < min_px or area > max_px:
            continue

        x = stats[comp, cv2.CC_STAT_LEFT]
        y = stats[comp, cv2.CC_STAT_TOP]
        w = stats[comp, cv2.CC_STAT_WIDTH]
        h = stats[comp, cv2.CC_STAT_HEIGHT]

        # Reject full-image spanning boxes
        if w > map_w * 0.85 or h > map_h * 0.85:
            continue

        # Mean activation inside component
        mask_comp  = (labels == comp).astype(np.float32)
        confidence = float((masked_map * mask_comp).sum() / mask_comp.sum())

        # Scale to original image coords
        sx = orig_w / map_w
        sy = orig_h / map_h

        x1 = max(0,      int(x       * sx) - 2)
        y1 = max(0,      int(y       * sy) - 2)
        x2 = min(orig_w, int((x + w) * sx) + 2)
        y2 = min(orig_h, int((y + h) * sy) + 2)

        # ── Lung overlap filter ───────────────────────────────────────────────
        # Check overlap against FULL-resolution lung mask
        overlap = _box_lung_overlap(x1, y1, x2, y2, lung_mask)
        if overlap < lung_overlap_min:
            continue    # box is mostly outside the lungs — discard

        boxes.append((x1, y1, x2, y2, confidence))

    boxes.sort(key=lambda b: b[4], reverse=True)
    return boxes[:max_boxes]


# ─── Public API ───────────────────────────────────────────────────────────────

def detect_boxes(model, tensor, probs, orig_size=(224, 224),
                 cam_threshold=CAM_THRESHOLD,
                 image_rgb_uint8=None):
    """
    Run detection pipeline for every detected class.

    Parameters
    ----------
    model            : LungMaxViTHybrid
    tensor           : (1, 3, 224, 224) tensor
    probs            : list of 6 sigmoid outputs
    orig_size        : (W, H) of the display image
    cam_threshold    : activation cutoff for boxes
    image_rgb_uint8  : (H, W, 3) uint8 array of the CT image — used to
                       build the lung mask.  If None, no lung filtering.

    Returns
    -------
    dict {label_key: {display, prob, boxes, color, cam_map, lung_mask}}
    """
    for param in model.parameters():
        param.requires_grad_(True)
    model.eval()

    # Build lung mask once — shared across all classes
    if image_rgb_uint8 is not None:
        lung_mask = _get_lung_mask(image_rgb_uint8)
    else:
        # No image provided — permissive mask (no filtering)
        lung_mask = np.ones((orig_size[1], orig_size[0]), dtype=np.uint8) * 255

    stage2_layer = [_get_stage_layer(model, 2)]
    stage3_layer = [_get_stage_layer(model, 3)]

    results = {}

    for i, key in enumerate(LABEL_KEYS):
        prob = probs[i]
        if prob < THRESHOLDS[key]:
            continue

        map2  = _cam_for_class(model, tensor, i, stage2_layer)
        map3  = _cam_for_class(model, tensor, i, stage3_layer)
        fused = _fuse_maps(map2, map3)
        boxes = _map_to_boxes(fused, orig_size, lung_mask,
                              threshold=cam_threshold)

        results[key] = {
            'display':   DISPLAY_NAMES[key],
            'prob':      prob,
            'boxes':     boxes,
            'color':     CLASS_COLORS_RGB[key],
            'cam_map':   fused,
            'lung_mask': lung_mask,
        }

    return results


def draw_boxes_on_image(image_rgb, detection_results,
                        line_thickness=2, show_confidence=False):
    """
    Draw bounding boxes on image_rgb (H×W×3 uint8).
    Labels are small and placed above each box.
    """
    annotated  = image_rgb.copy()
    font       = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.32
    thickness  = 1

    for key, data in detection_results.items():
        color   = data['color']
        bgr     = (color[2], color[1], color[0])
        display = data['display']
        prob    = data['prob']

        for x1, y1, x2, y2, conf in data['boxes']:
            cv2.rectangle(annotated, (x1, y1), (x2, y2), bgr, line_thickness)

            label_str = (f"{display} {prob:.0%} [{conf:.2f}]"
                         if show_confidence else f"{display} {prob:.0%}")

            (tw, th), baseline = cv2.getTextSize(
                label_str, font, font_scale, thickness)

            label_y = y1 - 3
            if label_y - th - baseline < 0:
                label_y = y2 + th + 3

            cv2.rectangle(annotated,
                          (x1, label_y - th - baseline - 1),
                          (x1 + tw + 4, label_y + 1),
                          bgr, cv2.FILLED)
            cv2.putText(annotated, label_str,
                        (x1 + 2, label_y - baseline),
                        font, font_scale,
                        (255, 255, 255), thickness, cv2.LINE_AA)

    return annotated


def render_bbox_figure(image_rgb_224, detection_results,
                       show_lung_mask=False):
    """
    Matplotlib figure:
      Left  — CT with all boxes + optional lung mask outline
      Right — per-class fused CAM heatmap with boxes
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    n_detected = len(detection_results)

    if n_detected == 0:
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        fig.patch.set_facecolor('#FFFFFF')
        ax.imshow(image_rgb_224, cmap='gray')
        ax.set_title('No conditions detected', color='gray', fontsize=10)
        ax.axis('off')
        return fig

    ncols     = 1 + n_detected
    fig, axes = plt.subplots(1, ncols, figsize=(4.5 * ncols, 4.5), dpi=120)
    fig.patch.set_facecolor('#FFFFFF')

    if ncols == 1:
        axes = [axes]

    # ── Left: annotated overview ──────────────────────────────────────────────
    img_uint8 = (image_rgb_224 * 255).astype(np.uint8)
    annotated = draw_boxes_on_image(img_uint8, detection_results,
                                    line_thickness=2)
    axes[0].imshow(annotated)

    # Optionally overlay lung mask outline
    if show_lung_mask and detection_results:
        first_key  = next(iter(detection_results))
        lung_mask  = detection_results[first_key].get('lung_mask')
        if lung_mask is not None:
            # Draw lung contour in cyan
            contours, _ = cv2.findContours(
                lung_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            lung_outline = annotated.copy()
            cv2.drawContours(lung_outline, contours, -1, (0, 255, 255), 1)
            axes[0].imshow(lung_outline)

    axes[0].set_title('Detected Regions', fontsize=10,
                      fontweight='bold', color='#1E3A08', pad=6)
    axes[0].axis('off')

    patches = []
    for key, data in detection_results.items():
        c = [v / 255 for v in data['color']]
        n = len(data['boxes'])
        patches.append(mpatches.Patch(
            color=c,
            label=f"{data['display']} {data['prob']:.0%} "
                  f"({n} region{'s' if n != 1 else ''})"))
    if patches:
        axes[0].legend(handles=patches, loc='lower left',
                       fontsize=7, framealpha=0.88,
                       facecolor='white', edgecolor='#cccccc')

    # ── Right: per-class CAM panels ───────────────────────────────────────────
    for col_idx, (key, data) in enumerate(detection_results.items(), start=1):
        ax    = axes[col_idx]
        cam   = data['cam_map']
        color = [v / 255 for v in data['color']]

        ax.imshow(image_rgb_224, cmap='gray', alpha=0.6)
        ax.imshow(cam, cmap='hot', alpha=0.5, vmin=0, vmax=1)

        for x1, y1, x2, y2, conf in data['boxes']:
            rect = plt.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=2, edgecolor=color, facecolor='none')
            ax.add_patch(rect)
            ax.text(x1 + 2, y1 - 3, f"{data['prob']:.0%}",
                    fontsize=7, color='white', fontweight='bold',
                    bbox=dict(facecolor=color, edgecolor='none',
                              alpha=0.85, pad=1))

        ax.set_title(
            f"{data['display']}\n{data['prob']:.1%}  (stage2+3 CAM)",
            fontsize=8, fontweight='bold', color='#1E3A08', pad=5)
        ax.axis('off')

    plt.tight_layout(pad=1.0)
    return fig