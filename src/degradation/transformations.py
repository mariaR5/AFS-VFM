"""
Deterministic image degradation transformations for AFS-VFM.

Each transformation accepts an RGB uint8 image and a severity value in
[0, 1], and returns a degraded RGB uint8 image of the *same* dimensions.

All transformations are fully deterministic — no randomness is used.
"""

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# 1. Motion Blur
# ---------------------------------------------------------------------------

def motion_blur(image: np.ndarray, severity: float) -> np.ndarray:
    """Apply horizontal motion blur with increasing kernel length.

    Kernel length scales linearly from 1 px (severity 0, no blur) to
    51 px (severity 1).  At severity 0 the image is returned untouched.

    Args:
        image: RGB uint8 array of shape (H, W, 3).
        severity: Float in [0, 1] controlling blur intensity.

    Returns:
        Blurred image with the same shape and dtype.
    """
    # At severity 0 → kernel_size 1 (identity), severity 1 → 51
    kernel_size = int(1 + severity * 50)  # 1 → 51
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel_size = max(1, kernel_size)

    # kernel_size 1 means no blur at all — return original
    if kernel_size <= 1:
        return image.copy()

    # Horizontal motion blur kernel
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    kernel[kernel_size // 2, :] = 1.0
    kernel /= kernel_size

    blurred = cv2.filter2D(image, -1, kernel)
    return blurred


# ---------------------------------------------------------------------------
# 2. Occlusion
# ---------------------------------------------------------------------------

def occlusion(image: np.ndarray, severity: float) -> np.ndarray:
    """Overlay a centred grey rectangle that grows with severity.

    The occluded area scales linearly from 0 % (severity 0, untouched)
    to 80 % of the total image area (severity 1).

    Args:
        image: RGB uint8 array of shape (H, W, 3).
        severity: Float in [0, 1].

    Returns:
        Occluded image with the same shape and dtype.
    """
    result = image.copy()

    # At severity 0 → no occlusion at all
    if severity <= 0.0:
        return result

    h, w = image.shape[:2]

    # Fraction of total area to occlude: 0 % → 80 %
    fraction = severity * 0.80
    side_frac = np.sqrt(fraction)  # equal scaling on both axes

    occ_h = int(h * side_frac)
    occ_w = int(w * side_frac)

    y1 = (h - occ_h) // 2
    x1 = (w - occ_w) // 2

    # Fill with mid-grey (128, 128, 128)
    result[y1:y1 + occ_h, x1:x1 + occ_w] = 128

    return result


# ---------------------------------------------------------------------------
# 3. Lighting Reduction
# ---------------------------------------------------------------------------

def lighting(image: np.ndarray, severity: float) -> np.ndarray:
    """Reduce brightness by progressively dimming the V channel in HSV.

    At severity 0 the image is unchanged; at severity 1 brightness is
    reduced by 95 %.

    Args:
        image: RGB uint8 array of shape (H, W, 3).
        severity: Float in [0, 1].

    Returns:
        Dimmed image with the same shape and dtype.
    """
    # Direct RGB → HSV conversion (no BGR intermediary needed)
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)

    # Scale the V (brightness) channel
    v_scale = 1.0 - 0.95 * severity
    hsv[:, :, 2] *= v_scale
    hsv[:, :, 2] = np.clip(hsv[:, :, 2], 0, 255)

    hsv = hsv.astype(np.uint8)
    rgb_out = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return rgb_out


# ---------------------------------------------------------------------------
# 4. Scale Reduction
# ---------------------------------------------------------------------------

def scale(image: np.ndarray, severity: float) -> np.ndarray:
    """Shrink the image and zero-pad borders to maintain dimensions.

    At severity 0 the image is full-size; at severity 1 it is shrunk
    to 10 % of the original dimensions.

    Args:
        image: RGB uint8 array of shape (H, W, 3).
        severity: Float in [0, 1].

    Returns:
        Scaled-and-padded image with the same shape and dtype.
    """
    h, w = image.shape[:2]

    # Scale factor decreases from 1.0 → 0.1
    factor = 1.0 - severity * 0.9
    new_h = max(1, int(h * factor))
    new_w = max(1, int(w * factor))

    shrunken = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Place on a black canvas at the centre
    canvas = np.zeros_like(image)
    y_off = (h - new_h) // 2
    x_off = (w - new_w) // 2
    canvas[y_off:y_off + new_h, x_off:x_off + new_w] = shrunken

    return canvas


# ---------------------------------------------------------------------------
# 5. Viewpoint Transformation
# ---------------------------------------------------------------------------

def viewpoint(image: np.ndarray, severity: float) -> np.ndarray:
    """Simulate camera rotation via affine warp.

    Rotation angle increases linearly from 0° to 45° around the image
    centre.  Borders are replicated to reduce black artefacts.

    Args:
        image: RGB uint8 array of shape (H, W, 3).
        severity: Float in [0, 1].

    Returns:
        Rotated image with the same shape and dtype.
    """
    h, w = image.shape[:2]
    angle = severity * 45.0  # 0° → 45°
    centre = (w / 2.0, h / 2.0)

    rotation_matrix = cv2.getRotationMatrix2D(centre, angle, scale=1.0)
    rotated = cv2.warpAffine(
        image,
        rotation_matrix,
        (w, h),
        borderMode=cv2.BORDER_REPLICATE,
    )
    return rotated
