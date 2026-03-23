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


