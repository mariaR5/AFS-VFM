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
    """Apply random black rectangular occlusions.
    
    At severity 0, no occlusion. At severity 1, up to 40% of the image 
    is heavily covered with black boxes. Uses a fixed random seed 
    based on severity to remain deterministic.
    """
    if severity <= 0.0:
        return image.copy()
        
    out = image.copy()
    h, w = out.shape[:2]
    
    # Deterministic seed based on severity to ensure repeatable sequences
    rs = np.random.RandomState(int(severity * 1000))
    
    # Scale max number of boxes and max box size with severity
    max_boxes = int(severity * 50)
    max_box_h = int(severity * (h * 0.3))
    max_box_w = int(severity * (w * 0.3))
    
    num_boxes = rs.randint(1, max(2, max_boxes + 1))
    
    for _ in range(num_boxes):
        box_h = rs.randint(10, max(11, max_box_h))
        box_w = rs.randint(10, max(11, max_box_w))
        
        y = rs.randint(0, max(1, h - box_h))
        x = rs.randint(0, max(1, w - box_w))
        
        out[y:y+box_h, x:x+box_w] = 0
        
    return out


# ---------------------------------------------------------------------------
# 3. Lighting (Low-Light)
# ---------------------------------------------------------------------------

def lighting(image: np.ndarray, severity: float) -> np.ndarray:
    """Progressively plunge the image into darkness.
    
    At severity 0, normal brightness. At severity 1, extremely dark.
    """
    if severity <= 0.0:
        return image.copy()
        
    # Decrease brightness (beta: 0 to -150)
    # Decrease contrast slightly (alpha: 1.0 to 0.5)
    alpha = max(0.5, 1.0 - (severity * 0.5))
    beta = int(-150 * severity)
    
    out = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return out


# ---------------------------------------------------------------------------
# 4. Scale (Pixelation / Compression)
# ---------------------------------------------------------------------------

def scale(image: np.ndarray, severity: float) -> np.ndarray:
    """Crush spatial resolution and repeatedly upsample it back.
    
    At severity 0, original resolution. At severity 1, heavily pixelated.
    """
    if severity <= 0.0:
        return image.copy()
        
    h, w = image.shape[:2]
    
    # Scale down factor: 1.0 (no scale) to ~0.02 (massive scale down)
    scale_factor = max(0.02, 1.0 - (severity * 0.98))
    
    small_w = max(1, int(w * scale_factor))
    small_h = max(1, int(h * scale_factor))
    
    # Downsample
    temp = cv2.resize(image, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
    # Upsample back to original size (using nearest neighbor for blocky artifacts)
    out = cv2.resize(temp, (w, h), interpolation=cv2.INTER_NEAREST)
    return out


# ---------------------------------------------------------------------------
# 5. Viewpoint (Perspective Shear)
# ---------------------------------------------------------------------------

def viewpoint(image: np.ndarray, severity: float) -> np.ndarray:
    """Apply an extreme perspective warp (shearing the image back).
    """
    if severity <= 0.0:
        return image.copy()
        
    h, w = image.shape[:2]
    
    # Source points: the 4 corners
    pts1 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
    
    # Destination points: pinch the top corners inwards dynamically
    pinch_x = int(severity * (w * 0.4))
    pinch_y = int(severity * (h * 0.2))
    
    pts2 = np.float32([
        [pinch_x, pinch_y],          # Top-left gets pushed right and down
        [w - pinch_x, pinch_y],      # Top-right gets pushed left and down
        [0, h],                      # Bottom-left stays
        [w, h]                       # Bottom-right stays
    ])
    
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    out = cv2.warpPerspective(image, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    
    return out

