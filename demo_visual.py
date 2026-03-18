"""
Visual Demo — AFS-VFM Degradation Engine
==========================================

Downloads a sample image and generates grid visualisations for every
degradation type, saving them into  data/demo_output/.

Usage:
    python demo_visual.py
"""

import os
import sys
import urllib.request

import cv2
import numpy as np

# Ensure the 'src' directory is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from degradation import generate_degradation_sequence

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
OUTPUT_DIR = os.path.join(DATA_DIR, "demo_output")
SAMPLE_IMAGE_PATH = os.path.join(DATA_DIR, "sample_dog.jpg")

# Public-domain sample image (golden retriever — classic ImageNet class)
SAMPLE_URL = (
    "https://upload.wikimedia.org/wikipedia/commons/thumb/"
    "b/bd/Golden_Retriever_Dukedestination.jpg/"
    "640px-Golden_Retriever_Dukedestination.jpg"
)

DEGRADATION_TYPES = ["blur", "occlusion", "lighting", "scale", "viewpoint"]

# How many frames to show in the grid (evenly spaced from the 20)
GRID_COLS = 5  # frames per row
GRID_ROWS = 2  # 2 rows x 5 cols = 10 selected frames out of 20
FRAME_INDICES = [0, 2, 4, 6, 8, 10, 12, 14, 17, 19]  # 10 evenly-ish spaced


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def download_sample_image() -> str:
    """Download a sample image if it doesn't already exist."""
    os.makedirs(DATA_DIR, exist_ok=True)

    if os.path.exists(SAMPLE_IMAGE_PATH):
        print(f"  Sample image already exists: {SAMPLE_IMAGE_PATH}")
        return SAMPLE_IMAGE_PATH

    print(f"  Downloading sample image ...")
    try:
        urllib.request.urlretrieve(SAMPLE_URL, SAMPLE_IMAGE_PATH)
        print(f"  Saved to: {SAMPLE_IMAGE_PATH}")
    except Exception as exc:
        print(f"  Download failed: {exc}")
        print("  Creating a synthetic test image instead ...")
        create_synthetic_image(SAMPLE_IMAGE_PATH)

    return SAMPLE_IMAGE_PATH


def create_synthetic_image(path: str) -> None:
    """Create a colourful 512x512 synthetic image as a fallback."""
    h, w = 512, 512
    img = np.zeros((h, w, 3), dtype=np.uint8)

    # Gradient background
    for y in range(h):
        for x in range(w):
            img[y, x] = [
                int(255 * x / w),         # R gradient
                int(255 * y / h),         # G gradient
                int(128 + 64 * np.sin(x * 0.05)),  # B wave
            ]

    # Draw some shapes to make it more interesting
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.circle(bgr, (256, 256), 100, (0, 255, 255), -1)
    cv2.rectangle(bgr, (50, 50), (200, 200), (255, 100, 50), -1)
    cv2.putText(bgr, "AFS-VFM", (150, 400), cv2.FONT_HERSHEY_SIMPLEX,
                2.0, (255, 255, 255), 3)
    cv2.imwrite(path, bgr)
    print(f"  Synthetic image saved to: {path}")


def add_label(image: np.ndarray, text: str) -> np.ndarray:
    """Add a small label to the top-left of an image (works on RGB)."""
    result = image.copy()
    # Convert to BGR for OpenCV drawing, then back
    bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
    cv2.putText(bgr, text, (8, 25), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (0, 255, 0), 2, cv2.LINE_AA)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def build_grid(frames: np.ndarray, deg_type: str) -> np.ndarray:
    """Select key frames and arrange them in a labelled grid.

    Returns an RGB image.
    """
    selected = [frames[i] for i in FRAME_INDICES]
    labelled = []
    for idx, frame in zip(FRAME_INDICES, selected):
        severity_pct = int(idx / 19 * 100)
        labelled.append(add_label(frame, f"F{idx+1} ({severity_pct}%)"))

    # Build rows
    rows = []
    for r in range(GRID_ROWS):
        start = r * GRID_COLS
        row_frames = labelled[start:start + GRID_COLS]
        rows.append(np.hstack(row_frames))

    grid = np.vstack(rows)
    return grid


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("  AFS-VFM Degradation Engine — Visual Demo")
    print("=" * 60)

    image_path = download_sample_image()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print()
    for deg_type in DEGRADATION_TYPES:
        print(f"  Generating {deg_type} sequence ...", end=" ")
        frames = generate_degradation_sequence(image_path, deg_type)

        # Build grid
        grid = build_grid(frames, deg_type)

        # Save the grid (convert RGB -> BGR for cv2.imwrite)
        out_path = os.path.join(OUTPUT_DIR, f"demo_{deg_type}.jpg")
        cv2.imwrite(out_path, cv2.cvtColor(grid, cv2.COLOR_RGB2BGR),
                    [cv2.IMWRITE_JPEG_QUALITY, 95])

        # Also save first and last frame individually
        first_path = os.path.join(OUTPUT_DIR, f"{deg_type}_frame01_clean.jpg")
        last_path = os.path.join(OUTPUT_DIR, f"{deg_type}_frame20_degraded.jpg")
        cv2.imwrite(first_path, cv2.cvtColor(frames[0], cv2.COLOR_RGB2BGR))
        cv2.imwrite(last_path, cv2.cvtColor(frames[-1], cv2.COLOR_RGB2BGR))

        print(f"saved -> {out_path}")

    print()
    print(f"  All outputs saved to: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
