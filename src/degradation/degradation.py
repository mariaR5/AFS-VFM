"""
Degradation pipeline for the AFS-VFM research project.

Usage
-----
>>> from degradation import generate_degradation_sequence
>>> frames = generate_degradation_sequence("image.jpg", "blur")
>>> print(frames.shape)   # (20, H, W, 3)

The module is designed for reuse: another developer can import
``generate_degradation_sequence`` and feed its output directly into a
model-inference pipeline.
"""

import numpy as np

from . import transformations
from .utils import load_image


# ── Supported degradation type aliases ──────────────────────────────────────
_TYPE_ALIASES: dict[str, str] = {
    "blur": "motion_blur",
    "motion_blur": "motion_blur",
    "occlusion": "occlusion",
    "occlude": "occlusion",
    "lighting": "lighting",
    "light": "lighting",
    "dark": "lighting",
    "scale": "scale",
    "zoom": "scale",
    "viewpoint": "viewpoint",
    "rotate": "viewpoint",
}


class DegradationPipeline:
    """Generates a sequence of progressively degraded image frames.

    Parameters
    ----------
    num_frames : int, optional
        Number of degradation steps in each sequence (default 20).
    """

    # Mapping from canonical degradation name → implementation function
    _TRANSFORMS: dict = {
        "motion_blur": transformations.motion_blur,
        "occlusion": transformations.occlusion,
        "lighting": transformations.lighting,
        "scale": transformations.scale,
        "viewpoint": transformations.viewpoint,
    }

    def __init__(self, num_frames: int = 20) -> None:
        self.num_frames = num_frames

    # ── Thin wrapper methods (for discoverability / IDE hints) ──────────

    @staticmethod
    def motion_blur(image: np.ndarray, severity: float) -> np.ndarray:
        """Apply horizontal motion blur. See ``transformations.motion_blur``."""
        return transformations.motion_blur(image, severity)

    @staticmethod
    def occlusion(image: np.ndarray, severity: float) -> np.ndarray:
        """Apply centre occlusion. See ``transformations.occlusion``."""
        return transformations.occlusion(image, severity)

    @staticmethod
    def lighting(image: np.ndarray, severity: float) -> np.ndarray:
        """Reduce lighting. See ``transformations.lighting``."""
        return transformations.lighting(image, severity)

    @staticmethod
    def scale(image: np.ndarray, severity: float) -> np.ndarray:
        """Scale-down with padding. See ``transformations.scale``."""
        return transformations.scale(image, severity)

    @staticmethod
    def viewpoint(image: np.ndarray, severity: float) -> np.ndarray:
        """Affine rotation. See ``transformations.viewpoint``."""
        return transformations.viewpoint(image, severity)

    # ── Core orchestration ──────────────────────────────────────────────

    def generate_sequence(
        self,
        image: np.ndarray,
        degradation_type: str,
    ) -> np.ndarray:
        """Produce a sequence of *num_frames* degraded versions of *image*.

        Severity increases linearly from frame 0 (mildest) to frame
        ``num_frames - 1`` (most severe).

        Parameters
        ----------
        image : np.ndarray
            Source image in RGB uint8 format, shape (H, W, 3).
        degradation_type : str
            Name of the degradation to apply.  Accepted values include
            ``"blur"``, ``"motion_blur"``, ``"occlusion"``, ``"lighting"``,
            ``"scale"``, ``"viewpoint"`` and common aliases.

        Returns
        -------
        np.ndarray
            Array of shape ``(num_frames, H, W, 3)`` with dtype ``uint8``.

        Raises
        ------
        ValueError
            If *degradation_type* is not recognised.
        """
        canonical = _TYPE_ALIASES.get(degradation_type.lower())
        if canonical is None:
            supported = sorted(set(_TYPE_ALIASES.keys()))
            raise ValueError(
                f"Unknown degradation type '{degradation_type}'. "
                f"Supported types: {supported}"
            )

        transform_fn = self._TRANSFORMS[canonical]
        h, w = image.shape[:2]
        frames = np.empty((self.num_frames, h, w, 3), dtype=np.uint8)

        for i in range(self.num_frames):
            severity = i / (self.num_frames - 1)  # 0.0 → 1.0 linearly
            frames[i] = transform_fn(image, severity)

        return frames


# ── Module-level convenience function ───────────────────────────────────────

def generate_degradation_sequence(
    image_path: str,
    degradation_type: str,
    num_frames: int = 20,
) -> np.ndarray:
    """Load an image and return a degradation sequence.

    This is the main public API of the module.

    Parameters
    ----------
    image_path : str
        Path to the source image.
    degradation_type : str
        Kind of degradation (e.g. ``"blur"``, ``"occlusion"``).
    num_frames : int, optional
        Number of degradation steps (default 20).

    Returns
    -------
    np.ndarray
        Array of shape ``(num_frames, H, W, 3)`` with dtype ``uint8``.
    """
    image = load_image(image_path)
    pipeline = DegradationPipeline(num_frames=num_frames)
    return pipeline.generate_sequence(image, degradation_type)
