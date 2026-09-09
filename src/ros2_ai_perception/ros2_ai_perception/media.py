"""Validated still-image, image-sequence, and video reading."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .config import validate_input_path

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


class UnsupportedMediaError(ValueError):
    """Raised when OpenCV cannot decode the requested media."""


class MediaSource:
    """Read frames from one image, an ordered directory, or an OpenCV video."""

    def __init__(self, path: str | Path, loop: bool = False) -> None:
        self.path = validate_input_path(path)
        self.loop = bool(loop)
        self._capture: cv2.VideoCapture | None = None
        self._images: list[Path] = []
        self._index = 0
        if self.path.is_dir():
            self._images = sorted(
                item for item in self.path.iterdir() if item.suffix.lower() in IMAGE_SUFFIXES
            )
            if not self._images:
                raise UnsupportedMediaError(f"directory contains no supported images: {self.path}")
        elif self.path.suffix.lower() in IMAGE_SUFFIXES:
            self._images = [self.path]
        else:
            capture = cv2.VideoCapture(str(self.path))
            if not capture.isOpened():
                capture.release()
                raise UnsupportedMediaError(f"OpenCV could not open media: {self.path}")
            self._capture = capture

    @property
    def is_still(self) -> bool:
        return self._capture is None and len(self._images) == 1

    def read(self) -> np.ndarray | None:
        """Return the next frame, or ``None`` at non-looping EOF."""
        if self._capture is not None:
            ok, frame = self._capture.read()
            if ok and frame is not None and frame.size:
                return frame
            if not self.loop:
                return None
            self._capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self._capture.read()
            return frame if ok and frame is not None and frame.size else None

        if self._index >= len(self._images):
            if not self.loop:
                return None
            self._index = 0
        frame = cv2.imread(str(self._images[self._index]), cv2.IMREAD_COLOR)
        source = self._images[self._index]
        self._index += 1
        if frame is None or not frame.size:
            raise UnsupportedMediaError(f"OpenCV could not decode image: {source}")
        return frame

    def close(self) -> None:
        """Release a video decoder, if one is active."""
        if self._capture is not None:
            self._capture.release()

    def __enter__(self) -> MediaSource:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
