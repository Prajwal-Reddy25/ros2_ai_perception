import cv2
import numpy as np
import pytest
from ros2_ai_perception.media import MediaSource, UnsupportedMediaError


def test_still_image_eof_and_loop(tmp_path):
    path = tmp_path / "frame.png"
    assert cv2.imwrite(str(path), np.full((10, 12, 3), 127, dtype=np.uint8))
    with MediaSource(path, loop=False) as source:
        assert source.read().shape == (10, 12, 3)
        assert source.read() is None
    with MediaSource(path, loop=True) as source:
        assert source.read() is not None
        assert source.read() is not None


def test_image_directory_is_sorted_and_filters_files(tmp_path):
    (tmp_path / "notes.txt").write_text("ignored", encoding="utf-8")
    assert cv2.imwrite(str(tmp_path / "b.png"), np.full((2, 2, 3), 2, dtype=np.uint8))
    assert cv2.imwrite(str(tmp_path / "a.png"), np.full((2, 2, 3), 1, dtype=np.uint8))
    with MediaSource(tmp_path) as source:
        assert int(source.read()[0, 0, 0]) == 1
        assert int(source.read()[0, 0, 0]) == 2
        assert source.read() is None


def test_empty_directory_and_invalid_file_are_rejected(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(UnsupportedMediaError, match="no supported images"):
        MediaSource(empty)
    invalid = tmp_path / "broken.mp4"
    invalid.write_text("not video", encoding="utf-8")
    with pytest.raises(UnsupportedMediaError, match="could not open"):
        MediaSource(invalid)
