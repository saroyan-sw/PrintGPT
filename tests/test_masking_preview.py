import numpy as np

from printgpt_masking import compose_preview


def test_preview_preserves_rgb_channels_and_source_array() -> None:
    page = np.full((4, 5, 3), (12, 34, 200), dtype=np.uint8)
    source = page.copy()
    mask = np.zeros((4, 5), dtype=np.uint8)
    mask[1, 2] = 255
    preview = compose_preview(page, mask)
    assert tuple(preview[0, 0]) == (12, 34, 200)
    assert tuple(preview[1, 2]) == (0, 0, 0)
    assert np.array_equal(page, source)


def test_preview_is_deterministic() -> None:
    page = np.arange(5 * 6 * 3, dtype=np.uint8).reshape(5, 6, 3)
    mask = np.zeros((5, 6), dtype=np.uint8)
    mask[2, 2:5] = 255
    assert np.array_equal(compose_preview(page, mask), compose_preview(page, mask))
