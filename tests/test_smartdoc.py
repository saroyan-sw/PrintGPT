import numpy as np

from printgpt_alignment.smartdoc import AnnotatedFrame, _select_diverse


def test_diverse_selection_is_deterministic() -> None:
    frames = [
        AnnotatedFrame(
            "sample",
            index,
            (
                (100.0 + index, 100.0),
                (700.0, 90.0 + index),
                (720.0 - index, 900.0),
                (90.0, 880.0 - index),
            ),
        )
        for index in range(1, 21)
    ]
    first = _select_diverse(frames, 5)
    second = _select_diverse(frames, 5)
    assert [frame.frame_index for frame in first] == [frame.frame_index for frame in second]
    assert len({frame.frame_index for frame in first}) == 5
