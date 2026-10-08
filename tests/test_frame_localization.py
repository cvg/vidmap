"""Tests for frame selection helpers of the all-frame localization."""

from types import SimpleNamespace

from vidmap.localization.frames import bracketing_keyframes, frame_timestamp, subsample_frames


def _names(times):
    return [f"{t:020.9f}.jpg" for t in times]


def test_frame_timestamp():
    assert frame_timestamp("video/0000000013.067000000.jpg") == 13.067


def test_subsample_frames():
    names = _names([i / 30 for i in range(90)])  # 3 s at 30 fps
    assert subsample_frames(list(reversed(names)), 0) == names  # all frames, sorted by time
    picked = subsample_frames(names, 10)
    assert len(picked) == 30
    assert picked[:2] == [names[0], names[3]]


def test_bracketing_keyframes():
    keyframes = [SimpleNamespace(name=n) for n in _names([0.0, 1.0, 2.0])]
    key_times = [0.0, 1.0, 2.0]
    assert bracketing_keyframes(1.2, keyframes, key_times) == [1, 2]
    assert bracketing_keyframes(1.8, keyframes, key_times) == [2, 1]
    assert bracketing_keyframes(5.0, keyframes, key_times) == [2]
    # Leave-one-out skips the excluded keyframe itself.
    assert bracketing_keyframes(1.0, keyframes, key_times, exclude=keyframes[1].name) == [0, 2]
