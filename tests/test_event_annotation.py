from __future__ import annotations

import numpy as np

from src.core.event_annotation import (
    EventAnnotationConfig,
    detect_event_annotations,
)


def test_detect_event_annotations_finds_multichannel_contractions():
    sampling_rate = 1000.0
    envelope = np.full((2, 2000), 0.05)
    envelope[0, 200:420] = np.hanning(220) + 0.2
    envelope[0, 900:1120] = np.hanning(220) + 0.25
    envelope[1, 500:760] = 0.8 * np.hanning(260) + 0.2

    events = detect_event_annotations(
        envelope,
        sampling_rate,
        EventAnnotationConfig(
            threshold_multiplier=2.0,
            min_duration_seconds=0.1,
            merge_gap_seconds=0.05,
            min_peak_distance_seconds=0.05,
        ),
    )

    assert [event.channel for event in events] == [0, 0, 1]
    assert events[0].start_sample == 200
    assert events[0].end_sample == 420
    assert 300 <= events[0].peak_sample <= 320
    assert events[1].start_sample == 900
    assert events[1].end_sample == 1120
    assert events[2].start_sample == 500
    assert events[2].end_sample == 760
    np.testing.assert_allclose(events[0].start_time, 0.2)
    np.testing.assert_allclose(events[0].end_time, 0.42)


def test_detect_event_annotations_merges_short_gaps():
    sampling_rate = 1000.0
    envelope = np.zeros((1, 1000))
    envelope[0, 100:220] = 1.0
    envelope[0, 260:380] = 1.1

    events = detect_event_annotations(
        envelope,
        sampling_rate,
        EventAnnotationConfig(
            threshold_multiplier=1.0,
            min_duration_seconds=0.05,
            merge_gap_seconds=0.05,
            min_peak_distance_seconds=0.05,
        ),
    )

    assert len(events) == 1
    assert events[0].start_sample == 100
    assert events[0].end_sample == 380


def test_detect_event_annotations_rejects_short_regions_and_noise():
    sampling_rate = 1000.0
    rng = np.random.default_rng(42)
    envelope = rng.normal(0.05, 0.005, size=(1, 1000))
    envelope[0, 100:120] = 1.0

    events = detect_event_annotations(
        envelope,
        sampling_rate,
        EventAnnotationConfig(
            threshold_multiplier=4.0,
            min_duration_seconds=0.05,
            merge_gap_seconds=0.02,
            min_peak_distance_seconds=0.05,
        ),
    )

    assert events == []
