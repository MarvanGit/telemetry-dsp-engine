from __future__ import annotations

import numpy as np

from src.core.fatigue_analysis import (
    FatigueAnalysisConfig,
    analyze_fatigue,
    compute_frequency_metrics,
    compute_welch,
)


def test_welch_finds_expected_sine_peak():
    sampling_rate = 1000.0
    time = np.arange(0.0, 4.0, 1.0 / sampling_rate)
    data = np.sin(2.0 * np.pi * 50.0 * time)

    frequencies, power = compute_welch(
        data,
        sampling_rate,
        FatigueAnalysisConfig(
            window_seconds=1.0,
            overlap_fraction=0.5,
            nfft=2048,
            min_frequency=20.0,
            max_frequency=120.0,
        ),
    )

    peak_frequency = frequencies[np.argmax(power[0])]

    assert abs(peak_frequency - 50.0) < 1.0


def test_analyze_fatigue_preserves_multichannel_result_shapes():
    sampling_rate = 1000.0
    time = np.arange(0.0, 3.0, 1.0 / sampling_rate)
    data = np.vstack(
        [
            np.sin(2.0 * np.pi * 45.0 * time),
            np.sin(2.0 * np.pi * 80.0 * time),
        ]
    )

    result = analyze_fatigue(
        data,
        sampling_rate,
        FatigueAnalysisConfig(
            window_seconds=0.5,
            overlap_fraction=0.5,
            nfft=1024,
            min_frequency=20.0,
            max_frequency=150.0,
        ),
    )

    assert result.welch_power.shape == (2, result.frequencies.size)
    assert result.spectrogram_power.shape == (
        2,
        result.frequencies.size,
        result.times.size,
    )
    assert result.median_frequency.shape == (2,)
    assert result.spectrogram_median_frequency.shape == (2, result.times.size)


def test_spectrogram_metrics_track_frequency_shift_over_time():
    sampling_rate = 1000.0
    first_time = np.arange(0.0, 2.0, 1.0 / sampling_rate)
    second_time = np.arange(0.0, 2.0, 1.0 / sampling_rate)
    data = np.concatenate(
        [
            np.sin(2.0 * np.pi * 40.0 * first_time),
            np.sin(2.0 * np.pi * 95.0 * second_time),
        ]
    )

    result = analyze_fatigue(
        data,
        sampling_rate,
        FatigueAnalysisConfig(
            window_seconds=0.5,
            overlap_fraction=0.0,
            nfft=2048,
            min_frequency=20.0,
            max_frequency=140.0,
        ),
    )

    trend = result.spectrogram_median_frequency[0]
    first_half = trend[result.times < 2.0]
    second_half = trend[result.times >= 2.0]

    assert np.nanmean(first_half) < 55.0
    assert np.nanmean(second_half) > 80.0


def test_frequency_metrics_return_nan_for_zero_power():
    frequencies = np.array([20.0, 40.0, 60.0])
    power = np.zeros((2, 3))

    median_frequency, mean_frequency, band_power = compute_frequency_metrics(
        frequencies,
        power,
    )

    assert np.all(np.isnan(median_frequency))
    assert np.all(np.isnan(mean_frequency))
    np.testing.assert_allclose(band_power, np.zeros(2))
