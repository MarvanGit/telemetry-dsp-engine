from __future__ import annotations

import numpy as np

from src.core.dsp_review import DSPReview


def test_review_bandpass_preserves_in_band_signal_and_rejects_low_frequency():
    sampling_rate = 1000.0
    duration_seconds = 4.0
    time = np.arange(0.0, duration_seconds, 1.0 / sampling_rate)

    in_band_signal = np.sin(2.0 * np.pi * 50.0 * time)
    low_frequency_noise = 0.8 * np.sin(2.0 * np.pi * 5.0 * time)
    data = np.vstack(
        [
            in_band_signal + low_frequency_noise,
            0.5 * in_band_signal + low_frequency_noise,
        ]
    )

    filtered = DSPReview.bandpass_filter(
        data,
        lowcut=20.0,
        highcut=100.0,
        fs=sampling_rate,
        order=4,
    )

    frequencies = np.fft.rfftfreq(filtered.shape[1], d=1.0 / sampling_rate)
    spectrum = np.abs(np.fft.rfft(filtered[0]))
    low_bin = np.argmin(np.abs(frequencies - 5.0))
    pass_bin = np.argmin(np.abs(frequencies - 50.0))

    assert filtered.shape == data.shape
    assert spectrum[pass_bin] > 50.0 * spectrum[low_bin]


def test_review_rms_envelope_matches_constant_channel_amplitudes():
    data = np.vstack(
        [
            np.full(400, -3.0),
            np.full(400, 4.0),
        ]
    )

    envelope = DSPReview().rms_envelope(
        data,
        sampling_rate=1000.0,
        window_ms=40.0,
    )

    assert envelope.shape == data.shape
    np.testing.assert_allclose(envelope[0], 3.0, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(envelope[1], 4.0, rtol=1e-12, atol=1e-12)
