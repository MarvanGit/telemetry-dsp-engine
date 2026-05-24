from __future__ import annotations

import numpy as np

from src.core.dsp_live import DSPLive


def make_live_dsp(num_channels: int) -> DSPLive:
    return DSPLive(
        num_channels=num_channels,
        sampling_rate=1000.0,
        lowcut=20.0,
        highcut=200.0,
        filter_order=4,
        rms_window_ms=50.0,
    )


def test_live_dsp_chunking_matches_single_pass_processing():
    rng = np.random.default_rng(20260524)
    data = rng.normal(size=(4, 640))

    single_pass = make_live_dsp(num_channels=data.shape[0]).process_chunk(data)

    chunked_dsp = make_live_dsp(num_channels=data.shape[0])
    chunked = np.concatenate(
        [
            chunked_dsp.process_chunk(data[:, start : start + 73])
            for start in range(0, data.shape[1], 73)
        ],
        axis=1,
    )

    np.testing.assert_allclose(chunked, single_pass, rtol=1e-12, atol=1e-12)
    assert chunked.shape == data.shape
    assert np.all(np.isfinite(chunked))
    assert np.all(chunked >= 0.0)


def test_live_dsp_reset_restores_initial_filter_state():
    rng = np.random.default_rng(24)
    data = rng.normal(size=(2, 256))
    dsp = make_live_dsp(num_channels=data.shape[0])

    first = dsp.process_chunk(data)
    dsp.process_chunk(data)
    dsp.reset_states()
    after_reset = dsp.process_chunk(data)

    np.testing.assert_allclose(after_reset, first, rtol=1e-12, atol=1e-12)
