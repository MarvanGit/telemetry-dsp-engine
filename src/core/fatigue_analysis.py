from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal


@dataclass(frozen=True)
class FatigueAnalysisConfig:
    window_seconds: float = 1.0
    overlap_fraction: float = 0.5
    nfft: int | None = None
    min_frequency: float = 20.0
    max_frequency: float | None = 450.0


@dataclass(frozen=True)
class FatigueAnalysisResult:
    frequencies: np.ndarray
    times: np.ndarray
    welch_power: np.ndarray
    spectrogram_power: np.ndarray
    median_frequency: np.ndarray
    mean_frequency: np.ndarray
    band_power: np.ndarray
    spectrogram_median_frequency: np.ndarray
    spectrogram_mean_frequency: np.ndarray
    spectrogram_band_power: np.ndarray


def analyze_fatigue(
    data: np.ndarray,
    sampling_rate: float,
    config: FatigueAnalysisConfig | None = None,
) -> FatigueAnalysisResult:
    analysis_config = config or FatigueAnalysisConfig()
    matrix = as_channel_matrix(data)

    frequencies, welch_power = compute_welch(matrix, sampling_rate, analysis_config)
    spectrogram_frequencies, times, spectrogram_power = compute_spectrogram(
        matrix,
        sampling_rate,
        analysis_config,
    )

    if not np.array_equal(frequencies, spectrogram_frequencies):
        raise ValueError("Welch and spectrogram frequency axes do not match.")

    median_frequency, mean_frequency, band_power = compute_frequency_metrics(
        frequencies,
        welch_power,
    )
    (
        spectrogram_median_frequency,
        spectrogram_mean_frequency,
        spectrogram_band_power,
    ) = compute_frequency_metrics(
        frequencies,
        spectrogram_power,
        frequency_axis=1,
    )

    return FatigueAnalysisResult(
        frequencies=frequencies,
        times=times,
        welch_power=welch_power,
        spectrogram_power=spectrogram_power,
        median_frequency=median_frequency,
        mean_frequency=mean_frequency,
        band_power=band_power,
        spectrogram_median_frequency=spectrogram_median_frequency,
        spectrogram_mean_frequency=spectrogram_mean_frequency,
        spectrogram_band_power=spectrogram_band_power,
    )


def compute_welch(
    data: np.ndarray,
    sampling_rate: float,
    config: FatigueAnalysisConfig | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    analysis_config = config or FatigueAnalysisConfig()
    matrix = as_channel_matrix(data)
    nperseg, noverlap, nfft = resolve_segment_settings(
        matrix.shape[1],
        sampling_rate,
        analysis_config,
    )

    frequencies, power = signal.welch(
        matrix,
        fs=sampling_rate,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=nfft,
        axis=-1,
        scaling="density",
    )
    frequency_mask = build_frequency_mask(frequencies, sampling_rate, analysis_config)
    return frequencies[frequency_mask], power[:, frequency_mask]


def compute_spectrogram(
    data: np.ndarray,
    sampling_rate: float,
    config: FatigueAnalysisConfig | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    analysis_config = config or FatigueAnalysisConfig()
    matrix = as_channel_matrix(data)
    nperseg, noverlap, nfft = resolve_segment_settings(
        matrix.shape[1],
        sampling_rate,
        analysis_config,
    )

    frequencies, times, power = signal.spectrogram(
        matrix,
        fs=sampling_rate,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=nfft,
        axis=-1,
        scaling="density",
        mode="psd",
    )
    frequency_mask = build_frequency_mask(frequencies, sampling_rate, analysis_config)
    return frequencies[frequency_mask], times, power[:, frequency_mask, :]


def compute_frequency_metrics(
    frequencies: np.ndarray,
    power: np.ndarray,
    frequency_axis: int = -1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    frequency_axis = int(frequency_axis)
    frequencies = np.asarray(frequencies, dtype=np.float64)
    power = np.asarray(power, dtype=np.float64)
    if frequencies.ndim != 1 or frequencies.size == 0:
        raise ValueError("frequencies must be a non-empty 1D array.")

    power_by_frequency = np.moveaxis(power, frequency_axis, -1)
    if power_by_frequency.shape[-1] != frequencies.size:
        raise ValueError("power frequency axis length must match frequencies.")

    total_power = np.sum(power_by_frequency, axis=-1)
    weighted_power = np.sum(power_by_frequency * frequencies, axis=-1)
    mean_frequency = np.divide(
        weighted_power,
        total_power,
        out=np.full(total_power.shape, np.nan, dtype=np.float64),
        where=total_power > 0.0,
    )

    cumulative_power = np.cumsum(power_by_frequency, axis=-1)
    median_indexes = np.argmax(cumulative_power >= (total_power[..., np.newaxis] * 0.5), axis=-1)
    median_frequency = frequencies[median_indexes]
    median_frequency = np.where(total_power > 0.0, median_frequency, np.nan)

    if frequencies.size > 1:
        band_power = np.trapezoid(power_by_frequency, frequencies, axis=-1)
    else:
        band_power = np.squeeze(power_by_frequency, axis=-1)
    return median_frequency, mean_frequency, band_power


def as_channel_matrix(data: np.ndarray) -> np.ndarray:
    matrix = np.asarray(data, dtype=np.float64)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    if matrix.ndim != 2:
        raise ValueError("data must be a 1D or 2D array shaped as channels x samples.")
    if matrix.shape[1] < 2:
        raise ValueError("data must contain at least two samples.")
    return matrix


def resolve_segment_settings(
    sample_count: int,
    sampling_rate: float,
    config: FatigueAnalysisConfig,
) -> tuple[int, int, int]:
    if sampling_rate <= 0:
        raise ValueError("sampling_rate must be greater than zero.")
    if config.window_seconds <= 0:
        raise ValueError("window_seconds must be greater than zero.")
    if not 0.0 <= config.overlap_fraction < 1.0:
        raise ValueError("overlap_fraction must be greater than or equal to 0 and less than 1.")

    requested_nperseg = max(2, int(round(config.window_seconds * sampling_rate)))
    nperseg = min(requested_nperseg, int(sample_count))
    noverlap = min(int(round(nperseg * config.overlap_fraction)), nperseg - 1)

    if config.nfft is None:
        nfft = nperseg
    else:
        nfft = max(int(config.nfft), nperseg)
    return nperseg, noverlap, nfft


def build_frequency_mask(
    frequencies: np.ndarray,
    sampling_rate: float,
    config: FatigueAnalysisConfig,
) -> np.ndarray:
    nyquist = sampling_rate * 0.5
    max_frequency = nyquist if config.max_frequency is None else min(config.max_frequency, nyquist)
    if config.min_frequency < 0:
        raise ValueError("min_frequency must be greater than or equal to zero.")
    if max_frequency <= config.min_frequency:
        raise ValueError("max_frequency must be greater than min_frequency.")

    frequency_mask = (frequencies >= config.min_frequency) & (frequencies <= max_frequency)
    if not np.any(frequency_mask):
        raise ValueError("frequency range does not contain any spectral bins.")
    return frequency_mask
