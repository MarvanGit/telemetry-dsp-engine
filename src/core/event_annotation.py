from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal

from src.core.fatigue_analysis import as_channel_matrix


@dataclass(frozen=True)
class EventAnnotationConfig:
    threshold_multiplier: float = 4.0
    min_duration_seconds: float = 0.15
    merge_gap_seconds: float = 0.1
    min_peak_distance_seconds: float = 0.1


@dataclass(frozen=True)
class EventAnnotation:
    channel: int
    start_sample: int
    end_sample: int
    peak_sample: int
    start_time: float
    end_time: float
    peak_time: float
    peak_value: float
    threshold: float


def detect_event_annotations(
    rms_envelope: np.ndarray,
    sampling_rate: float,
    config: EventAnnotationConfig | None = None,
) -> list[EventAnnotation]:
    annotation_config = config or EventAnnotationConfig()
    matrix = as_channel_matrix(rms_envelope)
    validate_config(sampling_rate, annotation_config)

    min_duration_samples = seconds_to_samples(
        annotation_config.min_duration_seconds,
        sampling_rate,
    )
    merge_gap_samples = seconds_to_samples(
        annotation_config.merge_gap_seconds,
        sampling_rate,
    )
    peak_distance_samples = seconds_to_samples(
        annotation_config.min_peak_distance_seconds,
        sampling_rate,
    )

    annotations: list[EventAnnotation] = []
    for channel_index, channel_envelope in enumerate(matrix):
        envelope = np.nan_to_num(
            np.asarray(channel_envelope, dtype=np.float64),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        envelope = np.maximum(envelope, 0.0)
        threshold = compute_adaptive_threshold(
            envelope,
            annotation_config.threshold_multiplier,
        )
        regions = find_active_regions(envelope > threshold)
        regions = merge_close_regions(regions, merge_gap_samples)

        peak_indices, _ = signal.find_peaks(
            envelope,
            height=threshold,
            distance=peak_distance_samples,
        )

        for start_sample, end_sample in regions:
            if end_sample - start_sample < min_duration_samples:
                continue

            region_peaks = peak_indices[
                (peak_indices >= start_sample) & (peak_indices < end_sample)
            ]
            if region_peaks.size > 0:
                peak_sample = int(region_peaks[np.argmax(envelope[region_peaks])])
            else:
                peak_sample = int(start_sample + np.argmax(envelope[start_sample:end_sample]))

            annotations.append(
                EventAnnotation(
                    channel=channel_index,
                    start_sample=start_sample,
                    end_sample=end_sample,
                    peak_sample=peak_sample,
                    start_time=start_sample / sampling_rate,
                    end_time=end_sample / sampling_rate,
                    peak_time=peak_sample / sampling_rate,
                    peak_value=float(envelope[peak_sample]),
                    threshold=threshold,
                )
            )

    return annotations


def validate_config(sampling_rate: float, config: EventAnnotationConfig) -> None:
    if sampling_rate <= 0:
        raise ValueError("sampling_rate must be positive.")
    if config.threshold_multiplier < 0:
        raise ValueError("threshold_multiplier cannot be negative.")
    if config.min_duration_seconds <= 0:
        raise ValueError("min_duration_seconds must be positive.")
    if config.merge_gap_seconds < 0:
        raise ValueError("merge_gap_seconds cannot be negative.")
    if config.min_peak_distance_seconds <= 0:
        raise ValueError("min_peak_distance_seconds must be positive.")


def seconds_to_samples(seconds: float, sampling_rate: float) -> int:
    return max(1, int(round(seconds * sampling_rate)))


def compute_adaptive_threshold(envelope: np.ndarray, multiplier: float) -> float:
    baseline = float(np.median(envelope))
    mad = float(np.median(np.abs(envelope - baseline)))
    robust_noise = 1.4826 * mad

    return baseline + multiplier * robust_noise


def find_active_regions(active_mask: np.ndarray) -> list[tuple[int, int]]:
    if active_mask.size == 0:
        return []

    padded_mask = np.concatenate(([False], active_mask, [False]))
    transitions = np.diff(padded_mask.astype(np.int8))
    starts = np.flatnonzero(transitions == 1)
    ends = np.flatnonzero(transitions == -1)
    return [(int(start), int(end)) for start, end in zip(starts, ends)]


def merge_close_regions(
    regions: list[tuple[int, int]],
    max_gap_samples: int,
) -> list[tuple[int, int]]:
    if not regions:
        return []

    merged_regions = [regions[0]]
    for start_sample, end_sample in regions[1:]:
        previous_start, previous_end = merged_regions[-1]
        if start_sample - previous_end <= max_gap_samples:
            merged_regions[-1] = (previous_start, end_sample)
        else:
            merged_regions.append((start_sample, end_sample))

    return merged_regions
