from __future__ import annotations

from threading import Lock

import numpy as np
from PySide6.QtCore import QThread, Signal

from src.core.fatigue_analysis import (
    FatigueAnalysisConfig,
    analyze_fatigue,
)


class FatigueReviewWorker(QThread):
    result_ready = Signal(object)
    error_signal = Signal(str)

    def __init__(
        self,
        data: np.ndarray,
        sampling_rate: float,
        config: FatigueAnalysisConfig,
        parent=None,
    ):
        super().__init__(parent)
        self.data = np.array(data, dtype=np.float64, copy=True)
        self.sampling_rate = sampling_rate
        self.config = config
        self._is_cancelled = False

    def cancel(self):
        self._is_cancelled = True

    def run(self):
        try:
            result = analyze_fatigue(self.data, self.sampling_rate, self.config)
            if not self._is_cancelled:
                self.result_ready.emit(result)
        except Exception as exc:
            if not self._is_cancelled:
                self.error_signal.emit(str(exc))


class FatigueLiveWorker(QThread):
    result_ready = Signal(object)
    error_signal = Signal(str)

    def __init__(
        self,
        sampling_rate: float,
        config: FatigueAnalysisConfig,
        buffer_seconds: float = 10.0,
        analysis_interval_ms: int = 1000,
        parent=None,
    ):
        super().__init__(parent)
        self.sampling_rate = sampling_rate
        self.config = config
        self.buffer_seconds = buffer_seconds
        self.analysis_interval_ms = analysis_interval_ms
        self.max_samples = max(2, int(round(buffer_seconds * sampling_rate)))
        self.minimum_samples = max(2, int(round(config.window_seconds * sampling_rate)))
        self._lock = Lock()
        self._pending_chunks: list[np.ndarray] = []
        self._buffer: np.ndarray | None = None
        self._is_running = False

    def add_chunk(self, data_chunk: np.ndarray):
        chunk = np.asarray(data_chunk, dtype=np.float64)
        if chunk.ndim == 1:
            chunk = chunk.reshape(1, -1)
        if chunk.ndim != 2 or chunk.shape[1] == 0:
            return

        with self._lock:
            self._pending_chunks.append(np.array(chunk, copy=True))
            if len(self._pending_chunks) > 8:
                self._pending_chunks = self._pending_chunks[-8:]

    def stop(self):
        self._is_running = False

    def run(self):
        self._is_running = True
        elapsed_ms = self.analysis_interval_ms

        while self._is_running:
            self._drain_pending_chunks()
            snapshot = self._snapshot_buffer()
            if (
                snapshot is not None
                and snapshot.shape[1] >= self.minimum_samples
                and elapsed_ms >= self.analysis_interval_ms
            ):
                self._analyze_snapshot(snapshot)
                elapsed_ms = 0

            self.msleep(50)
            elapsed_ms += 50

    def _drain_pending_chunks(self):
        with self._lock:
            pending_chunks = self._pending_chunks
            self._pending_chunks = []

        for chunk in pending_chunks:
            self._append_to_buffer(chunk)

    def _append_to_buffer(self, chunk: np.ndarray):
        if self._buffer is None or self._buffer.shape[0] != chunk.shape[0]:
            self._buffer = np.empty((chunk.shape[0], 0), dtype=np.float64)

        self._buffer = np.concatenate((self._buffer, chunk), axis=1)
        if self._buffer.shape[1] > self.max_samples:
            self._buffer = self._buffer[:, -self.max_samples :]

    def _snapshot_buffer(self) -> np.ndarray | None:
        if self._buffer is None:
            return None
        return np.array(self._buffer, copy=True)

    def _analyze_snapshot(self, snapshot: np.ndarray):
        try:
            result = analyze_fatigue(snapshot, self.sampling_rate, self.config)
            if self._is_running:
                self.result_ready.emit(result)
        except Exception as exc:
            self.error_signal.emit(str(exc))
