import os
import pickle

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QRectF, Qt
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from src.core.data_source import FileDataSource, LiveDataSource
from src.core.data_worker import DataWorker
from src.core.dsp_live import DSPLive
from src.core.dsp_review import DSPReview
from src.core.fatigue_analysis import FatigueAnalysisConfig
from src.ui.fatigue_worker import FatigueLiveWorker, FatigueReviewWorker


class MainWindow(QMainWindow):
    """
    Main application window for EMG visualization.

    The UI now exposes the application's two data modes:
    - Review: load a recorded dataset and render it statically.
    - Live: reserve the streaming path for the hardware-backed data source.
    """

    LIVE_MODE = "live"
    REVIEW_MODE = "review"
    ORIGINAL_SIGNAL = "original"
    FILTERED_SIGNAL = "filtered"
    RMS_SIGNAL = "rms"
    PROCESSED_SIGNAL = "processed"
    DEFAULT_REVIEW_SAMPLING_RATE = 2000
    LIVE_BUFFER_SAMPLES = 4000
    DSP_LOW_CUT = 20.0
    DSP_HIGH_CUT = 450.0
    DSP_FILTER_ORDER = 4
    DSP_RMS_WINDOW_MS = 100.0
    FATIGUE_WINDOW_SECONDS = 1.0
    FATIGUE_OVERLAP_PERCENT = 50
    FATIGUE_BUFFER_SECONDS = 10.0
    FATIGUE_ANALYSIS_INTERVAL_MS = 1000

    def __init__(self):
        super().__init__()

        self.data_file_path = os.path.join("data", "recording.pkl")
        self.current_mode = self.REVIEW_MODE
        self.data_source = None
        self.data_worker = None
        self.review_dsp = DSPReview()
        self.review_signal_cache = {}
        self.live_dsp = None
        self.num_channels = 0
        self.current_gain = 1
        self.current_signal_view = self.ORIGINAL_SIGNAL
        self.is_paused = False
        self.fatigue_review_worker = None
        self.fatigue_live_worker = None
        self.fatigue_result = None

        self.setWindowTitle("EMG Data Visualization")
        self.resize(1000, 800)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout()
        self.central_widget.setLayout(self.layout)

        metadata_layout = QHBoxLayout()
        self.mode_combo = QComboBox()
        self.mode_combo.addItem("Review", self.REVIEW_MODE)
        self.mode_combo.addItem("Live", self.LIVE_MODE)
        self.mode_combo.currentIndexChanged.connect(self.handle_mode_changed)
        self.current_file_label = QLabel(f"File: {self.data_file_path}")
        self.sample_rate_label = QLabel("Sample Rate: -")
        self.channel_count_label = QLabel("Channels: -")
        metadata_layout.addWidget(QLabel("Mode:"))
        metadata_layout.addWidget(self.mode_combo)
        metadata_layout.addWidget(self.current_file_label)
        metadata_layout.addWidget(self.sample_rate_label)
        metadata_layout.addWidget(self.channel_count_label)
        metadata_layout.addStretch()
        self.layout.addLayout(metadata_layout)

        controls_layout = QHBoxLayout()
        self.start_button = QPushButton("Start")
        self.start_button.clicked.connect(self.start_data_stream)
        self.pause_button = QPushButton("Pause")
        self.pause_button.setEnabled(False)
        self.pause_button.clicked.connect(self.toggle_pause_resume)
        self.stop_button = QPushButton("Stop")
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self.stop_data_stream)
        self.load_button = QPushButton("Load File")
        self.load_button.clicked.connect(self.open_data_file_dialog)
        self.signal_view_combo = QComboBox()
        self.signal_view_combo.currentIndexChanged.connect(self.handle_signal_view_changed)

        self.gain_slider = QSlider(Qt.Horizontal)
        self.gain_slider.setRange(1, 20)
        self.gain_slider.setValue(1)
        self.gain_slider.setTickInterval(1)
        self.gain_slider.setSingleStep(1)
        self.gain_slider.valueChanged.connect(self.on_gain_changed)
        self.gain_label = QLabel("Y Range: +/-1")

        controls_layout.addWidget(self.start_button)
        controls_layout.addWidget(self.pause_button)
        controls_layout.addWidget(self.stop_button)
        controls_layout.addWidget(self.load_button)
        controls_layout.addWidget(QLabel("Signal:"))
        controls_layout.addWidget(self.signal_view_combo)
        controls_layout.addStretch()
        controls_layout.addWidget(self.gain_label)
        controls_layout.addWidget(self.gain_slider)
        self.layout.addLayout(controls_layout)

        self.status_label = QLabel("Ready to load EMG data.")
        self.layout.addWidget(self.status_label)

        self.build_fatigue_controls()

        self.channel_combo = None
        self.channel_controls_layout = QHBoxLayout()
        self.layout.addLayout(self.channel_controls_layout)

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.graph_layout = pg.GraphicsLayoutWidget()
        self.scroll_area.setWidget(self.graph_layout)
        self.layout.addWidget(self.scroll_area)

        self.fatigue_graph_layout = pg.GraphicsLayoutWidget()
        self.fatigue_graph_layout.setMinimumHeight(320)
        self.layout.addWidget(self.fatigue_graph_layout)
        self.build_fatigue_plots()

        self.curves = []
        self.plot_items = []
        self.emg_data_buffer = np.zeros((1, self.LIVE_BUFFER_SAMPLES), dtype=np.float64)

        self.configure_signal_view_options()
        self.load_review_data_source(self.data_file_path)
        self.refresh_mode_controls()
        self.statusBar().showMessage("Ready")

    def build_fatigue_controls(self):
        fatigue_controls_layout = QHBoxLayout()

        self.fatigue_analyze_button = QPushButton("Analyze Fatigue")
        self.fatigue_analyze_button.clicked.connect(self.start_fatigue_analysis)
        self.fatigue_stop_button = QPushButton("Stop Analysis")
        self.fatigue_stop_button.setEnabled(False)
        self.fatigue_stop_button.clicked.connect(lambda: self.stop_fatigue_analysis())

        self.fatigue_channel_combo = QComboBox()
        self.fatigue_channel_combo.currentIndexChanged.connect(
            self.handle_fatigue_channel_changed
        )

        self.fatigue_window_spin = QDoubleSpinBox()
        self.fatigue_window_spin.setRange(0.1, 10.0)
        self.fatigue_window_spin.setDecimals(1)
        self.fatigue_window_spin.setSingleStep(0.1)
        self.fatigue_window_spin.setSuffix(" s")
        self.fatigue_window_spin.setValue(self.FATIGUE_WINDOW_SECONDS)

        self.fatigue_overlap_spin = QSpinBox()
        self.fatigue_overlap_spin.setRange(0, 90)
        self.fatigue_overlap_spin.setSuffix(" %")
        self.fatigue_overlap_spin.setValue(self.FATIGUE_OVERLAP_PERCENT)

        self.fatigue_nfft_spin = QSpinBox()
        self.fatigue_nfft_spin.setRange(0, 65536)
        self.fatigue_nfft_spin.setSingleStep(256)
        self.fatigue_nfft_spin.setSpecialValueText("Auto")
        self.fatigue_nfft_spin.setValue(0)

        self.fatigue_buffer_spin = QDoubleSpinBox()
        self.fatigue_buffer_spin.setRange(1.0, 60.0)
        self.fatigue_buffer_spin.setDecimals(1)
        self.fatigue_buffer_spin.setSingleStep(1.0)
        self.fatigue_buffer_spin.setSuffix(" s")
        self.fatigue_buffer_spin.setValue(self.FATIGUE_BUFFER_SECONDS)

        self.fatigue_metrics_label = QLabel("Fatigue: -")

        fatigue_controls_layout.addWidget(self.fatigue_analyze_button)
        fatigue_controls_layout.addWidget(self.fatigue_stop_button)
        fatigue_controls_layout.addWidget(QLabel("Fatigue Channel:"))
        fatigue_controls_layout.addWidget(self.fatigue_channel_combo)
        fatigue_controls_layout.addWidget(QLabel("Window:"))
        fatigue_controls_layout.addWidget(self.fatigue_window_spin)
        fatigue_controls_layout.addWidget(QLabel("Overlap:"))
        fatigue_controls_layout.addWidget(self.fatigue_overlap_spin)
        fatigue_controls_layout.addWidget(QLabel("FFT:"))
        fatigue_controls_layout.addWidget(self.fatigue_nfft_spin)
        fatigue_controls_layout.addWidget(QLabel("Live Buffer:"))
        fatigue_controls_layout.addWidget(self.fatigue_buffer_spin)
        fatigue_controls_layout.addStretch()
        fatigue_controls_layout.addWidget(self.fatigue_metrics_label)
        self.layout.addLayout(fatigue_controls_layout)

    def build_fatigue_plots(self):
        self.fatigue_graph_layout.clear()

        self.fatigue_psd_plot = self.fatigue_graph_layout.addPlot(
            row=0,
            col=0,
            title="Welch PSD",
        )
        self.fatigue_psd_plot.showGrid(x=True, y=True, alpha=0.3)
        self.fatigue_psd_plot.setLabel("bottom", "Frequency", units="Hz")
        self.fatigue_psd_plot.setLabel("left", "Power")
        self.fatigue_psd_curve = self.fatigue_psd_plot.plot(
            pen=pg.mkPen(color=(70, 140, 220), width=2)
        )

        self.fatigue_trend_plot = self.fatigue_graph_layout.addPlot(
            row=1,
            col=0,
            title="Median Frequency",
        )
        self.fatigue_trend_plot.showGrid(x=True, y=True, alpha=0.3)
        self.fatigue_trend_plot.setLabel("bottom", "Time", units="s")
        self.fatigue_trend_plot.setLabel("left", "Frequency", units="Hz")
        self.fatigue_median_curve = self.fatigue_trend_plot.plot(
            pen=pg.mkPen(color=(220, 120, 70), width=2)
        )

        self.fatigue_spectrogram_plot = self.fatigue_graph_layout.addPlot(
            row=0,
            col=1,
            rowspan=2,
            title="Spectrogram",
        )
        self.fatigue_spectrogram_plot.setLabel("bottom", "Time", units="s")
        self.fatigue_spectrogram_plot.setLabel("left", "Frequency", units="Hz")
        self.fatigue_spectrogram_item = pg.ImageItem()
        self.fatigue_spectrogram_plot.addItem(self.fatigue_spectrogram_item)
        self.clear_fatigue_result()

    def load_review_data_source(self, file_path: str):
        if self.data_worker is not None and self.data_worker.isRunning():
            self.stop_data_stream()
        self.stop_fatigue_workers(wait=False)
        self.clear_fatigue_result()

        try:
            sampling_rate = self.read_sampling_rate(file_path)
            self.data_source = FileDataSource(file_path=file_path, sampling_rate=sampling_rate)
            self.data_file_path = file_path
            original_data = self.data_source.get_data()
            self.review_signal_cache = {self.ORIGINAL_SIGNAL: original_data}
            self.emg_data_buffer = original_data
            self.num_channels = original_data.shape[0]
            self.current_file_label.setText(f"File: {os.path.basename(file_path)}")
            self.sample_rate_label.setText(f"Sample Rate: {self.data_source.sampling_rate} Hz")
            self.channel_count_label.setText(f"Channels: {self.num_channels}")
            self.update_status("Review dataset loaded.")
            self.build_graphs()
            self.render_review_data()
        except Exception as exc:
            self.data_source = None
            self.num_channels = 0
            self.review_signal_cache = {}
            self.emg_data_buffer = np.zeros((1, self.LIVE_BUFFER_SAMPLES), dtype=np.float64)
            self.current_file_label.setText("File: -")
            self.sample_rate_label.setText("Sample Rate: -")
            self.channel_count_label.setText("Channels: -")
            self.update_status(f"Failed to load file: {exc}")
            self.clear_graphs()
        finally:
            self.refresh_mode_controls()

    def read_sampling_rate(self, file_path: str) -> int:
        try:
            with open(file_path, "rb") as file:
                raw_data = pickle.load(file)
            sampling_rate = raw_data.get("device_information", {}).get(
                "sampling_frequency",
                self.DEFAULT_REVIEW_SAMPLING_RATE,
            )
            return int(sampling_rate)
        except Exception:
            return self.DEFAULT_REVIEW_SAMPLING_RATE

    def open_data_file_dialog(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select EMG dataset",
            os.path.abspath("data"),
            "Pickle files (*.pkl);;All files (*)",
        )
        if file_path:
            self.load_review_data_source(file_path)

    def handle_mode_changed(self):
        selected_mode = self.mode_combo.currentData()
        if selected_mode == self.current_mode:
            return

        if self.data_worker is not None and self.data_worker.isRunning():
            self.stop_data_stream()
        self.stop_fatigue_workers(wait=False)
        self.clear_fatigue_result()

        self.current_mode = selected_mode
        self.clear_graphs()
        self.configure_signal_view_options()

        if self.current_mode == self.REVIEW_MODE:
            self.load_review_data_source(self.data_file_path)
            return

        self.data_source = LiveDataSource()
        self.live_dsp = None
        self.num_channels = 0
        self.emg_data_buffer = np.zeros((1, self.LIVE_BUFFER_SAMPLES), dtype=np.float64)
        self.current_file_label.setText("File: -")
        self.sample_rate_label.setText("Sample Rate: -")
        self.channel_count_label.setText("Channels: -")
        self.update_status("Live mode selected. Hardware data source is not ready yet.")
        self.refresh_mode_controls()

    def refresh_mode_controls(self):
        is_streaming = self.data_worker is not None and self.data_worker.isRunning()
        is_review_mode = self.current_mode == self.REVIEW_MODE
        can_stream_live = self.current_mode == self.LIVE_MODE and self.can_stream_live_data()

        self.load_button.setEnabled(is_review_mode and not is_streaming)
        self.start_button.setEnabled(can_stream_live and not is_streaming)
        self.pause_button.setEnabled(is_streaming)
        self.stop_button.setEnabled(is_streaming)
        can_select_signal_view = (
            is_review_mode and self.num_channels > 0
        ) or self.current_mode == self.LIVE_MODE
        self.signal_view_combo.setEnabled(can_select_signal_view)
        if not is_streaming:
            self.pause_button.setText("Pause")
            self.is_paused = False
        self.refresh_fatigue_controls()

    def can_stream_live_data(self) -> bool:
        if self.data_source is None:
            return False

        get_chunk = getattr(self.data_source, "get_chunk", None)
        sampling_rate = getattr(self.data_source, "sampling_rate", None)
        chunk_size = getattr(self.data_source, "chunk_size", None)
        return callable(get_chunk) and sampling_rate is not None and chunk_size is not None

    def configure_signal_view_options(self):
        previous_view = self.current_signal_view
        if self.current_mode == self.REVIEW_MODE:
            options = (
                ("Original Signal", self.ORIGINAL_SIGNAL),
                ("Filtered", self.FILTERED_SIGNAL),
                ("RMS", self.RMS_SIGNAL),
            )
        else:
            options = (
                ("Original", self.ORIGINAL_SIGNAL),
                ("Processed", self.PROCESSED_SIGNAL),
            )

        self.signal_view_combo.blockSignals(True)
        self.signal_view_combo.clear()
        selected_index = 0
        for index, (label, value) in enumerate(options):
            self.signal_view_combo.addItem(label, value)
            if value == previous_view:
                selected_index = index
        self.signal_view_combo.setCurrentIndex(selected_index)
        self.current_signal_view = self.signal_view_combo.currentData()
        self.signal_view_combo.blockSignals(False)

    def handle_signal_view_changed(self):
        self.current_signal_view = self.signal_view_combo.currentData()

        if self.current_mode == self.REVIEW_MODE:
            self.render_review_data()
            return

        self.live_dsp = None
        if self.num_channels > 0:
            self.emg_data_buffer = np.zeros(
                (self.num_channels, self.LIVE_BUFFER_SAMPLES),
                dtype=np.float64,
            )
        self.update_status(f"Live view set to {self.signal_view_combo.currentText()}.")

    def get_dsp_band(self, sampling_rate: float) -> tuple[float, float]:
        nyquist = sampling_rate * 0.5
        highcut = min(self.DSP_HIGH_CUT, nyquist - 1.0)
        lowcut = min(self.DSP_LOW_CUT, highcut * 0.5)
        if lowcut <= 0 or highcut <= lowcut:
            raise ValueError(f"Invalid DSP band for {sampling_rate} Hz sampling rate.")
        return lowcut, highcut

    def get_current_sampling_rate(self) -> float:
        sampling_rate = getattr(self.data_source, "sampling_rate", None)
        try:
            sampling_rate = float(sampling_rate)
        except (TypeError, ValueError):
            return float(self.DEFAULT_REVIEW_SAMPLING_RATE)
        if sampling_rate <= 0:
            return float(self.DEFAULT_REVIEW_SAMPLING_RATE)
        return sampling_rate

    def get_review_time_axis(self, sample_count: int) -> np.ndarray:
        return np.arange(sample_count, dtype=np.float64) / self.get_current_sampling_rate()

    def get_live_time_axis(self, sample_count: int) -> np.ndarray:
        sampling_rate = self.get_current_sampling_rate()
        return (np.arange(sample_count, dtype=np.float64) - sample_count + 1) / sampling_rate

    def build_graphs(self):
        self.graph_layout.clear()
        self.curves = []
        self.plot_items = []
        self.clear_channel_controls()

        for channel_index in range(self.num_channels):
            plot_item = self.graph_layout.addPlot(row=channel_index, col=0)
            plot_item.showGrid(x=True, y=True, alpha=0.3)
            plot_item.setLabel("left", "EMG Amplitude (uV)")
            plot_item.setLabel("bottom", "Time (s)")
            plot_item.setYRange(-self.current_gain, self.current_gain)
            curve = plot_item.plot(
                pen=pg.mkPen(color=(channel_index * 20 % 255, 100, 200), width=1)
            )
            self.curves.append(curve)
            self.plot_items.append(plot_item)
            self.graph_layout.nextRow()

        self.channel_combo = None
        if self.num_channels > 0:
            self.channel_combo = QComboBox()
            for channel_index in range(self.num_channels):
                self.channel_combo.addItem(f"Channel {channel_index + 1}", channel_index)
            self.channel_combo.currentIndexChanged.connect(self.handle_channel_combo_changed)
            self.channel_controls_layout.addWidget(QLabel("Select Channel:"))
            self.channel_controls_layout.addWidget(self.channel_combo)
            self.handle_channel_combo_changed(0)
        self.channel_controls_layout.addStretch()
        self.refresh_fatigue_channel_options()

    def clear_channel_controls(self):
        while self.channel_controls_layout.count():
            item = self.channel_controls_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

    def clear_graphs(self):
        self.graph_layout.clear()
        self.curves = []
        self.plot_items = []
        self.channel_combo = None
        self.clear_channel_controls()
        self.refresh_fatigue_channel_options()

    def update_status(self, message: str):
        self.status_label.setText(message)
        self.statusBar().showMessage(message)

    def on_gain_changed(self, value: int):
        self.current_gain = value
        self.gain_label.setText(f"Y Range: +/-{value}")
        for plot_item in self.plot_items:
            plot_item.setYRange(-value, value)
        self.update_status(f"Vertical scale set to +/-{value}")

    def handle_channel_combo_changed(self, index):
        for idx, plot_item in enumerate(self.plot_items):
            plot_item.setVisible(idx == index)

    def refresh_fatigue_controls(self):
        if not hasattr(self, "fatigue_analyze_button"):
            return

        is_streaming = self.data_worker is not None and self.data_worker.isRunning()
        is_running = self.is_fatigue_analysis_running()
        review_ready = self.current_mode == self.REVIEW_MODE and self.num_channels > 0
        live_ready = self.current_mode == self.LIVE_MODE and is_streaming

        if self.current_mode == self.REVIEW_MODE:
            self.fatigue_analyze_button.setText("Analyze Fatigue")
        else:
            self.fatigue_analyze_button.setText("Start Fatigue")

        self.fatigue_analyze_button.setEnabled((review_ready or live_ready) and not is_running)
        self.fatigue_stop_button.setEnabled(is_running)
        self.fatigue_channel_combo.setEnabled(self.num_channels > 0)
        self.fatigue_window_spin.setEnabled(not is_running)
        self.fatigue_overlap_spin.setEnabled(not is_running)
        self.fatigue_nfft_spin.setEnabled(not is_running)
        self.fatigue_buffer_spin.setEnabled(
            self.current_mode == self.LIVE_MODE and not is_running
        )

    def is_fatigue_analysis_running(self) -> bool:
        review_running = (
            self.fatigue_review_worker is not None and self.fatigue_review_worker.isRunning()
        )
        live_running = (
            self.fatigue_live_worker is not None and self.fatigue_live_worker.isRunning()
        )
        return review_running or live_running

    def refresh_fatigue_channel_options(self):
        if not hasattr(self, "fatigue_channel_combo"):
            return

        previous_channel = self.fatigue_channel_combo.currentData()
        if previous_channel is None:
            previous_channel = 0

        self.fatigue_channel_combo.blockSignals(True)
        self.fatigue_channel_combo.clear()
        selected_index = 0
        for channel_index in range(self.num_channels):
            self.fatigue_channel_combo.addItem(f"Channel {channel_index + 1}", channel_index)
            if channel_index == previous_channel:
                selected_index = channel_index
        if self.num_channels > 0:
            self.fatigue_channel_combo.setCurrentIndex(min(selected_index, self.num_channels - 1))
        self.fatigue_channel_combo.blockSignals(False)
        self.refresh_fatigue_controls()
        self.render_fatigue_result()

    def handle_fatigue_channel_changed(self, index=None):
        self.render_fatigue_result()

    def get_selected_fatigue_channel(self) -> int:
        selected_channel = self.fatigue_channel_combo.currentData()
        if selected_channel is None:
            return 0
        return int(selected_channel)

    def get_fatigue_config(self) -> FatigueAnalysisConfig:
        sampling_rate = self.get_current_sampling_rate()
        lowcut, highcut = self.get_dsp_band(sampling_rate)
        nfft = self.fatigue_nfft_spin.value()
        return FatigueAnalysisConfig(
            window_seconds=self.fatigue_window_spin.value(),
            overlap_fraction=self.fatigue_overlap_spin.value() / 100.0,
            nfft=None if nfft == 0 else nfft,
            min_frequency=lowcut,
            max_frequency=highcut,
        )

    def start_fatigue_analysis(self):
        if self.is_fatigue_analysis_running():
            return

        if self.current_mode == self.REVIEW_MODE:
            self.start_review_fatigue_analysis()
            return

        self.start_live_fatigue_analysis()

    def start_review_fatigue_analysis(self):
        if self.ORIGINAL_SIGNAL not in self.review_signal_cache:
            self.update_status("No review dataset is loaded.")
            return

        try:
            config = self.get_fatigue_config()
        except Exception as exc:
            self.update_status(f"Invalid fatigue settings: {exc}")
            return

        source_data = self.review_signal_cache[self.ORIGINAL_SIGNAL]
        self.fatigue_review_worker = FatigueReviewWorker(
            data=source_data,
            sampling_rate=self.get_current_sampling_rate(),
            config=config,
        )
        self.fatigue_review_worker.result_ready.connect(self.handle_fatigue_result)
        self.fatigue_review_worker.error_signal.connect(self.handle_fatigue_error)
        self.fatigue_review_worker.finished.connect(self.on_fatigue_review_finished)
        self.fatigue_review_worker.start()
        self.update_status("Running review fatigue analysis.")
        self.refresh_fatigue_controls()

    def start_live_fatigue_analysis(self):
        if self.data_worker is None or not self.data_worker.isRunning():
            self.update_status("Start live streaming before fatigue analysis.")
            return

        try:
            config = self.get_fatigue_config()
        except Exception as exc:
            self.update_status(f"Invalid fatigue settings: {exc}")
            return

        self.fatigue_live_worker = FatigueLiveWorker(
            sampling_rate=self.get_current_sampling_rate(),
            config=config,
            buffer_seconds=self.fatigue_buffer_spin.value(),
            analysis_interval_ms=self.FATIGUE_ANALYSIS_INTERVAL_MS,
        )
        self.fatigue_live_worker.result_ready.connect(self.handle_fatigue_result)
        self.fatigue_live_worker.error_signal.connect(self.handle_fatigue_error)
        self.fatigue_live_worker.finished.connect(self.on_fatigue_live_finished)
        self.fatigue_live_worker.start()
        self.update_status("Live fatigue analysis started.")
        self.refresh_fatigue_controls()

    def stop_fatigue_analysis(self):
        self.stop_fatigue_workers(wait=False)
        self.update_status("Fatigue analysis stopped.")

    def stop_fatigue_workers(self, wait: bool):
        review_worker = self.fatigue_review_worker
        if review_worker is not None:
            review_worker.cancel()
            if wait and review_worker.isRunning():
                review_worker.wait(1000)

        live_worker = self.fatigue_live_worker
        if live_worker is not None:
            live_worker.stop()
            if wait and live_worker.isRunning():
                live_worker.wait(1000)

        self.refresh_fatigue_controls()

    def on_fatigue_review_finished(self):
        self.fatigue_review_worker = None
        self.refresh_fatigue_controls()

    def on_fatigue_live_finished(self):
        self.fatigue_live_worker = None
        self.refresh_fatigue_controls()

    def handle_fatigue_error(self, message: str):
        self.update_status(f"Fatigue analysis failed: {message}")

    def handle_fatigue_result(self, result):
        self.fatigue_result = result
        self.render_fatigue_result()
        self.update_status("Fatigue analysis updated.")

    def submit_live_fatigue_chunk(self, data_chunk: np.ndarray):
        if self.fatigue_live_worker is None or not self.fatigue_live_worker.isRunning():
            return

        try:
            self.fatigue_live_worker.add_chunk(data_chunk)
        except Exception as exc:
            self.update_status(f"Could not queue fatigue chunk: {exc}")

    def render_fatigue_result(self):
        if not hasattr(self, "fatigue_psd_curve") or self.fatigue_result is None:
            return
        if self.num_channels == 0:
            return

        result = self.fatigue_result
        channel_index = min(self.get_selected_fatigue_channel(), result.welch_power.shape[0] - 1)

        self.fatigue_psd_curve.setData(
            result.frequencies,
            result.welch_power[channel_index],
        )
        self.fatigue_median_curve.setData(
            result.times,
            result.spectrogram_median_frequency[channel_index],
        )

        power = result.spectrogram_power[channel_index]
        log_power = 10.0 * np.log10(np.maximum(power, np.finfo(np.float64).tiny))
        self.fatigue_spectrogram_item.setImage(log_power.T, autoLevels=True)
        self.set_fatigue_spectrogram_rect(result.times, result.frequencies)

        self.fatigue_metrics_label.setText(
            "Fatigue: "
            f"Median {self.format_frequency(result.median_frequency[channel_index])} | "
            f"Mean {self.format_frequency(result.mean_frequency[channel_index])} | "
            f"Power {self.format_power(result.band_power[channel_index])}"
        )

    def clear_fatigue_result(self):
        self.fatigue_result = None
        if hasattr(self, "fatigue_psd_curve"):
            self.fatigue_psd_curve.setData([], [])
        if hasattr(self, "fatigue_median_curve"):
            self.fatigue_median_curve.setData([], [])
        if hasattr(self, "fatigue_spectrogram_item"):
            self.fatigue_spectrogram_item.clear()
        if hasattr(self, "fatigue_metrics_label"):
            self.fatigue_metrics_label.setText("Fatigue: -")

    def set_fatigue_spectrogram_rect(self, times: np.ndarray, frequencies: np.ndarray):
        if times.size == 0 or frequencies.size == 0:
            return

        time_width = self.get_axis_width(times, self.fatigue_window_spin.value())
        frequency_height = self.get_axis_width(frequencies, 1.0)
        self.fatigue_spectrogram_item.setRect(
            QRectF(
                float(times[0]),
                float(frequencies[0]),
                time_width,
                frequency_height,
            )
        )

    @staticmethod
    def get_axis_width(values: np.ndarray, default_width: float) -> float:
        if values.size < 2:
            return float(default_width)
        width = float(values[-1] - values[0])
        return width if width > 0.0 else float(default_width)

    @staticmethod
    def format_frequency(value: float) -> str:
        if not np.isfinite(value):
            return "-"
        return f"{value:.1f} Hz"

    @staticmethod
    def format_power(value: float) -> str:
        if not np.isfinite(value):
            return "-"
        return f"{value:.3g}"

    def render_review_data(self):
        if self.current_mode != self.REVIEW_MODE or self.num_channels == 0:
            return

        try:
            self.emg_data_buffer = self.get_review_signal_data(self.current_signal_view)
        except Exception as exc:
            self.update_status(f"Failed to apply review DSP: {exc}")
            return

        time_axis = self.get_review_time_axis(self.emg_data_buffer.shape[1])
        for channel_index, curve in enumerate(self.curves):
            curve.setData(time_axis, self.emg_data_buffer[channel_index, :])
        self.update_status(f"Review view set to {self.signal_view_combo.currentText()}.")

    def get_review_signal_data(self, signal_view: str) -> np.ndarray:
        if signal_view in self.review_signal_cache:
            return self.review_signal_cache[signal_view]

        original_data = self.review_signal_cache[self.ORIGINAL_SIGNAL]
        sampling_rate = self.data_source.sampling_rate
        lowcut, highcut = self.get_dsp_band(sampling_rate)

        if self.FILTERED_SIGNAL not in self.review_signal_cache:
            self.review_signal_cache[self.FILTERED_SIGNAL] = DSPReview.bandpass_filter(
                original_data,
                lowcut=lowcut,
                highcut=highcut,
                fs=sampling_rate,
                order=self.DSP_FILTER_ORDER,
            )

        if signal_view == self.FILTERED_SIGNAL:
            return self.review_signal_cache[self.FILTERED_SIGNAL]

        if signal_view == self.RMS_SIGNAL:
            self.review_signal_cache[self.RMS_SIGNAL] = self.review_dsp.rms_envelope(
                self.review_signal_cache[self.FILTERED_SIGNAL],
                sampling_rate=sampling_rate,
                window_ms=self.DSP_RMS_WINDOW_MS,
            )
            return self.review_signal_cache[self.RMS_SIGNAL]

        return original_data

    def prepare_live_stream(self):
        self.live_dsp = None
        live_channels = self.get_live_num_channels()
        if live_channels <= 0:
            return

        self.num_channels = live_channels
        self.emg_data_buffer = np.zeros(
            (self.num_channels, self.LIVE_BUFFER_SAMPLES),
            dtype=np.float64,
        )
        self.channel_count_label.setText(f"Channels: {self.num_channels}")
        self.sample_rate_label.setText(f"Sample Rate: {self.data_source.sampling_rate} Hz")
        self.build_graphs()

    def get_live_num_channels(self) -> int:
        if self.data_source is None:
            return 0

        get_num_channels = getattr(self.data_source, "get_num_channels", None)
        if callable(get_num_channels):
            try:
                return int(get_num_channels())
            except Exception:
                return 0

        for attribute in ("num_channels", "number_of_channels"):
            value = getattr(self.data_source, attribute, None)
            if value is not None:
                return int(value)
        return 0

    def prepare_live_display_for_chunk(self, data_chunk: np.ndarray):
        if self.num_channels == data_chunk.shape[0]:
            return

        self.num_channels = data_chunk.shape[0]
        self.emg_data_buffer = np.zeros(
            (self.num_channels, self.LIVE_BUFFER_SAMPLES),
            dtype=np.float64,
        )
        self.channel_count_label.setText(f"Channels: {self.num_channels}")
        self.build_graphs()

    def process_live_chunk(self, data_chunk: np.ndarray) -> np.ndarray:
        if self.current_signal_view != self.PROCESSED_SIGNAL:
            return data_chunk

        if self.live_dsp is None or self.live_dsp.num_channels != data_chunk.shape[0]:
            lowcut, highcut = self.get_dsp_band(self.data_source.sampling_rate)
            self.live_dsp = DSPLive(
                num_channels=data_chunk.shape[0],
                sampling_rate=self.data_source.sampling_rate,
                lowcut=lowcut,
                highcut=highcut,
                filter_order=self.DSP_FILTER_ORDER,
                rms_window_ms=self.DSP_RMS_WINDOW_MS,
            )

        return self.live_dsp.process_chunk(data_chunk)

    def update_graph(self, data_chunk: np.ndarray):
        if data_chunk.size == 0:
            return

        data_chunk = np.asarray(data_chunk)
        if data_chunk.ndim == 1:
            data_chunk = data_chunk.reshape(1, -1)

        if self.current_mode == self.LIVE_MODE:
            self.submit_live_fatigue_chunk(data_chunk)
            data_chunk = self.process_live_chunk(data_chunk)
            self.prepare_live_display_for_chunk(data_chunk)

        self.emg_data_buffer = np.roll(self.emg_data_buffer, -data_chunk.shape[1], axis=1)
        self.emg_data_buffer[:, -data_chunk.shape[1]:] = data_chunk

        time_axis = self.get_live_time_axis(self.emg_data_buffer.shape[1])
        for channel_index, curve in enumerate(self.curves):
            if self.plot_items[channel_index].isVisible():
                curve.setData(time_axis, self.emg_data_buffer[channel_index, :])

    def start_data_stream(self):
        if self.current_mode == self.REVIEW_MODE:
            self.update_status("Review mode displays the loaded file directly.")
            return

        if self.data_source is None:
            self.update_status("No data source available.")
            return

        if not self.can_stream_live_data():
            self.update_status("Live data source is not ready for streaming yet.")
            return

        if self.data_worker is not None and self.data_worker.isRunning():
            self.update_status("Data stream already running.")
            return

        self.prepare_live_stream()
        self.data_worker = DataWorker(data_source=self.data_source)
        self.data_worker.data_chunk_signal.connect(self.update_graph)
        self.data_worker.finished.connect(self.on_worker_finished)
        self.data_worker.start()

        self.refresh_mode_controls()
        self.pause_button.setText("Pause")
        self.is_paused = False
        self.update_status("Streaming data in real time.")

    def toggle_pause_resume(self):
        if self.data_worker is None or not self.data_worker.isRunning():
            return

        if self.is_paused:
            self.data_worker.resume()
            self.pause_button.setText("Pause")
            self.update_status("Resumed data streaming.")
            self.is_paused = False
        else:
            self.data_worker.pause()
            self.pause_button.setText("Resume")
            self.update_status("Data stream paused.")
            self.is_paused = True

    def stop_data_stream(self):
        if self.data_worker is None:
            return

        self.stop_fatigue_workers(wait=False)
        self.data_worker.stop()
        self.data_worker = None
        self.refresh_mode_controls()
        self.update_status("Data stream stopped.")

    def on_worker_finished(self):
        self.data_worker = None
        self.refresh_mode_controls()
        self.update_status("Worker thread finished.")

    def closeEvent(self, event):
        self.stop_fatigue_workers(wait=True)
        if self.data_worker is not None and self.data_worker.isRunning():
            self.data_worker.stop()
        event.accept()
