import os
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QFileDialog,
    QSlider,
    QCheckBox,
    QPushButton,
    QComboBox
)
import pyqtgraph as pg
from src.core.data_source import MockDataSource
from src.core.data_worker import DataWorker


class MainWindow(QMainWindow):
    """
    The MainWindow class is responsible for setting up the main user interface of the application.
    It initializes the DataSource and DataWorker, and connects the data signals to the appropriate
    slots for processing and visualization. This class serves as the central hub for managing
    the application's data flow and user interactions.
    """

    def __init__(self):
        super().__init__()

        self.data_file_path = os.path.join("data", "recording.pkl")
        self.data_source = None
        self.data_worker = None
        self.num_channels = 0
        self.current_gain = 1
        self.is_paused = False

        self.setWindowTitle("EMG Data Visualization")
        self.resize(1000, 800)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout()
        self.central_widget.setLayout(self.layout)

        # Header metadata display
        metadata_layout = QHBoxLayout()
        self.current_file_label = QLabel(f"File: {self.data_file_path}")
        self.sample_rate_label = QLabel("Sample Rate: -")
        self.channel_count_label = QLabel("Channels: -")
        metadata_layout.addWidget(self.current_file_label)
        metadata_layout.addWidget(self.sample_rate_label)
        metadata_layout.addWidget(self.channel_count_label)
        metadata_layout.addStretch()
        self.layout.addLayout(metadata_layout)

        # Controls for runtime interaction
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

        self.gain_slider = QSlider(Qt.Horizontal)
        self.gain_slider.setRange(1, 20)
        self.gain_slider.setValue(1)
        self.gain_slider.setTickInterval(1)
        self.gain_slider.setSingleStep(1)
        self.gain_slider.valueChanged.connect(self.on_gain_changed)
        self.gain_label = QLabel("Y Range: ±1")

        controls_layout.addWidget(self.start_button)
        controls_layout.addWidget(self.pause_button)
        controls_layout.addWidget(self.stop_button)
        controls_layout.addWidget(self.load_button)
        controls_layout.addStretch()
        controls_layout.addWidget(self.gain_label)
        controls_layout.addWidget(self.gain_slider)
        self.layout.addLayout(controls_layout)

        # Status messages and information
        self.status_label = QLabel("Ready to load EMG data.")
        self.layout.addWidget(self.status_label)

        # Scrolling plot container
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.graph_layout = pg.GraphicsLayoutWidget()
        self.scroll_area.setWidget(self.graph_layout)
        self.layout.addWidget(self.scroll_area)

        self.curves = []
        self.plot_items = []
        self.emg_data_buffer = np.zeros((1, 4000), dtype=np.float64)
        # Channel selection combo box
        self.channel_combo = None
        self.channel_controls_layout = QHBoxLayout()
        self.layout.addLayout(self.channel_controls_layout)
        self.load_data_source(self.data_file_path)
        self.statusBar().showMessage("Ready")

    def load_data_source(self, file_path: str):
        if self.data_worker is not None and self.data_worker.isRunning():
            self.stop_data_stream()

        try:
            self.data_source = MockDataSource(file_path=file_path, chunk_size=50)
            self.num_channels = self.data_source.get_num_channels()
            self.emg_data_buffer = np.zeros((self.num_channels, 4000), dtype=np.float64)
            self.current_file_label.setText(f"File: {os.path.basename(file_path)}")
            self.sample_rate_label.setText(f"Sample Rate: {self.data_source.sampling_rate} Hz")
            self.channel_count_label.setText(f"Channels: {self.num_channels}")
            self.status_label.setText("Dataset loaded successfully.")
            self.update_status("Dataset loaded. Click Start to begin streaming.")
            self.build_graphs()
        except Exception as exc:
            self.data_source = None
            self.num_channels = 0
            self.emg_data_buffer = np.zeros((1, 4000), dtype=np.float64)
            self.current_file_label.setText("File: -")
            self.sample_rate_label.setText("Sample Rate: -")
            self.channel_count_label.setText("Channels: -")
            self.status_label.setText(f"Failed to load file: {exc}")
            self.update_status("Failed to load dataset.")
            self.clear_graphs()

    def open_data_file_dialog(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select EMG dataset",
            os.path.abspath("data"),
            "Pickle files (*.pkl);;All files (*)",
        )
        if file_path:
            self.load_data_source(file_path)

    def build_graphs(self):
        self.graph_layout.clear()
        self.curves = []
        self.plot_items = []
        self.channel_controls_layout.setParent(None)
        self.channel_controls_layout = QHBoxLayout()
        self.layout.insertLayout(3, self.channel_controls_layout)
        
        for channel_index in range(self.num_channels):
            plot_item = self.graph_layout.addPlot(row=channel_index, col=0)
            plot_item.showGrid(x=True, y=True, alpha=0.3)
            plot_item.setLabel("left", "EMG Amplitude (µV)")
            plot_item.setLabel("bottom", "Sample Index (n)")
            curve = plot_item.plot(pen=pg.mkPen(color=(channel_index * 20 % 255, 100, 200), width=1))
            self.curves.append(curve)
            self.plot_items.append(plot_item)
            self.graph_layout.nextRow()

        # Add combo box for channel selection
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

    def clear_graphs(self):
        self.graph_layout.clear()
        self.curves = []
        self.plot_items = []
        if self.channel_combo is not None:
            self.channel_combo.setParent(None)
            self.channel_combo = None

    def update_status(self, message: str):
        self.status_label.setText(message)
        self.statusBar().showMessage(message)

    def on_gain_changed(self, value: int):
        self.current_gain = value
        self.gain_label.setText(f"Y Range: ±{value}")
        for plot_item in self.plot_items:
            plot_item.setYRange(-value, value)
        self.update_status(f"Vertical scale set to ±{value}")

    def handle_channel_combo_changed(self, index):
        # Show only the selected channel, hide others
        for idx, plot_item in enumerate(self.plot_items):
            plot_item.setVisible(idx == index)

    def update_graph(self, data_chunk: np.ndarray):
        if self.num_channels == 0 or data_chunk.size == 0:
            return

        self.emg_data_buffer = np.roll(self.emg_data_buffer, -data_chunk.shape[1], axis=1)
        self.emg_data_buffer[:, -data_chunk.shape[1]:] = data_chunk

        for channel_index, curve in enumerate(self.curves):
            if self.plot_items[channel_index].isVisible():
                curve.setData(self.emg_data_buffer[channel_index, :])

    def start_data_stream(self):
        if self.data_source is None:
            self.update_status("No data source available.")
            return

        if self.data_worker is not None and self.data_worker.isRunning():
            self.update_status("Data stream already running.")
            return

        self.data_worker = DataWorker(data_source=self.data_source)
        self.data_worker.data_chunk_signal.connect(self.update_graph)
        self.data_worker.finished.connect(self.on_worker_finished)
        self.data_worker.start()

        self.start_button.setEnabled(False)
        self.pause_button.setEnabled(True)
        self.stop_button.setEnabled(True)
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

        self.data_worker.stop()
        self.data_worker = None

        self.start_button.setEnabled(True)
        self.pause_button.setEnabled(False)
        self.pause_button.setText("Pause")
        self.stop_button.setEnabled(False)
        self.is_paused = False
        self.update_status("Data stream stopped.")

    def on_worker_finished(self):
        self.start_button.setEnabled(True)
        self.pause_button.setEnabled(False)
        self.pause_button.setText("Pause")
        self.stop_button.setEnabled(False)
        self.update_status("Worker thread finished.")

    def closeEvent(self, event):
        if self.data_worker is not None and self.data_worker.isRunning():
            self.data_worker.stop()
        event.accept()



