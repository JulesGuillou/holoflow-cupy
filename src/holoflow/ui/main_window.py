"""
main_window.py — Qt main window.

Owns the pipeline lifecycle and drives the render loop via a QTimer.
All tuneable values come from the config dict loaded by main.py.
"""

import time

import cupyx.profiler
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QMainWindow, QVBoxLayout, QWidget

from holoflow.core.pipeline import DataLoader, FramePipeline
from holoflow.ui.gl_widget import HoloGLWidget


class MainWindow(QMainWindow):
    """Top-level window.  Starts the pipeline and polls it for new frames."""

    def __init__(self, config: dict) -> None:
        super().__init__()

        ui_cfg = config["ui"]
        pipe_cfg = config["pipeline"]

        self.setWindowTitle(ui_cfg["window_title"])
        self.resize(ui_cfg["window_width"], ui_cfg["window_height"])

        # Central widget
        central = QWidget()
        layout = QVBoxLayout(central)
        self._gl_viewer = HoloGLWidget()
        layout.addWidget(self._gl_viewer)
        self.setCentralWidget(central)

        # Pipeline
        loader = DataLoader(
            file_path=pipe_cfg["file_path"],
            start_frame=pipe_cfg["start_frame"],
            end_frame=pipe_cfg["end_frame"],
            batch_size=pipe_cfg["batch_size"],
            load_kind=pipe_cfg["load_kind"],
        )
        self._pipeline = FramePipeline(
            loader=loader,
            queue_depth=pipe_cfg["queue_depth"],
        )
        self._pipeline.start()

        self._window_base_title = ui_cfg["window_title"]

        # Sink FPS tracking
        self._frames_displayed = 0
        self._last_fps_time = time.monotonic()

        poll_ms = round(1000 / ui_cfg["poll_fps"])
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._poll_queue)
        self._timer.start(poll_ms)

    def _poll_queue(self) -> None:
        """Pull the latest frame from the pipeline and hand it to the renderer."""
        frame = self._pipeline.pull_latest_frame()
        if frame is None:
            return

        with cupyx.profiler.time_range("UI: update_frame", color_id=6):
            self._gl_viewer.update_frame(frame)
        self._pipeline.return_frame(frame)

        self._frames_displayed += 1
        now = time.monotonic()
        elapsed = now - self._last_fps_time
        if elapsed >= 1.0:
            sink_fps = self._frames_displayed / elapsed
            self._frames_displayed = 0
            self._last_fps_time = now

            input_fps = self._pipeline.pop_input_fps()
            input_str = f"{input_fps:.0f}" if input_fps is not None else "—"
            self.setWindowTitle(
                f"{self._window_base_title}  |  Input: {input_str} FPS  |  Display: {sink_fps:.0f} FPS"
            )

    def closeEvent(self, event) -> None:
        self._pipeline.stop()
        super().closeEvent(event)
