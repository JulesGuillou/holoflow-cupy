import time
import queue
from PySide6.QtWidgets import QMainWindow, QVBoxLayout, QWidget
from PySide6.QtCore import QTimer
import cupyx.profiler

from holoflow.ui.gl_widget import HoloGLWidget
from holoflow.core.pipeline import FramePipeline, DataLoader


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Holoflow - Real-time Doppler Holography")
        self.resize(800, 600)

        central_widget = QWidget()
        layout = QVBoxLayout(central_widget)

        self.gl_viewer = HoloGLWidget()
        layout.addWidget(self.gl_viewer)
        self.setCentralWidget(central_widget)

        # Initialize pipeline
        loader = DataLoader(
            file_path="C:\\Users\\guill\\Documents\\holofiles_data\\250527_GUJ_L_2.holo",
            start_frame=0,
            end_frame=16384,
            batch_size=128,
            load_kind="cpu",
        )
        self.pipeline = FramePipeline(
            loader=loader,
            queue_depth=128,
        )
        self.pipeline.start()

        # Sink FPS tracking variables
        self.frames_displayed = 0
        self.last_fps_time = time.time()

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.poll_queue)
        self.timer.start(1e3 / 100)

    def poll_queue(self):
        frame = self.pipeline.pull_latest_frame()

        if frame is not None:
            with cupyx.profiler.time_range("UI: update_frame", color_id=6):
                self.gl_viewer.update_frame(frame)
            self.pipeline.return_frame(frame)

            # --- Sink FPS Counter ---
            self.frames_displayed += 1
            now = time.time()
            if now - self.last_fps_time >= 1.0:
                fps = self.frames_displayed / (now - self.last_fps_time)
                print(f"[Sink] Display FPS: {fps:.1f}")
                self.frames_displayed = 0
                self.last_fps_time = now

    def closeEvent(self, event):
        self.pipeline.stop()
        super().closeEvent(event)
