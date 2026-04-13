"""
main.py — Application entry point.

Loads config.yaml from the project root, then launches the Qt application.
"""

import sys
from pathlib import Path

import yaml
from PySide6.QtWidgets import QApplication

from holoflow.ui.main_window import MainWindow

# config.yaml lives at the project root, two directories above this package.
_CONFIG_PATH = Path(__file__).parents[2] / "config.yaml"


def main() -> None:
    with open(_CONFIG_PATH) as f:
        config = yaml.safe_load(f)

    app = QApplication(sys.argv)
    window = MainWindow(config)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
