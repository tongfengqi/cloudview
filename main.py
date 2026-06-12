"""Point Cloud Viewer - Entry Point"""

import sys
import os

# Add tool directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt
from ui_mainwindow import MainWindow


def main():
    # High DPI support
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )

    app = QApplication(sys.argv)
    app.setApplicationName("Point Cloud Viewer")

    window = MainWindow()
    window.show()

    # Load file from command line if provided
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
        if os.path.isfile(filepath):
            window._load_file(filepath)

    sys.exit(app.exec())


if __name__ == '__main__':
    main()
