#!/usr/bin/env python
"""Entry point for PyInstaller build — launches py4D-browser with the fast-acbf plugin."""
import sys
import py4D_browser
from PyQt5.QtWidgets import QApplication
from py4D_browser.main_window import DataViewer

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = DataViewer(sys.argv)
    win.show()
    sys.exit(app.exec_())
