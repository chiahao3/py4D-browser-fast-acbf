"""fast-acbf py4D-browser plugin entry point."""

from __future__ import annotations

import traceback
from typing import TYPE_CHECKING

import numpy as np
from PyQt5.QtWidgets import QAction, QMessageBox, QWidget

from .config import FastAcbfConfig
from .dialogs import ConfigurationDialog, FastAcbfDashboard
from .worker import FastAcbfJobState, FastAcbfRunner

if TYPE_CHECKING:
    from py4D_browser import DataViewer


class FastAcbfPlugin(QWidget):
    plugin_id = "chiahao3.fast_acbf"
    display_name = "fast-acbf"
    uses_plugin_menu = True

    def __init__(self, parent: "DataViewer", plugin_menu, **kwargs):
        super().__init__(parent=parent)
        self.parent = parent
        self.fast_acbf_menu = plugin_menu
        self.config = FastAcbfConfig()
        self.job_state = FastAcbfJobState()
        self.runner: FastAcbfRunner | None = None
        self.dashboard: FastAcbfDashboard | None = None
        self._calibration_dialog = None

        self.dashboard_action = QAction("Interactive Dashboard", self)
        self.dashboard_action.triggered.connect(self.launch_dashboard)
        self.fast_acbf_menu.addAction(self.dashboard_action)

        self.quick_run_action = QAction("Quick Run (Last Config)", self)
        self.quick_run_action.triggered.connect(self.quick_run)
        self.fast_acbf_menu.addAction(self.quick_run_action)

        self.config_action = QAction("Configuration", self)
        self.config_action.triggered.connect(self.launch_config)
        self.fast_acbf_menu.addAction(self.config_action)

        signal = getattr(parent, "signal_datacube_changed", None)
        if signal is not None:
            signal.connect(self._datacube_changed)

    def close(self):
        if self.runner is not None and self.runner.isRunning():
            self.runner.wait(1000)
        if self.dashboard is not None:
            self.dashboard.close()

    def _datacube_changed(self) -> None:
        self.job_state = FastAcbfJobState()

    def _status(self, message: str, timeout: int = 5000) -> None:
        try:
            self.parent.statusBar().showMessage(message, timeout)
        except Exception:
            print(message)
        if self.dashboard is not None:
            self.dashboard.set_status(message)

    def _has_datacube(self) -> bool:
        if getattr(self.parent, "datacube", None) is None:
            QMessageBox.warning(self.parent, "fast-acbf", "Load a 4D datacube before running fast-acbf.")
            return False
        data = getattr(self.parent.datacube, "data", None)
        if data is None or len(getattr(data, "shape", ())) != 4:
            QMessageBox.warning(self.parent, "fast-acbf", "fast-acbf requires a 4D datacube.")
            return False
        return True

    def _resolved_config(self) -> FastAcbfConfig:
        return self.config.resolved_for(self.parent)

    def _refresh_calibration_display(self) -> None:
        refreshed = self._resolved_config()
        self.config.voltage_kv = refreshed.voltage_kv
        self.config.wavelength_angstrom = refreshed.wavelength_angstrom
        self.config.scan_step_angstrom = refreshed.scan_step_angstrom
        self.config.dk_inv_angstrom = refreshed.dk_inv_angstrom
        self.config.max_alpha_mrad = refreshed.max_alpha_mrad
        if self.dashboard is not None:
            self.dashboard.set_config(refreshed)

    def launch_dashboard(self) -> None:
        cfg = self._resolved_config()
        if self.dashboard is None:
            self.dashboard = FastAcbfDashboard(cfg, parent=self.parent)
            self.dashboard.run_requested.connect(self._dashboard_run_requested)
            self.dashboard.config_requested.connect(self.launch_config)
            self.dashboard.calibration_requested.connect(self.launch_py4d_calibration)
            self.dashboard.config_changed.connect(self._dashboard_config_changed)
        else:
            self.dashboard.set_config(cfg)
        self.dashboard.show()
        self.dashboard.raise_()

    def _dashboard_config_changed(self, config: FastAcbfConfig) -> None:
        self.config = config.copy()

    def _dashboard_run_requested(self, command: str) -> None:
        if self.dashboard is not None:
            self.config = self.dashboard.config.copy()
        if command == "apply":
            command = "manual"
        self._run(command)

    def quick_run(self) -> None:
        self._run("run")

    def launch_config(self) -> None:
        if self.dashboard is not None:
            self.config = self.dashboard.config.copy()
        dialog = ConfigurationDialog(self._resolved_config(), parent=self.parent)
        dialog.request_calibration.connect(self.launch_py4d_calibration)
        if dialog.exec_() == dialog.Accepted:
            self.config = dialog.config.copy()
            self.job_state = FastAcbfJobState()
            if self.dashboard is not None:
                self.dashboard.set_config(self.config)
            self._status("fast-acbf configuration updated.")

    def launch_py4d_calibration(self) -> None:
        if getattr(self.parent, "datacube", None) is None:
            QMessageBox.warning(self.parent, "Calibration", "Load a datacube before editing calibration.")
            return
        try:
            from py4d_browser_plugin.calibration_plugin.calibration_plugin import CalibrateDialog

            detector_info = self.parent.get_diffraction_detector()
            selector_size = None
            try:
                if detector_info["shape"].name == "CIRCLE":
                    selector_size = detector_info["geometry"]["R"]
            except Exception:
                selector_size = None
            dialog = CalibrateDialog(
                self.parent.datacube,
                parent=self.parent,
                diffraction_selector_size=selector_size,
            )
            self._prefill_calibration_dialog(dialog)
            dialog.finished.connect(lambda *_: self._refresh_calibration_display())
            dialog.destroyed.connect(lambda *_: self._refresh_calibration_display())
            self._calibration_dialog = dialog
            dialog.open()
        except Exception:
            QMessageBox.critical(self.parent, "Calibration", traceback.format_exc())

    def _prefill_calibration_dialog(self, dialog) -> None:
        datacube = self.parent.datacube
        calibration = datacube.calibration

        r_size = calibration.get_R_pixel_size()
        r_units = calibration.get_R_pixel_units()
        if r_units == "nm":
            dialog.realspace_unit_box.setCurrentText("nm")
        else:
            dialog.realspace_unit_box.setCurrentText("Å")
        dialog.realspace_pix_box.setText(f"{float(r_size):g}")
        dialog.realspace_fov_box.setText(f"{float(r_size) * datacube.R_Ny:g}")

        q_size = calibration.get_Q_pixel_size()
        q_units = calibration.get_Q_pixel_units()
        if q_units == "mrad":
            dialog.diff_unit_box.setCurrentText("mrad")
        else:
            dialog.diff_unit_box.setCurrentText("Å⁻¹")
        dialog.diff_pix_box.setText(f"{float(q_size):g}")
        dialog.diff_fov_box.setText(f"{float(q_size) * datacube.Q_Ny:g}")
        if dialog.diffraction_selector_size is not None:
            dialog.diff_selection_box.setText(f"{float(q_size) * dialog.diffraction_selector_size:g}")

        try:
            voltage = calibration["voltage"]
        except Exception:
            voltage = ""
        if voltage != "":
            dialog.kV_input.setText(f"{float(voltage):g}")

    def _set_actions_enabled(self, enabled: bool) -> None:
        self.dashboard_action.setEnabled(enabled)
        self.quick_run_action.setEnabled(enabled)
        self.config_action.setEnabled(enabled)

    def _run(self, command: str) -> None:
        if not self._has_datacube():
            return
        if self.runner is not None and self.runner.isRunning():
            QMessageBox.information(self.parent, "fast-acbf", "A fast-acbf job is already running.")
            return

        config = self._resolved_config()
        if self.dashboard is not None:
            self.dashboard.set_config(config)
        data = self.parent.datacube.data
        self.runner = FastAcbfRunner(
            command=command,
            data=data,
            config=config,
            state=self.job_state,
            parent=self,
        )
        self.runner.message.connect(lambda msg: self._status(msg, 0))
        self.runner.finished_result.connect(self._job_finished)
        self.runner.failed.connect(self._job_failed)
        self.runner.finished.connect(lambda: self._set_actions_enabled(True))
        self._set_actions_enabled(False)
        self._status("Starting fast-acbf...")
        self.runner.start()

    def _job_finished(self, result: dict) -> None:
        self.job_state.solver = result.get("solver")
        self.job_state.signature = result.get("signature")
        self.config = result.get("config", self.config).copy()
        image = np.asarray(result["image"])
        title = f"fast-acbf {result.get('mode', self.config.mode)}"

        try:
            if self.config.output_target == "result_image" and hasattr(self.parent, "set_result_image"):
                self.parent.set_result_image(
                    image,
                    reset=True,
                    title=title,
                    pixel_size=self.parent.datacube.calibration.get_R_pixel_size(),
                    pixel_units=self.parent.datacube.calibration.get_R_pixel_units(),
                )
            else:
                self.parent.set_virtual_image(image, reset=True)
        except Exception:
            QMessageBox.critical(self.parent, "fast-acbf display error", traceback.format_exc())
            raise

        if self.dashboard is not None:
            self.dashboard.set_config(self.config)
            self.dashboard.set_result(result)
        self._status(f"{title} complete on {result.get('device', 'device')}.")

    def _job_failed(self, trace: str) -> None:
        self._set_actions_enabled(True)
        self._status("fast-acbf failed.")
        QMessageBox.critical(self.parent, "fast-acbf error", trace)
        print(trace)
