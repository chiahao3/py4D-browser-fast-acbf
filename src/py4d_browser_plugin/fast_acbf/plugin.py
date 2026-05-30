"""fast-acbf py4D-browser plugin entry point."""

from __future__ import annotations

import gc
import traceback
from typing import TYPE_CHECKING

import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QAction, QMessageBox, QWidget

from .config import FastAcbfConfig
from .dialogs import ConfigurationDialog, FastAcbfDashboard, LiveDemoDialog
from .live import DEFAULT_GUI_FRAME_INTERVAL_MS, LiveSession, create_live_session, stop_live
from .live_view import (
    LIVE_OUTPUT_NONE,
    LiveViewDock,
    LiveViewSession,
    live_output_title,
    stop_live_view,
)
from .solver_job import PreviewJob, job_from_command
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
        self.live_demo: LiveDemoDialog | None = None
        self.live_session: LiveSession | None = None
        self.live_view_session: LiveViewSession | None = None
        self.live_view_dock: LiveViewDock | None = None
        self._live_view_callback_registered = False
        self._stopping_live_view = False
        self._live_view_display_keys: dict[str, tuple[str, tuple[int, ...]]] = {}
        self._calibration_dialog = None

        self.dashboard_action = QAction("Interactive Dashboard", self)
        self.dashboard_action.triggered.connect(self.launch_dashboard)
        self.fast_acbf_menu.addAction(self.dashboard_action)

        self.live_view_action = QAction("Live View", self)
        self.live_view_action.setCheckable(True)
        self.live_view_action.toggled.connect(self._live_view_toggled)
        self.fast_acbf_menu.addAction(self.live_view_action)

        self.live_demo_action = QAction("Live Demo", self)
        self.live_demo_action.triggered.connect(self.launch_live_demo)
        self.fast_acbf_menu.addAction(self.live_demo_action)

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
        self._stop_live()
        self._stop_live_view(restore_callback=False)
        if self.runner is not None and self.runner.isRunning():
            self.runner.wait(1000)
        if self.dashboard is not None:
            self.dashboard.close()
        if self.live_demo is not None:
            self.live_demo.close()

    def _datacube_changed(self) -> None:
        if self.live_view_session is not None:
            return
        self._stop_live()
        self._release_cached_solver()

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

    def _prepare_config_for_run(self, config: FastAcbfConfig) -> FastAcbfConfig | None:
        cfg = config.copy()
        messages = cfg.coerce_upscale_method_for_mode()
        try:
            cfg.validate_upscale_settings()
        except ValueError as exc:
            QMessageBox.warning(self.parent, "Invalid fast-acbf settings", str(exc))
            return None
        if messages:
            QMessageBox.warning(self.parent, "fast-acbf settings adjusted", "\n".join(messages))
            if self.live_view_session is None:
                self._release_cached_solver()
            else:
                self.job_state = FastAcbfJobState()
        return cfg

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

    def launch_live_demo(self) -> None:
        cfg = self._resolved_config()
        if self.live_demo is None:
            self.live_demo = LiveDemoDialog(cfg, parent=self.parent)
            self.live_demo.start_requested.connect(self._live_demo_start_requested)
            self.live_demo.stop_requested.connect(self._stop_live)
            self.live_demo.config_requested.connect(self.launch_config)
        else:
            self.live_demo.set_config(cfg)
        self.live_demo.show()
        self.live_demo.raise_()

    def _dashboard_config_changed(self, config: FastAcbfConfig) -> None:
        self.config = config.copy()
        self._update_live_view_config()

    def _dashboard_run_requested(self, job) -> None:
        if self.live_view_session is not None:
            QMessageBox.information(self.parent, "fast-acbf", "Stop Live View before running a preview.")
            return
        if self.live_session is not None:
            QMessageBox.information(self.parent, "fast-acbf", "Stop live mode before running a preview.")
            return
        if self.dashboard is not None:
            self.config = self.dashboard.config.copy()
        self._run(job)

    def quick_run(self) -> None:
        self._run(PreviewJob())

    def launch_config(self) -> None:
        if self.dashboard is not None:
            self.config = self.dashboard.config.copy()
        dialog = ConfigurationDialog(self._resolved_config(), parent=self.parent)
        dialog.request_calibration.connect(self.launch_py4d_calibration)
        if dialog.exec_() == dialog.Accepted:
            self.config = dialog.config.copy()
            if self.live_view_session is None:
                self._release_cached_solver()
            else:
                self.job_state = FastAcbfJobState()
            if self.dashboard is not None:
                self.dashboard.set_config(self.config)
            if self.live_demo is not None:
                self.live_demo.set_config(self._resolved_config())
            self._update_live_view_config()
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
        self.live_view_action.setEnabled(enabled)
        self.live_demo_action.setEnabled(enabled)
        self.quick_run_action.setEnabled(enabled)
        self.config_action.setEnabled(enabled)

    def _collect_device_memory(self) -> None:
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def _release_cached_solver(self) -> None:
        self.job_state = FastAcbfJobState()
        self._collect_device_memory()

    def _run(self, job) -> None:
        if isinstance(job, str):
            job = job_from_command(job if job != "apply" else "manual")
        if not self._has_datacube():
            return
        if self.live_view_session is not None:
            QMessageBox.information(self.parent, "fast-acbf", "Stop Live View before running fast-acbf.")
            return
        if self.live_session is not None:
            QMessageBox.information(self.parent, "fast-acbf", "Stop live mode before running fast-acbf.")
            return
        if self.runner is not None and self.runner.isRunning():
            QMessageBox.information(self.parent, "fast-acbf", "A fast-acbf job is already running.")
            return

        config = self._prepare_config_for_run(self._resolved_config())
        if config is None:
            return
        self.config = config.copy()
        if self.dashboard is not None:
            self.dashboard.set_config(config)
        data = self.parent.datacube.data
        self.runner = FastAcbfRunner(
            job=job,
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
                    pixel_size=self.config.output_pixel_size_angstrom(),
                    pixel_units="A",
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

    def _live_view_toggled(self, checked: bool) -> None:
        if checked:
            self._start_live_view()
        else:
            self._stop_live_view()

    def _start_live_view(self) -> None:
        if self.live_view_session is not None:
            return
        if not hasattr(self.parent, "register_result_callback"):
            QMessageBox.warning(
                self.parent,
                "fast-acbf Live View",
                "py4D-browser 1.5.0 or newer is required for Live View callbacks.",
            )
            self._set_live_view_action_checked(False)
            return
        if self.runner is not None and self.runner.isRunning():
            QMessageBox.information(self.parent, "fast-acbf", "A fast-acbf job is already running.")
            self._set_live_view_action_checked(False)
            return
        try:
            config = self._prepare_config_for_run(self._resolved_config())
            if config is None:
                self._set_live_view_action_checked(False)
                return
            self.config = config.copy()
            session = LiveViewSession(parent=self)
            session.frame_ready.connect(self._live_view_frame_ready)
            session.started_ready.connect(self._live_view_started)
            session.error.connect(self._live_view_failed)
            session.finished.connect(self._live_view_finished)
            self.live_view_session = session
            self._live_view_display_keys = {}
            self._ensure_live_view_dock(config)
            self.parent.register_result_callback(
                "fast-acbf Live View",
                cleanup=self._live_view_cleanup_requested,
                callback_datacube_changed=self._live_view_datacube_changed,
            )
            self._live_view_callback_registered = True
            session.start()
            self._status("fast-acbf Live View active; waiting for datacube updates.", 0)
            self._submit_live_view_current_datacube()
        except Exception:
            self._set_live_view_action_checked(False)
            self._stop_live_view(restore_callback=False)
            trace = traceback.format_exc()
            QMessageBox.critical(self.parent, "fast-acbf Live View error", trace)
            print(trace)

    def _ensure_live_view_dock(self, config: FastAcbfConfig) -> None:
        if self.live_view_dock is None:
            self.live_view_dock = LiveViewDock(config, parent=self.parent)
            self.parent.addDockWidget(Qt.TopDockWidgetArea, self.live_view_dock)
        else:
            self.live_view_dock.set_config(config)
        self.live_view_dock.show()

    def _set_live_view_action_checked(self, checked: bool) -> None:
        previous = self.live_view_action.blockSignals(True)
        self.live_view_action.setChecked(checked)
        self.live_view_action.blockSignals(previous)

    def _live_view_cleanup_requested(self) -> None:
        self._live_view_callback_registered = False
        self._stop_live_view(restore_callback=False)

    def _stop_live_view(self, status: str = "Live View stopped.", restore_callback: bool = True) -> None:
        if self._stopping_live_view:
            return
        self._stopping_live_view = True
        try:
            session = self.live_view_session
            self.live_view_session = None
            if session is not None:
                stop_live_view(session)
                self._collect_device_memory()
            if self.live_view_dock is not None:
                dock = self.live_view_dock
                self.live_view_dock = None
                dock.hide()
                try:
                    self.parent.removeDockWidget(dock)
                except Exception:
                    pass
                dock.deleteLater()
            self._live_view_display_keys = {}
            self._set_live_view_action_checked(False)
            if restore_callback and self._live_view_callback_registered:
                self._live_view_callback_registered = False
                restore = getattr(self.parent, "set_internal_result_callback", None)
                if restore is not None:
                    restore()
            self._status(status)
        finally:
            self._stopping_live_view = False

    def _update_live_view_config(self) -> None:
        if self.live_view_session is None:
            return
        config = self._prepare_config_for_run(self._resolved_config())
        if config is None:
            self._stop_live_view(status="Live View stopped: invalid configuration.")
            return
        self.config = config.copy()
        if self.live_view_dock is not None:
            self.live_view_dock.set_config(config)
        self._submit_live_view_current_datacube(config)

    def _submit_live_view_current_datacube(self, config: FastAcbfConfig | None = None) -> None:
        if self.live_view_session is None:
            return
        datacube = getattr(self.parent, "datacube", None)
        data = getattr(datacube, "data", None)
        if data is None or len(getattr(data, "shape", ())) != 4:
            if self.live_view_dock is not None:
                self.live_view_dock.set_status("Live View active; waiting for 4D datacube")
            return
        cfg = config or self._prepare_config_for_run(self._resolved_config())
        if cfg is None:
            self._stop_live_view(status="Live View stopped: invalid configuration.")
            return
        self.config = cfg.copy()
        self.live_view_session.submit(data, cfg)

    def _live_view_datacube_changed(self) -> None:
        self._submit_live_view_current_datacube()

    def _live_view_started(self, device: str) -> None:
        if self.live_view_dock is not None:
            self.live_view_dock.set_status("Live View active")
        self._status(f"fast-acbf Live View worker ready ({device}).", 0)

    def _live_view_frame_ready(self, payload: dict) -> None:
        config = payload.get("config", self.config)
        metrics = payload.get("metrics", {})
        routes = payload.get("routes", {})
        outputs = payload.get("outputs", {})
        force_reset = bool(payload.get("reset", False))
        if self.live_view_dock is not None:
            self.live_view_dock.set_config(config)
            self.live_view_dock.set_metrics(metrics)
        try:
            self._display_live_view_target("virtual", routes, outputs, force_reset, config)
            self._display_live_view_target("result", routes, outputs, force_reset, config)
        except Exception:
            trace = traceback.format_exc()
            QMessageBox.critical(self.parent, "fast-acbf Live View display error", trace)
            print(trace)
            self._stop_live_view(status="Live View stopped after display error.")
            return
        self._status("fast-acbf Live View updated.", 0)

    def _display_live_view_target(
        self,
        target: str,
        routes: dict,
        outputs: dict,
        force_reset: bool,
        config: FastAcbfConfig,
    ) -> None:
        kind = routes.get(target, LIVE_OUTPUT_NONE)
        if kind == LIVE_OUTPUT_NONE:
            return
        image = outputs.get(kind)
        if image is None:
            return
        image = np.asarray(image)
        key = (str(kind), tuple(image.shape))
        reset = force_reset or self._live_view_display_keys.get(target) != key
        self._live_view_display_keys[target] = key
        if target == "virtual":
            self.parent.set_virtual_image(image, reset=reset)
            return
        self.parent.set_result_image(
            image,
            reset=reset,
            title=live_output_title(kind),
            pixel_size=config.output_pixel_size_angstrom(),
            pixel_units="A",
        )

    def _live_view_failed(self, trace: str) -> None:
        self._status("fast-acbf Live View failed.")
        QMessageBox.critical(self.parent, "fast-acbf Live View error", trace)
        print(trace)
        self._stop_live_view(status="Live View failed.")

    def _live_view_finished(self) -> None:
        if self.live_view_session is not None:
            self._stop_live_view()

    def _live_demo_start_requested(self, options: dict) -> None:
        if self.live_view_session is not None:
            QMessageBox.information(self.parent, "fast-acbf", "Stop Live View before starting Live Demo.")
            if self.live_demo is not None:
                self.live_demo.set_live_active(False, "Stop Live View before starting Live Demo.")
            return
        if self.live_session is not None:
            return
        if not self._has_datacube():
            if self.live_demo is not None:
                self.live_demo.set_live_active(False, "Load a 4D datacube before starting live mode.")
            return
        if self.runner is not None and self.runner.isRunning():
            QMessageBox.information(self.parent, "fast-acbf", "A fast-acbf job is already running.")
            if self.live_demo is not None:
                self.live_demo.set_live_active(False, "A fast-acbf job is already running.")
            return

        try:
            config = self._resolved_config()
            config.mode = str(options.get("mode") or config.mode)
            config = self._prepare_config_for_run(config)
            if config is None:
                if self.live_demo is not None:
                    self.live_demo.set_live_active(False, "Invalid fast-acbf settings.")
                return
            self._release_cached_solver()
            data = self.parent.datacube.data
            session = create_live_session(
                config=config,
                datacube_data=data,
                options=options,
                frame_interval_ms=int(
                    options.get("frame_interval_ms", DEFAULT_GUI_FRAME_INTERVAL_MS)
                ),
                parent=self,
            )
        except Exception:
            trace = traceback.format_exc()
            if self.live_demo is not None:
                self.live_demo.set_live_active(False, "Live mode failed to start.")
            QMessageBox.critical(self.parent, "fast-acbf live error", trace)
            print(trace)
            return

        self.live_session = session
        session.worker.frame_ready.connect(self._live_frame_ready)
        session.worker.started_ready.connect(self._live_started)
        session.finished.connect(self._live_finished)
        session.error.connect(self._live_failed)
        if self.live_demo is not None:
            self.live_demo.set_config(config)
            self.live_demo.set_live_active(True, "Starting live mode...")
        self._status("Starting fast-acbf live mode...", 0)
        session.start()

    def _live_started(self, device: str) -> None:
        if self.live_demo is not None:
            self.live_demo.set_live_active(True, f"Live solver ready on {device}; waiting for frames...")
        self._status(f"fast-acbf live mode running on {device}.", 0)

    def _live_frame_ready(self, image: np.ndarray, metrics: dict) -> None:
        if self.live_demo is not None:
            self.live_demo.on_live_frame(image, metrics)

    def _live_failed(self, trace: str) -> None:
        self._status("fast-acbf live mode failed.")
        QMessageBox.critical(self.parent, "fast-acbf live error", trace)
        print(trace)
        self._stop_live(status="Live mode failed.")

    def _live_finished(self) -> None:
        self.live_session = None
        self._collect_device_memory()
        if self.live_demo is not None:
            self.live_demo.set_live_active(False, "Live mode stopped.")
        self._status("fast-acbf live mode stopped.")

    def _stop_live(self, status: str = "Live mode stopped.") -> None:
        session = self.live_session
        self.live_session = None
        if session is not None:
            stop_live(session)
            self._collect_device_memory()
        if self.live_demo is not None:
            self.live_demo.set_live_active(False, status)
