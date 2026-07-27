"""fast-acbf py4D-browser plugin entry point."""

from __future__ import annotations

import gc
import traceback
from typing import TYPE_CHECKING

import numpy as np
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import QAction, QMessageBox, QWidget

from .calibration import (
    is_calibration_unset,
    is_voltage_unset,
    resolved_wavelength_angstrom,
    sync_config_to_datacube_calibration,
)
from .config import FastAcbfConfig, lite_search_order
from .dialogs import (
    ConfigurationDialog,
    FastAcbfDashboard,
    LiteOrientationDialog,
    LiteSettingsDialog,
)
from .lite_dock import LiteTaskbarDock
from .live_view import (
    LIVE_OUTPUT_NONE,
    LiveViewDock,
    LiveViewSession,
    live_output_title,
    stop_live_view,
)
from .solver_job import LiteReconstructJob, OptimizeOrientationJob, RefineDefocusJob
from .worker import FastAcbfJobState, FastAcbfRunner

if TYPE_CHECKING:
    from py4D_browser import DataViewer


class FastAcbfPlugin(QWidget):
    plugin_id = "chiahao3.fast_acbf"
    display_name = "Fast tcBF/acBF"
    uses_plugin_menu = True

    def __init__(self, parent: "DataViewer", plugin_menu, **kwargs):
        super().__init__(parent=parent)
        self.parent = parent
        self.fast_acbf_menu = plugin_menu
        self.config = FastAcbfConfig()
        self.job_state = FastAcbfJobState()
        self.runner: FastAcbfRunner | None = None
        self.dashboard: FastAcbfDashboard | None = None
        self.live_view_session: LiveViewSession | None = None
        self.live_view_dock: LiveViewDock | None = None
        self.lite_dock: LiteTaskbarDock | None = None
        self.lite_orientation_dialog: LiteOrientationDialog | None = None
        self._pending_lite_acbf = False
        self._live_view_callback_registered = False
        self._stopping_live_view = False
        self._live_view_display_keys: dict[str, tuple[str, tuple[int, ...]]] = {}
        self._live_view_last_display: dict[str, tuple[str, np.ndarray, FastAcbfConfig]] = {}
        self._calibration_dialog = None

        self.lite_action = QAction("Show Simple Menu", self)
        self.lite_action.setCheckable(True)
        self.lite_action.toggled.connect(self._lite_taskbar_toggled)
        self.fast_acbf_menu.addAction(self.lite_action)

        self.dashboard_action = QAction("Advanced Dashboard", self)
        self.dashboard_action.triggered.connect(self.launch_dashboard)
        self.fast_acbf_menu.addAction(self.dashboard_action)

        self.live_view_action = QAction("Live View", self)
        self.live_view_action.setCheckable(True)
        self.live_view_action.toggled.connect(self._live_view_toggled)
        self.fast_acbf_menu.addAction(self.live_view_action)

        self.config_action = QAction("Configuration", self)
        self.config_action.triggered.connect(self.launch_config)
        self.fast_acbf_menu.addAction(self.config_action)

        signal = getattr(parent, "signal_datacube_changed", None)
        if signal is not None:
            signal.connect(self._datacube_changed)

    def close(self):
        self._stop_live_view(restore_callback=False)
        self._remove_lite_dock()
        if self.runner is not None and self.runner.isRunning():
            self.runner.wait(1000)
        if self.dashboard is not None:
            self.dashboard.close()
        if self.lite_orientation_dialog is not None:
            self.lite_orientation_dialog.close()

    def _datacube_changed(self) -> None:
        if self.live_view_session is not None:
            self._reassert_live_view_outputs()
            QTimer.singleShot(0, self._reassert_live_view_outputs)
            return
        self._release_cached_solver()

    def _status(self, message: str, timeout: int = 5000) -> None:
        try:
            self.parent.statusBar().showMessage(message, timeout)
        except Exception:
            print(message)
        if self.dashboard is not None:
            self.dashboard.set_status(message)
        if self.lite_orientation_dialog is not None:
            self.lite_orientation_dialog.set_status(message)

    def _on_lite_upscale_changed(self, upscale: float) -> None:
        self.config.upscale = upscale
        self._status(f"Upscale set to {upscale:g}")


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

    def _calibration_free_active(self) -> bool:
        """True when the Workflow taskbar should use the pixel-native calibration-free
        path: calibration-free is enabled, "read from py4D calibration" is on, and the
        datacube's own calibration is still at pixel defaults. Re-checked on every run,
        so as soon as real calibration is set (or the user disables use_calibration and
        types real values by hand), the very next run automatically falls back to the
        normal calibrated branch.
        """
        datacube = getattr(self.parent, "datacube", None)
        if datacube is None:
            return False
        return (
            bool(self.config.use_calibration)
            and bool(self.config.calibration_free)
            and is_calibration_unset(datacube)
        )

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
        if cfg.max_alpha_mrad is None:
            QMessageBox.warning(
                self.parent,
                "fast-acbf",
                "Max alpha is not set. Draw a circular BF detector selection, or set it "
                "under Configuration, before running.",
            )
            return None
        if cfg.wavelength_angstrom is None:
            if cfg.uses_acbf_reconstruction():
                # acBF's phase-based aberration correction genuinely needs a real
                # wavelength, unlike tcBF's pure shift-and-add.
                QMessageBox.warning(
                    self.parent,
                    "fast-acbf",
                    "Accelerating voltage is not set. Set it via Calibration or "
                    "Configuration before running acBF.",
                )
                return None
            # tcBF doesn't need a physically real wavelength, but the solver constructor
            # still requires a concrete number; use the same calibration-free placeholder
            # FastAcbfConfig.resolved_for uses internally.
            cfg.wavelength_angstrom = resolved_wavelength_angstrom(None)
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
        self._update_live_view_config(config=refreshed)
        self._resume_pending_lite_acbf()

    def _resume_pending_lite_acbf(self) -> None:
        """After the calibration dialog closes, auto-run a deferred Lite acBF if calibrated."""
        if not self._pending_lite_acbf:
            return
        # Clear before re-checking so the duplicate finished/destroyed callback is a no-op.
        self._pending_lite_acbf = False
        datacube = getattr(self.parent, "datacube", None)
        if datacube is None or is_calibration_unset(datacube) or is_voltage_unset(datacube):
            self._status("acBF needs calibration; run cancelled.")
            return
        QTimer.singleShot(0, lambda: self._run_lite("acBF"))

    def _sync_py4d_calibration_from_config(self, config: FastAcbfConfig) -> None:
        if not config.use_calibration:
            return
        datacube = getattr(self.parent, "datacube", None)
        if datacube is not None:
            sync_config_to_datacube_calibration(datacube, config)

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
        self._update_live_view_config(config=self.config)

    def _dashboard_run_requested(self, job) -> None:
        if self.live_view_session is not None:
            QMessageBox.information(self.parent, "fast-acbf", "Stop Live View before running a preview.")
            return
        if self.dashboard is not None:
            self.config = self.dashboard.config.copy()
        self._run(job)

    # ---- Lite taskbar

    def _lite_taskbar_toggled(self, checked: bool) -> None:
        if checked:
            self._ensure_lite_dock()
        else:
            self._remove_lite_dock()

    def _ensure_lite_dock(self) -> None:
        if self.lite_dock is None:
            self.lite_dock = LiteTaskbarDock(parent=self.parent)
            self.lite_dock.orientation_requested.connect(self.launch_lite_orientation)
            self.lite_dock.tcbf_requested.connect(lambda: self._run_lite("tcBF"))
            self.lite_dock.coarse_defocus_requested.connect(self._run_lite_coarse_defocus)
            self.lite_dock.refine_defocus_requested.connect(self._run_lite_refine_defocus)
            self.lite_dock.calibration_requested.connect(self.launch_py4d_calibration)
            self.lite_dock.acbf_requested.connect(lambda: self._run_lite("acBF"))
            self.lite_dock.settings_requested.connect(self.launch_lite_settings)
            self.lite_dock.advanced_requested.connect(self.launch_dashboard)
            self.lite_dock.upscale_changed.connect(self._on_lite_upscale_changed)
            self.lite_dock.closed.connect(lambda: self.lite_action.setChecked(False))
            # LiteTaskbarDock.toolbar is added to parent via its own constructor.
        self.lite_dock.show()

    def _remove_lite_dock(self) -> None:
        if self.lite_dock is None:
            return
        dock = self.lite_dock
        self.lite_dock = None
        dock.hide()
        try:
            self.parent.removeToolBar(dock.toolbar)
        except Exception:
            pass
        dock.deleteLater()

    def _run_lite_coarse_defocus(self) -> None:
        if not self._has_datacube():
            return
        if self.live_view_session is not None:
            QMessageBox.information(self.parent, "fast-acbf", "Stop Live View before running fast-acbf.")
            return

        cfg = self.config.copy()
        cfg.mode = "tcBF"
        cfg.refinement_mode = "tcBF"
        cfg.output_target = cfg.lite_output_target
        self.config = cfg

        job = RefineDefocusJob()
        self._run(job)

    def _run_lite_refine_defocus(self) -> None:
        if not self._has_datacube():
            return
        if self.live_view_session is not None:
            QMessageBox.information(self.parent, "fast-acbf", "Stop Live View before running fast-acbf.")
            return

        cfg = self.config.copy()
        cfg.mode = "tcBF"
        cfg.refinement_mode = "tcBF"
        cfg.output_target = cfg.lite_output_target
        cfg.defocus_method = "brent"
        self.config = cfg

        job = RefineDefocusJob()
        self._run(job)

    def _run_lite(self, mode: str) -> None:
        if not self._has_datacube():
            return
        if self.live_view_session is not None:
            QMessageBox.information(self.parent, "fast-acbf", "Stop Live View before running fast-acbf.")
            return
        # acBF's phase-based aberration correction genuinely needs real calibration
        # (it depends on wavelength nonlinearly, unlike tcBF's pure shift-and-add), so
        # its gate checks the raw calibration state (scan step, dk, and voltage) regardless
        # of calibration_free.
        if (
            mode == "acBF"
            and bool(self.config.use_calibration)
            and (is_calibration_unset(self.parent.datacube) or is_voltage_unset(self.parent.datacube))
        ):
            # acBF needs real calibration; prompt first and auto-run once it is saved.
            self._pending_lite_acbf = True
            self._status("acBF needs calibration; opening calibration...")
            self.launch_py4d_calibration()
            return

        level = self.config.lite_aberration_search
        order = lite_search_order(level)
        cfg = self.config.copy()
        cfg.mode = mode
        cfg.refinement_mode = "tcBF"
        cfg.output_target = cfg.lite_output_target
        if order > int(cfg.max_order):
            cfg.max_order = order
        self.config = cfg
        pixel_mode = mode == "tcBF" and self._calibration_free_active()
        job = LiteReconstructJob(
            aberration_search=level,
            pixel_mode=pixel_mode,
        )
        self._run(job)

    def launch_config(self) -> None:
        if self.dashboard is not None:
            self.config = self.dashboard.config.copy()
        dialog = ConfigurationDialog(self._resolved_config(), parent=self.parent)
        dialog.request_calibration.connect(self.launch_py4d_calibration)
        if dialog.exec_() == dialog.Accepted:
            self.config = dialog.config.copy()
            self._sync_py4d_calibration_from_config(self.config)
            if self.config.use_calibration:
                self.config = self._resolved_config()
            if self.live_view_session is None:
                self._release_cached_solver()
            else:
                self.job_state = FastAcbfJobState()
            if self.dashboard is not None:
                self.dashboard.set_config(self.config)
            if self.lite_orientation_dialog is not None:
                self.lite_orientation_dialog.set_config(self.config)
            self._update_live_view_config(config=self.config)
            self._status("fast-acbf configuration updated.")

    def launch_lite_orientation(self) -> None:
        cfg = self._resolved_config()
        if self.lite_orientation_dialog is None:
            self.lite_orientation_dialog = LiteOrientationDialog(cfg, parent=self.parent)
            self.lite_orientation_dialog.run_requested.connect(self._lite_orientation_run_requested)
            self.lite_orientation_dialog.config_changed.connect(self._dashboard_config_changed)
        else:
            self.lite_orientation_dialog.set_config(cfg)
        self.lite_orientation_dialog.show()
        self.lite_orientation_dialog.raise_()

    def _lite_orientation_run_requested(self, job) -> None:
        if self.live_view_session is not None:
            QMessageBox.information(self.parent, "fast-acbf", "Stop Live View before running a preview.")
            return
        if self.lite_orientation_dialog is not None:
            self.config = self.lite_orientation_dialog.config.copy()
        if isinstance(job, OptimizeOrientationJob):
            job = OptimizeOrientationJob(
                pixel_mode=self._calibration_free_active(),
            )
        self._run(job)

    def launch_lite_settings(self) -> None:
        dialog = LiteSettingsDialog(self._resolved_config(), parent=self.parent)
        if dialog.exec_() == dialog.Accepted:
            self.config = dialog.config.copy()
            if self.live_view_session is None:
                self._release_cached_solver()
            else:
                self.job_state = FastAcbfJobState()
            if self.dashboard is not None:
                self.dashboard.set_config(self.config)
            if self.lite_orientation_dialog is not None:
                self.lite_orientation_dialog.set_config(self.config)
            self._update_live_view_config(config=self.config)
            self._status("fast-acbf Lite settings updated.")

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
            dialog.finished.connect(lambda *_: self._refresh_calibration_display())
            dialog.destroyed.connect(lambda *_: self._refresh_calibration_display())
            self._calibration_dialog = dialog
            dialog.open()
        except Exception:
            QMessageBox.critical(self.parent, "Calibration", traceback.format_exc())

    def _set_actions_enabled(self, enabled: bool) -> None:
        self.lite_action.setEnabled(enabled)
        self.dashboard_action.setEnabled(enabled)
        self.live_view_action.setEnabled(enabled)
        self.config_action.setEnabled(enabled)
        if self.lite_dock is not None:
            self.lite_dock.set_enabled(enabled)

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
        if not self._has_datacube():
            return
        if self.live_view_session is not None:
            QMessageBox.information(self.parent, "fast-acbf", "Stop Live View before running fast-acbf.")
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
        if self.lite_orientation_dialog is not None:
            self.lite_orientation_dialog.set_config(config)
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
                self.parent.set_virtual_image(
                    image,
                    reset=True,
                    pixel_size=self.config.output_pixel_size_angstrom(),
                    pixel_units="A",
                )
        except Exception:
            QMessageBox.critical(self.parent, "fast-acbf display error", traceback.format_exc())
            raise

        if self.dashboard is not None:
            self.dashboard.set_config(self.config)
            self.dashboard.set_result(result)
        if self.lite_orientation_dialog is not None:
            self.lite_orientation_dialog.set_config(self.config)
        # self._status(f"{title} complete on {result.get('device', 'device')}.")

    def _job_failed(self, trace: str) -> None:
        self._set_actions_enabled(True)
        self._status("fast-acbf failed.")
        QMessageBox.critical(self.parent, "fast-acbf error", trace)
        print(trace)

    def _live_view_toggled(self, checked: bool) -> None:
        if checked:
            self._ensure_live_view_dock(self._resolved_config())
        else:
            self._stop_live_view(remove_dock=True)

    def _start_live_view(self) -> None:
        if self.live_view_session is not None:
            if self.live_view_dock is not None:
                self.live_view_dock.set_active(True)
            return
        if not hasattr(self.parent, "register_result_callback"):
            if self.live_view_dock is not None:
                self.live_view_dock.set_status("py4D-browser 1.5.0 or newer required")
            QMessageBox.warning(
                self.parent,
                "fast-acbf Live View",
                "py4D-browser 1.5.0 or newer is required for Live View callbacks.",
            )
            return
        if self.runner is not None and self.runner.isRunning():
            if self.live_view_dock is not None:
                self.live_view_dock.set_status("fast-acbf job already running")
            QMessageBox.information(self.parent, "fast-acbf", "A fast-acbf job is already running.")
            return
        try:
            config = self._prepare_config_for_run(self._resolved_config())
            if config is None:
                if self.live_view_dock is not None:
                    self.live_view_dock.set_status("Invalid Live View configuration")
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
            self._set_live_view_action_checked(True)
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
            self._stop_live_view(restore_callback=False, remove_dock=True)
            trace = traceback.format_exc()
            QMessageBox.critical(self.parent, "fast-acbf Live View error", trace)
            print(trace)

    def _ensure_live_view_dock(self, config: FastAcbfConfig) -> None:
        if self.live_view_dock is None:
            self.live_view_dock = LiveViewDock(
                config,
                start_callback=self._start_live_view,
                stop_callback=lambda: self._stop_live_view(remove_dock=False),
                configure_callback=self.launch_config,
                parent=self.parent,
            )
            self.live_view_dock.auto_refinement_changed.connect(
                self._live_view_auto_refinement_changed
            )
            self.live_view_dock.closed.connect(lambda: self.live_view_action.setChecked(False))
            self.parent.addDockWidget(Qt.TopDockWidgetArea, self.live_view_dock)
        else:
            self.live_view_dock.set_config(config)
        self.live_view_dock.set_active(self.live_view_session is not None)
        self.live_view_dock.show()

    def _set_live_view_action_checked(self, checked: bool) -> None:
        previous = self.live_view_action.blockSignals(True)
        self.live_view_action.setChecked(checked)
        self.live_view_action.blockSignals(previous)

    def _live_view_cleanup_requested(self) -> None:
        self._live_view_callback_registered = False
        self._stop_live_view(restore_callback=False, remove_dock=False)

    def _stop_live_view(
        self,
        status: str = "Live View stopped.",
        restore_callback: bool = True,
        *,
        remove_dock: bool = True,
    ) -> None:
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
                self.live_view_dock.set_active(False)
                self.live_view_dock.set_status("Live View stopped")
            if remove_dock:
                self._remove_live_view_dock()
            self._live_view_display_keys = {}
            self._live_view_last_display = {}
            if remove_dock:
                self._set_live_view_action_checked(False)
            if restore_callback and self._live_view_callback_registered:
                self._live_view_callback_registered = False
                restore = getattr(self.parent, "set_internal_result_callback", None)
                if restore is not None:
                    restore()
            self._status(status)
        finally:
            self._stopping_live_view = False

    def _remove_live_view_dock(self) -> None:
        if self.live_view_dock is None:
            return
        dock = self.live_view_dock
        self.live_view_dock = None
        dock.hide()
        try:
            self.parent.removeToolBar(dock.toolbar)
        except Exception:
            pass
        dock.deleteLater()

    def _update_live_view_config(self, config: FastAcbfConfig | None = None) -> None:
        if self.live_view_session is None:
            return
        resolved = config.copy() if config is not None else self._resolved_config()
        prepared = self._prepare_config_for_run(resolved)
        if prepared is None:
            self._stop_live_view(status="Live View stopped: invalid configuration.")
            return
        self.config = prepared.copy()
        if self.live_view_dock is not None:
            self.live_view_dock.set_config(prepared)
        self._sync_live_view_display_cache_to_config(prepared)
        self._submit_live_view_current_datacube(prepared)

    def _live_view_auto_refinement_changed(self, focus: bool, aberrations: bool) -> None:
        if self.live_view_session is None:
            return
        setter = getattr(self.live_view_session, "set_auto_refinement", None)
        if setter is not None:
            setter(focus=bool(focus), aberrations=bool(aberrations))

    def _sync_live_view_auto_refinement_to_worker(self) -> None:
        if self.live_view_session is None or self.live_view_dock is None:
            return
        state = getattr(self.live_view_dock, "auto_refinement_state", None)
        setter = getattr(self.live_view_session, "set_auto_refinement", None)
        if state is None or setter is None:
            return
        focus, aberrations = state()
        setter(focus=focus, aberrations=aberrations)

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
        self._sync_live_view_auto_refinement_to_worker()
        self.live_view_session.submit(data, cfg)

    def _live_view_datacube_changed(self) -> None:
        self._reassert_live_view_outputs()
        self._submit_live_view_current_datacube()
        QTimer.singleShot(0, self._reassert_live_view_outputs)

    def _live_view_started(self, device: str) -> None:
        if self.live_view_dock is not None:
            self.live_view_dock.set_status("Live View active")
            self.live_view_dock.set_active(True)
        self._status(f"fast-acbf Live View worker ready ({device}).", 0)

    def _live_view_frame_ready(self, payload: dict) -> None:
        config = payload.get("config", self.config)
        self.config = config.copy()
        metrics = payload.get("metrics", {})
        routes = payload.get("routes", {})
        outputs = payload.get("outputs", {})
        force_reset = bool(payload.get("reset", False))
        if self.live_view_dock is not None:
            self.live_view_dock.set_config(config)
            self.live_view_dock.set_active(True)
            self.live_view_dock.set_metrics(metrics)
            self._update_live_view_auto_refinement_from_metrics(metrics)
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
        if self.dashboard is not None:
            self.dashboard.set_config(self.config)

    def _update_live_view_auto_refinement_from_metrics(self, metrics: dict) -> None:
        messages = list(metrics.get("auto_refinement_messages") or [])
        if messages and self.live_view_dock is not None:
            self.live_view_dock.set_status(str(messages[-1]))
        if self.live_view_dock is None:
            return
        if "auto_focus_error" not in metrics and "auto_aberrations_error" not in metrics:
            return
        focus, aberrations = self.live_view_dock.auto_refinement_state()
        if "auto_focus_error" in metrics:
            focus = False
        if "auto_aberrations_error" in metrics:
            aberrations = False
        self.live_view_dock.set_auto_refinement_state(focus=focus, aberrations=aberrations)

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
            self._live_view_last_display.pop(target, None)
            return
        image = outputs.get(kind)
        if image is None:
            return
        image = np.asarray(image)
        key = (str(kind), tuple(image.shape))
        reset = force_reset or self._live_view_display_keys.get(target) != key
        self._live_view_display_keys[target] = key
        if target == "virtual":
            self.parent.set_virtual_image(
                image,
                reset=reset,
                pixel_size=config.output_pixel_size_angstrom(),
                pixel_units="A",
            )
            self._live_view_last_display[target] = (str(kind), image.copy(), config.copy())
            return
        self._set_result_scaling_linear()
        self.parent.set_result_image(
            image,
            reset=reset,
            title=live_output_title(kind),
            pixel_size=config.output_pixel_size_angstrom(),
            pixel_units="A",
        )
        self._live_view_last_display[target] = (str(kind), image.copy(), config.copy())

    def _reassert_live_view_outputs(self) -> None:
        if self.live_view_session is None:
            return
        for target in ("virtual", "result"):
            cached = self._live_view_last_display.get(target)
            if cached is None:
                continue
            kind, image, config = cached
            configured = self.config.live_virtual_output if target == "virtual" else self.config.live_result_output
            if str(configured) == LIVE_OUTPUT_NONE or str(configured) != kind:
                continue
            if target == "virtual":
                self.parent.set_virtual_image(
                    np.asarray(image),
                    reset=False,
                    pixel_size=config.output_pixel_size_angstrom(),
                    pixel_units="A",
                )
                continue
            self._set_result_scaling_linear()
            self.parent.set_result_image(
                np.asarray(image),
                reset=False,
                title=live_output_title(kind),
                pixel_size=config.output_pixel_size_angstrom(),
                pixel_units="A",
            )

    def _sync_live_view_display_cache_to_config(self, config: FastAcbfConfig) -> None:
        for target, output in (
            ("virtual", config.live_virtual_output),
            ("result", config.live_result_output),
        ):
            cached = self._live_view_last_display.get(target)
            if cached is None:
                continue
            if str(output) == LIVE_OUTPUT_NONE or str(output) != cached[0]:
                self._live_view_last_display.pop(target, None)

    def _set_result_scaling_linear(self) -> None:
        action = getattr(self.parent, "result_scale_linear_action", None)
        if action is not None:
            action.setChecked(True)
            return
        group = getattr(self.parent, "result_scaling_group", None)
        actions = getattr(group, "actions", lambda: [])()
        for candidate in actions:
            if candidate.text().replace("&", "") == "Linear":
                candidate.setChecked(True)
                return

    def _live_view_failed(self, trace: str) -> None:
        self._status("fast-acbf Live View failed.")
        QMessageBox.critical(self.parent, "fast-acbf Live View error", trace)
        print(trace)
        self._stop_live_view(status="Live View failed.", remove_dock=False)

    def _live_view_finished(self) -> None:
        if self.live_view_session is not None:
            self._stop_live_view()
