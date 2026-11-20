"""fast-acbf py4D-browser plugin entry point."""

from __future__ import annotations

import gc
import traceback
from typing import TYPE_CHECKING

import numpy as np
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import QAction, QMessageBox, QWidget
from PyQt5.QtGui import QKeySequence

from .calibration import (
    is_calibration_unset,
    is_voltage_unset,
    resolved_wavelength_angstrom,
    sync_config_to_datacube_calibration,
)
from .config import FastAcbfConfig, labels_for_order, simple_menu_search_order
from .dialogs import (
    ConfigurationDialog,
    FastAcbfDashboard,
    SimpleMenuOrientationDialog,
    SimpleMenuVolumeDialog,
)
from .simple_menu_toolbar import SimpleMenuToolbar
from .live_view import (
    LIVE_OUTPUT_NONE,
    LiveViewDock,
    LiveViewSession,
    stop_live_view,
)
from .solver_job import (
    OptimizeOrientationJob,
    PreviewJob,
    ReconstructVolumeJob,
    RefineDefocusJob,
    SimpleMenuReconstructJob,
)
from .worker import FastAcbfJobState, FastAcbfRunner

if TYPE_CHECKING:
    from py4D_browser import DataViewer


# acBF/tcBF reconstructions have strongly anisotropic resolution: the depth
# (defocus) axis is far coarser than the in-plane axes. Showing the volume at
# its physically correct anisotropic scale renders it very stretched, so by
# convention we display it with a fixed, compressed scale instead.
VOLUME_VOXEL_SIZE = (2.0, 1.0, 1.0)  # (depth, y, x)


class FastAcbfPlugin(QWidget):
    plugin_id = "chiahao3.fast_acbf"
    display_name = "Fast tcBF/acBF"
    uses_plugin_menu = True
    # Versioned plugin API (py4D-browser v1.6): the loader resolves this and
    # passes a matching PluginAPI broker as `api`. `full_access` additionally
    # hands us the raw DataViewer as `parent` for the Qt-window-only
    # operations the broker doesn't cover (toolbars, dock widgets).
    api_version = (1, 6)
    full_access = True

    def __init__(self, api, plugin_menu=None, plugin_action=None, parent=None, **kwargs):
        super().__init__(parent=parent)
        self.api = api            # PluginAPI broker — preferred access path
        self.parent = parent      # raw DataViewer (full_access) — toolbar/dock escape hatch
        self.fast_acbf_menu = plugin_menu
        self.config = FastAcbfConfig()
        self.job_state = FastAcbfJobState()
        self.runner: FastAcbfRunner | None = None
        self.dashboard: FastAcbfDashboard | None = None
        self.live_view_session: LiveViewSession | None = None
        self.live_view_dock: LiveViewDock | None = None
        self.simple_menu_toolbar: SimpleMenuToolbar | None = None
        self.simple_menu_orientation_dialog: SimpleMenuOrientationDialog | None = None
        self._pending_simple_menu_acbf = False
        self._pending_volume_mode: str | None = None
        self._stopping_live_view = False
        self._recon_image_tab = None
        self._recon_probe_tab = None
        self._live_view_tab = None
        self._opened_tabs: list = []
        self._live_view_display_key: tuple[str, tuple[int, ...]] | None = None
        self._calibration_dialog = None
        self._dataset_generation = 0

        self.simple_menu_action = QAction("Show Simple Menu", self)
        self.simple_menu_action.setCheckable(True)
        self.simple_menu_action.toggled.connect(self._simple_menu_toolbar_toggled)
        self.simple_menu_action.setShortcut(QKeySequence("Ctrl+Shift+T"))
        self.fast_acbf_menu.addAction(self.simple_menu_action)

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

        self.api.signal_datacube_changed.connect(self._datacube_changed)

    def close(self):
        # v1.6 unload lifecycle hook: the host calls this when the window
        # closes; release the runner, toolbar, dock, and dialogs.
        self._stop_live_view()
        self._remove_simple_menu_toolbar()
        if self.runner is not None and self.runner.isRunning():
            self.runner.wait(1000)
        if self.dashboard is not None:
            self.dashboard.close()
        if self.simple_menu_orientation_dialog is not None:
            self.simple_menu_orientation_dialog.close()
        self._close_all_plugin_tabs()

    def _datacube_changed(self) -> None:
        if self.live_view_session is not None:
            # a Live View session follows the new datacube instead of being reset by it
            self._submit_live_view_current_datacube()
            return
        self._dataset_generation += 1
        self._pending_simple_menu_acbf = False
        self._pending_volume_mode = None
        self._reset_diffraction_detector_selection()
        self.config = FastAcbfConfig()
        self._release_cached_solver()
        self._reset_offline_ui()
        self._close_all_plugin_tabs()
        self._status("New datacube loaded; fast-acbf state reset.")

    def _reset_diffraction_detector_selection(self) -> None:
        """Replace any area detector ROI with py4D's neutral Point selector."""
        group = getattr(self.parent, "detector_shape_group", None)
        actions = getattr(group, "actions", None)
        if actions is None:
            return
        point_action = next(
            (
                action
                for action in actions()
                if str(action.text()).replace("&", "").strip().lower() == "point"
            ),
            None,
        )
        if point_action is None:
            return
        point_action.setChecked(True)
        update = getattr(self.parent, "update_diffraction_detector", None)
        if update is not None:
            update()

    def _reset_offline_ui(self) -> None:
        if self.dashboard is not None:
            self.dashboard.reset_for_new_dataset(self.config)
        if self.simple_menu_orientation_dialog is not None:
            self.simple_menu_orientation_dialog.set_config(self.config)
        if self.simple_menu_toolbar is not None:
            self.simple_menu_toolbar.reset_for_new_dataset(self.config)
        if self.live_view_dock is not None:
            self.live_view_dock.set_config(self.config)

    def _status(self, message: str, timeout: int = 5000) -> None:
        try:
            self.api.status_bar.showMessage(message, timeout)
        except Exception:
            print(message)
        if self.dashboard is not None:
            self.dashboard.set_status(message)
        if self.simple_menu_orientation_dialog is not None:
            self.simple_menu_orientation_dialog.set_status(message)

    def _on_simple_menu_upscale_changed(self, upscale: float) -> None:
        self.config.upscale = upscale
        self._status(f"Upscale set to {upscale:g}")

    def _run_set_c10(self, value: float) -> None:
        """Set C10 to an absolute value and update the current reconstruction."""
        if not self._has_datacube():
            return
        if self.job_state.solver is None:
            self._status("Run a reconstruction first before adjusting defocus.", 0)
            return
        if self.live_view_session is not None:
            QMessageBox.information(self.api.qt_window, "fast-acbf", "Stop Live View before running fast-acbf.")
            return
        if self.runner is not None and self.runner.isRunning():
            QMessageBox.information(self.api.qt_window, "fast-acbf", "A fast-acbf job is already running.")
            return

        cfg = self.config.copy()
        cfg.aberrations["C10"] = float(value)
        self.config = cfg

        self._status(f"C10 set to {value:.2f} Å")
        self._run(PreviewJob())

    def _run_reset_aberrations(self) -> None:
        """Zero all active and already-known aberrations, then update the preview."""
        if not self._has_datacube():
            return
        if self.live_view_session is not None:
            QMessageBox.information(
                self.api.qt_window,
                "fast-acbf",
                "Stop Live View before running fast-acbf.",
            )
            return
        if self.runner is not None and self.runner.isRunning():
            QMessageBox.information(
                self.api.qt_window,
                "fast-acbf",
                "A fast-acbf job is already running.",
            )
            return

        cfg = self.config.copy()
        labels = set(cfg.aberrations)
        labels.update(labels_for_order(cfg.max_order))
        cfg.aberrations = {label: 0.0 for label in labels}
        self.config = cfg

        if self.simple_menu_toolbar is not None:
            self.simple_menu_toolbar.set_c10(0.0)
        if self.dashboard is not None:
            self.dashboard.set_config(cfg)
        if self.simple_menu_orientation_dialog is not None:
            self.simple_menu_orientation_dialog.set_config(cfg)
        self._status("Aberrations reset to zero.")
        self._run(PreviewJob())


    def _has_datacube(self) -> bool:
        if self.api.datacube is None:
            QMessageBox.warning(self.api.qt_window, "fast-acbf", "Load a 4D datacube before running fast-acbf.")
            return False
        data = getattr(self.api.datacube, "data", None)
        if data is None or len(getattr(data, "shape", ())) != 4:
            QMessageBox.warning(self.api.qt_window, "fast-acbf", "fast-acbf requires a 4D datacube.")
            return False
        return True

    def _resolved_config(self) -> FastAcbfConfig:
        return self.config.resolved_for(self.api)

    def _calibration_free_active(self) -> bool:
        """True when the Simple Menu should use the pixel-native calibration-free
        path: calibration-free is enabled, "read from py4D calibration" is on, and the
        datacube's own calibration is still at pixel defaults. Re-checked on every run,
        so as soon as real calibration is set (or the user disables use_calibration and
        types real values by hand), the very next run automatically falls back to the
        normal calibrated branch.
        """
        datacube = self.api.datacube
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
            QMessageBox.warning(self.api.qt_window, "Invalid fast-acbf settings", str(exc))
            return None
        if messages:
            QMessageBox.warning(self.api.qt_window, "fast-acbf settings adjusted", "\n".join(messages))
            if self.live_view_session is None:
                self._release_cached_solver()
            else:
                self.job_state = FastAcbfJobState()
        if cfg.max_alpha_mrad is None:
            QMessageBox.warning(
                self.api.qt_window,
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
                    self.api.qt_window,
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
        self._resume_pending_simple_menu_acbf()
        self._resume_pending_volume_mode()

    def _resume_pending_simple_menu_acbf(self) -> None:
        """After the calibration dialog closes, auto-run a deferred Simple Menu acBF if calibrated."""
        if not self._pending_simple_menu_acbf:
            return
        # Clear before re-checking so the duplicate finished/destroyed callback is a no-op.
        self._pending_simple_menu_acbf = False
        datacube = self.api.datacube
        if datacube is None or is_calibration_unset(datacube) or is_voltage_unset(datacube):
            self._status("acBF needs calibration; run cancelled.")
            return
        QTimer.singleShot(0, lambda: self._run_simple_menu("acBF"))

    def _resume_pending_volume_mode(self) -> None:
        """After the calibration dialog closes, auto-run a deferred volume reconstruction."""
        if self._pending_volume_mode is None:
            return
        mode = self._pending_volume_mode
        self._pending_volume_mode = None
        datacube = self.api.datacube
        if datacube is None or is_calibration_unset(datacube) or is_voltage_unset(datacube):
            self._status("acBF needs calibration; run cancelled.")
            return
        QTimer.singleShot(0, lambda: self._run_reconstruct_volume(mode))

    def _sync_py4d_calibration_from_config(self, config: FastAcbfConfig) -> None:
        if not config.use_calibration:
            return
        datacube = self.api.datacube
        if datacube is not None:
            sync_config_to_datacube_calibration(datacube, config)

    def launch_dashboard(self) -> None:
        cfg = self._resolved_config()
        if self.dashboard is None:
            self.dashboard = FastAcbfDashboard(cfg, parent=self.api.qt_window)
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
        self._sync_py4d_calibration_from_config(self.config)
        self._update_live_view_config(config=self.config)

    def _dashboard_run_requested(self, job) -> None:
        if self.live_view_session is not None:
            QMessageBox.information(self.api.qt_window, "fast-acbf", "Stop Live View before running a preview.")
            return
        if self.dashboard is not None:
            self.config = self.dashboard.config.copy()
        self._run(job)

    # ---- Simple Menu

    def _simple_menu_toolbar_toggled(self, checked: bool) -> None:
        if checked:
            self._ensure_simple_menu_toolbar()
        else:
            self._remove_simple_menu_toolbar()

    def _ensure_simple_menu_toolbar(self) -> None:
        if self.simple_menu_toolbar is None:
            self.simple_menu_toolbar = SimpleMenuToolbar(
                parent=self.parent,
                upscale=self.config.upscale,
            )
            self.simple_menu_toolbar.orientation_requested.connect(
                self.launch_simple_menu_orientation
            )
            self.simple_menu_toolbar.tcbf_requested.connect(lambda: self._run_simple_menu("tcBF"))
            self.simple_menu_toolbar.coarse_defocus_requested.connect(
                self._run_simple_menu_coarse_defocus
            )
            self.simple_menu_toolbar.refine_defocus_requested.connect(
                self._run_simple_menu_refine_defocus
            )
            self.simple_menu_toolbar.calibration_requested.connect(self.launch_py4d_calibration)
            self.simple_menu_toolbar.acbf_requested.connect(
                lambda: self._run_simple_menu("acBF")
            )
            self.simple_menu_toolbar.advanced_requested.connect(self.launch_dashboard)
            self.simple_menu_toolbar.upscale_changed.connect(
                self._on_simple_menu_upscale_changed
            )
            self.simple_menu_toolbar.c10_value_requested.connect(self._run_set_c10)
            self.simple_menu_toolbar.reset_aberrations_requested.connect(
                self._run_reset_aberrations
            )
            self.simple_menu_toolbar.volume_requested.connect(self._run_reconstruct_volume)
            self.simple_menu_toolbar.set_c10(self.config.aberrations.get("C10", 0.0))
            self.simple_menu_toolbar.visibilityChanged.connect(
                self._simple_menu_toolbar_visibility_changed
            )
            self.parent.addToolBar(self.simple_menu_toolbar)
        self.simple_menu_toolbar.show()

    def _simple_menu_toolbar_visibility_changed(self, visible: bool) -> None:
        if not visible and self.simple_menu_action.isChecked():
            self.simple_menu_action.setChecked(False)

    def _remove_simple_menu_toolbar(self) -> None:
        if self.simple_menu_toolbar is None:
            return
        toolbar = self.simple_menu_toolbar
        self.simple_menu_toolbar = None
        toolbar.hide()
        self.parent.removeToolBar(toolbar)
        toolbar.deleteLater()

    def _run_simple_menu_coarse_defocus(self) -> None:
        if not self._has_datacube():
            return
        if self.live_view_session is not None:
            QMessageBox.information(self.api.qt_window, "fast-acbf", "Stop Live View before running fast-acbf.")
            return

        cfg = self.config.copy()
        cfg.mode = "tcBF"
        cfg.refinement_mode = "tcBF"
        cfg.defocus_method = "max"
        self.config = cfg

        job = RefineDefocusJob()
        self._run(job)

    def _run_simple_menu_refine_defocus(self) -> None:
        if not self._has_datacube():
            return
        if self.live_view_session is not None:
            QMessageBox.information(self.api.qt_window, "fast-acbf", "Stop Live View before running fast-acbf.")
            return

        cfg = self.config.copy()
        cfg.mode = "tcBF"
        cfg.refinement_mode = "tcBF"
        cfg.defocus_method = "brent"
        self.config = cfg

        job = RefineDefocusJob()
        self._run(job)

    def _run_simple_menu(self, mode: str) -> None:
        if not self._has_datacube():
            return
        if self.live_view_session is not None:
            QMessageBox.information(self.api.qt_window, "fast-acbf", "Stop Live View before running fast-acbf.")
            return
        # acBF's phase-based aberration correction genuinely needs real calibration
        # (it depends on wavelength nonlinearly, unlike tcBF's pure shift-and-add), so
        # its gate checks the raw calibration state (scan step, dk, and voltage) regardless
        # of calibration_free.
        if (
            mode == "acBF"
            and bool(self.config.use_calibration)
            and (is_calibration_unset(self.api.datacube) or is_voltage_unset(self.api.datacube))
        ):
            # acBF needs real calibration; prompt first and auto-run once it is saved.
            self._pending_simple_menu_acbf = True
            self._status("acBF needs calibration; opening calibration...")
            self.launch_py4d_calibration()
            return

        level = self.config.simple_menu_aberration_search
        order = simple_menu_search_order(level)
        cfg = self.config.copy()
        cfg.mode = mode
        cfg.refinement_mode = "tcBF"
        if order > int(cfg.max_order):
            cfg.max_order = order
        self.config = cfg
        pixel_mode = mode == "tcBF" and self._calibration_free_active()
        job = SimpleMenuReconstructJob(
            aberration_search=level,
            pixel_mode=pixel_mode,
        )
        self._run(job)

    def _run_reconstruct_volume(self, mode: str) -> None:
        if not self._has_datacube():
            return
        if self.live_view_session is not None:
            QMessageBox.information(self.api.qt_window, "fast-acbf", "Stop Live View before running fast-acbf.")
            return
        if (
            mode == "acBF"
            and bool(self.config.use_calibration)
            and (is_calibration_unset(self.api.datacube) or is_voltage_unset(self.api.datacube))
        ):
            self._pending_volume_mode = mode
            self._status("acBF needs calibration; opening calibration...")
            self.launch_py4d_calibration()
            return
        dialog = SimpleMenuVolumeDialog(self._resolved_config(), mode, parent=self.api.qt_window)
        dialog.run_requested.connect(self._volume_run_requested)
        dialog.exec_()

    def _volume_run_requested(self, job: ReconstructVolumeJob) -> None:
        cfg = self.config.copy()
        cfg.mode = job.mode
        cfg.refinement_mode = "tcBF"
        self.config = cfg
        self._run(job)

    def launch_config(self) -> None:
        if self.dashboard is not None:
            self.config = self.dashboard.config.copy()
        dialog = ConfigurationDialog(self._resolved_config(), parent=self.api.qt_window)
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
            if self.simple_menu_orientation_dialog is not None:
                self.simple_menu_orientation_dialog.set_config(self.config)
            self._update_live_view_config(config=self.config)
            self._status("fast-acbf configuration updated.")

    def launch_simple_menu_orientation(self) -> None:
        cfg = self._resolved_config()
        if self.simple_menu_orientation_dialog is None:
            self.simple_menu_orientation_dialog = SimpleMenuOrientationDialog(cfg, parent=self.api.qt_window)
            self.simple_menu_orientation_dialog.run_requested.connect(self._simple_menu_orientation_run_requested)
            self.simple_menu_orientation_dialog.config_changed.connect(self._dashboard_config_changed)
        else:
            self.simple_menu_orientation_dialog.set_config(cfg)
        self.simple_menu_orientation_dialog.show()
        self.simple_menu_orientation_dialog.raise_()

    def _simple_menu_orientation_run_requested(self, job) -> None:
        if self.live_view_session is not None:
            QMessageBox.information(self.api.qt_window, "fast-acbf", "Stop Live View before running a preview.")
            return
        if self.simple_menu_orientation_dialog is not None:
            self.config = self.simple_menu_orientation_dialog.config.copy()
            self._sync_py4d_calibration_from_config(self.config)
        if isinstance(job, OptimizeOrientationJob):
            job = OptimizeOrientationJob(
                pixel_mode=self._calibration_free_active(),
            )
        self._run(job)

    def launch_py4d_calibration(self) -> None:
        if self.api.datacube is None:
            QMessageBox.warning(self.api.qt_window, "Calibration", "Load a datacube before editing calibration.")
            return
        try:
            from py4d_browser_plugin.calibration_plugin.calibration_plugin import CalibrateDialog

            detector_info = self.api.get_diffraction_detector()
            selector_size = None
            try:
                if detector_info["shape"].name == "CIRCLE":
                    selector_size = detector_info["geometry"]["R"]
            except Exception:
                selector_size = None
            dialog = CalibrateDialog(self.api, diffraction_selector_size=selector_size)
            dialog.finished.connect(lambda *_: self._refresh_calibration_display())
            dialog.destroyed.connect(lambda *_: self._refresh_calibration_display())
            self._calibration_dialog = dialog
            dialog.open()
        except Exception:
            QMessageBox.critical(self.api.qt_window, "Calibration", traceback.format_exc())

    def _set_actions_enabled(self, enabled: bool) -> None:
        self.simple_menu_action.setEnabled(enabled)
        self.dashboard_action.setEnabled(enabled)
        self.live_view_action.setEnabled(enabled)
        self.config_action.setEnabled(enabled)
        if self.simple_menu_toolbar is not None:
            self.simple_menu_toolbar.set_enabled(enabled)

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

    def _run(self, job, *, dataset_generation: int | None = None) -> None:
        generation = (
            self._dataset_generation if dataset_generation is None else dataset_generation
        )
        if generation != self._dataset_generation:
            return
        if not self._has_datacube():
            return
        if self.live_view_session is not None:
            QMessageBox.information(self.api.qt_window, "fast-acbf", "Stop Live View before running fast-acbf.")
            return
        if self.runner is not None and self.runner.isRunning():
            QMessageBox.information(self.api.qt_window, "fast-acbf", "A fast-acbf job is already running.")
            return

        config = self._prepare_config_for_run(self._resolved_config())
        if config is None:
            return
        self.config = config.copy()
        if self.dashboard is not None:
            self.dashboard.set_config(config)
        if self.simple_menu_orientation_dialog is not None:
            self.simple_menu_orientation_dialog.set_config(config)
        data = self.api.datacube.data
        self.runner = FastAcbfRunner(
            job=job,
            data=data,
            config=config,
            state=self.job_state,
            parent=self,
        )
        self.runner.message.connect(lambda msg: self._status(msg, 0))
        self.runner.finished_result.connect(
            lambda result: self._job_finished(result, generation)
        )
        self.runner.failed.connect(lambda trace: self._job_failed(trace, generation))
        self.runner.finished.connect(lambda: self._job_runner_finished(generation))
        self._set_actions_enabled(False)
        self._status("Starting fast-acbf...")
        self.runner.start()

    def _job_finished(
        self,
        result: dict,
        dataset_generation: int | None = None,
    ) -> None:
        if (
            dataset_generation is not None
            and dataset_generation != self._dataset_generation
        ):
            return
        # keep the solver in job_state only: the dashboard stores results, and a solver
        # left in one would keep its dataset in GPU memory after a rebuild
        solver = result.pop("solver", None)
        self.job_state.solver = solver
        self.job_state.signature = result.get("signature")
        self.config = result.get("config", self.config).copy()
        self._sync_py4d_calibration_from_config(self.config)
        if result.get("command") == "depth_stack":
            # a view of the current state, not a new one: no viewer update, no history
            if self.dashboard is not None:
                self.dashboard.set_result(result)
            return
        mode = result.get("mode", self.config.mode)
        if result.get("volume") is not None:
            self._show_volume_result(result, mode)
        else:
            self._show_image_result(result, mode)

        if self.dashboard is not None:
            self.dashboard.set_config(self.config)
            self.dashboard.set_result(result)
        if self.simple_menu_orientation_dialog is not None:
            self.simple_menu_orientation_dialog.set_config(self.config)
        if self.simple_menu_toolbar is not None:
            c10 = solver.ab_state.get_physical("C_1_0") if solver is not None else None
            self.simple_menu_toolbar.set_c10(c10)
        # self._status(f"{title} complete on {result.get('device', 'device')}.")

    def _get_or_create_recon_tab(self, slot: str, title: str, select: bool):
        """Return the plugin's image/probe tab for `slot`, honoring output_tab_mode.

        "reuse" keeps the existing tab (re-creating it only if closed) and just
        updates its title; "new" opens a fresh tab on every run. `select` only
        takes effect when a tab is actually created, so the image tab (which is
        created first with select=True) keeps the pane's focus over the probe.
        """
        if self.config.output_tab_mode == "new":
            tab = self.api.create_virtual_image_tab(title, select=select)
            setattr(self, slot, tab)
            self._opened_tabs.append(tab)
            return tab
        tab = getattr(self, slot)
        if tab is None or getattr(tab, "closed", False):
            tab = self.api.create_virtual_image_tab(title, select=select)
            setattr(self, slot, tab)
            self._opened_tabs.append(tab)
        tab.title = title
        return tab

    def _show_image_result(self, result: dict, mode: str) -> None:
        image = np.asarray(result["image"])
        probe = np.asarray(result["probe"])
        pixel_size = self.config.output_pixel_size_angstrom()
        try:
            image_tab = self._get_or_create_recon_tab("_recon_image_tab", f"fast-acbf {mode}", select=True)
            image_tab.set_image(image, reset=True, pixel_size=pixel_size, pixel_units="Å")
            probe_tab = self._get_or_create_recon_tab("_recon_probe_tab", f"fast-acbf {mode} probe", select=False)
            # the probe lives on its own 1/(N·dk) grid, not the reconstruction's scan-step grid
            probe_pixel_size = result.get("probe_pixel_size") or pixel_size
            probe_tab.set_image(probe, reset=True, pixel_size=probe_pixel_size, pixel_units="Å")
        except Exception:
            QMessageBox.critical(self.api.qt_window, "fast-acbf display error", traceback.format_exc())
            raise

    def _show_volume_result(self, result: dict, mode: str) -> None:
        volume = np.asarray(result["volume"])
        try:
            tab = self.api.create_volume_tab(f"fast-acbf {mode} volume", select=True)
        except RuntimeError:
            QMessageBox.critical(
                self.api.qt_window,
                "fast-acbf volume display error",
                "PyOpenGL is not available, so volume tabs cannot be shown. "
                "Install PyOpenGL to view 3-D results.",
            )
            return
        self._opened_tabs.append(tab)
        try:
            # The stack is (Nz, Ny, Nx); display it with the fixed, compressed
            # scale in VOLUME_VOXEL_SIZE rather than the physically correct but
            # very anisotropic voxel size.
            tab.set_volume(volume, voxel_size=VOLUME_VOXEL_SIZE)
        except Exception:
            QMessageBox.critical(self.api.qt_window, "fast-acbf display error", traceback.format_exc())
            raise

    def _job_runner_finished(self, dataset_generation: int) -> None:
        if dataset_generation == self._dataset_generation:
            self._set_actions_enabled(True)

    def _job_failed(
        self,
        trace: str,
        dataset_generation: int | None = None,
    ) -> None:
        if (
            dataset_generation is not None
            and dataset_generation != self._dataset_generation
        ):
            return
        self._set_actions_enabled(True)
        self._status("fast-acbf failed.")
        QMessageBox.critical(self.api.qt_window, "fast-acbf error", trace)
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
        if self.runner is not None and self.runner.isRunning():
            if self.live_view_dock is not None:
                self.live_view_dock.set_status("fast-acbf job already running")
            QMessageBox.information(self.api.qt_window, "fast-acbf", "A fast-acbf job is already running.")
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
            self._live_view_display_key = None
            self._ensure_live_view_tab()
            self._ensure_live_view_dock(config)
            self._set_live_view_action_checked(True)
            session.start()
            self._status("fast-acbf Live View active; waiting for datacube updates.", 0)
            self._submit_live_view_current_datacube()
        except Exception:
            self._set_live_view_action_checked(False)
            self._stop_live_view(remove_dock=True)
            trace = traceback.format_exc()
            QMessageBox.critical(self.api.qt_window, "fast-acbf Live View error", trace)
            print(trace)

    def _ensure_live_view_tab(self) -> None:
        if self._live_view_tab is None or getattr(self._live_view_tab, "closed", False):
            self._live_view_tab = self.api.create_virtual_image_tab("fast-acbf Live View", select=True)
            self._opened_tabs.append(self._live_view_tab)

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

    def _stop_live_view(
        self,
        status: str = "Live View stopped.",
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
            if self._live_view_tab is not None:
                self.api.close_virtual_image_tab(self._live_view_tab)
                self._live_view_tab = None
            self._live_view_display_key = None
            if remove_dock:
                self._set_live_view_action_checked(False)
            self._status(status)
        finally:
            self._stopping_live_view = False

    def _remove_live_view_dock(self) -> None:
        if self.live_view_dock is None:
            return
        dock = self.live_view_dock
        self.live_view_dock = None
        dock.hide()
        self.parent.removeDockWidget(dock)
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
        datacube = self.api.datacube
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

    def _live_view_started(self, device: str) -> None:
        if self.live_view_dock is not None:
            self.live_view_dock.set_status("Live View active")
            self.live_view_dock.set_active(True)
        self._status(f"fast-acbf Live View worker ready ({device}).", 0)

    def _live_view_frame_ready(self, payload: dict) -> None:
        config = payload.get("config", self.config)
        self.config = config.copy()
        metrics = payload.get("metrics", {})
        images = payload.get("images", {})
        force_reset = bool(payload.get("reset", False))
        if self.live_view_dock is not None:
            self.live_view_dock.set_config(config)
            self.live_view_dock.set_active(True)
            self.live_view_dock.set_metrics(metrics)
            self._update_live_view_auto_refinement_from_metrics(metrics)
        try:
            self._display_live_view_frame(images, config, force_reset)
        except Exception:
            trace = traceback.format_exc()
            QMessageBox.critical(self.api.qt_window, "fast-acbf Live View display error", trace)
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

    def _display_live_view_frame(self, images: dict, config: FastAcbfConfig, force_reset: bool) -> None:
        kind = str(config.live_output)
        if kind == LIVE_OUTPUT_NONE:
            self._live_view_display_key = None
            return
        image = images.get(kind)
        if image is None:
            return
        image = np.asarray(image)
        key = (kind, tuple(image.shape))
        reset = force_reset or self._live_view_display_key != key
        self._live_view_display_key = key
        self._ensure_live_view_tab()
        self._live_view_tab.set_image(
            image,
            reset=reset,
            pixel_size=config.output_pixel_size_angstrom(),
            pixel_units="Å",
        )

    def _live_view_failed(self, trace: str) -> None:
        self._status("fast-acbf Live View failed.")
        QMessageBox.critical(self.api.qt_window, "fast-acbf Live View error", trace)
        print(trace)
        self._stop_live_view(status="Live View failed.", remove_dock=False)

    def _live_view_finished(self) -> None:
        if self.live_view_session is not None:
            self._stop_live_view()

    def _close_all_plugin_tabs(self) -> None:
        for tab in self._opened_tabs:
            try:
                self.api.close_virtual_image_tab(tab)
            except Exception:
                pass
        self._opened_tabs = []
        self._recon_image_tab = None
        self._recon_probe_tab = None
        self._live_view_tab = None
