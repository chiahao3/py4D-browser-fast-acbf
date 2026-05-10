"""Qt lifecycle helpers for live fast-acbf dashboard sessions."""

from __future__ import annotations

from typing import Any

import numpy as np
from PyQt5.QtCore import QObject, QThread, pyqtSignal, pyqtSlot

from .config import FastAcbfConfig
from .live_worker import LiveSolverWorker
from .streamers import MockStreamer


def metadata_from_config(config: FastAcbfConfig, datacube_data: np.ndarray) -> dict[str, Any]:
    shape = tuple(getattr(datacube_data, "shape", ()))
    if len(shape) != 4:
        raise ValueError(f"Live mode requires a 4D datacube, got shape {shape}.")
    return {
        "wavelength": float(config.wavelength_angstrom),
        "max_alpha": float(config.max_alpha_mrad),
        "dk": float(config.dk_inv_angstrom),
        "scan_shape": tuple(shape[:2]),
        "scan_step_size": float(config.scan_step_angstrom),
        "rotation_deg": float(config.rotation_deg),
        "flipud": bool(config.flipud),
        "fliplr": bool(config.fliplr),
        "transpose": bool(config.transpose),
    }


def jitter_from_options(options: dict[str, Any] | None) -> dict[str, float]:
    options = options or {}
    jitter: dict[str, float] = {}
    rotation_sigma = float(options.get("jitter_rotation_deg") or 0.0)
    scan_step_sigma = float(options.get("jitter_scan_step_angstrom") or 0.0)
    if rotation_sigma > 0:
        jitter["rotation_deg"] = rotation_sigma
    if scan_step_sigma > 0:
        jitter["scan_step_size"] = scan_step_sigma
    return jitter


class _StreamerProducer(QObject):
    finished = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(
        self,
        streamer: MockStreamer,
        worker: LiveSolverWorker,
        *,
        frame_interval_ms: int = 0,
    ) -> None:
        super().__init__()
        self.streamer = streamer
        self.worker = worker
        self.frame_interval_ms = max(0, int(frame_interval_ms))
        self._stop = False

    def stop(self) -> None:
        self._stop = True

    @pyqtSlot()
    def run(self) -> None:
        try:
            for dataset, metadata in self.streamer:
                if self._stop:
                    break
                self.worker.submit(dataset, metadata)
                if self.frame_interval_ms > 0:
                    QThread.msleep(self.frame_interval_ms)
        except Exception as exc:
            self.error.emit(str(exc))
        finally:
            self.finished.emit()


class LiveSession(QObject):
    """Owns the worker and producer threads for one dashboard live run."""

    finished = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(
        self,
        *,
        worker: LiveSolverWorker,
        producer: _StreamerProducer,
        producer_thread: QThread,
        base_metadata: dict[str, Any],
        parent=None,
    ) -> None:
        super().__init__(parent=parent)
        self.worker = worker
        self.producer = producer
        self.producer_thread = producer_thread
        self.base_metadata = dict(base_metadata)
        self._stopping = False

        self.producer.moveToThread(self.producer_thread)
        self.producer_thread.started.connect(self.producer.run)
        self.producer.finished.connect(self.producer_thread.quit)
        self.producer.finished.connect(self.worker.stop)
        self.producer.error.connect(self.error)
        self.worker.error.connect(self.error)
        self.worker.started_ready.connect(self._start_producer)
        self.worker.finished.connect(self._worker_finished)

    def start(self) -> None:
        self.worker.start()

    @pyqtSlot(str)
    def _start_producer(self, _device: str) -> None:
        self.producer_thread.start()

    @pyqtSlot()
    def _worker_finished(self) -> None:
        self.producer.stop()
        self._join_producer()
        self.finished.emit()

    def _join_producer(self, timeout_ms: int = 2000) -> None:
        if self.producer_thread.isRunning():
            self.producer_thread.quit()
            self.producer_thread.wait(timeout_ms)

    def stop(self, timeout_ms: int = 2000) -> None:
        if self._stopping:
            return
        self._stopping = True
        self.producer.stop()
        self.worker.stop()
        self._join_producer(timeout_ms)
        if self.worker.isRunning():
            self.worker.wait(timeout_ms)


def create_live_session(
    *,
    config: FastAcbfConfig,
    datacube_data: np.ndarray,
    options: dict[str, Any] | None = None,
    parent=None,
    frame_interval_ms: int = 0,
) -> LiveSession:
    options = options or {}
    data = np.ascontiguousarray(np.asarray(datacube_data, dtype=np.float32))
    base_metadata = metadata_from_config(config, data)
    streamer = MockStreamer(
        data,
        base_metadata,
        jitter=jitter_from_options(options) or None,
        n_frames=options.get("n_frames"),
        seed=int(options.get("seed", 0)),
    )
    worker = LiveSolverWorker(
        cfg=config,
        initial_dataset=data,
        initial_metadata=base_metadata,
        parent=parent,
    )
    producer_thread = QThread(parent=parent)
    producer = _StreamerProducer(
        streamer,
        worker,
        frame_interval_ms=frame_interval_ms,
    )
    return LiveSession(
        worker=worker,
        producer=producer,
        producer_thread=producer_thread,
        base_metadata=base_metadata,
        parent=parent,
    )


def stop_live(session: LiveSession | None, timeout_ms: int = 2000) -> None:
    if session is not None:
        session.stop(timeout_ms=timeout_ms)
