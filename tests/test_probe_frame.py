"""The probe and χ the dashboard shows, against PtyRAD's (ported from ptydy).

PtyRAD flips the measured patterns with ``meas_flipT = [flipud, fliplr, transpose]``
(``np.flip`` / transpose, in that order) and builds its probe on the plain k grid of the
flipped patterns. fast-acbf keeps the raw patterns and applies the same flags, in the same
order, to the *k values* of each detector pixel. Hence (checked below with asymmetric
aberrations up to 3-fold astigmatism):

- detector-frame coefficients from fast-acbf (``get_aberrations_dict(frame="detector")``)
  are PtyRAD's coefficients when ``meas_flipT`` equals fast-acbf's flags; the scan rotation
  does not enter them
- ``get_probe(frame="detector")`` is that probe on the *raw* detector grid, so its
  aperture function equals PtyRAD's after mirroring about the k = 0 pixel (index N // 2).
  For odd N this is ``np.flip``; for even N it is ``np.flip`` plus a one-pixel roll, i.e.
  the raw array is **not** PtyRAD's probe array whenever a flip is set

The dashboard (and Live View) shows the probe and χ after ``utils.to_output_grid`` (that
mirror): in the detector frame PtyRAD's arrays, in the scan frame the scan-frame
coefficients' on the plain grid, rotated with the scan like the image.
"""

import itertools
import logging

import numpy as np
import pytest

fast_acbf = pytest.importorskip("fast_acbf")
probe_mod = pytest.importorskip("ptyrad.optics.probe")
constants = pytest.importorskip("ptyrad.optics.constants")

KV, ALPHA, DK = 80, 25.0, 0.03
ABERRATIONS = {"C10": 50.0, "C12a": 80.0, "C12b": -30.0, "C21a": 2000.0, "C21b": 900.0,
               "C23a": 1500.0, "C23b": -700.0}
FLAGS = [dict(zip(("flipud", "fliplr", "transpose"), map(bool, v), strict=True))
         for v in itertools.product([0, 1], repeat=3)]


def _ids(f):
    return "".join(str(int(v)) for v in f.values())


def _solver(n, flags, rotation_deg=0.0):
    # the aperture must fit in the grid (else the Nyquist row has no mirror partner)
    assert ALPHA / 1e3 / constants.get_wavelength_ang(kv=KV) < (n // 2 - 1) * DK
    data = np.random.default_rng(0).random((4, 4, n, n)).astype(np.float32) + 1
    return fast_acbf.BFSolver(
        dataset=data, max_alpha=ALPHA, scan_step_size=1.0, dk=DK,
        wavelength=constants.get_wavelength_ang(kv=KV), max_order=2,
        aberrations=dict(ABERRATIONS), device="cpu",
        coord_transform={**flags, "rotation_deg": rotation_deg})


def _kspace(probe):
    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(probe)))


def _meas_flipT(a, flags):
    """PtyRAD's ``meas_flipT`` on one pattern."""
    if flags["flipud"]:
        a = a[::-1]
    if flags["fliplr"]:
        a = a[:, ::-1]
    if flags["transpose"]:
        a = a.T
    return a


@pytest.fixture(autouse=True)
def _quiet():
    logging.disable(logging.CRITICAL)
    yield
    logging.disable(logging.NOTSET)


def _ptyrad_kspace(solver, n):
    coeffs = solver.get_aberrations_dict(frame="detector")
    return _kspace(probe_mod.make_stem_probe(KV, ALPHA, n, 1 / (n * DK), coeffs))


def _ptyrad_chi(solver, n, frame="detector"):
    """PtyRAD's χ (``make_stem_probe``'s grid and surface) for the frame's coefficients."""
    k = np.fft.fftshift(np.fft.fftfreq(n, 1 / (n * DK)))
    ky, kx = np.meshgrid(k, k, indexing="ij")
    return probe_mod.make_aberration_surface_krivanek_cartesian(
        solver.get_aberrations_dict(frame=frame), kx, ky, constants.get_wavelength_ang(kv=KV))


def test_to_output_grid_is_meas_flipT_mirrored_about_k0():
    from py4d_browser_plugin.fast_acbf.utils import to_output_grid

    for n in (6, 7):
        a = np.arange(n * n).reshape(n, n)
        c = n // 2  # the k = 0 pixel stays put
        for flags in FLAGS:
            out = to_output_grid(a, flags)
            assert out[c, c] == a[c, c]
            if n % 2:  # odd N: exactly PtyRAD's array flip
                np.testing.assert_array_equal(out, _meas_flipT(a, flags))
    np.testing.assert_array_equal(to_output_grid(a, None), a)
    stack = np.arange(2 * 36).reshape(2, 6, 6)  # leading axes untouched
    flags = {"flipud": True, "fliplr": False, "transpose": True}
    np.testing.assert_array_equal(to_output_grid(stack, flags)[1], to_output_grid(stack[1], flags))


@pytest.mark.parametrize("flags", FLAGS, ids=_ids)
def test_raw_detector_frame_probe_is_not_ptyrads_array(flags):
    """The premise: fast-acbf's own array is PtyRAD's only without a flip (even N)."""
    n = 64
    s = _solver(n, flags, rotation_deg=25.0)
    k_fa = _kspace(s.get_probe(frame="detector").detach().numpy())
    k_pr = _ptyrad_kspace(s, n)
    off = np.abs(k_fa - k_pr).max() > 0.1 * np.abs(k_pr).max()
    assert off == any(flags.values())


@pytest.mark.parametrize("n", [64, 63])
@pytest.mark.parametrize("flags", FLAGS, ids=_ids)
def test_dashboard_detector_frame_probe_and_chi_are_ptyrads(flags, n):
    """The probe and χ the dashboard shows in the detector frame are PtyRAD's arrays for
    ``meas_flipT`` = the flags (not the raw detector grid's), whatever the rotation."""
    from py4d_browser_plugin.fast_acbf.worker import optics_diagnostics

    s = _solver(n, flags, rotation_deg=25.0)
    d = optics_diagnostics(s, "detector", 1.0)
    disk = np.isfinite(d["chi"])
    expected = _ptyrad_chi(s, n)
    assert np.abs(d["chi"] - expected)[disk].max() < 1e-3 * np.abs(expected[disk]).max()
    k_pr = _ptyrad_kspace(s, n)
    assert np.abs(_kspace(d["probe_complex"]) - k_pr).max() < 1e-4 * np.abs(k_pr).max()


def test_dashboard_upscaled_probe_is_ptyrads():
    from py4d_browser_plugin.fast_acbf.worker import optics_diagnostics

    flags = {"flipud": True, "fliplr": False, "transpose": True}
    s = _solver(64, flags)
    probe = optics_diagnostics(s, "detector", 1.5)["probe_complex"]
    assert probe.shape == (96, 96)
    k_pr = _ptyrad_kspace(s, 96)
    assert np.abs(_kspace(probe) - k_pr).max() < 1e-4 * np.abs(k_pr).max()


@pytest.mark.parametrize("flags", FLAGS, ids=_ids)
def test_dashboard_scan_frame_probe_rotates_like_the_image(flags):
    """In the scan frame the probe is the scan coefficients' χ on the plain grid: rotated
    by ``rotation_deg`` (what brings the scan image to the detector frame) it is the
    detector-frame probe, for every flip."""
    import torch
    from torchvision.transforms.functional import rotate

    from py4d_browser_plugin.fast_acbf.worker import optics_diagnostics

    n, rot = 64, 30.0
    s = _solver(n, flags, rotation_deg=rot)
    det = np.abs(optics_diagnostics(s, "detector", 1.0)["probe_complex"])
    scan = optics_diagnostics(s, "scan", 1.0)
    disk = np.isfinite(scan["chi"])
    np.testing.assert_allclose(scan["chi"][disk], _ptyrad_chi(s, n, "scan")[disk], atol=1e-2)

    def corr(angle):
        r = rotate(torch.tensor(np.abs(scan["probe_complex"]))[None], angle=angle)
        return np.corrcoef(det.ravel(), r.squeeze(0).numpy().ravel())[0, 1]

    assert corr(rot) > 0.95 > corr(-rot) + 0.2


def test_depth_stack_probes_and_chi_are_on_the_output_grid():
    from py4d_browser_plugin.fast_acbf.config import FastAcbfConfig
    from py4d_browser_plugin.fast_acbf.worker import depth_stack, optics_diagnostics

    flags = {"flipud": True, "fliplr": False, "transpose": True}
    s = _solver(64, flags)
    cfg = FastAcbfConfig(**flags, output_frame="detector", upscale=1.0)
    d = optics_diagnostics(s, "detector", 1.0)
    out = depth_stack(s, cfg, 3, 10.0)
    np.testing.assert_allclose(out["probe_stack"][1], d["probe_complex"], atol=1e-6)
    np.testing.assert_allclose(out["chi_stack"][1], d["chi"], atol=1e-4)


def test_live_view_probe_and_chi_match_the_dashboard():
    from py4d_browser_plugin.fast_acbf.config import FastAcbfConfig
    from py4d_browser_plugin.fast_acbf.live_view.output import compute_one_live_output
    from py4d_browser_plugin.fast_acbf.worker import optics_diagnostics

    flags = {"flipud": True, "fliplr": False, "transpose": True}
    s = _solver(64, flags)
    cfg = FastAcbfConfig(**flags, output_frame="detector", upscale=1.0)
    d = optics_diagnostics(s, "detector", 1.0)
    np.testing.assert_allclose(compute_one_live_output(s, cfg, "probe"),
                               np.abs(d["probe_complex"]), atol=1e-6)
    disk = np.isfinite(d["chi"])
    wrapped = np.angle(np.exp(1j * d["chi"]))
    live = compute_one_live_output(s, cfg, "chi")
    assert np.abs(np.angle(np.exp(1j * (live - wrapped))))[disk].max() < 1e-4
