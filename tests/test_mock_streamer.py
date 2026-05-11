import numpy as np
import pytest

from py4d_browser_plugin.fast_acbf.streamers import MockStreamer


@pytest.fixture
def base_metadata():
    return {
        "wavelength": 0.0197,
        "max_alpha": 25.0,
        "dk": 0.05,
        "scan_shape": (4, 4),
        "scan_step_size": 0.2,
        "rotation_deg": 30.0,
        "flipud": False,
        "fliplr": False,
        "transpose": False,
    }


@pytest.fixture
def dataset():
    return np.zeros((4, 4, 8, 8), dtype=np.float32)


def test_yields_metadata_dict_with_expected_keys(dataset, base_metadata):
    s = MockStreamer(dataset, base_metadata, n_frames=1)
    data, meta = next(iter(s))
    assert data is dataset
    assert set(meta.keys()) == set(base_metadata.keys())


def test_default_yields_constant_metadata(dataset, base_metadata):
    s = MockStreamer(dataset, base_metadata, n_frames=5)
    metas = [meta for _, meta in s]
    for m in metas:
        assert m == base_metadata


def test_linear_and_cyclic_sweeps_are_deterministic(dataset, base_metadata):
    s = MockStreamer(
        dataset,
        base_metadata,
        linear_sweep={"rotation_deg": 2.0},
        cyclic_sweep={"scan_step_size": (0.1, 4)},
        n_frames=5,
    )
    metas = [meta for _, meta in s]

    assert [m["rotation_deg"] for m in metas] == [30.0, 32.0, 34.0, 36.0, 38.0]
    np.testing.assert_allclose(
        [m["scan_step_size"] for m in metas],
        [0.2, 0.3, 0.2, 0.1, 0.2],
        atol=1e-7,
    )


def test_n_frames_terminates_iteration(dataset, base_metadata):
    s = MockStreamer(dataset, base_metadata, n_frames=3)
    assert sum(1 for _ in s) == 3


def test_copy_dataset_yields_distinct_buffers(dataset, base_metadata):
    s = MockStreamer(dataset, base_metadata, n_frames=2, copy_dataset=True)
    bufs = [data for data, _ in s]
    assert bufs[0] is not bufs[1]
    assert bufs[0] is not dataset
    np.testing.assert_array_equal(bufs[0], dataset)


def test_default_yields_same_buffer(dataset, base_metadata):
    s = MockStreamer(dataset, base_metadata, n_frames=3)
    bufs = [data for data, _ in s]
    assert all(b is dataset for b in bufs)


def test_output_buffer_receives_frames(dataset, base_metadata):
    dataset[...] = 2.0
    out = np.empty_like(dataset)
    s = MockStreamer(dataset, base_metadata, n_frames=2, output_buffer=out)
    bufs = [data for data, _ in s]
    assert all(data is out for data in bufs)
    np.testing.assert_array_equal(out, dataset)
