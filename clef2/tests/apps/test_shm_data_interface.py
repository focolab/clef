"""Tests for SharedMemoryUint16DataInterface."""

import numpy as np
import pytest
from multiprocessing import shared_memory
from types import SimpleNamespace

from clef2.apps.io.input_device.shm_uint16_data_interface import (
    SharedMemoryUint16DataInterface,
)


def _make_di(height=4, width=4, num_z_planes=1, shm_name_prefix="shared_frame_memory"):
    """Helper: create a data interface with a mock input device and config."""
    device = SimpleNamespace(height=height, width=width)
    config = {"num_z_planes": num_z_planes, "shm_name_prefix": shm_name_prefix}
    return SharedMemoryUint16DataInterface(input_device=device, config=config)


@pytest.fixture
def di():
    """Create a configured SharedMemoryUint16DataInterface and clean up after."""
    d = _make_di()
    yield d
    d.close()


class TestConfigureSampling:

    def test_creates_shm_segments(self):
        di = _make_di(height=64, width=64, num_z_planes=3)
        di.configure_sampling()
        assert len(di.shm_names) == 3
        for name in di.shm_names:
            shm = shared_memory.SharedMemory(name=name, create=False)
            shm.close()
        di.close()

    def test_creates_image_count(self):
        di = _make_di(height=64, width=64)
        di.configure_sampling()
        shl = shared_memory.ShareableList(name=di.image_count_shm_name)
        assert shl[0] == 0
        shl.shm.close()
        di.close()

    def test_no_save_buffer_by_default(self):
        di = _make_di(height=64, width=64)
        di.configure_sampling()
        assert di._save_buffer is None
        di.close()

    def test_save_buffer_allocated_when_requested(self):
        di = _make_di(height=64, width=64)
        di.configure_sampling(num_samples=100, save_samples=True)
        assert di._save_buffer is not None
        assert di._save_buffer.shape == (100, 64, 64)
        assert di._save_buffer.dtype == np.uint16
        di.close()

    def test_skips_when_zero_dimensions(self):
        di = _make_di(height=0, width=0)
        di.configure_sampling()
        assert len(di.shm_names) == 0
        di.close()


class TestStoreInput:

    def test_writes_to_shm(self):
        di = _make_di()
        di.configure_sampling()
        frame = np.ones((4, 4), dtype=np.uint16) * 42
        di.store_input(frame)

        shm = shared_memory.SharedMemory(name=di.shm_names[0], create=False)
        arr = np.ndarray((4, 4), dtype=np.uint16, buffer=shm.buf)
        np.testing.assert_array_equal(arr, frame)
        shm.close()
        di.close()

    def test_increments_image_count(self):
        di = _make_di()
        di.configure_sampling()
        frame = np.zeros((4, 4), dtype=np.uint16)
        di.store_input(frame)
        di.store_input(frame)

        shl = shared_memory.ShareableList(name=di.image_count_shm_name)
        assert shl[0] == 2
        shl.shm.close()
        di.close()

    def test_cycles_z_planes(self):
        di = _make_di(num_z_planes=2)
        di.configure_sampling()
        frame0 = np.ones((4, 4), dtype=np.uint16) * 10
        frame1 = np.ones((4, 4), dtype=np.uint16) * 20
        frame2 = np.ones((4, 4), dtype=np.uint16) * 30

        di.store_input(frame0)  # z=0
        di.store_input(frame1)  # z=1
        di.store_input(frame2)  # z=0 again

        shm0 = shared_memory.SharedMemory(name=di.shm_names[0], create=False)
        arr0 = np.ndarray((4, 4), dtype=np.uint16, buffer=shm0.buf)
        np.testing.assert_array_equal(arr0, frame2)  # overwritten
        shm0.close()

        shm1 = shared_memory.SharedMemory(name=di.shm_names[1], create=False)
        arr1 = np.ndarray((4, 4), dtype=np.uint16, buffer=shm1.buf)
        np.testing.assert_array_equal(arr1, frame1)
        shm1.close()
        di.close()

    def test_writes_to_save_buffer(self):
        di = _make_di()
        di.configure_sampling(num_samples=10, save_samples=True)
        frame = np.ones((4, 4), dtype=np.uint16) * 7
        di.store_input(frame)
        np.testing.assert_array_equal(di._save_buffer[0], frame)
        di.close()

    def test_no_save_buffer_still_works(self):
        di = _make_di()
        di.configure_sampling()
        frame = np.ones((4, 4), dtype=np.uint16)
        di.store_input(frame)
        assert di._sample_index == 1
        di.close()

    def test_input_store_set(self):
        di = _make_di()
        di.configure_sampling()
        frame = np.ones((4, 4), dtype=np.uint16) * 5
        di.store_input(frame)
        np.testing.assert_array_equal(di.input_store, frame)
        di.close()

    def test_none_sample(self):
        di = _make_di()
        di.configure_sampling()
        di.store_input(None)
        assert di.input_store is None
        di.close()


class TestSaveData:

    def test_saves_tiff(self, tmp_path):
        di = _make_di()
        di.configure_sampling(num_samples=5, save_samples=True)
        for i in range(3):
            di.store_input(np.ones((4, 4), dtype=np.uint16) * i)

        filepath = tmp_path / "test"
        di.save_data(filepath=filepath)

        import tifffile as tf

        data = tf.imread(str(filepath) + ".tiff")
        assert data.shape == (3, 4, 4)
        di.close()

    def test_no_buffer_warns(self, tmp_path):
        di = _make_di()
        di.configure_sampling()
        di.save_data(filepath=tmp_path / "test")  # should not raise
        di.close()


class TestClose:

    def test_unlinks_shm(self):
        di = _make_di(num_z_planes=2)
        di.configure_sampling()
        names = list(di.shm_names)
        count_name = di.image_count_shm_name
        di.close()

        for name in names:
            with pytest.raises(FileNotFoundError):
                shared_memory.SharedMemory(name=name, create=False)

        with pytest.raises(FileNotFoundError):
            shared_memory.ShareableList(name=count_name)


class TestProperties:

    def test_frame_shape(self):
        di = _make_di(height=128, width=256)
        di.configure_sampling()
        assert di.frame_shape == (128, 256)
        di.close()

    def test_zsize(self):
        di = _make_di(num_z_planes=3)
        di.configure_sampling()
        assert di.zsize == 3
        di.close()

    def test_get_metadata(self):
        di = _make_di(num_z_planes=2)
        di.configure_sampling()
        meta = di.get_metadata()
        assert meta["zsize"] == 2
        assert len(meta["shm_names"]) == 2
        assert meta["data_interface_class"] == "shm_uint16_data_interface"
        di.close()
