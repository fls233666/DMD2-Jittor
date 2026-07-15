"""Smoke tests for the ImageNet-64 LMDB Jittor dataset.

Run from the project root in an environment with jittor and lmdb:

    python tests/test_imagenet64_lmdb.py
"""

import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import jittor as jt

try:
    import lmdb
except ModuleNotFoundError:
    lmdb = None


def setup_import_path():
    this_file = Path(__file__).resolve()
    project_root = this_file.parents[1]
    code_dir = project_root / "code"

    for path in (project_root, code_dir, code_dir / "datasets"):
        path = str(path)
        if path not in sys.path:
            sys.path.insert(0, path)


setup_import_path()

from datasets.imagenet64_lmdb import ImageNet64LMDBDataset, build_imagenet64_lmdb_loader


def configure_jittor():
    use_cuda = os.environ.get("JITTOR_USE_CUDA", "0") == "1"
    jt.flags.use_cuda = 1 if use_cuda else 0


def to_numpy(x):
    if isinstance(x, jt.Var):
        jt.sync_all()
        return np.asarray(x.numpy())
    return np.asarray(x)


def assert_shape(name, x, expected_shape):
    got = list(x.shape)
    expected = list(expected_shape)
    assert got == expected, f"{name}: got shape {got}, expected {expected}"


def write_lmdb(path, images, labels):
    if lmdb is None:
        raise ModuleNotFoundError("lmdb is required for this test")

    env = lmdb.open(path, map_size=64 * 1024 * 1024)
    with env.begin(write=True) as txn:
        txn.put(b"images_shape", " ".join(map(str, images.shape)).encode())
        txn.put(b"labels_shape", " ".join(map(str, labels.shape)).encode())
        for index, image in enumerate(images):
            txn.put(f"images_{index}_data".encode(), image.tobytes())
        for index, label in enumerate(labels):
            txn.put(f"labels_{index}_data".encode(), np.asarray(label).tobytes())
    env.close()


def make_chw_fixture(length=4):
    images = np.zeros([length, 3, 64, 64], dtype=np.uint8)
    labels = np.zeros([length, 1], dtype=np.int64)
    for index in range(length):
        images[index, 0, :, :] = index * 10
        images[index, 1, :, :] = 128
        images[index, 2, :, :] = 255 - index
        labels[index, 0] = index
    return images, labels


def test_imagenet64_lmdb_dataset_sample_and_loader():
    if lmdb is None:
        print("[SKIP] lmdb is not installed")
        return

    with tempfile.TemporaryDirectory() as tmpdir:
        images, labels = make_chw_fixture(length=4)
        write_lmdb(tmpdir, images, labels)

        dataset = ImageNet64LMDBDataset(
            root=tmpdir,
            max_samples=3,
            image_size=64,
            augment=False,
            seed=0,
            shuffle_subset=False,
        )
        assert len(dataset) == 3
        sample = dataset[0]
        assert set(sample.keys()) == {"image", "label", "class_id"}
        assert_shape("sample image", sample["image"], [3, 64, 64])
        assert_shape("sample label", sample["label"], [1000])
        assert sample["image"].dtype == np.float32
        assert sample["label"].dtype == np.float32
        assert int(sample["class_id"]) == 0
        assert -1.0 <= float(sample["image"].min()) <= 1.0
        assert -1.0 <= float(sample["image"].max()) <= 1.0
        assert np.isclose(sample["label"].sum(), 1.0)

        loader = build_imagenet64_lmdb_loader(
            root=tmpdir,
            max_samples=4,
            batch_size=2,
            shuffle=False,
            drop_last=True,
            num_workers=0,
            image_size=64,
            augment=False,
        )
        batch = next(iter(loader))
        assert_shape("batch image", batch["image"], [2, 3, 64, 64])
        assert_shape("batch label", batch["label"], [2, 1000])
        assert_shape("batch class id", batch["class_id"], [2])
        assert np.allclose(to_numpy(batch["label"]).sum(axis=1), 1.0)


def test_class_subset_uses_original_labels():
    if lmdb is None:
        print("[SKIP] lmdb is not installed")
        return

    with tempfile.TemporaryDirectory() as tmpdir:
        images, labels = make_chw_fixture(length=5)
        write_lmdb(tmpdir, images, labels)
        dataset = ImageNet64LMDBDataset(
            root=tmpdir,
            max_samples=None,
            image_size=64,
            augment=False,
            class_subset=[1, 3],
            shuffle_subset=False,
        )
        assert len(dataset) == 2
        assert int(dataset[0]["class_id"]) == 1
        assert int(dataset[1]["class_id"]) == 3


def run_all_tests():
    configure_jittor()

    tests = [
        test_imagenet64_lmdb_dataset_sample_and_loader,
        test_class_subset_uses_original_labels,
    ]

    for test in tests:
        print(f"[RUN] {test.__name__}")
        test()
        jt.sync_all()
        print(f"[OK]  {test.__name__}")


if __name__ == "__main__":
    run_all_tests()
