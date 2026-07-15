"""Smoke tests for code/datasets migration files.

Run from the project root:

    python tests/test_datasets.py
"""

import os
import sys
from pathlib import Path

import numpy as np
import jittor as jt
from PIL import Image


def setup_import_path():
    this_file = Path(__file__).resolve()
    project_root = this_file.parents[1]
    code_dir = project_root / "code"

    for path in (project_root, code_dir, code_dir / "datasets"):
        path = str(path)
        if path not in sys.path:
            sys.path.insert(0, path)


setup_import_path()

from datasets.cifar10 import CIFAR10DebugDataset, batch_to_real_train_dict, build_debug_indices
from datasets.transforms import image_to_nchw_float, make_cifar10_transform, one_hot


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


class FakeCIFAR10:
    def __init__(self, length=20):
        self.data = []
        self.targets = []
        for index in range(length):
            image = np.zeros([32, 32, 3], dtype=np.uint8)
            image[:, :, 0] = index
            image[:, :, 1] = index * 2
            image[:, :, 2] = 255 - index
            self.data.append(image)
            self.targets.append(index % 10)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return Image.fromarray(self.data[index]), self.targets[index]


def test_transform_range_and_shape():
    image = Image.fromarray(np.ones([32, 32, 3], dtype=np.uint8) * 255)
    transformed = image_to_nchw_float(image)

    assert_shape("transformed image", transformed, [3, 32, 32])
    assert transformed.dtype == np.float32
    assert np.allclose(transformed, 1.0)

    transform = make_cifar10_transform(train=False, image_size=32)
    transformed = transform(image)
    assert_shape("cifar transform image", transformed, [3, 32, 32])


def test_one_hot():
    label = one_hot(3, 10)
    assert_shape("one hot", label, [10])
    assert label.dtype == np.float32
    assert label[3] == 1.0
    assert label.sum() == 1.0


def test_debug_indices_are_reproducible():
    dataset = FakeCIFAR10(length=30)
    first = build_debug_indices(dataset, max_samples=8, seed=123)
    second = build_debug_indices(dataset, max_samples=8, seed=123)
    assert first == second

    subset = build_debug_indices(
        dataset,
        max_samples=6,
        class_subset=[1, 3],
        seed=0,
        shuffle_subset=False,
    )
    labels = [dataset.targets[index] for index in subset]
    assert all(label in (1, 3) for label in labels)


def test_cifar10_debug_dataset_sample():
    dataset = CIFAR10DebugDataset(
        train=True,
        max_samples=5,
        augment=False,
        download=False,
        base_dataset=FakeCIFAR10(length=12),
        shuffle_subset=False,
    )
    sample = dataset[0]

    assert set(sample.keys()) == {"image", "label", "class_id"}
    assert_shape("sample image", sample["image"], [3, 32, 32])
    assert_shape("sample label", sample["label"], [10])
    assert sample["image"].dtype == np.float32
    assert sample["label"].dtype == np.float32
    assert int(sample["class_id"]) == 0
    assert sample["image"].min() >= -1.0
    assert sample["image"].max() <= 1.0


def test_cifar10_debug_loader_batch_dict():
    loader = CIFAR10DebugDataset(
        train=True,
        max_samples=4,
        augment=False,
        download=False,
        base_dataset=FakeCIFAR10(length=8),
        shuffle_subset=False,
    ).set_attrs(batch_size=2, shuffle=False, drop_last=True, num_workers=0)

    batch = next(iter(loader))
    assert set(batch.keys()) == {"image", "label", "class_id"}
    assert_shape("batch image", batch["image"], [2, 3, 32, 32])
    assert_shape("batch label", batch["label"], [2, 10])
    assert_shape("batch class id", batch["class_id"], [2])

    real_train_dict = batch_to_real_train_dict(batch)
    assert set(real_train_dict.keys()) == {"real_image", "real_label"}
    assert np.allclose(to_numpy(real_train_dict["real_label"]), to_numpy(batch["label"]))


def run_all_tests():
    configure_jittor()

    tests = [
        test_transform_range_and_shape,
        test_one_hot,
        test_debug_indices_are_reproducible,
        test_cifar10_debug_dataset_sample,
        test_cifar10_debug_loader_batch_dict,
    ]

    for test in tests:
        print(f"[RUN] {test.__name__}")
        test()
        jt.sync_all()
        print(f"[OK]  {test.__name__}")


if __name__ == "__main__":
    run_all_tests()
