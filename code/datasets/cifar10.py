"""CIFAR-10 dataset helpers for small DMD2 debug training."""

import os

import numpy as np
from jittor.dataset import Dataset
from jittor.dataset.cifar import CIFAR10 as JittorCIFAR10

try:
    from .transforms import make_cifar10_transform, one_hot
except ImportError:
    from transforms import make_cifar10_transform, one_hot


CIFAR10_NUM_CLASSES = 10
CIFAR10_CLASS_NAMES = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)


def default_cifar10_root():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    return os.path.join(project_root, "data", "cifar10")


def _as_label_array(dataset):
    if hasattr(dataset, "targets"):
        return np.asarray(dataset.targets, dtype=np.int64)
    labels = []
    for index in range(len(dataset)):
        _, target = dataset[index]
        labels.append(int(target))
    return np.asarray(labels, dtype=np.int64)


def build_debug_indices(
    dataset,
    max_samples=None,
    class_subset=None,
    seed=0,
    shuffle_subset=True,
):
    indices = np.arange(len(dataset), dtype=np.int64)

    if class_subset is not None:
        class_subset = set(int(label) for label in class_subset)
        labels = _as_label_array(dataset)
        mask = np.asarray([label in class_subset for label in labels], dtype=bool)
        indices = indices[mask]

    if shuffle_subset:
        rng = np.random.default_rng(seed)
        indices = rng.permutation(indices)

    if max_samples is not None:
        indices = indices[: int(max_samples)]

    return [int(index) for index in indices]


class CIFAR10DebugDataset(Dataset):
    # Return DMD2-ready dict batches: image, one-hot label, and raw class id.
    def __init__(
        self,
        root=None,
        train=True,
        download=False,
        max_samples=5000,
        image_size=32,
        augment=True,
        crop_padding=4,
        hflip_prob=0.5,
        normalize=True,
        one_hot_labels=True,
        class_subset=None,
        seed=0,
        shuffle_subset=True,
        transform=None,
        base_dataset=None,
    ):
        super().__init__()

        self.root = default_cifar10_root() if root is None else root
        self.train = train
        self.num_classes = CIFAR10_NUM_CLASSES
        self.class_names = CIFAR10_CLASS_NAMES
        self.one_hot_labels = one_hot_labels

        if transform is None:
            transform = make_cifar10_transform(
                train=train,
                image_size=image_size,
                augment=augment,
                crop_padding=crop_padding,
                hflip_prob=hflip_prob,
                normalize=normalize,
            )
        self.transform = transform

        if base_dataset is None:
            base_dataset = JittorCIFAR10(
                root=self.root,
                train=train,
                transform=None,
                download=download,
            )
        self.base_dataset = base_dataset
        self.indices = build_debug_indices(
            dataset=base_dataset,
            max_samples=max_samples,
            class_subset=class_subset,
            seed=seed,
            shuffle_subset=shuffle_subset,
        )

        self.set_attrs(total_len=len(self.indices))

    def __getitem__(self, index):
        source_index = self.indices[index]
        image, class_id = self.base_dataset[source_index]
        class_id = int(class_id)

        image = self.transform(image).astype(np.float32)
        if self.one_hot_labels:
            label = one_hot(class_id, self.num_classes)
        else:
            label = np.asarray(class_id, dtype=np.int32)

        return {
            "image": image,
            "label": label,
            "class_id": np.asarray(class_id, dtype=np.int32),
        }

    def __len__(self):
        return len(self.indices)


def build_cifar10_debug_dataset(
    root=None,
    train=True,
    download=True,
    max_samples=5000,
    image_size=32,
    augment=True,
    seed=0,
    class_subset=None,
    **kwargs,
):
    return CIFAR10DebugDataset(
        root=root,
        train=train,
        download=download,
        max_samples=max_samples,
        image_size=image_size,
        augment=augment,
        seed=seed,
        class_subset=class_subset,
        **kwargs,
    )


def build_cifar10_debug_loader(
    root=None,
    train=True,
    download=True,
    max_samples=5000,
    batch_size=32,
    shuffle=True,
    drop_last=True,
    num_workers=0,
    image_size=32,
    augment=True,
    seed=0,
    class_subset=None,
    **kwargs,
):
    dataset = build_cifar10_debug_dataset(
        root=root,
        train=train,
        download=download,
        max_samples=max_samples,
        image_size=image_size,
        augment=augment,
        seed=seed,
        class_subset=class_subset,
        **kwargs,
    )
    return dataset.set_attrs(
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
    )


def batch_to_real_train_dict(batch):
    return {
        "real_image": batch["image"],
        "real_label": batch["label"],
    }
