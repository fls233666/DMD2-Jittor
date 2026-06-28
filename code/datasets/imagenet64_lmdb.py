"""ImageNet-64 LMDB dataset helpers for Jittor DMD2 training.

The LMDB layout matches ``DMD2-pytorch/main/data/lmdb_dataset.py``:

- ``images_shape`` and ``labels_shape`` store space-separated array shapes.
- ``images_{idx}_data`` stores one uint8 image row.
- ``labels_{idx}_data`` stores one int64 label row.
"""

import os

import numpy as np
from jittor.dataset import Dataset

try:
    import lmdb
except ModuleNotFoundError:
    lmdb = None

try:
    from .transforms import make_image_classification_transform, one_hot
except ImportError:
    from transforms import make_image_classification_transform, one_hot


IMAGENET64_NUM_CLASSES = 1000
IMAGENET64_LMDB_DIRNAME = "imagenet-64x64_lmdb"


def default_imagenet64_lmdb_root():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    return os.path.join(project_root, "data", IMAGENET64_LMDB_DIRNAME)


def parse_class_subset(value):
    if value is None or value == "":
        return None
    if isinstance(value, (list, tuple, set)):
        return [int(item) for item in value]

    result = []
    for part in str(value).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            result.extend(range(int(start), int(end) + 1))
        else:
            result.append(int(part))
    return result


def _require_lmdb():
    if lmdb is None:
        raise ModuleNotFoundError(
            "ImageNet64LMDBDataset requires the 'lmdb' package. "
            "Install it in the Jittor environment before training."
        )


def _open_lmdb(path):
    _require_lmdb()
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"ImageNet-64 LMDB not found: {path}. "
            "Expected the same imagenet-64x64_lmdb directory used by PyTorch."
        )
    return lmdb.open(path, readonly=True, lock=False, readahead=False, meminit=False)


def get_array_shape_from_lmdb(env, array_name):
    with env.begin() as txn:
        raw = txn.get(f"{array_name}_shape".encode())
    if raw is None:
        raise KeyError(f"Missing LMDB key: {array_name}_shape")
    return tuple(map(int, raw.decode().split()))


def retrieve_row_from_lmdb(env, array_name, dtype, shape, row_index):
    data_key = f"{array_name}_{int(row_index)}_data".encode()
    with env.begin() as txn:
        row_bytes = txn.get(data_key)
    if row_bytes is None:
        raise IndexError(f"Missing LMDB key: {data_key.decode()}")

    array = np.frombuffer(row_bytes, dtype=dtype)
    if len(shape) > 0:
        array = array.reshape(shape)
    return np.array(array, copy=True)


def _build_indices(length, label_getter=None, max_samples=None, class_subset=None, seed=0, shuffle_subset=True):
    indices = np.arange(int(length), dtype=np.int64)

    if class_subset is not None:
        if label_getter is None:
            raise ValueError("label_getter is required when class_subset is set")
        class_subset = set(int(label) for label in class_subset)
        keep = []
        for index in indices:
            keep.append(int(label_getter(int(index))) in class_subset)
        indices = indices[np.asarray(keep, dtype=bool)]

    if shuffle_subset:
        rng = np.random.default_rng(seed)
        indices = rng.permutation(indices)

    if max_samples is not None:
        indices = indices[: int(max_samples)]

    return [int(index) for index in indices]


class ImageNet64LMDBDataset(Dataset):
    # Return DMD2-ready dict batches: image in [-1, 1], one-hot label, class id.
    def __init__(
        self,
        root=None,
        train=True,
        max_samples=None,
        image_size=64,
        augment=False,
        crop_padding=0,
        hflip_prob=0.5,
        normalize=True,
        one_hot_labels=True,
        class_subset=None,
        num_classes=IMAGENET64_NUM_CLASSES,
        seed=0,
        shuffle_subset=True,
        transform=None,
    ):
        super().__init__()

        self.root = default_imagenet64_lmdb_root() if root is None else root
        self.train = train
        self.num_classes = int(num_classes)
        self.one_hot_labels = one_hot_labels
        self.env = None

        if self.num_classes != IMAGENET64_NUM_CLASSES:
            raise ValueError(
                "ImageNet-64 alignment expects 1000 classes; "
                f"got num_classes={self.num_classes}."
            )

        if transform is None:
            transform = make_image_classification_transform(
                train=train,
                image_size=image_size,
                augment=augment,
                crop_padding=crop_padding,
                hflip_prob=hflip_prob,
                normalize=normalize,
            )
        self.transform = transform

        env = self._env()
        self.image_shape = get_array_shape_from_lmdb(env, "images")
        self.label_shape = get_array_shape_from_lmdb(env, "labels")
        if len(self.image_shape) != 4:
            raise ValueError(f"Expected images_shape to have 4 dims, got {self.image_shape}")
        if self.image_shape[0] != self.label_shape[0]:
            raise ValueError(
                f"Image/label length mismatch: images={self.image_shape[0]}, "
                f"labels={self.label_shape[0]}"
            )

        class_subset = parse_class_subset(class_subset)
        self.indices = _build_indices(
            length=self.image_shape[0],
            label_getter=self._class_id_at,
            max_samples=max_samples,
            class_subset=class_subset,
            seed=seed,
            shuffle_subset=shuffle_subset,
        )
        self.set_attrs(total_len=len(self.indices))

    def _env(self):
        if self.env is None:
            self.env = _open_lmdb(self.root)
        return self.env

    def _class_id_at(self, source_index):
        label = retrieve_row_from_lmdb(
            self._env(),
            "labels",
            np.int64,
            self.label_shape[1:],
            source_index,
        )
        class_id = int(np.asarray(label).reshape(-1)[0])
        if class_id < 0 or class_id >= self.num_classes:
            raise ValueError(
                f"Class id {class_id} at LMDB row {source_index} is outside "
                f"[0, {self.num_classes})."
            )
        return class_id

    def __getitem__(self, index):
        source_index = self.indices[index]
        image = retrieve_row_from_lmdb(
            self._env(),
            "images",
            np.uint8,
            self.image_shape[1:],
            source_index,
        )
        class_id = self._class_id_at(source_index)

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

    def __getstate__(self):
        state = dict(self.__dict__)
        state["env"] = None
        return state

    def __del__(self):
        env = getattr(self, "env", None)
        if env is not None:
            env.close()


def build_imagenet64_lmdb_dataset(
    root=None,
    train=True,
    max_samples=None,
    image_size=64,
    augment=False,
    seed=0,
    class_subset=None,
    num_classes=IMAGENET64_NUM_CLASSES,
    **kwargs,
):
    return ImageNet64LMDBDataset(
        root=root,
        train=train,
        max_samples=max_samples,
        image_size=image_size,
        augment=augment,
        seed=seed,
        class_subset=class_subset,
        num_classes=num_classes,
        **kwargs,
    )


def build_imagenet64_lmdb_loader(
    root=None,
    train=True,
    max_samples=None,
    batch_size=32,
    shuffle=True,
    drop_last=True,
    num_workers=0,
    image_size=64,
    augment=False,
    seed=0,
    class_subset=None,
    num_classes=IMAGENET64_NUM_CLASSES,
    **kwargs,
):
    dataset = build_imagenet64_lmdb_dataset(
        root=root,
        train=train,
        max_samples=max_samples,
        image_size=image_size,
        augment=augment,
        seed=seed,
        class_subset=class_subset,
        num_classes=num_classes,
        **kwargs,
    )
    return dataset.set_attrs(
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
    )
