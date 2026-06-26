"""Tiny-ImageNet dataset helpers for Jittor DMD2 training."""

import os

import numpy as np
from PIL import Image
from jittor.dataset import Dataset

try:
    from .transforms import make_image_classification_transform, one_hot
except ImportError:
    from transforms import make_image_classification_transform, one_hot


TINY_IMAGENET_NUM_CLASSES = 200
TINY_IMAGENET_ARCHIVE = "tiny-imagenet-200.zip"
TINY_IMAGENET_DIRNAME = "tiny-imagenet-200"


def default_tiny_imagenet_root():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    return os.path.join(project_root, "data", TINY_IMAGENET_DIRNAME)


def read_words(root):
    words_path = os.path.join(root, "words.txt")
    result = {}
    if not os.path.exists(words_path):
        return result
    with open(words_path, "r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t", 1)
            if len(parts) == 2:
                result[parts[0]] = parts[1]
    return result


def read_wnids(root):
    wnids_path = os.path.join(root, "wnids.txt")
    with open(wnids_path, "r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


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


def collect_train_samples(root, wnids, word_to_label):
    samples = []
    for wnid in wnids:
        label = word_to_label[wnid]
        image_dir = os.path.join(root, "train", wnid, "images")
        if not os.path.isdir(image_dir):
            continue
        for filename in sorted(os.listdir(image_dir)):
            if filename.lower().endswith((".jpeg", ".jpg", ".png")):
                samples.append((os.path.join(image_dir, filename), label))
    return samples


def collect_val_samples(root, word_to_label):
    annotation_path = os.path.join(root, "val", "val_annotations.txt")
    image_dir = os.path.join(root, "val", "images")
    samples = []
    with open(annotation_path, "r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.strip().split("\t")
            if len(parts) < 2:
                continue
            filename, wnid = parts[0], parts[1]
            if wnid in word_to_label:
                samples.append((os.path.join(image_dir, filename), word_to_label[wnid]))
    return samples


def filter_samples(samples, max_samples=None, class_subset=None, seed=0, shuffle_subset=True):
    indices = np.arange(len(samples), dtype=np.int64)

    if class_subset is not None:
        class_subset = set(int(item) for item in class_subset)
        labels = np.asarray([label for _, label in samples], dtype=np.int64)
        mask = np.asarray([label in class_subset for label in labels], dtype=bool)
        indices = indices[mask]

    if shuffle_subset:
        rng = np.random.default_rng(seed)
        indices = rng.permutation(indices)

    if max_samples is not None:
        indices = indices[: int(max_samples)]

    return [samples[int(index)] for index in indices]


class TinyImageNetDataset(Dataset):
    # Return DMD2-ready dict batches: image, one-hot label, and raw class id.
    def __init__(
        self,
        root=None,
        train=True,
        max_samples=None,
        image_size=64,
        augment=True,
        crop_padding=4,
        hflip_prob=0.5,
        normalize=True,
        one_hot_labels=True,
        class_subset=None,
        num_classes=None,
        seed=0,
        shuffle_subset=True,
        transform=None,
    ):
        super().__init__()

        self.root = default_tiny_imagenet_root() if root is None else root
        self.train = train
        self.one_hot_labels = one_hot_labels

        if not os.path.exists(os.path.join(self.root, "wnids.txt")):
            raise FileNotFoundError(
                f"Tiny-ImageNet files not found in {self.root}. "
                "Run scripts/download_datasets.sh with DATASET=tiny-imagenet first."
            )

        self.wnids = read_wnids(self.root)
        self.word_to_label = {wnid: index for index, wnid in enumerate(self.wnids)}
        words = read_words(self.root)
        self.class_names = tuple(words.get(wnid, wnid) for wnid in self.wnids)
        self.num_classes = len(self.wnids) if num_classes is None else int(num_classes)

        class_subset = parse_class_subset(class_subset)
        if class_subset is not None and self.num_classes < len(self.wnids):
            invalid = [int(item) for item in class_subset if int(item) >= self.num_classes]
            if invalid:
                raise ValueError(
                    "Tiny-ImageNet subset runs with num_classes < 200 currently "
                    f"support only original class ids < {self.num_classes}; "
                    f"got invalid ids {invalid[:8]}."
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

        samples = (
            collect_train_samples(self.root, self.wnids, self.word_to_label)
            if train
            else collect_val_samples(self.root, self.word_to_label)
        )
        if self.num_classes < len(self.wnids):
            class_subset = list(range(self.num_classes)) if class_subset is None else class_subset
        self.samples = filter_samples(
            samples=samples,
            max_samples=max_samples,
            class_subset=class_subset,
            seed=seed,
            shuffle_subset=shuffle_subset,
        )

        self.set_attrs(total_len=len(self.samples))

    def __getitem__(self, index):
        path, class_id = self.samples[index]
        image = Image.open(path).convert("RGB")
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
        return len(self.samples)


def build_tiny_imagenet_dataset(
    root=None,
    train=True,
    max_samples=None,
    image_size=64,
    augment=True,
    seed=0,
    class_subset=None,
    num_classes=None,
    **kwargs,
):
    return TinyImageNetDataset(
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


def build_tiny_imagenet_loader(
    root=None,
    train=True,
    max_samples=None,
    batch_size=32,
    shuffle=True,
    drop_last=True,
    num_workers=0,
    image_size=64,
    augment=True,
    seed=0,
    class_subset=None,
    num_classes=None,
    **kwargs,
):
    dataset = build_tiny_imagenet_dataset(
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


def batch_to_real_train_dict(batch):
    return {
        "real_image": batch["image"],
        "real_label": batch["label"],
    }
