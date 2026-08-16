"""Download and load MNIST from the idx files. Cached under ./data."""
from __future__ import annotations

import gzip
import os
import urllib.request
from pathlib import Path

import numpy as np

BASE = "https://ossci-datasets.s3.amazonaws.com/mnist/"
FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}


def _download(data_dir: Path) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    for fname in FILES.values():
        dest = data_dir / fname
        if dest.exists():
            continue
        req = urllib.request.Request(BASE + fname, headers={"User-Agent": "nn-from-scratch"})
        with urllib.request.urlopen(req, timeout=60) as r, open(dest, "wb") as f:
            f.write(r.read())


def _read_images(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        data = f.read()
    # header: magic(4) + count(4) + rows(4) + cols(4), big-endian
    n = int.from_bytes(data[4:8], "big")
    rows = int.from_bytes(data[8:12], "big")
    cols = int.from_bytes(data[12:16], "big")
    images = np.frombuffer(data[16:], dtype=np.uint8).reshape(n, rows * cols)
    return images.astype(np.float32) / 255.0


def _read_labels(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        data = f.read()
    return np.frombuffer(data[8:], dtype=np.uint8).astype(np.int64)


def load_mnist(data_dir: str = "data"):
    """Return (x_train, y_train, x_test, y_test) with pixels scaled to [0, 1]."""
    d = Path(data_dir)
    _download(d)
    x_train = _read_images(d / FILES["train_images"])
    y_train = _read_labels(d / FILES["train_labels"])
    x_test = _read_images(d / FILES["test_images"])
    y_test = _read_labels(d / FILES["test_labels"])
    return x_train, y_train, x_test, y_test
