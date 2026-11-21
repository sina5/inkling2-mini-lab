"""Read the CIFAR-10 test set from the original Python pickle files."""

import io
import pickle
from pathlib import Path

import numpy as np
from PIL import Image

from inklab.constants import CIFAR_CLASSES, CIFAR_NATIVE_SIZE
from inklab.errors import DatasetError


class CifarTestSet:
    """The 10,000 CIFAR-10 test images and labels.

    Call load() before reading images.
    """

    def __init__(self, root: Path) -> None:
        self.root = root
        self._images: np.ndarray | None = None
        self._labels: np.ndarray | None = None

    def load(self) -> "CifarTestSet":
        """Read test_batch into memory.

        Raises:
            DatasetError: If the file is missing.
        """
        path = self.root / "test_batch"
        if not path.exists():
            raise DatasetError(f"CIFAR-10 test batch not found: {path}")
        with path.open("rb") as handle:
            batch = pickle.load(handle, encoding="bytes")
        images = np.asarray(batch[b"data"], dtype=np.uint8)
        self._images = images.reshape(-1, 3, CIFAR_NATIVE_SIZE, CIFAR_NATIVE_SIZE).transpose(0, 2, 3, 1)
        self._labels = np.asarray(batch[b"labels"], dtype=np.int64)
        return self

    def __len__(self) -> int:
        return len(self._require_labels())

    @property
    def labels(self) -> np.ndarray:
        """Return the class index of every image."""
        return self._require_labels()

    @property
    def class_names(self) -> tuple[str, ...]:
        """Return the 10 class names in label order."""
        return CIFAR_CLASSES

    def image(self, index: int) -> Image.Image:
        """Return one 32x32 RGB image."""
        if self._images is None:
            raise DatasetError("Call load() first")
        return Image.fromarray(self._images[index])

    def png(self, index: int, size: int) -> bytes:
        """Return one image resized to size x size with bicubic filtering, as PNG bytes."""
        image = self.image(index)
        if size != CIFAR_NATIVE_SIZE:
            image = image.resize((size, size), Image.BICUBIC)
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return buffer.getvalue()

    def _require_labels(self) -> np.ndarray:
        if self._labels is None:
            raise DatasetError("Call load() first")
        return self._labels
