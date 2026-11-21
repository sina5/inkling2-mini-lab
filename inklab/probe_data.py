"""Read the binary dumps of llama-inkling-probe and group them by tensor and cache slot.

A cache slot is one layer visit. Inkling 2 Mini has 24 physical layers; layers 6-17 run three
times, so slots 0-23 are the first visit of each layer and slots 24-47 are passes 2 and 3 of
the loop core.
"""

import re
import struct
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from inklab.errors import DatasetError

N_PHYS = 24
CORE_START = 6
CORE_END = 18
CORE_LEN = CORE_END - CORE_START
NAME = re.compile(r"^(?P<base>[a-z_]+?)(?:_(?P<ctrl>\d))?-(?P<slot>\d+)$")
CTRL_NAMES = ("q", "k", "v", "ffn")
FIRST_SPECIAL_ID = 200000
END_MESSAGE_ID = 200010


def slot_to_layer_pass(slot: int) -> tuple[int, int]:
    """Return (physical layer, pass number starting at 1) for a cache slot."""
    if slot < N_PHYS:
        return slot, 1
    offset = slot - N_PHYS
    return CORE_START + offset % CORE_LEN, 2 + offset // CORE_LEN


def layer_pass_to_slot(layer: int, pass_number: int) -> int:
    """Return the cache slot of a physical layer in a given pass."""
    if pass_number == 1:
        return layer
    if not CORE_START <= layer < CORE_END:
        raise ValueError(f"layer {layer} is not in the loop core")
    return N_PHYS + (pass_number - 2) * CORE_LEN + (layer - CORE_START)


@dataclass
class ProbeDump:
    """All records of one probe run.

    Attributes:
        tokens: Token ids in prompt order; -1 marks image positions.
        tensors: Map from (base name, ctrl or -1, slot) to an array of shape (n_tokens, ...).
    """

    tokens: np.ndarray
    tensors: dict[tuple[str, int, int], np.ndarray]

    @property
    def is_image(self) -> np.ndarray:
        """Return a boolean mask of image positions."""
        return self.tokens < 0

    @property
    def is_user_text(self) -> np.ndarray:
        """Return a mask of ordinary text tokens after the system message.

        This leaves out the chat-template tokens, which are the same in every prompt.
        """
        ends = np.flatnonzero(self.tokens == END_MESSAGE_ID)
        after_system = np.arange(len(self.tokens)) > (ends[0] if len(ends) else -1)
        return after_system & (self.tokens >= 0) & (self.tokens < FIRST_SPECIAL_ID)

    def topk(self, slot: int) -> np.ndarray:
        """Return the selected experts, shape (n_tokens, 8)."""
        return self.tensors[("ffn_moe_topk", -1, slot)]

    def hidden(self, slot: int) -> np.ndarray:
        """Return the layer output, shape (n_tokens, n_embd)."""
        return self.tensors[("l_out", -1, slot)]

    def gate(self, kind: str, ctrl: str, slot: int) -> np.ndarray:
        """Return per-token (mean, mean abs) of a loop gate, shape (n_tokens, 2)."""
        return self.tensors[(f"loop_{kind}", CTRL_NAMES.index(ctrl), slot)]


class ProbeReader:
    """Parse one probe file."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def read(self) -> ProbeDump:
        """Read and group every record.

        Raises:
            DatasetError: If the file has no tokens record.
        """
        parts: dict[tuple[str, int, int], list[np.ndarray]] = defaultdict(list)
        tokens = None
        for name, array in self._records():
            if name == "tokens":
                tokens = array.reshape(-1)
                continue
            match = NAME.match(name)
            if match is None:
                continue
            ctrl = int(match["ctrl"]) if match["ctrl"] is not None else -1
            parts[(match["base"], ctrl, int(match["slot"]))].append(array)
        if tokens is None:
            raise DatasetError(f"{self.path.name} has no tokens record")
        tensors = {key: np.concatenate(chunks, axis=0) for key, chunks in parts.items()}
        return ProbeDump(tokens=tokens, tensors=tensors)

    def _records(self):
        data = self.path.read_bytes()
        offset = 0
        while offset < len(data):
            (name_len,) = struct.unpack_from("<I", data, offset)
            offset += 4
            name = data[offset:offset + name_len].decode()
            offset += name_len
            kind, ne0, ne1 = struct.unpack_from("<Iqq", data, offset)
            offset += 20
            count = ne0 * ne1
            dtype = np.float32 if kind == 0 else np.int32
            array = np.frombuffer(data, dtype=dtype, count=count, offset=offset).reshape(ne1, ne0)
            offset += 4 * count
            yield name, self._token_major(name, array)

    @staticmethod
    def _token_major(name: str, array: np.ndarray) -> np.ndarray:
        # Gate records are stored as (2, n_tokens); every other record is (n_tokens, ne0) already.
        return array.T.copy() if name.startswith("loop_") else array
