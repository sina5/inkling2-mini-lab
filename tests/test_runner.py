"""Tests for the resumable JSON Lines runner and the probe file reader."""

import struct
from pathlib import Path

import numpy as np

from inklab.probe_data import ProbeReader
from inklab.runner import JsonlRunner, read_jsonl


def test_runner_writes_one_record_per_key(tmp_path: Path) -> None:
    runner = JsonlRunner(tmp_path / "out.jsonl", workers=2)
    assert runner.run([1, 2, 3], lambda k: {"value": k * 10}) == 3
    assert sorted(r["value"] for r in read_jsonl(tmp_path / "out.jsonl")) == [10, 20, 30]


def test_runner_skips_done_keys(tmp_path: Path) -> None:
    runner = JsonlRunner(tmp_path / "out.jsonl", workers=1)
    runner.run([1, 2], lambda k: {"value": k})
    calls = []
    added = runner.run([1, 2, 3], lambda k: calls.append(k) or {"value": k})
    assert added == 1
    assert calls == [3]


def _record(name: str, kind: int, array: np.ndarray) -> bytes:
    ne1, ne0 = array.shape
    head = struct.pack("<I", len(name)) + name.encode() + struct.pack("<Iqq", kind, ne0, ne1)
    return head + array.astype(np.float32 if kind == 0 else np.int32).tobytes()


def test_probe_reader_concatenates_ubatches(tmp_path: Path) -> None:
    path = tmp_path / "p.bin"
    topk_a = np.arange(16).reshape(2, 8)
    topk_b = np.arange(8).reshape(1, 8)
    gain = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    tokens = np.array([[5, -1, 7]])
    path.write_bytes(_record("ffn_moe_topk-3", 1, topk_a) + _record("ffn_moe_topk-3", 1, topk_b)
                     + _record("loop_gain_3-30", 0, gain) + _record("tokens", 1, tokens))
    dump = ProbeReader(path).read()
    assert dump.topk(3).shape == (3, 8)
    assert dump.gate("gain", "ffn", 30).shape == (3, 2)
    assert dump.is_image.tolist() == [False, True, False]
