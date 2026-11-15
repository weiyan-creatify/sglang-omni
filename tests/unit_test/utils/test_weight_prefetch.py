# SPDX-License-Identifier: Apache-2.0
"""Weight prefetch planning and the reader helper."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from sglang_omni.utils.weight_prefetch import plan_weight_files, start_weight_prefetch
from sglang_omni.utils.weight_prefetch_reader import CHUNK_BYTES, read_share


def test_plan_reads_each_weight_once_and_prefers_safetensors(tmp_path: Path) -> None:
    blob = tmp_path / "blobs" / "0123"
    blob.parent.mkdir()
    blob.write_bytes(b"s" * 7)
    snapshot = tmp_path / "snapshot"
    (snapshot / "speech_tokenizer").mkdir(parents=True)
    (snapshot / "model.safetensors").symlink_to(blob)
    (snapshot / "copy.safetensors").symlink_to(blob)
    (snapshot / "pytorch_model.bin").write_bytes(b"b" * 5)
    (snapshot / "config.json").write_bytes(b"{}")
    (snapshot / "speech_tokenizer" / "decoder.pt").write_bytes(b"p" * 3)
    (snapshot / "speech_tokenizer" / "empty.pt").write_bytes(b"")

    assert sorted(plan_weight_files(str(snapshot))) == sorted(
        [
            (os.path.realpath(blob), 0, 7),
            (os.path.realpath(snapshot / "speech_tokenizer" / "decoder.pt"), 0, 3),
        ]
    )


@pytest.mark.skipif(not hasattr(os, "preadv"), reason="the reader uses preadv")
def test_reader_shares_cover_every_byte_once(tmp_path: Path) -> None:
    weight_path = tmp_path / "model.safetensors"
    weight_size = 2 * CHUNK_BYTES + 11
    weight_path.write_bytes(b"w" * weight_size)
    ranges = [(str(weight_path), 0, weight_size)]

    assert (
        sum(read_share(ranges, share_index, 2, 3) for share_index in range(2))
        == weight_size
    )


@pytest.mark.skipif(sys.platform != "linux", reason="the prefetch reads /proc/meminfo")
def test_started_readers_exit_cleanly(tmp_path: Path) -> None:
    """The readers once died at startup: their script's directory shadowed a standard module."""
    (tmp_path / "model.safetensors").write_bytes(b"w" * 1024)

    readers = start_weight_prefetch(str(tmp_path))

    assert readers
    assert [reader.wait() for reader in readers] == [0] * len(readers)
