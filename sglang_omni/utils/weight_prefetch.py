# SPDX-License-Identifier: Apache-2.0
"""Read a model's weight files into the OS page cache while the server starts.

Startup reads every weight once, and on network volumes that read is bound by per-request latency, not bandwidth: one reader is far below what the storage serves to many. Helper processes read the files in parallel while the stage processes are still importing and waiting for their turn on the GPU, so the loaders later map pages that are already resident.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

from huggingface_hub import snapshot_download
from huggingface_hub.errors import LocalEntryNotFoundError

logger = logging.getLogger(__name__)

SAFETENSORS_SUFFIX = ".safetensors"
PICKLE_WEIGHT_SUFFIXES = (".bin", ".pt", ".pth")
# Measured on Modal B200 hosts reading a network volume: several processes read far faster than one with as many threads, and 8 x 4 was the fastest split of 32 readers tried.
READER_PROCESSES = 8
READER_THREADS_PER_PROCESS = 4
# Arbitrary; leaves room for the stage processes' own host memory while they load.
MAX_PAGE_CACHE_FRACTION = 0.8
READER_SCRIPT = Path(__file__).with_name("weight_prefetch_reader.py")


def plan_weight_files(model_root: str) -> list[tuple[str, int, int]]:
    """Whole-file (path, offset, length) ranges for the weight files under a model directory.

    A directory that holds safetensors files contributes only those, as the loaders prefer them over pickled copies of the same weights.
    """
    sizes: dict[str, int] = {}
    for directory, _, file_names in os.walk(model_root, followlinks=True):
        safetensors_names = [
            name for name in file_names if name.endswith(SAFETENSORS_SUFFIX)
        ]
        if safetensors_names:
            weight_names = safetensors_names
        else:
            weight_names = [
                name for name in file_names if name.endswith(PICKLE_WEIGHT_SUFFIXES)
            ]
        for weight_name in sorted(weight_names):
            # Hub caches link names to blobs, and one blob can back several names.
            real_path = os.path.realpath(os.path.join(directory, weight_name))
            if os.path.isfile(real_path):
                sizes[real_path] = os.path.getsize(real_path)
            else:
                pass
    return [(path, 0, size) for path, size in sizes.items() if size > 0]


def start_weight_prefetch(model_path: str) -> list[subprocess.Popen[str]]:
    """Start helper processes that read the model's weight files once; they exit when done.

    Starts none unless the model is already on local storage and its weights fit in the host memory the page cache can grow into.
    """
    if os.path.isdir(model_path):
        model_root = model_path
    else:
        try:
            model_root = snapshot_download(model_path, local_files_only=True)
        except (LocalEntryNotFoundError, OSError, ValueError):
            logger.info(
                f"Weight prefetch skipped: {model_path} is not on local storage"
            )
            return []
    ranges = plan_weight_files(model_root)
    total_bytes = sum(length for _, _, length in ranges)
    if not os.path.isfile("/proc/meminfo") or total_bytes == 0:
        return []
    else:
        pass
    with open("/proc/meminfo") as meminfo_file:
        headroom_bytes = next(
            int(line.split()[1]) * 1024
            for line in meminfo_file
            if line.startswith("MemAvailable:")
        )
    cgroup_limit_path = "/sys/fs/cgroup/memory.max"
    if os.path.isfile(cgroup_limit_path):
        cgroup_limit = Path(cgroup_limit_path).read_text().strip()
        if cgroup_limit != "max":
            cgroup_usage_bytes = int(Path("/sys/fs/cgroup/memory.current").read_text())
            headroom_bytes = min(headroom_bytes, int(cgroup_limit) - cgroup_usage_bytes)
        else:
            pass
    else:
        pass
    if total_bytes > MAX_PAGE_CACHE_FRACTION * headroom_bytes:
        logger.info(
            f"Weight prefetch skipped: {total_bytes / 1e9:.1f} GB of weights against "
            f"{headroom_bytes / 1e9:.1f} GB of free host memory"
        )
        return []
    else:
        pass

    plan_file_descriptor, plan_path = tempfile.mkstemp(
        prefix="sglang-omni-weight-prefetch-", suffix=".json"
    )
    with os.fdopen(plan_file_descriptor, "w") as plan_file:
        json.dump(ranges, plan_file)
    readers = [
        subprocess.Popen(
            [
                sys.executable,
                "-S",
                # The script's directory holds modules named like standard ones.
                "-P",
                str(READER_SCRIPT),
                str(reader_index),
                str(READER_PROCESSES),
                str(READER_THREADS_PER_PROCESS),
                plan_path,
            ],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for reader_index in range(READER_PROCESSES)
    ]
    started_at = time.monotonic()
    logger.info(
        f"Weight prefetch started: {len(ranges)} files, {total_bytes / 1e9:.1f} GB, "
        f"{READER_PROCESSES} reader processes"
    )

    def wait_for_readers() -> None:
        bytes_read = 0
        for reader in readers:
            reader_output, reader_error = reader.communicate()
            if reader.returncode == 0:
                bytes_read += int(reader_output)
            else:
                logger.warning(f"Weight prefetch reader failed: {reader_error.strip()}")
        os.unlink(plan_path)
        elapsed_seconds = time.monotonic() - started_at
        logger.info(
            f"Weight prefetch done: {bytes_read / 1e9:.1f} GB in {elapsed_seconds:.1f} s "
            f"({bytes_read / max(elapsed_seconds, 1e-9) / 1e9:.1f} GB/s)"
        )

    threading.Thread(
        target=wait_for_readers, name="weight-prefetch-readers", daemon=True
    ).start()
    return readers
