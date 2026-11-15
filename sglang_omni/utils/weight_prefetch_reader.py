# SPDX-License-Identifier: Apache-2.0
"""Helper process that reads its share of a weight prefetch plan into the page cache.

Runs as a plain script so it starts without importing the package; standard library only.
"""

from __future__ import annotations

import ctypes
import json
import os
import signal
import sys
import threading

CHUNK_BYTES = 32 << 20
PR_SET_PDEATHSIG = 1
# Arbitrary; the server's own startup work keeps priority over the readers.
NICE_INCREMENT = 5


def read_share(
    ranges: list[tuple[str, int, int]],
    share_index: int,
    share_count: int,
    thread_count: int,
) -> int:
    """Read every share_count-th chunk of the ranges from share_index; returns the bytes read."""
    chunks = [
        (path, offset, min(CHUNK_BYTES, begin + length - offset))
        for path, begin, length in ranges
        for offset in range(begin, begin + length, CHUNK_BYTES)
    ][share_index::share_count]
    next_chunk = iter(chunks)
    lock = threading.Lock()
    file_descriptors: dict[str, int] = {}
    bytes_read = 0

    def read_chunks() -> None:
        nonlocal bytes_read
        view = memoryview(bytearray(CHUNK_BYTES))
        while True:
            with lock:
                chunk = next(next_chunk, None)
                if chunk is None:
                    return
                else:
                    path, offset, length = chunk
                    if path not in file_descriptors:
                        file_descriptors[path] = os.open(path, os.O_RDONLY)
                    else:
                        pass
                    file_descriptor = file_descriptors[path]
            chunk_bytes_read = 0
            while chunk_bytes_read < length:
                read_size = os.preadv(
                    file_descriptor,
                    [view[chunk_bytes_read:length]],
                    offset + chunk_bytes_read,
                )
                if read_size <= 0:
                    break
                else:
                    chunk_bytes_read += read_size
            with lock:
                bytes_read += chunk_bytes_read

    threads = [threading.Thread(target=read_chunks) for _ in range(thread_count)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    for file_descriptor in file_descriptors.values():
        os.close(file_descriptor)
    return bytes_read


def main(arguments: list[str]) -> None:
    share_index, share_count, thread_count = (int(value) for value in arguments[:3])
    # Exit with the server that started this reader.
    ctypes.CDLL("libc.so.6", use_errno=True).prctl(PR_SET_PDEATHSIG, signal.SIGKILL)
    os.nice(NICE_INCREMENT)
    with open(arguments[3]) as plan_file:
        ranges = [(path, begin, length) for path, begin, length in json.load(plan_file)]
    print(read_share(ranges, share_index, share_count, thread_count), flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
else:
    pass
