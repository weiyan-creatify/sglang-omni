# SPDX-License-Identifier: Apache-2.0
"""Cached installed-package metadata for the libraries a process imports."""

from __future__ import annotations

import csv
import functools
import importlib.metadata
import inspect
import os
import sys
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

METADATA_FILE_NAMES = ("METADATA", "RECORD", "top_level.txt", "entry_points.txt")
# Arbitrary; the files are small, so reads are bound by per-file latency.
METADATA_READ_THREADS = 32


def packages_distributions() -> dict[str, list[str]]:
    """The mapping importlib.metadata.packages_distributions returns, without a stat per installed file.

    Existence is checked per top-level entry, so a package directory whose listed files were all deleted still maps its name.
    """
    distribution_names: defaultdict[str, list[str]] = defaultdict(list)
    for distribution in importlib.metadata.distributions():
        top_level_names = (distribution.read_text("top_level.txt") or "").split()
        if top_level_names:
            pass
        else:
            record = distribution.read_text("RECORD")
            if record is not None:
                installed_paths = [
                    row[0] for row in csv.reader(record.splitlines()) if row
                ]
            else:
                # Egg-info installs list their files elsewhere.
                installed_paths = [str(path) for path in distribution.files or ()]
            top_level_entries: dict[str, str] = {}
            for installed_path in installed_paths:
                parts = installed_path.split("/")
                if len(parts) > 1:
                    top_level_name = parts[0]
                else:
                    top_level_name = (
                        inspect.getmodulename(installed_path) or installed_path
                    )
                if "." not in top_level_name:
                    top_level_entries.setdefault(top_level_name, parts[0])
                else:
                    pass
            top_level_names = [
                top_level_name
                for top_level_name, entry in top_level_entries.items()
                if os.path.exists(distribution.locate_file(entry))
            ]
        for top_level_name in dict.fromkeys(top_level_names):
            distribution_names[top_level_name].append(distribution.metadata["Name"])
    return dict(distribution_names)


def read_metadata_files() -> None:
    """Read every installed distribution's metadata files so later lookups find them cached."""
    metadata_paths: list[str] = []
    for search_path in sys.path:
        if os.path.isdir(search_path or "."):
            for entry_name in os.listdir(search_path or "."):
                if entry_name.endswith((".dist-info", ".egg-info")):
                    metadata_paths.extend(
                        os.path.join(search_path, entry_name, file_name)
                        for file_name in METADATA_FILE_NAMES
                    )
                else:
                    pass
        else:
            pass

    def read_metadata_file(metadata_path: str) -> None:
        if os.path.isfile(metadata_path):
            with open(metadata_path, "rb") as metadata_file:
                metadata_file.read()
        else:
            pass

    with ThreadPoolExecutor(METADATA_READ_THREADS) as pool:
        list(pool.map(read_metadata_file, metadata_paths))


def cache_package_metadata() -> None:
    """Answer this process's package metadata lookups from one cached pass.

    transformers and diffusers probe which packages are installed while they import: the standard lookup reads every distribution's RECORD and stats each file it lists, and every version lookup reads a METADATA file. Installed packages do not change while a process runs.
    """
    threading.Thread(
        target=read_metadata_files, name="read-package-metadata", daemon=True
    ).start()
    importlib.metadata.packages_distributions = functools.cache(packages_distributions)
    importlib.metadata.version = functools.cache(importlib.metadata.version)
