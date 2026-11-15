# SPDX-License-Identifier: Apache-2.0
"""Cached package metadata lookups."""

from __future__ import annotations

import importlib.metadata

from sglang_omni.utils.package_metadata import packages_distributions


def test_packages_distributions_matches_the_standard_lookup() -> None:
    expected = importlib.metadata.packages_distributions()
    actual = packages_distributions()
    assert {name: sorted(distributions) for name, distributions in actual.items()} == {
        name: sorted(distributions) for name, distributions in expected.items()
    }
