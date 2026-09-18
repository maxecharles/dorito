"""Minimal test conftest.

This file intentionally avoids providing test-time stubs. Tests should
import and exercise the real third-party packages installed in the test
environment. Only collection-time tweaks (ignore the built `site/` dir)
are kept here.
"""


def pytest_ignore_collect(collection_path, config):
    try:
        p = str(collection_path)
    except Exception:
        return False
    if "/site/" in p or p.endswith("/site") or p.startswith("site"):
        return True
    return False
