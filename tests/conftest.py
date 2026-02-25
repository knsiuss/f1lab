# -*- coding: utf-8 -*-
"""Test infrastructure overrides for this repository."""

from __future__ import annotations

import shutil
import uuid
from pathlib import Path

import pytest


# Pytest's Windows tmp cleanup can fail in this environment due ACL issues on
# generated basetemp dirs. Guard the cleanup path so test reporting still
# completes and use a custom tmp_path fixture rooted in the workspace.
try:  # pragma: no cover - environment-specific workaround
    import _pytest.tmpdir as _pytest_tmpdir
    import _pytest.pathlib as _pytest_pathlib

    _orig_tmpdir_cleanup = getattr(_pytest_tmpdir, "cleanup_dead_symlinks", None)
    _orig_pathlib_cleanup = getattr(_pytest_pathlib, "cleanup_dead_symlinks", None)

    def _safe_cleanup_dead_symlinks(root):
        try:
            if _orig_pathlib_cleanup is not None:
                return _orig_pathlib_cleanup(root)
        except PermissionError:
            return None
        return None

    if _orig_tmpdir_cleanup is not None:
        _pytest_tmpdir.cleanup_dead_symlinks = _safe_cleanup_dead_symlinks
    if _orig_pathlib_cleanup is not None:
        _pytest_pathlib.cleanup_dead_symlinks = _safe_cleanup_dead_symlinks
except Exception:
    pass


@pytest.fixture
def tmp_path():
    """Workspace-local replacement for pytest's tmp_path fixture."""
    root = Path(__file__).parent.parent / ".pytest_local_tmp"
    root.mkdir(exist_ok=True)
    path = root / f"case_{uuid.uuid4().hex[:12]}"
    path.mkdir(parents=True, exist_ok=False)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)
