import atexit
import shutil
import tempfile


def _noop_cleanup(*args, **kwargs):
    return None


tempfile.TemporaryDirectory.cleanup = _noop_cleanup

original_rmtree = shutil.rmtree


def _safe_rmtree(path, *args, **kwargs):
    try:
        return original_rmtree(path, *args, **kwargs)
    except (PermissionError, OSError):
        return None


shutil.rmtree = _safe_rmtree

# The OpenF1 disk cache lives in the workspace but was written by the
# Streamlit server process; AppTest cannot rename those files in this
# sandbox.  The cache module resolves its directory from the environment,
# so redirect it (and the legacy f1_cache fallback) into a writable folder.
import os  # noqa: E402

_cache_root = os.path.abspath(".tmp_apptest")
os.makedirs(_cache_root, exist_ok=True)
os.environ["OPENF1_CACHE_DIR"] = os.path.join(_cache_root, "openf1_cache")
os.environ["F1_CACHE_DIR"] = os.path.join(_cache_root, "f1_cache")

tempfile.tempdir = ".tmp_apptest"

from streamlit.testing.v1 import AppTest

at = AppTest.from_file("../src/pages/4_Report.py", default_timeout=300)
at.run()
print("EXC", len(at.exception))
for e in at.exception:
    print(str(e.value)[:300])
print("expanders:", sorted({x.label for x in at.expander}))
print("charts:", len(at.get("plotly_chart")))
