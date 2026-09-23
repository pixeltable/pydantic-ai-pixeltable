"""Test isolation: tests run in their own Pixeltable catalog, never ~/.pixeltable.

Set before any test module imports pixeltable. The home is stable per pytest-xdist
worker, so later runs reuse the embedded Postgres server rather than each leaving
a new one running (each holds a shared-memory segment until stopped).
"""

import os
import tempfile

_worker = os.environ.get("PYTEST_XDIST_WORKER", "main")
os.environ["PIXELTABLE_HOME"] = os.path.join(tempfile.gettempdir(), f"pydantic-ai-pixeltable-tests-{_worker}")
