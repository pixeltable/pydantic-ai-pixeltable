"""Test isolation: each pytest process gets its own Pixeltable catalog.

Set before any test module imports pixeltable, so the embedded Postgres data
directory lands in a temp dir instead of ~/.pixeltable. pytest-xdist workers are
separate processes, so parallel runs get separate catalogs too.
"""

import os
import tempfile

os.environ["PIXELTABLE_HOME"] = tempfile.mkdtemp(prefix="pxt-test-")
