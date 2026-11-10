"""Pick which implementation the tests import.

    pytest                         # tests YOUR code in src/
    WPO_IMPL=solutions pytest      # tests the reference solutions
"""
import os
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
IMPL = os.environ.get("WPO_IMPL", "src")
sys.path.insert(0, str(ROOT / IMPL))
