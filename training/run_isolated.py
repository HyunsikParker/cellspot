"""Run a training script under `python -I` (no user site-packages, no current directory on the path): the XBAT+ files
are downloaded data, so nothing next to them should be importable.

usage: python -I run_isolated.py SCRIPT [ARGS ...]
"""
import runpy
import sys
from pathlib import Path

here = Path(__file__).resolve().parent
sys.path.insert(0, str(here))
script = here / sys.argv[1]
sys.argv = [str(script), *sys.argv[2:]]
runpy.run_path(str(script), run_name="__main__")
