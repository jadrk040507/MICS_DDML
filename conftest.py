"""Make the DDML package importable for pytest.

The pipeline lives in 'Do file/python' (a space-named parent, so it cannot be a
dotted package path). Putting that directory on sys.path lets the test modules
use bare imports (``from config import ...``, ``from models import ...``),
matching how the run_*.py scripts import.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "Do file" / "python"))
