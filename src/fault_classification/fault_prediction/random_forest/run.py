from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]
SRC_DIR = PROJECT_ROOT / "src"
SCRIPT_DIR = Path(__file__).resolve().parent
PACKAGE_DIR = SCRIPT_DIR.parent
for path in (str(SCRIPT_DIR), str(PACKAGE_DIR)):
    if path in sys.path:
        sys.path.remove(path)
sys.path.insert(0, str(SRC_DIR))

from fault_classification.fault_prediction.random_forest.main import main  # noqa: E402


if __name__ == "__main__":
    main()
