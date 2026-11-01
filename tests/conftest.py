from __future__ import annotations

import sys
from pathlib import Path


RELEASE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = RELEASE_ROOT.parent
for path in (
    RELEASE_ROOT / "src",
    RELEASE_ROOT / "vendor",
    PROJECT_ROOT / "code" / "models" / "nnUNet",
):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
