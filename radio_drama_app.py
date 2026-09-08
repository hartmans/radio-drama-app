#!/usr/bin/env python
"""Run the renderer from the source tree beside this helper script."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from radio_drama.cli import main


if __name__ == "__main__":
    main()
