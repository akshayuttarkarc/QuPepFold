#!/usr/bin/env python3
"""QuPepFold CLI runner script.

Usage:
    python3 ./run_fold.py fold --seq KHANAKPFEVPFLKF --out ./results
    python3 ./run_fold.py fold --seq APRLRFY --backend aer --spsa-iters 50 --out ./output
"""

import sys
import os

# Add src to path for development
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from qupepfold.cli import main

if __name__ == "__main__":
    main()
