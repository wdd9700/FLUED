"""
conftest.py — ensure project root is on sys.path so tests can import
the flued, bpe_baseline, and blt_baseline packages without installing them.
"""

import os
import sys

# Add the project root directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
