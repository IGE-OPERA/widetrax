"""Make the repository root importable so that `import widetrax` works in tests."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
