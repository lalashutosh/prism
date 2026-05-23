"""
conftest.py — project-root pytest configuration.

Adds the project root to sys.path so that absolute imports such as
`from core.types import ...` work from any test file.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
