"""
Standalone runner for test_session_cache.py.
Imports session_cache directly to avoid triggering config.py / DB imports.
Run: .venv\Scripts\python.exe run_session_tests.py
"""
import sys
import os

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Patch out the heavy __init__ by pre-loading session_cache directly
# before any package-level import can fire
import importlib.util

spec = importlib.util.spec_from_file_location(
    "services.chatbot.session_cache",
    os.path.join(os.path.dirname(__file__), "services", "chatbot", "session_cache.py"),
)
mod = importlib.util.load_from_spec(spec)  # type: ignore
spec.loader.exec_module(mod)  # type: ignore
sys.modules["services.chatbot.session_cache"] = mod

import unittest
loader = unittest.TestLoader()
suite  = loader.loadTestsFromName("tests.test_session_cache")
runner = unittest.TextTestRunner(verbosity=2)
result = runner.run(suite)
sys.exit(0 if result.wasSuccessful() else 1)
