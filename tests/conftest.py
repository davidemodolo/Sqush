import pytest


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: marks tests that need a GPU and model (deselect with '-m \"not slow\"')")
