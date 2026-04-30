"""Needed to pass the model name to the uMLIP test function after the venv was set up."""


def pytest_addoption(parser):
    """Add the model name to the test function."""
    parser.addoption("--model", action="store", default=None)
