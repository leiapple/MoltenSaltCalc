def pytest_addoption(parser):
    """Needed to pass the model name to the uMLIP test function after the venv was set up"""
    parser.addoption("--model", action="store", default=None)
