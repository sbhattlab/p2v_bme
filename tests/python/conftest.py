"""pytest setup shared by tests/python: the gpu marker."""


def pytest_configure(config):
    config.addinivalue_line("markers", "gpu: needs a GPU (ctest runs these as cli_gpu)")
