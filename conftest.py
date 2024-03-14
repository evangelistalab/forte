import pytest

@pytest.fixture(scope="function", autouse=True)
def set_up():
    import psi4
    import forte

    forte.clean_options()
