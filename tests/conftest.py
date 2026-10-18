import pytest

from agri.synthetic import generate


@pytest.fixture(scope="session")
def bundle():
    return generate()


@pytest.fixture(scope="session")
def corn_report(bundle):
    from agri.pipeline import analyse_contract

    return analyse_contract(bundle, "corn")
