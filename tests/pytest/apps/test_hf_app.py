import pytest
import forte.apps


def test_rhf_app():
    xyz = """
    H 0.0 0.0 0.0
    H 0.0 0.0 1.0
    """
    data = forte.apps.hf(geom=xyz, basis="cc-pVDZ", state={"charge": 0, "multiplicity": 1, "sym": "ag"})
    assert data.results.value("hf energy") == pytest.approx(-1.10015376479352, 1.0e-10)


if __name__ == "__main__":
    test_rhf_app()
