import psi4
import forte
import pytest


def test_scfinfo():
    """Test the SCFInfo class python API"""

    ref_energy = -99.50300245245828

    geom = """
    1 2
    H 0.0 0.0 0.0
    F 0.0 0.0 1.0
    """

    psi4.core.clean()

    mol = psi4.geometry(geom)
    psi4.set_options(
        {
            'basis': 'cc-pVDZ',
            'scf_type': 'pk',
            'reference': 'uhf',
            'docc': [3, 0, 1, 0],
            'socc': [0, 0, 0, 1]
        }
    )
    _, wfn = psi4.energy('scf', return_wfn=True, molecule=mol)

    # create an SCFInfo object from the psi4 wavefunction
    scfinfo = forte.SCFInfo(wfn)

    assert tuple(scfinfo.nmopi()) == (10, 1, 4, 4)
    assert tuple(scfinfo.doccpi()) == (3, 0, 1, 0)
    assert tuple(scfinfo.soccpi()) == (0, 0, 0, 1)
    assert scfinfo.reference_energy() == pytest.approx(ref_energy, 1.0e-10)
    assert scfinfo.epsilon_a().nph[0][0] == pytest.approx(-26.97737351, 1.0e-6)
    assert scfinfo.epsilon_b().nph[0][0] == pytest.approx(-26.92368455, 1.0e-6)


if __name__ == "__main__":
    test_scfinfo()
