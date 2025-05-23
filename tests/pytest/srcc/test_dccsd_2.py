import pytest
import forte
import psi4


def test_dccsd_2():
    """Test CCSD on H4 using RHF/DZ orbitals"""

    ref_energy = -2.225059801642  # from psi4

    molecule = psi4.geometry(
        """
     H 0.0 0.0 0.0
     H 0.0 0.0 1.0
     H 0.0 0.0 2.0
     H 0.0 0.0 3.0     
    """
    )

    data = forte.modules.ObjectsUtilPsi4(molecule=molecule, basis="DZ", options={"E_CONVERGENCE": 11}).run()
    scf_energy = data.psi_wfn.energy()
    cc = forte.modules.GeneralCC(cc_type="dcc", max_exc=2)
    data = cc.run(data)

    psi4.core.clean()

    energy = data.results.value("energy")

    print(f"  HF energy:   {scf_energy}")
    print(f"  CCSD energy: {energy}")
    print(f"  E - Eref:    {energy - ref_energy}")

    assert energy == pytest.approx(ref_energy, 5.0e-10)


if __name__ == "__main__":
    test_dccsd_2()
