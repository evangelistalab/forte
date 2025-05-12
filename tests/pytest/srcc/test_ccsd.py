import pytest
import forte
import psi4


def test_ccsd():
    """Test CCSD on H2 using RHF/DZ orbitals"""

    ref_energy = -1.126712715716011  # CCSD = FCI energy from psi4

    molecule = psi4.geometry(
        """
     H
     H 1 1.0
    """
    )

    data = forte.modules.ObjectsUtilPsi4(molecule=molecule, basis="DZ").run()
    cc = forte.modules.GeneralCC(cc_type="cc", max_exc=2, e_convergence=5.0e-10)
    data = cc.run(data)

    scf_energy = data.psi_wfn.energy()
    psi4.core.clean()

    energy = data.results.value("energy")

    print(f"  HF energy:   {scf_energy}")
    print(f"  CCSD energy: {energy}")
    print(f"  E - Eref:    {energy - ref_energy}")

    assert energy == pytest.approx(ref_energy, 5.0e-10)


if __name__ == "__main__":
    test_ccsd()
