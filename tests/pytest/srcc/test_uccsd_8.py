import pytest
import forte
import psi4


def test_uccsd_8():
    """Test projective unlinked UCCSD on H4 using RHF/STO-3G orbitals"""

    ref_energy = -1.9976233094  # from Jonathon

    molecule = psi4.geometry(
        """
     H 0.0 0.0 0.0
     H 0.0 0.0 1.5
     H 0.0 0.0 3.0
     H 0.0 0.0 4.5     
    """
    )

    data = forte.modules.ObjectsUtilPsi4(molecule=molecule, basis="sto-3g").run()
    scf_energy = data.psi_wfn.energy()
    cc = forte.modules.GeneralCC(cc_type="ucc", max_exc=2, e_convergence=1.0e-10, options={"linked": False})
    data = cc.run(data)

    psi4.core.clean()

    energy = data.results.value("proj. energy")

    print(f"  HF energy:    {scf_energy}")
    print(f"  CCSD energy:  {energy}")
    print(f"  corr. energy: {energy - scf_energy}")

    assert energy == pytest.approx(ref_energy, 1.0e-6)


if __name__ == "__main__":
    test_uccsd_8()
