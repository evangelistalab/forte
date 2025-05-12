import pytest
import forte
import psi4


def test_duccsd():
    """Test projective factorized UCCSDT on Ne using RHF/cc-pVDZ orbitals"""

    ref_energy = -128.677997285129
    molecule = psi4.geometry("Ne 0.0 0.0 0.0")

    data = forte.modules.ObjectsUtilPsi4(
        molecule=molecule, basis="cc-pVDZ", mo_spaces={"FROZEN_DOCC": [1, 0, 0, 0, 0, 0, 0, 0]}
    ).run()

    scf_energy = data.psi_wfn.energy()
    cc = forte.modules.GeneralCC(cc_type="ducc", max_exc=2, e_convergence=1.0e-10)
    data = cc.run(data)

    psi4.core.clean()

    energy = data.results.value("energy")

    print(f"  HF energy:     {scf_energy}")
    print(f"  DUCCSD energy: {energy}")
    print(f"  E - Eref:      {energy - ref_energy}")

    assert energy == pytest.approx(ref_energy, 5.0e-10)


if __name__ == "__main__":
    test_duccsd()
