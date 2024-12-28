import os.path
import pytest
import forte
import psi4


def test_uccsd_3():
    """Test projective UCCSD on Ne using RHF/cc-pVDZ orbitals"""

    ref_energy = -107.655681875111

    data = forte.modules.OptionsFactory(options={"FROZEN_DOCC": [2]}).run()
    data = forte.modules.ObjectsFromFCIDUMP(file=os.path.dirname(__file__) + "/INTDUMP").run(data)
    data = forte.modules.ActiveSpaceInts("CORRELATED", []).run(data)
    cc = forte.modules.GeneralCC(cc_type="ucc", max_exc=2, e_convergence=1.0e-10)

    data = cc.run(data)

    psi4.core.clean()

    energy = data.results.value("energy")

    print(f"  UCCSD energy: {energy}")
    assert energy == pytest.approx(ref_energy, 1.0e-9)


if __name__ == "__main__":
    test_uccsd_3()
