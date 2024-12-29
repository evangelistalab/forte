import os.path
import pytest
import forte
import psi4


def test_ul_uccsd_1():
    """Test projective unlinked UCCSD on H4 using RHF/STO-3G orbitals"""

    ref_energy = -1.9437216535661626  # from Jonathon

    psi4.set_options(
        {
            "FORTE__FCIDUMP_DOCC": [2],
            "FORTE__FROZEN_DOCC": [0],
        }
    )

    data = forte.modules.OptionsFactory().run()
    data = forte.modules.ObjectsFromFCIDUMP(file=os.path.dirname(__file__) + "/INTDUMP2").run(data)
    data = forte.modules.ActiveSpaceInts("CORRELATED", []).run(data)

    cc = forte.modules.GeneralCC(
        cc_type="ucc", max_exc=2, e_convergence=1.0e-10, options={"linked": False, "diis_start": 2}
    )
    data = cc.run(data)

    psi4.core.clean()

    energy = data.results.value("proj. energy")

    assert energy == pytest.approx(ref_energy, 1.0e-6)


if __name__ == "__main__":
    test_ul_uccsd_1()
