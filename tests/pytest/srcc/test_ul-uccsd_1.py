#!/usr/bin/env python
# -*- coding: utf-8 -*-


def test_ul_uccsd_1():
    """Test projective unlinked UCCSD on H4 using RHF/STO-3G orbitals"""

    import pytest
    import forte.proc.scc as scc
    import forte
    import psi4
    import os.path

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

    # data = forte.modules.ObjectsUtilPsi4(molecule=molecule, basis="sto-3g").run()
    # scf_energy = data.psi_wfn.energy()

    cc = forte.modules.GeneralCC(cc_type="ucc", max_exc=2, e_convergence=1.0e-10, options={"linked": False, "maxk": 1})
    data = cc.run(data)

    psi4.core.clean()

    energy = data.results.value("energy")

    # calc_data = scc.run_cc(
    #     data.as_ints,
    #     data.scf_info,
    #     data.mo_space_info,
    #     cc_type="ucc",
    #     max_exc=2,
    #     e_convergence=1.0e-10,
    #     linked=False,
    #     diis_start=2,
    # )

    # psi4.core.clean()

    # energy = calc_data[-1][2]

    assert energy == pytest.approx(ref_energy, 1.0e-6)


if __name__ == "__main__":
    test_ul_uccsd_1()
