#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest


def test_uccsd_3():
    """Test projective UCCSD on Ne using RHF/cc-pVDZ orbitals"""

    import forte.proc.scc as scc
    import forte
    import psi4
    import os.path

    ref_energy = -107.655681875111

    data = forte.modules.OptionsFactory(options={"FROZEN_DOCC": [2]}).run()
    data = forte.modules.ObjectsFromFCIDUMP(file=os.path.dirname(__file__) + "/INTDUMP").run(data)
    data = forte.modules.ActiveSpaceInts("CORRELATED", []).run(data)
    calc_data = scc.run_cc(
        data.as_ints, data.scf_info, data.mo_space_info, cc_type="ucc", max_exc=2, e_convergence=1.0e-10
    )

    psi4.core.clean()

    energy = calc_data[-1][1]

    print(f"  UCCSD energy: {energy}")
    assert energy == pytest.approx(ref_energy, 1.0e-9)


if __name__ == "__main__":
    test_uccsd_3()
