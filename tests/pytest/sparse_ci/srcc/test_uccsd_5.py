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

    psi4.set_options({'FORTE__FROZEN_DOCC': [2]})
    options = forte.prepare_forte_options()
    forte_objects = forte.prepare_forte_objects_from_fcidump(options, os.path.dirname(__file__))
    state_weights_map, mo_space_info, scf_info, fcidump = forte_objects
    ints = forte.make_ints_from_fcidump(fcidump, options, mo_space_info)
    as_ints = forte.make_active_space_ints(mo_space_info, ints, 'CORRELATED', [])
    calc_data = scc.run_cc(as_ints, scf_info, mo_space_info, cc_type='ucc', max_exc=2, e_convergence=1.0e-10)

    psi4.core.clean()

    energy = calc_data[-1][1]

    print(f'  UCCSD energy: {energy}')
    assert energy == pytest.approx(ref_energy, 1.0e-9)


if __name__ == "__main__":
    test_uccsd_3()
