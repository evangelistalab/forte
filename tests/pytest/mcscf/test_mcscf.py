import pytest
from forte.solvers import HF, ActiveSpaceSolver, MCSCF, input_factory


def test_mcscf():

    ref_mcscf_energy = -1.127298184169
    xyz = """
    H 0.0 0.0 0.0
    H 0.0 0.0 1.0
    """

    input = input_factory(molecule=xyz, basis='cc-pVDZ', int_type='DF', jkfit_aux_basis='cc-pVDZ-RI')
    state = input.state(charge=0, multiplicity=1, sym='ag')
    mo_spaces = input.mo_spaces(active=[1, 0, 0, 0, 0, 1, 0, 0])

    # create a HF object
    hf = HF(input, state=state)
    fci = ActiveSpaceSolver(hf, type='FCI', states=state, mo_spaces=mo_spaces)
    # pass the FCI object to MCSCF
    mcscf = MCSCF(fci, e_convergence=1.0e-11)  # <- use information in fci to get active space, etc.
    # run the computation
    mcscf.run()

    assert mcscf.value('mcscf energy')[state] == pytest.approx([ref_mcscf_energy], 1.0e-10)


def test_mcscf_detci():

    ref_mcscf_energy = -1.127298184169
    xyz = """
    H 0.0 0.0 0.0
    H 0.0 0.0 1.0
    """

    input = input_factory(molecule=xyz, basis='cc-pVDZ', int_type='DF', jkfit_aux_basis='cc-pVDZ-RI')
    state = input.state(charge=0, multiplicity=1, sym='ag')

    # create a HF object
    hf = HF(input, state=state)
    detci = ActiveSpaceSolver(hf, type='DETCI', states=state, mo_spaces={'ACTIVE': [1, 0, 0, 0, 0, 1, 0, 0]})
    # pass the DETCI object to MCSCF
    mcscf = MCSCF(detci, e_convergence=1.0e-11)  # <- use information in fci to get active space, etc.
    # run the computation
    mcscf.run()

    assert mcscf.value('mcscf energy')[state] == pytest.approx([ref_mcscf_energy], 1.0e-10)


def test_mcscf_aci():

    ref_mcscf_energy = -1.127298184169
    xyz = """
    H 0.0 0.0 0.0
    H 0.0 0.0 1.0
    """

    input = input_factory(molecule=xyz, basis='cc-pVDZ', int_type='DF', jkfit_aux_basis='cc-pVDZ-RI')
    state = input.state(charge=0, multiplicity=1, sym='ag')

    # create a HF object
    hf = HF(input, state=state)
    aci = ActiveSpaceSolver(
        hf, type='ACI', states=state, mo_spaces={'ACTIVE': [1, 0, 0, 0, 0, 1, 0, 0]}, options={'sigma': 0.0}
    )
    # pass the ACI object to MCSCF
    aci.run()
    mcscf = MCSCF(aci, e_convergence=1.0e-11)  # <- use information in fci to get active space, etc.
    # run the computation
    mcscf.run()

    assert mcscf.value('mcscf energy')[state] == pytest.approx([ref_mcscf_energy], 1.0e-10)


if __name__ == "__main__":
    test_mcscf()
    test_mcscf_detci()
    test_mcscf_aci()
