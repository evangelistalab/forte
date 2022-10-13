import pytest
from forte.solvers import HF, ActiveSpaceSolver, MCSCF, input_factory


def test_mcscf_nofc():
    """
    This tests that the MCSCF procedure with no frozen or restricted orbitals
    """

    ref_mcscf_energy = -76.073770507913
    xyz = """
    O
    H  1 1.00
    H  1 1.00 2 104.5
    """

    input = input_factory(molecule=xyz, basis='6-31g**')
    state = input.state(charge=0, multiplicity=1, sym='a1')
    mo_spaces = input.mo_spaces(active=[4, 0, 1, 2])

    hf = HF(input, state=state)
    fci = ActiveSpaceSolver(hf, type='FCI', states=state, mo_spaces=mo_spaces)
    mcscf = MCSCF(fci)
    mcscf.run()
    assert mcscf.value('mcscf energy')[state] == pytest.approx([ref_mcscf_energy], 1.0e-10)


def test_mcscf_rd():
    """
    This tests that the MCSCF procedure will optimize the restricted doubly occupied orbitals
    """
    ref_mcscf_energy = -76.073667296758
    xyz = """
    O
    H  1 1.00
    H  1 1.00 2 104.5
    """

    input = input_factory(molecule=xyz, basis='6-31g**')
    state = input.state(charge=0, multiplicity=1, sym='a1')
    mo_spaces = input.mo_spaces(restricted_docc=[1, 0, 0, 0], active=[3, 0, 1, 2])

    hf = HF(input, state=state)
    fci = ActiveSpaceSolver(hf, type='FCI', states=state, mo_spaces=mo_spaces)
    mcscf = MCSCF(fci)
    mcscf.run()
    assert mcscf.value('mcscf energy')[state] == pytest.approx([ref_mcscf_energy], 1.0e-10)


def test_mcscf_default_fc():
    """
    This tests for the default behaviour of treating the frozen doubly occupied as restricted docc.
    The MCSCF procedure will optimize the restricted doubly occupied orbitals.
    This is the desired behaviour the majority of the time
    """

    ref_mcscf_energy = -76.07366729675971
    xyz = """
    O
    H  1 1.00
    H  1 1.00 2 104.5
    """

    input = input_factory(molecule=xyz, basis='6-31g**')
    state = input.state(charge=0, multiplicity=1, sym='a1')
    mo_spaces = input.mo_spaces(frozen_docc=[1, 0, 0, 0], active=[3, 0, 1, 2])

    hf = HF(input, state=state)
    fci = ActiveSpaceSolver(hf, type='FCI', states=state, mo_spaces=mo_spaces)
    mcscf = MCSCF(fci)
    mcscf.run()
    assert mcscf.value('mcscf energy')[state] == pytest.approx([ref_mcscf_energy], 1.0e-10)


def test_mcscf_fc():
    """
    This tests that orbitals specified by frozen_docc are not optimized by MCSCF
    """
    ref_mcscf_energy = -76.073576310057
    xyz = """
    O
    H  1 1.00
    H  1 1.00 2 104.5
    """

    input = input_factory(molecule=xyz, basis='6-31g**')
    state = input.state(charge=0, multiplicity=1, sym='a1')
    mo_spaces = input.mo_spaces(frozen_docc=[1, 0, 0, 0], active=[3, 0, 1, 2])

    hf = HF(input, state=state)
    fci = ActiveSpaceSolver(hf, type='FCI', states=state, mo_spaces=mo_spaces)
    mcscf = MCSCF(fci, freeze_core=True)
    mcscf.run()
    assert mcscf.value('mcscf energy')[state] == pytest.approx([ref_mcscf_energy], 1.0e-10)


def test_mcscf_nofc_detci():
    """
    This tests that the MCSCF procedure with no frozen or restricted orbitals
    """

    ref_mcscf_energy = -76.073770507913
    xyz = """
    O
    H  1 1.00
    H  1 1.00 2 104.5
    """

    input = input_factory(molecule=xyz, basis='6-31g**')
    state = input.state(charge=0, multiplicity=1, sym='a1')
    mo_spaces = input.mo_spaces(active=[4, 0, 1, 2])

    hf = HF(input, state=state)
    fci = ActiveSpaceSolver(hf, type='DETCI', states=state, mo_spaces=mo_spaces)
    mcscf = MCSCF(fci)
    mcscf.run()
    assert mcscf.value('mcscf energy')[state] == pytest.approx([ref_mcscf_energy], 1.0e-10)


def test_mcscf_rd_detci():
    """
    This tests that the MCSCF procedure will optimize the restricted doubly occupied orbitals
    """
    ref_mcscf_energy = -76.073667296758
    xyz = """
    O
    H  1 1.00
    H  1 1.00 2 104.5
    """

    input = input_factory(molecule=xyz, basis='6-31g**')
    state = input.state(charge=0, multiplicity=1, sym='a1')
    mo_spaces = input.mo_spaces(restricted_docc=[1, 0, 0, 0], active=[3, 0, 1, 2])

    hf = HF(input, state=state)
    fci = ActiveSpaceSolver(hf, type='DETCI', states=state, mo_spaces=mo_spaces)
    mcscf = MCSCF(fci)
    mcscf.run()
    assert mcscf.value('mcscf energy')[state] == pytest.approx([ref_mcscf_energy], 1.0e-10)


def test_mcscf_default_fc_detci():
    """
    This tests for the default behaviour of treating the frozen doubly occupied as restricted docc.
    The MCSCF procedure will optimize the restricted doubly occupied orbitals.
    This is the desired behaviour the majority of the time
    """

    ref_mcscf_energy = -76.07366729675971
    xyz = """
    O
    H  1 1.00
    H  1 1.00 2 104.5
    """

    input = input_factory(molecule=xyz, basis='6-31g**')
    state = input.state(charge=0, multiplicity=1, sym='a1')
    mo_spaces = input.mo_spaces(frozen_docc=[1, 0, 0, 0], active=[3, 0, 1, 2])

    hf = HF(input, state=state)
    fci = ActiveSpaceSolver(hf, type='DETCI', states=state, mo_spaces=mo_spaces)
    mcscf = MCSCF(fci)
    mcscf.run()
    assert mcscf.value('mcscf energy')[state] == pytest.approx([ref_mcscf_energy], 1.0e-10)


def test_mcscf_fc_detci():
    """
    This tests that orbitals specified by frozen_docc are not optimized by MCSCF
    """
    ref_mcscf_energy = -76.073576310057
    xyz = """
    O
    H  1 1.00
    H  1 1.00 2 104.5
    """

    input = input_factory(molecule=xyz, basis='6-31g**')
    state = input.state(charge=0, multiplicity=1, sym='a1')
    mo_spaces = input.mo_spaces(frozen_docc=[1, 0, 0, 0], active=[3, 0, 1, 2])

    hf = HF(input, state=state)
    fci = ActiveSpaceSolver(hf, type='DETCI', states=state, mo_spaces=mo_spaces)
    mcscf = MCSCF(fci, freeze_core=True)
    mcscf.run()
    assert mcscf.value('mcscf energy')[state] == pytest.approx([ref_mcscf_energy], 1.0e-10)


if __name__ == "__main__":
    test_mcscf_nofc()
    test_mcscf_rd()
    test_mcscf_default_fc()
    test_mcscf_fc()
    test_mcscf_nofc_detci()
    test_mcscf_rd_detci()
    test_mcscf_default_fc_detci()
    test_mcscf_fc_detci()
