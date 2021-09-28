import pytest

from forte.solvers import solver_factory, HF, ActiveSpaceSolver


def test_fci_7():
    """
    Test FCI on the doublet state of CH and triplet state of CH+.
    It also verifies that computations with different values of M_S yield the same energy
    """

    ref_hf_energy = -37.43945401822133
    ref_fci_energy = -37.49081328115731

    # setup job
    xyz = """
    C
    H 1 1.0
    units bohr
    """
    input = solver_factory(molecule=xyz, basis='6-31G')
    state = input.state(charge=0, multiplicity=2, sym='b1')

    hf_doublet = HF(input, state=state, docc=[3, 0, 0, 0], socc=[0, 0, 1, 0])
    hf_doublet.run()

    # compute the FCI energy for the double B1 (M_S =  1/2) solution
    state_doublet_1 = input.state(charge=0, multiplicity=2, ms=0.5, sym='b1')
    fci_doublet_1 = ActiveSpaceSolver(hf_doublet, type='FCI', states=state_doublet_1)
    fci_doublet_1.run()

    # check results
    assert hf_doublet.value('hf energy') == pytest.approx(ref_hf_energy, 1.0e-10)
    assert fci_doublet_1.value('active space energy')[state_doublet_1] == pytest.approx([ref_fci_energy], 1.0e-10)

    # compute the FCI energy for the double B1 (M_S =  1/2) solution
    state_doublet_2 = input.state(charge=0, multiplicity=2, ms=-0.5, sym='b1')
    fci_doublet_2 = ActiveSpaceSolver(hf_doublet, type='FCI', states=state_doublet_2)
    fci_doublet_2.run()
    assert fci_doublet_2.value('active space energy')[state_doublet_2] == pytest.approx([ref_fci_energy], 1.0e-10)

    state = input.state(charge=1, multiplicity=3, sym='b1')

    ref_hf_triplet = -37.066693498042760
    ref_fci_triplet = -37.088876452204509

    hf_triplet = HF(input, state=state, docc=[2, 0, 0, 0], socc=[1, 0, 1, 0])
    hf_triplet.run()

    # compute the FCI energy for the triplet B1 (M_S =  1) solution
    state_triplet_1 = input.state(charge=1, multiplicity=3, ms=1.0, sym='b1')
    fci_triplet_1 = ActiveSpaceSolver(hf_triplet, type='FCI', states=state_triplet_1)
    fci_triplet_1.run()

    # check results
    assert hf_triplet.value('hf energy') == pytest.approx(ref_hf_triplet, 1.0e-10)
    assert fci_triplet_1.value('active space energy')[state_triplet_1] == pytest.approx([ref_fci_triplet], 1.0e-10)

    # compute the FCI energy for the triplet B1 (M_S =  0) solution
    state_triplet_2 = input.state(charge=1, multiplicity=3, ms=0.0, sym='b1')
    fci_triplet_2 = ActiveSpaceSolver(hf_triplet, type='FCI', states=state_triplet_2)
    fci_triplet_2.run()
    assert fci_triplet_2.value('active space energy')[state_triplet_2] == pytest.approx([ref_fci_triplet], 1.0e-10)

    # compute the FCI energy for the triplet B1 (M_S =  0) solution
    state_triplet_3 = input.state(charge=1, multiplicity=3, ms=-1.0, sym='b1')
    fci_triplet_3 = ActiveSpaceSolver(hf_triplet, type='FCI', states=state_triplet_3)
    fci_triplet_3.run()
    assert fci_triplet_3.value('active space energy')[state_triplet_3] == pytest.approx([ref_fci_triplet], 1.0e-10)


if __name__ == "__main__":
    test_fci_7()
