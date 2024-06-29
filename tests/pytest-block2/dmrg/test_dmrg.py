import pytest

from forte.solvers import solver_factory, HF, ActiveSpaceSolver


def test_dmrg():
    """Test DMRG on N2/STO-3G"""

    ref_hf_energy = -107.49650051193090
    ref_dmrg_energy = -107.65412244781278

    # setup job
    xyz = """
    N 0.0 0.0 0.0
    N 0.0 0.0 1.1
    """
    input = solver_factory(molecule=xyz, basis='sto-3g')
    state = input.state(charge=0, multiplicity=1, sym='ag')
    hf = HF(input, state=state)
    dmrg = ActiveSpaceSolver(hf, type='BLOCK2', states=state, options={'block2_sweep_davidson_tols': [1E-15]})
    dmrg.run()

    # check results
    assert hf.value('hf energy') == pytest.approx(ref_hf_energy, 1.0e-10)
    assert dmrg.value('active space energy')[state] == pytest.approx([ref_dmrg_energy], 1.0e-10)


if __name__ == "__main__":
    test_dmrg()
