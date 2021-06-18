"""Test the HF solver."""

import pytest

from forte.solvers import solver_factory, HF, CallbackHandler


def test_hf_callback():
    """Example of using a callback to check if UHF broke alpha/beta symmetry."""

    xyz = """
    H 0.0 0.0 0.0
    H 0.0 0.0 1.0
    """
    root = solver_factory(molecule=xyz, basis='cc-pVDZ')
    state = root.state(charge=0, multiplicity=1, sym='ag')

    cbh = CallbackHandler()

    def check_broken(cb, state):
        """Check if the MOs broke symmetry by computing |Ca - Cb|"""
        Ca = state.data.psi_wfn.Ca().clone()
        Cb = state.data.psi_wfn.Cb().clone()
        Ca.axpy(-1.0, Cb)
        # here we store the value so that it can be refererenced later
        cb.report = Ca.rms()

    cbh.add_callback('post hf', check_broken)
    hf = HF(root, state=state, restricted=False, cbh=cbh)
    hf.run()
    assert cbh.report('post hf') == 0.0


if __name__ == "__main__":
    test_hf_callback()
