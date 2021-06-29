from forte.solvers import solver_factory, HF, CallbackHandler


def test_hf_callback():
    """Example of using a callback to localize HF MOs."""

    xyz = """
    H 0.0 0.0 0.0
    H 0.0 0.0 1.0
    H 0.0 0.0 2.0
    H 0.0 0.0 3.0
    symmetry c1
    """
    input = solver_factory(molecule=xyz, basis='sto-3g')
    state = input.state(charge=0, multiplicity=1)

    cbh = CallbackHandler()

    def localize(cb, state):
        """Localize the orbitals after a HF computation"""
        import psi4
        wfn = state.data.psi_wfn
        basis_ = wfn.basisset()
        C = wfn.Ca_subset("AO", "ALL")
        Local = psi4.core.Localizer.build("PIPEK_MEZEY", basis_, C)
        Local.localize()
        new_C = Local.L
        wfn.Ca().copy(new_C)
        wfn.Cb().copy(new_C)

    cbh.add_callback('post hf', localize)
    hf = HF(input, state=state, restricted=False, cbh=cbh)
    hf.run()


if __name__ == "__main__":
    test_hf_callback()
