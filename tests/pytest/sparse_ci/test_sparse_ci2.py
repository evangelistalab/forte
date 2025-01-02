import psi4
import forte
import pytest


def test_sparse_ci2():

    ref_fci = -5.623851783330647

    psi4.core.clean()
    forte.clean_options()

    molecule = psi4.geometry(
        """
     He
     He 1 1.0
    """
    )

    wfn_sym = 0

    data = forte.modules.ObjectsUtilPsi4(molecule=molecule, basis="cc-pVDZ").run()
    wfn = data.psi_wfn
    na = wfn.nalpha()
    nb = wfn.nbeta()
    as_ints = data.as_ints

    wfn_sym = 0
    print("\n\n  => Sparse FCI Test <=")

    # Get the symmetry information of the MOs
    nirrep = wfn.nirrep()
    nmo = wfn.nmo()
    mo_sym = data.mo_space_info.symmetry("ALL")
    print(f"  Number of irreps: {nirrep}")
    print("  Symmetry of the MOs: ", mo_sym)

    # Generate the determinant basis using a utility function
    dets = forte.hilbert_space(nmo, na, nb, nirrep, mo_sym, wfn_sym)
    print(f"\n  Size of the derminant basis: {len(dets)}")

    # Diagonalize the Hamiltonian
    energy, evals, evecs, spin = forte.diag(dets, as_ints, 1, 1, "FULL")

    efci = energy[0] + as_ints.nuclear_repulsion_energy()
    print(f"\n\n  FCI Energy: {efci}")
    print(f"  FCI Energy: {ref_fci} (reference)")

    assert efci == pytest.approx(ref_fci, abs=1e-9)
    print("\n  Test passed!")


if __name__ == "__main__":
    test_sparse_ci2()
