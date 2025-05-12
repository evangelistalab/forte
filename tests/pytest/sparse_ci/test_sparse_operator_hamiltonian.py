import math
import psi4
import forte
import itertools
import numpy as np
import pytest
from forte import forte_options
from forte import ndarray_from_numpy


def test_sparse_operator_hamiltonian():
    psi4.core.clean()
    # need to clean the options otherwise this job will interfere
    forte.clean_options()

    molecule = psi4.geometry(
        """
     He
    """
    )

    data = forte.modules.ObjectsUtilPsi4(molecule=molecule, basis="cc-pVDZ").run()
    wfn = data.psi_wfn
    na = wfn.nalpha()
    nb = wfn.nbeta()
    nirrep = wfn.nirrep()
    wfn_symmetry = 0

    as_ints = data.as_ints

    hf_reference = forte.det("2")
    print(f"\n  Hartree-Fock determinant: {hf_reference.str(10)}")

    # Compute the HF energy using slater rules
    hf_energy = as_ints.nuclear_repulsion_energy() + as_ints.slater_rules(hf_reference, hf_reference)

    # Compute the energy using the sparse operator
    hf_state = forte.SparseState({hf_reference: 1.0})
    ham = forte.sparse_operator_hamiltonian(as_ints)
    Href = forte.apply_op(ham, hf_state)
    energy = forte.overlap(hf_state, Href)

    print(f"  Reference energy:                {hf_energy}")
    print(f"  Energy (via as_ints):            {energy}")

    assert energy == pytest.approx(hf_energy, abs=1e-9)

    scalar = as_ints.nuclear_repulsion_energy()
    nmo = as_ints.nmo()
    oei_a = np.zeros((nmo,) * 2)
    oei_b = np.zeros((nmo,) * 2)
    tei_aa = np.zeros((nmo,) * 4)
    tei_ab = np.zeros((nmo,) * 4)
    tei_bb = np.zeros((nmo,) * 4)
    for p in range(nmo):
        for q in range(nmo):
            oei_a[p, q] = as_ints.oei_a(p, q)
            oei_b[p, q] = as_ints.oei_b(p, q)
            for r in range(nmo):
                for s in range(nmo):
                    tei_aa[p, q, r, s] = as_ints.tei_aa(p, q, r, s)
                    tei_ab[p, q, r, s] = as_ints.tei_ab(p, q, r, s)
                    tei_bb[p, q, r, s] = as_ints.tei_bb(p, q, r, s)
    ham = forte.sparse_operator_hamiltonian(scalar,
                                            ndarray_from_numpy(oei_a), ndarray_from_numpy(oei_b),
                                            ndarray_from_numpy(tei_aa), ndarray_from_numpy(tei_ab),
                                            ndarray_from_numpy(tei_bb))
    
    Href = forte.apply_op(ham, hf_state)
    energy = forte.overlap(hf_state, Href)

    print(f"  Reference energy:                {hf_energy}")
    print(f"  Energy (via ndarray integrals):  {energy}")
    assert energy == pytest.approx(hf_energy, abs=1e-9)

if __name__ == "__main__":
    test_sparse_operator_hamiltonian()
