import time
from contextlib import contextmanager
import pytest
import psi4
import forte
import numpy as np

timings = []


@contextmanager
def time_block(description):
    start = time.perf_counter()
    try:
        yield
    finally:
        end = time.perf_counter()
        print(f"  {description:40} {end - start:12.6f} seconds")
        timings.append((description, end - start))


def sparse_operator_correctness():
    molecule = psi4.geometry(
        """
        0 1
        Be
        """
    )

    data = forte.modules.ObjectsUtilPsi4(molecule=molecule, basis="cc-pVDZ").run()
    wfn = data.psi_wfn
    na = wfn.nalpha()
    nb = wfn.nbeta()
    nirrep = wfn.nirrep()
    wfn_symmetry = 0
    mo_symmetry = data.mo_space_info.symmetry("ALL")

    as_ints = data.as_ints
    dets = forte.hilbert_space(wfn.nmo(), na, nb, nirrep, mo_symmetry, wfn_symmetry)

    # Build the Hamiltonian
    opH = forte.sparse_operator_hamiltonian(as_ints)

    print("\n\n  Number of determinants: ", len(dets))
    print("  Number of integrals: ", len(opH))

    # Apply the Hamiltonian to a state that spans the entire Hilbert space
    c = 1 / np.sqrt(len(dets))
    state = forte.SparseState({det: c for det in dets})
    opH_state = forte.apply_op(opH, state)


def sparse_operator_timing_1():
    psi4.core.clean()
    forte.clean_options()

    molecule = psi4.geometry(
        """
        0 1
        O
        H 1 0.96
        H 1 1.04 2 104.5
        """
    )

    data = forte.modules.ObjectsUtilPsi4(molecule=molecule, basis="DZ", mo_spaces={"FROZEN_DOCC": [2, 0]}).run()
    wfn = data.psi_wfn
    na = wfn.nalpha() - data.mo_space_info.size("FROZEN_DOCC")
    nb = wfn.nbeta() - data.mo_space_info.size("FROZEN_DOCC")
    nirrep = wfn.nirrep()
    nmo = wfn.nmo()
    wfn_sym = 0
    mo_sym = data.mo_space_info.symmetry("ALL")

    as_ints = data.as_ints
    dets = forte.hilbert_space(nmo, na, nb, nirrep, mo_sym, wfn_sym)

    # Build the Hamiltonian
    print()
    spH = forte.SparseHamiltonian(as_ints)
    opH = forte.sparse_operator_hamiltonian(as_ints)

    print("\n\n  Number of determinants: ", len(dets))
    print("  Number of integrals: ", len(opH))

    # Apply the Hamiltonian to a state that spans the entire Hilbert space
    c = 1 / np.sqrt(len(dets))
    state = forte.SparseState({det: c for det in dets})
    with time_block("Time to apply SparseHamiltonian (test 1):"):
        spH_state = spH.apply(state)
    with time_block("Time to apply SparseOperator    (test 1):"):
        opH_state = forte.apply_op(opH, state)
    spH_state -= opH_state
    assert spH_state.norm() < 1e-9

    print(f"  Norm of difference: {spH_state.norm()}")


def sparse_operator_timing_2():
    psi4.core.clean()
    forte.clean_options()

    molecule = psi4.geometry(
        """
        0 1
        O
        H 1 0.96
        H 1 1.04 2 104.5
        """
    )

    data = forte.modules.ObjectsUtilPsi4(molecule=molecule, basis="DZ", mo_spaces={"FROZEN_DOCC": [2, 0]}).run()
    wfn = data.psi_wfn
    na = wfn.nalpha() - data.mo_space_info.size("FROZEN_DOCC")
    nb = wfn.nbeta() - data.mo_space_info.size("FROZEN_DOCC")
    nirrep = wfn.nirrep()
    nmo = wfn.nmo()
    wfn_sym = 0
    mo_sym = data.mo_space_info.symmetry("ALL")

    as_ints = data.as_ints
    dets = forte.hilbert_space(nmo, na, nb, nirrep, mo_sym, wfn_sym)

    # Build the Hamiltonian
    print()
    spH = forte.SparseHamiltonian(as_ints)
    opH = forte.sparse_operator_hamiltonian(as_ints)

    ndets = 7000
    print("\n\n  Number of determinants: ", ndets)
    print("  Number of integrals: ", len(opH))

    # Apply the Hamiltonian to a state that spans 7000 determinants
    c = 1 / np.sqrt(ndets)
    state = forte.SparseState({det: c for det in dets[:ndets]})
    with time_block("Time to apply SparseHamiltonian (test 2):"):
        spH_state = spH.apply(state)
    with time_block("Time to apply SparseOperator    (test 2):"):
        opH_state = forte.apply_op(opH, state)
    diff_state = spH_state - opH_state
    assert diff_state.norm() < 1e-9
    print(f"  Norm of difference: {diff_state.norm()}")

    with time_block("Time to apply SparseHamiltonian^2 (test 2):"):
        spH_state = spH.apply(spH_state)
    with time_block("Time to apply SparseOperator^2    (test 2):"):
        opH_state = forte.apply_op(opH, opH_state)
    diff_state = spH_state - opH_state
    assert diff_state.norm() < 1e-9

    print(f"  Norm of difference: {diff_state.norm()}")


def sparse_operator_timing_3():
    psi4.core.clean()
    forte.clean_options()

    molecule = psi4.geometry(
        """
        0 1
        O
        H 1 0.96
        H 1 1.04 2 104.5
        """
    )

    data = forte.modules.ObjectsUtilPsi4(molecule=molecule, basis="DZ", mo_spaces={"FROZEN_DOCC": [1, 0]}).run()
    wfn = data.psi_wfn
    na = wfn.nalpha() - data.mo_space_info.size("FROZEN_DOCC")
    nb = wfn.nbeta() - data.mo_space_info.size("FROZEN_DOCC")
    nirrep = wfn.nirrep()
    nmo = wfn.nmo()
    wfn_sym = 0
    mo_sym = data.mo_space_info.symmetry("ALL")

    as_ints = data.as_ints
    dets = forte.hilbert_space(nmo, na, nb, nirrep, mo_sym, wfn_sym)

    # Build the Hamiltonian
    print()
    spH = forte.SparseHamiltonian(as_ints)
    opH = forte.sparse_operator_hamiltonian(as_ints)

    ndets = 20000
    print("\n\n  Number of determinants: ", ndets)
    print("  Number of integrals: ", len(opH))

    # Apply the Hamiltonian to a state that spans 7000 determinants
    c = 1 / np.sqrt(ndets)
    state = forte.SparseState({det: c for det in dets[:ndets]})
    with time_block("Time to apply SparseHamiltonian (test 3):"):
        spH_state = spH.apply(state)
    with time_block("Time to apply SparseOperator    (test 3):"):
        opH_state = forte.apply_op(opH, state)
    diff_state = spH_state - opH_state
    assert diff_state.norm() < 1e-9
    print(f"  Norm of difference: {diff_state.norm()}")

    with time_block("Time to apply SparseHamiltonian^2 (test 3):"):
        spH_state = spH.apply(spH_state)
    with time_block("Time to apply SparseOperator^2    (test 3):"):
        opH_state = forte.apply_op(opH, opH_state)
    diff_state = spH_state - opH_state
    assert diff_state.norm() < 1e-9

    print(f"  Norm of difference: {diff_state.norm()}")


if __name__ == "__main__":
    sparse_operator_correctness()
    sparse_operator_timing_1()
    sparse_operator_timing_2()
    sparse_operator_timing_3()
    for timing in timings:
        print(f"  {timing[0]:50} {timing[1]:12.6f} s")
