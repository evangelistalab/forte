import itertools
import functools
import time
import math
import copy

import numpy as np

import forte
import forte.utils


def run_cc(
    as_ints,
    scf_info,
    mo_space_info,
    cc_type=None,
    select_type=None,
    max_exc=None,
    omega=None,
    e_convergence=1.0e-10,
    r_convergence=1.0e-5,
    compute_threshold=1.0e-14,
    selection_threshold=1.0e-14,
    on_the_fly=False,
    linked=True,
    maxk=19,
    diis_start=3,
):
    """This function implements various CC methods

    Parameters
    ----------
    as_ints : ActiveSpaceIntegrals
        the molecular integrals
    mo_space_info : MOSpaceInfo
        the information about the molecular spaces
    scf_info : SCFInfo
        Information about the number of alpha/beta electrons
    cc_type : str
        The type of CC computation (cc/ucc/dcc/ducc)
    select_type : str
        The selection algorithm (default: None = no selection)
    max_exc : int
        The maximum excitation level (default: None)
    omega : float
        The selection threshold (default: None)
    e_convergence : float
        The energy convergence criterion (default = 1.0e-10)
    r_convergence : float
        The residual convergence criterion (default = 1.0e-5)
    compute_threshold : float
        The compute cutoff (default = 1.0e-14)
    selection_threshold : float
        The selection cutoff (default = 1.0e-14)
    on_the_fly : bool
        Use the on-the-fly algorithms? By default this code uses a caching algorithm
        that can be faster when using very small compute threshold values (default = False)
    Returns
    -------
    list(tuple(int,float))
        a list containing pairs of numbers that report the number of operators and the energy at each macroiteration
    """

    nirrep = mo_space_info.nirrep()

    if cc_type == None:
        raise ValueError("No type of CC computation was selected. Specify a valid value for the cc_type option.")

    nmo = mo_space_info.size("CORRELATED")
    nmopi = mo_space_info.dimension("CORRELATED").to_tuple()
    nfrzdocc = mo_space_info.dimension("FROZEN_DOCC").to_tuple()
    # the number of alpha electrons per irrep
    naelpi = (scf_info.doccpi() + scf_info.soccpi() - mo_space_info.dimension("FROZEN_DOCC")).to_tuple()
    # the number of beta electrons per irrep
    nbelpi = (scf_info.doccpi() - mo_space_info.dimension("FROZEN_DOCC")).to_tuple()

    print(f"Number of frozen orbitals per irrep:     {nfrzdocc}")
    print(f"Number of correlated orbitals per irrep: {nmopi}")
    print(f"Number of alpha electrons per irrep:     {naelpi}")
    print(f"Number of beta electrons per irrep:      {nbelpi}")

    # if not provided, define the maximum excitation level to be FCI
    if max_exc is None:
        max_exc = min(naelpi + nbelpi, nmo - naelpi + nbelpi)

    antihermitian = (cc_type != "cc") and (cc_type != "dcc")

    # create the operator pool
    op, denominators = make_cluster_operator(max_exc, naelpi, mo_space_info, scf_info)
    selected_op = forte.SparseOperator()

    # the list of operators selected from the full list
    if select_type is None:
        op_pool = list(range(op.size()))
        t = [0.0] * op.size()
        print(f"\n The excitation operator pool contains {op.size()} elements")
    else:
        raise RuntimeError("Selected CC methods are not implemented yet")
        print(f"\n Selecting operators using the {selec_type} scheme")
        t = []
        op_pool = []
        print(f"\n The selected operator pool contains {selected_op.size()} elements")

    old_e = 0.0
    start = time.time()

    hfref = make_hfref(naelpi, nbelpi, nmopi)
    eref = as_ints.slater_rules(hfref, hfref) + as_ints.nuclear_repulsion_energy()
    print(f"Reference determinant: {hfref.str(nmo)}")
    print(f"Energy of the reference determinant: {eref}")

    ref = forte.StateVector({hfref: 1.0})

    nops_old = 0

    sum_res_eval = 0
    max_macro_iter = 100
    calc_data = []

    print("=========================================================================")
    print("     Iter.         Energy       Delta Energy      Res. Norm       Time")
    print("                    (Eh)             (Eh)           (Eh)           (s)")
    print("-------------------------------------------------------------------------")

    for macro_iter in range(max_macro_iter):

        # solve the cc equations and update the amplitudes
        t, e, e_proj, micro_iter, timing = solve_cc_equations(
            cc_type,
            t,
            op,
            op,
            op_pool,
            denominators,
            ref,
            as_ints,
            compute_threshold,
            e_convergence,
            r_convergence,
            on_the_fly,
            linked,
            maxk,
            diis_start,
        )

        print(
            f"\n -> {macro_iter:4d} {e:20.12f}   {e - old_e:6e}                   {time.time() - start:8.3f}",
            flush=True,
        )

        nops = selected_op.size()

        calc_data.append((nops, e, e_proj))

        if nops_old == nops:
            break

        old_e = e
        nops_old = selected_op.size()

    print("=========================================================================")

    print(f"{cc_type.upper()} energy:           {e:20.12f}  Eh (Nops = {np.count_nonzero(t):8d})")
    print(f"{cc_type.upper()} corr. energy:     {e - eref:20.12f}")
    print(f"\n{cc_type.upper()} proj. energy:     {e_proj:20.12f}")
    print(f"{cc_type.upper()} proj. corr energy:{e_proj - eref:20.12f}")
    print(f"omega: {omega}")
    print(f"{timing}")

    return calc_data


def make_hfref(naelpi, nbelpi, nmopi):
    """Make the Hartree-Fock reference determinant

    Parameters
    ----------
    naelpi : psi4 Dimension
        The number of alpha electrons per irrep
    nbelpi : psi4 Dimension
        The number of beta electrons per irrep
    nmopi : psi4 Dimension
        The number of orbitals per irrep
    Returns
    -------
    Determinant
        The Hartree-Fock determinant
    """

    hfref = forte.Determinant()
    nirrep = len(nmopi)
    nmo = sum(nmopi)
    # we loop over each irrep and fill the occupied orbitals
    irrep_start = [sum(nmopi[:h]) for h in range(nirrep)]
    for h in range(nirrep):
        for i in range(naelpi[h]):
            hfref.set_alfa_bit(irrep_start[h] + i, True)
        for i in range(nbelpi[h]):
            hfref.set_beta_bit(irrep_start[h] + i, True)

    return hfref


def make_cluster_operator(max_exc, naelpi, mo_space_info, psi4_wfn):
    """Make the full cluster operator truncated to a given maximum excitation level (closed-shell case)

    Parameters
    ----------
    antihermitian : bool
        Is this operator antihermitian?
    max_exc : int
        The maximum excitation level (defaul: None)
    naelpi : psi4 Dimension
        The number of alpha electrons per irrep
    psi4_wfn : psi4 Wavefunction
        A psi4 Wavefunction object (to read the number of alpha/beta electrons)
    Returns
    -------
    tuple(SparseOperator, list)
        The cluster operator and a list of corresponding denominators
    """

    # Prepare the cluster operator (closed-shell case)

    nirrep = mo_space_info.nirrep()
    nmopi = mo_space_info.dimension("CORRELATED").to_tuple()

    irrep_start = [sum(nmopi[:h]) for h in range(nirrep)]

    occ_orbs = []
    vir_orbs = []
    for h in range(nirrep):
        for i in range(naelpi[h]):
            occ_orbs.append(irrep_start[h] + i)
        for i in range(naelpi[h], nmopi[h]):
            vir_orbs.append(irrep_start[h] + i)

    print(f"Occupied orbitals:    {occ_orbs}")
    print(f"Virtual orbitals:     {vir_orbs}")

    # get the symmetry of each active orbital
    symmetry = mo_space_info.symmetry("CORRELATED")

    sop = forte.SparseOperator()

    active_to_all = mo_space_info.absolute_mo("CORRELATED")

    ea = [psi4_wfn.epsilon_a().get(i) for i in active_to_all]
    eb = [psi4_wfn.epsilon_b().get(i) for i in active_to_all]

    denominators = []

    for n in range(1, max_exc + 1):
        for na in range(n + 1):
            nb = n - na
            for ao in itertools.combinations(occ_orbs, na):
                for av in itertools.combinations(vir_orbs, na):
                    for bo in itertools.combinations(occ_orbs, nb):
                        for bv in itertools.combinations(vir_orbs, nb):
                            aocc_sym = functools.reduce(lambda x, y: x ^ symmetry[y], ao, 0)
                            avir_sym = functools.reduce(lambda x, y: x ^ symmetry[y], av, 0)
                            bocc_sym = functools.reduce(lambda x, y: x ^ symmetry[y], bo, 0)
                            bvir_sym = functools.reduce(lambda x, y: x ^ symmetry[y], bv, 0)
                            # make sure the operators are total symmetric
                            if (aocc_sym ^ avir_sym) ^ (bocc_sym ^ bvir_sym) == 0:
                                # Create a list of tuples (creation, alpha, orb) where
                                #   creation : bool (true = creation, false = annihilation)
                                #   alpha    : bool (true = alpha, false = beta)
                                #   orb      : int  (the index of the mo)
                                op = []
                                for i in ao:
                                    op.append((False, True, i))
                                for i in bo:
                                    op.append((False, False, i))
                                for a in reversed(bv):
                                    op.append((True, False, a))
                                for a in reversed(av):
                                    op.append((True, True, a))

                                sop.add_term(op, 0.0)

                                e_aocc = functools.reduce(lambda x, y: x + ea[y], ao, 0.0)
                                e_avir = functools.reduce(lambda x, y: x + ea[y], av, 0.0)
                                e_bocc = functools.reduce(lambda x, y: x + eb[y], bo, 0.0)
                                e_bvir = functools.reduce(lambda x, y: x + eb[y], bv, 0.0)
                                den = e_bvir + e_avir - e_aocc - e_bocc
                                denominators.append(den)

    print(f"Number of amplitudes: {sop.size()}")
    return (sop, denominators)


def solve_cc_equations(
    cc_type,
    t,
    op,
    selected_op,
    op_pool,
    denominators,
    ref,
    as_ints,
    compute_threshold,
    e_convergence=1.0e-10,
    r_convergence=1.0e-5,
    on_the_fly=False,
    linked=True,
    maxk=19,
    diis_start=3,
    maxiter=200,
):
    """Solve the CC equations

    Parameters
    ----------
    cc_type : str
        The type of CC computation (cc/ucc/dcc/ducc)
    t : list
        The cluster amplitudes
    op : SparseOperator
        An operator that spans the full excitation space
    op_pool : list
        List of the operators that have been selected
    denominators: list
        Møller-Plesset denominators
    ref : dict(Denominator : float)
        The reference determinant
    as_ints : ActiveSpaceIntegrals
        the molecular integrals
    compute_threshold : float
        The compute cutoff
    e_convergence : float
        The energy convergence criterion (default = 1.0e-10)
    r_convergence : float
        The residual convergence criterion (default = 1.0e-5)
    on_the_fly : bool
        Use the on-the-fly algorithms? By default this code uses a caching algorithm
        that can be faster when using very small compute threshold values (default = False)
    maxiter : int
        The maximum number of iterations
    Returns
    -------
    tuple(t, e, e_proj, micro_iter + 1, exp.timings())
        Returns the a tuple containign the converged amplitudes, the energy, the projective energy,
        the number of iterations, and timings information
    """
    diis = DIIS(t, diis_start)
    ham = forte.SparseHamiltonian(as_ints)
    if cc_type == "cc" or cc_type == "ucc":
        exp = forte.SparseExp()
    if cc_type == "dcc" or cc_type == "ducc":
        exp = forte.SparseFactExp()

    old_e_micro = 0.0

    for micro_iter in range(maxiter):
        micro_start = time.time()
        t_old = copy.deepcopy(t)
        residual, e, e_proj = residual_equations(
            cc_type, t, op, selected_op, ref, ham, exp, compute_threshold, on_the_fly, linked, maxk
        )

        residual_norm = 0.0
        for l in range(selected_op.size()):
            t[l] -= residual[op_pool[l]] / denominators[op_pool[l]]
            residual_norm += residual[op_pool[l]] ** 2

        residual_norm = math.sqrt(residual_norm)

        t = diis.update(t, t_old)

        delta_e_micro = e - old_e_micro

        print(
            f"    {micro_iter:4d} {e:20.12f}   {delta_e_micro:+6e}   {residual_norm:+6e}   {time.time() - micro_start:8.3f}",
            flush=True,
        )

        if micro_iter > 2 and (abs(delta_e_micro) < e_convergence) and (residual_norm < r_convergence):
            break

        old_e_micro = e

    return (t, e, e_proj, micro_iter + 1, exp.timings())


def residual_equations(cc_type, t, op, sop, ref, ham, exp, compute_threshold, on_the_fly=False, linked=True, maxk=19):
    """Evaluate the residual equations

    Parameters
    ----------
    antihermitian : bool
        Is this operator antihermitian?
    max_exc : int
        The maximum excitation level (defaul: None)
    naelpi : psi4 Dimension
        The number of alpha electrons per irrep
    psi4_wfn : psi4 Wavefunction
        A psi4 Wavefunction object (to read the number of alpha/beta electrons)
    on_the_fly : bool
        Use the on-the-fly algorithms? By default this code uses a caching algorithm
        that can be faster when using very small compute threshold values (default = False)
    linked : bool
        Use a linked formulation of the CC equations (a commutator series)?
    Returns
    -------
    tuple(list,float,float)
        Returns the a tuple containign the residual (indexed as the operators/amplitudes),
        the average energy, and the projective energy
    """
    # update the amplitudes
    sop.set_coefficients(t)

    c0 = 0.0
    algorithm = "onthefly" if on_the_fly else "default"
    if cc_type == "cc":
        wfn = exp.apply_op(sop, ref, algorithm=algorithm, screen_thresh=compute_threshold, maxk=maxk)
        Hwfn = ham.compute_on_the_fly(wfn, compute_threshold)
        R = exp.apply_op(
            sop, Hwfn, scaling_factor=-1.0, algorithm=algorithm, screen_thresh=compute_threshold, maxk=maxk
        )
    elif cc_type == "ucc":
        wfn = exp.apply_antiherm(sop, ref, algorithm=algorithm, screen_thresh=compute_threshold, maxk=maxk)
        Hwfn = ham.compute_on_the_fly(wfn, compute_threshold)
        R = exp.apply_antiherm(
            sop, Hwfn, scaling_factor=-1.0, algorithm=algorithm, screen_thresh=compute_threshold, maxk=maxk
        )
    elif cc_type == "dcc":
        wfn = exp.apply_op(sop, ref, algorithm=algorithm, screen_thresh=compute_threshold)
        Hwfn = ham.compute_on_the_fly(wfn, compute_threshold)
        R = exp.apply_op(sop, Hwfn, inverse=True, algorithm=algorithm, screen_thresh=compute_threshold)
    elif cc_type == "ducc":
        wfn = exp.apply_antiherm(sop, ref, algorithm=algorithm, screen_thresh=compute_threshold)
        Hwfn = ham.compute_on_the_fly(wfn, compute_threshold)
        R = exp.apply_antiherm(sop, Hwfn, inverse=True, algorithm=algorithm, screen_thresh=compute_threshold)
    else:
        raise ValueError("Incorrect value for cc_type")

    # compute <ref|Psi>
    for d, c in ref.items():
        c0 += c * wfn[d]
    energy = 0.0
    energy_proj = 0.0
    if linked:
        # compute Eavg = <Psi|H|Psi> = <ref|U^+ H U|ref>
        for d, c in ref.items():
            energy += c * R[d]
        # compute Eproj = <ref|H|Psi> / <ref|Psi> = <ref|H U|ref> / <ref|U|ref>
        for d, c in ref.items():
            energy_proj += c * Hwfn[d] / c0
        # compute R = <exc|U^+ H U|ref>
        residual = forte.get_projection(op, ref, R)
    else:
        # compute the energy as the expectation value = <exc|U^+ H U|ref> / <exc|U^+U|ref>
        N2 = 0.0
        for d, c in wfn.items():
            N2 += c**2
            energy += c * Hwfn[d]
        energy = energy / N2
        # compute the projective energy as the expectation value = <ref|H U|ref> / <ref|U|ref>
        for d, c in ref.items():
            energy_proj += c * Hwfn[d] / c0
        # compute R = <exc|H U|ref> - E_proj <exc|U|ref>
        for d, c in wfn.items():
            Hwfn[d] -= c * energy_proj
        residual = forte.get_projection(op, ref, Hwfn)

    return (residual, energy, energy_proj)


class DIIS:
    """A class that implements DIIS for CC theory

        Parameters
    ----------
    diis_start : int
        Start the iterations when the DIIS dimension is greather than this parameter (default = 3)
    """

    def __init__(self, t, diis_start=3):
        self.t_diis = [t]
        self.e_diis = []
        self.diis_start = diis_start

    def update(self, t, t_old):
        """Update the DIIS object and return extrapolted amplitudes

        Parameters
        ----------
        t : list
            The updated amplitudes
        t_old : list
            The previous set of amplitudes
        Returns
        -------
        list
            The extrapolated amplitudes
        """

        if self.diis_start == -1:
            return t

        self.t_diis.append(t)
        self.e_diis.append(np.subtract(t, t_old))

        diis_dim = len(self.t_diis) - 1
        if (diis_dim >= self.diis_start) and (diis_dim < len(t)):
            # consturct diis B matrix (following Crawford Group github tutorial)
            B = np.ones((diis_dim + 1, diis_dim + 1)) * -1.0
            bsol = np.zeros(diis_dim + 1)
            B[-1, -1] = 0.0
            bsol[-1] = -1.0
            for i in range(len(self.e_diis)):
                for j in range(i, len(self.e_diis)):
                    B[i, j] = np.dot(np.real(self.e_diis[i]), np.real(self.e_diis[j]))
                    if i != j:
                        B[j, i] = B[i, j]
            B[:-1, :-1] /= np.abs(B[:-1, :-1]).max()
            x = np.linalg.solve(B, bsol)
            t_new = np.zeros((len(t)))
            for l in range(diis_dim):
                temp_ary = x[l] * np.asarray(self.t_diis[l + 1])
                t_new = np.add(t_new, temp_ary)
            return copy.deepcopy(list(np.real(t_new)))

        return t
