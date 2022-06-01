import itertools
import functools
import time

import numpy as np

import copy
import psi4
import forte
import forte.utils


def run_psi(geom, basis):
    psi4.set_options({'CI_MAXITER': 100})

    # run ccsd and fci
    (escf, psi4_wfn) = forte.utils.psi4_scf(geom, basis=basis, reference='rhf')
    eccsd = psi4.energy(f'ccsd/{basis}')
    efci = psi4.energy(f'fci/{basis}')

    energies = {'FCI': efci, 'CCSD': eccsd, 'HF': escf}
    print(f'SCF Energy:  {escf:16.9f}')
    print(f'CCSD Energy: {eccsd:16.9f}')
    print(f'FCI Energy:  {efci:16.9f}')
    return (psi4_wfn, energies)


def make_forte_objs(psi4_wfn, mo_spaces, localize=False):
    if psi4_wfn.nalphapi() != psi4_wfn.nbetapi():
        print("Cannot run computations on open-shell states")
    if localize:
        mo_spaces = {'RESTRICTED_DOCC': [psi4_wfn.nalpha()], 'GAS1': [0]}
        forte_objs = forte.utils.prepare_forte_objects(
            psi4_wfn,
            mo_spaces=mo_spaces,
            active_space='CORRELATED',
            core_spaces=[],
            localize=True,
            localize_spaces=['RESTRICTED_DOCC', 'RESTRICTED_UOCC']
        )
    else:
        forte_objs = forte.utils.prepare_forte_objects(psi4_wfn, mo_spaces=mo_spaces)

    ints, as_ints, scf_info, mo_space_info, state_weights_map = forte_objs

    nirrep = mo_space_info.nirrep()

    nmo = mo_space_info.size('CORRELATED')
    nmopi = mo_space_info.dimension('CORRELATED').to_tuple()
    # the number of alpha electrons per irrep
    naelpi = (psi4_wfn.nalphapi() - mo_space_info.dimension('FROZEN_DOCC')).to_tuple()
    # the number of beta electrons per irrep
    nbelpi = (psi4_wfn.nbetapi() - mo_space_info.dimension('FROZEN_DOCC')).to_tuple()

    print(f'Number of orbitals per irrep:        {nmopi}')
    print(f'Number of alpha electrons per irrep: {naelpi}')
    print(f'Number of beta electrons per irrep:  {nbelpi}')

    return forte_objs


def make_hfref(naelpi, nbelpi, nmopi):
    """Make the Hartree-Fock reference determinant"""
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
    print(f'Reference determinant: {hfref.str(nmo)}')
    return hfref


def make_cluster_operator(antihermitian, max_exc, naelpi, nmopi, mo_space_info, psi4_wfn):
    # Prepare the cluster operator (closed-shell case)

    nirrep = mo_space_info.nirrep()

    irrep_start = [sum(nmopi[:h]) for h in range(nirrep)]
    # list orbitals
    occ_orbs = []
    vir_orbs = []
    for h in range(nirrep):
        for i in range(naelpi[h]):
            occ_orbs.append(irrep_start[h] + i)
        for i in range(naelpi[h], nmopi[h]):
            vir_orbs.append(irrep_start[h] + i)

    print(f'Occupied orbitals:    {occ_orbs}')
    print(f'Virtual orbitals:     {vir_orbs}')

    # get the symmetry of each active orbital
    symmetry = mo_space_info.symmetry('CORRELATED')

    sop = forte.SparseOperator(antihermitian=antihermitian)

    active_to_all = mo_space_info.absolute_mo('CORRELATED')

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

    print(f'Number of amplitudes: {sop.nterms()}')
    return (sop, denominators)


import math


def residual_equations(cc_type, t, op, sop, ref, ham, exp, compute_threshold, on_the_fly=False):
    sop.set_coefficients(t)
    if on_the_fly:
        #         wfn = exp.compute(sop,ref)
        wfn = exp.compute_on_the_fly(sop, ref)
        Hwfn = ham.compute_on_the_fly(wfn, compute_threshold)
        if cc_type == 'cc' or cc_type == 'ucc':
            R = exp.compute_on_the_fly(sop, Hwfn, scaling_factor=-1.0)
        elif cc_type == 'ducc' or cc_type == 'fucc' or cc_type == 'factucc':
            #             R = exp.compute(sop,Hwfn,inverse=True)
            R = exp.compute_on_the_fly(sop, Hwfn, inverse=True)
        else:
            raise ValueError('Incorrect value for cc_type')
    else:
        wfn = exp.compute(sop, ref)
        Hwfn = ham.compute(wfn, compute_threshold)
        if cc_type == 'cc' or cc_type == 'ucc':
            R = exp.compute(sop, Hwfn, scaling_factor=-1.0)
        elif cc_type == 'ducc' or cc_type == 'fucc' or cc_type == 'factucc':
            R = exp.compute(sop, Hwfn, inverse=True)
        else:
            raise ValueError('Incorrect value for cc_type')
    residual = forte.get_projection(op, ref, R)
    energy = 0.0
    for d, c in ref.map().items():
        energy += c * R[d]
    return (residual, energy)


def select_operator_pool(sorted_res, omega):
    sum_r2 = 0.0
    excluded = 0
    for r, i in sorted_res:
        sum_r2 += r**2
        if sum_r2 >= omega**2:
            break
        excluded += 1
    new_ops = [i[1] for i in sorted_res[excluded:]]
    new_ops.reverse()  # so that operators are sorted from largest to smallest residual
    return new_ops


def solve_selected_cc_equations(
    cc_type,
    t1,
    op,
    selected_op,
    op_pool,
    denominators,
    ref,
    as_ints,
    compute_threshold,
    e_convergence=1.0e-10,
    r_convergence=1.0e-10,
    maxiter=100
):
    diis = DIIS(t1)
    ham = forte.SparseHamiltonian(as_ints)
    if cc_type == 'cc' or cc_type == 'ucc':
        exp = forte.SparseExp()
    if cc_type == 'ducc' or cc_type == 'fucc' or cc_type == 'factucc':
        exp = forte.SparseFactExp()

    old_e_micro = 0.0

    for micro_iter in range(maxiter):
        micro_start = time.time()
        t1_old = copy.deepcopy(t1)
        residual, e = residual_equations(cc_type, t1, op, selected_op, ref, ham, exp, compute_threshold)

        residual_norm = 0.0
        for l in range(selected_op.nterms()):
            t1[l] -= residual[op_pool[l]] / denominators[op_pool[l]]
            residual_norm += residual[op_pool[l]]**2

        residual_norm = math.sqrt(residual_norm)

        t1 = diis.update(t1, t1_old)

        delta_e_micro = e - old_e_micro

        print(
            f'    {micro_iter:4d} {e:20.12f}   {delta_e_micro:+6e}   {residual_norm:+6e}   {time.time() - micro_start:8.3f}',
            flush=True
        )

        if micro_iter > 2 and (abs(delta_e_micro) < e_convergence) and (residual_norm < r_convergence):
            break

        old_e_micro = e

    return (t1, e, micro_iter + 1, exp.time())


class DIIS():
    def __init__(self, t):
        self.t_diis = [t]
        self.e_diis = []

    def update(self, t, t_old):
        self.t_diis.append(t)
        self.e_diis.append(np.subtract(t, t_old))

        diis_dim = len(self.t_diis) - 1

        if (diis_dim >= 3) and (diis_dim < len(t)):
            #consturct diis B matrix (following Crawford Group github tutorial)
            B = np.ones((diis_dim + 1, diis_dim + 1)) * -1.0
            bsol = np.zeros(diis_dim + 1)
            B[-1, -1] = 0.0
            bsol[-1] = -1.0
            for i in range(len(self.e_diis)):
                for j in range(i, len(self.e_diis)):
                    B[i, j] = np.dot(np.real(self.e_diis[i]), np.real(self.e_diis[j]))
                    if (i != j):
                        B[j, i] = B[i, j]
            B[:-1, :-1] /= np.abs(B[:-1, :-1]).max()
            x = np.linalg.solve(B, bsol)
            t_new = np.zeros((len(t)))
            for l in range(diis_dim):
                temp_ary = x[l] * np.asarray(self.t_diis[l + 1])
                t_new = np.add(t_new, temp_ary)
            return copy.deepcopy(list(np.real(t_new)))

        return t


def run_selected_ucc(
    forte_objs,
    psi4_wfn,
    max_exc,
    omega,
    ordering=4,
    e_convergence=1.0e-12,
    r_convergence=1.0e-10,
    compute_threshold=1.0e-10,
    selection_threshold=1.0e-10
):
    """This function implements selected factorized UCC (also known as SPQE)"""

    as_ints = forte_objs['as_ints']
    mo_space_info = forte_objs['mo_space_info']
    nirrep = mo_space_info.nirrep()

    nmo = mo_space_info.size('CORRELATED')
    nmopi = mo_space_info.dimension('CORRELATED').to_tuple()
    # the number of alpha electrons per irrep
    naelpi = (psi4_wfn.nalphapi() - mo_space_info.dimension('FROZEN_DOCC')).to_tuple()
    # the number of beta electrons per irrep
    nbelpi = (psi4_wfn.nbetapi() - mo_space_info.dimension('FROZEN_DOCC')).to_tuple()

    print(f'Number of orbitals per irrep:        {nmopi}')
    print(f'Number of alpha electrons per irrep: {naelpi}')
    print(f'Number of beta electrons per irrep:  {nbelpi}')

    op, denominators = make_cluster_operator(True, max_exc, naelpi, nmopi, mo_space_info, psi4_wfn)

    old_e = 0.0
    start = time.time()

    selected_op = forte.SparseOperator()

    ref = forte.StateVector({make_hfref(naelpi, nbelpi, nmopi): 1.0})
    print(ref.str(nmo))

    t1 = []

    # the list of operators selected from the full list
    op_pool = []
    nops_old = 0

    sum_res_eval = 0
    max_macro_iter = 100
    calc_data = []

    print('=========================================================================')
    print('     Iter.         Energy       Delta Energy    Res. Norm         Time')
    print('     Iter.          (Eh)             (Eh)         (Eh)             (s)')
    print('-------------------------------------------------------------------------')
    for macro_iter in range(max_macro_iter):

        # step 1: select new operators

        # compute the full residual and sort operators according to its magnitude
        residual, e = residual_equations(cc_type, t1, op, selected_op, ref, as_ints, ham, exp, compute_threshold)

        sorted_res = sorted(zip(residual, range(len(residual))), key=lambda x: abs(x[0]), reverse=False)

        # get the list of operators to add
        new_ops = select_operator_pool(sorted_res, omega)

        if ordering == 1:
            # add these operators
            for j in new_ops:
                if j not in op_pool:
                    op_pool.append(j)
                    selected_op.add_term(op.get_term(j))
                    t1.append(0.0)

        if ordering == 2:
            # add these operators
            new_ops.reverse()  # to make sure operators will be sorted from smallest to largest
            for j in new_ops:
                if j not in op_pool:
                    op_pool.append(j)
                    selected_op.add_term(op.get_term(j))
                    t1.append(0.0)

        if ordering == 3:
            new_t1 = []
            new_op_pool = []
            new_selected_op = forte.SparseOperator()
            # add the operators
            for j in new_ops:
                if j not in op_pool:
                    new_op_pool.append(j)
                    new_selected_op.add_term(op.get_term(j))
                    new_t1.append(0.0)
            for j in range(selected_op.nterms()):
                new_selected_op.add_term(selected_op.get_term(j))
            new_t1.extend(t1)
            new_op_pool.extend(op_pool)
            t1 = new_t1
            op_pool = new_op_pool
            selected_op = new_selected_op

        if True:
            new_t1 = []
            new_op_pool = []
            new_selected_op = forte.SparseOperator()
            new_ops.reverse()  # to make sure operators will be sorted from smallest to largest
            # add the operators
            for j in new_ops:
                if j not in op_pool:
                    new_op_pool.append(j)
                    new_selected_op.add_term(op.get_term(j))
                    new_t1.append(0.0)
            for j in range(selected_op.nterms()):
                new_selected_op.add_term(selected_op.get_term(j))
            new_t1.extend(t1)
            new_op_pool.extend(op_pool)
            t1 = new_t1
            op_pool = new_op_pool
            selected_op = new_selected_op

        print(f'Number of operators selected: {selected_op.nterms()}')

        # step 2: solve the ucc equations and update the amplitudes
        t1, e, micro_iter = solve_selected_ucc_equations(
            t1, op, selected_op, op_pool, denominators, ref, as_ints, compute_threshold, e_convergence, r_convergence
        )
        print(
            f' -> {macro_iter:4d} {e:20.12f}   {delta_e_micro:+6e}             {time.time() - start:8.3f}', flush=True
        )

        nops = selected_op.nterms()

        sum_res_eval += micro_iter * nops

        calc_data.append((nops, e, sum_res_eval))

        if (nops_old == selected_op.nterms()):
            break

        old_e = e
        nops_old = selected_op.nterms()
    print('=========================================================================')

    print(f'omega: {omega}')
    print(f'ordering: {ordering}')
    print(f' sUCC energy (forte): {e:20.12f}  Eh (Nops = {np.count_nonzero(t1):8d})')

    print(f' Computation summary')
    print(f' Nops.  Energy (Eh)')
    for n, e, mi in calc_data:
        print(f'{n:6d} {e:20.12f} {mi}')

    return calc_data[-1]


def select_new_operators():
    # get the list of operators to add
    new_ops = select_operator_pool(sorted_res, omega)

    if ordering == 1:
        # add these operators
        for j in new_ops:
            if j not in op_pool:
                op_pool.append(j)
                selected_op.add_term(op.get_term(j))
                t1.append(0.0)

    if ordering == 2:
        # add these operators
        new_ops.reverse()  # to make sure operators will be sorted from smallest to largest
        for j in new_ops:
            if j not in op_pool:
                op_pool.append(j)
                selected_op.add_term(op.get_term(j))
                t1.append(0.0)

    if ordering == 3:
        new_t1 = []
        new_op_pool = []
        new_selected_op = forte.SparseOperator()
        # add the operators
        for j in new_ops:
            if j not in op_pool:
                new_op_pool.append(j)
                new_selected_op.add_term(op.get_term(j))
                new_t1.append(0.0)
        for j in range(selected_op.nterms()):
            new_selected_op.add_term(selected_op.get_term(j))
        new_t1.extend(t1)
        new_op_pool.extend(op_pool)
        t1 = new_t1
        op_pool = new_op_pool
        selected_op = new_selected_op

    if True:
        new_t1 = []
        new_op_pool = []
        new_selected_op = forte.SparseOperator()
        new_ops.reverse()  # to make sure operators will be sorted from smallest to largest
        # add the operators
        for j in new_ops:
            if j not in op_pool:
                new_op_pool.append(j)
                new_selected_op.add_term(op.get_term(j))
                new_t1.append(0.0)
        for j in range(selected_op.nterms()):
            new_selected_op.add_term(selected_op.get_term(j))
        new_t1.extend(t1)
        new_op_pool.extend(op_pool)
        t1 = new_t1
        op_pool = new_op_pool
        selected_op = new_selected_op

    print(f'Number of operators selected: {selected_op.nterms()}')


def run_cc(
    forte_objs,
    psi4_wfn,
    cc_type=None,
    select_type=None,
    max_exc=None,
    omega=None,
    e_convergence=1.0e-10,
    r_convergence=1.0e-5,
    compute_threshold=1.0e-16,
    selection_threshold=1.0e-16
):
    """This function implements selected CC """

    as_ints = forte_objs[1]
    mo_space_info = forte_objs[3]
    nirrep = mo_space_info.nirrep()

    nmo = mo_space_info.size('CORRELATED')
    nmopi = mo_space_info.dimension('CORRELATED').to_tuple()
    # the number of alpha electrons per irrep
    naelpi = (psi4_wfn.nalphapi() - mo_space_info.dimension('FROZEN_DOCC')).to_tuple()
    # the number of beta electrons per irrep
    nbelpi = (psi4_wfn.nbetapi() - mo_space_info.dimension('FROZEN_DOCC')).to_tuple()

    print(f'Number of orbitals per irrep:        {nmopi}')
    print(f'Number of alpha electrons per irrep: {naelpi}')
    print(f'Number of beta electrons per irrep:  {nbelpi}')

    # if not provided, define the maximum excitation level to be FCI
    if max_exc == None:
        max_exc = min(naelpi + nbelpi, nmo - naelpi + nbelpi)

    antihermitian = False if cc_type == 'cc' else True

    # create the operator pool
    op, denominators = make_cluster_operator(antihermitian, max_exc, naelpi, nmopi, mo_space_info, psi4_wfn)
    selected_op = forte.SparseOperator(antihermitian)

    # the list of operators selected from the full list
    if select_type is None:
        op_pool = list(range(op.size()))
        t1 = [0.0] * op.size()
        print(f'\n The excitation operator pool contains {op.size()} elements')
        print(f'\n The selected operator pool contains {selected_op.size()} elements')
    else:
        print('\n Selected methods are not implemented yet')
        t1 = []

    old_e = 0.0
    start = time.time()

    ref = forte.StateVector({make_hfref(naelpi, nbelpi, nmopi): 1.0})
    print(ref.str(nmo))

    nops_old = 0

    sum_res_eval = 0
    max_macro_iter = 100
    calc_data = []

    print('=========================================================================')
    print('     Iter.         Energy       Delta Energy      Res. Norm       Time')
    print('                    (Eh)             (Eh)           (Eh)           (s)')
    print('-------------------------------------------------------------------------')

    for macro_iter in range(max_macro_iter):

        # step 1: select new operators
        if select_type is not None:
            # compute the full residual and sort operators according to its magnitude
            residual, e = residual_equations(
                cc_type, t1, op, selected_op, ref, None, None, compute_threshold, on_the_fly=True
            )
            select_new_operators(select_type, residual, denominators, t1, op_pool)

            sorted_res = sorted(zip(residual, range(len(residual))), key=lambda x: abs(x[0]), reverse=False)

        # step 2: solve the ucc equations and update the amplitudes
        t1, e, micro_iter, timing = solve_selected_cc_equations(
            cc_type, t1, op, op, op_pool, denominators, ref, as_ints, compute_threshold, e_convergence, r_convergence
        )

        print(
            f'\n -> {macro_iter:4d} {e:20.12f}   {e - old_e:6e}                   {time.time() - start:8.3f}',
            flush=True
        )

        nops = selected_op.nterms()

        calc_data.append((nops, e))

        if (nops_old == selected_op.nterms()):
            break

        old_e = e
        nops_old = selected_op.nterms()
    print('=========================================================================')

    print(f'{cc_type.upper()} energy:{e:20.12f}  Eh (Nops = {np.count_nonzero(t1):8d})')
    print(f'omega: {omega}')
    print(f'{timing}')

    return calc_data