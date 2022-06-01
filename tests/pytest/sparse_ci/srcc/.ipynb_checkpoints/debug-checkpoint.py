import itertools
import functools
import time

import numpy as np

import psi4
import forte
import forte.utils
import copy
import math

forte.startup()


def make_geom(n,r):
    """Generate xyz coordinates for a linear chain of n hydrogen atoms with H-H distance equal to r"""
    geom = []
    for i in range(n):
        geom.append(f'H 0.0 0.0 {r * i:20.12f}')
    return '\n'.join(geom)
#+ '\nsymmetry c1'

geom = """
Ne
"""


# setup xyz geometry for linear H6
# geom = make_geom(10,1.5) #1.857421875)

print(geom)

psi4.set_options({'CI_MAXITER' : 100})

# run ccsd and fci
(escf, psi4_wfn) = forte.utils.psi4_scf(geom,basis='cc-pVDZ',reference='rhf')
# eccsd = psi4.energy('ccsd/sto-6g')
efci = psi4.energy('fci/cc-pVDZ')

print(f'SCF Energy:  {escf:16.9f}')
# print(f'CCSD Energy: {eccsd:16.9f}')
print(f'FCI Energy:  {efci:16.9f}')

mo_spaces = {'FROZEN_DOCC' : [1,0,0,0,0,0,0,0]}

from forte import forte_options

forte_objs = forte.utils.prepare_forte_objects(psi4_wfn,mo_spaces=mo_spaces)
ints, as_ints, scf_info, mo_space_info, state_weights_map = forte_objs

nirrep = mo_space_info.nirrep()

nmo = mo_space_info.size('ACTIVE')
nmopi = mo_space_info.dimension('ACTIVE').to_tuple()
# the number of alpha electrons per irrep
naelpi = (psi4_wfn.nalphapi() - mo_space_info.dimension('FROZEN_DOCC')).to_tuple()      
# the number of beta electrons per irrep
nbelpi = (psi4_wfn.nbetapi()- mo_space_info.dimension('FROZEN_DOCC')).to_tuple()           

print(f'Number of orbitals per irrep:        {nmopi}')
print(f'Number of alpha electrons per irrep: {naelpi}')
print(f'Number of beta electrons per irrep:  {nbelpi}')


def make_hfref(naelpi,nbelpi,nmopi):
    """Make the Hartree-Fock reference determinant"""
    hfref = forte.Determinant()
    # we loop over each irrep and fill the occupied orbitals 
    irrep_start = [sum(nmopi[:h]) for h in range(nirrep)]
    for h in range(nirrep):
        for i in range(naelpi[h]): hfref.set_alfa_bit(irrep_start[h] + i, True)
        for i in range(nbelpi[h]): hfref.set_beta_bit(irrep_start[h] + i, True)        
    print(f'Reference determinant: {hfref.str(nmo)}')
    return hfref

def make_cluster_operator(antihermitian,max_exc,naelpi,nmopi,mo_space_info):
    # Prepare the cluster operator (closed-shell case)
    
    irrep_start = [sum(nmopi[:h]) for h in range(nirrep)]
    # list orbitals
    occ_orbs = []
    vir_orbs = []
    for h in range(nirrep):
        for i in range(naelpi[h]): occ_orbs.append(irrep_start[h] + i)
        for i in range(naelpi[h],nmopi[h]): vir_orbs.append(irrep_start[h] + i)

    print(f'Occupied orbitals:    {occ_orbs}')
    print(f'Virtual orbitals:     {vir_orbs}')

    # get the symmetry of each active orbital
    symmetry = mo_space_info.symmetry('ACTIVE')

    sop = forte.SparseOperator(antihermitian=antihermitian)

    active_to_all = mo_space_info.absolute_mo('ACTIVE')

    ea = [psi4_wfn.epsilon_a().get(i) for i in active_to_all]
    eb = [psi4_wfn.epsilon_b().get(i) for i in active_to_all]
    
    print(ea)
    print(eb)    

    denominators = []

    for n in range(1,max_exc + 1):
        for na in range(n + 1):
            nb = n - na
            for ao in itertools.combinations(occ_orbs, na):
                for av in itertools.combinations(vir_orbs, na):            
                    for bo in itertools.combinations(occ_orbs, nb):                
                        for bv in itertools.combinations(vir_orbs, nb):
                            aocc_sym = functools.reduce(lambda x, y:  x ^ symmetry[y], ao,0)
                            avir_sym = functools.reduce(lambda x, y:  x ^ symmetry[y], av,0)
                            bocc_sym = functools.reduce(lambda x, y:  x ^ symmetry[y], bo,0)                    
                            bvir_sym = functools.reduce(lambda x, y:  x ^ symmetry[y], bv,0)                        
                            # make sure the operators are total symmetric
                            if (aocc_sym ^ avir_sym) ^ (bocc_sym ^ bvir_sym) == 0:
                                # Create a list of tuples (creation, alpha, orb) where
                                #   creation : bool (true = creation, false = annihilation)
                                #   alpha    : bool (true = alpha, false = beta)
                                #   orb      : int  (the index of the mo)
                                op = []
                                for i in ao: op.append((False,True,i))
                                for i in bo: op.append((False,False,i))                        
                                for a in reversed(bv): op.append((True,False,a))                                                                        
                                for a in reversed(av): op.append((True,True,a))

                                sop.add_term(op,0.0)                            

                                e_aocc = functools.reduce(lambda x, y: x + ea[y],ao,0.0)
                                e_avir = functools.reduce(lambda x, y: x + ea[y],av,0.0)                
                                e_bocc = functools.reduce(lambda x, y: x + eb[y],bo,0.0)
                                e_bvir = functools.reduce(lambda x, y: x + eb[y],bv,0.0)                                       
                                den = e_bvir + e_avir - e_aocc- e_bocc
                                denominators.append(den)

    print(f'Number of amplitudes: {sop.nterms()}')
    return (sop,denominators)


def residual_equations(t,op,sop,ref,as_ints,ham,exp,compute_threshold):
    sop.set_coefficients(t)

    wfn = forte.apply_exp_ah_factorized(sop,ref)
    Hwfn = forte.apply_hamiltonian(as_ints,wfn,compute_threshold)     
    R = forte.apply_exp_ah_factorized(sop,Hwfn,inverse=True)    
#     wfn = exp.compute_on_the_fly(sop,ref)
#     wfn = exp.compute(sop,ref)

#     Hwfn = ham.compute(wfn,compute_threshold)    
    

#     R = exp.compute(sop,Hwfn,inverse=True)

    residual = forte.get_projection(op,ref,R)
    energy = 0.0
    for d,c in ref.map().items():
        energy += c * R[d];
    return (residual, energy)
#     sop.set_coefficients(t)
#     wfn = forte.apply_exp_ah_factorized(sop,ref)
# #     wfn = exp.compute_on_the_fly(sop,ref)
# #     wfn = exp.compute(sop,ref)
# #     Hwfn = ham.compute_on_the_fly(wfn,compute_threshold)     
#     Hwfn = ham.compute(wfn,compute_threshold)    
    
#     R = forte.apply_exp_ah_factorized(sop,Hwfn,inverse=True)
# #     R = exp.compute_on_the_fly(sop,Hwfn,inverse=True)
# #     R = exp.compute(sop,Hwfn,inverse=True)
#     residual = forte.get_projection(op,ref,R)
#     energy = 0.0
#     for d,c in ref.map().items():
#         energy += c * R[d];
#     return (residual, energy)
  
def select_operator_pool(sorted_res,omega,residuals,denominators):
    sum_r2 = 0.0
    excluded = 0
    for r,i in sorted_res:
        sum_r2 += r**2        
        if sum_r2 >= omega**2:
            break
        excluded += 1        
#     selected = [(abs(residuals[i[1]]/denominators[i[1]]), i[1]) for i in sorted_res[excluded:]]
#     selected = sorted(selected, key= lambda x : x[0], reverse=False)
#     print(selected)
#     return [i[1] for i in selected]
    return [i[1] for i in sorted_res[excluded:]]

def solve_selected_ucc_equations(t1,op,selected_op,op_pool,denominators,ref,as_ints,compute_threshold,e_convergence = 1.0e-14, r_convergence = 1.0e-5,maxiter = 100):
    forte.SparseOperator.reset_timing()
    ham = forte.SparseHamiltonian(as_ints)
    exp = forte.SparseFactExp()
    diis = DIIS(t1)
    old_e_micro = 0.0
    for micro_iter in range(maxiter):
        t1_old = copy.deepcopy(t1)
        residual, e = residual_equations(t1,op,selected_op,ref,as_ints,ham,exp,compute_threshold)

        residual_norm = 0.0
        for l in range(selected_op.nterms()):
            t1[l] -= residual[op_pool[l]] / denominators[op_pool[l]]
            residual_norm += residual[op_pool[l]]**2
              
        residual_norm = math.sqrt(residual_norm)        
            
        t1 = diis.update(t1,t1_old)
    
        delta_e_micro = e - old_e_micro        
        if micro_iter > 2 and (abs(delta_e_micro) < e_convergence) and (residual_norm < r_convergence):
            break
        print(f'    {micro_iter:5d} {e:20.12f} {delta_e_micro:20.12f}', flush=True)                    
        old_e_micro = e
        
    print(f'    {micro_iter:5d} {e:20.12f} {delta_e_micro:20.12f}', flush=True)        
        
    print(forte.SparseOperator.timing())    
    print(f'Hamiltonian timing (iterations): {ham.time()}')
    print(f'Exponential timing (iterations): {exp.time()}')    
    return t1, e

class DIIS():
    def __init__(self,t,maxdiis = 1000000):
        self.t_diis = [t]
        self.e_diis = []
        self.maxdiis = maxdiis
    
    def update(self,t,t_old):
        self.t_diis.append(t)
        self.e_diis.append(np.subtract(t, t_old))
        
        diis_dim = len(self.t_diis) - 1
        
        if diis_dim >= 2:
            #consturct diis B matrix (following Crawford Group github tutorial)
            B = np.ones((diis_dim+1, diis_dim+1)) * -1.0
            bsol = np.zeros(diis_dim+1)
            B[-1, -1] = 0.0
            bsol[-1] = -1.0
            for i in range(len(self.e_diis)):
                for j in range(i, len(self.e_diis)):
                    B[i,j] = np.dot(np.real(self.e_diis[i]), np.real(self.e_diis[j]))
                    if(i!=j):
                        B[j,i] = B[i,j]
            B[:-1, :-1] /= np.abs(B[:-1, :-1]).max()
            x = np.linalg.solve(B, bsol)
            t_new = np.zeros(( len(t) ))
            for l in range(diis_dim):
                temp_ary = x[l] * np.asarray(self.t_diis[l+1])
                t_new = np.add(t_new, temp_ary)
            return copy.deepcopy(list(np.real(t_new)))
        
        if diis_dim > self.maxdiis:
            self.t_diis.pop(0)
            self.e_diis.pop(0)
            
        return t
    
    
ops_history = []
    
def run_selected_ucc(method, max_exc, omega, e_convergence = 1.0e-12,compute_threshold = 1.0e-15, selection_threshold = 1.0e-15):
    """This function implements selected factorized UCC (also known as SPQE)"""
    op, denominators = make_cluster_operator(True, max_exc, naelpi, nmopi, mo_space_info)
        
    old_e = 0.0
    start = time.time()

    selected_op = forte.SparseOperator()

    ref = forte.StateVector({ make_hfref(naelpi,nbelpi,nmopi) : 1.0})
    print(ref.str(nmo))
    
    t1 = []
    
    # the list of operators selected from the full list
    op_pool = []
    nops_old = 0

    max_macro_iter = 100
    calc_data = []
    

    print('=========================================================================')
    print('   Iteration     Energy (Eh)       Delta Energy (Eh)    Time (s)   Nops')   
    print('-------------------------------------------------------------------------')
    for macro_iter in range(max_macro_iter):
        
        # step 1: select new operators

        # compute the full residual and sort operators according to its magnitude
        ham = forte.SparseHamiltonian(as_ints)
        exp = forte.SparseFactExp()
        residual, e = residual_equations(t1,op,selected_op,ref,as_ints,ham,exp,compute_threshold)        
        sorted_res = sorted(zip(residual,range(len(residual))), key= lambda x : abs(x[0]), reverse=False)
        print(f'Hamiltonian timing: {ham.time()} (selection)')
            
        # get the list of operators to add
        new_ops = select_operator_pool(sorted_res,omega,residual,denominators)
        
        ordering = 5
        
        # 1. (k=kmax small->large)...(k=1 small->large) |phi_0>
        # 2. (k=kmax large->small)...(k=1 large->small) |phi_0>
        # 3. (k=1 small->large)...(k=kmax small->large) |phi_0>    [ version in article no convergence problems for H6 at r=2.0å]
        # 4. (k=1 large->small)...(k=kmax large->small) |phi_0>
    
        if ordering == 1:
            # need to reverse to add the operators with the largest residuals first
            new_ops.reverse()
            # add these operators 
            for j in new_ops:
                if j not in op_pool:    
                    op_pool.append(j)                
                    selected_op.add_term(op.get_term(j))
                    t1.append(0.0)
                    
        if ordering == 2:
            # add these operators 
            for j in new_ops:
                if j not in op_pool:    
                    op_pool.append(j)                
                    selected_op.add_term(op.get_term(j))
                    t1.append(0.0)                    

        if ordering == 3:
            new_t1 = []
            new_op_pool = []
            new_selected_op = forte.SparseOperator()            
            new_ops.reverse() # to make sure operators will be sorted from largest to smallest            
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

        if ordering == 4:
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

            
        if ordering == 5: # no ordering
            op_pool = list(range(op.size()))
            t1 = [0.0] * op.size()
            selected_op = op
            
        print(f'Number of operators selected: {selected_op.nterms()}')
        
#         with open(f'output-ref-{macro_iter}.txt', 'w') as file:  # Use file to refer to the file object
#             file.write("\n".join(selected_op.str()))
        
        # step 2: solve the ucc equations and update the amplitudes
        t1, e = solve_selected_ucc_equations(t1,op,selected_op,op_pool,denominators,ref,as_ints,compute_threshold,e_convergence)        

        ops_history.append(selected_op.str())
        
        print(f'\n{macro_iter:9d} {e:20.12f} {e - old_e:20.12f} {time.time() - start:11.3f}\n', flush=True)  

        if (nops_old == selected_op.nterms()):
            break
        
        old_e = e
        nops_old = selected_op.nterms()
        
        calc_data.append((nops_old,e,time.time() - start))
        
    print('=========================================================================')

    print(f' sUCC energy (forte): {e:20.12f} (Nops = {np.count_nonzero(t1):8d})')
    print(f' sUCC energy error  : {e - efci:20.12f}')


#     print(f' CCSD energy (psi4):  {eccsd:20.12f}')
    print(f' FCI  energy (forte): {efci:20.12f}')
    
    print(f' Computation summary')
    print(f' Nops.  Energy (Eh)')    
    for n,e,t in calc_data:
        print(f'{n:6d} {e:20.12f} {t:20.12f}')
        
    return calc_data    

for i in range(1):
    calc_data = run_selected_ucc('', 3, 0.0)

print(f'Energy error: {-3.904464823156 - calc_data[-1][1]}')
print(f'Total time  : {calc_data[-1][2]}')