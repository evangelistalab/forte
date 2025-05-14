import numpy as np
import psi4
import json
from opt_einsum import contract

def jsondump(wfn, frozen_docc, active_docc, active_socc, active_uocc, int_cutoff = 1e-13, fname="forte_molecule.json"):
    """
    Function to build a JSON file compatible with QForte.
    Data is extracted from a Psi4 Wavefunction object.
    Assumes ROHF, and that unpaired electrons don't exist outside the
    active space
    """

    print(f"\nSaving system info to {fname}.\n")

    mol = wfn.molecule()
    mints = psi4.core.MintsHelper(wfn.basisset())    
    scalars = wfn.scalar_variables()
    E_nuc = scalars["NUCLEAR REPULSION ENERGY"]
    
    print(f"Getting AO electronic integrals. (Not density fitted.)")
    ao_oeis = mints.ao_kinetic().np + mints.ao_potential().np
    print("Done")
    print(f"Getting AO electronic integrals. (Not density fitted.)")
    ao_teis = mints.ao_eri().np        
    print("Done")
    num_frozen = np.sum(frozen_docc)
    num_active = np.sum(active_docc + active_socc + active_uocc)
    
    
    #Reorder C so that instead of orbital energy, it is sorted core->active->virtual
    #Active space is chosen for aufbau ordering (i.e. doccs, soccs, uoccs)
        
    orbitals = []
    for irrep, block in enumerate(wfn.epsilon_a_subset("MO", "ALL").nph):
        for orbital in block:
            orbitals.append([orbital, irrep])
    
    
    orbitals.sort()
    orb_irreps_to_int = []
    for [eps, irrep] in orbitals:
        orb_irreps_to_int.append(irrep)

    
    categories = {"frozen_docc": [], "active_docc": [], "active_socc": [], "active_uocc": [], "frozen_uocc": []}
    
    
    irrep_dict = {}
    for i, irrep in enumerate(orb_irreps_to_int):
        if irrep not in irrep_dict:
            irrep_dict[irrep] = []
        irrep_dict[irrep].append(i)

    
    # Assign orbitals to categories based on count constraints
    for irrep, orbitals in irrep_dict.items():
        count_frozen = frozen_docc[irrep]
        count_active_docc = active_docc[irrep]
        count_active_socc = active_socc[irrep]
        count_active_uocc = active_uocc[irrep]
         
        categories["frozen_docc"].extend(orbitals[:count_frozen])
        categories["active_docc"].extend(orbitals[count_frozen:count_frozen + count_active_docc])
        categories["active_socc"].extend(orbitals[count_frozen + count_active_docc:count_frozen + count_active_docc + count_active_socc])
        categories["active_uocc"].extend(orbitals[count_frozen + count_active_docc + count_active_socc:count_frozen + count_active_docc + count_active_socc + count_active_uocc])
        categories["frozen_uocc"].extend(orbitals[count_frozen + count_active_docc + count_active_socc + count_active_uocc:])    

    for key in categories:
        categories[key].sort()

    new_order = (categories["frozen_docc"] +
                 categories["active_docc"] +
                 categories["active_socc"] +
                 categories["active_uocc"] +
                 categories["frozen_uocc"])
    
    C = wfn.Ca_subset("AO", "ALL").np[:, new_order]
    orb_irreps_to_int = np.array(orb_irreps_to_int)[new_order].tolist()
    
    #Compute frozen core energy and frozen core one electron integral.  (E_fc does NOT include nuclear repulsion.)
    
    Pc = contract('pi,si->ps', C[:,:num_frozen], C[:,:num_frozen], optimize = True)
    ao_hc = ao_oeis + 2*contract('psuv,ps->uv', ao_teis, Pc, optimize = True) - contract('puvs,ps->uv', ao_teis, Pc, optimize = True)
    E_fc = np.trace(Pc.T@(ao_hc + ao_oeis))
    
    
    
    mo_oeis = contract("ui,vj,uv->ij", C, C, ao_hc, optimize = True)     
    mo_teis = contract("pi,qj,rk,sl,pqrs->ijkl", C, C, C, C, ao_teis, optimize = True)

    mo_oeis = mo_oeis[num_frozen:num_active+num_frozen, num_frozen:num_active+num_frozen]
    mo_teis = mo_teis[num_frozen:num_active+num_frozen, num_frozen:num_active+num_frozen,
                      num_frozen:num_active+num_frozen, num_frozen:num_active+num_frozen]
    
    nalpha = wfn.nalpha() - np.sum(frozen_docc)
    nbeta = wfn.nbeta() - np.sum(frozen_docc)
    so_irreps = []
    for i in range(num_frozen,num_frozen+num_active):
        so_irreps += [orb_irreps_to_int[i],orb_irreps_to_int[i]]
    
    print(f"({nalpha+nbeta}, {num_active}) active space.\n")
    print(f"Ms = {(nalpha - nbeta)/2}\n")
    print(f"Nuclear repulsion energy: {E_nuc}") 
    print(f"Frozen core energy: {E_fc}")
    print(f"Total scalar energy: {E_fc + E_nuc}")
    
    external_data = {}
    external_data["scalar_energy"] = {}
    external_data["scalar_energy"]["data"] = E_fc + E_nuc 
    external_data["oei"] = {}
    external_data["oei"]["data"] = []
    
    external_data["tei"] = {}
    external_data["tei"]["data"] = []

    print("Effective integrals computed.  Writing them to json.")
    for p in range(num_active):
        pa = 2 * p
        pb = (2 * p) + 1
        for q in range(num_active):
            qa = 2 * q
            qb = (2 * q) + 1
            oei = float(mo_oeis[p,q])
            if abs(oei) > int_cutoff:
                external_data["oei"]["data"].append((pa, qa, oei))         
                external_data["oei"]["data"].append((pb, qb, oei))
            for r in range(num_active):
                ra = 2 * r
                rb = (2 * r) + 1
                for s in range(num_active):
                    sa = 2 * s
                    sb = (2 * s) + 1
                    pqrs = float(mo_teis[p,r,q,s])
                    pqsr = float(mo_teis[p,s,q,r])
                    if abs(pqrs - pqsr) > int_cutoff:
                        external_data["tei"]["data"].append((pa,qa,ra,sa, pqrs - pqsr))
                        external_data["tei"]["data"].append((pb,qb,rb,sb, pqrs - pqsr))
                    if abs(pqrs) > int_cutoff:
                        external_data["tei"]["data"].append((pa,qb,ra,sb, pqrs))
                        external_data["tei"]["data"].append((pb,qa,rb,sa, pqrs))
                    if abs(pqsr) > int_cutoff:
                        external_data["tei"]["data"].append((pb,qa,ra,sb, -pqsr))
                        external_data["tei"]["data"].append((pa,qb,rb,sa, -pqsr))
    
    external_data["nso"] = {}
    external_data["nso"]["data"] = 2 * int(num_active)
    external_data["na"] = {}
    external_data["na"]["data"] = int(nalpha)
    external_data["nb"] = {}
    external_data["nb"]["data"] = int(nbeta)
    external_data["point_group"] = {}
    external_data["point_group"]["data"] = mol.symmetry_from_input().lower()
    external_data["symmetry"] = {}
    external_data["symmetry"]["data"] = so_irreps

    with open(fname, "w") as f:
        json.dump(external_data, f, indent = 0)

    print(f"\nSystem info saved to {fname}.\n") 
    
    
    



