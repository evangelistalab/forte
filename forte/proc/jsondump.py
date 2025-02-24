import numpy as np
import psi4
import json

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
    
    ao_oeis = mints.ao_kinetic().np + mints.ao_potential().np
    ao_teis = mints.ao_eri().np        
    num_frozen = np.sum(frozen_docc)
    num_active = np.sum(active_docc + active_socc + active_uocc)

    #Reorder C so that instead of orbital energy, it is sorted core->active->virtual
    #Active space is chosen for aufbau ordering (i.e. doccs, soccs, uoccs)
        
    orbitals = []
    for irrep, block in enumerate(wfn.epsilon_a_subset("MO", "ACTIVE").nph):
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
    
    
    
    #Compute frozen core energy and frozen core one electron integral.  (E_fc does NOT include nuclear repulsion.)
    Pc = np.einsum('pi,si->ps', C[:,:num_frozen], C[:,:num_frozen])
    ao_hc = ao_oeis + 2*np.einsum('psuv,ps->uv', ao_teis, Pc) - np.einsum('puvs,ps->uv', ao_teis, Pc)
    E_fc = np.trace(Pc.T@(ao_hc + ao_oeis))
    
    mo_oeis = np.einsum("ui,vj,uv->ij", C, C, ao_oeis)[num_frozen:num_active+num_frozen,
                                                       num_frozen:num_active+num_frozen]
    
    mo_teis = np.einsum("pi,qj,rk,sl,pqrs->ijkl", C, C, C, C, ao_teis)[num_frozen:num_active+num_frozen,
                                                                  num_frozen:num_active+num_frozen,
                                                                  num_frozen:num_active+num_frozen,
                                                                  num_frozen:num_active+num_frozen]

    
 
    nalpha = wfn.nalpha() - np.sum(frozen_docc)
    nbeta = wfn.nbeta() - np.sum(frozen_docc)
    so_irreps = []
    for i in new_order:
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
    
    kept = []
    neglected = [0]
    for p in range(num_active):
        pa = p*2
        pb = p*2 + 1
        for q in range(num_active):
            qa = q*2
            qb = q*2 + 1
            oei = float(mo_oeis[p,q])
            if abs(oei) > int_cutoff:
                kept.append(oei)
                external_data["oei"]["data"].append((pa, qa, oei))         
                external_data["oei"]["data"].append((pb, qb, oei))
            else:
                neglected.append(oei)
            for r in range(num_active):
                ra = r*2
                rb = r*2 + 1
                for s in range(num_active):
                    sa = s*2
                    sb = s*2 + 1 
                    
                    tei_J = -float(mo_teis[p,s,q,r])
                    tei_K = -float(mo_teis[p,r,q,s])

                    if abs(tei_J - tei_K) > int_cutoff:
                        kept.append(tei_J - tei_K)
                        external_data["tei"]["data"].append((pa,qa,ra,sa, tei_J - tei_K))
                        external_data["tei"]["data"].append((pb,qb,rb,sb, tei_J - tei_K))
                    else:
                        neglected.append(tei_J - tei_K)
                    
                    if abs(tei_J) > int_cutoff:
                        kept.append(tei_J)
                        external_data["tei"]["data"].append((pa,qa,ra,sa,tei_J))
                        external_data["tei"]["data"].append((pb,qb,rb,sb,tei_J))
                    else:
                        neglected.append(tei_J)

    print(f"\nIntegral cutoff: {int_cutoff}\n")
    print(f"{len(kept)}/{pow(num_active,2)*2 + pow(num_active,4)*4} integrals stored.")
    print(f"Smallest included electron integral: {np.amin(abs(np.array(kept)))}")
    print(f"Largest neglected electron integral: {np.amax(abs(np.array(neglected)))}\n")
     
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
    
    
    



