import pathlib

import numpy as np
import psi4
import forte

from forte.data import ForteData
from .module import Module

from forte.register_forte_options import register_forte_options

import pyscf

def _make_ints_from_pyscf(pyscf_obj, data: ForteData, mo_coeff):
    """
    Make custom integrals from the PySCF wavefunction object
    """
    int_ao = pyscf_obj.mol.intor("int2e", aosym="s1")
    eri = pyscf.ao2mo.incore.full(int_ao, mo_coeff)
    nmo = pyscf_obj.mol.nao_nr()
    
    eri_aa = np.zeros((nmo, nmo, nmo, nmo))
    eri_ab = np.zeros((nmo, nmo, nmo, nmo))
    eri_bb = np.zeros((nmo, nmo, nmo, nmo))
    # <ij||kl> = (ik|jl) - (il|jk)
    eri_aa += np.einsum("ikjl->ijkl", eri)
    eri_aa -= np.einsum("iljk->ijkl", eri)
    # <ij|kl> = (ik|jl)
    eri_ab = np.einsum("ikjl->ijkl", eri)
    # <ij||kl> = (ik|jl) - (il|jk)
    eri_bb += np.einsum("ikjl->ijkl", eri)
    eri_bb -= np.einsum("iljk->ijkl", eri)
    
    enuc = pyscf_obj.mol.energy_nuc() # Should we also add the frozen-core energy?
    hcore_ao = pyscf_obj.get_hcore()
    
    hcore = np.einsum("uv,up,vq->pq", hcore_ao, mo_coeff.conj(), mo_coeff, optimize="optimal")
    
    ints = forte.make_custom_ints(
        data.options,
        data.mo_space_info,
        enuc,
        hcore.flatten(),
        hcore.flatten(),
        eri_aa.flatten(),
        eri_ab.flatten(),
        eri_bb.flatten(),
    )
    data.ints = ints
    
    data.as_ints = forte.make_active_space_ints(data.mo_space_info, data.ints, 'ACTIVE', ['RESTRICTED_DOCC'])
    
    return data

def _make_state_info_from_pyscf(pyscf_obj, options):
    nel = pyscf_obj.mol.nelectron
    multiplicity = pyscf_obj.mol.spin + 1
    
    twice_ms = (multiplicity + 1) % 2
    
    na = (nel + twice_ms) // 2
    nb = nel - na

    if isinstance(pyscf_obj, pyscf.mcscf.casci.CASCI):
        irrep = pyscf_obj.fcisolver.wfnsym
        if irrep is None:
            irrep = 0
    elif not options.is_none("ROOT_SYM"):
        irrep = options.get_int("ROOT_SYM")
    else:
        psi4.core.print_out(f"\n  Should run CASCI or CASSCF in Forte! Set Root_sym to 0 by default.")
        irrep = 0
    
    return forte.StateInfo(na, nb, multiplicity, twice_ms, irrep)

def _prepare_forte_objects_from_pyscf(data: ForteData, pyscf_obj) -> ForteData:
    options = data.options
    psi4.core.print_out(f"\n  Getting integral information from PySCF")
    nmo = pyscf_obj.mol.nao_nr()
    
    irrep_size = {"c1": 1, "ci": 2, "c2": 2, "cs": 2, "d2": 4, "c2v": 4, "c2h": 4, "d2h": 8}
    
    try:
        orbsym = pyscf_obj.mo_coeff.orbsym
    except:
        orbsym = np.zeros(nmo, dtype = int)
    
    if pyscf_obj.mol.symmetry is None:
        nirrep = 1
        nmopi_list = [nmo]
    else:
        if pyscf_obj.mol.groupname.lower() in ['coov', 'dooh', 'so3']:
            raise Exception(f"Forte: the value ({pyscf_obj.mol.groupname.lower()}) is not supported. Use an Abelian group.")
        nirrep = irrep_size[pyscf_obj.mol.groupname.lower()]
        nmopi_list = np.zeros(nirrep, dtype = int)
        for i in orbsym:
            nmopi_list[i] += 1
        
    # nmopi_offset = [sum(nmopi_list[0:h]) for h in range(nirrep)]

    nmopi = psi4.core.Dimension(list(nmopi_list))
    
    # Create the MOSpaceInfo object
    data.mo_space_info = forte.make_mo_space_info(nmopi, pyscf_obj.mol.groupname.lower(), options)
    
    # manufacture a SCFInfo object from the PySCF object.
    nel = pyscf_obj.mol.nelectron
    ms2 = pyscf_obj.mol.spin
    na = (nel + ms2) // 2
    nb = nel - na
    
    if pyscf_obj.mol.groupname.lower() == "c1":
        doccpi = psi4.core.Dimension([nb])
        soccpi = psi4.core.Dimension([ms2])
    else:
        if isinstance(pyscf_obj, pyscf.mcscf.casci.CASCI):
            mo_occ = pyscf_obj._scf.mo_occ
        elif isinstance(pyscf_obj, pyscf.scf.hf.SCF):
            mo_occ = pyscf_obj.mo_occ
        else:
            raise Exception(f"Forte: the object ({pyscf_obj}) is not supported. Use CASCI or SCF.")
        doccpi = np.zeros(nirrep, dtype = int)
        soccpi = np.zeros(nirrep, dtype = int)
        for i in range(len(mo_occ)):
            symm = orbsym[i]
            if mo_occ[i] == 2:
                doccpi[symm] += 1
            elif mo_occ[i] == 1:
                soccpi[symm] += 1
        
        doccpi = psi4.core.Dimension(list(doccpi))
        soccpi = psi4.core.Dimension(list(soccpi))  
    
    if isinstance(pyscf_obj, pyscf.mcscf.casci.CASCI):
        mo_energy_pyscf = pyscf_obj._scf.mo_energy
    elif isinstance(pyscf_obj, pyscf.scf.hf.SCF):
        mo_energy_pyscf = pyscf_obj.mo_energy
        
    mo_coeff_pyscf = pyscf_obj.mo_coeff
        
    if nirrep == 1:
        mo_energy = mo_energy_pyscf
        mo_coeff = mo_coeff_pyscf
    else:
        mo_energy = []
        mo_coeff = np.zeros((nmo, nmo))
        n_orb = 0
        for i in range(nirrep):
            for i_orb in range(nmo):
                if orbsym[i_orb] == int(i):
                    mo_energy.append(mo_energy_pyscf[i_orb])
                    mo_coeff[:,n_orb] = mo_coeff_pyscf[:,i_orb] 
                    n_orb += 1
        mo_energy = np.array(mo_energy)
    
    epsilon_a = psi4.core.Vector.from_array(mo_energy)
    epsilon_b = psi4.core.Vector.from_array(mo_energy)
    
    data.scf_info = forte.SCFInfo(nmopi, doccpi, soccpi, 0.0, epsilon_a, epsilon_b)
    
    state_info = _make_state_info_from_pyscf(pyscf_obj, options)
    data.state_weights_map = {state_info: [1.0]}
    data.psi_wfn = None
    
    return data, pyscf_obj, mo_coeff

class ObjectsFromPySCF(Module):
    """
    A module to prepare the ForteIntegrals, SCFInfo, and MOSpaceInfo objects from PySCF
    """

    def __init__(self, pyscf_obj, options: dict = None):
        """
        Parameters
        ----------
        options: dict
            A dictionary of options. Defaults to None, in which case the options are read from psi4.
        """
        super().__init__(options=options)
        self.pyscf_obj = pyscf_obj
        psi4.core.print_out("\n  Forte will use PySCF interfaces to prepare the objects\n")

    def _run(self, data: ForteData = None) -> ForteData:
        if "FIRST" in data.options.get_str("DERTYPE"):
            raise Exception("Energy gradients NOT available from PySCF yet!")

        psi4.core.print_out("\n  Preparing forte objects from PySCF\n")

        data, self.pyscf_obj, mo_coeff = _prepare_forte_objects_from_pyscf(data, self.pyscf_obj)

        # Make an integral object from the psi4 wavefunction object
        data = _make_ints_from_pyscf(self.pyscf_obj, data, mo_coeff)
        return data
