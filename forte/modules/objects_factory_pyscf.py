import pathlib

import numpy as np
import psi4
import forte

from forte.data import ForteData
from .module import Module

from forte.register_forte_options import register_forte_options

import pyscf

def _make_ints_from_pyscf(pyscf_obj, data: ForteData):
    """
    Make custom integrals from the PySCF wavefunction object
    """
    ints = forte.make_custom_ints(
        data.options,
        data.mo_space_info,
        fcidump["enuc"],
        fcidump["hcore"].flatten(),
        fcidump["hcore"].flatten(),
        eri_aa.flatten(),
        eri_ab.flatten(),
        eri_bb.flatten(),
    )
    data.ints = ints
    return data

def _make_state_info_from_pyscf(pyscf_obj, options):
    pass
    return forte.StateInfo(na, nb, multiplicity, twice_ms, irrep)

def _prepare_forte_objects_from_pyscf(data: ForteData, pyscf_obj) -> ForteData:
    options = data.options
    psi4.core.print_out(f"\n  Getting integral information from PySCF")
    nmo = pyscf_obj.mol.nao_nr()
    
    irrep_size = {"c1": 1, "ci": 2, "c2": 2, "cs": 2, "d2": 4, "c2v": 4, "c2h": 4, "d2h": 8}
    
    if pyscf_obj.mol.symmetry is not None:
        nirrep = None
        nmopi_list = None
    else:
        if pyscf_obj.mol.symmetry.lower() is bool:
            raise Exception(f"Forte: the value of pyscf_obj.mol.symmetry ({pyscf_obj.mol.symmetry}) is a boolean")
        nirrep = irrep_size[pyscf_obj.mol.symmetry.lower()]
        if isinstance(pyscf_obj, pyscf.scf.hf.SCF): 
            nmopi_list = np.zeros(nirrep)
            for i in pyscf_obj.orbsymm:
                symm = pyscf_obj.orbsymm[i]
                nmopi_list[symm] += 1
    
    nmopi_offset = [sum(nmopi_list[0:h]) for h in range(nirrep)]
    
    nmopi = 
    
        
    return data, pyscf_obj

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

        data, self.pyscf_obj = _prepare_forte_objects_from_pyscf(data, self.pyscf_obj)

        # Make an integral object from the psi4 wavefunction object
        data = _make_ints_from_pyscf(self.pyscf_obj, data)
        return data
