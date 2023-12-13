import pathlib

import numpy as np
import psi4
import forte

from forte.data import ForteData
from .module import Module

from forte.register_forte_options import register_forte_options

import pyscf

def _convert_molecule_to_pyscf(mol_psi4, basis):
    """
    Convert a psi4.core.Molecule object to a pyscf.gto.Mole object

    Parameters
    ----------
    mol_psi4: psi4.core.Molecule
        The molecule to convert
    basis: str
        The basis set to use, as psi4.core.Molecule does not store the basis set

    Returns
    -------
    pyscf.gto.Mole
        The converted molecule
    """
    # Convert the geometry to a string
    geom = mol_psi4.to_string('xyz').splitlines()[2:]
    # Convert the geometry string to a pyscf.gto.Mole object
    mol_pyscf = pyscf.gto.Mole()
    mol_pyscf.atom = geom
    mol_pyscf.basis = basis
    mol_pyscf.charge = mol_psi4.molecular_charge()
    mol_pyscf.spin = mol_psi4.multiplicity() - 1
    mol_pyscf.symmetry = mol_psi4.point_group().symbol()
    mol_pyscf.build()
    return mol_pyscf

def run_pyscf(ref_type, mol: pyscf.gto.Mole, options: dict = None):
    """
    Perform a new PySCF computation and return a PySCF Wavefunction object.

    :param ref_type: a Python string for reference type
    :param molecule: a Psi4 Molecule object on which computation is performed
    :param print_warning: Boolean for printing warnings on screen
    :param kwargs: named arguments associated with Psi4

    :return: a Psi4 Wavefunction object from the fresh Psi4 run
    """
    ref_type = ref_type.lower().strip()

    if ref_type in ["scf", "hf", "rhf", "rohf", "uhf"]:
        wfn = pyscf.scf.RHF(mol) if ref_type in ["scf", "hf", "rhf"] else pyscf.scf.UHF(mol)
        wfn.kernel()
    elif ref_type in ["casscf", "rasscf"]:
        wfn = pyscf.mcscf.CASSCF(mol, options["CAS_NORB"], options["CAS_NELEC"])
        wfn.mc2step()
    else:
        raise ValueError(f"Invalid REF_TYPE: {ref_type.upper()} not available!")

    return wfn

def _prepare_forte_objects_from_pyscf(mol: pyscf.gto.Mole, data: ForteData) -> ForteData:
    pass

def _make_ints_from_pyscf(mol: pyscf.gto.Mole, data: ForteData) -> ForteData:
    pass

class ObjectsFromPySCF(Module):
    """
    A module to prepare the ForteIntegrals, SCFInfo, and MOSpaceInfo objects from PySCF
    """

    def __init__(self, options: dict = None):
        """
        Parameters
        ----------
        options: dict
            A dictionary of options. Defaults to None, in which case the options are read from psi4.
        """
        super().__init__(options=options)
        psi4.core.print_out("\n  Forte will use PySCF interfaces to prepare the objects\n")

    def _run(self, data: ForteData = None) -> ForteData:
        if "FIRST" in data.options.get_str("DERTYPE"):
            raise Exception("Energy gradients NOT available from PySCF yet!")

        psi4.core.print_out("\n  Preparing forte objects from PySCF\n")

        mol = _convert_molecule_to_pyscf(data.molecule, data.options.get_str("BASIS"))
        data = _prepare_forte_objects_from_pyscf(mol, data)

        # Make an integral object from the psi4 wavefunction object
        data = _make_ints_from_pyscf(mol, data)
        return data
