import pathlib
import warnings
import os

import numpy as np

import psi4
import psi4.driver.p4util as p4util

from .check_mo_orthogonality import check_mo_orthonormality

from forte._forte import (
    make_mo_space_info_from_map,
    make_state_weights_map,
    SCFInfo,
    make_ints_from_psi4,
)

from forte.data import ForteData

from forte.proc.orbital_helpers import orbital_projection
from forte.proc.orbital_helpers import read_orbitals, ortho_orbs_forte
from forte.proc.external_active_space_solver import write_wavefunction, read_wavefunction

from .module import Module

from .helpers import make_mo_spaces_from_options

p4print = psi4.core.print_out


class ObjectsFromPsi4(Module):
    """
    A module to prepare the ForteIntegrals, SCFInfo, and MOSpaceInfo objects from a Psi4 Wavefunction object
    """

    def __init__(self, options: dict = None, **kwargs):
        """
        Parameters
        ----------
        options: dict
            A dictionary of options. Defaults to None, in which case the options are read from psi4.
        """
        super().__init__(options=options)
        self.kwargs = kwargs

    def _run(self, data: ForteData = None) -> ForteData:
        data = self.prepare_forte_objects(data, **self.kwargs)

        job_type = data.options.get_str("JOB_TYPE")
        if job_type == "NONE" and data.options.get_str("ORBITAL_TYPE") == "CANONICAL":
            psi4.core.set_scalar_variable("CURRENT ENERGY", 0.0)
            return data

        # these two functions are used by the external solver to read and write MO coefficients
        if data.options.get_bool("WRITE_WFN"):
            write_wavefunction(data)

        if data.options.get_bool("READ_WFN"):
            if not os.path.isfile("coeff.json"):
                print("No coefficient files in input folder, run a SCF first!")
                exit()
            read_wavefunction(data)

        if "FCIDUMP" in data.options.get_str("INT_TYPE"):
            pass
        else:
            p4print("\n  Forte will use psi4 integrals")
            # Make an integral object from the psi4 wavefunction object
            data.ints = make_ints_from_psi4(data.psi_wfn, data.options, data.scf_info, data.mo_space_info)

        # print(data.options)
        # print(data.scf_info)
        # print(data.state_weights_map)
        # print(data.ints)
        # print(data.mo_space_info)

        return data

    def prepare_forte_objects(self, data, **kwargs):
        """
        This is a very complex (sic) function that prepares the ForteIntegrals, SCFInfo, and MOSpaceInfo objects from a Psi4 Wavefunction object.
        Its complexity is due to the fact that we need to handle different scenarios and edge cases.

        Parameters
        ----------
        data : ForteData
            The ForteData object containing options
        kwargs : dict
            Keyword arguments that are passed to the Psi4 functions and may contain a molecule object and/or a psi4 wavefunction

        Returns
        -------
        data : ForteData
            The ForteData object containing the Forte objects prepared from the Psi4 wavefunction
        """
        options = data.options
        p4print("\n\n  Preparing forte objects from a Psi4 Wavefunction object")

        # Here we handle different scenarios:
        # The user specifies a molecule and guess orbitals via the psi_wfn
        # The user specifies a molecule and we compute the reference wavefunction with psi4

        # Step 1. Get the molecule from the kwargs or molecule in the psi_wfn
        data.molecule = self.get_molecule(**kwargs)
        kwargs.pop("molecule", None)

        # Step 2. Get or compute the reference wavefunction
        psi_wfn, is_psi_wfn_fresh = self.get_psi4_wavefunction(data, **kwargs)
        kwargs.pop("ref_wfn", None)
        data.psi_wfn = psi_wfn

        # Step 3: Create MO space information (this should be avoided if possible)
        temp_mo_space_info = self.create_mo_space_info(data, **kwargs)

        # Step 4: Read the orbitals from a file if READ_ORBITALS is set. If not reading orbitals, Ca is None.
        Ca = self.get_orbitals_from_file(data)

        # Step 5: Check if the orbitals we plan to use (which might come from a file or a computation done at at different geometry/state)
        # are orthonormal. If not, orthonormalize them. If the psi_wfn is not fresh, we will perform a new SCF at the current geometry.
        # This is necessary because there seems to be a memory effect in Psi4 when using different geometries.
        self.add_and_orthonormalize_orbitals(data, is_psi_wfn_fresh, Ca, temp_mo_space_info, **kwargs)

        # Step 6: Inject DF and MINAO basis sets in the psi4 wavefunction if specified in options
        self.set_basis_sets(data)

        # Step 7: Copy state information into the ForteOptions object from the wfn where applicable.
        # This **MUST** be done before the next function prepares states.
        # Note that we do not require that Psi4 and Forte agree. This allows, e.g., using singlet orbitals for a triplet.
        self.copy_state_info_options_from_wfn(data)

        # Step 8: Prepare Forte objects from the wavefunction, including a new MOSpaceInfo object
        self.prepare_forte_objects_from_wavefunction(data, temp_mo_space_info)

        # Once we are done, all the objects are stored in the ForteData object
        return data

    def get_molecule(self, **kwargs):
        """
        Get the molecule object from the kwargs or the ref_wfn.
        Keyword arguments supersede the ref_wfn.
        If no molecule is provided, raise an error.

        Parameters
        ----------
        kwargs : dict

        Returns
        -------
        molecule : psi4.core.Molecule
            The molecule object
        """
        molecule = kwargs.pop("molecule", None)
        if molecule:
            return molecule

        # no molecule provided, try to get it from a ref_wfn passed in kwargs
        ref_wfn = kwargs.get("ref_wfn", None)
        if ref_wfn:
            return ref_wfn.molecule()
        raise ValueError("No molecule provided to prepare_forte_objects.")

    def get_psi4_wavefunction(self, data, **kwargs):
        """
        Get or compute the reference Psi4 Wavefunction.

        Returns a tuple of (ref_wfn, is_fresh) where is_fresh is a boolean indicating whether the ref_wfn is fresh.

        Parameters
        ----------
        data : ForteData
            The ForteData object containing options and molecule
        kwargs : dict
            Keyword arguments
        """
        ref_wfn = kwargs.get("ref_wfn", None)
        if ref_wfn:
            return ref_wfn, False
        # if no ref_wfn provided, compute it
        ref_type = data.options.get_str("REF_TYPE")
        p4print(f"\n  No reference wavefunction provided. Computing {ref_type} orbitals using Psi4...\n")
        # Decide whether to run SCF or MCSCF based on REF_TYPE
        job_type = data.options.get_str("JOB_TYPE")
        do_mcscf = job_type in ["CASSCF", "MCSCF_TWO_STEP"] or data.options.get_bool("MCSCF_REFERENCE")
        ref_wfn = self.run_psi4(ref_type, data.molecule, not do_mcscf, **kwargs)
        return ref_wfn, True

    def run_psi4(self, ref_type, molecule, print_warning=False, **kwargs):
        """
        Perform a new Psi4 computation and return a Psi4 Wavefunction object.

        Parameters
        ----------
        ref_type : str
            A string for reference type
        molecule : psi4.core.Molecule
            The molecule object
        print_warning : bool
            A boolean for printing warnings on screen
        kwargs : dict
            Named arguments associated with Psi4

        Returns
        -------
        wfn : psi4.core.Wavefunction
            The Psi4 Wavefunction object from the fresh Psi4 run
        """
        ref_type = ref_type.lower().strip()

        print(self.options)

        # capitalize the options
        psi4_options = {k.upper(): v for k, v in self.options.items()} if self.options else {}

        # add the mandatory arguments
        if "basis" in kwargs:
            psi4_options["BASIS"] = kwargs["basis"]

        psi4.set_options(psi4_options)

        if ref_type in ["scf", "hf", "rhf", "rohf", "uhf"]:
            if print_warning:
                msg = [
                    "Forte is using orbitals from a Psi4 SCF reference.",
                    "This is not the best for multireference computations.",
                    "To use Psi4 CASSCF orbitals, set REF_TYPE to CASSCF.",
                ]
                msg = "\n  ".join(msg)
                warnings.warn(f"\n  {msg}\n", UserWarning)

            wfn = psi4.driver.scf_helper("forte", molecule=molecule, **kwargs)
        elif ref_type in ["casscf", "rasscf"]:
            wfn = psi4.proc.run_detcas(ref_type, molecule=molecule, **kwargs)
        else:
            raise ValueError(f"Invalid REF_TYPE: {ref_type.upper()} not available!")

        return wfn

    def get_orbitals_from_file(self, data):
        """
        Read orbitals from a file if the option READ_ORBITALS is set to True.

        Parameters
        ----------
        data : ForteData
            The ForteData object containing options or reference wavefunction
        """
        # Read orbitals from file if specified or else use the orbitals from the psi_wfn
        Ca = read_orbitals() if data.options.get_bool("READ_ORBITALS") else None
        if Ca:
            # test if input Ca has the correct dimension
            nsopi = data.psi_wfn.nsopi()
            nmopi = data.psi_wfn.nmopi()

            if Ca.rowdim() != nsopi or Ca.coldim() != nmopi:
                p4print("\n  Expecting orbital dimensions:\n")
                p4print("\n  row:    ")
                nsopi.print_out()
                p4print("  column: ")
                nmopi.print_out()
                p4print("\n  Actual orbital dimensions:\n")
                p4print("\n  row:    ")
                Ca.rowdim().print_out()
                p4print("  column: ")
                Ca.coldim().print_out()
                msg = "Invalid orbitals: different basis set / molecule! Check output for more."
                raise ValueError(msg)
        return Ca

    def create_mo_space_info(self, data, **kwargs):
        """
        Create the MOSpaceInfo object from options or kwargs.

        Parameters
        ----------
        data : ForteData
            The ForteData object containing options and reference wavefunction
        kwargs : dict
            Keyword arguments
        """
        nmopi = data.psi_wfn.nmopi()
        point_group = data.psi_wfn.molecule().point_group().symbol()
        mo_spaces = kwargs.get("mo_spaces", None)
        if mo_spaces is None:
            mo_spaces = make_mo_spaces_from_options(data.options)
        mo_space_info = make_mo_space_info_from_map(nmopi, point_group, mo_spaces)
        return mo_space_info

    def add_and_orthonormalize_orbitals(self, data, is_psi_wfn_fresh, Ca, temp_mo_space_info, **kwargs):
        """
        Check if orbitals are orthonormal and orthonormalize them if necessary.

        Parameters
        ----------
        data : ForteData
            The ForteData object
        is_psi_wfn_fresh : bool
            A boolean indicating whether the reference wavefunction is freshly computed
        Ca : psi4.core.Matrix
            The MO coefficient matrix
        temp_mo_space_info : Forte MOSpaceInfo
            The MO space information
        kwargs : dict
            Keyword arguments
        """
        # no orbitals read, then we use the orbitals from the psi_wfn
        if Ca is None:
            Ca = data.psi_wfn.Ca()
            # if this is the first time we are using the psi_wfn, we can use the orbitals directly
            if is_psi_wfn_fresh:
                return

        # Build overlap matrix
        new_S = psi4.core.Wavefunction.build(data.molecule, data.options.get_str("BASIS")).S()

        # if the orbitals are orthonormal, we can use them directly
        if check_mo_orthonormality(new_S, Ca):
            data.psi_wfn.Ca().copy(Ca)
        else:
            # if the psi_wfn is fresh just orthonormalize the orbitals
            if is_psi_wfn_fresh:
                data.psi_wfn.Ca().copy(ortho_orbs_forte(data.psi_wfn, temp_mo_space_info, Ca))
            # if the psi_wfn is not fresh, we need to perform a new SCF and orthonormalize the orbitals
            else:
                p4print("\n  Perform new SCF at current geometry ...\n")

                # run a new SCF
                wfn_new = self.run_psi4("scf", data.molecule, False, **kwargs)

                # orthonormalize orbitals
                wfn_new.Ca().copy(ortho_orbs_forte(wfn_new, temp_mo_space_info, Ca))

                # copy wfn_new to psi_wfn
                data.psi_wfn.shallow_copy(wfn_new)

    def set_basis_sets(self, data):
        """
        Set up DF and MINAO basis sets if specified in options.

        Parameters
        ----------
        data : ForteData
            The ForteData object
        """
        if "DF" in data.options.get_str("INT_TYPE"):
            aux_basis = psi4.core.BasisSet.build(
                data.molecule,
                "DF_BASIS_MP2",
                data.options.get_str("DF_BASIS_MP2"),
                "RIFIT",
                data.options.get_str("BASIS"),
                puream=data.psi_wfn.basisset().has_puream(),
            )
            data.psi_wfn.set_basisset("DF_BASIS_MP2", aux_basis)

        if data.options.get_str("MINAO_BASIS"):
            minao_basis = psi4.core.BasisSet.build(data.molecule, "MINAO_BASIS", data.options.get_str("MINAO_BASIS"))
            data.psi_wfn.set_basisset("MINAO_BASIS", minao_basis)

    def copy_state_info_options_from_wfn(self, data):
        if data.options.is_none("CHARGE"):
            data.options.set_int("CHARGE", data.molecule.molecular_charge())
        elif data.molecule.molecular_charge() != data.options.get_int("CHARGE"):
            warnings.warn(
                f"Psi4 and Forte options disagree about the molecular charge ({data.molecule.molecular_charge()} vs {data.options.get_int('CHARGE')}).",
                UserWarning,
            )
        if data.options.is_none("MULTIPLICITY"):
            data.options.set_int("MULTIPLICITY", data.molecule.multiplicity())
        elif data.molecule.multiplicity() != data.options.get_int("MULTIPLICITY"):
            warnings.warn(
                f"Psi4 and Forte options disagree about the multiplicity ({data.molecule.multiplicity()}) vs ({data.options.get_int('MULTIPLICITY')}).",
                UserWarning,
            )
        nel = int(sum(data.molecule.Z(i) for i in range(data.molecule.natom()))) - data.options.get_int("CHARGE")
        if data.options.is_none("NEL"):
            data.options.set_int("NEL", nel)
        elif nel != data.options.get_int("NEL"):
            warnings.warn(
                f"Psi4 and Forte options disagree about the number of electrons ({nel}) vs ({data.options.get_int('NEL')}).",
                UserWarning,
            )

    def prepare_forte_objects_from_wavefunction(self, data, temp_mo_space_info):
        """
        Prepare Forte objects from the Psi4 wavefunction and MO space info.

        Parameters
        ----------
        data : ForteData
            The ForteData object
        temp_mo_space_info : Forte MOSpaceInfo
            The MO space information object
        """
        # Call methods that project the orbitals (AVAS, embedding)
        data.mo_space_info = orbital_projection(data.psi_wfn, data.options, temp_mo_space_info)

        # Build Forte SCFInfo object
        data.scf_info = SCFInfo(data.psi_wfn)

        # Build a map from Forte StateInfo to the weights
        data.state_weights_map = make_state_weights_map(data.options, data.mo_space_info)
