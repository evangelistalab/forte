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

from forte.proc.orbital_helpers import add_orthogonal_vectors, orbital_projection
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

        # Step 1. Check if a molecule and a psi4 wavefunction are passed in the kwargs and if the basis set is specified in the options
        molecule = kwargs.pop("molecule", None)
        ref_wfn = kwargs.pop("ref_wfn", None)
        # we store the basis information in kwargs, so we can pass it to the psi4 wavefunction
        # we can use the basis set from the options or the one passed in kwargs, but if both are provided, we throw an error
        if "basis" not in kwargs:
            basis = data.options.get_str("BASIS")
            if not basis:
                basis = psi4.core.get_global_option("BASIS")
                p4print(f" Using basis set {basis} from Psi4 global options.")
            kwargs["basis"] = basis
        else:
            if data.options.get_str("BASIS"):
                raise ValueError("Both basis set in options and kwargs are provided. Please provide only one.")

        if "basis" not in kwargs:
            kwargs["basis"] = basis

        # Step 2: Get the molecule object from the kwargs or the ref_wfn
        data.molecule = self.get_molecule(molecule, ref_wfn)

        # Step 3. Get or compute the reference Psi4 Wavefunction
        data.psi_wfn = self.get_psi4_wavefunction(data, ref_wfn, **kwargs)

        # Step 4: Create MO space information (this should be avoided if possible)
        temp_mo_space_info = self.create_mo_space_info(data, **kwargs)

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

    def get_molecule(self, molecule, ref_wfn):
        """
        Get the molecule object from the kwargs or the ref_wfn.
        Keyword arguments supersede the ref_wfn.
        If no molecule is provided, raise an error.

        Parameters
        ----------
        molecule : psi4.core.Molecule
            The molecule object
        ref_wfn : psi4.core.Wavefunction
            The reference wavefunction object

        Returns
        -------
        molecule : psi4.core.Molecule
            The molecule object
        """
        # molecule provided in kwargs takes precedence
        if molecule is not None:
            return molecule

        # no molecule provided, try to get it from a ref_wfn passed in kwargs
        if ref_wfn is not None:
            return ref_wfn.molecule()
        raise ValueError("No molecule provided to prepare_forte_objects.")

    def get_psi4_wavefunction(self, data, ref_wfn, **kwargs):
        """
        Get or compute the reference Psi4 Wavefunction.

        Parameters
        ----------
        data : ForteData
            The ForteData object containing options and molecule
        """
        #  Here we handle twp different scenarios:
        #  Case | ref_wfn | Action
        #  -------------------------------------
        #  1    | None    | Run new SCF or MCSCF from Psi4 and return the wavefunction
        #  2    | Passed  | Use ref_wfn

        if not ref_wfn:
            ref_type = data.options.get_str("REF_TYPE")
            job_type = data.options.get_str("JOB_TYPE")
            do_mcscf = job_type in ["CASSCF", "MCSCF_TWO_STEP"] or data.options.get_bool("MCSCF_REFERENCE")
            ref_wfn = self.run_psi4(ref_type, data.molecule, not do_mcscf, **kwargs)
            return ref_wfn
        elif ref_wfn:
            # to test if we can use the orbitals directly, we compare the SO overlap matrices
            # metric matrix for the molecule built using the basis set from kwargs
            mol_S = psi4.core.Wavefunction.build(data.molecule, kwargs["basis"]).S()
            # metric matrix for the reference wavefunction
            ref_S = ref_wfn.S()

            # catch the case where the number of irreps is different
            if mol_S.nirrep() != ref_S.nirrep():
                raise ValueError(
                    "Different number of irreps in the overlap matrices of the reference wavefunction and the molecule."
                )

            # if the dimensions are the same, compute the difference between the two matrices
            if mol_S.rowdim() == ref_S.rowdim() and mol_S.coldim() == ref_S.coldim():
                diff_S = mol_S.clone()
                diff_S.subtract(ref_S)
                norm = diff_S.rms()
                # if the norm of the difference is less than 1e-6, we can use the orbitals directly
                if norm < 1e-10:
                    return ref_wfn

            # compute a new Wavefunction object
            new_wfn = self.run_psi4("scf", data.molecule, False, **kwargs)

            # project the orbitals from the reference wavefunction to the new wavefunction
            # note that this function may produce fewer orbitals than the new wavefunction
            pCa = new_wfn.basis_projection(ref_wfn.Ca(), ref_wfn.nmopi(), ref_wfn.basisset(), new_wfn.basisset())
            pCb = new_wfn.basis_projection(ref_wfn.Cb(), ref_wfn.nmopi(), ref_wfn.basisset(), new_wfn.basisset())

            # If the number of orbitals is different, we add orthogonal vectors to the projected orbitals
            if pCa.coldim() != new_wfn.Ca().coldim():
                nirrep = new_wfn.nirrep()
                Snp = new_wfn.S().nph
                pCanp = pCa.nph
                pCbnp = pCb.nph
                fullCa = []
                fullCb = []
                for h in range(nirrep):
                    fullCa.append(add_orthogonal_vectors(pCanp[h], Snp[h]))
                    fullCb.append(add_orthogonal_vectors(pCbnp[h], Snp[h]))

                # redefine the projected orbitals
                pCa = psi4.core.Matrix.from_array(fullCa)
                pCb = psi4.core.Matrix.from_array(fullCb)

            # copy the projected orbitals to the new wavefunction
            new_wfn.Ca().copy(pCa)
            new_wfn.Cb().copy(pCa)

            return new_wfn
        else:
            raise ValueError("Other cases not supported yet.")

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
