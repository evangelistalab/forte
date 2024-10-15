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

from forte.proc.orbital_helpers import read_orbitals, basis_projection, orbital_projection, ortho_orbs_new
from forte.proc.external_active_space_solver import write_wavefunction, read_wavefunction

from .module import Module

from .helpers import make_mo_spaces_from_options

# def run_psi4_ref(ref_type, molecule, print_warning=False, **kwargs):
#     """
#     Perform a new Psi4 computation and return a Psi4 Wavefunction object.

#     :param ref_type: a Python string for reference type
#     :param molecule: a Psi4 Molecule object on which computation is performed
#     :param print_warning: Boolean for printing warnings on screen
#     :param kwargs: named arguments associated with Psi4

#     :return: a Psi4 Wavefunction object from the fresh Psi4 run
#     """
#     ref_type = ref_type.lower().strip()

#     if ref_type in ["scf", "hf", "rhf", "rohf", "uhf"]:
#         if print_warning:
#             msg = [
#                 "Forte is using orbitals from a Psi4 SCF reference.",
#                 "This is not the best for multireference computations.",
#                 "To use Psi4 CASSCF orbitals, set REF_TYPE to CASSCF.",
#             ]
#             msg = "\n  ".join(msg)
#             warnings.warn(f"\n  {msg}\n", UserWarning)

#         wfn = psi4.driver.scf_helper("forte", molecule=molecule, **kwargs)
#     elif ref_type in ["casscf", "rasscf"]:
#         wfn = psi4.proc.run_detcas(ref_type, molecule=molecule, **kwargs)
#     else:
#         raise ValueError(f"Invalid REF_TYPE: {ref_type.upper()} not available!")

#     return wfn


# def prepare_psi4_ref_wfn(options, **kwargs):
#     """
#     Prepare a Psi4 Wavefunction as reference for Forte.
#     :param options: a ForteOptions object for options
#     :param kwargs: named arguments associated with Psi4
#     :return: (the processed Psi4 Wavefunction, a Forte MOSpaceInfo object)

#     Notes:
#         We will create a new Psi4 Wavefunction (wfn_new) if necessary.

#         1. For an empty ref_wfn, wfn_new will come from Psi4 SCF or MCSCF.

#         2. For a valid ref_wfn, we will test the orbital orthonormality against molecule.
#            If the orbitals from ref_wfn are consistent with the active geometry,
#            wfn_new will simply be a link to ref_wfn.
#            If not, we will rerun a Psi4 SCF and orthogonalize orbitals, where
#            wfn_new comes from this new Psi4 SCF computation.
#     """
#     p4print = psi4.core.print_out

#     # grab reference Wavefunction and Molecule from kwargs
#     kwargs = p4util.kwargs_lower(kwargs)

#     ref_wfn = kwargs.get("ref_wfn", None)

#     molecule = kwargs.pop("molecule", psi4.core.get_active_molecule())
#     point_group = molecule.point_group().symbol()

#     # try to read orbitals from file
#     Ca = read_orbitals() if options.get_bool("READ_ORBITALS") else None

#     need_orbital_check = True
#     fresh_ref_wfn = True if ref_wfn is None else False

#     if ref_wfn is None:
#         ref_type = options.get_str("REF_TYPE")
#         p4print("\n  No reference wave function provided for Forte." f" Computing {ref_type} orbitals using Psi4 ...\n")

#         # no warning printing for MCSCF
#         job_type = options.get_str("JOB_TYPE")
#         do_mcscf = job_type in ["CASSCF", "MCSCF_TWO_STEP"] or options.get_bool("MCSCF_REFERENCE")

#         # run Psi4 SCF or MCSCF
#         ref_wfn = run_psi4_ref(ref_type, molecule, not do_mcscf, **kwargs)

#         need_orbital_check = False if Ca is None else True
#     else:
#         # Ca from file has higher priority than that of ref_wfn
#         Ca = ref_wfn.Ca().clone() if Ca is None else Ca

#     # throw if molecule of ref_wfn is not the active one
#     if molecule.name() != ref_wfn.molecule().name():
#         msg = "Invalid ref_wfn: different molecule! "
#         msg += f"Expected: {molecule.name()} but got: {ref_wfn.molecule().name()}"
#         raise ValueError(msg)

#     # run new SCF if basis sets are inconsistent
#     bs_name = options.get_str("BASIS")
#     if bs_name != ref_wfn.basisset().name():
#         p4print(f"\n\n  Different basis sets between option ({bs_name}) and ref_wfn ({ref_wfn.basisset().name()})!")
#         p4print(f"\n  Perform new SCF using target basis set {bs_name} ...\n\n")
#         kwargs_copy = {k: v for k, v in kwargs.items() if k != "ref_wfn"}
#         wfn_new = run_psi4_ref("scf", molecule, False, **kwargs_copy)

#         # create a MOSpaceInfo object
#         nmopi = wfn_new.nmopi()
#         if kwargs.get("mo_spaces", None) is None:
#             mo_space_info = make_mo_space_info(nmopi, point_group, options)
#         else:
#             mo_space_info = make_mo_space_info_from_map(nmopi, point_group, kwargs.get("mo_spaces"), [])

#         mcscf_ignore_frozen = options.get_bool("MCSCF_IGNORE_FROZEN_ORBS")
#         if mo_space_info.size("FROZEN") > 0 and (not mcscf_ignore_frozen) and options.get_bool("MCSCF_REFERENCE"):
#             msg = "\n\n  WARNING: "
#             msg += "Frozen orbitals are detected for MCSCF starting from orbitals of a different basis set."
#             msg += "\n  This is currently not supported."
#             msg += "\n  Option MCSCF_IGNORE_FROZEN_ORBS is now set to TRUE to continue."
#             options.set_bool("MCSCF_IGNORE_FROZEN_ORBS", True)
#             print(msg)
#             p4print(msg)

#         p4print("\n\n  Perform basis projection for occupied orbitals ...")
#         Ca = basis_projection(ref_wfn, wfn_new, mo_space_info)
#         wfn_new.Ca().copy(Ca)
#     else:
#         # create a MOSpaceInfo object
#         nmopi = ref_wfn.nmopi()
#         if kwargs.get("mo_spaces", None) is None:
#             mo_space_info = make_mo_space_info(nmopi, point_group, options)
#         else:
#             mo_space_info = make_mo_space_info_from_map(nmopi, point_group, kwargs.get("mo_spaces"), [])

#     # do we need to check MO overlap?
#     if not need_orbital_check:
#         wfn_new = ref_wfn
#     else:
#         new_S = psi4.core.Wavefunction.build(molecule, options.get_str("BASIS")).S()
#         if check_mo_orthonormality(new_S, Ca):
#             if bs_name == ref_wfn.basisset().name():
#                 wfn_new = ref_wfn
#                 wfn_new.Ca().copy(Ca)
#         else:
#             if fresh_ref_wfn:
#                 wfn_new = ref_wfn
#                 wfn_new.Ca().copy(ortho_orbs_new(wfn_new.Ca(), wfn_new.S(), Ca, mo_space_info))
#             else:
#                 p4print("\n  Perform new SCF at current geometry ...\n")
#                 kwargs_copy = {k: v for k, v in kwargs.items() if k != "ref_wfn"}
#                 wfn_new = run_psi4_ref("scf", molecule, False, **kwargs_copy)
#                 wfn_new.Ca().copy(ortho_orbs_new(wfn_new.Ca(), wfn_new.S(), Ca, mo_space_info))
#                 ref_wfn.shallow_copy(wfn_new)

#     # set DF and MINAO basis
#     if "DF" in options.get_str("INT_TYPE"):
#         aux_basis = psi4.core.BasisSet.build(
#             molecule,
#             "DF_BASIS_MP2",
#             options.get_str("DF_BASIS_MP2"),
#             "RIFIT",
#             options.get_str("BASIS"),
#             puream=wfn_new.basisset().has_puream(),
#         )
#         wfn_new.set_basisset("DF_BASIS_MP2", aux_basis)

#     if options.get_str("MINAO_BASIS"):
#         minao_basis = psi4.core.BasisSet.build(molecule, "MINAO_BASIS", options.get_str("MINAO_BASIS"))
#         wfn_new.set_basisset("MINAO_BASIS", minao_basis)

#     return wfn_new, mo_space_info


# def prepare_forte_objects_from_psi4_wfn(options, wfn, mo_space_info):
#     """
#     Take a psi4 wavefunction object and prepare the ForteIntegrals, SCFInfo, and MOSpaceInfo objects

#     Parameters
#     ----------
#     options : ForteOptions
#         A Forte ForteOptions object
#     wfn : psi4 Wavefunction
#         A psi4 Wavefunction object
#     mo_space_info : the MO space info read from options
#         A Forte MOSpaceInfo object

#     Returns
#     -------
#     tuple(ForteIntegrals, SCFInfo, MOSpaceInfo)
#         a tuple containing the ForteIntegrals, SCFInfo, and MOSpaceInfo objects
#     """

#     # Call methods that project the orbitals (AVAS, embedding)
#     mo_space_info = orbital_projection(wfn, options, mo_space_info)

#     # Build Forte SCFInfo object
#     scf_info = SCFInfo(wfn)

#     # Build a map from Forte StateInfo to the weights
#     state_weights_map = make_state_weights_map(options, mo_space_info)

#     return (state_weights_map, mo_space_info, scf_info)


# def prepare_forte_objects(data, name, **kwargs):
#     """
#     Prepare the ForteIntegrals, SCFInfo, and MOSpaceInfo objects.
#     :param data: the ForteOptions object
#     :param name: the name of the module associated with Psi4
#     :param kwargs: named arguments associated with Psi4
#     :return: a tuple of (Wavefunction, ForteIntegrals, SCFInfo, MOSpaceInfo, FCIDUMP)
#     """
#     lowername = "forte"
#     options = data.options

#     psi4.core.print_out("\n\n  Preparing forte objects from a Psi4 Wavefunction object")
#     ref_wfn, mo_space_info = prepare_psi4_ref_wfn(options, **kwargs)

#     # Copy state information from the wfn where applicable.
#     # This **MUST** be done before the next function prepares states.
#     # Note that we do not require Psi4 and Forte agree. This allows, e.g., using singlet orbitals for a triplet.
#     molecule = ref_wfn.molecule()
#     if (data.options.is_none("CHARGE")):
#         data.options.set_int("CHARGE", molecule.molecular_charge())
#     elif molecule.molecular_charge() != data.options.get_int("CHARGE"):
#         warnings.warn(f"Psi4 and Forte options disagree about the molecular charge ({molecule.molecular_charge()} vs {data.options.get_int('CHARGE')}).", UserWarning)
#     if (data.options.is_none("MULTIPLICITY")):
#         data.options.set_int("MULTIPLICITY", molecule.multiplicity())
#     elif molecule.multiplicity() != data.options.get_int("MULTIPLICITY"):
#         warnings.warn(f"Psi4 and Forte options disagree about the multiplicity ({molecule.multiplicity()}) vs ({data.options.get_int('MULTIPLICITY')}).", UserWarning)
#     nel = int(sum(molecule.Z(i) for i in range(molecule.natom()))) - data.options.get_int("CHARGE")
#     if (data.options.is_none("NEL")):
#         data.options.set_int("NEL", nel)
#     elif nel != data.options.get_int("NEL"):
#         warnings.warn(f"Psi4 and Forte options disagree about the number of electorns ({nel}) vs ({data.options.get_int('NEL')}).", UserWarning)

#     forte_objects = prepare_forte_objects_from_psi4_wfn(options, ref_wfn, mo_space_info)
#     state_weights_map, mo_space_info, scf_info = forte_objects
#     fcidump = None

#     data.mo_space_info = mo_space_info
#     data.scf_info = scf_info
#     data.state_weights_map = state_weights_map
#     data.psi_wfn = ref_wfn

#     return data, fcidump

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

        # Check if a molecule and a psi4 wavefunction are passed in the kwargs
        molecule = kwargs.pop("molecule", None)
        ref_wfn = kwargs.pop("ref_wfn", None)
        data.molecule = self.get_molecule(molecule, ref_wfn)
        basis = data.options.get_str("BASIS")

        # Step 3. Get or compute the reference Psi4 Wavefunction depending on the scenario
        data.psi_wfn = self.get_psi4_wavefunction(data, basis, ref_wfn, **kwargs)

        # Step 3: Create MO space information (this should be avoided if possible)
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
            print("molecule provided in kwargs")
            return molecule

        # no molecule provided, try to get it from a ref_wfn passed in kwargs
        if ref_wfn is not None:
            print("molecule provided in ref_wfn")
            return ref_wfn.molecule()
        raise ValueError("No molecule provided to prepare_forte_objects.")

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
        return Ca

    def get_psi4_wavefunction(self, data, basis, ref_wfn, **kwargs):
        """
        Get or compute the reference Psi4 Wavefunction.

        Parameters
        ----------
        data : ForteData
            The ForteData object containing options and molecule
        """
        # Here we handle different scenarios:
        #  Case | basis  | ref_wfn | Action
        #  -----------------------------------------------------------------------------------------------
        #  1    | Passed | None    | Run new SCF
        #  2    | None   | Passed  | Use ref_wfn

        # Case 1
        if not ref_wfn:
            print("Case 1")
            ref_type = data.options.get_str("REF_TYPE")
            job_type = data.options.get_str("JOB_TYPE")
            do_mcscf = job_type in ["CASSCF", "MCSCF_TWO_STEP"] or data.options.get_bool("MCSCF_REFERENCE")
            ref_wfn = self.run_psi4(ref_type, data.molecule, not do_mcscf, **kwargs)
            return ref_wfn
        elif ref_wfn:
            print("Case 2")
            # check if the basis requested is consistent with the one in ref_wnf
            is_basis_same = basis == ref_wfn.basisset().name()
            if not is_basis_same:
                print("Basis set is different from the one in ref_wfn.")
                # print(f'BASIS = {data.options.get_str("BASIS")}')
                # print(f"REF_WFN BASIS = {ref_wfn.basisset().name()}")

            # check if the molecule is the same as the one in ref_wfn
            # is_molecule_same = False
            # geom_diff = data.molecule.geometry().to_array() - ref_wfn.molecule().geometry().to_array()
            # is_molecule_same = np.linalg.norm(geom_diff) < 1e-6

            # if not is_molecule_same:
            #     print("Molecule is different from the one in ref_wfn.")
            #     print(f"Molecule = {data.molecule.geometry().to_array()}")
            #     print(f"REF_WFN Molecule = {ref_wfn.molecule().geometry().to_array()}")

            print(f"{data.molecule = }")

            mol_S = psi4.core.Wavefunction.build(data.molecule, basis).S()
            ref_S = ref_wfn.S()

            # compare the dimensions of the overlap matrices
            if mol_S.nirrep() != ref_S.nirrep():
                raise ValueError("Different number of irreps in the overlap matrices.")
            # if the dimensions are the same, compute the difference between the two matrices
            if mol_S.rowdim() == ref_S.rowdim() and mol_S.coldim() == ref_S.coldim():
                diff_S = mol_S.clone()
                diff_S.subtract(ref_S)
                norm = diff_S.rms()
                if norm < 1e-6:
                    return ref_wfn

            # if the basis sets are different, we need to project the orbitals
            new_wfn = self.run_psi4("scf", data.molecule, False, **kwargs)

            pCa = new_wfn.basis_projection(ref_wfn.Ca(), ref_wfn.nmopi(), ref_wfn.basisset(), new_wfn.basisset())
            pCb = new_wfn.basis_projection(ref_wfn.Cb(), ref_wfn.nmopi(), ref_wfn.basisset(), new_wfn.basisset())
            new_wfn.guess_Ca(pCa)
            new_wfn.guess_Cb(pCb)

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
