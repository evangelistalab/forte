import pathlib
import warnings


import numpy as np
import psi4
import psi4.driver.p4util as p4util

import forte

from forte.data import ForteData

from .check_mo_orthogonality import check_mo_orthonormality
from .module import Module

from forte.register_forte_options import register_forte_options
from forte.proc.orbital_helpers import orbital_projection


def run_psi4_ref(ref_type, molecule, print_warning=False, **kwargs):
    """
    Perform a new Psi4 computation and return a Psi4 Wavefunction object.

    :param ref_type: a Python string for reference type
    :param molecule: a Psi4 Molecule object on which computation is performed
    :param print_warning: Boolean for printing warnings on screen
    :param kwargs: named arguments associated with Psi4

    :return: a Psi4 Wavefunction object from the fresh Psi4 run
    """
    ref_type = ref_type.lower().strip()

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


def prepare_psi4_ref_wfn(options, **kwargs):
    """
    Prepare a Psi4 Wavefunction as reference for Forte.
    :param options: a ForteOptions object for options
    :param kwargs: named arguments associated with Psi4
    :return: (the processed Psi4 Wavefunction, a Forte MOSpaceInfo object)

    Notes:
        We will create a new Psi4 Wavefunction (wfn_new) if necessary.

        1. For an empty ref_wfn, wfn_new will come from Psi4 SCF or MCSCF.

        2. For a valid ref_wfn, we will test the orbital orthonormality against molecule.
           If the orbitals from ref_wfn are consistent with the active geometry,
           wfn_new will simply be a link to ref_wfn.
           If not, we will rerun a Psi4 SCF and orthogonalize orbitals, where
           wfn_new comes from this new Psi4 SCF computation.
    """
    p4print = psi4.core.print_out

    # grab reference Wavefunction and Molecule from kwargs
    kwargs = p4util.kwargs_lower(kwargs)

    ref_wfn = kwargs.get("ref_wfn", None)

    molecule = kwargs.pop("molecule", psi4.core.get_active_molecule())
    point_group = molecule.point_group().symbol()

    # try to read orbitals from file
    Ca = read_orbitals() if options.get_bool("READ_ORBITALS") else None

    need_orbital_check = True
    fresh_ref_wfn = True if ref_wfn is None else False

    if ref_wfn is None:
        ref_type = options.get_str("REF_TYPE")
        p4print("\n  No reference wave function provided for Forte." f" Computing {ref_type} orbitals using Psi4 ...\n")

        # no warning printing for MCSCF
        job_type = options.get_str("JOB_TYPE")
        do_mcscf = job_type in ["CASSCF", "MCSCF_TWO_STEP"] or options.get_bool("CASSCF_REFERENCE")

        # run Psi4 SCF or MCSCF
        ref_wfn = run_psi4_ref(ref_type, molecule, not do_mcscf, **kwargs)

        need_orbital_check = False if Ca is None else True
    else:
        # Ca from file has higher priority than that of ref_wfn
        Ca = ref_wfn.Ca().clone() if Ca is None else Ca

    # build Forte MOSpaceInfo
    nmopi = ref_wfn.nmopi()
    mo_space_info = forte.make_mo_space_info(nmopi, point_group, options)

    # do we need to check MO overlap?
    if not need_orbital_check:
        wfn_new = ref_wfn
    else:
        # test if input Ca has the correct dimension
        if Ca.rowdim() != ref_wfn.nsopi() or Ca.coldim() != nmopi:
            p4print("\n  Expecting orbital dimensions:\n")
            p4print("\n  row:    ")
            ref_wfn.nsopi().print_out()
            p4print("  column: ")
            nmopi.print_out()
            p4print("\n  Actual orbital dimensions:\n")
            p4print("\n  row:    ")
            Ca.rowdim().print_out()
            p4print("  column: ")
            Ca.coldim().print_out()
            msg = "Invalid orbitals: different basis set / molecule! Check output for more."
            raise ValueError(msg)

        new_S = psi4.core.Wavefunction.build(molecule, options.get_str("BASIS")).S()

        if check_mo_orthonormality(new_S, Ca):
            wfn_new = ref_wfn
            wfn_new.Ca().copy(Ca)
        else:
            if fresh_ref_wfn:
                wfn_new = ref_wfn
                wfn_new.Ca().copy(ortho_orbs_forte(wfn_new, mo_space_info, Ca))
            else:
                p4print("\n  Perform new SCF at current geometry ...\n")

                kwargs_copy = {k: v for k, v in kwargs.items() if k != "ref_wfn"}
                wfn_new = run_psi4_ref("scf", molecule, False, **kwargs_copy)

                # orthonormalize orbitals
                wfn_new.Ca().copy(ortho_orbs_forte(wfn_new, mo_space_info, Ca))

                # copy wfn_new to ref_wfn
                ref_wfn.shallow_copy(wfn_new)

    # set DF and MINAO basis
    if "DF" in options.get_str("INT_TYPE"):
        aux_basis = psi4.core.BasisSet.build(
            molecule,
            "DF_BASIS_MP2",
            options.get_str("DF_BASIS_MP2"),
            "RIFIT",
            options.get_str("BASIS"),
            puream=wfn_new.basisset().has_puream(),
        )
        wfn_new.set_basisset("DF_BASIS_MP2", aux_basis)

    if options.get_str("MINAO_BASIS"):
        minao_basis = psi4.core.BasisSet.build(molecule, "MINAO_BASIS", options.get_str("MINAO_BASIS"))
        wfn_new.set_basisset("MINAO_BASIS", minao_basis)

    return wfn_new, mo_space_info


def prepare_forte_objects_from_psi4_wfn(options, wfn, mo_space_info):
    """
    Take a psi4 wavefunction object and prepare the ForteIntegrals, SCFInfo, and MOSpaceInfo objects

    Parameters
    ----------
    options : ForteOptions
        A Forte ForteOptions object
    wfn : psi4 Wavefunction
        A psi4 Wavefunction object
    mo_space_info : the MO space info read from options
        A Forte MOSpaceInfo object

    Returns
    -------
    tuple(ForteIntegrals, SCFInfo, MOSpaceInfo)
        a tuple containing the ForteIntegrals, SCFInfo, and MOSpaceInfo objects
    """

    # Call methods that project the orbitals (AVAS, embedding)
    mo_space_info = orbital_projection(wfn, options, mo_space_info)

    # Build Forte SCFInfo object
    scf_info = forte.SCFInfo(wfn)

    # Build a map from Forte StateInfo to the weights
    state_weights_map = forte.make_state_weights_map(options, mo_space_info)

    return (state_weights_map, mo_space_info, scf_info)


def prepare_forte_objects(data, name, **kwargs):
    """
    Prepare the ForteIntegrals, SCFInfo, and MOSpaceInfo objects.
    :param data: the ForteOptions object
    :param name: the name of the module associated with Psi4
    :param kwargs: named arguments associated with Psi4
    :return: a tuple of (Wavefunction, ForteIntegrals, SCFInfo, MOSpaceInfo, FCIDUMP)
    """
    lowername = "forte"
    options = data.options

    psi4.core.print_out("\n\n  Preparing forte objects from a Psi4 Wavefunction object")
    ref_wfn, mo_space_info = prepare_psi4_ref_wfn(options, **kwargs)
    forte_objects = prepare_forte_objects_from_psi4_wfn(options, ref_wfn, mo_space_info)
    state_weights_map, mo_space_info, scf_info = forte_objects
    fcidump = None

    data.mo_space_info = mo_space_info
    data.scf_info = scf_info
    data.state_weights_map = state_weights_map
    data.psi_wfn = ref_wfn

    return data, fcidump


class ObjectsFactoryPsi4(Module):
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
        name = "forte"
        data, fcidump = prepare_forte_objects(data, name, **self.kwargs)

        job_type = data.options.get_str("JOB_TYPE")
        if job_type == "NONE" and data.options.get_str("ORBITAL_TYPE") == "CANONICAL":
            psi4.core.set_scalar_variable("CURRENT ENERGY", 0.0)
            return data.psi_wfn

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
            psi4.core.print_out("\n  Forte will use psi4 integrals")
            # Make an integral object from the psi4 wavefunction object
            data.ints = forte.make_ints_from_psi4(data.psi_wfn, data.options, data.mo_space_info)

        return data
