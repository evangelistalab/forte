import pathlib

import numpy as np
import psi4
import forte

from forte.data import ForteData
from .module import Module

from forte.register_forte_options import register_forte_options


def _make_ints_from_fcidump(fcidump, data: ForteData):
    # transform two-electron integrals from chemist to physicist notation
    eri = fcidump["eri"]
    nmo = fcidump["norb"]
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


def _make_state_info_from_fcidump(fcidump, options):
    nel = fcidump["nelec"]
    if not options.is_none("NEL"):
        nel = options.get_int("NEL")

    multiplicity = fcidump["ms2"] + 1
    if not options.is_none("MULTIPLICITY"):
        multiplicity = options.get_int("MULTIPLICITY")

    # If the user did not specify ms determine the value from the input or
    # take the lowest value consistent with the value of "MULTIPLICITY"
    # For example:
    #    singlet: multiplicity = 1 -> twice_ms = 0 (ms = 0)
    #    doublet: multiplicity = 2 -> twice_ms = 1 (ms = 1/2)
    #    triplet: multiplicity = 3 -> twice_ms = 0 (ms = 0)
    twice_ms = (multiplicity + 1) % 2
    if not options.is_none("MS"):
        twice_ms = int(round(2.0 * options.get_double("MS")))

    if ((nel - twice_ms) % 2) != 0:
        raise Exception(f"Forte: the value of MS ({twice_ms}/2) is incompatible with the number of electrons ({nel})")

    na = (nel + twice_ms) // 2
    nb = nel - na

    irrep = fcidump["isym"]
    if not options.is_none("ROOT_SYM"):
        irrep = options.get_int("ROOT_SYM")

    return forte.StateInfo(na, nb, multiplicity, twice_ms, irrep)


def _prepare_forte_objects_from_fcidump(data, filename: str = None):
    options = data.options
    psi4.core.print_out(f"\n  Reading integral information from FCIDUMP file {filename}")
    fcidump = forte.proc.fcidump_from_file(filename, convert_to_psi4=True)

    irrep_size = {"c1": 1, "ci": 2, "c2": 2, "cs": 2, "d2": 4, "c2v": 4, "c2h": 4, "d2h": 8}

    nmo = len(fcidump["orbsym"])
    if "pntgrp" in fcidump:
        nirrep = irrep_size[fcidump["pntgrp"].lower()]
        nmopi_list = [fcidump["orbsym"].count(h) for h in range(nirrep)]
    else:
        fcidump["pntgrp"] = "C1"  # set the point group to C1
        fcidump["isym"] = 0  # shift by -1
        nirrep = 1
        nmopi_list = [nmo]

    nmopi_offset = [sum(nmopi_list[0:h]) for h in range(nirrep)]

    nmopi = psi4.core.Dimension(nmopi_list)

    # Create the MOSpaceInfo object
    data.mo_space_info = forte.make_mo_space_info(nmopi, fcidump["pntgrp"], options)

    # manufacture a SCFInfo object from the FCIDUMP file (this assumes C1 symmetry)
    nel = fcidump["nelec"]
    ms2 = fcidump["ms2"]
    na = (nel + ms2) // 2
    nb = nel - na
    if fcidump["pntgrp"] == "C1":
        doccpi = psi4.core.Dimension([nb])
        soccpi = psi4.core.Dimension([ms2])
    else:
        doccpi = options.get_int_list("FCIDUMP_DOCC")
        soccpi = options.get_int_list("FCIDUMP_SOCC")
        if len(doccpi) + len(soccpi) == 0:
            print("Reading a FCIDUMP file that uses symmetry but no DOCC and SOCC is specified.")
            print("Use the FCIDUMP_DOCC and FCIDUMP_SOCC options to specify the number of occupied orbitals per irrep.")
            doccpi = psi4.core.Dimension([0] * nirrep)
            soccpi = psi4.core.Dimension([0] * nirrep)

    if "epsilon" in fcidump:
        epsilon_a = psi4.core.Vector.from_array(fcidump["epsilon"])
        epsilon_b = psi4.core.Vector.from_array(fcidump["epsilon"])
    else:
        # manufacture Fock matrices
        epsilon_a = psi4.core.Vector(nmo)
        epsilon_b = psi4.core.Vector(nmo)
        hcore = fcidump["hcore"]
        eri = fcidump["eri"]
        nmo = fcidump["norb"]
        for i in range(nmo):
            val = hcore[i, i]
            for h in range(nirrep):
                for j in range(nmopi_offset[h], nmopi_offset[h] + doccpi[h] + soccpi[h]):
                    val += eri[i, i, j, j] - eri[i, j, i, j]
                for j in range(nmopi_offset[h], nmopi_offset[h] + doccpi[h]):
                    val += eri[i, i, j, j]
            epsilon_a.set(i, val)

            val = hcore[i, i]
            for h in range(nirrep):
                for j in range(nmopi_offset[h], nmopi_offset[h] + doccpi[h] + soccpi[h]):
                    val += eri[i, i, j, j]
                for j in range(nmopi_offset[h], nmopi_offset[h] + doccpi[h]):
                    val += eri[i, i, j, j] - eri[i, j, i, j]
            epsilon_b.set(i, val)

    data.scf_info = forte.SCFInfo(nmopi, doccpi, soccpi, 0.0, epsilon_a, epsilon_b)

    state_info = _make_state_info_from_fcidump(fcidump, options)
    data.state_weights_map = {state_info: [1.0]}
    data.psi_wfn = None

    return data, fcidump


class ObjectsFromFCIDUMP(Module):
    """
    A module to prepare the ForteIntegrals, SCFInfo, and MOSpaceInfo objects from a FCIDUMP file
    """

    def __init__(self, file=None, options: dict = None):
        """
        Parameters
        ----------
        file: str
            The name of the FCIDUMP file to read
        options: dict
            A dictionary of options. Defaults to None, in which case the options are read from psi4.
        """
        super().__init__(options=options)
        self.filename = file
        psi4.core.print_out("\n  Forte will use custom integrals")

    def _run(self, data: ForteData = None) -> ForteData:
        if "FIRST" in data.options.get_str("DERTYPE"):
            raise Exception("Energy gradients NOT available for custom integrals!")

        psi4.core.print_out("\n  Preparing forte objects from a custom source\n")
        if self.filename is None:
            self.filename = data.options.get_str("FCIDUMP_FILE")

        data, fcidump = _prepare_forte_objects_from_fcidump(data, self.filename)

        # Make an integral object from the psi4 wavefunction object
        data = _make_ints_from_fcidump(fcidump, data)
        return data
