#
# @BEGIN LICENSE
#
# Psi4: an open-source quantum chemistry software package
#
# Copyright (c) 2007-2019 The Psi4 Developers.
#
# The copyrights for code used from other parties are included in
# the corresponding files.
#
# This file is part of Psi4.
#
# Psi4 is free software; you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, version 3.
#
# Psi4 is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License along
# with Psi4; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
#
# @END LICENSE
#

import psi4
import forte
import numpy as np


def read_orbitals(filename='forte_Ca.npz'):
    """ Read orbitals from file. """
    psi4.core.print_out(f"\n\n  Forte: Read orbitals from file {filename} ...")
    try:
        Ca_loaded = np.load(filename)
        nirrep = len(Ca_loaded.files)
        Ca_list = [Ca_loaded[f'arr_{i}'] for i in range(nirrep)]  # to list
        Ca_mat = psi4.core.Matrix.from_array(Ca_list)
        psi4.core.print_out(" Done\n")
        return Ca_mat
    except FileNotFoundError:
        psi4.core.print_out(f" File {filename} NOT found!\n")
        return None


def dump_orbitals(wfn, filename='forte_Ca.npz'):
    """ Dump orbitals to file. """
    psi4.core.print_out(f"\n\n  Forte: Dump orbitals to file filename ...")

    Ca = wfn.Ca()
    Ca = [Ca.to_array()] if wfn.nirrep() == 1 else Ca.to_array()

    with open(filename, 'wb') as f:
        np.savez_compressed(f, *Ca)

    psi4.core.print_out(" Done\n")


def orbital_projection(ref_wfn, options, mo_space_info):
    """
    Functions that pre-rotate orbitals before calculations;
    Requires a set of reference orbitals and mo_space_info.

    AVAS: an automatic active space selection and projection;
    Embedding: simple frozen-orbital embedding with the overlap projector.

    Return a Forte MOSpaceInfo object
    """
    if options.get_bool("AVAS"):
        # Find the subspace projector
        # - Parse the subspace planes for pi orbitals
        from .aosubspace import parse_subspace_pi_planes
        pi_planes = parse_subspace_pi_planes(ref_wfn.molecule(), options.get_list("SUBSPACE_PI_PLANES"))

        # - Create the AO subspace projector
        ps = forte.make_aosubspace_projector(ref_wfn, options, pi_planes)

        # Apply the projector to rotate orbitals
        forte.make_avas(ref_wfn, options, ps)

    # Create the fragment(embedding) projector and apply to rotate orbitals
    if options.get_bool("EMBEDDING"):
        forte.print_method_banner(["Frozen-orbital Embedding", "Nan He"])
        fragment_projector, fragment_nbf = forte.make_fragment_projector(ref_wfn)
        return forte.make_embedding(ref_wfn, options, fragment_projector, fragment_nbf, mo_space_info)
    else:
        return mo_space_info


def ortho_orbs_forte(wfn, mo_space_info, Cold):
    """
    Read the set of orbitals from file and
    pass it to the current wave function as initial guess

    :param wfn: current Psi4 Wavefunction
    :param mo_space_info: the Forte MOSpaceInfo object
    :param Cold: MO coefficients from previous calculations
    :return: orthonormalized orbital coefficients
    """
    orbital_spaces = mo_space_info.space_names()

    # slices in the order of frozen core, core, active, virtual, frozen virtual
    occ_start = [psi4.core.Dimension([0] * mo_space_info.nirrep())]
    occ_end = []
    for space in orbital_spaces[:-1]:
        occpi = mo_space_info.dimension(space)
        occpi.name = space

        temp = occ_start[-1] + occpi
        occ_start.append(temp)
        occ_end.append(temp)
    occ_end.append(wfn.nmopi())

    slices = [psi4.core.Slice(b, e) for b, e in zip(occ_start, occ_end)]

    semi = True if wfn.Fa() else False  # Forte make_fock passes to wfn.Fa()

    return ortho_orbs_impl(Cold, wfn, slices, semi)


def ortho_orbs_impl(C1, wfn, slices, semi):
    """
    Orthonormalize orbitals C1 with overlap S2: (C1)^T S2 C1 = 1.

    :param C1: orbitals for an old geometry
    :param wfn: current Psi4 Wavefunction
    :param slices: a list of Psi4 Slice objects:
        [frozen core, frozen virtual, core, active, virtual]
    :param semi: whether semicanonicalize final orbitals
    :return: orthonormalized orbitals
    """
    title = "  ==> Orthogonalize Orbitals Between Different Geometries <=="
    psi4.core.print_out(f"\n\n{title}\n\n")

    # grab current orbitals and overlap
    C2 = wfn.Ca()
    S2 = wfn.S()

    # rearrange slices to: frozen core, frozen virtual, core, active, virtual
    nspaces = len(slices)
    perm = [0, nspaces - 1] + list(range(1, nspaces - 1))
    slices = [slices[i] for i in perm]

    # create subspaces: frozen core, frozen virtual, core, active, virtual
    psi4.core.print_out("    Preparing orbitals of subspaces ......... ")

    nirrep = wfn.nirrep()
    row = psi4.core.Slice(psi4.core.Dimension(nirrep), C2.rowdim())
    Csubs = [C1.get_block(row, col) for col in slices[2:]]
    Csubs = [C2.get_block(row, col) for col in slices[:2]] + Csubs

    psi4.core.print_out("Done\n")

    # orthogonalize subspaces
    psi4.core.print_out("    Orthogonalizing orbitals of subspaces ... ")
    for i, Csub in enumerate(Csubs[2:], 2):
        for j in range(i):
            Csub = projectout(Csub, Csubs[j], S2)
        Csubs[i] = ortho_subspace(Csub, S2)

    psi4.core.print_out("Done\n")

    # fill in data to the new combined orbitals
    psi4.core.print_out("    Combining orbitals of subspaces ......... ")
    Cnew = psi4.core.Matrix("new C", C1.rowdim(), C1.coldim())
    for Csub, col in zip(Csubs, slices):
        Cnew.set_block(row, col, Csub)
    psi4.core.print_out("Done\n")

    # semicanonicalize orbitals
    if semi:
        psi4.core.print_out("    Semicanonicalizing orbitals ............. ")
        U = semicanonicalize(wfn.Fa(), Cnew, slices[2:])
        Cnew = psi4.core.doublet(Cnew, U, False, False)
        psi4.core.print_out("Done\n\n")

    return Cnew


def ortho_orbs_psi4(wfn1, wfn2, semi=True):
    """
    Make orbitals of geometry 1 (old) orthonormal with the basis
    from geometry 2 (current):
    (C1)^T S2 C1 = 1, where C1 is the CASSCF orbitals at geometry 1
    and S2 is the SO overlap matrix at geometry 2.

    :param wfn1: Psi4 Wavefunction from geometry 1
    :param wfn2: Psi4 Wavefunction from geometry 2
    :param semi: Semicanonicalize resulting orbitals
    :return: orthogonal orbital coefficients

    Example:
        molecule HF {
        F
        H 1 R
        }

        set {
          basis cc-pvdz
          reference rhf
          restricted_docc [2,0,1,1]
          active [2,0,0,0]
        }

        HF.R = 1.0
        Ecas, wfn = energy('casscf', return_wfn=True)

        HF.R = 1.1
        Escf, wfnSCF = energy('scf', return_wfn=True)
        wfnSCF.Ca().copy(ortho_orbs_psi4(wfn, wfnSCF))
        Ecas = energy('casscf', ref_wfn=wfnSCF)
    """

    nirrep = wfn2.nirrep()

    orbital_spaces = ["FROZEN_DOCC", "RESTRICTED_DOCC", "ACTIVE", "RESTRICTED_UOCC", "FROZEN_UOCC"]
    dims = {space: None for space in orbital_spaces}
    dims["RESTRICTED_UOCC"] = wfn2.nmopi()

    # dimensions for orbital spaces
    for space in orbital_spaces:
        if space == "RESTRICTED_UOCC":
            continue

        occpi = psi4.core.Dimension.from_list(psi4.core.get_option("DETCI", space))
        if occpi.n() == 0:
            occpi.init(nirrep, space)
        else:
            occpi.name = space

        dims[space] = occpi
        dims["RESTRICTED_UOCC"] -= occpi

    # slices in the order of frozen core, core, active, virtual, frozen virtual
    occ_start = [psi4.core.Dimension(nirrep)]
    occ_end = []

    for space in orbital_spaces[:-1]:
        temp = occ_start[-1] + dims[space]
        occ_start.append(temp)
        occ_end.append(temp)
    occ_end.append(wfn2.nmopi())

    slices = [psi4.core.Slice(b, e) for b, e in zip(occ_start, occ_end)]

    # call the actual implementation
    return ortho_orbs_impl(wfn1.Ca(), wfn2, slices, semi)


def projectout(C, CP, S):
    """
    Project out CP contributions from C (Schmidt orthogonalization):
    Cp = (1 - P) C = C - CP (CP^T S C),
    where P = \sum_{q} |q><q| with q being the new MO of geometry 2.

    :param C: orbitals to be projected by P
    :param CP: orbitals of the projector
    :param S: SO overlap matrix
    :return: Cp = C - CP (CP^T S C)
    """

    M = psi4.core.triplet(CP, S, C, True, False, False)
    P = psi4.core.doublet(CP, M, False, False)
    Cp = C.clone()
    Cp.subtract(P)
    return Cp


def ortho_subspace(C, S):
    """
    Orthogonalize orbitals C.
    :param C: the orbitals of a subspace at geometry 1
    :param S: the SO overlap matrix at geometry 2
    :return: orthogonalized orbitals
    """

    M = psi4.core.triplet(C, S, C, True, False, False)
    X = canonicalX(M)

    Cnew = psi4.core.doublet(C, X, False, False)
    return Cnew


def canonicalX(S):
    """
    Compute the canonical orthogonalizing transformation matrix.
    :param: the overlap metric
    :return: X = U s^(-1/2)
    """

    nirrep = S.nirrep()
    rdim = S.rowdim()
    evals = psi4.core.Vector("evals", rdim)
    evecs = psi4.core.Matrix("evecs", rdim, rdim)

    S.diagonalize(evecs, evals, psi4.core.DiagonalizeOrder.Descending)
    shalf_inv = psi4.core.Matrix("s^(-1/2)", rdim, rdim)
    for h in range(nirrep):
        for i in range(rdim[h]):
            shalf_inv.set(h, i, i, evals.get(h, i)**-0.5)

    X = psi4.core.doublet(evecs, shalf_inv, False, False)
    return X


def semicanonicalize(Fso, C, block_slices):
    """
    Semicanonicalize orbitals.
    :param Fso: the SO Fock matrix at geometry 2
    :param C: molecular orbitals that transforms SO Fock to MO Fock matrix
    :param block_slices: a list of slices for each subspace
    :return: unitary matrix that transforms orbitals to semicanonical orbitals
    """

    # transform SO Fock to MO Fock
    Fmo = psi4.core.triplet(C, Fso, C, True, False, False)

    U = psi4.core.Matrix("U to semi", Fmo.rowdim(), Fmo.coldim())
    U.identity()

    # diagonalize each blcok of Fmo
    for col in block_slices:
        F = Fmo.get_block(col, col)

        dim = col.end() - col.begin()
        evals = psi4.core.Vector("F Evals", dim)
        evecs = psi4.core.Matrix("F Evecs", dim, dim)
        F.diagonalize(evecs, evals, psi4.core.DiagonalizeOrder.Ascending)

        U.set_block(col, col, evecs)

    return U
