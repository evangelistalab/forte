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

import numpy as np
import psi4
import qcelemental as qcel
import re


p4print = psi4.core.print_out


def parse_subspace_pi_planes(molecule, planes_expr, debug=False):
    """
    Parse the option "SUBSPACE_PI_PLANES" and return a direction vector on each plane atoms.
    :param molecule: a Psi4 Molecule object
    :param planes_expr: a list of plane expressions
    :param debug: debug flag for printing intermediate steps
    :return: a map from atom (atomic number, relative index in the molecule) to the direction vector

    This function parse a list of planes where each plane is defined by a list of atoms.
    The acceptable expressions for atoms include:
      - "C": all carbon atoms
      - "C2": the second carbon atom of the molecule
      - "C1-4": the first to fourth carbon atoms
    Examples for planes expressions:
      - [['C', 'H', 'O']]: only one plane consisting all C, H, and O atoms of the molecule.
      - [['C1-6'], ['N1-2', 'C9-11']]: plane 1 with the first six C atoms of the molecule,
                                       plane 2 with C9, C10, C11, N1 and N2 atoms.
      - [['C1-4'], ['C1-2', 'C5-6']]: plane 1 with the first four C atoms of the molecule,
                                      plane 2 with C1, C2, C5, and C6 atoms. Two planes share C1 and C2.

    Motivations:
      This function detects the directions of π orbitals for atoms forming the planes.
      The direction suggests how the atomic p orbitals should be linearly combined,
      such that the resulting orbital is perpendicular to the plane.
      This function can be useful for AVAS subspace selections on complicated molecules when:
        - the molecular xyz frame does not align with absolute xyz frame
        - the molecule contains multiple π systems
        - the plane is slightly distorted but an approximate averaged plane is desired

    Implementations:
      - Each plane is characterized by the plane unit normal.
        This normal is attached to all atoms that defines this plane.
      - If multiple planes share the same atom,
        the direction is obtained by summing over all unit normals of the planes and then normalize it.
      - The convention for the direction of a plane normal is defined such that the angle between
        the molecular centroid to the plane centroid and the the plane normal is acute.
      - The plane unit normal is computed as the smallest principal axis of the plane xyz coordinates.
        For a real plane, the singular value is zero.
        For an approximate plane, the singular value should be close to zero.

    PR #261: https://github.com/evangelistalab/forte/pull/261
    """
    # test input
    if not isinstance(molecule, psi4.core.Molecule):
        raise ValueError("Invalid argument for molecule!")

    if not isinstance(planes_expr, list):
        raise ValueError("Invalid plane expressions: layer 1 not a list!")
    else:
        for plane_atoms in planes_expr:
            if not isinstance(plane_atoms, list):
                raise ValueError("Invalid plane expressions: layer 2 not a list!")
            else:
                if not all(isinstance(i, str) for i in plane_atoms):
                    raise ValueError("Invalid plane expressions: atom expressions not string!")

    # print requested planes
    p4print("\n  ==> List of Planes Requested <==\n")
    for i, plane in enumerate(planes_expr):
        p4print(f"\n    Plane {i + 1:2d}")
        for j, atom in enumerate(plane):
            if j % 10 == 0:
                p4print("\n    ")
            p4print(f"{atom:>8s}")

    # create index map {'C': [absolute indices in molecule], 'BE': [...], ...}
    n_atoms = molecule.natom()
    abs_indices = {}
    for i in range(n_atoms):
        try:
            abs_indices[molecule.symbol(i).upper()].append(i)
        except KeyError:
            abs_indices[molecule.symbol(i).upper()] = [i]
    if debug:
        print(f"Index map: {abs_indices}")

    # put molecular geometry (Bohr) in numpy array format
    xyz = np.array([[molecule.x(i), molecule.y(i), molecule.z(i)] for i in range(n_atoms)])

    # centroid (geometric center) of the molecule
    centroid = np.mean(xyz, axis=0)
    if debug:
        print(f"Molecule centroid (Bohr): {centroid}")

    # parse planes
    atom_dirs = {}
    atom_regex = r"([A-Za-z]{1,2})\s*(\d*)\s*-?\s*(\d*)"

    for n, plane_atoms in enumerate(planes_expr):
        if debug:
            print(f"Process plane {n + 1}")

        plane = []  # absolute index for atoms forming the plane
        plane_z = []  # pair of atomic number and relative index

        # parse each plane entry
        for atom_expr in plane_atoms:
            atom_expr = atom_expr.upper()

            m = re.match(atom_regex, atom_expr)
            if not m:
                raise ValueError("Invalid expression of atoms!")

            atom, start_str, end_str = m.groups()
            if atom not in abs_indices:
                raise ValueError(f"Atom '{atom}' not in molecule!")

            start = 1
            end = int(end_str) if end_str else len(abs_indices[atom])
            if start_str:
                start = int(start_str)
                end = int(end_str) if end_str else start

            z = qcel.periodictable.to_Z(atom)
            for i in range(start - 1, end):
                plane.append(abs_indices[atom][i])
                plane_z.append((z, i))
            if debug:
                print(f"  parsed entry: {atom:2s} {start:>3d} - {end:d}")

        if debug:
            print(f"  atom indices of the plane: {plane}")

        # compute the plane unit normal (smallest principal axis)
        plane_xyz = xyz[plane]
        plane_centroid = np.mean(plane_xyz, axis=0)
        plane_xyz = plane_xyz - plane_centroid
        if debug:
            print(f"  plane centroid (Bohr): {plane_centroid}")
            print(f"  shifted plane xyz (Bohr):")
            for x, y, z in plane_xyz:
                print(f"    {x:13.10f}  {y:13.10f}  {z:13.10f}")

        # SVD the xyz coordinate
        u, s, vh = np.linalg.svd(plane_xyz)

        # fix phase
        p = plane_centroid - centroid
        plane_normal = vh[2] if np.inner(vh[2], p) >= 0.0 else vh[2] * -1.0
        if debug:
            print(f"  singular values: {s}")
            print(f"  plane unit normal: {plane_normal}")

        # attach each atom to the unit normal
        for z_i in plane_z:
            if z_i in atom_dirs:
                atom_dirs[z_i] = atom_dirs[z_i] + plane_normal
            else:
                atom_dirs[z_i] = plane_normal

    # normalize the directions on each requested atom
    atom_dirs = {z_i: n / np.linalg.norm(n) for z_i, n in atom_dirs.items()}
    if debug:
        print("Averaged vector perpendicular to the requested planes on each atom")
        for z, i in sorted(atom_dirs.keys()):
            n_str = ' '.join(f'{i:15.10f}' for i in atom_dirs[(z, i)])
            print(f"  Atom Z: {z:3d}, relative index: {i:3d}, direction: {n_str}")

    return atom_dirs
