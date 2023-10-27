/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2023 by its authors (see COPYING, COPYING.LESSER, AUTHORS).
 *
 * The copyrights for code used from other parties are included in
 * the corresponding files.
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program.  If not, see http://www.gnu.org/licenses/.
 *
 * @END LICENSE
 */

#ifndef _aosubspace_h_
#define _aosubspace_h_

#include "psi4/libmints/molecule.h"
#include "psi4/libmints/basisset.h"
#include "psi4/libmints/vector3.h"

#include <pybind11/pybind11.h>

namespace forte {
class ForteOptions;

/**
 * @brief The AOInfo class
 *
 * A class to store information about an atomic orbital
 */
class AOInfo {
  public:
    AOInfo(int A, int Z, int element_count, int n, int l, int m)
        : A_(A), Z_(Z), element_count_(element_count), n_(n), l_(l), m_(m) {}

    int A() const { return A_; }
    int Z() const { return Z_; }
    int element_count() const { return element_count_; }
    int n() const { return n_; }
    int l() const { return l_; }
    int m() const { return m_; }

  private:
    int A_;
    int Z_;
    int element_count_;
    int n_;
    int l_;
    int m_;
};

/**
 * @brief The AOSubspace class
 *
 * Typical usage:
 *
 *    // Find the AO subset
 *    std::shared_ptr<psi::Wavefunction> wfn = psi::Process::environment.wavefunction();
 *
 *    std::vector<std::string> subspace_str;
 *    if (options["SUBSPACE"].size() > 0){
 *        for (int entry = 0; entry < (int)options["SUBSPACE"].size(); ++entry){
 *            std::string s = options["SUBSPACE"][entry].to_string();
 *            subspace_str.push_back(s);
 *        }
 *    }
 *
 *    // Grab MINAO basis set
 *    std::shared_ptr<psi::BasisSet> min_basis = wfn->get_basisset("MINAO_BASIS");
 *
 *    // atom normals used to project atomic p orbitals
 *    // empty map: px, py, pz (3 orbitals) -> px, py, pz (3 orbitals)
 *    // nonempty: px, py, pz (3 orbitals) -> nx * px + ny * py + nz * pz (1 orbital)
 *    // implemented in Python side (see aosubspace.py in forte/proc)
 *    std::map<std::pair<int, int>, psi::Vector3> atom_normals;
 *
 *    // Debug printing
 *    debug = false;
 *
 *    // Create an AOSubspace object
 *    AOSubspace aosub(subspace_str, wfn->molecule(), min_basis, atom_normals, debug);
 *
 *    // Build a projector for computational (large) basis
 *    std::shared_ptr<psi::Matrix> Ps = aosub.build_projector(wfn->basisset());
 *
 *  Syntax:
 *
 *    Subspaces are specified by a string of the form "<element><range><ao set>"
 *
 *    <element> - the symbol of the element, e.g. 'Fe', 'C'
 *
 *    <range>   - the range of the atoms selected.  Possible choices are:
 *                1) '' (empty): all atoms that match <element> are selected
 *                2) 'i'       : select the i-th atom of type <element>
 *                3) 'i-j'     : select atoms i through j (included) of type
 * <element>
 *
 *    <ao set>  - the set of atomic orbitals to select.  Possible choices are:
 *                1) '' (empty): select all basis functions
 *                2) '(nl)'    : select the n-th level with angular momentum l
 *                               e.g. '(1s)', '(2s)', '(2p)',...
 *                               n = 1, 2, 3, ...
 *                               l = 's', 'p', 'd', 'f', 'g', ...
 *                3) '(nlm)'   : select the n-th level with angular momentum l and component m
 *                               e.g. '(2pz)', '(3dz2)', '(3dx2-y2)'
 *                               n = 1, 2, 3, ...
 *                               l = 's', 'p', 'd', 'f', 'g', ...
 *                               m = 'x', 'y', 'z', 'xy', 'xz', 'yz', 'z2', 'x2-y2'
 *
 *    Valid options include:
 *
 *    ["C"] - all carbon atoms
 *    ["C","N"] - all carbon and nitrogen atoms
 *    ["C1"] - carbon atom #1
 *    ["C1-3"] - carbon atoms #1, #2, #3
 *    ["C(2p)"] - the 2p subset of all carbon atoms
 *    ["C(1s)","C(2s)"] - the 1s/2s subsets of all carbon atoms
 *    ["C1-3(2s)"] - the 2s subsets of carbon atoms #1, #2, #3
 *    ["Ce(4fzx2-zy2)"] - the 4f zxx-zyy orbital of all Ce atoms
 */
class AOSubspace {
  public:
    // ==> Constructors <==

    /// Constructor using a list of subspaces and atom normals
    /// @param subspace_str: a list of subspace orbitals, e.g, {"C2", "N", "Fe(3d)", "Mo(3dx2-y2)"}
    /// @param molecule: a Psi4 Molecule object
    /// @param minao_basis: a Psi4 Basis object,
    ///                     a minimal basis set where subspace orbitals are selected from
    /// @param atom_normals: (optional) a map from the atom to its normal
    /// @param debug_mode: debug mode if True (more printing)
    ///
    /// For the argument of atom_normals:
    ///   An atom is characterized by its atomic number and the relative index.
    ///   For example, {6, 2} - the third carbon atom of the molecule (index is 0-based)
    ///
    ///   This argument is used to make the subspace p orbitals of an atom aligned to the normal:
    ///   px, py, pz (3 orbitals) -> nx * px + ny * py + nz * pz (1 orbital),
    ///   where the atom normal is a 3D unit vector (nx, ny, nz).
    ///
    ///   If the map is empty, the p orbitals are in the xyz frame defined the molecule.
    ///   Equivalently, 3 normals are attached to each atom: (1,0,0), (0,1,0), and (0,0,1).
    ///   As such, we still have a full set of p orbitals in the subspace.
    AOSubspace(std::vector<std::string> subspace_str, std::shared_ptr<psi::Molecule> molecule,
               std::shared_ptr<psi::BasisSet> minao_basis,
               std::map<std::pair<int, int>, psi::Vector3> atom_normals = {},
               bool debug_mode = false);

    // ==> User's interface <==

    /// Build the projector Pso = Ssl^T Sss^-1 Ssl
    /// Ssl: overlap between subspace and large (computational) orbitals
    /// Sss: overlap between subspace orbitals
    /// @param large_basis: the large computational basis set
    /// @return: the projector in SO basis
    std::shared_ptr<psi::Matrix> build_projector(const std::shared_ptr<psi::BasisSet>& large_basis);

  private:
    /// The vector of subspace descriptors passed by the user
    std::vector<std::string> subspace_str_;
    /// The molecule
    std::shared_ptr<psi::Molecule> molecule_;
    /// The AO basis set
    std::shared_ptr<psi::BasisSet> min_basis_;

    /// The l-label of atomic orbitals.
    /// l_labels_[l] returns the label for an orbital
    /// with angular momentum quantum number l
    std::vector<std::string> l_labels_;

    /// The label of spherical atomic orbitals.
    /// lm_labels_spherical_[l][m] returns the label for an orbital
    /// with angular momentum quantum number l and index m
    std::vector<std::vector<std::string>> lm_labels_spherical_;

    /// The map from labels to angular momentum quantum numbers
    /// e.g., labels_spherical_to_lm_["P"] = {{1,0}, {1,1}, {1,2}}
    /// e.g., labels_spherical_to_lm_["PZ"] = {{1,0}}
    /// e.g., labels_spherical_to_lm_["PX"] = {{1,1}}
    /// e.g., labels_spherical_to_lm_["PY"] = {{1,2}}
    /// ordering can be found in lm_labels_spherical_
    std::map<std::string, std::vector<std::pair<int, int>>> labels_spherical_to_lm_;

    /// The label of Cartesian atomic orbitals.
    /// lm_labels_cartesian_[l][m] returns the label for an orbital
    /// with angular momentum quantum number l and index m
    std::vector<std::vector<std::string>> lm_labels_cartesian_ = {
        {"S"},
        {"PX", "PY", "PZ"},
        {"DX2", "DXY", "DXZ", "DY2", "DYZ", "DZ2"},
        {"FX3", "FX2Y", "FX2Z", "FXY2", "FXYZ", "FXZ2", "FY3", "FY2Z", "FYZ2", "FZ3"}};

    /// The list of all AOs with their properties
    std::vector<AOInfo> aoinfo_vec_;

    /// A map from atomic number to its atomic orbitals
    std::map<int, std::vector<std::vector<int>>> atom_to_aos_;

    /// The AOs spanned by the subspace selected by the user
    /// AO position, subspace position, coefficient
    std::vector<std::tuple<int, int, double>> subspace_;

    /// Counter for the actual number of subspace orbitals
    int subspace_counter_;

    /// A map from <atomic number, relative index> to atom normal
    /// An atom normal is a 3D unit vector where the p orbitals are projected onto:
    /// px, py, pz (3 orbitals) -> nx * px + ny * py + nz * pz (1 orbital)
    std::map<std::pair<int, int>, psi::Vector3> atom_normals_;

    /// Debug flag
    bool debug_ = false;

    /// The startup function
    void startup();

    /// Parse the options object
    void parse_subspace();

    /// Parse the options object
    bool parse_subspace_entry(const std::string& s);

    /// Parse the AO basis set
    void parse_basis_set();
};

/// Make a projector using wfn and options with pruned atomic p orbitals for molecular pi orbitals
/// @param wfn: A Psi4 Wavefunction object
/// @param options: A ForteOptions object
/// @param atom_normals: The direction of 'pz' orbital on each atom
std::shared_ptr<psi::Matrix> make_aosubspace_projector(psi::SharedWavefunction wfn,
                                                       std::shared_ptr<ForteOptions> options,
                                                       const pybind11::dict& atom_normals);
} // namespace forte

#endif // _aosubspace_h_
