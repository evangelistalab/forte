/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2020 by its authors (see COPYING, COPYING.LESSER, AUTHORS).
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


#define _DEBUG_AOSUBSPACE_ 0

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
 *    // Create an AOSubspace object
 *    AOSubspace aosub(subspace_str,wfn->molecule(),wfn->basisset());
 *
 *    // Compute the subspaces
 *    aosub.find_subspace();
 *
 *    // Get the subspaces
 *    std::vector<int> subspace = aosub.subspace();
 *
 *    // Build a projector
 *    psi::SharedMatrix Ps =
 * aosub.build_projector(subspace,molecule,min_basis,basis);
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
 *                3) '(nlm)'   : select the n-th level with angular momentum l
 * and component m
 *                               e.g. '(2pz)', '(3dzz)', '(3dxx-yy)'
 *                               n = 1, 2, 3, ...
 *                               l = 's', 'p', 'd', 'f', 'g', ...
 *                               m = 'x', 'y', 'z', 'xy', 'xz', 'yz', 'zz',
 * 'xx-yy'
 *
 *    Valid options include:
 *
 *    ["C"] - all carbon atoms
 *    ["C","N"] - all carbon and nitrogen atoms
 *    ["C1"] - carbon atom #1
 *    ["C1-3"] - carbon atoms #1, #2, #3
 *    ["C(2p)"] - the 2p subset of all carbon atoms
 *    ["C(1s,2s)"] - the 1s/2s subsets of all carbon atoms
 *    ["C1-3(2s)"] - the 2s subsets of carbon atoms #1, #2, #3
 */
class AOSubspace {
  public:
    // ==> Constructors <==

    // Simple constructor
    AOSubspace(std::shared_ptr<psi::Molecule> molecule, std::shared_ptr<psi::BasisSet> basis);
    // Constructor with list of subspaces
    AOSubspace(std::vector<std::string> subspace_str, std::shared_ptr<psi::Molecule> molecule,
               std::shared_ptr<psi::BasisSet> basis);

    // ==> User's interface <==

    // Adds a subspace, e.g. add_subspace("C(1s,2s)")
    void add_subspace(std::string);

    // Compute the AOs in the subspace
    void find_subspace();

    // Return the index of the AOs that span the subspace selected
    const std::vector<int>& subspace();

    psi::SharedMatrix build_projector(const std::vector<int>& subspace,
                                      std::shared_ptr<psi::Molecule> molecule,
                                      std::shared_ptr<psi::BasisSet> min_basis,
                                      std::shared_ptr<psi::BasisSet> large_basis);

    /// Return a vector of labels for each atomic orbital.  This function
    /// accepts
    /// an optional argument that indicates the formatting that will be fed to
    /// boost::format.
    ///
    /// The field available for printing are:
    ///   1. Atom number (int)
    ///   2. Atom label, e.g. "C" (string)
    ///   3. Atom count, e.g. 3 = third atom of a given kind (int)
    ///   4. Energy level (n), 2 = 2s or 2p (int)
    ///   5. l/m label, e.g. "2px" (string)
    ///
    /// @arg str_format A string that specifies the output formatting
    std::vector<std::string> aolabels(std::string str_format = "%2$s%3$d (%4$d%5$s)") const;

    /// Return a vector of AOInfo objects
    const std::vector<AOInfo>& aoinfo() const;

  private:
    /// The vector of subspace descriptors passed by the user
    std::vector<std::string> subspace_str_;
    /// The molecule
    std::shared_ptr<psi::Molecule> molecule_;
    /// The AO basis set
    std::shared_ptr<psi::BasisSet> basis_;

    /// The label of Cartesian atomic orbitals.
    /// lm_labels_cartesian_[l][m] returns the label for an orbital
    /// with angular momentum quantum number l and index m
    std::vector<std::vector<std::string>> lm_labels_cartesian_;

    /// The l-label of atomic orbitals.
    /// l_labels_[l] returns the label for an orbital
    /// with angular momentum quantum number l
    std::vector<std::string> l_labels_;

    /// The label of Spherical atomic orbitals.
    /// lm_labels_sperical_[l][m] returns the label for an orbital
    /// with angular momentum quantum number l and index m
    std::vector<std::vector<std::string>> lm_labels_sperical_;

    std::map<std::string, std::vector<std::pair<int, int>>> labels_sperical_to_lm_;

    /// The list of all AOs with their properties
    std::vector<AOInfo> aoinfo_vec_;

    std::map<int, std::vector<std::vector<int>>> atom_to_aos_;

    /// The AOs spanned by the subspace selected by the user
    std::vector<int> subspace_;

    /// The AOs spanned by the subspace selected by the user
    std::vector<std::string> ao_info_;

    /// Counts how many copies of each element are there

    bool debug_ = false;

    /// The startup function
    void startup();

    /// Parse the options object
    void parse_subspace();

    /// Parse the options object
    void parse_subspace_entry(const std::string& s);

    /// Parse the AO basis set
    void parse_basis_set();
};

// Helper function to make a projector using info in wfn and options
psi::SharedMatrix make_aosubspace_projector(psi::SharedWavefunction wfn,
                                            std::shared_ptr<ForteOptions> options);
} // namespace forte

#endif // _aosubspace_h_
