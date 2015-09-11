/*
 *@BEGIN LICENSE
 *
 * Libadaptive: an ab initio quantum chemistry software package
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with this program; if not, write to the Free Software Foundation, Inc.,
 * 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 *
 *@END LICENSE
 */

#ifndef _fci_solver_h_
#define _fci_solver_h_

#include <libmints/wavefunction.h>
#include <liboptions/liboptions.h>
#include <physconst.h>

#include "fci_vector.h"

#include "helpers.h"
#include "integrals.h"
#include "string_lists.h"
#include "reference.h"


namespace psi{ namespace forte{

/**
 * @brief The FCISolver class
 * This class performs Full CI calculations.
 */
class FCISolver
{
public:
    // ==> Class Constructor and Destructor <==

    /**
     * @brief FCISolver
     * @param active_dim The dimension of the active orbital space
     * @param core_mo A vector of doubly occupied orbitals
     * @param active_mo A vector of active orbitals
     * @param na Number of alpha electrons
     * @param nb Number of beta electrons
     * @param multiplicity The spin multiplicity (2S + 1).  1 = singlet, 2 = doublet, ...
     * @param symmetry The irrep of the FCI wave function
     * @param ints An integral object
     */
    FCISolver(Dimension active_dim, std::vector<size_t> core_mo,
              std::vector<size_t> active_mo, size_t na, size_t nb,
              size_t multiplicity, size_t symmetry, ForteIntegrals* ints, std::shared_ptr<MOSpaceInfo> mo_space_info);

    ~FCISolver() {}

    /// Compute the FCI energy
    double compute_energy();

    /// Return a reference object
    Reference reference();

    /// When set to true before calling compute_energy(), it will test the
    /// reduce density matrices.  Watch out, this function is very slow!
    void test_rdms(bool value) {test_rdms_ = value;}

    void set_print(int value) {print_ = value;}
private:

    // ==> Class Data <==

    /// The Dimension object for the active space
    Dimension active_dim_;

    /// The orbitals frozen at the CI level
    std::vector<size_t> core_mo_;

    /// The orbitals treated at the CI level
    std::vector<size_t> active_mo_;

    /// A object that stores string information
    std::shared_ptr<StringLists> lists_;

    /// The molecular integrals
    ForteIntegrals* ints_;

    /// The FCI energy
    double energy_;

    /// The FCI wave function
    std::shared_ptr<FCIWfn> C_;

    /// The number of irreps
    int nirrep_;
    /// The symmetry of the wave function
    int symmetry_;
    /// The number of alpha electrons
    size_t na_;
    /// The number of beta electrons
    size_t nb_;
    /// The multiplicity (2S + 1) of the state to target.
    /// (1 = singlet, 2 = doublet, 3 = triplet, ...)
    size_t multiplicity_;
    /// The number of roots
    int nroot_;
    /// The number of trial guess vectors to generate per root
    size_t ntrial_per_root_ = 10;
    /// Test the RDMs?
    bool test_rdms_ = false;
    ///
    int print_ = 0;

    // ==> Class functions <==

    /// All that happens before we compute the energy
    void startup();
    /// The mo_space_info object
    std::shared_ptr<MOSpaceInfo> mo_space_info_;

    /// Initial CI wave function guess
    std::vector<std::vector<std::tuple<size_t, size_t, size_t, double> > >
    initial_guess(FCIWfn& diag,size_t n,size_t multiplicity,
                  std::shared_ptr<FCIIntegrals> fci_ints);
};


/**
 * @brief The FCI class
 * This class implements FCI
 */
class FCI : public Wavefunction
{
public:

    // ==> Class Constructor and Destructor <==

    /**
     * Constructor
     * @param wfn The main wavefunction object
     * @param options The main options object
     * @param ints A pointer to an allocated integral object
     */
    FCI(boost::shared_ptr<Wavefunction> wfn, Options &options, ForteIntegrals* ints, std::shared_ptr<MOSpaceInfo> mo_space_info);

    ~FCI();

    // ==> Class Interface <==

    /// Compute the energy
    virtual double compute_energy();

    /// Return a reference object
    Reference reference();

private:

    // ==> Class data <==

    /// A reference to the options object
    Options& options_;
    /// The molecular integrals
    ForteIntegrals* ints_;
    /// The information about the molecular orbital spaces
    std::shared_ptr<MOSpaceInfo> mo_space_info_;
    /// The information about the molecular orbital spaces
    FCISolver* fcisolver_ = nullptr;

    // ==> Class functions <==

    /// All that happens before we compute the energy
    void startup();
};

}}

#endif // _fci_solver_h_
