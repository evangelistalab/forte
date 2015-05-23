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


namespace psi{ namespace libadaptive{

/**
 * @brief The FCISolver class
 * This class performs Full CI calculations.
 */
class FCISolver
{
public:
    // ==> Class Constructor and Destructor <==

    FCISolver(Dimension active_dim, std::vector<size_t> core_mo, std::vector<size_t> active_mo, size_t na, size_t nb, size_t symmetry, ExplorerIntegrals* ints);

    ~FCISolver() {}

    /// Compute the FCI energy
    double compute_energy();

    /// Return a reference object
    Reference reference();

    /// When set to true before calling compute_energy(), it will test the
    /// reduce density matrices.  Watch out, this function is very slow!
    void test_rdms(bool value) {test_rdms_ = value;}

private:

    // ==> Class Data <==

    /// The Dimension object for the active space
    Dimension active_dim_;

    // The orbitals frozen at the CI level
    std::vector<size_t> core_mo_;

    // The orbitals treated at the CI level
    std::vector<size_t> active_mo_;

    // A object that stores string information
    boost::shared_ptr<StringLists> lists_;

    /// The molecular integrals
    ExplorerIntegrals* ints_;

    double energy_;
    std::shared_ptr<FCIWfn> C_;

    /// The number of irreps
    size_t nirrep_;
    /// The symmetry of the wave function
    size_t symmetry_;
    /// The number of alpha electrons
    size_t na_;
    /// The number of beta electrons
    size_t nb_;
    /// The number of roots
    size_t nroot_;
    /// Test the RDMs?
    bool test_rdms_ = false;

    // ==> Class functions <==

    /// All that happens before we compute the energy
    void startup();
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
    FCI(boost::shared_ptr<Wavefunction> wfn, Options &options, ExplorerIntegrals* ints, std::shared_ptr<MOSpaceInfo> mo_space_info);

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
    ExplorerIntegrals* ints_;
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
