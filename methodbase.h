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

#ifndef _methodbase_h_
#define _methodbase_h_

#include <fstream>

#include <liboptions/liboptions.h>
#include <libmints/wavefunction.h>

#include "integrals.h"
#include "tensor_basic.h"
#include "tensor_labeled.h"
#include "tensor_product.h"
#include "tensor_blocked.h"

namespace psi{

class PSIO;
class Chkpt;

namespace libadaptive{

/**
 * @brief The MethodBase class
 * This class provides basic functions to write electronic structure
 * pilot codes using the Tensor classes
 */
class MethodBase : public Wavefunction
{
protected:

    // => Class data <= //

    /// The molecular integrals required by MethodBase
    ExplorerIntegrals* ints_;

//    BlockedTensor Ha;
//    BlockedTensor Hb;
//    BlockedTensor Fa;
//    BlockedTensor Fb;

    // => Class initialization and termination <= //

    /// Called in the constructor
    void startup();
    /// Called in the destructor
    void cleanup();
public:

    // => Constructors <= //

    MethodBase(boost::shared_ptr<Wavefunction> wfn, Options &options, ExplorerIntegrals* ints);
    ~MethodBase();

//    /// The print level
//    int print_;
//    /// A reference to the wavefunction object
//    boost::shared_ptr<Wavefunction> wfn_;
//    /// A reference to the option object
//    Options& options_;
//    /// The number of irriducible representations
//    int nirrep_;
//    /// The wave function symmetry
//    int wavefunction_symmetry_;
//    /// The number of molecular orbitals
//    int nmo_;
//    /// The energy of the reference
//    double E0_;

//    /// The number of alpha electrons
//    int nalpha_;
//    /// The number of beta electrons
//    int nbeta_;
//    /// The number of molecular orbitals per irrep
//    Dimension nmopi_;
//    /// The number of molecular orbitals per irrep
//    Dimension doccpi_;
//    /// The number of molecular orbitals per irrep
//    Dimension soccpi_;
//    /// The number of alpha electrons per irrep
//    Dimension nalphapi_;
//    /// The number of beta electrons per irrep
//    Dimension nbetapi_;
//    /// The nuclear repulsion energy
//    double nuclear_repulsion_energy_;
//    /// The reference occupation numbers
//    MOTensor2 No_;
//    /// The reference complementary occupation numbers (1 - No_)
//    MOTensor2 Nv_;
//    /// The reference one-particle density matrix
//    MOTensor2 G1_;
//    /// The reference one-hole density matrix
//    MOTensor2 E1_;
//    /// The one-electron integrals Fock matrix
//    MOTensor2 H1_;
//    /// The generalized Fock matrix
//    MOTensor2 F_;
//    /// The two electron antisymmetrized integrals storeds as V[p][q][r][s] = <pq||rs>
//    BlockedTensor V_;



//    /// Build the generalized Fock matrix using the one-particle density matrix
//    void build_fock();
//    /// Sort the molecular integrals
//    void sort_integrals();
};

}} // End Namespaces

#endif // _methodbase_h_
