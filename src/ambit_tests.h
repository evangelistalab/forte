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

#ifndef _ambit_tests_h_
#define _ambit_tests_h_

#include <fstream>

#include <liboptions/liboptions.h>
#include <libmints/wavefunction.h>

#include "integrals.h"
#include "ambit/blocked_tensor.h"
#include "reference.h"

namespace psi{

namespace forte{

/**
 * @brief The MethodBase class
 * This class provides basic functions to write electronic structure
 * pilot codes using the Tensor classes
 */
class AmbitTests : public Wavefunction
{
protected:

    // => Class data <= //

    /// The reference object
    Reference reference_;

    /// The molecular integrals required by MethodBase
    ForteIntegrals* ints_;

    /// The number of correlated orbitals per irrep (excluding frozen core and virtuals)
    Dimension ncmopi_;
    /// The number of restricted doubly occupied orbitals per irrep (core)
    Dimension rdoccpi_;
    /// The number of active orbitals per irrep (active)
    Dimension actvpi_;
    /// The number of restricted unoccupied orbitals per irrep (virtual)
    Dimension ruoccpi_;

    /// List of alpha core MOs
    std::vector<size_t> acore_mos;
    /// List of alpha active MOs
    std::vector<size_t> aactv_mos;
    /// List of alpha virtual MOs
    std::vector<size_t> avirt_mos;

    /// List of beta core MOs
    std::vector<size_t> bcore_mos;
    /// List of beta active MOs
    std::vector<size_t> bactv_mos;
    /// List of beta virtual MOs
    std::vector<size_t> bvirt_mos;

    /// Map from all the MOs to the alpha core
    std::map<size_t,size_t> mos_to_acore;
    /// Map from all the MOs to the alpha active
    std::map<size_t,size_t> mos_to_aactv;
    /// Map from all the MOs to the alpha virtual
    std::map<size_t,size_t> mos_to_avirt;

    /// Map from all the MOs to the beta core
    std::map<size_t,size_t> mos_to_bcore;
    /// Map from all the MOs to the beta active
    std::map<size_t,size_t> mos_to_bactv;
    /// Map from all the MOs to the beta virtual
    std::map<size_t,size_t> mos_to_bvirt;

    /// The flow parameter
    double s_;

    /// Threshold for the Taylor expansion of f(z) = (1-exp(-z^2))/z
    double taylor_threshold_;
    /// Order of the Taylor expansion of f(z) = (1-exp(-z^2))/z
    int taylor_order_;

    ambit::TensorType tensor_type_;

    // => Tensors <= //

    ambit::BlockedTensor H;
    ambit::BlockedTensor F;
    ambit::BlockedTensor V;
    ambit::BlockedTensor Delta1;
    ambit::BlockedTensor Delta2;
    ambit::BlockedTensor R1;
    ambit::BlockedTensor R2;
    ambit::BlockedTensor T1;
    ambit::BlockedTensor T2;

    // => Class initialization and termination <= //

    /// Called in the constructor
    void startup();
    /// Called in the destructor
    void cleanup();
    /// Print a summary of the options
    void print_summary();

    /// Computes the t2 amplitudes for three different cases of spin (alpha all, beta all, and alpha beta)
    void compute_t2();

    /// Computes the t1 amplitudes for three different cases of spin (alpha all, beta all, and alpha beta)
    void compute_t1();

    // Print levels
    int print_;

public:

    // => Constructors <= //

    AmbitTests(Reference reference,boost::shared_ptr<Wavefunction> wfn, Options &options, ForteIntegrals* ints);

    ~AmbitTests();

    /// Compute the DSRG-MRPT2 energy
    double compute_energy();

    /// The energy of the reference
    double Eref;

    /// The frozen-core energy
    double frozen_core_energy;
};

}} // End Namespaces

#endif // _ambit_tests_h_
