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

#ifndef _so_mrdsrg_h_
#define _so_mrdsrg_h_

#include <fstream>
#include <boost/assign.hpp>

#include <liboptions/liboptions.h>
#include <libmints/wavefunction.h>

#include "helpers.h"
#include "integrals.h"
#include "ambit/blocked_tensor.h"
#include "reference.h"
#include "blockedtensorfactory.h"

namespace psi{

namespace libadaptive{

/**
 * @brief The SOMRDSRG class
 * This class implements the MR-DSRG(2) using a spin orbital formalism
 */
class SOMRDSRG : public Wavefunction
{
protected:

    // => Class data <= //

    /// The reference object
    Reference reference_;

    /// The molecular integrals required by MethodBase
    ExplorerIntegrals* ints_;

    /// The MOSpaceInfo object
    std::shared_ptr<MOSpaceInfo> mo_space_info_;

    /// The number of correlated orbitals per irrep (excluding frozen core and virtuals)
    Dimension ncmopi_;
    /// The number of restricted doubly occupied orbitals per irrep (core)
    Dimension rdoccpi_;
    /// The number of active orbitals per irrep (active)
    Dimension actvpi_;
    /// The number of restricted unoccupied orbitals per irrep (virtual)
    Dimension ruoccpi_;

    /// List of spin orbital core MOs
    std::vector<size_t> core_mos;
    /// List of alpha active MOs
    std::vector<size_t> actv_mos;
    /// List of alpha virtual MOs
    std::vector<size_t> virt_mos;

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

    /// Source operator
    std::string source_;

    /// Threshold for the Taylor expansion of f(z) = (1-exp(-z^2))/z
    double taylor_threshold_;
    /// Order of the Taylor expansion of f(z) = (1-exp(-z^2))/z
    int taylor_order_;

    ambit::TensorType tensor_type_;
    std::shared_ptr<BlockedTensorFactory> BTF;

    // => Tensors <= //

    ambit::BlockedTensor H;
    ambit::BlockedTensor F;
    ambit::BlockedTensor V;
    ambit::BlockedTensor DFL;
    ambit::BlockedTensor Gamma1;
    ambit::BlockedTensor Eta1;
    ambit::BlockedTensor Lambda2;
    ambit::BlockedTensor Lambda3;
    ambit::BlockedTensor Delta1;
    ambit::BlockedTensor Delta2;
    ambit::BlockedTensor RDelta1;
    ambit::BlockedTensor RDelta2;
    ambit::BlockedTensor T1;
    ambit::BlockedTensor T2;
    ambit::BlockedTensor RExp1;  // < one-particle exponential for renormalized Fock matrix
    ambit::BlockedTensor RExp2;  // < two-particle exponential for renormalized integral
    ambit::BlockedTensor Hbar1;  // < one-body term of effective Hamiltonian
    ambit::BlockedTensor Hbar2;  // < two-body term of effective Hamiltonian

    // => Class initialization and termination <= //

    /// Called in the constructor
    void startup();
    /// Called in the destructor
    void cleanup();
    /// Print a summary of the options
    void print_summary();

//    /// Renormalized denominator
//    double renormalized_denominator(double D);
//    double renormalized_denominator_amp(double V,double D);
//    double renormalized_denominator_emp2(double V,double D);
//    double renormalized_denominator_lamp(double V,double D);
//    double renormalized_denominator_lemp2(double V,double D);

//    /// Computes the t2 amplitudes for three different cases of spin (alpha all, beta all, and alpha beta)
//    void compute_t2();
//    void check_t2();
//    double T2norm;
//    double T2max;

//    /// Computes the t1 amplitudes for three different cases of spin (alpha all, beta all, and alpha beta)
//    void compute_t1();
//    void check_t1();
//    double T1norm;
//    double T1max;


public:

    // => Constructors <= //

    SOMRDSRG(Reference reference,
           boost::shared_ptr<Wavefunction> wfn,
           Options &options,
           ExplorerIntegrals* ints,
           std::shared_ptr<MOSpaceInfo> mo_space_info);

    ~SOMRDSRG();

    /// Compute the DSRG-MRPT2 energy
    double compute_energy();

    /// The energy of the reference
    double Eref;

    /// The frozen-core energy
    double frozen_core_energy;
};

}} // End Namespaces

#endif // _so_mrdsrg_h_
