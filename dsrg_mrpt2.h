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

#ifndef _dsrg_mrpt2_h_
#define _dsrg_mrpt2_h_

#include <fstream>
#include <boost/assign.hpp>

#include <liboptions/liboptions.h>
#include <libmints/wavefunction.h>

#include "integrals.h"
#include "ambit/blocked_tensor.h"
#include "reference.h"
#include "blockedtensorfactory.h"

namespace psi{

namespace libadaptive{

/**
 * @brief The MethodBase class
 * This class provides basic functions to write electronic structure
 * pilot codes using the Tensor classes
 */
class DSRG_MRPT2 : public Wavefunction
{
protected:

    // => Class data <= //

    enum sourceop{STANDARD, AMP, EMP2, LAMP, LEMP2};
    std::map<std::string, sourceop> sourcemap = boost::assign::map_list_of("STANDARD", STANDARD)("AMP", AMP)("EMP2", EMP2)("LAMP", LAMP)("LEMP2", LEMP2);

    /// The reference object
    Reference reference_;

    /// The molecular integrals required by MethodBase
    ExplorerIntegrals* ints_;

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

    /// Source operator
    std::string source_;

    /// Threshold for the Taylor expansion of f(z) = (1-exp(-z^2))/z
    double taylor_threshold_;
    /// Order of the Taylor expansion of f(z) = (1-exp(-z^2))/z
    int taylor_order_;

    TensorType tensor_type_;
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
    ambit::BlockedTensor RF;     // < effective 1st-order one-body term for 2nd-order Hbar
    ambit::BlockedTensor RV;     // < effective 1st-order two-body term for 2nd-order Hbar
    ambit::BlockedTensor Hbar1;  // < one-body term of effective Hamiltonian
    ambit::BlockedTensor Hbar2;  // < two-body term of effective Hamiltonian

    // => Class initialization and termination <= //

    /// Called in the constructor
    void startup();
    /// Called in the destructor
    void cleanup();
    /// Print a summary of the options
    void print_summary();

    /// Renormalized denominator
    double renormalized_denominator(double D);
    double renormalized_denominator_amp(double V,double D);
    double renormalized_denominator_emp2(double V,double D);
    double renormalized_denominator_lamp(double V,double D);
    double renormalized_denominator_lemp2(double V,double D);

    /// Computes the t2 amplitudes for three different cases of spin (alpha all, beta all, and alpha beta)
    void compute_t2();
    void check_t2();
    double T2norm;
    double T2max;

    /// Computes the t1 amplitudes for three different cases of spin (alpha all, beta all, and alpha beta)
    void compute_t1();
    void check_t1();
    double T1norm;
    double T1max;

    /// Renormalize Fock matrix and two-electron integral
    void renormalize_F();
    void renormalize_V();
    double renormalized_exp(double D) {return std::exp(-s_ * pow(D, 2.0));}
    double renormalized_exp_linear(double D) {return std::exp(-s_ * fabs(D));}

    /// Compute DSRG-PT2 correlation energy - Group of functions to calculate individual pieces of energy
    double compute_ref();
    double E_FT1();
    double E_VT1();
    double E_FT2();
    double E_VT2_2();
    double E_VT2_4PP();
    double E_VT2_4HH();
    double E_VT2_4PH();
    double E_VT2_6();

    /// Compute Hbar truncated to 2nd-order
    double Hbar0;
    void compute_Hbar1();
    void compute_Hbar2();

    // Print levels
    int print_;

    // Taylor Expansion of [1 - exp(-s * D^2)] / D = sqrt(s) * (\sum_{n=1} \frac{1}{n!} (-1)^{n+1} Z^{2n-1})
    double Taylor_Exp(const double& Z, const int& n){
        if(n > 0){
            double value = Z, tmp = Z;
            for(int x=0; x<(n-1); ++x){
                tmp *= -1.0 * pow(Z, 2.0) / (x+2);
                value += tmp;
            }
            return value;
        }else{return 0.0;}
    }

    // Taylor Expansion of [1 - exp(-s * |Z|)] / Z
    double Taylor_Exp_Linear(const double& Z, const int& n){
        bool Zabs = Z > 0.0 ? 1 : 0;
        if(n > 0){
            double value = 1, tmp = 1;
            for(int x=0; x<(n-1); ++x){
                tmp *= pow(-1.0, Zabs) * Z / (x+2);
                value += tmp;
            }
            return value * pow(-1.0, Zabs + 1);
        }else{return 0.0;}
    }

public:

    // => Constructors <= //

    DSRG_MRPT2(Reference reference,boost::shared_ptr<Wavefunction> wfn, Options &options, ExplorerIntegrals* ints);

    ~DSRG_MRPT2();

    /// Compute the DSRG-MRPT2 energy
    double compute_energy();

    /// The energy of the reference
    double Eref;

    /// The frozen-core energy
    double frozen_core_energy;

    /// Transfer integrals
    void transfer_integrals();
};

}} // End Namespaces

#endif // _dsrg_mrpt2_h_
