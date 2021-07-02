/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2021 by its authors (see COPYING, COPYING.LESSER, AUTHORS).
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

#ifndef _so_mrdsrg_h_
#define _so_mrdsrg_h_

#include <fstream>

#include "psi4/libmints/wavefunction.h"
#include "ambit/blocked_tensor.h"

#include "base_classes/dynamic_correlation_solver.h"
#include "base_classes/mo_space_info.h"
#include "base_classes/rdms.h"
#include "helpers/blockedtensorfactory.h"
#include "integrals/integrals.h"

namespace forte {

/**
 * @brief The SOMRDSRG class
 * This class implements the MR-DSRG(2) using a spin orbital formalism
 */
class SOMRDSRG : public DynamicCorrelationSolver {
  protected:
    // => Class data <= //

    int print_;

    /// The number of correlated orbitals per irrep (excluding frozen core and
    /// virtuals)
    psi::Dimension ncmopi_;
    /// The number of restricted doubly occupied orbitals per irrep (core)
    psi::Dimension rdoccpi_;
    /// The number of active orbitals per irrep (active)
    psi::Dimension actvpi_;
    /// The number of restricted unoccupied orbitals per irrep (virtual)
    psi::Dimension ruoccpi_;

    /// List of spin orbital core MOs
    std::vector<size_t> core_mos;
    /// List of alpha active MOs
    std::vector<size_t> actv_mos;
    /// List of alpha virtual MOs
    std::vector<size_t> virt_mos;

    /// Map from all the MOs to the alpha core
    std::map<size_t, size_t> mos_to_acore;
    /// Map from all the MOs to the alpha active
    std::map<size_t, size_t> mos_to_aactv;
    /// Map from all the MOs to the alpha virtual
    std::map<size_t, size_t> mos_to_avirt;

    /// Map from all the MOs to the beta core
    std::map<size_t, size_t> mos_to_bcore;
    /// Map from all the MOs to the beta active
    std::map<size_t, size_t> mos_to_bactv;
    /// Map from all the MOs to the beta virtual
    std::map<size_t, size_t> mos_to_bvirt;

    /// The flow parameter
    double s_;

    /// Source operator
    std::string source_;

    /// Threshold for the Taylor expansion of f(z) = (1-exp(-z^2))/z
    double taylor_threshold_;
    /// Order of the Taylor expansion of f(z) = (1-exp(-z^2))/z
    int taylor_order_;
    /// Robust routine to compute (1 - exp(-s D * D) / D
    double renormalized_denominator(double D);
    /// Taylor Expansion of [1 - exp(-s * D^2)] / D = sqrt(s) * (\sum_{n=1}
    /// \frac{1}{n!} (-1)^{n+1} Z^{2n-1})
    double Taylor_Exp(const double& Z, const int& n) {
        if (n > 0) {
            double value = Z, tmp = Z;
            for (int i = 0; i < n - 1; ++i) {
                tmp *= -1.0 * pow(Z, 2.0) / (static_cast<double>(i) + 2.0);
                value += tmp;
            }
            return value;
        }
        return 0.0;
    }

    ambit::TensorType tensor_type_;
    std::shared_ptr<BlockedTensorFactory> BTF;

    /// The energy of the reference wave function
    double E0_;
    double Hbar0;

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
    ambit::BlockedTensor RInvDelta1;
    ambit::BlockedTensor RInvDelta2;
    ambit::BlockedTensor T1;
    ambit::BlockedTensor T2;
    ambit::BlockedTensor DT1;
    ambit::BlockedTensor DT2;
    ambit::BlockedTensor R1;
    ambit::BlockedTensor R2;
    ambit::BlockedTensor C1;
    ambit::BlockedTensor C2;
    ambit::BlockedTensor O1;
    ambit::BlockedTensor O2;
    ambit::BlockedTensor RExp1; // < one-particle exponential for renormalized Fock matrix
    ambit::BlockedTensor RExp2; // < two-particle exponential for renormalized integral
    ambit::BlockedTensor Hbar1; // < one-body term of effective Hamiltonian
    ambit::BlockedTensor Hbar2; // < two-body term of effective Hamiltonian

    // => Class initialization and termination <= //

    /// Called in the constructor
    void startup();
    /// Called in the destructor
    void cleanup();
    /// Print a summary of the options
    void print_summary();

    // ==> Class functions <== //
    /// Compute MP2 amplitudes
    void mp2_guess();
    /// Compute Hbar
    double compute_hbar();
    /// Compute the commutator H <- [C,T]
    void H_eq_commutator_C_T(double factor, ambit::BlockedTensor& F, ambit::BlockedTensor& V,
                             ambit::BlockedTensor& T1, ambit::BlockedTensor& T2, double& H0,
                             ambit::BlockedTensor& H1, ambit::BlockedTensor& H2);

    /// T1 amplitude update
    void update_T1();
    /// T2 amplitude update
    void update_T2();

    //    /// Renormalized denominator
    //    double renormalized_denominator(double D);
    //    double renormalized_denominator_amp(double V,double D);
    //    double renormalized_denominator_emp2(double V,double D);
    //    double renormalized_denominator_lamp(double V,double D);
    //    double renormalized_denominator_lemp2(double V,double D);

    //    /// Computes the t2 amplitudes for three different cases of spin
    //    (alpha all, beta all, and alpha beta)
    //    void compute_t2();
    //    void check_t2();
    //    double T2norm;
    //    double T2max;

    //    /// Computes the t1 amplitudes for three different cases of spin
    //    (alpha all, beta all, and alpha beta)
    //    void compute_t1();
    //    void check_t1();
    //    double T1norm;
    //    double T1max;

  public:
    // => Constructors <= //

    SOMRDSRG(RDMs rdms, std::shared_ptr<SCFInfo> scf_info, std::shared_ptr<ForteOptions> options,
             std::shared_ptr<ForteIntegrals> ints, std::shared_ptr<MOSpaceInfo> mo_space_info);

    ~SOMRDSRG();

    /// Compute the DSRG-MRPT2 energy
    double compute_energy();

    /// DSRG transformed Hamiltonian (not implemented)
    std::shared_ptr<ActiveSpaceIntegrals> compute_Heff_actv();

    /// The energy of the reference
    double Eref;

    /// The frozen-core energy
    double frozen_core_energy;
};
}

#endif // _so_mrdsrg_h_
