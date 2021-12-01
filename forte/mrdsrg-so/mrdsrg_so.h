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

#ifndef _mrdsrg_so_h_
#define _mrdsrg_so_h_

#include <cmath>


#include "psi4/libmints/wavefunction.h"
#include "psi4/libpsio/psio.hpp"
#include "psi4/libpsio/psio.h"
#include "ambit/blocked_tensor.h"

#include "base_classes/forte_options.h"
#include "base_classes/dynamic_correlation_solver.h"
#include "base_classes/rdms.h"
#include "integrals/integrals.h"
#include "helpers/blockedtensorfactory.h"

using namespace ambit;

namespace forte {

class MRDSRG_SO : public DynamicCorrelationSolver {
  protected:
    // => Class initialization and termination <= //

    /// Called in the constructor
    void startup();
    /// Called in the destructor
    void cleanup();

    // => Class data <= //

    /// Print levels
    int print_;

    /// List of alpha core SOs
    std::vector<size_t> acore_sos;
    /// List of alpha active SOs
    std::vector<size_t> aactv_sos;
    /// List of alpha virtual SOs
    std::vector<size_t> avirt_sos;
    /// List of beta core SOs
    std::vector<size_t> bcore_sos;
    /// List of beta active SOs
    std::vector<size_t> bactv_sos;
    /// List of beta virtual SOs
    std::vector<size_t> bvirt_sos;

    /// List of core SOs
    std::vector<size_t> core_sos;
    /// List of active SOs
    std::vector<size_t> actv_sos;
    /// List of virtual SOs
    std::vector<size_t> virt_sos;

    /// Number of spin orbitals
    size_t nso_;
    /// Number of core spin orbitals
    size_t nc_;
    /// Number of active spin orbitals
    size_t na_;
    /// Number of virtual spin orbitals
    size_t nv_;
    /// Number of hole spin orbitals
    size_t nh_;
    /// Number of particle spin orbitals
    size_t np_;

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

    /// Map from space label to list of MOs
    std::map<char, std::vector<size_t>> label_to_spacemo;

    /// The flow parameter
    double s_;

    /// Source operator
    std::string source_;

    /// Threshold for the Taylor expansion of f(z) = (1-exp(-z^2))/z
    double taylor_threshold_;
    /// Order of the Taylor expansion of f(z) = (1-exp(-z^2))/z
    int taylor_order_;

    std::shared_ptr<BlockedTensorFactory> BTF_;
    TensorType tensor_type_;

    // => Tensors <= //

    ambit::BlockedTensor H;
    ambit::BlockedTensor F;
    ambit::BlockedTensor V;
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
    ambit::BlockedTensor RExp1; // < one-particle exponential for renormalized Fock matrix
    ambit::BlockedTensor RExp2; // < two-particle exponential for renormalized integral

    /// Diagonal elements of Fock matrix
    std::vector<double> Fd;

    /// Print a summary of the options
    void print_summary();

    /// Renormalized denominator
    double renormalized_denominator(double D);
    double renormalized_denominator_amp(double V, double D);
    double renormalized_denominator_emp2(double V, double D);
    double renormalized_denominator_lamp(double V, double D);
    double renormalized_denominator_lemp2(double V, double D);

    /// Number of amplitudes will be printed in amplitude summary
    int ntamp_;
    /// Print amplitudes summary
    void print_amp_summary(const std::string& name,
                           const std::vector<std::pair<std::vector<size_t>, double>>& list,
                           const double& norm, const size_t& number_nonzero);

    /// Threshold for amplitudes considered as intruders
    double intruder_tamp_;
    /// List of large amplitudes
    std::vector<std::pair<std::vector<size_t>, double>> lt1a;
    std::vector<std::pair<std::vector<size_t>, double>> lt1b;
    std::vector<std::pair<std::vector<size_t>, double>> lt2aa;
    std::vector<std::pair<std::vector<size_t>, double>> lt2ab;
    std::vector<std::pair<std::vector<size_t>, double>> lt2bb;
    /// Print intruder analysis
    void print_intruder(const std::string& name,
                        const std::vector<std::pair<std::vector<size_t>, double>>& list);

    /// Computes the t2 amplitudes for three different cases of spin (alpha all,
    /// beta all, and alpha beta)
    void guess_t2();
    void update_t2();
    void check_t2();
    double rms_t2;
    double T2norm;
    double T2max;

    /// Computes the t1 amplitudes for three different cases of spin (alpha all,
    /// beta all, and alpha beta)
    void guess_t1();
    void update_t1();
    void check_t1();
    double rms_t1;
    double T1norm;
    double T1max;

    /// Renormalize Fock matrix and two-electron integral
    void renormalize_F();
    void renormalize_V();
    double renormalized_exp(double D) { return std::exp(-s_ * pow(D, 2.0)); }
    double renormalized_exp_linear(double D) { return std::exp(-s_ * std::fabs(D)); }

    /// Effective Hamiltonian Hbar
    double Hbar0;
    ambit::BlockedTensor Hbar1;
    ambit::BlockedTensor Hbar2;
    void compute_hbar();
    void compute_qhbar();

    /// Compute zero-term term of commutator [H, T]
    void H1_T1_C0(BlockedTensor& H1, BlockedTensor& T1, const double& alpha, double& C0);
    void H2_T1_C0(BlockedTensor& H2, BlockedTensor& T1, const double& alpha, double& C0);
    void H1_T2_C0(BlockedTensor& H1, BlockedTensor& T2, const double& alpha, double& C0);
    void H2_T2_C0(BlockedTensor& H2, BlockedTensor& T2, const double& alpha, double& C0);

    /// Compute one-body term of commutator [H, T]
    void H1_T1_C1(BlockedTensor& H1, BlockedTensor& T1, const double& alpha, BlockedTensor& C1);
    void H1_T2_C1(BlockedTensor& H1, BlockedTensor& T2, const double& alpha, BlockedTensor& C1);
    void H2_T1_C1(BlockedTensor& H2, BlockedTensor& T1, const double& alpha, BlockedTensor& C1);
    void H2_T2_C1(BlockedTensor& H2, BlockedTensor& T2, const double& alpha, BlockedTensor& C1);
    void H3_T1_C1(BlockedTensor& H3, BlockedTensor& T1, const double& alpha, BlockedTensor& C1);
    void H3_T2_C1(BlockedTensor& H3, BlockedTensor& T2, const double& alpha, BlockedTensor& C1);

    /// Compute two-body term of commutator [H, T]
    void H2_T1_C2(BlockedTensor& H2, BlockedTensor& T1, const double& alpha, BlockedTensor& C2);
    void H1_T2_C2(BlockedTensor& H1, BlockedTensor& T2, const double& alpha, BlockedTensor& C2);
    void H2_T2_C2(BlockedTensor& H2, BlockedTensor& T2, const double& alpha, BlockedTensor& C2);
    void H3_T1_C2(BlockedTensor& H3, BlockedTensor& T1, const double& alpha, BlockedTensor& C2);
    void H3_T2_C2(BlockedTensor& H3, BlockedTensor& T2, const double& alpha, BlockedTensor& C2);

    /// Compute three-body term of commutator [H, T]
    void H2_T2_C3(BlockedTensor& H2, BlockedTensor& T2, const double& alpha, BlockedTensor& C3);

    // Taylor Expansion of [1 - exp(-s * D^2)] / D = sqrt(s) * (\sum_{n=1}
    // \frac{1}{n!} (-1)^{n+1} Z^{2n-1})
    double Taylor_Exp(const double& Z, const int& n) {
        if (n > 0) {
            double value = Z, tmp = Z;
            for (int x = 0; x < (n - 1); ++x) {
                tmp *= -1.0 * pow(Z, 2.0) / (x + 2);
                value += tmp;
            }
            return value;
        } else {
            return 0.0;
        }
    }

    // Taylor Expansion of [1 - exp(-s * |Z|)] / Z
    double Taylor_Exp_Linear(const double& Z, const int& n) {
        bool Zabs = Z > 0.0 ? 1 : 0;
        if (n > 0) {
            double value = 1, tmp = 1;
            for (int x = 0; x < (n - 1); ++x) {
                tmp *= pow(-1.0, Zabs) * Z / (x + 2);
                value += tmp;
            }
            return value * pow(-1.0, Zabs + 1);
        } else {
            return 0.0;
        }
    }

    // Non-Negative Integer Exponential
    size_t myPow(size_t x, size_t p) {
        size_t i = 1;
        for (size_t j = 1; j <= p; j++)
            i *= x;
        return i;
    }

  public:
    // => Constructors <= //

    MRDSRG_SO(RDMs rdms, std::shared_ptr<SCFInfo> scf_info, std::shared_ptr<ForteOptions> options,
              std::shared_ptr<ForteIntegrals> ints, std::shared_ptr<MOSpaceInfo> mo_space_info);

    ~MRDSRG_SO();

    /// Compute the DSRG-MRPT2 energy
    double compute_energy();

    /// DSRG transformed Hamiltonian (not implemented)
    std::shared_ptr<ActiveSpaceIntegrals> compute_Heff_actv();

    /// The energy of the reference
    double Eref;

    /// The frozen-core energy
    double frozen_core_energy;
};
} // namespace forte

#endif // _mrdsrg_so_h_
