/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2019 by its authors (see COPYING, COPYING.LESSER, AUTHORS).
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

#ifndef _cc_so_h_
#define _cc_so_h_

#include <cmath>

#include "psi4/liboptions/liboptions.h"
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

class CC_SO : public DynamicCorrelationSolver {
  public:
    // => Constructors <= //

    CC_SO(RDMs rdms, std::shared_ptr<SCFInfo> scf_info, std::shared_ptr<ForteOptions> options,
          std::shared_ptr<ForteIntegrals> ints, std::shared_ptr<MOSpaceInfo> mo_space_info);

    ~CC_SO();

    /// Compute the coupled cluster energy
    double compute_energy();

    /// Coupled cluster transformed Hamiltonian (not implemented)
    std::shared_ptr<ActiveSpaceIntegrals> compute_Heff_actv();

  protected:
    // => Class initialization and termination <= //

    /// Called in the constructor
    void startup();
    /// Called in the destructor
    void cleanup();

    // => Class data <= //

    /// Correlation level
    std::string corr_level_;

    /// The energy of the reference
    double Eref_;

    /// The frozen-core energy
    double Efrzc_;

    /// Convergence criteria
    double e_convergence_;
    double r_convergence_;
    int maxiter_;

    // => Triples related testing options <= //

    /// Include triples or not
    bool do_triples_;
    /// Perturbation order based on Fink Hamiltonian
    int fink_order_;

    /// Print levels
    int print_;

    /// List of alpha core SOs
    std::vector<size_t> acore_sos_;
    /// List of alpha virtual SOs
    std::vector<size_t> avirt_sos_;
    /// List of beta core SOs
    std::vector<size_t> bcore_sos_;
    /// List of beta virtual SOs
    std::vector<size_t> bvirt_sos_;

    /// List of core SOs
    std::vector<size_t> core_sos_;
    /// List of virtual SOs
    std::vector<size_t> virt_sos_;

    /// Number of spin orbitals
    size_t nso_;
    /// Number of core spin orbitals
    size_t nc_;
    /// Number of virtual spin orbitals
    size_t nv_;
    /// Number of spacial orbitals
    size_t nmo_;

    std::shared_ptr<BlockedTensorFactory> BTF_;
    TensorType tensor_type_;

    bool debug_flag_ = false;

    // => Tensors <= //

    ambit::BlockedTensor H_;
    ambit::BlockedTensor F_;
    ambit::BlockedTensor V_;
    ambit::BlockedTensor T1_;
    ambit::BlockedTensor T2_;
    ambit::BlockedTensor T3_;

    /// Diagonal elements of Fock matrix
    std::vector<double> Fd_;

    double Hbar0_;
    ambit::BlockedTensor Hbar1_;
    ambit::BlockedTensor Hbar2_;
    ambit::BlockedTensor Hbar3_;

    /// Print a summary of the options
    void print_summary();

    /// Number of amplitudes will be printed in amplitude summary
    int ntamp_;
    /// Print amplitudes summary
    void print_amp_summary(const std::string& name,
                           const std::vector<std::pair<std::vector<size_t>, double>>& list,
                           const double& norm, const size_t& number_nonzero);

    /// Computes T2 amplitudes
    void guess_t2();
    void update_t2();
    void check_t2();
    double rms_t2_;
    double T2norm_;
    double T2max_;

    /// Computes T1 amplitudes
    void guess_t1();
    void update_t1();
    void check_t1();
    double rms_t1_;
    double T1norm_;
    double T1max_;

    /// Compute T3 amplitudes
    void guess_t3();
    void update_t3();
    double rms_t3_ = 0.0;
    double T3norm_ = 0.0;
    double T3max_ = 0.0;

    void compute_ccsd_amp(BlockedTensor& H1, BlockedTensor& H2, BlockedTensor& T1,
                          BlockedTensor& T2, double& C0, BlockedTensor& C1, BlockedTensor& C2);
    void compute_ccsdt_amp(BlockedTensor& H1, BlockedTensor& H2, BlockedTensor& T1,
                           BlockedTensor& T2, BlockedTensor& T3, double& C0, BlockedTensor& C1,
                           BlockedTensor& C2, BlockedTensor& C3);

    void amplitudes(double factor, BlockedTensor& H1, BlockedTensor& H2, BlockedTensor& H3,
                    BlockedTensor& T1, BlockedTensor& T2, BlockedTensor& T3, double& C0,
                    BlockedTensor& C1, BlockedTensor& C2, BlockedTensor& C3);
};

std::unique_ptr<CC_SO> make_cc_so(RDMs rdms, std::shared_ptr<SCFInfo> scf_info,
                                  std::shared_ptr<ForteOptions> options,
                                  std::shared_ptr<ForteIntegrals> ints,
                                  std::shared_ptr<MOSpaceInfo> mo_space_info);
} // namespace forte

#endif // _cc_so_h_
