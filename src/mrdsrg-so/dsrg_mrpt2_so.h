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

#ifndef _dsrg_mrpt2_so_
#define _dsrg_mrpt2_so_

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
#include "mrdsrg-helper/dsrg_source.h"

using namespace ambit;

namespace forte {

class DSRG_MRPT2_SO : public DynamicCorrelationSolver {
  public:
    // => Constructors <= //

    DSRG_MRPT2_SO(RDMs rdms, std::shared_ptr<SCFInfo> scf_info,
                  std::shared_ptr<ForteOptions> options, std::shared_ptr<ForteIntegrals> ints,
                  std::shared_ptr<MOSpaceInfo> mo_space_info);

    /// Compute the DSRG-MRPT2 energy
    double compute_energy();

    /// DSRG transformed Hamiltonian (not implemented)
    std::shared_ptr<ActiveSpaceIntegrals> compute_Heff_actv();

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
    std::vector<size_t> acore_sos_;
    /// List of alpha active SOs
    std::vector<size_t> aactv_sos_;
    /// List of alpha virtual SOs
    std::vector<size_t> avirt_sos_;
    /// List of beta core SOs
    std::vector<size_t> bcore_sos_;
    /// List of beta active SOs
    std::vector<size_t> bactv_sos_;
    /// List of beta virtual SOs
    std::vector<size_t> bvirt_sos_;

    /// List of core SOs
    std::vector<size_t> core_sos_;
    /// List of active SOs
    std::vector<size_t> actv_sos_;
    /// List of virtual SOs
    std::vector<size_t> virt_sos_;

    /// Map from label to orbital indices
    std::map<std::string, std::vector<size_t>> label_to_sos_;

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

    /// The flow parameter
    double s_;

    /// Source operator
    std::string source_;
    /// The dsrg source operator
    std::shared_ptr<DSRG_SOURCE> dsrg_source_;
    /// Threshold for the Taylor expansion of f(z) = (1-exp(-z^2))/z
    double taylor_threshold_;

    std::shared_ptr<BlockedTensorFactory> BTF_;
    ambit::TensorType tensor_type_;

    // => Tensors <= //

    ambit::BlockedTensor H_; // OEI
    ambit::BlockedTensor F_; // Fock
    ambit::BlockedTensor Fc_; // inactive Fock
    ambit::BlockedTensor V_; // APTEI

    ambit::BlockedTensor V1_; // scaled by 2 * s * exp(-s * D) - (1 - exp(-s * D^2)) / D^2
    ambit::BlockedTensor V2_; // scaled by D * exp(-s * D^2)

    ambit::BlockedTensor M2_; // renormalized APTEI
    ambit::BlockedTensor Mbar2_; // multiplier for M2
    ambit::BlockedTensor Mdbar2_; // M2 double bar intermediate

    ambit::BlockedTensor T2_; // T2AMP
    ambit::BlockedTensor Tbar2_; // T2 bar * Delta
    ambit::BlockedTensor Tdbar2_; // T2 double bar intermediate

    ambit::BlockedTensor D1_; // 1-PDM
    ambit::BlockedTensor C1_; // 1-HDM
    ambit::BlockedTensor D2_; // 2-PDM

    ambit::BlockedTensor E1_; // 1-RDM like with CI multiplier
    ambit::BlockedTensor E2_; // 2-RDM like with CI multiplier

    ambit::BlockedTensor z_; // z diagonal blocks
    ambit::BlockedTensor Z_; // z off-diagonal blocks
    ambit::BlockedTensor W_; // orbital gradients

    /// Diagonal elements of Fock matrix
    std::vector<double> Fd_;

    /// Print a summary of the options
    void print_summary();

    /// Compute scaled two-electron integrals
    void compute_v_scaled();

    /// Compute the t2 amplitudes
    void compute_t2();
    /// Compute numerically stable t2 bar
    void compute_t2_bar();
    /// Compute the t2 double bar
    void compute_t2_double_bar();

    /// Compute the renormalized integrals
    void compute_m2();
    /// Compute multiplier for M2
    void compute_m2_bar();
    /// Compute m2 double bar
    void compute_m2_double_bar();

    /// Compute the reference energy
    double compute_reference_energy();
    /// Compute the correlation energy
    double compute_correlation_energy();

    /// Compute energy gradients
    void compute_gradients();

    /// Compute the diagonal elements of z
    void compute_z_diag();
    /// Compute the vv block of z
    void compute_z_vv();
    /// Compute the cc block of z
    void compute_z_cc();
    /// Compute the aa block of z
    void compute_z_aa();
    /// Compute z
    void compute_z();
    /// Compute Z
    void compute_Z();
    /// Compute parts of Z that needs iterative
    void compute_z_iter();
    /// Compute the orbital response
    void compute_orb_grad();
};
} // namespace forte

#endif // _mrdsrg_so_h_
