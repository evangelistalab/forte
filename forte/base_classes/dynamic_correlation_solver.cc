/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2023 by its authors (see COPYING, COPYING.LESSER,
 * AUTHORS).
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

#include "psi4/libmints/matrix.h"

#include "integrals/integrals.h"
#include "base_classes/dynamic_correlation_solver.h"
#include "base_classes/mo_space_info.h"
#include "base_classes/rdms.h"
#include "base_classes/state_info.h"

namespace forte {
DynamicCorrelationSolver::DynamicCorrelationSolver(std::shared_ptr<RDMs> rdms,
                                                   std::shared_ptr<SCFInfo> scf_info,
                                                   std::shared_ptr<ForteOptions> options,
                                                   std::shared_ptr<ForteIntegrals> ints,
                                                   std::shared_ptr<MOSpaceInfo> mo_space_info)
    : ints_(ints), mo_space_info_(mo_space_info), rdms_(rdms), scf_info_(scf_info),
      foptions_(options) {
    startup();
}

void DynamicCorrelationSolver::startup() {
    Enuc_ = ints_->nuclear_repulsion_energy();
    Efrzc_ = ints_->frozen_core_energy();

    print_ = foptions_->get_int("PRINT");

    eri_df_ = false;
    ints_type_ = foptions_->get_str("INT_TYPE");
    if (ints_type_ == "CHOLESKY" || ints_type_ == "DF" || ints_type_ == "DISKDF") {
        eri_df_ = true;
    }

    diis_start_ = foptions_->get_int("DSRG_DIIS_START");
    diis_freq_ = foptions_->get_int("DSRG_DIIS_FREQ");
    diis_min_vec_ = foptions_->get_int("DSRG_DIIS_MIN_VEC");
    diis_max_vec_ = foptions_->get_int("DSRG_DIIS_MAX_VEC");
    if (diis_min_vec_ < 1) {
        diis_min_vec_ = 1;
    }
    if (diis_max_vec_ <= diis_min_vec_) {
        diis_max_vec_ = diis_min_vec_ + 4;
    }
    if (diis_freq_ < 1) {
        diis_freq_ = 1;
    }
}

void DynamicCorrelationSolver::set_state_weights_map(
    const std::map<StateInfo, std::vector<double>>& state_to_weights) {
    state_to_weights_ = state_to_weights;
}

double DynamicCorrelationSolver::compute_reference_energy() {
    // Identical to the one in CASSCF_ORB_GRAD class.
    // Eref = Enuc + Eclosed + Fclosed["uv"] * D1["uv"] + 0.5 * (uv|xy) * D2["uxvy"]
    double Eref = Enuc_;

    std::shared_ptr<psi::Matrix> Fclosed;
    double Eclosed;
    auto dim_start = psi::Dimension(mo_space_info_->nirrep());
    auto dim_end = mo_space_info_->dimension("INACTIVE_DOCC");
    std::tie(Fclosed, std::ignore, Eclosed) = ints_->make_fock_inactive(dim_start, dim_end);
    Eref += Eclosed;

    auto nactv = mo_space_info_->size("ACTIVE");
    auto Fc = ambit::Tensor::build(ambit::CoreTensor, "F closed", {nactv, nactv});
    auto actv_relative_mos = mo_space_info_->relative_mo("ACTIVE");
    Fc.iterate([&](const std::vector<size_t>& i, double& value) {
        size_t h1, h2, p, q;
        std::tie(h1, p) = actv_relative_mos[i[0]];
        std::tie(h2, q) = actv_relative_mos[i[1]];
        if (h1 == h2)
            value = Fclosed->get(h1, p, q);
    });
    Eref += Fc("uv") * rdms_->SF_G1()("uv");

    auto actv_mos = mo_space_info_->corr_absolute_mo("ACTIVE");
    auto V = ints_->aptei_ab_block(actv_mos, actv_mos, actv_mos, actv_mos);
    Eref += 0.5 * V("uvxy") * rdms_->SF_G2()("uvxy");

    return Eref;
}

void DynamicCorrelationSolver::clean_checkpoints() {
    if (not t1_file_chk_.empty()) {
        if (remove(t1_file_chk_.c_str()) != 0) {
            perror("Error when deleting T1 checkpoint.");
        }
    }
    if (not t2_file_chk_.empty()) {
        if (remove(t2_file_chk_.c_str()) != 0) {
            perror("Error when deleting T2 checkpoint.");
        }
    }
}

std::shared_ptr<DynamicCorrelationSolver> make_dynamic_correlation_solver(
    const std::string& /*type*/, std::shared_ptr<ForteOptions> /*options*/,
    std::shared_ptr<ForteIntegrals> /*ints*/, std::shared_ptr<MOSpaceInfo> /*mo_space_info*/) {
    // TODO fill and return objects!
    //    return DynamicCorrelationSolver();
    return std::shared_ptr<DynamicCorrelationSolver>();
}

} // namespace forte
