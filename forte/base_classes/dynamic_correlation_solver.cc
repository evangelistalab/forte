/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2022 by its authors (see COPYING, COPYING.LESSER,
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

#include "psi4/psi4-dec.h"
#include "psi4/libpsi4util/PsiOutStream.h"
#include "psi4/libmints/matrix.h"

#include "base_classes/dynamic_correlation_solver.h"
#include "base_classes/mo_space_info.h"
#include "helpers/printing.h"

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

    internal_amp_ = foptions_->get_str("INTERNAL_AMP");
    internal_amp_select_ = foptions_->get_str("INTERNAL_AMP_SELECT");
    if (internal_amp_ != "NONE") {
        auto gas_spaces = mo_space_info_->nonzero_gas_spaces();

        if (gas_spaces.size() == 1 and internal_amp_select_ != "ALL") {
            internal_amp_select_ = "ALL";
        }

        for (const std::string& gas_name : gas_spaces) {
            gas_actv_rel_mos_[gas_name] = mo_space_info_->pos_in_space(gas_name, "ACTIVE");
        }

        if (internal_amp_.find("SINGLES") != std::string::npos) {
            build_t1_internal_types();
        }
        if (internal_amp_.find("DOUBLES") != std::string::npos) {
            build_t2_internal_types();
        }
    }
}

void DynamicCorrelationSolver::build_t1_internal_types() {
    t1_internals_.clear();

    auto gas_spaces = mo_space_info_->nonzero_gas_spaces();
    int n_gas = gas_spaces.size();

    // excitation O-V
    for (int o = 0; o < n_gas; ++o) {
        for (int v = o + 1; v < n_gas; ++v) {
            t1_internals_.emplace_back(gas_spaces[o], gas_spaces[v], false);
        }
    }

    // pure internal
    if (internal_amp_select_ == "ALL") {
        for (int x = 0; x < n_gas; ++x) {
            t1_internals_.emplace_back(gas_spaces[x], gas_spaces[x], true);
        }
    }

    if (print_ > 2) {
        print_h2("T1 Internal Amplitudes Types");
        for (const auto& t : t1_internals_) {
            const auto& [gas1, gas2, pure] = t;
            psi::outfile->Printf("\n  %s -> %s; pure internal: %d", gas1.c_str(), gas2.c_str(),
                                 pure);
        }
    }
}

void DynamicCorrelationSolver::build_t2_internal_types() {
    t2_internals_.clear();

    auto gas_spaces = mo_space_info_->nonzero_gas_spaces();
    int n_gas = gas_spaces.size();

    // pure excitation
    for (int o1 = 0; o1 < n_gas; ++o1) {
        for (int o2 = 0; o2 < n_gas; ++o2) {
            int o_max = o1 < o2 ? o2 : o1;

            for (int v1 = o_max + 1; v1 < n_gas; ++v1) {
                for (int v2 = o_max + 1; v2 < n_gas; ++v2) {
                    t2_internals_.emplace_back(gas_spaces[o1], gas_spaces[o2], gas_spaces[v1],
                                               gas_spaces[v2], false);
                }
            }
        }
    }

    // semi-internals
    if (internal_amp_select_ != "OOVV") {

        for (int o = 0; o < n_gas; ++o) {
            for (int v = o + 1; v < n_gas; ++v) {

                // oo->ov
                t2_internals_.emplace_back(gas_spaces[o], gas_spaces[o], gas_spaces[o],
                                           gas_spaces[v], false);
                t2_internals_.emplace_back(gas_spaces[o], gas_spaces[o], gas_spaces[v],
                                           gas_spaces[o], false);

                // ov->vv
                t2_internals_.emplace_back(gas_spaces[o], gas_spaces[v], gas_spaces[v],
                                           gas_spaces[v], false);
                t2_internals_.emplace_back(gas_spaces[v], gas_spaces[o], gas_spaces[v],
                                           gas_spaces[v], false);

                // ox->vx
                for (int x = 0; x < n_gas; ++x) {
                    if (x == o or x == v)
                        continue;

                    t2_internals_.emplace_back(gas_spaces[x], gas_spaces[o], gas_spaces[x],
                                               gas_spaces[v], false);
                    t2_internals_.emplace_back(gas_spaces[x], gas_spaces[o], gas_spaces[v],
                                               gas_spaces[x], false);
                    t2_internals_.emplace_back(gas_spaces[o], gas_spaces[x], gas_spaces[v],
                                               gas_spaces[x], false);
                    t2_internals_.emplace_back(gas_spaces[o], gas_spaces[x], gas_spaces[x],
                                               gas_spaces[v], false);
                }
            }
        }
    }

    // pure internal
    if (internal_amp_select_ == "ALL") {

        for (int x1 = 0; x1 < n_gas; ++x1) {

            t2_internals_.emplace_back(gas_spaces[x1], gas_spaces[x1], gas_spaces[x1],
                                       gas_spaces[x1], true);

            for (int x2 = x1 + 1; x2 < n_gas; ++x2) {

                t2_internals_.emplace_back(gas_spaces[x1], gas_spaces[x2], gas_spaces[x1],
                                           gas_spaces[x2], true);
                t2_internals_.emplace_back(gas_spaces[x1], gas_spaces[x2], gas_spaces[x2],
                                           gas_spaces[x1], true);
                t2_internals_.emplace_back(gas_spaces[x2], gas_spaces[x1], gas_spaces[x2],
                                           gas_spaces[x1], true);
                t2_internals_.emplace_back(gas_spaces[x2], gas_spaces[x1], gas_spaces[x1],
                                           gas_spaces[x2], true);
            }
        }
    }

    // debug printing
    if (print_ > 2) {
        print_h2("T2 Internal Amplitudes Types");
        for (const auto& t : t2_internals_) {
            const auto& [gas1, gas2, gas3, gas4, pure] = t;
            psi::outfile->Printf("\n  %s,%s -> %s,%s; pure internal: %d", gas1.c_str(),
                                 gas2.c_str(), gas3.c_str(), gas4.c_str(), pure);
        }
    }
}

double DynamicCorrelationSolver::compute_reference_energy() {
    // Identical to the one in CASSCF_ORB_GRAD class.
    // Eref = Enuc + Eclosed + Fclosed["uv"] * D1["uv"] + 0.5 * (uv|xy) * D2["uxvy"]
    double Eref = Enuc_;

    psi::SharedMatrix Fclosed;
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
