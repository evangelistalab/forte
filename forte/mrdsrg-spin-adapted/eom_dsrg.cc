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

// #include "helpers/timer.h"
// #include "sa_mrdsrg.h"
// using namespace psi;
// namespace forte {
// double SA_MRDSRG::compute_eom() {
//     // IP singles. Should add foptions.
//     size_t nhole = mo_space_info_->dimension("GENERALIZED HOLE").sum();
//     size_t nocc = mo_space_info_->dimension("RESTRICTED_DOCC").sum();
//     size_t nact = mo_space_info_->dimension("ACTIVE").sum();
//     auto EOM_Hbar = BTF_->build(tensor_type_, "EOM-Hbar", {"hh"}, true);
//     EOM_Hbar_mat_ = std::make_shared<psi::Matrix>("EOM-Hbar-Matrx", nhole, nhole);

//     /// Should do spin adaptation.
//     EOM_Hbar["nm"] -= Hbar1_["nm"];
//     // EOM_Hbar["mv"] = -Hbar1_["mu"] * L1_["uv"] + Hbar2_["mwux"] * L2_["uxwv"];
//     // EOM_Hbar["wx"] = -Hbar1_["vu"] * L1_["ux"] * L1_["wv"] + Hbar1_["vu"] * L2_["uwvx"] +
//     //                  0.5 * Hbar2_["yzuv"] * L1_["wy"] * L2_["uvzx"] -
//     //                  0.5 * Hbar2_["yzuv"] * L1_["vx"] * L2_["uwyz"] +
//     //                  0.25 * Hbar2_["yzuv"] * L3_["uvwyzx"];

//     EOM_Hbar.iterate([&](const std::vector<size_t>& i, const std::vector<SpinType>&,
//                          double& value) { EOM_Hbar_mat_->set(i[0], i[1], value); });

//     /// Overlap matrix
//     std::shared_ptr<psi::Matrix> S = std::make_shared<psi::Matrix>("EOM-S-sub", nhole, nhole);
//     std::shared_ptr<psi::Matrix> Sevec = std::make_shared<psi::Matrix>("S-evec", nhole, nhole);
//     std::shared_ptr<psi::Vector> Seval = std::make_shared<psi::Vector>("S-eval", nhole);
//     S->identity();
//     L1_.iterate([&](const std::vector<size_t>& i, const std::vector<SpinType>&, double& value) {
//         S->set(i[0] + nocc, i[1] + nocc, value);
//     });
//     S->diagonalize(Sevec, Seval);

//     EOM_Hbar_mat_ = psi::linalg::triplet(Sevec, EOM_Hbar_mat_, Sevec, true, false, false);

//     EOM_Hbar_mat_->diagonalize(Sevec, Seval);

//     Seval->print();

//     return Seval->get(0);
// }
// } // namespace forte