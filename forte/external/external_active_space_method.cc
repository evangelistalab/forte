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

#include <fstream>

#include "ambit/tensor.h"

#include "psi4/psi4-dec.h"
#include "psi4/libpsi4util/PsiOutStream.h"
#include "psi4/libpsi4util/process.h"

#include "lib/json/json.hpp"

#include "base_classes/rdms.h"
#include "base_classes/forte_options.h"
#include "base_classes/mo_space_info.h"

#include "integrals/active_space_integrals.h"

#include "helpers/printing.h"

#include "external_active_space_method.h"

using namespace nlohmann;

namespace forte {

class MOSpaceInfo;

auto read_json_file() {
    std::ifstream i("rdms.json");
    json j;
    i >> j;
    return j;
}

ExternalActiveSpaceMethod::ExternalActiveSpaceMethod(StateInfo state, size_t nroot,
                                                     std::shared_ptr<MOSpaceInfo> mo_space_info,
                                                     std::shared_ptr<ActiveSpaceIntegrals> as_ints)
    : ActiveSpaceMethod(state, nroot, mo_space_info, as_ints) {
    size_t ninact_docc = mo_space_info->size("INACTIVE_DOCC");
    na_ = state.na() - ninact_docc;
    nb_ = state.nb() - ninact_docc;
    nactv_ = mo_space_info->size("ACTIVE");
}

void ExternalActiveSpaceMethod::set_options(std::shared_ptr<ForteOptions> options) {
    twopdc_ = (options->get_str("TWOPDC") != "ZERO");
    threepdc_ = (options->get_str("THREEPDC") != "ZERO");
}

double ExternalActiveSpaceMethod::compute_energy() {
    print_method_banner({"External Active Space Solver"});

    // call python
    auto j = read_json_file();

    double energy = j["energy"]["data"];

    // Read 1-DM
    std::vector<std::tuple<int, int, double>> gamma1 = j["gamma1"]["data"];

    g1a_ = ambit::Tensor::build(ambit::CoreTensor, "g1a", std::vector<size_t>(2, nactv_));
    g1b_ = ambit::Tensor::build(ambit::CoreTensor, "g1b", std::vector<size_t>(2, nactv_));

    for (auto it1 = std::begin(gamma1); it1 != std::end(gamma1); ++it1) {
        size_t spin_case = std::get<0>(*it1) % 2;
        if (spin_case == 0) {
            size_t e1 = std::get<0>(*it1) / 2;
            size_t e2 = std::get<1>(*it1) / 2;
            g1a_.data()[e1 * nactv_ + e2] = std::get<2>(*it1);
        }
        if (spin_case == 1) {
            size_t e1 = (std::get<0>(*it1) - 1) / 2;
            size_t e2 = (std::get<1>(*it1) - 1) / 2;
            g1b_.data()[e1 * nactv_ + e2] = std::get<2>(*it1);
        }
    }

    // These must be allocated otherwise the code crashes
    g2aa_ = ambit::Tensor::build(ambit::CoreTensor, "g2aa", std::vector<size_t>(4, nactv_));
    g2ab_ = ambit::Tensor::build(ambit::CoreTensor, "g2ab", std::vector<size_t>(4, nactv_));
    g2bb_ = ambit::Tensor::build(ambit::CoreTensor, "g2bb", std::vector<size_t>(4, nactv_));

    // Read 2-DM
    if (j.contains("gamma2")) {
        std::vector<std::tuple<int, int, int, int, double>> gamma2 = j["gamma2"]["data"];

        for (auto it2 = std::begin(gamma2); it2 != std::end(gamma2); ++it2) {
            size_t e1 = std::get<0>(*it2) / 2;
            size_t e2 = std::get<1>(*it2) / 2;
            size_t e3 = std::get<2>(*it2) / 2;
            size_t e4 = std::get<3>(*it2) / 2;

            bool spin1 = (std::get<0>(*it2) % 2 == 0);
            bool spin2 = (std::get<1>(*it2) % 2 == 0);
            bool spin3 = (std::get<2>(*it2) % 2 == 0);
            bool spin4 = (std::get<3>(*it2) % 2 == 0);

            if (spin1 && spin2 && spin3 && spin4) {
                // Read form aaaa
                g2aa_.data()[e1 * nactv_ * nactv_ * nactv_ + e2 * nactv_ * nactv_ + e3 * nactv_ +
                             e4] = std::get<4>(*it2);
            } else if (!spin1 && !spin2 && !spin3 && !spin4) {
                // Read from bbbb
                g2bb_.data()[e1 * nactv_ * nactv_ * nactv_ + e2 * nactv_ * nactv_ + e3 * nactv_ +
                             e4] = std::get<4>(*it2);
            } else if (spin1 && !spin2 && spin3 && !spin4) {
                // Read from abab (should be enough)
                g2ab_.data()[e1 * nactv_ * nactv_ * nactv_ + e2 * nactv_ * nactv_ + e3 * nactv_ +
                             e4] = std::get<4>(*it2);
            }
        }
    } else {
        psi::outfile->Printf("\nThe json file does not contain data for the 2-RDM");
    }

    // These must be allocated otherwise the code crashes
    g3aaa_ = ambit::Tensor::build(ambit::CoreTensor, "g3aaa", std::vector<size_t>(6, nactv_));
    g3aab_ = ambit::Tensor::build(ambit::CoreTensor, "g3aab", std::vector<size_t>(6, nactv_));
    g3abb_ = ambit::Tensor::build(ambit::CoreTensor, "g3abb", std::vector<size_t>(6, nactv_));
    g3bbb_ = ambit::Tensor::build(ambit::CoreTensor, "g3bbb", std::vector<size_t>(6, nactv_));

    // Read 3-DM
    if (j.contains("gamma3")) {
        std::vector<std::tuple<int, int, int, int, int, int, double>> gamma3 = j["gamma3"]["data"];

        for (auto it3 = std::begin(gamma3); it3 != std::end(gamma3); ++it3) {
            size_t e1 = std::get<0>(*it3) / 2;
            size_t e2 = std::get<1>(*it3) / 2;
            size_t e3 = std::get<2>(*it3) / 2;
            size_t e4 = std::get<3>(*it3) / 2;
            size_t e5 = std::get<4>(*it3) / 2;
            size_t e6 = std::get<5>(*it3) / 2;

            bool spin1 = (std::get<0>(*it3) % 2 == 0);
            bool spin2 = (std::get<1>(*it3) % 2 == 0);
            bool spin3 = (std::get<2>(*it3) % 2 == 0);
            bool spin4 = (std::get<3>(*it3) % 2 == 0);
            bool spin5 = (std::get<4>(*it3) % 2 == 0);
            bool spin6 = (std::get<5>(*it3) % 2 == 0);

            if (spin1 && spin2 && spin3 && spin4 && spin5 && spin6) {
                // Read from aaaaaa
                g3aaa_
                    .data()[e1 * nactv_ * nactv_ * nactv_ * nactv_ * nactv_ +
                            e2 * nactv_ * nactv_ * nactv_ * nactv_ + e3 * nactv_ * nactv_ * nactv_ +
                            e4 * nactv_ * nactv_ + e5 * nactv_ + e6] = std::get<6>(*it3);
            } else if (spin1 && spin2 && !spin3 && spin4 && spin5 && !spin6) {
                // Read from aabaab
                g3aab_
                    .data()[e1 * nactv_ * nactv_ * nactv_ * nactv_ * nactv_ +
                            e2 * nactv_ * nactv_ * nactv_ * nactv_ + e3 * nactv_ * nactv_ * nactv_ +
                            e4 * nactv_ * nactv_ + e5 * nactv_ + e6] = std::get<6>(*it3);
            } else if (spin1 && !spin2 && !spin3 && spin4 && !spin5 && !spin6) {
                // Read from abbabb
                g3abb_
                    .data()[e1 * nactv_ * nactv_ * nactv_ * nactv_ * nactv_ +
                            e2 * nactv_ * nactv_ * nactv_ * nactv_ + e3 * nactv_ * nactv_ * nactv_ +
                            e4 * nactv_ * nactv_ + e5 * nactv_ + e6] = std::get<6>(*it3);
            } else if (!spin1 && !spin2 && !spin3 && !spin4 && !spin5 && !spin6) {
                // Read from bbbbbb
                g3bbb_
                    .data()[e1 * nactv_ * nactv_ * nactv_ * nactv_ * nactv_ +
                            e2 * nactv_ * nactv_ * nactv_ * nactv_ + e3 * nactv_ * nactv_ * nactv_ +
                            e4 * nactv_ * nactv_ + e5 * nactv_ + e6] = std::get<6>(*it3);
            }
        }
    } else {
        psi::outfile->Printf("\nThe json file does not contain data for the 3-RDM");
    }

    // Read reference energy
    energies_.push_back(energy);

    psi::Process::environment.globals["CURRENT ENERGY"] = energy;
    psi::Process::environment.globals["FCI ENERGY"] = energy;

    return energy;
}

std::vector<std::shared_ptr<RDMs>>
ExternalActiveSpaceMethod::rdms(const std::vector<std::pair<size_t, size_t>>& root_list,
                                int max_rdm_level, RDMsType type) {
    std::vector<std::shared_ptr<RDMs>> refs;

    if (type == RDMsType::spin_dependent) {
        if (max_rdm_level == 1) {
            refs.emplace_back(std::make_shared<RDMsSpinDependent>(g1a_, g1b_));
        }
        if (max_rdm_level == 2) {
            refs.emplace_back(std::make_shared<RDMsSpinDependent>(g1a_, g1b_, g2aa_, g2ab_, g2bb_));
        }
        if (max_rdm_level == 3) {
            refs.emplace_back(std::make_shared<RDMsSpinDependent>(g1a_, g1b_, g2aa_, g2ab_, g2bb_,
                                                                  g3aaa_, g3aab_, g3abb_, g3bbb_));
        }
    } else {
        ambit::Tensor g1sf, g2sf, g3sf;
        g1sf = g1a_.clone();
        g1sf("pq") += g1b_("pq");
        if (max_rdm_level > 1) {
            g2sf = g2aa_.clone();
            g2sf("pqrs") += g2ab_("pqrs") + g2ab_("qpsr");
            g2sf("pqrs") += g2bb_("pqrs");
        }
        if (max_rdm_level > 2) {
            g3sf = g3aaa_.clone();
            g3sf("pqrstu") += g3aab_("pqrstu") + g3aab_("prqsut") + g3aab_("qrptus");
            g3sf("pqrstu") += g3abb_("pqrstu") + g3abb_("qprtsu") + g3abb_("rpqust");
            g3sf("pqrstu") += g3bbb_("pqrstu");
        }
        if (max_rdm_level == 1)
            refs.emplace_back(std::make_shared<RDMsSpinFree>(g1sf));
        if (max_rdm_level == 2)
            refs.emplace_back(std::make_shared<RDMsSpinFree>(g1sf, g2sf));
        if (max_rdm_level == 3)
            refs.emplace_back(std::make_shared<RDMsSpinFree>(g1sf, g2sf, g3sf));
    }
    return refs;
}

std::vector<std::shared_ptr<RDMs>>
ExternalActiveSpaceMethod::transition_rdms(const std::vector<std::pair<size_t, size_t>>&,
                                           std::shared_ptr<ActiveSpaceMethod>, int, RDMsType) {
    throw std::runtime_error("ExternalActiveSpaceMethod::transition_rdms is not implemented!");
    return std::vector<std::shared_ptr<RDMs>>();
}

} // namespace forte
