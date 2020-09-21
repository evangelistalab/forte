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

#include "psi4/libpsi4util/process.h"

#include "external/json/json.hpp"

#include "base_classes/rdms.h"
#include "base_classes/forte_options.h"
#include "base_classes/mo_space_info.h"

#include "integrals/active_space_integrals.h"
//#include "sparse_ci/determinant.h"
//#include "sparse_ci/determinant_functions.hpp"
//#include "helpers/iterative_solvers.h"

//#include "helpers/helpers.h"
#include "helpers/printing.h"

#include "external_active_space_method.h"

//#ifdef HAVE_GA
//#include <ga.h>
//#include <macdecls.h>
//#endif

//#include "psi4/psi4-dec.h"

// using namespace psi;
using namespace nlohmann;

// int fci_debug_level = 4;

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
    //    set_e_convergence(options->get_double("E_CONVERGENCE"));
    //    set_r_convergence(options->get_double("R_CONVERGENCE"));
    //    set_print(options->get_int("PRINT"));
    //    set_root(options->get_int("ROOT"));

    twopdc_ = (options->get_str("TWOPDC") != "ZERO");
    threepdc_ = (options->get_str("THREEPDC") != "ZERO");
}

double ExternalActiveSpaceMethod::compute_energy() {
    print_method_banner({"External Active Space Solver"});

    // call python
    auto j = read_json_file();
    std::cout << j << std::endl;

    double energy = j["energy"]["data"];

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

    g1a_.print();
    g1b_.print();

    if (twopdc_) {
        std::vector<std::tuple<int, int, int, int, double>> gamma2 = j["gamma2"]["data"];

        g2aa_ = ambit::Tensor::build(ambit::CoreTensor, "g2aa", std::vector<size_t>(4, nactv_));
        g2ab_ = ambit::Tensor::build(ambit::CoreTensor, "g2ab", std::vector<size_t>(4, nactv_));
        g2bb_ = ambit::Tensor::build(ambit::CoreTensor, "g2bb", std::vector<size_t>(4, nactv_));

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
                // aaaa
                g2aa_.data()[e1 * nactv_ * nactv_ * nactv_ + e2 * nactv_ * nactv_ + e3 * nactv_ +
                             e4] = std::get<4>(*it2);
            } else if (!spin1 && !spin2 && !spin3 && !spin4) {
                // bbbb
                g2bb_.data()[e1 * nactv_ * nactv_ * nactv_ + e2 * nactv_ * nactv_ + e3 * nactv_ +
                             e4] = std::get<4>(*it2);
            } else {
                // abab abba
                g2ab_.data()[e1 * nactv_ * nactv_ * nactv_ + e2 * nactv_ * nactv_ + e3 * nactv_ +
                             e4] = std::get<4>(*it2);
            }
        }
    }

    g2aa_.print();
    g2ab_.print();
    g2bb_.print();

    // TODO (Nan) store the RDMs in ambit Tensors (like in the RDMs class)

    if (threepdc_) {
        std::vector<std::tuple<int, int, int, int, int, int, double>> gamma3 = j["gamma2"]["data"];

        g3aaa_ = ambit::Tensor::build(ambit::CoreTensor, "g3aaa", std::vector<size_t>(6, nactv_));
        g3aab_ = ambit::Tensor::build(ambit::CoreTensor, "g3aab", std::vector<size_t>(6, nactv_));
        g3abb_ = ambit::Tensor::build(ambit::CoreTensor, "g3abb", std::vector<size_t>(6, nactv_));
        g3bbb_ = ambit::Tensor::build(ambit::CoreTensor, "g3bbb", std::vector<size_t>(6, nactv_));

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

            int spin_case =
                int(spin1) + int(spin2) + int(spin3) + int(spin4) + int(spin5) + int(spin6);

            if (spin_case == 6) {
                g3aaa_
                    .data()[e1 * nactv_ * nactv_ * nactv_ * nactv_ * nactv_ +
                            e2 * nactv_ * nactv_ * nactv_ * nactv_ + e3 * nactv_ * nactv_ * nactv_ +
                            e4 * nactv_ * nactv_ + e5 * nactv_ + e6] = std::get<6>(*it3);
            } else if (spin_case == 4) {
                g3aab_
                    .data()[e1 * nactv_ * nactv_ * nactv_ * nactv_ * nactv_ +
                            e2 * nactv_ * nactv_ * nactv_ * nactv_ + e3 * nactv_ * nactv_ * nactv_ +
                            e4 * nactv_ * nactv_ + e5 * nactv_ + e6] = std::get<6>(*it3);
            } else if (spin_case == 2) {
                g3abb_
                    .data()[e1 * nactv_ * nactv_ * nactv_ * nactv_ * nactv_ +
                            e2 * nactv_ * nactv_ * nactv_ * nactv_ + e3 * nactv_ * nactv_ * nactv_ +
                            e4 * nactv_ * nactv_ + e5 * nactv_ + e6] = std::get<6>(*it3);
            } else {
                g3bbb_
                    .data()[e1 * nactv_ * nactv_ * nactv_ * nactv_ * nactv_ +
                            e2 * nactv_ * nactv_ * nactv_ * nactv_ + e3 * nactv_ * nactv_ * nactv_ +
                            e4 * nactv_ * nactv_ + e5 * nactv_ + e6] = std::get<6>(*it3);
            }
        }
    }

    energies_.push_back(energy);

    psi::Process::environment.globals["CURRENT ENERGY"] = energy;
    psi::Process::environment.globals["FCI ENERGY"] = energy;

    return energy;
}

std::vector<RDMs>
ExternalActiveSpaceMethod::rdms(const std::vector<std::pair<size_t, size_t>>& root_list,
                                int max_rdm_level) {
    // throw std::runtime_error("ExternalActiveSpaceMethod::rdms is not implemented!");
    std::vector<RDMs> refs;

    if (max_rdm_level <= 0)
        return refs;

    if (max_rdm_level == 1) {
        refs.push_back(RDMs(g1a_, g1b_));
    }
    if (max_rdm_level == 2) {
        refs.push_back(RDMs(g1a_, g1b_, g2aa_, g2ab_, g2bb_));
    }
    if (max_rdm_level == 3) {
        refs.push_back(RDMs(g1a_, g1b_, g2aa_, g2ab_, g2bb_, g3aaa_, g3aab_, g3abb_, g3bbb_));
    }

    return refs;
}

std::vector<RDMs> ExternalActiveSpaceMethod::transition_rdms(
    const std::vector<std::pair<size_t, size_t>>& /*root_list*/,
    std::shared_ptr<ActiveSpaceMethod> /*method2*/, int /*max_rdm_level*/) {
    std::vector<RDMs> refs;
    throw std::runtime_error("ExternalActiveSpaceMethod::transition_rdms is not implemented!");
    return refs;
}

} // namespace forte
