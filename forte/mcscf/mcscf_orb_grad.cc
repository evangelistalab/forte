﻿/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2024 by its authors (see COPYING, COPYING.LESSER, AUTHORS).
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

#include <ctype.h>
#include <numeric>

#include "psi4/psi4-dec.h"
#include "psi4/psifiles.h"
#include "psi4/libqt/qt.h"
#include "psi4/libmints/molecule.h"
#include "psi4/libmints/pointgrp.h"
#include "psi4/libmints/matrix.h"
#include "psi4/libmints/wavefunction.h"
#include "psi4/libmints/basisset.h"
#include "psi4/libmints/vector.h"
#include "psi4/libpsi4util/PsiOutStream.h"
#include "psi4/libpsi4util/process.h"
#include "psi4/libiwl/iwl.hpp"
#include "psi4/libpsio/psio.hpp"

#include "helpers/helpers.h"
#include "helpers/lbfgs/lbfgs.h"
#include "helpers/printing.h"
#include "helpers/timer.h"
#include "integrals/integrals.h"
#include "integrals/active_space_integrals.h"
#include "base_classes/rdms.h"

#include "mcscf/mcscf_orb_grad.h"

using namespace psi;
using namespace ambit;

namespace forte {

MCSCF_ORB_GRAD::MCSCF_ORB_GRAD(std::shared_ptr<ForteOptions> options,
                                 std::shared_ptr<MOSpaceInfo> mo_space_info,
                                 std::shared_ptr<ForteIntegrals> ints, bool freeze_core)
    : options_(options), mo_space_info_(mo_space_info), ints_(ints), freeze_core_(freeze_core) {
    startup();
}

void MCSCF_ORB_GRAD::startup() {
    // setup MO spaces
    setup_mos();

    // read and print options
    read_options();

    // nonredundant pairs
    nonredundant_pairs();

    // setup JK
    JK_ = ints_->jk();

    // allocate memory for tensors and matrices
    init_tensors();

    // compute the initial MO integrals
    build_mo_integrals();
}

void MCSCF_ORB_GRAD::setup_mos() {

    nirrep_ = mo_space_info_->nirrep();

    nsopi_ = ints_->nsopi();
    nmopi_ = mo_space_info_->dimension("ALL");

    if (freeze_core_) {
        ncmopi_ = mo_space_info_->dimension("CORRELATED");
    } else {
        ncmopi_ =
            mo_space_info_->dimension("FROZEN_DOCC") + mo_space_info_->dimension("CORRELATED");
    }

    ndoccpi_ = mo_space_info_->dimension("INACTIVE_DOCC");
    nfrzvpi_ = mo_space_info_->dimension("FROZEN_UOCC");

    if (freeze_core_) {
        nfrzcpi_ = mo_space_info_->dimension("FROZEN_DOCC");
        core_mos_ = mo_space_info_->absolute_mo("RESTRICTED_DOCC");
    } else {
        nfrzcpi_ = psi::Dimension(nirrep_);
        core_mos_ = mo_space_info_->absolute_mo("INACTIVE_DOCC");
    }

    nactvpi_ = mo_space_info_->dimension("ACTIVE");
    actv_mos_ = mo_space_info_->absolute_mo("ACTIVE");

    nso_ = nsopi_.sum();
    nmo_ = nmopi_.sum();
    ncmo_ = ncmopi_.sum();
    nactv_ = nactvpi_.sum();
    nfrzc_ = nfrzcpi_.sum();

    label_to_mos_.clear();

    if (freeze_core_) {
        label_to_mos_["f"] = mo_space_info_->absolute_mo("FROZEN_DOCC");
    } else {
        label_to_mos_["f"] = std::vector<size_t>();
    }
    label_to_mos_["c"] = core_mos_;
    label_to_mos_["a"] = actv_mos_;
    label_to_mos_["v"] = mo_space_info_->absolute_mo("RESTRICTED_UOCC");
    label_to_mos_["u"] = mo_space_info_->absolute_mo("FROZEN_UOCC");

    label_to_cmos_.clear();
    if (freeze_core_) {
        label_to_cmos_["c"] = mo_space_info_->corr_absolute_mo("RESTRICTED_DOCC");
        label_to_cmos_["a"] = mo_space_info_->corr_absolute_mo("ACTIVE");
        label_to_cmos_["v"] = mo_space_info_->corr_absolute_mo("RESTRICTED_UOCC");
    } else {
        label_to_cmos_["c"] = mo_space_info_->absolute_mo("INACTIVE_DOCC");
        label_to_cmos_["a"] = mo_space_info_->absolute_mo("ACTIVE");
        label_to_cmos_["v"] = mo_space_info_->absolute_mo("RESTRICTED_UOCC");
    }

    // in Pitzer ordering
    mos_rel_.resize(nmo_);
    for (int h = 0, offset = 0; h < nirrep_; ++h) {
        for (int i = 0; i < nmopi_[h]; ++i) {
            mos_rel_[i + offset] = std::make_pair(h, i);
        }
        offset += nmopi_[h];
    }

    // in Pitzer ordering
    mos_rel_space_.resize(nmo_);
    for (const std::string& space : {"f", "c", "a", "v", "u"}) {
        const auto& mos = label_to_mos_[space];
        for (size_t p = 0, size = mos.size(); p < size; ++p) {
            mos_rel_space_[mos[p]] = std::make_pair(space, p);
        }
    }

    // set up ambit spaces
    BlockedTensor::reset_mo_spaces();
    BlockedTensor::set_expert_mode(true);

    BlockedTensor::add_mo_space("f", "I,J", label_to_mos_["f"], NoSpin);
    BlockedTensor::add_mo_space("c", "i,j", core_mos_, NoSpin);
    BlockedTensor::add_mo_space("a", "t,u,v,w,y,x,z", actv_mos_, NoSpin);
    BlockedTensor::add_mo_space("v", "a,b", label_to_mos_["v"], NoSpin);
    BlockedTensor::add_mo_space("u", "A,B", label_to_mos_["u"], NoSpin);

    BlockedTensor::add_composite_mo_space("F", "M,N", {"f", "u"});
    BlockedTensor::add_composite_mo_space("g", "p,q,r,s", {"c", "a", "v"});
    BlockedTensor::add_composite_mo_space("G", "P,Q,R,S", {"f", "c", "a", "v", "u"});

    // if we are doing GAS
    gas_ref_ = mo_space_info_->nonzero_gas_spaces().size() > 1;
}

void MCSCF_ORB_GRAD::read_options() {
    print_ = options_->get_int("PRINT");
    debug_print_ = options_->get_bool("MCSCF_DEBUG_PRINTING");

    internal_rot_ = options_->get_bool("MCSCF_INTERNAL_ROT");

    ortho_trans_algo_ = options_->get_str("MCSCF_ORB_ORTHO_TRANS");

    // zero rotations
    zero_rots_.resize(nirrep_);
    auto zero_rots = options_->get_gen_list("MCSCF_ZERO_ROT");

    if (not zero_rots.empty()) {
        for (size_t i = 0, npairs = zero_rots.size(); i < npairs; ++i) {
            py::list pair = zero_rots[i];
            if (pair.size() != 3) {
                outfile->Printf("\n  Error: invalid input of MCSCF_ZERO_ROT.");
                outfile->Printf("\n  Each entry should take an array of three numbers.");
                throw std::runtime_error("Invalid input of MCSCF_ZERO_ROT");
            }

            int irrep = py::cast<int>(pair[0]);
            if (irrep >= nirrep_ or irrep < 0) {
                outfile->Printf("\n  Error: invalid irrep in MCSCF_ZERO_ROT.");
                outfile->Printf("\n  Check the input irrep (start from 0) not to exceed %d",
                                nirrep_ - 1);
                throw std::runtime_error("Invalid irrep in MCSCF_ZERO_ROT");
            }

            int i1 = py::cast<int>(pair[1]) - 1;
            int i2 = py::cast<int>(pair[2]) - 1;
            size_t n = nmopi_[irrep];
            if (static_cast<size_t>(i1) >= n or i1 < 0 or static_cast<size_t>(i2) >= n or i2 < 0) {
                outfile->Printf("\n  Error: invalid orbital indices in MCSCF_ZERO_ROT.");
                outfile->Printf("\n  The input orbital indices (start from 1) should not exceed "
                                "%zu (number of orbitals in irrep %d)",
                                n, irrep);
                throw std::runtime_error("Invalid orbital indices in MCSCF_ZERO_ROT");
            }

            zero_rots_[irrep][i1].emplace(i2);
            zero_rots_[irrep][i2].emplace(i1);
        }
    }

    auto frza_rot = options_->get_int_list("MCSCF_ACTIVE_FROZEN_ORBITAL");
    auto actv_rel_mos = mo_space_info_->relative_mo("ACTIVE");
    if (not frza_rot.empty()) {
        for (size_t u : frza_rot) {
            if (u >= nactv_) {
                outfile->Printf("\n  Error: invalid indices in MCSCF_ACTIVE_FROZEN_ORBITAL.");
                outfile->Printf("\n  Active orbitals include all of those in GAS1-GAS6");
                outfile->Printf("\n  Input indices (0 based wrt active) should not exceed %zu.",
                                nactv_ - 1);
                throw std::runtime_error("Invalid indices in MCSCF_ACTIVE_FROZEN_ORBITAL");
            }

            // zero between orbital u and all others
            int irrep = actv_rel_mos[u].first;
            auto nu = actv_rel_mos[u].second;
            for (int p = 0; p < nmopi_[irrep]; ++p) {
                zero_rots_[irrep][nu].emplace(p);
                zero_rots_[irrep][p].emplace(nu);
            }
        }
    }

    if (debug_print_ and !zero_rots_.empty()) {
        print_h2("Orbital Rotations Ignored (User Defined)");
        outfile->Printf("\n    Both irrep and indices are zero-based.\n");
        for (int h = 0; h < nirrep_; ++h) {
            for (const auto& index_map : zero_rots_[h]) {
                auto p = index_map.first;
                for (const auto& q : index_map.second) {
                    if (p <= q) {
                        outfile->Printf("\n    irrep: %d, pair: (%4zu,%4zu)", h, p, q);
                    }
                }
            }
        }
    }
}

void MCSCF_ORB_GRAD::nonredundant_pairs() {
    // prepare indices for rotation pairs
    rot_mos_irrep_.clear();
    rot_mos_block_.clear();

    // for printing
    std::map<std::string, std::vector<int>> nrots{{"vc", std::vector<int>(nirrep_, 0)},
                                                  {"va", std::vector<int>(nirrep_, 0)},
                                                  {"ac", std::vector<int>(nirrep_, 0)}};

    // if we want to zero an orbital pair
    auto in_zero_rots = [&](int h, size_t i, size_t j) {
        if (zero_rots_[h].find(i) != zero_rots_[h].end()) {
            if (zero_rots_[h][i].find(j) != zero_rots_[h][i].end())
                return true;
        }
        return false;
    };

    for (const std::string block : {"vc", "va", "ac"}) {
        const auto& mos1 = label_to_mos_[block.substr(0, 1)];
        const auto& mos2 = label_to_mos_[block.substr(1, 1)];

        for (int i = 0, si = mos1.size(); i < si; ++i) {
            int hi = mos_rel_[mos1[i]].first;
            auto ni = mos_rel_[mos1[i]].second;

            for (int j = 0, sj = mos2.size(); j < sj; ++j) {
                if (hi != mos_rel_[mos2[j]].first) // skip if i, j in different irreps
                    continue;

                auto nj = mos_rel_[mos2[j]].second;

                if (in_zero_rots(hi, ni, nj))
                    continue;

                rot_mos_irrep_.emplace_back(hi, ni, nj);
                rot_mos_block_.emplace_back(block, i, j);
                nrots[block][hi] += 1;
            }
        }
    }

    // GASm-GASn with m != n rotations
    auto gas_spaces = mo_space_info_->nonzero_gas_spaces();
    if (gas_ref_) {
        const auto& mos = label_to_mos_["a"];

        // loop over GASm spaces
        for (int g0 = 0, space_size = gas_spaces.size(); g0 < space_size; ++g0) {
            auto g0_in_actv = mo_space_info_->pos_in_space(gas_spaces[g0], "ACTIVE");

            // loop over GASn spaces
            for (int g1 = g0 + 1; g1 < space_size; ++g1) {
                auto g1_in_actv = mo_space_info_->pos_in_space(gas_spaces[g1], "ACTIVE");

                // space name for printing
                std::string space_name = gas_spaces[g0].substr(3) + gas_spaces[g1].substr(3);
                nrots[space_name] = std::vector<int>(nirrep_, 0);

                // loop over indices in GASm
                for (size_t u : g0_in_actv) {
                    int hu = mos_rel_[mos[u]].first;
                    auto nu = mos_rel_[mos[u]].second;

                    // loop over indices in GASn
                    for (size_t v : g1_in_actv) {
                        if (hu != mos_rel_[mos[v]].first)
                            continue;

                        auto nv = mos_rel_[mos[v]].second;

                        if (in_zero_rots(hu, nu, nv))
                            continue;

                        rot_mos_irrep_.emplace_back(hu, nv, nu);
                        rot_mos_block_.emplace_back("aa", v, u);
                        nrots[space_name][hu] += 1;
                    }
                }
            }
        }
    }

    // GASn-GASn rotations
    if (internal_rot_) {
        const auto& mos = label_to_mos_["a"];

        for (auto& gas_space : gas_spaces) {
            auto g_in_actv = mo_space_info_->pos_in_space(gas_space, "ACTIVE");

            // space name for printing, convert to 1-based GAS
            std::string space_name = gas_space.substr(3) + gas_space.substr(3);
            nrots[space_name] = std::vector<int>(nirrep_, 0);

            for (int u = 0, size = static_cast<int>(g_in_actv.size()); u < size; ++u) {
                int hu = mos_rel_[mos[g_in_actv[u]]].first;
                auto nu = mos_rel_[mos[g_in_actv[u]]].second;

                for (int v = u + 1; v < size; ++v) {
                    if (hu != mos_rel_[mos[g_in_actv[v]]].first)
                        continue;

                    auto nv = mos_rel_[mos[g_in_actv[v]]].second;

                    if (in_zero_rots(hu, nu, nv))
                        continue;

                    rot_mos_irrep_.emplace_back(hu, nv, nu);
                    rot_mos_block_.emplace_back("aa", g_in_actv[v], g_in_actv[u]);
                    nrots[space_name][hu] += 1;
                }
            }
        }
    }

    nrot_ = rot_mos_irrep_.size();

    // printing
    std::map<std::string, std::string> space_map{
        {"c", "RESTRICTED_DOCC"}, {"a", "ACTIVE"}, {"v", "RESTRICTED_UOCC"}};

    print_h2("Independent Orbital Rotations");
    outfile->Printf("\n    %-33s", "ORBITAL SPACES");
    for (int h = 0; h < nirrep_; ++h) {
        outfile->Printf(" %6s", mo_space_info_->irrep_label(h).c_str());
    }
    outfile->Printf("\n    %s", std::string(33 + nirrep_ * 7, '-').c_str());

    for (const auto& key_value : nrots) {
        const auto& key = key_value.first;
        auto block1 = isdigit(key[0]) ? "GAS" + key.substr(0, 1) : space_map[key.substr(0, 1)];
        auto block2 = isdigit(key[1]) ? "GAS" + key.substr(1, 1) : space_map[key.substr(1, 1)];
        outfile->Printf("\n    %15s / %15s", block1.c_str(), block2.c_str());

        const auto& value = key_value.second;
        for (int h = 0; h < nirrep_; ++h) {
            outfile->Printf(" %6zu", value[h]);
        }
    }
    outfile->Printf("\n    %s", std::string(33 + nirrep_ * 7, '-').c_str());
}

void MCSCF_ORB_GRAD::init_tensors() {
    // save a copy of initial MO
    C0_ = ints_->Ca()->clone();
    C0_->set_name("MCSCF Initial Orbital Coefficients");
    C_ = ints_->Ca()->clone();
    C_->set_name("MCSCF Orbital Coefficients");

    // Fock matrices
    Fd_.resize(nmo_);

    auto tensor_type = ambit::CoreTensor;
    Fc_ = ambit::BlockedTensor::build(tensor_type, "Fc", {"GG"});
    F_ = ambit::BlockedTensor::build(tensor_type, "F", {"gg"});

    // two-electron integrals
    V_ = ambit::BlockedTensor::build(tensor_type, "V", {"Gaaa"});

    // 1-RDM and 2-RDM
    D1_ = ambit::BlockedTensor::build(tensor_type, "1RDM", {"aa"});
    D2_ = ambit::BlockedTensor::build(tensor_type, "2RDM", {"aaaa"});
    rdm1_ = std::make_shared<psi::Matrix>("1RDM", nactvpi_, nactvpi_);

    // orbital gradients related
    A_ = ambit::BlockedTensor::build(tensor_type, "A", {"gg"});

    std::vector<std::string> g_blocks{"ac", "vc", "va"};
    if (internal_rot_ or gas_ref_) {
        g_blocks.emplace_back("aa");
        Guu_ = ambit::BlockedTensor::build(CoreTensor, "Guu", {"aa"});
        Guv_ = ambit::BlockedTensor::build(CoreTensor, "Guv", {"aa"});
        jk_internal_ = ambit::BlockedTensor::build(CoreTensor, "tei_internal", {"aaa"});
        d2_internal_ = ambit::BlockedTensor::build(CoreTensor, "rdm_internal", {"aaa"});
    }
    g_ = ambit::BlockedTensor::build(tensor_type, "g", g_blocks);
    h_diag_ = ambit::BlockedTensor::build(tensor_type, "h_diag", g_blocks);

    grad_ = std::make_shared<psi::Vector>("Gradient Vector", nrot_);
    hess_diag_ = std::make_shared<psi::Vector>("Diagonal Hessian", nrot_);

    // orbital rotation related
    R_ = std::make_shared<psi::Matrix>("Skew-Symmetric Orbital Rotation", nmopi_, nmopi_);
    U_ = std::make_shared<psi::Matrix>("Orthogonal Transformation", nmopi_, nmopi_);
    U_->identity();
}

void MCSCF_ORB_GRAD::build_mo_integrals() {
    // form closed-shell Fock matrix
    build_fock_inactive();
    // form the MO 2e-integrals
    if (ints_->integral_type() == Custom) {
        fill_tei_custom(V_);
    } else {
        build_tei_from_ao();
    }
}

void MCSCF_ORB_GRAD::fill_tei_custom(ambit::BlockedTensor V) {
    std::vector<std::string> blocks{"caaa", "aaaa", "vaaa"};
    for (const std::string& block : blocks) {
        if (not V.is_block(block))
            throw std::runtime_error(block + " not found in TEI!");

        const auto& mo_g = label_to_cmos_[block.substr(0, 1)];
        const auto& mo_a = label_to_cmos_["a"];

        V.block(block).iterate([&](const std::vector<size_t>& i, double& value) {
            value = ints_->aptei_ab(mo_g[i[0]], mo_a[i[2]], mo_a[i[1]], mo_a[i[3]]);
        });
    }
}

void MCSCF_ORB_GRAD::build_tei_from_ao() {
    if (nactv_ == 0)
        return;

    // This function will do an integral transformation using the JK builder,
    // and return the integrals of type <px|uy> = (pu|xy).
    timer_on("Build (pu|xy) integrals");

    // Transform C matrix to C1 symmetry
    // JK does not support mixed symmetry needed for 4-index integrals (York 09/09/2020)
    auto aotoso = ints_->wfn()->aotoso();
    auto C_nosym = std::make_shared<psi::Matrix>(nso_, nmo_);

    // Transform from the SO to the AO basis for the C matrix
    // MO in Pitzer ordering and only keep the non-frozen MOs
    for (int h = 0, index = 0; h < nirrep_; ++h) {
        for (int i = 0; i < nmopi_[h]; ++i) {
            int nao = nso_, nso_h = nsopi_[h];

            if (!nso_h)
                continue;

            C_DGEMV('N', nao, nso_h, 1.0, aotoso->pointer(h)[0], nso_h, &C_->pointer(h)[0][i],
                    nmopi_[h], 0.0, &C_nosym->pointer()[0][index], nmo_);

            index += 1;
        }
    }

    // set up the active part of the C matrix
    auto Cact = std::make_shared<psi::Matrix>("Cact", nso_, nactv_);
    std::vector<std::shared_ptr<psi::Matrix>> Cact_vec(nactv_);

    for (size_t x = 0; x < nactv_; ++x) {
        auto Ca_nosym_vec = C_nosym->get_column(0, actv_mos_[x]);
        Cact->set_column(0, x, Ca_nosym_vec);

        std::string name = "Cact slice " + std::to_string(x);
        auto temp = std::make_shared<psi::Matrix>(name, nso_, 1);
        temp->set_column(0, 0, Ca_nosym_vec);
        Cact_vec[x] = temp;
    }

    // The following type of integrals are needed:
    // (pu|xy) = C_{Mp}^T C_{Nu} C_{Rx}^T C_{Sy} (MN|RS)
    //         = C_{Mp}^T C_{Nu} J_{MN}^{xy}
    //         = C_{Mp}^T J_{MN}^{xy} C_{Nu}

    JK_->set_do_K(false);
    std::vector<std::shared_ptr<psi::Matrix>>& Cl = JK_->C_left();
    std::vector<std::shared_ptr<psi::Matrix>>& Cr = JK_->C_right();
    Cl.clear();
    Cr.clear();

    // figure out memory bottleneck
    size_t mem_sys = psi::Process::environment.get_memory() * 0.85;
    size_t max_elements = nactv_ * nactv_ * nso_ * nso_ * sizeof(double);
    size_t n_buckets = max_elements / mem_sys + (max_elements % mem_sys ? 1 : 0);

    size_t n_pairs = nactv_ * (nactv_ + 1) / 2;
    size_t n_pairspb = n_pairs / n_buckets;
    size_t n_mod = n_pairs - n_buckets * n_pairspb;

    // throw for JK's strange "same" test in compute_D() of jk.cc (York 09/09/2020)
    if (n_pairspb == 1 and nirrep_ != 1) {
        outfile->Printf("\n  Error: Problem for JK in compute_D() in this case");
        outfile->Printf("\n  If there is 1 active orbitals, try RHF/ROHF of Psi4.");
        outfile->Printf("\n  If not, try to increase the memory or compute in C1 symmetry.");
        throw std::runtime_error("JK does not work in this case. Try C1 symmetry.");
    }

    // put all (x,y) pairs to a vector for easy splittig to buckets
    std::vector<std::tuple<int, int>> pairs;
    pairs.reserve(nactv_ * (nactv_ + 1) / 2);
    for (size_t x = 0; x < nactv_; ++x) {
        for (size_t y = x; y < nactv_; ++y) {
            pairs.emplace_back(x, y);
        }
    }

    // JK compute
    size_t nactv2 = nactv_ * nactv_;
    size_t nactv3 = nactv2 * nactv_;
    for (size_t N = 0, offset = 0; N < n_buckets; ++N) {
        size_t n_pairs = N < n_mod ? n_pairspb + 1 : n_pairspb;

        Cl.clear();
        Cr.clear();

        for (size_t i = 0; i < n_pairs; ++i) {
            Cl.push_back(Cact_vec[std::get<0>(pairs[i + offset])]);
            Cr.push_back(Cact_vec[std::get<1>(pairs[i + offset])]);
        }
        JK_->compute();

        // transform to MO and fill V_
        for (size_t i = 0; i < n_pairs; ++i) {
            auto x = std::get<0>(pairs[i + offset]);
            auto y = std::get<1>(pairs[i + offset]);

            auto half_trans = psi::linalg::triplet(C_nosym, JK_->J()[i], Cact, true, false, false);

            for (size_t p = 0; p < nmo_; ++p) {
                size_t np = mos_rel_space_[p].second;

                std::string block = mos_rel_space_[p].first + "aaa";
                auto& data = V_.block(block).data();

                for (size_t u = 0; u < nactv_; ++u) {
                    double value = half_trans->get(p, u);
                    data[np * nactv3 + u * nactv2 + x * nactv_ + y] = value;
                    data[np * nactv3 + u * nactv2 + y * nactv_ + x] = value;
                }
            }
        }

        offset += n_pairs;
    }

    timer_off("Build (pu|xy) integrals");
}

void MCSCF_ORB_GRAD::build_fock(bool rebuild_inactive) {
    if (rebuild_inactive) {
        build_fock_inactive();
    }

    build_fock_active();

    Fock_->add(F_closed_);
    Fock_->set_name("Fock_MO");

    format_fock(Fock_, F_);

    // fill in diagonal Fock in Pitzer ordering
    for (const std::string& space : {"c", "a", "v"}) {
        std::string block = space + space;
        auto mos = label_to_mos_[space];
        for (size_t i = 0, size = mos.size(); i < size; ++i) {
            Fd_[mos[i]] = F_.block(block).data()[i * size + i];
        }
    }

    if (debug_print_) {
        Fock_->print();
    }
}

void MCSCF_ORB_GRAD::build_fock_inactive() {
    /* F_inactive = Hcore + F_frozen + F_restricted
     *
     * F_frozen = D_{uv}^{frozen} * (2 * (uv|rs) - (us|rv))
     * D_{uv}^{frozen} = \sum_{i}^{frozen} C_{ui} * C_{vi}
     *
     * F_restricted = D_{uv}^{restricted} * (2 * (uv|rs) - (us|rv))
     * D_{uv}^{restricted} = \sum_{i}^{restricted} C_{ui} * C_{vi}
     *
     * u,v,r,s: AO indices; i: MO indices
     */

    auto Ftuple = ints_->make_fock_inactive(psi::Dimension(nirrep_), ndoccpi_);
    std::tie(F_closed_, std::ignore, e_closed_) = Ftuple;
    F_closed_->set_name("Fock_inactive");

    // put into Ambit BlockedTensor format
    format_fock(F_closed_, Fc_);

    if (debug_print_) {
        F_closed_->print();
        outfile->Printf("\n  Frozen-core energy   %20.15f", ints_->frozen_core_energy());
        outfile->Printf("\n  Closed-shell energy  %20.15f", e_closed_);
    }
}

void MCSCF_ORB_GRAD::build_fock_active() {
    // Implementation Notes (in AO basis)
    // F_active = D_{uv}^{active} * ( (uv|rs) - 0.5 * (us|rv) )
    // D_{uv}^{active} = \sum_{xy}^{active} C_{ux} * C_{vy} * Gamma1_{xy}

    Fock_ = ints_->make_fock_active_restricted(rdm1_);
    Fock_->set_name("Fock_active");

    if (debug_print_) {
        Fock_->print();
    }
}

void MCSCF_ORB_GRAD::format_fock(std::shared_ptr<psi::Matrix> Fock, ambit::BlockedTensor F) {
    F.iterate([&](const std::vector<size_t>& i, const std::vector<SpinType>&, double& value) {
        auto irrep_index_pair1 = mos_rel_[i[0]];
        auto irrep_index_pair2 = mos_rel_[i[1]];

        int h1 = irrep_index_pair1.first;

        if (h1 == irrep_index_pair2.first) {
            auto p = irrep_index_pair1.second;
            auto q = irrep_index_pair2.second;
            value = Fock->get(h1, p, q);
        } else {
            value = 0.0;
        }
    });
}

std::shared_ptr<psi::Matrix> MCSCF_ORB_GRAD::fock(std::shared_ptr<RDMs> rdms) {
    // put spin-summed 1RDM to psi4 Matrix
    auto rdm1 = tensor_to_matrix(rdms->SF_G1(), nactvpi_);

    // use ForteIntegrals to build Fock
    auto Ftuple = ints_->make_fock_inactive(psi::Dimension(nirrep_), ndoccpi_);
    auto Fock = std::get<0>(Ftuple);
    Fock->add(ints_->make_fock_active_restricted(rdm1));
    Fock->set_name("Fock");

    return Fock;
}

double MCSCF_ORB_GRAD::evaluate(std::shared_ptr<psi::Vector> x, std::shared_ptr<psi::Vector> g,
                                 bool do_g) {
    // if need to update orbitals and integrals
    if (update_orbitals(x)) {
        build_mo_integrals();
    }

    compute_reference_energy();

    // if need to compute gradient
    if (do_g) {
        build_fock();
        compute_orbital_grad();
        g->copy(*grad_);
    }

    return energy_;
}

bool MCSCF_ORB_GRAD::update_orbitals(std::shared_ptr<psi::Vector> x) {
    // test if need to update orbitals
    auto dR = std::make_shared<psi::Matrix>("Delta Orbital Rotation", nmopi_, nmopi_);

    for (size_t n = 0; n < nrot_; ++n) {
        int h, i, j;
        std::tie(h, i, j) = rot_mos_irrep_[n];

        dR->set(h, i, j, x->get(n));
        dR->set(h, j, i, -x->get(n));
    }

    dR->subtract(R_);

    // incoming x consistent with R_, no need to update orbitals
    if (dR->absmax() < numerical_zero_)
        return false;

    // officially save progress of dR
    R_->add(dR);

    // compute incremental step dU and test against previous step
    std::shared_ptr<psi::Matrix> dU;
    if (ortho_trans_algo_ == "PADE") {
        dU = dR->clone();
        dU->expm(3);
    } else if (ortho_trans_algo_ == "POWER") {
        dU = matrix_exponential(dR, 3);
    } else {
        dU = cayley_trans(dR);
    }
    test_orbital_rotations(dU, "WARNING: Active Orbitals Different from the Previous Step");

    // U_new = U_old * exp(dR)
    U_ = psi::linalg::doublet(U_, dU, false, false);
    U_->set_name("Orthogonal Transformation");

    // update orbitals
    C_->gemm(false, false, 1.0, C0_, U_, 0.0);
    if (ints_->integral_type() == Custom) {
        ints_->update_orbitals(C_, C_);
    } else {
        // here we push the new orbitals to the integrals object but do not update the integrals
        // for efficiency
        ints_->update_orbitals(C_, C_, false);
    }

    // printing
    if (debug_print_) {
        dR->print();
        R_->print();
        U_->print();
        C_->print();
    }

    return true;
}

std::shared_ptr<psi::Matrix>
MCSCF_ORB_GRAD::matrix_exponential(const std::shared_ptr<psi::Matrix>& A, int n) {
    auto U = std::make_shared<psi::Matrix>("U = exp(A)", A->rowspi(), A->colspi());
    U->identity();
    U->add(A);

    if (n > 1) {
        auto M = A->clone();
        M->set_name("M");

        for (int i = 1; i < n; ++i) {
            auto B = psi::linalg::doublet(M, A, false, false);
            B->scale(1.0 / (i + 1));
            U->add(B);
            M->copy(B);
        }
    }

    U->schmidt();
    return U;
}

std::shared_ptr<psi::Matrix> MCSCF_ORB_GRAD::cayley_trans(const std::shared_ptr<psi::Matrix>& A) {
    auto n = std::make_shared<psi::Matrix>("I + A / 2", A->rowspi(), A->colspi());
    n->identity();

    auto d = n->clone();
    d->set_name("(I - A / 2)^-1");

    auto a = A->clone();
    a->scale(0.5);

    n->add(a);
    d->subtract(a);
    d->general_invert();

    auto c = psi::linalg::doublet(n, d, false, false);
    c->set_name("(I + A / 2) (I - A / 2)^-1");
    return c;
}

std::vector<std::tuple<int, int, int>>
MCSCF_ORB_GRAD::test_orbital_rotations(const std::shared_ptr<psi::Matrix>& U,
                                        const std::string& warning_msg) {
    // the overlap between new and old orbitals is simply U
    // O = Cold^T S Cnew = Cold^T S Cold U = U
    // MOM projection index: Pj = sum_{i}^{actv} O_ij
    std::vector<std::tuple<int, int, int>> out;

    for (int h = 0; h < nirrep_; ++h) {
        std::vector<std::pair<double, int>> p;
        for (int j = 0; j < nmopi_[h]; ++j) {
            double sum = 0.0;
            for (int i = 0; i < nactvpi_[h]; ++i) {
                sum += std::fabs(U->get(h, i + ndoccpi_[h], j));
            }
            p.emplace_back(sum, j);
        }

        // sort the Pj in descending order
        std::sort(p.rbegin(), p.rend());

        // find are there any orbitals rotated outside active
        std::vector<int> i_old(nactvpi_[h]);
        std::iota(i_old.begin(), i_old.end(), ndoccpi_[h]);
        std::vector<int> i_new;

        for (int i = 0; i < nactvpi_[h]; ++i) {
            auto j = p[i].second;
            if (j < ndoccpi_[h] or j >= ndoccpi_[h] + nactvpi_[h]) {
                i_new.push_back(j);
            } else {
                auto it = std::find(i_old.begin(), i_old.end(), j);
                if (it != i_old.end())
                    i_old.erase(it);
            }
        }

        // print warning if orbitals are rotated outside active
        for (size_t i = 0, ni = i_old.size(); i < ni; ++i) {
            out.emplace_back(h, i_old[i], i_new[i]);
        }
    }

    if (not out.empty()) {
        print_h2(warning_msg, "!!!", "!!!");
        outfile->Printf("\n    Based on projections of maximum overlap method:");
        for (const auto& tup : out) {
            int h, i_old, i_new;
            std::tie(h, i_old, i_new) = tup;
            std::string i_old_str = std::to_string(i_old) + mo_space_info_->irrep_label(h);
            std::string i_new_str = std::to_string(i_new) + mo_space_info_->irrep_label(h);
            std::string space = i_new < ndoccpi_[h] ? "RESTRICTED_DOCC" : "RESTRICTED_UOCC";
            outfile->Printf("\n    %9s -> %-9s %s", i_old_str.c_str(), i_new_str.c_str(),
                            space.c_str());
        }
    }

    return out;
}

void MCSCF_ORB_GRAD::compute_reference_energy() {
    // compute energy given that all useful MO integrals are available
    energy_ = e_closed_ + ints_->nuclear_repulsion_energy();
    energy_ += Fc_["uv"] * D1_["uv"];
    energy_ += 0.5 * V_["uvxy"] * D2_["uvxy"];

    if (debug_print_) {
        outfile->Printf("\n  Reference energy     %20.15f", energy_);
    }
}

void MCSCF_ORB_GRAD::compute_orbital_grad() {
    // build orbital response of energy with a factor of 0.5
    A_["ri"] = 2.0 * F_["ri"];
    A_["ru"] = Fc_["rt"] * D1_["tu"];
    A_["ru"] += V_["rtvw"] * D2_["tuvw"];

    // build orbital gradients
    g_["pq"] = 2.0 * A_["pq"];
    g_["pq"] -= 2.0 * A_["qp"];

    // reshape and format to std::shared_ptr<psi::Vector>
    reshape_rot_ambit(g_, grad_);

    if (debug_print_) {
        grad_->print();
    }
}

void MCSCF_ORB_GRAD::hess_diag(std::shared_ptr<psi::Vector>,
                                const std::shared_ptr<psi::Vector>& h0) {
    compute_orbital_hess_diag();
    h0->copy(*hess_diag_);
}

void MCSCF_ORB_GRAD::compute_orbital_hess_diag() {
    // modified diagonal Hessian from Theor. Chem. Acc. 97, 88-95 (1997)

    // virtual-core block
    h_diag_.block("vc").iterate([&](const std::vector<size_t>& i, double& value) {
        auto i0 = label_to_mos_["v"][i[0]];
        auto i1 = label_to_mos_["c"][i[1]];
        value = 4.0 * (Fd_[i0] - Fd_[i1]);
    });

    // virtual-active block
    auto& d1_data = D1_.block("aa").data();
    auto& a_data = A_.block("aa").data();

    h_diag_.block("va").iterate([&](const std::vector<size_t>& i, double& value) {
        auto i0 = label_to_mos_["v"][i[0]];
        auto i1 = i[1] * nactv_ + i[1];
        value = 2.0 * (Fd_[i0] * d1_data[i1] - a_data[i1]);
    });

    // active-core block
    h_diag_.block("ac").iterate([&](const std::vector<size_t>& i, double& value) {
        auto i0 = label_to_mos_["a"][i[0]];
        auto i1 = label_to_mos_["c"][i[1]];
        auto i1p = i[0] * nactv_ + i[0];
        value = 4.0 * (Fd_[i0] - Fd_[i1]);
        value += 2.0 * (Fd_[i1] * d1_data[i1p] - a_data[i1p]);
    });

    // active-active block [see SI of J. Chem. Phys. 152, 074102 (2020)]
    if (internal_rot_ or gas_ref_) {
        size_t nactv2 = nactv_ * nactv_;
        size_t nactv3 = nactv2 * nactv_;

        auto& fc_data = Fc_.block("aa").data();
        auto& v_data = V_.block("aaaa").data();
        auto& d2_data = D2_.block("aaaa").data();

        // G^{uu}_{vv}

        // (uu|xy)
        jk_internal_.block("aaa").iterate([&](const std::vector<size_t>& i, double& value) {
            auto idx = i[0] * nactv3 + i[0] * nactv2 + i[1] * nactv_ + i[2];
            value = v_data[idx];
        });

        // D_{vv,xy}
        d2_internal_.block("aaa").iterate([&](const std::vector<size_t>& i, double& value) {
            auto idx = i[0] * nactv3 + i[0] * nactv2 + i[1] * nactv_ + i[2];
            value = d2_data[idx];
        });

        Guu_["uv"] = jk_internal_["uxy"] * d2_internal_["vxy"];

        // (ux|uy)
        jk_internal_.block("aaa").iterate([&](const std::vector<size_t>& i, double& value) {
            auto idx = i[0] * nactv3 + i[1] * nactv2 + i[0] * nactv_ + i[2];
            value = v_data[idx];
        });

        // D_{vx,vy}
        d2_internal_.block("aaa").iterate([&](const std::vector<size_t>& i, double& value) {
            auto idx = i[0] * nactv3 + i[1] * nactv2 + i[0] * nactv_ + i[2];
            value = d2_data[idx];
        });

        Guu_["uv"] += 2.0 * jk_internal_["uxy"] * d2_internal_["vxy"];

        Guu_.block("aa").iterate([&](const std::vector<size_t>& i, double& value) {
            auto i0 = i[0] * nactv_ + i[0];
            auto i1 = i[1] * nactv_ + i[1];
            value += fc_data[i0] * d1_data[i1];
        });

        // G^{uv}_{vu}
        Guv_["uv"] = Fc_["uv"] * D1_["vu"];
        Guv_["uv"] += V_["uvxy"] * D2_["vuxy"];
        Guv_["uv"] += 2.0 * V_["uxvy"] * D2_["vxuy"];

        // build diagonal Hessian
        h_diag_["uv"] = 2.0 * Guu_["uv"];
        h_diag_["uv"] += 2.0 * Guu_["vu"];
        h_diag_["uv"] -= 2.0 * Guv_["uv"];
        h_diag_["uv"] -= 2.0 * Guv_["vu"];

        h_diag_.block("aa").iterate([&](const std::vector<size_t>& i, double& value) {
            auto i0 = i[0] * nactv_ + i[0];
            auto i1 = i[1] * nactv_ + i[1];
            value -= 2.0 * (a_data[i0] + a_data[i1]);
        });
    }

    // reshape and format to std::shared_ptr<psi::Vector>
    reshape_rot_ambit(h_diag_, hess_diag_);

    if (debug_print_) {
        hess_diag_->print();
    }
}

void MCSCF_ORB_GRAD::reshape_rot_ambit(ambit::BlockedTensor bt,
                                        const std::shared_ptr<psi::Vector>& sv) {
    size_t vec_size = sv->dimpi().sum();
    if (vec_size != nrot_) {
        throw std::runtime_error(
            "Inconsistent size between std::shared_ptr<psi::Vector> and number of rotations");
    }

    for (size_t n = 0; n < nrot_; ++n) {
        std::string block;
        int i, j;
        std::tie(block, i, j) = rot_mos_block_[n];
        auto tensor = bt.block(block);
        auto dim1 = tensor.dim(1);

        sv->set(n, tensor.data()[i * dim1 + j]);
    }
}

void MCSCF_ORB_GRAD::set_rdms(std::shared_ptr<RDMs> rdms) {
    // form spin-summed densities
    D1_.block("aa").copy(rdms->SF_G1());
    format_1rdm();

    // change to chemists' notation
    D2_.block("aaaa")("pqrs") = rdms->SF_G2()("prqs");
    D2_.block("aaaa")("pqrs") += rdms->SF_G2()("qrps");
    D2_.scale(0.5);
}

void MCSCF_ORB_GRAD::format_1rdm() {
    const auto& d1_data = D1_.block("aa").data();

    for (int h = 0, offset = 0; h < nirrep_; ++h) {
        for (int u = 0; u < nactvpi_[h]; ++u) {
            size_t nu = u + offset;
            for (int v = 0; v < nactvpi_[h]; ++v) {
                rdm1_->set(h, u, v, d1_data[nu * nactv_ + v + offset]);
            }
        }
        offset += nactvpi_[h];
    }

    if (debug_print_) {
        rdm1_->print();
    }
}

std::shared_ptr<ActiveSpaceIntegrals> MCSCF_ORB_GRAD::active_space_ints() {
    auto actv_sym = mo_space_info_->symmetry("ACTIVE");
    auto active_space_ints =
        std::make_shared<ActiveSpaceIntegrals>(ints_, actv_mos_, actv_sym, core_mos_);

    // build antisymmetrized TEI in physists' notation
    auto actv_ab = ambit::Tensor::build(CoreTensor, "tei_actv_aa", std::vector<size_t>(4, nactv_));
    actv_ab("pqrs") = V_.block("aaaa")("prqs");

    auto actv_aa = ambit::Tensor::build(CoreTensor, "tei_actv_aa", std::vector<size_t>(4, nactv_));
    actv_aa.copy(actv_ab);
    actv_aa("uvxy") -= actv_ab("uvyx");

    active_space_ints->set_active_integrals(actv_aa, actv_ab, actv_aa);

    auto& oei = Fc_.block("aa").data();
    active_space_ints->set_restricted_one_body_operator(oei, oei);

    active_space_ints->set_scalar_energy(e_closed_ - ints_->frozen_core_energy());

    return active_space_ints;
}

void MCSCF_ORB_GRAD::canonicalize_final(const std::shared_ptr<psi::Matrix>& U) {
    U_ = psi::linalg::doublet(U_, U, false, false);
    U_->set_name("Orthogonal Transformation");

    test_orbital_rotations(U_, "WARNING: Final Active Orbitals Maybe Different from the Original");

    C_->gemm(false, false, 1.0, C0_, U_, 0.0);

    if (ints_->integral_type() == Custom) {
        ints_->update_orbitals(C_, C_);
    } else {
        ints_->update_orbitals(C_, C_, false);
    }
    build_mo_integrals();
}
} // namespace forte
