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

#include <tuple>

#include "ambit/tensor.h"

#include "psi4/psi4-dec.h"
#include "psi4/libpsi4util/PsiOutStream.h"
#include "psi4/libpsi4util/process.h"
#include "psi4/libpsio/psio.hpp"

#include "base_classes/active_space_solver.h"

#include "embedding_density.h"

namespace forte {

EMBEDDING_DENSITY::EMBEDDING_DENSITY(
    const std::map<StateInfo, std::vector<double>>& state_weights_map,
    std::shared_ptr<SCFInfo> scf_info, std::shared_ptr<MOSpaceInfo> mo_space_info,
    std::shared_ptr<ForteIntegrals> ints, std::shared_ptr<ForteOptions> options)
    : state_weights_map_(state_weights_map), scf_info_(scf_info), mo_space_info_(mo_space_info),
      ints_(ints), options_(options) {

    start_up();
}

size_t index_exp(size_t na, size_t p, size_t q, size_t r, size_t s) {
    return na * na * na * p + na * na * q + na * r + s;
}

void EMBEDDING_DENSITY::start_up() {
    auto doccpi = scf_info_->doccpi();
    auto rdoccpi = mo_space_info_->dimension("RESTRICTED_DOCC");
    auto actvpi = mo_space_info_->dimension("ACTIVE");
    auto actvopi = doccpi - rdoccpi - mo_space_info_->dimension("FROZEN_DOCC");

    mos_actv_o_.clear();
    mos_actv_o_.reserve(actvopi.sum());
    for (int h = 0, shift = 0, nirrep = mo_space_info_->nirrep(); h < nirrep; ++h) {
        for (int i = 0; i < actvopi[h]; ++i) {
            mos_actv_o_.push_back(shift + i);
        }
        shift += actvpi[h];
    }
}

RDMs EMBEDDING_DENSITY::rhf_rdms() {
    size_t na = mo_space_info_->size("ACTIVE");
    psi::outfile->Printf("\n RHF(A) density will be written into the fragment (A) densiy blocks");

    // 1-RDM
    auto D1a = ambit::Tensor::build(ambit::CoreTensor, "D1a", std::vector<size_t>(2, na));
    auto D1b = ambit::Tensor::build(ambit::CoreTensor, "D1b", std::vector<size_t>(2, na));

    for (auto i : mos_actv_o_) {
        // psi::outfile->Printf("\n rdm(%d, %d)", i, i);
        D1a.data()[i * na + i] = 1.0;
        D1b.data()[i * na + i] = 1.0;
    }

    psi::outfile->Printf("\n D1 formed.");

    // 2-RDM
    auto D2aa = ambit::Tensor::build(ambit::CoreTensor, "D2aa", std::vector<size_t>(4, na));
    auto D2ab = ambit::Tensor::build(ambit::CoreTensor, "D2ab", std::vector<size_t>(4, na));
    auto D2bb = ambit::Tensor::build(ambit::CoreTensor, "D2bb", std::vector<size_t>(4, na));

    D2aa("pqrs") += D1a("pr") * D1a("qs");
    D2aa("pqrs") -= D1a("ps") * D1a("qr");

    D2bb("pqrs") += D1b("pr") * D1b("qs");
    D2bb("pqrs") -= D1b("ps") * D1b("qr");

    D2ab("pqrs") = D1a("pr") * D1b("qs");

    psi::outfile->Printf("\n D2 formed.");

    // 3-RDM
    if (options_->get_str("THREEPDC") != "ZERO") {
        auto D3aaa = ambit::Tensor::build(ambit::CoreTensor, "D3aaa", std::vector<size_t>(6, na));
        auto D3aab = ambit::Tensor::build(ambit::CoreTensor, "D3aab", std::vector<size_t>(6, na));
        auto D3abb = ambit::Tensor::build(ambit::CoreTensor, "D3abb", std::vector<size_t>(6, na));
        auto D3bbb = ambit::Tensor::build(ambit::CoreTensor, "D3bbb", std::vector<size_t>(6, na));

        D3aaa("pqrstu") += D1a("ps") * D1a("qt") * D1a("ru");
        D3aaa("pqrstu") -= D1a("ps") * D1a("rt") * D1a("qu");
        D3aaa("pqrstu") -= D1a("qs") * D1a("pt") * D1a("ru");
        D3aaa("pqrstu") += D1a("qs") * D1a("rt") * D1a("pu");
        D3aaa("pqrstu") -= D1a("rs") * D1a("qt") * D1a("pu");
        D3aaa("pqrstu") += D1a("rs") * D1a("pt") * D1a("qu");

        D3bbb("pqrstu") += D1b("ps") * D1b("qt") * D1b("ru");
        D3bbb("pqrstu") -= D1b("ps") * D1b("rt") * D1b("qu");
        D3bbb("pqrstu") -= D1b("qs") * D1b("pt") * D1b("ru");
        D3bbb("pqrstu") += D1b("qs") * D1b("rt") * D1b("pu");
        D3bbb("pqrstu") -= D1b("rs") * D1b("qt") * D1b("pu");
        D3bbb("pqrstu") += D1b("rs") * D1b("pt") * D1b("qu");

        D3aab("pqrstu") += D1a("ps") * D1a("qt") * D1b("ru");
        D3aab("pqrstu") -= D1a("qs") * D1a("pt") * D1b("ru");

        D3abb("pqrstu") += D1a("ps") * D1b("qt") * D1b("ru");
        D3abb("pqrstu") -= D1a("ps") * D1b("rt") * D1b("qu");

        psi::outfile->Printf("\n D3 formed.");

        return RDMs(D1a, D1b, D2aa, D2ab, D2bb, D3aaa, D3aab, D3abb, D3bbb);
    } else {
        psi::outfile->Printf("\n D3 skipped.");

        return RDMs(D1a, D1b, D2aa, D2ab, D2bb);
    }
}

RDMs EMBEDDING_DENSITY::cas_rdms(std::shared_ptr<MOSpaceInfo> mo_space_info_active) {
    // Return RDM from a CASCI/CASSCF computation
    RDMs ref_rdms;
    int print_level = 0;
    if (options_->get_str("FRAGMENT_DENSITY") == "CASCI") {
        std::string ci_type = options_->get_str("ACTIVE_SPACE_SOLVER");
        auto state_map = to_state_nroots_map(state_weights_map_);

        size_t nactv = mo_space_info_->size("ACTIVE");
        auto actv_mos = mo_space_info_active->corr_absolute_mo("ACTIVE");
        std::vector<int> actv_sym = mo_space_info_active->symmetry("ACTIVE");
        auto rdocc_mos = mo_space_info_active->corr_absolute_mo("RESTRICTED_DOCC");

        auto fci_ints =
            std::make_shared<ActiveSpaceIntegrals>(ints_, actv_mos, actv_sym, rdocc_mos);
        fci_ints->set_active_integrals_and_restricted_docc();

        auto active_space_solver = make_active_space_solver(
            ci_type, state_map, scf_info_, mo_space_info_active, fci_ints, options_);
        active_space_solver->set_print(print_level);
        active_space_solver->compute_energy();

        ref_rdms = active_space_solver->compute_average_rdms(state_weights_map_, 2);
        psi::outfile->Printf(
            "\n CASCI(A) density will be written into the fragment (A) densiy blocks");
    } else if (options_->get_str("FRAGMENT_DENSITY") == "CASSCF") {
        CASSCF cas_1(state_weights_map_, scf_info_, options_, mo_space_info_active, ints_);
        cas_1.compute_energy();
        psi::outfile->Printf(
            "\n CASSCF(A) density will be written into the fragment (A) densiy blocks");
        ref_rdms = cas_1.ref_rdms();
    }

    // Build RDMs with mo_space_info size, fill the blocks
    size_t na = mo_space_info_->size("ACTIVE");
    size_t na_in = mo_space_info_active->size("ACTIVE");

    // Outer-layer mospace
    auto doccpi = scf_info_->doccpi();
    auto rdoccpi = mo_space_info_->dimension("RESTRICTED_DOCC");
    auto actvpi = mo_space_info_->dimension("ACTIVE");
    auto actvopi = doccpi - rdoccpi - mo_space_info_->dimension("FROZEN_DOCC");

    // Inner-layer mospace
    auto rdoccpi_in = mo_space_info_active->dimension("RESTRICTED_DOCC");
    auto actvpi_in = mo_space_info_active->dimension("ACTIVE");
    // auto actvopi_in = doccpi - rdoccpi_in - mo_space_info_active->dimension("FROZEN_DOCC");

    std::vector<size_t> mos_oa;
    for (int i = 0; i < rdoccpi_in[0] + actvpi_in[0]; ++i) {
        mos_oa.push_back(i);
    }

    std::vector<size_t> mos_actv_in;
    for (int i = 0; i < actvpi_in[0]; ++i) {
        mos_actv_in.push_back(i);
    }

    // 1-RDM
    auto D1a = ambit::Tensor::build(ambit::CoreTensor, "D1a", std::vector<size_t>(2, na));
    auto D1b = ambit::Tensor::build(ambit::CoreTensor, "D1b", std::vector<size_t>(2, na));

    auto& D1a_data = D1a.data();
    auto& D1b_data = D1b.data();

    auto g1a_data = ref_rdms.g1a().data();
    auto g1b_data = ref_rdms.g1b().data();

    for (auto i : mos_oa) {
        for (auto j : mos_oa) {
            if (i >= rdoccpi_in[0] && j >= rdoccpi_in[0]) {
                size_t ip = i - rdoccpi_in[0];
                size_t jp = j - rdoccpi_in[0];
                D1a_data[i * na + j] = g1a_data[ip * na_in + jp];
                D1b_data[i * na + j] = g1b_data[ip * na_in + jp];
            } else {
                D1a_data[i * na + i] = 1.0;
                D1b_data[i * na + i] = 1.0;
            }
        }
    }

    // 2-RDM
    auto D2aa = ambit::Tensor::build(ambit::CoreTensor, "D2aa", std::vector<size_t>(4, na));
    auto D2ab = ambit::Tensor::build(ambit::CoreTensor, "D2ab", std::vector<size_t>(4, na));
    auto D2bb = ambit::Tensor::build(ambit::CoreTensor, "D2bb", std::vector<size_t>(4, na));

    D2aa("pqrs") += D1a("pr") * D1a("qs");
    D2aa("pqrs") -= D1a("ps") * D1a("qr");

    D2bb("pqrs") += D1b("pr") * D1b("qs");
    D2bb("pqrs") -= D1b("ps") * D1b("qr");

    D2ab("pqrs") = D1a("pr") * D1b("qs");

    size_t v = rdoccpi_in[0];

    auto& D2aa_data = D2aa.data();
    auto& D2ab_data = D2ab.data();
    auto& D2bb_data = D2bb.data();

    auto g2aa_data = ref_rdms.g2aa().data();
    auto g2bb_data = ref_rdms.g2bb().data();
    auto g2ab_data = ref_rdms.g2ab().data();

    for (auto p : mos_actv_in) {
        for (auto q : mos_actv_in) {
            for (auto r : mos_actv_in) {
                for (auto s : mos_actv_in) {
                    D2aa_data[index_exp(na, p + v, q + v, r + v, s + v)] =
                        g2aa_data[index_exp(na_in, p, q, r, s)];
                    D2bb_data[index_exp(na, p + v, q + v, r + v, s + v)] =
                        g2bb_data[index_exp(na_in, p, q, r, s)];
                    D2ab_data[index_exp(na, p + v, q + v, r + v, s + v)] =
                        g2ab_data[index_exp(na_in, p, q, r, s)];
                }
            }
        }
    }

    // Do not allow 3RDM when using CASCI/CASSCF density builder
    if (options_->get_str("THREEPDC") != "ZERO") {
        throw psi::PSIEXCEPTION("CASCI/CASSCF embedding density builder do not support 3-RDM!");
    } else {
        return RDMs(D1a, D1b, D2aa, D2ab, D2bb);
    }
}

} // namespace forte
