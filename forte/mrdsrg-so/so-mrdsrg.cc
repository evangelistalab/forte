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

#include <math.h>
#include <numeric>

#include "psi4/libpsi4util/process.h"
#include "psi4/libmints/molecule.h"
#include "psi4/libpsio/psio.h"
#include "psi4/libpsio/psio.hpp"
#include "psi4/libqt/qt.h"

#include "helpers/blockedtensorfactory.h"
#include "helpers/printing.h"
#include "helpers/helpers.h"
#include "so-mrdsrg.h"

#define ISA(x) (x < nactv)
#define ISB(x) (x >= nactv)
#define OFF(x) (x < nactv ? x : x - nactv)

using namespace ambit;

using namespace psi;

namespace forte {

SOMRDSRG::SOMRDSRG(RDMs rdms, std::shared_ptr<SCFInfo> scf_info,
                   std::shared_ptr<ForteOptions> options, std::shared_ptr<ForteIntegrals> ints,
                   std::shared_ptr<MOSpaceInfo> mo_space_info)
    : DynamicCorrelationSolver(rdms, scf_info, options, ints, mo_space_info),
      tensor_type_(CoreTensor), BTF(new BlockedTensorFactory()) {
    BlockedTensor::reset_mo_spaces();
    BlockedTensor::set_expert_mode(true);

    print_ = 2;

    print_method_banner({"Multireference Driven Similarity Renormalization Group",
                         "written by Francesco A. Evangelista"});

    startup();
    print_summary();
}

SOMRDSRG::~SOMRDSRG() {
    cleanup();
    BlockedTensor::set_expert_mode(false);
}

std::shared_ptr<ActiveSpaceIntegrals> SOMRDSRG::compute_Heff_actv() {
    throw psi::PSIEXCEPTION(
        "Computing active-space Hamiltonian is not yet implemented for spin-adapted code.");

    return std::make_shared<ActiveSpaceIntegrals>(
        ints_, mo_space_info_->corr_absolute_mo("ACTIVE"), mo_space_info_->symmetry("ACTIVE"),
        mo_space_info_->corr_absolute_mo("RESTRICTED_DOCC"));
}

void SOMRDSRG::startup() {
    Eref = compute_Eref_from_rdms(rdms_, ints_, mo_space_info_);

    frozen_core_energy = ints_->frozen_core_energy();

    ncmopi_ = mo_space_info_->dimension("CORRELATED");

    s_ = foptions_->get_double("DSRG_S");
    if (s_ < 0) {
        outfile->Printf("\n  S parameter for DSRG must >= 0!");
        exit(1);
    }
    taylor_threshold_ = foptions_->get_int("TAYLOR_THRESHOLD");
    if (taylor_threshold_ <= 0) {
        outfile->Printf("\n  Threshold for Taylor expansion must be an integer "
                        "greater than 0!");
        exit(1);
    }
    taylor_order_ = int(0.5 * (15.0 / taylor_threshold_ + 1)) + 1;

    std::vector<size_t> rdocc = mo_space_info_->corr_absolute_mo("RESTRICTED_DOCC");
    std::vector<size_t> actv = mo_space_info_->corr_absolute_mo("ACTIVE");
    std::vector<size_t> ruocc = mo_space_info_->corr_absolute_mo("RESTRICTED_UOCC");

    for (auto& space : {rdocc, actv, ruocc}) {
        outfile->Printf("\n");
        for (size_t mo : space) {
            outfile->Printf(" %3d", mo);
        }
    }

    size_t nactv = mo_space_info_->size("ACTIVE");

    std::vector<std::pair<size_t, SpinType>> rdocc_so;
    for (size_t p : rdocc)
        rdocc_so.push_back(std::make_pair(p, AlphaSpin));
    for (size_t p : rdocc)
        rdocc_so.push_back(std::make_pair(p, BetaSpin));

    std::vector<std::pair<size_t, SpinType>> actv_so;
    for (size_t p : actv)
        actv_so.push_back(std::make_pair(p, AlphaSpin));
    for (size_t p : actv)
        actv_so.push_back(std::make_pair(p, BetaSpin));

    std::vector<std::pair<size_t, SpinType>> ruocc_so;
    for (size_t p : ruocc)
        ruocc_so.push_back(std::make_pair(p, AlphaSpin));
    for (size_t p : ruocc)
        ruocc_so.push_back(std::make_pair(p, BetaSpin));

    BTF->add_mo_space("c", "m,n,o,c0,c1,c2,c3,c4,c5,c6,c7,c8,c9", rdocc_so);
    BTF->add_mo_space("a", "u,v,w,x,y,z,a0,a1,a2,a3,a4,a5,a6,a7,a8,a9", actv_so);
    BTF->add_mo_space("v", "e,f,g,h,v0,v1,v2,v3,v4,v5,v6,v7,v8,v9", ruocc_so);

    BTF->add_composite_mo_space("h", "i,j,k,l,h0,h1,h2,h3,h4,h5,h6,h7", {"c", "a"});
    BTF->add_composite_mo_space("p", "a,b,c,d,p0,p1,p2,p3,p4,p5,p6,p7", {"a", "v"});
    BTF->add_composite_mo_space("g", "p,q,r,s,g0,g1,g2,g3,g4,g5,g6,g7", {"c", "a", "v"});

    H = BTF->build(tensor_type_, "H", {"gg"});
    V = BTF->build(tensor_type_, "V", {"gggg"});

    Gamma1 = BTF->build(tensor_type_, "Gamma1", {"hh"});
    Eta1 = BTF->build(tensor_type_, "Eta1", {"pp"});
    Lambda2 = BTF->build(tensor_type_, "Lambda2", {"aaaa"});
    if (foptions_->get_str("THREEPDC") != "ZERO") {
        Lambda3 = BTF->build(tensor_type_, "Lambda3", {"aaaaaa"});
    }
    F = BTF->build(tensor_type_, "Fock", {"gg"});
    Delta1 = BTF->build(tensor_type_, "Delta1", {"hp"});
    Delta2 = BTF->build(tensor_type_, "Delta2", {"hhpp"});
    RInvDelta1 = BTF->build(tensor_type_, "Renormalized Inverse Delta1", {"hp"});
    RInvDelta2 = BTF->build(tensor_type_, "Renormalized Inverse Delta2", {"hhpp"});
    T1 = BTF->build(tensor_type_, "T1 Amplitudes", {"hp"});
    T2 = BTF->build(tensor_type_, "T2 Amplitudes", {"hhpp"});
    DT1 = BTF->build(tensor_type_, "Delta T1 Amplitudes", {"hp"});
    DT2 = BTF->build(tensor_type_, "Delta T2 Amplitudes", {"hhpp"});
    R1 = BTF->build(tensor_type_, "Residual T1 Amplitudes", {"hp"});
    R2 = BTF->build(tensor_type_, "Residual T2 Amplitudes", {"hhpp"});
    O1 = BTF->build(tensor_type_, "O1", {"gg"});
    O2 = BTF->build(tensor_type_, "O2", {"gggg"});
    C1 = BTF->build(tensor_type_, "C1", {"gg"});
    C2 = BTF->build(tensor_type_, "C2", {"gggg"});
    Hbar1 = BTF->build(tensor_type_, "One-body Hbar", {"gg"});
    Hbar2 = BTF->build(tensor_type_, "Two-body Hbar", {"gggg"});

    H.iterate([&](const std::vector<size_t>& i, const std::vector<SpinType>& spin, double& value) {
        if ((spin[0] == AlphaSpin) and (spin[1] == AlphaSpin))
            value = ints_->oei_a(i[0], i[1]);
        if ((spin[0] == BetaSpin) and (spin[1] == BetaSpin))
            value = ints_->oei_b(i[0], i[1]);
    });

    // Fill in the two-electron operator (V)
    V.iterate([&](const std::vector<size_t>& i, const std::vector<SpinType>& spin, double& value) {
        if ((spin[0] == AlphaSpin) and (spin[1] == AlphaSpin) and (spin[2] == AlphaSpin) and
            (spin[3] == AlphaSpin))
            value = +ints_->aptei_aa(i[0], i[1], i[2], i[3]);

        if ((spin[0] == AlphaSpin) and (spin[1] == BetaSpin) and (spin[2] == AlphaSpin) and
            (spin[3] == BetaSpin))
            value = +ints_->aptei_ab(i[0], i[1], i[2], i[3]);

        if ((spin[0] == AlphaSpin) and (spin[1] == BetaSpin) and (spin[2] == BetaSpin) and
            (spin[3] == AlphaSpin))
            value = -ints_->aptei_ab(i[0], i[1], i[3], i[2]);

        if ((spin[0] == BetaSpin) and (spin[1] == AlphaSpin) and (spin[2] == AlphaSpin) and
            (spin[3] == BetaSpin))
            value = -ints_->aptei_ab(i[1], i[0], i[2], i[3]);

        if ((spin[0] == BetaSpin) and (spin[1] == AlphaSpin) and (spin[2] == BetaSpin) and
            (spin[3] == AlphaSpin))
            value = +ints_->aptei_ab(i[1], i[0], i[3], i[2]);

        if ((spin[0] == BetaSpin) and (spin[1] == BetaSpin) and (spin[2] == BetaSpin) and
            (spin[3] == BetaSpin))
            value = +ints_->aptei_bb(i[0], i[1], i[2], i[3]);
    });

    ambit::Tensor Gamma1_cc = Gamma1.block("cc");
    ambit::Tensor Gamma1_aa = Gamma1.block("aa");

    ambit::Tensor Eta1_aa = Eta1.block("aa");
    ambit::Tensor Eta1_vv = Eta1.block("vv");

    Gamma1_cc.iterate(
        [&](const std::vector<size_t>& i, double& value) { value = i[0] == i[1] ? 1.0 : 0.0; });
    Eta1_aa.iterate(
        [&](const std::vector<size_t>& i, double& value) { value = i[0] == i[1] ? 1.0 : 0.0; });
    Eta1_vv.iterate(
        [&](const std::vector<size_t>& i, double& value) { value = i[0] == i[1] ? 1.0 : 0.0; });

    Matrix gamma_aa("Gamma_aa", nactv, nactv);
    Matrix gamma_AA("Gamma_AA", nactv, nactv);

    rdms_.g1a().iterate(
        [&](const std::vector<size_t>& i, double& value) { gamma_aa.set(i[0], i[1], value); });
    rdms_.g1b().iterate(
        [&](const std::vector<size_t>& i, double& value) { gamma_AA.set(i[0], i[1], value); });

    // Fill up the active part of Gamma
    Gamma1_aa.iterate([&](const std::vector<size_t>& i, double& value) {
        if (i[0] < nactv and i[1] < nactv) {
            value = gamma_aa.get(i[0], i[1]);
        }
        if (i[0] >= nactv and i[1] >= nactv) {
            value = gamma_AA.get(i[0] - nactv, i[1] - nactv);
        }
    });

    Eta1_aa("pq") -= Gamma1_aa("pq");

    // Fill out Lambda2 and Lambda3
    ambit::Tensor Lambda2_aa = Lambda2.block("aaaa");

    Matrix lambda2_aa("Lambda2_aa", nactv * nactv, nactv * nactv);
    Matrix lambda2_aA("Lambda2_aA", nactv * nactv, nactv * nactv);
    Matrix lambda2_AA("Lambda2_AA", nactv * nactv, nactv * nactv);

    rdms_.L2aa().iterate([&](const std::vector<size_t>& i, double& value) {
        size_t I = nactv * i[0] + i[1];
        size_t J = nactv * i[2] + i[3];
        lambda2_aa.set(I, J, value);
    });
    rdms_.L2ab().iterate([&](const std::vector<size_t>& i, double& value) {
        size_t I = nactv * i[0] + i[1];
        size_t J = nactv * i[2] + i[3];
        lambda2_aA.set(I, J, value);
    });

    rdms_.L2bb().iterate([&](const std::vector<size_t>& i, double& value) {
        size_t I = nactv * i[0] + i[1];
        size_t J = nactv * i[2] + i[3];
        lambda2_AA.set(I, J, value);
    });

    // Fill up the active part of Lamba2
    Lambda2_aa.iterate([&](const std::vector<size_t>& i, double& value) {
        // aa|aa
        if (ISA(i[0]) and ISA(i[1]) and ISA(i[2]) and ISA(i[3])) {
            size_t I = nactv * OFF(i[0]) + OFF(i[1]);
            size_t J = nactv * OFF(i[2]) + OFF(i[3]);
            value = lambda2_aa.get(I, J);
        }
        // ab|ab
        if (ISA(i[0]) and ISB(i[1]) and ISA(i[2]) and ISB(i[3])) {
            size_t I = nactv * OFF(i[0]) + OFF(i[1]);
            size_t J = nactv * OFF(i[2]) + OFF(i[3]);
            value = lambda2_aA.get(I, J);
        }
        // ab|ba
        if (ISA(i[0]) and ISB(i[1]) and ISB(i[2]) and ISA(i[3])) {
            size_t I = nactv * OFF(i[0]) + OFF(i[1]);
            size_t J = nactv * OFF(i[3]) + OFF(i[2]);
            value = -lambda2_aA.get(I, J);
        }
        // ba|ab
        if (ISB(i[0]) and ISA(i[1]) and ISA(i[2]) and ISB(i[3])) {
            size_t I = nactv * OFF(i[1]) + OFF(i[0]);
            size_t J = nactv * OFF(i[2]) + OFF(i[3]);
            value = -lambda2_aA.get(I, J);
        }
        // ba|ba
        if (ISB(i[0]) and ISA(i[1]) and ISB(i[2]) and ISA(i[3])) {
            size_t I = nactv * OFF(i[1]) + OFF(i[0]);
            size_t J = nactv * OFF(i[3]) + OFF(i[2]);
            value = lambda2_aA.get(I, J);
        }
        // bb|bb
        if (ISB(i[0]) and ISB(i[1]) and ISB(i[2]) and ISB(i[3])) {
            size_t I = nactv * OFF(i[0]) + OFF(i[1]);
            size_t J = nactv * OFF(i[2]) + OFF(i[3]);
            value = lambda2_AA.get(I, J);
        }
    });

    if (foptions_->get_str("THREEPDC") != "ZERO") {
        ambit::Tensor Lambda3_aaa = Lambda3.block("aaaaaa");

        Matrix lambda3_aaa("Lambda3_aaa", nactv * nactv * nactv, nactv * nactv * nactv);
        Matrix lambda3_aaA("Lambda3_aaA", nactv * nactv * nactv, nactv * nactv * nactv);
        Matrix lambda3_aAA("Lambda3_aAA", nactv * nactv * nactv, nactv * nactv * nactv);
        Matrix lambda3_AAA("Lambda3_AAA", nactv * nactv * nactv, nactv * nactv * nactv);

        rdms_.L3aaa().iterate([&](const std::vector<size_t>& i, double& value) {
            size_t I = nactv * nactv * i[0] + nactv * i[1] + i[2];
            size_t J = nactv * nactv * i[3] + nactv * i[4] + i[5];
            lambda3_aaa.set(I, J, value);
        });
        rdms_.L3aab().iterate([&](const std::vector<size_t>& i, double& value) {
            size_t I = nactv * nactv * i[0] + nactv * i[1] + i[2];
            size_t J = nactv * nactv * i[3] + nactv * i[4] + i[5];
            lambda3_aaA.set(I, J, value);
        });
        rdms_.L3abb().iterate([&](const std::vector<size_t>& i, double& value) {
            size_t I = nactv * nactv * i[0] + nactv * i[1] + i[2];
            size_t J = nactv * nactv * i[3] + nactv * i[4] + i[5];
            lambda3_aAA.set(I, J, value);
        });
        rdms_.L3bbb().iterate([&](const std::vector<size_t>& i, double& value) {
            size_t I = nactv * nactv * i[0] + nactv * i[1] + i[2];
            size_t J = nactv * nactv * i[3] + nactv * i[4] + i[5];
            lambda3_AAA.set(I, J, value);
        });

        // Fill up the active part of Lamba3
        Lambda3_aaa.iterate([&](const std::vector<size_t>& i, double& value) {
            // aaa|aaa
            if (ISA(i[0]) and ISA(i[1]) and ISA(i[2]) and ISA(i[3]) and ISA(i[4]) and ISA(i[5])) {
                size_t I = nactv * nactv * OFF(i[0]) + nactv * OFF(i[1]) + OFF(i[2]);
                size_t J = nactv * nactv * OFF(i[3]) + nactv * OFF(i[4]) + OFF(i[5]);
                value = lambda3_aaa.get(I, J);
            }
            // aab|aab
            if (ISA(i[0]) and ISA(i[1]) and ISB(i[2]) and ISA(i[3]) and ISA(i[4]) and ISB(i[5])) {
                size_t I = nactv * nactv * OFF(i[0]) + nactv * OFF(i[1]) + OFF(i[2]);
                size_t J = nactv * nactv * OFF(i[3]) + nactv * OFF(i[4]) + OFF(i[5]);
                value = lambda3_aaA.get(I, J);
            }
            // aab|aba
            if (ISA(i[0]) and ISA(i[1]) and ISB(i[2]) and ISA(i[3]) and ISB(i[4]) and ISA(i[5])) {
                size_t I = nactv * nactv * OFF(i[0]) + nactv * OFF(i[1]) + OFF(i[2]);
                size_t J = nactv * nactv * OFF(i[3]) + nactv * OFF(i[5]) + OFF(i[4]);
                value = -lambda3_aaA.get(I, J);
            }
            // aab|baa
            if (ISA(i[0]) and ISA(i[1]) and ISB(i[2]) and ISB(i[3]) and ISA(i[4]) and ISA(i[5])) {
                size_t I = nactv * nactv * OFF(i[0]) + nactv * OFF(i[1]) + OFF(i[2]);
                size_t J = nactv * nactv * OFF(i[4]) + nactv * OFF(i[5]) + OFF(i[3]);
                value = lambda3_aaA.get(I, J);
            }

            // aba|aab
            if (ISA(i[0]) and ISB(i[1]) and ISA(i[2]) and ISA(i[3]) and ISA(i[4]) and ISB(i[5])) {
                size_t I = nactv * nactv * OFF(i[0]) + nactv * OFF(i[2]) + OFF(i[1]);
                size_t J = nactv * nactv * OFF(i[3]) + nactv * OFF(i[4]) + OFF(i[5]);
                value = -lambda3_aaA.get(I, J);
            }
            // aba|aba
            if (ISA(i[0]) and ISB(i[1]) and ISA(i[2]) and ISA(i[3]) and ISB(i[4]) and ISA(i[5])) {
                size_t I = nactv * nactv * OFF(i[0]) + nactv * OFF(i[2]) + OFF(i[1]);
                size_t J = nactv * nactv * OFF(i[3]) + nactv * OFF(i[5]) + OFF(i[4]);
                value = lambda3_aaA.get(I, J);
            }
            // aba|baa
            if (ISA(i[0]) and ISB(i[1]) and ISA(i[2]) and ISB(i[3]) and ISA(i[4]) and ISA(i[5])) {
                size_t I = nactv * nactv * OFF(i[0]) + nactv * OFF(i[2]) + OFF(i[1]);
                size_t J = nactv * nactv * OFF(i[4]) + nactv * OFF(i[5]) + OFF(i[3]);
                value = -lambda3_aaA.get(I, J);
            }

            // baa|aab
            if (ISB(i[0]) and ISA(i[1]) and ISA(i[2]) and ISA(i[3]) and ISA(i[4]) and ISB(i[5])) {
                size_t I = nactv * nactv * OFF(i[1]) + nactv * OFF(i[2]) + OFF(i[0]);
                size_t J = nactv * nactv * OFF(i[3]) + nactv * OFF(i[4]) + OFF(i[5]);
                value = lambda3_aaA.get(I, J);
            }
            // baa|aba
            if (ISB(i[0]) and ISA(i[1]) and ISA(i[2]) and ISA(i[3]) and ISB(i[4]) and ISA(i[5])) {
                size_t I = nactv * nactv * OFF(i[1]) + nactv * OFF(i[2]) + OFF(i[0]);
                size_t J = nactv * nactv * OFF(i[3]) + nactv * OFF(i[5]) + OFF(i[4]);
                value = -lambda3_aaA.get(I, J);
            }
            // baa|baa
            if (ISB(i[0]) and ISA(i[1]) and ISA(i[2]) and ISB(i[3]) and ISA(i[4]) and ISA(i[5])) {
                size_t I = nactv * nactv * OFF(i[1]) + nactv * OFF(i[2]) + OFF(i[0]);
                size_t J = nactv * nactv * OFF(i[4]) + nactv * OFF(i[5]) + OFF(i[3]);
                value = lambda3_aaA.get(I, J);
            }

            // abb|abb
            if (ISA(i[0]) and ISB(i[1]) and ISB(i[2]) and ISA(i[3]) and ISB(i[4]) and ISB(i[5])) {
                size_t I = nactv * nactv * OFF(i[0]) + nactv * OFF(i[1]) + OFF(i[2]);
                size_t J = nactv * nactv * OFF(i[3]) + nactv * OFF(i[4]) + OFF(i[5]);
                value = lambda3_aAA.get(I, J);
            }
            // abb|bab
            if (ISA(i[0]) and ISB(i[1]) and ISB(i[2]) and ISB(i[3]) and ISA(i[4]) and ISB(i[5])) {
                size_t I = nactv * nactv * OFF(i[0]) + nactv * OFF(i[1]) + OFF(i[2]);
                size_t J = nactv * nactv * OFF(i[4]) + nactv * OFF(i[3]) + OFF(i[5]);
                value = -lambda3_aAA.get(I, J);
            }
            // abb|bba
            if (ISA(i[0]) and ISB(i[1]) and ISB(i[2]) and ISB(i[3]) and ISB(i[4]) and ISA(i[5])) {
                size_t I = nactv * nactv * OFF(i[0]) + nactv * OFF(i[1]) + OFF(i[2]);
                size_t J = nactv * nactv * OFF(i[5]) + nactv * OFF(i[3]) + OFF(i[4]);
                value = lambda3_aAA.get(I, J);
            }

            // bab|abb
            if (ISB(i[0]) and ISA(i[1]) and ISB(i[2]) and ISA(i[3]) and ISB(i[4]) and ISB(i[5])) {
                size_t I = nactv * nactv * OFF(i[1]) + nactv * OFF(i[0]) + OFF(i[2]);
                size_t J = nactv * nactv * OFF(i[3]) + nactv * OFF(i[4]) + OFF(i[5]);
                value = -lambda3_aAA.get(I, J);
            }
            // bab|bab
            if (ISB(i[0]) and ISA(i[1]) and ISB(i[2]) and ISB(i[3]) and ISA(i[4]) and ISB(i[5])) {
                size_t I = nactv * nactv * OFF(i[1]) + nactv * OFF(i[0]) + OFF(i[2]);
                size_t J = nactv * nactv * OFF(i[4]) + nactv * OFF(i[3]) + OFF(i[5]);
                value = lambda3_aAA.get(I, J);
            }
            // bab|bba
            if (ISB(i[0]) and ISA(i[1]) and ISB(i[2]) and ISB(i[3]) and ISB(i[4]) and ISA(i[5])) {
                size_t I = nactv * nactv * OFF(i[1]) + nactv * OFF(i[0]) + OFF(i[2]);
                size_t J = nactv * nactv * OFF(i[5]) + nactv * OFF(i[3]) + OFF(i[4]);
                value = -lambda3_aAA.get(I, J);
            }

            // bba|abb
            if (ISB(i[0]) and ISB(i[1]) and ISA(i[2]) and ISA(i[3]) and ISB(i[4]) and ISB(i[5])) {
                size_t I = nactv * nactv * OFF(i[2]) + nactv * OFF(i[0]) + OFF(i[1]);
                size_t J = nactv * nactv * OFF(i[3]) + nactv * OFF(i[4]) + OFF(i[5]);
                value = lambda3_aAA.get(I, J);
            }
            // bba|bab
            if (ISB(i[0]) and ISB(i[1]) and ISA(i[2]) and ISB(i[3]) and ISA(i[4]) and ISB(i[5])) {
                size_t I = nactv * nactv * OFF(i[2]) + nactv * OFF(i[0]) + OFF(i[1]);
                size_t J = nactv * nactv * OFF(i[4]) + nactv * OFF(i[3]) + OFF(i[5]);
                value = -lambda3_aAA.get(I, J);
            }
            // bba|bba
            if (ISB(i[0]) and ISB(i[1]) and ISA(i[2]) and ISB(i[3]) and ISB(i[4]) and ISA(i[5])) {
                size_t I = nactv * nactv * OFF(i[2]) + nactv * OFF(i[0]) + OFF(i[1]);
                size_t J = nactv * nactv * OFF(i[5]) + nactv * OFF(i[3]) + OFF(i[4]);
                value = lambda3_aAA.get(I, J);
            }

            // bbb|bbb
            if (ISB(i[0]) and ISB(i[1]) and ISB(i[2]) and ISB(i[3]) and ISB(i[4]) and ISB(i[5])) {
                size_t I = nactv * nactv * OFF(i[0]) + nactv * OFF(i[1]) + OFF(i[2]);
                size_t J = nactv * nactv * OFF(i[3]) + nactv * OFF(i[4]) + OFF(i[5]);
                value = lambda3_AAA.get(I, J);
            }
        });
    }

    // Form the Fock matrix
    F["pq"] = H["pq"];
    F["pq"] += V["pjqi"] * Gamma1["ij"];

    size_t ncmo_ = mo_space_info_->size("CORRELATED");
    std::vector<double> Fa(ncmo_);
    std::vector<double> Fb(ncmo_);

    F.iterate([&](const std::vector<size_t>& i, const std::vector<SpinType>& spin, double& value) {
        if (spin[0] == AlphaSpin and spin[1] == AlphaSpin and (i[0] == i[1])) {
            Fa[i[0]] = value;
        }
        if (spin[0] == BetaSpin and spin[1] == BetaSpin and (i[0] == i[1])) {
            Fb[i[0]] = value;
        }
    });

    Delta1.iterate(
        [&](const std::vector<size_t>& i, const std::vector<SpinType>& spin, double& value) {
            if (spin[0] == AlphaSpin and spin[1] == AlphaSpin) {
                value = Fa[i[0]] - Fa[i[1]];
            } else if (spin[0] == BetaSpin and spin[1] == BetaSpin) {
                value = Fb[i[0]] - Fb[i[1]];
            } else {
                value = 0.0;
            }
        });

    Delta2.iterate(
        [&](const std::vector<size_t>& i, const std::vector<SpinType>& spin, double& value) {
            if (spin[0] == AlphaSpin and spin[1] == AlphaSpin and spin[2] == AlphaSpin and
                spin[3] == AlphaSpin) {
                value = Fa[i[0]] + Fa[i[1]] - Fa[i[2]] - Fa[i[3]];
            } else if (spin[0] == AlphaSpin and spin[1] == BetaSpin and spin[2] == AlphaSpin and
                       spin[3] == BetaSpin) {
                value = Fa[i[0]] + Fb[i[1]] - Fa[i[2]] - Fb[i[3]];
            } else if (spin[0] == AlphaSpin and spin[1] == BetaSpin and spin[2] == BetaSpin and
                       spin[3] == AlphaSpin) {
                value = Fa[i[0]] + Fb[i[1]] - Fb[i[2]] - Fa[i[3]];
            } else if (spin[0] == BetaSpin and spin[1] == AlphaSpin and spin[2] == AlphaSpin and
                       spin[3] == BetaSpin) {
                value = Fb[i[0]] + Fa[i[1]] - Fa[i[2]] - Fb[i[3]];
            } else if (spin[0] == BetaSpin and spin[1] == AlphaSpin and spin[2] == BetaSpin and
                       spin[3] == AlphaSpin) {
                value = Fb[i[0]] + Fa[i[1]] - Fb[i[2]] - Fa[i[3]];
            } else if (spin[0] == BetaSpin and spin[1] == BetaSpin and spin[2] == BetaSpin and
                       spin[3] == BetaSpin) {
                value = Fb[i[0]] + Fb[i[1]] - Fb[i[2]] - Fb[i[3]];
            } else {
                value = 0.0;
            }
        });

    RInvDelta1.iterate(
        [&](const std::vector<size_t>& i, const std::vector<SpinType>& spin, double& value) {
            if (spin[0] == AlphaSpin and spin[1] == AlphaSpin) {
                value = renormalized_denominator(Fa[i[0]] - Fa[i[1]]);
            } else if (spin[0] == BetaSpin and spin[1] == BetaSpin) {
                value = renormalized_denominator(Fb[i[0]] - Fb[i[1]]);
            } else {
                value = 0.0;
            }
        });

    RInvDelta2.iterate(
        [&](const std::vector<size_t>& i, const std::vector<SpinType>& spin, double& value) {
            if (spin[0] == AlphaSpin and spin[1] == AlphaSpin and spin[2] == AlphaSpin and
                spin[3] == AlphaSpin) {
                value = Fa[i[0]] + Fa[i[1]] - Fa[i[2]] - Fa[i[3]];
            } else if (spin[0] == AlphaSpin and spin[1] == BetaSpin and spin[2] == AlphaSpin and
                       spin[3] == BetaSpin) {
                value = Fa[i[0]] + Fb[i[1]] - Fa[i[2]] - Fb[i[3]];
            } else if (spin[0] == AlphaSpin and spin[1] == BetaSpin and spin[2] == BetaSpin and
                       spin[3] == AlphaSpin) {
                value = Fa[i[0]] + Fb[i[1]] - Fb[i[2]] - Fa[i[3]];
            } else if (spin[0] == BetaSpin and spin[1] == AlphaSpin and spin[2] == AlphaSpin and
                       spin[3] == BetaSpin) {
                value = Fb[i[0]] + Fa[i[1]] - Fa[i[2]] - Fb[i[3]];
            } else if (spin[0] == BetaSpin and spin[1] == AlphaSpin and spin[2] == BetaSpin and
                       spin[3] == AlphaSpin) {
                value = Fb[i[0]] + Fa[i[1]] - Fb[i[2]] - Fa[i[3]];
            } else if (spin[0] == BetaSpin and spin[1] == BetaSpin and spin[2] == BetaSpin and
                       spin[3] == BetaSpin) {
                value = Fb[i[0]] + Fb[i[1]] - Fb[i[2]] - Fb[i[3]];
            } else {
                value = 0.0;
            }
            value = renormalized_denominator(value);
        });

    // Prepare Hbar
    Hbar1["pq"] = F["pq"];
    Hbar2["pqrs"] = V["pqrs"];

    //    // Print levels
    //    print_ = foptions_->get_int("PRINT");
    //    if(print_ > 1){
    //        Gamma1.print(stdout);
    //        Eta1.print(stdout);
    //        F.print(stdout);
    //    }
    //    if(print_ > 2){
    //        V.print(stdout);
    //        Lambda2.print(stdout);
    //    }
    //    if(print_ > 3){
    //        Lambda3.print(stdout);
    //    }
}

void SOMRDSRG::print_summary() {
    // Print a summary
    std::vector<std::pair<std::string, int>> calculation_info;

    std::vector<std::pair<std::string, double>> calculation_info_double{
        {"flow parameter", s_},
        {"taylor expansion threshold", pow(10.0, -double(taylor_threshold_))}};

    std::vector<std::pair<std::string, std::string>> calculation_info_string{
        {"int_type", foptions_->get_str("INT_TYPE")}, {"source operator", source_}};

    // Print some information
    outfile->Printf("\n\n  ==> Calculation Information <==\n");
    for (auto& str_dim : calculation_info) {
        outfile->Printf("\n    %-39s %10d", str_dim.first.c_str(), str_dim.second);
    }
    for (auto& str_dim : calculation_info_double) {
        outfile->Printf("\n    %-39s %10.3e", str_dim.first.c_str(), str_dim.second);
    }
    for (auto& str_dim : calculation_info_string) {
        outfile->Printf("\n    %-39s %10s", str_dim.first.c_str(), str_dim.second.c_str());
    }
}

double SOMRDSRG::compute_energy() {
    print_h2("Computing the SO-MR-DSRG(2) energy");

    E0_ = ints_->nuclear_repulsion_energy() + ints_->frozen_core_energy();
    E0_ += F["qp"] * Gamma1["pq"];
    E0_ += -0.5 * V["rspq"] * Gamma1["pr"] * Gamma1["qs"];
    E0_ += 0.25 * V["rspq"] * Lambda2["pqrs"];

    outfile->Printf("\n  * RDMs total energy            = %25.15f\n", E0_);

    // Start the SO-MR-DSRG cycle
    double old_energy = 0.0;
    bool converged = false;
    int cycle = 0;

    outfile->Printf("\n  "
                    "----------------------------------------------------------"
                    "----------------------------------------");
    outfile->Printf("\n         Cycle     Energy (a.u.)     Delta(E)   |Hbar1| "
                    "   |Hbar2|     |S1|    |S2|  max(S1) max(S2)");
    outfile->Printf("\n  "
                    "----------------------------------------------------------"
                    "----------------------------------------");

    mp2_guess();

    compute_hbar();

    while (!converged) {
        if (print_ > 1) {
            outfile->Printf("\n  Updating the S amplitudes...");
        }

        update_T1();
        update_T2();

        if (print_ > 1) {
            outfile->Printf("\n  --------------------------------------------");
            outfile->Printf("\n  nExc           |S|                  |R|");
            outfile->Printf("\n  --------------------------------------------");
            outfile->Printf("\n    1     %15e      %15e", T1.norm(), 0.0);
            outfile->Printf("\n    2     %15e      %15e", T2.norm(), 0.0);
            outfile->Printf("\n  --------------------------------------------");

            //            auto max_S2aa = S2.block("oovv")->max_abs_element();
            //            auto max_S2ab = S2.block("oOvV")->max_abs_element();
            //            auto max_S2bb = S2.block("OOVV")->max_abs_element();
            //            outfile->Printf("\n  Largest S2 (aa): %20.12f
            //            ",max_S2aa.first);
            //            for (size_t index: max_S2aa.second){
            //                outfile->Printf(" %zu",index);
            //            }
            //            outfile->Printf("\n  Largest S2 (ab): %20.12f
            //            ",max_S2ab.first);
            //            for (size_t index: max_S2ab.second){
            //                outfile->Printf(" %zu",index);
            //            }
            //            outfile->Printf("\n  Largest S2 (bb): %20.12f
            //            ",max_S2bb.first);
            //            for (size_t index: max_S2bb.second){
            //                outfile->Printf(" %zu",index);
            //            }
        }

        if (print_ > 1) {
            outfile->Printf(" done.");
        }
        //        if(diis_manager){
        //            if (do_dsrg){
        //                diis_manager->add_entry(10,
        //                                        &(DS1.block("ov").data()[0]),
        //                                        &(DS1.block("OV").data()[0]),
        //                                        &(DS2.block("oovv").data()[0]),
        //                                        &(DS2.block("oOvV").data()[0]),
        //                                        &(DS2.block("OOVV").data()[0]),
        //                                        &(S1.block("ov").data()[0]),
        //                                        &(S1.block("OV").data()[0]),
        //                                        &(S2.block("oovv").data()[0]),
        //                                        &(S2.block("oOvV").data()[0]),
        //                                        &(S2.block("OOVV").data()[0]));
        //            }else{
        //                diis_manager->add_entry(10,
        //                                        &(Hbar1.block("ov").data()[0]),
        //                                        &(Hbar1.block("OV").data()[0]),
        //                                        &(Hbar2.block("oovv").data()[0]),
        //                                        &(Hbar2.block("oOvV").data()[0]),
        //                                        &(Hbar2.block("OOVV").data()[0]),
        //                                        &(S1.block("ov").data()[0]),
        //                                        &(S1.block("OV").data()[0]),
        //                                        &(S2.block("oovv").data()[0]),
        //                                        &(S2.block("oOvV").data()[0]),
        //                                        &(S2.block("OOVV").data()[0]));
        //            }
        //            if (cycle > max_diis_vectors){
        //                if (cycle % max_diis_vectors == 2){
        //                    outfile->Printf(" -> DIIS");
        //                    diis_manager->extrapolate(5,
        //                                             &(S1.block("ov").data()[0]),
        //                                             &(S1.block("OV").data()[0]),
        //                                             &(S2.block("oovv").data()[0]),
        //                                             &(S2.block("oOvV").data()[0]),
        //                                             &(S2.block("OOVV").data()[0]));
        //                }
        //            }
        //        }
        if (print_ > 1) {
            outfile->Printf("\n  Compute recursive single commutator...");
        }

        // Compute the new similarity-transformed Hamiltonian
        double energy = E0_ + compute_hbar();

        if (print_ > 1) {
            outfile->Printf(" done.");
        }

        double delta_energy = energy - old_energy;
        old_energy = energy;

        double max_T1 = 0.0;
        double max_T2 = 0.0;
        T1.citerate(
            [&](const std::vector<size_t>&, const std::vector<SpinType>&, const double& value) {
                if (std::fabs(value) > std::fabs(max_T1))
                    max_T1 = value;
            });

        T2.citerate(
            [&](const std::vector<size_t>&, const std::vector<SpinType>&, const double& value) {
                if (std::fabs(value) > std::fabs(max_T2))
                    max_T2 = value;
            });

        //        double norm_H1 = Hbar1.block("ov").norm();
        //        double norm_H2 = Hbar2.block("oovv").norm();

        //        double norm_Hbar1_ex = std::sqrt(norm_H1a * norm_H1a +
        //        norm_H1b * norm_H1b);
        //        double norm_Hbar2_ex = std::sqrt(0.25 * norm_H2aa * norm_H2aa
        //        + norm_H2ab * norm_H2ab + 0.25 * norm_H2bb * norm_H2bb);

        double norm_T1 = T1.norm();
        double norm_T2 = T2.norm();

        outfile->Printf("\n    @SO-MR-DSRG %4d %20.12f %11.3e %10.3e %10.3e "
                        "%7.4f %7.4f %7.4f %7.4f",
                        cycle, energy, delta_energy, 0, 0, norm_T1, norm_T2, max_T1, max_T2);

        if (std::fabs(delta_energy) < foptions_->get_double("E_CONVERGENCE")) {
            converged = true;
        }

        if (cycle > foptions_->get_int("MAXITER")) {
            outfile->Printf("\n\n\tThe calculation did not converge in %d "
                            "cycles\n\tQuitting.\n",
                            foptions_->get_int("MAXITER"));

            converged = true;
            old_energy = 0.0;
        }

        cycle++;
    }
    outfile->Printf("\n  "
                    "----------------------------------------------------------"
                    "----------------------------------------");

    outfile->Printf("\n\n\n    SO-MR-DSRG correlation energy      = %25.15f", old_energy - E0_);
    outfile->Printf("\n  * SO-MR-DSRG total energy            = %25.15f\n", old_energy);

    // Set some environment variables
    psi::Process::environment.globals["CURRENT ENERGY"] = old_energy;
    psi::Process::environment.globals["SO-MR-DSRG ENERGY"] = old_energy;

    return old_energy;
}

void SOMRDSRG::mp2_guess() {
    T1["ia"] = F["ia"] * RInvDelta1["ia"];
    T2["ijab"] = V["ijab"] * RInvDelta2["ijab"];
    ;

    // Zero internal amplitudes
    T1.block("aa").zero();
    T2.block("aaaa").zero();

    double mp2_correlation_energy = 0.25 * T2["ijab"] * V["ijab"];

    outfile->Printf("\n\n    SCF energy                            = %20.15f", E0_);
    outfile->Printf("\n    SRG-PT2 correlation energy            = %20.15f",
                    mp2_correlation_energy);
    outfile->Printf("\n  * SRG-PT2 total energy                  = %20.15f\n",
                    E0_ + mp2_correlation_energy);
}

double SOMRDSRG::compute_hbar() {
    if (print_ > 1) {
        outfile->Printf("\n\n  Computing the similarity-transformed Hamiltonian");
        outfile->Printf("\n  "
                        "------------------------------------------------------"
                        "-----------");
        outfile->Printf("\n  nComm           C0                 |C1|           "
                        "       |C2|");
        outfile->Printf("\n  "
                        "------------------------------------------------------"
                        "-----------");
    }

    // Initialize Hbar and O with the normal ordered Hamiltonian
    Hbar0 = 0.0;
    Hbar1["pq"] = F["pq"];
    Hbar2["pqrs"] = V["pqrs"];

    O1["pq"] = F["pq"];
    O2["pqrs"] = V["pqrs"];

    if (print_ > 1) {
        outfile->Printf("\n  %2d %20.12f %20e %20e", 0, Hbar0, Hbar1.norm(), Hbar2.norm());
    }

    int maxn = foptions_->get_int("DSRG_RSC_NCOMM");
    double ct_threshold = foptions_->get_double("DSRG_RSC_THRESHOLD");
    for (int n = 1; n <= maxn; ++n) {
        double factor = 1.0 / static_cast<double>(n);

        double C0 = 0;
        C1.zero();
        C2.zero();

        // Compute the commutator C = 1/factor [O,S]
        H_eq_commutator_C_T(factor, O1, O2, T1, T2, C0, C1, C2);

        // Hbar += C
        Hbar0 += C0;
        Hbar1["pq"] += C1["pq"];
        Hbar2["pqrs"] += C2["pqrs"];

        // O = C
        O1["pq"] = C1["pq"];
        O2["pqrs"] = C2["pqrs"];

        // Check |C|
        double norm_C1 = C1.norm();
        double norm_C2 = C2.norm();

        if (print_ > 1) {
            outfile->Printf("\n  %2d %20.12f %20e %20e", n, C0, norm_C1, norm_C2);
        }
        if (std::sqrt(norm_C2 * norm_C2 + norm_C1 * norm_C1) < ct_threshold) {
            break;
        }
    }
    if (print_ > 1) {
        outfile->Printf("\n  "
                        "------------------------------------------------------"
                        "-----------");
    }
    return Hbar0;
}

void SOMRDSRG::H_eq_commutator_C_T(double factor, BlockedTensor& F, BlockedTensor& V,
                                   BlockedTensor& T1, BlockedTensor& T2, double& H0,
                                   BlockedTensor& H1, BlockedTensor& H2) {
    H0 += 1.000000 * Eta1["p1,p0"] * F["p0,h0"] * Gamma1["h0,h1"] * T1["h1,p1"];
    H0 += -0.500000 * F["a0,c0"] * Lambda2["a2,a3,a1,a0"] * T2["a1,c0,a2,a3"];
    H0 += 0.500000 * F["v0,a0"] * Lambda2["a0,a3,a1,a2"] * T2["a1,a2,v0,a3"];
    H0 += 0.500000 * Lambda2["a2,a3,a0,a1"] * T1["a0,v0"] * V["v0,a1,a2,a3"];
    H0 += -0.500000 * Lambda2["a0,a3,a1,a2"] * T1["c0,a0"] * V["a1,a2,c0,a3"];
    H0 += 0.250000 * Eta1["p2,p0"] * Eta1["p3,p1"] * Gamma1["h0,h2"] * Gamma1["h1,h3"] *
          T2["h2,h3,p2,p3"] * V["p0,p1,h0,h1"];
    H0 += 0.125000 * Eta1["p2,p0"] * Eta1["p3,p1"] * Lambda2["a0,a1,a2,a3"] * T2["a2,a3,p2,p3"] *
          V["p0,p1,a0,a1"];
    H0 += 0.125000 * Gamma1["h0,h2"] * Gamma1["h1,h3"] * Lambda2["a2,a3,a0,a1"] *
          T2["h2,h3,a2,a3"] * V["a0,a1,h0,h1"];
    H0 += 1.000000 * Eta1["p1,p0"] * Gamma1["h0,h1"] * Lambda2["a0,a2,a3,a1"] * T2["h1,a3,p1,a2"] *
          V["a1,p0,h0,a0"];
    if (foptions_->get_str("THREEPDC") != "ZERO") {
        H0 += 0.250000 * Lambda3["a3,a4,a0,a1,a2,a5"] * T2["h0,a5,a3,a4"] * V["a1,a2,h0,a0"];
        H0 += 0.250000 * Lambda3["a0,a1,a3,a4,a5,a2"] * T2["a4,a5,p0,a3"] * V["a2,p0,a0,a1"];
    }
    H1["h0,g0"] += 1.000000 * F["p0,g0"] * T1["h0,p0"];
    H1["g0,p0"] += -1.000000 * F["g0,h0"] * T1["h0,p0"];
    H1["h0,p0"] += 1.000000 * F["p1,h2"] * Gamma1["h2,h1"] * T2["h0,h1,p0,p1"];
    H1["h0,p0"] += -1.000000 * F["a0,h1"] * Gamma1["a1,a0"] * T2["h0,h1,p0,a1"];
    H1["g1,g0"] += 1.000000 * Gamma1["h1,h0"] * T1["h0,p0"] * V["p0,g1,h1,g0"];
    H1["g1,g0"] += -1.000000 * Gamma1["a1,a0"] * T1["c0,a0"] * V["a1,g1,c0,g0"];

    H1["g0,p0"] +=
        -0.500000 * Gamma1["h1,h0"] * Gamma1["h3,h2"] * T2["h0,h2,p0,p1"] * V["g0,p1,h1,h3"];
    H1["g0,p0"] += -0.500000 * Gamma1["a1,a0"] * T2["h0,h1,p0,a1"] * V["g0,a0,h0,h1"];
    H1["g0,p0"] +=
        1.000000 * Gamma1["a1,a0"] * Gamma1["h1,h0"] * T2["h0,h2,p0,a1"] * V["g0,a0,h1,h2"];
    H1["h0,g0"] += 0.500000 * Gamma1["h2,h1"] * T2["h0,h1,p0,p1"] * V["p0,p1,g0,h2"];
    H1["h0,g0"] +=
        0.500000 * Gamma1["a1,a0"] * Gamma1["a3,a2"] * T2["h0,h1,a1,a3"] * V["a0,a2,g0,h1"];
    H1["h0,g0"] +=
        -1.000000 * Gamma1["a1,a0"] * Gamma1["h2,h1"] * T2["h0,h1,p0,a1"] * V["p0,a0,g0,h2"];
    H1["g0,p0"] += -0.250000 * Lambda2["a2,a3,a0,a1"] * T2["a0,a1,p0,p1"] * V["g0,p1,a2,a3"];
    H1["h0,g0"] += 0.250000 * Lambda2["a0,a1,a2,a3"] * T2["h0,h1,a0,a1"] * V["a2,a3,g0,h1"];
    H1["h0,g0"] += 1.000000 * Lambda2["a2,a0,a3,a1"] * T2["h0,a1,p0,a0"] * V["p0,a3,g0,a2"];
    H1["g0,p0"] += -1.000000 * Lambda2["a2,a0,a3,a1"] * T2["h0,a1,p0,a0"] * V["g0,a3,h0,a2"];
    H1["h0,p0"] += 0.500000 * Lambda2["a1,a2,a0,a3"] * T2["a0,h0,p1,p0"] * V["p1,a3,a1,a2"];
    H1["h0,p0"] += -0.500000 * Lambda2["a0,a1,a2,a3"] * T2["h1,h0,a0,p0"] * V["a2,a3,h1,a1"];
    H1["g1,g0"] += 0.500000 * Lambda2["a3,a0,a1,a2"] * T2["a1,a2,v0,a0"] * V["v0,g1,a3,g0"];
    H1["g1,g0"] += -0.500000 * Lambda2["a0,a1,a3,a2"] * T2["c0,a2,a0,a1"] * V["a3,g1,c0,g0"];

    H2["h0,h1,g0,p0"] += 1.000000 * F["p1,g0"] * T2["h0,h1,p1,p0"];
    H2["h0,h1,p0,g0"] += 1.000000 * F["p1,g0"] * T2["h0,h1,p0,p1"];
    H2["g0,h0,p0,p1"] += -1.000000 * F["g0,h1"] * T2["h1,h0,p0,p1"];
    H2["h0,g0,p0,p1"] += -1.000000 * F["g0,h1"] * T2["h0,h1,p0,p1"];
    H2["h0,g2,g0,g1"] += 1.000000 * T1["h0,p0"] * V["p0,g2,g0,g1"];
    H2["g2,h0,g0,g1"] += 1.000000 * T1["h0,p0"] * V["g2,p0,g0,g1"];
    H2["g1,g2,p0,g0"] += -1.000000 * T1["h0,p0"] * V["g1,g2,h0,g0"];
    H2["g1,g2,g0,p0"] += -1.000000 * T1["h0,p0"] * V["g1,g2,g0,h0"];
    H2["g0,g1,p0,p1"] += 0.500000 * T2["h0,h1,p0,p1"] * V["g0,g1,h0,h1"];
    H2["g0,g1,p0,p1"] += -1.000000 * Eta1["a1,a0"] * T2["a0,h0,p0,p1"] * V["g0,g1,a1,h0"];
    H2["h0,h1,g0,g1"] += 0.500000 * T2["h0,h1,p0,p1"] * V["p0,p1,g0,g1"];
    H2["h0,h1,g0,g1"] += -1.000000 * Gamma1["a0,a1"] * T2["h0,h1,a0,p0"] * V["a1,p0,g0,g1"];
    H2["g1,h0,g0,p0"] += 1.000000 * Gamma1["h2,h1"] * T2["h1,h0,p1,p0"] * V["p1,g1,h2,g0"];
    H2["h0,g1,g0,p0"] += 1.000000 * Gamma1["h2,h1"] * T2["h0,h1,p1,p0"] * V["p1,g1,h2,g0"];
    H2["g1,h0,p0,g0"] += 1.000000 * Gamma1["h2,h1"] * T2["h1,h0,p0,p1"] * V["p1,g1,h2,g0"];
    H2["h0,g1,p0,g0"] += 1.000000 * Gamma1["h2,h1"] * T2["h0,h1,p0,p1"] * V["p1,g1,h2,g0"];
    H2["g1,h0,g0,p0"] += -1.000000 * Gamma1["a0,a1"] * T2["h1,h0,a0,p0"] * V["a1,g1,h1,g0"];
    H2["h0,g1,g0,p0"] += -1.000000 * Gamma1["a0,a1"] * T2["h0,h1,a0,p0"] * V["a1,g1,h1,g0"];
    H2["g1,h0,p0,g0"] += -1.000000 * Gamma1["a0,a1"] * T2["h1,h0,p0,a0"] * V["a1,g1,h1,g0"];
    H2["h0,g1,p0,g0"] += -1.000000 * Gamma1["a0,a1"] * T2["h0,h1,p0,a0"] * V["a1,g1,h1,g0"];

    outfile->Printf("\n   H0  = %20.12f", H0);
    outfile->Printf("\n  |H1| = %20.12f", H1.norm(1));
    outfile->Printf("\n  |H2| = %20.12f", H2.norm(1));

    // Scale by factor
    H0 *= factor;
    H1.scale(factor);
    H2.scale(factor);

    // => Add the term  + [F + V,T1 + T2]^+ <= //
    H0 *= 2.0;
    F["pq"] = H1["pq"];
    H1["pq"] += F["qp"];

    V["pqrs"] = H2["pqrs"];
    H2["pqrs"] += V["rspq"];
}

void SOMRDSRG::update_T1() {
    R1.zero();

    R1["ia"] = Hbar1["ia"] * RInvDelta1["ia"];
    R1["ia"] += T1["ia"] * Delta1["ia"] * RInvDelta1["ia"];

    // Compute the change in amplitudes
    DT1["ia"] = T1["ia"];
    DT1["ia"] -= R1["ia"];

    T1["ia"] = R1["ia"];

    // Zero internal amplitudes
    T1.block("aa").zero();
}

void SOMRDSRG::update_T2() {
    R2.zero();

    R2["ijab"] = Hbar2["ijab"] * RInvDelta2["ijab"];
    R2["ijab"] += T2["ijab"] * Delta2["ijab"] * RInvDelta2["ijab"];

    // Compute the change in amplitudes
    DT2["ijab"] = T2["ijab"];
    DT2["ijab"] -= R2["ijab"];

    T2["ijab"] = R2["ijab"];

    // Zero internal amplitudes
    T2.block("aaaa").zero();
}

void SOMRDSRG::cleanup() {}

double SOMRDSRG::renormalized_denominator(double D) {
    double Z = std::sqrt(s_) * D;
    if (std::fabs(Z) < std::pow(0.1, taylor_threshold_)) {
        return Taylor_Exp(Z, taylor_order_) * std::sqrt(s_);
    } else {
        return (1.0 - std::exp(-s_ * std::pow(D, 2.0))) / D;
    }
}
} // namespace forte
