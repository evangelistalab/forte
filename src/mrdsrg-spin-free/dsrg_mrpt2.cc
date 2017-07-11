/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2017 by its authors (see COPYING, COPYING.LESSER,
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

#include <algorithm>
#include <math.h>
#include <numeric>
#include <ctype.h>

#include "psi4/libpsio/psio.h"
#include "psi4/libpsio/psio.hpp"
#include "psi4/libqt/qt.h"

#include "../ci_rdms.h"
#include "../fci/fci.h"
#include "../mini-boost/boost/format.hpp"
#include "../fci_mo.h"
#include "../fci/fci_solver.h"
#include "dsrg_mrpt2.h"

using namespace ambit;

namespace psi {
namespace forte {

DSRG_MRPT2::DSRG_MRPT2(Reference reference, SharedWavefunction ref_wfn, Options& options,
                       std::shared_ptr<ForteIntegrals> ints,
                       std::shared_ptr<MOSpaceInfo> mo_space_info)
    : Wavefunction(options), reference_(reference), ints_(ints), mo_space_info_(mo_space_info),
      tensor_type_(ambit::CoreTensor), BTF_(new BlockedTensorFactory(options)) {
    // Copy the wavefunction information
    reference_wavefunction_ = ref_wfn;
    shallow_copy(ref_wfn);

    print_method_banner({"Driven Similarity Renormalization Group MBPT2",
                         "Chenyang Li, Kevin Hannon, Francesco Evangelista"});
    outfile->Printf("\n    References:");
    outfile->Printf("\n      u-DSRG-MRPT2:    J. Chem. Theory Comput. 2015, 11, 2097.");
    outfile->Printf("\n      (pr-)DSRG-MRPT2: J. Chem. Phys. 2017, 146, 124132.");

    startup();
    print_summary();
}

DSRG_MRPT2::~DSRG_MRPT2() { cleanup(); }

void DSRG_MRPT2::startup() {
    s_ = options_.get_double("DSRG_S");
    if (s_ < 0) {
        outfile->Printf("\n  S parameter for DSRG must >= 0!");
        throw PSIEXCEPTION("S parameter for DSRG must >= 0!");
    }
    taylor_threshold_ = options_.get_int("TAYLOR_THRESHOLD");
    if (taylor_threshold_ <= 0) {
        outfile->Printf("\n  Threshold for Taylor expansion must be an integer "
                        "greater than 0!");
        throw PSIEXCEPTION("Threshold for Taylor expansion must be an integer "
                           "greater than 0!");
    }

    source_ = options_.get_str("SOURCE");
    if (source_ != "STANDARD" && source_ != "LABS" && source_ != "DYSON") {
        outfile->Printf("\n  Warning: SOURCE option \"%s\" is not implemented "
                        "in DSRG-MRPT2. Changed to STANDARD.",
                        source_.c_str());
        source_ = "STANDARD";
    }
    if (source_ == "STANDARD") {
        dsrg_source_ = std::make_shared<STD_SOURCE>(s_, taylor_threshold_);
    } else if (source_ == "LABS") {
        dsrg_source_ = std::make_shared<LABS_SOURCE>(s_, taylor_threshold_);
    } else if (source_ == "DYSON") {
        dsrg_source_ = std::make_shared<DYSON_SOURCE>(s_, taylor_threshold_);
    }

    ntamp_ = options_.get_int("NTAMP");
    intruder_tamp_ = options_.get_double("INTRUDER_TAMP");
    internal_amp_ = options_.get_str("INTERNAL_AMP") != "NONE";
    internal_amp_select_ = options_.get_str("INTERNAL_AMP_SELECT");

    // get frozen core energy
    frozen_core_energy_ = ints_->frozen_core_energy();

    // get reference energy
    Eref_ = reference_.get_Eref();

    // orbital spaces
    BlockedTensor::reset_mo_spaces();
    BlockedTensor::set_expert_mode(true);
    acore_mos_ = mo_space_info_->get_corr_abs_mo("RESTRICTED_DOCC");
    bcore_mos_ = mo_space_info_->get_corr_abs_mo("RESTRICTED_DOCC");
    aactv_mos_ = mo_space_info_->get_corr_abs_mo("ACTIVE");
    bactv_mos_ = mo_space_info_->get_corr_abs_mo("ACTIVE");
    avirt_mos_ = mo_space_info_->get_corr_abs_mo("RESTRICTED_UOCC");
    bvirt_mos_ = mo_space_info_->get_corr_abs_mo("RESTRICTED_UOCC");

    // define space labels
    acore_label_ = "c";
    aactv_label_ = "a";
    avirt_label_ = "v";
    bcore_label_ = "C";
    bactv_label_ = "A";
    bvirt_label_ = "V";
    BTF_->add_mo_space(acore_label_, "mn", acore_mos_, AlphaSpin);
    BTF_->add_mo_space(bcore_label_, "MN", bcore_mos_, BetaSpin);
    BTF_->add_mo_space(aactv_label_, "uvwxyz123", aactv_mos_, AlphaSpin);
    BTF_->add_mo_space(bactv_label_, "UVWXYZ!@#", bactv_mos_, BetaSpin);
    BTF_->add_mo_space(avirt_label_, "ef", avirt_mos_, AlphaSpin);
    BTF_->add_mo_space(bvirt_label_, "EF", bvirt_mos_, BetaSpin);

    // map space labels to mo spaces
    label_to_spacemo_[acore_label_[0]] = acore_mos_;
    label_to_spacemo_[bcore_label_[0]] = bcore_mos_;
    label_to_spacemo_[aactv_label_[0]] = aactv_mos_;
    label_to_spacemo_[bactv_label_[0]] = bactv_mos_;
    label_to_spacemo_[avirt_label_[0]] = avirt_mos_;
    label_to_spacemo_[bvirt_label_[0]] = bvirt_mos_;

    // define composite spaces
    BTF_->add_composite_mo_space("h", "ijkl", {acore_label_, aactv_label_});
    BTF_->add_composite_mo_space("H", "IJKL", {bcore_label_, bactv_label_});
    BTF_->add_composite_mo_space("p", "abcd", {aactv_label_, avirt_label_});
    BTF_->add_composite_mo_space("P", "ABCD", {bactv_label_, bvirt_label_});
    BTF_->add_composite_mo_space("g", "pqrs", {acore_label_, aactv_label_, avirt_label_});
    BTF_->add_composite_mo_space("G", "PQRS", {bcore_label_, bactv_label_, bvirt_label_});

    // fill in density matrices
    Eta1_ = BTF_->build(tensor_type_, "Eta1", spin_cases({"aa"}));
    Gamma1_ = BTF_->build(tensor_type_, "Gamma1", spin_cases({"aa"}));
    Lambda2_ = BTF_->build(tensor_type_, "Lambda2", spin_cases({"aaaa"}));
    if (options_.get_str("THREEPDC") != "ZERO") {
        Lambda3_ = BTF_->build(tensor_type_, "Lambda3", spin_cases({"aaaaaa"}));
    }
    build_density();

    // prepare integrals
    V_ = BTF_->build(tensor_type_, "V", spin_cases({"pphh"}));
    build_ints();

    // build Fock matrix and its diagonal elements
    size_t ncmo = mo_space_info_->size("CORRELATED");
    Fa_ = std::vector<double>(ncmo);
    Fb_ = std::vector<double>(ncmo);
    F_ = BTF_->build(tensor_type_, "Fock", spin_cases({"gc", "pa", "vv"}));
    build_fock();

    multi_state_ = options_["AVG_STATE"].size() != 0;
    relax_ref_ = options_.get_str("RELAX_REF");
    if (relax_ref_ != "NONE" && relax_ref_ != "ONCE") {
        outfile->Printf("\n\n  Warning: RELAX_REF option \"%s\" is not "
                        "supported. Change to ONCE",
                        relax_ref_.c_str());
        relax_ref_ = "ONCE";
    }

    // Prepare Hbar
    if (relax_ref_ != "NONE" || multi_state_) {
        Hbar1_ = BTF_->build(tensor_type_, "1-body Hbar", spin_cases({"aa"}));
        Hbar2_ = BTF_->build(tensor_type_, "2-body Hbar", spin_cases({"aaaa"}));
        Hbar1_["uv"] = F_["uv"];
        Hbar1_["UV"] = F_["UV"];
        Hbar2_["uvxy"] = V_["uvxy"];
        Hbar2_["uVxY"] = V_["uVxY"];
        Hbar2_["UVXY"] = V_["UVXY"];

        if (options_.get_bool("FORM_HBAR3")) {
            Hbar3_ = BTF_->build(tensor_type_, "3-body Hbar", spin_cases({"aaaaaa"}));
        }
    }

    // ignore semicanonical test
    std::string actv_type = options_.get_str("FCIMO_ACTV_TYPE");
    if (actv_type != "COMPLETE" && actv_type != "DOCI") {
        ignore_semicanonical_ = true;
    }

    // initialize timer for commutator
    dsrg_time_ = DSRG_TIME();

    // print levels
    print_ = options_.get_int("PRINT");
    if (print_ > 1) {
        Gamma1_.print(stdout);
        Eta1_.print(stdout);
        F_.print(stdout);
    }
    if (print_ > 2) {
        V_.print(stdout);
        Lambda2_.print(stdout);
    }
    if (print_ > 3) {
        Lambda3_.print(stdout);
    }
}

void DSRG_MRPT2::build_density() {
    // prepare one-particle and one-hole densities
    Gamma1_.block("aa")("pq") = reference_.L1a()("pq");
    Gamma1_.block("AA")("pq") = reference_.L1a()("pq");

    (Eta1_.block("aa")).iterate([&](const std::vector<size_t>& i, double& value) {
        value = i[0] == i[1] ? 1.0 : 0.0;
    });
    (Eta1_.block("AA")).iterate([&](const std::vector<size_t>& i, double& value) {
        value = i[0] == i[1] ? 1.0 : 0.0;
    });
    Eta1_.block("aa")("pq") -= reference_.L1a()("pq");
    Eta1_.block("AA")("pq") -= reference_.L1a()("pq");

    // prepare two-body density cumulants
    Lambda2_.block("aaaa")("pqrs") = reference_.L2aa()("pqrs");
    Lambda2_.block("aAaA")("pqrs") = reference_.L2ab()("pqrs");
    Lambda2_.block("AAAA")("pqrs") = reference_.L2bb()("pqrs");

    // prepare three-body density cumulants
    if (options_.get_str("THREEPDC") != "ZERO") {
        Lambda3_.block("aaaaaa")("pqrstu") = reference_.L3aaa()("pqrstu");
        Lambda3_.block("aaAaaA")("pqrstu") = reference_.L3aab()("pqrstu");
        Lambda3_.block("aAAaAA")("pqrstu") = reference_.L3abb()("pqrstu");
        Lambda3_.block("AAAAAA")("pqrstu") = reference_.L3bbb()("pqrstu");
    }
}

void DSRG_MRPT2::build_ints() {
    V_.iterate([&](const std::vector<size_t>& i, const std::vector<SpinType>& spin, double& value) {
        if ((spin[0] == AlphaSpin) and (spin[1] == AlphaSpin))
            value = ints_->aptei_aa(i[0], i[1], i[2], i[3]);
        if ((spin[0] == AlphaSpin) and (spin[1] == BetaSpin))
            value = ints_->aptei_ab(i[0], i[1], i[2], i[3]);
        if ((spin[0] == BetaSpin) and (spin[1] == BetaSpin))
            value = ints_->aptei_bb(i[0], i[1], i[2], i[3]);
    });
}

void DSRG_MRPT2::build_fock() {
    // build Fock matrix
    for (const auto& block : F_.block_labels()) {
        // lowercase: alpha spin
        if (islower(block[0])) {
            F_.block(block).iterate([&](const std::vector<size_t>& i, double& value) {
                size_t np = label_to_spacemo_[block[0]][i[0]];
                size_t nq = label_to_spacemo_[block[1]][i[1]];
                value = ints_->oei_a(np, nq);

                for (const size_t& nm : acore_mos_) {
                    value += ints_->aptei_aa(np, nm, nq, nm);
                    value += ints_->aptei_ab(np, nm, nq, nm);
                }
            });
        } else {
            F_.block(block).iterate([&](const std::vector<size_t>& i, double& value) {
                size_t np = label_to_spacemo_[block[0]][i[0]];
                size_t nq = label_to_spacemo_[block[1]][i[1]];
                value = ints_->oei_b(np, nq);

                for (const size_t& nm : bcore_mos_) {
                    value += ints_->aptei_bb(np, nm, nq, nm);
                    value += ints_->aptei_ab(nm, np, nm, nq);
                }
            });
        }
    }

    // core-core block
    BlockedTensor VFock =
        ambit::BlockedTensor::build(tensor_type_, "VFock", {"caca", "cAcA", "aCaC", "CACA"});
    VFock.iterate(
        [&](const std::vector<size_t>& i, const std::vector<SpinType>& spin, double& value) {
            if ((spin[0] == AlphaSpin) and (spin[1] == AlphaSpin))
                value = ints_->aptei_aa(i[0], i[1], i[2], i[3]);
            if ((spin[0] == AlphaSpin) and (spin[1] == BetaSpin))
                value = ints_->aptei_ab(i[0], i[1], i[2], i[3]);
            if ((spin[0] == BetaSpin) and (spin[1] == BetaSpin))
                value = ints_->aptei_bb(i[0], i[1], i[2], i[3]);
        });
    F_["mn"] += VFock["mvnu"] * Gamma1_["uv"];
    F_["mn"] += VFock["mVnU"] * Gamma1_["UV"];
    F_["MN"] += VFock["vMuN"] * Gamma1_["uv"];
    F_["MN"] += VFock["MVNU"] * Gamma1_["UV"];

    // virtual-virtual block
    VFock = ambit::BlockedTensor::build(tensor_type_, "VFock", {"vava", "vAvA", "aVaV", "VAVA"});
    VFock.iterate(
        [&](const std::vector<size_t>& i, const std::vector<SpinType>& spin, double& value) {
            if ((spin[0] == AlphaSpin) and (spin[1] == AlphaSpin))
                value = ints_->aptei_aa(i[0], i[1], i[2], i[3]);
            if ((spin[0] == AlphaSpin) and (spin[1] == BetaSpin))
                value = ints_->aptei_ab(i[0], i[1], i[2], i[3]);
            if ((spin[0] == BetaSpin) and (spin[1] == BetaSpin))
                value = ints_->aptei_bb(i[0], i[1], i[2], i[3]);
        });
    F_["ef"] += VFock["evfu"] * Gamma1_["uv"];
    F_["ef"] += VFock["eVfU"] * Gamma1_["UV"];
    F_["EF"] += VFock["vEuF"] * Gamma1_["uv"];
    F_["EF"] += VFock["EVFU"] * Gamma1_["UV"];

    // off-diagonal and all-active blocks
    VFock = ambit::BlockedTensor::build(tensor_type_, "VFock", {"paha", "pAhA", "aPaH", "PAHA"});
    VFock.iterate(
        [&](const std::vector<size_t>& i, const std::vector<SpinType>& spin, double& value) {
            if ((spin[0] == AlphaSpin) and (spin[1] == AlphaSpin))
                value = ints_->aptei_aa(i[0], i[1], i[2], i[3]);
            if ((spin[0] == AlphaSpin) and (spin[1] == BetaSpin))
                value = ints_->aptei_ab(i[0], i[1], i[2], i[3]);
            if ((spin[0] == BetaSpin) and (spin[1] == BetaSpin))
                value = ints_->aptei_bb(i[0], i[1], i[2], i[3]);
        });
    F_["ai"] += VFock["aviu"] * Gamma1_["uv"];
    F_["ai"] += VFock["aViU"] * Gamma1_["UV"];
    F_["AI"] += VFock["vAuI"] * Gamma1_["uv"];
    F_["AI"] += VFock["AVIU"] * Gamma1_["UV"];

    // obtain diagonal elements of Fock matrix
    F_.iterate([&](const std::vector<size_t>& i, const std::vector<SpinType>& spin, double& value) {
        if (spin[0] == AlphaSpin and (i[0] == i[1])) {
            Fa_[i[0]] = value;
        }
        if (spin[0] == BetaSpin and (i[0] == i[1])) {
            Fb_[i[0]] = value;
        }
    });
}

bool DSRG_MRPT2::check_semicanonical() {
    print_h2("Checking Orbitals");

    // zero diagonal elements
    F_.iterate([&](const std::vector<size_t>& i, const std::vector<SpinType>& spin, double& value) {
        if (spin[0] == AlphaSpin and (i[0] == i[1])) {
            value = 0.0;
        }
        if (spin[0] == BetaSpin and (i[0] == i[1])) {
            value = 0.0;
        }
    });

    // off diagonal elements of diagonal blocks
    double Foff_sum = 0.0;
    std::vector<double> Foff;
    for (const auto& block : {"cc", "aa", "vv", "CC", "AA", "VV"}) {
        double value = F_.block(block).norm();
        Foff.emplace_back(value);
        Foff_sum += value;
    }

    // add diagonal elements back
    F_.iterate([&](const std::vector<size_t>& i, const std::vector<SpinType>& spin, double& value) {
        if (spin[0] == AlphaSpin and (i[0] == i[1])) {
            value = Fa_[i[0]];
        }
        if (spin[0] == BetaSpin and (i[0] == i[1])) {
            value = Fb_[i[0]];
        }
    });

    bool semi = false;
    double threshold = 0.1 * std::sqrt(options_.get_double("E_CONVERGENCE"));
    if (Foff_sum > threshold) {
        std::string sep(3 + 16 * 3, '-');
        outfile->Printf("\n    Warning! Orbitals are not semi-canonicalized!");
        outfile->Printf("\n    Off-Diagonal norms of the core, active, virtual "
                        "blocks of Fock matrix");
        outfile->Printf("\n       %15s %15s %15s", "core", "active", "virtual");
        outfile->Printf("\n    %s", sep.c_str());
        outfile->Printf("\n    Fa %15.10f %15.10f %15.10f", Foff[0], Foff[1], Foff[2]);
        outfile->Printf("\n    Fb %15.10f %15.10f %15.10f", Foff[3], Foff[4], Foff[5]);
        outfile->Printf("\n    %s\n", sep.c_str());

        outfile->Printf("\n    DSRG energy is reliable roughly to the same "
                        "digit as max(|F_ij|, i != j), F: Fock diag. blocks.");
    } else {
        outfile->Printf("\n    Orbitals are semi-canonicalized.");
        semi = true;
    }

    if (ignore_semicanonical_ && Foff_sum > threshold) {
        std::string actv_type = options_.get_str("FCIMO_ACTV_TYPE");
        if (actv_type == "CIS" || actv_type == "CISD") {
            outfile->Printf("\n    It is OK for Fock (active) not being diagonal because %s "
                            "active space is incomplete.",
                            actv_type.c_str());
            outfile->Printf("\n    Please inspect if the Fock diag. blocks (C, AH, AP, V) "
                            "are diagonal or not in the prior CI step.");

        } else {
            outfile->Printf("\n    Warning: ignore testing of semi-canonical orbitals.");
            outfile->Printf("\n    Please inspect if the Fock diag. blocks (C, A, V) are "
                            "diagonal or not.");
        }

        semi = true;
    }
    outfile->Printf("\n");

    return semi;
}

void DSRG_MRPT2::print_summary() {
    // Print a summary
    std::vector<std::pair<std::string, int>> calculation_info{{"ntamp", ntamp_}};

    std::vector<std::pair<std::string, double>> calculation_info_double{
        {"flow parameter", s_},
        {"taylor expansion threshold", pow(10.0, -double(taylor_threshold_))},
        {"intruder_tamp", intruder_tamp_}};

    std::vector<std::pair<std::string, std::string>> calculation_info_string{
        {"int_type", options_.get_str("INT_TYPE")},
        {"source operator", source_},
        {"reference relaxation", relax_ref_}};

    if (multi_state_) {
        calculation_info_string.push_back({"state_type", "MULTI-STATE"});
        calculation_info_string.push_back(
            {"multi-state type", options_.get_str("DSRG_MULTI_STATE")});
    } else {
        calculation_info_string.push_back({"state_type", "STATE-SPECIFIC"});
    }

    if (internal_amp_) {
        calculation_info_string.push_back({"internal_amp", options_.get_str("INTERNAL_AMP")});
        calculation_info_string.push_back({"internal_amp_select", internal_amp_select_});
    }

    if (options_.get_bool("FORM_HBAR3")) {
        calculation_info_string.push_back({"form Hbar3", "TRUE"});
    } else {
        calculation_info_string.push_back({"form Hbar3", "FALSE"});
    }

    // Print some information
    outfile->Printf("\n\n  ==> Calculation Information <==\n");
    for (auto& str_dim : calculation_info) {
        outfile->Printf("\n    %-40s %15d", str_dim.first.c_str(), str_dim.second);
    }
    for (auto& str_dim : calculation_info_double) {
        outfile->Printf("\n    %-40s %15.3e", str_dim.first.c_str(), str_dim.second);
    }
    for (auto& str_dim : calculation_info_string) {
        outfile->Printf("\n    %-40s %15s", str_dim.first.c_str(), str_dim.second.c_str());
    }

    if (options_.get_bool("MEMORY_SUMMARY")) {
        BTF_->print_memory_info();
    }
}

void DSRG_MRPT2::cleanup() { dsrg_time_.print_comm_time(); }

double DSRG_MRPT2::compute_ref() {
    Timer timer;
    std::string str = "Computing reference energy";
    outfile->Printf("\n    %-40s ...", str.c_str());
    double E = 0.0;

    for (const std::string block : {"cc", "CC"}) {
        F_.block(block).iterate([&](const std::vector<size_t>& i, double& value) {
            if (i[0] == i[1]) {
                E += 0.5 * value;
            }
        });
        Hoei_.block(block).iterate([&](const std::vector<size_t>& i, double& value) {
            if (i[0] == i[1]) {
                E += 0.5 * value;
            }
        });
    }

    E += 0.5 * Hoei_["uv"] * Gamma1_["vu"];
    E += 0.5 * Hoei_["UV"] * Gamma1_["VU"];
    E += 0.5 * F_["uv"] * Gamma1_["vu"];
    E += 0.5 * F_["UV"] * Gamma1_["VU"];

    E += 0.25 * V_["uvxy"] * Lambda2_["xyuv"];
    E += 0.25 * V_["UVXY"] * Lambda2_["XYUV"];
    E += V_["uVxY"] * Lambda2_["xYuV"];

    double Enuc = Process::environment.molecule()->nuclear_repulsion_energy();

    outfile->Printf("  Done. Timing %15.6f s", timer.get());
    return E + frozen_core_energy_ + Enuc;
}

double DSRG_MRPT2::compute_energy() {
    // check semi-canonical orbitals
    print_h2("Checking Orbitals");
    semi_canonical_ = check_semicanonical();
    if (!semi_canonical_) {
        outfile->Printf("\n    Orbital invariant formalism is employed for "
                        "DSRG-MRPT2.");
        U_ = ambit::BlockedTensor::build(tensor_type_, "U", spin_cases({"gg"}));
        std::vector<std::vector<double>> eigens = diagonalize_Fock_diagblocks(U_);
        Fa_ = eigens[0];
        Fb_ = eigens[1];
    }

    Timer DSRG_energy;
    outfile->Printf("\n\n  ==> Computing DSRG-MRPT2 ... <==\n");

    // Compute T2 and T1
    T1_ = BTF_->build(tensor_type_, "T1 Amplitudes", spin_cases({"hp"}));
    T2_ = BTF_->build(tensor_type_, "T2 Amplitudes", spin_cases({"hhpp"}));
    compute_t2();
    compute_t1();

    // Compute effective integrals
    renormalize_V();
    renormalize_F();
    if (print_ > 1)
        F_.print(stdout);
    if (print_ > 2) {
        T1_.print(stdout);
        T2_.print(stdout);
        V_.print(stdout);
    }

    // Compute DSRG-MRPT2 correlation energy
    double Etemp = 0.0;
    double EVT2 = 0.0;
    double Ecorr = 0.0;
    double Etotal = 0.0;
    std::vector<std::pair<std::string, double>> energy;
    energy.push_back({"E0 (reference)", Eref_});

    Etemp = E_FT1();
    Ecorr += Etemp;
    energy.push_back({"<[F, T1]>", Etemp});

    Etemp = E_FT2();
    Ecorr += Etemp;
    energy.push_back({"<[F, T2]>", Etemp});

    Etemp = E_VT1();
    Ecorr += Etemp;
    energy.push_back({"<[V, T1]>", Etemp});

    Etemp = E_VT2_2();
    EVT2 += Etemp;
    energy.push_back({"<[V, T2]> (C_2)^4", Etemp});

    Etemp = E_VT2_4HH();
    EVT2 += Etemp;
    energy.push_back({"<[V, T2]> C_4 (C_2)^2 HH", Etemp});

    Etemp = E_VT2_4PP();
    EVT2 += Etemp;
    energy.push_back({"<[V, T2]> C_4 (C_2)^2 PP", Etemp});

    Etemp = E_VT2_4PH();
    EVT2 += Etemp;
    energy.push_back({"<[V, T2]> C_4 (C_2)^2 PH", Etemp});

    if (options_.get_str("THREEPDC") != "ZERO") {
        Etemp = E_VT2_6();
    } else {
        Etemp = 0.0;
    }
    EVT2 += Etemp;
    energy.push_back({"<[V, T2]> C_6 C_2", Etemp});

    Ecorr += EVT2;
    Etotal = Ecorr + Eref_;
    energy.push_back({"<[V, T2]>", EVT2});
    energy.push_back({"DSRG-MRPT2 correlation energy", Ecorr});
    energy.push_back({"DSRG-MRPT2 total energy", Etotal});
    Hbar0_ = Ecorr;

    // Analyze T1 and T2
    outfile->Printf("\n\n  ==> Excitation Amplitudes Summary <==\n");
    outfile->Printf("\n    Active Indices: ");
    int c = 0;
    for (const auto& idx : aactv_mos_) {
        outfile->Printf("%4zu ", idx);
        if (++c % 10 == 0)
            outfile->Printf("\n    %16c", ' ');
    }
    check_t1();
    check_t2();
    energy.push_back({"max(T1)", T1max_});
    energy.push_back({"max(T2)", T2max_});
    energy.push_back({"||T1||", T1norm_});
    energy.push_back({"||T2||", T2norm_});

    outfile->Printf("\n\n  ==> Possible Intruders <==\n");
    print_intruder("A", lt1a_);
    print_intruder("B", lt1b_);
    print_intruder("AA", lt2aa_);
    print_intruder("AB", lt2ab_);
    print_intruder("BB", lt2bb_);

    // Print energy summary
    outfile->Printf("\n\n  ==> DSRG-MRPT2 Energy Summary <==\n");
    for (auto& str_dim : energy) {
        outfile->Printf("\n    %-30s = %22.15f", str_dim.first.c_str(), str_dim.second);
    }

    Process::environment.globals["CURRENT ENERGY"] = Etotal;
    outfile->Printf("\n\n  Energy took %10.3f s", DSRG_energy.get());
    outfile->Printf("\n");

    // relax reference
    if (relax_ref_ != "NONE" || multi_state_) {
        BlockedTensor C1 = BTF_->build(tensor_type_, "C1", spin_cases({"aa"}));
        BlockedTensor C2 = BTF_->build(tensor_type_, "C2", spin_cases({"aaaa"}));
        H1_T1_C1(F_, T1_, 0.5, C1);
        H1_T2_C1(F_, T2_, 0.5, C1);
        H2_T1_C1(V_, T1_, 0.5, C1);
        H2_T2_C1(V_, T2_, 0.5, C1);
        H1_T2_C2(F_, T2_, 0.5, C2);
        H2_T1_C2(V_, T1_, 0.5, C2);
        H2_T2_C2(V_, T2_, 0.5, C2);

        Hbar1_["ij"] += C1["ij"];
        Hbar1_["ij"] += C1["ji"];
        Hbar1_["IJ"] += C1["IJ"];
        Hbar1_["IJ"] += C1["JI"];
        Hbar2_["ijkl"] += C2["ijkl"];
        Hbar2_["ijkl"] += C2["klij"];
        Hbar2_["iJkL"] += C2["iJkL"];
        Hbar2_["iJkL"] += C2["kLiJ"];
        Hbar2_["IJKL"] += C2["IJKL"];
        Hbar2_["IJKL"] += C2["KLIJ"];

        if (options_.get_bool("FORM_HBAR3")) {
            BlockedTensor C3 = BTF_->build(tensor_type_, "C3", spin_cases({"aaaaaa"}));
            H2_T2_C3(V_, T2_, 0.5, C3);

            Hbar3_["uvwxyz"] += C3["uvwxyz"];
            Hbar3_["uvwxyz"] += C3["xyzuvw"];
            Hbar3_["uvWxyZ"] += C3["uvWxyZ"];
            Hbar3_["uvWxyZ"] += C3["xyZuvW"];
            Hbar3_["uVWxYZ"] += C3["uVWxYZ"];
            Hbar3_["uVWxYZ"] += C3["xYZuVW"];
            Hbar3_["UVWXYZ"] += C3["UVWXYZ"];
            Hbar3_["UVWXYZ"] += C3["XYZUVW"];
        }
    }

    return Etotal;
}

void DSRG_MRPT2::compute_t2() {
    Timer timer;
    std::string str = "Computing T2 amplitudes";
    outfile->Printf("\n    %-40s ...", str.c_str());

    T2_["ijab"] = V_["abij"];
    T2_["iJaB"] = V_["aBiJ"];
    T2_["IJAB"] = V_["ABIJ"];

    // transform to semi-canonical basis
    if (!semi_canonical_) {
        BlockedTensor tempT2 =
            ambit::BlockedTensor::build(tensor_type_, "Temp T2", spin_cases({"hhpp"}));
        tempT2["klab"] = U_["ki"] * U_["lj"] * T2_["ijab"];
        tempT2["kLaB"] = U_["ki"] * U_["LJ"] * T2_["iJaB"];
        tempT2["KLAB"] = U_["KI"] * U_["LJ"] * T2_["IJAB"];
        T2_["ijcd"] = tempT2["ijab"] * U_["db"] * U_["ca"];
        T2_["iJcD"] = tempT2["iJaB"] * U_["DB"] * U_["ca"];
        T2_["IJCD"] = tempT2["IJAB"] * U_["DB"] * U_["CA"];
    }

    T2_.iterate(
        [&](const std::vector<size_t>& i, const std::vector<SpinType>& spin, double& value) {
            if (std::fabs(value) > 1.0e-12) {
                if ((spin[0] == AlphaSpin) and (spin[1] == AlphaSpin)) {
                    value *= dsrg_source_->compute_renormalized_denominator(Fa_[i[0]] + Fa_[i[1]] -
                                                                            Fa_[i[2]] - Fa_[i[3]]);
                } else if ((spin[0] == AlphaSpin) and (spin[1] == BetaSpin)) {
                    value *= dsrg_source_->compute_renormalized_denominator(Fa_[i[0]] + Fb_[i[1]] -
                                                                            Fa_[i[2]] - Fb_[i[3]]);
                } else if ((spin[0] == BetaSpin) and (spin[1] == BetaSpin)) {
                    value *= dsrg_source_->compute_renormalized_denominator(Fb_[i[0]] + Fb_[i[1]] -
                                                                            Fb_[i[2]] - Fb_[i[3]]);
                }
            } else {
                value = 0.0;
            }
        });

    // transform back to non-canonical basis
    if (!semi_canonical_) {
        BlockedTensor tempT2 =
            ambit::BlockedTensor::build(tensor_type_, "Temp T2", spin_cases({"hhpp"}));
        tempT2["klab"] = U_["ik"] * U_["jl"] * T2_["ijab"];
        tempT2["kLaB"] = U_["ik"] * U_["JL"] * T2_["iJaB"];
        tempT2["KLAB"] = U_["IK"] * U_["JL"] * T2_["IJAB"];
        T2_["ijcd"] = tempT2["ijab"] * U_["bd"] * U_["ac"];
        T2_["iJcD"] = tempT2["iJaB"] * U_["BD"] * U_["ac"];
        T2_["IJCD"] = tempT2["IJAB"] * U_["BD"] * U_["AC"];
    }

    // internal amplitudes (AA->AA)
    std::string internal_amp = options_.get_str("INTERNAL_AMP");
    if (internal_amp.find("DOUBLES") != string::npos) {
        size_t nactv1 = mo_space_info_->size("ACTIVE");
        size_t nactv2 = nactv1 * nactv1;
        size_t nactv3 = nactv2 * nactv1;
        size_t nactv_occ = actv_occ_mos_.size();
        size_t nactv_uocc = actv_uocc_mos_.size();

        if (internal_amp_select_ == "ALL") {
            for (size_t i = 0; i < nactv1; ++i) {
                for (size_t j = 0; j < nactv1; ++j) {
                    size_t c = i * nactv1 + j;

                    for (size_t a = 0; a < nactv1; ++a) {
                        for (size_t b = 0; b < nactv1; ++b) {
                            size_t v = a * nactv1 + b;

                            if (c >= v) {
                                size_t idx = i * nactv3 + j * nactv2 + a * nactv1 + b;
                                for (const std::string& block : {"aaaa", "aAaA", "AAAA"}) {
                                    T2_.block(block).data()[idx] = 0.0;
                                }
                            }
                        }
                    }
                }
            }
        } else if (internal_amp_select_ == "OOVV") {
            for (const std::string& block : {"aaaa", "aAaA", "AAAA"}) {
                // copy original data
                std::vector<double> data(T2_.block(block).data());

                T2_.block(block).zero();
                for (size_t I = 0; I < nactv_occ; ++I) {
                    for (size_t J = 0; J < nactv_occ; ++J) {
                        for (size_t A = 0; A < nactv_uocc; ++A) {
                            for (size_t B = 0; B < nactv_uocc; ++B) {
                                size_t idx = actv_occ_mos_[I] * nactv3 + actv_occ_mos_[J] * nactv2 +
                                             actv_uocc_mos_[A] * nactv1 + actv_uocc_mos_[B];
                                T2_.block(block).data()[idx] = data[idx];
                            }
                        }
                    }
                }
            }
        } else {
            for (const std::string& block : {"aaaa", "aAaA", "AAAA"}) {
                // copy original data
                std::vector<double> data(T2_.block(block).data());
                T2_.block(block).zero();

                // OO->VV
                for (size_t I = 0; I < nactv_occ; ++I) {
                    for (size_t J = 0; J < nactv_occ; ++J) {
                        for (size_t A = 0; A < nactv_uocc; ++A) {
                            for (size_t B = 0; B < nactv_uocc; ++B) {
                                size_t idx = actv_occ_mos_[I] * nactv3 + actv_occ_mos_[J] * nactv2 +
                                             actv_uocc_mos_[A] * nactv1 + actv_uocc_mos_[B];
                                T2_.block(block).data()[idx] = data[idx];
                            }
                        }
                    }
                }

                // OO->OV, OO->VO
                for (size_t I = 0; I < nactv_occ; ++I) {
                    for (size_t J = 0; J < nactv_occ; ++J) {
                        for (size_t K = 0; K < nactv_occ; ++K) {
                            for (size_t A = 0; A < nactv_uocc; ++A) {
                                size_t idx = actv_occ_mos_[I] * nactv3 + actv_occ_mos_[J] * nactv2 +
                                             actv_occ_mos_[K] * nactv1 + actv_uocc_mos_[A];
                                T2_.block(block).data()[idx] = data[idx];

                                idx = actv_occ_mos_[I] * nactv3 + actv_occ_mos_[J] * nactv2 +
                                      actv_uocc_mos_[A] * nactv1 + actv_occ_mos_[K];
                                T2_.block(block).data()[idx] = data[idx];
                            }
                        }
                    }
                }

                // OV->VV, VO->VV
                for (size_t I = 0; I < nactv_occ; ++I) {
                    for (size_t A = 0; A < nactv_uocc; ++A) {
                        for (size_t B = 0; B < nactv_uocc; ++B) {
                            for (size_t C = 0; C < nactv_uocc; ++C) {
                                size_t idx = actv_occ_mos_[I] * nactv3 +
                                             actv_uocc_mos_[A] * nactv2 +
                                             actv_uocc_mos_[B] * nactv1 + actv_uocc_mos_[C];
                                T2_.block(block).data()[idx] = data[idx];

                                idx = actv_uocc_mos_[A] * nactv3 + actv_occ_mos_[I] * nactv2 +
                                      actv_uocc_mos_[B] * nactv1 + actv_uocc_mos_[C];
                                T2_.block(block).data()[idx] = data[idx];
                            }
                        }
                    }
                }
            }
        }

    } else {
        T2_.block("aaaa").zero();
        T2_.block("aAaA").zero();
        T2_.block("AAAA").zero();
    }

    // This is used to print the tensor out for further analysis.
    // Only used as a test for some future tensor factorizations and other
    bool print_denom = options_.get_bool("PRINT_DENOM2");

    if (print_denom) {
        std::ofstream myfile;
        myfile.open("Deltaijab.txt");
        myfile << acore_mos_.size() + aactv_mos_.size() << " "
               << acore_mos_.size() + aactv_mos_.size() << " "
               << aactv_mos_.size() + avirt_mos_.size() << " "
               << aactv_mos_.size() + avirt_mos_.size() << " \n";
        T2_.iterate([&](const std::vector<size_t>& i, const std::vector<SpinType>& spin, double&) {
            if ((spin[0] == AlphaSpin) and (spin[1] == AlphaSpin)) {
                double D = dsrg_source_->compute_renormalized_denominator(Fa_[i[0]] + Fa_[i[1]] -
                                                                          Fa_[i[2]] - Fa_[i[3]]);
                D *= 1.0 +
                     dsrg_source_->compute_renormalized(Fa_[i[0]] + Fa_[i[1]] - Fa_[i[2]] -
                                                        Fa_[i[3]]);
                myfile << i[0] << " " << i[1] << " " << i[2] << " " << i[3] << " " << D << " \n";
            }
        });
    }

    outfile->Printf("  Done. Timing %15.6f s", timer.get());
}

void DSRG_MRPT2::compute_t1() {
    Timer timer;
    std::string str = "Computing T1 amplitudes";
    outfile->Printf("\n    %-40s ...", str.c_str());

    BlockedTensor temp = BTF_->build(tensor_type_, "temp", spin_cases({"aa"}));
    temp["xu"] = Gamma1_["xu"];
    temp["XU"] = Gamma1_["XU"];
    // transform to semi-canonical basis
    if (!semi_canonical_) {
        BlockedTensor tempG =
            ambit::BlockedTensor::build(tensor_type_, "Temp Gamma", spin_cases({"aa"}));
        tempG["uv"] = U_["ux"] * temp["xy"] * U_["vy"];
        tempG["UV"] = U_["UX"] * temp["XY"] * U_["VY"];
        temp["uv"] = tempG["uv"];
        temp["UV"] = tempG["UV"];
    }
    // scale by delta
    temp.iterate(
        [&](const std::vector<size_t>& i, const std::vector<SpinType>& spin, double& value) {
            if (spin[0] == AlphaSpin) {
                value *= Fa_[i[0]] - Fa_[i[1]];
            } else {
                value *= Fb_[i[0]] - Fb_[i[1]];
            }
        });
    // transform back to non-canonical basis
    if (!semi_canonical_) {
        BlockedTensor tempG =
            ambit::BlockedTensor::build(tensor_type_, "Temp Gamma", spin_cases({"aa"}));
        tempG["uv"] = U_["xu"] * temp["xy"] * U_["yv"];
        tempG["UV"] = U_["XU"] * temp["XY"] * U_["YV"];
        temp["uv"] = tempG["uv"];
        temp["UV"] = tempG["UV"];
    }

    T1_["ia"] = F_["ai"];
    T1_["ia"] += temp["xu"] * T2_["iuax"];
    T1_["ia"] += temp["XU"] * T2_["iUaX"];

    T1_["IA"] = F_["AI"];
    T1_["IA"] += temp["xu"] * T2_["uIxA"];
    T1_["IA"] += temp["XU"] * T2_["IUAX"];

    // transform to semi-canonical basis
    if (!semi_canonical_) {
        BlockedTensor tempT1 =
            ambit::BlockedTensor::build(tensor_type_, "Temp T1", spin_cases({"hp"}));
        tempT1["jb"] = U_["ji"] * T1_["ia"] * U_["ba"];
        tempT1["JB"] = U_["JI"] * T1_["IA"] * U_["BA"];
        T1_["ia"] = tempT1["ia"];
        T1_["IA"] = tempT1["IA"];
    }

    T1_.iterate(
        [&](const std::vector<size_t>& i, const std::vector<SpinType>& spin, double& value) {
            if (std::fabs(value) > 1.0e-12) {
                if (spin[0] == AlphaSpin) {
                    value *= dsrg_source_->compute_renormalized_denominator(Fa_[i[0]] - Fa_[i[1]]);
                } else {
                    value *= dsrg_source_->compute_renormalized_denominator(Fb_[i[0]] - Fb_[i[1]]);
                }
            } else {
                value = 0.0;
            }
        });

    // transform back to non-canonical basis
    if (!semi_canonical_) {
        BlockedTensor tempT1 =
            ambit::BlockedTensor::build(tensor_type_, "Temp T1", spin_cases({"hp"}));
        tempT1["jb"] = U_["ij"] * T1_["ia"] * U_["ab"];
        tempT1["JB"] = U_["IJ"] * T1_["IA"] * U_["AB"];
        T1_["ia"] = tempT1["ia"];
        T1_["IA"] = tempT1["IA"];
    }

    // internal amplitudes (A->A)
    std::string internal_amp = options_.get_str("INTERNAL_AMP");
    if (internal_amp.find("SINGLES") != std::string::npos) {
        size_t nactv = mo_space_info_->size("ACTIVE");

        // zero half internals to avoid double counting
        for (size_t i = 0; i < nactv; ++i) {
            for (size_t a = 0; a < nactv; ++a) {
                if (i >= a) {
                    size_t idx = i * nactv + a;
                    for (const std::string& block : {"aa", "AA"}) {
                        T1_.block(block).data()[idx] = 0.0;
                    }
                }
            }
        }

        if (internal_amp_select_ != "ALL") {
            size_t nactv_occ = actv_occ_mos_.size();
            size_t nactv_uocc = actv_uocc_mos_.size();

            // zero O->O internals
            for (size_t I = 0; I < nactv_occ; ++I) {
                for (size_t J = 0; J < nactv_occ; ++J) {
                    size_t idx = actv_occ_mos_[I] * nactv + actv_occ_mos_[J];
                    for (const std::string& block : {"aa", "AA"}) {
                        T1_.block(block).data()[idx] = 0.0;
                    }
                }
            }

            // zero V->V internals
            for (size_t A = 0; A < nactv_uocc; ++A) {
                for (size_t B = 0; B < nactv_uocc; ++B) {
                    size_t idx = actv_uocc_mos_[A] * nactv + actv_uocc_mos_[B];
                    for (const std::string& block : {"aa", "AA"}) {
                        T1_.block(block).data()[idx] = 0.0;
                    }
                }
            }
        }
    } else {
        T1_.block("AA").zero();
        T1_.block("aa").zero();
    }

    outfile->Printf("  Done. Timing %15.6f s", timer.get());
}

void DSRG_MRPT2::renormalize_V() {
    Timer timer;
    std::string str = "Renormalizing two-electron integrals";
    outfile->Printf("\n    %-40s ...", str.c_str());

    // transform to semi-canonical basis
    if (!semi_canonical_) {
        BlockedTensor tempV =
            ambit::BlockedTensor::build(tensor_type_, "Temp V", spin_cases({"pphh"}));
        tempV["cdij"] = U_["ca"] * U_["db"] * V_["abij"];
        tempV["cDiJ"] = U_["ca"] * U_["DB"] * V_["aBiJ"];
        tempV["CDIJ"] = U_["CA"] * U_["DB"] * V_["ABIJ"];
        V_["abkl"] = tempV["abij"] * U_["lj"] * U_["ki"];
        V_["aBkL"] = tempV["aBiJ"] * U_["LJ"] * U_["ki"];
        V_["ABKL"] = tempV["ABIJ"] * U_["LJ"] * U_["KI"];
    }

    V_.iterate([&](const std::vector<size_t>& i, const std::vector<SpinType>& spin, double& value) {
        if (std::fabs(value) > 1.0e-12) {
            if ((spin[0] == AlphaSpin) and (spin[1] == AlphaSpin)) {
                value *= 1.0 +
                         dsrg_source_->compute_renormalized(Fa_[i[0]] + Fa_[i[1]] - Fa_[i[2]] -
                                                            Fa_[i[3]]);
            } else if ((spin[0] == AlphaSpin) and (spin[1] == BetaSpin)) {
                value *= 1.0 +
                         dsrg_source_->compute_renormalized(Fa_[i[0]] + Fb_[i[1]] - Fa_[i[2]] -
                                                            Fb_[i[3]]);
            } else if ((spin[0] == BetaSpin) and (spin[1] == BetaSpin)) {
                value *= 1.0 +
                         dsrg_source_->compute_renormalized(Fb_[i[0]] + Fb_[i[1]] - Fb_[i[2]] -
                                                            Fb_[i[3]]);
            }
        } else {
            value = 0.0;
        }
    });

    // transform back to non-canonical basis
    if (!semi_canonical_) {
        BlockedTensor tempV =
            ambit::BlockedTensor::build(tensor_type_, "Temp V", spin_cases({"pphh"}));
        tempV["cdij"] = U_["ac"] * U_["bd"] * V_["abij"];
        tempV["cDiJ"] = U_["ac"] * U_["BD"] * V_["aBiJ"];
        tempV["CDIJ"] = U_["AC"] * U_["BD"] * V_["ABIJ"];
        V_["abkl"] = tempV["abij"] * U_["jl"] * U_["ik"];
        V_["aBkL"] = tempV["aBiJ"] * U_["JL"] * U_["ik"];
        V_["ABKL"] = tempV["ABIJ"] * U_["JL"] * U_["IK"];
    }

    outfile->Printf("  Done. Timing %15.6f s", timer.get());
}

void DSRG_MRPT2::renormalize_F() {
    Timer timer;
    std::string str = "Renormalizing Fock matrix elements";
    outfile->Printf("\n    %-40s ...", str.c_str());

    BlockedTensor temp = BTF_->build(tensor_type_, "temp", spin_cases({"aa"}));
    temp["xu"] = Gamma1_["xu"];
    temp["XU"] = Gamma1_["XU"];
    // transform to semi-canonical basis
    if (!semi_canonical_) {
        BlockedTensor tempG =
            ambit::BlockedTensor::build(tensor_type_, "Temp Gamma", spin_cases({"aa"}));
        tempG["uv"] = U_["ux"] * temp["xy"] * U_["vy"];
        tempG["UV"] = U_["UX"] * temp["XY"] * U_["VY"];
        temp["uv"] = tempG["uv"];
        temp["UV"] = tempG["UV"];
    }
    // scale by delta
    temp.iterate(
        [&](const std::vector<size_t>& i, const std::vector<SpinType>& spin, double& value) {
            if (spin[0] == AlphaSpin) {
                value *= Fa_[i[0]] - Fa_[i[1]];
            } else {
                value *= Fb_[i[0]] - Fb_[i[1]];
            }
        });
    // transform back to non-canonical basis
    if (!semi_canonical_) {
        BlockedTensor tempG =
            ambit::BlockedTensor::build(tensor_type_, "Temp Gamma", spin_cases({"aa"}));
        tempG["uv"] = U_["xu"] * temp["xy"] * U_["yv"];
        tempG["UV"] = U_["XU"] * temp["XY"] * U_["YV"];
        temp["uv"] = tempG["uv"];
        temp["UV"] = tempG["UV"];
    }

    BlockedTensor sum = ambit::BlockedTensor::build(tensor_type_, "Temp sum", spin_cases({"ph"}));
    sum["ai"] = F_["ai"];
    sum["ai"] += temp["xu"] * T2_["iuax"];
    sum["ai"] += temp["XU"] * T2_["iUaX"];

    sum["AI"] = F_["AI"];
    sum["AI"] += temp["xu"] * T2_["uIxA"];
    sum["AI"] += temp["XU"] * T2_["IUAX"];

    // transform to semi-canonical basis
    if (!semi_canonical_) {
        BlockedTensor tempF =
            ambit::BlockedTensor::build(tensor_type_, "Temp F", spin_cases({"ph"}));
        tempF["bj"] = U_["ba"] * sum["ai"] * U_["ji"];
        tempF["BJ"] = U_["BA"] * sum["AI"] * U_["JI"];
        sum["ai"] = tempF["ai"];
        sum["AI"] = tempF["AI"];
    }

    sum.iterate(
        [&](const std::vector<size_t>& i, const std::vector<SpinType>& spin, double& value) {
            if (std::fabs(value) > 1.0e-12) {
                if (spin[0] == AlphaSpin) {
                    value *= dsrg_source_->compute_renormalized(Fa_[i[0]] - Fa_[i[1]]);
                } else {
                    value *= dsrg_source_->compute_renormalized(Fb_[i[0]] - Fb_[i[1]]);
                }
            } else {
                value = 0.0;
            }
        });

    // transform back to non-canonical basis
    if (!semi_canonical_) {
        BlockedTensor tempF =
            ambit::BlockedTensor::build(tensor_type_, "Temp F", spin_cases({"ph"}));
        tempF["bj"] = U_["ab"] * sum["ai"] * U_["ij"];
        tempF["BJ"] = U_["AB"] * sum["AI"] * U_["IJ"];
        sum["ai"] = tempF["ai"];
        sum["AI"] = tempF["AI"];
    }

    // add to original Fock
    F_["ai"] += sum["ai"];
    F_["AI"] += sum["AI"];

    outfile->Printf("  Done. Timing %15.6f s", timer.get());
}

double DSRG_MRPT2::E_FT1() {
    Timer timer;
    std::string str = "Computing <[F, T1]>";
    outfile->Printf("\n    %-40s ...", str.c_str());

    double E = 0.0;
    E += F_["em"] * T1_["me"];
    E += F_["ex"] * T1_["ye"] * Gamma1_["xy"];
    E += F_["xm"] * T1_["my"] * Eta1_["yx"];

    E += F_["EM"] * T1_["ME"];
    E += F_["EX"] * T1_["YE"] * Gamma1_["XY"];
    E += F_["XM"] * T1_["MY"] * Eta1_["YX"];

    if (internal_amp_) {
        E += F_["xv"] * T1_["ux"] * Gamma1_["vu"];
        E -= F_["yu"] * T1_["ux"] * Gamma1_["xy"];

        E += F_["XV"] * T1_["UX"] * Gamma1_["VU"];
        E -= F_["YU"] * T1_["UX"] * Gamma1_["XY"];
    }

    outfile->Printf("  Done. Timing %15.6f s", timer.get());
    dsrg_time_.add("110", timer.get());
    return E;
}

double DSRG_MRPT2::E_VT1() {
    Timer timer;
    std::string str = "Computing <[V, T1]>";
    outfile->Printf("\n    %-40s ...", str.c_str());

    double E = 0.0;
    BlockedTensor temp = BTF_->build(tensor_type_, "temp", spin_cases({"aaaa"}), true);

    temp["uvxy"] += V_["evxy"] * T1_["ue"];
    temp["uvxy"] -= V_["uvmy"] * T1_["mx"];

    temp["UVXY"] += V_["EVXY"] * T1_["UE"];
    temp["UVXY"] -= V_["UVMY"] * T1_["MX"];

    temp["uVxY"] += V_["eVxY"] * T1_["ue"];
    temp["uVxY"] += V_["uExY"] * T1_["VE"];
    temp["uVxY"] -= V_["uVmY"] * T1_["mx"];
    temp["uVxY"] -= V_["uVxM"] * T1_["MY"];

    E += 0.5 * temp["uvxy"] * Lambda2_["xyuv"];
    E += 0.5 * temp["UVXY"] * Lambda2_["XYUV"];
    E += temp["uVxY"] * Lambda2_["xYuV"];

    if (internal_amp_) {
        temp.zero();

        temp["uvxy"] += V_["wvxy"] * T1_["uw"];
        temp["uvxy"] -= V_["uvwy"] * T1_["wx"];

        temp["UVXY"] += V_["WVXY"] * T1_["UW"];
        temp["UVXY"] -= V_["UVWY"] * T1_["WX"];

        temp["uVxY"] += V_["wVxY"] * T1_["uw"];
        temp["uVxY"] += V_["uWxY"] * T1_["VW"];
        temp["uVxY"] -= V_["uVwY"] * T1_["wx"];
        temp["uVxY"] -= V_["uVxW"] * T1_["WY"];

        E += 0.5 * temp["uvxy"] * Lambda2_["xyuv"];
        E += 0.5 * temp["UVXY"] * Lambda2_["XYUV"];
        E += temp["uVxY"] * Lambda2_["xYuV"];
    }

    outfile->Printf("  Done. Timing %15.6f s", timer.get());
    dsrg_time_.add("210", timer.get());
    return E;
}

double DSRG_MRPT2::E_FT2() {
    Timer timer;
    std::string str = "Computing <[F, T2]>";
    outfile->Printf("\n    %-40s ...", str.c_str());

    double E = 0.0;
    BlockedTensor temp = BTF_->build(tensor_type_, "temp", spin_cases({"aaaa"}), true);

    temp["uvxy"] += F_["ex"] * T2_["uvey"];
    temp["uvxy"] -= F_["vm"] * T2_["umxy"];

    temp["UVXY"] += F_["EX"] * T2_["UVEY"];
    temp["UVXY"] -= F_["VM"] * T2_["UMXY"];

    temp["uVxY"] += F_["ex"] * T2_["uVeY"];
    temp["uVxY"] += F_["EY"] * T2_["uVxE"];
    temp["uVxY"] -= F_["VM"] * T2_["uMxY"];
    temp["uVxY"] -= F_["um"] * T2_["mVxY"];

    E += 0.5 * temp["uvxy"] * Lambda2_["xyuv"];
    E += 0.5 * temp["UVXY"] * Lambda2_["XYUV"];
    E += temp["uVxY"] * Lambda2_["xYuV"];

    if (internal_amp_) {
        temp.zero();

        temp["uvxy"] += F_["wx"] * T2_["uvwy"];
        temp["uvxy"] -= F_["vw"] * T2_["uwxy"];

        temp["UVXY"] += F_["WX"] * T2_["UVWY"];
        temp["UVXY"] -= F_["VW"] * T2_["UWXY"];

        temp["uVxY"] += F_["wx"] * T2_["uVwY"];
        temp["uVxY"] += F_["WY"] * T2_["uVxW"];
        temp["uVxY"] -= F_["VW"] * T2_["uWxY"];
        temp["uVxY"] -= F_["uw"] * T2_["wVxY"];

        E += 0.5 * temp["uvxy"] * Lambda2_["xyuv"];
        E += 0.5 * temp["UVXY"] * Lambda2_["XYUV"];
        E += temp["uVxY"] * Lambda2_["xYuV"];
    }

    outfile->Printf("  Done. Timing %15.6f s", timer.get());
    dsrg_time_.add("120", timer.get());
    return E;
}

double DSRG_MRPT2::E_VT2_2() {
    Timer timer;
    std::string str = "Computing <[V, T2]> (C_2)^4";
    outfile->Printf("\n    %-40s ...", str.c_str());

    double E = 0.0;
    E += 0.25 * V_["efmn"] * T2_["mnef"];
    E += 0.25 * V_["EFMN"] * T2_["MNEF"];
    E += V_["eFmN"] * T2_["mNeF"];

    BlockedTensor temp = BTF_->build(tensor_type_, "temp", spin_cases({"aa"}), true);
    temp["vu"] += 0.5 * V_["efmu"] * T2_["mvef"];
    temp["vu"] += V_["fEuM"] * T2_["vMfE"];
    temp["VU"] += 0.5 * V_["EFMU"] * T2_["MVEF"];
    temp["VU"] += V_["eFmU"] * T2_["mVeF"];
    E += temp["vu"] * Gamma1_["uv"];
    E += temp["VU"] * Gamma1_["UV"];

    temp.zero();
    temp["vu"] += 0.5 * V_["vemn"] * T2_["mnue"];
    temp["vu"] += V_["vEmN"] * T2_["mNuE"];
    temp["VU"] += 0.5 * V_["VEMN"] * T2_["MNUE"];
    temp["VU"] += V_["eVnM"] * T2_["nMeU"];
    E += temp["vu"] * Eta1_["uv"];
    E += temp["VU"] * Eta1_["UV"];

    temp = BTF_->build(tensor_type_, "temp", spin_cases({"aaaa"}), true);
    temp["yvxu"] += V_["efxu"] * T2_["yvef"];
    temp["yVxU"] += V_["eFxU"] * T2_["yVeF"];
    temp["YVXU"] += V_["EFXU"] * T2_["YVEF"];
    E += 0.25 * temp["yvxu"] * Gamma1_["xy"] * Gamma1_["uv"];
    E += temp["yVxU"] * Gamma1_["UV"] * Gamma1_["xy"];
    E += 0.25 * temp["YVXU"] * Gamma1_["XY"] * Gamma1_["UV"];

    temp.zero();
    temp["vyux"] += V_["vymn"] * T2_["mnux"];
    temp["vYuX"] += V_["vYmN"] * T2_["mNuX"];
    temp["VYUX"] += V_["VYMN"] * T2_["MNUX"];
    E += 0.25 * temp["vyux"] * Eta1_["uv"] * Eta1_["xy"];
    E += temp["vYuX"] * Eta1_["uv"] * Eta1_["XY"];
    E += 0.25 * temp["VYUX"] * Eta1_["UV"] * Eta1_["XY"];

    temp.zero();
    temp["vyux"] += V_["vemx"] * T2_["myue"];
    temp["vyux"] += V_["vExM"] * T2_["yMuE"];
    temp["VYUX"] += V_["eVmX"] * T2_["mYeU"];
    temp["VYUX"] += V_["VEXM"] * T2_["YMUE"];
    E += temp["vyux"] * Gamma1_["xy"] * Eta1_["uv"];
    E += temp["VYUX"] * Gamma1_["XY"] * Eta1_["UV"];
    temp["yVxU"] = V_["eVxM"] * T2_["yMeU"];
    E += temp["yVxU"] * Gamma1_["xy"] * Eta1_["UV"];
    temp["vYuX"] = V_["vEmX"] * T2_["mYuE"];
    E += temp["vYuX"] * Gamma1_["XY"] * Eta1_["uv"];

    temp.zero();
    temp["yvxu"] += 0.5 * Gamma1_["wz"] * V_["vexw"] * T2_["yzue"];
    temp["yvxu"] += Gamma1_["WZ"] * V_["vExW"] * T2_["yZuE"];
    temp["yvxu"] += 0.5 * Eta1_["wz"] * T2_["myuw"] * V_["vzmx"];
    temp["yvxu"] += Eta1_["WZ"] * T2_["yMuW"] * V_["vZxM"];
    E += temp["yvxu"] * Gamma1_["xy"] * Eta1_["uv"];

    temp["YVXU"] += 0.5 * Gamma1_["WZ"] * V_["VEXW"] * T2_["YZUE"];
    temp["YVXU"] += Gamma1_["wz"] * V_["eVwX"] * T2_["zYeU"];
    temp["YVXU"] += 0.5 * Eta1_["WZ"] * T2_["MYUW"] * V_["VZMX"];
    temp["YVXU"] += Eta1_["wz"] * V_["zVmX"] * T2_["mYwU"];
    E += temp["YVXU"] * Gamma1_["XY"] * Eta1_["UV"];

    if (internal_amp_) {
        temp.zero();
        temp["uvxy"] += 0.25 * V_["uvwz"] * Gamma1_["wx"] * Gamma1_["zy"];
        temp["uVxY"] += V_["uVwZ"] * Gamma1_["wx"] * Gamma1_["ZY"];
        temp["UVXY"] += 0.25 * V_["UVWZ"] * Gamma1_["WX"] * Gamma1_["ZY"];

        temp["uvxy"] -= 0.25 * V_["wzxy"] * Gamma1_["uw"] * Gamma1_["vz"];
        temp["uVxY"] -= V_["wZxY"] * Gamma1_["uw"] * Gamma1_["VZ"];
        temp["UVXY"] -= 0.25 * V_["WZXY"] * Gamma1_["UW"] * Gamma1_["VZ"];

        temp["uvxy"] -= 0.5 * V_["u1wz"] * Gamma1_["v1"] * Gamma1_["wx"] * Gamma1_["zy"];
        temp["uVxY"] -= V_["u!wZ"] * Gamma1_["V!"] * Gamma1_["wx"] * Gamma1_["ZY"];
        temp["uVxY"] -= V_["1VwZ"] * Gamma1_["u1"] * Gamma1_["wx"] * Gamma1_["ZY"];
        temp["UVXY"] -= 0.5 * V_["U!WZ"] * Gamma1_["V!"] * Gamma1_["WX"] * Gamma1_["ZY"];

        temp["uvxy"] += 0.5 * V_["wzx1"] * Gamma1_["uw"] * Gamma1_["vz"] * Gamma1_["1y"];
        temp["uVxY"] += V_["wZx!"] * Gamma1_["uw"] * Gamma1_["VZ"] * Gamma1_["!Y"];
        temp["uVxY"] += V_["wZ1Y"] * Gamma1_["uw"] * Gamma1_["VZ"] * Gamma1_["1x"];
        temp["UVXY"] += 0.5 * V_["WZX!"] * Gamma1_["UW"] * Gamma1_["VZ"] * Gamma1_["!Y"];

        E += temp["uvxy"] * T2_["xyuv"];
        E += temp["uVxY"] * T2_["xYuV"];
        E += temp["UVXY"] * T2_["XYUV"];
    }

    outfile->Printf("  Done. Timing %15.6f s", timer.get());
    dsrg_time_.add("220", timer.get());
    return E;
}

double DSRG_MRPT2::E_VT2_4HH() {
    Timer timer;
    std::string str = "Computing <[V, T2]> C_4 (C_2)^2 HH";
    outfile->Printf("\n    %-40s ...", str.c_str());

    double E = 0.0;
    BlockedTensor temp = BTF_->build(tensor_type_, "temp", spin_cases({"aaaa"}), true);

    temp["uvxy"] += 0.125 * V_["uvmn"] * T2_["mnxy"];
    temp["uvxy"] += 0.25 * Gamma1_["wz"] * V_["uvmw"] * T2_["mzxy"];
    temp["uVxY"] += V_["uVmN"] * T2_["mNxY"];
    temp["uVxY"] += Gamma1_["wz"] * T2_["zMxY"] * V_["uVwM"];
    temp["uVxY"] += Gamma1_["WZ"] * V_["uVmW"] * T2_["mZxY"];
    temp["UVXY"] += 0.125 * V_["UVMN"] * T2_["MNXY"];
    temp["UVXY"] += 0.25 * Gamma1_["WZ"] * V_["UVMW"] * T2_["MZXY"];

    E += Lambda2_["xyuv"] * temp["uvxy"];
    E += Lambda2_["xYuV"] * temp["uVxY"];
    E += Lambda2_["XYUV"] * temp["UVXY"];

    if (internal_amp_) {
        temp.zero();
        temp["uvxy"] -= 0.125 * V_["uvwz"] * T2_["wzxy"];
        temp["uVxY"] -= V_["uVwZ"] * T2_["wZxY"];
        temp["UVXY"] -= 0.125 * V_["UVWZ"] * T2_["WZXY"];

        temp["uvxy"] += 0.25 * V_["uv1w"] * T2_["1zxy"] * Gamma1_["wz"];
        temp["uVxY"] += V_["uV1W"] * T2_["1ZxY"] * Gamma1_["WZ"];
        temp["uVxY"] += V_["uVw!"] * T2_["z!xY"] * Gamma1_["wz"];
        temp["UVXY"] += 0.25 * V_["UV!W"] * T2_["!ZXY"] * Gamma1_["WZ"];

        E += Lambda2_["xyuv"] * temp["uvxy"];
        E += Lambda2_["XYUV"] * temp["UVXY"];
        E += Lambda2_["xYuV"] * temp["uVxY"];
    }

    outfile->Printf("  Done. Timing %15.6f s", timer.get());
    dsrg_time_.add("220", timer.get());
    return E;
}

double DSRG_MRPT2::E_VT2_4PP() {
    Timer timer;
    std::string str = "Computing <[V, T2]> C_4 (C_2)^2 PP";
    outfile->Printf("\n    %-40s ...", str.c_str());

    double E = 0.0;
    BlockedTensor temp = BTF_->build(tensor_type_, "temp", spin_cases({"aaaa"}), true);

    temp["uvxy"] += 0.125 * V_["efxy"] * T2_["uvef"];
    temp["uvxy"] += 0.25 * Eta1_["wz"] * T2_["uvew"] * V_["ezxy"];
    temp["uVxY"] += V_["eFxY"] * T2_["uVeF"];
    temp["uVxY"] += Eta1_["wz"] * V_["zExY"] * T2_["uVwE"];
    temp["uVxY"] += Eta1_["WZ"] * T2_["uVeW"] * V_["eZxY"];
    temp["UVXY"] += 0.125 * V_["EFXY"] * T2_["UVEF"];
    temp["UVXY"] += 0.25 * Eta1_["WZ"] * T2_["UVEW"] * V_["EZXY"];

    E += Lambda2_["xyuv"] * temp["uvxy"];
    E += Lambda2_["xYuV"] * temp["uVxY"];
    E += Lambda2_["XYUV"] * temp["UVXY"];

    if (internal_amp_) {
        temp.zero();
        temp["uvxy"] += 0.125 * V_["wzxy"] * T2_["uvwz"];
        temp["uVxY"] += V_["wZxY"] * T2_["uVwZ"];
        temp["UVXY"] += 0.125 * V_["WZXY"] * T2_["UVWZ"];

        temp["uvxy"] -= 0.25 * V_["1zxy"] * T2_["uv1w"] * Gamma1_["wz"];
        temp["uVxY"] -= V_["1ZxY"] * T2_["uV1W"] * Gamma1_["WZ"];
        temp["uVxY"] -= V_["z!xY"] * T2_["uVw!"] * Gamma1_["wz"];
        temp["UVXY"] -= 0.25 * V_["!ZXY"] * T2_["UV!W"] * Gamma1_["WZ"];

        E += Lambda2_["xyuv"] * temp["uvxy"];
        E += Lambda2_["xYuV"] * temp["uVxY"];
        E += Lambda2_["XYUV"] * temp["UVXY"];
    }

    outfile->Printf("  Done. Timing %15.6f s", timer.get());
    dsrg_time_.add("220", timer.get());
    return E;
}

double DSRG_MRPT2::E_VT2_4PH() {
    Timer timer;
    std::string str = "Computing <[V, T2]> C_4 (C_2)^2 PH";
    outfile->Printf("\n    %-40s ...", str.c_str());

    double E = 0.0;
    BlockedTensor temp = BTF_->build(tensor_type_, "temp", spin_cases({"aaaa"}), true);

    temp["uvxy"] += V_["eumx"] * T2_["mvey"];
    temp["uvxy"] += V_["uExM"] * T2_["vMyE"];
    temp["uvxy"] += Gamma1_["wz"] * T2_["zvey"] * V_["euwx"];
    temp["uvxy"] += Gamma1_["WZ"] * V_["uExW"] * T2_["vZyE"];
    temp["uvxy"] += Eta1_["zw"] * V_["wumx"] * T2_["mvzy"];
    temp["uvxy"] += Eta1_["ZW"] * T2_["vMyZ"] * V_["uWxM"];
    E += temp["uvxy"] * Lambda2_["xyuv"];

    temp["UVXY"] += V_["eUmX"] * T2_["mVeY"];
    temp["UVXY"] += V_["EUMX"] * T2_["MVEY"];
    temp["UVXY"] += Gamma1_["wz"] * T2_["zVeY"] * V_["eUwX"];
    temp["UVXY"] += Gamma1_["WZ"] * T2_["ZVEY"] * V_["EUWX"];
    temp["UVXY"] += Eta1_["zw"] * V_["wUmX"] * T2_["mVzY"];
    temp["UVXY"] += Eta1_["ZW"] * V_["WUMX"] * T2_["MVZY"];
    E += temp["UVXY"] * Lambda2_["XYUV"];

    temp["uVxY"] += V_["uexm"] * T2_["mVeY"];
    temp["uVxY"] += V_["uExM"] * T2_["MVEY"];
    temp["uVxY"] -= V_["eVxM"] * T2_["uMeY"];
    temp["uVxY"] -= V_["uEmY"] * T2_["mVxE"];
    temp["uVxY"] += V_["eVmY"] * T2_["umxe"];
    temp["uVxY"] += V_["EVMY"] * T2_["uMxE"];

    temp["uVxY"] += Gamma1_["wz"] * T2_["zVeY"] * V_["uexw"];
    temp["uVxY"] += Gamma1_["WZ"] * T2_["ZVEY"] * V_["uExW"];
    temp["uVxY"] -= Gamma1_["WZ"] * V_["eVxW"] * T2_["uZeY"];
    temp["uVxY"] -= Gamma1_["wz"] * T2_["zVxE"] * V_["uEwY"];
    temp["uVxY"] += Gamma1_["wz"] * T2_["zuex"] * V_["eVwY"];
    temp["uVxY"] -= Gamma1_["WZ"] * V_["EVYW"] * T2_["uZxE"];

    temp["uVxY"] += Eta1_["zw"] * V_["wumx"] * T2_["mVzY"];
    temp["uVxY"] += Eta1_["ZW"] * T2_["VMYZ"] * V_["uWxM"];
    temp["uVxY"] -= Eta1_["zw"] * V_["wVxM"] * T2_["uMzY"];
    temp["uVxY"] -= Eta1_["ZW"] * T2_["mVxZ"] * V_["uWmY"];
    temp["uVxY"] += Eta1_["zw"] * T2_["umxz"] * V_["wVmY"];
    temp["uVxY"] += Eta1_["ZW"] * V_["WVMY"] * T2_["uMxZ"];
    E += temp["uVxY"] * Lambda2_["xYuV"];

    if (internal_amp_) {
        temp.zero();
        temp["uvxy"] -= V_["v1xw"] * T2_["zu1y"] * Gamma1_["wz"];
        temp["uvxy"] -= V_["v!xW"] * T2_["uZy!"] * Gamma1_["WZ"];
        temp["uvxy"] += V_["vzx1"] * T2_["1uwy"] * Gamma1_["wz"];
        temp["uvxy"] += V_["vZx!"] * T2_["u!yW"] * Gamma1_["WZ"];
        E += temp["uvxy"] * Lambda2_["xyuv"];

        temp["UVXY"] -= V_["V!XW"] * T2_["ZU!Y"] * Gamma1_["WZ"];
        temp["UVXY"] -= V_["1VwX"] * T2_["zU1Y"] * Gamma1_["wz"];
        temp["UVXY"] += V_["VZX!"] * T2_["!UWY"] * Gamma1_["WZ"];
        temp["UVXY"] += V_["zV1X"] * T2_["1UwY"] * Gamma1_["wz"];
        E += temp["UVXY"] * Lambda2_["XYUV"];

        temp["uVxY"] -= V_["1VxW"] * T2_["uZ1Y"] * Gamma1_["WZ"];
        temp["uVxY"] -= V_["u!wY"] * T2_["zVx!"] * Gamma1_["wz"];
        temp["uVxY"] += V_["u1xw"] * T2_["zV1Y"] * Gamma1_["wz"];
        temp["uVxY"] += V_["u!xW"] * T2_["ZV!Y"] * Gamma1_["WZ"];
        temp["uVxY"] += V_["1VwY"] * T2_["zu1x"] * Gamma1_["wz"];
        temp["uVxY"] += V_["!VWY"] * T2_["uZx!"] * Gamma1_["WZ"];

        temp["uVxY"] += V_["zVx!"] * T2_["u!wY"] * Gamma1_["wz"];
        temp["uVxY"] += V_["uZ1Y"] * T2_["1VxW"] * Gamma1_["WZ"];
        temp["uVxY"] -= V_["uzx1"] * T2_["1VwY"] * Gamma1_["wz"];
        temp["uVxY"] -= V_["uZx!"] * T2_["!VWY"] * Gamma1_["WZ"];
        temp["uVxY"] -= V_["zV1Y"] * T2_["1uwx"] * Gamma1_["wz"];
        temp["uVxY"] -= V_["ZV!Y"] * T2_["u!xW"] * Gamma1_["WZ"];
        E += temp["uVxY"] * Lambda2_["xYuV"];
    }

    outfile->Printf("  Done. Timing %15.6f s", timer.get());
    dsrg_time_.add("220", timer.get());
    return E;
}

double DSRG_MRPT2::E_VT2_6() {
    Timer timer;
    std::string str = "Computing <[V, T2]> C_6 C_2";
    outfile->Printf("\n    %-40s ...", str.c_str());

    double E = 0.0;

    // aaa
    BlockedTensor temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"aaaaaa"});
    temp["uvwxyz"] += V_["uvmz"] * T2_["mwxy"];
    temp["uvwxyz"] += V_["wexy"] * T2_["uvez"];

    if (internal_amp_) {
        temp["uvwxyz"] += V_["uv1z"] * T2_["1wxy"];
        temp["uvwxyz"] += V_["w1xy"] * T2_["uv1z"];
    }
    E += 0.25 * temp["uvwxyz"] * Lambda3_["xyzuvw"];

    // bbb
    temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"AAAAAA"});
    temp["UVWXYZ"] += V_["UVMZ"] * T2_["MWXY"];
    temp["UVWXYZ"] += V_["WEXY"] * T2_["UVEZ"];

    if (internal_amp_) {
        temp["UVWXYZ"] += V_["UV!Z"] * T2_["!WXY"];
        temp["UVWXYZ"] += V_["W!XY"] * T2_["UV!Z"];
    }
    E += 0.25 * temp["UVWXYZ"] * Lambda3_["XYZUVW"];

    // aab
    temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"aaAaaA"});
    temp["uvWxyZ"] -= V_["uvmy"] * T2_["mWxZ"];
    temp["uvWxyZ"] -= V_["uWmZ"] * T2_["mvxy"];
    temp["uvWxyZ"] += 2.0 * V_["uWyM"] * T2_["vMxZ"];

    temp["uvWxyZ"] += V_["eWxZ"] * T2_["uvey"];
    temp["uvWxyZ"] -= V_["vexy"] * T2_["uWeZ"];
    temp["uvWxyZ"] -= 2.0 * V_["vExZ"] * T2_["uWyE"];

    if (internal_amp_) {
        temp["uvWxyZ"] -= V_["uv1y"] * T2_["1WxZ"];
        temp["uvWxyZ"] -= V_["uW1Z"] * T2_["1vxy"];
        temp["uvWxyZ"] += 2.0 * V_["uWy!"] * T2_["v!xZ"];

        temp["uvWxyZ"] += V_["1WxZ"] * T2_["uv1y"];
        temp["uvWxyZ"] -= V_["v1xy"] * T2_["uW1Z"];
        temp["uvWxyZ"] -= 2.0 * V_["v!xZ"] * T2_["uWy!"];
    }
    E += 0.50 * temp["uvWxyZ"] * Lambda3_["xyZuvW"];

    // abb
    temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"aAAaAA"});
    temp["uVWxYZ"] -= V_["VWMZ"] * T2_["uMxY"];
    temp["uVWxYZ"] -= V_["uVxM"] * T2_["MWYZ"];
    temp["uVWxYZ"] += 2.0 * V_["uVmZ"] * T2_["mWxY"];

    temp["uVWxYZ"] += V_["uExY"] * T2_["VWEZ"];
    temp["uVWxYZ"] -= V_["WEYZ"] * T2_["uVxE"];
    temp["uVWxYZ"] -= 2.0 * V_["eWxY"] * T2_["uVeZ"];

    if (internal_amp_) {
        temp["uVWxYZ"] -= V_["VW!Z"] * T2_["u!xY"];
        temp["uVWxYZ"] -= V_["uVx!"] * T2_["!WYZ"];
        temp["uVWxYZ"] += 2.0 * V_["uV1Z"] * T2_["1WxY"];

        temp["uVWxYZ"] += V_["u!xY"] * T2_["VW!Z"];
        temp["uVWxYZ"] -= V_["W!YZ"] * T2_["uVx!"];
        temp["uVWxYZ"] -= 2.0 * V_["1WxY"] * T2_["uV1Z"];
    }
    E += 0.5 * temp["uVWxYZ"] * Lambda3_["xYZuVW"];

    outfile->Printf("  Done. Timing %15.6f s", timer.get());
    dsrg_time_.add("220", timer.get());
    return E;
}

double DSRG_MRPT2::compute_energy_relaxed() {

    // reference relaxation
    double Edsrg = 0.0, Erelax = 0.0;

    // compute energy with fixed ref.
    Edsrg = compute_energy();

    // transfer integrals
    transfer_integrals();

    // diagonalize Hbar depending on CAS_TYPE
    if (options_.get_str("CAS_TYPE") == "CAS") {

        FCI_MO fci_mo(reference_wavefunction_, options_, ints_, mo_space_info_);
        fci_mo.set_form_Fock(false);
        Erelax = fci_mo.compute_energy();

    } else {

        // it is simpler here to call FCI instead of FCISolver
        FCI fci(reference_wavefunction_, options_, ints_, mo_space_info_);
        fci.set_max_rdm_level(1);
        Erelax = fci.compute_energy();

        //        // setup for FCISolver
        //        Dimension active_dim = mo_space_info_->get_dimension("ACTIVE");
        //        int charge = Process::environment.molecule()->molecular_charge();
        //        if (options_["CHARGE"].has_changed()) {
        //            charge = options_.get_int("CHARGE");
        //        }
        //        auto nelec = 0;
        //        int natom = Process::environment.molecule()->natom();
        //        for (int i = 0; i < natom; ++i) {
        //            nelec += Process::environment.molecule()->fZ(i);
        //        }
        //        nelec -= charge;
        //        int multi = Process::environment.molecule()->multiplicity();
        //        if (options_["MULTIPLICITY"].has_changed()) {
        //            multi = options_.get_int("MULTIPLICITY");
        //        }
        //        int twice_ms = (multi + 1) % 2;
        //        if (options_["MS"].has_changed()) {
        //            twice_ms = std::round(2.0 * options_.get_double("MS"));
        //        }
        //        auto nelec_actv =
        //            nelec - 2 * mo_space_info_->size("FROZEN_DOCC") - 2 *
        //            acore_mos_.size();
        //        auto na = (nelec_actv + twice_ms) / 2;
        //        auto nb = nelec_actv - na;

        //        // diagonalize the Hamiltonian
        //        FCISolver fcisolver(active_dim, acore_mos_, aactv_mos_, na,
        //                            nb, multi,
        //                            options_.get_int("ROOT_SYM"), ints_,
        //                            mo_space_info_,
        //                            options_.get_int("NTRIAL_PER_ROOT"),
        //                            print_,
        //                            options_);
        //        fcisolver.set_max_rdm_level(1);
        //        fcisolver.set_nroot(options_.get_int("FCI_NROOT"));
        //        fcisolver.set_root(options_.get_int("FCI_ROOT"));
        //        fcisolver.set_fci_iterations(options_.get_int("FCI_MAXITER"));
        //        fcisolver.set_collapse_per_root(
        //                    options_.get_int("DL_COLLAPSE_PER_ROOT"));
        //        fcisolver.set_subspace_per_root(
        //                    options_.get_int("DL_SUBSPACE_PER_ROOT"));
        //        Erelax = fcisolver.compute_energy();
    }

    // printing
    print_h2("DSRG-MRPT2 Energy Summary");
    outfile->Printf("\n    %-30s = %22.15f", "DSRG-MRPT2 Total Energy (fixed)  ", Edsrg);
    outfile->Printf("\n    %-30s = %22.15f", "DSRG-MRPT2 Total Energy (relaxed)", Erelax);
    outfile->Printf("\n");

    Process::environment.globals["CURRENT ENERGY"] = Erelax;
    return Erelax;
}

void DSRG_MRPT2::transfer_integrals() {
    // printing
    print_h2("De-Normal-Order the DSRG Transformed Hamiltonian");

    // compute scalar term (all active only)
    Timer t_scalar;
    std::string str = "Computing the scalar term   ...";
    outfile->Printf("\n    %-35s", str.c_str());
    double scalar0 =
        Eref_ + Hbar0_ - molecule_->nuclear_repulsion_energy() - ints_->frozen_core_energy();

    // scalar from Hbar1
    double scalar1 = 0.0;
    scalar1 -= Hbar1_["vu"] * Gamma1_["uv"];
    scalar1 -= Hbar1_["VU"] * Gamma1_["UV"];

    // scalar from Hbar2
    double scalar2 = 0.0;
    scalar2 += 0.5 * Gamma1_["uv"] * Hbar2_["vyux"] * Gamma1_["xy"];
    scalar2 += 0.5 * Gamma1_["UV"] * Hbar2_["VYUX"] * Gamma1_["XY"];
    scalar2 += Gamma1_["uv"] * Hbar2_["vYuX"] * Gamma1_["XY"];

    scalar2 -= 0.25 * Hbar2_["xyuv"] * Lambda2_["uvxy"];
    scalar2 -= 0.25 * Hbar2_["XYUV"] * Lambda2_["UVXY"];
    scalar2 -= Hbar2_["xYuV"] * Lambda2_["uVxY"];

    double scalar = scalar0 + scalar1 + scalar2;

    bool form_hbar3 = options_.get_bool("FORM_HBAR3");
    double scalar3 = 0.0;
    if (form_hbar3) {
        scalar3 -= (1.0 / 36) * Hbar3_["xyzuvw"] * Lambda3_["uvwxyz"];
        scalar3 -= (1.0 / 36) * Hbar3_["XYZUVW"] * Lambda3_["UVWXYZ"];
        scalar3 -= 0.25 * Hbar3_["xyZuvW"] * Lambda3_["uvWxyZ"];
        scalar3 -= 0.25 * Hbar3_["xYZuVW"] * Lambda3_["uVWxYZ"];

        scalar3 += 0.25 * Hbar3_["xyzuvw"] * Gamma1_["wz"] * Lambda2_["uvxy"];
        scalar3 += 0.25 * Hbar3_["XYZUVW"] * Gamma1_["WZ"] * Lambda2_["UVXY"];
        scalar3 += 0.25 * Hbar3_["xyZuvW"] * Gamma1_["WZ"] * Lambda2_["uvxy"];
        scalar3 += Hbar3_["xzYuwV"] * Gamma1_["wz"] * Lambda2_["uVxY"];
        scalar3 += 0.25 * Hbar3_["zXYwUV"] * Gamma1_["wz"] * Lambda2_["UVXY"];
        scalar3 += Hbar3_["xZYuWV"] * Gamma1_["WZ"] * Lambda2_["uVxY"];

        scalar3 -= (1.0 / 6) * Hbar3_["xyzuvw"] * Gamma1_["ux"] * Gamma1_["vy"] * Gamma1_["wz"];
        scalar3 -= (1.0 / 6) * Hbar3_["XYZUVW"] * Gamma1_["UX"] * Gamma1_["VY"] * Gamma1_["WZ"];
        scalar3 -= 0.5 * Hbar3_["xyZuvW"] * Gamma1_["ux"] * Gamma1_["vy"] * Gamma1_["WZ"];
        scalar3 -= 0.5 * Hbar3_["xYZuVW"] * Gamma1_["ux"] * Gamma1_["VY"] * Gamma1_["WZ"];

        scalar += scalar3;
    }

    outfile->Printf("  Done. Timing %10.3f s", t_scalar.get());

    // compute one-body term
    Timer t_one;
    str = "Computing the one-body term ...";
    outfile->Printf("\n    %-35s", str.c_str());
    BlockedTensor temp1 = BTF_->build(tensor_type_, "temp1", spin_cases({"aa"}));
    temp1["uv"] = Hbar1_["uv"];
    temp1["UV"] = Hbar1_["UV"];
    temp1["uv"] -= Hbar2_["uxvy"] * Gamma1_["yx"];
    temp1["uv"] -= Hbar2_["uXvY"] * Gamma1_["YX"];
    temp1["UV"] -= Hbar2_["xUyV"] * Gamma1_["yx"];
    temp1["UV"] -= Hbar2_["UXVY"] * Gamma1_["YX"];

    if (form_hbar3) {
        temp1["uv"] += 0.5 * Hbar3_["uyzvxw"] * Gamma1_["xy"] * Gamma1_["wz"];
        temp1["uv"] += 0.5 * Hbar3_["uYZvXW"] * Gamma1_["XY"] * Gamma1_["WZ"];
        temp1["uv"] += Hbar3_["uyZvxW"] * Gamma1_["xy"] * Gamma1_["WZ"];

        temp1["UV"] += 0.5 * Hbar3_["UYZVXW"] * Gamma1_["XY"] * Gamma1_["WZ"];
        temp1["UV"] += 0.5 * Hbar3_["yzUxwV"] * Gamma1_["xy"] * Gamma1_["wz"];
        temp1["UV"] += Hbar3_["yUZxVW"] * Gamma1_["xy"] * Gamma1_["WZ"];

        temp1["uv"] -= 0.25 * Hbar3_["uxyvwz"] * Lambda2_["wzxy"];
        temp1["uv"] -= 0.25 * Hbar3_["uXYvWZ"] * Lambda2_["WZXY"];
        temp1["uv"] -= Hbar3_["uxYvwZ"] * Lambda2_["wZxY"];

        temp1["UV"] -= 0.25 * Hbar3_["UXYVWZ"] * Lambda2_["WZXY"];
        temp1["UV"] -= 0.25 * Hbar3_["xyUwzV"] * Lambda2_["wzxy"];
        temp1["UV"] -= Hbar3_["xUYwVZ"] * Lambda2_["wZxY"];
    }

    outfile->Printf("  Done. Timing %10.3f s", t_one.get());

    // compute two-body term
    BlockedTensor temp2;
    if (form_hbar3) {
        temp2 = BTF_->build(tensor_type_, "temp2", spin_cases({"aaaa"}));
        str = "Computing the two-body term ...";
        outfile->Printf("\n    %-35s", str.c_str());

        temp2["uvxy"] = Hbar2_["uvxy"];
        temp2["uVxY"] = Hbar2_["uVxY"];
        temp2["UVXY"] = Hbar2_["UVXY"];

        temp2["xyuv"] -= Hbar3_["xyzuvw"] * Gamma1_["wz"];
        temp2["xyuv"] -= Hbar3_["xyZuvW"] * Gamma1_["WZ"];
        temp2["xYuV"] -= Hbar3_["xYZuVW"] * Gamma1_["WZ"];
        temp2["xYuV"] -= Hbar3_["xzYuwV"] * Gamma1_["wz"];
        temp2["XYUV"] -= Hbar3_["XYZUVW"] * Gamma1_["WZ"];
        temp2["XYUV"] -= Hbar3_["zXYwUV"] * Gamma1_["wz"];

        outfile->Printf("  Done. Timing %10.3f s", t_one.get());
    }

    // update integrals
    Timer t_int;
    str = "Updating integrals          ...";
    outfile->Printf("\n    %-35s", str.c_str());
    ints_->set_scalar(scalar);

    //   a) zero hole integrals
    std::vector<size_t> hole_mos = acore_mos_;
    hole_mos.insert(hole_mos.end(), aactv_mos_.begin(), aactv_mos_.end());
    for (const size_t& i : hole_mos) {
        for (const size_t& j : hole_mos) {
            ints_->set_oei(i, j, 0.0, true);
            ints_->set_oei(i, j, 0.0, false);
            for (const size_t& k : hole_mos) {
                for (const size_t& l : hole_mos) {
                    ints_->set_tei(i, j, k, l, 0.0, true, true);
                    ints_->set_tei(i, j, k, l, 0.0, true, false);
                    ints_->set_tei(i, j, k, l, 0.0, false, false);
                }
            }
        }
    }

    //   b) copy all active part
    temp1.citerate(
        [&](const std::vector<size_t>& i, const std::vector<SpinType>& spin, const double& value) {
            if (spin[0] == AlphaSpin) {
                ints_->set_oei(i[0], i[1], value, true);
            } else {
                ints_->set_oei(i[0], i[1], value, false);
            }
        });

    if (!form_hbar3) {
        Hbar2_.citerate([&](const std::vector<size_t>& i, const std::vector<SpinType>& spin,
                            const double& value) {
            if ((spin[0] == AlphaSpin) && (spin[1] == AlphaSpin)) {
                ints_->set_tei(i[0], i[1], i[2], i[3], value, true, true);
            } else if ((spin[0] == AlphaSpin) && (spin[1] == BetaSpin)) {
                ints_->set_tei(i[0], i[1], i[2], i[3], value, true, false);
            } else if ((spin[0] == BetaSpin) && (spin[1] == BetaSpin)) {
                ints_->set_tei(i[0], i[1], i[2], i[3], value, false, false);
            }
        });
    } else {
        temp2.citerate([&](const std::vector<size_t>& i, const std::vector<SpinType>& spin,
                           const double& value) {
            if ((spin[0] == AlphaSpin) && (spin[1] == AlphaSpin)) {
                ints_->set_tei(i[0], i[1], i[2], i[3], value, true, true);
            } else if ((spin[0] == AlphaSpin) && (spin[1] == BetaSpin)) {
                ints_->set_tei(i[0], i[1], i[2], i[3], value, true, false);
            } else if ((spin[0] == BetaSpin) && (spin[1] == BetaSpin)) {
                ints_->set_tei(i[0], i[1], i[2], i[3], value, false, false);
            }
        });
    }

    outfile->Printf("  Done. Timing %10.3f s", t_int.get());

    // print scalar
    double scalar_include_fc = scalar + ints_->frozen_core_energy();
    print_h2("Scalar of the DSRG Hamiltonian (WRT True Vacuum)");
    outfile->Printf("\n    %-30s = %22.15f", "Scalar0", scalar0);
    outfile->Printf("\n    %-30s = %22.15f", "Scalar1", scalar1);
    outfile->Printf("\n    %-30s = %22.15f", "Scalar2", scalar2);
    if (form_hbar3) {
        outfile->Printf("\n    %-30s = %22.15f", "Scalar3", scalar3);
    }
    outfile->Printf("\n    %-30s = %22.15f", "Total Scalar W/O Frozen-Core", scalar);
    outfile->Printf("\n    %-30s = %22.15f", "Total Scalar W/  Frozen-Core", scalar_include_fc);

    // test if de-normal-ordering is correct
    print_h2("Test De-Normal-Ordered Hamiltonian");
    double Etest = scalar_include_fc + molecule_->nuclear_repulsion_energy();

    double Etest1 = 0.0;
    if (!form_hbar3) {
        Etest1 += temp1["uv"] * Gamma1_["vu"];
        Etest1 += temp1["UV"] * Gamma1_["VU"];

        Etest1 += Hbar1_["uv"] * Gamma1_["vu"];
        Etest1 += Hbar1_["UV"] * Gamma1_["VU"];
        Etest1 *= 0.5;
    } else {
        Etest1 += temp1["uv"] * Gamma1_["vu"];
        Etest1 += temp1["UV"] * Gamma1_["VU"];
    }

    double Etest2 = 0.0;
    Etest2 += 0.25 * Hbar2_["uvxy"] * Lambda2_["xyuv"];
    Etest2 += 0.25 * Hbar2_["UVXY"] * Lambda2_["XYUV"];
    Etest2 += Hbar2_["uVxY"] * Lambda2_["xYuV"];

    if (form_hbar3) {
        Etest2 += 0.5 * temp2["xyuv"] * Gamma1_["ux"] * Gamma1_["vy"];
        Etest2 += 0.5 * temp2["XYUV"] * Gamma1_["UX"] * Gamma1_["VY"];
        Etest2 += temp2["xYuV"] * Gamma1_["ux"] * Gamma1_["VY"];
    }

    Etest += Etest1 + Etest2;
    outfile->Printf("\n    %-30s = %22.15f", "One-Body Energy (after)", Etest1);
    outfile->Printf("\n    %-30s = %22.15f", "Two-Body Energy (after)", Etest2);

    if (form_hbar3) {
        double Etest3 = 0.0;
        Etest3 += (1.0 / 6) * Hbar3_["xyzuvw"] * Gamma1_["ux"] * Gamma1_["vy"] * Gamma1_["wz"];
        Etest3 += (1.0 / 6) * Hbar3_["XYZUVW"] * Gamma1_["UX"] * Gamma1_["VY"] * Gamma1_["WZ"];
        Etest3 += 0.5 * Hbar3_["xyZuvW"] * Gamma1_["ux"] * Gamma1_["vy"] * Gamma1_["WZ"];
        Etest3 += 0.5 * Hbar3_["xYZuVW"] * Gamma1_["ux"] * Gamma1_["VY"] * Gamma1_["WZ"];

        Etest3 += (1.0 / 36) * Hbar3_["xyzuvw"] * Lambda3_["uvwxyz"];
        Etest3 += (1.0 / 36) * Hbar3_["XYZUVW"] * Lambda3_["UVWXYZ"];
        Etest3 += 0.25 * Hbar3_["xyZuvW"] * Lambda3_["uvWxyZ"];
        Etest3 += 0.25 * Hbar3_["xYZuVW"] * Lambda3_["uVWxYZ"];

        outfile->Printf("\n    %-30s = %22.15f", "Three-Body Energy (after)", Etest3);
        Etest += Etest3;
    }

    outfile->Printf("\n    %-30s = %22.15f", "Total Energy (after)", Etest);
    outfile->Printf("\n    %-30s = %22.15f", "Total Energy (before)", Eref_ + Hbar0_);

    if (std::fabs(Etest - Eref_ - Hbar0_) > 100.0 * options_.get_double("E_CONVERGENCE")) {
        throw PSIEXCEPTION("De-normal-odering failed.");
    } else {
        ints_->update_integrals(false);
    }
}

void DSRG_MRPT2::H1_T1_C1(BlockedTensor& H1, BlockedTensor& T1, const double& alpha,
                          BlockedTensor& C1) {
    Timer timer;

    C1["ij"] += alpha * H1["aj"] * T1["ia"];
    C1["ju"] -= alpha * T1["iu"] * H1["ji"];

    C1["IJ"] += alpha * H1["AJ"] * T1["IA"];
    C1["JU"] -= alpha * T1["IU"] * H1["JI"];

    if (print_ > 2) {
        outfile->Printf("\n    Time for [H1, T1] -> C1 : %12.3f", timer.get());
    }
    dsrg_time_.add("111", timer.get());
}

void DSRG_MRPT2::H1_T2_C1(BlockedTensor& H1, BlockedTensor& T2, const double& alpha,
                          BlockedTensor& C1) {
    Timer timer;

    C1["iu"] += alpha * H1["bm"] * T2["imub"];
    C1["ix"] += alpha * H1["bu"] * T2["ivxb"] * Gamma1_["uv"];
    C1["ix"] -= alpha * H1["vj"] * T2["ijxu"] * Gamma1_["uv"];
    C1["iu"] += alpha * H1["BM"] * T2["iMuB"];
    C1["ix"] += alpha * H1["BU"] * T2["iVxB"] * Gamma1_["UV"];
    C1["ix"] -= alpha * H1["VJ"] * T2["iJxU"] * Gamma1_["UV"];

    C1["IU"] += alpha * H1["bm"] * T2["mIbU"];
    C1["IX"] += alpha * H1["bu"] * Gamma1_["uv"] * T2["vIbX"];
    C1["IX"] -= alpha * H1["vj"] * T2["jIuX"] * Gamma1_["uv"];
    C1["IU"] += alpha * H1["BM"] * T2["IMUB"];
    C1["IX"] += alpha * H1["BU"] * T2["IVXB"] * Gamma1_["UV"];
    C1["IX"] -= alpha * H1["VJ"] * T2["IJXU"] * Gamma1_["UV"];

    if (print_ > 2) {
        outfile->Printf("\n    Time for [H1, T2] -> C1 : %12.3f", timer.get());
    }
    dsrg_time_.add("121", timer.get());
}

void DSRG_MRPT2::H2_T1_C1(BlockedTensor& H2, BlockedTensor& T1, const double& alpha,
                          BlockedTensor& C1) {
    Timer timer;

    C1["ij"] += alpha * T1["ma"] * H2["iajm"];
    C1["ij"] += alpha * T1["xe"] * Gamma1_["yx"] * H2["iejy"];
    C1["ij"] -= alpha * T1["mu"] * Gamma1_["uv"] * H2["ivjm"];
    C1["ij"] += alpha * T1["MA"] * H2["iAjM"];
    C1["ij"] += alpha * T1["XE"] * Gamma1_["YX"] * H2["iEjY"];
    C1["ij"] -= alpha * T1["MU"] * Gamma1_["UV"] * H2["iVjM"];

    C1["IJ"] += alpha * T1["ma"] * H2["aImJ"];
    C1["IJ"] += alpha * T1["xe"] * Gamma1_["yx"] * H2["eIyJ"];
    C1["IJ"] -= alpha * T1["mu"] * Gamma1_["uv"] * H2["vImJ"];
    C1["IJ"] += alpha * T1["MA"] * H2["IAJM"];
    C1["IJ"] += alpha * T1["XE"] * Gamma1_["YX"] * H2["IEJY"];
    C1["IJ"] -= alpha * T1["MU"] * Gamma1_["UV"] * H2["IVJM"];

    if (print_ > 2) {
        outfile->Printf("\n    Time for [H2, T1] -> C1 : %12.3f", timer.get());
    }
    dsrg_time_.add("211", timer.get());
}

void DSRG_MRPT2::H2_T2_C1(BlockedTensor& H2, BlockedTensor& T2, const double& alpha,
                          BlockedTensor& C1) {
    Timer timer;
    BlockedTensor temp;

    // [Hbar2, T2] (C_2)^3 -> C1 particle contractions
    C1["ij"] += 0.5 * alpha * H2["abjm"] * T2["imab"];
    C1["ij"] += alpha * H2["aBjM"] * T2["iMaB"];
    C1["IJ"] += 0.5 * alpha * H2["ABJM"] * T2["IMAB"];
    C1["IJ"] += alpha * H2["aBmJ"] * T2["mIaB"];

    C1["ij"] += 0.5 * alpha * Gamma1_["uv"] * H2["abju"] * T2["ivab"];
    C1["ij"] += alpha * Gamma1_["UV"] * H2["aBjU"] * T2["iVaB"];
    C1["IJ"] += 0.5 * alpha * Gamma1_["UV"] * H2["ABJU"] * T2["IVAB"];
    C1["IJ"] += alpha * Gamma1_["uv"] * H2["aBuJ"] * T2["vIaB"];

    C1["ik"] += 0.5 * alpha * T2["ijux"] * Gamma1_["xy"] * Gamma1_["uv"] * H2["vykj"];
    C1["IK"] += 0.5 * alpha * T2["IJUX"] * Gamma1_["XY"] * Gamma1_["UV"] * H2["VYKJ"];
    temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"hHaA"});
    temp["iJvY"] = T2["iJuX"] * Gamma1_["XY"] * Gamma1_["uv"];
    C1["ik"] += alpha * temp["iJvY"] * H2["vYkJ"];
    C1["IK"] += alpha * temp["jIvY"] * H2["vYjK"];

    C1["ij"] -= alpha * Gamma1_["uv"] * H2["vbjm"] * T2["imub"];
    C1["ij"] -= alpha * Gamma1_["uv"] * H2["vBjM"] * T2["iMuB"];
    C1["ij"] -= alpha * Gamma1_["UV"] * T2["iMbU"] * H2["bVjM"];
    C1["IJ"] -= alpha * Gamma1_["UV"] * H2["VBJM"] * T2["IMUB"];
    C1["IJ"] -= alpha * Gamma1_["UV"] * H2["bVmJ"] * T2["mIbU"];
    C1["IJ"] -= alpha * Gamma1_["uv"] * H2["vBmJ"] * T2["mIuB"];

    C1["ij"] -= alpha * H2["vbjx"] * Gamma1_["uv"] * Gamma1_["xy"] * T2["iyub"];
    C1["ij"] -= alpha * H2["vBjX"] * Gamma1_["uv"] * Gamma1_["XY"] * T2["iYuB"];
    C1["ij"] -= alpha * H2["bVjX"] * Gamma1_["XY"] * Gamma1_["UV"] * T2["iYbU"];
    C1["IJ"] -= alpha * H2["VBJX"] * Gamma1_["UV"] * Gamma1_["XY"] * T2["IYUB"];
    C1["IJ"] -= alpha * H2["vBxJ"] * Gamma1_["uv"] * Gamma1_["xy"] * T2["yIuB"];
    C1["IJ"] -= alpha * T2["yIbU"] * Gamma1_["UV"] * Gamma1_["xy"] * H2["bVxJ"];

    // [Hbar2, T2] (C_2)^3 -> C1 hole contractions
    C1["ku"] -= 0.5 * alpha * H2["keij"] * T2["ijue"];
    C1["ku"] -= alpha * H2["kEiJ"] * T2["iJuE"];
    C1["KU"] -= 0.5 * alpha * H2["KEIJ"] * T2["IJUE"];
    C1["KU"] -= alpha * H2["eKiJ"] * T2["iJeU"];

    C1["kx"] -= 0.5 * alpha * Eta1_["uv"] * T2["ijxu"] * H2["kvij"];
    C1["kx"] -= alpha * Eta1_["UV"] * T2["iJxU"] * H2["kViJ"];
    C1["KX"] -= 0.5 * alpha * Eta1_["UV"] * T2["IJXU"] * H2["KVIJ"];
    C1["KX"] -= alpha * Eta1_["uv"] * T2["iJuX"] * H2["vKiJ"];

    C1["kw"] -= 0.5 * alpha * T2["vywb"] * Eta1_["uv"] * Eta1_["xy"] * H2["kbux"];
    C1["KW"] -= 0.5 * alpha * T2["VYWB"] * Eta1_["UV"] * Eta1_["XY"] * H2["KBUX"];
    temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"aApP"});
    temp["uXaB"] = T2["vYaB"] * Eta1_["uv"] * Eta1_["XY"];
    C1["kw"] -= alpha * H2["kBuX"] * temp["uXwB"];
    C1["KW"] -= alpha * H2["bKuX"] * temp["uXbW"];

    C1["kx"] += alpha * Eta1_["uv"] * T2["vjxe"] * H2["keuj"];
    C1["kx"] += alpha * Eta1_["uv"] * T2["vJxE"] * H2["kEuJ"];
    C1["kx"] += alpha * Eta1_["UV"] * H2["kEjU"] * T2["jVxE"];
    C1["KX"] += alpha * Eta1_["UV"] * T2["VJXE"] * H2["KEUJ"];
    C1["KX"] += alpha * Eta1_["uv"] * T2["vJeX"] * H2["eKuJ"];
    C1["KX"] += alpha * Eta1_["UV"] * H2["eKjU"] * T2["jVeX"];

    C1["kw"] += alpha * T2["vjwx"] * Eta1_["uv"] * Eta1_["xy"] * H2["kyuj"];
    C1["kw"] += alpha * T2["vJwX"] * Eta1_["uv"] * Eta1_["XY"] * H2["kYuJ"];
    C1["kw"] += alpha * T2["jVwX"] * Eta1_["XY"] * Eta1_["UV"] * H2["kYjU"];
    C1["KW"] += alpha * T2["VJWX"] * Eta1_["UV"] * Eta1_["XY"] * H2["KYUJ"];
    C1["KW"] += alpha * T2["vJxW"] * Eta1_["uv"] * Eta1_["xy"] * H2["yKuJ"];
    C1["KW"] += alpha * H2["yKjU"] * Eta1_["UV"] * Eta1_["xy"] * T2["jVxW"];

    // [Hbar2, T2] C_4 C_2 2:2 -> C1
    C1["ik"] += 0.25 * alpha * T2["ijxy"] * Lambda2_["xyuv"] * H2["uvkj"];
    C1["IK"] += 0.25 * alpha * T2["IJXY"] * Lambda2_["XYUV"] * H2["UVKJ"];
    temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"hHaA"});
    temp["iJuV"] = T2["iJxY"] * Lambda2_["xYuV"];
    C1["ik"] += alpha * H2["uVkJ"] * temp["iJuV"];
    C1["IK"] += alpha * H2["uVjK"] * temp["jIuV"];

    C1["iw"] -= 0.25 * alpha * Lambda2_["xyuv"] * T2["uvwb"] * H2["ibxy"];
    C1["IW"] -= 0.25 * alpha * Lambda2_["XYUV"] * T2["UVWB"] * H2["IBXY"];
    temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"aApP"});
    temp["xYaB"] = T2["uVaB"] * Lambda2_["xYuV"];
    C1["iw"] -= alpha * H2["iBxY"] * temp["xYwB"];
    C1["IW"] -= alpha * H2["bIxY"] * temp["xYbW"];

    C1["ij"] -= alpha * Lambda2_["yXuV"] * T2["iVyA"] * H2["uAjX"];
    C1["IJ"] -= alpha * Lambda2_["xYvU"] * T2["vIaY"] * H2["aUxJ"];
    C1["kw"] += alpha * Lambda2_["xYvU"] * T2["vIwY"] * H2["kUxI"];
    C1["KW"] += alpha * Lambda2_["yXuV"] * T2["iVyW"] * H2["uKiX"];

    temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"hapa"});
    temp["ixau"] += Lambda2_["xyuv"] * T2["ivay"];
    temp["ixau"] += Lambda2_["xYuV"] * T2["iVaY"];
    C1["ij"] += alpha * temp["ixau"] * H2["aujx"];
    C1["kw"] -= alpha * H2["kuix"] * temp["ixwu"];
    temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"hApA"});
    temp["iXaU"] += Lambda2_["XYUV"] * T2["iVaY"];
    temp["iXaU"] += Lambda2_["yXvU"] * T2["ivay"];
    C1["ij"] += alpha * temp["iXaU"] * H2["aUjX"];
    C1["kw"] -= alpha * H2["kUiX"] * temp["iXwU"];
    temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"aHaP"});
    temp["xIuA"] += Lambda2_["xyuv"] * T2["vIyA"];
    temp["xIuA"] += Lambda2_["xYuV"] * T2["VIYA"];
    C1["IJ"] += alpha * temp["xIuA"] * H2["uAxJ"];
    C1["KW"] -= alpha * H2["uKxI"] * temp["xIuW"];
    temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"HAPA"});
    temp["IXAU"] += Lambda2_["XYUV"] * T2["IVAY"];
    temp["IXAU"] += Lambda2_["yXvU"] * T2["vIyA"];
    C1["IJ"] += alpha * temp["IXAU"] * H2["AUJX"];
    C1["KW"] -= alpha * H2["KUIX"] * temp["IXWU"];

    // [Hbar2, T2] C_4 C_2 1:3 -> C1
    temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"pa"});
    temp["au"] += 0.5 * Lambda2_["xyuv"] * H2["avxy"];
    temp["au"] += Lambda2_["xYuV"] * H2["aVxY"];
    C1["jx"] += alpha * temp["au"] * T2["ujax"];
    C1["JX"] += alpha * temp["au"] * T2["uJaX"];
    temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"PA"});
    temp["AU"] += 0.5 * Lambda2_["XYUV"] * H2["AVXY"];
    temp["AU"] += Lambda2_["xYvU"] * H2["vAxY"];
    C1["jx"] += alpha * temp["AU"] * T2["jUxA"];
    C1["JX"] += alpha * temp["AU"] * T2["UJAX"];

    temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"ah"});
    temp["xi"] += 0.5 * Lambda2_["xyuv"] * H2["uviy"];
    temp["xi"] += Lambda2_["xYuV"] * H2["uViY"];
    C1["ju"] -= alpha * temp["xi"] * T2["ijxu"];
    C1["JU"] -= alpha * temp["xi"] * T2["iJxU"];
    temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"AH"});
    temp["XI"] += 0.5 * Lambda2_["XYUV"] * H2["UVIY"];
    temp["XI"] += Lambda2_["yXvU"] * H2["vUyI"];
    C1["ju"] -= alpha * temp["XI"] * T2["jIuX"];
    C1["JU"] -= alpha * temp["XI"] * T2["IJXU"];

    temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"av"});
    temp["xe"] += 0.5 * T2["uvey"] * Lambda2_["xyuv"];
    temp["xe"] += T2["uVeY"] * Lambda2_["xYuV"];
    C1["ij"] += alpha * temp["xe"] * H2["eixj"];
    C1["IJ"] += alpha * temp["xe"] * H2["eIxJ"];
    temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"AV"});
    temp["XE"] += 0.5 * T2["UVEY"] * Lambda2_["XYUV"];
    temp["XE"] += T2["uVyE"] * Lambda2_["yXuV"];
    C1["ij"] += alpha * temp["XE"] * H2["iEjX"];
    C1["IJ"] += alpha * temp["XE"] * H2["EIXJ"];

    temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"ca"});
    temp["mu"] += 0.5 * T2["mvxy"] * Lambda2_["xyuv"];
    temp["mu"] += T2["mVxY"] * Lambda2_["xYuV"];
    C1["ij"] -= alpha * temp["mu"] * H2["uimj"];
    C1["IJ"] -= alpha * temp["mu"] * H2["uImJ"];
    temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"CA"});
    temp["MU"] += 0.5 * T2["MVXY"] * Lambda2_["XYUV"];
    temp["MU"] += T2["vMxY"] * Lambda2_["xYvU"];
    C1["ij"] -= alpha * temp["MU"] * H2["iUjM"];
    C1["IJ"] -= alpha * temp["MU"] * H2["UIMJ"];

    if (print_ > 2) {
        outfile->Printf("\n    Time for [H2, T2] -> C1 : %12.3f", timer.get());
    }
    dsrg_time_.add("221", timer.get());
}

void DSRG_MRPT2::H1_T2_C2(BlockedTensor& H1, BlockedTensor& T2, const double& alpha,
                          BlockedTensor& C2) {
    Timer timer;

    C2["ijku"] += alpha * T2["ijau"] * H1["ak"];
    C2["ijuk"] += alpha * T2["ijub"] * H1["bk"];
    C2["kjuv"] -= alpha * T2["ijuv"] * H1["ki"];
    C2["ikuv"] -= alpha * T2["ijuv"] * H1["kj"];

    C2["iJkU"] += alpha * T2["iJaU"] * H1["ak"];
    C2["iJuK"] += alpha * T2["iJuB"] * H1["BK"];
    C2["kJuV"] -= alpha * T2["iJuV"] * H1["ki"];
    C2["iKuV"] -= alpha * T2["iJuV"] * H1["KJ"];

    C2["IJKU"] += alpha * T2["IJAU"] * H1["AK"];
    C2["IJUK"] += alpha * T2["IJUB"] * H1["BK"];
    C2["KJUV"] -= alpha * T2["IJUV"] * H1["KI"];
    C2["IKUV"] -= alpha * T2["IJUV"] * H1["KJ"];

    if (print_ > 2) {
        outfile->Printf("\n    Time for [H1, T2] -> C2 : %12.3f", timer.get());
    }
    dsrg_time_.add("122", timer.get());
}

void DSRG_MRPT2::H2_T1_C2(BlockedTensor& H2, BlockedTensor& T1, const double& alpha,
                          BlockedTensor& C2) {
    Timer timer;

    C2["ijkl"] += alpha * T1["ia"] * H2["ajkl"];
    C2["jikl"] += alpha * T1["ia"] * H2["jakl"];
    C2["kluj"] -= alpha * T1["iu"] * H2["klij"];
    C2["klju"] -= alpha * T1["iu"] * H2["klji"];

    C2["iJkL"] += alpha * T1["ia"] * H2["aJkL"];
    C2["jIkL"] += alpha * T1["IA"] * H2["jAkL"];
    C2["kLuJ"] -= alpha * T1["iu"] * H2["kLiJ"];
    C2["kLjU"] -= alpha * T1["IU"] * H2["kLjI"];

    C2["IJKL"] += alpha * T1["IA"] * H2["AJKL"];
    C2["JIKL"] += alpha * T1["IA"] * H2["JAKL"];
    C2["KLUJ"] -= alpha * T1["IU"] * H2["KLIJ"];
    C2["KLJU"] -= alpha * T1["IU"] * H2["KLJI"];

    if (print_ > 2) {
        outfile->Printf("\n    Time for [H2, T1] -> C2 : %12.3f", timer.get());
    }
    dsrg_time_.add("212", timer.get());
}

void DSRG_MRPT2::H2_T2_C2(BlockedTensor& H2, BlockedTensor& T2, const double& alpha,
                          BlockedTensor& C2) {
    Timer timer;

    // particle-particle contractions
    C2["ijkl"] += 0.5 * alpha * H2["abkl"] * T2["ijab"];
    C2["iJkL"] += alpha * H2["aBkL"] * T2["iJaB"];
    C2["IJKL"] += 0.5 * alpha * H2["ABKL"] * T2["IJAB"];

    C2["ijkl"] -= alpha * Gamma1_["xy"] * H2["ybkl"] * T2["ijxb"];
    C2["iJkL"] -= alpha * Gamma1_["xy"] * H2["yBkL"] * T2["iJxB"];
    C2["iJkL"] -= alpha * Gamma1_["XY"] * T2["iJbX"] * H2["bYkL"];
    C2["IJKL"] -= alpha * Gamma1_["XY"] * H2["YBKL"] * T2["IJXB"];

    // hole-hole contractions
    C2["kluv"] += 0.5 * alpha * H2["klij"] * T2["ijuv"];
    C2["kLuV"] += alpha * H2["kLiJ"] * T2["iJuV"];
    C2["KLUV"] += 0.5 * alpha * H2["KLIJ"] * T2["IJUV"];

    C2["kluv"] -= alpha * Eta1_["xy"] * T2["yjuv"] * H2["klxj"];
    C2["kLuV"] -= alpha * Eta1_["xy"] * T2["yJuV"] * H2["kLxJ"];
    C2["kLuV"] -= alpha * Eta1_["XY"] * H2["kLjX"] * T2["jYuV"];
    C2["KLUV"] -= alpha * Eta1_["XY"] * T2["YJUV"] * H2["KLXJ"];

    // hole-particle contractions
    BlockedTensor temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"aaaa"});
    temp["kjlu"] += alpha * H2["akml"] * T2["mjau"];
    temp["kjlu"] += alpha * H2["kAlM"] * T2["jMuA"];
    temp["kjlu"] += alpha * Gamma1_["xy"] * T2["yjau"] * H2["akxl"];
    temp["kjlu"] += alpha * Gamma1_["XY"] * T2["jYuA"] * H2["kAlX"];
    temp["kjlu"] -= alpha * Gamma1_["xy"] * H2["ykil"] * T2["ijxu"];
    temp["kjlu"] -= alpha * Gamma1_["XY"] * H2["kYlI"] * T2["jIuX"];
    C2["kjlu"] += temp["kjlu"];
    C2["jklu"] -= temp["kjlu"];
    C2["kjul"] -= temp["kjlu"];
    C2["jkul"] += temp["kjlu"];

    temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"AAAA"});
    temp["KJLU"] += alpha * H2["AKML"] * T2["MJAU"];
    temp["KJLU"] += alpha * H2["aKmL"] * T2["mJaU"];
    temp["KJLU"] += alpha * Gamma1_["XY"] * T2["YJAU"] * H2["AKXL"];
    temp["KJLU"] += alpha * Gamma1_["xy"] * T2["yJaU"] * H2["aKxL"];
    temp["KJLU"] -= alpha * Gamma1_["XY"] * H2["YKIL"] * T2["IJXU"];
    temp["KJLU"] -= alpha * Gamma1_["xy"] * H2["yKiL"] * T2["iJxU"];
    C2["KJLU"] += temp["KJLU"];
    C2["JKLU"] -= temp["KJLU"];
    C2["KJUL"] -= temp["KJLU"];
    C2["JKUL"] += temp["KJLU"];

    C2["kJlU"] += alpha * H2["akml"] * T2["mJaU"];
    C2["kJlU"] += alpha * H2["kAlM"] * T2["MJAU"];
    C2["kJlU"] += alpha * Gamma1_["xy"] * T2["yJaU"] * H2["akxl"];
    C2["kJlU"] += alpha * Gamma1_["XY"] * T2["YJAU"] * H2["kAlX"];
    C2["kJlU"] -= alpha * Gamma1_["xy"] * H2["ykil"] * T2["iJxU"];
    C2["kJlU"] -= alpha * Gamma1_["XY"] * H2["kYlI"] * T2["IJXU"];

    C2["iKlU"] -= alpha * T2["iMaU"] * H2["aKlM"];
    C2["iKlU"] -= alpha * Gamma1_["XY"] * T2["iYaU"] * H2["aKlX"];
    C2["iKlU"] += alpha * Gamma1_["xy"] * H2["yKlJ"] * T2["iJxU"];

    C2["kJuL"] -= alpha * T2["mJuB"] * H2["kBmL"];
    C2["kJuL"] -= alpha * Gamma1_["xy"] * T2["yJuB"] * H2["kBxL"];
    C2["kJuL"] += alpha * Gamma1_["XY"] * H2["kYiL"] * T2["iJuX"];

    C2["iKuL"] += alpha * T2["imub"] * H2["bKmL"];
    C2["iKuL"] += alpha * T2["iMuB"] * H2["BKML"];
    C2["iKuL"] += alpha * Gamma1_["xy"] * T2["iyub"] * H2["bKxL"];
    C2["iKuL"] += alpha * Gamma1_["XY"] * T2["iYuB"] * H2["BKXL"];
    C2["iKuL"] -= alpha * Gamma1_["xy"] * H2["yKjL"] * T2["ijux"];
    C2["iKuL"] -= alpha * Gamma1_["XY"] * H2["YKJL"] * T2["iJuX"];

    if (print_ > 2) {
        outfile->Printf("\n    Time for [H2, T2] -> C2 : %12.3f", timer.get());
    }
    dsrg_time_.add("222", timer.get());
}

void DSRG_MRPT2::H2_T2_C3(BlockedTensor& H2, BlockedTensor& T2, const double& alpha,
                          BlockedTensor& C3) {
    dsrg_time_.create_code("223");
    Timer timer;

    // compute only all active !

    // aaa
    BlockedTensor temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"aaaaaa"});
    temp["xyzuvw"] += alpha * H2["xymw"] * T2["mzuv"];
    temp["xyzuvw"] -= alpha * H2["ezuv"] * T2["xyew"];
    std::vector<std::string> label{"xyzuvw", "xyzwvu", "zyxwvu", "xyzuwv", "zyxuwv",
                                   "xzyuvw", "xzywvu", "zyxuvw", "xzyuwv"}; // ordering matters
    for (int i = 0, sign = 1; i < 9; ++i) {
        C3[label[i]] += sign * temp["xyzuvw"];
        sign *= -1;
    }

    // bbb
    temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"AAAAAA"});
    temp["XYZUVW"] += alpha * H2["XYMW"] * T2["MZUV"];
    temp["XYZUVW"] -= alpha * H2["EZUV"] * T2["XYEW"];
    for (int i = 0, sign = 1; i < 9; ++i) {
        std::string this_label = label[i];
        std::transform(this_label.begin(), this_label.end(), this_label.begin(),
                       (int (*)(int))toupper);
        C3[this_label] += sign * temp["XYZUVW"];
        sign *= -1;
    }

    // aab
    temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"aaAaaA"});
    temp["xyZwuV"] += alpha * H2["xymw"] * T2["mZuV"];
    temp["xyZwuV"] -= alpha * H2["eZuV"] * T2["xyew"];
    C3["xyZwuV"] += temp["xyZwuV"];
    C3["xyZuwV"] -= temp["xyZwuV"];

    temp.zero();
    temp["zxYuvW"] += alpha * H2["xYmW"] * T2["mzuv"];
    temp["zxYuvW"] -= alpha * H2["ezuv"] * T2["xYeW"];
    C3["zxYuvW"] += temp["zxYuvW"];
    C3["xzYuvW"] -= temp["zxYuvW"];

    temp.zero();
    temp["zxYwuV"] += alpha * H2["xYwM"] * T2["zMuV"];
    temp["zxYwuV"] -= alpha * H2["zEuV"] * T2["xYwE"];
    C3["zxYwuV"] += temp["zxYwuV"];
    C3["xzYwuV"] -= temp["zxYwuV"];
    C3["zxYuwV"] -= temp["zxYwuV"];
    C3["xzYuwV"] += temp["zxYwuV"];

    // abb
    temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"aAAaAA"});
    temp["zYXuVW"] += alpha * H2["XYMW"] * T2["zMuV"];
    temp["zYXuVW"] -= alpha * H2["zEuV"] * T2["XYEW"];
    C3["zYXuVW"] += temp["zYXuVW"];
    C3["zYXuWV"] -= temp["zYXuVW"];

    temp.zero();
    temp["xYZwVU"] += alpha * H2["xYwM"] * T2["MZUV"];
    temp["xYZwVU"] -= alpha * H2["EZUV"] * T2["xYwE"];
    C3["xYZwVU"] += temp["xYZwVU"];
    C3["xZYwVU"] -= temp["xYZwVU"];

    temp.zero();
    temp["xYZuVW"] += alpha * H2["xYmW"] * T2["mZuV"];
    temp["xYZuVW"] -= alpha * H2["eZuV"] * T2["xYeW"];
    C3["xYZuVW"] += temp["xYZuVW"];
    C3["xZYuVW"] -= temp["xYZuVW"];
    C3["xYZuWV"] -= temp["xYZuVW"];
    C3["xZYuWV"] += temp["xYZuVW"];

    if (print_ > 2) {
        outfile->Printf("\n    Time for [H2, T2] -> C3 : %12.3f", timer.get());
    }
    dsrg_time_.add("223", timer.get());
}

std::vector<std::vector<double>> DSRG_MRPT2::diagonalize_Fock_diagblocks(BlockedTensor& U) {
    // diagonal blocks identifiers (C-A-V ordering)
    std::vector<std::string> blocks{"cc", "aa", "vv", "CC", "AA", "VV"};

    // map MO space label to its Dimension
    std::map<std::string, Dimension> MOlabel_to_dimension;
    MOlabel_to_dimension[acore_label_] = mo_space_info_->get_dimension("RESTRICTED_DOCC");
    MOlabel_to_dimension[aactv_label_] = mo_space_info_->get_dimension("ACTIVE");
    MOlabel_to_dimension[avirt_label_] = mo_space_info_->get_dimension("RESTRICTED_UOCC");

    // eigen values to be returned
    size_t ncmo = mo_space_info_->size("CORRELATED");
    Dimension corr = mo_space_info_->get_dimension("CORRELATED");
    std::vector<double> eigenvalues_a(ncmo, 0.0);
    std::vector<double> eigenvalues_b(ncmo, 0.0);

    // map MO space label to its offset Dimension
    std::map<std::string, Dimension> MOlabel_to_offset_dimension;
    int nirrep = corr.n();
    MOlabel_to_offset_dimension["c"] = Dimension(std::vector<int>(nirrep, 0));
    MOlabel_to_offset_dimension["a"] = mo_space_info_->get_dimension("RESTRICTED_DOCC");
    MOlabel_to_offset_dimension["v"] =
        mo_space_info_->get_dimension("RESTRICTED_DOCC") + mo_space_info_->get_dimension("ACTIVE");

    // figure out index
    auto fill_eigen = [&](std::string block_label, int irrep, std::vector<double> values) {
        int h = irrep;
        size_t idx_begin = 0;
        while ((--h) >= 0)
            idx_begin += corr[h];

        std::string label(1, tolower(block_label[0]));
        idx_begin += MOlabel_to_offset_dimension[label][irrep];

        bool spin_alpha = islower(block_label[0]);
        size_t nvalues = values.size();
        if (spin_alpha) {
            for (size_t i = 0; i < nvalues; ++i) {
                eigenvalues_a[i + idx_begin] = values[i];
            }
        } else {
            for (size_t i = 0; i < nvalues; ++i) {
                eigenvalues_b[i + idx_begin] = values[i];
            }
        }
    };

    // diagonalize diagonal blocks
    for (const auto& block : blocks) {
        size_t dim = F_.block(block).dim(0);
        if (dim == 0) {
            continue;
        } else {
            std::string label(1, tolower(block[0]));
            Dimension space = MOlabel_to_dimension[label];
            int nirrep = space.n();

            // separate Fock with irrep
            for (int h = 0; h < nirrep; ++h) {
                size_t h_dim = space[h];
                ambit::Tensor U_h;
                if (h_dim == 0) {
                    continue;
                } else if (h_dim == 1) {
                    U_h = ambit::Tensor::build(tensor_type_, "U_h", std::vector<size_t>(2, h_dim));
                    U_h.data()[0] = 1.0;
                    ambit::Tensor F_block =
                        ambit::Tensor::build(tensor_type_, "F_block", F_.block(block).dims());
                    F_block.data() = F_.block(block).data();
                    ambit::Tensor T_h = separate_tensor(F_block, space, h);
                    fill_eigen(block, h, T_h.data());
                } else {
                    ambit::Tensor F_block =
                        ambit::Tensor::build(tensor_type_, "F_block", F_.block(block).dims());
                    F_block.data() = F_.block(block).data();
                    ambit::Tensor T_h = separate_tensor(F_block, space, h);
                    auto Feigen = T_h.syev(AscendingEigenvalue);
                    U_h = ambit::Tensor::build(tensor_type_, "U_h", std::vector<size_t>(2, h_dim));
                    U_h("pq") = Feigen["eigenvectors"]("pq");
                    fill_eigen(block, h, Feigen["eigenvalues"].data());
                }
                ambit::Tensor U_out = U.block(block);
                combine_tensor(U_out, U_h, space, h);
            }
        }
    }
    return {eigenvalues_a, eigenvalues_b};
}

ambit::Tensor DSRG_MRPT2::separate_tensor(ambit::Tensor& tens, const Dimension& irrep,
                                          const int& h) {
    // test tens and irrep
    int tens_dim = static_cast<int>(tens.dim(0));
    if (tens_dim != irrep.sum() || tens_dim != tens.dim(1)) {
        throw PSIEXCEPTION("Wrong dimension for the to-be-separated ambit Tensor.");
    }
    if (h >= irrep.n()) {
        throw PSIEXCEPTION("Ask for wrong irrep.");
    }

    // from relative (blocks) to absolute (big tensor) index
    auto rel_to_abs = [&](size_t i, size_t j, size_t offset) {
        return (i + offset) * tens_dim + (j + offset);
    };

    // compute offset
    size_t offset = 0, h_dim = irrep[h];
    int h_local = h;
    while ((--h_local) >= 0)
        offset += irrep[h_local];

    // fill in values
    ambit::Tensor T_h = ambit::Tensor::build(tensor_type_, "T_h", std::vector<size_t>(2, h_dim));
    for (size_t i = 0; i < h_dim; ++i) {
        for (size_t j = 0; j < h_dim; ++j) {
            size_t abs_idx = rel_to_abs(i, j, offset);
            T_h.data()[i * h_dim + j] = tens.data()[abs_idx];
        }
    }

    return T_h;
}

void DSRG_MRPT2::combine_tensor(ambit::Tensor& tens, ambit::Tensor& tens_h, const Dimension& irrep,
                                const int& h) {
    // test tens and irrep
    if (h >= irrep.n()) {
        throw PSIEXCEPTION("Ask for wrong irrep.");
    }
    size_t tens_h_dim = tens_h.dim(0), h_dim = irrep[h];
    if (tens_h_dim != h_dim || tens_h_dim != tens_h.dim(1)) {
        throw PSIEXCEPTION("Wrong dimension for the to-be-combined ambit Tensor.");
    }

    // from relative (blocks) to absolute (big tensor) index
    size_t tens_dim = tens.dim(0);
    auto rel_to_abs = [&](size_t i, size_t j, size_t offset) {
        return (i + offset) * tens_dim + (j + offset);
    };

    // compute offset
    size_t offset = 0;
    int h_local = h;
    while ((--h_local) >= 0)
        offset += irrep[h_local];

    // fill in values
    for (size_t i = 0; i < h_dim; ++i) {
        for (size_t j = 0; j < h_dim; ++j) {
            size_t abs_idx = rel_to_abs(i, j, offset);
            tens.data()[abs_idx] = tens_h.data()[i * h_dim + j];
        }
    }
}

ambit::BlockedTensor DSRG_MRPT2::get_T1(const std::vector<std::string>& blocks) {
    for (const std::string& block : blocks) {
        if (!T1_.is_block(block)) {
            std::string error = "Error from T1(blocks): cannot find block " + block;
            throw PSIEXCEPTION(error);
        }
    }
    ambit::BlockedTensor out = ambit::BlockedTensor::build(tensor_type_, "T1 selected", blocks);
    out["ia"] = T1_["ia"];
    out["IA"] = T1_["IA"];
    return out;
}

ambit::BlockedTensor DSRG_MRPT2::get_T1deGNO(const std::vector<std::string>& blocks) {
    for (const std::string& block : blocks) {
        if (!T1eff_.is_block(block)) {
            std::string error = "Error from T1deGNO(blocks): cannot find block " + block;
            throw PSIEXCEPTION(error);
        }
    }
    ambit::BlockedTensor out =
        ambit::BlockedTensor::build(tensor_type_, "T1deGNO selected", blocks);
    out["ia"] = T1eff_["ia"];
    out["IA"] = T1eff_["IA"];
    return out;
}

ambit::BlockedTensor DSRG_MRPT2::get_T2(const std::vector<std::string>& blocks) {
    for (const std::string& block : blocks) {
        if (!T2_.is_block(block)) {
            std::string error = "Error from T2(blocks): cannot find block " + block;
            throw PSIEXCEPTION(error);
        }
    }
    ambit::BlockedTensor out = ambit::BlockedTensor::build(tensor_type_, "T2 selected", blocks);
    out["ijab"] = T2_["ijab"];
    out["iJaB"] = T2_["iJaB"];
    out["IJAB"] = T2_["IJAB"];
    return out;
}

void DSRG_MRPT2::rotate_amp(SharedMatrix Ua, SharedMatrix Ub, const bool& transpose,
                            const bool& t1eff) {
    ambit::BlockedTensor U = BTF_->build(tensor_type_, "Uorb", spin_cases({"gg"}));

    std::map<char, std::vector<std::pair<size_t, size_t>>> space_to_relmo;
    space_to_relmo['c'] = mo_space_info_->get_relative_mo("RESTRICTED_DOCC");
    space_to_relmo['a'] = mo_space_info_->get_relative_mo("ACTIVE");
    space_to_relmo['v'] = mo_space_info_->get_relative_mo("RESTRICTED_UOCC");

    // alpha
    for (const std::string& block : {"cc", "aa", "vv"}) {
        char space = block[0];

        U.block(block).iterate([&](const std::vector<size_t>& i, double& value) {
            std::pair<size_t, size_t> p0 = space_to_relmo[space][i[0]];
            std::pair<size_t, size_t> p1 = space_to_relmo[space][i[1]];
            size_t h0 = p0.first, h1 = p1.first;
            size_t i0 = p0.second, i1 = p1.second;

            if (h0 == h1) {
                if (transpose) {
                    value = Ua->get(h0, i1, i0);
                } else {
                    value = Ua->get(h0, i0, i1);
                }
            }
        });
    }

    // beta
    for (const std::string& block : {"CC", "AA", "VV"}) {
        char space = tolower(block[0]);

        U.block(block).iterate([&](const std::vector<size_t>& i, double& value) {
            std::pair<size_t, size_t> p0 = space_to_relmo[space][i[0]];
            std::pair<size_t, size_t> p1 = space_to_relmo[space][i[1]];
            size_t h0 = p0.first, h1 = p1.first;
            size_t i0 = p0.second, i1 = p1.second;

            if (h0 == h1) {
                if (transpose) {
                    value = Ub->get(h0, i1, i0);
                } else {
                    value = Ub->get(h0, i0, i1);
                }
            }
        });
    }

    // rotate amplitudes
    BlockedTensor temp = ambit::BlockedTensor::build(tensor_type_, "Temp T2", spin_cases({"hhpp"}));
    temp["klab"] = U["ik"] * U["jl"] * T2_["ijab"];
    temp["kLaB"] = U["ik"] * U["JL"] * T2_["iJaB"];
    temp["KLAB"] = U["IK"] * U["JL"] * T2_["IJAB"];
    T2_["ijcd"] = temp["ijab"] * U["bd"] * U["ac"];
    T2_["iJcD"] = temp["iJaB"] * U["BD"] * U["ac"];
    T2_["IJCD"] = temp["IJAB"] * U["BD"] * U["AC"];

    temp = ambit::BlockedTensor::build(tensor_type_, "Temp T1", spin_cases({"hp"}));
    temp["jb"] = U["ij"] * T1_["ia"] * U["ab"];
    temp["JB"] = U["IJ"] * T1_["IA"] * U["AB"];
    T1_["ia"] = temp["ia"];
    T1_["IA"] = temp["IA"];

    if (t1eff) {
        temp["jb"] = U["ij"] * T1eff_["ia"] * U["ab"];
        temp["JB"] = U["IJ"] * T1eff_["IA"] * U["AB"];
        T1eff_["ia"] = temp["ia"];
        T1eff_["IA"] = temp["IA"];
    }
}

// Binary function to achieve sorting a vector of pair<vector, double>
// according to the double value in decending order
template <class T1, class T2, class G3 = std::greater<T2>> struct rsort_pair_second {
    bool operator()(const std::pair<T1, T2>& left, const std::pair<T1, T2>& right) {
        G3 p;
        return p(std::fabs(left.second), std::fabs(right.second));
    }
};

void DSRG_MRPT2::check_t2() {
    T2norm_ = 0.0;
    T2max_ = 0.0;
    double T2aanorm = 0.0, T2abnorm = 0.0, T2bbnorm = 0.0;
    size_t nonzero_aa = 0, nonzero_ab = 0, nonzero_bb = 0;
    std::vector<std::pair<std::vector<size_t>, double>> t2aa, t2ab, t2bb;

    // create all knids of spin maps; 0: aa, 1: ab, 2:bb
    std::map<int, double> spin_to_norm;
    std::map<int, double> spin_to_nonzero;
    std::map<int, std::vector<std::pair<std::vector<size_t>, double>>> spin_to_t2;
    std::map<int, std::vector<std::pair<std::vector<size_t>, double>>> spin_to_lt2;

    for (const std::string& block : T2_.block_labels()) {
        int spin = bool(isupper(block[0])) + bool(isupper(block[1]));
        std::vector<std::pair<std::vector<size_t>, double>>& temp_t2 = spin_to_t2[spin];
        std::vector<std::pair<std::vector<size_t>, double>>& temp_lt2 = spin_to_lt2[spin];

        T2_.block(block).citerate([&](const std::vector<size_t>& i, const double& value) {
            if (std::fabs(value) != 0.0) {
                size_t idx0 = label_to_spacemo_[block[0]][i[0]];
                size_t idx1 = label_to_spacemo_[block[1]][i[1]];
                size_t idx2 = label_to_spacemo_[block[2]][i[2]];
                size_t idx3 = label_to_spacemo_[block[3]][i[3]];

                ++spin_to_nonzero[spin];
                spin_to_norm[spin] += pow(value, 2.0);

                if ((idx0 <= idx1) && (idx2 <= idx3)) {
                    std::vector<size_t> indices = {idx0, idx1, idx2, idx3};
                    std::pair<std::vector<size_t>, double> idx_value =
                        std::make_pair(indices, value);

                    temp_t2.push_back(idx_value);
                    std::sort(temp_t2.begin(), temp_t2.end(),
                              rsort_pair_second<std::vector<size_t>, double>());
                    if (temp_t2.size() == ntamp_ + 1) {
                        temp_t2.pop_back();
                    }

                    if (std::fabs(value) > std::fabs(intruder_tamp_)) {
                        temp_lt2.push_back(idx_value);
                    }
                    std::sort(temp_lt2.begin(), temp_lt2.end(),
                              rsort_pair_second<std::vector<size_t>, double>());
                }
                T2max_ = T2max_ > std::fabs(value) ? T2max_ : std::fabs(value);
            }
        });
    }

    // update values
    T2aanorm = spin_to_norm[0];
    T2abnorm = spin_to_norm[1];
    T2bbnorm = spin_to_norm[2];
    T2norm_ = sqrt(T2aanorm + T2bbnorm + 4 * T2abnorm);

    nonzero_aa = spin_to_nonzero[0];
    nonzero_ab = spin_to_nonzero[1];
    nonzero_bb = spin_to_nonzero[2];

    t2aa = spin_to_t2[0];
    t2ab = spin_to_t2[1];
    t2bb = spin_to_t2[2];

    lt2aa_ = spin_to_lt2[0];
    lt2ab_ = spin_to_lt2[1];
    lt2bb_ = spin_to_lt2[2];

    // print summary
    print_amp_summary("AA", t2aa, sqrt(T2aanorm), nonzero_aa);
    print_amp_summary("AB", t2ab, sqrt(T2abnorm), nonzero_ab);
    print_amp_summary("BB", t2bb, sqrt(T2bbnorm), nonzero_bb);
}

void DSRG_MRPT2::check_t1() {
    T1max_ = 0.0;
    T1norm_ = 0.0;
    double T1anorm = 0.0, T1bnorm = 0.0;
    size_t nonzero_a = 0, nonzero_b = 0;
    std::vector<std::pair<std::vector<size_t>, double>> t1a, t1b;

    // create all kinds of spin maps; true: a, false: b
    std::map<bool, double> spin_to_norm;
    std::map<bool, double> spin_to_nonzero;
    std::map<bool, std::vector<std::pair<std::vector<size_t>, double>>> spin_to_t1;
    std::map<bool, std::vector<std::pair<std::vector<size_t>, double>>> spin_to_lt1;

    for (const std::string& block : T1_.block_labels()) {
        bool spin_alpha = islower(block[0]) ? true : false;
        std::vector<std::pair<std::vector<size_t>, double>>& temp_t1 = spin_to_t1[spin_alpha];
        std::vector<std::pair<std::vector<size_t>, double>>& temp_lt1 = spin_to_lt1[spin_alpha];

        T1_.block(block).citerate([&](const std::vector<size_t>& i, const double& value) {
            if (std::fabs(value) != 0.0) {
                size_t idx0 = label_to_spacemo_[block[0]][i[0]];
                size_t idx1 = label_to_spacemo_[block[1]][i[1]];

                std::vector<size_t> indices = {idx0, idx1};
                std::pair<std::vector<size_t>, double> idx_value = std::make_pair(indices, value);

                ++spin_to_nonzero[spin_alpha];
                spin_to_norm[spin_alpha] += pow(value, 2.0);

                temp_t1.push_back(idx_value);
                std::sort(temp_t1.begin(), temp_t1.end(),
                          rsort_pair_second<std::vector<size_t>, double>());
                if (temp_t1.size() == ntamp_ + 1) {
                    temp_t1.pop_back();
                }

                if (std::fabs(value) > std::fabs(intruder_tamp_)) {
                    temp_lt1.push_back(idx_value);
                }
                std::sort(temp_lt1.begin(), temp_lt1.end(),
                          rsort_pair_second<std::vector<size_t>, double>());

                T1max_ = T1max_ > std::fabs(value) ? T1max_ : std::fabs(value);
            }
        });
    }

    // update value
    T1anorm = spin_to_norm[true];
    T1bnorm = spin_to_norm[false];
    T1norm_ = sqrt(T1anorm + T1bnorm);

    nonzero_a = spin_to_nonzero[true];
    nonzero_b = spin_to_nonzero[false];

    t1a = spin_to_t1[true];
    t1b = spin_to_t1[false];

    lt1a_ = spin_to_lt1[true];
    lt1b_ = spin_to_lt1[false];

    // print summary
    print_amp_summary("A", t1a, sqrt(T1anorm), nonzero_a);
    print_amp_summary("B", t1b, sqrt(T1bnorm), nonzero_b);
}

void DSRG_MRPT2::print_amp_summary(const std::string& name,
                                   const std::vector<std::pair<std::vector<size_t>, double>>& list,
                                   const double& norm, const size_t& number_nonzero) {
    int rank = name.size();
    std::map<char, std::string> spin_case;
    spin_case['A'] = " ";
    spin_case['B'] = "_";

    std::string indent(4, ' ');
    std::string title =
        indent + "Largest T" + std::to_string(rank) + " amplitudes for spin case " + name + ":";
    std::string spin_title;
    std::string mo_title;
    std::string line;
    std::string output;
    std::string summary;

    auto extendstr = [&](std::string s, int n) {
        std::string o(s);
        while ((--n) > 0)
            o += s;
        return o;
    };

    if (rank == 1) {
        spin_title += str(boost::format(" %3c %3c %3c %3c %9c ") % spin_case[name[0]] % ' ' %
                          spin_case[name[0]] % ' ' % ' ');
        if (spin_title.find_first_not_of(' ') != std::string::npos) {
            spin_title = "\n" + indent + extendstr(spin_title, 3);
        } else {
            spin_title = "";
        }
        mo_title += str(boost::format(" %3c %3c %3c %3c %9c ") % 'i' % ' ' % 'a' % ' ' % ' ');
        mo_title = "\n" + indent + extendstr(mo_title, 3);
        for (size_t n = 0; n != list.size(); ++n) {
            if (n % 3 == 0)
                output += "\n" + indent;
            const auto& datapair = list[n];
            std::vector<size_t> idx = datapair.first;
            output += str(boost::format("[%3d %3c %3d %3c]%9.6f ") % idx[0] % ' ' % idx[1] % ' ' %
                          datapair.second);
        }
    } else if (rank == 2) {
        spin_title += str(boost::format(" %3c %3c %3c %3c %9c ") % spin_case[name[0]] %
                          spin_case[name[1]] % spin_case[name[0]] % spin_case[name[1]] % ' ');
        if (spin_title.find_first_not_of(' ') != std::string::npos) {
            spin_title = "\n" + indent + extendstr(spin_title, 3);
        } else {
            spin_title = "";
        }
        mo_title += str(boost::format(" %3c %3c %3c %3c %9c ") % 'i' % 'j' % 'a' % 'b' % ' ');
        mo_title = "\n" + indent + extendstr(mo_title, 3);
        for (size_t n = 0; n != list.size(); ++n) {
            if (n % 3 == 0)
                output += "\n" + indent;
            const auto& datapair = list[n];
            std::vector<size_t> idx = datapair.first;
            output += str(boost::format("[%3d %3d %3d %3d]%9.6f ") % idx[0] % idx[1] % idx[2] %
                          idx[3] % datapair.second);
        }
    } else {
        outfile->Printf("\n    Printing of amplitude is implemented only for T1 and T2!");
        return;
    }

    if (output.size() != 0) {
        int linesize = mo_title.size() - 2;
        line = "\n" + indent + std::string(linesize - indent.size(), '-');
        summary = "\n" + indent + "Norm of T" + std::to_string(rank) + name +
                  " vector: (nonzero elements: " + std::to_string(number_nonzero) + ")";
        std::string strnorm = str(boost::format("%.15f.") % norm);
        std::string blank(linesize - summary.size() - strnorm.size() + 1, ' ');
        summary += blank + strnorm;

        output = title + spin_title + mo_title + line + output + line + summary + line;
    }
    outfile->Printf("\n%s", output.c_str());
}

void DSRG_MRPT2::print_intruder(const std::string& name,
                                const std::vector<std::pair<std::vector<size_t>, double>>& list) {
    int rank = name.size();
    std::map<char, std::vector<double>> spin_to_F;
    spin_to_F['A'] = Fa_;
    spin_to_F['B'] = Fb_;

    std::string indent(4, ' ');
    std::string title = indent + "T" + std::to_string(rank) + " amplitudes larger than " +
                        str(boost::format("%.4f") % intruder_tamp_) + " for spin case " + name +
                        ":";
    std::string col_title;
    std::string line;
    std::string output;

    if (rank == 1) {
        int x = 30 + 2 * 3 + 2 - 11;
        std::string blank(x / 2, ' ');
        col_title += "\n" + indent + "    Amplitude         Value     " + blank + "Denominator" +
                     std::string(x - blank.size(), ' ');
        line = "\n" + indent + std::string(col_title.size() - indent.size() - 1, '-');

        for (size_t n = 0; n != list.size(); ++n) {
            const auto& datapair = list[n];
            std::vector<size_t> idx = datapair.first;
            size_t i = idx[0], a = idx[1];
            double fi = spin_to_F[name[0]][i], fa = spin_to_F[name[0]][a];
            double down = fi - fa;
            double v = datapair.second;

            output += "\n" + indent +
                      str(boost::format("[%3d %3c %3d %3c] %13.8f (%10.6f - %10.6f = %10.6f)") % i %
                          ' ' % a % ' ' % v % fi % fa % down);
        }
    } else if (rank == 2) {
        int x = 50 + 4 * 3 + 2 - 11;
        std::string blank(x / 2, ' ');
        col_title += "\n" + indent + "    Amplitude         Value     " + blank + "Denominator" +
                     std::string(x - blank.size(), ' ');
        line = "\n" + indent + std::string(col_title.size() - indent.size() - 1, '-');
        for (size_t n = 0; n != list.size(); ++n) {
            const auto& datapair = list[n];
            std::vector<size_t> idx = datapair.first;
            size_t i = idx[0], j = idx[1], a = idx[2], b = idx[3];
            double fi = spin_to_F[name[0]][i], fj = spin_to_F[name[1]][j];
            double fa = spin_to_F[name[0]][a], fb = spin_to_F[name[1]][b];
            double down = fi + fj - fa - fb;
            double v = datapair.second;

            output += "\n" + indent + str(boost::format("[%3d %3d %3d %3d] %13.8f (%10.6f + "
                                                        "%10.6f - %10.6f - %10.6f = %10.6f)") %
                                          i % j % a % b % v % fi % fj % fa % fb % down);
        }
    } else {
        outfile->Printf("\n    Printing of amplitude is implemented only for T1 and T2!");
        return;
    }

    if (output.size() != 0) {
        output = title + col_title + line + output + line;
    } else {
        output = title + " NULL";
    }
    outfile->Printf("\n%s", output.c_str());
}
}
} // End Namespaces
