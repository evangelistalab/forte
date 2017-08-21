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
#include "psi4/libmints/dipole.h"

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
    : MASTER_DSRG(reference, ref_wfn, options, ints, mo_space_info) {

    print_method_banner({"MR-DSRG Second-Order Perturbation Theory",
                         "Chenyang Li, Kevin Hannon, Francesco Evangelista"});
    outfile->Printf("\n    References:");
    outfile->Printf("\n      u-DSRG-MRPT2:    J. Chem. Theory Comput. 2015, 11, 2097.");
    outfile->Printf("\n      (pr-)DSRG-MRPT2: J. Chem. Phys. 2017, 146, 124132.");

    startup();
    print_summary();
}

DSRG_MRPT2::~DSRG_MRPT2() { cleanup(); }

void DSRG_MRPT2::startup() {
    //    s_ = options_.get_double("DSRG_S");
    //    if (s_ < 0) {
    //        outfile->Printf("\n  S parameter for DSRG must >= 0!");
    //        throw PSIEXCEPTION("S parameter for DSRG must >= 0!");
    //    }
    //    taylor_threshold_ = options_.get_int("TAYLOR_THRESHOLD");
    //    if (taylor_threshold_ <= 0) {
    //        outfile->Printf("\n  Threshold for Taylor expansion must be an integer "
    //                        "greater than 0!");
    //        throw PSIEXCEPTION("Threshold for Taylor expansion must be an integer "
    //                           "greater than 0!");
    //    }

    //    source_ = options_.get_str("SOURCE");
    //    if (source_ != "STANDARD" && source_ != "LABS" && source_ != "DYSON") {
    //        outfile->Printf("\n  Warning: SOURCE option \"%s\" is not implemented "
    //                        "in DSRG-MRPT2. Changed to STANDARD.",
    //                        source_.c_str());
    //        source_ = "STANDARD";
    //    }
    //    if (source_ == "STANDARD") {
    //        dsrg_source_ = std::make_shared<STD_SOURCE>(s_, taylor_threshold_);
    //    } else if (source_ == "LABS") {
    //        dsrg_source_ = std::make_shared<LABS_SOURCE>(s_, taylor_threshold_);
    //    } else if (source_ == "DYSON") {
    //        dsrg_source_ = std::make_shared<DYSON_SOURCE>(s_, taylor_threshold_);
    //    }

    //    ntamp_ = options_.get_int("NTAMP");
    //    intruder_tamp_ = options_.get_double("INTRUDER_TAMP");
    internal_amp_ = options_.get_str("INTERNAL_AMP") != "NONE";
    internal_amp_select_ = options_.get_str("INTERNAL_AMP_SELECT");

    //    // get frozen core energy
    //    Efrzc_ = ints_->frozen_core_energy();

    //    // get reference energy
    //    Eref_ = reference_.get_Eref();

    //    // orbital spaces
    //    BlockedTensor::reset_mo_spaces();
    //    BlockedTensor::set_expert_mode(true);
    //    acore_mos_ = mo_space_info_->get_corr_abs_mo("RESTRICTED_DOCC");
    //    bcore_mos_ = mo_space_info_->get_corr_abs_mo("RESTRICTED_DOCC");
    //    aactv_mos_ = mo_space_info_->get_corr_abs_mo("ACTIVE");
    //    bactv_mos_ = mo_space_info_->get_corr_abs_mo("ACTIVE");
    //    avirt_mos_ = mo_space_info_->get_corr_abs_mo("RESTRICTED_UOCC");
    //    bvirt_mos_ = mo_space_info_->get_corr_abs_mo("RESTRICTED_UOCC");

    //    // define space labels
    //    acore_label_ = "c";
    //    aactv_label_ = "a";
    //    avirt_label_ = "v";
    //    bcore_label_ = "C";
    //    bactv_label_ = "A";
    //    bvirt_label_ = "V";
    //    BTF_->add_mo_space(acore_label_, "mn", acore_mos_, AlphaSpin);
    //    BTF_->add_mo_space(bcore_label_, "MN", bcore_mos_, BetaSpin);
    //    BTF_->add_mo_space(aactv_label_, "uvwxyz123", aactv_mos_, AlphaSpin);
    //    BTF_->add_mo_space(bactv_label_, "UVWXYZ!@#", bactv_mos_, BetaSpin);
    //    BTF_->add_mo_space(avirt_label_, "ef", avirt_mos_, AlphaSpin);
    //    BTF_->add_mo_space(bvirt_label_, "EF", bvirt_mos_, BetaSpin);

    //    // map space labels to mo spaces
    //    label_to_spacemo_[acore_label_[0]] = acore_mos_;
    //    label_to_spacemo_[bcore_label_[0]] = bcore_mos_;
    //    label_to_spacemo_[aactv_label_[0]] = aactv_mos_;
    //    label_to_spacemo_[bactv_label_[0]] = bactv_mos_;
    //    label_to_spacemo_[avirt_label_[0]] = avirt_mos_;
    //    label_to_spacemo_[bvirt_label_[0]] = bvirt_mos_;

    //    // define composite spaces
    //    BTF_->add_composite_mo_space("h", "ijkl", {acore_label_, aactv_label_});
    //    BTF_->add_composite_mo_space("H", "IJKL", {bcore_label_, bactv_label_});
    //    BTF_->add_composite_mo_space("p", "abcd", {aactv_label_, avirt_label_});
    //    BTF_->add_composite_mo_space("P", "ABCD", {bactv_label_, bvirt_label_});
    //    BTF_->add_composite_mo_space("g", "pqrs", {acore_label_, aactv_label_, avirt_label_});
    //    BTF_->add_composite_mo_space("G", "PQRS", {bcore_label_, bactv_label_, bvirt_label_});

    //    // fill in density matrices
    //    Eta1_ = BTF_->build(tensor_type_, "Eta1", spin_cases({"aa"}));
    //    Gamma1_ = BTF_->build(tensor_type_, "Gamma1", spin_cases({"aa"}));
    //    Lambda2_ = BTF_->build(tensor_type_, "Lambda2", spin_cases({"aaaa"}));
    //    if (options_.get_str("THREEPDC") != "ZERO") {
    //        Lambda3_ = BTF_->build(tensor_type_, "Lambda3", spin_cases({"aaaaaa"}));
    //    }
    //    build_density();

    // prepare integrals
    V_ = BTF_->build(tensor_type_, "V", spin_cases({"pphh"}));
    build_ints();

    // build Fock matrix and its diagonal elements
    size_t ncmo = mo_space_info_->size("CORRELATED");
    Fa_ = std::vector<double>(ncmo);
    Fb_ = std::vector<double>(ncmo);
    F_ = BTF_->build(tensor_type_, "Fock", spin_cases({"gc", "pa", "vv"}));
    build_fock();

    //    multi_state_ = options_["AVG_STATE"].size() != 0;
    //    relax_ref_ = options_.get_str("RELAX_REF");
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

    //    MOdipole_ints_ = ints_->compute_MOdipole_ints(true, true);
    //    Mx_ = BTF_->build(tensor_type_, "Dipole X", spin_cases({"gg"}));
    //    My_ = BTF_->build(tensor_type_, "Dipole Y", spin_cases({"gg"}));
    //    Mz_ = BTF_->build(tensor_type_, "Dipole Z", spin_cases({"gg"}));

    //    Mbar1x_ = BTF_->build(tensor_type_, "DSRG Dipole 1 X", spin_cases({"aa"}));
    //    Mbar1y_ = BTF_->build(tensor_type_, "DSRG Dipole 1 Y", spin_cases({"aa"}));
    //    Mbar1z_ = BTF_->build(tensor_type_, "DSRG Dipole 1 Z", spin_cases({"aa"}));

    //    Mbar2x_ = BTF_->build(tensor_type_, "DSRG Dipole 2 X", spin_cases({"aaaa"}));
    //    Mbar2y_ = BTF_->build(tensor_type_, "DSRG Dipole 2 Y", spin_cases({"aaaa"}));
    //    Mbar2z_ = BTF_->build(tensor_type_, "DSRG Dipole 2 Z", spin_cases({"aaaa"}));

    //    fill_bare_dipoles();
    //    compute_ref_dipoles();

    // ignore semicanonical test
    std::string actv_type = options_.get_str("FCIMO_ACTV_TYPE");
    if (actv_type != "COMPLETE" && actv_type != "DOCI") {
        ignore_semicanonical_ = true;
    }

    //    // initialize timer for commutator
    //    dsrg_time_ = DSRG_TIME();

    // print levels
    //    print_ = options_.get_int("PRINT");
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
        reference_.L3aaa().print();
        reference_.L3aab().print();
        reference_.L3abb().print();
        reference_.L3bbb().print();
    }
}

// void DSRG_MRPT2::build_density() {
//    // prepare one-particle and one-hole densities
//    Gamma1_.block("aa")("pq") = reference_.L1a()("pq");
//    Gamma1_.block("AA")("pq") = reference_.L1a()("pq");

//    (Eta1_.block("aa")).iterate([&](const std::vector<size_t>& i, double& value) {
//        value = i[0] == i[1] ? 1.0 : 0.0;
//    });
//    (Eta1_.block("AA")).iterate([&](const std::vector<size_t>& i, double& value) {
//        value = i[0] == i[1] ? 1.0 : 0.0;
//    });
//    Eta1_.block("aa")("pq") -= reference_.L1a()("pq");
//    Eta1_.block("AA")("pq") -= reference_.L1a()("pq");

//    // prepare two-body density cumulants
//    Lambda2_.block("aaaa")("pqrs") = reference_.L2aa()("pqrs");
//    Lambda2_.block("aAaA")("pqrs") = reference_.L2ab()("pqrs");
//    Lambda2_.block("AAAA")("pqrs") = reference_.L2bb()("pqrs");

//    // prepare three-body density cumulants
//    if (options_.get_str("THREEPDC") != "ZERO") {
//        Lambda3_.block("aaaaaa")("pqrstu") = reference_.L3aaa()("pqrstu");
//        Lambda3_.block("aaAaaA")("pqrstu") = reference_.L3aab()("pqrstu");
//        Lambda3_.block("aAAaAA")("pqrstu") = reference_.L3abb()("pqrstu");
//        Lambda3_.block("AAAAAA")("pqrstu") = reference_.L3bbb()("pqrstu");
//    }
//}

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

                for (const size_t& nm : core_mos_) {
                    value += ints_->aptei_aa(np, nm, nq, nm);
                    value += ints_->aptei_ab(np, nm, nq, nm);
                }
            });
        } else {
            F_.block(block).iterate([&](const std::vector<size_t>& i, double& value) {
                size_t np = label_to_spacemo_[block[0]][i[0]];
                size_t nq = label_to_spacemo_[block[1]][i[1]];
                value = ints_->oei_b(np, nq);

                for (const size_t& nm : core_mos_) {
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
        std::string sep(2 + 16 * 3, '-');
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
    return E + Efrzc_ + Enuc;
}

double DSRG_MRPT2::compute_energy() {
    // check semi-canonical orbitals
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
    for (const auto& idx : actv_mos_) {
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

    Process::environment.globals["UNRELAXED ENERGY"] = Etotal;
    Process::environment.globals["CURRENT ENERGY"] = Etotal;
    outfile->Printf("\n\n  Energy took %10.3f s", DSRG_energy.get());
    outfile->Printf("\n");

    // transform dipole integrals
    if (do_dm_) {
        compute_pt2_dm();
    }

    // relax reference
    if (relax_ref_ != "NONE" || multi_state_) {
        BlockedTensor C1 = BTF_->build(tensor_type_, "C1", spin_cases({"aa"}));
        BlockedTensor C2 = BTF_->build(tensor_type_, "C2", spin_cases({"aaaa"}));
        H1_T1_C1aa(F_, T1_, 0.5, C1);
        H1_T2_C1aa(F_, T2_, 0.5, C1);
        H2_T1_C1aa(V_, T1_, 0.5, C1);
        H2_T2_C1aa(V_, T2_, 0.5, C1);
        H1_T2_C2aaaa(F_, T2_, 0.5, C2);
        H2_T1_C2aaaa(V_, T1_, 0.5, C2);
        H2_T2_C2aaaa(V_, T2_, 0.5, C2);

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
            H2_T2_C3aaaaaa(V_, T2_, 0.5, C3);

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
        myfile << core_mos_.size() + actv_mos_.size() << " " << core_mos_.size() + actv_mos_.size()
               << " " << actv_mos_.size() + virt_mos_.size() << " "
               << actv_mos_.size() + virt_mos_.size() << " \n";
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
    E += 0.25 * temp.block("aaaaaa")("uvwxyz") * reference_.L3aaa()("xyzuvw");

    // bbb
    temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"AAAAAA"});
    temp["UVWXYZ"] += V_["UVMZ"] * T2_["MWXY"];
    temp["UVWXYZ"] += V_["WEXY"] * T2_["UVEZ"];

    if (internal_amp_) {
        temp["UVWXYZ"] += V_["UV!Z"] * T2_["!WXY"];
        temp["UVWXYZ"] += V_["W!XY"] * T2_["UV!Z"];
    }
    E += 0.25 * temp.block("AAAAAA")("UVWXYZ") * reference_.L3bbb()("XYZUVW");

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
    E += 0.5 * temp.block("aaAaaA")("uvWxyZ") * reference_.L3aab()("xyZuvW");

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
    E += 0.5 * temp.block("aAAaAA")("uVWxYZ") * reference_.L3abb()("xYZuVW");

    outfile->Printf("  Done. Timing %15.6f s", timer.get());
    dsrg_time_.add("220", timer.get());
    return E;
}

void DSRG_MRPT2::compute_pt2_dm() {
    print_h2("DSRG-MRPT2 (unrelaxed) Dipole Moments (a.u.)");

    double nx = dm_nuc_[0];
    double ny = dm_nuc_[1];
    double nz = dm_nuc_[2];
    outfile->Printf("\n    Nuclear dipole moment:");
    outfile->Printf("\n      X: %10.6f  Y: %10.6f  Z: %10.6f\n", nx, ny, nz);

    double rx = dm_ref_[0];
    double ry = dm_ref_[1];
    double rz = dm_ref_[2];
    outfile->Printf("\n    Reference electronic dipole moment:");
    outfile->Printf("\n      X: %10.6f  Y: %10.6f  Z: %10.6f\n", rx, ry, rz);

    // compute DSRG-MRPT2 dressed dipoles
    Mbar0_ = std::vector<double>{rx, ry, rz};
    for (int i = 0; i < 3; ++i) {
        compute_pt2_dm_helper(dm_[i], Mbar0_[i], Mbar1_[i], Mbar2_[i]);
    }

    double x = Mbar0_[0];
    double y = Mbar0_[1];
    double z = Mbar0_[2];
    outfile->Printf("\n    DSRG-MRPT2 electronic dipole moment:");
    outfile->Printf("\n      X: %10.6f  Y: %10.6f  Z: %10.6f\n", x, y, z);

    rx += nx;
    ry += ny;
    rz += nz;
    double rt = std::sqrt(rx * rx + ry * ry + rz * rz);
    outfile->Printf("\n    Reference dipole moment:");
    outfile->Printf("\n      X: %10.6f  Y: %10.6f  Z: %10.6f  Total: %10.6f\n", rx, ry, rz, rt);

    x += nx;
    y += ny;
    z += nz;
    double t = std::sqrt(x * x + y * y + z * z);
    outfile->Printf("\n    DSRG-MRPT2 dipole moment:");
    outfile->Printf("\n      X: %10.6f  Y: %10.6f  Z: %10.6f  Total: %10.6f\n", x, y, z, t);

    Process::environment.globals["UNRELAXED DIPOLE"] = t;
}

void DSRG_MRPT2::compute_pt2_dm_helper(BlockedTensor& M, double& Mbar0, BlockedTensor& Mbar1,
                                       BlockedTensor& Mbar2) {
    /// Mbar = M + [M, A] + 0.5 * [[M, A], A]

    // compute [M, A] fully contracted terms
    // 2.0 accounts for [M, T]^dag
    H1_T1_C0(M, T1_, 2.0, Mbar0);
    H1_T2_C0(M, T2_, 2.0, Mbar0);

    // compute O = [M, A] nondiagonal one- and two-body terms
    BlockedTensor O1, O2, temp1, temp2;
    O1 = BTF_->build(tensor_type_, "O1", spin_cases({"pc", "va"}), true);
    O2 = BTF_->build(tensor_type_, "O2", spin_cases({"ppch", "ppac", "vpaa", "avaa"}), true);
    H1_T1_C1(M, T1_, 1.0, O1);
    H1_T2_C1(M, T2_, 1.0, O1);
    H1_T2_C2(M, T2_, 1.0, O2);

    temp1 = BTF_->build(tensor_type_, "temp1", spin_cases({"cp", "av"}), true);
    temp2 = BTF_->build(tensor_type_, "temp2", spin_cases({"chpp", "acpp", "aavp", "aaav"}), true);
    H1_T1_C1(M, T1_, 1.0, temp1);
    H1_T2_C1(M, T2_, 1.0, temp1);
    H1_T2_C2(M, T2_, 1.0, temp2);

    O1["ai"] += temp1["ia"];
    O1["AI"] += temp1["IA"];
    O2["abij"] += temp2["ijab"];
    O2["aBiJ"] += temp2["iJaB"];
    O2["ABIJ"] += temp2["IJAB"];

    // compute Mbar = 0.5 * [O, A]
    // fully contracted term
    H1_T1_C0(O1, T1_, 1.0, Mbar0);
    H1_T2_C0(O1, T2_, 1.0, Mbar0);
    H2_T1_C0(O2, T1_, 1.0, Mbar0);
    H2_T2_C0(O2, T2_, 1.0, Mbar0);

    // cases when we need Mbar1 and Mbar2
    if (relax_ref_ != "NONE" || multi_state_) {
        // set to bare
        Mbar1["uv"] = M["uv"];
        Mbar1["UV"] = M["UV"];

        // compute [M, T] active 1- and 2-body terms
        BlockedTensor C1, C2;
        C1 = BTF_->build(tensor_type_, "C1", spin_cases({"aa"}), true);
        C2 = BTF_->build(tensor_type_, "C2", spin_cases({"aaaa"}), true);
        H1_T1_C1aa(M, T1_, 1.0, C1);
        H1_T2_C1aa(M, T2_, 1.0, C1);
        H1_T2_C2aaaa(M, T2_, 1.0, C2);

        H1_T1_C1aa(O1, T1_, 0.5, C1);
        H1_T2_C1aa(O1, T2_, 0.5, C1);
        H2_T1_C1aa(O2, T1_, 0.5, C1);
        H2_T2_C1aa(O2, T2_, 0.5, C1);
        H1_T2_C2aaaa(O1, T2_, 0.5, C2);
        H2_T1_C2aaaa(O2, T1_, 0.5, C2);
        H2_T2_C2aaaa(O2, T2_, 0.5, C2);

        // add to C1 and C1^dag to Mbar
        Mbar1["uv"] += C1["uv"];
        Mbar1["uv"] += C1["vu"];
        Mbar1["UV"] += C1["UV"];
        Mbar1["UV"] += C1["VU"];
        Mbar2["uvxy"] += C2["uvxy"];
        Mbar2["uvxy"] += C2["xyuv"];
        Mbar2["uVxY"] += C2["uVxY"];
        Mbar2["uVxY"] += C2["xYuV"];
        Mbar2["UVXY"] += C2["UVXY"];
        Mbar2["UVXY"] += C2["XYUV"];
    }
}

double DSRG_MRPT2::compute_energy_relaxed() {
    double Edsrg = 0.0, Erelax = 0.0;

    // compute energy with fixed ref.
    Edsrg = compute_energy();

    // transfer integrals
    transfer_integrals();

    // dipole related
    std::vector<double> dm_dsrg(Mbar0_);
    std::map<std::string, std::vector<double>> dm_relax;

    // diagonalize Hbar depending on CAS_TYPE
    if (options_.get_str("CAS_TYPE") == "CAS") {

        FCI_MO fci_mo(reference_wavefunction_, options_, ints_, mo_space_info_);
        Erelax = fci_mo.compute_energy();

        if (do_dm_) {
            // de-normal-order DSRG dipole integrals
            for (int z = 0; z < 3; ++z) {
                if (do_dm_dirs_[z]) {
                    std::string name = "Dipole " + dm_dirs_[z] + " Integrals";
                    deGNO_ints(name, Mbar0_[z], Mbar1_[z], Mbar2_[z]);
                }
            }

            // compute permanent dipoles
            dm_relax = fci_mo.compute_relaxed_dm(Mbar0_, Mbar1_, Mbar2_);
        }

    } else {

        // it is simpler here to call FCI instead of FCISolver
        FCI fci(reference_wavefunction_, options_, ints_, mo_space_info_);
        fci.set_max_rdm_level(1);
        Erelax = fci.compute_energy();
    }

    // printing
    print_h2("DSRG-MRPT2 Energy Summary");
    outfile->Printf("\n    %-30s = %22.15f", "DSRG-MRPT2 Total Energy (fixed)  ", Edsrg);
    outfile->Printf("\n    %-30s = %22.15f", "DSRG-MRPT2 Total Energy (relaxed)", Erelax);
    outfile->Printf("\n");

    if (do_dm_) {
        print_h2("DSRG-MRPT2 Dipole Moment Summary");
        const double& nx = dm_nuc_[0];
        const double& ny = dm_nuc_[1];
        const double& nz = dm_nuc_[2];

        double x = dm_dsrg[0] + nx;
        double y = dm_dsrg[1] + ny;
        double z = dm_dsrg[2] + nz;
        double t = std::sqrt(x * x + y * y + z * z);
        outfile->Printf("\n    DSRG-MRPT2 unrelaxed dipole moment:");
        outfile->Printf("\n      X: %10.6f  Y: %10.6f  Z: %10.6f  Total: %10.6f\n", x, y, z, t);
        Process::environment.globals["UNRELAXED DIPOLE"] = t;

        // there should be only one entry for state-specific computations
        if (dm_relax.size() != 0) {
            for (const auto& p : dm_relax) {
                x = p.second[0] + nx;
                y = p.second[1] + ny;
                z = p.second[2] + nz;
                t = std::sqrt(x * x + y * y + z * z);
            }
            outfile->Printf("\n    DSRG-MRPT2 partially relaxed dipole moment:");
            outfile->Printf("\n      X: %10.6f  Y: %10.6f  Z: %10.6f  Total: %10.6f\n", x, y, z, t);
            Process::environment.globals["PARTIALLY RELAXED DIPOLE"] = t;
        }
    }

    Process::environment.globals["UNRELAXED ENERGY"] = Edsrg;
    Process::environment.globals["PARTIALLY RELAXED ENERGY"] = Erelax;
    Process::environment.globals["CURRENT ENERGY"] = Erelax;
    return Erelax;
}

std::shared_ptr<FCIIntegrals> DSRG_MRPT2::compute_Heff() {
    // de-normal-order DSRG transformed Hamiltonian
    double Edsrg = Eref_ + Hbar0_;
    if (options_.get_bool("FORM_HBAR3")) {
        deGNO_ints("Hamiltonian", Edsrg, Hbar1_, Hbar2_, Hbar3_);
    } else {
        deGNO_ints("Hamiltonian", Edsrg, Hbar1_, Hbar2_);
    }

    // transfer integrals to ForteIntegrals
    ints_->set_scalar(Edsrg - Enuc_ - Efrzc_);

    // TODO: before zero hhhh integrals, is is probably good to save a copy
    std::vector<size_t> hole_mos = core_mos_;
    hole_mos.insert(hole_mos.end(), actv_mos_.begin(), actv_mos_.end());
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

    Hbar1_.citerate(
        [&](const std::vector<size_t>& i, const std::vector<SpinType>& spin, const double& value) {
            if (spin[0] == AlphaSpin) {
                ints_->set_oei(i[0], i[1], value, true);
            } else {
                ints_->set_oei(i[0], i[1], value, false);
            }
        });

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

    ints_->update_integrals(false);

    // create FCIIntegral shared_ptr
    std::shared_ptr<FCIIntegrals> fci_ints =
        std::make_shared<FCIIntegrals>(ints_, actv_mos_, core_mos_);
    fci_ints->set_active_integrals(Hbar2_.block("aaaa"), Hbar2_.block("aAaA"),
                                   Hbar2_.block("AAAA"));
    fci_ints->compute_restricted_one_body_operator();

    return fci_ints;
}

void DSRG_MRPT2::transfer_integrals() {
    // printing
    print_h2("De-Normal-Order the DSRG Transformed Hamiltonian");

    //    double Edsrg = Eref_ + Hbar0_;
    //    deGNO_ints("Hamiltonian", Edsrg, Hbar1_, Hbar2_);

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
        scalar3 -= (1.0 / 36) * Hbar3_.block("aaaaaa")("xyzuvw") * reference_.L3aaa()("xyzuvw");
        scalar3 -= (1.0 / 36) * Hbar3_.block("AAAAAA")("XYZUVW") * reference_.L3bbb()("XYZUVW");
        scalar3 -= 0.25 * Hbar3_.block("aaAaaA")("xyZuvW") * reference_.L3aab()("xyZuvW");
        scalar3 -= 0.25 * Hbar3_.block("aAAaAA")("xYZuVW") * reference_.L3abb()("xYZuVW");

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
    //    ints_->set_scalar(Edsrg - Enuc_ - Efrzc_);
    ints_->set_scalar(scalar);

    //   a) zero hole integrals
    std::vector<size_t> hole_mos = core_mos_;
    hole_mos.insert(hole_mos.end(), actv_mos_.begin(), actv_mos_.end());
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
    //    Hbar1_.citerate(
    //        [&](const std::vector<size_t>& i, const std::vector<SpinType>& spin, const double&
    //        value) {
    //            if (spin[0] == AlphaSpin) {
    //                ints_->set_oei(i[0], i[1], value, true);
    //            } else {
    //                ints_->set_oei(i[0], i[1], value, false);
    //            }
    //        });
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

        Etest3 += (1.0 / 36) * Hbar3_.block("aaaaaa")("xyzuvw") * reference_.L3aaa()("xyzuvw");
        Etest3 += (1.0 / 36) * Hbar3_.block("AAAAAA")("XYZUVW") * reference_.L3bbb()("XYZUVW");
        Etest3 += 0.25 * Hbar3_.block("aaAaaA")("xyZuvW") * reference_.L3aab()("xyZuvW");
        Etest3 += 0.25 * Hbar3_.block("aAAaAA")("xYZuVW") * reference_.L3abb()("xYZuVW");

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

void DSRG_MRPT2::H1_T1_C1aa(BlockedTensor& H1, BlockedTensor& T1, const double& alpha,
                            BlockedTensor& C1) {
    Timer timer;

    C1["uv"] += alpha * H1["av"] * T1["ua"];
    C1["vu"] -= alpha * T1["iu"] * H1["vi"];

    C1["UV"] += alpha * H1["AV"] * T1["UA"];
    C1["VU"] -= alpha * T1["IU"] * H1["VI"];

    if (print_ > 2) {
        outfile->Printf("\n    Time for [H1, T1] -> C1 : %12.3f", timer.get());
    }
    dsrg_time_.add("111", timer.get());
}

void DSRG_MRPT2::H1_T2_C1aa(BlockedTensor& H1, BlockedTensor& T2, const double& alpha,
                            BlockedTensor& C1) {
    Timer timer;

    C1["vu"] += alpha * H1["bm"] * T2["vmub"];
    C1["yx"] += alpha * H1["bu"] * T2["yvxb"] * Gamma1_["uv"];
    C1["yx"] -= alpha * H1["vj"] * T2["yjxu"] * Gamma1_["uv"];
    C1["vu"] += alpha * H1["BM"] * T2["vMuB"];
    C1["yx"] += alpha * H1["BU"] * T2["yVxB"] * Gamma1_["UV"];
    C1["yx"] -= alpha * H1["VJ"] * T2["yJxU"] * Gamma1_["UV"];

    C1["VU"] += alpha * H1["bm"] * T2["mVbU"];
    C1["YX"] += alpha * H1["bu"] * Gamma1_["uv"] * T2["vYbX"];
    C1["YX"] -= alpha * H1["vj"] * T2["jYuX"] * Gamma1_["uv"];
    C1["VU"] += alpha * H1["BM"] * T2["VMUB"];
    C1["YX"] += alpha * H1["BU"] * T2["YVXB"] * Gamma1_["UV"];
    C1["YX"] -= alpha * H1["VJ"] * T2["YJXU"] * Gamma1_["UV"];

    if (print_ > 2) {
        outfile->Printf("\n    Time for [H1, T2] -> C1 : %12.3f", timer.get());
    }
    dsrg_time_.add("121", timer.get());
}

void DSRG_MRPT2::H2_T1_C1aa(BlockedTensor& H2, BlockedTensor& T1, const double& alpha,
                            BlockedTensor& C1) {
    Timer timer;

    C1["uv"] += alpha * T1["ma"] * H2["uavm"];
    C1["uv"] += alpha * T1["xe"] * Gamma1_["yx"] * H2["uevy"];
    C1["uv"] -= alpha * T1["mx"] * Gamma1_["xy"] * H2["uyvm"];
    C1["uv"] += alpha * T1["MA"] * H2["uAvM"];
    C1["uv"] += alpha * T1["XE"] * Gamma1_["YX"] * H2["uEvY"];
    C1["uv"] -= alpha * T1["MX"] * Gamma1_["XY"] * H2["uYvM"];

    C1["UV"] += alpha * T1["ma"] * H2["aUmV"];
    C1["UV"] += alpha * T1["xe"] * Gamma1_["yx"] * H2["eUyV"];
    C1["UV"] -= alpha * T1["mx"] * Gamma1_["xy"] * H2["yUmV"];
    C1["UV"] += alpha * T1["MA"] * H2["UAVM"];
    C1["UV"] += alpha * T1["XE"] * Gamma1_["YX"] * H2["UEVY"];
    C1["UV"] -= alpha * T1["MX"] * Gamma1_["XY"] * H2["UYVM"];

    if (print_ > 2) {
        outfile->Printf("\n    Time for [H2, T1] -> C1 : %12.3f", timer.get());
    }
    dsrg_time_.add("211", timer.get());
}

void DSRG_MRPT2::H2_T2_C1aa(BlockedTensor& H2, BlockedTensor& T2, const double& alpha,
                            BlockedTensor& C1) {
    Timer timer;
    BlockedTensor temp;

    // [Hbar2, T2] (C_2)^3 -> C1 particle contractions
    C1["uv"] += 0.5 * alpha * H2["abvm"] * T2["umab"];
    C1["uv"] += alpha * H2["aBvM"] * T2["uMaB"];
    C1["UV"] += 0.5 * alpha * H2["ABVM"] * T2["UMAB"];
    C1["UV"] += alpha * H2["aBmV"] * T2["mUaB"];

    C1["xy"] += 0.5 * alpha * Gamma1_["uv"] * H2["abyu"] * T2["xvab"];
    C1["xy"] += alpha * Gamma1_["UV"] * H2["aByU"] * T2["xVaB"];
    C1["XY"] += 0.5 * alpha * Gamma1_["UV"] * H2["ABYU"] * T2["XVAB"];
    C1["XY"] += alpha * Gamma1_["uv"] * H2["aBuY"] * T2["vXaB"];

    C1["wz"] += 0.5 * alpha * T2["wjux"] * Gamma1_["xy"] * Gamma1_["uv"] * H2["vyzj"];
    C1["WZ"] += 0.5 * alpha * T2["WJUX"] * Gamma1_["XY"] * Gamma1_["UV"] * H2["VYZJ"];
    temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"hHaA"});
    temp["iJvY"] = T2["iJuX"] * Gamma1_["XY"] * Gamma1_["uv"];
    C1["wz"] += alpha * temp["wJvY"] * H2["vYzJ"];
    C1["WZ"] += alpha * temp["jWvY"] * H2["vYjZ"];

    C1["xy"] -= alpha * Gamma1_["uv"] * H2["vbym"] * T2["xmub"];
    C1["xy"] -= alpha * Gamma1_["uv"] * H2["vByM"] * T2["xMuB"];
    C1["xy"] -= alpha * Gamma1_["UV"] * T2["xMbU"] * H2["bVyM"];
    C1["XY"] -= alpha * Gamma1_["UV"] * H2["VBYM"] * T2["XMUB"];
    C1["XY"] -= alpha * Gamma1_["UV"] * H2["bVmY"] * T2["mXbU"];
    C1["XY"] -= alpha * Gamma1_["uv"] * H2["vBmY"] * T2["mXuB"];

    C1["wz"] -= alpha * H2["vbzx"] * Gamma1_["uv"] * Gamma1_["xy"] * T2["wyub"];
    C1["wz"] -= alpha * H2["vBzX"] * Gamma1_["uv"] * Gamma1_["XY"] * T2["wYuB"];
    C1["wz"] -= alpha * H2["bVzX"] * Gamma1_["XY"] * Gamma1_["UV"] * T2["wYbU"];
    C1["WZ"] -= alpha * H2["VBZX"] * Gamma1_["UV"] * Gamma1_["XY"] * T2["WYUB"];
    C1["WZ"] -= alpha * H2["vBxZ"] * Gamma1_["uv"] * Gamma1_["xy"] * T2["yWuB"];
    C1["WZ"] -= alpha * T2["yWbU"] * Gamma1_["UV"] * Gamma1_["xy"] * H2["bVxZ"];

    // [Hbar2, T2] (C_2)^3 -> C1 hole contractions
    C1["zu"] -= 0.5 * alpha * H2["zeij"] * T2["ijue"];
    C1["zu"] -= alpha * H2["zEiJ"] * T2["iJuE"];
    C1["ZU"] -= 0.5 * alpha * H2["ZEIJ"] * T2["IJUE"];
    C1["ZU"] -= alpha * H2["eZiJ"] * T2["iJeU"];

    C1["zx"] -= 0.5 * alpha * Eta1_["uv"] * T2["ijxu"] * H2["zvij"];
    C1["zx"] -= alpha * Eta1_["UV"] * T2["iJxU"] * H2["zViJ"];
    C1["ZX"] -= 0.5 * alpha * Eta1_["UV"] * T2["IJXU"] * H2["ZVIJ"];
    C1["ZX"] -= alpha * Eta1_["uv"] * T2["iJuX"] * H2["vZiJ"];

    C1["zw"] -= 0.5 * alpha * T2["vywb"] * Eta1_["uv"] * Eta1_["xy"] * H2["zbux"];
    C1["ZW"] -= 0.5 * alpha * T2["VYWB"] * Eta1_["UV"] * Eta1_["XY"] * H2["ZBUX"];
    temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"aApP"});
    temp["uXaB"] = T2["vYaB"] * Eta1_["uv"] * Eta1_["XY"];
    C1["zw"] -= alpha * H2["zBuX"] * temp["uXwB"];
    C1["ZW"] -= alpha * H2["bZuX"] * temp["uXbW"];

    C1["zx"] += alpha * Eta1_["uv"] * T2["vjxe"] * H2["zeuj"];
    C1["zx"] += alpha * Eta1_["uv"] * T2["vJxE"] * H2["zEuJ"];
    C1["zx"] += alpha * Eta1_["UV"] * H2["zEjU"] * T2["jVxE"];
    C1["ZX"] += alpha * Eta1_["UV"] * T2["VJXE"] * H2["ZEUJ"];
    C1["ZX"] += alpha * Eta1_["uv"] * T2["vJeX"] * H2["eZuJ"];
    C1["ZX"] += alpha * Eta1_["UV"] * H2["eZjU"] * T2["jVeX"];

    C1["zw"] += alpha * T2["vjwx"] * Eta1_["uv"] * Eta1_["xy"] * H2["zyuj"];
    C1["zw"] += alpha * T2["vJwX"] * Eta1_["uv"] * Eta1_["XY"] * H2["zYuJ"];
    C1["zw"] += alpha * T2["jVwX"] * Eta1_["XY"] * Eta1_["UV"] * H2["zYjU"];
    C1["ZW"] += alpha * T2["VJWX"] * Eta1_["UV"] * Eta1_["XY"] * H2["ZYUJ"];
    C1["ZW"] += alpha * T2["vJxW"] * Eta1_["uv"] * Eta1_["xy"] * H2["yZuJ"];
    C1["ZW"] += alpha * H2["yZjU"] * Eta1_["UV"] * Eta1_["xy"] * T2["jVxW"];

    // [Hbar2, T2] C_4 C_2 2:2 -> C1
    C1["wz"] += 0.25 * alpha * T2["wjxy"] * Lambda2_["xyuv"] * H2["uvzj"];
    C1["WZ"] += 0.25 * alpha * T2["WJXY"] * Lambda2_["XYUV"] * H2["UVZJ"];
    temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"hHaA"});
    temp["iJuV"] = T2["iJxY"] * Lambda2_["xYuV"];
    C1["wz"] += alpha * H2["uVzJ"] * temp["wJuV"];
    C1["WZ"] += alpha * H2["uVjZ"] * temp["jWuV"];

    C1["zw"] -= 0.25 * alpha * Lambda2_["xyuv"] * T2["uvwb"] * H2["zbxy"];
    C1["ZW"] -= 0.25 * alpha * Lambda2_["XYUV"] * T2["UVWB"] * H2["ZBXY"];
    temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"aApP"});
    temp["xYaB"] = T2["uVaB"] * Lambda2_["xYuV"];
    C1["zw"] -= alpha * H2["zBxY"] * temp["xYwB"];
    C1["ZW"] -= alpha * H2["bZxY"] * temp["xYbW"];

    C1["wz"] -= alpha * Lambda2_["yXuV"] * T2["wVyA"] * H2["uAzX"];
    C1["WZ"] -= alpha * Lambda2_["xYvU"] * T2["vWaY"] * H2["aUxZ"];
    C1["zw"] += alpha * Lambda2_["xYvU"] * T2["vIwY"] * H2["zUxI"];
    C1["ZW"] += alpha * Lambda2_["yXuV"] * T2["iVyW"] * H2["uZiX"];

    temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"hapa"});
    temp["ixau"] += Lambda2_["xyuv"] * T2["ivay"];
    temp["ixau"] += Lambda2_["xYuV"] * T2["iVaY"];
    C1["wz"] += alpha * temp["wxau"] * H2["auzx"];
    C1["zw"] -= alpha * H2["zuix"] * temp["ixwu"];
    temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"hApA"});
    temp["iXaU"] += Lambda2_["XYUV"] * T2["iVaY"];
    temp["iXaU"] += Lambda2_["yXvU"] * T2["ivay"];
    C1["wz"] += alpha * temp["wXaU"] * H2["aUzX"];
    C1["zw"] -= alpha * H2["zUiX"] * temp["iXwU"];
    temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"aHaP"});
    temp["xIuA"] += Lambda2_["xyuv"] * T2["vIyA"];
    temp["xIuA"] += Lambda2_["xYuV"] * T2["VIYA"];
    C1["WZ"] += alpha * temp["xWuA"] * H2["uAxZ"];
    C1["ZW"] -= alpha * H2["uZxI"] * temp["xIuW"];
    temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"HAPA"});
    temp["IXAU"] += Lambda2_["XYUV"] * T2["IVAY"];
    temp["IXAU"] += Lambda2_["yXvU"] * T2["vIyA"];
    C1["WZ"] += alpha * temp["WXAU"] * H2["AUZX"];
    C1["ZW"] -= alpha * H2["ZUIX"] * temp["IXWU"];

    // [Hbar2, T2] C_4 C_2 1:3 -> C1
    temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"pa"});
    temp["au"] += 0.5 * Lambda2_["xyuv"] * H2["avxy"];
    temp["au"] += Lambda2_["xYuV"] * H2["aVxY"];
    C1["wx"] += alpha * temp["au"] * T2["uwax"];
    C1["WX"] += alpha * temp["au"] * T2["uWaX"];
    temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"PA"});
    temp["AU"] += 0.5 * Lambda2_["XYUV"] * H2["AVXY"];
    temp["AU"] += Lambda2_["xYvU"] * H2["vAxY"];
    C1["wx"] += alpha * temp["AU"] * T2["wUxA"];
    C1["WX"] += alpha * temp["AU"] * T2["UWAX"];

    temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"ah"});
    temp["xi"] += 0.5 * Lambda2_["xyuv"] * H2["uviy"];
    temp["xi"] += Lambda2_["xYuV"] * H2["uViY"];
    C1["vu"] -= alpha * temp["xi"] * T2["ivxu"];
    C1["VU"] -= alpha * temp["xi"] * T2["iVxU"];
    temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"AH"});
    temp["XI"] += 0.5 * Lambda2_["XYUV"] * H2["UVIY"];
    temp["XI"] += Lambda2_["yXvU"] * H2["vUyI"];
    C1["vu"] -= alpha * temp["XI"] * T2["vIuX"];
    C1["VU"] -= alpha * temp["XI"] * T2["IVXU"];

    temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"av"});
    temp["xe"] += 0.5 * T2["uvey"] * Lambda2_["xyuv"];
    temp["xe"] += T2["uVeY"] * Lambda2_["xYuV"];
    C1["uv"] += alpha * temp["xe"] * H2["euxv"];
    C1["UV"] += alpha * temp["xe"] * H2["eUxV"];
    temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"AV"});
    temp["XE"] += 0.5 * T2["UVEY"] * Lambda2_["XYUV"];
    temp["XE"] += T2["uVyE"] * Lambda2_["yXuV"];
    C1["uv"] += alpha * temp["XE"] * H2["uEvX"];
    C1["UV"] += alpha * temp["XE"] * H2["EUXV"];

    temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"ca"});
    temp["mu"] += 0.5 * T2["mvxy"] * Lambda2_["xyuv"];
    temp["mu"] += T2["mVxY"] * Lambda2_["xYuV"];
    C1["xy"] -= alpha * temp["mu"] * H2["uxmy"];
    C1["XY"] -= alpha * temp["mu"] * H2["uXmY"];
    temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"CA"});
    temp["MU"] += 0.5 * T2["MVXY"] * Lambda2_["XYUV"];
    temp["MU"] += T2["vMxY"] * Lambda2_["xYvU"];
    C1["xy"] -= alpha * temp["MU"] * H2["xUyM"];
    C1["XY"] -= alpha * temp["MU"] * H2["UXMY"];

    if (print_ > 2) {
        outfile->Printf("\n    Time for [H2, T2] -> C1 : %12.3f", timer.get());
    }
    dsrg_time_.add("221", timer.get());
}

void DSRG_MRPT2::H1_T2_C2aaaa(BlockedTensor& H1, BlockedTensor& T2, const double& alpha,
                              BlockedTensor& C2) {
    Timer timer;

    C2["uvxy"] += alpha * T2["uvay"] * H1["ax"];
    C2["uvxy"] += alpha * T2["uvxb"] * H1["by"];
    C2["uvxy"] -= alpha * T2["ivxy"] * H1["ui"];
    C2["uvxy"] -= alpha * T2["ujxy"] * H1["vj"];

    C2["uVxY"] += alpha * T2["uVaY"] * H1["ax"];
    C2["uVxY"] += alpha * T2["uVxB"] * H1["BY"];
    C2["uVxY"] -= alpha * T2["iVxY"] * H1["ui"];
    C2["uVxY"] -= alpha * T2["uJxY"] * H1["VJ"];

    C2["UVXY"] += alpha * T2["UVAY"] * H1["AX"];
    C2["UVXY"] += alpha * T2["UVXB"] * H1["BY"];
    C2["UVXY"] -= alpha * T2["IVXY"] * H1["UI"];
    C2["UVXY"] -= alpha * T2["UJXY"] * H1["VJ"];

    if (print_ > 2) {
        outfile->Printf("\n    Time for [H1, T2] -> C2 : %12.3f", timer.get());
    }
    dsrg_time_.add("122", timer.get());
}

void DSRG_MRPT2::H2_T1_C2aaaa(BlockedTensor& H2, BlockedTensor& T1, const double& alpha,
                              BlockedTensor& C2) {
    Timer timer;

    C2["uvxy"] += alpha * T1["ua"] * H2["avxy"];
    C2["uvxy"] += alpha * T1["va"] * H2["uaxy"];
    C2["uvxy"] -= alpha * T1["ix"] * H2["uviy"];
    C2["uvxy"] -= alpha * T1["iy"] * H2["uvxi"];

    C2["iJkL"] += alpha * T1["ia"] * H2["aJkL"];
    C2["jIkL"] += alpha * T1["IA"] * H2["jAkL"];
    C2["kLuJ"] -= alpha * T1["iu"] * H2["kLiJ"];
    C2["kLjU"] -= alpha * T1["IU"] * H2["kLjI"];

    C2["UVXY"] += alpha * T1["UA"] * H2["AVXY"];
    C2["UVXY"] += alpha * T1["VA"] * H2["UAXY"];
    C2["UVXY"] -= alpha * T1["IX"] * H2["UVIY"];
    C2["UVXY"] -= alpha * T1["IY"] * H2["UVXI"];

    if (print_ > 2) {
        outfile->Printf("\n    Time for [H2, T1] -> C2 : %12.3f", timer.get());
    }
    dsrg_time_.add("212", timer.get());
}

void DSRG_MRPT2::H2_T2_C2aaaa(BlockedTensor& H2, BlockedTensor& T2, const double& alpha,
                              BlockedTensor& C2) {
    Timer timer;

    // particle-particle contractions
    C2["uvxy"] += 0.5 * alpha * H2["abxy"] * T2["uvab"];
    C2["uVxY"] += alpha * H2["aBxY"] * T2["uVaB"];
    C2["UVXY"] += 0.5 * alpha * H2["ABXY"] * T2["UVAB"];

    C2["uvxy"] -= alpha * Gamma1_["wz"] * H2["zbxy"] * T2["uvwb"];
    C2["uVxY"] -= alpha * Gamma1_["wz"] * H2["zBxY"] * T2["uVwB"];
    C2["uVxY"] -= alpha * Gamma1_["WZ"] * T2["uVbW"] * H2["bZxY"];
    C2["UVXY"] -= alpha * Gamma1_["WZ"] * H2["ZBXY"] * T2["UVWB"];

    // hole-hole contractions
    C2["xyuv"] += 0.5 * alpha * H2["xyij"] * T2["ijuv"];
    C2["xYuV"] += alpha * H2["xYiJ"] * T2["iJuV"];
    C2["XYUV"] += 0.5 * alpha * H2["XYIJ"] * T2["IJUV"];

    C2["wzuv"] -= alpha * Eta1_["xy"] * T2["yjuv"] * H2["wzxj"];
    C2["wZuV"] -= alpha * Eta1_["xy"] * T2["yJuV"] * H2["wZxJ"];
    C2["wZuV"] -= alpha * Eta1_["XY"] * H2["wZjX"] * T2["jYuV"];
    C2["WZUV"] -= alpha * Eta1_["XY"] * T2["YJUV"] * H2["WZXJ"];

    // hole-particle contractions
    BlockedTensor temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"aaaa"});
    temp["uvxy"] += alpha * H2["aumx"] * T2["mvay"];
    temp["uvxy"] += alpha * H2["uAxM"] * T2["vMyA"];
    temp["uvxy"] += alpha * Gamma1_["wz"] * T2["zvay"] * H2["auwx"];
    temp["uvxy"] += alpha * Gamma1_["WZ"] * T2["vZyA"] * H2["uAxW"];
    temp["uvxy"] -= alpha * Gamma1_["wz"] * H2["zuix"] * T2["ivwy"];
    temp["uvxy"] -= alpha * Gamma1_["WZ"] * H2["uZxI"] * T2["vIyW"];
    C2["uvxy"] += temp["uvxy"];
    C2["vuxy"] -= temp["uvxy"];
    C2["uvyx"] -= temp["uvxy"];
    C2["vuyx"] += temp["uvxy"];

    temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"AAAA"});
    temp["UVXY"] += alpha * H2["AUMX"] * T2["MVAY"];
    temp["UVXY"] += alpha * H2["aUmX"] * T2["mVaY"];
    temp["UVXY"] += alpha * Gamma1_["WZ"] * T2["ZVAY"] * H2["AUWX"];
    temp["UVXY"] += alpha * Gamma1_["wz"] * T2["zVaY"] * H2["aUwX"];
    temp["UVXY"] -= alpha * Gamma1_["WZ"] * H2["ZUIX"] * T2["IVWY"];
    temp["UVXY"] -= alpha * Gamma1_["wz"] * H2["zUiX"] * T2["iVwY"];
    C2["UVXY"] += temp["UVXY"];
    C2["VUXY"] -= temp["UVXY"];
    C2["UVYX"] -= temp["UVXY"];
    C2["VUYX"] += temp["UVXY"];

    C2["uVxY"] += alpha * H2["aumx"] * T2["mVaY"];
    C2["uVxY"] += alpha * H2["uAxM"] * T2["MVAY"];
    C2["uVxY"] += alpha * Gamma1_["wz"] * T2["zVaY"] * H2["auwx"];
    C2["uVxY"] += alpha * Gamma1_["WZ"] * T2["ZVAY"] * H2["uAxW"];
    C2["uVxY"] -= alpha * Gamma1_["wz"] * H2["zuix"] * T2["iVwY"];
    C2["uVxY"] -= alpha * Gamma1_["WZ"] * H2["uZxI"] * T2["IVWY"];

    C2["uVxY"] -= alpha * T2["uMaY"] * H2["aVxM"];
    C2["uVxY"] -= alpha * Gamma1_["WZ"] * T2["uZaY"] * H2["aVxW"];
    C2["uVxY"] += alpha * Gamma1_["wz"] * H2["zVxJ"] * T2["uJwY"];

    C2["uVxY"] -= alpha * T2["mVxB"] * H2["uBmY"];
    C2["uVxY"] -= alpha * Gamma1_["wz"] * T2["zVxB"] * H2["uBwY"];
    C2["uVxY"] += alpha * Gamma1_["WZ"] * H2["uZiY"] * T2["iVxW"];

    C2["uVxY"] += alpha * T2["umxb"] * H2["bVmY"];
    C2["uVxY"] += alpha * T2["uMxB"] * H2["BVMY"];
    C2["uVxY"] += alpha * Gamma1_["wz"] * T2["uzxb"] * H2["bVwY"];
    C2["uVxY"] += alpha * Gamma1_["WZ"] * T2["uZxB"] * H2["BVWY"];
    C2["uVxY"] -= alpha * Gamma1_["wz"] * H2["zVjY"] * T2["ujxw"];
    C2["uVxY"] -= alpha * Gamma1_["WZ"] * H2["ZVJY"] * T2["uJxW"];

    if (print_ > 2) {
        outfile->Printf("\n    Time for [H2, T2] -> C2 : %12.3f", timer.get());
    }
    dsrg_time_.add("222", timer.get());
}

void DSRG_MRPT2::H2_T2_C3aaaaaa(BlockedTensor& H2, BlockedTensor& T2, const double& alpha,
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

// void DSRG_MRPT2::H1_T1_C0(BlockedTensor& H1, BlockedTensor& T1, const double& alpha, double& C0)
// {
//    Timer timer;

//    double E = 0.0;
//    E += H1["em"] * T1["me"];
//    E += H1["ex"] * T1["ye"] * Gamma1_["xy"];
//    E += H1["xm"] * T1["my"] * Eta1_["yx"];

//    E += H1["EM"] * T1["ME"];
//    E += H1["EX"] * T1["YE"] * Gamma1_["XY"];
//    E += H1["XM"] * T1["MY"] * Eta1_["YX"];

//    E *= alpha;
//    C0 += E;

//    if (print_ > 2) {
//        outfile->Printf("\n    Time for [H1, T1] -> C0 : %12.3f", timer.get());
//    }
//    dsrg_time_.add("110", timer.get());
//}

// void DSRG_MRPT2::H1_T2_C0(BlockedTensor& H1, BlockedTensor& T2, const double& alpha, double& C0)
// {
//    Timer timer;
//    BlockedTensor temp;
//    double E = 0.0;

//    temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"aaaa"});
//    temp["uvxy"] += H1["ex"] * T2["uvey"];
//    temp["uvxy"] -= H1["vm"] * T2["umxy"];
//    E += 0.5 * temp["uvxy"] * Lambda2_["xyuv"];

//    temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"AAAA"});
//    temp["UVXY"] += H1["EX"] * T2["UVEY"];
//    temp["UVXY"] -= H1["VM"] * T2["UMXY"];
//    E += 0.5 * temp["UVXY"] * Lambda2_["XYUV"];

//    temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"aAaA"});
//    temp["uVxY"] += H1["ex"] * T2["uVeY"];
//    temp["uVxY"] += H1["EY"] * T2["uVxE"];
//    temp["uVxY"] -= H1["VM"] * T2["uMxY"];
//    temp["uVxY"] -= H1["um"] * T2["mVxY"];
//    E += temp["uVxY"] * Lambda2_["xYuV"];

//    E *= alpha;
//    C0 += E;

//    if (print_ > 2) {
//        outfile->Printf("\n    Time for [H1, T2] -> C0 : %12.3f", timer.get());
//    }
//    dsrg_time_.add("120", timer.get());
//}

// void DSRG_MRPT2::H2_T1_C0(BlockedTensor& H2, BlockedTensor& T1, const double& alpha, double& C0)
// {
//    Timer timer;
//    BlockedTensor temp;
//    double E = 0.0;

//    temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"aaaa"});
//    temp["uvxy"] += H2["evxy"] * T1["ue"];
//    temp["uvxy"] -= H2["uvmy"] * T1["mx"];
//    E += 0.5 * temp["uvxy"] * Lambda2_["xyuv"];

//    temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"AAAA"});
//    temp["UVXY"] += H2["EVXY"] * T1["UE"];
//    temp["UVXY"] -= H2["UVMY"] * T1["MX"];
//    E += 0.5 * temp["UVXY"] * Lambda2_["XYUV"];

//    temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"aAaA"});
//    temp["uVxY"] += H2["eVxY"] * T1["ue"];
//    temp["uVxY"] += H2["uExY"] * T1["VE"];
//    temp["uVxY"] -= H2["uVmY"] * T1["mx"];
//    temp["uVxY"] -= H2["uVxM"] * T1["MY"];
//    E += temp["uVxY"] * Lambda2_["xYuV"];

//    E *= alpha;
//    C0 += E;

//    if (print_ > 2) {
//        outfile->Printf("\n    Time for [H2, T1] -> C0 : %12.3f", timer.get());
//    }
//    dsrg_time_.add("210", timer.get());
//}

// void DSRG_MRPT2::H2_T2_C0(BlockedTensor& H2, BlockedTensor& T2, const double& alpha, double& C0)
// {
//    Timer timer;

//    // <[Hbar2, T2]> (C_2)^4
//    double E = H2["eFmN"] * T2["mNeF"];
//    E += 0.25 * H2["efmn"] * T2["mnef"];
//    E += 0.25 * H2["EFMN"] * T2["MNEF"];

//    BlockedTensor temp = ambit::BlockedTensor::build(tensor_type_, "temp", spin_cases({"aa"}));
//    temp["vu"] += 0.5 * H2["efmu"] * T2["mvef"];
//    temp["vu"] += H2["fEuM"] * T2["vMfE"];
//    temp["VU"] += 0.5 * H2["EFMU"] * T2["MVEF"];
//    temp["VU"] += H2["eFmU"] * T2["mVeF"];
//    E += temp["vu"] * Gamma1_["uv"];
//    E += temp["VU"] * Gamma1_["UV"];

//    temp.zero();
//    temp["vu"] += 0.5 * H2["vemn"] * T2["mnue"];
//    temp["vu"] += H2["vEmN"] * T2["mNuE"];
//    temp["VU"] += 0.5 * H2["VEMN"] * T2["MNUE"];
//    temp["VU"] += H2["eVnM"] * T2["nMeU"];
//    E += temp["vu"] * Eta1_["uv"];
//    E += temp["VU"] * Eta1_["UV"];

//    temp = ambit::BlockedTensor::build(tensor_type_, "temp", spin_cases({"aaaa"}));
//    temp["yvxu"] += H2["efxu"] * T2["yvef"];
//    temp["yVxU"] += H2["eFxU"] * T2["yVeF"];
//    temp["YVXU"] += H2["EFXU"] * T2["YVEF"];
//    E += 0.25 * temp["yvxu"] * Gamma1_["xy"] * Gamma1_["uv"];
//    E += temp["yVxU"] * Gamma1_["UV"] * Gamma1_["xy"];
//    E += 0.25 * temp["YVXU"] * Gamma1_["XY"] * Gamma1_["UV"];

//    temp.zero();
//    temp["vyux"] += H2["vymn"] * T2["mnux"];
//    temp["vYuX"] += H2["vYmN"] * T2["mNuX"];
//    temp["VYUX"] += H2["VYMN"] * T2["MNUX"];
//    E += 0.25 * temp["vyux"] * Eta1_["uv"] * Eta1_["xy"];
//    E += temp["vYuX"] * Eta1_["uv"] * Eta1_["XY"];
//    E += 0.25 * temp["VYUX"] * Eta1_["UV"] * Eta1_["XY"];

//    temp.zero();
//    temp["vyux"] += H2["vemx"] * T2["myue"];
//    temp["vyux"] += H2["vExM"] * T2["yMuE"];
//    temp["VYUX"] += H2["eVmX"] * T2["mYeU"];
//    temp["VYUX"] += H2["VEXM"] * T2["YMUE"];
//    E += temp["vyux"] * Gamma1_["xy"] * Eta1_["uv"];
//    E += temp["VYUX"] * Gamma1_["XY"] * Eta1_["UV"];
//    temp["yVxU"] = H2["eVxM"] * T2["yMeU"];
//    E += temp["yVxU"] * Gamma1_["xy"] * Eta1_["UV"];
//    temp["vYuX"] = H2["vEmX"] * T2["mYuE"];
//    E += temp["vYuX"] * Gamma1_["XY"] * Eta1_["uv"];

//    temp.zero();
//    temp["yvxu"] += 0.5 * Gamma1_["wz"] * H2["vexw"] * T2["yzue"];
//    temp["yvxu"] += Gamma1_["WZ"] * H2["vExW"] * T2["yZuE"];
//    temp["yvxu"] += 0.5 * Eta1_["wz"] * T2["myuw"] * H2["vzmx"];
//    temp["yvxu"] += Eta1_["WZ"] * T2["yMuW"] * H2["vZxM"];
//    E += temp["yvxu"] * Gamma1_["xy"] * Eta1_["uv"];

//    temp["YVXU"] += 0.5 * Gamma1_["WZ"] * H2["VEXW"] * T2["YZUE"];
//    temp["YVXU"] += Gamma1_["wz"] * H2["eVwX"] * T2["zYeU"];
//    temp["YVXU"] += 0.5 * Eta1_["WZ"] * T2["MYUW"] * H2["VZMX"];
//    temp["YVXU"] += Eta1_["wz"] * H2["zVmX"] * T2["mYwU"];
//    E += temp["YVXU"] * Gamma1_["XY"] * Eta1_["UV"];

//    // <[Hbar2, T2]> C_4 (C_2)^2 HH -- combined with PH
//    temp = ambit::BlockedTensor::build(tensor_type_, "temp", spin_cases({"aaaa"}));
//    temp["uvxy"] += 0.125 * H2["uvmn"] * T2["mnxy"];
//    temp["uvxy"] += 0.25 * Gamma1_["wz"] * H2["uvmw"] * T2["mzxy"];
//    temp["uVxY"] += H2["uVmN"] * T2["mNxY"];
//    temp["uVxY"] += Gamma1_["wz"] * T2["zMxY"] * H2["uVwM"];
//    temp["uVxY"] += Gamma1_["WZ"] * H2["uVmW"] * T2["mZxY"];
//    temp["UVXY"] += 0.125 * H2["UVMN"] * T2["MNXY"];
//    temp["UVXY"] += 0.25 * Gamma1_["WZ"] * H2["UVMW"] * T2["MZXY"];

//    // <[Hbar2, T2]> C_4 (C_2)^2 PP -- combined with PH
//    temp["uvxy"] += 0.125 * H2["efxy"] * T2["uvef"];
//    temp["uvxy"] += 0.25 * Eta1_["wz"] * T2["uvew"] * H2["ezxy"];
//    temp["uVxY"] += H2["eFxY"] * T2["uVeF"];
//    temp["uVxY"] += Eta1_["wz"] * H2["zExY"] * T2["uVwE"];
//    temp["uVxY"] += Eta1_["WZ"] * T2["uVeW"] * H2["eZxY"];
//    temp["UVXY"] += 0.125 * H2["EFXY"] * T2["UVEF"];
//    temp["UVXY"] += 0.25 * Eta1_["WZ"] * T2["UVEW"] * H2["EZXY"];

//    // <[Hbar2, T2]> C_4 (C_2)^2 PH
//    temp["uvxy"] += H2["eumx"] * T2["mvey"];
//    temp["uvxy"] += H2["uExM"] * T2["vMyE"];
//    temp["uvxy"] += Gamma1_["wz"] * T2["zvey"] * H2["euwx"];
//    temp["uvxy"] += Gamma1_["WZ"] * H2["uExW"] * T2["vZyE"];
//    temp["uvxy"] += Eta1_["zw"] * H2["wumx"] * T2["mvzy"];
//    temp["uvxy"] += Eta1_["ZW"] * T2["vMyZ"] * H2["uWxM"];
//    E += temp["uvxy"] * Lambda2_["xyuv"];

//    temp["UVXY"] += H2["eUmX"] * T2["mVeY"];
//    temp["UVXY"] += H2["EUMX"] * T2["MVEY"];
//    temp["UVXY"] += Gamma1_["wz"] * T2["zVeY"] * H2["eUwX"];
//    temp["UVXY"] += Gamma1_["WZ"] * T2["ZVEY"] * H2["EUWX"];
//    temp["UVXY"] += Eta1_["zw"] * H2["wUmX"] * T2["mVzY"];
//    temp["UVXY"] += Eta1_["ZW"] * H2["WUMX"] * T2["MVZY"];
//    E += temp["UVXY"] * Lambda2_["XYUV"];

//    temp["uVxY"] += H2["uexm"] * T2["mVeY"];
//    temp["uVxY"] += H2["uExM"] * T2["MVEY"];
//    temp["uVxY"] -= H2["eVxM"] * T2["uMeY"];
//    temp["uVxY"] -= H2["uEmY"] * T2["mVxE"];
//    temp["uVxY"] += H2["eVmY"] * T2["umxe"];
//    temp["uVxY"] += H2["EVMY"] * T2["uMxE"];

//    temp["uVxY"] += Gamma1_["wz"] * T2["zVeY"] * H2["uexw"];
//    temp["uVxY"] += Gamma1_["WZ"] * T2["ZVEY"] * H2["uExW"];
//    temp["uVxY"] -= Gamma1_["WZ"] * H2["eVxW"] * T2["uZeY"];
//    temp["uVxY"] -= Gamma1_["wz"] * T2["zVxE"] * H2["uEwY"];
//    temp["uVxY"] += Gamma1_["wz"] * T2["zuex"] * H2["eVwY"];
//    temp["uVxY"] -= Gamma1_["WZ"] * H2["EVYW"] * T2["uZxE"];

//    temp["uVxY"] += Eta1_["zw"] * H2["wumx"] * T2["mVzY"];
//    temp["uVxY"] += Eta1_["ZW"] * T2["VMYZ"] * H2["uWxM"];
//    temp["uVxY"] -= Eta1_["zw"] * H2["wVxM"] * T2["uMzY"];
//    temp["uVxY"] -= Eta1_["ZW"] * T2["mVxZ"] * H2["uWmY"];
//    temp["uVxY"] += Eta1_["zw"] * T2["umxz"] * H2["wVmY"];
//    temp["uVxY"] += Eta1_["ZW"] * H2["WVMY"] * T2["uMxZ"];
//    E += temp["uVxY"] * Lambda2_["xYuV"];

//    // <[Hbar2, T2]> C_6 C_2
//    if (options_.get_str("THREEPDC") != "ZERO") {
//        temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"aaaaaa"});
//        temp["uvwxyz"] += H2["uviz"] * T2["iwxy"];
//        temp["uvwxyz"] += H2["waxy"] * T2["uvaz"];
//        E += 0.25 * temp.block("aaaaaa")("uvwxyz") * reference_.L3aaa()("xyzuvw");

//        temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"AAAAAA"});
//        temp["UVWXYZ"] += H2["UVIZ"] * T2["IWXY"];
//        temp["UVWXYZ"] += H2["WAXY"] * T2["UVAZ"];
//        E += 0.25 * temp.block("AAAAAA")("UVWXYZ") * reference_.L3bbb()("XYZUVW");

//        temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"aaAaaA"});
//        temp["uvWxyZ"] -= H2["uviy"] * T2["iWxZ"];
//        temp["uvWxyZ"] -= H2["uWiZ"] * T2["ivxy"];
//        temp["uvWxyZ"] += 2.0 * H2["uWyI"] * T2["vIxZ"];

//        temp["uvWxyZ"] += H2["aWxZ"] * T2["uvay"];
//        temp["uvWxyZ"] -= H2["vaxy"] * T2["uWaZ"];
//        temp["uvWxyZ"] -= 2.0 * H2["vAxZ"] * T2["uWyA"];
//        E += 0.5 * temp.block("aaAaaA")("uvWxyZ") * reference_.L3aab()("xyZuvW");

//        temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"aAAaAA"});
//        temp["uVWxYZ"] -= H2["VWIZ"] * T2["uIxY"];
//        temp["uVWxYZ"] -= H2["uVxI"] * T2["IWYZ"];
//        temp["uVWxYZ"] += 2.0 * H2["uViZ"] * T2["iWxY"];

//        temp["uVWxYZ"] += H2["uAxY"] * T2["VWAZ"];
//        temp["uVWxYZ"] -= H2["WAYZ"] * T2["uVxA"];
//        temp["uVWxYZ"] -= 2.0 * H2["aWxY"] * T2["uVaZ"];
//        E += 0.5 * temp.block("aAAaAA")("uVWxYZ") * reference_.L3abb()("xYZuVW");
//    }

//    // multiply prefactor and copy to C0
//    E *= alpha;
//    C0 += E;

//    if (print_ > 2) {
//        outfile->Printf("\n    Time for [H2, T2] -> C0 : %12.3f", timer.get());
//    }
//    dsrg_time_.add("220", timer.get());
//}

// void DSRG_MRPT2::H1_T1_C1(BlockedTensor& H1, BlockedTensor& T1, const double& alpha,
//                          BlockedTensor& C1) {
//    Timer timer;

//    C1["ip"] += alpha * H1["ap"] * T1["ia"];
//    C1["qa"] -= alpha * T1["ia"] * H1["qi"];
//    C1["IP"] += alpha * H1["AP"] * T1["IA"];
//    C1["QA"] -= alpha * T1["IA"] * H1["QI"];

//    if (print_ > 2) {
//        outfile->Printf("\n    Time for [H1, T1] -> C1 : %12.3f", timer.get());
//    }
//    dsrg_time_.add("111", timer.get());
//}

// void DSRG_MRPT2::H1_T2_C1(BlockedTensor& H1, BlockedTensor& T2, const double& alpha,
//                          BlockedTensor& C1) {
//    Timer timer;

//    C1["ia"] += alpha * H1["bm"] * T2["imab"];
//    C1["ia"] += alpha * H1["bu"] * Gamma1_["uv"] * T2["ivab"];
//    C1["ia"] -= alpha * H1["vj"] * Gamma1_["uv"] * T2["ijau"];
//    C1["ia"] += alpha * H1["BM"] * T2["iMaB"];
//    C1["ia"] += alpha * H1["BU"] * Gamma1_["UV"] * T2["iVaB"];
//    C1["ia"] -= alpha * H1["VJ"] * Gamma1_["UV"] * T2["iJaU"];

//    C1["IA"] += alpha * H1["bm"] * T2["mIbA"];
//    C1["IA"] += alpha * H1["bu"] * Gamma1_["uv"] * T2["vIbA"];
//    C1["IA"] -= alpha * H1["vj"] * Gamma1_["uv"] * T2["jIuA"];
//    C1["IA"] += alpha * H1["BM"] * T2["IMAB"];
//    C1["IA"] += alpha * H1["BU"] * Gamma1_["UV"] * T2["IVAB"];
//    C1["IA"] -= alpha * H1["VJ"] * Gamma1_["UV"] * T2["IJAU"];

//    if (print_ > 2) {
//        outfile->Printf("\n    Time for [H1, T2] -> C1 : %12.3f", timer.get());
//    }
//    dsrg_time_.add("121", timer.get());
//}

// void DSRG_MRPT2::H2_T1_C1(BlockedTensor& H2, BlockedTensor& T1, const double& alpha,
//                          BlockedTensor& C1) {
//    Timer timer;

//    C1["qp"] += alpha * T1["ma"] * H2["qapm"];
//    C1["qp"] += alpha * T1["xe"] * Gamma1_["yx"] * H2["qepy"];
//    C1["qp"] -= alpha * T1["mu"] * Gamma1_["uv"] * H2["qvpm"];
//    C1["qp"] += alpha * T1["MA"] * H2["qApM"];
//    C1["qp"] += alpha * T1["XE"] * Gamma1_["YX"] * H2["qEpY"];
//    C1["qp"] -= alpha * T1["MU"] * Gamma1_["UV"] * H2["qVpM"];

//    C1["QP"] += alpha * T1["ma"] * H2["aQmP"];
//    C1["QP"] += alpha * T1["xe"] * Gamma1_["yx"] * H2["eQyP"];
//    C1["QP"] -= alpha * T1["mu"] * Gamma1_["uv"] * H2["vQmP"];
//    C1["QP"] += alpha * T1["MA"] * H2["QAPM"];
//    C1["QP"] += alpha * T1["XE"] * Gamma1_["YX"] * H2["QEPY"];
//    C1["QP"] -= alpha * T1["MU"] * Gamma1_["UV"] * H2["QVPM"];

//    if (print_ > 2) {
//        outfile->Printf("\n    Time for [H2, T1] -> C1 : %12.3f", timer.get());
//    }
//    dsrg_time_.add("211", timer.get());
//}

// void DSRG_MRPT2::H2_T2_C1(BlockedTensor& H2, BlockedTensor& T2, const double& alpha,
//                          BlockedTensor& C1) {
//    Timer timer;
//    BlockedTensor temp;

//    /// minimum memory requirement: h * a * p * p

//    // [Hbar2, T2] (C_2)^3 -> C1 particle contractions
//    C1["ir"] += 0.5 * alpha * H2["abrm"] * T2["imab"];
//    C1["ir"] += alpha * H2["aBrM"] * T2["iMaB"];
//    C1["IR"] += 0.5 * alpha * H2["ABRM"] * T2["IMAB"];
//    C1["IR"] += alpha * H2["aBmR"] * T2["mIaB"];

//    C1["ir"] += 0.5 * alpha * Gamma1_["uv"] * T2["ivab"] * H2["abru"];
//    C1["ir"] += alpha * Gamma1_["UV"] * T2["iVaB"] * H2["aBrU"];
//    C1["IR"] += 0.5 * alpha * Gamma1_["UV"] * T2["IVAB"] * H2["ABRU"];
//    C1["IR"] += alpha * Gamma1_["uv"] * T2["vIaB"] * H2["aBuR"];

//    C1["ir"] += 0.5 * alpha * T2["ijux"] * Gamma1_["xy"] * Gamma1_["uv"] * H2["vyrj"];
//    C1["IR"] += 0.5 * alpha * T2["IJUX"] * Gamma1_["XY"] * Gamma1_["UV"] * H2["VYRJ"];
//    temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"hHaA"});
//    temp["iJvY"] = T2["iJuX"] * Gamma1_["XY"] * Gamma1_["uv"];
//    C1["ir"] += alpha * temp["iJvY"] * H2["vYrJ"];
//    C1["IR"] += alpha * temp["jIvY"] * H2["vYjR"];

//    C1["ir"] -= alpha * Gamma1_["uv"] * T2["imub"] * H2["vbrm"];
//    C1["ir"] -= alpha * Gamma1_["uv"] * T2["iMuB"] * H2["vBrM"];
//    C1["ir"] -= alpha * Gamma1_["UV"] * T2["iMbU"] * H2["bVrM"];
//    C1["IR"] -= alpha * Gamma1_["UV"] * T2["IMUB"] * H2["VBRM"];
//    C1["IR"] -= alpha * Gamma1_["UV"] * T2["mIbU"] * H2["bVmR"];
//    C1["IR"] -= alpha * Gamma1_["uv"] * T2["mIuB"] * H2["vBmR"];

//    C1["ir"] -= alpha * T2["iyub"] * Gamma1_["uv"] * Gamma1_["xy"] * H2["vbrx"];
//    C1["ir"] -= alpha * T2["iYuB"] * Gamma1_["uv"] * Gamma1_["XY"] * H2["vBrX"];
//    C1["ir"] -= alpha * T2["iYbU"] * Gamma1_["XY"] * Gamma1_["UV"] * H2["bVrX"];
//    C1["IR"] -= alpha * T2["IYUB"] * Gamma1_["UV"] * Gamma1_["XY"] * H2["VBRX"];
//    C1["IR"] -= alpha * T2["yIuB"] * Gamma1_["uv"] * Gamma1_["xy"] * H2["vBxR"];
//    C1["IR"] -= alpha * T2["yIbU"] * Gamma1_["UV"] * Gamma1_["xy"] * H2["bVxR"];

//    // [Hbar2, T2] (C_2)^3 -> C1 hole contractions
//    C1["pa"] -= 0.5 * alpha * H2["peij"] * T2["ijae"];
//    C1["pa"] -= alpha * H2["pEiJ"] * T2["iJaE"];
//    C1["PA"] -= 0.5 * alpha * H2["PEIJ"] * T2["IJAE"];
//    C1["PA"] -= alpha * H2["ePiJ"] * T2["iJeA"];

//    C1["pa"] -= 0.5 * alpha * Eta1_["uv"] * T2["ijau"] * H2["pvij"];
//    C1["pa"] -= alpha * Eta1_["UV"] * T2["iJaU"] * H2["pViJ"];
//    C1["PA"] -= 0.5 * alpha * Eta1_["UV"] * T2["IJAU"] * H2["PVIJ"];
//    C1["PA"] -= alpha * Eta1_["uv"] * T2["iJuA"] * H2["vPiJ"];

//    C1["pa"] -= 0.5 * alpha * T2["vyab"] * Eta1_["uv"] * Eta1_["xy"] * H2["pbux"];
//    C1["PA"] -= 0.5 * alpha * T2["VYAB"] * Eta1_["UV"] * Eta1_["XY"] * H2["PBUX"];
//    temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"aApP"});
//    temp["uXaB"] = T2["vYaB"] * Eta1_["uv"] * Eta1_["XY"];
//    C1["pa"] -= alpha * H2["pBuX"] * temp["uXaB"];
//    C1["PA"] -= alpha * H2["bPuX"] * temp["uXbA"];

//    C1["pa"] += alpha * Eta1_["uv"] * T2["vjae"] * H2["peuj"];
//    C1["pa"] += alpha * Eta1_["uv"] * T2["vJaE"] * H2["pEuJ"];
//    C1["pa"] += alpha * Eta1_["UV"] * T2["jVaE"] * H2["pEjU"];
//    C1["PA"] += alpha * Eta1_["UV"] * T2["VJAE"] * H2["PEUJ"];
//    C1["PA"] += alpha * Eta1_["uv"] * T2["vJeA"] * H2["ePuJ"];
//    C1["PA"] += alpha * Eta1_["UV"] * T2["jVeA"] * H2["ePjU"];

//    C1["pa"] += alpha * T2["vjax"] * Eta1_["uv"] * Eta1_["xy"] * H2["pyuj"];
//    C1["pa"] += alpha * T2["vJaX"] * Eta1_["uv"] * Eta1_["XY"] * H2["pYuJ"];
//    C1["pa"] += alpha * T2["jVaX"] * Eta1_["XY"] * Eta1_["UV"] * H2["pYjU"];
//    C1["PA"] += alpha * T2["VJAX"] * Eta1_["UV"] * Eta1_["XY"] * H2["PYUJ"];
//    C1["PA"] += alpha * T2["vJxA"] * Eta1_["uv"] * Eta1_["xy"] * H2["yPuJ"];
//    C1["PA"] += alpha * T2["jVxA"] * Eta1_["UV"] * Eta1_["xy"] * H2["yPjU"];

//    // [Hbar2, T2] C_4 C_2 2:2 -> C1
//    C1["ir"] += 0.25 * alpha * T2["ijxy"] * Lambda2_["xyuv"] * H2["uvrj"];
//    C1["IR"] += 0.25 * alpha * T2["IJXY"] * Lambda2_["XYUV"] * H2["UVRJ"];
//    temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"hHaA"});
//    temp["iJuV"] = T2["iJxY"] * Lambda2_["xYuV"];
//    C1["ir"] += alpha * H2["uVrJ"] * temp["iJuV"];
//    C1["IR"] += alpha * H2["uVjR"] * temp["jIuV"];

//    C1["pa"] -= 0.25 * alpha * Lambda2_["xyuv"] * T2["uvab"] * H2["pbxy"];
//    C1["PA"] -= 0.25 * alpha * Lambda2_["XYUV"] * T2["UVAB"] * H2["PBXY"];
//    temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"aApP"});
//    temp["xYaB"] = T2["uVaB"] * Lambda2_["xYuV"];
//    C1["pa"] -= alpha * H2["pBxY"] * temp["xYaB"];
//    C1["PA"] -= alpha * H2["bPxY"] * temp["xYbA"];

//    C1["ir"] -= alpha * Lambda2_["yXuV"] * T2["iVyA"] * H2["uArX"];
//    C1["IR"] -= alpha * Lambda2_["xYvU"] * T2["vIaY"] * H2["aUxR"];
//    C1["pa"] += alpha * Lambda2_["xYvU"] * T2["vIaY"] * H2["pUxI"];
//    C1["PA"] += alpha * Lambda2_["yXuV"] * T2["iVyA"] * H2["uPiX"];

//    temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"hapa"});
//    temp["ixau"] += Lambda2_["xyuv"] * T2["ivay"];
//    temp["ixau"] += Lambda2_["xYuV"] * T2["iVaY"];
//    C1["ir"] += alpha * temp["ixau"] * H2["aurx"];
//    C1["pa"] -= alpha * H2["puix"] * temp["ixau"];
//    temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"hApA"});
//    temp["iXaU"] += Lambda2_["XYUV"] * T2["iVaY"];
//    temp["iXaU"] += Lambda2_["yXvU"] * T2["ivay"];
//    C1["ir"] += alpha * temp["iXaU"] * H2["aUrX"];
//    C1["pa"] -= alpha * H2["pUiX"] * temp["iXaU"];
//    temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"aHaP"});
//    temp["xIuA"] += Lambda2_["xyuv"] * T2["vIyA"];
//    temp["xIuA"] += Lambda2_["xYuV"] * T2["VIYA"];
//    C1["IR"] += alpha * temp["xIuA"] * H2["uAxR"];
//    C1["PA"] -= alpha * H2["uPxI"] * temp["xIuA"];
//    temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"HAPA"});
//    temp["IXAU"] += Lambda2_["XYUV"] * T2["IVAY"];
//    temp["IXAU"] += Lambda2_["yXvU"] * T2["vIyA"];
//    C1["IR"] += alpha * temp["IXAU"] * H2["AURX"];
//    C1["PA"] -= alpha * H2["PUIX"] * temp["IXAU"];

//    // [Hbar2, T2] C_4 C_2 1:3 -> C1
//    temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"pa"});
//    temp["au"] += 0.5 * Lambda2_["xyuv"] * H2["avxy"];
//    temp["au"] += Lambda2_["xYuV"] * H2["aVxY"];
//    C1["jb"] += alpha * temp["au"] * T2["ujab"];
//    C1["JB"] += alpha * temp["au"] * T2["uJaB"];
//    temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"PA"});
//    temp["AU"] += 0.5 * Lambda2_["XYUV"] * H2["AVXY"];
//    temp["AU"] += Lambda2_["xYvU"] * H2["vAxY"];
//    C1["jb"] += alpha * temp["AU"] * T2["jUbA"];
//    C1["JB"] += alpha * temp["AU"] * T2["UJAB"];

//    temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"ah"});
//    temp["xi"] += 0.5 * Lambda2_["xyuv"] * H2["uviy"];
//    temp["xi"] += Lambda2_["xYuV"] * H2["uViY"];
//    C1["jb"] -= alpha * temp["xi"] * T2["ijxb"];
//    C1["JB"] -= alpha * temp["xi"] * T2["iJxB"];
//    temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"AH"});
//    temp["XI"] += 0.5 * Lambda2_["XYUV"] * H2["UVIY"];
//    temp["XI"] += Lambda2_["yXvU"] * H2["vUyI"];
//    C1["jb"] -= alpha * temp["XI"] * T2["jIbX"];
//    C1["JB"] -= alpha * temp["XI"] * T2["IJXB"];

//    temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"av"});
//    temp["xe"] += 0.5 * T2["uvey"] * Lambda2_["xyuv"];
//    temp["xe"] += T2["uVeY"] * Lambda2_["xYuV"];
//    C1["qs"] += alpha * temp["xe"] * H2["eqxs"];
//    C1["QS"] += alpha * temp["xe"] * H2["eQxS"];
//    temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"AV"});
//    temp["XE"] += 0.5 * T2["UVEY"] * Lambda2_["XYUV"];
//    temp["XE"] += T2["uVyE"] * Lambda2_["yXuV"];
//    C1["qs"] += alpha * temp["XE"] * H2["qEsX"];
//    C1["QS"] += alpha * temp["XE"] * H2["EQXS"];

//    temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"ca"});
//    temp["mu"] += 0.5 * T2["mvxy"] * Lambda2_["xyuv"];
//    temp["mu"] += T2["mVxY"] * Lambda2_["xYuV"];
//    C1["qs"] -= alpha * temp["mu"] * H2["uqms"];
//    C1["QS"] -= alpha * temp["mu"] * H2["uQmS"];
//    temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"CA"});
//    temp["MU"] += 0.5 * T2["MVXY"] * Lambda2_["XYUV"];
//    temp["MU"] += T2["vMxY"] * Lambda2_["xYvU"];
//    C1["qs"] -= alpha * temp["MU"] * H2["qUsM"];
//    C1["QS"] -= alpha * temp["MU"] * H2["UQMS"];

//    if (print_ > 2) {
//        outfile->Printf("\n    Time for [H2, T2] -> C1 : %12.3f", timer.get());
//    }
//    dsrg_time_.add("221", timer.get());
//}

// void DSRG_MRPT2::H1_T2_C2(BlockedTensor& H1, BlockedTensor& T2, const double& alpha,
//                          BlockedTensor& C2) {
//    Timer timer;

//    C2["ijpb"] += alpha * T2["ijab"] * H1["ap"];
//    C2["ijap"] += alpha * T2["ijab"] * H1["bp"];
//    C2["qjab"] -= alpha * T2["ijab"] * H1["qi"];
//    C2["iqab"] -= alpha * T2["ijab"] * H1["qj"];

//    C2["iJpB"] += alpha * T2["iJaB"] * H1["ap"];
//    C2["iJaP"] += alpha * T2["iJaB"] * H1["BP"];
//    C2["qJaB"] -= alpha * T2["iJaB"] * H1["qi"];
//    C2["iQaB"] -= alpha * T2["iJaB"] * H1["QJ"];

//    C2["IJPB"] += alpha * T2["IJAB"] * H1["AP"];
//    C2["IJAP"] += alpha * T2["IJAB"] * H1["BP"];
//    C2["QJAB"] -= alpha * T2["IJAB"] * H1["QI"];
//    C2["IQAB"] -= alpha * T2["IJAB"] * H1["QJ"];

//    if (print_ > 2) {
//        outfile->Printf("\n    Time for [H1, T2] -> C2 : %12.3f", timer.get());
//    }
//    dsrg_time_.add("122", timer.get());
//}

// void DSRG_MRPT2::H2_T1_C2(BlockedTensor& H2, BlockedTensor& T1, const double& alpha,
//                          BlockedTensor& C2) {
//    Timer timer;

//    C2["irpq"] += alpha * T1["ia"] * H2["arpq"];
//    C2["ripq"] += alpha * T1["ia"] * H2["rapq"];
//    C2["rsaq"] -= alpha * T1["ia"] * H2["rsiq"];
//    C2["rspa"] -= alpha * T1["ia"] * H2["rspi"];

//    C2["iRpQ"] += alpha * T1["ia"] * H2["aRpQ"];
//    C2["rIpQ"] += alpha * T1["IA"] * H2["rApQ"];
//    C2["rSaQ"] -= alpha * T1["ia"] * H2["rSiQ"];
//    C2["rSpA"] -= alpha * T1["IA"] * H2["rSpI"];

//    C2["IRPQ"] += alpha * T1["IA"] * H2["ARPQ"];
//    C2["RIPQ"] += alpha * T1["IA"] * H2["RAPQ"];
//    C2["RSAQ"] -= alpha * T1["IA"] * H2["RSIQ"];
//    C2["RSPA"] -= alpha * T1["IA"] * H2["RSPI"];

//    if (print_ > 2) {
//        outfile->Printf("\n    Time for [H2, T1] -> C2 : %12.3f", timer.get());
//    }
//    dsrg_time_.add("212", timer.get());
//}

// void DSRG_MRPT2::H2_T2_C2(BlockedTensor& H2, BlockedTensor& T2, const double& alpha,
//                          BlockedTensor& C2) {
//    Timer timer;

//    /// minimum memory requirement: g * g * p * p

//    // particle-particle contractions
//    C2["ijrs"] += 0.5 * alpha * H2["abrs"] * T2["ijab"];
//    C2["iJrS"] += alpha * H2["aBrS"] * T2["iJaB"];
//    C2["IJRS"] += 0.5 * alpha * H2["ABRS"] * T2["IJAB"];

//    C2["ijrs"] -= alpha * Gamma1_["xy"] * T2["ijxb"] * H2["ybrs"];
//    C2["iJrS"] -= alpha * Gamma1_["xy"] * T2["iJxB"] * H2["yBrS"];
//    C2["iJrS"] -= alpha * Gamma1_["XY"] * T2["iJbX"] * H2["bYrS"];
//    C2["IJRS"] -= alpha * Gamma1_["XY"] * T2["IJXB"] * H2["YBRS"];

//    // hole-hole contractions
//    C2["pqab"] += 0.5 * alpha * H2["pqij"] * T2["ijab"];
//    C2["pQaB"] += alpha * H2["pQiJ"] * T2["iJaB"];
//    C2["PQAB"] += 0.5 * alpha * H2["PQIJ"] * T2["IJAB"];

//    C2["pqab"] -= alpha * Eta1_["xy"] * T2["yjab"] * H2["pqxj"];
//    C2["pQaB"] -= alpha * Eta1_["xy"] * T2["yJaB"] * H2["pQxJ"];
//    C2["pQaB"] -= alpha * Eta1_["XY"] * T2["jYaB"] * H2["pQjX"];
//    C2["PQAB"] -= alpha * Eta1_["XY"] * T2["YJAB"] * H2["PQXJ"];

//    // hole-particle contractions
//    // figure out useful blocks of temp (assume symmetric C2 blocks, if cavv exists => acvv
//    exists)
//    std::vector<std::string> C2_blocks(C2.block_labels());
//    std::sort(C2_blocks.begin(), C2_blocks.end());
//    std::vector<std::string> temp_blocks;
//    for (const std::string& p : {"c", "a", "v"}) {
//        for (const std::string& q : {"c", "a"}) {
//            for (const std::string& r : {"c", "a", "v"}) {
//                for (const std::string& s : {"a", "v"}) {
//                    temp_blocks.emplace_back(p + q + r + s);
//                }
//            }
//        }
//    }
//    std::sort(temp_blocks.begin(), temp_blocks.end());
//    std::vector<std::string> blocks;
//    std::set_intersection(temp_blocks.begin(), temp_blocks.end(), C2_blocks.begin(),
//                          C2_blocks.end(), std::back_inserter(blocks));
//    BlockedTensor temp = ambit::BlockedTensor::build(tensor_type_, "temp", blocks);
//    temp["qjsb"] += alpha * H2["aqms"] * T2["mjab"];
//    temp["qjsb"] += alpha * H2["qAsM"] * T2["jMbA"];
//    temp["qjsb"] += alpha * Gamma1_["xy"] * T2["yjab"] * H2["aqxs"];
//    temp["qjsb"] += alpha * Gamma1_["XY"] * T2["jYbA"] * H2["qAsX"];
//    temp["qjsb"] -= alpha * Gamma1_["xy"] * T2["ijxb"] * H2["yqis"];
//    temp["qjsb"] -= alpha * Gamma1_["XY"] * T2["jIbX"] * H2["qYsI"];
//    C2["qjsb"] += temp["qjsb"];
//    C2["jqsb"] -= temp["qjsb"];
//    C2["qjbs"] -= temp["qjsb"];
//    C2["jqbs"] += temp["qjsb"];

//    // figure out useful blocks of temp (assume symmetric C2 blocks, if cavv exists => acvv
//    exists)
//    temp_blocks.clear();
//    for (const std::string& p : {"C", "A", "V"}) {
//        for (const std::string& q : {"C", "A"}) {
//            for (const std::string& r : {"C", "A", "V"}) {
//                for (const std::string& s : {"A", "V"}) {
//                    temp_blocks.emplace_back(p + q + r + s);
//                }
//            }
//        }
//    }
//    std::sort(temp_blocks.begin(), temp_blocks.end());
//    blocks.clear();
//    std::set_intersection(temp_blocks.begin(), temp_blocks.end(), C2_blocks.begin(),
//                          C2_blocks.end(), std::back_inserter(blocks));
//    temp = ambit::BlockedTensor::build(tensor_type_, "temp", blocks);
//    temp["QJSB"] += alpha * H2["AQMS"] * T2["MJAB"];
//    temp["QJSB"] += alpha * H2["aQmS"] * T2["mJaB"];
//    temp["QJSB"] += alpha * Gamma1_["XY"] * T2["YJAB"] * H2["AQXS"];
//    temp["QJSB"] += alpha * Gamma1_["xy"] * T2["yJaB"] * H2["aQxS"];
//    temp["QJSB"] -= alpha * Gamma1_["XY"] * T2["IJXB"] * H2["YQIS"];
//    temp["QJSB"] -= alpha * Gamma1_["xy"] * T2["iJxB"] * H2["yQiS"];
//    C2["QJSB"] += temp["QJSB"];
//    C2["JQSB"] -= temp["QJSB"];
//    C2["QJBS"] -= temp["QJSB"];
//    C2["JQBS"] += temp["QJSB"];

//    C2["qJsB"] += alpha * H2["aqms"] * T2["mJaB"];
//    C2["qJsB"] += alpha * H2["qAsM"] * T2["MJAB"];
//    C2["qJsB"] += alpha * Gamma1_["xy"] * T2["yJaB"] * H2["aqxs"];
//    C2["qJsB"] += alpha * Gamma1_["XY"] * T2["YJAB"] * H2["qAsX"];
//    C2["qJsB"] -= alpha * Gamma1_["xy"] * T2["iJxB"] * H2["yqis"];
//    C2["qJsB"] -= alpha * Gamma1_["XY"] * T2["IJXB"] * H2["qYsI"];

//    C2["iQsB"] -= alpha * T2["iMaB"] * H2["aQsM"];
//    C2["iQsB"] -= alpha * Gamma1_["XY"] * T2["iYaB"] * H2["aQsX"];
//    C2["iQsB"] += alpha * Gamma1_["xy"] * T2["iJxB"] * H2["yQsJ"];

//    C2["qJaS"] -= alpha * T2["mJaB"] * H2["qBmS"];
//    C2["qJaS"] -= alpha * Gamma1_["xy"] * T2["yJaB"] * H2["qBxS"];
//    C2["qJaS"] += alpha * Gamma1_["XY"] * T2["iJaX"] * H2["qYiS"];

//    C2["iQaS"] += alpha * T2["imab"] * H2["bQmS"];
//    C2["iQaS"] += alpha * T2["iMaB"] * H2["BQMS"];
//    C2["iQaS"] += alpha * Gamma1_["xy"] * T2["iyab"] * H2["bQxS"];
//    C2["iQaS"] += alpha * Gamma1_["XY"] * T2["iYaB"] * H2["BQXS"];
//    C2["iQaS"] -= alpha * Gamma1_["xy"] * T2["ijax"] * H2["yQjS"];
//    C2["iQaS"] -= alpha * Gamma1_["XY"] * T2["iJaX"] * H2["YQJS"];

//    if (print_ > 2) {
//        outfile->Printf("\n    Time for [H2, T2] -> C2 : %12.3f", timer.get());
//    }
//    dsrg_time_.add("222", timer.get());
//}

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
