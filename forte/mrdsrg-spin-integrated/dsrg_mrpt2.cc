/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2021 by its authors (see COPYING, COPYING.LESSER,
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

#include "psi4/libpsi4util/PsiOutStream.h"
#include "psi4/libpsi4util/process.h"
#include "psi4/libpsio/psio.h"
#include "psi4/libpsio/psio.hpp"
#include "psi4/libqt/qt.h"
#include "psi4/libmints/dipole.h"

#include "helpers/timer.h"
#include "ci_rdm/ci_rdms.h"
#include "boost/format.hpp"
#include "sci/fci_mo.h"
#include "fci/fci_solver.h"
#include "helpers/printing.h"
#include "dsrg_mrpt2.h"

using namespace ambit;

using namespace psi;

namespace forte {

DSRG_MRPT2::DSRG_MRPT2(RDMs rdms, std::shared_ptr<SCFInfo> scf_info,
                       std::shared_ptr<ForteOptions> options, std::shared_ptr<ForteIntegrals> ints,
                       std::shared_ptr<MOSpaceInfo> mo_space_info)
    : MASTER_DSRG(rdms, scf_info, options, ints, mo_space_info) {

    print_method_banner({"MR-DSRG Second-Order Perturbation Theory",
                         "Chenyang Li, Kevin Hannon, Francesco Evangelista"});
    outfile->Printf("\n    References:");
    outfile->Printf("\n      u-DSRG-MRPT2:    J. Chem. Theory Comput. 2015, 11, 2097.");
    outfile->Printf("\n      (pr-)DSRG-MRPT2: J. Chem. Phys. 2017, 146, 124132.");

    startup();
    print_options_summary();
}

DSRG_MRPT2::~DSRG_MRPT2() { cleanup(); }

void DSRG_MRPT2::startup() {
    // options for internal
    internal_amp_ = foptions_->get_str("INTERNAL_AMP") != "NONE";
    internal_amp_select_ = foptions_->get_str("INTERNAL_AMP_SELECT");

    // prepare integrals
    V_ = BTF_->build(tensor_type_, "V", spin_cases({"pphh"}));
    build_ints();

    // copy Fock matrix from master_dsrg
    F_ = BTF_->build(tensor_type_, "Fock", spin_cases({"gc", "pa", "vv"}));
    F_["pq"] = Fock_["pq"];
    F_["PQ"] = Fock_["PQ"];
    Fa_ = Fdiag_a_;
    Fb_ = Fdiag_b_;

    // Prepare Hbar
    if (relax_ref_ != "NONE" || multi_state_) {
        Hbar1_ = BTF_->build(tensor_type_, "1-body Hbar", spin_cases({"aa"}));
        Hbar2_ = BTF_->build(tensor_type_, "2-body Hbar", spin_cases({"aaaa"}));
        Hbar1_["uv"] = F_["uv"];
        Hbar1_["UV"] = F_["UV"];
        Hbar2_["uvxy"] = V_["uvxy"];
        Hbar2_["uVxY"] = V_["uVxY"];
        Hbar2_["UVXY"] = V_["UVXY"];

        if (foptions_->get_bool("FORM_HBAR3")) {
            Hbar3_ = BTF_->build(tensor_type_, "3-body Hbar", spin_cases({"aaaaaa"}));
        }
    }

    // print levels
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
        rdms_.L3aaa().print();
        rdms_.L3aab().print();
        rdms_.L3abb().print();
        rdms_.L3bbb().print();
    }
}

void DSRG_MRPT2::build_ints() {
    if (eri_df_) {
        // a simple trick when we cannot store <pq|rs> but can store <ij|ab>
        BlockedTensor B = BTF_->build(tensor_type_, "B", {"Lph", "LPH"});

        for (const std::string& block : B.block_labels()) {
            std::vector<size_t> iaux = label_to_spacemo_[block[0]];
            std::vector<size_t> ip = label_to_spacemo_[block[1]];
            std::vector<size_t> ih = label_to_spacemo_[block[2]];

            ambit::Tensor Bblock = ints_->three_integral_block(iaux, ip, ih);
            B.block(block).copy(Bblock);
        }

        V_["abij"] = B["gai"] * B["gbj"];
        V_["abij"] -= B["gaj"] * B["gbi"];

        V_["aBiJ"] = B["gai"] * B["gBJ"];

        V_["ABIJ"] = B["gAI"] * B["gBJ"];
        V_["ABIJ"] -= B["gAJ"] * B["gBI"];

    } else {
        V_.iterate(
            [&](const std::vector<size_t>& i, const std::vector<SpinType>& spin, double& value) {
                if ((spin[0] == AlphaSpin) and (spin[1] == AlphaSpin))
                    value = ints_->aptei_aa(i[0], i[1], i[2], i[3]);
                if ((spin[0] == AlphaSpin) and (spin[1] == BetaSpin))
                    value = ints_->aptei_ab(i[0], i[1], i[2], i[3]);
                if ((spin[0] == BetaSpin) and (spin[1] == BetaSpin))
                    value = ints_->aptei_bb(i[0], i[1], i[2], i[3]);
            });
    }
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

    // set F_ to Fock_ in master_dsrg because check_semi_orbs use Fock_
    Fock_["pq"] = F_["pq"];
    Fock_["PQ"] = F_["PQ"];
}

void DSRG_MRPT2::print_options_summary() {
    // Print a summary
    std::vector<std::pair<std::string, int>> calculation_info_int{{"ntamp", ntamp_}};

    std::vector<std::pair<std::string, double>> calculation_info_double{
        {"flow parameter", s_},
        {"taylor expansion threshold", pow(10.0, -double(taylor_threshold_))},
        {"intruder_tamp", intruder_tamp_}};

    std::vector<std::pair<std::string, std::string>> calculation_info_string{
        {"int_type", ints_type_},
        {"source operator", source_},
        {"reference relaxation", relax_ref_}};

    if (multi_state_) {
        calculation_info_string.push_back({"state_type", "MULTI-STATE"});
        calculation_info_string.push_back({"multi-state type", multi_state_algorithm_});
    } else {
        calculation_info_string.push_back({"state_type", "STATE-SPECIFIC"});
    }

    if (internal_amp_) {
        calculation_info_string.push_back({"internal_amp", foptions_->get_str("INTERNAL_AMP")});
        calculation_info_string.push_back({"internal_amp_select", internal_amp_select_});
    }

    std::vector<std::pair<std::string, bool>> calculation_info_bool{
        {"form Hbar3", foptions_->get_bool("FORM_HBAR3")}};

    // print information
    print_selected_options("Calculation Information", calculation_info_string,
                           calculation_info_bool, calculation_info_double, calculation_info_int);

    if (foptions_->get_bool("MEMORY_SUMMARY")) {
        BTF_->print_memory_info();
    }
}

void DSRG_MRPT2::cleanup() {}

double DSRG_MRPT2::compute_ref() {
    local_timer timer;
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

    outfile->Printf("  Done. Timing %15.6f s", timer.get());
    return E + Efrzc_ + Enuc_;
}

double DSRG_MRPT2::compute_energy() {
    // check semi-canonical orbitals
    semi_canonical_ = check_semi_orbs();
    if (!semi_canonical_) {
        outfile->Printf("\n    Orbital invariant formalism will be employed for DSRG-MRPT2.");
        U_ = ambit::BlockedTensor::build(tensor_type_, "U", spin_cases({"gg"}));
        std::vector<std::vector<double>> eigens = diagonalize_Fock_diagblocks(U_);
        Fa_ = eigens[0];
        Fb_ = eigens[1];
    }

    local_timer DSRG_energy;
    print_h2("Computing DSRG-MRPT2 ...");

    // Compute T2 and T1
    T1_ = BTF_->build(tensor_type_, "T1 Amplitudes", spin_cases({"hp"}));
    T2_ = BTF_->build(tensor_type_, "T2 Amplitudes", spin_cases({"hhpp"}));
    compute_t2();
    compute_t1();

    // Compute effective integrals
    if (foptions_->get_bool("DSRGPT")) {
        renormalize_V();
        renormalize_F();
    } else {
        outfile->Printf("\n    Ignore [H0th, A1st] in DSRG-MRPT2!");
    }
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

    if (foptions_->get_str("THREEPDC") != "ZERO") {
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
    print_h2("Excitation Amplitudes Summary");
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

    print_h2("Possible Intruders");
    print_intruder("A", lt1a_);
    print_intruder("B", lt1b_);
    print_intruder("AA", lt2aa_);
    print_intruder("AB", lt2ab_);
    print_intruder("BB", lt2bb_);

    // Print energy summary
    print_h2("DSRG-MRPT2 Energy Summary");
    for (auto& str_dim : energy) {
        outfile->Printf("\n    %-30s = %22.15f", str_dim.first.c_str(), str_dim.second);
    }

    psi::Process::environment.globals["UNRELAXED ENERGY"] = Etotal;
    psi::Process::environment.globals["CURRENT ENERGY"] = Etotal;
    outfile->Printf("\n\n  Energy took %10.3f s", DSRG_energy.get());
    outfile->Printf("\n");

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

        if (foptions_->get_bool("FORM_HBAR3")) {
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

    // transform dipole integrals
    if (do_dm_) {
        print_h2("Transforming Dipole Integrals ... ");
        Mbar0_ = {dm_ref_[0], dm_ref_[1], dm_ref_[2]};
        for (int i = 0; i < 3; ++i) {
            local_timer timer;
            std::string name = "Computing direction " + dm_dirs_[i];
            outfile->Printf("\n    %-30s ...", name.c_str());

            if (relax_ref_ != "NONE" || multi_state_) {
                Mbar1_[i]["uv"] = dm_[i]["uv"];
                Mbar1_[i]["UV"] = dm_[i]["UV"];
            }

            if (do_dm_dirs_[i] || multi_state_) {
                if (foptions_->get_bool("FORM_MBAR3")) {
                    compute_dm1d_pt2(dm_[i], Mbar0_[i], Mbar1_[i], Mbar2_[i], Mbar3_[i]);
                } else {
                    compute_dm1d_pt2(dm_[i], Mbar0_[i], Mbar1_[i], Mbar2_[i]);
                }
            }

            outfile->Printf("  Done. Timing %15.6f s", timer.get());
        }
        print_dm_pt2();
    }

    return Etotal;
}

void DSRG_MRPT2::compute_t2() {
    local_timer timer;
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
            if (std::fabs(value) > 1.0e-15) {
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
    std::string internal_amp = foptions_->get_str("INTERNAL_AMP");
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
    bool print_denom = foptions_->get_bool("PRINT_DENOM2");

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
                D *= 1.0 + dsrg_source_->compute_renormalized(Fa_[i[0]] + Fa_[i[1]] - Fa_[i[2]] -
                                                              Fa_[i[3]]);
                myfile << i[0] << " " << i[1] << " " << i[2] << " " << i[3] << " " << D << " \n";
            }
        });
    }

    outfile->Printf("  Done. Timing %15.6f s", timer.get());
}

void DSRG_MRPT2::compute_t1() {
    local_timer timer;
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
            if (std::fabs(value) > 1.0e-15) {
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
    std::string internal_amp = foptions_->get_str("INTERNAL_AMP");
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
    local_timer timer;
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
        if (std::fabs(value) > 1.0e-15) {
            if ((spin[0] == AlphaSpin) and (spin[1] == AlphaSpin)) {
                value *= 1.0 + dsrg_source_->compute_renormalized(Fa_[i[0]] + Fa_[i[1]] -
                                                                  Fa_[i[2]] - Fa_[i[3]]);
            } else if ((spin[0] == AlphaSpin) and (spin[1] == BetaSpin)) {
                value *= 1.0 + dsrg_source_->compute_renormalized(Fa_[i[0]] + Fb_[i[1]] -
                                                                  Fa_[i[2]] - Fb_[i[3]]);
            } else if ((spin[0] == BetaSpin) and (spin[1] == BetaSpin)) {
                value *= 1.0 + dsrg_source_->compute_renormalized(Fb_[i[0]] + Fb_[i[1]] -
                                                                  Fb_[i[2]] - Fb_[i[3]]);
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
    local_timer timer;
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
            if (std::fabs(value) > 1.0e-15) {
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
    local_timer timer;
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
    local_timer timer;
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
    local_timer timer;
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
    local_timer timer;
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
    local_timer timer;
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
    local_timer timer;
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
    local_timer timer;
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
    local_timer timer;
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
    E += 0.25 * temp.block("aaaaaa")("uvwxyz") * rdms_.L3aaa()("xyzuvw");

    // bbb
    temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"AAAAAA"});
    temp["UVWXYZ"] += V_["UVMZ"] * T2_["MWXY"];
    temp["UVWXYZ"] += V_["WEXY"] * T2_["UVEZ"];

    if (internal_amp_) {
        temp["UVWXYZ"] += V_["UV!Z"] * T2_["!WXY"];
        temp["UVWXYZ"] += V_["W!XY"] * T2_["UV!Z"];
    }
    E += 0.25 * temp.block("AAAAAA")("UVWXYZ") * rdms_.L3bbb()("XYZUVW");

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
    E += 0.5 * temp.block("aaAaaA")("uvWxyZ") * rdms_.L3aab()("xyZuvW");

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
    E += 0.5 * temp.block("aAAaAA")("uVWxYZ") * rdms_.L3abb()("xYZuVW");

    outfile->Printf("  Done. Timing %15.6f s", timer.get());
    dsrg_time_.add("220", timer.get());
    return E;
}

void DSRG_MRPT2::print_dm_pt2() {
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

    psi::Process::environment.globals["UNRELAXED DIPOLE X"] = x;
    psi::Process::environment.globals["UNRELAXED DIPOLE Y"] = y;
    psi::Process::environment.globals["UNRELAXED DIPOLE Z"] = z;
    psi::Process::environment.globals["UNRELAXED DIPOLE"] = t;
}

void DSRG_MRPT2::compute_dm1d_pt2(BlockedTensor& M, double& Mbar0, BlockedTensor& Mbar1,
                                  BlockedTensor& Mbar2) {
    /// Mbar = M + [M, A] + 0.5 * [[M, A], A]

    //    BlockedTensor D1 = BTF_->build(tensor_type_, "D1", spin_cases({"gg"}), true);
    //    for (const auto& block : {"cc", "CC"}) {
    //        D1.block(block).iterate([&](const std::vector<size_t>& i, double& value) {
    //            if (i[0] == i[1]) {
    //                value = 1.0;
    //            }
    //        });
    //    }
    //    D1["uv"] = Gamma1_["uv"];
    //    D1["UV"] = Gamma1_["UV"];

    //    D1["ij"] -= 0.5 * T2_["jnab"] * T2_["inab"];
    //    D1["ij"] -= T2_["jNaB"] * T2_["iNaB"];
    //    D1["IJ"] -= 0.5 * T2_["JNAB"] * T2_["INAB"];
    //    D1["IJ"] -= T2_["nJaB"] * T2_["nIaB"];

    //    D1["ab"] += 0.5 * T2_["mnbc"] * T2_["mnac"];
    //    D1["ab"] += T2_["mNbC"] * T2_["mNaC"];
    //    D1["AB"] += 0.5 * T2_["MNBC"] * T2_["MNAC"];
    //    D1["AB"] += T2_["mNcB"] * T2_["mNcA"];

    //    // transform D1 with a irrep psi::SharedMatrix
    //    psi::SharedMatrix SOdens(new psi::Matrix("SO density ", this->nmopi(), this->nmopi()));

    //    for (const auto& pair: mo_space_info_->relative_mo("FROZEN_DOCC")) {
    //        size_t h = pair.first;
    //        size_t i = pair.second;
    //        SOdens->set(h, i, i, 1.0);
    //    }

    //    std::map<size_t, std::pair<size_t, size_t>> momap;
    //    for (const std::string& block: {"RESTRICTED_DOCC", "ACTIVE", "RESTRICTED_UOCC"}) {
    //        size_t size = mo_space_info_->size(block);
    //        for (size_t i = 0; i < size; ++i) {
    //            momap[mo_space_info_->corr_absolute_mo(block)[i]] =
    //            mo_space_info_->relative_mo(block)[i];
    //        }
    //    }

    //    D1.citerate(
    //        [&](const std::vector<size_t>& i, const std::vector<SpinType>& spin, const double&
    //        value) {
    //            if (spin[0] == AlphaSpin) {
    //                size_t h0 = momap[i[0]].first;
    //                size_t m0 = momap[i[0]].second;
    //                size_t h1 = momap[i[1]].first;
    //                size_t m1 = momap[i[1]].second;
    //                if (h0 == h1) {
    //                    SOdens->set(h0, m0, m1, value);
    //                }
    //            }
    //        });

    //    SOdens->back_transform(this->Ca());

    //    psi::SharedMatrix sotoao(this->aotoso()->transpose());
    //    size_t nao = sotoao->coldim(0);
    //    psi::SharedMatrix AOdens(new psi::Matrix("AO density ", nao, nao));
    //    AOdens->remove_symmetry(SOdens, sotoao);

    //    std::vector<psi::SharedMatrix> aodipole_ints = ints_->ao_dipole_ints();
    //    std::vector<double> de(4, 0.0);
    //    for (int i = 0; i < 3; ++i) {
    //        de[i] = 2.0 * AOdens->vector_dot(aodipole_ints[i]); // 2.0 for beta spin
    //        de[3] += de[i] * de[i];
    //    }
    //    de[3] = sqrt(de[3]);

    //    outfile->Printf("\n  Permanent dipole moments (a.u.):  X: %7.5f  Y: "
    //                    "%7.5f  Z: %7.5f  Total: %7.5f", de[0], de[1], de[2], de[3]);

    //    double correct = 0.0;
    //    correct -= 0.5 * M["ji"] * T2_["jnab"] * T2_["inab"];
    //    correct -= M["ji"] * T2_["jNaB"] * T2_["iNaB"];
    //    correct -= M["JI"] * T2_["nJaB"] * T2_["nIaB"];
    //    correct -= 0.5 * M["JI"] * T2_["JNAB"] * T2_["INAB"];

    //    correct += 0.5 * M["ab"] * T2_["mnbc"] * T2_["mnac"];
    //    correct += M["ab"] * T2_["mNbC"] * T2_["mNaC"];
    //    correct += M["AB"] * T2_["mNcB"] * T2_["mNcA"];
    //    correct += 0.5 * M["AB"] * T2_["MNBC"] * T2_["MNAC"];
    //    outfile->Printf("\n  Correct value: %.15f", correct);

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

void DSRG_MRPT2::compute_dm1d_pt2(BlockedTensor& M, double& Mbar0, BlockedTensor& Mbar1,
                                  BlockedTensor& Mbar2, BlockedTensor& Mbar3) {
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

    // cases when we need Mbar1, Mbar2, and Mbar3
    if (relax_ref_ != "NONE" || multi_state_) {
        BlockedTensor C1, C2, C3;
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

        C3 = BTF_->build(tensor_type_, "C3", spin_cases({"aaaaaa"}), true);

        /// This needs to be checked
        H2_T2_C3(O2, T2_, 0.5, C3);

        Mbar3["uvwxyz"] += C3["uvwxyz"];
        Mbar3["uvwxyz"] += C3["xyzuvw"];
        Mbar3["uvWxyZ"] += C3["uvWxyZ"];
        Mbar3["uvWxyZ"] += C3["xyZuvW"];
        Mbar3["uVWxYZ"] += C3["uVWxYZ"];
        Mbar3["uVWxYZ"] += C3["uVWxYZ"];
        Mbar3["UVWXYZ"] += C3["UVWXYZ"];
        Mbar3["UVWXYZ"] += C3["UVWXYZ"];
    }
}

// double DSRG_MRPT2::compute_energy_relaxed() {
//    double Edsrg = 0.0, Erelax = 0.0;

//    // compute energy with fixed ref.
//    Edsrg = compute_energy();

//    // unrelaxed dipole from compute_energy
//    std::vector<double> dm_dsrg(Mbar0_);
//    std::map<std::string, std::vector<double>> dm_relax;

//    // obtain the all-active DSRG transformed Hamiltonian
//    auto fci_ints = compute_Heff_actv();

//    size_t nroot = foptions_->get_int("NROOT");

//    // diagonalize Hbar depending on CAS_TYPE
//    if (foptions_->get_str("CAS_TYPE") == "CAS") {
//        auto state = make_state_info_from_psi(ints_->wfn());
//        FCI_MO fci_mo(state, nroot, scf_info_, foptions_, mo_space_info_, fci_ints);
//        fci_mo.set_localize_actv(false);
//        Erelax = fci_mo.compute_energy();

//        if (do_dm_) {
//            // de-normal-order DSRG dipole integrals
//            for (int z = 0; z < 3; ++z) {
//                if (do_dm_dirs_[z]) {
//                    std::string name = "Dipole " + dm_dirs_[z] + " Integrals";
//                    if (foptions_->get_bool("FORM_MBAR3")) {
//                        deGNO_ints(name, Mbar0_[z], Mbar1_[z], Mbar2_[z], Mbar3_[z]);
//                        rotate_ints_semi_to_origin(name, Mbar1_[z], Mbar2_[z], Mbar3_[z]);
//                    } else {
//                        deGNO_ints(name, Mbar0_[z], Mbar1_[z], Mbar2_[z]);
//                        rotate_ints_semi_to_origin(name, Mbar1_[z], Mbar2_[z]);
//                    }
//                }
//            }

//            // compute permanent dipoles
//            if (foptions_->get_bool("FORM_MBAR3")) {
//                dm_relax = fci_mo.compute_ref_relaxed_dm(Mbar0_, Mbar1_, Mbar2_, Mbar3_);
//            } else {
//                dm_relax = fci_mo.compute_ref_relaxed_dm(Mbar0_, Mbar1_, Mbar2_);
//            }
//        }
//    } else if (foptions_->get_str("CAS_TYPE") == "ACI") {

//        auto state = make_state_info_from_psi(ints_->wfn());
//        AdaptiveCI aci(state, nroot, scf_info_, foptions_, mo_space_info_, fci_ints);

//        Erelax = aci.compute_energy();
//    } else {
//        auto state = make_state_info_from_psi(ints_->wfn());
//        auto fci = make_active_space_method("FCI", state, nroot, scf_info_, mo_space_info_, ints_,
//                                            foptions_);
//        fci->set_active_space_integrals(fci_ints);
//        Erelax = fci->compute_energy();
//    }

//    // printing
//    print_h2("DSRG-MRPT2 Energy Summary");
//    outfile->Printf("\n    %-30s = %22.15f", "DSRG-MRPT2 Total Energy (fixed)  ", Edsrg);
//    outfile->Printf("\n    %-30s = %22.15f\n", "DSRG-MRPT2 Total Energy (relaxed)", Erelax);

//    if (do_dm_) {
//        print_h2("DSRG-MRPT2 Dipole Moment Summary");
//        const double& nx = dm_nuc_[0];
//        const double& ny = dm_nuc_[1];
//        const double& nz = dm_nuc_[2];

//        double x = dm_dsrg[0] + nx;
//        double y = dm_dsrg[1] + ny;
//        double z = dm_dsrg[2] + nz;
//        double t = std::sqrt(x * x + y * y + z * z);
//        outfile->Printf("\n    DSRG-MRPT2 unrelaxed dipole moment:");
//        outfile->Printf("\n      X: %10.6f  Y: %10.6f  Z: %10.6f  Total: %10.6f\n", x, y, z, t);
//        psi::Process::environment.globals["UNRELAXED DIPOLE"] = t;

//        // there should be only one entry for state-specific computations
//        if (dm_relax.size() == 1) {
//            for (const auto& p : dm_relax) {
//                x = p.second[0] + nx;
//                y = p.second[1] + ny;
//                z = p.second[2] + nz;
//                t = std::sqrt(x * x + y * y + z * z);
//            }
//            outfile->Printf("\n    DSRG-MRPT2 partially relaxed dipole moment:");
//            outfile->Printf("\n      X: %10.6f  Y: %10.6f  Z: %10.6f  Total: %10.6f\n", x, y, z,
//            t); psi::Process::environment.globals["PARTIALLY RELAXED DIPOLE"] = t;
//        }
//    }

//    psi::Process::environment.globals["UNRELAXED ENERGY"] = Edsrg;
//    psi::Process::environment.globals["PARTIALLY RELAXED ENERGY"] = Erelax;
//    psi::Process::environment.globals["CURRENT ENERGY"] = Erelax;
//    return Erelax;
//}

// ambit::BlockedTensor DSRG_MRPT2::compute_OE_density(BlockedTensor& T1, BlockedTensor& T2,
//                                                    BlockedTensor& D1, BlockedTensor& D2,
//                                                    BlockedTensor& D3, const bool& transition) {
//    BlockedTensor O = BTF_->build(tensor_type_, "OE D1", spin_cases({"gg"}));
//    BlockedTensor temp;

//    if (!transition) {
//        // copy the reference D1 to O if not transition one density
//        for (const auto& block : {"cc", "CC"}) {
//            O.block(block).iterate([&](const std::vector<size_t>& i, double& value) {
//                if (i[0] == i[1]) {
//                    value = 1.0;
//                }
//            });
//        }
//        O["uv"] = D1["uv"];
//        O["UV"] = D1["UV"];

//        // compute fully contracted term

//        // im/mi
//        temp = BTF_->build(tensor_type_, "temp", spin_cases({"hc"}));
//        temp["im"] -= 0.5 * T1["ma"] * T1["ia"];
//        temp["im"] -= 0.25 * T2["mnab"] * T2["inab"];
//        temp["im"] -= 0.5 * T2["mNaB"] * T2["iNaB"];

//        temp["IM"] -= 0.5 * T1["MA"] * T1["IA"];
//        temp["IM"] -= 0.25 * T2["MNAB"] * T2["INAB"];
//        temp["IM"] -= 0.5 * T2["nMaB"] * T2["nIaB"];

//        O["im"] += temp["im"];
//        O["IM"] += temp["IM"];
//        O["mi"] += temp["im"];
//        O["MI"] += temp["IM"];

//        // am/ma
//        temp = BTF_->build(tensor_type_, "temp", spin_cases({"pc"}));
//        temp["am"] = T1["ma"];
//        temp["am"] += 0.5 * T1["mu"] * T1["ua"];
//        temp["am"] += 0.5 * T1["nb"] * T2["mnab"];
//        temp["am"] += 0.5 * T1["NB"] * T2["mNaB"];

//        temp["AM"] = T1["MA"];
//        temp["AM"] += 0.5 * T1["MU"] * T1["UA"];
//        temp["AM"] += 0.5 * T1["NB"] * T2["MNAB"];
//        temp["AM"] += 0.5 * T1["nb"] * T2["nMbA"];

//        O["am"] += temp["am"];
//        O["AM"] += temp["AM"];
//        O["ma"] += temp["am"];
//        O["MA"] += temp["AM"];

//        // ab
//        O["ba"] = T1["mb"] * T1["ma"];
//        O["ba"] += 0.5 * T2["mnbc"] * T2["mnac"];
//        O["ba"] += T2["mNbC"] * T2["mNaC"];

//        O["BA"] = T1["MB"] * T1["MA"];
//        O["BA"] += 0.5 * T2["MNBC"] * T2["MNAC"];
//        O["BA"] += T2["mNcB"] * T2["mNcA"];
//    }

//    // ui/iu with D1
//    temp = BTF_->build(tensor_type_, "temp", spin_cases({"ha"}));
//    temp["iv"] -= T1["iv"];
//    temp["iv"] -= 0.5 * T1["va"] * T1["ia"];
//    temp["iv"] -= 0.5 * T1["ma"] * T2["imva"];
//    temp["iv"] -= 0.5 * T1["MA"] * T2["iMvA"];
//    temp["iv"] -= 0.25 * T2["vmab"] * T2["imab"];
//    temp["iv"] -= 0.5 * T2["vMaB"] * T2["iMaB"];

//    temp["IV"] -= T1["IV"];
//    temp["IV"] -= 0.5 * T1["VA"] * T1["IA"];
//    temp["IV"] -= 0.5 * T1["ma"] * T2["mIaV"];
//    temp["IV"] -= 0.5 * T1["MA"] * T2["IMVA"];
//    temp["IV"] -= 0.25 * T2["VMAB"] * T2["IMAB"];
//    temp["IV"] -= 0.5 * T2["mVaB"] * T2["mIaB"];

//    if (internal_amp_) {
//        temp["iw"] += 0.5 * T1["iv"] * T1["vw"];
//        temp["IW"] += 0.5 * T1["IV"] * T1["VW"];
//    }

//    O["iu"] += temp["iv"] * D1["vu"];
//    O["IU"] += temp["IV"] * D1["VU"];
//    O["ui"] += temp["iv"] * D1["uv"];
//    O["UI"] += temp["IV"] * D1["UV"];

//    // ua/ua with D1
//    temp = BTF_->build(tensor_type_, "temp", spin_cases({"pa"}));
//    temp["av"] = T1["va"];
//    temp["av"] -= 0.5 * T1["iv"] * T1["ia"];
//    temp["av"] -= 0.5 * T1["mb"] * T2["mvab"];
//    temp["av"] += 0.5 * T1["MB"] * T2["vMaB"];
//    temp["av"] -= 0.25 * T2["mnab"] * T2["mnvb"];
//    temp["av"] -= 0.5 * T2["mNaB"] * T2["mNvB"];

//    temp["AV"] = T1["VA"];
//    temp["AV"] -= 0.5 * T1["IV"] * T1["IA"];
//    temp["AV"] -= 0.5 * T1["MB"] * T2["MVAB"];
//    temp["av"] += 0.5 * T1["mb"] * T2["mVbA"];
//    temp["AV"] -= 0.25 * T2["MNAB"] * T2["MNVB"];
//    temp["AV"] -= 0.5 * T2["mNbA"] * T2["mNbV"];

//    if (internal_amp_) {
//        temp["av"] += 0.5 * T1["wa"] * T1["vw"];
//        temp["av"] += 0.5 * T1["WA"] * T1["VW"];
//    }

//    O["au"] += temp["av"] * D1["vu"];
//    O["AU"] += temp["AV"] * D1["VU"];
//    O["ua"] += temp["av"] * D1["uv"];
//    O["UA"] += temp["AV"] * D1["UV"];

//    // mi/im with D1
//    temp = BTF_->build(tensor_type_, "temp", {"aahc"});
//    temp["uvim"] += 0.5 * T1["ia"] * T2["mvua"];
//    temp["uvim"] += 0.5 * T1["ma"] * T2["iuva"];
//    temp["uvim"] -= 0.25 * T2["mvab"] * T2["iuab"];
//    temp["uvim"] += 0.5 * T2["nmua"] * T2["niva"];
//    temp["uvim"] += 0.5 * T2["mNuA"] * T2["iNvA"];

//    O["im"] += temp["uvim"] * D1["vu"];
//    O["mi"] += temp["uvim"] * D1["uv"];

//    temp = BTF_->build(tensor_type_, "temp", {"AAhc"});
//    temp["UVim"] -= 0.5 * T1["ia"] * T2["mVaU"];
//    temp["UVim"] -= 0.5 * T1["ma"] * T2["iUaV"];
//    temp["UVim"] -= 0.5 * T2["mVaB"] * T2["iUaB"];
//    temp["UVim"] += 0.5 * T2["mNaU"] * T2["iNaV"];

//    O["im"] += temp["UVim"] * D1["VU"];
//    O["mi"] += temp["UVim"] * D1["UV"];

//    temp = BTF_->build(tensor_type_, "temp", {"AAHC"});
//    temp["UVIM"] += 0.5 * T1["IA"] * T2["MVUA"];
//    temp["UVIM"] += 0.5 * T1["MA"] * T2["IUVA"];
//    temp["UVIM"] -= 0.25 * T2["MVAB"] * T2["IUAB"];
//    temp["UVIM"] += 0.5 * T2["nMaU"] * T2["nIaV"];
//    temp["UVIM"] += 0.5 * T2["MNUA"] * T2["INVA"];

//    O["IM"] += temp["UVIM"] * D1["VU"];
//    O["MI"] += temp["UVIM"] * D1["UV"];

//    temp = BTF_->build(tensor_type_, "temp", {"aaHC"});
//    temp["uvIM"] -= 0.5 * T1["IA"] * T2["vMuA"];
//    temp["uvIM"] -= 0.5 * T1["MA"] * T2["uIvA"];
//    temp["uvIM"] -= 0.5 * T2["vMaB"] * T2["uIaB"];
//    temp["uvIM"] += 0.5 * T2["nMuA"] * T2["nIvA"];

//    O["IM"] += temp["uvIM"] * D1["vu"];
//    O["MI"] += temp["uvIM"] * D1["uv"];

//    // ma/am with D1
////    temp = BTF_->build(tensor_type_, "temp", {"capa"});
////    temp["muav"] = T2["muav"];
////    temp["muav"] += 0.5 * T1["ua"] * T2["mvuw"];
////    temp["muav"] += 0.5 * T1["ub"] * T2["mvab"];
////    temp["muav"] += 0.5 * T1["iu"] * T2["imav"];
////    temp["muav"] += 0.5 * T1["mu"] * T2["uvaw"];

//    return O;
//}

// void DSRG_MRPT2::transfer_integrals() {
//    // printing
//    print_h2("De-Normal-Order the DSRG Transformed Hamiltonian");

//    // compute scalar term (all active only)
//    local_timer t_scalar;
//    std::string str = "Computing the scalar term   ...";
//    outfile->Printf("\n    %-35s", str.c_str());
//    double scalar0 = Eref_ + Hbar0_ - Enuc_ - Efrzc_;

//    // scalar from Hbar1
//    double scalar1 = 0.0;
//    scalar1 -= Hbar1_["vu"] * Gamma1_["uv"];
//    scalar1 -= Hbar1_["VU"] * Gamma1_["UV"];

//    // scalar from Hbar2
//    double scalar2 = 0.0;
//    scalar2 += 0.5 * Gamma1_["uv"] * Hbar2_["vyux"] * Gamma1_["xy"];
//    scalar2 += 0.5 * Gamma1_["UV"] * Hbar2_["VYUX"] * Gamma1_["XY"];
//    scalar2 += Gamma1_["uv"] * Hbar2_["vYuX"] * Gamma1_["XY"];

//    scalar2 -= 0.25 * Hbar2_["xyuv"] * Lambda2_["uvxy"];
//    scalar2 -= 0.25 * Hbar2_["XYUV"] * Lambda2_["UVXY"];
//    scalar2 -= Hbar2_["xYuV"] * Lambda2_["uVxY"];

//    double scalar = scalar0 + scalar1 + scalar2;

//    bool form_hbar3 = foptions_->get_bool("FORM_HBAR3");
//    double scalar3 = 0.0;
//    if (form_hbar3) {
//        scalar3 -= (1.0 / 36) * Hbar3_.block("aaaaaa")("xyzuvw") * rdms_.L3aaa()("xyzuvw");
//        scalar3 -= (1.0 / 36) * Hbar3_.block("AAAAAA")("XYZUVW") * rdms_.L3bbb()("XYZUVW");
//        scalar3 -= 0.25 * Hbar3_.block("aaAaaA")("xyZuvW") * rdms_.L3aab()("xyZuvW");
//        scalar3 -= 0.25 * Hbar3_.block("aAAaAA")("xYZuVW") * rdms_.L3abb()("xYZuVW");

//        scalar3 += 0.25 * Hbar3_["xyzuvw"] * Gamma1_["wz"] * Lambda2_["uvxy"];
//        scalar3 += 0.25 * Hbar3_["XYZUVW"] * Gamma1_["WZ"] * Lambda2_["UVXY"];
//        scalar3 += 0.25 * Hbar3_["xyZuvW"] * Gamma1_["WZ"] * Lambda2_["uvxy"];
//        scalar3 += Hbar3_["xzYuwV"] * Gamma1_["wz"] * Lambda2_["uVxY"];
//        scalar3 += 0.25 * Hbar3_["zXYwUV"] * Gamma1_["wz"] * Lambda2_["UVXY"];
//        scalar3 += Hbar3_["xZYuWV"] * Gamma1_["WZ"] * Lambda2_["uVxY"];

//        scalar3 -= (1.0 / 6) * Hbar3_["xyzuvw"] * Gamma1_["ux"] * Gamma1_["vy"] * Gamma1_["wz"];
//        scalar3 -= (1.0 / 6) * Hbar3_["XYZUVW"] * Gamma1_["UX"] * Gamma1_["VY"] * Gamma1_["WZ"];
//        scalar3 -= 0.5 * Hbar3_["xyZuvW"] * Gamma1_["ux"] * Gamma1_["vy"] * Gamma1_["WZ"];
//        scalar3 -= 0.5 * Hbar3_["xYZuVW"] * Gamma1_["ux"] * Gamma1_["VY"] * Gamma1_["WZ"];

//        scalar += scalar3;
//    }

//    outfile->Printf("  Done. Timing %10.3f s", t_scalar.get());

//    // compute one-body term
//    local_timer t_one;
//    str = "Computing the one-body term ...";
//    outfile->Printf("\n    %-35s", str.c_str());
//    BlockedTensor temp1 = BTF_->build(tensor_type_, "temp1", spin_cases({"aa"}));
//    temp1["uv"] = Hbar1_["uv"];
//    temp1["UV"] = Hbar1_["UV"];
//    temp1["uv"] -= Hbar2_["uxvy"] * Gamma1_["yx"];
//    temp1["uv"] -= Hbar2_["uXvY"] * Gamma1_["YX"];
//    temp1["UV"] -= Hbar2_["xUyV"] * Gamma1_["yx"];
//    temp1["UV"] -= Hbar2_["UXVY"] * Gamma1_["YX"];

//    if (form_hbar3) {
//        temp1["uv"] += 0.5 * Hbar3_["uyzvxw"] * Gamma1_["xy"] * Gamma1_["wz"];
//        temp1["uv"] += 0.5 * Hbar3_["uYZvXW"] * Gamma1_["XY"] * Gamma1_["WZ"];
//        temp1["uv"] += Hbar3_["uyZvxW"] * Gamma1_["xy"] * Gamma1_["WZ"];

//        temp1["UV"] += 0.5 * Hbar3_["UYZVXW"] * Gamma1_["XY"] * Gamma1_["WZ"];
//        temp1["UV"] += 0.5 * Hbar3_["yzUxwV"] * Gamma1_["xy"] * Gamma1_["wz"];
//        temp1["UV"] += Hbar3_["yUZxVW"] * Gamma1_["xy"] * Gamma1_["WZ"];

//        temp1["uv"] -= 0.25 * Hbar3_["uxyvwz"] * Lambda2_["wzxy"];
//        temp1["uv"] -= 0.25 * Hbar3_["uXYvWZ"] * Lambda2_["WZXY"];
//        temp1["uv"] -= Hbar3_["uxYvwZ"] * Lambda2_["wZxY"];

//        temp1["UV"] -= 0.25 * Hbar3_["UXYVWZ"] * Lambda2_["WZXY"];
//        temp1["UV"] -= 0.25 * Hbar3_["xyUwzV"] * Lambda2_["wzxy"];
//        temp1["UV"] -= Hbar3_["xUYwVZ"] * Lambda2_["wZxY"];
//    }

//    outfile->Printf("  Done. Timing %10.3f s", t_one.get());

//    // compute two-body term
//    BlockedTensor temp2;
//    if (form_hbar3) {
//        temp2 = BTF_->build(tensor_type_, "temp2", spin_cases({"aaaa"}));
//        str = "Computing the two-body term ...";
//        outfile->Printf("\n    %-35s", str.c_str());

//        temp2["uvxy"] = Hbar2_["uvxy"];
//        temp2["uVxY"] = Hbar2_["uVxY"];
//        temp2["UVXY"] = Hbar2_["UVXY"];

//        temp2["xyuv"] -= Hbar3_["xyzuvw"] * Gamma1_["wz"];
//        temp2["xyuv"] -= Hbar3_["xyZuvW"] * Gamma1_["WZ"];
//        temp2["xYuV"] -= Hbar3_["xYZuVW"] * Gamma1_["WZ"];
//        temp2["xYuV"] -= Hbar3_["xzYuwV"] * Gamma1_["wz"];
//        temp2["XYUV"] -= Hbar3_["XYZUVW"] * Gamma1_["WZ"];
//        temp2["XYUV"] -= Hbar3_["zXYwUV"] * Gamma1_["wz"];

//        outfile->Printf("  Done. Timing %10.3f s", t_one.get());
//    }

//    // update integrals
//    local_timer t_int;
//    str = "Updating integrals          ...";
//    outfile->Printf("\n    %-35s", str.c_str());
//    //    ints_->set_scalar(Edsrg - Enuc_ - Efrzc_);
//    ints_->set_scalar(scalar);

//    //   a) zero hole integrals
//    std::vector<size_t> hole_mos = core_mos_;
//    hole_mos.insert(hole_mos.end(), actv_mos_.begin(), actv_mos_.end());
//    for (const size_t& i : hole_mos) {
//        for (const size_t& j : hole_mos) {
//            ints_->set_oei(i, j, 0.0, true);
//            ints_->set_oei(i, j, 0.0, false);
//            for (const size_t& k : hole_mos) {
//                for (const size_t& l : hole_mos) {
//                    ints_->set_tei(i, j, k, l, 0.0, true, true);
//                    ints_->set_tei(i, j, k, l, 0.0, true, false);
//                    ints_->set_tei(i, j, k, l, 0.0, false, false);
//                }
//            }
//        }
//    }

//    //   b) copy all active part
//    //    Hbar1_.citerate(
//    //        [&](const std::vector<size_t>& i, const std::vector<SpinType>& spin, const double&
//    //        value) {
//    //            if (spin[0] == AlphaSpin) {
//    //                ints_->set_oei(i[0], i[1], value, true);
//    //            } else {
//    //                ints_->set_oei(i[0], i[1], value, false);
//    //            }
//    //        });
//    temp1.citerate(
//        [&](const std::vector<size_t>& i, const std::vector<SpinType>& spin, const double& value)
//        {
//            if (spin[0] == AlphaSpin) {
//                ints_->set_oei(i[0], i[1], value, true);
//            } else {
//                ints_->set_oei(i[0], i[1], value, false);
//            }
//        });

//    if (!form_hbar3) {
//        Hbar2_.citerate([&](const std::vector<size_t>& i, const std::vector<SpinType>& spin,
//                            const double& value) {
//            if ((spin[0] == AlphaSpin) && (spin[1] == AlphaSpin)) {
//                ints_->set_tei(i[0], i[1], i[2], i[3], value, true, true);
//            } else if ((spin[0] == AlphaSpin) && (spin[1] == BetaSpin)) {
//                ints_->set_tei(i[0], i[1], i[2], i[3], value, true, false);
//            } else if ((spin[0] == BetaSpin) && (spin[1] == BetaSpin)) {
//                ints_->set_tei(i[0], i[1], i[2], i[3], value, false, false);
//            }
//        });
//    } else {
//        temp2.citerate([&](const std::vector<size_t>& i, const std::vector<SpinType>& spin,
//                           const double& value) {
//            if ((spin[0] == AlphaSpin) && (spin[1] == AlphaSpin)) {
//                ints_->set_tei(i[0], i[1], i[2], i[3], value, true, true);
//            } else if ((spin[0] == AlphaSpin) && (spin[1] == BetaSpin)) {
//                ints_->set_tei(i[0], i[1], i[2], i[3], value, true, false);
//            } else if ((spin[0] == BetaSpin) && (spin[1] == BetaSpin)) {
//                ints_->set_tei(i[0], i[1], i[2], i[3], value, false, false);
//            }
//        });
//    }

//    outfile->Printf("  Done. Timing %10.3f s", t_int.get());

//    // print scalar
//    double scalar_include_fc = scalar + ints_->frozen_core_energy();
//    print_h2("Scalar of the DSRG Hamiltonian (WRT True Vacuum)");
//    outfile->Printf("\n    %-30s = %22.15f", "Scalar0", scalar0);
//    outfile->Printf("\n    %-30s = %22.15f", "Scalar1", scalar1);
//    outfile->Printf("\n    %-30s = %22.15f", "Scalar2", scalar2);
//    if (form_hbar3) {
//        outfile->Printf("\n    %-30s = %22.15f", "Scalar3", scalar3);
//    }
//    outfile->Printf("\n    %-30s = %22.15f", "Total Scalar W/O Frozen-Core", scalar);
//    outfile->Printf("\n    %-30s = %22.15f", "Total Scalar W/  Frozen-Core", scalar_include_fc);

//    // test if de-normal-ordering is correct
//    print_h2("Test De-Normal-Ordered Hamiltonian");
//    double Etest = scalar_include_fc + Enuc_;

//    double Etest1 = 0.0;
//    if (!form_hbar3) {
//        Etest1 += temp1["uv"] * Gamma1_["vu"];
//        Etest1 += temp1["UV"] * Gamma1_["VU"];

//        Etest1 += Hbar1_["uv"] * Gamma1_["vu"];
//        Etest1 += Hbar1_["UV"] * Gamma1_["VU"];
//        Etest1 *= 0.5;
//    } else {
//        Etest1 += temp1["uv"] * Gamma1_["vu"];
//        Etest1 += temp1["UV"] * Gamma1_["VU"];
//    }

//    double Etest2 = 0.0;
//    Etest2 += 0.25 * Hbar2_["uvxy"] * Lambda2_["xyuv"];
//    Etest2 += 0.25 * Hbar2_["UVXY"] * Lambda2_["XYUV"];
//    Etest2 += Hbar2_["uVxY"] * Lambda2_["xYuV"];

//    if (form_hbar3) {
//        Etest2 += 0.5 * temp2["xyuv"] * Gamma1_["ux"] * Gamma1_["vy"];
//        Etest2 += 0.5 * temp2["XYUV"] * Gamma1_["UX"] * Gamma1_["VY"];
//        Etest2 += temp2["xYuV"] * Gamma1_["ux"] * Gamma1_["VY"];
//    }

//    Etest += Etest1 + Etest2;
//    outfile->Printf("\n    %-30s = %22.15f", "One-Body Energy (after)", Etest1);
//    outfile->Printf("\n    %-30s = %22.15f", "Two-Body Energy (after)", Etest2);

//    if (form_hbar3) {
//        double Etest3 = 0.0;
//        Etest3 += (1.0 / 6) * Hbar3_["xyzuvw"] * Gamma1_["ux"] * Gamma1_["vy"] * Gamma1_["wz"];
//        Etest3 += (1.0 / 6) * Hbar3_["XYZUVW"] * Gamma1_["UX"] * Gamma1_["VY"] * Gamma1_["WZ"];
//        Etest3 += 0.5 * Hbar3_["xyZuvW"] * Gamma1_["ux"] * Gamma1_["vy"] * Gamma1_["WZ"];
//        Etest3 += 0.5 * Hbar3_["xYZuVW"] * Gamma1_["ux"] * Gamma1_["VY"] * Gamma1_["WZ"];

//        Etest3 += (1.0 / 36) * Hbar3_.block("aaaaaa")("xyzuvw") * rdms_.L3aaa()("xyzuvw");
//        Etest3 += (1.0 / 36) * Hbar3_.block("AAAAAA")("XYZUVW") * rdms_.L3bbb()("XYZUVW");
//        Etest3 += 0.25 * Hbar3_.block("aaAaaA")("xyZuvW") * rdms_.L3aab()("xyZuvW");
//        Etest3 += 0.25 * Hbar3_.block("aAAaAA")("xYZuVW") * rdms_.L3abb()("xYZuVW");

//        outfile->Printf("\n    %-30s = %22.15f", "Three-Body Energy (after)", Etest3);
//        Etest += Etest3;
//    }

//    outfile->Printf("\n    %-30s = %22.15f", "Total Energy (after)", Etest);
//    outfile->Printf("\n    %-30s = %22.15f", "Total Energy (before)", Eref_ + Hbar0_);

//    if (std::fabs(Etest - Eref_ - Hbar0_) > 100.0 * foptions_->get_double("E_CONVERGENCE")) {
//        throw psi::PSIEXCEPTION("De-normal-odering failed.");
//    }
//}

void DSRG_MRPT2::H1_T1_C1aa(BlockedTensor& H1, BlockedTensor& T1, const double& alpha,
                            BlockedTensor& C1) {
    local_timer timer;

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
    local_timer timer;

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
    local_timer timer;

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
    local_timer timer;
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
    local_timer timer;

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
    local_timer timer;

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
    local_timer timer;

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
    local_timer timer;

    // compute only all active !

    // aaa
    BlockedTensor temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"aaaaaa"});
    temp["xyzuvw"] -= alpha * H2["xywi"] * T2["izuv"];
    temp["xyzuvw"] += alpha * H2["azuv"] * T2["xywa"];
    C3["xyzuvw"] += temp["xyzuvw"];
    C3["zxyuvw"] += temp["xyzuvw"];
    C3["xzyuvw"] -= temp["xyzuvw"];
    C3["xyzwuv"] += temp["xyzuvw"];
    C3["zxywuv"] += temp["xyzuvw"];
    C3["xzywuv"] -= temp["xyzuvw"];
    C3["xyzuwv"] -= temp["xyzuvw"];
    C3["zxyuwv"] -= temp["xyzuvw"];
    C3["xzyuwv"] += temp["xyzuvw"];

    // bbb
    temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"AAAAAA"});
    temp["XYZUVW"] -= alpha * H2["XYWI"] * T2["IZUV"];
    temp["XYZUVW"] += alpha * H2["AZUV"] * T2["XYWA"];
    C3["XYZUVW"] += temp["XYZUVW"];
    C3["ZXYUVW"] += temp["XYZUVW"];
    C3["XZYUVW"] -= temp["XYZUVW"];
    C3["XYZWUV"] += temp["XYZUVW"];
    C3["ZXYWUV"] += temp["XYZUVW"];
    C3["XZYWUV"] -= temp["XYZUVW"];
    C3["XYZUWV"] -= temp["XYZUVW"];
    C3["ZXYUWV"] -= temp["XYZUVW"];
    C3["XZYUWV"] += temp["XYZUVW"];

    // aab
    temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"aaAaaA"});
    temp["xyZwuV"] -= alpha * H2["xywi"] * T2["iZuV"];
    temp["xyZwuV"] += alpha * H2["aZuV"] * T2["xywa"];
    C3["xyZwuV"] += temp["xyZwuV"];
    C3["xyZuwV"] -= temp["xyZwuV"];

    temp.zero();
    temp["zxYuvW"] += alpha * H2["xYiW"] * T2["izuv"];
    temp["zxYuvW"] -= alpha * H2["azuv"] * T2["xYaW"];
    C3["zxYuvW"] += temp["zxYuvW"];
    C3["xzYuvW"] -= temp["zxYuvW"];

    temp.zero();
    temp["zxYwuV"] += alpha * H2["xYwI"] * T2["zIuV"];
    temp["zxYwuV"] -= alpha * H2["zAuV"] * T2["xYwA"];
    C3["zxYwuV"] += temp["zxYwuV"];
    C3["xzYwuV"] -= temp["zxYwuV"];
    C3["zxYuwV"] -= temp["zxYwuV"];
    C3["xzYuwV"] += temp["zxYwuV"];

    // abb
    temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"aAAaAA"});
    temp["zYXuVW"] -= alpha * H2["XYWI"] * T2["zIuV"];
    temp["zYXuVW"] += alpha * H2["zAuV"] * T2["XYWA"];
    C3["zYXuVW"] += temp["zYXuVW"];
    C3["zYXuWV"] -= temp["zYXuVW"];

    temp.zero();
    temp["xYZwVU"] += alpha * H2["xYwI"] * T2["IZUV"];
    temp["xYZwVU"] -= alpha * H2["AZUV"] * T2["xYwA"];
    C3["xYZwVU"] += temp["xYZwVU"];
    C3["xZYwVU"] -= temp["xYZwVU"];

    temp.zero();
    temp["xYZuVW"] += alpha * H2["xYiW"] * T2["iZuV"];
    temp["xYZuVW"] -= alpha * H2["aZuV"] * T2["xYaW"];
    C3["xYZuVW"] += temp["xYZuVW"];
    C3["xZYuVW"] -= temp["xYZuVW"];
    C3["xYZuWV"] -= temp["xYZuVW"];
    C3["xZYuWV"] += temp["xYZuVW"];

    if (print_ > 2) {
        outfile->Printf("\n    Time for [H2, T2] -> C3 : %12.3f", timer.get());
    }
    dsrg_time_.add("223", timer.get());
}

// std::vector<std::vector<double>> DSRG_MRPT2::diagonalize_Fock_diagblocks(BlockedTensor& U) {
//    // diagonal blocks identifiers (C-A-V ordering)
//    std::vector<std::string> blocks{"cc", "aa", "vv", "CC", "AA", "VV"};

//    // map MO space label to its psi::Dimension
//    std::map<std::string, psi::Dimension> MOlabel_to_dimension;
//    MOlabel_to_dimension[acore_label_] = mo_space_info_->dimension("RESTRICTED_DOCC");
//    MOlabel_to_dimension[aactv_label_] = mo_space_info_->dimension("ACTIVE");
//    MOlabel_to_dimension[avirt_label_] = mo_space_info_->dimension("RESTRICTED_UOCC");

//    // eigen values to be returned
//    size_t ncmo = mo_space_info_->size("CORRELATED");
//    psi::Dimension corr = mo_space_info_->dimension("CORRELATED");
//    std::vector<double> eigenvalues_a(ncmo, 0.0);
//    std::vector<double> eigenvalues_b(ncmo, 0.0);

//    // map MO space label to its offset psi::Dimension
//    std::map<std::string, psi::Dimension> MOlabel_to_offset_dimension;
//    int nirrep = corr.n();
//    MOlabel_to_offset_dimension["c"] = psi::Dimension(std::vector<int>(nirrep, 0));
//    MOlabel_to_offset_dimension["a"] = mo_space_info_->dimension("RESTRICTED_DOCC");
//    MOlabel_to_offset_dimension["v"] =
//        mo_space_info_->dimension("RESTRICTED_DOCC") + mo_space_info_->dimension("ACTIVE");

//    // figure out index
//    auto fill_eigen = [&](std::string block_label, int irrep, std::vector<double> values) {
//        int h = irrep;
//        size_t idx_begin = 0;
//        while ((--h) >= 0)
//            idx_begin += corr[h];

//        std::string label(1, tolower(block_label[0]));
//        idx_begin += MOlabel_to_offset_dimension[label][irrep];

//        bool spin_alpha = islower(block_label[0]);
//        size_t nvalues = values.size();
//        if (spin_alpha) {
//            for (size_t i = 0; i < nvalues; ++i) {
//                eigenvalues_a[i + idx_begin] = values[i];
//            }
//        } else {
//            for (size_t i = 0; i < nvalues; ++i) {
//                eigenvalues_b[i + idx_begin] = values[i];
//            }
//        }
//    };

//    // diagonalize diagonal blocks
//    for (const auto& block : blocks) {
//        size_t dim = F_.block(block).dim(0);
//        if (dim == 0) {
//            continue;
//        } else {
//            std::string label(1, tolower(block[0]));
//            psi::Dimension space = MOlabel_to_dimension[label];
//            int nirrep = space.n();

//            // separate Fock with irrep
//            for (int h = 0; h < nirrep; ++h) {
//                size_t h_dim = space[h];
//                ambit::Tensor U_h;
//                if (h_dim == 0) {
//                    continue;
//                } else if (h_dim == 1) {
//                    U_h = ambit::Tensor::build(tensor_type_, "U_h", std::vector<size_t>(2,
//                    h_dim)); U_h.data()[0] = 1.0; ambit::Tensor F_block =
//                        ambit::Tensor::build(tensor_type_, "F_block", F_.block(block).dims());
//                    F_block.data() = F_.block(block).data();
//                    ambit::Tensor T_h = separate_tensor(F_block, space, h);
//                    fill_eigen(block, h, T_h.data());
//                } else {
//                    ambit::Tensor F_block =
//                        ambit::Tensor::build(tensor_type_, "F_block", F_.block(block).dims());
//                    F_block.data() = F_.block(block).data();
//                    ambit::Tensor T_h = separate_tensor(F_block, space, h);
//                    auto Feigen = T_h.syev(AscendingEigenvalue);
//                    U_h = ambit::Tensor::build(tensor_type_, "U_h", std::vector<size_t>(2,
//                    h_dim)); U_h("pq") = Feigen["eigenvectors"]("pq"); fill_eigen(block, h,
//                    Feigen["eigenvalues"].data());
//                }
//                ambit::Tensor U_out = U.block(block);
//                combine_tensor(U_out, U_h, space, h);
//            }
//        }
//    }
//    return {eigenvalues_a, eigenvalues_b};
//}

// ambit::Tensor DSRG_MRPT2::separate_tensor(ambit::Tensor& tens, const psi::Dimension& irrep,
//                                          const int& h) {
//    // test tens and irrep
//    size_t tens_dim = tens.dim(0);
//    if (tens_dim != static_cast<size_t>(irrep.sum()) || tens_dim != tens.dim(1)) {
//        throw psi::PSIEXCEPTION("Wrong dimension for the to-be-separated ambit Tensor.");
//    }
//    if (h >= irrep.n()) {
//        throw psi::PSIEXCEPTION("Ask for wrong irrep.");
//    }

//    // from relative (blocks) to absolute (big tensor) index
//    auto rel_to_abs = [&](size_t i, size_t j, size_t offset) {
//        return (i + offset) * tens_dim + (j + offset);
//    };

//    // compute offset
//    size_t offset = 0, h_dim = irrep[h];
//    int h_local = h;
//    while ((--h_local) >= 0)
//        offset += irrep[h_local];

//    // fill in values
//    ambit::Tensor T_h = ambit::Tensor::build(tensor_type_, "T_h", std::vector<size_t>(2, h_dim));
//    for (size_t i = 0; i < h_dim; ++i) {
//        for (size_t j = 0; j < h_dim; ++j) {
//            size_t abs_idx = rel_to_abs(i, j, offset);
//            T_h.data()[i * h_dim + j] = tens.data()[abs_idx];
//        }
//    }

//    return T_h;
//}

// void DSRG_MRPT2::combine_tensor(ambit::Tensor& tens, ambit::Tensor& tens_h,
//                                const psi::Dimension& irrep, const int& h) {
//    // test tens and irrep
//    if (h >= irrep.n()) {
//        throw psi::PSIEXCEPTION("Ask for wrong irrep.");
//    }
//    size_t tens_h_dim = tens_h.dim(0), h_dim = irrep[h];
//    if (tens_h_dim != h_dim || tens_h_dim != tens_h.dim(1)) {
//        throw psi::PSIEXCEPTION("Wrong dimension for the to-be-combined ambit Tensor.");
//    }

//    // from relative (blocks) to absolute (big tensor) index
//    size_t tens_dim = tens.dim(0);
//    auto rel_to_abs = [&](size_t i, size_t j, size_t offset) {
//        return (i + offset) * tens_dim + (j + offset);
//    };

//    // compute offset
//    size_t offset = 0;
//    int h_local = h;
//    while ((--h_local) >= 0)
//        offset += irrep[h_local];

//    // fill in values
//    for (size_t i = 0; i < h_dim; ++i) {
//        for (size_t j = 0; j < h_dim; ++j) {
//            size_t abs_idx = rel_to_abs(i, j, offset);
//            tens.data()[abs_idx] = tens_h.data()[i * h_dim + j];
//        }
//    }
//}

ambit::BlockedTensor DSRG_MRPT2::get_T1deGNO(double& T0deGNO) {
    ambit::BlockedTensor T1eff = deGNO_Tamp(T1_, T2_, Gamma1_);

    if (internal_amp_) {
        // the scalar term of amplitudes when de-normal-ordering
        T0deGNO -= T1_["uv"] * Gamma1_["vu"];
        T0deGNO -= T1_["UV"] * Gamma1_["VU"];

        T0deGNO -= 0.25 * T2_["xyuv"] * Lambda2_["uvxy"];
        T0deGNO -= 0.25 * T2_["XYUV"] * Lambda2_["UVXY"];
        T0deGNO -= T2_["xYuV"] * Lambda2_["uVxY"];

        T0deGNO += 0.5 * T2_["xyuv"] * Gamma1_["ux"] * Gamma1_["vy"];
        T0deGNO += 0.5 * T2_["XYUV"] * Gamma1_["UX"] * Gamma1_["VY"];
        T0deGNO += T2_["xYuV"] * Gamma1_["ux"] * Gamma1_["VY"];
    }

    return T1eff;
}

ambit::BlockedTensor DSRG_MRPT2::get_T2(const std::vector<std::string>& blocks) {
    for (const std::string& block : blocks) {
        if (!T2_.is_block(block)) {
            std::string error = "Error from T2(blocks): cannot find block " + block;
            throw psi::PSIEXCEPTION(error);
        }
    }
    ambit::BlockedTensor out = ambit::BlockedTensor::build(tensor_type_, "T2 selected", blocks);
    out["ijab"] = T2_["ijab"];
    out["iJaB"] = T2_["iJaB"];
    out["IJAB"] = T2_["IJAB"];
    return out;
}

ambit::BlockedTensor DSRG_MRPT2::get_RH1deGNO() {
    ambit::BlockedTensor RH1eff = BTF_->build(tensor_type_, "RH1 from deGNO", spin_cases({"ph"}));

    RH1eff["ai"] = F_["ai"];
    RH1eff["AI"] = F_["AI"];

    RH1eff["ai"] -= V_["auiv"] * Gamma1_["vu"];
    RH1eff["ai"] -= V_["aUiV"] * Gamma1_["VU"];
    RH1eff["AI"] -= V_["uAvI"] * Gamma1_["vu"];
    RH1eff["AI"] -= V_["AUIV"] * Gamma1_["VU"];

    return RH1eff;
}

void DSRG_MRPT2::rotate_amp(psi::SharedMatrix Ua, psi::SharedMatrix Ub, const bool& transpose,
                            const bool& t1eff) {
    ambit::BlockedTensor U = BTF_->build(tensor_type_, "Uorb", spin_cases({"gg"}));

    std::map<char, std::vector<std::pair<size_t, size_t>>> space_to_relmo;
    space_to_relmo['c'] = mo_space_info_->relative_mo("RESTRICTED_DOCC");
    space_to_relmo['a'] = mo_space_info_->relative_mo("ACTIVE");
    space_to_relmo['v'] = mo_space_info_->relative_mo("RESTRICTED_UOCC");

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

            output += "\n" + indent +
                      str(boost::format("[%3d %3d %3d %3d] %13.8f (%10.6f + "
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
} // namespace forte
