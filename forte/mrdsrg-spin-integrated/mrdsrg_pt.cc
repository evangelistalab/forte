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

#include <algorithm>
#include <map>
#include <memory>
#include <vector>

#include "psi4/libpsi4util/PsiOutStream.h"
#include "psi4/libdiis/diismanager.h"
#include "helpers/printing.h"

#include "base_classes/mo_space_info.h"
#include "helpers/timer.h"
#include "mrdsrg.h"

using namespace psi;

namespace forte {

double MRDSRG::compute_energy_pt2() {
    // print title
    outfile->Printf("\n\n  ==> Second-Order Perturbation DSRG-MRPT2 <==\n");
    outfile->Printf("\n    References:");
    outfile->Printf("\n      J. Chem. Theory Comput. 2015, 11, 2097-2108.");
    outfile->Printf("\n      J. Chem. Phys. 2017, 146, 124132.\n");

    // form full 2-e integrals if using density fitting
    if (eri_df_) {
        std::stringstream ss;
        ss << "DSRG-PT2 is not density fitted in MRDSRG module. Please try THREE-DSRG-MRPT2.";
        outfile->Printf("\n  %s", ss.str().c_str());
        outfile->Printf("\n  DSRG-PT2 of MRDSRG is mainly for testing ideas.");
        outfile->Printf("\n  If DF/CD is insisted, please uncomment lines 62-69 of mrdsrg_pt.cc "
                        "and recompile FORTE.");
        outfile->Printf("\n  However, this will result in building a full set of 2e integrals.");
        throw psi::PSIEXCEPTION(ss.str() + "\nPlease advise the output file.");

        //        V_ = BTF_->build(tensor_type_, "V", spin_cases({"gggg"}));
        //        V_["pqrs"] = B_["gpr"] * B_["gqs"];
        //        V_["pqrs"] -= B_["gps"] * B_["gqr"];

        //        V_["pQrS"] = B_["gpr"] * B_["gQS"];

        //        V_["PQRS"] = B_["gPR"] * B_["gQS"];
        //        V_["PQRS"] -= B_["gPS"] * B_["gQR"];
    }

    // create zeroth-order Hamiltonian
    H0th_ = BTF_->build(tensor_type_, "Zeroth-order H", spin_cases({"gg"}));
    for (const auto block : {"cc", "CC", "aa", "AA", "vv", "VV"}) {
        H0th_.block(block)("pq") = F_.block(block)("pq");
    }

    // compute MRPT2 energy and Hbar
    std::vector<std::pair<std::string, double>> energy;
    if (pt2_h0th_ == "FFULL") {
        energy = compute_energy_pt2_Ffull();
    } else if (pt2_h0th_ == "FDIAG_VACTV" || pt2_h0th_ == "FDIAG_VDIAG") {
        energy = compute_energy_pt2_FdiagV();
    } else {
        energy = compute_energy_pt2_Fdiag();
    }

    outfile->Printf("\n\n  ==> DSRG-MRPT2 Energy Summary <==\n");
    for (auto& str_dim : energy) {
        outfile->Printf("\n    %-30s = %22.15f", str_dim.first.c_str(), str_dim.second);
    }

    // <0>: description, <1>: energy value
    double Ecorr = std::get<1>(energy.back()) - Eref_;

    Hbar0_ = Ecorr;
    return Ecorr;
}

std::vector<std::pair<std::string, double>> MRDSRG::compute_energy_pt2_Fdiag() {
    // initialize Hbar with bare Hamiltonian
    Hbar1_ = BTF_->build(tensor_type_, "Hbar1", spin_cases({"gg"}));
    Hbar2_ = BTF_->build(tensor_type_, "Hbar2", spin_cases({"gggg"}));
    Hbar1_["pq"] = F_["pq"];
    Hbar1_["PQ"] = F_["PQ"];
    Hbar2_["pqrs"] = V_["pqrs"];
    Hbar2_["pQrS"] = V_["pQrS"];
    Hbar2_["PQRS"] = V_["PQRS"];

    // compute H0th contribution to H1 and H2
    O1_ = BTF_->build(tensor_type_, "temp1", spin_cases({"gg"}));
    O2_ = BTF_->build(tensor_type_, "temp2", spin_cases({"gggg"}));
    H1_T1_C1(H0th_, T1_, 0.5, O1_);
    H1_T2_C1(H0th_, T2_, 0.5, O1_);
    H1_T2_C2(H0th_, T2_, 0.5, O2_);

    // [H, A] = [H, T] + [H, T]^dagger
    Hbar1_["pq"] += O1_["pq"];
    Hbar1_["PQ"] += O1_["PQ"];
    Hbar2_["pqrs"] += O2_["pqrs"];
    Hbar2_["pQrS"] += O2_["pQrS"];
    Hbar2_["PQRS"] += O2_["PQRS"];

    Hbar1_["pq"] += O1_["qp"];
    Hbar1_["PQ"] += O1_["QP"];
    Hbar2_["pqrs"] += O2_["rspq"];
    Hbar2_["pQrS"] += O2_["rSpQ"];
    Hbar2_["PQRS"] += O2_["RSPQ"];

    // compute PT2 energy
    std::vector<std::pair<std::string, double>> energy;
    energy.push_back({"E0 (reference)", Eref_});
    double Ecorr = 0.0, Etemp = 0.0;

    //    H1_T1_C0(Hbar1_,T1_,1.0,Ecorr);
    BlockedTensor::set_expert_mode(true);
    H1_G1_C0(Hbar1_, T1_, 1.0, Ecorr);
    energy.push_back({"<[F, A1]>", 2 * (Ecorr - Etemp)});
    Etemp = Ecorr;

    //    H1_T2_C0(Hbar1_,T2_,1.0,Ecorr);
    H1_G2_C0(Hbar1_, T2_, 1.0, Ecorr);
    energy.push_back({"<[F, A2]>", 2 * (Ecorr - Etemp)});
    Etemp = Ecorr;

    //    H2_T1_C0(Hbar2_,T1_,1.0,Ecorr);
    H1_G2_C0(T1_, Hbar2_, -1.0, Ecorr);
    energy.push_back({"<[V, A1]>", 2 * (Ecorr - Etemp)});
    Etemp = Ecorr;

    //    H2_T2_C0(Hbar2_,T2_,1.0,Ecorr);
    H2_T2_C0(Hbar2_, T2_, 1.0, Ecorr);
    energy.push_back({"<[V, A2]>", 2 * (Ecorr - Etemp)});
    Etemp = Ecorr;

    // <[H, A]> = 2 * <[H, T]>
    Ecorr *= 2.0;

    energy.push_back({"DSRG-MRPT2 correlation energy", Ecorr});
    energy.push_back({"DSRG-MRPT2 total energy", Eref_ + Ecorr});

    bool multi_state = foptions_->get_gen_list("AVG_STATE").size() != 0;

    // reference relaxation
    if (foptions_->get_str("RELAX_REF") != "NONE" || multi_state) {
        O1_.zero();
        O2_.zero();

        H1_T1_C1(Hbar1_, T1_, 1.0, O1_);
        H1_T2_C1(Hbar1_, T2_, 1.0, O1_);
        H2_T1_C1(Hbar2_, T1_, 1.0, O1_);
        H2_T2_C1(Hbar2_, T2_, 1.0, O1_);

        H1_T2_C2(Hbar1_, T2_, 1.0, O2_);
        H2_T1_C2(Hbar2_, T1_, 1.0, O2_);
        H2_T2_C2(Hbar2_, T2_, 1.0, O2_);

        //        H1_G1_C1(Hbar1_,T1_,1.0,O1_);
        //        H1_G2_C1(Hbar1_,T2_,1.0,O1_);
        //        H1_G2_C1(T1_,Hbar2_,-1.0,O1_);
        //        H2_G2_C1(Hbar2_,T2_,1.0,O1_);

        //        H1_G2_C2(Hbar1_,T2_,1.0,O2_);
        //        H1_G2_C2(T1_,Hbar2_,-1.0,O2_);
        //        H2_G2_C2(Hbar2_,T2_,1.0,O2_);

        Hbar1_["pq"] += O1_["pq"];
        Hbar1_["pq"] += O1_["qp"];
        Hbar1_["PQ"] += O1_["PQ"];
        Hbar1_["PQ"] += O1_["QP"];
        Hbar2_["pqrs"] += O2_["pqrs"];
        Hbar2_["pqrs"] += O2_["rspq"];
        Hbar2_["pQrS"] += O2_["pQrS"];
        Hbar2_["pQrS"] += O2_["rSpQ"];
        Hbar2_["PQRS"] += O2_["PQRS"];
        Hbar2_["PQRS"] += O2_["RSPQ"];
    }

    return energy;
}

std::vector<std::pair<std::string, double>> MRDSRG::compute_energy_pt2_FdiagV() {
    // figure out off-diagonal block labels
    std::vector<std::string> blocks1 = od_one_labels_hp();
    std::vector<std::string> blocks2 = od_two_labels_hhpp();

    // solve first-order amplitudes
    int maxiter = foptions_->get_int("MAXITER");
    double r_conv = foptions_->get_double("R_CONVERGENCE");
    bool converged = false;

    Hbar1_ = BTF_->build(tensor_type_, "Hbar1", spin_cases({"gg"}));
    Hbar2_ = BTF_->build(tensor_type_, "Hbar2", spin_cases({"gggg"}));
    O1_ = BTF_->build(tensor_type_, "O1", od_one_labels_hp());
    O2_ = BTF_->build(tensor_type_, "O2", od_two_labels_hhpp());
    DT1_ = BTF_->build(tensor_type_, "DT1", spin_cases({"hp"}));
    DT2_ = BTF_->build(tensor_type_, "DT2", spin_cases({"hhpp"}));

    // setup DIIS
    if (diis_start_ > 0) {
        diis_manager_init();
    }

    // turn on expert mode in ambit
    BlockedTensor::set_expert_mode(true);

    // two-body zeroth-order Hamiltonian
    BlockedTensor V0th;
    if (pt2_h0th_ == "FDIAG_VACTV") {
        V0th = BTF_->build(tensor_type_, "V0th", spin_cases({"aaaa"}));
    } else if (pt2_h0th_ == "FDIAG_VDIAG") {
        V0th = BTF_->build(tensor_type_, "V0th", re_two_labels());
    }

    V0th["pqrs"] = V_["pqrs"];
    V0th["pQrS"] = V_["pQrS"];
    V0th["PQRS"] = V_["PQRS"];

    // print title
    print_h2("Solve first-order amplitudes");
    std::string indent(4, ' ');
    std::string dash(76, '-');
    std::string title;
    title += indent + "         Non-Diagonal Norm        Amplitude RMS         Timings (s)\n";
    title += indent + "       ---------------------  ---------------------  -----------------\n";
    title +=
        indent + "Iter.     Hbar1      Hbar2        T1         T2        Hbar     Amp.    DIIS\n";
    title += indent + dash;
    outfile->Printf("\n%s", title.c_str());

    // start iteration
    for (int cycle = 1; cycle <= maxiter; ++cycle) {
        // compute Hbar
        local_timer t_hbar;
        Hbar1_["pq"] = F_["pq"];
        Hbar1_["PQ"] = F_["PQ"];
        Hbar1_["pq"] -= H0th_["pq"];
        Hbar1_["PQ"] -= H0th_["PQ"];

        Hbar2_["pqrs"] = V_["pqrs"];
        Hbar2_["pQrS"] = V_["pQrS"];
        Hbar2_["PQRS"] = V_["PQRS"];
        Hbar2_["pqrs"] -= V0th["pqrs"];
        Hbar2_["pQrS"] -= V0th["pQrS"];
        Hbar2_["PQRS"] -= V0th["PQRS"];

        O1_.zero();
        O2_.zero();
        H1_T1_C1(H0th_, T1_, 1.0, O1_);
        H1_T2_C1(H0th_, T2_, 1.0, O1_);
        H1_T2_C2(H0th_, T2_, 1.0, O2_);
        H2_T1_C2(V0th, T1_, 1.0, O2_);
        H2_T2_C1(V0th, T2_, 1.0, O1_);
        H2_T2_C2(V0th, T2_, 1.0, O2_);

        Hbar1_["pq"] += O1_["pq"];
        Hbar1_["PQ"] += O1_["PQ"];
        Hbar2_["pqrs"] += O2_["pqrs"];
        Hbar2_["pQrS"] += O2_["pQrS"];
        Hbar2_["PQRS"] += O2_["PQRS"];

        Hbar1_["pq"] += O1_["qp"];
        Hbar1_["PQ"] += O1_["QP"];
        Hbar2_["pqrs"] += O2_["rspq"];
        Hbar2_["pQrS"] += O2_["rSpQ"];
        Hbar2_["PQRS"] += O2_["RSPQ"];

        double time_hbar = t_hbar.get();

        // compute norms of off-diagonal Hbar
        double Hbar1od = Hbar1od_norm(blocks1);
        double Hbar2od = Hbar2od_norm(blocks2);

        // update amplitudes
        local_timer t_amp;
        update_t();
        double time_amp = t_amp.get();

        // printing
        outfile->Printf("\n    %4d   %10.3e %10.3e  %10.3e %10.3e  %8.3f %8.3f", cycle, Hbar1od,
                        Hbar2od, T1rms_, T2rms_, time_hbar, time_amp);

        // DIIS amplitudes
        if (diis_start_ > 0 and cycle >= diis_start_) {
            diis_manager_add_entry();
            outfile->Printf("  S");

            if ((cycle - diis_start_) % diis_freq_ == 0 and
                diis_manager_->subspace_size() >= diis_min_vec_) {
                diis_manager_extrapolate();
                outfile->Printf("/E");
            }
        }

        // test convergence
        double rms = T1rms_ > T2rms_ ? T1rms_ : T2rms_;
        if (rms < r_conv) {
            converged = true;
            break;
        }

        if (cycle == maxiter) {
            outfile->Printf(
                "\n\n    First-order amplitudes do not converge in %d iterations! Quitting.\n",
                maxiter);
        }
        if (cycle > 5 and std::fabs(rms) > 10.0) {
            outfile->Printf("\n\n    Large RMS for amplitudes. Likely no convergence. Quitting.\n");
        }
    }
    outfile->Printf("\n    %s", dash.c_str());

    // clean up raw pointers used in DIIS
    if (diis_start_ > 0) {
        diis_manager_cleanup();
    }

    // analyze converged amplitudes
    analyze_amplitudes("First-Order (Iter.)", T1_, T2_);

    // fail to converge
    if (!converged) {
        throw psi::PSIEXCEPTION("First-order amplitudes do not converge in DSRG-MRPT2.");
    }

    // reset Hbar to 1st-order H
    Hbar1_["pq"] = F_["pq"];
    Hbar1_["PQ"] = F_["PQ"];
    Hbar1_["pq"] -= H0th_["pq"];
    Hbar1_["PQ"] -= H0th_["PQ"];

    Hbar2_["pqrs"] = V_["pqrs"];
    Hbar2_["pQrS"] = V_["pQrS"];
    Hbar2_["PQRS"] = V_["PQRS"];
    Hbar2_["pqrs"] -= V0th["pqrs"];
    Hbar2_["pQrS"] -= V0th["pQrS"];
    Hbar2_["PQRS"] -= V0th["PQRS"];

    // add 0.5 * [H^0th, A^1st] to Hbar
    Hbar1_["pq"] += 0.5 * O1_["pq"];
    Hbar1_["PQ"] += 0.5 * O1_["PQ"];
    Hbar2_["pqrs"] += 0.5 * O2_["pqrs"];
    Hbar2_["pQrS"] += 0.5 * O2_["pQrS"];
    Hbar2_["PQRS"] += 0.5 * O2_["PQRS"];

    Hbar1_["pq"] += 0.5 * O1_["qp"];
    Hbar1_["PQ"] += 0.5 * O1_["QP"];
    Hbar2_["pqrs"] += 0.5 * O2_["rspq"];
    Hbar2_["pQrS"] += 0.5 * O2_["rSpQ"];
    Hbar2_["PQRS"] += 0.5 * O2_["RSPQ"];

    // compute PT2 energy
    double Ecorr = 0.0, Etemp = 0.0;
    std::vector<std::pair<std::string, double>> energy;
    energy.push_back({"E0 (reference)", Eref_});

    H1_T1_C0(Hbar1_, T1_, 1.0, Ecorr);
    energy.push_back({"<[Htilde1, A1]>", 2 * (Ecorr - Etemp)});
    Etemp = Ecorr;

    H1_T2_C0(Hbar1_, T2_, 1.0, Ecorr);
    energy.push_back({"<[Htilde1, A2]>", 2 * (Ecorr - Etemp)});
    Etemp = Ecorr;

    H2_T1_C0(Hbar2_, T1_, 1.0, Ecorr);
    energy.push_back({"<[Htilde2, A1]>", 2 * (Ecorr - Etemp)});
    Etemp = Ecorr;

    H2_T2_C0(Hbar2_, T2_, 1.0, Ecorr);
    energy.push_back({"<[Htilde2, A2]>", 2 * (Ecorr - Etemp)});
    Etemp = Ecorr;

    // <[H, A]> = 2 * <[H, T]>
    Ecorr *= 2.0;

    energy.push_back({"DSRG-MRPT2 correlation energy", Ecorr});
    energy.push_back({"DSRG-MRPT2 total energy", Eref_ + Ecorr});

    bool multi_state = foptions_->get_gen_list("AVG_STATE").size() != 0;

    // reference relaxation
    if (foptions_->get_str("RELAX_REF") != "NONE" || multi_state) {
        O1_ = BTF_->build(tensor_type_, "O1", spin_cases({"aa"}));
        O2_ = BTF_->build(tensor_type_, "O2", spin_cases({"aaaa"}));

        H1_T1_C1(Hbar1_, T1_, 1.0, O1_);
        H1_T2_C1(Hbar1_, T2_, 1.0, O1_);
        H2_T1_C1(Hbar2_, T1_, 1.0, O1_);
        H2_T2_C1(Hbar2_, T2_, 1.0, O1_);

        H1_T2_C2(Hbar1_, T2_, 1.0, O2_);
        H2_T1_C2(Hbar2_, T1_, 1.0, O2_);
        H2_T2_C2(Hbar2_, T2_, 1.0, O2_);

        Hbar1_["pq"] = F_["pq"];
        Hbar1_["PQ"] = F_["PQ"];
        Hbar2_["pqrs"] = V_["pqrs"];
        Hbar2_["pQrS"] = V_["pQrS"];
        Hbar2_["PQRS"] = V_["PQRS"];

        Hbar1_["pq"] += O1_["pq"];
        Hbar1_["pq"] += O1_["qp"];
        Hbar1_["PQ"] += O1_["PQ"];
        Hbar1_["PQ"] += O1_["QP"];
        Hbar2_["pqrs"] += O2_["pqrs"];
        Hbar2_["pqrs"] += O2_["rspq"];
        Hbar2_["pQrS"] += O2_["pQrS"];
        Hbar2_["pQrS"] += O2_["rSpQ"];
        Hbar2_["PQRS"] += O2_["PQRS"];
        Hbar2_["PQRS"] += O2_["RSPQ"];
    }

    return energy;
}

std::vector<std::pair<std::string, double>> MRDSRG::compute_energy_pt2_Ffull() {
    // figure out off-diagonal block labels
    std::vector<std::string> blocks1 = od_one_labels_hp();
    std::vector<std::string> blocks2 = od_two_labels_hhpp();

    // solve first-order amplitudes
    double Ecorr = 0.0, E1st = 0.0;
    std::vector<std::pair<std::string, double>> energy;
    int maxiter = foptions_->get_int("MAXITER");
    double e_conv = foptions_->get_double("E_CONVERGENCE");
    double r_conv = foptions_->get_double("R_CONVERGENCE");
    bool converged = false;
    Hbar1_ = BTF_->build(tensor_type_, "Hbar1", spin_cases({"gg"}));
    Hbar2_ = BTF_->build(tensor_type_, "Hbar2", spin_cases({"gggg"}));
    O1_ = BTF_->build(tensor_type_, "O1", spin_cases({"gg"}));
    O2_ = BTF_->build(tensor_type_, "O2", spin_cases({"gggg"}));
    C1_ = BTF_->build(tensor_type_, "C1", spin_cases({"gg"}));
    C2_ = BTF_->build(tensor_type_, "C2", spin_cases({"gggg"}));
    DT1_ = BTF_->build(tensor_type_, "DT1", spin_cases({"hp"}));
    DT2_ = BTF_->build(tensor_type_, "DT2", spin_cases({"hhpp"}));
    auto Hbar1_actv = BTF_->build(tensor_type_, "Hbar1 active", spin_cases({"aa"}));
    auto Hbar2_actv = BTF_->build(tensor_type_, "Hbar2 active", spin_cases({"aaaa"}));

    // setup DIIS
    if (diis_start_ > 0) {
        diis_manager_init();
    }

    // print title
    print_h2("Solve first-order amplitudes");
    std::string indent(4, ' ');
    std::string dash(105, '-');
    std::string title;

    title += indent + "              Energy (a.u.)           Non-Diagonal Norm        Amplitude "
                      "RMS         Timings (s)\n";
    title += indent + "       ---------------------------  ---------------------  "
                      "---------------------  -----------------\n";
    title += indent + "Iter.        Corr.         Delta       Hbar1      Hbar2        T1         "
                      "T2        Hbar     Amp.    DIIS\n";
    title += indent + dash;

    outfile->Printf("\n%s", title.c_str());

    // start iteration
    for (int cycle = 1; cycle <= maxiter; ++cycle) {
        // compute Hbar
        local_timer t_hbar;
        Hbar1_.zero();
        Hbar2_["pqrs"] = V_["pqrs"];
        Hbar2_["pQrS"] = V_["pQrS"];
        Hbar2_["PQRS"] = V_["PQRS"];

        double C0 = 0.0;
        C1_.zero();
        C2_.zero();
        H1_T1_C0(F_, T1_, 1.0, C0);
        H1_T2_C0(F_, T2_, 1.0, C0);
        H1_T1_C1(F_, T1_, 1.0, C1_);
        H1_T2_C1(F_, T2_, 1.0, C1_);
        H1_T2_C2(F_, T2_, 1.0, C2_);

        Hbar1_["pq"] += C1_["pq"];
        Hbar1_["PQ"] += C1_["PQ"];
        Hbar2_["pqrs"] += C2_["pqrs"];
        Hbar2_["pQrS"] += C2_["pQrS"];
        Hbar2_["PQRS"] += C2_["PQRS"];

        Hbar1_["pq"] += C1_["qp"];
        Hbar1_["PQ"] += C1_["QP"];
        Hbar2_["pqrs"] += C2_["rspq"];
        Hbar2_["pQrS"] += C2_["rSpQ"];
        Hbar2_["PQRS"] += C2_["RSPQ"];

        Hbar0_ = 2 * C0;
        double Edelta = Hbar0_ - Ecorr;
        Ecorr = Hbar0_;
        double time_hbar = t_hbar.get();

        // compute norms of off-diagonal Hbar
        double Hbar1od = Hbar1od_norm(blocks1);
        double Hbar2od = Hbar2od_norm(blocks2);

        // update amplitudes
        local_timer t_amp;
        update_t();
        double time_amp = t_amp.get();

        // printing
        outfile->Printf("\n    %5d  %16.12f %10.3e  %10.3e %10.3e  %10.3e %10.3e  %8.3f %8.3f",
                        cycle, Ecorr, Edelta, Hbar1od, Hbar2od, T1rms_, T2rms_, time_hbar,
                        time_amp);

        // DIIS amplitudes
        if (diis_start_ > 0 and cycle >= diis_start_) {
            diis_manager_add_entry();
            outfile->Printf("  S");

            if ((cycle - diis_start_) % diis_freq_ == 0 and
                diis_manager_->subspace_size() >= diis_min_vec_) {
                diis_manager_extrapolate();
                outfile->Printf("/E");
            }
        }

        // test convergence
        double rms = T1rms_ > T2rms_ ? T1rms_ : T2rms_;
        if (std::fabs(Edelta) < e_conv && rms < r_conv) {
            converged = true;
            break;
        }

        if (cycle == maxiter) {
            outfile->Printf(
                "\n\n    First-order amplitudes do not converge in %d iterations! Quitting.\n",
                maxiter);
        }
        if (cycle > 5 and std::fabs(rms) > 10.0) {
            outfile->Printf("\n\n    Large RMS for amplitudes. Likely no convergence. Quitting.\n");
        }
    }
    outfile->Printf("\n    %s", dash.c_str());

    // clean up raw pointers used in DIIS
    if (diis_start_ > 0) {
        diis_manager_cleanup();
    }

    // analyze converged amplitudes
    analyze_amplitudes("First-order (iter)", T1_, T2_);

    // fail to converge
    if (!converged) {
        throw psi::PSIEXCEPTION("First-order amplitudes do not converge in DSRG-MRPT2.");
    }

    E1st = Ecorr;
    energy.push_back({"1st-order correlation energy", E1st});

    bool multi_state = foptions_->get_gen_list("AVG_STATE").size() != 0;

    // save active part for reference relxation
    if (foptions_->get_str("RELAX_REF") != "NONE" || multi_state) {
        Hbar1_actv["uv"] = F_["uv"];
        Hbar1_actv["UV"] = F_["UV"];
        Hbar1_actv["uv"] += Hbar1_["uv"];
        Hbar1_actv["UV"] += Hbar1_["UV"];
        Hbar2_actv["uvxy"] = Hbar2_["uvxy"];
        Hbar2_actv["uVxY"] = Hbar2_["uVxY"];
        Hbar2_actv["UVXY"] = Hbar2_["UVXY"];
    }

    // compute second-order energy from first-order A
    Hbar2_["pqrs"] += V_["pqrs"];
    Hbar2_["pQrS"] += V_["pQrS"];
    Hbar2_["PQRS"] += V_["PQRS"];

    double Etemp = Ecorr, E2nd = 0.0;

    H1_T1_C0(Hbar1_, T1_, 1.0, Ecorr);
    energy.push_back({"<[Hbar1, A1]>", Ecorr - Etemp});
    Etemp = Ecorr;

    H1_T2_C0(Hbar1_, T2_, 1.0, Ecorr);
    energy.push_back({"<[Hbar1, A2]>", Ecorr - Etemp});
    Etemp = Ecorr;

    H2_T1_C0(Hbar2_, T1_, 1.0, Ecorr);
    energy.push_back({"<[Htilde2, A1]>", Ecorr - Etemp});
    Etemp = Ecorr;

    H2_T2_C0(Hbar2_, T2_, 1.0, Ecorr);
    energy.push_back({"<[Htilde2, A2]>", Ecorr - Etemp});
    Etemp = Ecorr;

    E2nd += Ecorr - E1st;

    // save 0.5 * [H1st + Hbar1st, T] to O
    H1_T1_C1(Hbar1_, T1_, 0.5, O1_);
    H1_T2_C1(Hbar1_, T2_, 0.5, O1_);
    H2_T1_C1(Hbar2_, T1_, 0.5, O1_);
    H2_T2_C1(Hbar2_, T2_, 0.5, O1_);
    H1_T2_C2(Hbar1_, T2_, 0.5, O2_);
    H2_T1_C2(Hbar2_, T1_, 0.5, O2_);
    H2_T2_C2(Hbar2_, T2_, 0.5, O2_);

    // solve second-order amplitudes
    Ecorr = 0.0;
    converged = false;
    T1_.zero();
    T2_.zero();
    if (diis_start_ > 0) {
        diis_manager_init();
    }

    // print title
    print_h2("Solve second-order amplitudes");
    outfile->Printf("\n%s", title.c_str());

    // start iteration
    for (int cycle = 1; cycle <= maxiter; ++cycle) {
        // compute Hbar
        local_timer t_hbar;

        Hbar1_["pq"] = O1_["pq"];
        Hbar1_["PQ"] = O1_["PQ"];
        Hbar2_["pqrs"] = O2_["pqrs"];
        Hbar2_["pQrS"] = O2_["pQrS"];
        Hbar2_["PQRS"] = O2_["PQRS"];

        Hbar1_["pq"] += O1_["qp"];
        Hbar1_["PQ"] += O1_["QP"];
        Hbar2_["pqrs"] += O2_["rspq"];
        Hbar2_["pQrS"] += O2_["rSpQ"];
        Hbar2_["PQRS"] += O2_["RSPQ"];

        double C0 = 0.0;
        C1_.zero();
        C2_.zero();
        H1_T1_C0(F_, T1_, 1.0, C0);
        H1_T2_C0(F_, T2_, 1.0, C0);
        H1_T1_C1(F_, T1_, 1.0, C1_);
        H1_T2_C1(F_, T2_, 1.0, C1_);
        H1_T2_C2(F_, T2_, 1.0, C2_);

        Hbar1_["pq"] += C1_["pq"];
        Hbar1_["PQ"] += C1_["PQ"];
        Hbar2_["pqrs"] += C2_["pqrs"];
        Hbar2_["pQrS"] += C2_["pQrS"];
        Hbar2_["PQRS"] += C2_["PQRS"];

        Hbar1_["pq"] += C1_["qp"];
        Hbar1_["PQ"] += C1_["QP"];
        Hbar2_["pqrs"] += C2_["rspq"];
        Hbar2_["pQrS"] += C2_["rSpQ"];
        Hbar2_["PQRS"] += C2_["RSPQ"];

        Hbar0_ = 2 * C0;
        double Edelta = Hbar0_ - Ecorr;
        Ecorr = Hbar0_;
        double time_hbar = t_hbar.get();

        // compute norms of off-diagonal Hbar
        double Hbar1od = Hbar1od_norm(blocks1);
        double Hbar2od = Hbar2od_norm(blocks2);

        // update amplitudes
        local_timer t_amp;
        update_t();
        double time_amp = t_amp.get();

        // printing
        outfile->Printf("\n    %5d  %16.12f %10.3e  %10.3e %10.3e  %10.3e "
                        "%10.3e  %8.3f %8.3f",
                        cycle, Ecorr, Edelta, Hbar1od, Hbar2od, T1rms_, T2rms_, time_hbar,
                        time_amp);

        // DIIS amplitudes
        if (diis_start_ > 0 and cycle >= diis_start_) {
            diis_manager_add_entry();
            outfile->Printf("  S");

            if ((cycle - diis_start_) % diis_freq_ == 0 and
                diis_manager_->subspace_size() >= diis_min_vec_) {
                diis_manager_extrapolate();
                outfile->Printf("/E");
            }
        }

        // test convergence
        double rms = T1rms_ > T2rms_ ? T1rms_ : T2rms_;
        if (std::fabs(Edelta) < e_conv && rms < r_conv) {
            converged = true;
            break;
        }

        if (cycle == maxiter) {
            outfile->Printf(
                "\n\n    Second-order amplitudes do not converge in %d iterations! Quitting.\n",
                maxiter);
        }
        if (cycle > 5 and std::fabs(rms) > 10.0) {
            outfile->Printf("\n\n    Large RMS for amplitudes. Likely no convergence. Quitting.\n");
        }
    }
    outfile->Printf("\n    %s", dash.c_str());

    // clean up raw pointers used in DIIS
    if (diis_start_ > 0) {
        diis_manager_cleanup();
    }

    // analyze converged amplitudes
    analyze_amplitudes("Second-order (iter)", T1_, T2_);

    // fail to converge
    if (!converged) {
        throw psi::PSIEXCEPTION("Second-order amplitudes do not converge in DSRG-MRPT2.");
    }

    E2nd += Ecorr;
    energy.push_back({"<[H_0th, A_2nd]>", Ecorr});
    energy.push_back({"2nd-order correlation energy", E2nd});

    Ecorr = E1st + E2nd;
    energy.push_back({"DSRG-MRPT2 correlation energy", Ecorr});
    energy.push_back({"DSRG-MRPT2 total energy", Eref_ + Ecorr});

    // add 0th- and 1st-order Hbar for reference relaxation
    if (foptions_->get_str("RELAX_REF") != "NONE" || multi_state) {
        Hbar1_["pq"] += Hbar1_actv["pq"];
        Hbar1_["PQ"] += Hbar1_actv["PQ"];
        Hbar2_["pqrs"] += Hbar2_actv["pqrs"];
        Hbar2_["pQrS"] += Hbar2_actv["pQrS"];
        Hbar2_["PQRS"] += Hbar2_actv["PQRS"];
    }

    return energy;
}

double MRDSRG::compute_energy_pt3() {
    // print title
    outfile->Printf("\n\n  ==> Third-Order Perturbation DSRG-MRPT3 <==\n");
    outfile->Printf("\n    Reference:");
    outfile->Printf("\n      J. Chem. Phys. 2017, 146, 124132.\n");
    std::vector<std::pair<std::string, double>> energy;
    energy.push_back({"E0 (reference)", Eref_});

    // form full 2-e integrals if using density fitting
    if (eri_df_) {
        std::stringstream ss;
        ss << "DSRG-PT3 is not density fitted in MRDSRG module. Please try DSRG-MRPT3.";
        outfile->Printf("\n  %s", ss.str().c_str());
        outfile->Printf("\n  DSRG-PT3 of MRDSRG is for testing correctness of DSRG-MRPT3 code.");
        outfile->Printf("\n  If DF/CD is insisted, please uncomment lines 1198-1205 of "
                        "mrdsrg_pt.cc and recompile FORTE.");
        outfile->Printf("\n  However, this will result in building a full set of 2e integrals.");
        throw psi::PSIEXCEPTION(ss.str() + "\nPlease advise the output file.");

        //        V_ = BTF_->build(tensor_type_, "V", spin_cases({"gggg"}));
        //        V_["pqrs"] = B_["gpr"] * B_["gqs"];
        //        V_["pqrs"] -= B_["gps"] * B_["gqr"];

        //        V_["pQrS"] = B_["gpr"] * B_["gQS"];

        //        V_["PQRS"] = B_["gPR"] * B_["gQS"];
        //        V_["PQRS"] -= B_["gPS"] * B_["gQR"];
    }

    // create zeroth-order Hamiltonian
    H0th_ = BTF_->build(tensor_type_, "Zeroth-order H", spin_cases({"gg"}));
    for (const auto block : {"cc", "CC", "aa", "AA", "vv", "VV"}) {
        H0th_.block(block)("pq") = F_.block(block)("pq");
    }

    // create first-order bare Hamiltonian
    BlockedTensor H1st_1 = BTF_->build(tensor_type_, "H1st_1", spin_cases({"gg"}));
    BlockedTensor H1st_2 = BTF_->build(tensor_type_, "H1st_2", spin_cases({"gggg"}));
    H1st_1["pq"] = F_["pq"];
    H1st_1["PQ"] = F_["PQ"];
    for (auto block : {"cc", "aa", "vv", "CC", "AA", "VV"}) {
        H1st_1.block(block).zero();
    }
    H1st_2["pqrs"] = V_["pqrs"];
    H1st_2["pQrS"] = V_["pQrS"];
    H1st_2["PQRS"] = V_["PQRS"];

    // compute [H0th, T1st]
    O1_ = BTF_->build(tensor_type_, "temp1", spin_cases({"gg"}));
    O2_ = BTF_->build(tensor_type_, "temp2", spin_cases({"gggg"}));
    H1_T1_C1(H0th_, T1_, 1.0, O1_);
    H1_T2_C1(H0th_, T2_, 1.0, O1_);
    H1_T2_C2(H0th_, T2_, 1.0, O2_);

    // compute H~1st = H1st + 0.5 * [H0th, A1st]
    Hbar1_ = BTF_->build(tensor_type_, "temp1", spin_cases({"gg"}));
    Hbar2_ = BTF_->build(tensor_type_, "temp2", spin_cases({"gggg"}));
    Hbar1_["pq"] += 0.5 * O1_["pq"];
    Hbar1_["PQ"] += 0.5 * O1_["PQ"];
    Hbar2_["pqrs"] += 0.5 * O2_["pqrs"];
    Hbar2_["pQrS"] += 0.5 * O2_["pQrS"];
    Hbar2_["PQRS"] += 0.5 * O2_["PQRS"];

    if (dsrg_trans_type_ == "UNITARY") {
        Hbar1_["pq"] += 0.5 * O1_["qp"];
        Hbar1_["PQ"] += 0.5 * O1_["QP"];
        Hbar2_["pqrs"] += 0.5 * O2_["rspq"];
        Hbar2_["pQrS"] += 0.5 * O2_["rSpQ"];
        Hbar2_["PQRS"] += 0.5 * O2_["RSPQ"];
    }

    Hbar1_["pq"] += H1st_1["pq"];
    Hbar1_["PQ"] += H1st_1["PQ"];
    Hbar2_["pqrs"] += H1st_2["pqrs"];
    Hbar2_["pQrS"] += H1st_2["pQrS"];
    Hbar2_["PQRS"] += H1st_2["PQRS"];

    // compute second-order energy
    double Ept2 = 0.0;
    H1_T1_C0(Hbar1_, T1_, 1.0, Ept2);
    H1_T2_C0(Hbar1_, T2_, 1.0, Ept2);
    H2_T1_C0(Hbar2_, T1_, 1.0, Ept2);
    H2_T2_C0(Hbar2_, T2_, 1.0, Ept2);
    if (dsrg_trans_type_ == "UNITARY") {
        Ept2 *= 2.0;
    }
    energy.push_back({"2nd-order correlation energy", Ept2});

    // compute [H~1st, T1st]
    O1_.zero();
    O2_.zero();
    H1_T1_C1(Hbar1_, T1_, 1.0, O1_);
    H1_T2_C1(Hbar1_, T2_, 1.0, O1_);
    H2_T1_C1(Hbar2_, T1_, 1.0, O1_);
    H2_T2_C1(Hbar2_, T2_, 1.0, O1_);
    H1_T2_C2(Hbar1_, T2_, 1.0, O2_);
    H2_T1_C2(Hbar2_, T1_, 1.0, O2_);
    H2_T2_C2(Hbar2_, T2_, 1.0, O2_);

    // compute second-order Hbar
    BlockedTensor Hbar2nd_1 = BTF_->build(tensor_type_, "Hbar2nd_1", spin_cases({"gg"}));
    BlockedTensor Hbar2nd_2 = BTF_->build(tensor_type_, "Hbar2nd_2", spin_cases({"gggg"}));
    Hbar2nd_1["pq"] += O1_["pq"];
    Hbar2nd_1["PQ"] += O1_["PQ"];
    Hbar2nd_2["pqrs"] += O2_["pqrs"];
    Hbar2nd_2["pQrS"] += O2_["pQrS"];
    Hbar2nd_2["PQRS"] += O2_["PQRS"];

    if (dsrg_trans_type_ == "UNITARY") {
        Hbar2nd_1["pq"] += O1_["qp"];
        Hbar2nd_1["PQ"] += O1_["QP"];
        Hbar2nd_2["pqrs"] += O2_["rspq"];
        Hbar2nd_2["pQrS"] += O2_["rSpQ"];
        Hbar2nd_2["PQRS"] += O2_["RSPQ"];
    }

    // compute second-order amplitudes
    BlockedTensor T2nd_1 = BTF_->build(tensor_type_, "temp1", spin_cases({"hp"}));
    BlockedTensor T2nd_2 = BTF_->build(tensor_type_, "temp2", spin_cases({"hhpp"}));
    guess_t(Hbar2nd_2, T2nd_2, Hbar2nd_1, T2nd_1);
    analyze_amplitudes("Second-Order", T2nd_1, T2nd_2);

    // compute <[H~1st, A2nd]>
    double Ecorr1 = 0.0;
    H1_T1_C0(Hbar1_, T2nd_1, 1.0, Ecorr1);
    H1_T2_C0(Hbar1_, T2nd_2, 1.0, Ecorr1);
    H2_T1_C0(Hbar2_, T2nd_1, 1.0, Ecorr1);
    H2_T2_C0(Hbar2_, T2nd_2, 1.0, Ecorr1);
    if (dsrg_trans_type_ == "UNITARY") {
        Ecorr1 *= 2.0;
        energy.push_back({"<[H~1st, A2nd]>", Ecorr1});
    } else {
        energy.push_back({"<[H~1st, T2nd]>", Ecorr1});
    }

    // compute 1 / 3 * [H~1st, A1st]
    Hbar2nd_1.scale(1.0 / 3.0);
    Hbar2nd_2.scale(1.0 / 3.0);

    // compute 1 / 2 * [H0th, A2nd] + 1 / 6 * [H1st, A1st]
    O1_.zero();
    O2_.zero();
    H1_T1_C1(H0th_, T2nd_1, 0.5, O1_);
    H1_T2_C1(H0th_, T2nd_2, 0.5, O1_);
    H1_T2_C2(H0th_, T2nd_2, 0.5, O2_);

    H1_T1_C1(H1st_1, T1_, 1.0 / 6.0, O1_);
    H1_T2_C1(H1st_1, T2_, 1.0 / 6.0, O1_);
    H2_T1_C1(H1st_2, T1_, 1.0 / 6.0, O1_);
    H2_T2_C1(H1st_2, T2_, 1.0 / 6.0, O1_);
    H1_T2_C2(H1st_1, T2_, 1.0 / 6.0, O2_);
    H2_T1_C2(H1st_2, T1_, 1.0 / 6.0, O2_);
    H2_T2_C2(H1st_2, T2_, 1.0 / 6.0, O2_);

    Hbar2nd_1["pq"] += O1_["pq"];
    Hbar2nd_1["PQ"] += O1_["PQ"];
    Hbar2nd_2["pqrs"] += O2_["pqrs"];
    Hbar2nd_2["pQrS"] += O2_["pQrS"];
    Hbar2nd_2["PQRS"] += O2_["PQRS"];

    if (dsrg_trans_type_ == "UNITARY") {
        Hbar2nd_1["pq"] += O1_["qp"];
        Hbar2nd_1["PQ"] += O1_["QP"];
        Hbar2nd_2["pqrs"] += O2_["rspq"];
        Hbar2nd_2["pQrS"] += O2_["rSpQ"];
        Hbar2nd_2["PQRS"] += O2_["RSPQ"];
    }

    // compute <[H~2nd, A1st]>
    double Ecorr2 = 0.0;
    H1_T1_C0(Hbar2nd_1, T1_, 1.0, Ecorr2);
    H1_T2_C0(Hbar2nd_1, T2_, 1.0, Ecorr2);
    H2_T1_C0(Hbar2nd_2, T1_, 1.0, Ecorr2);
    H2_T2_C0(Hbar2nd_2, T2_, 1.0, Ecorr2);
    if (dsrg_trans_type_ == "UNITARY") {
        Ecorr2 *= 2.0;
        energy.push_back({"<[H~2nd, A1st]>", Ecorr2});
    } else {
        energy.push_back({"<[H~2nd, T1st]>", Ecorr2});
    }

    // print summary
    double Ecorr = Ecorr1 + Ecorr2;
    energy.push_back({"3rd-order correlation energy", Ecorr});
    Ecorr += Ept2;
    energy.push_back({"DSRG-MRPT3 correlation energy", Ecorr});
    energy.push_back({"DSRG-MRPT3 total energy", Eref_ + Ecorr});

    outfile->Printf("\n\n  ==> DSRG-MRPT3 Energy Summary <==\n");
    for (auto& str_dim : energy) {
        outfile->Printf("\n    %-30s = %22.15f", str_dim.first.c_str(), str_dim.second);
    }

    bool multi_state = foptions_->get_gen_list("AVG_STATE").size() != 0;

    if (foptions_->get_str("RELAX_REF") != "NONE" || multi_state) {
        O1_.zero();
        O2_.zero();

        // [H~1st, A1st]
        H1_T1_C1(Hbar1_, T1_, 1.0, O1_);
        H1_T2_C1(Hbar1_, T2_, 1.0, O1_);
        H2_T1_C1(Hbar2_, T1_, 1.0, O1_);
        H2_T2_C1(Hbar2_, T2_, 1.0, O1_);
        H1_T2_C2(Hbar1_, T2_, 1.0, O2_);
        H2_T1_C2(Hbar2_, T1_, 1.0, O2_);
        H2_T2_C2(Hbar2_, T2_, 1.0, O2_);

        // [H~2nd, A1st]
        H1_T1_C1(Hbar2nd_1, T1_, 1.0, O1_);
        H1_T2_C1(Hbar2nd_1, T2_, 1.0, O1_);
        H2_T1_C1(Hbar2nd_2, T1_, 1.0, O1_);
        H2_T2_C1(Hbar2nd_2, T2_, 1.0, O1_);
        H1_T2_C2(Hbar2nd_1, T2_, 1.0, O2_);
        H2_T1_C2(Hbar2nd_2, T1_, 1.0, O2_);
        H2_T2_C2(Hbar2nd_2, T2_, 1.0, O2_);

        // [H~1st, A2nd]
        H1_T1_C1(Hbar1_, T2nd_1, 1.0, O1_);
        H1_T2_C1(Hbar1_, T2nd_2, 1.0, O1_);
        H2_T1_C1(Hbar2_, T2nd_1, 1.0, O1_);
        H2_T2_C1(Hbar2_, T2nd_2, 1.0, O1_);
        H1_T2_C2(Hbar1_, T2nd_2, 1.0, O2_);
        H2_T1_C2(Hbar2_, T2nd_1, 1.0, O2_);
        H2_T2_C2(Hbar2_, T2nd_2, 1.0, O2_);

        Hbar1_["pq"] += O1_["pq"];
        Hbar1_["pq"] += O1_["qp"];
        Hbar1_["PQ"] += O1_["PQ"];
        Hbar1_["PQ"] += O1_["QP"];
        Hbar2_["pqrs"] += O2_["pqrs"];
        Hbar2_["pqrs"] += O2_["rspq"];
        Hbar2_["pQrS"] += O2_["pQrS"];
        Hbar2_["pQrS"] += O2_["rSpQ"];
        Hbar2_["PQRS"] += O2_["PQRS"];
        Hbar2_["PQRS"] += O2_["RSPQ"];

        Hbar1_["pq"] += H0th_["pq"];
        Hbar1_["PQ"] += H0th_["PQ"];
    }

    Hbar0_ = Ecorr;
    return Ecorr;
}
} // namespace forte
