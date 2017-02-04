/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2017 by its authors (see LICENSE, AUTHORS).
 *
 * The copyrights for code used from other parties are included in
 * the corresponding files.
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see http://www.gnu.org/licenses/.
 *
 * @END LICENSE
 */

#include <algorithm>
#include <vector>
#include <map>
#include "mini-boost/boost/format.hpp"

#include "psi4/libdiis/diismanager.h"

#include "helpers.h"
#include "mrdsrg.h"

namespace psi {
namespace forte {

double MRDSRG::compute_energy_pt2() {
    // print title
    outfile->Printf("\n\n  ==> Second-Order Perturbation DSRG-MRPT2 <==\n");
    outfile->Printf("\n    Reference:");
    outfile->Printf("\n      J. Chem. Theory Comput. 2015, 11, 2097-2108.");
    outfile->Printf("\n      J. Chem. Phys. 2016 (in preparation)\n");

    // create zeroth-order Hamiltonian
    H0th_ = BTF_->build(tensor_type_, "Zeroth-order H", spin_cases({"gg"}));
    H0th_.iterate([&](const std::vector<size_t>& i,
                      const std::vector<SpinType>& spin, double& value) {
        if (i[0] == i[1]) {
            if (spin[0] == AlphaSpin) {
                value = Fa_[i[0]];
            } else {
                value = Fb_[i[0]];
            }
        }
    });

    // test orbitals are semi-canonicalized
    if (!check_semicanonical()) {
        outfile->Printf("\n    DSRG-MRPT2 is currently only formulated using "
                        "semi-canonical orbitals!");
        throw PSIEXCEPTION("Orbitals are not semi-canonicalized.");
    }

    // compute MRPT2 energy and Hbar
    std::vector<std::pair<std::string, double>> energy;
    std::string H0th = options_.get_str("H0TH");
    if (H0th == "FFULL") {
        energy = compute_energy_pt2_Ffull();
    } else if (H0th == "FDIAG_VACTV" || H0th == "FDIAG_VDIAG") {
        energy = compute_energy_pt2_FdiagV();
    } else {
        energy = compute_energy_pt2_Fdiag();
    }

    outfile->Printf("\n\n  ==> DSRG-MRPT2 Energy Summary <==\n");
    for (auto& str_dim : energy) {
        outfile->Printf("\n    %-30s = %22.15f", str_dim.first.c_str(),
                        str_dim.second);
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

    // reference relaxation
    if (options_.get_str("RELAX_REF") != "NONE") {
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

std::vector<std::pair<std::string, double>>
MRDSRG::compute_energy_pt2_FdiagV() {
    // figure out off-diagonal block labels
    std::vector<std::string> blocks1 = od_one_labels_hp();
    std::vector<std::string> blocks2 = od_two_labels_hhpp();

    // solve first-order amplitudes
    int cycle = 0, maxiter = options_.get_int("MAXITER");
    double r_conv = options_.get_double("R_CONVERGENCE");
    bool converged = false, failed = false;

    Hbar1_ = BTF_->build(tensor_type_, "Hbar1", spin_cases({"gg"}));
    Hbar2_ = BTF_->build(tensor_type_, "Hbar2", spin_cases({"gggg"}));
    O1_ = BTF_->build(tensor_type_, "O1", od_one_labels_hp());
    O2_ = BTF_->build(tensor_type_, "O2", od_two_labels_hhpp());
    DT1_ = BTF_->build(tensor_type_, "DT1", spin_cases({"hp"}));
    DT2_ = BTF_->build(tensor_type_, "DT2", spin_cases({"hhpp"}));
    std::vector<double> big_T, big_DT;
    size_t numel = vector_size_diis(T1_, blocks1, T2_, blocks2);

    // setup DIIS
    std::shared_ptr<DIISManager> diis_manager;
    int max_diis_vectors = options_.get_int("DIIS_MAX_VECS");
    int min_diis_vectors = options_.get_int("DIIS_MIN_VECS");
    if (max_diis_vectors > 0) {
        diis_manager = std::shared_ptr<DIISManager>(
            new DIISManager(max_diis_vectors, "MRPT2 DIIS T",
                            DIISManager::LargestError, DIISManager::InCore));
        diis_manager->set_error_vector_size(1, DIISEntry::Pointer, numel);
        diis_manager->set_vector_size(1, DIISEntry::Pointer, numel);
    }

    // turn on expert mode in ambit
    BlockedTensor::set_expert_mode(true);

    // two-body zeroth-order Hamiltonian
    BlockedTensor V0th;
    std::string H0th_string = options_.get_str("H0TH");
    if (H0th_string == "FDIAG_VACTV") {
        V0th = BTF_->build(tensor_type_, "V0th", spin_cases({"aaaa"}));
    } else if (H0th_string == "FDIAG_VDIAG") {
        V0th = BTF_->build(tensor_type_, "V0th", re_two_labels());
    }
    //    for(auto& x: V0th.block_labels()){
    //        outfile->Printf("\n  V0th block %s", x.c_str());
    //    }
    V0th["pqrs"] = V_["pqrs"];
    V0th["pQrS"] = V_["pQrS"];
    V0th["PQRS"] = V_["PQRS"];

    // print title
    print_h2("Solve first-order amplitudes");
    std::string indent(4, ' ');
    std::string dash(76, '-');
    std::string title;
    title += indent +
             str(boost::format("%5c  %=21s  %=21s  %=17s  %=4s\n") % ' ' %
                 "Non-Diagonal Norm" % "Amplitude RMS" % "Timings (s)" % " ");
    title += indent + std::string(7, ' ') + std::string(21, '-') + "  " +
             std::string(21, '-') + "  " + std::string(17, '-') + "\n";
    title +=
        indent +
        str(boost::format("%5s  %=10s %=10s  %=10s %=10s  %=8s %=8s  %4s\n") %
            "Iter." % "Hbar1" % "Hbar2" % "T1" % "T2" % "Hbar" % "Amp." %
            "DIIS");
    title += indent + dash;
    outfile->Printf("\n%s", title.c_str());

    // start iteration
    do {
        // compute Hbar
        ForteTimer t_hbar;
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

        double time_hbar = t_hbar.elapsed();

        // compute norms of off-diagonal Hbar
        double Hbar1od = Hbar1od_norm(blocks1);
        double Hbar2od = Hbar2od_norm(blocks2);

        // update amplitudes
        ForteTimer t_amp;
        update_t();
        double time_amp = t_amp.elapsed();

        // copy amplitudes to the big vector
        big_T = copy_amp_diis(T1_, blocks1, T2_, blocks2);
        big_DT = copy_amp_diis(DT1_, blocks1, DT2_, blocks2);

        // printing
        outfile->Printf(
            "\n    %5d  %10.3e %10.3e  %10.3e %10.3e  %8.3f %8.3f  ", cycle,
            Hbar1od, Hbar2od, T1rms_, T2rms_, time_hbar, time_amp);

        // DIIS amplitudes
        if (diis_manager) {
            if (cycle >= min_diis_vectors) {
                diis_manager->add_entry(2, &(big_DT[0]), &(big_T[0]));
                outfile->Printf("S");
                outfile->Flush();
            }
            if (cycle > max_diis_vectors) {
                if (diis_manager->subspace_size() >= min_diis_vectors &&
                    cycle) {
                    outfile->Printf("/E");
                    outfile->Flush();
                    diis_manager->extrapolate(1, &(big_T[0]));
                    return_amp_diis(T1_, blocks1, T2_, blocks2, big_T);
                }
            }
        }

        // test convergence
        double rms = T1rms_ > T2rms_ ? T1rms_ : T2rms_;
        if (rms < r_conv) {
            converged = true;
        }
        if (cycle > maxiter) {
            outfile->Printf("\n\n    First-order amplitudes do not converge in "
                            "%d iterations! Quitting.\n",
                            maxiter);
            converged = true;
            failed = true;
        }
        outfile->Flush();
        ++cycle;
    } while (!converged);
    outfile->Printf("\n    %s", dash.c_str());

    // analyze converged amplitudes
    analyze_amplitudes("First-Order (Iter.)", T1_, T2_);

    // fail to converge
    if (failed) {
        throw PSIEXCEPTION(
            "First-order amplitudes do not converge in DSRG-MRPT2.");
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
    energy.push_back({"<[F, A1]>", 2 * (Ecorr - Etemp)});
    Etemp = Ecorr;

    H1_T2_C0(Hbar1_, T2_, 1.0, Ecorr);
    energy.push_back({"<[F, A2]>", 2 * (Ecorr - Etemp)});
    Etemp = Ecorr;

    H2_T1_C0(Hbar2_, T1_, 1.0, Ecorr);
    energy.push_back({"<[V, A1]>", 2 * (Ecorr - Etemp)});
    Etemp = Ecorr;

    H2_T2_C0(Hbar2_, T2_, 1.0, Ecorr);
    energy.push_back({"<[V, A2]>", 2 * (Ecorr - Etemp)});
    Etemp = Ecorr;

    // <[H, A]> = 2 * <[H, T]>
    Ecorr *= 2.0;

    energy.push_back({"DSRG-MRPT2 correlation energy", Ecorr});
    energy.push_back({"DSRG-MRPT2 total energy", Eref_ + Ecorr});

    // reference relaxation
    if (options_.get_str("RELAX_REF") != "NONE") {
        O1_ = BTF_->build(tensor_type_, "O1", spin_cases({"hh"}));
        O2_ = BTF_->build(tensor_type_, "O2", spin_cases({"hhhh"}));

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

std::vector<std::pair<std::string, double>>
MRDSRG::compute_energy_pt2_FdiagVdiag() {
    // figure out off-diagonal block labels
    std::vector<std::string> blocks1 = od_one_labels_hp();
    std::vector<std::string> blocks2 = od_two_labels_hhpp();

    // solve first-order amplitudes
    double Ecorr = 0.0, E1st = 0.0;
    std::vector<std::pair<std::string, double>> energy;
    int cycle = 0, maxiter = options_.get_int("MAXITER");
    double e_conv = options_.get_double("E_CONVERGENCE");
    double r_conv = options_.get_double("R_CONVERGENCE");
    bool converged = false, failed = false;

    Hbar1_ = BTF_->build(tensor_type_, "Hbar1", spin_cases({"gg"}));
    Hbar2_ = BTF_->build(tensor_type_, "Hbar2", spin_cases({"gggg"}));
    O1_ = BTF_->build(tensor_type_, "O1", spin_cases({"gg"}));
    O2_ = BTF_->build(tensor_type_, "O2", spin_cases({"gggg"}));
    DT1_ = BTF_->build(tensor_type_, "DT1", spin_cases({"hp"}));
    DT2_ = BTF_->build(tensor_type_, "DT2", spin_cases({"hhpp"}));
    std::vector<double> big_T, big_DT;
    size_t numel = vector_size_diis(T1_, blocks1, T2_, blocks2);

    // setup DIIS
    std::shared_ptr<DIISManager> diis_manager;
    int max_diis_vectors = options_.get_int("DIIS_MAX_VECS");
    int min_diis_vectors = options_.get_int("DIIS_MIN_VECS");
    if (max_diis_vectors > 0) {
        diis_manager = std::shared_ptr<DIISManager>(
            new DIISManager(max_diis_vectors, "MRPT2 DIIS T",
                            DIISManager::LargestError, DIISManager::InCore));
        diis_manager->set_error_vector_size(1, DIISEntry::Pointer, numel);
        diis_manager->set_vector_size(1, DIISEntry::Pointer, numel);
    }

    // turn on expert mode in ambit
    BlockedTensor::set_expert_mode(true);

    // two-body zeroth-order Hamiltonian
    BlockedTensor V0th = BTF_->build(tensor_type_, "V0th", re_two_labels());
    V0th["pqrs"] = V_["pqrs"];
    V0th["pQrS"] = V_["pQrS"];
    V0th["PQRS"] = V_["PQRS"];

    // print title
    print_h2("Solve first-order amplitudes");
    std::string indent(4, ' ');
    std::string dash(70, '-');
    std::string title;
    title +=
        indent + str(boost::format("%5c  %=21s  %=21s  %=17s\n") % ' ' %
                     "Non-Diagonal Norm" % "Amplitude RMS" % "Timings (s)");
    title += indent + std::string(7, ' ') + std::string(21, '-') + "  " +
             std::string(21, '-') + "  " + std::string(17, '-') + "\n";
    title += indent +
             str(boost::format("%5s  %=10s %=10s  %=10s %=10s  %=8s %=8s\n") %
                 "Iter." % "Hbar1" % "Hbar2" % "T1" % "T2" % "Hbar" % "Amp.");
    title += indent + dash;
    outfile->Printf("\n%s", title.c_str());

    // start iteration
    do {
        // compute Hbar
        ForteTimer t_hbar;
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

        double time_hbar = t_hbar.elapsed();

        // compute norms of off-diagonal Hbar
        double Hbar1od = Hbar1od_norm(blocks1);
        double Hbar2od = Hbar2od_norm(blocks2);

        // update amplitudes
        ForteTimer t_amp;
        update_t();
        double time_amp = t_amp.elapsed();

        // copy amplitudes to the big vector
        big_T = copy_amp_diis(T1_, blocks1, T2_, blocks2);
        big_DT = copy_amp_diis(DT1_, blocks1, DT2_, blocks2);

        // DIIS amplitudes
        if (diis_manager) {
            if (cycle >= min_diis_vectors) {
                diis_manager->add_entry(2, &(big_DT[0]), &(big_T[0]));
            }
            if (cycle > max_diis_vectors) {
                if (diis_manager->subspace_size() >= min_diis_vectors &&
                    cycle) {
                    outfile->Printf(" -> DIIS");
                    outfile->Flush();
                    diis_manager->extrapolate(1, &(big_T[0]));
                    return_amp_diis(T1_, blocks1, T2_, blocks2, big_T);
                }
            }
        }

        // printing
        outfile->Printf("\n    %5d  %10.3e %10.3e  %10.3e %10.3e  %8.3f %8.3f",
                        cycle, Hbar1od, Hbar2od, T1rms_, T2rms_, time_hbar,
                        time_amp);

        // test convergence
        double rms = T1rms_ > T2rms_ ? T1rms_ : T2rms_;
        if (rms < r_conv) {
            converged = true;
        }
        if (cycle > maxiter) {
            outfile->Printf("\n\n    First-order amplitudes do not converge in "
                            "%d iterations! Quitting.\n",
                            maxiter);
            converged = true;
            failed = true;
        }
        outfile->Flush();
        ++cycle;
    } while (!converged);
    outfile->Printf("\n    %s", dash.c_str());

    // analyze converged amplitudes
    analyze_amplitudes("First-Order (Iter.)", T1_, T2_);

    // fail to converge
    if (failed) {
        throw PSIEXCEPTION(
            "First-order amplitudes do not converge in DSRG-MRPT2.");
    }

    // reset Hbar to 1st-order Hamiltonian
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
    energy.push_back({"E0 (reference)", Eref_});
    double Etemp = 0.0;

    H1_T1_C0(Hbar1_, T1_, 1.0, Ecorr);
    energy.push_back({"<[F, A1]>", 2 * (Ecorr - Etemp)});
    Etemp = Ecorr;

    H1_T2_C0(Hbar1_, T2_, 1.0, Ecorr);
    energy.push_back({"<[F, A2]>", 2 * (Ecorr - Etemp)});
    Etemp = Ecorr;

    H2_T1_C0(Hbar2_, T1_, 1.0, Ecorr);
    energy.push_back({"<[V, A1]>", 2 * (Ecorr - Etemp)});
    Etemp = Ecorr;

    H2_T2_C0(Hbar2_, T2_, 1.0, Ecorr);
    energy.push_back({"<[V, A2]>", 2 * (Ecorr - Etemp)});
    Etemp = Ecorr;

    // <[H, A]> = 2 * <[H, T]>
    Ecorr *= 2.0;

    energy.push_back({"DSRG-MRPT2 correlation energy", Ecorr});
    energy.push_back({"DSRG-MRPT2 total energy", Eref_ + Ecorr});

    // reference relaxation
    if (options_.get_str("RELAX_REF") != "NONE") {
        // save the hole part of [H^0th, A^1st]
        BlockedTensor H0A1_1 =
            BTF_->build(tensor_type_, "H0A1_1", spin_cases({"gg"}));
        BlockedTensor H0A1_2 =
            BTF_->build(tensor_type_, "H0A1_2", spin_cases({"gggg"}));
        H0A1_1["pq"] = O1_["pq"];
        H0A1_1["PQ"] = O1_["PQ"];
        H0A1_1["pq"] += O1_["qp"];
        H0A1_1["PQ"] += O1_["QP"];
        H0A1_2["pqrs"] = O2_["pqrs"];
        H0A1_2["pQrS"] = O2_["pQrS"];
        H0A1_2["PQRS"] = O2_["PQRS"];
        H0A1_2["pqrs"] += O2_["rspq"];
        H0A1_2["pQrS"] += O2_["rSpQ"];
        H0A1_2["PQRS"] += O2_["RSPQ"];

        // save [H^1st + 0.5 * [H^0th, A^1st], A^1st]
        C1_ = BTF_->build(tensor_type_, "C1", spin_cases({"gg"}));
        C2_ = BTF_->build(tensor_type_, "C2", spin_cases({"gggg"}));

        O1_.zero();
        O2_.zero();
        H1_T1_C1(Hbar1_, T1_, 1.0, O1_);
        H1_T2_C1(Hbar1_, T2_, 1.0, O1_);
        H2_T1_C1(Hbar2_, T1_, 1.0, O1_);
        H2_T2_C1(Hbar2_, T2_, 1.0, O1_);

        H1_T2_C2(Hbar1_, T2_, 1.0, O2_);
        H2_T1_C2(Hbar2_, T1_, 1.0, O2_);
        H2_T2_C2(Hbar2_, T2_, 1.0, O2_);

        C1_["pq"] = O1_["pq"];
        C1_["PQ"] = O1_["PQ"];
        C1_["pq"] += O1_["qp"];
        C1_["PQ"] += O1_["QP"];

        C2_["pqrs"] = O2_["pqrs"];
        C2_["pQrS"] = O2_["pQrS"];
        C2_["PQRS"] = O2_["PQRS"];
        C2_["pqrs"] += O2_["rspq"];
        C2_["pQrS"] += O2_["rSpQ"];
        C2_["PQRS"] += O2_["RSPQ"];

        // solve 2nd-order amplitudes
        cycle = 0;
        converged = false, failed = false;
        T1_.zero();
        T2_.zero();
        if (max_diis_vectors > 0) {
            diis_manager = std::shared_ptr<DIISManager>(new DIISManager(
                max_diis_vectors, "MRPT2 DIIS T", DIISManager::LargestError,
                DIISManager::InCore));
            diis_manager->set_error_vector_size(1, DIISEntry::Pointer, numel);
            diis_manager->set_vector_size(1, DIISEntry::Pointer, numel);
        }

        // print title
        print_h2("Solve second-order amplitudes");
        outfile->Printf("\n%s", title.c_str());

        // start iteration
        do {
            // compute Hbar
            ForteTimer t_hbar;
            Hbar1_["pq"] = C1_["pq"];
            Hbar1_["PQ"] = C1_["PQ"];
            Hbar2_["pqrs"] = C2_["pqrs"];
            Hbar2_["pQrS"] = C2_["pQrS"];
            Hbar2_["PQRS"] = C2_["PQRS"];

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

            double time_hbar = t_hbar.elapsed();

            // compute norms of off-diagonal Hbar
            double Hbar1od = Hbar1od_norm(blocks1);
            double Hbar2od = Hbar2od_norm(blocks2);

            // update amplitudes
            ForteTimer t_amp;
            update_t();
            double time_amp = t_amp.elapsed();

            // copy amplitudes to the big vector
            big_T = copy_amp_diis(T1_, blocks1, T2_, blocks2);
            big_DT = copy_amp_diis(DT1_, blocks1, DT2_, blocks2);

            // DIIS amplitudes
            if (diis_manager) {
                if (cycle >= min_diis_vectors) {
                    diis_manager->add_entry(2, &(big_DT[0]), &(big_T[0]));
                }
                if (cycle > max_diis_vectors) {
                    if (diis_manager->subspace_size() >= min_diis_vectors &&
                        cycle) {
                        outfile->Printf(" -> DIIS");
                        outfile->Flush();
                        diis_manager->extrapolate(1, &(big_T[0]));
                        return_amp_diis(T1_, blocks1, T2_, blocks2, big_T);
                    }
                }
            }

            // printing
            outfile->Printf(
                "\n    %5d  %10.3e %10.3e  %10.3e %10.3e  %8.3f %8.3f", cycle,
                Hbar1od, Hbar2od, T1rms_, T2rms_, time_hbar, time_amp);

            // test convergence
            double rms = T1rms_ > T2rms_ ? T1rms_ : T2rms_;
            if (rms < r_conv) {
                converged = true;
            }
            if (cycle > maxiter) {
                outfile->Printf("\n\n    First-order amplitudes do not "
                                "converge in %d iterations! Quitting.\n",
                                maxiter);
                converged = true;
                failed = true;
            }
            outfile->Flush();
            ++cycle;
        } while (!converged);
        outfile->Printf("\n    %s", dash.c_str());

        // analyze converged amplitudes
        analyze_amplitudes("Second-Order (Iter.)", T1_, T2_);

        // fail to converge
        if (failed) {
            throw PSIEXCEPTION(
                "Second-order amplitudes do not converge in DSRG-MRPT2.");
        }

        // build Hbar correct till 2nd order
        Hbar1_["pq"] = F_["pq"];
        Hbar1_["PQ"] = F_["PQ"];
        Hbar2_["pqrs"] = V_["pqrs"];
        Hbar2_["pQrS"] = V_["pQrS"];
        Hbar2_["PQRS"] = V_["PQRS"];

        Hbar1_["pq"] += H0A1_1["pq"];
        Hbar1_["PQ"] += H0A1_1["PQ"];
        Hbar2_["pqrs"] += H0A1_2["pqrs"];
        Hbar2_["pQrS"] += H0A1_2["pQrS"];
        Hbar2_["PQRS"] += H0A1_2["PQRS"];

        Hbar1_["pq"] += C1_["pq"];
        Hbar1_["PQ"] += C1_["PQ"];
        Hbar2_["pqrs"] += C2_["pqrs"];
        Hbar2_["pQrS"] += C2_["pQrS"];
        Hbar2_["PQRS"] += C2_["PQRS"];

        Hbar1_["pq"] += O1_["pq"];
        Hbar1_["PQ"] += O1_["PQ"];
        Hbar1_["pq"] += O1_["qp"];
        Hbar1_["PQ"] += O1_["QP"];
        Hbar2_["pqrs"] += O2_["pqrs"];
        Hbar2_["pQrS"] += O2_["pQrS"];
        Hbar2_["PQRS"] += O2_["PQRS"];
        Hbar2_["pqrs"] += O2_["rspq"];
        Hbar2_["pQrS"] += O2_["rSpQ"];
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
    int cycle = 0, maxiter = options_.get_int("MAXITER");
    double e_conv = options_.get_double("E_CONVERGENCE");
    double r_conv = options_.get_double("R_CONVERGENCE");
    bool converged = false, failed = false;
    Hbar1_ = BTF_->build(tensor_type_, "Hbar1", spin_cases({"gg"}));
    Hbar2_ = BTF_->build(tensor_type_, "Hbar2", spin_cases({"gggg"}));
    O1_ = BTF_->build(tensor_type_, "O1", spin_cases({"gg"}));
    O2_ = BTF_->build(tensor_type_, "O2", spin_cases({"gggg"}));
    C1_ = BTF_->build(tensor_type_, "C1", spin_cases({"gg"}));
    C2_ = BTF_->build(tensor_type_, "C2", spin_cases({"gggg"}));
    DT1_ = BTF_->build(tensor_type_, "DT1", spin_cases({"hp"}));
    DT2_ = BTF_->build(tensor_type_, "DT2", spin_cases({"hhpp"}));
    std::vector<double> big_T, big_DT;
    size_t numel = vector_size_diis(T1_, blocks1, T2_, blocks2);

    // setup DIIS
    std::shared_ptr<DIISManager> diis_manager;
    int max_diis_vectors = options_.get_int("DIIS_MAX_VECS");
    int min_diis_vectors = options_.get_int("DIIS_MIN_VECS");
    if (max_diis_vectors > 0) {
        diis_manager = std::shared_ptr<DIISManager>(
            new DIISManager(max_diis_vectors, "MRPT2 DIIS T",
                            DIISManager::LargestError, DIISManager::InCore));
        diis_manager->set_error_vector_size(1, DIISEntry::Pointer, numel);
        diis_manager->set_vector_size(1, DIISEntry::Pointer, numel);
    }

    // print title
    print_h2("Solve first-order amplitudes");
    std::string indent(4, ' ');
    std::string dash(99, '-');
    std::string title;
    title += indent + str(boost::format("%5c  %=27s  %=21s  %=21s  %=17s\n") %
                          ' ' % "Energy (a.u.)" % "Non-Diagonal Norm" %
                          "Amplitude RMS" % "Timings (s)");
    title += indent + std::string(7, ' ') + std::string(27, '-') + "  " +
             std::string(21, '-') + "  " + std::string(21, '-') + "  " +
             std::string(17, '-') + "\n";
    title +=
        indent +
        str(boost::format(
                "%5s  %=16s %=10s  %=10s %=10s  %=10s %=10s  %=8s %=8s\n") %
            "Iter." % "Corr." % "Delta" % "Hbar1" % "Hbar2" % "T1" % "T2" %
            "Hbar" % "Amp.");
    title += indent + dash;
    outfile->Printf("\n%s", title.c_str());

    // start iteration
    do {
        // compute Hbar
        ForteTimer t_hbar;
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
        double time_hbar = t_hbar.elapsed();

        // compute norms of off-diagonal Hbar
        double Hbar1od = Hbar1od_norm(blocks1);
        double Hbar2od = Hbar2od_norm(blocks2);

        // update amplitudes
        ForteTimer t_amp;
        update_t();
        double time_amp = t_amp.elapsed();

        // copy amplitudes to the big vector
        big_T = copy_amp_diis(T1_, blocks1, T2_, blocks2);
        big_DT = copy_amp_diis(DT1_, blocks1, DT2_, blocks2);

        // DIIS amplitudes
        if (diis_manager) {
            if (cycle >= min_diis_vectors) {
                diis_manager->add_entry(2, &(big_DT[0]), &(big_T[0]));
            }
            if (cycle > max_diis_vectors) {
                if (diis_manager->subspace_size() >= min_diis_vectors &&
                    cycle) {
                    outfile->Printf(" -> DIIS");
                    outfile->Flush();
                    diis_manager->extrapolate(1, &(big_T[0]));
                    return_amp_diis(T1_, blocks1, T2_, blocks2, big_T);
                }
            }
        }

        // printing
        outfile->Printf("\n    %5d  %16.12f %10.3e  %10.3e %10.3e  %10.3e "
                        "%10.3e  %8.3f %8.3f",
                        cycle, Ecorr, Edelta, Hbar1od, Hbar2od, T1rms_, T2rms_,
                        time_hbar, time_amp);

        // test convergence
        double rms = T1rms_ > T2rms_ ? T1rms_ : T2rms_;
        if (fabs(Edelta) < e_conv && rms < r_conv) {
            converged = true;
        }
        if (cycle > maxiter) {
            outfile->Printf("\n\n    First-order amplitudes do not converge in "
                            "%d iterations! Quitting.\n",
                            maxiter);
            converged = true;
            failed = true;
        }
        outfile->Flush();
        ++cycle;
    } while (!converged);

    // analyze converged amplitudes
    analyze_amplitudes("First-order (iter)", T1_, T2_);

    // fail to converge
    if (failed) {
        throw PSIEXCEPTION(
            "First-order amplitudes do not converge in DSRG-MRPT2.");
    }

    E1st = Ecorr;
    energy.push_back({"1st-order correlation energy", E1st});

    // compute second-order energy from first-order A
    Hbar2_["pqrs"] += V_["pqrs"];
    Hbar2_["pQrS"] += V_["pQrS"];
    Hbar2_["PQRS"] += V_["PQRS"];

    double Etemp = Ecorr, E2nd = 0.0;

    H1_T1_C0(Hbar1_, T1_, 1.0, Ecorr);
    energy.push_back({"<[F, A1]>", Ecorr - Etemp});
    Etemp = Ecorr;

    H1_T2_C0(Hbar1_, T2_, 1.0, Ecorr);
    energy.push_back({"<[F, A2]>", Ecorr - Etemp});
    Etemp = Ecorr;

    H2_T1_C0(Hbar2_, T1_, 1.0, Ecorr);
    energy.push_back({"<[V, A1]>", Ecorr - Etemp});
    Etemp = Ecorr;

    H2_T2_C0(Hbar2_, T2_, 1.0, Ecorr);
    energy.push_back({"<[V, A2]>", Ecorr - Etemp});
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

    // save first-order amplitudes
    BlockedTensor T1_1st =
        BTF_->build(tensor_type_, "T1_1st", spin_cases({"hp"}));
    BlockedTensor T2_1st =
        BTF_->build(tensor_type_, "T2_1st", spin_cases({"hhpp"}));
    T1_1st["ia"] = T1_["ia"];
    T1_1st["IA"] = T1_["IA"];
    T2_1st["ijab"] = T2_["ijab"];
    T2_1st["iJaB"] = T2_["iJaB"];
    T2_1st["IJAB"] = T2_["IJAB"];

    // solve second-order amplitudes
    Ecorr = 0.0;
    cycle = 0;
    converged = false, failed = false;
    T1_.zero();
    T2_.zero();
    if (max_diis_vectors > 0) {
        diis_manager = std::shared_ptr<DIISManager>(
            new DIISManager(max_diis_vectors, "MRPT2 DIIS T",
                            DIISManager::LargestError, DIISManager::InCore));
        diis_manager->set_error_vector_size(1, DIISEntry::Pointer, numel);
        diis_manager->set_vector_size(1, DIISEntry::Pointer, numel);
    }

    // print title
    print_h2("Solve second-order amplitudes");
    outfile->Printf("\n%s", title.c_str());

    // start iteration
    do {
        // compute Hbar
        ForteTimer t_hbar;

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
        double time_hbar = t_hbar.elapsed();

        // compute norms of off-diagonal Hbar
        double Hbar1od = Hbar1od_norm(blocks1);
        double Hbar2od = Hbar2od_norm(blocks2);

        // update amplitudes
        ForteTimer t_amp;
        update_t();
        double time_amp = t_amp.elapsed();

        // copy amplitudes to the big vector
        big_T = copy_amp_diis(T1_, blocks1, T2_, blocks2);
        big_DT = copy_amp_diis(DT1_, blocks1, DT2_, blocks2);

        // DIIS amplitudes
        if (diis_manager) {
            if (cycle >= min_diis_vectors) {
                diis_manager->add_entry(2, &(big_DT[0]), &(big_T[0]));
            }
            if (cycle > max_diis_vectors) {
                if (diis_manager->subspace_size() >= min_diis_vectors &&
                    cycle % 4 == 0) {
                    outfile->Printf(" -> DIIS");
                    outfile->Flush();
                    diis_manager->extrapolate(1, &(big_T[0]));
                    return_amp_diis(T1_, blocks1, T2_, blocks2, big_T);
                }
            }
        }

        // printing
        outfile->Printf("\n    %5d  %16.12f %10.3e  %10.3e %10.3e  %10.3e "
                        "%10.3e  %8.3f %8.3f",
                        cycle, Ecorr, Edelta, Hbar1od, Hbar2od, T1rms_, T2rms_,
                        time_hbar, time_amp);

        // test convergence
        double rms = T1rms_ > T2rms_ ? T1rms_ : T2rms_;
        if (fabs(Edelta) < e_conv && rms < r_conv) {
            converged = true;
        }
        if (cycle > maxiter) {
            outfile->Printf("\n\n    Second-order amplitudes do not converge "
                            "in %d iterations! Quitting.\n",
                            maxiter);
            converged = true;
            failed = true;
        }
        outfile->Flush();
        ++cycle;
    } while (!converged);

    // analyze converged amplitudes
    analyze_amplitudes("Second-order (iter)", T1_, T2_);

    // fail to converge
    if (failed) {
        throw PSIEXCEPTION(
            "First-order amplitudes do not converge in DSRG-MRPT2.");
    }

    E2nd += Ecorr;
    energy.push_back({"<[H_0th, A_2nd]>", Ecorr});
    energy.push_back({"2nd-order correlation energy", E2nd});

    Ecorr = E1st + E2nd;
    energy.push_back({"DSRG-MRPT2 correlation energy", Ecorr});
    energy.push_back({"DSRG-MRPT2 total energy", Eref_ + Ecorr});

    // reference relaxation
    if (options_.get_str("RELAX_REF") != "NONE") {
        Hbar1_["pq"] += F_["pq"];
        Hbar1_["PQ"] += F_["PQ"];
        Hbar2_["pqrs"] += V_["pqrs"];
        Hbar2_["pQrS"] += V_["pQrS"];
        Hbar2_["PQRS"] += V_["PQRS"];

        C1_.zero();
        C2_.zero();
        H1_T1_C1(F_, T1_1st, 1.0, C1_);
        H1_T2_C1(F_, T2_1st, 1.0, C1_);
        H1_T2_C2(F_, T2_1st, 1.0, C2_);

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
    }

    return energy;
}

double MRDSRG::compute_energy_pt3() {
    // print title
    outfile->Printf("\n\n  ==> Third-Order Perturbation DSRG-MRPT3 <==\n");
    outfile->Printf("\n    Reference:");
    outfile->Printf("\n      J. Chem. Phys. 2016 (in preparation)\n");
    std::vector<std::pair<std::string, double>> energy;
    energy.push_back({"E0 (reference)", Eref_});

    // create zeroth-order Hamiltonian
    H0th_ = BTF_->build(tensor_type_, "Zeroth-order H", spin_cases({"gg"}));
    H0th_.iterate([&](const std::vector<size_t>& i,
                      const std::vector<SpinType>& spin, double& value) {
        if (i[0] == i[1]) {
            if (spin[0] == AlphaSpin) {
                value = Fa_[i[0]];
            } else {
                value = Fb_[i[0]];
            }
        }
    });

    // test orbitals are semi-canonicalized
    if (!check_semicanonical()) {
        outfile->Printf("\n    DSRG-MRPT3 is currently only formulated using "
                        "semi-canonical orbitals!");
        throw PSIEXCEPTION("Orbitals are not semi-canonicalized.");
    }

    // create first-order bare Hamiltonian
    BlockedTensor H1st_1 =
        BTF_->build(tensor_type_, "H1st_1", spin_cases({"gg"}));
    BlockedTensor H1st_2 =
        BTF_->build(tensor_type_, "H1st_2", spin_cases({"gggg"}));
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

    std::string dsrg_op = options_.get_str("DSRG_TRANS_TYPE");
    if (dsrg_op == "UNITARY") {
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
    if (dsrg_op == "UNITARY") {
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
    BlockedTensor Hbar2nd_1 =
        BTF_->build(tensor_type_, "Hbar2nd_1", spin_cases({"gg"}));
    BlockedTensor Hbar2nd_2 =
        BTF_->build(tensor_type_, "Hbar2nd_2", spin_cases({"gggg"}));
    Hbar2nd_1["pq"] += O1_["pq"];
    Hbar2nd_1["PQ"] += O1_["PQ"];
    Hbar2nd_2["pqrs"] += O2_["pqrs"];
    Hbar2nd_2["pQrS"] += O2_["pQrS"];
    Hbar2nd_2["PQRS"] += O2_["PQRS"];

    if (dsrg_op == "UNITARY") {
        Hbar2nd_1["pq"] += O1_["qp"];
        Hbar2nd_1["PQ"] += O1_["QP"];
        Hbar2nd_2["pqrs"] += O2_["rspq"];
        Hbar2nd_2["pQrS"] += O2_["rSpQ"];
        Hbar2nd_2["PQRS"] += O2_["RSPQ"];
    }

    // compute second-order amplitudes
    BlockedTensor T2nd_1 =
        BTF_->build(tensor_type_, "temp1", spin_cases({"hp"}));
    BlockedTensor T2nd_2 =
        BTF_->build(tensor_type_, "temp2", spin_cases({"hhpp"}));
    guess_t(Hbar2nd_2, T2nd_2, Hbar2nd_1, T2nd_1);
    analyze_amplitudes("Second-Order", T2nd_1, T2nd_2);

    // compute <[H~1st, A2nd]>
    double Ecorr1 = 0.0;
    H1_T1_C0(Hbar1_, T2nd_1, 1.0, Ecorr1);
    H1_T2_C0(Hbar1_, T2nd_2, 1.0, Ecorr1);
    H2_T1_C0(Hbar2_, T2nd_1, 1.0, Ecorr1);
    H2_T2_C0(Hbar2_, T2nd_2, 1.0, Ecorr1);
    if (dsrg_op == "UNITARY") {
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

    if (dsrg_op == "UNITARY") {
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
    if (dsrg_op == "UNITARY") {
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
        outfile->Printf("\n    %-30s = %22.15f", str_dim.first.c_str(),
                        str_dim.second);
    }

    if (options_.get_str("RELAX_REF") != "NONE") {
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

bool MRDSRG::check_semicanonical() {
    outfile->Printf("\n    Checking if orbitals are semi-canonicalized ...");

    H0th_.iterate([&](const std::vector<size_t>& i,
                      const std::vector<SpinType>& spin, double& value) {
        if (i[0] == i[1]) {
            if (spin[0] == AlphaSpin) {
                value = Fa_[i[0]];
            } else {
                value = Fb_[i[0]];
            }
        }
    });

    std::vector<std::string> blocks = diag_one_labels();
    std::vector<double> Foff;
    double Foff_sum = 0.0;
    for (auto& block : blocks) {
        size_t dim = F_.block(block).dim(0);
        ambit::Tensor diff = ambit::Tensor::build(tensor_type_, "F - H0",
                                                  std::vector<size_t>(2, dim));
        diff("pq") = F_.block(block)("pq");
        diff("pq") -= H0th_.block(block)("pq");
        double value = diff.norm();
        Foff.emplace_back(value);
        Foff_sum += value;
    }

    double threshold = 0.5 * std::sqrt(options_.get_double("E_CONVERGENCE"));
    bool semi = false;
    if (Foff_sum > threshold) {
        std::string sep(3 + 16 * 3, '-');
        outfile->Printf("\n    Warning! Orbitals are not semi-canonicalized!");
        outfile->Printf("\n    Off-Diagonal norms of the core, active, virtual "
                        "blocks of Fock matrix");
        outfile->Printf("\n       %15s %15s %15s", "core", "active", "virtual");
        outfile->Printf("\n    %s", sep.c_str());
        outfile->Printf("\n    Fa %15.10f %15.10f %15.10f", Foff[0], Foff[1],
                        Foff[2]);
        outfile->Printf("\n    Fb %15.10f %15.10f %15.10f", Foff[3], Foff[4],
                        Foff[5]);
        outfile->Printf("\n    %s\n", sep.c_str());
    } else {
        outfile->Printf("     OK.");
        semi = true;
    }

    return semi;
}
}
}
