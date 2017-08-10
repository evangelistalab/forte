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
#include <cctype>
#include <map>
#include <memory>
#include <vector>

#include "psi4/libdiis/diismanager.h"

#include "../helpers.h"
#include "../mini-boost/boost/format.hpp"
#include "mrdsrg.h"

namespace psi {
namespace forte {

void MRDSRG::compute_hbar() {
    if (print_ > 2) {
        outfile->Printf("\n\n  ==> Computing the DSRG Transformed Hamiltonian <==\n");
    }

    // copy bare Hamiltonian to Hbar
    Hbar0_ = 0.0;
    Hbar1_["pq"] = F_["pq"];
    Hbar1_["PQ"] = F_["PQ"];
    Hbar2_["pqrs"] = V_["pqrs"];
    Hbar2_["pQrS"] = V_["pQrS"];
    Hbar2_["PQRS"] = V_["PQRS"];

    // temporary Hamiltonian used in every iteration
    O1_["pq"] = F_["pq"];
    O1_["PQ"] = F_["PQ"];
    O2_["pqrs"] = V_["pqrs"];
    O2_["pQrS"] = V_["pQrS"];
    O2_["PQRS"] = V_["PQRS"];

    // iteration variables
    bool converged = false;
    int maxn = options_.get_int("DSRG_RSC_NCOMM");
    double ct_threshold = options_.get_double("SRG_RSC_THRESHOLD");
    std::string dsrg_op = options_.get_str("DSRG_TRANS_TYPE");

    // compute Hbar recursively
    for (int n = 1; n <= maxn; ++n) {
        // prefactor before n-nested commutator
        double factor = 1.0 / n;

        // Compute the commutator C = 1/n [O, T]
        double C0 = 0.0;
        C1_.zero();
        C2_.zero();

        // printing level
        if (print_ > 2) {
            std::string dash(38, '-');
            outfile->Printf("\n    %s", dash.c_str());
        }

        // zero-body
        H1_T1_C0(O1_, T1_, factor, C0);
        H1_T2_C0(O1_, T2_, factor, C0);
        H2_T1_C0(O2_, T1_, factor, C0);
        H2_T2_C0(O2_, T2_, factor, C0);
        // one-body
        H1_T1_C1(O1_, T1_, factor, C1_);
        H1_T2_C1(O1_, T2_, factor, C1_);
        H2_T1_C1(O2_, T1_, factor, C1_);
        if (options_.get_str("SRG_COMM") == "STANDARD") {
            H2_T2_C1(O2_, T2_, factor, C1_);
        } else if (options_.get_str("SRG_COMM") == "FO") {
            BlockedTensor C1p = BTF_->build(tensor_type_, "C1p", spin_cases({"gg"}));
            H2_T2_C1(O2_, T2_, factor, C1p);
            C1p.block("cc").scale(2.0);
            C1p.block("aa").scale(2.0);
            C1p.block("vv").scale(2.0);
            C1p.block("CC").scale(2.0);
            C1p.block("AA").scale(2.0);
            C1p.block("VV").scale(2.0);
            C1_["pq"] += C1p["pq"];
            C1_["PQ"] += C1p["PQ"];
        }
        // two-body
        if ((options_.get_str("SRG_COMM") == "STANDARD") or n < 2) {
            H1_T2_C2(O1_, T2_, factor, C2_);
        } else if (options_.get_str("SRG_COMM") == "FO2") {
            O1_.block("cc").scale(2.0);
            O1_.block("aa").scale(2.0);
            O1_.block("vv").scale(2.0);
            O1_.block("CC").scale(2.0);
            O1_.block("AA").scale(2.0);
            O1_.block("VV").scale(2.0);
            H1_T2_C2(O1_, T2_, factor, C2_);
            O1_.block("cc").scale(0.5);
            O1_.block("aa").scale(0.5);
            O1_.block("vv").scale(0.5);
            O1_.block("CC").scale(0.5);
            O1_.block("AA").scale(0.5);
            O1_.block("VV").scale(0.5);
        }
        H2_T1_C2(O2_, T1_, factor, C2_);
        H2_T2_C2(O2_, T2_, factor, C2_);

        // printing level
        if (print_ > 2) {
            std::string dash(38, '-');
            outfile->Printf("\n    %s\n", dash.c_str());
        }

        // [H, A] = [H, T] + [H, T]^dagger
        if (dsrg_op == "UNITARY") {
            C0 *= 2.0;
            O1_["pq"] = C1_["pq"];
            O1_["PQ"] = C1_["PQ"];
            C1_["pq"] += O1_["qp"];
            C1_["PQ"] += O1_["QP"];
            O2_["pqrs"] = C2_["pqrs"];
            O2_["pQrS"] = C2_["pQrS"];
            O2_["PQRS"] = C2_["PQRS"];
            C2_["pqrs"] += O2_["rspq"];
            C2_["pQrS"] += O2_["rSpQ"];
            C2_["PQRS"] += O2_["RSPQ"];
        }

        // Hbar += C
        Hbar0_ += C0;
        Hbar1_["pq"] += C1_["pq"];
        Hbar1_["PQ"] += C1_["PQ"];
        Hbar2_["pqrs"] += C2_["pqrs"];
        Hbar2_["pQrS"] += C2_["pQrS"];
        Hbar2_["PQRS"] += C2_["PQRS"];

        // copy C to O for next level commutator
        O1_["pq"] = C1_["pq"];
        O1_["PQ"] = C1_["PQ"];
        O2_["pqrs"] = C2_["pqrs"];
        O2_["pQrS"] = C2_["pQrS"];
        O2_["PQRS"] = C2_["PQRS"];

        // test convergence of C
        double norm_C1 = C1_.norm();
        double norm_C2 = C2_.norm();
        if (print_ > 2) {
            outfile->Printf("\n  n = %3d, C1norm = %20.15f, C2norm = %20.15f", n, norm_C1, norm_C2);
        }
        if (std::sqrt(norm_C2 * norm_C2 + norm_C1 * norm_C1) < ct_threshold) {
            converged = true;
            break;
        }
    }
    if (!converged) {
        outfile->Printf("\n    Warning! Hbar is not converged in %3d-nested commutators!", maxn);
        outfile->Printf("\n    Please increase DSRG_RSC_NCOMM.");
    }
}

void MRDSRG::compute_hbar_sequential() {
    if (print_ > 2) {
        outfile->Printf("\n\n  ==> Computing the DSRG Transformed Hamiltonian <==\n");
    }
    //    outfile->Printf("\n\n  ==> compute_hbar_sequential() <==\n");

    // copy bare Hamiltonian to Hbar
    Hbar0_ = 0.0;
    Hbar1_["pq"] = F_["pq"];
    Hbar1_["PQ"] = F_["PQ"];
    Hbar2_["pqrs"] = V_["pqrs"];
    Hbar2_["pQrS"] = V_["pQrS"];
    Hbar2_["PQRS"] = V_["PQRS"];

    // temporary Hamiltonian used in every iteration
    O1_["pq"] = F_["pq"];
    O1_["PQ"] = F_["PQ"];
    O2_["pqrs"] = V_["pqrs"];
    O2_["pQrS"] = V_["pQrS"];
    O2_["PQRS"] = V_["PQRS"];

    // iteration variables
    bool converged = false;
    int maxn = options_.get_int("DSRG_RSC_NCOMM");
    double ct_threshold = options_.get_double("SRG_RSC_THRESHOLD");
    std::string dsrg_op = options_.get_str("DSRG_TRANS_TYPE");

    // compute Hbar recursively
    for (int n = 1; n <= maxn; ++n) {
        // prefactor before n-nested commutator
        double factor = 1.0 / n;

        // Compute the commutator C = 1/n [O, T]
        double C0 = 0.0;
        C1_.zero();
        C2_.zero();

        // printing level
        if (print_ > 2) {
            std::string dash(38, '-');
            outfile->Printf("\n    %s", dash.c_str());
        }

        // zero-body
        H1_T1_C0(O1_, T1_, factor, C0);
        H2_T1_C0(O2_, T1_, factor, C0);
        // one-body
        H1_T1_C1(O1_, T1_, factor, C1_);
        H2_T1_C1(O2_, T1_, factor, C1_);
        // two-body
        H2_T1_C2(O2_, T1_, factor, C2_);

        // printing level
        if (print_ > 2) {
            std::string dash(38, '-');
            outfile->Printf("\n    %s\n", dash.c_str());
        }

        // [H, A] = [H, T] + [H, T]^dagger
        if (dsrg_op == "UNITARY") {
            C0 *= 2.0;
            O1_["pq"] = C1_["pq"];
            O1_["PQ"] = C1_["PQ"];
            C1_["pq"] += O1_["qp"];
            C1_["PQ"] += O1_["QP"];
            O2_["pqrs"] = C2_["pqrs"];
            O2_["pQrS"] = C2_["pQrS"];
            O2_["PQRS"] = C2_["PQRS"];
            C2_["pqrs"] += O2_["rspq"];
            C2_["pQrS"] += O2_["rSpQ"];
            C2_["PQRS"] += O2_["RSPQ"];
        }

        // Hbar += C
        Hbar0_ += C0;
        Hbar1_["pq"] += C1_["pq"];
        Hbar1_["PQ"] += C1_["PQ"];
        Hbar2_["pqrs"] += C2_["pqrs"];
        Hbar2_["pQrS"] += C2_["pQrS"];
        Hbar2_["PQRS"] += C2_["PQRS"];

        // copy C to O for next level commutator
        O1_["pq"] = C1_["pq"];
        O1_["PQ"] = C1_["PQ"];
        O2_["pqrs"] = C2_["pqrs"];
        O2_["pQrS"] = C2_["pQrS"];
        O2_["PQRS"] = C2_["PQRS"];

        // test convergence of C
        double norm_C1 = C1_.norm();
        double norm_C2 = C2_.norm();
        if (print_ > 2) {
            outfile->Printf("\n  n = %3d, C1norm = %20.15f, C2norm = %20.15f", n, norm_C1, norm_C2);
        }
        if (std::sqrt(norm_C2 * norm_C2 + norm_C1 * norm_C1) < ct_threshold) {
            converged = true;
            break;
        }
    }
    if (!converged) {
        outfile->Printf("\n    Warning! Hbar is not converged in %3d-nested commutators!", maxn);
        outfile->Printf("\n    Please increase DSRG_RSC_NCOMM.");
    }

    // temporary Hamiltonian used in every iteration
    O1_["pq"] = Hbar1_["pq"];
    O1_["PQ"] = Hbar1_["PQ"];
    O2_["pqrs"] = Hbar2_["pqrs"];
    O2_["pQrS"] = Hbar2_["pQrS"];
    O2_["PQRS"] = Hbar2_["PQRS"];

    // iteration variables
    converged = false;

    // compute Hbar recursively
    for (int n = 1; n <= maxn; ++n) {
        // prefactor before n-nested commutator
        double factor = 1.0 / n;

        // Compute the commutator C = 1/n [O, T]
        double C0 = 0.0;
        C1_.zero();
        C2_.zero();

        // printing level
        if (print_ > 2) {
            std::string dash(38, '-');
            outfile->Printf("\n    %s", dash.c_str());
        }

        // zero-body
        H1_T2_C0(O1_, T2_, factor, C0);
        H2_T2_C0(O2_, T2_, factor, C0);
        // one-body
        H1_T2_C1(O1_, T2_, factor, C1_);
        H2_T2_C1(O2_, T2_, factor, C1_);
        // two-body
        H1_T2_C2(O1_, T2_, factor, C2_);
        H2_T2_C2(O2_, T2_, factor, C2_);

        // printing level
        if (print_ > 2) {
            std::string dash(38, '-');
            outfile->Printf("\n    %s\n", dash.c_str());
        }

        // [H, A] = [H, T] + [H, T]^dagger
        if (dsrg_op == "UNITARY") {
            C0 *= 2.0;
            O1_["pq"] = C1_["pq"];
            O1_["PQ"] = C1_["PQ"];
            C1_["pq"] += O1_["qp"];
            C1_["PQ"] += O1_["QP"];
            O2_["pqrs"] = C2_["pqrs"];
            O2_["pQrS"] = C2_["pQrS"];
            O2_["PQRS"] = C2_["PQRS"];
            C2_["pqrs"] += O2_["rspq"];
            C2_["pQrS"] += O2_["rSpQ"];
            C2_["PQRS"] += O2_["RSPQ"];
        }

        // Hbar += C
        Hbar0_ += C0;
        Hbar1_["pq"] += C1_["pq"];
        Hbar1_["PQ"] += C1_["PQ"];
        Hbar2_["pqrs"] += C2_["pqrs"];
        Hbar2_["pQrS"] += C2_["pQrS"];
        Hbar2_["PQRS"] += C2_["PQRS"];

        // copy C to O for next level commutator
        O1_["pq"] = C1_["pq"];
        O1_["PQ"] = C1_["PQ"];
        O2_["pqrs"] = C2_["pqrs"];
        O2_["pQrS"] = C2_["pQrS"];
        O2_["PQRS"] = C2_["PQRS"];

        // test convergence of C
        double norm_C1 = C1_.norm();
        double norm_C2 = C2_.norm();
        if (print_ > 2) {
            outfile->Printf("\n  n = %3d, C1norm = %20.15f, C2norm = %20.15f", n, norm_C1, norm_C2);
        }
        if (std::sqrt(norm_C2 * norm_C2 + norm_C1 * norm_C1) < ct_threshold) {
            converged = true;
            break;
        }
    }
    if (!converged) {
        outfile->Printf("\n    Warning! Hbar is not converged in %3d-nested commutators!", maxn);
        outfile->Printf("\n    Please increase DSRG_RSC_NCOMM.");
    }
}

double MRDSRG::compute_energy_ldsrg2() {
    // print title
    outfile->Printf("\n\n  ==> Computing MR-LDSRG(2) Energy <==\n");
    outfile->Printf("\n    Reference:");
    outfile->Printf("\n      J. Chem. Phys. 2016, 144, 164114.\n");
    if (options_.get_str("THREEPDC") == "ZERO") {
        outfile->Printf("\n    Skip Lambda3 contributions in [Hbar2, T2].");
    }
    std::string indent(4, ' ');
    std::string dash(99, '-');
    std::string title;
    title += indent + str(boost::format("%5c  %=27s  %=21s  %=21s  %=17s\n") % ' ' %
                          "Energy (a.u.)" % "Non-Diagonal Norm" % "Amplitude RMS" % "Timings (s)");
    title += indent + std::string(7, ' ') + std::string(27, '-') + "  " + std::string(21, '-') +
             "  " + std::string(21, '-') + "  " + std::string(17, '-') + "\n";
    title += indent +
             str(boost::format("%5s  %=16s %=10s  %=10s %=10s  %=10s %=10s  %=8s %=8s\n") %
                 "Iter." % "Corr." % "Delta" % "Hbar1" % "Hbar2" % "T1" % "T2" % "Hbar" % "Amp.");
    title += indent + dash;
    outfile->Printf("\n%s", title.c_str());

    // figure out off-diagonal block labels for Hbar1
    std::vector<std::string> blocks1 = od_one_labels_hp();

    // figure out off-diagonal block labels for Hbar2
    std::vector<std::string> blocks2 = od_two_labels_hhpp();

    // iteration variables
    double Ecorr = 0.0;
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
        diis_manager = std::shared_ptr<DIISManager>(new DIISManager(
            max_diis_vectors, "LDSRG2 DIIS T", DIISManager::LargestError, DIISManager::InCore));
        diis_manager->set_error_vector_size(1, DIISEntry::Pointer, numel);
        diis_manager->set_vector_size(1, DIISEntry::Pointer, numel);
    }

    // start iteration
    do {
        // compute Hbar
        ForteTimer t_hbar;
        if (sequential_Hbar_) {
            compute_hbar_sequential();
        } else {
            compute_hbar();
        }
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
                if (diis_manager->subspace_size() >= min_diis_vectors && cycle) {
                    outfile->Printf(" -> DIIS");

                    diis_manager->extrapolate(1, &(big_T[0]));
                    return_amp_diis(T1_, blocks1, T2_, blocks2, big_T);
                }
            }
        }

        // printing
        outfile->Printf("\n    %5d  %16.12f %10.3e  %10.3e %10.3e  %10.3e "
                        "%10.3e  %8.3f %8.3f",
                        cycle, Ecorr, Edelta, Hbar1od, Hbar2od, T1rms_, T2rms_, time_hbar,
                        time_amp);

        // test convergence
        double rms = T1rms_ > T2rms_ ? T1rms_ : T2rms_;
        if (std::fabs(Edelta) < e_conv && rms < r_conv) {
            converged = true;

            // rebuild Hbar because it is destroyed when updating amplitudes
            if (options_.get_str("RELAX_REF") != "NONE" || options_["AVG_STATE"].size() != 0) {
                if (sequential_Hbar_) {
                    compute_hbar_sequential();
                } else {
                    compute_hbar();
                }
            }
        }
        if (cycle > maxiter) {
            outfile->Printf("\n\n    The computation does not converge in %d "
                            "iterations! Quitting.\n",
                            maxiter);
            converged = true;
            failed = true;
        }

        ++cycle;
    } while (!converged);

    // print summary
    outfile->Printf("\n    %s", dash.c_str());
    outfile->Printf("\n\n  ==> MR-LDSRG(2) Energy Summary <==\n");
    std::vector<std::pair<std::string, double>> energy;
    energy.push_back({"E0 (reference)", Eref_});
    energy.push_back({"MR-LDSRG(2) correlation energy", Ecorr});
    energy.push_back({"MR-LDSRG(2) total energy", Eref_ + Ecorr});
    for (auto& str_dim : energy) {
        outfile->Printf("\n    %-30s = %23.15f", str_dim.first.c_str(), str_dim.second);
    }

    // analyze converged amplitudes
    analyze_amplitudes("Final", T1_, T2_);

    // fail to converge
    if (failed) {
        throw PSIEXCEPTION("The MR-LDSRG(2) computation does not converge.");
    }

    Hbar0_ = Ecorr;
    return Ecorr;
}

void MRDSRG::compute_hbar_qc() {
    std::string dsrg_op = options_.get_str("DSRG_TRANS_TYPE");

    // initialize Hbar with bare H
    Hbar0_ = 0.0;
    Hbar1_["ia"] = F_["ia"];
    Hbar1_["IA"] = F_["IA"];
    Hbar2_["ijab"] = V_["ijab"];
    Hbar2_["iJaB"] = V_["iJaB"];
    Hbar2_["IJAB"] = V_["IJAB"];

    // compute S1 = H + 0.5 * [H, A]
    BlockedTensor S1 = BTF_->build(tensor_type_, "S1", spin_cases({"gg"}), true);
    H1_T1_C1(F_, T1_, 0.5, S1);
    H1_T2_C1(F_, T2_, 0.5, S1);
    H2_T1_C1(V_, T1_, 0.5, S1);
    H2_T2_C1(V_, T2_, 0.5, S1);

    BlockedTensor temp;
    if (dsrg_op == "UNITARY") {
        temp = BTF_->build(tensor_type_, "temp", spin_cases({"gg"}), true);
        temp["pq"] = S1["pq"];
        temp["PQ"] = S1["PQ"];
        S1["pq"] += temp["qp"];
        S1["PQ"] += temp["QP"];
    }

    S1["pq"] += F_["pq"];
    S1["PQ"] += F_["PQ"];

    // compute Hbar = [S1, A]
    //   Step 1: [S1, T]_{ab}^{ij}
    H1_T1_C0(S1, T1_, 2.0, Hbar0_);
    H1_T2_C0(S1, T2_, 2.0, Hbar0_);
    H1_T1_C1(S1, T1_, 1.0, Hbar1_);
    H1_T2_C1(S1, T2_, 1.0, Hbar1_);
    H1_T2_C2(S1, T2_, 1.0, Hbar2_);

    //   Step 2: [S1, T]_{ij}^{ab}
    if (dsrg_op == "UNITARY") {
        temp = BTF_->build(tensor_type_, "temp", spin_cases({"ph"}), true);
        H1_T1_C1(S1, T1_, 1.0, temp);
        H1_T2_C1(S1, T2_, 1.0, temp);
        Hbar1_["ia"] += temp["ai"];
        Hbar1_["IA"] += temp["AI"];

        for (const std::string& block : {"pphh", "pPhH", "PPHH"}) {
            // spin cases
            std::string ijab{"ijab"};
            std::string abij{"abij"};
            if (isupper(block[1])) {
                ijab = "iJaB";
                abij = "aBiJ";
            }
            if (isupper(block[0])) {
                ijab = "IJAB";
                abij = "ABIJ";
            }

            temp = BTF_->build(tensor_type_, "temp", {block}, true);
            H1_T2_C2(S1, T2_, 1.0, temp);
            Hbar2_[ijab] += temp[abij];
        }
    }

    // compute Hbar = [S2, A]
    // compute S2 = H + 0.5 * [H, A] in batches of spin
    for (const std::string& block : {"gggg", "gGgG", "GGGG"}) {
        // spin cases for S2
        int spin = 0;
        std::string pqrs{"pqrs"};
        if (isupper(block[1])) {
            spin = 1;
            pqrs = "pQrS";
        }
        if (isupper(block[0])) {
            spin = 2;
            pqrs = "PQRS";
        }

        // 0.5 * [H, T]
        BlockedTensor S2 = BTF_->build(tensor_type_, "S2", {block}, true);
        H1_T2_C2(F_, T2_, 0.5, S2);
        H2_T1_C2(V_, T1_, 0.5, S2);
        H2_T2_C2(V_, T2_, 0.5, S2);

        // 0.5 * [H, T]^+
        if (dsrg_op == "UNITARY") {
            if (spin == 0) {
                tensor_add_HC_aa(S2);
            } else if (spin == 1) {
                tensor_add_HC_ab(S2);
            } else if (spin == 2) {
                tensor_add_HC_aa(S2, false);
            }
        }

        // add bare Hamiltonian contribution
        S2[pqrs] += V_[pqrs];

        // compute Hbar = [S2, A]
        //   Step 1: [S2, T]_{ab}^{ij}
        H2_T1_C0(S2, T1_, 2.0, Hbar0_);
        H2_T2_C0(S2, T2_, 2.0, Hbar0_);
        H2_T1_C1(S2, T1_, 1.0, Hbar1_);
        H2_T2_C1(S2, T2_, 1.0, Hbar1_);
        H2_T1_C2(S2, T1_, 1.0, Hbar2_);
        H2_T2_C2(S2, T2_, 1.0, Hbar2_);

        //   Step 2: [S2, T]_{ij}^{ab}
        if (dsrg_op == "UNITARY") {
            temp = BTF_->build(tensor_type_, "temp", spin_cases({"ph"}), true);
            H2_T1_C1(S2, T1_, 1.0, temp);
            H2_T2_C1(S2, T2_, 1.0, temp);
            Hbar1_["ia"] += temp["ai"];
            Hbar1_["IA"] += temp["AI"];

            for (const std::string& block : {"pphh", "pPhH", "PPHH"}) {
                // spin cases
                std::string ijab{"ijab"};
                std::string abij{"abij"};
                if (isupper(block[1])) {
                    ijab = "iJaB";
                    abij = "aBiJ";
                }
                if (isupper(block[0])) {
                    ijab = "IJAB";
                    abij = "ABIJ";
                }

                temp = BTF_->build(tensor_type_, "temp", {block}, true);
                H2_T1_C2(S2, T1_, 1.0, temp);
                H2_T2_C2(S2, T2_, 1.0, temp);
                Hbar2_[ijab] += temp[abij];
            }
        }
    }
}

void MRDSRG::compute_hbar_qc_sequential() {

    outfile->Printf("\n  ==> compute_hbar_qc_sequential() <==\n");
    std::string dsrg_op = options_.get_str("DSRG_TRANS_TYPE");

    // initialize Hbar with bare H
    Hbar0_ = 0.0;
    Hbar1_["ia"] = F_["ia"];
    Hbar1_["IA"] = F_["IA"];
    Hbar2_["ijab"] = V_["ijab"];
    Hbar2_["iJaB"] = V_["iJaB"];
    Hbar2_["IJAB"] = V_["IJAB"];

    // compute S1 = H + 0.5 * [H, A]
    BlockedTensor S1 = BTF_->build(tensor_type_, "S1", spin_cases({"gg"}), true);
    H1_T1_C1(F_, T1_, 0.5, S1);
    H2_T1_C1(V_, T1_, 0.5, S1);

    BlockedTensor temp;
    if (dsrg_op == "UNITARY") {
        temp = BTF_->build(tensor_type_, "temp", spin_cases({"gg"}), true);
        temp["pq"] = S1["pq"];
        temp["PQ"] = S1["PQ"];
        S1["pq"] += temp["qp"];
        S1["PQ"] += temp["QP"];
    }

    S1["pq"] += F_["pq"];
    S1["PQ"] += F_["PQ"];

    // compute Hbar = [S1, A]
    //   Step 1: [S1, T]_{ab}^{ij}
    H1_T1_C0(S1, T1_, 2.0, Hbar0_);
    H1_T1_C1(S1, T1_, 1.0, Hbar1_);

    //   Step 2: [S1, T]_{ij}^{ab}
    if (dsrg_op == "UNITARY") {
        temp = BTF_->build(tensor_type_, "temp", spin_cases({"ph"}), true);
        H1_T1_C1(S1, T1_, 1.0, temp);
        Hbar1_["ia"] += temp["ai"];
        Hbar1_["IA"] += temp["AI"];

        for (const std::string& block : {"pphh", "pPhH", "PPHH"}) {
            // spin cases
            std::string ijab{"ijab"};
            std::string abij{"abij"};
            if (isupper(block[1])) {
                ijab = "iJaB";
                abij = "aBiJ";
            }
            if (isupper(block[0])) {
                ijab = "IJAB";
                abij = "ABIJ";
            }

            temp = BTF_->build(tensor_type_, "temp", {block}, true);
            Hbar2_[ijab] += temp[abij];
        }
    }

    // compute Hbar = [S2, A]
    // compute S2 = H + 0.5 * [H, A] in batches of spin
    for (const std::string& block : {"gggg", "gGgG", "GGGG"}) {
        // spin cases for S2
        int spin = 0;
        std::string pqrs{"pqrs"};
        if (isupper(block[1])) {
            spin = 1;
            pqrs = "pQrS";
        }
        if (isupper(block[0])) {
            spin = 2;
            pqrs = "PQRS";
        }

        // 0.5 * [H, T]
        BlockedTensor S2 = BTF_->build(tensor_type_, "S2", {block}, true);
        H2_T1_C2(V_, T1_, 0.5, S2);

        // 0.5 * [H, T]^+
        if (dsrg_op == "UNITARY") {
            if (spin == 0) {
                tensor_add_HC_aa(S2);
            } else if (spin == 1) {
                tensor_add_HC_ab(S2);
            } else if (spin == 2) {
                tensor_add_HC_aa(S2, false);
            }
        }

        // add bare Hamiltonian contribution
        S2[pqrs] += V_[pqrs];

        // compute Hbar = [S2, A]
        //   Step 1: [S2, T]_{ab}^{ij}
        H2_T1_C0(S2, T1_, 2.0, Hbar0_);
        H2_T1_C1(S2, T1_, 1.0, Hbar1_);
        H2_T1_C2(S2, T1_, 1.0, Hbar2_);

        //   Step 2: [S2, T]_{ij}^{ab}
        if (dsrg_op == "UNITARY") {
            temp = BTF_->build(tensor_type_, "temp", spin_cases({"ph"}), true);
            H2_T1_C1(S2, T1_, 1.0, temp);
            Hbar1_["ia"] += temp["ai"];
            Hbar1_["IA"] += temp["AI"];

            for (const std::string& block : {"pphh", "pPhH", "PPHH"}) {
                // spin cases
                std::string ijab{"ijab"};
                std::string abij{"abij"};
                if (isupper(block[1])) {
                    ijab = "iJaB";
                    abij = "aBiJ";
                }
                if (isupper(block[0])) {
                    ijab = "IJAB";
                    abij = "ABIJ";
                }

                temp = BTF_->build(tensor_type_, "temp", {block}, true);
                H2_T1_C2(S2, T1_, 1.0, temp);
                Hbar2_[ijab] += temp[abij];
            }
        }
    }

    // initialize Hbar with bare H
//    Hbar0_ = 0.0;
//    Hbar1_["ia"] = F_["ia"];
//    Hbar1_["IA"] = F_["IA"];
//    Hbar2_["ijab"] = V_["ijab"];
//    Hbar2_["iJaB"] = V_["iJaB"];
//    Hbar2_["IJAB"] = V_["IJAB"];

    // compute S1 = H + 0.5 * [H, A]
    S1 = BTF_->build(tensor_type_, "S1", spin_cases({"gg"}), true);
    H1_T2_C1(F_, T2_, 0.5, S1);
    H2_T2_C1(V_, T2_, 0.5, S1);

    if (dsrg_op == "UNITARY") {
        temp = BTF_->build(tensor_type_, "temp", spin_cases({"gg"}), true);
        temp["pq"] = S1["pq"];
        temp["PQ"] = S1["PQ"];
        S1["pq"] += temp["qp"];
        S1["PQ"] += temp["QP"];
    }

    S1["pq"] += F_["pq"];
    S1["PQ"] += F_["PQ"];

    // compute Hbar = [S1, A]
    //   Step 1: [S1, T]_{ab}^{ij}
    H1_T2_C0(S1, T2_, 2.0, Hbar0_);
    H1_T2_C1(S1, T2_, 1.0, Hbar1_);
    H1_T2_C2(S1, T2_, 1.0, Hbar2_);

    //   Step 2: [S1, T]_{ij}^{ab}
    if (dsrg_op == "UNITARY") {
        temp = BTF_->build(tensor_type_, "temp", spin_cases({"ph"}), true);
        H1_T2_C1(S1, T2_, 1.0, temp);
        Hbar1_["ia"] += temp["ai"];
        Hbar1_["IA"] += temp["AI"];

        for (const std::string& block : {"pphh", "pPhH", "PPHH"}) {
            // spin cases
            std::string ijab{"ijab"};
            std::string abij{"abij"};
            if (isupper(block[1])) {
                ijab = "iJaB";
                abij = "aBiJ";
            }
            if (isupper(block[0])) {
                ijab = "IJAB";
                abij = "ABIJ";
            }

            temp = BTF_->build(tensor_type_, "temp", {block}, true);
            H1_T2_C2(S1, T2_, 1.0, temp);
            Hbar2_[ijab] += temp[abij];
        }
    }

    // compute Hbar = [S2, A]
    // compute S2 = H + 0.5 * [H, A] in batches of spin
    for (const std::string& block : {"gggg", "gGgG", "GGGG"}) {
        // spin cases for S2
        int spin = 0;
        std::string pqrs{"pqrs"};
        if (isupper(block[1])) {
            spin = 1;
            pqrs = "pQrS";
        }
        if (isupper(block[0])) {
            spin = 2;
            pqrs = "PQRS";
        }

        // 0.5 * [H, T]
        BlockedTensor S2 = BTF_->build(tensor_type_, "S2", {block}, true);
        H1_T2_C2(F_, T2_, 0.5, S2);
        H2_T2_C2(V_, T2_, 0.5, S2);

        // 0.5 * [H, T]^+
        if (dsrg_op == "UNITARY") {
            if (spin == 0) {
                tensor_add_HC_aa(S2);
            } else if (spin == 1) {
                tensor_add_HC_ab(S2);
            } else if (spin == 2) {
                tensor_add_HC_aa(S2, false);
            }
        }

        // add bare Hamiltonian contribution
        S2[pqrs] += V_[pqrs];

        // compute Hbar = [S2, A]
        //   Step 1: [S2, T]_{ab}^{ij}
        H2_T2_C0(S2, T2_, 2.0, Hbar0_);
        H2_T2_C1(S2, T2_, 1.0, Hbar1_);
        H2_T2_C2(S2, T2_, 1.0, Hbar2_);

        //   Step 2: [S2, T]_{ij}^{ab}
        if (dsrg_op == "UNITARY") {
            temp = BTF_->build(tensor_type_, "temp", spin_cases({"ph"}), true);
            H2_T2_C1(S2, T2_, 1.0, temp);
            Hbar1_["ia"] += temp["ai"];
            Hbar1_["IA"] += temp["AI"];

            for (const std::string& block : {"pphh", "pPhH", "PPHH"}) {
                // spin cases
                std::string ijab{"ijab"};
                std::string abij{"abij"};
                if (isupper(block[1])) {
                    ijab = "iJaB";
                    abij = "aBiJ";
                }
                if (isupper(block[0])) {
                    ijab = "IJAB";
                    abij = "ABIJ";
                }

                temp = BTF_->build(tensor_type_, "temp", {block}, true);
                H2_T2_C2(S2, T2_, 1.0, temp);
                Hbar2_[ijab] += temp[abij];
            }
        }
    }
}

double MRDSRG::compute_energy_ldsrg2_qc() {
    // print title
    outfile->Printf("\n\n  ==> Computing MR-LDSRG(2)-QC Energy <==\n");
    outfile->Printf("\n    DSRG transformed Hamiltonian is truncated to "
                    "quadratic nested commutator.");
    outfile->Printf("\n    Reference:");
    outfile->Printf("\n      J. Chem. Phys. (in preparation)\n");
    std::string indent(4, ' ');
    std::string dash(99, '-');
    std::string title;
    title += indent + str(boost::format("%5c  %=27s  %=21s  %=21s  %=17s\n") % ' ' %
                          "Energy (a.u.)" % "Non-Diagonal Norm" % "Amplitude RMS" % "Timings (s)");
    title += indent + std::string(7, ' ') + std::string(27, '-') + "  " + std::string(21, '-') +
             "  " + std::string(21, '-') + "  " + std::string(17, '-') + "\n";
    title += indent +
             str(boost::format("%5s  %=16s %=10s  %=10s %=10s  %=10s %=10s  %=8s %=8s\n") %
                 "Iter." % "Corr." % "Delta" % "Hbar1" % "Hbar2" % "T1" % "T2" % "Hbar" % "Amp.");
    title += indent + dash;
    outfile->Printf("\n%s", title.c_str());

    // figure out off-diagonal blocks labels
    std::vector<std::string> blocks1 = od_one_labels_hp();
    std::vector<std::string> blocks2 = od_two_labels_hhpp();

    // iteration variables
    double Ecorr = 0.0;
    int cycle = 0, maxiter = options_.get_int("MAXITER");
    double e_conv = options_.get_double("E_CONVERGENCE");
    double r_conv = options_.get_double("R_CONVERGENCE");
    bool converged = false, failed = false;
    Hbar1_ = BTF_->build(tensor_type_, "Hbar1", spin_cases({"hp"}));
    Hbar2_ = BTF_->build(tensor_type_, "Hbar2", spin_cases({"hhpp"}));
    O1_ = BTF_->build(tensor_type_, "O1", spin_cases({"aa"}));
    DT1_ = BTF_->build(tensor_type_, "DT1", spin_cases({"hp"}));
    DT2_ = BTF_->build(tensor_type_, "DT2", spin_cases({"hhpp"}));
    std::vector<double> big_T, big_DT;
    size_t numel = vector_size_diis(T1_, blocks1, T2_, blocks2);
    BlockedTensor::set_expert_mode(true);

    // setup DIIS
    std::shared_ptr<DIISManager> diis_manager;
    int max_diis_vectors = options_.get_int("DIIS_MAX_VECS");
    int min_diis_vectors = options_.get_int("DIIS_MIN_VECS");
    if (max_diis_vectors > 0) {
        diis_manager = std::shared_ptr<DIISManager>(new DIISManager(
            max_diis_vectors, "LDSRG2 DIIS T", DIISManager::LargestError, DIISManager::InCore));
        diis_manager->set_error_vector_size(1, DIISEntry::Pointer, numel);
        diis_manager->set_vector_size(1, DIISEntry::Pointer, numel);
    }

    // start iteration
    do {
        // compute Hbar
        ForteTimer t_hbar;
        if (sequential_Hbar_) {
            compute_hbar_qc_sequential();
        } else {
            compute_hbar_qc();
        }
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
                if (diis_manager->subspace_size() >= min_diis_vectors && cycle) {
                    outfile->Printf(" -> DIIS");

                    diis_manager->extrapolate(1, &(big_T[0]));
                    return_amp_diis(T1_, blocks1, T2_, blocks2, big_T);
                }
            }
        }

        // printing
        outfile->Printf("\n    %5d  %16.12f %10.3e  %10.3e %10.3e  %10.3e "
                        "%10.3e  %8.3f %8.3f",
                        cycle, Ecorr, Edelta, Hbar1od, Hbar2od, T1rms_, T2rms_, time_hbar,
                        time_amp);

        // test convergence
        double rms = T1rms_ > T2rms_ ? T1rms_ : T2rms_;
        if (std::fabs(Edelta) < e_conv && rms < r_conv) {
            converged = true;

            // rebuild Hbar because it is destroyed when updating amplitudes
            if (options_.get_str("RELAX_REF") != "NONE" || options_["AVG_STATE"].size() != 0) {
                if (sequential_Hbar_) {
                    compute_hbar_qc_sequential();
                } else {
                    compute_hbar_qc();
                }
            }
        }
        if (cycle > maxiter) {
            outfile->Printf("\n\n    The computation does not converge in %d "
                            "iterations! Quitting.\n",
                            maxiter);
            converged = true;
            failed = true;
        }

        ++cycle;
    } while (!converged);

    // print summary
    outfile->Printf("\n    %s", dash.c_str());
    outfile->Printf("\n\n  ==> MR-LDSRG(2)-QC Energy Summary <==\n");
    std::vector<std::pair<std::string, double>> energy;
    energy.push_back({"E0 (reference)", Eref_});
    energy.push_back({"MR-LDSRG(2)-QC correlation energy", Ecorr});
    energy.push_back({"MR-LDSRG(2)-QC total energy", Eref_ + Ecorr});
    for (auto& str_dim : energy) {
        outfile->Printf("\n    %-35s = %23.15f", str_dim.first.c_str(), str_dim.second);
    }

    // analyze converged amplitudes
    analyze_amplitudes("Final", T1_, T2_);

    // fail to converge
    if (failed) {
        throw PSIEXCEPTION("The MR-LDSRG(2)-QC computation does not converge.");
    }

    Hbar0_ = Ecorr;
    return Ecorr;
}

void MRDSRG::tensor_add_HC_aa(BlockedTensor& H2, const bool& spin_alpha) {
    // unique blocks
    std::vector<std::string> diag_blocks{"cccc", "caca", "cvcv", "aaaa", "avav", "vvvv"};
    std::vector<std::string> od_blocks{"ccca", "cccv", "ccaa", "ccav", "ccvv",
                                       "cacv", "caaa", "caav", "cavv", "cvaa",
                                       "cvav", "cvvv", "aaav", "aavv", "avvv"};
    if (!spin_alpha) {
        for (std::string& block : diag_blocks) {
            for (int i = 0; i < 4; ++i) {
                block[i] = std::toupper(block[i]);
            }
        }
        for (std::string& block : od_blocks) {
            for (int i = 0; i < 4; ++i) {
                block[i] = std::toupper(block[i]);
            }
        }
    }

    // figure out if permutations needed
    std::vector<std::vector<int>> diag_blocks_permutation;
    std::vector<std::vector<int>> od_blocks_permutation;
    for (const std::string& block : diag_blocks) {
        int m = block[0] == block[1] ? 0 : 1;
        int n = block[2] == block[3] ? 0 : 1;
        diag_blocks_permutation.push_back({m, n});
    }
    for (const std::string& block : od_blocks) {
        int m = block[0] == block[1] ? 0 : 1;
        int n = block[2] == block[3] ? 0 : 1;
        od_blocks_permutation.push_back({m, n});
    }

    // add Hermitian conjugate to itself for diagonal blocks (square matrix)
    for (const std::string& block : diag_blocks) {
        const std::vector<size_t> dims = H2.block(block).dims();
        size_t half = dims[0] * dims[1];
        for (size_t i = 0; i < half; ++i) {
            for (size_t j = i; j < half; ++j) {
                size_t idx = i * half + j;
                size_t idx_hc = j * half + i;
                H2.block(block).data()[idx] += H2.block(block).data()[idx_hc];
                H2.block(block).data()[idx_hc] = H2.block(block).data()[idx];
            }
        }

        // consider permutations
        if (block[0] != block[1]) {
            std::string p(1, block[0]);
            std::string q(1, block[1]);
            std::string block1 = q + p + p + q;
            std::string block2 = p + q + q + p;
            std::string block3 = q + p + q + p;
            H2.block(block1)("qprs") = -1.0 * H2.block(block)("pqrs");
            H2.block(block2)("pqsr") = -1.0 * H2.block(block)("pqrs");
            H2.block(block3)("qpsr") = H2.block(block)("pqrs");
        }
    }

    // add Hermitian conjugate to itself for off-diagonal blocks
    for (const std::string& block : od_blocks) {
        // figure out Hermitian conjugate block label
        std::string p(1, block[0]);
        std::string q(1, block[1]);
        std::string r(1, block[2]);
        std::string s(1, block[3]);
        std::string block_hc = r + s + p + q;

        // add Hermitian conjugate
        H2.block(block)("pqrs") += H2.block(block_hc)("rspq");
        H2.block(block_hc)("rspq") = H2.block(block)("pqrs");

        // consider permutations
        bool permute_1st_half = (block[0] == block[1]) ? false : true;
        bool permute_2nd_half = (block[2] == block[3]) ? false : true;

        if (permute_1st_half) {
            std::string block1 = q + p + r + s;
            std::string block1_hc = r + s + q + p;
            H2.block(block1)("qprs") = -1.0 * H2.block(block)("pqrs");
            H2.block(block1_hc)("rsqp") = -1.0 * H2.block(block_hc)("rspq");
        }

        if (permute_2nd_half) {
            std::string block2 = p + q + s + r;
            std::string block2_hc = s + r + p + q;
            H2.block(block2)("pqsr") = -1.0 * H2.block(block)("pqrs");
            H2.block(block2_hc)("srpq") = -1.0 * H2.block(block_hc)("rspq");
        }

        if (permute_1st_half && permute_2nd_half) {
            std::string block3 = q + p + s + r;
            std::string block3_hc = s + r + q + p;
            H2.block(block3)("qpsr") = H2.block(block)("pqrs");
            H2.block(block3_hc)("srqp") = H2.block(block_hc)("rspq");
        }
    }
}

void MRDSRG::tensor_add_HC_ab(BlockedTensor& H2) {
    // labels for half of tensor
    std::vector<std::string> labels_half{"cC", "cA", "cV", "aC", "aA", "aV", "vC", "vA", "vV"};
    std::vector<std::string> diag_blocks;
    std::vector<std::string> od_blocks;

    // figure out diagonal blocks labels
    for (const std::string& label : labels_half) {
        diag_blocks.push_back(label + label);
    }

    // figure out off-diagonal blocks labels
    for (int i = 0; i < 9; ++i) {
        for (int j = i + 1; j < 9; ++j) {
            od_blocks.push_back(labels_half[i] + labels_half[j]);
        }
    }

    // add Hermitian conjugate to itself for diagonal blocks (square matrix)
    for (const std::string& block : diag_blocks) {
        const std::vector<size_t> dims = H2.block(block).dims();
        size_t half = dims[0] * dims[1];
        for (size_t i = 0; i < half; ++i) {
            for (size_t j = i; j < half; ++j) {
                size_t idx = i * half + j;
                size_t idx_hc = j * half + i;
                H2.block(block).data()[idx] += H2.block(block).data()[idx_hc];
                H2.block(block).data()[idx_hc] = H2.block(block).data()[idx];
            }
        }
    }

    // add Hermitian conjugate to itself for off-diagonal blocks
    for (const std::string& block : od_blocks) {
        // figure out Hermitian conjugate block label
        std::string p(1, block[0]);
        std::string q(1, block[1]);
        std::string r(1, block[2]);
        std::string s(1, block[3]);
        std::string block_hc = r + s + p + q;

        // add Hermitian conjugate
        H2.block(block)("pqrs") += H2.block(block_hc)("rspq");
        H2.block(block_hc)("rspq") = H2.block(block)("pqrs");
    }
}

double MRDSRG::Hbar1od_norm(const std::vector<std::string>& blocks) {
    double norm = 0.0;

    for (auto& block : blocks) {
        double norm_block = Hbar1_.block(block).norm();
        norm += 2.0 * norm_block * norm_block;
    }
    norm = std::sqrt(norm);

    return norm;
}

double MRDSRG::Hbar2od_norm(const std::vector<std::string>& blocks) {
    double norm = 0.0;

    for (auto& block : blocks) {
        double norm_block = Hbar2_.block(block).norm();
        norm += 2.0 * norm_block * norm_block;
    }
    norm = std::sqrt(norm);

    return norm;
}

std::vector<double> MRDSRG::copy_amp_diis(BlockedTensor& T1, const std::vector<std::string>& label1,
                                          BlockedTensor& T2,
                                          const std::vector<std::string>& label2) {
    std::vector<double> out;

    for (const auto& block : label1) {
        out.insert(out.end(), T1.block(block).data().begin(), T1.block(block).data().end());
    }
    for (const auto& block : label2) {
        out.insert(out.end(), T2.block(block).data().begin(), T2.block(block).data().end());
    }

    return out;
}

size_t MRDSRG::vector_size_diis(BlockedTensor& T1, const std::vector<std::string>& label1,
                                BlockedTensor& T2, const std::vector<std::string>& label2) {
    size_t total_elements = 0;
    for (const auto& block : label1) {
        total_elements += T1.block(block).numel();
    }
    for (const auto& block : label2) {
        total_elements += T2.block(block).numel();
    }
    return total_elements;
}

void MRDSRG::return_amp_diis(BlockedTensor& T1, const std::vector<std::string>& label1,
                             BlockedTensor& T2, const std::vector<std::string>& label2,
                             const std::vector<double>& data) {
    // test data
    std::map<std::string, size_t> num_elements;
    size_t total_elements = 0;

    for (const auto& block : label1) {
        size_t numel = T1.block(block).numel();
        num_elements[block] = total_elements;
        total_elements += numel;
    }
    for (const auto& block : label2) {
        size_t numel = T2.block(block).numel();
        num_elements[block] = total_elements;
        total_elements += numel;
    }

    if (data.size() != total_elements) {
        throw PSIEXCEPTION("Number of elements in T1 and T2 do not match the bid data vector");
    }

    // transfer data
    for (const auto& block : label1) {
        std::vector<double>::const_iterator start = data.begin() + num_elements[block];
        std::vector<double>::const_iterator end = start + T1.block(block).numel();
        std::vector<double> T1_this_block(start, end);
        T1.block(block).data() = T1_this_block;
    }
    for (const auto& block : label2) {
        std::vector<double>::const_iterator start = data.begin() + num_elements[block];
        std::vector<double>::const_iterator end = start + T2.block(block).numel();
        std::vector<double> T2_this_block(start, end);
        T2.block(block).data() = T2_this_block;
    }
}
}
}
