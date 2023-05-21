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

#include <cctype>

#include "psi4/libdiis/diismanager.h"
#include "psi4/libpsi4util/PsiOutStream.h"

#include "helpers/timer.h"
#include "sa_mrdsrg.h"

using namespace psi;

namespace forte {

double SA_MRDSRG::compute_energy_ldsrg2() {
    // variant name
    std::string level = (corrlv_string_ == "LDSRG2_QC") ? "-C2" : "";

    // print title
    outfile->Printf("\n\n  ==> Computing MR-LDSRG(2)%s Energy <==\n", level.c_str());
    outfile->Printf("\n    Reference:");
    outfile->Printf("\n      J. Chem. Phys. 2016, 144, 164114.\n");

    timer ldsrg2("Energy LDSRG(2)");

    if (!do_cu3_) {
        outfile->Printf("\n    Skip 3-cumulant contributions in [O2, T2].");
    }

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

    // figure out off-diagonal block labels for Hbar1 and Hbar2
    std::vector<std::string> blocks1 = od_one_labels_hp();
    std::vector<std::string> blocks2 = od_two_labels_hhpp();

    // iteration variables
    double Ecorr = 0.0;
    bool converged = false;

    setup_ldsrg2_tensors();

    // setup DIIS
    if (diis_start_ > 0) {
        diis_manager_init();
    }

    // start iteration
    for (int cycle = 1; cycle <= maxiter_; ++cycle) {
        // use DT2_ as an intermediate used for compute Hbar
        DT2_["ijab"] = 2.0 * T2_["ijab"];
        DT2_["ijab"] -= T2_["ijba"];

        // compute Hbar
        local_timer t_hbar;
        timer hbar("Compute Hbar");
        if (corrlv_string_ == "LDSRG2_QC") {
            compute_hbar_qc();
        } else {
            if (sequential_Hbar_) {
                compute_hbar_sequential();
            } else {
                compute_hbar();
            }
        }
        hbar.stop();
        double Edelta = Hbar0_ - Ecorr;
        Ecorr = Hbar0_;
        double time_hbar = t_hbar.get();

        timer od("Off-diagonal Hbar");
        // compute norms of off-diagonal Hbar
        double Hbar1od = Hbar_od_norm(1, blocks1);
        double Hbar2od = Hbar_od_norm(2, blocks2);

        // update amplitudes
        local_timer t_amp;
        update_t();
        double time_amp = t_amp.get();
        od.stop();

        // printing
        outfile->Printf("\n    %4d   %16.12f %10.3e  %10.3e %10.3e  %10.3e %10.3e  %8.3f %8.3f",
                        cycle, Ecorr, Edelta, Hbar1od, Hbar2od, T1rms_, T2rms_, time_hbar,
                        time_amp);

        timer diis("DIIS");
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
        diis.stop();

        // test convergence
        double rms = T1rms_ > T2rms_ ? T1rms_ : T2rms_;
        if (std::fabs(Edelta) < e_conv_ && rms < r_conv_) {
            converged = true;
            break;
        }

        if (cycle == maxiter_) {
            outfile->Printf("\n\n    The computation does not converge in %d iterations!\n",
                            maxiter_);
        }
        if (cycle > 5 and std::fabs(rms) > 10.0) {
            outfile->Printf("\n\n    Large RMS for amplitudes. Likely no convergence. Quitting.\n");
        }
    }

    // clean up raw pointers used in DIIS
    if (diis_start_ > 0) {
        diis_manager_cleanup();
    }

    timer final("Summary LDSRG(2)");
    // print summary
    outfile->Printf("\n    %s", dash.c_str());
    outfile->Printf("\n\n  ==> MR-LDSRG(2)%s Energy Summary <==\n", level.c_str());
    std::vector<std::pair<std::string, double>> energy;
    energy.push_back({"E0 (reference)", Eref_});
    energy.push_back({"MR-LDSRG(2) correlation energy", Ecorr});
    energy.push_back({"MR-LDSRG(2) total energy", Eref_ + Ecorr});
    for (auto& str_dim : energy) {
        outfile->Printf("\n    %-30s = %23.15f", str_dim.first.c_str(), str_dim.second);
    }

    // analyze converged amplitudes
    analyze_amplitudes("Final", T1_, T2_);

    // dump amplitudes to disk
    dump_amps_to_disk();

    // fail to converge
    if (!converged) {
        clean_checkpoints(); // clean amplitudes in scratch directory
        throw psi::PSIEXCEPTION("The MR-LDSRG(2) computation does not converge.");
    }
    final.stop();

    Hbar0_ = Ecorr;
    return Ecorr;
}

void SA_MRDSRG::compute_hbar() {
    if (print_ > 2) {
        outfile->Printf("\n\n  ==> Computing the DSRG Transformed Hamiltonian <==\n");
    }

    // copy bare Hamiltonian to Hbar
    Hbar0_ = 0.0;
    Hbar1_["pq"] = F_["pq"];

    if (eri_df_) {
        Hbar2_["pqrs"] = B_["gpr"] * B_["gqs"];
    } else {
        Hbar2_["pqrs"] = V_["pqrs"];
        O2_["pqrs"] = Hbar2_["pqrs"];
    }

    // temporary Hamiltonian used in every iteration
    O1_["pq"] = F_["pq"];

    // iteration variables
    bool converged = false;

    // compute Hbar recursively
    for (int n = 1; n <= rsc_ncomm_; ++n) {
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
        if (n == 1 && eri_df_) {
            V_T1_C0_DF(B_, T1_, factor, C0);
            V_T2_C0_DF(B_, T2_, DT2_, factor, C0);
        } else {
            H2_T1_C0(O2_, T1_, factor, C0);
            H2_T2_C0(O2_, T2_, DT2_, factor, C0);
        }

        // one-body
        H1_T1_C1(O1_, T1_, factor, C1_);
        H1_T2_C1(O1_, T2_, factor, C1_);
        if (n == 1 && eri_df_) {
            V_T1_C1_DF(B_, T1_, factor, C1_);
            V_T2_C1_DF(B_, T2_, DT2_, factor, C1_);
        } else {
            H2_T1_C1(O2_, T1_, factor, C1_);
            H2_T2_C1(O2_, T2_, DT2_, factor, C1_);
        }

        // two-body
        H1_T2_C2(O1_, T2_, factor, C2_);
        if (n == 1 && eri_df_) {
            V_T1_C2_DF(B_, T1_, factor, C2_);
            V_T2_C2_DF(B_, T2_, DT2_, factor, C2_);
        } else {
            H2_T1_C2(O2_, T1_, factor, C2_);
            H2_T2_C2(O2_, T2_, DT2_, factor, C2_);
        }

        // printing level
        if (print_ > 2) {
            std::string dash(38, '-');
            outfile->Printf("\n    %s\n", dash.c_str());
        }

        // [H, A] = [H, T] + [H, T]^dagger
        C0 *= 2.0;
        O1_["pq"] = C1_["pq"];
        C1_["pq"] += O1_["qp"];
        O2_["pqrs"] = C2_["pqrs"];
        C2_["pqrs"] += O2_["rspq"];

        // Hbar += C
        Hbar0_ += C0;
        Hbar1_["pq"] += C1_["pq"];
        Hbar2_["pqrs"] += C2_["pqrs"];

        // copy C to O for next level commutator
        O1_["pq"] = C1_["pq"];
        O2_["pqrs"] = C2_["pqrs"];

        // test convergence of C
        double norm_C1 = C1_.norm();
        double norm_C2 = C2_.norm();
        if (print_ > 2) {
            outfile->Printf("\n  n: %3d, C0: %20.15f, C1 max: %20.15f, C2 max: %20.15f", n, C0,
                            C1_.norm(0), C2_.norm(0));
        }
        if (std::sqrt(norm_C2 * norm_C2 + norm_C1 * norm_C1) < rsc_conv_) {
            converged = true;
            break;
        }
    }
    if (!converged) {
        outfile->Printf("\n    Warning! Hbar is not converged in %3d-nested commutators!",
                        rsc_ncomm_);
        outfile->Printf("\n    Please increase DSRG_RSC_NCOMM.");
    }
}

void SA_MRDSRG::compute_hbar_sequential() {
    if (print_ > 2) {
        outfile->Printf("\n\n  ==> Computing the DSRG Transformed Hamiltonian <==\n");
    }

    timer rotation("Hbar T1 rotation");

    ambit::BlockedTensor A1;
    A1 = BTF_->build(tensor_type_, "A1 Amplitudes", {"gg"}, true);
    A1["ia"] = T1_["ia"];
    A1["ai"] -= T1_["ia"];

    size_t ncmo = core_mos_.size() + actv_mos_.size() + virt_mos_.size();

    auto A1_m = std::make_shared<psi::Matrix>("A1 alpha", ncmo, ncmo);
    A1.iterate([&](const std::vector<size_t>& i, const std::vector<SpinType>&, double& value) {
        A1_m->set(i[0], i[1], value);
    });

    // >=3 is required for high energy convergence
    A1_m->expm(3);

    ambit::BlockedTensor U1;
    U1 = BTF_->build(tensor_type_, "Transformer", {"gg"}, true);
    U1.iterate([&](const std::vector<size_t>& i, const std::vector<SpinType>&, double& value) {
        value = A1_m->get(i[0], i[1]);
    });

    /// Recompute Hbar0 (ref. energy + T1 correlation), Hbar1 (Fock), and Hbar2 (aptei)
    /// E = 0.5 * ( H["ji"] + F["ji] ) * D1["ij"] + 0.25 * V["xyuv"] * L2["uvxy"]

    // Hbar1 is now "bare H1"
    Hbar1_["rs"] = U1["rp"] * H_["pq"] * U1["sq"];

    Hbar0_ = 0.0;
    Hbar1_.block("cc").iterate([&](const std::vector<size_t>& i, double& value) {
        if (i[0] == i[1])
            Hbar0_ += value;
    });
    Hbar0_ += 0.5 * Hbar1_["uv"] * L1_["vu"];

    // for simplicity, create a core-core density matrix
    BlockedTensor D1c = BTF_->build(tensor_type_, "L1 core", {"cc"});
    for (size_t m = 0, nc = core_mos_.size(); m < nc; ++m) {
        D1c.block("cc").data()[m * nc + m] = 2.0;
    }

    // Hbar1 becomes "Fock"
    ambit::BlockedTensor B;
    if (eri_df_) {
        B = BTF_->build(tensor_type_, "B 3-idx", {"Lgg"}, true);
        B["grs"] = U1["rp"] * B_["gpq"] * U1["sq"];

        BlockedTensor temp = BTF_->build(tensor_type_, "B temp", {"L"}, true);
        temp["g"] = B["gmn"] * D1c["mn"];
        temp["g"] += B["guv"] * L1_["uv"];
        Hbar1_["pq"] += temp["g"] * B["gpq"];

        Hbar1_["pq"] -= 0.5 * B["gpm"] * B["gnq"] * D1c["mn"];
        Hbar1_["pq"] -= 0.5 * B["gpu"] * B["gvq"] * L1_["uv"];
    } else {
        Hbar2_["pqrs"] = U1["pt"] * U1["qo"] * V_["t,o,g0,g1"] * U1["r,g0"] * U1["s,g1"];

        Hbar1_["pq"] += Hbar2_["pnqm"] * D1c["mn"];
        Hbar1_["pq"] -= 0.5 * Hbar2_["npqm"] * D1c["mn"];

        Hbar1_["pq"] += Hbar2_["pvqu"] * L1_["uv"];
        Hbar1_["pq"] -= 0.5 * Hbar2_["vpqu"] * L1_["uv"];
    }

    // compute fully contracted term from T1
    Hbar1_.block("cc").iterate([&](const std::vector<size_t>& i, double& value) {
        if (i[0] == i[1])
            Hbar0_ += value;
    });
    Hbar0_ += 0.5 * Hbar1_["uv"] * L1_["vu"];

    if (eri_df_) {
        Hbar0_ += 0.5 * B["gux"] * B["gvy"] * L2_["xyuv"];
    } else {
        Hbar0_ += 0.5 * Hbar2_["uvxy"] * L2_["xyuv"];
    }

    Hbar0_ += Efrzc_ + Enuc_ - Eref_;

    rotation.stop();

    ////////////////////////////////////////////////////////////////////////////////////

    // iteration variables
    bool converged = false;

    timer comm("Hbar T2 commutator");

    // temporary Hamiltonian used in every iteration
    O1_["pq"] = Hbar1_["pq"];
    if (eri_df_) {
        Hbar2_["pqrs"] = B["gpr"] * B["gqs"];
    } else {
        O2_["pqrs"] = Hbar2_["pqrs"];
    }

    // iteration variables
    converged = false;

    // compute Hbar recursively
    for (int n = 1; n <= rsc_ncomm_; ++n) {
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

        if (n == 1 && eri_df_) {
            // zero-body
            H1_T2_C0(O1_, T2_, factor, C0);
            V_T2_C0_DF(B, T2_, DT2_, factor, C0);
            // one-body
            H1_T2_C1(O1_, T2_, factor, C1_);
            V_T2_C1_DF(B, T2_, DT2_, factor, C1_);
            // two-body
            H1_T2_C2(O1_, T2_, factor, C2_);
            V_T2_C2_DF(B, T2_, DT2_, factor, C2_);
        } else {
            // zero-body
            H1_T2_C0(O1_, T2_, factor, C0);
            H2_T2_C0(O2_, T2_, DT2_, factor, C0);
            // one-body
            H1_T2_C1(O1_, T2_, factor, C1_);
            H2_T2_C1(O2_, T2_, DT2_, factor, C1_);
            // two-body
            H1_T2_C2(O1_, T2_, factor, C2_);
            H2_T2_C2(O2_, T2_, DT2_, factor, C2_);
        }

        // printing level
        if (print_ > 2) {
            std::string dash(38, '-');
            outfile->Printf("\n    %s\n", dash.c_str());
        }

        // [H, A] = [H, T] + [H, T]^dagger
        C0 *= 2.0;
        O1_["pq"] = C1_["pq"];
        C1_["pq"] += O1_["qp"];
        O2_["pqrs"] = C2_["pqrs"];
        C2_["pqrs"] += O2_["rspq"];

        // Hbar += C
        Hbar0_ += C0;
        Hbar1_["pq"] += C1_["pq"];
        Hbar2_["pqrs"] += C2_["pqrs"];

        // copy C to O for next level commutator
        O1_["pq"] = C1_["pq"];
        O2_["pqrs"] = C2_["pqrs"];

        // test convergence of C
        double norm_C1 = C1_.norm();
        double norm_C2 = C2_.norm();
        if (print_ > 2) {
            outfile->Printf("\n  n: %3d, C0: %20.15f, C1 max: %20.15f, C2 max: %20.15f", n, C0,
                            C1_.norm(0), C2_.norm(0));
        }
        if (std::sqrt(norm_C2 * norm_C2 + norm_C1 * norm_C1) < rsc_conv_) {
            converged = true;
            break;
        }
    }
    if (!converged) {
        outfile->Printf("\n    Warning! Hbar is not converged in %3d-nested commutators!",
                        rsc_ncomm_);
        outfile->Printf("\n    Please increase DSRG_RSC_NCOMM.");
    }
}

void SA_MRDSRG::compute_hbar_qc() {
    // initialize Hbar with bare H
    Hbar0_ = 0.0;
    Hbar1_["ia"] = F_["ia"];
    if (eri_df_) {
        Hbar2_["ijab"] = B_["gia"] * B_["gjb"];
    } else {
        Hbar2_["ijab"] = V_["ijab"];
    }

    // compute S1 = H + 0.5 * [H, A]
    BlockedTensor S1 = BTF_->build(tensor_type_, "S1", {"gg"}, true);
    H1_T1_C1(F_, T1_, 0.5, S1);
    H1_T2_C1(F_, T2_, 0.5, S1);
    if (eri_df_) {
        V_T1_C1_DF(B_, T1_, 0.5, S1);
        V_T2_C1_DF(B_, T2_, DT2_, 0.5, S1);
    } else {
        H2_T1_C1(V_, T1_, 0.5, S1);
        H2_T2_C1(V_, T2_, DT2_, 0.5, S1);
    }

    auto temp = BTF_->build(tensor_type_, "temp", {"gg"}, true);
    temp["pq"] = S1["pq"];
    S1["pq"] += temp["qp"];

    S1["pq"] += F_["pq"];

    // compute Hbar = [S1, A]
    //   Step 1: [S1, T]_{ab}^{ij}
    H1_T1_C0(S1, T1_, 2.0, Hbar0_);
    H1_T2_C0(S1, T2_, 2.0, Hbar0_);
    H1_T1_C1(S1, T1_, 1.0, Hbar1_);
    H1_T2_C1(S1, T2_, 1.0, Hbar1_);
    H1_T2_C2(S1, T2_, 1.0, Hbar2_);

    //   Step 2: [S1, T]_{ij}^{ab}
    temp = BTF_->build(tensor_type_, "temp", {"ph"}, true);
    H1_T1_C1(S1, T1_, 1.0, temp);
    H1_T2_C1(S1, T2_, 1.0, temp);
    Hbar1_["ia"] += temp["ai"];

    temp = BTF_->build(tensor_type_, "temp", {"pphh"}, true);
    H1_T2_C2(S1, T2_, 1.0, temp);
    Hbar2_["ijab"] += temp["abij"];

    // compute S2 = H + 0.5 * [H, A]
    // 0.5 * [H, T]
    BlockedTensor S2 = BTF_->build(tensor_type_, "S2", {"gggg"}, true);
    H1_T2_C2(F_, T2_, 0.5, S2);
    if (eri_df_) {
        V_T1_C2_DF(B_, T1_, 0.5, S2);
        V_T2_C2_DF(B_, T2_, DT2_, 0.5, S2);
    } else {
        H2_T1_C2(V_, T1_, 0.5, S2);
        H2_T2_C2(V_, T2_, DT2_, 0.5, S2);
    }

    // 0.5 * [H, T]^+
    add_hermitian_conjugate(S2);

    // add bare Hamiltonian contribution
    if (eri_df_) {
        S2["pqrs"] += B_["gpr"] * B_["gqs"];
    } else {
        S2["pqrs"] += V_["pqrs"];
    }

    // compute Hbar = [S2, A]
    //   Step 1: [S2, T]_{ab}^{ij}
    H2_T1_C0(S2, T1_, 2.0, Hbar0_);
    H2_T2_C0(S2, T2_, DT2_, 2.0, Hbar0_);
    H2_T1_C1(S2, T1_, 1.0, Hbar1_);
    H2_T2_C1(S2, T2_, DT2_, 1.0, Hbar1_);
    H2_T1_C2(S2, T1_, 1.0, Hbar2_);
    H2_T2_C2(S2, T2_, DT2_, 1.0, Hbar2_);

    //   Step 2: [S2, T]_{ij}^{ab}
    temp.zero();
    H2_T1_C2(S2, T1_, 1.0, temp);
    H2_T2_C2(S2, T2_, DT2_, 1.0, temp);
    Hbar2_["ijab"] += temp["abij"];

    temp = BTF_->build(tensor_type_, "temp", {"ph"}, true);
    H2_T1_C1(S2, T1_, 1.0, temp);
    H2_T2_C1(S2, T2_, DT2_, 1.0, temp);
    Hbar1_["ia"] += temp["ai"];
}

void SA_MRDSRG::setup_ldsrg2_tensors() {
    BlockedTensor::set_expert_mode(true);

    if (corrlv_string_ == "LDSRG2_QC") {
        Hbar1_ = BTF_->build(tensor_type_, "Hbar1", {"hp"});
        Hbar2_ = BTF_->build(tensor_type_, "Hbar2", {"hhpp"});
    } else {
        if (nivo_) {
            // Generate blocks for Hbar2_, O2_ and C2_
            std::vector<std::string> blocks_exclude_V3 = nivo_labels();
            Hbar2_ = BTF_->build(tensor_type_, "Hbar2", blocks_exclude_V3);
            O2_ = BTF_->build(tensor_type_, "O2", blocks_exclude_V3);
            C2_ = BTF_->build(tensor_type_, "C2", blocks_exclude_V3);
        } else {
            Hbar2_ = BTF_->build(tensor_type_, "Hbar2", {"gggg"});
            O2_ = BTF_->build(tensor_type_, "O2", {"gggg"});
            C2_ = BTF_->build(tensor_type_, "C2", {"gggg"});
        }

        Hbar1_ = BTF_->build(tensor_type_, "Hbar1", {"gg"});
        O1_ = BTF_->build(tensor_type_, "O1", {"gg"});
        C1_ = BTF_->build(tensor_type_, "C1", {"gg"});
    }

    DT1_ = BTF_->build(tensor_type_, "DT1", {"hp"});
    DT2_ = BTF_->build(tensor_type_, "DT2", {"hhpp"});
}

void SA_MRDSRG::add_hermitian_conjugate(BlockedTensor& H2) {
    // labels for half of tensor
    std::vector<std::string> labels_half{"cc", "ca", "cv", "ac", "aa", "av", "vc", "va", "vv"};
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

void SA_MRDSRG::compute_mbar_ldsrg2(const ambit::BlockedTensor& M, int max_level, int ind) {
    // compute reference multipole
    {
        Mbar0_[ind] = M["uv"] * L1_["vu"];
        auto& M1c = M.block("cc").data();
        for (size_t m = 0, ncore = core_mos_.size(); m < ncore; ++m) {
            Mbar0_[ind] += 2.0 * M1c[m * ncore + m];
        }
        Mbar1_[ind]["uv"] = M["uv"];
    }

    O1_["pq"] = M["pq"];

    bool converged = false;
    for (int n = 1; n <= rsc_ncomm_; ++n) {
        // prefactor before n-nested commutator
        double factor = 1.0 / n;

        // Compute the commutator C = 1/n [O, T]
        double C0 = 0.0;
        C1_.zero();
        if (max_level > 1)
            C2_.zero();

        // zero-body
        H1_T1_C0(O1_, T1_, factor, C0);
        H1_T2_C0(O1_, T2_, factor, C0);
        if (max_level > 1 and n != 1) {
            H2_T1_C0(O2_, T1_, factor, C0);
            H2_T2_C0(O2_, T2_, DT2_, factor, C0);
        }

        // one-body
        H1_T1_C1(O1_, T1_, factor, C1_);
        H1_T2_C1(O1_, T2_, factor, C1_);
        if (max_level > 1 and n != 1) {
            H2_T1_C1(O2_, T1_, factor, C1_);
            H2_T2_C1(O2_, T2_, DT2_, factor, C1_);
        }

        // two-body
        if (max_level > 1) {
            H1_T2_C2(O1_, T2_, factor, C2_);
            if (n != 1) {
                H2_T1_C2(O2_, T1_, factor, C2_);
                H2_T2_C2(O2_, T2_, DT2_, factor, C2_);
            }
        }

        // [M, A] = [M, T] + [M, T]^dagger
        C0 *= 2.0;
        O1_["pq"] = C1_["pq"];
        C1_["pq"] += O1_["qp"];
        if (max_level > 1) {
            O2_["pqrs"] = C2_["pqrs"];
            C2_["pqrs"] += O2_["rspq"];
        }

        // Mbar += C
        Mbar0_[ind] += C0;
        Mbar1_[ind]["pq"] += C1_["pq"];
        if (max_level > 1) {
            Mbar2_[ind]["pqrs"] += C2_["pqrs"];
        }

        // copy C to O for next level commutator
        O1_["pq"] = C1_["pq"];
        if (max_level > 1) {
            O2_["pqrs"] = C2_["pqrs"];
        }

        // test convergence of C
        double norm_C1 = C1_.norm();
        double norm_C2 = (max_level > 1) ? C2_.norm() : 0.0;
        if (std::sqrt(norm_C2 * norm_C2 + norm_C1 * norm_C1) < rsc_conv_) {
            converged = true;
            break;
        }
    }

    if (!converged) {
        outfile->Printf("\n    Warning! Mbar is not converged in %3d-nested commutators!",
                        rsc_ncomm_);
        outfile->Printf("\n    Please increase DSRG_RSC_NCOMM.");
    }
}

} // namespace forte
