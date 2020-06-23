/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2020 by its authors (see COPYING, COPYING.LESSER,
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

#include "psi4/libpsi4util/process.h"
#include "psi4/libdiis/diismanager.h"
#include "psi4/libmints/molecule.h"
#include "psi4/libqt/qt.h"

#include "helpers/timer.h"
#include "base_classes/mo_space_info.h"
#include "mrdsrg.h"

using namespace psi;

namespace forte {

void MRDSRG::compute_hbar() {
    if (print_ > 2) {
        outfile->Printf("\n\n  ==> Computing the DSRG Transformed Hamiltonian <==\n");
    }

    // copy bare Hamiltonian to Hbar
    Hbar0_ = 0.0;
    Hbar1_["pq"] = F_["pq"];
    Hbar1_["PQ"] = F_["PQ"];
    if (eri_df_) {
        Hbar2_["pqrs"] = B_["gpr"] * B_["gqs"];
        Hbar2_["pqrs"] -= B_["gps"] * B_["gqr"];

        Hbar2_["pQrS"] = B_["gpr"] * B_["gQS"];

        Hbar2_["PQRS"] = B_["gPR"] * B_["gQS"];
        Hbar2_["PQRS"] -= B_["gPS"] * B_["gQR"];
    } else {
        Hbar2_["pqrs"] = V_["pqrs"];
        Hbar2_["pQrS"] = V_["pQrS"];
        Hbar2_["PQRS"] = V_["PQRS"];
        O2_["pqrs"] = Hbar2_["pqrs"];
        O2_["pQrS"] = Hbar2_["pQrS"];
        O2_["PQRS"] = Hbar2_["PQRS"];
    }

    // temporary Hamiltonian used in every iteration
    O1_["pq"] = F_["pq"];
    O1_["PQ"] = F_["PQ"];

    // iteration variables
    bool converged = false;
    int maxn = foptions_->get_int("DSRG_RSC_NCOMM");
    double ct_threshold = foptions_->get_double("DSRG_RSC_THRESHOLD");
    std::string dsrg_op = foptions_->get_str("DSRG_TRANS_TYPE");

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
        if (n == 1 && eri_df_) {
            H2_T1_C0_DF(B_, T1_, factor, C0);
            H2_T2_C0_DF(B_, T2_, factor, C0);
        } else {
            H2_T1_C0(O2_, T1_, factor, C0);
            H2_T2_C0(O2_, T2_, factor, C0);
        }
        // one-body
        H1_T1_C1(O1_, T1_, factor, C1_);
        H1_T2_C1(O1_, T2_, factor, C1_);
        if (n == 1 && eri_df_) {
            H2_T1_C1_DF(B_, T1_, factor, C1_);
        } else {
            H2_T1_C1(O2_, T1_, factor, C1_);
        }
        if (foptions_->get_str("SRG_COMM") == "STANDARD") {
            if (n == 1 && eri_df_) {
                H2_T2_C1_DF(B_, T2_, factor, C1_);
            } else {
                H2_T2_C1(O2_, T2_, factor, C1_);
            }
        } else if (foptions_->get_str("SRG_COMM") == "FO") {
            BlockedTensor C1p = BTF_->build(tensor_type_, "C1p", spin_cases({"gg"}));
            if (n == 1 && eri_df_) {
                H2_T2_C1_DF(B_, T2_, factor, C1p);
            } else {
                H2_T2_C1(O2_, T2_, factor, C1p);
            }
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
        if ((foptions_->get_str("SRG_COMM") == "STANDARD") or n < 2) {
            H1_T2_C2(O1_, T2_, factor, C2_);
        } else if (foptions_->get_str("SRG_COMM") == "FO2") {
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
        if (n == 1 && eri_df_) {
            H2_T1_C2_DF(B_, T1_, factor, C2_);
            H2_T2_C2_DF(B_, T2_, factor, C2_);
        } else {
            H2_T1_C2(O2_, T1_, factor, C2_);
            H2_T2_C2(O2_, T2_, factor, C2_);
        }

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
            outfile->Printf("\n  n: %3d, C0: %20.15f, C1 max: %20.15f, C2 max: %20.15f", n, C0,
                            C1_.norm(0), C2_.norm(0));
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
    int maxn = foptions_->get_int("DSRG_RSC_NCOMM");
    double ct_threshold = foptions_->get_double("DSRG_RSC_THRESHOLD");
    std::string dsrg_op = foptions_->get_str("DSRG_TRANS_TYPE");

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
            outfile->Printf("\n  n: %3d, C0: %20.15f, C1 max: %20.15f, C2 max: %20.15f", n, C0,
                            C1_.norm(0), C2_.norm(0));
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

void MRDSRG::compute_hbar_sequential_rotation() {
    if (print_ > 2) {
        outfile->Printf("\n\n  ==> Computing the DSRG Transformed Hamiltonian <==\n");
    }

    timer rotation("Hbar T1 rotation");

    ambit::BlockedTensor A1;
    A1 = BTF_->build(tensor_type_, "A1 Amplitudes", spin_cases({"gg"}));
    A1["ia"] = T1_["ia"];
    A1["ai"] -= T1_["ia"];
    A1["IA"] = T1_["IA"];
    A1["AI"] -= T1_["IA"];

    size_t ncmo = core_mos_.size() + actv_mos_.size() + virt_mos_.size();

    psi::SharedMatrix aA1_m(new psi::Matrix("A1 alpha", ncmo, ncmo));
    psi::SharedMatrix bA1_m(new psi::Matrix("A1 beta", ncmo, ncmo));
    A1.iterate([&](const std::vector<size_t>& i, const std::vector<SpinType>& spin, double& value) {
        if (spin[0] == AlphaSpin)
            aA1_m->set(i[0], i[1], value);
        else
            bA1_m->set(i[0], i[1], value);
    });

    // >=3 is required for high energy convergence
    aA1_m->expm(3);
    bA1_m->expm(3);

    ambit::BlockedTensor U1;
    U1 = BTF_->build(tensor_type_, "Transformer", spin_cases({"gg"}));
    U1.iterate([&](const std::vector<size_t>& i, const std::vector<SpinType>& spin, double& value) {
        if (spin[0] == AlphaSpin)
            value = aA1_m->get(i[0], i[1]);
        else
            value = bA1_m->get(i[0], i[1]);
    });

    /// Recompute Hbar0 (ref. energy + T1 correlation), Hbar1 (Fock), and Hbar2 (aptei)
    /// E = 0.5 * ( H["ji"] + F["ji] ) * D1["ij"] + 0.25 * V["xyuv"] * L2["uvxy"]

    Hbar1_["rs"] = U1["rp"] * H_["pq"] * U1["sq"];
    Hbar1_["RS"] = U1["RP"] * H_["PQ"] * U1["SQ"];

    Hbar0_ = 0.0;
    for (const std::string block : {"cc", "CC"}) {
        Hbar1_.block(block).iterate([&](const std::vector<size_t>& i, double& value) {
            if (i[0] == i[1]) {
                Hbar0_ += 0.5 * value;
            }
        });
    }
    Hbar0_ += 0.5 * Hbar1_["uv"] * Gamma1_["vu"];
    Hbar0_ += 0.5 * Hbar1_["UV"] * Gamma1_["VU"];

    ambit::BlockedTensor B;
    if (eri_df_) {
        B = BTF_->build(tensor_type_, "B 3-idx", {"Lgg", "LGG"});
        B["grs"] = U1["rp"] * B_["gpq"] * U1["sq"];
        B["gRS"] = U1["RP"] * B_["gPQ"] * U1["SQ"];

        // for simplicity, create a core-core density matrix
        BlockedTensor D1c = BTF_->build(tensor_type_, "Gamma1 core", spin_cases({"cc"}));
        for (size_t m = 0, nc = core_mos_.size(); m < nc; ++m) {
            D1c.block("cc").data()[m * nc + m] = 1.0;
            D1c.block("CC").data()[m * nc + m] = 1.0;
        }

        BlockedTensor temp = BTF_->build(tensor_type_, "B temp", {"L"});
        temp["g"] = B["gmn"] * D1c["mn"];
        temp["g"] += B["guv"] * Gamma1_["uv"];
        Hbar1_["pq"] += temp["g"] * B["gpq"];
        Hbar1_["PQ"] += temp["g"] * B["gPQ"];

        temp["g"] = B["gMN"] * D1c["MN"];
        temp["g"] += B["gUV"] * Gamma1_["UV"];
        Hbar1_["pq"] += temp["g"] * B["gpq"];
        Hbar1_["PQ"] += temp["g"] * B["gPQ"];

        Hbar1_["pq"] -= B["gpn"] * B["gmq"] * D1c["mn"];
        Hbar1_["pq"] -= B["gpv"] * B["guq"] * Gamma1_["uv"];

        Hbar1_["PQ"] -= B["gPN"] * B["gMQ"] * D1c["MN"];
        Hbar1_["PQ"] -= B["gPV"] * B["gUQ"] * Gamma1_["UV"];
    } else {
        Hbar2_["pqrs"] = U1["pt"] * U1["qo"] * V_["to45"] * U1["r4"] * U1["s5"];
        Hbar2_["pQrS"] = U1["pt"] * U1["QO"] * V_["tO49"] * U1["r4"] * U1["S9"];
        Hbar2_["PQRS"] = U1["PT"] * U1["QO"] * V_["TO89"] * U1["R8"] * U1["S9"];

        // for simplicity, create a core-core density matrix
        BlockedTensor D1c = BTF_->build(tensor_type_, "Gamma1 core", spin_cases({"cc"}));
        for (size_t m = 0, nc = core_mos_.size(); m < nc; ++m) {
            D1c.block("cc").data()[m * nc + m] = 1.0;
            D1c.block("CC").data()[m * nc + m] = 1.0;
        }

        Hbar1_["pq"] += Hbar2_["pnqm"] * D1c["mn"];
        Hbar1_["pq"] += Hbar2_["pNqM"] * D1c["MN"];
        Hbar1_["pq"] += Hbar2_["pvqu"] * Gamma1_["uv"];
        Hbar1_["pq"] += Hbar2_["pVqU"] * Gamma1_["UV"];

        Hbar1_["PQ"] += Hbar2_["nPmQ"] * D1c["mn"];
        Hbar1_["PQ"] += Hbar2_["PNQM"] * D1c["MN"];
        Hbar1_["PQ"] += Hbar2_["vPuQ"] * Gamma1_["uv"];
        Hbar1_["PQ"] += Hbar2_["PVQU"] * Gamma1_["UV"];
    }

    // compute fully contracted term from T1
    for (const std::string block : {"cc", "CC"}) {
        Hbar1_.block(block).iterate([&](const std::vector<size_t>& i, double& value) {
            if (i[0] == i[1]) {
                Hbar0_ += 0.5 * value;
            }
        });
    }
    Hbar0_ += 0.5 * Hbar1_["uv"] * Gamma1_["vu"];
    Hbar0_ += 0.5 * Hbar1_["UV"] * Gamma1_["VU"];

    if (eri_df_) {
        Hbar0_ += 0.25 * B["gux"] * B["gvy"] * Lambda2_["xyuv"];
        Hbar0_ -= 0.25 * B["guy"] * B["gvx"] * Lambda2_["xyuv"];
        Hbar0_ += 0.25 * B["gUX"] * B["gVY"] * Lambda2_["XYUV"];
        Hbar0_ -= 0.25 * B["gUY"] * B["gVX"] * Lambda2_["XYUV"];
        Hbar0_ += B["gux"] * B["gVY"] * Lambda2_["xYuV"];
    } else {
        Hbar0_ += 0.25 * Hbar2_["uvxy"] * Lambda2_["xyuv"];
        Hbar0_ += 0.25 * Hbar2_["UVXY"] * Lambda2_["XYUV"];
        Hbar0_ += Hbar2_["uVxY"] * Lambda2_["xYuV"];
    }

    Hbar0_ += Efrzc_ + Enuc_ - Eref_;

    rotation.stop();

    ////////////////////////////////////////////////////////////////////////////////////

    // iteration variables
    bool converged = false;
    int maxn = foptions_->get_int("DSRG_RSC_NCOMM");
    double ct_threshold = foptions_->get_double("DSRG_RSC_THRESHOLD");

    timer comm("Hbar T2 commutator");

    // temporary Hamiltonian used in every iteration
    O1_["pq"] = Hbar1_["pq"];
    O1_["PQ"] = Hbar1_["PQ"];
    if (eri_df_) {
        Hbar2_["pqrs"] = B["gpr"] * B["gqs"];
        Hbar2_["pqrs"] -= B["gps"] * B["gqr"];

        Hbar2_["pQrS"] = B["gpr"] * B["gQS"];

        Hbar2_["PQRS"] = B["gPR"] * B["gQS"];
        Hbar2_["PQRS"] -= B["gPS"] * B["gQR"];
    } else {
        O2_["pqrs"] = Hbar2_["pqrs"];
        O2_["pQrS"] = Hbar2_["pQrS"];
        O2_["PQRS"] = Hbar2_["PQRS"];
    }

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

        if (n == 1 && eri_df_) {
            // zero-body
            H1_T2_C0(O1_, T2_, factor, C0);
            H2_T2_C0_DF(B, T2_, factor, C0);
            // one-body
            H1_T2_C1(O1_, T2_, factor, C1_);
            H2_T2_C1_DF(B, T2_, factor, C1_);
            // two-body
            H1_T2_C2(O1_, T2_, factor, C2_);
            H2_T2_C2_DF(B, T2_, factor, C2_);
        } else {
            // zero-body
            H1_T2_C0(O1_, T2_, factor, C0);
            H2_T2_C0(O2_, T2_, factor, C0);
            // one-body
            H1_T2_C1(O1_, T2_, factor, C1_);
            H2_T2_C1(O2_, T2_, factor, C1_);
            // two-body
            H1_T2_C2(O1_, T2_, factor, C2_);
            H2_T2_C2(O2_, T2_, factor, C2_);
        }

        // printing level
        if (print_ > 2) {
            std::string dash(38, '-');
            outfile->Printf("\n    %s\n", dash.c_str());
        }

        // [H, A] = [H, T] + [H, T]^dagger
        if (dsrg_trans_type_ == "UNITARY") {
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
            outfile->Printf("\n  n: %3d, C0: %20.15f, C1 max: %20.15f, C2 max: %20.15f", n, C0,
                            C1_.norm(0), C2_.norm(0));
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

    timer ldsrg2("Energy_ldsrg2");

    if (foptions_->get_str("THREEPDC") == "ZERO") {
        outfile->Printf("\n    Skip Lambda3 contributions in [Hbar2, T2].");
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

    // figure out off-diagonal block labels for Hbar1
    std::vector<std::string> blocks1 = od_one_labels_hp();

    // figure out off-diagonal block labels for Hbar2
    std::vector<std::string> blocks2 = od_two_labels_hhpp();

    BlockedTensor::set_expert_mode(true);
    if (nivo_) {
        // Generate blocks for Hbar2_, O2_ and C2_
        std::vector<std::string> blocks_exclude_V3;
        for (std::string s0 : {"c", "a", "v"}) {
            for (std::string s1 : {"c", "a", "v"}) {
                for (std::string s2 : {"c", "a", "v"}) {
                    for (std::string s3 : {"c", "a", "v"}) {
                        std::string s = s0 + s1 + s2 + s3;
                        if (std::count(s.begin(), s.end(), 'v') < 3) {
                            blocks_exclude_V3.push_back(s);
                        }
                    }
                }
            }
        }
        Hbar2_ = BTF_->build(tensor_type_, "Hbar2", spin_cases(blocks_exclude_V3));
        O2_ = BTF_->build(tensor_type_, "O2", spin_cases(blocks_exclude_V3));
        C2_ = BTF_->build(tensor_type_, "C2", spin_cases(blocks_exclude_V3));
    } else {
        Hbar2_ = BTF_->build(tensor_type_, "Hbar2", spin_cases({"gggg"}));
        O2_ = BTF_->build(tensor_type_, "O2", spin_cases({"gggg"}));
        C2_ = BTF_->build(tensor_type_, "C2", spin_cases({"gggg"}));
    }

    // iteration variables
    double Ecorr = 0.0;
    int maxiter = foptions_->get_int("MAXITER");
    double e_conv = foptions_->get_double("E_CONVERGENCE");
    double r_conv = foptions_->get_double("R_CONVERGENCE");
    bool converged = false;
    Hbar1_ = BTF_->build(tensor_type_, "Hbar1", spin_cases({"gg"}));
    O1_ = BTF_->build(tensor_type_, "O1", spin_cases({"gg"}));
    C1_ = BTF_->build(tensor_type_, "C1", spin_cases({"gg"}));
    DT1_ = BTF_->build(tensor_type_, "DT1", spin_cases({"hp"}));
    DT2_ = BTF_->build(tensor_type_, "DT2", spin_cases({"hhpp"}));

    // setup DIIS
    if (diis_start_ > 0) {
        diis_manager_init();
    }

    // start iteration
    for (int cycle = 1; cycle <= maxiter; ++cycle) {
        // compute Hbar
        local_timer t_hbar;
        timer hbar("Compute Hbar");
        if (sequential_Hbar_) {
            compute_hbar_sequential_rotation();
        } else {
            compute_hbar();
        }
        hbar.stop();
        double Edelta = Hbar0_ - Ecorr;
        Ecorr = Hbar0_;
        double time_hbar = t_hbar.get();

        timer od("Off-diagonal Hbar");
        // compute norms of off-diagonal Hbar
        double Hbar1od = Hbar1od_norm(blocks1);
        double Hbar2od = Hbar2od_norm(blocks2);

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
        if (std::fabs(Edelta) < e_conv && rms < r_conv) {
            converged = true;
            break;
        }

        if (cycle == maxiter) {
            outfile->Printf(
                "\n\n    The computation does not converge in %d iterations! Quitting.\n", maxiter);
        }
        if (cycle > 5 and std::fabs(rms) > 10.0) {
            outfile->Printf("\n\n    Large RMS for amplitudes. Likely no convergence. Quitting.\n");
        }
    }

    // clean up raw pointers used in DIIS
    if (diis_start_ > 0) {
        diis_manager_cleanup();
    }

    timer final("Summary");
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
    if (!converged) {
        throw psi::PSIEXCEPTION("The MR-LDSRG(2) computation does not converge.");
    }
    final.stop();

    Hbar0_ = Ecorr;
    return Ecorr;
}

void MRDSRG::compute_hbar_qc() {
    std::string dsrg_op = foptions_->get_str("DSRG_TRANS_TYPE");

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

double MRDSRG::compute_energy_ldsrg2_qc() {
    // print title
    outfile->Printf("\n\n  ==> Computing MR-LDSRG(2)-QC Energy <==\n");
    outfile->Printf("\n    DSRG transformed Hamiltonian is truncated to "
                    "quadratic nested commutator.");
    outfile->Printf("\n    Reference:");
    outfile->Printf("\n      J. Chem. Phys. (in preparation)\n");

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

    // figure out off-diagonal blocks labels
    std::vector<std::string> blocks1 = od_one_labels_hp();
    std::vector<std::string> blocks2 = od_two_labels_hhpp();

    // iteration variables
    double Ecorr = 0.0;
    int maxiter = foptions_->get_int("MAXITER");
    double e_conv = foptions_->get_double("E_CONVERGENCE");
    double r_conv = foptions_->get_double("R_CONVERGENCE");
    bool converged = false;
    Hbar1_ = BTF_->build(tensor_type_, "Hbar1", spin_cases({"hp"}));
    Hbar2_ = BTF_->build(tensor_type_, "Hbar2", spin_cases({"hhpp"}));
    O1_ = BTF_->build(tensor_type_, "O1", spin_cases({"aa"}));
    DT1_ = BTF_->build(tensor_type_, "DT1", spin_cases({"hp"}));
    DT2_ = BTF_->build(tensor_type_, "DT2", spin_cases({"hhpp"}));
    BlockedTensor::set_expert_mode(true);

    // initialize V_ here
    if (eri_df_) {
        V_ = BTF_->build(tensor_type_, "V", spin_cases({"gggg"}));

        V_["pqrs"] = B_["gpr"] * B_["gqs"];
        V_["pqrs"] -= B_["gps"] * B_["gqr"];

        V_["pQrS"] = B_["gpr"] * B_["gQS"];

        V_["PQRS"] = B_["gPR"] * B_["gQS"];
        V_["PQRS"] -= B_["gPS"] * B_["gQR"];
    }

    // setup DIIS
    if (diis_start_ > 0) {
        diis_manager_init();
    }

    // start iteration
    for (int cycle = 1; cycle <= maxiter; ++cycle) {
        // compute Hbar
        local_timer t_hbar;
        compute_hbar_qc();
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
                "\n\n    The computation does not converge in %d iterations! Quitting.\n", maxiter);
        }
        if (cycle > 5 and std::fabs(rms) > 10.0) {
            outfile->Printf("\n\n    Large RMS for amplitudes. Likely no convergence. Quitting.\n");
        }
    }

    // clean up raw pointers used in DIIS
    if (diis_start_ > 0) {
        diis_manager_cleanup();
    }

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
    if (!converged) {
        throw psi::PSIEXCEPTION("The MR-LDSRG(2)-QC computation does not converge.");
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

} // namespace forte
