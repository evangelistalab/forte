/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2017 by its authors (see COPYING, COPYING.LESSER, AUTHORS).
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

#include <cmath>

#include "psi4/libpsi4util/process.h"
#include "psi4/libdiis/diismanager.h"

#include "tensorsrg.h"

using namespace psi;
using namespace ambit;

namespace psi {
namespace forte {

double TensorSRG::compute_ct_energy() {
    bool do_dsrg = false;
    if (options_.get_double("DSRG_S") > 0.0) {
        do_dsrg = true;
        compute_mp2_guess_driven_srg();
    } else {
        compute_mp2_guess();
    }

    // Start the CTSD cycle
    double dsrg_s = options_.get_double("DSRG_S");
    double old_energy = 0.0;
    bool converged = false;
    int cycle = 0;

    std::shared_ptr<DIISManager> diis_manager;

    int max_diis_vectors = options_.get_int("DIIS_MAX_VECS");
    if (max_diis_vectors > 0) {
        diis_manager = std::shared_ptr<DIISManager>(new DIISManager(
            max_diis_vectors, "L-CTSD DIIS vector", DIISManager::OldestAdded, DIISManager::InCore));
        diis_manager->set_error_vector_size(5, DIISEntry::Pointer, S1.block("ov").numel(),
                                            DIISEntry::Pointer, S1.block("OV").numel(),
                                            DIISEntry::Pointer, S2.block("oovv").numel(),
                                            DIISEntry::Pointer, S2.block("oOvV").numel(),
                                            DIISEntry::Pointer, S2.block("OOVV").numel());
        diis_manager->set_vector_size(5, DIISEntry::Pointer, S1.block("ov").numel(),
                                      DIISEntry::Pointer, S1.block("OV").numel(),
                                      DIISEntry::Pointer, S2.block("oovv").numel(),
                                      DIISEntry::Pointer, S2.block("oOvV").numel(),
                                      DIISEntry::Pointer, S2.block("OOVV").numel());
    }

    if (dsrg_s == 0.0) {
        outfile->Printf("\n  Linearized Canonical Transformation Theory with "
                        "Singles and Doubles");
    } else {
        outfile->Printf("\n  Driven Similarity Renormalization Group with "
                        "Singles and Doubles (s = %f a.u.)",
                        dsrg_s);
    }
    outfile->Printf("\n  "
                    "----------------------------------------------------------"
                    "----------------------------------------");
    outfile->Printf("\n         Cycle     Energy (a.u.)     Delta(E)   |Hbar1| "
                    "   |Hbar2|     |S1|    |S2|  max(S1) max(S2)");
    outfile->Printf("\n  "
                    "----------------------------------------------------------"
                    "----------------------------------------");

    compute_hbar();

    while (!converged) {
        if (print_ > 1) {
            outfile->Printf("\n  Updating the S amplitudes...");
        }

        if (do_dsrg) {
            update_S1_dsrg();
            update_S2_dsrg();
        } else {
            update_S1();
            update_S2();
        }

        if (print_ > 1) {
            outfile->Printf("\n  --------------------------------------------");
            outfile->Printf("\n  nExc           |S|                  |R|");
            outfile->Printf("\n  --------------------------------------------");
            outfile->Printf("\n    1     %15e      %15e", S1.norm(), 0.0);
            outfile->Printf("\n    2     %15e      %15e", S2.norm(), 0.0);
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
        if (diis_manager) {
            if (do_dsrg) {
                diis_manager->add_entry(
                    10, &(DS1.block("ov").data()[0]), &(DS1.block("OV").data()[0]),
                    &(DS2.block("oovv").data()[0]), &(DS2.block("oOvV").data()[0]),
                    &(DS2.block("OOVV").data()[0]), &(S1.block("ov").data()[0]),
                    &(S1.block("OV").data()[0]), &(S2.block("oovv").data()[0]),
                    &(S2.block("oOvV").data()[0]), &(S2.block("OOVV").data()[0]));
            } else {
                diis_manager->add_entry(
                    10, &(Hbar1.block("ov").data()[0]), &(Hbar1.block("OV").data()[0]),
                    &(Hbar2.block("oovv").data()[0]), &(Hbar2.block("oOvV").data()[0]),
                    &(Hbar2.block("OOVV").data()[0]), &(S1.block("ov").data()[0]),
                    &(S1.block("OV").data()[0]), &(S2.block("oovv").data()[0]),
                    &(S2.block("oOvV").data()[0]), &(S2.block("OOVV").data()[0]));
            }
            if (cycle > max_diis_vectors) {
                if (cycle % max_diis_vectors == 2) {
                    outfile->Printf(" -> DIIS");
                    diis_manager->extrapolate(
                        5, &(S1.block("ov").data()[0]), &(S1.block("OV").data()[0]),
                        &(S2.block("oovv").data()[0]), &(S2.block("oOvV").data()[0]),
                        &(S2.block("OOVV").data()[0]));
                }
            }
        }
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

        double max_S1 = 0.0;
        double max_S2 = 0.0;
        S1.citerate(
            [&](const std::vector<size_t>&, const std::vector<SpinType>&, const double& value) {
                if (std::fabs(value) > std::fabs(max_S1))
                    max_S1 = value;
            });

        S2.citerate(
            [&](const std::vector<size_t>&, const std::vector<SpinType>&, const double& value) {
                if (std::fabs(value) > std::fabs(max_S2))
                    max_S2 = value;
            });

        double norm_H1a = Hbar1.block("ov").norm();
        double norm_H1b = Hbar1.block("OV").norm();
        double norm_H2aa = Hbar2.block("oovv").norm();
        double norm_H2ab = Hbar2.block("oOvV").norm();
        double norm_H2bb = Hbar2.block("OOVV").norm();

        double norm_Hbar1_ex = std::sqrt(norm_H1a * norm_H1a + norm_H1b * norm_H1b);
        double norm_Hbar2_ex = std::sqrt(0.25 * norm_H2aa * norm_H2aa + norm_H2ab * norm_H2ab +
                                         0.25 * norm_H2bb * norm_H2bb);

        double norm_S1a = S1.block("ov").norm();
        double norm_S1b = S1.block("OV").norm();
        double norm_S2aa = S2.block("oovv").norm();
        double norm_S2ab = S2.block("oOvV").norm();
        double norm_S2bb = S2.block("OOVV").norm();

        double norm_S1 = std::sqrt(norm_S1a * norm_S1a + norm_S1b * norm_S1b);
        double norm_S2 = std::sqrt(0.25 * norm_S2aa * norm_S2aa + norm_S2ab * norm_S2ab +
                                   0.25 * norm_S2bb * norm_S2bb);

        outfile->Printf("\n    @CT %4d %20.12f %11.3e %10.3e %10.3e %7.4f "
                        "%7.4f %7.4f %7.4f",
                        cycle, energy, delta_energy, norm_Hbar1_ex, norm_Hbar2_ex, norm_S1, norm_S2,
                        max_S1, max_S2);

        if (std::fabs(delta_energy) < options_.get_double("E_CONVERGENCE")) {
            converged = true;
        }

        if (cycle > options_.get_int("MAXITER")) {
            outfile->Printf("\n\n\tThe calculation did not converge in %d "
                            "cycles\n\tQuitting.\n",
                            options_.get_int("MAXITER"));

            converged = true;
            old_energy = 0.0;
        }

        cycle++;
    }
    outfile->Printf("\n  "
                    "----------------------------------------------------------"
                    "----------------------------------------");

    if (dsrg_s == 0.0) {
        outfile->Printf("\n\n\n    L-CTSD correlation energy      = %25.15f", old_energy - E0_);
        outfile->Printf("\n  * L-CTSD total energy            = %25.15f\n", old_energy);
    } else {
        outfile->Printf("\n\n\n    DSRG-SD correlation energy      = %25.15f", old_energy - E0_);
        outfile->Printf("\n  * DSRG-SD total energy            = %25.15f\n", old_energy);
    }
    // Set some environment variables
    Process::environment.globals["CURRENT ENERGY"] = old_energy;
    Process::environment.globals["CTSD ENERGY"] = old_energy;
    Process::environment.globals["LCTSD ENERGY"] = old_energy;

    if (options_.get_bool("SAVE_HBAR")) {
        save_hbar();
    }

    return old_energy;
}

double TensorSRG::compute_hbar() {
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
    Hbar1["PQ"] = F["PQ"];
    Hbar2["pqrs"] = V["pqrs"];
    Hbar2["pQrS"] = V["pQrS"];
    Hbar2["PQRS"] = V["PQRS"];

    O1["pq"] = F["pq"];
    O1["PQ"] = F["PQ"];
    O2["pqrs"] = V["pqrs"];
    O2["pQrS"] = V["pQrS"];
    O2["PQRS"] = V["PQRS"];

    if (print_ > 1) {
        outfile->Printf("\n  %2d %20.12f %20e %20e", 0, Hbar0, Hbar1.norm(), Hbar2.norm());
    }

    int maxn = options_.get_int("DSRG_RSC_NCOMM");
    double ct_threshold = options_.get_double("SRG_RSC_THRESHOLD");
    for (int n = 1; n <= maxn; ++n) {
        double factor = 1.0 / static_cast<double>(n);

        double C0 = 0;
        C1.zero();
        C2.zero();

        // Compute the commutator C = 1/n [O,S]
        commutator_A_B_C(factor, O1, O2, S1, S2, C0, C1, C2, n);

        // Hbar += C
        Hbar0 += C0;
        Hbar1["pq"] += C1["pq"];
        Hbar1["PQ"] += C1["PQ"];
        Hbar2["pqrs"] += C2["pqrs"];
        Hbar2["pQrS"] += C2["pQrS"];
        Hbar2["PQRS"] += C2["PQRS"];

        // O = C
        O1["pq"] = C1["pq"];
        O1["PQ"] = C1["PQ"];
        O2["pqrs"] = C2["pqrs"];
        O2["pQrS"] = C2["pQrS"];
        O2["PQRS"] = C2["PQRS"];

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
    print_ = 0;
    return Hbar0;
}

void TensorSRG::update_S1() {
    S1["ia"] += Hbar1["ia"] * InvD1["ia"];
    S1["IA"] += Hbar1["IA"] * InvD1["IA"];
}

void TensorSRG::update_S2() {
    S2["ijab"] += Hbar2["ijab"] * InvD2["ijab"];
    S2["iJaB"] += Hbar2["iJaB"] * InvD2["iJaB"];
    S2["IJAB"] += Hbar2["IJAB"] * InvD2["IJAB"];
}

void TensorSRG::update_S1_dsrg() {
    R1.zero();

    R1["ia"] = Hbar1["ia"] * RInvD1["ia"];
    R1["ia"] += S1["ia"] * D1["ia"] * RInvD1["ia"];

    R1["IA"] = Hbar1["IA"] * RInvD1["IA"];
    R1["IA"] += S1["IA"] * D1["IA"] * RInvD1["IA"];

    // Compute the change in amplitudes
    DS1["ia"] = S1["ia"];
    DS1["ia"] -= R1["ia"];
    DS1["IA"] = S1["IA"];
    DS1["IA"] -= R1["IA"];

    S1["ia"] = R1["ia"];
    S1["IA"] = R1["IA"];

    //    outfile->Printf("\n ||R1|| = %f, ||S1|| = %f,",R1.norm(),S1.norm());
}

void TensorSRG::update_S2_dsrg() {
    R2.zero();

    R2["ijab"] = Hbar2["ijab"] * RInvD2["ijab"];
    R2["ijab"] += S2["ijab"] * D2["ijab"] * RInvD2["ijab"];

    R2["iJaB"] = Hbar2["iJaB"] * RInvD2["iJaB"];
    R2["iJaB"] += S2["iJaB"] * D2["iJaB"] * RInvD2["iJaB"];

    R2["IJAB"] = Hbar2["IJAB"] * RInvD2["IJAB"];
    R2["IJAB"] += S2["IJAB"] * D2["IJAB"] * RInvD2["IJAB"];

    // Compute the change in amplitudes
    DS2["ijab"] = S2["ijab"];
    DS2["iJaB"] = S2["iJaB"];
    DS2["IJAB"] = S2["IJAB"];
    DS2["ijab"] -= R2["ijab"];
    DS2["iJaB"] -= R2["iJaB"];
    DS2["IJAB"] -= R2["IJAB"];

    S2["ijab"] = R2["ijab"];
    S2["iJaB"] = R2["iJaB"];
    S2["IJAB"] = R2["IJAB"];

    // outfile->Printf("\n ||Hbar2|| = %f, ||R2|| = %f, ||S2|| =
    // %f,",Hbar2.norm(),R2.norm(),S2.norm());
}
}
} // EndNamespaces
