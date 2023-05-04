/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2023 by its authors (see COPYING, COPYING.LESSER, AUTHORS).
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

#include <iomanip>
#include <sstream>
#include <complex>
#include <stdlib.h>

#include "psi4/libpsi4util/process.h"
#include "psi4/libmints/molecule.h"
#include "psi4/libmints/pointgrp.h"
#include "psi4/libpsio/psio.hpp"
#include "base_classes/active_space_solver.h"

#include "helpers/printing.h"
#include "helpers/helpers.h"
#include "tdci.h"

using namespace psi;

/* CHEEV prototype */
extern "C" {
extern void zheev(char* jobz, char* uplo, int* n, std::complex<double>* a, int* lda, double* w,
                  std::complex<double>* work, int* lwork, double* rwork, int* info);
}

namespace forte {

TDCI::TDCI(std::shared_ptr<ActiveSpaceMethod> active_space_method,
           std::shared_ptr<SCFInfo> scf_info, std::shared_ptr<ForteOptions> options,
           std::shared_ptr<MOSpaceInfo> mo_space_info,
           std::shared_ptr<ActiveSpaceIntegrals> as_ints)
    : active_space_method_(active_space_method), scf_info_(scf_info), as_ints_(as_ints),
      options_(options), mo_space_info_(mo_space_info) {}

TDCI::~TDCI() {}

double TDCI::compute_energy() {

    double en = 0.0;
    int nact = mo_space_info_->size("ACTIVE");
    int hole = options_->get_int("TDCI_HOLE");

    // skip some steps if doing screening
    bool build_full_H = true;

    std::string propagate_type = options_->get_str("TDCI_PROPAGATOR");

    if ((propagate_type == "EXACT_SELECT") or (propagate_type == "RK4_SELECT") or
        (propagate_type == "RK4_LIST") or (propagate_type == "RK4_SELECT_LIST")) {

        build_full_H = false;
    }

    // 1. Grab the CI wavefunction
    DeterminantHashVec aci_dets = active_space_method_->get_PQ_space();
    SharedMatrix aci_coeffs = active_space_method_->get_PQ_evecs();

    // 1.5 Compute ACI occs and save to file
    std::vector<double> ref_occs(nact, 0.0);
    for (size_t I = 0, maxI = aci_dets.size(); I < maxI; ++I) {
        const Determinant& det = aci_dets.get_det(I);
        const double CI = aci_coeffs->get(I, 0);
        for (int p = 0; p < nact; ++p) {
            double value = 0.0;
            if (det.get_alfa_bit(p) == true) {
                value += CI * CI;
            }
            if (det.get_beta_bit(p) == true) {
                value += CI * CI;
            }
            ref_occs[p] += value;
        }
    }
    save_vector(ref_occs, "aci-occ.txt");

    // 2. Generate the n-1 Determinants (not just core)
    for (int i = 0; i < nact; ++i) {
        annihilate_wfn(aci_dets, ann_dets_, i);
    }
    size_t nann = ann_dets_.size();
    outfile->Printf("\n  Number of cationic determinants: %zu", ann_dets_.size());

    // 3. Build the full n-1 Hamiltonian if not screening
    std::vector<std::string> det_str(nann);
    SharedMatrix full_aH = std::make_shared<Matrix>("aH", nann, nann);
    if (build_full_H) {
        for (size_t I = 0; I < nann; ++I) {
            Determinant detI = ann_dets_.get_det(I);
            det_str[I] = str(detI, nact).c_str();
            for (size_t J = I; J < nann; ++J) {
                Determinant detJ = ann_dets_.get_det(J);
                double value = as_ints_->slater_rules(detI, detJ);
                full_aH->set(I, J, value);
                full_aH->set(J, I, value);
            }
        }
        save_vector(det_str, "determinants.txt");
    }

    // 4. Prepare initial state by removing an electron from aci wfn
    // DeterminantHashVec core_dets;
    SharedVector core_coeffs = std::make_shared<Vector>("init", nann);
    core_coeffs->zero();

    const det_hashvec& dets = aci_dets.wfn_hash();
    size_t ndet = dets.size();
    // size_t ncore = 0;
    for (size_t I = 0; I < ndet; ++I) {
        auto& detI = dets[I];
        if (detI.get_alfa_bit(hole) == true) {
            Determinant adet(detI);
            adet.set_alfa_bit(hole, false);
            size_t idx = ann_dets_.get_idx(adet);
            core_coeffs->set(idx,
                             core_coeffs->get(idx) + aci_coeffs->get(aci_dets.get_idx(detI), 0));
            core_dets_.add(adet);
        }
    }
    outfile->Printf("\n  Size of initial state: %zu", core_dets_.size());

    // 5. Renormalize wave function
    outfile->Printf("\n  Renormalizing wave function");
    double norm = core_coeffs->norm();
    norm = 1.0 / norm;
    core_coeffs->scale(norm);

    outfile->Printf("\n  Using %s propagator", propagate_type.c_str());
    // 5. Propagate
    if (propagate_type == "EXACT") {
        propagate_exact(core_coeffs, full_aH);
    } else if (propagate_type == "CN") {
        propagate_cn(core_coeffs, full_aH);
    } else if (propagate_type == "QCN") {
        propagate_QCN(core_coeffs, full_aH);
    } else if (propagate_type == "LINEAR") {
        propagate_taylor1(core_coeffs, full_aH);
    } else if (propagate_type == "QUADRATIC") {
        propagate_taylor2(core_coeffs, full_aH);
    } else if (propagate_type == "RK4") {
        propagate_RK4(core_coeffs, full_aH);
    } else if (propagate_type == "RK4_LIST") {
        propagate_list(core_coeffs);
    } else if (propagate_type == "LANCZOS") {
        propagate_lanczos(core_coeffs, full_aH);
    } else if (propagate_type == "EXACT_SELECT" or propagate_type == "RK4_SELECT" or
               propagate_type == "RK4_SELECT_LIST") {
        compute_tdci_select(core_coeffs);
    } else if (propagate_type == "ALL") {
        propagate_exact(core_coeffs, full_aH);
        propagate_cn(core_coeffs, full_aH);
        propagate_taylor1(core_coeffs, full_aH);
        propagate_taylor2(core_coeffs, full_aH);
        propagate_RK4(core_coeffs, full_aH);
        propagate_QCN(core_coeffs, full_aH);
        propagate_lanczos(core_coeffs, full_aH);
    }

    if (options_->get_bool("TDCI_TEST_OCC")) {
        double tval = test_occ();
    }

    return en;
}

void TDCI::propagate_list(SharedVector C0) {

    Timer t1;

    // A list of orbitals to compute occupations_ during propagation
    std::vector<int> orbs = options_->get_int_list("TDCI_OCC_ORB");

    // Timestep details
    int nstep = options_->get_int("TDCI_NSTEP");
    double dt = options_->get_double("TDCI_TIMESTEP");
    double conv = 1.0 / 24.18884326505;
    dt *= conv;
    double time = dt;

    occupations_.resize(orbs.size());

    size_t n_ann_dets = ann_dets_.size();
    // set initial coefficients
    std::vector<double> PQ_coeffs_r(n_ann_dets, 0.0);
    std::vector<double> PQ_coeffs_i(n_ann_dets, 0.0);

    for (size_t I = 0; I < n_ann_dets; ++I) {
        PQ_coeffs_r[I] = C0->get(I);
    }

    // Compute couplings for sigma builds
    auto mo_sym = mo_space_info_->symmetry("ACTIVE");
    DeterminantSubstitutionLists op(as_ints_);
    op.set_quiet_mode(true);

    op.build_strings(ann_dets_);
    op.op_s_lists(ann_dets_);
    op.tp_s_lists(ann_dets_);

    // Begin the timesteps
    for (int N = 0; N < nstep; ++N) {

        //    if (options_->get_str("TDCI_PROPAGATOR") == "RK4_LIST") {
        propagate_RK4_list(PQ_coeffs_r, PQ_coeffs_i, ann_dets_, op, dt);
        //    } else if (options_->get_str("TDCI_PROPAGATOR") == "LANCZOS_LIST") {
        //        propagate_lanczos_list(PQ_coeffs_r, PQ_coeffs_i, PQ_space, dt);
        //    }

        if (std::fabs((time / conv) - round(time / conv)) <= 1e-8) {
            outfile->Printf("\n t = %1.3f as", time / conv);
            if (options_->get_bool("TDCI_PRINT_WFN")) {
                std::stringstream ss;
                ss << std::fixed << std::setprecision(3) << time / conv;
                save_vector(PQ_coeffs_r, "rk4_list_" + ss.str() + "_r.txt");
                save_vector(PQ_coeffs_i, "rk4_list_" + ss.str() + "_i.txt");
            }
            //  std::vector<double> occ = compute_occupation(ct_r, ct_i, orbs);
            std::vector<double> occ = compute_occupation(ann_dets_, PQ_coeffs_r, PQ_coeffs_i, orbs);
            for (size_t i = 0; i < orbs.size(); ++i) {
                occupations_[i].push_back(occ[i]);
            }
        }

        time += dt;
    }
    for (size_t i = 0; i < orbs.size(); ++i) {
        save_vector(occupations_[i], "occupations_" + std::to_string(orbs[i]) + ".txt");
    }

    outfile->Printf("\n Time spent propagating: %1.6f s", t1.get());
}

void TDCI::propagate_exact(SharedVector C0, SharedMatrix H) {

    std::vector<int> orbs = options_->get_int_list("TDCI_OCC_ORB");

    Timer t1;
    size_t ndet = C0->dim();

    // Diagonalize the full Hamiltonian
    SharedMatrix evecs = std::make_shared<Matrix>("evecs", ndet, ndet);
    SharedVector evals = std::make_shared<Vector>("evals", ndet);

    outfile->Printf("\n  Diagonalizing Hamiltonian");
    H->diagonalize(evecs, evals);

    int nstep = options_->get_int("TDCI_NSTEP");
    double dt = options_->get_double("TDCI_TIMESTEP");

    SharedVector ct_r = std::make_shared<Vector>("ct_R", ndet);
    SharedVector ct_i = std::make_shared<Vector>("ct_I", ndet);
    ct_r->zero();
    ct_i->zero();

    // Convert to a.u. from as
    double conv = 1.0 / 24.18884326505;
    double time = dt;

    occupations_.resize(orbs.size());
    SharedVector int1 = std::make_shared<Vector>("int1", ndet);
    SharedVector int1r = std::make_shared<Vector>("int2r", ndet);
    SharedVector int1i = std::make_shared<Vector>("int2i", ndet);
    // First multiply the evecs by the initial vector
    int1->gemv(true, 1.0, *evecs, *C0, 0.0);

    for (int n = 0; n < nstep; ++n) {
        for (size_t I = 0; I < ndet; ++I) {
            int1r->set(I, int1->get(I) * std::cos(-1.0 * evals->get(I) * time * conv));
            int1i->set(I, int1->get(I) * std::sin(-1.0 * evals->get(I) * time * conv));
        }
        ct_r->gemv(false, 1.0, *evecs, *int1r, 0.0);
        ct_i->gemv(false, 1.0, *evecs, *int1i, 0.0);

        if (std::abs(time - round(time)) <= 1e-8) {
            if (options_->get_bool("TDCI_PRINT_WFN")) {
                std::stringstream ss;
                ss << std::fixed << std::setprecision(3) << time;
                //  save_vector(mag,"exact_" + ss.str()+ ".txt");
                save_vector(ct_r, "exact_" + ss.str() + "_r.txt");
                save_vector(ct_i, "exact_" + ss.str() + "_i.txt");
            }
            std::vector<double> occ = compute_occupation(ct_r, ct_i, orbs);
            for (size_t i = 0; i < orbs.size(); ++i) {
                occupations_[i].push_back(occ[i]);
            }
        }
        int1r->zero();
        int1i->zero();

        time += dt;
    }
    for (size_t i = 0; i < orbs.size(); ++i) {
        save_vector(occupations_[i], "occupations_" + std::to_string(orbs[i]) + ".txt");
    }
    outfile->Printf("\n Time spent propagating (exact): %1.6f s", t1.get());
}

void TDCI::propagate_cn(SharedVector C0, SharedMatrix H) {

    outfile->Printf("\n  Propogating with Crank-Nicholson algorithm");

    Timer total;
    size_t ndet = C0->dim();

    int nstep = options_->get_int("TDCI_NSTEP");
    double dt = options_->get_double("TDCI_TIMESTEP");
    double conv = 1.0 / 24.18884326505;
    dt *= conv;
    double time = dt;

    // Copy initial state into iteratively updated vectors
    SharedVector ct_r = std::make_shared<Vector>("ct_R", ndet);
    SharedVector ct_i = std::make_shared<Vector>("ct_I", ndet);

    ct_r->copy(C0->clone());
    ct_i->zero();

    std::vector<int> orbs = options_->get_int_list("TDCI_OCC_ORB");

    occupations_.resize(orbs.size());
    for (int n = 1; n <= nstep; ++n) {
        //        outfile->Printf("\n  Propogating at t = %1.6f", time/conv);
        // Form b vector
        Timer b;
        SharedVector b_r = std::make_shared<Vector>("br", ndet);
        SharedVector b_i = std::make_shared<Vector>("bi", ndet);

        // -iHdt|Psi>
        b_r->gemv(false, 0.5 * dt, *H, *ct_i, 0.0);
        b_i->gemv(false, -1.0 * 0.5 * dt, *H, *ct_r, 0.0);

        b_r->add(*ct_r);
        b_i->add(*ct_i);

        // Converge C(t+dt)
        bool converged = false;
        SharedVector ct_r_new = std::make_shared<Vector>("ct_R", ndet);
        SharedVector ct_i_new = std::make_shared<Vector>("ct_I", ndet);

        while (!converged) {

            ct_r_new->gemv(false, 0.5 * dt, *H, *ct_i, 0.0);
            ct_i_new->gemv(false, -1.0 * 0.5 * dt, *H, *ct_r, 0.0);

            ct_r_new->add(*b_r);
            ct_i_new->add(*b_i);

            // Test convergence
            SharedVector err = std::make_shared<Vector>("err", ndet);
            double norm = 0.0;
            for (size_t I = 0; I < ndet; ++I) {
                double rn = ct_r_new->get(I);
                double in = ct_i_new->get(I);
                norm += (rn * rn + in * in);
            }
            ct_r_new->scale(1.0 / sqrt(norm));
            ct_i_new->scale(1.0 / sqrt(norm));
            for (size_t I = 0; I < ndet; ++I) {
                double rn = ct_r_new->get(I);
                double in = ct_i_new->get(I);
                double ro = ct_r->get(I);
                double io = ct_i->get(I);
                err->set(I, (rn * rn + in * in) - (ro * ro + io * io));
            }

            //            outfile->Printf("\n  %1.9f", err->norm());
            if (err->norm() <= options_->get_double("TDCI_CN_CONVERGENCE")) {
                converged = true;
            }

            ct_r->copy(ct_r_new->clone());
            ct_i->copy(ct_i_new->clone());
        }

        double norm = 0.0;
        SharedVector mag = std::make_shared<Vector>("mag", ndet);
        for (size_t I = 0; I < ndet; ++I) {
            double re = ct_r->get(I);
            double im = ct_i->get(I);
            mag->set(I, (re * re) + (im * im));
            norm += (re * re) + (im * im);
        }
        norm = std::sqrt(norm);
        //        outfile->Printf("\n norm: %1.6f", norm);
        ct_r->scale(1.0 / norm);
        ct_i->scale(1.0 / norm);

        //        outfile->Printf("\n  norm(t=%1.3f) = %1.5f", time/conv, norm);

        if (std::abs(time / conv - round(time / conv)) <= 1e-8) {
            outfile->Printf("\n t = %1.3f as", time / conv);
            std::stringstream ss;
            ss << std::fixed << std::setprecision(3) << time / conv;
            if (options_->get_bool("TDCI_PRINT_WFN")) {
                save_vector(ct_r, "CN_" + ss.str() + "_r.txt");
                save_vector(ct_i, "CN_" + ss.str() + "_i.txt");
            }
            std::vector<double> occ = compute_occupation(ct_r, ct_i, orbs);
            for (size_t i = 0; i < orbs.size(); ++i) {
                occupations_[i].push_back(occ[i]);
            }
        }

        time += dt;
    }
    for (size_t i = 0; i < orbs.size(); ++i) {
        save_vector(occupations_[i], "occupations_" + std::to_string(orbs[i]) + ".txt");
    }
    outfile->Printf("\n  Time spent propagating (CN): %1.6f", total.get());
}

void TDCI::propagate_taylor1(SharedVector C0, SharedMatrix H) {
    outfile->Printf("\n  Propogating with linear Taylor algorithm");

    Timer t1;
    std::vector<int> orbs = options_->get_int_list("TDCI_OCC_ORB");
    // The screening criterion
    // double eta = options_->get_double("TDCI_ETA_P");
    double d_tau = options_->get_double("TDCI_TIMESTEP") * 0.0413413745758;
    double tau = 0.0;
    int nstep = options_->get_int("TDCI_NSTEP");

    size_t ndet = C0->dim();
    // The imaginary part
    SharedVector C0_r = std::make_shared<Vector>("C0r", ndet);
    SharedVector C0_i = std::make_shared<Vector>("C0i", ndet);
    SharedVector Ct_r = std::make_shared<Vector>("Ctr", ndet);
    SharedVector Ct_i = std::make_shared<Vector>("Cti", ndet);

    C0_r->copy(C0->clone());
    C0_i->zero();

    occupations_.resize(orbs.size());

    for (int N = 0; N < nstep; ++N) {
        std::vector<size_t> counter(ndet, 0);
        tau += d_tau;

        SharedVector sigma_r = std::make_shared<Vector>("Sr", ndet);
        SharedVector sigma_i = std::make_shared<Vector>("Si", ndet);

        Ct_r->zero();
        Ct_i->zero();
        sigma_r->zero();
        sigma_i->zero();

        // Compute first order correction
        sigma_r->gemv(false, 1.0, *H, *C0_r, 0.0);
        sigma_i->gemv(false, 1.0, *H, *C0_i, 0.0);

        sigma_r->scale(d_tau);
        sigma_i->scale(d_tau);

        Ct_r->add(*C0_r);
        Ct_i->add(*C0_i);

        Ct_r->add(*sigma_i);
        Ct_i->subtract(*sigma_r);

        // Renormalize C_tau
        double norm = 0.0;
        for (size_t I = 0; I < ndet; ++I) {
            double re = Ct_r->get(I);
            double im = Ct_i->get(I);
            norm += re * re + im * im;
        }
        norm = 1.0 / std::sqrt(norm);
        Ct_r->scale(norm);
        Ct_i->scale(norm);

        // print the wavefunction
        if (std::abs((tau / 0.0413413745758) - round(tau / 0.0413413745758)) <= 1e-8) {
            outfile->Printf("\n t = %1.3f as", tau / 0.0413413745758);
            if (options_->get_bool("TDCI_PRINT_WFN")) {
                std::stringstream ss;
                ss << std::fixed << std::setprecision(3) << tau / 0.0413413745758;
                save_vector(Ct_r, "taylor_" + ss.str() + "_r.txt");
                save_vector(Ct_i, "taylor_" + ss.str() + "_i.txt");
            }
            std::vector<double> occ = compute_occupation(Ct_r, Ct_i, orbs);
            for (size_t i = 0; i < orbs.size(); ++i) {
                occupations_[i].push_back(occ[i]);
            }
        }
        C0_r->copy(Ct_r->clone());
        C0_i->copy(Ct_i->clone());
    }
    for (size_t i = 0; i < orbs.size(); ++i) {
        save_vector(occupations_[i], "occupations_" + std::to_string(orbs[i]) + ".txt");
    }
    outfile->Printf("\n  Time spent propagating (linear): %1.6f", t1.get());
}

void TDCI::propagate_taylor2(SharedVector C0, SharedMatrix H) {

    outfile->Printf("\n  Propogating with quadratic Taylor algorithm");
    Timer t2;
    std::vector<int> orbs = options_->get_int_list("TDCI_OCC_ORB");

    int nstep = options_->get_int("TDCI_NSTEP");
    double dt = options_->get_double("TDCI_TIMESTEP");
    double conv = 1.0 / 24.18884326505;
    dt *= conv;
    double time = dt;

    size_t ndet = C0->dim();
    occupations_.resize(orbs.size());
    // The imaginary part
    SharedVector C0_r = std::make_shared<Vector>("C0r", ndet);
    SharedVector C0_i = std::make_shared<Vector>("C0i", ndet);
    SharedVector Ct_r = std::make_shared<Vector>("Ctr", ndet);
    SharedVector Ct_i = std::make_shared<Vector>("Cti", ndet);

    C0_r->copy(C0->clone());
    C0_i->zero();

    for (int N = 0; N < nstep; ++N) {
        std::vector<size_t> counter(ndet, 0);
        //  outfile->Printf("\n  Propogating at t = %1.6f", time/conv);
        SharedVector sigma_r = std::make_shared<Vector>("Sr", ndet);
        SharedVector sigma_i = std::make_shared<Vector>("Si", ndet);

        Ct_r->zero();
        Ct_i->zero();
        sigma_r->zero();
        sigma_i->zero();

        // Compute first order correction
        sigma_r->gemv(false, 1.0, *H, *C0_r, 0.0);
        sigma_i->gemv(false, 1.0, *H, *C0_i, 0.0);

        sigma_r->scale(dt);
        sigma_i->scale(dt);

        Ct_r->add(*C0_r);
        Ct_i->add(*C0_i);

        Ct_r->add(*sigma_i);
        Ct_i->subtract(*sigma_r);
        // Quadratic correction
        SharedVector sigmaq_r = std::make_shared<Vector>("Sr", ndet);
        SharedVector sigmaq_i = std::make_shared<Vector>("Si", ndet);
        sigma_r->scale(1.0 / dt);
        sigma_i->scale(1.0 / dt);
        sigmaq_r->gemv(false, 1.0, *H, *sigma_r, 0.0);
        sigmaq_i->gemv(false, 1.0, *H, *sigma_i, 0.0);

        sigmaq_r->scale(dt * dt * 0.5);
        sigmaq_i->scale(dt * dt * 0.5);

        Ct_r->subtract(*sigmaq_r);
        Ct_i->subtract(*sigmaq_i);

        // Renormalize C_tau
        double norm = 0.0;
        for (size_t I = 0; I < ndet; ++I) {
            double re = Ct_r->get(I);
            double im = Ct_i->get(I);
            norm += re * re + im * im;
        }
        norm = 1.0 / std::sqrt(norm);
        Ct_r->scale(norm);
        Ct_i->scale(norm);

        // print the wavefunction
        if (std::fabs((time / conv) - round(time / conv)) <= 1e-8) {
            outfile->Printf("\n t = %1.3f as", time / conv);
            if (options_->get_bool("TDCI_PRINT_WFN")) {
                std::stringstream ss;
                ss << std::fixed << std::setprecision(3) << time / conv;
                save_vector(Ct_r, "t2_" + ss.str() + "_r.txt");
                save_vector(Ct_i, "t2_" + ss.str() + "_i.txt");
            }
            std::vector<double> occ = compute_occupation(Ct_r, Ct_i, orbs);
            for (size_t i = 0; i < orbs.size(); ++i) {
                occupations_[i].push_back(occ[i]);
            }
        }
        C0_r->copy(Ct_r->clone());
        C0_i->copy(Ct_i->clone());
        time += dt;
    }
    for (size_t i = 0; i < orbs.size(); ++i) {
        save_vector(occupations_[i], "occupations_" + std::to_string(orbs[i]) + ".txt");
    }
    outfile->Printf("\n  Time spent propagating (quadratic): %1.6f", t2.get());
}

void TDCI::propagate_RK4(SharedVector C0, SharedMatrix H) {

    outfile->Printf("\n  Propogating with 4th order Runge-Kutta algorithm");

    Timer total;
    size_t ndet = C0->dim();

    int nstep = options_->get_int("TDCI_NSTEP");
    double dt = options_->get_double("TDCI_TIMESTEP");
    double conv = 1.0 / 24.18884326505;
    dt *= conv;
    double time = dt;

    std::vector<int> orbs = options_->get_int_list("TDCI_OCC_ORB");
    occupations_.resize(orbs.size());
    // Copy initial state into iteratively updated vectors
    SharedVector ct_r = std::make_shared<Vector>("ct_R", ndet);
    SharedVector ct_i = std::make_shared<Vector>("ct_I", ndet);

    ct_r->copy(C0->clone());
    ct_i->zero();

    for (int n = 1; n <= nstep; ++n) {

        // k1
        SharedVector k1r = std::make_shared<Vector>("k1r", ndet);
        SharedVector k1i = std::make_shared<Vector>("k1i", ndet);

        k1r->gemv(false, 1.0, *H, *ct_i, 0.0);
        k1i->gemv(false, -1.0, *H, *ct_r, 0.0);

        k1r->scale(dt);
        k1i->scale(dt);

        // k2
        SharedVector intr = std::make_shared<Vector>("intr", ndet);
        SharedVector inti = std::make_shared<Vector>("inti", ndet);

        intr->copy(ct_r->clone());
        inti->copy(ct_i->clone());

        k1r->scale(0.5);
        k1i->scale(0.5);
        intr->add(*k1r);
        inti->add(*k1i);
        k1r->scale(2.0);
        k1i->scale(2.0);

        SharedVector k2r = std::make_shared<Vector>("k2r", ndet);
        SharedVector k2i = std::make_shared<Vector>("k2i", ndet);

        k2r->gemv(false, 1.0, *H, *inti, 0.0);
        k2i->gemv(false, -1.0, *H, *intr, 0.0);
        k2r->scale(dt);
        k2i->scale(dt);

        // k3
        intr->copy(ct_r->clone());
        inti->copy(ct_i->clone());

        k2r->scale(0.5);
        k2i->scale(0.5);
        intr->add(*k2r);
        inti->add(*k2i);
        k2r->scale(2.0);
        k2i->scale(2.0);

        SharedVector k3r = std::make_shared<Vector>("k3r", ndet);
        SharedVector k3i = std::make_shared<Vector>("k3i", ndet);

        k3r->gemv(false, 1.0, *H, *inti, 0.0);
        k3i->gemv(false, -1.0, *H, *intr, 0.0);
        k3r->scale(dt);
        k3i->scale(dt);

        // k4
        intr->copy(ct_r->clone());
        inti->copy(ct_i->clone());

        intr->add(*k3r);
        inti->add(*k3i);

        SharedVector k4r = std::make_shared<Vector>("k4r", ndet);
        SharedVector k4i = std::make_shared<Vector>("k4i", ndet);

        k4r->gemv(false, 1.0, *H, *inti, 0.0);
        k4i->gemv(false, -1.0, *H, *intr, 0.0);
        k4r->scale(dt);
        k4i->scale(dt);

        // Compile all intermediates

        k1r->scale(1.0 / 6.0);
        k2r->scale(1.0 / 3.0);
        k3r->scale(1.0 / 3.0);
        k4r->scale(1.0 / 6.0);
        k1r->add(*k2r);
        k1r->add(*k3r);
        k1r->add(*k4r);
        ct_r->add(*k1r);

        k1i->scale(1.0 / 6.0);
        k2i->scale(1.0 / 3.0);
        k3i->scale(1.0 / 3.0);
        k4i->scale(1.0 / 6.0);
        k1i->add(*k2i);
        k1i->add(*k3i);
        k1i->add(*k4i);
        ct_i->add(*k1i);

        double norm = 0.0;
        for (size_t I = 0; I < ndet; ++I) {
            double re = ct_r->get(I);
            double im = ct_i->get(I);
            norm += (re * re) + (im * im);
        }
        norm = std::sqrt(norm);
        ct_r->scale(1.0 / norm);
        ct_i->scale(1.0 / norm);

        if (std::fabs((time / conv) - round(time / conv)) <= 1e-8) {
            outfile->Printf("\n t = %1.3f as", time / conv);
            if (options_->get_bool("TDCI_PRINT_WFN")) {
                std::stringstream ss;
                ss << std::fixed << std::setprecision(3) << time / conv;
                save_vector(ct_r, "RK4_" + ss.str() + "_r.txt");
                save_vector(ct_i, "RK4_" + ss.str() + "_i.txt");
            }
            std::vector<double> occ = compute_occupation(ct_r, ct_i, orbs);
            for (size_t i = 0; i < orbs.size(); ++i) {
                occupations_[i].push_back(occ[i]);
            }
        }

        time += dt;
    }
    for (size_t i = 0; i < orbs.size(); ++i) {
        save_vector(occupations_[i], "occupations_" + std::to_string(orbs[i]) + ".txt");
    }
    outfile->Printf("\n  Time spent propagating (RK4): %1.6f", total.get());
}

void TDCI::propagate_QCN(SharedVector C0, SharedMatrix H) {

    Timer total;
    size_t ndet = C0->dim();

    size_t nstep = options_->get_int("TDCI_NSTEP");
    double dt = options_->get_double("TDCI_TIMESTEP");
    double conv = 1.0 / 24.18884326505;
    dt *= conv;
    double time = dt;

    std::vector<int> orbs = options_->get_int_list("TDCI_OCC_ORB");
    occupations_.resize(orbs.size());

    // Copy initial state into iteratively updated vectors
    SharedVector ct_r = std::make_shared<Vector>("ct_R", ndet);
    SharedVector ct_i = std::make_shared<Vector>("ct_I", ndet);

    ct_r->copy(C0->clone());
    ct_i->zero();

    for (size_t n = 1; n <= nstep; ++n) {

        SharedVector b_r = std::make_shared<Vector>("br", ndet);
        SharedVector b_i = std::make_shared<Vector>("bi", ndet);

        // Quadratic propagator for b

        b_r->copy(ct_r->clone());
        b_i->copy(ct_i->clone());

        SharedVector sigma_r = std::make_shared<Vector>("sr", ndet);
        SharedVector sigma_i = std::make_shared<Vector>("si", ndet);
        sigma_r->gemv(false, 0.5 * dt, *H, *ct_r, 0.0);
        sigma_i->gemv(false, 0.5 * dt, *H, *ct_i, 0.0);

        b_r->add(*sigma_i);
        b_i->subtract(*sigma_r);
        // Quadratic correction
        b_r->gemv(false, -0.5 * dt, *H, *sigma_r, 1.0);
        b_i->gemv(false, -0.5 * dt, *H, *sigma_i, 1.0);

        bool converged = false;
        SharedVector ct_r_new = std::make_shared<Vector>("ct_R", ndet);
        SharedVector ct_i_new = std::make_shared<Vector>("ct_I", ndet);

        while (!converged) {
            ct_r_new->copy(b_r->clone());
            ct_i_new->copy(b_i->clone());

            SharedVector tmp_r = std::make_shared<Vector>("t_r", ndet);
            SharedVector tmp_i = std::make_shared<Vector>("t_i", ndet);

            tmp_r->gemv(false, 0.5 * dt, *H, *ct_r, 0.0);
            tmp_i->gemv(false, 0.5 * dt, *H, *ct_i, 0.0);

            ct_r_new->add(*tmp_i);
            ct_i_new->subtract(*tmp_r);

            ct_r_new->gemv(false, 0.5 * dt, *H, *tmp_r, 1.0);
            ct_i_new->gemv(false, 0.5 * dt, *H, *tmp_i, 1.0);

            // Test convergence
            SharedVector err = std::make_shared<Vector>("err", ndet);
            double norm = 0.0;
            for (size_t I = 0; I < ndet; ++I) {
                double rn = ct_r_new->get(I);
                double in = ct_i_new->get(I);
                norm += (rn * rn + in * in);
            }
            ct_r_new->scale(1.0 / sqrt(norm));
            ct_i_new->scale(1.0 / sqrt(norm));
            for (size_t I = 0; I < ndet; ++I) {
                double rn = ct_r_new->get(I);
                double in = ct_i_new->get(I);
                double ro = ct_r->get(I);
                double io = ct_i->get(I);
                err->set(I, (rn * rn + in * in) - (ro * ro + io * io));
            }

            //           outfile->Printf("\n  %1.9f", err->norm());
            if (err->norm() <= options_->get_double("TDCI_CN_CONVERGENCE")) {
                converged = true;
            }

            ct_r->copy(ct_r_new->clone());
            ct_i->copy(ct_i_new->clone());
        }
        if (std::fabs((time / conv) - round(time / conv)) <= 1e-8) {
            outfile->Printf("\n t = %1.3f as", time / conv);
            if (options_->get_bool("TDCI_PRINT_WFN")) {
                std::stringstream ss;
                ss << std::fixed << std::setprecision(3) << time / conv;
                save_vector(ct_r, "QCN_" + ss.str() + "_r.txt");
                save_vector(ct_i, "QCN_" + ss.str() + "_i.txt");
            }
            std::vector<double> occ = compute_occupation(ct_r, ct_i, orbs);
            for (size_t i = 0; i < orbs.size(); ++i) {
                occupations_[i].push_back(occ[i]);
            }
        }
        time += dt;
    }
    for (size_t i = 0; i < orbs.size(); ++i) {
        save_vector(occupations_[i], "occupations_" + std::to_string(orbs[i]) + ".txt");
    }
}

void TDCI::propagate_lanczos(SharedVector C0, SharedMatrix H) {

    std::vector<int> orbs = options_->get_int_list("TDCI_OCC_ORB");

    outfile->Printf("\n  Propogating with Arnoldi-Lanzcos algorithm");
    Timer total;
    size_t ndet = C0->dim();

    int nstep = options_->get_int("TDCI_NSTEP");
    double dt = options_->get_double("TDCI_TIMESTEP");
    double conv = 1.0 / 24.18884326505;
    dt *= conv;
    double time = dt;

    // Copy initial state into iteratively updated vectors
    SharedVector ct_r = std::make_shared<Vector>("ct_R", ndet);
    SharedVector ct_i = std::make_shared<Vector>("ct_I", ndet);

    ct_r->copy(C0->clone());
    ct_i->zero();

    int krylov_dim = options_->get_int("TDCI_KRYLOV_DIM");

    occupations_.resize(orbs.size());
    SharedMatrix Kn_r = std::make_shared<Matrix>("knr", ndet, krylov_dim);
    SharedMatrix Kn_i = std::make_shared<Matrix>("kni", ndet, krylov_dim);
    for (int N = 0; N < nstep; ++N) {

        // 1. Form the Krylov subspace vectors and subspace hamiltonian simultaneously
        // std::vector<std::pair<SharedVector,SharedVector>> Kn(krylov_dim);
        Kn_r->zero();
        Kn_i->zero();

        // Form the first vector
        std::vector<std::complex<double>> Hs(krylov_dim * krylov_dim);
        Kn_r->set_column(0, 0, ct_r);
        Kn_i->set_column(0, 0, ct_i);
        for (int k = 0; k < krylov_dim; ++k) {

            // Need to get last diagonal
            SharedVector wk_r = std::make_shared<Vector>("r", ndet);
            SharedVector wk_i = std::make_shared<Vector>("i", ndet);
            wk_r->zero();
            wk_i->zero();

            SharedVector qk_r = std::make_shared<Vector>("r", ndet);
            SharedVector qk_i = std::make_shared<Vector>("i", ndet);
            qk_r->zero();
            qk_i->zero();

            qk_r->add(*Kn_r->get_column(0, k));
            qk_i->add(*Kn_i->get_column(0, k));

            wk_r->gemv(false, 1.0, *H, *qk_r, 0.0);
            wk_i->gemv(false, 1.0, *H, *qk_i, 0.0);
            // Modified Gram-Schmidt
            for (int i = 0; i <= k; ++i) {
                SharedVector qi_r = std::make_shared<Vector>("r", ndet);
                SharedVector qi_i = std::make_shared<Vector>("i", ndet);
                qi_r->zero();
                qi_i->zero();
                qi_r->add(*Kn_r->get_column(0, i));
                qi_i->add(*Kn_i->get_column(0, i));

                double hik_r = qi_r->vector_dot(*wk_r) + qi_i->vector_dot(*wk_i);
                double hik_i = qi_i->vector_dot(*wk_r) - qi_r->vector_dot(*wk_i);

                Hs[krylov_dim * i + k] = {0.0, 0.0};
                Hs[krylov_dim * k + i] = {hik_r, hik_i};

                wk_r->axpy(-1.0 * hik_r, *qi_r);
                wk_r->axpy(1.0 * hik_i, *qi_i);
                wk_i->axpy(-1.0 * hik_r, *qi_i);
                wk_i->axpy(-1.0 * hik_i, *qi_r);
            }

            double norm = 0.0;
            for (size_t I = 0; I < ndet; ++I) {
                double re = wk_r->get(I);
                double im = wk_i->get(I);
                norm += re * re + im * im;
            }
            norm = sqrt(norm);
            wk_r->scale(1.0 / norm);
            wk_i->scale(1.0 / norm);

            if (k < (krylov_dim - 1)) {
                Hs[krylov_dim * (k + 1) + k] = {norm, 0.0};
                Kn_r->set_column(0, k + 1, wk_r);
                Kn_i->set_column(0, k + 1, wk_i);
            }
        }

        // Diagonalize matrix in Krylov subspace
        int n = krylov_dim, lda = krylov_dim, info, lwork;
        /* Local arrays */
        /* rwork dimension should be at least max(1,3*n-2) */
        double w[n], rwork[3 * n - 2];
        lwork = 2 * n - 1;
        std::vector<std::complex<double>> work(lwork);
        zheev("V", "L", &n, Hs.data(), &lda, w, work.data(), &lwork, rwork, &info);
        // Evecs are stored in Hs, let's unpack it and the energy

        SharedMatrix evecs_r = std::make_shared<Matrix>("er", n, n);
        SharedMatrix evecs_i = std::make_shared<Matrix>("ei", n, n);
        SharedVector evals = std::make_shared<Vector>("evals", n);
        for (int i = 0; i < krylov_dim; ++i) {
            evals->set(i, w[i]);
            for (int j = 0; j < krylov_dim; ++j) {
                evecs_r->set(i, j, Hs[krylov_dim * i + j].real());
                evecs_i->set(i, j, Hs[krylov_dim * i + j].imag());
            }
        }

        // Do the propagation
        SharedVector ct_int_r = std::make_shared<Vector>("ct_R", krylov_dim);
        SharedVector ct_int_i = std::make_shared<Vector>("ct_I", krylov_dim);

        SharedVector kd_r = std::make_shared<Vector>("ct_R", krylov_dim);
        SharedVector kd_i = std::make_shared<Vector>("ct_I", krylov_dim);
        for (int i = 0; i < krylov_dim; ++i) {
            kd_r->set(i, Kn_r->get_column(0, i)->vector_dot(*ct_r));
            kd_r->add(i, Kn_i->get_column(0, i)->vector_dot(*ct_i));

            kd_i->set(i, Kn_r->get_column(0, i)->vector_dot(*ct_i));
            kd_i->add(i, -1.0 * Kn_i->get_column(0, i)->vector_dot(*ct_r));
        }

        ct_int_r->gemv(true, 1.0, *evecs_r, *kd_r, 0.0);
        ct_int_r->gemv(true, 1.0, *evecs_i, *kd_i, 1.0);

        ct_int_i->gemv(true, 1.0, *evecs_r, *kd_i, 0.0);
        ct_int_i->gemv(true, -1.0, *evecs_i, *kd_r, 1.0);

        for (int I = 0; I < krylov_dim; ++I) {
            double rval = ct_int_r->get(I);
            double ival = ct_int_i->get(I);
            double eval = evals->get(I);
            ct_int_r->set(I, rval * std::cos(eval * dt) + ival * std::sin(eval * dt));
            ct_int_i->set(I, ival * std::cos(eval * dt) - rval * std::sin(eval * dt));
        }

        kd_r->gemv(false, 1.0, *evecs_r, *ct_int_r, 0.0);
        kd_r->gemv(false, -1.0, *evecs_i, *ct_int_i, 1.0);

        kd_i->gemv(false, 1.0, *evecs_r, *ct_int_i, 0.0);
        kd_i->gemv(false, 1.0, *evecs_i, *ct_int_r, 1.0);

        ct_r->zero();
        ct_i->zero();

        for (size_t i = 0; i < ndet; ++i) {
            ct_r->set(i, Kn_r->get_row(0, i)->vector_dot(*kd_r));
            ct_r->add(i, -1.0 * Kn_i->get_row(0, i)->vector_dot(*kd_i));

            ct_i->set(i, Kn_r->get_row(0, i)->vector_dot(*kd_i));
            ct_i->add(i, Kn_i->get_row(0, i)->vector_dot(*kd_r));
        }

        double norm = 0.0;
        for (size_t I = 0; I < ndet; ++I) {
            double re = ct_r->get(I);
            double im = ct_i->get(I);
            norm += (re * re) + (im * im);
        }
        //  outfile->Printf("\n  norm: %1.6f", norm);
        ct_r->scale(1.0 / sqrt(norm));
        ct_i->scale(1.0 / sqrt(norm));

        if (std::abs((time / conv) - round((time / conv))) <= 1e-8) {
            outfile->Printf("\n  t = %1.3f as", time / conv);
            if (options_->get_bool("TDCI_PRINT_WFN")) {
                std::stringstream ss;
                ss << std::fixed << std::setprecision(3) << time / conv;
                save_vector(ct_r, "lanczos_" + ss.str() + "_r.txt");
                save_vector(ct_i, "lanczos_" + ss.str() + "_i.txt");
            }
            std::vector<double> occ = compute_occupation(ct_r, ct_i, orbs);
            for (size_t i = 0; i < orbs.size(); ++i) {
                occupations_[i].push_back(occ[i]);
            }
        }
        time += dt;
    }
    for (size_t i = 0; i < orbs.size(); ++i) {
        save_vector(occupations_[i], "occupations_" + std::to_string(orbs[i]) + ".txt");
    }

    outfile->Printf("\n  Time spent propagating (Lanzcos): %1.6f", total.get());
}

void TDCI::save_matrix(SharedMatrix mat, std::string name) {

    size_t dim = mat->nrow();
    std::ofstream file;
    file.open(name, std::ofstream::out | std::ofstream::trunc);
    for (size_t I = 0; I < dim; ++I) {
        for (size_t J = 0; J < dim; ++J) {
            file << std::setw(12) << std::setprecision(11) << mat->get(I, J) << " ";
        }
        file << "\n";
    }
}
void TDCI::save_vector(SharedVector vec, std::string name) {

    size_t dim = vec->dim();
    std::ofstream file;
    file.open(name, std::ofstream::out | std::ofstream::trunc);
    for (size_t I = 0; I < dim; ++I) {
        file << std::setw(12) << std::setprecision(11) << vec->get(I) << "\n";
    }
}
void TDCI::save_vector(std::vector<std::string>& vec, std::string name) {

    size_t dim = vec.size();
    std::ofstream file;
    file.open(name, std::ofstream::out | std::ofstream::trunc);
    for (size_t I = 0; I < dim; ++I) {
        file << vec[I] << "\n";
    }
}

void TDCI::save_vector(std::vector<size_t>& vec, std::string name) {

    size_t dim = vec.size();
    std::ofstream file;
    file.open(name, std::ofstream::out | std::ofstream::trunc);
    for (size_t I = 0; I < dim; ++I) {
        file << vec[I] << "\n";
    }
}
void TDCI::save_vector(std::vector<double>& vec, std::string name) {

    size_t dim = vec.size();
    std::ofstream file;
    file.open(name, std::ofstream::out | std::ofstream::trunc);
    for (size_t I = 0; I < dim; ++I) {
        file << std::setw(12) << std::setprecision(11) << vec[I] << "\n";
    }
}
void TDCI::annihilate_wfn(DeterminantHashVec& olddets, DeterminantHashVec& anndets, int frz_orb) {

    // Loop through determinants, annihilate frz_orb(alpha)
    const det_hashvec& dets = olddets.wfn_hash();
    size_t ndet = dets.size();
    for (size_t I = 0; I < ndet; ++I) {
        auto& detI = dets[I];
        if (detI.get_alfa_bit(frz_orb) == true) {
            Determinant new_det(detI);
            new_det.set_alfa_bit(frz_orb, false);
            if (!anndets.has_det(new_det)) {
                //                outfile->Printf("\n %s", new_det.str(ncmo ).c_str());
                anndets.add(new_det);
            }
        }
    }
}

std::vector<double> TDCI::compute_occupation(DeterminantHashVec& dets, std::vector<double>& Cr,
                                             std::vector<double>& Ci, std::vector<int>& orbs) {

    size_t nact = Cr.size();
    std::vector<double> occ_vec(orbs.size(), 0.0);

    for (size_t i = 0; i < orbs.size(); ++i) {
        double occ = 0.0;
        int orb = orbs[i];
        for (size_t I = 0; I < nact; ++I) {

            const Determinant& detI = dets.get_det(I);
            if (detI.get_alfa_bit(orb) == true) {
                size_t idx = dets.get_idx(detI);
                double re = Cr[idx];
                double im = Ci[idx];
                occ += re * re + im * im;
            }
        }
        occ_vec[i] = occ;
    }
    return occ_vec;
}

std::vector<double> TDCI::compute_occupation(SharedVector Cr, SharedVector Ci,
                                             std::vector<int>& orbs) {

    size_t nact = Cr->dim();
    std::vector<double> occ_vec(orbs.size(), 0.0);

    for (size_t i = 0; i < orbs.size(); ++i) {
        double occ = 0.0;
        int orb = orbs[i];
        for (size_t I = 0; I < nact; ++I) {

            const Determinant& detI = ann_dets_.get_det(I);
            if (detI.get_alfa_bit(orb) == true) {
                size_t idx = ann_dets_.get_idx(detI);
                double re = Cr->get(idx);
                double im = Ci->get(idx);
                occ += re * re + im * im;
            }
        }
        occ_vec[i] = occ;
    }
    return occ_vec;
}

void TDCI::compute_tdci_select(SharedVector C0) {

    Timer t1;
    double eta = options_->get_double("TDCI_ETA_P");
    int nact = mo_space_info_->size("ACTIVE");

    // A list of orbitals to compute occupations_ during propagation
    std::vector<int> orbs = options_->get_int_list("TDCI_OCC_ORB");

    // Timestep details
    int nstep = options_->get_int("TDCI_NSTEP");
    double dt = options_->get_double("TDCI_TIMESTEP");
    double conv = 1.0 / 24.18884326505;
    dt *= conv;
    double time = dt;

    occupations_.resize(orbs.size());
    // Get the initial P and Q space determinants
    DeterminantHashVec P_space;
    DeterminantHashVec PQ_space;

    const det_hashvec& dets = ann_dets_.wfn_hash();
    size_t n_core_dets = core_dets_.size();
    size_t n_ann_dets = ann_dets_.size();
    std::vector<std::pair<double, size_t>> sorted_dets(n_ann_dets);
    for (size_t I = 0; I < n_ann_dets; ++I) {
        double cI = C0->get(I);
        sorted_dets[I] = std::make_pair(cI * cI, I);
    }
    // Sort by abs value of coefficient
    std::sort(sorted_dets.begin(), sorted_dets.end());
    //    for( size_t I = n_ann_dets-n_core_dets; I < n_ann_dets; ++I ){
    //        auto d_pair = sorted_dets[I];
    //        double ci = C0->get(d_pair.second);
    ////        outfile->Printf("\n  %12.10f: %s",ci, dets[d_pair.second].str(nact).c_str());
    //    }

    std::vector<double> P_coeffs_r;
    std::vector<double> PQ_coeffs_r;
    std::vector<double> PQ_coeffs_i;
    double sum = 0.0;
    size_t n_excluded = 0;
    for (size_t I = (n_ann_dets - n_core_dets); I < n_ann_dets; ++I) {
        auto d_pair = sorted_dets[I];
        double cI = d_pair.first;
        Determinant det = dets[d_pair.second];
        if (sum + cI < eta) {
            sum += cI;
            n_excluded++;
            //            outfile->Printf("\n (%6.4f) %10.6f: %s", sum, cI, det.str(nact).c_str());
        } else {
            //            break; // I think this is faster
            P_space.add(det);
            P_coeffs_r.push_back(C0->get(d_pair.second));
        }
    }
    //    outfile->Printf("\n  Remove %zu out of %zu", n_excluded, n_core_dets);
    std::vector<double> P_coeffs_i(P_space.size(), 0.0);
    double norm = 0.0;
    for (size_t I = 0, maxI = P_space.size(); I < maxI; ++I) {
        double val_r = P_coeffs_r[I];
        norm += val_r * val_r;
    }
    norm = 1.0 / std::sqrt(norm);
    for (size_t I = 0, maxI = P_space.size(); I < maxI; ++I) {
        double cr = P_coeffs_r[I];
        cr *= norm;
        P_coeffs_r[I] = cr;
    }

    //   for( size_t I = n_excluded; I < n_core_dets; ++I ){
    //       auto d_pair = sorted_dets[I];
    //       double cI = d_pair.first;
    //       Determinant det = d_pair.second;
    //
    //       outfile->Printf("\n  %10.6f: %s", cI, det.str(nact).c_str());
    //       P_coeffs_r[I - n_excluded] = cI;
    //       P_space.add(det);
    //   }

    // Begin the timesteps
    for (int N = 0; N < nstep; ++N) {
        Timer total;

        // 1. Get 1st order opproximation to current basis for propagation
        Timer pq;
        get_PQ_space(P_space, P_coeffs_r, P_coeffs_i, PQ_space, PQ_coeffs_r, PQ_coeffs_i);

        // 2. Propogate in PQ space
        Timer prop;

        if (options_->get_str("TDCI_PROPAGATOR") == "EXACT_SELECT") {
            propagate_exact_select(PQ_coeffs_r, PQ_coeffs_i, PQ_space, dt);
        } else if (options_->get_str("TDCI_PROPAGATOR") == "RK4_SELECT") {
            propagate_RK4_select(PQ_coeffs_r, PQ_coeffs_i, PQ_space, dt);
        } else if (options_->get_str("TDCI_PROPAGATOR") == "RK4_SELECT_LIST") {
            // build coupling lists
            auto mo_sym = mo_space_info_->symmetry("ACTIVE");
            DeterminantSubstitutionLists op(as_ints_);
            op.set_quiet_mode(true);

            op.build_strings(PQ_space);
            op.op_s_lists(PQ_space);
            op.tp_s_lists(PQ_space);
            propagate_RK4_list(PQ_coeffs_r, PQ_coeffs_i, PQ_space, op, dt);
        }

        //        outfile->Printf("\n  propagate: %1.6f", prop.get());

        //        const det_hashvec& PQ_dets = PQ_space.wfn_hash();
        //        for( size_t I = 0; I < PQ_space.size(); ++I ){
        //            const Determinant& det = PQ_dets[I];
        //            if( P_space.has_det(det) ){
        //                size_t p_idx = P_space.get_idx(det);
        //                outfile->Printf("\n  %11.8f  %11.8f  %s", P_coeffs_r[p_idx],
        //                PQ_coeffs_r[I], det.str(nact).c_str());
        //            } else {
        //                outfile->Printf("\n  %11.8f  %11.8f  %s", 0.0, PQ_coeffs_r[I],
        //                det.str(nact).c_str());
        //            }
        //        }

        if (std::abs((time / conv) - round(time / conv)) <= 1e-8) {
            outfile->Printf("\n  (t = %10.2f)  P: %6zu, PQ: %6zu   (%1.6f)", time / conv,
                            P_space.size(), PQ_space.size(), pq.get());
            if (options_->get_bool("TDCI_PRINT_WFN")) {
                std::stringstream ss;
                ss << std::fixed << std::setprecision(3) << time / conv;
                save_vector(PQ_coeffs_r, "select_" + ss.str() + "_r.txt");
                save_vector(PQ_coeffs_i, "select_" + ss.str() + "_i.txt");

                size_t npq = PQ_space.size();
                std::vector<std::string> det_str(npq);
                const det_hashvec& PQ_dets = PQ_space.wfn_hash();
                for (size_t I = 0; I < npq; ++I) {
                    auto detI = PQ_dets[I];
                    /// det_str[I] = detI.str(nact).c_str();
                    det_str[I] = str(detI, nact).c_str();
                }
                save_vector(det_str, "determinants_" + ss.str() + ".txt");
            }
            // 3. Save wfn/occ to file
            std::vector<double> occ = compute_occupation(PQ_space, PQ_coeffs_r, PQ_coeffs_i, orbs);
            for (size_t i = 0; i < orbs.size(); ++i) {
                occupations_[i].push_back(occ[i]);
            }
        }

        // 4. Update P space
        Timer prune;
        update_P_space(P_space, P_coeffs_r, P_coeffs_i, PQ_space, PQ_coeffs_r, PQ_coeffs_i);
        // outfile->Printf("\n  prune: %1.6f", prune.get());

        time += dt;

        //      outfile->Printf("\n total: %8.6f s", total.get());
    }
    for (size_t i = 0; i < orbs.size(); ++i) {
        save_vector(occupations_[i], "occupations_" + std::to_string(orbs[i]) + ".txt");
    }

    outfile->Printf("\n Time spent propagating (exact): %1.6f s", t1.get());
}

void TDCI::get_PQ_space(DeterminantHashVec& P_space, std::vector<double>& P_coeffs_r,
                        std::vector<double>& P_coeffs_i, DeterminantHashVec& PQ_space,
                        std::vector<double>& PQ_coeffs_r, std::vector<double>& PQ_coeffs_i) {

    double eta = options_->get_double("TDCI_ETA_PQ");
    double dt = options_->get_double("TDCI_TIMESTEP");
    double conv = 1.0 / 24.18884326505;
    dt *= conv;
    int nact = mo_space_info_->size("ACTIVE");
    auto mo_sym = mo_space_info_->symmetry("ACTIVE");
    double thresh = options_->get_double("TDCI_PRESCREEN_THRESH");

    size_t max_P = P_space.size();
    const det_hashvec& P_dets = P_space.wfn_hash();

    DeterminantHashVec F_space;
    std::vector<double> F_approx_r;
    std::vector<double> F_approx_i;

    //#pragma omp parallel
    //    {
    //        int ntd = omp_get_num_threads();
    //        int tid = omp_get_thread_num();
    //        int bin_size = max_P / ntd;
    //        bin_size += (tid < (max_P % ntd)) ? 1 : 0;
    //        int start_idx = (tid < (max_P%ntd)) ? tid*bin_size : (max_P%ntd)*(bin_size+1) + (tid -
    //        (max_P%ntd))*bin_size; int end_idx = start_idx + bin_size;
    size_t start_idx = 0;
    size_t end_idx = max_P;

    det_hash<std::pair<double, double>> Hc_t;

    // First do diagonal part
    for (size_t P = start_idx; P < end_idx; ++P) {
        const Determinant& det = P_dets[P];
        double Cp_r = P_coeffs_r[P];
        double Cp_i = P_coeffs_i[P];
        Hc_t[det] = std::make_pair(Cp_r * as_ints_->energy(det), Cp_i * as_ints_->energy(det));
    }

    for (size_t P = start_idx; P < end_idx; ++P) {
        const Determinant& det = P_dets[P];
        double Cp_r = P_coeffs_r[P];
        double Cp_i = P_coeffs_i[P];
        double Cmag_sq = Cp_r * Cp_r + Cp_i * Cp_i;

        std::vector<int> aocc = det.get_alfa_occ(nact);
        std::vector<int> bocc = det.get_beta_occ(nact);
        std::vector<int> avir = det.get_alfa_vir(nact);
        std::vector<int> bvir = det.get_beta_vir(nact);

        size_t noalpha = aocc.size();
        size_t nobeta = bocc.size();
        size_t nvalpha = avir.size();
        size_t nvbeta = bvir.size();
        //            outfile->Printf("\n  %10.6f: %s", Cp_r, det.str(nact).c_str());

        Determinant new_det(det);
        // single alpha
        for (size_t i = 0; i < noalpha; ++i) {
            size_t ii = aocc[i];
            for (size_t a = 0; a < nvalpha; ++a) {
                size_t aa = avir[a];
                if ((mo_sym[ii] ^ mo_sym[aa]) == 0) {
                    double int_ia = as_ints_->slater_rules_single_alpha(det, ii, aa);
                    if (std::fabs(int_ia * Cmag_sq) >= thresh) {

                        new_det = det;
                        new_det.set_alfa_bit(ii, false);
                        new_det.set_alfa_bit(aa, true);

                        double HIJ_r = int_ia * Cp_r;
                        double HIJ_i = int_ia * Cp_i;
                        auto it = Hc_t.find(new_det);
                        if (it != Hc_t.end()) {
                            auto pair = Hc_t[new_det];
                            double cr = pair.first;
                            double ci = pair.second;
                            cr += HIJ_r;
                            ci += HIJ_i;
                            Hc_t[new_det] = std::make_pair(cr, ci);
                        } else {
                            Hc_t[new_det] = std::make_pair(HIJ_r, HIJ_i);
                        }
                        //                        outfile->Printf("\n %s",
                        //                        new_det.str(nact).c_str());
                    }
                }
            }
        }
        // single beta
        for (size_t i = 0; i < nobeta; ++i) {
            size_t ii = bocc[i];
            for (size_t a = 0; a < nvbeta; ++a) {
                size_t aa = bvir[a];
                if ((mo_sym[ii] ^ mo_sym[aa]) == 0) {
                    double int_ia = as_ints_->slater_rules_single_beta(det, ii, aa);
                    if (std::fabs(int_ia * Cmag_sq) >= thresh) {
                        new_det = det;
                        new_det.set_beta_bit(ii, false);
                        new_det.set_beta_bit(aa, true);
                        double HIJ_r = int_ia * Cp_r;
                        double HIJ_i = int_ia * Cp_i;
                        auto it = Hc_t.find(new_det);
                        if (it != Hc_t.end()) {
                            auto pair = Hc_t[new_det];
                            double cr = pair.first;
                            double ci = pair.second;
                            cr += HIJ_r;
                            ci += HIJ_i;
                            Hc_t[new_det] = std::make_pair(cr, ci);
                        } else {
                            Hc_t[new_det] = std::make_pair(HIJ_r, HIJ_i);
                        }
                    }
                }
            }
        }
        // aabb doubles
        for (size_t i = 0; i < noalpha; ++i) {
            size_t ii = aocc[i];
            for (size_t j = 0; j < nobeta; ++j) {
                size_t jj = bocc[j];
                for (size_t a = 0; a < nvalpha; ++a) {
                    size_t aa = avir[a];
                    for (size_t b = 0; b < nvbeta; ++b) {
                        size_t bb = bvir[b];
                        if ((mo_sym[ii] ^ mo_sym[jj] ^ mo_sym[aa] ^ mo_sym[bb]) == 0) {
                            double int_ijab = as_ints_->tei_ab(ii, jj, aa, bb);
                            if (std::fabs(int_ijab * Cmag_sq) >= thresh) {
                                new_det = det;
                                int_ijab *= new_det.double_excitation_ab(ii, jj, aa, bb);
                                double HIJ_r = int_ijab * Cp_r;
                                double HIJ_i = int_ijab * Cp_i;

                                auto it = Hc_t.find(new_det);
                                if (it != Hc_t.end()) {
                                    auto pair = Hc_t[new_det];
                                    double cr = pair.first;
                                    double ci = pair.second;
                                    cr += HIJ_r;
                                    ci += HIJ_i;
                                    Hc_t[new_det] = std::make_pair(cr, ci);
                                } else {
                                    Hc_t[new_det] = std::make_pair(HIJ_r, HIJ_i);
                                }
                            }
                        }
                    }
                }
            }
        }
        // aaaa doubles
        for (size_t i = 0; i < noalpha; ++i) {
            size_t ii = aocc[i];
            for (size_t j = i + 1; j < noalpha; ++j) {
                size_t jj = aocc[j];
                for (size_t a = 0; a < nvalpha; ++a) {
                    size_t aa = avir[a];
                    for (size_t b = a + 1; b < nvalpha; ++b) {
                        size_t bb = avir[b];
                        if ((mo_sym[ii] ^ mo_sym[jj] ^ mo_sym[aa] ^ mo_sym[bb]) == 0) {
                            double int_ijab = as_ints_->tei_aa(ii, jj, aa, bb);
                            if (std::fabs(int_ijab * Cmag_sq) >= thresh) {
                                new_det = det;
                                int_ijab *= new_det.double_excitation_aa(ii, jj, aa, bb);

                                double HIJ_r = int_ijab * Cp_r;
                                double HIJ_i = int_ijab * Cp_i;

                                auto it = Hc_t.find(new_det);
                                if (it != Hc_t.end()) {
                                    auto pair = Hc_t[new_det];
                                    double cr = pair.first;
                                    double ci = pair.second;
                                    cr += HIJ_r;
                                    ci += HIJ_i;
                                    Hc_t[new_det] = std::make_pair(cr, ci);
                                } else {
                                    Hc_t[new_det] = std::make_pair(HIJ_r, HIJ_i);
                                }
                            }
                        }
                    }
                }
            }
        }
        // bbbb doubles
        for (size_t i = 0; i < nobeta; ++i) {
            size_t ii = bocc[i];
            for (size_t j = i + 1; j < nobeta; ++j) {
                size_t jj = bocc[j];
                for (size_t a = 0; a < nvbeta; ++a) {
                    size_t aa = bvir[a];
                    for (size_t b = a + 1; b < nvbeta; ++b) {
                        size_t bb = bvir[b];
                        if ((mo_sym[ii] ^ mo_sym[jj] ^ mo_sym[aa] ^ mo_sym[bb]) == 0) {
                            double int_ijab = as_ints_->tei_bb(ii, jj, aa, bb);
                            if (std::fabs(int_ijab * Cmag_sq) >= thresh) {
                                new_det = det;
                                int_ijab *= new_det.double_excitation_bb(ii, jj, aa, bb);

                                double HIJ_r = int_ijab * Cp_r;
                                double HIJ_i = int_ijab * Cp_i;

                                auto it = Hc_t.find(new_det);
                                if (it != Hc_t.end()) {
                                    auto pair = Hc_t[new_det];
                                    double cr = pair.first;
                                    double ci = pair.second;
                                    cr += HIJ_r;
                                    ci += HIJ_i;
                                    Hc_t[new_det] = std::make_pair(cr, ci);
                                } else {
                                    Hc_t[new_det] = std::make_pair(HIJ_r, HIJ_i);
                                }
                            }
                        }
                    }
                }
            }
        }
    } // loop over reference

    //      // Merge
    //      #pragma omp critical
    //      {
    for (auto& pair : Hc_t) {
        const Determinant& det = pair.first;
        if (F_space.has_det(det)) {
            size_t idx = F_space.get_idx(det);
            F_approx_i[idx] -= pair.second.first * dt;
            F_approx_r[idx] += pair.second.second * dt;
        } else {
            F_space.add(det);
            //                    size_t idx = F_space.get_idx(det);
            F_approx_i.push_back(pair.second.first * -1.0 * dt);
            F_approx_r.push_back(pair.second.second * dt);
        }
    }
    //      }
    //   } // close threads

    // Compute full correction vector
    size_t nF = F_space.size();
    const det_hashvec& F_dets = F_space.wfn_hash();
    std::vector<std::pair<double, Determinant>> sorted_dets(nF);

    double norm = 0.0;
    for (size_t I = 0; I < nF; ++I) {
        double cr = F_approx_r[I];
        double ci = F_approx_i[I];

        const Determinant& det = F_dets[I];
        if (P_space.has_det(det)) {
            size_t idx = P_space.get_idx(det);
            cr += P_coeffs_r[idx];
            ci += P_coeffs_i[idx];
        }

        F_approx_r[I] = cr;
        F_approx_i[I] = ci;

        norm += cr * cr + ci * ci;
    }

    // Copy normalized vectors into sortable list
    norm = 1.0 / std::sqrt(norm);
    for (size_t I = 0; I < nF; ++I) {
        double cr = F_approx_r[I] * norm;
        double ci = F_approx_i[I] * norm;
        const Determinant& det = F_dets[I];
        sorted_dets[I] = std::make_pair(cr * cr + ci * ci, det);
    }

    // Now, screen the determinants
    std::sort(sorted_dets.begin(), sorted_dets.end());

    // Get a copy of PQ space
    DeterminantHashVec PQ_copy(PQ_space);
    std::vector<double> PQ_copy_r = PQ_coeffs_r;
    std::vector<double> PQ_copy_i = PQ_coeffs_i;

    PQ_space.clear();
    double sum = 0.0;
    for (size_t I = 0; I < nF; ++I) {

        const auto& dpair = sorted_dets[I];
        const double cI = dpair.first;

        if ((sum + cI) < eta) {
            sum += cI;
        } else {
            PQ_space.add(dpair.second);
        }
    }
    // This will be the initial state for propagation
    size_t npq = PQ_space.size();
    PQ_coeffs_r.clear();
    PQ_coeffs_i.clear();

    PQ_coeffs_r.resize(npq, 0.0);
    PQ_coeffs_i.resize(npq, 0.0);

    const det_hashvec& PQ_dets = PQ_space.wfn_hash();

    norm = 0.0;
    for (size_t I = 0; I < npq; ++I) {
        const Determinant& det = PQ_dets[I];

        if (P_space.has_det(det)) {
            size_t p_idx = P_space.get_idx(det);
            double& cr = PQ_coeffs_r[I];
            double& ci = PQ_coeffs_i[I];
            cr = P_coeffs_r[p_idx];
            ci = P_coeffs_i[p_idx];

            norm += (cr * cr + ci * ci);
        } else if (PQ_copy.has_det(det)) {
            size_t pq_idx = PQ_copy.get_idx(det);
            double& cr = PQ_coeffs_r[I];
            double& ci = PQ_coeffs_i[I];
            cr = PQ_copy_r[pq_idx];
            ci = PQ_copy_i[pq_idx];
            norm += (cr * cr + ci * ci);
        }
    }

    norm = 1.0 / std::sqrt(norm);
    std::transform(PQ_coeffs_r.begin(), PQ_coeffs_r.end(), PQ_coeffs_r.begin(),
                   std::bind(std::multiplies<double>(), std::placeholders::_1, norm));

    std::transform(PQ_coeffs_i.begin(), PQ_coeffs_i.end(), PQ_coeffs_i.begin(),
                   std::bind(std::multiplies<double>(), std::placeholders::_1, norm));
}

void TDCI::propagate_exact_select(std::vector<double>& PQ_coeffs_r,
                                  std::vector<double>& PQ_coeffs_i, DeterminantHashVec& PQ_space,
                                  double dt) {

    // Build a full Hamiltonian in the PQ space
    size_t npq = PQ_space.size();
    SharedMatrix H = std::make_shared<Matrix>("H", npq, npq);

    const det_hashvec& PQ_dets = PQ_space.wfn_hash();
    for (size_t I = 0; I < npq; ++I) {
        const Determinant& detI = PQ_dets[I];
        for (size_t J = I; J < npq; ++J) {
            const Determinant& detJ = PQ_dets[J];
            double value = as_ints_->slater_rules(detI, detJ);
            H->set(I, J, value);
            H->set(J, I, value);
        }
    }

    // Diagonalize the Hamiltonian
    SharedMatrix evecs = std::make_shared<Matrix>("evecs", npq, npq);
    SharedVector evals = std::make_shared<Vector>("evals", npq);
    H->diagonalize(evecs, evals);

    std::vector<double> int_r(npq, 0.0);
    std::vector<double> int_i(npq, 0.0);

    C_DGEMV('t', npq, npq, 1.0, &(evecs->pointer(0)[0][0]), npq, &(PQ_coeffs_r[0]), 1, 0.0,
            &(int_r[0]), 1);
    C_DGEMV('t', npq, npq, 1.0, &(evecs->pointer(0)[0][0]), npq, &(PQ_coeffs_i[0]), 1, 0.0,
            &(int_i[0]), 1);

    for (size_t I = 0; I < npq; ++I) {
        double rval = int_r[I];
        double ival = int_i[I];
        double eval = evals->get(I);
        int_r[I] = std::cos(eval * dt) * rval + std::sin(eval * dt) * ival;
        int_i[I] = std::cos(eval * dt) * ival - std::sin(eval * dt) * rval;
    }
    C_DGEMV('n', npq, npq, 1.0, &(evecs->pointer(0)[0][0]), npq, &(int_r[0]), 1, 0.0,
            &(PQ_coeffs_r[0]), 1);
    C_DGEMV('n', npq, npq, 1.0, &(evecs->pointer(0)[0][0]), npq, &(int_i[0]), 1, 0.0,
            &(PQ_coeffs_i[0]), 1);
}

void TDCI::update_P_space(DeterminantHashVec& P_space, std::vector<double>& P_coeffs_r,
                          std::vector<double>& P_coeffs_i, DeterminantHashVec& PQ_space,
                          std::vector<double>& PQ_coeffs_r, std::vector<double>& PQ_coeffs_i) {

    // Clear the P space
    P_space.clear();
    P_coeffs_r.clear();
    P_coeffs_i.clear();

    size_t npq = PQ_space.size();

    // Put PQ |c_I|^2 into sortable list
    std::vector<std::pair<double, size_t>> sorted_dets(npq);
    for (size_t I = 0; I < npq; ++I) {
        double cr = PQ_coeffs_r[I];
        double ci = PQ_coeffs_i[I];

        sorted_dets[I] = std::make_pair(cr * cr + ci * ci, I);
    }

    std::sort(sorted_dets.begin(), sorted_dets.end());

    double eta = options_->get_double("TDCI_ETA_P");
    const det_hashvec& PQ_dets = PQ_space.wfn_hash();

    double sum = 0.0;
    size_t last = 0;
    for (size_t I = 0; I < npq; ++I) {
        double mag = sorted_dets[I].first;

        if (mag + sum < eta) {
            sum += mag;
            last = I;
        } else {
            size_t idx = sorted_dets[I].second;
            P_space.add(PQ_dets[idx]);
        }
    }

    size_t np = P_space.size();
    P_coeffs_r.resize(np, 0.0);
    P_coeffs_i.resize(np, 0.0);
    const det_hashvec& P_dets = P_space.wfn_hash();

    double norm = 0.0;
    for (size_t I = 0; I < np; ++I) {
        const Determinant& det = P_dets[I];
        size_t idx = PQ_space.get_idx(det);
        double val_r = PQ_coeffs_r[idx];
        double val_i = PQ_coeffs_i[idx];
        P_coeffs_r[I] = val_r;
        P_coeffs_i[I] = val_i;
        norm += val_r * val_r + val_i * val_i;
    }
    //    outfile->Printf("\n  norm: %14.8f", norm);
    norm = 1.0 / std::sqrt(norm);
    std::transform(P_coeffs_r.begin(), P_coeffs_r.end(), P_coeffs_r.begin(),
                   std::bind(std::multiplies<double>(), std::placeholders::_1, norm));

    std::transform(P_coeffs_i.begin(), P_coeffs_i.end(), P_coeffs_i.begin(),
                   std::bind(std::multiplies<double>(), std::placeholders::_1, norm));
    //  for( size_t I = 0; I < np; ++I ){
    //      double& cr = P_coeffs_r[I];
    //      double& ci = P_coeffs_i[I];
    //      cr *= norm;
    //      ci *= norm;
    //  }
}

void TDCI::propagate_RK4_select(std::vector<double>& PQ_coeffs_r, std::vector<double>& PQ_coeffs_i,
                                DeterminantHashVec& PQ_space, double dt) {

    Timer total;
    size_t npq = PQ_space.size();

    SharedMatrix H = std::make_shared<Matrix>("H", npq, npq);

    // dumb implementation for now:
    SharedVector ct_r = std::make_shared<Vector>("ctr", npq);
    SharedVector ct_i = std::make_shared<Vector>("ctr", npq);

    const det_hashvec& PQ_dets = PQ_space.wfn_hash();
    for (size_t I = 0; I < npq; ++I) {
        const Determinant& detI = PQ_dets[I];

        ct_r->set(I, PQ_coeffs_r[I]);
        ct_i->set(I, PQ_coeffs_i[I]);

        for (size_t J = I; J < npq; ++J) {
            const Determinant& detJ = PQ_dets[J];
            double value = as_ints_->slater_rules(detI, detJ);
            H->set(I, J, value);
            H->set(J, I, value);
        }
    }

    //   outfile->Printf("\n    Build H: %1.6f", total.get());

    // k1
    SharedVector k1r = std::make_shared<Vector>("k1r", npq);
    SharedVector k1i = std::make_shared<Vector>("k1i", npq);

    k1r->gemv(false, 1.0, *H, *ct_i, 0.0);
    k1i->gemv(false, -1.0, *H, *ct_r, 0.0);

    // k2
    SharedVector intr = std::make_shared<Vector>("intr", npq);
    SharedVector inti = std::make_shared<Vector>("inti", npq);

    intr->copy(ct_r->clone());
    inti->copy(ct_i->clone());

    k1r->scale(0.5 * dt);
    k1i->scale(0.5 * dt);
    intr->add(*k1r);
    inti->add(*k1i);
    k1r->scale(2.0 / dt);
    k1i->scale(2.0 / dt);

    SharedVector k2r = std::make_shared<Vector>("k2r", npq);
    SharedVector k2i = std::make_shared<Vector>("k2i", npq);

    k2r->gemv(false, 1.0, *H, *inti, 0.0);
    k2i->gemv(false, -1.0, *H, *intr, 0.0);

    // k3
    intr->copy(ct_r->clone());
    inti->copy(ct_i->clone());

    k2r->scale(0.5 * dt);
    k2i->scale(0.5 * dt);
    intr->add(*k2r);
    inti->add(*k2i);
    k2r->scale(2.0 * 1.0 / dt);
    k2i->scale(2.0 * 1.0 / dt);

    SharedVector k3r = std::make_shared<Vector>("k3r", npq);
    SharedVector k3i = std::make_shared<Vector>("k3i", npq);

    k3r->gemv(false, 1.0, *H, *inti, 0.0);
    k3i->gemv(false, -1.0, *H, *intr, 0.0);

    // k4
    intr->copy(ct_r->clone());
    inti->copy(ct_i->clone());

    k3r->scale(dt);
    k3i->scale(dt);
    intr->add(*k3r);
    inti->add(*k3i);
    k3r->scale(1.0 / dt);
    k3i->scale(1.0 / dt);

    SharedVector k4r = std::make_shared<Vector>("k4r", npq);
    SharedVector k4i = std::make_shared<Vector>("k4i", npq);

    k4r->gemv(false, 1.0, *H, *inti, 0.0);
    k4i->gemv(false, -1.0, *H, *intr, 0.0);

    // Compile all intermediates

    k1r->scale(dt / 6.0);
    k2r->scale(dt / 3.0);
    k3r->scale(dt / 3.0);
    k4r->scale(dt / 6.0);
    k1r->add(*k2r);
    k1r->add(*k3r);
    k1r->add(*k4r);
    ct_r->add(*k1r);

    k1i->scale(dt / 6.0);
    k2i->scale(dt / 3.0);
    k3i->scale(dt / 3.0);
    k4i->scale(dt / 6.0);
    k1i->add(*k2i);
    k1i->add(*k3i);
    k1i->add(*k4i);
    ct_i->add(*k1i);

    double norm = 0.0;
    for (size_t I = 0; I < npq; ++I) {
        double re = ct_r->get(I);
        double im = ct_i->get(I);
        norm += (re * re) + (im * im);
    }
    norm = std::sqrt(norm);
    ct_r->scale(1.0 / norm);
    ct_i->scale(1.0 / norm);

    for (size_t I = 0; I < npq; ++I) {
        PQ_coeffs_r[I] = ct_r->get(I);
        PQ_coeffs_i[I] = ct_i->get(I);
    }

    // outfile->Printf("\n  Time spent propagating (RK4): %1.6f", total.get());
}

void TDCI::propagate_RK4_list(std::vector<double>& PQ_coeffs_r, std::vector<double>& PQ_coeffs_i,
                              DeterminantHashVec& PQ_space, DeterminantSubstitutionLists& op,
                              double dt) {

    Timer total;
    size_t npq = PQ_space.size();

    // k1 = -iH|Psi>
    std::vector<double> k1r(npq, 0.0);
    std::vector<double> k1i(npq, 0.0);

    complex_sigma_build(k1i, k1r, PQ_coeffs_r, PQ_coeffs_i, PQ_space, op);

    // k2
    std::vector<double> intr = PQ_coeffs_r;
    std::vector<double> inti = PQ_coeffs_i;

    double half_dt = 0.5 * dt;
#pragma omp parallel for
    for (size_t I = 0; I < npq; ++I) {
        intr[I] += k1r[I] * half_dt;
        inti[I] -= k1i[I] * half_dt;
    }

    std::vector<double> k2r(npq, 0.0);
    std::vector<double> k2i(npq, 0.0);
    complex_sigma_build(k2i, k2r, intr, inti, PQ_space, op);

    // k3
#pragma omp parallel for
    for (size_t I = 0; I < npq; ++I) {
        intr[I] = PQ_coeffs_r[I] + k2r[I] * half_dt;
        inti[I] = PQ_coeffs_i[I] - k2i[I] * half_dt;
    }

    std::vector<double> k3r(npq, 0.0);
    std::vector<double> k3i(npq, 0.0);
    complex_sigma_build(k3i, k3r, intr, inti, PQ_space, op);
    // k4
#pragma omp parallel for
    for (size_t I = 0; I < npq; ++I) {
        intr[I] = PQ_coeffs_r[I] + k3r[I] * dt;
        inti[I] = PQ_coeffs_i[I] - k3i[I] * dt;
    }

    std::vector<double> k4r(npq, 0.0);
    std::vector<double> k4i(npq, 0.0);
    complex_sigma_build(k4i, k4r, intr, inti, PQ_space, op);

    // Compile all intermediates

    double s_dt = dt / 6.0;
#pragma omp parallel for
    for (size_t I = 0; I < npq; ++I) {
        PQ_coeffs_r[I] += s_dt * (k1r[I] + 2 * k2r[I] + 2 * k3r[I] + k4r[I]);
        PQ_coeffs_i[I] -= s_dt * (k1i[I] + 2 * k2i[I] + 2 * k3i[I] + k4i[I]);
    }

    double norm = 0.0;
    for (size_t I = 0; I < npq; ++I) {
        double re = PQ_coeffs_r[I];
        double im = PQ_coeffs_i[I];
        norm += (re * re) + (im * im);
    }
    norm = 1.0 / std::sqrt(norm);

    for (size_t I = 0; I < npq; ++I) {
        PQ_coeffs_r[I] *= norm;
        PQ_coeffs_i[I] *= norm;
    }
    // outfile->Printf("\n  Time spent propagating (RK4): %1.6f", total.get());
}

void TDCI::complex_sigma_build(std::vector<double>& sigma_r, std::vector<double>& sigma_i,
                               std::vector<double>& c_r, std::vector<double>& c_i,
                               DeterminantHashVec& dethash, DeterminantSubstitutionLists& op) {

    auto& dets = dethash.wfn_hash();
    size_t size = dets.size();

    // Get the lists
    auto& a_list = op.a_list_;
    auto& b_list = op.b_list_;
    auto& aa_list = op.aa_list_;
    auto& ab_list = op.ab_list_;
    auto& bb_list = op.bb_list_;

#pragma omp parallel
    {
        size_t num_thread = omp_get_max_threads();
        size_t tid = omp_get_thread_num();

        size_t bin_size = size / num_thread;
        bin_size += (tid < (size % num_thread)) ? 1 : 0;
        size_t start_idx = (tid < (size % num_thread)) ? tid * bin_size
                                                       : (size % num_thread) * (bin_size + 1) +
                                                             (tid - (size % num_thread)) * bin_size;
        size_t end_idx = start_idx + bin_size;

        for (size_t J = start_idx; J < end_idx; ++J) {
            double diag_J = as_ints_->energy(dets[J]);
            sigma_r[J] += diag_J * c_r[J];
            sigma_i[J] += diag_J * c_i[J];
        }

        // Each thread gets local copy of sigma
        std::vector<double> sigma_t_r(size, 0.0);
        std::vector<double> sigma_t_i(size, 0.0);

        // a singles
        size_t end_a_idx = a_list.size();
        size_t start_a_idx = 0;
        for (size_t K = start_a_idx, max_K = end_a_idx; K < max_K; ++K) {
            if ((K % num_thread) == tid) {
                const std::vector<std::pair<size_t, short>>& c_dets = a_list[K];
                size_t max_det = c_dets.size();
                for (size_t det = 0; det < max_det; ++det) {
                    auto& detJ = c_dets[det];
                    const size_t J = detJ.first;
                    const size_t p = std::abs(detJ.second) - 1;
                    double sign_p = detJ.second > 0.0 ? 1.0 : -1.0;
                    for (size_t det2 = det + 1; det2 < max_det; ++det2) {
                        auto& detI = c_dets[det2];
                        const size_t q = std::abs(detI.second) - 1;
                        if (p != q) {
                            const size_t I = detI.first;
                            double sign_q = detI.second > 0.0 ? 1.0 : -1.0;
                            const double HIJ =
                                as_ints_->slater_rules_single_alpha_abs(dets[J], p, q) * sign_p *
                                sign_q;
                            sigma_t_r[I] += HIJ * c_r[J];
                            sigma_t_r[J] += HIJ * c_r[I];

                            sigma_t_i[I] += HIJ * c_i[J];
                            sigma_t_i[J] += HIJ * c_i[I];
                        }
                    }
                }
            }
        }

        // b singles
        size_t end_b_idx = b_list.size();
        size_t start_b_idx = 0;
        for (size_t K = start_b_idx, max_K = end_b_idx; K < max_K; ++K) {
            // aa singles
            if ((K % num_thread) == tid) {
                const std::vector<std::pair<size_t, short>>& c_dets = b_list[K];
                size_t max_det = c_dets.size();
                for (size_t det = 0; det < max_det; ++det) {
                    auto& detJ = c_dets[det];
                    const size_t J = detJ.first;
                    const size_t p = std::abs(detJ.second) - 1;
                    double sign_p = detJ.second > 0.0 ? 1.0 : -1.0;
                    for (size_t det2 = det + 1; det2 < max_det; ++det2) {
                        auto& detI = c_dets[det2];
                        const size_t q = std::abs(detI.second) - 1;
                        if (p != q) {
                            const size_t I = detI.first;
                            double sign_q = detI.second > 0.0 ? 1.0 : -1.0;
                            const double HIJ =
                                as_ints_->slater_rules_single_beta_abs(dets[J], p, q) * sign_p *
                                sign_q;
                            sigma_t_r[I] += HIJ * c_r[J];
                            sigma_t_r[J] += HIJ * c_r[I];

                            sigma_t_i[I] += HIJ * c_i[J];
                            sigma_t_i[J] += HIJ * c_i[I];
                        }
                    }
                }
            }
        }

        // AA doubles
        size_t aa_size = aa_list.size();
        //      size_t bin_aa_size = aa_size / num_thread;
        //      bin_aa_size += (tid < (aa_size % num_thread)) ? 1 : 0;
        //      size_t start_aa_idx = (tid < (aa_size % num_thread))
        //                             ? tid * bin_aa_size
        //                             : (aa_size % num_thread) * (bin_aa_size + 1) +
        //                                   (tid - (aa_size % num_thread)) * bin_aa_size;
        //      size_t end_aa_idx = start_aa_idx + bin_aa_size;
        for (size_t K = 0, max_K = aa_size; K < max_K; ++K) {
            if ((K % num_thread) == tid) {
                const std::vector<std::tuple<size_t, short, short>>& c_dets = aa_list[K];
                size_t max_det = c_dets.size();
                for (size_t det = 0; det < max_det; ++det) {
                    auto& detJ = c_dets[det];
                    size_t J = std::get<0>(detJ);
                    short p = std::abs(std::get<1>(detJ)) - 1;
                    short q = std::get<2>(detJ);
                    double sign_p = std::get<1>(detJ) > 0.0 ? 1.0 : -1.0;
                    for (size_t det2 = det + 1; det2 < max_det; ++det2) {
                        auto& detI = c_dets[det2];
                        short r = std::abs(std::get<1>(detI)) - 1;
                        short s = std::get<2>(detI);
                        if ((p != r) and (q != s) and (p != s) and (q != r)) {
                            size_t I = std::get<0>(detI);
                            double sign_q = std::get<1>(detI) > 0.0 ? 1.0 : -1.0;
                            double HIJ = sign_p * sign_q * as_ints_->tei_aa(p, q, r, s);
                            sigma_t_r[I] += HIJ * c_r[J];
                            sigma_t_r[J] += HIJ * c_r[I];

                            sigma_t_i[I] += HIJ * c_i[J];
                            sigma_t_i[J] += HIJ * c_i[I];
                        }
                    }
                }
            }
        }

        // BB doubles
        for (size_t K = 0, max_K = bb_list.size(); K < max_K; ++K) {
            if ((K % num_thread) == tid) {
                const std::vector<std::tuple<size_t, short, short>>& c_dets = bb_list[K];
                size_t max_det = c_dets.size();
                for (size_t det = 0; det < max_det; ++det) {
                    auto& detJ = c_dets[det];
                    size_t J = std::get<0>(detJ);
                    short p = std::abs(std::get<1>(detJ)) - 1;
                    short q = std::get<2>(detJ);
                    double sign_p = std::get<1>(detJ) > 0.0 ? 1.0 : -1.0;
                    for (size_t det2 = det + 1; det2 < max_det; ++det2) {
                        auto& detI = c_dets[det2];
                        short r = std::abs(std::get<1>(detI)) - 1;
                        short s = std::get<2>(detI);
                        if ((p != r) and (q != s) and (p != s) and (q != r)) {
                            size_t I = std::get<0>(detI);
                            double sign_q = std::get<1>(detI) > 0.0 ? 1.0 : -1.0;
                            double HIJ = sign_p * sign_q * as_ints_->tei_bb(p, q, r, s);
                            sigma_t_r[I] += HIJ * c_r[J];
                            sigma_t_r[J] += HIJ * c_r[I];

                            sigma_t_i[I] += HIJ * c_i[J];
                            sigma_t_i[J] += HIJ * c_i[I];
                        }
                    }
                }
            }
        }
        for (size_t K = 0, max_K = ab_list.size(); K < max_K; ++K) {
            if ((K % num_thread) == tid) {
                const std::vector<std::tuple<size_t, short, short>>& c_dets = ab_list[K];
                size_t max_det = c_dets.size();
                for (size_t det = 0; det < max_det; ++det) {
                    auto detJ = c_dets[det];
                    size_t J = std::get<0>(detJ);
                    short p = std::abs(std::get<1>(detJ)) - 1;
                    short q = std::get<2>(detJ);
                    double sign_p = std::get<1>(detJ) > 0.0 ? 1.0 : -1.0;
                    for (size_t det2 = det + 1; det2 < max_det; ++det2) {
                        auto& detI = c_dets[det2];
                        short r = std::abs(std::get<1>(detI)) - 1;
                        short s = std::get<2>(detI);
                        if ((p != r) and (q != s)) {
                            size_t I = std::get<0>(detI);
                            double sign_q = std::get<1>(detI) > 0.0 ? 1.0 : -1.0;
                            double HIJ = sign_p * sign_q * as_ints_->tei_ab(p, q, r, s);
                            sigma_t_r[I] += HIJ * c_r[J];
                            sigma_t_r[J] += HIJ * c_r[I];

                            sigma_t_i[I] += HIJ * c_i[J];
                            sigma_t_i[J] += HIJ * c_i[I];
                        }
                    }
                }
            }
        }

#pragma omp critical
        {
            for (size_t I = 0; I < size; ++I) {
                // #pragma omp atomic update
                sigma_r[I] += sigma_t_r[I];
                sigma_i[I] += sigma_t_i[I];
            }
        }
    }
}

double TDCI::test_occ() {

    // only test lowest orb
    int orb = options_->get_int_list("TDCI_OCC_ORB")[0];

    // Load reference file
    std::ifstream ref_file("ref_occ_" + std::to_string(orb) + ".txt");

    std::vector<double> ref_occ;
    if (ref_file) {
        double value;
        while (ref_file >> value) {
            ref_occ.push_back(value);
        }
    } else {
        outfile->Printf("\n File not found!");
        exit(1);
    }

    std::vector<double>& curr_occ = occupations_[0];

    double diff = 0.0;
    for (size_t I = 0, maxI = curr_occ.size(); I < maxI; ++I) {
        double ref = ref_occ[I];
        double cur = curr_occ[I];
        diff += (cur - ref) * (cur - ref);
    }
    diff = sqrt(diff);

    // TODO: this is just a temporary solution
    psi::Process::environment.globals["OCCUPATION ERROR"] = diff;

    outfile->Printf("\n  Norm of error vector: %10.6f", diff);

    return diff;
}

} // namespace forte
