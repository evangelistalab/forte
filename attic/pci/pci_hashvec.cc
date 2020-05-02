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
#include <cmath>
#include <functional>
#include <tuple>
#include <unordered_map>
#include <utility>

#include "boost/format.hpp"
#include "boost/math/special_functions/bessel.hpp"

#include "psi4/libpsi4util/process.h"
#include "psi4/libciomr/libciomr.h"
#include "psi4/libmints/matrix.h"
#include "psi4/libmints/molecule.h"
#include "psi4/libmints/vector.h"
#include "psi4/libpsio/psio.h"
#include "psi4/libpsio/psio.hpp"
#include "psi4/libqt/qt.h"

#include "pci_hashvec.h"
#include "helpers/timer.h"
#include "sparse_ci/ci_reference.h"
#include "base_classes/state_info.h"
#include "base_classes/rdms.h"

using namespace psi;
using namespace forte::GeneratorType_HashVec;

#define USE_HASH 1
#define DO_STATS 0
#define ENFORCE_SYM 1

namespace forte {
#ifdef _OPENMP
#include <omp.h>
bool ProjectorCI_HashVec::have_omp_ = true;
#else
#define omp_get_max_threads() 1
#define omp_get_thread_num() 0
bool ProjectorCI_HashVec::have_omp_ = false;
#endif

void scale(std::vector<double>& A, double alpha);
double normalize(std::vector<double>& C);
double dot(std::vector<double>& C1, std::vector<double>& C2);
void add(std::vector<double>& a, double k, std::vector<double>& b);
void Wall_Chebyshev_generator_coefs(std::vector<double>& coefs, int order, double range);
void print_polynomial(std::vector<double>& coefs);

void add(const det_hashvec& A, std::vector<double>& Ca, double beta, const det_hashvec& B,
         const std::vector<double> Cb);

double dot(const det_hashvec& A, const std::vector<double> Ca, const det_hashvec& B,
           const std::vector<double> Cb);

void ProjectorCI_HashVec::sortHashVecByCoefficient(det_hashvec& dets_hashvec,
                                                   std::vector<double>& C) {
    size_t dets_size = dets_hashvec.size();
    std::vector<std::pair<double, size_t>> det_weight(dets_size);
    for (size_t I = 0; I < dets_size; ++I) {
        det_weight[I] = std::make_pair(std::fabs(C[I]), I);
    }
    std::sort(det_weight.begin(), det_weight.end(), std::greater<std::pair<double, size_t>>());
    std::vector<size_t> order_map(dets_size);
    for (size_t I = 0; I < dets_size; ++I) {
        order_map[det_weight[I].second] = I;
    }

    dets_hashvec.map_order(order_map);

    std::vector<double> new_C(dets_size);
    std::vector<std::pair<double, double>> new_dets_max_couplings(dets_size);
    for (size_t I = 0; I < dets_size; ++I) {
        new_C[order_map[I]] = C[I];
        new_dets_max_couplings[order_map[I]] = dets_max_couplings_[I];
    }
    C = std::move(new_C);
    dets_max_couplings_ = std::move(new_dets_max_couplings);
}

ProjectorCI_HashVec::ProjectorCI_HashVec(StateInfo state, size_t nroot,
                                         std::shared_ptr<forte::SCFInfo> scf_info,
                                         std::shared_ptr<ForteOptions> options,
                                         std::shared_ptr<MOSpaceInfo> mo_space_info,
                                         std::shared_ptr<ActiveSpaceIntegrals> as_ints)
    : ActiveSpaceMethod(state, nroot, mo_space_info, as_ints), scf_info_(scf_info),
      options_(options), fast_variational_estimate_(false) {
    // Copy the wavefunction information
    startup();
}

std::vector<RDMs> ProjectorCI_HashVec::rdms(const std::vector<std::pair<size_t, size_t>>& root_list,
                                            int max_rdm_level) {
    std::vector<RDMs> pci_ref;
    // TODO: implement
    throw std::runtime_error("ProjectorCI_HashVec::rdms is not implemented!");
    return pci_ref;
}

std::vector<RDMs>
ProjectorCI_HashVec::transition_rdms(const std::vector<std::pair<size_t, size_t>>& root_list,
                                     std::shared_ptr<ActiveSpaceMethod> method2,
                                     int max_rdm_level) {
    std::vector<RDMs> refs;
    throw std::runtime_error("ProjectorCI_HashVec::transition_rdms is not implemented!");
    return refs;
}

void ProjectorCI_HashVec::startup() {
    // The number of correlated molecular orbitals
    nact_ = mo_space_info_->corr_absolute_mo("ACTIVE").size();
    nactpi_ = mo_space_info_->dimension("ACTIVE");

    // Include frozen_docc and restricted_docc
    frzcpi_ = mo_space_info_->dimension("INACTIVE_DOCC");
    nfrzc_ = mo_space_info_->size("INACTIVE_DOCC");

    nuclear_repulsion_energy_ = as_ints_->ints()->nuclear_repulsion_energy();

    mo_symmetry_ = mo_space_info_->symmetry("ACTIVE");

    wavefunction_symmetry_ = state_.irrep();
    wavefunction_multiplicity_ = state_.multiplicity();

    // Number of correlated electrons
    nactel_ = 0;
    nalpha_ = 0;
    nbeta_ = 0;
    int nel = state_.na() + state_.nb();
    nirrep_ = mo_space_info_->nirrep();

    int ms = wavefunction_multiplicity_ - 1;
    nactel_ = nel - 2 * nfrzc_;
    nalpha_ = (nactel_ + ms) / 2;
    nbeta_ = nactel_ - nalpha_;

    // Build the reference determinant and compute its energy
    std::vector<Determinant> reference_vec;
    CI_Reference ref(scf_info_, options_, mo_space_info_, as_ints_, wavefunction_multiplicity_, ms,
                wavefunction_symmetry_);
    ref.set_ref_type("HF");
    ref.build_reference(reference_vec);
    reference_determinant_ = reference_vec[0];

    //    outfile->Printf("\n  The reference determinant is:\n");
    //    reference_determinant_.print();

    nroot_ = options_->get_int("PCI_NROOT");
    current_root_ = -1;
    post_diagonalization_ = options_->get_bool("PCI_POST_DIAGONALIZE");
    diag_method_ = DLSolver;
    if (options_->has_changed("DIAG_ALGORITHM")) {
        if (options_->get_str("DIAG_ALGORITHM") == "FULL") {
            diag_method_ = Full;
        } else if (options_->get_str("DIAG_ALGORITHM") == "DLSTRING") {
            diag_method_ = DLString;
        } else if (options_->get_str("DIAG_ALGORITHM") == "DLDISK") {
            diag_method_ = DLDisk;
        }
    }
    //    /-> Define appropriate variable: post_diagonalization_ =
    //    options_->get_bool("EX_ALGORITHM");

    spawning_threshold_ = options_->get_double("PCI_SPAWNING_THRESHOLD");
    initial_guess_spawning_threshold_ = options_->get_double("PCI_GUESS_SPAWNING_THRESHOLD");
    if (initial_guess_spawning_threshold_ < 0.0)
        initial_guess_spawning_threshold_ = 10.0 * spawning_threshold_;
    time_step_ = options_->get_double("PCI_TAU");
    maxiter_ = options_->get_int("PCI_MAXBETA") / time_step_;
    max_Davidson_iter_ = options_->get_int("PCI_MAX_DAVIDSON_ITER");
    davidson_collapse_per_root_ = options_->get_int("PCI_DL_COLLAPSE_PER_ROOT");
    davidson_subspace_per_root_ = options_->get_int("PCI_DL_SUBSPACE_PER_ROOT");
    e_convergence_ = options_->get_double("PCI_E_CONVERGENCE");
    energy_estimate_threshold_ = options_->get_double("PCI_ENERGY_ESTIMATE_THRESHOLD");
    evar_max_error_ = options_->get_double("PCI_EVAR_MAX_ERROR");

    max_guess_size_ = options_->get_int("PCI_MAX_GUESS_SIZE");
    energy_estimate_freq_ = options_->get_int("PCI_ENERGY_ESTIMATE_FREQ");

    fast_variational_estimate_ = options_->get_bool("PCI_FAST_EVAR");
    do_shift_ = options_->get_bool("PCI_USE_SHIFT");
    use_inter_norm_ = options_->get_bool("PCI_USE_INTER_NORM");
    do_perturb_analysis_ = options_->get_bool("PCI_PERTURB_ANALYSIS");
    stop_higher_new_low_ = options_->get_bool("PCI_STOP_HIGHER_NEW_LOW");
    chebyshev_order_ = options_->get_int("PCI_CHEBYSHEV_ORDER");
    krylov_order_ = options_->get_int("PCI_KRYLOV_ORDER");

    variational_estimate_ = options_->get_bool("PCI_VAR_ESTIMATE");
    print_full_wavefunction_ = options_->get_bool("PCI_PRINT_FULL_WAVEFUNCTION");

    approx_E_tau_ = 1.0;
    approx_E_S_ = 0.0;

    if (options_->get_str("PCI_GENERATOR") == "WALL-CHEBYSHEV") {
        generator_ = WallChebyshevGenerator;
        generator_description_ = "Wall-Chebyshev";
        time_step_ = 1.0;
        if (chebyshev_order_ <= 0) {
            outfile->Printf("\n\n  Warning! Chebyshev order %d out of bound, "
                            "automatically adjusted to 5.",
                            chebyshev_order_);
            chebyshev_order_ = 5;
        }
    } else if (options_->get_str("PCI_GENERATOR") == "DL") {
        generator_ = DLGenerator;
        generator_description_ = "Davidson-Liu by Tianyuan";
        time_step_ = 1.0;
        if (krylov_order_ <= 0) {
            outfile->Printf("\n\n  Warning! Krylov order %d out of bound, "
                            "automatically adjusted to 8.",
                            krylov_order_);
            krylov_order_ = 8;
        }
    } else {
        outfile->Printf("\n\n  Warning! Generator Unsupported.");
        abort();
    }

    num_threads_ = omp_get_max_threads();
}

void ProjectorCI_HashVec::print_info() {
    // Print a summary
    std::vector<std::pair<std::string, int>> calculation_info{
        {"Symmetry", wavefunction_symmetry_},
        {"Multiplicity", wavefunction_multiplicity_},
        {"Number of roots", nroot_},
        {"Root used for properties", options_->get_int("ROOT")},
        {"Maximum number of iterations", maxiter_},
        {"Energy estimation frequency", energy_estimate_freq_},
        {"Number of threads", num_threads_}};

    std::vector<std::pair<std::string, double>> calculation_info_double{
        {"Time step (beta)", time_step_},
        {"Spawning threshold", spawning_threshold_},
        {"Initial guess spawning threshold", initial_guess_spawning_threshold_},
        {"Convergence threshold", e_convergence_},
        {"Energy estimate tollerance", energy_estimate_threshold_}};

    std::vector<std::pair<std::string, std::string>> calculation_info_string{
        {"Generator type", generator_description_},
        {"Shift the energy", do_shift_ ? "YES" : "NO"},
        {"Use intermediate normalization", use_inter_norm_ ? "YES" : "NO"},
        {"Fast variational estimate", fast_variational_estimate_ ? "YES" : "NO"},
        {"Result perturbation analysis", do_perturb_analysis_ ? "YES" : "NO"},
        {"Using OpenMP", have_omp_ ? "YES" : "NO"},
    };

    // Print some information
    outfile->Printf("\n\n  ==> Calculation Information <==\n");
    for (auto& str_dim : calculation_info) {
        outfile->Printf("\n    %-39s %10d", str_dim.first.c_str(), str_dim.second);
    }
    for (auto& str_dim : calculation_info_double) {
        outfile->Printf("\n    %-39s %10.3e", str_dim.first.c_str(), str_dim.second);
    }
    for (auto& str_dim : calculation_info_string) {
        outfile->Printf("\n    %-39s %10s", str_dim.first.c_str(), str_dim.second.c_str());
    }
}

double ProjectorCI_HashVec::estimate_high_energy() {
    double high_obt_energy = 0.0;
    int nea = 0, neb = 0;
    std::vector<std::pair<double, int>> obt_energies;
    Determinant high_det(reference_determinant_);
    for (size_t i = 0; i < nact_; i++) {
        if (reference_determinant_.get_alfa_bit(i)) {
            ++nea;
            high_det.destroy_alfa_bit(i);
        }
        if (reference_determinant_.get_beta_bit(i)) {
            ++neb;
            high_det.destroy_beta_bit(i);
        }

        double temp = as_ints_->oei_a(i, i);
        for (size_t p = 0; p < nact_; ++p) {
            if (reference_determinant_.get_alfa_bit(p)) {
                temp += as_ints_->tei_aa(i, p, i, p);
            }
            if (reference_determinant_.get_beta_bit(p)) {
                temp += as_ints_->tei_ab(i, p, i, p);
            }
        }
        obt_energies.push_back(std::make_pair(temp, i));
    }
    std::sort(obt_energies.begin(), obt_energies.end());

    for (int i = 1; i <= nea; i++) {
        high_obt_energy += obt_energies[obt_energies.size() - i].first;
        high_det.create_alfa_bit(obt_energies[obt_energies.size() - i].second);
    }
    for (int i = 1; i <= neb; i++) {
        high_obt_energy += obt_energies[obt_energies.size() - i].first;
        high_det.create_beta_bit(obt_energies[obt_energies.size() - i].second);
    }

    //    if (ne % 2) {
    //        high_obt_energy +=
    //        obt_energies[obt_energies.size()-1-Ndocc].first;
    //        high_det.create_alfa_bit(obt_energies[obt_energies.size()-1-Ndocc].second);
    //    }
    lambda_h_ = high_obt_energy + as_ints_->frozen_core_energy() + as_ints_->scalar_energy();

    double lambda_h_G = as_ints_->energy(high_det) + as_ints_->scalar_energy();
    std::vector<int> aocc = high_det.get_alfa_occ(nact_);
    std::vector<int> bocc = high_det.get_beta_occ(nact_);
    std::vector<int> avir = high_det.get_alfa_vir(nact_);
    std::vector<int> bvir = high_det.get_beta_vir(nact_);
    std::vector<int> aocc_offset(nirrep_ + 1);
    std::vector<int> bocc_offset(nirrep_ + 1);
    std::vector<int> avir_offset(nirrep_ + 1);
    std::vector<int> bvir_offset(nirrep_ + 1);

    int noalpha = aocc.size();
    int nobeta = bocc.size();
    int nvalpha = avir.size();
    int nvbeta = bvir.size();

    for (int i = 0; i < noalpha; ++i)
        aocc_offset[mo_symmetry_[aocc[i]] + 1] += 1;
    for (int a = 0; a < nvalpha; ++a)
        avir_offset[mo_symmetry_[avir[a]] + 1] += 1;
    for (int i = 0; i < nobeta; ++i)
        bocc_offset[mo_symmetry_[bocc[i]] + 1] += 1;
    for (int a = 0; a < nvbeta; ++a)
        bvir_offset[mo_symmetry_[bvir[a]] + 1] += 1;
    for (int h = 1; h < nirrep_ + 1; ++h) {
        aocc_offset[h] += aocc_offset[h - 1];
        avir_offset[h] += avir_offset[h - 1];
        bocc_offset[h] += bocc_offset[h - 1];
        bvir_offset[h] += bvir_offset[h - 1];
    }

    // Generate aa excitations
    for (int h = 0; h < nirrep_; ++h) {
        for (int i = aocc_offset[h]; i < aocc_offset[h + 1]; ++i) {
            int ii = aocc[i];
            for (int a = avir_offset[h]; a < avir_offset[h + 1]; ++a) {
                int aa = avir[a];
                double HJI = as_ints_->slater_rules_single_alpha(high_det, ii, aa);
                lambda_h_G += std::fabs(HJI);
            }
        }
    }
    // Generate bb excitations
    for (int h = 0; h < nirrep_; ++h) {
        for (int i = bocc_offset[h]; i < bocc_offset[h + 1]; ++i) {
            int ii = bocc[i];
            for (int a = bvir_offset[h]; a < bvir_offset[h + 1]; ++a) {
                int aa = bvir[a];
                double HJI = as_ints_->slater_rules_single_beta(high_det, ii, aa);
                lambda_h_G += std::fabs(HJI);
            }
        }
    }

    for (int i = 0; i < noalpha; ++i) {
        int ii = aocc[i];
        for (int j = i + 1; j < noalpha; ++j) {
            int jj = aocc[j];
            for (int a = 0; a < nvalpha; ++a) {
                int aa = avir[a];
                int h = mo_symmetry_[ii] ^ mo_symmetry_[jj] ^ mo_symmetry_[aa];
                if (h < mo_symmetry_[aa])
                    continue;
                int minb = h == mo_symmetry_[aa] ? a + 1 : avir_offset[h];
                int maxb = avir_offset[h + 1];
                for (int b = minb; b < maxb; ++b) {
                    int bb = avir[b];
                    double HJI = as_ints_->tei_aa(ii, jj, aa, bb);
                    lambda_h_G += std::fabs(HJI);
                }
            }
        }
    }

    for (int i = 0; i < noalpha; ++i) {
        int ii = aocc[i];
        for (int j = 0; j < nobeta; ++j) {
            int jj = bocc[j];
            for (int a = 0; a < nvalpha; ++a) {
                int aa = avir[a];
                int h = mo_symmetry_[ii] ^ mo_symmetry_[jj] ^ mo_symmetry_[aa];
                int minb = bvir_offset[h];
                int maxb = bvir_offset[h + 1];
                for (int b = minb; b < maxb; ++b) {
                    int bb = bvir[b];
                    double HJI = as_ints_->tei_ab(ii, jj, aa, bb);
                    lambda_h_G += std::fabs(HJI);
                }
            }
        }
    }
    for (int i = 0; i < nobeta; ++i) {
        int ii = bocc[i];
        for (int j = i + 1; j < nobeta; ++j) {
            int jj = bocc[j];
            for (int a = 0; a < nvbeta; ++a) {
                int aa = bvir[a];
                int h = mo_symmetry_[ii] ^ mo_symmetry_[jj] ^ mo_symmetry_[aa];
                if (h < mo_symmetry_[aa])
                    continue;
                int minb = h == mo_symmetry_[aa] ? a + 1 : bvir_offset[h];
                int maxb = bvir_offset[h + 1];
                for (int b = minb; b < maxb; ++b) {
                    int bb = bvir[b];
                    double HJI = as_ints_->tei_bb(ii, jj, aa, bb);
                    lambda_h_G += std::fabs(HJI);
                }
            }
        }
    }
    outfile->Printf("\n\n  ==> Estimate highest excitation energy <==");
    outfile->Printf("\n  Highest Excited determinant:");
    outfile->Printf("\n  %s", high_det.str().c_str());
    outfile->Printf("\n  Determinant Energy                    :  %.12f",
                    as_ints_->energy(high_det) + nuclear_repulsion_energy_ +
                        as_ints_->scalar_energy());
    outfile->Printf("\n  Highest Energy Gershgorin circle Est. :  %.12f",
                    lambda_h_G + nuclear_repulsion_energy_);
    lambda_h_ = lambda_h_G;
    return lambda_h_;
}

void ProjectorCI_HashVec::convergence_analysis() {
    estimate_high_energy();
    compute_characteristic_function();
    print_characteristic_function();
}

void ProjectorCI_HashVec::compute_characteristic_function() {
    shift_ = (lambda_h_ + lambda_1_) / 2.0;
    range_ = (lambda_h_ - lambda_1_) / 2.0;
    switch (generator_) {
    case WallChebyshevGenerator:
        Wall_Chebyshev_generator_coefs(cha_func_coefs_, chebyshev_order_, range_);
    default:
        break;
    }
}

void ProjectorCI_HashVec::print_characteristic_function() {
    outfile->Printf("\n\n  ==> Characteristic Function <==");
    print_polynomial(cha_func_coefs_);
    outfile->Printf("\n    with tau = %e, shift = %.12f, range = %.12f", time_step_, shift_,
                    range_);
    outfile->Printf("\n    Initial guess: lambda_1= %s%.12f", lambda_1_ >= 0.0 ? " " : "",
                    lambda_1_ + nuclear_repulsion_energy_);
    outfile->Printf("\n    Est. Highest eigenvalue= %s%.12f", lambda_h_ >= 0.0 ? " " : "",
                    lambda_h_ + nuclear_repulsion_energy_);
}

double ProjectorCI_HashVec::compute_energy() {
    timer_on("PCI:Energy");
    local_timer t_apici;

    // Increase the root counter (ground state = 0)
    current_root_ += 1;
    lastLow = 0.0;
    previous_go_up = false;

    outfile->Printf("\n\n\t  ---------------------------------------------------------");
    outfile->Printf("\n\t    Projector Configuration Interaction HashVector "
                    "implementation");
    outfile->Printf("\n\t         by Francesco A. Evangelista and Tianyuan Zhang");
    outfile->Printf("\n\t                      version Jul. 28 2017");
    outfile->Printf("\n\t                    %4d thread(s) %s", num_threads_,
                    have_omp_ ? "(OMP)" : "");
    outfile->Printf("\n\t  ---------------------------------------------------------");

    // Print a summary of the options
    print_info();

    /// A vector of determinants in the P space
    det_hashvec dets_hashvec;
    std::vector<double> C;

    SparseCISolver sparse_solver(as_ints_);
    sparse_solver.set_parallel(true);
    sparse_solver.set_e_convergence(options_->get_double("E_CONVERGENCE"));
    sparse_solver.set_maxiter_davidson(options_->get_int("DL_MAXITER"));
    sparse_solver.set_spin_project(true);

    pqpq_aa_ = new double[nact_ * nact_];
    pqpq_ab_ = new double[nact_ * nact_];
    pqpq_bb_ = new double[nact_ * nact_];

    for (size_t i = 0; i < (size_t)nact_; ++i) {
        for (size_t j = 0; j < (size_t)nact_; ++j) {
            double temp_aa = sqrt(std::fabs(as_ints_->tei_aa(i, j, i, j)));
            pqpq_aa_[i * nact_ + j] = temp_aa;
            if (temp_aa > pqpq_max_aa_)
                pqpq_max_aa_ = temp_aa;
            double temp_ab = sqrt(std::fabs(as_ints_->tei_ab(i, j, i, j)));
            pqpq_ab_[i * nact_ + j] = temp_ab;
            if (temp_ab > pqpq_max_ab_)
                pqpq_max_ab_ = temp_ab;
            double temp_bb = sqrt(std::fabs(as_ints_->tei_bb(i, j, i, j)));
            pqpq_bb_[i * nact_ + j] = temp_bb;
            if (temp_bb > pqpq_max_bb_)
                pqpq_max_bb_ = temp_bb;
        }
    }

    timer_on("PCI:Couplings");
    compute_single_couplings(spawning_threshold_);
    compute_double_couplings(spawning_threshold_);
    timer_off("PCI:Couplings");

    // Compute the initial guess
    outfile->Printf("\n\n  ==> Initial Guess <==");
    approx_E_flag_ = true;
    double var_energy = initial_guess(dets_hashvec, C);
    double proj_energy = var_energy;

    timer_on("PCI:sort");
    sortHashVecByCoefficient(dets_hashvec, C);
    timer_off("PCI:sort");

    print_wfn(dets_hashvec, C);
    //    det_hash<> old_space_map;
    //    for (size_t I = 0; I < dets_hashvec.size(); ++I) {
    //        old_space_map[dets_hashvec[I]] = C[I];
    //    }

    convergence_analysis();

    //    for (Determinant det : dets) {
    //        count_hash(det);
    //    }

    // Main iterations
    outfile->Printf("\n\n  ==> PCI Iterations <==");
    if (variational_estimate_) {
        outfile->Printf("\n\n  "
                        "------------------------------------------------------"
                        "------------------------------------------------------"
                        "----------------------------------");
        outfile->Printf("\n    Steps  Beta/Eh      Ndets      NoffDiag     Proj. Energy/Eh   "
                        "  dEp/dt      Var. Energy/Eh      dEp/dt      Approx. "
                        "Energy/Eh   dEv/dt");
        outfile->Printf("\n  "
                        "------------------------------------------------------"
                        "------------------------------------------------------"
                        "----------------------------------");
    } else {
        outfile->Printf("\n\n  "
                        "------------------------------------------------------"
                        "--------------------------------------------------------");
        outfile->Printf("\n    Steps  Beta/Eh      Ndets      NoffDiag     Proj. Energy/Eh   "
                        "  dEp/dt      Approx. Energy/Eh   dEv/dt");
        outfile->Printf("\n  "
                        "------------------------------------------------------"
                        "--------------------------------------------------------");
    }

    int maxcycle = maxiter_;
    double old_var_energy = var_energy;
    double old_proj_energy = proj_energy;
    double beta = 0.0;
    bool converged = false;

    approx_E_flag_ = true;

    for (int cycle = 0; cycle < maxcycle; ++cycle) {
        iter_ = cycle;

        timer_on("PCI:Step");
        if (use_inter_norm_) {
            auto minmax_C = std::minmax_element(C.begin(), C.end());
            double min_C_abs = std::fabs(*minmax_C.first);
            double max_C = *minmax_C.second;
            max_C = max_C > min_C_abs ? max_C : min_C_abs;
            propagate(generator_, dets_hashvec, C, spawning_threshold_ * max_C);
        } else {
            propagate(generator_, dets_hashvec, C, spawning_threshold_);
        }
        timer_off("PCI:Step");

        // Orthogonalize this solution with respect to the previous ones
        timer_on("PCI:Ortho");
        if (current_root_ > 0) {
            orthogonalize(dets_hashvec, C, solutions_);
            normalize(C);
        }
        timer_off("PCI:Ortho");

        // Compute the energy and check for convergence
        if (cycle % energy_estimate_freq_ == 0) {
            approx_E_flag_ = true;
            timer_on("PCI:<E>");
            std::map<std::string, double> results = estimate_energy(dets_hashvec, C);
            timer_off("PCI:<E>");

            proj_energy = results["PROJECTIVE ENERGY"];

            double proj_energy_gradient =
                (proj_energy - old_proj_energy) / (time_step_ * energy_estimate_freq_);
            double approx_energy_gradient =
                (approx_energy_ - old_approx_energy_) / (time_step_ * energy_estimate_freq_);
            if (cycle == 0)
                approx_energy_gradient = 10.0 * e_convergence_ + 1.0;

            switch (generator_) {
            case DLGenerator:
                outfile->Printf("\n%9d %8d %10zu %13zu %20.12f %10.3e", cycle,
                                current_davidson_iter_, C.size(), num_off_diag_elem_, proj_energy,
                                proj_energy_gradient);
                break;
            default:
                outfile->Printf("\n%9d %8.2f %10zu %13zu %20.12f %10.3e", cycle, beta, C.size(),
                                num_off_diag_elem_, proj_energy, proj_energy_gradient);
                break;
            }

            if (variational_estimate_) {
                var_energy = results["VARIATIONAL ENERGY"];
                double var_energy_gradient =
                    (var_energy - old_var_energy) / (time_step_ * energy_estimate_freq_);
                outfile->Printf(" %20.12f %10.3e", var_energy, var_energy_gradient);
            }

            old_var_energy = var_energy;
            old_proj_energy = proj_energy;

            iter_Evar_steps_.push_back(std::make_pair(iter_, var_energy));

            if (std::fabs(approx_energy_gradient) < e_convergence_ && cycle > 1) {
                converged = true;
                break;
            }
            if (converge_test()) {
                break;
            }
            if (do_shift_) {
                lambda_1_ = approx_energy_ - nuclear_repulsion_energy_;
                compute_characteristic_function();
            }
        }
        beta += time_step_;
    }

    if (variational_estimate_) {
        outfile->Printf("\n  "
                        "------------------------------------------------------"
                        "------------------------------------------------------"
                        "----------------------------------");
    } else {
        outfile->Printf("\n  "
                        "------------------------------------------------------"
                        "--------------------------------------------------------");
    }

    if (converged) {
        outfile->Printf("\n\n  Calculation converged.");
    } else {
        outfile->Printf("\n\n  Calculation %s",
                        iter_ != maxiter_ ? "stoped in appearance of higher new low."
                                          : "did not converge!");
    }

    if (do_shift_) {
        outfile->Printf("\n\n  Shift applied during iteration, the "
                        "characteristic function may change every step.\n  "
                        "Characteristic function at last step:");
        print_characteristic_function();
    }

    outfile->Printf("\n\n  ==> Post-Iterations <==\n");
    outfile->Printf("\n  * Size of CI space                    = %zu", C.size());
    outfile->Printf("\n  * Number of off-diagonal elements     = %zu", num_off_diag_elem_);
    outfile->Printf("\n  * Projector-CI Approximate Energy     = %18.12f Eh", 1, approx_energy_);
    outfile->Printf("\n  * Projector-CI Projective  Energy     = %18.12f Eh", 1, proj_energy);

    timer_on("PCI:sort");
    sortHashVecByCoefficient(dets_hashvec, C);
    timer_off("PCI:sort");

    if (print_full_wavefunction_) {
        print_wfn(dets_hashvec, C, C.size());
    } else {
        print_wfn(dets_hashvec, C);
    }

    outfile->Printf("\n  %s: %f s\n", "Projector-CI (bitset) steps finished in  ", t_apici.get());

    timer_on("PCI:<E>end_v");
    if (fast_variational_estimate_) {
        var_energy = estimate_var_energy_sparse(dets_hashvec, C, evar_max_error_);
    } else {
        var_energy = estimate_var_energy_within_error_sigma(dets_hashvec, C, evar_max_error_);
    }
    timer_off("PCI:<E>end_v");

    psi::Process::environment.globals["PCI ENERGY"] = var_energy;

    outfile->Printf("\n  * Projector-CI Variational Energy     = %18.12f Eh", 1, var_energy);
    outfile->Printf("\n  * Projector-CI Var. Corr.  Energy     = %18.12f Eh", 1,
                    var_energy - as_ints_->energy(reference_determinant_) -
                        nuclear_repulsion_energy_ - as_ints_->scalar_energy());

    outfile->Printf("\n  * 1st order perturbation   Energy     = %18.12f Eh", 1,
                    var_energy - approx_energy_);

    outfile->Printf("\n\n  %s: %f s", "Projector-CI (bitset) ran in  ", t_apici.get());

    if (current_root_ < nroot_ - 1) {
        save_wfn(dets_hashvec, C, solutions_);
    }

    if (post_diagonalization_) {
        outfile->Printf("\n\n  ==> Post-Diagonalization <==\n");
        timer_on("PCI:Post_Diag");
        psi::SharedMatrix apfci_evecs(new psi::Matrix("Eigenvectors", C.size(), nroot_));
        psi::SharedVector apfci_evals(new Vector("Eigenvalues", nroot_));

        WFNOperator op(mo_symmetry_, as_ints_);
        DeterminantHashVec det_map(std::move(dets_hashvec));
        op.build_strings(det_map);
        op.op_s_lists(det_map);
        op.tp_s_lists(det_map);

        // set options
        sparse_solver.set_sigma_method("SPARSE");
        sparse_solver.set_e_convergence(e_convergence_);
        sparse_solver.set_spin_project(true);
        sparse_solver.set_spin_project_full(false);

        sparse_solver.diagonalize_hamiltonian_map(det_map, op, apfci_evals, apfci_evecs, nroot_,
                                                  wavefunction_multiplicity_, diag_method_);
        det_map.swap(dets_hashvec);

        timer_off("PCI:Post_Diag");

        double post_diag_energy =
            apfci_evals->get(current_root_) + nuclear_repulsion_energy_ + as_ints_->scalar_energy();
        psi::Process::environment.globals["PCI POST DIAG ENERGY"] = post_diag_energy;

        outfile->Printf("\n\n  * Projector-CI Post-diag   Energy     = %18.12f Eh", 1,
                        post_diag_energy);
        outfile->Printf("\n  * Projector-CI Var. Corr.  Energy     = %18.12f Eh", 1,
                        post_diag_energy - as_ints_->energy(reference_determinant_) -
                            nuclear_repulsion_energy_ - as_ints_->scalar_energy());

        std::vector<double> diag_C(C.size());

        for (size_t I = 0; I < C.size(); ++I) {
            diag_C[I] = apfci_evecs->get(I, current_root_);
        }

        timer_on("PCI:sort");
        sortHashVecByCoefficient(dets_hashvec, diag_C);
        timer_off("PCI:sort");

        if (print_full_wavefunction_) {
            print_wfn(dets_hashvec, diag_C, diag_C.size());
        } else {
            print_wfn(dets_hashvec, diag_C);
        }
    }

    delete[] pqpq_aa_;
    delete[] pqpq_ab_;
    delete[] pqpq_bb_;
    energies_.push_back(var_energy);

    timer_off("PCI:Energy");
    return var_energy;
}

bool ProjectorCI_HashVec::converge_test() {
    if (!stop_higher_new_low_) {
        return false;
    }
    if (approx_energy_ > old_approx_energy_ && !previous_go_up) {
        if (old_approx_energy_ > lastLow) {
            lastLow = old_approx_energy_;
            return true;
        }
        lastLow = old_approx_energy_;
        previous_go_up = true;
    }
    if (approx_energy_ < old_approx_energy_) {
        previous_go_up = false;
    }
    return false;
}

double ProjectorCI_HashVec::initial_guess(det_hashvec& dets_hashvec, std::vector<double>& C) {

    // Do one time step starting from the reference determinant
    Determinant bs_det(reference_determinant_);
    dets_hashvec.clear();
    dets_hashvec.add(bs_det);
    dets_max_couplings_.resize(dets_hashvec.size());

    apply_tau_H_symm(time_step_, initial_guess_spawning_threshold_, dets_hashvec, {1.0}, C, 0.0);

    size_t guess_size = dets_hashvec.size();
    if (guess_size > max_guess_size_) {
        // Consider the 1000 largest contributions
        std::vector<std::pair<double, size_t>> det_weight;
        for (size_t I = 0, max_I = C.size(); I < max_I; ++I) {
            det_weight.push_back(std::make_pair(std::fabs(C[I]), I));
            // dets[I].print();
        }
        std::sort(det_weight.begin(), det_weight.end());
        std::reverse(det_weight.begin(), det_weight.end());

        //        det_vec new_dets;
        det_hashvec new_dets;
        std::vector<std::pair<double, double>> new_dets_max_couplings;
        new_dets_max_couplings.reserve(max_guess_size_);
        for (size_t sI = 0; sI < max_guess_size_; ++sI) {
            size_t I = det_weight[sI].second;
            new_dets.add(dets_hashvec[I]);
            new_dets_max_couplings.push_back(dets_max_couplings_[I]);
        }
        dets_hashvec.swap(new_dets);
        dets_max_couplings_.swap(new_dets_max_couplings);
        guess_size = dets_hashvec.size();
        C.resize(guess_size);
    }

    outfile->Printf("\n\n  Initial guess size = %zu", guess_size);

    SparseCISolver sparse_solver(as_ints_);
    sparse_solver.set_parallel(true);
    sparse_solver.set_e_convergence(options_->get_double("E_CONVERGENCE"));
    sparse_solver.set_maxiter_davidson(options_->get_int("DL_MAXITER"));
    sparse_solver.set_spin_project(true);

    psi::SharedMatrix evecs(new psi::Matrix("Eigenvectors", guess_size, nroot_));
    psi::SharedVector evals(new Vector("Eigenvalues", nroot_));
    //  std::vector<DynamicBitsetDeterminant> dyn_dets;
    // for (auto& d : dets){
    //   DynamicBitsetDeterminant dbs = d.to_dynamic_bitset();
    //  dyn_dets.push_back(dbs);
    // }
    sparse_solver.diagonalize_hamiltonian(dets_hashvec.toVector(), evals, evecs, nroot_,
                                          wavefunction_multiplicity_, DLSolver);
    double var_energy =
        evals->get(current_root_) + nuclear_repulsion_energy_ + as_ints_->scalar_energy();
    outfile->Printf("\n\n  Initial guess energy (variational) = %20.12f Eh (root = %d)", var_energy,
                    current_root_ + 1);

    lambda_1_ = evals->get(current_root_) + as_ints_->scalar_energy();

    // Copy the ground state eigenvector
    for (size_t I = 0; I < guess_size; ++I) {
        C[I] = evecs->get(I, current_root_);
    }
    //    outfile->Printf("\n\n  Reached here");
    return var_energy;
}

void ProjectorCI_HashVec::propagate(GeneratorType generator, det_hashvec& dets_hashvec,
                                    std::vector<double>& C, double spawning_threshold) {
    //    det_hashvec dets_hashvec(dets);
    //    det_vec dets;
    switch (generator) {
    case WallChebyshevGenerator:
        propagate_wallCh(dets_hashvec, C, spawning_threshold);
        break;
    case DLGenerator:
        //        dets = dets_hashvec.toVector();
        propagate_DL(dets_hashvec, C, spawning_threshold);
        //        dets_hashvec = det_hashvec(dets);
        break;
    default:
        outfile->Printf("\n\n  Selected Generator Unsupported in HashVector version!!!");
        abort();
        break;
    }
    normalize(C);
    //    dets = dets_hashvec.toVector();
}

void ProjectorCI_HashVec::propagate_wallCh(det_hashvec& dets_hashvec, std::vector<double>& C,
                                           double spawning_threshold) {
    //    det_hashvec dets_hashvec(dets);
    // A map that contains the pair (determinant,coefficient)
    const double PI = 2 * acos(0.0);
    //    det_hash<> dets_C_hash;
    const std::vector<double> ref_C(C);

    double root = -cos(((double)chebyshev_order_) * PI / (chebyshev_order_ + 0.5));
    apply_tau_H_symm(-1.0, spawning_threshold, dets_hashvec, ref_C, C, range_ * root + shift_);
    normalize(C);

    for (int i = chebyshev_order_ - 1; i > 0; i--) {
        //        outfile->Printf("\nCurrent root:%.12lf",range_ * root +
        //        shift_);
        //        apply_tau_H(-1.0/range_,spawning_threshold,dets,C,dets_C_hash,
        //        range_ * root + shift_);
        double root = -cos(((double)i) * PI / (chebyshev_order_ + 0.5));
        //        dets = dets_hashvec.toVector();
        std::vector<double> result_C;
        apply_tau_H_ref_C_symm(-1.0, spawning_threshold, dets_hashvec, ref_C, C, result_C,
                               range_ * root + shift_);
        C.swap(result_C);
        //        copy_hash_to_vec_order_ref(dets_C_hash, dets, C);
        //        dets_hashvec = det_hashvec(dets_C_hash, C);
        //        dets = dets_hashvec.toVector();
        //        dets_hashvec = det_hashvec(dets);

        //        dets_C_hash.clear();
        normalize(C);
    }
    //    dets = dets_hashvec.toVector();
}

void ProjectorCI_HashVec::propagate_DL(det_hashvec& dets_hashvec, std::vector<double>& C,
                                       double spawning_threshold) {
    size_t ref_size = C.size();
    std::vector<std::vector<double>> b_vec(davidson_subspace_per_root_);
    std::vector<std::vector<double>> sigma_vec(davidson_subspace_per_root_);
    std::vector<double> alpha_vec(davidson_subspace_per_root_);
    psi::SharedMatrix A(new psi::Matrix(davidson_subspace_per_root_, davidson_subspace_per_root_));
    //    det_hash<> dets_C_hash;
    //    apply_tau_H_ref_C_symm(1.0, spawning_threshold, dets, b_vec[0], C,
    //                           dets_C_hash, 0.0);
    //    copy_hash_to_vec_order_ref(dets_C_hash, dets, sigma_vec[0]);
    //    det_hashvec dets_hashvec(dets);
    apply_tau_H_symm(1.0, spawning_threshold, dets_hashvec, C, sigma_vec[0], 0.0);
    //    dets = dets_hashvec.toVector();
    if (ref_size <= 1) {
        C = sigma_vec[0];
        outfile->Printf("\nDavidson break because the reference space have "
                        "only 1 determinant.");
        current_davidson_iter_ = 1;
        return;
    }

    size_t dets_size = dets_hashvec.size();
    b_vec[0] = C;
    A->set(0, 0, dot(b_vec[0], sigma_vec[0]));
    b_vec[0].resize(dets_size, 0.0);

    std::vector<double> diag_vec(dets_size);
#pragma omp parallel for
    for (size_t i = 0; i < dets_size; i++) {
        diag_vec[i] = as_ints_->energy(dets_hashvec[i]) + as_ints_->scalar_energy();
    }

    double lambda = A->get(0, 0);
    alpha_vec[0] = 1.0;
    std::vector<double> delta_vec(dets_size, 0.0);
    size_t current_order = 1;

    int i = 1;
    for (i = 1; i < max_Davidson_iter_; i++) {

        for (size_t k = 0; k < current_order; k++) {
#pragma omp parallel for
            for (size_t j = 0; j < dets_size; j++) {
                delta_vec[j] += alpha_vec[k] * (sigma_vec[k][j] - lambda * b_vec[k][j]);
            }
        }
#pragma omp parallel for
        for (size_t j = 0; j < dets_size; j++) {
            delta_vec[j] /= lambda - diag_vec[j];
        }

        normalize(delta_vec);
        for (size_t m = 0; m < current_order; m++) {
            double delta_dot_bm = dot(delta_vec, b_vec[m]);
            add(delta_vec, -delta_dot_bm, b_vec[m]);
        }
        double correct_norm = normalize(delta_vec);
        if (correct_norm < 1e-4) {
            outfile->Printf("\nDavidson break at %d-th iter because the "
                            "correction norm %10.3e is too small.",
                            i, correct_norm);
            break;
        }
        if (correct_norm > 1e1) {
            outfile->Printf("\nDavidson break at %d-th iter because the "
                            "correction norm %10.3e is too large.",
                            i, correct_norm);
            break;
        }
        //        print_vector(delta_vec, "delta_vec");
        b_vec[current_order] = delta_vec;

        //        dets_C_hash.clear();
        //        apply_tau_H_ref_C_symm(1.0, spawning_threshold, dets,
        //                               b_vec[current_order], C, dets_C_hash,
        //                               0.0);
        //        copy_hash_to_vec_order_ref(dets_C_hash, dets,
        //        sigma_vec[current_order]);
        //        dets_hashvec = det_hashvec(dets);
        apply_tau_H_ref_C_symm(1.0, spawning_threshold, dets_hashvec, C, b_vec[current_order],
                               sigma_vec[current_order], 0.0);
        //        dets = dets_hashvec.toVector();
        for (size_t m = 0; m < current_order; m++) {
            double b_dot_sigma_m = dot(b_vec[current_order], sigma_vec[m]);
            A->set(current_order, m, b_dot_sigma_m);
            A->set(m, current_order, b_dot_sigma_m);
        }
        A->set(current_order, current_order, dot(b_vec[current_order], sigma_vec[current_order]));

        current_order++;
        psi::SharedMatrix G(new psi::Matrix(current_order, current_order));

        for (size_t k = 0; k < current_order; k++) {
            for (size_t j = 0; j < current_order; j++) {
                G->set(k, j, A->get(k, j));
            }
        }
        psi::SharedMatrix evecs(new psi::Matrix(current_order, current_order));
        psi::SharedVector eigs(new Vector(current_order));
        G->diagonalize(evecs, eigs);

        double e_gradiant = -lambda;

        lambda = eigs->get(0);
        for (size_t j = 0; j < current_order; j++) {
            alpha_vec[j] = evecs->get(j, 0);
        }
        e_gradiant += lambda;
        outfile->Printf("\nDavidson iter %4d order %4d correction norm %10.3e dE %10.3e E %18.12f.",
                        i, current_order, correct_norm, e_gradiant,
                        lambda + nuclear_repulsion_energy_ + as_ints_->scalar_energy());
        if (std::fabs(e_gradiant) < e_convergence_) {
            i++;
            break;
        }
        if (current_order >= davidson_subspace_per_root_) {
#pragma omp parallel for
            for (size_t j = 0; j < dets_size; j++) {
                std::vector<double> b_j(davidson_collapse_per_root_, 0.0);
                std::vector<double> sigma_j(davidson_collapse_per_root_, 0.0);
                for (size_t l = 0; l < davidson_collapse_per_root_; l++) {
                    for (size_t k = 0; k < current_order; k++) {
                        b_j[l] += evecs->get(k, l) * b_vec[k][j];
                        sigma_j[l] += evecs->get(k, l) * sigma_vec[k][j];
                    }
                }
                for (size_t l = 0; l < davidson_collapse_per_root_; l++) {
                    b_vec[l][j] = b_j[l];
                    sigma_vec[l][j] = sigma_j[l];
                }
            }
            for (size_t l = davidson_collapse_per_root_; l < davidson_subspace_per_root_; l++) {
                b_vec[l].clear();
                sigma_vec[l].clear();
            }
            for (size_t m = 0; m < davidson_collapse_per_root_; m++) {
                for (size_t n = 0; n <= m; n++) {
                    double n_dot_sigma_m = dot(b_vec[n], sigma_vec[m]);
                    A->set(n, m, n_dot_sigma_m);
                    A->set(m, n, n_dot_sigma_m);
                }
            }
            alpha_vec[0] = 1.0;
            for (size_t l = 1; l < davidson_subspace_per_root_; l++) {
                alpha_vec[l] = 0.0;
            }
            outfile->Printf("\nDavidson collapsed from %d vectors to %d vectors.", current_order,
                            davidson_collapse_per_root_);
            current_order = davidson_collapse_per_root_;
        }
    }

    //    for (int i = 0; i < krylov_order_; i++) {
    //        print_vector(b_vec[i], "b_vec["+std::to_string(i)+"]");
    //    }

    current_davidson_iter_ = i;

    //    scale(C, alpha_vec[0]);
    //    C.clear();
    C = b_vec[0];
    scale(C, alpha_vec[0]);
    C.resize(dets_hashvec.size(), 0.0);
    //    b_vec[0].resize(dets.size(), 0.0);
    for (size_t i = 1; i < current_order; i++) {
#pragma omp parallel for
        for (size_t j = 0; j < dets_size; j++) {
            C[j] += alpha_vec[i] * b_vec[i][j];
        }
    }
    //    dets = dets_hashvec.toVector();

    //    std::vector<double> C2;
    //    C2.resize(dets.size(), 0.0);
    //    for (int i = 0; i < current_order; i++) {
    //        for (int j = 0; j < b_vec[i].size(); j++) {
    //            C2[j] += alpha_vec[i] * b_vec[i][j];
    //        }
    //    }
    //    add(C2, -1.0, C);
    //    outfile->Printf("\nC2 norm %10.3e", norm(C2));
}

void ProjectorCI_HashVec::apply_tau_H_symm(double tau, double spawning_threshold,
                                           det_hashvec& dets_hashvec, const std::vector<double>& C,
                                           std::vector<double>& result_C, double S) {

    size_t ref_size = dets_hashvec.size();
    result_C.clear();
    result_C.resize(ref_size, 0.0);
    det_hashvec extra_dets;
    std::vector<double> extra_C;

    std::vector<std::vector<std::pair<Determinant, double>>> thread_det_C_vecs(num_threads_);
    num_off_diag_elem_ = 0;

#pragma omp parallel for
    for (size_t I = 0; I < ref_size; ++I) {
        size_t current_rank = omp_get_thread_num();
        std::pair<double, double>& max_coupling = dets_max_couplings_[I];
        thread_det_C_vecs[current_rank].clear();
        apply_tau_H_symm_det_dynamic_HBCI_2(tau, spawning_threshold, dets_hashvec, C, I, C[I],
                                            result_C, thread_det_C_vecs[current_rank], S,
                                            max_coupling);
#pragma omp critical
        {
            merge(extra_dets, extra_C, thread_det_C_vecs[current_rank],
                  std::function<double(double, double)>(std::plus<double>()), 0.0, false);
        }
    }

    dets_hashvec.merge(extra_dets);
    result_C.insert(result_C.end(), extra_C.begin(), extra_C.end());
    dets_max_couplings_.resize(dets_hashvec.size());

    if (approx_E_flag_) {
        timer_on("PCI:<E>a");
        double CHC_energy = 0.0;
#pragma omp parallel for reduction(+ : CHC_energy)
        for (size_t I = 0; I < ref_size; ++I) {
            CHC_energy += C[I] * result_C[I];
        }
        CHC_energy = CHC_energy / tau + S + nuclear_repulsion_energy_;
        timer_off("PCI:<E>a");
        double CHC_energy_gradient =
            (CHC_energy - approx_energy_) / (time_step_ * energy_estimate_freq_);
        old_approx_energy_ = approx_energy_;
        approx_energy_ = CHC_energy;
        approx_E_flag_ = false;
        approx_E_tau_ = tau;
        approx_E_S_ = S;
        if (iter_ != 0)
            outfile->Printf(" %20.12f %10.3e", approx_energy_, CHC_energy_gradient);
    }
}

void ProjectorCI_HashVec::apply_tau_H_symm_det_dynamic_HBCI_2(
    double tau, double spawning_threshold, const det_hashvec& dets_hashvec,
    const std::vector<double>& pre_C, size_t I, double CI, std::vector<double>& result_C,
    std::vector<std::pair<Determinant, double>>& new_det_C_vec, double E0,
    std::pair<double, double>& max_coupling) {

    const Determinant& detI = dets_hashvec[I];
    size_t pre_C_size = pre_C.size();

    bool do_singles_1 = max_coupling.first == 0.0 and
                        std::fabs(dets_single_max_coupling_ * CI) >= spawning_threshold;
    bool do_singles = std::fabs(max_coupling.first * CI) >= spawning_threshold;
    bool do_doubles_1 = max_coupling.second == 0.0 and
                        std::fabs(dets_double_max_coupling_ * CI) >= spawning_threshold;
    bool do_doubles = std::fabs(max_coupling.second * CI) >= spawning_threshold;

    // Diagonal contributions
    // parallel_timer_on("PCI:diagonal", omp_get_thread_num());

    double det_energy = as_ints_->energy(detI) + as_ints_->scalar_energy();
#pragma omp atomic
    result_C[I] += tau * (det_energy - E0) * CI;

    // parallel_timer_off("PCI:diagonal", omp_get_thread_num());

    Determinant detJ(detI);
    if (do_singles) {
        // parallel_timer_on("PCI:singles", omp_get_thread_num());
        // Generate alpha excitations
        for (size_t x = 0; x < a_couplings_size_; ++x) {
            double HJI_max = std::get<1>(a_couplings_[x]);
            if (std::fabs(HJI_max * CI) < spawning_threshold) {
                break;
            }
            int i = std::get<0>(a_couplings_[x]);
            if (detI.get_alfa_bit(i)) {
                std::vector<std::tuple<int, double>>& sub_couplings = std::get<2>(a_couplings_[x]);
                size_t sub_couplings_size = sub_couplings.size();
                for (size_t y = 0; y < sub_couplings_size; ++y) {
                    int a;
                    double HJI_bound;
                    std::tie(a, HJI_bound) = sub_couplings[y];
                    if (std::fabs(HJI_bound * CI) < spawning_threshold) {
                        break;
                    }
                    if (!detI.get_alfa_bit(a)) {
                        Determinant detJ(detI);

                        double HJI = as_ints_->slater_rules_single_alpha_abs(detJ, i, a);

                        if (std::fabs(HJI * CI) >= spawning_threshold) {
                            HJI *= detJ.single_excitation_a(i, a);

                            size_t index = dets_hashvec.find(detJ);
                            if (index >= pre_C_size) {
                                new_det_C_vec.push_back(std::make_pair(detJ, tau * HJI * CI));
#pragma omp atomic
                                num_off_diag_elem_ += 2;
                            } else {
#pragma omp atomic
                                result_C[index] += tau * HJI * CI;
#pragma omp atomic
                                ++num_off_diag_elem_;
                                if (std::fabs(HJI * pre_C[index]) < spawning_threshold) {
#pragma omp atomic
                                    result_C[I] += tau * HJI * pre_C[index];
#pragma omp atomic
                                    ++num_off_diag_elem_;
                                }
                            }

                            detJ.set_alfa_bit(i, true);
                            detJ.set_alfa_bit(a, false);
                        }
                    }
                }
            }
        }
        // Generate beta excitations
        for (size_t x = 0; x < b_couplings_size_; ++x) {
            double HJI_max = std::get<1>(b_couplings_[x]);
            if (std::fabs(HJI_max * CI) < spawning_threshold) {
                break;
            }
            int i = std::get<0>(b_couplings_[x]);
            if (detI.get_beta_bit(i)) {
                std::vector<std::tuple<int, double>>& sub_couplings = std::get<2>(b_couplings_[x]);
                size_t sub_couplings_size = sub_couplings.size();
                for (size_t y = 0; y < sub_couplings_size; ++y) {
                    int a;
                    double HJI_bound;
                    std::tie(a, HJI_bound) = sub_couplings[y];
                    if (std::fabs(HJI_bound * CI) < spawning_threshold) {
                        break;
                    }
                    if (!detI.get_beta_bit(a)) {
                        Determinant detJ(detI);

                        double HJI = as_ints_->slater_rules_single_beta_abs(detJ, i, a);

                        if (std::fabs(HJI * CI) >= spawning_threshold) {
                            HJI *= detJ.single_excitation_b(i, a);

                            size_t index = dets_hashvec.find(detJ);
                            if (index >= pre_C_size) {
                                new_det_C_vec.push_back(std::make_pair(detJ, tau * HJI * CI));
#pragma omp atomic
                                num_off_diag_elem_ += 2;
                            } else {
#pragma omp atomic
                                result_C[index] += tau * HJI * CI;
#pragma omp atomic
                                ++num_off_diag_elem_;
                                if (std::fabs(HJI * pre_C[index]) < spawning_threshold) {
#pragma omp atomic
                                    result_C[I] += tau * HJI * pre_C[index];
#pragma omp atomic
                                    ++num_off_diag_elem_;
                                }
                            }

                            detJ.set_beta_bit(i, true);
                            detJ.set_beta_bit(a, false);
                        }
                    }
                }
            }
        }
        // parallel_timer_off("PCI:singles", omp_get_thread_num());
    } else if (do_singles_1) {
        // parallel_timer_on("PCI:singles", omp_get_thread_num());
        // Generate alpha excitations
        for (size_t x = 0; x < a_couplings_size_; ++x) {
            double HJI_max = std::get<1>(a_couplings_[x]);
            if (std::fabs(HJI_max * CI) < spawning_threshold) {
                break;
            }
            int i = std::get<0>(a_couplings_[x]);
            if (detI.get_alfa_bit(i)) {
                std::vector<std::tuple<int, double>>& sub_couplings = std::get<2>(a_couplings_[x]);
                size_t sub_couplings_size = sub_couplings.size();
                for (size_t y = 0; y < sub_couplings_size; ++y) {
                    int a;
                    double HJI_bound;
                    std::tie(a, HJI_bound) = sub_couplings[y];
                    if (std::fabs(HJI_bound * CI) < spawning_threshold) {
                        break;
                    }
                    if (!detI.get_alfa_bit(a)) {
                        Determinant detJ(detI);

                        double HJI = as_ints_->slater_rules_single_alpha_abs(detJ, i, a);

                        max_coupling.first = std::max(max_coupling.first, std::fabs(HJI));
                        if (std::fabs(HJI * CI) >= spawning_threshold) {
                            HJI *= detJ.single_excitation_a(i, a);

                            size_t index = dets_hashvec.find(detJ);
                            if (index >= pre_C_size) {
                                new_det_C_vec.push_back(std::make_pair(detJ, tau * HJI * CI));
#pragma omp atomic
                                num_off_diag_elem_ += 2;
                            } else {
#pragma omp atomic
                                result_C[index] += tau * HJI * CI;
#pragma omp atomic
                                ++num_off_diag_elem_;
                                if (std::fabs(HJI * pre_C[index]) < spawning_threshold) {
#pragma omp atomic
                                    result_C[I] += tau * HJI * pre_C[index];
#pragma omp atomic
                                    ++num_off_diag_elem_;
                                }
                            }

                            detJ.set_alfa_bit(i, true);
                            detJ.set_alfa_bit(a, false);
                        }
                    }
                }
            }
        }
        // Generate beta excitations
        for (size_t x = 0; x < b_couplings_size_; ++x) {
            double HJI_max = std::get<1>(b_couplings_[x]);
            if (std::fabs(HJI_max * CI) < spawning_threshold) {
                break;
            }
            int i = std::get<0>(b_couplings_[x]);
            if (detI.get_beta_bit(i)) {
                std::vector<std::tuple<int, double>>& sub_couplings = std::get<2>(b_couplings_[x]);
                size_t sub_couplings_size = sub_couplings.size();
                for (size_t y = 0; y < sub_couplings_size; ++y) {
                    int a;
                    double HJI_bound;
                    std::tie(a, HJI_bound) = sub_couplings[y];
                    if (std::fabs(HJI_bound * CI) < spawning_threshold) {
                        break;
                    }
                    if (!detI.get_beta_bit(a)) {
                        Determinant detJ(detI);

                        double HJI = as_ints_->slater_rules_single_beta_abs(detJ, i, a);

                        max_coupling.first = std::max(max_coupling.first, std::fabs(HJI));
                        if (std::fabs(HJI * CI) >= spawning_threshold) {
                            HJI *= detJ.single_excitation_b(i, a);

                            size_t index = dets_hashvec.find(detJ);
                            if (index >= pre_C_size) {
                                new_det_C_vec.push_back(std::make_pair(detJ, tau * HJI * CI));
#pragma omp atomic
                                num_off_diag_elem_ += 2;
                            } else {
#pragma omp atomic
                                result_C[index] += tau * HJI * CI;
#pragma omp atomic
                                ++num_off_diag_elem_;
                                if (std::fabs(HJI * pre_C[index]) < spawning_threshold) {
#pragma omp atomic
                                    result_C[I] += tau * HJI * pre_C[index];
#pragma omp atomic
                                    ++num_off_diag_elem_;
                                }
                            }

                            detJ.set_beta_bit(i, true);
                            detJ.set_beta_bit(a, false);
                        }
                    }
                }
            }
        }
        // parallel_timer_off("PCI:singles", omp_get_thread_num());
    }

    if (do_doubles) {
        // parallel_timer_on("PCI:doubles", omp_get_thread_num());
        // Generate alpha-alpha excitations
        for (size_t x = 0; x < aa_couplings_size_; ++x) {
            double HJI_max = std::get<2>(aa_couplings_[x]);
            if (std::fabs(HJI_max * CI) < spawning_threshold) {
                break;
            }
            int i = std::get<0>(aa_couplings_[x]);
            int j = std::get<1>(aa_couplings_[x]);
            if (detI.get_alfa_bit(i) and detI.get_alfa_bit(j)) {
                std::vector<std::tuple<int, int, double>>& sub_couplings =
                    std::get<3>(aa_couplings_[x]);
                size_t sub_couplings_size = sub_couplings.size();
                for (size_t y = 0; y < sub_couplings_size; ++y) {
                    int a, b;
                    double HJI;
                    std::tie(a, b, HJI) = sub_couplings[y];
                    if (std::fabs(HJI * CI) < spawning_threshold) {
                        break;
                    }
                    if (!(detI.get_alfa_bit(a) or detI.get_alfa_bit(b))) {
                        HJI *= detJ.double_excitation_aa(i, j, a, b);

                        size_t index = dets_hashvec.find(detJ);
                        if (index >= pre_C_size) {
                            new_det_C_vec.push_back(std::make_pair(detJ, tau * HJI * CI));
#pragma omp atomic
                            num_off_diag_elem_ += 2;
                        } else {
#pragma omp atomic
                            result_C[index] += tau * HJI * CI;
#pragma omp atomic
                            ++num_off_diag_elem_;
                            if (std::fabs(HJI * pre_C[index]) < spawning_threshold) {
#pragma omp atomic
                                result_C[I] += tau * HJI * pre_C[index];
#pragma omp atomic
                                ++num_off_diag_elem_;
                            }
                        }

                        detJ.set_alfa_bit(i, true);
                        detJ.set_alfa_bit(j, true);
                        detJ.set_alfa_bit(a, false);
                        detJ.set_alfa_bit(b, false);
                    }
                }
            }
        }
        // Generate alpha-beta excitations
        for (size_t x = 0; x < ab_couplings_size_; ++x) {
            double HJI_max = std::get<2>(ab_couplings_[x]);
            if (std::fabs(HJI_max * CI) < spawning_threshold) {
                break;
            }
            int i = std::get<0>(ab_couplings_[x]);
            int j = std::get<1>(ab_couplings_[x]);
            if (detI.get_alfa_bit(i) and detI.get_beta_bit(j)) {
                std::vector<std::tuple<int, int, double>>& sub_couplings =
                    std::get<3>(ab_couplings_[x]);
                size_t sub_couplings_size = sub_couplings.size();
                for (size_t y = 0; y < sub_couplings_size; ++y) {
                    int a, b;
                    double HJI;
                    std::tie(a, b, HJI) = sub_couplings[y];
                    if (std::fabs(HJI * CI) < spawning_threshold) {
                        break;
                    }
                    if (!(detI.get_alfa_bit(a) or detI.get_beta_bit(b))) {
                        HJI *= detJ.double_excitation_ab(i, j, a, b);

                        size_t index = dets_hashvec.find(detJ);
                        if (index >= pre_C_size) {
                            new_det_C_vec.push_back(std::make_pair(detJ, tau * HJI * CI));
#pragma omp atomic
                            num_off_diag_elem_ += 2;
                        } else {
#pragma omp atomic
                            result_C[index] += tau * HJI * CI;
#pragma omp atomic
                            ++num_off_diag_elem_;
                            if (std::fabs(HJI * pre_C[index]) < spawning_threshold) {
#pragma omp atomic
                                result_C[I] += tau * HJI * pre_C[index];
#pragma omp atomic
                                ++num_off_diag_elem_;
                            }
                        }

                        detJ.set_alfa_bit(i, true);
                        detJ.set_beta_bit(j, true);
                        detJ.set_alfa_bit(a, false);
                        detJ.set_beta_bit(b, false);
                    }
                }
            }
        }
        // Generate beta-beta excitations
        for (size_t x = 0; x < bb_couplings_size_; ++x) {
            double HJI_max = std::get<2>(bb_couplings_[x]);
            if (std::fabs(HJI_max * CI) < spawning_threshold) {
                break;
            }
            int i = std::get<0>(bb_couplings_[x]);
            int j = std::get<1>(bb_couplings_[x]);
            if (detI.get_beta_bit(i) and detI.get_beta_bit(j)) {
                std::vector<std::tuple<int, int, double>>& sub_couplings =
                    std::get<3>(bb_couplings_[x]);
                size_t sub_couplings_size = sub_couplings.size();
                for (size_t y = 0; y < sub_couplings_size; ++y) {
                    int a, b;
                    double HJI;
                    std::tie(a, b, HJI) = sub_couplings[y];
                    if (std::fabs(HJI * CI) < spawning_threshold) {
                        break;
                    }
                    if (!(detI.get_beta_bit(a) or detI.get_beta_bit(b))) {
                        HJI *= detJ.double_excitation_bb(i, j, a, b);

                        size_t index = dets_hashvec.find(detJ);
                        if (index >= pre_C_size) {
                            new_det_C_vec.push_back(std::make_pair(detJ, tau * HJI * CI));
#pragma omp atomic
                            num_off_diag_elem_ += 2;
                        } else {
#pragma omp atomic
                            result_C[index] += tau * HJI * CI;
#pragma omp atomic
                            ++num_off_diag_elem_;
                            if (std::fabs(HJI * pre_C[index]) < spawning_threshold) {
#pragma omp atomic
                                result_C[I] += tau * HJI * pre_C[index];
#pragma omp atomic
                                ++num_off_diag_elem_;
                            }
                        }

                        detJ.set_beta_bit(i, true);
                        detJ.set_beta_bit(j, true);
                        detJ.set_beta_bit(a, false);
                        detJ.set_beta_bit(b, false);
                    }
                }
            }
        }
        // parallel_timer_off("PCI:doubles", omp_get_thread_num());
    } else if (do_doubles_1) {
        // parallel_timer_on("PCI:doubles", omp_get_thread_num());
        // Generate alpha-alpha excitations
        for (size_t x = 0; x < aa_couplings_size_; ++x) {
            double HJI_max = std::get<2>(aa_couplings_[x]);
            if (std::fabs(HJI_max * CI) < spawning_threshold) {
                break;
            }
            int i = std::get<0>(aa_couplings_[x]);
            int j = std::get<1>(aa_couplings_[x]);
            if (detI.get_alfa_bit(i) and detI.get_alfa_bit(j)) {
                std::vector<std::tuple<int, int, double>>& sub_couplings =
                    std::get<3>(aa_couplings_[x]);
                size_t sub_couplings_size = sub_couplings.size();
                for (size_t y = 0; y < sub_couplings_size; ++y) {
                    int a, b;
                    double HJI;
                    std::tie(a, b, HJI) = sub_couplings[y];
                    if (std::fabs(HJI * CI) < spawning_threshold) {
                        break;
                    }
                    if (!(detI.get_alfa_bit(a) or detI.get_alfa_bit(b))) {
                        max_coupling.second = std::max(max_coupling.second, std::fabs(HJI));
                        HJI *= detJ.double_excitation_aa(i, j, a, b);

                        size_t index = dets_hashvec.find(detJ);
                        if (index >= pre_C_size) {
                            new_det_C_vec.push_back(std::make_pair(detJ, tau * HJI * CI));
#pragma omp atomic
                            num_off_diag_elem_ += 2;
                        } else {
#pragma omp atomic
                            result_C[index] += tau * HJI * CI;
#pragma omp atomic
                            ++num_off_diag_elem_;
                            if (std::fabs(HJI * pre_C[index]) < spawning_threshold) {
#pragma omp atomic
                                result_C[I] += tau * HJI * pre_C[index];
#pragma omp atomic
                                ++num_off_diag_elem_;
                            }
                        }

                        detJ.set_alfa_bit(i, true);
                        detJ.set_alfa_bit(j, true);
                        detJ.set_alfa_bit(a, false);
                        detJ.set_alfa_bit(b, false);
                    }
                }
            }
        }
        // Generate alpha-beta excitations
        for (size_t x = 0; x < ab_couplings_size_; ++x) {
            double HJI_max = std::get<2>(ab_couplings_[x]);
            if (std::fabs(HJI_max * CI) < spawning_threshold) {
                break;
            }
            int i = std::get<0>(ab_couplings_[x]);
            int j = std::get<1>(ab_couplings_[x]);
            if (detI.get_alfa_bit(i) and detI.get_beta_bit(j)) {
                std::vector<std::tuple<int, int, double>>& sub_couplings =
                    std::get<3>(ab_couplings_[x]);
                size_t sub_couplings_size = sub_couplings.size();
                for (size_t y = 0; y < sub_couplings_size; ++y) {
                    int a, b;
                    double HJI;
                    std::tie(a, b, HJI) = sub_couplings[y];
                    if (std::fabs(HJI * CI) < spawning_threshold) {
                        break;
                    }
                    if (!(detI.get_alfa_bit(a) or detI.get_beta_bit(b))) {
                        max_coupling.second = std::max(max_coupling.second, std::fabs(HJI));
                        HJI *= detJ.double_excitation_ab(i, j, a, b);

                        size_t index = dets_hashvec.find(detJ);
                        if (index >= pre_C_size) {
                            new_det_C_vec.push_back(std::make_pair(detJ, tau * HJI * CI));
#pragma omp atomic
                            num_off_diag_elem_ += 2;
                        } else {
#pragma omp atomic
                            result_C[index] += tau * HJI * CI;
#pragma omp atomic
                            ++num_off_diag_elem_;
                            if (std::fabs(HJI * pre_C[index]) < spawning_threshold) {
#pragma omp atomic
                                result_C[I] += tau * HJI * pre_C[index];
#pragma omp atomic
                                ++num_off_diag_elem_;
                            }
                        }

                        detJ.set_alfa_bit(i, true);
                        detJ.set_beta_bit(j, true);
                        detJ.set_alfa_bit(a, false);
                        detJ.set_beta_bit(b, false);
                    }
                }
            }
        }
        // Generate beta-beta excitations
        for (size_t x = 0; x < bb_couplings_size_; ++x) {
            double HJI_max = std::get<2>(bb_couplings_[x]);
            if (std::fabs(HJI_max * CI) < spawning_threshold) {
                break;
            }
            int i = std::get<0>(bb_couplings_[x]);
            int j = std::get<1>(bb_couplings_[x]);
            if (detI.get_beta_bit(i) and detI.get_beta_bit(j)) {
                std::vector<std::tuple<int, int, double>>& sub_couplings =
                    std::get<3>(bb_couplings_[x]);
                size_t sub_couplings_size = sub_couplings.size();
                for (size_t y = 0; y < sub_couplings_size; ++y) {
                    int a, b;
                    double HJI;
                    std::tie(a, b, HJI) = sub_couplings[y];
                    if (std::fabs(HJI * CI) < spawning_threshold) {
                        break;
                    }
                    if (!(detI.get_beta_bit(a) or detI.get_beta_bit(b))) {
                        max_coupling.second = std::max(max_coupling.second, std::fabs(HJI));
                        HJI *= detJ.double_excitation_bb(i, j, a, b);

                        size_t index = dets_hashvec.find(detJ);
                        if (index >= pre_C_size) {
                            new_det_C_vec.push_back(std::make_pair(detJ, tau * HJI * CI));
#pragma omp atomic
                            num_off_diag_elem_ += 2;
                        } else {
#pragma omp atomic
                            result_C[index] += tau * HJI * CI;
#pragma omp atomic
                            ++num_off_diag_elem_;
                            if (std::fabs(HJI * pre_C[index]) < spawning_threshold) {
#pragma omp atomic
                                result_C[I] += tau * HJI * pre_C[index];
#pragma omp atomic
                                ++num_off_diag_elem_;
                            }
                        }

                        detJ.set_beta_bit(i, true);
                        detJ.set_beta_bit(j, true);
                        detJ.set_beta_bit(a, false);
                        detJ.set_beta_bit(b, false);
                    }
                }
            }
        }
        // parallel_timer_off("PCI:doubles", omp_get_thread_num());
    }
}

void ProjectorCI_HashVec::apply_tau_H_ref_C_symm(double tau, double spawning_threshold,
                                                 const det_hashvec& dets_hashvec,
                                                 const std::vector<double>& ref_C,
                                                 const std::vector<double>& pre_C,
                                                 std::vector<double>& result_C, double S) {

    result_C.clear();
    result_C.resize(dets_hashvec.size(), 0.0);

    size_t ref_max_I = ref_C.size();
#pragma omp parallel for
    for (size_t I = 0; I < ref_max_I; ++I) {
        std::pair<double, double> max_coupling;
        max_coupling = dets_max_couplings_[I];
        apply_tau_H_ref_C_symm_det_dynamic_HBCI_2(tau, spawning_threshold, dets_hashvec, pre_C,
                                                  ref_C, I, pre_C[I], ref_C[I], result_C, S,
                                                  max_coupling);
    }
    size_t max_I = pre_C.size();
#pragma omp parallel for
    for (size_t I = ref_max_I; I < max_I; ++I) {
        // Diagonal contribution
        // parallel_timer_on("PCI:diagonal", omp_get_thread_num());
        double det_energy = as_ints_->energy(dets_hashvec[I]) + as_ints_->scalar_energy();
        // parallel_timer_off("PCI:diagonal", omp_get_thread_num());
        // Diagonal contributions
        result_C[I] += tau * (det_energy - S) * pre_C[I];
    }
}

void ProjectorCI_HashVec::apply_tau_H_ref_C_symm_det_dynamic_HBCI_2(
    double tau, double spawning_threshold, const det_hashvec& dets_hashvec,
    const std::vector<double>& pre_C, const std::vector<double>& ref_C, size_t I, double CI,
    double ref_CI, std::vector<double>& result_C, double E0,
    const std::pair<double, double>& max_coupling) {

    const Determinant& detI = dets_hashvec[I];
    size_t ref_C_size = ref_C.size();

    bool do_singles = std::fabs(max_coupling.first * ref_CI) >= spawning_threshold;
    bool do_doubles = std::fabs(max_coupling.second * ref_CI) >= spawning_threshold;

    // Diagonal contributions
    // parallel_timer_on("PCI:diagonal", omp_get_thread_num());

    double det_energy = as_ints_->energy(detI) + as_ints_->scalar_energy();

#pragma omp atomic
    result_C[I] += tau * (det_energy - E0) * CI;

    // parallel_timer_off("PCI:diagonal", omp_get_thread_num());

    Determinant detJ(detI);
    if (do_singles) {
        // parallel_timer_on("PCI:singles", omp_get_thread_num());
        // Generate alpha excitations
        for (size_t x = 0; x < a_couplings_size_; ++x) {
            double HJI_max = std::get<1>(a_couplings_[x]);
            if (std::fabs(HJI_max * ref_CI) < spawning_threshold) {
                break;
            }
            int i = std::get<0>(a_couplings_[x]);
            if (detI.get_alfa_bit(i)) {
                std::vector<std::tuple<int, double>>& sub_couplings = std::get<2>(a_couplings_[x]);
                size_t sub_couplings_size = sub_couplings.size();
                for (size_t y = 0; y < sub_couplings_size; ++y) {
                    int a;
                    double HJI_bound;
                    std::tie(a, HJI_bound) = sub_couplings[y];
                    if (std::fabs(HJI_bound * ref_CI) < spawning_threshold) {
                        break;
                    }
                    if (!detI.get_alfa_bit(a)) {
                        Determinant detJ(detI);

                        double HJI = as_ints_->slater_rules_single_alpha_abs(detJ, i, a);

                        if (std::fabs(HJI * ref_CI) >= spawning_threshold) {
                            HJI *= detJ.single_excitation_a(i, a);

                            size_t index = dets_hashvec.find(detJ);
#pragma omp atomic
                            result_C[index] += tau * HJI * CI;

                            if ((index < ref_C_size &&
                                 std::fabs(HJI * ref_C[index]) < spawning_threshold) ||
                                index >= ref_C_size) {
#pragma omp atomic
                                result_C[I] += tau * HJI * pre_C[index];
                            }
                            detJ.set_alfa_bit(i, true);
                            detJ.set_alfa_bit(a, false);
                        }
                    }
                }
            }
        }
        // Generate beta excitations
        for (size_t x = 0; x < b_couplings_size_; ++x) {
            double HJI_max = std::get<1>(b_couplings_[x]);
            if (std::fabs(HJI_max * ref_CI) < spawning_threshold) {
                break;
            }
            int i = std::get<0>(b_couplings_[x]);
            if (detI.get_beta_bit(i)) {
                std::vector<std::tuple<int, double>>& sub_couplings = std::get<2>(b_couplings_[x]);
                size_t sub_couplings_size = sub_couplings.size();
                for (size_t y = 0; y < sub_couplings_size; ++y) {
                    int a;
                    double HJI_bound;
                    std::tie(a, HJI_bound) = sub_couplings[y];
                    if (std::fabs(HJI_bound * ref_CI) < spawning_threshold) {
                        break;
                    }
                    if (!detI.get_beta_bit(a)) {
                        Determinant detJ(detI);

                        double HJI = as_ints_->slater_rules_single_beta_abs(detJ, i, a);

                        if (std::fabs(HJI * ref_CI) >= spawning_threshold) {
                            HJI *= detJ.single_excitation_b(i, a);

                            size_t index = dets_hashvec.find(detJ);
#pragma omp atomic
                            result_C[index] += tau * HJI * CI;

                            if ((index < ref_C_size &&
                                 std::fabs(HJI * ref_C[index]) < spawning_threshold) ||
                                index >= ref_C_size) {
#pragma omp atomic
                                result_C[I] += tau * HJI * pre_C[index];
                            }
                            detJ.set_beta_bit(i, true);
                            detJ.set_beta_bit(a, false);
                        }
                    }
                }
            }
        }
        // parallel_timer_off("PCI:singles", omp_get_thread_num());
    }

    if (do_doubles) {
        // parallel_timer_on("PCI:doubles", omp_get_thread_num());
        // Generate alpha-alpha excitations
        for (size_t x = 0; x < aa_couplings_size_; ++x) {
            double HJI_max = std::get<2>(aa_couplings_[x]);
            if (std::fabs(HJI_max * ref_CI) < spawning_threshold) {
                break;
            }
            int i = std::get<0>(aa_couplings_[x]);
            int j = std::get<1>(aa_couplings_[x]);
            if (detI.get_alfa_bit(i) and detI.get_alfa_bit(j)) {
                std::vector<std::tuple<int, int, double>>& sub_couplings =
                    std::get<3>(aa_couplings_[x]);
                size_t sub_couplings_size = sub_couplings.size();
                for (size_t y = 0; y < sub_couplings_size; ++y) {
                    int a, b;
                    double HJI;
                    std::tie(a, b, HJI) = sub_couplings[y];
                    if (std::fabs(HJI * ref_CI) < spawning_threshold) {
                        break;
                    }
                    if (!(detI.get_alfa_bit(a) or detI.get_alfa_bit(b))) {
                        HJI *= detJ.double_excitation_aa(i, j, a, b);

                        size_t index = dets_hashvec.find(detJ);
#pragma omp atomic
                        result_C[index] += tau * HJI * CI;

                        if ((index < ref_C_size &&
                             std::fabs(HJI * ref_C[index]) < spawning_threshold) ||
                            index >= ref_C_size) {
#pragma omp atomic
                            result_C[I] += tau * HJI * pre_C[index];
                        }
                        detJ.set_alfa_bit(i, true);
                        detJ.set_alfa_bit(j, true);
                        detJ.set_alfa_bit(a, false);
                        detJ.set_alfa_bit(b, false);
                    }
                }
            }
        }
        // Generate alpha-beta excitations
        for (size_t x = 0; x < ab_couplings_size_; ++x) {
            double HJI_max = std::get<2>(ab_couplings_[x]);
            if (std::fabs(HJI_max * ref_CI) < spawning_threshold) {
                break;
            }
            int i = std::get<0>(ab_couplings_[x]);
            int j = std::get<1>(ab_couplings_[x]);
            if (detI.get_alfa_bit(i) and detI.get_beta_bit(j)) {
                std::vector<std::tuple<int, int, double>>& sub_couplings =
                    std::get<3>(ab_couplings_[x]);
                size_t sub_couplings_size = sub_couplings.size();
                for (size_t y = 0; y < sub_couplings_size; ++y) {
                    int a, b;
                    double HJI;
                    std::tie(a, b, HJI) = sub_couplings[y];
                    if (std::fabs(HJI * ref_CI) < spawning_threshold) {
                        break;
                    }
                    if (!(detI.get_alfa_bit(a) or detI.get_beta_bit(b))) {
                        HJI *= detJ.double_excitation_ab(i, j, a, b);

                        size_t index = dets_hashvec.find(detJ);
#pragma omp atomic
                        result_C[index] += tau * HJI * CI;

                        if ((index < ref_C_size &&
                             std::fabs(HJI * ref_C[index]) < spawning_threshold) ||
                            index >= ref_C_size) {
#pragma omp atomic
                            result_C[I] += tau * HJI * pre_C[index];
                        }
                        detJ.set_alfa_bit(i, true);
                        detJ.set_beta_bit(j, true);
                        detJ.set_alfa_bit(a, false);
                        detJ.set_beta_bit(b, false);
                    }
                }
            }
        }
        // Generate beta-beta excitations
        for (size_t x = 0; x < bb_couplings_size_; ++x) {
            double HJI_max = std::get<2>(bb_couplings_[x]);
            if (std::fabs(HJI_max * ref_CI) < spawning_threshold) {
                break;
            }
            int i = std::get<0>(bb_couplings_[x]);
            int j = std::get<1>(bb_couplings_[x]);
            if (detI.get_beta_bit(i) and detI.get_beta_bit(j)) {
                std::vector<std::tuple<int, int, double>>& sub_couplings =
                    std::get<3>(bb_couplings_[x]);
                size_t sub_couplings_size = sub_couplings.size();
                for (size_t y = 0; y < sub_couplings_size; ++y) {
                    int a, b;
                    double HJI;
                    std::tie(a, b, HJI) = sub_couplings[y];
                    if (std::fabs(HJI * ref_CI) < spawning_threshold) {
                        break;
                    }
                    if (!(detI.get_beta_bit(a) or detI.get_beta_bit(b))) {
                        HJI *= detJ.double_excitation_bb(i, j, a, b);

                        size_t index = dets_hashvec.find(detJ);
#pragma omp atomic
                        result_C[index] += tau * HJI * CI;

                        if ((index < ref_C_size &&
                             std::fabs(HJI * ref_C[index]) < spawning_threshold) ||
                            index >= ref_C_size) {
#pragma omp atomic
                            result_C[I] += tau * HJI * pre_C[index];
                        }
                        detJ.set_beta_bit(i, true);
                        detJ.set_beta_bit(j, true);
                        detJ.set_beta_bit(a, false);
                        detJ.set_beta_bit(b, false);
                    }
                }
            }
        }
        // parallel_timer_off("PCI:doubles", omp_get_thread_num());
    }
}

std::map<std::string, double> ProjectorCI_HashVec::estimate_energy(const det_hashvec& dets_hashvec,
                                                                   std::vector<double>& C) {
    std::map<std::string, double> results;
    //    det_hashvec dets_hashvec(dets);
    //    dets = dets_hashvec.toVector();
    timer_on("PCI:<E>p");
    results["PROJECTIVE ENERGY"] = estimate_proj_energy(dets_hashvec, C);
    timer_off("PCI:<E>p");

    if (variational_estimate_) {
        if (fast_variational_estimate_) {
            timer_on("PCI:<E>vs");
            results["VARIATIONAL ENERGY"] =
                estimate_var_energy_sparse(dets_hashvec, C, energy_estimate_threshold_);
            timer_off("PCI:<E>vs");
        } else {
            timer_on("PCI:<E>v");
            results["VARIATIONAL ENERGY"] =
                estimate_var_energy(dets_hashvec, C, energy_estimate_threshold_);
            timer_off("PCI:<E>v");
        }
    }
    //    dets_hashvec = det_hashvec(dets);
    //    dets = dets_hashvec.toVector();
    return results;
}

static bool abs_compare(double a, double b) { return (std::abs(a) < std::abs(b)); }

double ProjectorCI_HashVec::estimate_proj_energy(const det_hashvec& dets_hashvec,
                                                 std::vector<double>& C) {
    // Find the determinant with the largest value of C
    auto result = std::max_element(C.begin(), C.end(), abs_compare);
    size_t J = std::distance(C.begin(), result);
    double CJ = C[J];

    // Compute the projective energy
    double projective_energy_estimator = 0.0;
    for (int I = 0, max_I = dets_hashvec.size(); I < max_I; ++I) {
        double HIJ = as_ints_->slater_rules(dets_hashvec[I], (dets_hashvec[J]));
        projective_energy_estimator += HIJ * C[I] / CJ;
    }
    return projective_energy_estimator + nuclear_repulsion_energy_ + as_ints_->scalar_energy();
}

double ProjectorCI_HashVec::estimate_var_energy(const det_hashvec& dets_hashvec,
                                                std::vector<double>& C, double tollerance) {
    // Compute a variational estimator of the energy
    size_t size = dets_hashvec.size();
    double variational_energy_estimator = 0.0;
#pragma omp parallel for reduction(+ : variational_energy_estimator)
    for (size_t I = 0; I < size; ++I) {
        const Determinant& detI = dets_hashvec[I];
        variational_energy_estimator += C[I] * C[I] * as_ints_->energy(detI);
        for (size_t J = I + 1; J < size; ++J) {
            if (std::fabs(C[I] * C[J]) > tollerance) {
                double HIJ = as_ints_->slater_rules(dets_hashvec[I], (dets_hashvec[J]));
                variational_energy_estimator += 2.0 * C[I] * HIJ * C[J];
            }
        }
    }
    return variational_energy_estimator + nuclear_repulsion_energy_ + as_ints_->scalar_energy();
}

double ProjectorCI_HashVec::estimate_var_energy_within_error(const det_hashvec& dets_hashvec,
                                                             std::vector<double>& C,
                                                             double max_error) {
    // Compute a variational estimator of the energy
    size_t cut_index = dets_hashvec.size() - 1;
    double max_HIJ = dets_single_max_coupling_ > dets_double_max_coupling_
                         ? dets_single_max_coupling_
                         : dets_double_max_coupling_;
    double ignore_bound = max_error * max_error / (2.0 * max_HIJ * max_HIJ);
    double cume_ignore = 0.0;
    for (; cut_index > 0; --cut_index) {
        cume_ignore += C[cut_index] * C[cut_index];
        if (cume_ignore >= ignore_bound) {
            break;
        }
    }
    cume_ignore -= C[cut_index] * C[cut_index];
    if (cut_index < dets_hashvec.size() - 1) {
        ++cut_index;
    }
    outfile->Printf(
        "\n  Variational energy estimated with %zu determinants to meet the max error %e",
        cut_index + 1, max_error);
    double variational_energy_estimator = 0.0;
#pragma omp parallel for reduction(+ : variational_energy_estimator)
    for (size_t I = 0; I <= cut_index; ++I) {
        const Determinant& detI = dets_hashvec[I];
        variational_energy_estimator += C[I] * C[I] * as_ints_->energy(detI);
        for (size_t J = I + 1; J <= cut_index; ++J) {
            double HIJ = as_ints_->slater_rules(dets_hashvec[I], dets_hashvec[J]);
            variational_energy_estimator += 2.0 * C[I] * HIJ * C[J];
        }
    }
    variational_energy_estimator /= 1.0 - cume_ignore;
    return variational_energy_estimator + nuclear_repulsion_energy_ + as_ints_->scalar_energy();
}

double ProjectorCI_HashVec::estimate_var_energy_within_error_sigma(const det_hashvec& dets_hashvec,
                                                                   std::vector<double>& C,
                                                                   double max_error) {
    // Compute a variational estimator of the energy
    size_t cut_index = dets_hashvec.size() - 1;
    double max_HIJ = dets_single_max_coupling_ > dets_double_max_coupling_
                         ? dets_single_max_coupling_
                         : dets_double_max_coupling_;
    double ignore_bound = max_error * max_error / (2.0 * max_HIJ * max_HIJ);
    double cume_ignore = 0.0;
    for (; cut_index > 0; --cut_index) {
        cume_ignore += C[cut_index] * C[cut_index];
        if (cume_ignore >= ignore_bound) {
            break;
        }
    }
    cume_ignore -= C[cut_index] * C[cut_index];
    if (cut_index < dets_hashvec.size() - 1) {
        ++cut_index;
    }
    outfile->Printf(
        "\n  Variational energy estimated with %zu determinants to meet the max error %e",
        cut_index + 1, max_error);
    double variational_energy_estimator = 0.0;

    WFNOperator op(mo_symmetry_, as_ints_);
    std::vector<Determinant> sub_dets = dets_hashvec.toVector();
    sub_dets.erase(sub_dets.begin() + cut_index + 1, sub_dets.end());
    DeterminantHashVec det_map(sub_dets);
    op.build_strings(det_map);
    op.op_s_lists(det_map);
    op.tp_s_lists(det_map);
    //    std::vector<std::pair<std::vector<size_t>, std::vector<double>>> H =
    //    op.build_H_sparse(det_map);
    //    SigmaVectorSparse svs(H);
    SigmaVectorWfn2 svs(det_map, op, as_ints_);
    size_t sub_size = svs.size();
    // allocate vectors
    psi::SharedVector b(new Vector("b", sub_size));
    psi::SharedVector sigma(new Vector("sigma", sub_size));
    for (size_t i = 0; i < sub_size; ++i) {
        b->set(i, C[i]);
    }
    svs.compute_sigma(sigma, b);
    variational_energy_estimator = sigma->dot(b.get());

    variational_energy_estimator /= 1.0 - cume_ignore;
    return variational_energy_estimator + nuclear_repulsion_energy_ + as_ints_->scalar_energy();
}

double ProjectorCI_HashVec::estimate_var_energy_sparse(const det_hashvec& dets_hashvec,
                                                       std::vector<double>& C, double max_error) {
    size_t cut_index = dets_hashvec.size() - 1;
    double max_HIJ = dets_single_max_coupling_ > dets_double_max_coupling_
                         ? dets_single_max_coupling_
                         : dets_double_max_coupling_;
    double ignore_bound = max_error * max_error / (2.0 * max_HIJ * max_HIJ);
    double cume_ignore = 0.0;
    for (; cut_index > 0; --cut_index) {
        cume_ignore += C[cut_index] * C[cut_index];
        if (cume_ignore >= ignore_bound) {
            break;
        }
    }
    cume_ignore -= C[cut_index] * C[cut_index];
    if (cut_index < dets_hashvec.size() - 1) {
        ++cut_index;
    }
    outfile->Printf(
        "\n  Variational energy estimated with %zu determinants to meet the max error %e",
        cut_index + 1, max_error);

    timer_on("PCI:Couplings");
    compute_couplings_half(dets_hashvec, cut_index + 1);
    timer_off("PCI:Couplings");

    double variational_energy_estimator = 0.0;
    std::vector<double> energy(num_threads_, 0.0);

#pragma omp parallel for
    for (size_t I = 0; I <= cut_index; ++I) {
        energy[omp_get_thread_num()] += form_H_C(dets_hashvec, C, I);
    }
    for (int t = 0; t < num_threads_; ++t) {
        variational_energy_estimator += energy[t];
    }
    variational_energy_estimator /= 1.0 - cume_ignore;
    return variational_energy_estimator + nuclear_repulsion_energy_ + as_ints_->scalar_energy();
}

void ProjectorCI_HashVec::print_wfn(const det_hashvec& space_hashvec, std::vector<double>& C,
                                    size_t max_output) {
    outfile->Printf("\n\n  Most important contributions to the wave function:\n");

    size_t max_dets = std::min(int(max_output), int(C.size()));
    for (size_t I = 0; I < max_dets; ++I) {
        outfile->Printf("\n  %3zu  %13.6g %13.6g  %10zu %s  %18.12f", I, C[I], C[I] * C[I], I,
                        space_hashvec[I].str().c_str(),
                        as_ints_->energy(space_hashvec[I]) + as_ints_->scalar_energy());
    }

    // Compute the expectation value of the spin
    size_t max_sample = 1000;
    size_t max_I = 0;
    double sum_weight = 0.0;
    double wfn_threshold = 0.95;
    for (size_t I = 0; I < space_hashvec.size(); ++I) {
        if ((sum_weight < wfn_threshold) and (I < max_sample)) {
            sum_weight += C[I] * C[I];
            max_I++;
        } else if (std::fabs(C[I - 1]) - std::fabs(C[I]) < 1.0e-6) {
            // Special case, if there are several equivalent determinants
            sum_weight += C[I] * C[I];
            max_I++;
        } else {
            break;
        }
    }

    double norm = 0.0;
    double S2 = 0.0;
    for (size_t I = 0; I < max_I; ++I) {
        for (size_t J = 0; J < max_I; ++J) {
            if (std::fabs(C[I] * C[J]) > 1.0e-12) {
                const double S2IJ = spin2(space_hashvec[I], space_hashvec[J]);
                S2 += C[I] * C[J] * S2IJ;
            }
        }
        norm += C[I] * C[I];
    }
    S2 /= norm;
    double S = std::fabs(0.5 * (std::sqrt(1.0 + 4.0 * S2) - 1.0));

    std::vector<std::string> s2_labels({"singlet", "doublet", "triplet", "quartet", "quintet",
                                        "sextet", "septet", "octet", "nonet", "decaet"});
    size_t nLet = std::round(S * 2.0);
    std::string state_label;
    if (nLet < 10) {
        state_label = s2_labels[nLet];
    } else {
        state_label = std::to_string(nLet) + "-let";
    }

    outfile->Printf("\n\n  Spin State: S^2 = %5.3f, S = %5.3f, %s (from %zu "
                    "determinants,%.2f%%)",
                    S2, S, state_label.c_str(), max_I, 100.0 * sum_weight);
}

void ProjectorCI_HashVec::save_wfn(
    det_hashvec& space, std::vector<double>& C,
    std::vector<std::pair<det_hashvec, std::vector<double>>>& solutions) {
    outfile->Printf("\n\n  Saving the wave function:\n");

    //    det_hash<> solution;
    //    for (size_t I = 0; I < space.size(); ++I) {
    //        solution[space[I]] = C[I];
    //    }
    //    solutions.push_back(std::move(solution));
    solutions.push_back(std::make_pair(space, C));
}

void ProjectorCI_HashVec::orthogonalize(
    det_hashvec& space, std::vector<double>& C,
    std::vector<std::pair<det_hashvec, std::vector<double>>>& solutions) {
    //    det_hash<> det_C;
    //    //    for (size_t I = 0; I < space.size(); ++I) {
    //    //        det_C[space[I]] = C[I];
    //    //    }
    //    det_C = space.toUnordered_map(C);
    //    for (size_t n = 0; n < solutions.size(); ++n) {
    //        double dot_prod = dot(det_C, solutions[n]);
    //        add(det_C, -dot_prod, solutions[n]);
    //    }
    //    normalize(det_C);
    //    //    copy_hash_to_vec(det_C, space, C);
    //    space = det_hashvec(det_C, C);
    for (size_t n = 0; n < solutions.size(); ++n) {
        double dot_prod = dot(space, C, solutions[n].first, solutions[n].second);
        add(space, C, -dot_prod, solutions[n].first, solutions[n].second);
    }
    //    normalize(C);
}

double ProjectorCI_HashVec::form_H_C(const det_hashvec& dets_hashvec, std::vector<double>& C,
                                     size_t I) {
    const Determinant& detI = dets_hashvec[I];
    double CI = C[I];

    // diagonal contribution
    double result = CI * CI * as_ints_->energy(detI);

    std::vector<int> aocc = detI.get_alfa_occ(nact_);
    std::vector<int> bocc = detI.get_beta_occ(nact_);
    std::vector<int> avir = detI.get_alfa_vir(nact_);
    std::vector<int> bvir = detI.get_beta_vir(nact_);

    int noalpha = aocc.size();
    int nobeta = bocc.size();
    int nvalpha = avir.size();
    int nvbeta = bvir.size();

    Determinant detJ(detI);
    double HJI, sign;
    // Generate aa excitations
    for (int i = 0; i < noalpha; ++i) {
        int ii = aocc[i];
        for (int a = 0; a < nvalpha; ++a) {
            int aa = avir[a];
            if ((mo_symmetry_[ii] ^ mo_symmetry_[aa]) == 0) {
                detJ.set_alfa_bit(ii, false);
                detJ.set_alfa_bit(aa, true);
                size_t index = dets_hashvec.find(detJ);
                if (index < I) {
                    HJI = as_ints_->slater_rules_single_alpha(detI, ii, aa);
                    result += 2.0 * HJI * CI * C[index];
                }
                detJ.set_alfa_bit(ii, true);
                detJ.set_alfa_bit(aa, false);
            }
        }
    }

    for (int i = 0; i < nobeta; ++i) {
        int ii = bocc[i];
        for (int a = 0; a < nvbeta; ++a) {
            int aa = bvir[a];
            if ((mo_symmetry_[ii] ^ mo_symmetry_[aa]) == 0) {
                detJ.set_beta_bit(ii, false);
                detJ.set_beta_bit(aa, true);
                size_t index = dets_hashvec.find(detJ);
                if (index < I) {
                    HJI = as_ints_->slater_rules_single_beta(detI, ii, aa);
                    result += 2.0 * HJI * CI * C[index];
                }
                detJ.set_beta_bit(ii, true);
                detJ.set_beta_bit(aa, false);
            }
        }
    }

    // Generate aa excitations
    for (int i = 0; i < noalpha; ++i) {
        int ii = aocc[i];
        for (int j = i + 1; j < noalpha; ++j) {
            int jj = aocc[j];
            for (int a = 0; a < nvalpha; ++a) {
                int aa = avir[a];
                for (int b = a + 1; b < nvalpha; ++b) {
                    int bb = avir[b];
                    if ((mo_symmetry_[ii] ^ mo_symmetry_[jj] ^ mo_symmetry_[aa] ^
                         mo_symmetry_[bb]) == 0) {
                        detJ.set_alfa_bit(ii, false);
                        detJ.set_alfa_bit(jj, false);
                        detJ.set_alfa_bit(aa, true);
                        detJ.set_alfa_bit(bb, true);
                        size_t index = dets_hashvec.find(detJ);
                        if (index < I) {
                            sign = detJ.double_excitation_aa(aa, bb, ii, jj);
                            HJI = as_ints_->tei_aa(ii, jj, aa, bb);
                            result += 2.0 * sign * HJI * CI * C[index];
                        } else {
                            detJ.set_alfa_bit(ii, true);
                            detJ.set_alfa_bit(jj, true);
                            detJ.set_alfa_bit(aa, false);
                            detJ.set_alfa_bit(bb, false);
                        }
                    }
                }
            }
        }
    }

    for (int i = 0; i < noalpha; ++i) {
        int ii = aocc[i];
        for (int j = 0; j < nobeta; ++j) {
            int jj = bocc[j];
            for (int a = 0; a < nvalpha; ++a) {
                int aa = avir[a];
                for (int b = 0; b < nvbeta; ++b) {
                    int bb = bvir[b];
                    if ((mo_symmetry_[ii] ^ mo_symmetry_[jj] ^ mo_symmetry_[aa] ^
                         mo_symmetry_[bb]) == 0) {
                        detJ.set_alfa_bit(ii, false);
                        detJ.set_beta_bit(jj, false);
                        detJ.set_alfa_bit(aa, true);
                        detJ.set_beta_bit(bb, true);
                        size_t index = dets_hashvec.find(detJ);
                        if (index < I) {
                            sign = detJ.double_excitation_ab(aa, bb, ii, jj);
                            HJI = as_ints_->tei_ab(ii, jj, aa, bb);
                            result += 2.0 * sign * HJI * CI * C[index];
                        } else {
                            detJ.set_alfa_bit(ii, true);
                            detJ.set_beta_bit(jj, true);
                            detJ.set_alfa_bit(aa, false);
                            detJ.set_beta_bit(bb, false);
                        }
                    }
                }
            }
        }
    }
    for (int i = 0; i < nobeta; ++i) {
        int ii = bocc[i];
        for (int j = i + 1; j < nobeta; ++j) {
            int jj = bocc[j];
            for (int a = 0; a < nvbeta; ++a) {
                int aa = bvir[a];
                for (int b = a + 1; b < nvbeta; ++b) {
                    int bb = bvir[b];
                    if ((mo_symmetry_[ii] ^
                         (mo_symmetry_[jj] ^ (mo_symmetry_[aa] ^ mo_symmetry_[bb]))) == 0) {
                        detJ.set_beta_bit(ii, false);
                        detJ.set_beta_bit(jj, false);
                        detJ.set_beta_bit(aa, true);
                        detJ.set_beta_bit(bb, true);
                        size_t index = dets_hashvec.find(detJ);
                        if (index < I) {
                            sign = detJ.double_excitation_bb(aa, bb, ii, jj);
                            HJI = as_ints_->tei_bb(ii, jj, aa, bb);
                            result += 2.0 * sign * HJI * CI * C[index];
                        } else {
                            detJ.set_beta_bit(ii, true);
                            detJ.set_beta_bit(jj, true);
                            detJ.set_beta_bit(aa, false);
                            detJ.set_beta_bit(bb, false);
                        }
                    }
                }
            }
        }
    }
    return result;
}

double ProjectorCI_HashVec::form_H_C_2(const det_hashvec& dets_hashvec, std::vector<double>& C,
                                       size_t I, size_t cut_index) {
    const Determinant& detI = dets_hashvec[I];
    double CI = C[I];

    // diagonal contribution
    double result = CI * CI * as_ints_->energy(detI);

    Determinant detJ(detI);
    double HJI;
    for (size_t x = 0; x < a_couplings_size_; ++x) {
        int i = std::get<0>(a_couplings_[x]);
        if (detI.get_alfa_bit(i)) {
            std::vector<std::tuple<int, double>>& sub_couplings = std::get<2>(a_couplings_[x]);
            size_t sub_couplings_size = sub_couplings.size();
            for (size_t y = 0; y < sub_couplings_size; ++y) {
                int a;
                std::tie(a, HJI) = sub_couplings[y];
                if (!detI.get_alfa_bit(a)) {
                    //                    size_t max_bit = 2 * nact_;
                    //                    bit_t& bits = detJ.bits_;
                    //                    std::vector<double>& double_couplings =
                    //                        single_alpha_excite_double_couplings_[i][a];
                    //                    for (size_t p = 0; p < max_bit; ++p) {
                    //                        if (bits[p]) {
                    //                            HJI += double_couplings[p];
                    //                        }
                    //                    }
                    //                    HJI *= detJ.single_excitation_a(i, a);
                    HJI = as_ints_->slater_rules_single_alpha_abs(detJ, i, a);
                    HJI *= detJ.single_excitation_a(i, a);
                    size_t index = dets_hashvec.find(detJ);
                    if (index <= cut_index) {
                        result += 2.0 * HJI * CI * C[index];
                    }
                    detJ.set_alfa_bit(i, true);
                    detJ.set_alfa_bit(a, false);
                }
            }
        }
    }

    for (size_t x = 0; x < b_couplings_size_; ++x) {
        int i = std::get<0>(b_couplings_[x]);
        if (detI.get_beta_bit(i)) {
            std::vector<std::tuple<int, double>>& sub_couplings = std::get<2>(b_couplings_[x]);
            size_t sub_couplings_size = sub_couplings.size();
            for (size_t y = 0; y < sub_couplings_size; ++y) {
                int a;
                std::tie(a, HJI) = sub_couplings[y];
                if (!detI.get_beta_bit(a)) {
                    //                    size_t max_bit = 2 * nact_;
                    //                    bit_t& bits = detJ.bits_;
                    //                    std::vector<double>& double_couplings =
                    //                        single_beta_excite_double_couplings_[i][a];
                    //                    for (size_t p = 0; p < max_bit; ++p) {
                    //                        if (bits[p]) {
                    //                            HJI += double_couplings[p];
                    //                        }
                    //                    }
                    //                    HJI *= detJ.single_excitation_b(i, a);
                    HJI = as_ints_->slater_rules_single_beta_abs(detJ, i, a);
                    HJI *= detJ.single_excitation_b(i, a);
                    size_t index = dets_hashvec.find(detJ);
                    if (index <= cut_index) {
                        result += 2.0 * HJI * CI * C[index];
                    }
                    detJ.set_beta_bit(i, true);
                    detJ.set_beta_bit(a, false);
                }
            }
        }
    }

    // Generate aa excitations
    for (size_t x = 0; x < aa_couplings_size_; ++x) {
        int i = std::get<0>(aa_couplings_[x]);
        int j = std::get<1>(aa_couplings_[x]);
        if (detI.get_alfa_bit(i) and detI.get_alfa_bit(j)) {
            std::vector<std::tuple<int, int, double>>& sub_couplings =
                std::get<3>(aa_couplings_[x]);
            size_t sub_couplings_size = sub_couplings.size();
            for (size_t y = 0; y < sub_couplings_size; ++y) {
                int a, b;
                std::tie(a, b, HJI) = sub_couplings[y];
                if (!(detI.get_alfa_bit(a) or detI.get_alfa_bit(b))) {
                    HJI *= detJ.double_excitation_aa(i, j, a, b);
                    size_t index = dets_hashvec.find(detJ);
                    if (index <= cut_index) {
                        result += 2.0 * HJI * CI * C[index];
                    }
                    detJ.set_alfa_bit(i, true);
                    detJ.set_alfa_bit(j, true);
                    detJ.set_alfa_bit(a, false);
                    detJ.set_alfa_bit(b, false);
                }
            }
        }
    }

    for (size_t x = 0; x < ab_couplings_size_; ++x) {
        int i = std::get<0>(ab_couplings_[x]);
        int j = std::get<1>(ab_couplings_[x]);
        if (detI.get_alfa_bit(i) and detI.get_beta_bit(j)) {
            std::vector<std::tuple<int, int, double>>& sub_couplings =
                std::get<3>(ab_couplings_[x]);
            size_t sub_couplings_size = sub_couplings.size();
            for (size_t y = 0; y < sub_couplings_size; ++y) {
                int a, b;
                std::tie(a, b, HJI) = sub_couplings[y];
                if (!(detI.get_alfa_bit(a) or detI.get_beta_bit(b))) {
                    HJI *= detJ.double_excitation_ab(i, j, a, b);
                    size_t index = dets_hashvec.find(detJ);
                    if (index <= cut_index) {
                        result += 2.0 * HJI * CI * C[index];
                    }
                    detJ.set_alfa_bit(i, true);
                    detJ.set_beta_bit(j, true);
                    detJ.set_alfa_bit(a, false);
                    detJ.set_beta_bit(b, false);
                }
            }
        }
    }

    for (size_t x = 0; x < bb_couplings_size_; ++x) {
        int i = std::get<0>(bb_couplings_[x]);
        int j = std::get<1>(bb_couplings_[x]);
        if (detI.get_beta_bit(i) and detI.get_beta_bit(j)) {
            std::vector<std::tuple<int, int, double>>& sub_couplings =
                std::get<3>(bb_couplings_[x]);
            size_t sub_couplings_size = sub_couplings.size();
            for (size_t y = 0; y < sub_couplings_size; ++y) {
                int a, b;
                std::tie(a, b, HJI) = sub_couplings[y];
                if (!(detI.get_beta_bit(a) or detI.get_beta_bit(b))) {
                    HJI *= detJ.double_excitation_bb(i, j, a, b);
                    size_t index = dets_hashvec.find(detJ);
                    if (index <= cut_index) {
                        result += 2.0 * HJI * CI * C[index];
                    }
                    detJ.set_beta_bit(i, true);
                    detJ.set_beta_bit(j, true);
                    detJ.set_beta_bit(a, false);
                    detJ.set_beta_bit(b, false);
                }
            }
        }
    }
    return result;
}

void ProjectorCI_HashVec::compute_single_couplings(double single_coupling_threshold) {
    struct {
        bool operator()(std::tuple<int, double> first, std::tuple<int, double> second) const {
            double H1 = std::get<1>(first);
            double H2 = std::get<1>(second);
            return H1 > H2;
        }
    } CouplingCompare;

    struct {
        bool
        operator()(std::tuple<int, double, std::vector<std::tuple<int, double>>> first,
                   std::tuple<int, double, std::vector<std::tuple<int, double>>> second) const {
            double H1 = std::get<1>(first);
            double H2 = std::get<1>(second);
            return H1 > H2;
        }
    } MaxCouplingCompare;

    //    single_alpha_excite_double_couplings_.clear();
    //    single_beta_excite_double_couplings_.clear();
    //    single_alpha_excite_double_couplings_.resize(nact_,
    //    std::vector<std::vector<double>>(nact_));
    //    single_beta_excite_double_couplings_.resize(nact_,
    //    std::vector<std::vector<double>>(nact_));

    dets_single_max_coupling_ = 0.0;
    a_couplings_.clear();
    a_couplings_.resize(nact_);
    for (size_t i = 0; i < nact_; ++i) {
        for (size_t a = i + 1; a < nact_; ++a) {
            if ((mo_symmetry_[i] ^ mo_symmetry_[a]) == 0) {
                double Hia = as_ints_->oei_a(i, a);
                std::vector<double> aa_double_couplings(nact_);
                std::vector<double> ab_double_couplings(nact_);
                //                single_alpha_excite_double_couplings_[i][a].resize(2 * nact_);
                //                single_alpha_excite_double_couplings_[a][i].resize(2 * nact_);
                for (size_t p = 0; p < nact_; ++p) {
                    aa_double_couplings[p] = as_ints_->tei_aa(i, p, a, p);
                    ab_double_couplings[p] = as_ints_->tei_ab(i, p, a, p);
                    //                    single_alpha_excite_double_couplings_[i][a][p] =
                    //                    aa_double_couplings[p];
                    //                    single_alpha_excite_double_couplings_[a][i][p] =
                    //                    aa_double_couplings[p];
                    //                    single_alpha_excite_double_couplings_[i][a][p + nact_] =
                    //                    ab_double_couplings[p];
                    //                    single_alpha_excite_double_couplings_[a][i][p + nact_] =
                    //                    ab_double_couplings[p];
                }
                std::sort(aa_double_couplings.begin(), aa_double_couplings.end());
                std::sort(ab_double_couplings.begin(), ab_double_couplings.end());
                double H1 = Hia, H2 = Hia;
                for (int x = 0; x < nalpha_; x++) {
                    H1 += aa_double_couplings[x];
                    H2 += aa_double_couplings[nact_ - 1 - x];
                }
                for (int y = 0; y < nbeta_; y++) {
                    H1 += ab_double_couplings[y];
                    H2 += ab_double_couplings[nact_ - 1 - y];
                }
                Hia = std::fabs(H1) > std::fabs(H2) ? std::fabs(H1) : std::fabs(H2);
                if (Hia >= single_coupling_threshold) {
                    std::get<2>(a_couplings_[i]).push_back(std::make_tuple(a, Hia));
                    std::get<2>(a_couplings_[a]).push_back(std::make_tuple(i, Hia));
                }
            }
        }
        if (std::get<2>(a_couplings_[i]).size() != 0) {
            std::sort(std::get<2>(a_couplings_[i]).begin(), std::get<2>(a_couplings_[i]).end(),
                      CouplingCompare);
            std::get<1>(a_couplings_[i]) = std::get<1>(std::get<2>(a_couplings_[i])[0]);
        } else {
            std::get<1>(a_couplings_[i]) = 0.0;
        }
        std::get<0>(a_couplings_[i]) = i;
    }
    std::sort(a_couplings_.begin(), a_couplings_.end(), MaxCouplingCompare);
    while (std::get<1>(a_couplings_.back()) == 0.0) {
        a_couplings_.pop_back();
    }
    a_couplings_size_ = a_couplings_.size();
    dets_single_max_coupling_ = std::get<1>(a_couplings_[0]);

    b_couplings_.clear();
    b_couplings_.resize(nact_);
    for (size_t i = 0; i < nact_; ++i) {
        for (size_t a = i + 1; a < nact_; ++a) {
            if ((mo_symmetry_[i] ^ mo_symmetry_[a]) == 0) {
                double Hia = as_ints_->oei_b(i, a);
                std::vector<double> ab_double_couplings(nact_);
                std::vector<double> bb_double_couplings(nact_);
                //                single_beta_excite_double_couplings_[i][a].resize(2 * nact_);
                //                single_beta_excite_double_couplings_[a][i].resize(2 * nact_);
                for (size_t p = 0; p < nact_; ++p) {
                    ab_double_couplings[p] = as_ints_->tei_ab(p, i, p, a);
                    bb_double_couplings[p] = as_ints_->tei_bb(i, p, a, p);
                    //                    single_beta_excite_double_couplings_[i][a][p] =
                    //                    ab_double_couplings[p];
                    //                    single_beta_excite_double_couplings_[a][i][p] =
                    //                    ab_double_couplings[p];
                    //                    single_beta_excite_double_couplings_[i][a][p + nact_] =
                    //                    bb_double_couplings[p];
                    //                    single_beta_excite_double_couplings_[a][i][p + nact_] =
                    //                    bb_double_couplings[p];
                }
                std::sort(ab_double_couplings.begin(), ab_double_couplings.end());
                std::sort(bb_double_couplings.begin(), bb_double_couplings.end());
                double H1 = Hia, H2 = Hia;
                for (int x = 0; x < nalpha_; x++) {
                    H1 += ab_double_couplings[x];
                    H2 += ab_double_couplings[nact_ - 1 - x];
                }
                for (int y = 0; y < nbeta_; y++) {
                    H1 += bb_double_couplings[y];
                    H2 += bb_double_couplings[nact_ - 1 - y];
                }
                Hia = std::fabs(H1) > std::fabs(H2) ? std::fabs(H1) : std::fabs(H2);
                if (Hia >= single_coupling_threshold) {
                    std::get<2>(b_couplings_[i]).push_back(std::make_tuple(a, Hia));
                    std::get<2>(b_couplings_[a]).push_back(std::make_tuple(i, Hia));
                }
            }
        }
        if (std::get<2>(b_couplings_[i]).size() != 0) {
            std::sort(std::get<2>(b_couplings_[i]).begin(), std::get<2>(b_couplings_[i]).end(),
                      CouplingCompare);
            std::get<1>(b_couplings_[i]) = std::get<1>(std::get<2>(b_couplings_[i])[0]);
        } else {
            std::get<1>(b_couplings_[i]) = 0.0;
        }
        std::get<0>(b_couplings_[i]) = i;
    }
    std::sort(b_couplings_.begin(), b_couplings_.end(), MaxCouplingCompare);
    while (std::get<1>(b_couplings_.back()) == 0.0) {
        b_couplings_.pop_back();
    }
    b_couplings_size_ = b_couplings_.size();
    if (dets_single_max_coupling_ < std::get<1>(b_couplings_[0])) {
        dets_single_max_coupling_ = std::get<1>(b_couplings_[0]);
    }
}

void ProjectorCI_HashVec::compute_double_couplings(double double_coupling_threshold) {
    struct {
        bool operator()(std::tuple<int, int, double> first,
                        std::tuple<int, int, double> second) const {
            double H1 = std::get<2>(first);
            double H2 = std::get<2>(second);
            return std::fabs(H1) > std::fabs(H2);
        }
    } CouplingCompare;

    struct {
        bool operator()(
            std::tuple<int, int, double, std::vector<std::tuple<int, int, double>>> first,
            std::tuple<int, int, double, std::vector<std::tuple<int, int, double>>> second) const {
            double H1 = std::get<2>(first);
            double H2 = std::get<2>(second);
            return H1 > H2;
        }
    } MaxCouplingCompare;

    dets_double_max_coupling_ = 0.0;

    aa_couplings_.clear();
    for (size_t i = 0; i < nact_; ++i) {
        for (size_t j = i + 1; j < nact_; ++j) {
            std::vector<std::tuple<int, int, double>> ij_couplings;
            for (size_t a = 0; a < nact_; ++a) {
                if (a == i or a == j)
                    continue;
                for (size_t b = a + 1; b < nact_; ++b) {
                    if (b == i or b == j)
                        continue;
                    if ((mo_symmetry_[i] ^ mo_symmetry_[j] ^ mo_symmetry_[a] ^ mo_symmetry_[b]) ==
                        0) {
                        double Hijab = as_ints_->tei_aa(i, j, a, b);
                        if (std::fabs(Hijab) >= double_coupling_threshold) {
                            ij_couplings.push_back(std::make_tuple(a, b, Hijab));
                        }
                    }
                }
            }
            double max_ij_coupling = 0;
            if (ij_couplings.size() != 0) {
                std::sort(ij_couplings.begin(), ij_couplings.end(), CouplingCompare);
                max_ij_coupling = std::get<2>(ij_couplings[0]);
                aa_couplings_.push_back(
                    std::make_tuple(i, j, std::fabs(max_ij_coupling), ij_couplings));
            }
        }
    }
    aa_couplings_size_ = aa_couplings_.size();
    if (aa_couplings_size_ != 0) {
        std::sort(aa_couplings_.begin(), aa_couplings_.end(), MaxCouplingCompare);
        max_aa_coupling_ = std::get<2>(aa_couplings_[0]);
        dets_double_max_coupling_ = max_aa_coupling_;
    }

    ab_couplings_.clear();
    for (size_t i = 0; i < nact_; ++i) {
        for (size_t j = 0; j < nact_; ++j) {
            std::vector<std::tuple<int, int, double>> ij_couplings;
            for (size_t a = 0; a < nact_; ++a) {
                if (a == i)
                    continue;
                for (size_t b = 0; b < nact_; ++b) {
                    if (b == j)
                        continue;
                    if ((mo_symmetry_[i] ^ mo_symmetry_[j] ^ mo_symmetry_[a] ^ mo_symmetry_[b]) ==
                        0) {
                        double Hijab = as_ints_->tei_ab(i, j, a, b);
                        if (std::fabs(Hijab) >= double_coupling_threshold) {
                            ij_couplings.push_back(std::make_tuple(a, b, Hijab));
                        }
                    }
                }
            }
            double max_ij_coupling = 0;
            if (ij_couplings.size() != 0) {
                std::sort(ij_couplings.begin(), ij_couplings.end(), CouplingCompare);
                max_ij_coupling = std::get<2>(ij_couplings[0]);
                ab_couplings_.push_back(
                    std::make_tuple(i, j, std::fabs(max_ij_coupling), ij_couplings));
            }
        }
    }
    ab_couplings_size_ = ab_couplings_.size();
    if (ab_couplings_size_ != 0) {
        std::sort(ab_couplings_.begin(), ab_couplings_.end(), MaxCouplingCompare);
        max_ab_coupling_ = std::get<2>(ab_couplings_[0]);
        dets_double_max_coupling_ = dets_double_max_coupling_ > max_ab_coupling_
                                        ? dets_double_max_coupling_
                                        : max_ab_coupling_;
    }

    bb_couplings_.clear();
    for (size_t i = 0; i < nact_; ++i) {
        for (size_t j = i + 1; j < nact_; ++j) {
            std::vector<std::tuple<int, int, double>> ij_couplings;
            for (size_t a = 0; a < nact_; ++a) {
                if (a == i or a == j)
                    continue;
                for (size_t b = a + 1; b < nact_; ++b) {
                    if (b == i or b == j)
                        continue;
                    if ((mo_symmetry_[i] ^ mo_symmetry_[j] ^ mo_symmetry_[a] ^ mo_symmetry_[b]) ==
                        0) {
                        double Hijab = as_ints_->tei_bb(i, j, a, b);
                        if (std::fabs(Hijab) >= double_coupling_threshold) {
                            ij_couplings.push_back(std::make_tuple(a, b, Hijab));
                        }
                    }
                }
            }
            double max_ij_coupling = 0;
            if (ij_couplings.size() != 0) {
                std::sort(ij_couplings.begin(), ij_couplings.end(), CouplingCompare);
                max_ij_coupling = std::get<2>(ij_couplings[0]);
                bb_couplings_.push_back(
                    std::make_tuple(i, j, std::fabs(max_ij_coupling), ij_couplings));
            }
        }
    }
    bb_couplings_size_ = bb_couplings_.size();
    if (bb_couplings_size_ != 0) {
        std::sort(bb_couplings_.begin(), bb_couplings_.end(), MaxCouplingCompare);
        max_bb_coupling_ = std::get<2>(bb_couplings_[0]);
        dets_double_max_coupling_ = dets_double_max_coupling_ > max_bb_coupling_
                                        ? dets_double_max_coupling_
                                        : max_bb_coupling_;
    }
}

void ProjectorCI_HashVec::compute_couplings_half(const det_hashvec& dets, size_t cut_size) {
    Determinant andBits(dets[0]), orBits(dets[0]);
    andBits.flip();
    for (size_t i = 0; i < cut_size; ++i) {
        andBits = common_occupation(andBits, dets[i]);
        orBits = union_occupation(orBits, dets[i]);
    }
    Determinant actBits = different_occupation(andBits, orBits);

    a_couplings_.clear();
    a_couplings_.resize(nact_);
    for (size_t i = 0; i < nact_; ++i) {
        if (!actBits.get_alfa_bit(i))
            continue;
        std::vector<std::tuple<int, double>> i_couplings;
        for (size_t a = i + 1; a < nact_; ++a) {
            if (!actBits.get_alfa_bit(a))
                continue;
            if ((mo_symmetry_[i] ^ mo_symmetry_[a]) == 0) {
                double Hia = as_ints_->oei_a(i, a);
                i_couplings.push_back(std::make_tuple(a, Hia));
            }
        }
        if (i_couplings.size() != 0) {
            a_couplings_.push_back(std::make_tuple(i, 0.0, i_couplings));
        }
    }
    a_couplings_size_ = a_couplings_.size();

    b_couplings_.clear();
    b_couplings_.resize(nact_);
    for (size_t i = 0; i < nact_; ++i) {
        if (!actBits.get_beta_bit(i))
            continue;
        std::vector<std::tuple<int, double>> i_couplings;
        for (size_t a = i + 1; a < nact_; ++a) {
            if (!actBits.get_beta_bit(a))
                continue;
            if ((mo_symmetry_[i] ^ mo_symmetry_[a]) == 0) {
                double Hia = as_ints_->oei_b(i, a);
                i_couplings.push_back(std::make_tuple(a, Hia));
            }
        }
        if (i_couplings.size() != 0) {
            b_couplings_.push_back(std::make_tuple(i, 0.0, i_couplings));
        }
    }
    b_couplings_size_ = b_couplings_.size();

    aa_couplings_.clear();
    for (size_t i = 0; i < nact_; ++i) {
        if (!actBits.get_alfa_bit(i))
            continue;
        for (size_t j = i + 1; j < nact_; ++j) {
            if (!actBits.get_alfa_bit(j))
                continue;
            std::vector<std::tuple<int, int, double>> ij_couplings;
            for (size_t a = i + 1; a < nact_; ++a) {
                if (a == j or !actBits.get_alfa_bit(a))
                    continue;
                for (size_t b = a + 1; b < nact_; ++b) {
                    if (b == j or !actBits.get_alfa_bit(b))
                        continue;
                    if ((mo_symmetry_[i] ^ mo_symmetry_[j] ^ mo_symmetry_[a] ^ mo_symmetry_[b]) ==
                        0) {
                        double Hijab = as_ints_->tei_aa(i, j, a, b);
                        ij_couplings.push_back(std::make_tuple(a, b, Hijab));
                    }
                }
            }
            if (ij_couplings.size() != 0) {
                aa_couplings_.push_back(std::make_tuple(i, j, 0.0, ij_couplings));
            }
        }
    }
    aa_couplings_size_ = aa_couplings_.size();

    ab_couplings_.clear();
    for (size_t i = 0; i < nact_; ++i) {
        if (!actBits.get_alfa_bit(i))
            continue;
        for (size_t j = 0; j < nact_; ++j) {
            if (!actBits.get_beta_bit(j))
                continue;
            std::vector<std::tuple<int, int, double>> ij_couplings;
            for (size_t a = i + 1; a < nact_; ++a) {
                if (a == i or !actBits.get_alfa_bit(a))
                    continue;
                for (size_t b = 0; b < nact_; ++b) {
                    if (b == j or !actBits.get_beta_bit(b))
                        continue;
                    if ((mo_symmetry_[i] ^ mo_symmetry_[j] ^ mo_symmetry_[a] ^ mo_symmetry_[b]) ==
                        0) {
                        double Hijab = as_ints_->tei_ab(i, j, a, b);
                        ij_couplings.push_back(std::make_tuple(a, b, Hijab));
                    }
                }
            }
            if (ij_couplings.size() != 0) {
                ab_couplings_.push_back(std::make_tuple(i, j, 0.0, ij_couplings));
            }
        }
    }
    ab_couplings_size_ = ab_couplings_.size();

    bb_couplings_.clear();
    for (size_t i = 0; i < nact_; ++i) {
        if (!actBits.get_beta_bit(i))
            continue;
        for (size_t j = i + 1; j < nact_; ++j) {
            if (!actBits.get_beta_bit(j))
                continue;
            std::vector<std::tuple<int, int, double>> ij_couplings;
            for (size_t a = i + 1; a < nact_; ++a) {
                if (a == j or !actBits.get_beta_bit(a))
                    continue;
                for (size_t b = a + 1; b < nact_; ++b) {
                    if (b == j or !actBits.get_beta_bit(b))
                        continue;
                    if ((mo_symmetry_[i] ^ mo_symmetry_[j] ^ mo_symmetry_[a] ^ mo_symmetry_[b]) ==
                        0) {
                        double Hijab = as_ints_->tei_bb(i, j, a, b);
                        ij_couplings.push_back(std::make_tuple(a, b, Hijab));
                    }
                }
            }
            if (ij_couplings.size() != 0) {
                bb_couplings_.push_back(std::make_tuple(i, j, 0.0, ij_couplings));
            }
        }
    }
    bb_couplings_size_ = bb_couplings_.size();
}

std::vector<std::tuple<double, int, int>>
ProjectorCI_HashVec::sym_labeled_orbitals(std::string type) {
    std::vector<std::tuple<double, int, int>> labeled_orb;

    if (type == "RHF" or type == "ROHF" or type == "ALFA") {

        // Create a vector of orbital energy and index pairs
        std::vector<std::pair<double, int>> orb_e;
        int cumidx = 0;
        for (int h = 0; h < nirrep_; ++h) {
            for (int a = 0; a < nactpi_[h]; ++a) {
                orb_e.push_back(
                    std::make_pair(scf_info_->epsilon_a()->get(h, frzcpi_[h] + a), a + cumidx));
            }
            cumidx += nactpi_[h];
        }

        // Create a vector that stores the orbital energy, symmetry, and idx
        for (size_t a = 0; a < nact_; ++a) {
            labeled_orb.push_back(
                std::make_tuple(orb_e[a].first, mo_symmetry_[a], orb_e[a].second));
        }
        // Order by energy, low to high
        std::sort(labeled_orb.begin(), labeled_orb.end());
    }
    if (type == "BETA") {
        // Create a vector of orbital energies and index pairs
        std::vector<std::pair<double, int>> orb_e;
        int cumidx = 0;
        for (int h = 0; h < nirrep_; ++h) {
            for (size_t a = 0, max = nactpi_[h]; a < max; ++a) {
                orb_e.push_back(
                    std::make_pair(scf_info_->epsilon_b()->get(h, frzcpi_[h] + a), a + cumidx));
            }
            cumidx += nactpi_[h];
        }

        // Create a vector that stores the orbital energy, sym, and idx
        for (size_t a = 0; a < nact_; ++a) {
            labeled_orb.push_back(
                std::make_tuple(orb_e[a].first, mo_symmetry_[a], orb_e[a].second));
        }
        std::sort(labeled_orb.begin(), labeled_orb.end());
    }
    return labeled_orb;
}
} // namespace forte
