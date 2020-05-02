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
#include <unordered_map>

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

#include "pci.h"
#include "helpers/timer.h"
#include "sparse_ci/ci_reference.h"
#include "base_classes/state_info.h"
#include "base_classes/rdms.h"

using namespace psi;
using namespace forte::GeneratorType_;

#define USE_HASH 1
#define DO_STATS 0
#define ENFORCE_SYM 1

namespace forte {
#ifdef _OPENMP
#include <omp.h>
bool ProjectorCI::have_omp_ = true;
#else
#define omp_get_max_threads() 1
#define omp_get_thread_num() 0
bool ProjectorCI::have_omp_ = false;
#endif

void combine_hashes(std::vector<det_hash<>>& thread_det_C_map, det_hash<>& dets_C_hash);
void combine_hashes(det_hash<>& dets_C_hash_A, det_hash<>& dets_C_hash_B);
void combine_hashes_into_hash(std::vector<det_hash<>>& thread_det_C_hash, det_hash<>& dets_C_hash);
void copy_hash_to_vec(det_hash<>& dets_C_hash, det_vec& dets, std::vector<double>& C);
void copy_hash_to_vec_order_ref(det_hash<>& dets_C_hash, det_vec& dets, std::vector<double>& C);
void copy_vec_to_hash(det_vec& dets, const std::vector<double>& C, det_hash<>& dets_C_hash);
void scale(std::vector<double>& A, double alpha);
void scale(det_hash<>& A, double alpha);
double normalize(std::vector<double>& C);
double normalize(det_hash<>& dets_C);
double norm(std::vector<double>& C);
double norm(det_hash<>& dets_C);
double dot(det_hash<>& A, det_hash<>& B);
double dot(std::vector<double>& C1, std::vector<double>& C2);
size_t ortho_norm(std::vector<std::vector<double>>& H_n_C, std::vector<double>& norms, Matrix& A,
                  double colinear_threshold);
void add(det_hash<>& A, double beta, det_hash<>& B);
void add(std::vector<double>& a, double k, std::vector<double>& b);
double factorial(int n);
void binomial_coefs(std::vector<double>& coefs, int order, double a, double b);
void Taylor_generator_coefs(std::vector<double>& coefs, int order, double tau, double S);
void Taylor_polynomial_coefs(std::vector<double>& coefs, int order);
void Chebyshev_polynomial_coefs(std::vector<double>& coefs, int order);
void Exp_Chebyshev_generator_coefs(std::vector<double>& coefs, int order, double tau, double range);
void Chebyshev_generator_coefs(std::vector<double>& coefs, int order, double range);
void Wall_Chebyshev_generator_coefs(std::vector<double>& coefs, int order, double range);
void print_polynomial(std::vector<double>& coefs);

void print_vector(const std::vector<double>& C, std::string description) {
    outfile->Printf("\n%s :", description.c_str());
    for (size_t i = 0, C_size = C.size(); i < C_size; i++) {
        outfile->Printf(" %.12lf ", C[i]);
    }
    outfile->Printf("\n");
}

ProjectorCI::ProjectorCI(StateInfo state, size_t nroot, std::shared_ptr<forte::SCFInfo> scf_info,
                         std::shared_ptr<ForteOptions> options,
                         std::shared_ptr<MOSpaceInfo> mo_space_info,
                         std::shared_ptr<ActiveSpaceIntegrals> as_ints)
    : ActiveSpaceMethod(state, nroot, mo_space_info, as_ints), scf_info_(scf_info),
      options_(options), prescreening_tollerance_factor_(1.5), fast_variational_estimate_(false) {
    // Copy the wavefunction information
    startup();
}

std::vector<RDMs> ProjectorCI::rdms(const std::vector<std::pair<size_t, size_t>>& root_list,
                                    int max_rdm_level) {
    std::vector<RDMs> refs;
    // TODO: implement
    throw std::runtime_error("ProjectorCI::rdms is not implemented!");
    return refs;
}

std::vector<RDMs>
ProjectorCI::transition_rdms(const std::vector<std::pair<size_t, size_t>>& root_list,
                             std::shared_ptr<ActiveSpaceMethod> method2, int max_rdm_level) {
    std::vector<RDMs> refs;
    throw std::runtime_error("ProjectorCI::transition_rdms is not implemented!");
    return refs;
}

void ProjectorCI::startup() {
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
    initiator_approx_factor_ = options_->get_double("PCI_INITIATOR_APPROX_FACTOR");
    colinear_threshold_ = options_->get_double("PCI_COLINEAR_THRESHOLD");

    max_guess_size_ = options_->get_int("PCI_MAX_GUESS_SIZE");
    energy_estimate_freq_ = options_->get_int("PCI_ENERGY_ESTIMATE_FREQ");

    adaptive_beta_ = options_->get_bool("PCI_ADAPTIVE_BETA");
    fast_variational_estimate_ = options_->get_bool("PCI_FAST_EVAR");
    do_shift_ = options_->get_bool("PCI_USE_SHIFT");
    use_inter_norm_ = options_->get_bool("PCI_USE_INTER_NORM");
    do_simple_prescreening_ = options_->get_bool("PCI_SIMPLE_PRESCREENING");
    do_dynamic_prescreening_ = options_->get_bool("PCI_DYNAMIC_PRESCREENING");
    do_schwarz_prescreening_ = options_->get_bool("PCI_SCHWARZ_PRESCREENING");
    do_initiator_approx_ = options_->get_bool("PCI_INITIATOR_APPROX");
    do_perturb_analysis_ = options_->get_bool("PCI_PERTURB_ANALYSIS");
    stop_higher_new_low_ = options_->get_bool("PCI_STOP_HIGHER_NEW_LOW");
    chebyshev_order_ = options_->get_int("PCI_CHEBYSHEV_ORDER");
    krylov_order_ = options_->get_int("PCI_KRYLOV_ORDER");
    symm_approx_H_ = options_->get_bool("PCI_SYMM_APPROX_H");
    reference_spawning_ = options_->get_bool("PCI_REFERENCE_SPAWNING");
    if (reference_spawning_ && !symm_approx_H_) {
        symm_approx_H_ = true;
        outfile->Printf("\n\n  Warning! Use of symmetric approximated "
                        "Hamiltonian is enforced by using reference spawning.");
    }

    variational_estimate_ = options_->get_bool("PCI_VAR_ESTIMATE");
    print_full_wavefunction_ = options_->get_bool("PCI_PRINT_FULL_WAVEFUNCTION");

    approx_E_tau_ = 1.0;
    approx_E_S_ = 0.0;

    if (options_->get_str("PCI_GENERATOR") == "LINEAR") {
        generator_ = LinearGenerator;
        generator_description_ = "Linear";
    } else if (options_->get_str("PCI_GENERATOR") == "TROTTER") {
        generator_ = TrotterLinear;
        generator_description_ = "Trotter";
    } else if (options_->get_str("PCI_GENERATOR") == "QUADRATIC") {
        generator_ = QuadraticGenerator;
        generator_description_ = "Quadratic";
    } else if (options_->get_str("PCI_GENERATOR") == "CUBIC") {
        generator_ = CubicGenerator;
        generator_description_ = "Cubic";
    } else if (options_->get_str("PCI_GENERATOR") == "QUARTIC") {
        generator_ = QuarticGenerator;
        generator_description_ = "Quartic";
    } else if (options_->get_str("PCI_GENERATOR") == "POWER") {
        generator_ = PowerGenerator;
        generator_description_ = "Power";
        time_step_ = 1.0;
    } else if (options_->get_str("PCI_GENERATOR") == "OLSEN") {
        generator_ = OlsenGenerator;
        generator_description_ = "Olsen";
        // Make sure that do_shift_ is set to true
        do_shift_ = true;
    } else if (options_->get_str("PCI_GENERATOR") == "DAVIDSON") {
        generator_ = DavidsonLiuGenerator;
        generator_description_ = "Davidson-Liu";
        // Make sure that do_shift_ is set to true
        do_shift_ = true;
    } else if (options_->get_str("PCI_GENERATOR") == "EXP-CHEBYSHEV") {
        generator_ = ExpChebyshevGenerator;
        generator_description_ = "Exp-Chebyshev";
        if (chebyshev_order_ <= 0) {
            outfile->Printf("\n\n  Warning! Chebyshev order %d out of bound, "
                            "automatically adjusted to 5.",
                            chebyshev_order_);
            chebyshev_order_ = 5;
        }
    } else if (options_->get_str("PCI_GENERATOR") == "CHEBYSHEV") {
        generator_ = ChebyshevGenerator;
        generator_description_ = "Chebyshev";
        time_step_ = 1.0;
        if (chebyshev_order_ <= 0) {
            outfile->Printf("\n\n  Warning! Chebyshev order %d out of bound, "
                            "automatically adjusted to 5.",
                            chebyshev_order_);
            chebyshev_order_ = 5;
        }
    } else if (options_->get_str("PCI_GENERATOR") == "WALL-CHEBYSHEV") {
        generator_ = WallChebyshevGenerator;
        generator_description_ = "Wall-Chebyshev";
        time_step_ = 1.0;
        if (chebyshev_order_ <= 0) {
            outfile->Printf("\n\n  Warning! Chebyshev order %d out of bound, "
                            "automatically adjusted to 5.",
                            chebyshev_order_);
            chebyshev_order_ = 5;
        }
    } else if (options_->get_str("PCI_GENERATOR") == "LANCZOS") {
        generator_ = LanczosGenerator;
        generator_description_ = "Lanczos";
        time_step_ = 1.0;
        if (krylov_order_ <= 0) {
            outfile->Printf("\n\n  Warning! Krylov order %d out of bound, "
                            "automatically adjusted to 5.",
                            krylov_order_);
            krylov_order_ = 5;
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
    }

    num_threads_ = omp_get_max_threads();
}

void ProjectorCI::print_info() {
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
        {"Prescreening tollerance factor", prescreening_tollerance_factor_},
        {"Energy estimate tollerance", energy_estimate_threshold_}};

    std::vector<std::pair<std::string, std::string>> calculation_info_string{
        {"Generator type", generator_description_},
        {"Adaptive time step", adaptive_beta_ ? "YES" : "NO"},
        {"Shift the energy", do_shift_ ? "YES" : "NO"},
        {"Use intermediate normalization", use_inter_norm_ ? "YES" : "NO"},
        {"Prescreen spawning", do_simple_prescreening_ ? "YES" : "NO"},
        {"Dynamic prescreening", do_dynamic_prescreening_ ? "YES" : "NO"},
        {"Schwarz prescreening", do_schwarz_prescreening_ ? "YES" : "NO"},
        {"Initiator approximation", do_initiator_approx_ ? "YES" : "NO"},
        {"Fast variational estimate", fast_variational_estimate_ ? "YES" : "NO"},
        {"Result perturbation analysis", do_perturb_analysis_ ? "YES" : "NO"},
        {"Using OpenMP", have_omp_ ? "YES" : "NO"},
    };
    //    {"Number of electrons",nel},
    //    {"Number of correlated alpha electrons",nalpha_},
    //    {"Number of correlated beta electrons",nbeta_},
    //    {"Number of restricted docc electrons",rdoccpi_.sum()},
    //    {"Charge",charge},
    //    {"Multiplicity",multiplicity},

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

void print_polynomial(std::vector<double>& coefs);

double ProjectorCI::estimate_high_energy() {
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

    // Begin debug
    //    outfile->Printf("\n\n  Estimating high energy, size of obt_energies:%d",
    //    obt_energies.size());
    //    for (auto item : obt_energies)
    //        outfile -> Printf("  %lf", item);
    //    outfile->Printf("\n  Printing oei and tei_aa tei_ab for obtI = 4:");
    //    int obtI = 4;
    //    outfile->Printf("\n  oei_a(%d,%d) = %lf", obtI, obtI,as_ints_->oei_a(obtI, obtI));
    //    for (int p = 0; p < nact_; ++p) {
    //        if (bits_[p]) {
    //            outfile->Printf("\n  tei_aa(%d,%d,%d,%d) = %lf",obtI, p, obtI, p,
    //            as_ints_->tei_aa(obtI, p, obtI, p));
    //        }
    //        if (bits_[nact_ + p]) {
    //            outfile->Printf("\n  tei_ab(%d,%d,%d,%d) = %lf",obtI, p, obtI, p,
    //            as_ints_->tei_ab(obtI, p, obtI, p));
    //        }
    //    }
    // End debug

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

void ProjectorCI::convergence_analysis() {
    estimate_high_energy();
    compute_characteristic_function();
    print_characteristic_function();
}

void ProjectorCI::compute_characteristic_function() {
    shift_ = (lambda_h_ + lambda_1_) / 2.0;
    range_ = (lambda_h_ - lambda_1_) / 2.0;
    switch (generator_) {
    case PowerGenerator:
        cha_func_coefs_.clear();
        cha_func_coefs_.push_back(0.0);
        cha_func_coefs_.push_back(-1.0);
        break;
    case LinearGenerator:
        Taylor_generator_coefs(cha_func_coefs_, 1, time_step_, range_);
        break;
    case QuadraticGenerator:
        Taylor_generator_coefs(cha_func_coefs_, 2, time_step_, range_);
        break;
    case CubicGenerator:
        Taylor_generator_coefs(cha_func_coefs_, 3, time_step_, range_);
        break;
    case QuarticGenerator:
        Taylor_generator_coefs(cha_func_coefs_, 4, time_step_, range_);
        break;
    case ExpChebyshevGenerator:
        Exp_Chebyshev_generator_coefs(cha_func_coefs_, chebyshev_order_, time_step_, range_);
        break;
    case ChebyshevGenerator:
        Chebyshev_generator_coefs(cha_func_coefs_, chebyshev_order_, range_);
        break;
    case WallChebyshevGenerator:
        Wall_Chebyshev_generator_coefs(cha_func_coefs_, chebyshev_order_, range_);
    default:
        break;
    }
}

void ProjectorCI::print_characteristic_function() {
    outfile->Printf("\n\n  ==> Characteristic Function <==");
    print_polynomial(cha_func_coefs_);
    outfile->Printf("\n    with tau = %e, shift = %.12f, range = %.12f", time_step_, shift_,
                    range_);
    outfile->Printf("\n    Initial guess: lambda_1= %s%.12f", lambda_1_ >= 0.0 ? " " : "",
                    lambda_1_ + nuclear_repulsion_energy_);
    outfile->Printf("\n    Est. Highest eigenvalue= %s%.12f", lambda_h_ >= 0.0 ? " " : "",
                    lambda_h_ + nuclear_repulsion_energy_);
}

double factorial(int n);

void binomial_coefs(std::vector<double>& coefs, int order, double a, double b);

void Polynomial_generator_coefs(std::vector<double>& coefs, std::vector<double>& poly_coefs,
                                double a, double b);

void Taylor_polynomial_coefs(std::vector<double>& coefs, int order) {
    coefs.clear();
    for (int i = 0; i <= order; i++) {
        coefs.push_back(1.0 / factorial(i));
    }
}

void Taylor_generator_coefs(std::vector<double>& coefs, int order, double tau, double S) {
    coefs.clear();
    std::vector<double> poly_coefs;
    Taylor_polynomial_coefs(poly_coefs, order);
    Polynomial_generator_coefs(coefs, poly_coefs, -tau, -tau * S);
    //    coefs.clear();
    //    for (int i=0; i <= order; i++) {
    //        coefs.push_back(0.0);
    //        for (int j=0; j <= i; j++) {
    //            coefs[j] += 1.0/(factorial(j)*factorial(i-j))*pow(-tau,
    //            i)*pow(-S, i-j);
    //        }
    //    }
}

void Chebyshev_polynomial_coefs(std::vector<double>& coefs, int order);

void Exp_Chebyshev_generator_coefs(std::vector<double>& coefs, int order, double tau,
                                   double range) {
    coefs.clear();
    std::vector<double> poly_coefs;
    for (int i = 0; i <= order; i++) {
        poly_coefs.push_back(0.0);
    }

    for (int i = 0; i <= order; i++) {
        std::vector<double> chbv_poly_coefs;
        Chebyshev_polynomial_coefs(chbv_poly_coefs, i);
        //        outfile->Printf("\n\n  chebyshev poly in step %d", i);
        //        print_polynomial(chbv_poly_coefs);
        for (int j = 0; j <= i; j++) {
            poly_coefs[j] += (i == 0 ? 1.0 : 2.0) * boost::math::cyl_bessel_i(i, tau * range) *
                             chbv_poly_coefs[j];
        }
        //        outfile->Printf("\n\n  propagate poly in step %d", i);
        //        print_polynomial(poly_coefs);
    }
    Polynomial_generator_coefs(coefs, poly_coefs, -1.0 / range, 0.0);
}

void Chebyshev_generator_coefs(std::vector<double>& coefs, int order, double range) {
    coefs.clear();
    std::vector<double> poly_coefs;
    Chebyshev_polynomial_coefs(poly_coefs, order);

    Polynomial_generator_coefs(coefs, poly_coefs, -1.0 / range, 0.0);
}

void Wall_Chebyshev_generator_coefs(std::vector<double>& coefs, int order, double range);

double ProjectorCI::compute_energy() {
    timer_on("PCI:Energy");
    local_timer t_apici;
    old_max_one_HJI_ = 1e100;
    new_max_one_HJI_ = 1e100;
    old_max_two_HJI_ = 1e100;
    new_max_two_HJI_ = 1e100;

    // Increase the root counter (ground state = 0)
    current_root_ += 1;
    lastLow = 0.0;
    previous_go_up = false;

    outfile->Printf("\n\n\t  ---------------------------------------------------------");
    outfile->Printf("\n\t      Adaptive Path-Integral Full Configuration Interaction");
    outfile->Printf("\n\t         by Francesco A. Evangelista and Tianyuan Zhang");
    outfile->Printf("\n\t                      version Jun. 4 2017");
    outfile->Printf("\n\t                    %4d thread(s) %s", num_threads_,
                    have_omp_ ? "(OMP)" : "");
    outfile->Printf("\n\t  ---------------------------------------------------------");

    // Print a summary of the options
    print_info();

    /// A vector of determinants in the P space
    det_vec dets;
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

    // Compute the initial guess
    outfile->Printf("\n\n  ==> Initial Guess <==");
    double var_energy = initial_guess(dets, C);
    double proj_energy = var_energy;

    print_wfn(dets, C);
    det_hash<> old_space_map;
    for (size_t I = 0; I < dets.size(); ++I) {
        old_space_map[dets[I]] = C[I];
    }

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
                        "--------------------");
        outfile->Printf("\n    Steps  Beta/Eh      Ndets     Proj. Energy/Eh   "
                        "  dEp/dt      Var. Energy/Eh      dEp/dt      Approx. "
                        "Energy/Eh   dEv/dt");
        outfile->Printf("\n  "
                        "------------------------------------------------------"
                        "------------------------------------------------------"
                        "--------------------");
    } else {
        outfile->Printf("\n\n  "
                        "------------------------------------------------------"
                        "------------------------------------------");
        outfile->Printf("\n    Steps  Beta/Eh      Ndets     Proj. Energy/Eh   "
                        "  dEp/dt      Approx. Energy/Eh   dEv/dt");
        outfile->Printf("\n  "
                        "------------------------------------------------------"
                        "------------------------------------------");
    }

    int maxcycle = maxiter_;
    double old_var_energy = var_energy;
    double old_proj_energy = proj_energy;
    double beta = 0.0;
    bool converged = false;

    schwarz_succ_ = 0;
    schwarz_total_ = 0;

    approx_E_flag_ = true;

    for (int cycle = 0; cycle < maxcycle; ++cycle) {
        iter_ = cycle;

        timer_on("PCI:Step");
        if (use_inter_norm_) {
            auto minmax_C = std::minmax_element(C.begin(), C.end());
            double min_C_abs = std::fabs(*minmax_C.first);
            double max_C = *minmax_C.second;
            max_C = max_C > min_C_abs ? max_C : min_C_abs;
            propagate(generator_, dets, C, time_step_, spawning_threshold_ * max_C, shift_);
        } else {
            propagate(generator_, dets, C, time_step_, spawning_threshold_, shift_);
        }
        timer_off("PCI:Step");
        if (generator_ == DavidsonLiuGenerator)
            break;

        // Orthogonalize this solution with respect to the previous ones
        timer_on("PCI:Ortho");
        if (current_root_ > 0) {
            orthogonalize(dets, C, solutions_);
        }
        timer_off("PCI:Ortho");

        // Compute the energy and check for convergence
        if (cycle % energy_estimate_freq_ == 0) {
            approx_E_flag_ = true;
            timer_on("PCI:<E>");
            std::map<std::string, double> results = estimate_energy(dets, C);
            timer_off("PCI:<E>");

            proj_energy = results["PROJECTIVE ENERGY"];

            double proj_energy_gradient =
                (proj_energy - old_proj_energy) / (time_step_ * energy_estimate_freq_);
            double approx_energy_gradient =
                (approx_energy_ - old_approx_energy_) / (time_step_ * energy_estimate_freq_);
            if (cycle == 0)
                approx_energy_gradient = 10.0 * e_convergence_ + 1.0;

            switch (generator_) {
            case LanczosGenerator:
                outfile->Printf("\n%9d %8d %10zu %20.12f %10.3e", cycle, krylov_order_, C.size(),
                                proj_energy, proj_energy_gradient);
                break;
            case DLGenerator:
                outfile->Printf("\n%9d %8d %10zu %20.12f %10.3e", cycle, current_davidson_iter_,
                                C.size(), proj_energy, proj_energy_gradient);
                break;
            default:
                outfile->Printf("\n%9d %8.2f %10zu %20.12f %10.3e", cycle, beta, C.size(),
                                proj_energy, proj_energy_gradient);
                break;
            }

            //            if (generator_ != LanczosGenerator) {
            //                outfile->Printf("\n%9d %8.2f %10zu %20.12f
            //                %10.3e",cycle,beta,C.size(),
            //                                proj_energy,proj_energy_gradient);
            //            } else {
            //                outfile->Printf("\n%9d %8d %10zu %20.12f
            //                %10.3e",cycle,krylov_order_,C.size(),
            //                                proj_energy,proj_energy_gradient);
            //            }

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

    //    det_hash<> dets_C_hash;
    //    if (symm_approx_H_) {
    //        apply_tau_H_symm(approx_E_tau_,spawning_threshold_,dets,C,dets_C_hash,
    //        approx_E_S_);
    //    } else {
    //        apply_tau_H(approx_E_tau_,spawning_threshold_,dets,C,dets_C_hash,
    //        approx_E_S_);
    //    }
    //    dets_C_hash.clear();
    if (variational_estimate_) {
        outfile->Printf("\n  "
                        "------------------------------------------------------"
                        "------------------------------------------------------"
                        "--------------------");
    } else {
        outfile->Printf("\n  "
                        "------------------------------------------------------"
                        "------------------------------------------");
    }

    outfile->Printf("\n\n  Calculation %s", converged ? "converged." : "did not converge!");

    if (do_shift_) {
        outfile->Printf("\n\n  Shift applied during iteration, the "
                        "characteristic function may change every step.\n  "
                        "Characteristic function at last step:");
        print_characteristic_function();
    }

    //    for (size_t i = 0, i_max = statistic_vec.size(); i < i_max; i++) {
    //        statistic_vec[i].print();
    //        outfile->Printf(",%zu,%zu", statistic_hash[statistic_vec[i]],
    //        Determinant::Hash()(statistic_vec[i]));
    //    }

    timer_on("PCI:<E>end_v");

    if (fast_variational_estimate_) {
        var_energy = estimate_var_energy_sparse(dets, C, 1.0e-14);
    } else {
        var_energy = estimate_var_energy(dets, C, 1.0e-14);
    }
    timer_off("PCI:<E>end_v");

    psi::Process::environment.globals["PCI ENERGY"] = var_energy;

    outfile->Printf("\n\n  ==> Post-Iterations <==\n");
    outfile->Printf("\n  * Projector-CI Variational Energy     = %18.12f Eh", 1, var_energy);
    outfile->Printf("\n  * Projector-CI Projective  Energy     = %18.12f Eh", 1, proj_energy);

    outfile->Printf("\n\n  * Projector-CI Approximate Energy     = %18.12f Eh", 1, approx_energy_);
    outfile->Printf("\n  * 1st order perturbation   Energy     = %18.12f Eh", 1,
                    var_energy - approx_energy_);

    outfile->Printf("\n  * Projector-CI Var. Corr.  Energy     = %18.12f Eh", 1,
                    var_energy - as_ints_->energy(reference_determinant_) -
                        nuclear_repulsion_energy_ - as_ints_->scalar_energy());

    //    double pfError = estimate_path_filtering_error(dets, C,
    //    spawning_threshold_);
    //    outfile->Printf("\n\n  * Projector-CI Path-filtering Error  <= %18.12f
    //    Eh",1,pfError);

    if (do_perturb_analysis_) {
        double error_2nd_perturb_sub, error_2nd_perturb_full;
        std::tie(error_2nd_perturb_sub, error_2nd_perturb_full) =
            estimate_perturbation(dets, C, spawning_threshold_);
        outfile->Printf("\n  * 2nd order perturbation est. Energy = %18.12f Eh", 1,
                        error_2nd_perturb_sub);
    }

    outfile->Printf("\n\n  * Size of CI space                    = %zu", C.size());

    if (do_schwarz_prescreening_) {
        outfile->Printf("\n  * Schwarz prescreening total attempt = %zu", schwarz_total_);
        outfile->Printf("\n  * Schwarz prescreening succeed       = %zu", schwarz_succ_);
    }

    outfile->Printf("\n\n  %s: %f s", "Projector-CI (bitset) ran in  ", t_apici.get());

    if (print_full_wavefunction_) {
        print_wfn(dets, C, C.size());
    } else {
        print_wfn(dets, C);
    }

    if (current_root_ < nroot_ - 1) {
        save_wfn(dets, C, solutions_);
    }

    if (post_diagonalization_) {
        outfile->Printf("\n\n  ==> Post-Diagonalization <==\n");
        timer_on("PCI:Post_Diag");
        //        sparse_solver.diagonalize_hamiltonian(dets,apfci_evals,apfci_evecs,nroot_,DavidsonLiuList);
        psi::SharedMatrix apfci_evecs(new psi::Matrix("Eigenvectors", C.size(), nroot_));
        psi::SharedVector apfci_evals(new Vector("Eigenvalues", nroot_));

        sparse_solver.diagonalize_hamiltonian(dets, apfci_evals, apfci_evecs, nroot_,
                                              wavefunction_multiplicity_, diag_method_);

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

        if (print_full_wavefunction_) {
            print_wfn(dets, diag_C, diag_C.size());
        } else {
            print_wfn(dets, diag_C);
        }
    }

    energies_.push_back(var_energy);

    delete[] pqpq_aa_;
    delete[] pqpq_ab_;
    delete[] pqpq_bb_;

    timer_off("PCI:Energy");
    return var_energy;
}

bool ProjectorCI::converge_test() {
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

double ProjectorCI::initial_guess(det_vec& dets, std::vector<double>& C) {
    det_hash<> dets_C;

    // Do one time step starting from the reference determinant
    Determinant bs_det(reference_determinant_);
    det_vec guess_dets{bs_det};

    apply_tau_H(time_step_, initial_guess_spawning_threshold_, guess_dets, {1.0}, dets_C, 0.0);

    // Save the list of determinants
    copy_hash_to_vec(dets_C, dets, C);

    size_t guess_size = dets.size();
    if (guess_size > max_guess_size_) {
        // Consider the 1000 largest contributions
        std::vector<std::pair<double, size_t>> det_weight;
        for (size_t I = 0, max_I = C.size(); I < max_I; ++I) {
            det_weight.push_back(std::make_pair(std::fabs(C[I]), I));
            // dets[I].print();
        }
        std::sort(det_weight.begin(), det_weight.end());
        std::reverse(det_weight.begin(), det_weight.end());

        det_vec new_dets;
        for (size_t sI = 0; sI < max_guess_size_; ++sI) {
            size_t I = det_weight[sI].second;
            new_dets.push_back(dets[I]);
        }
        dets = new_dets;
        C.resize(guess_size);
        guess_size = dets.size();
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
    sparse_solver.diagonalize_hamiltonian(dets, evals, evecs, nroot_, wavefunction_multiplicity_,
                                          DLSolver);
    double var_energy =
        evals->get(current_root_) + nuclear_repulsion_energy_ + as_ints_->scalar_energy();
    outfile->Printf("\n\n  Initial guess energy (variational) = %20.12f Eh (root = %d)", var_energy,
                    current_root_ + 1);
    lambda_1_ = evals->get(current_root_) + as_ints_->scalar_energy();

    // Copy the ground state eigenvector
    for (size_t I = 0; I < guess_size; ++I) {
        C[I] = evecs->get(I, current_root_);
    }
    return var_energy;
}

void ProjectorCI::propagate(GeneratorType generator, det_vec& dets, std::vector<double>& C,
                            double tau, double spawning_threshold, double S) {
    // Reset prescreening boundary
    if (do_simple_prescreening_) {
        new_max_one_HJI_ = 0.0;
        new_max_two_HJI_ = 0.0;
    }

    switch (generator) {
    case WallChebyshevGenerator:
        propagate_wallCh(dets, C, spawning_threshold);
        break;
    case LanczosGenerator:
        propagate_Lanczos(dets, C, spawning_threshold);
        break;
    case DLGenerator:
        propagate_DL(dets, C, spawning_threshold);
        break;
    case ChebyshevGenerator:
        propagate_Chebyshev(dets, C, spawning_threshold);
        break;
    case LinearGenerator:
        propagate_Linear(dets, C, tau, spawning_threshold, lambda_1_);
    case TrotterLinear:
        propagate_Trotter_linear(dets, C, tau, spawning_threshold, S);
        break;
    default:
        propagate_Polynomial(dets, C, cha_func_coefs_, spawning_threshold);
        break;
    }

    //    // Evaluate (1-beta H) |C>
    //    if (generator == LinearGenerator){
    //        propagate_first_order(dets,C,tau,spawning_threshold,S);
    //    }else if (generator == TrotterLinear){
    //        propagate_Trotter_linear(dets,C,tau,spawning_threshold,S);
    //    }else if (generator == ChebyshevGenerator){
    //        propagate_Chebyshev(dets,C,tau,spawning_threshold,S);
    //    }else if (generator == QuadraticGenerator){
    //        propagate_Taylor(2,dets,C,tau,spawning_threshold,S);
    //    }else if (generator == CubicGenerator){
    //        propagate_Taylor(3,dets,C,tau,spawning_threshold,S);
    //    }else if (generator == QuarticGenerator){
    //        propagate_Taylor(4,dets,C,tau,spawning_threshold,S);
    //    }else if (generator == PowerGenerator){
    //        propagate_power(dets,C,spawning_threshold,S);
    //    }else if (generator == OlsenGenerator){
    //        propagate_Olsen(dets,C,spawning_threshold,S);
    //    }else if (generator == DavidsonLiuGenerator){
    //        propagate_DavidsonLiu(dets,C,spawning_threshold);
    //    }

    // Update prescreening boundary
    if (do_simple_prescreening_) {
        old_max_one_HJI_ = new_max_one_HJI_;
        old_max_two_HJI_ = new_max_two_HJI_;
    }
    normalize(C);
}

void ProjectorCI::propagate_wallCh(det_vec& dets, std::vector<double>& C,
                                   double spawning_threshold) {

    // A map that contains the pair (determinant,coefficient)
    const double PI = 2 * acos(0.0);
    det_hash<> dets_C_hash;
    std::vector<double> ref_C;
    if (reference_spawning_) {
        ref_C = C;
    }
    for (int i = chebyshev_order_; i > 0; i--) {
        //        outfile->Printf("\nCurrent root:%.12lf",range_ * root +
        //        shift_);
        //        apply_tau_H(-1.0/range_,spawning_threshold,dets,C,dets_C_hash,
        //        range_ * root + shift_);
        double root = -cos(((double)i) * PI / (chebyshev_order_ + 0.5));
        if (reference_spawning_) {
            apply_tau_H_ref_C_symm(-1.0, spawning_threshold, dets, C, ref_C, dets_C_hash,
                                   range_ * root + shift_);
            copy_hash_to_vec_order_ref(dets_C_hash, dets, C);
        } else {
            if (symm_approx_H_) {
                apply_tau_H_symm(-1.0, spawning_threshold, dets, C, dets_C_hash,
                                 range_ * root + shift_);
            } else {
                apply_tau_H(-1.0, spawning_threshold, dets, C, dets_C_hash, range_ * root + shift_);
            }
            copy_hash_to_vec(dets_C_hash, dets, C);
        }

        dets_C_hash.clear();
        normalize(C);
    }
}

void ProjectorCI::propagate_Lanczos(det_vec& dets, std::vector<double>& C,
                                    double spawning_threshold) {
    size_t ref_size = C.size();
    size_t krylov_order = krylov_order_ < ref_size + 1 ? krylov_order_ : ref_size + 1;
    std::vector<std::vector<double>> H_n_C(krylov_order + 1);
    H_n_C[0] = C;
    det_hash<> dets_C_hash;
    std::vector<double> norms(krylov_order);
    //    apply_tau_H_ref_C_symm(1.0, spawning_threshold, dets, H_n_C[0], C,
    //    dets_C_hash, 0.0);
    //    apply_tau_H_symm(1.0,spawning_threshold,dets,C,dets_C_hash, 0.0);
    //    for (det_hash_it it = dets_C_hash.begin(), endit = dets_C_hash.end();
    //    it != endit; ++it){
    //        outfile -> Printf(" %lf ", it->second);
    //    }

    //    copy_hash_to_vec_order_ref(dets_C_hash, dets, H_n_C[1]);

    //    norms[0] = normalize(H_n_C[1]);

    for (size_t i = 1; i <= krylov_order; i++) {
        //        for (int k = 0; k < H_n_C[i-1].size(); k++) {
        //            outfile -> Printf(" %lf ", H_n_C[i-1][k]);
        //        }
        dets_C_hash.clear();

        if (reference_spawning_) {
            apply_tau_H_ref_C_symm(1.0, spawning_threshold, dets, H_n_C[i - 1], C, dets_C_hash,
                                   0.0);
        } else if (symm_approx_H_) {
            apply_tau_H_symm(1.0, spawning_threshold, dets, H_n_C[i - 1], dets_C_hash, 0.0);
        } else {
            apply_tau_H(1.0, spawning_threshold, dets, H_n_C[i - 1], dets_C_hash, 0.0);
        }

        copy_hash_to_vec_order_ref(dets_C_hash, dets, H_n_C[i]);
        norms[i - 1] = normalize(H_n_C[i]);
    }

    //    for (int i = 0; i <= krylov_order; i++) {
    //        print_vector(H_n_C[i], "H_n_C["+std::to_string(i)+"]");
    //    }

    //    print_vector(norms, "Norms: ");

    //// Orthogonal solver BEGIN
    Matrix A(krylov_order, krylov_order);

    for (size_t i = 0; i < krylov_order; i++) {
        for (size_t j = i; j < krylov_order; j++) {
            double dotIJ = norms[j] * dot(H_n_C[i], H_n_C[j + 1]);
            A.set(i, j, dotIJ);
            A.set(j, i, dotIJ);
        }
    }

    //    outfile -> Printf("\n  A:");
    //    A.print();

    size_t current_order = ortho_norm(H_n_C, norms, A, colinear_threshold_);

    //    for (int i = 0; i <= krylov_order; i++) {
    //        print_vector(H_n_C[i], "H_n_C["+std::to_string(i)+"]");
    //    }

    if (current_order < krylov_order)
        outfile->Printf("\n  Near linear dependency spotted, reduced krylov_order to %d",
                        current_order);

    psi::SharedMatrix H(new psi::Matrix(current_order, current_order));

    for (size_t i = 0; i < current_order; i++) {
        for (size_t j = 0; j < current_order; j++) {
            H->set(i, j, A.get(i, j));
        }
    }
    psi::SharedMatrix evecs(new psi::Matrix(current_order, current_order));
    psi::SharedVector eigs(new Vector(current_order));
    H->diagonalize(evecs, eigs);

    //    outfile -> Printf("\n  H:");
    //    H->print();
    //    outfile -> Printf("\n  evecs:");
    //    evecs->print();
    //    outfile -> Printf("\n  eigs:");
    //    eigs->print();

    int ground_index = 0;
    double abs_ground_coef = std::fabs(evecs->get(0, ground_index));
    for (size_t i = 1; i < current_order; i++) {
        if (std::fabs(evecs->get(0, i)) > abs_ground_coef) {
            abs_ground_coef = std::fabs(evecs->get(0, i));
            ground_index = i;
        }
    }

    while (ground_index != 0) {
        current_order -= ground_index;
        H.reset(new psi::Matrix(current_order, current_order));
        for (size_t i = 0; i < current_order; i++) {
            for (size_t j = 0; j < current_order; j++) {
                H->set(i, j, A.get(i, j));
            }
        }
        evecs.reset(new psi::Matrix(current_order, current_order));
        eigs.reset(new Vector(current_order));
        H->diagonalize(evecs, eigs);

        //        outfile -> Printf("\nRebuilt  H:");
        //        H->print();
        //        outfile -> Printf("\nRebuilt  evecs:");
        //        evecs->print();
        //        outfile -> Printf("\nRebuilt  eigs:");
        //        eigs->print();

        ground_index = 0;
        abs_ground_coef = std::fabs(evecs->get(0, ground_index));
        for (size_t i = 1; i < current_order; i++) {
            if (std::fabs(evecs->get(0, i)) > abs_ground_coef) {
                abs_ground_coef = std::fabs(evecs->get(0, i));
                ground_index = i;
            }
        }
    }
    if (current_order <= 1) {
        outfile->Printf("\n\n  Current Krylov order <= 1 !!!");
        abort();
    }

    scale(C, evecs->get(0, 0));
    C.resize(dets.size());
    for (size_t i = 1; i < current_order; i++) {
        for (size_t j = 0, H_n_C_i_size = H_n_C[i].size(); j < H_n_C_i_size; j++) {
            C[j] += evecs->get(i, 0) * H_n_C[i][j];
        }
    }
    krylov_order_ = current_order;

    //// Orthogonal solver BEGIN

    //// print Hamiltonian
    //    outfile->Printf("\n[");
    //    for (int i = 0; i < dets.size(); i++) {
    //        outfile->Printf("[");
    //        for (int j = 0; j < dets.size(); j++) {
    //            outfile->Printf("%.12lf, ", dets[i].slater_rules(dets[j]));
    //        }
    //        outfile->Printf("],\n");
    //    }
    //    outfile->Printf("]\n");

    //    for (int i = 0; i <= krylov_order; i++) {
    //        print_vector(H_n_C[i], "H_n_C["+std::to_string(i)+"]");
    //    }
    //    outfile -> Printf("\n  Norms: ");
    //    for (int i = 0; i < krylov_order; i++) {
    //        outfile -> Printf(" %lf ", norms[i]);
    //    }
    //    outfile -> Printf("\n");

    //// Diagonalize overlap matrix BEGIN
    //    psi::SharedMatrix A(new psi::Matrix (krylov_order, krylov_order));
    //    psi::SharedMatrix B(new psi::Matrix (krylov_order, krylov_order));

    //    for (int i = 0; i < krylov_order; i++) {
    //        for (int j = i; j < krylov_order; j++) {
    //            double dotIJ = dot(H_n_C[i], H_n_C[j]);
    //            B->set(i,j, dotIJ);
    //            B->set(j,i, dotIJ);
    //        }
    //    }

    //    for (int i = 0; i < krylov_order; i++) {
    //        for (int j = i; j < krylov_order; j++) {
    //            double dotIJ = norms[j] * dot(H_n_C[i], H_n_C[j+1]);
    //            A->set(i,j, dotIJ);
    //            A->set(j,i, dotIJ);
    //        }
    //    }
    //    A->print();
    //    B->print();

    //    psi::SharedMatrix evecs(new psi::Matrix (krylov_order, krylov_order));
    //    psi::SharedVector evals(new Vector(krylov_order));
    //    A->diagonalize(B, evecs, evals);

    //    evals -> print();
    //    evecs -> print();
    //    A -> print();
    //    krylov_order = 2;
    //// Diagonalize overlap matrix END

    //// Generalized eigenvalue problem solver BEGIN
    //    Matrix t(krylov_order, krylov_order);
    //    Matrix m(krylov_order, krylov_order);

    //    for (int i = 0; i < krylov_order; i++) {
    //        for (int j = i; j < krylov_order; j++) {
    //            double dotIJ = dot(H_n_C[i], H_n_C[j]);
    //            m.set(i,j, dotIJ);
    //            m.set(j,i, dotIJ);
    //        }
    //    }

    //    for (int i = 0; i < krylov_order; i++) {
    //        for (int j = i; j < krylov_order; j++) {
    //            double dotIJ = norms[j] * dot(H_n_C[i], H_n_C[j+1]);
    //            t.set(i,j, dotIJ);
    //            t.set(j,i, dotIJ);
    //        }
    //    }

    ////    t.print();
    ////    m.print();

    //    Matrix S_evecs(krylov_order, krylov_order);
    //    Vector S_eigs(krylov_order);
    //    m.diagonalize(S_evecs, S_eigs, 0);
    ////    m.print();
    ////    S_evecs.print();
    ////    S_eigs.print();
    ////    outfile->Printf("\n E-value of overlap matrix : %.2e",
    /// S_eigs.get(0));
    ////    for (int i = 1; i < krylov_order; i++) {
    ////        outfile->Printf(" %.2e", S_eigs.get(i));
    ////    }

    //    int lwork = 3*krylov_order;
    //    double *work = new double[lwork];
    //    psi::SharedVector evals(new Vector(krylov_order));

    //    int err = C_DSYGV(1, 'V', 'U',
    //                      krylov_order, t.get_pointer(0),
    //                      krylov_order, m.get_pointer(0),
    //                      krylov_order, evals->pointer(0),
    //                      work, lwork);
    //    if (err != 0) {
    //        if (err < 0) {
    //            outfile->Printf( "Matrix::diagonalize with metric: C_DSYGV:
    //            argument %d has invalid parameter.\n", -err);

    //            abort();
    //        }
    //        if (err > 0) {
    //            outfile->Printf( "Matrix::diagonalize with metric: C_DSYGV:
    //            error value: %d\n", err);

    //            abort();
    //        }
    //    }
    //    delete[] work;

    ////    t.print();
    ////    m.print();
    ////    evals->print();

    //    scale(C, t.get(0,0,0));
    //    C.resize(dets.size());
    //    for (int i = 1; i < krylov_order; i++) {
    //        for (int j = 0; j < H_n_C[i].size(); j++) {
    //            C[j] += t.get(0,0,i) * H_n_C[i][j];
    //        }
    //    }
    //// Generalized eigenvalue problem solver END
}

void ProjectorCI::propagate_DL(det_vec& dets, std::vector<double>& C, double spawning_threshold) {
    size_t ref_size = C.size();
    std::vector<std::vector<double>> b_vec(davidson_subspace_per_root_);
    std::vector<std::vector<double>> sigma_vec(davidson_subspace_per_root_);
    std::vector<double> alpha_vec(davidson_subspace_per_root_);
    psi::SharedMatrix A(new psi::Matrix(davidson_subspace_per_root_, davidson_subspace_per_root_));
    b_vec[0] = C;
    det_hash<> dets_C_hash;
    if (reference_spawning_) {
        apply_tau_H_ref_C_symm(1.0, spawning_threshold, dets, b_vec[0], C, dets_C_hash, 0.0);
    } else if (symm_approx_H_) {
        apply_tau_H_symm(1.0, spawning_threshold, dets, b_vec[0], dets_C_hash, 0.0);
    } else {
        apply_tau_H(1.0, spawning_threshold, dets, b_vec[0], dets_C_hash, 0.0);
    }
    copy_hash_to_vec_order_ref(dets_C_hash, dets, sigma_vec[0]);
    if (ref_size <= 1) {
        C = sigma_vec[0];
        outfile->Printf("\nDavidson break because the reference space have "
                        "only 1 determinant.");
        current_davidson_iter_ = 1;
        return;
    }

    A->set(0, 0, dot(b_vec[0], sigma_vec[0]));

    size_t dets_size = dets.size();
    std::vector<double> diag_vec(dets_size);
    for (size_t i = 0; i < dets_size; i++) {
        diag_vec[i] = as_ints_->energy(dets[i]) + as_ints_->scalar_energy();
    }

    double lambda = A->get(0, 0);
    alpha_vec[0] = 1.0;
    std::vector<double> delta_vec(dets_size, 0.0);
    int current_order = 1;

    int i = 1;
    for (i = 1; i < max_Davidson_iter_; i++) {

        for (int k = 0; k < current_order; k++) {
            for (int j = 0, jmax = b_vec[k].size(); j < jmax; j++) {
                delta_vec[j] += alpha_vec[k] * (sigma_vec[k][j] - lambda * b_vec[k][j]);
            }
        }
        for (size_t j = 0; j < dets_size; j++) {
            delta_vec[j] /= lambda - diag_vec[j];
        }

        normalize(delta_vec);
        for (int m = 0; m < current_order; m++) {
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

        dets_C_hash.clear();
        if (reference_spawning_) {
            apply_tau_H_ref_C_symm(1.0, spawning_threshold, dets, b_vec[current_order], C,
                                   dets_C_hash, 0.0);
        } else if (symm_approx_H_) {
            apply_tau_H_symm(1.0, spawning_threshold, dets, b_vec[current_order], dets_C_hash, 0.0);
        } else {
            apply_tau_H(1.0, spawning_threshold, dets, b_vec[current_order], dets_C_hash, 0.0);
        }
        copy_hash_to_vec_order_ref(dets_C_hash, dets, sigma_vec[current_order]);
        for (int m = 0; m < current_order; m++) {
            double b_dot_sigma_m = dot(b_vec[current_order], sigma_vec[m]);
            A->set(current_order, m, b_dot_sigma_m);
            A->set(m, current_order, b_dot_sigma_m);
        }
        A->set(current_order, current_order, dot(b_vec[current_order], sigma_vec[current_order]));

        current_order++;
        psi::SharedMatrix G(new psi::Matrix(current_order, current_order));

        for (int k = 0; k < current_order; k++) {
            for (int j = 0; j < current_order; j++) {
                G->set(k, j, A->get(k, j));
            }
        }
        psi::SharedMatrix evecs(new psi::Matrix(current_order, current_order));
        psi::SharedVector eigs(new Vector(current_order));
        G->diagonalize(evecs, eigs);

        double e_gradiant = -lambda;

        lambda = eigs->get(0);
        for (int j = 0; j < current_order; j++) {
            alpha_vec[j] = evecs->get(j, 0);
        }
        e_gradiant += lambda;
        outfile->Printf("\nDavidson iter %4d order %4d correction norm %10.3e dE %10.3e.", i,
                        current_order, correct_norm, e_gradiant);
        if (std::fabs(e_gradiant) < e_convergence_) {
            i++;
            break;
        }
        if (current_order >= davidson_subspace_per_root_) {
            b_vec[0].resize(dets_size, 0.0);
            for (int j = 0, jmax = dets.size(); j < jmax; j++) {
                std::vector<double> b_j(davidson_collapse_per_root_, 0.0);
                std::vector<double> sigma_j(davidson_collapse_per_root_, 0.0);
                for (int l = 0; l < davidson_collapse_per_root_; l++) {
                    for (int k = 0; k < current_order; k++) {
                        b_j[l] += evecs->get(k, l) * b_vec[k][j];
                        sigma_j[l] += evecs->get(k, l) * sigma_vec[k][j];
                    }
                }
                for (int l = 0; l < davidson_collapse_per_root_; l++) {
                    b_vec[l][j] = b_j[l];
                    sigma_vec[l][j] = sigma_j[l];
                }
            }
            for (int l = davidson_collapse_per_root_; l < davidson_subspace_per_root_; l++) {
                b_vec[l].clear();
                sigma_vec[l].clear();
            }
            for (int m = 0; m < davidson_collapse_per_root_; m++) {
                for (int n = 0; n <= m; n++) {
                    double n_dot_sigma_m = dot(b_vec[n], sigma_vec[m]);
                    A->set(n, m, n_dot_sigma_m);
                    A->set(m, n, n_dot_sigma_m);
                }
            }
            alpha_vec[0] = 1.0;
            for (int l = 1; l < davidson_subspace_per_root_; l++) {
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
    C.resize(dets.size(), 0.0);
    //    b_vec[0].resize(dets.size(), 0.0);
    for (int i = 1; i < current_order; i++) {
        for (int j = 0, jmax = b_vec[i].size(); j < jmax; j++) {
            C[j] += alpha_vec[i] * b_vec[i][j];
        }
    }

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

void ProjectorCI::propagate_Chebyshev(det_vec& dets, std::vector<double>& C,
                                      double spawning_threshold) {
    // A map that contains the pair (determinant,coefficient)
    const double PI = 2 * acos(0.0);
    det_hash<> dets_C_hash;
    for (int i = 0; i < chebyshev_order_; i++) {
        double root = 0.0;
        if (i < 0) {
            root = 1.0;
        } else {
            root = cos((2.0 * i + 1) * PI / (2.0 * chebyshev_order_));
        }

        apply_tau_H(-1.0, spawning_threshold, dets, C, dets_C_hash, range_ * root + shift_);
        copy_hash_to_vec(dets_C_hash, dets, C);
        dets_C_hash.clear();
        normalize(C);
    }
}

void ProjectorCI::propagate_Linear(det_vec& dets, std::vector<double>& C, double tau,
                                   double spawning_threshold, double S) {
    // A map that contains the pair (determinant,coefficient)
    det_hash<> dets_C_hash;

    apply_tau_H(-tau, spawning_threshold, dets, C, dets_C_hash, S + 1.0 / tau);

    // Overwrite the input vectors with the updated wave function
    copy_hash_to_vec(dets_C_hash, dets, C);
}

void ProjectorCI::propagate_Taylor(int order, det_vec& dets, std::vector<double>& C, double tau,
                                   double spawning_threshold, double S) {
    // A map that contains the pair (determinant,coefficient)
    det_hash<> dets_C_hash;
    det_hash<> dets_sum_map;
    // A vector of maps that hold (determinant,coefficient)

    // Propagate the wave function for one time step using |n+1> = (1 - tau
    // (H-S) + tau^2 (H-S)^2 / 2)|n>

    // Term 1. |n>
    for (size_t I = 0, max_I = dets.size(); I < max_I; ++I) {
        dets_sum_map[dets[I]] = C[I];
    }

    for (int j = 1; j <= order; ++j) {
        double delta_tau = -tau / double(j);
        apply_tau_H(delta_tau, spawning_threshold, dets, C, dets_C_hash, S);

        // Add this term to the total vector
        combine_hashes(dets_C_hash, dets_sum_map);
        // Copy the wave function to a vector
        if (j < order) {
            copy_hash_to_vec(dets_C_hash, dets, C);
        }
        dets_C_hash.clear();
        //        if(iter_ % energy_estimate_freq_ == 0){
        //            double norm = 0.0;
        //            for (double CI : C) norm += CI * CI;
        //            norm = std::sqrt(norm);
        //            outfile->Printf("\n  ||C(%d)-C(%d)|| = %e
        //            (%f)",j,j-1,norm,delta_tau);
        //        }
    }
    copy_hash_to_vec(dets_sum_map, dets, C);
}

void ProjectorCI::propagate_power(det_vec& dets, std::vector<double>& C, double spawning_threshold,
                                  double S) {
    // A map that contains the pair (determinant,coefficient)
    det_hash<> dets_C_hash;

    apply_tau_H(1.0, spawning_threshold, dets, C, dets_C_hash, S);

    // Overwrite the input vectors with the updated wave function
    copy_hash_to_vec(dets_C_hash, dets, C);
}

void ProjectorCI::propagate_Polynomial(det_vec& dets, std::vector<double>& C,
                                       std::vector<double>& coef, double spawning_threshold) {
    int order = coef.size() - 1;
    if (order <= 0)
        return;
    // A map that contains the pair (determinant,coefficient)
    det_hash<> dets_C_hash;
    det_hash<> dets_sum_map;
    // A vector of maps that hold (determinant,coefficient)

    // Term 1. |n>
    for (size_t I = 0, max_I = dets.size(); I < max_I; ++I) {
        dets_sum_map[dets[I]] = coef[0] * C[I];
    }
    if (symm_approx_H_) {
        apply_tau_H_symm(coef[1], spawning_threshold, dets, C, dets_C_hash, 0.0);
    } else {
        apply_tau_H(coef[1], spawning_threshold, dets, C, dets_C_hash, 0.0);
    }

    // Add this term to the total vector
    combine_hashes(dets_C_hash, dets_sum_map);

    for (int j = 2; j <= order; ++j) {
        // Copy the wave function to a vector
        copy_hash_to_vec(dets_C_hash, dets, C);
        double current_spawning = spawning_threshold * norm(dets_C_hash);
        dets_C_hash.clear();
        //        apply_tau_H(coef[j]/coef[j-1],spawning_threshold,dets,C,dets_C_hash,0.0);
        apply_tau_H_subset(coef[j] / coef[j - 1], current_spawning, dets, C, dets_sum_map,
                           dets_C_hash, shift_);

        // Add this term to the total vector
        combine_hashes(dets_C_hash, dets_sum_map);
    }
    copy_hash_to_vec(dets_sum_map, dets, C);
}

void ProjectorCI::propagate_Trotter_linear(det_vec& dets, std::vector<double>& C, double tau,
                                           double spawning_threshold, double S) {
    // A map that contains the pair (determinant,coefficient)
    det_hash<> dets_C_hash;

    // Term 1. |n>
    for (size_t I = 0, max_I = dets.size(); I < max_I; ++I) {
        dets_C_hash[dets[I]] = C[I];
    }
    // Term 2. -tau (H - S)|n>
    apply_tau_H(-tau, spawning_threshold, dets, C, dets_C_hash, S);

    // Correct the diagonals
    for (size_t I = 0, max_I = dets.size(); I < max_I; ++I) {
        double det_energy = as_ints_->energy(dets[I]) + as_ints_->scalar_energy();
        double CI = dets_C_hash[dets[I]];
        dets_C_hash[dets[I]] += tau * (det_energy - S) * CI;
        dets_C_hash[dets[I]] += exp(-tau * (det_energy - S)) * CI;
    }

    // Overwrite the input vectors with the updated wave function
    copy_hash_to_vec(dets_C_hash, dets, C);
}

void ProjectorCI::propagate_Olsen(det_vec& dets, std::vector<double>& C, double spawning_threshold,
                                  double S) {
    // A map that contains the pair (determinant,coefficient)
    det_hash<> dets_C_hash;

    // 1.  Compute H - E (S = E)
    apply_tau_H(1.0, spawning_threshold, dets, C, dets_C_hash, S);

    double delta_E_num = 0.0;
    double delta_E_den = 0.0;
    for (size_t I = 0, max_I = dets.size(); I < max_I; ++I) {
        double CI = C[I];
        double EI = as_ints_->energy(dets[I]) + as_ints_->scalar_energy();
        double sigma_I = dets_C_hash[dets[I]];
        delta_E_num += CI * sigma_I / (EI - S);
        delta_E_den += CI * CI / (EI - S);
    }
    double delta_E = delta_E_num / delta_E_den;

    for (size_t I = 0, max_I = dets.size(); I < max_I; ++I) {
        dets_C_hash[dets[I]] -= C[I] * delta_E;
    }

    double step_norm = 0.0;
    for (auto& det_C : dets_C_hash) {
        double EI = as_ints_->energy(det_C.first) + as_ints_->scalar_energy();
        det_C.second /= -(EI - S);
        step_norm += det_C.second * det_C.second;
    }
    step_norm = std::sqrt(step_norm);

    double max_norm = 0.05;
    if (step_norm > max_norm) {
        outfile->Printf("\n\t  Step norm = %f is greather than %f.  Rescaling Olsen step.",
                        step_norm, max_norm);
        double factor = max_norm / step_norm;
        for (auto& det_C : dets_C_hash) {
            det_C.second *= factor;
        }
    }

    double sum = 0.0;
    for (size_t I = 0, max_I = dets.size(); I < max_I; ++I) {
        sum += std::fabs(dets_C_hash[dets[I]]);
        dets_C_hash[dets[I]] += C[I];
    }

    double norm = 0.0;
    for (auto& det_C : dets_C_hash) {
        norm += std::pow(det_C.second, 2.0);
    }
    norm = std::sqrt(norm);
    for (auto& det_C : dets_C_hash) {
        det_C.second /= norm;
    }

    // Overwrite the input vectors with the updated wave function
    copy_hash_to_vec(dets_C_hash, dets, C);
}

void ProjectorCI::apply_tau_H_symm_det_dynamic(
    double tau, double spawning_threshold, det_hash<>& pre_dets_C_hash, const Determinant& detI,
    double CI, std::vector<std::pair<Determinant, double>>& new_space_C_vec, double E0,
    std::pair<double, double>& max_coupling) {
    bool do_singles =
        (max_coupling.first == 0.0) or (std::fabs(max_coupling.first * CI) >= spawning_threshold);
    bool do_doubles =
        (max_coupling.second == 0.0) or (std::fabs(max_coupling.second * CI) >= spawning_threshold);

    // Diagonal contributions
    double det_energy = as_ints_->energy(detI) + as_ints_->scalar_energy();
    new_space_C_vec.push_back(std::make_pair(detI, tau * (det_energy - E0) * CI));

    if (do_singles or do_doubles) {

        std::vector<int> aocc = detI.get_alfa_occ(nact_);
        std::vector<int> bocc = detI.get_beta_occ(nact_);
        std::vector<int> avir = detI.get_alfa_vir(nact_);
        std::vector<int> bvir = detI.get_beta_vir(nact_);

        int noalpha = aocc.size();
        int nobeta = bocc.size();
        int nvalpha = avir.size();
        int nvbeta = bvir.size();

        if (do_singles) {
            // Generate alpha excitations
            for (int i = 0; i < noalpha; ++i) {
                int ii = aocc[i];
                for (int a = 0; a < nvalpha; ++a) {
                    int aa = avir[a];
                    if ((mo_symmetry_[ii] ^ mo_symmetry_[aa]) == 0) {
                        Determinant detJ(detI);
                        detJ.set_alfa_bit(ii, false);
                        detJ.set_alfa_bit(aa, true);
                        double HJI = as_ints_->slater_rules(detJ, detI);
                        max_coupling.first = std::max(max_coupling.first, std::fabs(HJI));
                        if (std::fabs(HJI * CI) >= spawning_threshold) {
                            new_space_C_vec.push_back(std::make_pair(detJ, tau * HJI * CI));
                            det_hash_it it = pre_dets_C_hash.find(detJ);
                            if (it != pre_dets_C_hash.end() &&
                                std::fabs(HJI * it->second) < spawning_threshold) {
                                //                                outfile->Printf("\nTouched:
                                //                                %.12f
                                //                                previous:
                                //                                %.12f",CI,
                                //                                new_space_C_vec[0].second);
                                new_space_C_vec[0].second += tau * HJI * it->second;
                                //                                outfile->Printf(",
                                //                                then: %.12f",
                                //                                new_space_C_vec[0].second);
                            }
                        }
                    }
                }
            }
            // Generate beta excitations
            for (int i = 0; i < nobeta; ++i) {
                int ii = bocc[i];
                for (int a = 0; a < nvbeta; ++a) {
                    int aa = bvir[a];
                    if ((mo_symmetry_[ii] ^ mo_symmetry_[aa]) == 0) {
                        Determinant detJ(detI);
                        detJ.set_beta_bit(ii, false);
                        detJ.set_beta_bit(aa, true);
                        double HJI = as_ints_->slater_rules(detJ, detI);
                        max_coupling.first = std::max(max_coupling.first, std::fabs(HJI));
                        if (std::fabs(HJI * CI) >= spawning_threshold) {
                            new_space_C_vec.push_back(std::make_pair(detJ, tau * HJI * CI));
                            det_hash_it it = pre_dets_C_hash.find(detJ);
                            if (it != pre_dets_C_hash.end() &&
                                std::fabs(HJI * it->second) < spawning_threshold) {
                                new_space_C_vec[0].second += tau * HJI * it->second;
                            }
                        }
                    }
                }
            }
        }

        if (do_doubles) {
            // Generate alpha-alpha excitations
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
                                double HJI = as_ints_->tei_aa(ii, jj, aa, bb);
                                max_coupling.second = std::max(max_coupling.second, std::fabs(HJI));
                                if (std::fabs(HJI * CI) >= spawning_threshold) {
                                    Determinant detJ(detI);
                                    HJI *= detJ.double_excitation_aa(ii, jj, aa, bb);
                                    new_space_C_vec.push_back(std::make_pair(detJ, tau * HJI * CI));
                                    det_hash_it it = pre_dets_C_hash.find(detJ);
                                    if (it != pre_dets_C_hash.end() &&
                                        std::fabs(HJI * it->second) < spawning_threshold) {
                                        new_space_C_vec[0].second += tau * HJI * it->second;
                                    }
                                }
                            }
                        }
                    }
                }
            }
            // Generate alpha-beta excitations
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
                                double HJI = as_ints_->tei_ab(ii, jj, aa, bb);
                                max_coupling.second = std::max(max_coupling.second, std::fabs(HJI));
                                if (std::fabs(HJI * CI) >= spawning_threshold) {
                                    Determinant detJ(detI);
                                    HJI *= detJ.double_excitation_ab(ii, jj, aa, bb);
                                    new_space_C_vec.push_back(std::make_pair(detJ, tau * HJI * CI));
                                    det_hash_it it = pre_dets_C_hash.find(detJ);
                                    if (it != pre_dets_C_hash.end() &&
                                        std::fabs(HJI * it->second) < spawning_threshold) {
                                        new_space_C_vec[0].second += tau * HJI * it->second;
                                    }
                                }
                            }
                        }
                    }
                }
            }
            // Generate beta-beta excitations
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
                                double HJI = as_ints_->tei_bb(ii, jj, aa, bb);
                                max_coupling.second = std::max(max_coupling.second, std::fabs(HJI));
                                if (std::fabs(HJI * CI) >= spawning_threshold) {
                                    Determinant detJ(detI);
                                    HJI *= detJ.double_excitation_bb(ii, jj, aa, bb);
                                    new_space_C_vec.push_back(std::make_pair(detJ, tau * HJI * CI));
                                    det_hash_it it = pre_dets_C_hash.find(detJ);
                                    if (it != pre_dets_C_hash.end() &&
                                        std::fabs(HJI * it->second) < spawning_threshold) {
                                        new_space_C_vec[0].second += tau * HJI * it->second;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

void ProjectorCI::apply_tau_H_symm(double tau, double spawning_threshold, det_vec& dets,
                                   const std::vector<double>& C, det_hash<>& dets_C_hash,
                                   double S) {
    // A vector of maps that hold (determinant,coefficient)
    //    std::vector<det_hash<>> thread_det_C_hash(num_threads_);
    std::vector<std::pair<double, double>> thread_max_HJI(num_threads_);
    //    outfile->Printf("\nSymmetric used.\n");

    if (do_dynamic_prescreening_) {
        det_hash<> pre_dets_C_hash;
        copy_vec_to_hash(dets, C, pre_dets_C_hash);

        //        outfile->Printf("\nSize of pre: %zu", pre_dets_C_hash.size());

        size_t max_I = dets.size();
#pragma omp parallel for
        for (size_t I = 0; I < max_I; ++I) {
            std::pair<double, double> zero_pair(0.0, 0.0);
            // Update the list of couplings
            std::pair<double, double> max_coupling;
#pragma omp critical
            { max_coupling = dets_max_couplings_[dets[I]]; }
            if (max_coupling == zero_pair) {
                std::vector<std::pair<Determinant, double>> thread_det_C_vec;
                apply_tau_H_symm_det_dynamic(tau, spawning_threshold, pre_dets_C_hash, dets[I],
                                             C[I], thread_det_C_vec, S, max_coupling);
#pragma omp critical
                {
                    for (auto det_C : thread_det_C_vec) {
                        dets_C_hash[det_C.first] += det_C.second;
                    }
                }
#pragma omp critical
                { dets_max_couplings_[dets[I]] = max_coupling; }
            } else {
                std::vector<std::pair<Determinant, double>> thread_det_C_vec;
                apply_tau_H_symm_det_dynamic(tau, spawning_threshold, pre_dets_C_hash, dets[I],
                                             C[I], thread_det_C_vec, S, max_coupling);
#pragma omp critical
                {
                    for (auto det_C : thread_det_C_vec) {
                        dets_C_hash[det_C.first] += det_C.second;
                    }
                }
            }
        }
    } else {
        outfile->Printf("Symmetric approximated hamiltonian only implemented "
                        "on dynamic prescreening.");
    }
    if (approx_E_flag_) {
        timer_on("PCI:<E>a");
        size_t max_I = dets.size();
        double CHC_energy = 0.0;
#pragma omp parallel for reduction(+ : CHC_energy)
        for (size_t I = 0; I < max_I; ++I) {
            CHC_energy += C[I] * dets_C_hash[dets[I]];
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

void ProjectorCI::apply_tau_H_ref_C_symm(double tau, double spawning_threshold, det_vec& dets,
                                         const std::vector<double>& C,
                                         const std::vector<double>& ref_C, det_hash<>& dets_C_hash,
                                         double S) {
    //    outfile -> Printf("\napply_tau_H_ref_C_symm : Beginning args:");
    //    (ref_C, "ref_C");
    //    (C, "C");

    // A vector of maps that hold (determinant,coefficient)
    //    std::vector<det_hash<>> thread_det_C_hash(num_threads_);
    std::vector<std::pair<double, double>> thread_max_HJI(num_threads_);
    //    outfile->Printf("\nSymmetric used.\n");

    if (do_dynamic_prescreening_) {
        det_hash<> pre_dets_C_hash;
        copy_vec_to_hash(dets, C, pre_dets_C_hash);
        det_hash<> ref_dets_C_hash;
        copy_vec_to_hash(dets, ref_C, ref_dets_C_hash);
        //        outfile -> Printf("\napply_tau_H_ref_C_symm : Converted to
        //        hashs:");
        //        print_hash(pre_dets_C_hash, "pre_dets_C_hash");
        //        print_hash(ref_dets_C_hash, "ref_dets_C_hash");

        size_t ref_max_I = ref_C.size();
#pragma omp parallel for
        for (size_t I = 0; I < ref_max_I; ++I) {
            //            outfile -> Printf("\napply_tau_H_ref_C_symm :
            //            Det[%d]:\n", I);
            std::pair<double, double> zero_pair(0.0, 0.0);
            // Update the list of couplings
            std::pair<double, double> max_coupling;
#pragma omp critical
            { max_coupling = dets_max_couplings_[dets[I]]; }
            if (max_coupling == zero_pair) {
                std::vector<std::pair<Determinant, double>> thread_det_C_vec;
                apply_tau_H_ref_C_symm_det_dynamic(tau, spawning_threshold, pre_dets_C_hash,
                                                   ref_dets_C_hash, dets[I], C[I], ref_C[I],
                                                   thread_det_C_vec, S, max_coupling);
//                apply_tau_H_ref_C_symm_det_dynamic_smooth(tau,spawning_threshold,
//                pre_dets_C_hash, ref_dets_C_hash,dets[I],C[I], ref_C[I],
//                thread_det_C_vec,S,max_coupling);
#pragma omp critical
                {
                    for (auto det_C : thread_det_C_vec) {
                        dets_C_hash[det_C.first] += det_C.second;
                        //                        count_hash(det_C.first);
                        //                        det_C.first.print();
                        //                        outfile->Printf(" %.4lf ",
                        //                        det_C.second);
                    }
                }
#pragma omp critical
                { dets_max_couplings_[dets[I]] = max_coupling; }
            } else {
                std::vector<std::pair<Determinant, double>> thread_det_C_vec;
                apply_tau_H_ref_C_symm_det_dynamic(tau, spawning_threshold, pre_dets_C_hash,
                                                   ref_dets_C_hash, dets[I], C[I], ref_C[I],
                                                   thread_det_C_vec, S, max_coupling);
//                apply_tau_H_ref_C_symm_det_dynamic_smooth(tau,spawning_threshold,
//                pre_dets_C_hash, ref_dets_C_hash,dets[I],C[I], ref_C[I],
//                thread_det_C_vec,S,max_coupling);
#pragma omp critical
                {
                    for (auto det_C : thread_det_C_vec) {
                        dets_C_hash[det_C.first] += det_C.second;
                        //                        count_hash(det_C.first);
                        //                        det_C.first.print();
                        //                        outfile->Printf(" %.4lf ",
                        //                        det_C.second);
                    }
                }
            }
        }
        size_t max_I = C.size();
        for (size_t I = ref_max_I; I < max_I; ++I) {
            // Diagonal contribution
            double det_energy = as_ints_->energy(dets[I]) + as_ints_->scalar_energy();
// Diagonal contributions
#pragma omp critical
            {
                dets_C_hash[dets[I]] += tau * (det_energy - S) * C[I];
                //                count_hash(dets[I]);
            }
        }
    } else {
        outfile->Printf("Symmetric approximated hamiltonian only implemented "
                        "on dynamic prescreening.");
    }
    if (approx_E_flag_) {
        timer_on("PCI:<E>a");
        size_t max_I = dets.size();
        double CHC_energy = 0.0;
#pragma omp parallel for reduction(+ : CHC_energy)
        for (size_t I = 0; I < max_I; ++I) {
            CHC_energy += C[I] * dets_C_hash[dets[I]];
            //            count_hash(dets[I]);
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

    //    outfile -> Printf("\napply_tau_H_ref_C_symm : End:");
    //    print_hash(dets_C_hash, "dets_C_hash", true);
}

// void ProjectorCI::apply_tau_H_ref_C_symm_det_dynamic_stat(double tau, double
// spawning_threshold, det_hash<> &pre_dets_C_hash, det_hash<> &ref_dets_C_hash,
// const Determinant &detI, double CI, double ref_CI,
// std::vector<std::pair<Determinant, double> > &new_space_C_vec, double E0,
// std::pair<double,double>& max_coupling)
//{
////    outfile -> Printf("\napply_tau_H_ref_C_symm_det_dynamic : Beginning
/// args:");
////    outfile -> Printf("\n CI: %lf, ref_CI: %lf\n", CI, ref_CI);

//    bool do_singles = (max_coupling.first == 0.0) or
//    (std::fabs(max_coupling.first * ref_CI) >= spawning_threshold);
//    bool do_doubles = (max_coupling.second == 0.0) or
//    (std::fabs(max_coupling.second  * ref_CI) >= spawning_threshold);

//    // Diagonal contributions
//    double det_energy = detI.energy();
//    new_space_C_vec.push_back(std::make_pair(detI, tau * (det_energy - E0) *
//    CI));

//    if (do_singles or do_doubles){

//        std::vector<int> aocc = detI.get_alfa_occ(nact_);
//        std::vector<int> bocc = detI.get_beta_occ(nact_);
//        std::vector<int> avir = detI.get_alfa_vir(nact_);
//        std::vector<int> bvir = detI.get_beta_vir(nact_);

//        int noalpha = aocc.size();
//        int nobeta  = bocc.size();
//        int nvalpha = avir.size();
//        int nvbeta  = bvir.size();

//        if (do_singles){
//            // Generate alpha excitations
//            for (int i = 0; i < noalpha; ++i){
//                int ii = aocc[i];
//                for (int a = 0; a < nvalpha; ++a){
//                    int aa = avir[a];
//                    if ((mo_symmetry_[ii] ^ mo_symmetry_[aa]) == 0){
//                        Determinant detJ(detI);
//                        detJ.set_alfa_bit(ii,false);
//                        detJ.set_alfa_bit(aa,true);
//                        double HJI = detJ.slater_rules(detI);
//                        max_coupling.first =
//                        std::max(max_coupling.first,std::fabs(HJI));
//                        if (std::fabs(HJI * ref_CI) >= spawning_threshold){
//                            new_space_C_vec.push_back(std::make_pair(detJ, tau
//                            * HJI * CI));

//                            det_hash_it it_pre = pre_dets_C_hash.find(detJ);
//                            if (it_pre != pre_dets_C_hash.end()) {
//                                det_hash_it it_ref =
//                                ref_dets_C_hash.find(detJ);
//                                count_hash(detJ);
//                                count_hash(detJ);
//                                if (it_ref == ref_dets_C_hash.end() ||
//                                std::fabs(HJI * it_ref -> second) <
//                                spawning_threshold){
//    //                                if (it == pre_dets_C_hash.end()) {
//    //                                    outfile -> Printf("\n\nERROR:
//    apply_tau_H_ref_C_symm_det_dynamic aa det NOT FOUND in pre_dets_C_hash");
//    //                                }
//                                    new_space_C_vec[0].second += tau * HJI *
//                                    it_pre -> second;
//    //                                outfile->Printf(", then: %.12f",
//    new_space_C_vec[0].second);
//                                }
//                            }

//                        }
//                    }
//                }
//            }
//            // Generate beta excitations
//            for (int i = 0; i < nobeta; ++i){
//                int ii = bocc[i];
//                for (int a = 0; a < nvbeta; ++a){
//                    int aa = bvir[a];
//                    if ((mo_symmetry_[ii] ^ mo_symmetry_[aa])  == 0){
//                        Determinant detJ(detI);
//                        detJ.set_beta_bit(ii,false);
//                        detJ.set_beta_bit(aa,true);
//                        double HJI = detJ.slater_rules(detI);
//                        max_coupling.first =
//                        std::max(max_coupling.first,std::fabs(HJI));
//                        if (std::fabs(HJI * ref_CI) >= spawning_threshold){
//                            new_space_C_vec.push_back(std::make_pair(detJ, tau
//                            * HJI * CI));

//                            det_hash_it it_pre = pre_dets_C_hash.find(detJ);
//                            if (it_pre != pre_dets_C_hash.end()) {
//                                det_hash_it it_ref =
//                                ref_dets_C_hash.find(detJ);
//                                count_hash(detJ);
//                                count_hash(detJ);
//                                if (it_ref == ref_dets_C_hash.end() ||
//                                std::fabs(HJI * it_ref -> second) <
//                                spawning_threshold){
//    //                                if (it == pre_dets_C_hash.end()) {
//    //                                    outfile -> Printf("\n\nERROR:
//    apply_tau_H_ref_C_symm_det_dynamic aa det NOT FOUND in pre_dets_C_hash");
//    //                                }
//                                    new_space_C_vec[0].second += tau * HJI *
//                                    it_pre -> second;
//    //                                outfile->Printf(", then: %.12f",
//    new_space_C_vec[0].second);
//                                }
//                            }

//                        }
//                    }
//                }
//            }
//        }

//        if (do_doubles){
//            // Generate alpha-alpha excitations
//            for (int i = 0; i < noalpha; ++i){
//                int ii = aocc[i];
//                for (int j = i + 1; j < noalpha; ++j){
//                    int jj = aocc[j];
//                    for (int a = 0; a < nvalpha; ++a){
//                        int aa = avir[a];
//                        for (int b = a + 1; b < nvalpha; ++b){
//                            int bb = avir[b];
//                            if ((mo_symmetry_[ii] ^ mo_symmetry_[jj] ^
//                            mo_symmetry_[aa] ^ mo_symmetry_[bb]) == 0){
//                                double HJI = as_ints_->tei_aa(ii,jj,aa,bb);
//                                max_coupling.second =
//                                std::max(max_coupling.second,std::fabs(HJI));
//                                if (std::fabs(HJI * ref_CI) >=
//                                spawning_threshold){
//                                    Determinant detJ(detI);
//                                    HJI *=
//                                    detJ.double_excitation_aa(ii,jj,aa,bb);
//                                    new_space_C_vec.push_back(std::make_pair(detJ,
//                                    tau * HJI * CI));

//                                    det_hash_it it_pre =
//                                    pre_dets_C_hash.find(detJ);
//                                    if (it_pre != pre_dets_C_hash.end()) {
//                                        det_hash_it it_ref =
//                                        ref_dets_C_hash.find(detJ);
//                                        count_hash(detJ);
//                                        count_hash(detJ);
//                                        if (it_ref == ref_dets_C_hash.end() ||
//                                        std::fabs(HJI * it_ref -> second) <
//                                        spawning_threshold){
//            //                                if (it == pre_dets_C_hash.end())
//            {
//            //                                    outfile ->
//            Printf("\n\nERROR: apply_tau_H_ref_C_symm_det_dynamic aa det NOT
//            FOUND in pre_dets_C_hash");
//            //                                }
//                                            new_space_C_vec[0].second += tau *
//                                            HJI * it_pre -> second;
//            //                                outfile->Printf(", then: %.12f",
//            new_space_C_vec[0].second);
//                                        }
//                                    }

//                                }
//                            }
//                        }
//                    }
//                }
//            }
//            // Generate alpha-beta excitations
//            for (int i = 0; i < noalpha; ++i){
//                int ii = aocc[i];
//                for (int j = 0; j < nobeta; ++j){
//                    int jj = bocc[j];
//                    for (int a = 0; a < nvalpha; ++a){
//                        int aa = avir[a];
//                        for (int b = 0; b < nvbeta; ++b){
//                            int bb = bvir[b];
//                            if ((mo_symmetry_[ii] ^ mo_symmetry_[jj] ^
//                            mo_symmetry_[aa] ^ mo_symmetry_[bb]) == 0){
//                                double HJI = as_ints_->tei_ab(ii,jj,aa,bb);
//                                max_coupling.second =
//                                std::max(max_coupling.second,std::fabs(HJI));
//                                if (std::fabs(HJI * ref_CI) >=
//                                spawning_threshold){
//                                    Determinant detJ(detI);
//                                    HJI *=
//                                    detJ.double_excitation_ab(ii,jj,aa,bb);
//                                    new_space_C_vec.push_back(std::make_pair(detJ,
//                                    tau * HJI * CI));

//                                    det_hash_it it_pre =
//                                    pre_dets_C_hash.find(detJ);
//                                    if (it_pre != pre_dets_C_hash.end()) {
//                                        det_hash_it it_ref =
//                                        ref_dets_C_hash.find(detJ);
//                                        count_hash(detJ);
//                                        count_hash(detJ);
//                                        if (it_ref == ref_dets_C_hash.end() ||
//                                        std::fabs(HJI * it_ref -> second) <
//                                        spawning_threshold){
//            //                                if (it == pre_dets_C_hash.end())
//            {
//            //                                    outfile ->
//            Printf("\n\nERROR: apply_tau_H_ref_C_symm_det_dynamic aa det NOT
//            FOUND in pre_dets_C_hash");
//            //                                }
//                                            new_space_C_vec[0].second += tau *
//                                            HJI * it_pre -> second;
//            //                                outfile->Printf(", then: %.12f",
//            new_space_C_vec[0].second);
//                                        }
//                                    }

//                                }
//                            }
//                        }
//                    }
//                }
//            }
//            // Generate beta-beta excitations
//            for (int i = 0; i < nobeta; ++i){
//                int ii = bocc[i];
//                for (int j = i + 1; j < nobeta; ++j){
//                    int jj = bocc[j];
//                    for (int a = 0; a < nvbeta; ++a){
//                        int aa = bvir[a];
//                        for (int b = a + 1; b < nvbeta; ++b){
//                            int bb = bvir[b];
//                            if ((mo_symmetry_[ii] ^ (mo_symmetry_[jj] ^
//                            (mo_symmetry_[aa] ^ mo_symmetry_[bb]))) == 0){
//                                double HJI = as_ints_->tei_bb(ii,jj,aa,bb);
//                                max_coupling.second =
//                                std::max(max_coupling.second,std::fabs(HJI));
//                                if (std::fabs(HJI * ref_CI) >=
//                                spawning_threshold){
//                                    Determinant detJ(detI);
//                                    HJI *=
//                                    detJ.double_excitation_bb(ii,jj,aa,bb);
//                                    new_space_C_vec.push_back(std::make_pair(detJ,
//                                    tau * HJI * CI));

//                                    det_hash_it it_pre =
//                                    pre_dets_C_hash.find(detJ);
//                                    if (it_pre != pre_dets_C_hash.end()) {
//                                        det_hash_it it_ref =
//                                        ref_dets_C_hash.find(detJ);
//                                        count_hash(detJ);
//                                        count_hash(detJ);
//                                        if (it_ref == ref_dets_C_hash.end() ||
//                                        std::fabs(HJI * it_ref -> second) <
//                                        spawning_threshold){
//            //                                if (it == pre_dets_C_hash.end())
//            {
//            //                                    outfile ->
//            Printf("\n\nERROR: apply_tau_H_ref_C_symm_det_dynamic aa det NOT
//            FOUND in pre_dets_C_hash");
//            //                                }
//                                            new_space_C_vec[0].second += tau *
//                                            HJI * it_pre -> second;
//            //                                outfile->Printf(", then: %.12f",
//            new_space_C_vec[0].second);
//                                        }
//                                    }

//                                }
//                            }
//                        }
//                    }
//                }
//            }
//        }
//    }
//}

inline double polynomial_smoother(double edge0, double edge1, double x) {
    if (x <= edge0) {
        return 0.0;
    }
    if (x >= edge1) {
        return 1.0;
    }
    // Scale, and clamp x to 0..1 range
    x = (x - edge0) / (edge1 - edge0);
    // Evaluate polynomial
    return x * x * x * (x * (x * 6. - 15.) + 10.);
}

void ProjectorCI::apply_tau_H_ref_C_symm_det_dynamic_smooth(
    double tau, double spawning_threshold, det_hash<>& pre_dets_C_hash, det_hash<>& ref_dets_C_hash,
    const Determinant& detI, double CI, double ref_CI,
    std::vector<std::pair<Determinant, double>>& new_space_C_vec, double E0,
    std::pair<double, double>& max_coupling) {
    //    outfile -> Printf("\napply_tau_H_ref_C_symm_det_dynamic : Beginning
    //    args:");
    //    outfile -> Printf("\n CI: %lf, ref_CI: %lf\n", CI, ref_CI);
    const double smooth_edge1 = 1.5 * spawning_threshold;
    bool do_singles = (max_coupling.first == 0.0) or
                      (std::fabs(max_coupling.first * ref_CI) >= spawning_threshold);
    bool do_doubles = (max_coupling.second == 0.0) or
                      (std::fabs(max_coupling.second * ref_CI) >= spawning_threshold);

    // Diagonal contributions
    double det_energy = as_ints_->energy(detI) + as_ints_->scalar_energy();
    new_space_C_vec.push_back(std::make_pair(detI, tau * (det_energy - E0) * CI));

    if (do_singles or do_doubles) {

        std::vector<int> aocc = detI.get_alfa_occ(nact_);
        std::vector<int> bocc = detI.get_beta_occ(nact_);
        std::vector<int> avir = detI.get_alfa_vir(nact_);
        std::vector<int> bvir = detI.get_beta_vir(nact_);

        int noalpha = aocc.size();
        int nobeta = bocc.size();
        int nvalpha = avir.size();
        int nvbeta = bvir.size();

        if (do_singles) {
            // Generate alpha excitations
            for (int i = 0; i < noalpha; ++i) {
                int ii = aocc[i];
                for (int a = 0; a < nvalpha; ++a) {
                    int aa = avir[a];
                    if ((mo_symmetry_[ii] ^ mo_symmetry_[aa]) == 0) {
                        Determinant detJ(detI);
                        detJ.set_alfa_bit(ii, false);
                        detJ.set_alfa_bit(aa, true);
                        double HJI = as_ints_->slater_rules(detJ, detI);
                        max_coupling.first = std::max(max_coupling.first, std::fabs(HJI));

                        if (std::fabs(HJI * ref_CI) >= spawning_threshold) {
                            double smooth_factor = polynomial_smoother(
                                spawning_threshold, smooth_edge1, std::fabs(HJI * ref_CI));
                            new_space_C_vec.push_back(
                                std::make_pair(detJ, smooth_factor * tau * HJI * CI));

                            det_hash_it it_pre = pre_dets_C_hash.find(detJ);
                            if (it_pre != pre_dets_C_hash.end()) {
                                det_hash_it it_ref = ref_dets_C_hash.find(detJ);
                                if (it_ref == ref_dets_C_hash.end() ||
                                    std::fabs(HJI * it_ref->second) < spawning_threshold) {
                                    //                                if (it ==
                                    //                                pre_dets_C_hash.end())
                                    //                                {
                                    //                                    outfile
                                    //                                    ->
                                    //                                    Printf("\n\nERROR:
                                    //                                    apply_tau_H_ref_C_symm_det_dynamic
                                    //                                    aa det
                                    //                                    NOT
                                    //                                    FOUND
                                    //                                    in
                                    //                                    pre_dets_C_hash");
                                    //                                }
                                    new_space_C_vec[0].second +=
                                        smooth_factor * tau * HJI * it_pre->second;
                                    //                                outfile->Printf(",
                                    //                                then:
                                    //                                %.12f",
                                    //                                new_space_C_vec[0].second);
                                }
                            }
                        }
                    }
                }
            }
            // Generate beta excitations
            for (int i = 0; i < nobeta; ++i) {
                int ii = bocc[i];
                for (int a = 0; a < nvbeta; ++a) {
                    int aa = bvir[a];
                    if ((mo_symmetry_[ii] ^ mo_symmetry_[aa]) == 0) {
                        Determinant detJ(detI);
                        detJ.set_beta_bit(ii, false);
                        detJ.set_beta_bit(aa, true);
                        double HJI = as_ints_->slater_rules(detJ, detI);
                        max_coupling.first = std::max(max_coupling.first, std::fabs(HJI));
                        if (std::fabs(HJI * ref_CI) >= spawning_threshold) {
                            double smooth_factor = polynomial_smoother(
                                spawning_threshold, smooth_edge1, std::fabs(HJI * ref_CI));
                            new_space_C_vec.push_back(
                                std::make_pair(detJ, smooth_factor * tau * HJI * CI));

                            det_hash_it it_pre = pre_dets_C_hash.find(detJ);
                            if (it_pre != pre_dets_C_hash.end()) {
                                det_hash_it it_ref = ref_dets_C_hash.find(detJ);
                                if (it_ref == ref_dets_C_hash.end() ||
                                    std::fabs(HJI * it_ref->second) < spawning_threshold) {
                                    //                                if (it ==
                                    //                                pre_dets_C_hash.end())
                                    //                                {
                                    //                                    outfile
                                    //                                    ->
                                    //                                    Printf("\n\nERROR:
                                    //                                    apply_tau_H_ref_C_symm_det_dynamic
                                    //                                    aa det
                                    //                                    NOT
                                    //                                    FOUND
                                    //                                    in
                                    //                                    pre_dets_C_hash");
                                    //                                }
                                    new_space_C_vec[0].second +=
                                        smooth_factor * tau * HJI * it_pre->second;
                                    //                                outfile->Printf(",
                                    //                                then:
                                    //                                %.12f",
                                    //                                new_space_C_vec[0].second);
                                }
                            }
                        }
                    }
                }
            }
        }

        if (do_doubles) {
            // Generate alpha-alpha excitations
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
                                double HJI = as_ints_->tei_aa(ii, jj, aa, bb);
                                max_coupling.second = std::max(max_coupling.second, std::fabs(HJI));
                                if (std::fabs(HJI * ref_CI) >= spawning_threshold) {
                                    double smooth_factor = polynomial_smoother(
                                        spawning_threshold, smooth_edge1, std::fabs(HJI * ref_CI));
                                    Determinant detJ(detI);
                                    HJI *= detJ.double_excitation_aa(ii, jj, aa, bb);
                                    new_space_C_vec.push_back(
                                        std::make_pair(detJ, smooth_factor * tau * HJI * CI));

                                    det_hash_it it_pre = pre_dets_C_hash.find(detJ);
                                    if (it_pre != pre_dets_C_hash.end()) {
                                        det_hash_it it_ref = ref_dets_C_hash.find(detJ);
                                        if (it_ref == ref_dets_C_hash.end() ||
                                            std::fabs(HJI * it_ref->second) < spawning_threshold) {
                                            //                                if
                                            //                                (it
                                            //                                ==
                                            //                                pre_dets_C_hash.end())
                                            //                                {
                                            //                                    outfile ->
                                            //                                    Printf("\n\nERROR:
                                            //                                    apply_tau_H_ref_C_symm_det_dynamic
                                            //                                    aa det NOT FOUND
                                            //                                    in
                                            //                                    pre_dets_C_hash");
                                            //                                }
                                            new_space_C_vec[0].second +=
                                                smooth_factor * tau * HJI * it_pre->second;
                                            //                                outfile->Printf(",
                                            //                                then:
                                            //                                %.12f",
                                            //                                new_space_C_vec[0].second);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            // Generate alpha-beta excitations
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
                                double HJI = as_ints_->tei_ab(ii, jj, aa, bb);
                                max_coupling.second = std::max(max_coupling.second, std::fabs(HJI));
                                if (std::fabs(HJI * ref_CI) >= spawning_threshold) {
                                    double smooth_factor = polynomial_smoother(
                                        spawning_threshold, smooth_edge1, std::fabs(HJI * ref_CI));
                                    Determinant detJ(detI);
                                    HJI *= detJ.double_excitation_ab(ii, jj, aa, bb);
                                    new_space_C_vec.push_back(
                                        std::make_pair(detJ, smooth_factor * tau * HJI * CI));

                                    det_hash_it it_pre = pre_dets_C_hash.find(detJ);
                                    if (it_pre != pre_dets_C_hash.end()) {
                                        det_hash_it it_ref = ref_dets_C_hash.find(detJ);
                                        if (it_ref == ref_dets_C_hash.end() ||
                                            std::fabs(HJI * it_ref->second) < spawning_threshold) {
                                            //                                if
                                            //                                (it
                                            //                                ==
                                            //                                pre_dets_C_hash.end())
                                            //                                {
                                            //                                    outfile ->
                                            //                                    Printf("\n\nERROR:
                                            //                                    apply_tau_H_ref_C_symm_det_dynamic
                                            //                                    aa det NOT FOUND
                                            //                                    in
                                            //                                    pre_dets_C_hash");
                                            //                                }
                                            new_space_C_vec[0].second +=
                                                smooth_factor * tau * HJI * it_pre->second;
                                            //                                outfile->Printf(",
                                            //                                then:
                                            //                                %.12f",
                                            //                                new_space_C_vec[0].second);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            // Generate beta-beta excitations
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
                                double HJI = as_ints_->tei_bb(ii, jj, aa, bb);
                                max_coupling.second = std::max(max_coupling.second, std::fabs(HJI));
                                if (std::fabs(HJI * ref_CI) >= spawning_threshold) {
                                    double smooth_factor = polynomial_smoother(
                                        spawning_threshold, smooth_edge1, std::fabs(HJI * ref_CI));
                                    Determinant detJ(detI);
                                    HJI *= detJ.double_excitation_bb(ii, jj, aa, bb);
                                    new_space_C_vec.push_back(
                                        std::make_pair(detJ, smooth_factor * tau * HJI * CI));

                                    det_hash_it it_pre = pre_dets_C_hash.find(detJ);
                                    if (it_pre != pre_dets_C_hash.end()) {
                                        det_hash_it it_ref = ref_dets_C_hash.find(detJ);
                                        if (it_ref == ref_dets_C_hash.end() ||
                                            std::fabs(HJI * it_ref->second) < spawning_threshold) {
                                            //                                if
                                            //                                (it
                                            //                                ==
                                            //                                pre_dets_C_hash.end())
                                            //                                {
                                            //                                    outfile ->
                                            //                                    Printf("\n\nERROR:
                                            //                                    apply_tau_H_ref_C_symm_det_dynamic
                                            //                                    aa det NOT FOUND
                                            //                                    in
                                            //                                    pre_dets_C_hash");
                                            //                                }
                                            new_space_C_vec[0].second +=
                                                smooth_factor * tau * HJI * it_pre->second;
                                            //                                outfile->Printf(",
                                            //                                then:
                                            //                                %.12f",
                                            //                                new_space_C_vec[0].second);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

void ProjectorCI::apply_tau_H_ref_C_symm_det_dynamic(
    double tau, double spawning_threshold, det_hash<>& pre_dets_C_hash, det_hash<>& ref_dets_C_hash,
    const Determinant& detI, double CI, double ref_CI,
    std::vector<std::pair<Determinant, double>>& new_space_C_vec, double E0,
    std::pair<double, double>& max_coupling) {
    //    outfile -> Printf("\napply_tau_H_ref_C_symm_det_dynamic : Beginning
    //    args:");
    //    outfile -> Printf("\n CI: %lf, ref_CI: %lf\n", CI, ref_CI);
    bool do_singles = (max_coupling.first == 0.0) or
                      (std::fabs(max_coupling.first * ref_CI) >= spawning_threshold);
    bool do_doubles = (max_coupling.second == 0.0) or
                      (std::fabs(max_coupling.second * ref_CI) >= spawning_threshold);

    // Diagonal contributions
    double det_energy = as_ints_->energy(detI) + as_ints_->scalar_energy();
    new_space_C_vec.push_back(std::make_pair(detI, tau * (det_energy - E0) * CI));

    if (do_singles or do_doubles) {

        std::vector<int> aocc = detI.get_alfa_occ(nact_);
        std::vector<int> bocc = detI.get_beta_occ(nact_);
        std::vector<int> avir = detI.get_alfa_vir(nact_);
        std::vector<int> bvir = detI.get_beta_vir(nact_);

        int noalpha = aocc.size();
        int nobeta = bocc.size();
        int nvalpha = avir.size();
        int nvbeta = bvir.size();

        if (do_singles) {
            // Generate alpha excitations
            for (int i = 0; i < noalpha; ++i) {
                int ii = aocc[i];
                for (int a = 0; a < nvalpha; ++a) {
                    int aa = avir[a];
                    if ((mo_symmetry_[ii] ^ mo_symmetry_[aa]) == 0) {
                        Determinant detJ(detI);
                        detJ.set_alfa_bit(ii, false);
                        detJ.set_alfa_bit(aa, true);
                        double HJI = as_ints_->slater_rules(detJ, detI);
                        max_coupling.first = std::max(max_coupling.first, std::fabs(HJI));
                        if (std::fabs(HJI * ref_CI) >= spawning_threshold) {
                            new_space_C_vec.push_back(std::make_pair(detJ, tau * HJI * CI));

                            det_hash_it it_pre = pre_dets_C_hash.find(detJ);
                            if (it_pre != pre_dets_C_hash.end()) {
                                det_hash_it it_ref = ref_dets_C_hash.find(detJ);
                                if (it_ref == ref_dets_C_hash.end() ||
                                    std::fabs(HJI * it_ref->second) < spawning_threshold) {
                                    //                                if (it ==
                                    //                                pre_dets_C_hash.end())
                                    //                                {
                                    //                                    outfile
                                    //                                    ->
                                    //                                    Printf("\n\nERROR:
                                    //                                    apply_tau_H_ref_C_symm_det_dynamic
                                    //                                    aa det
                                    //                                    NOT
                                    //                                    FOUND
                                    //                                    in
                                    //                                    pre_dets_C_hash");
                                    //                                }
                                    new_space_C_vec[0].second += tau * HJI * it_pre->second;
                                    //                                outfile->Printf(",
                                    //                                then:
                                    //                                %.12f",
                                    //                                new_space_C_vec[0].second);
                                }
                            }
                        }
                    }
                }
            }
            // Generate beta excitations
            for (int i = 0; i < nobeta; ++i) {
                int ii = bocc[i];
                for (int a = 0; a < nvbeta; ++a) {
                    int aa = bvir[a];
                    if ((mo_symmetry_[ii] ^ mo_symmetry_[aa]) == 0) {
                        Determinant detJ(detI);
                        detJ.set_beta_bit(ii, false);
                        detJ.set_beta_bit(aa, true);
                        double HJI = as_ints_->slater_rules(detJ, detI);
                        max_coupling.first = std::max(max_coupling.first, std::fabs(HJI));
                        if (std::fabs(HJI * ref_CI) >= spawning_threshold) {
                            new_space_C_vec.push_back(std::make_pair(detJ, tau * HJI * CI));

                            det_hash_it it_pre = pre_dets_C_hash.find(detJ);
                            if (it_pre != pre_dets_C_hash.end()) {
                                det_hash_it it_ref = ref_dets_C_hash.find(detJ);
                                if (it_ref == ref_dets_C_hash.end() ||
                                    std::fabs(HJI * it_ref->second) < spawning_threshold) {
                                    //                                if (it ==
                                    //                                pre_dets_C_hash.end())
                                    //                                {
                                    //                                    outfile
                                    //                                    ->
                                    //                                    Printf("\n\nERROR:
                                    //                                    apply_tau_H_ref_C_symm_det_dynamic
                                    //                                    aa det
                                    //                                    NOT
                                    //                                    FOUND
                                    //                                    in
                                    //                                    pre_dets_C_hash");
                                    //                                }
                                    new_space_C_vec[0].second += tau * HJI * it_pre->second;
                                    //                                outfile->Printf(",
                                    //                                then:
                                    //                                %.12f",
                                    //                                new_space_C_vec[0].second);
                                }
                            }
                        }
                    }
                }
            }
        }

        if (do_doubles) {
            // Generate alpha-alpha excitations
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
                                double HJI = as_ints_->tei_aa(ii, jj, aa, bb);
                                max_coupling.second = std::max(max_coupling.second, std::fabs(HJI));
                                if (std::fabs(HJI * ref_CI) >= spawning_threshold) {
                                    Determinant detJ(detI);
                                    HJI *= detJ.double_excitation_aa(ii, jj, aa, bb);
                                    new_space_C_vec.push_back(std::make_pair(detJ, tau * HJI * CI));

                                    det_hash_it it_pre = pre_dets_C_hash.find(detJ);
                                    if (it_pre != pre_dets_C_hash.end()) {
                                        det_hash_it it_ref = ref_dets_C_hash.find(detJ);
                                        if (it_ref == ref_dets_C_hash.end() ||
                                            std::fabs(HJI * it_ref->second) < spawning_threshold) {
                                            //                                if
                                            //                                (it
                                            //                                ==
                                            //                                pre_dets_C_hash.end())
                                            //                                {
                                            //                                    outfile ->
                                            //                                    Printf("\n\nERROR:
                                            //                                    apply_tau_H_ref_C_symm_det_dynamic
                                            //                                    aa det NOT FOUND
                                            //                                    in
                                            //                                    pre_dets_C_hash");
                                            //                                }
                                            new_space_C_vec[0].second += tau * HJI * it_pre->second;
                                            //                                outfile->Printf(",
                                            //                                then:
                                            //                                %.12f",
                                            //                                new_space_C_vec[0].second);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            // Generate alpha-beta excitations
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
                                double HJI = as_ints_->tei_ab(ii, jj, aa, bb);
                                max_coupling.second = std::max(max_coupling.second, std::fabs(HJI));
                                if (std::fabs(HJI * ref_CI) >= spawning_threshold) {
                                    Determinant detJ(detI);
                                    HJI *= detJ.double_excitation_ab(ii, jj, aa, bb);
                                    new_space_C_vec.push_back(std::make_pair(detJ, tau * HJI * CI));

                                    det_hash_it it_pre = pre_dets_C_hash.find(detJ);
                                    if (it_pre != pre_dets_C_hash.end()) {
                                        det_hash_it it_ref = ref_dets_C_hash.find(detJ);
                                        if (it_ref == ref_dets_C_hash.end() ||
                                            std::fabs(HJI * it_ref->second) < spawning_threshold) {
                                            //                                if
                                            //                                (it
                                            //                                ==
                                            //                                pre_dets_C_hash.end())
                                            //                                {
                                            //                                    outfile ->
                                            //                                    Printf("\n\nERROR:
                                            //                                    apply_tau_H_ref_C_symm_det_dynamic
                                            //                                    aa det NOT FOUND
                                            //                                    in
                                            //                                    pre_dets_C_hash");
                                            //                                }
                                            new_space_C_vec[0].second += tau * HJI * it_pre->second;
                                            //                                outfile->Printf(",
                                            //                                then:
                                            //                                %.12f",
                                            //                                new_space_C_vec[0].second);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            // Generate beta-beta excitations
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
                                double HJI = as_ints_->tei_bb(ii, jj, aa, bb);
                                max_coupling.second = std::max(max_coupling.second, std::fabs(HJI));
                                if (std::fabs(HJI * ref_CI) >= spawning_threshold) {
                                    Determinant detJ(detI);
                                    HJI *= detJ.double_excitation_bb(ii, jj, aa, bb);
                                    new_space_C_vec.push_back(std::make_pair(detJ, tau * HJI * CI));

                                    det_hash_it it_pre = pre_dets_C_hash.find(detJ);
                                    if (it_pre != pre_dets_C_hash.end()) {
                                        det_hash_it it_ref = ref_dets_C_hash.find(detJ);
                                        if (it_ref == ref_dets_C_hash.end() ||
                                            std::fabs(HJI * it_ref->second) < spawning_threshold) {
                                            //                                if
                                            //                                (it
                                            //                                ==
                                            //                                pre_dets_C_hash.end())
                                            //                                {
                                            //                                    outfile ->
                                            //                                    Printf("\n\nERROR:
                                            //                                    apply_tau_H_ref_C_symm_det_dynamic
                                            //                                    aa det NOT FOUND
                                            //                                    in
                                            //                                    pre_dets_C_hash");
                                            //                                }
                                            new_space_C_vec[0].second += tau * HJI * it_pre->second;
                                            //                                outfile->Printf(",
                                            //                                then:
                                            //                                %.12f",
                                            //                                new_space_C_vec[0].second);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

void ProjectorCI::apply_tau_H(double tau, double spawning_threshold, det_vec& dets,
                              const std::vector<double>& C, det_hash<>& dets_C_hash, double S) {
    // A vector of maps that hold (determinant,coefficient)
    //    std::vector<det_hash<>> thread_det_C_hash(num_threads_);
    std::vector<std::pair<double, double>> thread_max_HJI(num_threads_);

    if (do_dynamic_prescreening_) {
        size_t max_I = dets.size();
#pragma omp parallel for
        for (size_t I = 0; I < max_I; ++I) {
            std::pair<double, double> zero_pair(0.0, 0.0);
            // Update the list of couplings
            std::pair<double, double> max_coupling;
#pragma omp critical
            { max_coupling = dets_max_couplings_[dets[I]]; }
            if (max_coupling == zero_pair) {
                std::vector<std::pair<Determinant, double>> thread_det_C_vec;
                apply_tau_H_det_dynamic(tau, spawning_threshold, dets[I], C[I], thread_det_C_vec, S,
                                        max_coupling);
#pragma omp critical
                {
                    for (auto det_C : thread_det_C_vec) {
                        dets_C_hash[det_C.first] += det_C.second;
                    }
                }
#pragma omp critical
                { dets_max_couplings_[dets[I]] = max_coupling; }
            } else {
                std::vector<std::pair<Determinant, double>> thread_det_C_vec;
                apply_tau_H_det_dynamic(tau, spawning_threshold, dets[I], C[I], thread_det_C_vec, S,
                                        max_coupling);
#pragma omp critical
                {
                    for (auto det_C : thread_det_C_vec) {
                        dets_C_hash[det_C.first] += det_C.second;
                    }
                }
            }
        }
        // Combine the results of all the threads
    } else if (do_schwarz_prescreening_) {
        if (do_initiator_approx_) {
            size_t max_I = dets.size();
#pragma omp parallel for
            for (size_t I = 0; I < max_I; ++I) {
                if (std::fabs(C[I]) >= initiator_approx_factor_ * spawning_threshold) {
                    std::vector<std::pair<Determinant, double>> thread_det_C_vec;
                    apply_tau_H_det_schwarz(tau, spawning_threshold, dets[I], C[I],
                                            thread_det_C_vec, S);
#pragma omp critical
                    {
                        for (auto det_C : thread_det_C_vec) {
                            dets_C_hash[det_C.first] += det_C.second;
                        }
                    }
                } else {
                    // Diagonal contribution
                    double det_energy = as_ints_->energy(dets[I]) + as_ints_->scalar_energy();
// Diagonal contributions
#pragma omp critical
                    { dets_C_hash[dets[I]] += tau * (det_energy - S) * C[I]; }
                }
            }
        } else {
            size_t max_I = dets.size();
#pragma omp parallel for
            for (size_t I = 0; I < max_I; ++I) {
                std::vector<std::pair<Determinant, double>> thread_det_C_vec;
                apply_tau_H_det_schwarz(tau, spawning_threshold, dets[I], C[I], thread_det_C_vec,
                                        S);
#pragma omp critical
                {
                    for (auto det_C : thread_det_C_vec) {
                        dets_C_hash[det_C.first] += det_C.second;
                    }
                }
            }
        }

        // Combine the results of all the threads
    } else if (do_initiator_approx_) {
        size_t max_I = dets.size();
#pragma omp parallel for
        for (size_t I = 0; I < max_I; ++I) {
            int thread_id = omp_get_thread_num();
            if (std::fabs(C[I]) >= initiator_approx_factor_ * spawning_threshold) {
                std::vector<std::pair<Determinant, double>> thread_det_C_vec;
                const std::pair<double, double> max_HJI = apply_tau_H_det_prescreening(
                    tau, spawning_threshold, dets[I], C[I], thread_det_C_vec, S);
#pragma omp critical
                {
                    for (auto det_C : thread_det_C_vec) {
                        dets_C_hash[det_C.first] += det_C.second;
                    }
                }
                thread_max_HJI[thread_id].first =
                    std::max(thread_max_HJI[thread_id].first,
                             max_HJI.first); // to avoid race condition
                thread_max_HJI[thread_id].second =
                    std::max(thread_max_HJI[thread_id].second,
                             max_HJI.second); // to avoid race condition
            } else {
                double det_energy = as_ints_->energy(dets[I]) + as_ints_->scalar_energy();
// Diagonal contributions
#pragma omp critical
                { dets_C_hash[dets[I]] += tau * (det_energy - S) * C[I]; }
            }
        }
        // Combine the bounds on the couplings
        for (int t = 0; t < num_threads_; ++t) {
            new_max_one_HJI_ = std::max(thread_max_HJI[t].first, new_max_one_HJI_);
            new_max_two_HJI_ = std::max(thread_max_HJI[t].second, new_max_two_HJI_);
        }
    } else {
        size_t max_I = dets.size();
#pragma omp parallel for
        for (size_t I = 0; I < max_I; ++I) {
            int thread_id = omp_get_thread_num();
            std::vector<std::pair<Determinant, double>> thread_det_C_vec;
            const std::pair<double, double> max_HJI = apply_tau_H_det_prescreening(
                tau, spawning_threshold, dets[I], C[I], thread_det_C_vec, S);
#pragma omp critical
            {
                for (auto det_C : thread_det_C_vec) {
                    dets_C_hash[det_C.first] += det_C.second;
                }
            }
            thread_max_HJI[thread_id].first = std::max(thread_max_HJI[thread_id].first,
                                                       max_HJI.first); // to avoid race condition
            thread_max_HJI[thread_id].second = std::max(thread_max_HJI[thread_id].second,
                                                        max_HJI.second); // to avoid race condition
        }
        // Combine the bounds on the couplings
        for (int t = 0; t < num_threads_; ++t) {
            new_max_one_HJI_ = std::max(thread_max_HJI[t].first, new_max_one_HJI_);
            new_max_two_HJI_ = std::max(thread_max_HJI[t].second, new_max_two_HJI_);
        }
    }
    if (approx_E_flag_) {
        timer_on("PCI:<E>a");
        size_t max_I = dets.size();
        double CHC_energy = 0.0;
#pragma omp parallel for reduction(+ : CHC_energy)
        for (size_t I = 0; I < max_I; ++I) {
            CHC_energy += C[I] * dets_C_hash[dets[I]];
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

void ProjectorCI::apply_tau_H_det_subset(
    double tau, Determinant& detI, double CI, det_hash<>& dets_sum_map,
    std::vector<std::pair<Determinant, double>>& new_space_C_vec, double E0) {
    // Diagonal contributions
    double det_energy = as_ints_->energy(detI) + as_ints_->scalar_energy();
    //    new_space_C[detI] += tau * (det_energy - E0) * CI;
    new_space_C_vec.push_back(std::make_pair(detI, tau * (det_energy - E0) * CI));

    std::vector<int> aocc = detI.get_alfa_occ(nact_);
    std::vector<int> bocc = detI.get_beta_occ(nact_);
    std::vector<int> avir = detI.get_alfa_vir(nact_);
    std::vector<int> bvir = detI.get_beta_vir(nact_);
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

    // do_singles
    Determinant detJ(detI);
    // Generate aa excitations
    for (int h = 0; h < nirrep_; ++h) {
        for (int i = aocc_offset[h]; i < aocc_offset[h + 1]; ++i) {
            int ii = aocc[i];
            for (int a = avir_offset[h]; a < avir_offset[h + 1]; ++a) {
                int aa = avir[a];
                detJ = detI;
                detJ.set_alfa_bit(ii, false);
                detJ.set_alfa_bit(aa, true);
                if (dets_sum_map.end() != dets_sum_map.find(detJ)) {
                    double HJI = as_ints_->slater_rules_single_alpha(detI, ii, aa);
                    new_space_C_vec.push_back(std::make_pair(detJ, tau * HJI * CI));
                }
            }
        }
    }
    // Generate bb excitations
    for (int h = 0; h < nirrep_; ++h) {
        for (int i = bocc_offset[h]; i < bocc_offset[h + 1]; ++i) {
            int ii = bocc[i];
            for (int a = bvir_offset[h]; a < bvir_offset[h + 1]; ++a) {
                int aa = bvir[a];
                detJ = detI;
                detJ.set_beta_bit(ii, false);
                detJ.set_beta_bit(aa, true);
                if (dets_sum_map.end() != dets_sum_map.find(detJ)) {
                    double HJI = as_ints_->slater_rules_single_beta(detI, ii, aa);
                    new_space_C_vec.push_back(std::make_pair(detJ, tau * HJI * CI));
                }
            }
        }
    }

    // do_doubles
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
                    detJ = detI;
                    double HJI = detJ.double_excitation_aa(ii, jj, aa, bb);
                    if (dets_sum_map.end() != dets_sum_map.find(detJ)) {
                        HJI *= as_ints_->tei_aa(ii, jj, aa, bb);
                        new_space_C_vec.push_back(std::make_pair(detJ, tau * HJI * CI));
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
                int h = mo_symmetry_[ii] ^ mo_symmetry_[jj] ^ mo_symmetry_[aa];
                int minb = bvir_offset[h];
                int maxb = bvir_offset[h + 1];
                for (int b = minb; b < maxb; ++b) {
                    int bb = bvir[b];
                    detJ = detI;
                    double HJI = detJ.double_excitation_ab(ii, jj, aa, bb);
                    if (dets_sum_map.end() != dets_sum_map.find(detJ)) {
                        HJI *= as_ints_->tei_ab(ii, jj, aa, bb);
                        new_space_C_vec.push_back(std::make_pair(detJ, tau * HJI * CI));
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
                int h = mo_symmetry_[ii] ^ mo_symmetry_[jj] ^ mo_symmetry_[aa];
                if (h < mo_symmetry_[aa])
                    continue;
                int minb = h == mo_symmetry_[aa] ? a + 1 : bvir_offset[h];
                int maxb = bvir_offset[h + 1];
                for (int b = minb; b < maxb; ++b) {
                    int bb = bvir[b];
                    detJ = detI;
                    double HJI = detJ.double_excitation_bb(ii, jj, aa, bb);
                    if (dets_sum_map.end() != dets_sum_map.find(detJ)) {
                        HJI *= as_ints_->tei_bb(ii, jj, aa, bb);
                        new_space_C_vec.push_back(std::make_pair(detJ, tau * HJI * CI));
                    }
                }
            }
        }
    }
}

void ProjectorCI::apply_tau_H_det_subset_prescreening(
    double tau, double spawning_threshold, Determinant& detI, double CI, det_hash<>& dets_sum_map,
    std::vector<std::pair<Determinant, double>>& new_space_C_vec, double E0) {
    // Diagonal contributions
    double det_energy = as_ints_->energy(detI) + as_ints_->scalar_energy();
    //    new_space_C[detI] += tau * (det_energy - E0) * CI;
    new_space_C_vec.push_back(std::make_pair(detI, tau * (det_energy - E0) * CI));

    std::vector<int> aocc = detI.get_alfa_occ(nact_);
    std::vector<int> bocc = detI.get_beta_occ(nact_);
    std::vector<int> avir = detI.get_alfa_vir(nact_);
    std::vector<int> bvir = detI.get_beta_vir(nact_);
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

    // do_singles
    Determinant detJ(detI);
    // Generate aa excitations
    for (int h = 0; h < nirrep_; ++h) {
        for (int i = aocc_offset[h]; i < aocc_offset[h + 1]; ++i) {
            int ii = aocc[i];
            for (int a = avir_offset[h]; a < avir_offset[h + 1]; ++a) {
                int aa = avir[a];
                double HJI = as_ints_->slater_rules_single_alpha(detI, ii, aa);
                //                my_new_max_one_HJI =
                //                std::max(my_new_max_one_HJI,std::fabs(HJI));
                if (std::fabs(HJI * CI) >= spawning_threshold) {
                    detJ = detI;
                    detJ.set_alfa_bit(ii, false);
                    detJ.set_alfa_bit(aa, true);
                    if (dets_sum_map.end() != dets_sum_map.find(detJ)) {
                        new_space_C_vec.push_back(std::make_pair(detJ, tau * HJI * CI));
                    }
                }
                //                detJ = detI;
                //                detJ.set_alfa_bit(ii,false);
                //                detJ.set_alfa_bit(aa,true);
                //                if (dets_sum_map.end() !=
                //                dets_sum_map.find(detJ)){
                //                    double HJI =
                //                    detI.slater_rules_single_alpha(ii,aa);
                //                    new_space_C_vec.push_back(std::make_pair(detJ,
                //                    tau * HJI * CI));
                //                }
            }
        }
    }
    // Generate bb excitations
    for (int h = 0; h < nirrep_; ++h) {
        for (int i = bocc_offset[h]; i < bocc_offset[h + 1]; ++i) {
            int ii = bocc[i];
            for (int a = bvir_offset[h]; a < bvir_offset[h + 1]; ++a) {
                int aa = bvir[a];
                double HJI = as_ints_->slater_rules_single_beta(detI, ii, aa);
                //                my_new_max_one_HJI =
                //                std::max(my_new_max_one_HJI,std::fabs(HJI));
                if (std::fabs(HJI * CI) >= spawning_threshold) {
                    detJ = detI;
                    detJ.set_beta_bit(ii, false);
                    detJ.set_beta_bit(aa, true);
                    if (dets_sum_map.end() != dets_sum_map.find(detJ)) {
                        new_space_C_vec.push_back(std::make_pair(detJ, tau * HJI * CI));
                    }
                }
                //                detJ = detI;
                //                detJ.set_beta_bit(ii,false);
                //                detJ.set_beta_bit(aa,true);
                //                if (dets_sum_map.end() !=
                //                dets_sum_map.find(detJ)){
                //                    double HJI =
                //                    detI.slater_rules_single_beta(ii,aa);
                //                    new_space_C_vec.push_back(std::make_pair(detJ,
                //                    tau * HJI * CI));
                //                }
            }
        }
    }

    // do_doubles
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
                    //                    my_new_max_two_HJI =
                    //                    std::max(my_new_max_two_HJI,std::fabs(HJI));
                    if (std::fabs(HJI * CI) >= spawning_threshold) {
                        detJ = detI;
                        HJI *= detJ.double_excitation_aa(ii, jj, aa, bb);
                        if (dets_sum_map.end() != dets_sum_map.find(detJ)) {
                            new_space_C_vec.push_back(std::make_pair(detJ, tau * HJI * CI));
                        }
                    }
                    //                    detJ = detI;
                    //                    double HJI =
                    //                    detJ.double_excitation_aa(ii,jj,aa,bb);
                    //                    if (dets_sum_map.end() !=
                    //                    dets_sum_map.find(detJ)){
                    //                        HJI *=
                    //                        as_ints_->tei_aa(ii,jj,aa,bb);
                    //                        new_space_C_vec.push_back(std::make_pair(detJ,
                    //                        tau * HJI * CI));
                    //                    }
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
                    //                    my_new_max_two_HJI =
                    //                    std::max(my_new_max_two_HJI,std::fabs(HJI));

                    if (std::fabs(HJI * CI) >= spawning_threshold) {
                        detJ = detI;
                        HJI *= detJ.double_excitation_ab(ii, jj, aa, bb);
                        if (dets_sum_map.end() != dets_sum_map.find(detJ)) {
                            new_space_C_vec.push_back(std::make_pair(detJ, tau * HJI * CI));
                        }
                    }
                    //                    detJ = detI;
                    //                    double HJI =
                    //                    detJ.double_excitation_ab(ii,jj,aa,bb);
                    //                    if (dets_sum_map.end() !=
                    //                    dets_sum_map.find(detJ)){
                    //                        HJI *=
                    //                        as_ints_->tei_ab(ii,jj,aa,bb);
                    //                        new_space_C_vec.push_back(std::make_pair(detJ,
                    //                        tau * HJI * CI));
                    //                    }
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
                    //                    my_new_max_two_HJI =
                    //                    std::max(my_new_max_two_HJI,std::fabs(HJI));

                    if (std::fabs(HJI * CI) >= spawning_threshold) {
                        detJ = detI;
                        HJI *= detJ.double_excitation_bb(ii, jj, aa, bb);
                        if (dets_sum_map.end() != dets_sum_map.find(detJ)) {
                            new_space_C_vec.push_back(std::make_pair(detJ, tau * HJI * CI));
                        }
                    }
                    //                    detJ = detI;
                    //                    double HJI =
                    //                    detJ.double_excitation_bb(ii,jj,aa,bb);
                    //                    if (dets_sum_map.end() !=
                    //                    dets_sum_map.find(detJ)){
                    //                        HJI *=
                    //                        as_ints_->tei_bb(ii,jj,aa,bb);
                    //                        new_space_C_vec.push_back(std::make_pair(detJ,
                    //                        tau * HJI * CI));
                    //                    }
                }
            }
        }
    }
}

void ProjectorCI::apply_tau_H_subset(double tau, double spawning_threshold, det_vec& dets,
                                     const std::vector<double>& C, det_hash<>& dets_sum_map,
                                     det_hash<>& dets_C_hash, double S) {
    //    size_t max_I = dets.size();
    //#pragma omp parallel for
    //    for (size_t I = 0; I < max_I; ++I){
    //        double CI = 0.0;
    //        for (size_t j = 0; j < max_I; ++j) {
    //            CI += dets[I].slater_rules(dets[j]) * C[j];
    //        }
    //        CI *= tau;
    //        #pragma omp critical
    //        {
    //            dets_C_hash[dets[I]] = CI;
    //        }
    //    }
    size_t max_I = dets.size();
#pragma omp parallel for
    for (size_t I = 0; I < max_I; ++I) {
        std::vector<std::pair<Determinant, double>> thread_det_C_vec;
        //        apply_tau_H_det_subset(tau,dets[I],C[I],
        //        dets_sum_map,thread_det_C_vec,S);
        apply_tau_H_det_subset_prescreening(tau, spawning_threshold, dets[I], C[I], dets_sum_map,
                                            thread_det_C_vec, S);
#pragma omp critical
        {
            for (auto det_C : thread_det_C_vec) {
                dets_C_hash[det_C.first] += det_C.second;
            }
        }
    }
}

std::pair<double, double> ProjectorCI::apply_tau_H_det_prescreening(
    double tau, double spawning_threshold, Determinant& detI, double CI,
    std::vector<std::pair<Determinant, double>>& new_space_C_vec, double E0) {
    bool do_singles =
        std::fabs(prescreening_tollerance_factor_ * old_max_one_HJI_ * CI) >= spawning_threshold;
    bool do_doubles =
        std::fabs(prescreening_tollerance_factor_ * old_max_two_HJI_ * CI) >= spawning_threshold;

    // Diagonal contributions
    double det_energy = as_ints_->energy(detI) + as_ints_->scalar_energy();
    //    new_space_C[detI] += tau * (det_energy - E0) * CI;
    new_space_C_vec.push_back(std::make_pair(detI, tau * (det_energy - E0) * CI));

    if (do_singles or do_doubles) {
        std::vector<int> aocc = detI.get_alfa_occ(nact_);
        std::vector<int> bocc = detI.get_beta_occ(nact_);
        std::vector<int> avir = detI.get_alfa_vir(nact_);
        std::vector<int> bvir = detI.get_beta_vir(nact_);
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

        double my_new_max_one_HJI = 0.0;
        double my_new_max_two_HJI = 0.0;

        if (do_singles) {
            Determinant detJ(detI);
#if ENFORCE_SYM
            // Generate aa excitations
            for (int h = 0; h < nirrep_; ++h) {
                for (int i = aocc_offset[h]; i < aocc_offset[h + 1]; ++i) {
                    int ii = aocc[i];
                    for (int a = avir_offset[h]; a < avir_offset[h + 1]; ++a) {
                        int aa = avir[a];
                        double HJI = as_ints_->slater_rules_single_alpha(detI, ii, aa);
                        my_new_max_one_HJI = std::max(my_new_max_one_HJI, std::fabs(HJI));
                        if (std::fabs(HJI * CI) >= spawning_threshold) {
                            detJ = detI;
                            detJ.set_alfa_bit(ii, false);
                            detJ.set_alfa_bit(aa, true);
                            //                            new_space_C[detJ] +=
                            //                            tau * HJI * CI;
                            new_space_C_vec.push_back(std::make_pair(detJ, tau * HJI * CI));
                        }
                    }
                }
            }
            // Generate bb excitations
            for (int h = 0; h < nirrep_; ++h) {
                for (int i = bocc_offset[h]; i < bocc_offset[h + 1]; ++i) {
                    int ii = bocc[i];
                    for (int a = bvir_offset[h]; a < bvir_offset[h + 1]; ++a) {
                        int aa = bvir[a];
                        double HJI = as_ints_->slater_rules_single_beta(detI, ii, aa);
                        my_new_max_one_HJI = std::max(my_new_max_one_HJI, std::fabs(HJI));
                        if (std::fabs(HJI * CI) >= spawning_threshold) {
                            detJ = detI;
                            detJ.set_beta_bit(ii, false);
                            detJ.set_beta_bit(aa, true);
                            //                            new_space_C[detJ] +=
                            //                            tau * HJI * CI;
                            new_space_C_vec.push_back(std::make_pair(detJ, tau * HJI * CI));
                        }
                    }
                }
            }
        }

        if (do_doubles) {
            Determinant detJ(detI);
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
                            my_new_max_two_HJI = std::max(my_new_max_two_HJI, std::fabs(HJI));
                            if (std::fabs(HJI * CI) >= spawning_threshold) {
                                detJ = detI;
                                HJI *= detJ.double_excitation_aa(ii, jj, aa, bb);
                                //                                new_space_C[detJ]
                                //                                += tau * HJI *
                                //                                CI;
                                new_space_C_vec.push_back(std::make_pair(detJ, tau * HJI * CI));
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
                        int h = mo_symmetry_[ii] ^ mo_symmetry_[jj] ^ mo_symmetry_[aa];
                        int minb = bvir_offset[h];
                        int maxb = bvir_offset[h + 1];
                        for (int b = minb; b < maxb; ++b) {
                            int bb = bvir[b];
                            double HJI = as_ints_->tei_ab(ii, jj, aa, bb);
                            my_new_max_two_HJI = std::max(my_new_max_two_HJI, std::fabs(HJI));

                            if (std::fabs(HJI * CI) >= spawning_threshold) {
                                detJ = detI;
                                HJI *= detJ.double_excitation_ab(ii, jj, aa, bb);
                                //                                new_space_C[detJ]
                                //                                += tau * HJI *
                                //                                CI;
                                new_space_C_vec.push_back(std::make_pair(detJ, tau * HJI * CI));
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
                        int h = mo_symmetry_[ii] ^ mo_symmetry_[jj] ^ mo_symmetry_[aa];
                        if (h < mo_symmetry_[aa])
                            continue;
                        int minb = h == mo_symmetry_[aa] ? a + 1 : bvir_offset[h];
                        int maxb = bvir_offset[h + 1];
                        for (int b = minb; b < maxb; ++b) {
                            int bb = bvir[b];
                            double HJI = as_ints_->tei_bb(ii, jj, aa, bb);
                            my_new_max_two_HJI = std::max(my_new_max_two_HJI, std::fabs(HJI));

                            if (std::fabs(HJI * CI) >= spawning_threshold) {
                                detJ = detI;
                                HJI *= detJ.double_excitation_bb(ii, jj, aa, bb);
                                //                                new_space_C[detJ]
                                //                                += tau * HJI *
                                //                                CI;
                                new_space_C_vec.push_back(std::make_pair(detJ, tau * HJI * CI));
                            }
                        }
                    }
                }
            }
        }
#else
            for (int i = 0; i < noalpha; ++i) {
                int ii = aocc[i];
                for (int a = 0; a < nvalpha; ++a) {
                    int aa = avir[a];
                    if ((mo_symmetry_[ii] ^ mo_symmetry_[aa]) == 0) {
                        double HJI = as_ints_->slater_rules_single_alpha(detI, ii, aa);
                        my_new_max_one_HJI = std::max(my_new_max_one_HJI, std::fabs(HJI));
                        if (std::fabs(HJI * CI) >= spawning_threshold) {
                            detJ = detI;
                            detJ.set_alfa_bit(ii, false);
                            detJ.set_alfa_bit(aa, true);
                            new_space_C[detJ] += tau * HJI * CI;
                        }
                    }
                }
            }

            for (int i = 0; i < nobeta; ++i) {
                int ii = bocc[i];
                for (int a = 0; a < nvbeta; ++a) {
                    int aa = bvir[a];
                    if ((mo_symmetry_[ii] ^ mo_symmetry_[aa]) == 0) {
                        double HJI = as_ints_->slater_rules_single_beta(detI, ii, aa);
                        my_new_max_one_HJI = std::max(my_new_max_one_HJI, std::fabs(HJI));
                        if (std::fabs(HJI * CI) >= spawning_threshold) {
                            detJ = detI;
                            detJ.set_beta_bit(ii, false);
                            detJ.set_beta_bit(aa, true);
                            new_space_C[detJ] += tau * HJI * CI;
                        }
                    }
                }
            }
        }

        if (do_doubles) {
            Determinant detJ(detI);
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
                                double HJI = as_ints_->tei_aa(ii, jj, aa, bb);
                                my_new_max_two_HJI = std::max(my_new_max_two_HJI, std::fabs(HJI));
                                if (std::fabs(HJI * CI) >= spawning_threshold) {
                                    detJ = detI;
                                    detJ.set_alfa_bit(ii, false);
                                    detJ.set_alfa_bit(jj, false);
                                    detJ.set_alfa_bit(aa, true);
                                    detJ.set_alfa_bit(bb, true);

                                    // grap the alpha bits of both determinants
                                    const Determinant::bit_t& Ia = detI.alfa_bits();
                                    const Determinant::bit_t& Ja = detJ.alfa_bits();

                                    // compute the sign of the matrix element
                                    HJI *= Determinant::SlaterSign(Ia, ii) *
                                           Determinant::SlaterSign(Ia, jj) *
                                           Determinant::SlaterSign(Ja, aa) *
                                           Determinant::SlaterSign(Ja, bb);
                                    new_space_C[detJ] += tau * HJI * CI;
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
                                double HJI = as_ints_->tei_ab(ii, jj, aa, bb);
                                my_new_max_two_HJI = std::max(my_new_max_two_HJI, std::fabs(HJI));

                                if (std::fabs(HJI * CI) >= spawning_threshold) {
                                    detJ = detI;
                                    detJ.set_alfa_bit(ii, false);
                                    detJ.set_beta_bit(jj, false);
                                    detJ.set_alfa_bit(aa, true);
                                    detJ.set_beta_bit(bb, true);

                                    // grap the alpha bits of both determinants
                                    const Determinant::bit_t& Ia = detI.alfa_bits();
                                    const Determinant::bit_t& Ib = detI.beta_bits();
                                    const Determinant::bit_t& Ja = detJ.alfa_bits();
                                    const Determinant::bit_t& Jb = detJ.beta_bits();

                                    // compute the sign of the matrix element
                                    HJI *= Determinant::SlaterSign(Ia, ii) *
                                           Determinant::SlaterSign(Ib, jj) *
                                           Determinant::SlaterSign(Ja, aa) *
                                           Determinant::SlaterSign(Jb, bb);
                                    new_space_C[detJ] += tau * HJI * CI;
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
                                double HJI = as_ints_->tei_bb(ii, jj, aa, bb);
                                my_new_max_two_HJI = std::max(my_new_max_two_HJI, std::fabs(HJI));

                                if (std::fabs(HJI * CI) >= spawning_threshold) {
                                    detJ = detI;
                                    detJ.set_beta_bit(ii, false);
                                    detJ.set_beta_bit(jj, false);
                                    detJ.set_beta_bit(aa, true);
                                    detJ.set_beta_bit(bb, true);

                                    // grap the alpha bits of both determinants
                                    const Determinant::bit_t& Ib = detI.beta_bits();
                                    const Determinant::bit_t& Jb = detJ.beta_bits();

                                    // compute the sign of the matrix element
                                    HJI *= Determinant::SlaterSign(Ib, ii) *
                                           Determinant::SlaterSign(Ib, jj) *
                                           Determinant::SlaterSign(Jb, aa) *
                                           Determinant::SlaterSign(Jb, bb);
                                    new_space_C[detJ] += tau * HJI * CI;
                                }
                            }
                        }
                    }
                }
            }
        }
#endif
        return std::make_pair(my_new_max_one_HJI, my_new_max_two_HJI);
    }
    return std::make_pair(0.0, 0.0);
}

void ProjectorCI::apply_tau_H_det_schwarz(
    double tau, double spawning_threshold, const Determinant& detI, double CI,
    std::vector<std::pair<Determinant, double>>& new_space_C_vec, double E0) {
    std::vector<int> aocc = detI.get_alfa_occ(nact_);
    std::vector<int> bocc = detI.get_beta_occ(nact_);
    std::vector<int> avir = detI.get_alfa_vir(nact_);
    std::vector<int> bvir = detI.get_beta_vir(nact_);

    int noalpha = aocc.size();
    int nobeta = bocc.size();
    int nvalpha = avir.size();
    int nvbeta = bvir.size();

    // Diagonal contributions
    double det_energy = as_ints_->energy(detI) + as_ints_->scalar_energy();
    new_space_C_vec.push_back(std::make_pair(detI, tau * (det_energy - E0) * CI));

    // Generate aa excitations
    for (int i = 0; i < noalpha; ++i) {
        int ii = aocc[i];
        for (int a = 0; a < nvalpha; ++a) {
            int aa = avir[a];
            if ((mo_symmetry_[ii] ^ mo_symmetry_[aa]) == 0) {
                Determinant detJ(detI);
                detJ.set_alfa_bit(ii, false);
                detJ.set_alfa_bit(aa, true);
                double HJI = as_ints_->slater_rules(detJ, detI);
                if (std::fabs(HJI * CI) >= spawning_threshold) {
                    new_space_C_vec.push_back(std::make_pair(detJ, tau * HJI * CI));
                }
            }
        }
    }

    for (int i = 0; i < nobeta; ++i) {
        int ii = bocc[i];
        for (int a = 0; a < nvbeta; ++a) {
            int aa = bvir[a];
            if ((mo_symmetry_[ii] ^ mo_symmetry_[aa]) == 0) {
                Determinant detJ(detI);
                detJ.set_beta_bit(ii, false);
                detJ.set_beta_bit(aa, true);
                double HJI = as_ints_->slater_rules(detJ, detI);
                if (std::fabs(HJI * CI) >= spawning_threshold) {
                    new_space_C_vec.push_back(std::make_pair(detJ, tau * HJI * CI));
                }
            }
        }
    }

    // Generate aa excitations
    double pqpq_max_ab = 0.0;
    for (int a = 0; a < nvalpha; ++a) {
        int aa = avir[a];
        for (int b = a + 1; b < nvalpha; ++b) {
            int bb = avir[b];
            pqpq_max_ab = std::max(pqpq_aa_[aa * nact_ + bb], pqpq_max_ab);
        }
    }
    for (int i = 0; i < noalpha; ++i) {
        int ii = aocc[i];
        for (int j = i + 1; j < noalpha; ++j) {
            int jj = aocc[j];
            ++schwarz_total_;
            if (std::fabs(pqpq_aa_[ii * nact_ + jj] * pqpq_max_ab * CI) < spawning_threshold) {
                ++schwarz_succ_;
                continue;
            }
            for (int a = 0; a < nvalpha; ++a) {
                int aa = avir[a];
                for (int b = a + 1; b < nvalpha; ++b) {
                    int bb = avir[b];
                    if ((mo_symmetry_[ii] ^ mo_symmetry_[jj] ^ mo_symmetry_[aa] ^
                         mo_symmetry_[bb]) == 0) {
                        double HJI = as_ints_->tei_aa(ii, jj, aa, bb);

                        if (std::fabs(HJI * CI) >= spawning_threshold) {
                            Determinant detJ(detI);
                            HJI *= detJ.double_excitation_aa(ii, jj, aa, bb);
                            new_space_C_vec.push_back(std::make_pair(detJ, tau * HJI * CI));
                        }
                    }
                }
            }
        }
    }

    pqpq_max_ab = 0.0;
    for (int a = 0; a < nvalpha; ++a) {
        int aa = avir[a];
        for (int b = 0; b < nvbeta; ++b) {
            int bb = bvir[b];
            pqpq_max_ab = std::max(pqpq_ab_[aa * nact_ + bb], pqpq_max_ab);
        }
    }
    for (int i = 0; i < noalpha; ++i) {
        int ii = aocc[i];
        for (int j = 0; j < nobeta; ++j) {
            int jj = bocc[j];
            ++schwarz_total_;
            if (std::fabs(pqpq_ab_[ii * nact_ + jj] * pqpq_max_ab * CI) < spawning_threshold) {
                ++schwarz_succ_;
                continue;
            }
            for (int a = 0; a < nvalpha; ++a) {
                int aa = avir[a];
                for (int b = 0; b < nvbeta; ++b) {
                    int bb = bvir[b];
                    if ((mo_symmetry_[ii] ^ mo_symmetry_[jj] ^ mo_symmetry_[aa] ^
                         mo_symmetry_[bb]) == 0) {
                        double HJI = as_ints_->tei_ab(ii, jj, aa, bb);

                        if (std::fabs(HJI * CI) >= spawning_threshold) {
                            Determinant detJ(detI);
                            HJI *= detJ.double_excitation_ab(ii, jj, aa, bb);
                            new_space_C_vec.push_back(std::make_pair(detJ, tau * HJI * CI));
                        }
                    }
                }
            }
        }
    }
    pqpq_max_ab = 0.0;
    for (int a = 0; a < nvalpha; ++a) {
        int aa = bvir[a];
        for (int b = a + 1; b < nvbeta; ++b) {
            int bb = bvir[b];
            pqpq_max_ab = std::max(pqpq_bb_[aa * nact_ + bb], pqpq_max_ab);
        }
    }
    for (int i = 0; i < nobeta; ++i) {
        int ii = bocc[i];
        for (int j = i + 1; j < nobeta; ++j) {
            int jj = bocc[j];
            ++schwarz_total_;
            if (std::fabs(pqpq_bb_[ii * nact_ + jj] * pqpq_max_ab * CI) < spawning_threshold) {
                ++schwarz_succ_;
                continue;
            }
            for (int a = 0; a < nvbeta; ++a) {
                int aa = bvir[a];
                for (int b = a + 1; b < nvbeta; ++b) {
                    int bb = bvir[b];
                    if ((mo_symmetry_[ii] ^
                         (mo_symmetry_[jj] ^ (mo_symmetry_[aa] ^ mo_symmetry_[bb]))) == 0) {
                        double HJI = as_ints_->tei_bb(ii, jj, aa, bb);
                        if (std::fabs(HJI * CI) >= spawning_threshold) {
                            Determinant detJ(detI);
                            HJI *= detJ.double_excitation_bb(ii, jj, aa, bb);
                            new_space_C_vec.push_back(std::make_pair(detJ, tau * HJI * CI));
                        }
                    }
                }
            }
        }
    }
}

void ProjectorCI::apply_tau_H_det_dynamic(
    double tau, double spawning_threshold, const Determinant& detI, double CI,
    std::vector<std::pair<Determinant, double>>& new_space_C_vec, double E0,
    std::pair<double, double>& max_coupling) {
    bool do_singles =
        (max_coupling.first == 0.0) or (std::fabs(max_coupling.first * CI) >= spawning_threshold);
    bool do_doubles =
        (max_coupling.second == 0.0) or (std::fabs(max_coupling.second * CI) >= spawning_threshold);

    // Diagonal contributions
    double det_energy = as_ints_->energy(detI) + as_ints_->scalar_energy();
    new_space_C_vec.push_back(std::make_pair(detI, tau * (det_energy - E0) * CI));

    if (do_singles or do_doubles) {
        std::vector<int> aocc = detI.get_alfa_occ(nact_);
        std::vector<int> bocc = detI.get_beta_occ(nact_);
        std::vector<int> avir = detI.get_alfa_vir(nact_);
        std::vector<int> bvir = detI.get_beta_vir(nact_);

        int noalpha = aocc.size();
        int nobeta = bocc.size();
        int nvalpha = avir.size();
        int nvbeta = bvir.size();

        if (do_singles) {
            // Generate alpha excitations
            for (int i = 0; i < noalpha; ++i) {
                int ii = aocc[i];
                for (int a = 0; a < nvalpha; ++a) {
                    int aa = avir[a];
                    if ((mo_symmetry_[ii] ^ mo_symmetry_[aa]) == 0) {
                        Determinant detJ(detI);
                        detJ.set_alfa_bit(ii, false);
                        detJ.set_alfa_bit(aa, true);
                        double HJI = as_ints_->slater_rules(detJ, detI);
                        max_coupling.first = std::max(max_coupling.first, std::fabs(HJI));
                        if (std::fabs(HJI * CI) >= spawning_threshold) {
                            new_space_C_vec.push_back(std::make_pair(detJ, tau * HJI * CI));
                        }
                    }
                }
            }
            // Generate beta excitations
            for (int i = 0; i < nobeta; ++i) {
                int ii = bocc[i];
                for (int a = 0; a < nvbeta; ++a) {
                    int aa = bvir[a];
                    if ((mo_symmetry_[ii] ^ mo_symmetry_[aa]) == 0) {
                        Determinant detJ(detI);
                        detJ.set_beta_bit(ii, false);
                        detJ.set_beta_bit(aa, true);
                        double HJI = as_ints_->slater_rules(detJ, detI);
                        max_coupling.first = std::max(max_coupling.first, std::fabs(HJI));
                        if (std::fabs(HJI * CI) >= spawning_threshold) {
                            new_space_C_vec.push_back(std::make_pair(detJ, tau * HJI * CI));
                        }
                    }
                }
            }
        }

        if (do_doubles) {
            // Generate alpha-alpha excitations
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
                                double HJI = as_ints_->tei_aa(ii, jj, aa, bb);
                                max_coupling.second = std::max(max_coupling.second, std::fabs(HJI));
                                if (std::fabs(HJI * CI) >= spawning_threshold) {
                                    Determinant detJ(detI);
                                    HJI *= detJ.double_excitation_aa(ii, jj, aa, bb);
                                    new_space_C_vec.push_back(std::make_pair(detJ, tau * HJI * CI));
                                }
                            }
                        }
                    }
                }
            }
            // Generate alpha-beta excitations
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
                                double HJI = as_ints_->tei_ab(ii, jj, aa, bb);
                                max_coupling.second = std::max(max_coupling.second, std::fabs(HJI));
                                if (std::fabs(HJI * CI) >= spawning_threshold) {
                                    Determinant detJ(detI);
                                    HJI *= detJ.double_excitation_ab(ii, jj, aa, bb);
                                    new_space_C_vec.push_back(std::make_pair(detJ, tau * HJI * CI));
                                }
                            }
                        }
                    }
                }
            }
            // Generate beta-beta excitations
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
                                double HJI = as_ints_->tei_bb(ii, jj, aa, bb);
                                max_coupling.second = std::max(max_coupling.second, std::fabs(HJI));
                                if (std::fabs(HJI * CI) >= spawning_threshold) {
                                    Determinant detJ(detI);
                                    HJI *= detJ.double_excitation_bb(ii, jj, aa, bb);
                                    new_space_C_vec.push_back(std::make_pair(detJ, tau * HJI * CI));
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

std::map<std::string, double> ProjectorCI::estimate_energy(det_vec& dets, std::vector<double>& C) {
    std::map<std::string, double> results;

    timer_on("PCI:<E>p");
    results["PROJECTIVE ENERGY"] = estimate_proj_energy(dets, C);
    timer_off("PCI:<E>p");

    if (variational_estimate_) {
        if (fast_variational_estimate_) {
            timer_on("PCI:<E>vs");
            results["VARIATIONAL ENERGY"] =
                estimate_var_energy_sparse(dets, C, energy_estimate_threshold_);
            timer_off("PCI:<E>vs");
        } else {
            timer_on("PCI:<E>v");
            results["VARIATIONAL ENERGY"] =
                estimate_var_energy(dets, C, energy_estimate_threshold_);
            timer_off("PCI:<E>v");
        }
    }
    return results;
}

static bool abs_compare(double a, double b) { return (std::abs(a) < std::abs(b)); }

double ProjectorCI::estimate_proj_energy(det_vec& dets, std::vector<double>& C) {
    // Find the determinant with the largest value of C
    auto result = std::max_element(C.begin(), C.end(), abs_compare);
    size_t J = std::distance(C.begin(), result);
    double CJ = C[J];

    // Compute the projective energy
    double projective_energy_estimator = 0.0;
    for (int I = 0, max_I = dets.size(); I < max_I; ++I) {
        double HIJ = as_ints_->slater_rules(dets[I], dets[J]);
        projective_energy_estimator += HIJ * C[I] / CJ;
    }
    return projective_energy_estimator + nuclear_repulsion_energy_ + as_ints_->scalar_energy();
}

double ProjectorCI::estimate_var_energy(det_vec& dets, std::vector<double>& C, double tollerance) {
    // Compute a variational estimator of the energy
    size_t size = dets.size();
    double variational_energy_estimator = 0.0;
#pragma omp parallel for reduction(+ : variational_energy_estimator)
    for (size_t I = 0; I < size; ++I) {
        const Determinant& detI = dets[I];
        variational_energy_estimator += C[I] * C[I] * as_ints_->energy(detI);
        for (size_t J = I + 1; J < size; ++J) {
            if (std::fabs(C[I] * C[J]) > tollerance) {
                double HIJ = as_ints_->slater_rules(dets[I], dets[J]);
                variational_energy_estimator += 2.0 * C[I] * HIJ * C[J];
            }
        }
    }
    return variational_energy_estimator + nuclear_repulsion_energy_ + as_ints_->scalar_energy();
}

double ProjectorCI::estimate_var_energy_sparse(det_vec& dets, std::vector<double>& C,
                                               double tollerance) {
    // A map that contains the pair (determinant,coefficient)
    det_hash<> dets_C_hash;

    // double tau = time_step_;
    double variational_energy_estimator = 0.0;
    std::vector<double> energy(num_threads_, 0.0);

    size_t max_I = dets.size();
    for (size_t I = 0; I < max_I; ++I) {
        dets_C_hash[dets[I]] = C[I];
    }

    std::pair<double, double> zero(0.0, 0.0);
#pragma omp parallel for
    for (size_t I = 0; I < max_I; ++I) {
        int thread_id = omp_get_thread_num();
        // Update the list of couplings
        std::pair<double, double> max_coupling;
#pragma omp critical
        { max_coupling = dets_max_couplings_[dets[I]]; }
        if (max_coupling == zero) {
            max_coupling = {1.0, 1.0};
        }
        //        energy[thread_id] +=
        //        form_H_C_sym(1.0,tollerance,dets[I],C[I],dets_C_hash,max_coupling);
        energy[thread_id] += form_H_C(1.0, tollerance, dets[I], C[I], dets_C_hash, max_coupling);
    }

    for (size_t I = 0; I < max_I; ++I) {
        variational_energy_estimator += C[I] * C[I] * as_ints_->energy(dets[I]);
    }
    for (int t = 0; t < num_threads_; ++t) {
        variational_energy_estimator += energy[t];
    }

    return variational_energy_estimator + nuclear_repulsion_energy_ + as_ints_->scalar_energy();
}

double ProjectorCI::estimate_1st_order_perturbation(det_vec& dets, std::vector<double>& C,
                                                    double spawning_threshold) {
    // Compute a variational estimator of the energy
    size_t size = dets.size();
    double perturbation_energy_estimator = 0.0;
#pragma omp parallel for reduction(+ : perturbation_energy_estimator)
    for (size_t I = 0; I < size; ++I) {
        for (size_t J = 0; J < size; ++J) {
            double HIJ = as_ints_->slater_rules(dets[I], dets[J]);
            if (std::fabs(C[I] * HIJ) < spawning_threshold && J != I) {
                perturbation_energy_estimator += C[I] * HIJ * C[J];
            }
        }
    }
    return perturbation_energy_estimator;
}

double ProjectorCI::estimate_2nd_order_perturbation_sub(det_vec& dets, std::vector<double>& C,
                                                        double spawning_threshold) {
    // Compute a variational estimator of the energy
    size_t size = dets.size();
    double perturbation_energy_estimator = 0.0;
#pragma omp parallel for reduction(+ : perturbation_energy_estimator)
    for (size_t I = 0; I < size; ++I) {
        for (size_t J = 0; J < size; ++J) {
            double HIJ = as_ints_->slater_rules(dets[I], dets[J]);
            if (std::fabs(C[I] * HIJ) < spawning_threshold && J != I) {
                perturbation_energy_estimator += C[I] * HIJ * C[J];
            }
        }
    }
    return perturbation_energy_estimator;
}

std::tuple<double, double> ProjectorCI::estimate_perturbation(det_vec& dets, std::vector<double>& C,
                                                              double spawning_threshold) {
    //    double first_order_perturb = estimate_1st_order_perturbation(dets, C,
    //    spawning_threshold);
    //    return std::make_tuple(first_order_perturb, 0.0, 0.0);
    // Compute a variational estimator of the energy
    size_t size = dets.size();
    double variational_energy_estimator = approx_energy_ - nuclear_repulsion_energy_;
    //#pragma omp parallel for reduction(+:variational_energy_estimator,
    // perturbation_1st_energy_estimator)
    //    for (size_t I = 0; I < size; ++I){
    //        for (size_t J = 0; J < size; ++J){
    //            double HIJ = dets[I].slater_rules(dets[J]);
    //            if (std::fabs(C[I] * HIJ) < spawning_threshold && J != I){
    //                perturbation_1st_energy_estimator += C[I] * HIJ * C[J];
    //            } else {
    //                variational_energy_estimator += C[I] * HIJ * C[J];
    //            }
    //        }
    //    }
    double perturbation_2nd_energy_estimator_sub = 0.0;
#pragma omp parallel for reduction(+ : perturbation_2nd_energy_estimator_sub)
    for (size_t I = 0; I < size; ++I) {
        double current_V = 0.0;
        for (size_t J = 0; J < size; ++J) {
            double HIJ = as_ints_->slater_rules(dets[J], dets[I]);
            if (symm_approx_H_) {
                if (std::fabs(C[J] * HIJ) < spawning_threshold &&
                    std::fabs(C[I] * HIJ) < spawning_threshold && J != I) {
                    current_V += HIJ * C[J];
                }
            } else {
                if (std::fabs(C[J] * HIJ) < spawning_threshold && J != I) {
                    current_V += HIJ * C[J];
                }
            }
        }
        current_V *= C[I];
        double delta =
            variational_energy_estimator - as_ints_->energy(dets[I]) - as_ints_->scalar_energy();
        perturbation_2nd_energy_estimator_sub += current_V * current_V / delta;
        //            0.5 * (delta - sqrt(delta * delta + 4 * current_V *
        //            current_V));
    }
    return std::make_tuple(perturbation_2nd_energy_estimator_sub, 0.0);
}

double ProjectorCI::estimate_path_filtering_error(det_vec& dets, std::vector<double>& C,
                                                  double spawning_threshold) {
    size_t size = dets.size();
    double pfError = 0.0;
#pragma omp parallel for reduction(max : pfError)
    for (size_t I = 0; I < size; ++I) {
        double current_pf = 0.0;
        for (size_t J = 0; J < size; ++J) {
            double HIJ = as_ints_->slater_rules(dets[J], dets[I]);
            if (std::fabs(C[J] * HIJ) < spawning_threshold && J != I) {
                current_pf += std::fabs(HIJ * C[J]);
            }
        }
        if (current_pf > pfError)
            pfError = current_pf;
    }
    return pfError;
}

void ProjectorCI::print_wfn(det_vec& space, std::vector<double>& C, size_t max_output) {
    outfile->Printf("\n\n  Most important contributions to the wave function:\n");

    std::vector<std::pair<double, size_t>> det_weight;
    for (size_t I = 0; I < space.size(); ++I) {
        det_weight.push_back(std::make_pair(std::fabs(C[I]), I));
    }
    std::sort(det_weight.begin(), det_weight.end());
    std::reverse(det_weight.begin(), det_weight.end());
    size_t max_dets = std::min(int(max_output), int(C.size()));
    for (size_t I = 0; I < max_dets; ++I) {
        outfile->Printf("\n  %3zu  %13.6g %13.6g  %10zu %s  %18.12f", I, C[det_weight[I].second],
                        det_weight[I].first * det_weight[I].first, det_weight[I].second,
                        space[det_weight[I].second].str().c_str(),
                        as_ints_->energy(space[det_weight[I].second]) + as_ints_->scalar_energy());
    }

    // Compute the expectation value of the spin
    size_t max_sample = 1000;
    size_t max_I = 0;
    double sum_weight = 0.0;
    double wfn_threshold = 0.95;
    for (size_t I = 0; I < space.size(); ++I) {
        if ((sum_weight < wfn_threshold) and (I < max_sample)) {
            sum_weight += std::pow(det_weight[I].first, 2.0);
            max_I++;
        } else if (std::fabs(det_weight[I].first - det_weight[I - 1].first) < 1.0e-6) {
            // Special case, if there are several equivalent determinants
            sum_weight += std::pow(det_weight[I].first, 2.0);
            max_I++;
        } else {
            break;
        }
    }

    double norm = 0.0;
    double S2 = 0.0;
    for (size_t sI = 0; sI < max_I; ++sI) {
        size_t I = det_weight[sI].second;
        for (size_t sJ = 0; sJ < max_I; ++sJ) {
            size_t J = det_weight[sJ].second;
            if (std::fabs(C[I] * C[J]) > 1.0e-12) {
                const double S2IJ = spin2(space[I], space[J]);
                S2 += C[I] * C[J] * S2IJ;
            }
        }
        norm += std::pow(C[I], 2.0);
    }
    S2 /= norm;
    double S = std::fabs(0.5 * (std::sqrt(1.0 + 4.0 * S2) - 1.0));

    std::vector<std::string> s2_labels({"singlet", "doublet", "triplet", "quartet", "quintet",
                                        "sextet", "septet", "octet", "nonet", "decaet"});
    std::string state_label = s2_labels[std::round(S * 2.0)];
    outfile->Printf("\n\n  Spin State: S^2 = %5.3f, S = %5.3f, %s (from %zu "
                    "determinants,%.2f%%)",
                    S2, S, state_label.c_str(), max_I, 100.0 * sum_weight);
}

void ProjectorCI::save_wfn(det_vec& space, std::vector<double>& C,
                           std::vector<det_hash<>>& solutions) {
    outfile->Printf("\n\n  Saving the wave function:\n");

    det_hash<> solution;
    for (size_t I = 0; I < space.size(); ++I) {
        solution[space[I]] = C[I];
    }
    solutions.push_back(std::move(solution));
}

void ProjectorCI::orthogonalize(det_vec& space, std::vector<double>& C,
                                std::vector<det_hash<>>& solutions) {
    det_hash<> det_C;
    for (size_t I = 0; I < space.size(); ++I) {
        det_C[space[I]] = C[I];
    }
    for (size_t n = 0; n < solutions.size(); ++n) {
        double dot_prod = dot(det_C, solutions[n]);
        add(det_C, -dot_prod, solutions[n]);
    }
    normalize(det_C);
    copy_hash_to_vec(det_C, space, C);
}

void combine_hashes(std::vector<det_hash<>>& thread_det_C_map, det_hash<>& dets_C_hash) {
    // Combine the content of varius wave functions stored as maps
    for (size_t t = 0; t < thread_det_C_map.size(); ++t) {
        for (det_hash_it it = thread_det_C_map[t].begin(), endit = thread_det_C_map[t].end();
             it != endit; ++it) {
            dets_C_hash[it->first] += it->second;
        }
    }
}

void combine_hashes(det_hash<>& dets_C_hash_A, det_hash<>& dets_C_hash_B) {
    // Combine the content of varius wave functions stored as maps
    for (det_hash_it it = dets_C_hash_A.begin(), endit = dets_C_hash_A.end(); it != endit; ++it) {
        dets_C_hash_B[it->first] += it->second;
    }
}

void combine_hashes_into_hash(std::vector<det_hash<>>& thread_det_C_hash, det_hash<>& dets_C_hash) {
    // Combine the content of varius wave functions stored as maps
    for (size_t t = 0; t < thread_det_C_hash.size(); ++t) {
        for (auto& kv : thread_det_C_hash[t]) {
            dets_C_hash[kv.first] += kv.second;
        }
    }
}

void copy_hash_to_vec(det_hash<>& dets_C_hash, det_vec& dets, std::vector<double>& C) {
    size_t size = dets_C_hash.size();
    dets.resize(size, Determinant());
    C.resize(size);

    size_t I = 0;
    for (det_hash_it it = dets_C_hash.begin(), endit = dets_C_hash.end(); it != endit; ++it) {
        dets[I] = it->first;
        C[I] = it->second;
        I++;
    }
}

void copy_hash_to_vec_order_ref(det_hash<>& dets_C_hash, det_vec& dets, std::vector<double>& C) {
    size_t hash_size = dets_C_hash.size();
    size_t size = dets.size();
    dets.resize(hash_size, Determinant());
    C.resize(hash_size);

    for (size_t I = 0; I < size; ++I) {
        C[I] = dets_C_hash[dets[I]];
        dets_C_hash.erase(dets[I]);
    }
    size_t I = size;
    for (det_hash_it it = dets_C_hash.begin(), endit = dets_C_hash.end(); it != endit; ++it) {
        dets[I] = it->first;
        C[I] = it->second;
        I++;
    }
}

void copy_vec_to_hash(det_vec& dets, const std::vector<double>& C, det_hash<>& dets_C_hash) {
    size_t size = C.size();
    for (size_t I = 0; I < size; ++I) {
        dets_C_hash[dets[I]] = C[I];
    }
}

double normalize(std::vector<double>& C);

double normalize(det_hash<>& dets_C) {
    double norm = 0.0;
    for (auto& det_C : dets_C) {
        norm += det_C.second * det_C.second;
    }
    norm = std::sqrt(norm);
    for (auto& det_C : dets_C) {
        det_C.second /= norm;
    }
    return norm;
}

double norm(std::vector<double>& C) {
    size_t size = C.size();
    double norm = 0.0;
    for (size_t I = 0; I < size; ++I) {
        norm += C[I] * C[I];
    }
    norm = std::sqrt(norm);
    return norm;
}

double norm(det_hash<>& dets_C) {
    double norm = 0.0;
    for (auto& det_C : dets_C) {
        norm += det_C.second * det_C.second;
    }
    norm = std::sqrt(norm);
    return norm;
}

double dot(det_hash<>& A, det_hash<>& B) {
    double res = 0.0;
    for (auto& det_C : A) {
        res += det_C.second * B[det_C.first];
    }
    return res;
}

double dot(std::vector<double>& C1, std::vector<double>& C2) {
    size_t size1 = C1.size(), size2 = C2.size();
    size1 = size1 < size2 ? size1 : size2;
    double result = 0.0;
#pragma omp parallel for reduction(+ : result)
    for (size_t i = 0; i < size1; i++) {
        result += C1[i] * C2[i];
    }
    return result;
}

size_t ortho_norm(std::vector<std::vector<double>>& H_n_C, std::vector<double>& norms, Matrix& A,
                  double colinear_threshold) {
    size_t order = H_n_C.size() - 1;
    Vector N(order);
    Matrix QC(order, order), QHC(order, order);

    N.set(0, 1.0);
    QC.set(0, 0, 1.0);
    QHC.set(0, 0, A.get(0, 0));

    for (size_t n = 1; n < order; n++) {
        for (size_t m = 0; m < n; m++) {
            double qm_dot_Cn = dot(H_n_C[m], H_n_C[n]);
            QC.set(m, n, qm_dot_Cn);
            add(H_n_C[n], -qm_dot_Cn, H_n_C[m]);
        }
        double norm = normalize(H_n_C[n]);
        //        outfile->Printf("\nCorrect norm order %d: %e", n, norm);
        if (norm < colinear_threshold && n > 1)
            return n;
        N.set(n, 1.0 / norm);
        for (size_t m = 0; m < n; m++) {
            double correction = 0.0;
            for (size_t i = 0; i < n; i++) {
                correction += QC.get(i, n) * A.get(i, m);
            }
            double QmHCn = norms[n] * dot(H_n_C[m], H_n_C[n + 1]);
            QHC.set(m, n, QmHCn);
            double Anm = N.get(n) * (QmHCn - correction);
            A.set(m, n, Anm);
            A.set(n, m, Anm);
        }
        double QCQHC_correction = 0.0, QCAQC_correction = 0.0;
        for (size_t i = 0; i < n; i++) {
            QCQHC_correction += QC.get(i, n) * QHC.get(i, n);
            for (size_t j = 0; j < n; j++) {
                QCAQC_correction += QC.get(i, n) * A.get(i, j) * QC.get(j, n);
            }
        }
        double Ann =
            N.get(n) * N.get(n) * (A.get(n, n) - 2.0 * QCQHC_correction + QCAQC_correction);
        A.set(n, n, Ann);
    }

    //    outfile -> Printf("\n  QC:");
    //    QC.print();
    //    outfile -> Printf("\n  QHC:");
    //    QHC.print();
    //    outfile -> Printf("\n  N:");
    //    N.print();

    return order;
}

void add(det_hash<>& A, double beta, det_hash<>& B) {
    // A += beta B
    for (auto& det_C : B) {
        A[det_C.first] += beta * det_C.second;
    }
}

void add(std::vector<double>& a, double k, std::vector<double>& b) {
    size_t sizeA = a.size(), sizeB = b.size();
    if (sizeA < sizeB)
        a.resize(sizeB);
#pragma omp parallel for
    for (size_t i = 0; i < sizeB; i++) {
        a[i] += k * b[i];
    }
}

void scale(std::vector<double>& A, double alpha) {
    size_t size = A.size();
#pragma omp parallel for
    for (size_t I = 0; I < size; ++I) {
        A[I] *= alpha;
    }
}

void scale(det_hash<>& A, double alpha) {
    for (auto& det_C : A) {
        A[det_C.first] *= alpha;
    }
}

/*
double ProjectorCI::form_H_C_sym(double tau,double
spawning_threshold,Determinant& detI, double CI, bit_hash&
det_C,std::pair<double,double>& max_coupling)
{
    double result = 0.0;

    if ((max_coupling.second != 0.0) and (std::fabs(max_coupling.second  * CI) <
spawning_threshold) and (std::fabs(max_coupling.first * CI) <
spawning_threshold)){
        return result;
    }

    std::vector<std::vector<int>> naoccs(nirrep_), navirs(nirrep_),
nboccs(nirrep_), nbvirs(nirrep_);

    for (int i = 0; i<nact_; i++) {
        if (detI.get_alfa_bit(i)) {
            naoccs[mo_symmetry_[i]].push_back(i);
        } else {
            navirs[mo_symmetry_[i]].push_back(i);
        }
        if (detI.get_beta_bit(i)) {
            nboccs[mo_symmetry_[i]].push_back(i);
        } else {
            nbvirs[mo_symmetry_[i]].push_back(i);
        }
    }

    size_t spawned = 0;

    // No diagonal contributions

    if ((std::fabs(max_coupling.first * CI) >= spawning_threshold)){
        for (int i = 0; i<nirrep_; i++) {
            for (int ix=0; ix<naoccs[i].size(); ix++) {
                int ii = naoccs[i][ix];
                for (int ax=0; ax<navirs[i].size(); ax++) {
                    int aa = navirs[i][ax];
                    Determinant detJ(detI);
                    detJ.set_alfa_bit(ii,false);
                    detJ.set_alfa_bit(aa,true);
                    det_hash_it it = det_C.find(detJ);
                    if (it != det_C.end()){
                        double HJI = detJ.slater_rules(detI);
                        if (std::fabs(HJI * CI) >= spawning_threshold){
                            result += tau * HJI * CI * it->second;
                            spawned++;
                        }
                    }
                    ndet_visited_++;
                }
            }
        }
        for (int i = 0; i<nirrep_; i++) {
            for (int ix=0; ix<nboccs[i].size(); ix++) {
                int ii = nboccs[i][ix];
                for (int ax=0; ax<nbvirs[i].size(); ax++) {
                    int aa = nbvirs[i][ax];
                    Determinant detJ(detI);
                    detJ.set_beta_bit(ii,false);
                    detJ.set_beta_bit(aa,true);
                    det_hash_it it = det_C.find(detJ);
                    if (it != det_C.end()){
                        double HJI = detJ.slater_rules(detI);
                        if (std::fabs(HJI * CI) >= spawning_threshold){
                            result += tau * HJI * CI * it->second;
                            spawned++;
                        }
                    }
                    ndet_visited_++;
                }
            }
        }
    }

    if ((max_coupling.second == 0.0) or (std::fabs(max_coupling.second  * CI) >=
spawning_threshold)){
        for (int i = 0; i<nirrep_; i++) {
            for (int j = 0; j<nirrep_; j++) {
                for (int a = 0; a<nirrep_; a++){
                    int b = i^j^a;
                    for (int ix=0; ix<naoccs[i].size(); ix++) {
                        int ii = naoccs[i][ix];
                        for (int jx=0; jx<nboccs[j].size(); jx++) {
                            int jj = nboccs[j][jx];
                            for (int ax=0; ax<navirs[a].size(); ax++) {
                                int aa = navirs[a][ax];
                                for (int bx=0; bx<nbvirs[b].size(); bx++) {
                                    int bb = nbvirs[b][bx];
                                    double HJI = as_ints_->tei_ab(ii,jj,aa,bb);

                                    if (std::fabs(HJI * CI) >=
spawning_threshold){
                                        Determinant detJ(detI);
                                        double sign =
detJ.double_excitation_ab(ii,jj,aa,bb);
//                                        detJ.set_alfa_bit(ii,false);
//                                        detJ.set_beta_bit(jj,false);
//                                        detJ.set_alfa_bit(aa,true);
//                                        detJ.set_beta_bit(bb,true);

                                        det_hash_it it = det_C.find(detJ);
                                        if (it != det_C.end()){
//                                            // grap the alpha bits of both
determinants
//                                            const Determinant::bit_t& Ia =
detI.alfa_bits();
//                                            const Determinant::bit_t& Ib =
detI.beta_bits();
//                                            const Determinant::bit_t& Ja =
detJ.alfa_bits();
//                                            const Determinant::bit_t& Jb =
detJ.beta_bits();

                                            // compute the sign of the matrix
element
//                                            HJI *=
Determinant::SlaterSign(Ia,ii) * Determinant::SlaterSign(Ib,jj) *
Determinant::SlaterSign(Ja,aa) * Determinant::SlaterSign(Jb,bb);

                                            result += sign * tau * HJI * CI *
it->second;

                                        }
                                    }
                                    ndet_visited_++;

                                }
                            }
                        }
                    }
                }
            }
        }
        for (int i = 0; i<nirrep_; i++) {
            for (int a = 0; a<nirrep_; a++) {
                for (int ix=0; ix<naoccs[i].size(); ix++) {
                    int ii = naoccs[i][ix];
                    for (int jx=ix+1; jx<naoccs[i].size(); jx++) {
                        int jj = naoccs[i][jx];
                        for (int ax=0; ax<navirs[a].size(); ax++) {
                            int aa = navirs[a][ax];
                            for (int bx=ax+1; bx<navirs[a].size(); bx++) {
                                int bb = navirs[a][bx];
                                double HJI = as_ints_->tei_aa(ii,jj,aa,bb);

                                if (std::fabs(HJI * CI) >= spawning_threshold){
                                    Determinant detJ(detI);
                                    double sign =
detJ.double_excitation_aa(ii,jj,aa,bb);

//                                    detJ.set_alfa_bit(ii,false);
//                                    detJ.set_alfa_bit(jj,false);
//                                    detJ.set_alfa_bit(aa,true);
//                                    detJ.set_alfa_bit(bb,true);

                                    det_hash_it it = det_C.find(detJ);
                                    if (it != det_C.end()){
//                                        // grap the alpha bits of both
determinants
//                                        const Determinant::bit_t& Ia =
detI.alfa_bits();
//                                        const Determinant::bit_t& Ja =
detJ.alfa_bits();

//                                        // compute the sign of the matrix
element
//                                        HJI *= Determinant::SlaterSign(Ia,ii)
* Determinant::SlaterSign(Ia,jj) * Determinant::SlaterSign(Ja,aa) *
Determinant::SlaterSign(Ja,bb);

                                        result += sign * tau * HJI * CI *
it->second;

                                    }
                                }
                                ndet_visited_++;
                            }
                        }
                    }
                }
            }
        }
        for (int i = 0; i<nirrep_; i++) {
            for (int j = i+1; j<nirrep_; j++) {
                for (int a = 0; a<nirrep_; a++){
                    int b = i^j^a;
                    if (b > a) {
                        for (int ix=0; ix<naoccs[i].size(); ix++) {
                            int ii = naoccs[i][ix];
                            for (int jx=0; jx<naoccs[j].size(); jx++) {
                                int jj = naoccs[j][jx];
                                for (int ax=0; ax<navirs[a].size(); ax++) {
                                    int aa = navirs[a][ax];
                                    for (int bx=0; bx<navirs[b].size(); bx++) {
                                        int bb = navirs[b][bx];
                                        double HJI =
as_ints_->tei_aa(ii,jj,aa,bb);

                                        if (std::fabs(HJI * CI) >=
spawning_threshold){
                                            Determinant detJ(detI);
                                            detJ.set_alfa_bit(ii,false);
                                            detJ.set_alfa_bit(jj,false);
                                            detJ.set_alfa_bit(aa,true);
                                            detJ.set_alfa_bit(bb,true);

                                            det_hash_it it = det_C.find(detJ);
                                            if (it != det_C.end()){
                                                // grap the alpha bits of both
determinants
                                                const Determinant::bit_t& Ia =
detI.alfa_bits();
                                                const Determinant::bit_t& Ja =
detJ.alfa_bits();

                                                // compute the sign of the
matrix element
                                                HJI *=
Determinant::SlaterSign(Ia,ii) * Determinant::SlaterSign(Ia,jj) *
Determinant::SlaterSign(Ja,aa) * Determinant::SlaterSign(Ja,bb);

                                                result += tau * HJI * CI *
it->second;

                                            }
                                        }
                                        ndet_visited_++;
                                    }
                                }
                            }
                        }
                    }

                }
            }
        }
        for (int i = 0; i<nirrep_; i++) {
            for (int a = 0; a<nirrep_; a++) {
                for (int ix=0; ix<nboccs[i].size(); ix++) {
                    int ii = nboccs[i][ix];
                    for (int jx=ix+1; jx<nboccs[i].size(); jx++) {
                        int jj = nboccs[i][jx];
                        for (int ax=0; ax<nbvirs[a].size(); ax++) {
                            int aa = nbvirs[a][ax];
                            for (int bx=ax+1; bx<nbvirs[a].size(); bx++) {
                                int bb = nbvirs[a][bx];
                                double HJI = as_ints_->tei_bb(ii,jj,aa,bb);

                                if (std::fabs(HJI * CI) >= spawning_threshold){
                                    Determinant detJ(detI);
                                    detJ.set_beta_bit(ii,false);
                                    detJ.set_beta_bit(jj,false);
                                    detJ.set_beta_bit(aa,true);
                                    detJ.set_beta_bit(bb,true);

                                    det_hash_it it = det_C.find(detJ);
                                    if (it != det_C.end()){
                                        // grap the alpha bits of both
determinants
                                        const Determinant::bit_t& Ib =
detI.beta_bits();
                                        const Determinant::bit_t& Jb =
detJ.beta_bits();

                                        // compute the sign of the matrix
element
                                        HJI *= Determinant::SlaterSign(Ib,ii) *
Determinant::SlaterSign(Ib,jj) * Determinant::SlaterSign(Jb,aa) *
Determinant::SlaterSign(Jb,bb);

                                        result += tau * HJI * CI * it->second;

                                    }
                                }
                                ndet_visited_++;
                            }
                        }
                    }
                }
            }
        }
        for (int i = 0; i<nirrep_; i++) {
            for (int j = i+1; j<nirrep_; j++) {
                for (int a = 0; a<nirrep_; a++){
                    int b = i^j^a;
                    if (b > a) {
                        for (int ix=0; ix<nboccs[i].size(); ix++) {
                            int ii = nboccs[i][ix];
                            for (int jx=0; jx<nboccs[j].size(); jx++) {
                                int jj = nboccs[j][jx];
                                for (int ax=0; ax<nbvirs[a].size(); ax++) {
                                    int aa = nbvirs[a][ax];
                                    for (int bx=0; bx<nbvirs[b].size(); bx++) {
                                        int bb = nbvirs[b][bx];
                                        double HJI =
as_ints_->tei_bb(ii,jj,aa,bb);

                                        if (std::fabs(HJI * CI) >=
spawning_threshold){
                                            Determinant detJ(detI);
                                            detJ.set_beta_bit(ii,false);
                                            detJ.set_beta_bit(jj,false);
                                            detJ.set_beta_bit(aa,true);
                                            detJ.set_beta_bit(bb,true);

                                            det_hash_it it = det_C.find(detJ);
                                            if (it != det_C.end()){
                                                // grap the alpha bits of both
determinants
                                                const Determinant::bit_t& Ib =
detI.beta_bits();
                                                const Determinant::bit_t& Jb =
detJ.beta_bits();

                                                // compute the sign of the
matrix element
                                                HJI *=
Determinant::SlaterSign(Ib,ii) * Determinant::SlaterSign(Ib,jj) *
Determinant::SlaterSign(Jb,aa) * Determinant::SlaterSign(Jb,bb);

                                                result += tau * HJI * CI *
it->second;

                                            }
                                        }
                                        ndet_visited_++;
                                    }
                                }
                            }
                        }
                    }

                }
            }
        }
    }

    return result;
}
*/

double ProjectorCI::form_H_C(double tau, double spawning_threshold, Determinant& detI, double CI,
                             det_hash<>& det_C, std::pair<double, double>& max_coupling) {
    double result = 0.0;

    std::vector<int> aocc = detI.get_alfa_occ(nact_);
    std::vector<int> bocc = detI.get_beta_occ(nact_);
    std::vector<int> avir = detI.get_alfa_vir(nact_);
    std::vector<int> bvir = detI.get_beta_vir(nact_);

    int noalpha = aocc.size();
    int nobeta = bocc.size();
    int nvalpha = avir.size();
    int nvbeta = bvir.size();

    // No diagonal contributions

    if ((std::fabs(max_coupling.first * CI) >= spawning_threshold)) {
        // Generate aa excitations
        for (int i = 0; i < noalpha; ++i) {
            int ii = aocc[i];
            for (int a = 0; a < nvalpha; ++a) {
                int aa = avir[a];
                if ((mo_symmetry_[ii] ^ mo_symmetry_[aa]) == 0) {
                    Determinant detJ(detI);
                    detJ.set_alfa_bit(ii, false);
                    detJ.set_alfa_bit(aa, true);
                    det_hash_it it = det_C.find(detJ);
                    if (it != det_C.end()) {
                        double HJI = as_ints_->slater_rules(detJ, detI);
                        if (std::fabs(HJI * CI) >= spawning_threshold) {
                            result += tau * HJI * CI * it->second;
                        }
                    }
                }
            }
        }

        for (int i = 0; i < nobeta; ++i) {
            int ii = bocc[i];
            for (int a = 0; a < nvbeta; ++a) {
                int aa = bvir[a];
                if ((mo_symmetry_[ii] ^ mo_symmetry_[aa]) == 0) {
                    Determinant detJ(detI);
                    detJ.set_beta_bit(ii, false);
                    detJ.set_beta_bit(aa, true);
                    det_hash_it it = det_C.find(detJ);
                    if (it != det_C.end()) {
                        double HJI = as_ints_->slater_rules(detJ, detI);
                        if (std::fabs(HJI * CI) >= spawning_threshold) {
                            result += tau * HJI * CI * it->second;
                        }
                    }
                }
            }
        }
    }

    if ((max_coupling.second == 0.0) or
        (std::fabs(max_coupling.second * CI) >= spawning_threshold)) {
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
                            double HJI = as_ints_->tei_aa(ii, jj, aa, bb);

                            if (std::fabs(HJI * CI) >= spawning_threshold) {
                                Determinant detJ(detI);
                                double sign = detJ.double_excitation_aa(ii, jj, aa, bb);
                                det_hash_it it = det_C.find(detJ);
                                if (it != det_C.end()) {
                                    result += sign * tau * HJI * CI * it->second;
                                }
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
                            double HJI = as_ints_->tei_ab(ii, jj, aa, bb);

                            if (std::fabs(HJI * CI) >= spawning_threshold) {
                                Determinant detJ(detI);
                                double sign = detJ.double_excitation_ab(ii, jj, aa, bb);
                                det_hash_it it = det_C.find(detJ);
                                if (it != det_C.end()) {
                                    result += sign * tau * HJI * CI * it->second;
                                }
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
                            double HJI = as_ints_->tei_bb(ii, jj, aa, bb);
                            if (std::fabs(HJI * CI) >= spawning_threshold) {
                                Determinant detJ(detI);
                                double sign = detJ.double_excitation_bb(ii, jj, aa, bb);
                                det_hash_it it = det_C.find(detJ);
                                if (it != det_C.end()) {
                                    result += sign * tau * HJI * CI * it->second;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    return result;
}

std::vector<std::tuple<double, int, int>> ProjectorCI::sym_labeled_orbitals(std::string type) {
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
