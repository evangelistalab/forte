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
#include <cmath>
#include <limits>
#include <cfloat>
#include <functional>
#include <tuple>
#include <unordered_map>
#include <utility>

#include "psi4/libpsi4util/PsiOutStream.h"
#include "psi4/libpsi4util/process.h"
#include "psi4/libmints/matrix.h"

#include "helpers/timer.h"
#include "sparse_ci/ci_reference.h"
#include "sparse_ci/determinant_functions.hpp"
#include "pci.h"
#include "pci_sigma.h"

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

std::vector<double> to_std_vector(psi::SharedVector c);
void set_psi_Vector(psi::SharedVector c_psi, const std::vector<double>& c_vec);

double normalize(std::vector<double>& C) {
    size_t size = C.size();
    double norm = 0.0;
    for (size_t I = 0; I < size; ++I) {
        norm += C[I] * C[I];
    }
    norm = std::sqrt(norm);
    for (size_t I = 0; I < size; ++I) {
        C[I] /= norm;
    }
    return norm;
}

double factorial(int n) { return (n == 1 || n == 0) ? 1.0 : factorial(n - 1) * n; }

void binomial_coefs(std::vector<double>& coefs, int order, double a, double b) {
    coefs.clear();
    for (int i = 0; i <= order; i++) {
        coefs.push_back(factorial(order) / (factorial(i) * factorial(order - i)) * pow(a, i) *
                        pow(b, order - i));
    }
}

void Polynomial_generator_coefs(std::vector<double>& coefs, std::vector<double>& poly_coefs,
                                double a, double b) {
    coefs.clear();
    int order = poly_coefs.size() - 1;
    for (int i = 0; i <= order; i++) {
        coefs.push_back(0.0);
    }
    for (int i = 0; i <= order; i++) {
        std::vector<double> bino_coefs;
        binomial_coefs(bino_coefs, i, a, b);
        for (int j = 0; j <= i; j++) {
            coefs[j] += poly_coefs[i] * bino_coefs[j];
        }
    }
}

void Chebyshev_polynomial_coefs(std::vector<double>& coefs, int order) {
    coefs.clear();
    std::vector<double> coefs_0, coefs_1;
    if (order == 0) {
        coefs.push_back(1.0);
        return;
    } else
        coefs_0.push_back(1.0);
    if (order == 1) {
        coefs.push_back(0.0);
        coefs.push_back(1.0);
        return;
    } else {
        coefs_1.push_back(0.0);
        coefs_1.push_back(1.0);
    }
    for (int i = 2; i <= order; i++) {
        coefs.clear();
        for (int j = 0; j <= i; j++) {
            coefs.push_back(0.0);
        }
        for (int j = 0; j <= i - 2; j++) {
            coefs[j] -= coefs_0[j];
        }
        for (int j = 0; j <= i - 1; j++) {
            coefs[j + 1] += 2.0 * coefs_1[j];
        }
        coefs_0 = coefs_1;
        coefs_1 = coefs;
    }
}

void Wall_Chebyshev_generator_coefs(std::vector<double>& coefs, int order, double range) {
    coefs.clear();
    std::vector<double> poly_coefs;
    for (int i = 0; i <= order; i++) {
        poly_coefs.push_back(0.0);
    }

    for (int i = 0; i <= order; i++) {
        std::vector<double> chbv_poly_coefs;
        Chebyshev_polynomial_coefs(chbv_poly_coefs, i);
        //        psi::outfile->Printf("\n\n  chebyshev poly in step %d", i);
        //        print_polynomial(chbv_poly_coefs);
        for (int j = 0; j <= i; j++) {
            poly_coefs[j] += (i == 0 ? 1.0 : 2.0) * chbv_poly_coefs[j];
        }
        //        psi::outfile->Printf("\n\n  propagate poly in step %d", i);
        //        print_polynomial(poly_coefs);
    }
    Polynomial_generator_coefs(coefs, poly_coefs, -1.0 / range, 0.0);
}

void print_polynomial(std::vector<double>& coefs) {
    psi::outfile->Printf("\n    f(x) = ");
    for (int i = coefs.size() - 1; i >= 0; i--) {
        switch (i) {
        case 0:
            psi::outfile->Printf("%s%e", coefs[i] >= 0 ? "+" : "", coefs[i]);
            break;
        case 1:
            psi::outfile->Printf("%s%e * x ", coefs[i] >= 0 ? "+" : "", coefs[i]);
            break;
        default:
            psi::outfile->Printf("%s%e * x^%d ", coefs[i] >= 0 ? "+" : "", coefs[i], i);
            break;
        }
    }
}

void add(const det_hashvec& A, std::vector<double>& Ca, double beta, const det_hashvec& B,
         const std::vector<double> Cb) {
    size_t A_size = A.size(), B_size = B.size();
#pragma omp parallel for
    for (size_t i = 0; i < A_size; ++i) {
        size_t B_index = B.find(A[i]);
        if (B_index < B_size)
            Ca[i] += beta * Cb[B_index];
    }
}

double dot(const det_hashvec& A, const std::vector<double> Ca, const det_hashvec& B,
           const std::vector<double> Cb) {
    double res = 0.0;
    size_t A_size = A.size(), B_size = B.size();
#pragma omp parallel for reduction(+ : res)
    for (size_t i = 0; i < A_size; ++i) {
        size_t B_index = B.find(A[i]);
        if (B_index < B_size)
            res += Ca[i] * Cb[B_index];
    }
    return res;
}

void ProjectorCI::sortHashVecByCoefficient(det_hashvec& dets_hashvec, std::vector<double>& C) {
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
    for (size_t I = 0; I < dets_size; ++I) {
        new_C[order_map[I]] = C[I];
    }
    C = std::move(new_C);
}

ProjectorCI::ProjectorCI(StateInfo state, size_t nroot, std::shared_ptr<SCFInfo> scf_info,
                         std::shared_ptr<ForteOptions> options,
                         std::shared_ptr<MOSpaceInfo> mo_space_info,
                         std::shared_ptr<ActiveSpaceIntegrals> as_ints)
    : SelectedCIMethod(state, nroot, scf_info, options, mo_space_info, as_ints) {
    // Copy the wavefunction information
    startup();
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

    approx_E_tau_ = 1.0;
    approx_E_S_ = 0.0;

    sparse_solver_.set_parallel(true);
    sparse_solver_.set_spin_project_full(true);

    num_threads_ = omp_get_max_threads();
}

void ProjectorCI::set_options(std::shared_ptr<ForteOptions> options) {
    // Build the reference determinant and compute its energy
    int ms = wavefunction_multiplicity_ - 1;
    std::vector<Determinant> reference_vec;
    CI_Reference ref(scf_info_, options, mo_space_info_, as_ints_, wavefunction_multiplicity_, ms,
                     wavefunction_symmetry_,state_);
    ref.set_ref_type("HF");
    ref.build_reference(reference_vec);
    reference_determinant_ = reference_vec[0];

    //    psi::outfile->Printf("\n  The reference determinant is:\n");
    //    reference_determinant_.print();

    post_diagonalization_ = options->get_bool("PCI_POST_DIAGONALIZE");
    //    /-> Define appropriate variable: post_diagonalization_ =
    //    options_->get_bool("EX_ALGORITHM");

    spawning_threshold_ = options->get_double("PCI_SPAWNING_THRESHOLD");
    initial_guess_spawning_threshold_ = options->get_double("PCI_GUESS_SPAWNING_THRESHOLD");
    if (initial_guess_spawning_threshold_ < 0.0)
        initial_guess_spawning_threshold_ = 10.0 * spawning_threshold_;
    max_cycle_ = options->get_int("SCI_MAX_CYCLE");
    max_Davidson_iter_ = options->get_int("PCI_MAX_DAVIDSON_ITER");
    davidson_collapse_per_root_ = options->get_int("PCI_DL_COLLAPSE_PER_ROOT");
    davidson_subspace_per_root_ = options->get_int("PCI_DL_SUBSPACE_PER_ROOT");
    e_convergence_ = options->get_double("PCI_E_CONVERGENCE");
    energy_estimate_threshold_ = options->get_double("PCI_ENERGY_ESTIMATE_THRESHOLD");
    evar_max_error_ = options->get_double("PCI_EVAR_MAX_ERROR");

    max_guess_size_ = options->get_int("PCI_MAX_GUESS_SIZE");
    energy_estimate_freq_ = options->get_int("PCI_ENERGY_ESTIMATE_FREQ");

    fast_variational_estimate_ = options->get_bool("PCI_FAST_EVAR");
    do_shift_ = options->get_bool("PCI_USE_SHIFT");
    use_inter_norm_ = options->get_bool("PCI_USE_INTER_NORM");
    do_perturb_analysis_ = options->get_bool("PCI_PERTURB_ANALYSIS");
    stop_higher_new_low_ = options->get_bool("PCI_STOP_HIGHER_NEW_LOW");
    chebyshev_order_ = options->get_int("PCI_CHEBYSHEV_ORDER");
    krylov_order_ = options->get_int("PCI_KRYLOV_ORDER");

    variational_estimate_ = options->get_bool("PCI_VAR_ESTIMATE");
    print_full_wavefunction_ = options->get_bool("PCI_PRINT_FULL_WAVEFUNCTION");

    if (options->get_str("PCI_GENERATOR") == "WALL-CHEBYSHEV") {
        generator_ = WallChebyshevGenerator;
        generator_description_ = "Wall-Chebyshev";
        if (chebyshev_order_ <= 0) {
            psi::outfile->Printf("\n\n  Warning! Chebyshev order %d out of bound, "
                                 "automatically adjusted to 5.",
                                 chebyshev_order_);
            chebyshev_order_ = 5;
        }
    } else if (options->get_str("PCI_GENERATOR") == "DL") {
        generator_ = DLGenerator;
        generator_description_ = "Davidson-Liu by Tianyuan";
        if (krylov_order_ <= 0) {
            psi::outfile->Printf("\n\n  Warning! Krylov order %d out of bound, "
                                 "automatically adjusted to 8.",
                                 krylov_order_);
            krylov_order_ = 8;
        }
    } else {
        psi::outfile->Printf("\n\n  Warning! Generator Unsupported.");
        abort();
    }

    if (options->get_str("PCI_FUNCTIONAL") == "MAX") {
        if (std::numeric_limits<double>::has_infinity) {
            functional_order_ = std::numeric_limits<double>::infinity();
        } else {
            functional_order_ = std::numeric_limits<double>::max();
        }
        prescreen_H_CI_ = [](double HJI, double CI, double spawning_threshold) {
            return std::fabs(HJI * CI) >= spawning_threshold;
        };
        important_H_CI_CJ_ = [](double, double, double, double) { return true; };
        functional_description_ = "|Hij|*max(|Ci|,|Cj|)";
    } else if (options->get_str("PCI_FUNCTIONAL") == "SUM") {
        functional_order_ = 1.0;
        prescreen_H_CI_ = [](double HJI, double CI, double spawning_threshold) {
            return std::fabs(HJI * CI) >= 0.5 * spawning_threshold;
        };
        important_H_CI_CJ_ = [](double HJI, double CI, double CJ, double spawning_threshold) {
            return std::fabs(HJI * CI) + std::fabs(HJI * CJ) >= spawning_threshold;
        };
        functional_description_ = "|Hij|*(|Ci|+|Cj|)";
    } else if (options->get_str("PCI_FUNCTIONAL") == "SQUARE") {
        functional_order_ = 2.0;
        prescreen_H_CI_ = [](double HJI, double CI, double spawning_threshold) {
            return std::fabs(HJI * CI) >= 1.4142135623730952 * spawning_threshold;
        };
        important_H_CI_CJ_ = [](double HJI, double CI, double CJ, double spawning_threshold) {
            return std::fabs(HJI) * std::sqrt(CI * CI + CJ * CJ) >= spawning_threshold;
        };
        functional_description_ = "|Hij|*sqrt(Ci^2+Cj^2)";
    } else if (options->get_str("PCI_FUNCTIONAL") == "SQRT") {
        functional_order_ = 0.5;
        prescreen_H_CI_ = [](double HJI, double CI, double spawning_threshold) {
            return std::fabs(HJI * CI) >= 0.25 * spawning_threshold;
        };
        important_H_CI_CJ_ = [](double HJI, double CI, double CJ, double spawning_threshold) {
            return std::fabs(HJI) *
                       std::pow(std::sqrt(std::fabs(CI)) + std::sqrt(std::fabs(CJ)), 2) >=
                   spawning_threshold;
        };
        functional_description_ = "|Hij|*(sqrt(|Ci|)+sqrt(|Cj|))^2";
    } else if (options->get_str("PCI_FUNCTIONAL") == "SPECIFY-ORDER") {
        functional_order_ = options->get_double("PCI_FUNCTIONAL_ORDER");
        double factor = std::pow(2.0, 1.0 / functional_order_);
        prescreen_H_CI_ = [factor](double HJI, double CI, double spawning_threshold) {
            return std::fabs(HJI * CI) * factor >= spawning_threshold;
        };
        double order = functional_order_;
        important_H_CI_CJ_ = [order](double HJI, double CI, double CJ, double spawning_threshold) {
            return std::fabs(HJI) *
                       std::pow(std::pow(std::fabs(CI), order) + std::pow(std::fabs(CJ), order),
                                1.0 / order) >=
                   spawning_threshold;
        };
        functional_description_ = "|Hij|*((|Ci|^" + std::to_string(order) + ")+(|Cj|^" +
                                  std::to_string(order) + "))^" + std::to_string(1.0 / order);
    } else {
        psi::outfile->Printf("\n\n  Warning! Functional Unsupported.");
        abort();
    }

    sparse_solver_.set_e_convergence(options->get_double("PCI_E_CONVERGENCE"));
    sparse_solver_.set_r_convergence(options->get_double("PCI_R_CONVERGENCE"));
    sparse_solver_.set_maxiter_davidson(options->get_int("DL_MAXITER"));
}

void ProjectorCI::print_info() {
    psi::outfile->Printf("\n\n\t  ---------------------------------------------------------");
    psi::outfile->Printf("\n\t              Projector Configuration Interaction");
    psi::outfile->Printf("\n\t         by Francesco A. Evangelista and Tianyuan Zhang");
    psi::outfile->Printf("\n\t                    %4d thread(s) %s", num_threads_,
                         have_omp_ ? "(OMP)" : "");
    psi::outfile->Printf("\n\t  ---------------------------------------------------------");

    // Print a summary
    std::vector<std::pair<std::string, int>> calculation_info{
        {"Symmetry", wavefunction_symmetry_},
        {"Multiplicity", wavefunction_multiplicity_},
        {"Number of roots", nroot_},
        {"Root used for properties", current_root_},
        {"Maximum number of iterations", max_cycle_},
        {"Energy estimation frequency", energy_estimate_freq_},
        {"Number of threads", num_threads_}};

    std::vector<std::pair<std::string, double>> calculation_info_double{
        {"Spawning threshold", spawning_threshold_},
        {"Initial guess spawning threshold", initial_guess_spawning_threshold_},
        {"Convergence threshold", e_convergence_},
        {"Energy estimate tollerance", energy_estimate_threshold_}};

    std::vector<std::pair<std::string, std::string>> calculation_info_string{
        {"Generator type", generator_description_},
        {"Importance functional", functional_description_},
        {"Shift the energy", do_shift_ ? "YES" : "NO"},
        {"Use intermediate normalization", use_inter_norm_ ? "YES" : "NO"},
        {"Fast variational estimate", fast_variational_estimate_ ? "YES" : "NO"},
        {"Result perturbation analysis", do_perturb_analysis_ ? "YES" : "NO"},
        {"Using OpenMP", have_omp_ ? "YES" : "NO"},
    };

    // Print some information
    psi::outfile->Printf("\n\n  ==> Calculation Information <==\n");
    for (auto& str_dim : calculation_info) {
        psi::outfile->Printf("\n    %-39s %10d", str_dim.first.c_str(), str_dim.second);
    }
    for (auto& str_dim : calculation_info_double) {
        psi::outfile->Printf("\n    %-39s %10.3e", str_dim.first.c_str(), str_dim.second);
    }
    for (auto& str_dim : calculation_info_string) {
        psi::outfile->Printf("\n    %-39s %10s", str_dim.first.c_str(), str_dim.second.c_str());
    }
}

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
    lambda_h_ = high_obt_energy;

    double lambda_h_G = as_ints_->energy(high_det);
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
    lambda_h_ = lambda_h_G;
    psi::outfile->Printf("\n\n  ==> Estimate highest excitation energy <==");
    psi::outfile->Printf("\n  Highest Excited determinant:");
    psi::outfile->Printf("\n  %s", str(high_det).c_str());
    psi::outfile->Printf("\n  Determinant Energy                    :  %.12f",
                         as_ints_->energy(high_det));
    psi::outfile->Printf("\n  Highest Energy Gershgorin circle Est. :  %.12f", lambda_h_);
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
    case WallChebyshevGenerator:
        Wall_Chebyshev_generator_coefs(cha_func_coefs_, chebyshev_order_, range_);
    default:
        break;
    }
}

void ProjectorCI::print_characteristic_function() {
    psi::outfile->Printf("\n\n  ==> Characteristic Function <==");
    print_polynomial(cha_func_coefs_);
    psi::outfile->Printf("\n    with shift = %.12f, range = %.12f", shift_, range_);
    psi::outfile->Printf("\n    Initial guess: lambda_1= %s%.12f", lambda_1_ >= 0.0 ? " " : "",
                         lambda_1_);
    psi::outfile->Printf("\n    Est. Highest eigenvalue= %s%.12f", lambda_h_ >= 0.0 ? " " : "",
                         lambda_h_);
}

void ProjectorCI::pre_iter_preparation() {
    t_pci_.reset();

    lastLow = 0.0;
    previous_go_up = false;

    if (!std::numeric_limits<double>::has_quiet_NaN) {
        psi::outfile->Printf("\n\n  The implementation does not support quiet_NaN.");
        psi::outfile->Printf("\n  Program behavior undefined. Abort.");
        abort();
    }

    dets_hashvec_.clear();
    C_.clear();

    psi::timer_on("PCI:Couplings");
    double factor = std::max(1.0, std::pow(2.0, 1.0 / functional_order_ - 0.5));
    compute_single_couplings(spawning_threshold_ / factor);
    compute_double_couplings(spawning_threshold_ / factor);
    psi::timer_off("PCI:Couplings");

    // Compute the initial guess
    psi::outfile->Printf("\n\n  ==> Initial Guess <==");
    approx_E_flag_ = true;
    var_energy_ = initial_guess(dets_hashvec_, C_);
    proj_energy_ = var_energy_;

    psi::timer_on("PCI:sort");
    sortHashVecByCoefficient(dets_hashvec_, C_);
    psi::timer_off("PCI:sort");

    //    print_wfn(dets_hashvec_, C_, 1); TODO: re-enable [sci_cleanup]
    //    det_hash<> old_space_map;
    //    for (size_t I = 0; I < dets_hashvec.size(); ++I) {
    //        old_space_map[dets_hashvec[I]] = C[I];
    //    }

    convergence_analysis();

    //    for (Determinant det : dets) {
    //        count_hash(det);
    //    }

    // Main iterations
    psi::outfile->Printf("\n\n  ==> PCI Iterations <==");
    if (variational_estimate_) {
        psi::outfile->Printf("\n\n  "
                             "------------------------------------------------------"
                             "------------------------------------------------------"
                             "----------------------------------~");
        psi::outfile->Printf("\n    Steps  Beta/Eh      Ndets      NoffDiag     Proj. Energy/Eh   "
                             "  dEp/dt      Var. Energy/Eh      dEp/dt      Approx. "
                             "Energy/Eh   dEv/dt      ~");
        psi::outfile->Printf("\n  "
                             "------------------------------------------------------"
                             "------------------------------------------------------"
                             "----------------------------------~");
    } else {
        psi::outfile->Printf("\n\n  "
                             "------------------------------------------------------"
                             "--------------------------------------------------------~");
        psi::outfile->Printf("\n    Steps  Beta/Eh      Ndets      NoffDiag     Proj. Energy/Eh   "
                             "  dEp/dt      Approx. Energy/Eh   dEv/dt      ~");
        psi::outfile->Printf("\n  "
                             "------------------------------------------------------"
                             "--------------------------------------------------------~");
    }

    old_var_energy_ = var_energy_;
    old_proj_energy_ = proj_energy_;
    converged_ = false;

    approx_E_flag_ = true;
}

void ProjectorCI::diagonalize_P_space() {}

void ProjectorCI::find_q_space() {}

void ProjectorCI::diagonalize_PQ_space() {
    psi::timer_on("PCI:Step");
    if (use_inter_norm_) {
        double max_C = std::fabs(C_[0]);
        propagate(generator_, dets_hashvec_, C_, spawning_threshold_ * max_C);
    } else {
        propagate(generator_, dets_hashvec_, C_, spawning_threshold_);
    }
    psi::timer_off("PCI:Step");

    // Orthogonalize this solution with respect to the previous ones
    psi::timer_on("PCI:Ortho");
    orthogonalize(dets_hashvec_, C_, solutions_);
    normalize(C_);
    psi::timer_off("PCI:Ortho");

    psi::timer_on("PCI:sort");
    sortHashVecByCoefficient(dets_hashvec_, C_);
    psi::timer_off("PCI:sort");
}

bool ProjectorCI::check_convergence() {
    // Compute the energy and check for convergence
    if (cycle_ % energy_estimate_freq_ == 0) {
        approx_E_flag_ = true;
        psi::timer_on("PCI:<E>");
        std::map<std::string, double> results = estimate_energy(dets_hashvec_, C_);
        psi::timer_off("PCI:<E>");

        proj_energy_ = results["PROJECTIVE ENERGY"];

        double proj_energy_gradient = (proj_energy_ - old_proj_energy_) / energy_estimate_freq_;
        double approx_energy_gradient =
            (approx_energy_ - old_approx_energy_) / energy_estimate_freq_;
        if (cycle_ == 0)
            approx_energy_gradient = 10.0 * e_convergence_ + 1.0;

        switch (generator_) {
        case DLGenerator:
            psi::outfile->Printf("\n%9d %8d %10zu %13zu %20.12f %10.3e %20.12f %10.3e     ~",
                                 cycle_, current_davidson_iter_, C_.size(), num_off_diag_elem_,
                                 proj_energy_, proj_energy_gradient, approx_energy_,
                                 approx_energy_gradient);
            break;
        default:
            psi::outfile->Printf("\n%9d %8d %10zu %13zu %20.12f %10.3e ", cycle_, cycle_, C_.size(),
                                 num_off_diag_elem_, proj_energy_, proj_energy_gradient);
            break;
        }

        if (variational_estimate_) {
            var_energy_ = results["VARIATIONAL ENERGY"];
            double var_energy_gradient = (var_energy_ - old_var_energy_) / energy_estimate_freq_;
            psi::outfile->Printf(" %20.12f %10.3e", var_energy_, var_energy_gradient);
        }

        old_var_energy_ = var_energy_;
        old_proj_energy_ = proj_energy_;

        iter_Evar_steps_.push_back(std::make_pair(cycle_, var_energy_));

        if (std::fabs(approx_energy_gradient) < e_convergence_ && cycle_ > 1) {
            converged_ = true;
            return true;
        }
        if (converge_test()) {
            return true;
        }
        if (do_shift_) {
            lambda_1_ = approx_energy_ - as_ints_->scalar_energy() - nuclear_repulsion_energy_;
            compute_characteristic_function();
        }
    }
    return false;
}

void ProjectorCI::prune_PQ_to_P() {}

void ProjectorCI::post_iter_process() {

    if (variational_estimate_) {
        psi::outfile->Printf("\n  "
                             "------------------------------------------------------"
                             "------------------------------------------------------"
                             "----------------------------------~");
    } else {
        psi::outfile->Printf("\n  "
                             "------------------------------------------------------"
                             "--------------------------------------------------------~");
    }

    if (converged_) {
        psi::outfile->Printf("\n\n  Calculation converged.");
    } else {
        psi::outfile->Printf("\n\n  Calculation %s", cycle_ != max_cycle_
                                                         ? "stoped in appearance of higher new low."
                                                         : "did not converge!");
    }

    if (do_shift_) {
        psi::outfile->Printf("\n\n  Shift applied during iteration, the "
                             "characteristic function may change every step.\n  "
                             "Characteristic function at last step:");
        print_characteristic_function();
    }

    psi::outfile->Printf("\n\n  ==> Post-Iterations <==\n");
    psi::outfile->Printf("\n  * Size of CI space                    = %zu", C_.size());
    psi::outfile->Printf("\n  * Number of off-diagonal elements     = %zu", num_off_diag_elem_);
    psi::outfile->Printf("\n  * ProjectorCI Approximate Energy    = %18.12f Eh", 1, approx_energy_);
    psi::outfile->Printf("\n  * ProjectorCI Projective  Energy    = %18.12f Eh", 1, proj_energy_);

    psi::timer_on("PCI:sort");
    sortHashVecByCoefficient(dets_hashvec_, C_);
    psi::timer_off("PCI:sort");

    if (print_full_wavefunction_) {
        //        print_wfn(dets_hashvec_, C_, 1, C_.size()); TODO: re-enable [sci_cleanup]
    } else {
        //        print_wfn(dets_hashvec_, C_, 1); TODO: re-enable [sci_cleanup]
    }

    psi::outfile->Printf("\n  %s: %f s\n", "ProjectorCI (bitset) steps finished in  ",
                         t_pci_.get());

    psi::timer_on("PCI:<E>end_v");
    if (fast_variational_estimate_) {
        var_energy_ = estimate_var_energy_sparse(dets_hashvec_, C_, evar_max_error_);
    } else {
        var_energy_ = estimate_var_energy_within_error_sigma(dets_hashvec_, C_, evar_max_error_);
    }
    psi::timer_off("PCI:<E>end_v");

    psi::Process::environment.globals["PCI ENERGY"] = var_energy_;

    psi::outfile->Printf("\n  * ProjectorCI Variational Energy    = %18.12f Eh", 1, var_energy_);
    psi::outfile->Printf("\n  * ProjectorCI Var. Corr.  Energy    = %18.12f Eh", 1,
                         var_energy_ - as_ints_->energy(reference_determinant_) -
                             nuclear_repulsion_energy_ - as_ints_->scalar_energy());

    psi::outfile->Printf("\n  * 1st order perturbation   Energy     = %18.12f Eh", 1,
                         var_energy_ - approx_energy_);

    psi::outfile->Printf("\n\n  %s: %f s", "ProjectorCI (bitset) ran in  ", t_pci_.get());

    save_wfn(dets_hashvec_, C_, solutions_);

    if (post_diagonalization_) {
        psi::outfile->Printf("\n\n  ==> Post-Diagonalization <==\n");
        psi::timer_on("PCI:Post_Diag");
        psi::SharedMatrix apfci_evecs(new psi::Matrix("Eigenvectors", C_.size(), nroot_));
        psi::SharedVector apfci_evals(new psi::Vector("Eigenvalues", nroot_));

        DeterminantHashVec det_map(std::move(dets_hashvec_));

        // set SparseCISolver options
        sparse_solver_.set_spin_project(options_->get_bool("SCI_PROJECT_OUT_SPIN_CONTAMINANTS"));
        sparse_solver_.manual_guess(false);
        sparse_solver_.set_force_diag(false);

        auto sigma_vector = make_sigma_vector(det_map, as_ints_, 0, SigmaVectorType::SparseList);
        std::tie(apfci_evals, apfci_evecs) = sparse_solver_.diagonalize_hamiltonian(
            det_map, sigma_vector, nroot_, wavefunction_multiplicity_);
        det_map.swap(dets_hashvec_);

        psi::timer_off("PCI:Post_Diag");

        double post_diag_energy =
            apfci_evals->get(current_root_) + nuclear_repulsion_energy_ + as_ints_->scalar_energy();
        psi::Process::environment.globals["PCI POST DIAG ENERGY"] = post_diag_energy;

        psi::outfile->Printf("\n\n  * ProjectorCI Post-diag   Energy    = %18.12f Eh", 1,
                             post_diag_energy);
        psi::outfile->Printf("\n  * ProjectorCI Var. Corr.  Energy    = %18.12f Eh", 1,
                             post_diag_energy - as_ints_->energy(reference_determinant_) -
                                 nuclear_repulsion_energy_ - as_ints_->scalar_energy());

        //        std::vector<double> diag_C(C_.size());

        //        for (size_t I = 0; I < C_.size(); ++I) {
        //            diag_C[I] = apfci_evecs->get(I, current_root_);
        //        }

        psi::timer_on("PCI:sort");
        sortHashVecByCoefficient(dets_hashvec_, C_);
        psi::timer_off("PCI:sort");

        if (print_full_wavefunction_) {
            //            print_wfn(dets_hashvec_, apfci_evecs, 1, diag_C.size()); TODO: re-enable
            //            [sci_cleanup]
        } else {
            //            print_wfn(dets_hashvec_, apfci_evecs, 1); TODO: re-enable [sci_cleanup]
        }
    }
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

double ProjectorCI::initial_guess(det_hashvec& dets_hashvec, std::vector<double>& C) {

    // Do one time step starting from the reference determinant
    Determinant bs_det(reference_determinant_);
    dets_hashvec.clear();
    dets_hashvec.add(bs_det);
    std::vector<double> start_C(1, 1.0);
    //    det_hashvec result_dets;
    //    size_t overlap_size;

    PCISigmaVector sigma_vector(dets_hashvec, start_C, initial_guess_spawning_threshold_, as_ints_,
                                prescreen_H_CI_, important_H_CI_CJ_, a_couplings_, b_couplings_,
                                aa_couplings_, ab_couplings_, bb_couplings_, dets_max_couplings_,
                                dets_single_max_coupling_, dets_double_max_coupling_, solutions_);

    //    overlap_size = C.size();
    psi::SharedVector C_psi = std::make_shared<psi::Vector>(sigma_vector.size()),
                      sigma_psi = std::make_shared<psi::Vector>(sigma_vector.size());
    set_psi_Vector(C_psi, start_C);
    sigma_vector.compute_sigma(sigma_psi, C_psi);
    C = to_std_vector(sigma_psi);
    num_off_diag_elem_ = sigma_vector.get_num_off_diag();

    //    apply_tau_H_symm(time_step_, initial_guess_spawning_threshold_, dets_hashvec, start_C, C,
    //    0.0,
    //                     overlap_size);

    //    dets_hashvec.swap(result_dets);
    size_t guess_size = dets_hashvec.size();
    if (guess_size == 0) {
        dets_hashvec.add(bs_det);
        C.push_back(1.0);
        ++guess_size;
    }

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
        for (size_t sI = 0; sI < max_guess_size_; ++sI) {
            size_t I = det_weight[sI].second;
            new_dets.add(dets_hashvec[I]);
        }
        dets_hashvec.swap(new_dets);
        guess_size = dets_hashvec.size();
        C.resize(guess_size);
    }

    psi::outfile->Printf("\n\n  Initial guess size = %zu", guess_size);

    psi::SharedMatrix evecs(new psi::Matrix("Eigenvectors", guess_size, nroot_));
    psi::SharedVector evals(new psi::Vector("Eigenvalues", nroot_));
    //  std::vector<DynamicBitsetDeterminant> dyn_dets;
    // for (auto& d : dets){
    //   DynamicBitsetDeterminant dbs = d.to_dynamic_bitset();
    //  dyn_dets.push_back(dbs);
    // }
    sparse_solver_.set_spin_project(true);
    sparse_solver_.manual_guess(false);
    sparse_solver_.set_force_diag(false);

    DeterminantHashVec det_map(dets_hashvec_);
    auto sigma_vector_diag = make_sigma_vector(det_map, as_ints_, 0, SigmaVectorType::SparseList);
    std::tie(evals, evecs) = sparse_solver_.diagonalize_hamiltonian(
        det_map, sigma_vector_diag, nroot_, wavefunction_multiplicity_);

    double var_energy =
        evals->get(current_root_) + nuclear_repulsion_energy_ + as_ints_->scalar_energy();
    psi::outfile->Printf("\n\n  Initial guess energy (variational) = %20.12f Eh (root = %d)",
                         var_energy, current_root_ + 1);

    lambda_1_ = evals->get(current_root_);

    // Copy the ground state eigenvector
    for (size_t I = 0; I < guess_size; ++I) {
        C[I] = evecs->get(I, current_root_);
    }
    //    psi::outfile->Printf("\n\n  Reached here");
    return var_energy;
}

void ProjectorCI::propagate(GeneratorType generator, det_hashvec& dets_hashvec,
                            std::vector<double>& C, double spawning_threshold) {
    switch (generator) {
    case WallChebyshevGenerator:
        propagate_wallCh(dets_hashvec, C, spawning_threshold);
        break;
    case DLGenerator:
        propagate_DL(dets_hashvec, C, spawning_threshold);
        break;
    default:
        psi::outfile->Printf("\n\n  Selected Generator Unsupported!!!");
        abort();
        break;
    }
    normalize(C);
}

void ProjectorCI::propagate_wallCh(det_hashvec& dets_hashvec, std::vector<double>& C,
                                   double spawning_threshold) {
    //    det_hashvec dets_hashvec(dets);
    // A map that contains the pair (determinant,coefficient)
    const double PI = 2 * acos(0.0);
    //    det_hash<> dets_C_hash;
    std::vector<double> ref_C(C);
    //    det_hashvec result_dets;
    size_t overlap_size;

    double root = -cos(((double)chebyshev_order_) * PI / (chebyshev_order_ + 0.5));

    PCISigmaVector sigma_vector(dets_hashvec, ref_C, spawning_threshold, as_ints_, prescreen_H_CI_,
                                important_H_CI_CJ_, a_couplings_, b_couplings_, aa_couplings_,
                                ab_couplings_, bb_couplings_, dets_max_couplings_,
                                dets_single_max_coupling_, dets_double_max_coupling_, solutions_);

    overlap_size = ref_C.size();
    psi::SharedVector C_psi = std::make_shared<psi::Vector>(sigma_vector.size()),
                      sigma_psi = std::make_shared<psi::Vector>(sigma_vector.size());
    set_psi_Vector(C_psi, ref_C);
    sigma_vector.compute_sigma(sigma_psi, C_psi);
    sigma_psi->scale(-1.0);
    C = to_std_vector(sigma_psi);
    num_off_diag_elem_ = sigma_vector.get_num_off_diag();

    double S = range_ * root + shift_;
#pragma omp parallel for
    for (size_t I = 0; I < overlap_size; ++I) {
        C[I] += S * ref_C[I];
    }

    if (approx_E_flag_) {
        psi::timer_on("PCI:<E>a");
        double CHC_energy = 0.0;
#pragma omp parallel for reduction(+ : CHC_energy)
        for (size_t I = 0; I < overlap_size; ++I) {
            CHC_energy += ref_C[I] * C[I];
        }
        CHC_energy = CHC_energy / -1.0 + (range_ * root + shift_) + as_ints_->scalar_energy() +
                     nuclear_repulsion_energy_;
        psi::timer_off("PCI:<E>a");
        double CHC_energy_gradient = (CHC_energy - approx_energy_) / energy_estimate_freq_;
        old_approx_energy_ = approx_energy_;
        approx_energy_ = CHC_energy;
        approx_E_flag_ = false;
        approx_E_tau_ = -1.0;
        approx_E_S_ = S;
        if (cycle_ != 0)
            psi::outfile->Printf(" %20.12f %10.3e     ~", approx_energy_, CHC_energy_gradient);
    }
    //    apply_tau_H_symm(-1.0, spawning_threshold, dets_hashvec, ref_C, C, range_ * root + shift_,
    //                     overlap_size);
    normalize(C);

    for (int i = chebyshev_order_ - 1; i > 0; i--) {
        //        psi::outfile->Printf("\nCurrent root:%.12lf",range_ * root +
        //        shift_);
        //        apply_tau_H(-1.0/range_,spawning_threshold,dets,C,dets_C_hash,
        //        range_ * root + shift_);
        double root = -cos(((double)i) * PI / (chebyshev_order_ + 0.5));
        //        dets = dets_hashvec.toVector();
        std::vector<double> result_C;

        set_psi_Vector(C_psi, C);
        sigma_vector.compute_sigma(sigma_psi, C_psi);
        sigma_psi->scale(-1.0);
        result_C = to_std_vector(sigma_psi);
        S = range_ * root + shift_;
#pragma omp parallel for
        for (size_t I = 0; I < sigma_vector.size(); ++I) {
            result_C[I] += S * C[I];
        }
        //        apply_tau_H_ref_C_symm(-1.0, spawning_threshold, dets_hashvec, ref_C, C, result_C,
        //                               overlap_size, range_ * root + shift_);
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

void ProjectorCI::propagate_DL(det_hashvec& dets_hashvec, std::vector<double>& C,
                               double spawning_threshold) {
    auto sigma_vector = std::make_shared<PCISigmaVector>(
        dets_hashvec, C, spawning_threshold, as_ints_, prescreen_H_CI_, important_H_CI_CJ_,
        a_couplings_, b_couplings_, aa_couplings_, ab_couplings_, bb_couplings_,
        dets_max_couplings_, dets_single_max_coupling_, dets_double_max_coupling_, solutions_);
    num_off_diag_elem_ = sigma_vector->get_num_off_diag();
    size_t ref_size = C.size();

    //    outfile->Printf("\n\n result_size = %zu", result_size);

    std::vector<std::pair<size_t, double>> guess(ref_size);
    for (size_t I = 0; I < ref_size; ++I) {
        guess[I] = std::make_pair(I, C[I]);
    }
    sparse_solver_.set_initial_guess({guess});
    sparse_solver_.set_spin_project(false);
    sparse_solver_.set_force_diag(true);
    psi::SharedMatrix PQ_evecs_;
    psi::SharedVector PQ_evals_;

    DeterminantHashVec det_map(dets_hashvec_);

    // set SparseCISolver options
    sparse_solver_.set_spin_project(true);
    sparse_solver_.manual_guess(false);
    sparse_solver_.set_force_diag(false);

    auto sigma_vector2 = make_sigma_vector(det_map, as_ints_, 0, SigmaVectorType::SparseList);

    size_t result_size = sigma_vector->size();

    std::tie(PQ_evals_, PQ_evecs_) = sparse_solver_.diagonalize_hamiltonian(
        det_map, sigma_vector, nroot_, state_.multiplicity());

    current_davidson_iter_ = sigma_vector->get_sigma_build_count();
    old_approx_energy_ = approx_energy_;
    approx_energy_ = PQ_evals_->get(0) + as_ints_->scalar_energy() + nuclear_repulsion_energy_;
    C.resize(result_size);
    for (size_t I = 0; I < result_size; ++I) {
        C[I] = PQ_evecs_->get(I, 0);
    }
}

std::map<std::string, double> ProjectorCI::estimate_energy(const det_hashvec& dets_hashvec,
                                                           std::vector<double>& C) {
    std::map<std::string, double> results;
    //    det_hashvec dets_hashvec(dets);
    //    dets = dets_hashvec.toVector();
    psi::timer_on("PCI:<E>p");
    results["PROJECTIVE ENERGY"] = estimate_proj_energy(dets_hashvec, C);
    psi::timer_off("PCI:<E>p");

    if (variational_estimate_) {
        if (fast_variational_estimate_) {
            psi::timer_on("PCI:<E>vs");
            results["VARIATIONAL ENERGY"] =
                estimate_var_energy_sparse(dets_hashvec, C, energy_estimate_threshold_);
            psi::timer_off("PCI:<E>vs");
        } else {
            psi::timer_on("PCI:<E>v");
            results["VARIATIONAL ENERGY"] =
                estimate_var_energy(dets_hashvec, C, energy_estimate_threshold_);
            psi::timer_off("PCI:<E>v");
        }
    }
    //    dets_hashvec = det_hashvec(dets);
    //    dets = dets_hashvec.toVector();
    return results;
}

static bool abs_compare(double a, double b) { return (std::abs(a) < std::abs(b)); }

double ProjectorCI::estimate_proj_energy(const det_hashvec& dets_hashvec, std::vector<double>& C) {
    // Find the determinant with the largest value of C
    auto result = std::max_element(C.begin(), C.end(), abs_compare);
    size_t J = std::distance(C.begin(), result);
    double CJ = C[J];

    // Compute the projective energy
    double projective_energy_estimator = 0.0;
    for (int I = 0, max_I = dets_hashvec.size(); I < max_I; ++I) {
        double HIJ = as_ints_->slater_rules(dets_hashvec[I], dets_hashvec[J]);
        projective_energy_estimator += HIJ * C[I] / CJ;
    }
    return projective_energy_estimator + nuclear_repulsion_energy_ + as_ints_->scalar_energy();
}

double ProjectorCI::estimate_var_energy(const det_hashvec& dets_hashvec, std::vector<double>& C,
                                        double tollerance) {
    // Compute a variational estimator of the energy
    size_t size = dets_hashvec.size();
    double variational_energy_estimator = 0.0;
#pragma omp parallel for reduction(+ : variational_energy_estimator)
    for (size_t I = 0; I < size; ++I) {
        const Determinant& detI = dets_hashvec[I];
        variational_energy_estimator += C[I] * C[I] * as_ints_->energy(detI);
        for (size_t J = I + 1; J < size; ++J) {
            if (std::fabs(C[I] * C[J]) > tollerance) {
                double HIJ = as_ints_->slater_rules(dets_hashvec[I], dets_hashvec[J]);
                variational_energy_estimator += 2.0 * C[I] * HIJ * C[J];
            }
        }
    }
    return variational_energy_estimator + nuclear_repulsion_energy_ + as_ints_->scalar_energy();
}

double ProjectorCI::estimate_var_energy_within_error(const det_hashvec& dets_hashvec,
                                                     std::vector<double>& C, double max_error) {
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
    psi::outfile->Printf(
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

double ProjectorCI::estimate_var_energy_within_error_sigma(const det_hashvec& dets_hashvec,
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
    psi::outfile->Printf(
        "\n  Variational energy estimated with %zu determinants to meet the max error %e",
        cut_index + 1, max_error);
    double variational_energy_estimator = 0.0;

    std::vector<Determinant> sub_dets = dets_hashvec.toVector();
    sub_dets.erase(sub_dets.begin() + cut_index + 1, sub_dets.end());
    DeterminantHashVec det_map(sub_dets);
    auto sigma_vector = make_sigma_vector(det_map, as_ints_, 0, SigmaVectorType::SparseList);
    size_t sub_size = sigma_vector->size();
    // allocate vectors
    psi::SharedVector b(new psi::Vector("b", sub_size));
    psi::SharedVector sigma(new psi::Vector("sigma", sub_size));
    for (size_t i = 0; i < sub_size; ++i) {
        b->set(i, C[i]);
    }
    sigma_vector->compute_sigma(sigma, b);
    variational_energy_estimator = sigma->dot(b.get());

    variational_energy_estimator /= 1.0 - cume_ignore;
    return variational_energy_estimator + nuclear_repulsion_energy_ + as_ints_->scalar_energy();
}

double ProjectorCI::estimate_var_energy_sparse(const det_hashvec& dets_hashvec,
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
    psi::outfile->Printf(
        "\n  Variational energy estimated with %zu determinants to meet the max error %e",
        cut_index + 1, max_error);

    psi::timer_on("PCI:Couplings");
    compute_couplings_half(dets_hashvec, cut_index + 1);
    psi::timer_off("PCI:Couplings");

    double variational_energy_estimator = 0.0;
    std::vector<double> energy(num_threads_, 0.0);

    size_t full_num_off_diag_elem = 0;
#pragma omp parallel for reduction(+ : full_num_off_diag_elem)
    for (size_t I = 0; I <= cut_index; ++I) {
        size_t thread_num_off_diag_elem = 0;
        energy[omp_get_thread_num()] += form_H_C(dets_hashvec, C, I, thread_num_off_diag_elem);
        full_num_off_diag_elem += thread_num_off_diag_elem;
    }
    psi::outfile->Printf("\n  * Subspace Hamiltonian number of off-diagonal elements = %zu",
                         full_num_off_diag_elem);
    for (int t = 0; t < num_threads_; ++t) {
        variational_energy_estimator += energy[t];
    }
    variational_energy_estimator /= 1.0 - cume_ignore;
    return variational_energy_estimator + nuclear_repulsion_energy_ + as_ints_->scalar_energy();
}

// void ProjectorCI::print_wfn(const det_hashvec& space_hashvec, std::vector<double>& C,
//                            size_t max_output) {
//    psi::outfile->Printf("\n\n  Most important contributions to the wave function:\n");

//    size_t max_dets = std::min(int(max_output), int(C.size()));
//    for (size_t I = 0; I < max_dets; ++I) {
//        psi::outfile->Printf("\n  %3zu  %13.6g %13.6g  %10zu %s  %18.12f", I, C[I], C[I] * C[I],
//        I,
//                             str(space_hashvec[I]).c_str(),
//                             as_ints_->energy(space_hashvec[I]) + as_ints_->scalar_energy());
//    }

//    // Compute the expectation value of the spin
//    size_t max_sample = 1000;
//    size_t max_I = 0;
//    double sum_weight = 0.0;
//    double wfn_threshold = 0.95;
//    for (size_t I = 0; I < space_hashvec.size(); ++I) {
//        if ((sum_weight < wfn_threshold) and (I < max_sample)) {
//            sum_weight += C[I] * C[I];
//            max_I++;
//        } else if (std::fabs(C[I - 1]) - std::fabs(C[I]) < 1.0e-6) {
//            // Special case, if there are several equivalent determinants
//            sum_weight += C[I] * C[I];
//            max_I++;
//        } else {
//            break;
//        }
//    }

//    double norm = 0.0;
//    double S2 = 0.0;
//    for (size_t I = 0; I < max_I; ++I) {
//        for (size_t J = 0; J < max_I; ++J) {
//            if (std::fabs(C[I] * C[J]) > 1.0e-12) {
//                const double S2IJ = spin2(space_hashvec[I], space_hashvec[J]);
//                S2 += C[I] * C[J] * S2IJ;
//            }
//        }
//        norm += C[I] * C[I];
//    }
//    S2 /= norm;
//    double S = std::fabs(0.5 * (std::sqrt(1.0 + 4.0 * S2) - 1.0));

//    std::vector<std::string> s2_labels({"singlet", "doublet", "triplet", "quartet", "quintet",
//                                        "sextet", "septet", "octet", "nonet", "decaet"});
//    size_t nLet = std::round(S * 2.0);
//    std::string state_label;
//    if (nLet < 10) {
//        state_label = s2_labels[nLet];
//    } else {
//        state_label = std::to_string(nLet) + "-let";
//    }

//    psi::outfile->Printf("\n\n  Spin State: S^2 = %5.3f, S = %5.3f, %s (from %zu "
//                         "determinants,%.2f%%)",
//                         S2, S, state_label.c_str(), max_I, 100.0 * sum_weight);
//}

void ProjectorCI::save_wfn(det_hashvec& space, std::vector<double>& C,
                           std::vector<std::pair<det_hashvec, std::vector<double>>>& solutions) {
    psi::outfile->Printf("\n\n  Saving the wave function:\n");

    //    det_hash<> solution;
    //    for (size_t I = 0; I < space.size(); ++I) {
    //        solution[space[I]] = C[I];
    //    }
    //    solutions.push_back(std::move(solution));
    solutions.push_back(std::make_pair(space, C));
}

void ProjectorCI::orthogonalize(
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

double ProjectorCI::form_H_C(const det_hashvec& dets_hashvec, std::vector<double>& C, size_t I,
                             size_t& thread_num_off_diag_elem) {
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
                    thread_num_off_diag_elem += 2;
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
                    thread_num_off_diag_elem += 2;
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
                            thread_num_off_diag_elem += 2;
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
                            thread_num_off_diag_elem += 2;
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
                            thread_num_off_diag_elem += 2;
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

double ProjectorCI::form_H_C_2(const det_hashvec& dets_hashvec, std::vector<double>& C, size_t I,
                               size_t cut_index) {
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

void ProjectorCI::compute_single_couplings(double single_coupling_threshold) {
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

void ProjectorCI::compute_double_couplings(double double_coupling_threshold) {
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

void ProjectorCI::compute_couplings_half(const det_hashvec& dets, size_t cut_size) {
    Determinant andBits(dets[0]), orBits(dets[0]);
    andBits.flip();
    for (size_t i = 0; i < cut_size; ++i) {
        andBits = andBits & dets[i]; // common_occupation(andBits, dets[i]);
        orBits = orBits | dets[i];   // union_occupation(orBits, dets[i]);
    }
    Determinant actBits = andBits ^ orBits; // different_occupation(andBits, orBits);

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

void ProjectorCI::set_method_variables(
    std::string ex_alg, size_t nroot_method, size_t root,
    const std::vector<std::vector<std::pair<Determinant, double>>>& old_roots) {
    if (!((ex_alg == "ROOT_ORTHOGONALIZE") or (ex_alg == "NONE"))) {
        throw psi::PSIEXCEPTION(ex_alg + " has not been implemented in PCI.");
    }
    nroot_ = nroot_method;
    current_root_ = root;

    solutions_.clear();
    solutions_.reserve(old_roots.size());
    for (const auto& old_root : old_roots) {
        std::vector<double> C;
        C.reserve(old_root.size());
        det_hashvec dets;
        dets.reserve(old_root.size());
        merge(dets, C, old_root);
        solutions_.push_back(std::make_pair(dets, C));
    }
}

DeterminantHashVec ProjectorCI::get_PQ_space() { return solutions_[solutions_.size() - 1].first; }
psi::SharedMatrix ProjectorCI::get_PQ_evecs() {
    const auto& C = solutions_[solutions_.size() - 1].second;
    size_t nDet = C.size();
    psi::SharedMatrix evecs = std::make_shared<psi::Matrix>("U", nDet, nroot_);
    for (size_t i = 0; i < nDet; ++i) {
        evecs->set(i, 0, C[i]);
    }
    return evecs;
}

psi::SharedVector ProjectorCI::get_PQ_evals() {
    psi::SharedVector evals = std::make_shared<psi::Vector>("e", nroot_);
    evals->set(0, approx_energy_ - as_ints_->scalar_energy() - nuclear_repulsion_energy_);
    return evals;
}

size_t ProjectorCI::get_ref_root() { return current_root_; }
std::vector<double> ProjectorCI::get_multistate_pt2_energy_correction() {
    return std::vector<double>(nroot_);
}
} // namespace forte
