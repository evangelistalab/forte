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

#include <cmath>

#include "psi4/libpsi4util/PsiOutStream.h"

#include "helpers/timer.h"
#include "base_classes/mo_space_info.h"
#include "helpers/iterative_solvers.h"

#define PRINT_VARS(msg)                                                                            \
    // std::vector<std::pair<size_t, std::string>> v = {{collapse_size_, "collapse_size_"},           \
    //                                                  {subspace_size_, "subspace_size_"},           \
    //                                                  {basis_size_, "basis_size_"},                 \
    //                                                  {sigma_size_, "sigma_size_"},                 \
    //                                                  {nroot_, "nroot_"}};                          \
    // outfile->Printf("\n\n => %s <=", msg);                                                         \
    // for (auto vk : v) {                                                                            \
    //     outfile->Printf("\n    %-30s  %zu", vk.second.c_str(), vk.first);                          \
    // }

using namespace psi;

namespace forte {

DavidsonLiuSolver::DavidsonLiuSolver(size_t size, size_t nroot) : size_(size), nroot_(nroot) {
    if (size_ == 0)
        throw std::runtime_error("DavidsonLiuSolver called with space of dimension zero.");
    residual_.resize(nroot, 0.0);
}

void DavidsonLiuSolver::startup(psi::SharedVector diagonal) {
    // set space size
    collapse_size_ = std::min(collapse_per_root_ * nroot_, size_);
    subspace_size_ = std::min(subspace_per_root_ * nroot_, size_);

    basis_size_ = 0; // start with no vectors
    sigma_size_ = 0; // start with no sigma vectors
    iter_ = 0;
    converged_ = 0;

    // store the basis vectors (each row is a vector)
    b_ = std::make_shared<psi::Matrix>("b", subspace_size_, size_);
    b_->zero();
    bnew = std::make_shared<psi::Matrix>("bnew", subspace_size_, size_);
    f = std::make_shared<psi::Matrix>("f", subspace_size_, size_);
    sigma_ = std::make_shared<psi::Matrix>("sigma", size_, subspace_size_);
    sigma_->zero();

    G = std::make_shared<psi::Matrix>("G", subspace_size_, subspace_size_);
    S = std::make_shared<psi::Matrix>("S", subspace_size_, subspace_size_);
    alpha = std::make_shared<psi::Matrix>("alpha", subspace_size_, subspace_size_);

    lambda = std::make_shared<psi::Vector>("lambda", subspace_size_);
    lambda_old = std::make_shared<psi::Vector>("lambda_old", subspace_size_);
    h_diag = std::make_shared<psi::Vector>("h_diag", size_);

    h_diag->copy(*diagonal);
}

DavidsonLiuSolver::~DavidsonLiuSolver() {}

void DavidsonLiuSolver::set_print_level(size_t n) { print_level_ = n; }

void DavidsonLiuSolver::set_e_convergence(double value) { e_convergence_ = value; }

void DavidsonLiuSolver::set_r_convergence(double value) { r_convergence_ = value; }

double DavidsonLiuSolver::get_e_convergence() const { return e_convergence_; }

double DavidsonLiuSolver::get_r_convergence() const { return r_convergence_; }

void DavidsonLiuSolver::set_collapse_per_root(int value) { collapse_per_root_ = value; }

void DavidsonLiuSolver::set_subspace_per_root(int value) { subspace_per_root_ = value; }

size_t DavidsonLiuSolver::collapse_size() const { return collapse_size_; }

size_t DavidsonLiuSolver::sigma_size() const { return sigma_size_; }

size_t DavidsonLiuSolver::basis_size() const { return basis_size_; }

void DavidsonLiuSolver::add_guess(psi::SharedVector vec) {
    // Give the next b that does not have a sigma
    for (size_t j = 0; j < size_; ++j) {
        b_->set(basis_size_, j, vec->get(j));
    }
    basis_size_++;
}

void DavidsonLiuSolver::get_b(psi::SharedVector vec) {
    PRINT_VARS("get_b")
    // Give the next b that does not have a sigma
    for (size_t j = 0; j < size_; ++j) {
        vec->set(j, b_->get(sigma_size_, j));
    }
}

void DavidsonLiuSolver::get_b(psi::SharedVector vec, size_t i) {
    PRINT_VARS("get_b")
    // Give the i-th b
    for (size_t j = 0; j < size_; ++j) {
        vec->set(j, b_->get(i, j));
    }
}

bool DavidsonLiuSolver::add_sigma(psi::SharedVector vec) {
    PRINT_VARS("add_sigma")
    // Place the new sigma vector at the end
    for (size_t j = 0; j < size_; ++j) {
        sigma_->set(j, sigma_size_, vec->get(j));
    }
    sigma_size_++;
    return (sigma_size_ < basis_size_);
}

void DavidsonLiuSolver::set_sigma(psi::SharedVector vec, size_t i) {
    PRINT_VARS("set_sigma")
    // Set the i-th sigma vector
    for (size_t j = 0; j < size_; ++j) {
        sigma_->set(j, i, vec->get(j));
    }
}

void DavidsonLiuSolver::set_hdiag(psi::SharedVector hdiag) {
    h_diag->copy(*hdiag);
}

void DavidsonLiuSolver::set_project_out(std::vector<sparse_vec> project_out) {
    project_out_ = project_out;
}

psi::SharedVector DavidsonLiuSolver::eigenvalues() const { return lambda; }

psi::SharedMatrix DavidsonLiuSolver::eigenvectors() const { return bnew; }

psi::SharedVector DavidsonLiuSolver::eigenvector(size_t n) const {
    double** v = bnew->pointer();

    psi::SharedVector evec(new psi::Vector("V", size_));
    for (size_t I = 0; I < size_; I++) {
        evec->set(I, v[n][I]);
    }
    return evec;
}

std::vector<double> DavidsonLiuSolver::residuals() const { return residual_; }

void DavidsonLiuSolver::reset_convergence() { converged_ = 0; }

SolverStatus DavidsonLiuSolver::update() {
    // If converged or exceeded the maximum number of iterations return true
    // if ((converged_ >= nroot_) or (iter_ > maxiter_)) return
    // SolverStatus::Converged;
    if ((converged_ >= nroot_))
        return SolverStatus::Converged;

    PRINT_VARS("update")

    local_timer t_davidson;

    // form and diagonalize mini-matrix
    G->zero();
    G->gemm(false, false, 1.0, b_, sigma_, 0.0);
    check_G_hermiticity();
    G->diagonalize(alpha, lambda);

    bool is_energy_converged = false;
    bool is_residual_converged = false;
    if (not last_update_collapsed_) {
        auto converged = check_convergence();
        is_energy_converged = converged.first;
        is_residual_converged = converged.second;
        if (is_energy_converged and is_residual_converged) {
            get_results();
            return SolverStatus::Converged;
        }
    }

    if (size_ == 1)
        return SolverStatus::Converged;

    check_orthogonality();

    // Do subspace collapse
    if (subspace_collapse()) {
        last_update_collapsed_ = true;
        return SolverStatus::Collapse;
    } else {
        last_update_collapsed_ = false;
    }

    // Step #3: Build the Correction Vectors
    // form preconditioned residue vectors
    form_correction_vectors();

    // Step #3b: Project out undesired roots
    project_out_roots(f);

    // Step #4: Normalize the Correction Vectors
    auto f_norm = normalize_vectors(f, nroot_);

    // schmidt orthogonalize the f[k] against the set of b[i] and add new
    // vectors
    size_t num_added = 0;
    for (size_t k = 0; k < nroot_; k++) {
        if (basis_size_ < subspace_size_) {
            // check that the norm of the correction vector (before normalization) is "not small"
            if (f_norm[k] > 0.01 * r_convergence_) {
                // Schmidt-orthogonalize the correction vector
                if (schmidt_add(b_, basis_size_, size_, f, k)) {
                    basis_size_++; // <- Increase L if we add one more basis vector
                    num_added += 1;
                } else {
                    outfile->Printf("\n  Rejected new correction vector %d with norm: %f", k,
                                    f_norm[k]);
                }
            }
        }
    }

    // if we do not add any new vector then we are in trouble and we better finish the computation
    if ((num_added == 0) and is_energy_converged) {
        return SolverStatus::Converged;
    }

    iter_++;

    timing_ += t_davidson.get();

    return SolverStatus::NotConverged;
}

void DavidsonLiuSolver::form_correction_vectors() {
    f->zero();
    double* lambda_p = lambda->pointer();
    double* Adiag_p = h_diag->pointer();
    double** b_p = b_->pointer();
    double** f_p = f->pointer();
    double** alpha_p = alpha->pointer();
    double** sigma_p = sigma_->pointer();

    for (size_t k = 0; k < nroot_; k++) { // loop over roots
        residual_[k] = 0.0;
        for (size_t I = 0; I < size_; I++) { // loop over elements
            for (size_t i = 0; i < basis_size_; i++) {
                f_p[k][I] += alpha_p[i][k] * (sigma_p[I][i] - lambda_p[k] * b_p[i][I]);
            }
            residual_[k] += std::pow(f_p[k][I], 2.0);

            double denom = lambda_p[k] - Adiag_p[I];
            if (std::fabs(denom) > 1.0e-6) {
                f_p[k][I] /= denom;
            } else {
                f_p[k][I] = 0.0;
            }
        }
        residual_[k] = std::sqrt(residual_[k]);
    }
}

void DavidsonLiuSolver::compute_residual_norm() {
    double* lambda_p = lambda->pointer();
    double** b_p = b_->pointer();
    double** alpha_p = alpha->pointer();
    double** sigma_p = sigma_->pointer();

    for (size_t k = 0; k < nroot_; k++) { // loop over roots
        residual_[k] = 0.0;
        for (size_t I = 0; I < size_; I++) { // loop over elements
            double r = 0.0;
            for (size_t i = 0; i < basis_size_; i++) {
                r += alpha_p[i][k] * (sigma_p[I][i] - lambda_p[k] * b_p[i][I]);
            }
            residual_[k] += std::pow(r, 2.0);
        }
        residual_[k] = std::sqrt(residual_[k]);
    }
}

void DavidsonLiuSolver::project_out_roots(psi::SharedMatrix v) {
    double** v_p = v->pointer();
    for (size_t k = 0; k < nroot_; k++) {
        for (auto& bad_root : project_out_) {
            double overlap = 0.0;
            for (auto& I_CI : bad_root) {
                size_t I = I_CI.first;
                double CI = I_CI.second;
                overlap += v_p[k][I] * CI;
            }
            for (auto& I_CI : bad_root) {
                size_t I = I_CI.first;
                double CI = I_CI.second;
                v_p[k][I] -= overlap * CI;
            }
        }
    }
}

std::vector<double> DavidsonLiuSolver::normalize_vectors(psi::SharedMatrix v, size_t n) {
    // normalize each residual
    std::vector<double> v_norm;
    double** v_p = v->pointer();
    for (size_t k = 0; k < n; k++) {
        double norm = 0.0;
        for (size_t I = 0; I < size_; I++) {
            norm += v_p[k][I] * v_p[k][I];
        }
        norm = std::sqrt(norm);
        v_norm.push_back(norm);
        for (size_t I = 0; I < size_; I++) {
            v_p[k][I] /= norm;
        }
    }
    return v_norm;
}

bool DavidsonLiuSolver::subspace_collapse() {
    if (collapse_size_ + nroot_ > subspace_size_) { // in this case I will never
                                                    // be able to add new
                                                    // vectors
        // collapse vectors
        collapse_vectors();

        // normalize new vectors
        normalize_vectors(bnew, collapse_size_);

        // Copy them into place
        b_->zero();
        sigma_->zero();
        basis_size_ = 0;
        sigma_size_ = 0;
        for (size_t k = 0; k < collapse_size_; k++) {
            double norm_bnew_k = std::fabs(bnew->get_row(0, k)->norm());
            if (norm_bnew_k > schmidt_threshold_) {
                if (schmidt_add(b_, k, size_, bnew, k)) {
                    basis_size_++; // <- Increase L if we add one more basis vector
                }
            }
        }
        return false;
    }

    // If L is close to maxdim, collapse to one guess per root */
    if (nroot_ + basis_size_ > subspace_size_) { // this means that next
                                                 // iteration I cannot add more
                                                 // roots so I need to collapse
        if (print_level_ > 1) {
            outfile->Printf("\nSubspace too large: max subspace size = %d, "
                            "basis size = %d\n",
                            subspace_size_, basis_size_);
            outfile->Printf("Collapsing eigenvectors.\n");
        }
        // collapse vectors
        collapse_vectors();

        // normalize new vectors
        normalize_vectors(bnew, collapse_size_);

        // Copy them into place
        b_->zero();
        sigma_->zero();
        basis_size_ = 0;
        sigma_size_ = 0;
        for (size_t k = 0; k < collapse_size_; k++) {
            if (schmidt_add(b_, k, size_, bnew, k)) {
                basis_size_++; // <- Increase L if we add one more basis vector
            }
        }

        /// Need new sigma vectors to continue, so return control to caller
        return true;
    }
    return false;
}

void DavidsonLiuSolver::collapse_vectors() {
    bnew->zero();
    double** alpha_p = alpha->pointer();
    double** b_p = b_->pointer();
    double** bnew_p = bnew->pointer();
    for (size_t i = 0; i < collapse_size_; i++) {
        for (size_t j = 0; j < basis_size_; j++) {
            for (size_t k = 0; k < size_; k++) {
                bnew_p[i][k] += alpha_p[j][i] * b_p[j][k];
            }
        }
    }
}

/// add a new vector and orthogonalize it against the existing ones
bool DavidsonLiuSolver::schmidt_add(psi::SharedMatrix Amat, size_t rows, size_t cols,
                                    psi::SharedMatrix vvec, int l) {
    double** A = Amat->pointer();
    double* v = vvec->pointer()[l];

    for (size_t i = 0; i < rows; i++) {
        const auto dotval = C_DDOT(cols, A[i], 1, v, 1);
        for (size_t I = 0; I < cols; I++)
            v[I] -= dotval * A[i][I];
    }

    const auto normval = std::sqrt(C_DDOT(cols, v, 1, v, 1));
    if (normval < schmidt_threshold_)
        return false;
    for (size_t I = 0; I < cols; I++)
        A[rows][I] = v[I] / normval;
    return true;
}

std::pair<bool, bool> DavidsonLiuSolver::check_convergence() {
    compute_residual_norm();
    // check convergence on all roots
    size_t num_converged_energy = 0;
    size_t num_converged_residual = 0;
    converged_ = 0;
    if (print_level_ > 1) {
        outfile->Printf("\n  Root      Eigenvalue        Delta   Converged?\n");
        outfile->Printf("  ---- -------------------- --------- ----------\n");
    }
    for (size_t k = 0; k < nroot_; k++) {
        // check if the energy converged
        double diff = std::fabs(lambda->get(k) - lambda_old->get(k));
        bool this_energy_converged = (diff < e_convergence_);
        // check if the residual converged
        bool this_residual_converged = (residual_[k] < r_convergence_);
        // update counters
        num_converged_energy += this_energy_converged;
        num_converged_residual += this_residual_converged;

        if (this_energy_converged and this_residual_converged) {
            converged_++;
        }
        // update the old eigenvalue
        lambda_old->set(k, lambda->get(k));

        if (print_level_ > 1) {
            bool this_converged = (this_energy_converged and this_residual_converged);
            outfile->Printf("  %3d  %20.14f %4.3e      %1s\n", k, lambda->get(k), diff,
                            this_converged ? "Y" : "N");
        }
    }

    bool is_energy_converged = (num_converged_energy == nroot_);
    bool is_residual_converged = (num_converged_residual == nroot_);
    return std::make_pair(is_energy_converged, is_residual_converged);
}

void DavidsonLiuSolver::get_results() {
    /* generate final eigenvalues and eigenvectors */
    double** alpha_p = alpha->pointer();
    double** b_p = b_->pointer();
    double* eps = lambda_old->pointer();
    double** v = bnew->pointer();
    bnew->zero();

    for (size_t i = 0; i < nroot_; i++) {
        eps[i] = lambda->get(i);
        for (size_t j = 0; j < basis_size_; j++) {
            for (size_t I = 0; I < size_; I++) {
                v[i][I] += alpha_p[j][i] * b_p[j][I];
            }
        }
        // Normalize v
        double norm = 0.0;
        for (size_t I = 0; I < size_; I++) {
            norm += v[i][I] * v[i][I];
        }
        norm = std::sqrt(norm);
        for (size_t I = 0; I < size_; I++) {
            v[i][I] /= norm;
        }
    }
    //    if (print_level_){
    //        outfile->Printf("\n  The Davidson-Liu algorithm converged in %d
    //        iterations.", iter_);
    //        outfile->Printf("\n  %s: %f s","Time spent diagonalizing
    //        H",timing_);
    //    }
}

void DavidsonLiuSolver::check_orthogonality() {
    // Compute the overlap matrix
    S->gemm(false, true, 1.0, b_, b_, 0.0);

    // Check for normalization
    double maxdiag = 0.0;
    for (size_t i = 0; i < basis_size_; ++i) {
        maxdiag = std::max(maxdiag, std::fabs(S->get(i, i) - 1.0));
    }
    if (maxdiag > orthogonality_threshold_) {
        S->print();
        outfile->Printf("\n  Maximum absolute deviation from normalization: %e", maxdiag);
        std::string msg =
            "DavidsonLiuSolver::check_orthogonality(): eigenvectors are not normalized!";
        throw std::runtime_error(msg);
    }

    // Check for orthogonality
    double maxoffdiag = 0.0;
    for (size_t i = 0; i < basis_size_; ++i) {
        for (size_t j = 0; j < basis_size_; ++j) {
            if (i != j) {
                maxoffdiag = std::max(maxoffdiag, std::fabs(S->get(i, j)));
            }
        }
    }
    if (maxoffdiag > orthogonality_threshold_) {
        S->print();
        outfile->Printf("\n  Maximum absolute off-diagonal element of the overlap: %e", maxoffdiag);
        std::string msg =
            "DavidsonLiuSolver::check_orthogonality(): eigenvectors are not orthogonal!";
        throw std::runtime_error(msg);
    }
}

void DavidsonLiuSolver::check_G_hermiticity() {
    double maxnonherm = 0.0;
    for (size_t i = 0; i < basis_size_; ++i) {
        for (size_t j = i + 1; j < basis_size_; ++j) {
            maxnonherm = std::max(maxnonherm, std::fabs(G->get(i, j) - G->get(j, i)));
        }
    }
    if (maxnonherm > nonhermitian_G_threshold_) {
        G->print();
        outfile->Printf("\n  Maximum absolute off-diagonal element of the Hamiltonian: %e",
                        maxnonherm);
        std::string msg =
            "DavidsonLiuSolver::check_G_hermiticity(): the Hamiltonian in not Hermitian";
        throw std::runtime_error(msg);
    } else {
        for (size_t i = 0; i < basis_size_; ++i) {
            for (size_t j = i + 1; j < basis_size_; ++j) {
                auto od = 0.5 * (G->get(i, j) + G->get(j, i));
                G->set(i, j, od);
                G->set(j, i, od);
            }
        }
    }
}

} // namespace forte
