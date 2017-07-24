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

#include "psi4/libciomr/libciomr.h"
#include "psi4/libmints/matrix.h"
#include "psi4/libmints/vector.h"
#include "psi4/libpsio/psio.hpp"
#include "psi4/libqt/qt.h"

#include "forte-def.h"
#include "iterative_solvers.h"
#include "sparse_ci_solver_tmp.h"
//#include "fci/fci_vector.h"

struct PairHash {
    size_t operator()(const std::pair<size_t, size_t>& p) const {
        return (p.first * 1000) + p.second;
    }
};

namespace psi {
namespace forte {

#ifdef _OPENMP
#include <omp.h>
#else
#define omp_get_max_threads() 1
#define omp_get_thread_num() 0
#define omp_get_num_threads() 1
#endif

SigmaVector2::SigmaVector2(const DeterminantMap2& space, WFNOperator2& op,
                           std::shared_ptr<FCIIntegrals> fci_ints)
    : SigmaVector(space.size()), space_(space), fci_ints_(fci_ints), a_list_(op.a_list_),
      b_list_(op.b_list_), aa_list_(op.aa_list_), ab_list_(op.ab_list_), bb_list_(op.bb_list_) {

    stldet_hash<size_t> detmap = space_.wfn_hash();
    diag_.resize(space_.size());
    for (stldet_hash<size_t>::const_iterator it = detmap.begin(), endit = detmap.end(); it != endit;
         ++it) {
        diag_[it->second] = fci_ints_->energy(it->first);
    }
}

void SigmaVector2::add_bad_roots(std::vector<std::vector<std::pair<size_t, double>>>& roots) {
    bad_states_.clear();
    for (int i = 0, max_i = roots.size(); i < max_i; ++i) {
        bad_states_.push_back(roots[i]);
    }
}

void SigmaVector2::get_diagonal(Vector& diag) {
    for (size_t I = 0; I < diag_.size(); ++I) {
        diag.set(I, diag_[I]);
    }
}

void SigmaVector2::compute_sigma(SharedVector sigma, SharedVector b) {
    sigma->zero();

    size_t ncmo = fci_ints_->nmo();

    double* sigma_p = sigma->pointer();
    double* b_p = b->pointer();

    // Compute the overlap with each root
    int nbad = bad_states_.size();
    std::vector<double> overlap(nbad);
    if (nbad != 0) {
        for (int n = 0; n < nbad; ++n) {
            std::vector<std::pair<size_t, double>>& bad_state = bad_states_[n];
            double dprd = 0.0;
            for (size_t det = 0, ndet = bad_state.size(); det < ndet; ++det) {
                dprd += bad_state[det].second * b_p[bad_state[det].first];
            }
            overlap[n] = dprd;
        }
        // outfile->Printf("\n Overlap: %1.6f", overlap[0]);

        for (int n = 0; n < nbad; ++n) {
            std::vector<std::pair<size_t, double>>& bad_state = bad_states_[n];
            size_t ndet = bad_state.size();

#pragma omp parallel for
            for (size_t det = 0; det < ndet; ++det) {
                b_p[bad_state[det].first] -= bad_state[det].second * overlap[n];
            }
        }
    }
    const std::vector<STLDeterminant>& dets = space_.determinants();

#pragma omp parallel
    {
        int num_thread = omp_get_max_threads();
        int tid = omp_get_thread_num();

        // Each thread gets local copy of sigma
        std::vector<double> sigma_t(size_);

        size_t bin_size = size_ / num_thread;
        bin_size += (tid < (size_ % num_thread)) ? 1 : 0;
        size_t start_idx =
            (tid < (size_ % num_thread))
                ? tid * bin_size
                : (size_ % num_thread) * (bin_size + 1) + (tid - (size_ % num_thread)) * bin_size;
        size_t end_idx = start_idx + bin_size;

        for (size_t J = start_idx; J < end_idx; ++J) {
            sigma_t[J] += diag_[J] * b_p[J]; // Make DDOT
        }

        // a singles
        size_t end_a_idx = a_list_.size();
        size_t start_a_idx = 0;
        for (size_t K = start_a_idx, max_K = end_a_idx; K < max_K; ++K) {
            if ((K % num_thread) == tid) {
                for (auto& detJ : a_list_[K]) { // Each gives unique J
                    const size_t J = detJ.first;
                    const size_t p = std::abs(detJ.second) - 1;
                    double sign_p = detJ.second > 0.0 ? 1.0 : -1.0;
                    for (auto& detI : a_list_[K]) {
                        const size_t q = std::abs(detI.second) - 1;
                        if (p != q) {
                            const size_t I = detI.first;
                            double sign_q = detI.second > 0.0 ? 1.0 : -1.0;
                            const double HIJ =
                                fci_ints_->slater_rules_single_alpha_abs(dets[J], p, q) * sign_p *
                                sign_q;
                            sigma_t[I] += HIJ * b_p[J];
                        }
                    }
                }
            }
        }

        // b singles
        size_t end_b_idx = b_list_.size();
        size_t start_b_idx = 0;
        for (size_t K = start_b_idx, max_K = end_b_idx; K < max_K; ++K) {
            // aa singles
            if ((K % num_thread) == tid) {
                for (auto& detJ : b_list_[K]) {
                    const size_t J = detJ.first;
                    const size_t p = std::abs(detJ.second) - 1;
                    double sign_p = detJ.second > 0.0 ? 1.0 : -1.0;
                    for (auto& detI : b_list_[K]) {
                        const size_t q = std::abs(detI.second) - 1;
                        if (p != q) {
                            const size_t I = detI.first;
                            double sign_q = detI.second > 0.0 ? 1.0 : -1.0;
                            const double HIJ =
                                fci_ints_->slater_rules_single_beta_abs(dets[J], p, q) * sign_p *
                                sign_q;
                            sigma_t[I] += HIJ * b_p[J];
                        }
                    }
                }
            }
        }

        // AA doubles
        size_t aa_size = aa_list_.size();
        //      size_t bin_aa_size = aa_size / num_thread;
        //      bin_aa_size += (tid < (aa_size % num_thread)) ? 1 : 0;
        //      size_t start_aa_idx = (tid < (aa_size % num_thread))
        //                             ? tid * bin_aa_size
        //                             : (aa_size % num_thread) * (bin_aa_size + 1) +
        //                                   (tid - (aa_size % num_thread)) * bin_aa_size;
        //      size_t end_aa_idx = start_aa_idx + bin_aa_size;
        for (size_t K = 0, max_K = aa_size; K < max_K; ++K) {
            if ((K % num_thread) == tid) {
                const std::vector<std::tuple<size_t, short, short>>& c_dets = aa_list_[K];
                for (auto& detJ : c_dets) {
                    size_t J = std::get<0>(detJ);
                    short p = std::abs(std::get<1>(detJ)) - 1;
                    short q = std::get<2>(detJ);
                    double sign_p = std::get<1>(detJ) > 0.0 ? 1.0 : -1.0;
                    for (auto& detI : c_dets) {
                        short r = std::abs(std::get<1>(detI)) - 1;
                        short s = std::get<2>(detI);
                        if ((p != r) and (q != s) and (p != s) and (q != r)) {
                            size_t I = std::get<0>(detI);
                            double sign_q = std::get<1>(detI) > 0.0 ? 1.0 : -1.0;
                            double HIJ = sign_p * sign_q * fci_ints_->tei_aa(p, q, r, s);
                            sigma_t[I] += HIJ * b_p[J];
                        }
                    }
                }
            }
        }

        // BB doubles
        for (size_t K = 0, max_K = bb_list_.size(); K < max_K; ++K) {
            const std::vector<std::tuple<size_t, short, short>>& c_dets = bb_list_[K];
            if ((K % num_thread) == tid) {
                for (auto& detJ : c_dets) {
                    size_t J = std::get<0>(detJ);
                    short p = std::abs(std::get<1>(detJ)) - 1;
                    short q = std::get<2>(detJ);
                    double sign_p = std::get<1>(detJ) > 0.0 ? 1.0 : -1.0;
                    for (auto& detI : c_dets) {
                        short r = std::abs(std::get<1>(detI)) - 1;
                        short s = std::get<2>(detI);
                        if ((p != r) and (q != s) and (p != s) and (q != r)) {
                            size_t I = std::get<0>(detI);
                            double sign_q = std::get<1>(detI) > 0.0 ? 1.0 : -1.0;
                            double HIJ = sign_p * sign_q * fci_ints_->tei_bb(p, q, r, s);
                            sigma_t[I] += HIJ * b_p[J];
                        }
                    }
                }
            }
        }
        for (size_t K = 0, max_K = ab_list_.size(); K < max_K; ++K) {
            if ((K % num_thread) == tid) {
                const std::vector<std::tuple<size_t, short, short>>& c_dets = ab_list_[K];
                size_t max_det = c_dets.size();
                for (size_t det = 0; det < max_det; ++det) {
                    auto detJ = c_dets[det];
                    size_t J = std::get<0>(detJ);
                    short p = std::abs(std::get<1>(detJ)) - 1;
                    short q = std::get<2>(detJ);
                    double sign_p = std::get<1>(detJ) > 0.0 ? 1.0 : -1.0;
                    for (auto& detI : c_dets) {
                        short r = std::abs(std::get<1>(detI)) - 1;
                        short s = std::get<2>(detI);
                        if ((p != r) and (q != s)) {
                            size_t I = std::get<0>(detI);
                            double sign_q = std::get<1>(detI) > 0.0 ? 1.0 : -1.0;
                            double HIJ = sign_p * sign_q * fci_ints_->tei_ab(p, q, r, s);
                            sigma_t[I] += HIJ * b_p[J];
                        }
                    }
                }
            }
        }

        //        #pragma omp critical
        //        {
        for (size_t I = 0; I < size_; ++I) {
#pragma omp atomic update
            sigma_p[I] += sigma_t[I];
        }
        //        }
    }
}

SparseCISolver2::SparseCISolver2(std::shared_ptr<FCIIntegrals> fci_ints) { fci_ints_ = fci_ints; }

void SparseCISolver2::set_spin_project(bool value) { spin_project_ = value; }

void SparseCISolver2::set_e_convergence(double value) { e_convergence_ = value; }

void SparseCISolver2::set_maxiter_davidson(int value) { maxiter_davidson_ = value; }

void SparseCISolver2::set_spin_project_full(bool value) { spin_project_full_ = value; }

void SparseCISolver2::set_sigma_method(std::string value) { sigma_method_ = value; }

void SparseCISolver2::diagonalize_hamiltonian_map(const DeterminantMap2& space, WFNOperator2& op,
                                                  SharedVector& evals, SharedMatrix& evecs,
                                                  int nroot, int multiplicity,
                                                  DiagonalizationMethod diag_method) {
    if (space.size() <= 200 or diag_method == Full) {
        const std::vector<STLDeterminant> dets = space.determinants();
        diagonalize_full(dets, evals, evecs, nroot, multiplicity);
    } else {
        diagonalize_dl(space, op, evals, evecs, nroot, multiplicity);
    }
}

void SparseCISolver2::diagonalize_dl(const DeterminantMap2& space, WFNOperator2& op,
                                     SharedVector& evals, SharedMatrix& evecs, int nroot,
                                     int multiplicity) {
    if (print_details_) {
        outfile->Printf("\n\n  Davidson-Liu solver algorithm");
        outfile->Printf("\n  Using %s sigma builder", sigma_method_.c_str());
    }
    size_t dim_space = space.size();
    evecs.reset(new Matrix("U", dim_space, nroot));
    evals.reset(new Vector("e", nroot));
    SigmaVector* sigma_vector = 0;

    if (sigma_vec_ != nullptr) {
        sigma_vec_->add_bad_roots(bad_states_);
        davidson_liu_solver_map(space, sigma_vec_, evals, evecs, nroot, multiplicity);
        return;
    }

    SigmaVector2 svw(space, op, fci_ints_);
    sigma_vector = &svw;
    sigma_vector->add_bad_roots(bad_states_);
    davidson_liu_solver_map(space, sigma_vector, evals, evecs, nroot, multiplicity);
}

void SparseCISolver2::diagonalize_full(const std::vector<STLDeterminant>& space,
                                       SharedVector& evals, SharedMatrix& evecs, int nroot,
                                       int multiplicity) {

    size_t dim_space = space.size();
    evecs.reset(new Matrix("U", dim_space, nroot));
    evals.reset(new Vector("e", nroot));

    if (spin_project_full_) {
        // Diagonalize S^2 matrix
        Matrix S2("S^2", dim_space, dim_space);
        for (size_t I = 0; I < dim_space; ++I) {
            for (size_t J = 0; J < dim_space; ++J) {
                double S2IJ = space[I].spin2(space[J]);
                S2.set(I, J, S2IJ);
            }
        }
        Vector S2vals("S^2 Eigen Values", dim_space);
        Matrix S2vecs("S^2 Eigen Vectors", dim_space, dim_space);
        S2.diagonalize(S2vecs, S2vals);

        // Map multiplcity to index
        double Stollerance = 1.0e-4;
        std::map<int, std::vector<int>> multi_list;
        for (size_t i = 0; i < dim_space; ++i) {
            double multi = std::sqrt(1.0 + 4.0 * S2vals.get(i));
            double error = std::round(multi) - multi;
            if (std::fabs(error) < Stollerance) {
                int multi_round = std::round(multi);
                multi_list[multi_round].push_back(i);
            } else {
                if (print_details_) {
                    outfile->Printf("\n  Spin multiplicity of root %zu not close to integer (%.4f)",
                                    i, multi);
                }
            }
        }

        // Test S^2 eigen values
        int nfound = 0;
        for (const auto& mi : multi_list) {
            int multi = mi.first;
            size_t multi_size = mi.second.size();
            std::string mark = " *";
            if (multi == multiplicity) {
                nfound = static_cast<int>(multi_size);
            } else {
                mark = "";
            }
            if (print_details_) {
                outfile->Printf("\n  Found %zu roots with 2S+1 = %d%s", multi_size, multi,
                                mark.c_str());
            }
        }
        if (nfound < nroot) {
            outfile->Printf("\n  Error: ask for %d roots with 2S+1 = %d but only "
                            "%d were found!",
                            nroot, multiplicity, nfound);
            throw PSIEXCEPTION("Too many roots of interest in full diag. of sparce_ci_solver.");
        }

        // Select sub eigen vectors of S^2 with correct multiplicity
        SharedMatrix S2vecs_sub(new Matrix("Spin Selected S^2 Eigen Vectors", dim_space, nfound));
        for (size_t i = 0; i < nfound; ++i) {
            SharedVector vec = S2vecs.get_column(0, multi_list[multiplicity][i]);
            S2vecs_sub->set_column(0, i, vec);
        }

        // Build spin selected Hamiltonian
        SharedMatrix H = build_full_hamiltonian(space);
        SharedMatrix Hss = Matrix::triplet(S2vecs_sub, H, S2vecs_sub, true, false, false);
        Hss->set_name("Hss");

        // Obtain spin selected eigen values and vectors
        SharedVector Hss_vals(new Vector("Hss Eigen Values", nfound));
        SharedMatrix Hss_vecs(new Matrix("Hss Eigen Vectors", nfound, nfound));
        Hss->diagonalize(Hss_vecs, Hss_vals);

        // Project Hss_vecs back to original manifold
        SharedMatrix H_vecs = Matrix::doublet(S2vecs_sub, Hss_vecs);
        H_vecs->set_name("H Eigen Vectors");

        // Fill in results
        for (int i = 0; i < nroot; ++i) {
            evals->set(i, Hss_vals->get(i));
            evecs->set_column(0, i, H_vecs->get_column(0, i));
        }
    } else {
        // Find all the eigenvalues and eigenvectors of the Hamiltonian
        SharedMatrix H = build_full_hamiltonian(space);

        evecs.reset(new Matrix("U", dim_space, dim_space));
        evals.reset(new Vector("e", dim_space));

        // Diagonalize H
        H->diagonalize(evecs, evals);
    }
}

SharedMatrix SparseCISolver2::build_full_hamiltonian(const std::vector<STLDeterminant>& space) {
    // Build the H matrix
    size_t dim_space = space.size();
    SharedMatrix H(new Matrix("H", dim_space, dim_space));
    // If you are using DiskDF, Kevin found that openmp does not like this!
    int threads = 0;
    if (fci_ints_->get_integral_type() == DiskDF) {
        threads = 1;
    } else {
        threads = omp_get_max_threads();
    }
#pragma omp parallel for schedule(dynamic) num_threads(threads)
    for (size_t I = 0; I < dim_space; ++I) {
        const STLDeterminant& detI = space[I];
        for (size_t J = I; J < dim_space; ++J) {
            const STLDeterminant& detJ = space[J];
            double HIJ = fci_ints_->slater_rules(detI, detJ);
            H->set(I, J, HIJ);
            H->set(J, I, HIJ);
        }
    }

    if (root_project_) {
        // Form the projection matrix
        for (int n = 0, max_n = bad_states_.size(); n < max_n; ++n) {
            SharedMatrix P(new Matrix("P", dim_space, dim_space));
            P->identity();
            std::vector<std::pair<size_t, double>>& bad_state = bad_states_[n];
            for (size_t det1 = 0, ndet = bad_state.size(); det1 < ndet; ++det1) {
                for (size_t det2 = 0; det2 < ndet; ++det2) {
                    size_t& I = bad_state[det1].first;
                    size_t& J = bad_state[det2].first;
                    double& el1 = bad_state[det1].second;
                    double& el2 = bad_state[det2].second;
                    P->set(I, J, P->get(I, J) - el1 * el2);
                }
            }
            H->transform(P);
        }
    }

    return H;
}

std::vector<std::pair<double, std::vector<std::pair<size_t, double>>>>
SparseCISolver2::initial_guess_map(const DeterminantMap2& space, int nroot, int multiplicity) {
    size_t ndets = space.size();
    size_t nguess = std::min(static_cast<size_t>(nroot) * dl_guess_, ndets);
    std::vector<std::pair<double, std::vector<std::pair<size_t, double>>>> guess(nguess);

    // Find the ntrial lowest diagonals
    std::vector<std::pair<STLDeterminant, size_t>> guess_dets_pos;
    std::vector<std::pair<double, STLDeterminant>> smallest;
    const stldet_hash<size_t>& detmap = space.wfn_hash();

    for (stldet_hash<size_t>::const_iterator it = detmap.begin(), endit = detmap.end(); it != endit;
         ++it) {
        smallest.push_back(std::make_pair(fci_ints_->energy(it->first), it->first));
    }
    std::sort(smallest.begin(), smallest.end());

    std::vector<STLDeterminant> guess_det;
    for (size_t i = 0; i < nguess; i++) {
        STLDeterminant detI = smallest[i].second;
        guess_dets_pos.push_back(
            std::make_pair(detI, space.get_idx(detI))); // store a det and its position
        guess_det.push_back(detI);
    }

    if (spin_project_) {
        STLDeterminant det = guess_det[0];
        det.enforce_spin_completeness(guess_det);
        if (guess_det.size() > nguess) {
            size_t nnew_dets = guess_det.size() - nguess;
            if (print_details_)
                outfile->Printf("\n  Initial guess space is incomplete.\n  "
                                "Trying to add %d determinant(s).",
                                nnew_dets);
            int nfound = 0;
            for (size_t i = 0; i < nnew_dets; ++i) {
                for (size_t j = nguess; j < ndets; ++j) {
                    STLDeterminant detJ = smallest[j].second;
                    if (detJ == guess_det[nguess + i]) {
                        guess_dets_pos.push_back(std::make_pair(
                            detJ, space.get_idx(detJ))); // store a det and its position
                        nfound++;
                        break;
                    }
                }
            }
            if (print_details_)
                outfile->Printf("  %d determinant(s) added.", nfound);
        }
        nguess = guess_dets_pos.size();
    }

    // Form the S^2 operator matrix and diagonalize it
    Matrix S2("S^2", nguess, nguess);
    for (size_t I = 0; I < nguess; I++) {
        for (size_t J = I; J < nguess; J++) {
            const STLDeterminant& detI = guess_dets_pos[I].first;
            const STLDeterminant& detJ = guess_dets_pos[J].first;
            double S2IJ = detI.spin2(detJ);
            S2.set(I, J, S2IJ);
            S2.set(J, I, S2IJ);
        }
    }
    Matrix S2evecs("S^2", nguess, nguess);
    Vector S2evals("S^2", nguess);
    S2.diagonalize(S2evecs, S2evals);

    // Form the Hamiltonian
    Matrix H("H", nguess, nguess);
    for (size_t I = 0; I < nguess; I++) {
        for (size_t J = I; J < nguess; J++) {
            const STLDeterminant& detI = guess_dets_pos[I].first;
            const STLDeterminant& detJ = guess_dets_pos[J].first;
            double HIJ = fci_ints_->slater_rules(detI, detJ);
            H.set(I, J, HIJ);
            H.set(J, I, HIJ);
        }
    }
    // H.print();
    // Project H onto the spin-adapted subspace
    H.transform(S2evecs);

    // Find groups of solutions with same spin
    double Stollerance = 1.0e-6;
    std::map<int, std::vector<int>> mult_list;
    for (size_t i = 0; i < nguess; ++i) {
        double mult = std::sqrt(1.0 + 4.0 * S2evals.get(i)); // 2S + 1 = Sqrt(1 + 4 S (S + 1))
        int mult_int = std::round(mult);
        double error = mult - static_cast<double>(mult_int);
        if (std::fabs(error) < Stollerance) {
            mult_list[mult_int].push_back(i);
        } else if (print_details_) {
            outfile->Printf("\n  Found a guess vector with spin not close to "
                            "integer value (%f)",
                            mult);
        }
    }
    if (mult_list[multiplicity].size() < static_cast<size_t>(nroot)) {
        size_t nfound = mult_list[multiplicity].size();
        outfile->Printf("\n  Error: %d guess vectors with 2S+1 = %d but only "
                        "%d were found!",
                        nguess, multiplicity, nfound);
        if (nfound == 0) {
            exit(1);
        }
    }

    std::vector<int> mult_vals;
    for (auto kv : mult_list) {
        mult_vals.push_back(kv.first);
    }
    std::sort(mult_vals.begin(), mult_vals.end());

    for (int m : mult_vals) {
        std::vector<int>& mult_list_s = mult_list[m];
        int nspin_states = mult_list_s.size();
        if (print_details_)
            outfile->Printf("\n  Initial guess found %d solutions with 2S+1 = %d %c", nspin_states,
                            m, m == multiplicity ? '*' : ' ');
        // Extract the spin manifold
        Matrix HS2("HS2", nspin_states, nspin_states);
        Vector HS2evals("HS2", nspin_states);
        Matrix HS2evecs("HS2", nspin_states, nspin_states);
        for (int I = 0; I < nspin_states; I++) {
            for (int J = 0; J < nspin_states; J++) {
                HS2.set(I, J, H.get(mult_list_s[I], mult_list_s[J]));
            }
        }
        HS2.diagonalize(HS2evecs, HS2evals);

        // Project the spin-adapted solution onto the full manifold
        for (int r = 0; r < nspin_states; ++r) {
            std::vector<std::pair<size_t, double>> det_C;
            for (size_t I = 0; I < nguess; I++) {
                double CIr = 0.0;
                for (int J = 0; J < nspin_states; ++J) {
                    CIr += S2evecs.get(I, mult_list_s[J]) * HS2evecs(J, r);
                }
                det_C.push_back(std::make_pair(guess_dets_pos[I].second, CIr));
            }
            guess.push_back(std::make_pair(m, det_C));
        }
    }

    return guess;
}

void SparseCISolver2::add_bad_states(std::vector<std::vector<std::pair<size_t, double>>>& roots) {
    bad_states_.clear();
    for (int i = 0, max_i = roots.size(); i < max_i; ++i) {
        bad_states_.push_back(roots[i]);
    }
}

void SparseCISolver2::set_root_project(bool value) { root_project_ = value; }

void SparseCISolver2::manual_guess(bool value) { set_guess_ = value; }

void SparseCISolver2::set_initial_guess(std::vector<std::pair<size_t, double>>& guess) {
    set_guess_ = true;
    guess_.clear();

    for (size_t I = 0, max_I = guess.size(); I < max_I; ++I) {
        guess_.push_back(guess[I]);
    }
}

void SparseCISolver2::set_num_vecs(size_t value) { nvec_ = value; }

bool SparseCISolver2::davidson_liu_solver_map(const DeterminantMap2& space,
                                              SigmaVector* sigma_vector, SharedVector Eigenvalues,
                                              SharedMatrix Eigenvectors, int nroot,
                                              int multiplicity) {
    //    print_details_ = true;
    Timer dl;
    size_t fci_size = sigma_vector->size();
    DavidsonLiuSolver dls(fci_size, nroot);
    dls.set_e_convergence(e_convergence_);
    dls.set_print_level(0);

    // allocate vectors
    SharedVector b(new Vector("b", fci_size));
    SharedVector sigma(new Vector("sigma", fci_size));

    // get and pass diagonal
    sigma_vector->get_diagonal(*sigma);
    dls.startup(sigma);

    std::vector<std::vector<std::pair<size_t, double>>> bad_roots;
    size_t guess_size = std::min(nvec_, dls.collapse_size());

    auto guess = initial_guess_map(space, nroot, multiplicity);
    if (!set_guess_) {
        std::vector<int> guess_list;
        for (size_t g = 0; g < guess.size(); ++g) {
            if (guess[g].first == multiplicity)
                guess_list.push_back(g);
        }

        // number of guess to be used
        size_t nguess = std::min(guess_list.size(), guess_size);

        if (nguess == 0) {
            throw PSIEXCEPTION("\n\n  Found zero FCI guesses with the "
                               "requested multiplicity.\n\n");
        }

        for (size_t n = 0; n < nguess; ++n) {
            b->zero();
            for (auto& guess_vec_info : guess[guess_list[n]].second) {
                b->set(guess_vec_info.first, guess_vec_info.second);
            }
            if (print_details_)
                outfile->Printf("\n  Adding guess %d (multiplicity = %f)", n,
                                guess[guess_list[n]].first);

            dls.add_guess(b);
        }
    }

    // Prepare a list of bad roots to project out and pass them to the solver
    for (auto& g : guess) {
        if (g.first != multiplicity)
            bad_roots.push_back(g.second);
    }
    dls.set_project_out(bad_roots);

    if (set_guess_) {
        // Use previous solution as guess
        b->zero();
        for (size_t I = 0, max_I = guess_.size(); I < max_I; ++I) {
            b->set(guess_[I].first, guess_[I].second);
        }
        double norm = sqrt(1.0 / b->norm());
        b->scale(norm);
        dls.add_guess(b);
    }

    SolverStatus converged = SolverStatus::NotConverged;

    if (print_details_) {
        outfile->Printf("\n\n  ==> Diagonalizing Hamiltonian <==\n");
        outfile->Printf("\n  ----------------------------------------");
        outfile->Printf("\n    Iter.      Avg. Energy       Delta_E");
        outfile->Printf("\n  ----------------------------------------");
    }

    double old_avg_energy = 0.0;
    int real_cycle = 1;

    //    maxiter_davidson_ = 2;
    //    b->print();
    for (int cycle = 0; cycle < maxiter_davidson_; ++cycle) {
        bool add_sigma = true;
        do {
            dls.get_b(b);
            sigma_vector->compute_sigma(sigma, b);

            add_sigma = dls.add_sigma(sigma);
        } while (add_sigma);

        converged = dls.update();

        if (converged != SolverStatus::Collapse) {
            double avg_energy = 0.0;
            for (int r = 0; r < nroot; ++r)
                avg_energy += dls.eigenvalues()->get(r);
            avg_energy /= static_cast<double>(nroot);
            if (print_details_) {
                outfile->Printf("\n    %3d  %20.12f  %+.3e", real_cycle, avg_energy,
                                avg_energy - old_avg_energy);
            }
            old_avg_energy = avg_energy;
            real_cycle++;
        }

        if (converged == SolverStatus::Converged)
            break;
    }

    if (print_details_) {
        outfile->Printf("\n  ----------------------------------------");
        if (converged == SolverStatus::Converged) {
            outfile->Printf("\n  The Davidson-Liu algorithm converged in %d iterations.",
                            real_cycle);
        }
    }

    if (converged == SolverStatus::NotConverged) {
        outfile->Printf("\n  FCI did not converge!");
        exit(1);
    }

    //    dls.get_results();
    SharedVector evals = dls.eigenvalues();
    SharedMatrix evecs = dls.eigenvectors();
    for (int r = 0; r < nroot; ++r) {
        Eigenvalues->set(r, evals->get(r));
        for (size_t I = 0; I < fci_size; ++I) {
            Eigenvectors->set(I, r, evecs->get(r, I));
        }
    }
    if (print_details_) {
        outfile->Printf("\n  Davidson-Liu procedure took  %1.6f s", dl.get());
    }

    return true;
}
}
}
