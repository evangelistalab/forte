/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2020 by its authors (see COPYING, COPYING.LESSER, AUTHORS).
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

//#include <algorithm>
//#include <numeric>

//#include "helpers/string_algorithms.h"

#include "sparse_ci/sparse_operations.h"

namespace forte {

SparseHamiltonian::SparseHamiltonian(std::shared_ptr<ActiveSpaceIntegrals> as_ints)
    : as_ints_(as_ints) {}

StateVector SparseHamiltonian::compute(const StateVector& state, double screen_thresh) {
    std::vector<Determinant> new_dets;
    std::vector<double> state_c(state.size());

    // find new determinants
    for (const auto& det_c : state) {
        const Determinant& det = det_c.first;
        const double c = det_c.second;
        // if we already have precomputed this det, put the coefficient where it belongs
        if (state_hash_.has_det(det)) {
            state_c[state_hash_.get_idx(det)] = c;
        } else {
            size_t idx = state_hash_.add(det);
            state_c[idx] = c;
            new_dets.push_back(det);
        }
    }

    // compute the new couplings
    compute_new_couplings(new_dets, screen_thresh);

    // compute sigma
    return compute_sigma(state_c, screen_thresh);
}

void SparseHamiltonian::compute_new_couplings(const std::vector<Determinant>& new_dets,
                                              double screen_thresh) {
    local_timer t;

    size_t nmo = as_ints_->nmo();

    auto symm = as_ints_->active_mo_symmetry();

    Determinant new_det;

    for (const auto& det : new_dets) {
        size_t det_idx = state_hash_.get_idx(det);

        // diagonal coupling
        double E_0 = as_ints_->nuclear_repulsion_energy() + as_ints_->scalar_energy();

        // index of det in the sigma hash
        size_t det_sigma_idx = sigma_hash_.add(det);

        couplings_.emplace_back(det_idx, det_sigma_idx, E_0 + as_ints_->slater_rules(det, det));

        std::vector<int> aocc = det.get_alfa_occ(nmo);
        std::vector<int> bocc = det.get_beta_occ(nmo);
        std::vector<int> avir = det.get_alfa_vir(nmo);
        std::vector<int> bvir = det.get_beta_vir(nmo);

        size_t noalpha = aocc.size();
        size_t nobeta = bocc.size();
        size_t nvalpha = avir.size();
        size_t nvbeta = bvir.size();

        // aa singles
        for (size_t i : aocc) {
            for (size_t a : avir) {
                if ((symm[i] ^ symm[a]) == 0) {
                    double DHIJ = as_ints_->slater_rules_single_alpha(det, i, a);
                    if (std::abs(DHIJ) >= screen_thresh) {
                        new_det = det;
                        new_det.set_alfa_bit(i, false);
                        new_det.set_alfa_bit(a, true);
                        size_t new_det_sigma_idx = sigma_hash_.add(new_det);
                        couplings_.emplace_back(det_idx, new_det_sigma_idx, DHIJ);
                    }
                }
            }
        }
        // bb singles
        for (size_t i : bocc) {
            for (size_t a : bvir) {
                if ((symm[i] ^ symm[a]) == 0) {
                    double DHIJ = as_ints_->slater_rules_single_beta(det, i, a);
                    if (std::abs(DHIJ) >= screen_thresh) {
                        new_det = det;
                        new_det.set_beta_bit(i, false);
                        new_det.set_beta_bit(a, true);
                        size_t new_det_sigma_idx = sigma_hash_.add(new_det);
                        couplings_.emplace_back(det_idx, new_det_sigma_idx, DHIJ);
                    }
                }
            }
        }
        // Generate aa excitations
        for (size_t ii = 0; ii < noalpha; ++ii) {
            size_t i = aocc[ii];
            for (size_t jj = ii + 1; jj < noalpha; ++jj) {
                size_t j = aocc[jj];
                for (size_t aa = 0; aa < nvalpha; ++aa) {
                    size_t a = avir[aa];
                    for (size_t bb = aa + 1; bb < nvalpha; ++bb) {
                        size_t b = avir[bb];
                        if ((symm[i] ^ symm[j] ^ symm[a] ^ symm[b]) == 0) {
                            double DHIJ = as_ints_->tei_aa(i, j, a, b);
                            if (std::abs(DHIJ) >= screen_thresh) {
                                new_det = det;
                                DHIJ *= new_det.double_excitation_aa(i, j, a, b);
                                size_t new_det_sigma_idx = sigma_hash_.add(new_det);
                                couplings_.emplace_back(det_idx, new_det_sigma_idx, DHIJ);
                            }
                        }
                    }
                }
            }
        }
        // Generate ab excitations
        for (size_t i : aocc) {
            for (size_t j : bocc) {
                for (size_t a : avir) {
                    for (size_t b : bvir) {
                        if ((symm[i] ^ symm[j] ^ symm[a] ^ symm[b]) == 0) {
                            double DHIJ = as_ints_->tei_ab(i, j, a, b);
                            if (std::abs(DHIJ) >= screen_thresh) {
                                new_det = det;
                                DHIJ *= new_det.double_excitation_ab(i, j, a, b);
                                size_t new_det_sigma_idx = sigma_hash_.add(new_det);
                                couplings_.emplace_back(det_idx, new_det_sigma_idx, DHIJ);
                            }
                        }
                    }
                }
            }
        }
        // Generate bb excitations
        for (size_t ii = 0; ii < nobeta; ++ii) {
            size_t i = bocc[ii];
            for (size_t jj = ii + 1; jj < nobeta; ++jj) {
                size_t j = bocc[jj];
                for (size_t aa = 0; aa < nvbeta; ++aa) {
                    size_t a = bvir[aa];
                    for (size_t bb = aa + 1; bb < nvbeta; ++bb) {
                        size_t b = bvir[bb];
                        if ((symm[i] ^ symm[j] ^ symm[a] ^ symm[b]) == 0) {
                            double DHIJ = as_ints_->tei_bb(i, j, a, b);
                            if (std::abs(DHIJ) >= screen_thresh) {
                                new_det = det;
                                DHIJ *= new_det.double_excitation_bb(i, j, a, b);
                                size_t new_det_sigma_idx = sigma_hash_.add(new_det);
                                couplings_.emplace_back(det_idx, new_det_sigma_idx, DHIJ);
                            }
                        }
                    }
                }
            }
        }
    }
    couplings_time_ += t.get();
    time_ += t.get();
}

StateVector SparseHamiltonian::compute_sigma(const std::vector<double>& state_c,
                                             double screen_thresh) {
    local_timer t;
    // initialize a state object
    std::vector<double> sigma_c(sigma_hash_.size(), 0.0);
    StateVector sigma;
    for (const auto& coupling : couplings_) {
        const size_t det_idx = std::get<0>(coupling);
        const size_t new_det_idx = std::get<1>(coupling);
        const double h = std::get<2>(coupling);
        const double c = state_c[det_idx];
        if (std::fabs(c * h) > screen_thresh) {
            sigma_c[new_det_idx] += c * h;
        }
    }
    for (size_t n = 0, maxn = sigma_hash_.size(); n < maxn; n++) {
        sigma[sigma_hash_.get_det(n)] = sigma_c[n];
    }

    sigma_time_ += t.get();
    time_ += t.get();
    return sigma;
}

std::map<std::string, double> SparseHamiltonian::time() const {
    std::map<std::string, double> t;
    t["time"] = time_;
    t["couplings_time"] = couplings_time_;
    t["sigma_time"] = sigma_time_;
    t["on_the_fly_time"] = on_the_fly_time_;
    return t;
}

StateVector SparseHamiltonian::compute_on_the_fly(const StateVector& state, double screen_thresh) {
    local_timer t;

    // initialize a state object
    StateVector sigma;

    size_t nmo = as_ints_->nmo();

    auto symm = as_ints_->active_mo_symmetry();

    Determinant new_det;

    for (const auto& det_c : state) {
        const Determinant& det = det_c.first;
        const double c = det_c.second;

        std::vector<int> aocc = det.get_alfa_occ(nmo);
        std::vector<int> bocc = det.get_beta_occ(nmo);
        std::vector<int> avir = det.get_alfa_vir(nmo);
        std::vector<int> bvir = det.get_beta_vir(nmo);

        size_t noalpha = aocc.size();
        size_t nobeta = bocc.size();
        size_t nvalpha = avir.size();
        size_t nvbeta = bvir.size();

        double E_0 = as_ints_->nuclear_repulsion_energy() + as_ints_->scalar_energy();

        sigma[det] += (E_0 + as_ints_->slater_rules(det, det)) * c;
        // aa singles
        for (size_t i : aocc) {
            for (size_t a : avir) {
                if ((symm[i] ^ symm[a]) == 0) {
                    double DHIJ = as_ints_->slater_rules_single_alpha(det, i, a);
                    if (std::abs(DHIJ * c) >= screen_thresh) {
                        new_det = det;
                        new_det.set_alfa_bit(i, false);
                        new_det.set_alfa_bit(a, true);
                        sigma[new_det] += DHIJ * c;
                    }
                }
            }
        }
        // bb singles
        for (size_t i : bocc) {
            for (size_t a : bvir) {
                if ((symm[i] ^ symm[a]) == 0) {
                    double DHIJ = as_ints_->slater_rules_single_beta(det, i, a);
                    if (std::abs(DHIJ * c) >= screen_thresh) {
                        new_det = det;
                        new_det.set_beta_bit(i, false);
                        new_det.set_beta_bit(a, true);
                        sigma[new_det] += DHIJ * c;
                    }
                }
            }
        }
        // Generate aa excitations
        for (size_t ii = 0; ii < noalpha; ++ii) {
            size_t i = aocc[ii];
            for (size_t jj = ii + 1; jj < noalpha; ++jj) {
                size_t j = aocc[jj];
                for (size_t aa = 0; aa < nvalpha; ++aa) {
                    size_t a = avir[aa];
                    for (size_t bb = aa + 1; bb < nvalpha; ++bb) {
                        size_t b = avir[bb];
                        if ((symm[i] ^ symm[j] ^ symm[a] ^ symm[b]) == 0) {
                            double DHIJ = as_ints_->tei_aa(i, j, a, b);
                            if (std::abs(DHIJ * c) >= screen_thresh) {
                                new_det = det;
                                DHIJ *= new_det.double_excitation_aa(i, j, a, b);
                                sigma[new_det] += DHIJ * c;
                            }
                        }
                    }
                }
            }
        }
        // Generate ab excitations
        for (size_t i : aocc) {
            for (size_t j : bocc) {
                for (size_t a : avir) {
                    for (size_t b : bvir) {
                        if ((symm[i] ^ symm[j] ^ symm[a] ^ symm[b]) == 0) {
                            double DHIJ = as_ints_->tei_ab(i, j, a, b);
                            if (std::abs(DHIJ * c) >= screen_thresh) {
                                new_det = det;
                                DHIJ *= new_det.double_excitation_ab(i, j, a, b);
                                sigma[new_det] += DHIJ * c;
                            }
                        }
                    }
                }
            }
        }
        // Generate bb excitations
        for (size_t ii = 0; ii < nobeta; ++ii) {
            size_t i = bocc[ii];
            for (size_t jj = ii + 1; jj < nobeta; ++jj) {
                size_t j = bocc[jj];
                for (size_t aa = 0; aa < nvbeta; ++aa) {
                    size_t a = bvir[aa];
                    for (size_t bb = aa + 1; bb < nvbeta; ++bb) {
                        size_t b = bvir[bb];
                        if ((symm[i] ^ symm[j] ^ symm[a] ^ symm[b]) == 0) {
                            double DHIJ = as_ints_->tei_bb(i, j, a, b);
                            if (std::abs(DHIJ * c) >= screen_thresh) {
                                new_det = det;
                                DHIJ *= new_det.double_excitation_bb(i, j, a, b);
                                sigma[new_det] += DHIJ * c;
                            }
                        }
                    }
                }
            }
        }
    }
    on_the_fly_time_ += t.get();
    return sigma;
}

SparseFactExp::SparseFactExp() {}

StateVector SparseFactExp::compute(SparseOperator& sop, const StateVector& state, bool inverse,
                                   double screen_thresh) {
    for (const auto& det_c : state) {
        const Determinant& det = det_c.first;
        const double c = det_c.second;
        exp_hash_.add(det);
    }

    if (inverse and (not initialized_inverse_)) {
        // compute the couplings
        compute_couplings(sop, state, inverse, screen_thresh);
        initialized_inverse_ = true;
    }
    if (not inverse and (not initialized_)) {
        // compute the couplings
        compute_couplings(sop, state, inverse, screen_thresh);
        initialized_ = true;
    }
    return compute_exp_size_t(sop, state, inverse, screen_thresh);
}

void SparseFactExp::compute_couplings(const SparseOperator& sop, const StateVector& state0,
                                      bool inverse, const double screen_thresh) {
    local_timer t;
    const auto& op_list = sop.op_list();

    // initialize a state object
    StateVector state(state0);
    StateVector new_terms;

    // loop over all operators
    for (size_t m = 0, nterms = sop.nterms(); m < nterms; m++) {
        size_t n = inverse ? nterms - m - 1 : m;

        std::vector<std::tuple<Determinant, Determinant, double>> d_couplings;
        std::vector<std::tuple<size_t, size_t, double>> d_couplings2;

        // zero the new terms
        new_terms.clear();

        const SQOperator& sqop = op_list[n];
        const Determinant ucre = sqop.cre() - sqop.ann();
        const Determinant uann = sqop.ann() - sqop.cre();
        const double sign = inverse ? -1.0 : 1.0;
        Determinant new_d;
        // loop over all determinants
        for (const auto& det_c : state) {
            const Determinant& d = det_c.first;

            // test if we can apply this operator to this determinant
            if (d.fast_a_and_b_equal_b(sqop.ann()) and d.fast_a_and_b_eq_zero(ucre)) {
                new_d = d;
                double f = sign * apply_op(new_d, sqop.cre(), sqop.ann());
                d_couplings.emplace_back(d, new_d, f);

                size_t d_idx = exp_hash_.add(d);
                size_t new_d_idx = exp_hash_.add(new_d);
                d_couplings2.emplace_back(d_idx, new_d_idx, f);

                new_terms[new_d] += 1.0;
            } else if (d.fast_a_and_b_equal_b(sqop.cre()) and d.fast_a_and_b_eq_zero(uann)) {
                new_d = d;
                double f = -sign * apply_op(new_d, sqop.ann(), sqop.cre());
                d_couplings.emplace_back(d, new_d, f);

                size_t d_idx = exp_hash_.add(d);
                size_t new_d_idx = exp_hash_.add(new_d);
                d_couplings2.emplace_back(d_idx, new_d_idx, f);

                new_terms[new_d] += 1.0;
            }
        }
        for (const auto& d_c : new_terms) {
            state[d_c.first] = 1.0;
        }
        if (inverse) {
            inverse_couplings_.push_back(d_couplings);
            inverse_couplings2_.push_back(d_couplings2);
        } else {
            couplings_.push_back(d_couplings);
            couplings2_.push_back(d_couplings2);
        }
    }
    time_ += t.get();
    couplings_time_ += t.get();
}

StateVector SparseFactExp::compute_exp(const SparseOperator& sop, const StateVector& state0,
                                       bool inverse, const double screen_thresh) {
    local_timer t;
    const auto& op_list = sop.op_list();

    // initialize a state object
    StateVector state(state0);
    StateVector new_terms;

    // loop over all operators
    for (size_t m = 0, nterms = sop.nterms(); m < nterms; m++) {
        size_t n = inverse ? nterms - m - 1 : m;

        double amp = sop.get_term(n).factor();

        const std::vector<std::tuple<Determinant, Determinant, double>>& d_couplings =
            inverse ? inverse_couplings_[m] : couplings_[m];

        // zero the new terms
        new_terms.clear();

        for (const auto& coupling : d_couplings) {
            const Determinant& d = std::get<0>(coupling);
            const Determinant& new_d = std::get<1>(coupling);
            const double f = amp * std::get<2>(coupling);
            const double c = state[d];
            new_terms[d] += c * (std::cos(f) - 1.0);
            new_terms[new_d] += c * std::sin(f);
        }

        for (const auto& d_c : new_terms) {
            state[d_c.first] += d_c.second;
        }
    }

    time_ += t.get();
    exp_time_ += t.get();
    return state;
}

StateVector SparseFactExp::compute_exp_size_t(const SparseOperator& sop, const StateVector& state0,
                                              bool inverse, const double screen_thresh) {
    local_timer t;
    const auto& op_list = sop.op_list();

    // create and fill in the state vector
    std::vector<double> state_c(exp_hash_.size(), 0.0);

    // temporary space to store new elements
    std::vector<std::pair<size_t, double>> new_terms2(10000);

    for (const auto& det_c : state0) {
        const Determinant& d = det_c.first;
        double c = det_c.second;
        size_t d_idx = exp_hash_.get_idx(d);
        state_c[d_idx] = c;
    }

    // loop over all operators
    for (size_t m = 0, nterms = sop.nterms(); m < nterms; m++) {
        size_t n = inverse ? nterms - m - 1 : m;

        double amp = sop.get_term(n).factor();

        const std::vector<std::tuple<size_t, size_t, double>>& d_couplings2 =
            inverse ? inverse_couplings2_[m] : couplings2_[m];

        // zero the new terms
        size_t k = 0;
        const size_t vec_size = new_terms2.size();
        for (const auto& coupling : d_couplings2) {
            const size_t d_idx = std::get<0>(coupling);
            const size_t new_d_idx = std::get<1>(coupling);
            const double f = amp * std::get<2>(coupling);
            const double c = state_c[d_idx];
            if (k < vec_size) {
                new_terms2[k] = std::make_pair(d_idx, c * (std::cos(f) - 1.0));
                new_terms2[k + 1] = std::make_pair(new_d_idx, c * std::sin(f));
            } else {
                new_terms2.push_back(std::make_pair(d_idx, c * (std::cos(f) - 1.0)));
                new_terms2.push_back(std::make_pair(new_d_idx, c * std::sin(f)));
            }
            k += 2;
        }
        for (size_t j = 0; j < k; j++) {
            state_c[new_terms2[j].first] += new_terms2[j].second;
        }
    }
    StateVector state;
    for (size_t idx = 0, maxidx = exp_hash_.size(); idx < maxidx; idx++) {
        const Determinant& d = exp_hash_.get_det(idx);
        state[d] = state_c[idx];
    }
    time_ += t.get();
    exp_time_ += t.get();
    return state;
}

void SparseFactExp::apply_exp_op_fast(const Determinant& d, Determinant& new_d,
                                      const Determinant& cre, const Determinant& ann, double amp,
                                      double c, StateVector& new_terms) {
    new_d = d;
    const double f = apply_op(new_d, cre, ann) * amp;
    // this is to deal with number operators (should be removed)
    if (d != new_d) {
        new_terms[d] += c * (std::cos(f) - 1.0);
        new_terms[new_d] += c * std::sin(f);
    }
}

StateVector SparseFactExp::compute_on_the_fly(SparseOperator& sop, const StateVector& state0,
                                              bool inverse, double screen_thresh) {
    local_timer t;
    const auto& op_list = sop.op_list();

    // initialize a state object
    StateVector state(state0);
    StateVector new_terms;

    for (size_t m = 0, nterms = sop.nterms(); m < nterms; m++) {
        size_t n = inverse ? nterms - m - 1 : m;

        // zero the new terms
        new_terms.clear();

        const SQOperator& sqop = op_list[n];
        const Determinant ucre = sqop.cre() - sqop.ann();
        const Determinant uann = sqop.ann() - sqop.cre();
        const double tau = (inverse ? -1.0 : 1.0) * sqop.factor();
        Determinant new_d;
        // loop over all determinants
        for (const auto& det_c : state) {
            const Determinant& d = det_c.first;

            // test if we can apply this operator to this determinant
            if (d.fast_a_and_b_equal_b(sqop.ann()) and d.fast_a_and_b_eq_zero(ucre)) {
                const double c = det_c.second;
                apply_exp_op_fast(d, new_d, sqop.cre(), sqop.ann(), tau, c, new_terms);
            } else if (d.fast_a_and_b_equal_b(sqop.cre()) and d.fast_a_and_b_eq_zero(uann)) {
                const double c = det_c.second;
                apply_exp_op_fast(d, new_d, sqop.ann(), sqop.cre(), -tau, c, new_terms);
            }
        }
        for (const auto& d_c : new_terms) {
            if (std::fabs(d_c.second) > screen_thresh) {
                state[d_c.first] += d_c.second;
            }
        }
    }
    on_the_fly_time_ += t.get();
    return state;
}

std::map<std::string, double> SparseFactExp::time() const {
    std::map<std::string, double> t;
    t["time"] = time_;
    t["couplings_time"] = couplings_time_;
    t["exp_time"] = exp_time_;
    t["on_the_fly_time"] = on_the_fly_time_;
    return t;
}

} // namespace forte
