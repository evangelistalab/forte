/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2021 by its authors (see COPYING, COPYING.LESSER, AUTHORS).
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

#include "sparse_ci/sparse_hamiltonian.h"

namespace forte {

SparseHamiltonian::SparseHamiltonian(std::shared_ptr<ActiveSpaceIntegrals> as_ints)
    : as_ints_(as_ints) {}

StateVector SparseHamiltonian::compute(const StateVector& state, double screen_thresh) {
    // store a list of determinants that we have never encountered before
    std::vector<Determinant> new_dets;

    // find new determinants
    for (const auto& det_c : state) {
        const Determinant& det = det_c.first;
        // if we already have precomputed this det add it to the list of new dets
        if (not state_hash_.has_det(det)) {
            new_dets.push_back(det);
        }
    }

    // compute the new couplings
    compute_new_couplings(new_dets, screen_thresh);

    // compute sigma
    return compute_sigma(state, screen_thresh);
}

void SparseHamiltonian::compute_new_couplings(const std::vector<Determinant>& new_dets,
                                              double screen_thresh) {
    local_timer t;

    size_t nmo = as_ints_->nmo();
    auto symm = as_ints_->active_mo_symmetry();

    Determinant new_det;

    // contribution to the diagonal elements
    double E_0 = as_ints_->nuclear_repulsion_energy() + as_ints_->scalar_energy();

    for (const auto& det : new_dets) {
        std::vector<std::pair<size_t, double>> det_couplings;

        size_t det_sigma_idx = sigma_hash_.add(det);

        // diagonal couplings
        // couplings_.emplace_back(det_idx, det_sigma_idx, E_0 + as_ints_->slater_rules(det, det));
        det_couplings.emplace_back(det_sigma_idx, E_0 + as_ints_->slater_rules(det, det));

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
                        det_couplings.emplace_back(new_det_sigma_idx, DHIJ);
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
                        det_couplings.emplace_back(new_det_sigma_idx, DHIJ);
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
                                det_couplings.emplace_back(new_det_sigma_idx, DHIJ);
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
                                det_couplings.emplace_back(new_det_sigma_idx, DHIJ);
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
                                det_couplings.emplace_back(new_det_sigma_idx, DHIJ);
                            }
                        }
                    }
                }
            }
        }
        // here we sort the couplings in decresing magnitude to help with the screening later
        sort(begin(det_couplings), end(det_couplings), [](auto const& a, auto const& b) {
            return std::fabs(a.second) > std::fabs(b.second);
        });
        couplings_[det] = det_couplings;
    }
    timings_["coupling_time"] += t.get();
    timings_["time"] += t.get();
}

StateVector SparseHamiltonian::compute_sigma(const StateVector& state, double screen_thresh) {
    local_timer t;

    std::vector<double> sigma_c(sigma_hash_.size(), 0.0);

    // compute the sigma vector
    for (const auto& det_c : state) {
        const auto& det = det_c.first;
        const double c = det_c.second;
        for (const auto& new_det_HIJ : couplings_[det]) {
            const size_t new_det_idx = new_det_HIJ.first;
            const double h = new_det_HIJ.second;
            // since the couplings are sorted in decreasing magnitude
            // once an element falls below the threshold we can just
            // terminate the loop
            if (std::fabs(c * h) > screen_thresh) {
                sigma_c[new_det_idx] += c * h;
            } else {
                break;
            }
        }
    }

    // copy data to a StateVector object
    StateVector sigma;
    for (size_t n = 0, maxn = sigma_hash_.size(); n < maxn; n++) {
        sigma[sigma_hash_.get_det(n)] = sigma_c[n];
    }

    timings_["total"] += t.get();
    timings_["sigma"] += t.get();
    return sigma;
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
    timings_["total"] += t.get();
    timings_["on_the_fly"] += t.get();
    return sigma;
}

std::map<std::string, double> SparseHamiltonian::timings() const { return timings_; }

} // namespace forte
