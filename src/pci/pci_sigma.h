/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2019 by its authors (see COPYING, COPYING.LESSER,
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

#ifndef _pci_sigma_h_
#define _pci_sigma_h_

#include "sparse_ci/sigma_vector.h"

namespace forte {

class PCISigmaVector : public SigmaVector {
  public:
    PCISigmaVector(
        det_hashvec& dets_hashvec, std::vector<double>& ref_C, double spawning_threshold,
        std::shared_ptr<ActiveSpaceIntegrals> as_ints,
        std::function<bool(double, double, double)> prescreen_H_CI,
        std::function<bool(double, double, double, double)> important_H_CI_CJ,
        const std::vector<std::tuple<int, double, std::vector<std::tuple<int, double>>>>&
            a_couplings,
        const std::vector<std::tuple<int, double, std::vector<std::tuple<int, double>>>>&
            b_couplings,
        const std::vector<std::tuple<int, int, double, std::vector<std::tuple<int, int, double>>>>&
            aa_couplings,
        const std::vector<std::tuple<int, int, double, std::vector<std::tuple<int, int, double>>>>&
            ab_couplings,
        const std::vector<std::tuple<int, int, double, std::vector<std::tuple<int, int, double>>>>&
            bb_couplings,
        double dets_single_max_coupling, double dets_double_max_coupling);
    void compute_sigma(psi::SharedVector sigma, psi::SharedVector b) override;
    void get_diagonal(psi::Vector& diag) override;
    void add_bad_roots(std::vector<std::vector<std::pair<size_t, double>>>& bad_states) override;

  private:
    det_hashvec& dets_;
    std::vector<double>& ref_C_;
    double spawning_threshold_;
    /// The molecular integrals for the active space
    /// This object holds only the integrals for the orbital contained in the active_mo_ vector.
    /// The one-electron integrals and scalar energy contains contributions from the
    /// doubly occupied orbitals specified by the core_mo_ vector.
    std::shared_ptr<ActiveSpaceIntegrals> as_ints_;
    /// Function for prescreening with one coefficient
    std::function<bool(double, double, double)> prescreen_H_CI_;
    /// Function for important matrix element
    std::function<bool(double, double, double, double)> important_H_CI_CJ_;
    /// The maximum number of threads
    int num_threads_;
    /// The number of off-diagonal elements
    size_t num_off_diag_elem_;
    /// A map used to store the largest absolute value of the couplings of a
    /// determinant to all of its singly and doubly excited states.
    /// Bounds are stored as a pair (f_max,v_max) where f_max and v_max are
    /// the couplings to the singles and doubles, respectively.
    std::unordered_map<Determinant, std::pair<double, double>, Determinant::Hash>
        dets_max_couplings_;
    double dets_single_max_coupling_;
    const std::vector<std::tuple<int, double, std::vector<std::tuple<int, double>>>>&a_couplings_,
        &b_couplings_;
    size_t a_couplings_size_, b_couplings_size_;
    double dets_double_max_coupling_;
    const std::vector<std::tuple<int, int, double, std::vector<std::tuple<int, int, double>>>>
        &aa_couplings_, &ab_couplings_, &bb_couplings_;
    size_t aa_couplings_size_, ab_couplings_size_, bb_couplings_size_;

    /// Apply symmetric approx tau H to a set of determinants with selection
    /// according to reference coefficients
    void apply_tau_H_symm(double tau, double spawning_threshold, det_hashvec& dets_hashvec,
                          std::vector<double>& ref_C, std::vector<double>& result_C, double S,
                          size_t& overlap_size);

    /// Apply symmetric approx tau H to a determinant using dynamic screening
    /// with selection according to a reference coefficient
    /// and with HBCI sorting scheme with singles screening
    void apply_tau_H_symm_det_dynamic_HBCI_2(
        double tau, double spawning_threshold, const det_hashvec& dets_hashvec,
        const std::vector<double>& pre_C, size_t I, double CI, std::vector<double>& result_C,
        std::vector<std::pair<Determinant, double>>& new_det_C_vec,
        std::pair<double, double>& max_coupling);
    /// Apply symmetric approx tau H to a set of determinants with selection
    /// according to reference coefficients
    void apply_tau_H_ref_C_symm(double tau, double spawning_threshold,
                                const det_hashvec& result_dets, const std::vector<double>& ref_C,
                                const std::vector<double>& pre_C, std::vector<double>& result_C,
                                const size_t overlap_size, double S);

    /// Apply symmetric approx tau H to a determinant using dynamic screening
    /// with selection according to a reference coefficient
    /// and with HBCI sorting scheme with singles screening
    void apply_tau_H_ref_C_symm_det_dynamic_HBCI_2(
        double tau, double spawning_threshold, const det_hashvec& dets_hashvec,
        const std::vector<double>& pre_C, const std::vector<double>& ref_C, size_t I, double CI,
        double ref_CI, const size_t overlap_size, std::vector<double>& result_C,
        const std::pair<double, double>& max_coupling);
};
}
#endif // _pci_sigma_h_
