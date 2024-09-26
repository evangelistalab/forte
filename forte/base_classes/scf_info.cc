/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2024 by its authors (see COPYING, COPYING.LESSER, AUTHORS).
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

#include "psi4/libmints/wavefunction.h"
#include "psi4/libmints/matrix.h"
#include "psi4/libmints/vector.h"

#include "scf_info.h"

namespace forte {

SCFInfo::SCFInfo(const psi::Dimension& nmopi, const psi::Dimension& doccpi,
                 const psi::Dimension& soccpi, double reference_energy,
                 std::shared_ptr<psi::Vector> epsilon_a, std::shared_ptr<psi::Vector> epsilon_b,
                 std::shared_ptr<psi::Matrix> Ca, std::shared_ptr<psi::Matrix> Cb) {
    initialize(nmopi, doccpi, soccpi, reference_energy, epsilon_a, epsilon_b, Ca, Cb);
}

SCFInfo::SCFInfo(std::shared_ptr<psi::Wavefunction> psi4_wfn) : psi4_wfn_(psi4_wfn) {
    initialize(psi4_wfn->nmopi(), psi4_wfn->doccpi(), psi4_wfn->soccpi(), psi4_wfn->energy(),
               psi4_wfn->epsilon_a(), psi4_wfn->epsilon_b(), psi4_wfn->Ca(), psi4_wfn->Cb());
}

void SCFInfo::initialize(const psi::Dimension& nmopi, const psi::Dimension& doccpi,
                         const psi::Dimension& soccpi, double reference_energy,
                         std::shared_ptr<psi::Vector> epsilon_a,
                         std::shared_ptr<psi::Vector> epsilon_b, std::shared_ptr<psi::Matrix> Ca,
                         std::shared_ptr<psi::Matrix> Cb) {
    nmopi_ = nmopi;
    doccpi_ = doccpi;
    soccpi_ = soccpi;
    energy_ = reference_energy;
    epsilon_a_ = std::make_shared<psi::Vector>(epsilon_a->clone());
    epsilon_b_ = std::make_shared<psi::Vector>(epsilon_b->clone());
    Ca_ = Ca->clone();
    Cb_ = Cb->clone();
}

psi::Dimension SCFInfo::nmopi() { return nmopi_; }

psi::Dimension SCFInfo::doccpi() { return doccpi_; }

psi::Dimension SCFInfo::soccpi() { return soccpi_; }

double SCFInfo::reference_energy() { return energy_; }

std::shared_ptr<psi::Vector> SCFInfo::epsilon_a() { return epsilon_a_; }

std::shared_ptr<psi::Vector> SCFInfo::epsilon_b() { return epsilon_b_; }

std::shared_ptr<psi::Matrix> SCFInfo::_Ca() { return Ca_; }

std::shared_ptr<psi::Matrix> SCFInfo::_Cb() { return Cb_; }

std::shared_ptr<const psi::Matrix> SCFInfo::Ca() const { return Ca_; }

std::shared_ptr<const psi::Matrix> SCFInfo::Cb() const { return Cb_; }

void SCFInfo::update_psi4_wavefunction() {
    if (psi4_wfn_) {
        psi4_wfn_->Ca()->copy(Ca_);
        psi4_wfn_->Cb()->copy(Cb_);
    }
}

/// This is the main function to update the orbitals in the SCFInfo object
/// All other functions that update the orbitals should call this function
void SCFInfo::update_orbitals(std::shared_ptr<psi::Matrix> Ca, std::shared_ptr<psi::Matrix> Cb,
                              bool transform_ints) {
    // 1. Copy the new orbital coefficients
    Ca_->copy(Ca);
    Cb_->copy(Cb);

    // 2. Update the Psi4 Wavefunction (if available)
    update_psi4_wavefunction();

    // 3. Notify the observers (e.g., ForteIntegral) that the orbitals have been rotated
    std::vector<std::string> messages = {"update_orbitals"};
    if (transform_ints) {
        messages.push_back("transform_ints");
    }
    notify_observers(messages);
}

void SCFInfo::rotate_orbitals(std::shared_ptr<psi::Matrix> Ua, std::shared_ptr<psi::Matrix> Ub,
                              bool transform_ints) {
    // 1. Create the rotated orbital coefficients
    auto Ca_rotated = psi::linalg::doublet(Ca_, Ua);
    auto Cb_rotated = psi::linalg::doublet(Ca_, Ub);

    // 2. Update the orbital coefficients and optionally re-transform the integrals
    update_orbitals(Ca_rotated, Cb_rotated, transform_ints);
}

void SCFInfo::reorder_orbitals(const std::vector<std::vector<size_t>>& new_order) {
    auto Ca_new = std::make_shared<psi::Matrix>("Ca", Ca_->rowspi(), Ca_->colspi());
    auto Cb_new = std::make_shared<psi::Matrix>("Cb", Cb_->rowspi(), Cb_->colspi());

    auto epsilon_a_old = epsilon_a_->clone();
    auto epsilon_b_old = epsilon_b_->clone();

    size_t nirrep = nmopi().n();

    if (new_order.size() != nirrep) {
        throw std::runtime_error("The number of MOs in the new order does not match the number of "
                                 "MOs in the old order.");
    }
    for (size_t h = 0; h < nirrep; ++h) {
        size_t nmo_h = nmopi()[h];
        if (new_order[h].size() != nmo_h) {
            throw std::runtime_error(
                "The number of MOs in the new order does not match the number of "
                "MOs in the old order.");
        }
        for (size_t p = 0; p < nmo_h; ++p) {
            auto p_new = new_order[h][p];
            Ca_new->set_column(h, p_new, Ca_->get_column(h, p));
            Cb_new->set_column(h, p_new, Cb_->get_column(h, p));
            epsilon_a_->set(h, p_new, epsilon_a_old.get(h, p));
            epsilon_b_->set(h, p_new, epsilon_b_old.get(h, p));
        }
    }

    update_orbitals(Ca_new, Cb_new, true);
}

} // namespace forte
