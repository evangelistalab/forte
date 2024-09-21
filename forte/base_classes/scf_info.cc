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

SCFInfo::SCFInfo(psi::SharedWavefunction wfn)
    : nmopi_(wfn->nmopi()), doccpi_(wfn->doccpi()), soccpi_(wfn->soccpi()), energy_(wfn->energy()),
      epsilon_a_(wfn->epsilon_a()), epsilon_b_(wfn->epsilon_b()), Ca_(wfn->Ca()), Cb_(wfn->Cb()) {}

SCFInfo::SCFInfo(const psi::Dimension& nmopi, const psi::Dimension& doccpi,
                 const psi::Dimension& soccpi, double reference_energy,
                 std::shared_ptr<psi::Vector> epsilon_a, std::shared_ptr<psi::Vector> epsilon_b,
                 std::shared_ptr<psi::Matrix> Ca, std::shared_ptr<psi::Matrix> Cb)
    : nmopi_(nmopi), doccpi_(doccpi), soccpi_(soccpi), energy_(reference_energy),
      epsilon_a_(epsilon_a), epsilon_b_(epsilon_b), Ca_(Ca), Cb_(Cb) {}

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

std::shared_ptr<SCFInfo> reorder_orbitals(std::shared_ptr<SCFInfo> scf_info,
                                          const std::vector<std::vector<size_t>>& new_order,
                                          std::shared_ptr<psi::Wavefunction> wfn) {
    auto Ca_old = scf_info->_Ca();
    auto Cb_old = scf_info->_Cb();
    auto Ca_new = std::make_shared<psi::Matrix>("Ca", Ca_old->rowspi(), Ca_old->colspi());
    auto Cb_new = std::make_shared<psi::Matrix>("Cb", Cb_old->rowspi(), Cb_old->colspi());

    auto eps_a_old = scf_info->epsilon_a();
    auto eps_b_old = scf_info->epsilon_b();
    auto eps_a_new = std::make_shared<psi::Vector>("eps_a", eps_a_old->dimpi());
    auto eps_b_new = std::make_shared<psi::Vector>("eps_b", eps_b_old->dimpi());

    size_t nirrep = scf_info->nmopi().n();

    if (new_order.size() != nirrep) {
        throw std::runtime_error("The number of MOs in the new order does not match the number of "
                                 "MOs in the old order.");
    }
    for (size_t h = 0; h < nirrep; ++h) {
        size_t nmo_h = scf_info->nmopi()[h];
        if (new_order[h].size() != nmo_h) {
            throw std::runtime_error(
                "The number of MOs in the new order does not match the number of "
                "MOs in the old order.");
        }
        for (size_t p = 0; p < nmo_h; ++p) {
            auto p_new = new_order[h][p];
            Ca_new->set_column(h, p_new, Ca_old->get_column(h, p));
            Cb_new->set_column(h, p_new, Cb_old->get_column(h, p));
            eps_a_new->set(h, p_new, eps_a_old->get(h, p));
            eps_b_new->set(h, p_new, eps_b_old->get(h, p));
        }
    }
    // Copy to psi::Wavefunction
    if (wfn) {
        wfn->Ca()->copy(Ca_new);
        wfn->Cb()->copy(Cb_new);
    }

    // Make a new SCFInfo object
    return std::make_shared<SCFInfo>(scf_info->nmopi(), scf_info->doccpi(), scf_info->soccpi(),
                                     scf_info->reference_energy(), scf_info->epsilon_a(),
                                     scf_info->epsilon_b(), Ca_new, Cb_new);
}

} // namespace forte
