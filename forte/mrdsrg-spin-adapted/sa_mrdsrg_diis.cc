/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2022 by its authors (see COPYING, COPYING.LESSER, AUTHORS).
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

#include "psi4/libdiis/diismanager.h"

#include "sa_mrdsrg.h"

using namespace psi;

namespace forte {

void SA_MRDSRG::diis_manager_init() {
    diis_manager_ = std::make_shared<DIISManager>(diis_max_vec_, "SA_MRDSRG DIIS",
                                                  DIISManager::RemovalPolicy::LargestError,
                                                  DIISManager::StoragePolicy::OnDisk);

    diis_manager_->set_error_vector_size(DT1_, DT2_);

    diis_manager_->set_vector_size(T1_, T2_);
}

void SA_MRDSRG::diis_manager_add_entry() {
    diis_manager_->add_entry(DT1_, DT2_, T1_, T2_);
}

void SA_MRDSRG::diis_manager_extrapolate() {
    diis_manager_->extrapolate(T1_, T2_);
}

void SA_MRDSRG::diis_manager_cleanup() {
    diis_manager_->reset_subspace();
    diis_manager_->delete_diis_file();
}
} // namespace forte
