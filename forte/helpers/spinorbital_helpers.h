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

#ifndef _spinorbital_helpers_h_
#define _spinorbital_helpers_h_

#include <vector>
#include "base_classes/rdms.h"

namespace ambit {
class Tensor;
} // namespace ambit

namespace forte {
class ForteIntegrals;

/**
 * @brief spinorbital_oei A helper function to return the one-electron integrals
 *        in a spin orbital basis of the form <p|h|q>. This function will compute
 *        a block of integrals at a time.
 * @param ints the ForteIntegrals object
 * @param p a vector of MO indices that define the block of integrals to generate
 * @param q a vector of MO indices that define the block of integrals to generate
 * @return a vector of RDMs up to the max rank stored in the RDMs object
 */
ambit::Tensor spinorbital_oei(const std::shared_ptr<ForteIntegrals> ints,
                              const std::vector<size_t>& p, const std::vector<size_t>& q);

/**
 * @brief spinorbital_tei A helper function to return the two-electron integrals
 *        in a spin orbital basis of the form <pq||rs>. This function will compute
 *        a block of integrals at a time.
 * @param ints the ForteIntegrals object
 * @param p a vector of MO indices that define the block of integrals to generate
 * @param q a vector of MO indices that define the block of integrals to generate
 * @param r a vector of MO indices that define the block of integrals to generate
 * @param s a vector of MO indices that define the block of integrals to generate*
 * @return a vector of RDMs up to the max rank stored in the RDMs object
 */
ambit::Tensor spinorbital_tei(const std::shared_ptr<ForteIntegrals> ints,
                              const std::vector<size_t>& p, const std::vector<size_t>& q,
                              const std::vector<size_t>& r, const std::vector<size_t>& s);

/**
 * @brief spinorbital_fock A helper function to return the fock matrix elements <p|f|q>
 *        in a spin orbital basis. This function will compute a block of integrals at a time.
 * @param ints the ForteIntegrals object
 * @param p a vector of MO indices that define the block of integrals to generate
 * @param q a vector of MO indices that define the block of integrals to generate
 * @param docc a vector of doubly occupied MOs
 * @return a vector of RDMs up to the max rank stored in the RDMs object
 */
ambit::Tensor spinorbital_fock(const std::shared_ptr<ForteIntegrals> ints,
                               const std::vector<size_t>& p, const std::vector<size_t>& q,
                               const std::vector<size_t>& docc);

/**
 * @brief spinorbital_rdms A helper function to return RDMs in a spin orbital basis
 * @param rdms the RDMs object
 * @return a vector of RDMs up to the max rank stored in the RDMs object
 */
std::vector<ambit::Tensor> spinorbital_rdms(RDMs& rdms);

/**
 * @brief spinorbital_rdms A helper function to return density cumulants in a spin orbital basis
 * @param rdms the RDMs object
 * @return a vector of cumulants up to the max rank stored in the RDMs object
 */
std::vector<ambit::Tensor> spinorbital_cumulants(RDMs& rdms);

} // namespace forte

#endif // _spinorbital_helpers_h_
