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

#ifndef DSRG_WICK_H
#define DSRG_WICK_H

#include "ambit/blocked_tensor.h"
#include "sq.h"
#include "helpers/mo_space_info.h"

typedef std::vector<double> d1;
typedef std::vector<d1> d2;
typedef std::vector<d2> d3;
typedef std::vector<d3> d4;

namespace psi {
namespace forte {

class DSRG_WICK {
  public:
    DSRG_WICK(std::shared_ptr<MOSpaceInfo> mo_space_info,
              ambit::BlockedTensor Fock, ambit::BlockedTensor RTEI,
              ambit::BlockedTensor T1, ambit::BlockedTensor T2);

  private:
    // MO space info
    std::shared_ptr<MOSpaceInfo> mo_space_info_;
    // Fock operator
    Operator F_;
    // Antisymmetrized two electron integral
    Operator V_;
    // Single excitation operator
    Operator T1_;
    // Double excitation operator
    Operator T2_;

    /// List of alpha core MOs
    std::vector<size_t> acore_mos;
    /// List of alpha active MOs
    std::vector<size_t> aactv_mos;
    /// List of alpha virtual MOs
    std::vector<size_t> avirt_mos;
    /// List of alpha hole MOs
    std::vector<size_t> ahole_mos;
    /// List of alpha particle MOs
    std::vector<size_t> apart_mos;

    /// List of beta core MOs
    std::vector<size_t> bcore_mos;
    /// List of beta active MOs
    std::vector<size_t> bactv_mos;
    /// List of beta virtual MOs
    std::vector<size_t> bvirt_mos;
    /// List of beta hole MOs
    std::vector<size_t> bhole_mos;
    /// List of beta particle MOs
    std::vector<size_t> bpart_mos;

    /// A map between space label and space mos
    std::map<char, std::vector<size_t>> label_to_spacemo;

    // size of spin orbital space
    size_t ncso_;
    size_t nc_;
    size_t na_;
    size_t nv_;
    size_t nh_;
    size_t np_;

    // setup Fock operator (BlockedTensor -> Operator)
    void setup(ambit::BlockedTensor Fock, ambit::BlockedTensor RTEI,
               ambit::BlockedTensor T1, ambit::BlockedTensor T2);
};
}
}

#endif // DSRG_WICK_H
