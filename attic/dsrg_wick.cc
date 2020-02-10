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

#include "dsrg_wick.h"
#include "ambit/tensor.h"

namespace psi {
namespace forte {

DSRG_WICK::DSRG_WICK(std::shared_ptr<MOSpaceInfo> mo_space_info, ambit::BlockedTensor Fock,
                     ambit::BlockedTensor RTEI, ambit::BlockedTensor T1, ambit::BlockedTensor T2)
    : mo_space_info_(mo_space_info) {
    // put all beta behind alpha
    size_t mo_shift = mo_space_info_->size("RESTRICTED_DOCC") + mo_space_info_->size("ACTIVE") +
                      mo_space_info_->size("RESTRICTED_UOCC");

    acore_mos = mo_space_info_->corr_absolute_mo("RESTRICTED_DOCC");
    for (size_t idx : acore_mos) {
        bcore_mos.push_back(idx + mo_shift);
    }

    aactv_mos = mo_space_info_->corr_absolute_mo("ACTIVE");
    for (size_t idx : aactv_mos) {
        bactv_mos.push_back(idx + mo_shift);
    }

    avirt_mos = mo_space_info_->corr_absolute_mo("RESTRICTED_UOCC");
    for (size_t idx : avirt_mos) {
        bvirt_mos.push_back(idx + mo_shift);
    }

    ncso_ = 2 * mo_shift;
    nc_ = 2 * acore_mos.size();
    na_ = 2 * aactv_mos.size();
    nv_ = 2 * avirt_mos.size();
    nh_ = nc_ + na_;
    np_ = nv_ + na_;

    // block indices: c for core, a for active, v for virtual
    // lowercase for alpha and uppercase for beta
    label_to_spacemo['c'] = acore_mos;
    label_to_spacemo['C'] = bcore_mos;
    label_to_spacemo['a'] = aactv_mos;
    label_to_spacemo['A'] = bactv_mos;
    label_to_spacemo['v'] = avirt_mos;
    label_to_spacemo['V'] = bvirt_mos;

    // setup necessary operators
    //    setup(Fock, RTEI, T1, T2);
}

void DSRG_WICK::setup(ambit::BlockedTensor Fock, ambit::BlockedTensor RTEI, ambit::BlockedTensor T1,
                      ambit::BlockedTensor T2) {
    // Setup Fock operator
    for (std::string block : Fock.block_labels()) {
        Fock.block(block).citerate([&](const std::vector<size_t>& i, const double& value) {
            int idx0 = label_to_spacemo[block[0]][i[0]];
            int idx1 = label_to_spacemo[block[1]][i[1]];
            SqOperator op_pq({idx0}, {idx1});
            F_.add(value, op_pq);
        });
    }
    //    outfile->Printf("\n  %s", F_.str().c_str());

    // Setup two-electron integral
    for (std::string block : RTEI.block_labels()) {
        RTEI.block(block).citerate([&](const std::vector<size_t>& i, const double& value) {
            int idx0 = label_to_spacemo[block[0]][i[0]];
            int idx1 = label_to_spacemo[block[1]][i[1]];
            int idx2 = label_to_spacemo[block[2]][i[2]];
            int idx3 = label_to_spacemo[block[3]][i[3]];
            if (islower(block[0]) && isupper(block[1])) {
                SqOperator abab({idx0, idx1}, {idx2, idx3});
                SqOperator abba({idx0, idx1}, {idx3, idx2});
                SqOperator baab({idx1, idx0}, {idx2, idx3});
                SqOperator baba({idx1, idx0}, {idx3, idx2});
                V_.add(value, abab);
                V_.add(value, baba);
                V_.add(-value, abba);
                V_.add(-value, baab);
            } else {
                SqOperator op_pqrs({idx0, idx1}, {idx2, idx3});
                V_.add(value, op_pqrs);
            }
            //            outfile->Printf("\n  [%d][%d][%d][%d] = %.15f",
            //            idx0, idx1, idx2, idx3, value);
        });
    }

    // Setup T1
    for (std::string block : T1.block_labels()) {
        T1.block(block).citerate([&](const std::vector<size_t>& i, const double& value) {
            int idx0 = label_to_spacemo[block[0]][i[0]];
            int idx1 = label_to_spacemo[block[1]][i[1]];
            SqOperator op_pq({idx0}, {idx1});
            T1_.add(value, op_pq);
        });
    }

    // Setup T2
    for (std::string block : T2.block_labels()) {
        T2.block(block).citerate([&](const std::vector<size_t>& i, const double& value) {
            int idx0 = label_to_spacemo[block[0]][i[0]];
            int idx1 = label_to_spacemo[block[1]][i[1]];
            int idx2 = label_to_spacemo[block[2]][i[2]];
            int idx3 = label_to_spacemo[block[3]][i[3]];
            if (islower(block[0]) && isupper(block[1])) {
                SqOperator abab({idx0, idx1}, {idx2, idx3});
                SqOperator abba({idx0, idx1}, {idx3, idx2});
                SqOperator baab({idx1, idx0}, {idx2, idx3});
                SqOperator baba({idx1, idx0}, {idx3, idx2});
                T2_.add(value, abab);
                T2_.add(value, baba);
                T2_.add(-value, abba);
                T2_.add(-value, baab);
            } else {
                SqOperator op_pqrs({idx0, idx1}, {idx2, idx3});
                T2_.add(value, op_pqrs);
            }
        });
    }
}
}
}
