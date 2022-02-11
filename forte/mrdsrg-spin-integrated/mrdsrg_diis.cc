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
#include "ambit/blocked_tensor.h"

#include "mrdsrg.h"

using namespace psi;

namespace forte {

std::vector<double> MRDSRG::copy_amp_diis(BlockedTensor& T1, const std::vector<std::string>& label1,
                                          BlockedTensor& T2,
                                          const std::vector<std::string>& label2) {
    std::vector<double> out;

    for (const auto& block : label1) {
        out.insert(out.end(), T1.block(block).data().begin(), T1.block(block).data().end());
    }
    for (const auto& block : label2) {
        out.insert(out.end(), T2.block(block).data().begin(), T2.block(block).data().end());
    }

    return out;
}

size_t MRDSRG::vector_size_diis(BlockedTensor& T1, const std::vector<std::string>& label1,
                                BlockedTensor& T2, const std::vector<std::string>& label2) {
    size_t total_elements = 0;
    for (const auto& block : label1) {
        total_elements += T1.block(block).numel();
    }
    for (const auto& block : label2) {
        total_elements += T2.block(block).numel();
    }
    return total_elements;
}

void MRDSRG::return_amp_diis(BlockedTensor& T1, const std::vector<std::string>& label1,
                             BlockedTensor& T2, const std::vector<std::string>& label2,
                             const std::vector<double>& data) {
    // test data
    std::map<std::string, size_t> num_elements;
    size_t total_elements = 0;

    for (const auto& block : label1) {
        size_t numel = T1.block(block).numel();
        num_elements[block] = total_elements;
        total_elements += numel;
    }
    for (const auto& block : label2) {
        size_t numel = T2.block(block).numel();
        num_elements[block] = total_elements;
        total_elements += numel;
    }

    if (data.size() != total_elements) {
        throw psi::PSIEXCEPTION("Number of elements in T1 and T2 do not match the bid data vector");
    }

    // transfer data
    for (const auto& block : label1) {
        std::vector<double>::const_iterator start = data.begin() + num_elements[block];
        std::vector<double>::const_iterator end = start + T1.block(block).numel();
        std::vector<double> T1_this_block(start, end);
        T1.block(block).data() = T1_this_block;
    }
    for (const auto& block : label2) {
        std::vector<double>::const_iterator start = data.begin() + num_elements[block];
        std::vector<double>::const_iterator end = start + T2.block(block).numel();
        std::vector<double> T2_this_block(start, end);
        T2.block(block).data() = T2_this_block;
    }
}

void MRDSRG::diis_manager_init() {
    diis_manager_ = std::make_shared<DIISManager>(diis_max_vec_, "MRDSRG DIIS",
                                                  DIISManager::RemovalPolicy::LargestError,
                                                  DIISManager::StoragePolicy::OnDisk);

    amp_ptrs_.clear();
    res_ptrs_.clear();

    std::vector<std::string> blocks{
        "ca",   "cv",   "av",   "CA",   "CV",   "AV",   "ccaa", "ccav", "ccva", "ccvv", "caaa",
        "caav", "cava", "cavv", "acaa", "acav", "acva", "acvv", "aaav", "aava", "aavv", "cCaA",
        "cCaV", "cCvA", "cCvV", "cAaA", "cAaV", "cAvA", "cAvV", "aCaA", "aCaV", "aCvA", "aCvV",
        "aAaV", "aAvA", "aAvV", "CCAA", "CCAV", "CCVA", "CCVV", "CAAA", "CAAV", "CAVA", "CAVV",
        "ACAA", "ACAV", "ACVA", "ACVV", "AAAV", "AAVA", "AAVV"};

    std::vector<size_t> sizes(51);
    for (int i = 0; i < 51; ++i) {
        auto block = blocks[i];
        if (block.size() == 2) {
            sizes[i] = T1_.block(block).numel();
            amp_ptrs_.push_back(T1_.block(block).data().data());
            res_ptrs_.push_back(DT1_.block(block).data().data());
        } else {
            sizes[i] = T2_.block(block).numel();
            amp_ptrs_.push_back(T2_.block(block).data().data());
            res_ptrs_.push_back(DT2_.block(block).data().data());
        }
    }

    diis_manager_->set_error_vector_size(
        sizes[0], sizes[1], sizes[2], sizes[3], sizes[4], sizes[5],
        sizes[6], sizes[7], sizes[8], sizes[9], sizes[10], sizes[11],
        sizes[12], sizes[13], sizes[14], sizes[15], sizes[16], sizes[17],
        sizes[18], sizes[19], sizes[20], sizes[21], sizes[22], sizes[23],
        sizes[24], sizes[25], sizes[26], sizes[27], sizes[28], sizes[29],
        sizes[30], sizes[31], sizes[32], sizes[33], sizes[34], sizes[35],
        sizes[36], sizes[37], sizes[38], sizes[39], sizes[40], sizes[41],
        sizes[42], sizes[43], sizes[44], sizes[45], sizes[46], sizes[47],
        sizes[48], sizes[49], sizes[50]);

    diis_manager_->set_vector_size(
        sizes[0], sizes[1], sizes[2], sizes[3], sizes[4], sizes[5],
        sizes[6], sizes[7], sizes[8], sizes[9], sizes[10], sizes[11],
        sizes[12], sizes[13], sizes[14], sizes[15], sizes[16], sizes[17],
        sizes[18], sizes[19], sizes[20], sizes[21], sizes[22], sizes[23],
        sizes[24], sizes[25], sizes[26], sizes[27], sizes[28], sizes[29],
        sizes[30], sizes[31], sizes[32], sizes[33], sizes[34], sizes[35],
        sizes[36], sizes[37], sizes[38], sizes[39], sizes[40], sizes[41],
        sizes[42], sizes[43], sizes[44], sizes[45], sizes[46], sizes[47],
        sizes[48], sizes[49], sizes[50]);
}

void MRDSRG::diis_manager_add_entry() {
    diis_manager_->add_entry(
        102, res_ptrs_[0], res_ptrs_[1], res_ptrs_[2], res_ptrs_[3], res_ptrs_[4], res_ptrs_[5],
        res_ptrs_[6], res_ptrs_[7], res_ptrs_[8], res_ptrs_[9], res_ptrs_[10], res_ptrs_[11],
        res_ptrs_[12], res_ptrs_[13], res_ptrs_[14], res_ptrs_[15], res_ptrs_[16], res_ptrs_[17],
        res_ptrs_[18], res_ptrs_[19], res_ptrs_[20], res_ptrs_[21], res_ptrs_[22], res_ptrs_[23],
        res_ptrs_[24], res_ptrs_[25], res_ptrs_[26], res_ptrs_[27], res_ptrs_[28], res_ptrs_[29],
        res_ptrs_[30], res_ptrs_[31], res_ptrs_[32], res_ptrs_[33], res_ptrs_[34], res_ptrs_[35],
        res_ptrs_[36], res_ptrs_[37], res_ptrs_[38], res_ptrs_[39], res_ptrs_[40], res_ptrs_[41],
        res_ptrs_[42], res_ptrs_[43], res_ptrs_[44], res_ptrs_[45], res_ptrs_[46], res_ptrs_[47],
        res_ptrs_[48], res_ptrs_[49], res_ptrs_[50], amp_ptrs_[0], amp_ptrs_[1], amp_ptrs_[2],
        amp_ptrs_[3], amp_ptrs_[4], amp_ptrs_[5], amp_ptrs_[6], amp_ptrs_[7], amp_ptrs_[8],
        amp_ptrs_[9], amp_ptrs_[10], amp_ptrs_[11], amp_ptrs_[12], amp_ptrs_[13], amp_ptrs_[14],
        amp_ptrs_[15], amp_ptrs_[16], amp_ptrs_[17], amp_ptrs_[18], amp_ptrs_[19], amp_ptrs_[20],
        amp_ptrs_[21], amp_ptrs_[22], amp_ptrs_[23], amp_ptrs_[24], amp_ptrs_[25], amp_ptrs_[26],
        amp_ptrs_[27], amp_ptrs_[28], amp_ptrs_[29], amp_ptrs_[30], amp_ptrs_[31], amp_ptrs_[32],
        amp_ptrs_[33], amp_ptrs_[34], amp_ptrs_[35], amp_ptrs_[36], amp_ptrs_[37], amp_ptrs_[38],
        amp_ptrs_[39], amp_ptrs_[40], amp_ptrs_[41], amp_ptrs_[42], amp_ptrs_[43], amp_ptrs_[44],
        amp_ptrs_[45], amp_ptrs_[46], amp_ptrs_[47], amp_ptrs_[48], amp_ptrs_[49], amp_ptrs_[50]);
}

void MRDSRG::diis_manager_extrapolate() {
    diis_manager_->extrapolate(
        51, amp_ptrs_[0], amp_ptrs_[1], amp_ptrs_[2], amp_ptrs_[3], amp_ptrs_[4], amp_ptrs_[5],
        amp_ptrs_[6], amp_ptrs_[7], amp_ptrs_[8], amp_ptrs_[9], amp_ptrs_[10], amp_ptrs_[11],
        amp_ptrs_[12], amp_ptrs_[13], amp_ptrs_[14], amp_ptrs_[15], amp_ptrs_[16], amp_ptrs_[17],
        amp_ptrs_[18], amp_ptrs_[19], amp_ptrs_[20], amp_ptrs_[21], amp_ptrs_[22], amp_ptrs_[23],
        amp_ptrs_[24], amp_ptrs_[25], amp_ptrs_[26], amp_ptrs_[27], amp_ptrs_[28], amp_ptrs_[29],
        amp_ptrs_[30], amp_ptrs_[31], amp_ptrs_[32], amp_ptrs_[33], amp_ptrs_[34], amp_ptrs_[35],
        amp_ptrs_[36], amp_ptrs_[37], amp_ptrs_[38], amp_ptrs_[39], amp_ptrs_[40], amp_ptrs_[41],
        amp_ptrs_[42], amp_ptrs_[43], amp_ptrs_[44], amp_ptrs_[45], amp_ptrs_[46], amp_ptrs_[47],
        amp_ptrs_[48], amp_ptrs_[49], amp_ptrs_[50]);
}

void MRDSRG::diis_manager_cleanup() {
    amp_ptrs_.clear();
    res_ptrs_.clear();
    diis_manager_->reset_subspace();
    diis_manager_->delete_diis_file();
}
} // namespace forte
