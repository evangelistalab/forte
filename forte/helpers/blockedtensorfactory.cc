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

#include "blockedtensorfactory.h"
#include "psi4/psi4-dec.h"
#include "psi4/libmints/molecule.h"
#include "psi4/libmints/wavefunction.h"

#include "psi4/libpsio/psio.h"
#include "psi4/libpsio/psio.hpp"
#include "psi4/libpsi4util/process.h"

#include <algorithm>
#include <string>
#include <tuple>
#include <vector>

using namespace psi;

namespace forte {

BlockedTensorFactory::BlockedTensorFactory() {
    memory_ = psi::Process::environment.get_memory() / 1073741824.0;
    number_of_tensors_ = 0;
}

BlockedTensorFactory::~BlockedTensorFactory() {
    if (print_memory_) {
        memory_summary();
    }
}

ambit::BlockedTensor BlockedTensorFactory::build(ambit::TensorType storage, const std::string& name,
                                                 const std::vector<std::string>& spin_stuff,
                                                 bool is_local_variable) {
    if (print_memory_) {
        outfile->Printf("\n Creating %s ", name.c_str());
    }
    ambit::BlockedTensor BT = ambit::BlockedTensor::build(storage, name, spin_stuff);
    number_of_tensors_ += 1;
    memory_information(BT, is_local_variable);
    if (memory_ < 0.0) {
        outfile->Printf("\n\n Created %s and out of memory", name.c_str());
        outfile->Printf("\n DANGER DANGER Will Robinson\n");
        outfile->Printf("\n Your memory requirements were underestimated.  "
                        "Please be more careful! \n");
    }

    return BT;
}

void BlockedTensorFactory::add_mo_space(const std::string& name, const std::string& mo_indices,
                                        std::vector<size_t> mos, ambit::SpinType spin) {
    ambit::BlockedTensor::add_mo_space(name, mo_indices, mos, spin);
    molabel_to_index_[name] = mos;
}

void BlockedTensorFactory::add_mo_space(const std::string& name, const std::string& mo_indices,
                                        std::vector<std::pair<size_t, ambit::SpinType>> mo_spin) {
    ambit::BlockedTensor::add_mo_space(name, mo_indices, mo_spin);
}

void BlockedTensorFactory::add_composite_mo_space(const std::string& name,
                                                  const std::string& mo_indices,
                                                  const std::vector<std::string>& subspaces) {
    ambit::BlockedTensor::add_composite_mo_space(name, mo_indices, subspaces);
}

// This is a utility function for generating all the necessary strings for the
// orbital spaces
std::vector<std::string> BlockedTensorFactory::generate_indices(const std::string in_str,
                                                                const std::string type) {
    std::vector<std::string> return_string;

    // Hardlined for 4 character strings
    if (type == "all") {
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                for (int k = 0; k < 3; k++) {
                    for (int l = 0; l < 3; l++) {
                        std::string one_string_lower;
                        std::string one_string_upper;
                        std::string one_string_mixed;

                        one_string_lower.push_back(in_str[i]);
                        one_string_lower.push_back(in_str[j]);
                        one_string_lower.push_back(in_str[k]);
                        one_string_lower.push_back(in_str[l]);

                        one_string_upper.push_back(std::toupper(in_str[i]));
                        one_string_upper.push_back(std::toupper(in_str[j]));
                        one_string_upper.push_back(std::toupper(in_str[k]));
                        one_string_upper.push_back(std::toupper(in_str[l]));

                        one_string_mixed.push_back(in_str[i]);
                        one_string_mixed.push_back(std::toupper(in_str[j]));
                        one_string_mixed.push_back(in_str[k]);
                        one_string_mixed.push_back(std::toupper(in_str[l]));

                        return_string.push_back(one_string_lower);
                        return_string.push_back(one_string_upper);
                        return_string.push_back(one_string_mixed);
                    }
                }
            }
        }
    } else {

        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) {
                for (int k = 0; k < 2; k++) {
                    for (int l = 0; l < 2; l++) {
                        std::string one_string_lower;
                        std::string one_string_upper;
                        std::string one_string_mixed;

                        one_string_lower.push_back(in_str[i]);
                        one_string_lower.push_back(in_str[j]);
                        one_string_lower.push_back(in_str[k + 1]);
                        one_string_lower.push_back(in_str[l + 1]);

                        one_string_upper.push_back(std::toupper(in_str[i]));
                        one_string_upper.push_back(std::toupper(in_str[j]));
                        one_string_upper.push_back(std::toupper(in_str[k + 1]));
                        one_string_upper.push_back(std::toupper(in_str[l + 1]));

                        one_string_mixed.push_back(in_str[i]);
                        one_string_mixed.push_back(std::toupper(in_str[j]));
                        one_string_mixed.push_back(in_str[k + 1]);
                        one_string_mixed.push_back(std::toupper(in_str[l + 1]));

                        return_string.push_back(one_string_lower);
                        return_string.push_back(one_string_upper);
                        return_string.push_back(one_string_mixed);
                    }
                }
            }
        }
    }

    return return_string;
}

void BlockedTensorFactory::memory_information(ambit::BlockedTensor BT, bool is_local_variable) {
    double size_of_tensor = 0.0;
    std::vector<std::string> BTblocks = BT.block_labels();
    for (const std::string& block : BTblocks) {
        size_of_tensor += BT.block(block).numel();
    }
    double memory_of_tensor = (size_of_tensor * 8.0) / 1073741824.0;
    tensors_information_.push_back(std::make_pair(BT.name(), memory_of_tensor));
    number_of_blocks_.push_back(BT.numblocks());

    if (!(is_local_variable)) {
        memory_ -= memory_of_tensor;
    }
    if (print_memory_) {
        outfile->Printf("\n For tensor %s, this will take up %6.6f GB", BT.name().c_str(),
                        memory_of_tensor);
        outfile->Printf("\n %6.6f GB of memory left over", memory_);
    }
}

void BlockedTensorFactory::memory_summary() {
    outfile->Printf("\n Memory Summary of the %u tensors \n", number_of_tensors_);
    outfile->Printf("\n TensorName \t Number_of_blocks \t memory gb");
    for (size_t i = 0; i < tensors_information_.size(); i++) {
        outfile->Printf("\n %-25s  %u    %8.8f GB", tensors_information_[i].first.c_str(),
                        number_of_blocks_[i], tensors_information_[i].second);
    }
    outfile->Printf("\n Memory left over: %8.6f GB\n", memory_);
}

std::vector<std::string>
BlockedTensorFactory::spin_cases_avoid(const std::vector<std::string>& in_str_vec,
                                       int how_many_active) {

    std::vector<std::string> out_str_vec;
    if (how_many_active == 1) {
        for (const std::string spin : in_str_vec) {
            size_t spin_ind = spin.find('a');
            size_t spin_ind2 = spin.find('A');
            if (spin_ind != std::string::npos || spin_ind2 != std::string::npos) {
                out_str_vec.push_back(spin);
            }
        }
    } else if (how_many_active > 1) {
        for (const std::string spin : in_str_vec) {
            std::string spin_transform = spin;
            std::transform(spin_transform.begin(), spin_transform.end(), spin_transform.begin(),
                           ::tolower);
            int anum = std::count(spin_transform.begin(), spin_transform.end(), 'a');
            if (anum >= how_many_active) {
                out_str_vec.push_back(spin);
            }
        }
    }
    return out_str_vec;
}

void BlockedTensorFactory::memory_summary_per_block(ambit::BlockedTensor& tensor) {

    std::vector<std::string> Tensor_label = tensor.block_labels();
    outfile->Printf("\n\n\n\n Memory Summary for %s\n\n", tensor.name().c_str());
    for (auto& block : Tensor_label) {
        double memory_per_block = (tensor.block(block).numel() * sizeof(double)) / 1073741824.0;
        outfile->Printf("\n %s   %8.8f GB", block.c_str(), memory_per_block);
    }
}
} // namespace forte
