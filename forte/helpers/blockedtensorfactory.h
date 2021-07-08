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

#ifndef BLOCKEDTENSORFACTORY_H
#define BLOCKEDTENSORFACTORY_H

/// A class used to create BlockedTensors similar
/// to matrix factory.
/// All blockedTensor functions with strings should be placed here
/// Creates MO SPACES
#include <vector>
#include <tuple>

#include "ambit/blocked_tensor.h"
#include "psi4/libmints/dimension.h"

#include "integrals/integrals.h"


namespace forte {

class BlockedTensorFactory {
  protected:
    // Whether the tensor is CoreTensor, kDisk, Agnostic
    // TODO:  Make this more generic -
    ambit::TensorType tensor_type_ = ambit::CoreTensor;
    // Overall memory that tensors are taking up
    double memory_;
    // Number of BlockedTensors used
    int number_of_tensors_;
    // String of all the tensors
    std::vector<std::string> tensor_names_;
    // Name of Tensor and memory requirements
    std::vector<std::pair<std::string, double>> tensors_information_;
    std::vector<size_t> number_of_blocks_;
    // Used to control printing for memory summary
    bool print_memory_ = false;

    std::map<std::string, std::vector<size_t>> molabel_to_index_;

  public:
    BlockedTensorFactory();
    ~BlockedTensorFactory();
    /* -- Builds a block tensor
    //@params  storage-> Core, disk, or agnostic (program decides) Core has been
    tested
    //@params  std::string name -> name of tensor
    //@params  spin_stuff -> A vector of strings listing all possible spin
    components
    //ie for a hhpp, {hhpp, hHpP, HHPP}
    //Returns -> BlockTensor
    */
    ambit::BlockedTensor build(ambit::TensorType storage, const std::string& name,
                               const std::vector<std::string>& spin_stuff,
                               bool is_local_variable = false);
    /* -- add_mo_space
      Adds a mo space -> core, active, virtual
    //@params std::string name -> name of the space
    //@params std::string mo_indicies -> string index ie m,n
    //@params vector of indices that correspond to the name ie {0,1,2,3,4,5,6}
    //@params spin -> NoSpin, AlphaSpin, BetaSpin
    */
    void add_mo_space(const std::string& name, const std::string& mo_indices,
                      std::vector<size_t> mos, ambit::SpinType spin);
    void add_mo_space(const std::string& name, const std::string& mo_indices,
                      std::vector<std::pair<size_t, ambit::SpinType>> mo_spin);
    // Adds a composite_mo_space -> combines mo_space -> h = c + a
    void add_composite_mo_space(const std::string& name, const std::string& mo_indices,
                                const std::vector<std::string>& subspaces);
    // Reset mo_space
    void reset_mo_space() {
        ambit::BlockedTensor::reset_mo_spaces();
        molabel_to_index_.clear();
    }
    void memory_info(ambit::BlockedTensor BT);
    /* - This function generates all possible MO spaces and spin components
    /// Param:  std::string is the lables - "cav"
    /// Will take a string like cav and generate all possible combinations of
    this
    /// for a four character string
    */
    std::vector<std::string> generate_indices(const std::string in_str, const std::string type);
    // Lets the user know how much memory is left
    double memory_left() { return memory_; }
    // Calculates the amount of memory BlockedTensor takes up
    // is_local_variable says whether to increment memory
    void memory_information(ambit::BlockedTensor, bool is_local_variable);
    // Array of all things memory
    void memory_summary();
    // controls printing information
    void print_memory_info() { print_memory_ = true; }
    void memory_summary_per_block(ambit::BlockedTensor&);
    std::vector<std::string> spin_cases_avoid(const std::vector<std::string>& vecstring,
                                              int how_many_active);
    std::map<std::string, std::vector<size_t>> get_mo_to_index() { return molabel_to_index_; }
};
}

#endif // BLOCKEDTENSORFACTORY_H
