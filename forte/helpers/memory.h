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

#pragma once

#include <cassert>
#include <map>
#include <string>
#include <utility> // std::forward
#include <vector>

namespace forte {

/**
 * @brief Compute the memory requirement to store a given number of elements
 * @param T The type of the data stored
 * @param n The number of elements for storage
 * @return A pair (size, "unit") with the size given in appropriate unit (B, KB, MB, GB, TB, PB)
 */
template <typename T> std::pair<double, std::string> to_xb2(size_t nele) {
    constexpr size_t type_size = sizeof(T);
    // map the size
    std::map<std::string, double> to_XB;
    to_XB["B"] = 1.0;
    to_XB["KB"] = 1000.0; // use 1000.0 for safety
    to_XB["MB"] = 1000000.0;
    to_XB["GB"] = 1000000000.0;
    to_XB["TB"] = 1000000000000.0;
    to_XB["PB"] = 1000000000000000.0;

    // convert to appropriate unit
    size_t bytes = nele * type_size;
    std::pair<double, std::string> out;
    for (auto& XB : to_XB) {
        double xb = bytes / XB.second;
        if (xb >= 0.9 && xb < 900.0) {
            out = std::make_pair(xb, XB.first);
            break;
        }
    }
    return out;
}

/// @brief A templated buffer class
template <typename T> class Buffer {
  public:
    Buffer(size_t initial_size = 0) : buffer_(initial_size) {}

    // Non-modifying (const) begin and end iterators
    auto begin() const noexcept { return buffer_.cbegin(); }
    auto end() const noexcept { return buffer_.cbegin() + num_stored_; }

    size_t size() const noexcept { return num_stored_; }

    size_t capacity() const noexcept { return buffer_.size(); }

    void reset() noexcept { num_stored_ = 0; }

    void push_back(const T& el) {
        if (num_stored_ < buffer_.size()) {
            buffer_[num_stored_] = el;
        } else {
            buffer_.push_back(el);
        }
        num_stored_++;
    }

    template <typename... Args> void emplace_back(Args&&... args) {
        if (num_stored_ < buffer_.size()) {
            buffer_[num_stored_] = T(std::forward<Args>(args)...);
        } else {
            buffer_.emplace_back(std::forward<Args>(args)...);
        }
        ++num_stored_;
    }

    const T& operator[](size_t n) const {
        assert(n < num_stored_); // Add bounds checking to prevent undefined behavior
        return buffer_[n];
    }

  private:
    // the number of elements currently stored in the buffer
    size_t num_stored_ = 0;
    // the buffer element container
    std::vector<T> buffer_;
};

} // namespace forte
