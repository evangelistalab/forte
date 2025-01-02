/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2025 by its authors (see COPYING, COPYING.LESSER, AUTHORS).
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

/// @brief A templated buffer class used to store elements of type T
/// This class is a simple wrapper around std::vector<T> which provides a variable-size buffer.
/// The buffer is initialized with a given size and is automatically resized when needed.
/// The buffer provides a begin and end iterator to access the stored elements.
/// The buffer also provides a reset method to start storing elements from the beginning.
/// This avoids the need to reallocate memory for each new buffer.
/// Here is an example of how to use the buffer class:
/// @code
/// forte::Buffer<int> buffer(10);
/// for (int i = 0; i < 10; ++i) {
///     buffer.push_back(i);
/// }
/// for (auto it = buffer.begin(); it != buffer.end(); ++it) {
///     std::cout << *it << " ";
/// }
/// std::cout << std::endl;
/// buffer.reset(); // reset the buffer to start storing elements from the beginning
/// for (int i = 0; i < 20; ++i) {
///     buffer.push_back(i);
/// }
/// for (auto it = buffer.begin(); it != buffer.end(); ++it) {
///     std::cout << *it << " ";
/// }
/// std::cout << std::endl;
/// @endcode

template <typename T> class Buffer {
  public:
    /// @brief Construct a buffer with an initial size
    /// @param initial_size number of elements to allocate in the buffer
    Buffer(size_t initial_size = 0) : buffer_(initial_size) {}

    // Non-modifying (const) begin and end iterators
    /// @brief Get a const iterator to the beginning of the buffer
    /// @return const iterator to the beginning of the buffer
    auto begin() const noexcept { return buffer_.cbegin(); }

    /// @brief Get a const iterator to the end of the buffer
    /// @return const iterator to the end of the buffer
    auto end() const noexcept { return buffer_.cbegin() + num_stored_; }

    /// @brief Get the number of elements stored in the buffer
    size_t size() const noexcept { return num_stored_; }

    /// @brief Get the capacity of the buffer
    size_t capacity() const noexcept { return buffer_.size(); }

    /// @brief Reset the buffer to start storing elements from the beginning
    /// This avoids memory reallocation and reuses the existing buffer
    void reset() noexcept { num_stored_ = 0; }

    /// @brief Push an element to the back of the buffer
    void push_back(const T& el) {
        if (num_stored_ < buffer_.size()) {
            buffer_[num_stored_] = el;
        } else {
            buffer_.push_back(el);
        }
        num_stored_++;
    }

    /// @brief Emplace an element to the back of the buffer
    template <typename... Args> void emplace_back(Args&&... args) {
        if (num_stored_ < buffer_.size()) {
            buffer_[num_stored_] = T(std::forward<Args>(args)...);
        } else {
            buffer_.emplace_back(std::forward<Args>(args)...);
        }
        ++num_stored_;
    }

    /// @brief Access an element in the buffer
    const T& operator[](size_t n) const {
        assert(n < num_stored_); // Add bounds checking to prevent undefined behavior
        return buffer_[n];
    }

  private:
    // the number of elements currently handled by the buffer. Note that this is not the same as the
    // buffer size (capacity), which is the total number of elements that can be stored in the
    // buffer.
    size_t num_stored_ = 0;
    // the buffer element container
    std::vector<T> buffer_;
};

} // namespace forte
