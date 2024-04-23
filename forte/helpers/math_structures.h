/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2024 by its authors (see COPYING, COPYING.LESSER,
 * AUTHORS).
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

#include <algorithm>
#include <cassert>
#include <complex>
#include <cmath>
#include <concepts>
#include <unordered_map>
#include <vector>

namespace forte {

// Defining a function to calculate the conjugate of a value that works both for real/complex fields
template <typename F> F conjugate(const F& value) { return value; }
template <typename Real> std::complex<Real> conjugate(const std::complex<Real>& value) {
    return std::conj(value);
}

// Defining a concept for arithmetic types
template <typename F>
concept Arithmetic = std::is_arithmetic_v<F>;

/// @brief A template class to define a vector space over a field F for a given type T
/// @tparam Derived The derived class
/// @tparam T The type of the vector space
/// @tparam F The field of the vector space
/// @tparam Hash The hash function for the unordered_map
/// @details The class uses an unordered_map to store the elements of the vector space.
/// Here we use the Curiously Recurring Template Pattern (CRTP) to define a template class
/// VectorSpace that supports basic operations for vector spaces over a field F for a given type T.
/// The class is templated over the derived class, the type T, the field F, and an optional hash
/// function.
// The class provides basic operations such as addition, subtraction, scalar multiplication, scalar
/// division, and norm calculation.
/// To use the class, the derived class should be implemented in the following way:
///
/// class Derived : public VectorSpace<Derived, T, F> {
///     // Implement the derived class here
/// };
///
template <typename Derived, typename T, Arithmetic F, typename Hash = std::hash<T>>
class VectorSpace {
  public:
    using container = std::unordered_map<T, F, Hash>;

    // Constructor
    VectorSpace() = default;
    // Copy constructor
    VectorSpace(const VectorSpace& other) : elements_(other.elements_) {}
    // Constructor from a map/dictionary (python friendly)
    VectorSpace(const container& elements) : elements_(elements) {}

    constexpr static F zero_{0};
    constexpr static F small_{1.0e-12};

    /// @return the list of operators
    const container& elements() const { return elements_; }

    inline auto self() { return static_cast<Derived&>(*this); }
    inline auto self() const { return static_cast<const Derived&>(*this); }

    // implement a copy operator
    VectorSpace& operator=(const VectorSpace& other) {
        elements_ = other.elements_;
        return *this;
    }

    void copy(const Derived& other) { elements_ = other.elements_; }

    // implement the move constructor
    VectorSpace(VectorSpace&& other) : elements_(std::move(other.elements_)) {}

    size_t size() const { return elements_.size(); }

    /// @return the beginning of the map
    auto begin() { return elements_.begin(); }
    /// @return the beginning of the map (const)
    auto begin() const { return elements_.begin(); }
    /// @return the end of the map
    auto end() { return elements_.end(); }
    /// @return the end of the map (const)
    auto end() const { return elements_.end(); }

    const F& operator[](const T& e) const {
        auto it = elements_.find(e);
        if (it == elements_.end()) {
            return zero_;
        }
        return it->second;
    }

    inline F& operator[](const T& e) { return elements_[e]; }

    inline F operator()(const T& e) const { return (*this)[e]; }

    /// @return count how many times an element appears in the vector space
    inline auto count(const T& e) const { return elements_.count(e); }

    inline auto find(const T& e) const { return elements_.find(e); }

    F norm(int p = 2) const {
        F result{0};
        // If p is -1, we calculate the infinity norm
        if (p == -1) {
            for (const auto& [_, c] : elements_) {
                result = std::max(result, std::abs(c));
            }
            return result;
        }
        // Otherwise, we calculate the p-norm
        for (const auto& [_, c] : elements_) {
            result += std::pow(std::abs(c), p);
        }
        return std::pow(result, 1. / static_cast<F>(p));
    }

    void add(const T& e, F c) { elements_[e] += c; }

    F remove(const T& e) {
        auto it = elements_.find(e);
        if (it == elements_.end()) {
            return zero_;
        }
        F c = it->second;
        elements_.erase(it);
        return c;
    }

    Derived operator+(const Derived& rhs) const {
        Derived result = self();
        result += rhs;
        return result;
    }

    Derived operator-(const Derived& rhs) const {
        Derived result = self();
        result -= rhs;
        return result;
    }

    Derived operator*(F scalar) const {
        Derived result = self();
        result *= scalar;
        return result;
    }

    Derived operator/(F scalar) const {
        assert(scalar != 0); // Prevent division by zero
        Derived result = self();
        result /= scalar;
        return result;
    }

    Derived& operator+=(const Derived& rhs) {
        for (const auto& [e, c] : rhs.elements_) {
            elements_[e] += c;
        }
        return static_cast<Derived&>(*this);
    }

    Derived& operator-=(const Derived& rhs) {
        for (const auto& [e, c] : rhs.elements_) {
            elements_[e] -= c;
        }
        return static_cast<Derived&>(*this);
    }

    Derived& operator*=(F scalar) {
        for (auto& [_, c] : elements_) {
            c *= scalar;
        }
        return static_cast<Derived&>(*this);
    }

    Derived& operator/=(F scalar) {
        assert(scalar != 0); // Prevent division by zero
        for (auto& [_, c] : elements_) {
            c /= scalar;
        }
        return static_cast<Derived&>(*this);
    }

    bool operator==(const Derived& other) const {
        const double nonzero = 1.0e-14;
        const auto& smaller = size() < other.size() ? elements() : other.elements();
        const auto& larger = size() < other.size() ? other.elements() : elements();

        // edge case: if smaller is empty, we need to make sure the elements in the other operator
        // are zero
        if (smaller.size() == 0) {
            if (larger.size() == 0) {
                return true;
            } else {
                for (const auto& [sqop, c] : larger) {
                    if (std::abs(c) > nonzero) {
                        return false;
                    }
                }
                return true;
            }
        }

        // Check if the two operators have the same terms
        for (const auto& [sqop, c] : smaller) {
            if (larger.find(sqop) == larger.end()) {
                return false;
            }
            if (std::abs(c - larger.at(sqop)) > nonzero) {
                return false;
            }
        }
        return true;
    }

    Derived adjoint() const {
        Derived result;
        for (const auto& [e, c] : elements_) {
            result.add(e.adjoint(), conjugate(c));
        }
        return result;
    }

    F dot(const Derived& other) const {
        F result{0};
        const auto& smaller = size() < other.size() ? elements() : other.elements();
        const auto& larger = size() < other.size() ? other.elements() : elements();
        for (const auto& [e, c] : smaller) {
            if (const auto it = larger.find(e); it != larger.end()) {
                result += conjugate(c) * it->second;
            }
        }
        return result;
    }

    void insert(const T& element, const F& value) { elements_[element] = value; }

  private:
    // Using an unordered_map with a custom hash function
    container elements_;
};

/// @brief A template class to define an ordered list of vector space elements over a field F for a
/// given type T
/// @tparam Derived The derived class
/// @tparam T The type of the vector space
/// @tparam F The field of the vector space
/// @tparam Hash The hash function for the unordered_map
/// @details The class uses a std::vector to store the elements of the list.
/// Here we use the Curiously Recurring Template Pattern (CRTP) to define a template class
/// VectorSpaceList that supports basic operations for vector spaces over a field F for a given type
/// T. The class is templated over the derived class, the type T, the field F, and an optional hash
/// function.
/// To use the class, the derived class should be implemented in the following way:
///
/// class Derived : public VectorSpace<Derived, T, F> {
///     // Implement the derived class here
/// };
///
template <typename Derived, typename T, Arithmetic F> class VectorSpaceList {
  public:
    using container = std::vector<std::pair<T, F>>;

    VectorSpaceList() = default;

    constexpr static F zero_{0};

    /// @return the list of operators
    const container& elements() const { return elements_; }

    // implement the copy constructor
    VectorSpaceList(const VectorSpaceList& other) : elements_(other.elements_) {}

    // implement a copy operator
    VectorSpaceList& operator=(const VectorSpaceList& other) {
        elements_ = other.elements_;
        return *this;
    }

    void copy(const VectorSpaceList& other) { elements_ = other.elements_; }

    // implement the move constructor
    VectorSpaceList(VectorSpaceList&& other) : elements_(std::move(other.elements_)) {}

    size_t size() const { return elements_.size(); }

    const F& operator[](size_t n) const { return elements_[n].second; }

    F& operator[](size_t n) { return elements_[n].second; }

    const auto& operator()(size_t n) const { return elements_[n]; }

    F norm() const {
        F result = 0;
        for (const auto& [e, c] : elements_) {
            result += c * c;
        }
        return result;
    }

    void add(const T& e, F c) { elements_.emplace_back(e, c); }

    Derived& operator*=(F scalar) {
        for (auto& [e, c] : elements_) {
            c *= scalar;
        }
        return static_cast<Derived&>(*this);
    }

    Derived& operator/=(F scalar) {
        assert(scalar != 0); // Prevent division by zero
        for (auto& [e, c] : elements_) {
            c /= scalar;
        }
        return static_cast<Derived&>(*this);
    }

    bool operator==(const VectorSpaceList& rhs) const { return elements_ == rhs.elements_; }

    void insert(const T& e, const F& c) { elements_.emplace_back(e, c); }

    VectorSpaceList adjoint() const {
        VectorSpaceList result;
        for (const auto& [e, c] : elements_) {
            result.insert(e.adjoint(), conjugate(c));
        }
        return result;
    }

  private:
    // Using an unordered_map with a custom hash function
    container elements_;
};

} // namespace forte
