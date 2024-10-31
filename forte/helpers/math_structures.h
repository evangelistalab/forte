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
template <typename Derived, typename T, typename F, typename Hash = std::hash<T>>
class VectorSpace {
  public:
    using container = std::unordered_map<T, F, Hash>;

    /// @brief Constructor
    VectorSpace() = default;
    /// @brief Copy constructor
    VectorSpace(const VectorSpace& other) : elements_(other.elements_) {}
    /// @brief Constructor from a map/dictionary (python friendly)
    VectorSpace(const container& elements) : elements_(elements) {}
    /// Move constructor
    VectorSpace(VectorSpace&& other) : elements_(std::move(other.elements_)) {}

    /// @brief Zero element of the field
    constexpr static F zero_{0};
    /// @brief Small number for comparison
    constexpr static F small_{1.0e-12};

    /// @return the list of operators
    const container& elements() const { return elements_; }

    /// @return convert this object to the derived class
    inline auto self() { return static_cast<Derived&>(*this); }
    /// @return convert this object to the derived class (const)
    inline auto self() const { return static_cast<const Derived&>(*this); }

    /// @brief Copy operator
    VectorSpace& operator=(const VectorSpace& other) {
        elements_ = other.elements_;
        return *this;
    }

    /// @brief Copy function
    void copy(const Derived& other) { elements_ = other.elements_; }

    /// @return the number of elements in the vector space
    size_t size() const { return elements_.size(); }

    /// @return an iterator to the beginning of the object
    inline auto begin() { return elements_.begin(); }
    /// @return an iterator to the beginning of the object (const)
    inline auto begin() const { return elements_.begin(); }
    /// @return an iterator to the end of the object
    inline auto end() { return elements_.end(); }
    /// @return an iterator to the end of the object (const)
    inline auto end() const { return elements_.end(); }

    /// @return the element corresponding to the key e
    const F& operator[](const T& e) const {
        auto it = elements_.find(e);
        if (it == elements_.end()) {
            return zero_;
        }
        return it->second;
    }

    /// @return the element corresponding to the key e
    inline F& operator[](const T& e) { return elements_[e]; }

    /// @return a copy the element corresponding to the key e
    inline F operator()(const T& e) const { return (*this)[e]; }

    /// @return count how many times an element appears in the vector space
    inline auto count(const T& e) const { return elements_.count(e); }

    /// @return find an element in the vector space
    inline auto find(const T& e) const { return elements_.find(e); }

    /// @return the norm of the vector space
    /// @param p the norm to calculate (default is 2, -1 is infinity norm)
    double norm(int p = 2) const {
        double result{0};
        // If p is -1, we calculate the infinity norm
        if (p == -1) {
            for (const auto& [_, c] : elements_) {
                result = std::max(std::abs(result), std::abs(c));
            }
            return result;
        }
        // Otherwise, we calculate the p-norm
        for (const auto& [_, c] : elements_) {
            result += std::pow(std::abs(c), p);
        }
        return std::pow(result, 1. / static_cast<double>(p));
    }

    /// @brief Add an element to the vector space
    void add(const T& e, F c) { elements_[e] += c; }

    /// @brief Remove an element from the vector space
    F remove(const T& e) {
        auto it = elements_.find(e);
        if (it == elements_.end()) {
            return zero_;
        }
        F c = it->second;
        elements_.erase(it);
        return c;
    }

    /// @brief Negate a vector
    Derived operator-() const {
        Derived result = self();
        result *= -1;
        return result;
    }

    /// @brief Add two vectors
    Derived operator+(const Derived& rhs) const {
        Derived result = self();
        result += rhs;
        return result;
    }

    /// @brief Subtract two vectors
    Derived operator-(const Derived& rhs) const {
        Derived result = self();
        result -= rhs;
        return result;
    }

    /// @brief Multiply a vector by a scalar
    /// @param scalar the scalar to multiply by
    /// @return the result of the multiplication
    Derived operator*(F scalar) const {
        Derived result = self();
        result *= scalar;
        return result;
    }

    /// @brief Divide a vector by a scalar
    /// @param scalar the scalar to divide by
    /// @return the result of the division
    Derived operator/(F scalar) const {
        assert(scalar != F(0)); // Prevent division by zero
        Derived result = self();
        result /= scalar;
        return result;
    }

    /// @brief Add two vectors
    Derived& operator+=(const Derived& rhs) {
        for (const auto& [e, c] : rhs.elements_) {
            elements_[e] += c;
        }
        return static_cast<Derived&>(*this);
    }

    /// @brief Subtract two vectors
    Derived& operator-=(const Derived& rhs) {
        for (const auto& [e, c] : rhs.elements_) {
            elements_[e] -= c;
        }
        return static_cast<Derived&>(*this);
    }

    /// @brief Multiply a vector by a scalar
    Derived& operator*=(F scalar) {
        for (auto& [_, c] : elements_) {
            c *= scalar;
        }
        return static_cast<Derived&>(*this);
    }

    /// @brief Divide a vector by a scalar
    Derived& operator/=(F scalar) {
        assert(scalar != F(0)); // Prevent division by zero
        for (auto& [_, c] : elements_) {
            c /= scalar;
        }
        return static_cast<Derived&>(*this);
    }

    /// @brief Check if two vectors are equal
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

    /// @brief Get the adjoint of the vector
    Derived adjoint() const {
        Derived result;
        for (const auto& [e, c] : elements_) {
            result.add(e.adjoint(), conjugate(c));
        }
        return result;
    }

    /// @brief Calculate the dot product of two vectors
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
template <typename Derived, typename T, typename F> class VectorSpaceList {
  public:
    using container = std::vector<std::pair<T, F>>;

    /// @brief Constructor
    VectorSpaceList() = default;
    /// @brief Copy constructor
    VectorSpaceList(const VectorSpaceList& other) : elements_(other.elements_) {}
    /// Move constructor
    VectorSpaceList(VectorSpaceList&& other) : elements_(std::move(other.elements_)) {}

    /// @brief Zero element of the field
    constexpr static F zero_{0};

    /// @return the list of operators
    const container& elements() const { return elements_; }

    /// @brief Copy operator
    VectorSpaceList& operator=(const VectorSpaceList& other) {
        elements_ = other.elements_;
        return *this;
    }

    /// @brief Copy function
    void copy(const VectorSpaceList& other) { elements_ = other.elements_; }

    /// @return the number of elements in the vector
    size_t size() const { return elements_.size(); }

    /// @return an element of the vector
    const F& operator[](size_t n) const { return elements_[n].second; }

    /// @return an element of the vector
    F& operator[](size_t n) { return elements_[n].second; }

    /// @return an element of the vector
    const auto& operator()(size_t n) const { return elements_[n]; }

    /// @return the norm of the vector space
    /// @param p the norm to calculate (default is 2, -1 is infinity norm)
    double norm(int p = 2) const {
        double result{zero_};
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
        return std::pow(result, 1. / static_cast<double>(p));
    }

    /// @brief Add an element to the vector space
    void add(const T& e, const F& c) { elements_.emplace_back(e, c); }

    /// @brief Multiply a vector by a scalar
    /// @param scalar
    /// @return the result of the multiplication
    Derived& operator*=(F scalar) {
        for (auto& [e, c] : elements_) {
            c *= scalar;
        }
        return static_cast<Derived&>(*this);
    }

    /// @brief Divide a vector by a scalar
    /// @param scalar
    /// @return the result of the division
    Derived& operator/=(F scalar) {
        assert(scalar != F(0)); // Prevent division by zero
        for (auto& [e, c] : elements_) {
            c /= scalar;
        }
        return static_cast<Derived&>(*this);
    }

    /// @brief Check if two vectors lists are equal
    bool operator==(const VectorSpaceList& rhs) const { return elements_ == rhs.elements_; }

    /// @brief  Get the adjoint of the vector
    VectorSpaceList adjoint() const {
        VectorSpaceList result;
        for (const auto& [e, c] : elements_) {
            result.add(e.adjoint(), conjugate(c));
        }
        return result;
    }

  private:
    // Using an unordered_map with a custom hash function
    container elements_;
};

} // namespace forte
