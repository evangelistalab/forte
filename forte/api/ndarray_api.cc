/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2024 by its authors (see LICENSE, AUTHORS).
 *
 * The copyrights for code used from other parties are included in
 * the corresponding files.
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see http://www.gnu.org/licenses/.
 *
 * @END LICENSE
 */

#include <variant>

#include "helpers/tensors/ndarray.hpp"

#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace forte {
// Template specializations of ndarray<T>::dtype
// template <> const DataType ndarray<float>::dtype("float32", 'f', sizeof(float));

// template <> const DataType ndarray<double>::dtype("float64", 'f', sizeof(double));

// template <>
// const DataType ndarray<std::complex<float>>::dtype("complex64", 'c',
// sizeof(std::complex<float>));

// template <>
// const DataType ndarray<std::complex<double>>::dtype("complex128", 'c',
//                                                     sizeof(std::complex<double>));

// Include necessary headers for type traits
#include <complex>
#include <type_traits>

// Define a helper struct to map C++ types to DataType
template <typename T> struct DataTypeTraits;

template <> struct DataTypeTraits<int> {
    static DataType dtype() { return DataType("int32", 'i', sizeof(int)); }
};

template <> struct DataTypeTraits<double> {
    static DataType dtype() { return DataType("float64", 'f', sizeof(double)); }
};

template <> struct DataTypeTraits<float> {
    static DataType dtype() { return DataType("float32", 'f', sizeof(float)); }
};

template <> struct DataTypeTraits<std::complex<float>> {
    static DataType dtype() { return DataType("complex64", 'c', sizeof(std::complex<float>)); }
};

template <> struct DataTypeTraits<std::complex<double>> {
    static DataType dtype() { return DataType("complex128", 'c', sizeof(std::complex<double>)); }
};

// General template for types not explicitly specialized
template <typename T> struct DataTypeTraits {
    static DataType dtype() {
        return DataType(typeid(T).name(), '?', sizeof(T)); // '?' as unknown kind
    }
};

// Now, define the static dtype member using DataTypeTraits
template <typename T> const DataType ndarray<T>::dtype = DataTypeTraits<T>::dtype();

/// @brief Allowed data types
enum class DataTypeEnum { Float32, Float64, Complex64, Complex128 };

DataTypeEnum string_to_dtype(const std::string& dtype) {
    if (dtype == "float32") {
        return DataTypeEnum::Float32;
    } else if (dtype == "float64") {
        return DataTypeEnum::Float64;
    } else if (dtype == "complex64") {
        return DataTypeEnum::Complex64;
    } else if (dtype == "complex128") {
        return DataTypeEnum::Complex128;
    } else {
        throw std::runtime_error("Unknown dtype: " + dtype);
    }
}

void export_ndarray(py::module& m) {
    ndarray<float>::bind(m, "ftensor");
    ndarray<std::complex<float>>::bind(m, "cftensor");
    ndarray<double>::bind(m, "dtensor");
    ndarray<std::complex<double>>::bind(m, "cdtensor");

    py::enum_<DataTypeEnum>(m, "dtype")
        .value("float32", DataTypeEnum::Float32)
        .value("float64", DataTypeEnum::Float64)
        .value("complex64", DataTypeEnum::Complex64)
        .value("complex128", DataTypeEnum::Complex128)
        .export_values();

    m.attr("float32") = DataTypeEnum::Float32;
    m.attr("float64") = DataTypeEnum::Float64;
    m.attr("complex64") = DataTypeEnum::Complex64;
    m.attr("complex128") = DataTypeEnum::Complex128;

    m.def(
        "ndarray",
        [](std::vector<size_t> shape, std::string dtype)
            -> std::variant<ndarray<float>, ndarray<double>, ndarray<std::complex<float>>,
                            ndarray<std::complex<double>>> {
            if (dtype == "float32") {
                return ndarray<float>(shape);
            } else if (dtype == "complex64") {
                return ndarray<std::complex<float>>(shape);
            } else if (dtype == "float64") {
                return ndarray<double>(shape);
            } else if (dtype == "complex128") {
                return ndarray<std::complex<double>>(shape);
            } else {
                throw std::runtime_error("Unknown dtype");
            }
        },
        py::arg("shape"), py::arg("dtype") = "float64");

    m.def(
        "ndarray",
        [](std::vector<size_t> shape, DataTypeEnum dtype)
            -> std::variant<ndarray<float>, ndarray<double>, ndarray<std::complex<float>>,
                            ndarray<std::complex<double>>> {
            if (dtype == DataTypeEnum::Float32) {
                return ndarray<float>(shape);
            } else if (dtype == DataTypeEnum::Complex64) {
                return ndarray<std::complex<float>>(shape);
            } else if (dtype == DataTypeEnum::Float64) {
                return ndarray<double>(shape);
            } else if (dtype == DataTypeEnum::Complex128) {
                return ndarray<std::complex<double>>(shape);
            } else {
                throw std::runtime_error("Unknown dtype");
            }
        },
        py::arg("shape"), py::arg("dtype") = DataTypeEnum::Float64);

    m.def(
        "ndarray_from_numpy",
        [](py::array array)
            -> std::variant<ndarray<float>, ndarray<double>, ndarray<std::complex<float>>,
                            ndarray<std::complex<double>>> {
            auto itemsize = array.dtype().itemsize();
            auto kind = array.dtype().kind();
            if (kind == 'f' and itemsize == sizeof(float)) {
                return ndarray<float>::from_numpy(array);
            } else if (kind == 'f' and itemsize == sizeof(double)) {
                return ndarray<double>::from_numpy(array);
            } else if (kind == 'c' and itemsize == sizeof(std::complex<float>)) {
                return ndarray<std::complex<float>>::from_numpy(array);
            } else if (kind == 'c' and itemsize == sizeof(std::complex<double>)) {
                return ndarray<std::complex<double>>::from_numpy(array);
            } else {
                throw py::type_error("Unknown dtype: kind = " + std::to_string(kind) +
                                     " itemsize = " + std::to_string(itemsize));
            }
        },
        py::arg("array"));

    py::class_<DataType>(m, "DataType")
        .def_property_readonly("name", &DataType::name)
        .def_property_readonly("kind", &DataType::kind)
        .def_property_readonly("itemsize", &DataType::itemsize);
}
} // namespace forte