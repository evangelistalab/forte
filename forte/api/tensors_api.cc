/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * t    hat implements a variety of quantum chemistry methods for strongly
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

#include "helpers/tensors/epictensors.hpp"

#include <pybind11/stl.h>

// Template specializations of Tensor<T>::dtype
template <> const DataType Tensor<float>::dtype("float32", 'f', sizeof(float));

template <> const DataType Tensor<double>::dtype("float64", 'd', sizeof(double));

template <>
const DataType Tensor<std::complex<float>>::dtype("complex64", 'c', sizeof(std::complex<float>));

template <>
const DataType Tensor<std::complex<double>>::dtype("complex128", 'c', sizeof(std::complex<double>));

namespace forte {

/// @brief Allowed data types
enum class DataTypeEnum { Float32, Float64, Complex64, Complex128 };

void export_EPICTensors(py::module& m) {
    Tensor<float>::bind(m, "ftensor");
    Tensor<std::complex<float>>::bind(m, "cftensor");
    Tensor<double>::bind(m, "dtensor");
    Tensor<std::complex<double>>::bind(m, "cdtensor");

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
        "array",
        [](std::vector<size_t> shape, std::string dtype)
            -> std::variant<Tensor<float>, Tensor<double>, Tensor<std::complex<float>>,
                            Tensor<std::complex<double>>> {
            if (dtype == "float32") {
                return Tensor<float>(shape);
            } else if (dtype == "complex64") {
                return Tensor<std::complex<float>>(shape);
            } else if (dtype == "float64") {
                return Tensor<double>(shape);
            } else if (dtype == "complex128") {
                return Tensor<std::complex<double>>(shape);
            } else {
                throw std::runtime_error("Unknown dtype");
            }
        },
        py::arg("shape"), py::arg("dtype") = "float64");

    m.def(
        "array",
        [](std::vector<size_t> shape, DataTypeEnum dtype)
            -> std::variant<Tensor<float>, Tensor<double>, Tensor<std::complex<float>>,
                            Tensor<std::complex<double>>> {
            if (dtype == DataTypeEnum::Float32) {
                return Tensor<float>(shape);
            } else if (dtype == DataTypeEnum::Complex64) {
                return Tensor<std::complex<float>>(shape);
            } else if (dtype == DataTypeEnum::Float64) {
                return Tensor<double>(shape);
            } else if (dtype == DataTypeEnum::Complex128) {
                return Tensor<std::complex<double>>(shape);
            } else {
                throw std::runtime_error("Unknown dtype");
            }
        },
        py::arg("shape"), py::arg("dtype") = DataTypeEnum::Float64);

    py::class_<DataType>(m, "DataType")
        .def_property_readonly("name", &DataType::name)
        .def_property_readonly("kind", &DataType::kind)
        .def_property_readonly("itemsize", &DataType::itemsize);
}
} // namespace forte