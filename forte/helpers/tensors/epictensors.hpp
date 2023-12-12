#pragma once

#include <complex>
#include <iostream>
#include <memory>
#include <span>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "datatype.hpp"

namespace py = pybind11;

constexpr bool epictensors_debug = false;

/// @brief Class to represent a tensor
template <typename T> class Tensor {
    /// @brief Shape of tensor
    using shape_t = std::vector<size_t>;
    /// @brief Strides of tensor
    using strides_t = std::vector<size_t>;
    /// @brief Data of tensor
    using container_t = std::vector<T>;

  public:
    Tensor() = default;

    /// @brief Construct a new Tensor object from a shape. This will allocate
    /// the data and the Tensor object will own it.
    /// @param shape
    Tensor(const shape_t& shape) : shape_(shape) {
        size_t size = size_from_shape(shape);
        owned_data_ = std::make_shared<container_t>(size);
        data_span_ = std::span<T>(owned_data_->data(), size);
        compute_strides();
    }

    /// @brief Construct from raw pointer, size, and shape. In this case the
    /// data is not owned by the Tensor object.
    /// @param data pointer to data
    /// @param size size of data
    /// @param shape shape of tensor
    Tensor(T* data, size_t size, shape_t shape_list) : shape_(shape_list), data_span_(data, size) {
        compute_strides();
    }

    /// @brief Return the shape of the tensor
    const auto& shape() const { return shape_; }

    /// @brief Return the size of the tensor
    size_t size() const { return data_span_.size(); }

    /// @brief Return the strides of the tensor
    const auto& strides() const { return strides_; }

    /// @brief Return the data of the tensor at the given indices
    /// @param indices indices of the data to return
    /// @return A reference to the data at the given indices
    T& at(const std::vector<size_t>& indices) {
        size_t index = 0;
        size_t multiplier = 1;
        for (int i = indices.size() - 1; i >= 0; --i) {
            index += indices[i] * multiplier;
            multiplier *= shape_[i];
        }
        return data_span_[index];
    }

    T& operator[](size_t index) { return data_span_[index]; }

    /// @brief Set
    void set_at(const std::vector<size_t>& indices, T value) { at(indices) = value; }

    void set_to(const std::vector<T>& values) {
        if (values.size() != data_span_.size()) {
            throw std::runtime_error("Incorrect number of values");
        }
        std::copy(values.begin(), values.end(), data_span_.begin());
    }

    void fill(const T& value) { std::fill(data_span_.begin(), data_span_.end(), value); }

    /// @brief Return a numpy array view of the tensor
    /// @note This is a zero-copy operation, so the data is not copied. The
    /// Python array view holds a reference to the C++ object, ensuring the data
    /// isn't deleted while Python still has a reference to it. For example, in
    /// python:
    /// ```
    /// t = dtensor([2])
    /// t.at([0, 0]) = 1.0
    /// npt = t.numpy_array()
    /// npt[0, 0] = 7.0
    /// print(t.at([0, 0])) # prints 7.0
    /// del t # this is ok, the data is still owned by the numpy array
    /// print(npt[0, 0]) # prints 7.0
    /// del npt # this is ok, the data is deallocated
    /// ```
    /// @return A numpy array view of the tensor
    py::array_t<T> numpy_array() {
        // pass the shape, data pointer, and a pointer to the class instance
        return py::array_t<T>(shape_, data_span_.data(), py::cast(this));
    }

    /// @brief Return a numpy array view of the tensor to a const object
    py::array_t<T> numpy_array_const() const {
        // pass the shape, data pointer, and a pointer to the class instance
        return py::array_t<T>(shape_, data_span_.data(), py::cast(this));
    }

    /// @brief Construct a Tensor object from a numpy array
    static Tensor<T> from_numpy(py::array_t<T> array) {
        std::vector<size_t> shape(array.shape(), array.shape() + array.ndim());
        Tensor<T> tensor(const_cast<T*>(array.data()), array.size(), shape);
        return tensor;
    }

    std::string to_string() const {
        std::ostringstream oss;
        oss << "Tensor(shape=[";
        for (size_t i = 0; i < shape_.size(); ++i) {
            oss << shape_[i];
            if (i != shape_.size() - 1) {
                oss << ", ";
            }
        }
        oss << "], data=[";
        for (size_t i = 0; const auto& x : data_span_) {
            oss << x;
            if (i != data_span_.size() - 1) {
                oss << ", ";
            }
            i += 1;
        }
        oss << "], strides=[";
        for (size_t i = 0; i < strides_.size(); ++i) {
            oss << strides_[i];
            if (i != strides_.size() - 1) {
                oss << ", ";
            }
        }
        oss << "], data_type=" << typeid(T).name() << "])";
        return oss.str();
    }

    static Tensor<T> einsum(const std::string& equation, const std::vector<Tensor<T>>& tensors) {
        if (tensors.empty()) {
            throw std::runtime_error("No tensors provided to einsum.");
        }

        // Convert all tensors to numpy arrays and store them in a vector
        std::vector<py::array_t<T>> numpy_tensors;
        numpy_tensors.reserve(tensors.size());
        for (const auto& tensor : tensors) {
            numpy_tensors.push_back(tensor.numpy_array_const());
        }

        // Import numpy and call einsum with all numpy arrays at once
        py::object numpy = py::module::import("numpy");
        py::array_t<T> result =
            numpy.attr("einsum")(equation, *numpy_tensors.data()).template cast<py::array_t<T>>();

        return Tensor<T>::from_numpy(result);
    }

  private:
    /// @brief  Data type of tensor
    static const DataType dtype;
    /// @brief Shape of tensor
    shape_t shape_;
    /// @brief Strides of tensor
    strides_t strides_;
    /// @brief Data of tensor (if owned)
    std::shared_ptr<container_t> owned_data_;
    /// @brief Span of the data used to access the data
    std::span<T> data_span_;

    /// @brief Compute the size of a tensor from its shape
    [[nodiscard]] auto size_from_shape(const shape_t& shape) const -> size_t {
        return std::accumulate(shape.begin(), shape.end(), 1u, std::multiplies<>());
    }

    /// @brief Compute the strides of a tensor from its shape
    void compute_strides() {
        strides_.resize(shape_.size());
        size_t stride = 1;
        for (size_t i = shape_.size(); i > 0; --i) {
            strides_[i - 1] = stride;
            stride *= shape_[i - 1];
        }
    }

    template <typename... Indices> T& get(Indices... indices) {
        return data_span_[offset(indices...)];
    }

    template <typename... Indices> size_t offset(Indices... indices) const {
        if constexpr (epictensors_debug) {
            if (sizeof...(indices) != shape_.size()) {
                throw std::out_of_range("Incorrect number of indices");
            }
        }
        return offset_impl(0, indices...);
    }

    size_t offset_impl(size_t) const { return 0; }

    template <typename First, typename... Rest>
    size_t offset_impl(size_t dim, First first, Rest... rest) const {
        if constexpr (epictensors_debug) {
            if (first >= shape_[dim]) {
                throw std::out_of_range("Index out of bounds for dimension");
            }
        }
        return first * strides_[dim] + offset_impl(dim + 1, rest...);
    }

  public:
    static void bind(py::module& m, const std::string& name) {
        py::class_<Tensor<T>>(m, name.c_str())
            .def(py::init<const std::vector<size_t>&>())
            .def("__str__", &Tensor<T>::to_string)
            .def("at", &Tensor<T>::at)
            .def("set_at", &Tensor<T>::set_at)
            .def("set_to", &Tensor<T>::set_to)
            .def("fill", &Tensor<T>::fill)
            // .def("get", &Tensor<T>::get)
            .def("numpy_array", &Tensor<T>::numpy_array)
            .def("__repr__", &Tensor<T>::to_string)
            .def("__str__", &Tensor<T>::to_string)
            // TODO: this is dangerous because we ignore the args and kwargs
            .def("__array__", [](Tensor<T>& self, py::args args,
                                 py::kwargs kwargs) { return self.numpy_array(); })

            .def_property_readonly("shape", &Tensor<T>::shape)
            .def_static("from_numpy", &Tensor<T>::from_numpy)
            .def_property_readonly_static("dtype", [](py::object) { return Tensor<T>::dtype; })
            .def_static("einsum", &Tensor<T>::einsum);
    }
};

template <typename T> std::ostream& operator<<(std::ostream& os, const Tensor<T>& tensor) {
    os << tensor.to_string();
    return os;
}