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
using namespace pybind11::literals;

constexpr bool ndarray_debug = true;

namespace forte {

/// @brief Class to represent a tensor
template <typename T> class ndarray {
    /// @brief Shape of tensor
    using shape_t = std::vector<size_t>;
    /// @brief Strides of tensor
    using strides_t = std::vector<size_t>;
    /// @brief Data of tensor
    using container_t = std::vector<T>;

  public:
    ndarray() = default;

    /// @brief Construct a new ndarray object from a shape. This will allocate
    /// the data and the ndarray object will own it. The data is initialized to
    /// zero.
    /// @param shape
    ndarray(const shape_t& shape) : shape_(shape) {
        size_t size = size_from_shape(shape);
        owned_data_ = std::make_shared<container_t>(size);
        data_span_ = std::span<T>(owned_data_->data(), size);
        compute_strides();
    }

    /// @brief Construct from raw pointer, size, and shape. In this case the
    /// data is not owned by the ndarray object.
    /// @param data pointer to data
    /// @param size size of data
    /// @param shape shape of tensor
    ndarray(T* data, shape_t shape) : shape_(shape) {
        size_t size = size_from_shape(shape);
        data_span_ = std::span<T>(data, size);
        compute_strides();
    }

    /// @brief Return the shape of the tensor
    const auto& shape() const { return shape_; }

    /// @brief Return the size of the tensor
    size_t size() const { return data_span_.size(); }

    /// @brief Return the strides of the tensor
    const auto& strides() const { return strides_; }

    /// @brief Return the rank of the tensor
    size_t rank() const { return shape_.size(); }

    /// @brief Return the data of the tensor at the given indices
    /// @param indices indices of the data to return
    /// @return A reference to the data at the given indices
    T& at(const std::vector<size_t>& indices) {
        if constexpr (ndarray_debug) {
            if (indices.size() != shape_.size()) {
                throw std::out_of_range("Incorrect number of indices. Expected " +
                                        std::to_string(shape_.size()) + " but got " +
                                        std::to_string(indices.size()));
            }
        }
        size_t index = 0;
        for (int i = indices.size() - 1; i >= 0; --i) {
            if constexpr (ndarray_debug) {
                if (indices[i] >= shape_[i]) {
                    throw std::out_of_range("Index " + std::to_string(indices[i]) +
                                            " out of bounds for dimension " + std::to_string(i) +
                                            " with size " + std::to_string(shape_[i]));
                }
            }
            index += indices[i] * strides_[i];
        }
        return data_span_[index];
    }

    /// @brief Return the data of the tensor at the given indices
    /// @param indices indices of the data to return
    /// @return A reference to the data at the given indices
    template <typename... Indices> T& at(Indices... indices) {
        // return data_span_[offset(indices...)];
        size_t index = calculate_index_variadic(0, indices...);
        return data_span_[index];
    }

    /// @brief Return the data of the tensor at the given flat index
    T& operator[](size_t index) { return data_span_[index]; }

    /// @brief Set the value of a tensor element at the given indices
    void set_at(const std::vector<size_t>& indices, T value) { at(indices) = value; }

    /// @brief Set the values of the tensor to a vector of values
    void set_to(const std::vector<T>& values) {
        if (values.size() != data_span_.size()) {
            throw std::runtime_error("Incorrect number of values");
        }
        std::copy(values.begin(), values.end(), data_span_.begin());
    }

    /// @brief Fill the tensor with a value
    void fill(const T& value) { std::fill(data_span_.begin(), data_span_.end(), value); }

    /// @brief Return a numpy array view of the tensor
    /// @note This is a zero-copy operation, so the data is not copied. The
    /// Python array view holds a reference to the C++ object, ensuring the data
    /// isn't deleted while Python still has a reference to it. For example, in
    /// python:
    /// ```
    /// t = ndarray([2,2])
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

    /// @brief Construct a ndarray object from a numpy array
    static ndarray<T> from_numpy(py::array_t<T> array) {
        // Request buffer info
        py::buffer_info info = array.request();
        if (info.readonly) {
            throw std::runtime_error("Cannot create ndarray from a read-only NumPy array.");
        }
        std::vector<size_t> shape(info.shape.begin(), info.shape.end());
        T* data_ptr = static_cast<T*>(info.ptr);
        ndarray<T> tensor(data_ptr, shape);
        return tensor;
    }

    /// @brief Create a new ndarray that contains a copy of a numpy array
    static ndarray<T> copy_from_numpy(py::array_t<T> array) {
        // Request buffer info
        py::buffer_info info = array.request();
        std::vector<size_t> shape(info.shape.begin(), info.shape.end());
        ndarray<T> tensor(shape);
        std::copy(static_cast<T*>(info.ptr), static_cast<T*>(info.ptr) + array.size(),
                  tensor.data_span_.begin());
        return tensor;
    }

    /// @brief Create a new ndarray that contains a copy of the data passed via a raw pointer
    static ndarray<T> copy_from_pointer(const T* const data, const std::vector<size_t>& shape) {
        ndarray<T> tensor(shape);
        std::copy(data, data + tensor.size(), tensor.data_span_.begin());
        return tensor;
    }

    /// @brief Return a string representation of the tensor
    std::string to_string() const {
        std::ostringstream oss;
        oss << "ndarray(shape=[";
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
        oss << "], data_type=" << ndarray<T>::dtype.to_string() << "])";
        return oss.str();
    }

  private:
    /// @brief  Data type of tensor
    static const DataType dtype;
    /// @brief Shape of tensor
    shape_t shape_;
    /// @brief Strides of tensor
    strides_t strides_;
    /// @brief Span of the data used to access the data
    std::span<T> data_span_;
    /// @brief Data of tensor (if owned)
    std::shared_ptr<container_t> owned_data_;

    /// @brief Compute the size of a tensor from its shape
    /// @param shape Shape of the tensor
    /// @return Size of the tensor
    /// @note This function can even handle zero dimensional tensors (empty shapes), in which case
    /// the returned size is 1.
    [[nodiscard]] auto static size_from_shape(const shape_t& shape) -> size_t {
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

    /// @brief Return the data of the tensor at the given indices
    /// @param indices indices of the data to return
    /// @return A reference to the data at the given indices
    template <typename First, typename... Rest>
    size_t calculate_index_variadic(size_t dim, First first, Rest... rest) const {
        if constexpr (ndarray_debug) {
            if (dim >= shape_.size()) {
                throw std::out_of_range("Too many indices");
            }
            if (first >= shape_[dim]) {
                throw std::out_of_range("Index " + std::to_string(first) +
                                        " out of bounds for dimension " + std::to_string(dim) +
                                        " with size " + std::to_string(shape_[dim]));
            }
        }
        size_t offset = first * strides_[dim];
        if constexpr (sizeof...(rest) == 0) {
            return offset;
        } else {
            return offset + calculate_index_variadic(dim + 1, rest...);
        }
    }

  public:
    static void bind(py::module& m, const std::string& name) {
        py::class_<ndarray<T>>(m, name.c_str())
            .def(py::init<const std::vector<size_t>&>())
            .def_static("from_numpy", &ndarray<T>::from_numpy)
            .def_static("copy_from_numpy", &ndarray<T>::copy_from_numpy)
            .def_property_readonly("rank", &ndarray<T>::rank, "Return the rank of the tensor")
            .def_property_readonly(
                "shape", &ndarray<T>::shape,
                "Return the shape of the tensor (number of elements in each dimension)")
            .def_property_readonly("size", &ndarray<T>::size,
                                   "Return the size of the tensor (number of elements)")
            .def_property_readonly("strides", &ndarray<T>::strides,
                                   "Return the strides of the tensor")
            .def_property_readonly_static("dtype", [](py::object) { return ndarray<T>::dtype; })
            .def("__repr__", &ndarray<T>::to_string, "Return a string representation of the tensor")
            .def("__str__", &ndarray<T>::to_string, "Return a string representation of the tensor")
            .def("__len__", &ndarray<T>::size, "Return the size of the tensor")
            // TODO: this is dangerous because we ignore the args and kwargs
            .def(
                "__array__",
                [](ndarray<T>& self, py::args, py::kwargs) { return self.numpy_array(); },
                "Return a numpy array view of the tensor")
            .def(
                "__getitem__",
                [](ndarray<T>& self, py::tuple index) -> T& {
                    std::vector<size_t> indices;
                    for (auto item : index) {
                        indices.push_back(item.cast<size_t>());
                    }
                    return self.at(indices);
                },
                "Get the value at the given indices")
            .def(
                "__getitem__",
                [](ndarray<T>& self, size_t index) -> T& { return self.at({index}); },
                "Get the value at the given index")
            .def(
                "__setitem__",
                [](ndarray<T>& self, py::tuple index, T value) {
                    std::vector<size_t> indices;
                    for (auto item : index) {
                        indices.push_back(item.cast<size_t>());
                    }
                    self.set_at(indices, value);
                },
                "Set the value at the given indices")
            .def("__setitem__",
                 [](ndarray<T>& self, size_t index, T value) { self.set_at({index}, value); })
            // iterator support
            .def(
                "__iter__",
                [](ndarray<T>& self) {
                    return py::make_iterator(self.data_span_.begin(), self.data_span_.end());
                },
                py::keep_alive<0, 1>())
            .def("numpy_array", &ndarray<T>::numpy_array, "Return a numpy array view of the tensor")
            .def("fill", &ndarray<T>::fill, "value"_a, "Fill the tensor with a value")
            .def("set_at", &ndarray<T>::set_at, "indices"_a, "value"_a,
                 "Set the value at the given indices (passed as a list)")
            .def("set_to", &ndarray<T>::set_to, "values"_a,
                 "Set the values of the tensor to a vector of values")
            // addressing functions
            .def("at",
                 [](ndarray<T>& self, const std::vector<size_t>& indices) -> T& {
                     return self.at(indices);
                 })
            .def(
                "at",
                [](ndarray<T>& self, size_t i1) -> T& {
                    if constexpr (ndarray_debug) {
                        if (self.rank() != 1) {
                            throw std::out_of_range(
                                "Rank " + std::to_string(self.rank()) +
                                " array addressed with incorrect number of indices (1)");
                        }
                    }
                    return self.at(i1);
                },
                "Get the value at the given index")
            .def(
                "at",
                [](ndarray<T>& self, size_t i1, size_t i2) -> T& {
                    if constexpr (ndarray_debug) {
                        if (self.rank() != 2) {
                            throw std::out_of_range(
                                "Rank " + std::to_string(self.rank()) +
                                " array addressed with incorrect number of indices (2)");
                        }
                    }
                    return self.at(i1, i2);
                },
                "Get the value at the given indices")
            .def(
                "at",
                [](ndarray<T>& self, size_t i1, size_t i2, size_t i3) -> T& {
                    if constexpr (ndarray_debug) {
                        if (self.rank() != 3) {
                            throw std::out_of_range(
                                "Rank " + std::to_string(self.rank()) +
                                " array addressed with incorrect number of indices (3)");
                        }
                    }
                    return self.at(i1, i2, i3);
                },
                "Get the value at the given indices")
            .def(
                "at",
                [](ndarray<T>& self, size_t i1, size_t i2, size_t i3, size_t i4) -> T& {
                    if constexpr (ndarray_debug) {
                        if (self.rank() != 4) {
                            throw std::out_of_range(
                                "Rank " + std::to_string(self.rank()) +
                                " array addressed with incorrect number of indices (4)");
                        }
                    }
                    return self.at(i1, i2, i3, i4);
                },
                "Get the value at the given indices")
            .def(
                "at",
                [](ndarray<T>& self, size_t i1, size_t i2, size_t i3, size_t i4, size_t i5) -> T& {
                    if constexpr (ndarray_debug) {
                        if (self.rank() != 5) {
                            throw std::out_of_range(
                                "Rank " + std::to_string(self.rank()) +
                                " array addressed with incorrect number of indices (5)");
                        }
                    }

                    return self.at(i1, i2, i3, i4, i5);
                },
                "Get the value at the given indices")
            .def(
                "at",
                [](ndarray<T>& self, size_t i1, size_t i2, size_t i3, size_t i4, size_t i5,
                   size_t i6) -> T& {
                    if constexpr (ndarray_debug) {
                        if (self.rank() != 6) {
                            throw std::out_of_range(
                                "Rank " + std::to_string(self.rank()) +
                                " array addressed with incorrect number of indices (6)");
                        }
                    }
                    return self.at(i1, i2, i3, i4, i5, i6);
                },
                "Get the value at the given indices")
            .def(
                "at",
                [](ndarray<T>& self, size_t i1, size_t i2, size_t i3, size_t i4, size_t i5,
                   size_t i6, size_t i7) -> T& {
                    if constexpr (ndarray_debug) {
                        if (self.rank() != 7) {
                            throw std::out_of_range(
                                "Rank " + std::to_string(self.rank()) +
                                " array addressed with incorrect number of indices (7)");
                        }
                    }
                    return self.at(i1, i2, i3, i4, i5, i6, i7);
                },
                "Get the value at the given indices")
            .def(
                "at",
                [](ndarray<T>& self, size_t i1, size_t i2, size_t i3, size_t i4, size_t i5,
                   size_t i6, size_t i7, size_t i8) -> T& {
                    if constexpr (ndarray_debug) {
                        if (self.rank() != 8) {
                            throw std::out_of_range(
                                "Rank " + std::to_string(self.rank()) +
                                " array addressed with incorrect number of indices (8)");
                        }
                    }

                    return self.at(i1, i2, i3, i4, i5, i6, i7, i8);
                },
                "Get the value at the given indices")
            .def(
                "at",
                [](ndarray<T>& self, size_t i1, size_t i2, size_t i3, size_t i4, size_t i5,
                   size_t i6, size_t i7, size_t i8, size_t i9) -> T& {
                    if constexpr (ndarray_debug) {
                        if (self.rank() != 9) {
                            throw std::out_of_range(
                                "Rank " + std::to_string(self.rank()) +
                                " array addressed with incorrect number of indices (9)");
                        }
                    }
                    return self.at(i1, i2, i3, i4, i5, i6, i7, i8, i9);
                },
                "Get the value at the given indices")
            .def(
                "at",
                [](ndarray<T>& self, size_t i1, size_t i2, size_t i3, size_t i4, size_t i5,
                   size_t i6, size_t i7, size_t i8, size_t i9, size_t i10) -> T& {
                    if constexpr (ndarray_debug) {
                        if (self.rank() != 10) {
                            throw std::out_of_range(
                                "Rank " + std::to_string(self.rank()) +
                                " array addressed with incorrect number of indices (10)");
                        }
                    }
                    return self.at(i1, i2, i3, i4, i5, i6, i7, i8, i9, i10);
                },
                "Get the value at the given indices");
    }
};

/// @brief Output operator for tensor
template <typename T> std::ostream& operator<<(std::ostream& os, const ndarray<T>& a) {
    os << a.to_string();
    return os;
}

} // namespace forte