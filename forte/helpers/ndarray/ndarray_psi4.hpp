#pragma once

#include "ndarray.hpp"

namespace forte {

/// @brief A function to convert a numpy array to a Psi4 Vector
/// This function will turn a matrix with symmetry into a full matrix
template <typename T> ndarray<T> from_numpy(const psi::Matrix& m) {
    // determine the shape of the matrix
    std::vector<size_t> shape = {m.nrows(), m.ncols()};
}

ndarray<double> psi_vector_to_ndarray(const psi::Vector& v) {
    std::vector<size_t> shape = {v.dimpi().sum()};
    ndarray<double>::copy_from_pointer(v.pointer(), shape);

    for (size_t i = 0; i < v.dim(); ++i) {
        a.at(i) = v.get(i);
    }
    return a;
}

std::shared_ptr<psi::Vector> ndarray_to_psi_vector(const ndarray<double>& a) {
    if (a.rank() != 1) {
        throw std::runtime_error("Unable to convert: Tensor rank is not 1!");
    }
    const auto& shape = a.shape();
    auto v = std::make_shared<psi::Vector>("V", shape[0]);
    for (size_t i = 0; i < shape[0]; ++i) {
        v->set(i, a.at(i));
    }
    return v;
}

std::shared_ptr<psi::Matrix> ndarray_to_psi_matrix(const ndarray<double>& a) {
    if (a.rank() != 2) {
        throw std::runtime_error("Unable to convert: Tensor rank is not 2!");
    }
    const auto& shape = a.shape();
    auto m = std::make_shared<psi::Matrix>("M", shape[0], shape[1]);
    for (size_t i = 0; i < shape[0]; ++i) {
        for (size_t j = 0; j < shape[1]; ++j) {
            m->set(i, j, a.at(i, j));
        }
    }
    return m;
}

// std::shared_ptr<psi::Matrix> tensor_to_matrix(const ndarray<double>& a, psi::Dimension rowdims,
//                                               psi::Dimension coldims = psi::Dimension()) {
//     if (a.rank() != 2) {
//         throw std::runtime_error("Unable to convert: Tensor rank is not 2!");
//     }
//     const auto& shape = a.shape();

//     if (coldims.sum() == 0) {
//         coldims = rowdims;
//     }

//     auto m = std::make_shared<psi::Matrix>("M", rowdims, coldims);
//     size_t sym = 0;

//     auto nirrep = static_cast<size_t>(rowdims.n());

//     // Loop over the blocks of this matrix
//     for (size_t hr = 0, row_offset = 0; col_offset = 0; hr < nirrep; ++hr) {
//         size_t hc = sym ^ hr;
//         auto rowsize = static_cast<size_t>(rowdims[hr]);
//         auto colsize = static_cast<size_t>(coldims[hc]);
//         for (size_t p = 0; p < rowsize; ++p) {
//             auto np = p + row_offset;
//             for (size_t q = 0; q < colsize; ++q) {
//                 auto nq = q + col_offset;
//                 m->set(hr, p, q, a.at(np, nq));
//             }
//         }
//         row_offset += rowdims[hr];
//         col_offset += coldims[hc];
//     }
//     return m;
// }

} // namespace forte