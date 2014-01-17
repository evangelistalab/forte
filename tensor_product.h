/*
 *@BEGIN LICENSE
 *
 * Basic Tensor Library: a library to perform common tensor operations
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with this program; if not, write to the Free Software Foundation, Inc.,
 * 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 *
 *@END LICENSE
 */


#ifndef _tensor_product_h_
#define _tensor_product_h_

#include <string>
#include <vector>

class Tensor;
class LabeledTensor;

/// A product of tensors labeled with indices.
/// A TensorProduct object can hold an arbitrary number of tensors.
///
/// Sample usage:
///    A("pqab") * B("abrs") * C("pqrs");
class LabeledTensorProduct
{
public:
    LabeledTensorProduct() {};
    LabeledTensorProduct(std::initializer_list<LabeledTensor> l) : tensors_(l) {}

    /// Return the n-th labeled tensor
    LabeledTensor tensor(size_t n);

    /// Append one labeled tensor
    void append(LabeledTensor tl);
    /// Append a list of labeled tensors
    void append(std::initializer_list<LabeledTensor> l);

    /// Compute the flop and memory costs of performing a contraction
    /// of this tensor product using the factorization pattern defined by perm
    std::pair<double,double> compute_contraction_cost(std::vector<size_t> perm);

    void print();

    /// Multiply two tensors with indices
    LabeledTensorProduct operator*(LabeledTensor lhs);

    size_t size() {return tensors_.size();}
private:
    std::vector<LabeledTensor> tensors_;
};

#endif // _tensor_product_h_
