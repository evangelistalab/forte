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


#ifndef _tensor_labeled_h_
#define _tensor_labeled_h_

#include <string>
#include <vector>

class Tensor;
class LabeledTensorProduct;

/// Represent a tensor labeled with indices.
/// This class does not own the Tensor, it only holds a pointer.
///
/// Sample usage:
///    C("pqrs") = 0.5 * A("pqab") * B("abrs");
class LabeledTensor
{
public:
    LabeledTensor(const std::vector<std::string> indices,Tensor* tensor) :
        factor_(1.0), indices_(indices), tensor_(tensor) {};
    LabeledTensor(const double factor, const std::vector<std::string> indices,Tensor* tensor) :
        factor_(factor), indices_(indices), tensor_(tensor) {};

    double factor() {return factor_;}
    std::vector<std::string> indices() const {return indices_;}
    Tensor* tensor() {return tensor_;}
    Tensor* tensor() const {return tensor_;}

    /// Return a string representing the indexed tensor
    void print();
    /// Return a string representing the indexed tensor
    std::string str() const;
    /// Multiply two tensors with indices
    LabeledTensorProduct operator*(LabeledTensor lhs);
    /// Add a product of two tensors to a tensor with indices (contraction)
    void operator+=(LabeledTensorProduct tp);
    /// Set this equal to a product of two tensors (contraction)
    void operator=(LabeledTensorProduct tp);
    /// Add a tensor to this tensors (copy)
    void operator+=(LabeledTensor lhs);
    /// Set this tensor equal to another tensors (copy)
    void operator=(LabeledTensor lhs);
private:
    double factor_;
    std::vector<std::string> indices_;
    Tensor* tensor_;
};

LabeledTensor operator*(double factor,LabeledTensor ti);

#endif // _tensor_h_

