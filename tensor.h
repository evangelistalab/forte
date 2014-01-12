/*
 *@BEGIN LICENSE
 *
 * Libadaptive: an ab initio quantum chemistry software package
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

#include <vector>
#include <string>

#ifndef _tensor_h_
#define _tensor_h_

class TensorIndexed;
class TensorProduct;

class Tensor
{
public:
    Tensor();
    Tensor(std::string label,std::vector<size_t> dims);
    ~Tensor();

    TensorIndexed operator()(std::string indices);

    void resize(std::vector<size_t> dims);
    void resize(std::string label,std::vector<size_t> dims);

    double& operator()(size_t i1,size_t i2);
    const double& operator()(size_t i1,size_t i2) const;
    double& operator()(size_t i1,size_t i2,size_t i3,size_t i4);
    const double& operator()(size_t i1,size_t i2,size_t i3,size_t i4) const;

    /// Return the number of tensor components
    size_t ndims() {return ndims_;}
    /// Return the dimensions of the tensor components
    std::vector<size_t>& dims() {return dims_;}
    /// Return the dimensions of the tensor components
    size_t dims(size_t n) {return dims_[n];}
    /// Return the tensor data
    double* t() {return t_;}
    /// Return the tensor label
    std::string& label() {return label_;}

    // Operations on the tensor
    /// Zero all the tensor elements
    void zero();
    /// Compute the k-norm of the vector.  The default is the 2-norm.
    double norm(int power = 2);

    /// Contract two tensors
    /// Performs either C += A * B
    ///              or C  = A * B
    /// @param A
    /// @param B
    /// @param C
    /// @param addition Add A * B to C?
    static void evaluate(TensorIndexed A, TensorIndexed B, TensorIndexed C, bool addition);

    /// Add or copy a tensor
    /// Performs either B += A
    ///              or B  = A
    static void add(TensorIndexed A, TensorIndexed B, bool addition);

    /// Functions that deal with the temporary data
    static void initialize_class(size_t nmo);
    static void finalize_class();
private:
    void allocate(std::vector<size_t> dims);
    void release();

    size_t one_address(size_t i0) const;
//    size_t two_address(size_t i0,size_t i1) const;
    size_t two_address(size_t i0,size_t i1) const {
        return(i0 * add_[0] + i1);
    }
    size_t three_address(size_t i0,size_t i1,size_t i2) const;
    size_t four_address(size_t i0,size_t i1,size_t i2,size_t i3) const {
        return(i0 * add_[0] + i1 * add_[1] + i2 * add_[2] + i3);
    }

    /// The label of the tensor
    std::string label_;
    /// The number of tensor components
    size_t ndims_;
    /// The dimensions of the tensor components
    std::vector<size_t> dims_;
    /// An addressing array
    std::vector<size_t> add_;
    /// The number of the tensor elements
    size_t nelements_;
    /// The tensor data
    double* t_;

    static std::pair<size_t,size_t> tensor_to_matrix_sort(TensorIndexed T,
                               std::vector<std::string> T_left,
                               std::vector<std::string> T_right,
                               double* t, bool direct);

    // Class static functions
    /// Set the use of DGEMM for tensor contractions
    void set_use_dgemm(bool value) {use_dgemm_ = value;}
    void set_print(int value) {print_ = value;}

    template<typename Sorter>
    void sort_me(std::vector<size_t> itoj,double*& matrix,bool direct,Sorter sorter);

    /// Temporary arrays
    static double* tA;
    static double* tB;
    static double* tC;
    static double* tD;
    static size_t nwork_;

    // Class options
    ///  Use of DGEMM for tensor contractions?
    static bool use_dgemm_;
    /// The print level of the Tensor class
    /// 0 : no printing
    /// 1 : print contraction information
    /// 2 : print contraction and sorting information
    /// 3 : print everything
    static int print_;
};


class TensorProduct;

/// This class is used to represent a tensor with a specific
/// indexing to be used in a tensor operation
class TensorIndexed
{
public:
    TensorIndexed(const std::vector<std::string> indices,Tensor* tensor) :
        factor_(1.0), indices_(indices), tensor_(tensor) {};
    TensorIndexed(const double factor, const std::vector<std::string> indices,Tensor* tensor) :
        factor_(factor), indices_(indices), tensor_(tensor) {};

    double factor() {return factor_;}
    std::vector<std::string> indices() {return indices_;}
    Tensor* tensor() {return tensor_;}

    void print();
    /// Multiply two tensors with indices
    TensorProduct operator*(TensorIndexed lhs);
    /// Add a product of two tensors to a tensor with indices (contraction)
    void operator+=(TensorProduct tp);
    /// Set this equal to a product of two tensors (contraction)
    void operator=(TensorProduct tp);
    /// Add a tensor to this tensors (copy)
    void operator+=(TensorIndexed lhs);
    /// Set this tensor equal to another tensors (copy)
    void operator=(TensorIndexed lhs);
private:
    double factor_;
    std::vector<std::string> indices_;
    Tensor* tensor_;
};

TensorIndexed operator*(double factor,TensorIndexed ti);

/// This class is used to represent the product of tensors
class TensorProduct
{
public:
    TensorProduct(TensorIndexed A, TensorIndexed B) :
        A_(A), B_(B) {};
    TensorIndexed A() {return A_;}
    TensorIndexed B() {return B_;}
    void print();
private:
    TensorIndexed A_;
    TensorIndexed B_;
};

///// This class is used to evaluate contractions of tensors
//class TensorContractor
//{
//public:
//    TensorContractor();
//    ~TensorContractor();
//    void evaluate(TensorIndexed A,TensorProduct tp);
//private:
//    double* t1;
//    double* t2;
//};

bool test_tensor_class(bool verbose);

#endif // _tensor_h_

