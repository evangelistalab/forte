#include <boost/algorithm/string/join.hpp>

#include <psi4-dec.h>

#include "tensor.h"

using namespace std;
using namespace psi;

Tensor::Tensor(std::string label, std::vector<size_t> dims)
    : label_(label),
      ndims_(dims.size()),
      dims_(dims),
      t_(nullptr),
      nelements_(0)
{
    // compute the number of elements
    nelements_ = 1;
    for (size_t d : dims_) nelements_ *= d;

    // compute the addressing array
    add_.resize(ndims_,1);
    for (size_t n = 0; n < ndims_ - 1; ++n){
        size_t i = ndims_ - 2 - n;
        add_[i] = add_[i + 1] * dims_[i + 1];
        fprintf(outfile,"\n add[%zu] = %zu",i,add_[i]);
    }

    allocate();
}

Tensor::~Tensor()
{
    release();
}

void Tensor::allocate()
{
    std::vector<std::string> str;
    for_each(dims_.begin(),dims_.end(),[&](size_t d){str.push_back(std::to_string(d));});
    std::string joined = boost::algorithm::join(str, ",");

    fprintf(outfile,"\nAllocating the tensor %s[%s] with %zu elements",label_.c_str(),joined.c_str(),nelements_);

    if (t_ != nullptr) delete[] t_;
    t_ = new double[nelements_];
}

void Tensor::release()
{
    if (t_ != nullptr) delete t_;
}

inline size_t Tensor::two_address(size_t i1,size_t i2) const
{
//    return(i1 * dims_[1] + i2);
    return(i1 * add_[0] + i2);
}

inline size_t Tensor::four_address(size_t i1,size_t i2,size_t i3,size_t i4) const
{
    return(i1 * add_[0] + i2 * add_[1] + i3 * add_[2] + i4);
//    return(i1 * dims_[1] * dims_[2] * dims_[3] + i2 * dims_[2] * dims_[3]  + i3 * dims_[3] + i4);
}

double& Tensor::operator()(size_t i1,size_t i2)
{
    return t_[two_address(i1,i2)];
}

const double& Tensor::operator()(size_t i1,size_t i2) const
{
    return t_[two_address(i1,i2)];
}

double& Tensor::operator()(size_t i1,size_t i2,size_t i3,size_t i4)
{
    return t_[four_address(i1,i2,i3,i4)];
}

const double& Tensor::operator()(size_t i1,size_t i2,size_t i3,size_t i4) const
{
    return t_[four_address(i1,i2,i3,i4)];
}

TensorIndexed Tensor::operator()(std::string indices)
{
    // Input format 1: a series of characters ""
    std::vector<std::string> i_vec;
    for (char c : indices){
        i_vec.push_back(string(1,c));
    }
    return TensorIndexed(i_vec,this);
}

TensorIndexed operator*(double factor,TensorIndexed ti)
{
    factor *= ti.factor();
    TensorIndexed result(factor,ti.indices(),ti.tensor());
    return result;
}

void TensorIndexed::print()
{
    if (factor_ != 1.0) fprintf(outfile,"%+f ",factor_);
    fprintf(outfile,"%s(",tensor_->label().c_str());
    fprintf(outfile,"%s)",boost::algorithm::join(indices_, ",").c_str());
}

void TensorIndexed::operator+=(TensorProduct tp)
{
    Tensor::evaluate(tp.A(),tp.B(),*this);
}

TensorProduct TensorIndexed::operator*(TensorIndexed lhs)
{
    return TensorProduct(*this,lhs);
}

void TensorProduct::print()
{
    A_.print();
    fprintf(outfile," * ");
    B_.print();
}
//void Tensor::sort()
//{
//    int maxi1 = ;
//    int maxi2 = ;
//    /// 1-1-Tensor
//    for (int i1 = 0; i1 < maxi1; ++i1){
//        for (int i2 = 0; i2 < maxi2; ++i2){
//            index_sort(i1,i2);
//        }
//    }
//}
