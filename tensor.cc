#include <cmath>

#include <boost/algorithm/string/join.hpp>

#include <psi4-dec.h>

#include "tensor.h"

using namespace std;
using namespace psi;

bool Tensor::use_dgemm_ = true;
int Tensor::print_ = 0;

Tensor::Tensor()
    : t_(nullptr), nelements_(0)
{
}

Tensor::Tensor(std::string label,std::vector<size_t> dims)
    : label_(label), t_(nullptr), nelements_(0)
{
    allocate(dims);
}

Tensor::~Tensor()
{
    release();
}

void Tensor::resize(std::vector<size_t> dims)
{
    allocate(dims);
}

void Tensor::resize(std::string label, std::vector<size_t> dims)
{
    label_ = label;
    resize(dims);
}

void Tensor::allocate(std::vector<size_t> dims)
{
    // set the dimensions
    ndims_ = dims.size();
    dims_ = dims;

    // compute the number of elements
    nelements_ = 1;
    for (size_t d : dims_) nelements_ *= d;

    // allocate the memory and zero
    if (t_ != nullptr) delete[] t_;
    t_ = new double[nelements_];
    for (size_t i = 0; i < nelements_; ++i) t_[i] = 0.0;

    // compute the addressing array
    add_.resize(ndims_,1);
    for (size_t n = 0; n < ndims_ - 1; ++n){
        size_t i = ndims_ - 2 - n;
        add_[i] = add_[i + 1] * dims_[i + 1];
        fprintf(outfile,"\n add[%zu] = %zu",i,add_[i]);
    }

    // print fancy things
    std::vector<std::string> str;
    for_each(dims_.begin(),dims_.end(),[&](size_t d){str.push_back(std::to_string(d));});
    std::string joined = boost::algorithm::join(str, ",");
    fprintf(outfile,"\nAllocating the tensor %s[%s] with %zu elements",label_.c_str(),joined.c_str(),nelements_);
}

void Tensor::release()
{
    if (t_ != nullptr) delete t_;
    t_ = nullptr;
}

inline size_t Tensor::one_address(size_t i0) const
{
    return(i0);
}

inline size_t Tensor::three_address(size_t i0,size_t i1,size_t i2) const
{
    return(i0 * add_[0] + i1 * add_[1] + i2);
}

double& Tensor::operator()(size_t i0,size_t i1)
{
    return t_[two_address(i0,i1)];
}

const double& Tensor::operator()(size_t i0,size_t i1) const
{
    return t_[two_address(i0,i1)];
}

double& Tensor::operator()(size_t i0,size_t i1,size_t i2,size_t i3)
{
    return t_[four_address(i0,i1,i2,i3)];
}

const double& Tensor::operator()(size_t i0,size_t i1,size_t i2,size_t i3) const
{
    return t_[four_address(i0,i1,i2,i3)];
}

void Tensor::zero()
{
    std::fill(t_,t_ + nelements_,0.0);
}

double Tensor::norm(int power)
{
    double sum = 0.0;
    double p = power;
    for (int n = 0; n < nelements_; ++n){
        sum += std::pow(t_[n],p);
    }
    return std::pow(sum,1.0 / p);
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
    Tensor::evaluate(tp.A(),tp.B(),*this,true);
}

void TensorIndexed::operator=(TensorProduct tp)
{
    Tensor::evaluate(tp.A(),tp.B(),*this,false);
}

void TensorIndexed::operator+=(TensorIndexed ti)
{
    Tensor::add(ti,*this,true);
}

void TensorIndexed::operator=(TensorIndexed ti)
{
    Tensor::add(ti,*this,false);
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
