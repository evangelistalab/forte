#include <cmath>

#include <boost/algorithm/string/join.hpp>
#include <boost/format.hpp>

#include <psi4-dec.h>

#include "tensor.h"

using namespace std;
using namespace psi;

bool Tensor::use_dgemm_ = true;
int Tensor::print_level_ = 0;

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
    }

    if (print_level_ > 0){
        // print fancy things
        std::vector<std::string> str;
        for_each(dims_.begin(),dims_.end(),[&](size_t d){str.push_back(std::to_string(d));});
        std::string joined = boost::algorithm::join(str, ",");
        fprintf(outfile,"\nAllocating the tensor %s[%s] with %zu elements",label_.c_str(),joined.c_str(),nelements_);
    }
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

std::string TensorIndexed::str() const
{
    std::string label;
    if (factor_ != 1.0){
        label += boost::str(boost::format("%+f") % factor_);
    }
    label += tensor_->label() + "(" + boost::algorithm::join(indices_, ",") + ")";
    return label;
}

void TensorIndexed::operator+=(TensorProduct tp)
{
    Tensor::contract(tp,*this,true);
}

void TensorIndexed::operator=(TensorProduct tp)
{
    Tensor::contract(tp,*this,false);
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
    return TensorProduct({*this,lhs});
}

void TensorProduct::print()
{
    for (auto t : tensors_){
        t.print();
        fprintf(outfile," ");
    }
}

TensorProduct TensorProduct::operator*(TensorIndexed lhs)
{
    append(lhs);
    return *this;
}

/// Return the memory and computational cost of a given contraction pattern
std::pair<double,double> TensorProduct::compute_contraction_cost(std::vector<size_t> perm)
{
    std::pair<double,double> cpu_memory_cost;
    if (Tensor::print_level() > 1){
        fprintf(outfile,"\n\n  Testing the cost of the contraction pattern:");
        for (size_t p : perm){
            fprintf(outfile,"[");
        }
        for (size_t p : perm){
            const TensorIndexed& ti = tensor(p);
            fprintf(outfile," %s] ",ti.str().c_str());
        }
    }

    std::map<std::string,size_t> indices_to_size;

    for (const TensorIndexed& ti: tensors_){
        const std::vector<std::string> indices = ti.indices();
        for (size_t i = 0; i < indices.size(); ++i){
            indices_to_size[indices[i]] = ti.tensor()->dims(i);
        }
    }

    double cpu_cost_total = 0.0;
    double memory_cost_max = 0.0;
    std::vector<std::string> first = tensors_[perm[0]].indices();
    for (size_t i = 1; i < perm.size(); ++i){
        std::vector<std::string> second = tensors_[perm[i]].indices();
        std::sort(first.begin(),first.end());
        std::sort(second.begin(),second.end());
        std::vector<std::string> common, first_unique, second_unique;


        // cannot use common.begin() here, need to use back_inserter() because common.begin() of an
        // empty vector is not a valid output iterator
        std::set_intersection(first.begin(),first.end(),second.begin(),second.end(),back_inserter(common));
        std::set_difference(first.begin(),first.end(),second.begin(),second.end(),back_inserter(first_unique));
        std::set_difference(second.begin(),second.end(),first.begin(),first.end(),back_inserter(second_unique));

        double common_size = 1.0;
        for (std::string s : common) common_size *= indices_to_size[s];
        double first_size = 1.0;
        for (std::string s : first) first_size *= indices_to_size[s];
        double second_size = 1.0;
        for (std::string s : second) second_size *= indices_to_size[s];
        double first_unique_size = 1.0;
        for (std::string s : first_unique) first_unique_size *= indices_to_size[s];
        double second_unique_size = 1.0;
        for (std::string s : second_unique) second_unique_size *= indices_to_size[s];
        double result_size = first_unique_size + second_unique_size;


        std::vector<std::string> stored_indices(first_unique);
        stored_indices.insert(stored_indices.end(),second_unique.begin(),second_unique.end());

        double cpu_cost = common_size * result_size;
        double memory_cost = first_size + second_size + result_size;
        cpu_cost_total += cpu_cost;
        memory_cost_max = std::max({memory_cost_max,memory_cost});

        if (Tensor::print_level() > 1){
            fprintf(outfile,"\n  First indices        : %s",boost::algorithm::join(first, " ").c_str());
            fprintf(outfile,"\n  Second indices       : %s",boost::algorithm::join(second, " ").c_str());

            fprintf(outfile,"\n  Common indices       : %s (%.0f)",boost::algorithm::join(common, " ").c_str(),common_size);
            fprintf(outfile,"\n  First unique indices : %s (%.0f)",boost::algorithm::join(first_unique, " ").c_str(),first_unique_size);
            fprintf(outfile,"\n  Second unique indices: %s (%.0f)",boost::algorithm::join(second_unique, " ").c_str(),second_unique_size);

            fprintf(outfile,"\n  CPU cost for this step    : %f.0",cpu_cost);
            fprintf(outfile,"\n  Memory cost for this step : %f.0 = %f.0 + %f.0 + %f.0",memory_cost,first_size,second_size,result_size);

            fprintf(outfile,"\n  Stored indices       : %s",boost::algorithm::join(stored_indices, " ").c_str());
            fflush(outfile);
        }
        first = stored_indices;
    }
    if (Tensor::print_level() > 1){
        fprintf(outfile,"\n  Total CPU cost                : %f.0",cpu_cost_total);
        fprintf(outfile,"\n  Maximum memory cost           : %f.0",memory_cost_max);
    }

    return cpu_memory_cost;
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
