#include "tensor.h"
#include "tensor_labeled.h"
#include "tensor_product.h"

#include <boost/algorithm/string/join.hpp>
#include <boost/format.hpp>

#include <psi4-dec.h>

using namespace psi;

void LabeledTensor::print()
{
    if (factor_ != 1.0) fprintf(outfile,"%+f ",factor_);
    fprintf(outfile,"%s(",tensor_->label().c_str());
    fprintf(outfile,"%s)",boost::algorithm::join(indices_, ",").c_str());
}

std::string LabeledTensor::str() const
{
    std::string label;
    if (factor_ != 1.0){
        label += boost::str(boost::format("%+f") % factor_);
    }
    label += tensor_->label() + "(" + boost::algorithm::join(indices_, ",") + ")";
    return label;
}

void LabeledTensor::operator+=(LabeledTensorProduct tp)
{
    Tensor::contract(tp,*this,true);
}

void LabeledTensor::operator=(LabeledTensorProduct tp)
{
    Tensor::contract(tp,*this,false);
}

void LabeledTensor::operator+=(LabeledTensor ti)
{
    Tensor::add(ti,*this,true);
}

void LabeledTensor::operator=(LabeledTensor ti)
{
    Tensor::add(ti,*this,false);
}

LabeledTensorProduct LabeledTensor::operator*(LabeledTensor lhs)
{
    return LabeledTensorProduct({*this,lhs});
}
