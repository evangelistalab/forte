#include "tensor.h"
#include "tensor_labeled.h"
#include "tensor_product.h"

#include <boost/algorithm/string/join.hpp>
#include <boost/format.hpp>

#include <psi4-dec.h>

using namespace psi;

void LabeledTensorProduct::print()
{
    for (auto t : tensors_){
        t.print();
        fprintf(outfile," ");
    }
}

LabeledTensorProduct LabeledTensorProduct::operator*(LabeledTensor lhs)
{
    append(lhs);
    return *this;
}

LabeledTensor LabeledTensorProduct::tensor(size_t n)
{
    return tensors_[n];
}

void LabeledTensorProduct::append(LabeledTensor tl){
    tensors_.push_back(tl);
}

void LabeledTensorProduct::append(std::initializer_list<LabeledTensor> l)
{
    tensors_.insert(tensors_.end(),l.begin(),l.end());
}

/// Return the memory and computational cost of a given contraction pattern
std::pair<double,double> LabeledTensorProduct::compute_contraction_cost(std::vector<size_t> perm)
{
    std::pair<double,double> cpu_memory_cost;
    if (Tensor::print_level() > 1){
        fprintf(outfile,"\n\n  Testing the cost of the contraction pattern:");
        for (size_t p : perm){
            fprintf(outfile,"[");
        }
        for (size_t p : perm){
            const LabeledTensor& ti = tensor(p);
            fprintf(outfile," %s] ",ti.str().c_str());
        }
    }

    std::map<std::string,size_t> indices_to_size;

    for (const LabeledTensor& ti: tensors_){
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
