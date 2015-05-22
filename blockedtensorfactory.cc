#include "blockedtensorfactory.h"
#include <libpsio/psio.hpp>
#include <libpsio/psio.h>
#include <libmints/molecule.h>
#include <libmints/wavefunction.h>
#include <liboptions/liboptions.h>
#include <tuple>
#include <vector>
#include <string>


namespace psi{ namespace libadaptive{

BlockedTensorFactory::BlockedTensorFactory(Options& options)

{
    memory_ = Process::environment.get_memory()/8L;
    number_of_tensors_ = 0;

}
BlockedTensorFactory::~BlockedTensorFactory()
{
    if(print_memory_)
    {
        memory_summary();
    }

}

ambit::BlockedTensor BlockedTensorFactory::build(ambit::TensorType storage,const std::string& name,const std::vector<std::string>& spin_stuff)
{
    ambit::BlockedTensor BT = ambit::BlockedTensor::build(storage, name, spin_stuff);
    number_of_tensors_+= 1;
    memory_information(BT);
    if(memory_ < 0.0)
    {
        outfile->Printf("\n DANGER DANGER Will Robinson\n");
        outfile->Printf("\n Your memory requirements were underestimated.  Please be more careful! \n");
    }

    return BT;
}
void BlockedTensorFactory::add_mo_space(const std::string& name,const std::string& mo_indices,std::vector<size_t> mos,ambit::SpinType spin)
{
    ambit::BlockedTensor::add_mo_space(name, mo_indices, mos, spin);

}
void BlockedTensorFactory::add_mo_space(const std::string& name,const std::string& mo_indices,std::vector<std::pair<size_t,ambit::SpinType>> mo_spin)
{
    ambit::BlockedTensor::add_mo_space(name, mo_indices, mo_spin);
}
void BlockedTensorFactory::add_composite_mo_space(const std::string& name,const std::string& mo_indices,const std::vector<std::string>& subspaces)
{
    ambit::BlockedTensor::add_composite_mo_space(name, mo_indices, subspaces);
}
std::vector<std::string> BlockedTensorFactory::generate_indices(const std::string in_str, const std::string type)
{
    std::vector<std::string> return_string;

    //Hardlined for 4 character strings
    if(type=="all")
    {
        for(int i = 0; i < 3; i++){
            for(int j = 0; j < 3; j++){
                for(int k = 0; k < 3; k++){
                    for(int l = 0; l < 3; l++){
                        std::string one_string_lower;
                        std::string one_string_upper;
                        std::string one_string_mixed;

                        one_string_lower.push_back(in_str[i]);
                        one_string_lower.push_back(in_str[j]);
                        one_string_lower.push_back(in_str[k]);
                        one_string_lower.push_back(in_str[l]);

                        one_string_upper.push_back(std::toupper(in_str[i]));
                        one_string_upper.push_back(std::toupper(in_str[j]));
                        one_string_upper.push_back(std::toupper(in_str[k]));
                        one_string_upper.push_back(std::toupper(in_str[l]));

                        one_string_mixed.push_back(in_str[i]);
                        one_string_mixed.push_back(std::toupper(in_str[j]));
                        one_string_mixed.push_back(in_str[k]);
                        one_string_mixed.push_back(std::toupper(in_str[l]));

                        return_string.push_back(one_string_lower);
                        return_string.push_back(one_string_upper);
                        return_string.push_back(one_string_mixed);
                    }

                }
            }
        }
    }
    else if(type=="hhpp")
    {

        for(int i = 0; i < 2; i++){
            for(int j = 0; j < 2; j++){
                for(int k = 0; k < 2; k++){
                    for(int l = 0; l < 2; l++){
                        std::string one_string_lower;
                        std::string one_string_upper;
                        std::string one_string_mixed;

                        one_string_lower.push_back(in_str[i]);
                        one_string_lower.push_back(in_str[j]);
                        one_string_lower.push_back(in_str[k + 1]);
                        one_string_lower.push_back(in_str[l + 1]);

                        one_string_upper.push_back(std::toupper(in_str[i]));
                        one_string_upper.push_back(std::toupper(in_str[j]));
                        one_string_upper.push_back(std::toupper(in_str[k + 1]));
                        one_string_upper.push_back(std::toupper(in_str[l + 1]));

                        one_string_mixed.push_back(in_str[i]);
                        one_string_mixed.push_back(std::toupper(in_str[j]));
                        one_string_mixed.push_back(in_str[k + 1]);
                        one_string_mixed.push_back(std::toupper(in_str[l + 1]));

                        return_string.push_back(one_string_lower);
                        return_string.push_back(one_string_upper);
                        return_string.push_back(one_string_mixed);
                    }

                }
            }
        }
    }


    return return_string;

}
void BlockedTensorFactory::memory_information(ambit::BlockedTensor BT)
{
    double size_of_tensor = 0.0;
    std::vector<std::string> BTblocks = BT.block_labels();
    for(const std::string& block: BTblocks){
        ambit::Tensor temp = BT.block(block);
        size_of_tensor += temp.numel();
    }
    double memory_of_tensor = size_of_tensor * 8.0 /1073741824;
    tensors_information_.push_back(std::make_pair(BT.name(), memory_of_tensor));
    number_of_blocks_.push_back(BT.numblocks());
    memory_ -= memory_of_tensor;

}
void BlockedTensorFactory::memory_summary()
{
    outfile->Printf("\n Memory Summary of all tensors \n");
    outfile->Printf("\n TensorName \t Number_of_blocks \t memory gb");
    for(int i = 0; i < tensors_information_.size(); i++)
    {
        outfile->Printf("\n %-25s  %u    %8.8 GB", tensors_information_[i].first.c_str(), number_of_blocks_[i],tensors_information_[i].second);
    }
    outfile->Printf("\n Memory left over: %8.6f GB\n", memory_/1073741824);
}
std::vector<std::string> BlockedTensorFactory::spin_cases_avoid(const std::vector<std::string>& in_str_vec)
{

    std::vector<std::string> out_str_vec;
    for(const std::string spin : in_str_vec){
        size_t spin_ind  = spin.find('a');
        size_t spin_ind2 = spin.find('A');
        if(spin_ind != std::string::npos|| spin_ind2 != std::string::npos){
            out_str_vec.push_back(spin);
        }
    }
    return out_str_vec;
}
}}
