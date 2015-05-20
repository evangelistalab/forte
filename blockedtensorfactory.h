#ifndef BLOCKEDTENSORFACTORY_H
#define BLOCKEDTENSORFACTORY_H

///A class used to create BlockedTensors similar
/// to matrix factory.
/// All blockedTensor functions with strings should be placed here
/// Creates MO SPACES
#include <ambit/blocked_tensor.h>
#include "integrals.h"
#include <vector>
#include <tuple>


namespace psi{ namespace libadaptive{

class BlockedTensorFactory
{
protected:
    // Whether the tensor is kCore, kDisk, Agnostic
    // TODO:  Make this more generic -
    ambit::TensorType tensor_type_ = ambit::kCore;
    // Overall memory that tensors are taking up
    double memory_;
    //Number of BlockedTensors used
    int number_of_tensors_;
    //String of all the tensors
    std::vector<std::string> tensor_names_;
    //Name of Tensor and memory requirements
    std::vector<std::pair<std::string, double > > tensors_information_;
    std::vector<size_t> number_of_blocks_;
    //Used to control printing for memory summary
    bool print_memory_ = false;

public:
    BlockedTensorFactory(Options &options);
    ~BlockedTensorFactory();
    /* -- Builds a block tensor
    //@params  storage-> Core, disk, or agnostic (program decides) Core has been tested
    //@params  std::string name -> name of tensor
    //@params  spin_stuff -> A vector of strings listing all possible spin components
    //ie for a hhpp, {hhpp, hHpP, HHPP}
    //Returns -> BlockTensor
    */
    ambit::BlockedTensor build(ambit::TensorType storage, const std::string &name, const std::vector<std::string> &spin_stuff) ;
    /* -- add_mo_space
      Adds a mo space -> core, active, virtual
    //@params std::string name -> name of the space
    //@params std::string mo_indicies -> string index ie m,n
    //@params vector of indices that correspond to the name ie {0,1,2,3,4,5,6}
    //@params spin -> NoSpin, AlphaSpin, BetaSpin
    */
    void add_mo_space(const std::string& name,const std::string& mo_indices,std::vector<size_t> mos,ambit::SpinType spin);
    void add_mo_space(const std::string& name,const std::string& mo_indices,std::vector<std::pair<size_t,ambit::SpinType>> mo_spin);
    //Adds a composite_mo_space -> combines mo_space -> h = c + a
    void add_composite_mo_space(const std::string& name,const std::string& mo_indices,const std::vector<std::string>& subspaces);
    void memory_info(ambit::BlockedTensor BT);
    /* - This function generates all possible MO spaces and spin components
    /// Param:  std::string is the lables - "cav"
    /// Will take a string like cav and generate all possible combinations of this
    /// for a four character string
    */
    std::vector<std::string> generate_indices(const std::string in_str, const std::string type);
    //Lets the user know how much memory is left
    double memory_left(){return memory_;}
    //Calculates the amount of memory BlockedTensor takes up
    void memory_information(ambit::BlockedTensor);
    //Array of all things memory
    void memory_summary();
    //controls printing information
    void print_memory_info(){print_memory_=true;}
};
}}


#endif // BLOCKEDTENSORFACTORY_H
