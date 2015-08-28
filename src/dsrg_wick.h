#ifndef DSRG_WICK_H
#define DSRG_WICK_H

#include "ambit/blocked_tensor.h"
#include "sq.h"
#include "helpers.h"

typedef std::vector<double> d1;
typedef std::vector<d1> d2;
typedef std::vector<d2> d3;
typedef std::vector<d3> d4;

namespace psi { namespace libadaptive {

class DSRG_WICK
{
public:
    DSRG_WICK(std::shared_ptr<MOSpaceInfo> mo_space_info,
              ambit::BlockedTensor Fock, ambit::BlockedTensor RTEI,
              ambit::BlockedTensor T1, ambit::BlockedTensor T2);
private:
    // MO space info
    std::shared_ptr<MOSpaceInfo> mo_space_info_;
    // Fock operator
    Operator F_;
    // Antisymmetrized two electron integral
    Operator V_;
    // Single excitation operator
    Operator T1_;
    // Double excitation operator
    Operator T2_;

    /// List of alpha core MOs
    std::vector<size_t> acore_mos;
    /// List of alpha active MOs
    std::vector<size_t> aactv_mos;
    /// List of alpha virtual MOs
    std::vector<size_t> avirt_mos;
    /// List of alpha hole MOs
    std::vector<size_t> ahole_mos;
    /// List of alpha particle MOs
    std::vector<size_t> apart_mos;

    /// List of beta core MOs
    std::vector<size_t> bcore_mos;
    /// List of beta active MOs
    std::vector<size_t> bactv_mos;
    /// List of beta virtual MOs
    std::vector<size_t> bvirt_mos;
    /// List of beta hole MOs
    std::vector<size_t> bhole_mos;
    /// List of beta particle MOs
    std::vector<size_t> bpart_mos;

    /// A map between space label and space mos
    std::map<char, std::vector<size_t>> label_to_spacemo;

    // size of spin orbital space
    size_t ncso_;
    size_t nc_;
    size_t na_;
    size_t nv_;
    size_t nh_;
    size_t np_;

    // setup Fock operator (BlockedTensor -> Operator)
    void setup(ambit::BlockedTensor Fock, ambit::BlockedTensor RTEI,
               ambit::BlockedTensor T1, ambit::BlockedTensor T2);
};

}}

#endif // DSRG_WICK_H
