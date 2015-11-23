#include <vector>
#include <numeric>
#include <algorithm>

#include "helpers.h"
#include "mrdsrg.h"

namespace psi{ namespace forte{

double MRDSRG::make_s_smart(){
    double Edelta = 0.0, dsrg_s = 0.0;
    switch (smartsmap[options_.get_str("SMART_DSRG_S")]){
    case SMART_S::MIN_DELTA1:{
        Edelta = smart_s_min_delta1();
        break;
    }
    case SMART_S::DAVG_MIN_DELTA1:{
        Edelta = smart_s_davg_min_delta1();
        break;
    }
    case SMART_S::MAX_DELTA1:{
        Edelta = smart_s_max_delta1();
        break;
    }
    case SMART_S::DAVG_MAX_DELTA1:{
        Edelta = smart_s_davg_max_delta1();
        break;
    }
    default:{
        dsrg_s = s_;
        return dsrg_s;
    }}

    if (source_ == "LABS"){
        dsrg_s = 1.0 / Edelta;
    }else{
        dsrg_s = 1.0 / (Edelta * Edelta);
    }

    return dsrg_s;
}

double MRDSRG::smart_s_min_delta1(){
    Dimension virt = mo_space_info_->get_dimension("RESTRICTED_UOCC");
    int nirrep = virt.n();
    std::vector<double> lowest_virt;
    for(int h = 0; h < nirrep; ++h){
        size_t index = 0;
        int h_local = h;
        while (--h_local >= 0)  index += virt[h_local];
        lowest_virt.emplace_back(Fa_[avirt_mos_[index]]);
    }

    double Edelta = 100.0, dsrg_s = 0.0;
    std::vector<int> actv_sym = mo_space_info_->symmetry("ACTIVE");
    size_t nactv = actv_sym.size();
    for(size_t i = 0; i < nactv; ++i){
        size_t idx = aactv_mos_[i];
        double diff = lowest_virt[actv_sym[i]] - Fa_[idx];
        if (Edelta > diff)  Edelta = diff;
    }

    return Edelta;
}

double MRDSRG::smart_s_max_delta1(){
    Dimension virt = mo_space_info_->get_dimension("RESTRICTED_UOCC");
    int nirrep = virt.n();
    std::vector<double> lowest_virt;
    for(int h = 0; h < nirrep; ++h){
        size_t index = 0;
        int h_local = h;
        while (--h_local >= 0)  index += virt[h_local];
        lowest_virt.emplace_back(Fa_[avirt_mos_[index]]);
    }

    double Edelta = 0.0, dsrg_s = 0.0;
    std::vector<int> actv_sym = mo_space_info_->symmetry("ACTIVE");
    size_t nactv = actv_sym.size();
    for(size_t i = 0; i < nactv; ++i){
        size_t idx = aactv_mos_[i];
        double diff = lowest_virt[actv_sym[i]] - Fa_[idx];
        if (Edelta < diff)  Edelta = diff;
    }

    return Edelta;
}

double MRDSRG::smart_s_davg_min_delta1(){
    // obtain a vector of the lowest virtual energies with irrep
    Dimension virt = mo_space_info_->get_dimension("RESTRICTED_UOCC");
    int nirrep = virt.n();
    std::vector<double> lowest_virt;
    for(int h = 0; h < nirrep; ++h){
        size_t index = 0;
        int h_local = h;
        while (--h_local >= 0)  index += virt[h_local];
        lowest_virt.emplace_back(Fa_[avirt_mos_[index]]);
    }

    // normalize diagonal density
    std::vector<double> davg;
    std::vector<int> actv_sym = mo_space_info_->symmetry("ACTIVE");
    size_t nactv = actv_sym.size();
    for(size_t i = 0; i < nactv; ++i){
        davg.emplace_back(Eta1_.block("aa").data()[i * nactv + i]);
    }
    double davg_sum = std::accumulate(davg.begin(), davg.end(), 0.0);
    std::transform(davg.begin(), davg.end(), davg.begin(), std::bind1st(std::multiplies<double>(),1.0/davg_sum));

    // density averaged denorminator
    double Edelta = 0.0;
    for(size_t i = 0; i < nactv; ++i){
        size_t idx = aactv_mos_[i];
        double diff = lowest_virt[actv_sym[i]] - Fa_[idx];
        Edelta += diff * davg[i];
    }

    return Edelta;
}

double MRDSRG::smart_s_davg_max_delta1(){
    // obtain a vector of the lowest virtual energies with irrep
    Dimension virt = mo_space_info_->get_dimension("RESTRICTED_UOCC");
    int nirrep = virt.n();
    std::vector<double> lowest_virt;
    for(int h = 0; h < nirrep; ++h){
        size_t index = 0;
        int h_local = h;
        while (--h_local >= 0)  index += virt[h_local];
        lowest_virt.emplace_back(Fa_[avirt_mos_[index]]);
    }

    // normalize diagonal density
    std::vector<double> davg;
    std::vector<int> actv_sym = mo_space_info_->symmetry("ACTIVE");
    size_t nactv = actv_sym.size();
    for(size_t i = 0; i < nactv; ++i){
        davg.emplace_back(Gamma1_.block("aa").data()[i * nactv + i]);
    }
    double davg_sum = std::accumulate(davg.begin(), davg.end(), 0.0);
    std::transform(davg.begin(), davg.end(), davg.begin(), std::bind1st(std::multiplies<double>(),1.0/davg_sum));

    // density averaged denorminator
    double Edelta = 0.0;
    for(size_t i = 0; i < nactv; ++i){
        size_t idx = aactv_mos_[i];
        double diff = lowest_virt[actv_sym[i]] - Fa_[idx];
        Edelta += diff * davg[i];
    }

    return Edelta;
}

}}
