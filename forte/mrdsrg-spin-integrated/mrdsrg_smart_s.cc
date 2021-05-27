/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2020 by its authors (see COPYING, COPYING.LESSER, AUTHORS).
 *
 * The copyrights for code used from other parties are included in
 * the corresponding files.
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program.  If not, see http://www.gnu.org/licenses/.
 *
 * @END LICENSE
 */

#include <algorithm>
#include <numeric>
#include <vector>

#include "base_classes/mo_space_info.h"
#include "mrdsrg.h"


namespace forte {

double MRDSRG::make_s_smart() {
    double Edelta = 0.0, dsrg_s = 0.0;
    switch (smartsmap[foptions_->get_str("SMART_DSRG_S")]) {
    case SMART_S::MIN_DELTA1: {
        Edelta = smart_s_min_delta1();
        break;
    }
    case SMART_S::DAVG_MIN_DELTA1: {
        Edelta = smart_s_davg_min_delta1();
        break;
    }
    case SMART_S::MAX_DELTA1: {
        Edelta = smart_s_max_delta1();
        break;
    }
    case SMART_S::DAVG_MAX_DELTA1: {
        Edelta = smart_s_davg_max_delta1();
        break;
    }
    default: {
        dsrg_s = s_;
        return dsrg_s;
    }
    }

    if (source_ == "LABS") {
        dsrg_s = 1.0 / Edelta;
    } else {
        dsrg_s = 1.0 / (Edelta * Edelta);
    }

    return dsrg_s;
}

double MRDSRG::smart_s_min_delta1() {
    psi::Dimension virt = mo_space_info_->dimension("RESTRICTED_UOCC");
    int nirrep = virt.n();
    std::vector<double> lowest_virt;
    for (int h = 0; h < nirrep; ++h) {
        size_t index = 0;
        int h_local = h;
        while (--h_local >= 0)
            index += virt[h_local];
        lowest_virt.emplace_back(Fa_[virt_mos_[index]]);
    }

    double Edelta = 100.0;
    std::vector<int> actv_sym = mo_space_info_->symmetry("ACTIVE");
    size_t nactv = actv_sym.size();
    for (size_t i = 0; i < nactv; ++i) {
        size_t idx = actv_mos_[i];
        double diff = lowest_virt[actv_sym[i]] - Fa_[idx];
        if (Edelta > diff)
            Edelta = diff;
    }

    return Edelta;
}

double MRDSRG::smart_s_max_delta1() {
    psi::Dimension virt = mo_space_info_->dimension("RESTRICTED_UOCC");
    int nirrep = virt.n();
    std::vector<double> lowest_virt;
    for (int h = 0; h < nirrep; ++h) {
        size_t index = 0;
        int h_local = h;
        while (--h_local >= 0)
            index += virt[h_local];
        lowest_virt.emplace_back(Fa_[virt_mos_[index]]);
    }

    double Edelta = 0.0;
    std::vector<int> actv_sym = mo_space_info_->symmetry("ACTIVE");
    size_t nactv = actv_sym.size();
    for (size_t i = 0; i < nactv; ++i) {
        size_t idx = actv_mos_[i];
        double diff = lowest_virt[actv_sym[i]] - Fa_[idx];
        if (Edelta < diff)
            Edelta = diff;
    }

    return Edelta;
}

double MRDSRG::smart_s_davg_min_delta1() {
    // obtain a vector of the lowest virtual energies with irrep
    psi::Dimension virt = mo_space_info_->dimension("RESTRICTED_UOCC");
    int nirrep = virt.n();
    std::vector<double> lowest_virt;
    for (int h = 0; h < nirrep; ++h) {
        size_t index = 0;
        int h_local = h;
        while (--h_local >= 0)
            index += virt[h_local];
        lowest_virt.emplace_back(Fa_[virt_mos_[index]]);
    }

    // normalize diagonal density
    std::vector<double> davg;
    std::vector<int> actv_sym = mo_space_info_->symmetry("ACTIVE");
    size_t nactv = actv_sym.size();
    for (size_t i = 0; i < nactv; ++i) {
        davg.emplace_back(Eta1_.block("aa").data()[i * nactv + i]);
    }
    double davg_sum = std::accumulate(davg.begin(), davg.end(), 0.0);
    std::transform(davg.begin(), davg.end(), davg.begin(), [&](double x) { return x / davg_sum; });

    // density averaged denorminator
    double Edelta = 0.0;
    for (size_t i = 0; i < nactv; ++i) {
        size_t idx = actv_mos_[i];
        double diff = lowest_virt[actv_sym[i]] - Fa_[idx];
        Edelta += diff * davg[i];
    }

    return Edelta;
}

double MRDSRG::smart_s_davg_max_delta1() {
    // obtain a vector of the lowest virtual energies with irrep
    psi::Dimension virt = mo_space_info_->dimension("RESTRICTED_UOCC");
    int nirrep = virt.n();
    std::vector<double> lowest_virt;
    for (int h = 0; h < nirrep; ++h) {
        size_t index = 0;
        int h_local = h;
        while (--h_local >= 0)
            index += virt[h_local];
        lowest_virt.emplace_back(Fa_[virt_mos_[index]]);
    }

    // normalize diagonal density
    std::vector<double> davg;
    std::vector<int> actv_sym = mo_space_info_->symmetry("ACTIVE");
    size_t nactv = actv_sym.size();
    for (size_t i = 0; i < nactv; ++i) {
        davg.emplace_back(Gamma1_.block("aa").data()[i * nactv + i]);
    }
    double davg_sum = std::accumulate(davg.begin(), davg.end(), 0.0);
    std::transform(davg.begin(), davg.end(), davg.begin(), [&](double x) { return x / davg_sum; });

    // density averaged denorminator
    double Edelta = 0.0;
    for (size_t i = 0; i < nactv; ++i) {
        size_t idx = actv_mos_[i];
        double diff = lowest_virt[actv_sym[i]] - Fa_[idx];
        Edelta += diff * davg[i];
    }

    return Edelta;
}
}
