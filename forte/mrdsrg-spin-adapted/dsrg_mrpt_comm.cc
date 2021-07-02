/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2021 by its authors (see COPYING, COPYING.LESSER, AUTHORS).
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

#include "psi4/libpsi4util/PsiOutStream.h"

#include "dsrg_mrpt.h"
#include "helpers/timer.h"

using namespace psi;

namespace forte {

void DSRG_MRPT::H1_T1_C0(BlockedTensor& H1, BlockedTensor& T1, const double& alpha, double& C0) {
    local_timer timer;

    double E = 0.0;
    E += 2.0 * H1["ma"] * T1["ma"];
    ambit::BlockedTensor temp = ambit::BlockedTensor::build(tensor_type_, "Temp110", {"aa"});
    temp["uv"] += H1["ve"] * T1["ue"];
    temp["uv"] -= H1["mu"] * T1["mv"];
    E += L1_["vu"] * temp["uv"];

    E *= alpha;
    C0 += E;

    if (print_ > 2) {
        outfile->Printf("\n    Time for [H1, T1] -> C0 : %10.3f", timer.get());
    }
    dsrg_time_.add("110", timer.get());
}

void DSRG_MRPT::H1_T2_C0(BlockedTensor& H1, BlockedTensor& T2, const double& alpha, double& C0) {
    local_timer timer;

    double E = 0.0;
    ambit::BlockedTensor temp = ambit::BlockedTensor::build(tensor_type_, "Temp120", {"aaaa"});
    temp["uvxy"] += H1["xe"] * T2["uvey"];
    temp["uvxy"] -= H1["mv"] * T2["umxy"];
    temp["uvxy"] += H1["ye"] * T2["uvxe"];
    temp["uvxy"] -= H1["mu"] * T2["mvxy"];
    E += 0.5 * L2_["xyuv"] * temp["uvxy"];

    E *= alpha;
    C0 += E;

    if (print_ > 2) {
        outfile->Printf("\n    Time for [H1, T2] -> C0 : %10.3f", timer.get());
    }
    dsrg_time_.add("120", timer.get());
}

void DSRG_MRPT::H2_T1_C0(BlockedTensor& H2, BlockedTensor& T1, const double& alpha, double& C0) {
    local_timer timer;

    double E = 0.0;
    ambit::BlockedTensor temp = ambit::BlockedTensor::build(tensor_type_, "Temp120", {"aaaa"});
    temp["uvxy"] += H2["xyev"] * T1["ue"];
    temp["uvxy"] -= H2["myuv"] * T1["mx"];
    temp["uvxy"] += H2["xyue"] * T1["ve"];
    temp["uvxy"] -= H2["xmuv"] * T1["my"];
    E += 0.5 * L2_["xyuv"] * temp["uvxy"];

    E *= alpha;
    C0 += E;

    if (print_ > 2) {
        outfile->Printf("\n    Time for [H2, T1] -> C0 : %10.3f", timer.get());
    }
    dsrg_time_.add("210", timer.get());
}

void DSRG_MRPT::H2_T2_C0(BlockedTensor& H2, BlockedTensor& T2, const double& alpha, double& C0,
                         const bool& stored) {
    local_timer timer;

    double E = 0.0;

    H2_T2_C0_L1(H2, T2, 1.0, E, stored);
    H2_T2_C0_L2(H2, T2, 1.0, E);
    H2_T2_C0_L3(H2, T2, 1.0, E);

    E *= alpha;
    C0 += E;

    if (print_ > 2) {
        outfile->Printf("\n    Time for [H2, T2] -> C0 : %10.3f", timer.get());
    }
    dsrg_time_.add("220", timer.get());
}

void DSRG_MRPT::H2_T2_C0_L1(BlockedTensor& H2, BlockedTensor& T2, const double& alpha, double& C0,
                            const bool& stored) {
    local_timer timer;
    double E = 0.0;

    if (!stored) {
        // I decide to keep actv_mos_ as a whole, otherwise the cavv and ccav
        // terms would be a headache.
        // separte core_mos_ to nbatch_ subvectors
        size_t core_size = core_mos_.size();
        std::vector<std::vector<size_t>> nb_core_mos;
        size_t nc = nbatch_;
        if (core_size < nc && core_size != 0)
            nc = core_size;
        size_t even = core_size / nc;
        size_t left = core_size % nc;

        for (size_t i = 0, start = 0; i < nc; ++i) {
            size_t end;
            if (i < left) {
                end = start + (even + 1);
            } else {
                end = start + even;
            }
            std::vector<size_t> part(core_mos_.begin() + start, core_mos_.begin() + end);
            nb_core_mos.emplace_back(part);
            start = end;
        }

        // [V, T2] from ccvv; this will change mo_spaces.
        E += V_T2_C0_L1_ccvv(nb_core_mos);

        // [V, T2] from cavv; this will change mo_spaces.
        E += V_T2_C0_L1_cavv(nb_core_mos);

        // [V, T2] from cavv; this will change mo_spaces.
        E += V_T2_C0_L1_ccav();
    } else {
        // [H2, T2] from ccvv
        E += 2.0 * H2["mnef"] * T2["mnef"];
        E -= H2["nmef"] * T2["mnef"];

        // [H2, T2] from cavv
        ambit::BlockedTensor temp = ambit::BlockedTensor::build(tensor_type_, "temp_cavv", {"aa"});
        temp["vu"] += 2.0 * H2["muef"] * T2["mvef"];
        //        temp["vu"] += H2["umef"] * T2["vmef"];
        temp["vu"] -= H2["muef"] * T2["vmef"];
        E += temp["vu"] * L1_["uv"];

        // [H2, T2] from ccav
        temp.zero();
        temp["vu"] += 2.0 * H2["mnve"] * T2["mnue"];
        //        temp["vu"] += H2["mnev"] * T2["mneu"];
        temp["vu"] -= H2["mnev"] * T2["mnue"];
        E += temp["vu"] * Eta1_["uv"];
    }

    // reset the mo_spaces back to origin
    ambit::BlockedTensor::reset_mo_spaces();
    ambit::BlockedTensor::add_mo_space("c", "mn", core_mos_, NoSpin);
    ambit::BlockedTensor::add_mo_space("a", "uvwxyz", actv_mos_, NoSpin);
    ambit::BlockedTensor::add_mo_space("v", "ef", virt_mos_, NoSpin);
    ambit::BlockedTensor::add_composite_mo_space("h", "ijkl", {"c", "a"});
    ambit::BlockedTensor::add_composite_mo_space("p", "abcd", {"a", "v"});
    ambit::BlockedTensor::add_composite_mo_space("g", "pqrs", {"c", "a", "v"});

    // rest of the terms from [H2, T2] involving only L1
    ambit::BlockedTensor temp = ambit::BlockedTensor::build(tensor_type_, "temp_aavv", {"aaaa"});
    temp["yvxu"] += 2.0 * H2["xuef"] * T2["yvef"];
    temp["yvxu"] -= H2["uxef"] * T2["yvef"];
    E += 0.25 * temp["yvxu"] * L1_["uv"] * L1_["xy"];

    temp.zero();
    temp.set_name("temp_ccaa");
    temp["vyux"] += 2.0 * H2["mnvy"] * T2["mnux"];
    temp["vyux"] -= H2["mnyv"] * T2["mnux"];
    E += 0.25 * temp["vyux"] * Eta1_["uv"] * Eta1_["xy"];

    temp.zero();
    temp.set_name("temp_caav");
    temp["uxyv"] += H2["mxve"] * T2["myue"];
    temp["uxyv"] += H2["xmve"] * T2["ymue"];
    temp["uxyv"] -= 0.5 * H2["xmve"] * T2["myue"];
    temp["uxyv"] -= 0.5 * H2["mxve"] * T2["ymue"];
    E += temp["uxyv"] * Eta1_["uv"] * L1_["xy"];

    temp.zero();
    temp.set_name("temp_aaav_caaa");
    temp["uxyv"] += 0.5 * H2["xwve"] * T2["yzue"] * L1_["wz"];
    temp["uxyv"] -= 0.25 * H2["wxve"] * T2["yzue"] * L1_["wz"];
    temp["uxyv"] += 0.5 * H2["mxvz"] * T2["myuw"] * Eta1_["wz"];
    temp["uxyv"] -= 0.25 * H2["xmvz"] * T2["myuw"] * Eta1_["wz"];
    E += temp["uxyv"] * Eta1_["uv"] * L1_["xy"];

    E *= alpha;
    C0 += E;
    dsrg_time_.add("220", timer.get());
}

double DSRG_MRPT::V_T2_C0_L1_ccvv(const std::vector<std::vector<size_t>>& small_core_mo) {
    double E = 0.0;

    for (size_t i = 0; i < small_core_mo.size(); ++i) {
        for (size_t j = i; j < small_core_mo.size(); ++j) {

            // reset the mo_spaces for BlockedTensor
            ambit::BlockedTensor::reset_mo_spaces();
            ambit::BlockedTensor::add_mo_space("i", "m1,n1", small_core_mo[i], NoSpin);
            ambit::BlockedTensor::add_mo_space("j", "m2,n2", small_core_mo[j], NoSpin);
            ambit::BlockedTensor::add_mo_space("v", "e,f", virt_mos_, NoSpin);

            // fill in V and T
            ambit::BlockedTensor V = ambit::BlockedTensor::build(tensor_type_, "V_ccvv", {"ijvv"});
            ambit::BlockedTensor T = ambit::BlockedTensor::build(tensor_type_, "T2_ccvv", {"ijvv"});

            V.iterate([&](const std::vector<size_t>& i, const std::vector<SpinType>&,
                          double& value) { value = ints_->aptei_ab(i[0], i[1], i[2], i[3]); });

            T["m1,m2,e,f"] = V["m1,m2,e,f"];
            if (ccvv_source_ == "ZERO") {
                BT_scaled_by_D(T);
            } else if (ccvv_source_ == "NORMAL") {
                BT_scaled_by_RD(T);
                BT_scaled_by_Rplus1(V);
            }

            // compute the Coulomb part
            double value = 0.0;
            value += 2.0 * V["m1,m2,e,f"] * T["m1,m2,e,f"];

            // modify V for the exchange part
            V = ambit::BlockedTensor::build(tensor_type_, "V_ccvv", {"jivv"});
            V.iterate([&](const std::vector<size_t>& i, const std::vector<SpinType>&,
                          double& value) { value = ints_->aptei_ab(i[0], i[1], i[2], i[3]); });
            if (ccvv_source_ == "NORMAL") {
                BT_scaled_by_Rplus1(V);
            }
            value -= V["m2,m1,e,f"] * T["m1,m2,e,f"];

            if (i == j) {
                E += value;
            } else {
                E += 2.0 * value;
            }
        }
    }
    return E;
}

double DSRG_MRPT::V_T2_C0_L1_cavv(const std::vector<std::vector<size_t>>& small_core_mo) {
    double E = 0.0;

    for (size_t i = 0; i < small_core_mo.size(); ++i) {
        // reset the mo_spaces for BlockedTensor
        ambit::BlockedTensor::reset_mo_spaces();
        ambit::BlockedTensor::add_mo_space("c", "mn", small_core_mo[i], NoSpin);
        ambit::BlockedTensor::add_mo_space("a", "uv", actv_mos_, NoSpin);
        ambit::BlockedTensor::add_mo_space("v", "ef", virt_mos_, NoSpin);

        // create a temp for contraction with L1
        ambit::BlockedTensor temp = ambit::BlockedTensor::build(tensor_type_, "temp_cavv", {"aa"});

        // fill in V and T
        ambit::BlockedTensor V = ambit::BlockedTensor::build(tensor_type_, "V_cavv", {"cavv"});
        ambit::BlockedTensor T = ambit::BlockedTensor::build(tensor_type_, "T2_cavv", {"cavv"});

        // compute the first term
        V.iterate([&](const std::vector<size_t>& i, const std::vector<SpinType>&, double& value) {
            value = ints_->aptei_ab(i[0], i[1], i[2], i[3]);
        });

        T["muef"] = V["muef"];
        if (ccvv_source_ == "ZERO") {
            BT_scaled_by_D(T);
        } else if (ccvv_source_ == "NORMAL") {
            BT_scaled_by_RD(T);
            BT_scaled_by_Rplus1(V);
        }
        temp["vu"] += 2.0 * V["muef"] * T["mvef"];

        // modify T for the third term
        T = ambit::BlockedTensor::build(tensor_type_, "T2_acvv", {"acvv"});
        T.iterate([&](const std::vector<size_t>& i, const std::vector<SpinType>&, double& value) {
            value = ints_->aptei_ab(i[0], i[1], i[2], i[3]);
        });
        if (ccvv_source_ == "ZERO") {
            BT_scaled_by_D(T);
        } else if (ccvv_source_ == "NORMAL") {
            BT_scaled_by_RD(T);
        }
        temp["vu"] -= V["muef"] * T["vmef"];

        //        // modify V for the second term
        //        V =
        //        ambit::BlockedTensor::build(tensor_type_,"V_acvv",{"acvv"});
        //        V.iterate([&](const std::vector<size_t>& i,const
        //        std::vector<SpinType>&,double& value){
        //            value = ints_->aptei_ab(i[0],i[1],i[2],i[3]);
        //        });
        //        if(ccvv_source_ == "NORMAL"){
        //            BT_scaled_by_Rplus1(V);
        //        }
        //        temp["vu"] += V["umef"] * T["vmef"];

        E += temp["vu"] * L1_["uv"];
    }
    return E;
}

double DSRG_MRPT::V_T2_C0_L1_ccav() {
    double E = 0.0;
    ambit::BlockedTensor::reset_mo_spaces();
    ambit::BlockedTensor::add_mo_space("c", "mn", core_mos_, NoSpin);
    ambit::BlockedTensor::add_mo_space("a", "uvwxyz", actv_mos_, NoSpin);
    ambit::BlockedTensor::add_mo_space("v", "ef", virt_mos_, NoSpin);

    // create a temp for contraction with L1
    ambit::BlockedTensor temp = ambit::BlockedTensor::build(tensor_type_, "temp_cavv", {"aa"});

    // fill in V and T
    ambit::BlockedTensor V = ambit::BlockedTensor::build(tensor_type_, "V_ccav", {"ccav"});
    ambit::BlockedTensor T = ambit::BlockedTensor::build(tensor_type_, "T2_ccav", {"ccav"});

    // compute the first term
    V.iterate([&](const std::vector<size_t>& i, const std::vector<SpinType>&, double& value) {
        value = ints_->aptei_ab(i[0], i[1], i[2], i[3]);
    });

    T["mnue"] = V["mnue"];
    if (ccvv_source_ == "ZERO") {
        BT_scaled_by_D(T);
    } else if (ccvv_source_ == "NORMAL") {
        BT_scaled_by_RD(T);
        BT_scaled_by_Rplus1(V);
    }
    temp["vu"] += 2.0 * V["mnve"] * T["mnue"];

    // modify T for the third term
    T = ambit::BlockedTensor::build(tensor_type_, "T2_ccva", {"ccva"});
    T.iterate([&](const std::vector<size_t>& i, const std::vector<SpinType>&, double& value) {
        value = ints_->aptei_ab(i[0], i[1], i[2], i[3]);
    });
    if (ccvv_source_ == "ZERO") {
        BT_scaled_by_D(T);
    } else if (ccvv_source_ == "NORMAL") {
        BT_scaled_by_RD(T);
    }
    temp["vu"] -= V["mnve"] * T["mneu"];

    //    // modify V for the second term
    //    V = ambit::BlockedTensor::build(tensor_type_,"V_ccva",{"ccva"});
    //    V.iterate([&](const std::vector<size_t>& i,const
    //    std::vector<SpinType>&,double& value){
    //        value = ints_->aptei_ab(i[0],i[1],i[2],i[3]);
    //    });
    //    if(ccvv_source_ == "NORMAL"){
    //        BT_scaled_by_Rplus1(V);
    //    }
    //    temp["vu"] += V["mnev"] * T["mneu"];

    E += temp["vu"] * Eta1_["uv"];
    return E;
}

void DSRG_MRPT::H2_T2_C0_L2(BlockedTensor& H2, BlockedTensor& T2, const double& alpha, double& C0) {
    local_timer timer;
    double E = 0.0;
    ambit::BlockedTensor temp = ambit::BlockedTensor::build(tensor_type_, "temp_ccaa", {"aaaa"});

    // HH
    temp["uvxy"] += 0.5 * H2["mnuv"] * T2["mnxy"];
    temp["uvxy"] += 0.5 * H2["mwuv"] * T2["mzxy"] * L1_["wz"];

    // PP
    temp["uvxy"] += 0.5 * H2["xyef"] * T2["uvef"];
    temp["uvxy"] += 0.5 * H2["xyez"] * T2["uvew"] * Eta1_["wz"];

    // HP
    temp["uvxy"] += 2.0 * H2["xmue"] * T2["vmye"];
    temp["uvxy"] -= H2["xmue"] * T2["mvye"];
    temp["uvxy"] -= H2["mxue"] * T2["vmye"];
    temp["uvxy"] -= H2["mxve"] * T2["umey"];

    // HP with Gamma1
    temp["uvxy"] += H2["wxeu"] * T2["zvey"] * L1_["wz"];
    temp["uvxy"] -= 0.5 * H2["wxeu"] * T2["vzey"] * L1_["wz"];
    temp["uvxy"] -= 0.5 * H2["xweu"] * T2["zvey"] * L1_["wz"];
    temp["uvxy"] -= 0.5 * H2["xwev"] * T2["uzey"] * L1_["wz"];

    // HP with Eta1
    temp["uvxy"] += H2["mxwu"] * T2["mvzy"] * Eta1_["wz"];
    temp["uvxy"] -= 0.5 * H2["mxwu"] * T2["mvyz"] * Eta1_["wz"];
    temp["uvxy"] -= 0.5 * H2["mxuw"] * T2["mvzy"] * Eta1_["wz"];
    temp["uvxy"] -= 0.5 * H2["mxvw"] * T2["muyz"] * Eta1_["wz"];

    E += temp["uvxy"] * L2_["uvxy"];

    E *= alpha;
    C0 += E;
    dsrg_time_.add("220", timer.get());
}

void DSRG_MRPT::H2_T2_C0_L3(BlockedTensor& H2, BlockedTensor& T2, const double& alpha, double& C0) {
    local_timer timer;
    double E = 0.0;

    E += H2["xyew"] * T2["uvez"] * L3_["xyzuwv"];
    E -= H2["mzuv"] * T2["mwxy"] * L3_["xyzuwv"];

    E *= alpha;
    C0 += E;
    dsrg_time_.add("220", timer.get());
}
} // namespace forte

