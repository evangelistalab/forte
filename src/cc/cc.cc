/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2017 by its authors (see COPYING, COPYING.LESSER, AUTHORS).
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
#include <cmath>
#include <map>
#include <vector>

#include "psi4/libmints/molecule.h"

#include "cc/cc.h"
#include "helpers/mo_space_info.h"

namespace psi {
namespace forte {

CC::CC(SharedWavefunction ref_wfn, Options& options, std::shared_ptr<ForteIntegrals> ints,
       std::shared_ptr<MOSpaceInfo> mo_space_info)
    : Wavefunction(options), ints_(ints), mo_space_info_(mo_space_info),
      BTF_(new BlockedTensorFactory(options)), tensor_type_(CoreTensor) {
    startup();
}

/// Destructor
CC::~CC() {}

/// Compute the corr_level energy with fixed reference
double CC::compute_energy() { return 0.0; }

// MRDSRG::MRDSRG(Reference reference, SharedWavefunction ref_wfn, Options&
// options,
//               std::shared_ptr<ForteIntegrals> ints,
//               std::shared_ptr<MOSpaceInfo> mo_space_info)
//    : Wavefunction(options), reference_(reference), ints_(ints),
//      mo_space_info_(mo_space_info), BTF_(new BlockedTensorFactory(options)),
//      tensor_type_(CoreTensor)
//{
//    shallow_copy(ref_wfn);
//    reference_wavefunction_ = ref_wfn;

//    print_method_banner({"Multireference Driven Similarity Renormalization
//    Group","Chenyang Li"});
//    read_options();
//    print_options();
//
//}

// MRDSRG::~MRDSRG(){
//    cleanup();
//}

// void MRDSRG::cleanup(){
//    dsrg_time_.print_comm_time();
//}

// void MRDSRG::read_options(){

//    print_ = options_.get_int("PRINT");

//    s_ = options_.get_double("DSRG_S");
//    if(s_ < 0){
//        outfile->Printf("\n  S parameter for DSRG must >= 0!");
//        throw PSIEXCEPTION("S parameter for DSRG must >= 0!");
//    }
//    taylor_threshold_ = options_.get_int("TAYLOR_THRESHOLD");
//    if(taylor_threshold_ <= 0){
//        outfile->Printf("\n  Threshold for Taylor expansion must be an integer
//        greater than 0!");
//        throw PSIEXCEPTION("Threshold for Taylor expansion must be an integer
//        greater than 0!");
//    }

//    source_ = options_.get_str("SOURCE");
//    if(source_ != "STANDARD" && source_ != "LABS" && source_ != "DYSON"){
//        outfile->Printf("\n  Warning: SOURCE option \"%s\" is not implemented
//        in MRDSRG. Changed to STANDARD.", source_.c_str());
//        source_ = "STANDARD";
//    }
//    if(source_ == "STANDARD"){
//        dsrg_source_ = std::make_shared<STD_SOURCE>(s_,taylor_threshold_);
//    }else if(source_ == "LABS"){
//        dsrg_source_ = std::make_shared<LABS_SOURCE>(s_,taylor_threshold_);
//    }else if(source_ == "DYSON"){
//        dsrg_source_ = std::make_shared<DYSON_SOURCE>(s_,taylor_threshold_);
//    }

//    ntamp_ = options_.get_int("NTAMP");
//    intruder_tamp_ = options_.get_double("INTRUDER_TAMP");
//}

void CC::startup() {
    // frozen-core energy
    frozen_core_energy_ = ints_->frozen_core_energy();

    // orbital spaces
    BlockedTensor::reset_mo_spaces();
    aocc_mos_ = mo_space_info_->get_corr_abs_mo("RESTRICTED_DOCC");
    bocc_mos_ = mo_space_info_->get_corr_abs_mo("RESTRICTED_DOCC");
    avir_mos_ = mo_space_info_->get_corr_abs_mo("RESTRICTED_UOCC");
    bvir_mos_ = mo_space_info_->get_corr_abs_mo("RESTRICTED_UOCC");

    // define space labels
    aocc_label_ = "o";
    avir_label_ = "v";
    bocc_label_ = "o";
    bvir_label_ = "v";

    BTF_->add_mo_space(aocc_label_, "ijklmn", aocc_mos_, AlphaSpin);
    BTF_->add_mo_space(bocc_label_, "IJKLMN", bocc_mos_, BetaSpin);
    BTF_->add_mo_space(avir_label_, "abcdef", avir_mos_, AlphaSpin);
    BTF_->add_mo_space(bvir_label_, "ABCDEF", bvir_mos_, BetaSpin);

    //    // map space labels to mo spaces
    //    label_to_spacemo_[acore_label_[0]] = acore_mos_;
    //    label_to_spacemo_[bcore_label_[0]] = bcore_mos_;
    //    label_to_spacemo_[aactv_label_[0]] = aactv_mos_;
    //    label_to_spacemo_[bactv_label_[0]] = bactv_mos_;
    //    label_to_spacemo_[avirt_label_[0]] = avirt_mos_;
    //    label_to_spacemo_[bvirt_label_[0]] = bvirt_mos_;

    // define composite spaces
    BTF_->add_composite_mo_space("g", "pqrsto", {aocc_label_, avir_label_});
    BTF_->add_composite_mo_space("G", "PQRSTO", {bocc_label_, bvir_label_});

    // prepare integrals
    H_ = BTF_->build(tensor_type_, "H", spin_cases({"gg"}));
    V_ = BTF_->build(tensor_type_, "V", spin_cases({"gggg"}));
    //    build_ints();

    // build Fock matrix
    F_ = BTF_->build(tensor_type_, "Fock", spin_cases({"gg"}));
    //    build_fock(H_, V_);
}

// void MRDSRG::build_ints(){
//    // prepare one-electron integrals
//    H_.iterate([&](const std::vector<size_t>& i,const std::vector<SpinType>&
//    spin,double& value){
//        if (spin[0] == AlphaSpin) value = ints_->oei_a(i[0],i[1]);
//        else value = ints_->oei_b(i[0],i[1]);
//    });

//    // prepare two-electron integrals
//    V_.iterate([&](const std::vector<size_t>& i,const std::vector<SpinType>&
//    spin,double& value){
//        if ((spin[0] == AlphaSpin) && (spin[1] == AlphaSpin)) value =
//        ints_->aptei_aa(i[0],i[1],i[2],i[3]);
//        if ((spin[0] == AlphaSpin) && (spin[1] == BetaSpin))  value =
//        ints_->aptei_ab(i[0],i[1],i[2],i[3]);
//        if ((spin[0] == BetaSpin)  && (spin[1] == BetaSpin))  value =
//        ints_->aptei_bb(i[0],i[1],i[2],i[3]);
//    });
//}

// void MRDSRG::build_density(){
//    // prepare density matrices
//    (Gamma1_.block("cc")).iterate([&](const std::vector<size_t>& i,double&
//    value){
//        value = i[0] == i[1] ? 1.0 : 0.0;});
//    (Gamma1_.block("CC")).iterate([&](const std::vector<size_t>& i,double&
//    value){
//        value = i[0] == i[1] ? 1.0 : 0.0;});
//    (Eta1_.block("aa")).iterate([&](const std::vector<size_t>& i,double&
//    value){
//        value = i[0] == i[1] ? 1.0 : 0.0;});
//    (Eta1_.block("AA")).iterate([&](const std::vector<size_t>& i,double&
//    value){
//        value = i[0] == i[1] ? 1.0 : 0.0;});
//    (Eta1_.block("vv")).iterate([&](const std::vector<size_t>& i,double&
//    value){
//        value = i[0] == i[1] ? 1.0 : 0.0;});
//    (Eta1_.block("VV")).iterate([&](const std::vector<size_t>& i,double&
//    value){
//        value = i[0] == i[1] ? 1.0 : 0.0;});
//    // symmetrize beta spin
//    outfile->Printf("\n  Warning: I am forcing density Db = Da to avoid spin
//    symmetry breaking.");
//    outfile->Printf("\n  If this is not desired, go to mrdsrg.cc
//    build_density() around line 190.");
//    Gamma1_.block("aa")("pq") = reference_.L1a()("pq");
//    Gamma1_.block("AA")("pq") = reference_.L1a()("pq");
//    Eta1_.block("aa")("pq") -= reference_.L1a()("pq");
//    Eta1_.block("AA")("pq") -= reference_.L1a()("pq");

////    ambit::Tensor Diff =
/// ambit::Tensor::build(tensor_type_,"Diff",reference_.L1a().dims());
////    Diff.data() = reference_.L1a().data();
////    Diff("pq") -= reference_.L1b()("pq");
////    outfile->Printf("\n  L1a diff Here !!!!");
////    Diff.citerate([&](const std::vector<size_t>& i,const double& value){
////        if(value != 0.0){
////            outfile->Printf("\n  [%zu][%zu] = %20.15f",i[0],i[1],value);
////        }
////    });

//    // prepare two-body density cumulants
//    ambit::Tensor Lambda2_aa = Lambda2_.block("aaaa");
//    ambit::Tensor Lambda2_aA = Lambda2_.block("aAaA");
//    ambit::Tensor Lambda2_AA = Lambda2_.block("AAAA");
//    Lambda2_aa("pqrs") = reference_.L2aa()("pqrs");
//    Lambda2_aA("pqrs") = reference_.L2ab()("pqrs");
//    Lambda2_AA("pqrs") = reference_.L2bb()("pqrs");

////    Diff =
/// ambit::Tensor::build(tensor_type_,"Diff",reference_.L2aa().dims());
////    Diff.data() = reference_.L2aa().data();
////    Diff("pqrs") -= reference_.L2bb()("pqrs");
////    outfile->Printf("\n  L2aa diff Here !!!!");
////    Diff.citerate([&](const std::vector<size_t>& i,const double& value){
////        if(value != 0.0){
////            outfile->Printf("\n  [%zu][%zu][%zu][%zu] =
///%20.15f",i[0],i[1],i[2],i[3],value);
////        }
////    });

//    // prepare three-body density cumulants
//    if(options_.get_str("THREEPDC") != "ZERO"){
//        ambit::Tensor Lambda3_aaa = Lambda3_.block("aaaaaa");
//        ambit::Tensor Lambda3_aaA = Lambda3_.block("aaAaaA");
//        ambit::Tensor Lambda3_aAA = Lambda3_.block("aAAaAA");
//        ambit::Tensor Lambda3_AAA = Lambda3_.block("AAAAAA");
//        Lambda3_aaa("pqrstu") = reference_.L3aaa()("pqrstu");
//        Lambda3_aaA("pqrstu") = reference_.L3aab()("pqrstu");
//        Lambda3_aAA("pqrstu") = reference_.L3abb()("pqrstu");
//        Lambda3_AAA("pqrstu") = reference_.L3bbb()("pqrstu");
//    }

//    // check cumulants
//    print_cumulant_summary();
//}

// void MRDSRG::build_fock(BlockedTensor& H, BlockedTensor& V){
//    // build Fock matrix
//    F_["pq"]  = H["pq"];
//    F_["pq"] += V["pjqi"] * Gamma1_["ij"];
//    F_["pq"] += V["pJqI"] * Gamma1_["IJ"];
//    F_["PQ"]  = H["PQ"];
//    F_["PQ"] += V["jPiQ"] * Gamma1_["ij"];
//    F_["PQ"] += V["PJQI"] * Gamma1_["IJ"];

//    // obtain diagonal elements of Fock matrix
//    size_t ncmo_ = mo_space_info_->size("CORRELATED");
//    Fa_ = std::vector<double>(ncmo_);
//    Fb_ = std::vector<double>(ncmo_);
//    F_.iterate([&](const std::vector<size_t>& i,const std::vector<SpinType>&
//    spin,double& value){
//        if (spin[0] == AlphaSpin and (i[0] == i[1])){
//            Fa_[i[0]] = value;
//        }
//        if (spin[0] == BetaSpin and (i[0] == i[1])){
//            Fb_[i[0]] = value;
//        }
//    });
//}

// void MRDSRG::print_options()
//{
//    // fill in information
//    std::vector<std::pair<std::string,int>> calculation_info{
//        {"ntamp", ntamp_},
//        {"diis_min_vecs", options_.get_int("DIIS_MIN_VECS")},
//        {"diis_max_vecs", options_.get_int("DIIS_MAX_VECS")}};

//    std::vector<std::pair<std::string,double>> calculation_info_double{
//        {"flow parameter",s_},
//        {"taylor expansion threshold",pow(10.0,-double(taylor_threshold_))},
//        {"intruder_tamp", intruder_tamp_}};

//    std::vector<std::pair<std::string,std::string>> calculation_info_string{
//        {"corr_level", options_.get_str("CORR_LEVEL")},
//        {"int_type", options_.get_str("INT_TYPE")},
//        {"source operator", source_},
//        {"smart_dsrg_s", options_.get_str("SMART_DSRG_S")},
//        {"reference relaxation", options_.get_str("RELAX_REF")},
//        {"dsrg transformation type", options_.get_str("DSRG_TRANS_TYPE")},
//        {"core virtual source type", options_.get_str("CCVV_SOURCE")}};

//    // print some information
//    print_h2("Calculation Information");
//    for (auto& str_dim : calculation_info){
//        outfile->Printf("\n    %-35s
//        %15d",str_dim.first.c_str(),str_dim.second);
//    }
//    for (auto& str_dim : calculation_info_double){
//        outfile->Printf("\n    %-35s
//        %15.3e",str_dim.first.c_str(),str_dim.second);
//    }
//    for (auto& str_dim : calculation_info_string){
//        outfile->Printf("\n    %-35s
//        %15s",str_dim.first.c_str(),str_dim.second.c_str());
//    }
//    outfile->Printf("\n");
//
//}
}
}
