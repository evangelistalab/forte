#include <numeric>
#include <math.h>

#include <libpsio/psio.hpp>
#include <libpsio/psio.h>
#include <libmints/molecule.h>
#include <libqt/qt.h>

#include "so-mrdsrg.h"
#include "blockedtensorfactory.h"

using namespace ambit;

namespace psi{ namespace libadaptive{

SOMRDSRG::SOMRDSRG(Reference reference, boost::shared_ptr<Wavefunction> wfn,
                   Options &options, ExplorerIntegrals* ints,
                   std::shared_ptr<MOSpaceInfo> mo_space_info)
    : Wavefunction(options,_default_psio_lib_),
      reference_(reference),
      ints_(ints),
      mo_space_info_(mo_space_info),
      tensor_type_(kCore),
      BTF(new BlockedTensorFactory(options))
{
    // Copy the wavefunction information
    copy(wfn);

    print_method_banner({"Multireference Driven Similarity Renormalization Group","written by Francesco A. Evangelista"});
    if(options.get_bool("MEMORY_SUMMARY"))
    {
        BTF->print_memory_info();
    }

    startup();
    print_summary();
}

SOMRDSRG::~SOMRDSRG()
{
    cleanup();
}

void SOMRDSRG::startup()
{
    Eref = reference_.get_Eref();

    frozen_core_energy = ints_->frozen_core_energy();

    ncmopi_ = ints_->ncmopi();

    s_ = options_.get_double("DSRG_S");
    if(s_ < 0){
        outfile->Printf("\n  S parameter for DSRG must >= 0!");
        exit(1);
    }
    taylor_threshold_ = options_.get_int("TAYLOR_THRESHOLD");
    if(taylor_threshold_ <= 0){
        outfile->Printf("\n  Threshold for Taylor expansion must be an integer greater than 0!");
        exit(1);
    }
    taylor_order_ = int(0.5 * (15.0 / taylor_threshold_ + 1)) + 1;

    std::vector<size_t> rdocc = mo_space_info_->get_corr_abs_mo("RESTRICTED_DOCC");
    std::vector<size_t> actv  = mo_space_info_->get_corr_abs_mo("ACTIVE");
    std::vector<size_t> ruocc = mo_space_info_->get_corr_abs_mo("RESTRICTED_UOCC");

    std::vector<std::pair<size_t,SpinType> > rdocc_so;
    for (size_t p : rdocc)
        rdocc_so.push_back(std::make_pair(p,AlphaSpin));
    for (size_t p : rdocc)
        rdocc_so.push_back(std::make_pair(p,BetaSpin));

    std::vector<std::pair<size_t,SpinType> > actv_so;
    for (size_t p : actv)
        actv_so.push_back(std::make_pair(p,AlphaSpin));
    for (size_t p : actv)
        actv_so.push_back(std::make_pair(p,BetaSpin));

    std::vector<std::pair<size_t,SpinType> > ruocc_so;
    for (size_t p : ruocc)
        ruocc_so.push_back(std::make_pair(p,AlphaSpin));
    for (size_t p : ruocc)
        ruocc_so.push_back(std::make_pair(p,BetaSpin));

//    // Form the maps from MOs to orbital sets
//    for (size_t p = 0; p < acore_mos.size(); ++p) mos_to_acore[acore_mos[p]] = p;
//    for (size_t p = 0; p < bcore_mos.size(); ++p) mos_to_bcore[bcore_mos[p]] = p;
//    for (size_t p = 0; p < aactv_mos.size(); ++p) mos_to_aactv[aactv_mos[p]] = p;
//    for (size_t p = 0; p < bactv_mos.size(); ++p) mos_to_bactv[bactv_mos[p]] = p;
//    for (size_t p = 0; p < avirt_mos.size(); ++p) mos_to_avirt[avirt_mos[p]] = p;
//    for (size_t p = 0; p < bvirt_mos.size(); ++p) mos_to_bvirt[bvirt_mos[p]] = p;

    BTF->add_mo_space("c","mno",rdocc_so);
    BTF->add_mo_space("a","uvwxyz",actv_so);
    BTF->add_mo_space("v","efgh",actv_so);

    BTF->add_composite_mo_space("h","ijkl",{"c","a"});

    BTF->add_composite_mo_space("p","abcd",{"a","v"});

    BTF->add_composite_mo_space("g","pqrs",{"c","a","v"});

    H = BTF->build(tensor_type_,"H",{"gg"});
    V = BTF->build(tensor_type_,"V",{"gggg"});

    Gamma1 = BTF->build(tensor_type_,"Gamma1",{"hh"});
    Eta1 = BTF->build(tensor_type_,"Eta1",{"pp"});
    Lambda2 = BTF->build(tensor_type_,"Lambda2",{"aaaa"});
    Lambda3 = BTF->build(tensor_type_,"Lambda3",{"aaaaaa"});
    F = BTF->build(tensor_type_,"Fock",{"gg"});
    Delta1 = BTF->build(tensor_type_,"Delta1",{"hp"});
    Delta2 = BTF->build(tensor_type_,"Delta2",{"hhpp"});
    RDelta1 = BTF->build(tensor_type_,"RDelta1",{"hp"});
    RDelta2 = BTF->build(tensor_type_,"RDelta2",{"hhpp"});
    T1 = BTF->build(tensor_type_,"T1 Amplitudes",{"hp"});
    T2 = BTF->build(tensor_type_,"T2 Amplitudes",{"hhpp"});
//    RExp1 = BTF->build(tensor_type_,"RExp1",spin_cases({"hp"}));
//    RExp2 = BTF->build(tensor_type_,"RExp2",spin_cases({"hhpp"}));
//    Hbar1 = BTF->build(tensor_type_,"One-body Hbar",spin_cases({"hh"}));
//    Hbar2 = BTF->build(tensor_type_,"Two-body Hbar",spin_cases({"hhhh"}));

    H.iterate([&](const std::vector<size_t>& i,const std::vector<SpinType>& spin,double& value){
        if ((spin[0] == AlphaSpin) and (spin[1] == AlphaSpin))
            value = ints_->oei_a(i[0],i[1]);
        if ((spin[0] == BetaSpin) and (spin[1] == BetaSpin))
            value = ints_->oei_b(i[0],i[1]);
    });

    // Fill in the two-electron operator (V)
    V.iterate([&](const std::vector<size_t>& i,const std::vector<SpinType>& spin,double& value){
        if ((spin[0] == AlphaSpin) and (spin[1] == AlphaSpin) and (spin[2] == AlphaSpin) and (spin[3] == AlphaSpin))
            value = ints_->aptei_aa(i[0],i[1],i[2],i[3]);

        if ((spin[0] == AlphaSpin) and (spin[1] == BetaSpin) and (spin[2] == AlphaSpin) and (spin[3] == BetaSpin))
            value = +ints_->aptei_ab(i[0],i[1],i[2],i[3]);

        if ((spin[0] == AlphaSpin) and (spin[1] == BetaSpin) and (spin[2] == BetaSpin) and (spin[3] == AlphaSpin))
            value = -ints_->aptei_ab(i[0],i[1],i[3],i[2]);

        if ((spin[0] == BetaSpin) and (spin[1] == AlphaSpin) and (spin[2] == AlphaSpin) and (spin[3] == BetaSpin))
            value = -ints_->aptei_ab(i[1],i[0],i[2],i[3]);

        if ((spin[0] == BetaSpin) and (spin[1] == AlphaSpin) and (spin[2] == BetaSpin) and (spin[3] == AlphaSpin))
            value = +ints_->aptei_ab(i[1],i[0],i[3],i[2]);

        if ((spin[0] == BetaSpin) and (spin[1] == BetaSpin) and (spin[2] == BetaSpin) and (spin[3] == BetaSpin))
            value = ints_->aptei_bb(i[0],i[1],i[2],i[3]);
    });

    ambit::Tensor Gamma1_cc = Gamma1.block("cc");
    ambit::Tensor Gamma1_aa = Gamma1.block("aa");

    ambit::Tensor Eta1_aa = Eta1.block("aa");
    ambit::Tensor Eta1_vv = Eta1.block("vv");
//    ambit::Tensor Eta1_AA = Eta1.block("AA");
//    ambit::Tensor Eta1_VV = Eta1.block("VV");

    Gamma1_cc.iterate([&](const std::vector<size_t>& i,double& value){
        value = i[0] == i[1] ? 1.0 : 0.0;});
    Eta1_aa.iterate([&](const std::vector<size_t>& i,double& value){
        value = i[0] == i[1] ? 1.0 : 0.0;});
    Eta1_vv.iterate([&](const std::vector<size_t>& i,double& value){
        value = i[0] == i[1] ? 1.0 : 0.0;});

    Gamma1.print();

//    Gamma1_aa("pq") = reference_.L1a()("pq");

//    Eta1_VV.iterate([&](const std::vector<size_t>& i,double& value){
//        value = i[0] == i[1] ? 1.0 : 0.0;});

//    Gamma1_aa("pq") = reference_.L1a()("pq");
//    Gamma1_AA("pq") = reference_.L1b()("pq");

//    Eta1_aa("pq") -= reference_.L1a()("pq");
//    Eta1_AA("pq") -= reference_.L1b()("pq");

//    // Fill out Lambda2 and Lambda3
//    ambit::Tensor Lambda2_aa = Lambda2.block("aaaa");
//    ambit::Tensor Lambda2_aA = Lambda2.block("aAaA");
//    ambit::Tensor Lambda2_AA = Lambda2.block("AAAA");
//    Lambda2_aa("pqrs") = reference_.L2aa()("pqrs");
//    Lambda2_aA("pqrs") = reference_.L2ab()("pqrs");
//    Lambda2_AA("pqrs") = reference_.L2bb()("pqrs");

//    ambit::Tensor Lambda3_aaa = Lambda3.block("aaaaaa");
//    ambit::Tensor Lambda3_aaA = Lambda3.block("aaAaaA");
//    ambit::Tensor Lambda3_aAA = Lambda3.block("aAAaAA");
//    ambit::Tensor Lambda3_AAA = Lambda3.block("AAAAAA");
//    Lambda3_aaa("pqrstu") = reference_.L3aaa()("pqrstu");
//    Lambda3_aaA("pqrstu") = reference_.L3aab()("pqrstu");
//    Lambda3_aAA("pqrstu") = reference_.L3abb()("pqrstu");
//    Lambda3_AAA("pqrstu") = reference_.L3bbb()("pqrstu");

//    // Form the Fock matrix
//    F["pq"]  = H["pq"];
//    F["pq"] += V["pjqi"] * Gamma1["ij"];
//    F["pq"] += V["pJqI"] * Gamma1["IJ"];

//    F["PQ"]  = H["PQ"];
//    F["PQ"] += V["jPiQ"] * Gamma1["ij"];
//    F["PQ"] += V["PJQI"] * Gamma1["IJ"];

//    size_t ncmo_ = ints_->ncmo();
//    std::vector<double> Fa(ncmo_);
//    std::vector<double> Fb(ncmo_);

//    F.iterate([&](const std::vector<size_t>& i,const std::vector<SpinType>& spin,double& value){
//        if (spin[0] == AlphaSpin and (i[0] == i[1])){
//            Fa[i[0]] = value;
//        }
//        if (spin[0] == BetaSpin and (i[0] == i[1])){
//            Fb[i[0]] = value;
//        }
//    });

//    Delta1.iterate([&](const std::vector<size_t>& i,const std::vector<SpinType>& spin,double& value){
//        if (spin[0] == AlphaSpin){
//            value = Fa[i[0]] - Fa[i[1]];
//        }else if (spin[0]  == BetaSpin){
//            value = Fb[i[0]] - Fb[i[1]];
//        }
//    });

//    RDelta1.iterate([&](const std::vector<size_t>& i,const std::vector<SpinType>& spin,double& value){
//        if (spin[0]  == AlphaSpin){
//            value = renormalized_denominator(Fa[i[0]] - Fa[i[1]]);
//        }else if (spin[0]  == BetaSpin){
//            value = renormalized_denominator(Fb[i[0]] - Fb[i[1]]);
//        }
//    });

//    RDelta2.iterate([&](const std::vector<size_t>& i,const std::vector<SpinType>& spin,double& value){
//        if ((spin[0] == AlphaSpin) and (spin[1] == AlphaSpin)){
//            value = renormalized_denominator(Fa[i[0]] + Fa[i[1]] - Fa[i[2]] - Fa[i[3]]);
//        }else if ((spin[0] == AlphaSpin) and (spin[1] == BetaSpin) ){
//            value = renormalized_denominator(Fa[i[0]] + Fb[i[1]] - Fa[i[2]] - Fb[i[3]]);
//        }else if ((spin[0] == BetaSpin)  and (spin[1] == BetaSpin) ){
//            value = renormalized_denominator(Fb[i[0]] + Fb[i[1]] - Fb[i[2]] - Fb[i[3]]);
//        }
//    });

//    // Prepare exponential tensors for effective Fock matrix and integrals
//    RExp1.iterate([&](const std::vector<size_t>& i,const std::vector<SpinType>& spin,double& value){
//        if (spin[0]  == AlphaSpin){
//            value = renormalized_exp(Fa[i[0]] - Fa[i[1]]);
//        }else if (spin[0]  == BetaSpin){
//            value = renormalized_exp(Fb[i[0]] - Fb[i[1]]);
//        }
//    });

//    RExp2.iterate([&](const std::vector<size_t>& i,const std::vector<SpinType>& spin,double& value){
//        if ((spin[0] == AlphaSpin) and (spin[1] == AlphaSpin)){
//            value = renormalized_exp(Fa[i[0]] + Fa[i[1]] - Fa[i[2]] - Fa[i[3]]);
//        }else if ((spin[0] == AlphaSpin) and (spin[1] == BetaSpin) ){
//            value = renormalized_exp(Fa[i[0]] + Fb[i[1]] - Fa[i[2]] - Fb[i[3]]);
//        }else if ((spin[0] == BetaSpin)  and (spin[1] == BetaSpin) ){
//            value = renormalized_exp(Fb[i[0]] + Fb[i[1]] - Fb[i[2]] - Fb[i[3]]);
//        }
//    });

//    // Prepare Hbar
//    Hbar1["ij"] = F["ij"];
//    Hbar1["IJ"] = F["IJ"];
//    Hbar2["ijkl"] = V["ijkl"];
//    Hbar2["iJkL"] = V["iJkL"];
//    Hbar2["IJKL"] = V["IJKL"];
////    Hbar1.print(stdout);
////    Hbar2.print(stdout);

//    // Print levels
//    print_ = options_.get_int("PRINT");
//    if(print_ > 1){
//        Gamma1.print(stdout);
//        Eta1.print(stdout);
//        F.print(stdout);
//    }
//    if(print_ > 2){
//        V.print(stdout);
//        Lambda2.print(stdout);
//    }
//    if(print_ > 3){
//        Lambda3.print(stdout);
//    }
}

void SOMRDSRG::print_summary()
{
    // Print a summary
    std::vector<std::pair<std::string,int>> calculation_info;

    std::vector<std::pair<std::string,double>> calculation_info_double{
        {"flow parameter",s_},
        {"taylor expansion threshold",pow(10.0,-double(taylor_threshold_))}};

    std::vector<std::pair<std::string,std::string>> calculation_info_string{
        {"int_type", options_.get_str("INT_TYPE")},
        {"source operator", source_}};

    // Print some information
    outfile->Printf("\n\n  ==> Calculation Information <==\n");
    for (auto& str_dim : calculation_info){
        outfile->Printf("\n    %-39s %10d",str_dim.first.c_str(),str_dim.second);
    }
    for (auto& str_dim : calculation_info_double){
        outfile->Printf("\n    %-39s %10.3e",str_dim.first.c_str(),str_dim.second);
    }
    for (auto& str_dim : calculation_info_string){
        outfile->Printf("\n    %-39s %10s",str_dim.first.c_str(),str_dim.second.c_str());
    }
    outfile->Flush();
}

double SOMRDSRG::compute_energy()
{
    print_h2("Computing the SO-MR-DSRG(2) energy");
    return 0.0;
}

void SOMRDSRG::cleanup()
{
}

}} // End Namespaces
