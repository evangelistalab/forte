#include <numeric>

#include <libpsio/psio.hpp>
#include <libpsio/psio.h>
#include <libmints/molecule.h>
#include <libqt/qt.h>

#include "dsrg_mrpt2.h"

using namespace ambit;

namespace psi{ namespace libadaptive{

DSRG_MRPT2::DSRG_MRPT2(Reference reference, boost::shared_ptr<Wavefunction> wfn, Options &options, ExplorerIntegrals* ints)
    : Wavefunction(options,_default_psio_lib_), reference_(reference), ints_(ints), tensor_type_(kCore)
{
    // Copy the wavefunction information
    copy(wfn);

    outfile->Printf("\n\n\t  ---------------------------------------------------------");
    outfile->Printf("\n\t      Driven Similarity Renormalization Group MBPT2");
    outfile->Printf("\n\t  ---------------------------------------------------------");

    startup();
    print_summary();
}

DSRG_MRPT2::~DSRG_MRPT2()
{
    cleanup();
}

void DSRG_MRPT2::startup()
{
    Eref = reference_.get_Eref();
    outfile->Printf("\n  Reference Energy = %.15f", Eref);

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

    rdoccpi_ = Dimension (nirrep_, "Restricted Occupied MOs");
    actvpi_  = Dimension (nirrep_, "Active MOs");
    ruoccpi_ = Dimension (nirrep_, "Restricted Unoccupied MOs");
    if (options_["RESTRICTED_DOCC"].size() == nirrep_){
        for (int h = 0; h < nirrep_; ++h){
            rdoccpi_[h] = options_["RESTRICTED_DOCC"][h].to_integer();
        }
    }else{
        outfile->Printf("\n  The size of RESTRICTED_DOCC occupation does not match the number of Irrep.");
        exit(1);
    }
    if ((options_["ACTIVE"].has_changed()) && (options_["ACTIVE"].size() == nirrep_)){
        for (int h = 0; h < nirrep_; ++h){
            actvpi_[h] = options_["ACTIVE"][h].to_integer();
            ruoccpi_[h] = ncmopi_[h] - rdoccpi_[h] - actvpi_[h];
        }
    }else{
        outfile->Printf("\n  The size of ACTIVE occupation does not match the number of Irrep.");
        exit(1);
    }

    // Populate the core, active, and virtuall arrays
    for (int h = 0, p = 0; h < nirrep_; ++h){
        for (int i = 0; i < rdoccpi_[h]; ++i,++p){
            acore_mos.push_back(p);
            bcore_mos.push_back(p);
        }
        for (int i = 0; i < actvpi_[h]; ++i,++p){
            aactv_mos.push_back(p);
            bactv_mos.push_back(p);
        }
        for (int a = 0; a < ruoccpi_[h]; ++a,++p){
            avirt_mos.push_back(p);
            bvirt_mos.push_back(p);
        }
    }

    // Form the maps from MOs to orbital sets
    for (size_t p = 0; p < acore_mos.size(); ++p) mos_to_acore[acore_mos[p]] = p;
    for (size_t p = 0; p < bcore_mos.size(); ++p) mos_to_bcore[bcore_mos[p]] = p;
    for (size_t p = 0; p < aactv_mos.size(); ++p) mos_to_aactv[aactv_mos[p]] = p;
    for (size_t p = 0; p < bactv_mos.size(); ++p) mos_to_bactv[bactv_mos[p]] = p;
    for (size_t p = 0; p < avirt_mos.size(); ++p) mos_to_avirt[avirt_mos[p]] = p;
    for (size_t p = 0; p < bvirt_mos.size(); ++p) mos_to_bvirt[bvirt_mos[p]] = p;

    BlockedTensor::add_mo_space("c","mn",acore_mos,AlphaSpin);
    BlockedTensor::add_mo_space("C","MN",bcore_mos,BetaSpin);

    BlockedTensor::add_mo_space("a","uvwxyz",aactv_mos,AlphaSpin);
    BlockedTensor::add_mo_space("A","UVWXYZ",bactv_mos,BetaSpin);

    BlockedTensor::add_mo_space("v","ef",avirt_mos,AlphaSpin);
    BlockedTensor::add_mo_space("V","EF",bvirt_mos,BetaSpin);

    BlockedTensor::add_composite_mo_space("h","ijkl",{"c","a"});
    BlockedTensor::add_composite_mo_space("H","IJKL",{"C","A"});

    BlockedTensor::add_composite_mo_space("p","abcd",{"a","v"});
    BlockedTensor::add_composite_mo_space("P","ABCD",{"A","V"});

    BlockedTensor::add_composite_mo_space("g","pqrs",{"c","a","v"});
    BlockedTensor::add_composite_mo_space("G","PQRS",{"C","A","V"});

    H = BlockedTensor::build(tensor_type_,"H",spin_cases({"gg"}));
    V = BlockedTensor::build(tensor_type_,"V",spin_cases({"gggg"}));

    Gamma1 = BlockedTensor::build(tensor_type_,"Gamma1",spin_cases({"hh"}));
    Eta1 = BlockedTensor::build(tensor_type_,"Eta1",spin_cases({"pp"}));
    Lambda2 = BlockedTensor::build(tensor_type_,"Lambda2",spin_cases({"aaaa"}));
    Lambda3 = BlockedTensor::build(tensor_type_,"Lambda3",spin_cases({"aaaaaa"}));
    F = BlockedTensor::build(tensor_type_,"Fock",spin_cases({"gg"}));
    Delta1 = BlockedTensor::build(tensor_type_,"Delta1",spin_cases({"hp"}));
    Delta2 = BlockedTensor::build(tensor_type_,"Delta2",spin_cases({"hhpp"}));
    RDelta1 = BlockedTensor::build(tensor_type_,"RDelta1",spin_cases({"hp"}));
    RDelta2 = BlockedTensor::build(tensor_type_,"RDelta2",spin_cases({"hhpp"}));
    T1 = BlockedTensor::build(tensor_type_,"T1 Amplitudes",spin_cases({"hp"}));
    T2 = BlockedTensor::build(tensor_type_,"T2 Amplitudes",spin_cases({"hhpp"}));
    RExp1 = BlockedTensor::build(tensor_type_,"RExp1",spin_cases({"hp"}));
    RExp2 = BlockedTensor::build(tensor_type_,"RExp2",spin_cases({"hhpp"}));

    H.iterate([&](const std::vector<size_t>& i,const std::vector<SpinType>& spin,double& value){
        if (spin[0] == AlphaSpin)
            value = ints_->oei_a(i[0],i[1]);
        else
            value = ints_->oei_b(i[0],i[1]);
    });

    ambit::Tensor Gamma1_cc = Gamma1.block("cc");
    ambit::Tensor Gamma1_aa = Gamma1.block("aa");
    ambit::Tensor Gamma1_CC = Gamma1.block("CC");
    ambit::Tensor Gamma1_AA = Gamma1.block("AA");

    ambit::Tensor Eta1_aa = Eta1.block("aa");
    ambit::Tensor Eta1_vv = Eta1.block("vv");
    ambit::Tensor Eta1_AA = Eta1.block("AA");
    ambit::Tensor Eta1_VV = Eta1.block("VV");

    Gamma1_cc.iterate([&](const std::vector<size_t>& i,double& value){
        value = i[0] == i[1] ? 1.0 : 0.0;});
    Gamma1_CC.iterate([&](const std::vector<size_t>& i,double& value){
        value = i[0] == i[1] ? 1.0 : 0.0;});

    Eta1_aa.iterate([&](const std::vector<size_t>& i,double& value){
        value = i[0] == i[1] ? 1.0 : 0.0;});
    Eta1_AA.iterate([&](const std::vector<size_t>& i,double& value){
        value = i[0] == i[1] ? 1.0 : 0.0;});

    Eta1_vv.iterate([&](const std::vector<size_t>& i,double& value){
        value = i[0] == i[1] ? 1.0 : 0.0;});
    Eta1_VV.iterate([&](const std::vector<size_t>& i,double& value){
        value = i[0] == i[1] ? 1.0 : 0.0;});

    Gamma1_aa("pq") = reference_.L1a()("pq");
    Gamma1_AA("pq") = reference_.L1b()("pq");

    Eta1_aa("pq") -= reference_.L1a()("pq");
    Eta1_AA("pq") -= reference_.L1b()("pq");

    // Fill in the two-electron operator (V)
    V.iterate([&](const std::vector<size_t>& i,const std::vector<SpinType>& spin,double& value){
        if ((spin[0] == AlphaSpin) and (spin[1] == AlphaSpin)) value = ints_->aptei_aa(i[0],i[1],i[2],i[3]);
        if ((spin[0] == AlphaSpin) and (spin[1] == BetaSpin) ) value = ints_->aptei_ab(i[0],i[1],i[2],i[3]);
        if ((spin[0] == BetaSpin)  and (spin[1] == BetaSpin) ) value = ints_->aptei_bb(i[0],i[1],i[2],i[3]);
    });
    // Form the Fock matrix
    F["pq"]  = H["pq"];
    F["pq"] += V["pjqi"] * Gamma1["ij"];
    F["pq"] += V["pJqI"] * Gamma1["IJ"];

    F["PQ"] += H["PQ"];
    F["PQ"] += V["jPiQ"] * Gamma1["ij"];
    F["PQ"] += V["PJQI"] * Gamma1["IJ"];

  
    Tensor Fa_cc = F.block("cc");
    Tensor Fa_aa = F.block("aa");
    Tensor Fa_vv = F.block("vv");
    Tensor Fb_CC = F.block("CC");
    Tensor Fb_AA = F.block("AA");
    Tensor Fb_VV = F.block("VV");

    size_t ncmo_ = ints_->ncmo();
    std::vector<double> Fa(ncmo_);
    std::vector<double> Fb(ncmo_);

    F.iterate([&](const std::vector<size_t>& i,const std::vector<SpinType>& spin,double& value){
        if (spin[0] == AlphaSpin and (i[0] == i[1])){
            Fa[i[0]] = value;
        }
        if (spin[0] == BetaSpin and (i[0] == i[1])){
            Fb[i[0]] = value;
        }
    });

    Delta1.iterate([&](const std::vector<size_t>& i,const std::vector<SpinType>& spin,double& value){
        if (spin[0] == AlphaSpin){
            value = Fa[i[0]] - Fa[i[1]];
        }else if (spin[0]  == BetaSpin){
            value = Fb[i[0]] - Fb[i[1]];
        }
    });

    RDelta1.iterate([&](const std::vector<size_t>& i,const std::vector<SpinType>& spin,double& value){
        if (spin[0]  == AlphaSpin){
            value = renormalized_denominator(Fa[i[0]] - Fa[i[1]]);
        }else if (spin[0]  == BetaSpin){
            value = renormalized_denominator(Fb[i[0]] - Fb[i[1]]);
        }
    });

    RDelta2.iterate([&](const std::vector<size_t>& i,const std::vector<SpinType>& spin,double& value){
        if ((spin[0] == AlphaSpin) and (spin[1] == AlphaSpin)){
            value = renormalized_denominator(Fa[i[0]] + Fa[i[1]] - Fa[i[2]] - Fa[i[3]]);
        }else if ((spin[0] == AlphaSpin) and (spin[1] == BetaSpin) ){
            value = renormalized_denominator(Fa[i[0]] + Fb[i[1]] - Fa[i[2]] - Fb[i[3]]);
        }else if ((spin[0] == BetaSpin)  and (spin[1] == BetaSpin) ){
            value = renormalized_denominator(Fb[i[0]] + Fb[i[1]] - Fb[i[2]] - Fb[i[3]]);
        }
    });
    // Fill out Lambda2 and Lambda3
    Tensor Lambda2_aa = Lambda2.block("aaaa");
    Tensor Lambda2_aA = Lambda2.block("aAaA");
    Tensor Lambda2_AA = Lambda2.block("AAAA");
    Lambda2_aa("pqrs") = reference_.L2aa()("pqrs");
    Lambda2_aA("pqrs") = reference_.L2ab()("pqrs");
    Lambda2_AA("pqrs") = reference_.L2bb()("pqrs");

    Tensor Lambda3_aaa = Lambda3.block("aaaaaa");
    Tensor Lambda3_aaA = Lambda3.block("aaAaaA");
    Tensor Lambda3_aAA = Lambda3.block("aAAaAA");
    Tensor Lambda3_AAA = Lambda3.block("AAAAAA");
    Lambda3_aaa("pqrstu") = reference_.L3aaa()("pqrstu");
    Lambda3_aaA("pqrstu") = reference_.L3aab()("pqrstu");
    Lambda3_aAA("pqrstu") = reference_.L3abb()("pqrstu");
    Lambda3_AAA("pqrstu") = reference_.L3bbb()("pqrstu");

    // Prepare exponential tensors for effective Fock matrix and integrals

    RExp1.iterate([&](const std::vector<size_t>& i,const std::vector<SpinType>& spin,double& value){
        if (spin[0]  == AlphaSpin){
            value = renormalized_exp(Fa[i[0]] - Fa[i[1]]);
        }else if (spin[0]  == BetaSpin){
            value = renormalized_exp(Fb[i[0]] - Fb[i[1]]);
        }
    });

    RExp2.iterate([&](const std::vector<size_t>& i,const std::vector<SpinType>& spin,double& value){
        if ((spin[0] == AlphaSpin) and (spin[1] == AlphaSpin)){
            value = renormalized_exp(Fa[i[0]] + Fa[i[1]] - Fa[i[2]] - Fa[i[3]]);
        }else if ((spin[0] == AlphaSpin) and (spin[1] == BetaSpin) ){
            value = renormalized_exp(Fa[i[0]] + Fb[i[1]] - Fa[i[2]] - Fb[i[3]]);
        }else if ((spin[0] == BetaSpin)  and (spin[1] == BetaSpin) ){
            value = renormalized_exp(Fb[i[0]] + Fb[i[1]] - Fb[i[2]] - Fb[i[3]]);
        }
    });

    // Print levels
    print_ = options_.get_int("PRINT");
    if(print_ > 1){
        Gamma1.print(stdout);
        Eta1.print(stdout);
        F.print(stdout);
    }
    if(print_ > 2){
        V.print(stdout);
        Lambda2.print(stdout);
    }
    if(print_ > 3){
        Lambda3.print(stdout);
    }
}

void DSRG_MRPT2::print_summary()
{
    // Print a summary
    std::vector<std::pair<std::string,int>> calculation_info;

    std::vector<std::pair<std::string,double>> calculation_info_double{
        {"Flow parameter",s_},
        {"Taylor expansion threshold",std::pow(10.0,-double(taylor_threshold_))}};

    std::vector<std::pair<std::string,std::string>> calculation_info_string{
        {"int_type", options_.get_str("INT_TYPE")}};

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

void DSRG_MRPT2::cleanup()
{
}

double DSRG_MRPT2::renormalized_denominator(double D)
{
    double Z = std::sqrt(s_) * D;
    if(std::fabs(Z) < std::pow(0.1, taylor_threshold_)){
        return Taylor_Exp(Z, taylor_order_) * std::sqrt(s_);
    }else{
        return (1.0 - std::exp(-s_ * std::pow(D, 2.0))) / D;
    }
}

// Computes (1 - exp(-s D^2/V^2))
double DSRG_MRPT2::renormalized_denominator_amp(double V,double D)
{
    double Z = std::sqrt(s_) * (D / V);
    if (V == 0.0){
        return 0.0;
    }
    if(std::fabs(Z) < std::pow(0.1, taylor_threshold_)){
        return V * Taylor_Exp(Z, taylor_order_) * std::sqrt(s_);
    }else{
        return (1.0 - std::exp(-s_ * std::pow(D/V, 2.0))) / D;
    }
}

double DSRG_MRPT2::compute_energy()
{
    // Compute reference
//    Eref = compute_ref();

    // Compute T2 and T1
    compute_t2();
    compute_t1();

    // Compute effective integrals
    renormalize_V();
    renormalize_F();       
    if(print_ > 1)  F.print(stdout); // The actv-actv block is different but OK.
    if(print_ > 2){
        T1.print(stdout);
        T2.print(stdout);
        V.print(stdout);
    }

    // Compute DSRG-MRPT2 correlation energy
    double Ecorr = 0.0;
    timer_on("E_FT1");
    Ecorr += E_FT1();
    timer_off("E_FT1");

    timer_on("E_VT1");
    Ecorr += E_VT1();
    timer_off("E_VT1");

    timer_on("E_FT2");
    Ecorr += E_FT2();
    timer_off("E_FT2");

    timer_on("E_VT2_2");
    Ecorr += E_VT2_2();
    timer_off("E_VT2_2");

    timer_on("E_VT2_4HH");
    Ecorr += E_VT2_4HH();
    timer_off("E_VT2_4HH");

    timer_on("E_VT2_4PP");
    Ecorr += E_VT2_4PP();
    timer_off("E_VT2_4PP");

    timer_on("E_VT2_4PH");
    Ecorr += E_VT2_4PH();
    timer_off("E_VT2_4PH");

    timer_on("E_VT2_6");
    Ecorr += E_VT2_6();
    timer_off("E_VT2_6");

    outfile->Printf("\n  E(DSRG-PT2) %17c = %22.15lf", ' ', Ecorr);
    Process::environment.globals["CURRENT ENERGY"] = Ecorr + Eref;
    return Ecorr + Eref;
}

double DSRG_MRPT2::compute_ref()
{
    timer_on("Compute Eref");
    double E = 0.0;

    E  = 0.5 * H["ij"] * Gamma1["ij"];
    E += 0.5 * F["ij"] * Gamma1["ij"];
    E += 0.5 * H["IJ"] * Gamma1["IJ"];
    E += 0.5 * F["IJ"] * Gamma1["IJ"];

    E += 0.25 * V["uvxy"] * Lambda2["uvxy"];
    E += 0.25 * V["UVXY"] * Lambda2["UVXY"];
    E += V["uVxY"] * Lambda2["uVxY"];

    boost::shared_ptr<Molecule> molecule = Process::environment.molecule();
    double Enuc = molecule->nuclear_repulsion_energy();

    timer_off("Compute Eref");
    return E + frozen_core_energy + Enuc;
}

void DSRG_MRPT2::compute_t2()
{
    timer_on("Compute T2");
    T2["ijab"] = V["ijab"] * RDelta2["ijab"];
    T2["iJaB"] = V["iJaB"] * RDelta2["iJaB"];
    T2["IJAB"] = V["IJAB"] * RDelta2["IJAB"];

    // zero internal amplitudes
    T2.block("aaaa").zero();
    T2.block("aAaA").zero();
    T2.block("AAAA").zero();

    // norm and maximum of T2 amplitudes
    T2norm = 0.0; T2max = 0.0;
//    std::map<std::vector<std::string>,SharedTensor>& T2blocks = T2.blocks();
//    for(const auto& block: T2blocks){
//        std::vector<std::string> index_vec = block.first;
//        std::string name;
//        for(const std::string& index: index_vec)
//            name += index;

//        if(islower(name[0]) && isupper(name[1])){
//            T2norm += 4 * pow(T2.block(name)->norm(), 2.0);
//        }else{
//            T2norm += pow(T2.block(name)->norm(), 2.0);
//        }
////        double max = T2.block(name)->max_abs_vec()[0];
////        T2max = T2max > max ? T2max : max;
//    }
    T2norm = sqrt(T2norm);    
    outfile->Printf("\n  ||T2|| %22c = %22.15lf", ' ', T2.norm());
//    outfile->Printf("\n  max(T2) %21c = %22.15lf", ' ', T2max);
    timer_off("Compute T2");
}

void DSRG_MRPT2::compute_t1()
{
    timer_on("Compute T1");
    //A temporary tensor to use for the building of T1
    //Francesco's library does not handle repeating indices between 3 different terms, so need to form an intermediate
    //via a pointwise multiplcation
    BlockedTensor temp = BlockedTensor::build(tensor_type_,"temp",spin_cases({"aa"}));
    temp["xu"] = Gamma1["xu"] * Delta1["xu"];
    temp["XU"] = Gamma1["XU"] * Delta1["XU"];

    //Form the T1 amplitudes
    //Note:  The equations are changed slightly from York's equations.
    //Tensor libary does not handle beta alpha beta alpha, only alpha beta alpha beta.
    //Did some permuting to get the correct format

    BlockedTensor N = BlockedTensor::build(tensor_type_,"N",spin_cases({"hp"}));

    N["ia"]  = F["ia"];
    N["ia"] += temp["xu"] * T2["iuax"];
    N["ia"] += temp["XU"] * T2["iUaX"];

    N["IA"]  = F["IA"];
    N["IA"] += temp["xu"] * T2["uIxA"];
    N["IA"] += temp["XU"] * T2["IUAX"];
    //->    DN["ia"] = Delta1["ia"] / N["ia"];

//    if (){
//        std::vector<std::string> blocks(spin_cases({"ca","cv","av"}));
//        for (const std::string& block : blocks){
//            std::vector<double>& rd = RDelta1.block(block).data();
//            std::vector<double>& n = N.block(block).data();
//            std::vector<double>& d = Delta1.block(block).data();
//            for (size_t i = 0; i < rd.size(); ++i){
//                rd[i] = renormalized_exp_amp(n[i],d[i]);
//            }
//        }
//    }

    T1["ia"] = N["ia"] * RDelta1["ia"];

    T1["IA"] = N["IA"] * RDelta1["IA"];

    T1.block("AA").zero();
    T1.block("aa").zero();

    // norm and maximum of T1 amplitudes
    double Fnorm = F.norm(2);
    double RDelta1norm = RDelta1.norm(2);
//    T1norm = T1.norm(2); T1max = 0.0;
//    std::map<std::vector<std::string>,SharedTensor>& T1blocks = T1.blocks();
//    for(const auto& block: T1blocks){
//        std::vector<std::string> index_vec = block.first;
//        std::string name;
//        for(const std::string& index: index_vec)
//            name += index;
//        double max = T1.block(name)->max_abs_vec()[0];
//        T1max = T1max > max ? T1max : max;
//    }
    outfile->Printf("\n  ||T1|| %22c = %22.15lf", ' ', T1.norm());
//    outfile->Printf("\n  max(T1) %21c = %22.15lf", ' ', T1max);
    timer_off("Compute T1");
}

void DSRG_MRPT2::renormalize_V()
{
    timer_on("Renorm. V");
    V["ijab"] += V["ijab"] * RExp2["ijab"];
    V["iJaB"] += V["iJaB"] * RExp2["iJaB"];
    V["IJAB"] += V["IJAB"] * RExp2["IJAB"];

    V["abij"] += V["abij"] * RExp2["ijab"];
    V["aBiJ"] += V["aBiJ"] * RExp2["iJaB"];
    V["ABIJ"] += V["ABIJ"] * RExp2["IJAB"];
    timer_off("Renorm. V");
}

void DSRG_MRPT2::renormalize_F()
{
    timer_on("Renorm. F");
    BlockedTensor temp_aa = BlockedTensor::build(tensor_type_,"temp_aa",spin_cases({"aa"}));
    temp_aa["xu"] = Gamma1["xu"] * Delta1["xu"];
    temp_aa["XU"] = Gamma1["XU"] * Delta1["XU"];

    BlockedTensor temp1 = BlockedTensor::build(tensor_type_,"temp1",spin_cases({"hp"}));
    BlockedTensor temp2 = BlockedTensor::build(tensor_type_,"temp2",spin_cases({"hp"}));

    temp1["ia"] += temp_aa["xu"] * T2["iuax"];
    temp1["ia"] += temp_aa["XU"] * T2["iUaX"];
    temp2["ia"] += F["ia"] * RExp1["ia"];
    temp2["ia"] += temp1["ia"] * RExp1["ia"];

    temp1["IA"] += temp_aa["xu"] * T2["uIxA"];
    temp1["IA"] += temp_aa["XU"] * T2["IUAX"];
    temp2["IA"] += F["IA"] * RExp1["IA"];
    temp2["IA"] += temp1["IA"] * RExp1["IA"];

//    temp["ai"] += temp_aa["xu"] * T2["iuax"];
//    temp["ai"] += temp_aa["XU"] * T2["iUaX"];
//    F["ai"] += F["ai"] % RExp1["ia"];  // TODO <- is this legal in ambit???
//    F["ai"] += temp["ai"] % RExp1["ia"];
    F["ia"] += temp2["ia"];
    temp2.block("aa").zero();
    F["ai"] += temp2["ia"];

//    temp["AI"] += temp_aa["xu"] * T2["uIxA"];
//    temp["AI"] += temp_aa["XU"] * T2["IUAX"];
//    F["AI"] += F["AI"] % RExp1["IA"];
//    F["AI"] += temp["AI"] % RExp1["IA"];
    F["IA"] += temp2["IA"];
    temp2.block("AA").zero();
    F["AI"] += temp2["IA"];
    timer_off("Renorm. F");
}

double DSRG_MRPT2::E_FT1()
{
    double E = 0.0;
    BlockedTensor temp;
    temp = BlockedTensor::build(tensor_type_,"temp",spin_cases({"hp"}));

    temp["jb"] += T1["ia"] * Eta1["ab"] * Gamma1["ji"];
    temp["JB"] += T1["IA"] * Eta1["AB"] * Gamma1["JI"];

    E += temp["jb"] * F["bj"];
    E += temp["JB"] * F["BJ"];

    outfile->Printf("\n  E([F, T1]) %18c = %22.15lf", ' ', E);
    return E;
}

double DSRG_MRPT2::E_VT1()
{
    double E = 0.0;
    BlockedTensor temp;
    temp = BlockedTensor::build(tensor_type_,"temp", spin_cases({"aaaa"}));

    temp["uvxy"] += V["evxy"] * T1["ue"];
    temp["uvxy"] -= V["uvmy"] * T1["mx"];

    temp["UVXY"] += V["EVXY"] * T1["UE"];
    temp["UVXY"] -= V["UVMY"] * T1["MX"];

    temp["uVxY"] += V["eVxY"] * T1["ue"];
    temp["uVxY"] += V["uExY"] * T1["VE"];
    temp["uVxY"] -= V["uVmY"] * T1["mx"];
    temp["uVxY"] -= V["uVxM"] * T1["MY"];

    E += 0.5 * temp["uvxy"] * Lambda2["xyuv"];
    E += 0.5 * temp["UVXY"] * Lambda2["XYUV"];
    E += temp["uVxY"] * Lambda2["xYuV"];

    outfile->Printf("\n  E([V, T1]) %18c = %22.15lf", ' ', E);
    return E;
}

double DSRG_MRPT2::E_FT2()
{
    double E = 0.0;
    BlockedTensor temp;
    temp = BlockedTensor::build(tensor_type_,"temp",spin_cases({"aaaa"}));

    temp["uvxy"] += F["ex"] * T2["uvey"];
    temp["uvxy"] -= F["vm"] * T2["umxy"];

    temp["UVXY"] += F["EX"] * T2["UVEY"];
    temp["UVXY"] -= F["VM"] * T2["UMXY"];

    temp["uVxY"] += F["ex"] * T2["uVeY"];
    temp["uVxY"] += F["EY"] * T2["uVxE"];
    temp["uVxY"] -= F["VM"] * T2["uMxY"];
    temp["uVxY"] -= F["um"] * T2["mVxY"];

    E += 0.5 * temp["uvxy"] * Lambda2["xyuv"];
    E += 0.5 * temp["UVXY"] * Lambda2["XYUV"];
    E += temp["uVxY"] * Lambda2["xYuV"];

    outfile->Printf("\n  E([F, T2]) %18c = %22.15lf", ' ', E);
    return E;
}

double DSRG_MRPT2::E_VT2_2()
{
    double E = 0.0;

    BlockedTensor temp1;
    BlockedTensor temp2;
    temp1 = BlockedTensor::build(tensor_type_,"temp1",spin_cases({"hhpp"}));
    temp2 = BlockedTensor::build(tensor_type_,"temp2",spin_cases({"hhpp"}));

    temp1["klab"] += T2["ijab"] * Gamma1["ki"] * Gamma1["lj"];
    temp2["klcd"] += temp1["klab"] * Eta1["ac"] * Eta1["bd"];

    temp1["KLAB"] += T2["IJAB"] * Gamma1["KI"] * Gamma1["LJ"];
    temp2["KLCD"] += temp1["KLAB"] * Eta1["AC"] * Eta1["BD"];

    temp1["kLaB"] += T2["iJaB"] * Gamma1["ki"] * Gamma1["LJ"];
    temp2["kLcD"] += temp1["kLaB"] * Eta1["ac"] * Eta1["BD"];

    E += 0.25 * V["cdkl"] * temp2["klcd"];
    E += 0.25 * V["CDKL"] * temp2["KLCD"];
    E += V["cDkL"] * temp2["kLcD"];

    outfile->Printf("\n  E([V, T2] C_2^4) %12c = %22.15lf", ' ', E);
    return E;
}

double DSRG_MRPT2::E_VT2_4HH()
{
    double E = 0.0;
    BlockedTensor temp1;
    BlockedTensor temp2;
    temp1 = BlockedTensor::build(tensor_type_,"temp1", spin_cases({"aahh"}));
    temp2 = BlockedTensor::build(tensor_type_,"temp2", spin_cases({"aaaa"}));

    temp1["uvij"] += V["uvkl"] * Gamma1["ki"] * Gamma1["lj"];
    temp1["UVIJ"] += V["UVKL"] * Gamma1["KI"] * Gamma1["LJ"];
    temp1["uViJ"] += V["uVkL"] * Gamma1["ki"] * Gamma1["LJ"];

    temp2["uvxy"] += temp1["uvij"] * T2["ijxy"];
    temp2["UVXY"] += temp1["UVIJ"] * T2["IJXY"];
    temp2["uVxY"] += temp1["uViJ"] * T2["iJxY"];

    E += 0.125 * Lambda2["xyuv"] * temp2["uvxy"];
    E += 0.125 * Lambda2["XYUV"] * temp2["UVXY"];
    E += Lambda2["xYuV"] * temp2["uVxY"];

    outfile->Printf("\n  E([V, T2] C_2^2 * C_4: HH) %2c = %22.15lf", ' ', E);
    return E;
}

double DSRG_MRPT2::E_VT2_4PP()
{
    double E = 0.0;
    BlockedTensor temp1;
    BlockedTensor temp2;
    temp1 = BlockedTensor::build(tensor_type_,"temp1", spin_cases({"aapp"}));
    temp2 = BlockedTensor::build(tensor_type_,"temp2", spin_cases({"aaaa"}));

    temp1["uvcd"] += T2["uvab"] * Eta1["ac"] * Eta1["bd"];
    temp1["UVCD"] += T2["UVAB"] * Eta1["AC"] * Eta1["BD"];
    temp1["uVcD"] += T2["uVaB"] * Eta1["ac"] * Eta1["BD"];

    temp2["uvxy"] += temp1["uvcd"] * V["cdxy"];
    temp2["UVXY"] += temp1["UVCD"] * V["CDXY"];
    temp2["uVxY"] += temp1["uVcD"] * V["cDxY"];

    E += 0.125 * Lambda2["xyuv"] * temp2["uvxy"];
    E += 0.125 * Lambda2["XYUV"] * temp2["UVXY"];
    E += Lambda2["xYuV"] * temp2["uVxY"];

    outfile->Printf("\n  E([V, T2] C_2^2 * C_4: PP) %2c = %22.15lf", ' ', E);
    return E;
}

double DSRG_MRPT2::E_VT2_4PH()
{
    double E = 0.0;
//    BlockedTensor temp;
//    temp = BlockedTensor::build(tensor_type_,"temp", "aaaa");

//    temp["uvxy"] += V["vbjx"] * T2["iuay"] * Gamma1["ji"] * Eta1["ab"];
//    temp["uvxy"] -= V["vBxJ"] * T2["uIyA"] * Gamma1["JI"] * Eta1["AB"];
//    E += BlockedTensor::dot(temp["uvxy"], Lambda2["xyuv"]);

//    temp["UVXY"] += V["VBJX"] * T2["IUAY"] * Gamma1["JI"] * Eta1["AB"];
//    temp["UVXY"] -= V["bVjX"] * T2["iUaY"] * Gamma1["ji"] * Eta1["ab"];
//    E += BlockedTensor::dot(temp["UVXY"], Lambda2["XYUV"]);

//    temp["uVxY"] -= V["ubjx"] * T2["iVaY"] * Gamma1["ji"] * Eta1["ab"];
//    temp["uVxY"] += V["uBxJ"] * T2["IVAY"] * Gamma1["JI"] * Eta1["AB"];
//    temp["uVxY"] += V["bVjY"] * T2["iuax"] * Gamma1["ji"] * Eta1["ab"];
//    temp["uVxY"] -= V["VBJY"] * T2["uIxA"] * Gamma1["JI"] * Eta1["AB"];
//    temp["uVxY"] -= V["bVxJ"] * T2["uIaY"] * Gamma1["JI"] * Eta1["ab"];
//    temp["uVxY"] -= V["uBjY"] * T2["iVxA"] * Gamma1["ji"] * Eta1["AB"];
//    E += BlockedTensor::dot(temp["uVxY"], Lambda2["xYuV"]);

    BlockedTensor temp1;
    BlockedTensor temp2;
    temp1 = BlockedTensor::build(tensor_type_,"temp1", spin_cases({"hhpp"}));
    temp2 = BlockedTensor::build(tensor_type_,"temp2", spin_cases({"aaaa"}));

    temp1["juby"]  = T2["iuay"] * Gamma1["ji"] * Eta1["ab"];
    temp2["uvxy"] += V["vbjx"] * temp1["juby"];

    temp1["uJyB"]  = T2["uIyA"] * Gamma1["JI"] * Eta1["AB"];
    temp2["uvxy"] -= V["vBxJ"] * temp1["uJyB"];
    E += temp2["uvxy"] * Lambda2["xyuv"];

    temp1["JUBY"]  = T2["IUAY"] * Gamma1["IJ"] * Eta1["AB"];
    temp2["UVXY"] += V["VBJX"] * temp1["JUBY"];

    temp1["jUbY"]  = T2["iUaY"] * Gamma1["ji"] * Eta1["ab"];
    temp2["UVXY"] -= V["bVjX"] * temp1["jUbY"];
    E += temp2["UVXY"] * Lambda2["XYUV"];

    temp1["jVbY"]  = T2["iVaY"] * Gamma1["ji"] * Eta1["ab"];
    temp2["uVxY"] -= V["ubjx"] * temp1["jVbY"];

    temp1["JVBY"]  = T2["IVAY"] * Gamma1["JI"] * Eta1["AB"];
    temp2["uVxY"] += V["uBxJ"] * temp1["JVBY"];

    temp1["jubx"]  = T2["iuax"] * Gamma1["ji"] * Eta1["ab"];
    temp2["uVxY"] += V["bVjY"] * temp1["jubx"];

    temp1["uJxB"]  = T2["uIxA"] * Gamma1["JI"] * Eta1["AB"];
    temp2["uVxY"] -= V["VBJY"] * temp1["uJxB"];

    temp1["uJbY"]  = T2["uIaY"] * Gamma1["JI"] * Eta1["ab"];
    temp2["uVxY"] -= V["bVxJ"] * temp1["uJbY"];

    temp1["jVxB"]  = T2["iVxA"] * Gamma1["ji"] * Eta1["AB"];
    temp2["uVxY"] -= V["uBjY"] * temp1["jVxB"];
    E += temp2["uVxY"] * Lambda2["xYuV"];

    outfile->Printf("\n  E([V, T2] C_2^2 * C_4: PH) %2c = %22.15lf", ' ', E);
    return E;
}

double DSRG_MRPT2::E_VT2_6()
{
    double E = 0.0;
    BlockedTensor temp;
    temp = BlockedTensor::build(tensor_type_,"temp", spin_cases({"aaaaaa"}));

    temp["uvwxyz"] += V["uviz"] * T2["iwxy"];      //  aaaaaa from hole
    temp["uvwxyz"] += V["waxy"] * T2["uvaz"];      //  aaaaaa from particle
    temp["UVWXYZ"] += V["UVIZ"] * T2["IWXY"];      //  AAAAAA from hole
    temp["UVWXYZ"] += V["WAXY"] * T2["UVAZ"];      //  AAAAAA from particle
    E += 0.25 * temp["uvwxyz"] * Lambda3["xyzuvw"];
    E += 0.25 * temp["UVWXYZ"] * Lambda3["XYZUVW"];

    temp["uvWxyZ"] -= V["uviy"] * T2["iWxZ"];      //  aaAaaA from hole
    temp["uvWxyZ"] -= V["uWiZ"] * T2["ivxy"];      //  aaAaaA from hole
    temp["uvWxyZ"] += V["uWyI"] * T2["vIxZ"];      //  aaAaaA from hole
    temp["uvWxyZ"] += V["uWyI"] * T2["vIxZ"];      //  aaAaaA from hole

    temp["uvWxyZ"] += V["aWxZ"] * T2["uvay"];      //  aaAaaA from particle
    temp["uvWxyZ"] -= V["vaxy"] * T2["uWaZ"];      //  aaAaaA from particle
    temp["uvWxyZ"] -= V["vAxZ"] * T2["uWyA"];      //  aaAaaA from particle
    temp["uvWxyZ"] -= V["vAxZ"] * T2["uWyA"];      //  aaAaaA from particle

    E += 0.50 * temp["uvWxyZ"] * Lambda3["xyZuvW"];

    temp["uVWxYZ"] -= V["VWIZ"] * T2["uIxY"];      //  aAAaAA from hole
    temp["uVWxYZ"] -= V["uVxI"] * T2["IWYZ"];      //  aAAaAA from hole
    temp["uVWxYZ"] += V["uViZ"] * T2["iWxY"];      //  aAAaAA from hole
    temp["uVWxYZ"] += V["uViZ"] * T2["iWxY"];      //  aAAaAA from hole

    temp["uVWxYZ"] += V["uAxY"] * T2["VWAZ"];      //  aAAaAA from particle
    temp["uVWxYZ"] -= V["WAYZ"] * T2["uVxA"];      //  aAAaAA from particle
    temp["uVWxYZ"] -= V["aWxY"] * T2["uVaZ"];      //  aAAaAA from particle
    temp["uVWxYZ"] -= V["aWxY"] * T2["uVaZ"];      //  aAAaAA from particle

    E += 0.5 * temp["uVWxYZ"] * Lambda3["xYZuVW"];

    outfile->Printf("\n  E([V, T2] C_2 * C_6) %8c = %22.15lf", ' ', E);
    return E;
}

}} // End Namespaces
