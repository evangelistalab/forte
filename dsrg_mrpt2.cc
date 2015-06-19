#include <numeric>
#include <math.h>

#include <libpsio/psio.hpp>
#include <libpsio/psio.h>
#include <libmints/molecule.h>
#include <libqt/qt.h>

#include "dsrg_mrpt2.h"
#include "blockedtensorfactory.h"

using namespace ambit;

namespace psi{ namespace libadaptive{

DSRG_MRPT2::DSRG_MRPT2(Reference reference, boost::shared_ptr<Wavefunction> wfn, Options &options, ExplorerIntegrals* ints)
    : Wavefunction(options,_default_psio_lib_),
      reference_(reference),
      ints_(ints),
      tensor_type_(kCore),
      BTF(new BlockedTensorFactory(options))
{
    // Copy the wavefunction information
    copy(wfn);

    outfile->Printf("\n\n\t  ---------------------------------------------------------");
    outfile->Printf("\n\t      Driven Similarity Renormalization Group MBPT2");
    outfile->Printf("\n\t  ---------------------------------------------------------");
    if(options.get_bool("MEMORY_SUMMARY"))
    {
    BTF->print_memory_info();
    }

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

    source_ = options_.get_str("SOURCE");

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

    // Populate the core, active, and virtual arrays
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

    BTF->add_mo_space("c","mn",acore_mos,AlphaSpin);
    BTF->add_mo_space("C","MN",bcore_mos,BetaSpin);

    BTF->add_mo_space("a","uvwxyz",aactv_mos,AlphaSpin);
    BTF->add_mo_space("A","UVWXYZ",bactv_mos,BetaSpin);

    BTF->add_mo_space("v","ef",avirt_mos,AlphaSpin);
    BTF->add_mo_space("V","EF",bvirt_mos,BetaSpin);

    BTF->add_composite_mo_space("h","ijkl",{"c","a"});
    BTF->add_composite_mo_space("H","IJKL",{"C","A"});

    BTF->add_composite_mo_space("p","abcd",{"a","v"});
    BTF->add_composite_mo_space("P","ABCD",{"A","V"});

    BTF->add_composite_mo_space("g","pqrs",{"c","a","v"});
    BTF->add_composite_mo_space("G","PQRS",{"C","A","V"});

    H = BTF->build(tensor_type_,"H",spin_cases({"gg"}));
    V = BTF->build(tensor_type_,"V",spin_cases({"gggg"}));

    Gamma1 = BTF->build(tensor_type_,"Gamma1",spin_cases({"hh"}));
    Eta1 = BTF->build(tensor_type_,"Eta1",spin_cases({"pp"}));
    Lambda2 = BTF->build(tensor_type_,"Lambda2",spin_cases({"aaaa"}));
    Lambda3 = BTF->build(tensor_type_,"Lambda3",spin_cases({"aaaaaa"}));
    F = BTF->build(tensor_type_,"Fock",spin_cases({"gg"}));
    Delta1 = BTF->build(tensor_type_,"Delta1",spin_cases({"hp"}));
    Delta2 = BTF->build(tensor_type_,"Delta2",spin_cases({"hhpp"}));
    RDelta1 = BTF->build(tensor_type_,"RDelta1",spin_cases({"hp"}));
    RDelta2 = BTF->build(tensor_type_,"RDelta2",spin_cases({"hhpp"}));
    T1 = BTF->build(tensor_type_,"T1 Amplitudes",spin_cases({"hp"}));
    T2 = BTF->build(tensor_type_,"T2 Amplitudes",spin_cases({"hhpp"}));
    RExp1 = BTF->build(tensor_type_,"RExp1",spin_cases({"hp"}));
    RExp2 = BTF->build(tensor_type_,"RExp2",spin_cases({"hhpp"}));
    Hbar1 = BTF->build(tensor_type_,"One-body Hbar",spin_cases({"hh"}));
    Hbar2 = BTF->build(tensor_type_,"Two-body Hbar",spin_cases({"hhhh"}));

    H.iterate([&](const std::vector<size_t>& i,const std::vector<SpinType>& spin,double& value){
        if (spin[0] == AlphaSpin)
            value = ints_->oei_a(i[0],i[1]);
        else
            value = ints_->oei_b(i[0],i[1]);
    });

    // Fill in the two-electron operator (V)
    V.iterate([&](const std::vector<size_t>& i,const std::vector<SpinType>& spin,double& value){
        if ((spin[0] == AlphaSpin) and (spin[1] == AlphaSpin)) value = ints_->aptei_aa(i[0],i[1],i[2],i[3]);
        if ((spin[0] == AlphaSpin) and (spin[1] == BetaSpin) ) value = ints_->aptei_ab(i[0],i[1],i[2],i[3]);
        if ((spin[0] == BetaSpin)  and (spin[1] == BetaSpin) ) value = ints_->aptei_bb(i[0],i[1],i[2],i[3]);
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

    // Fill out Lambda2 and Lambda3
    ambit::Tensor Lambda2_aa = Lambda2.block("aaaa");
    ambit::Tensor Lambda2_aA = Lambda2.block("aAaA");
    ambit::Tensor Lambda2_AA = Lambda2.block("AAAA");
    Lambda2_aa("pqrs") = reference_.L2aa()("pqrs");
    Lambda2_aA("pqrs") = reference_.L2ab()("pqrs");
    Lambda2_AA("pqrs") = reference_.L2bb()("pqrs");

    ambit::Tensor Lambda3_aaa = Lambda3.block("aaaaaa");
    ambit::Tensor Lambda3_aaA = Lambda3.block("aaAaaA");
    ambit::Tensor Lambda3_aAA = Lambda3.block("aAAaAA");
    ambit::Tensor Lambda3_AAA = Lambda3.block("AAAAAA");
    Lambda3_aaa("pqrstu") = reference_.L3aaa()("pqrstu");
    Lambda3_aaA("pqrstu") = reference_.L3aab()("pqrstu");
    Lambda3_aAA("pqrstu") = reference_.L3abb()("pqrstu");
    Lambda3_AAA("pqrstu") = reference_.L3bbb()("pqrstu");

    // Form the Fock matrix
    F["pq"]  = H["pq"];
    F["pq"] += V["pjqi"] * Gamma1["ij"];
    F["pq"] += V["pJqI"] * Gamma1["IJ"];

    F["PQ"]  = H["PQ"];
    F["PQ"] += V["jPiQ"] * Gamma1["ij"];
    F["PQ"] += V["PJQI"] * Gamma1["IJ"];

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

    // Prepare Hbar
    Hbar1["ij"] = F["ij"];
    Hbar1["IJ"] = F["IJ"];
    Hbar2["ijkl"] = V["ijkl"];
    Hbar2["iJkL"] = V["iJkL"];
    Hbar2["IJKL"] = V["IJKL"];
//    Hbar1.print(stdout);
//    Hbar2.print(stdout);

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

void DSRG_MRPT2::cleanup()
{
}

double DSRG_MRPT2::renormalized_denominator(double D)
{
    double Z = sqrt(s_) * D;
    if(fabs(Z) < pow(0.1, taylor_threshold_)){
        return Taylor_Exp(Z, taylor_order_) * sqrt(s_);
    }else{
        return (1.0 - std::exp(-s_ * pow(D, 2.0))) / D;
    }
}

// Computes (1 - exp(-s D^2/V^2)) * V / D
double DSRG_MRPT2::renormalized_denominator_amp(double V, double D)
{
    if (fabs(V) < 1.0e-10){return 0.0;}

    double RD = D / V;
    double Z = sqrt(s_) * RD;
    if(fabs(Z) < pow(0.1, taylor_threshold_)){
        return Taylor_Exp(Z, taylor_order_) * sqrt(s_);
    }else{
        return (1.0 - std::exp(-s_ * pow(RD, 2.0))) * V / D;
    }
}

// Computes (1 - exp(-s D^2/V^4)) * V / D
double DSRG_MRPT2::renormalized_denominator_emp2(double V, double D)
{
    if (fabs(V) < 1.0e-10){return 0.0;}

    double RD = D / V;
    double Z = sqrt(s_) * (RD / V);
    if(fabs(Z) < pow(0.1, taylor_threshold_)){
        return Taylor_Exp(Z, taylor_order_) * sqrt(s_) / V;
    }else{
        return (1.0 - std::exp(-s_ * pow(RD/V, 2.0))) * V / D;
    }
}

// Computes (1 - exp(-s |D / V|)) * V / D
double DSRG_MRPT2::renormalized_denominator_lamp(double V, double D)
{
    if (fabs(V) < 1.0e-10){return 0.0;}

    double RD = D / V;
    double Z = s_ * RD;
    if(fabs(Z) < pow(0.1, taylor_threshold_)){
        return Taylor_Exp_Linear(Z, taylor_order_) * s_;
    }else{
        return (1.0 - exp(-s_ * fabs(RD))) * V / D;
    }
}

double DSRG_MRPT2::compute_energy()
{
    outfile->Printf("\n\n  ==> Computing DSRG-MRPT2 ... <==\n");
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
    double Etemp  = 0.0;
    double EVT2   = 0.0;
    double Ecorr  = 0.0;
    double Etotal = 0.0;
    std::vector<std::pair<std::string,double>> energy;
    energy.push_back({"E0 (reference)", Eref});

    Etemp  = 2 * E_FT1();
    Ecorr += Etemp;
    energy.push_back({"<[F, T1]>", Etemp});

    Etemp  = 2 * E_FT2();
    Ecorr += Etemp;
    energy.push_back({"<[F, T2]>", Etemp});

    Etemp  = 2 * E_VT1();
    Ecorr += Etemp;
    energy.push_back({"<[V, T1]>", Etemp});

    Etemp  = 2 * E_VT2_2();
    EVT2 += Etemp;
    energy.push_back({"<[V, T2]> (C_2)^4", Etemp});

    Etemp  = 2 * E_VT2_4HH();
    EVT2 += Etemp;
    energy.push_back({"<[V, T2]> C_4 (C_2)^2 HH", Etemp});

    Etemp  = 2 * E_VT2_4PP();
    EVT2 += Etemp;
    energy.push_back({"<[V, T2]> C_4 (C_2)^2 PP", Etemp});

    Etemp  = 2 * E_VT2_4PH();
    EVT2 += Etemp;
    energy.push_back({"<[V, T2]> C_4 (C_2)^2 PH", Etemp});

    Etemp  = 2 * E_VT2_6();
    EVT2 += Etemp;
    energy.push_back({"<[V, T2]> C_6 C_2", Etemp});

    Ecorr += EVT2;
    Etotal = Ecorr + Eref;
    energy.push_back({"<[V, T2]>", EVT2});
    energy.push_back({"DSRG-MRPT2 correlation energy", Ecorr});
    energy.push_back({"DSRG-MRPT2 total energy", Etotal});
    Hbar0 = Etotal;

    // Analyze T1 and T2
    check_t1();
    check_t2();
    energy.push_back({"max(T1)", T1max});
    energy.push_back({"max(T2)", T2max});
    energy.push_back({"||T1||", T1norm});
    energy.push_back({"||T2||", T2norm});

    // Print energy summary
    outfile->Printf("\n\n  ==> DSRG-MRPT2 Energy Summary <==\n");
    for (auto& str_dim : energy){
        outfile->Printf("\n    %-30s = %22.15f",str_dim.first.c_str(),str_dim.second);
    }

    Process::environment.globals["CURRENT ENERGY"] = Etotal;

    return Etotal;
}

double DSRG_MRPT2::compute_ref()
{
    Timer timer;
    std::string str = "Computing reference energy";
    outfile->Printf("\n    %-40s ...", str.c_str());
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

    outfile->Printf("  Done. Timing %15.6f s", timer.get());
    return E + frozen_core_energy + Enuc;
}

void DSRG_MRPT2::compute_t2()
{
    Timer timer;
    std::string str = "Computing T2 amplitudes";
    outfile->Printf("\n    %-40s ...", str.c_str());

    T2["ijab"] = V["ijab"] * RDelta2["ijab"];
    T2["iJaB"] = V["iJaB"] * RDelta2["iJaB"];
    T2["IJAB"] = V["IJAB"] * RDelta2["IJAB"];

    // zero internal amplitudes
    T2.block("aaaa").zero();
    T2.block("aAaA").zero();
    T2.block("AAAA").zero();

    outfile->Printf("  Done. Timing %15.6f s", timer.get());
}

void DSRG_MRPT2::compute_t1()
{
    Timer timer;
    std::string str = "Computing T1 amplitudes";
    outfile->Printf("\n    %-40s ...", str.c_str());

    //A temporary tensor to use for the building of T1
    //Francesco's library does not handle repeating indices between 3 different terms, so need to form an intermediate
    //via a pointwise multiplcation
    BlockedTensor temp = BTF->build(tensor_type_,"temp",spin_cases({"aa"}));
    temp["xu"] = Gamma1["xu"] * Delta1["xu"];
    temp["XU"] = Gamma1["XU"] * Delta1["XU"];

    //Form the T1 amplitudes
    //Note:  The equations are changed slightly from York's equations.
    //Tensor libary does not handle beta alpha beta alpha, only alpha beta alpha beta.
    //Did some permuting to get the correct format

    BlockedTensor N = BTF->build(tensor_type_,"Numerator",spin_cases({"hp"}));

    N["ia"]  = F["ia"];
    N["ia"] += temp["xu"] * T2["iuax"];
    N["ia"] += temp["XU"] * T2["iUaX"];

    N["IA"]  = F["IA"];
    N["IA"] += temp["xu"] * T2["uIxA"];
    N["IA"] += temp["XU"] * T2["IUAX"];

    T1["ia"] = N["ia"] * RDelta1["ia"];
    T1["IA"] = N["IA"] * RDelta1["IA"];

    T1.block("AA").zero();
    T1.block("aa").zero();

    outfile->Printf("  Done. Timing %15.6f s", timer.get());
}

void DSRG_MRPT2::check_t2()
{
    // norm and maximum of T2 amplitudes
    T2norm = 0.0; T2max = 0.0;
    std::vector<std::string> T2blocks = T2.block_labels();
    for(const std::string& block: T2blocks){
        Tensor temp = T2.block(block);
        if(islower(block[0]) && isupper(block[1])){
            T2norm += 4 * pow(temp.norm(), 2.0);
        }else{
            T2norm += pow(temp.norm(), 2.0);
        }
        temp.iterate([&](const std::vector<size_t>& i,double& value){
                T2max = T2max > fabs(value) ? T2max : fabs(value);
        });
    }
    T2norm = sqrt(T2norm);
}

void DSRG_MRPT2::check_t1()
{
    // norm and maximum of T1 amplitudes
    T1norm = T1.norm(); T1max = 0.0;
    T1.iterate([&](const std::vector<size_t>& i,const std::vector<SpinType>& spin,double& value){
            T1max = T1max > fabs(value) ? T1max : fabs(value);
    });
}

void DSRG_MRPT2::renormalize_V()
{
    Timer timer;
    std::string str = "Renormalizing two-electron integrals";
    outfile->Printf("\n    %-40s ...", str.c_str());

    BlockedTensor temp1 = BTF->build(tensor_type_,"temp1",spin_cases({"hhpp"}));
    BlockedTensor temp2 = BTF->build(tensor_type_,"temp2",spin_cases({"hhpp"}));

    temp1["ijab"] = V["ijab"] * RExp2["ijab"];
    temp1["iJaB"] = V["iJaB"] * RExp2["iJaB"];
    temp1["IJAB"] = V["IJAB"] * RExp2["IJAB"];

    // Non-diagonal Hbar2
    Hbar2["ijuv"] = temp1["ijuv"];
    Hbar2["iJuV"] = temp1["iJuV"];
    Hbar2["IJUV"] = temp1["IJUV"];
    Hbar2["uvij"] = temp1["ijuv"];
    Hbar2["uViJ"] = temp1["iJuV"];
    Hbar2["UVIJ"] = temp1["IJUV"];
//    Hbar2.print(stdout);

    // Back to renormalized V
    temp2["ijab"] = temp1["ijab"];
    temp2["iJaB"] = temp1["iJaB"];
    temp2["IJAB"] = temp1["IJAB"];
    temp2["ijab"] += V["ijab"];
    temp2["iJaB"] += V["iJaB"];
    temp2["IJAB"] += V["IJAB"];
//    temp2.print(stdout);

//    temp2["ijab"] = temp1["ijab"] + V["ijab"];
//    temp2["IJAB"] = temp1["IJAB"] + V["IJAB"];
//    temp2["iJaB"] = temp1["iJaB"] + V["iJaB"];
//    temp2.print(stdout);

    V["ijab"] = 0.5 * temp2["ijab"];
    V["iJaB"] = 0.5 * temp2["iJaB"];
    V["IJAB"] = 0.5 * temp2["IJAB"];

    V["abij"] = 0.5 * temp2["ijab"];
    V["aBiJ"] = 0.5 * temp2["iJaB"];
    V["ABIJ"] = 0.5 * temp2["IJAB"];

    outfile->Printf("  Done. Timing %15.6f s", timer.get());
}

void DSRG_MRPT2::renormalize_F()
{
    Timer timer;
    std::string str = "Renormalizing Fock matrix elements";
    outfile->Printf("\n    %-40s ...", str.c_str());

    BlockedTensor temp_aa = BTF->build(tensor_type_,"temp_aa",spin_cases({"aa"}));
    temp_aa["xu"] = Gamma1["xu"] * Delta1["xu"];
    temp_aa["XU"] = Gamma1["XU"] * Delta1["XU"];

    BlockedTensor temp1 = BTF->build(tensor_type_,"temp1",spin_cases({"hp"}));
    BlockedTensor temp2 = BTF->build(tensor_type_,"temp2",spin_cases({"hp"}));

    temp1["ia"] += temp_aa["xu"] * T2["iuax"];
    temp1["ia"] += temp_aa["XU"] * T2["iUaX"];
    temp1["IA"] += temp_aa["xu"] * T2["uIxA"];
    temp1["IA"] += temp_aa["XU"] * T2["IUAX"];

    temp2["ia"] += F["ia"] * RExp1["ia"];
    temp2["ia"] += temp1["ia"] * RExp1["ia"];
    temp2["IA"] += F["IA"] * RExp1["IA"];
    temp2["IA"] += temp1["IA"] * RExp1["IA"];

    // Non-diagonal Hbar1
    Hbar1["iv"] = temp2["iv"];
    Hbar1["IV"] = temp2["IV"];
    Hbar1["vi"] = temp2["iv"];
    Hbar1["VI"] = temp2["IV"];
//    Hbar1.print(stdout);

    // Back to renormalized F
    temp2["ia"] += F["ia"];
    temp2["IA"] += F["IA"];

    F["ia"] = 0.5 * temp2["ia"];
    F["IA"] = 0.5 * temp2["IA"];

    F["ai"] = 0.5 * temp2["ia"];
    F["AI"] = 0.5 * temp2["IA"];

    outfile->Printf("  Done. Timing %15.6f s", timer.get());
}

double DSRG_MRPT2::E_FT1()
{
    Timer timer;
    std::string str = "Computing <[F, T1]>";
    outfile->Printf("\n    %-40s ...", str.c_str());

    double E = 0.0;
    BlockedTensor temp;
    temp = BTF->build(tensor_type_,"temp",spin_cases({"hp"}));

    temp["jb"] += T1["ia"] * Eta1["ab"] * Gamma1["ji"];
    temp["JB"] += T1["IA"] * Eta1["AB"] * Gamma1["JI"];

    E += temp["jb"] * F["bj"];
    E += temp["JB"] * F["BJ"];

    outfile->Printf("  Done. Timing %15.6f s", timer.get());
    return E;
}

double DSRG_MRPT2::E_VT1()
{
    Timer timer;
    std::string str = "Computing <[V, T1]>";
    outfile->Printf("\n    %-40s ...", str.c_str());

    double E = 0.0;
    BlockedTensor temp;
    temp = BTF->build(tensor_type_,"temp", spin_cases({"aaaa"}));

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

    outfile->Printf("  Done. Timing %15.6f s", timer.get());
    return E;
}

double DSRG_MRPT2::E_FT2()
{
    Timer timer;
    std::string str = "Computing <[F, T2]>";
    outfile->Printf("\n    %-40s ...", str.c_str());

    double E = 0.0;
    BlockedTensor temp;
    temp = BTF->build(tensor_type_,"temp",spin_cases({"aaaa"}));

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

    outfile->Printf("  Done. Timing %15.6f s", timer.get());
    return E;
}

double DSRG_MRPT2::E_VT2_2()
{
    Timer timer;
    std::string str = "Computing <[V, T2]> (C_2)^4";
    outfile->Printf("\n    %-40s ...", str.c_str());

    double E = 0.0;
    BlockedTensor temp1;
    BlockedTensor temp2;
    temp1 = BTF->build(tensor_type_,"temp1",spin_cases({"hhpp"}));
    temp2 = BTF->build(tensor_type_,"temp2",spin_cases({"hhpp"}));

    temp1["klab"] += T2["ijab"] * Gamma1["ki"] * Gamma1["lj"];
    temp2["klcd"] += temp1["klab"] * Eta1["ac"] * Eta1["bd"];

    temp1["KLAB"] += T2["IJAB"] * Gamma1["KI"] * Gamma1["LJ"];
    temp2["KLCD"] += temp1["KLAB"] * Eta1["AC"] * Eta1["BD"];

    temp1["kLaB"] += T2["iJaB"] * Gamma1["ki"] * Gamma1["LJ"];
    temp2["kLcD"] += temp1["kLaB"] * Eta1["ac"] * Eta1["BD"];

    E += 0.25 * V["cdkl"] * temp2["klcd"];
    E += 0.25 * V["CDKL"] * temp2["KLCD"];
    E += V["cDkL"] * temp2["kLcD"];

    outfile->Printf("  Done. Timing %15.6f s", timer.get());
    return E;
}

double DSRG_MRPT2::E_VT2_4HH()
{
    Timer timer;
    std::string str = "Computing <[V, T2]> C_4 (C_2)^2 HH";
    outfile->Printf("\n    %-40s ...", str.c_str());

    double E = 0.0;
    BlockedTensor temp1;
    BlockedTensor temp2;
    temp1 = BTF->build(tensor_type_,"temp1", spin_cases({"aahh"}));
    temp2 = BTF->build(tensor_type_,"temp2", spin_cases({"aaaa"}));

    temp1["uvij"] += V["uvkl"] * Gamma1["ki"] * Gamma1["lj"];
    temp1["UVIJ"] += V["UVKL"] * Gamma1["KI"] * Gamma1["LJ"];
    temp1["uViJ"] += V["uVkL"] * Gamma1["ki"] * Gamma1["LJ"];

    temp2["uvxy"] += temp1["uvij"] * T2["ijxy"];
    temp2["UVXY"] += temp1["UVIJ"] * T2["IJXY"];
    temp2["uVxY"] += temp1["uViJ"] * T2["iJxY"];

    E += 0.125 * Lambda2["xyuv"] * temp2["uvxy"];
    E += 0.125 * Lambda2["XYUV"] * temp2["UVXY"];
    E += Lambda2["xYuV"] * temp2["uVxY"];

    outfile->Printf("  Done. Timing %15.6f s", timer.get());
    return E;
}

double DSRG_MRPT2::E_VT2_4PP()
{
    Timer timer;
    std::string str = "Computing <[V, T2]> C_4 (C_2)^2 PP";
    outfile->Printf("\n    %-40s ...", str.c_str());

    double E = 0.0;
    BlockedTensor temp1;
    BlockedTensor temp2;
    temp1 = BTF->build(tensor_type_,"temp1", spin_cases({"aapp"}));
    temp2 = BTF->build(tensor_type_,"temp2", spin_cases({"aaaa"}));

    temp1["uvcd"] += T2["uvab"] * Eta1["ac"] * Eta1["bd"];
    temp1["UVCD"] += T2["UVAB"] * Eta1["AC"] * Eta1["BD"];
    temp1["uVcD"] += T2["uVaB"] * Eta1["ac"] * Eta1["BD"];

    temp2["uvxy"] += temp1["uvcd"] * V["cdxy"];
    temp2["UVXY"] += temp1["UVCD"] * V["CDXY"];
    temp2["uVxY"] += temp1["uVcD"] * V["cDxY"];

    E += 0.125 * Lambda2["xyuv"] * temp2["uvxy"];
    E += 0.125 * Lambda2["XYUV"] * temp2["UVXY"];
    E += Lambda2["xYuV"] * temp2["uVxY"];

    outfile->Printf("  Done. Timing %15.6f s", timer.get());
    return E;
}

double DSRG_MRPT2::E_VT2_4PH()
{
    Timer timer;
    std::string str = "Computing <[V, T2]> C_4 (C_2)^2 PH";
    outfile->Printf("\n    %-40s ...", str.c_str());

    double E = 0.0;

    BlockedTensor temp1;
    BlockedTensor temp2;
    temp1 = BTF->build(tensor_type_,"temp1", spin_cases({"hhpp"}));
    temp2 = BTF->build(tensor_type_,"temp2", spin_cases({"aaaa"}));

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

    outfile->Printf("  Done. Timing %15.6f s", timer.get());
    return E;
}

double DSRG_MRPT2::E_VT2_6()
{
    Timer timer;
    std::string str = "Computing <[V, T2]> C_6 C_2";
    outfile->Printf("\n    %-40s ...", str.c_str());

    double E = 0.0;
    BlockedTensor temp;
    temp = BTF->build(tensor_type_,"temp", spin_cases({"aaaaaa"}));

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

    outfile->Printf("  Done. Timing %15.6f s", timer.get());
    return E;
}

void DSRG_MRPT2::transform_integrals(){

    outfile->Printf("\n\n  ==> Building effective Hamiltonian Hbar ... <==\n");

    // Compute second-order one- and two-body Hbar
    // Cautious: V and F are renormalized !
    Hbar1_FT1();
    Hbar1_FT2();
    Hbar2_FT2();
    Hbar1_VT1();
    Hbar2_VT1();
    Hbar2_VT2_HH();
    Hbar2_VT2_PP();
    Hbar2_VT2_PH();
    Hbar1_VT2_2();
//    Hbar1_VT2_4_22();
//    Hbar1_VT2_4_13();
//    Hbar1.print(stdout);
//    Hbar2.print(stdout);

    // Scalar term
    outfile->Printf("\n\n  ==> Computing scalar term in Hbar ... <==\n");
    boost::shared_ptr<Molecule> molecule = Process::environment.molecule();
    double Enuc = molecule->nuclear_repulsion_energy();
    double scalar = Hbar0 - Enuc;

    scalar -= Hbar1["ji"] * Gamma1["ij"];
    scalar -= Hbar1["JI"] * Gamma1["IJ"];

    scalar -= 0.25 * Hbar2["xyuv"] * Lambda2["uvxy"];
    scalar -= 0.25 * Hbar2["XYUV"] * Lambda2["UVXY"];
    scalar -= Hbar2["xYuV"] * Lambda2["uVxY"];

    scalar += 0.5 * Hbar2["klij"] * Gamma1["ik"] * Gamma1["jl"];
    scalar += 0.5 * Hbar2["KLIJ"] * Gamma1["IK"] * Gamma1["JL"];
    scalar += Hbar2["kLiJ"] * Gamma1["ik"] * Gamma1["JL"];

    outfile->Printf("\n    %-30s = %22.15f", "Hbar scalar", scalar);

    // Update integrals
    ints_->set_scalar(scalar);
    outfile->Printf("\n\n  ==> Updating all integrals ... <==\n");

    // Compute one-body operator (two-body operator is Hbar2)
    Timer o1;
    std::string str = "Computing new one-electron integrals";
    outfile->Printf("\n    %-40s ...", str.c_str());
    BlockedTensor O1 = BTF->build(tensor_type_, "One-body operator", spin_cases({"hh"}));
    O1["ij"]  = Hbar1["ij"];
    O1["ij"] -= Hbar2["ikjl"] * Gamma1["lk"];
    O1["ij"] -= Hbar2["iKjL"] * Gamma1["LK"];
    O1["IJ"]  = Hbar1["IJ"];
    O1["IJ"] -= Hbar2["kIlJ"] * Gamma1["lk"];
    O1["IJ"] -= Hbar2["IKJL"] * Gamma1["LK"];
    outfile->Printf("  Done. Timing %15.6f s", o1.get());

    // Create a map between space label and space mos
    std::map<char, std::vector<size_t>> label_to_spacemo;
    label_to_spacemo['c'] = acore_mos;
    label_to_spacemo['C'] = bcore_mos;
    label_to_spacemo['a'] = aactv_mos;
    label_to_spacemo['A'] = bactv_mos;

    // Fill out ints->oei
    Timer fill1;
    str = "Updating one-electron integrals";
    outfile->Printf("\n    %-40s ...", str.c_str());
    for(std::string block: O1.block_labels()){
        bool spin = islower(block[0]) ? true : false;
        O1.block(block).citerate([&](const std::vector<size_t>& i,const double& value){
            size_t idx0 = label_to_spacemo[block[0]][i[0]];
            size_t idx1 = label_to_spacemo[block[1]][i[1]];
            ints_->set_oei(idx0,idx1,value,spin);
        });
    }
    outfile->Printf("  Done. Timing %15.6f s", fill1.get());

    // Fill out ints->tei
    Timer fill2;
    str = "Updating two-electron integrals";
    outfile->Printf("\n    %-40s ...", str.c_str());
    for(std::string block: Hbar2.block_labels()){
        bool spin0 = islower(block[0]) ? true : false;
        bool spin1 = islower(block[1]) ? true : false;
        Hbar2.block(block).citerate([&](const std::vector<size_t>& i,const double& value){
            size_t idx0 = label_to_spacemo[block[0]][i[0]];
            size_t idx1 = label_to_spacemo[block[1]][i[1]];
            size_t idx2 = label_to_spacemo[block[2]][i[2]];
            size_t idx3 = label_to_spacemo[block[3]][i[3]];
            ints_->set_tei(idx0,idx1,idx2,idx3,value,spin0,spin1);
        });
    }
    outfile->Printf("  Done. Timing %15.6f s", fill2.get());

    ints_->update_integrals(false);
}

void DSRG_MRPT2::Hbar1_FT1(){
    Timer timer;
    std::string str = "Computing [F, T1] for Hbar1";
    outfile->Printf("\n    %-40s ...", str.c_str());

//    Hbar1.print(stdout);
    BlockedTensor temp = BTF->build(tensor_type_, "temp", spin_cases({"hh"}));
    temp["ij"] = F["aj"] * T1["ia"];
    temp["IJ"] = F["AJ"] * T1["IA"];

    Hbar1["ij"] += temp["ij"];
    Hbar1["ij"] += temp["ji"];
    Hbar1["IJ"] += temp["IJ"];
    Hbar1["IJ"] += temp["JI"];

    temp = BTF->build(tensor_type_, "temp", spin_cases({"ha"}));
    temp["iu"] = F["im"] * T1["mu"];
    temp["IU"] = F["IM"] * T1["MU"];

    Hbar1["iu"] -= temp["iu"];
    Hbar1["ui"] -= temp["iu"];
    Hbar1["IU"] -= temp["IU"];
    Hbar1["UI"] -= temp["IU"];

//    Hbar1.print(stdout);
    outfile->Printf("  Done. Timing %15.6f s", timer.get());
}

void DSRG_MRPT2::Hbar1_FT2(){
    Timer timer;
    std::string str = "Computing [F, T2] for Hbar1";
    outfile->Printf("\n    %-40s ...", str.c_str());

//    Hbar1.print(stdout);
    BlockedTensor temp = BTF->build(tensor_type_, "temp", spin_cases({"ha"}));
    temp["iu"] += T2["ijub"] * F["bk"] * Gamma1["kj"];
    temp["iu"] += T2["iJuB"] * F["BK"] * Gamma1["KJ"];
    temp["iu"] -= T2["ijuv"] * F["xj"] * Gamma1["vx"];
    temp["iu"] -= T2["iJuV"] * F["XJ"] * Gamma1["VX"];

    temp["IU"] += T2["jIbU"] * F["bk"] * Gamma1["kj"];
    temp["IU"] += T2["IJUB"] * F["BK"] * Gamma1["KJ"];
    temp["IU"] -= T2["jIvU"] * F["xj"] * Gamma1["vx"];
    temp["IU"] -= T2["IJUV"] * F["XJ"] * Gamma1["VX"];

    Hbar1["iu"] += temp["iu"];
    Hbar1["ui"] += temp["iu"];
    Hbar1["IU"] += temp["IU"];
    Hbar1["UI"] += temp["IU"];

//    Hbar1.print(stdout);
    outfile->Printf("  Done. Timing %15.6f s", timer.get());
}

void DSRG_MRPT2::Hbar2_FT2(){
    Timer timer;
    std::string str = "Computing [F, T2] for Hbar2";
    outfile->Printf("\n    %-40s ...", str.c_str());

//    Hbar2.print(stdout);
    BlockedTensor temp = BTF->build(tensor_type_, "temp", spin_cases({"hhhh"}));
    temp["ijku"] += T2["ijau"] * F["ak"];
    temp["iJkU"] += T2["iJaU"] * F["ak"];
    temp["IJKU"] += T2["IJAU"] * F["AK"];
    Hbar2["ijku"] += temp["ijku"];
    Hbar2["iJkU"] += temp["iJkU"];
    Hbar2["IJKU"] += temp["IJKU"];
    Hbar2["kuij"] += temp["ijku"];
    Hbar2["kUiJ"] += temp["iJkU"];
    Hbar2["KUIJ"] += temp["IJKU"];

    // No need to recompute temp["ijuk"] because of the permutation symmetry
    temp["iJuK"]   = T2["iJuA"] * F["AK"];
    Hbar2["ijuk"] -= temp["ijku"];
    Hbar2["iJuK"] += temp["iJuK"];
    Hbar2["IJUK"] -= temp["IJKU"];
    Hbar2["ukij"] -= temp["ijku"];
    Hbar2["uKiJ"] += temp["iJuK"];
    Hbar2["UKIJ"] -= temp["IJKU"];

    temp["ijuv"]  = T2["kjuv"] * F["ik"];
    Hbar2["ijuv"] -= temp["ijuv"];
    Hbar2["ijuv"] += temp["jiuv"];
    Hbar2["uvij"] -= temp["ijuv"];
    Hbar2["uvij"] += temp["jiuv"];

    temp["iJuV"]  = T2["kJuV"] * F["ik"];
    temp["iJuV"] += T2["iKuV"] * F["JK"];
    Hbar2["iJuV"] -= temp["iJuV"];
    Hbar2["uViJ"] -= temp["iJuV"];

    temp["IJUV"]  = T2["KJUV"] * F["IK"];
    Hbar2["IJUV"] -= temp["IJUV"];
    Hbar2["IJUV"] += temp["JIUV"];
    Hbar2["UVIJ"] -= temp["IJUV"];
    Hbar2["UVIJ"] += temp["JIUV"];

//    Hbar2.print(stdout);
    outfile->Printf("  Done. Timing %15.6f s", timer.get());
}

void DSRG_MRPT2::Hbar1_VT1(){
    Timer timer;
    std::string str = "Computing [V, T1] for Hbar1";
    outfile->Printf("\n    %-40s ...", str.c_str());

//    Hbar1.print(stdout);
    BlockedTensor temp = BTF->build(tensor_type_, "temp", spin_cases({"hhhh"}));
    temp["likj"] = V["lakj"] * T1["ia"];
    Hbar1["lk"] += temp["likj"] * Gamma1["ji"];
    Hbar1["lk"] += temp["kilj"] * Gamma1["ji"];

    temp["lIkJ"] = V["lAkJ"] * T1["IA"];
    Hbar1["lk"] += temp["lIkJ"] * Gamma1["JI"];
    Hbar1["lk"] += temp["kIlJ"] * Gamma1["JI"];

    temp["lukv"] = V["lukm"] * T1["mv"];
    Hbar1["lk"] -= temp["lukv"] * Gamma1["vu"];
    Hbar1["lk"] -= temp["kulv"] * Gamma1["vu"];

    temp["lUkV"] = V["lUkM"] * T1["MV"];
    Hbar1["lk"] -= temp["lUkV"] * Gamma1["VU"];
    Hbar1["lk"] -= temp["kUlV"] * Gamma1["VU"];

    temp["iLjK"] = V["aLjK"] * T1["ia"];
    Hbar1["LK"] += temp["iLjK"] * Gamma1["ji"];
    Hbar1["LK"] += temp["iKjL"] * Gamma1["ji"];

    temp["LIKJ"] = V["LAKJ"] * T1["IA"];
    Hbar1["LK"] += temp["LIKJ"] * Gamma1["JI"];
    Hbar1["LK"] += temp["KILJ"] * Gamma1["JI"];

    temp["uLvK"] = V["uLmK"] * T1["mv"];
    Hbar1["LK"] -= temp["uLvK"] * Gamma1["vu"];
    Hbar1["LK"] -= temp["uKvL"] * Gamma1["uv"];

    temp["LUKV"] = V["LUKM"] * T1["MV"];
    Hbar1["LK"] -= temp["LUKV"] * Gamma1["VU"];
    Hbar1["LK"] -= temp["KULV"] * Gamma1["UV"];

//    Hbar1.print(stdout);
    outfile->Printf("  Done. Timing %15.6f s", timer.get());
}

void DSRG_MRPT2::Hbar2_VT1(){
    Timer timer;
    std::string str = "Computing [V, T1] for Hbar2";
    outfile->Printf("\n    %-40s ...", str.c_str());

//    Hbar2.print(stdout);
    BlockedTensor temp = BTF->build(tensor_type_, "temp", spin_cases({"hhhh"}));
    temp["ijkl"] = T1["ia"] * V["ajkl"];
    Hbar2["ijkl"] += temp["ijkl"];
    Hbar2["ijkl"] += temp["klij"];
    Hbar2["ijkl"] -= temp["jikl"];
    Hbar2["ijkl"] -= temp["lkij"];

    temp["iJkL"] = T1["ia"] * V["aJkL"];
    Hbar2["iJkL"] += temp["iJkL"];
    Hbar2["iJkL"] += temp["kLiJ"];
    temp["iJkL"] = T1["JA"] * V["iAkL"];
    Hbar2["iJkL"] += temp["iJkL"];
    Hbar2["iJkL"] += temp["kLiJ"];

    temp["IJKL"] = T1["IA"] * V["AJKL"];
    Hbar2["IJKL"] += temp["IJKL"];
    Hbar2["IJKL"] += temp["KLIJ"];
    Hbar2["IJKL"] -= temp["JIKL"];
    Hbar2["IJKL"] -= temp["LKIJ"];

    temp["ijku"] = T1["mu"] * V["ijkm"];
    temp["iJkU"] = T1["MU"] * V["iJkM"];
    temp["IJKU"] = T1["MU"] * V["IJKM"];
    Hbar2["ijku"] -= temp["ijku"];
    Hbar2["iJkU"] -= temp["iJkU"];
    Hbar2["IJKU"] -= temp["IJKU"];
    Hbar2["kuij"] -= temp["ijku"];
    Hbar2["kUiJ"] -= temp["iJkU"];
    Hbar2["KUIJ"] -= temp["IJKU"];

    temp["iJuK"] = T1["mu"] * V["iJmK"];
    Hbar2["ijuk"] += temp["ijku"];
    Hbar2["iJuK"] -= temp["iJuK"];
    Hbar2["IJUK"] += temp["IJKU"];
    Hbar2["ukij"] += temp["ijku"];
    Hbar2["uKiJ"] -= temp["iJuK"];
    Hbar2["UKIJ"] += temp["IJKU"];

//    Hbar2.print(stdout);
    outfile->Printf("  Done. Timing %15.6f s", timer.get());
}

void DSRG_MRPT2::Hbar2_VT2_PP(){
    Timer timer;
    std::string str = "Computing [V, T2] PP for Hbar2";
    outfile->Printf("\n    %-40s ...", str.c_str());

//    Hbar2.print(stdout);
    BlockedTensor temp = BTF->build(tensor_type_, "temp", spin_cases({"hhpp"}));
    temp["ijcd"]  = T2["ijab"] * Eta1["ac"] * Eta1["bd"];
    Hbar2["ijkl"] += 0.5 * temp["ijcd"] * V["cdkl"];
    Hbar2["ijkl"] += 0.5 * temp["klcd"] * V["cdij"];

    temp["ijvy"] = T2["ijux"] * Gamma1["uv"] * Gamma1["xy"];
    Hbar2["ijkl"] -= 0.5 * temp["ijvy"] * V["vykl"];
    Hbar2["ijkl"] -= 0.5 * temp["klvy"] * V["vyij"];

    temp["iJcD"] = T2["iJaB"] * Eta1["ac"] * Eta1["BD"];
    Hbar2["iJkL"] += temp["iJcD"] * V["cDkL"];
    Hbar2["iJkL"] += temp["kLcD"] * V["cDiJ"];

    temp["iJvY"] = T2["iJuX"] * Gamma1["uv"] * Gamma1["XY"];
    Hbar2["iJkL"] -= temp["iJvY"] * V["vYkL"];
    Hbar2["iJkL"] -= temp["kLvY"] * V["vYiJ"];

    temp["IJCD"] = T2["IJAB"] * Eta1["AC"] * Eta1["BD"];
    Hbar2["IJKL"] += 0.5 * temp["IJCD"] * V["CDKL"];
    Hbar2["IJKL"] += 0.5 * temp["KLCD"] * V["CDIJ"];

    temp["IJVY"] = T2["IJUX"] * Gamma1["UV"] * Gamma1["XY"];
    Hbar2["IJKL"] -= 0.5 * temp["IJVY"] * V["VYKL"];
    Hbar2["IJKL"] -= 0.5 * temp["KLVY"] * V["VYIJ"];

//    Hbar2.print(stdout);
    outfile->Printf("  Done. Timing %15.6f s", timer.get());
}

void DSRG_MRPT2::Hbar2_VT2_HH(){
    Timer timer;
    std::string str = "Computing [V, T2] HH for Hbar2";
    outfile->Printf("\n    %-40s ...", str.c_str());

//    Hbar2.print(stdout);
    BlockedTensor temp = BTF->build(tensor_type_, "temp", spin_cases({"hhaa"}));
    BlockedTensor final = BTF->build(tensor_type_, "final_temp", spin_cases({"hhaa"}));
    temp["ijuv"] = T2["kluv"] * Gamma1["ik"] * Gamma1["jl"];
    temp["iJuV"] = T2["kLuV"] * Gamma1["ik"] * Gamma1["JL"];
    temp["IJUV"] = T2["KLUV"] * Gamma1["IK"] * Gamma1["JL"];

    final["ijuv"] = V["ijkl"] * temp["kluv"];
    final["iJuV"] = V["iJkL"] * temp["kLuV"];
    final["IJUV"] = V["IJKL"] * temp["KLUV"];

    Hbar2["ijuv"] += 0.5 * final["ijuv"];
    Hbar2["uvij"] += 0.5 * final["ijuv"];

    Hbar2["iJuV"] += final["iJuV"];
    Hbar2["uViJ"] += final["iJuV"];

    Hbar2["IJUV"] += 0.5 * final["IJUV"];
    Hbar2["UVIJ"] += 0.5 * final["IJUV"];

//    Hbar2.print(stdout);
    outfile->Printf("  Done. Timing %15.6f s", timer.get());
}

void DSRG_MRPT2::Hbar2_VT2_PH(){
    Timer timer;
    std::string str = "Computing [V, T2] PH for Hbar2";
    outfile->Printf("\n    %-40s ...", str.c_str());

//    Hbar2.print(stdout);
    BlockedTensor temp3 = BTF->build(tensor_type_, "temp3", spin_cases({"hhhh"}));
    BlockedTensor temp2 = BTF->build(tensor_type_, "temp2", spin_cases({"hhhh"}));
    BlockedTensor temp1 = BTF->build(tensor_type_, "temp1", spin_cases({"hhpp"}));

    // alpha-alpha
    temp1["ljau"]  = T2["ijau"] * Gamma1["li"];
    temp2["ijku"]  = V["ailk"] * temp1["ljau"];
    temp3["ijku"]  = temp2["ijku"];
    temp3["ijku"] -= temp2["jiku"];

    temp1["jLuA"]  = T2["jIuA"] * Gamma1["LI"];
    temp2["ijku"]  = V["iAkL"] * temp1["jLuA"];
    temp3["ijku"] += temp2["ijku"];
    temp3["ijku"] -= temp2["jiku"];

    Hbar2["ijku"] += temp3["ijku"];
    Hbar2["kuij"] += temp3["ijku"];
    Hbar2["ijuk"] -= temp3["ijku"];
    Hbar2["ukij"] -= temp3["ijku"];

    temp1["ljyu"]  = T2["ljxu"] * Gamma1["xy"];
    temp2["ijku"]  = V["yilk"] * temp1["ljyu"];
    temp3["ijku"]  = temp2["ijku"];
    temp3["ijku"] -= temp2["jiku"];

    temp1["jLuY"]  = T2["jLuX"] * Gamma1["XY"];
    temp2["ijku"]  = V["iYkL"] * temp1["jLuY"];
    temp3["ijku"] += temp2["ijku"];
    temp3["ijku"] -= temp2["jiku"];

    Hbar2["ijku"] -= temp3["ijku"];
    Hbar2["kuij"] -= temp3["ijku"];
    Hbar2["ijuk"] += temp3["ijku"];
    Hbar2["ukij"] += temp3["ijku"];

    // beta-beta
    temp1["lJaU"]  = T2["iJaU"] * Gamma1["li"];
    temp2["IJKU"]  = V["aIlK"] * temp1["lJaU"];
    temp3["IJKU"]  = temp2["IJKU"];
    temp3["IJKU"] -= temp2["JIKU"];

    temp1["LJAU"]  = T2["IJAU"] * Gamma1["LI"];
    temp2["IJKU"]  = V["AILK"] * temp1["LJAU"];
    temp3["IJKU"] += temp2["IJKU"];
    temp3["IJKU"] -= temp2["JIKU"];

    Hbar2["IJKU"] += temp3["IJKU"];
    Hbar2["KUIJ"] += temp3["IJKU"];
    Hbar2["IJUK"] -= temp3["IJKU"];
    Hbar2["UKIJ"] -= temp3["IJKU"];

    temp1["lJyU"]  = T2["lJxU"] * Gamma1["xy"];
    temp2["IJKU"]  = V["yIlK"] * temp1["lJyU"];
    temp3["IJKU"]  = temp2["IJKU"];
    temp3["IJKU"] -= temp2["JIKU"];

    temp1["LJYU"]  = T2["LJXU"] * Gamma1["XY"];
    temp2["IJKU"]  = V["YILK"] * temp1["LJYU"];
    temp3["IJKU"] += temp2["IJKU"];
    temp3["IJKU"] -= temp2["JIKU"];

    Hbar2["IJKU"] -= temp3["IJKU"];
    Hbar2["KUIJ"] -= temp3["IJKU"];
    Hbar2["IJUK"] += temp3["IJKU"];
    Hbar2["UKIJ"] += temp3["IJKU"];

    // alpha-beta ["iJkU"]
    temp1["lJaU"]  = T2["iJaU"] * Gamma1["li"];
    temp2["iJkU"]  = V["ailk"] * temp1["lJaU"];
    temp3["iJkU"]  = temp2["iJkU"];

    temp1["LJAU"]  = T2["IJAU"] * Gamma1["LI"];
    temp2["iJkU"]  = V["iAkL"] * temp1["LJAU"];
    temp3["iJkU"] += temp2["iJkU"];

    temp1["iLaU"]  = T2["iJaU"] * Gamma1["LJ"];
    temp2["iJkU"]  = -1.0 * V["aJkL"] * temp1["iLaU"];
    temp3["iJkU"] += temp2["iJkU"];

    temp1["lJyU"]  = T2["lJxU"] * Gamma1["xy"];
    temp2["iJkU"]  = V["yilk"] * temp1["lJyU"];
    temp3["iJkU"] -= temp2["iJkU"];

    temp1["LJYU"]  = T2["LJXU"] * Gamma1["XY"];
    temp2["iJkU"]  = V["iYkL"] * temp1["LJYU"];
    temp3["iJkU"] -= temp2["iJkU"];

    temp1["iLyU"]  = T2["iLxU"] * Gamma1["xy"];
    temp2["iJkU"]  = -1.0 * V["yJkL"] * temp1["iLyU"];
    temp3["iJkU"] -= temp2["iJkU"];

    Hbar2["iJkU"] += temp3["iJkU"];
    Hbar2["kUiJ"] += temp3["iJkU"];

    // alpha-beta ["iJuK"]
    temp1["ilub"]  = T2["ijub"] * Gamma1["lj"];
    temp2["iJuK"]  = V["bJlK"] * temp1["ilub"];
    temp3["iJuK"]  = temp2["iJuK"];

    temp1["iLuB"]  = T2["iJuB"] * Gamma1["LJ"];
    temp2["iJuK"]  = V["BJLK"] * temp1["iLuB"];
    temp3["iJuK"] += temp2["iJuK"];

    temp1["lJuB"]  = T2["iJuB"] * Gamma1["li"];
    temp2["iJuK"]  = -1.0 * V["iBlK"] * temp1["lJuB"];
    temp3["iJuK"] += temp2["iJuK"];

    temp1["iluy"]  = T2["ilux"] * Gamma1["xy"];
    temp2["iJuK"]  = V["yJlK"] * temp1["iluy"];
    temp3["iJuK"] -= temp2["iJuK"];

    temp1["iLuY"]  = T2["iLuX"] * Gamma1["XY"];
    temp2["iJuK"]  = V["YJLK"] * temp1["iLuY"];
    temp3["iJuK"] -= temp2["iJuK"];

    temp1["lJuY"]  = T2["lJuX"] * Gamma1["XY"];
    temp2["iJuK"]  = -1.0 * V["iYlK"] * temp1["lJuY"];
    temp3["iJuK"] -= temp2["iJuK"];

    Hbar2["iJuK"] += temp3["iJuK"];
    Hbar2["uKiJ"] += temp3["iJuK"];

//    Hbar2.print(stdout);
    outfile->Printf("  Done. Timing %15.6f s", timer.get());
}

void DSRG_MRPT2::Hbar1_VT2_2(){
    Timer timer;
    std::string str = "Computing [V, T2] (C_2)^3 for Hbar1";
    outfile->Printf("\n    %-40s ...", str.c_str());

    BlockedTensor temp1 = BTF->build(tensor_type_, "temp1", spin_cases({"hhpp"}));
    BlockedTensor temp2 = BTF->build(tensor_type_, "temp2", spin_cases({"hhhh"}));
    BlockedTensor temp3 = BTF->build(tensor_type_, "temp3", spin_cases({"hh"}));

    // hole-hole Gamma * Eta * Eta
    temp1["ikcd"] = T2["ikab"] * Eta1["bd"] * Eta1["ac"];
    temp1["iKcD"] = T2["iKaB"] * Eta1["BD"] * Eta1["ac"];
    temp1["IKCD"] = T2["IKAB"] * Eta1["BD"] * Eta1["AC"];

    temp2["ikjl"] = V["cdjl"] * temp1["ikcd"];
    temp3["ij"] = temp2["ikjl"] * Gamma1["lk"];
    Hbar1["ij"] += 0.5 * temp3["ij"];
    Hbar1["ij"] += 0.5 * temp3["ji"];

    temp2["iKjL"] = V["cDjL"] * temp1["iKcD"];
    temp3["ij"] = temp2["iKjL"] * Gamma1["LK"];
    Hbar1["ij"] += temp3["ij"];
    Hbar1["ij"] += temp3["ji"];

    temp3["IJ"] = temp2["kIlJ"] * Gamma1["lk"];
    Hbar1["IJ"] += temp3["IJ"];
    Hbar1["IJ"] += temp3["JI"];

    temp2["IKJL"] = V["CDJL"] * temp1["IKCD"];
    temp3["IJ"] = temp2["IKJL"] * Gamma1["LK"];
    Hbar1["IJ"] += 0.5 * temp3["IJ"];
    Hbar1["IJ"] += 0.5 * temp3["JI"];

    // hole-hole Gamma * Gamma * Eta
    temp1["izxy"] = T2["izuv"] * Gamma1["ux"] * Gamma1["vy"];
    temp2["izjw"] = V["xyjw"] * temp1["izxy"];
    temp3["ij"] = temp2["izjw"] * Eta1["wz"];
    Hbar1["ij"] += 0.5 * temp3["ij"];
    Hbar1["ij"] += 0.5 * temp3["ji"];

    temp1["iZxY"] = T2["iZuV"] * Gamma1["ux"] * Gamma1["VY"];
    temp2["iZjW"] = V["xYjW"] * temp1["iZxY"];
    temp3["ij"] = temp2["iZjW"] * Eta1["WZ"];
    Hbar1["ij"] += temp3["ij"];
    Hbar1["ij"] += temp3["ji"];

    temp1["zIyX"] = T2["zIvU"] * Gamma1["UX"] * Gamma1["vy"];
    temp2["zIwJ"] = V["yXwJ"] * temp1["zIyX"];
    temp3["IJ"] = temp2["zIwJ"] * Eta1["wz"];
    Hbar1["IJ"] += temp3["IJ"];
    Hbar1["IJ"] += temp3["JI"];

    temp1["IZXY"] = T2["IZUV"] * Gamma1["UX"] * Gamma1["VY"];
    temp2["IZJW"] = V["XYJW"] * temp1["IZXY"];
    temp3["IJ"] = temp2["IZJW"] * Eta1["WZ"];
    Hbar1["IJ"] += 0.5 * temp3["IJ"];
    Hbar1["IJ"] += 0.5 * temp3["JI"];

    // hole-active
    temp1["liub"] = T2["kjub"] * Gamma1["lk"] * Gamma1["ij"];
    temp3["iu"] = 0.5 * temp1["ljub"] * Eta1["ba"] * V["ialj"];
    temp1["lIuB"] = T2["kJuB"] * Gamma1["lk"] * Gamma1["IJ"];
    temp3["iu"] += temp1["lJuB"] * Eta1["BA"] * V["iAlJ"];
    Hbar1["iu"] -= temp3["iu"];
    Hbar1["ui"] -= temp3["iu"];

    temp1["iLbU"] = T2["jKbU"] * Gamma1["LK"] * Gamma1["ij"];
    temp3["IU"] = temp1["jLbU"] * Eta1["ba"] * V["aIjL"];
    temp1["LIUB"] = T2["KJUB"] * Gamma1["LK"] * Gamma1["IJ"];
    temp3["IU"] += 0.5 * temp1["LJUB"] * Eta1["BA"] * V["IALJ"];
    Hbar1["IU"] -= temp3["IU"];
    Hbar1["UI"] -= temp3["IU"];

//    Hbar1.print(stdout);
    outfile->Printf("  Done. Timing %15.6f s", timer.get());
}

void DSRG_MRPT2::Hbar1_VT2_4_22(){
    Timer timer;
    std::string str = "Computing [V, T2] C_4 C_2 2-2 for Hbar1";
    outfile->Printf("\n    %-40s ...", str.c_str());

//    Hbar1.print(stdout);
    outfile->Printf("  Done. Timing %15.6f s", timer.get());
}

void DSRG_MRPT2::Hbar1_VT2_4_13(){
    Timer timer;
    std::string str = "Computing [V, T2] C_4 C_2 1-3 for Hbar1";
    outfile->Printf("\n    %-40s ...", str.c_str());

//    Hbar1.print(stdout);
    outfile->Printf("  Done. Timing %15.6f s", timer.get());
}

}} // End Namespaces
