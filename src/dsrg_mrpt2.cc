#include <numeric>
#include <math.h>
#include <boost/format.hpp>

#include <libpsio/psio.hpp>
#include <libpsio/psio.h>
#include <libmints/molecule.h>
#include <libqt/qt.h>
#include <iostream>
#include <fstream>

#include "dsrg_mrpt2.h"
#include "blockedtensorfactory.h"

using namespace ambit;

namespace psi{ namespace forte{

DSRG_MRPT2::DSRG_MRPT2(Reference reference, boost::shared_ptr<Wavefunction> wfn, Options &options,
                       std::shared_ptr<ForteIntegrals> ints, std::shared_ptr<MOSpaceInfo> mo_space_info)
    : Wavefunction(options,_default_psio_lib_),
      reference_(reference),
      ints_(ints),
      mo_space_info_(mo_space_info),
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
    print_ = options_.get_int("PRINT");

    frozen_core_energy = ints_->frozen_core_energy();

    source_ = options_.get_str("SOURCE");

    s_ = options_.get_double("DSRG_S");
    if(s_ < 0){
        outfile->Printf("\n  S parameter for DSRG must >= 0!");
        throw PSIEXCEPTION("S parameter for DSRG must >= 0!");
    }
    taylor_threshold_ = options_.get_int("TAYLOR_THRESHOLD");
    if(taylor_threshold_ <= 0){
        outfile->Printf("\n  Threshold for Taylor expansion must be an integer greater than 0!");
        throw PSIEXCEPTION("Threshold for Taylor expansion must be an integer greater than 0!");
    }
    taylor_order_ = int(0.5 * (15.0 / taylor_threshold_ + 1)) + 1;

    ntamp_ = options_.get_int("NTAMP");
    intruder_tamp_ = options_.get_double("INTRUDER_TAMP");

    // orbital spaces
    BlockedTensor::reset_mo_spaces();
    acore_mos = mo_space_info_->get_corr_abs_mo("RESTRICTED_DOCC");
    bcore_mos = mo_space_info_->get_corr_abs_mo("RESTRICTED_DOCC");
    aactv_mos = mo_space_info_->get_corr_abs_mo("ACTIVE");
    bactv_mos = mo_space_info_->get_corr_abs_mo("ACTIVE");
    avirt_mos = mo_space_info_->get_corr_abs_mo("RESTRICTED_UOCC");
    bvirt_mos = mo_space_info_->get_corr_abs_mo("RESTRICTED_UOCC");

    // define space labels
    acore_label = "c";
    aactv_label = "a";
    avirt_label = "v";
    bcore_label = "C";
    bactv_label = "A";
    bvirt_label = "V";
    BTF->add_mo_space(acore_label,"mn",acore_mos,AlphaSpin);
    BTF->add_mo_space(bcore_label,"MN",bcore_mos,BetaSpin);
    BTF->add_mo_space(aactv_label,"uvwxyz",aactv_mos,AlphaSpin);
    BTF->add_mo_space(bactv_label,"UVWXYZ",bactv_mos,BetaSpin);
    BTF->add_mo_space(avirt_label,"ef",avirt_mos,AlphaSpin);
    BTF->add_mo_space(bvirt_label,"EF",bvirt_mos,BetaSpin);

    // map space labels to mo spaces
    label_to_spacemo[acore_label[0]] = acore_mos;
    label_to_spacemo[bcore_label[0]] = bcore_mos;
    label_to_spacemo[aactv_label[0]] = aactv_mos;
    label_to_spacemo[bactv_label[0]] = bactv_mos;
    label_to_spacemo[avirt_label[0]] = avirt_mos;
    label_to_spacemo[bvirt_label[0]] = bvirt_mos;

    // define composite spaces
    BTF->add_composite_mo_space("h","ijkl",{acore_label,aactv_label});
    BTF->add_composite_mo_space("H","IJKL",{bcore_label,bactv_label});
    BTF->add_composite_mo_space("p","abcd",{aactv_label,avirt_label});
    BTF->add_composite_mo_space("P","ABCD",{bactv_label,bvirt_label});
    BTF->add_composite_mo_space("g","pqrs",{acore_label,aactv_label,avirt_label});
    BTF->add_composite_mo_space("G","PQRS",{bcore_label,bactv_label,bvirt_label});

    // get reference energy
    Eref = reference_.get_Eref();

    // prepare integrals
    H = BTF->build(tensor_type_,"H",spin_cases({"gg"}));
    H.iterate([&](const std::vector<size_t>& i,const std::vector<SpinType>& spin,double& value){
        if (spin[0] == AlphaSpin)
            value = ints_->oei_a(i[0],i[1]);
        else
            value = ints_->oei_b(i[0],i[1]);
    });

    V = BTF->build(tensor_type_,"V",spin_cases({"gggg"}));
    V.iterate([&](const std::vector<size_t>& i,const std::vector<SpinType>& spin,double& value){
        if ((spin[0] == AlphaSpin) and (spin[1] == AlphaSpin)) value = ints_->aptei_aa(i[0],i[1],i[2],i[3]);
        if ((spin[0] == AlphaSpin) and (spin[1] == BetaSpin) ) value = ints_->aptei_ab(i[0],i[1],i[2],i[3]);
        if ((spin[0] == BetaSpin)  and (spin[1] == BetaSpin) ) value = ints_->aptei_bb(i[0],i[1],i[2],i[3]);
    });

    // prepare density matrix
    Gamma1 = BTF->build(tensor_type_,"Gamma1",spin_cases({"hh"}));
    Eta1 = BTF->build(tensor_type_,"Eta1",spin_cases({"pp"}));

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

    // prepare density cumulants
    Lambda2 = BTF->build(tensor_type_,"Lambda2",spin_cases({"aaaa"}));
    ambit::Tensor Lambda2_aa = Lambda2.block("aaaa");
    ambit::Tensor Lambda2_aA = Lambda2.block("aAaA");
    ambit::Tensor Lambda2_AA = Lambda2.block("AAAA");
    Lambda2_aa("pqrs") = reference_.L2aa()("pqrs");
    Lambda2_aA("pqrs") = reference_.L2ab()("pqrs");
    Lambda2_AA("pqrs") = reference_.L2bb()("pqrs");

    if(options_.get_str("THREEPDC") != "ZERO"){
        Lambda3 = BTF->build(tensor_type_,"Lambda3",spin_cases({"aaaaaa"}));
        ambit::Tensor Lambda3_aaa = Lambda3.block("aaaaaa");
        ambit::Tensor Lambda3_aaA = Lambda3.block("aaAaaA");
        ambit::Tensor Lambda3_aAA = Lambda3.block("aAAaAA");
        ambit::Tensor Lambda3_AAA = Lambda3.block("AAAAAA");
        Lambda3_aaa("pqrstu") = reference_.L3aaa()("pqrstu");
        Lambda3_aaA("pqrstu") = reference_.L3aab()("pqrstu");
        Lambda3_aAA("pqrstu") = reference_.L3abb()("pqrstu");
        Lambda3_AAA("pqrstu") = reference_.L3bbb()("pqrstu");
    }

    // Form the Fock matrix
    F = BTF->build(tensor_type_,"Fock",spin_cases({"gg"}));

    F["pq"]  = H["pq"];
    F["pq"] += V["pjqi"] * Gamma1["ij"];
    F["pq"] += V["pJqI"] * Gamma1["IJ"];

    F["PQ"]  = H["PQ"];
    F["PQ"] += V["jPiQ"] * Gamma1["ij"];
    F["PQ"] += V["PJQI"] * Gamma1["IJ"];

    size_t ncmo_ = mo_space_info_->size("CORRELATED");
    Fa = std::vector<double>(ncmo_);
    Fb = std::vector<double>(ncmo_);

    F.iterate([&](const std::vector<size_t>& i,const std::vector<SpinType>& spin,double& value){
        if (spin[0] == AlphaSpin and (i[0] == i[1])){
            Fa[i[0]] = value;
        }
        if (spin[0] == BetaSpin and (i[0] == i[1])){
            Fb[i[0]] = value;
        }
    });

    // keep Delta1 for renormalize_F
    Delta1 = BTF->build(tensor_type_,"Delta1",spin_cases({"hp"}));
    Delta1.iterate([&](const std::vector<size_t>& i,const std::vector<SpinType>& spin,double& value){
        if (spin[0] == AlphaSpin){
            value = Fa[i[0]] - Fa[i[1]];
        }else if (spin[0]  == BetaSpin){
            value = Fb[i[0]] - Fb[i[1]];
        }
    });

    // Prepare exponential tensors for effective Fock matrix and integrals
    RExp1 = BTF->build(tensor_type_,"RExp1",spin_cases({"hp"}));
    RExp1.iterate([&](const std::vector<size_t>& i,const std::vector<SpinType>& spin,double& value){
        if (spin[0]  == AlphaSpin){
            value = renormalized_exp(Fa[i[0]] - Fa[i[1]]);
        }else if (spin[0]  == BetaSpin){
            value = renormalized_exp(Fb[i[0]] - Fb[i[1]]);
        }
    });

    RExp2 = BTF->build(tensor_type_,"RExp2",spin_cases({"hhpp"}));
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
    if(options_.get_str("RELAX_REF") != "NONE"){
        Hbar1 = BTF->build(tensor_type_,"One-body Hbar",spin_cases({"hh"}));
        Hbar2 = BTF->build(tensor_type_,"Two-body Hbar",spin_cases({"hhhh"}));
        Hbar1["ij"] = F["ij"];
        Hbar1["IJ"] = F["IJ"];
        Hbar2["ijkl"] = V["ijkl"];
        Hbar2["iJkL"] = V["iJkL"];
        Hbar2["IJKL"] = V["IJKL"];
    }

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
    std::vector<std::pair<std::string,int>> calculation_info{
        {"ntamp", ntamp_}};

    std::vector<std::pair<std::string,double>> calculation_info_double{
        {"flow parameter",s_},
        {"taylor expansion threshold",pow(10.0,-double(taylor_threshold_))},
        {"intruder_tamp", intruder_tamp_}};

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
double DSRG_MRPT2::renormalized_denominator_ts(double D)
{
    double Z = sqrt(s_) * D;
    if(fabs(Z) < pow(0.1, taylor_threshold_)){
        return Taylor_Exp(Z, taylor_order_) * sqrt(2.0 * s_);
    }else{
        return (1.0 - std::exp(-2.0 * s_ * pow(D, 2.0))) / D;
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
    Timer DSRG_energy;
    outfile->Printf("\n\n  ==> Computing DSRG-MRPT2 ... <==\n");
    // Compute reference
//    Eref = compute_ref();

    // Compute T2 and T1
    compute_t2();
    compute_t1();

    // Compute effective integrals
    renormalize_V();
    renormalize_F();
    if(print_ > 1)  F.print(stdout);
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
    outfile->Printf("\n\n  ==> Excitation Amplitudes Summary <==\n");
    outfile->Printf("\n    Active Indices: ");
    int c = 0;
    for(const auto& idx: aactv_mos){
        outfile->Printf("%4zu ", idx);
        if(++c % 10 == 0)
            outfile->Printf("\n    %16c", ' ');
    }
    check_t1();
    check_t2();
    energy.push_back({"max(T1)", T1max});
    energy.push_back({"max(T2)", T2max});
    energy.push_back({"||T1||", T1norm});
    energy.push_back({"||T2||", T2norm});

    outfile->Printf("\n\n  ==> Possible Intruders <==\n");
    print_intruder("A", lt1a);
    print_intruder("B", lt1b);
    print_intruder("AA", lt2aa);
    print_intruder("AB", lt2ab);
    print_intruder("BB", lt2bb);

    // Print energy summary
    outfile->Printf("\n\n  ==> DSRG-MRPT2 Energy Summary <==\n");
    for (auto& str_dim : energy){
        outfile->Printf("\n    %-30s = %22.15f",str_dim.first.c_str(),str_dim.second);
    }

    Process::environment.globals["CURRENT ENERGY"] = Etotal;
    outfile->Printf("\n\n\n Energy took %8.8f s", DSRG_energy.get());

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

    T2 = BTF->build(tensor_type_,"T2 Amplitudes",spin_cases({"hhpp"}));
    T2["ijab"] = V["ijab"];
    T2["iJaB"] = V["iJaB"];
    T2["IJAB"] = V["IJAB"];

    // This is used to print the tensor out for further analysis.
    // Only used as a test for some future tensor factorizations and other
    bool print_denom;
    print_denom = options_.get_bool("PRINT_DENOM2");
    
    if(print_denom)
    {
        std::ofstream myfile;
        myfile.open ("Deltaijab.txt");
        myfile << acore_mos.size() + aactv_mos.size() << " "
               << acore_mos.size() + aactv_mos.size() << " "
               << aactv_mos.size() + avirt_mos.size() << " "
               << aactv_mos.size() + avirt_mos.size() << " \n";
        T2.iterate([&](const std::vector<size_t>& i,const std::vector<SpinType>& spin,double& value){
            if ((spin[0] == AlphaSpin) and (spin[1] == AlphaSpin)){
                double D = renormalized_denominator_ts(Fa[i[0]] + Fa[i[1]] - Fa[i[2]] - Fa[i[3]]);
                myfile << i[0] << " " << i[1] << " "
                               << i[2] << " " << i[3] << " " << D << " \n";
                }
            });
    }

    T2.iterate([&](const std::vector<size_t>& i,const std::vector<SpinType>& spin,double& value){
        if ((spin[0] == AlphaSpin) and (spin[1] == AlphaSpin)){
            value *= renormalized_denominator(Fa[i[0]] + Fa[i[1]] - Fa[i[2]] - Fa[i[3]]);
        }else if ((spin[0] == AlphaSpin) and (spin[1] == BetaSpin) ){
            value *= renormalized_denominator(Fa[i[0]] + Fb[i[1]] - Fa[i[2]] - Fb[i[3]]);
        }else if ((spin[0] == BetaSpin)  and (spin[1] == BetaSpin) ){
            value *= renormalized_denominator(Fb[i[0]] + Fb[i[1]] - Fb[i[2]] - Fb[i[3]]);
        }
    });

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

    T1 = BTF->build(tensor_type_,"T1 Amplitudes",spin_cases({"hp"}));

    BlockedTensor temp = BTF->build(tensor_type_,"temp",spin_cases({"aa"}));
    temp["xu"] = Gamma1["xu"];
    temp["XU"] = Gamma1["XU"];
    temp.iterate([&](const std::vector<size_t>& i,const std::vector<SpinType>& spin,double& value){
        if (spin[0] == AlphaSpin){
            value *= Fa[i[0]] - Fa[i[1]];
        }else{
            value *= Fb[i[0]] - Fb[i[1]];
        }
    });

    T1["ia"]  = F["ia"];
    T1["ia"] += temp["xu"] * T2["iuax"];
    T1["ia"] += temp["XU"] * T2["iUaX"];

    T1["IA"]  = F["IA"];
    T1["IA"] += temp["xu"] * T2["uIxA"];
    T1["IA"] += temp["XU"] * T2["IUAX"];

    T1.iterate([&](const std::vector<size_t>& i,const std::vector<SpinType>& spin,double& value){
        if (spin[0]  == AlphaSpin){
            value *= renormalized_denominator(Fa[i[0]] - Fa[i[1]]);
        }else{
            value *= renormalized_denominator(Fb[i[0]] - Fb[i[1]]);
        }
    });

    // zero internal amplitudes
    T1.block("AA").zero();
    T1.block("aa").zero();

    outfile->Printf("  Done. Timing %15.6f s", timer.get());
}

// Binary function to achieve sorting a vector of pair<vector, double>
// according to the double value in decending order
template <class T1, class T2, class G3 = std::greater<T2> >
struct rsort_pair_second {
    bool operator()(const std::pair<T1,T2>& left, const std::pair<T1,T2>& right) {
        G3 p;
        return p(fabs(left.second), fabs(right.second));
    }
};

void DSRG_MRPT2::check_t2()
{
    T2norm = 0.0; T2max = 0.0;
    double T2aanorm = 0.0, T2abnorm = 0.0, T2bbnorm = 0.0;
    size_t nonzero_aa = 0, nonzero_ab = 0, nonzero_bb = 0;
    std::vector<std::pair<std::vector<size_t>, double>> t2aa, t2ab, t2bb;

    // create all knids of spin maps; 0: aa, 1: ab, 2:bb
    std::map<int, double> spin_to_norm;
    std::map<int, double> spin_to_nonzero;
    std::map<int, std::vector<std::pair<std::vector<size_t>, double>>> spin_to_t2;
    std::map<int, std::vector<std::pair<std::vector<size_t>, double>>> spin_to_lt2;

    for(const std::string& block: T2.block_labels()){
        int spin = bool(isupper(block[0])) + bool(isupper(block[1]));
        std::vector<std::pair<std::vector<size_t>, double>>& temp_t2 = spin_to_t2[spin];
        std::vector<std::pair<std::vector<size_t>, double>>& temp_lt2 = spin_to_lt2[spin];

        T2.block(block).citerate([&](const std::vector<size_t>& i, const double& value){
            if(fabs(value) != 0.0){
                size_t idx0 = label_to_spacemo[block[0]][i[0]];
                size_t idx1 = label_to_spacemo[block[1]][i[1]];
                size_t idx2 = label_to_spacemo[block[2]][i[2]];
                size_t idx3 = label_to_spacemo[block[3]][i[3]];

                ++spin_to_nonzero[spin];
                spin_to_norm[spin] += pow(value, 2.0);

                if((idx0 <= idx1) && (idx2 <= idx3)){
                    std::vector<size_t> indices = {idx0, idx1, idx2, idx3};
                    std::pair<std::vector<size_t>, double> idx_value = std::make_pair(indices, value);

                    temp_t2.push_back(idx_value);
                    std::sort(temp_t2.begin(), temp_t2.end(), rsort_pair_second<std::vector<size_t>, double>());
                    if(temp_t2.size() == ntamp_ + 1){
                        temp_t2.pop_back();
                    }

                    if(fabs(value) > fabs(intruder_tamp_)){
                        temp_lt2.push_back(idx_value);
                    }
                    std::sort(temp_lt2.begin(), temp_lt2.end(), rsort_pair_second<std::vector<size_t>, double>());
                }
                T2max = T2max > fabs(value) ? T2max : fabs(value);
            }
        });
    }

    // update values
    T2aanorm = spin_to_norm[0];
    T2abnorm = spin_to_norm[1];
    T2bbnorm = spin_to_norm[2];
    T2norm = sqrt(T2aanorm + T2bbnorm + 4 * T2abnorm);

    nonzero_aa = spin_to_nonzero[0];
    nonzero_ab = spin_to_nonzero[1];
    nonzero_bb = spin_to_nonzero[2];

    t2aa = spin_to_t2[0];
    t2ab = spin_to_t2[1];
    t2bb = spin_to_t2[2];

    lt2aa = spin_to_lt2[0];
    lt2ab = spin_to_lt2[1];
    lt2bb = spin_to_lt2[2];

    // print summary
    print_amp_summary("AA", t2aa, sqrt(T2aanorm), nonzero_aa);
    print_amp_summary("AB", t2ab, sqrt(T2abnorm), nonzero_ab);
    print_amp_summary("BB", t2bb, sqrt(T2bbnorm), nonzero_bb);
}

void DSRG_MRPT2::check_t1()
{
    T1max = 0.0; T1norm = 0.0;
    double T1anorm = 0.0, T1bnorm = 0.0;
    size_t nonzero_a = 0, nonzero_b = 0;
    std::vector<std::pair<std::vector<size_t>, double>> t1a, t1b;

    // create all kinds of spin maps; true: a, false: b
    std::map<bool, double> spin_to_norm;
    std::map<bool, double> spin_to_nonzero;
    std::map<bool, std::vector<std::pair<std::vector<size_t>, double>>> spin_to_t1;
    std::map<bool, std::vector<std::pair<std::vector<size_t>, double>>> spin_to_lt1;

    for(const std::string& block: T1.block_labels()){
        bool spin_alpha = islower(block[0]) ? true : false;
        std::vector<std::pair<std::vector<size_t>, double>>& temp_t1 = spin_to_t1[spin_alpha];
        std::vector<std::pair<std::vector<size_t>, double>>& temp_lt1 = spin_to_lt1[spin_alpha];

        T1.block(block).citerate([&](const std::vector<size_t>& i, const double& value){
            if(fabs(value) != 0.0){
                size_t idx0 = label_to_spacemo[block[0]][i[0]];
                size_t idx1 = label_to_spacemo[block[1]][i[1]];

                std::vector<size_t> indices = {idx0, idx1};
                std::pair<std::vector<size_t>, double> idx_value = std::make_pair(indices, value);

                ++spin_to_nonzero[spin_alpha];
                spin_to_norm[spin_alpha] += pow(value, 2.0);

                temp_t1.push_back(idx_value);
                std::sort(temp_t1.begin(), temp_t1.end(), rsort_pair_second<std::vector<size_t>, double>());
                if(temp_t1.size() == ntamp_ + 1){
                    temp_t1.pop_back();
                }

                if(fabs(value) > fabs(intruder_tamp_)){
                    temp_lt1.push_back(idx_value);
                }
                std::sort(temp_lt1.begin(), temp_lt1.end(), rsort_pair_second<std::vector<size_t>, double>());

                T1max = T1max > fabs(value) ? T1max : fabs(value);
            }
        });
    }

    // update value
    T1anorm = spin_to_norm[true];
    T1bnorm = spin_to_norm[false];
    T1norm = sqrt(T1anorm + T1bnorm);

    nonzero_a = spin_to_nonzero[true];
    nonzero_b = spin_to_nonzero[false];

    t1a = spin_to_t1[true];
    t1b = spin_to_t1[false];

    lt1a = spin_to_lt1[true];
    lt1b = spin_to_lt1[false];

    // print summary
    print_amp_summary("A", t1a, sqrt(T1anorm), nonzero_a);
    print_amp_summary("B", t1b, sqrt(T1bnorm), nonzero_b);
}

void DSRG_MRPT2::print_amp_summary(const std::string &name,
                                   const std::vector<std::pair<std::vector<size_t>, double> > &list,
                                   const double &norm, const size_t &number_nonzero)
{
    int rank = name.size();
    std::map<char, std::string> spin_case;
    spin_case['A'] = " ";
    spin_case['B'] = "_";

    std::string indent(4, ' ');
    std::string title = indent + "Largest T" + std::to_string(rank)
            + " amplitudes for spin case " + name + ":";
    std::string spin_title;
    std::string mo_title;
    std::string line;
    std::string output;
    std::string summary;

    auto extendstr = [&](std::string s, int n){
        std::string o(s);
        while((--n) > 0) o += s;
        return o;
    };

    if(rank == 1){
        spin_title += str(boost::format(" %3c %3c %3c %3c %9c ") % spin_case[name[0]] % ' ' % spin_case[name[0]] % ' ' % ' ');
        if(spin_title.find_first_not_of(' ') != std::string::npos){
            spin_title = "\n" + indent + extendstr(spin_title, 3);
        }else{
            spin_title = "";
        }
        mo_title += str(boost::format(" %3c %3c %3c %3c %9c ") % 'i' % ' ' % 'a' % ' ' % ' ');
        mo_title = "\n" + indent + extendstr(mo_title, 3);
        for(size_t n = 0; n != list.size(); ++n){
            if(n % 3 == 0) output += "\n" + indent;
            const auto& datapair = list[n];
            std::vector<size_t> idx = datapair.first;
            output += str(boost::format("[%3d %3c %3d %3c]%9.6f ") % idx[0] % ' ' % idx[1] % ' ' % datapair.second);
        }
    }
    else if(rank == 2){
        spin_title += str(boost::format(" %3c %3c %3c %3c %9c ") % spin_case[name[0]] % spin_case[name[1]] % spin_case[name[0]] % spin_case[name[1]] % ' ');
        if(spin_title.find_first_not_of(' ') != std::string::npos){
            spin_title = "\n" + indent + extendstr(spin_title, 3);
        }else{
            spin_title = "";
        }
        mo_title += str(boost::format(" %3c %3c %3c %3c %9c ") % 'i' % 'j' % 'a' % 'b' % ' ');
        mo_title = "\n" + indent + extendstr(mo_title, 3);
        for(size_t n = 0; n != list.size(); ++n){
            if(n % 3 == 0) output += "\n" + indent;
            const auto& datapair = list[n];
            std::vector<size_t> idx = datapair.first;
            output += str(boost::format("[%3d %3d %3d %3d]%9.6f ") % idx[0] % idx[1] % idx[2] % idx[3] % datapair.second);
        }
    }else{
        outfile->Printf("\n    Printing of amplitude is implemented only for T1 and T2!");
        return;
    }

    if(output.size() != 0){
        int linesize = mo_title.size() - 2;
        line = "\n" + indent + std::string(linesize - indent.size(), '-');
        summary = "\n" + indent + "Norm of T" + std::to_string(rank) + name
                + " vector: (nonzero elements: " + std::to_string(number_nonzero) + ")";
        std::string strnorm = str(boost::format("%.15f.") % norm);
        std::string blank(linesize - summary.size() - strnorm.size() + 1, ' ');
        summary += blank + strnorm;

        output = title + spin_title + mo_title + line + output + line + summary + line;
    }
    outfile->Printf("\n%s", output.c_str());
}

void DSRG_MRPT2::print_intruder(const std::string &name,
                                const std::vector<std::pair<std::vector<size_t>, double> > &list)
{
    int rank = name.size();
    std::map<char, std::vector<double>> spin_to_F;
    spin_to_F['A'] = Fa;
    spin_to_F['B'] = Fb;

    std::string indent(4, ' ');
    std::string title = indent + "T" + std::to_string(rank) + " amplitudes larger than "
            + str(boost::format("%.4f") % intruder_tamp_) + " for spin case "  + name + ":";
    std::string col_title;
    std::string line;
    std::string output;

    if(rank == 1){
        int x = 30 + 2 * 3 + 2 - 11;
        std::string blank(x / 2, ' ');
        col_title += "\n" + indent + "    Amplitude         Value     "
                + blank + "Denominator" + std::string(x - blank.size(), ' ');
        line = "\n" + indent + std::string(col_title.size() - indent.size() - 1, '-');

        for(size_t n = 0; n != list.size(); ++n){
            const auto& datapair = list[n];
            std::vector<size_t> idx = datapair.first;
            size_t i = idx[0], a = idx[1];
            double fi = spin_to_F[name[0]][i], fa = spin_to_F[name[0]][a];
            double down = fi - fa;
            double v = datapair.second;

            output += "\n" + indent
                    + str(boost::format("[%3d %3c %3d %3c] %13.8f (%10.6f - %10.6f = %10.6f)")
                          % i % ' ' % a % ' ' % v % fi % fa % down);
        }
    }
    else if(rank == 2){
        int x = 50 + 4 * 3 + 2 - 11;
        std::string blank(x / 2, ' ');
        col_title += "\n" + indent + "    Amplitude         Value     "
                + blank + "Denominator" + std::string(x - blank.size(), ' ');
        line = "\n" + indent + std::string(col_title.size() - indent.size() - 1, '-');
        for(size_t n = 0; n != list.size(); ++n){
            const auto& datapair = list[n];
            std::vector<size_t> idx = datapair.first;
            size_t i = idx[0], j = idx[1], a = idx[2], b = idx[3];
            double fi = spin_to_F[name[0]][i], fj = spin_to_F[name[1]][j];
            double fa = spin_to_F[name[0]][a], fb = spin_to_F[name[1]][b];
            double down = fi + fj - fa - fb;
            double v = datapair.second;

            output += "\n" + indent
                    + str(boost::format("[%3d %3d %3d %3d] %13.8f (%10.6f + %10.6f - %10.6f - %10.6f = %10.6f)")
                          % i % j % a % b % v % fi % fj % fa % fb % down);
        }
    }else{
        outfile->Printf("\n    Printing of amplitude is implemented only for T1 and T2!");
        return;
    }

    if(output.size() != 0){
        output = title + col_title + line + output + line;
    }else{
        output = title + " NULL";
    }
    outfile->Printf("\n%s", output.c_str());
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
    if(options_.get_str("RELAX_REF") != "NONE"){
        Hbar2["ijuv"] = temp1["ijuv"];
        Hbar2["iJuV"] = temp1["iJuV"];
        Hbar2["IJUV"] = temp1["IJUV"];
        Hbar2["uvij"] = temp1["ijuv"];
        Hbar2["uViJ"] = temp1["iJuV"];
        Hbar2["UVIJ"] = temp1["IJUV"];

        // acv-acv-acv-acv block should be zero
        Hbar2.block("aaaa").zero();
        Hbar2.block("AAAA").zero();
//        Hbar2.print(stdout);
    }

    // Back to renormalized V
    temp2["ijab"] = temp1["ijab"];
    temp2["iJaB"] = temp1["iJaB"];
    temp2["IJAB"] = temp1["IJAB"];
    temp2["ijab"] += V["ijab"];
    temp2["iJaB"] += V["iJaB"];
    temp2["IJAB"] += V["IJAB"];
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

    // Non-diagonal Hbar1 (acv-acv block should be diagonal)
    if(options_.get_str("RELAX_REF") != "NONE"){
        Hbar1["mv"] = temp2["mv"];
        Hbar1["MV"] = temp2["MV"];
        Hbar1["vm"] = temp2["mv"];
        Hbar1["VM"] = temp2["MV"];
//        Hbar1.print(stdout);
    }

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
    E += F["em"] * T1["me"];
    E += F["ex"] * T1["ye"] * Gamma1["xy"];
    E += F["xm"] * T1["my"] * Eta1["yx"];

    E += F["EM"] * T1["ME"];
    E += F["EX"] * T1["YE"] * Gamma1["XY"];
    E += F["XM"] * T1["MY"] * Eta1["YX"];

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
    E += 0.25 * V["efmn"] * T2["mnef"];
    E += 0.25 * V["EFMN"] * T2["MNEF"];
    E += V["eFmN"] * T2["mNeF"];

    BlockedTensor temp = BTF->build(tensor_type_,"temp",spin_cases({"aa"}));
    temp["vu"] += 0.5 * V["efmu"] * T2["mvef"];
    temp["vu"] += V["fEuM"] * T2["vMfE"];
    temp["VU"] += 0.5 * V["EFMU"] * T2["MVEF"];
    temp["VU"] += V["eFmU"] * T2["mVeF"];
    E += temp["vu"] * Gamma1["uv"];
    E += temp["VU"] * Gamma1["UV"];

    temp.zero();
    temp["vu"] += 0.5 * V["vemn"] * T2["mnue"];
    temp["vu"] += V["vEmN"] * T2["mNuE"];
    temp["VU"] += 0.5 * V["VEMN"] * T2["MNUE"];
    temp["VU"] += V["eVnM"] * T2["nMeU"];
    E += temp["vu"] * Eta1["uv"];
    E += temp["VU"] * Eta1["UV"];

    temp = BTF->build(tensor_type_,"temp",spin_cases({"aaaa"}));
    temp["yvxu"] += V["efxu"] * T2["yvef"];
    temp["yVxU"] += V["eFxU"] * T2["yVeF"];
    temp["YVXU"] += V["EFXU"] * T2["YVEF"];
    E += 0.25 * temp["yvxu"] * Gamma1["xy"] * Gamma1["uv"];
    E += temp["yVxU"] * Gamma1["UV"] * Gamma1["xy"];
    E += 0.25 * temp["YVXU"] * Gamma1["XY"] * Gamma1["UV"];

    temp.zero();
    temp["vyux"] += V["vymn"] * T2["mnux"];
    temp["vYuX"] += V["vYmN"] * T2["mNuX"];
    temp["VYUX"] += V["VYMN"] * T2["MNUX"];
    E += 0.25 * temp["vyux"] * Eta1["uv"] * Eta1["xy"];
    E += temp["vYuX"] * Eta1["uv"] * Eta1["XY"];
    E += 0.25 * temp["VYUX"] * Eta1["UV"] * Eta1["XY"];

    temp.zero();
    temp["vyux"] += V["vemx"] * T2["myue"];
    temp["vyux"] += V["vExM"] * T2["yMuE"];
    temp["VYUX"] += V["eVmX"] * T2["mYeU"];
    temp["VYUX"] += V["VEXM"] * T2["YMUE"];
    E += temp["vyux"] * Gamma1["xy"] * Eta1["uv"];
    E += temp["VYUX"] * Gamma1["XY"] * Eta1["UV"];
    temp["yVxU"] = V["eVxM"] * T2["yMeU"];
    E += temp["yVxU"] * Gamma1["xy"] * Eta1["UV"];
    temp["vYuX"] = V["vEmX"] * T2["mYuE"];
    E += temp["vYuX"] * Gamma1["XY"] * Eta1["uv"];

    temp.zero();
    temp["yvxu"] += 0.5 * Gamma1["wz"] * V["vexw"] * T2["yzue"];
    temp["yvxu"] += Gamma1["WZ"] * V["vExW"] * T2["yZuE"];
    temp["yvxu"] += 0.5 * Eta1["wz"] * T2["myuw"] * V["vzmx"];
    temp["yvxu"] += Eta1["WZ"] * T2["yMuW"] * V["vZxM"];
    E += temp["yvxu"] * Gamma1["xy"] * Eta1["uv"];

    temp["YVXU"] += 0.5 * Gamma1["WZ"] * V["VEXW"] * T2["YZUE"];
    temp["YVXU"] += Gamma1["wz"] * V["eVwX"] * T2["zYeU"];
    temp["YVXU"] += 0.5 * Eta1["WZ"] * T2["MYUW"] * V["VZMX"];
    temp["YVXU"] += Eta1["wz"] * V["zVmX"] * T2["mYwU"];
    E += temp["YVXU"] * Gamma1["XY"] * Eta1["UV"];

    outfile->Printf("  Done. Timing %15.6f s", timer.get());
    return E;
}

double DSRG_MRPT2::E_VT2_4HH()
{
    Timer timer;
    std::string str = "Computing <[V, T2]> C_4 (C_2)^2 HH";
    outfile->Printf("\n    %-40s ...", str.c_str());

    double E = 0.0;
    BlockedTensor temp = BTF->build(tensor_type_,"temp",spin_cases({"aaaa"}));

    temp["uvxy"] += 0.125 * V["uvmn"] * T2["mnxy"];
    temp["uvxy"] += 0.25 * Gamma1["wz"] * V["uvmw"] * T2["mzxy"];
    temp["uVxY"] += V["uVmN"] * T2["mNxY"];
    temp["uVxY"] += Gamma1["wz"] * T2["zMxY"] * V["uVwM"];
    temp["uVxY"] += Gamma1["WZ"] * V["uVmW"] * T2["mZxY"];
    temp["UVXY"] += 0.125 * V["UVMN"] * T2["MNXY"];
    temp["UVXY"] += 0.25 * Gamma1["WZ"] * V["UVMW"] * T2["MZXY"];

    E += Lambda2["xyuv"] * temp["uvxy"];
    E += Lambda2["xYuV"] * temp["uVxY"];
    E += Lambda2["XYUV"] * temp["UVXY"];

    outfile->Printf("  Done. Timing %15.6f s", timer.get());
    return E;
}

double DSRG_MRPT2::E_VT2_4PP()
{
    Timer timer;
    std::string str = "Computing <[V, T2]> C_4 (C_2)^2 PP";
    outfile->Printf("\n    %-40s ...", str.c_str());

    double E = 0.0;
    BlockedTensor temp = BTF->build(tensor_type_,"temp",spin_cases({"aaaa"}));

    temp["uvxy"] += 0.125 * V["efxy"] * T2["uvef"];
    temp["uvxy"] += 0.25 * Eta1["wz"] * T2["uvew"] * V["ezxy"];
    temp["uVxY"] += V["eFxY"] * T2["uVeF"];
    temp["uVxY"] += Eta1["wz"] * V["zExY"] * T2["uVwE"];
    temp["uVxY"] += Eta1["WZ"] * T2["uVeW"] * V["eZxY"];
    temp["UVXY"] += 0.125 * V["EFXY"] * T2["UVEF"];
    temp["UVXY"] += 0.25 * Eta1["WZ"] * T2["UVEW"] * V["EZXY"];

    E += Lambda2["xyuv"] * temp["uvxy"];
    E += Lambda2["xYuV"] * temp["uVxY"];
    E += Lambda2["XYUV"] * temp["UVXY"];

    outfile->Printf("  Done. Timing %15.6f s", timer.get());
    return E;
}

double DSRG_MRPT2::E_VT2_4PH()
{
    Timer timer;
    std::string str = "Computing <[V, T2]> C_4 (C_2)^2 PH";
    outfile->Printf("\n    %-40s ...", str.c_str());

    double E = 0.0;
    BlockedTensor temp = BTF->build(tensor_type_,"temp",spin_cases({"aaaa"}));

    temp["uvxy"] += V["eumx"] * T2["mvey"];
    temp["uvxy"] += V["uExM"] * T2["vMyE"];
    temp["uvxy"] += Gamma1["wz"] * T2["zvey"] * V["euwx"];
    temp["uvxy"] += Gamma1["WZ"] * V["uExW"] * T2["vZyE"];
    temp["uvxy"] += Eta1["zw"] * V["wumx"] * T2["mvzy"];
    temp["uvxy"] += Eta1["ZW"] * T2["vMyZ"] * V["uWxM"];
    E += temp["uvxy"] * Lambda2["xyuv"];

    temp["UVXY"] += V["eUmX"] * T2["mVeY"];
    temp["UVXY"] += V["EUMX"] * T2["MVEY"];
    temp["UVXY"] += Gamma1["wz"] * T2["zVeY"] * V["eUwX"];
    temp["UVXY"] += Gamma1["WZ"] * T2["ZVEY"] * V["EUWX"];
    temp["UVXY"] += Eta1["zw"] * V["wUmX"] * T2["mVzY"];
    temp["UVXY"] += Eta1["ZW"] * V["WUMX"] * T2["MVZY"];
    E += temp["UVXY"] * Lambda2["XYUV"];

    temp["uVxY"] += V["uexm"] * T2["mVeY"];
    temp["uVxY"] += V["uExM"] * T2["MVEY"];
    temp["uVxY"] -= V["eVxM"] * T2["uMeY"];
    temp["uVxY"] -= V["uEmY"] * T2["mVxE"];
    temp["uVxY"] += V["eVmY"] * T2["umxe"];
    temp["uVxY"] += V["EVMY"] * T2["uMxE"];

    temp["uVxY"] += Gamma1["wz"] * T2["zVeY"] * V["uexw"];
    temp["uVxY"] += Gamma1["WZ"] * T2["ZVEY"] * V["uExW"];
    temp["uVxY"] -= Gamma1["WZ"] * V["eVxW"] * T2["uZeY"];
    temp["uVxY"] -= Gamma1["wz"] * T2["zVxE"] * V["uEwY"];
    temp["uVxY"] += Gamma1["wz"] * T2["zuex"] * V["eVwY"];
    temp["uVxY"] -= Gamma1["WZ"] * V["EVYW"] * T2["uZxE"];

    temp["uVxY"] += Eta1["zw"] * V["wumx"] * T2["mVzY"];
    temp["uVxY"] += Eta1["ZW"] * T2["VMYZ"] * V["uWxM"];
    temp["uVxY"] -= Eta1["zw"] * V["wVxM"] * T2["uMzY"];
    temp["uVxY"] -= Eta1["ZW"] * T2["mVxZ"] * V["uWmY"];
    temp["uVxY"] += Eta1["zw"] * T2["umxz"] * V["wVmY"];
    temp["uVxY"] += Eta1["ZW"] * V["WVMY"] * T2["uMxZ"];
    E += temp["uVxY"] * Lambda2["xYuV"];

    outfile->Printf("  Done. Timing %15.6f s", timer.get());
    return E;
}

double DSRG_MRPT2::E_VT2_6()
{
    Timer timer;
    std::string str = "Computing <[V, T2]> C_6 C_2";
    outfile->Printf("\n    %-40s ...", str.c_str());

    double E = 0.0;
    if(options_.get_str("THREEPDC") != "ZERO"){
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
    }

    outfile->Printf("  Done. Timing %15.6f s", timer.get());
    return E;
}

void DSRG_MRPT2::transform_integrals(){

    outfile->Printf("\n\n  ==> Building effective Hamiltonian Hbar ... <==\n");

    // Compute second-order one- and two-body Hbar
    // Cautious: V and F are renormalized !
//    Hbar1_FT1();
//    Hbar1_FT2();
//    Hbar2_FT2();
//    Hbar1_VT1();
//    Hbar2_VT1();
//    Hbar2_VT2_HH();
//    Hbar2_VT2_PP();
//    Hbar2_VT2_PH();
//    Hbar1_VT2_2();
//    Hbar1_VT2_4_22();
//    Hbar1_VT2_4_13();
    Hbar1.print(stdout);
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

    // Fill out ints->oei
    Timer fill1;
    str = "Updating one-electron integrals";
    outfile->Printf("\n    %-40s ...", str.c_str());
    for(const std::string& block: O1.block_labels()){
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
    for(const std::string& block: Hbar2.block_labels()){
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
