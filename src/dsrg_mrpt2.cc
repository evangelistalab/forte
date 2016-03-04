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
#include "fci_solver.h"

using namespace ambit;

namespace psi{ namespace forte{

DSRG_MRPT2::DSRG_MRPT2(Reference reference, SharedWavefunction ref_wfn, Options &options,
                       std::shared_ptr<ForteIntegrals> ints, std::shared_ptr<MOSpaceInfo> mo_space_info)
    : Wavefunction(options),
      reference_(reference),
      ints_(ints),
      mo_space_info_(mo_space_info),
      tensor_type_(CoreTensor),
      BTF(new BlockedTensorFactory(options))
{
    // Copy the wavefunction information
    shallow_copy(ref_wfn);

    print_method_banner({"Driven Similarity Renormalization Group MBPT2", "Chenyang Li, Kevin Hannon, Francesco Evangelista"});
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

    frozen_core_energy_ = ints_->frozen_core_energy();

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
    Eref_ = reference_.get_Eref();

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

    // initialize timer for commutator
    dsrg_time_ = DSRG_TIME();
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
    dsrg_time_.print_comm_time();
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
    energy.push_back({"E0 (reference)", Eref_});

    Etemp  = E_FT1();
    Ecorr += Etemp;
    energy.push_back({"<[F, T1]>", Etemp});

    Etemp  = E_FT2();
    Ecorr += Etemp;
    energy.push_back({"<[F, T2]>", Etemp});

    Etemp  = E_VT1();
    Ecorr += Etemp;
    energy.push_back({"<[V, T1]>", Etemp});

    Etemp  = E_VT2_2();
    EVT2 += Etemp;
    energy.push_back({"<[V, T2]> (C_2)^4", Etemp});

    Etemp  = E_VT2_4HH();
    EVT2 += Etemp;
    energy.push_back({"<[V, T2]> C_4 (C_2)^2 HH", Etemp});

    Etemp  = E_VT2_4PP();
    EVT2 += Etemp;
    energy.push_back({"<[V, T2]> C_4 (C_2)^2 PP", Etemp});

    Etemp  = E_VT2_4PH();
    EVT2 += Etemp;
    energy.push_back({"<[V, T2]> C_4 (C_2)^2 PH", Etemp});

    Etemp  = E_VT2_6();
    EVT2 += Etemp;
    energy.push_back({"<[V, T2]> C_6 C_2", Etemp});

    Ecorr += EVT2;
    Etotal = Ecorr + Eref_;
    energy.push_back({"<[V, T2]>", EVT2});
    energy.push_back({"DSRG-MRPT2 correlation energy", Ecorr});
    energy.push_back({"DSRG-MRPT2 total energy", Etotal});
    Hbar0_ = Ecorr;

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
    outfile->Printf("\n\n  Energy took %8.8f s", DSRG_energy.get());

    // relax reference
    if(options_.get_str("RELAX_REF") != "NONE"){
        C1 = BTF->build(tensor_type_,"C1",spin_cases({"hh"}));
        C2 = BTF->build(tensor_type_,"C2",spin_cases({"hhhh"}));
        H1_T1_C1(F,T1,0.5,C1);
        H1_T2_C1(F,T2,0.5,C1);
        H2_T1_C1(V,T1,0.5,C1);
        H2_T2_C1(V,T2,0.5,C1);
        H1_T2_C2(F,T2,0.5,C2);
        H2_T1_C2(V,T1,0.5,C2);
        H2_T2_C2(V,T2,0.5,C2);

        Hbar1["ij"] += C1["ij"];
        Hbar1["ij"] += C1["ji"];
        Hbar1["IJ"] += C1["IJ"];
        Hbar1["IJ"] += C1["JI"];
        Hbar2["ijkl"] += C2["ijkl"];
        Hbar2["ijkl"] += C2["klij"];
        Hbar2["iJkL"] += C2["iJkL"];
        Hbar2["iJkL"] += C2["kLiJ"];
        Hbar2["IJKL"] += C2["IJKL"];
        Hbar2["IJKL"] += C2["KLIJ"];
    }

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
    return E + frozen_core_energy_ + Enuc;
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
        T2.iterate([&](const std::vector<size_t>& i,const std::vector<SpinType>& spin,double&){
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

    temp1["ijab"] = V["ijab"] * RExp2["ijab"];
    temp1["iJaB"] = V["iJaB"] * RExp2["iJaB"];
    temp1["IJAB"] = V["IJAB"] * RExp2["IJAB"];

    temp1["ijab"] += V["ijab"];
    temp1["iJaB"] += V["iJaB"];
    temp1["IJAB"] += V["IJAB"];

    V["ijab"] = temp1["ijab"];
    V["iJaB"] = temp1["iJaB"];
    V["IJAB"] = temp1["IJAB"];

    V["abij"] = temp1["ijab"];
    V["aBiJ"] = temp1["iJaB"];
    V["ABIJ"] = temp1["IJAB"];

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

    F["ia"] += temp2["ia"];
    F["IA"] += temp2["IA"];

    // avoid double counting the a-a block
    F["am"] += temp2["ma"];
    F["AM"] += temp2["MA"];
    F["eu"] += temp2["ue"];
    F["EU"] += temp2["UE"];

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
    dsrg_time_.add("110",timer.get());
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
    dsrg_time_.add("210",timer.get());
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
    dsrg_time_.add("120",timer.get());
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
    dsrg_time_.add("220",timer.get());
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
    dsrg_time_.add("220",timer.get());
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
    dsrg_time_.add("220",timer.get());
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
    dsrg_time_.add("220",timer.get());
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
        temp["uvWxyZ"] += 2.0 * V["uWyI"] * T2["vIxZ"];//  aaAaaA from hole

        temp["uvWxyZ"] += V["aWxZ"] * T2["uvay"];      //  aaAaaA from particle
        temp["uvWxyZ"] -= V["vaxy"] * T2["uWaZ"];      //  aaAaaA from particle
        temp["uvWxyZ"] -= 2.0 * V["vAxZ"] * T2["uWyA"];//  aaAaaA from particle
        E += 0.50 * temp["uvWxyZ"] * Lambda3["xyZuvW"];

        temp["uVWxYZ"] -= V["VWIZ"] * T2["uIxY"];      //  aAAaAA from hole
        temp["uVWxYZ"] -= V["uVxI"] * T2["IWYZ"];      //  aAAaAA from hole
        temp["uVWxYZ"] += 2.0 * V["uViZ"] * T2["iWxY"];//  aAAaAA from hole

        temp["uVWxYZ"] += V["uAxY"] * T2["VWAZ"];      //  aAAaAA from particle
        temp["uVWxYZ"] -= V["WAYZ"] * T2["uVxA"];      //  aAAaAA from particle
        temp["uVWxYZ"] -= 2.0 * V["aWxY"] * T2["uVaZ"];//  aAAaAA from particle
        E += 0.5 * temp["uVWxYZ"] * Lambda3["xYZuVW"];
    }

    outfile->Printf("  Done. Timing %15.6f s", timer.get());
    dsrg_time_.add("220",timer.get());
    return E;
}

void DSRG_MRPT2::build_density(){
    // prepare density matrices
    (Gamma1.block("cc")).iterate([&](const std::vector<size_t>& i,double& value){
        value = i[0] == i[1] ? 1.0 : 0.0;});
    (Gamma1.block("CC")).iterate([&](const std::vector<size_t>& i,double& value){
        value = i[0] == i[1] ? 1.0 : 0.0;});
    (Eta1.block("aa")).iterate([&](const std::vector<size_t>& i,double& value){
        value = i[0] == i[1] ? 1.0 : 0.0;});
    (Eta1.block("AA")).iterate([&](const std::vector<size_t>& i,double& value){
        value = i[0] == i[1] ? 1.0 : 0.0;});
    (Eta1.block("vv")).iterate([&](const std::vector<size_t>& i,double& value){
        value = i[0] == i[1] ? 1.0 : 0.0;});
    (Eta1.block("VV")).iterate([&](const std::vector<size_t>& i,double& value){
        value = i[0] == i[1] ? 1.0 : 0.0;});
    // symmetrize beta spin
    outfile->Printf("\n  Warning: I am forcing density Db = Da to avoid spin symmetry breaking.");
    outfile->Printf("\n  If this is not desired, go to dsrg_mrpt2.cc around line 1250.");
    Gamma1.block("aa")("pq") = reference_.L1a()("pq");
    Gamma1.block("AA")("pq") = reference_.L1a()("pq");
    Eta1.block("aa")("pq") -= reference_.L1a()("pq");
    Eta1.block("AA")("pq") -= reference_.L1a()("pq");

    // prepare two-body density cumulants
    ambit::Tensor Lambda2_aa = Lambda2.block("aaaa");
    ambit::Tensor Lambda2_aA = Lambda2.block("aAaA");
    ambit::Tensor Lambda2_AA = Lambda2.block("AAAA");
    Lambda2_aa("pqrs") = reference_.L2aa()("pqrs");
    Lambda2_aA("pqrs") = reference_.L2ab()("pqrs");
    Lambda2_AA("pqrs") = reference_.L2bb()("pqrs");

    // prepare three-body density cumulants
    if(options_.get_str("THREEPDC") != "ZERO"){
        ambit::Tensor Lambda3_aaa = Lambda3.block("aaaaaa");
        ambit::Tensor Lambda3_aaA = Lambda3.block("aaAaaA");
        ambit::Tensor Lambda3_aAA = Lambda3.block("aAAaAA");
        ambit::Tensor Lambda3_AAA = Lambda3.block("AAAAAA");
        Lambda3_aaa("pqrstu") = reference_.L3aaa()("pqrstu");
        Lambda3_aaA("pqrstu") = reference_.L3aab()("pqrstu");
        Lambda3_aAA("pqrstu") = reference_.L3abb()("pqrstu");
        Lambda3_AAA("pqrstu") = reference_.L3bbb()("pqrstu");
    }
}

void DSRG_MRPT2::build_fock(BlockedTensor& H, BlockedTensor& V){
    // build Fock matrix
    F["pq"]  = H["pq"];
    F["pq"] += V["pjqi"] * Gamma1["ij"];
    F["pq"] += V["pJqI"] * Gamma1["IJ"];
    F["PQ"]  = H["PQ"];
    F["PQ"] += V["jPiQ"] * Gamma1["ij"];
    F["PQ"] += V["PJQI"] * Gamma1["IJ"];

    // obtain diagonal elements of Fock matrix
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
}

void DSRG_MRPT2::reset_ints(BlockedTensor& H, BlockedTensor& V){
    ints_->set_scalar(0.0);
    H.citerate([&](const std::vector<size_t>& i,const std::vector<SpinType>& spin,const double& value){
        if (spin[0] == AlphaSpin){
            ints_->set_oei(i[0],i[1],value,true);
        }else{
            ints_->set_oei(i[0],i[1],value,false);
        }
    });
    V.citerate([&](const std::vector<size_t>& i,const std::vector<SpinType>& spin,const double& value){
        if ((spin[0] == AlphaSpin) && (spin[1] == AlphaSpin)){
            ints_->set_tei(i[0],i[1],i[2],i[3],value,true,true);
        }else if ((spin[0] == AlphaSpin) && (spin[1] == BetaSpin)){
            ints_->set_tei(i[0],i[1],i[2],i[3],value,true,false);
        }else if ((spin[0] == BetaSpin)  && (spin[1] == BetaSpin)){
            ints_->set_tei(i[0],i[1],i[2],i[3],value,false,false);
        }
    });
}

double DSRG_MRPT2::compute_energy_relaxed(){
    // setup for FCISolver
    Dimension active_dim = mo_space_info_->get_dimension("ACTIVE");
    int charge = Process::environment.molecule()->molecular_charge();
    if(options_["CHARGE"].has_changed()){
        charge = options_.get_int("CHARGE");
    }
    auto nelec = 0;
    int natom = Process::environment.molecule()->natom();
    for(int i = 0; i < natom; ++i){
        nelec += Process::environment.molecule()->fZ(i);
    }
    nelec -= charge;
    int multi = Process::environment.molecule()->multiplicity();
    if(options_["MULTIPLICITY"].has_changed()){
        multi = options_.get_int("MULTIPLICITY");
    }
    int ms = multi - 1;
    if(options_["MS"].has_changed()){
        ms = options_.get_int("MS");
    }
    auto nelec_actv = nelec - 2 * mo_space_info_->size("FROZEN_DOCC") - 2 * acore_mos.size();
    auto na = (nelec_actv + ms) / 2;
    auto nb =  nelec_actv - na;

    // reference relaxation
    double Edsrg = 0.0, Erelax = 0.0;
    std::string relax_algorithm = options_.get_str("RELAX_REF");

    // only relax once, otherwise we have to store another
    if(relax_algorithm != "NONE"){
        // compute energy with fixed ref.
        Edsrg = compute_energy();

//        Hbar2.print();

        // transfer integrals
        transfer_integrals();

        // diagonalize the Hamiltonian
        FCISolver fcisolver(active_dim,acore_mos,aactv_mos,na,nb,multi,options_.get_int("ROOT_SYM"),ints_, mo_space_info_,
                                             options_.get_int("NTRIAL_PER_ROOT"),print_, options_);
        fcisolver.set_max_rdm_level(2);
        Erelax = fcisolver.compute_energy();

        // printing
        print_h2("MRDSRG Energy Summary");
        outfile->Printf("\n    %-30s = %22.15f", "MRDSRG Total Energy (fixed)", Edsrg);
        outfile->Printf("\n    %-30s = %22.15f", "MRDSRG Total Energy (relaxed)", Erelax);
        outfile->Printf("\n");
    }

    Process::environment.globals["CURRENT ENERGY"] = Erelax;
    return Erelax;
}

void DSRG_MRPT2::transfer_integrals(){
    // printing
    print_h2("De-Normal-Order the DSRG Transformed Hamiltonian");

    // compute scalar term
    Timer t_scalar;
    std::string str = "Computing the scalar term   ...";
    outfile->Printf("\n    %-35s", str.c_str());
    double scalar0 = Eref_ + Hbar0_ - molecule_->nuclear_repulsion_energy()
            - ints_->frozen_core_energy();

    // scalar from Hbar1
    double scalar1 = 0.0;
    Hbar1.block("cc").citerate([&](const std::vector<size_t>& i,const double& value){
        if (i[0] == i[1]) scalar1 -= value;
    });
    Hbar1.block("CC").citerate([&](const std::vector<size_t>& i,const double& value){
        if (i[0] == i[1]) scalar1 -= value;
    });
    scalar1 -= Hbar1["vu"] * Gamma1["uv"];
    scalar1 -= Hbar1["VU"] * Gamma1["UV"];

    // scalar from Hbar2
    double scalar2 = 0.0;
    scalar2 -= 0.25 * Hbar2["xyuv"] * Lambda2["uvxy"];
    scalar2 -= 0.25 * Hbar2["XYUV"] * Lambda2["UVXY"];
    scalar2 -= Hbar2["xYuV"] * Lambda2["uVxY"];
    Hbar2.block("cccc").citerate([&](const std::vector<size_t>& i,const double& value){
        if ((i[0] == i[2]) && (i[1] == i[3])) scalar2 += 0.5 * value;
    });
    Hbar2.block("cCcC").citerate([&](const std::vector<size_t>& i,const double& value){
        if ((i[0] == i[2]) && (i[1] == i[3])) scalar2 += value;
    });
    Hbar2.block("CCCC").citerate([&](const std::vector<size_t>& i,const double& value){
        if ((i[0] == i[2]) && (i[1] == i[3])) scalar2 += 0.5 * value;
    });

    C1.zero();
    C1["ij"] += Hbar2["iujv"] * Gamma1["vu"];
    C1["ij"] += Hbar2["iUjV"] * Gamma1["VU"];
    C1["IJ"] += Hbar2["uIvJ"] * Gamma1["vu"];
    C1["IJ"] += Hbar2["IUJV"] * Gamma1["VU"];
    C1.block("cc").citerate([&](const std::vector<size_t>& i,const double& value){
        if (i[0] == i[1]) scalar2 += value;
    });
    C1.block("CC").citerate([&](const std::vector<size_t>& i,const double& value){
        if (i[0] == i[1]) scalar2 += value;
    });
    scalar2 += 0.5 * Gamma1["uv"] * Hbar2["vyux"] * Gamma1["xy"];
    scalar2 += 0.5 * Gamma1["UV"] * Hbar2["VYUX"] * Gamma1["XY"];
    scalar2 += Gamma1["uv"] * Hbar2["vYuX"] * Gamma1["XY"];

    double scalar = scalar0 + scalar1 + scalar2;
    outfile->Printf("  Done. Timing %10.3f s", t_scalar.get());

    // compute one-body term
    Timer t_one;
    str = "Computing the one-body term ...";
    outfile->Printf("\n    %-35s", str.c_str());
    C1.scale(-1.0);
    C1["ij"] += Hbar1["ij"];
    C1["IJ"] += Hbar1["IJ"];
    BlockedTensor temp = BTF->build(tensor_type_,"temp",spin_cases({"cc"}));
    temp.iterate([&](const std::vector<size_t>& i,const std::vector<SpinType>&,double& value){
        if (i[0] == i[1]) value = 1.0;
    });
    C1["ij"] -= Hbar2["imjn"] * temp["nm"];
    C1["ij"] -= Hbar2["iMjN"] * temp["NM"];
    C1["IJ"] -= Hbar2["mInJ"] * temp["nm"];
    C1["IJ"] -= Hbar2["IMJN"] * temp["NM"];
    outfile->Printf("  Done. Timing %10.3f s", t_one.get());

    // update integrals
    Timer t_int;
    str = "Updating integrals          ...";
    outfile->Printf("\n    %-35s", str.c_str());
    ints_->set_scalar(scalar);
    C1.citerate([&](const std::vector<size_t>& i,const std::vector<SpinType>& spin,const double& value){
        if (spin[0] == AlphaSpin){
            ints_->set_oei(i[0],i[1],value,true);
        }else{
            ints_->set_oei(i[0],i[1],value,false);
        }
    });
    Hbar2.citerate([&](const std::vector<size_t>& i,const std::vector<SpinType>& spin,const double& value){
        if ((spin[0] == AlphaSpin) && (spin[1] == AlphaSpin)){
            ints_->set_tei(i[0],i[1],i[2],i[3],value,true,true);
        }else if ((spin[0] == AlphaSpin) && (spin[1] == BetaSpin)){
            ints_->set_tei(i[0],i[1],i[2],i[3],value,true,false);
        }else if ((spin[0] == BetaSpin)  && (spin[1] == BetaSpin)){
            ints_->set_tei(i[0],i[1],i[2],i[3],value,false,false);
        }
    });
    outfile->Printf("  Done. Timing %10.3f s", t_int.get());

    // print scalar
    double scalar_include_fc = scalar + ints_->frozen_core_energy();
    print_h2("Scalar of the DSRG Hamiltonian (WRT True Vacuum)");
    outfile->Printf("\n    %-30s = %22.15f", "Scalar0", scalar0);
    outfile->Printf("\n    %-30s = %22.15f", "Scalar1", scalar1);
    outfile->Printf("\n    %-30s = %22.15f", "Scalar2", scalar2);
    outfile->Printf("\n    %-30s = %22.15f", "Total Scalar W/O Frozen-Core", scalar);
    outfile->Printf("\n    %-30s = %22.15f", "Total Scalar W/  Frozen-Core", scalar_include_fc);

    // test if de-normal-ordering is correct
    print_h2("Test De-Normal-Ordered Hamiltonian");
    double Etest = scalar_include_fc + molecule_->nuclear_repulsion_energy();

    double Etest1 = 0.0;
    C1.block("cc").citerate([&](const std::vector<size_t>& i,const double& value){
        if (i[0] == i[1]) Etest1 += value;
    });
    C1.block("CC").citerate([&](const std::vector<size_t>& i,const double& value){
        if (i[0] == i[1]) Etest1 += value;
    });
    Etest1 += C1["uv"] * Gamma1["vu"];
    Etest1 += C1["UV"] * Gamma1["VU"];

    double Etest2 = 0.0;
    Hbar2.block("cccc").citerate([&](const std::vector<size_t>& i,const double& value){
        if ((i[0] == i[2]) && (i[1] == i[3])) Etest2 += 0.5 * value;
    });
    Hbar2.block("cCcC").citerate([&](const std::vector<size_t>& i,const double& value){
        if ((i[0] == i[2]) && (i[1] == i[3])) Etest2 += value;
    });
    Hbar2.block("CCCC").citerate([&](const std::vector<size_t>& i,const double& value){
        if ((i[0] == i[2]) && (i[1] == i[3])) Etest2 += 0.5 * value;
    });

    Etest2 += Hbar2["munv"] * temp["nm"] * Gamma1["vu"];
    Etest2 += Hbar2["uMvN"] * temp["NM"] * Gamma1["vu"];
    Etest2 += Hbar2["mUnV"] * temp["nm"] * Gamma1["VU"];
    Etest2 += Hbar2["MUNV"] * temp["NM"] * Gamma1["VU"];

    Etest2 += 0.5 * Gamma1["vu"] * Hbar2["uxvy"] * Gamma1["yx"];
    Etest2 += 0.5 * Gamma1["VU"] * Hbar2["UXVY"] * Gamma1["YX"];
    Etest2 += Gamma1["vu"] * Hbar2["uXvY"] * Gamma1["YX"];

    Etest2 += 0.25 * Hbar2["uvxy"] * Lambda2["xyuv"];
    Etest2 += 0.25 * Hbar2["UVXY"] * Lambda2["XYUV"];
    Etest2 += Hbar2["uVxY"] * Lambda2["xYuV"];

    Etest += Etest1 + Etest2;
    outfile->Printf("\n    %-30s = %22.15f", "One-Body Energy (after)", Etest1);
    outfile->Printf("\n    %-30s = %22.15f", "Two-Body Energy (after)", Etest2);
    outfile->Printf("\n    %-30s = %22.15f", "Total Energy (after)", Etest);
    outfile->Printf("\n    %-30s = %22.15f", "Total Energy (before)", Eref_ + Hbar0_);

    if(fabs(Etest - Eref_ - Hbar0_) > 100.0 * options_.get_double("E_CONVERGENCE")){
        throw PSIEXCEPTION("De-normal-odering failed.");
    }else{
        ints_->update_integrals(false);
    }
}

//void DSRG_MRPT2::transfer_integrals(){

//    outfile->Printf("\n\n  ==> Building effective Hamiltonian Hbar ... <==\n");

//    // Compute second-order one- and two-body Hbar
//    // Cautious: V and F are renormalized !
////    Hbar1_FT1();
////    Hbar1_FT2();
////    Hbar2_FT2();
////    Hbar1_VT1();
////    Hbar2_VT1();
////    Hbar2_VT2_HH();
////    Hbar2_VT2_PP();
////    Hbar2_VT2_PH();
////    Hbar1_VT2_2();
////    Hbar1_VT2_4_22();
////    Hbar1_VT2_4_13();
//    Hbar1.print(stdout);
////    Hbar2.print(stdout);

//    // Scalar term
//    outfile->Printf("\n\n  ==> Computing scalar term in Hbar ... <==\n");
//    boost::shared_ptr<Molecule> molecule = Process::environment.molecule();
//    double Enuc = molecule->nuclear_repulsion_energy();
//    double scalar = Hbar0_ - Enuc;

//    scalar -= Hbar1["ji"] * Gamma1["ij"];
//    scalar -= Hbar1["JI"] * Gamma1["IJ"];

//    scalar -= 0.25 * Hbar2["xyuv"] * Lambda2["uvxy"];
//    scalar -= 0.25 * Hbar2["XYUV"] * Lambda2["UVXY"];
//    scalar -= Hbar2["xYuV"] * Lambda2["uVxY"];

//    scalar += 0.5 * Hbar2["klij"] * Gamma1["ik"] * Gamma1["jl"];
//    scalar += 0.5 * Hbar2["KLIJ"] * Gamma1["IK"] * Gamma1["JL"];
//    scalar += Hbar2["kLiJ"] * Gamma1["ik"] * Gamma1["JL"];

//    outfile->Printf("\n    %-30s = %22.15f", "Hbar scalar", scalar);

//    // Update integrals
//    ints_->set_scalar(scalar);
//    outfile->Printf("\n\n  ==> Updating all integrals ... <==\n");

//    // Compute one-body operator (two-body operator is Hbar2)
//    Timer o1;
//    std::string str = "Computing new one-electron integrals";
//    outfile->Printf("\n    %-40s ...", str.c_str());
//    BlockedTensor O1 = BTF->build(tensor_type_, "One-body operator", spin_cases({"hh"}));
//    O1["ij"]  = Hbar1["ij"];
//    O1["ij"] -= Hbar2["ikjl"] * Gamma1["lk"];
//    O1["ij"] -= Hbar2["iKjL"] * Gamma1["LK"];
//    O1["IJ"]  = Hbar1["IJ"];
//    O1["IJ"] -= Hbar2["kIlJ"] * Gamma1["lk"];
//    O1["IJ"] -= Hbar2["IKJL"] * Gamma1["LK"];
//    outfile->Printf("  Done. Timing %15.6f s", o1.get());

//    // Fill out ints->oei
//    Timer fill1;
//    str = "Updating one-electron integrals";
//    outfile->Printf("\n    %-40s ...", str.c_str());
//    for(const std::string& block: O1.block_labels()){
//        bool spin = islower(block[0]) ? true : false;
//        O1.block(block).citerate([&](const std::vector<size_t>& i,const double& value){
//            size_t idx0 = label_to_spacemo[block[0]][i[0]];
//            size_t idx1 = label_to_spacemo[block[1]][i[1]];
//            ints_->set_oei(idx0,idx1,value,spin);
//        });
//    }
//    outfile->Printf("  Done. Timing %15.6f s", fill1.get());

//    // Fill out ints->tei
//    Timer fill2;
//    str = "Updating two-electron integrals";
//    outfile->Printf("\n    %-40s ...", str.c_str());
//    for(const std::string& block: Hbar2.block_labels()){
//        bool spin0 = islower(block[0]) ? true : false;
//        bool spin1 = islower(block[1]) ? true : false;
//        Hbar2.block(block).citerate([&](const std::vector<size_t>& i,const double& value){
//            size_t idx0 = label_to_spacemo[block[0]][i[0]];
//            size_t idx1 = label_to_spacemo[block[1]][i[1]];
//            size_t idx2 = label_to_spacemo[block[2]][i[2]];
//            size_t idx3 = label_to_spacemo[block[3]][i[3]];
//            ints_->set_tei(idx0,idx1,idx2,idx3,value,spin0,spin1);
//        });
//    }
//    outfile->Printf("  Done. Timing %15.6f s", fill2.get());

//    ints_->update_integrals(false);
//}

void DSRG_MRPT2::H1_T1_C1(BlockedTensor& H1, BlockedTensor& T1, const double& alpha, BlockedTensor& C1){
    Timer timer;

    C1["ij"] += alpha * H1["aj"] * T1["ia"];
    C1["ju"] -= alpha * T1["iu"] * H1["ji"];

    C1["IJ"] += alpha * H1["AJ"] * T1["IA"];
    C1["JU"] -= alpha * T1["IU"] * H1["JI"];

    if(print_ > 2){
        outfile->Printf("\n    Time for [H1, T1] -> C1 : %12.3f",timer.get());
    }
    dsrg_time_.add("111",timer.get());
}

void DSRG_MRPT2::H1_T2_C1(BlockedTensor& H1, BlockedTensor& T2, const double& alpha, BlockedTensor& C1){
    Timer timer;

    C1["iu"] += alpha * H1["bm"] * T2["imub"];
    C1["ix"] += alpha * H1["bu"] * T2["ivxb"] * Gamma1["uv"];
    C1["ix"] -= alpha * H1["vj"] * T2["ijxu"] * Gamma1["uv"];
    C1["iu"] += alpha * H1["BM"] * T2["iMuB"];
    C1["ix"] += alpha * H1["BU"] * T2["iVxB"] * Gamma1["UV"];
    C1["ix"] -= alpha * H1["VJ"] * T2["iJxU"] * Gamma1["UV"];

    C1["IU"] += alpha * H1["bm"] * T2["mIbU"];
    C1["IX"] += alpha * H1["bu"] * Gamma1["uv"] * T2["vIbX"];
    C1["IX"] -= alpha * H1["vj"] * T2["jIuX"] * Gamma1["uv"];
    C1["IU"] += alpha * H1["BM"] * T2["IMUB"];
    C1["IX"] += alpha * H1["BU"] * T2["IVXB"] * Gamma1["UV"];
    C1["IX"] -= alpha * H1["VJ"] * T2["IJXU"] * Gamma1["UV"];

    if(print_ > 2){
        outfile->Printf("\n    Time for [H1, T2] -> C1 : %12.3f",timer.get());
    }
    dsrg_time_.add("121",timer.get());
}

void DSRG_MRPT2::H2_T1_C1(BlockedTensor& H2, BlockedTensor& T1, const double& alpha, BlockedTensor& C1){
    Timer timer;

    C1["ij"] += alpha * T1["ma"] * H2["iajm"];
    C1["ij"] += alpha * T1["xe"] * Gamma1["yx"] * H2["iejy"];
    C1["ij"] -= alpha * T1["mu"] * Gamma1["uv"] * H2["ivjm"];
    C1["ij"] += alpha * T1["MA"] * H2["iAjM"];
    C1["ij"] += alpha * T1["XE"] * Gamma1["YX"] * H2["iEjY"];
    C1["ij"] -= alpha * T1["MU"] * Gamma1["UV"] * H2["iVjM"];

    C1["IJ"] += alpha * T1["ma"] * H2["aImJ"];
    C1["IJ"] += alpha * T1["xe"] * Gamma1["yx"] * H2["eIyJ"];
    C1["IJ"] -= alpha * T1["mu"] * Gamma1["uv"] * H2["vImJ"];
    C1["IJ"] += alpha * T1["MA"] * H2["IAJM"];
    C1["IJ"] += alpha * T1["XE"] * Gamma1["YX"] * H2["IEJY"];
    C1["IJ"] -= alpha * T1["MU"] * Gamma1["UV"] * H2["IVJM"];

    if(print_ > 2){
        outfile->Printf("\n    Time for [H2, T1] -> C1 : %12.3f",timer.get());
    }
    dsrg_time_.add("211",timer.get());
}

void DSRG_MRPT2::H2_T2_C1(BlockedTensor& H2, BlockedTensor& T2, const double& alpha, BlockedTensor& C1){
    Timer timer;
    BlockedTensor temp;

    // [Hbar2, T2] (C_2)^3 -> C1 particle contractions
    C1["ij"] += 0.5 * alpha * H2["abjm"] * T2["imab"];
    C1["ij"] += alpha * H2["aBjM"] * T2["iMaB"];
    C1["IJ"] += 0.5 * alpha * H2["ABJM"] * T2["IMAB"];
    C1["IJ"] += alpha * H2["aBmJ"] * T2["mIaB"];

    C1["ij"] += 0.5 * alpha * Gamma1["uv"] * H2["abju"] * T2["ivab"];
    C1["ij"] += alpha * Gamma1["UV"] * H2["aBjU"] * T2["iVaB"];
    C1["IJ"] += 0.5 * alpha * Gamma1["UV"] * H2["ABJU"] * T2["IVAB"];
    C1["IJ"] += alpha * Gamma1["uv"] * H2["aBuJ"] * T2["vIaB"];

    C1["ik"] += 0.5 * alpha * T2["ijux"] * Gamma1["xy"] * Gamma1["uv"] * H2["vykj"];
    C1["IK"] += 0.5 * alpha * T2["IJUX"] * Gamma1["XY"] * Gamma1["UV"] * H2["VYKJ"];
    temp = ambit::BlockedTensor::build(tensor_type_,"temp",{"hHaA"});
    temp["iJvY"] = T2["iJuX"] * Gamma1["XY"] * Gamma1["uv"];
    C1["ik"] += alpha * temp["iJvY"] * H2["vYkJ"];
    C1["IK"] += alpha * temp["jIvY"] * H2["vYjK"];

    C1["ij"] -= alpha * Gamma1["uv"] * H2["vbjm"] * T2["imub"];
    C1["ij"] -= alpha * Gamma1["uv"] * H2["vBjM"] * T2["iMuB"];
    C1["ij"] -= alpha * Gamma1["UV"] * T2["iMbU"] * H2["bVjM"];
    C1["IJ"] -= alpha * Gamma1["UV"] * H2["VBJM"] * T2["IMUB"];
    C1["IJ"] -= alpha * Gamma1["UV"] * H2["bVmJ"] * T2["mIbU"];
    C1["IJ"] -= alpha * Gamma1["uv"] * H2["vBmJ"] * T2["mIuB"];

    C1["ij"] -= alpha * H2["vbjx"] * Gamma1["uv"] * Gamma1["xy"] * T2["iyub"];
    C1["ij"] -= alpha * H2["vBjX"] * Gamma1["uv"] * Gamma1["XY"] * T2["iYuB"];
    C1["ij"] -= alpha * H2["bVjX"] * Gamma1["XY"] * Gamma1["UV"] * T2["iYbU"];
    C1["IJ"] -= alpha * H2["VBJX"] * Gamma1["UV"] * Gamma1["XY"] * T2["IYUB"];
    C1["IJ"] -= alpha * H2["vBxJ"] * Gamma1["uv"] * Gamma1["xy"] * T2["yIuB"];
    C1["IJ"] -= alpha * T2["yIbU"] * Gamma1["UV"] * Gamma1["xy"] * H2["bVxJ"];

    // [Hbar2, T2] (C_2)^3 -> C1 hole contractions
    C1["ku"] -= 0.5 * alpha * H2["keij"] * T2["ijue"];
    C1["ku"] -= alpha * H2["kEiJ"] * T2["iJuE"];
    C1["KU"] -= 0.5 * alpha * H2["KEIJ"] * T2["IJUE"];
    C1["KU"] -= alpha * H2["eKiJ"] * T2["iJeU"];

    C1["kx"] -= 0.5 * alpha * Eta1["uv"] * T2["ijxu"] * H2["kvij"];
    C1["kx"] -= alpha * Eta1["UV"] * T2["iJxU"] * H2["kViJ"];
    C1["KX"] -= 0.5 * alpha * Eta1["UV"] * T2["IJXU"] * H2["KVIJ"];
    C1["KX"] -= alpha * Eta1["uv"] * T2["iJuX"] * H2["vKiJ"];

    C1["kw"] -= 0.5 * alpha * T2["vywb"] * Eta1["uv"] * Eta1["xy"] * H2["kbux"];
    C1["KW"] -= 0.5 * alpha * T2["VYWB"] * Eta1["UV"] * Eta1["XY"] * H2["KBUX"];
    temp = ambit::BlockedTensor::build(tensor_type_,"temp",{"aApP"});
    temp["uXaB"] = T2["vYaB"] * Eta1["uv"] * Eta1["XY"];
    C1["kw"] -= alpha * H2["kBuX"] * temp["uXwB"];
    C1["KW"] -= alpha * H2["bKuX"] * temp["uXbW"];

    C1["kx"] += alpha * Eta1["uv"] * T2["vjxe"] * H2["keuj"];
    C1["kx"] += alpha * Eta1["uv"] * T2["vJxE"] * H2["kEuJ"];
    C1["kx"] += alpha * Eta1["UV"] * H2["kEjU"] * T2["jVxE"];
    C1["KX"] += alpha * Eta1["UV"] * T2["VJXE"] * H2["KEUJ"];
    C1["KX"] += alpha * Eta1["uv"] * T2["vJeX"] * H2["eKuJ"];
    C1["KX"] += alpha * Eta1["UV"] * H2["eKjU"] * T2["jVeX"];

    C1["kw"] += alpha * T2["vjwx"] * Eta1["uv"] * Eta1["xy"] * H2["kyuj"];
    C1["kw"] += alpha * T2["vJwX"] * Eta1["uv"] * Eta1["XY"] * H2["kYuJ"];
    C1["kw"] += alpha * T2["jVwX"] * Eta1["XY"] * Eta1["UV"] * H2["kYjU"];
    C1["KW"] += alpha * T2["VJWX"] * Eta1["UV"] * Eta1["XY"] * H2["KYUJ"];
    C1["KW"] += alpha * T2["vJxW"] * Eta1["uv"] * Eta1["xy"] * H2["yKuJ"];
    C1["KW"] += alpha * H2["yKjU"] * Eta1["UV"] * Eta1["xy"] * T2["jVxW"];

    // [Hbar2, T2] C_4 C_2 2:2 -> C1
    C1["ik"] +=  0.25 * alpha * T2["ijxy"] * Lambda2["xyuv"] * H2["uvkj"];
    C1["IK"] +=  0.25 * alpha * T2["IJXY"] * Lambda2["XYUV"] * H2["UVKJ"];
    temp = ambit::BlockedTensor::build(tensor_type_,"temp",{"hHaA"});
    temp["iJuV"] = T2["iJxY"] * Lambda2["xYuV"];
    C1["ik"] += alpha * H2["uVkJ"] * temp["iJuV"];
    C1["IK"] += alpha * H2["uVjK"] * temp["jIuV"];

    C1["iw"] -=  0.25 * alpha * Lambda2["xyuv"] * T2["uvwb"] * H2["ibxy"];
    C1["IW"] -=  0.25 * alpha * Lambda2["XYUV"] * T2["UVWB"] * H2["IBXY"];
    temp = ambit::BlockedTensor::build(tensor_type_,"temp",{"aApP"});
    temp["xYaB"] = T2["uVaB"] * Lambda2["xYuV"];
    C1["iw"] -= alpha * H2["iBxY"] * temp["xYwB"];
    C1["IW"] -= alpha * H2["bIxY"] * temp["xYbW"];

    C1["ij"] -= alpha * Lambda2["yXuV"] * T2["iVyA"] * H2["uAjX"];
    C1["IJ"] -= alpha * Lambda2["xYvU"] * T2["vIaY"] * H2["aUxJ"];
    C1["kw"] += alpha * Lambda2["xYvU"] * T2["vIwY"] * H2["kUxI"];
    C1["KW"] += alpha * Lambda2["yXuV"] * T2["iVyW"] * H2["uKiX"];

    temp = ambit::BlockedTensor::build(tensor_type_,"temp",{"hapa"});
    temp["ixau"] += Lambda2["xyuv"] * T2["ivay"];
    temp["ixau"] += Lambda2["xYuV"] * T2["iVaY"];
    C1["ij"] += alpha * temp["ixau"] * H2["aujx"];
    C1["kw"] -= alpha * H2["kuix"] * temp["ixwu"];
    temp = ambit::BlockedTensor::build(tensor_type_,"temp",{"hApA"});
    temp["iXaU"] += Lambda2["XYUV"] * T2["iVaY"];
    temp["iXaU"] += Lambda2["yXvU"] * T2["ivay"];
    C1["ij"] += alpha * temp["iXaU"] * H2["aUjX"];
    C1["kw"] -= alpha * H2["kUiX"] * temp["iXwU"];
    temp = ambit::BlockedTensor::build(tensor_type_,"temp",{"aHaP"});
    temp["xIuA"] += Lambda2["xyuv"] * T2["vIyA"];
    temp["xIuA"] += Lambda2["xYuV"] * T2["VIYA"];
    C1["IJ"] += alpha * temp["xIuA"] * H2["uAxJ"];
    C1["KW"] -= alpha * H2["uKxI"] * temp["xIuW"];
    temp = ambit::BlockedTensor::build(tensor_type_,"temp",{"HAPA"});
    temp["IXAU"] += Lambda2["XYUV"] * T2["IVAY"];
    temp["IXAU"] += Lambda2["yXvU"] * T2["vIyA"];
    C1["IJ"] += alpha * temp["IXAU"] * H2["AUJX"];
    C1["KW"] -= alpha * H2["KUIX"] * temp["IXWU"];

    // [Hbar2, T2] C_4 C_2 1:3 -> C1
    temp = ambit::BlockedTensor::build(tensor_type_,"temp",{"pa"});
    temp["au"] += 0.5 * Lambda2["xyuv"] * H2["avxy"];
    temp["au"] += Lambda2["xYuV"] * H2["aVxY"];
    C1["jx"] += alpha * temp["au"] * T2["ujax"];
    C1["JX"] += alpha * temp["au"] * T2["uJaX"];
    temp = ambit::BlockedTensor::build(tensor_type_,"temp",{"PA"});
    temp["AU"] += 0.5 * Lambda2["XYUV"] * H2["AVXY"];
    temp["AU"] += Lambda2["xYvU"] * H2["vAxY"];
    C1["jx"] += alpha * temp["AU"] * T2["jUxA"];
    C1["JX"] += alpha * temp["AU"] * T2["UJAX"];

    temp = ambit::BlockedTensor::build(tensor_type_,"temp",{"ah"});
    temp["xi"] += 0.5 * Lambda2["xyuv"] * H2["uviy"];
    temp["xi"] += Lambda2["xYuV"] * H2["uViY"];
    C1["ju"] -= alpha * temp["xi"] * T2["ijxu"];
    C1["JU"] -= alpha * temp["xi"] * T2["iJxU"];
    temp = ambit::BlockedTensor::build(tensor_type_,"temp",{"AH"});
    temp["XI"] += 0.5 * Lambda2["XYUV"] * H2["UVIY"];
    temp["XI"] += Lambda2["yXvU"] * H2["vUyI"];
    C1["ju"] -= alpha * temp["XI"] * T2["jIuX"];
    C1["JU"] -= alpha * temp["XI"] * T2["IJXU"];

    temp = ambit::BlockedTensor::build(tensor_type_,"temp",{"av"});
    temp["xe"] += 0.5 * T2["uvey"] * Lambda2["xyuv"];
    temp["xe"] += T2["uVeY"] * Lambda2["xYuV"];
    C1["ij"] += alpha * temp["xe"] * H2["eixj"];
    C1["IJ"] += alpha * temp["xe"] * H2["eIxJ"];
    temp = ambit::BlockedTensor::build(tensor_type_,"temp",{"AV"});
    temp["XE"] += 0.5 * T2["UVEY"] * Lambda2["XYUV"];
    temp["XE"] += T2["uVyE"] * Lambda2["yXuV"];
    C1["ij"] += alpha * temp["XE"] * H2["iEjX"];
    C1["IJ"] += alpha * temp["XE"] * H2["EIXJ"];

    temp = ambit::BlockedTensor::build(tensor_type_,"temp",{"ca"});
    temp["mu"] += 0.5 * T2["mvxy"] * Lambda2["xyuv"];
    temp["mu"] += T2["mVxY"] * Lambda2["xYuV"];
    C1["ij"] -= alpha * temp["mu"] * H2["uimj"];
    C1["IJ"] -= alpha * temp["mu"] * H2["uImJ"];
    temp = ambit::BlockedTensor::build(tensor_type_,"temp",{"CA"});
    temp["MU"] += 0.5 * T2["MVXY"] * Lambda2["XYUV"];
    temp["MU"] += T2["vMxY"] * Lambda2["xYvU"];
    C1["ij"] -= alpha * temp["MU"] * H2["iUjM"];
    C1["IJ"] -= alpha * temp["MU"] * H2["UIMJ"];

    if(print_ > 2){
        outfile->Printf("\n    Time for [H2, T2] -> C1 : %12.3f",timer.get());
    }
    dsrg_time_.add("221",timer.get());
}

void DSRG_MRPT2::H1_T2_C2(BlockedTensor& H1, BlockedTensor& T2, const double& alpha, BlockedTensor& C2){
    Timer timer;

    C2["ijku"] += alpha * T2["ijau"] * H1["ak"];
    C2["ijuk"] += alpha * T2["ijub"] * H1["bk"];
    C2["kjuv"] -= alpha * T2["ijuv"] * H1["ki"];
    C2["ikuv"] -= alpha * T2["ijuv"] * H1["kj"];

    C2["iJkU"] += alpha * T2["iJaU"] * H1["ak"];
    C2["iJuK"] += alpha * T2["iJuB"] * H1["BK"];
    C2["kJuV"] -= alpha * T2["iJuV"] * H1["ki"];
    C2["iKuV"] -= alpha * T2["iJuV"] * H1["KJ"];

    C2["IJKU"] += alpha * T2["IJAU"] * H1["AK"];
    C2["IJUK"] += alpha * T2["IJUB"] * H1["BK"];
    C2["KJUV"] -= alpha * T2["IJUV"] * H1["KI"];
    C2["IKUV"] -= alpha * T2["IJUV"] * H1["KJ"];

    if(print_ > 2){
        outfile->Printf("\n    Time for [H1, T2] -> C2 : %12.3f",timer.get());
    }
    dsrg_time_.add("122",timer.get());
}

void DSRG_MRPT2::H2_T1_C2(BlockedTensor& H2, BlockedTensor& T1, const double& alpha, BlockedTensor& C2){
    Timer timer;

    C2["ijkl"] += alpha * T1["ia"] * H2["ajkl"];
    C2["jikl"] += alpha * T1["ia"] * H2["jakl"];
    C2["kluj"] -= alpha * T1["iu"] * H2["klij"];
    C2["klju"] -= alpha * T1["iu"] * H2["klji"];

    C2["iJkL"] += alpha * T1["ia"] * H2["aJkL"];
    C2["jIkL"] += alpha * T1["IA"] * H2["jAkL"];
    C2["kLuJ"] -= alpha * T1["iu"] * H2["kLiJ"];
    C2["kLjU"] -= alpha * T1["IU"] * H2["kLjI"];

    C2["IJKL"] += alpha * T1["IA"] * H2["AJKL"];
    C2["JIKL"] += alpha * T1["IA"] * H2["JAKL"];
    C2["KLUJ"] -= alpha * T1["IU"] * H2["KLIJ"];
    C2["KLJU"] -= alpha * T1["IU"] * H2["KLJI"];

    if(print_ > 2){
        outfile->Printf("\n    Time for [H2, T1] -> C2 : %12.3f",timer.get());
    }
    dsrg_time_.add("212",timer.get());
}

void DSRG_MRPT2::H2_T2_C2(BlockedTensor& H2, BlockedTensor& T2, const double& alpha, BlockedTensor& C2){
    Timer timer;

    // particle-particle contractions
    C2["ijkl"] += 0.5 * alpha * H2["abkl"] * T2["ijab"];
    C2["iJkL"] += alpha * H2["aBkL"] * T2["iJaB"];
    C2["IJKL"] += 0.5 * alpha * H2["ABKL"] * T2["IJAB"];

    C2["ijkl"] -= alpha * Gamma1["xy"] * H2["ybkl"] * T2["ijxb"];
    C2["iJkL"] -= alpha * Gamma1["xy"] * H2["yBkL"] * T2["iJxB"];
    C2["iJkL"] -= alpha * Gamma1["XY"] * T2["iJbX"] * H2["bYkL"];
    C2["IJKL"] -= alpha * Gamma1["XY"] * H2["YBKL"] * T2["IJXB"];

    // hole-hole contractions
    C2["kluv"] += 0.5 * alpha * H2["klij"] * T2["ijuv"];
    C2["kLuV"] += alpha * H2["kLiJ"] * T2["iJuV"];
    C2["KLUV"] += 0.5 * alpha * H2["KLIJ"] * T2["IJUV"];

    C2["kluv"] -= alpha * Eta1["xy"] * T2["yjuv"] * H2["klxj"];
    C2["kLuV"] -= alpha * Eta1["xy"] * T2["yJuV"] * H2["kLxJ"];
    C2["kLuV"] -= alpha * Eta1["XY"] * H2["kLjX"] * T2["jYuV"];
    C2["KLUV"] -= alpha * Eta1["XY"] * T2["YJUV"] * H2["KLXJ"];

    // hole-particle contractions
    BlockedTensor temp = ambit::BlockedTensor::build(tensor_type_,"temp",{"hhha"});
    temp["kjlu"] += alpha * H2["akml"] * T2["mjau"];
    temp["kjlu"] += alpha * H2["kAlM"] * T2["jMuA"];
    temp["kjlu"] += alpha * Gamma1["xy"] * T2["yjau"] * H2["akxl"];
    temp["kjlu"] += alpha * Gamma1["XY"] * T2["jYuA"] * H2["kAlX"];
    temp["kjlu"] -= alpha * Gamma1["xy"] * H2["ykil"] * T2["ijxu"];
    temp["kjlu"] -= alpha * Gamma1["XY"] * H2["kYlI"] * T2["jIuX"];
    C2["kjlu"] += temp["kjlu"];
    C2["jklu"] -= temp["kjlu"];
    C2["kjul"] -= temp["kjlu"];
    C2["jkul"] += temp["kjlu"];

    temp = ambit::BlockedTensor::build(tensor_type_,"temp",{"HHHA"});
    temp["KJLU"] += alpha * H2["AKML"] * T2["MJAU"];
    temp["KJLU"] += alpha * H2["aKmL"] * T2["mJaU"];
    temp["KJLU"] += alpha * Gamma1["XY"] * T2["YJAU"] * H2["AKXL"];
    temp["KJLU"] += alpha * Gamma1["xy"] * T2["yJaU"] * H2["aKxL"];
    temp["KJLU"] -= alpha * Gamma1["XY"] * H2["YKIL"] * T2["IJXU"];
    temp["KJLU"] -= alpha * Gamma1["xy"] * H2["yKiL"] * T2["iJxU"];
    C2["KJLU"] += temp["KJLU"];
    C2["JKLU"] -= temp["KJLU"];
    C2["KJUL"] -= temp["KJLU"];
    C2["JKUL"] += temp["KJLU"];

    C2["kJlU"] += alpha * H2["akml"] * T2["mJaU"];
    C2["kJlU"] += alpha * H2["kAlM"] * T2["MJAU"];
    C2["kJlU"] += alpha * Gamma1["xy"] * T2["yJaU"] * H2["akxl"];
    C2["kJlU"] += alpha * Gamma1["XY"] * T2["YJAU"] * H2["kAlX"];
    C2["kJlU"] -= alpha * Gamma1["xy"] * H2["ykil"] * T2["iJxU"];
    C2["kJlU"] -= alpha * Gamma1["XY"] * H2["kYlI"] * T2["IJXU"];

    C2["iKlU"] -= alpha * T2["iMaU"] * H2["aKlM"];
    C2["iKlU"] -= alpha * Gamma1["XY"] * T2["iYaU"] * H2["aKlX"];
    C2["iKlU"] += alpha * Gamma1["xy"] * H2["yKlJ"] * T2["iJxU"];

    C2["kJuL"] -= alpha * T2["mJuB"] * H2["kBmL"];
    C2["kJuL"] -= alpha * Gamma1["xy"] * T2["yJuB"] * H2["kBxL"];
    C2["kJuL"] += alpha * Gamma1["XY"] * H2["kYiL"] * T2["iJuX"];

    C2["iKuL"] += alpha * T2["imub"] * H2["bKmL"];
    C2["iKuL"] += alpha * T2["iMuB"] * H2["BKML"];
    C2["iKuL"] += alpha * Gamma1["xy"] * T2["iyub"] * H2["bKxL"];
    C2["iKuL"] += alpha * Gamma1["XY"] * T2["iYuB"] * H2["BKXL"];
    C2["iKuL"] -= alpha * Gamma1["xy"] * H2["yKjL"] * T2["ijux"];
    C2["iKuL"] -= alpha * Gamma1["XY"] * H2["YKJL"] * T2["iJuX"];

    if(print_ > 2){
        outfile->Printf("\n    Time for [H2, T2] -> C2 : %12.3f",timer.get());
    }
    dsrg_time_.add("222",timer.get());
}

//std::vector<std::vector<double>> DSRG_MRPT2::diagonalize_Fock_diagblocks(BlockedTensor& U){
//    // diagonal blocks identifiers (C-A-V ordering)
//    std::vector<std::string> blocks = diag_labels();

//    // map MO space label to its Dimension
//    std::map<std::string, Dimension> MOlabel_to_dimension;
//    MOlabel_to_dimension[acore_label_] = mo_space_info_->get_dimension("RESTRICTED_DOCC");
//    MOlabel_to_dimension[aactv_label_] = mo_space_info_->get_dimension("ACTIVE");
//    MOlabel_to_dimension[avirt_label_] = mo_space_info_->get_dimension("RESTRICTED_UOCC");

//    // eigen values to be returned
//    size_t ncmo = mo_space_info_->size("CORRELATED");
//    Dimension corr = mo_space_info_->get_dimension("CORRELATED");
//    std::vector<double> eigenvalues_a(ncmo, 0.0);
//    std::vector<double> eigenvalues_b(ncmo, 0.0);

//    // map MO space label to its offset Dimension
//    std::map<std::string, Dimension> MOlabel_to_offset_dimension;
//    int nirrep = corr.n();
//    MOlabel_to_offset_dimension[acore_label_] = Dimension(std::vector<int> (nirrep, 0));
//    MOlabel_to_offset_dimension[aactv_label_] = mo_space_info_->get_dimension("RESTRICTED_DOCC");
//    MOlabel_to_offset_dimension[avirt_label_] = mo_space_info_->get_dimension("RESTRICTED_DOCC") + mo_space_info_->get_dimension("ACTIVE");

//    // figure out index
//    auto fill_eigen = [&](std::string block_label, int irrep, std::vector<double> values){
//        int h = irrep;
//        size_t idx_begin = 0;
//        while((--h) >= 0) idx_begin += corr[h];

//        std::string label (1, tolower(block_label[0]));
//        idx_begin += MOlabel_to_offset_dimension[label][irrep];

//        bool spin_alpha = islower(block_label[0]);
//        size_t nvalues = values.size();
//        if(spin_alpha){
//            for(size_t i = 0; i < nvalues; ++i){
//                eigenvalues_a[i + idx_begin] = values[i];
//            }
//        }else{
//            for(size_t i = 0; i < nvalues; ++i){
//                eigenvalues_b[i + idx_begin] = values[i];
//            }
//        }
//    };

//    // diagonalize diagonal blocks
//    for(const auto& block: blocks){
//        size_t dim = F_.block(block).dim(0);
//        if(dim == 0){
//            continue;
//        }else{
//            std::string label (1, tolower(block[0]));
//            Dimension space = MOlabel_to_dimension[label];
//            int nirrep = space.n();

//            // separate Fock with irrep
//            for(int h = 0; h < nirrep; ++h){
//                size_t h_dim = space[h];
//                ambit::Tensor U_h;
//                if(h_dim == 0){
//                    continue;
//                }else if(h_dim == 1){
//                    U_h = ambit::Tensor::build(tensor_type_,"U_h",std::vector<size_t> (2, h_dim));
//                    U_h.data()[0] = 1.0;
//                    ambit::Tensor F_block = ambit::Tensor::build(tensor_type_,"F_block",F_.block(block).dims());
//                    F_block.data() = F_.block(block).data();
//                    ambit::Tensor T_h = separate_tensor(F_block,space,h);
//                    fill_eigen(block,h,T_h.data());
//                }else{
//                    ambit::Tensor F_block = ambit::Tensor::build(tensor_type_,"F_block",F_.block(block).dims());
//                    F_block.data() = F_.block(block).data();
//                    ambit::Tensor T_h = separate_tensor(F_block,space,h);
//                    auto Feigen = T_h.syev(AscendingEigenvalue);
//                    U_h = ambit::Tensor::build(tensor_type_,"U_h",std::vector<size_t> (2, h_dim));
//                    U_h("pq") = Feigen["eigenvectors"]("pq");
//                    fill_eigen(block,h,Feigen["eigenvalues"].data());
//                }
//                ambit::Tensor U_out = U.block(block);
//                combine_tensor(U_out,U_h,space,h);
//            }
//        }
//    }
//    return {eigenvalues_a, eigenvalues_b};
//}

//ambit::Tensor DSRG_MRPT2::separate_tensor(ambit::Tensor& tens, const Dimension& irrep, const int& h){
//    // test tens and irrep
//    int tens_dim = static_cast<int>(tens.dim(0));
//    if(tens_dim != irrep.sum() || tens_dim != tens.dim(1)){
//        throw PSIEXCEPTION("Wrong dimension for the to-be-separated ambit Tensor.");
//    }
//    if(h >= irrep.n()){
//        throw PSIEXCEPTION("Ask for wrong irrep.");
//    }

//    // from relative (blocks) to absolute (big tensor) index
//    auto rel_to_abs = [&](size_t i, size_t j, size_t offset){
//        return (i + offset) * tens_dim + (j + offset);
//    };

//    // compute offset
//    size_t offset = 0, h_dim = irrep[h];
//    int h_local = h;
//    while((--h_local) >= 0) offset += irrep[h_local];

//    // fill in values
//    ambit::Tensor T_h = ambit::Tensor::build(tensor_type_,"T_h",std::vector<size_t> (2, h_dim));
//    for(size_t i = 0; i < h_dim; ++i){
//        for(size_t j = 0; j < h_dim; ++j){
//            size_t abs_idx = rel_to_abs(i, j, offset);
//            T_h.data()[i * h_dim + j] = tens.data()[abs_idx];
//        }
//    }

//    return T_h;
//}

//void DSRG_MRPT2::combine_tensor(ambit::Tensor& tens, ambit::Tensor& tens_h, const Dimension& irrep, const int& h){
//    // test tens and irrep
//    if(h >= irrep.n()){
//        throw PSIEXCEPTION("Ask for wrong irrep.");
//    }
//    size_t tens_h_dim = tens_h.dim(0), h_dim = irrep[h];
//    if(tens_h_dim != h_dim || tens_h_dim != tens_h.dim(1)){
//        throw PSIEXCEPTION("Wrong dimension for the to-be-combined ambit Tensor.");
//    }

//    // from relative (blocks) to absolute (big tensor) index
//    size_t tens_dim = tens.dim(0);
//    auto rel_to_abs = [&](size_t i, size_t j, size_t offset){
//        return (i + offset) * tens_dim + (j + offset);
//    };

//    // compute offset
//    size_t offset = 0;
//    int h_local = h;
//    while((--h_local) >= 0) offset += irrep[h_local];

//    // fill in values
//    for(size_t i = 0; i < h_dim; ++i){
//        for(size_t j = 0; j < h_dim; ++j){
//            size_t abs_idx = rel_to_abs(i, j, offset);
//            tens.data()[abs_idx] = tens_h.data()[i * h_dim + j];
//        }
//    }
//}

}} // End Namespaces
