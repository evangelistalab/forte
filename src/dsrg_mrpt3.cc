#include <numeric>
#include <math.h>
#include <chrono>
#include <boost/format.hpp>

#include <libpsio/psio.hpp>
#include <libpsio/psio.h>
#include <libmints/molecule.h>
#include <libqt/qt.h>
#include <iostream>
#include <fstream>

#include "dsrg_mrpt3.h"
#include "blockedtensorfactory.h"
#include "fci_solver.h"

using namespace ambit;

namespace psi{ namespace forte{

DSRG_MRPT3::DSRG_MRPT3(Reference reference, SharedWavefunction ref_wfn, Options &options,
                       std::shared_ptr<ForteIntegrals> ints, std::shared_ptr<MOSpaceInfo> mo_space_info)
    : Wavefunction(options), reference_(reference), ints_(ints), mo_space_info_(mo_space_info),
      tensor_type_(ambit::CoreTensor), BTF_(new BlockedTensorFactory(options))
{
    // Copy the wavefunction information
    shallow_copy(ref_wfn);

    print_method_banner({"Driven Similarity Renormalization Group MBPT3", "Chenyang Li"});
    outfile->Printf("\n    Reference:");
    outfile->Printf("\n      J. Chem. Phys. 2016 (in preparation)");

    startup();
    print_summary();
}

DSRG_MRPT3::~DSRG_MRPT3()
{
    cleanup();
}

void DSRG_MRPT3::startup()
{
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

    source_ = options_.get_str("SOURCE");
    if(source_ != "STANDARD" && source_ != "LABS" && source_ != "DYSON"){
        outfile->Printf("\n  Warning: SOURCE option \"%s\" is not implemented in DSRG-MRPT3. Changed to STANDARD.", source_.c_str());
        source_ = "STANDARD";
    }
    if(source_ == "STANDARD"){
        dsrg_source_ = std::make_shared<STD_SOURCE>(s_,taylor_threshold_);
    }else if(source_ == "LABS"){
        dsrg_source_ = std::make_shared<LABS_SOURCE>(s_,taylor_threshold_);
    }else if(source_ == "DYSON"){
        dsrg_source_ = std::make_shared<DYSON_SOURCE>(s_,taylor_threshold_);
    }

    ntamp_ = options_.get_int("NTAMP");
    intruder_tamp_ = options_.get_double("INTRUDER_TAMP");

    // get frozen core energy
    frozen_core_energy_ = ints_->frozen_core_energy();

    // get reference energy
    Eref_ = reference_.get_Eref();

    // orbital spaces
    BlockedTensor::reset_mo_spaces();
    BlockedTensor::set_expert_mode(true);
    acore_mos_ = mo_space_info_->get_corr_abs_mo("RESTRICTED_DOCC");
    bcore_mos_ = mo_space_info_->get_corr_abs_mo("RESTRICTED_DOCC");
    aactv_mos_ = mo_space_info_->get_corr_abs_mo("ACTIVE");
    bactv_mos_ = mo_space_info_->get_corr_abs_mo("ACTIVE");
    avirt_mos_ = mo_space_info_->get_corr_abs_mo("RESTRICTED_UOCC");
    bvirt_mos_ = mo_space_info_->get_corr_abs_mo("RESTRICTED_UOCC");

    // define space labels
    acore_label_ = "c";
    aactv_label_ = "a";
    avirt_label_ = "v";
    bcore_label_ = "C";
    bactv_label_ = "A";
    bvirt_label_ = "V";
    BTF_->add_mo_space(acore_label_,"mn",acore_mos_,AlphaSpin);
    BTF_->add_mo_space(bcore_label_,"MN",bcore_mos_,BetaSpin);
    BTF_->add_mo_space(aactv_label_,"uvwxyz",aactv_mos_,AlphaSpin);
    BTF_->add_mo_space(bactv_label_,"UVWXYZ",bactv_mos_,BetaSpin);
    BTF_->add_mo_space(avirt_label_,"ef",avirt_mos_,AlphaSpin);
    BTF_->add_mo_space(bvirt_label_,"EF",bvirt_mos_,BetaSpin);

    // map space labels to mo spaces
    label_to_spacemo_[acore_label_[0]] = acore_mos_;
    label_to_spacemo_[bcore_label_[0]] = bcore_mos_;
    label_to_spacemo_[aactv_label_[0]] = aactv_mos_;
    label_to_spacemo_[bactv_label_[0]] = bactv_mos_;
    label_to_spacemo_[avirt_label_[0]] = avirt_mos_;
    label_to_spacemo_[bvirt_label_[0]] = bvirt_mos_;

    // define composite spaces
    BTF_->add_composite_mo_space("h","ijkl",{acore_label_,aactv_label_});
    BTF_->add_composite_mo_space("H","IJKL",{bcore_label_,bactv_label_});
    BTF_->add_composite_mo_space("p","abcd",{aactv_label_,avirt_label_});
    BTF_->add_composite_mo_space("P","ABCD",{bactv_label_,bvirt_label_});
    BTF_->add_composite_mo_space("g","pqrsto",{acore_label_,aactv_label_,avirt_label_});
    BTF_->add_composite_mo_space("G","PQRSTO",{bcore_label_,bactv_label_,bvirt_label_});

    // fill in density matrices
    Eta1_ = BTF_->build(tensor_type_,"Eta1",spin_cases({"aa"}));
    Gamma1_ = BTF_->build(tensor_type_,"Gamma1",spin_cases({"aa"}));
    Lambda2_ = BTF_->build(tensor_type_,"Lambda2",spin_cases({"aaaa"}));
    if(options_.get_str("THREEPDC") != "ZERO"){
        Lambda3_ = BTF_->build(tensor_type_,"Lambda3",spin_cases({"aaaaaa"}));
    }
    build_density();

    // prepare integrals
    V_ = BTF_->build(tensor_type_,"V",spin_cases({"pphh"}));
    build_tei();

    // build Fock matrix and its diagonal elements
    size_t ncmo = mo_space_info_->size("CORRELATED");
    Fa_ = std::vector<double>(ncmo);
    Fb_ = std::vector<double>(ncmo);
    F_ = BTF_->build(tensor_type_,"Fock",spin_cases({"gc","pa","vv"}));
    build_fock_half();

    // check semi-canonical orbitals
    print_h2("Checking Orbitals");
    semi_canonical_ = check_semicanonical();
    if(!semi_canonical_){
        outfile->Printf("\n    Orbital invariant formalism is employed for DSRG-MRPT3.");
        U_ = ambit::BlockedTensor::build(tensor_type_,"U",spin_cases({"gg"}));
        std::vector<std::vector<double>> eigens = diagonalize_Fock_diagblocks(U_);
        Fa_ = eigens[0];
        Fb_ = eigens[1];
    }else{
        outfile->Printf("\n    Orbitals are semi-canonicalized.");
    }

    // Prepare Hbar
    relax_ref_ = options_.get_str("RELAX_REF");
    if(relax_ref_ != "NONE"){
        if(relax_ref_ != "ONCE"){
            outfile->Printf("\n  Warning: RELAX_REF option \"%s\" is not supported. Change to ONCE", relax_ref_.c_str());
            relax_ref_ = "ONCE";
        }

        Hbar1_ = BTF_->build(tensor_type_,"One-body Hbar",spin_cases({"aa"}));
        Hbar2_ = BTF_->build(tensor_type_,"Two-body Hbar",spin_cases({"aaaa"}));
        Hbar1_["uv"] = F_["uv"];
        Hbar1_["UV"] = F_["UV"];
        Hbar2_["uvxy"] = V_["uvxy"];
        Hbar2_["uVxY"] = V_["uVxY"];
        Hbar2_["UVXY"] = V_["UVXY"];
    }

    // initialize timer for commutator
    dsrg_time_ = DSRG_TIME();

    // Print levels
    print_ = options_.get_int("PRINT");
    if(print_ > 1){
        Gamma1_.print(stdout);
        Eta1_.print(stdout);
        F_.print(stdout);
    }
    if(print_ > 2){
        V_.print(stdout);
        Lambda2_.print(stdout);
    }
    if(print_ > 3){
        Lambda3_.print(stdout);
    }
}

void DSRG_MRPT3::build_density(){
    // prepare one-particle and one-hole densities
    Gamma1_.block("aa")("pq") = reference_.L1a()("pq");
    Gamma1_.block("AA")("pq") = reference_.L1a()("pq");

    (Eta1_.block("aa")).iterate([&](const std::vector<size_t>& i,double& value){
        value = i[0] == i[1] ? 1.0 : 0.0;
    });
    (Eta1_.block("AA")).iterate([&](const std::vector<size_t>& i,double& value){
        value = i[0] == i[1] ? 1.0 : 0.0;
    });
    Eta1_.block("aa")("pq") -= reference_.L1a()("pq");
    Eta1_.block("AA")("pq") -= reference_.L1a()("pq");

    // prepare two-body density cumulants
    Lambda2_.block("aaaa")("pqrs") = reference_.L2aa()("pqrs");
    Lambda2_.block("aAaA")("pqrs") = reference_.L2ab()("pqrs");
    Lambda2_.block("AAAA")("pqrs") = reference_.L2bb()("pqrs");

    // prepare three-body density cumulants
    if(options_.get_str("THREEPDC") != "ZERO"){
        Lambda3_.block("aaaaaa")("pqrstu") = reference_.L3aaa()("pqrstu");
        Lambda3_.block("aaAaaA")("pqrstu") = reference_.L3aab()("pqrstu");
        Lambda3_.block("aAAaAA")("pqrstu") = reference_.L3abb()("pqrstu");
        Lambda3_.block("AAAAAA")("pqrstu") = reference_.L3bbb()("pqrstu");
    }
}

void DSRG_MRPT3::build_tei(){
    V_.iterate([&](const std::vector<size_t>& i,const std::vector<SpinType>& spin,double& value){
        if ((spin[0] == AlphaSpin) and (spin[1] == AlphaSpin)) value = ints_->aptei_aa(i[0],i[1],i[2],i[3]);
        if ((spin[0] == AlphaSpin) and (spin[1] == BetaSpin) ) value = ints_->aptei_ab(i[0],i[1],i[2],i[3]);
        if ((spin[0] == BetaSpin)  and (spin[1] == BetaSpin) ) value = ints_->aptei_bb(i[0],i[1],i[2],i[3]);
    });
}

void DSRG_MRPT3::build_fock_half(){
    for(const auto& block: F_.block_labels()){
        // lowercase: alpha spin
        if(islower(block[0])){
            F_.block(block).iterate([&](const std::vector<size_t>& i,double& value){
                size_t np = label_to_spacemo_[block[0]][i[0]];
                size_t nq = label_to_spacemo_[block[1]][i[1]];
                value = ints_->oei_a(np,nq);

                for(const size_t& nm: acore_mos_) {
                    value += ints_->aptei_aa(np,nm,nq,nm);
                    value += ints_->aptei_ab(np,nm,nq,nm);
                }
            });
        } else {
            F_.block(block).iterate([&](const std::vector<size_t>& i,double& value){
                size_t np = label_to_spacemo_[block[0]][i[0]];
                size_t nq = label_to_spacemo_[block[1]][i[1]];
                value = ints_->oei_b(np,nq);

                for(const size_t& nm: bcore_mos_) {
                    value += ints_->aptei_bb(np,nm,nq,nm);
                    value += ints_->aptei_ab(nm,np,nm,nq);
                }
            });
        }
    }

    // core-core block
    BlockedTensor VFock = ambit::BlockedTensor::build(tensor_type_,"VFock",{"caca","cAcA","aCaC","CACA"});
    VFock.iterate([&](const std::vector<size_t>& i,const std::vector<SpinType>& spin,double& value){
        if ((spin[0] == AlphaSpin) and (spin[1] == AlphaSpin)) value = ints_->aptei_aa(i[0],i[1],i[2],i[3]);
        if ((spin[0] == AlphaSpin) and (spin[1] == BetaSpin) ) value = ints_->aptei_ab(i[0],i[1],i[2],i[3]);
        if ((spin[0] == BetaSpin)  and (spin[1] == BetaSpin) ) value = ints_->aptei_bb(i[0],i[1],i[2],i[3]);
    });
    F_["mn"] += VFock["mvnu"] * Gamma1_["uv"];
    F_["mn"] += VFock["mVnU"] * Gamma1_["UV"];
    F_["MN"] += VFock["vMuN"] * Gamma1_["uv"];
    F_["MN"] += VFock["MVNU"] * Gamma1_["UV"];

    // virtual-virtual block
    VFock = ambit::BlockedTensor::build(tensor_type_,"VFock",{"vava","vAvA","aVaV","VAVA"});
    VFock.iterate([&](const std::vector<size_t>& i,const std::vector<SpinType>& spin,double& value){
        if ((spin[0] == AlphaSpin) and (spin[1] == AlphaSpin)) value = ints_->aptei_aa(i[0],i[1],i[2],i[3]);
        if ((spin[0] == AlphaSpin) and (spin[1] == BetaSpin) ) value = ints_->aptei_ab(i[0],i[1],i[2],i[3]);
        if ((spin[0] == BetaSpin)  and (spin[1] == BetaSpin) ) value = ints_->aptei_bb(i[0],i[1],i[2],i[3]);
    });
    F_["ef"] += VFock["evfu"] * Gamma1_["uv"];
    F_["ef"] += VFock["eVfU"] * Gamma1_["UV"];
    F_["EF"] += VFock["vEuF"] * Gamma1_["uv"];
    F_["EF"] += VFock["EVFU"] * Gamma1_["UV"];

    // off-diagonal and all-active blocks
    F_["ai"] += V_["aviu"] * Gamma1_["uv"];
    F_["ai"] += V_["aViU"] * Gamma1_["UV"];
    F_["AI"] += V_["vAuI"] * Gamma1_["uv"];
    F_["AI"] += V_["AVIU"] * Gamma1_["UV"];

    // obtain diagonal elements of Fock matrix
    F_.iterate([&](const std::vector<size_t>& i,const std::vector<SpinType>& spin,double& value){
        if (spin[0] == AlphaSpin and (i[0] == i[1])){
            Fa_[i[0]] = value;
        }
        if (spin[0] == BetaSpin and (i[0] == i[1])){
            Fb_[i[0]] = value;
        }
    });
}

void DSRG_MRPT3::build_fock_full(){
    // copy one-electron integrals and core part of two-electron integrals
    F_.iterate([&](const std::vector<size_t>& i,const std::vector<SpinType>& spin,double& value){
        if (spin[0] == AlphaSpin) {
            value = ints_->oei_a(i[0],i[1]);
            for(const size_t& nm: acore_mos_) {
                value += ints_->aptei_aa(i[0],nm,i[1],nm);
                value += ints_->aptei_ab(i[0],nm,i[1],nm);
            }
        }
        else {
            value = ints_->oei_b(i[0],i[1]);
            for(const size_t& nm: bcore_mos_) {
                value += ints_->aptei_bb(i[0],nm,i[1],nm);
                value += ints_->aptei_ab(nm,i[0],nm,i[1]);
            }
        }
    });

    // active part of two-electron integrals
    F_["pq"] += V_["pvqu"] * Gamma1_["uv"];
    F_["pq"] += V_["pVqU"] * Gamma1_["UV"];
    F_["PQ"] += V_["vPuQ"] * Gamma1_["uv"];
    F_["PQ"] += V_["PVQU"] * Gamma1_["UV"];

    // obtain diagonal elements of Fock matrix
    F_.iterate([&](const std::vector<size_t>& i,const std::vector<SpinType>& spin,double& value){
        if (spin[0] == AlphaSpin and (i[0] == i[1])){
            Fa_[i[0]] = value;
        }
        if (spin[0] == BetaSpin and (i[0] == i[1])){
            Fb_[i[0]] = value;
        }
    });
}

bool DSRG_MRPT3::check_semicanonical(){
    outfile->Printf("\n    Checking if orbitals are semi-canonicalized ...");

    // zero diagonal elements
    F_.iterate([&](const std::vector<size_t>& i,const std::vector<SpinType>& spin,double& value){
        if (spin[0] == AlphaSpin and (i[0] == i[1])){
            value = 0.0;
        }
        if (spin[0] == BetaSpin and (i[0] == i[1])){
            value = 0.0;
        }
    });

    // off diagonal elements of diagonal blocks
    double Foff_sum = 0.0;
    std::vector<double> Foff;
    for(const auto& block: {"cc", "aa", "vv", "CC", "AA", "VV"}){
        double value = F_.block(block).norm();
        Foff.emplace_back(value);
        Foff_sum += value;
    }

    // add diagonal elements back
    F_.iterate([&](const std::vector<size_t>& i,const std::vector<SpinType>& spin,double& value){
        if (spin[0] == AlphaSpin and (i[0] == i[1])){
            value = Fa_[i[0]];
        }
        if (spin[0] == BetaSpin and (i[0] == i[1])){
            value = Fb_[i[0]];
        }
    });

    bool semi = false;
    double threshold = 0.1 * std::sqrt(options_.get_double("E_CONVERGENCE"));
    if(Foff_sum > threshold){
        std::string sep(3 + 16 * 3, '-');
        outfile->Printf("\n    Warning! Orbitals are not semi-canonicalized!");
        outfile->Printf("\n    Off-Diagonal norms of the core, active, virtual blocks of Fock matrix");
        outfile->Printf("\n       %15s %15s %15s", "core", "active", "virtual");
        outfile->Printf("\n    %s", sep.c_str());
        outfile->Printf("\n    Fa %15.10f %15.10f %15.10f", Foff[0], Foff[1], Foff[2]);
        outfile->Printf("\n    Fb %15.10f %15.10f %15.10f", Foff[3], Foff[4], Foff[5]);
        outfile->Printf("\n    %s\n", sep.c_str());
    }else{
        outfile->Printf("     OK.");
        semi = true;
    }
    return semi;
}

void DSRG_MRPT3::print_summary()
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
        {"source operator", source_},
        {"reference relaxation", relax_ref_}};

    // Print some information
    outfile->Printf("\n\n  ==> Calculation Information <==\n");
    for (auto& str_dim : calculation_info){
        outfile->Printf("\n    %-39s %15d",str_dim.first.c_str(),str_dim.second);
    }
    for (auto& str_dim : calculation_info_double){
        outfile->Printf("\n    %-39s %15.3e",str_dim.first.c_str(),str_dim.second);
    }
    for (auto& str_dim : calculation_info_string){
        outfile->Printf("\n    %-39s %15s",str_dim.first.c_str(),str_dim.second.c_str());
    }

    if(options_.get_bool("MEMORY_SUMMARY")) {
        BTF_->print_memory_info();
    }
    outfile->Flush();
}

void DSRG_MRPT3::cleanup()
{
    dsrg_time_.print_comm_time();
}

double DSRG_MRPT3::compute_energy()
{
    // Compute first-order T2 and T1
    print_h2("First-Order Amplitudes");
    T1_ = BTF_->build(tensor_type_,"T1 Amplitudes",spin_cases({"cp","av"}));
    T2_ = BTF_->build(tensor_type_,"T2 Amplitudes",spin_cases({"chpp","acpp","aavp","aaav"}));
    compute_t2();
    compute_t1();

    // analyze amplitudes
    outfile->Printf("\n    Active Indices: ");
    int c = 0;
    for(const auto& idx: aactv_mos_){
        outfile->Printf("%4zu ", idx);
        if(++c % 10 == 0)
            outfile->Printf("\n    %16c", ' ');
    }
    check_t1();
    check_t2();

    print_h2("Possible Intruders");
    print_intruder("A", lt1a_);
    print_intruder("B", lt1b_);
    print_intruder("AA", lt2aa_);
    print_intruder("AB", lt2ab_);
    print_intruder("BB", lt2bb_);

    // compute energy contributions, note: ordering matters!
    double Ept3_1 = compute_energy_pt3_1();
    double Ept2 = compute_energy_pt2();
    double Ept3_2 = compute_energy_pt3_2();
    double Ept3_3 = compute_energy_pt3_3();
    double Ept3 = Ept3_1 + Ept3_2 + Ept3_3;
    Hbar0_ = Ept3 + Ept2;
    double Etotal = Hbar0_ + Eref_;

    // print energy summary
    std::vector<std::pair<std::string,double>> energy;
    energy.push_back({"E0 (reference)", Eref_});
    energy.push_back({"2nd-order corr. energy", Ept2});
    energy.push_back({"3rd-order corr. energy part 1", Ept3_1});
    energy.push_back({"3rd-order corr. energy part 2", Ept3_2});
    energy.push_back({"3rd-order corr. energy part 3", Ept3_3});
    energy.push_back({"3rd-order corr. energy", Ept3});
    energy.push_back({"DSRG-MRPT3 corr. energy", Hbar0_});
    energy.push_back({"DSRG-MRPT3 total energy", Etotal});

    print_h2("DSRG-MRPT3 Energy Summary");
    for (auto& str_dim : energy){
        outfile->Printf("\n    %-35s = %22.15f",str_dim.first.c_str(),str_dim.second);
    }
    outfile->Printf("\n\n    Notes:");
    outfile->Printf("\n      3rd-order energy part 1: -1.0 / 12.0 * [[[H0th, A1st], A1st], A1st]");
    outfile->Printf("\n      3rd-order energy part 2: 0.5 * [H1st + Hbar1st, A2nd]");
    outfile->Printf("\n      3rd-order energy part 3: 0.5 * [Hbar2nd, A1st]");
    outfile->Printf("\n      Hbar1st = H1st + [H0th, A1st]");
    outfile->Printf("\n      Hbar2nd = 0.5 * [H1st + Hbar1st, A1st] + [H0th, A2nd]");

    Process::environment.globals["CURRENT ENERGY"] = Etotal;

    return Etotal;
}

double DSRG_MRPT3::compute_energy_pt2(){
    print_h2("Computing 2nd-Order Correlation Energy");

    // Compute effective integrals
    renormalize_V();
    renormalize_F();

    // Compute DSRG-MRPT2 correlation energy
    double Ept2 = 0.0;
    Timer t1;
    std::string str = "Computing 2nd-order energy";
    outfile->Printf("\n    %-40s ...", str.c_str());
    H1_T1_C0(F_,T1_,1.0,Ept2);
    H1_T2_C0(F_,T2_,1.0,Ept2);
    H2_T1_C0(V_,T1_,1.0,Ept2);
    H2_T2_C0(V_,T2_,1.0,Ept2);
    outfile->Printf("  Done. Timing %10.3f s", t1.get());

    // relax reference
    if(relax_ref_ != "NONE"){
        Timer t2;
        str = "Computing integrals for ref. relaxation";
        outfile->Printf("\n    %-40s ...", str.c_str());

        BlockedTensor C1 = BTF_->build(tensor_type_,"C1",spin_cases({"aa"}));
        BlockedTensor C2 = BTF_->build(tensor_type_,"C2",spin_cases({"aaaa"}));
        H1_T1_C1(F_,T1_,0.5,C1);
        H1_T2_C1(F_,T2_,0.5,C1);
        H2_T1_C1(V_,T1_,0.5,C1);
        H2_T2_C1(V_,T2_,0.5,C1);
        H1_T2_C2(F_,T2_,0.5,C2);
        H2_T1_C2(V_,T1_,0.5,C2);
        H2_T2_C2(V_,T2_,0.5,C2);

        Hbar1_["uv"] += C1["uv"];
        Hbar1_["uv"] += C1["vu"];
        Hbar1_["UV"] += C1["UV"];
        Hbar1_["UV"] += C1["VU"];
        Hbar2_["uvxy"] += C2["uvxy"];
        Hbar2_["uvxy"] += C2["xyuv"];
        Hbar2_["uVxY"] += C2["uVxY"];
        Hbar2_["uVxY"] += C2["xYuV"];
        Hbar2_["UVXY"] += C2["UVXY"];
        Hbar2_["UVXY"] += C2["XYUV"];

        outfile->Printf("  Done. Timing %10.3f s", t2.get());
    }

    return Ept2;
}

double DSRG_MRPT3::compute_energy_pt3_1(){
    print_h2("Computing 3rd-Order Energy Contribution (1/3)");

    Timer t1;
    std::string str = "Computing 3rd-order energy (1/3)";
    outfile->Printf("\n    %-40s ...", str.c_str());

    // compute -[H0th,A1st] = Delta * T
    // Step 1: create 0th-order Hamiltonian
    BlockedTensor H0th = BTF_->build(tensor_type_,"T1 1st",spin_cases({"cc","aa","vv"}));
    H0th["pq"] = F_["pq"];
    H0th["PQ"] = F_["PQ"];

    // Step 2: compute -[H0th,A1st] and save to C1 and C2
    BlockedTensor C1 = BTF_->build(tensor_type_,"C1",spin_cases({"cp","av","pc","va"}));
    BlockedTensor C2 = BTF_->build(tensor_type_,"C2",spin_cases({"chpp","acpp","aavp","aaav","ppch","ppac","vpaa","avaa"}));
    H1_T1_C1(H0th,T1_,-1.0,C1);
    H1_T2_C1(H0th,T2_,-1.0,C1);
    H1_T2_C2(H0th,T2_,-1.0,C2);

    C1["ai"] = C1["ia"];
    C1["AI"] = C1["IA"];
    C2["abij"] = C2["ijab"];
    C2["aBiJ"] = C2["iJaB"];
    C2["ABIJ"] = C2["IJAB"];

    // compute -[[H0th,A1st],A1st]
    // Step 1: ph and pphh part
    BlockedTensor O1 = BTF_->build(tensor_type_,"O1 PT3 1/3",spin_cases({"pc","va"}));
    BlockedTensor O2 = BTF_->build(tensor_type_,"O2 PT3 1/3",spin_cases({"ppch","ppac","vpaa","avaa"}));
    H1_T1_C1(C1,T1_,1.0,O1);
    H1_T2_C1(C1,T2_,1.0,O1);
    H2_T1_C1(C2,T1_,1.0,O1);
    H2_T2_C1(C2,T2_,1.0,O1);
    H1_T2_C2(C1,T2_,1.0,O2);
    H2_T1_C2(C2,T1_,1.0,O2);
    H2_T2_C2(C2,T2_,1.0,O2);

    // Step 2: hp and hhpp part
    BlockedTensor temp1 = BTF_->build(tensor_type_,"temp1 pt3 1/3",spin_cases({"cp","av"}));
    BlockedTensor temp2 = BTF_->build(tensor_type_,"temp2 pt3 1/3",spin_cases({"chpp","acpp","aavp","aaav"}));
    H1_T1_C1(C1,T1_,1.0,temp1);
    H1_T2_C1(C1,T2_,1.0,temp1);
    H2_T1_C1(C2,T1_,1.0,temp1);
    H2_T2_C1(C2,T2_,1.0,temp1);
    H1_T2_C2(C1,T2_,1.0,temp2);
    H2_T1_C2(C2,T1_,1.0,temp2);
    H2_T2_C2(C2,T2_,1.0,temp2);

    // Step 3: add hp and hhpp to O1 and O2
    O1["ai"] += temp1["ia"];
    O1["AI"] += temp1["IA"];
    O2["abij"] += temp2["ijab"];
    O2["aBiJ"] += temp2["iJaB"];
    O2["ABIJ"] += temp2["IJAB"];

    // compute -1.0 / 12.0 * [[[H0th,A1st],A1st],A1st]
    double Ereturn = 0.0;
    double factor  = 1.0 / 6.0;
    H1_T1_C0(O1,T1_,factor,Ereturn);
    H1_T2_C0(O1,T2_,factor,Ereturn);
    H2_T1_C0(O2,T1_,factor,Ereturn);
    H2_T2_C0(O2,T2_,factor,Ereturn);

    outfile->Printf("  Done. Timing %10.3f s", t1.get());

    if(relax_ref_ != "NONE"){
        Timer t2;
        str = "Computing integrals for ref. relaxation";
        outfile->Printf("\n    %-40s ...", str.c_str());

        factor = 1.0 / 12.0;
        C1 = BTF_->build(tensor_type_,"C1",spin_cases({"aa"}));
        C2 = BTF_->build(tensor_type_,"C2",spin_cases({"aaaa"}));
        H1_T1_C1(O1,T1_,factor,C1);
        H1_T2_C1(O1,T2_,factor,C1);
        H2_T1_C1(O2,T1_,factor,C1);
        H2_T2_C1(O2,T2_,factor,C1);
        H1_T2_C2(O1,T2_,factor,C2);
        H2_T1_C2(O2,T1_,factor,C2);
        H2_T2_C2(O2,T2_,factor,C2);

        Hbar1_["uv"] += C1["uv"];
        Hbar1_["uv"] += C1["vu"];
        Hbar1_["UV"] += C1["UV"];
        Hbar1_["UV"] += C1["VU"];
        Hbar2_["uvxy"] += C2["uvxy"];
        Hbar2_["uvxy"] += C2["xyuv"];
        Hbar2_["uVxY"] += C2["uVxY"];
        Hbar2_["uVxY"] += C2["xYuV"];
        Hbar2_["UVXY"] += C2["UVXY"];
        Hbar2_["UVXY"] += C2["XYUV"];

        outfile->Printf("  Done. Timing %10.3f s", t2.get());
    }

    return Ereturn;
}

double DSRG_MRPT3::compute_energy_pt3_2(){
    print_h2("Computing 3rd-Order Energy Contribution (2/3)");

    Timer t1;
    std::string str = "Preparing 2nd-order amplitudes";
    outfile->Printf("\n    %-40s ...", str.c_str());

    // compute 2nd-order amplitudes
    // Step 1: compute 0.5 * [H1st + Hbar1st, A1st]
    //     a) one-body H1st + Hbar1st
    BlockedTensor O1 = BTF_->build(tensor_type_,"O1 pt3 2/3",spin_cases({"pc","va","cp","av"}));
    O1["ai"] = F_["ai"];
    O1["ia"] = F_["ai"];
    O1["AI"] = F_["AI"];
    O1["IA"] = F_["AI"];

    //     b) keep a copy of two-body H1st + Hbar1st
    BlockedTensor O2 = BTF_->build(tensor_type_,"O2 pt3 2/3",spin_cases({"ppch","ppac","vpaa","avaa"}));
    O2["abij"] = V_["abij"];
    O2["aBiJ"] = V_["aBiJ"];
    O2["ABIJ"] = V_["ABIJ"];

    //     c) compute contraction from one-body term
    F_.zero();
    V_.zero();
    H1_T1_C1(O1,T1_,0.5,F_);
    H1_T2_C1(O1,T2_,0.5,F_);
    H1_T2_C2(O1,T2_,0.5,V_);

    BlockedTensor HP1 = BTF_->build(tensor_type_,"HP2 pt3 2/3",spin_cases({"cp","av"}));
    BlockedTensor HP2 = BTF_->build(tensor_type_,"HP2 pt3 2/3",spin_cases({"chpp","acpp","aavp","aaav"}));
    H1_T1_C1(O1,T1_,0.5,HP1);
    H1_T2_C1(O1,T2_,0.5,HP1);
    H1_T2_C2(O1,T2_,0.5,HP2);

    F_["ai"] += HP1["ia"];
    F_["AI"] += HP1["IA"];
    V_["abij"] += HP2["ijab"];
    V_["aBiJ"] += HP2["iJaB"];
    V_["ABIJ"] += HP2["IJAB"];

    //     d) compute contraction in batches of spin cases
    for(const std::string& block: {"gggg", "gGgG", "GGGG"}){
        // spin cases: 0 -> AA; 1 -> AB; 2 -> BB
        std::string abij {"abij"};
        std::string ijab {"ijab"};

        if (isupper(block[1])) {
            abij = "aBiJ";
            ijab = "iJaB";
        }
        if (isupper(block[0])) {
            abij = "ABIJ";
            ijab = "IJAB";
        }

        BlockedTensor C2 = BTF_->build(tensor_type_,"C2 pt3 2/3",{block});
        C2.iterate([&](const std::vector<size_t>& i,const std::vector<SpinType>& spin,double& value){
            if ((spin[0] == AlphaSpin) and (spin[1] == AlphaSpin)) value = 2.0 * ints_->aptei_aa(i[0],i[1],i[2],i[3]);
            if ((spin[0] == AlphaSpin) and (spin[1] == BetaSpin) ) value = 2.0 * ints_->aptei_ab(i[0],i[1],i[2],i[3]);
            if ((spin[0] == BetaSpin)  and (spin[1] == BetaSpin) ) value = 2.0 * ints_->aptei_bb(i[0],i[1],i[2],i[3]);
        });
        C2[abij] = O2[abij];
        C2[ijab] = O2[abij];

        // pphh part
        H2_T1_C1(C2,T1_,0.5,F_);
        H2_T2_C1(C2,T2_,0.5,F_);
        H2_T1_C2(C2,T1_,0.5,V_);
        H2_T2_C2(C2,T2_,0.5,V_);

        // hhpp part
        HP1.zero();
        HP2.zero();
        H2_T1_C1(C2,T1_,0.5,HP1);
        H2_T2_C1(C2,T2_,0.5,HP1);
        H2_T1_C2(C2,T1_,0.5,HP2);
        H2_T2_C2(C2,T2_,0.5,HP2);

        F_["ai"] += HP1["ia"];
        F_["AI"] += HP1["IA"];
        V_["abij"] += HP2["ijab"];
        V_["aBiJ"] += HP2["iJaB"];
        V_["ABIJ"] += HP2["IJAB"];
    }
    outfile->Printf("  Done. Timing %10.3f s", t1.get());

    // Step 2: compute amplitdes
    //     a) first save 1st-order amplitudes for later use
    T1_1st_ = BTF_->build(tensor_type_,"T1 1st",spin_cases({"cp","av"}));
    T2_1st_ = BTF_->build(tensor_type_,"T2 1st",spin_cases({"chpp","acpp","aavp","aaav"}));
    T1_1st_["ia"] = T1_["ia"];
    T1_1st_["IA"] = T1_["IA"];
    T2_1st_["ijab"] = T2_["ijab"];
    T2_1st_["iJaB"] = T2_["iJaB"];
    T2_1st_["IJAB"] = T2_["IJAB"];

    //     b) compute 2nd-order amplitdes
    compute_t2();
    compute_t1();

    // compute energy from 0.5 * [[H1st + Hbar1st, A1st], A2nd]
    Timer t2;
    str = "Computing 3rd-order energy (2/3)";
    outfile->Printf("\n    %-40s ...", str.c_str());
    double Ereturn = 0.0;
    H1_T1_C0(O1,T1_,1.0,Ereturn);
    H1_T2_C0(O1,T2_,1.0,Ereturn);
    H2_T1_C0(O2,T1_,1.0,Ereturn);
    H2_T2_C0(O2,T2_,1.0,Ereturn);
    outfile->Printf("  Done. Timing %10.3f s", t2.get());

    if(relax_ref_ != "NONE"){
        Timer t3;
        str = "Computing integrals for ref. relaxation";
        outfile->Printf("\n    %-40s ...", str.c_str());

        double factor = 0.5;
        BlockedTensor A1 = BTF_->build(tensor_type_,"A1",spin_cases({"aa"}));
        BlockedTensor A2 = BTF_->build(tensor_type_,"A2",spin_cases({"aaaa"}));
        H1_T1_C1(O1,T1_,factor,A1);
        H1_T2_C1(O1,T2_,factor,A1);
        H2_T1_C1(O2,T1_,factor,A1);
        H2_T2_C1(O2,T2_,factor,A1);
        H1_T2_C2(O1,T2_,factor,A2);
        H2_T1_C2(O2,T1_,factor,A2);
        H2_T2_C2(O2,T2_,factor,A2);

        Hbar1_["uv"] += A1["uv"];
        Hbar1_["uv"] += A1["vu"];
        Hbar1_["UV"] += A1["UV"];
        Hbar1_["UV"] += A1["VU"];
        Hbar2_["uvxy"] += A2["uvxy"];
        Hbar2_["uvxy"] += A2["xyuv"];
        Hbar2_["uVxY"] += A2["uVxY"];
        Hbar2_["uVxY"] += A2["xYuV"];
        Hbar2_["UVXY"] += A2["UVXY"];
        Hbar2_["UVXY"] += A2["XYUV"];

        outfile->Printf("  Done. Timing %10.3f s", t3.get());
    }

    // analyze amplitudes
    print_h2("Second-Order Amplitudes Summary");
    outfile->Printf("\n    Active Indices: ");
    int c = 0;
    for(const auto& idx: aactv_mos_){
        outfile->Printf("%4zu ", idx);
        if(++c % 10 == 0)
            outfile->Printf("\n    %16c", ' ');
    }
    check_t1();
    check_t2();

    return Ereturn;
}

double DSRG_MRPT3::compute_energy_pt3_3(){
    print_h2("Computing 3rd-Order Energy Contribution (3/3)");

    // scale F and V by exponential delta
    renormalize_F(false);
    renormalize_V(false);

    // compute energy of 0.5 * [Hbar2nd, A1st]
    double Ereturn = 0.0;
    Timer t1;
    std::string str = "Computing 3rd-order energy (3/3)";
    outfile->Printf("\n    %-40s ...", str.c_str());
    H1_T1_C0(F_,T1_1st_,1.0,Ereturn);
    H1_T2_C0(F_,T2_1st_,1.0,Ereturn);
    H2_T1_C0(V_,T1_1st_,1.0,Ereturn);
    H2_T2_C0(V_,T2_1st_,1.0,Ereturn);
    outfile->Printf("  Done. Timing %10.3f s", t1.get());

    // relax reference
    if(relax_ref_ != "NONE"){
        Timer t2;
        str = "Computing integrals for ref. relaxation";
        outfile->Printf("\n    %-40s ...", str.c_str());

        BlockedTensor C1 = BTF_->build(tensor_type_,"C1",spin_cases({"aa"}));
        BlockedTensor C2 = BTF_->build(tensor_type_,"C2",spin_cases({"aaaa"}));
        H1_T1_C1(F_,T1_1st_,0.5,C1);
        H1_T2_C1(F_,T2_1st_,0.5,C1);
        H2_T1_C1(V_,T1_1st_,0.5,C1);
        H2_T2_C1(V_,T2_1st_,0.5,C1);
        H1_T2_C2(F_,T2_1st_,0.5,C2);
        H2_T1_C2(V_,T1_1st_,0.5,C2);
        H2_T2_C2(V_,T2_1st_,0.5,C2);

        Hbar1_["uv"] += C1["uv"];
        Hbar1_["uv"] += C1["vu"];
        Hbar1_["UV"] += C1["UV"];
        Hbar1_["UV"] += C1["VU"];
        Hbar2_["uvxy"] += C2["uvxy"];
        Hbar2_["uvxy"] += C2["xyuv"];
        Hbar2_["uVxY"] += C2["uVxY"];
        Hbar2_["uVxY"] += C2["xYuV"];
        Hbar2_["UVXY"] += C2["UVXY"];
        Hbar2_["UVXY"] += C2["XYUV"];

        outfile->Printf("  Done. Timing %10.3f s", t2.get());
    }

    return Ereturn;
}

void DSRG_MRPT3::compute_t2()
{
    Timer timer;
    std::string str = "Computing T2 amplitudes";
    outfile->Printf("\n    %-40s ...", str.c_str());

    T2_["ijab"] = V_["abij"];
    T2_["iJaB"] = V_["aBiJ"];
    T2_["IJAB"] = V_["ABIJ"];

    // transform to semi-canonical basis
    if(!semi_canonical_){
        BlockedTensor tempT2 = ambit::BlockedTensor::build(tensor_type_,"Temp T2",spin_cases({"hhpp"}));
        tempT2["klab"] = U_["ki"] * U_["lj"] * T2_["ijab"];
        tempT2["kLaB"] = U_["ki"] * U_["LJ"] * T2_["iJaB"];
        tempT2["KLAB"] = U_["KI"] * U_["LJ"] * T2_["IJAB"];
        T2_["ijcd"] = tempT2["ijab"] * U_["db"] * U_["ca"];
        T2_["iJcD"] = tempT2["iJaB"] * U_["DB"] * U_["ca"];
        T2_["IJCD"] = tempT2["IJAB"] * U_["DB"] * U_["CA"];
    }

    T2_.iterate([&](const std::vector<size_t>& i,const std::vector<SpinType>& spin,double& value){
        if (fabs(value) > 1.0e-12){
            if ((spin[0] == AlphaSpin) and (spin[1] == AlphaSpin)){
                value *= dsrg_source_->compute_renormalized_denominator(Fa_[i[0]] + Fa_[i[1]] - Fa_[i[2]] - Fa_[i[3]]);
            }
            else if ((spin[0] == AlphaSpin) and (spin[1] == BetaSpin) ){
                value *= dsrg_source_->compute_renormalized_denominator(Fa_[i[0]] + Fb_[i[1]] - Fa_[i[2]] - Fb_[i[3]]);
            }
            else if ((spin[0] == BetaSpin)  and (spin[1] == BetaSpin) ){
                value *= dsrg_source_->compute_renormalized_denominator(Fb_[i[0]] + Fb_[i[1]] - Fb_[i[2]] - Fb_[i[3]]);
            }
        } else {
            value = 0.0;
        }
    });

    // transform back to non-canonical basis
    if(!semi_canonical_){
        BlockedTensor tempT2 = ambit::BlockedTensor::build(tensor_type_,"Temp T2",spin_cases({"hhpp"}));
        tempT2["klab"] = U_["ik"] * U_["jl"] * T2_["ijab"];
        tempT2["kLaB"] = U_["ik"] * U_["JL"] * T2_["iJaB"];
        tempT2["KLAB"] = U_["IK"] * U_["JL"] * T2_["IJAB"];
        T2_["ijcd"] = tempT2["ijab"] * U_["bd"] * U_["ac"];
        T2_["iJcD"] = tempT2["iJaB"] * U_["BD"] * U_["ac"];
        T2_["IJCD"] = tempT2["IJAB"] * U_["BD"] * U_["AC"];
    }

    // no internal amplitudes, otherwise need to zero them

    outfile->Printf("  Done. Timing %10.3f s", timer.get());
}

void DSRG_MRPT3::compute_t1()
{
    Timer timer;
    std::string str = "Computing T1 amplitudes";
    outfile->Printf("\n    %-40s ...", str.c_str());

    BlockedTensor temp = BTF_->build(tensor_type_,"temp",spin_cases({"aa"}));
    temp["xu"] = Gamma1_["xu"];
    temp["XU"] = Gamma1_["XU"];
    // transform to semi-canonical basis
    if(!semi_canonical_){
        BlockedTensor tempG = ambit::BlockedTensor::build(tensor_type_,"Temp Gamma",spin_cases({"aa"}));
        tempG["uv"] = U_["ux"] * temp["xy"] * U_["vy"];
        tempG["UV"] = U_["UX"] * temp["XY"] * U_["VY"];
        temp["uv"] = tempG["uv"];
        temp["UV"] = tempG["UV"];
    }
    // scale by delta
    temp.iterate([&](const std::vector<size_t>& i,const std::vector<SpinType>& spin,double& value){
        if (spin[0] == AlphaSpin){
            value *= Fa_[i[0]] - Fa_[i[1]];
        }else{
            value *= Fb_[i[0]] - Fb_[i[1]];
        }
    });
    // transform back to non-canonical basis
    if(!semi_canonical_){
        BlockedTensor tempG = ambit::BlockedTensor::build(tensor_type_,"Temp Gamma",spin_cases({"aa"}));
        tempG["uv"] = U_["xu"] * temp["xy"] * U_["yv"];
        tempG["UV"] = U_["XU"] * temp["XY"] * U_["YV"];
        temp["uv"] = tempG["uv"];
        temp["UV"] = tempG["UV"];
    }

    T1_["ia"]  = F_["ai"];
    T1_["ia"] += temp["xu"] * T2_["iuax"];
    T1_["ia"] += temp["XU"] * T2_["iUaX"];

    T1_["IA"]  = F_["AI"];
    T1_["IA"] += temp["xu"] * T2_["uIxA"];
    T1_["IA"] += temp["XU"] * T2_["IUAX"];

    // transform to semi-canonical basis
    if(!semi_canonical_){
        BlockedTensor tempT1 = ambit::BlockedTensor::build(tensor_type_,"Temp T1",spin_cases({"hp"}));
        tempT1["jb"] = U_["ji"] * T1_["ia"] * U_["ba"];
        tempT1["JB"] = U_["JI"] * T1_["IA"] * U_["BA"];
        T1_["ia"] = tempT1["ia"];
        T1_["IA"] = tempT1["IA"];
    }

    T1_.iterate([&](const std::vector<size_t>& i,const std::vector<SpinType>& spin,double& value){
        if(fabs(value) > 1.0e-12){
            if (spin[0]  == AlphaSpin){
                value *= dsrg_source_->compute_renormalized_denominator(Fa_[i[0]] - Fa_[i[1]]);
            }else{
                value *= dsrg_source_->compute_renormalized_denominator(Fb_[i[0]] - Fb_[i[1]]);
            }
        } else {
            value = 0.0;
        }
    });

    // transform back to non-canonical basis
    if(!semi_canonical_){
        BlockedTensor tempT1 = ambit::BlockedTensor::build(tensor_type_,"Temp T1",spin_cases({"hp"}));
        tempT1["jb"] = U_["ij"] * T1_["ia"] * U_["ab"];
        tempT1["JB"] = U_["IJ"] * T1_["IA"] * U_["AB"];
        T1_["ia"] = tempT1["ia"];
        T1_["IA"] = tempT1["IA"];
    }

    // no internal amplitudes, otherwise need to zero them

    outfile->Printf("  Done. Timing %10.3f s", timer.get());
}


void DSRG_MRPT3::renormalize_V(const bool& plusone)
{
    Timer timer;
    std::string str = "Renormalizing two-electron integrals";
    outfile->Printf("\n    %-40s ...", str.c_str());

    // transform to semi-canonical basis
    if(!semi_canonical_){
        BlockedTensor tempV = ambit::BlockedTensor::build(tensor_type_,"Temp V",spin_cases({"pphh"}));
        tempV["cdij"] = U_["ca"] * U_["db"] * V_["abij"];
        tempV["cDiJ"] = U_["ca"] * U_["DB"] * V_["aBiJ"];
        tempV["CDIJ"] = U_["CA"] * U_["DB"] * V_["ABIJ"];
        V_["abkl"] = tempV["abij"] * U_["lj"] * U_["ki"];
        V_["aBkL"] = tempV["aBiJ"] * U_["LJ"] * U_["ki"];
        V_["ABKL"] = tempV["ABIJ"] * U_["LJ"] * U_["KI"];
    }

    if (plusone) {
        V_.iterate([&](const std::vector<size_t>& i,const std::vector<SpinType>& spin,double& value){
            if (fabs(value) > 1.0e-12){
                if ((spin[0] == AlphaSpin) and (spin[1] == AlphaSpin)){
                    value *= 1.0 + dsrg_source_->compute_renormalized(Fa_[i[0]] + Fa_[i[1]] - Fa_[i[2]] - Fa_[i[3]]);
                }
                else if ((spin[0] == AlphaSpin) and (spin[1] == BetaSpin) ){
                    value *= 1.0 + dsrg_source_->compute_renormalized(Fa_[i[0]] + Fb_[i[1]] - Fa_[i[2]] - Fb_[i[3]]);
                }
                else if ((spin[0] == BetaSpin)  and (spin[1] == BetaSpin) ){
                    value *= 1.0 + dsrg_source_->compute_renormalized(Fb_[i[0]] + Fb_[i[1]] - Fb_[i[2]] - Fb_[i[3]]);
                }
            } else {
                value = 0.0;
            }
        });
    } else {
        V_.iterate([&](const std::vector<size_t>& i,const std::vector<SpinType>& spin,double& value){
            if (fabs(value) > 1.0e-12){
                if ((spin[0] == AlphaSpin) and (spin[1] == AlphaSpin)){
                    value *= dsrg_source_->compute_renormalized(Fa_[i[0]] + Fa_[i[1]] - Fa_[i[2]] - Fa_[i[3]]);
                }
                else if ((spin[0] == AlphaSpin) and (spin[1] == BetaSpin) ){
                    value *= dsrg_source_->compute_renormalized(Fa_[i[0]] + Fb_[i[1]] - Fa_[i[2]] - Fb_[i[3]]);
                }
                else if ((spin[0] == BetaSpin)  and (spin[1] == BetaSpin) ){
                    value *= dsrg_source_->compute_renormalized(Fb_[i[0]] + Fb_[i[1]] - Fb_[i[2]] - Fb_[i[3]]);
                }
            } else {
                value = 0.0;
            }
        });
    }

    // transform back to non-canonical basis
    if(!semi_canonical_){
        BlockedTensor tempV = ambit::BlockedTensor::build(tensor_type_,"Temp V",spin_cases({"pphh"}));
        tempV["cdij"] = U_["ac"] * U_["bd"] * V_["abij"];
        tempV["cDiJ"] = U_["ac"] * U_["BD"] * V_["aBiJ"];
        tempV["CDIJ"] = U_["AC"] * U_["BD"] * V_["ABIJ"];
        V_["abkl"] = tempV["abij"] * U_["jl"] * U_["ik"];
        V_["aBkL"] = tempV["aBiJ"] * U_["JL"] * U_["ik"];
        V_["ABKL"] = tempV["ABIJ"] * U_["JL"] * U_["IK"];
    }

    outfile->Printf("  Done. Timing %10.3f s", timer.get());
}

void DSRG_MRPT3::renormalize_F(const bool& plusone)
{
    Timer timer;
    std::string str = "Renormalizing Fock matrix elements";
    outfile->Printf("\n    %-40s ...", str.c_str());

    BlockedTensor temp = BTF_->build(tensor_type_,"temp",spin_cases({"aa"}));
    temp["xu"] = Gamma1_["xu"];
    temp["XU"] = Gamma1_["XU"];
    // transform to semi-canonical basis
    if(!semi_canonical_){
        BlockedTensor tempG = ambit::BlockedTensor::build(tensor_type_,"Temp Gamma",spin_cases({"aa"}));
        tempG["uv"] = U_["ux"] * temp["xy"] * U_["vy"];
        tempG["UV"] = U_["UX"] * temp["XY"] * U_["VY"];
        temp["uv"] = tempG["uv"];
        temp["UV"] = tempG["UV"];
    }
    // scale by delta
    temp.iterate([&](const std::vector<size_t>& i,const std::vector<SpinType>& spin,double& value){
        if (spin[0] == AlphaSpin){
            value *= Fa_[i[0]] - Fa_[i[1]];
        }else{
            value *= Fb_[i[0]] - Fb_[i[1]];
        }
    });
    // transform back to non-canonical basis
    if(!semi_canonical_){
        BlockedTensor tempG = ambit::BlockedTensor::build(tensor_type_,"Temp Gamma",spin_cases({"aa"}));
        tempG["uv"] = U_["xu"] * temp["xy"] * U_["yv"];
        tempG["UV"] = U_["XU"] * temp["XY"] * U_["YV"];
        temp["uv"] = tempG["uv"];
        temp["UV"] = tempG["UV"];
    }

    BlockedTensor sum = ambit::BlockedTensor::build(tensor_type_,"Temp sum",spin_cases({"ph"}));
    sum["ai"]  = F_["ai"];
    sum["ai"] += temp["xu"] * T2_["iuax"];
    sum["ai"] += temp["XU"] * T2_["iUaX"];

    sum["AI"]  = F_["AI"];
    sum["AI"] += temp["xu"] * T2_["uIxA"];
    sum["AI"] += temp["XU"] * T2_["IUAX"];

    // transform to semi-canonical basis
    if(!semi_canonical_){
        BlockedTensor tempF = ambit::BlockedTensor::build(tensor_type_,"Temp F",spin_cases({"ph"}));
        tempF["bj"] = U_["ba"] * sum["ai"] * U_["ji"];
        tempF["BJ"] = U_["BA"] * sum["AI"] * U_["JI"];
        sum["ai"] = tempF["ai"];
        sum["AI"] = tempF["AI"];
    }

    sum.iterate([&](const std::vector<size_t>& i,const std::vector<SpinType>& spin,double& value){
        if(fabs(value) > 1.0e-12){
            if (spin[0]  == AlphaSpin){
                value *= dsrg_source_->compute_renormalized(Fa_[i[0]] - Fa_[i[1]]);
            }else{
                value *= dsrg_source_->compute_renormalized(Fb_[i[0]] - Fb_[i[1]]);
            }
        } else {
            value = 0.0;
        }
    });

    // transform back to non-canonical basis
    if(!semi_canonical_){
        BlockedTensor tempF = ambit::BlockedTensor::build(tensor_type_,"Temp F",spin_cases({"ph"}));
        tempF["bj"] = U_["ab"] * sum["ai"] * U_["ij"];
        tempF["BJ"] = U_["AB"] * sum["AI"] * U_["IJ"];
        sum["ai"] = tempF["ai"];
        sum["AI"] = tempF["AI"];
    }

    // add to original Fock
    if (plusone){
        F_["ai"] += sum["ai"];
        F_["AI"] += sum["AI"];
    } else {
        F_["ai"] = sum["ai"];
        F_["AI"] = sum["AI"];
    }

    outfile->Printf("  Done. Timing %10.3f s", timer.get());
}

double DSRG_MRPT3::compute_energy_relaxed(){
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
    auto nelec_actv = nelec - 2 * mo_space_info_->size("FROZEN_DOCC") - 2 * acore_mos_.size();
    auto na = (nelec_actv + ms) / 2;
    auto nb =  nelec_actv - na;

    // reference relaxation
    double Edsrg = 0.0, Erelax = 0.0;

    // only relax once, otherwise we have to store another
    if(relax_ref_ != "NONE"){
        // compute energy with fixed ref.
        Edsrg = compute_energy();

        // transfer integrals
        transfer_integrals();

        // diagonalize the Hamiltonian
        FCISolver fcisolver(active_dim,acore_mos_,aactv_mos_,na,nb,multi,options_.get_int("ROOT_SYM"),ints_, mo_space_info_,
                                             options_.get_int("NTRIAL_PER_ROOT"),print_, options_);
        fcisolver.set_max_rdm_level(1);
        fcisolver.set_fci_iterations(options_.get_int("FCI_ITERATIONS"));
        fcisolver.set_collapse_per_root(options_.get_int("DAVIDSON_COLLAPSE_PER_ROOT"));
        fcisolver.set_subspace_per_root(options_.get_int("DAVIDSON_COLLAPSE_PER_ROOT"));
        Erelax = fcisolver.compute_energy();

        // printing
        print_h2("MRDSRG Energy Summary");
        outfile->Printf("\n    %-35s = %22.15f", "MRDSRG Total Energy (fixed)", Edsrg);
        outfile->Printf("\n    %-35s = %22.15f", "MRDSRG Total Energy (relaxed)", Erelax);
        outfile->Printf("\n");
    }

    Process::environment.globals["CURRENT ENERGY"] = Erelax;
    return Erelax;
}

void DSRG_MRPT3::transfer_integrals(){
    // printing
    print_h2("De-Normal-Order the DSRG Transformed Hamiltonian");

    // compute scalar term (all active only)
    Timer t_scalar;
    std::string str = "Computing the scalar term";
    outfile->Printf("\n    %-40s ...", str.c_str());
    double scalar0 = Eref_ + Hbar0_ - molecule_->nuclear_repulsion_energy()
            - ints_->frozen_core_energy();

    // scalar from Hbar1
    double scalar1 = 0.0;
    scalar1 -= Hbar1_["vu"] * Gamma1_["uv"];
    scalar1 -= Hbar1_["VU"] * Gamma1_["UV"];

    // scalar from Hbar2
    double scalar2 = 0.0;
    scalar2 += 0.5 * Gamma1_["uv"] * Hbar2_["vyux"] * Gamma1_["xy"];
    scalar2 += 0.5 * Gamma1_["UV"] * Hbar2_["VYUX"] * Gamma1_["XY"];
    scalar2 += Gamma1_["uv"] * Hbar2_["vYuX"] * Gamma1_["XY"];

    scalar2 -= 0.25 * Hbar2_["xyuv"] * Lambda2_["uvxy"];
    scalar2 -= 0.25 * Hbar2_["XYUV"] * Lambda2_["UVXY"];
    scalar2 -= Hbar2_["xYuV"] * Lambda2_["uVxY"];

    double scalar = scalar0 + scalar1 + scalar2;
    outfile->Printf("  Done. Timing %10.3f s", t_scalar.get());

    // compute one-body term
    Timer t_one;
    str = "Computing the one-body term";
    outfile->Printf("\n    %-40s ...", str.c_str());
    BlockedTensor temp1 = BTF_->build(tensor_type_,"temp1",spin_cases({"aa"}));
    temp1["uv"]  = Hbar1_["uv"];
    temp1["UV"]  = Hbar1_["UV"];
    temp1["uv"] -= Hbar2_["uxvy"] * Gamma1_["yx"];
    temp1["uv"] -= Hbar2_["uXvY"] * Gamma1_["YX"];
    temp1["UV"] -= Hbar2_["xUyV"] * Gamma1_["yx"];
    temp1["UV"] -= Hbar2_["UXVY"] * Gamma1_["YX"];
    outfile->Printf("  Done. Timing %10.3f s", t_one.get());

    // update integrals
    Timer t_int;
    str = "Updating integrals";
    outfile->Printf("\n    %-40s ...", str.c_str());
    ints_->set_scalar(scalar);

    //   a) zero hole integrals
    std::vector<size_t> hole_mos = acore_mos_;
    hole_mos.insert(hole_mos.end(),aactv_mos_.begin(),aactv_mos_.end());
    for(const size_t& i: hole_mos){
        for(const size_t& j: hole_mos){
            ints_->set_oei(i,j,0.0,true);
            ints_->set_oei(i,j,0.0,false);
            for(const size_t& k: hole_mos){
                for(const size_t& l: hole_mos){
                    ints_->set_tei(i,j,k,l,0.0,true,true);
                    ints_->set_tei(i,j,k,l,0.0,true,false);
                    ints_->set_tei(i,j,k,l,0.0,false,false);
                }
            }
        }
    }

    //   b) copy all active part
    temp1.citerate([&](const std::vector<size_t>& i,const std::vector<SpinType>& spin,const double& value){
        if (spin[0] == AlphaSpin){
            ints_->set_oei(i[0],i[1],value,true);
        }else{
            ints_->set_oei(i[0],i[1],value,false);
        }
    });

    BlockedTensor temp2 = BTF_->build(tensor_type_,"temp2",spin_cases({"aaaa"}));
    temp2["uvxy"] = Hbar2_["uvxy"];
    temp2["uVxY"] = Hbar2_["uVxY"];
    temp2["UVXY"] = Hbar2_["UVXY"];
    temp2.citerate([&](const std::vector<size_t>& i,const std::vector<SpinType>& spin,const double& value){
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
    outfile->Printf("\n    %-35s = %22.15f", "Scalar0", scalar0);
    outfile->Printf("\n    %-35s = %22.15f", "Scalar1", scalar1);
    outfile->Printf("\n    %-35s = %22.15f", "Scalar2", scalar2);
    outfile->Printf("\n    %-35s = %22.15f", "Total Scalar W/O Frozen-Core", scalar);
    outfile->Printf("\n    %-35s = %22.15f", "Total Scalar W/  Frozen-Core", scalar_include_fc);

    // test if de-normal-ordering is correct
    print_h2("Test De-Normal-Ordered Hamiltonian");
    double Etest = scalar_include_fc + molecule_->nuclear_repulsion_energy();

    double Etest1 = 0.0;
    Etest1 += temp1["uv"] * Gamma1_["vu"];
    Etest1 += temp1["UV"] * Gamma1_["VU"];

    Etest1 += Hbar1_["uv"] * Gamma1_["vu"];
    Etest1 += Hbar1_["UV"] * Gamma1_["VU"];
    Etest1 *= 0.5;

    double Etest2 = 0.0;
    Etest2 += 0.25 * Hbar2_["uvxy"] * Lambda2_["xyuv"];
    Etest2 += 0.25 * Hbar2_["UVXY"] * Lambda2_["XYUV"];
    Etest2 += Hbar2_["uVxY"] * Lambda2_["xYuV"];

    Etest += Etest1 + Etest2;
    outfile->Printf("\n    %-35s = %22.15f", "One-Body Energy (after)", Etest1);
    outfile->Printf("\n    %-35s = %22.15f", "Two-Body Energy (after)", Etest2);
    outfile->Printf("\n    %-35s = %22.15f", "Total Energy (after)", Etest);
    outfile->Printf("\n    %-35s = %22.15f", "Total Energy (before)", Eref_ + Hbar0_);

    if(fabs(Etest - Eref_ - Hbar0_) > 100.0 * options_.get_double("E_CONVERGENCE")){
        throw PSIEXCEPTION("De-normal-odering failed.");
    }else{
        ints_->update_integrals(false);
    }
}

void DSRG_MRPT3::H1_T1_C0(BlockedTensor& H1, BlockedTensor& T1, const double& alpha, double& C0){
    Timer timer;

    double E = 0.0;
    E += H1["em"] * T1["me"];
    E += H1["ex"] * T1["ye"] * Gamma1_["xy"];
    E += H1["xm"] * T1["my"] * Eta1_["yx"];

    E += H1["EM"] * T1["ME"];
    E += H1["EX"] * T1["YE"] * Gamma1_["XY"];
    E += H1["XM"] * T1["MY"] * Eta1_["YX"];

    E *= alpha;
    C0 += E;

    if(print_ > 2){
        outfile->Printf("\n    Time for [H1, T1] -> C0 : %12.3f",timer.get());
    }
    dsrg_time_.add("110",timer.get());
}

void DSRG_MRPT3::H1_T2_C0(BlockedTensor& H1, BlockedTensor& T2, const double& alpha, double& C0){
    Timer timer;
    BlockedTensor temp;
    double E = 0.0;

    temp = ambit::BlockedTensor::build(tensor_type_,"temp",{"aaaa"});
    temp["uvxy"] += H1["ex"] * T2["uvey"];
    temp["uvxy"] -= H1["vm"] * T2["umxy"];
    E += 0.5 * temp["uvxy"] * Lambda2_["xyuv"];

    temp = ambit::BlockedTensor::build(tensor_type_,"temp",{"AAAA"});
    temp["UVXY"] += H1["EX"] * T2["UVEY"];
    temp["UVXY"] -= H1["VM"] * T2["UMXY"];
    E += 0.5 * temp["UVXY"] * Lambda2_["XYUV"];

    temp = ambit::BlockedTensor::build(tensor_type_,"temp",{"aAaA"});
    temp["uVxY"] += H1["ex"] * T2["uVeY"];
    temp["uVxY"] += H1["EY"] * T2["uVxE"];
    temp["uVxY"] -= H1["VM"] * T2["uMxY"];
    temp["uVxY"] -= H1["um"] * T2["mVxY"];
    E += temp["uVxY"] * Lambda2_["xYuV"];

    E  *= alpha;
    C0 += E;

    if(print_ > 2){
        outfile->Printf("\n    Time for [H1, T2] -> C0 : %12.3f",timer.get());
    }
    dsrg_time_.add("120",timer.get());
}

void DSRG_MRPT3::H2_T1_C0(BlockedTensor& H2, BlockedTensor& T1, const double& alpha, double& C0){
    Timer timer;
    BlockedTensor temp;
    double E = 0.0;

    temp = ambit::BlockedTensor::build(tensor_type_,"temp",{"aaaa"});
    temp["uvxy"] += H2["evxy"] * T1["ue"];
    temp["uvxy"] -= H2["uvmy"] * T1["mx"];
    E += 0.5 * temp["uvxy"] * Lambda2_["xyuv"];

    temp = ambit::BlockedTensor::build(tensor_type_,"temp",{"AAAA"});
    temp["UVXY"] += H2["EVXY"] * T1["UE"];
    temp["UVXY"] -= H2["UVMY"] * T1["MX"];
    E += 0.5 * temp["UVXY"] * Lambda2_["XYUV"];

    temp = ambit::BlockedTensor::build(tensor_type_,"temp",{"aAaA"});
    temp["uVxY"] += H2["eVxY"] * T1["ue"];
    temp["uVxY"] += H2["uExY"] * T1["VE"];
    temp["uVxY"] -= H2["uVmY"] * T1["mx"];
    temp["uVxY"] -= H2["uVxM"] * T1["MY"];
    E += temp["uVxY"] * Lambda2_["xYuV"];

    E  *= alpha;
    C0 += E;

    if(print_ > 2){
        outfile->Printf("\n    Time for [H2, T1] -> C0 : %12.3f",timer.get());
    }
    dsrg_time_.add("210",timer.get());
}

void DSRG_MRPT3::H2_T2_C0(BlockedTensor& H2, BlockedTensor& T2, const double& alpha, double& C0){
    Timer timer;

    // <[Hbar2, T2]> (C_2)^4
    double E = H2["eFmN"] * T2["mNeF"];
    E += 0.25 * H2["efmn"] * T2["mnef"];
    E += 0.25 * H2["EFMN"] * T2["MNEF"];

    BlockedTensor temp = ambit::BlockedTensor::build(tensor_type_,"temp",spin_cases({"aa"}));
    temp["vu"] += 0.5 * H2["efmu"] * T2["mvef"];
    temp["vu"] += H2["fEuM"] * T2["vMfE"];
    temp["VU"] += 0.5 * H2["EFMU"] * T2["MVEF"];
    temp["VU"] += H2["eFmU"] * T2["mVeF"];
    E += temp["vu"] * Gamma1_["uv"];
    E += temp["VU"] * Gamma1_["UV"];

    temp.zero();
    temp["vu"] += 0.5 * H2["vemn"] * T2["mnue"];
    temp["vu"] += H2["vEmN"] * T2["mNuE"];
    temp["VU"] += 0.5 * H2["VEMN"] * T2["MNUE"];
    temp["VU"] += H2["eVnM"] * T2["nMeU"];
    E += temp["vu"] * Eta1_["uv"];
    E += temp["VU"] * Eta1_["UV"];

    temp = BTF_->build(tensor_type_,"temp",spin_cases({"aaaa"}));
    temp["yvxu"] += H2["efxu"] * T2["yvef"];
    temp["yVxU"] += H2["eFxU"] * T2["yVeF"];
    temp["YVXU"] += H2["EFXU"] * T2["YVEF"];
    E += 0.25 * temp["yvxu"] * Gamma1_["xy"] * Gamma1_["uv"];
    E += temp["yVxU"] * Gamma1_["UV"] * Gamma1_["xy"];
    E += 0.25 * temp["YVXU"] * Gamma1_["XY"] * Gamma1_["UV"];

    temp.zero();
    temp["vyux"] += H2["vymn"] * T2["mnux"];
    temp["vYuX"] += H2["vYmN"] * T2["mNuX"];
    temp["VYUX"] += H2["VYMN"] * T2["MNUX"];
    E += 0.25 * temp["vyux"] * Eta1_["uv"] * Eta1_["xy"];
    E += temp["vYuX"] * Eta1_["uv"] * Eta1_["XY"];
    E += 0.25 * temp["VYUX"] * Eta1_["UV"] * Eta1_["XY"];

    temp.zero();
    temp["vyux"] += H2["vemx"] * T2["myue"];
    temp["vyux"] += H2["vExM"] * T2["yMuE"];
    temp["VYUX"] += H2["eVmX"] * T2["mYeU"];
    temp["VYUX"] += H2["VEXM"] * T2["YMUE"];
    E += temp["vyux"] * Gamma1_["xy"] * Eta1_["uv"];
    E += temp["VYUX"] * Gamma1_["XY"] * Eta1_["UV"];
    temp["yVxU"] = H2["eVxM"] * T2["yMeU"];
    E += temp["yVxU"] * Gamma1_["xy"] * Eta1_["UV"];
    temp["vYuX"] = H2["vEmX"] * T2["mYuE"];
    E += temp["vYuX"] * Gamma1_["XY"] * Eta1_["uv"];

    temp.zero();
    temp["yvxu"] += 0.5 * Gamma1_["wz"] * H2["vexw"] * T2["yzue"];
    temp["yvxu"] += Gamma1_["WZ"] * H2["vExW"] * T2["yZuE"];
    temp["yvxu"] += 0.5 * Eta1_["wz"] * T2["myuw"] * H2["vzmx"];
    temp["yvxu"] += Eta1_["WZ"] * T2["yMuW"] * H2["vZxM"];
    E += temp["yvxu"] * Gamma1_["xy"] * Eta1_["uv"];

    temp["YVXU"] += 0.5 * Gamma1_["WZ"] * H2["VEXW"] * T2["YZUE"];
    temp["YVXU"] += Gamma1_["wz"] * H2["eVwX"] * T2["zYeU"];
    temp["YVXU"] += 0.5 * Eta1_["WZ"] * T2["MYUW"] * H2["VZMX"];
    temp["YVXU"] += Eta1_["wz"] * H2["zVmX"] * T2["mYwU"];
    E += temp["YVXU"] * Gamma1_["XY"] * Eta1_["UV"];

    // <[Hbar2, T2]> C_4 (C_2)^2 HH -- combined with PH
    temp = ambit::BlockedTensor::build(tensor_type_,"temp",spin_cases({"aaaa"}));
    temp["uvxy"] += 0.125 * H2["uvmn"] * T2["mnxy"];
    temp["uvxy"] += 0.25 * Gamma1_["wz"] * H2["uvmw"] * T2["mzxy"];
    temp["uVxY"] += H2["uVmN"] * T2["mNxY"];
    temp["uVxY"] += Gamma1_["wz"] * T2["zMxY"] * H2["uVwM"];
    temp["uVxY"] += Gamma1_["WZ"] * H2["uVmW"] * T2["mZxY"];
    temp["UVXY"] += 0.125 * H2["UVMN"] * T2["MNXY"];
    temp["UVXY"] += 0.25 * Gamma1_["WZ"] * H2["UVMW"] * T2["MZXY"];

    // <[Hbar2, T2]> C_4 (C_2)^2 PP -- combined with PH
    temp["uvxy"] += 0.125 * H2["efxy"] * T2["uvef"];
    temp["uvxy"] += 0.25 * Eta1_["wz"] * T2["uvew"] * H2["ezxy"];
    temp["uVxY"] += H2["eFxY"] * T2["uVeF"];
    temp["uVxY"] += Eta1_["wz"] * H2["zExY"] * T2["uVwE"];
    temp["uVxY"] += Eta1_["WZ"] * T2["uVeW"] * H2["eZxY"];
    temp["UVXY"] += 0.125 * H2["EFXY"] * T2["UVEF"];
    temp["UVXY"] += 0.25 * Eta1_["WZ"] * T2["UVEW"] * H2["EZXY"];

    // <[Hbar2, T2]> C_4 (C_2)^2 PH
    temp["uvxy"] += H2["eumx"] * T2["mvey"];
    temp["uvxy"] += H2["uExM"] * T2["vMyE"];
    temp["uvxy"] += Gamma1_["wz"] * T2["zvey"] * H2["euwx"];
    temp["uvxy"] += Gamma1_["WZ"] * H2["uExW"] * T2["vZyE"];
    temp["uvxy"] += Eta1_["zw"] * H2["wumx"] * T2["mvzy"];
    temp["uvxy"] += Eta1_["ZW"] * T2["vMyZ"] * H2["uWxM"];
    E += temp["uvxy"] * Lambda2_["xyuv"];

    temp["UVXY"] += H2["eUmX"] * T2["mVeY"];
    temp["UVXY"] += H2["EUMX"] * T2["MVEY"];
    temp["UVXY"] += Gamma1_["wz"] * T2["zVeY"] * H2["eUwX"];
    temp["UVXY"] += Gamma1_["WZ"] * T2["ZVEY"] * H2["EUWX"];
    temp["UVXY"] += Eta1_["zw"] * H2["wUmX"] * T2["mVzY"];
    temp["UVXY"] += Eta1_["ZW"] * H2["WUMX"] * T2["MVZY"];
    E += temp["UVXY"] * Lambda2_["XYUV"];

    temp["uVxY"] += H2["uexm"] * T2["mVeY"];
    temp["uVxY"] += H2["uExM"] * T2["MVEY"];
    temp["uVxY"] -= H2["eVxM"] * T2["uMeY"];
    temp["uVxY"] -= H2["uEmY"] * T2["mVxE"];
    temp["uVxY"] += H2["eVmY"] * T2["umxe"];
    temp["uVxY"] += H2["EVMY"] * T2["uMxE"];

    temp["uVxY"] += Gamma1_["wz"] * T2["zVeY"] * H2["uexw"];
    temp["uVxY"] += Gamma1_["WZ"] * T2["ZVEY"] * H2["uExW"];
    temp["uVxY"] -= Gamma1_["WZ"] * H2["eVxW"] * T2["uZeY"];
    temp["uVxY"] -= Gamma1_["wz"] * T2["zVxE"] * H2["uEwY"];
    temp["uVxY"] += Gamma1_["wz"] * T2["zuex"] * H2["eVwY"];
    temp["uVxY"] -= Gamma1_["WZ"] * H2["EVYW"] * T2["uZxE"];

    temp["uVxY"] += Eta1_["zw"] * H2["wumx"] * T2["mVzY"];
    temp["uVxY"] += Eta1_["ZW"] * T2["VMYZ"] * H2["uWxM"];
    temp["uVxY"] -= Eta1_["zw"] * H2["wVxM"] * T2["uMzY"];
    temp["uVxY"] -= Eta1_["ZW"] * T2["mVxZ"] * H2["uWmY"];
    temp["uVxY"] += Eta1_["zw"] * T2["umxz"] * H2["wVmY"];
    temp["uVxY"] += Eta1_["ZW"] * H2["WVMY"] * T2["uMxZ"];
    E += temp["uVxY"] * Lambda2_["xYuV"];

    // <[Hbar2, T2]> C_6 C_2
    if(options_.get_str("THREEPDC") != "ZERO"){
        temp = ambit::BlockedTensor::build(tensor_type_,"temp",{"aaaaaa"});
        temp["uvwxyz"] += H2["uviz"] * T2["iwxy"];      //  aaaaaa from hole
        temp["uvwxyz"] += H2["waxy"] * T2["uvaz"];      //  aaaaaa from particle
        E += 0.25 * temp["uvwxyz"] * Lambda3_["xyzuvw"];

        temp = ambit::BlockedTensor::build(tensor_type_,"temp",{"AAAAAA"});
        temp["UVWXYZ"] += H2["UVIZ"] * T2["IWXY"];      //  AAAAAA from hole
        temp["UVWXYZ"] += H2["WAXY"] * T2["UVAZ"];      //  AAAAAA from particle
        E += 0.25 * temp["UVWXYZ"] * Lambda3_["XYZUVW"];

        temp = ambit::BlockedTensor::build(tensor_type_,"temp",{"aaAaaA"});
        temp["uvWxyZ"] -= H2["uviy"] * T2["iWxZ"];      //  aaAaaA from hole
        temp["uvWxyZ"] -= H2["uWiZ"] * T2["ivxy"];      //  aaAaaA from hole
        temp["uvWxyZ"] += 2.0 * H2["uWyI"] * T2["vIxZ"];//  aaAaaA from hole

        temp["uvWxyZ"] += H2["aWxZ"] * T2["uvay"];      //  aaAaaA from particle
        temp["uvWxyZ"] -= H2["vaxy"] * T2["uWaZ"];      //  aaAaaA from particle
        temp["uvWxyZ"] -= 2.0 * H2["vAxZ"] * T2["uWyA"];//  aaAaaA from particle
        E += 0.5 * temp["uvWxyZ"] * Lambda3_["xyZuvW"];

        temp = ambit::BlockedTensor::build(tensor_type_,"temp",{"aAAaAA"});
        temp["uVWxYZ"] -= H2["VWIZ"] * T2["uIxY"];      //  aAAaAA from hole
        temp["uVWxYZ"] -= H2["uVxI"] * T2["IWYZ"];      //  aAAaAA from hole
        temp["uVWxYZ"] += 2.0 * H2["uViZ"] * T2["iWxY"];//  aAAaAA from hole

        temp["uVWxYZ"] += H2["uAxY"] * T2["VWAZ"];      //  aAAaAA from particle
        temp["uVWxYZ"] -= H2["WAYZ"] * T2["uVxA"];      //  aAAaAA from particle
        temp["uVWxYZ"] -= 2.0 * H2["aWxY"] * T2["uVaZ"];//  aAAaAA from particle
        E += 0.5 * temp["uVWxYZ"] * Lambda3_["xYZuVW"];
    }

    // multiply prefactor and copy to C0
    E  *= alpha;
    C0 += E;

    if(print_ > 2){
        outfile->Printf("\n    Time for [H2, T2] -> C0 : %12.3f",timer.get());
    }
    dsrg_time_.add("220",timer.get());
}

void DSRG_MRPT3::H1_T1_C1(BlockedTensor& H1, BlockedTensor& T1, const double& alpha, BlockedTensor& C1){
    Timer timer;

    C1["ip"] += alpha * H1["ap"] * T1["ia"];
    C1["qa"] -= alpha * T1["ia"] * H1["qi"];

    C1["IP"] += alpha * H1["AP"] * T1["IA"];
    C1["QA"] -= alpha * T1["IA"] * H1["QI"];

    if(print_ > 2){
        outfile->Printf("\n    Time for [H1, T1] -> C1 : %12.3f",timer.get());
    }
    dsrg_time_.add("111",timer.get());
}

void DSRG_MRPT3::H1_T2_C1(BlockedTensor& H1, BlockedTensor& T2, const double& alpha, BlockedTensor& C1){
    Timer timer;

    C1["ia"] += alpha * H1["bm"] * T2["imab"];
    C1["ia"] += alpha * H1["bu"] * T2["ivab"] * Gamma1_["uv"];
    C1["ia"] -= alpha * H1["vj"] * T2["ijau"] * Gamma1_["uv"];
    C1["ia"] += alpha * H1["BM"] * T2["iMaB"];
    C1["ia"] += alpha * H1["BU"] * T2["iVaB"] * Gamma1_["UV"];
    C1["ia"] -= alpha * H1["VJ"] * T2["iJaU"] * Gamma1_["UV"];

    C1["IA"] += alpha * H1["bm"] * T2["mIbA"];
    C1["IA"] += alpha * H1["bu"] * Gamma1_["uv"] * T2["vIbA"];
    C1["IA"] -= alpha * H1["vj"] * T2["jIuA"] * Gamma1_["uv"];
    C1["IA"] += alpha * H1["BM"] * T2["IMAB"];
    C1["IA"] += alpha * H1["BU"] * T2["IVAB"] * Gamma1_["UV"];
    C1["IA"] -= alpha * H1["VJ"] * T2["IJAU"] * Gamma1_["UV"];

    if(print_ > 2){
        outfile->Printf("\n    Time for [H1, T2] -> C1 : %12.3f",timer.get());
    }
    dsrg_time_.add("121",timer.get());
}

void DSRG_MRPT3::H2_T1_C1(BlockedTensor& H2, BlockedTensor& T1, const double& alpha, BlockedTensor& C1){
    Timer timer;

    C1["qp"] += alpha * T1["ma"] * H2["qapm"];
    C1["qp"] += alpha * T1["xe"] * Gamma1_["yx"] * H2["qepy"];
    C1["qp"] -= alpha * T1["mu"] * Gamma1_["uv"] * H2["qvpm"];
    C1["qp"] += alpha * T1["MA"] * H2["qApM"];
    C1["qp"] += alpha * T1["XE"] * Gamma1_["YX"] * H2["qEpY"];
    C1["qp"] -= alpha * T1["MU"] * Gamma1_["UV"] * H2["qVpM"];

    C1["QP"] += alpha * T1["ma"] * H2["aQmP"];
    C1["QP"] += alpha * T1["xe"] * Gamma1_["yx"] * H2["eQyP"];
    C1["QP"] -= alpha * T1["mu"] * Gamma1_["uv"] * H2["vQmP"];
    C1["QP"] += alpha * T1["MA"] * H2["QAPM"];
    C1["QP"] += alpha * T1["XE"] * Gamma1_["YX"] * H2["QEPY"];
    C1["QP"] -= alpha * T1["MU"] * Gamma1_["UV"] * H2["QVPM"];

    if(print_ > 2){
        outfile->Printf("\n    Time for [H2, T1] -> C1 : %12.3f",timer.get());
    }
    dsrg_time_.add("211",timer.get());
}

void DSRG_MRPT3::H2_T2_C1(BlockedTensor& H2, BlockedTensor& T2, const double& alpha, BlockedTensor& C1){
    Timer timer;
    BlockedTensor temp;

    // [Hbar2, T2] (C_2)^3 -> C1 particle contractions
    C1["ir"] += 0.5 * alpha * H2["abrm"] * T2["imab"];
    C1["ir"] += alpha * H2["aBrM"] * T2["iMaB"];
    C1["IR"] += 0.5 * alpha * H2["ABRM"] * T2["IMAB"];
    C1["IR"] += alpha * H2["aBmR"] * T2["mIaB"];

    C1["ir"] += 0.5 * alpha * Gamma1_["uv"] * H2["abru"] * T2["ivab"];
    C1["ir"] += alpha * Gamma1_["UV"] * H2["aBrU"] * T2["iVaB"];
    C1["IR"] += 0.5 * alpha * Gamma1_["UV"] * H2["ABRU"] * T2["IVAB"];
    C1["IR"] += alpha * Gamma1_["uv"] * H2["aBuR"] * T2["vIaB"];

    C1["ir"] += 0.5 * alpha * T2["ijux"] * Gamma1_["xy"] * Gamma1_["uv"] * H2["vyrj"];
    C1["IR"] += 0.5 * alpha * T2["IJUX"] * Gamma1_["XY"] * Gamma1_["UV"] * H2["VYRJ"];
    temp = ambit::BlockedTensor::build(tensor_type_,"temp",{"hHaA"});
    temp["iJvY"] = T2["iJuX"] * Gamma1_["XY"] * Gamma1_["uv"];
    C1["ir"] += alpha * temp["iJvY"] * H2["vYrJ"];
    C1["IR"] += alpha * temp["jIvY"] * H2["vYjR"];

    C1["ir"] -= alpha * Gamma1_["uv"] * H2["vbrm"] * T2["imub"];
    C1["ir"] -= alpha * Gamma1_["uv"] * H2["vBrM"] * T2["iMuB"];
    C1["ir"] -= alpha * Gamma1_["UV"] * T2["iMbU"] * H2["bVrM"];
    C1["IR"] -= alpha * Gamma1_["UV"] * H2["VBRM"] * T2["IMUB"];
    C1["IR"] -= alpha * Gamma1_["UV"] * H2["bVmR"] * T2["mIbU"];
    C1["IR"] -= alpha * Gamma1_["uv"] * H2["vBmR"] * T2["mIuB"];

    C1["ir"] -= alpha * H2["vbrx"] * Gamma1_["uv"] * Gamma1_["xy"] * T2["iyub"];
    C1["ir"] -= alpha * H2["vBrX"] * Gamma1_["uv"] * Gamma1_["XY"] * T2["iYuB"];
    C1["ir"] -= alpha * H2["bVrX"] * Gamma1_["XY"] * Gamma1_["UV"] * T2["iYbU"];
    C1["IR"] -= alpha * H2["VBRX"] * Gamma1_["UV"] * Gamma1_["XY"] * T2["IYUB"];
    C1["IR"] -= alpha * H2["vBxR"] * Gamma1_["uv"] * Gamma1_["xy"] * T2["yIuB"];
    C1["IR"] -= alpha * T2["yIbU"] * Gamma1_["UV"] * Gamma1_["xy"] * H2["bVxR"];

    // [Hbar2, T2] (C_2)^3 -> C1 hole contractions
    C1["pa"] -= 0.5 * alpha * H2["peij"] * T2["ijae"];
    C1["pa"] -= alpha * H2["pEiJ"] * T2["iJaE"];
    C1["PA"] -= 0.5 * alpha * H2["PEIJ"] * T2["IJAE"];
    C1["PA"] -= alpha * H2["ePiJ"] * T2["iJeA"];

    C1["pa"] -= 0.5 * alpha * Eta1_["uv"] * T2["ijau"] * H2["pvij"];
    C1["pa"] -= alpha * Eta1_["UV"] * T2["iJaU"] * H2["pViJ"];
    C1["PA"] -= 0.5 * alpha * Eta1_["UV"] * T2["IJAU"] * H2["PVIJ"];
    C1["PA"] -= alpha * Eta1_["uv"] * T2["iJuA"] * H2["vPiJ"];

    C1["pa"] -= 0.5 * alpha * T2["vyab"] * Eta1_["uv"] * Eta1_["xy"] * H2["pbux"];
    C1["PA"] -= 0.5 * alpha * T2["VYAB"] * Eta1_["UV"] * Eta1_["XY"] * H2["PBUX"];
    temp = ambit::BlockedTensor::build(tensor_type_,"temp",{"aApP"});
    temp["uXaB"] = T2["vYaB"] * Eta1_["uv"] * Eta1_["XY"];
    C1["pa"] -= alpha * H2["pBuX"] * temp["uXaB"];
    C1["PA"] -= alpha * H2["bPuX"] * temp["uXbA"];

    C1["pa"] += alpha * Eta1_["uv"] * T2["vjae"] * H2["peuj"];
    C1["pa"] += alpha * Eta1_["uv"] * T2["vJaE"] * H2["pEuJ"];
    C1["pa"] += alpha * Eta1_["UV"] * H2["pEjU"] * T2["jVaE"];
    C1["PA"] += alpha * Eta1_["UV"] * T2["VJAE"] * H2["PEUJ"];
    C1["PA"] += alpha * Eta1_["uv"] * T2["vJeA"] * H2["ePuJ"];
    C1["PA"] += alpha * Eta1_["UV"] * H2["ePjU"] * T2["jVeA"];

    C1["pa"] += alpha * T2["vjax"] * Eta1_["uv"] * Eta1_["xy"] * H2["pyuj"];
    C1["pa"] += alpha * T2["vJaX"] * Eta1_["uv"] * Eta1_["XY"] * H2["pYuJ"];
    C1["pa"] += alpha * T2["jVaX"] * Eta1_["XY"] * Eta1_["UV"] * H2["pYjU"];
    C1["PA"] += alpha * T2["VJAX"] * Eta1_["UV"] * Eta1_["XY"] * H2["PYUJ"];
    C1["PA"] += alpha * T2["vJxA"] * Eta1_["uv"] * Eta1_["xy"] * H2["yPuJ"];
    C1["PA"] += alpha * H2["yPjU"] * Eta1_["UV"] * Eta1_["xy"] * T2["jVxA"];

    // [Hbar2, T2] C_4 C_2 2:2 -> C1
    C1["ir"] +=  0.25 * alpha * T2["ijxy"] * Lambda2_["xyuv"] * H2["uvrj"];
    C1["IR"] +=  0.25 * alpha * T2["IJXY"] * Lambda2_["XYUV"] * H2["UVRJ"];
    temp = ambit::BlockedTensor::build(tensor_type_,"temp",{"hHaA"});
    temp["iJuV"] = T2["iJxY"] * Lambda2_["xYuV"];
    C1["ir"] += alpha * H2["uVrJ"] * temp["iJuV"];
    C1["IR"] += alpha * H2["uVjR"] * temp["jIuV"];

    C1["pa"] -=  0.25 * alpha * Lambda2_["xyuv"] * T2["uvab"] * H2["pbxy"];
    C1["PA"] -=  0.25 * alpha * Lambda2_["XYUV"] * T2["UVAB"] * H2["PBXY"];
    temp = ambit::BlockedTensor::build(tensor_type_,"temp",{"aApP"});
    temp["xYaB"] = T2["uVaB"] * Lambda2_["xYuV"];
    C1["pa"] -= alpha * H2["pBxY"] * temp["xYaB"];
    C1["PA"] -= alpha * H2["bPxY"] * temp["xYbA"];

    C1["ir"] -= alpha * Lambda2_["yXuV"] * T2["iVyA"] * H2["uArX"];
    C1["IR"] -= alpha * Lambda2_["xYvU"] * T2["vIaY"] * H2["aUxR"];
    C1["pa"] += alpha * Lambda2_["xYvU"] * T2["vIaY"] * H2["pUxI"];
    C1["PA"] += alpha * Lambda2_["yXuV"] * T2["iVyA"] * H2["uPiX"];

    temp = ambit::BlockedTensor::build(tensor_type_,"temp",{"hapa"});
    temp["ixau"] += Lambda2_["xyuv"] * T2["ivay"];
    temp["ixau"] += Lambda2_["xYuV"] * T2["iVaY"];
    C1["ir"] += alpha * temp["ixau"] * H2["aurx"];
    C1["pa"] -= alpha * H2["puix"] * temp["ixau"];
    temp = ambit::BlockedTensor::build(tensor_type_,"temp",{"hApA"});
    temp["iXaU"] += Lambda2_["XYUV"] * T2["iVaY"];
    temp["iXaU"] += Lambda2_["yXvU"] * T2["ivay"];
    C1["ir"] += alpha * temp["iXaU"] * H2["aUrX"];
    C1["pa"] -= alpha * H2["pUiX"] * temp["iXaU"];
    temp = ambit::BlockedTensor::build(tensor_type_,"temp",{"aHaP"});
    temp["xIuA"] += Lambda2_["xyuv"] * T2["vIyA"];
    temp["xIuA"] += Lambda2_["xYuV"] * T2["VIYA"];
    C1["IR"] += alpha * temp["xIuA"] * H2["uAxR"];
    C1["PA"] -= alpha * H2["uPxI"] * temp["xIuA"];
    temp = ambit::BlockedTensor::build(tensor_type_,"temp",{"HAPA"});
    temp["IXAU"] += Lambda2_["XYUV"] * T2["IVAY"];
    temp["IXAU"] += Lambda2_["yXvU"] * T2["vIyA"];
    C1["IR"] += alpha * temp["IXAU"] * H2["AURX"];
    C1["PA"] -= alpha * H2["PUIX"] * temp["IXAU"];

    // [Hbar2, T2] C_4 C_2 1:3 -> C1
    temp = ambit::BlockedTensor::build(tensor_type_,"temp",{"pa"});
    temp["au"] += 0.5 * Lambda2_["xyuv"] * H2["avxy"];
    temp["au"] += Lambda2_["xYuV"] * H2["aVxY"];
    C1["jb"] += alpha * temp["au"] * T2["ujab"];
    C1["JB"] += alpha * temp["au"] * T2["uJaB"];
    temp = ambit::BlockedTensor::build(tensor_type_,"temp",{"PA"});
    temp["AU"] += 0.5 * Lambda2_["XYUV"] * H2["AVXY"];
    temp["AU"] += Lambda2_["xYvU"] * H2["vAxY"];
    C1["jb"] += alpha * temp["AU"] * T2["jUbA"];
    C1["JB"] += alpha * temp["AU"] * T2["UJAB"];

    temp = ambit::BlockedTensor::build(tensor_type_,"temp",{"ah"});
    temp["xi"] += 0.5 * Lambda2_["xyuv"] * H2["uviy"];
    temp["xi"] += Lambda2_["xYuV"] * H2["uViY"];
    C1["jb"] -= alpha * temp["xi"] * T2["ijxb"];
    C1["JB"] -= alpha * temp["xi"] * T2["iJxB"];
    temp = ambit::BlockedTensor::build(tensor_type_,"temp",{"AH"});
    temp["XI"] += 0.5 * Lambda2_["XYUV"] * H2["UVIY"];
    temp["XI"] += Lambda2_["yXvU"] * H2["vUyI"];
    C1["jb"] -= alpha * temp["XI"] * T2["jIbX"];
    C1["JB"] -= alpha * temp["XI"] * T2["IJXB"];

    temp = ambit::BlockedTensor::build(tensor_type_,"temp",{"av"});
    temp["xe"] += 0.5 * T2["uvey"] * Lambda2_["xyuv"];
    temp["xe"] += T2["uVeY"] * Lambda2_["xYuV"];
    C1["qs"] += alpha * temp["xe"] * H2["eqxs"];
    C1["QS"] += alpha * temp["xe"] * H2["eQxS"];
    temp = ambit::BlockedTensor::build(tensor_type_,"temp",{"AV"});
    temp["XE"] += 0.5 * T2["UVEY"] * Lambda2_["XYUV"];
    temp["XE"] += T2["uVyE"] * Lambda2_["yXuV"];
    C1["qs"] += alpha * temp["XE"] * H2["qEsX"];
    C1["QS"] += alpha * temp["XE"] * H2["EQXS"];

    temp = ambit::BlockedTensor::build(tensor_type_,"temp",{"ca"});
    temp["mu"] += 0.5 * T2["mvxy"] * Lambda2_["xyuv"];
    temp["mu"] += T2["mVxY"] * Lambda2_["xYuV"];
    C1["qs"] -= alpha * temp["mu"] * H2["uqms"];
    C1["QS"] -= alpha * temp["mu"] * H2["uQmS"];
    temp = ambit::BlockedTensor::build(tensor_type_,"temp",{"CA"});
    temp["MU"] += 0.5 * T2["MVXY"] * Lambda2_["XYUV"];
    temp["MU"] += T2["vMxY"] * Lambda2_["xYvU"];
    C1["qs"] -= alpha * temp["MU"] * H2["qUsM"];
    C1["QS"] -= alpha * temp["MU"] * H2["UQMS"];

    if(print_ > 2){
        outfile->Printf("\n    Time for [H2, T2] -> C1 : %12.3f",timer.get());
    }
    dsrg_time_.add("221",timer.get());
}

void DSRG_MRPT3::H1_T2_C2(BlockedTensor& H1, BlockedTensor& T2, const double& alpha, BlockedTensor& C2){
    Timer timer;

    C2["ijpb"] += alpha * T2["ijab"] * H1["ap"];
    C2["ijap"] += alpha * T2["ijab"] * H1["bp"];
    C2["qjab"] -= alpha * T2["ijab"] * H1["qi"];
    C2["iqab"] -= alpha * T2["ijab"] * H1["qj"];

    C2["iJpB"] += alpha * T2["iJaB"] * H1["ap"];
    C2["iJaP"] += alpha * T2["iJaB"] * H1["BP"];
    C2["qJaB"] -= alpha * T2["iJaB"] * H1["qi"];
    C2["iQaB"] -= alpha * T2["iJaB"] * H1["QJ"];

    C2["IJPB"] += alpha * T2["IJAB"] * H1["AP"];
    C2["IJAP"] += alpha * T2["IJAB"] * H1["BP"];
    C2["QJAB"] -= alpha * T2["IJAB"] * H1["QI"];
    C2["IQAB"] -= alpha * T2["IJAB"] * H1["QJ"];

    if(print_ > 2){
        outfile->Printf("\n    Time for [H1, T2] -> C2 : %12.3f",timer.get());
    }
    dsrg_time_.add("122",timer.get());
}

void DSRG_MRPT3::H2_T1_C2(BlockedTensor& H2, BlockedTensor& T1, const double& alpha, BlockedTensor& C2){
    Timer timer;

    C2["irpq"] += alpha * T1["ia"] * H2["arpq"];
    C2["ripq"] += alpha * T1["ia"] * H2["rapq"];
    C2["rsaq"] -= alpha * T1["ia"] * H2["rsiq"];
    C2["rspa"] -= alpha * T1["ia"] * H2["rspi"];

    C2["iRpQ"] += alpha * T1["ia"] * H2["aRpQ"];
    C2["rIpQ"] += alpha * T1["IA"] * H2["rApQ"];
    C2["rSaQ"] -= alpha * T1["ia"] * H2["rSiQ"];
    C2["rSpA"] -= alpha * T1["IA"] * H2["rSpI"];

    C2["IRPQ"] += alpha * T1["IA"] * H2["ARPQ"];
    C2["RIPQ"] += alpha * T1["IA"] * H2["RAPQ"];
    C2["RSAQ"] -= alpha * T1["IA"] * H2["RSIQ"];
    C2["RSPA"] -= alpha * T1["IA"] * H2["RSPI"];

    if(print_ > 2){
        outfile->Printf("\n    Time for [H2, T1] -> C2 : %12.3f",timer.get());
    }
    dsrg_time_.add("212",timer.get());
}

void DSRG_MRPT3::H2_T2_C2(BlockedTensor& H2, BlockedTensor& T2, const double& alpha, BlockedTensor& C2){
    Timer timer;

    // particle-particle contractions
    C2["ijrs"] += 0.5 * alpha * H2["abrs"] * T2["ijab"];
    C2["iJrS"] += alpha * H2["aBrS"] * T2["iJaB"];
    C2["IJRS"] += 0.5 * alpha * H2["ABRS"] * T2["IJAB"];

    C2["ijrs"] -= alpha * Gamma1_["xy"] * H2["ybrs"] * T2["ijxb"];
    C2["iJrS"] -= alpha * Gamma1_["xy"] * H2["yBrS"] * T2["iJxB"];
    C2["iJrS"] -= alpha * Gamma1_["XY"] * T2["iJbX"] * H2["bYrS"];
    C2["IJRS"] -= alpha * Gamma1_["XY"] * H2["YBRS"] * T2["IJXB"];

    // hole-hole contractions
    C2["pqab"] += 0.5 * alpha * H2["pqij"] * T2["ijab"];
    C2["pQaB"] += alpha * H2["pQiJ"] * T2["iJaB"];
    C2["PQAB"] += 0.5 * alpha * H2["PQIJ"] * T2["IJAB"];

    C2["pqab"] -= alpha * Eta1_["xy"] * T2["yjab"] * H2["pqxj"];
    C2["pQaB"] -= alpha * Eta1_["xy"] * T2["yJaB"] * H2["pQxJ"];
    C2["pQaB"] -= alpha * Eta1_["XY"] * H2["pQjX"] * T2["jYaB"];
    C2["PQAB"] -= alpha * Eta1_["XY"] * T2["YJAB"] * H2["PQXJ"];

    // hole-particle contractions
    BlockedTensor temp = ambit::BlockedTensor::build(tensor_type_,"temp",{"ghgp"});
    temp["qjsb"] += alpha * H2["aqms"] * T2["mjab"];
    temp["qjsb"] += alpha * H2["qAsM"] * T2["jMbA"];
    temp["qjsb"] += alpha * Gamma1_["xy"] * T2["yjab"] * H2["aqxs"];
    temp["qjsb"] += alpha * Gamma1_["XY"] * T2["jYbA"] * H2["qAsX"];
    temp["qjsb"] -= alpha * Gamma1_["xy"] * H2["yqis"] * T2["ijxb"];
    temp["qjsb"] -= alpha * Gamma1_["XY"] * H2["qYsI"] * T2["jIbX"];
    C2["qjsb"] += temp["qjsb"];
    C2["jqsb"] -= temp["qjsb"];
    C2["qjbs"] -= temp["qjsb"];
    C2["jqbs"] += temp["qjsb"];

    temp = ambit::BlockedTensor::build(tensor_type_,"temp",{"GHGP"});
    temp["QJSB"] += alpha * H2["AQMS"] * T2["MJAB"];
    temp["QJSB"] += alpha * H2["aQmS"] * T2["mJaB"];
    temp["QJSB"] += alpha * Gamma1_["XY"] * T2["YJAB"] * H2["AQXS"];
    temp["QJSB"] += alpha * Gamma1_["xy"] * T2["yJaB"] * H2["aQxS"];
    temp["QJSB"] -= alpha * Gamma1_["XY"] * H2["YQIS"] * T2["IJXB"];
    temp["QJSB"] -= alpha * Gamma1_["xy"] * H2["yQiS"] * T2["iJxB"];
    C2["QJSB"] += temp["QJSB"];
    C2["JQSB"] -= temp["QJSB"];
    C2["QJBS"] -= temp["QJSB"];
    C2["JQBS"] += temp["QJSB"];

    C2["qJsB"] += alpha * H2["aqms"] * T2["mJaB"];
    C2["qJsB"] += alpha * H2["qAsM"] * T2["MJAB"];
    C2["qJsB"] += alpha * Gamma1_["xy"] * T2["yJaB"] * H2["aqxs"];
    C2["qJsB"] += alpha * Gamma1_["XY"] * T2["YJAB"] * H2["qAsX"];
    C2["qJsB"] -= alpha * Gamma1_["xy"] * H2["yqis"] * T2["iJxB"];
    C2["qJsB"] -= alpha * Gamma1_["XY"] * H2["qYsI"] * T2["IJXB"];

    C2["iQsB"] -= alpha * T2["iMaB"] * H2["aQsM"];
    C2["iQsB"] -= alpha * Gamma1_["XY"] * T2["iYaB"] * H2["aQsX"];
    C2["iQsB"] += alpha * Gamma1_["xy"] * H2["yQsJ"] * T2["iJxB"];

    C2["qJaS"] -= alpha * T2["mJaB"] * H2["qBmS"];
    C2["qJaS"] -= alpha * Gamma1_["xy"] * T2["yJaB"] * H2["qBxS"];
    C2["qJaS"] += alpha * Gamma1_["XY"] * H2["qYiS"] * T2["iJaX"];

    C2["iQaS"] += alpha * T2["imab"] * H2["bQmS"];
    C2["iQaS"] += alpha * T2["iMaB"] * H2["BQMS"];
    C2["iQaS"] += alpha * Gamma1_["xy"] * T2["iyab"] * H2["bQxS"];
    C2["iQaS"] += alpha * Gamma1_["XY"] * T2["iYaB"] * H2["BQXS"];
    C2["iQaS"] -= alpha * Gamma1_["xy"] * H2["yQjS"] * T2["ijax"];
    C2["iQaS"] -= alpha * Gamma1_["XY"] * H2["YQJS"] * T2["iJaX"];

    if(print_ > 2){
        outfile->Printf("\n    Time for [H2, T2] -> C2 : %12.3f",timer.get());
    }
    dsrg_time_.add("222",timer.get());
}

std::vector<std::vector<double>> DSRG_MRPT3::diagonalize_Fock_diagblocks(BlockedTensor& U){
    // diagonal blocks identifiers (C-A-V ordering)
    std::vector<std::string> blocks {"cc","aa","vv","CC","AA","VV"};

    // map MO space label to its Dimension
    std::map<std::string, Dimension> MOlabel_to_dimension;
    MOlabel_to_dimension[acore_label_] = mo_space_info_->get_dimension("RESTRICTED_DOCC");
    MOlabel_to_dimension[aactv_label_] = mo_space_info_->get_dimension("ACTIVE");
    MOlabel_to_dimension[avirt_label_] = mo_space_info_->get_dimension("RESTRICTED_UOCC");

    // eigen values to be returned
    size_t ncmo = mo_space_info_->size("CORRELATED");
    Dimension corr = mo_space_info_->get_dimension("CORRELATED");
    std::vector<double> eigenvalues_a(ncmo, 0.0);
    std::vector<double> eigenvalues_b(ncmo, 0.0);

    // map MO space label to its offset Dimension
    std::map<std::string, Dimension> MOlabel_to_offset_dimension;
    int nirrep = corr.n();
    MOlabel_to_offset_dimension["c"] = Dimension(std::vector<int> (nirrep, 0));
    MOlabel_to_offset_dimension["a"] = mo_space_info_->get_dimension("RESTRICTED_DOCC");
    MOlabel_to_offset_dimension["v"] = mo_space_info_->get_dimension("RESTRICTED_DOCC") + mo_space_info_->get_dimension("ACTIVE");

    // figure out index
    auto fill_eigen = [&](std::string block_label, int irrep, std::vector<double> values){
        int h = irrep;
        size_t idx_begin = 0;
        while((--h) >= 0) idx_begin += corr[h];

        std::string label (1, tolower(block_label[0]));
        idx_begin += MOlabel_to_offset_dimension[label][irrep];

        bool spin_alpha = islower(block_label[0]);
        size_t nvalues = values.size();
        if(spin_alpha){
            for(size_t i = 0; i < nvalues; ++i){
                eigenvalues_a[i + idx_begin] = values[i];
            }
        }else{
            for(size_t i = 0; i < nvalues; ++i){
                eigenvalues_b[i + idx_begin] = values[i];
            }
        }
    };

    // diagonalize diagonal blocks
    for(const auto& block: blocks){
        size_t dim = F_.block(block).dim(0);
        if(dim == 0){
            continue;
        }else{
            std::string label (1, tolower(block[0]));
            Dimension space = MOlabel_to_dimension[label];
            int nirrep = space.n();

            // separate Fock with irrep
            for(int h = 0; h < nirrep; ++h){
                size_t h_dim = space[h];
                ambit::Tensor U_h;
                if(h_dim == 0){
                    continue;
                }else if(h_dim == 1){
                    U_h = ambit::Tensor::build(tensor_type_,"U_h",std::vector<size_t> (2, h_dim));
                    U_h.data()[0] = 1.0;
                    ambit::Tensor F_block = ambit::Tensor::build(tensor_type_,"F_block",F_.block(block).dims());
                    F_block.data() = F_.block(block).data();
                    ambit::Tensor T_h = separate_tensor(F_block,space,h);
                    fill_eigen(block,h,T_h.data());
                }else{
                    ambit::Tensor F_block = ambit::Tensor::build(tensor_type_,"F_block",F_.block(block).dims());
                    F_block.data() = F_.block(block).data();
                    ambit::Tensor T_h = separate_tensor(F_block,space,h);
                    auto Feigen = T_h.syev(AscendingEigenvalue);
                    U_h = ambit::Tensor::build(tensor_type_,"U_h",std::vector<size_t> (2, h_dim));
                    U_h("pq") = Feigen["eigenvectors"]("pq");
                    fill_eigen(block,h,Feigen["eigenvalues"].data());
                }
                ambit::Tensor U_out = U.block(block);
                combine_tensor(U_out,U_h,space,h);
            }
        }
    }
    return {eigenvalues_a, eigenvalues_b};
}

ambit::Tensor DSRG_MRPT3::separate_tensor(ambit::Tensor& tens, const Dimension& irrep, const int& h){
    // test tens and irrep
    int tens_dim = static_cast<int>(tens.dim(0));
    if(tens_dim != irrep.sum() || tens_dim != tens.dim(1)){
        throw PSIEXCEPTION("Wrong dimension for the to-be-separated ambit Tensor.");
    }
    if(h >= irrep.n()){
        throw PSIEXCEPTION("Ask for wrong irrep.");
    }

    // from relative (blocks) to absolute (big tensor) index
    auto rel_to_abs = [&](size_t i, size_t j, size_t offset){
        return (i + offset) * tens_dim + (j + offset);
    };

    // compute offset
    size_t offset = 0, h_dim = irrep[h];
    int h_local = h;
    while((--h_local) >= 0) offset += irrep[h_local];

    // fill in values
    ambit::Tensor T_h = ambit::Tensor::build(tensor_type_,"T_h",std::vector<size_t> (2, h_dim));
    for(size_t i = 0; i < h_dim; ++i){
        for(size_t j = 0; j < h_dim; ++j){
            size_t abs_idx = rel_to_abs(i, j, offset);
            T_h.data()[i * h_dim + j] = tens.data()[abs_idx];
        }
    }

    return T_h;
}

void DSRG_MRPT3::combine_tensor(ambit::Tensor& tens, ambit::Tensor& tens_h, const Dimension& irrep, const int& h){
    // test tens and irrep
    if(h >= irrep.n()){
        throw PSIEXCEPTION("Ask for wrong irrep.");
    }
    size_t tens_h_dim = tens_h.dim(0), h_dim = irrep[h];
    if(tens_h_dim != h_dim || tens_h_dim != tens_h.dim(1)){
        throw PSIEXCEPTION("Wrong dimension for the to-be-combined ambit Tensor.");
    }

    // from relative (blocks) to absolute (big tensor) index
    size_t tens_dim = tens.dim(0);
    auto rel_to_abs = [&](size_t i, size_t j, size_t offset){
        return (i + offset) * tens_dim + (j + offset);
    };

    // compute offset
    size_t offset = 0;
    int h_local = h;
    while((--h_local) >= 0) offset += irrep[h_local];

    // fill in values
    for(size_t i = 0; i < h_dim; ++i){
        for(size_t j = 0; j < h_dim; ++j){
            size_t abs_idx = rel_to_abs(i, j, offset);
            tens.data()[abs_idx] = tens_h.data()[i * h_dim + j];
        }
    }
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

void DSRG_MRPT3::check_t2()
{
    T2norm_ = 0.0; T2max_ = 0.0;
    double T2aanorm = 0.0, T2abnorm = 0.0, T2bbnorm = 0.0;
    size_t nonzero_aa = 0, nonzero_ab = 0, nonzero_bb = 0;
    std::vector<std::pair<std::vector<size_t>, double>> t2aa, t2ab, t2bb;

    // create all knids of spin maps; 0: aa, 1: ab, 2:bb
    std::map<int, double> spin_to_norm;
    std::map<int, double> spin_to_nonzero;
    std::map<int, std::vector<std::pair<std::vector<size_t>, double>>> spin_to_t2;
    std::map<int, std::vector<std::pair<std::vector<size_t>, double>>> spin_to_lt2;

    for(const std::string& block: T2_.block_labels()){
        int spin = bool(isupper(block[0])) + bool(isupper(block[1]));
        std::vector<std::pair<std::vector<size_t>, double>>& temp_t2 = spin_to_t2[spin];
        std::vector<std::pair<std::vector<size_t>, double>>& temp_lt2 = spin_to_lt2[spin];

        T2_.block(block).citerate([&](const std::vector<size_t>& i, const double& value){
            if(fabs(value) != 0.0){
                size_t idx0 = label_to_spacemo_[block[0]][i[0]];
                size_t idx1 = label_to_spacemo_[block[1]][i[1]];
                size_t idx2 = label_to_spacemo_[block[2]][i[2]];
                size_t idx3 = label_to_spacemo_[block[3]][i[3]];

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
                T2max_ = T2max_ > fabs(value) ? T2max_ : fabs(value);
            }
        });
    }

    // update values
    T2aanorm = spin_to_norm[0];
    T2abnorm = spin_to_norm[1];
    T2bbnorm = spin_to_norm[2];
    T2norm_ = sqrt(T2aanorm + T2bbnorm + 4 * T2abnorm);

    nonzero_aa = spin_to_nonzero[0];
    nonzero_ab = spin_to_nonzero[1];
    nonzero_bb = spin_to_nonzero[2];

    t2aa = spin_to_t2[0];
    t2ab = spin_to_t2[1];
    t2bb = spin_to_t2[2];

    lt2aa_ = spin_to_lt2[0];
    lt2ab_ = spin_to_lt2[1];
    lt2bb_ = spin_to_lt2[2];

    // print summary
    print_amp_summary("AA", t2aa, sqrt(T2aanorm), nonzero_aa);
    print_amp_summary("AB", t2ab, sqrt(T2abnorm), nonzero_ab);
    print_amp_summary("BB", t2bb, sqrt(T2bbnorm), nonzero_bb);
}

void DSRG_MRPT3::check_t1()
{
    T1max_ = 0.0; T1norm_ = 0.0;
    double T1anorm = 0.0, T1bnorm = 0.0;
    size_t nonzero_a = 0, nonzero_b = 0;
    std::vector<std::pair<std::vector<size_t>, double>> t1a, t1b;

    // create all kinds of spin maps; true: a, false: b
    std::map<bool, double> spin_to_norm;
    std::map<bool, double> spin_to_nonzero;
    std::map<bool, std::vector<std::pair<std::vector<size_t>, double>>> spin_to_t1;
    std::map<bool, std::vector<std::pair<std::vector<size_t>, double>>> spin_to_lt1;

    for(const std::string& block: T1_.block_labels()){
        bool spin_alpha = islower(block[0]) ? true : false;
        std::vector<std::pair<std::vector<size_t>, double>>& temp_t1 = spin_to_t1[spin_alpha];
        std::vector<std::pair<std::vector<size_t>, double>>& temp_lt1 = spin_to_lt1[spin_alpha];

        T1_.block(block).citerate([&](const std::vector<size_t>& i, const double& value){
            if(fabs(value) != 0.0){
                size_t idx0 = label_to_spacemo_[block[0]][i[0]];
                size_t idx1 = label_to_spacemo_[block[1]][i[1]];

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

                T1max_ = T1max_ > fabs(value) ? T1max_ : fabs(value);
            }
        });
    }

    // update value
    T1anorm = spin_to_norm[true];
    T1bnorm = spin_to_norm[false];
    T1norm_ = sqrt(T1anorm + T1bnorm);

    nonzero_a = spin_to_nonzero[true];
    nonzero_b = spin_to_nonzero[false];

    t1a = spin_to_t1[true];
    t1b = spin_to_t1[false];

    lt1a_ = spin_to_lt1[true];
    lt1b_ = spin_to_lt1[false];

    // print summary
    print_amp_summary("A", t1a, sqrt(T1anorm), nonzero_a);
    print_amp_summary("B", t1b, sqrt(T1bnorm), nonzero_b);
}

void DSRG_MRPT3::print_amp_summary(const std::string &name,
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

void DSRG_MRPT3::print_intruder(const std::string &name,
                                const std::vector<std::pair<std::vector<size_t>, double> > &list)
{
    int rank = name.size();
    std::map<char, std::vector<double>> spin_to_F;
    spin_to_F['A'] = Fa_;
    spin_to_F['B'] = Fb_;

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

}} // End Namespaces
