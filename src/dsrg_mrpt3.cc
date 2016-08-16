#include <numeric>
#include <algorithm>
#include <math.h>
#include "mini-boost/boost/format.hpp"

#include <libpsio/psio.hpp>
#include <libpsio/psio.h>
#include <libmints/molecule.h>
#include <libqt/qt.h>
#include <iostream>
#include <fstream>

#include "dsrg_mrpt3.h"
#include "blockedtensorfactory.h"
#include "fci_solver.h"
#include "fci_mo.h"

using namespace ambit;

namespace psi{ namespace forte{

DSRG_MRPT3::DSRG_MRPT3(Reference reference, SharedWavefunction ref_wfn, Options &options,
                       std::shared_ptr<ForteIntegrals> ints, std::shared_ptr<MOSpaceInfo> mo_space_info)
    : Wavefunction(options), reference_(reference), ints_(ints), mo_space_info_(mo_space_info),
      tensor_type_(ambit::CoreTensor), BTF_(new BlockedTensorFactory(options))
{
    // Copy the wavefunction information
    shallow_copy(ref_wfn);

    print_method_banner({"Driven Similarity Renormalization Group",
                         "Third-Order Perturbation Theory", "Chenyang Li"});
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

    // density fitted ERI?
    eri_df_ = false;
    std::string int_type = options_.get_str("INT_TYPE");
    if(int_type == "CHOLESKY" || int_type == "DF" || int_type == "DISKDF") {
        eri_df_ = true;
    }

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

    // if density fitted
    if(eri_df_){
        aux_label_ = "L";
        aux_mos_ = std::vector<size_t> (ints_->nthree());
        std::iota(aux_mos_.begin(),aux_mos_.end(),0);

        BTF_->add_mo_space(aux_label_,"g",aux_mos_,NoSpin);
        label_to_spacemo_[aux_label_[0]] = aux_mos_;

        B_ = BTF_->build(tensor_type_,"B 3-idx",{"Lgg", "LGG"});
        B_.iterate([&](const std::vector<size_t>& i,const std::vector<SpinType>&, double& value){
            value = ints_->three_integral(i[0],i[1],i[2]);
        });
    }

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
    build_tei(V_);

    // build Fock matrix and its diagonal elements
    size_t ncmo = mo_space_info_->size("CORRELATED");
    Fa_ = std::vector<double>(ncmo);
    Fb_ = std::vector<double>(ncmo);
    F_ = BTF_->build(tensor_type_,"Fock",spin_cases({"gc","pa","vv"}));
    build_fock_half();

    // save a copy of zeroth-order Hamiltonian
    F0th_ = BTF_->build(tensor_type_,"T1 1st",spin_cases({"cc","aa","vv"}));
    F0th_["pq"] = F_["pq"];
    F0th_["PQ"] = F_["PQ"];

    // save a copy of first-order Fock matrix
    F1st_ = BTF_->build(tensor_type_,"Fock",spin_cases({"pc","va","cp","av"}));
    F1st_["ai"] = F_["ai"];
    F1st_["ia"] = F_["ai"];
    F1st_["AI"] = F_["AI"];
    F1st_["IA"] = F_["AI"];

    // Prepare Hbar
    relax_ref_ = options_.get_str("RELAX_REF");
    multi_state_ = options_["AVG_STATE"].has_changed();
    if(relax_ref_ != "NONE"){
        if(relax_ref_ != "ONCE"){
            outfile->Printf("\n  Warning: RELAX_REF option \"%s\" is not supported. Change to ONCE", relax_ref_.c_str());
            relax_ref_ = "ONCE";
        }
        if(multi_state_){
            outfile->Printf("\n\n  Multi-state computations ignore RELAX_REF option.");
        }

        Hbar1_ = BTF_->build(tensor_type_,"One-body Hbar",spin_cases({"aa"}));
        Hbar2_ = BTF_->build(tensor_type_,"Two-body Hbar",spin_cases({"aaaa"}));
        Hbar1_["uv"] = F_["uv"];
        Hbar1_["UV"] = F_["UV"];
        Hbar2_["uvxy"] = V_["uvxy"];
        Hbar2_["uVxY"] = V_["uVxY"];
        Hbar2_["UVXY"] = V_["UVXY"];
    } else {
        if(multi_state_){
            Hbar1_ = BTF_->build(tensor_type_,"One-body Hbar",spin_cases({"aa"}));
            Hbar2_ = BTF_->build(tensor_type_,"Two-body Hbar",spin_cases({"aaaa"}));
            Hbar1_["uv"] = F_["uv"];
            Hbar1_["UV"] = F_["UV"];
            Hbar2_["uvxy"] = V_["uvxy"];
            Hbar2_["uVxY"] = V_["uVxY"];
            Hbar2_["UVXY"] = V_["UVXY"];
        }
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
    if(print_ > 3){
        V_.print(stdout);;
    }

    profile_print_ = options_.get_bool("PRINT_TIME_PROFILE");
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

void DSRG_MRPT3::build_tei(BlockedTensor& V){
    if(eri_df_) {
        V["pqrs"]  = B_["gpr"] * B_["gqs"];
        V["pqrs"] -= B_["gps"] * B_["gqr"];

        V["pQrS"]  = B_["gpr"] * B_["gQS"];

        V["PQRS"]  = B_["gPR"] * B_["gQS"];
        V["PQRS"] -= B_["gPS"] * B_["gQR"];
    } else {
        V.iterate([&](const std::vector<size_t>& i,const std::vector<SpinType>& spin,double& value){
            if ((spin[0] == AlphaSpin) and (spin[1] == AlphaSpin)) value = ints_->aptei_aa(i[0],i[1],i[2],i[3]);
            if ((spin[0] == AlphaSpin) and (spin[1] == BetaSpin) ) value = ints_->aptei_ab(i[0],i[1],i[2],i[3]);
            if ((spin[0] == BetaSpin)  and (spin[1] == BetaSpin) ) value = ints_->aptei_bb(i[0],i[1],i[2],i[3]);
        });
    }
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
    build_tei(VFock);
    F_["mn"] += VFock["mvnu"] * Gamma1_["uv"];
    F_["mn"] += VFock["mVnU"] * Gamma1_["UV"];
    F_["MN"] += VFock["vMuN"] * Gamma1_["uv"];
    F_["MN"] += VFock["MVNU"] * Gamma1_["UV"];

    // virtual-virtual block
    VFock = ambit::BlockedTensor::build(tensor_type_,"VFock",{"vava","vAvA","aVaV","VAVA"});
    build_tei(VFock);
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
    F0th_.iterate([&](const std::vector<size_t>& i,const std::vector<SpinType>&,double& value){
        if (i[0] == i[1]){
            value = 0.0;
        }
    });

    // off diagonal elements of diagonal blocks
    double Foff_sum = 0.0;
    std::vector<double> Foff;
    for(const auto& block: {"cc", "aa", "vv", "CC", "AA", "VV"}){
        double value = F0th_.block(block).norm();
        Foff.emplace_back(value);
        Foff_sum += value;
    }

    // add diagonal elements back
    F0th_.iterate([&](const std::vector<size_t>& i,const std::vector<SpinType>& spin,double& value){
        if (spin[0] == AlphaSpin && (i[0] == i[1])){
            value = Fa_[i[0]];
        }
        if (spin[0] == BetaSpin && (i[0] == i[1])){
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
        {"state_type", multi_state_ ? "MULTI_STATE" : "STATE_SPECIFIC"},
        {"reference relaxation", relax_ref_}};

    // Print some information
    outfile->Printf("\n\n  ==> Calculation Information <==\n");
    for (auto& str_dim : calculation_info){
        outfile->Printf("\n    %-40s %15d",str_dim.first.c_str(),str_dim.second);
    }
    for (auto& str_dim : calculation_info_double){
        outfile->Printf("\n    %-40s %15.3e",str_dim.first.c_str(),str_dim.second);
    }
    for (auto& str_dim : calculation_info_string){
        outfile->Printf("\n    %-40s %15s",str_dim.first.c_str(),str_dim.second.c_str());
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
    // check semi-canonical orbitals
    print_h2("Checking Orbitals");
    semi_canonical_ = check_semicanonical();
    if(!semi_canonical_){
        if(!ignore_semicanonical_){
            outfile->Printf("\n    Orbital invariant formalism is employed for DSRG-MRPT3.");
            U_ = ambit::BlockedTensor::build(tensor_type_,"U",spin_cases({"gg"}));
            std::vector<std::vector<double>> eigens = diagonalize_Fock_diagblocks(U_);
            Fa_ = eigens[0];
            Fb_ = eigens[1];
        } else {
            outfile->Printf("\n    Warning: ignore testing of semi-canonical orbitals. DSRG-MRPT3 energy may be meaningless.");
            semi_canonical_ = true;
        }
    }else{
        outfile->Printf("\n    Orbitals are semi-canonicalized.");
    }

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


    // Step 2: compute -[H0th,A1st] and save to C1 and C2
    BlockedTensor C1 = BTF_->build(tensor_type_,"C1",spin_cases({"cp","av","pc","va"}));
    BlockedTensor C2 = BTF_->build(tensor_type_,"C2",spin_cases({"chpp","acpp","aavp","aaav","ppch","ppac","vpaa","avaa"}));
    H1_T1_C1(F0th_,T1_,-1.0,C1);
    H1_T2_C1(F0th_,T2_,-1.0,C1);
    H1_T2_C2(F0th_,T2_,-1.0,C2);

    C1["ai"] = C1["ia"];
    C1["AI"] = C1["IA"];
    C2["abij"] = C2["ijab"];
    C2["aBiJ"] = C2["iJaB"];
    C2["ABIJ"] = C2["IJAB"];

    // compute -[[H0th,A1st],A1st]
    // Step 1: ph and pphh part
    O1_ = BTF_->build(tensor_type_,"O1 PT3 1/3",spin_cases({"pc","va"}));
    O2_ = BTF_->build(tensor_type_,"O2 PT3 1/3",spin_cases({"ppch","ppac","vpaa","avaa"}));
    H1_T1_C1(C1,T1_,1.0,O1_);
    H1_T2_C1(C1,T2_,1.0,O1_);
    H2_T1_C1(C2,T1_,1.0,O1_);
    H2_T2_C1(C2,T2_,1.0,O1_);
    H1_T2_C2(C1,T2_,1.0,O2_);
    H2_T1_C2(C2,T1_,1.0,O2_);
    H2_T2_C2(C2,T2_,1.0,O2_);

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
    O1_["ai"] += temp1["ia"];
    O1_["AI"] += temp1["IA"];
    O2_["abij"] += temp2["ijab"];
    O2_["aBiJ"] += temp2["iJaB"];
    O2_["ABIJ"] += temp2["IJAB"];

    // compute -1.0 / 12.0 * [[[H0th,A1st],A1st],A1st]
    double Ereturn = 0.0;
    double factor  = 1.0 / 6.0;
    H1_T1_C0(O1_,T1_,factor,Ereturn);
    H1_T2_C0(O1_,T2_,factor,Ereturn);
    H2_T1_C0(O2_,T1_,factor,Ereturn);
    H2_T2_C0(O2_,T2_,factor,Ereturn);

    outfile->Printf("  Done. Timing %10.3f s", t1.get());

    if(relax_ref_ != "NONE"){
        Timer t2;
        str = "Computing integrals for ref. relaxation";
        outfile->Printf("\n    %-40s ...", str.c_str());

        factor = 1.0 / 12.0;
        C1 = BTF_->build(tensor_type_,"C1",spin_cases({"aa"}));
        C2 = BTF_->build(tensor_type_,"C2",spin_cases({"aaaa"}));
        H1_T1_C1(O1_,T1_,factor,C1);
        H1_T2_C1(O1_,T2_,factor,C1);
        H2_T1_C1(O2_,T1_,factor,C1);
        H2_T2_C1(O2_,T2_,factor,C1);
        H1_T2_C2(O1_,T2_,factor,C2);
        H2_T1_C2(O2_,T1_,factor,C2);
        H2_T2_C2(O2_,T2_,factor,C2);

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
    // Step 1: compute 0.5 * [H1st + Hbar1st, A1st] = [H1st, A1st] + 0.5 * [[H0th, A1st], A1st]
    //     a) keep a copy of H1st + Hbar1st
    BlockedTensor O1 = BTF_->build(tensor_type_,"O1 pt3 2/3",spin_cases({"pc","va"}));
    BlockedTensor O2 = BTF_->build(tensor_type_,"O2 pt3 2/3",spin_cases({"ppch","ppac","vpaa","avaa"}));
    O1["ai"] = F_["ai"];
    O1["AI"] = F_["AI"];
    O2["abij"] = V_["abij"];
    O2["aBiJ"] = V_["aBiJ"];
    O2["ABIJ"] = V_["ABIJ"];

    //     b) scale -[[H0th, A1st], A1st] by -0.5, computed in compute_energy_pt3_1
    O1_.scale(-0.5);
    O2_.scale(-0.5);

    //     c) prepare V and F
    F_.zero();
    V_.zero();
    F_["ai"] = O1_["ai"];
    F_["AI"] = O1_["AI"];
    V_["abij"] = O2_["abij"];
    V_["aBiJ"] = O2_["aBiJ"];
    V_["ABIJ"] = O2_["ABIJ"];

    //     d) compute contraction from one-body term (first-order bare Fock)
    H1_T1_C1(F1st_,T1_,1.0,F_);
    H1_T2_C1(F1st_,T2_,1.0,F_);
    H1_T2_C2(F1st_,T2_,1.0,V_);

    O1_ = BTF_->build(tensor_type_,"HP2 pt3 2/3",spin_cases({"cp","av"}));
    O2_ = BTF_->build(tensor_type_,"HP2 pt3 2/3",spin_cases({"chpp","acpp","aavp","aaav"}));
    H1_T1_C1(F1st_,T1_,1.0,O1_);
    H1_T2_C1(F1st_,T2_,1.0,O1_);
    H1_T2_C2(F1st_,T2_,1.0,O2_);

    F_["ai"] += O1_["ia"];
    F_["AI"] += O1_["IA"];
    V_["abij"] += O2_["ijab"];
    V_["aBiJ"] += O2_["iJaB"];
    V_["ABIJ"] += O2_["IJAB"];

    //     e) compute contraction in batches of spin cases
    if(eri_df_) {
        // pphh part
        V_T1_C1_DF(B_,T1_,1.0,F_);
        V_T2_C1_DF(B_,T2_,1.0,F_);
        V_T1_C2_DF(B_,T1_,1.0,V_);
        V_T2_C2_DF(B_,T2_,1.0,V_);

        // hhpp part
        O1_.zero();
        O2_.zero();
        V_T1_C1_DF(B_,T1_,1.0,O1_);
        V_T2_C1_DF(B_,T2_,1.0,O1_);
        V_T1_C2_DF(B_,T1_,1.0,O2_);
        V_T2_C2_DF(B_,T2_,1.0,O2_);

        F_["ai"] += O1_["ia"];
        F_["AI"] += O1_["IA"];
        V_["abij"] += O2_["ijab"];
        V_["aBiJ"] += O2_["iJaB"];
        V_["ABIJ"] += O2_["IJAB"];
    } else {
        for(const std::string& block: {"gggg", "gGgG", "GGGG"}){
            BlockedTensor C2 = BTF_->build(tensor_type_,"C2 pt3 2/3",{block});
            C2.iterate([&](const std::vector<size_t>& i,const std::vector<SpinType>& spin,double& value){
                if ((spin[0] == AlphaSpin) and (spin[1] == AlphaSpin)) value = ints_->aptei_aa(i[0],i[1],i[2],i[3]);
                if ((spin[0] == AlphaSpin) and (spin[1] == BetaSpin) ) value = ints_->aptei_ab(i[0],i[1],i[2],i[3]);
                if ((spin[0] == BetaSpin)  and (spin[1] == BetaSpin) ) value = ints_->aptei_bb(i[0],i[1],i[2],i[3]);
            });

            // pphh part
            H2_T1_C1(C2,T1_,1.0,F_);
            H2_T2_C1(C2,T2_,1.0,F_);
            H2_T1_C2(C2,T1_,1.0,V_);
            H2_T2_C2(C2,T2_,1.0,V_);

            // hhpp part
            O1_.zero();
            O2_.zero();
            H2_T1_C1(C2,T1_,1.0,O1_);
            H2_T2_C1(C2,T2_,1.0,O1_);
            H2_T1_C2(C2,T1_,1.0,O2_);
            H2_T2_C2(C2,T2_,1.0,O2_);

            F_["ai"] += O1_["ia"];
            F_["AI"] += O1_["IA"];
            V_["abij"] += O2_["ijab"];
            V_["aBiJ"] += O2_["iJaB"];
            V_["ABIJ"] += O2_["IJAB"];
        }
    }
    outfile->Printf("  Done. Timing %10.3f s", t1.get());

    // Step 2: compute amplitdes
    //     a) save 1st-order amplitudes for later use
    O1_.set_name("T1 1st");
    O2_.set_name("T2 1st");
    O1_["ia"] = T1_["ia"];
    O1_["IA"] = T1_["IA"];
    O2_["ijab"] = T2_["ijab"];
    O2_["iJaB"] = T2_["iJaB"];
    O2_["IJAB"] = T2_["IJAB"];

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
    H1_T1_C0(F_,O1_,1.0,Ereturn);
    H1_T2_C0(F_,O2_,1.0,Ereturn);
    H2_T1_C0(V_,O1_,1.0,Ereturn);
    H2_T2_C0(V_,O2_,1.0,Ereturn);
    outfile->Printf("  Done. Timing %10.3f s", t1.get());

    // relax reference
    if(relax_ref_ != "NONE"){
        Timer t2;
        str = "Computing integrals for ref. relaxation";
        outfile->Printf("\n    %-40s ...", str.c_str());

        BlockedTensor C1 = BTF_->build(tensor_type_,"C1",spin_cases({"aa"}));
        BlockedTensor C2 = BTF_->build(tensor_type_,"C2",spin_cases({"aaaa"}));
        H1_T1_C1(F_,O1_,0.5,C1);
        H1_T2_C1(F_,O2_,0.5,C1);
        H2_T1_C1(V_,O1_,0.5,C1);
        H2_T2_C1(V_,O2_,0.5,C1);
        H1_T2_C2(F_,O2_,0.5,C2);
        H2_T1_C2(V_,O1_,0.5,C2);
        H2_T2_C2(V_,O2_,0.5,C2);

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

double DSRG_MRPT3::compute_energy_multi_state(){
    // compute DSRG-MRPT3 energy
    compute_energy();

    // transfer integrals
    transfer_integrals();

    // prepare FCI integrals
    std::shared_ptr<FCIIntegrals> fci_ints = std::make_shared<FCIIntegrals>(ints_, aactv_mos_, acore_mos_);
    ambit::Tensor tei_active_aa = ints_->aptei_aa_block(aactv_mos_, aactv_mos_, aactv_mos_, aactv_mos_);
    ambit::Tensor tei_active_ab = ints_->aptei_ab_block(aactv_mos_, aactv_mos_, aactv_mos_, aactv_mos_);
    ambit::Tensor tei_active_bb = ints_->aptei_bb_block(aactv_mos_, aactv_mos_, aactv_mos_, aactv_mos_);
    fci_ints->set_active_integrals(tei_active_aa, tei_active_ab, tei_active_bb);
    fci_ints->compute_restricted_one_body_operator();

    // get character table
    CharacterTable ct = Process::environment.molecule()->point_group()->char_table();
    std::vector<std::string> irrep_symbol;
    for(int h = 0; h < this->nirrep(); ++h){
        irrep_symbol.push_back(std::string(ct.gamma(h).symbol()));
    }

    // multiplicity table
    std::vector<std::string> multi_label{"Singlet","Doublet","Triplet","Quartet","Quintet","Sextet","Septet","Octet",
                                  "Nonet","Decaet","11-et","12-et","13-et","14-et","15-et","16-et","17-et","18-et",
                                  "19-et","20-et","21-et","22-et","23-et","24-et"};

    // size of 1rdm and 2rdm
    size_t na = mo_space_info_->size("ACTIVE");
    size_t nele1 = na * na;
    size_t nele2 = na * nele1;

    // get effective one-electron integral (DSRG transformed)
    BlockedTensor oei = BTF_->build(tensor_type_,"temp1",spin_cases({"aa"}));
    oei.block("aa").data() = fci_ints->oei_a_vector();
    oei.block("AA").data() = fci_ints->oei_b_vector();

    // get nuclear repulsion energy
    boost::shared_ptr<Molecule> molecule = Process::environment.molecule();
    double Enuc = molecule->nuclear_repulsion_energy();

    // loop over entries of AVG_STATE
    print_h2("Diagonalize Effective Hamiltonian");
    outfile->Printf("\n");
    int nentry = eigens_.size();
    std::vector<std::vector<double>> Edsrg_sa (nentry, std::vector<double> ());
    for(int n = 0; n < nentry; ++n){
        int irrep = options_["AVG_STATE"][n][0].to_integer();
        int multi = options_["AVG_STATE"][n][1].to_integer();
        int nstates = options_["AVG_STATE"][n][2].to_integer();

        int dim = (eigens_[n][0].first)->dim();
        SharedMatrix evecs (new Matrix("evecs",dim,dim));
        for(int i = 0; i < eigens_[n].size(); ++i){
            evecs->set_column(0,i,(eigens_[n][i]).first);
        }

        SharedMatrix Heff (new Matrix("Heff " + multi_label[multi - 1]
                           + " " + irrep_symbol[irrep], nstates, nstates));
        for(int A = 0; A < nstates; ++A){
            for(int B = A; B < nstates; ++B){

                // compute rdms
                CI_RDMS ci_rdms (options_,fci_ints,p_space_,evecs,A,B);
                ci_rdms.set_symmetry(irrep);

                std::vector<double> opdm_a (nele1, 0.0);
                std::vector<double> opdm_b (nele1, 0.0);
                ci_rdms.compute_1rdm(opdm_a,opdm_b);

                std::vector<double> tpdm_aa (nele2, 0.0);
                std::vector<double> tpdm_ab (nele2, 0.0);
                std::vector<double> tpdm_bb (nele2, 0.0);
                ci_rdms.compute_2rdm(tpdm_aa,tpdm_ab,tpdm_bb);

                // put rdms in tensor format
                BlockedTensor D1 = BTF_->build(tensor_type_,"D1",spin_cases({"aa"}),true);
                D1.block("aa").data() = opdm_a;
                D1.block("AA").data() = opdm_b;

                BlockedTensor D2 = BTF_->build(tensor_type_,"D2",spin_cases({"aaaa"}),true);
                D2.block("aaaa").data() = tpdm_aa;
                D2.block("aAaA").data() = tpdm_ab;
                D2.block("AAAA").data() = tpdm_bb;

                double H_AB = 0.0;
                H_AB += oei["uv"] * D1["uv"];
                H_AB += oei["UV"] * D1["UV"];
                H_AB += 0.25 * Hbar2_["uvxy"] * D2["xyuv"];
                H_AB += 0.25 * Hbar2_["UVXY"] * D2["XYUV"];
                H_AB += Hbar2_["uVxY"] * D2["xYuV"];

                if(A == B){
                    H_AB += ints_->frozen_core_energy() + fci_ints->scalar_energy() + Enuc;
                    Heff->set(A,B,H_AB);
                } else {
                    Heff->set(A,B,H_AB);
                    Heff->set(B,A,H_AB);
                }

            }
        } // end forming effective Hamiltonian

        Heff->print();

        SharedMatrix U (new Matrix("U of Heff", nstates, nstates));
        SharedVector Ems (new Vector("MS Energies", nstates));
        Heff->diagonalize(U, Ems);

        // fill in Edsrg_sa
        for(int i = 0; i < nstates; ++i){
            Edsrg_sa[n].push_back(Ems->get(i));
        }

    } // end looping averaged states

    // energy summuary
    print_h2("Multi-State DSRG-MRPT3 Energy Summary");

    outfile->Printf("\n    Multi.  Irrep.  No.   DSRG-MRPT2 Energy");
    std::string dash(41, '-');
    outfile->Printf("\n    %s", dash.c_str());

    for(int n = 0; n < nentry; ++n){
        int irrep = options_["AVG_STATE"][n][0].to_integer();
        int multi = options_["AVG_STATE"][n][1].to_integer();
        int nstates = options_["AVG_STATE"][n][2].to_integer();

        for(int i = 0; i < nstates; ++i){
            outfile->Printf("\n     %3d     %3s    %3d  %20.15f",
                            multi, irrep_symbol[irrep].c_str(), i, Edsrg_sa[n][i]);
        }
        outfile->Printf("\n    %s", dash.c_str());
    }

    Process::environment.globals["CURRENT ENERGY"] = Edsrg_sa[0][0];
    return Edsrg_sa[0][0];
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

        // nroot and root
        int root  = options_.get_int("ROOT");
        int nroot = options_.get_int("NROOT");

        // diagonalize the Hamiltonian
        FCISolver fcisolver(active_dim,acore_mos_,aactv_mos_,na,nb,multi,options_.get_int("ROOT_SYM"),ints_, mo_space_info_,
                                             options_.get_int("NTRIAL_PER_ROOT"),print_, options_);
        fcisolver.set_max_rdm_level(1);
        fcisolver.set_nroot(nroot);
        fcisolver.set_root(root);
        fcisolver.set_fci_iterations(options_.get_int("FCI_ITERATIONS"));
        fcisolver.set_collapse_per_root(options_.get_int("DAVIDSON_COLLAPSE_PER_ROOT"));
        fcisolver.set_subspace_per_root(options_.get_int("DAVIDSON_SUBSPACE_PER_ROOT"));

        // create FCIIntegrals manually
        if(eri_df_){
            std::shared_ptr<FCIIntegrals> fci_ints;
            fci_ints = std::make_shared<FCIIntegrals>(ints_, aactv_mos_, acore_mos_);
            fcisolver.use_user_integrals_and_restricted_docc(true);
            fci_ints->set_active_integrals(Hbar2_.block("aaaa"), Hbar2_.block("aAaA"),Hbar2_.block("AAAA"));
            fci_ints->set_restricted_one_body_operator(aone_eff_,bone_eff_);
            fci_ints->set_scalar_energy(ints_->scalar());
            fcisolver.set_integral_pointer(fci_ints);
        }

        Erelax = fcisolver.compute_energy();

        // test the overlap of the relaxed wfn with original ref. wfn
        if(fciwfn0_->size() != 0){
            fciwfn_ = fcisolver.get_FCIWFN();
            double overlap = fciwfn0_->dot(fciwfn_);
            const double threshold = 0.90;
            int more_roots = (fciwfn0_->size() < 5) ? fciwfn0_->size() : 5;

            if(fabs(overlap) > threshold){
                outfile->Printf("\n    Overlap: <Phi0_unrelaxed|Phi0_relaxed> = %4.3f.", fabs(overlap));
            } else {
                outfile->Printf("\n    Warning: overlap <Phi0_unrelaxed|Phi0_relaxed> = %4.3f < %.2f.", fabs(overlap), threshold);
                outfile->Printf("\n    FCI seems to find the wrong root. Try %d more roots.", more_roots);

                bool find = false;
                double Etemp = 0.0;
                for(int i = 0; i < more_roots; ++i){
                    if(root < nroot -1){
                        ++root;
                    } else {
                        ++root;
                        ++nroot;
                    }
                    fcisolver.set_nroot(nroot);
                    fcisolver.set_root(root);
                    Etemp = fcisolver.compute_energy();

                    fciwfn_ = fcisolver.get_FCIWFN();
                    overlap = fciwfn0_->dot(fciwfn_);

                    outfile->Printf("\n    Current overlap <Phi0_unrelaxed|Phi0_relaxed> = %4.3f.", fabs(overlap));
                    if(fabs(overlap) >= threshold){
                        outfile->Printf("\n    This root seems to be OK. Stop trying more roots.");
                        find = true;
                        break;
                    } else {
                        outfile->Printf("\n    Keep looking for better overlap. Tries left: %d.", more_roots - 1 - i);
                    }
                }

                if(find){
                    Erelax = Etemp;
                } else {
                    outfile->Printf("\n    Fatal Warning: Do not find root of enough overlap (> %.2f) with original FCI wavefunction.", threshold);
                    outfile->Printf("\n    The DSRG-MRPT3 relaxed energy might be meaningless.");
                }
            }

        }

        // printing
        print_h2("MRDSRG Energy Summary");
        outfile->Printf("\n    %-35s = %22.15f", "DSRG-MRPT3 Total Energy (fixed)", Edsrg);
        outfile->Printf("\n    %-35s = %22.15f", "DSRG-MRPT3 Total Energy (relaxed)", Erelax);
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
    if(eri_df_){
        aone_eff_ = temp1.block("aa").data();
        bone_eff_ = temp1.block("AA").data();
    }
    outfile->Printf("  Done. Timing %10.3f s", t_one.get());

    // update integrals
    Timer t_int;
    str = "Updating integrals";
    outfile->Printf("\n    %-40s ...", str.c_str());
    ints_->set_scalar(scalar);

    if(!eri_df_){
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

        Hbar2_.citerate([&](const std::vector<size_t>& i,const std::vector<SpinType>& spin,const double& value){
            if ((spin[0] == AlphaSpin) && (spin[1] == AlphaSpin)){
                ints_->set_tei(i[0],i[1],i[2],i[3],value,true,true);
            }else if ((spin[0] == AlphaSpin) && (spin[1] == BetaSpin)){
                ints_->set_tei(i[0],i[1],i[2],i[3],value,true,false);
            }else if ((spin[0] == BetaSpin)  && (spin[1] == BetaSpin)){
                ints_->set_tei(i[0],i[1],i[2],i[3],value,false,false);
            }
        });
    }
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
        if(!eri_df_){
            ints_->update_integrals(false);
        }
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
    C2["qjsb"] += alpha * H2["aqms"] * T2["mjab"];
    C2["qjsb"] += alpha * H2["qAsM"] * T2["jMbA"];
    C2["qjsb"] += alpha * Gamma1_["xy"] * T2["yjab"] * H2["aqxs"];
    C2["qjsb"] += alpha * Gamma1_["XY"] * T2["jYbA"] * H2["qAsX"];
    C2["qjsb"] -= alpha * Gamma1_["xy"] * H2["yqis"] * T2["ijxb"];
    C2["qjsb"] -= alpha * Gamma1_["XY"] * H2["qYsI"] * T2["jIbX"];

    C2["jqsb"] -= alpha * H2["aqms"] * T2["mjab"];
    C2["jqsb"] -= alpha * H2["qAsM"] * T2["jMbA"];
    C2["jqsb"] -= alpha * Gamma1_["xy"] * T2["yjab"] * H2["aqxs"];
    C2["jqsb"] -= alpha * Gamma1_["XY"] * T2["jYbA"] * H2["qAsX"];
    C2["jqsb"] += alpha * Gamma1_["xy"] * H2["yqis"] * T2["ijxb"];
    C2["jqsb"] += alpha * Gamma1_["XY"] * H2["qYsI"] * T2["jIbX"];

    C2["qjbs"] -= alpha * H2["aqms"] * T2["mjab"];
    C2["qjbs"] -= alpha * H2["qAsM"] * T2["jMbA"];
    C2["qjbs"] -= alpha * Gamma1_["xy"] * T2["yjab"] * H2["aqxs"];
    C2["qjbs"] -= alpha * Gamma1_["XY"] * T2["jYbA"] * H2["qAsX"];
    C2["qjbs"] += alpha * Gamma1_["xy"] * H2["yqis"] * T2["ijxb"];
    C2["qjbs"] += alpha * Gamma1_["XY"] * H2["qYsI"] * T2["jIbX"];

    C2["jqbs"] += alpha * H2["aqms"] * T2["mjab"];
    C2["jqbs"] += alpha * H2["qAsM"] * T2["jMbA"];
    C2["jqbs"] += alpha * Gamma1_["xy"] * T2["yjab"] * H2["aqxs"];
    C2["jqbs"] += alpha * Gamma1_["XY"] * T2["jYbA"] * H2["qAsX"];
    C2["jqbs"] -= alpha * Gamma1_["xy"] * H2["yqis"] * T2["ijxb"];
    C2["jqbs"] -= alpha * Gamma1_["XY"] * H2["qYsI"] * T2["jIbX"];

    C2["QJSB"] += alpha * H2["AQMS"] * T2["MJAB"];
    C2["QJSB"] += alpha * H2["aQmS"] * T2["mJaB"];
    C2["QJSB"] += alpha * Gamma1_["XY"] * T2["YJAB"] * H2["AQXS"];
    C2["QJSB"] += alpha * Gamma1_["xy"] * T2["yJaB"] * H2["aQxS"];
    C2["QJSB"] -= alpha * Gamma1_["XY"] * H2["YQIS"] * T2["IJXB"];
    C2["QJSB"] -= alpha * Gamma1_["xy"] * H2["yQiS"] * T2["iJxB"];

    C2["JQSB"] -= alpha * H2["AQMS"] * T2["MJAB"];
    C2["JQSB"] -= alpha * H2["aQmS"] * T2["mJaB"];
    C2["JQSB"] -= alpha * Gamma1_["XY"] * T2["YJAB"] * H2["AQXS"];
    C2["JQSB"] -= alpha * Gamma1_["xy"] * T2["yJaB"] * H2["aQxS"];
    C2["JQSB"] += alpha * Gamma1_["XY"] * H2["YQIS"] * T2["IJXB"];
    C2["JQSB"] += alpha * Gamma1_["xy"] * H2["yQiS"] * T2["iJxB"];

    C2["QJBS"] -= alpha * H2["AQMS"] * T2["MJAB"];
    C2["QJBS"] -= alpha * H2["aQmS"] * T2["mJaB"];
    C2["QJBS"] -= alpha * Gamma1_["XY"] * T2["YJAB"] * H2["AQXS"];
    C2["QJBS"] -= alpha * Gamma1_["xy"] * T2["yJaB"] * H2["aQxS"];
    C2["QJBS"] += alpha * Gamma1_["XY"] * H2["YQIS"] * T2["IJXB"];
    C2["QJBS"] += alpha * Gamma1_["xy"] * H2["yQiS"] * T2["iJxB"];

    C2["JQBS"] += alpha * H2["AQMS"] * T2["MJAB"];
    C2["JQBS"] += alpha * H2["aQmS"] * T2["mJaB"];
    C2["JQBS"] += alpha * Gamma1_["XY"] * T2["YJAB"] * H2["AQXS"];
    C2["JQBS"] += alpha * Gamma1_["xy"] * T2["yJaB"] * H2["aQxS"];
    C2["JQBS"] -= alpha * Gamma1_["XY"] * H2["YQIS"] * T2["IJXB"];
    C2["JQBS"] -= alpha * Gamma1_["xy"] * H2["yQiS"] * T2["iJxB"];

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

void DSRG_MRPT3::V_T1_C1_DF(BlockedTensor& B, BlockedTensor& T1, const double& alpha, BlockedTensor& C1){
    Timer timer;

    BlockedTensor temp = BTF_->build(tensor_type_,"temp VT1->C1 DF",{"L"},true);
    temp["g"] += T1["ma"] * B["gam"];
    temp["g"] += T1["xe"] * Gamma1_["yx"] * B["gey"];
    temp["g"] -= T1["mu"] * Gamma1_["uv"] * B["gvm"];
    temp["g"] += T1["MA"] * B["gAM"];
    temp["g"] += T1["XE"] * Gamma1_["YX"] * B["gEY"];
    temp["g"] -= T1["MU"] * Gamma1_["UV"] * B["gVM"];

    C1["qp"] += alpha * temp["g"] * B["gqp"];
    C1["QP"] += alpha * temp["g"] * B["gQP"];

    temp = BTF_->build(tensor_type_,"temp VT1->C1 DF",{"Lgh"},true);
    temp["gpm"] -= T1["ma"] * B["gap"];
    temp["gpy"] -= T1["xe"] * Gamma1_["yx"] * B["gep"];
    temp["gpm"] += T1["mu"] * Gamma1_["uv"] * B["gvp"];

    C1["qp"] += alpha * temp["gpm"] * B["gqm"];
    C1["qp"] += alpha * temp["gpy"] * B["gqy"];

    temp = BTF_->build(tensor_type_,"temp VT1->C1 DF",{"LGH"},true);
    temp["gPM"] -= T1["MA"] * B["gAP"];
    temp["gPY"] -= T1["XE"] * Gamma1_["YX"] * B["gEP"];
    temp["gPM"] += T1["MU"] * Gamma1_["UV"] * B["gVP"];

    C1["QP"] += alpha * temp["gPM"] * B["gQM"];
    C1["QP"] += alpha * temp["gPY"] * B["gQY"];

    if(print_ > 2){
        outfile->Printf("\n    Time for [H2, T1] -> C1 : %12.3f",timer.get());
    }
    dsrg_time_.add("211",timer.get());
}

void DSRG_MRPT3::V_T1_C2_DF(BlockedTensor& B, BlockedTensor& T1, const double& alpha, BlockedTensor& C2){
    Timer timer;

    BlockedTensor temp = BTF_->build(tensor_type_,"temp VT1->C2 DF",{"Lhg"},true);
    temp["gip"] = T1["ia"] * B["gpa"];
    C2["irpq"] += alpha * temp["gip"] * B["grq"];
    C2["irqp"] -= alpha * temp["gip"] * B["grq"];
    C2["riqp"] += alpha * temp["gip"] * B["grq"];
    C2["ripq"] -= alpha * temp["gip"] * B["grq"];
    C2["iRpQ"] += alpha * temp["gip"] * B["gRQ"];

    temp = BTF_->build(tensor_type_,"temp VT1->C2 DF",{"Lgp"},true);
    temp["gra"] = T1["ia"] * B["gri"];
    C2["rsaq"] -= alpha * temp["gra"] * B["gsq"];
    C2["sraq"] += alpha * temp["gra"] * B["gsq"];
    C2["srqa"] -= alpha * temp["gra"] * B["gsq"];
    C2["rsqa"] += alpha * temp["gra"] * B["gsq"];
    C2["rSaQ"] -= alpha * temp["gra"] * B["gSQ"];

    temp = BTF_->build(tensor_type_,"temp VT1->C2 DF",{"LHG"},true);
    temp["gIP"] = T1["IA"] * B["gPA"];
    C2["IRPQ"] += alpha * temp["gIP"] * B["gRQ"];
    C2["IRQP"] -= alpha * temp["gIP"] * B["gRQ"];
    C2["RIQP"] += alpha * temp["gIP"] * B["gRQ"];
    C2["RIPQ"] -= alpha * temp["gIP"] * B["gRQ"];
    C2["rIqP"] += alpha * temp["gIP"] * B["grq"];

    temp = BTF_->build(tensor_type_,"temp VT1->C2 DF",{"LGP"},true);
    temp["gRA"] = T1["IA"] * B["gRI"];
    C2["RSAQ"] -= alpha * temp["gRA"] * B["gSQ"];
    C2["SRAQ"] += alpha * temp["gRA"] * B["gSQ"];
    C2["SRQA"] -= alpha * temp["gRA"] * B["gSQ"];
    C2["RSQA"] += alpha * temp["gRA"] * B["gSQ"];
    C2["sRpA"] -= alpha * temp["gRA"] * B["gsp"];

    if(print_ > 2){
        outfile->Printf("\n    Time for [H2, T1] -> C2 : %12.3f",timer.get());
    }
    dsrg_time_.add("212",timer.get());
}

void DSRG_MRPT3::V_T2_C1_DF(BlockedTensor& B, BlockedTensor& T2, const double& alpha, BlockedTensor& C1){
    Timer timer;
    BlockedTensor temp;

    // [Hbar2, T2] (C_2)^3 -> C1 particle contractions
    start_ = std::chrono::system_clock::now();
    tt1_ = std::chrono::system_clock::to_time_t(start_);
    if(profile_print_){
        outfile->Printf("\n    [V, T2] (C_2)^3 -> C1 P started: %s", std::ctime(&tt1_));
    }

    temp = BTF_->build(tensor_type_,"temp VT2->C1 DF",{"Lhp"},true);
    temp["gia"] += B["gmb"] * T2["imab"];
    temp["gia"] += B["gMB"] * T2["iMaB"];
    temp["gia"] += Gamma1_["uv"] * B["gbu"] * T2["ivab"];
    temp["gia"] += Gamma1_["UV"] * B["gBU"] * T2["iVaB"];
    temp["gia"] += Gamma1_["uv"] * B["gmv"] * T2["imua"];
    temp["gia"] -= Gamma1_["UV"] * B["gMV"] * T2["iMaU"];
    temp["gia"] += B["gvx"] * Gamma1_["uv"] * Gamma1_["xy"] * T2["yiau"];
    temp["gia"] -= B["gVX"] * Gamma1_["XY"] * Gamma1_["UV"] * T2["iYaU"];
    C1["ir"] += alpha * temp["gia"] * B["gra"];

    temp = BTF_->build(tensor_type_,"temp VT2->C1 DF",{"LHP"},true);
    temp["gIA"] += B["gMB"] * T2["IMAB"];
    temp["gIA"] += B["gbm"] * T2["mIbA"];
    temp["gIA"] += Gamma1_["UV"] * B["gBU"] * T2["IVAB"];
    temp["gIA"] += Gamma1_["uv"] * B["gbu"] * T2["vIbA"];
    temp["gIA"] += Gamma1_["UV"] * B["gVM"] * T2["MIAU"];
    temp["gIA"] -= Gamma1_["uv"] * B["gvm"] * T2["mIuA"];
    temp["gIA"] += B["gVX"] * Gamma1_["UV"] * Gamma1_["XY"] * T2["IYUA"];
    temp["gIA"] -= B["gvx"] * Gamma1_["uv"] * Gamma1_["xy"] * T2["yIuA"];
    C1["IR"] += alpha * temp["gIA"] * B["gRA"];

    temp = BTF_->build(tensor_type_,"temp VT2->C1 DF",{"Lha"},true);
    temp["giv"] += T2["jixu"] * Gamma1_["xy"] * Gamma1_["uv"] * B["gyj"];
    temp["giv"] += T2["iJuX"] * Gamma1_["XY"] * Gamma1_["uv"] * B["gJY"];
    temp["giv"] += Gamma1_["uv"] * B["gmb"] * T2["miub"];
    temp["giv"] -= Gamma1_["uv"] * B["gMB"] * T2["iMuB"];
    temp["giv"] -= B["gbx"] * T2["yibu"] * Gamma1_["uv"] * Gamma1_["xy"];
    temp["giv"] -= B["gXB"] * T2["iYuB"] * Gamma1_["uv"] * Gamma1_["XY"];
    C1["ir"] += alpha * temp["giv"] * B["grv"];

    temp = BTF_->build(tensor_type_,"temp VT2->C1 DF",{"LHA"},true);
    temp["gIV"] += T2["JIXU"] * Gamma1_["XY"] * Gamma1_["UV"] * B["gYJ"];
    temp["gIV"] += T2["jIxU"] * Gamma1_["UV"] * Gamma1_["xy"] * B["gyj"];
    temp["gIV"] += Gamma1_["UV"] * B["gMB"] * T2["MIUB"];
    temp["gIV"] -= Gamma1_["UV"] * B["gbm"] * T2["mIbU"];
    temp["gIV"] -= B["gXB"] * T2["IYUB"] * Gamma1_["UV"] * Gamma1_["XY"];
    temp["gIV"] -= B["gbx"] * T2["yIbU"] * Gamma1_["UV"] * Gamma1_["xy"];
    C1["IR"] += alpha * temp["gIV"] * B["gRV"];

    end_ = std::chrono::system_clock::now();
    tt2_ = std::chrono::system_clock::to_time_t(end_);
    if(profile_print_){
        outfile->Printf("    [V, T2] (C_2)^3 -> C1 P ended:   %s", std::ctime(&tt2_));
        outfile->Printf("    [V, T2] (C_2)^3 -> C1 P wall time %.1f s.",
                        compute_elapsed_time(start_,end_).count());
    }

    // [Hbar2, T2] (C_2)^3 -> C1 hole contractions
    start_ = std::chrono::system_clock::now();
    tt1_ = std::chrono::system_clock::to_time_t(start_);
    if(profile_print_){
        outfile->Printf("\n    [V, T2] (C_2)^3 -> C1 H started: %s", std::ctime(&tt1_));
    }

    temp = BTF_->build(tensor_type_,"temp VT2->C1 DF",{"Lph"},true);
    temp["gai"] += B["gje"] * T2["jiae"];
    temp["gai"] -= B["gJE"] * T2["iJaE"];
    temp["gai"] += B["gvj"] * T2["jiau"] * Eta1_["uv"];
    temp["gai"] -= B["gJV"] * T2["iJaU"] * Eta1_["UV"];
    temp["gai"] -= B["gue"] * T2["viae"] * Eta1_["uv"];
    temp["gai"] += B["gUE"] * T2["iVaE"] * Eta1_["UV"];
    temp["gai"] -= B["gyu"] * T2["viax"] * Eta1_["uv"] * Eta1_["xy"];
    temp["gai"] += B["gYU"] * T2["iVaX"] * Eta1_["XY"] * Eta1_["UV"];
    C1["pa"] += alpha * temp["gai"] * B["gpi"];

    temp = BTF_->build(tensor_type_,"temp VT2->C1 DF",{"LPH"},true);
    temp["gAI"] += B["gJE"] * T2["JIAE"];
    temp["gAI"] -= B["gej"] * T2["jIeA"];
    temp["gAI"] += B["gVJ"] * T2["JIAU"] * Eta1_["UV"];
    temp["gAI"] -= B["gvj"] * T2["jIuA"] * Eta1_["uv"];
    temp["gAI"] -= B["gUE"] * T2["VIAE"] * Eta1_["UV"];
    temp["gAI"] += B["geu"] * T2["vIeA"] * Eta1_["uv"];
    temp["gAI"] -= B["gYU"] * T2["VIAX"] * Eta1_["UV"] * Eta1_["XY"];
    temp["gAI"] += B["gyu"] * T2["vIxA"] * Eta1_["uv"] * Eta1_["xy"];
    C1["PA"] += alpha * temp["gAI"] * B["gPI"];

    temp = BTF_->build(tensor_type_,"temp VT2->C1 DF",{"Lpa"},true);
    temp["gau"] -= B["gbx"] * T2["vyab"] * Eta1_["uv"] * Eta1_["xy"];
    temp["gau"] -= B["gBX"] * T2["vYaB"] * Eta1_["uv"] * Eta1_["XY"];
    temp["gau"] += B["gje"] * T2["vjae"] * Eta1_["uv"];
    temp["gau"] += B["gJE"] * T2["vJaE"] * Eta1_["uv"];
    temp["gau"] += B["gjy"] * T2["vjax"] * Eta1_["uv"] * Eta1_["xy"];
    temp["gau"] += B["gJY"] * T2["vJaX"] * Eta1_["uv"] * Eta1_["XY"];
    C1["pa"] += alpha * temp["gau"] * B["gpu"];

    temp = BTF_->build(tensor_type_,"temp VT2->C1 DF",{"LPA"},true);
    temp["gAU"] -= B["gBX"] * Eta1_["XY"] * T2["VYAB"] * Eta1_["UV"];
    temp["gAU"] -= B["gbx"] * Eta1_["xy"] * T2["yVbA"] * Eta1_["UV"];
    temp["gAU"] += B["gJE"] * T2["VJAE"] * Eta1_["UV"];
    temp["gAU"] += B["gej"] * T2["jVeA"] * Eta1_["UV"];
    temp["gAU"] += B["gYJ"] * T2["VJAX"] * Eta1_["UV"] * Eta1_["XY"];
    temp["gAU"] += B["gyj"] * T2["jVxA"] * Eta1_["UV"] * Eta1_["xy"];
    C1["PA"] += alpha * temp["gAU"] * B["gPU"];

    end_ = std::chrono::system_clock::now();
    tt2_ = std::chrono::system_clock::to_time_t(end_);
    if(profile_print_){
        outfile->Printf("    [V, T2] (C_2)^3 -> C1 H ended:   %s", std::ctime(&tt2_));
        outfile->Printf("    [V, T2] (C_2)^3 -> C1 H wall time %.1f s.",
                        compute_elapsed_time(start_,end_).count());
    }

    // [Hbar2, T2] C_4 C_2 2:2 -> C1
    start_ = std::chrono::system_clock::now();
    tt1_ = std::chrono::system_clock::to_time_t(start_);
    if(profile_print_){
        outfile->Printf("\n    [V, T2] C_4 C_2 2:2 -> C1 started: %s", std::ctime(&tt1_));
    }

    temp = BTF_->build(tensor_type_,"temp VT2->C1 DF",{"Lha"},true);
    temp["giu"] += 0.5 * T2["ijxy"] * Lambda2_["xyuv"] * B["gjv"];
    temp["giu"] += T2["iJxY"] * Lambda2_["xYuV"] * B["gJV"];
    temp["giu"] -= T2["iVyA"] * Lambda2_["yXuV"] * B["gXA"];
    C1["ir"] += alpha * temp["giu"]* B["gur"];

    temp = BTF_->build(tensor_type_,"temp VT2->C1 DF",{"LHA"},true);
    temp["gIU"] += 0.5 * T2["IJXY"] * Lambda2_["XYUV"] * B["gJV"];
    temp["gIU"] += T2["jIxY"] * Lambda2_["xYvU"] * B["gvj"];
    temp["gIU"] -= T2["vIaY"] * Lambda2_["xYvU"] * B["gax"];
    C1["IR"] += alpha * temp["gIU"]* B["gUR"];

    temp = BTF_->build(tensor_type_,"temp VT2->C1 DF",{"Lpa"},true);
    temp["gax"] -= 0.5 * T2["uvab"] * Lambda2_["yxvu"] * B["gyb"];
    temp["gax"] -= T2["uVaB"] * Lambda2_["xYuV"] * B["gBY"];
    temp["gax"] += T2["vIaY"] * Lambda2_["xYvU"] * B["gIU"];
    C1["pa"] += alpha * temp["gax"]* B["gpx"];

    temp = BTF_->build(tensor_type_,"temp VT2->C1 DF",{"LPA"},true);
    temp["gAX"] -= 0.5 * T2["UVAB"] * Lambda2_["YXVU"] * B["gYB"];
    temp["gAX"] -= T2["uVbA"] * Lambda2_["yXuV"] * B["gby"];
    temp["gAX"] += T2["iVyA"] * Lambda2_["yXuV"] * B["gui"];
    C1["PA"] += alpha * temp["gAX"]* B["gPX"];

    temp = ambit::BlockedTensor::build(tensor_type_,"temp",{"hapa"});
    temp["ixau"] += Lambda2_["xyuv"] * T2["ivay"];
    temp["ixau"] += Lambda2_["xYuV"] * T2["iVaY"];
    C1["ir"] += alpha * temp["ixau"] * B["gar"] * B["gux"];
    C1["ir"] -= alpha * temp["ixau"] * B["gax"] * B["gur"];
    C1["pa"] -= alpha * B["gpi"] * B["gux"] * temp["ixau"];
    C1["pa"] += alpha * B["gpx"] * B["gui"] * temp["ixau"];

    temp = ambit::BlockedTensor::build(tensor_type_,"temp",{"hApA"});
    temp["iXaU"] += Lambda2_["XYUV"] * T2["iVaY"];
    temp["iXaU"] += Lambda2_["yXvU"] * T2["ivay"];
    C1["ir"] += alpha * temp["iXaU"] * B["gar"] * B["gUX"];
    C1["pa"] -= alpha * B["gpi"] * B["gUX"] * temp["iXaU"];

    temp = ambit::BlockedTensor::build(tensor_type_,"temp",{"aHaP"});
    temp["xIuA"] += Lambda2_["xyuv"] * T2["vIyA"];
    temp["xIuA"] += Lambda2_["xYuV"] * T2["VIYA"];
    C1["IR"] += alpha * temp["xIuA"] * B["gux"] * B["gAR"];
    C1["PA"] -= alpha * B["gux"] * B["gPI"] * temp["xIuA"];

    temp = ambit::BlockedTensor::build(tensor_type_,"temp",{"HAPA"});
    temp["IXAU"] += Lambda2_["XYUV"] * T2["IVAY"];
    temp["IXAU"] += Lambda2_["yXvU"] * T2["vIyA"];
    C1["IR"] += alpha * temp["IXAU"] * B["gAR"] * B["gUX"];
    C1["IR"] -= alpha * temp["IXAU"] * B["gAX"] * B["gUR"];
    C1["PA"] -= alpha * B["gPI"] * B["gUX"] * temp["IXAU"];
    C1["PA"] += alpha * B["gPX"] * B["gUI"] * temp["IXAU"];

    end_ = std::chrono::system_clock::now();
    tt2_ = std::chrono::system_clock::to_time_t(end_);
    if(profile_print_){
        outfile->Printf("    [V, T2] C_4 C_2 2:2 -> C1 ended:   %s", std::ctime(&tt2_));
        outfile->Printf("    [V, T2] C_4 C_2 2:2 -> C1 wall time %.1f s.",
                        compute_elapsed_time(start_,end_).count());
    }

    // [Hbar2, T2] C_4 C_2 1:3 -> C1
    start_ = std::chrono::system_clock::now();
    tt1_ = std::chrono::system_clock::to_time_t(start_);
    if(profile_print_){
        outfile->Printf("\n    [V, T2] C_4 C_2 1:3 -> C1 started: %s", std::ctime(&tt1_));
    }

    temp = ambit::BlockedTensor::build(tensor_type_,"temp",{"pa"});
    temp["au"] += Lambda2_["xyuv"] * B["gax"] * B["gyv"];
    temp["au"] += Lambda2_["xYuV"] * B["gax"] * B["gYV"];
    C1["jb"] += alpha * temp["au"] * T2["ujab"];
    C1["JB"] += alpha * temp["au"] * T2["uJaB"];
    temp = ambit::BlockedTensor::build(tensor_type_,"temp",{"PA"});
    temp["AU"] += Lambda2_["XYUV"] * B["gAX"] * B["gYV"];
    temp["AU"] += Lambda2_["xYvU"] * B["gvx"] * B["gAY"];
    C1["jb"] += alpha * temp["AU"] * T2["jUbA"];
    C1["JB"] += alpha * temp["AU"] * T2["UJAB"];

    temp = ambit::BlockedTensor::build(tensor_type_,"temp",{"ah"});
    temp["xi"] += Lambda2_["yxvu"] * B["giu"] * B["gvy"];
    temp["xi"] += Lambda2_["xYuV"] * B["giu"] * B["gYV"];
    C1["jb"] -= alpha * temp["xi"] * T2["ijxb"];
    C1["JB"] -= alpha * temp["xi"] * T2["iJxB"];
    temp = ambit::BlockedTensor::build(tensor_type_,"temp",{"AH"});
    temp["XI"] += Lambda2_["YXVU"] * B["gIU"] * B["gVY"];
    temp["XI"] += Lambda2_["yXvU"] * B["gvy"] * B["gIU"];
    C1["jb"] -= alpha * temp["XI"] * T2["jIbX"];
    C1["JB"] -= alpha * temp["XI"] * T2["IJXB"];

    temp = ambit::BlockedTensor::build(tensor_type_,"temp",{"av"});
    temp["xe"] += 0.5 * T2["uvey"] * Lambda2_["yxvu"];
    temp["xe"] += T2["uVeY"] * Lambda2_["xYuV"];
    C1["qs"] += alpha * temp["xe"] * B["gxe"] * B["gqs"];
    C1["qs"] -= alpha * temp["xe"] * B["gse"] * B["gqx"];
    C1["QS"] += alpha * temp["xe"] * B["gxe"] * B["gQS"];
    temp = ambit::BlockedTensor::build(tensor_type_,"temp",{"AV"});
    temp["XE"] += 0.5 * T2["UVEY"] * Lambda2_["YXVU"];
    temp["XE"] += T2["uVyE"] * Lambda2_["yXuV"];
    C1["qs"] += alpha * temp["XE"] * B["gqs"] * B["gXE"];
    C1["QS"] += alpha * temp["XE"] * B["gXE"] * B["gQS"];
    C1["QS"] -= alpha * temp["XE"] * B["gSE"] * B["gQX"];

    temp = ambit::BlockedTensor::build(tensor_type_,"temp",{"ca"});
    temp["mu"] += 0.5 * T2["mvxy"] * Lambda2_["xyuv"];
    temp["mu"] += T2["mVxY"] * Lambda2_["xYuV"];
    C1["qs"] -= alpha * temp["mu"] * B["gum"] * B["gqs"];
    C1["qs"] += alpha * temp["mu"] * B["gsu"] * B["gqm"];
    C1["QS"] -= alpha * temp["mu"] * B["gum"] * B["gQS"];
    temp = ambit::BlockedTensor::build(tensor_type_,"temp",{"CA"});
    temp["MU"] += 0.5 * T2["MVXY"] * Lambda2_["XYUV"];
    temp["MU"] += T2["vMxY"] * Lambda2_["xYvU"];
    C1["qs"] -= alpha * temp["MU"] * B["gqs"] * B["gUM"];
    C1["QS"] -= alpha * temp["MU"] * B["gUM"] * B["gQS"];
    C1["QS"] += alpha * temp["MU"] * B["gSU"] * B["gQM"];

    end_ = std::chrono::system_clock::now();
    tt2_ = std::chrono::system_clock::to_time_t(end_);
    if(profile_print_){
        outfile->Printf("    [V, T2] C_4 C_2 1:3 -> C1 ended:   %s", std::ctime(&tt2_));
        outfile->Printf("    [V, T2] C_4 C_2 1:3 -> C1 wall time %.1f s.",
                        compute_elapsed_time(start_,end_).count());
    }

    if(print_ > 2){
        outfile->Printf("\n    Time for [H2, T2] -> C1 : %12.3f",timer.get());
    }
    dsrg_time_.add("221",timer.get());
}

void DSRG_MRPT3::V_T2_C2_DF(BlockedTensor& B, BlockedTensor& T2, const double& alpha, BlockedTensor& C2){
    Timer timer;

    // hole-hole contractions
    // saving hhgg should not be a problem until h > 200 and g > 600
    {
        start_ = std::chrono::system_clock::now();
        tt1_ = std::chrono::system_clock::to_time_t(start_);
        if(profile_print_){
            outfile->Printf("\n    [V, T2] DF -> C2 HH started: %s", std::ctime(&tt1_));
        }

        BlockedTensor H2 = BTF_->build(tensor_type_,"VT2->C2 H2",{"gghh"},true);
        BlockedTensor X2 = BTF_->build(tensor_type_,"T2*Eta1",{"ahpp"},true);
        H2["pqij"] = B["gpi"] * B["gqj"];
        X2["xjab"] = Eta1_["xy"] * T2["yjab"];
        C2["pqab"] += alpha * H2["pqij"] * T2["ijab"];
        C2["pqab"] -= alpha * H2["pqxj"] * X2["xjab"];
        C2["pqab"] += alpha * H2["pqjx"] * X2["xjab"];

        H2 = BTF_->build(tensor_type_,"VT2->C2 H2",{"gGhH"},true);
        H2["pQiJ"] = B["gpi"] * B["gQJ"];
        C2["pQaB"] += alpha * H2["pQiJ"] * T2["iJaB"];
        X2 = BTF_->build(tensor_type_,"T2*Eta1",{"aHpP"},true);
        X2["xJaB"] = Eta1_["xy"] * T2["yJaB"];
        C2["pQaB"] -= alpha * H2["pQxJ"] * X2["xJaB"];
        X2 = BTF_->build(tensor_type_,"T2*Eta1",{"hApP"},true);
        X2["jXaB"] = Eta1_["XY"] * T2["jYaB"];
        C2["pQaB"] -= alpha * H2["pQjX"] * X2["jXaB"];

        H2 = BTF_->build(tensor_type_,"VT2->C2 H2",{"GGHH"},true);
        X2 = BTF_->build(tensor_type_,"T2*Eta1",{"AHPP"},true);
        H2["PQIJ"] = B["gPI"] * B["gQJ"];
        X2["XJAB"] = Eta1_["XY"] * T2["YJAB"];
        C2["PQAB"] += alpha * H2["PQIJ"] * T2["IJAB"];
        C2["PQAB"] -= alpha * H2["PQXJ"] * X2["XJAB"];
        C2["PQAB"] += alpha * H2["PQJX"] * X2["XJAB"];

        end_ = std::chrono::system_clock::now();
        tt2_ = std::chrono::system_clock::to_time_t(end_);
        if(profile_print_){
            outfile->Printf("    [V, T2] DF -> C2 HH ended:   %s", std::ctime(&tt2_));
            outfile->Printf("    [V, T2] DF -> C2 HH wall time %.1f s.",
                            compute_elapsed_time(start_,end_).count());
        }
    }

    // separte virtual index adaptive to the memory
    // get the dimension of each block
    size_t c = acore_mos_.size();
    size_t a = aactv_mos_.size();
    size_t v = avirt_mos_.size();
    size_t h = c + a;
    size_t p = a + v;
    size_t g = c + p;
    size_t L = aux_mos_.size();

    long long int mem_total = Process::environment.get_memory();
    mem_total -= sizeof(double) * (9 * h * h * p * p + (2 * L + h * h) * g * g + 8 * g * g); // estimated global variables saved in memory
    size_t mem_max2 = p * p * g * g * sizeof(double);
    size_t mem_max3 = p * h * g * g * sizeof(double);
    if(mem_total < 0){
        outfile->Printf("\n    Not enough memory for batching.");
        throw PSIEXCEPTION("Not enough memory for batching at DSRG-MRPT3 V_T2_C2_DF");
    }

    // particle-particle contractions
    if (mem_max2 < mem_total) {
        start_ = std::chrono::system_clock::now();
        tt1_ = std::chrono::system_clock::to_time_t(start_);
        if(profile_print_){
            outfile->Printf("\n    [V, T2] DF -> C2 PP started: %s", std::ctime(&tt1_));
        }

        // if we can save four-index
        BlockedTensor H2 = BTF_->build(tensor_type_,"VT2->C2 H2",{"ggpp"},true);
        BlockedTensor X2 = BTF_->build(tensor_type_,"T2*Gamma1",{"hhap"},true);
        H2["rsab"] = B["gar"] * B["gbs"];
        X2["ijyb"] = Gamma1_["xy"] * T2["ijxb"];
        C2["ijrs"] += alpha * H2["rsab"] * T2["ijab"];
        C2["ijrs"] -= alpha * H2["rsyb"] * X2["ijyb"];
        C2["ijrs"] += alpha * H2["rsby"] * X2["ijyb"];

        H2 = BTF_->build(tensor_type_,"VT2->C2 H2",{"gGpP"},true);
        H2["rSaB"] = B["gar"] * B["gBS"];
        C2["iJrS"] += alpha * H2["rSaB"] * T2["iJaB"];
        X2 = BTF_->build(tensor_type_,"T2*Gamma1",{"hHaP"},true);
        X2["iJyB"] = Gamma1_["xy"] * T2["iJxB"];
        C2["iJrS"] -= alpha * H2["rSyB"] * X2["iJyB"];
        X2 = BTF_->build(tensor_type_,"T2*Gamma1",{"hHpA"},true);
        X2["iJbY"] = Gamma1_["XY"] * T2["iJbX"];
        C2["iJrS"] -= alpha * H2["rSbY"] * X2["iJbY"];

        H2 = BTF_->build(tensor_type_,"VT2->C2 H2",{"GGPP"},true);
        X2 = BTF_->build(tensor_type_,"T2*Gamma1",{"HHAP"},true);
        H2["RSAB"] = B["gAR"] * B["gBS"];
        X2["IJYB"] = Gamma1_["XY"] * T2["IJXB"];
        C2["IJRS"] += alpha * H2["RSAB"] * T2["IJAB"];
        C2["IJRS"] -= alpha * H2["RSYB"] * X2["IJYB"];
        C2["IJRS"] += alpha * H2["RSBY"] * X2["IJYB"];

        end_ = std::chrono::system_clock::now();
        tt2_ = std::chrono::system_clock::to_time_t(end_);
        if(profile_print_){
            outfile->Printf("    [V, T2] DF -> C2 PP ended:   %s", std::ctime(&tt2_));
            outfile->Printf("    [V, T2] DF -> C2 PP wall time %.1f s.",
                            compute_elapsed_time(start_,end_).count());
        }
    } else {
        // figure out how to separate the virtual block
        size_t nbatch = 2;
        size_t nele = v / nbatch;
        size_t total_ele = v * nele * (v * nele + 2 * L) + h * h * nele * nele;
        while(total_ele * sizeof(double) > mem_total){
            nbatch += 1;
            nele = v / nbatch;
            total_ele = v * nele * (v * nele + 2 * L) + h * h * nele * nele;
        }
        std::pair<double, std::string> mem_use = to_xb(total_ele, sizeof(double));
        outfile->Printf("\n    Computing [V, T2] -> C2 PP in %zu batches using %.2f %s",
                        (nbatch + 2)  * nbatch, mem_use.first, mem_use.second.c_str());

        std::vector<std::vector<size_t>> sub_virt_mos; // relative virtual indices
        size_t divisible = v / nbatch;
        size_t modulo = v % nbatch;
        for(size_t i = 0, start = 0; i < nbatch; ++i){
            size_t end;
            if(i < modulo){
                end = start + (divisible + 1);
            } else {
                end = start + divisible;
            }

            std::vector<size_t> sub_virt;
            for(size_t i = start; i < end; ++i){
                sub_virt.push_back(i);
            }
            sub_virt_mos.push_back(sub_virt);
            start = end;
        }

        V_T2_C2_DF_PP(B,T2,alpha,C2,sub_virt_mos);
    }

    // hole-particle contractions
    // memory friendly (Coulomb) part B[gqs] * ...
    {
        start_ = std::chrono::system_clock::now();
        tt1_ = std::chrono::system_clock::to_time_t(start_);
        if(profile_print_){
            outfile->Printf("\n    [V, T2] DF -> C2 PH Coulomb started: %s", std::ctime(&tt1_));
        }

        BlockedTensor H2 = BTF_->build(tensor_type_,"B temp",{"Lhp"},true);
        H2["gjb"] += B["gam"] * T2["mjab"];
        H2["gjb"] += B["gAM"] * T2["jMbA"];
        H2["gjb"] += B["gax"] * Gamma1_["xy"] * T2["yjab"];
        H2["gjb"] += B["gAX"] * Gamma1_["XY"] * T2["jYbA"];
        H2["gjb"] -= B["gyi"] * Gamma1_["xy"] * T2["ijxb"];
        H2["gjb"] -= B["gYI"] * Gamma1_["XY"] * T2["jIbX"];
        C2["qjsb"] += alpha * H2["gjb"] * B["gqs"];
        C2["jqsb"] -= alpha * H2["gjb"] * B["gqs"];
        C2["qjbs"] -= alpha * H2["gjb"] * B["gqs"];
        C2["jqbs"] += alpha * H2["gjb"] * B["gqs"];
        C2["jQbS"] += alpha * H2["gjb"] * B["gQS"];

        H2 = BTF_->build(tensor_type_,"B temp",{"LHP"},true);
        H2["gJB"] += B["gam"] * T2["mJaB"];
        H2["gJB"] += B["gAM"] * T2["MJAB"];
        H2["gJB"] += B["gax"] * Gamma1_["xy"] * T2["yJaB"];
        H2["gJB"] += B["gAX"] * Gamma1_["XY"] * T2["YJAB"];
        H2["gJB"] -= B["gyi"] * Gamma1_["xy"] * T2["iJxB"];
        H2["gJB"] -= B["gYI"] * Gamma1_["XY"] * T2["IJXB"];
        C2["QJSB"] += alpha * H2["gJB"] * B["gQS"];
        C2["JQSB"] -= alpha * H2["gJB"] * B["gQS"];
        C2["QJBS"] -= alpha * H2["gJB"] * B["gQS"];
        C2["JQBS"] += alpha * H2["gJB"] * B["gQS"];
        C2["qJsB"] += alpha * H2["gJB"] * B["gqs"];

        end_ = std::chrono::system_clock::now();
        tt2_ = std::chrono::system_clock::to_time_t(end_);
        if(profile_print_){
            outfile->Printf("    [V, T2] DF -> C2 PH Coulomb ended:   %s", std::ctime(&tt2_));
            outfile->Printf("    [V, T2] DF -> C2 PH Coulomb wall time %.1f s.",
                            compute_elapsed_time(start_,end_).count());
        }
    }

    // difficult (exchange) part
    if (2 * mem_max3 < mem_total) {
        start_ = std::chrono::system_clock::now();
        tt1_ = std::chrono::system_clock::to_time_t(start_);
        if(profile_print_){
            outfile->Printf("\n    [V, T2] DF -> C2 PH exchange started: %s", std::ctime(&tt1_));
        }

        BlockedTensor H2 = BTF_->build(tensor_type_,"VT2->H2",{"ggph"},true);
        BlockedTensor O2 = BTF_->build(tensor_type_,"VT2->H2 O2",{"ghgp"},true);
        H2["qsai"]  = B["gas"] * B["gqi"];
        O2["qjsb"] -= alpha * H2["qsam"] * T2["mjab"];
        BlockedTensor X2 = BTF_->build(tensor_type_,"T2*Gamma1",{"ahpp"},true);
        X2["xjab"] = T2["yjab"] * Gamma1_["xy"];
        O2["qjsb"] -= alpha * H2["qsax"] * X2["xjab"];
        X2 = BTF_->build(tensor_type_,"T2*Gamma1",{"hhap"},true);
        X2["ijyb"] = T2["ijxb"] * Gamma1_["xy"];
        O2["qjsb"] += alpha * H2["qsyi"] * X2["ijyb"];
        C2["qjsb"] += O2["qjsb"];
        C2["jqsb"] -= O2["qjsb"];
        C2["qjbs"] -= O2["qjsb"];
        C2["jqbs"] += O2["qjsb"];

        C2["qJsB"] -= alpha * H2["qsam"] * T2["mJaB"];
        X2 = BTF_->build(tensor_type_,"T2*Gamma1",{"aHpP"},true);
        X2["xJaB"] = T2["yJaB"] * Gamma1_["xy"];
        C2["qJsB"] -= alpha * H2["qsax"] * X2["xJaB"];
        X2 = BTF_->build(tensor_type_,"T2*Gamma1",{"hHaP"},true);
        X2["iJyB"] = T2["iJxB"] * Gamma1_["xy"];
        C2["qJsB"] += alpha * H2["qsyi"] * X2["iJyB"];

        H2 = BTF_->build(tensor_type_,"VT2->H2",{"GGPH"},true);
        O2 = BTF_->build(tensor_type_,"VT2->H2 O2",{"GHGP"},true);
        H2["QSAI"]  = B["gAS"] * B["gQI"];
        O2["QJSB"] -= alpha * H2["QSAM"] * T2["MJAB"];
        X2 = BTF_->build(tensor_type_,"T2*Gamma1",{"AHPP"},true);
        X2["XJAB"] = T2["YJAB"] * Gamma1_["XY"];
        O2["QJSB"] -= alpha * H2["QSAX"] * X2["XJAB"];
        X2 = BTF_->build(tensor_type_,"T2*Gamma1",{"HHAP"},true);
        X2["IJYB"] = T2["IJXB"] * Gamma1_["XY"];
        O2["QJSB"] += alpha * H2["QSYI"] * X2["IJYB"];
        C2["QJSB"] += O2["QJSB"];
        C2["JQSB"] -= O2["QJSB"];
        C2["QJBS"] -= O2["QJSB"];
        C2["JQBS"] += O2["QJSB"];

        C2["iQaS"] -= alpha * H2["QSBM"] * T2["iMaB"];
        X2 = BTF_->build(tensor_type_,"T2*Gamma1",{"hApP"},true);
        X2["iXaB"] = T2["iYaB"] * Gamma1_["XY"];
        C2["iQaS"] -= alpha * H2["QSBX"] * X2["iXaB"];
        X2 = BTF_->build(tensor_type_,"T2*Gamma1",{"hHpA"},true);
        X2["iJaY"] = T2["iJaX"] * Gamma1_["XY"];
        C2["iQaS"] += alpha * H2["QSYJ"] * X2["iJaY"];

        H2 = BTF_->build(tensor_type_,"VT2->H2",{"gGpH"},true);
        H2["sQaI"]  = B["gas"] * B["gQI"];
        C2["iQsB"] -= alpha * H2["sQaM"] * T2["iMaB"];
        X2 = BTF_->build(tensor_type_,"T2*Gamma1",{"hApP"},true);
        X2["iXaB"] = T2["iYaB"] * Gamma1_["XY"];
        C2["iQsB"] -= alpha * H2["sQaX"] * X2["iXaB"];
        X2 = BTF_->build(tensor_type_,"T2*Gamma1",{"hHaP"},true);
        X2["iJyB"] = T2["iJxB"] * Gamma1_["xy"];
        C2["iQsB"] += alpha * H2["sQyJ"] * X2["iJyB"];

        H2 = BTF_->build(tensor_type_,"VT2->H2",{"gGhP"},true);
        H2["qSiA"]  = B["gAS"] * B["gqi"];
        C2["qJaS"] -= alpha * H2["qSmB"] * T2["mJaB"];
        X2 = BTF_->build(tensor_type_,"T2*Gamma1",{"aHpP"},true);
        X2["xJaB"] = T2["yJaB"] * Gamma1_["xy"];
        C2["qJaS"] -= alpha * H2["qSxB"] * X2["xJaB"];
        X2 = BTF_->build(tensor_type_,"T2*Gamma1",{"hHpA"},true);
        X2["iJaY"] = T2["iJaX"] * Gamma1_["XY"];
        C2["qJaS"] += alpha * H2["qSiY"] * X2["iJaY"];

        end_ = std::chrono::system_clock::now();
        tt2_ = std::chrono::system_clock::to_time_t(end_);
        if(profile_print_){
            outfile->Printf("    [V, T2] DF -> C2 PH exchange ended:   %s", std::ctime(&tt2_));
            outfile->Printf("    [V, T2] DF -> C2 PH exchange wall time %.1f s.",
                            compute_elapsed_time(start_,end_).count());
        }
    } else {
        if(sizeof(double) * g * g * a * h * 2 > mem_total){
            std::pair<double, std::string> mem_use = to_xb(g * g * a * h * 2, sizeof(double));
            outfile->Printf("\n    Four-index tensor ggah * 2 is too large (%.2f %s) for current batching algorithm.",
                            mem_use.first, mem_use.second.c_str());
            throw PSIEXCEPTION("Not enough memory to store two ggah tensors at DSRG-MRPT3 V_T2_C2_DF batching.");
        }

        // figure out how to separate the virtual block
        size_t nbatch = 2;
        size_t nele = v / nbatch;
        size_t total_ele = g * g * a * h + (v * v + c * a) * c * nele;
        while(total_ele * sizeof(double) > mem_total){
            nbatch += 1;
            nele = v / nbatch;
            total_ele = g * g * a * h + (v * v + c * a) * c * nele;
        }
        std::pair<double, std::string> mem_use = to_xb(total_ele, sizeof(double));
        outfile->Printf("\n    Computing [V, T2] -> C2 PH in %zu batches using %.2f %s",
                        (nbatch + 2)  * nbatch, mem_use.first, mem_use.second.c_str());

        std::vector<std::vector<size_t>> sub_virt_mos; // relative virtual indices
        size_t divisible = v / nbatch;
        size_t modulo = v % nbatch;
        for(size_t i = 0, start = 0; i < nbatch; ++i){
            size_t end;
            if(i < modulo){
                end = start + (divisible + 1);
            } else {
                end = start + divisible;
            }

            std::vector<size_t> sub_virt;
            for(size_t i = start; i < end; ++i){
                sub_virt.push_back(i);
            }
            sub_virt_mos.push_back(sub_virt);
            start = end;
        }

        V_T2_C2_DF_PH_EX(B,T2,alpha,C2,sub_virt_mos);
    }

    if(print_ > 2){
        outfile->Printf("\n    Time for [H2, T2] -> C2 : %12.3f",timer.get());
    }
    dsrg_time_.add("222",timer.get());
}

void DSRG_MRPT3::V_T2_C2_DF_PP(BlockedTensor& B, BlockedTensor& T2, const double& alpha, BlockedTensor& C2, const std::vector<std::vector<size_t>>& sub_virt_mos){
    // 1) aa block: H2[rsuv]
    {
        start_ = std::chrono::system_clock::now();
        tt1_ = std::chrono::system_clock::to_time_t(start_);
        if(profile_print_){
            outfile->Printf("\n    [V, T2] DF -> C2 PP AA started: %s", std::ctime(&tt1_));
        }

        BlockedTensor H2 = BTF_->build(tensor_type_,"VT2->C2 H2",{"ggaa"},true);
        BlockedTensor X2 = BTF_->build(tensor_type_,"T2*Gamma1",{"hhaa"},true);
        X2["ijyv"] = Gamma1_["xy"] * T2["ijxv"];
        H2["rsuv"] = B["gur"] * B["gvs"];
        C2["ijrs"] += alpha * H2["rsuv"] * T2["ijuv"];
        C2["ijrs"] -= alpha * H2["rsyv"] * X2["ijyv"];
        C2["ijsr"] += alpha * H2["rsyv"] * X2["ijyv"];

        H2 = BTF_->build(tensor_type_,"VT2->C2 H2",{"gGaA"},true);
        H2["rSuV"] = B["gur"] * B["gVS"];
        C2["iJrS"] += alpha * H2["rSuV"] * T2["iJuV"];
        X2 = BTF_->build(tensor_type_,"T2*Gamma1",{"hHaA"},true);
        X2["iJyV"] = Gamma1_["xy"] * T2["iJxV"];
        C2["iJrS"] -= alpha * H2["rSyV"] * X2["iJyV"];
        X2["iJvY"] = Gamma1_["XY"] * T2["iJvX"];
        C2["iJrS"] -= alpha * H2["rSvY"] * X2["iJvY"];

        H2 = BTF_->build(tensor_type_,"VT2->C2 H2",{"GGAA"},true);
        H2["RSUV"] = B["gUR"] * B["gVS"];
        C2["IJRS"] += alpha * H2["RSUV"] * T2["IJUV"];
        X2 = BTF_->build(tensor_type_,"T2*Gamma1",{"HHAA"},true);
        X2["IJYV"] = Gamma1_["XY"] * T2["IJXV"];
        C2["IJRS"] -= alpha * H2["RSYV"] * X2["IJYV"];
        C2["IJSR"] += alpha * H2["RSYV"] * X2["IJYV"];

        end_ = std::chrono::system_clock::now();
        tt2_ = std::chrono::system_clock::to_time_t(end_);
        if(profile_print_){
            outfile->Printf("    [V, T2] DF -> C2 PP AA ended:   %s", std::ctime(&tt2_));
            outfile->Printf("    [V, T2] DF -> C2 PP AA wall time %.1f s.",
                            compute_elapsed_time(start_,end_).count());
        }
    }

    // figure out unique block labels of C2[ijrs]
    std::vector<std::string> C2_1st_half_labels;
    std::vector<std::string> C2_2nd_half_labels;
    for(const std::string& block: C2.block_labels()){
        char idx0 = block[0];
        char idx1 = block[1];
        if(idx0 != 'v' && idx0 != 'V' && idx1 != 'v' && idx1 != 'V'){
            C2_1st_half_labels.push_back(std::string (1, idx0) + std::string (1, idx1));
        }

        std::string idx2 (1, block[2]);
        std::string idx3 (1, block[3]);
        C2_2nd_half_labels.push_back(idx2 + idx3);
    }
    std::sort(C2_1st_half_labels.begin(), C2_1st_half_labels.end());
    C2_1st_half_labels.erase(std::unique(C2_1st_half_labels.begin(), C2_1st_half_labels.end()),
                             C2_1st_half_labels.end());
    std::sort(C2_2nd_half_labels.begin(), C2_2nd_half_labels.end());
    C2_2nd_half_labels.erase(std::unique(C2_2nd_half_labels.begin(), C2_2nd_half_labels.end()),
                             C2_2nd_half_labels.end());

    // compute in batches
    for(const std::string& C2_2nd_label: C2_2nd_half_labels){
        // the last two indices of C2 are the first two indices of H2
        std::string idx0 (1, C2_2nd_label[0]);
        std::string idx1 (1, C2_2nd_label[1]);
        size_t size0 = label_to_spacemo_[C2_2nd_label[0]].size();
        size_t size1 = label_to_spacemo_[C2_2nd_label[1]].size();

        // the last two indices of H2: aa, av, va, and vv
        std::string actv_idx0 (1, 'a');
        std::string actv_idx1 (1, 'a');
        std::string virt_idx0 (1, 'v');
        std::string virt_idx1 (1, 'v');
        if(isupper(C2_2nd_label[1])) actv_idx1 = "A";
        if(isupper(C2_2nd_label[0])) actv_idx0 = "A";
        if(isupper(C2_2nd_label[1])) virt_idx1 = "V";
        if(isupper(C2_2nd_label[0])) virt_idx0 = "V";

        // figure out the first two indices of C2
        std::vector<std::string> hole_labels {"cc", "ca", "ac", "aa"};
        if(isupper(actv_idx1[0])){
            hole_labels = {"cC", "cA", "aC", "aA"};
        }
        if(isupper(actv_idx0[0])){
            hole_labels = {"CC", "CA", "AC", "AA"};
        }
        std::sort(hole_labels.begin(), hole_labels.end());

        std::vector<std::string> C2_1st_half_hole_labels;
        std::set_intersection(hole_labels.begin(), hole_labels.end(),
                              C2_1st_half_labels.begin(), C2_1st_half_labels.end(),
                              std::back_inserter(C2_1st_half_hole_labels));

        // 2) av and va blocks
        start_ = std::chrono::system_clock::now();
        tt1_ = std::chrono::system_clock::to_time_t(start_);
        if(profile_print_){
            outfile->Printf("\n    [V, T2] DF -> C2 PP AV/VA block %s started: %s",
                            C2_2nd_label.c_str(), std::ctime(&tt1_));
        }

        size_t sizea = aactv_mos_.size();
        size_t sizev = avirt_mos_.size();
        size_t sizeL = aux_mos_.size();
        for(const auto& sub_virt_mo: sub_virt_mos){
            // av block
            size_t sub_virt_size = sub_virt_mo.size();
            ambit:: Tensor H2 = ambit::Tensor::build(tensor_type_,"H2 av",{size0, size1, sizea, sub_virt_size});

            std::string Blabel0 = "L" + actv_idx0 + idx0;
            std::string Blabel1 = "L" + virt_idx1 + idx1;

            ambit::Tensor B1 = ambit::Tensor::build(tensor_type_,"B1 av",{sizeL, sub_virt_size, size1});
            B1.iterate([&](const std::vector<size_t>& i, double& value){
                size_t idx = i[0] * sizev * size1 + sub_virt_mo[i[1]] * size1 + i[2];
                value = B.block(Blabel1).data()[idx];
            });

            H2("rsue") = B.block(Blabel0)("gur") * B1("ges");

            for(const std::string& C2_1st_label: C2_1st_half_hole_labels){
                std::string C2label = C2_1st_label + idx0 + idx1;
                if(!C2.is_block(C2label)) continue;

                std::string T2label = C2_1st_label + actv_idx0 + virt_idx1;
                std::string G1label = "aa";
                if(isupper(C2_1st_label[0])) G1label = "AA";

                // get subset of T2 amplitudes
                size_t hole_size0 = label_to_spacemo_[C2_1st_label[0]].size();
                size_t hole_size1 = label_to_spacemo_[C2_1st_label[1]].size();
                ambit::Tensor T2small = ambit::Tensor::build(tensor_type_,"T2 small av",
                                        {hole_size0, hole_size1, sizea, sub_virt_size});
                T2small.iterate([&](const std::vector<size_t>& i, double& value){
                    size_t idx = i[0] * hole_size1 * sizea * sizev
                            + i[1] * sizea * sizev + i[2] * sizev + sub_virt_mo[i[3]];
                    value = T2.block(T2label).data()[idx];
                });

                C2.block(C2label)("ijrs") += alpha * H2("rsue") * T2small("ijue");

                ambit::Tensor X2 = ambit::Tensor::build(tensor_type_,"T2 small av * Gamma1",
                                   {hole_size0, hole_size1, sizea, sub_virt_size});
                X2("ijye") = Gamma1_.block(G1label)("xy") * T2small("ijxe");
                C2.block(C2label)("ijrs") -= alpha * H2("rsye") * X2("ijye");
            }

            // va block
            H2 = ambit::Tensor::build(tensor_type_,"H2 va",{size0, size1, sub_virt_size, sizea});

            Blabel0 = "L" + virt_idx0 + idx0;
            Blabel1 = "L" + actv_idx1 + idx1;

            ambit::Tensor B0 = ambit::Tensor::build(tensor_type_,"B0 va",{sizeL, sub_virt_size, size0});
            B0.iterate([&](const std::vector<size_t>& i, double& value){
                size_t idx = i[0] * sizev * size0 + sub_virt_mo[i[1]] * size0 + i[2];
                value = B.block(Blabel0).data()[idx];
            });

            H2("rseu") = B.block(Blabel1)("gus") * B0("ger");

            for(const std::string& C2_1st_label: C2_1st_half_hole_labels){
                std::string C2label = C2_1st_label + idx0 + idx1;
                if(!C2.is_block(C2label)) continue;

                std::string T2label = C2_1st_label + virt_idx0 + actv_idx1;
                std::string G1label = "AA";
                if(islower(C2_1st_label[1])) G1label = "aa";

                // get subset of T2 amplitudes
                size_t hole_size0 = label_to_spacemo_[C2_1st_label[0]].size();
                size_t hole_size1 = label_to_spacemo_[C2_1st_label[1]].size();
                ambit::Tensor T2small = ambit::Tensor::build(tensor_type_, "T2 small va",
                                        {hole_size0, hole_size1, sub_virt_size, sizea});
                T2small.iterate([&](const std::vector<size_t>& i, double& value){
                    size_t idx = i[0] * hole_size1 * sizea * sizev + i[1] * sizea * sizev
                            + sub_virt_mo[i[2]] * sizea + i[3];
                    value = T2.block(T2label).data()[idx];
                });

                C2.block(C2label)("ijrs") += alpha * H2("rseu") * T2small("ijeu");

                ambit::Tensor X2 = ambit::Tensor::build(tensor_type_,"T2 small av * Gamma1",
                                   {hole_size0, hole_size1, sub_virt_size, sizea});
                X2("ijey") = Gamma1_.block(G1label)("xy") * T2small("ijex");
                C2.block(C2label)("ijrs") -= alpha * H2("rsey") * X2("ijey");
            }
        }

        end_ = std::chrono::system_clock::now();
        tt2_ = std::chrono::system_clock::to_time_t(end_);
        if(profile_print_){
            outfile->Printf("    [V, T2] DF -> C2 PP AV/VA block %s ended:   %s",
                            C2_2nd_label.c_str(), std::ctime(&tt2_));
            outfile->Printf("    [V, T2] DF -> C2 PP AV/VA block %s wall time %.1f s.",
                            C2_2nd_label.c_str(), compute_elapsed_time(start_,end_).count());
        }

        // 3) vv block
        start_ = std::chrono::system_clock::now();
        tt1_ = std::chrono::system_clock::to_time_t(start_);
        if(profile_print_){
            outfile->Printf("\n    [V, T2] DF -> C2 PP VV block %s started: %s",
                            C2_2nd_label.c_str(), std::ctime(&tt1_));
        }

        for(const auto& sub_virt_mo0: sub_virt_mos){
            size_t sub_virt_size0 = sub_virt_mo0.size();
            for(const auto& sub_virt_mo1: sub_virt_mos){
                size_t sub_virt_size1 = sub_virt_mo1.size();

                ambit::Tensor H2 = ambit::Tensor::build(tensor_type_,"H2 vv",
                                   {size0, size1, sub_virt_size0, sub_virt_size1});

                std::string Blabel0 = "L" + virt_idx0 + idx0;
                std::string Blabel1 = "L" + virt_idx1 + idx1;

                ambit::Tensor B0 = ambit::Tensor::build(tensor_type_,"B0 vv",{sizeL, sub_virt_size0, size0});
                ambit::Tensor B1 = ambit::Tensor::build(tensor_type_,"B1 vv",{sizeL, sub_virt_size1, size1});
                B0.iterate([&](const std::vector<size_t>& i, double& value){
                    size_t idx = i[0] * sizev * size0 + sub_virt_mo0[i[1]] * size0 + i[2];
                    value = B.block(Blabel0).data()[idx];
                });
                B1.iterate([&](const std::vector<size_t>& i, double& value){
                    size_t idx = i[0] * sizev * size1 + sub_virt_mo1[i[1]] * size1 + i[2];
                    value = B.block(Blabel1).data()[idx];
                });

                H2("rsef") = B0("ger") * B1("gfs");

                for(const std::string& C2_1st_label: C2_1st_half_hole_labels){
                    std::string C2label = C2_1st_label + idx0 + idx1;
                    if(!C2.is_block(C2label)) continue;

                    std::string T2label = C2_1st_label + virt_idx0 + virt_idx1;

                    // get subset of T2 amplitudes
                    size_t hole_size0 = label_to_spacemo_[C2_1st_label[0]].size();
                    size_t hole_size1 = label_to_spacemo_[C2_1st_label[1]].size();
                    ambit::Tensor T2small = ambit::Tensor::build(tensor_type_,"T2 small vv",
                                            {hole_size0, hole_size1, sub_virt_size0, sub_virt_size1});
                    T2small.iterate([&](const std::vector<size_t>& i, double& value){
                        size_t idx = i[0] * hole_size1 * sizev * sizev + i[1] * sizev * sizev
                                + sub_virt_mo0[i[2]] * sizev + sub_virt_mo1[i[3]];
                        value = T2.block(T2label).data()[idx];
                    });

                    C2.block(C2label)("ijrs") += alpha * H2("rsef") * T2small("ijef");
                }
            }
        }

        end_ = std::chrono::system_clock::now();
        tt2_ = std::chrono::system_clock::to_time_t(end_);
        if(profile_print_){
            outfile->Printf("    [V, T2] DF -> C2 PP VV block %s ended:   %s",
                            C2_2nd_label.c_str(), std::ctime(&tt2_));
            outfile->Printf("    [V, T2] DF -> C2 PP VV block %s wall time %.1f s.",
                            C2_2nd_label.c_str(), compute_elapsed_time(start_,end_).count());
        }
    }
}

void DSRG_MRPT3::V_T2_C2_DF_PH_EX(BlockedTensor& B, BlockedTensor& T2, const double& alpha, BlockedTensor& C2, const std::vector<std::vector<size_t>>& sub_virt_mos){
    size_t sizec = acore_mos_.size();
    size_t sizea = aactv_mos_.size();
    size_t sizev = avirt_mos_.size();
    size_t sizeL = aux_mos_.size();

    // figure out unique indices "qs" and "jb" of C2
    std::vector<std::string> C2label_qs_aa, C2label_qs_bb, C2label_qs_ba, C2label_qs_ab;
    std::vector<std::string> C2label_jb_aa, C2label_jb_bb, C2label_jb_ab, C2label_jb_ba;
    auto jb_is_hp = [&](char idxh, char idxp, bool alpha) -> bool {
        bool j, b;
        if (alpha) {
            j = (idxh == 'c' || idxh == 'a');
            b = (idxp == 'a' || idxp == 'v');
        } else {
            j = (idxh == 'C' || idxh == 'A');
            b = (idxp == 'A' || idxp == 'V');
        }
        return j && b;
    };
    auto fill_C2label = [&](char i, char j){
        if(islower(i) && islower(j)){
            C2label_qs_aa.push_back(std::string (1, i) + std::string (1, j));
            if(jb_is_hp(i,j,true)){
                C2label_jb_aa.push_back(std::string (1, i) + std::string (1, j));
            }
        }
        if(isupper(i) && isupper(j)){
            C2label_qs_bb.push_back(std::string (1, i) + std::string (1, j));
            if(jb_is_hp(i,j,false)){
                C2label_jb_bb.push_back(std::string (1, i) + std::string (1, j));
            }
        }
    };

    for(const std::string& block: C2.block_labels()){
        char idx0 = block[0];
        char idx1 = block[1];
        char idx2 = block[2];
        char idx3 = block[3];

        // aa and bb
        fill_C2label(idx0,idx2);
        fill_C2label(idx1,idx2);
        fill_C2label(idx0,idx3);
        fill_C2label(idx1,idx3);

        // ba
        if(isupper(idx1) && islower(idx2)){
            C2label_qs_ba.push_back(std::string (1, idx1) + std::string (1, idx2));
            bool j = (idx1 == 'C' || idx1 == 'A');
            bool b = (idx2 == 'a' || idx2 == 'v');
            if(j && b){
                C2label_jb_ba.push_back(std::string (1, idx1) + std::string (1, idx2));
            }
        }

        // ab
        if(islower(idx0) && isupper(idx3)){
            C2label_qs_ab.push_back(std::string (1, idx0) + std::string (1, idx3));
            bool j = (idx0 == 'c' || idx0 == 'a');
            bool b = (idx3 == 'A' || idx3 == 'V');
            if(j && b){
                C2label_jb_ab.push_back(std::string (1, idx0) + std::string (1, idx3));
            }
        }
    }

    auto keep_unique = [](std::vector<std::string> vec_str){
        std::sort(vec_str.begin(), vec_str.end());
        vec_str.erase(std::unique(vec_str.begin(), vec_str.end()), vec_str.end());
        return vec_str;
    };

    C2label_qs_aa = keep_unique(C2label_qs_aa);
    C2label_qs_bb = keep_unique(C2label_qs_bb);
    C2label_qs_ba = keep_unique(C2label_qs_ba);
    C2label_qs_ab = keep_unique(C2label_qs_ab);
    std::vector<std::string> C2label_qs_spinpure (C2label_qs_aa);
    C2label_qs_spinpure.insert(C2label_qs_spinpure.end(), C2label_qs_bb.begin(), C2label_qs_bb.end());

    C2label_jb_aa = keep_unique(C2label_jb_aa);
    C2label_jb_bb = keep_unique(C2label_jb_bb);
    C2label_jb_ba = keep_unique(C2label_jb_ba);
    C2label_jb_ab = keep_unique(C2label_jb_ab);

    // H2[qsai] * T2[ijab], separate "a" and "b" in batches
    // Step 1: "a" -> active, "b" -> active (C2 aa / bb), "b" -> particle (C2 ab)
    {
        start_ = std::chrono::system_clock::now();
        tt1_ = std::chrono::system_clock::to_time_t(start_);
        if(profile_print_){
            outfile->Printf("\n    [V, T2] DF -> C2aa PH exchange AA started: %s", std::ctime(&tt1_));
        }

        BlockedTensor X2;
        BlockedTensor H2 = BTF_->build(tensor_type_,"VT2->H2",{"ggah"},true);
        {
            BlockedTensor O2 = BTF_->build(tensor_type_,"VT2->H2 O2",{"ghga"},true);
            H2["qsui"]  = B["gus"] * B["gqi"];
            O2["qjsv"] -= alpha * H2["qsum"] * T2["mjuv"];
            X2 = BTF_->build(tensor_type_,"T2*Gamma1",{"ahaa"},true);
            X2["xjuv"] = T2["yjuv"] * Gamma1_["xy"];
            O2["qjsv"] -= alpha * H2["qsux"] * X2["xjuv"];
            X2 = BTF_->build(tensor_type_,"T2*Gamma1",{"hhaa"},true);
            X2["ijyv"] = T2["ijxv"] * Gamma1_["xy"];
            O2["qjsv"] += alpha * H2["qsyi"] * X2["ijyv"];
            C2["qjsv"] += O2["qjsv"];
            C2["jqsv"] -= O2["qjsv"];
            C2["qjvs"] -= O2["qjsv"];
            C2["jqvs"] += O2["qjsv"];
        }

        C2["qJsB"] -= alpha * H2["qsum"] * T2["mJuB"];
        X2 = BTF_->build(tensor_type_,"T2*Gamma1",{"aHaP"},true);
        X2["xJuB"] = T2["yJuB"] * Gamma1_["xy"];
        C2["qJsB"] -= alpha * H2["qsux"] * X2["xJuB"];
        X2 = BTF_->build(tensor_type_,"T2*Gamma1",{"hHaP"},true);
        X2["iJyB"] = T2["iJxB"] * Gamma1_["xy"];
        C2["qJsB"] += alpha * H2["qsyi"] * X2["iJyB"];

        end_ = std::chrono::system_clock::now();
        tt2_ = std::chrono::system_clock::to_time_t(end_);
        if(profile_print_){
            outfile->Printf("    [V, T2] DF -> C2aa PH exchange AA ended:   %s", std::ctime(&tt2_));
            outfile->Printf("    [V, T2] DF -> C2aa PH exchange AA wall time %.1f s.",
                            compute_elapsed_time(start_,end_).count());
        }

        //   sidestep: separate "b" -> virtual to subblocks for C2 aa spin
        start_ = std::chrono::system_clock::now();
        tt1_ = std::chrono::system_clock::to_time_t(start_);
        if(profile_print_){
            outfile->Printf("\n    [V, T2] DF -> C2aa PH exchange AV started: %s", std::ctime(&tt1_));
        }

        for(const std::string& idx_qs: C2label_qs_aa){
            std::string idx0 (1, idx_qs[0]);
            std::string idx1 (1, idx_qs[1]);
            size_t size0 = label_to_spacemo_[idx0[0]].size();
            size_t size1 = label_to_spacemo_[idx1[0]].size();

            // alpha spin of "j"
            for(const std::string& idx_h: {"c","a"}){

                std::string C2label_P0 = idx0 + idx_h + idx1 + "v";
                std::string C2label_P1 = idx_h + idx0 + idx1 + "v";
                std::string C2label_P2 = idx0 + idx_h + "v" + idx1;
                std::string C2label_P3 = idx_h + idx0 + "v" + idx1;
                bool is_C2label_P0 = C2.is_block(C2label_P0);
                bool is_C2label_P1 = C2.is_block(C2label_P1);
                bool is_C2label_P2 = C2.is_block(C2label_P2);
                bool is_C2label_P3 = C2.is_block(C2label_P3);
                if(!is_C2label_P0 && !is_C2label_P1 && !is_C2label_P2 && !is_C2label_P3)
                    continue;

                std::string T2label_c = "c" + idx_h + "a" + "v";
                std::string H2label_c = idx0 + idx1 + "a" + "c";
                std::string T2label_a = "a" + idx_h + "a" + "v";
                std::string H2label_a = idx0 + idx1 + "a" + "a";

                size_t sizeh = label_to_spacemo_[idx_h[0]].size();
                for(const auto& sub_virt_mo: sub_virt_mos){
                    size_t sub_virt_size = sub_virt_mo.size();

                    // T2[mjab]
                    ambit::Tensor T2sub = ambit::Tensor::build(tensor_type_,"T2 sub",
                    {sizec, sizeh, sizea, sub_virt_size});
                    T2sub.iterate([&](const std::vector<size_t>& i, double& value){
                        size_t idx = i[0] * sizeh * sizea * sizev + i[1] * sizea * sizev
                                + i[2] * sizev + sub_virt_mo[i[3]];
                        value = T2.block(T2label_c).data()[idx];
                    });

                    ambit::Tensor O2sub = ambit::Tensor::build(tensor_type_,"O2 sub",
                    {size0, sizeh, size1, sub_virt_size});

                    O2sub("qjse") -= alpha * H2.block(H2label_c)("qsum") * T2sub("mjue");

                    // T2[yjab]
                    T2sub = ambit::Tensor::build(tensor_type_,"T2 sub",{sizea, sizeh, sizea, sub_virt_size});
                    T2sub.iterate([&](const std::vector<size_t>& i, double& value){
                        size_t idx = i[0] * sizeh * sizea * sizev + i[1] * sizea * sizev
                                + i[2] * sizev + sub_virt_mo[i[3]];
                        value = T2.block(T2label_a).data()[idx];
                    });

                    ambit::Tensor Y2 = ambit::Tensor::build(tensor_type_,"T2 sub * Gamma1",
                    {sizea, sizeh, sizea, sub_virt_size});
                    Y2("xjue") = Gamma1_.block("aa")("xy") * T2sub("yjue");
                    O2sub("qjse") -= alpha * H2.block(H2label_a)("qsux") * Y2("xjue");

                    // T2[ijxb]
                    for(const std::string& idx_h0: {"c","a"}){
                        size_t sizeh1 = label_to_spacemo_[idx_h0[0]].size();
                        std::string H2label_h1 = idx0 + idx1 + "a" + idx_h0;
                        std::string T2label_h1 = idx_h0 + idx_h + "a" + "v";

                        T2sub = ambit::Tensor::build(tensor_type_,"T2 sub",{sizeh1, sizeh, sizea, sub_virt_size});
                        T2sub.iterate([&](const std::vector<size_t>& i, double& value){
                            size_t idx = i[0] * sizeh * sizea * sizev + i[1] * sizea * sizev
                                    + i[2] * sizev + sub_virt_mo[i[3]];
                            value = T2.block(T2label_h1).data()[idx];
                        });

                        Y2 = ambit::Tensor::build(tensor_type_,"T2 sub * Gamma1",{sizeh1, sizeh, sizea, sub_virt_size});
                        Y2("ijye") = Gamma1_.block("aa")("xy") * T2sub("ijxe");
                        O2sub("qjse") += alpha * H2.block(H2label_h1)("qsyi") * Y2("ijye");
                    }

                    // adding O2sub to C2 with permutations
                    O2sub.iterate([&](const std::vector<size_t>& i, double& value){
                        if(is_C2label_P0){
                            size_t idx = i[0] * sizeh * size1 * sizev + i[1] * size1 * sizev
                                    + i[2] * sizev + sub_virt_mo[i[3]];
                            C2.block(C2label_P0).data()[idx] += value;
                        }
                        if(is_C2label_P1){
                            size_t idx = i[1] * size0 * size1 * sizev + i[0] * size1 * sizev
                                    + i[2] * sizev + sub_virt_mo[i[3]];
                            C2.block(C2label_P1).data()[idx] -= value;
                        }
                        if(is_C2label_P2){
                            size_t idx = i[0] * sizeh * size1 * sizev + i[1] * size1 * sizev
                                    + sub_virt_mo[i[3]] * size1 + i[2];
                            C2.block(C2label_P2).data()[idx] -= value;
                        }
                        if(is_C2label_P3){
                            size_t idx = i[1] * size0 * size1 * sizev + i[0] * size1 * sizev
                                    + sub_virt_mo[i[3]] * size1 + i[2];
                            C2.block(C2label_P3).data()[idx] += value;
                        }
                    });

                } // end virtual partition
            } // end idx_h: "j"
        } // end C2label_qs_aa

        end_ = std::chrono::system_clock::now();
        tt2_ = std::chrono::system_clock::to_time_t(end_);
        if(profile_print_){
            outfile->Printf("    [V, T2] DF -> C2aa PH exchange AV ended:   %s", std::ctime(&tt2_));
            outfile->Printf("    [V, T2] DF -> C2aa PH exchange AV wall time %.1f s.",
                            compute_elapsed_time(start_,end_).count());
        }

        start_ = std::chrono::system_clock::now();
        tt1_ = std::chrono::system_clock::to_time_t(start_);
        if(profile_print_){
            outfile->Printf("\n    [V, T2] DF -> C2bb PH exchange AA started: %s", std::ctime(&tt1_));
        }

        H2 = BTF_->build(tensor_type_,"VT2->H2",{"GGAH"},true);
        {
            BlockedTensor O2 = BTF_->build(tensor_type_,"VT2->H2 O2",{"GHGA"},true);
            H2["QSUI"]  = B["gUS"] * B["gQI"];
            O2["QJSV"] -= alpha * H2["QSUM"] * T2["MJUV"];
            X2 = BTF_->build(tensor_type_,"T2*Gamma1",{"AHAA"},true);
            X2["XJUV"] = T2["YJUV"] * Gamma1_["XY"];
            O2["QJSV"] -= alpha * H2["QSUX"] * X2["XJUV"];
            X2 = BTF_->build(tensor_type_,"T2*Gamma1",{"HHAA"},true);
            X2["IJYV"] = T2["IJXV"] * Gamma1_["XY"];
            O2["QJSV"] += alpha * H2["QSYI"] * X2["IJYV"];
            C2["QJSV"] += O2["QJSV"];
            C2["JQSV"] -= O2["QJSV"];
            C2["QJVS"] -= O2["QJSV"];
            C2["JQVS"] += O2["QJSV"];
        }

        C2["iQaS"] -= alpha * H2["QSUM"] * T2["iMaU"];
        X2 = BTF_->build(tensor_type_,"T2*Gamma1",{"hApA"},true);
        X2["iXaU"] = T2["iYaU"] * Gamma1_["XY"];
        C2["iQaS"] -= alpha * H2["QSUX"] * X2["iXaU"];
        X2 = BTF_->build(tensor_type_,"T2*Gamma1",{"hHpA"},true);
        X2["iJaY"] = T2["iJaX"] * Gamma1_["XY"];
        C2["iQaS"] += alpha * H2["QSYJ"] * X2["iJaY"];

        end_ = std::chrono::system_clock::now();
        tt2_ = std::chrono::system_clock::to_time_t(end_);
        if(profile_print_){
            outfile->Printf("    [V, T2] DF -> C2bb PH exchange AA ended:   %s", std::ctime(&tt2_));
            outfile->Printf("    [V, T2] DF -> C2bb PH exchange AA wall time %.1f s.",
                            compute_elapsed_time(start_,end_).count());
        }

        //   sizestep: separate "b" -> virtual to subblocks for C2 bb spin
        start_ = std::chrono::system_clock::now();
        tt1_ = std::chrono::system_clock::to_time_t(start_);
        if(profile_print_){
            outfile->Printf("\n    [V, T2] DF -> C2bb PH exchange AV started: %s", std::ctime(&tt1_));
        }

        for(const std::string& idx_qs: C2label_qs_bb){
            std::string idx0 (1, idx_qs[0]);
            std::string idx1 (1, idx_qs[1]);
            size_t size0 = label_to_spacemo_[idx0[0]].size();
            size_t size1 = label_to_spacemo_[idx1[0]].size();

            // beta spin of "J"
            for(const std::string& idx_h: {"C","A"}){

                std::string C2label_P0 = idx0 + idx_h + idx1 + "V";
                std::string C2label_P1 = idx_h + idx0 + idx1 + "V";
                std::string C2label_P2 = idx0 + idx_h + "V" + idx1;
                std::string C2label_P3 = idx_h + idx0 + "V" + idx1;
                bool is_C2label_P0 = C2.is_block(C2label_P0);
                bool is_C2label_P1 = C2.is_block(C2label_P1);
                bool is_C2label_P2 = C2.is_block(C2label_P2);
                bool is_C2label_P3 = C2.is_block(C2label_P3);
                if(!is_C2label_P0 && !is_C2label_P1 && !is_C2label_P2 && !is_C2label_P3)
                    continue;

                std::string T2label_c = "C" + idx_h + "A" + "V";
                std::string H2label_c = idx0 + idx1 + "A" + "C";
                std::string T2label_a = "A" + idx_h + "A" + "V";
                std::string H2label_a = idx0 + idx1 + "A" + "A";

                size_t sizeh = label_to_spacemo_[idx_h[0]].size();
                for(const auto& sub_virt_mo: sub_virt_mos){
                    size_t sub_virt_size = sub_virt_mo.size();

                    // T2[mjab]
                    ambit::Tensor T2sub = ambit::Tensor::build(tensor_type_,"T2 sub",
                    {sizec, sizeh, sizea, sub_virt_size});
                    T2sub.iterate([&](const std::vector<size_t>& i, double& value){
                        size_t idx = i[0] * sizeh * sizea * sizev + i[1] * sizea * sizev
                                + i[2] * sizev + sub_virt_mo[i[3]];
                        value = T2.block(T2label_c).data()[idx];
                    });

                    ambit::Tensor O2sub = ambit::Tensor::build(tensor_type_,"O2 sub",
                    {size0, sizeh, size1, sub_virt_size});

                    O2sub("qjse") -= alpha * H2.block(H2label_c)("qsum") * T2sub("mjue");

                    // T2[yjab]
                    T2sub = ambit::Tensor::build(tensor_type_,"T2 sub",
                    {sizea, sizeh, sizea, sub_virt_size});
                    T2sub.iterate([&](const std::vector<size_t>& i, double& value){
                        size_t idx = i[0] * sizeh * sizea * sizev + i[1] * sizea * sizev
                                + i[2] * sizev + sub_virt_mo[i[3]];
                        value = T2.block(T2label_a).data()[idx];
                    });

                    ambit::Tensor Y2 = ambit::Tensor::build(tensor_type_,"T2 sub * Gamma1",
                    {sizea, sizeh, sizea, sub_virt_size});
                    Y2("xjue") = Gamma1_.block("AA")("xy") * T2sub("yjue");
                    O2sub("qjse") -= alpha * H2.block(H2label_a)("qsux") * Y2("xjue");

                    // T2[ijxb]
                    for(const std::string& idx_h0: {"C","A"}){
                        size_t sizeh1 = label_to_spacemo_[idx_h0[0]].size();
                        std::string H2label_h1 = idx0 + idx1 + "A" + idx_h0;
                        std::string T2label_h1 = idx_h0 + idx_h + "A" + "V";

                        T2sub = ambit::Tensor::build(tensor_type_,"T2 sub",
                        {sizeh1, sizeh, sizea, sub_virt_size});
                        T2sub.iterate([&](const std::vector<size_t>& i, double& value){
                            size_t idx = i[0] * sizeh * sizea * sizev + i[1] * sizea * sizev
                                    + i[2] * sizev + sub_virt_mo[i[3]];
                            value = T2.block(T2label_h1).data()[idx];
                        });

                        Y2 = ambit::Tensor::build(tensor_type_,"T2 sub * Gamma1",
                        {sizeh1, sizeh, sizea, sub_virt_size});
                        Y2("ijye") = Gamma1_.block("AA")("xy") * T2sub("ijxe");
                        O2sub("qjse") += alpha * H2.block(H2label_h1)("qsyi") * Y2("ijye");
                    }

                    // adding O2sub to C2 with permutations
                    O2sub.iterate([&](const std::vector<size_t>& i, double& value){
                        if(is_C2label_P0){
                            size_t idx = i[0] * sizeh * size1 * sizev + i[1] * size1 * sizev
                                    + i[2] * sizev + sub_virt_mo[i[3]];
                            C2.block(C2label_P0).data()[idx] += value;
                        }
                        if(is_C2label_P1){
                            size_t idx = i[1] * size0 * size1 * sizev + i[0] * size1 * sizev
                                    + i[2] * sizev + sub_virt_mo[i[3]];
                            C2.block(C2label_P1).data()[idx] -= value;
                        }
                        if(is_C2label_P2){
                            size_t idx = i[0] * sizeh * size1 * sizev + i[1] * size1 * sizev
                                    + sub_virt_mo[i[3]] * size1 + i[2];
                            C2.block(C2label_P2).data()[idx] -= value;
                        }
                        if(is_C2label_P3){
                            size_t idx = i[1] * size0 * size1 * sizev + i[0] * size1 * sizev
                                    + sub_virt_mo[i[3]] * size1 + i[2];
                            C2.block(C2label_P3).data()[idx] += value;
                        }
                    });

                } // end virtual partition
            } // end idx_h: "J"
        } // end C2label_qs_bb

        end_ = std::chrono::system_clock::now();
        tt2_ = std::chrono::system_clock::to_time_t(end_);
        if(profile_print_){
            outfile->Printf("    [V, T2] DF -> C2bb PH exchange AV ended:   %s", std::ctime(&tt2_));
            outfile->Printf("    [V, T2] DF -> C2bb PH exchange AV wall time %.1f s.",
                            compute_elapsed_time(start_,end_).count());
        }

        start_ = std::chrono::system_clock::now();
        tt1_ = std::chrono::system_clock::to_time_t(start_);
        if(profile_print_){
            outfile->Printf("\n    [V, T2] DF -> C2ab PH exchange AP started: %s", std::ctime(&tt1_));
        }

        H2 = BTF_->build(tensor_type_,"VT2->H2",{"gGaH"},true);
        H2["sQuI"]  = B["gus"] * B["gQI"];
        C2["iQsB"] -= alpha * H2["sQuM"] * T2["iMuB"];
        X2 = BTF_->build(tensor_type_,"T2*Gamma1",{"hAaP"},true);
        X2["iXuB"] = T2["iYuB"] * Gamma1_["XY"];
        C2["iQsB"] -= alpha * H2["sQuX"] * X2["iXuB"];
        X2 = BTF_->build(tensor_type_,"T2*Gamma1",{"hHaP"},true);
        X2["iJyB"] = T2["iJxB"] * Gamma1_["xy"];
        C2["iQsB"] += alpha * H2["sQyJ"] * X2["iJyB"];

        H2 = BTF_->build(tensor_type_,"VT2->H2",{"gGhA"},true);
        H2["qSiU"]  = B["gUS"] * B["gqi"];
        C2["qJaS"] -= alpha * H2["qSmU"] * T2["mJaU"];
        X2 = BTF_->build(tensor_type_,"T2*Gamma1",{"aHpA"},true);
        X2["xJaU"] = T2["yJaU"] * Gamma1_["xy"];
        C2["qJaS"] -= alpha * H2["qSxU"] * X2["xJaU"];
        X2 = BTF_->build(tensor_type_,"T2*Gamma1",{"hHpA"},true);
        X2["iJaY"] = T2["iJaX"] * Gamma1_["XY"];
        C2["qJaS"] += alpha * H2["qSiY"] * X2["iJaY"];

        end_ = std::chrono::system_clock::now();
        tt2_ = std::chrono::system_clock::to_time_t(end_);
        if(profile_print_){
            outfile->Printf("    [V, T2] DF -> C2ab PH exchange AP ended:   %s", std::ctime(&tt2_));
            outfile->Printf("    [V, T2] DF -> C2ab PH exchange AP wall time %.1f s.",
                            compute_elapsed_time(start_,end_).count());
        }
    }

    // Step 2: "a" -> virtual
    start_ = std::chrono::system_clock::now();
    tt1_ = std::chrono::system_clock::to_time_t(start_);
    if(profile_print_){
        outfile->Printf("\n    [V, T2] DF -> C2aa/bb PH exchange VP started: %s", std::ctime(&tt1_));
    }

    // q: idx0, s: idx1
    for(const std::string& idxqs: C2label_qs_spinpure){
        std::string idx0 (1, idxqs[0]);
        std::string idx1 (1, idxqs[1]);
        size_t size0 = label_to_spacemo_[idx0[0]].size();
        size_t size1 = label_to_spacemo_[idx1[0]].size();
        bool beta = isupper(idx0[0]);

        std::vector<std::string> vec_idxh0 {"c","a"};
        std::vector<std::string> vec_idxh1p1 = C2label_jb_aa;
        if(beta){
            vec_idxh0 = {"C","A"};
            vec_idxh1p1 = C2label_jb_bb;
        }

        // i: idxh0
        for(const std::string& idxh0: vec_idxh0){
            size_t sizeh0 = label_to_spacemo_[idxh0[0]].size();
            std::string Blabel0 = "Lv" + idx1;
            std::string Blabel1 = "L" + idx0 + idxh0;
            if(beta){ Blabel0 = "LV" + idx1; }

            // a: sub_virt_mo0
            for(const auto& sub_virt_mo0: sub_virt_mos){
                size_t sub_virt_size0 = sub_virt_mo0.size();

                ambit::Tensor Bsub = ambit::Tensor::build(tensor_type_,"B sub",
                {sizeL, sub_virt_size0, size1});
                Bsub.iterate([&](const std::vector<size_t>& i, double& value){
                    size_t idx = i[0] * sizev * size1 + sub_virt_mo0[i[1]] * size1 + i[2];
                    value = B.block(Blabel0).data()[idx];
                });

                ambit::Tensor H2 = ambit::Tensor::build(tensor_type_,"H2 sub",
                {size0, size1, sub_virt_size0, sizeh0});
                H2("qsei") = Bsub("ges") * B.block(Blabel1)("gqi");

                // j: idxh1, b: idxp1, C2 aa / bb spin
                for(const std::string& idxhp: vec_idxh1p1){
                    std::string idxh1 (1, idxhp[0]);
                    std::string idxp1 (1, idxhp[1]);
                    size_t sizeh1 = label_to_spacemo_[idxh1[0]].size();

                    std::string C2label_P0 = idx0 + idxh1 + idx1 + idxp1;
                    std::string C2label_P1 = idxh1 + idx0 + idx1 + idxp1;
                    std::string C2label_P2 = idx0 + idxh1 + idxp1 + idx1;
                    std::string C2label_P3 = idxh1 + idx0 + idxp1 + idx1;
                    bool is_C2label_P0 = C2.is_block(C2label_P0);
                    bool is_C2label_P1 = C2.is_block(C2label_P1);
                    bool is_C2label_P2 = C2.is_block(C2label_P2);
                    bool is_C2label_P3 = C2.is_block(C2label_P3);
                    if(!is_C2label_P0 && !is_C2label_P1 && !is_C2label_P2 && !is_C2label_P3)
                        continue;

                    // "b" -> active
                    if(idxhp[1] == 'a' || idxhp[1] == 'A'){
                        std::string idxa (idxp1);

                        std::string T2label = idxh0 + idxh1 + "v" + idxa;
                        if(beta){
                            T2label = idxh0 + idxh1 + "V" + idxa;
                        }
                        ambit::Tensor T2sub = ambit::Tensor::build(tensor_type_,"T2 sub",
                        {sizeh0,sizeh1,sub_virt_size0,sizea});
                        T2sub.iterate([&](const std::vector<size_t>& i, double& value){
                            size_t idx = i[0] * sizeh1 * sizev * sizea + i[1] * sizev * sizea
                                    + sub_virt_mo0[i[2]] * sizea + i[3];
                            value = T2.block(T2label).data()[idx];
                        });

                        ambit::Tensor O2sub = ambit::Tensor::build(tensor_type_,"O2 sub",
                        {size0,sizeh1,size1,sizea});

                        if(idxh0 == "c" || idxh0 == "C"){
                            O2sub("qjsv") -= alpha * H2("qsam") * T2sub("mjav");
                        } else {
                            ambit::Tensor Y2 = ambit::Tensor::build(tensor_type_,"T2 sub * Gamma1",
                            {sizeh0,sizeh1,sub_virt_size0,sizea});
                            Y2("xjav") = Gamma1_.block(idxa + idxa)("xy") * T2sub("yjav");
                            O2sub("qjsv") -= alpha * H2("qsax") * Y2("xjav");
                        }

                        // adding O2sub to C2 with permutations
                        O2sub.iterate([&](const std::vector<size_t>& i, double& value){
                            if(is_C2label_P0){
                                size_t idx = i[0] * sizeh1 * size1 * sizea + i[1] * size1 * sizea
                                        + i[2] * sizea + i[3];
                                C2.block(C2label_P0).data()[idx] += value;
                            }
                            if(is_C2label_P1){
                                size_t idx = i[1] * size0 * size1 * sizea + i[0] * size1 * sizea
                                        + i[2] * sizea + i[3];
                                C2.block(C2label_P1).data()[idx] -= value;
                            }
                            if(is_C2label_P2){
                                size_t idx = i[0] * sizeh1 * size1 * sizea + i[1] * size1 * sizea
                                        + i[3] * size1 + i[2];
                                C2.block(C2label_P2).data()[idx] -= value;
                            }
                            if(is_C2label_P3){
                                size_t idx = i[1] * size0 * size1 * sizea + i[0] * size1 * sizea
                                        + i[3] * size1 + i[2];
                                C2.block(C2label_P3).data()[idx] += value;
                            }
                        });
                    }

                    // "b" -> virtual
                    if(idxhp[1] == 'v' || idxhp[1] == 'V'){
                        std::string idxv (idxp1);

                        for(const auto& sub_virt_mo1: sub_virt_mos){
                            size_t sub_virt_size1 = sub_virt_mo1.size();

                            std::string T2label = idxh0 + idxh1 + "v" + idxv;
                            if(beta){
                                T2label = idxh0 + idxh1 + "V" + idxv;
                            }
                            ambit::Tensor T2sub = ambit::Tensor::build(tensor_type_,"T2 sub",
                            {sizeh0,sizeh1,sub_virt_size0,sub_virt_size1});
                            T2sub.iterate([&](const std::vector<size_t>& i, double& value){
                                size_t idx = i[0] * sizeh1 * sizev * sizev + i[1] * sizev * sizev
                                        + sub_virt_mo0[i[2]] * sizev + sub_virt_mo1[i[3]];
                                value = T2.block(T2label).data()[idx];
                            });

                            ambit::Tensor O2sub = ambit::Tensor::build(tensor_type_,"O2 sub",
                            {size0,sizeh1,size1,sub_virt_size1});

                            if(idxh0 == "c" || idxh0 == "C"){
                                O2sub("qjse") -= alpha * H2("qsam") * T2sub("mjae");
                            } else {
                                std::string G1label = "aa";
                                if(beta){ G1label = "AA"; }

                                ambit::Tensor Y2 = ambit::Tensor::build(tensor_type_,"T2 sub * Gamma1",
                                {sizeh0,sizeh1,sub_virt_size0,sub_virt_size1});
                                Y2("xjae") = Gamma1_.block(G1label)("xy") * T2sub("yjae");
                                O2sub("qjse") -= alpha * H2("qsax") * Y2("xjae");
                            }

                            // adding O2sub to C2 with permutations
                            O2sub.iterate([&](const std::vector<size_t>& i, double& value){
                                if(is_C2label_P0){
                                    size_t idx = i[0] * sizeh1 * size1 * sizev + i[1] * size1 * sizev
                                            + i[2] * sizev + sub_virt_mo1[i[3]];
                                    C2.block(C2label_P0).data()[idx] += value;
                                }
                                if(is_C2label_P1){
                                    size_t idx = i[1] * size0 * size1 * sizev + i[0] * size1 * sizev
                                            + i[2] * sizev + sub_virt_mo1[i[3]];
                                    C2.block(C2label_P1).data()[idx] -= value;
                                }
                                if(is_C2label_P2){
                                    size_t idx = i[0] * sizeh1 * size1 * sizev + i[1] * size1 * sizev
                                            + sub_virt_mo1[i[3]] * size1 + i[2];
                                    C2.block(C2label_P2).data()[idx] -= value;
                                }
                                if(is_C2label_P3){
                                    size_t idx = i[1] * size0 * size1 * sizev + i[0] * size1 * sizev
                                            + sub_virt_mo1[i[3]] * size1 + i[2];
                                    C2.block(C2label_P3).data()[idx] += value;
                                }
                            });
                        }
                    }
                }

                // C2 ab spin
                if(!beta){
                    // index J, B
                    for(const std::string& idxhp: C2label_jb_bb){
                        std::string idxh1 (1, idxhp[0]);
                        std::string idxp1 (1, idxhp[1]);
                        size_t sizeh1 = label_to_spacemo_[idxh1[0]].size();
                        size_t sizep1 = label_to_spacemo_[idxp1[0]].size();

                        std::string C2label = idx0 + idxh1 + idx1 + idxp1;
                        if(!C2.is_block(C2label)) continue;

                        std::string T2label = idxh0 + idxh1 + "v" + idxp1;
                        ambit::Tensor T2sub = ambit::Tensor::build(tensor_type_,"T2 sub",
                        {sizeh0,sizeh1,sub_virt_size0,sizep1});
                        T2sub.iterate([&](const std::vector<size_t>& i, double& value){
                            size_t idx = i[0] * sizeh1 * sizev * sizep1 + i[1] * sizev * sizep1
                                    + sub_virt_mo0[i[2]] * sizep1 + i[3];
                            value = T2.block(T2label).data()[idx];
                        });

                        if(idxh0 == "c"){
                            C2.block(C2label)("qJsB") -= alpha * H2("qsam") * T2sub("mJaB");
                        } else {
                            ambit::Tensor Y2 = ambit::Tensor::build(tensor_type_,"T2 sub * Gamma1",
                            {sizeh0,sizeh1,sub_virt_size0,sizep1});
                            Y2("xJaB") = Gamma1_.block("aa")("xy") * T2sub("yJaB");
                            C2.block(C2label)("qJsB") -= alpha * H2("qsax") * Y2("xJaB");
                        }
                    }

                } else {
                    // index i, a
                    for(const std::string& idxhp: C2label_jb_aa){
                        std::string idxh1 (1, idxhp[0]);
                        std::string idxp1 (1, idxhp[1]);
                        size_t sizeh1 = label_to_spacemo_[idxh1[0]].size();
                        size_t sizep1 = label_to_spacemo_[idxp1[0]].size();

                        std::string C2label = idxh1 + idx0 + idxp1 + idx1;
                        if(!C2.is_block(C2label)) continue;

                        std::string T2label = idxh1 + idxh0 + idxp1 + "V";
                        ambit::Tensor T2sub = ambit::Tensor::build(tensor_type_,"T2 sub",
                        {sizeh1,sizeh0,sizep1,sub_virt_size0});
                        T2sub.iterate([&](const std::vector<size_t>& i, double& value){
                            size_t idx = i[0] * sizeh0 * sizev * sizep1 + i[1] * sizev * sizep1
                                    + i[2] * sizev + sub_virt_mo0[i[3]];
                            value = T2.block(T2label).data()[idx];
                        });

                        if(idxh0 == "C"){
                            C2.block(C2label)("iQaS") -= alpha * H2("QSBM") * T2sub("iMaB");
                        } else {
                            ambit::Tensor Y2 = ambit::Tensor::build(tensor_type_,"T2 sub * Gamma1",
                            {sizeh1,sizeh0,sizep1,sub_virt_size0});
                            Y2("iXaB") = Gamma1_.block("AA")("XY") * T2sub("iYaB");
                            C2.block(C2label)("iQaS") -= alpha * H2("QSBX") * Y2("iXaB");
                        }
                    }
                }

            } // end virtual "a" partition
        } // end hole index "i" in H2[qsai]
    } // end spin-pure general indices "q" and "s" in H2[qsai]

    end_ = std::chrono::system_clock::now();
    tt2_ = std::chrono::system_clock::to_time_t(end_);
    if(profile_print_){
        outfile->Printf("    [V, T2] DF -> C2aa/bb PH exchange VP ended:   %s", std::ctime(&tt2_));
        outfile->Printf("    [V, T2] DF -> C2aa/bb PH exchange VP wall time %.1f s.",
                        compute_elapsed_time(start_,end_).count());
    }

    // C2 ba spin, H2[sQaI]
    start_ = std::chrono::system_clock::now();
    tt1_ = std::chrono::system_clock::to_time_t(start_);
    if(profile_print_){
        outfile->Printf("\n    [V, T2] DF -> C2ba PH exchange VP started: %s", std::ctime(&tt1_));
    }

    for(const std::string& idxqs: C2label_qs_ba){
        std::string idx0 (1, idxqs[0]);
        std::string idx1 (1, idxqs[1]);
        size_t size0 = label_to_spacemo_[idx0[0]].size();
        size_t size1 = label_to_spacemo_[idx1[0]].size();

        for(const std::string& idxh0: {"C","A"}){
            size_t sizeh0 = label_to_spacemo_[idxh0[0]].size();
            std::string Blabel0 = "Lv" + idx1;
            std::string Blabel1 = "L" + idx0 + idxh0;

            for(const auto& sub_virt_mo: sub_virt_mos){
                size_t sub_virt_size = sub_virt_mo.size();

                ambit::Tensor Bsub = ambit::Tensor::build(tensor_type_,"B sub",
                {sizeL, sub_virt_size, size1});
                Bsub.iterate([&](const std::vector<size_t>& i, double& value){
                    size_t idx = i[0] * sizev * size1 + sub_virt_mo[i[1]] * size1 + i[2];
                    value = B.block(Blabel0).data()[idx];
                });

                ambit::Tensor H2 = ambit::Tensor::build(tensor_type_,"H2 sub",
                {size1, size0, sub_virt_size, sizeh0});
                H2("sQaI") = Bsub("gas") * B.block(Blabel1)("gQI");

                for(const std::string& idxjb: C2label_jb_ab){
                    std::string idxh1 (1, idxjb[0]);
                    std::string idxp1 (1, idxjb[1]);
                    size_t sizeh1 = label_to_spacemo_[idxh1[0]].size();
                    size_t sizep1 = label_to_spacemo_[idxp1[0]].size();

                    std::string C2label = idxh1 + idx0 + idx1 + idxp1;
                    if(!C2.is_block(C2label)) continue;

                    std::string T2label = idxh1 + idxh0 + "v" + idxp1;
                    ambit::Tensor T2sub = ambit::Tensor::build(tensor_type_,"T2 sub",
                    {sizeh1,sizeh0,sub_virt_size,sizep1});
                    T2sub.iterate([&](const std::vector<size_t>& i, double& value){
                        size_t idx = i[0] * sizeh0 * sizev * sizep1 + i[1] * sizev * sizep1
                                + sub_virt_mo[i[2]] * sizep1 + i[3];
                        value = T2.block(T2label).data()[idx];
                    });

                    if(idxh0 == "C"){
                        C2.block(C2label)("iQsB") -= alpha * H2("sQaM") * T2sub("iMaB");
                    } else {
                        ambit::Tensor Y2 = ambit::Tensor::build(tensor_type_,"T2 sub * Gamma1",
                        {sizeh1,sizeh0,sub_virt_size,sizep1});
                        Y2("iXaB") = Gamma1_.block("AA")("XY") * T2sub("iYaB");
                        C2.block(C2label)("iQsB") -= alpha * H2("sQaX") * Y2("iXaB");
                    }
                }

            } // end virtual "a" partition
        } // end hole index "I" in H2[sQaI]
    } // end general indices "Q" and "s" in H2[sQaI]

    end_ = std::chrono::system_clock::now();
    tt2_ = std::chrono::system_clock::to_time_t(end_);
    if(profile_print_){
        outfile->Printf("    [V, T2] DF -> C2ba PH exchange VP ended:   %s", std::ctime(&tt2_));
        outfile->Printf("    [V, T2] DF -> C2ba PH exchange VP wall time %.1f s.",
                        compute_elapsed_time(start_,end_).count());
    }

    // C2 ab spin, H2[qSiA]
    start_ = std::chrono::system_clock::now();
    tt1_ = std::chrono::system_clock::to_time_t(start_);
    if(profile_print_){
        outfile->Printf("\n    [V, T2] DF -> C2ab PH exchange VP started: %s", std::ctime(&tt1_));
    }

    for(const std::string& idxqs: C2label_qs_ab){
        std::string idx0 (1, idxqs[0]);
        std::string idx1 (1, idxqs[1]);
        size_t size0 = label_to_spacemo_[idx0[0]].size();
        size_t size1 = label_to_spacemo_[idx1[0]].size();

        for(const std::string& idxh0: {"c","a"}){
            size_t sizeh0 = label_to_spacemo_[idxh0[0]].size();
            std::string Blabel0 = "LV" + idx1;
            std::string Blabel1 = "L" + idx0 + idxh0;

            for(const auto& sub_virt_mo: sub_virt_mos){
                size_t sub_virt_size = sub_virt_mo.size();

                ambit::Tensor Bsub = ambit::Tensor::build(tensor_type_,"B sub",
                {sizeL, sub_virt_size, size1});
                Bsub.iterate([&](const std::vector<size_t>& i, double& value){
                    size_t idx = i[0] * sizev * size1 + sub_virt_mo[i[1]] * size1 + i[2];
                    value = B.block(Blabel0).data()[idx];
                });

                ambit::Tensor H2 = ambit::Tensor::build(tensor_type_,"H2 sub",
                {size0, size1, sizeh0, sub_virt_size});
                H2("qSiA") = Bsub("gAS") * B.block(Blabel1)("gqi");

                for(const std::string& idxjb: C2label_jb_ba){
                    std::string idxh1 (1, idxjb[0]);
                    std::string idxp1 (1, idxjb[1]);
                    size_t sizeh1 = label_to_spacemo_[idxh1[0]].size();
                    size_t sizep1 = label_to_spacemo_[idxp1[0]].size();

                    std::string C2label = idx0 + idxh1 + idxp1 + idx1;
                    if(!C2.is_block(C2label)) continue;

                    std::string T2label = idxh0 + idxh1 + idxp1 + "V";
                    ambit::Tensor T2sub = ambit::Tensor::build(tensor_type_,"T2 sub",
                    {sizeh0,sizeh1,sizep1,sub_virt_size});
                    T2sub.iterate([&](const std::vector<size_t>& i, double& value){
                        size_t idx = i[0] * sizeh1 * sizev * sizep1 + i[1] * sizev * sizep1
                                + i[2] * sizev + sub_virt_mo[i[3]];
                        value = T2.block(T2label).data()[idx];
                    });

                    if(idxh0 == "c"){
                        C2.block(C2label)("qJaS") -= alpha * H2("qSmB") * T2sub("mJaB");
                    } else {
                        ambit::Tensor Y2 = ambit::Tensor::build(tensor_type_,"T2 sub * Gamma1",
                        {sizeh0,sizeh1,sizep1,sub_virt_size});
                        Y2("xJaB") = Gamma1_.block("aa")("xy") * T2sub("yJaB");
                        C2.block(C2label)("qJaS") -= alpha * H2("qSxB") * Y2("xJaB");
                    }
                }

            } // end virtual "A" partition
        } // end hole index "i" in H2[qSiA]
    } // end general indices "q" and "S" in H2[qSiA]

    end_ = std::chrono::system_clock::now();
    tt2_ = std::chrono::system_clock::to_time_t(end_);
    if(profile_print_){
        outfile->Printf("    [V, T2] DF -> C2ab PH exchange VP ended:   %s", std::ctime(&tt2_));
        outfile->Printf("    [V, T2] DF -> C2ab PH exchange VP wall time %.1f s.",
                        compute_elapsed_time(start_,end_).count());
    }
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
