#include <numeric>

#include <libpsio/psio.hpp>
#include <libpsio/psio.h>
#include <libmints/molecule.h>
#include <libmints/matrix.h>
#include <libmints/vector.h>
#include <libqt/qt.h>
#include "blockedtensorfactory.h"

#include "three_dsrg_mrpt2.h"
#include <vector>
#include <string>
#include <algorithm>
using namespace ambit;

namespace psi{ namespace libadaptive{

#ifdef _OPENMP
	#include <omp.h>
	bool THREE_DSRG_MRPT2::have_omp_ = true;
#else
   #define omp_get_max_threads() 1
   #define omp_get_thread_num() 0
   bool THREE_DSRG_MRPT2::have_omp_ = false;
#endif


THREE_DSRG_MRPT2::THREE_DSRG_MRPT2(Reference reference, boost::shared_ptr<Wavefunction> wfn, Options &options, ExplorerIntegrals* ints)
    : Wavefunction(options,_default_psio_lib_),
      reference_(reference),
      ints_(ints),
      tensor_type_(kCore),
      BTF(new BlockedTensorFactory(options))
{
    ///Need to erase all mo_space information
    ambit::BlockedTensor::reset_mo_spaces();
    // Copy the wavefunction information
    copy(wfn);

	num_threads_ = omp_get_max_threads();

    outfile->Printf("\n\n\t  ---------------------------------------------------------");
    outfile->Printf("\n\t      DF/CD - Driven Similarity Renormalization Group MBPT2");
    outfile->Printf("\n\t                   Kevin Hannon and Chenyang (York) Li");
    outfile->Printf("\n\t                    %4d thread(s) %s",num_threads_,have_omp_ ? "(OMP)" : "");
    outfile->Printf("\n\t  ---------------------------------------------------------");

    if(options_.get_bool("MEMORY_SUMMARY"))
    {
        BTF->print_memory_info();
    }

    startup();
    //if(false){
    //frozen_natural_orbitals();}
    print_summary();
}

THREE_DSRG_MRPT2::~THREE_DSRG_MRPT2()
{
    cleanup();
}

void THREE_DSRG_MRPT2::startup()
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

    BlockedTensor::set_expert_mode(true);

    //boost::shared_ptr<BlockedTensorFactory> BTFBlockedTensorFactory(options);

    //BlockedTensor::add_mo_space("c","m,n,µ,π",acore_mos,AlphaSpin);
    //BlockedTensor::add_mo_space("C","M,N,Ω,∏",bcore_mos,BetaSpin);
    BTF->add_mo_space("c","m,n,µ,π",acore_mos,AlphaSpin);
    BTF->add_mo_space("C","M,N,Ω,∏",bcore_mos,BetaSpin);

    core_ = acore_mos.size();

    //BlockedTensor::add_mo_space("a","uvwxyz",aactv_mos,AlphaSpin);
    //BlockedTensor::add_mo_space("A","UVWXYZ",bactv_mos,BetaSpin);
    BTF->add_mo_space("a","uvwxyz",aactv_mos,AlphaSpin);
    BTF->add_mo_space("A","UVWXYZ",bactv_mos,BetaSpin);
    active_ = aactv_mos.size();

    //BlockedTensor::add_mo_space("v","e,f,ε,φ",avirt_mos,AlphaSpin);
    //BlockedTensor::add_mo_space("V","E,F,Ƒ,Ǝ",bvirt_mos,BetaSpin);
    BTF->add_mo_space("v","e,f,ε,φ",avirt_mos,AlphaSpin);
    BTF->add_mo_space("V","E,F,Ƒ,Ǝ",bvirt_mos,BetaSpin);
    virtual_ = avirt_mos.size();

    //BlockedTensor::add_composite_mo_space("h","ijkl",{"c","a"});
    //BlockedTensor::add_composite_mo_space("H","IJKL",{"C","A"});

    //BlockedTensor::add_composite_mo_space("p","abcd",{"a","v"});
    //BlockedTensor::add_composite_mo_space("P","ABCD",{"A","V"});

    //BlockedTensor::add_composite_mo_space("g","pqrs",{"c","a","v"});
    //BlockedTensor::add_composite_mo_space("G","PQRS",{"C","A","V"});

    BTF->add_composite_mo_space("h","ijkl",{"c","a"});
    BTF->add_composite_mo_space("H","IJKL",{"C","A"});

    BTF->add_composite_mo_space("p","abcd",{"a","v"});
    BTF->add_composite_mo_space("P","ABCD",{"A","V"});

    BTF->add_composite_mo_space("g","pqrs",{"c","a","v"});
    BTF->add_composite_mo_space("G","PQRS",{"C","A","V"});
    // These two blocks of functions create a Blocked tensor
    std::vector<std::string> hhpp_no_cv = BTF->generate_indices("cav", "hhpp");
    no_hhpp_ = hhpp_no_cv;


    // These two blocks of functions create a Blocked tensor
    // And fill the tensor
    // Just need to fill full spin cases.  Mixed alpha beta is created via alphaalpha beta beta

    size_t nthree = ints_->nthree();
    std::vector<size_t> nauxpi(nthree);
    std::iota(nauxpi.begin(), nauxpi.end(),0);

    //BlockedTensor::add_mo_space("@","$",nauxpi,NoSpin);
    //BlockedTensor::add_mo_space("d","g",nauxpi,NoSpin);
    BTF->add_mo_space("d","g",nauxpi,NoSpin);

    ThreeIntegral = BTF->build(tensor_type_,"ThreeInt",{"dgg","dGG"});
    //ThreeIntegral = BTF->build(tensor_type_,"ThreeInt",{"dgg","dGG"});

    ThreeIntegral.iterate([&](const std::vector<size_t>& i,const std::vector<SpinType>& spin,double& value){
        value = ints_->get_three_integral(i[0],i[1],i[2]);
    });


    H = BTF->build(tensor_type_,"H",spin_cases({"gg"}));
    //Returns a vector of all combinations for gggg
    std::vector<std::string> list_of_entire_space = BTF->generate_indices("cav", "all");
    //Function below will return a list of pphh
    std::vector<std::string> list_of_pphh_V = BTF->generate_indices("vac", "pphh");

    //Avoiding building pqrs integrals just abij -> some tricks needed to get this to work.
    //See Vpphh
    V = BTF->build(tensor_type_,"V",BTF->spin_cases_avoid(list_of_pphh_V));


    Gamma1 = BTF->build(tensor_type_,"Gamma1",spin_cases({"hh"}));
    Eta1 = BTF->build(tensor_type_,"Eta1",spin_cases({"pp"}));
    Lambda2 = BTF->build(tensor_type_,"Lambda2",spin_cases({"aaaa"}));
    Lambda3 = BTF->build(tensor_type_,"Lambda3",spin_cases({"aaaaaa"}));
    F = BTF->build(tensor_type_,"Fock",spin_cases({"gg"}));
    Delta1 = BTF->build(tensor_type_,"Delta1",spin_cases({"hp"}));

    Delta2 = BTF->build(tensor_type_,"Delta2",BTF->spin_cases_avoid(hhpp_no_cv));

    RDelta1 = BTF->build(tensor_type_,"RDelta1",spin_cases({"hp"}));

    //Need to avoid building ccvv part of this
    //ccvv only used for creating T2
    RDelta2 = BTF->build(tensor_type_,"RDelta2",BTF->spin_cases_avoid(hhpp_no_cv));


    T1 = BTF->build(tensor_type_,"T1 Amplitudes",spin_cases({"hp"}));

    RExp1 = BTF->build(tensor_type_,"RExp1",spin_cases({"hp"}));
    RExp2 = BTF->build(tensor_type_,"RExp2",BTF->spin_cases_avoid(hhpp_no_cv));
    //all_spin = RExp2.get.();
    T2pr   = BTF->build(tensor_type_,"T2 Amplitudes not all",
             BTF->spin_cases_avoid(no_hhpp_));

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
    //V.iterate([&](const std::vector<size_t>& i,const std::vector<SpinType>& spin,double& value){
    //    if ((spin[0] == AlphaSpin) and (spin[1] == AlphaSpin)) value = ints_->aptei_aa(i[0],i[1],i[2],i[3]);
    //    if ((spin[0] == AlphaSpin) and (spin[1] == BetaSpin) ) value = ints_->aptei_ab(i[0],i[1],i[2],i[3]);
    //    if ((spin[0] == BetaSpin)  and (spin[1] == BetaSpin) ) value = ints_->aptei_bb(i[0],i[1],i[2],i[3]);
    //});

    V["abij"] =  ThreeIntegral["gai"]*ThreeIntegral["gbj"];
    V["abij"] -= ThreeIntegral["gaj"]*ThreeIntegral["gbi"];

    V["aBiJ"] =  ThreeIntegral["gai"]*ThreeIntegral["gBJ"];

    V["ABIJ"] =  ThreeIntegral["gAI"]*ThreeIntegral["gBJ"];
    V["ABIJ"] -= ThreeIntegral["gAJ"]*ThreeIntegral["gBI"];



 //   V.iterate([&](const std::vector<size_t>& i,const std::vector<SpinType>& spin,double& value){
 //       if ((spin[0] == AlphaSpin) and (spin[1] == AlphaSpin)) value = ints_->aptei_aa(i[0],i[1],i[2],i[3]);
 //       if ((spin[0] == AlphaSpin) and (spin[1] == BetaSpin) ) value = ints_->aptei_ab(i[0],i[1],i[2],i[3]);
 //       if ((spin[0] == BetaSpin)  and (spin[1] == BetaSpin) ) value = ints_->aptei_bb(i[0],i[1],i[2],i[3]);
 //   });
    F["pq"]  = H["pq"];
    F["pq"] += ThreeIntegral["gpq"]*ThreeIntegral["gji"]*Gamma1["ij"];
    F["pq"] -= ThreeIntegral["gpi"]*ThreeIntegral["gjq"]*Gamma1["ij"];
    F["pq"] += ThreeIntegral["gpq"]*ThreeIntegral["gJI"]*Gamma1["IJ"];

    F["PQ"]  = H["PQ"];
    F["PQ"] += ThreeIntegral["gji"]*ThreeIntegral["gPQ"]*Gamma1["ij"];
    F["PQ"] += ThreeIntegral["gPQ"]*ThreeIntegral["gJI"]*Gamma1["IJ"];
    F["PQ"] -= ThreeIntegral["gPI"]*ThreeIntegral["gJQ"]*Gamma1["IJ"];

    size_t ncmo_ = ints_->ncmo();

    Fa.reserve(ncmo_);
    Fb.reserve(ncmo_);

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
    Delta2.iterate([&](const std::vector<size_t>& i,const std::vector<SpinType>& spin,double& value){
        if (spin[0] == AlphaSpin){
            value = Fa[i[0]]  + Fa[i[1]] - Fa[i[2]]- Fa[i[3]];
        }else if (spin[0]  == BetaSpin){
            value = Fb[i[0]]  + Fb[i[1]] - Fb[i[2]]- Fb[i[3]];
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
//    for (size_t i = 0; i < naocc*navir; i++)
//    {
//        if(RExpEigs->get(i) < -1.0e-8) count++;
//    }

    // Print levels
    print_ = options_.get_int("PRINT");

    if(print_ > 1){
        Gamma1.print(stdout);
        Eta1.print(stdout);
        F.print(stdout);
        H.print(stdout);
    }
    if(print_ > 2){
        V.print(stdout);
        Lambda2.print(stdout);
    }
    if(print_ > 3){
        Lambda3.print(stdout);
    }
}

void THREE_DSRG_MRPT2::print_summary()
{
    // Print a summary
    std::vector<std::pair<std::string,int>> calculation_info;

    std::vector<std::pair<std::string,double>> calculation_info_double{
        {"Flow parameter",s_},
        {"Cholesky Tolerance", options_.get_double("CHOLESKY_TOLERANCE")},
        {"Taylor expansion threshold",std::pow(10.0,-double(taylor_threshold_))}};

    std::vector<std::pair<std::string,std::string>> calculation_info_string{
        {"int_type", options_.get_str("INT_TYPE")},
        {"ccvv_algorithm",options_.get_str("ccvv_algorithm")}};

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

void THREE_DSRG_MRPT2::cleanup()
{
}

double THREE_DSRG_MRPT2::renormalized_denominator(double D)
{
    double Z = std::sqrt(s_) * D;
    if(std::fabs(Z) < std::pow(0.1, taylor_threshold_)){
        return Taylor_Exp(Z, taylor_order_) * std::sqrt(s_);
    }else{
        return (1.0 - std::exp(-s_ * std::pow(D, 2.0))) / D;
    }
}

double THREE_DSRG_MRPT2::compute_energy()
{
    // Compute reference
    //    Eref = compute_ref();

        // Compute T2 and T1
        compute_t2();
        check_t2();
        compute_t1();
        check_t1();

        // Compute effective integrals
        renormalize_V();
        renormalize_F();
        if(print_ > 1)  F.print(stdout); // The actv-actv block is different but OK.
        if(print_ > 2){
            T1.print(stdout);
            T2pr.print(stdout);
            V.print(stdout);
        }

        // Compute DSRG-MRPT2 correlation energy
        // Compute DSRG-MRPT2 correlation energy
        double Etemp  = 0.0;
        double EVT2   = 0.0;
        double Ecorr  = 0.0;
        double Etotal = 0.0;
        std::vector<std::pair<std::string,double>> energy;
        energy.push_back({"E0 (reference)", Eref});

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
        Etotal = Ecorr + Eref;
        energy.push_back({"<[V, T2]>", EVT2});
        energy.push_back({"DSRG-MRPT2 correlation energy", Ecorr});
        energy.push_back({"DSRG-MRPT2 total energy", Etotal});

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

    double THREE_DSRG_MRPT2::compute_ref()
{
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

    outfile->Printf("\n Reference Energy = %12.8f", E + frozen_core_energy + Enuc );
    return E + frozen_core_energy + Enuc;
}

void THREE_DSRG_MRPT2::compute_t2()
{
    std::string str = "Computing T2";
    outfile->Printf("\n    %-36s ...", str.c_str());
    Timer timer;

    T2pr["ijab"] = V["abij"] * RDelta2["ijab"];
    T2pr["iJaB"] = V["aBiJ"] * RDelta2["iJaB"];
    T2pr["IJAB"] = V["ABIJ"] * RDelta2["IJAB"];

    // zero internal amplitudes
    T2pr.block("aaaa").zero();
    T2pr.block("aAaA").zero();
    T2pr.block("AAAA").zero();


    outfile->Printf("...Done. Timing %15.6f s", timer.get());

}
void THREE_DSRG_MRPT2::check_t2()
{
    // norm and maximum of T2 amplitudes
    T2norm = 0.0; T2max = 0.0;
    std::vector<std::string> T2blocks = T2pr.block_labels();
    for(const std::string& block: T2blocks){
        Tensor temp = T2pr.block(block);
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

void THREE_DSRG_MRPT2::compute_t1()
{
    //A temporary tensor to use for the building of T1
    //Francesco's library does not handle repeating indices between 3 different terms, so need to form an intermediate
    //via a pointwise multiplcation
    std::string str = "Computing T1";
    outfile->Printf("\n    %-36s ...", str.c_str());
    Timer timer;
    BlockedTensor temp;
    temp = BTF->build(tensor_type_,"temp",spin_cases({"aa"}));
    temp["xu"] = Gamma1["xu"] * Delta1["xu"];
    temp["XU"] = Gamma1["XU"] * Delta1["XU"];

    //Form the T1 amplitudes

    BlockedTensor N = BTF->build(tensor_type_,"N",spin_cases({"hp"}));

    N["ia"]  = F["ia"];
    N["ia"] += temp["xu"] * T2pr["iuax"];
    N["ia"] += temp["XU"] * T2pr["iUaX"];

    T1["ia"] = N["ia"] * RDelta1["ia"];

    N["IA"]  = F["IA"];
    N["IA"] += temp["xu"] * T2pr["uIxA"];
    N["IA"] += temp["XU"] * T2pr["IUAX"];
    T1["IA"] = N["IA"] * RDelta1["IA"];

    T1.block("AA").zero();
    T1.block("aa").zero();

    outfile->Printf("...Done. Timing %15.6f s", timer.get());
}
void THREE_DSRG_MRPT2::check_t1()
{
    // norm and maximum of T1 amplitudes
    T1norm = T1.norm(); T1max = 0.0;
    T1.iterate([&](const std::vector<size_t>& i,const std::vector<SpinType>& spin,double& value){
            T1max = T1max > fabs(value) ? T1max : fabs(value);
    });
}

void THREE_DSRG_MRPT2::renormalize_V()
{
    Timer timer;
    std::string str = "Renormalizing V";
    outfile->Printf("\n    %-36s ...", str.c_str());
    std::vector<std::string> list_of_pphh_V = BTF->generate_indices("vac", "pphh");

    // Put RExp2 into a shared matrix.
    BlockedTensor v = BTF->build(tensor_type_,"v",BTF->spin_cases_avoid(list_of_pphh_V));
    v["abij"] = V["abij"];
    v["aBiJ"] = V["aBiJ"];
    v["ABIJ"] = V["ABIJ"];

    //V["ijab"] += v["ijab"] * RExp2["ijab"];
    //V["iJaB"] += v["iJaB"] * RExp2["iJaB"];
    //V["IJAB"] += v["IJAB"] * RExp2["IJAB"];

    V["abij"] += v["abij"] * RExp2["ijab"];
    V["aBiJ"] += v["aBiJ"] * RExp2["iJaB"];
    V["ABIJ"] += v["ABIJ"] * RExp2["IJAB"];

    outfile->Printf("...Done. Timing %15.6f s", timer.get());
}

void THREE_DSRG_MRPT2::renormalize_F()
{
    Timer timer;

    std::string str = "Renormalizing F";
    outfile->Printf("\n    %-36s ...", str.c_str());

    BlockedTensor temp_aa = BTF->build(tensor_type_,"temp_aa",spin_cases({"aa"}));
    temp_aa["xu"] = Gamma1["xu"] * Delta1["xu"];
    temp_aa["XU"] = Gamma1["XU"] * Delta1["XU"];

    BlockedTensor temp1 = BTF->build(tensor_type_,"temp1",spin_cases({"hp"}));
    BlockedTensor temp2 = BTF->build(tensor_type_,"temp2",spin_cases({"hp"}));

    temp1["ia"] += temp_aa["xu"] * T2pr["iuax"];
    temp1["ia"] += temp_aa["XU"] * T2pr["iUaX"];
    temp2["ia"] += F["ia"] * RExp1["ia"];
    temp2["ia"] += temp1["ia"] * RExp1["ia"];

    temp1["IA"] += temp_aa["xu"] * T2pr["uIxA"];
    temp1["IA"] += temp_aa["XU"] * T2pr["IUAX"];
    temp2["IA"] += F["IA"] * RExp1["IA"];
    temp2["IA"] += temp1["IA"] * RExp1["IA"];

//    temp["ai"] += temp_aa["xu"] * T2["iuax"];
//    temp["ai"] += temp_aa["XU"] * T2["iUaX"];
//    F["ai"] += F["ai"] % RExp1["ia"];  // TODO <- is this legal in ambit???
//    F["ai"] += temp["ai"] % RExp1["ia"];
    F["ia"] += temp2["ia"];
    F["ai"] += temp2["ia"];

//    temp["AI"] += temp_aa["xu"] * T2["uIxA"];
//    temp["AI"] += temp_aa["XU"] * T2["IUAX"];
//    F["AI"] += F["AI"] % RExp1["IA"];
//    F["AI"] += temp["AI"] % RExp1["IA"];
    F["IA"] += temp2["IA"];
    F["AI"] += temp2["IA"];
    outfile->Printf("...Done. Timing %15.6f s", timer.get());
}

double THREE_DSRG_MRPT2::E_FT1()
{
    Timer timer;
    std::string str = "Computing <[F,T1]>";
    outfile->Printf("\n    %-36s ...", str.c_str());
    double E = 0.0;
    BlockedTensor temp;
    temp = BTF->build(tensor_type_,"temp",spin_cases({"hp"}));

    temp["jb"] += T1["ia"] * Eta1["ab"] * Gamma1["ji"];
    temp["JB"] += T1["IA"] * Eta1["AB"] * Gamma1["JI"];

    //E += T1["ia"]*Eta1["ab"]* Gamma1["ji"] * F["bj"];
    E += temp["jb"] * F["bj"];
    //E += T1["IA"]*Eta1["AB"]* Gamma1["JI"] * F["BJ"];
    E += temp["JB"] * F["BJ"];

    outfile->Printf("...Done. Timing %15.6f s", timer.get());

    return E;
}

double THREE_DSRG_MRPT2::E_VT1()
{
    Timer timer;
    std::string str = "Computing <[V, T1]>";
    outfile->Printf("\n    %-36s ...", str.c_str());
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

    outfile->Printf("...Done. Timing %15.6f s", timer.get());

    return E;
}

double THREE_DSRG_MRPT2::E_FT2()
{
    Timer timer;
    std::string str = "Computing <[F, T2]>";
    outfile->Printf("\n    %-36s ...", str.c_str());
    double E = 0.0;
    BlockedTensor temp;
    temp = BTF->build(tensor_type_,"temp",spin_cases({"aaaa"}));


    temp["uvxy"] += F["ex"] * T2pr["uvey"];
    temp["uvxy"] -= F["vm"] * T2pr["umxy"];

    temp["UVXY"] += F["EX"] * T2pr["UVEY"];
    temp["UVXY"] -= F["VM"] * T2pr["UMXY"];

    temp["uVxY"] += F["ex"] * T2pr["uVeY"];
    temp["uVxY"] += F["EY"] * T2pr["uVxE"];
    temp["uVxY"] -= F["VM"] * T2pr["uMxY"];
    temp["uVxY"] -= F["um"] * T2pr["mVxY"];

    E += 0.5 * temp["uvxy"] * Lambda2["xyuv"];
    E += 0.5 * temp["UVXY"] * Lambda2["XYUV"];
    E += temp["uVxY"] * Lambda2["xYuV"];

    outfile->Printf("...Done. Timing %15.6f s", timer.get());
    return E;
}

double THREE_DSRG_MRPT2::E_VT2_2()
{
    double E = 0.0;
    Timer timer;
    std::string str = "Computing <[V, T2]> (C_2)^4 (no ccvv)";
    outfile->Printf("\n    %-36s ...", str.c_str());

    BlockedTensor temp1;
    BlockedTensor temp2;
    temp1 = BTF->build(tensor_type_,"temp1",BTF->spin_cases_avoid(no_hhpp_));
    temp2 = BTF->build(tensor_type_,"temp2",BTF->spin_cases_avoid(no_hhpp_));


    //Calculates all but ccvv, cCvV, and CCVV energies

    temp1["klab"] += T2pr["ijab"] * Gamma1["ki"] * Gamma1["lj"];
    temp2["klcd"] += temp1["klab"] * Eta1["ac"] * Eta1["bd"];

    temp1["KLAB"] += T2pr["IJAB"] * Gamma1["KI"] * Gamma1["LJ"];
    temp2["KLCD"] += temp1["KLAB"] * Eta1["AC"] * Eta1["BD"];

    temp1["kLaB"] += T2pr["iJaB"] * Gamma1["ki"] * Gamma1["LJ"];
    temp2["kLcD"] += temp1["kLaB"] * Eta1["ac"] * Eta1["BD"];

    E += 0.25 * V["CDKL"] * temp2["KLCD"];
    E += 0.25 * V["cdkl"] * temp2["klcd"];
    E += V["cDkL"] * temp2["kLcD"];
    outfile->Printf("...Done. Timing %15.6f s", timer.get());


    double Eccvv = 0.0;
    std::string strccvv = "Computing <[V, T2]> (C_2)^4 ccvv";
    outfile->Printf("\n    %-36s ...", strccvv.c_str());
    Timer ccvv_timer;

    if(options_.get_str("ccvv_algorithm")=="CORE")
    {
        Eccvv = E_VT2_2_core();

    }
    else if(options_.get_str("ccvv_algorithm")=="FLY_OPENMP")
    {
        Eccvv = E_VT2_2_fly_openmp();
    }
    else if(options_.get_str("ccvv_algorithm")=="FLY")
    {
        Eccvv = E_VT2_2_ambit();
    }
    else
    {
        outfile->Printf("\n Specify a correct algorithm string");

    }
    outfile->Printf("...Done. Timing %15.6f s", ccvv_timer.get());


    return (E + Eccvv);
}

double THREE_DSRG_MRPT2::E_VT2_4HH()
{
    Timer timer;
    std::string str = "Computing <[V, T2]> 4HH";
    outfile->Printf("\n    %-36s ...", str.c_str());
    double E = 0.0;
    BlockedTensor temp1;
    BlockedTensor temp2;
    temp1 = BTF->build(tensor_type_,"temp1", spin_cases({"aahh"}));
    temp2 = BTF->build(tensor_type_,"temp2", spin_cases({"aaaa"}));

    temp1["uvij"] += V["uvkl"] * Gamma1["ki"] * Gamma1["lj"];
    temp1["UVIJ"] += V["UVKL"] * Gamma1["KI"] * Gamma1["LJ"];
    temp1["uViJ"] += V["uVkL"] * Gamma1["ki"] * Gamma1["LJ"];

    temp2["uvxy"] += temp1["uvij"] * T2pr["ijxy"];
    temp2["UVXY"] += temp1["UVIJ"] * T2pr["IJXY"];
    temp2["uVxY"] += temp1["uViJ"] * T2pr["iJxY"];

    E += 0.125 * Lambda2["xyuv"] * temp2["uvxy"];
    E += 0.125 * Lambda2["XYUV"] * temp2["UVXY"];
    E += Lambda2["xYuV"] * temp2["uVxY"];

    outfile->Printf("...Done. Timing %15.6f s", timer.get());

    return E;
}

double THREE_DSRG_MRPT2::E_VT2_4PP()
{
    Timer timer;
    double E = 0.0;
    BlockedTensor temp1;
    BlockedTensor temp2;

    std::string str = "Computing <V, T2]> 4PP";
    outfile->Printf("\n    %-36s ...", str.c_str());

    temp1 = BTF->build(tensor_type_,"temp1", spin_cases({"aapp"}));
    temp2 = BTF->build(tensor_type_,"temp2", spin_cases({"aaaa"}));

    temp1["uvcd"] += T2pr["uvab"] * Eta1["ac"] * Eta1["bd"];
    temp1["UVCD"] += T2pr["UVAB"] * Eta1["AC"] * Eta1["BD"];
    temp1["uVcD"] += T2pr["uVaB"] * Eta1["ac"] * Eta1["BD"];

    temp2["uvxy"] += temp1["uvcd"] * V["cdxy"];
    temp2["UVXY"] += temp1["UVCD"] * V["CDXY"];
    temp2["uVxY"] += temp1["uVcD"] * V["cDxY"];

    E += 0.125 * Lambda2["xyuv"] * temp2["uvxy"];
    E += 0.125 * Lambda2["XYUV"] * temp2["UVXY"];
    E += Lambda2["xYuV"] * temp2["uVxY"];

    outfile->Printf("  Done. Timing %15.6f s", timer.get());
    return E;
}

double THREE_DSRG_MRPT2::E_VT2_4PH()
{
    Timer timer;
    double E = 0.0;
    std::string str = "Computing [V, T2] 4PH";
    outfile->Printf("\n    %-36s ...", str.c_str());
//    BlockedTensor temp;
//    temp = BTF->build(tensor_type_,"temp", "aaaa");

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
    temp1 = BTF->build(tensor_type_,"temp1", BTF->spin_cases_avoid(no_hhpp_));
    temp2 = BTF->build(tensor_type_,"temp2", spin_cases({"aaaa"}));

    temp1["juby"]  = T2pr["iuay"] * Gamma1["ji"] * Eta1["ab"];
    temp2["uvxy"] += V["vbjx"] * temp1["juby"];

    temp1["uJyB"]  = T2pr["uIyA"] * Gamma1["JI"] * Eta1["AB"];
    temp2["uvxy"] -= V["vBxJ"] * temp1["uJyB"];
    E += temp2["uvxy"] * Lambda2["xyuv"];

    temp1["JUBY"]  = T2pr["IUAY"] * Gamma1["IJ"] * Eta1["AB"];
    temp2["UVXY"] += V["VBJX"] * temp1["JUBY"];

    temp1["jUbY"]  = T2pr["iUaY"] * Gamma1["ji"] * Eta1["ab"];
    temp2["UVXY"] -= V["bVjX"] * temp1["jUbY"];
    E += temp2["UVXY"] * Lambda2["XYUV"];

    temp1["jVbY"]  = T2pr["iVaY"] * Gamma1["ji"] * Eta1["ab"];
    temp2["uVxY"] -= V["ubjx"] * temp1["jVbY"];

    temp1["JVBY"]  = T2pr["IVAY"] * Gamma1["JI"] * Eta1["AB"];
    temp2["uVxY"] += V["uBxJ"] * temp1["JVBY"];

    temp1["jubx"]  = T2pr["iuax"] * Gamma1["ji"] * Eta1["ab"];
    temp2["uVxY"] += V["bVjY"] * temp1["jubx"];

    temp1["uJxB"]  = T2pr["uIxA"] * Gamma1["JI"] * Eta1["AB"];
    temp2["uVxY"] -= V["VBJY"] * temp1["uJxB"];

    temp1["uJbY"]  = T2pr["uIaY"] * Gamma1["JI"] * Eta1["ab"];
    temp2["uVxY"] -= V["bVxJ"] * temp1["uJbY"];

    temp1["jVxB"]  = T2pr["iVxA"] * Gamma1["ji"] * Eta1["AB"];
    temp2["uVxY"] -= V["uBjY"] * temp1["jVxB"];
    E += temp2["uVxY"] * Lambda2["xYuV"];

    outfile->Printf("  Done. Timing %15.6f s", timer.get());
    return E;
}

double THREE_DSRG_MRPT2::E_VT2_6()
{
    Timer timer;
    std::string str = "Computing [V, T2] λ3";
    outfile->Printf("\n    %-36s ...", str.c_str());
    double E = 0.0;
    BlockedTensor temp;
    temp = BTF->build(tensor_type_,"temp", spin_cases({"aaaaaa"}));

    temp["uvwxyz"] += V["uviz"] * T2pr["iwxy"];      //  aaaaaa from hole
    temp["uvwxyz"] += V["waxy"] * T2pr["uvaz"];      //  aaaaaa from particle
    temp["UVWXYZ"] += V["UVIZ"] * T2pr["IWXY"];      //  AAAAAA from hole
    temp["UVWXYZ"] += V["WAXY"] * T2pr["UVAZ"];      //  AAAAAA from particle
    E += 0.25 * temp["uvwxyz"] * Lambda3["xyzuvw"];
    E += 0.25 * temp["UVWXYZ"] * Lambda3["XYZUVW"];

    temp["uvWxyZ"] -= V["uviy"] * T2pr["iWxZ"];      //  aaAaaA from hole
    temp["uvWxyZ"] -= V["uWiZ"] * T2pr["ivxy"];      //  aaAaaA from hole
    temp["uvWxyZ"] += V["uWyI"] * T2pr["vIxZ"];      //  aaAaaA from hole
    temp["uvWxyZ"] += V["uWyI"] * T2pr["vIxZ"];      //  aaAaaA from hole

    temp["uvWxyZ"] += V["aWxZ"] * T2pr["uvay"];      //  aaAaaA from particle
    temp["uvWxyZ"] -= V["vaxy"] * T2pr["uWaZ"];      //  aaAaaA from particle
    temp["uvWxyZ"] -= V["vAxZ"] * T2pr["uWyA"];      //  aaAaaA from particle
    temp["uvWxyZ"] -= V["vAxZ"] * T2pr["uWyA"];      //  aaAaaA from particle

    E += 0.50 * temp["uvWxyZ"] * Lambda3["xyZuvW"];

    temp["uVWxYZ"] -= V["VWIZ"] * T2pr["uIxY"];      //  aAAaAA from hole
    temp["uVWxYZ"] -= V["uVxI"] * T2pr["IWYZ"];      //  aAAaAA from hole
    temp["uVWxYZ"] += V["uViZ"] * T2pr["iWxY"];      //  aAAaAA from hole
    temp["uVWxYZ"] += V["uViZ"] * T2pr["iWxY"];      //  aAAaAA from hole

    temp["uVWxYZ"] += V["uAxY"] * T2pr["VWAZ"];      //  aAAaAA from particle
    temp["uVWxYZ"] -= V["WAYZ"] * T2pr["uVxA"];      //  aAAaAA from particle
    temp["uVWxYZ"] -= V["aWxY"] * T2pr["uVaZ"];      //  aAAaAA from particle
    temp["uVWxYZ"] -= V["aWxY"] * T2pr["uVaZ"];      //  aAAaAA from particle

    E += 0.5 * temp["uVWxYZ"] * Lambda3["xYZuVW"];

    outfile->Printf("...Done. Timing %15.6f s", timer.get());
    return E;
}
std::vector<std::string> THREE_DSRG_MRPT2::spin_cases_avoid(const std::vector<std::string>& in_str_vec)
{

    std::vector<std::string> out_str_vec;
    for(const std::string spin : in_str_vec){
        size_t spin_ind  = spin.find('a');
        size_t spin_ind2 = spin.find('A');
        if(spin_ind != std::string::npos|| spin_ind2 != std::string::npos){
            out_str_vec.push_back(spin);
        }
    }
    return out_str_vec;
}
std::vector<std::string> THREE_DSRG_MRPT2::generate_all_indices(const std::string in_str, const std::string type)
{
    std::vector<std::string> return_string;

    //Hardlined for 4 character strings
    if(type=="all")
    {
        for(int i = 0; i < 3; i++){
            for(int j = 0; j < 3; j++){
                for(int k = 0; k < 3; k++){
                    for(int l = 0; l < 3; l++){
                        std::string one_string_lower;
                        std::string one_string_upper;
                        std::string one_string_mixed;

                        one_string_lower.push_back(in_str[i]);
                        one_string_lower.push_back(in_str[j]);
                        one_string_lower.push_back(in_str[k]);
                        one_string_lower.push_back(in_str[l]);

                        one_string_upper.push_back(std::toupper(in_str[i]));
                        one_string_upper.push_back(std::toupper(in_str[j]));
                        one_string_upper.push_back(std::toupper(in_str[k]));
                        one_string_upper.push_back(std::toupper(in_str[l]));

                        one_string_mixed.push_back(in_str[i]);
                        one_string_mixed.push_back(std::toupper(in_str[j]));
                        one_string_mixed.push_back(in_str[k]);
                        one_string_mixed.push_back(std::toupper(in_str[l]));

                        return_string.push_back(one_string_lower);
                        return_string.push_back(one_string_upper);
                        return_string.push_back(one_string_mixed);
                    }

                }
            }
        }
    }
    else 
    {
       //This batch of code will take a string of three letter string specifiying core, active, or virtual
       //cav-> generates all possible kinds of spaces
        for(int i = 0; i < 2; i++){
            for(int j = 0; j < 2; j++){
                for(int k = 0; k < 2; k++){
                    for(int l = 0; l < 2; l++){
                        std::string one_string_lower;
                        std::string one_string_upper;
                        std::string one_string_mixed;

                        one_string_lower.push_back(in_str[i]);
                        one_string_lower.push_back(in_str[j]);
                        one_string_lower.push_back(in_str[k + 1]);
                        one_string_lower.push_back(in_str[l + 1]);

                        one_string_upper.push_back(std::toupper(in_str[i]));
                        one_string_upper.push_back(std::toupper(in_str[j]));
                        one_string_upper.push_back(std::toupper(in_str[k + 1]));
                        one_string_upper.push_back(std::toupper(in_str[l + 1]));

                        one_string_mixed.push_back(in_str[i]);
                        one_string_mixed.push_back(std::toupper(in_str[j]));
                        one_string_mixed.push_back(in_str[k + 1]);
                        one_string_mixed.push_back(std::toupper(in_str[l + 1]));

                        return_string.push_back(one_string_lower);
                        return_string.push_back(one_string_upper);
                        return_string.push_back(one_string_mixed);
                    }

                }
            }
        }
    }


    return return_string;

}
void THREE_DSRG_MRPT2::frozen_natural_orbitals()
{
     //outfile->Printf("\n About to compute MP2-like frozen natural orbitals");
    BlockedTensor Dfv = BTF->build(tensor_type_,"MP2Density", spin_cases({"pp"}));
    BlockedTensor Vhap = BTF->build(tensor_type_,"V", spin_cases({"ppvv"}));
    Vhap = V;
    

    Dfv["ef"] += 0.5 * Vhap["εfij"]*Vhap["ijεe"] * Delta2["εfij"] * Delta2["εeij"];
     


}
double THREE_DSRG_MRPT2::E_VT2_2_fly_openmp()
{
    double Eflyalpha = 0.0;
    double Eflybeta = 0.0;
    double Eflymixed = 0.0;
    double Efly = 0.0;
    size_t nthree = ints_->nthree();
    size_t ncmo   = ints_->ncmo();
    size_t nmo_   = ints_->nmo();
    #pragma omp parallel for num_threads(num_threads_) \
    schedule(dynamic) \
    reduction(+:Eflyalpha, Eflybeta, Eflymixed)
    for(size_t mind = 0; mind < core_; mind++){
        for(size_t nind = 0; nind < core_; nind++){
            for(size_t eind = 0; eind < virtual_; eind++){
                for(size_t find = 0; find < virtual_; find++){
                    //These are used because active is not partitioned as simple as
                    //core orbs -- active orbs -- virtual
                    //This also takes in account symmetry labeled
                    size_t m = acore_mos[mind];
                    size_t n = acore_mos[nind];
                    size_t e = avirt_mos[eind];
                    size_t f = bvirt_mos[find];
                    size_t mb = bcore_mos[mind];
                    size_t nb = bcore_mos[nind];
                    size_t eb = bvirt_mos[eind];
                    size_t fb = bvirt_mos[find];
                    double vmnefalpha = 0.0;

                    double vmnefalphaR = 0.0;
                    double vmnefbeta = 0.0;
                    double vmnefalphaC = 0.0;
                    double vmnefalphaE = 0.0;
                    double vmnefbetaC = 0.0;
                    double vmnefbetaE = 0.0;
                    double vmnefbetaR = 0.0;
                    double vmnefmixed = 0.0;
                    double vmnefmixedC = 0.0;
                    double vmnefmixedR = 0.0;
                    double t2alpha = 0.0;
                    double t2mixed = 0.0;
                    double t2beta = 0.0;
                    vmnefalphaC = C_DDOT(nthree,
                            &(ints_->get_three_integral_pointer()[0][m * ncmo + e]),nmo_ * nmo_,
                            &(ints_->get_three_integral_pointer()[0][n * ncmo + f]),nmo_ * nmo_);
                     vmnefalphaE = C_DDOT(nthree,
                            &(ints_->get_three_integral_pointer()[0][m * ncmo + f]),nmo_ * nmo_,
                            &(ints_->get_three_integral_pointer()[0][n * ncmo + e]),nmo_ * nmo_);
                    vmnefbetaC = C_DDOT(nthree,
                            &(ints_->get_three_integral_pointer()[0][mb * ncmo + eb]),nmo_ * nmo_,
                            &(ints_->get_three_integral_pointer()[0][nb * ncmo + fb]),nmo_ * nmo_);
                     vmnefbetaE = C_DDOT(nthree,
                            &(ints_->get_three_integral_pointer()[0][mb * ncmo + fb]),nmo_ * nmo_,
                            &(ints_->get_three_integral_pointer()[0][nb * ncmo + eb]),nmo_ * nmo_);
                    vmnefmixedC = C_DDOT(nthree,
                            &(ints_->get_three_integral_pointer()[0][m * ncmo + eb]),nmo_ * nmo_,
                            &(ints_->get_three_integral_pointer()[0][n * ncmo + fb]),nmo_ * nmo_);

                    vmnefalpha = vmnefalphaC - vmnefalphaE;
                    vmnefbeta = vmnefbetaC - vmnefbetaE;
                    vmnefmixed = vmnefmixedC;

                    t2alpha = vmnefalpha * renormalized_denominator(Fa[m] + Fa[n] - Fa[e] - Fa[f ]);
                    t2beta  = vmnefbeta  * renormalized_denominator(Fb[m] + Fb[n] - Fb[e] - Fb[f ]);
                    t2mixed = vmnefmixed * renormalized_denominator(Fa[m] + Fb[n] - Fa[e] - Fb[f ]);

                    vmnefalphaR  = vmnefalpha;
                    vmnefbetaR   = vmnefbeta;
                    vmnefmixedR  = vmnefmixed;
                    vmnefalphaR += vmnefalpha * renormalized_exp(Fa[m] + Fa[n] - Fa[e] - Fa[f]);
                    vmnefbetaR  += vmnefbeta * renormalized_exp(Fb[m] + Fb[n]  - Fb[e] - Fb[f]);
                    vmnefmixedR += vmnefmixed * renormalized_exp(Fa[m] + Fb[n] - Fa[e] - Fb[f]);

                    Eflyalpha+=0.25 * vmnefalphaR * t2alpha;
                    Eflybeta+=0.25 * vmnefbetaR * t2beta;
                    Eflymixed+=vmnefmixedR * t2mixed;
                }
            }
        }
    }
    Efly = Eflyalpha + Eflybeta + Eflymixed;

    return Efly;
}
double THREE_DSRG_MRPT2::E_VT2_2_ambit()
{
    double Efly = 0.0;
    size_t nthree= ints_->nthree();
    // Compute <[V, T2]> (C_2)^4 ccvv term; (me|nf) = B(L|me) * B(L|nf)
    // For a given m and n, form Bm(L|e) and Bn(L|f)
    // Bef(ef) = Bm(L|e) * Bn(L|f)
    ambit::Tensor Ba = ambit::Tensor::build(tensor_type_,"Ba",{core_,nthree,virtual_});
    ambit::Tensor Bb = ambit::Tensor::build(tensor_type_,"Bb",{core_,nthree,virtual_});
    Ba("mge") = (ThreeIntegral.block("dcv"))("gme");
    Bb("MgE") = (ThreeIntegral.block("dCV"))("gME");

    size_t dim = nthree * virtual_;
    double Emp2 = 0.0;

    double Ealpha = 0.0;
    double Ebeta  = 0.0;
    double Emixed = 0.0;
    //ambit::Tensor Bma = ambit::Tensor::build(tensor_type_,"Bma",{nthree,virtual_});
    //ambit::Tensor Bna = ambit::Tensor::build(tensor_type_,"Bna",{nthree,virtual_});
    //ambit::Tensor Bmb = ambit::Tensor::build(tensor_type_,"Bmb",{nthree,virtual_});
    //ambit::Tensor Bnb = ambit::Tensor::build(tensor_type_,"Bnb",{nthree,virtual_});
    //ambit::Tensor Bef = ambit::Tensor::build(tensor_type_,"Bef",{virtual_,virtual_});
    //ambit::Tensor BefJK = ambit::Tensor::build(tensor_type_,"BefJK",{virtual_,virtual_});
    //ambit::Tensor RD = ambit::Tensor::build(tensor_type_,"RD",{virtual_,virtual_});
    int nthread = 1;
    #ifdef _OPENMP
        nthread = omp_get_max_threads();
    #endif
    std::vector<ambit::Tensor> BmaVec;
    std::vector<ambit::Tensor> BnaVec;
    std::vector<ambit::Tensor> BmbVec;
    std::vector<ambit::Tensor> BnbVec;
    std::vector<ambit::Tensor> BefVec;
    std::vector<ambit::Tensor> BefJKVec;
    std::vector<ambit::Tensor> RDVec;
    for (int i = 0; i < nthread; i++)
    {
     BmaVec.push_back(ambit::Tensor::build(tensor_type_,"Bma",{nthree,virtual_}));
     BnaVec.push_back(ambit::Tensor::build(tensor_type_,"Bna",{nthree,virtual_}));
     BmbVec.push_back(ambit::Tensor::build(tensor_type_,"Bmb",{nthree,virtual_}));
     BnbVec.push_back(ambit::Tensor::build(tensor_type_,"Bnb",{nthree,virtual_}));
     BefVec.push_back(ambit::Tensor::build(tensor_type_,"Bef",{virtual_,virtual_}));
     BefJKVec.push_back(ambit::Tensor::build(tensor_type_,"BefJK",{virtual_,virtual_}));
     RDVec.push_back(ambit::Tensor::build(tensor_type_,"RD",{virtual_,virtual_}));
    }
    #pragma omp parallel for num_threads(num_threads_) \
    schedule(dynamic) \
    reduction(+:Ealpha, Ebeta, Emixed) \
    shared(Ba,Bb)

    for(size_t m = 0; m < core_; ++m){
        int thread = 0;
        #ifdef _OPENMP
            thread = omp_get_thread_num();
        #endif
        size_t ma = acore_mos[m];
        size_t mb = bcore_mos[m];
        #pragma omp critical
        {
        std::copy(&Ba.data()[m * dim], &Ba.data()[m * dim + dim], BmaVec[thread].data().begin());
        std::copy(&Bb.data()[m * dim], &Bb.data()[m * dim + dim], BmbVec[thread].data().begin());
        }
        for(size_t n = 0; n < core_; ++n){
            size_t na = acore_mos[n];
            size_t nb = bcore_mos[n];
            #pragma omp critical
            {
            std::copy(&Ba.data()[n * dim], &Ba.data()[n * dim + dim], BnaVec[thread].data().begin());
            std::copy(&Bb.data()[n * dim], &Bb.data()[n * dim + dim], BnbVec[thread].data().begin());
            }

            // alpha-aplha
            BefVec[thread]("ef") = BmaVec[thread]("ge") * BnaVec[thread]("gf");
            BefJKVec[thread]("ef")  = BefVec[thread]("ef") * BefVec[thread]("ef");
            BefJKVec[thread]("ef") -= BefVec[thread]("ef") * BefVec[thread]("fe");
            RDVec[thread].iterate([&](const std::vector<size_t>& i,double& value){
                double D = Fa[ma] + Fa[na] - Fa[avirt_mos[i[0]]] - Fa[avirt_mos[i[1]]];
                value = renormalized_denominator(D) * (1.0 + renormalized_exp(D));});
            Ealpha += 0.5 * BefJKVec[thread]("ef") * RDVec[thread]("ef");

            // beta-beta
            BefVec[thread]("EF") = BmbVec[thread]("gE") * BnbVec[thread]("gF");
            BefJKVec[thread]("EF")  = BefVec[thread]("EF") * BefVec[thread]("EF");
            BefJKVec[thread]("EF") -= BefVec[thread]("EF") * BefVec[thread]("FE");
            RDVec[thread].iterate([&](const std::vector<size_t>& i,double& value){
                double D = Fb[mb] + Fb[nb] - Fb[bvirt_mos[i[0]]] - Fb[bvirt_mos[i[1]]];
                value = renormalized_denominator(D) * (1.0 + renormalized_exp(D));});
            Ebeta += 0.5 * BefJKVec[thread]("EF") * RDVec[thread]("EF");

            // alpha-beta
            BefVec[thread]("eF") = BmaVec[thread]("ge") * BnbVec[thread]("gF");
            BefJKVec[thread]("eF")  = BefVec[thread]("eF") * BefVec[thread]("eF");
            RDVec[thread].iterate([&](const std::vector<size_t>& i,double& value){
                double D = Fa[ma] + Fb[nb] - Fa[avirt_mos[i[0]]] - Fb[bvirt_mos[i[1]]];
                value = renormalized_denominator(D) * (1.0 + renormalized_exp(D));});
            Emixed += BefJKVec[thread]("eF") * RDVec[thread]("eF");
        }
    }

    return (Ealpha + Ebeta + Emixed);
}
double THREE_DSRG_MRPT2::E_VT2_2_core()
{
    double E2_core = 0.0;
    BlockedTensor T2ccvv = BTF->build(tensor_type_,"T2ccvv", spin_cases({"ccvv"}));
    BlockedTensor RD2_ccvv = BTF->build(tensor_type_, "RDelta2ccvv", spin_cases({"ccvv"}));
    BlockedTensor v    = BTF->build(tensor_type_, "Vccvv", spin_cases({"ccvv"}));
    BlockedTensor RExp2ccvv = BTF->build(tensor_type_, "RExp2ccvv", spin_cases({"ccvv"}));

    v("mnef") = ThreeIntegral("gme") * ThreeIntegral("gnf");
    v("mnef") -= ThreeIntegral("gmf") * ThreeIntegral("gne");
    v("MNEF") = ThreeIntegral("gME") * ThreeIntegral("gNF");
    v("MNEF") -= ThreeIntegral("gMF") * ThreeIntegral("gNE");
    v("mNeF") = ThreeIntegral("gme") * ThreeIntegral("gNF");

    RD2_ccvv.iterate([&](const std::vector<size_t>& i,const std::vector<SpinType>& spin,double& value){
        if ((spin[0] == AlphaSpin) and (spin[1] == AlphaSpin)){
            value = renormalized_denominator(Fa[i[0]] + Fa[i[1]] - Fa[i[2]] - Fa[i[3]]);
        }else if ((spin[0] == AlphaSpin) and (spin[1] == BetaSpin) ){
            value = renormalized_denominator(Fa[i[0]] + Fb[i[1]] - Fa[i[2]] - Fb[i[3]]);
        }else if ((spin[0] == BetaSpin)  and (spin[1] == BetaSpin) ){
            value = renormalized_denominator(Fb[i[0]] + Fb[i[1]] - Fb[i[2]] - Fb[i[3]]);
        }
    });
    RExp2ccvv.iterate([&](const std::vector<size_t>& i,const std::vector<SpinType>& spin,double& value){
        if ((spin[0] == AlphaSpin) and (spin[1] == AlphaSpin)){
            value = renormalized_exp(Fa[i[0]] + Fa[i[1]] - Fa[i[2]] - Fa[i[3]]);
        }else if ((spin[0] == AlphaSpin) and (spin[1] == BetaSpin) ){
            value = renormalized_exp(Fa[i[0]] + Fb[i[1]] - Fa[i[2]] - Fb[i[3]]);
        }else if ((spin[0] == BetaSpin)  and (spin[1] == BetaSpin) ){
            value = renormalized_exp(Fb[i[0]] + Fb[i[1]] - Fb[i[2]] - Fb[i[3]]);
        }
    });
    BlockedTensor Rv = BTF->build(tensor_type_, "ReV", spin_cases({"ccvv"}));
    Rv("mnef") = v("mnef");
    Rv("mNeF") = v("mNeF");
    Rv("MNEF") = v("MNEF");
    Rv("mnef") += v("mnef")*RExp2ccvv("mnef");
    Rv("MNEF") += v("MNEF")*RExp2ccvv("MNEF");
    Rv("mNeF") += v("mNeF")*RExp2ccvv("mNeF");


    T2ccvv["MNEF"] = v["MNEF"] * RD2_ccvv["MNEF"];
    T2ccvv["mnef"] = v["mnef"] * RD2_ccvv["mnef"];
    T2ccvv["mNeF"] = v["mNeF"] * RD2_ccvv["mNeF"];
    E2_core += 0.25 * T2ccvv["mnef"] * Rv["mnef"];
    E2_core += 0.25 * T2ccvv["MNEF"] * Rv["MNEF"];
    E2_core += T2ccvv["mNeF"] * Rv["mNeF"];

    return E2_core;
}

}} // End Namespaces
