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

namespace psi{ namespace forte{

#ifdef _OPENMP
	#include <omp.h>
	bool THREE_DSRG_MRPT2::have_omp_ = true;
#else
   #define omp_get_max_threads() 1
   #define omp_get_thread_num() 0
   bool THREE_DSRG_MRPT2::have_omp_ = false;
#endif


THREE_DSRG_MRPT2::THREE_DSRG_MRPT2(Reference reference, boost::shared_ptr<Wavefunction> wfn, Options &options, std::shared_ptr<ForteIntegrals>  ints,
std::shared_ptr<MOSpaceInfo> mo_space_info)
    : Wavefunction(options,_default_psio_lib_),
      reference_(reference),
      ints_(ints),
      tensor_type_(kCore),
      BTF_(new BlockedTensorFactory(options)),
      mo_space_info_(mo_space_info)
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
        BTF_->print_memory_info();
    }
    ref_type_ = options_.get_str("REFERENCE");
    outfile->Printf("\n Reference = %s", ref_type_.c_str());

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
    Eref_ = reference_.get_Eref();
    outfile->Printf("\n  Reference Energy = %.15f", Eref_);

    frozen_core_energy_ = ints_->frozen_core_energy();

    ncmopi_ = mo_space_info_->get_dimension("CORRELATED");
    size_t ncmo_ = mo_space_info_->size("CORRELATED");

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

    rdoccpi_ = mo_space_info_->get_dimension("RESTRICTED_DOCC");
    actvpi_  = mo_space_info_->get_dimension ("ACTIVE");
    ruoccpi_ = mo_space_info_->get_dimension ("RESTRICTED_UOCC");

    
    acore_mos_ = mo_space_info_->get_corr_abs_mo("RESTRICTED_DOCC");
    bcore_mos_ = mo_space_info_->get_corr_abs_mo("RESTRICTED_DOCC");
    aactv_mos_ = mo_space_info_->get_corr_abs_mo("ACTIVE");
    bactv_mos_ = mo_space_info_->get_corr_abs_mo("ACTIVE");
    avirt_mos_ = mo_space_info_->get_corr_abs_mo("RESTRICTED_UOCC");
    bvirt_mos_ = mo_space_info_->get_corr_abs_mo("RESTRICTED_UOCC");

    BlockedTensor::set_expert_mode(true);

    //boost::shared_ptr<BlockedTensorFa_ctory> BTFBlockedTensorFa_ctory(options);

    //BlockedTensor::add_mo_space("c","m,n,µ,π",acore_mos,AlphaSpin);
    //BlockedTensor::add_mo_space("C","M,N,Ω,∏",bcore_mos,BetaSpin);
    BTF_->add_mo_space("c","m,n,µ,π",acore_mos_,AlphaSpin);
    BTF_->add_mo_space("C","M,N,Ω,∏",bcore_mos_,BetaSpin);

    core_ = acore_mos_.size();

    BTF_->add_mo_space("a","uvwxyz",aactv_mos_,AlphaSpin);
    BTF_->add_mo_space("A","UVWXYZ",bactv_mos_,BetaSpin);
    active_ = aactv_mos_.size();

    BTF_->add_mo_space("v","e,f,ε,φ",avirt_mos_,AlphaSpin);
    BTF_->add_mo_space("V","E,F,Ƒ,Ǝ",bvirt_mos_,BetaSpin);
    virtual_ = avirt_mos_.size();

    BTF_->add_composite_mo_space("h","ijkl",{"c","a"});
    BTF_->add_composite_mo_space("H","IJKL",{"C","A"});

    BTF_->add_composite_mo_space("p","abcd",{"a","v"});
    BTF_->add_composite_mo_space("P","ABCD",{"A","V"});

    BTF_->add_composite_mo_space("g","pqrs",{"c","a","v"});
    BTF_->add_composite_mo_space("G","PQRS",{"C","A","V"});

    // These two blocks of functions create a Blocked tensor
    std::vector<std::string> hhpp_no_cv = BTF_->generate_indices("cav", "hhpp");
    no_hhpp_ = hhpp_no_cv;

    nthree_ = ints_->nthree();
    std::vector<size_t> nauxpi(nthree_);
    std::iota(nauxpi.begin(), nauxpi.end(),0);

    std::vector<std::string> list_of_pphh_V = BTF_->generate_indices("vac", "pphh");
    //BlockedTensor::add_mo_space("@","$",nauxpi,NoSpin);
    //BlockedTensor::add_mo_space("d","g",nauxpi,NoSpin);
    BTF_->add_mo_space("d","g",nauxpi,NoSpin);

    H_ = BTF_->build(tensor_type_,"H",spin_cases({"gg"}));

    Gamma1_ = BTF_->build(tensor_type_,"Gamma1_",spin_cases({"hh"}));
    Eta1_ = BTF_->build(tensor_type_,"Eta1_",spin_cases({"pp"}));
    Lambda2_ = BTF_->build(tensor_type_,"Lambda2_",spin_cases({"aaaa"}));
    Lambda3_ = BTF_->build(tensor_type_,"Lambda3_",spin_cases({"aaaaaa"}));
    F_ = BTF_->build(tensor_type_,"Fock",spin_cases({"gg"}));
    Delta1_ = BTF_->build(tensor_type_,"Delta1_",spin_cases({"aa"}));

    RDelta1_ = BTF_->build(tensor_type_,"RDelta1_",spin_cases({"hp"}));

    T1_ = BTF_->build(tensor_type_,"T1 Amplitudes",spin_cases({"hp"}));

    RExp1_ = BTF_->build(tensor_type_,"RExp1",spin_cases({"hp"}));

    H_.iterate([&](const std::vector<size_t>& i,const std::vector<SpinType>& spin,double& value){
        if (spin[0] == AlphaSpin)
            value = ints_->oei_a(i[0],i[1]);
        else
            value = ints_->oei_b(i[0],i[1]);
    });

    ambit::Tensor Gamma1_cc = Gamma1_.block("cc");
    ambit::Tensor Gamma1_aa = Gamma1_.block("aa");
    ambit::Tensor Gamma1_CC = Gamma1_.block("CC");
    ambit::Tensor Gamma1_AA = Gamma1_.block("AA");

    ambit::Tensor Eta1_aa = Eta1_.block("aa");
    ambit::Tensor Eta1_vv = Eta1_.block("vv");
    ambit::Tensor Eta1_AA = Eta1_.block("AA");
    ambit::Tensor Eta1_VV = Eta1_.block("VV");

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


    //Compute the fock matrix from the reference.  Make sure fock matrix is updated in integrals class.  
    boost::shared_ptr<Matrix> Gamma1_matrixA(new Matrix("Gamma1_RDM", ncmo_, ncmo_));
    boost::shared_ptr<Matrix> Gamma1_matrixB(new Matrix("Gamma1_RDM", ncmo_, ncmo_));
    for(size_t m = 0; m < core_; m++){
            Gamma1_matrixA->set(acore_mos_[m],acore_mos_[m],1.0);
            Gamma1_matrixB->set(bcore_mos_[m],bcore_mos_[m],1.0);
    }
    Gamma1_aa.iterate([&](const std::vector<size_t>& i,double& value){
     Gamma1_matrixA->set(aactv_mos_[i[0]], aactv_mos_[i[1]], value);   });

    Gamma1_aa.iterate([&](const std::vector<size_t>& i,double& value){
     Gamma1_matrixB->set(bactv_mos_[i[0]], bactv_mos_[i[1]], value);   
     });

    ints_->make_fock_matrix(Gamma1_matrixA, Gamma1_matrixB);

    if(ref_type_ == "RHF" || ref_type_ == "ROHF" || ref_type_ == "TWOCON" || ref_type_ == "RKS")
    {
    F_.iterate([&](const std::vector<size_t>& i,const std::vector<SpinType>& spin,double& value)
    {
        if (spin[0] == AlphaSpin){
            value = ints_->get_fock_a(i[0],i[1]);
        }else if (spin[0]  == BetaSpin){
            value = ints_->get_fock_b(i[0],i[1]);
        }
    });

    }
    else
    {
    F_.iterate([&](const std::vector<size_t>& i,const std::vector<SpinType>& spin,double& value)
    {
        if (spin[0] == AlphaSpin){
            value = ints_->get_fock_a(i[0],i[1]);
        }else if (spin[0]  == BetaSpin){
            value = ints_->get_fock_b(i[0],i[1]);
        }
    });
    }

    Dimension ncmopi_ = mo_space_info_->get_dimension("CORRELATED");

    Fa_.reserve(ncmo_);
    Fb_.reserve(ncmo_);

    for(size_t p = 0; p < ncmo_; p++)
    {
        Fa_[p] = ints_->get_fock_a(p,p);
        Fb_[p] = ints_->get_fock_b(p,p);
    }

    Delta1_.iterate([&](const std::vector<size_t>& i,const std::vector<SpinType>& spin,double& value){
        if (spin[0] == AlphaSpin){
            value = Fa_[i[0]] - Fa_[i[1]];
        }else if (spin[0]  == BetaSpin){
            value = Fb_[i[0]] - Fb_[i[1]];
        }
    });

    RDelta1_.iterate([&](const std::vector<size_t>& i,const std::vector<SpinType>& spin,double& value){
        if (spin[0]  == AlphaSpin){
            value = renormalized_denominator(Fa_[i[0]] - Fa_[i[1]]);
        }else if (spin[0]  == BetaSpin){
            value = renormalized_denominator(Fb_[i[0]] - Fb_[i[1]]);
        }
    });

    // Fill out Lambda2_ and Lambda3_
    ambit::Tensor Lambda2_aa = Lambda2_.block("aaaa");
    ambit::Tensor Lambda2_aA = Lambda2_.block("aAaA");
    ambit::Tensor Lambda2_AA = Lambda2_.block("AAAA");
    Lambda2_aa("pqrs") = reference_.L2aa()("pqrs");
    Lambda2_aA("pqrs") = reference_.L2ab()("pqrs");
    Lambda2_AA("pqrs") = reference_.L2bb()("pqrs");

    ambit::Tensor Lambda3_aaa = Lambda3_.block("aaaaaa");
    ambit::Tensor Lambda3_aaA = Lambda3_.block("aaAaaA");
    ambit::Tensor Lambda3_aAA = Lambda3_.block("aAAaAA");
    ambit::Tensor Lambda3_AAA = Lambda3_.block("AAAAAA");
    Lambda3_aaa("pqrstu") = reference_.L3aaa()("pqrstu");
    Lambda3_aaA("pqrstu") = reference_.L3aab()("pqrstu");
    Lambda3_aAA("pqrstu") = reference_.L3abb()("pqrstu");
    Lambda3_AAA("pqrstu") = reference_.L3bbb()("pqrstu");

    // Prepare exponential tensors for effective Fock matrix and integrals

    RExp1_.iterate([&](const std::vector<size_t>& i,const std::vector<SpinType>& spin,double& value){
        if (spin[0]  == AlphaSpin){
            value = renormalized_exp(Fa_[i[0]] - Fa_[i[1]]);
        }else if (spin[0]  == BetaSpin){
            value = renormalized_exp(Fb_[i[0]] - Fb_[i[1]]);
        }
    });


    print_ = options_.get_int("PRINT");

    if(print_ > 1){
        Gamma1_.print(stdout);
        Eta1_.print(stdout);
        F_.print(stdout);
        H_.print(stdout);
    }
    if(print_ > 2){
        Lambda2_.print(stdout);
    }
    if(print_ > 3){
        Lambda3_.print(stdout);
    }

    integral_type_ = ints_->integral_type();

    if(integral_type_!=DiskDF)
    {
        V_ = BTF_->build(tensor_type_,"V_", BTF_->spin_cases_avoid(list_of_pphh_V, 1));
        T2_ = BTF_->build(tensor_type_, "T2 Amplitudes", BTF_->spin_cases_avoid(no_hhpp_,1));
        ThreeIntegral_ = BTF_->build(tensor_type_,"ThreeInt",{"dph", "dPH"});

        std::vector<std::string> ThreeInt_block = ThreeIntegral_.block_labels();

        std::map<std::string, std::vector<size_t> > mo_to_index = BTF_->get_mo_to_index();

        for(std::string& string_block : ThreeInt_block)
        {
                std::string pos1(1, string_block[0]);
                std::string pos2(1, string_block[1]);
                std::string pos3(1, string_block[2]);

                std::vector<size_t> first_index = mo_to_index[pos1];
                std::vector<size_t> second_index = mo_to_index[pos2];
                std::vector<size_t> third_index = mo_to_index[pos3];

                ambit::Tensor ThreeIntegral_block = ints_->three_integral_block(first_index, second_index, third_index);
                ThreeIntegral_.block(string_block).copy(ThreeIntegral_block);
        }
        V_["abij"] =  ThreeIntegral_["gai"]*ThreeIntegral_["gbj"];
        V_["abij"] -= ThreeIntegral_["gaj"]*ThreeIntegral_["gbi"];

        V_["aBiJ"] =  ThreeIntegral_["gai"]*ThreeIntegral_["gBJ"];

        V_["ABIJ"] =  ThreeIntegral_["gAI"]*ThreeIntegral_["gBJ"];
        V_["ABIJ"] -= ThreeIntegral_["gAJ"]*ThreeIntegral_["gBI"];

    }
    // If the integral_type is DiskDF, we will compute these integral stuff later in each funtion

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
        {"ccvv_algorithm",options_.get_str("ccvv_algorithm")},
        {"ccvv_source", options_.get_str("CCVV_SOURCE")}};

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
    Timer ComputeEnergy;
    // Compute reference
        //double Eref;
        //Eref = compute_ref();

        // Compute T2 and T1
        if(integral_type_!=DiskDF){compute_t2();}
        compute_t1();
        check_t1();
        if(integral_type_!=DiskDF){renormalize_V();}

        // Compute effective integrals
        renormalize_F();
        if(print_ > 1)  F_.print(stdout); // The actv-actv block is different but OK.
        if(print_ > 2){
            T1_.print(stdout);
        }

        // Compute DSRG-MRPT2 correlation energy
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

        // Analyze T1 and T2
        check_t1();
        energy.push_back({"max(T1)", T1max_});
        energy.push_back({"||T1||", T1norm_});

        // Print energy summary
        outfile->Printf("\n\n  ==> DSRG-MRPT2 Energy Summary <==\n");
        for (auto& str_dim : energy){
            outfile->Printf("\n    %-30s = %22.15f",str_dim.first.c_str(),str_dim.second);
        }

        Process::environment.globals["CURRENT ENERGY"] = Etotal;


        outfile->Printf("\n\n\n    CD/DF-DSRG-MRPT2 took   %8.8f s.", ComputeEnergy.get());
        return Etotal;
}

double THREE_DSRG_MRPT2::compute_ref()
{
    double E = 0.0;

    E  = 0.5 * H_["ij"] * Gamma1_["ij"];
    E += 0.5 * F_["ij"] * Gamma1_["ij"];
    E += 0.5 * H_["IJ"] * Gamma1_["IJ"];
    E += 0.5 * F_["IJ"] * Gamma1_["IJ"];

    if(integral_type_==DiskDF)
    {
        V_ = compute_V_minimal({"aaaa", "AAAA", "aAaA"}, false);
    }
    E += 0.25 * V_["uvxy"] * Lambda2_["uvxy"];
    E += 0.25 * V_["UVXY"] * Lambda2_["UVXY"];
    E += V_["uVxY"] * Lambda2_["uVxY"];

    boost::shared_ptr<Molecule> molecule = Process::environment.molecule();
    double Enuc = molecule->nuclear_repulsion_energy();

    return E + frozen_core_energy_+ Enuc;
}
void THREE_DSRG_MRPT2::compute_t2()
{
    std::string str = "Computing T2";
    outfile->Printf("\n    %-36s ...", str.c_str());
    Timer timer;

    T2_["ijab"] = V_["abij"];
    T2_["iJaB"] = V_["aBiJ"];
    T2_["IJAB"] = V_["ABIJ"];
    T2_.iterate([&](const std::vector<size_t>& i,const std::vector<SpinType>& spin,double& value){
        if (spin[0] == AlphaSpin && spin[1] == AlphaSpin)
        {
            value *= renormalized_denominator(Fa_[i[0]] + Fa_[i[1]] - Fa_[i[2]] - Fa_[i[3]]);
        }
        else if(spin[0]==BetaSpin && spin[1] == BetaSpin)
        {
            value *= renormalized_denominator(Fb_[i[0]] + Fb_[i[1]] - Fb_[i[2]] - Fb_[i[3]]);
        }
        else
        {
            value *= renormalized_denominator(Fa_[i[0]] + Fb_[i[1]] - Fa_[i[2]] - Fb_[i[3]]);
        }
    });

    // zero internal amplitudes
    T2_.block("aaaa").zero();
    T2_.block("aAaA").zero();
    T2_.block("AAAA").zero();


    outfile->Printf("...Done. Timing %15.6f s", timer.get());

}

ambit::BlockedTensor THREE_DSRG_MRPT2::compute_T2_minimal(const std::vector<std::string>& t2_spaces)
{
    ambit::BlockedTensor T2min;

    T2min = BTF_->build(tensor_type_, "T2min", t2_spaces, true);
    ambit::BlockedTensor ThreeInt = compute_B_minimal(t2_spaces);
    T2min["ijab"] =  (ThreeInt["gia"] * ThreeInt["gjb"]);
    T2min["ijab"] -= (ThreeInt["gib"] * ThreeInt["gja"]);
    T2min["IJAB"] =  (ThreeInt["gIA"] * ThreeInt["gJB"]);
    T2min["IJAB"] -= (ThreeInt["gIB"] * ThreeInt["gJA"]);
    T2min["iJaB"] =  (ThreeInt["gia"] * ThreeInt["gJB"]);

    T2min.iterate([&](const std::vector<size_t>& i,const std::vector<SpinType>& spin,double& value){
        if (spin[0] == AlphaSpin && spin[1] == AlphaSpin)
        {
            value *= renormalized_denominator(Fa_[i[0]] + Fa_[i[1]] - Fa_[i[2]] - Fa_[i[3]]);
        }
        else if(spin[0]==BetaSpin && spin[1] == BetaSpin)
        {
            value *= renormalized_denominator(Fb_[i[0]] + Fb_[i[1]] - Fb_[i[2]] - Fb_[i[3]]);
        }
        else
        {
            value *= renormalized_denominator(Fa_[i[0]] + Fb_[i[1]] - Fa_[i[2]] - Fb_[i[3]]);
        }
    });


    // zero internal amplitudes
    if(std::find(t2_spaces.begin(), t2_spaces.end(), "aaaa")!=t2_spaces.end())
        T2min.block("aaaa").zero();
    if(std::find(t2_spaces.begin(), t2_spaces.end(), "aAaA")!=t2_spaces.end())
        T2min.block("aAaA").zero();
    if(std::find(t2_spaces.begin(), t2_spaces.end(), "AAAA")!=t2_spaces.end())
        T2min.block("AAAA").zero();
    return T2min;


}
ambit::BlockedTensor THREE_DSRG_MRPT2::compute_V_minimal(const std::vector<std::string>& spaces, bool renormalize)
{
    

    ambit::BlockedTensor Vmin = BTF_->build(tensor_type_,"Vmin",spaces, true);
    ambit::BlockedTensor ThreeInt;
    ThreeInt = compute_B_minimal(spaces);
    Vmin["abij"] =   ThreeInt["gai"]*ThreeInt["gbj"];
    Vmin["abij"] -=  ThreeInt["gaj"]*ThreeInt["gbi"];
    Vmin["ABIJ"] =   ThreeInt["gAI"]*ThreeInt["gBJ"];
    Vmin["ABIJ"] -=  ThreeInt["gAJ"]*ThreeInt["gBI"];
    Vmin["aBiJ"] =   ThreeInt["gai"]*ThreeInt["gBJ"];

    if(renormalize)
    {
        Vmin.iterate([&](const std::vector<size_t>& i,const std::vector<SpinType>& spin,double& value){
            if ((spin[0] == AlphaSpin) and (spin[1] == AlphaSpin)){
                value = (value + value * renormalized_exp(Fa_[i[0]] + Fa_[i[1]] - Fa_[i[2]] - Fa_[i[3]]));
            }else if ((spin[0] == AlphaSpin) and (spin[1] == BetaSpin) ){
                value = (value + value * renormalized_exp(Fa_[i[0]] + Fb_[i[1]] - Fa_[i[2]] - Fb_[i[3]]));
            }else if ((spin[0] == BetaSpin)  and (spin[1] == BetaSpin) ){
                value = (value + value * renormalized_exp(Fb_[i[0]] + Fb_[i[1]] - Fb_[i[2]] - Fb_[i[3]]));
            }
        });
    }
    return Vmin;
}

ambit::BlockedTensor THREE_DSRG_MRPT2::compute_B_minimal(const std::vector<std::string>& spaces)
{
    std::vector<size_t> nauxpi(nthree_);
    std::iota(nauxpi.begin(), nauxpi.end(),0);

    //BlockedTensor::add_mo_space("@","$",nauxpi,NoSpin);
    //BlockedTensor::add_mo_space("d","g",nauxpi,NoSpin);
    std::vector<std::string> ThreeIntegral_labels;
    for(const auto& label : spaces)
    {   
        std::string left_threeint;
        std::string right_threeint;
        left_threeint+="d";
        right_threeint+="d";
       
        //Since aAaA-> (aa)(AA) -> ThreeInt
        if(std::islower(label[0]) && std::isupper(label[1]) && std::islower(label[2]) && std::isupper(label[3]))
        {
            left_threeint+=label[0];
            left_threeint+=label[2];

            if(std::find(ThreeIntegral_labels.begin(), ThreeIntegral_labels.end(), left_threeint)==ThreeIntegral_labels.end())
            {
                ThreeIntegral_labels.push_back(left_threeint);
            }

            right_threeint+=label[1];
            right_threeint+=label[3];
            if(std::find(ThreeIntegral_labels.begin(), ThreeIntegral_labels.end(), right_threeint)==ThreeIntegral_labels.end())
            {
                ThreeIntegral_labels.push_back(right_threeint);
            }
            
        }
        //Since acac -> (aa)(cc) - (ac)(ac)
        else if(std::islower(label[0]) && std::islower(label[1]) && std::islower(label[2]) && std::islower(label[3]))
        {
            //Declare a string for the Kexchange part
            std::string left_threeintK;
            std::string right_threeintK;

            //Next section of code is standard J-like term
            left_threeint+=label[0];
            left_threeint+=label[2];

            if(std::find(ThreeIntegral_labels.begin(), ThreeIntegral_labels.end(), left_threeint)==ThreeIntegral_labels.end())
            {
                ThreeIntegral_labels.push_back(left_threeint);
            }
            right_threeint+=label[1];
            right_threeint+=label[3];
            if(std::find(ThreeIntegral_labels.begin(), ThreeIntegral_labels.end(), right_threeint)==ThreeIntegral_labels.end())
            {
                ThreeIntegral_labels.push_back(right_threeint);
            }

            //Add the exchange part of ThreeInt
            left_threeintK+="d";
            left_threeintK+=label[0];
            left_threeintK+=label[3];;
            if(std::find(ThreeIntegral_labels.begin(), ThreeIntegral_labels.end(), left_threeintK)==ThreeIntegral_labels.end())
            {
                ThreeIntegral_labels.push_back(left_threeintK);
            }
            right_threeintK+="d";
            right_threeintK+=label[1];
            right_threeintK+=label[2];;
            if(std::find(ThreeIntegral_labels.begin(), ThreeIntegral_labels.end(), right_threeintK)==ThreeIntegral_labels.end())
            {
                ThreeIntegral_labels.push_back(right_threeintK);
            }
            
            
        }
        else if(std::isupper(label[0]) && std::isupper(label[1]) && std::isupper(label[2]) && std::isupper(label[3]))
        {
            //Declare a string for the Kexchange part
            std::string left_threeintK;
            std::string right_threeintK;

            //Next section of code is standard J-like term
            left_threeint+=label[0];
            left_threeint+=label[2];

            if(std::find(ThreeIntegral_labels.begin(), ThreeIntegral_labels.end(), left_threeint)==ThreeIntegral_labels.end())
            {
                ThreeIntegral_labels.push_back(left_threeint);
            }
            right_threeint+=label[1];
            right_threeint+=label[3];
            if(std::find(ThreeIntegral_labels.begin(), ThreeIntegral_labels.end(), right_threeint)==ThreeIntegral_labels.end())
            {
                ThreeIntegral_labels.push_back(right_threeint);
            }

            //Add the exchange part of ThreeInt
            left_threeintK+="d";
            left_threeintK+=label[0];
            left_threeintK+=label[3];;
            if(std::find(ThreeIntegral_labels.begin(), ThreeIntegral_labels.end(), left_threeintK)==ThreeIntegral_labels.end())
            {
                ThreeIntegral_labels.push_back(left_threeintK);
            }
            right_threeintK+="d";
            right_threeintK+=label[1];
            right_threeintK+=label[2];;
            if(std::find(ThreeIntegral_labels.begin(), ThreeIntegral_labels.end(), right_threeintK)==ThreeIntegral_labels.end())
            {
                ThreeIntegral_labels.push_back(right_threeintK);
            }

        }

    
    }

    ambit::BlockedTensor ThreeInt = BTF_->build(tensor_type_, "ThreeIntMin", ThreeIntegral_labels, true);


    std::vector<std::string> ThreeInt_block = ThreeInt.block_labels();

    std::map<std::string, std::vector<size_t> > mo_to_index = BTF_->get_mo_to_index();

    for(std::string& string_block : ThreeInt_block)
    {
        std::string pos1(1, string_block[0]);
        std::string pos2(1, string_block[1]);
        std::string pos3(1, string_block[2]);

        std::vector<size_t> first_index = mo_to_index[pos1];
        std::vector<size_t> second_index = mo_to_index[pos2];
        std::vector<size_t> third_index = mo_to_index[pos3];

        ambit::Tensor ThreeIntegral_block = ints_->three_integral_block(first_index, second_index, third_index);
        ThreeInt.block(string_block).copy(ThreeIntegral_block);
    }

    return ThreeInt;
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
    temp = BTF_->build(tensor_type_,"temp",spin_cases({"aa"}), true);
    temp["xu"] = Gamma1_["xu"] * Delta1_["xu"];
    temp["XU"] = Gamma1_["XU"] * Delta1_["XU"];

    //Form the T1 amplitudes

    BlockedTensor N = BTF_->build(tensor_type_,"N",spin_cases({"hp"}));
    if(integral_type_==DiskDF)
    {
        T2_  = compute_T2_minimal({"cava", "caaa", "aaaa","aava", "cAvA", "aAvA", "cAaA", "aCaV", "aAaA", "aCaA", "aAaV", "CAVA", "CAAA",
        "AAVA", "AAAA"});
    }

    N["ia"]  = F_["ia"];
    N["ia"] += temp["xu"] * T2_["iuax"];
    N["ia"] += temp["XU"] * T2_["iUaX"];
    T1_["ia"] = N["ia"] * RDelta1_["ia"];


    N["IA"]  = F_["IA"];
    N["IA"] += temp["xu"] * T2_["uIxA"];
    N["IA"] += temp["XU"] * T2_["IUAX"];
    T1_["IA"] = N["IA"] * RDelta1_["IA"];

    T1_.block("AA").zero();
    T1_.block("aa").zero();

    outfile->Printf("...Done. Timing %15.6f s", timer.get());
}
void THREE_DSRG_MRPT2::check_t1()
{
    // norm and maximum of T1 amplitudes
    T1norm_ = T1_.norm(); T1max_ = 0.0;
    T1_.iterate([&](const std::vector<size_t>&,const std::vector<SpinType>&,double& value){
            T1max_ = T1max_ > fabs(value) ? T1max_ : fabs(value);
    });
}

void THREE_DSRG_MRPT2::renormalize_V()
{
    Timer timer;
    std::string str = "Renormalizing V";
    outfile->Printf("\n    %-36s ...", str.c_str());

    V_.iterate([&](const std::vector<size_t>& i,const std::vector<SpinType>& spin,double& value){
        if ((spin[0] == AlphaSpin) and (spin[1] == AlphaSpin)){
            value = (value + value * renormalized_exp(Fa_[i[0]] + Fa_[i[1]] - Fa_[i[2]] - Fa_[i[3]]));
        }else if ((spin[0] == AlphaSpin) and (spin[1] == BetaSpin) ){
            value = (value + value * renormalized_exp(Fa_[i[0]] + Fb_[i[1]] - Fa_[i[2]] - Fb_[i[3]]));
        }else if ((spin[0] == BetaSpin)  and (spin[1] == BetaSpin) ){
            value = (value + value * renormalized_exp(Fb_[i[0]] + Fb_[i[1]] - Fb_[i[2]] - Fb_[i[3]]));
        }
    });




    outfile->Printf("...Done. Timing %15.6f s", timer.get());
}

void THREE_DSRG_MRPT2::renormalize_F()
{
    Timer timer;

    std::string str = "Renormalizing F";
    outfile->Printf("\n    %-36s ...", str.c_str());

    BlockedTensor temp_aa = BTF_->build(tensor_type_,"temp_aa",spin_cases({"aa"}), true);
    temp_aa["xu"] = Gamma1_["xu"] * Delta1_["xu"];
    temp_aa["XU"] = Gamma1_["XU"] * Delta1_["XU"];

    BlockedTensor temp1 = BTF_->build(tensor_type_,"temp1",spin_cases({"hp"}));
    BlockedTensor temp2 = BTF_->build(tensor_type_,"temp2",spin_cases({"hp"}));
    if(integral_type_==DiskDF)
    {
        T2_  = compute_T2_minimal({"cava", "caaa", "aaaa", "cAvA","aava", "aAvA", "cAaA", "aCaV", "aAaA", "aCaA", "aAaV", "CAVA", "CAAA",
        "AAVA", "AAAA"});
    }

    temp1["ia"] += temp_aa["xu"] * T2_["iuax"];
    temp1["ia"] += temp_aa["XU"] * T2_["iUaX"];
    temp2["ia"] += F_["ia"] * RExp1_["ia"];
    temp2["ia"] += temp1["ia"] * RExp1_["ia"];

    temp1["IA"] += temp_aa["xu"] * T2_["uIxA"];
    temp1["IA"] += temp_aa["XU"] * T2_["IUAX"];
    temp2["IA"] += F_["IA"] * RExp1_["IA"];
    temp2["IA"] += temp1["IA"] * RExp1_["IA"];

    F_["ia"] += temp2["ia"];
    F_["ai"] += temp2["ia"];

    F_["IA"] += temp2["IA"];
    F_["AI"] += temp2["IA"];
    outfile->Printf("...Done. Timing %15.6f s", timer.get());
}

double THREE_DSRG_MRPT2::E_FT1()
{
    Timer timer;
    std::string str = "Computing <[F, T1]>";
    outfile->Printf("\n    %-36s ...", str.c_str());
    double E = 0.0;
    BlockedTensor temp;
    temp = BTF_->build(tensor_type_,"temp",spin_cases({"hp"}), true);

    temp["jb"] += T1_["ia"] * Eta1_["ab"] * Gamma1_["ji"];
    temp["JB"] += T1_["IA"] * Eta1_["AB"] * Gamma1_["JI"];

    E += temp["jb"] * F_["bj"];
    E += temp["JB"] * F_["BJ"];

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
    temp = BTF_->build(tensor_type_,"temp", spin_cases({"aaaa"}));
    if(integral_type_==DiskDF){
        V_ = compute_V_minimal({"vaaa", "aaca", "VAAA", "AACA", "vAaA", "aVaA", "aAcA", "aAaC"}, true);
    }
    

    temp["uvxy"] += V_["evxy"] * T1_["ue"];
    temp["uvxy"] -= V_["uvmy"] * T1_["mx"];

    temp["UVXY"] += V_["EVXY"] * T1_["UE"];
    temp["UVXY"] -= V_["UVMY"] * T1_["MX"];

    temp["uVxY"] += V_["eVxY"] * T1_["ue"];
    temp["uVxY"] += V_["uExY"] * T1_["VE"];
    temp["uVxY"] -= V_["uVmY"] * T1_["mx"];
    temp["uVxY"] -= V_["uVxM"] * T1_["MY"];

    E += 0.5 * temp["uvxy"] * Lambda2_["xyuv"];
    E += 0.5 * temp["UVXY"] * Lambda2_["XYUV"];
    E += temp["uVxY"] * Lambda2_["xYuV"];

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
    temp = BTF_->build(tensor_type_,"temp",spin_cases({"aaaa"}));
    if(integral_type_==DiskDF)
    {
        T2_ = compute_T2_minimal({"aava", "acaa", "AAVA", "ACAA", "aAvA", "aAaV", "aCaA", "cAaA"});
    }
    
    temp["uvxy"] += F_["xe"] * T2_["uvey"];
    temp["uvxy"] -= F_["mv"] * T2_["umxy"];

    temp["UVXY"] += F_["XE"] * T2_["UVEY"];
    temp["UVXY"] -= F_["MV"] * T2_["UMXY"];

    temp["uVxY"] += F_["xe"] * T2_["uVeY"];
    temp["uVxY"] += F_["YE"] * T2_["uVxE"];
    temp["uVxY"] -= F_["MV"] * T2_["uMxY"];
    temp["uVxY"] -= F_["mu"] * T2_["mVxY"];

    E += 0.5 * temp["uvxy"] * Lambda2_["xyuv"];
    E += 0.5 * temp["UVXY"] * Lambda2_["XYUV"];
    E += temp["uVxY"] * Lambda2_["xYuV"];

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
    std::vector<std::string> list_of_pphh_V = BTF_->generate_indices("vac", "pphh");
    
    if(integral_type_==DiskDF)
    {
        T2_ = compute_T2_minimal(BTF_->spin_cases_avoid(no_hhpp_,1));
        V_  = compute_V_minimal(BTF_->spin_cases_avoid(list_of_pphh_V, 1));
    }

    //TODO: Implement these without storing V and/or T2 by using blocking
    BlockedTensor temp = BTF_->build(tensor_type_,"temp",spin_cases({"aa"}));
    temp["vu"] += 0.5 * V_["efmu"] * T2_["mvef"];
    temp["vu"] += V_["fEuM"] * T2_["vMfE"];
    temp["VU"] += 0.5 * V_["EFMU"] * T2_["MVEF"];
    temp["VU"] += V_["eFmU"] * T2_["mVeF"];
    E += temp["vu"] * Gamma1_["uv"];
    E += temp["VU"] * Gamma1_["UV"];

    temp.zero();
    temp["vu"] += 0.5 * V_["vemn"] * T2_["mnue"];
    temp["vu"] += V_["vEmN"] * T2_["mNuE"];
    temp["VU"] += 0.5 * V_["VEMN"] * T2_["MNUE"];
    temp["VU"] += V_["eVnM"] * T2_["nMeU"];
    E += temp["vu"] * Eta1_["uv"];
    E += temp["VU"] * Eta1_["UV"];
    /// These terms all have two active indices -> I will assume these can be store in core.

    temp = BTF_->build(tensor_type_,"temp",spin_cases({"aaaa"}), true);
    temp["yvxu"] += V_["efxu"] * T2_["yvef"];
    temp["yVxU"] += V_["eFxU"] * T2_["yVeF"];
    temp["YVXU"] += V_["EFXU"] * T2_["YVEF"];
    E += 0.25 * temp["yvxu"] * Gamma1_["xy"] * Gamma1_["uv"];
    E += temp["yVxU"] * Gamma1_["UV"] * Gamma1_["xy"];
    E += 0.25 * temp["YVXU"] * Gamma1_["XY"] * Gamma1_["UV"];

    temp.zero();
    temp["vyux"] += V_["vymn"] * T2_["mnux"];
    temp["vYuX"] += V_["vYmN"] * T2_["mNuX"];
    temp["VYUX"] += V_["VYMN"] * T2_["MNUX"];
    E += 0.25 * temp["vyux"] * Eta1_["uv"] * Eta1_["xy"];
    E += temp["vYuX"] * Eta1_["uv"] * Eta1_["XY"];
    E += 0.25 * temp["VYUX"] * Eta1_["UV"] * Eta1_["XY"];

    temp.zero();
    temp["vyux"] += V_["vemx"] * T2_["myue"];
    temp["vyux"] += V_["vExM"] * T2_["yMuE"];
    temp["VYUX"] += V_["eVmX"] * T2_["mYeU"];
    temp["VYUX"] += V_["VEXM"] * T2_["YMUE"];
    E += temp["vyux"] * Gamma1_["xy"] * Eta1_["uv"];
    E += temp["VYUX"] * Gamma1_["XY"] * Eta1_["UV"];
    temp["yVxU"] = V_["eVxM"] * T2_["yMeU"];
    E += temp["yVxU"] * Gamma1_["xy"] * Eta1_["UV"];
    temp["vYuX"] = V_["vEmX"] * T2_["mYuE"];
    E += temp["vYuX"] * Gamma1_["XY"] * Eta1_["uv"];

    temp.zero();
    temp["yvxu"] += 0.5 * Gamma1_["wz"] * V_["vexw"] * T2_["yzue"];
    temp["yvxu"] += Gamma1_["WZ"] * V_["vExW"] * T2_["yZuE"];
    temp["yvxu"] += 0.5 * Eta1_["wz"] * T2_["myuw"] * V_["vzmx"];
    temp["yvxu"] += Eta1_["WZ"] * T2_["yMuW"] * V_["vZxM"];
    E += temp["yvxu"] * Gamma1_["xy"] * Eta1_["uv"];

    temp["YVXU"] += 0.5 * Gamma1_["WZ"] * V_["VEXW"] * T2_["YZUE"];
    temp["YVXU"] += Gamma1_["wz"] * V_["eVwX"] * T2_["zYeU"];
    temp["YVXU"] += 0.5 * Eta1_["WZ"] * T2_["MYUW"] * V_["VZMX"];
    temp["YVXU"] += Eta1_["wz"] * V_["zVmX"] * T2_["mYwU"];
    E += temp["YVXU"] * Gamma1_["XY"] * Eta1_["UV"];

    //Calculates all but ccvv, cCvV, and CCVV energies


    double Eccvv = 0.0;
    outfile->Printf("...Done. Timing %15.6f s", timer.get());
    std::string strccvv = "Computing <[V, T2]> (C_2)^4 ccvv";
    outfile->Printf("\n    %-36s ...", strccvv.c_str());

    Timer ccvv_timer;
    //TODO:  Make this smarter and automatically switch to right algorithm for size
    //Small size -> use core algorithm
    //Large size -> use fly_ambit

    if(options_.get_str("ccvv_algorithm")=="CORE")
    {
        Eccvv = E_VT2_2_core();

    }
    else if(options_.get_str("ccvv_algorithm")=="FLY_LOOP")
    {
        Eccvv = E_VT2_2_fly_openmp();
    }
    else if(options_.get_str("ccvv_algorithm")=="FLY_AMBIT")
    {
        Eccvv = E_VT2_2_ambit();
    }
    else
    {
        outfile->Printf("\n Specify a correct algorithm string");
        throw PSIEXCEPTION("Specify either CORE FLY_LOOP or FLY_AMBIT");
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
    temp1 = BTF_->build(tensor_type_,"temp1", spin_cases({"aahh"}));
    temp2 = BTF_->build(tensor_type_,"temp2", spin_cases({"aaaa"}));

    if(integral_type_==DiskDF)
    {
        V_ = compute_V_minimal(temp1.block_labels());
        T2_ = compute_T2_minimal({"ccaa", "caaa", "acaa", "aaaa", "CCAA", "CAAA", "ACAA", "AAAA", "cCaA", "cAaA", "aAaA", "aCaA"});
    }

    temp1["uvij"] += V_["uvkl"] * Gamma1_["ki"] * Gamma1_["lj"];
    temp1["UVIJ"] += V_["UVKL"] * Gamma1_["KI"] * Gamma1_["LJ"];
    temp1["uViJ"] += V_["uVkL"] * Gamma1_["ki"] * Gamma1_["LJ"];

    temp2["uvxy"] += temp1["uvij"] * T2_["ijxy"];
    temp2["UVXY"] += temp1["UVIJ"] * T2_["IJXY"];
    temp2["uVxY"] += temp1["uViJ"] * T2_["iJxY"];

    E += 0.125 * Lambda2_["xyuv"] * temp2["uvxy"];
    E += 0.125 * Lambda2_["XYUV"] * temp2["UVXY"];
    E += Lambda2_["xYuV"] * temp2["uVxY"];

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

    temp1 = BTF_->build(tensor_type_,"temp1", spin_cases({"aapp"}));
    temp2 = BTF_->build(tensor_type_,"temp2", spin_cases({"aaaa"}));
    if(integral_type_==DiskDF)
    {
        T2_ = compute_T2_minimal(temp1.block_labels());
        V_ = compute_V_minimal({"aaaa", "avaa", "vvaa", "vaaa", "AAAA", "AVAA", "VVAA", "VAAA", "aAaA", "aVaA", "vVaA", "vAaA"});
    }   

    temp1["uvcd"] += T2_["uvab"] * Eta1_["ac"] * Eta1_["bd"];
    temp1["UVCD"] += T2_["UVAB"] * Eta1_["AC"] * Eta1_["BD"];
    temp1["uVcD"] += T2_["uVaB"] * Eta1_["ac"] * Eta1_["BD"];

    temp2["uvxy"] += temp1["uvcd"] * V_["cdxy"];
    temp2["UVXY"] += temp1["UVCD"] * V_["CDXY"];
    temp2["uVxY"] += temp1["uVcD"] * V_["cDxY"];

    E += 0.125 * Lambda2_["xyuv"] * temp2["uvxy"];
    E += 0.125 * Lambda2_["XYUV"] * temp2["UVXY"];
    E += Lambda2_["xYuV"] * temp2["uVxY"];

    outfile->Printf("  Done. Timing %15.6f s", timer.get());
    return E;
}

double THREE_DSRG_MRPT2::E_VT2_4PH()
{
    Timer timer;
    double E = 0.0;
    std::string str = "Computing [V, T2] 4PH";
    outfile->Printf("\n    %-36s ...", str.c_str());

    BlockedTensor temp1;
    BlockedTensor temp2;
    temp1 = BTF_->build(tensor_type_,"temp1",{"hapa", "HAPA", "hApA", "ahap", "AHAP", "aHaP", "aHpA", "hAaP"});
    temp2 = BTF_->build(tensor_type_,"temp2", spin_cases({"aaaa"}));
    std::vector<std::string> list_of_pphh_V = BTF_->generate_indices("vac", "pphh");
    if(integral_type_==DiskDF)
    {
        T2_ = compute_T2_minimal(temp1.block_labels());
        V_  = compute_V_minimal(BTF_->spin_cases_avoid(list_of_pphh_V,2));
    }

    

    temp1["juby"]  =  T2_["iuay"] * Gamma1_["ji"] * Eta1_["ab"];
    temp2["uvxy"] +=  V_["vbjx"] * temp1["juby"];

    temp1["uJyB"]  =  T2_["uIyA"] * Gamma1_["JI"] * Eta1_["AB"];
    temp2["uvxy"] -=  V_["vBxJ"] * temp1["uJyB"];
    E += temp2["uvxy"] * Lambda2_["xyuv"];

    temp1["JUBY"]  = T2_["IUAY"] * Gamma1_["IJ"] * Eta1_["AB"];
    temp2["UVXY"] += V_["VBJX"] * temp1["JUBY"];

    temp1["jUbY"]  = T2_["iUaY"] * Gamma1_["ji"] * Eta1_["ab"];
    temp2["UVXY"] -= V_["bVjX"] * temp1["jUbY"];
    E += temp2["UVXY"] * Lambda2_["XYUV"];

    temp1["jVbY"]  = T2_["iVaY"] * Gamma1_["ji"] * Eta1_["ab"];
    temp2["uVxY"] -= V_["ubjx"] * temp1["jVbY"];

    temp1["JVBY"]  = T2_["IVAY"] * Gamma1_["JI"] * Eta1_["AB"];
    temp2["uVxY"] += V_["uBxJ"] * temp1["JVBY"];

    temp1["jubx"]  = T2_["iuax"] * Gamma1_["ji"] * Eta1_["ab"];
    temp2["uVxY"] += V_["bVjY"] * temp1["jubx"];

    temp1["uJxB"]  = T2_["uIxA"] * Gamma1_["JI"] * Eta1_["AB"];
    temp2["uVxY"] -= V_["VBJY"] * temp1["uJxB"];

    temp1["uJbY"]  = T2_["uIaY"] * Gamma1_["JI"] * Eta1_["ab"];
    temp2["uVxY"] -= V_["bVxJ"] * temp1["uJbY"];

    temp1["jVxB"]  = T2_["iVxA"] * Gamma1_["ji"] * Eta1_["AB"];
    temp2["uVxY"] -= V_["uBjY"] * temp1["jVxB"];
    E += temp2["uVxY"] * Lambda2_["xYuV"];

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
    temp = BTF_->build(tensor_type_,"temp", spin_cases({"aaaaaa"}));
    std::vector<std::string> list_of_pphh_V = BTF_->generate_indices("vac", "pphh");
    
    if(integral_type_==DiskDF)
    {
        T2_ = compute_T2_minimal(BTF_->spin_cases_avoid(no_hhpp_,3));
        V_  = compute_V_minimal(BTF_->spin_cases_avoid(list_of_pphh_V, 3));
    }

    temp["uvwxyz"] += V_["uviz"] * T2_["iwxy"];
    temp["uvwxyz"] += V_["waxy"] * T2_["uvaz"];      //  aaaaaa from particle
    temp["UVWXYZ"] += V_["UVIZ"] * T2_["IWXY"];      //  AAAAAA from hole
    temp["UVWXYZ"] += V_["WAXY"] * T2_["UVAZ"];      //  AAAAAA from particle
    E += 0.25 * temp["uvwxyz"] * Lambda3_["xyzuvw"];
    E += 0.25 * temp["UVWXYZ"] * Lambda3_["XYZUVW"];

    temp["uvWxyZ"] -= V_["uviy"] * T2_["iWxZ"];      //  aaAaaA from hole
    temp["uvWxyZ"] -= V_["uWiZ"] * T2_["ivxy"];      //  aaAaaA from hole
    temp["uvWxyZ"] += V_["uWyI"] * T2_["vIxZ"];      //  aaAaaA from hole
    temp["uvWxyZ"] += V_["uWyI"] * T2_["vIxZ"];      //  aaAaaA from hole

    temp["uvWxyZ"] += V_["aWxZ"] * T2_["uvay"];      //  aaAaaA from particle
    temp["uvWxyZ"] -= V_["vaxy"] * T2_["uWaZ"];      //  aaAaaA from particle
    temp["uvWxyZ"] -= V_["vAxZ"] * T2_["uWyA"];      //  aaAaaA from particle
    temp["uvWxyZ"] -= V_["vAxZ"] * T2_["uWyA"];      //  aaAaaA from particle

    E += 0.50 * temp["uvWxyZ"] * Lambda3_["xyZuvW"];

    temp["uVWxYZ"] -= V_["VWIZ"] * T2_["uIxY"];      //  aAAaAA from hole
    temp["uVWxYZ"] -= V_["uVxI"] * T2_["IWYZ"];      //  aAAaAA from hole
    temp["uVWxYZ"] += V_["uViZ"] * T2_["iWxY"];      //  aAAaAA from hole
    temp["uVWxYZ"] += V_["uViZ"] * T2_["iWxY"];      //  aAAaAA from hole

    temp["uVWxYZ"] += V_["uAxY"] * T2_["VWAZ"];      //  aAAaAA from particle
    temp["uVWxYZ"] -= V_["WAYZ"] * T2_["uVxA"];      //  aAAaAA from particle
    temp["uVWxYZ"] -= V_["aWxY"] * T2_["uVaZ"];      //  aAAaAA from particle
    temp["uVWxYZ"] -= V_["aWxY"] * T2_["uVaZ"];      //  aAAaAA from particle

    E += 0.5 * temp["uVWxYZ"] * Lambda3_["xYZuVW"];

    outfile->Printf("...Done. Timing %15.6f s", timer.get());
    return E;
}

double THREE_DSRG_MRPT2::E_VT2_2_fly_openmp()
{
    double Eflyalpha = 0.0;
    double Eflybeta = 0.0;
    double Eflymixed = 0.0;
    double Efly = 0.0;
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
                    size_t m =  acore_mos_[mind];
                    size_t n =  acore_mos_[nind];
                    size_t e =  avirt_mos_[eind];
                    size_t f =  bvirt_mos_[find];
                    size_t mb = bcore_mos_[mind];
                    size_t nb = bcore_mos_[nind];
                    size_t eb = bvirt_mos_[eind];
                    size_t fb = bvirt_mos_[find];
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
                    vmnefalphaC = C_DDOT(nthree_,
                            &(ints_->three_integral_pointer()[0][m * ncmo_ + e]),nmo_ * nmo_,
                            &(ints_->three_integral_pointer()[0][n * ncmo_ + f]),nmo_ * nmo_);
                     vmnefalphaE = C_DDOT(nthree_,
                            &(ints_->three_integral_pointer()[0][m * ncmo_ + f]),nmo_ * nmo_,
                            &(ints_->three_integral_pointer()[0][n * ncmo_ + e]),nmo_ * nmo_);
                    vmnefbetaC = C_DDOT(nthree_,
                            &(ints_->three_integral_pointer()[0][mb * ncmo_ + eb]),nmo_ * nmo_,
                            &(ints_->three_integral_pointer()[0][nb * ncmo_ + fb]),nmo_ * nmo_);
                     vmnefbetaE = C_DDOT(nthree_,
                            &(ints_->three_integral_pointer()[0][mb * ncmo_ + fb]),nmo_ * nmo_,
                            &(ints_->three_integral_pointer()[0][nb * ncmo_ + eb]),nmo_ * nmo_);
                    vmnefmixedC = C_DDOT(nthree_,
                            &(ints_->three_integral_pointer()[0][m * ncmo_ + eb]),nmo_ * nmo_,
                            &(ints_->three_integral_pointer()[0][n * ncmo_ + fb]),nmo_ * nmo_);

                    vmnefalpha = vmnefalphaC - vmnefalphaE;
                    vmnefbeta = vmnefbetaC - vmnefbetaE;
                    vmnefmixed = vmnefmixedC;

                    t2alpha = vmnefalpha * renormalized_denominator(Fa_[m] + Fa_[n] - Fa_[e] - Fa_[f ]);
                    t2beta  = vmnefbeta  * renormalized_denominator(Fb_[m] + Fb_[n] - Fb_[e] - Fb_[f ]);
                    t2mixed = vmnefmixed * renormalized_denominator(Fa_[m] + Fb_[n] - Fa_[e] - Fb_[f ]);

                    vmnefalphaR  = vmnefalpha;
                    vmnefbetaR   = vmnefbeta;
                    vmnefmixedR  = vmnefmixed;
                    vmnefalphaR += vmnefalpha * renormalized_exp(Fa_[m] + Fa_[n] - Fa_[e] - Fa_[f]);
                    vmnefbetaR  += vmnefbeta * renormalized_exp(Fb_[m] + Fb_[n]  - Fb_[e] - Fb_[f]);
                    vmnefmixedR += vmnefmixed * renormalized_exp(Fa_[m] + Fb_[n] - Fa_[e] - Fb_[f]);

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
    // Compute <[V, T2]> (C_2)^4 ccvv term; (me|nf) = B(L|me) * B(L|nf)
    // For a given m and n, form Bm(L|e) and Bn(L|f)
    // Bef(ef) = Bm(L|e) * Bn(L|f)
    size_t dim = nthree_ * virtual_;
    std::vector<size_t> naux(nthree_);
    std::iota(naux.begin(), naux.end(), 0);

    std::vector<size_t> virt_mos = mo_space_info_->get_corr_abs_mo("RESTRICTED_UOCC");

    double Ealpha = 0.0;
    double Ebeta  = 0.0;
    double Emixed = 0.0;
    int nthread = 1;
    #ifdef _OPENMP
        nthread = omp_get_max_threads();
    #endif
    std::vector<ambit::Tensor> BefVec;
    std::vector<ambit::Tensor> BefJKVec;
    std::vector<ambit::Tensor> RDVec;
    std::vector<ambit::Tensor> BmaVec_three;
    std::vector<ambit::Tensor> BnaVec_three;
    std::vector<ambit::Tensor> BmbVec_three;
    std::vector<ambit::Tensor> BnbVec_three;
    std::vector<ambit::Tensor> BmaVec;
    std::vector<ambit::Tensor> BnaVec;
    std::vector<ambit::Tensor> BmbVec;
    std::vector<ambit::Tensor> BnbVec;
    std::vector<std::vector<size_t>> ma_vec;
    std::vector<std::vector<size_t>> mb_vec;
    std::vector<std::vector<size_t>> na_vec;
    std::vector<std::vector<size_t>> nb_vec;
    for (int i = 0; i < nthread; i++)
    {
        BmaVec.push_back(ambit::Tensor::build(tensor_type_,"Bma",{nthree_,virtual_}));
        BnaVec.push_back(ambit::Tensor::build(tensor_type_,"Bna",{nthree_,virtual_}));
        BmbVec.push_back(ambit::Tensor::build(tensor_type_,"Bmb",{nthree_,virtual_}));
        BnbVec.push_back(ambit::Tensor::build(tensor_type_,"Bnb",{nthree_,virtual_}));
        BefVec.push_back(ambit::Tensor::build(tensor_type_,"Bef",{virtual_,virtual_}));
        BefJKVec.push_back(ambit::Tensor::build(tensor_type_,"BefJK",{virtual_,virtual_}));
        BmaVec_three.push_back(ambit::Tensor::build(tensor_type_,"Bma",{nthree_,1,virtual_}));
        BnaVec_three.push_back(ambit::Tensor::build(tensor_type_,"Bna",{nthree_,1,virtual_}));
        BmbVec_three.push_back(ambit::Tensor::build(tensor_type_,"Bmb",{nthree_,1,virtual_}));
        BnbVec_three.push_back(ambit::Tensor::build(tensor_type_,"Bnb",{nthree_,1,virtual_}));
        RDVec.push_back(ambit::Tensor::build(tensor_type_, "RDVec", {virtual_, virtual_}));
        ma_vec.push_back(std::vector<size_t>(1));
        mb_vec.push_back(std::vector<size_t>(1));
        na_vec.push_back(std::vector<size_t>(1));
        nb_vec.push_back(std::vector<size_t>(1));

    }
    
    #pragma omp parallel for num_threads(num_threads_) \
    schedule(dynamic) \
    reduction(+:Ealpha, Ebeta, Emixed) 

    for(size_t m = 0; m < core_; ++m){
         
        int thread = 0;
        #ifdef _OPENMP
            thread = omp_get_thread_num();
        #endif
        size_t ma = acore_mos_[m];
        size_t mb = bcore_mos_[m];
        #pragma omp critical
        {
            ma_vec[thread][0] = ma;
            mb_vec[thread][0] = mb;
            BmaVec_three[thread] = ints_->three_integral_block(naux, ma_vec[thread], virt_mos);
            BmbVec_three[thread] = ints_->three_integral_block(naux, mb_vec[thread], virt_mos);
            std::copy(&BmaVec_three[thread].data()[0], &BmaVec_three[thread].data()[dim], BmaVec[thread].data().begin());
            std::copy(&BmbVec_three[thread].data()[0], &BmbVec_three[thread].data()[dim], BmbVec[thread].data().begin());
        }
        for(size_t n = 0; n < core_; ++n){
            size_t na = acore_mos_[n];
            size_t nb = bcore_mos_[n];
            na_vec[thread][0] = na;
            nb_vec[thread][0] = nb;
            #pragma omp critical
            {
                BnaVec_three[thread] = ints_->three_integral_block(naux, na_vec[thread], virt_mos);
                BnbVec_three[thread] = ints_->three_integral_block(naux, nb_vec[thread], virt_mos);
                std::copy(&BnaVec_three[thread].data()[0], &BnaVec_three[thread].data()[dim], BnaVec[thread].data().begin());
                std::copy(&BnbVec_three[thread].data()[0], &BnbVec_three[thread].data()[dim], BnbVec[thread].data().begin());
            }

            // alpha-aplha
            BefVec[thread]("ef") = BmaVec[thread]("ge") * BnaVec[thread]("gf");
            BefJKVec[thread]("ef")  = BefVec[thread]("ef") * BefVec[thread]("ef");
            BefJKVec[thread]("ef") -= BefVec[thread]("ef") * BefVec[thread]("fe");
            RDVec[thread].iterate([&](const std::vector<size_t>& i,double& value){
                double D = Fa_[ma] + Fa_[na] - Fa_[avirt_mos_[i[0]]] - Fa_[avirt_mos_[i[1]]];
                value = renormalized_denominator(D) * (1.0 + renormalized_exp(D));});
            Ealpha += 0.5 * BefJKVec[thread]("ef") * RDVec[thread]("ef");

            // beta-beta
            BefVec[thread]("EF") = BmbVec[thread]("gE") * BnbVec[thread]("gF");
            BefJKVec[thread]("EF")  = BefVec[thread]("EF") * BefVec[thread]("EF");
            BefJKVec[thread]("EF") -= BefVec[thread]("EF") * BefVec[thread]("FE");
            RDVec[thread].iterate([&](const std::vector<size_t>& i,double& value){
                double D = Fb_[mb] + Fb_[nb] - Fb_[bvirt_mos_[i[0]]] - Fb_[bvirt_mos_[i[1]]];
                value = renormalized_denominator(D) * (1.0 + renormalized_exp(D));});
            Ebeta += 0.5 * BefJKVec[thread]("EF") * RDVec[thread]("EF");

            // alpha-beta
            BefVec[thread]("eF") = BmaVec[thread]("ge") * BnbVec[thread]("gF");
            BefJKVec[thread]("eF")  = BefVec[thread]("eF") * BefVec[thread]("eF");
            RDVec[thread].iterate([&](const std::vector<size_t>& i,double& value){
                double D = Fa_[ma] + Fb_[nb] - Fa_[avirt_mos_[i[0]]] - Fb_[bvirt_mos_[i[1]]];
                value = renormalized_denominator(D) * (1.0 + renormalized_exp(D));});
            Emixed += BefJKVec[thread]("eF") * RDVec[thread]("eF");
        }
    }

    return (Ealpha + Ebeta + Emixed);
}
double THREE_DSRG_MRPT2::E_VT2_2_core()
{
    double E2_core = 0.0;
    BlockedTensor T2ccvv = BTF_->build(tensor_type_,"T2ccvv", spin_cases({"ccvv"}));
    BlockedTensor v    = BTF_->build(tensor_type_, "Vccvv", spin_cases({"ccvv"}));

    BlockedTensor ThreeIntegral = BTF_->build(tensor_type_,"ThreeInt",{"dph","dPH"});
    ThreeIntegral.iterate([&](const std::vector<size_t>& i,const std::vector<SpinType>& spin,double& value){
        value = ints_->three_integral(i[0],i[1],i[2]);
    });

    v("mnef") = ThreeIntegral("gem") * ThreeIntegral("gfn");
    v("mnef") -= ThreeIntegral("gfm") * ThreeIntegral("gen");
    v("MNEF") = ThreeIntegral("gEM") * ThreeIntegral("gFN");
    v("MNEF") -= ThreeIntegral("gFM") * ThreeIntegral("gEN");
    v("mNeF") = ThreeIntegral("gem") * ThreeIntegral("gFN");

    if(options_.get_str("CCVV_SOURCE")=="NORMAL")
    {
        BlockedTensor RD2_ccvv = BTF_->build(tensor_type_, "RDelta2ccvv", spin_cases({"ccvv"}));
        BlockedTensor RExp2ccvv = BTF_->build(tensor_type_, "RExp2ccvv", spin_cases({"ccvv"}));
        RD2_ccvv.iterate([&](const std::vector<size_t>& i,const std::vector<SpinType>& spin,double& value){
            if ((spin[0] == AlphaSpin) and (spin[1] == AlphaSpin)){
                value = renormalized_denominator(Fa_[i[0]] + Fa_[i[1]] - Fa_[i[2]] - Fa_[i[3]]);
            }else if ((spin[0] == AlphaSpin) and (spin[1] == BetaSpin) ){
                value = renormalized_denominator(Fa_[i[0]] + Fb_[i[1]] - Fa_[i[2]] - Fb_[i[3]]);
            }else if ((spin[0] == BetaSpin)  and (spin[1] == BetaSpin) ){
                value = renormalized_denominator(Fb_[i[0]] + Fb_[i[1]] - Fb_[i[2]] - Fb_[i[3]]);
            }
        });
        RExp2ccvv.iterate([&](const std::vector<size_t>& i,const std::vector<SpinType>& spin,double& value){
            if ((spin[0] == AlphaSpin) and (spin[1] == AlphaSpin)){
                value = renormalized_exp(Fa_[i[0]] + Fa_[i[1]] - Fa_[i[2]] - Fa_[i[3]]);
            }else if ((spin[0] == AlphaSpin) and (spin[1] == BetaSpin) ){
                value = renormalized_exp(Fa_[i[0]] + Fb_[i[1]] - Fa_[i[2]] - Fb_[i[3]]);
            }else if ((spin[0] == BetaSpin)  and (spin[1] == BetaSpin) ){
                value = renormalized_exp(Fb_[i[0]] + Fb_[i[1]] - Fb_[i[2]] - Fb_[i[3]]);
            }
        });
        BlockedTensor Rv = BTF_->build(tensor_type_, "ReV", spin_cases({"ccvv"}));
        Rv("mnef") = v("mnef");
        Rv("mNeF") = v("mNeF");
        Rv("MNEF") = v("MNEF");
        Rv("mnef") += v("mnef")*RExp2ccvv("mnef");
        Rv("MNEF") += v("MNEF")*RExp2ccvv("MNEF");
        Rv("mNeF") += v("mNeF")*RExp2ccvv("mNeF");


        T2ccvv["MNEF"] = V_["MNEF"] * RD2_ccvv["MNEF"];
        T2ccvv["mnef"] = V_["mnef"] * RD2_ccvv["mnef"];
        T2ccvv["mNeF"] = V_["mNeF"] * RD2_ccvv["mNeF"];
        E2_core += 0.25 * T2ccvv["mnef"] * Rv["mnef"];
        E2_core += 0.25 * T2ccvv["MNEF"] * Rv["MNEF"];
        E2_core += T2ccvv["mNeF"] * Rv["mNeF"];
    }
    else if (options_.get_str("CCVV_SOURCE")=="ZERO")
    {
        BlockedTensor Denom = BTF_->build(tensor_type_,"Mp2Denom", spin_cases({"ccvv"}));
        Denom.iterate([&](const std::vector<size_t>& i,const std::vector<SpinType>& spin,double& value){
        if ((spin[0] == AlphaSpin) and (spin[1] == AlphaSpin)){
            value = 1.0/(Fa_[i[0]] + Fa_[i[1]] - Fa_[i[2]] - Fa_[i[3]]);
        }else if ((spin[0] == AlphaSpin) and (spin[1] == BetaSpin) ){
            value = 1.0/(Fa_[i[0]] + Fb_[i[1]] - Fa_[i[2]] - Fb_[i[3]]);
        }else if ((spin[0] == BetaSpin)  and (spin[1] == BetaSpin) ){
            value = 1.0/(Fb_[i[0]] + Fb_[i[1]] - Fb_[i[2]] - Fb_[i[3]]);
        }
    });
        T2ccvv["MNEF"] = V_["MNEF"]*Denom["MNEF"];
        T2ccvv["mnef"] = V_["mnef"]*Denom["mnef"];
        T2ccvv["mNeF"] = V_["mNeF"]*Denom["mNeF"];

        E2_core += 0.25 * T2ccvv["mnef"] * V_["mnef"];
        E2_core += 0.25 * T2ccvv["MNEF"] * V_["MNEF"];
        E2_core += T2ccvv["mNeF"] * V_["mNeF"];

    }

    return E2_core;
}

}} // End Namespaces
