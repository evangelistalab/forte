#include <numeric>

#include <libpsio/psio.hpp>
#include <libpsio/psio.h>
#include <libmints/molecule.h>
#include <libmints/matrix.h>
#include <libmints/vector.h>
#include <libqt/qt.h>
#include "blockedtensorfactory.h"
#include "fci_solver.h"
#include "fci_vector.h"

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
      tensor_type_(ambit::CoreTensor),
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
    outfile->Printf("\n  Reference = %s", ref_type_.c_str());

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

    //BlockedTensor::add_mo_space("@","$",nauxpi,NoSpin);
    //BlockedTensor::add_mo_space("d","g",nauxpi,NoSpin);
    BTF_->add_mo_space("d","g",nauxpi,NoSpin);

    H_ = BTF_->build(tensor_type_,"H",spin_cases({"gg"}));

    Gamma1_ = BTF_->build(tensor_type_,"Gamma1_",spin_cases({"hh"}));
    Eta1_ = BTF_->build(tensor_type_,"Eta1_",spin_cases({"pp"}));
    F_ = BTF_->build(tensor_type_,"Fock",spin_cases({"gg"}));
    F_no_renorm_ = BTF_->build(tensor_type_,"Fock",spin_cases({"gg"}));
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

    Gamma1_AA.iterate([&](const std::vector<size_t>& i,double& value){
     Gamma1_matrixB->set(bactv_mos_[i[0]], bactv_mos_[i[1]], value);   
     });

    ints_->make_fock_matrix(Gamma1_matrixA, Gamma1_matrixB);

    F_.iterate([&](const std::vector<size_t>& i,const std::vector<SpinType>& spin,double& value)
    {
        if (spin[0] == AlphaSpin){
            value = ints_->get_fock_a(i[0],i[1]);
        }else if (spin[0]  == BetaSpin){
            value = ints_->get_fock_b(i[0],i[1]);
        }
    });
    F_no_renorm_["pq"] = F_["pq"];
    F_no_renorm_["PQ"] = F_["PQ"];

    Dimension ncmopi_ = mo_space_info_->get_dimension("CORRELATED");

    Fa_.reserve(ncmo_);
    Fb_.reserve(ncmo_);

    for(size_t p = 0; p < ncmo_; p++)
    {
        Fa_[p] = ints_->get_fock_a(p,p);
        Fb_[p] = ints_->get_fock_b(p,p);
    }
    if(options_.get_bool("MOLDEN_WRITE_FORTE"))
    {
        Dimension nmopi_ = mo_space_info_->get_dimension("ALL");
        int nirrep = Process::environment.wavefunction()->nirrep();
        boost::shared_ptr<Vector>occ_vector(new Vector(nirrep, nmopi_));

        for(auto orb_energy : Fa_)
        {
            outfile->Printf(" %8.8f", orb_energy);
        }
        ///Not right, but I do not care about occupation.  Look at orbital energies
        for(int h = 0; h < nirrep; h++)
            for(int i = 0; i < nmopi_[h]; i++)
                occ_vector->set(h, i, 0.0);


        view_modified_orbitals(Process::environment.wavefunction()->Ca(), Process::environment.wavefunction()->epsilon_a(), occ_vector );
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
    Lambda2_ = BTF_->build(tensor_type_,"Lambda2_",spin_cases({"aaaa"}));
    ambit::Tensor Lambda2_aa = Lambda2_.block("aaaa");
    ambit::Tensor Lambda2_aA = Lambda2_.block("aAaA");
    ambit::Tensor Lambda2_AA = Lambda2_.block("AAAA");
    Lambda2_aa("pqrs") = reference_.L2aa()("pqrs");
    Lambda2_aA("pqrs") = reference_.L2ab()("pqrs");
    Lambda2_AA("pqrs") = reference_.L2bb()("pqrs");

    Lambda3_ = BTF_->build(tensor_type_,"Lambda3_",spin_cases({"aaaaaa"}));
    if(options_.get_str("THREEPDC") != "ZERO"){
        ambit::Tensor Lambda3_aaa = Lambda3_.block("aaaaaa");
        ambit::Tensor Lambda3_aaA = Lambda3_.block("aaAaaA");
        ambit::Tensor Lambda3_aAA = Lambda3_.block("aAAaAA");
        ambit::Tensor Lambda3_AAA = Lambda3_.block("AAAAAA");
        Lambda3_aaa("pqrstu") = reference_.L3aaa()("pqrstu");
        Lambda3_aaA("pqrstu") = reference_.L3aab()("pqrstu");
        Lambda3_aAA("pqrstu") = reference_.L3abb()("pqrstu");
        Lambda3_AAA("pqrstu") = reference_.L3bbb()("pqrstu");
    }

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
        std::vector<std::string> list_of_pphh_V = BTF_->generate_indices("vac", "pphh");
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
    if(integral_type_!=DiskDF){renormalize_V();}
    if(integral_type_==DiskDF)
    {
        size_t memory_cost = nmo_ * nmo_ * nmo_ * active_ * 16;
        bool exceed_memory = memory_cost < Process::environment.get_memory();
        exceed_memory = false;

        std::vector<std::string> list_of_pphh_V = BTF_->generate_indices("vac", "pphh");
        std::string str = "Computing T2";
        outfile->Printf("\n    %-36s ...", str.c_str());
        Timer T2timer;

        // If exceed memory, use diskbased algorithm
        //for all terms with <= 1 active idex
        //If not, just compute V in the beginning

        if(!exceed_memory)
        {
            T2_ = compute_T2_minimal(BTF_->spin_cases_avoid(no_hhpp_,2));
        }
        else { T2_ = compute_T2_minimal(BTF_->spin_cases_avoid(no_hhpp_, 1));}
        outfile->Printf("...Done. Timing %15.6f s", T2timer.get());

        std::string strV = "Computing V and Renormalizing";
        outfile->Printf("\n    %-36s ...", strV.c_str());
        Timer Vtimer;
        if(!exceed_memory)
        {
            V_  = compute_V_minimal(BTF_->spin_cases_avoid(list_of_pphh_V, 2));
        }
        else {V_ = compute_V_minimal(BTF_->spin_cases_avoid(list_of_pphh_V, 1));}
        outfile->Printf("...Done. Timing %15.6f s", Vtimer.get());
    }
    compute_t1();
    check_t1();


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

    if(options_.get_bool("PRINT_DENOM2"))
    {
        std::ofstream myfile;
        myfile.open ("DENOM.txt");
        ambit::BlockedTensor Delta2 = BTF_->build(tensor_type_,"Delta1_",{"cavv", "ccvv", "ccva"});
        Delta2.iterate([&](const std::vector<size_t>& i,const std::vector<SpinType>& spin,double& value){
            if (spin[0]  == AlphaSpin){
                value = 1 / (Fa_[i[0]] + Fa_[i[1]]- Fa_[i[2]] - Fa_[i[3]]);
            }
        });
        ambit::Tensor Delta2ccvv = Delta2.block("ccvv");
        ambit::Tensor Delta2cavv = Delta2.block("cavv");
        ambit::Tensor Delta2ccva = Delta2.block("ccva");
        myfile << "ccvv DELTA2\n";
        int count = 0;
        Delta2ccvv.iterate([&](const std::vector<size_t>&,double& value){
            myfile << count << "  " << value << "\n";
        });
        myfile << "cavv DELTA2\n";

        count = 0;
        Delta2cavv.iterate([&](const std::vector<size_t>&,double& value){
            myfile << count << "  " << value << "\n";
        });
        myfile << "ccva DELTA2\n";

        count = 0;
        Delta2ccva.iterate([&](const std::vector<size_t>&,double& value){
            myfile << count << "  " << value << "\n";
        });

    }
    Hbar0_ = Etotal - Eref_;
    if(options_.get_str("RELAX_REF") != "NONE")
    {
        relax_reference_once();
    }
    return Etotal;
}

double THREE_DSRG_MRPT2::compute_ref()
{
    double E = 0.0;

    E  = 0.5 * H_["ij"] * Gamma1_["ij"];
    E += 0.5 * F_["ij"] * Gamma1_["ij"];
    E += 0.5 * H_["IJ"] * Gamma1_["IJ"];
    E += 0.5 * F_["IJ"] * Gamma1_["IJ"];

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
    for(auto space : spaces)
    {
        outfile->Printf("\n space: %s ThreeIntNorm: %8.6f", space.c_str(), Vmin.block(space).norm(2));
    }

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

    //TODO: Implement these without storing V and/or T2 by using blocking
    ambit::BlockedTensor temp = BTF_->build(tensor_type_, "temp",{"aa", "AA"});

    if( (integral_type_!=DiskDF))
    {
        temp.zero();
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
    }
    else
    {
        E += E_VT2_2_one_active();
    }
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

    if(options_.get_str("THREEPDC") != "ZERO"){
        BlockedTensor temp;
        temp = BTF_->build(tensor_type_,"temp", spin_cases({"aaaaaa"}));

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
    }

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
    /// This block of code assumes that ThreeIntegral are not stored as a member variable.  Requires the reading from aptei_block which makes code
    ///general for all, but makes it slow for DiskDF.

    if(integral_type_==DiskDF)
    {
        std::vector<ambit::Tensor> BefVec;
        std::vector<ambit::Tensor> BefJKVec;
        std::vector<ambit::Tensor> RDVec;
        std::vector<ambit::Tensor> BmaVec;
        std::vector<ambit::Tensor> BnaVec;
        std::vector<ambit::Tensor> BmbVec;
        std::vector<ambit::Tensor> BnbVec;

        for (int i = 0; i < nthread; i++)
        {
            BmaVec.push_back(ambit::Tensor::build(tensor_type_,"Bma",{nthree_,virtual_}));
            BnaVec.push_back(ambit::Tensor::build(tensor_type_,"Bna",{nthree_,virtual_}));
            BmbVec.push_back(ambit::Tensor::build(tensor_type_,"Bmb",{nthree_,virtual_}));
            BnbVec.push_back(ambit::Tensor::build(tensor_type_,"Bnb",{nthree_,virtual_}));
            BefVec.push_back(ambit::Tensor::build(tensor_type_,"Bef",{virtual_,virtual_}));
            BefJKVec.push_back(ambit::Tensor::build(tensor_type_,"BefJK",{virtual_,virtual_}));
            RDVec.push_back(ambit::Tensor::build(tensor_type_, "RDVec", {virtual_, virtual_}));

        }
        
        #pragma omp parallel for num_threads(num_threads_) \
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
                BmaVec[thread] = ints_->three_integral_block_two_index(naux, ma, virt_mos);
                BmbVec[thread] = ints_->three_integral_block_two_index(naux, ma, virt_mos);
            }
            for(size_t n = 0; n < core_; ++n){
                size_t na = acore_mos_[n];
                size_t nb = bcore_mos_[n];
                #pragma omp critical
                {
                    BnaVec[thread] = ints_->three_integral_block_two_index(naux, na, virt_mos);
                    BnbVec[thread] = ints_->three_integral_block_two_index(naux, na, virt_mos);
                }

                // alpha-aplha
                BefVec[thread].zero();
                BefJKVec[thread].zero();
                RDVec[thread].zero();

                BefVec[thread]("ef") = BmaVec[thread]("ge") * BnaVec[thread]("gf");
                BefJKVec[thread]("ef")  = BefVec[thread]("ef") * BefVec[thread]("ef");
                BefJKVec[thread]("ef") -= BefVec[thread]("ef") * BefVec[thread]("fe");
                RDVec[thread].iterate([&](const std::vector<size_t>& i,double& value){
                    double D = Fa_[ma] + Fa_[na] - Fa_[avirt_mos_[i[0]]] - Fa_[avirt_mos_[i[1]]];
                    value = renormalized_denominator(D) * (1.0 + renormalized_exp(D));});
                Ealpha += 0.5 * BefJKVec[thread]("ef") * RDVec[thread]("ef");

                BefVec[thread].zero();
                BefJKVec[thread].zero();
                RDVec[thread].zero();

                // beta-beta
                BefVec[thread]("EF") = BmbVec[thread]("gE") * BnbVec[thread]("gF");
                BefJKVec[thread]("EF")  = BefVec[thread]("EF") * BefVec[thread]("EF");
                BefJKVec[thread]("EF") -= BefVec[thread]("EF") * BefVec[thread]("FE");
                RDVec[thread].iterate([&](const std::vector<size_t>& i,double& value){
                    double D = Fb_[mb] + Fb_[nb] - Fb_[bvirt_mos_[i[0]]] - Fb_[bvirt_mos_[i[1]]];
                    value = renormalized_denominator(D) * (1.0 + renormalized_exp(D));});
                Ebeta += 0.5 * BefJKVec[thread]("EF") * RDVec[thread]("EF");

                // alpha-beta
                BefVec[thread].zero();
                BefJKVec[thread].zero();

                BefVec[thread]("eF") = BmaVec[thread]("ge") * BnbVec[thread]("gF");
                BefJKVec[thread]("eF")  = BefVec[thread]("eF") * BefVec[thread]("eF");
                RDVec[thread].iterate([&](const std::vector<size_t>& i,double& value){
                    double D = Fa_[ma] + Fb_[nb] - Fa_[avirt_mos_[i[0]]] - Fb_[bvirt_mos_[i[1]]];
                    value = renormalized_denominator(D) * (1.0 + renormalized_exp(D));});
                Emixed += BefJKVec[thread]("eF") * RDVec[thread]("eF");
            }
        }
    }
    /// This block of code runs with DF and assumes that ThreeIntegral_ is created in startup.  Will fail for systems around 800 or 900 BF
    else
    {
        ambit::Tensor Ba = ambit::Tensor::build(tensor_type_,"Ba",{core_,nthree_,virtual_});
        ambit::Tensor Bb = ambit::Tensor::build(tensor_type_,"Bb",{core_,nthree_,virtual_});
        Ba("mge") = (ThreeIntegral_.block("dvc"))("gem");
        Bb("MgE") = (ThreeIntegral_.block("dvc"))("gEM");

        std::vector<ambit::Tensor> BmaVec;
        std::vector<ambit::Tensor> BnaVec;
        std::vector<ambit::Tensor> BmbVec;
        std::vector<ambit::Tensor> BnbVec;
        std::vector<ambit::Tensor> BefVec;
        std::vector<ambit::Tensor> BefJKVec;
        std::vector<ambit::Tensor> RDVec;
        for (int i = 0; i < nthread; i++)
        {
         BmaVec.push_back(ambit::Tensor::build(tensor_type_,"Bma",{nthree_,virtual_}));
         BnaVec.push_back(ambit::Tensor::build(tensor_type_,"Bna",{nthree_,virtual_}));
         BmbVec.push_back(ambit::Tensor::build(tensor_type_,"Bmb",{nthree_,virtual_}));
         BnbVec.push_back(ambit::Tensor::build(tensor_type_,"Bnb",{nthree_,virtual_}));
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
            size_t ma = acore_mos_[m];
            size_t mb = bcore_mos_[m];

            std::copy(&Ba.data()[m * dim], &Ba.data()[m * dim + dim], BmaVec[thread].data().begin());
            //std::copy(&Bb.data()[m * dim], &Bb.data()[m * dim + dim], BmbVec[thread].data().begin());
            std::copy(&Ba.data()[m * dim], &Ba.data()[m * dim + dim], BmbVec[thread].data().begin());

            for(size_t n = 0; n < core_; ++n){
                size_t na = acore_mos_[n];
                size_t nb = bcore_mos_[n];
                
                std::copy(&Ba.data()[n * dim], &Ba.data()[n * dim + dim], BnaVec[thread].data().begin());
                //std::copy(&Bb.data()[n * dim], &Bb.data()[n * dim + dim], BnbVec[thread].data().begin());
                std::copy(&Ba.data()[n * dim], &Ba.data()[n * dim + dim], BnbVec[thread].data().begin());

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
    }

    return (Ealpha + Ebeta + Emixed);
}
double THREE_DSRG_MRPT2::E_VT2_2_core()
{
    double E2_core = 0.0;
    BlockedTensor T2ccvv = BTF_->build(tensor_type_,"T2ccvv", spin_cases({"ccvv"}));
    BlockedTensor v    = BTF_->build(tensor_type_, "Vccvv", spin_cases({"ccvv"}));

    BlockedTensor ThreeIntegral = BTF_->build(tensor_type_,"ThreeInt",{"dph","dPH"});
    ThreeIntegral.iterate([&](const std::vector<size_t>& i,const std::vector<SpinType>& ,double& value){
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
double THREE_DSRG_MRPT2::E_VT2_2_one_active()
{
    double Eccva = 0;
    double Eacvv = 0;
    int nthread = 1;
    int thread  = 0;
    #ifdef _OPENMP
        nthread = omp_get_max_threads();
        thread  = omp_get_thread_num();
    #endif
/// This block of code assumes that ThreeIntegral are not stored as a member variable.  Requires the reading from aptei_block which makes code
    std::vector<size_t> naux(nthree_);
    std::iota(naux.begin(), naux.end(), 0);
    ambit::Tensor  Gamma1_aa  = Gamma1_.block("aa");
    ambit::Tensor  Gamma1_AA  = Gamma1_.block("AA");


    std::vector<ambit::Tensor>  Bm_Qe;
    std::vector<ambit::Tensor>  Bm_Qf;

    std::vector<ambit::Tensor>  Vefu;
    std::vector<ambit::Tensor>  Tefv;
    std::vector<ambit::Tensor>  tempTaa;
    std::vector<ambit::Tensor>  tempTAA;

    Timer ccvaTimer;
    for(int thread = 0; thread < nthread; thread++)
    {
        Bm_Qe.push_back(ambit::Tensor::build(tensor_type_, "BemQ", {nthree_, virtual_}));
        Bm_Qf.push_back(ambit::Tensor::build(tensor_type_, "Bmq", {nthree_, virtual_}));

        Vefu.push_back(ambit::Tensor::build(tensor_type_, "muJK", {virtual_, virtual_,active_}));
        Tefv.push_back(ambit::Tensor::build(tensor_type_, "T2", {virtual_, virtual_, active_}));

        tempTaa.push_back(ambit::Tensor::build(tensor_type_, "TEMPaa", {active_, active_}));
        tempTAA.push_back(ambit::Tensor::build(tensor_type_, "TEMPAA", {active_, active_}));

    }
    //ambit::Tensor BemQ = ints_->three_integral_block(naux,  acore_mos_, avirt_mos_);
    //ambit::Tensor BeuQ = ints_->three_integral_block(naux,  aactv_mos_, avirt_mos_);

    ///Loop over e and f to compute V

    ambit::Tensor BeuQ = ints_->three_integral_block(naux, avirt_mos_, aactv_mos_);

    //std::vector<double>& BemQ_data = BemQ.data();

    ///I think this loop is typically too small to allow efficient use of
    ///OpenMP.  Should probably test this assumption.
    #pragma omp parallel for num_threads(num_threads_) if (core_ > 1000)
    for(size_t m = 0; m < core_; m++)
    {
        int thread = 0;
        #ifdef _OPENMP
            thread = omp_get_thread_num();
        #endif
        size_t ma = acore_mos_[m];

        //V[efu]_m = B_{em}^Q * B_{fu}^Q - B_{eu}^Q B_{fm}^Q
        //V[efu]_m = V[efmu] + V[efmu] * exp[efmu]
        //T2["mvef"] = V["mvef"] * D["mvef"]
        //temp["uv"] = V * T2
        #pragma omp critical
        {
            Bm_Qe[thread] = ints_->three_integral_block_two_index(naux, ma, avirt_mos_);
        }


        Vefu[thread]("e, f, u") =  Bm_Qe[thread]("Q, e") * BeuQ("Q, f, u");
        Vefu[thread]("e, f, u") -= BeuQ("Q, e, u") * Bm_Qe[thread]("Q, f");

        //E = V["efmu"] (1 + Exp(-s * D^{ef}_{mu}) * V^{mv}_{ef} * Denom^{mv}_{ef}
        #pragma omp critical
        {
            Tefv[thread].data() = Vefu[thread].data();

            std::vector<double>& T_mv_data = Tefv[thread].data();
            Vefu[thread].iterate([&](const std::vector<size_t>& i,double& value){
                double Exp = Fa_[avirt_mos_[i[0]]] + Fa_[avirt_mos_[i[1]]] - Fa_[aactv_mos_[i[2]]] - Fa_[ma];
                double D = -1.0 * (Fa_[avirt_mos_[i[0]]] + Fa_[avirt_mos_[i[1]]] - Fa_[aactv_mos_[i[2]]] - Fa_[ma]);
                value = value + value * renormalized_exp(Exp);
                T_mv_data[i[0] * virtual_ * active_ + i[1] * active_ + i[2]] *= renormalized_denominator(D);
            });
        }

            //T_mv[thread].iterate([&](const std::vector<size_t>& i,double& value){
            //    double D = Fa_[aactv_mos_[i[1]]] + Fa_[acore_mos_[i[0]]] - Fa_[ea] - Fa_[fa];
            //    value = value * renormalized_denominator(D);});

        tempTaa[thread]("u,v")+= 0.5 * Vefu[thread]("e, f, u") * Tefv[thread]("e, f, v");
        Vefu[thread].zero();
        Tefv[thread].zero();

        Vefu[thread].zero();
        Vefu[thread]("e, f, u") =  Bm_Qe[thread]("Q, e") * BeuQ("Q, f, u");

        //E = V["efmu"] (1 + Exp(-s * D^{ef}_{mu}) * V^{mv}_{ef} * Denom^{mv}_{ef}
        #pragma omp critical
        {
            Tefv[thread].data() = Vefu[thread].data();
            std::vector<double>& T_mv_data = Tefv[thread].data();
            T_mv_data = Tefv[thread].data();
            Vefu[thread].iterate([&](const std::vector<size_t>& i,double& value){
                double Exp = Fa_[avirt_mos_[i[0]]] + Fb_[avirt_mos_[i[1]]] - Fa_[aactv_mos_[i[2]]] - Fb_[ma];
                double D = -1.0 * (Fa_[avirt_mos_[i[0]]] + Fb_[avirt_mos_[i[1]]] - Fa_[aactv_mos_[i[2]]] - Fb_[ma]);
                value = value + value * renormalized_exp(Exp);
                T_mv_data[i[0] * virtual_ * active_ + i[1] * active_ + i[2]] *= renormalized_denominator(D);
            });
        }

            //T_mv[thread].iterate([&](const std::vector<size_t>& i,double& value){
            //    double D = Fa_[aactv_mos_[i[1]]] + Fa_[acore_mos_[i[0]]] - Fa_[ea] - Fa_[fa];
            //    value = value * renormalized_denominator(D);});

        tempTAA[thread]("vu")+=Vefu[thread]("e, f, u") * Tefv[thread]("e,f, v");
        tempTaa[thread]("vu")+=Vefu[thread]("e,f, u") * Tefv[thread]("e,f, v");
        Vefu[thread].zero();
        Tefv[thread].zero();

        Vefu[thread]("e, f, u") =  Bm_Qe[thread]("Q, e") * BeuQ("Q, f, u");
        Vefu[thread]("e, f, u") -= BeuQ("Q, e, u") * Bm_Qe[thread]("Q, f");

        //E = V["efmu"] (1 + Exp(-s * D^{ef}_{mu}) * V^{mv}_{ef} * Denom^{mv}_{ef}

        #pragma omp critical 
        {
            Tefv[thread].data() = Vefu[thread].data();
            std::vector<double>& T_mv_data = Tefv[thread].data();

            T_mv_data = Tefv[thread].data();
            Vefu[thread].iterate([&](const std::vector<size_t>& i,double& value){
                double Exp = Fa_[bvirt_mos_[i[0]]] + Fb_[bvirt_mos_[i[1]]] - Fb_[bactv_mos_[i[2]]] - Fb_[ma];
                double D = -1.0 * (Fa_[bvirt_mos_[i[0]]] + Fa_[bvirt_mos_[i[1]]] - Fb_[bactv_mos_[i[2]]] - Fb_[ma]);
                value = value + value * renormalized_exp(Exp);
                T_mv_data[i[0] * virtual_ * active_ + i[1] * active_ + i[2]] *= renormalized_denominator(D);
            });
        }


        tempTaa[thread]("u,v")+= 0.5 * Vefu[thread]("e, f, u") * Tefv[thread]("e, f, v");

    }

    ambit::Tensor tempTAA_all = ambit::Tensor::build(tensor_type_, "tempTAA_all", {active_, active_});
    ambit::Tensor tempTaa_all = ambit::Tensor::build(tensor_type_, "tempTaa_all", {active_, active_});
    for(int thread = 0; thread < nthread; thread++)
    {
        tempTAA_all("v, u") += tempTAA[thread]("v, u");
        tempTaa_all("v, u") += tempTaa[thread]("v, u");
    }

    Eacvv += tempTAA_all("v,u") * Gamma1_AA("v,u");
    Eacvv += tempTaa_all("v,u") * Gamma1_aa("v,u");

    if(print_ > 0)
    {
        outfile->Printf("\n\n CAVV computation takes %8.8f", ccvaTimer.get());
        outfile->Printf("\n\n Eacvv   =  %8.8f", Eacvv);
    }

    std::vector<ambit::Tensor>  Bm_vQ;
    std::vector<ambit::Tensor>  Bn_eQ;
    std::vector<ambit::Tensor>  Bm_eQ;
    std::vector<ambit::Tensor>  Bn_vQ;

    std::vector<ambit::Tensor>  V_eu;
    std::vector<ambit::Tensor>  T_ev;
    std::vector<ambit::Tensor>  tempTaa_e;
    std::vector<ambit::Tensor>  tempTAA_e;

    ambit::Tensor BmvQ = ints_->three_integral_block(naux, acore_mos_, aactv_mos_);

    std::vector<double>& BmvQ_data = BmvQ.data();

    Timer cavvTimer;
    for(int thread = 0; thread < nthread; thread++)
    {
        Bm_vQ.push_back(ambit::Tensor::build(tensor_type_, "BemQ", {nthree_, active_}));
        Bn_eQ.push_back(ambit::Tensor::build(tensor_type_, "Bf_uQ", {nthree_, virtual_}));
        Bm_eQ.push_back(ambit::Tensor::build(tensor_type_, "Bmq", {nthree_, virtual_}));
        Bn_vQ.push_back(ambit::Tensor::build(tensor_type_, "Bmq", {nthree_, active_}));

        V_eu.push_back(ambit::Tensor::build(tensor_type_, "muJK", {virtual_, active_}));
        T_ev.push_back(ambit::Tensor::build(tensor_type_, "T2",   {virtual_, active_}));

        tempTaa_e.push_back(ambit::Tensor::build(tensor_type_, "TEMPaa", {active_, active_}));
        tempTAA_e.push_back(ambit::Tensor::build(tensor_type_, "TEMPAA", {active_, active_}));
    }
    ambit::Tensor Eta1_aa = Eta1_.block("aa");
    ambit::Tensor Eta1_AA = Eta1_.block("AA");

    #pragma omp parallel for num_threads(num_threads_)
    for(size_t m = 0; m < core_; ++m){
        size_t ma = acore_mos_[m];
        size_t mb = bcore_mos_[m];
        int thread = 0;
        #ifdef _OPENMP
            thread = omp_get_thread_num();
        #endif

        #pragma omp critical
        {
            Bm_eQ[thread] = ints_->three_integral_block_two_index(naux, ma, avirt_mos_);
            Bm_vQ[thread].iterate([&](const std::vector<size_t>& i,double& value){
                value = BmvQ_data[i[0] * core_ * active_ + m * active_ + i[1] ];
            });
        }

        for(size_t n = 0; n < core_; ++n){
        // alpha-aplha
            size_t na = acore_mos_[n];
            size_t nb = bcore_mos_[n];

            #pragma omp critical
            {
                Bn_vQ[thread].iterate([&](const std::vector<size_t>& i,double& value){
                    value = BmvQ_data[i[0] * core_ * active_ + n * active_ + i[1] ];
                });
                Bn_eQ[thread] = ints_->three_integral_block_two_index(naux, na, avirt_mos_);
            }

            // B_{mv}^{Q} * B_{ne}^{Q} - B_{me}^Q * B_{nv}
            V_eu[thread]("e, u") = Bm_vQ[thread]("Q, u") * Bn_eQ[thread]("Q, e");
            V_eu[thread]("e, u")-= Bm_eQ[thread]("Q, e") * Bn_vQ[thread]("Q, u");
            //E = V["efmu"] (1 + Exp(-s * D^{ef}_{mu}) * V^{mv}_{ef} * Denom^{mv}_{ef}
            T_ev[thread].data() = V_eu[thread].data();

            V_eu[thread].iterate([&](const std::vector<size_t>& i,double& value){
                double Exp = Fa_[aactv_mos_[i[1]]] + Fa_[avirt_mos_[i[0]]] - Fa_[ma] - Fa_[na];
                value = value + value * renormalized_exp(Exp);});

            T_ev[thread].iterate([&](const std::vector<size_t>& i,double& value){
                double D = Fa_[ma] + Fa_[na] - Fa_[aactv_mos_[i[1]]] - Fa_[avirt_mos_[i[0]]];
                value = value * renormalized_denominator(D);});

            tempTaa_e[thread]("u,v")+= 0.5 * V_eu[thread]("e,u") * T_ev[thread]("e,v");
            V_eu[thread].zero();
            T_ev[thread].zero();


            //alpha-beta
            //temp["vu"] += V_["vEmN"] * T2_["mNuE"];
            //
            V_eu[thread]("E,u") = Bm_vQ[thread]("Q, u") * Bn_eQ[thread]("Q, E");
            T_ev[thread].data() = V_eu[thread].data();
            V_eu[thread].iterate([&](const std::vector<size_t>& i,double& value){
                double Exp = Fa_[aactv_mos_[i[1]]] + Fb_[bvirt_mos_[i[0]]] - Fa_[ma] - Fb_[nb];
                value = value + value * renormalized_exp(Exp);});

            T_ev[thread].iterate([&](const std::vector<size_t>& i,double& value){
                double D = Fa_[ma] + Fb_[nb] - Fa_[aactv_mos_[i[1]]] - Fb_[bvirt_mos_[i[0]]];
                value = value * renormalized_denominator(D);});

            tempTAA_e[thread]("vu")+=V_eu[thread]("M,v") * T_ev[thread]("M,u");
            tempTaa_e[thread]("vu")+=V_eu[thread]("M,v") * T_ev[thread]("M, u");

            //beta-beta
            V_eu[thread].zero();
            T_ev[thread].zero();
            V_eu[thread]("E,U") = Bm_vQ[thread]("Q, U") * Bn_eQ[thread]("Q,E");
            V_eu[thread]("E,U")-= Bm_eQ[thread]("Q, E") * Bn_vQ[thread]("Q, U");
            T_ev[thread].data() = V_eu[thread].data();

            V_eu[thread].iterate([&](const std::vector<size_t>& i,double& value){
                double Exp = Fb_[mb] + Fb_[nb] - Fb_[bactv_mos_[i[1]]] - Fb_[bvirt_mos_[i[0]]];
                value = value + value * renormalized_exp(Exp);});

            T_ev[thread].iterate([&](const std::vector<size_t>& i,double& value){
                double D = Fb_[mb] + Fb_[nb] - Fb_[bactv_mos_[i[1]]] - Fb_[bvirt_mos_[i[0]]];
                value = value * renormalized_denominator(D);});
            tempTAA_e[thread]("v,u")+= 0.5 * V_eu[thread]("M,v") * T_ev[thread]("M,u");
            V_eu[thread].zero();
            T_ev[thread].zero();
        }
    }

    tempTAA_all = ambit::Tensor::build(tensor_type_, "tempTAA_all", {active_, active_});
    tempTaa_all = ambit::Tensor::build(tensor_type_, "tempTaa_all", {active_, active_});
    for(int thread = 0; thread < nthread; thread++)
    {
        tempTAA_all("u, v") += tempTAA_e[thread]("u,v");
        tempTaa_all("u, v") += tempTaa_e[thread]("u,v");
    }
    Eccva += tempTaa_all("vu") * Eta1_aa("uv");
    Eccva += tempTAA_all("VU") * Eta1_AA("UV");
    if(print_ > 0)
    {
        outfile->Printf("\n\n CCVA takes %8.8f", cavvTimer.get());
    }

    return (Eacvv + Eccva);

}
void THREE_DSRG_MRPT2::relax_reference_once()
{
    /// Time to relax this reference!
        O1_ = BTF_->build(tensor_type_,"OneBody", spin_cases({"gg"}));
        O2_ = BTF_->build(tensor_type_, "TwoBody", spin_cases({"gggg"}));
        BlockedTensor T2all = BTF_->build(tensor_type_, "T2all", spin_cases({"hhpp"}));
        H0_ = BTF_->build(tensor_type_, "ZeroBody", spin_cases({"gg"}));
        H0_.iterate([&](const std::vector<size_t>& i,const std::vector<SpinType>& spin,double& value){
            if(i[0] == i[1]){
                if(spin[0] == AlphaSpin){
                    value = Fa_[i[0]];
                }else{
                    value = Fb_[i[0]];
                }
            }
        });
        Hbar1_ = BTF_->build(tensor_type_, "OneBody", spin_cases({"gg"}));
        Hbar2_ = BTF_->build(tensor_type_, "TwoBody", spin_cases({"gggg"}));

        BlockedTensor Vint = BTF_->build(tensor_type_, "AllV", spin_cases({"gggg"}));
        BlockedTensor ThreeInt = compute_B_minimal(Vint.block_labels());
        Vint["pqrs"] =   ThreeInt["gpr"]*ThreeInt["gqs"];
        Vint["pqrs"] -=  ThreeInt["gps"]*ThreeInt["gqr"];
        Vint["PQRS"] =   ThreeInt["gPR"]*ThreeInt["gQS"];
        Vint["PQRS"] -=  ThreeInt["gPS"]*ThreeInt["gQR"];
        Vint["qPsR"] =   ThreeInt["gPR"]*ThreeInt["gqs"];
        Hbar1_("pq") = F_no_renorm_("pq");
        Hbar1_("PQ") = F_no_renorm_("PQ");

        Hbar2_["pqrs"] = Vint["pqrs"];
        Hbar2_["pQrS"] = Vint["pQrS"];
        Hbar2_["PQRS"] = Vint["PQRS"];

        T2all["ijab"] = Vint["ijab"];
        T2all["IJAB"] = Vint["IJAB"];
        T2all["iJaB"] = Vint["iJaB"];
        T2all.iterate([&](const std::vector<size_t>& i,const std::vector<SpinType>& spin,double& value){
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
        T2all.block("aaaa").zero();
        T2all.block("AAAA").zero();
        T2all.block("aAaA").zero();

        H1_T1_C1(H0_,T1_,0.5,O1_);
        H1_T2_C1(H0_,T2all,0.5,O1_);
        H1_T2_C2(H0_,T2all,0.5,O2_);

        Hbar1_["pq"] += O1_["pq"];
        Hbar1_["PQ"] += O1_["PQ"];
        Hbar2_["pqrs"] += O2_["pqrs"];
        Hbar2_["pQrS"] += O2_["pQrS"];
        Hbar2_["PQRS"] += O2_["PQRS"];

        Hbar1_["pq"] += O1_["qp"];
        Hbar1_["PQ"] += O1_["QP"];
        Hbar2_["pqrs"] += O2_["rspq"];
        Hbar2_["pQrS"] += O2_["rSpQ"];
        Hbar2_["PQRS"] += O2_["RSPQ"];
        if(true)
        {
            double Ecorr = 0.0;
            double Etemp = 0.0;
            std::vector<std::pair<std::string, double> > energy;
            H1_T1_C0(Hbar1_,T1_,1.0,Ecorr);
            energy.push_back({"<[F, A1]>", 2 * (Ecorr - Etemp)});
            Etemp = Ecorr;

            H1_T2_C0(Hbar1_,T2all,1.0,Ecorr);
            energy.push_back({"<[F, A2]>", 2 * (Ecorr - Etemp)});
            Etemp = Ecorr;

            H2_T1_C0(Hbar2_,T1_,1.0,Ecorr);
            energy.push_back({"<[V, A1]>", 2 * (Ecorr - Etemp)});
            Etemp = Ecorr;

            H2_T2_C0(Hbar2_,T2all,1.0,Ecorr);
            energy.push_back({"<[V, A2]>", 2 * (Ecorr - Etemp)});
            Etemp = Ecorr;

            // <[H, A]> = 2 * <[H, T]>
            Ecorr *= 2.0;
            energy.push_back({"DSRG-MRPT2 correlation energy", Ecorr});
            energy.push_back({"DSRG-MRPT2 total energy", Eref_ + Ecorr});

            outfile->Printf("\n\n  ==> DSRG-MRPT2 Energy Summary <==\n");
            for (auto& str_dim : energy){
                outfile->Printf("\n    %-30s = %22.15f",str_dim.first.c_str(),str_dim.second);
            }
        }

        O1_.zero();
        O2_.zero();

        H1_T1_C1(Hbar1_,T1_,1.0,O1_);
        H1_T2_C1(Hbar1_,T2all,1.0,O1_);
        H2_T1_C1(Hbar2_,T1_,1.0,O1_);
        H2_T2_C1(Hbar2_,T2all,1.0,O1_);

        H1_T2_C2(Hbar1_,T2all,1.0,O2_);
        H2_T1_C2(Hbar2_,T1_,1.0,O2_);
        H2_T2_C2(Hbar2_,T2all,1.0,O2_);


        Hbar1_["pq"] += O1_["pq"];
        Hbar1_["pq"] += O1_["qp"];
        Hbar1_["PQ"] += O1_["PQ"];
        Hbar1_["PQ"] += O1_["QP"];
        Hbar2_["pqrs"] += O2_["pqrs"];
        Hbar2_["pqrs"] += O2_["rspq"];
        Hbar2_["pQrS"] += O2_["pQrS"];
        Hbar2_["pQrS"] += O2_["rSpQ"];
        Hbar2_["PQRS"] += O2_["PQRS"];
        Hbar2_["PQRS"] += O2_["RSPQ"];

        de_normal_order();

        double E_relax = relaxed_energy();

        Process::environment.globals["CURRENT ENERGY"] = E_relax;


        // printing
        print_h2("MRDSRG Energy Summary");
        outfile->Printf("\n    %-30s = %22.15f", "DSRG-MRPT2 (fixed)", Hbar0_ + Eref_);
        outfile->Printf("\n    %-30s = %22.15f", "DSRG-MRPT2 (relax)", E_relax);
        outfile->Printf("\n");

}
double THREE_DSRG_MRPT2::relaxed_energy()
{
    // setup for FCISolver
    std::vector<size_t> rdocc = mo_space_info_->get_corr_abs_mo("RESTRICTED_DOCC");
    std::vector<size_t> active = mo_space_info_->get_corr_abs_mo("ACTIVE");
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
    O1_.citerate([&](const std::vector<size_t>& i,const std::vector<SpinType>& spin,const double& value){
        if (spin[0] == AlphaSpin){
            ints_->set_oei(i[0],i[1],value,true);
        }else{
            ints_->set_oei(i[0],i[1],value,false);
        }
    });

    // reference relaxation
    double Erelax = 0.0;
        // diagonalize the Hamiltonian
    FCISolver fcisolver(active_dim,acore_mos_,aactv_mos_,na,nb,multi,options_.get_int("ROOT_SYM"),ints_, mo_space_info_,
                                             options_.get_int("NTRIAL_PER_ROOT"),print_, options_);
    fcisolver.set_max_rdm_level(2);
    fcisolver.set_nroot(options_.get_int("NROOT"));
    fcisolver.set_root(options_.get_int("ROOT"));
    fcisolver.test_rdms(options_.get_bool("TEST_RDMS"));
    fcisolver.set_fci_iterations(options_.get_int("FCI_ITERATIONS"));
    fcisolver.set_collapse_per_root(options_.get_int("DAVIDSON_COLLAPSE_PER_ROOT"));
    fcisolver.set_subspace_per_root(options_.get_int("DAVIDSON_SUBSPACE_PER_ROOT"));
    fcisolver.print_no(false);

    std::shared_ptr<FCIIntegrals> fci_ints = std::make_shared<FCIIntegrals>(ints_, active, rdocc);
    auto na_array = mo_space_info_->get_corr_abs_mo("ACTIVE");
    fcisolver.use_user_integrals_and_restricted_docc(true);
    fci_ints->set_active_integrals(Hbar2_.block("aaaa"), Hbar2_.block("aAaA"),Hbar2_.block("AAAA"));
    std::vector<std::vector<double> > oei_vector;
    if((core_  + mo_space_info_->size("FROZEN_DOCC")) > 0)
    {
        oei_vector = compute_restricted_docc_operator_dsrg();
        fci_ints->set_restricted_one_body_operator(oei_vector[0], oei_vector[1]);
        fci_ints->set_scalar_energy(scalar_energy_fci_);
        fcisolver.set_integral_pointer(fci_ints);
    }
    else{
        std::vector<double> oei_a(active_ * active_);
        std::vector<double> oei_b(active_ * active_);

        for (size_t p = 0; p < active_; ++p){
            size_t pp = active[p];
            for (size_t q = 0; q < active_; ++q){
                size_t qq = active[q];
                size_t idx = active_ * p + q;
                oei_a[idx] = ints_->oei_a(pp,qq);
                oei_b[idx] = ints_->oei_b(pp,qq);
            }
        }
        oei_vector.push_back(oei_a);
        oei_vector.push_back(oei_b);
        fci_ints->set_restricted_one_body_operator(oei_vector[0], oei_vector[1]);
        fci_ints->set_scalar_energy(scalar_energy_fci_);
        fcisolver.set_integral_pointer(fci_ints);
    }

    Erelax = fcisolver.compute_energy();
    return Erelax;

}
void THREE_DSRG_MRPT2::H1_T1_C0(BlockedTensor& H1, BlockedTensor& T1, const double& alpha, double& C0){
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
    time_H1_T1_C0 += timer.get();
}

void THREE_DSRG_MRPT2::H1_T2_C0(BlockedTensor& H1, BlockedTensor& T2, const double& alpha, double& C0){
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
    time_H1_T2_C0 += timer.get();
}

void THREE_DSRG_MRPT2::H2_T1_C0(BlockedTensor& H2, BlockedTensor& T1, const double& alpha, double& C0){
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
    time_H2_T1_C0 += timer.get();
}

void THREE_DSRG_MRPT2::H2_T2_C0(BlockedTensor& H2, BlockedTensor& T2, const double& alpha, double& C0){
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
        temp["uvWxyZ"] += H2["uWyI"] * T2["vIxZ"];      //  aaAaaA from hole
        temp["uvWxyZ"] += H2["uWyI"] * T2["vIxZ"];      //  aaAaaA from hole

        temp["uvWxyZ"] += H2["aWxZ"] * T2["uvay"];      //  aaAaaA from particle
        temp["uvWxyZ"] -= H2["vaxy"] * T2["uWaZ"];      //  aaAaaA from particle
        temp["uvWxyZ"] -= H2["vAxZ"] * T2["uWyA"];      //  aaAaaA from particle
        temp["uvWxyZ"] -= H2["vAxZ"] * T2["uWyA"];      //  aaAaaA from particle
        E += 0.5 * temp["uvWxyZ"] * Lambda3_["xyZuvW"];

        temp = ambit::BlockedTensor::build(tensor_type_,"temp",{"aAAaAA"});
        temp["uVWxYZ"] -= H2["VWIZ"] * T2["uIxY"];      //  aAAaAA from hole
        temp["uVWxYZ"] -= H2["uVxI"] * T2["IWYZ"];      //  aAAaAA from hole
        temp["uVWxYZ"] += H2["uViZ"] * T2["iWxY"];      //  aAAaAA from hole
        temp["uVWxYZ"] += H2["uViZ"] * T2["iWxY"];      //  aAAaAA from hole

        temp["uVWxYZ"] += H2["uAxY"] * T2["VWAZ"];      //  aAAaAA from particle
        temp["uVWxYZ"] -= H2["WAYZ"] * T2["uVxA"];      //  aAAaAA from particle
        temp["uVWxYZ"] -= H2["aWxY"] * T2["uVaZ"];      //  aAAaAA from particle
        temp["uVWxYZ"] -= H2["aWxY"] * T2["uVaZ"];      //  aAAaAA from particle
        E += 0.5 * temp["uVWxYZ"] * Lambda3_["xYZuVW"];
    }

    // multiply prefactor and copy to C0
    E  *= alpha;
    C0 += E;

    if(print_ > 2){
        outfile->Printf("\n    Time for [H2, T2] -> C0 : %12.3f",timer.get());
    }
    time_H2_T2_C0 += timer.get();
}

void THREE_DSRG_MRPT2::H1_T1_C1(BlockedTensor& H1, BlockedTensor& T1, const double& alpha, BlockedTensor& C1){
    Timer timer;

    C1["ip"] += alpha * H1["ap"] * T1["ia"];
    C1["qa"] -= alpha * T1["ia"] * H1["qi"];

    C1["IP"] += alpha * H1["AP"] * T1["IA"];
    C1["QA"] -= alpha * T1["IA"] * H1["QI"];

    if(print_ > 2){
        outfile->Printf("\n    Time for [H1, T1] -> C1 : %12.3f",timer.get());
    }
    time_H1_T1_C1 += timer.get();
}

void THREE_DSRG_MRPT2::H1_T2_C1(BlockedTensor& H1, BlockedTensor& T2, const double& alpha, BlockedTensor& C1){
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
    time_H1_T2_C1 += timer.get();
}

void THREE_DSRG_MRPT2::H2_T1_C1(BlockedTensor& H2, BlockedTensor& T1, const double& alpha, BlockedTensor& C1){
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
    time_H2_T1_C1 += timer.get();
}

void THREE_DSRG_MRPT2::H2_T2_C1(BlockedTensor& H2, BlockedTensor& T2, const double& alpha, BlockedTensor& C1){
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
    time_H2_T2_C1 += timer.get();
}

void THREE_DSRG_MRPT2::H1_T2_C2(BlockedTensor& H1, BlockedTensor& T2, const double& alpha, BlockedTensor& C2){
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

//    // probably not worth doing the following because contracting one index should be fast
//    BlockedTensor temp = ambit::BlockedTensor::build(tensor_type_,"temp",{"hhpg","HHPG"});
//    temp["ijap"] = alpha * T2["ijab"] * H1["bp"];
//    temp["IJAP"] = alpha * T2["IJAB"] * H1["BP"];

//    C2["ijpb"] -= temp["ijbp"]; // use permutation of temp
//    C2["ijap"] += temp["ijap"]; // explicitly evaluate by temp
//    C2["iJpA"] += alpha * T2["iJbA"] * H1["bp"];
//    C2["iJaP"] += alpha * T2["iJaB"] * H1["BP"];
//    C2["IJPB"] -= temp["IJBP"]; // use permutation of temp
//    C2["IJAP"] += temp["IJAP"]; // explicitly evaluate by temp

//    temp = ambit::BlockedTensor::build(tensor_type_,"temp",{"ghpp","GHPP"});
//    temp["qjab"] = alpha * T2["ijab"] * H1["qi"];
//    temp["QJAB"] = alpha * T2["IJAB"] * H1["QI"];

//    C2["qjab"] -= temp["qjab"]; // explicitly evaluate by temp
//    C2["iqab"] += temp["qiab"]; // use permutation of temp
//    C2["qJaB"] -= alpha * T2["iJaB"] * H1["qi"];
//    C2["iQaB"] -= alpha * T2["iJaB"] * H1["QJ"];
//    C2["QJAB"] -= temp["QJAB"]; // explicitly evaluate by temp
//    C2["IQAB"] += temp["QIAB"]; // use permutation of temp

    if(print_ > 2){
        outfile->Printf("\n    Time for [H1, T2] -> C2 : %12.3f",timer.get());
    }
    time_H1_T2_C2 += timer.get();
}

void THREE_DSRG_MRPT2::H2_T1_C2(BlockedTensor& H2, BlockedTensor& T1, const double& alpha, BlockedTensor& C2){
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
    time_H2_T1_C2 += timer.get();
}

void THREE_DSRG_MRPT2::H2_T2_C2(BlockedTensor& H2, BlockedTensor& T2, const double& alpha, BlockedTensor& C2){
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
    time_H2_T2_C2 += timer.get();
}
void THREE_DSRG_MRPT2::de_normal_order()
{
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
    Hbar1_.block("cc").citerate([&](const std::vector<size_t>& i,const double& value){
        if (i[0] == i[1]) scalar1 -= value;
    });
    Hbar1_.block("CC").citerate([&](const std::vector<size_t>& i,const double& value){
        if (i[0] == i[1]) scalar1 -= value;
    });
    scalar1 -= Hbar1_["vu"] * Gamma1_["uv"];
    scalar1 -= Hbar1_["VU"] * Gamma1_["UV"];

    // scalar from Hbar2
    double scalar2 = 0.0;
    scalar2 -= 0.25 * Hbar2_["xyuv"] * Lambda2_["uvxy"];
    scalar2 -= 0.25 * Hbar2_["XYUV"] * Lambda2_["UVXY"];
    scalar2 -= Hbar2_["xYuV"] * Lambda2_["uVxY"];
    Hbar2_.block("cccc").citerate([&](const std::vector<size_t>& i,const double& value){
        if ((i[0] == i[2]) && (i[1] == i[3])) scalar2 += 0.5 * value;
    });
    Hbar2_.block("cCcC").citerate([&](const std::vector<size_t>& i,const double& value){
        if ((i[0] == i[2]) && (i[1] == i[3])) scalar2 += value;
    });
    Hbar2_.block("CCCC").citerate([&](const std::vector<size_t>& i,const double& value){
        if ((i[0] == i[2]) && (i[1] == i[3])) scalar2 += 0.5 * value;
    });

    O1_.zero();
    O1_["pq"] += Hbar2_["puqv"] * Gamma1_["vu"];
    O1_["pq"] += Hbar2_["pUqV"] * Gamma1_["VU"];
    O1_["PQ"] += Hbar2_["uPvQ"] * Gamma1_["vu"];
    O1_["PQ"] += Hbar2_["PUQV"] * Gamma1_["VU"];
    O1_.block("cc").citerate([&](const std::vector<size_t>& i,const double& value){
        if (i[0] == i[1]) scalar2 += value;
    });
    O1_.block("CC").citerate([&](const std::vector<size_t>& i,const double& value){
        if (i[0] == i[1]) scalar2 += value;
    });
    scalar2 += 0.5 * Gamma1_["uv"] * Hbar2_["vyux"] * Gamma1_["xy"];
    scalar2 += 0.5 * Gamma1_["UV"] * Hbar2_["VYUX"] * Gamma1_["XY"];
    scalar2 += Gamma1_["uv"] * Hbar2_["vYuX"] * Gamma1_["XY"];

    double scalar = scalar0 + scalar1 + scalar2;
    outfile->Printf("  Done. Timing %10.3f s", t_scalar.get());

    // compute one-body term
    Timer t_one;
    str = "Computing the one-body term ...";
    outfile->Printf("\n    %-35s", str.c_str());
    O1_.scale(-1.0);
    O1_["pq"] += Hbar1_["pq"];
    O1_["PQ"] += Hbar1_["PQ"];
    BlockedTensor temp = BTF_->build(tensor_type_,"temp",spin_cases({"cc"}));
    temp.iterate([&](const std::vector<size_t>& i,const std::vector<SpinType>&,double& value){
        if (i[0] == i[1]) value = 1.0;
    });
    O1_["pq"] -= Hbar2_["pmqn"] * temp["nm"];
    O1_["pq"] -= Hbar2_["pMqN"] * temp["NM"];
    O1_["PQ"] -= Hbar2_["mPnQ"] * temp["nm"];
    O1_["PQ"] -= Hbar2_["PMQN"] * temp["NM"];
    outfile->Printf("  Done. Timing %10.3f s", t_one.get());

    ints_->set_scalar(scalar);

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
    O1_.block("cc").citerate([&](const std::vector<size_t>& i,const double& value){
        if (i[0] == i[1]) Etest1 += value;
    });
    O1_.block("CC").citerate([&](const std::vector<size_t>& i,const double& value){
        if (i[0] == i[1]) Etest1 += value;
    });
    Etest1 += O1_["uv"] * Gamma1_["vu"];
    Etest1 += O1_["UV"] * Gamma1_["VU"];

    double Etest2 = 0.0;
    Hbar2_.block("cccc").citerate([&](const std::vector<size_t>& i,const double& value){
        if ((i[0] == i[2]) && (i[1] == i[3])) Etest2 += 0.5 * value;
    });
    Hbar2_.block("cCcC").citerate([&](const std::vector<size_t>& i,const double& value){
        if ((i[0] == i[2]) && (i[1] == i[3])) Etest2 += value;
    });
    Hbar2_.block("CCCC").citerate([&](const std::vector<size_t>& i,const double& value){
        if ((i[0] == i[2]) && (i[1] == i[3])) Etest2 += 0.5 * value;
    });

    Etest2 += Hbar2_["munv"] * temp["nm"] * Gamma1_["vu"];
    Etest2 += Hbar2_["uMvN"] * temp["NM"] * Gamma1_["vu"];
    Etest2 += Hbar2_["mUnV"] * temp["nm"] * Gamma1_["VU"];
    Etest2 += Hbar2_["MUNV"] * temp["NM"] * Gamma1_["VU"];

    Etest2 += 0.5 * Gamma1_["vu"] * Hbar2_["uxvy"] * Gamma1_["yx"];
    Etest2 += 0.5 * Gamma1_["VU"] * Hbar2_["UXVY"] * Gamma1_["YX"];
    Etest2 += Gamma1_["vu"] * Hbar2_["uXvY"] * Gamma1_["YX"];

    Etest2 += 0.25 * Hbar2_["uvxy"] * Lambda2_["xyuv"];
    Etest2 += 0.25 * Hbar2_["UVXY"] * Lambda2_["XYUV"];
    Etest2 += Hbar2_["uVxY"] * Lambda2_["xYuV"];

    Etest += Etest1 + Etest2;
    outfile->Printf("\n    %-30s = %22.15f", "One-Body Energy (after)", Etest1);
    outfile->Printf("\n    %-30s = %22.15f", "Two-Body Energy (after)", Etest2);
    outfile->Printf("\n    %-30s = %22.15f", "Total Energy (after)", Etest);
    outfile->Printf("\n    %-30s = %22.15f", "Total Energy (before)", Eref_ + Hbar0_);

    if(fabs(Etest - Eref_ - Hbar0_) > 100.0 * options_.get_double("E_CONVERGENCE")){
        throw PSIEXCEPTION("De-normal-odering failed.");
    }
}
std::vector<std::vector<double> > THREE_DSRG_MRPT2::compute_restricted_docc_operator_dsrg()
{
    size_t nfomo = mo_space_info_->size("RESTRICTED_DOCC");
    size_t na = mo_space_info_->size("ACTIVE");
    auto fomo_to_mo = mo_space_info_->get_corr_abs_mo("RESTRICTED_DOCC");
    auto cmo_to_mo = mo_space_info_->get_corr_abs_mo("ACTIVE");
    std::vector<double> oei_a(na * na);
    std::vector<double> oei_b(na * na);
    std::vector<double> tei_rdocc_aa;
    std::vector<double> tei_rdocc_ab;
    std::vector<double> tei_rdocc_bb;

    std::vector<double> tei_gh_aa;
    std::vector<double> tei_gh_ab;
    std::vector<double> tei_gh_bb;

    ambit::Tensor rdocc_aa = Hbar2_.block("cccc");
    ambit::Tensor rdocc_ab = Hbar2_.block("cCcC");
    ambit::Tensor rdocc_bb = Hbar2_.block("CCCC");
    tei_rdocc_aa = rdocc_aa.data();
    tei_rdocc_ab = rdocc_ab.data();
    tei_rdocc_bb = rdocc_bb.data();

    ambit::Tensor gh_aa  = Hbar2_.block("acac");
    ambit::Tensor gh_ab  = Hbar2_.block("aCaC");
    ambit::Tensor gh_bb  = Hbar2_.block("ACAC");

    tei_gh_aa  = gh_aa.data();
    tei_gh_ab  = gh_ab.data();
    tei_gh_bb  = gh_bb.data();
    O1_.citerate([&](const std::vector<size_t>& i,const std::vector<SpinType>& spin,const double& value){
        if (spin[0] == AlphaSpin){
            ints_->set_oei(i[0],i[1],value,true);
        }else{
            ints_->set_oei(i[0],i[1],value,false);
        }
    });

    // Compute the scalar contribution to the energy that comes from
    // the restricted occupied orbitals
    double scalar_energy = ints_->scalar();
    for (size_t i = 0; i < nfomo; ++i){
        size_t ii = fomo_to_mo[i];
        scalar_energy += ints_->oei_a(ii,ii);
        scalar_energy += ints_->oei_b(ii,ii);
        for (size_t j = 0; j < nfomo; ++j){
            size_t index = nfomo*nfomo*nfomo*i + nfomo*nfomo*j + nfomo*i + j;
            scalar_energy += 0.5 * tei_rdocc_aa[index];
            scalar_energy += 1.0 * tei_rdocc_ab[index];
            scalar_energy += 0.5 * tei_rdocc_bb[index];
        }
    }
    scalar_energy_fci_ = scalar_energy;


    for (size_t p = 0; p < na; ++p){
        size_t pp = cmo_to_mo[p];
        for (size_t q = 0; q < na; ++q){
            size_t qq = cmo_to_mo[q];
            size_t idx = na * p + q;
            oei_a[idx] = ints_->oei_a(pp,qq);
            oei_b[idx] = ints_->oei_b(pp,qq);
            // Compute the one-body contribution to the energy that comes from
            // the restricted occupied orbitals
            for (size_t f = 0; f < nfomo; ++f){
                size_t index  = nfomo * na * nfomo * p + na * nfomo * f + nfomo * q + f;
                oei_a[idx] += tei_gh_aa[index];
                oei_a[idx] += tei_gh_ab[index];
                oei_b[idx] += tei_gh_bb[index];
                oei_b[idx] += tei_gh_ab[index]; // TODO check these factors 0.5
            }
        }
    }
    std::vector<std::vector<double> > return_oei(2);
    return_oei[0] = oei_a;
    return_oei[1] = oei_b;

    return return_oei;

}
}} // End Namespaces
