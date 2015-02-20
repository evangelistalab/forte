#include <numeric>

#include <libpsio/psio.hpp>
#include <libpsio/psio.h>
#include <libmints/molecule.h>

#include "dsrg_mrpt2.h"

namespace psi{ namespace libadaptive{

DSRG_MRPT2::DSRG_MRPT2(Reference reference, boost::shared_ptr<Wavefunction> wfn, Options &options, ExplorerIntegrals* ints)
    : Wavefunction(options,_default_psio_lib_), reference_(reference), ints_(ints)
{
    // Copy the wavefunction information
    copy(wfn);
    Tensor::set_print_level(debug_);
    startup();
    print_summary();
}

DSRG_MRPT2::~DSRG_MRPT2()
{
    cleanup();
}

void DSRG_MRPT2::startup()
{
    Eref_ = reference_.get_Eref();
    outfile->Printf("\n  Reference Energy = %.15f", Eref_);

    double frozen_core_energy = ints_->frozen_core_energy();

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



    BlockedTensor::add_primitive_mo_space("c","mn",acore_mos,Alpha);
    BlockedTensor::add_primitive_mo_space("C","MN",bcore_mos,Beta);

    BlockedTensor::add_primitive_mo_space("a","uvwxyz",aactv_mos,Alpha);
    BlockedTensor::add_primitive_mo_space("A","UVWXYZ",bactv_mos,Beta);

    BlockedTensor::add_primitive_mo_space("v","ef",avirt_mos,Alpha);
    BlockedTensor::add_primitive_mo_space("V","EF",bvirt_mos,Beta);

    BlockedTensor::add_composite_mo_space("h","ijkl",{"c","a"});
    BlockedTensor::add_composite_mo_space("H","IJKL",{"C","A"});

    BlockedTensor::add_composite_mo_space("p","abcd",{"a","v"});
    BlockedTensor::add_composite_mo_space("P","ABCD",{"A","V"});

    BlockedTensor::add_composite_mo_space("g","pqrs",{"c","a","v"});
    BlockedTensor::add_composite_mo_space("G","PQRS",{"C","A","V"});


    if(options_.get_str("INT_TYPE")=="CHOLESKY")
    {
        size_t nL = ints_->nL();
        std::vector<size_t> nauxpi(nL);
        std::iota(nauxpi.begin(), nauxpi.end(),0);
        BlockedTensor::add_primitive_mo_space("d","g",nauxpi,Alpha);
        //BlockedTensor::add_composite_mo_space("o","@#$",{"d","c",;

    }
    else if(options_.get_str("INT_TYPE")=="DF")
    {
        size_t nDF = ints_->naux();
        std::vector<size_t> nauxpi(nDF);
        std::iota(nauxpi.begin(), nauxpi.end(),0);
        BlockedTensor::add_primitive_mo_space("d","g",nauxpi,Alpha);
    }
    size_t ndf = 10;
    std::vector<size_t> ndfpi(ndf);
    std::iota(ndfpi.begin(),ndfpi.end(),0);
    BlockedTensor::add_primitive_mo_space("d","g",ndfpi,Alpha);

    H.resize_spin_components("H","gg");
    V.resize_spin_components("V","gggg");

    Gamma1.resize_spin_components("Gamma1","hh");
    Eta1.resize_spin_components("Eta1","pp");
    Lambda2.resize_spin_components("Lambda2","aaaa");
    Lambda3.resize_spin_components("Lambda3","aaaaaa");
    F.resize_spin_components("Fock","gg");
    Delta1.resize_spin_components("Delta1","hp");
    Delta2.resize_spin_components("Delta2","hhpp");
    RDelta1.resize_spin_components("RDelta1","hp");
    RDelta2.resize_spin_components("RDelta2","hhpp");
    T1.resize_spin_components("T1 Amplitudes","hp");
    T2.resize_spin_components("T2 Amplitudes","hhpp");

    // Fill in the one-electron operator (H)
    H.fill_one_electron_spin([&](size_t p,MOSetSpinType sp,size_t q,MOSetSpinType sq){
        return (sp == Alpha) ? ints_->oei_a(p,q) : ints_->oei_b(p,q);
    });

    Tensor& Gamma1_cc = *Gamma1.block("cc");
    Tensor& Gamma1_aa = *Gamma1.block("aa");
    Tensor& Gamma1_CC = *Gamma1.block("CC");
    Tensor& Gamma1_AA = *Gamma1.block("AA");

    Tensor& Eta1_aa = *Eta1.block("aa");
    Tensor& Eta1_vv = *Eta1.block("vv");
    Tensor& Eta1_AA = *Eta1.block("AA");
    Tensor& Eta1_VV = *Eta1.block("VV");


    for (Tensor::iterator it = Gamma1_cc.begin(),endit = Gamma1_cc.end(); it != endit; ++it){
        std::vector<size_t>& i = it.address();
        *it = i[0] == i[1] ? 1.0 : 0.0;
    }
    for (Tensor::iterator it = Gamma1_CC.begin(),endit = Gamma1_CC.end(); it != endit; ++it){
        std::vector<size_t>& i = it.address();
        *it = i[0] == i[1] ? 1.0 : 0.0;
    }

    for (Tensor::iterator it = Eta1_aa.begin(),endit = Eta1_aa.end(); it != endit; ++it){
        std::vector<size_t>& i = it.address();
        *it = i[0] == i[1] ? 1.0 : 0.0;
    }
    for (Tensor::iterator it = Eta1_AA.begin(),endit = Eta1_AA.end(); it != endit; ++it){
        std::vector<size_t>& i = it.address();
        *it = i[0] == i[1] ? 1.0 : 0.0;
    }

    for (Tensor::iterator it = Eta1_vv.begin(),endit = Eta1_vv.end(); it != endit; ++it){
        std::vector<size_t>& i = it.address();
        *it = i[0] == i[1] ? 1.0 : 0.0;
    }
    for (Tensor::iterator it = Eta1_VV.begin(),endit = Eta1_VV.end(); it != endit; ++it){
        std::vector<size_t>& i = it.address();
        *it = i[0] == i[1] ? 1.0 : 0.0;
    }

    Gamma1_aa["pq"] = (*reference_.L1a())["pq"];
    Gamma1_AA["pq"] = (*reference_.L1b())["pq"];

    Eta1_aa["pq"] -= (*reference_.L1a())["pq"];
    Eta1_AA["pq"] -= (*reference_.L1b())["pq"];

    // Fill in the two-electron operator (V)
    V.fill_two_electron_spin([&](size_t p,MOSetSpinType sp,
                                           size_t q,MOSetSpinType sq,
                                           size_t r,MOSetSpinType sr,
                                           size_t s,MOSetSpinType ss){
        if ((sp == Alpha) and (sq == Alpha)) return ints_->aptei_aa(p,q,r,s);
        if ((sp == Alpha) and (sq == Beta) ) return ints_->aptei_ab(p,q,r,s);
        if ((sp == Beta)  and (sq == Beta) ) return ints_->aptei_bb(p,q,r,s);
        return 0.0;
    });

    // Form the Fock matrix
    F["pq"]  = H["pq"];
    F["pq"] += V["prqs"] * Gamma1["sr"];
    F["pq"] += V["pRqS"] * Gamma1["SR"];

    F["PQ"] += H["PQ"];
    F["PQ"] += V["rPsQ"] * Gamma1["sr"];
    F["PQ"] += V["PRQS"] * Gamma1["SR"];
  
    F.print();

    Tensor& Fa_cc = *F.block("cc");
    Tensor& Fa_aa = *F.block("aa");
    Tensor& Fa_vv = *F.block("vv");
    Tensor& Fb_CC = *F.block("CC");
    Tensor& Fb_AA = *F.block("AA");
    Tensor& Fb_VV = *F.block("VV");

    size_t ncmo_ = ints_->ncmo();
    std::vector<double> Fa(ncmo_);
    std::vector<double> Fb(ncmo_);
    for (Tensor::iterator it = Fa_cc.begin(),endit = Fa_cc.end(); it != endit; ++it){
        std::vector<size_t>& i = it.address();
        if(i[0] == i[1]){
            Fa[acore_mos[i[0]]] = *it;
        }
    }
    for (Tensor::iterator it = Fa_aa.begin(),endit = Fa_aa.end(); it != endit; ++it){
        std::vector<size_t>& i = it.address();
        if(i[0] == i[1]){
            Fa[aactv_mos[i[0]]] = *it;
        }
    }
    for (Tensor::iterator it = Fa_vv.begin(),endit = Fa_vv.end(); it != endit; ++it){
        std::vector<size_t>& i = it.address();
        if(i[0] == i[1]){
            Fa[avirt_mos[i[0]]] = *it;
        }
    }
    for (Tensor::iterator it = Fb_CC.begin(),endit = Fb_CC.end(); it != endit; ++it){
        std::vector<size_t>& i = it.address();
        if(i[0] == i[1]){
            Fb[bcore_mos[i[0]]] = *it;
        }
    }
    for (Tensor::iterator it = Fb_AA.begin(),endit = Fb_AA.end(); it != endit; ++it){
        std::vector<size_t>& i = it.address();
        if(i[0] == i[1]){
            Fb[bactv_mos[i[0]]] = *it;
        }
    }
    for (Tensor::iterator it = Fb_VV.begin(),endit = Fb_VV.end(); it != endit; ++it){
        std::vector<size_t>& i = it.address();
        if(i[0] == i[1]){
            Fb[bvirt_mos[i[0]]] = *it;
        }
    }

    Delta1.fill_one_electron_spin([&](size_t p,MOSetSpinType sp,size_t q,MOSetSpinType sq){
        if (sp  == Alpha){
            return Fa[p] - Fa[q];
        }else if (sp  == Beta){
            return Fb[p] - Fb[q];
        }
        return 0.0;
    });

    RDelta1.fill_one_electron_spin([&](size_t p,MOSetSpinType sp,size_t q,MOSetSpinType sq){
        if (sp  == Alpha){
            return renormalized_denominator(Fa[p] - Fa[q]);
        }else if (sp  == Beta){
            return renormalized_denominator(Fb[p] - Fb[q]);
        }
        return 0.0;
    });

    RDelta2.fill_two_electron_spin([&](size_t p,MOSetSpinType sp,
                                      size_t q,MOSetSpinType sq,
                                      size_t r,MOSetSpinType sr,
                                      size_t s,MOSetSpinType ss){
        if ((sp == Alpha) and (sq == Alpha)){
            return renormalized_denominator(Fa[p] + Fa[q] - Fa[r] - Fa[s]);
        }else if ((sp == Alpha) and (sq == Beta) ){
            return renormalized_denominator(Fa[p] + Fb[q] - Fa[r] - Fb[s]);
        }else if ((sp == Beta)  and (sq == Beta) ){
            return renormalized_denominator(Fb[p] + Fb[q] - Fb[r] - Fb[s]);
        }
        return 0.0;
    });

    // Fill out Lambda2 and Lambda3
    Tensor& Lambda2_aa = *Lambda2.block("aaaa");
    Tensor& Lambda2_aA = *Lambda2.block("aAaA");
    Tensor& Lambda2_AA = *Lambda2.block("AAAA");
    Lambda2_aa["pqrs"] = (*reference_.L2aa())["pqrs"];
    Lambda2_aA["pqrs"] = (*reference_.L2ab())["pqrs"];
    Lambda2_AA["pqrs"] = (*reference_.L2bb())["pqrs"];

    // TODO Lambda3
    Tensor& Lambda3_aaa = *Lambda3.block("aaaaaa");
    Tensor& Lambda3_aaA = *Lambda3.block("aaAaaA");
    Tensor& Lambda3_aAA = *Lambda3.block("aAAaAA");
    Tensor& Lambda3_AAA = *Lambda3.block("AAAAAA");
    Lambda3_aaa["pqrstu"] = (*reference_.L3aaa())["pqrstu"];
    Lambda3_aaA["pqrstu"] = (*reference_.L3aab())["pqrstu"];
    Lambda3_aAA["pqrstu"] = (*reference_.L3abb())["pqrstu"];
    Lambda3_AAA["pqrstu"] = (*reference_.L3bbb())["pqrstu"];
    Lambda3.print();

    // Prepare exponential tensors for effective Fock matrix and integrals
    RExp1.resize_spin_components("RExp1","hp");
    RExp2.resize_spin_components("RExp2","hhpp");

    RExp2.fill_two_electron_spin([&](size_t p,MOSetSpinType sp,
                                      size_t q,MOSetSpinType sq,
                                      size_t r,MOSetSpinType sr,
                                      size_t s,MOSetSpinType ss){
        if ((sp == Alpha) and (sq == Alpha)){
            return renormalized_exp(Fa[p] + Fa[q] - Fa[r] - Fa[s]);
        }else if ((sp == Alpha) and (sq == Beta) ){
            return renormalized_exp(Fa[p] + Fb[q] - Fa[r] - Fb[s]);
        }else if ((sp == Beta)  and (sq == Beta) ){
            return renormalized_exp(Fb[p] + Fb[q] - Fb[r] - Fb[s]);
        }
        return 0.0;
    });

    RExp1.fill_one_electron_spin([&](size_t p,MOSetSpinType sp,size_t q,MOSetSpinType sq){
        if (sp  == Alpha){
            return renormalized_exp(Fa[p] - Fa[q]);
        }else if (sp  == Beta){
            return renormalized_exp(Fb[p] - Fb[q]);
        }
        return 0.0;
    });
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

double DSRG_MRPT2::compute_energy()
{
    // Compute reference
//    double E0 = 0.5 * BlockedTensor::dot(H["ij"],Gamma1["ij"]);
//    E0 += 0.5 * BlockedTensor::dot(F["ij"],Gamma1["ij"]);
//    E0 += 0.5 * BlockedTensor::dot(H["IJ"],Gamma1["IJ"]);
//    E0 += 0.5 * BlockedTensor::dot(F["IJ"],Gamma1["IJ"]);

//    E0 += 0.25 * BlockedTensor::dot(V["uvxy"],Lambda2["uvxy"]);
//    E0 += 0.25 * BlockedTensor::dot(V["UVXY"],Lambda2["UVXY"]);
//    E0 += BlockedTensor::dot(V["uVxY"],Lambda2["uVxY"]);

    // Compute T2 and T1
    compute_t2();
    compute_t1();

    // Compute effective integrals
    renormalize_V();
    renormalize_F();

    // Compute DSRG-MRPT2 correlation energy
    double Ecorr = 0.0;
    Ecorr += E_FT1();

    return Ecorr + Eref_;
}

void DSRG_MRPT2::compute_t2()
{

    T2["ijab"] = V["ijab"] % RDelta2["ijab"];
    T2["iJaB"] = V["iJaB"] % RDelta2["iJaB"];
    T2["IJAB"] = V["IJAB"] % RDelta2["IJAB"];

    // zero internal amplitudes
    T2.block("aaaa")->zero();
    T2.block("aAaA")->zero();
    T2.block("AAAA")->zero();

    // norm and maximum of T2 amplitudes
    T2norm = 0.0; T2max = 0.0;
    std::vector<std::string> first; first.push_back("a"); first.push_back("c");
    std::vector<std::string> second; second.push_back("a"); second.push_back("c");
    std::vector<std::string> third; third.push_back("a"); third.push_back("v");
    std::vector<std::string> fourth; fourth.push_back("a"); fourth.push_back("v");
    for(const std::string& x1: first){
        for(const std::string& x2: second){
            for(const std::string& x3: third){
                for(const std::string& x4: fourth){
                    std::string index (x1 + x2 + x3 + x4);
                    T2norm += pow(T2.block(index)->norm(), 2.0);
                    double max = T2.block(index)->max_abs_vec()[0];
                    T2max = T2max > max ? T2max : max;
                }
            }
        }
    }
    second.clear(); second.push_back("A"); second.push_back("C");
    fourth.clear(); fourth.push_back("A"); fourth.push_back("V");
    for(const std::string& x1: first){
        for(const std::string& x2: second){
            for(const std::string& x3: third){
                for(const std::string& x4: fourth){
                    std::string index (x1 + x2 + x3 + x4);
                    T2norm += 4 * pow(T2.block(index)->norm(), 2.0);
                    double max = T2.block(index)->max_abs_vec()[0];
                    T2max = T2max > max ? T2max : max;
                }
            }
        }
    }
    first.clear(); first.push_back("A"); first.push_back("C");
    third.clear(); third.push_back("A"); third.push_back("V");
    for(const std::string& x1: first){
        for(const std::string& x2: second){
            for(const std::string& x3: third){
                for(const std::string& x4: fourth){
                    std::string index (x1 + x2 + x3 + x4);
                    T2norm += pow(T2.block(index)->norm(), 2.0);
                    double max = T2.block(index)->max_abs_vec()[0];
                    T2max = T2max > max ? T2max : max;
                }
            }
        }
    }
    T2norm = sqrt(T2norm);
    outfile->Printf("\n T2 norm: \t %20.15f", T2norm);
    outfile->Printf("\n T2 max: \t %20.15f", T2max);
}
void DSRG_MRPT2::compute_t1()
{
   //A temporary tensor to use for the building of T1
   //Francesco's library does not handle repeating indices between 3 different terms, so need to form an intermediate
   //via a pointwise multiplcation
   BlockedTensor temp;
   temp.resize_spin_components("temp","aa");
   temp["xu"] = Gamma1["xu"] % Delta1["xu"];
   temp["XU"] = Gamma1["XU"] % Delta1["XU"];

   //Form the T1 amplitudes
   //Note:  The equations are changed slightly from York's equations.
   //Tensor libary does not handle beta alpha beta alpha, only alpha beta alpha beta.
   //Did some permuting to get the correct format

   T1["ia"]  = F["ia"];
   T1["ia"] += temp["xu"] * T2["iuax"];
   T1["ia"] += temp["XU"] * T2["iUaX"];

   T1["ia"]  = T1["ia"] % RDelta1["ia"];

   T1["IA"]  = F["IA"];
   T1["IA"] += temp["xu"] * T2["uIxA"];
   T1["IA"] += temp["XU"] * T2["IUAX"];
   T1["IA"]  = T1["IA"] % RDelta1["IA"];

   T1.block("AA")->zero();
   T1.block("aa")->zero();

//   T1.print();
//   T1.print_norm_of_blocks();

   T1norm = T1.norm(); T1max = 0.0;
   std::vector<std::string> first; first.push_back("a"); first.push_back("c");
   std::vector<std::string> second; second.push_back("a"); second.push_back("v");
   for(const std::string& x1: first){
       for(const std::string& x2: second){
           std::string index (x1 + x2);
           double max = T1.block(index)->max_abs_vec()[0];
           T1max = T1max > max ? T1max : max;
       }
   }
   first.clear(); first.push_back("A"); first.push_back("C");
   second.clear(); second.push_back("A"); second.push_back("V");
   for(const std::string& x1: first){
       for(const std::string& x2: second){
           std::string index (x1 + x2);
           double max = T1.block(index)->max_abs_vec()[0];
           T1max = T1max > max ? T1max : max;
       }
   }
   outfile->Printf("\n T1 norm: \t %20.15f", T1norm);
   outfile->Printf("\n T1 max: \t %20.15f", T1max);
}

void DSRG_MRPT2::renormalize_V()
{
    V["ijab"] += V["ijab"] % RExp2["ijab"];
    V["iJaB"] += V["iJaB"] % RExp2["iJaB"];
    V["IJAB"] += V["IJAB"] % RExp2["IJAB"];

    V["abij"] += V["abij"] % RExp2["ijab"];
    V["aBiJ"] += V["aBiJ"] % RExp2["iJaB"];
    V["ABIJ"] += V["ABIJ"] % RExp2["IJAB"];
}

void DSRG_MRPT2::renormalize_F()
{
    BlockedTensor temp_aa;
    temp_aa.resize_spin_components("temp_aa","aa");
    temp_aa["xu"] = Gamma1["xu"] % Delta1["xu"];
    temp_aa["XU"] = Gamma1["XU"] % Delta1["XU"];

    BlockedTensor temp_hp;
    temp_hp.resize_spin_components("temp_hp","hp");

    temp_hp["ia"] += temp_aa["xu"] * T2["iuax"];
    temp_hp["ia"] += temp_aa["XU"] * T2["iUaX"];
    F["ia"] += F["ia"] % RExp1["ia"];
    F["ia"] += temp_hp["ia"] % RExp1["ia"];

    temp_hp["ai"] += temp_aa["xu"] * T2["auix"];
    temp_hp["ai"] += temp_aa["XU"] * T2["aUiX"];
    F["ai"] += F["ai"] % RExp1["ia"];
    F["ai"] += temp_hp["ai"] % RExp1["ia"];

    temp_hp["IA"] += temp_aa["xu"] * T2["uIxA"];
    temp_hp["IA"] += temp_aa["XU"] * T2["IUAX"];
    F["IA"] += F["IA"] % RExp1["IA"];
    F["IA"] += temp_hp["IA"] % RExp1["IA"];

    temp_hp["AI"] += temp_aa["xu"] * T2["uAxI"];
    temp_hp["AI"] += temp_aa["XU"] * T2["AUIX"];
    F["AI"] += F["AI"] % RExp1["IA"];
    F["AI"] += temp_hp["AI"] % RExp1["IA"];

//    F.print();  // The actv-actv block is different but it should not matter.
}

double DSRG_MRPT2::E_FT1()
{
    double E = 0.0;
    BlockedTensor temp1;
    BlockedTensor temp2;
    temp1.resize_spin_components("temp1","hp");
    temp2.resize_spin_components("temp2","hp");

//    temp1.zero();
    temp1["ib"] += T1["ia"] * Eta1["ab"];
    temp1["IB"] += T1["IA"] * Eta1["AB"];
    temp2["jb"] += temp1["ib"] * Gamma1["ji"];
    temp2["JB"] += temp1["IB"] * Gamma1["JI"];

    E += BlockedTensor::dot(temp2["jb"], F["bj"]);
    E += BlockedTensor::dot(temp2["JB"], F["BJ"]);

//    Gamma1.print();
//    T1.print();
//    temp1.print();
//    temp2.print();
//    F.print();

    outfile->Printf("\n  E([F, T1]) %18c = %22.15lf", ' ', E);
    return E;
}

}} // End Namespaces
