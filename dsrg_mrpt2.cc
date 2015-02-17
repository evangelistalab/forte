#include "dsrg_mrpt2.h"

#include "libmints/molecule.h"

#include <libpsio/psio.h>
#include <libpsio/psio.hpp>

namespace psi{ namespace libadaptive{

DSRG_MRPT2::DSRG_MRPT2(boost::shared_ptr<Wavefunction> wfn, Options &options, ExplorerIntegrals* ints)
    : Wavefunction(options,_default_psio_lib_), ints_(ints)
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
//    // Number of Electrons and Orbitals
////    boost::shared_ptr<Molecule> molecule = Process::environment.molecule();
//    int natom = molecule_->natom();
//    size_t nelec = 0;
//    for(int i = 0; i<natom; ++i){
//        nelec += molecule_->fZ(i);
//    }
//    int charge = molecule_->molecular_charge();
//    if(options_["CHARGE"].has_changed()){
//        charge = options_.get_int("CHARGE");
//    }
//    int ms_ = options_.get_int("MS");
//    if(ms_ < 0){
//        outfile->Printf("\n  MS must be no less than 0.");
//        exit(1);
//    }
//    int multi_ = molecule_->multiplicity();
//    if(options_["MULTI"].has_changed()){
//        multi_ = options_.get_int("MULTI");
//    }
//    if(multi_ < 1){
//        outfile->Printf("\n  MULTI must be no less than 1. Check Multiplicity!");
//        exit(1);
//    }
//    int nalfa_ = (nelec - charge + ms_ * (ms_ + 1)) / 2;
//    int nbeta_ = (nelec - charge - ms_ * (ms_ + 1)) / 2;
//    if(nalfa_ < 0 || nbeta_ < 0){
//        outfile->Printf("\n  Check the Charge and Multiplicity! \n");
//        exit(1);
//    }
//    if(nalfa_ - nc_ - nfrzc_ > na_){
//        outfile->Printf("\n  Not enough active orbitals to arrange electrons!");
//        outfile->Printf("\n  Check core and active orbitals! \n");
//        exit(1);
//    }

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

    H.resize_spin_components("H","gg");
    V.resize_spin_components("V","gggg");

    Gamma1.resize_spin_components("Gamma1","hh");
    Eta1.resize_spin_components("Eta1","pp");
    Lambda2.resize_spin_components("Lambda2","aaaa");
    Lambda3.resize_spin_components("Lambda3","aaaaaa");
    F.resize_spin_components("Fock","gg");
    Delta1.resize_spin_components("Delta1","hp");
    Delta2.resize_spin_components("Delta2","hhpp");

    // Fill in the one-electron operator (H)
    H.fill_one_electron_spin([&](size_t p,MOSetSpinType sp,size_t q,MOSetSpinType sq){
        return (sp == Alpha) ? ints_->oei_a(p,q) : ints_->oei_b(p,q);
    });

    Gamma1.fill_one_electron_spin([&](size_t p,MOSetSpinType sp,size_t q,MOSetSpinType sq){
        return (p == q ? 1.0 : 0.0);
    });

    Eta1.fill_one_electron_spin([&](size_t p,MOSetSpinType sp,size_t q,MOSetSpinType sq){
        return (p == q ? 1.0 : 0.0);
    });

    // DC1 dc1 = fci->get_lambda_1({3,4,5});  // Return lambda_1 in the range 3-5
    // DC2 dc2 = fci->get_lambda_2({3,4,5});  // Return lambda_1 in the range 3-5
    // dsrg_mrpt2->set_lambda_1(dc1);
    // dsrg_mrpt2->set_lambda_2(dc2);


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
//    if (print_ > 2){
//        G1.print();
//        CG1.print();
//        H.print();
//        F.print();
//    }

//    Tensor& Fa_oo = *F.block("oo");
//    Tensor& Fa_vv = *F.block("vv");
//    Tensor& Fb_OO = *F.block("OO");
//    Tensor& Fb_VV = *F.block("VV");

//    D1.fill_one_electron_spin([&](size_t p,MOSetSpinType sp,size_t q,MOSetSpinType sq){
//        if (sp  == Alpha){
//            size_t pp = mos_to_aocc[p];
//            size_t qq = mos_to_avir[q];
//            return Fa_oo(pp,pp) - Fa_vv(qq,qq);
//        }else if (sp  == Beta){
//            size_t pp = mos_to_bocc[p];
//            size_t qq = mos_to_bvir[q];
//            return Fb_OO(pp,pp) - Fb_VV(qq,qq);
//        }
//        return 0.0;
//    });

//    D2.fill_two_electron_spin([&](size_t p,MOSetSpinType sp,
//                                           size_t q,MOSetSpinType sq,
//                                           size_t r,MOSetSpinType sr,
//                                           size_t s,MOSetSpinType ss){
//        if ((sp == Alpha) and (sq == Alpha)){
//            size_t pp = mos_to_aocc[p];
//            size_t qq = mos_to_aocc[q];
//            size_t rr = mos_to_avir[r];
//            size_t ss = mos_to_avir[s];
//            return Fa_oo(pp,pp) + Fa_oo(qq,qq) - Fa_vv(rr,rr) - Fa_vv(ss,ss);
//        }else if ((sp == Alpha) and (sq == Beta) ){
//            size_t pp = mos_to_aocc[p];
//            size_t qq = mos_to_bocc[q];
//            size_t rr = mos_to_avir[r];
//            size_t ss = mos_to_bvir[s];
//            return Fa_oo(pp,pp) + Fb_OO(qq,qq) - Fa_vv(rr,rr) - Fb_VV(ss,ss);
//        }else if ((sp == Beta)  and (sq == Beta) ){
//            size_t pp = mos_to_bocc[p];
//            size_t qq = mos_to_bocc[q];
//            size_t rr = mos_to_bvir[r];
//            size_t ss = mos_to_bvir[s];
//            return Fb_OO(pp,pp) + Fb_OO(qq,qq) - Fb_VV(rr,rr) - Fb_VV(ss,ss);
//        }
//        return 0.0;
//    });
}

void DSRG_MRPT2::print_summary()
{
    // Print a summary
    std::vector<std::pair<std::string,int>> calculation_info;

    std::vector<std::pair<std::string,double>> calculation_info_double{
        {"Flow parameter",s_},
        {"Taylor expansion threshold",std::pow(10.0,-double(taylor_threshold_))}};

    std::vector<std::pair<std::string,std::string>> calculation_info_string;

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

double DSRG_MRPT2::compute_energy()
{
    return 0.0;
}

}} // End Namespaces
