#include "methodbase.h"

#include <libpsio/psio.h>
#include <libpsio/psio.hpp>

namespace psi{ namespace libadaptive{

MethodBase::MethodBase(boost::shared_ptr<Wavefunction> wfn, Options &options, ExplorerIntegrals* ints)
    : Wavefunction(options,_default_psio_lib_), ints_(ints)
{
    // Copy the wavefunction information
    copy(wfn);
    Tensor::set_print_level(debug_);
    startup();
}

MethodBase::~MethodBase()
{
    cleanup();
}

void MethodBase::startup()
{
    std::vector<size_t> a_occ_mos;
    std::vector<size_t> b_occ_mos;
    std::vector<size_t> a_vir_mos;
    std::vector<size_t> b_vir_mos;

    std::map<size_t,size_t> mos_to_aocc;
    std::map<size_t,size_t> mos_to_bocc;
    std::map<size_t,size_t> mos_to_avir;
    std::map<size_t,size_t> mos_to_bvir;

    for (int h = 0, p = 0; h < nirrep_; ++h){
        for (int i = 0; i < doccpi_[h]; ++i,++p){
            a_occ_mos.push_back(p);
            b_occ_mos.push_back(p);
            mos_to_aocc[p] = a_occ_mos.size()-1;
            mos_to_bocc[p] = a_occ_mos.size()-1;
        }
        for (int i = 0; i < soccpi_[h]; ++i,++p){
            a_occ_mos.push_back(p);
            b_vir_mos.push_back(p);
        }
        for (int a = 0; a < nmopi_[h] - doccpi_[h] - soccpi_[h]; ++a,++p){
            a_vir_mos.push_back(p);
            b_vir_mos.push_back(p);
        }
    }

    for (size_t p = 0; p < a_occ_mos.size(); ++p) mos_to_aocc[a_occ_mos[p]] = p;
    for (size_t p = 0; p < b_occ_mos.size(); ++p) mos_to_bocc[b_occ_mos[p]] = p;
    for (size_t p = 0; p < a_vir_mos.size(); ++p) mos_to_avir[a_vir_mos[p]] = p;
    for (size_t p = 0; p < b_vir_mos.size(); ++p) mos_to_bvir[b_vir_mos[p]] = p;

    size_t naocc = a_occ_mos.size();
    size_t nbocc = b_occ_mos.size();
    size_t navir = a_vir_mos.size();
    size_t nbvir = b_vir_mos.size();

    BlockedTensor::add_primitive_mo_space("o","ijklmn",a_occ_mos,Alpha);
    BlockedTensor::add_primitive_mo_space("O","IJKLMN",b_occ_mos,Beta);
    BlockedTensor::add_primitive_mo_space("v","abcdef",a_vir_mos,Alpha);
    BlockedTensor::add_primitive_mo_space("V","ABCDEF",b_vir_mos,Beta);
    BlockedTensor::add_composite_mo_space("i","pqrstuvwxyz",{"o","v"});
    BlockedTensor::add_composite_mo_space("I","PQRSTUVWXYZ",{"O","V"});

    H.resize_spin_components("H","ii");
    G1.resize_spin_components("G1","oo");
    CG1.resize_spin_components("CG1","oo");
    F.resize_spin_components("F","ii");
    V.resize_spin_components("V","iiii");
    D1.resize_spin_components("D1","ov");
    D2.resize_spin_components("D2","oovv");
//    H.a.resize("Ha","ii");
//    H.b.resize("Ha","II");
//    F.a.resize("Fa","ii");
//    F.b.resize("Fb","II");
//    G1.a.resize("G1a","oo");
//    G1.b.resize("G1b","OO");
//    V.aa.resize("Vaa","iiii");
//    V.ab.resize("Vab","iIiI");
//    V.bb.resize("Vbb","IIII");
//    D2.aa.resize("D2aa","oovv");
//    D2.ab.resize("D2ab","oOvV");
//    D2.bb.resize("D2bb","OOVV");

//    Ha.resize("Ha","ii");
//    Hb.resize("Ha","II");
//    Fa.resize("Fa","ii");
//    Fb.resize("Fb","II");
//    G1a.resize("G1a","oo");
//    G1b.resize("G1b","OO");
//    Vaa.resize("Vaa","iiii");
//    Vab.resize("Vab","iIiI");
//    Vbb.resize("Vbb","IIII");
//    D2aa.resize("D2aa","oovv");
//    D2ab.resize("D2ab","oOvV");
//    D2bb.resize("D2bb","OOVV");

    // Fill in the one-electron operator (H)
    H.fill_one_electron_spin([&](size_t p,MOSetSpinType sp,size_t q,MOSetSpinType sq){
        return (sp == Alpha) ? ints_->oei_a(p,q) : ints_->oei_b(p,q);
    });

    G1.fill_one_electron_spin([&](size_t p,MOSetSpinType sp,size_t q,MOSetSpinType sq){
        return (p == q ? 1.0 : 0.0);
    });

    CG1.fill_one_electron_spin([&](size_t p,MOSetSpinType sp,size_t q,MOSetSpinType sq){
        return (p == q ? 1.0 : 0.0);
    });

    G1.print();
    CG1.print();

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

//    V.ab.fill_two_electron([&](size_t p,size_t q,size_t r,size_t s){return ints_->aptei_ab(p,q,r,s);});
//    V.bb.fill_two_electron([&](size_t p,size_t q,size_t r,size_t s){return ints_->aptei_bb(p,q,r,s);});

//    G1.a.fill_one_electron([&](size_t p,size_t q){return (p == q ? 1.0 : 0.0);});
//    G1.b.fill_one_electron([&](size_t p,size_t q){return (p == q ? 1.0 : 0.0);});

//    // Fill in the one-electron operator (H)
//    H.a.fill_one_electron([&](size_t p,size_t q){return ints_->oei_a(p,q);});
//    H.b.fill_one_electron([&](size_t p,size_t q){return ints_->oei_b(p,q);});
//    // Fill in the two-electron operator (V)
//    V.aa.fill_two_electron([&](size_t p,size_t q,size_t r,size_t s){return ints_->aptei_aa(p,q,r,s);});
//    V.ab.fill_two_electron([&](size_t p,size_t q,size_t r,size_t s){return ints_->aptei_ab(p,q,r,s);});
//    V.bb.fill_two_electron([&](size_t p,size_t q,size_t r,size_t s){return ints_->aptei_bb(p,q,r,s);});

//    G1.a.fill_one_electron([&](size_t p,size_t q){return (p == q ? 1.0 : 0.0);});
//    G1.b.fill_one_electron([&](size_t p,size_t q){return (p == q ? 1.0 : 0.0);});

    H.print();

    // Form the Fock matrix

    F["pq"]  = H["pq"];
    F["pq"] += V["prqs"] * G1["sr"];
    F["pq"] += V["pRqS"] * G1["SR"];

    F["PQ"] += H["PQ"];
    F["PQ"] += V["rPsQ"] * G1["sr"];
    F["PQ"] += V["PRQS"] * G1["SR"];


    F.print();

//    F.a["pq"]  = H.a["pq"];
//    F.a["pq"] += V.aa["prqs"] * G1.a["sr"];
//    F.a["pq"] += V.ab["pRqS"] * G1.b["SR"];

//    F.b["PQ"]  = H.b["PQ"];
//    F.b["PQ"] += V.ab["rPsQ"] * G1.a["sr"];
//    F.b["PQ"] += V.bb["PRQS"] * G1.b["SR"];
//    F.b["PQ"]  = H.b["PQ"];
//    F.b["PQ"] += V.ab["rPsQ"] * G1.a["sr"];
//    F.b["PQ"] += V.bb["PRQS"] * G1.b["SR"];

    Tensor& Fa_oo = *F.block("oo");
    Tensor& Fa_vv = *F.block("vv");
    Tensor& Fb_OO = *F.block("OO");
    Tensor& Fb_VV = *F.block("VV");

    D1.fill_one_electron_spin([&](size_t p,MOSetSpinType sp,size_t q,MOSetSpinType sq){
        if (sp  == Alpha){
            size_t pp = mos_to_aocc[p];
            size_t qq = mos_to_avir[q];
            return Fa_oo(pp,pp) - Fa_vv(qq,qq);
        }else if (sp  == Beta){
            size_t pp = mos_to_bocc[p];
            size_t qq = mos_to_bvir[q];
            return Fb_OO(pp,pp) - Fb_VV(qq,qq);
        }
        return 0.0;
    });

    D2.fill_two_electron_spin([&](size_t p,MOSetSpinType sp,
                                           size_t q,MOSetSpinType sq,
                                           size_t r,MOSetSpinType sr,
                                           size_t s,MOSetSpinType ss){
        if ((sp == Alpha) and (sq == Alpha)){
            size_t pp = mos_to_aocc[p];
            size_t qq = mos_to_aocc[q];
            size_t rr = mos_to_avir[r];
            size_t ss = mos_to_avir[s];
            return Fa_oo(pp,pp) + Fa_oo(qq,qq) - Fa_vv(rr,rr) - Fa_vv(ss,ss);
        }else if ((sp == Alpha) and (sq == Beta) ){
            size_t pp = mos_to_aocc[p];
            size_t qq = mos_to_bocc[q];
            size_t rr = mos_to_avir[r];
            size_t ss = mos_to_bvir[s];
            return Fa_oo(pp,pp) + Fb_OO(qq,qq) - Fa_vv(rr,rr) - Fb_VV(ss,ss);
        }else if ((sp == Beta)  and (sq == Beta) ){
            size_t pp = mos_to_bocc[p];
            size_t qq = mos_to_bocc[q];
            size_t rr = mos_to_bvir[r];
            size_t ss = mos_to_bvir[s];
            return Fb_OO(pp,pp) + Fb_OO(qq,qq) - Fb_VV(rr,rr) - Fb_VV(ss,ss);
        }
        return 0.0;
    });

//    D2.aa.fill_two_electron(
//                [&](size_t p,size_t q,size_t r,size_t s){
//        size_t pp = mos_to_aocc[p];
//        size_t qq = mos_to_aocc[q];
//        size_t rr = mos_to_avir[r];
//        size_t ss = mos_to_avir[s];
//        return Fa_oo(pp,pp) + Fa_oo(qq,qq) - Fa_vv(rr,rr) - Fa_vv(ss,ss);});
//    D2.ab.fill_two_electron(
//                [&](size_t p,size_t q,size_t r,size_t s){
//        size_t pp = mos_to_aocc[p];
//        size_t qq = mos_to_bocc[q];
//        size_t rr = mos_to_avir[r];
//        size_t ss = mos_to_bvir[s];
//        return Fa_oo(pp,pp) + Fb_OO(qq,qq) - Fa_vv(rr,rr) - Fb_VV(ss,ss);});
//    D2.bb.fill_two_electron(
//                [&](size_t p,size_t q,size_t r,size_t s){
//        size_t pp = mos_to_bocc[p];
//        size_t qq = mos_to_bocc[q];
//        size_t rr = mos_to_bvir[r];
//        size_t ss = mos_to_bvir[s];
//        return Fb_OO(pp,pp) + Fb_OO(qq,qq) - Fb_VV(rr,rr) - Fb_VV(ss,ss);});
}

void MethodBase::cleanup()
{
}

}} // End Namespaces
