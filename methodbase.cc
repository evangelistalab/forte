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

    BlockedTensor::add_primitive_mo_space("o","ijkl",a_occ_mos);
    BlockedTensor::add_primitive_mo_space("O","IJKL",b_occ_mos);
    BlockedTensor::add_primitive_mo_space("v","abcd",a_vir_mos);
    BlockedTensor::add_primitive_mo_space("V","ABCD",b_vir_mos);
    BlockedTensor::add_composite_mo_space("i","pqrstu",{"o","v"});
    BlockedTensor::add_composite_mo_space("I","PQRSTU",{"O","V"});

    Ha.resize("Ha","ii");
    Hb.resize("Ha","II");
    Fa.resize("Fa","ii");
    Fb.resize("Fb","II");
    G1a.resize("G1a","oo");
    G1b.resize("G1b","OO");
    Vaa.resize("Vaa","iiii");
    Vab.resize("Vab","iIiI");
    Vbb.resize("Vbb","IIII");
    D2aa.resize("D2aa","oovv");
    D2ab.resize("D2ab","oOvV");
    D2bb.resize("D2bb","OOVV");

    // Fill in the one-electron operator (H)
    Ha.fill_one_electron([&](size_t p,size_t q){return ints_->oei_a(p,q);});
    Hb.fill_one_electron([&](size_t p,size_t q){return ints_->oei_b(p,q);});
    // Fill in the two-electron operator (V)
    Vaa.fill_two_electron([&](size_t p,size_t q,size_t r,size_t s){return ints_->aptei_aa(p,q,r,s);});
    Vab.fill_two_electron([&](size_t p,size_t q,size_t r,size_t s){return ints_->aptei_ab(p,q,r,s);});
    Vbb.fill_two_electron([&](size_t p,size_t q,size_t r,size_t s){return ints_->aptei_bb(p,q,r,s);});

    G1a.fill_one_electron([&](size_t p,size_t q){return (p == q ? 1.0 : 0.0);});
    G1b.fill_one_electron([&](size_t p,size_t q){return (p == q ? 1.0 : 0.0);});

    // Form the Fock matrix
    Fa["pq"]  = Ha["pq"];
    Fa["pq"] += Vaa["prqs"] * G1a["sr"];
    Fa["pq"] += Vab["pRqS"] * G1b["SR"];

    Fb["PQ"]  = Hb["PQ"];
    Fb["PQ"] += Vab["rPsQ"] * G1a["sr"];
    Fb["PQ"] += Vbb["PRQS"] * G1b["SR"];

    Tensor& Fa_oo = *Fa.block("oo");
    Tensor& Fa_vv = *Fa.block("vv");
    Tensor& Fb_OO = *Fb.block("OO");
    Tensor& Fb_VV = *Fb.block("VV");

    D2aa.fill_two_electron(
                [&](size_t p,size_t q,size_t r,size_t s){
        size_t pp = mos_to_aocc[p];
        size_t qq = mos_to_aocc[q];
        size_t rr = mos_to_avir[r];
        size_t ss = mos_to_avir[s];
        return Fa_oo(pp,pp) + Fa_oo(qq,qq) - Fa_vv(rr,rr) - Fa_vv(ss,ss);});
    D2ab.fill_two_electron(
                [&](size_t p,size_t q,size_t r,size_t s){
        size_t pp = mos_to_aocc[p];
        size_t qq = mos_to_bocc[q];
        size_t rr = mos_to_avir[r];
        size_t ss = mos_to_bvir[s];
        return Fa_oo(pp,pp) + Fb_OO(qq,qq) - Fa_vv(rr,rr) - Fb_VV(ss,ss);});
    D2bb.fill_two_electron(
                [&](size_t p,size_t q,size_t r,size_t s){
        size_t pp = mos_to_bocc[p];
        size_t qq = mos_to_bocc[q];
        size_t rr = mos_to_bvir[r];
        size_t ss = mos_to_bvir[s];
        return Fb_OO(pp,pp) + Fb_OO(qq,qq) - Fb_VV(rr,rr) - Fb_VV(ss,ss);});
}

void MethodBase::cleanup()
{
}
    
    //void MethodBase::sort_integrals()
    //{
    ////    loop_mo_p loop_mo_q{
    ////        H1_.aa[p][q] = ints_->oei_a(p,q);
    ////        H1_.bb[p][q] = ints_->oei_b(p,q);
    ////    }
    ////    loop_mo_p loop_mo_q loop_mo_r loop_mo_s{
    ////        V_.aaaa[p][q][r][s] = ints_->aptei_aa(p,q,r,s); //ints_->rtei(p,r,q,s) - ints_->rtei(p,s,q,r);
    ////        V_.abab[p][q][r][s] = ints_->aptei_ab(p,q,r,s); //ints_->rtei(p,r,q,s);
    ////        V_.bbbb[p][q][r][s] = ints_->aptei_bb(p,q,r,s); //ints_->rtei(p,r,q,s) - ints_->rtei(p,s,q,r);
    ////    }
    //}
    
    //void MethodBase::build_fock()
    //{
    ////    boost::shared_ptr<Wavefunction> wfn = Process::environment.wavefunction();
    
    ////    boost::shared_ptr<Molecule> molecule_ = wfn->molecule();
    ////    nuclear_repulsion_energy_ = molecule_->nuclear_repulsion_energy();
    
    ////    // Compute the reference energy
    ////    E0_ = nuclear_repulsion_energy_;
    ////    loop_mo_p loop_mo_q{
    ////        E0_ += H1_.aa[p][q] * G1_.aa[q][p];
    ////        E0_ += H1_.bb[p][q] * G1_.bb[q][p];
    ////    }
    ////    loop_mo_p loop_mo_q loop_mo_r loop_mo_s{
    ////        E0_ += 0.25 * V_.aaaa[p][q][r][s] * (G1_.aa[p][r] * G1_.aa[q][s] - G1_.aa[p][s] * G1_.aa[q][r]);
    ////        E0_ += 0.25 * V_.bbbb[p][q][r][s] * (G1_.bb[p][r] * G1_.bb[q][s] - G1_.bb[p][s] * G1_.bb[q][r]);
    ////        E0_ +=  1.0 * V_.abab[p][q][r][s] * G1_.aa[p][r] * G1_.bb[q][s];
    ////    }
    ////    // Compute the fock matrix
    ////    loop_mo_p loop_mo_q{
    ////        F_.aa[p][q] = H1_.aa[p][q];
    ////        F_.bb[p][q] = H1_.bb[p][q];
    ////        loop_mo_r loop_mo_s{
    ////            F_.aa[p][q] += V_.aaaa[p][r][q][s] * G1_.aa[s][r] + V_.abab[p][r][q][s] * G1_.bb[s][r];
    ////            F_.bb[p][q] += V_.bbbb[p][r][q][s] * G1_.bb[s][r] + V_.abab[r][p][s][q] * G1_.aa[s][r];
    ////        }
    ////    }
    
    ////    fprintf(outfile,"\n  The energy of the reference is: %20.12f Eh",E0_);
    ////    fprintf(outfile,"\n  Diagonal elements of the Fock matrix:");
    ////    fprintf(outfile,"\n  SO            Epsilon         ON");
    ////    loop_mo_p {
    ////        fprintf(outfile,"\n  %2d  %20.12f   %8.6f  %20.12f   %8.6f",p,F_.aa[p][p],G1_.aa[p][p],F_.bb[p][p],G1_.bb[p][p]);
    ////    }
    //}

}} // End Namespaces
