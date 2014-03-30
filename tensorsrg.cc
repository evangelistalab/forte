//#ifdef _HAS_LIBBTL_
#include "tensor_basic.h"
#include "tensor_labeled.h"
#include "tensor_product.h"
#include "tensorsrg.h"


#include <cmath>

#include <boost/numeric/odeint.hpp>

#include <libpsio/psio.hpp>
#include <libmints/wavefunction.h>
#include <libmints/molecule.h>

#include "mosrg.h"

using namespace std;
using namespace psi;

namespace psi{ namespace libadaptive{

TensorSRG::TensorSRG(boost::shared_ptr<Wavefunction> wfn, Options &options, ExplorerIntegrals* ints)
    : MethodBase(wfn,options,ints)
{
    startup();
}

TensorSRG::~TensorSRG()
{
}

void TensorSRG::startup()
{
    fprintf(outfile,"\n\n      --------------------------------------");
    fprintf(outfile,"\n          Similarity Renormalization Group");
    fprintf(outfile,"\n                tensor-based code");
    fprintf(outfile,"\n");
    fprintf(outfile,"\n                Version 0.1.0");
    fprintf(outfile,"\n");
    fprintf(outfile,"\n       written by Francesco A. Evangelista");
    fprintf(outfile,"\n      --------------------------------------\n");
    fflush(outfile);
}

void TensorSRG::cleanup()
{
}

double TensorSRG::compute_energy()
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
    BlockedTensor::print_mo_spaces();
    BlockedTensor T2aa("T2aa","oovv");
    BlockedTensor T2ab("T2ab","oOvV");
    BlockedTensor T2bb("T2bb","OOVV");
    BlockedTensor D2aa("D2aa","oovv");
    BlockedTensor D2ab("D2ab","oOvV");
    BlockedTensor D2bb("D2bb","OOVV");
    BlockedTensor Ha("H","ii");
    BlockedTensor Hb("H","II");
    BlockedTensor Fa("F","ii");
    BlockedTensor Fb("F","II");
    BlockedTensor G1a("gamma_1 alpha","oo");
    BlockedTensor G1b("gamma_1 beta","OO");
    BlockedTensor Vaa("Vaa","iiii");
    BlockedTensor Vab("Vab","iIiI");
    BlockedTensor Vbb("Vbb","IIII");

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


    T2aa["ijab"] = Vaa["ijab"] / D2aa["ijab"];
    T2ab["iJaB"] = Vab["iJaB"] / D2ab["iJaB"];
    T2bb["IJAB"] = Vbb["IJAB"] / D2bb["IJAB"];

    double Eaa = 0.25 * BlockedTensor::dot(T2aa["ijab"],Vaa["ijab"]);
    double Eab = BlockedTensor::dot(T2ab["iJaB"],Vab["iJaB"]);
    double Ebb = 0.25 * BlockedTensor::dot(T2bb["IJAB"],Vbb["IJAB"]);

    double mp2_correlation_energy = Eaa + Eab + Ebb;
    double ref_energy = reference_energy();
    fprintf(outfile,"\n  SCF energy                            = %20.15f",ref_energy);
    fprintf(outfile,"\n  MP2 correlation energy                = %20.15f",mp2_correlation_energy);
    fprintf(outfile,"\n* MP2 total energy                      = %20.15f",ref_energy + mp2_correlation_energy);
    return 0.0;
}

}} // EndNamespaces
























