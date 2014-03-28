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

TensorSRG::TensorSRG(Options &options, ExplorerIntegrals* ints)
    : MethodBase(options,ints)
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
    startup();
}

TensorSRG::~TensorSRG()
{
}

void TensorSRG::startup()
{
    doccpi_.print();
//    std::vector<size_t> n2 = {nmo_,nmo_};
//    std::vector<size_t> n4 = {nmo_,nmo_,nmo_,nmo_};

    std::vector<size_t> occ_mos;
    std::vector<size_t> vir_mos;
    for (int h = 0, p = 0; h < nirrep_; ++h){
        for (int i = 0; i < doccpi_[h]; ++i,++p){
            occ_mos.push_back(p);
        }
        for (int a = 0; a < nmopi_[h] - doccpi_[h]; ++a,++p){
            vir_mos.push_back(p);
        }
    }
    BlockedTensor::add_primitive_mo_space("O","ijkl",occ_mos);
    BlockedTensor::add_primitive_mo_space("V","abcd",vir_mos);
    BlockedTensor::add_composite_mo_space("I","pqrstu",{"O","V"});
    BlockedTensor::print_mo_spaces();
    BlockedTensor T2("T","OOVV");
    BlockedTensor Ha("H","II");
    BlockedTensor Hb("H","II");
    BlockedTensor Fa("F","II");
    BlockedTensor Fb("F","II");
    BlockedTensor G1a("gamma_1 alpha","OO");
    BlockedTensor G1b("gamma_1 beta","OO");
    BlockedTensor Vaa("V","IIII");
    BlockedTensor Vab("V","IIII");
    BlockedTensor Vbb("V","IIII");

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
    Fa["pq"] += Vab["prqs"] * G1b["sr"];
    Fa.print();
}

void TensorSRG::cleanup()
{
}

}} // EndNamespaces
