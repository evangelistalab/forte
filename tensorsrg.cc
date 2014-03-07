//#ifdef _HAS_LIBBTL_
#include "tensor.h"
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
    BlockedTensor::add_mo_space("O","ijkl",occ_mos);
    BlockedTensor::add_mo_space("V","abcd",vir_mos);
    BlockedTensor::add_composite_mo_space("I","pqrs",{"O","V"});
    BlockedTensor::print_mo_spaces();
    BlockedTensor T2("T","OOVV");
    BlockedTensor V("V","IIII");
    std::map<std::vector<std::string>,Tensor>& v_blocks = V.blocks();
    for (block_iterator it = v_blocks.begin(), endit = v_blocks.end(); it != endit; ++it){
        std::vector<size_t> mos[4];
        for (int i = 0; i < 4; ++i){
            fprintf(outfile,"\n  %d %s",i,it->first[i].c_str());
//            BlockedTensor::mo_label_to_sets[it->first[i]][0].mo();
        }
////        for (string s : it->first){

////        }
    }
}

void TensorSRG::cleanup()
{
}

}} // EndNamespaces
