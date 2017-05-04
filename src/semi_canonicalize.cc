/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2017 by its authors (see LICENSE, AUTHORS).
 *
 * The copyrights for code used from other parties are included in
 * the corresponding files.
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see http://www.gnu.org/licenses/.
 *
 * @END LICENSE
 */

#include "ambit/blocked_tensor.h"

#include "psi4/libpsio/psio.h"
#include "psi4/libpsio/psio.hpp"

#include "psi4/libmints/matrix.h"
#include "psi4/libmints/vector.h"

#include "helpers.h"
#include "semi_canonicalize.h"
#include "../blockedtensorfactory.h"

namespace psi {
namespace forte {

using namespace ambit;

SemiCanonical::SemiCanonical(std::shared_ptr<Wavefunction> wfn,
                             Options& options,
                             std::shared_ptr<ForteIntegrals> ints,
                             std::shared_ptr<MOSpaceInfo> mo_space_info,
                             Reference& reference) {
    print_method_banner(
        {"Semi-Canonical Orbitals", "Francesco A. Evangelista"});
    Timer SemiCanonicalize;

    // 1. Build the Fock matrix
    int nirrep = wfn->nirrep();
    size_t ncmo = mo_space_info->size("CORRELATED");
    size_t nact = mo_space_info->size("ACTIVE");
    Dimension nmopi = wfn->nmopi();
    Dimension ncmopi = mo_space_info->get_dimension("CORRELATED");
    Dimension fdocc = mo_space_info->get_dimension("FROZEN_DOCC");
    Dimension rdocc = mo_space_info->get_dimension("RESTRICTED_DOCC");
    Dimension actv = mo_space_info->get_dimension("ACTIVE");
    Dimension ruocc = mo_space_info->get_dimension("RESTRICTED_UOCC");

    SharedMatrix Da(new Matrix("Da", ncmo, ncmo));
    SharedMatrix Db(new Matrix("Db", ncmo, ncmo));

    Matrix L1a = tensor_to_matrix(reference.L1a(), actv);
    Matrix L1b = tensor_to_matrix(reference.L1b(), actv);

    for (int h = 0, offset = 0; h < nirrep; ++h) {
        // core block (diagonal)
        for (int i = 0; i < rdocc[h]; ++i) {
            Da->set(offset + i, offset + i, 1.0);
            Db->set(offset + i, offset + i, 1.0);
        }

        offset += rdocc[h];

        // active block
        for (int u = 0; u < actv[h]; ++u) {
            for (int v = 0; v < actv[h]; ++v) {
                Da->set(offset + u, offset + v, L1a.get(h, u, v));
                Db->set(offset + u, offset + v, L1b.get(h, u, v));
            }
        }

        offset += ncmopi[h] - rdocc[h];
    }

    Timer FockTime;
    ints->make_fock_matrix(Da, Db);
    outfile->Printf("\n Took %8.6f s to build fock matrix", FockTime.get());

    // 2. Diagonalize the diagonal blocks of the Fock matrix
    SharedMatrix Fc_a(new Matrix("Fock core alpha", rdocc, rdocc));
    SharedMatrix Fc_b(new Matrix("Fock core beta", rdocc, rdocc));
    SharedMatrix Fa_a(new Matrix("Fock active alpha", actv, actv));
    SharedMatrix Fa_b(new Matrix("Fock active beta", actv, actv));
    SharedMatrix Fv_a(new Matrix("Fock virtual alpha", ruocc, ruocc));
    SharedMatrix Fv_b(new Matrix("Fock virtual beta", ruocc, ruocc));

    for (int h = 0, offset = 0; h < nirrep; ++h) {
        // core block
        for (int i = 0; i < rdocc[h]; ++i) {
            for (int j = 0; j < rdocc[h]; ++j) {
                Fc_a->set(h, i, j, ints->fock_a(offset + i, offset + j));
                Fc_b->set(h, i, j, ints->fock_b(offset + i, offset + j));
            }
        }
        offset += rdocc[h];

        // active block
        for (int u = 0; u < actv[h]; ++u) {
            for (int v = 0; v < actv[h]; ++v) {
                Fa_a->set(h, u, v, ints->fock_a(offset + u, offset + v));
                Fa_b->set(h, u, v, ints->fock_b(offset + u, offset + v));
            }
        }
        offset += actv[h];

        // virtual block
        for (int a = 0; a < ruocc[h]; ++a) {
            for (int b = 0; b < ruocc[h]; ++b) {
                Fv_a->set(h, a, b, ints->fock_a(offset + a, offset + b));
                Fv_b->set(h, a, b, ints->fock_b(offset + a, offset + b));
            }
        }
        offset += ruocc[h];
    }

    // Diagonalize each block of the Fock matrix
    std::vector<SharedMatrix> evecs;
    std::vector<SharedVector> evals;
    for (auto F : {Fc_a, Fc_b, Fa_a, Fa_b, Fv_a, Fv_b}) {
        SharedMatrix U(new Matrix("U", F->rowspi(), F->colspi()));
        SharedVector lambda(new Vector("lambda", F->rowspi()));
        F->diagonalize(U, lambda);
        evecs.push_back(U);
        evals.push_back(lambda);
    }
    //    Fv_a->print();
    //    SharedMatrix Uv = evecs[4];
    //    Fv_a->transform(Uv);
    //    Fv_a->print();

    // 3. Build the unitary matrices
    Matrix Ua("Ua", nmopi, nmopi);
    Matrix Ub("Ub", nmopi, nmopi);

    Matrix Ua_copy(nact,nact);
    Matrix Ub_copy(nact,nact);

    size_t act_off = 0;
    for (int h = 0; h < nirrep; ++h) {
        size_t offset = 0;

        // Set the matrices to the identity,
        // this takes care of the frozen core and virtual spaces
        for (int p = 0; p < nmopi[h]; ++p) {
            Ua.set(h, p, p, 1.0);
            Ub.set(h, p, p, 1.0);
        }

        offset += fdocc[h];

        // core block
        for (int i = 0; i < rdocc[h]; ++i) {
            for (int j = 0; j < rdocc[h]; ++j) {
                Ua.set(h, offset + i, offset + j, evecs[0]->get(h, i, j));
                Ub.set(h, offset + i, offset + j, evecs[1]->get(h, i, j));
            }
        }
        offset += rdocc[h];

        // active block
        for (int u = 0; u < actv[h]; ++u) {
            for (int v = 0; v < actv[h]; ++v) {
                Ua.set(h, offset + u, offset + v, evecs[2]->get(h, u, v));
                Ub.set(h, offset + u, offset + v, evecs[3]->get(h, u, v));
                Ua_copy.set(act_off + v, act_off + u, evecs[2]->get(h, u, v));
                Ub_copy.set(act_off + v, act_off + u, evecs[3]->get(h, u, v));
            }
        }
        act_off += actv[h];
        offset += actv[h];

        // virtual block
        for (int a = 0; a < ruocc[h]; ++a) {
            for (int b = 0; b < ruocc[h]; ++b) {
                Ua.set(h, offset + a, offset + b, evecs[4]->get(h, a, b));
                Ub.set(h, offset + a, offset + b, evecs[5]->get(h, a, b));
            }
        }
    }

    // 4. Transform the MO coefficients
    SharedMatrix Ca = wfn->Ca();
    SharedMatrix Cb = wfn->Cb();
    SharedMatrix Ca_new(Ca->clone());
    SharedMatrix Cb_new(Cb->clone());
    Ca_new->gemm(false, false, 1.0, Ca, Ua, 0.0);
    Cb_new->gemm(false, false, 1.0, Cb, Ub, 0.0);
    Ca->copy(Ca_new);
    Cb->copy(Cb_new);

   // Ca_new->print();
   // Ca_copy->print();

    // 5. Retransform the integrals in the new basis
    print_h2("Integral transformation");
    ints->retransform_integrals();

    // 6. Transform cumulants in the new basis
    BlockedTensor::reset_mo_spaces();
    BlockedTensor::set_expert_mode(true);

    std::vector<size_t> core_mo = mo_space_info->get_corr_abs_mo("RESTRICTED_DOCC");
    std::vector<size_t> vir_mo = mo_space_info->get_corr_abs_mo("RESTRICTED_UOCC");
    std::vector<size_t> active_mo(nact);
    for( int i = 0; i < nact; ++i ){
        active_mo[i] = i;
    }
    
//    BlockedTensor::add_mo_space("c", "mn",core_mo , AlphaSpin); 
//    BlockedTensor::add_mo_space("C", "MN", core_mo, BetaSpin); 
    BlockedTensor::add_mo_space("a", "abcdpqrsijkl", active_mo, AlphaSpin); 
    BlockedTensor::add_mo_space("A", "ABCDPQRSIJKL", active_mo, BetaSpin); 
//    BlockedTensor::add_mo_space("v", "xy", vir_mo, AlphaSpin); 
//    BlockedTensor::add_mo_space("V", "XY", vir_mo, BetaSpin); 
  
    // First build transformation matrices
    ambit::BlockedTensor U = BlockedTensor::build(CoreTensor, "U", spin_cases({"aa"}));
    U.iterate([&]( const std::vector<size_t>& i, const std::vector<SpinType>& spin, double& value ) {
        if( spin[0] == AlphaSpin ){
            value = Ua_copy.get(i[0],i[1]);
        } else { 
            value = Ub_copy.get(i[0],i[1]);
        }
    });


    // Transform the 1-cumulants
    ambit::BlockedTensor gamma1 = BlockedTensor::build(CoreTensor, "Gamma1", spin_cases({"aa"}));
    ambit::BlockedTensor rdm1 = BlockedTensor::build(CoreTensor, "rdm1", spin_cases({"aa"}));

    rdm1.block("aa")("pq") = reference.L1a()("pq");
    rdm1.block("AA")("pq") = reference.L1b()("pq");

    gamma1["pq"] = U["pa"] * rdm1["ab"] * U["qb"]; 
    gamma1["PQ"] = U["PA"] * rdm1["AB"] * U["QB"];


    // Transform 2-cumulants
    ambit::BlockedTensor tmpL2 = BlockedTensor::build(CoreTensor, "tmpL2", spin_cases({"aaaa"}));
    ambit::BlockedTensor L2 = BlockedTensor::build(CoreTensor, "L2", spin_cases({"aaaa"}));
    
    L2.block("aaaa")("abij") = reference.L2aa()("abij"); 
    L2.block("aAaA")("abij") = reference.L2ab()("abij"); 
    L2.block("AAAA")("abij") = reference.L2bb()("abij"); 

    tmpL2["cdij"] = U["ca"] * U["db"] * L2["abij"];
    tmpL2["cDiJ"] = U["ca"] * U["DB"] * L2["aBiJ"];
    tmpL2["CDIJ"] = U["CA"] * U["DB"] * L2["ABIJ"];

    L2["abkl"] = tmpL2["abij"] * U["lj"] * U["ki"];
    L2["aBkL"] = tmpL2["aBiJ"] * U["LJ"] * U["ki"];
    L2["ABKL"] = tmpL2["ABIJ"] * U["LJ"] * U["KI"];

    ambit::BlockedTensor g2 = BlockedTensor::build(CoreTensor, "g2", spin_cases({"aaaa"}));
    ambit::BlockedTensor tmpg2 = BlockedTensor::build(CoreTensor, "tmpg2", spin_cases({"aaaa"}));

    g2.block("aaaa")("abij") = reference.g2aa()("abij"); 
    g2.block("aAaA")("abij") = reference.g2ab()("abij"); 
    g2.block("AAAA")("abij") = reference.g2bb()("abij"); 

    tmpg2["cdij"] = U["ca"] * U["db"] * g2["abij"];
    tmpg2["cDiJ"] = U["ca"] * U["DB"] * g2["aBiJ"];
    tmpg2["CDIJ"] = U["CA"] * U["DB"] * g2["ABIJ"];

    g2["abkl"] = tmpg2["abij"] * U["lj"] * U["ki"];
    g2["aBkL"] = tmpg2["aBiJ"] * U["LJ"] * U["ki"];
    g2["ABKL"] = tmpg2["ABIJ"] * U["LJ"] * U["KI"];

    // Transform 3 cumulants
    ambit::BlockedTensor tmpL3 = BlockedTensor::build(CoreTensor, "Gamma3", spin_cases({"aaaaaa"}));
    ambit::BlockedTensor L3 = BlockedTensor::build(CoreTensor, "L3", spin_cases({"aaaaaa"}));

    L3.block("aaaaaa")("abcijk") = reference.L3aaa()("abcijk");
    L3.block("aaAaaA")("abcijk") = reference.L3aab()("abcijk");
    L3.block("aAAaAA")("abcijk") = reference.L3abb()("abcijk");
    L3.block("AAAAAA")("abcijk") = reference.L3bbb()("abcijk");
     
    tmpL3["dpqijk"] = U["da"] * U["pb"] * U["qc"] * L3["abcijk"]; 
    tmpL3["dpQijK"] = U["da"] * U["pb"] * U["QC"] * L3["abCijK"]; 
    tmpL3["dPQiJK"] = U["da"] * U["PB"] * U["QC"] * L3["aBCiJK"]; 
    tmpL3["DPQIJK"] = U["DA"] * U["PB"] * U["QC"] * L3["ABCIJK"]; 

    L3["abclrs"] = tmpL3["abcijk"] * U["li"] * U["rj"] * U["sk"]; 
    L3["abClrS"] = tmpL3["abCijK"] * U["LI"] * U["rj"] * U["sk"]; 
    L3["aBClRS"] = tmpL3["aBCiJK"] * U["LI"] * U["RJ"] * U["sk"]; 
    L3["ABCLRS"] = tmpL3["ABCIJK"] * U["LI"] * U["RJ"] * U["SK"]; 
    // Recompute the energy 

    // Update the reference
    reference.set_L1a( gamma1.block("aa") );    
    reference.set_L1b( gamma1.block("AA") );    

    reference.set_L2aa( L2.block("aaaa") );    
    reference.set_L2ab( L2.block("aAaA") );    
    reference.set_L2bb( L2.block("AAAA") );    

    reference.set_g2aa( g2.block("aaaa") );    
    reference.set_g2ab( g2.block("aAaA") );    
    reference.set_g2bb( g2.block("AAAA") );    

    reference.set_L3aaa( L3.block("aaaaaa"));    
    reference.set_L3aab( L3.block("aaAaaA"));    
    reference.set_L3abb( L3.block("aAAaAA"));    
    reference.set_L3bbb( L3.block("AAAAAA"));    

    outfile->Printf("\n SemiCanonicalize takes %8.6f s.",
                    SemiCanonicalize.get());
}
}
} // End Namespaces
