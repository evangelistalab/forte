/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2017 by its authors (see COPYING, COPYING.LESSER, AUTHORS).
 *
 * The copyrights for code used from other parties are included in
 * the corresponding files.
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program.  If not, see http://www.gnu.org/licenses/.
 *
 * @END LICENSE
 */

#include "ambit/blocked_tensor.h"

#include "psi4/libpsio/psio.h"
#include "psi4/libpsio/psio.hpp"

#include "psi4/libmints/matrix.h"
#include "psi4/libmints/vector.h"

#include "../blockedtensorfactory.h"
#include "helpers.h"
#include "semi_canonicalize.h"

namespace psi {
namespace forte {

using namespace ambit;

SemiCanonical::SemiCanonical(std::shared_ptr<Wavefunction> wfn, Options& options,
                             std::shared_ptr<ForteIntegrals> ints,
                             std::shared_ptr<MOSpaceInfo> mo_space_info, Reference& reference,
                             const bool& quiet)
    : wfn_(wfn), mo_space_info_(mo_space_info), ints_(ints), quiet_(quiet) {

    if (!quiet) {
        print_method_banner(
            {"Semi-Canonical Orbitals", "Jeffrey B. Schriber and Francesco A. Evangelista"});
    }

    // TODO: remove options, reference?

    // 0. initialize the dimension objects
    nirrep_ = wfn_->nirrep();
    ncmo_ = mo_space_info_->size("CORRELATED");
    nact_ = mo_space_info_->size("ACTIVE");
    nmopi_ = wfn_->nmopi();
    ncmopi_ = mo_space_info_->get_dimension("CORRELATED");
    fdocc_ = mo_space_info_->get_dimension("FROZEN_DOCC");
    rdocc_ = mo_space_info_->get_dimension("RESTRICTED_DOCC");
    actv_ = mo_space_info_->get_dimension("ACTIVE");
    ruocc_ = mo_space_info_->get_dimension("RESTRICTED_UOCC");
}

void SemiCanonical::semicanonicalize(Reference& reference) {
    Timer SemiCanonicalize;

    // 1. Build the Fock matrix
    build_fock_matrix(reference);

    // TODO: it is probably useful to test the Fock matrix
    // maybe the orbitals are semicanonical already

    // 2. Build transformation matrices from diagononalizing blocks in F

    // These transform all MOs
    SharedMatrix Ua(new Matrix("Ua", nmopi_, nmopi_));
    SharedMatrix Ub(new Matrix("Ub", nmopi_, nmopi_));

    // This transforms only within ACTIVE MOs
    // Use ambit Tensor here so that the ambit mo_spaces remains the same

    //    std::vector<size_t> active_mo(nact_);
    //    for (int i = 0; i < nact_; ++i) {
    //        active_mo[i] = i;
    //    }
    //    BlockedTensor::reset_mo_spaces();
    //    BlockedTensor::set_expert_mode(true);
    //    BlockedTensor::add_mo_space("a", "abcdpqrstuijk", active_mo, AlphaSpin);
    //    BlockedTensor::add_mo_space("A", "ABCDPQRSTUIJK", active_mo, BetaSpin);
    //    ambit::BlockedTensor U = BlockedTensor::build(CoreTensor, "U", spin_cases({"aa"}));

    ambit::Tensor Ua_t = ambit::Tensor::build(ambit::CoreTensor, "Ua", {nact_, nact_});
    ambit::Tensor Ub_t = ambit::Tensor::build(ambit::CoreTensor, "Ub", {nact_, nact_});

    // TODO: add ways to treat active hole and active particle orbitals
    // TODO: it might be also useful to diagonalize Fock block only when it is not diagonal

    build_transformation_matrices(Ua, Ub, Ua_t, Ub_t);

    // 3. Retransform integrals
    transform_ints(Ua, Ub);

    // 4. Transform RMDs/cumulants
    transform_reference(Ua_t, Ub_t, reference);

    outfile->Printf("\n SemiCanonicalize takes %8.6f s.", SemiCanonicalize.get());
}

void SemiCanonical::build_fock_matrix(Reference& reference) {
    // 1. Build the Fock matrix

    SharedMatrix Da(new Matrix("Da", ncmo_, ncmo_));
    SharedMatrix Db(new Matrix("Db", ncmo_, ncmo_));

    Matrix L1a = tensor_to_matrix(reference.L1a(), actv_);
    Matrix L1b = tensor_to_matrix(reference.L1b(), actv_);

    for (int h = 0, offset = 0; h < nirrep_; ++h) {
        // core block (diagonal)
        for (int i = 0; i < rdocc_[h]; ++i) {
            Da->set(offset + i, offset + i, 1.0);
            Db->set(offset + i, offset + i, 1.0);
        }

        offset += rdocc_[h];

        // active block
        for (int u = 0; u < actv_[h]; ++u) {
            for (int v = 0; v < actv_[h]; ++v) {
                Da->set(offset + u, offset + v, L1a.get(h, u, v));
                Db->set(offset + u, offset + v, L1b.get(h, u, v));
            }
        }

        offset += ncmopi_[h] - rdocc_[h];
    }

    Timer FockTime;
    ints_->make_fock_matrix(Da, Db);
    outfile->Printf("\n Took %8.6f s to build fock matrix", FockTime.get());
}

void SemiCanonical::build_transformation_matrices(SharedMatrix& Ua, SharedMatrix& Ub,
                                                  ambit::Tensor& Ua_t, ambit::Tensor& Ub_t) {
    // 2. Diagonalize the diagonal blocks of the Fock matrix
    SharedMatrix Fc_a(new Matrix("Fock core alpha", rdocc_, rdocc_));
    SharedMatrix Fc_b(new Matrix("Fock core beta", rdocc_, rdocc_));
    SharedMatrix Fa_a(new Matrix("Fock active alpha", actv_, actv_));
    SharedMatrix Fa_b(new Matrix("Fock active beta", actv_, actv_));
    SharedMatrix Fv_a(new Matrix("Fock virtual alpha", ruocc_, ruocc_));
    SharedMatrix Fv_b(new Matrix("Fock virtual beta", ruocc_, ruocc_));

    for (int h = 0, offset = 0; h < nirrep_; ++h) {
        // core block
        for (int i = 0; i < rdocc_[h]; ++i) {
            for (int j = 0; j < rdocc_[h]; ++j) {
                Fc_a->set(h, i, j, ints_->fock_a(offset + i, offset + j));
                Fc_b->set(h, i, j, ints_->fock_b(offset + i, offset + j));
            }
        }
        offset += rdocc_[h];

        // active block
        for (int u = 0; u < actv_[h]; ++u) {
            for (int v = 0; v < actv_[h]; ++v) {
                Fa_a->set(h, u, v, ints_->fock_a(offset + u, offset + v));
                Fa_b->set(h, u, v, ints_->fock_b(offset + u, offset + v));
            }
        }
        offset += actv_[h];

        // virtual block
        for (int a = 0; a < ruocc_[h]; ++a) {
            for (int b = 0; b < ruocc_[h]; ++b) {
                Fv_a->set(h, a, b, ints_->fock_a(offset + a, offset + b));
                Fv_b->set(h, a, b, ints_->fock_b(offset + a, offset + b));
            }
        }
        offset += ruocc_[h];
    }

    // Diagonalize each block of the Fock matrix
    std::vector<SharedMatrix> evecs;
    std::vector<SharedVector> evals;
    for (auto F : {Fc_a, Fc_b, Fa_a, Fa_b, Fv_a, Fv_b}) {
        SharedMatrix U_tmp(new Matrix("U", F->rowspi(), F->colspi()));
        SharedVector lambda(new Vector("lambda", F->rowspi()));
        F->diagonalize(U_tmp, lambda);
        evecs.push_back(U_tmp);
        evals.push_back(lambda);
    }
    //    Fv_a->print();
    //    SharedMatrix Uv = evecs[4];
    //    Fv_a->transform(Uv);
    //    Fv_a->print();

    // 3. Build the unitary matrices

    std::vector<double> UaData(nact_ * nact_, 0.0);
    std::vector<double> UbData(nact_ * nact_, 0.0);
    //    Matrix Ua_copy(nact_, nact_);
    //    Matrix Ub_copy(nact_, nact_);

    //    size_t act_off = 0;
    for (int h = 0, act_off = 0; h < nirrep_; ++h) {
        //        size_t offset = 0;
        int offset = 0;

        // Set the matrices to the identity,
        // this takes care of the frozen core and virtual spaces
        for (int p = 0; p < nmopi_[h]; ++p) {
            Ua->set(h, p, p, 1.0);
            Ub->set(h, p, p, 1.0);
        }
        offset += fdocc_[h];

        // core block
        for (int i = 0; i < rdocc_[h]; ++i) {
            for (int j = 0; j < rdocc_[h]; ++j) {
                Ua->set(h, offset + i, offset + j, evecs[0]->get(h, i, j));
                Ub->set(h, offset + i, offset + j, evecs[1]->get(h, i, j));
            }
        }
        offset += rdocc_[h];

        // active block
        for (int u = 0; u < actv_[h]; ++u) {
            for (int v = 0; v < actv_[h]; ++v) {
                Ua->set(h, offset + u, offset + v, evecs[2]->get(h, u, v));
                Ub->set(h, offset + u, offset + v, evecs[3]->get(h, u, v));

                int nu = act_off + u;
                int nv = act_off + v;
                UaData[nu * nact_ + nv] = evecs[2]->get(h, u, v);
                UbData[nu * nact_ + nv] = evecs[3]->get(h, u, v);

                //                Ua_copy.set(act_off + u, act_off + v, evecs[2]->get(h, u, v));
                //                Ub_copy.set(act_off + u, act_off + v, evecs[3]->get(h, u, v));
            }
        }
        act_off += actv_[h];
        offset += actv_[h];

        // virtual block
        for (int a = 0; a < ruocc_[h]; ++a) {
            for (int b = 0; b < ruocc_[h]; ++b) {
                Ua->set(h, offset + a, offset + b, evecs[4]->get(h, a, b));
                Ub->set(h, offset + a, offset + b, evecs[5]->get(h, a, b));
            }
        }
    }

    // copy active data to ambit tensors
    Ua_t.data() = UaData;
    Ub_t.data() = UbData;
    //    Ua_t.iterate(
    //        [&](const std::vector<size_t>& i, const std::vector<SpinType>& spin, double& value) {
    //            if (spin[0] == AlphaSpin) {
    //                value = Ua_copy.get(i[0], i[1]);
    //            } else {
    //                value = Ub_copy.get(i[0], i[1]);
    //            }
    //        });
}

void SemiCanonical::transform_ints(SharedMatrix& Ua, SharedMatrix& Ub) {
    SharedMatrix Ca = wfn_->Ca();
    SharedMatrix Cb = wfn_->Cb();
    SharedMatrix Ca_new(Ca->clone());
    SharedMatrix Cb_new(Cb->clone());
    Ca_new->gemm(false, false, 1.0, Ca, Ua, 0.0);
    Cb_new->gemm(false, false, 1.0, Cb, Ub, 0.0);
    Ca->copy(Ca_new);
    Cb->copy(Cb_new);

    // 5. Retransform the integrals in the new basis
    print_h2("Integral Transformation");
    ints_->retransform_integrals();
}

void SemiCanonical::transform_reference(ambit::Tensor& Ua, ambit::Tensor& Ub,
                                        Reference& reference) {
    print_h2("Reference Transformation");

    // Transform the 1-cumulants
    ambit::Tensor L1a0 = reference.L1a();
    ambit::Tensor L1b0 = reference.L1b();

    ambit::Tensor L1aT = ambit::Tensor::build(ambit::CoreTensor, "Transformed L1a", {nact_, nact_});
    ambit::Tensor L1bT = ambit::Tensor::build(ambit::CoreTensor, "Transformed L1b", {nact_, nact_});
    L1aT("pq") = Ua("ap") * L1a0("ab") * Ua("bq");
    L1bT("PQ") = Ub("AP") * L1b0("AB") * Ub("BQ");

    reference.set_L1a(L1aT);
    reference.set_L1b(L1bT);
    outfile->Printf("\n    Transformed 1 cumulants.");

    //    ambit::BlockedTensor gamma1 = BlockedTensor::build(CoreTensor, "Gamma1",
    //    spin_cases({"aa"}));
    //    ambit::BlockedTensor rdm1 = BlockedTensor::build(CoreTensor, "rdm1", spin_cases({"aa"}));

    //    rdm1.block("aa")("pq") = reference.L1a()("pq");
    //    rdm1.block("AA")("pq") = reference.L1b()("pq");

    //    gamma1["pq"] = Ua["ap"] * rdm1["ab"] * Ua["bq"];
    //    gamma1["PQ"] = Ua["AP"] * rdm1["AB"] * Ua["BQ"];

    // Transform 2-cumulants
    ambit::Tensor L2aa0 = reference.L2aa();
    ambit::Tensor L2ab0 = reference.L2ab();
    ambit::Tensor L2bb0 = reference.L2bb();

    ambit::Tensor L2aaT =
        ambit::Tensor::build(ambit::CoreTensor, "Transformed L2aa", {nact_, nact_, nact_, nact_});
    ambit::Tensor L2abT =
        ambit::Tensor::build(ambit::CoreTensor, "Transformed L2ab", {nact_, nact_, nact_, nact_});
    ambit::Tensor L2bbT =
        ambit::Tensor::build(ambit::CoreTensor, "Transformed L2bb", {nact_, nact_, nact_, nact_});

    L2aaT("pqrs") = Ua("ap") * Ua("bq") * L2aa0("abcd") * Ua("cr") * Ua("ds");
    L2abT("pQrS") = Ua("ap") * Ub("BQ") * L2ab0("aBcD") * Ua("cr") * Ub("DS");
    L2bbT("PQRS") = Ub("AP") * Ub("BQ") * L2bb0("ABCD") * Ub("CR") * Ub("DS");

    reference.set_L2aa(L2aaT);
    reference.set_L2ab(L2abT);
    reference.set_L2bb(L2bbT);
    outfile->Printf("\n    Transformed 2 cumulants.");

    // Transform 2-RDMs
    // Recompute 2-RDMs using transformed cumulants
    ambit::Tensor G2aaT =
        ambit::Tensor::build(ambit::CoreTensor, "Transformed G2aa", {nact_, nact_, nact_, nact_});
    ambit::Tensor G2abT =
        ambit::Tensor::build(ambit::CoreTensor, "Transformed G2ab", {nact_, nact_, nact_, nact_});
    ambit::Tensor G2bbT =
        ambit::Tensor::build(ambit::CoreTensor, "Transformed G2bb", {nact_, nact_, nact_, nact_});

    G2aaT.copy(L2aaT);
    G2abT.copy(L2abT);
    G2bbT.copy(L2bbT);

    G2aaT("pqrs") += L1aT("pr") * L1aT("qs");
    G2aaT("pqrs") -= L1aT("ps") * L1aT("qr");

    G2abT("pqrs") += L1aT("pr") * L1bT("qs");

    G2bbT("pqrs") += L1bT("pr") * L1bT("qs");
    G2bbT("pqrs") -= L1bT("ps") * L1bT("qr");

    //    L2aa0 = reference.g2aa();
    //    L2ab0 = reference.g2ab();
    //    L2bb0 = reference.g2bb();

    //    G2aaT("pqrs") = Ua("ap") * Ua("bq") * L2aa0("abcd") * Ua("cr") * Ua("ds");
    //    G2abT("pQrS") = Ua("ap") * Ub("BQ") * L2ab0("aBcD") * Ua("cr") * Ub("DS");
    //    G2bbT("PQRS") = Ub("AP") * Ub("BQ") * L2bb0("ABCD") * Ub("CR") * Ub("DS");

    reference.set_g2aa(G2aaT);
    reference.set_g2ab(G2abT);
    reference.set_g2bb(G2bbT);
    outfile->Printf("\n    Transformed 2 RDMs.");

    //    ambit::BlockedTensor tmpL2 = BlockedTensor::build(CoreTensor, "tmpL2",
    //    spin_cases({"aaaa"}));
    //    ambit::BlockedTensor L2 = BlockedTensor::build(CoreTensor, "L2", spin_cases({"aaaa"}));

    //    tmpL2.block("aaaa")("abcd") = reference.L2aa()("abcd");
    //    tmpL2.block("aAaA")("abcd") = reference.L2ab()("abcd");
    //    tmpL2.block("AAAA")("abcd") = reference.L2bb()("abcd");

    //    L2["pqrs"] = Ua["ap"] * Ua["bq"] * tmpL2["abcd"] * Ua["cr"] * Ua["ds"];
    //    L2["pQrS"] = Ua["ap"] * Ua["BQ"] * tmpL2["aBcD"] * Ua["cr"] * Ua["DS"];
    //    L2["PQRS"] = Ua["AP"] * Ua["BQ"] * tmpL2["ABCD"] * Ua["CR"] * Ua["DS"];

    //    ambit::BlockedTensor g2 = BlockedTensor::build(CoreTensor, "g2", spin_cases({"aaaa"}));
    //    ambit::BlockedTensor tmpg2 = BlockedTensor::build(CoreTensor, "tmpg2",
    //    spin_cases({"aaaa"}));

    //    tmpg2.block("aaaa")("abcd") = reference.g2aa()("abcd");
    //    tmpg2.block("aAaA")("abcd") = reference.g2ab()("abcd");
    //    tmpg2.block("AAAA")("abcd") = reference.g2bb()("abcd");

    //    g2["pqrs"] = Ua["ap"] * Ua["bq"] * tmpg2["abcd"] * Ua["cr"] * Ua["ds"];
    //    g2["pQrS"] = Ua["ap"] * Ua["BQ"] * tmpg2["aBcD"] * Ua["cr"] * Ua["DS"];
    //    g2["PQRS"] = Ua["AP"] * Ua["BQ"] * tmpg2["ABCD"] * Ua["CR"] * Ua["DS"];

    // Transform 3 cumulants
    ambit::Tensor L3aaa0 = reference.L3aaa();
    ambit::Tensor L3aab0 = reference.L3aab();
    ambit::Tensor L3abb0 = reference.L3abb();
    ambit::Tensor L3bbb0 = reference.L3bbb();

    ambit::Tensor L3aaaT =
        ambit::Tensor::build(ambit::CoreTensor, "Transformed L3aaa", std::vector<size_t>(6, nact_));
    ambit::Tensor L3aabT =
        ambit::Tensor::build(ambit::CoreTensor, "Transformed L3aab", std::vector<size_t>(6, nact_));
    ambit::Tensor L3abbT =
        ambit::Tensor::build(ambit::CoreTensor, "Transformed L3abb", std::vector<size_t>(6, nact_));
    ambit::Tensor L3bbbT =
        ambit::Tensor::build(ambit::CoreTensor, "Transformed L3bbb", std::vector<size_t>(6, nact_));

    L3aaaT("pqrstu") =
        Ua("ap") * Ua("bq") * Ua("cr") * L3aaa0("abcijk") * Ua("is") * Ua("jt") * Ua("ku");
    L3aabT("pqRstU") =
        Ua("ap") * Ua("bq") * Ub("CR") * L3aab0("abCijK") * Ua("is") * Ua("jt") * Ua("KU");
    L3abbT("pQRsTU") =
        Ua("ap") * Ub("BQ") * Ub("CR") * L3abb0("aBCiJK") * Ua("is") * Ub("JT") * Ua("KU");
    L3bbbT("PQRSTU") =
        Ub("AP") * Ub("BQ") * Ub("CR") * L3bbb0("ABCIJK") * Ub("IS") * Ub("JT") * Ua("KU");

    reference.set_L3aaa(L3aaaT);
    reference.set_L3aab(L3aabT);
    reference.set_L3abb(L3abbT);
    reference.set_L3bbb(L3bbbT);
    outfile->Printf("\n    Transformed 3 cumulants.");

    //    ambit::BlockedTensor tmpL3 = BlockedTensor::build(CoreTensor, "Gamma3",
    //    spin_cases({"aaaaaa"}));
    //    ambit::BlockedTensor L3 = BlockedTensor::build(CoreTensor, "L3", spin_cases({"aaaaaa"}));

    //    tmpL3.block("aaaaaa")("pqrstu") = reference.L3aaa()("pqrstu");
    //    tmpL3.block("aaAaaA")("pqrstu") = reference.L3aab()("pqrstu");
    //    tmpL3.block("aAAaAA")("pqrstu") = reference.L3abb()("pqrstu");
    //    tmpL3.block("AAAAAA")("pqrstu") = reference.L3bbb()("pqrstu");

    //    L3["pqrstu"] = Ua["ap"] * Ua["bq"] * Ua["cr"] * tmpL3["abcijk"] * Ua["is"] * Ua["jt"] *
    //    Ua["ku"];
    //    L3["pqRstU"] = Ua["ap"] * Ua["bq"] * Ua["CR"] * tmpL3["abCijK"] * Ua["is"] * Ua["jt"] *
    //    Ua["KU"];
    //    L3["pQRsTU"] = Ua["ap"] * Ua["BQ"] * Ua["CR"] * tmpL3["aBCiJK"] * Ua["is"] * Ua["JT"] *
    //    Ua["KU"];
    //    L3["PQRSTU"] = Ua["AP"] * Ua["BQ"] * Ua["CR"] * tmpL3["ABCIJK"] * Ua["IS"] * Ua["JT"] *
    //    Ua["KU"];

    //    // Recompute the energy

    //    // Update the reference
    //    reference.set_L1a(gamma1.block("aa"));
    //    reference.set_L1b(gamma1.block("AA"));

    //    reference.set_L2aa(L2.block("aaaa"));
    //    reference.set_L2ab(L2.block("aAaA"));
    //    reference.set_L2bb(L2.block("AAAA"));

    //    reference.set_g2aa(g2.block("aaaa"));
    //    reference.set_g2ab(g2.block("aAaA"));
    //    reference.set_g2bb(g2.block("AAAA"));

    //    reference.set_L3aaa(L3.block("aaaaaa"));
    //    reference.set_L3aab(L3.block("aaAaaA"));
    //    reference.set_L3abb(L3.block("aAAaAA"));
    //    reference.set_L3bbb(L3.block("AAAAAA"));
}
}
} // End Namespaces
