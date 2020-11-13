/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2020 by its authors (see COPYING, COPYING.LESSER, AUTHORS).
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

#include <cmath>
#include <numeric>

#include "psi4/libpsi4util/process.h"
#include "psi4/libmints/basisset.h"
#include "psi4/libmints/matrix.h"
#include "psi4/libmints/vector.h"
#include "psi4/libmints/wavefunction.h"
#include "psi4/libqt/qt.h"

#include "base_classes/mo_space_info.h"

#ifdef HAVE_GA
#include <ga.h>
#include <macdecls.h>
#endif

#include "helpers/blockedtensorfactory.h"
#include "helpers/printing.h"
#include "helpers/timer.h"
#include "diskdf_integrals.h"

using namespace ambit;
using namespace psi;

namespace forte {

DISKDFIntegrals::DISKDFIntegrals(std::shared_ptr<ForteOptions> options,
                                 std::shared_ptr<psi::Wavefunction> ref_wfn,
                                 std::shared_ptr<MOSpaceInfo> mo_space_info,
                                 IntegralSpinRestriction restricted)
    : Psi4Integrals(options, ref_wfn, mo_space_info, DiskDF, restricted) {
    initialize();
}

void DISKDFIntegrals::initialize() {
    print_info();
    local_timer int_timer;

    int my_proc = 0;
#ifdef HAVE_GA
    my_proc = GA_Nodeid();
#endif
    if (my_proc == 0) {
        gather_integrals();
        freeze_core_orbitals();
        print_timing("disk-based density-fitted integrals", int_timer.get());
    }
}

double DISKDFIntegrals::aptei_aa(size_t p, size_t q, size_t r, size_t s) {
    size_t pn, qn, rn, sn;

    if (frzcpi_.sum() > 0 && ncmo_ == aptei_idx_) {
        pn = cmotomo_[p];
        qn = cmotomo_[q];
        rn = cmotomo_[r];
        sn = cmotomo_[s];
    } else {
        pn = p;
        qn = q;
        rn = r;
        sn = s;
    }

    std::vector<size_t> A_range = {0, nthree_};
    std::vector<size_t> p_range = {pn, pn + 1};
    std::vector<size_t> q_range = {qn, qn + 1};
    std::vector<size_t> r_range = {rn, rn + 1};
    std::vector<size_t> s_range = {sn, sn + 1};

    double vpqrsalphaC = 0.0;
    double vpqrsalphaE = 0.0;

    std::shared_ptr<psi::Matrix> B1(new psi::Matrix(1, nthree_));
    std::shared_ptr<psi::Matrix> B2(new psi::Matrix(1, nthree_));

    df_->fill_tensor("B", B1, A_range, p_range, r_range);
    df_->fill_tensor("B", B2, A_range, q_range, s_range);

    vpqrsalphaC = B1->vector_dot(B2);

    B1->zero();
    B2->zero();

    df_->fill_tensor("B", B1, A_range, p_range, s_range);
    df_->fill_tensor("B", B2, A_range, q_range, r_range);

    vpqrsalphaE = B1->vector_dot(B2);

    return (vpqrsalphaC - vpqrsalphaE);
}

double DISKDFIntegrals::aptei_ab(size_t p, size_t q, size_t r, size_t s) {
    size_t pn, qn, rn, sn;
    if (frzcpi_.sum() > 0 && ncmo_ == aptei_idx_) {
        pn = cmotomo_[p];
        qn = cmotomo_[q];
        rn = cmotomo_[r];
        sn = cmotomo_[s];
    } else {
        pn = p;
        qn = q;
        rn = r;
        sn = s;
    }
    std::vector<size_t> A_range = {0, nthree_};
    std::vector<size_t> p_range = {pn, pn + 1};
    std::vector<size_t> q_range = {qn, qn + 1};
    std::vector<size_t> r_range = {rn, rn + 1};
    std::vector<size_t> s_range = {sn, sn + 1};

    double vpqrsalphaC = 0.0;
    std::shared_ptr<psi::Matrix> B1(new psi::Matrix(1, nthree_));
    std::shared_ptr<psi::Matrix> B2(new psi::Matrix(1, nthree_));

    df_->fill_tensor("B", B1, A_range, p_range, r_range);
    df_->fill_tensor("B", B2, A_range, q_range, s_range);

    vpqrsalphaC = B1->vector_dot(B2);

    return (vpqrsalphaC);
}

double DISKDFIntegrals::aptei_bb(size_t p, size_t q, size_t r, size_t s) {
    size_t pn, qn, rn, sn;

    if (frzcpi_.sum() > 0 && ncmo_ == aptei_idx_) {
        pn = cmotomo_[p];
        qn = cmotomo_[q];
        rn = cmotomo_[r];
        sn = cmotomo_[s];
    } else {
        pn = p;
        qn = q;
        rn = r;
        sn = s;
    }
    std::vector<size_t> A_range = {0, nthree_};
    std::vector<size_t> p_range = {pn, pn + 1};
    std::vector<size_t> q_range = {qn, qn + 1};
    std::vector<size_t> r_range = {rn, rn + 1};
    std::vector<size_t> s_range = {sn, sn + 1};

    double vpqrsalphaC = 0.0;
    double vpqrsalphaE = 0.0;

    std::shared_ptr<psi::Matrix> B1(new psi::Matrix(1, nthree_));
    std::shared_ptr<psi::Matrix> B2(new psi::Matrix(1, nthree_));

    df_->fill_tensor("B", B1, A_range, p_range, r_range);
    df_->fill_tensor("B", B2, A_range, q_range, s_range);

    vpqrsalphaC = B1->vector_dot(B2);

    B1->zero();
    B2->zero();

    df_->fill_tensor("B", B1, A_range, p_range, s_range);
    df_->fill_tensor("B", B2, A_range, q_range, r_range);

    vpqrsalphaE = B1->vector_dot(B2);

    return (vpqrsalphaC - vpqrsalphaE);
}

ambit::Tensor
DISKDFIntegrals::aptei_aa_block(const std::vector<size_t>& p, const std::vector<size_t>& q,
                                const std::vector<size_t>& r, const std::vector<size_t>& s)

{
    ambit::Tensor ThreeIntpr =
        ambit::Tensor::build(tensor_type_, "ThreeInt", {nthree_, p.size(), r.size()});
    ambit::Tensor ThreeIntqs =
        ambit::Tensor::build(tensor_type_, "ThreeInt", {nthree_, q.size(), s.size()});
    std::vector<size_t> Avec(nthree_);
    std::iota(Avec.begin(), Avec.end(), 0);

    ThreeIntpr = three_integral_block(Avec, p, r);
    ThreeIntqs = three_integral_block(Avec, q, s);

    ambit::Tensor ReturnTensor =
        ambit::Tensor::build(tensor_type_, "Return", {p.size(), q.size(), r.size(), s.size()});
    ReturnTensor("p,q,r,s") = ThreeIntpr("A,p,r") * ThreeIntqs("A,q,s");

    /// If p != q != r !=s need to form the Exchange part separately
    if (r != s) {
        ambit::Tensor ThreeIntpsK =
            ambit::Tensor::build(tensor_type_, "ThreeIntK", {nthree_, p.size(), s.size()});
        ambit::Tensor ThreeIntqrK =
            ambit::Tensor::build(tensor_type_, "ThreeIntK", {nthree_, q.size(), r.size()});
        ThreeIntpsK = three_integral_block(Avec, p, s);
        ThreeIntqrK = three_integral_block(Avec, q, r);
        ReturnTensor("p, q, r, s") -= ThreeIntpsK("A, p, s") * ThreeIntqrK("A, q, r");
    } else {
        ReturnTensor("p,q,r,s") -= ThreeIntpr("A,p,s") * ThreeIntqs("A,q,r");
    }

    return ReturnTensor;
}

ambit::Tensor DISKDFIntegrals::aptei_ab_block(const std::vector<size_t>& p,
                                              const std::vector<size_t>& q,
                                              const std::vector<size_t>& r,
                                              const std::vector<size_t>& s) {
    ambit::Tensor ThreeIntpr =
        ambit::Tensor::build(tensor_type_, "ThreeInt", {nthree_, p.size(), r.size()});
    ambit::Tensor ThreeIntqs =
        ambit::Tensor::build(tensor_type_, "ThreeInt", {nthree_, q.size(), s.size()});
    std::vector<size_t> Avec(nthree_);
    std::iota(Avec.begin(), Avec.end(), 0);

    ThreeIntpr = three_integral_block(Avec, p, r);
    ThreeIntqs = three_integral_block(Avec, q, s);

    ambit::Tensor ReturnTensor =
        ambit::Tensor::build(tensor_type_, "Return", {p.size(), q.size(), r.size(), s.size()});
    ReturnTensor("p,q,r,s") = ThreeIntpr("A,p,r") * ThreeIntqs("A,q,s");

    return ReturnTensor;
}

ambit::Tensor DISKDFIntegrals::aptei_bb_block(const std::vector<size_t>& p,
                                              const std::vector<size_t>& q,
                                              const std::vector<size_t>& r,
                                              const std::vector<size_t>& s) {
    ambit::Tensor ThreeIntpr =
        ambit::Tensor::build(tensor_type_, "ThreeInt", {nthree_, p.size(), r.size()});
    ambit::Tensor ThreeIntqs =
        ambit::Tensor::build(tensor_type_, "ThreeInt", {nthree_, q.size(), s.size()});
    std::vector<size_t> Avec(nthree_);
    std::iota(Avec.begin(), Avec.end(), 0);

    ThreeIntpr = three_integral_block(Avec, p, r);
    ThreeIntqs = three_integral_block(Avec, q, s);

    ambit::Tensor ReturnTensor =
        ambit::Tensor::build(tensor_type_, "Return", {p.size(), q.size(), r.size(), s.size()});
    ReturnTensor("p,q,r,s") = ThreeIntpr("A,p,r") * ThreeIntqs("A,q,s");

    /// If p != q != r !=s need to form the Exchane part separately
    if (r != s) {
        ambit::Tensor ThreeIntpsK =
            ambit::Tensor::build(tensor_type_, "ThreeIntK", {nthree_, p.size(), s.size()});
        ambit::Tensor ThreeIntqrK =
            ambit::Tensor::build(tensor_type_, "ThreeIntK", {nthree_, q.size(), r.size()});
        ThreeIntpsK = three_integral_block(Avec, p, s);
        ThreeIntqrK = three_integral_block(Avec, q, r);
        ReturnTensor("p, q, r, s") -= ThreeIntpsK("A, p, s") * ThreeIntqrK("A, q, r");
    } else {
        ReturnTensor("p,q,r,s") -= ThreeIntpr("A,p,s") * ThreeIntqs("A,q,r");
    }
    return ReturnTensor;
}

double** DISKDFIntegrals::three_integral_pointer() { return (ThreeIntegral_->pointer()); }

ambit::Tensor DISKDFIntegrals::three_integral_block(const std::vector<size_t>& A,
                                                    const std::vector<size_t>& p,
                                                    const std::vector<size_t>& q) {
    ambit::Tensor ReturnTensor =
        ambit::Tensor::build(tensor_type_, "Return", {A.size(), p.size(), q.size()});
    std::vector<double>& ReturnTensorV = ReturnTensor.data();

    bool frozen_core = false;

    // Take care of frozen orbitals
    if (frzcpi_.sum() && aptei_idx_ == ncmo_) {
        frozen_core = true;
    }

    size_t pn, qn;
    if (nthree_ == A.size()) {
        std::vector<std::shared_ptr<psi::Matrix>> p_by_Aq;
        for (auto p_block : p) {
            if (frozen_core) {
                pn = cmotomo_[p_block];
            } else {
                pn = p_block;
            }

            std::shared_ptr<psi::Matrix> Aq(new psi::Matrix("Aq", nthree_, nmo_));

            std::vector<size_t> A_range = {A[0], A.back() + 1};
            std::vector<size_t> p_range = {pn, pn + 1};
            std::vector<size_t> q_range = {0, nmo_};

            df_->fill_tensor("B", Aq, A_range, p_range, q_range);
            p_by_Aq.push_back(Aq);
        }
        if (frozen_core) {
            ReturnTensor.iterate([&](const std::vector<size_t>& i, double& value) {
                value = p_by_Aq[i[1]]->get(A[i[0]], cmotomo_[q[i[2]]]);
            });
        } else {
            ReturnTensor.iterate([&](const std::vector<size_t>& i, double& value) {
                value = p_by_Aq[i[1]]->get(A[i[0]], q[i[2]]);
            });
        }
    } else {
        // If user wants blocking in A
        pn = 0;
        qn = 0;
        // If p and q are not just nmo_, this map corrects that.
        // If orbital 0, 5, 10, 15 is frozen, this corresponds to 0, 1, 2, 3.
        // This map says p_map[5] = 1.
        // Used in correct ordering for the tensor.
        std::map<size_t, size_t> p_map;
        std::map<size_t, size_t> q_map;

        int p_idx = 0;
        int q_idx = 0;
        for (size_t p_block : p) {
            p_map[p_block] = p_idx;
            p_idx++;
        }
        for (size_t q_block : q) {
            q_map[q_block] = q_idx;
            q_idx++;
        }
        for (size_t p_block : p) {
            pn = frozen_core ? cmotomo_[p_block] : p_block;

            for (size_t q_block : q) {
                qn = frozen_core ? cmotomo_[q_block] : q_block;

                std::vector<size_t> A_range = {A[0], A.size()};
                std::vector<size_t> p_range = {pn, pn};
                std::vector<size_t> q_range = {qn, qn};

                double* A_chunk = nullptr;

                df_->fill_tensor("B", A_chunk, A_range, p_range, q_range);

                for (size_t a = 0; a < A.size(); a++) {
                    // Weird way the tensor is formatted
                    // Fill the tensor for every chunk of A
                    ReturnTensorV[a * p.size() * q.size() + p_map[p_block] * q.size() +
                                  q_map[q_block]] = A_chunk[a];
                }
                delete[] A_chunk;
            }
        }
    }
    return ReturnTensor;
}

void DISKDFIntegrals::set_tei(size_t, size_t, size_t, size_t, double, bool, bool) {
    outfile->Printf("\n  DISKDFIntegrals::set_tei : DISKDF integrals are read only");
    throw psi::PSIEXCEPTION("DISKDFIntegrals::set_tei : DISKDF integrals are read only");
}

void DISKDFIntegrals::gather_integrals() {
    outfile->Printf("\n Computing Density fitted integrals \n");

    std::shared_ptr<psi::BasisSet> primary = wfn_->basisset();
    std::shared_ptr<psi::BasisSet> auxiliary = wfn_->get_basisset("DF_BASIS_MP2");

    size_t nprim = primary->nbf();
    size_t naux = auxiliary->nbf();
    nthree_ = naux;
    outfile->Printf("\n  Number of auxiliary basis functions:  %u", naux);
    outfile->Printf("\n  Need %8.6f GB to store DF integrals\n",
                    (nprim * nprim * naux * sizeof(double) / 1073741824.0));
    int_mem_ = (nprim * nprim * naux * sizeof(double));

    psi::Dimension nsopi_ = wfn_->nsopi();
    std::shared_ptr<psi::Matrix> aotoso = wfn_->aotoso();
    std::shared_ptr<psi::Matrix> Ca = wfn_->Ca();
    std::shared_ptr<psi::Matrix> Ca_ao(new psi::Matrix("Ca_ao", nso_, nmopi_.sum()));

    // Transform from the SO to the AO basis
    for (int h = 0, index = 0; h < nirrep_; ++h) {
        for (int i = 0; i < nmopi_[h]; ++i) {
            size_t nao = nso_;
            size_t nso = nsopi_[h];

            if (!nso)
                continue;

            C_DGEMV('N', nao, nso, 1.0, aotoso->pointer(h)[0], nso, &Ca->pointer(h)[0][i],
                    nmopi_[h], 0.0, &Ca_ao->pointer()[0][index], nmopi_.sum());

            index += 1;
        }
    }

    // B_{pq}^Q -> MO without frozen core

    // Constructs the DF function
    // I used this version of build as this doesn't build all the apces and
    // assume a RHF/UHF reference
    df_ = std::make_shared<psi::DFHelper>(primary, auxiliary);
    df_->set_memory(psi::Process::environment.get_memory() / sizeof(double));
    df_->initialize();
    df_->set_MO_core(false);
    // set_C clears all the orbital spaces, so this creates the space
    // This space creates the total nmo_.
    // This assumes that everything is correlated.
    df_->add_space("ALL", Ca_ao);
    // Does not add the pair_space, but says which one is should use
    df_->add_transformation("B", "ALL", "ALL", "Qpq");

    // Finally computes the df integrals
    // Does the timings also
    local_timer timer;
    outfile->Printf("\n  Computing DF Integrals");
    df_->transform();
    print_timing("computing density-fitted integrals", timer.get());
}

void DISKDFIntegrals::resort_integrals_after_freezing() {
    local_timer resort_integrals;
    outfile->Printf("\n  Resorting integrals after freezing core.");

    // Create an array that maps the CMOs to the MOs (cmo2mo).
    // resort_three(ThreeIntegral_,cmo2mo);

    print_timing("resorting integrals", resort_integrals.get());
}

ambit::Tensor DISKDFIntegrals::three_integral_block_two_index(const std::vector<size_t>& A,
                                                              size_t p,
                                                              const std::vector<size_t>& q) {

    ambit::Tensor ReturnTensor = ambit::Tensor::build(tensor_type_, "Return", {A.size(), q.size()});

    size_t p_max, p_min;
    bool frozen_core = false;
    if (frzcpi_.sum() && aptei_idx_ == ncmo_) {
        frozen_core = true;
        p_min = cmotomo_[p];
        p_max = p_min + 1;
    } else {
        p_min = p;
        p_max = p_min + 1;
    }

    if (nthree_ == A.size()) {

        std::vector<size_t> arange = {0, nthree_};
        std::vector<size_t> qrange = {0, nmo_};
        std::vector<size_t> prange = {p_min, p_max};

        std::shared_ptr<psi::Matrix> Aq(new psi::Matrix("Aq", nthree_, nmo_));
        df_->fill_tensor("B", Aq, arange, prange, qrange);

        if (frozen_core) {
            ReturnTensor.iterate([&](const std::vector<size_t>& i, double& value) {
                value = Aq->get(A[i[0]], cmotomo_[q[i[1]]]);
            });
        } else {
            ReturnTensor.iterate([&](const std::vector<size_t>& i, double& value) {
                value = Aq->get(A[i[0]], q[i[1]]);
            });
        }
    } else {
        outfile->Printf("\n Not implemened for variable size in A");
        throw psi::PSIEXCEPTION("Can only use if 2nd parameter is a size_t and A.size==nthree_");
    }
    return ReturnTensor;
}

size_t DISKDFIntegrals::nthree() const { return nthree_; }

} // namespace forte
