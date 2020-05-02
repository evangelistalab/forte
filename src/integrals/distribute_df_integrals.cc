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

#ifdef HAVE_GA

#include <cassert>
#include <cmath>
#include <numeric>

#include "integrals.h"
#include "psi4/libmints/basisset.h"
#include "psi4/libmints/matrix.h"
#include "psi4/libqt/qt.h"

#ifdef _OPENMP
#include <omp.h>
#else
#define omp_get_max_threads() 1
#define omp_get_thread_num() 0
#define omp_get_num_threads() 1
#endif

#ifdef HAVE_GA
#include "paralleldfmo.h"
#include "helpers/timer.h"
#include <ga.h>
#include <macdecls.h>
#include <mpi.h>
#endif

#include "helpers/blockedtensorfactory.h"

using namespace ambit;

namespace forte {

DistDFIntegrals::DistDFIntegrals(std::shared_ptr<ForteOptions> options, std::shared_ptr<psi::Wavefunction> ref_wfn,
                                 IntegralSpinRestriction restricted,
                                 IntegralFrozenCore resort_frozen_core,
                                 std::shared_ptr<MOSpaceInfo> mo_space_info)
    : ForteIntegrals(psi::Options, ref_wfn, restricted, resort_frozen_core, mo_space_info) {

    wfn_ = ref_wfn;

    integral_type_ = DistDF;
    outfile->Printf("\n  DistDFIntegrals overall time with %d MPI Process and %d threads",
                    GA_Nnodes(), omp_get_max_threads());
    local_timer DFInt;
#define omp_get_max_threads() 1
    allocate();

    int my_proc = 0;
#ifdef HAVE_GA
    my_proc = GA_Nodeid();
#endif

    gather_integrals();
    if (my_proc == 0) {
        test_distributed_integrals();
    }
    freeze_core_orbitals();

    outfile->Printf("\n  DistDFIntegrals take %15.8f s", DFInt.get());
}

void DistDFIntegrals::test_distributed_integrals() {
    outfile->Printf("\n Computing Density fitted integrals \n");

    std::shared_ptr<psi::BasisSet> primary = wfn_->basisset();
    if (options_.get_str("DF_BASIS_MP2").length() == 0) {
        outfile->Printf("\n Please set a DF_BASIS_MP2 option to a specified "
                        "auxiliary basis set");
        throw psi::PSIEXCEPTION("Select a DF_BASIS_MP2 for use with DFIntegrals");
    }

    // std::shared_ptr<psi::BasisSet> auxiliary =
    // psi::BasisSet::pyconstruct_orbital(primary->molecule(),
    // "DF_BASIS_MP2",options_.get_str("DF_BASIS_MP2"));

    std::shared_ptr<psi::BasisSet> auxiliary = wfn_->get_basisset("DF_BASIS_MP2");

    size_t nprim = primary->nbf();
    size_t naux = auxiliary->nbf();
    nthree_ = naux;
    outfile->Printf("\n Number of auxiliary basis functions:  %u", naux);
    outfile->Printf("\n Need %8.6f GB to store DF integrals\n",
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
    std::shared_ptr<DFERI> df = DFERI::build(primary, auxiliary, options_);

    // Pushes a C matrix that is ordered in pitzer ordering
    // into the C_matrix object
    df->set_C(Ca_ao);
    // set_C clears all the orbital spaces, so this creates the space
    // This space creates the total nmo_.
    // This assumes that everything is correlated.
    df->add_space("ALL", 0, nmo_);
    // Does not add the pair_space, but says which one is should use
    df->add_pair_space("B", "ALL", "ALL");
    df->set_memory(psi::Process::environment.get_memory() / 8L);

    // Finally computes the df integrals
    // Does the timings also
    local_timer timer;
    std::string str = "Computing DF Integrals";
    outfile->Printf("\n    %-36s ...", str.c_str());
    df->compute();
    outfile->Printf("...Done. Timing %15.6f s", timer.get());

    std::shared_ptr<Tensor> B = df->ints()["B"];
    B_ = B;
    df.reset();
    int dim[2], chunk[2];
    dim[0] = nthree_;
    dim[1] = nmo_ * nmo_;
    chunk[0] = -1;
    chunk[1] = nmo_ * nmo_;

    int MY_DF = NGA_Create(C_DBL, 2, dim, (char*)"DistributedDF", chunk);
    if (MY_DF == 0)
        GA_Error("DistributedDF failed on creating the tensor", 1);
    // GA_Print_distribution(MY_DF);
    // GA_Print_distribution(DistDF_ga_);
    int my_proc = GA_Nodeid();
    int num_proc = GA_Nnodes();
    if (my_proc == 0) {
        for (int iproc = 0; iproc < num_proc; iproc++) {
            int begin_offset[2];
            int end_offset[2];
            int stride[1];
            stride[0] = nmo_ * nmo_;
            NGA_Distribution(MY_DF, iproc, begin_offset, end_offset);
            std::vector<int> begin_offset_vec = {begin_offset[0], begin_offset[1]};
            std::vector<int> end_offset_vec = {end_offset[0], end_offset[1]};
            ambit::Tensor B_per_process = read_integral_chunk(B_, begin_offset_vec, end_offset_vec);
            NGA_Put(MY_DF, begin_offset, end_offset, &(B_per_process.data()[0]), stride);
        }
    }
    double positive_one = 1.0;
    double negative_one = -1.0;
    GA_Add(&positive_one, MY_DF, &negative_one, DistDF_ga_, MY_DF);
    std::vector<double> my_df_norm(2, 0.0);
    GA_Norm_infinity(MY_DF, &my_df_norm[0]);
    for (auto my_norm : my_df_norm) {
        outfile->Printf("\n ||SERIAL_DF - DistDF||_{\infinity} = %4.16f", my_norm);
        if (my_norm > 1e-4)
            throw psi::PSIEXCEPTION("DF and DistDF do not agree");
    }
    /// Test getting entire three_integral_object
    std::vector<size_t> Avec(nthree_, 0);
    std::vector<size_t> p(nmo_, 0);
    std::iota(Avec.begin(), Avec.end(), 0);
    std::iota(p.begin(), p.end(), 0);

    /// Test whether DFIntegrals is same as DistributedDF
    ForteIntegrals* test_int =
        new DFIntegrals(options_, wfn_, UnrestrictedMOs, RemoveFrozenMOs, mo_space_info_);
    ambit::Tensor entire_b_df = test_int->three_integral_block(Avec, p, p);
    ambit::Tensor entire_b_dist = three_integral_block(Avec, p, p);
    entire_b_df("Q, p, q") -= entire_b_dist("Q, p, q");
    outfile->Printf("\n Test read entire A: %8.8f", entire_b_df.norm(2.0));
    if (entire_b_df.norm(2.0) > 1.0e-6)
        throw psi::PSIEXCEPTION("three_integral_block for all integrals does not work");

    /// Test partial nthree
    int block = nthree_ / 2;
    std::vector<size_t> Apartial(block, 0);
    std::iota(Apartial.begin(), Apartial.end(), 0);
    ambit::Tensor partial_b_df = test_int->three_integral_block(Apartial, p, p);
    ambit::Tensor partial_b_dist = three_integral_block(Apartial, p, p);
    partial_b_df("Q, p, q") -= partial_b_dist("Q, p, q");
    outfile->Printf("\n Test partial A: %8.8f", partial_b_df.norm(2.0));
    if (partial_b_df.norm(2.0) > 1.0e-6)
        throw psi::PSIEXCEPTION("three_integral_block for partial nthree integrals does not work");
    /// Test rdocc

    ambit::Tensor b_zero_df = test_int->three_integral_block(Avec, {0}, {0});
    ambit::Tensor b_zero_dist = three_integral_block(Avec, {0}, {0});
    b_zero_df("Q, p, q") -= b_zero_dist("Q, p, q");
    outfile->Printf("\n Test Q_00: %8.8f", b_zero_df.norm(2.0));
    if (b_zero_df.norm(2.0) > 1.0e-6)
        throw psi::PSIEXCEPTION("three_integral_block for B_00");

    ambit::Tensor b_one_df = test_int->three_integral_block(Avec, {1}, {1});
    ambit::Tensor b_one_dist = three_integral_block(Avec, {1}, {1});
    b_one_df("Q, p, q") -= b_one_dist("Q, p, q");
    outfile->Printf("\n Test Q_{11}: %8.8f", b_one_df.norm(2.0));
    if (b_one_df.norm(2.0) > 1.0e-6)
        throw psi::PSIEXCEPTION("three_integral_block for B_11");

    auto rdocc = mo_space_info_->corr_absolute_mo("RESTRICTED_DOCC");
    auto active = mo_space_info_->corr_absolute_mo("ACTIVE");
    ambit::Tensor b_mn_df = test_int->three_integral_block(Avec, rdocc, rdocc);
    ambit::Tensor b_mn_dist = three_integral_block(Avec, rdocc, rdocc);
    b_mn_df("Q, p, q") -= b_mn_dist("Q, p, q");
    outfile->Printf("\n Test Q_mn: %8.8f", b_mn_df.norm(2.0));
    if (b_mn_df.norm(2.0) > 1.0e-6)
        throw psi::PSIEXCEPTION("three_integral_block for B_mn");

    ambit::Tensor b_mu_df = test_int->three_integral_block(Avec, rdocc, active);
    ambit::Tensor b_mu_dist = three_integral_block(Avec, rdocc, active);
    b_mu_df("Q, p, q") -= b_mu_dist("Q, p, q");
    outfile->Printf("\n Test Q_{mu}: %8.8f", b_mu_df.norm(2.0));
    if (b_mu_df.norm(2.0) > 1.0e-6)
        throw psi::PSIEXCEPTION("three_integral_block for B_11");

    delete test_int;
}

ambit::Tensor DistDFIntegrals::read_integral_chunk(std::shared_ptr<Tensor>& B, std::vector<int>& lo,
                                                   std::vector<int>& hi) {
    assert(lo.size() == 2);
    assert(hi.size() == 2);
    /// This tells what block of naux is on the processor that calls this
    size_t naux_block_size = hi[0] - lo[0] + 1;
    ambit::Tensor ReturnTensor =
        ambit::Tensor::build(tensor_type_, "Return", {naux_block_size, nmo_, nmo_});
    std::vector<double>& ReturnTensorV = ReturnTensor.data();
    /// Allocate a vector that is blocked via naux dimension
    std::vector<size_t> naux(naux_block_size, 0);
    /// Fill the tensor starting from start of block and end once it hits end
    std::iota(naux.begin(), naux.end(), lo[0]);
    // If p and q are not just nmo_, this map corrects that.
    // If orbital 0, 5, 10, 15 is frozen, this corresponds to 0, 1, 2, 3.
    // This map says p_map[5] = 1.
    // Used in correct ordering for the tensor.

    std::vector<size_t> all_mos = mo_space_info_->corr_absolute_mo("ALL");
    int p_idx = 0;
    int q_idx = 0;
    std::vector<size_t> p_map(nmo_);
    std::vector<size_t> q_map(nmo_);

    // for(size_t p_block : all_mos)
    //{
    //    p_map[p_block] = p_idx;
    //    p_idx++;
    //}
    // for(size_t q_block : all_mos)
    //{
    //    q_map[q_block] = q_idx;
    //    q_idx++;
    //}

    for (size_t p_block = 0; p_block < nmo_; p_block++) {
        size_t pn = p_block;
        for (size_t q_block = 0; q_block < nmo_; q_block++) {
            size_t qn = q_block;
            double* A_chunk = new double[naux_block_size];
            size_t offset = pn * nthree_ * nmo_ + qn * nthree_ + naux[0];
            fseek(B_->file_pointer(), offset * sizeof(double), SEEK_SET);
            fread(&(A_chunk[0]), sizeof(double), naux_block_size, B_->file_pointer());
            for (size_t a = 0; a < naux_block_size; a++) {
                // Weird way the tensor is formatted
                // Fill the tensor for every chunk of A
                // ReturnTensorV[a * nmo_ * nmo_ + p_map[p_block] * nmo_ +
                // q_map[q_block]] = A_chunk[a];
                ReturnTensorV[a * nmo_ * nmo_ + p_block * nmo_ + q_block] = A_chunk[a];
            }
            delete[] A_chunk;
        }
    }
    return ReturnTensor;
}
void DistDFIntegrals::deallocate() {
    GA_Destroy(DistDF_ga_);
    // delete[] diagonal_aphys_tei_aa;
    // delete[] diagonal_aphys_tei_ab;
    // delete[] diagonal_aphys_tei_bb;
}
void DistDFIntegrals::allocate() {
    diagonal_aphys_tei_aa = new double[nmo_ * nmo_];
    diagonal_aphys_tei_ab = new double[nmo_ * nmo_];
    diagonal_aphys_tei_bb = new double[nmo_ * nmo_];
}
// double DistDFIntegrals::aptei_aa(size_t p, size_t q, size_t r, size_t s)
//{
//    ambit::Tensor pqrs = aptei_aa_block({p}, {q}, {r}, {s});
//    return pqrs.data()[0];
//}
// double DistDFIntegrals::aptei_ab(size_t p, size_t q, size_t r, size_t s)
//{
//    ambit::Tensor pqrs = aptei_ab_block({p}, {q}, {r}, {s});
//    return pqrs.data()[0];
//}
// double DistDFIntegrals::aptei_bb(size_t p, size_t q, size_t r, size_t s)
//{
//    ambit::Tensor pqrs = aptei_bb_block({p}, {q}, {r}, {s});
//    return pqrs.data()[0];
//}
ambit::Tensor DistDFIntegrals::three_integral_block(const std::vector<size_t>& A,
                                                    const std::vector<size_t>& p,
                                                    const std::vector<size_t>& q) {
    ambit::Tensor ReturnTensor =
        ambit::Tensor::build(tensor_type_, "Return", {A.size(), p.size(), q.size()});
    std::vector<double>& ReturnTensorV = ReturnTensor.data();
    bool frozen_core = false;

    if (frzcpi_.sum() && aptei_idx_ == ncmo_)
        frozen_core = true;

    size_t pn, qn;
    /// DistDF_ga_ is distributed via A dimension.
    /// A lot of logic needs to be done to figure out where information lies
    int subscript_begin[2];
    int subscript_end[2];
    /// If user wants blocking in A
    // GA_Print_distribution(DistDF_ga_);
    if (p.size() == nmo_ && q.size() == nmo_ && A.size() < nthree_) {
        int ld[1];
        ld[0] = nmo_ * nmo_;
        subscript_begin[0] = A[0];
        subscript_begin[1] = 0;
        subscript_end[0] = A[A.size() - 1];
        subscript_end[1] = nmo_ * nmo_ - 1;
        for (int i = 0; i < 2; i++)
            outfile->Printf("\n subscript[%d] = (%d, %d)", i, subscript_begin[i], subscript_end[i]);
        NGA_Get(DistDF_ga_, subscript_begin, subscript_end, &ReturnTensorV[0], ld);
        return ReturnTensor;
    } else if (p.size() == nmo_ && q.size() == nmo_ && A.size() == nthree_) {
        int ld[1];
        ld[0] = nmo_ * nmo_;
        subscript_begin[0] = 0;
        subscript_begin[1] = 0;
        subscript_end[0] = nthree_ - 1;
        subscript_end[1] = nmo_ * nmo_ - 1;
        for (int i = 0; i < 2; i++)
            outfile->Printf("\n subscript[%d] = (%d, %d)", i, subscript_begin[i], subscript_end[i]);
        NGA_Get(DistDF_ga_, subscript_begin, subscript_end, &ReturnTensorV[0], ld);
        return ReturnTensor;
    } else if (A.size() == nthree_ and (p.size() != nmo_ or q.size() != nmo_)) {
        /// DF_Tensor is packed in like (NAUX, p, q) where fast dimension is q.
        /// To read tensor from ga, I will assume in that auxiliary index is
        /// nthree_
        /// Let's say I want (nthree_,1,nmo_), I have to translate that to lo
        /// and hi index.
        /// lo[0] = 0, hi[0] = nthree_ - 1
        /// lo[1] = 1 * nmo_ + 0
        /// hi[1] = 1 * nmo_ + nmo_
        /// This means lo[1] = p[idx] * nmo_ + q[idx_min]
        ///                           hi[1] = p[idx] * nmo_ + q[idx_max]
        /// Few ways to do this:  Read all of q (ignore q vector) for every p
        /// Then put the select bits into the tensor
        /// Slower way:  Read only 1 entry at a time for p and q
        /// No reason to do this.  Maybe if you want a dumb algorithm

        int ld[1];
        ld[0] = nmo_;
        subscript_begin[0] = 0;
        subscript_end[0] = nthree_ - 1;
        std::vector<std::vector<double>> ga_buf_vector;
        for (size_t plist = 0; plist < p.size(); plist++) {
            std::vector<double> ga_buf(nthree_ * nmo_);
            size_t p_index = p[plist];
            subscript_begin[1] = (p_index * nmo_);
            subscript_end[1] = (p_index + 1) * nmo_ - 1;
            for (int i = 0; i < 2; i++)
                outfile->Printf("\n subscript[%d] = (%d, %d)", i, subscript_begin[i],
                                subscript_end[i]);

            NGA_Get(DistDF_ga_, subscript_begin, subscript_end, &ga_buf[0], ld);
            ga_buf_vector.push_back(ga_buf);
            // for(int qlist = 0; qlist < q.size(); qlist++){
            //    size_t q_index = q[qlist];

            //}
            // ReturnTensorV = ga_buf;
        }
        ReturnTensor.iterate([&](const std::vector<size_t>& i, double& value) {
            size_t p_index = p[i[1]];
            size_t q_index = q[i[2]];
            size_t A_index = A[i[0]];
            value = ga_buf_vector[i[1]][A_index * nmo_ + q[i[2]]];
        });

        return ReturnTensor;
    } else if (A.size() != nthree_ and p.size() != nmo_ and q.size() != nmo_) {
    }
}
void DistDFIntegrals::gather_integrals() {
    // std::shared_ptr<psi::BasisSet> auxiliary =
    // psi::BasisSet::pyconstruct_orbital(wfn_->molecule(),
    // "DF_BASIS_MP2",options_.get_str("DF_BASIS_MP2"));
    std::shared_ptr<psi::BasisSet> auxiliary = wfn_->get_basisset("DF_BASIS_MP2");
    std::shared_ptr<psi::Matrix> Ca = wfn_->Ca();
    std::shared_ptr<psi::Matrix> Ca_ao(new psi::Matrix("CA_AO", wfn_->nso(), wfn_->nmo()));
    for (size_t h = 0, index = 0; h < wfn_->nirrep(); ++h) {
        for (size_t i = 0; i < wfn_->nmopi()[h]; ++i) {
            size_t nao = wfn_->nso();
            size_t nso = wfn_->nsopi()[h];

            if (!nso)
                continue;

            C_DGEMV('N', nao, nso, 1.0, wfn_->aotoso()->pointer(h)[0], nso, &Ca->pointer(h)[0][i],
                    wfn_->nmopi()[h], 0.0, &Ca_ao->pointer()[0][index], wfn_->nmopi().sum());

            index += 1;
        }
    }

    ParallelDFMO DFMO = ParallelDFMO(wfn_->basisset(), auxiliary);
    DFMO.set_C(Ca_ao);
    DFMO.compute_integrals();
    DistDF_ga_ = DFMO.Q_PQ();
    int dim[3];
    int chunk[3];
    /// Note:  I assume that we always store all the 3index integrals
    /// Frozen core does not gain any benefits in storage
    /// Accessing elements needs to use absolute index.  Will convert to
    /// absolute in API
    dim[0] = nthree_;
    dim[1] = nmo_;
    dim[2] = nmo_;
    chunk[0] = -1;
    chunk[1] = nmo_;
    chunk[2] = nmo_;
}
void DistDFIntegrals::retransform_integrals() {
    aptei_idx_ = nmo_;
    transform_one_electron_integrals();
    int my_proc = 0;
    my_proc = GA_Nodeid();
    outfile->Printf("\n Integrals are about to be computed.");
    gather_integrals();
    outfile->Printf("\n Integrals are about to be updated.");
    freeze_core_orbitals();
}
} // namespace forte

#endif // HAVE_GA
