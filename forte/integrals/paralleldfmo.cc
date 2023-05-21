/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2023 by its authors (see COPYING, COPYING.LESSER, AUTHORS).
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

#include "psi4/psi4-dec.h"
#include "psi4/psifiles.h"

#include "psi4/lib3index/3index.h"

#include "psi4/libqt/qt.h"

#include "psi4/libmints/integral.h"
#include "psi4/libmints/matrix.h"
#include "psi4/libmints/sieve.h"

#include "helpers/timer.h"

#include "psi4/libfock/jk.h"
#ifdef _OPENMP
#include <omp.h>
#endif
#include "paralleldfmo.h"
#ifdef HAVE_GA
#include <ga.h>
#include <macdecls.h>
#endif

namespace forte {

ParallelDFMO::ParallelDFMO(std::shared_ptr<psi::BasisSet> primary,
                           std::shared_ptr<psi::BasisSet> auxiliary)
    : primary_(primary), auxiliary_(auxiliary) {
    memory_ = psi::Process::environment.get_memory();
}
void ParallelDFMO::compute_integrals() {
    local_timer compute_integrals_time;
    timer_on("DFMO: transform_integrals()");
    transform_integrals();
    printf("\n P%d compute_integrals_time: %8.6f ", GA_Nodeid(), compute_integrals_time.get());
    timer_off("DFMO: transform_integrals()");
}
void ParallelDFMO::transform_integrals() {
    // > Sizing < //

    int nso = primary_->nbf();
    int naux = auxiliary_->nbf();
    nmo_ = Ca_->colspi()[0];

    // > Threading < //

    int nthread = 1;
#ifdef _OPENMP
    nthread = omp_get_max_threads();
#endif

    // > Maximum orbital sizes < //

    size_t max1 = nmo_;
    size_t max12 = max1 * max1;

    // > Row requirements < //

    unsigned long int per_row = 0L;
    // (Q|mn)
    per_row += nso * (unsigned long int)nso;
    // (Q|mi)
    per_row += max1 * (unsigned long int)nso;
    // (Q|ia)
    per_row += max12;

    // > Maximum number of rows < //

    unsigned long int max_rows = (memory_ / per_row);
    // max_rows = 3L * auxiliary_->max_function_per_shell(); // Debug
    if (max_rows < auxiliary_->max_function_per_shell()) {
        throw psi::PSIEXCEPTION("Out of memory in DFERI.");
    }
    max_rows = (max_rows > auxiliary_->nbf() ? auxiliary_->nbf() : max_rows);
    int shell_per_process = 0;
    int shell_start = -1;
    int shell_end = -1;
    /// MPI Environment
    int my_rank = GA_Nodeid();
    int num_proc = GA_Nnodes();

    if (auxiliary_->nbf() == max_rows) {
        shell_per_process = auxiliary_->nshell() / num_proc;
    } else {
        throw psi::PSIEXCEPTION("Have not implemented memory bound df integrals");
    }
    /// Have first proc be from 0 to shell_per_process
    /// Last proc is shell_per_process * my_rank to naux
    if (my_rank != (num_proc - 1)) {
        shell_start = shell_per_process * my_rank;
        shell_end = shell_per_process * (my_rank + 1);
    } else {
        shell_start = shell_per_process * my_rank;
        shell_end = (auxiliary_->nshell() % num_proc == 0 ? shell_per_process * (my_rank + 1)
                                                          : auxiliary_->nshell());
    }

    // printf("\n P%d shell_per_process: %d shell_start:%d  shell_end:%d",
    // my_rank, shell_per_process, shell_start, shell_end);
    // > Shell block assignments < //

    // int fcount = auxiliary_->shell(0).nfunction();
    // std::vector<int> shell_starts(auxiliary_->nshell() + 1, 0);
    // shell_starts[0] = 0;
    // for (int Q = 1; Q < auxiliary_->nshell(); Q++) {
    //    if (fcount + auxiliary_->shell(Q).nfunction() > max_rows) {
    //        shell_starts[Q] = Q;
    //        fcount = auxiliary_->shell(Q).nfunction();
    //    } else {
    //        fcount += auxiliary_->shell(Q).nfunction();
    //    }
    //}
    // shell_starts.push_back(auxiliary_->nshell());

    int function_start = auxiliary_->shell(shell_start).function_index();
    int function_end =
        (shell_end == auxiliary_->nshell() ? auxiliary_->nbf()
                                           : auxiliary_->shell(shell_end).function_index());
    int dims[2];
    int chunk[2];
    dims[0] = naux;
    dims[1] = nso * nso;
    chunk[0] = GA_Nnodes();
    chunk[1] = 1;
    int map[GA_Nnodes() + 1];
    for (int iproc = 0; iproc < GA_Nnodes(); iproc++) {
        int shell_start = 0;
        int shell_end = 0;
        if (iproc != (num_proc - 1)) {
            shell_start = shell_per_process * iproc;
            shell_end = shell_per_process * (iproc + 1);
        } else {
            shell_start = shell_per_process * iproc;
            shell_end = (auxiliary_->nshell() % num_proc == 0 ? shell_per_process * (iproc + 1)
                                                              : auxiliary_->nshell());
        }
        int function_start = auxiliary_->shell(shell_start).function_index();
        int function_end =
            (shell_end == auxiliary_->nshell() ? auxiliary_->nbf()
                                               : auxiliary_->shell(shell_end).function_index());
        map[iproc] = function_start;
        outfile->Printf("\n  P%d shell_start: %d shell_end: %d function_start: "
                        "%d function_end: %d",
                        iproc, shell_start, shell_end, function_start, function_end);
    }
    map[GA_Nnodes()] = 0;
    int Aia_ga = NGA_Create_irreg(C_DBL, 2, dims, (char*)"Aia_temp", chunk, map);
    if (not Aia_ga) {
        throw psi::PSIEXCEPTION("GA failed on creating Aia_ga");
    }
    GA_Q_PQ_ = GA_Duplicate(Aia_ga, (char*)"(Q|pq)");
    if (not GA_Q_PQ_) {
        throw psi::PSIEXCEPTION("GA failed on creating GA_Q_PQ");
    }

    // => ERI Objects <= //

    std::shared_ptr<IntegralFactory> factory(
        new IntegralFactory(auxiliary_, psi::BasisSet::zero_ao_basis_set(), primary_, primary_));
    std::vector<std::shared_ptr<TwoBodyAOInt>> eri;
    for (int thread = 0; thread < nthread; thread++) {
        eri.push_back(std::shared_ptr<TwoBodyAOInt>(factory->eri()));
    }

    // => ERI Sieve <= //

    std::shared_ptr<ERISieve> sieve(new ERISieve(primary_, 1e-10));
    const std::vector<std::pair<int, int>>& shell_pairs = sieve->shell_pairs();
    long int nshell_pairs = (long int)shell_pairs.size();

    // => Temporary Tensors <= //

    // > Three-index buffers < //
    auto Amn = std::make_shared<psi::Matrix>("(A|mn)", max_rows, nso * (unsigned long int)nso);
    auto Ami = std::make_shared<psi::Matrix>("(A|mi)", max_rows, nso * (unsigned long int)max1);
    auto Aia = std::make_shared<psi::Matrix>("(A|ia)", naux, max12);
    double** Amnp = Amn->pointer();
    double** Amip = Ami->pointer();
    double** Aiap = Aia->pointer();

    // > C-matrix weirdness < //

    double** Cp = Ca_->pointer();
    int lda = nmo_;

    //// ==> Master Loop <== //

    int Aia_begin[2];
    int Aia_end[2];
    /// SIMD
    /// shell_start represents the start of shells for this processor
    /// shell_end represents the end of shells for this processor
    /// NOTE:  This code will have terrible load balance (shells do not
    /// correspond to equal number of functions
    local_timer compute_Aia;
    {
        int Pstart = shell_start;
        int Pstop = shell_end;
        int nPshell = Pstop - Pstart;
        int pstart = auxiliary_->shell(Pstart).function_index();
        int pstop = (Pstop == auxiliary_->nshell() ? auxiliary_->nbf()
                                                   : auxiliary_->shell(Pstop).function_index());
        int rows = pstop - pstart;

        // > (Q|mn) ERIs < //

        ::memset((void*)Amnp[0], '\0', sizeof(double) * rows * nso * nso);

#pragma omp parallel for schedule(dynamic) num_threads(nthread)
        for (long int PMN = 0L; PMN < nPshell * nshell_pairs; PMN++) {

            int thread = 0;
#ifdef _OPENMP
            thread = omp_get_thread_num();
#endif

            int P = PMN / nshell_pairs + Pstart;
            int MN = PMN % nshell_pairs;
            std::pair<int, int> pair = shell_pairs[MN];
            int M = pair.first;
            int N = pair.second;

            eri[thread]->compute_shell(P, 0, M, N);

            int nm = primary_->shell(M).nfunction();
            int nn = primary_->shell(N).nfunction();
            int np = auxiliary_->shell(P).nfunction();
            int om = primary_->shell(M).function_index();
            int on = primary_->shell(N).function_index();
            int op = auxiliary_->shell(P).function_index();

            const double* buffer = eri[thread]->buffer();

            for (int p = 0; p < np; p++) {
                for (int m = 0; m < nm; m++) {
                    for (int n = 0; n < nn; n++) {
                        Amnp[p + op - pstart][(m + om) * nso + (n + on)] =
                            Amnp[p + op - pstart][(n + on) * nso + (m + om)] = (*buffer++);
                    }
                }
            }
        }

        // for (int ind1 = 0; ind1 < tasks.size(); ind1++) {

        //    std::string space1 = pair_spaces_[tasks[ind1][0]].first;
        //    int start1 = spaces_[space1].first;
        //    int end1   = spaces_[space1].second;
        //    int n1      = end1 - start1;
        //    double* C1p = &Cp[0][start1];
        int start1 = 0;
        int end1 = nmo_;
        int n1 = end1 - start1;
        double* C1p = &Cp[0][start1];

        C_DGEMM('N', 'N', rows * nso, n1, nso, 1.0, Amnp[0], nso, C1p, lda, 0.0, Amip[0], n1);

        // for (int ind2 = 0; ind2 < tasks[ind1].size(); ind2++) {
        // std::string space2 = pair_spaces_[tasks[ind1][ind2]].second;
        // int start2 = spaces_[space2].first;
        // int end2   = spaces_[space2].second;
        // int n2      = end2 - start2;
        // double* C2p = &Cp[0][start2];
        int n2 = nmo_;
        double* C2p = &Cp[0][0];

        size_t n12 = n1 * (size_t)n2;
        size_t no1 = nso * (size_t)n1;

        // std::string name = tasks[ind1][ind2];
        bool transpose12 = false;

        if (transpose12) {
#pragma omp parallel for num_threads(nthread)
            for (int Q = 0; Q < rows; Q++) {
                C_DGEMM('T', 'N', n2, n1, nso, 1.0, C2p, lda, Amip[0] + Q * no1, n1, 0.0,
                        Aiap[0] + Q * n12, n1);
            }
        } else {
#pragma omp parallel for num_threads(nthread)
            for (int Q = 0; Q < rows; Q++) {
                C_DGEMM('T', 'N', n1, n2, nso, 1.0, Amip[0] + Q * no1, n1, C2p, lda, 0.0,
                        Aiap[0] + Q * n12, n2);
            }
        }

        // Amn->print();
        // Ami->print();
        // Aia->print();

        // std::shared_ptr<Tensor> A = ints_[name + "_temp"];
        // FILE* fh = A->file_pointer();
        // fwrite(Aiap[0],sizeof(double),rows*n12,fh);
        int ld = nmo_ * nmo_;

        NGA_Distribution(Aia_ga, GA_Nodeid(), Aia_begin, Aia_end);
        NGA_Put(Aia_ga, Aia_begin, Aia_end, Aiap[0], &(ld));
        //}
        //}
    }
    printf("\n  P%d Aia took %8.6f s.", GA_Nodeid(), compute_Aia.get());

    local_timer J_one_half_time;
    J_one_half();
    printf("\n  P%d J^({-1/2}} took %8.6f s.", GA_Nodeid(), J_one_half_time.get());

    local_timer GA_DGEMM;
    GA_Dgemm('T', 'N', naux, nmo_ * nmo_, naux, 1.0, GA_J_onehalf_, Aia_ga, 0.0, GA_Q_PQ_);
    printf("\n  P%d DGEMM took %8.6f s.", GA_Nodeid(), GA_DGEMM.get());
    GA_Destroy(GA_J_onehalf_);
    GA_Destroy(Aia_ga);
}
void ParallelDFMO::J_one_half() {
    // Everybody likes them some inverse square root metric, eh?

    int nthread = 1;
#ifdef _OPENMP
    nthread = omp_get_max_threads();
#endif

    int naux = auxiliary_->nbf();

    auto J = std::make_shared<psi::Matrix>("J", naux, naux);
    double** Jp = J->pointer();

    int dims[2];
    int chunk[2];
    dims[0] = naux;
    dims[1] = naux;
    chunk[0] = -1;
    chunk[1] = naux;
    GA_J_onehalf_ = NGA_Create(C_DBL, 2, dims, (char*)"J_1/2", chunk);
    if (not GA_J_onehalf_)
        throw psi::PSIEXCEPTION("Failure in creating J_^(-1/2) in GA");

    // if(GA_Nodeid() == 0)
    {
        std::shared_ptr<IntegralFactory> Jfactory(
            new IntegralFactory(auxiliary_, psi::BasisSet::zero_ao_basis_set(), auxiliary_,
                                psi::BasisSet::zero_ao_basis_set()));
        std::vector<std::shared_ptr<TwoBodyAOInt>> Jeri;
        for (int thread = 0; thread < nthread; thread++) {
            Jeri.push_back(std::shared_ptr<TwoBodyAOInt>(Jfactory->eri()));
        }

        std::vector<std::pair<int, int>> Jpairs;
        for (int M = 0; M < auxiliary_->nshell(); M++) {
            for (int N = 0; N <= M; N++) {
                Jpairs.push_back(std::pair<int, int>(M, N));
            }
        }
        long int num_Jpairs = Jpairs.size();

#pragma omp parallel for schedule(dynamic) num_threads(nthread)
        for (long int PQ = 0L; PQ < num_Jpairs; PQ++) {

            int thread = 0;
#ifdef _OPENMP
            thread = omp_get_thread_num();
#endif

            std::pair<int, int> pair = Jpairs[PQ];
            int P = pair.first;
            int Q = pair.second;

            Jeri[thread]->compute_shell(P, 0, Q, 0);

            int np = auxiliary_->shell(P).nfunction();
            int op = auxiliary_->shell(P).function_index();
            int nq = auxiliary_->shell(Q).nfunction();
            int oq = auxiliary_->shell(Q).function_index();

            const double* buffer = Jeri[thread]->buffer();

            for (int p = 0; p < np; p++) {
                for (int q = 0; q < nq; q++) {
                    Jp[p + op][q + oq] = Jp[q + oq][p + op] = (*buffer++);
                }
            }
        }
        Jfactory.reset();
        Jeri.clear();

        // > Invert J < //

        J->power(-1.0 / 2.0, 1e-10);
        if (GA_Nodeid() == 0) {
            for (int me = 0; me < GA_Nnodes(); me++) {
                int begin_offset[2];
                int end_offset[2];
                NGA_Distribution(GA_J_onehalf_, me, begin_offset, end_offset);
                int offset = begin_offset[0];
                NGA_Put(GA_J_onehalf_, begin_offset, end_offset, J->pointer()[offset], &naux);
            }
        }
    }
}
} // namespace forte

#endif
