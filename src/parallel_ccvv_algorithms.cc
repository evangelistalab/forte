#include <numeric>

#include <libpsio/psio.hpp>
#include <libpsio/psio.h>
#include <libmints/molecule.h>
#include <libmints/matrix.h>
#include <libmints/vector.h>
#include <libqt/qt.h>
#include "blockedtensorfactory.h"
#include "fci_solver.h"
#include "fci_vector.h"

#include "three_dsrg_mrpt2.h"
#include <vector>
#include <string>
#include <algorithm>
#ifdef HAVE_MPI
#include "mpi.h"
#endif
#ifdef _OPENMP
#include "omp.h"
#endif
using namespace ambit;

namespace psi{ namespace forte{

#ifdef HAVE_GA
    #include <ga.h>
    #include <macdecls.h>
    #include <omp.h>
#else 
    #define GA_Nnodes() 1
    #define GA_Nodeid() 0
#endif


double THREE_DSRG_MRPT2::E_VT2_2_batch_core_ga()
{
    bool debug_print = options_.get_bool("DSRG_MRPT2_DEBUG");
    double Ealpha = 0.0;
    double Emixed = 0.0;
    double Ebeta  = 0.0;
    int my_proc  = GA_Nodeid();
    // Compute <[V, T2]> (C_2)^4 ccvv term; (me|nf) = B(L|me) * B(L|nf)
    // For a given m and n, form Bm(L|e) and Bn(L|f)
    // Bef(ef) = Bm(L|e) * Bn(L|f)
    if(debug_print) printf("\n Computing V_T2_2 in batch algorithm with P%d\n", my_proc);
    if(debug_print) printf("\n Batching algorithm is going over m and n");
    outfile->Printf("\n Computing V_T2_2 in batch algorithm\n");
    outfile->Printf("v\n Batching algorithm is going over m and n");
    size_t dim = nthree_ * virtual_;
    int nthread = 1;
    #ifdef _OPENMP
        nthread = omp_get_max_threads();
    #endif

    ///Step 1:  Figure out the largest chunk of B_{me}^{Q} and B_{nf}^{Q} can be stored in core.  
    /// In Parallel, make sure to limit memory per core.  
    int num_proc = GA_Nnodes();
    outfile->Printf("\n\n====Blocking information==========\n");
    /// Memory keyword is global (compute per node memory here)
    size_t int_mem_int = 0;
    size_t memory_input = 0;
    int num_block = 0;
    int block_size = 0;
    if(my_proc ==0) int_mem_int = (nthree_ * core_ * virtual_) * sizeof(double);
    if(my_proc ==0) memory_input = Process::environment.get_memory() * 0.75 * 1.0 / num_proc;
    if(my_proc == 0) num_block = int_mem_int / memory_input < 1 ? 1 : int_mem_int / memory_input;
    if(my_proc == 0) block_size = core_ / num_block;
    MPI_Bcast(&int_mem_int,  1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&memory_input, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&num_block,    1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&block_size,   1, MPI_INT, 0, MPI_COMM_WORLD);
    //printf("\n memory_input: %d num_block: %d int_mem_int: %d", memory_input, num_block, int_mem_int);

    ///Since the integrals compute Fa_, need to make sure Fa is distributed to all cores
    if(my_proc != 0)
    {
        Fa_.resize(ncmo_);
        Fb_.resize(ncmo_);
    }
    Timer F_BCAST;
    MPI_Bcast(&Fa_[0], ncmo_, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&Fb_[0], ncmo_, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    if(debug_print) printf("\n P%d done with F_BCAST: %8.8f s", my_proc, F_BCAST.get());
    if(debug_print) printf("\n P%d ncmo_: %d nthree_: %d virtual_: %d core_: %d", my_proc, ncmo_, nthree_, virtual_, core_);

    if(memory_input > int_mem_int)
    {
        if(my_proc == 0)
        {
            block_size = core_ / num_proc;
            num_block = num_proc;
        }
        MPI_Bcast(&num_block,  1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&block_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }
    if(options_.get_int("CCVV_BATCH_NUMBER") != -1)
    {
        num_block = options_.get_int("CCVV_BATCH_NUMBER");
        block_size = core_ / num_block;
    }

    if(block_size < 1)
    {
        printf("\n P%d found this", my_proc);
        printf("\n\n Block size is FUBAR.");
        printf("\n Block size is %d with P%d", block_size, my_proc);
        throw PSIEXCEPTION("Block size is either 0 or negative.  Fix this problem");
    }
    if(num_block > core_)
    {
        printf("\n Number of blocks can not be larger than core_ on P%d", my_proc);
        printf("\n num_block: %d core_: %d on P%d", num_block, core_, my_proc);
        throw PSIEXCEPTION("Number of blocks is larger than core.  Fix num_block or check source code");
    }
    if(num_block < num_proc)
    {
        outfile->Printf("\n Set number of processors larger");
        outfile->Printf("\n This algorithm uses P processors to block DF tensors");
        printf("\n num_block = %d and num_proc = %d with P%d", num_block, num_proc, my_proc);
        throw PSIEXCEPTION("Set number of processors larger.  See output for details.");
    }
    if(num_block >= 1)
    {
        outfile->Printf("\n  %lu / %lu = %lu", int_mem_int, memory_input, int_mem_int / memory_input);
        outfile->Printf("\n  Block_size = %lu num_block = %lu", block_size, num_block);
        outfile->Printf("\n core_: %d num_proc: %d", core_, num_proc);
    }
    ///Create a Global Array with B_{me}^{Q}
    ///My algorithm assumes that the tensor is distributed as m(iproc)B_{e}^{Q}
    ///Each processor holds a chunk of B distributed through the core index
    ///dims-> nthree_, core_, virtual_
    
    int dims[2];
    int B_chunk[2];
    int ld[1];
    ld[0] = nthree_ * virtual_;
    if(debug_print) printf("\n myproc: %d block_size: %d", my_proc, block_size);
    dims[0] = core_;
    dims[1] = nthree_ * virtual_;
    int dim_chunk[2];
    dim_chunk[0] = num_block;
    dim_chunk[1] = 1;
    int map[num_block + 1];
    for(int i = 0; i < num_block; i++)
    {
        map[i] = i * block_size;
    }
    map[num_block] = 0;
    if(debug_print)
    {
        outfile->Printf("\n dim_chunk[0]: %d dim_chunk[1]: %d", dim_chunk[0], dim_chunk[1]);
        for(int i = 0; i < num_block + 1; i++)
            outfile->Printf("\n map[%d] = %d", i, map[i]);
    }
    int mBe = NGA_Create_irreg(C_DBL, 2, dims, (char *)"mBe", dim_chunk, map);
    if(not mBe)
    {
        GA_Error((char *)"Create mBe failed", 0);
        throw PSIEXCEPTION("Error in creating GA for B");
    }
    if(debug_print) GA_Print_distribution(mBe);
    if(my_proc == 0 && debug_print) printf("Created mBe tensor");
    std::vector<size_t> virt_mos = mo_space_info_->get_corr_abs_mo("RESTRICTED_UOCC");
    std::vector<size_t> naux(nthree_);
    std::iota(naux.begin(), naux.end(), 0);
    /// Take m_blocks and split it up between processors
    std::pair<std::vector<int>, std::vector<int> > my_tasks = split_up_tasks(num_block, num_proc);
    ///Since B is stored on disk on processor 0, only read from p0
    for(int iproc = 0; iproc < num_proc; iproc++)
    {
        if(my_proc == 0)
        {
            for(int m_blocks = my_tasks.first[iproc]; m_blocks < my_tasks.second[iproc]; m_blocks++)
            {
                std::vector<size_t> m_batch;
                if(core_ % num_block == 0)
                {
                    /// Fill the mbatch from block_begin to block_end
                    /// This is done so I can pass a block to IntegralsAPI to read a chunk
                    m_batch.resize(block_size);
                    /// copy used to get correct indices for B.  
                    std::copy(acore_mos_.begin() + (m_blocks * block_size), acore_mos_.begin() + ((m_blocks + 1) * block_size), m_batch.begin());
                }
                else
                {
                    ///If last_block is shorter or long, fill the rest
                    size_t gimp_block_size = m_blocks==(num_block - 1) ? block_size + core_ % num_block : block_size;
                    m_batch.resize(gimp_block_size);
                    //std::iota(m_batch.begin(), m_batch.end(), m_blocks * (core_ / num_block));
                     std::copy(acore_mos_.begin() + (m_blocks)  * block_size, acore_mos_.begin() + (m_blocks) * block_size +  gimp_block_size, m_batch.begin());
                }
                ambit::Tensor BmQe = ambit::Tensor::build(tensor_type_, "BmQE", {m_batch.size(), nthree_, virtual_});
                ambit::Tensor B = ints_->three_integral_block(naux, m_batch, virt_mos);
                BmQe("mQe") = B("Qme");

                int begin_offset[2];
                int end_offset[2];
                NGA_Distribution(mBe, iproc, begin_offset, end_offset);
                NGA_Put(mBe, begin_offset, end_offset, &BmQe.data()[0], ld);
                for(int i = 0; i < 2; i++)
                {
                    outfile->Printf("\n my_proc: %d offsets[%d] = (%d, %d)", iproc, i, begin_offset[i], end_offset[i]);
                }
                
                //#ifdef HAVE_GA
                //#endif
            }
        }
    }

    //if(my_proc == 0)
    //{
    //    ambit::Tensor Bcorrect = ints_->three_integral_block(naux, acore_mos_, virt_mos);
    //    ambit::Tensor Bcorrect_trans = ambit::Tensor::build(tensor_type_, "BFull", {core_, nthree_, virtual_});
    //    Bcorrect_trans("mQe") = Bcorrect("Qme");
    //}
    if(debug_print)
    {
        ambit::Tensor B_global = ambit::Tensor::build(tensor_type_, "BGlobal", {core_, nthree_, virtual_});
        printf("\n P%d going to NGA_GET", my_proc);
        outfile->Printf("\n");
        if(my_proc == 0)
        {
            for(int iproc = 0; iproc < num_proc; iproc++)
            {
                int begin_offset_get[2];
                int end_offset_get[2];
                NGA_Distribution(mBe, iproc, begin_offset_get, end_offset_get);
                for(int i = 0; i < 2; i++)
                {
                    outfile->Printf("\n my_proc: %d offsets[%d] = (%d, %d)", iproc, i, begin_offset_get[i], end_offset_get[i]);
                }
                NGA_Get(mBe, begin_offset_get, end_offset_get, &(B_global.data()[begin_offset_get[0] * nthree_ * virtual_]), ld);
            }
        }
        GA_Sync();
    if(my_proc == 0 && debug_print) printf("\n B_Global norm: %8.8f", B_global.norm(2.0));
    }
    GA_Sync();
    if(debug_print) GA_Print(mBe);
    if(debug_print) GA_Print_distribution(mBe);

    /// Race condition if each thread access ambit tensors
    /// Force each thread to have its own copy of matrices (memory NQ * V)
    std::vector<ambit::Tensor> BefVec;
    std::vector<ambit::Tensor> BefJKVec;
    std::vector<ambit::Tensor> RDVec;
    std::vector<ambit::Tensor> BmaVec;
    std::vector<ambit::Tensor> BnaVec;
    std::vector<ambit::Tensor> BmbVec;
    std::vector<ambit::Tensor> BnbVec;

    for (int i = 0; i < nthread; i++)
    {
        BmaVec.push_back(ambit::Tensor::build(tensor_type_,"Bma",{nthree_,virtual_}));
        BnaVec.push_back(ambit::Tensor::build(tensor_type_,"Bna",{nthree_,virtual_}));
        BmbVec.push_back(ambit::Tensor::build(tensor_type_,"Bmb",{nthree_,virtual_}));
        BnbVec.push_back(ambit::Tensor::build(tensor_type_,"Bnb",{nthree_,virtual_}));
        BefVec.push_back(ambit::Tensor::build(tensor_type_,"Bef",{virtual_,virtual_}));
        BefJKVec.push_back(ambit::Tensor::build(tensor_type_,"BefJK",{virtual_,virtual_}));
        RDVec.push_back(ambit::Tensor::build(tensor_type_, "RDVec", {virtual_, virtual_}));

    }

    std::vector<std::pair<int, int> > mn_tasks;
    for(int m = 0; m < num_block; m++)
        for(int n = 0; n <= m; n++)
            mn_tasks.push_back({m, n});

    std::pair<std::vector<int>, std::vector<int> > my_mn_tasks = split_up_tasks(mn_tasks.size(), num_proc);
    for(int p = 0; p < num_proc; p++)
    {
        outfile->Printf("\n my_tasks[%d].first: %d my_tasks[%d].second: %d", p, my_mn_tasks.first[p], p, my_mn_tasks.second[p]);
    }
    ///Step 2:  Loop over memory allowed blocks of m and n
    /// Get batch sizes and create vectors of mblock length
    for(size_t tasks = my_mn_tasks.first[my_proc]; tasks < my_mn_tasks.second[my_proc];tasks++)
    {
        int m_blocks = mn_tasks[tasks].first;
        int n_blocks = mn_tasks[tasks].second;
        if(debug_print) printf("\n my_proc: %d tasks: %d m_blocks: %d n_blocks: %d", my_proc, tasks, m_blocks, n_blocks);
        std::vector<size_t> m_batch;
        size_t gimp_block_size = 0;
        ///If core_ goes into num_block equally, all blocks are equal
        if(core_ % num_block == 0)
        {
            /// Fill the mbatch from block_begin to block_end
            /// This is done so I can pass a block to IntegralsAPI to read a chunk
            m_batch.resize(block_size);
            /// copy used to get correct indices for B.  
            std::copy(acore_mos_.begin() + (m_blocks * block_size), acore_mos_.begin() + ((m_blocks + 1) * block_size), m_batch.begin());
            gimp_block_size = block_size;
        }
        else
        {
            ///If last_block is shorter or long, fill the rest
            gimp_block_size = m_blocks==(num_block - 1) ? block_size + core_ % num_block : block_size;
            m_batch.resize(gimp_block_size);
            //std::iota(m_batch.begin(), m_batch.end(), m_blocks * (core_ / num_block));
             std::copy(acore_mos_.begin() + (m_blocks)  * block_size, acore_mos_.begin() + (m_blocks) * block_size +  gimp_block_size, m_batch.begin());
        }

        ///Get the correct chunk for m_batch 
        ///Since every processor has different chunk (I can't assume locality)
        ambit::Tensor BmQe = ambit::Tensor::build(tensor_type_, "BmQE", {m_batch.size(), nthree_, virtual_});

        int m_begin_offset[2];
        int m_end_offset[2];
        //int m_offset_start = m_blocks * block_size;
        //int m_offset_end   = m_blocks * block_size + m_batch.size() - 1;
        //m_begin_offset[0] = m_offset_start;
        //m_begin_offset[1] = 0;
        //m_end_offset[0]  = m_offset_end;
        //m_end_offset[1]  = nthree_ * virtual_ - 1;
        //if(debug_print)
        //{
        //    for(int i = 0; i < 2; i++)
        //        printf("\n my_proc: %d offset[%d] = (%d, %d)", my_proc, i, m_begin_offset[i], m_end_offset[i]);
        //}
        //int* map_array;
        //int proc_list[num_proc];
        //int NGA_INFO = NGA_Locate_region(mBe, m_begin_offset, m_end_offset, map_array, proc_list);
        int subscript[2];
        subscript[0] = m_blocks * block_size;
        subscript[1] = 0;
        int NGA_INFO = NGA_Locate(mBe, subscript);
        
        if(NGA_INFO == -1)
        {
            printf("\n NGA_INFO: %d", NGA_INFO);
            printf("\n Found multiple blocks that hold this region");
            printf("\n Could not locate block of mBe");
            printf("\n my_proc: %d block_size: %d m_blocks: %d", my_proc, block_size, m_blocks);
            //for(int i = 0; i < num_proc; i++)
            //    printf("\n PL[%d] = %d", i, proc_list[i]);
            throw PSIEXCEPTION("GA could not locate region of B_m^{Qe}");
        }
        int begin_offset[2];
        int end_offset[2];
        NGA_Distribution(mBe, NGA_INFO, begin_offset, end_offset);
        NGA_Get(mBe, begin_offset, end_offset, &(BmQe.data()[0]), ld);

        std::vector<size_t> n_batch;
        ///If core_ goes into num_block equally, all blocks are equal
        if(core_ % num_block == 0)
        {
            /// Fill the mbatch from block_begin to block_end
            /// This is done so I can pass a block to IntegralsAPI to read a chunk
            n_batch.resize(block_size);
            std::copy(acore_mos_.begin() + n_blocks * block_size, acore_mos_.begin() + ((n_blocks + 1) * block_size), n_batch.begin());
            gimp_block_size = block_size;
        }
        else
        {
             ///If last_block is longer, block_size + remainder
             gimp_block_size = n_blocks==(num_block - 1) ? block_size +core_ % num_block : block_size;
             n_batch.resize(gimp_block_size);
             std::copy(acore_mos_.begin() + (n_blocks) * block_size, acore_mos_.begin() + (n_blocks  * block_size) + gimp_block_size , n_batch.begin());
         }

         ambit::Tensor BnQf = ambit::Tensor::build(tensor_type_, "BnQf", {n_batch.size(), nthree_, virtual_});

         if(n_blocks == m_blocks)
         {
             BnQf.copy(BmQe);
         }
         else
         {
            int locate_offset[2];
            locate_offset[0] = n_blocks * block_size;
            locate_offset[1] = 0;
            NGA_INFO = NGA_Locate(mBe, locate_offset);
            if(NGA_INFO == -1)
            {
                printf("\n Could not locate block of mBe");
                printf("\n locate_offset[0]: %d", locate_offset[0]);
                printf("\n nblocks: %d block_size: %d", n_blocks, block_size);
                throw PSIEXCEPTION("GA could not locate region of B_m^{Qe}");
            }
            int begin_offset_n[2];
            int end_offset_n[2];
            NGA_Distribution(mBe, NGA_INFO, begin_offset_n, end_offset_n);
            NGA_Get(mBe, begin_offset_n, end_offset_n, &(BnQf.data()[0]), ld);
         }
         size_t m_size = m_batch.size();
         size_t n_size = n_batch.size();
         #pragma omp parallel for \
             schedule(static) \
             reduction(+:Ealpha, Emixed) 
         for(size_t mn = 0; mn < m_size * n_size; ++mn){
             int thread = 0;
             size_t m = mn / n_size + m_batch[0];
             size_t n = mn % n_size + n_batch[0];
             if(n > m) continue;
             double factor = (m == n ? 1.0 : 2.0);
             #ifdef _OPENMP
                 thread = omp_get_thread_num();
             #endif
             ///Since loop over mn is collapsed, need to use fancy offset tricks
             /// m_in_loop = mn / n_size -> corresponds to m increment (m++) 
             /// n_in_loop = mn % n_size -> corresponds to n increment (n++)
             /// m_batch[m_in_loop] corresponds to the absolute index
             size_t m_in_loop = mn / n_size;
             size_t n_in_loop = mn % n_size;
             size_t ma = m_batch[m_in_loop ];
             size_t mb = m_batch[m_in_loop ];

             size_t na = n_batch[n_in_loop ];
             size_t nb = n_batch[n_in_loop ];

             std::copy(BmQe.data().begin() + (m_in_loop) * dim, BmQe.data().begin() +  (m_in_loop) * dim + dim, BmaVec[thread].data().begin());

             std::copy(BnQf.data().begin() + (mn % n_size) * dim, BnQf.data().begin() + (n_in_loop) * dim + dim, BnaVec[thread].data().begin());
             std::copy(BnQf.data().begin() + (mn % n_size) * dim, BnQf.data().begin() + (n_in_loop) * dim + dim, BnbVec[thread].data().begin());


             //// alpha-aplha
             BefVec[thread]("ef") = BmaVec[thread]("ge") * BnaVec[thread]("gf");
             BefJKVec[thread]("ef")  = BefVec[thread]("ef") * BefVec[thread]("ef");
             BefJKVec[thread]("ef") -= BefVec[thread]("ef") * BefVec[thread]("fe");
             RDVec[thread].iterate([&](const std::vector<size_t>& i,double& value){
                 double D = Fa_[ma] + Fa_[na] - Fa_[avirt_mos_[i[0]]] - Fa_[avirt_mos_[i[1]]];
                 value = renormalized_denominator(D) * (1.0 + renormalized_exp(D));});
             Ealpha += factor * 1.0 * BefJKVec[thread]("ef") * RDVec[thread]("ef");

             //// beta-beta
             ////BefVec[thread]("EF") = BmbVec[thread]("gE") * BnbVec[thread]("gF");
             ////BefJKVec[thread]("EF")  = BefVec[thread]("EF") * BefVec[thread]("EF");
             ////BefJKVec[thread]("EF") -= BefVec[thread]("EF") * BefVec[thread]("FE");
             ////RDVec[thread].iterate([&](const std::vector<size_t>& i,double& value){
             ////    double D = Fb_[mb] + Fb_[nb] - Fb_[bvirt_mos_[i[0]]] - Fb_[bvirt_mos_[i[1]]];
             ////    value = renormalized_denominator(D) * (1.0 + renormalized_exp(D));});
             ////Ebeta += 0.5 * BefJKVec[thread]("EF") * RDVec[thread]("EF");

             //// alpha-beta
             BefVec[thread]("eF") = BmaVec[thread]("ge") * BnbVec[thread]("gF");
             BefJKVec[thread]("eF")  = BefVec[thread]("eF") * BefVec[thread]("eF");
             RDVec[thread].iterate([&](const std::vector<size_t>& i,double& value){
                 double D = Fa_[ma] + Fb_[nb] - Fa_[avirt_mos_[i[0]]] - Fb_[bvirt_mos_[i[1]]];
                 value = renormalized_denominator(D) * (1.0 + renormalized_exp(D));});
             Emixed += factor * BefJKVec[thread]("eF") * RDVec[thread]("eF");
             if(debug_print)
             {
                 printf("\n my_proc: %d m_size: %d n_size: %d m: %d n:%d", my_proc, m_size, n_size, m, n);
                 printf("\n my_proc: %d m: %d n:%d Ealpha = %8.8f Emixed = %8.8f Sum = %8.8f", my_proc, m, n, Ealpha , Emixed, Ealpha + Emixed);
             }
         }
     }
     double local_sum = Ealpha + Emixed;
     double total_sum;
     MPI_Reduce(&local_sum, &total_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
     return total_sum;
}
double THREE_DSRG_MRPT2::E_VT2_2_batch_core_rep()
{
    bool debug_print = options_.get_bool("DSRG_MRPT2_DEBUG");
    double Ealpha = 0.0;
    double Emixed = 0.0;
    double Ebeta  = 0.0;
    int num_proc = MPI::COMM_WORLD.Get_size();
    int my_proc  = MPI::COMM_WORLD.Get_rank();
    // Compute <[V, T2]> (C_2)^4 ccvv term; (me|nf) = B(L|me) * B(L|nf)
    // For a given m and n, form Bm(L|e) and Bn(L|f)
    // Bef(ef) = Bm(L|e) * Bn(L|f)
    outfile->Printf("\n Computing V_T2_2 in batch algorithm\n");
    outfile->Printf("\n Batching algorithm is going over m and n");
    if(debug_print)
        printf("\n P%d is in batch_core_rep", my_proc);
    size_t dim = nthree_ * virtual_;
    int nthread = 1;
    #ifdef _OPENMP
        nthread = omp_get_max_threads();
    #endif
    outfile->Printf("\n Algorithm uses %d processors and %d threads", num_proc, nthread);

    ///Step 1:  Figure out the largest chunk of B_{me}^{Q} and B_{nf}^{Q} can be stored in core.  
    outfile->Printf("\n\n====Blocking information==========\n");
    size_t int_mem_int = 0;
    size_t memory_input = 0;
    size_t num_block = 0;
    size_t block_size = 0;
    if(my_proc == 0)
    {
        int_mem_int = (nthree_ * core_ * virtual_) * sizeof(double);
        memory_input = Process::environment.get_memory() * 0.75 * 1.0 / num_proc;
        num_block = int_mem_int / memory_input < 1 ? 1 : int_mem_int / memory_input;
        block_size = core_ / num_block;
    }
    MPI_Bcast(&int_mem_int, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&memory_input, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&num_block, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&block_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if(options_.get_int("CCVV_BATCH_NUMBER") != -1)
    {
        if(my_proc == 0) num_block = options_.get_int("CCVV_BATCH_NUMBER");
        MPI_Bcast(&num_block, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }
    if(memory_input > int_mem_int)
    {
        if(my_proc == 0) block_size = core_ / num_proc;
        if(my_proc == 0) num_block = num_proc;
        MPI_Bcast(&block_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&num_block, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }
    if(num_block >= 1)
    {
        outfile->Printf("\n Block Information");
        outfile->Printf("\n  %lu / %lu = %lu", int_mem_int, memory_input, int_mem_int / memory_input);
        outfile->Printf("\n  Block_size = %lu num_block = %lu P%d", block_size, num_block);
    }

    if(block_size < 1)
    {
        printf("\n\n Block size is FUBAR on P%d.", my_proc);
        printf("\n Block size is %d", block_size);
        printf("\n Block Information for P%d\n", my_proc);
        printf("\n  %lu / %lu = %lu P%d", int_mem_int, memory_input, int_mem_int / memory_input, my_proc);
        printf("\n  Block_size = %lu num_block = %lu P%d", block_size, num_block, my_proc);
        printf("\n Core_: %d virtual_: %d nthree_: %d", core_, virtual_, nthree_);
        throw PSIEXCEPTION("Block size is either 0 or negative.  Fix this problem");
    }    
    if(num_block > core_)
    {
        printf("\n P%d says that num_block is %d", my_proc, num_block);
        outfile->Printf("\n Number of blocks can not be larger than core_");
        throw PSIEXCEPTION("Number of blocks is larger than core.  Fix num_block or check source code");
    }

    if(num_block < num_proc)
    {
        outfile->Printf("\n Set number of processors larger");
        outfile->Printf("\n This algorithm uses P processors to block DF tensors");
        outfile->Printf("\n num_block = %d and num_proc = %d", num_block, num_proc);
        throw PSIEXCEPTION("Set number of processors larger.  See output for details.");
    }
    if(debug_print)
    {
        printf("\n P%d is complete with all block information", my_proc);
        printf("\n P%d num_block: %d core_: %d virtual_: %d nthree_: %d ncmo_: %d num_proc: %d block_size: %d", my_proc, num_block, core_, virtual_, nthree_, ncmo_, num_proc, block_size);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    
    printf("\n P%d virt_mos begin", my_proc);
    std::vector<size_t> virt_mos = mo_space_info_->get_corr_abs_mo("RESTRICTED_UOCC");
    printf("\n P%d virt_mos end", my_proc);
    MPI_Barrier(MPI_COMM_WORLD);
    printf("\n P%d Naux init with nthree_:%d", my_proc, nthree_);
    std::vector<size_t> naux;
    naux.resize(nthree_);
    printf("\n P%d Naux init correctly", my_proc);
    MPI_Barrier(MPI_COMM_WORLD);
    printf("\n P%d naux about to be filled", my_proc);
    std::iota(naux.begin(), naux.begin() + nthree_, 0);
    printf("\n P%d virt_mos, naux, and filling naux ending", my_proc);

    /// Race condition if each thread access ambit tensors
    /// Force each thread to have its own copy of matrices (memory NQ * V)
    std::vector<ambit::Tensor> BefVec;
    std::vector<ambit::Tensor> BefJKVec;
    std::vector<ambit::Tensor> RDVec;
    std::vector<ambit::Tensor> BmaVec;
    std::vector<ambit::Tensor> BnaVec;
    std::vector<ambit::Tensor> BmbVec;
    std::vector<ambit::Tensor> BnbVec;
    if(debug_print) printf("\n P%d going to allocate tensors", my_proc);

    for (int i = 0; i < nthread; i++)
    {
        BmaVec.push_back(ambit::Tensor::build(tensor_type_,"Bma",{nthree_,virtual_}));
        BnaVec.push_back(ambit::Tensor::build(tensor_type_,"Bna",{nthree_,virtual_}));
        BmbVec.push_back(ambit::Tensor::build(tensor_type_,"Bmb",{nthree_,virtual_}));
        BnbVec.push_back(ambit::Tensor::build(tensor_type_,"Bnb",{nthree_,virtual_}));
        BefVec.push_back(ambit::Tensor::build(tensor_type_,"Bef",{virtual_,virtual_}));
        BefJKVec.push_back(ambit::Tensor::build(tensor_type_,"BefJK",{virtual_,virtual_}));
        RDVec.push_back(ambit::Tensor::build(tensor_type_, "RDVec", {virtual_, virtual_}));

    }
    if(debug_print) printf("\n P%d done with allocating tensors", my_proc);

    std::vector<std::pair<int, int> > mn_tasks;
    for(int m = 0, idx = 0; m < num_block; m++)
    {
        for(int n = 0; n <= m; n++, idx++)
        {
            mn_tasks.push_back({m, n});
        }
    }
    
    std::pair<std::vector<int>, std::vector<int> > my_tasks = split_up_tasks(mn_tasks.size(), num_proc);
    std::vector<int> batch_start = my_tasks.first;
    std::vector<int> batch_end =   my_tasks.second;
    /// B tensor will be broadcasted to all processors (very memory heavy)
    /// F matrix will be broadcasted to all processors (N^2)
    MPI_Barrier(MPI_COMM_WORLD);
    if(debug_print) printf("\n P%d going to allocate tensor", my_proc);
    ambit::Tensor BmQe = ambit::Tensor::build(tensor_type_, "BmQE", {core_, nthree_, virtual_});
    if(debug_print) printf("\n P%d done with allocatating tensor", my_proc);
    if(my_proc != 0)
    {
        Fa_.resize(ncmo_);
        Fb_.resize(ncmo_);
    }
    if(my_proc == 0)
    {
        ambit::Tensor B = ints_->three_integral_block(naux, acore_mos_, virt_mos);
        BmQe("mQe") = B("Qme");
    }
    Timer F_Bcast;
    if(debug_print) printf("\n F_Bcast for F about to start on P%d", my_proc);
    MPI_Bcast(&Fa_[0], ncmo_, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&Fb_[0], ncmo_, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    if(debug_print) printf("\n F_Bcast for F end on P%d", my_proc);
    if(debug_print) printf("\n F_Bcast for F took %8.6f s on P%d.", F_Bcast.get(), my_proc);
    if(debug_print) printf("\n B_Bcast for B about to start on P%d", my_proc);
    Timer B_Bcast;
    MPI_Bcast(&BmQe.data()[0], nthree_ * virtual_ * core_, MPI_DOUBLE, 0,MPI_COMM_WORLD);
    if(debug_print) printf("\n B_Bcast took %8.8f on P%d", B_Bcast.get(), my_proc);
    if(debug_print) printf("\n BmQe norm: %8.8f on P%d", BmQe.norm(2.0), my_proc);
    ///Step 2:  Loop over memory allowed blocks of m and n
    /// Get batch sizes and create vectors of mblock length

    for(int tasks = my_tasks.first[my_proc]; tasks < my_tasks.second[my_proc]; tasks++)
    {
        if(debug_print) printf("\n tasks: %d my-tasks.first[%d] = %d my_tasks.second = %d", tasks, my_proc, my_tasks.first[my_proc], my_tasks.second[my_proc]);
        int m_blocks = mn_tasks[tasks].first;
        int n_blocks = mn_tasks[tasks].second;
        if(debug_print) printf("\n m_blocks: %d n_blocks: %d", m_blocks, n_blocks);
        std::vector<size_t> m_batch;
        size_t gimp_block_size = 0;
        ///If core_ goes into num_block equally, all blocks are equal
        if(core_ % num_block == 0)
        {
            /// Fill the mbatch from block_begin to block_end
            /// This is done so I can pass a block to IntegralsAPI to read a chunk
            m_batch.resize(block_size);
            /// copy used to get correct indices for B.  
            std::copy(acore_mos_.begin() + (m_blocks * block_size), acore_mos_.begin() + ((m_blocks + 1) * block_size), m_batch.begin());
            gimp_block_size = block_size;
        }
        else
        {
            ///If last_block is shorter or long, fill the rest
            gimp_block_size = m_blocks==(num_block - 1) ? block_size + core_ % num_block : block_size;
            m_batch.resize(gimp_block_size);
            //std::iota(m_batch.begin(), m_batch.end(), m_blocks * (core_ / num_block));
             std::copy(acore_mos_.begin() + (m_blocks)  * block_size, acore_mos_.begin() + (m_blocks) * block_size +  gimp_block_size, m_batch.begin());
        }

        ambit::Tensor BmQe_batch = ambit::Tensor::build(tensor_type_, "BmQE", {m_batch.size(), nthree_, virtual_});
        std::copy(BmQe.data().begin() + (m_blocks * block_size) * nthree_ * virtual_, BmQe.data().begin() + (m_blocks  * block_size) * nthree_ * virtual_ + gimp_block_size* nthree_ * virtual_, BmQe_batch.data().begin());
        if(debug_print)
        {
            printf("\n BmQe norm: %8.8f", BmQe_batch.norm(2.0));
            printf("\n m_block: %d", m_blocks);
            int count = 0;
            for(auto mb : m_batch)
            {
                printf("m_batch[%d] =  %d ",count, mb);
                count++;
            }
            printf("\n Core indice list");
            for(auto coremo : acore_mos_)
            {
                outfile->Printf(" %d " , coremo);
            }
        }
        
        std::vector<size_t> n_batch;
        ///If core_ goes into num_block equally, all blocks are equal
        if(core_ % num_block == 0)
        {
            /// Fill the mbatch from block_begin to block_end
            /// This is done so I can pass a block to IntegralsAPI to read a chunk
            n_batch.resize(block_size);
            std::copy(acore_mos_.begin() + n_blocks * block_size, acore_mos_.begin() + ((n_blocks + 1) * block_size), n_batch.begin());
            gimp_block_size = block_size;
        }
        else
        {
            ///If last_block is longer, block_size + remainder
            gimp_block_size = n_blocks==(num_block - 1) ? block_size +core_ % num_block : block_size;
            n_batch.resize(gimp_block_size);
            std::copy(acore_mos_.begin() + (n_blocks) * block_size, acore_mos_.begin() + (n_blocks  * block_size) + gimp_block_size , n_batch.begin());
        }

        ambit::Tensor BnQf_batch = ambit::Tensor::build(tensor_type_, "BnQf", {n_batch.size(), nthree_, virtual_});

        if(n_blocks == m_blocks)
        {
            BnQf_batch.copy(BmQe_batch);
        }
        else
        {
            std::copy(BmQe.data().begin() + (n_blocks * block_size) * nthree_ * virtual_, BmQe.data().begin() + (n_blocks * block_size) * nthree_ * virtual_ + gimp_block_size * nthree_ * virtual_, BnQf_batch.data().begin());
        }
        if(debug_print)
        {
            printf("\n BnQf norm: %8.8f", BnQf_batch.norm(2.0));
            printf("\n n_block: %d", n_blocks);
            int count = 0;
            for(auto nb : n_batch)
            {
                printf("n_batch[%d] =  %d ", count, nb);
                count++;
            }
        }
        size_t m_size = m_batch.size();
        size_t n_size = n_batch.size();
        #pragma omp parallel for num_threads(num_threads_)\
            schedule(static) \
            reduction(+:Ealpha, Emixed) 
        for(size_t mn = 0; mn < m_size * n_size; ++mn){
            int thread = 0;
            size_t m = mn / n_size + m_batch[0];
            size_t n = mn % n_size + n_batch[0];
            if(n > m) continue;
            double factor = (m == n ? 1.0 : 2.0);
            #ifdef _OPENMP
                thread = omp_get_thread_num();
            #endif
            ///Since loop over mn is collapsed, need to use fancy offset tricks
            /// m_in_loop = mn / n_size -> corresponds to m increment (m++) 
            /// n_in_loop = mn % n_size -> corresponds to n increment (n++)
            /// m_batch[m_in_loop] corresponds to the absolute index
            size_t m_in_loop = mn / n_size;
            size_t n_in_loop = mn % n_size;
            size_t ma = m_batch[m_in_loop ];
            size_t mb = m_batch[m_in_loop ];

            size_t na = n_batch[n_in_loop ];
            size_t nb = n_batch[n_in_loop ];

            std::copy(BmQe_batch.data().begin() + (m_in_loop) * dim, BmQe_batch.data().begin() +  (m_in_loop) * dim + dim, BmaVec[thread].data().begin());

            std::copy(BnQf_batch.data().begin() + (mn % n_size) * dim, BnQf_batch.data().begin() + (n_in_loop) * dim + dim, BnaVec[thread].data().begin());
            std::copy(BnQf_batch.data().begin() + (mn % n_size) * dim, BnQf_batch.data().begin() + (n_in_loop) * dim + dim, BnbVec[thread].data().begin());


            //// alpha-aplha
            BefVec[thread]("ef") = BmaVec[thread]("ge") * BnaVec[thread]("gf");
            BefJKVec[thread]("ef")  = BefVec[thread]("ef") * BefVec[thread]("ef");
            BefJKVec[thread]("ef") -= BefVec[thread]("ef") * BefVec[thread]("fe");
            RDVec[thread].iterate([&](const std::vector<size_t>& i,double& value){
                double D = Fa_[ma] + Fa_[na] - Fa_[avirt_mos_[i[0]]] - Fa_[avirt_mos_[i[1]]];
                value = renormalized_denominator(D) * (1.0 + renormalized_exp(D));});
            Ealpha += factor * 1.0 * BefJKVec[thread]("ef") * RDVec[thread]("ef");

            //// beta-beta
            ////BefVec[thread]("EF") = BmbVec[thread]("gE") * BnbVec[thread]("gF");
            ////BefJKVec[thread]("EF")  = BefVec[thread]("EF") * BefVec[thread]("EF");
            ////BefJKVec[thread]("EF") -= BefVec[thread]("EF") * BefVec[thread]("FE");
            ////RDVec[thread].iterate([&](const std::vector<size_t>& i,double& value){
            ////    double D = Fb_[mb] + Fb_[nb] - Fb_[bvirt_mos_[i[0]]] - Fb_[bvirt_mos_[i[1]]];
            ////    value = renormalized_denominator(D) * (1.0 + renormalized_exp(D));});
            ////Ebeta += 0.5 * BefJKVec[thread]("EF") * RDVec[thread]("EF");

            //// alpha-beta
            BefVec[thread]("eF") = BmaVec[thread]("ge") * BnbVec[thread]("gF");
                BefJKVec[thread]("eF")  = BefVec[thread]("eF") * BefVec[thread]("eF");
                RDVec[thread].iterate([&](const std::vector<size_t>& i,double& value){
                    double D = Fa_[ma] + Fb_[nb] - Fa_[avirt_mos_[i[0]]] - Fb_[bvirt_mos_[i[1]]];
                    value = renormalized_denominator(D) * (1.0 + renormalized_exp(D));});
            Emixed += factor * BefJKVec[thread]("eF") * RDVec[thread]("eF");
            if(debug_print)
            {
                 printf("\n my_proc: %d m_size: %d n_size: %d m: %d n:%d", my_proc, m_size, n_size, m, n);
                 printf("\n my_proc: %d m: %d n:%d Ealpha = %8.8f Emixed = %8.8f Sum = %8.8f", my_proc, m, n, Ealpha , Emixed, Ealpha + Emixed);
            }
        }
    }
    double local_sum = Ealpha + Emixed;
    double total_sum;
    MPI_Reduce(&local_sum, &total_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    //return (Ealpha + Ebeta + Emixed);
    
    return (total_sum);
}
double THREE_DSRG_MRPT2::E_VT2_2_batch_virtual_ga()
{
    bool debug_print = options_.get_bool("DSRG_MRPT2_DEBUG");
    double Ealpha = 0.0;
    double Emixed = 0.0;
    double Ebeta  = 0.0;
    // Compute <[V, T2]> (C_2)^4 ccvv term; (me|nf) = B(L|me) * B(L|nf)
    // For a given e and f, form Be(L|m) and Bf(L|n)
    // Bef(mn) = Be(L|m) * Bf(L|n)
    outfile->Printf("\n Computing V_T2_2 in batch algorithm\n");
    outfile->Printf("\n Batching algorithm is going over e and f");
    size_t dim = nthree_ * core_;
    int nthread = 1;
    #ifdef _OPENMP
        nthread = omp_get_max_threads();
    #endif

    ///Step 1:  Figure out the largest chunk of B_{me}^{Q} and B_{nf}^{Q} can be stored in core.  
    outfile->Printf("\n\n====Blocking information==========\n");
    size_t int_mem_int = (nthree_ * core_ * virtual_) * sizeof(double);
    size_t memory_input = Process::environment.get_memory() * 0.75;
    size_t num_block = int_mem_int / memory_input < 1 ? 1 : int_mem_int / memory_input;

    if(options_.get_int("CCVV_BATCH_NUMBER") != -1)
    {
        num_block = options_.get_int("CCVV_BATCH_NUMBER");
    }
    size_t block_size = virtual_ / num_block;

    if(block_size < 1)
    {
        outfile->Printf("\n\n Block size is FUBAR.");
        outfile->Printf("\n Block size is %d", block_size);
        throw PSIEXCEPTION("Block size is either 0 or negative.  Fix this problem");
    }
    if(num_block > virtual_)
    {
        outfile->Printf("\n Number of blocks can not be larger than core_");
        throw PSIEXCEPTION("Number of blocks is larger than core.  Fix num_block or check source code");
    }

    if(num_block >= 1)
    {
        outfile->Printf("\n  %lu / %lu = %lu", int_mem_int, memory_input, int_mem_int / memory_input);
        outfile->Printf("\n  Block_size = %lu num_block = %lu", block_size, num_block);
    }

    
    std::vector<size_t> virt_mos = mo_space_info_->get_corr_abs_mo("RESTRICTED_UOCC");
    std::vector<size_t> naux(nthree_);
    std::iota(naux.begin(), naux.end(), 0);

    /// Race condition if each thread access ambit tensors
    /// Force each thread to have its own copy of matrices (memory NQ * V)
    std::vector<ambit::Tensor> BmnVec;
    std::vector<ambit::Tensor> BmnJKVec;
    std::vector<ambit::Tensor> RDVec;
    std::vector<ambit::Tensor> BmaVec;
    std::vector<ambit::Tensor> BnaVec;
    std::vector<ambit::Tensor> BmbVec;
    std::vector<ambit::Tensor> BnbVec;

    for (int i = 0; i < nthread; i++)
    {
        BmaVec.push_back(ambit::Tensor::build(tensor_type_,"Bma",{nthree_,core_}));
        BnaVec.push_back(ambit::Tensor::build(tensor_type_,"Bna",{nthree_,core_}));
        BmbVec.push_back(ambit::Tensor::build(tensor_type_,"Bmb",{nthree_,core_}));
        BnbVec.push_back(ambit::Tensor::build(tensor_type_,"Bnb",{nthree_,core_}));
        BmnVec.push_back(ambit::Tensor::build(tensor_type_,"Bmn",{core_,core_}));
        BmnJKVec.push_back(ambit::Tensor::build(tensor_type_,"BmnJK",{core_,core_}));
        RDVec.push_back(ambit::Tensor::build(tensor_type_, "RDVec", {core_, core_}));

    }

    ///Step 2:  Loop over memory allowed blocks of m and n
    /// Get batch sizes and create vectors of mblock length
    for(size_t e_blocks = 0; e_blocks < num_block; e_blocks++)
    {
        std::vector<size_t> e_batch;
        ///If core_ goes into num_block equally, all blocks are equal
        if(virtual_ % num_block == 0)
        {
            /// Fill the mbatch from block_begin to block_end
            /// This is done so I can pass a block to IntegralsAPI to read a chunk
            e_batch.resize(block_size);
            /// copy used to get correct indices for B.  
            std::copy(virt_mos.begin() + (e_blocks * block_size), virt_mos.begin() + ((e_blocks + 1) * block_size), e_batch.begin());
        }
        else
        {
            ///If last_block is shorter or long, fill the rest
            size_t gimp_block_size = e_blocks==(num_block - 1) ? block_size + virtual_ % num_block : block_size;
            e_batch.resize(gimp_block_size);
            //std::iota(m_batch.begin(), m_batch.end(), m_blocks * (core_ / num_block));
             std::copy(virt_mos.begin() + (e_blocks)  * block_size, virt_mos.begin() + (e_blocks) * block_size +  gimp_block_size, e_batch.begin());
        }

        ambit::Tensor B = ints_->three_integral_block(naux, e_batch, acore_mos_);
        ambit::Tensor BeQm = ambit::Tensor::build(tensor_type_, "BmQE", {e_batch.size(), nthree_, core_});
        BeQm("eQm") = B("Qem");
        B.reset();

        if(debug_print)
        {
            outfile->Printf("\n BeQm norm: %8.8f", BeQm.norm(2.0));
            outfile->Printf("\n e_block: %d", e_blocks);
            int count = 0;
            for(auto e : e_batch)
            {
                outfile->Printf("e_batch[%d] =  %d ",count, e);
                count++;
            }
            outfile->Printf("\n Virtual index list");
            for(auto virtualmo : virt_mos)
            {
                outfile->Printf(" %d " , virtualmo);
            }
        }
        
        for(size_t f_blocks = 0; f_blocks <= e_blocks; f_blocks++)
        {
            std::vector<size_t> f_batch;
        ///If core_ goes into num_block equally, all blocks are equal
            if(virtual_ % num_block == 0)
            {
                /// Fill the mbatch from block_begin to block_end
                /// This is done so I can pass a block to IntegralsAPI to read a chunk
                f_batch.resize(block_size);
                std::copy(virt_mos.begin() + f_blocks * block_size, virt_mos.begin() + ((f_blocks + 1) * block_size), f_batch.begin());
            }
            else
            {
                ///If last_block is longer, block_size + remainder
                size_t gimp_block_size = f_blocks==(num_block - 1) ? block_size +virtual_ % num_block : block_size;
                f_batch.resize(gimp_block_size);
                std::copy(virt_mos.begin() + (f_blocks) * block_size, virt_mos.begin() + (f_blocks  * block_size) + gimp_block_size , f_batch.begin());
            }
            ambit::Tensor BfQn = ambit::Tensor::build(tensor_type_, "BnQf", {f_batch.size(), nthree_, core_});
            if(f_blocks == e_blocks)
            {
                BfQn.copy(BeQm);
            }
            else
            {
                ambit::Tensor B = ints_->three_integral_block(naux, f_batch, acore_mos_);
                BfQn("eQm") = B("Qem");
                B.reset();
            }
            if(debug_print)
            {
                outfile->Printf("\n BfQn norm: %8.8f", BfQn.norm(2.0));
                outfile->Printf("\n f_block: %d", f_blocks);
                int count = 0;
                for(auto nf : f_batch)
                {
                    outfile->Printf("f_batch[%d] =  %d ", count, nf);
                    count++;
                }
            }
            size_t e_size = e_batch.size();
            size_t f_size = f_batch.size();
            #pragma omp parallel for \
                schedule(static) \
                reduction(+:Ealpha, Emixed) 
            for(size_t ef = 0; ef < e_size * f_size; ++ef){
                int thread = 0;
                size_t e = ef / e_size + e_batch[0];
                size_t f = ef % f_size + f_batch[0];
                if(f > e) continue;
                double factor = (e == f ? 1.0 : 2.0);
                #ifdef _OPENMP
                    thread = omp_get_thread_num();
                #endif
                ///Since loop over mn is collapsed, need to use fancy offset tricks
                /// m_in_loop = mn / n_size -> corresponds to m increment (m++) 
                /// n_in_loop = mn % n_size -> corresponds to n increment (n++)
                /// m_batch[m_in_loop] corresponds to the absolute index
                size_t e_in_loop = ef / f_size;
                size_t f_in_loop = ef % f_size;
                size_t ea = e_batch[e_in_loop ];
                size_t eb = e_batch[e_in_loop ];

                size_t fa = f_batch[f_in_loop ];
                size_t fb = f_batch[f_in_loop ];

                std::copy(BeQm.data().begin() + (e_in_loop) * dim, BeQm.data().begin() +  (e_in_loop) * dim + dim, BmaVec[thread].data().begin());

                std::copy(BfQn.data().begin() + f_in_loop * dim, BfQn.data().begin() + (f_in_loop) * dim + dim, BnaVec[thread].data().begin());
                std::copy(BfQn.data().begin() + f_in_loop * dim, BfQn.data().begin() + (f_in_loop) * dim + dim, BnbVec[thread].data().begin());


                //// alpha-aplha
                BmnVec[thread]("mn") = BmaVec[thread]("gm") * BnaVec[thread]("gn");
                BmnJKVec[thread]("mn")  = BmnVec[thread]("mn") * BmnVec[thread]("mn");
                BmnJKVec[thread]("mn") -= BmnVec[thread]("mn") * BmnVec[thread]("nm");
                RDVec[thread].iterate([&](const std::vector<size_t>& i,double& value){
                    double D = Fa_[acore_mos_[i[0]]] + Fa_[acore_mos_[i[1]]] - Fa_[ea] - Fa_[fa];
                    value = renormalized_denominator(D) * (1.0 + renormalized_exp(D));});
                Ealpha += factor * 1.0 * BmnJKVec[thread]("mn") * RDVec[thread]("mn");


                //// alpha-beta
                BmnVec[thread]("mN") = BmaVec[thread]("gm") * BnbVec[thread]("gN");
                BmnJKVec[thread]("mN")  = BmnVec[thread]("mN") * BmnVec[thread]("mN");
                RDVec[thread].iterate([&](const std::vector<size_t>& i,double& value){
                    double D = Fa_[acore_mos_[i[0]]] + Fa_[acore_mos_[i[1]]] - Fa_[ea] - Fa_[fb];
                    value = renormalized_denominator(D) * (1.0 + renormalized_exp(D));});
                Emixed += factor * BmnJKVec[thread]("mN") * RDVec[thread]("mN");
                if(debug_print)
                {
                    outfile->Printf("\n e_size: %d f_size: %d e: %d f:%d", e_size, f_size, e, f);
                    outfile->Printf("\n e: %d f:%d Ealpha = %8.8f Emixed = %8.8f Sum = %8.8f", e, f, Ealpha , Emixed, Ealpha + Emixed);
                }
            }
        }
    }
    //return (Ealpha + Ebeta + Emixed);
    return (Ealpha + Ebeta + Emixed);
}
}}
