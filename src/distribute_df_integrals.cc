//[forte-public]
#include <cmath>
#include <numeric>

#include <libmints/matrix.h>
#include <libthce/lreri.h>
#include <libmints/basisset.h>
#include <libthce/thce.h>
#include <libqt/qt.h>
#include "integrals.h"
#include <cassert>
#ifdef HAVE_GA
    #include <ga.h>
    #include <macdecls.h>
    #include <mpi.h>
#endif


#include "blockedtensorfactory.h"

using namespace ambit;
namespace psi{ namespace forte{

DistDFIntegrals::DistDFIntegrals(psi::Options &options, SharedWavefunction ref_wfn, IntegralSpinRestriction restricted, IntegralFrozenCore resort_frozen_core,
std::shared_ptr<MOSpaceInfo> mo_space_info)
    : ForteIntegrals(options, ref_wfn, restricted, resort_frozen_core, mo_space_info){

    wfn_ = ref_wfn;

    integral_type_ = DistDF;
    outfile->Printf("\n  DistDFIntegrals overall time");
    Timer DFInt;
    allocate();

    //Form a correlated mo to mo before I create integrals
    std::vector<size_t> cmo2mo;
    for (int h = 0, q = 0; h < nirrep_; ++h){
        q += frzcpi_[h]; // skip the frozen core
        for (int r = 0; r < ncmopi_[h]; ++r){
            cmo2mo.push_back(q);
            q++;
        }
        q += frzvpi_[h]; // skip the frozen virtual
    }
    cmotomo_ = cmo2mo;
    int my_proc = 0;
    #ifdef HAVE_GA
        my_proc = GA_Nodeid();
    #endif

    if(my_proc == 0) gather_integrals();
    MPI_Bcast(&nthree_, 1, MPI_INT, 0, MPI_COMM_WORLD);
    create_dist_df();
    //make_diagonal_integrals();
    if (ncmo_ < nmo_){
        if(my_proc == 0) freeze_core_orbitals();
        // Set the new value of the number of orbitals to be used in indexing routines
        aptei_idx_ = ncmo_;
    }

    outfile->Printf("\n  DISKDFIntegrals take %15.8f s", DFInt.get());
}

DistDFIntegrals::~DistDFIntegrals()
{
    deallocate();
}
void DistDFIntegrals::gather_integrals()
{
        outfile->Printf("\n Computing Density fitted integrals \n");

        boost::shared_ptr<BasisSet> primary = wfn_->basisset();
        if(options_.get_str("DF_BASIS_MP2").length() == 0)
        {
            outfile->Printf("\n Please set a DF_BASIS_MP2 option to a specified auxiliary basis set");
            throw PSIEXCEPTION("Select a DF_BASIS_MP2 for use with DFIntegrals");
        }

        boost::shared_ptr<BasisSet> auxiliary = BasisSet::pyconstruct_orbital(primary->molecule(), "DF_BASIS_MP2",options_.get_str("DF_BASIS_MP2"));

        size_t nprim = primary->nbf();
        size_t naux  = auxiliary->nbf();
        nthree_ = naux;
        outfile->Printf("\n Number of auxiliary basis functions:  %u", naux);
        outfile->Printf("\n Need %8.6f GB to store DF integrals\n", (nprim * nprim * naux * sizeof(double)/1073741824.0));
        int_mem_ = (nprim * nprim * naux * sizeof(double));

        Dimension nsopi_ = wfn_->nsopi();
        SharedMatrix aotoso = wfn_->aotoso();
        SharedMatrix Ca = wfn_->Ca();
        SharedMatrix Ca_ao(new Matrix("Ca_ao",nso_,nmopi_.sum()));

        // Transform from the SO to the AO basis
        for (int h = 0, index = 0; h < nirrep_; ++h){
            for (int i = 0; i < nmopi_[h]; ++i){
                size_t nao = nso_;
                size_t nso = nsopi_[h];

                if (!nso) continue;

                C_DGEMV('N',nao,nso,1.0,aotoso->pointer(h)[0],nso,&Ca->pointer(h)[0][i],nmopi_[h],0.0,&Ca_ao->pointer()[0][index],nmopi_.sum());

                index += 1;
            }

        }


        //B_{pq}^Q -> MO without frozen core

        //Constructs the DF function
        //I used this version of build as this doesn't build all the apces and assume a RHF/UHF reference
        boost::shared_ptr<DFERI> df = DFERI::build(primary,auxiliary,options_);

        //Pushes a C matrix that is ordered in pitzer ordering
        //into the C_matrix object
        df->set_C(Ca_ao);
        //set_C clears all the orbital spaces, so this creates the space
        //This space creates the total nmo_.
        //This assumes that everything is correlated.
        df->add_space("ALL", 0, nmo_);
        //Does not add the pair_space, but says which one is should use
        df->add_pair_space("B", "ALL", "ALL");
        df->set_memory(Process::environment.get_memory()/8L);

        //Finally computes the df integrals
        //Does the timings also
        Timer timer;
        std::string str= "Computing DF Integrals";
        outfile->Printf("\n    %-36s ...", str.c_str());
        df->compute();
        outfile->Printf("...Done. Timing %15.6f s", timer.get());

        boost::shared_ptr<Tensor> B = df->ints()["B"];
        B_ = B;
        df.reset();
}
void DistDFIntegrals::create_dist_df()
{
    int dim[2], chunk[2];
    dim[0] = nthree_;
    dim[1] = nmo_ * nmo_;
    chunk[0] = -1;
    chunk[1] = nmo_ * nmo_;

    DistDF_ga_ = NGA_Create(C_DBL, 2, dim, (char *)"DistributedDF", chunk);
    if(DistDF_ga_ == 0)
        GA_Error("DistributedDF failed on creating the tensor", 1);
    GA_Print_distribution(DistDF_ga_);
    int my_proc = GA_Nodeid();
    int num_proc = GA_Nnodes();
    if(my_proc == 0)
    {
        for(int iproc = 0; iproc < num_proc; iproc++)
        {
            int begin_offset[2];
            int end_offset[2];
            int stride[1];
            stride[0] = nmo_ * nmo_;
            NGA_Distribution(DistDF_ga_, iproc, begin_offset, end_offset);
            std::vector<int> begin_offset_vec = {begin_offset[0], begin_offset[1]};
            std::vector<int> end_offset_vec = {end_offset[0], end_offset[1]};
            ambit::Tensor B_per_process = read_integral_chunk(B_, begin_offset_vec, end_offset_vec);
            NGA_Put(DistDF_ga_, begin_offset, end_offset, &(B_per_process.data()[0]), stride);
        }
    }
    GA_Print_distribution(DistDF_ga_);
    GA_Print(DistDF_ga_);
}

ambit::Tensor DistDFIntegrals::read_integral_chunk(boost::shared_ptr<Tensor>& B,std::vector<int>& lo, std::vector<int>& hi)
{
    assert(lo.size()==2);
    assert(hi.size()==2);
    /// This tells what block of naux is on the processor that calls this
    size_t naux_block_size = hi[0] - lo[0] + 1; 
    ambit::Tensor ReturnTensor = ambit::Tensor::build(tensor_type_, "Return", {naux_block_size, nmo_, nmo_});
    std::vector<double>& ReturnTensorV = ReturnTensor.data();
    /// Allocate a vector that is blocked via naux dimension
    std::vector<size_t> naux(naux_block_size, 0);
    /// Fill the tensor starting from start of block and end once it hits end
    std::iota(naux.begin(), naux.end(), lo[0]);
    //If p and q are not just nmo_, this map corrects that.
    //If orbital 0, 5, 10, 15 is frozen, this corresponds to 0, 1, 2, 3.
    //This map says p_map[5] = 1.
    //Used in correct ordering for the tensor.
    
    std::vector<size_t> all_mos = mo_space_info_->get_corr_abs_mo("ALL");
    int p_idx = 0;
    int q_idx = 0;
    std::vector<size_t> p_map(nmo_);
    std::vector<size_t> q_map(nmo_);

    for(size_t p_block : all_mos)
    {
        p_map[p_block] = p_idx;
        p_idx++;
    }
    for(size_t q_block : all_mos)
    {
        q_map[q_block] = q_idx;
        q_idx++;
    }

    for(size_t p_block = 0; p_block < nmo_; p_block++)
    {
        size_t pn = p_block;
        for(size_t q_block = 0; q_block < nmo_; q_block++)
        {
            size_t qn = q_block;
            double* A_chunk = new double[naux_block_size];
            size_t offset = pn * nthree_ * nmo_ + qn * nthree_ + naux[0];
            fseek(B_->file_pointer(), offset * sizeof(double), SEEK_SET);
            fread(&(A_chunk[0]), sizeof(double), naux_block_size, B_->file_pointer());
            for(size_t a = 0; a < naux_block_size; a++)
            {
                //Weird way the tensor is formatted
                //Fill the tensor for every chunk of A
                //ReturnTensorV[a * nmo_ * nmo_ + p_map[p_block] * nmo_ + q_map[q_block]] = A_chunk[a];
                ReturnTensorV[a * nmo_ * nmo_ + p_block * nmo_ + q_block] = A_chunk[a];

            }
            delete[] A_chunk;
    
        }
    }
    return ReturnTensor;
}
void DistDFIntegrals::deallocate()
{
    GA_Destroy(DistDF_ga_);
    delete[] diagonal_aphys_tei_aa;
    delete[] diagonal_aphys_tei_ab;
    delete[] diagonal_aphys_tei_bb;
}
void DistDFIntegrals::allocate()
{
    diagonal_aphys_tei_aa = new double[nmo_ * nmo_];
    diagonal_aphys_tei_ab = new double[nmo_ * nmo_];
    diagonal_aphys_tei_bb = new double[nmo_ * nmo_];
}
//double DistDFIntegrals::aptei_aa(size_t p, size_t q, size_t r, size_t s)
//{
//    ambit::Tensor pqrs = aptei_aa_block({p}, {q}, {r}, {s});
//    return pqrs.data()[0];
//}
//double DistDFIntegrals::aptei_ab(size_t p, size_t q, size_t r, size_t s)
//{
//    ambit::Tensor pqrs = aptei_ab_block({p}, {q}, {r}, {s});
//    return pqrs.data()[0];
//}
//double DistDFIntegrals::aptei_bb(size_t p, size_t q, size_t r, size_t s)
//{
//    ambit::Tensor pqrs = aptei_bb_block({p}, {q}, {r}, {s});
//    return pqrs.data()[0];
//}
ambit::Tensor DistDFIntegrals::three_integral_block(const std::vector<size_t>& A, const std::vector<size_t>& p, const std::vector<size_t>& q)
{
    ambit::Tensor ReturnTensor = ambit::Tensor::build(tensor_type_, "Return", {A.size(), p.size(), q.size()});
    std::vector<double>& ReturnTensorV = ReturnTensor.data();
    bool frozen_core = false;

    if(frzcpi_.sum() && aptei_idx_ == ncmo_)
        frozen_core = true;

    size_t pn, qn;
    /// DistDF_ga_ is distributed via A dimension.  
    /// A lot of logic needs to be done to figure out where information lies
    bool on_one_core = false;
    int subscript_begin[2];
    int subscript_end[2];
    subscript_begin[0] = A[0];
    subscript_begin[1] = 0;
    subscript_end[0] = A[A.size() - 1];
    subscript_end[1] = nmo_ * nmo_;
    int proc_begin = NGA_Locate(DistDF_ga_, subscript_begin);
    int proc_end   = NGA_Locate(DistDF_ga_, subscript_end);
    if(proc_begin == -1 or proc_end == -1)
    {
        printf("\n proc_begin: %d", proc_begin);
        printf("\n proc_end: %d", proc_end);
        printf("\n Found multiple blocks that hold this region");
        printf("\n A = (%d, %d), p = (%d, %d), q = (%d, %d)", A[0], A[A.size() - 1], p[0], p[p.size() - 1], q[0], q[q.size() - 1]);
        printf("\n my_proc: %d", GA_Nodeid());
        throw PSIEXCEPTION("GA could not locate region of THREE_INTEGRAL");
    }
    /// If NGA_Locate gives same processor for begin and end,
    /// it is given that the integral is located on one core
    if(proc_begin == proc_end)
    {
        on_one_core = true;
    }
    if(on_one_core)
    {
        int one_core_offset_begin[2];
        int one_core_offset_end[2];
        int ld[1];
        ld[0] = nmo_ * nmo_;
        ///Read entire chunk of requested array
        NGA_Distribution(DistDF_ga_, proc_begin, one_core_offset_begin, one_core_offset_end);
        std::vector<double> buffer;
        ///Buffer can be larger than what was requested
        ///If(p or q < nmo_), then will read more than I want, but thats okay(I Think).
        size_t buffer_size = one_core_offset_end[0] - one_core_offset_begin[0] + 1 + nmo_ * nmo_;
        buffer.resize(buffer_size);
        NGA_Get(DistDF_ga_, one_core_offset_begin, one_core_offset_end, &buffer[0], ld);
        return ReturnTensor;

    }





}


}}

