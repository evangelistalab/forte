//[forte-public]
#include <cmath>
#include <numeric>

#include <libmints/matrix.h>
#include <libthce/lreri.h>
#include <libmints/basisset.h>
#include <libthce/thce.h>
#include <libqt/qt.h>
#include "integrals.h"
#ifdef HAVE_GA
    #include <ga.h>
    #include <macdecls.h>
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

    if(my_proc == 0)
    {
        gather_integrals();
        //make_diagonal_integrals();
        if (ncmo_ < nmo_){
            freeze_core_orbitals();
            // Set the new value of the number of orbitals to be used in indexing routines
            aptei_idx_ = ncmo_;
        }

        outfile->Printf("\n  DISKDFIntegrals take %15.8f s", DFInt.get());
    }
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
    int dim[2], chunk[2];
    dim[0] = nthree_;
    dim[1] = nmo_ * nmo_;
    chunk[0] = -1;
    chunk[1] = nmo_ * nmo_;

    DistDF_ga_ = NGA_Create(C_DBL, 2, dim, (char *)"DistributedDF", chunk);

    //outfile->Printf("\n %8.8f integral", aptei_ab(10,8,5,2));

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


}}

