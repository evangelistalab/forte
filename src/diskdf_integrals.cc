#include <cmath>

#include <psifiles.h>
#include <libiwl/iwl.h>
#include <libtrans/integraltransform.h>
#include <libpsio/psio.hpp>
#include <libmints/matrix.h>
#include <libmints/basisset.h>
#include <libthce/thce.h>
#include <libthce/thcew.h>
#include <libthce/lreri.h>
#include <lib3index/cholesky.h>
#include <libmints/mints.h>
#include <libqt/qt.h>
#include <libfock/jk.h>
#include <algorithm>
#include <numeric>
#include "blockedtensorfactory.h"
using namespace ambit;
namespace psi{ namespace forte{
DISKDFIntegrals::~DISKDFIntegrals()
{
    deallocate();
}

void DISKDFIntegrals::allocate()
{
    // Allocate the memory required to store the one-electron integrals
    // Allocate the memory required to store the two-electron integrals
    diagonal_aphys_tei_aa = new double[nmo_ * nmo_];
    diagonal_aphys_tei_ab = new double[nmo_ * nmo_];
    diagonal_aphys_tei_bb = new double[nmo_ * nmo_];

    //qt_pitzer_ = new int[nmo_];
}

double DISKDFIntegrals::aptei_aa(size_t p, size_t q, size_t r, size_t s)
{
    size_t pn, qn, rn, sn;

    if(frzcpi_.sum() > 0 && ncmo_ == aptei_idx_)
    {
        pn = cmotomo_[p];
        qn = cmotomo_[q];
        rn = cmotomo_[r];
        sn = cmotomo_[s];
    }
    else
    {
        pn = p;
        qn = q;
        rn = r;
        sn = s;
    }

    size_t offset1 = rn * nthree_ + pn * (nthree_ * nmo_);
    size_t offset2 = sn * nthree_ + qn * (nthree_ * nmo_);
    double vpqrsalphaC = 0.0;
    double vpqrsalphaE = 0.0;

    SharedVector B1(new Vector("B1", nthree_));
    SharedVector B2(new Vector("B2", nthree_));


    // Read a block of Vectors for the Columb term
    fseek(B_->file_pointer(), offset1 * sizeof(double), SEEK_SET);
    fread(&(B1->pointer()[0]), sizeof(double), nthree_, B_->file_pointer());
    fseek(B_->file_pointer(), offset2 * sizeof(double), SEEK_SET);
    fread(&(B2->pointer()[0]), sizeof(double), nthree_, B_->file_pointer());

    vpqrsalphaC = C_DDOT(nthree_,
            &(B1->pointer()[0]),1, &(B2->pointer()[0]),1);


     B1->zero();
     B2->zero();
     offset1 = 0;
     offset2 = 0;

    offset1 = sn * nthree_ + pn * (nthree_ * nmo_);
    offset2 = rn * nthree_ + qn * (nthree_ * nmo_);

    fseek(B_->file_pointer(), offset1 * sizeof(double), SEEK_SET);
    fread(&(B1->pointer()[0]), sizeof(double), nthree_, B_->file_pointer());
    fseek(B_->file_pointer(), offset2 * sizeof(double), SEEK_SET);
    fread(&(B2->pointer()[0]), sizeof(double), nthree_, B_->file_pointer());
     vpqrsalphaE = C_DDOT(nthree_,
            &(B1->pointer()[0]),1, &(B2->pointer()[0]),1);

    return (vpqrsalphaC - vpqrsalphaE);

}

double DISKDFIntegrals::aptei_ab(size_t p, size_t q, size_t r, size_t s)
{

   size_t pn, qn, rn, sn;
   if(frzcpi_.sum() > 0 && ncmo_ == aptei_idx_)
   {
       pn = cmotomo_[p];
       qn = cmotomo_[q];
       rn = cmotomo_[r];
       sn = cmotomo_[s];
   }
   else
   {
       pn = p;
       qn = q;
       rn = r;
       sn = s;
   }

   size_t offset1 = rn * nthree_ + pn * (nthree_ * nmo_);
   size_t offset2 = sn * nthree_ + qn * (nthree_ * nmo_);
   double vpqrsalphaC = 0.0;

   SharedVector B1(new Vector("B1", nthree_));
   SharedVector B2(new Vector("B2", nthree_));

   // Read a block of Vectors for the Columb term
   fseek(B_->file_pointer(), offset1 * sizeof(double), SEEK_SET);
   fread(&(B1->pointer()[0]), sizeof(double), nthree_, B_->file_pointer());
   fseek(B_->file_pointer(), offset2 * sizeof(double), SEEK_SET);
   fread(&(B2->pointer()[0]), sizeof(double), nthree_, B_->file_pointer());

   vpqrsalphaC = C_DDOT(nthree_,
                        &(B1->pointer()[0]),1, &(B2->pointer()[0]),1);

   return (vpqrsalphaC);

}

double DISKDFIntegrals::aptei_bb(size_t p, size_t q, size_t r, size_t s)
{
    size_t pn, qn, rn, sn;

    if(frzcpi_.sum() > 0 && ncmo_ == aptei_idx_)
    {
        pn = cmotomo_[p];
        qn = cmotomo_[q];
        rn = cmotomo_[r];
        sn = cmotomo_[s];
    }
    else
    {
        pn = p;
        qn = q;
        rn = r;
        sn = s;
    }

    size_t offset1 = rn * nthree_ + pn * (nthree_ * nmo_);
    size_t offset2 = sn * nthree_ + qn * (nthree_ * nmo_);
    double vpqrsalphaC = 0.0;
    double vpqrsalphaE = 0.0;

    SharedVector B1(new Vector("B1", nthree_));
    SharedVector B2(new Vector("B2", nthree_));

    // Read a block of Vectors for the Columb term
    fseek(B_->file_pointer(), offset1 * sizeof(double), SEEK_SET);
    fread(&(B1->pointer()[0]), sizeof(double), nthree_, B_->file_pointer());
    fseek(B_->file_pointer(), offset2 * sizeof(double), SEEK_SET);
    fread(&(B2->pointer()[0]), sizeof(double), nthree_, B_->file_pointer());

    vpqrsalphaC = C_DDOT(nthree_,
            &(B1->pointer()[0]),1, &(B2->pointer()[0]),1);

     B1->zero();
     B2->zero();
     offset1 = 0;
     offset2 = 0;

    offset1 = sn * nthree_ + pn * (nthree_ * nmo_);
    offset2 = rn * nthree_ + qn * (nthree_ * nmo_);
    fseek(B_->file_pointer(), offset1 * sizeof(double), SEEK_SET);
    fread(&(B1->pointer()[0]), sizeof(double), nthree_, B_->file_pointer());
    fseek(B_->file_pointer(), offset2 * sizeof(double), SEEK_SET);
    fread(&(B2->pointer()[0]), sizeof(double), nthree_, B_->file_pointer());
     vpqrsalphaE = C_DDOT(nthree_,
            &(B1->pointer()[0]),1, &(B2->pointer()[0]),1);

    return (vpqrsalphaC - vpqrsalphaE);

}
ambit::Tensor DISKDFIntegrals::aptei_aa_block(const std::vector<size_t>& p, const std::vector<size_t>& q, const std::vector<size_t>& r,
    const std::vector<size_t> & s)

{
    ambit::Tensor ThreeIntpr = ambit::Tensor::build(tensor_type_, "ThreeInt", {nthree_, p.size(), r.size()});
    ambit::Tensor ThreeIntqs = ambit::Tensor::build(tensor_type_, "ThreeInt", {nthree_, q.size(), s.size()});
    std::vector<size_t> Avec(nthree_);
    std::iota(Avec.begin(), Avec.end(), 0);

    ThreeIntpr = get_three_integral_block(Avec, p, r);
    ThreeIntqs = get_three_integral_block(Avec, q, s);

    ambit::Tensor ReturnTensor = ambit::Tensor::build(tensor_type_,"Return",{p.size(),q.size(), r.size(), s.size()});
    ReturnTensor ("p,q,r,s") = ThreeIntpr("A,p,r") * ThreeIntqs("A,q,s");
    ReturnTensor ("p,q,r,s") -= ThreeIntpr("A,p,s") * ThreeIntqs("A,q,r");

    return ReturnTensor;
}

ambit::Tensor DISKDFIntegrals::aptei_ab_block(const std::vector<size_t>& p, const std::vector<size_t>& q, const std::vector<size_t>& r,
    const std::vector<size_t> & s)
{
    ambit::Tensor ThreeIntpr = ambit::Tensor::build(tensor_type_, "ThreeInt", {nthree_, p.size(), r.size()});
    ambit::Tensor ThreeIntqs = ambit::Tensor::build(tensor_type_, "ThreeInt", {nthree_, q.size(), s.size()});
    std::vector<size_t> Avec(nthree_);
    std::iota(Avec.begin(), Avec.end(), 0);

    ThreeIntpr = get_three_integral_block(Avec, p, r);
    ThreeIntqs = get_three_integral_block(Avec, q, s);

    ambit::Tensor ReturnTensor = ambit::Tensor::build(tensor_type_,"Return",{p.size(),q.size(), r.size(), s.size()});
    ReturnTensor ("p,q,r,s") = ThreeIntpr("A,p,r") * ThreeIntqs("A,q,s");

    return ReturnTensor;
}

ambit::Tensor DISKDFIntegrals::aptei_bb_block(const std::vector<size_t>& p, const std::vector<size_t>& q, const std::vector<size_t>& r,
    const std::vector<size_t> & s)
{
    ambit::Tensor ThreeIntpr = ambit::Tensor::build(tensor_type_, "ThreeInt", {nthree_, p.size(), r.size()});
    ambit::Tensor ThreeIntqs = ambit::Tensor::build(tensor_type_, "ThreeInt", {nthree_, q.size(), s.size()});
    std::vector<size_t> Avec(nthree_);
    std::iota(Avec.begin(), Avec.end(), 0);

    ThreeIntpr = get_three_integral_block(Avec, p, r);
    ThreeIntqs = get_three_integral_block(Avec, q, s);

    ambit::Tensor ReturnTensor = ambit::Tensor::build(tensor_type_,"Return",{p.size(),q.size(), r.size(), s.size()});
    ReturnTensor ("p,q,r,s") = ThreeIntpr("A,p,r") * ThreeIntqs("A,q,s");
    ReturnTensor ("p,q,r,s") -= ThreeIntpr("A,p,s") * ThreeIntqs("A,q,r");
    return ReturnTensor;
}
double DISKDFIntegrals::get_three_integral(size_t A, size_t p, size_t q)
{
    size_t pn, qn;
    if(frzcpi_.sum() > 0 && ncmo_ == aptei_idx_)
    {
        pn = cmotomo_[p];
        qn = cmotomo_[q];
    }
    else
    {
        pn = p;
        qn = q;
    }


    size_t offset1 = pn * (nthree_ * nmo_) + qn * nthree_ + A;
    double value = 0.0;
    fseek(B_->file_pointer(), offset1 * sizeof(double), SEEK_SET);
    fread(&value, sizeof(double), 1, B_->file_pointer());
    return value;

}
ambit::Tensor DISKDFIntegrals::get_three_integral_block(const std::vector<size_t> &A, const std::vector<size_t> &p, const std::vector<size_t> &q)
{
    //Since file is formatted as p by A * q
    bool frozen_core = false;

    ambit::Tensor ReturnTensor = ambit::Tensor::build(tensor_type_,"Return",{A.size(), p.size(), q.size()});
    std::vector<double>& ReturnTensorV = ReturnTensor.data();

    if(frzcpi_.sum() && aptei_idx_==ncmo_)
    {
        frozen_core = true;
    }

    size_t pn, qn;
    if(nthree_ == A.size())
    {
        std::vector<boost::shared_ptr<Matrix> > p_by_Aq;
        for (auto p_block : p)
        {
            if(frozen_core)
            {
                pn = cmotomo_[p_block];
            }
            else
            {
                pn = p_block;
            }

            boost::shared_ptr<Matrix> Aq(new Matrix("Aq", nmo_, nthree_));

            fseek(B_->file_pointer(), pn*nthree_*nmo_*sizeof(double), SEEK_SET);
            fread(&(Aq->pointer()[0][0]), sizeof(double), nmo_ * nthree_, B_->file_pointer());
            p_by_Aq.push_back(Aq);


        }
        if(frozen_core)
        {
            ReturnTensor.iterate([&](const std::vector<size_t>& i,double& value){
                value = p_by_Aq[i[1]]->get(cmotomo_[q[i[2]]], A[i[0]]);
            });
        }
        else
        {
            ReturnTensor.iterate([&](const std::vector<size_t>& i,double& value){
                value = p_by_Aq[i[1]]->get(q[i[2]], A[i[0]]);
            });

        }
    }
    else
    {
        //If user wants blocking in A
        pn = 0;
        qn = 0;
        //If p and q are not just nmo_, this map corrects that.
        //If orbital 0, 5, 10, 15 is frozen, this corresponds to 0, 1, 2, 3.
        //This map says p_map[5] = 1.
        //Used in correct ordering for the tensor.
        std::map<size_t, size_t> p_map;
        std::map<size_t, size_t> q_map;

        int p_idx = 0;
        int q_idx = 0;
    for(size_t p_block : p)
    {
        p_map[p_block] = p_idx;
        p_idx++;
    }
    for(size_t q_block : q)
    {
        q_map[q_block] = q_idx;
        q_idx++;
    }
        for(size_t p_block : p)
        {
            pn = frozen_core ? cmotomo_[p_block] : p_block;

            for(size_t q_block : q)
            {
                qn = frozen_core ? cmotomo_[q_block] : q_block;

                double* A_chunk = new double[A.size()];
                size_t offset = pn * nthree_ * nmo_ + qn * nthree_ + A[0];
                fseek(B_->file_pointer(), offset * sizeof(double), SEEK_SET);
                fread(&(A_chunk[0]), sizeof(double), A.size(), B_->file_pointer());
                for(size_t a = 0; a < A.size(); a++)
                {
                    //Weird way the tensor is formatted
                    //Fill the tensor for every chunk of A
                    ReturnTensorV[a * p.size() * q.size() + p_map[p_block] * q.size() + q_map[q_block]] = A_chunk[a];
                }
                delete[] A_chunk;

            }
        }
    }
    return ReturnTensor;
}

void DISKDFIntegrals::set_tei(size_t p, size_t q, size_t r,size_t s,double value,bool alpha1,bool alpha2)
{
    outfile->Printf("\n If you are using this, you are ruining the advantages of DF/CD");
    throw PSIEXCEPTION("Don't use DF/CD if you use set_tei");
}

void DISKDFIntegrals::gather_integrals()
{
    boost::shared_ptr<Wavefunction> wfn = Process::environment.wavefunction();

    outfile->Printf("\n Computing Density fitted integrals \n");

    boost::shared_ptr<BasisSet> primary = wfn->basisset();
    boost::shared_ptr<BasisSet> auxiliary = BasisSet::pyconstruct_orbital(primary->molecule(), "DF_BASIS_MP2",options_.get_str("DF_BASIS_MP2"));

    size_t nprim = primary->nbf();
    size_t naux  = auxiliary->nbf();
    nthree_ = naux;
    outfile->Printf("\n Number of auxiliary basis functions:  %u", naux);
    outfile->Printf("\n Need %8.6f GB to store DF integrals\n", (nprim * nprim * naux * sizeof(double)/1073741824.0));
    int_mem_ = (nprim * nprim * naux * sizeof(double));

    Dimension nsopi_ = wfn->nsopi();
    SharedMatrix aotoso = wfn->aotoso();
    SharedMatrix Ca = wfn->Ca();
    SharedMatrix Ca_ao(new Matrix("Ca_ao",nso_,nmopi_.sum()));

    // Transform from the SO to the AO basis
    for (size_t h = 0, index = 0; h < nirrep_; ++h){
        for (size_t i = 0; i < nmopi_[h]; ++i){
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
//    df->set_C(C_ord);
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
    df->print_header();
    df->compute();
    outfile->Printf("...Done. Timing %15.6f s", timer.get());

    boost::shared_ptr<Tensor> B = df->ints()["B"];
    B_ = B;
    df.reset();

    //outfile->Printf("\n %8.8f integral", aptei_ab(10,8,5,2));

}

void DISKDFIntegrals::make_diagonal_integrals()
{
    for(size_t p = 0; p < nmo_; ++p){
        for(size_t q = 0; q < nmo_; ++q){
            diagonal_aphys_tei_aa[p * nmo_ + q] = aptei_aa(p,q,p,q);
            diagonal_aphys_tei_ab[p * nmo_ + q] = aptei_ab(p,q,p,q);
            diagonal_aphys_tei_bb[p * nmo_ + q] = aptei_bb(p,q,p,q);
        }
    }
}

DISKDFIntegrals::DISKDFIntegrals(psi::Options &options, IntegralSpinRestriction restricted, IntegralFrozenCore resort_frozen_core)
    : ForteIntegrals(options, restricted, resort_frozen_core){

    integral_type_ = DiskDF;
    outfile->Printf("\n DISKDFIntegrals overall time");
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

    gather_integrals();
    make_diagonal_integrals();
    if (ncmo_ < nmo_){
        freeze_core_orbitals();
        // Set the new value of the number of orbitals to be used in indexing routines
        aptei_idx_ = ncmo_;
    }

    outfile->Printf("\n DISKDFIntegrals take %15.8f s", DFInt.get());
}

void DISKDFIntegrals::update_integrals(bool freeze_core)
{
    make_diagonal_integrals();
    if (freeze_core){
        if (ncmo_ < nmo_){
            freeze_core_orbitals();
            aptei_idx_ = ncmo_;
        }
    }
}

void DISKDFIntegrals::retransform_integrals()
{
    aptei_idx_ = nmo_;
    transform_one_electron_integrals();
    //TODO:  Remove this function from retransform
    //For DF, reread integrals and then transfrom to new basis
    gather_integrals();
    update_integrals();
}

void DISKDFIntegrals::deallocate()
{

    // Deallocate the memory required to store the one-electron integrals
    // Allocate the memory required to store the two-electron integrals

    delete[] diagonal_aphys_tei_aa;
    delete[] diagonal_aphys_tei_ab;
    delete[] diagonal_aphys_tei_bb;
    //delete[] qt_pitzer_;
}
void DISKDFIntegrals::make_fock_matrix(SharedMatrix gamma_aM,SharedMatrix gamma_bM)
{
    //Efficient calculation of fock matrix from disk
    //Since gamma_aM is very sparse (diagonal elements of core and active block)
    //Only nonzero contributions are on diagonal elements
    //Grab the nonzero elements and put to a vector

    TensorType tensor_type = kCore;

    //Create the fock_a and fock_b globally
    //Choose to block over naux rather than ncmo_
    ambit::Tensor fock_a = ambit::Tensor::build(tensor_type, "Fock_a",{aptei_idx_, aptei_idx_});
    ambit::Tensor fock_b = ambit::Tensor::build(tensor_type, "Fock_b",{aptei_idx_, aptei_idx_});

    std::vector<size_t> nonzero;
    //Figure out exactly what I need to contract the Coloumb term
    for(int i = 0; i < ncmo_; i++)
    {
        if(gamma_aM->get(i,i) > 1e-10)
        {
            nonzero.push_back(i);
        }
    }

    fock_a.iterate([&](const std::vector<size_t>& i,double& value){
        value = one_electron_integrals_a[i[0] * aptei_idx_ + i[1]];
    });

    fock_b.iterate([&](const std::vector<size_t>& i,double& value){
        value = one_electron_integrals_b[i[0] * aptei_idx_ + i[1]];
    });


    std::vector<size_t> A(nthree_);
    std::iota(A.begin(), A.end(), 0);

    std::vector<size_t> P(aptei_idx_);
    std::iota(P.begin(), P.end(), 0);

    //Create a gamma that contains only nonzero terms
    ambit::Tensor gamma_a = ambit::Tensor::build(tensor_type, "Gamma_a",{nonzero.size(), nonzero.size()});
    ambit::Tensor gamma_b = ambit::Tensor::build(tensor_type, "Gamma_b",{nonzero.size(), nonzero.size()});
    //Create the full gamma (K is not nearly as sparse as J)
    ambit::Tensor gamma_a_full = ambit::Tensor::build(tensor_type, "Gamma_a",{aptei_idx_, aptei_idx_});
    ambit::Tensor gamma_b_full = ambit::Tensor::build(tensor_type, "Gamma_b",{aptei_idx_, aptei_idx_});

    gamma_a.iterate([&](const std::vector<size_t>& i,double& value){
        value = gamma_aM->get(nonzero[i[0]],nonzero[i[1]]);
    });
    gamma_b.iterate([&](const std::vector<size_t>& i,double& value){
        value = gamma_bM->get(nonzero[i[0]],nonzero[i[1]]);
    });
    ambit::Tensor ThreeIntC2 = ambit::Tensor::build(tensor_type, "ThreeInkC", {nthree_,nonzero.size(), nonzero.size()});
    ThreeIntC2 = get_three_integral_block(A, nonzero, nonzero);

    ambit::Tensor BQA = ambit::Tensor::build(tensor_type, "BQ", {nthree_});
    ambit::Tensor BQB = ambit::Tensor::build(tensor_type, "BQ", {nthree_});
    //Do a contraction over Naux * n_h^2 * n_h^2 -> naux * n_h^2
    BQA("B") = ThreeIntC2("B,r,s") * gamma_a("r,s");
    BQB("B") = ThreeIntC2("B,r,s") * gamma_b("r,s");
    //Grab the data from this for the block iteration
    std::vector<double>& BQAv = BQA.data();
    std::vector<double>& BQBv = BQB.data();

    gamma_a_full.iterate([&](const std::vector<size_t>& i,double& value){
        value = gamma_aM->get(i[0],i[1]);
    });
    gamma_b_full.iterate([&](const std::vector<size_t>& i,double& value){
        value = gamma_bM->get(i[0],i[1]);
    });

    //====Blocking information==========
    size_t int_mem_int_ = (nthree_ * ncmo_ * ncmo_) * sizeof(double);
    int memory_input = Process::environment.get_memory();
    int num_block = std::ceil(double(int_mem_int_) / memory_input);
    //Hard wires num_block for testing

    int block_size = nthree_ / num_block;

    if(num_block != 1)
    {
        outfile->Printf("\n\n\n\n\t---------Blocking Information-------\n\n\n\n\t");
    outfile->Printf("\n  %d / %d = %d", int_mem_int_, memory_input, int_mem_int_ / memory_input);
    outfile->Printf("\n  Block_size = %d\n num_block = %d", block_size, num_block);
    }

    Timer block_read;
    for(int i = 0; i < num_block; i++)
    {
        std::vector<size_t> A_block;
        if(nthree_ % num_block == 0)
        {
            A_block.resize(block_size);
            std::iota(A_block.begin(), A_block.end(), i * block_size);
        }
        else
        {
            block_size = i==(num_block - 1) ? block_size + nthree_ % num_block : block_size;
            A_block.resize(block_size);
            std::iota(A_block.begin(), A_block.end(), i * (nthree_ / num_block));
        }

        //Create a tensor of TI("Q,r,p")
        ambit::Tensor BQA_small = ambit::Tensor::build(tensor_type, "BQ", {A_block.size()});
        ambit::Tensor BQB_small = ambit::Tensor::build(tensor_type, "BQ", {A_block.size()});

       //Calculate the smaller block of A from the global block of prior Brs * gamma_rs
       BQA_small.iterate([&](const std::vector<size_t>& i,double& value){
            value = BQAv[A_block[i[0]]];
        });
       BQB_small.iterate([&](const std::vector<size_t>& i,double& value){
            value = BQBv[A_block[i[0]]];
        });
        ambit::Tensor ThreeIntegralTensor = ambit::Tensor::build(tensor_type,"ThreeIndex",{A_block.size(),aptei_idx_, aptei_idx_});

        //ThreeIntegralTensor.iterate([&](const std::vector<size_t>& i,double& value){
        //    value = get_three_integral(A_block[i[0]], i[1], i[2]);
        //});

        //Return a tensor of ThreeInt given the smaller block of A
        ThreeIntegralTensor = get_three_integral_block(A_block, P, P);


        //Need to rewrite this to at least read in chunks of nthree_
        //ThreeIntegralTensor = get_three_integral_block(A_block, P,P );

        fock_a("p,q") +=  ThreeIntegralTensor("Q,p,q") * BQA_small("Q");
        fock_a("p,q") -=  ThreeIntegralTensor("Q,p,r") * ThreeIntegralTensor("Q,q,s") * gamma_a_full("r,s");
        fock_a("p,q") +=  ThreeIntegralTensor("Q,p,q") * BQB_small("Q");

        fock_b("p,q") +=  ThreeIntegralTensor("Q,p,q") * BQB_small("Q");
        fock_b("p,q") -=  ThreeIntegralTensor("Q,p,r") * ThreeIntegralTensor("Q,q,s") * gamma_b_full("r,s");
        fock_b("p,q") +=  ThreeIntegralTensor("Q,p,q") * BQA_small("Q");

        A_block.clear();
    }
    fock_a.iterate([&](const std::vector<size_t>& i,double& value){
        fock_matrix_a[i[0] * aptei_idx_ + i[1]] = value;
    });
    fock_b.iterate([&](const std::vector<size_t>& i,double& value){
        fock_matrix_b[i[0] * aptei_idx_ + i[1]] = value;
    });

    if(num_block!=1)
    {
        outfile->Printf("\n Created Fock matrix %8.8f s", block_read.get());
    }

}

void DISKDFIntegrals::make_fock_matrix(bool* Ia, bool* Ib)
{
    for(size_t p = 0; p < ncmo_; ++p){
        for(size_t q = 0; q < ncmo_; ++q){
            // Builf Fock Diagonal alpha-alpha
            fock_matrix_a[p * ncmo_ + q] = oei_a(p,q);
            // Add the non-frozen alfa part, the forzen core part is already included in oei
            for (int k = 0; k < ncmo_; ++k) {
                if (Ia[k]) {
                    fock_matrix_a[p * ncmo_ + q] += aptei_aa(p,k,q,k);
                }
                if (Ib[k]) {
                    fock_matrix_a[p * ncmo_ + q] += aptei_ab(p,k,q,k);
                }
            }
            fock_matrix_b[p * ncmo_ + q] = oei_b(p,q);
            // Add the non-frozen alfa part, the forzen core part is already included in oei
            for (int k = 0; k < ncmo_; ++k) {
                if (Ib[k]) {
                    fock_matrix_b[p * ncmo_ + q] += aptei_bb(p,k,q,k);
                }
                if (Ia[k]) {
                    fock_matrix_b[p * ncmo_ + q] += aptei_ab(p,k,q,k);
                }
            }
        }
    }
}

void DISKDFIntegrals::make_fock_matrix(const boost::dynamic_bitset<>& Ia,const boost::dynamic_bitset<>& Ib)
{
    for(size_t p = 0; p < ncmo_; ++p){
        for(size_t q = p; q < ncmo_; ++q){
            // Builf Fock Diagonal alpha-alpha
            double fock_a_pq = oei_a(p,q);
            //            fock_matrix_a[p * ncmo_ + q] = oei_a(p,q);
            // Add the non-frozen alfa part, the forzen core part is already included in oei
            for (int k = 0; k < ncmo_; ++k) {
                if (Ia[k]) {
                    fock_a_pq += aptei_aa(p,k,q,k);
                }
                if (Ib[k]) {
                    fock_a_pq += aptei_ab(p,k,q,k);
                }
            }
            fock_matrix_a[p * ncmo_ + q] = fock_matrix_a[q * ncmo_ + p] = fock_a_pq;
            double fock_b_pq = oei_b(p,q);
            // Add the non-frozen alfa part, the forzen core part is already included in oei
            for (int k = 0; k < ncmo_; ++k) {
                if (Ib[k]) {
                    fock_b_pq += aptei_bb(p,k,q,k);
                }
                if (Ia[k]) {
                    fock_b_pq += aptei_ab(p,k,q,k);
                }
            }
            fock_matrix_b[p * ncmo_ + q] = fock_matrix_b[q * ncmo_ + p] = fock_b_pq;
        }
    }
}

void DISKDFIntegrals::make_fock_diagonal(bool* Ia, bool* Ib, std::pair<std::vector<double>, std::vector<double> > &fock_diagonals)
{
    std::vector<double>& fock_diagonal_alpha = fock_diagonals.first;
    std::vector<double>& fock_diagonal_beta = fock_diagonals.second;
    for(size_t p = 0; p < ncmo_; ++p){
        // Builf Fock Diagonal alpha-alpha
        fock_diagonal_alpha[p] =  oei_a(p,p);// roei(p,p);
        // Add the non-frozen alfa part, the forzen core part is already included in oei
        for (int k = 0; k < ncmo_; ++k) {
            if (Ia[k]) {
                //                fock_diagonal_alpha[p] += diag_ce_rtei(p,k); //rtei(p,p,k,k) - rtei(p,k,p,k);
                fock_diagonal_alpha[p] += diag_aptei_aa(p,k); //rtei(p,p,k,k) - rtei(p,k,p,k);
            }
            if (Ib[k]) {
                //                fock_diagonal_alpha[p] += diag_c_rtei(p,k); //rtei(p,p,k,k);
                fock_diagonal_alpha[p] += diag_aptei_ab(p,k); //rtei(p,p,k,k);
            }
        }
        fock_diagonal_beta[p] =  oei_b(p,p);
        // Add the non-frozen alfa part, the forzen core part is already included in oei
        for (int k = 0; k < ncmo_; ++k) {
            if (Ib[k]) {
                //                fock_diagonal_beta[p] += diag_ce_rtei(p,k); //rtei(p,p,k,k) - rtei(p,k,p,k);
                fock_diagonal_beta[p] += diag_aptei_bb(p,k); //rtei(p,p,k,k) - rtei(p,k,p,k);
            }
            if (Ia[k]) {
                //                fock_diagonal_beta[p] += diag_c_rtei(p,k); //rtei(p,p,k,k);
                fock_diagonal_beta[p] += diag_aptei_ab(p,k); //rtei(p,p,k,k);
            }
        }
    }
}

void DISKDFIntegrals::make_alpha_fock_diagonal(bool* Ia, bool* Ib,std::vector<double> &fock_diagonal)
{
    for(size_t p = 0; p < ncmo_; ++p){
        // Builf Fock Diagonal alpha-alpha
        fock_diagonal[p] = oei_a(p,p);
        // Add the non-frozen alfa part, the forzen core part is already included in oei
        for (int k = 0; k < ncmo_; ++k) {
            if (Ia[k]) {
                fock_diagonal[p] += diag_aptei_aa(p,k);  //diag_ce_rtei(p,k); //rtei(p,p,k,k) - rtei(p,k,p,k);
            }
            if (Ib[k]) {
                fock_diagonal[p] += diag_aptei_ab(p,k); // diag_c_rtei(p,k); //rtei(p,p,k,k);
            }
        }
    }
}

void DISKDFIntegrals::make_beta_fock_diagonal(bool* Ia, bool* Ib, std::vector<double> &fock_diagonals)
{
    for(size_t p = 0; p < ncmo_; ++p){
        fock_diagonals[p] = oei_b(p,p);
        // Add the non-frozen alfa part, the forzen core part is already included in oei
        for (int k = 0; k < ncmo_; ++k) {
            if (Ia[k]) {
                fock_diagonals[p] += diag_aptei_ab(p,k);  //diag_c_rtei(p,k); //rtei(p,p,k,k);
            }
            if (Ib[k]) {
                fock_diagonals[p] += diag_aptei_bb(p,k);  //diag_ce_rtei(p,k); //rtei(p,p,k,k) - rtei(p,k,p,k);
            }
        }
    }
}

void DISKDFIntegrals::resort_three(SharedMatrix& threeint,std::vector<size_t>& map)
{
    outfile->Printf("No need to resort a file.  dummy!");
}

void DISKDFIntegrals::freeze_core_orbitals()
{
    Timer freezeOrbs;
    compute_frozen_core_energy();
    compute_frozen_one_body_operator();
    if (resort_frozen_core_ == RemoveFrozenMOs){
        resort_integrals_after_freezing();
    }
    outfile->Printf("\n Frozen Orbitals takes %8.8f s", freezeOrbs.get());
}

void DISKDFIntegrals::compute_frozen_core_energy()
{
    Timer FrozenEnergy;
    frozen_core_energy_ = 0.0;

    for (int hi = 0, p = 0; hi < nirrep_; ++hi){
        for (int i = 0; i < frzcpi_[hi]; ++i){
            frozen_core_energy_ += oei_a(p + i,p + i) + oei_b(p + i,p + i);

            for (int hj = 0, q = 0; hj < nirrep_; ++hj){
                for (int j = 0; j < frzcpi_[hj]; ++j){
                    frozen_core_energy_ += 0.5 * diag_aptei_aa(p + i,q + j) + 0.5 * diag_aptei_bb(p + i,q + j) + diag_aptei_ab(p + i,q + j);
                }
                q += nmopi_[hj]; // orbital offset for the irrep hj
            }
        }
        p += nmopi_[hi]; // orbital offset for the irrep hi
    }
    outfile->Printf("\n  Frozen-core energy        %20.12f a.u.",frozen_core_energy_);
    outfile->Printf("\n\n Frozen_Core_Energy takes   %8.8f s", FrozenEnergy.get());
}

void DISKDFIntegrals::compute_frozen_one_body_operator()
{
    Timer FrozenOneBody;
    boost::shared_ptr<BlockedTensorFactory>BTF(new BlockedTensorFactory(options_));
    ambit::BlockedTensor::reset_mo_spaces();

    size_t f = 0; // The Offset for irrep
    size_t r = 0; // The MO number for frozen core
    size_t g = 0; //the Offset for irrep
    std::vector<size_t> frozen_vec;
    std::vector<size_t> corrleated_vec;

    for (size_t hi = 0; hi < nirrep_; ++hi){
        for (size_t i = 0; i < frzcpi_[hi]; ++i){
            r = f + i;
            frozen_vec.push_back(r);
        }
        f += nmopi_[hi];
        for (size_t p = frzcpi_[hi]; p < nmopi_[hi]; p++)
        {
            size_t mo = p + g;
            corrleated_vec.push_back(mo);
        }
        g += nmopi_[hi];
    }
    //Get the size of frozen MO
    size_t frozen_size = frozen_vec.size();

    //Form a map that says mo_to_rel[ABS_MO] =relative in frozen array
    std::map<size_t, size_t>  mo_to_rel;
    std::vector<size_t> motofrozen(frozen_size);
    std::iota(motofrozen.begin(), motofrozen.end(), 0);

    int i = 0;
    for (auto frozen : frozen_vec)
    {
        mo_to_rel[frozen] = motofrozen[i];
        i++;
    }

    std::vector<size_t> nauxpi(nthree_);
    std::iota(nauxpi.begin(), nauxpi.end(),0);

    std::vector<size_t> P(nmo_);
    std::iota(P.begin(), P.end(), 0);

    // <rp || rq> + <rp | rq>
    // = B_{rr}^{Q} * B_{pq}^{Q} - B_{rp}^{Q} B_{qr}^{Q} + B_{rr}^{Q} * B_{pq}^{Q}
    // => Assume that I can store all but B_{pq}^{Q} in core (maybe a big assumption . . . . . )
    //Kevin is lazy.  Going to use ambit to perform this contraction
    //Create three tensors.  Forgive my terrible naming
    //B_{rr}^{Q}
    // B_{rp}^{Q}
    ambit::Tensor BQrp = ambit::Tensor::build(tensor_type_, "BQrp", {nthree_, frozen_vec.size(), nmo_});
    // B_{pr}^{Q}
    ambit::Tensor BQpr = ambit::Tensor::build(tensor_type_, "BQrp", {nthree_, nmo_,frozen_vec.size()});

    //Read these into a tensor
    BQrp = get_three_integral_block(nauxpi, frozen_vec, P);
    BQpr = get_three_integral_block(nauxpi, P, frozen_vec);

    ambit::Tensor rpq = ambit::Tensor::build(tensor_type_, "rpq", {frozen_vec.size(), nmo_, nmo_});
    ambit::Tensor rpqK = ambit::Tensor::build(tensor_type_, "rpqK", {frozen_vec.size(), nmo_, nmo_});

    //Form the exchange part of this out of loop for blocks
    //Need to see if this ever gets too large
    rpqK("r,p,q") = BQrp("Q,r,p") * BQpr("Q,q,r");

    //====Blocking information==========
    //Hope this is smart enough to figure out not to store B_{pq}^{Q}
    //Major problem with this is that other tensors are not that small.
    //Maybe won't work
    size_t int_mem_int_ = (nthree_ * nmo_ * nmo_) * sizeof(double);
    int memory_input = Process::environment.get_memory();
    int num_block = std::ceil(double(int_mem_int_) / memory_input);
    //Hard wires num_block for testing

    int block_size = nthree_ / num_block;

    if(num_block != 1)
    {
        outfile->Printf("\n\n\n\n\t---------Blocking Information-------\n\n\n\n\t");
        outfile->Printf("\n  %d / %d = %d", int_mem_int_, memory_input, int_mem_int_ / memory_input);
        outfile->Printf("\nTotal size = %d  Block_size = %d\n num_block = %d",nthree_ ,block_size, num_block);
        outfile->Printf("\n\n rpq is %4.4f GB", nthree_ * frozen_size * nmo_ * 8.0 / (1024 * 1024 * 1024));
        outfile->Printf("\n\n BpqQ is %4.4f GB", nthree_ * nmo_ * nmo_ * 8.0 / (1024 * 1024 * 1024));
    }

    for(int i = 0; i < num_block; i++)
    {
        std::vector<size_t> A_block;
        if(nthree_ % num_block == 0)
        {
            A_block.resize(block_size);
            std::iota(A_block.begin(), A_block.end(), i * block_size);
        }
        else
        {
            block_size = i==(num_block - 1) ? block_size + nthree_ % num_block : block_size;
            A_block.resize(block_size);
            std::iota(A_block.begin(), A_block.end(), i * (nthree_ / num_block));
        }
        //This tensor is extremely large for big systems
        //Needs to be blocked over A

        ambit::Tensor BQpq = ambit::Tensor::build(tensor_type_, "BQpq", {A_block.size(), nmo_, nmo_});
        BQpq = get_three_integral_block(A_block, P, P);

        //A_block is the split of the naux -> (0, . . . ,block_size)

        ambit::Tensor Qr = ambit::Tensor::build(tensor_type_, "Qr", {A_block.size(),frozen_vec.size()});
        ambit::Tensor BQr = get_three_integral_block(A_block, frozen_vec, frozen_vec);



        //Go from B_{rr}^{Q} -> B_{r}^{Q}.  Tensor library can not do this.

        std::vector<double>& BQrP = BQr.data(); // get the data from BQr
        Qr.iterate([&](const std::vector<size_t>& i,double& value){
            value = BQrP[i[0] * frozen_size * frozen_size + i[1] * frozen_size + i[1]] ;
        });
        //^^^^^ -> B_{rr}^{Q} -> B_{r}^{Q}
        rpq("r,p,q") += 2.0 * Qr("Q,r") * BQpq("Q,p,q");

    }
    std::vector<double>& rpqP = rpq.data();
    std::vector<double>& rpqKP = rpqK.data();
    f = 0;
    for (size_t hi = 0; hi < nirrep_; ++hi){
        for (size_t i = 0; i < frzcpi_[hi]; ++i){
            size_t r = f + i;
            outfile->Printf("\n  Freezing MO %lu", r);
            #pragma omp parallel for num_threads(num_threads_) \
            schedule(dynamic)
            for(size_t p = 0; p < nmo_; ++p){
                for(size_t q = 0; q < nmo_; ++q){
                    one_electron_integrals_a[p * nmo_ + q] += rpqP[mo_to_rel[r] * nmo_ * nmo_ + p * nmo_ +  q]
                            - rpqKP[mo_to_rel[r] * nmo_ * nmo_ + p * nmo_ + q];
                    one_electron_integrals_b[p * nmo_ + q] += rpqP[mo_to_rel[r] * nmo_ * nmo_ +p * nmo_ + q]
                            - rpqKP[mo_to_rel[r] * nmo_ * nmo_ + p * nmo_ + q];

                }
            }
        }
        f += nmopi_[hi];
    }
    ambit::BlockedTensor::reset_mo_spaces();

    outfile->Printf("\n\n FrozenOneBody Operator takes  %8.8f s", FrozenOneBody.get());

}

void DISKDFIntegrals::resort_integrals_after_freezing()
{
    Timer resort_integrals;
    outfile->Printf("\n  Resorting integrals after freezing core.");

    // Create an array that maps the CMOs to the MOs (cmo2mo).

    // Resort the integrals
    resort_two(one_electron_integrals_a,cmotomo_);
    resort_two(one_electron_integrals_b,cmotomo_);
    resort_two(diagonal_aphys_tei_aa,cmotomo_);
    resort_two(diagonal_aphys_tei_ab,cmotomo_);
    resort_two(diagonal_aphys_tei_bb,cmotomo_);

    //resort_three(ThreeIntegral_,cmo2mo);

    outfile->Printf("\n Resorting integrals takes   %8.8fs", resort_integrals.get());
}
}}
