//[forte-public]
#include <cmath>

#include <libmints/wavefunction.h>
#include <libmints/integral.h>
#include <lib3index/cholesky.h>
#include <libmints/basisset.h>
#include <libqt/qt.h>

#include "integrals.h"

using namespace ambit;

namespace psi{ namespace forte{

/**
     * @brief CholeskyIntegrals::CholeskyIntegrals
     * @param options - psi options class
     * @param restricted - type of integral transformation
     * @param resort_frozen_core -
     */
CholeskyIntegrals::CholeskyIntegrals(psi::Options &options, SharedWavefunction ref_wfn,  IntegralSpinRestriction restricted, IntegralFrozenCore resort_frozen_core,
std::shared_ptr<MOSpaceInfo> mo_space_info)
    : ForteIntegrals(options, restricted, resort_frozen_core, mo_space_info){

    wfn_ = ref_wfn;

    integral_type_ = Cholesky;
    outfile->Printf("\n  Cholesky integrals time");
    Timer CholInt;
    allocate();
    gather_integrals();
    make_diagonal_integrals();
    if (ncmo_ < nmo_){
        freeze_core_orbitals();
        // Set the new value of the number of orbitals to be used in indexing routines
        aptei_idx_ = ncmo_;
    }
    outfile->Printf("\n  CholeskyIntegrals take %8.8f", CholInt.get());
}

CholeskyIntegrals::~CholeskyIntegrals()
{
    deallocate();
}
void CholeskyIntegrals::make_diagonal_integrals()
{
    for(size_t p = 0; p < nmo_; ++p){
        for(size_t q = 0; q < nmo_; ++q){
            diagonal_aphys_tei_aa[p * nmo_ + q] = aptei_aa(p,q,p,q);
            diagonal_aphys_tei_ab[p * nmo_ + q] = aptei_ab(p,q,p,q);
            diagonal_aphys_tei_bb[p * nmo_ + q] = aptei_bb(p,q,p,q);
        }
    }
}

double CholeskyIntegrals::aptei_aa(size_t p, size_t q, size_t r, size_t s)
{
    double vpqrsalphaC = 0.0;
    double vpqrsalphaE = 0.0;
    vpqrsalphaC = C_DDOT(nthree_,
            &(ThreeIntegral_->pointer()[p*aptei_idx_ + r][0]),1,
            &(ThreeIntegral_->pointer()[q*aptei_idx_ + s][0]),1);
     vpqrsalphaE = C_DDOT(nthree_,
            &(ThreeIntegral_->pointer()[p*aptei_idx_ + s][0]),1,
            &(ThreeIntegral_->pointer()[q*aptei_idx_ + r][0]),1);

    return (vpqrsalphaC - vpqrsalphaE);

}

double CholeskyIntegrals::aptei_ab(size_t p, size_t q, size_t r, size_t s)
{
    double vpqrsalphaC = 0.0;
    vpqrsalphaC = C_DDOT(nthree_,
            &(ThreeIntegral_->pointer()[p*aptei_idx_ + r][0]),1 ,
            &(ThreeIntegral_->pointer()[q*aptei_idx_ + s][0]),1 );
    return (vpqrsalphaC);
}

double CholeskyIntegrals::aptei_bb(size_t p, size_t q, size_t r, size_t s)
{
    double vpqrsalphaC = 0.0, vpqrsalphaE = 0.0;
    vpqrsalphaC = C_DDOT(nthree_,
            &(ThreeIntegral_->pointer()[p*aptei_idx_ + r][0]),1,
            &(ThreeIntegral_->pointer()[q*aptei_idx_ + s][0]),1);
     vpqrsalphaE = C_DDOT(nthree_,
            &(ThreeIntegral_->pointer()[p*aptei_idx_ + s][0]),1,
            &(ThreeIntegral_->pointer()[q*aptei_idx_ + r][0]),1);
    //}

    return (vpqrsalphaC - vpqrsalphaE);
}
ambit::Tensor CholeskyIntegrals::aptei_aa_block(const std::vector<size_t>& p, const std::vector<size_t>& q, const std::vector<size_t>& r,
    const std::vector<size_t> & s)
{
    ambit::Tensor ReturnTensor = ambit::Tensor::build(tensor_type_,"Return",{p.size(),q.size(), r.size(), s.size()});
    ReturnTensor.iterate([&](const std::vector<size_t>& i,double& value){
        value = aptei_aa(p[i[0]], q[i[1]], r[i[2]], s[i[3]]);
    });
    return ReturnTensor;
}

ambit::Tensor CholeskyIntegrals::aptei_ab_block(const std::vector<size_t>& p, const std::vector<size_t>& q, const std::vector<size_t>& r,
    const std::vector<size_t> & s)
{
    ambit::Tensor ReturnTensor = ambit::Tensor::build(tensor_type_,"Return",{p.size(),q.size(), r.size(), s.size()});
    ReturnTensor.iterate([&](const std::vector<size_t>& i,double& value){
        value = aptei_ab(p[i[0]], q[i[1]], r[i[2]], s[i[3]]);
    });
    return ReturnTensor;
}

ambit::Tensor CholeskyIntegrals::aptei_bb_block(const std::vector<size_t>& p, const std::vector<size_t>& q, const std::vector<size_t>& r,
    const std::vector<size_t> & s)
{
    ambit::Tensor ReturnTensor = ambit::Tensor::build(tensor_type_,"Return",{p.size(),q.size(), r.size(), s.size()});
    ReturnTensor.iterate([&](const std::vector<size_t>& i,double& value){
        value = aptei_bb(p[i[0]], q[i[1]], r[i[2]], s[i[3]]);
    });
    return ReturnTensor;
}
ambit::Tensor CholeskyIntegrals::three_integral_block(const std::vector<size_t> &A, const std::vector<size_t> &p, const std::vector<size_t> &q)
{
    ambit::Tensor ReturnTensor = ambit::Tensor::build(tensor_type_,"Return",{A.size(), p.size(), q.size()});
    ReturnTensor.iterate([&](const std::vector<size_t>& i,double& value){
        value = three_integral(A[i[0]], p[i[1]], q[i[2]]);
    });
    return ReturnTensor;
}

void CholeskyIntegrals::gather_integrals()
{
    if(print_){outfile->Printf("\n Computing the Cholesky Vectors \n");}



    boost::shared_ptr<BasisSet> primary = wfn_->basisset();


    size_t nbf = primary->nbf();


    boost::shared_ptr<IntegralFactory> integral(new IntegralFactory(primary, primary, primary, primary));
    double tol_cd = options_.get_double("CHOLESKY_TOLERANCE");

    //This is creates the cholesky decomposed AO integrals
    Timer timer;
    std::string str= "Computing CD Integrals";
    if(print_){outfile->Printf("\n    %-36s ...", str.c_str());}
    boost::shared_ptr<CholeskyERI> Ch (new CholeskyERI(boost::shared_ptr<TwoBodyAOInt>(integral->eri()),0.0 ,tol_cd, Process::environment.get_memory()));
    //Computes the cholesky integrals
    Ch->choleskify();
    if(print_){outfile->Printf("...Done. Timing %15.6f s", timer.get());}

    //The number of vectors required to do cholesky factorization
    size_t nL = Ch->Q();
    nthree_ = nL;
    if(print_){outfile->Printf("\n Need %8.6f GB to store cd integrals in core\n",nL * nbf * nbf * sizeof(double) / 1073741824.0 );}
    int_mem_ = (nL * nbf * nbf * sizeof(double) / 1073741824.0);

    TensorType tensor_type = CoreTensor;

    if(print_){outfile->Printf("\n Number of cholesky vectors %d to satisfy %20.12f tolerance\n", nL,tol_cd);}
    SharedMatrix Lao = Ch->L();
    SharedMatrix L(new Matrix("Lmo", nL, (nmo_)*(nmo_)));
    SharedMatrix Ca_ao(new Matrix("Ca_ao",nso_,nmopi_.sum()));
    SharedMatrix Ca = wfn_->Ca();
    SharedMatrix aotoso = wfn_->aotoso();

    // Transform from the SO to the AO basis
    Dimension nsopi_ = wfn_->nsopi();
    for (int h = 0, index = 0; h < nirrep_; ++h){
        for (int i = 0; i < nmopi_[h]; ++i){
            int nao = nso_;
            int nso = nsopi_[h];

            if (!nso) continue;

            C_DGEMV('N',nao,nso,1.0,aotoso->pointer(h)[0],nso,&Ca->pointer(h)[0][i],nmopi_[h],0.0,&Ca_ao->pointer()[0][index],nmopi_.sum());

            index += 1;
        }

    }
//    Ca_ = Ca_ao;

    ambit::Tensor ThreeIntegral_ao = ambit::Tensor::build(tensor_type,"ThreeIndex",{nthree_,nmo_, nmo_ });
    ambit::Tensor Cpq_tensor = ambit::Tensor::build(tensor_type,"C_sorted",{nbf,nmo_});
    ambit::Tensor ThreeIntegral = ambit::Tensor::build(tensor_type,"ThreeIndex",{nthree_,nmo_, nmo_ });

    Cpq_tensor.iterate([&](const std::vector<size_t>& i,double& value){
        value = Ca_ao->get(i[0],i[1]);
    });
    ThreeIntegral_ao.iterate([&](const std::vector<size_t>& i,double& value){
        value = Lao->get(i[0],i[1]*nbf + i[2]);
    });
    SharedMatrix ThreeInt(new Matrix("Lmo", (nmo_)*(nmo_), nL));
    ThreeIntegral_ =ThreeInt;


    ThreeIntegral("L,p,q") = ThreeIntegral_ao("L,m,n")*Cpq_tensor("m,p")*Cpq_tensor("n,q");

    ThreeIntegral.iterate([&](const std::vector<size_t>& i,double& value){
        ThreeIntegral_->set(i[1]*nmo_ + i[2],i[0],value);
    });
}

void CholeskyIntegrals::allocate()
{
    // Allocate the memory required to store the one-electron integrals

    // Allocate the memory required to store the two-electron integrals
    diagonal_aphys_tei_aa = new double[nmo_ * nmo_];
    diagonal_aphys_tei_ab = new double[nmo_ * nmo_];
    diagonal_aphys_tei_bb = new double[nmo_ * nmo_];

}

void CholeskyIntegrals::deallocate()
{

    // Deallocate the memory required to store the one-electron integrals

    delete[] diagonal_aphys_tei_aa;
    delete[] diagonal_aphys_tei_ab;
    delete[] diagonal_aphys_tei_bb;

    //delete[] qt_pitzer_;
}

void CholeskyIntegrals::make_fock_matrix(SharedMatrix gamma_aM,SharedMatrix gamma_bM)
{
    TensorType tensor_type = CoreTensor;
    ambit::Tensor ThreeIntegralTensor = ambit::Tensor::build(tensor_type,"ThreeIndex",{nthree_,ncmo_, ncmo_ });
    ambit::Tensor gamma_a = ambit::Tensor::build(tensor_type, "Gamma_a",{ncmo_, ncmo_});
    ambit::Tensor gamma_b = ambit::Tensor::build(tensor_type, "Gamma_b",{ncmo_, ncmo_});
    ambit::Tensor fock_a = ambit::Tensor::build(tensor_type, "Fock_a",{ncmo_, ncmo_});
    ambit::Tensor fock_b = ambit::Tensor::build(tensor_type, "Fock_b",{ncmo_, ncmo_});

    ThreeIntegralTensor.iterate([&](const std::vector<size_t>& i,double& value){
        value = ThreeIntegral_->get(i[1]*aptei_idx_ + i[2], i[0]);
    });
    gamma_a.iterate([&](const std::vector<size_t>& i,double& value){
        value = gamma_aM->get(i[0],i[1]);
    });
    gamma_b.iterate([&](const std::vector<size_t>& i,double& value){
        value = gamma_bM->get(i[0],i[1]);
    });

    fock_a.iterate([&](const std::vector<size_t>& i,double& value){
        value = one_electron_integrals_a[i[0] * aptei_idx_ + i[1]];
    });

    fock_b.iterate([&](const std::vector<size_t>& i,double& value){
        value = one_electron_integrals_b[i[0] * aptei_idx_ + i[1]];
    });


    fock_a("p,q") +=  ThreeIntegralTensor("Q,p,q") * ThreeIntegralTensor("Q,r,s") * gamma_a("r,s");
    fock_a("p,q") -= ThreeIntegralTensor("Q,p,r") * ThreeIntegralTensor("Q,q,s") * gamma_a("r,s");
    fock_a("p,q") +=  ThreeIntegralTensor("Q,p,q") * ThreeIntegralTensor("Q,r,s") * gamma_b("r,s");

    fock_b("p,q") +=  ThreeIntegralTensor("Q,p,q") * ThreeIntegralTensor("Q,r,s") * gamma_b("r,s");
    fock_b("p,q") -= ThreeIntegralTensor("Q,p,r") * ThreeIntegralTensor("Q,q,s") * gamma_b("r,s");
    fock_b("p,q") +=  ThreeIntegralTensor("Q,p,q") * ThreeIntegralTensor("Q,r,s") * gamma_a("r,s");

    fock_a.iterate([&](const std::vector<size_t>& i,double& value){
        fock_matrix_a[i[0] * aptei_idx_ + i[1]] = value;
    });
    fock_b.iterate([&](const std::vector<size_t>& i,double& value){
        fock_matrix_b[i[0] * aptei_idx_ + i[1]] = value;
    });
}

void CholeskyIntegrals::resort_integrals_after_freezing()
{
    outfile->Printf("\n  Resorting integrals after freezing core.");

    // Create an array that maps the CMOs to the MOs (cmo2mo).
    std::vector<size_t> cmo2mo;
    for (int h = 0, q = 0; h < nirrep_; ++h){
        q += frzcpi_[h]; // skip the frozen core
        for (int r = 0; r < ncmopi_[h]; ++r){
            cmo2mo.push_back(q);
            q++;
        }
        q += frzvpi_[h]; // skip the frozen virtual
    }
    cmotomo_ = (cmo2mo);

    // Resort the integrals
    resort_two(one_electron_integrals_a,cmo2mo);
    resort_two(one_electron_integrals_b,cmo2mo);
    resort_two(diagonal_aphys_tei_aa,cmo2mo);
    resort_two(diagonal_aphys_tei_ab,cmo2mo);
    resort_two(diagonal_aphys_tei_bb,cmo2mo);

    resort_three(ThreeIntegral_,cmo2mo);

}
void CholeskyIntegrals::resort_three(boost::shared_ptr<Matrix>& threeint,std::vector<size_t>& map)
{
    //Create a temperature threeint matrix
    SharedMatrix temp_threeint(threeint->clone());
    temp_threeint->zero();

    // Borrwed from resort_four.
    // Since L is not sorted, only need to sort the columns
    // Surprisingly, this was pretty easy.
    for (size_t L = 0; L < nthree_; ++L){
        for (size_t q = 0; q < ncmo_; ++q){
            for (size_t r = 0; r < ncmo_; ++r){
                size_t Lpq_cmo  = q * ncmo_ + r;
                size_t Lpq_mo  = map[q] * nmo_ + map[r];
                temp_threeint->set(Lpq_cmo, L, threeint->get(Lpq_mo, L));

            }
        }
    }

    //This copies the resorted integrals and the data is changed to the sorted
    //matrix
    threeint->copy(temp_threeint);
}

void CholeskyIntegrals::set_tei(size_t, size_t, size_t, size_t, double, bool, bool)
{
    outfile->Printf("\n If you are using this, you are ruining the advantages of DF/CD");
    throw PSIEXCEPTION("Don't use DF/CD if you use set_tei");
}

}}
