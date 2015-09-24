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
#include <libqt/qt.h>
#include <algorithm>
#include <numeric>
#include "blockedtensorfactory.h"
#include <libfock/jk.h>

using namespace ambit;
namespace psi{ namespace forte{
//Class for the DF Integrals
//Generates DF Integrals.  Freezes Core orbitals, computes integrals, and resorts integrals.  Also computes fock matrix

DFIntegrals::~DFIntegrals()
{
    deallocate();
}

void DFIntegrals::allocate()
{
    // Allocate the memory required to store the one-electron integrals
    // Allocate the memory required to store the two-electron integrals
    diagonal_aphys_tei_aa = new double[nmo_ * nmo_];
    diagonal_aphys_tei_ab = new double[nmo_ * nmo_];
    diagonal_aphys_tei_bb = new double[nmo_ * nmo_];

    //qt_pitzer_ = new int[nmo_];
}

double DFIntegrals::aptei_aa(size_t p, size_t q, size_t r, size_t s)
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

double DFIntegrals::aptei_ab(size_t p, size_t q, size_t r, size_t s)
{
    double vpqrsalphaC = 0.0;
    vpqrsalphaC = C_DDOT(nthree_,
            &(ThreeIntegral_->pointer()[p*aptei_idx_ + r][0]),1,
            &(ThreeIntegral_->pointer()[q*aptei_idx_ + s][0]),1);

    return (vpqrsalphaC);

}

double DFIntegrals::aptei_bb(size_t p, size_t q, size_t r, size_t s)
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

ambit::Tensor DFIntegrals::aptei_aa_block(const std::vector<size_t>& p, const std::vector<size_t>& q, const std::vector<size_t>& r,
    const std::vector<size_t> & s)
{
    ambit::Tensor ReturnTensor = ambit::Tensor::build(tensor_type_,"Return",{p.size(),q.size(), r.size(), s.size()});
    ReturnTensor.iterate([&](const std::vector<size_t>& i,double& value){
        value = aptei_aa(p[i[0]], q[i[1]], r[i[2]], s[i[3]]);
    });
    return ReturnTensor;
}

ambit::Tensor DFIntegrals::aptei_ab_block(const std::vector<size_t>& p, const std::vector<size_t>& q, const std::vector<size_t>& r,
    const std::vector<size_t> & s)
{
    ambit::Tensor ReturnTensor = ambit::Tensor::build(tensor_type_,"Return",{p.size(),q.size(), r.size(), s.size()});
    ReturnTensor.iterate([&](const std::vector<size_t>& i,double& value){
        value = aptei_ab(p[i[0]], q[i[1]], r[i[2]], s[i[3]]);
    });
    return ReturnTensor;
}

ambit::Tensor DFIntegrals::aptei_bb_block(const std::vector<size_t>& p, const std::vector<size_t>& q, const std::vector<size_t>& r,
    const std::vector<size_t> & s)
{
    ambit::Tensor ReturnTensor = ambit::Tensor::build(tensor_type_,"Return",{p.size(),q.size(), r.size(), s.size()});
    ReturnTensor.iterate([&](const std::vector<size_t>& i,double& value){
        value = aptei_bb(p[i[0]], q[i[1]], r[i[2]], s[i[3]]);
    });
    return ReturnTensor;
}
ambit::Tensor DFIntegrals::three_integral_block(const std::vector<size_t> &A, const std::vector<size_t> &p, const std::vector<size_t> &q)
{
    ambit::Tensor ReturnTensor = ambit::Tensor::build(tensor_type_,"Return",{A.size(), p.size(), q.size()});
    ReturnTensor.iterate([&](const std::vector<size_t>& i,double& value){
        value = three_integral(A[i[0]], p[i[1]], q[i[2]]);
    });
    return ReturnTensor;
}

void DFIntegrals::set_tei(size_t, size_t, size_t ,size_t,double,bool, bool)
{
    outfile->Printf("\n If you are using this, you are ruining the advantages of DF/CD");
    throw PSIEXCEPTION("Don't use DF/CD if you use set_tei");
}

void DFIntegrals::gather_integrals()
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

    Dimension nsopi_ = wfn->nsopi();
    SharedMatrix aotoso = wfn->aotoso();
    SharedMatrix Ca = wfn->Ca();
    //SharedMatrix Ca_ao(new Matrix("Ca_ao",nso_,nmopi_.sum()));
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
    Ca_ = Ca_ao;
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

    boost::shared_ptr<psi::Tensor> B = df->ints()["B"];
    df.reset();

    FILE* Bf = B->file_pointer();
    //SharedMatrix Bpq(new Matrix("Bpq", nmo_, nmo_ * naux));
    SharedMatrix Bpq(new Matrix("Bpq", nmo_ * nmo_, naux));

    //Reads the DF integrals into Bpq.  Stores them as nmo by (nmo*naux)

    std::string str_seek= "Seeking DF Integrals";
    outfile->Printf("\n    %-36s ...", str_seek.c_str());
    fseek(Bf,0L, SEEK_SET);
    outfile->Printf("...Done. Timing %15.6f s", timer.get());

    std::string str_read = "Reading DF Integrals";
    outfile->Printf("\n   %-36s . . .", str_read.c_str());
    fread(&(Bpq->pointer()[0][0]), sizeof(double),naux*(nmo_)*(nmo_), Bf);
    outfile->Printf("...Done. Timing %15.6f s", timer.get());

    //This has a different dimension than two_electron_integrals in the integral code that francesco wrote.
    //This is because francesco reads only the nonzero integrals
    //I store all of them into this array.

    // Store the integrals in the form of nmo*nmo by B
    //Makes a gemm call very easy
    //std::string re_sort = "Resorting DF Integrals";
    //outfile->Printf("\n   %-36s ...",re_sort.c_str());
    //for (size_t p = 0; p < nmo_; ++p){
    //    for (size_t q = 0; q < nmo_; ++q){
    //        // <pq||rs> = <pq|rs> - <pq|sr> = (pr|qs) - (ps|qr)
    //        for(size_t B = 0; B < naux; B++){
    //            size_t qB = q * naux + B;
    //            tBpq->set(B,p*nmo_ + q, Bpq->get(p,qB));
    //        }
    //     }
    //}
    outfile->Printf("...Done.  Timing %15.6f s", timer.get());

    ThreeIntegral_ = Bpq;
    //outfile->Printf("\n %8.8f integral", aptei_ab(10,8,5,2));

}

void DFIntegrals::make_diagonal_integrals()
{
    for(size_t p = 0; p < nmo_; ++p){
        for(size_t q = 0; q < nmo_; ++q){
            diagonal_aphys_tei_aa[p * nmo_ + q] = aptei_aa(p,q,p,q);
            diagonal_aphys_tei_ab[p * nmo_ + q] = aptei_ab(p,q,p,q);
            diagonal_aphys_tei_bb[p * nmo_ + q] = aptei_bb(p,q,p,q);
        }
    }
}

DFIntegrals::DFIntegrals(psi::Options &options, IntegralSpinRestriction restricted, IntegralFrozenCore resort_frozen_core,
std::shared_ptr<MOSpaceInfo> mo_space_info)
    : ForteIntegrals(options, restricted, resort_frozen_core, mo_space_info){
    integral_type_ = DF;

    outfile->Printf("\n DFIntegrals overall time");
    Timer DFInt;
    allocate();
    gather_integrals();
    make_diagonal_integrals();
    if (ncmo_ < nmo_){
        freeze_core_orbitals();
        // Set the new value of the number of orbitals to be used in indexing routines
        aptei_idx_ = ncmo_;
    }
    outfile->Printf("\n DFIntegrals take %15.8f s", DFInt.get());
}

void DFIntegrals::update_integrals(bool freeze_core)
{
    make_diagonal_integrals();
    if (freeze_core){
        if (ncmo_ < nmo_){
            freeze_core_orbitals();
            aptei_idx_ = ncmo_;
        }
    }
}

void DFIntegrals::retransform_integrals()
{
    aptei_idx_ = nmo_;
    transform_one_electron_integrals();
    //TODO:  Remove this function from retransform
    //For DF, reread integrals and then transfrom to new basis
    gather_integrals();
    update_integrals();
}

void DFIntegrals::deallocate()
{

    // Deallocate the memory required to store the one-electron integrals
    // Allocate the memory required to store the two-electron integrals

    delete[] diagonal_aphys_tei_aa;
    delete[] diagonal_aphys_tei_ab;
    delete[] diagonal_aphys_tei_bb;
    //delete[] qt_pitzer_;
}
void DFIntegrals::make_fock_matrix(SharedMatrix gamma_aM,SharedMatrix gamma_bM)
{
    TensorType tensor_type = kCore;
    ambit::Tensor ThreeIntegralTensor = ambit::Tensor::build(tensor_type,"ThreeIndex",{nthree_,ncmo_, ncmo_ });
    ambit::Tensor gamma_a = ambit::Tensor::build(tensor_type, "Gamma_a",{ncmo_, ncmo_});
    ambit::Tensor gamma_b = ambit::Tensor::build(tensor_type, "Gamma_b",{ncmo_, ncmo_});
    ambit::Tensor fock_a = ambit::Tensor::build(tensor_type, "Fock_a",{ncmo_, ncmo_});
    ambit::Tensor fock_b = ambit::Tensor::build(tensor_type, "Fock_b",{ncmo_, ncmo_});

    //ThreeIntegralTensor.iterate([&](const std::vector<size_t>& i,double& value){
    //    value = ThreeIntegral_->get(i[0],i[1]*aptei_idx_ + i[2]);
    //});
    std::vector<size_t> vQ(nthree_);
    std::iota(vQ.begin(), vQ.end(), 0);
    std::vector<size_t> vP(ncmo_);
    std::iota(vP.begin(), vP.end(), 0);


    ThreeIntegralTensor = three_integral_block(vQ, vP, vP);

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

    ///Changing the Q_pr * Q_qs  to Q_rp * Q_sq for convience for reading

    //ambit::Tensor test = ambit::Tensor::build(tensor_type, "Fock_b",{nthree_});
    //test("Q") = ThreeIntegralTensor("Q,r,s") * gamma_a("r,s");
    //fock_a("p,q") += ThreeIntegralTensor("Q,p,q")*test("Q");
    fock_a("p,q") +=  ThreeIntegralTensor("Q,p,q") * ThreeIntegralTensor("Q,r,s") * gamma_a("r,s");
    fock_a("p,q") -=  ThreeIntegralTensor("Q,r,p") * ThreeIntegralTensor("Q,s,q") * gamma_a("r,s");
    fock_a("p,q") +=  ThreeIntegralTensor("Q,p,q") * ThreeIntegralTensor("Q,r,s") * gamma_b("r,s");

    fock_b("p,q") +=  ThreeIntegralTensor("Q,p,q") * ThreeIntegralTensor("Q,r,s") * gamma_b("r,s");
    fock_b("p,q") -=  ThreeIntegralTensor("Q,r,p") * ThreeIntegralTensor("Q,s,q") * gamma_b("r,s");
    fock_b("p,q") +=  ThreeIntegralTensor("Q,p,q") * ThreeIntegralTensor("Q,r,s") * gamma_a("r,s");



    fock_a.iterate([&](const std::vector<size_t>& i,double& value){
        fock_matrix_a[i[0] * aptei_idx_ + i[1]] = value;
    });
    fock_b.iterate([&](const std::vector<size_t>& i,double& value){
        fock_matrix_b[i[0] * aptei_idx_ + i[1]] = value;
    });

}

void DFIntegrals::make_fock_matrix(bool* Ia, bool* Ib)
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

void DFIntegrals::make_fock_matrix(const boost::dynamic_bitset<>& Ia,const boost::dynamic_bitset<>& Ib)
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

void DFIntegrals::make_fock_diagonal(bool* Ia, bool* Ib, std::pair<std::vector<double>, std::vector<double> > &fock_diagonals)
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

void DFIntegrals::make_alpha_fock_diagonal(bool* Ia, bool* Ib,std::vector<double> &fock_diagonal)
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

void DFIntegrals::make_beta_fock_diagonal(bool* Ia, bool* Ib, std::vector<double> &fock_diagonals)
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

void DFIntegrals::resort_three(SharedMatrix& threeint,std::vector<size_t>& map)
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
                temp_threeint->set(Lpq_cmo, L , threeint->get(Lpq_mo, L));

            }
        }
    }

    //This copies the resorted integrals and the data is changed to the sorted
    //matrix
    outfile->Printf("\n Done with resorting");
    threeint->copy(temp_threeint);
}

void DFIntegrals::freeze_core_orbitals()
{
    Timer freezeOrbs;
    compute_frozen_core_energy();
    compute_frozen_one_body_operator();
    if (resort_frozen_core_ == RemoveFrozenMOs){
        resort_integrals_after_freezing();
    }
    outfile->Printf("\n Frozen Orbitals takes %8.8f s", freezeOrbs.get());
}

void DFIntegrals::compute_frozen_core_energy()
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

void DFIntegrals::compute_frozen_one_body_operator()
{
    Timer FrozenOneBody;

    std::vector<size_t> frozen_dim_abs = mo_space_info_->get_absolute_mo("FROZEN_DOCC");
    SharedMatrix C_core(new Matrix("C_core",nmo_, frozen_dim_abs.size()));
    // Need to get the frozen block of the C matrix
    for(size_t mu = 0; mu < nmo_; mu++){
        for(size_t i = 0; i < frozen_dim_abs.size(); i++){
            C_core->set(mu, i, Ca_->get(mu, frozen_dim_abs[i]));
        }
    }

    boost::shared_ptr<JK> JK_core = JK::build_JK();

    JK_core->set_memory(Process::environment.get_memory() * 0.8);
    /// Already transform everything to C1 so make sure JK does not do this.
    JK_core->set_allow_desymmetrization(false);

    /////TODO: Make this an option in my code
    //JK_core->set_cutoff(options_.get_double("INTEGRAL_SCREENING"));
    JK_core->set_cutoff(options_.get_double("INTEGRAL_SCREENING"));
    JK_core->initialize();

    std::vector<boost::shared_ptr<Matrix> >&Cl = JK_core->C_left();

    Cl.clear();
    Cl.push_back(C_core);

    JK_core->compute();

    SharedMatrix F_core = JK_core->J()[0];
    SharedMatrix K_core = JK_core->K()[0];

    F_core->scale(2.0);
    F_core->subtract(K_core);
    F_core->transform(Ca_);
    for(size_t p = 0; p < nmo_; ++p){
        for(size_t q = 0; q < nmo_; ++q){
            one_electron_integrals_a[p * nmo_ + q] += F_core->get(p, q);
            one_electron_integrals_b[p * nmo_ + q] += F_core->get(p ,q);
        }
    }


    ambit::BlockedTensor::reset_mo_spaces();
    outfile->Printf("\n\n FrozenOneBody Operator takes  %8.8f s", FrozenOneBody.get());

}

void DFIntegrals::resort_integrals_after_freezing()
{
    Timer resort_integrals;
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

    outfile->Printf("\n Resorting integrals takes   %8.8fs", resort_integrals.get());
}


} }
