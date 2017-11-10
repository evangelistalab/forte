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

#include <cmath>
#include <numeric>

#include "psi4/libpsi4util/process.h"
#include "psi4/libmints/basisset.h"
#include "psi4/libmints/matrix.h"
#include "psi4/libmints/vector.h"
#include "psi4/libmints/wavefunction.h"
#include "psi4/libqt/qt.h"

#include "../helpers.h"

#ifdef HAVE_GA
#include <ga.h>
#include <macdecls.h>
#endif

#include "../blockedtensorfactory.h"

using namespace ambit;
namespace psi {
namespace forte {

DISKDFIntegrals::DISKDFIntegrals(psi::Options& options, SharedWavefunction ref_wfn,
                                 IntegralSpinRestriction restricted,
                                 IntegralFrozenCore resort_frozen_core,
                                 std::shared_ptr<MOSpaceInfo> mo_space_info)
    : ForteIntegrals(options, ref_wfn, restricted, resort_frozen_core, mo_space_info) {

    wfn_ = ref_wfn;

    integral_type_ = DiskDF;
    outfile->Printf("\n  DISKDFIntegrals overall time");
    Timer DFInt;
    allocate();

    // Form a correlated mo to mo before I create integrals
    std::vector<size_t> cmo2mo;
    for (int h = 0, q = 0; h < nirrep_; ++h) {
        q += frzcpi_[h]; // skip the frozen core
        for (int r = 0; r < ncmopi_[h]; ++r) {
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

    if (my_proc == 0) {
        gather_integrals();
        // make_diagonal_integrals();
        if (ncmo_ < nmo_) {
            freeze_core_orbitals();
            // Set the new value of the number of orbitals to be used in
            // indexing routines
            aptei_idx_ = ncmo_;
        }

        outfile->Printf("\n  DISKDFIntegrals take %15.8f s", DFInt.get());
    }
}

DISKDFIntegrals::~DISKDFIntegrals() { deallocate(); }

void DISKDFIntegrals::allocate() {
    // Allocate the memory required to store the one-electron integrals
    // Allocate the memory required to store the two-electron integrals
    diagonal_aphys_tei_aa = new double[nmo_ * nmo_];
    diagonal_aphys_tei_ab = new double[nmo_ * nmo_];
    diagonal_aphys_tei_bb = new double[nmo_ * nmo_];

    // qt_pitzer_ = new int[nmo_];
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
    std::vector<size_t> p_range = {pn,pn+1};
    std::vector<size_t> q_range = {qn,qn+1};
    std::vector<size_t> r_range = {rn,rn+1};
    std::vector<size_t> s_range = {sn,sn+1};

    double vpqrsalphaC = 0.0;
    double vpqrsalphaE = 0.0;

    SharedMatrix B1(new Matrix(1,nthree_));
    SharedMatrix B2(new Matrix(1,nthree_));

    df_->fill_tensor("B",B1, A_range, p_range, r_range ); 
    df_->fill_tensor("B",B2, A_range, q_range, s_range ); 

    vpqrsalphaC = B1->vector_dot(B2);

    B1->zero();
    B2->zero();

    df_->fill_tensor("B",B1, A_range, p_range, s_range ); 
    df_->fill_tensor("B",B2, A_range, q_range, r_range ); 

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
    std::vector<size_t> p_range = {pn,pn+1};
    std::vector<size_t> q_range = {qn,qn+1};
    std::vector<size_t> r_range = {rn,rn+1};
    std::vector<size_t> s_range = {sn,sn+1};

    double vpqrsalphaC = 0.0;
    SharedMatrix B1(new Matrix(1,nthree_));
    SharedMatrix B2(new Matrix(1,nthree_));

    df_->fill_tensor("B",B1, A_range, p_range, r_range ); 
    df_->fill_tensor("B",B2, A_range, q_range, s_range ); 

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
    std::vector<size_t> p_range = {pn,pn+1};
    std::vector<size_t> q_range = {qn,qn+1};
    std::vector<size_t> r_range = {rn,rn+1};
    std::vector<size_t> s_range = {sn,sn+1};

    double vpqrsalphaC = 0.0;
    double vpqrsalphaE = 0.0;

    SharedMatrix B1(new Matrix(1,nthree_));
    SharedMatrix B2(new Matrix(1,nthree_));

    df_->fill_tensor("B",B1, A_range, p_range, r_range ); 
    df_->fill_tensor("B",B2, A_range, q_range, s_range ); 

    vpqrsalphaC = B1->vector_dot(B2);

    B1->zero();
    B2->zero();

    df_->fill_tensor("B",B1, A_range, p_range, s_range ); 
    df_->fill_tensor("B",B2, A_range, q_range, r_range ); 

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
double DISKDFIntegrals::three_integral(size_t A, size_t p, size_t q) {
    size_t pn, qn;
    if (frzcpi_.sum() > 0 && ncmo_ == aptei_idx_) {
        pn = cmotomo_[p];
        qn = cmotomo_[q];
    } else {
        pn = p;
        qn = q;
    }
    std::vector<size_t> A_range = {A,A+1};
    std::vector<size_t> p_range = {pn,pn+1};
    std::vector<size_t> q_range = {qn,qn+1};
    
    double* Apq;

    df_->fill_tensor("B",Apq, A_range, p_range, q_range ); 

    double value = *Apq;

    delete[] Apq;

    return value;
}
ambit::Tensor DISKDFIntegrals::three_integral_block(const std::vector<size_t>& A,
                                                    const std::vector<size_t>& p,
                                                    const std::vector<size_t>& q) {
    ambit::Tensor ReturnTensor =
        ambit::Tensor::build(tensor_type_, "Return", {A.size(), p.size(), q.size()});
    std::vector<double>& ReturnTensorV = ReturnTensor.data();

    bool frozen_core = false;

    // Take care of frozen orbitals
//    size_t p_max, p_min, q_max, q_min;
    if (frzcpi_.sum() && aptei_idx_ == ncmo_) {
//        p_min = cmotomo_[p[0]];
//        p_min = cmotomo_[p[0]];
//        p_max = cmotomo_[p.back()];
//        p_max = cmotomo_[p.back()];
//
//        q_min = cmotomo_[q[0]];
//        q_min = cmotomo_[q[0]];
//        q_max = cmotomo_[q.back()];
//        q_max = cmotomo_[q.back()];
//    } else {
//        p_min = p[0];
//        p_min = p[0];
//        p_max = p.back();
//        p_max = p.back();
//
//        q_min = q[0];
//        q_min = q[0];
//        q_max = q.back();
//        q_max = q.back();
        frozen_core = true;
    }

//
//    std::vector<size_t> A_range = {A[0], A.back() + 1};
//    std::vector<size_t> p_range = {p_min, p_max + 1};
//    std::vector<size_t> q_range = {q_min, q_max + 1};
//
//    SharedMatrix Bpq = std::make_shared<Matrix>(nthree_, p.size() * q.size());
//
//    df_->fill_tensor( "B", Bpq, A_range, p_range, q_range ); 
//
//    // Integrals stored as (A, p*q)
//    // Put them into ambit tensor
//    ReturnTensor.iterate([&](const std::vector<size_t>& i, double& value) {
//        value = Bpq->get(i[0], i[1]*p_max + i[2] );
//    });
//
    size_t pn, qn;
    if (nthree_ == A.size()) {
outfile->Printf("\n here");
        std::vector<std::shared_ptr<Matrix>> p_by_Aq;
        for (auto p_block : p) {
            if (frozen_core) {
                pn = cmotomo_[p_block];
            } else {
                pn = p_block;
            }

            std::shared_ptr<Matrix> Aq(new Matrix("Aq", nthree_, nmo_));

           // fseek(B_->file_pointer(), pn * nthree_ * nmo_ * sizeof(double), SEEK_SET);
           // fread(&(Aq->pointer()[0][0]), sizeof(double), nmo_ * nthree_, B_->file_pointer());
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
outfile->Printf("\n here done");
    } else {
outfile->Printf("\n here 2");
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
                std::vector<size_t> p_range = {pn, pn };
                std::vector<size_t> q_range = {qn, qn };

                double* A_chunk;// = new double[A.size()];
                //size_t offset = pn * nthree_ * nmo_ + qn * nthree_ + A[0];
                //fseek(B_->file_pointer(), offset * sizeof(double), SEEK_SET);
                //fread(&(A_chunk[0]), sizeof(double), A.size(), B_->file_pointer());
            
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
    outfile->Printf("\n If you are using this, you are ruining the advantages of DF/CD");
    throw PSIEXCEPTION("Don't use DF/CD if you use set_tei");
}

void DISKDFIntegrals::gather_integrals() {
    outfile->Printf("\n Computing Density fitted integrals \n");

    std::shared_ptr<BasisSet> primary = wfn_->basisset();
    if (options_.get_str("DF_BASIS_MP2").length() == 0) {
        outfile->Printf("\n Please set a DF_BASIS_MP2 option to a specified "
                        "auxiliary basis set");
        throw PSIEXCEPTION("Select a DF_BASIS_MP2 for use with DFIntegrals");
    }

    std::shared_ptr<BasisSet> auxiliary = wfn_->get_basisset("DF_BASIS_MP2");
    //            BasisSet::pyconstruct_orbital(primary->molecule(),
    //            "DF_BASIS_MP2",options_.get_str("DF_BASIS_MP2"));

    size_t nprim = primary->nbf();
    size_t naux = auxiliary->nbf();
    nthree_ = naux;
    outfile->Printf("\n Number of auxiliary basis functions:  %u", naux);
    outfile->Printf("\n Need %8.6f GB to store DF integrals\n",
                    (nprim * nprim * naux * sizeof(double) / 1073741824.0));
    int_mem_ = (nprim * nprim * naux * sizeof(double));

    Dimension nsopi_ = wfn_->nsopi();
    SharedMatrix aotoso = wfn_->aotoso();
    SharedMatrix Ca = wfn_->Ca();
    SharedMatrix Ca_ao(new Matrix("Ca_ao", nso_, nmopi_.sum()));

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
    df_ = std::make_shared<DF_Helper>(primary,auxiliary);
    df_->initialize();
    df_->set_MO_core(false);
    // set_C clears all the orbital spaces, so this creates the space
    // This space creates the total nmo_.
    // This assumes that everything is correlated.
    df_->add_space("ALL", Ca_ao);
    // Does not add the pair_space, but says which one is should use
    df_->add_transformation("B", "ALL", "ALL", "Qpq");
    df_->set_memory(Process::environment.get_memory() / 8L);

    // Finally computes the df integrals
    // Does the timings also
    Timer timer;
    std::string str = "Computing DF Integrals";
    outfile->Printf("\n    %-36s ...", str.c_str());
    df_->transform();
    outfile->Printf("...Done. Timing %15.6f s", timer.get());
}

void DISKDFIntegrals::make_diagonal_integrals() {
    /// TODO:  Need to rewrite this to avoid using aptei_aa.
    Timer MakeDiagonalIntegrals;
    // for(size_t p = 0; p < nmo_; ++p){
    //    for(size_t q = 0; q < nmo_; ++q){
    //        //diagonal_aphys_tei_aa[p * nmo_ + q] = aptei_aa(p,q,p,q);
    //        //diagonal_aphys_tei_ab[p * nmo_ + q] = aptei_ab(p,q,p,q);
    //        //diagonal_aphys_tei_bb[p * nmo_ + q] = aptei_bb(p,q,p,q);
    //        diagonal_aphys_tei_aa[p * nmo_ + q] = 0.0;
    //        diagonal_aphys_tei_ab[p * nmo_ + q] = 0.0;
    //        diagonal_aphys_tei_bb[p * nmo_ + q] = 0.0;

    //    }
    //}
    if (print_) {
        outfile->Printf("\n Make diagonal integrals in DISKDF took %6.6f s",
                        MakeDiagonalIntegrals.get());
    }
}

void DISKDFIntegrals::deallocate() {

    // Deallocate the memory required to store the one-electron integrals
    // Allocate the memory required to store the two-electron integrals

    delete[] diagonal_aphys_tei_aa;
    delete[] diagonal_aphys_tei_ab;
    delete[] diagonal_aphys_tei_bb;
}
void DISKDFIntegrals::make_fock_matrix(SharedMatrix gamma_aM, SharedMatrix gamma_bM) {
    // Efficient calculation of fock matrix from disk
    // Since gamma_aM is very sparse (diagonal elements of core and active
    // block)
    // Only nonzero contributions are on diagonal elements
    // Grab the nonzero elements and put to a vector

    TensorType tensor_type = ambit::CoreTensor;
    outfile->Printf("\n Making a fock matrix \n");

    // Create the fock_a and fock_b globally
    // Choose to block over naux rather than ncmo_
    ambit::Tensor fock_a = ambit::Tensor::build(tensor_type, "Fock_a", {aptei_idx_, aptei_idx_});
    ambit::Tensor fock_b = ambit::Tensor::build(tensor_type, "Fock_b", {aptei_idx_, aptei_idx_});

    ////Figure out exactly what I need to contract the Coloumb term
    /// Only h + a is nonzero for RDM
    std::vector<size_t> generalized_hole = mo_space_info_->get_corr_abs_mo("GENERALIZED HOLE");

    fock_a.iterate([&](const std::vector<size_t>& i, double& value) {
        value = one_electron_integrals_a[i[0] * aptei_idx_ + i[1]];
    });

    fock_b.iterate([&](const std::vector<size_t>& i, double& value) {
        value = one_electron_integrals_b[i[0] * aptei_idx_ + i[1]];
    });

    std::vector<size_t> A(nthree_);
    std::iota(A.begin(), A.end(), 0);

    std::vector<size_t> P(aptei_idx_);
    std::iota(P.begin(), P.end(), 0);

    // Create a gamma that contains only nonzero terms
    ambit::Tensor gamma_a = ambit::Tensor::build(
        tensor_type, "Gamma_a", {generalized_hole.size(), generalized_hole.size()});
    ambit::Tensor gamma_b = ambit::Tensor::build(
        tensor_type, "Gamma_b", {generalized_hole.size(), generalized_hole.size()});
    // Create the full gamma (K is not nearly as sparse as J)

    gamma_a.iterate([&](const std::vector<size_t>& i, double& value) {
        value = gamma_aM->get(generalized_hole[i[0]], generalized_hole[i[1]]);
    });
    gamma_b.iterate([&](const std::vector<size_t>& i, double& value) {
        value = gamma_bM->get(generalized_hole[i[0]], generalized_hole[i[1]]);
    });
    ambit::Tensor ThreeIntC2 = ambit::Tensor::build(
        tensor_type, "ThreeInkC", {nthree_, generalized_hole.size(), generalized_hole.size()});
    ThreeIntC2 = three_integral_block(A, generalized_hole, generalized_hole);

    ambit::Tensor BQA = ambit::Tensor::build(tensor_type, "BQ", {nthree_});
    ambit::Tensor BQB = ambit::Tensor::build(tensor_type, "BQ", {nthree_});
    // Do a contraction over Naux * n_h^2 * n_h^2 -> naux * n_h^2
    BQA("B") = ThreeIntC2("B,r,s") * gamma_a("r,s");
    BQB("B") = ThreeIntC2("B,r,s") * gamma_b("r,s");

    // Grab the data from this for the block iteration
    std::vector<double>& BQAv = BQA.data();
    std::vector<double>& BQBv = BQB.data();

    //====Blocking information==========
    size_t int_mem_int = (nthree_ * ncmo_ * ncmo_) * sizeof(double);
    size_t memory_input = Process::environment.get_memory() * 0.75;
    size_t num_block = int_mem_int / memory_input < 1 ? 1 : int_mem_int / memory_input;

    int block_size = nthree_ / num_block;
    if (block_size < 1) {
        outfile->Printf("\n\n Block size is FUBAR.");
        outfile->Printf("\n Block size is %d", block_size);
        throw PSIEXCEPTION("Block size is either 0 or negative.  Fix this problem");
    }
    if (num_block >= 1) {
        outfile->Printf("\n---------Blocking Information-------\n");
        outfile->Printf("\n  %lu / %lu = %lu", int_mem_int, memory_input,
                        int_mem_int / memory_input);
        outfile->Printf("\n  Block_size = %lu num_block = %lu", block_size, num_block);
    }
    ambit::Tensor ThreeIntegralTensorK;
    ambit::Tensor ThreeIntegralTensorJ;

    Timer block_read;
    for (size_t i = 0; i < num_block; i++) {
        std::vector<size_t> A_block;
        if (nthree_ % num_block == 0) {
            A_block.resize(block_size);
            std::iota(A_block.begin(), A_block.end(), i * block_size);
        } else {
            block_size = i == (num_block - 1) ? block_size + nthree_ % num_block : block_size;
            A_block.resize(block_size);
            std::iota(A_block.begin(), A_block.end(), i * (nthree_ / num_block));
        }

        // Create a tensor of TI("Q,r,p")
        ambit::Tensor BQA_small = ambit::Tensor::build(tensor_type, "BQ", {A_block.size()});
        ambit::Tensor BQB_small = ambit::Tensor::build(tensor_type, "BQ", {A_block.size()});

        // Calculate the smaller block of A from the global block of prior Brs *
        // gamma_rs
        BQA_small.iterate(
            [&](const std::vector<size_t>& i, double& value) { value = BQAv[A_block[i[0]]]; });
        BQB_small.iterate(
            [&](const std::vector<size_t>& i, double& value) { value = BQBv[A_block[i[0]]]; });

        ThreeIntegralTensorK = ambit::Tensor::build(
            tensor_type, "ThreeIndex", {A_block.size(), aptei_idx_, generalized_hole.size()});
        ThreeIntegralTensorJ = ambit::Tensor::build(tensor_type, "ThreeIndex",
                                                    {A_block.size(), aptei_idx_, aptei_idx_});

        ThreeIntegralTensorJ = three_integral_block(A_block, P, P);
        std::vector<double>& dataJ = ThreeIntegralTensorJ.data();
        ThreeIntegralTensorK.iterate([&](const std::vector<size_t>& i, double& value) {
            value =
                dataJ[i[0] * aptei_idx_ * aptei_idx_ + i[1] * aptei_idx_ + generalized_hole[i[2]]];
        });

        ambit::Tensor KGA = ambit::Tensor::build(
            tensor_type, "K * G", {A_block.size(), aptei_idx_, generalized_hole.size()});
        ambit::Tensor KGB = ambit::Tensor::build(
            tensor_type, "K * G b", {A_block.size(), aptei_idx_, generalized_hole.size()});

        KGA("Q,q,r") = ThreeIntegralTensorK("Q, q, s") * gamma_a("s, r");
        KGB("Q,q,r") = ThreeIntegralTensorK("Q, q, s") * gamma_b("s, r");

        // Need to rewrite this to at least read in chunks of nthree_
        // ThreeIntegralTensor = three_integral_block(A_block, P,P );

        fock_a("p,q") += ThreeIntegralTensorJ("Q,p,q") * BQA_small("Q");
        fock_a("p,q") -= ThreeIntegralTensorK("Q,p,r") * KGA("Q, q, r ");
        fock_a("p,q") += ThreeIntegralTensorJ("Q,p,q") * BQB_small("Q");

        fock_b("p,q") += ThreeIntegralTensorJ("Q,p,q") * BQB_small("Q");
        fock_b("p,q") -= ThreeIntegralTensorK("Q,p,r") * KGB("Q, q, r");
        fock_b("p,q") += ThreeIntegralTensorJ("Q,p,q") * BQA_small("Q");

        A_block.clear();
    }
    fock_a.iterate([&](const std::vector<size_t>& i, double& value) {
        fock_matrix_a[i[0] * aptei_idx_ + i[1]] = value;
    });
    fock_b.iterate([&](const std::vector<size_t>& i, double& value) {
        fock_matrix_b[i[0] * aptei_idx_ + i[1]] = value;
    });

    if (num_block != 1) {
        outfile->Printf("\n Created Fock matrix %8.8f s", block_read.get());
    }
}

void DISKDFIntegrals::resort_three(SharedMatrix&, std::vector<size_t>&) {
    outfile->Printf("No need to resort a file.  dummy!");
}

void DISKDFIntegrals::resort_integrals_after_freezing() {
    Timer resort_integrals;
    outfile->Printf("\n  Resorting integrals after freezing core.");

    // Create an array that maps the CMOs to the MOs (cmo2mo).

    // Resort the integrals
    resort_two(one_electron_integrals_a, cmotomo_);
    resort_two(one_electron_integrals_b, cmotomo_);
    // resort_two(diagonal_aphys_tei_aa,cmotomo_);
    // resort_two(diagonal_aphys_tei_ab,cmotomo_);
    // resort_two(diagonal_aphys_tei_bb,cmotomo_);

    // resort_three(ThreeIntegral_,cmo2mo);

    outfile->Printf("\n Resorting integrals takes   %8.8fs", resort_integrals.get());
}
ambit::Tensor DISKDFIntegrals::three_integral_block_two_index(const std::vector<size_t>& A,
                                                              size_t p,
                                                              const std::vector<size_t>& q) {

    ambit::Tensor ReturnTensor = ambit::Tensor::build(tensor_type_, "Return", {A.size(), q.size()});
 
outfile->Printf("\n here tidx");
    size_t p_max, p_min;
    bool frozen_core = false;
    if (frzcpi_.sum() && aptei_idx_ == ncmo_) {
        frozen_core = true; 
        p_min = cmotomo_[p];
        p_max = p_min +1;
    } else {
        p_min = p;
        p_max = p_min +1;
    }

    if (nthree_ == A.size()) {

        std::vector<size_t> arange = {0, nthree_};
        std::vector<size_t> qrange = {0, nmo_}; 
        std::vector<size_t> prange = {p_min, p_max}; 

        std::shared_ptr<Matrix> Aq(new Matrix("Aq", nthree_, nmo_));
        df_->fill_tensor("B", Aq, arange, prange, qrange ); 

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
        throw PSIEXCEPTION("Can only use if 2nd parameter is a size_t and A.size==nthree_");
    }
outfile->Printf("\n here tidx done");
    return ReturnTensor;
}

double DISKDFIntegrals::diag_aptei_aa(size_t, size_t) {
    outfile->Printf("\n Kevin seemed to find that nobody uses this function.  "
                    "It is quite slow in DISKDF");
    outfile->Printf("\n Bribe Kevin with things and he will implement it if it is needed");
    throw PSIEXCEPTION("diag_aptei_aa is not implemented for DISKDF");
}

double DISKDFIntegrals::diag_aptei_ab(size_t, size_t) {
    outfile->Printf("\n Kevin seemed to find that nobody uses this function.  "
                    "It is quite slow in DISKDF");
    outfile->Printf("\n Bribe Kevin with things and he will implement it if it is needed");
    throw PSIEXCEPTION("diag_aptei_ab is not implemented for DISKDF");
}
double DISKDFIntegrals::diag_aptei_bb(size_t, size_t) {
    outfile->Printf("\n Kevin seemed to find that nobody uses this function.  "
                    "It is quite slow in DISKDF");
    outfile->Printf("\n Bribe Kevin with things and he will implement it if it is needed");
    throw PSIEXCEPTION("diag_aptei_bb is not implemented for DISKDF");
}
}
}
