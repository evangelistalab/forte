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
#include "psi4/libmints/basisset.h"
#include "psi4/libmints/wavefunction.h"
#include "psi4/libqt/qt.h"
#include "psi4/libthce/lreri.h"
#include "psi4/libthce/thce.h"

#ifdef HAVE_GA
#include <ga.h>
#include <macdecls.h>
#endif
#ifdef HAVE_MPI
#include <mpi.h>
#endif

#include "../blockedtensorfactory.h"

using namespace ambit;
namespace psi {
namespace forte {
// Class for the DF Integrals
// Generates DF Integrals.  Freezes Core orbitals, computes integrals, and
// resorts integrals.  Also computes fock matrix
DFIntegrals::DFIntegrals(psi::Options& options, SharedWavefunction ref_wfn,
                         IntegralSpinRestriction restricted, IntegralFrozenCore resort_frozen_core,
                         std::shared_ptr<MOSpaceInfo> mo_space_info)
    : ForteIntegrals(options, ref_wfn, restricted, resort_frozen_core, mo_space_info) {
    integral_type_ = DF;
    // If code calls constructor print things
    // But if someone calls retransform integrals do not print it

    wfn_ = ref_wfn;

    outfile->Printf("\n  DFIntegrals overall time");
    Timer DFInt;
    allocate();
    int my_proc = 0;
#ifdef HAVE_GA
    my_proc = GA_Nodeid();
#endif
    if (my_proc == 0) {
        gather_integrals();
        make_diagonal_integrals();
        if (ncmo_ < nmo_) {
            freeze_core_orbitals();
            // Set the new value of the number of orbitals to be used in
            // indexing routines
            aptei_idx_ = ncmo_;
        }
    }
    outfile->Printf("\n  DFIntegrals take %15.8f s", DFInt.get());
}

DFIntegrals::~DFIntegrals() { deallocate(); }

void DFIntegrals::allocate() {
    // Allocate the memory required to store the one-electron integrals
    // Allocate the memory required to store the two-electron integrals
    diagonal_aphys_tei_aa = new double[nmo_ * nmo_];
    diagonal_aphys_tei_ab = new double[nmo_ * nmo_];
    diagonal_aphys_tei_bb = new double[nmo_ * nmo_];

    // qt_pitzer_ = new int[nmo_];
}

double DFIntegrals::aptei_aa(size_t p, size_t q, size_t r, size_t s) {
    double vpqrsalphaC = 0.0;
    double vpqrsalphaE = 0.0;
    vpqrsalphaC = C_DDOT(nthree_, &(ThreeIntegral_->pointer()[p * aptei_idx_ + r][0]), 1,
                         &(ThreeIntegral_->pointer()[q * aptei_idx_ + s][0]), 1);
    vpqrsalphaE = C_DDOT(nthree_, &(ThreeIntegral_->pointer()[p * aptei_idx_ + s][0]), 1,
                         &(ThreeIntegral_->pointer()[q * aptei_idx_ + r][0]), 1);

    return (vpqrsalphaC - vpqrsalphaE);
}

double DFIntegrals::aptei_ab(size_t p, size_t q, size_t r, size_t s) {
    double vpqrsalphaC = 0.0;
    vpqrsalphaC = C_DDOT(nthree_, &(ThreeIntegral_->pointer()[p * aptei_idx_ + r][0]), 1,
                         &(ThreeIntegral_->pointer()[q * aptei_idx_ + s][0]), 1);

    return (vpqrsalphaC);
}

double DFIntegrals::aptei_bb(size_t p, size_t q, size_t r, size_t s) {
    double vpqrsalphaC = 0.0;
    double vpqrsalphaE = 0.0;
    vpqrsalphaC = C_DDOT(nthree_, &(ThreeIntegral_->pointer()[p * aptei_idx_ + r][0]), 1,
                         &(ThreeIntegral_->pointer()[q * aptei_idx_ + s][0]), 1);
    vpqrsalphaE = C_DDOT(nthree_, &(ThreeIntegral_->pointer()[p * aptei_idx_ + s][0]), 1,
                         &(ThreeIntegral_->pointer()[q * aptei_idx_ + r][0]), 1);

    return (vpqrsalphaC - vpqrsalphaE);
}

ambit::Tensor DFIntegrals::aptei_aa_block(const std::vector<size_t>& p,
                                          const std::vector<size_t>& q,
                                          const std::vector<size_t>& r,
                                          const std::vector<size_t>& s) {
    ambit::Tensor ReturnTensor =
        ambit::Tensor::build(tensor_type_, "Return", {p.size(), q.size(), r.size(), s.size()});
    ReturnTensor.iterate([&](const std::vector<size_t>& i, double& value) {
        value = aptei_aa(p[i[0]], q[i[1]], r[i[2]], s[i[3]]);
    });
    return ReturnTensor;
}

ambit::Tensor DFIntegrals::aptei_ab_block(const std::vector<size_t>& p,
                                          const std::vector<size_t>& q,
                                          const std::vector<size_t>& r,
                                          const std::vector<size_t>& s) {
    ambit::Tensor ReturnTensor =
        ambit::Tensor::build(tensor_type_, "Return", {p.size(), q.size(), r.size(), s.size()});
    ReturnTensor.iterate([&](const std::vector<size_t>& i, double& value) {
        value = aptei_ab(p[i[0]], q[i[1]], r[i[2]], s[i[3]]);
    });
    return ReturnTensor;
}

ambit::Tensor DFIntegrals::aptei_bb_block(const std::vector<size_t>& p,
                                          const std::vector<size_t>& q,
                                          const std::vector<size_t>& r,
                                          const std::vector<size_t>& s) {
    ambit::Tensor ReturnTensor =
        ambit::Tensor::build(tensor_type_, "Return", {p.size(), q.size(), r.size(), s.size()});
    ReturnTensor.iterate([&](const std::vector<size_t>& i, double& value) {
        value = aptei_bb(p[i[0]], q[i[1]], r[i[2]], s[i[3]]);
    });
    return ReturnTensor;
}
ambit::Tensor DFIntegrals::three_integral_block(const std::vector<size_t>& A,
                                                const std::vector<size_t>& p,
                                                const std::vector<size_t>& q) {
    ambit::Tensor ReturnTensor =
        ambit::Tensor::build(tensor_type_, "Return", {A.size(), p.size(), q.size()});
    ReturnTensor.iterate([&](const std::vector<size_t>& i, double& value) {
        value = three_integral(A[i[0]], p[i[1]], q[i[2]]);
    });
    return ReturnTensor;
}

void DFIntegrals::set_tei(size_t, size_t, size_t, size_t, double, bool, bool) {
    outfile->Printf("\n If you are using this, you are ruining the advantages of DF/CD");
    throw PSIEXCEPTION("Don't use DF/CD if you use set_tei");
}

void DFIntegrals::gather_integrals() {

    if (print_ > 0) {
        outfile->Printf("\n Computing Density fitted integrals \n");
    }
    if (options_.get_str("DF_BASIS_MP2").length() == 0) {
        outfile->Printf("\n Please set a DF_BASIS_MP2 option to a specified "
                        "auxiliary basis set");
#ifdef HAVE_MPI
        MPI_Abort(MPI_COMM_WORLD, 0);
#endif
        throw PSIEXCEPTION("Select a DF_BASIS_MP2 for use with DFIntegrals");
    }

    std::shared_ptr<BasisSet> primary = wfn_->basisset();
    std::shared_ptr<BasisSet> auxiliary = wfn_->get_basisset("DF_BASIS_MP2");

    size_t nprim = primary->nbf();
    size_t naux = auxiliary->nbf();
    nthree_ = naux;
    if (print_ > 0) {
        outfile->Printf("\n Number of auxiliary basis functions:  %u", naux);
        outfile->Printf("\n Need %8.6f GB to store DF integrals\n",
                        (nprim * nprim * naux * sizeof(double) / 1073741824.0));
    }

    Dimension nsopi_ = wfn_->nsopi();
    SharedMatrix aotoso = wfn_->aotoso();
    SharedMatrix Ca = wfn_->Ca();
    // SharedMatrix Ca_ao(new Matrix("Ca_ao",nso_,nmopi_.sum()));
    SharedMatrix Ca_ao(new Matrix("Ca_ao", nso_, nmopi_.sum()));

    // Transform from the SO to the AO basis
    for (size_t h = 0, index = 0; h < nirrep_; ++h) {
        for (size_t i = 0; i < nmopi_[h]; ++i) {
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
    //    df->set_C(C_ord);
    df->set_C(Ca_ao);
    //    Ca_ = Ca_ao;
    // set_C clears all the orbital spaces, so this creates the space
    // This space creates the total nmo_.
    // This assumes that everything is correlated.
    df->add_space("ALL", 0, nmo_);
    // Does not add the pair_space, but says which one is should use
    df->add_pair_space("B", "ALL", "ALL");
    df->set_memory(Process::environment.get_memory() / 8L);

    // Finally computes the df integrals
    // Does the timings also
    Timer timer;
    std::string str = "Computing DF Integrals";
    if (print_ > 0) {
        outfile->Printf("\n  %-36s ...", str.c_str());
    }
    df->compute();
    if (print_ > 0) {
        outfile->Printf("...Done. Timing %15.6f s", timer.get());
    }

    std::shared_ptr<psi::Tensor> B = df->ints()["B"];
    df.reset();

    FILE* Bf = B->file_pointer();
    // SharedMatrix Bpq(new Matrix("Bpq", nmo_, nmo_ * naux));
    SharedMatrix Bpq(new Matrix("Bpq", nmo_ * nmo_, naux));

    // Reads the DF integrals into Bpq.  Stores them as nmo by (nmo*naux)

    std::string str_seek = "Seeking DF Integrals";
    if (print_ > 0) {
        outfile->Printf("\n  %-36s ...", str_seek.c_str());
    }
    fseek(Bf, 0L, SEEK_SET);
    if (print_ > 0) {
        outfile->Printf("...Done. Timing %15.6f s", timer.get());
    }

    std::string str_read = "Reading DF Integrals";
    if (print_ > 0) {
        outfile->Printf("\n  %-36s ...", str_read.c_str());
    }
    fread(&(Bpq->pointer()[0][0]), sizeof(double), naux * (nmo_) * (nmo_), Bf);
    if (print_ > 0) {
        outfile->Printf("...Done. Timing %15.6f s", timer.get());
    }

    ThreeIntegral_ = Bpq;
}

void DFIntegrals::make_diagonal_integrals() {
    for (size_t p = 0; p < nmo_; ++p) {
        for (size_t q = 0; q < nmo_; ++q) {
            diagonal_aphys_tei_aa[p * nmo_ + q] = aptei_aa(p, q, p, q);
            diagonal_aphys_tei_ab[p * nmo_ + q] = aptei_ab(p, q, p, q);
            diagonal_aphys_tei_bb[p * nmo_ + q] = aptei_bb(p, q, p, q);
        }
    }
}

void DFIntegrals::deallocate() {

    // Deallocate the memory required to store the one-electron integrals
    // Allocate the memory required to store the two-electron integrals

    delete[] diagonal_aphys_tei_aa;
    delete[] diagonal_aphys_tei_ab;
    delete[] diagonal_aphys_tei_bb;
    // delete[] qt_pitzer_;
}

void DFIntegrals::make_fock_matrix(SharedMatrix gamma_aM, SharedMatrix gamma_bM) {
    TensorType tensor_type = ambit::CoreTensor;
    ambit::Tensor ThreeIntegralTensor =
        ambit::Tensor::build(tensor_type, "ThreeIndex", {nthree_, ncmo_, ncmo_});
    ambit::Tensor gamma_a = ambit::Tensor::build(tensor_type, "Gamma_a", {ncmo_, ncmo_});
    ambit::Tensor gamma_b = ambit::Tensor::build(tensor_type, "Gamma_b", {ncmo_, ncmo_});
    ambit::Tensor fock_a = ambit::Tensor::build(tensor_type, "Fock_a", {ncmo_, ncmo_});
    ambit::Tensor fock_b = ambit::Tensor::build(tensor_type, "Fock_b", {ncmo_, ncmo_});

    // ThreeIntegralTensor.iterate([&](const std::vector<size_t>& i,double&
    // value){
    //    value = ThreeIntegral_->get(i[0],i[1]*aptei_idx_ + i[2]);
    //});
    std::vector<size_t> vQ(nthree_);
    std::iota(vQ.begin(), vQ.end(), 0);
    std::vector<size_t> vP(ncmo_);
    std::iota(vP.begin(), vP.end(), 0);

    ThreeIntegralTensor = three_integral_block(vQ, vP, vP);

    gamma_a.iterate(
        [&](const std::vector<size_t>& i, double& value) { value = gamma_aM->get(i[0], i[1]); });
    gamma_b.iterate(
        [&](const std::vector<size_t>& i, double& value) { value = gamma_bM->get(i[0], i[1]); });

    fock_a.iterate([&](const std::vector<size_t>& i, double& value) {
        value = one_electron_integrals_a[i[0] * aptei_idx_ + i[1]];
    });

    fock_b.iterate([&](const std::vector<size_t>& i, double& value) {
        value = one_electron_integrals_b[i[0] * aptei_idx_ + i[1]];
    });

    /// Changing the Q_pr * Q_qs  to Q_rp * Q_sq for convience for reading

    // ambit::Tensor test = ambit::Tensor::build(tensor_type,
    // "Fock_b",{nthree_});
    // test("Q") = ThreeIntegralTensor("Q,r,s") * gamma_a("r,s");
    // fock_a("p,q") += ThreeIntegralTensor("Q,p,q")*test("Q");
    fock_a("p,q") += ThreeIntegralTensor("Q,p,q") * ThreeIntegralTensor("Q,r,s") * gamma_a("r,s");
    fock_a("p,q") -= ThreeIntegralTensor("Q,r,p") * ThreeIntegralTensor("Q,s,q") * gamma_a("r,s");
    fock_a("p,q") += ThreeIntegralTensor("Q,p,q") * ThreeIntegralTensor("Q,r,s") * gamma_b("r,s");

    fock_b("p,q") += ThreeIntegralTensor("Q,p,q") * ThreeIntegralTensor("Q,r,s") * gamma_b("r,s");
    fock_b("p,q") -= ThreeIntegralTensor("Q,r,p") * ThreeIntegralTensor("Q,s,q") * gamma_b("r,s");
    fock_b("p,q") += ThreeIntegralTensor("Q,p,q") * ThreeIntegralTensor("Q,r,s") * gamma_a("r,s");

    fock_a.iterate([&](const std::vector<size_t>& i, double& value) {
        fock_matrix_a[i[0] * aptei_idx_ + i[1]] = value;
    });
    fock_b.iterate([&](const std::vector<size_t>& i, double& value) {
        fock_matrix_b[i[0] * aptei_idx_ + i[1]] = value;
    });

    /// Form with JK builders
}

void DFIntegrals::resort_three(SharedMatrix& threeint, std::vector<size_t>& map) {
    // Create a temperature threeint matrix
    SharedMatrix temp_threeint(new Matrix("tmp", ncmo_ * ncmo_, nthree_));
    temp_threeint->zero();

    // Borrwed from resort_four.
    // Since L is not sorted, only need to sort the columns
    // Surprisingly, this was pretty easy.
    for (size_t L = 0; L < nthree_; ++L) {
        for (size_t q = 0; q < ncmo_; ++q) {
            for (size_t r = 0; r < ncmo_; ++r) {
                size_t Lpq_cmo = q * ncmo_ + r;
                size_t Lpq_mo = map[q] * nmo_ + map[r];
                temp_threeint->set(Lpq_cmo, L, threeint->get(Lpq_mo, L));
            }
        }
    }

    // This copies the resorted integrals and the data is changed to the sorted
    // matrix
    if (print_ > 0) {
        outfile->Printf("\n Done with resorting");
    }
    threeint->copy(temp_threeint);
}

void DFIntegrals::resort_integrals_after_freezing() {
    Timer resort_integrals;
    if (print_ > 0) {
        outfile->Printf("\n  Resorting integrals after freezing core.");
    }

    // Create an array that maps the CMOs to the MOs (cmo2mo).
    std::vector<size_t> cmo2mo;
    for (int h = 0, q = 0; h < nirrep_; ++h) {
        q += frzcpi_[h]; // skip the frozen core
        for (int r = 0; r < ncmopi_[h]; ++r) {
            cmo2mo.push_back(q);
            q++;
        }
        q += frzvpi_[h]; // skip the frozen virtual
    }
    cmotomo_ = (cmo2mo);

    // Resort the integrals
    resort_two(one_electron_integrals_a, cmo2mo);
    resort_two(one_electron_integrals_b, cmo2mo);
    resort_two(diagonal_aphys_tei_aa, cmo2mo);
    resort_two(diagonal_aphys_tei_ab, cmo2mo);
    resort_two(diagonal_aphys_tei_bb, cmo2mo);

    resort_three(ThreeIntegral_, cmo2mo);

    if (print_ > 0) {
        outfile->Printf("\n Resorting integrals takes   %8.8fs", resort_integrals.get());
    }
}
}
}
