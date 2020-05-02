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

#include <cmath>
#include <numeric>

#include "psi4/libpsi4util/process.h"
#include "psi4/libmints/basisset.h"
#include "psi4/libmints/matrix.h"
#include "psi4/libmints/wavefunction.h"
#include "psi4/libqt/qt.h"
#include "psi4/lib3index/dfhelper.h"

#ifdef HAVE_GA
#include <ga.h>
#include <macdecls.h>
#endif
#ifdef HAVE_MPI
#include <mpi.h>
#endif

#include "helpers/blockedtensorfactory.h"
#include "helpers/printing.h"
#include "helpers/timer.h"
#include "helpers/memory.h"

#include "df_integrals.h"

using namespace ambit;
using namespace psi;

namespace forte {

DFIntegrals::DFIntegrals(std::shared_ptr<ForteOptions> options, std::shared_ptr<psi::Wavefunction> ref_wfn,
                         std::shared_ptr<MOSpaceInfo> mo_space_info,
                         IntegralSpinRestriction restricted)
    : ForteIntegrals(options, ref_wfn, mo_space_info, restricted) {
    integral_type_ = DF;
    // If code calls constructor print things
    // But if someone calls retransform integrals do not print it
    print_info();
    local_timer int_timer;

    int my_proc = 0;
#ifdef HAVE_GA
    my_proc = GA_Nodeid();
#endif
    if (my_proc == 0) {
        gather_integrals();
        freeze_core_orbitals();
    }
    print_timing("computing density-fitted integrals", int_timer.get());
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

double DFIntegrals::three_integral(size_t A, size_t p, size_t q) {
    return ThreeIntegral_->get(p * aptei_idx_ + q, A);
}

double** DFIntegrals::three_integral_pointer() { return ThreeIntegral_->pointer(); }

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

ambit::Tensor DFIntegrals::three_integral_block_two_index(const std::vector<size_t>&, size_t,
                                                          const std::vector<size_t>&) {
    outfile->Printf("\n Oh no! this isn't here");
    throw psi::PSIEXCEPTION("INT_TYPE=DISKDF");
}

void DFIntegrals::set_tei(size_t, size_t, size_t, size_t, double, bool, bool) {
    outfile->Printf("\n If you are using this, you are ruining the advantages of DF/CD");
    throw psi::PSIEXCEPTION("Don't use DF/CD if you use set_tei");
}

void DFIntegrals::gather_integrals() {

    if (print_ > 0) {
        outfile->Printf("\n  Computing Density fitted integrals \n");
    }

    std::shared_ptr<psi::BasisSet> primary = wfn_->basisset();
    std::shared_ptr<psi::BasisSet> auxiliary = wfn_->get_basisset("DF_BASIS_MP2");

    size_t nprim = primary->nbf();
    size_t naux = auxiliary->nbf();
    nthree_ = naux;
    if (print_ > 0) {
        outfile->Printf("\n  Number of auxiliary basis functions:  %u", naux);
        auto mem_info = to_xb2<double>(nprim * nprim * naux);
        outfile->Printf("\n  Need %.2f %s to store DF integrals\n", mem_info.first,
                        mem_info.second.c_str());
    }

    psi::Dimension nsopi_ = wfn_->nsopi();
    std::shared_ptr<psi::Matrix> aotoso = wfn_->aotoso();
    std::shared_ptr<psi::Matrix> Ca = wfn_->Ca();
    // std::shared_ptr<psi::Matrix> Ca_ao(new psi::Matrix("Ca_ao",nso_,nmopi_.sum()));
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
    // assume a RHF/UHF reference
    std::shared_ptr<psi::DFHelper> df(new DFHelper(primary, auxiliary));
    df->initialize();
    // Pushes a C matrix that is ordered in pitzer ordering
    // into the C_matrix object

    df->add_space("ALL", Ca_ao);

    // set_C clears all the orbital spaces, so this creates the space
    // This space creates the total nmo_.
    // This assumes that everything is correlated.
    // Does not add the pair_space, but says which one is should use
    df->add_transformation("B", "ALL", "ALL", "Qpq");
    df->set_memory(psi::Process::environment.get_memory() / 8L);

    // Finally computes the df integrals
    // Does the timings also
    local_timer timer;
    std::string str = "Transforming DF Integrals";
    if (print_ > 0) {
        outfile->Printf("\n  %-36s ...", str.c_str());
    }
    df->transform();
    if (print_ > 0) {
        outfile->Printf("...Done.");
        print_timing("density-fitting transformation", timer.get());
        outfile->Printf("\n");
    }

    std::shared_ptr<psi::Matrix> Bpq(new psi::Matrix("Bpq", naux, nmo_ * nmo_));

    Bpq = df->get_tensor("B");

    // Store as transpose for now
    ThreeIntegral_ = Bpq->transpose()->clone();
}

void DFIntegrals::make_fock_matrix(std::shared_ptr<psi::Matrix> gamma_aM,
                                   std::shared_ptr<psi::Matrix> gamma_bM) {
    TensorType tensor_type = ambit::CoreTensor;
    ambit::Tensor ThreeIntegralTensor =
        // ambit::Tensor::build(tensor_type, "ThreeIndex", {ncmo_, ncmo_, nthree_});
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
        value = one_electron_integrals_a_[i[0] * aptei_idx_ + i[1]];
    });

    fock_b.iterate([&](const std::vector<size_t>& i, double& value) {
        value = one_electron_integrals_b_[i[0] * aptei_idx_ + i[1]];
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
        fock_matrix_a_[i[0] * aptei_idx_ + i[1]] = value;
    });
    fock_b.iterate([&](const std::vector<size_t>& i, double& value) {
        fock_matrix_b_[i[0] * aptei_idx_ + i[1]] = value;
    });

    /// Form with JK builders
}

void DFIntegrals::resort_three(std::shared_ptr<psi::Matrix>& threeint, std::vector<size_t>& map) {
    // Create a temperature threeint matrix
    std::shared_ptr<psi::Matrix> temp_threeint(new psi::Matrix("tmp", ncmo_ * ncmo_, nthree_));
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
    threeint->copy(temp_threeint);
}

void DFIntegrals::resort_integrals_after_freezing() {
    local_timer timer_resort;
    if (print_ > 0) {
        outfile->Printf("\n  Resorting integrals after freezing core.");
    }

    resort_three(ThreeIntegral_, cmotomo_);

    if (print_ > 0) {
        print_timing("resorting DF integrals", timer_resort.get());
    }
}

size_t DFIntegrals::nthree() const { return nthree_; }

} // namespace forte
