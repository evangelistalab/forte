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

#include "forte-def.h"
#include "helpers/blockedtensorfactory.h"
#include "helpers/helpers.h"
#include "helpers/printing.h"
#include "helpers/timer.h"
#include "helpers/memory.h"

#include "df_integrals.h"

using namespace ambit;
using namespace psi;

namespace forte {

DFIntegrals::DFIntegrals(std::shared_ptr<ForteOptions> options,
                         std::shared_ptr<psi::Wavefunction> ref_wfn,
                         std::shared_ptr<MOSpaceInfo> mo_space_info,
                         IntegralSpinRestriction restricted)
    : Psi4Integrals(options, ref_wfn, mo_space_info, DF, restricted) {
    initialize();
}

void DFIntegrals::initialize() {
    // If code calls constructor print things
    // But if someone calls retransform integrals do not print it
    print_info();

    int my_proc = 0;
#ifdef HAVE_GA
    my_proc = GA_Nodeid();
#endif
    if (my_proc == 0 and (not skip_build_)) {
        local_timer int_timer;
        gather_integrals();
        freeze_core_orbitals();
        print_timing("computing density-fitted integrals", int_timer.get());
    }
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
                                                const std::vector<size_t>& q,
                                                ThreeIntsBlockOrder order) {
    ambit::Tensor ReturnTensor;
    if (order == pqQ) {
        ReturnTensor = ambit::Tensor::build(tensor_type_, "Return", {p.size(), q.size(), A.size()});
        ReturnTensor.iterate([&](const std::vector<size_t>& i, double& value) {
            value = three_integral(A[i[2]], p[i[0]], q[i[1]]);
        });
    } else {
        ReturnTensor = ambit::Tensor::build(tensor_type_, "Return", {A.size(), p.size(), q.size()});
        ReturnTensor.iterate([&](const std::vector<size_t>& i, double& value) {
            value = three_integral(A[i[0]], p[i[1]], q[i[2]]);
        });
    }
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
        outfile->Printf("\n  Computing density fitted integrals\n");
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

    // B_{pq}^Q -> MO without frozen core

    // Constructs the DF function
    // assume a RHF/UHF reference
    auto df = std::make_shared<psi::DFHelper>(primary, auxiliary);
    size_t mem_sys = psi::Process::environment.get_memory() * 0.9 / sizeof(double);
    int64_t mem = mem_sys;
    if (JK_status_ == JKStatus::initialized) {
        mem = mem_sys - JK_->memory_estimate();
        if (mem < 0) {
            auto xb = to_xb(static_cast<size_t>(-mem), sizeof(double));
            std::string msg = "Not enough memory! Need at least ";
            msg += std::to_string(xb.first) + " " + xb.second.c_str() + " more.";
            outfile->Printf("\n  %s", msg.c_str());
            throw psi::PSIEXCEPTION(msg);
        }
    }
    df->set_schwarz_cutoff(schwarz_cutoff_);
    df->set_fitting_condition(df_fitting_cutoff_);
    df->set_memory(static_cast<size_t>(mem));
    df->set_nthreads(omp_get_max_threads());
    df->set_print_lvl(1);
    df->initialize();
    df->print_header();
    // Pushes a C matrix that is ordered in pitzer ordering
    // into the C_matrix object
    df->add_space("ALL", Ca_AO());

    // set_C clears all the orbital spaces, so this creates the space
    // This space creates the total nmo_.
    // This assumes that everything is correlated.
    // Does not add the pair_space, but says which one is should use
    df->add_transformation("B", "ALL", "ALL", "Qpq");

    // Finally computes the df integrals
    // Does the timings also
    local_timer timer;
    if (print_ > 0) {
        outfile->Printf("\n  Transforming DF Integrals");
    }
    df->transform();
    if (print_ > 0) {
        print_timing("density-fitting transformation", timer.get());
        outfile->Printf("\n");
    }

    auto Bpq = std::make_shared<psi::Matrix>("Bpq", naux, nmo_ * nmo_);

    Bpq = df->get_tensor("B");

    // Store as transpose for now
    ThreeIntegral_ = Bpq->transpose()->clone();
}

void DFIntegrals::resort_three(std::shared_ptr<psi::Matrix>& threeint, std::vector<size_t>& map) {
    // Create a temperature threeint matrix
    auto temp_threeint = std::make_shared<psi::Matrix>("tmp", ncmo_ * ncmo_, nthree_);
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
