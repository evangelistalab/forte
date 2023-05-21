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
#include <cstring>

#include "psi4/libpsi4util/process.h"
#include "psi4/libmints/basisset.h"
#include "psi4/libmints/wavefunction.h"
#include "psi4/libmints/integral.h"
#include "psi4/libmints/matrix.h"
#include "psi4/lib3index/cholesky.h"
#include "psi4/libpsio/psio.hpp"
#include "psi4/libqt/qt.h"
#include "psi4/psifiles.h"

#include "helpers/timer.h"
#include "helpers/printing.h"
#include "helpers/memory.h"

#include "base_classes/forte_options.h"

#include "cholesky_integrals.h"

using namespace ambit;
using namespace psi;

namespace forte {

CholeskyIntegrals::CholeskyIntegrals(std::shared_ptr<ForteOptions> options,
                                     std::shared_ptr<psi::Wavefunction> ref_wfn,
                                     std::shared_ptr<MOSpaceInfo> mo_space_info,
                                     IntegralSpinRestriction restricted)
    : Psi4Integrals(options, ref_wfn, mo_space_info, Cholesky, restricted) {
    initialize();
}

void CholeskyIntegrals::initialize() {
    print_info();

    if (not skip_build_) {
        local_timer int_timer;
        gather_integrals();
        freeze_core_orbitals();
        print_timing("computing Cholesky integrals", int_timer.get());
    }
}

double CholeskyIntegrals::aptei_aa(size_t p, size_t q, size_t r, size_t s) {
    double vpqrsalphaC = 0.0;
    double vpqrsalphaE = 0.0;
    vpqrsalphaC = C_DDOT(nthree_, &(ThreeIntegral_->pointer()[p * aptei_idx_ + r][0]), 1,
                         &(ThreeIntegral_->pointer()[q * aptei_idx_ + s][0]), 1);
    vpqrsalphaE = C_DDOT(nthree_, &(ThreeIntegral_->pointer()[p * aptei_idx_ + s][0]), 1,
                         &(ThreeIntegral_->pointer()[q * aptei_idx_ + r][0]), 1);

    return (vpqrsalphaC - vpqrsalphaE);
}

double CholeskyIntegrals::aptei_ab(size_t p, size_t q, size_t r, size_t s) {
    double vpqrsalphaC = 0.0;
    vpqrsalphaC = C_DDOT(nthree_, &(ThreeIntegral_->pointer()[p * aptei_idx_ + r][0]), 1,
                         &(ThreeIntegral_->pointer()[q * aptei_idx_ + s][0]), 1);
    return (vpqrsalphaC);
}

double CholeskyIntegrals::aptei_bb(size_t p, size_t q, size_t r, size_t s) {
    double vpqrsalphaC = 0.0, vpqrsalphaE = 0.0;
    vpqrsalphaC = C_DDOT(nthree_, &(ThreeIntegral_->pointer()[p * aptei_idx_ + r][0]), 1,
                         &(ThreeIntegral_->pointer()[q * aptei_idx_ + s][0]), 1);
    vpqrsalphaE = C_DDOT(nthree_, &(ThreeIntegral_->pointer()[p * aptei_idx_ + s][0]), 1,
                         &(ThreeIntegral_->pointer()[q * aptei_idx_ + r][0]), 1);

    return (vpqrsalphaC - vpqrsalphaE);
}
ambit::Tensor CholeskyIntegrals::aptei_aa_block(const std::vector<size_t>& p,
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

ambit::Tensor CholeskyIntegrals::aptei_ab_block(const std::vector<size_t>& p,
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

ambit::Tensor CholeskyIntegrals::aptei_bb_block(const std::vector<size_t>& p,
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

double CholeskyIntegrals::three_integral(size_t A, size_t p, size_t q) const {
    return ThreeIntegral_->get(p * aptei_idx_ + q, A);
}

double** CholeskyIntegrals::three_integral_pointer() { return ThreeIntegral_->pointer(); }

ambit::Tensor CholeskyIntegrals::three_integral_block(const std::vector<size_t>& A,
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

ambit::Tensor CholeskyIntegrals::three_integral_block_two_index(const std::vector<size_t>&, size_t,
                                                                const std::vector<size_t>&) {
    outfile->Printf("\n Oh no! this isn't here");
    throw psi::PSIEXCEPTION("INT_TYPE=DISKDF");
}

void CholeskyIntegrals::gather_integrals() {
    if (print_) {
        outfile->Printf("\n  Computing the Cholesky Vectors \n");
    }
    std::shared_ptr<psi::BasisSet> primary = wfn_->basisset();
    size_t nbf = primary->nbf();

    /// Needed to generate sieve information
    auto integral = std::make_shared<IntegralFactory>(primary, primary, primary, primary);

    double tol_cd = options_->get_double("CHOLESKY_TOLERANCE");

    // This is creates the cholesky decomposed AO integrals
    local_timer timer;
    auto twobodyaoints = std::shared_ptr<TwoBodyAOInt>(integral->eri());
    auto Ch = std::make_shared<CholeskyERI>(twobodyaoints, options_->get_double("INTS_TOLERANCE"),
                                            tol_cd, psi::Process::environment.get_memory());

    if (options_->get_str("DF_INTS_IO") == "LOAD") {
        // std::shared_ptr<ERISieve> sieve(
        //     new ERISieve(primary, options_->get_double("INTS_TOLERANCE")));
        const std::vector<std::pair<int, int>>& function_pairs = twobodyaoints->function_pairs();
        size_t ntri = twobodyaoints->function_pairs().size();
        size_t nbf = primary->nbf();
        std::string str = "Reading CD Integrals";
        if (print_) {
            outfile->Printf("\n    %-36s ...", str.c_str());
        }

        auto psio = std::make_shared<PSIO>();
        int file_unit = PSIF_DFSCF_BJ;

        if (psio->exists(file_unit)) {
            psio->open(file_unit, PSIO_OPEN_OLD);
            psio->read_entry(file_unit, "length", (char*)&nthree_, sizeof(long int));
            auto L_tri = std::make_shared<psi::Matrix>("Partial Cholesky", nthree_, ntri);
            double** Lp = L_tri->pointer();
            psio->read_entry(file_unit, "(Q|mn) Integrals", (char*)Lp[0],
                             sizeof(double) * nthree_ * ntri);
            psio->close(file_unit, 1);
            auto L_ao = std::make_shared<psi::Matrix>("Partial Cholesky", nthree_, nbf * nbf);
            for (size_t mn = 0; mn < ntri; mn++) {
                size_t m = function_pairs[mn].first;
                size_t n = function_pairs[mn].second;
                for (size_t P = 0; P < nthree_; P++) {
                    L_ao->set(P, (m * nbf) + n, L_tri->get(P, mn));
                    L_ao->set(P, (n * nbf) + m, L_tri->get(P, mn));
                }
            }
            L_ao_ = L_ao;
            if (print_) {
                outfile->Printf("...Done. Timing %15.6f s", timer.get());
            }
        } else {
            outfile->Printf("\n File PSIF_DFSCF_BJ(cholesky integrals) was not generated");
            outfile->Printf("\n");
            if (print_) {
                outfile->Printf("\n  Computing CD Integrals");
            }
            Ch->choleskify();
            nthree_ = Ch->Q();
            L_ao_ = Ch->L();
            if (print_) {
                print_timing("cholesky transformation", timer.get());
            }
        }
    } else {
        if (print_) {
            outfile->Printf("\n  Computing CD Integrals");
        }
        Ch->choleskify();
        nthree_ = Ch->Q();
        L_ao_ = Ch->L();
        if (print_) {
            print_timing("cholesky transformation", timer.get());
        }
    }

    // The number of vectors required to do cholesky factorization
    if (print_) {
        auto mem_info = to_xb2<double>(nthree_ * nbf * nbf);
        outfile->Printf("\n  Need %.2f %s to store CD integrals in core\n", mem_info.first,
                        mem_info.second.c_str());
    }
    int_mem_ = (nthree_ * nbf * nbf * sizeof(double) / 1073741824.0);

    if (print_) {
        outfile->Printf("\n  Number of Cholesky vectors required for %.3e tolerance: %d\n", tol_cd,
                        nthree_);
    }
    transform_integrals();
}

void CholeskyIntegrals::transform_integrals() {
    TensorType tensor_type = CoreTensor;

    auto Ca_ao = Ca_AO();

    auto ThreeIntegral_ao = ambit::Tensor::build(tensor_type, "ThreeIndex", {nthree_, nso_, nso_});
    auto Cpq_tensor = ambit::Tensor::build(tensor_type, "C_sorted", {nso_, nmo_});
    auto ThreeIntegral = ambit::Tensor::build(tensor_type, "ThreeIndex", {nthree_, nmo_, nmo_});

    Cpq_tensor.iterate(
        [&](const std::vector<size_t>& i, double& value) { value = Ca_ao->get(i[0], i[1]); });
    ThreeIntegral_ao.iterate([&](const std::vector<size_t>& i, double& value) {
        value = L_ao_->get(i[0], i[1] * nso_ + i[2]);
    });
    auto ThreeInt = std::make_shared<psi::Matrix>("Lmo", nmo_ * nmo_, nthree_);
    ThreeIntegral_ = ThreeInt;

    ThreeIntegral("L,p,q") = ThreeIntegral_ao("L,m,n") * Cpq_tensor("m,p") * Cpq_tensor("n,q");

    ThreeIntegral.iterate([&](const std::vector<size_t>& i, double& value) {
        ThreeIntegral_->set(i[1] * nmo_ + i[2], i[0], value);
    });
}

void CholeskyIntegrals::resort_integrals_after_freezing() {
    if (print_ > 0) {
        outfile->Printf("\n  Resorting integrals after freezing core.");
    }
    // Resort the three-index integrals
    resort_three(ThreeIntegral_, cmotomo_);
}

void CholeskyIntegrals::resort_three(std::shared_ptr<psi::Matrix>& threeint,
                                     std::vector<size_t>& map) {
    // Create a temperature threeint matrix
    std::shared_ptr<psi::Matrix> temp_threeint(threeint->clone());
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

void CholeskyIntegrals::set_tei(size_t, size_t, size_t, size_t, double, bool, bool) {
    outfile->Printf("\n If you are using this, you are ruining the advantages of DF/CD");
    throw psi::PSIEXCEPTION("Don't use DF/CD if you use set_tei");
}

size_t CholeskyIntegrals::nthree() const { return nthree_; }
} // namespace forte
