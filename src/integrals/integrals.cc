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

#include <algorithm>
#include <cmath>
#include <numeric>

#include "psi4/libpsi4util/PsiOutStream.h"
#include "psi4/libmints/wavefunction.h"

//#include "psi4/libmints/molecule.h"
//#include "psi4/libpsi4util/process.h"
//#include "psi4/libfock/jk.h"
//#include "psi4/libmints/basisset.h"
//#include "psi4/libmints/factory.h"
//#include "psi4/libmints/integral.h"
//#include "psi4/libmints/mintshelper.h"
//#include "psi4/libqt/qt.h"
//#include "psi4/libtrans/integraltransform.h"
//#include "psi4/psi4-dec.h"
//#include "psi4/psifiles.h"

#include "helpers/blockedtensorfactory.h"
#include "base_classes/forte_options.h"
#include "base_classes/mo_space_info.h"
#include "helpers/printing.h"
#include "helpers/timer.h"
#include "integrals.h"
#include "memory.h"

#ifdef HAVE_GA
#include <ga.h>
#include <macdecls.h>
#endif

using namespace psi;
using namespace ambit;

namespace forte {

#ifdef _OPENMP
#include <omp.h>
#else
#define omp_get_max_threads() 1
#define omp_get_thread_num() 0
#endif

std::map<IntegralType, std::string> int_type_label{
    {Conventional, "Conventional"},          {DF, "Density fitting"},
    {Cholesky, "Cholesky decomposition"},    {DiskDF, "Disk-based density fitting"},
    {DistDF, "Distributed density fitting"}, {Custom, "Custom"}};

ForteIntegrals::ForteIntegrals(std::shared_ptr<ForteOptions> options,
                               std::shared_ptr<psi::Wavefunction> ref_wfn,
                               std::shared_ptr<MOSpaceInfo> mo_space_info,
                               IntegralType integral_type, IntegralSpinRestriction restricted)
    : options_(options), wfn_(ref_wfn), integral_type_(integral_type),
      spin_restriction_(restricted), frozen_core_energy_(0.0), scalar_energy_(0.0),
      mo_space_info_(mo_space_info) {}

ForteIntegrals::ForteIntegrals(std::shared_ptr<ForteOptions> options,
                               std::shared_ptr<MOSpaceInfo> mo_space_info,
                               IntegralType integral_type, IntegralSpinRestriction restricted)
    : options_(options), integral_type_(integral_type), spin_restriction_(restricted),
      frozen_core_energy_(0.0), scalar_energy_(0.0), mo_space_info_(mo_space_info) {}

std::shared_ptr<psi::Matrix> ForteIntegrals::Ca() const { return Ca_; }

std::shared_ptr<psi::Matrix> ForteIntegrals::Cb() const { return Cb_; }

double ForteIntegrals::nuclear_repulsion_energy() const { return nucrep_; }

std::shared_ptr<psi::Wavefunction> ForteIntegrals::wfn() { return wfn_; }

size_t ForteIntegrals::nmo() const { return nmo_; }

int ForteIntegrals::nirrep() const { return nirrep_; }

psi::Dimension& ForteIntegrals::frzcpi() { return frzcpi_; }

psi::Dimension& ForteIntegrals::frzvpi() { return frzvpi_; }

psi::Dimension& ForteIntegrals::ncmopi() { return ncmopi_; }

size_t ForteIntegrals::ncmo() const { return ncmo_; }

void ForteIntegrals::set_print(int print) { print_ = print; }

double ForteIntegrals::frozen_core_energy() { return frozen_core_energy_; }

double ForteIntegrals::scalar() const { return scalar_energy_; }

double ForteIntegrals::oei_a(size_t p, size_t q) const {
    return one_electron_integrals_a_[p * aptei_idx_ + q];
}

double ForteIntegrals::oei_b(size_t p, size_t q) const {
    return one_electron_integrals_b_[p * aptei_idx_ + q];
}

ambit::Tensor ForteIntegrals::oei_a_block(const std::vector<size_t>& p,
                                          const std::vector<size_t>& q) {
    ambit::Tensor t = ambit::Tensor::build(ambit::CoreTensor, "oei_a", {p.size(), q.size()});
    t.iterate(
        [&](const std::vector<size_t>& i, double& value) { value = oei_a(p[i[0]], q[i[1]]); });
    return t;
}

ambit::Tensor ForteIntegrals::oei_b_block(const std::vector<size_t>& p,
                                          const std::vector<size_t>& q) {
    ambit::Tensor t = ambit::Tensor::build(ambit::CoreTensor, "oei_b", {p.size(), q.size()});
    t.iterate(
        [&](const std::vector<size_t>& i, double& value) { value = oei_b(p[i[0]], q[i[1]]); });
    return t;
}

double ForteIntegrals::get_fock_a(size_t p, size_t q) const {
    return fock_matrix_a_[p * aptei_idx_ + q];
}

double ForteIntegrals::get_fock_b(size_t p, size_t q) const {
    return fock_matrix_b_[p * aptei_idx_ + q];
}

std::vector<double> ForteIntegrals::get_fock_a() const { return fock_matrix_a_; }

std::vector<double> ForteIntegrals::get_fock_b() const { return fock_matrix_b_; }

void ForteIntegrals::set_scalar(double value) { scalar_energy_ = value; }

IntegralSpinRestriction ForteIntegrals::spin_restriction() const { return spin_restriction_; }

IntegralType ForteIntegrals::integral_type() const { return integral_type_; }

std::shared_ptr<psi::Matrix> ForteIntegrals::OneBody_symm() const { return OneBody_symm_; }

std::shared_ptr<psi::Matrix> ForteIntegrals::OneBodyAO() const { return OneIntsAO_; }

int ForteIntegrals::ga_handle() { return 0; }

std::vector<std::shared_ptr<psi::Matrix>> ForteIntegrals::ao_dipole_ints() const {
    return dipole_ints_ao_;
}

void ForteIntegrals::set_fock_a(const std::vector<double>& fock_stl) {
    size_t fock_size = fock_stl.size();
    if (fock_size != ncmo_ * ncmo_) {
        throw psi::PSIEXCEPTION("Cannot fill in fock_matrix_a because the vector is out-of-range.");
    } else {
        fock_matrix_a_ = fock_stl;
    }
}

/// Set the beta fock matrix
void ForteIntegrals::set_fock_b(const std::vector<double>& fock_stl) {
    size_t fock_size = fock_stl.size();
    if (fock_size != ncmo_ * ncmo_) {
        throw psi::PSIEXCEPTION("Cannot fill in fock_matrix_b because the vector is out-of-range.");
    } else {
        fock_matrix_b_ = fock_stl;
    }
}

void ForteIntegrals::set_oei(double** ints, bool alpha) {
    std::vector<double>& p_oei = alpha ? one_electron_integrals_a_ : one_electron_integrals_b_;
    for (size_t p = 0; p < aptei_idx_; ++p) {
        for (size_t q = 0; q < aptei_idx_; ++q) {
            p_oei[p * aptei_idx_ + q] = ints[p][q];
        }
    }
}

void ForteIntegrals::set_oei(size_t p, size_t q, double value, bool alpha) {
    std::vector<double>& p_oei = alpha ? one_electron_integrals_a_ : one_electron_integrals_b_;
    p_oei[p * aptei_idx_ + q] = value;
}

bool ForteIntegrals::test_orbital_spin_restriction(std::shared_ptr<psi::Matrix> A,
                                                   std::shared_ptr<psi::Matrix> B) const {
    std::shared_ptr<psi::Matrix> A_minus_B = A->clone();
    A_minus_B->subtract(B);
    return (A_minus_B->absmax() < 1.0e-7 ? true : false);
}

void ForteIntegrals::freeze_core_orbitals() {
    local_timer freeze_timer;
    if (ncmo_ < nmo_) {
        compute_frozen_one_body_operator();
        resort_integrals_after_freezing();
        aptei_idx_ = ncmo_;
    }
    if (print_) {
        print_timing("freezing core and virtual orbitals", freeze_timer.get());
    }
}

void ForteIntegrals::print_info() {
    outfile->Printf("\n\n  ==> Integral Transformation <==\n");
    outfile->Printf("\n  Number of molecular orbitals:            %10d", nmopi_.sum());
    outfile->Printf("\n  Number of correlated molecular orbitals: %10zu", ncmo_);
    outfile->Printf("\n  Number of frozen occupied orbitals:      %10d", frzcpi_.sum());
    outfile->Printf("\n  Number of frozen unoccupied orbitals:    %10d", frzvpi_.sum());
    outfile->Printf("\n  Two-electron integral type:              %10s\n\n",
                    int_type_label[integral_type()].c_str());
}

void ForteIntegrals::print_ints() {

    Ca_->print();
    Cb_->print();

    outfile->Printf("\n  Alpha one-electron integrals (T + V_{en})");
    Matrix ha(" Alpha one-electron integrals (T + V_{en})", nmo_, nmo_);
    for (size_t p = 0; p < nmo_; ++p) {
        for (size_t q = 0; q < nmo_; ++q) {
            //       ha.set(p, q, oei_a(p, q));
            if (std::abs(oei_a(p, q)) >= 1e-14)
                outfile->Printf("\n  h[%6d][%6d] = %20.12f", p, q, oei_a(p, q));
        }
    }
    //  ha.print();

    outfile->Printf("\n  Beta one-electron integrals (T + V_{en})");
    for (size_t p = 0; p < nmo_; ++p) {
        for (size_t q = 0; q < nmo_; ++q) {
            if (std::abs(oei_b(p, q)) >= 1e-14)
                outfile->Printf("\n  h[%6d][%6d] = %20.12f", p, q, oei_b(p, q));
        }
    }
    /*
        outfile->Printf("\n  Alpha-alpha two-electron integrals <pq||rs>");
        for (size_t p = 0; p < nmo_; ++p) {
            for (size_t q = 0; q < nmo_; ++q) {
                for (size_t r = 0; r < nmo_; ++r) {
                    for (size_t s = 0; s < nmo_; ++s) {
                    if( std::abs(aptei_aa(p,q,r,s)) >= 1e-14 )
                        outfile->Printf("\n  v[%6d][%6d][%6d][%6d] = %20.12f", p, q, r, s,
                                        aptei_aa(p, q, r, s));
                    }
                }
            }
        }

        outfile->Printf("\n  Alpha-beta two-electron integrals <pq||rs>");
        for (size_t p = 0; p < nmo_; ++p) {
            for (size_t q = 0; q < nmo_; ++q) {
                for (size_t r = 0; r < nmo_; ++r) {
                    for (size_t s = 0; s < nmo_; ++s) {
                    if( std::abs(aptei_ab(p,q,r,s)) >= 1e-14 )
                        outfile->Printf("\n  v[%6d][%6d][%6d][%6d] = %20.12f", p, q, r, s,
                                        aptei_ab(p, q, r, s));
                    }
                }
            }
        }
        outfile->Printf("\n  Beta-beta two-electron integrals <pq||rs>");
        for (size_t p = 0; p < nmo_; ++p) {
            for (size_t q = 0; q < nmo_; ++q) {
                for (size_t r = 0; r < nmo_; ++r) {
                    for (size_t s = 0; s < nmo_; ++s) {
                    if( std::abs(aptei_bb(p,q,r,s)) >= 1e-14 )
                        outfile->Printf("\n  v[%6d][%6d][%6d][%6d] = %20.12f", p, q, r, s,
                                        aptei_bb(p, q, r, s));
                    }
                }
            }
        }
    */
}

void ForteIntegrals::build_dipole_ints_ao() { _undefined_function("build_dipole_ints_ao"); }

std::vector<std::shared_ptr<psi::Matrix>>
ForteIntegrals::dipole_ints_mo_helper(std::shared_ptr<psi::Matrix> Cao, psi::SharedVector epsilon,
                                      const bool& resort) {
    std::vector<std::shared_ptr<psi::Matrix>> MOdipole_ints;
    _undefined_function("dipole_ints_mo_helper");

    return MOdipole_ints;
}

// The following functions throw an error by default

void ForteIntegrals::rotate_orbitals(std::shared_ptr<psi::Matrix> Ua,
                                     std::shared_ptr<psi::Matrix> Ub) {
    _undefined_function("rotate_orbitals");
}

void ForteIntegrals::update_orbitals(std::shared_ptr<psi::Matrix> Ca,
                                     std::shared_ptr<psi::Matrix> Cb) {
    _undefined_function("update_orbitals");
}

ambit::Tensor ForteIntegrals::three_integral_block(const std::vector<size_t>&,
                                                   const std::vector<size_t>&,
                                                   const std::vector<size_t>&) {
    _undefined_function("three_integral_block");
}

ambit::Tensor ForteIntegrals::three_integral_block_two_index(const std::vector<size_t>&, size_t,
                                                             const std::vector<size_t>&) {
    _undefined_function("three_integral_block_two_index");
}

double** ForteIntegrals::three_integral_pointer() { _undefined_function("three_integral_pointer"); }

void ForteIntegrals::rotate_mos() { _undefined_function("rotate_mos"); }

std::vector<std::shared_ptr<psi::Matrix>> ForteIntegrals::mo_dipole_ints(const bool& alpha,
                                                                         const bool& resort) {
    std::vector<std::shared_ptr<psi::Matrix>> mo_dipole_ints_;
    _undefined_function("compute_MOdipole_ints");
    return mo_dipole_ints_;
}

void ForteIntegrals::_undefined_function(const std::string& method) {
    outfile->Printf("\n  ForteIntegrals::" + method + "not supported for integral type " +
                    std::to_string(integral_type()));
    throw std::runtime_error("ForteIntegrals::" + method + " not supported for integral type " +
                             std::to_string(integral_type()));
}
} // namespace forte
