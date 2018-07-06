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

#include <algorithm>
#include <cmath>
#include <numeric>

#include "psi4/libfock/jk.h"
#include "psi4/libmints/basisset.h"
#include "psi4/libmints/factory.h"
#include "psi4/libmints/integral.h"
#include "psi4/libmints/mintshelper.h"
#include "psi4/libmints/wavefunction.h"
#include "psi4/libpsio/psio.hpp"
#include "psi4/libqt/qt.h"
#include "psi4/libtrans/integraltransform.h"
#include "psi4/psi4-dec.h"
#include "psi4/psifiles.h"

#include "../blockedtensorfactory.h"
#include "../forte_options.h"
#include "../helpers.h"
#include "integrals.h"
#include "memory.h"

#ifdef HAVE_GA
#include <ga.h>
#include <macdecls.h>
#endif

using namespace psi;
using namespace ambit;

namespace psi {
namespace forte {

#ifdef _OPENMP
#include <omp.h>
bool ForteIntegrals::have_omp_ = true;
#else
#define omp_get_max_threads() 1
#define omp_get_thread_num() 0
bool ForteIntegrals::have_omp_ = false;
#endif

void set_INT_options(ForteOptions& foptions) {
    /*- The algorithm used to screen the determinant
     *  - CONVENTIONAL Conventional two-electron integrals
     *  - DF Density fitted two-electron integrals
     *  - CHOLESKY Cholesky decomposed two-electron integrals -*/
    foptions.add_str("INT_TYPE", "CONVENTIONAL",
                     {"CONVENTIONAL", "DF", "CHOLESKY", "DISKDF", "DISTDF", "ALL", "OWNINTEGRALS"},
                     "The integral type");

    /*- The screening for JK builds and DF libraries -*/
    foptions.add_double("INTEGRAL_SCREENING", 1e-12,
                        "The screening for JK builds and DF libraries");

    /* - The tolerance for cholesky integrals */
    foptions.add_double("CHOLESKY_TOLERANCE", 1e-6, "The tolerance for cholesky integrals");

    foptions.add_bool("PRINT_INTS", false, "Print the one- and two-electron integrals?");
}

ForteIntegrals::ForteIntegrals(psi::Options& options, SharedWavefunction ref_wfn,
                               IntegralSpinRestriction restricted,
                               IntegralFrozenCore resort_frozen_core,
                               std::shared_ptr<MOSpaceInfo> mo_space_info)
    : options_(options), wfn_(ref_wfn), restricted_(restricted),
      resort_frozen_core_(resort_frozen_core), frozen_core_energy_(0.0), scalar_(0.0),
      mo_space_info_(mo_space_info) {
    // Copy the Wavefunction object

    startup();
    allocate();
    transform_one_electron_integrals();
    build_AOdipole_ints();
}

ForteIntegrals::~ForteIntegrals() { deallocate(); }

void ForteIntegrals::startup() {
    // Grab the global (default) PSIO object, for file I/O
    std::shared_ptr<PSIO> psio(_default_psio_lib_);

    if (not wfn_) {
        outfile->Printf("\n  No wave function object found!  Run a scf "
                        "calculation first!\n");

        exit(1);
    }

    nirrep_ = wfn_->nirrep();
    nso_ = wfn_->nso();
    nmo_ = wfn_->nmo();
    nsopi_ = wfn_->nsopi();
    nmopi_ = wfn_->nmopi();
    frzcpi_ = wfn_->frzcpi();
    frzvpi_ = wfn_->frzvpi();
    frzcpi_ = mo_space_info_->get_dimension("FROZEN_DOCC");
    frzvpi_ = mo_space_info_->get_dimension("FROZEN_UOCC");
    ncmopi_ = mo_space_info_->get_dimension("CORRELATED");

    ncmo_ = ncmopi_.sum();

    outfile->Printf("\n\n  ==> Integral Transformation <==\n");
    outfile->Printf("\n  Number of molecular orbitals:            %5d", nmopi_.sum());
    outfile->Printf("\n  Number of correlated molecular orbitals: %5zu", ncmo_);
    outfile->Printf("\n  Number of frozen occupied orbitals:      %5d", frzcpi_.sum());
    outfile->Printf("\n  Number of frozen unoccupied orbitals:    %5d\n\n", frzvpi_.sum());

    // Indexing
    // This is important!  Set the indexing to work using the number of
    // molecular integrals
    aptei_idx_ = nmo_;
    num_oei = INDEX2(nmo_ - 1, nmo_ - 1) + 1;
    num_tei = INDEX4(nmo_ - 1, nmo_ - 1, nmo_ - 1, nmo_ - 1) + 1;
    num_aptei = nmo_ * nmo_ * nmo_ * nmo_;
    num_threads_ = omp_get_max_threads();
    print_ = options_.get_int("PRINT");
    /// If MO_ROTATE is set in option, call rotate_mos.
    /// Wasn't really sure where to put this function, but since, integrals is
    /// always called, this seems like a good spot.
    if (options_["ROTATE_MOS"].size() > 0) {
        rotate_mos();
    }
}

void ForteIntegrals::ForteIntegrals::allocate() {
    // Allocate the memory required to store the one-electron integrals
    one_electron_integrals_a = new double[nmo_ * nmo_];
    one_electron_integrals_b = new double[nmo_ * nmo_];

    fock_matrix_a = new double[nmo_ * nmo_];
    fock_matrix_b = new double[nmo_ * nmo_];
}

void ForteIntegrals::ForteIntegrals::deallocate() {
    // Deallocate the memory required to store the one-electron integrals
    delete[] one_electron_integrals_a;
    delete[] one_electron_integrals_b;

    delete[] fock_matrix_a;
    delete[] fock_matrix_b;
}

void ForteIntegrals::ForteIntegrals::resort_two(double*& ints, std::vector<size_t>& map) {
    // Store the integrals in a temporary array of dimension nmo x nmo
    double* temp_ints = new double[nmo_ * nmo_];
    for (size_t p = 0; p < nmo_ * nmo_; ++p) {
        temp_ints[p] = 0.0;
    }
    for (size_t p = 0; p < ncmo_; ++p) {
        for (size_t q = 0; q < ncmo_; ++q) {
            temp_ints[p * ncmo_ + q] = ints[map[p] * nmo_ + map[q]];
        }
    }
    // Delete old integrals and assign the pointer
    delete[] ints;
    ints = temp_ints;
}

void ForteIntegrals::ForteIntegrals::set_oei(double** ints, bool alpha) {
    double* p_oei = alpha ? one_electron_integrals_a : one_electron_integrals_b;
    for (size_t p = 0; p < aptei_idx_; ++p) {
        for (size_t q = 0; q < aptei_idx_; ++q) {
            p_oei[p * aptei_idx_ + q] = ints[p][q];
        }
    }
}

void ForteIntegrals::set_oei(size_t p, size_t q, double value, bool alpha) {
    double* p_oei = alpha ? one_electron_integrals_a : one_electron_integrals_b;
    p_oei[p * aptei_idx_ + q] = value;
}

void ForteIntegrals::transform_one_electron_integrals() {
    // Now we want the reference (SCF) wavefunction
    std::shared_ptr<PSIO> psio_ = PSIO::shared_object();

    SharedMatrix T = SharedMatrix(wfn_->matrix_factory()->create_matrix(PSIF_SO_T));
    SharedMatrix V = SharedMatrix(wfn_->matrix_factory()->create_matrix(PSIF_SO_V));

    MintsHelper mints(wfn_);
    T = mints.so_kinetic();
    V = mints.so_potential();

    SharedMatrix Ca = wfn_->Ca();
    SharedMatrix Cb = wfn_->Cb();

    SharedMatrix Ha = T->clone();
    SharedMatrix Hb = T->clone();
    Ha->add(V);
    Hb->add(V);

    OneIntsAO_ = Ha->clone();

    Ha->transform(Ca);
    Hb->transform(Cb);

    OneBody_symm_ = Ha;

    for (size_t pq = 0; pq < nmo_ * nmo_; ++pq)
        one_electron_integrals_a[pq] = 0.0;
    for (size_t pq = 0; pq < nmo_ * nmo_; ++pq)
        one_electron_integrals_b[pq] = 0.0;

    // Read the one-electron integrals (T + V, restricted)
    int offset = 0;
    for (int h = 0; h < nirrep_; ++h) {
        for (int p = 0; p < nmopi_[h]; ++p) {
            for (int q = 0; q < nmopi_[h]; ++q) {
                one_electron_integrals_a[(p + offset) * nmo_ + q + offset] = Ha->get(h, p, q);
                one_electron_integrals_b[(p + offset) * nmo_ + q + offset] = Hb->get(h, p, q);
            }
        }
        offset += nmopi_[h];
    }
}

void ForteIntegrals::compute_frozen_one_body_operator() {
    Timer FrozenOneBody;

    Dimension frozen_dim = mo_space_info_->get_dimension("FROZEN_DOCC");
    Dimension nmopi = mo_space_info_->get_dimension("ALL");
    // Need to get the inactive block of the C matrix
    Dimension nsopi = wfn_->nsopi();
    SharedMatrix Ca = wfn_->Ca();
    SharedMatrix C_core(new Matrix("C_core", nirrep_, nsopi, frozen_dim));

    for (int h = 0; h < nirrep_; h++) {
        for (int mu = 0; mu < nsopi[h]; mu++) {
            for (int i = 0; i < frozen_dim[h]; i++) {
                C_core->set(h, mu, i, Ca->get(h, mu, i));
            }
        }
    }

    std::shared_ptr<JK> JK_core;
    if (options_.get_str("SCF_TYPE") == "GTFOCK") {
#ifdef HAVE_JK_FACTORY
        Process::environment.set_legacy_molecule(wfn_->molecule());
        JK_core = std::shared_ptr<JK>(new GTFockJK(wfn_->basisset()));
#else
        throw PSIEXCEPTION("GTFock was not compiled in this version");
#endif
    } else {
        if (options_.get_str("SCF_TYPE") == "DF") {
            JK_core = JK::build_JK(wfn_->basisset(), wfn_->get_basisset("DF_BASIS_MP2"), options_, "MEM_DF");
        } else {
            JK_core = JK::build_JK(wfn_->basisset(), BasisSet::zero_ao_basis_set(), options_);
        }
    }

    JK_core->set_memory(Process::environment.get_memory() * 0.8);
    /// Already transform everything to C1 so make sure JK does not do this.

    // JK_core->set_cutoff(options_.get_double("INTEGRAL_SCREENING"));
    JK_core->set_cutoff(options_.get_double("INTEGRAL_SCREENING"));
    JK_core->initialize();
    JK_core->set_do_J(true);
    // JK_core->set_allow_desymmetrization(true);
    JK_core->set_do_K(true);

    std::vector<std::shared_ptr<Matrix>>& Cl = JK_core->C_left();
    std::vector<std::shared_ptr<Matrix>>& Cr = JK_core->C_right();

    Cl.clear();
    Cr.clear();
    Cl.push_back(C_core);
    Cr.push_back(C_core);

    JK_core->compute();

    SharedMatrix F_core = JK_core->J()[0];
    SharedMatrix K_core = JK_core->K()[0];

    F_core->scale(2.0);
    F_core->subtract(K_core);
    F_core->transform(Ca);
    int offset = 0;
    for (int h = 0; h < nirrep_; h++) {
        for (int p = 0; p < nmopi[h]; ++p) {
            for (int q = 0; q < nmopi[h]; ++q) {
                one_electron_integrals_a[(p + offset) * nmo_ + (q + offset)] +=
                    F_core->get(h, p, q);
                one_electron_integrals_b[(p + offset) * nmo_ + (q + offset)] +=
                    F_core->get(h, p, q);
            }
        }
        offset += nmopi[h];
    }
    F_core->add(OneBody_symm_);

    frozen_core_energy_ = 0.0;
    double E_frozen = 0.0;
    for (int h = 0; h < nirrep_; h++) {
        for (int fr = 0; fr < frozen_dim[h]; fr++) {
            E_frozen += OneBody_symm_->get(h, fr, fr) + F_core->get(h, fr, fr);
        }
    }

    OneBody_symm_ = F_core;
    frozen_core_energy_ = E_frozen;

    if (print_ > 0) {
        outfile->Printf("\n  Frozen-core energy        %20.12f a.u.", frozen_core_energy_);

        outfile->Printf("\n\n  FrozenOneBody Operator takes  %8.8f s", FrozenOneBody.get());
    }
}

void ForteIntegrals::update_integrals(bool freeze_core) {
    Timer freezeOrbs;
    make_diagonal_integrals();
    if (freeze_core) {
        if (ncmo_ < nmo_) {
            freeze_core_orbitals();
            if (resort_frozen_core_ == RemoveFrozenMOs) {
                aptei_idx_ = ncmo_;
            }
        }
    }
    if (print_) {
        outfile->Printf("\n  Frozen Orbitals takes %9.3f s.", freezeOrbs.get());
    }
}

void ForteIntegrals::retransform_integrals() {
    aptei_idx_ = nmo_;
    transform_one_electron_integrals();
    int my_proc = 0;
#ifdef HAVE_GA
    my_proc = GA_Nodeid();
#endif
    if (my_proc == 0) {
        outfile->Printf("\n Integrals are about to be computed.");
        gather_integrals();
        outfile->Printf("\n Integrals are about to be updated.");
        update_integrals();
    }
}

void ForteIntegrals::freeze_core_orbitals() {
    compute_frozen_one_body_operator();
    if (resort_frozen_core_ == RemoveFrozenMOs) {
        resort_integrals_after_freezing();
    }
}

void ForteIntegrals::rotate_mos() {
    int size_mo_rotate = options_["ROTATE_MOS"].size();
    outfile->Printf("\n\n\n  ==> ROTATING MOS <==");
    if (size_mo_rotate % 3 != 0) {
        outfile->Printf("\n Check ROTATE_MOS array");
        outfile->Printf("\nFormat should be in group of 3s");
        outfile->Printf("\n Irrep, rotate_1, rotate_2, irrep, rotate_3, rotate_4");
        throw PSIEXCEPTION("User specifed ROTATE_MOS incorrectly.  Check output for notes");
    }
    int orbital_rotate_group = (size_mo_rotate / 3);
    std::vector<std::vector<int>> rotate_mo_list;
    outfile->Printf("\n\n  IRREP  MO_1  MO_2\n");
    for (int a = 0; a < orbital_rotate_group; a++) {
        std::vector<int> rotate_mo_group(3);
        int offset_a = 3 * a;
        rotate_mo_group[0] = options_["ROTATE_MOS"][offset_a].to_integer() - 1;
        if (rotate_mo_group[0] > nirrep_) {
            outfile->Printf("\n Irrep:%d does not match wfn_ symmetry:%d", rotate_mo_group[0],
                            nirrep_);
            throw PSIEXCEPTION("Irrep does not match wavefunction symmetry");
        }
        rotate_mo_group[1] = options_["ROTATE_MOS"][offset_a + 1].to_integer() - 1;
        rotate_mo_group[2] = options_["ROTATE_MOS"][offset_a + 2].to_integer() - 1;
        rotate_mo_list.push_back(rotate_mo_group);

        outfile->Printf("   %d   %d   %d\n", rotate_mo_group[0], rotate_mo_group[1],
                        rotate_mo_group[2]);
    }
    SharedMatrix C_old = wfn_->Ca();
    SharedMatrix C_new(C_old->clone());

    for (auto mo_group : rotate_mo_list) {
        SharedVector C_mo1 = C_old->get_column(mo_group[0], mo_group[1]);
        SharedVector C_mo2 = C_old->get_column(mo_group[0], mo_group[2]);
        C_new->set_column(mo_group[0], mo_group[2], C_mo1);
        C_new->set_column(mo_group[0], mo_group[1], C_mo2);
    }
    C_old->copy(C_new);
    
    SharedMatrix Cb_old = wfn_->Cb();
    Cb_old->copy(C_new);
}

void ForteIntegrals::print_ints() {

    wfn_->Ca()->print();
    wfn_->Cb()->print();

    outfile->Printf("\n  Alpha one-electron integrals (T + V_{en})");
    Matrix ha(" Alpha one-electron integrals (T + V_{en})", nmo_, nmo_);
    for (size_t p = 0; p < nmo_; ++p) {
        for (size_t q = 0; q < nmo_; ++q) {
            ha.set(p, q, oei_a(p, q));
            //            outfile->Printf("\n  h[%6d][%6d] = %20.12f", p, q, oei_a(p, q));
        }
    }
    ha.print();

    outfile->Printf("\n  Beta one-electron integrals (T + V_{en})");
    for (size_t p = 0; p < nmo_; ++p) {
        for (size_t q = 0; q < nmo_; ++q) {
            outfile->Printf("\n  h[%6d][%6d] = %20.12f", p, q, oei_b(p, q));
        }
    }

    outfile->Printf("\n  Alpha-alpha two-electron integrals <pq||rs>");
    for (size_t p = 0; p < nmo_; ++p) {
        for (size_t q = 0; q < nmo_; ++q) {
            for (size_t r = 0; r < nmo_; ++r) {
                for (size_t s = 0; s < nmo_; ++s) {
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
                    outfile->Printf("\n  v[%6d][%6d][%6d][%6d] = %20.12f", p, q, r, s,
                                    aptei_bb(p, q, r, s));
                }
            }
        }
    }
}

void ForteIntegrals::build_AOdipole_ints() {
    std::shared_ptr<BasisSet> basisset = wfn_->basisset();
    std::shared_ptr<IntegralFactory> ints_fac = std::make_shared<IntegralFactory>(basisset);
    int nbf = basisset->nbf();

    AOdipole_ints_.clear();
    for (const std::string& direction : {"X", "Y", "Z"}) {
        std::string name = "AO Dipole " + direction;
        AOdipole_ints_.push_back(SharedMatrix(new Matrix(name, nbf, nbf)));
    }
    std::shared_ptr<OneBodyAOInt> aodOBI(ints_fac->ao_dipole());
    aodOBI->compute(AOdipole_ints_);
}

std::vector<SharedMatrix> ForteIntegrals::compute_MOdipole_ints(const bool& alpha,
                                                                const bool& resort) {
    if (alpha) {
        return MOdipole_ints_helper(wfn_->Ca_subset("AO"), wfn_->epsilon_a(), resort);
    } else {
        return MOdipole_ints_helper(wfn_->Cb_subset("AO"), wfn_->epsilon_b(), resort);
    }
}

std::vector<SharedMatrix>
ForteIntegrals::MOdipole_ints_helper(SharedMatrix Cao, SharedVector epsilon, const bool& resort) {
    std::vector<SharedMatrix> MOdipole_ints;
    std::vector<std::string> names{"X", "Y", "Z"};
    for (int i = 0; i < 3; ++i) {
        SharedMatrix modipole(AOdipole_ints_[i]->clone());
        modipole->set_name("MO Dipole " + names[i]);
        modipole->transform(Cao);
        MOdipole_ints.push_back(modipole);
    }

    if (resort) {
        // figure out the correspondance between C1 and Pitzer
        std::vector<std::tuple<double, int, int>> order;
        for (int h = 0; h < nirrep_; ++h) {
            for (int i = 0; i < nmopi_[h]; ++i) {
                order.push_back(std::tuple<double, int, int>(epsilon->get(h, i), i, h));
            }
        }
        std::sort(order.begin(), order.end(), std::less<std::tuple<double, int, int>>());

        std::vector<int> irrep_offset(nirrep_, 0);
        for (int h = 1, sum = 0; h < nirrep_; ++h) {
            sum += nmopi_[h - 1];
            irrep_offset[h] = sum;
        }

        std::vector<int> indices;
        for (int iC1 = 0; iC1 < (int)nmo_; ++iC1) {
            int i = std::get<1>(order[iC1]);
            int h = std::get<2>(order[iC1]);
            indices.push_back(irrep_offset[h] + i);
        }

        for (int i = 0; i < 3; ++i) {
            SharedMatrix modipole(new Matrix("MO Dipole " + names[i], (int)nmo_, (int)nmo_));
            for (int p = 0; p < (int)nmo_; ++p) {
                int np = indices[p];
                for (int q = 0; q < (int)nmo_; ++q) {
                    int nq = indices[q];
                    modipole->set(np, nq, MOdipole_ints[i]->get(p, q));
                }
            }
            MOdipole_ints[i] = modipole;
        }
    }

    return MOdipole_ints;
}
}
}
