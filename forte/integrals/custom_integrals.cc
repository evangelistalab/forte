/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2021 by its authors (see COPYING, COPYING.LESSER, AUTHORS).
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
#include <fstream>

#include "psi4/libdpd/dpd.h"
#include "psi4/libmints/mintshelper.h"
#include "psi4/libmints/matrix.h"
#include "psi4/libpsio/psio.hpp"
#include "psi4/psi4-dec.h"
#include "psi4/psifiles.h"

#include "base_classes/mo_space_info.h"
#include "helpers/blockedtensorfactory.h"
#include "helpers/string_algorithms.h"
#include "helpers/timer.h"
#include "helpers/printing.h"

#include "custom_integrals.h"

#define IOFFINDEX(i) (i * (i + 1) / 2)
#define PAIRINDEX(i, j) ((i > j) ? (IOFFINDEX(i) + (j)) : (IOFFINDEX(j) + (i)))
#define four(i, j, k, l) PAIRINDEX(PAIRINDEX(i, j), PAIRINDEX(k, l))

using namespace psi;

namespace forte {

CustomIntegrals::CustomIntegrals(std::shared_ptr<ForteOptions> options,
                                 std::shared_ptr<MOSpaceInfo> mo_space_info,
                                 IntegralSpinRestriction restricted, double scalar,
                                 const std::vector<double>& oei_a, const std::vector<double>& oei_b,
                                 const std::vector<double>& tei_aa,
                                 const std::vector<double>& tei_ab,
                                 const std::vector<double>& tei_bb)
    : ForteIntegrals(options, mo_space_info, Custom, restricted), full_aphys_tei_aa_(tei_aa),
      full_aphys_tei_ab_(tei_ab), full_aphys_tei_bb_(tei_bb) {
    set_nuclear_repulsion(scalar);
    set_oei_all(oei_a, oei_b);
    initialize();
}

void CustomIntegrals::initialize() {
    Ca_ = std::make_shared<psi::Matrix>(nmopi_, nmopi_);
    Cb_ = std::make_shared<psi::Matrix>(nmopi_, nmopi_);
    Ca_->identity();
    Cb_->identity();
    nsopi_ = nmopi_;
    nso_ = nmo_;

    print_info();
    local_timer int_timer;
    gather_integrals();
    freeze_core_orbitals();
    print_timing("preparing custom (FCIDUMP) integrals", int_timer.get());
}

double CustomIntegrals::aptei_aa(size_t p, size_t q, size_t r, size_t s) {
    return aphys_tei_aa_[aptei_index(p, q, r, s)];
}

double CustomIntegrals::aptei_ab(size_t p, size_t q, size_t r, size_t s) {
    return aphys_tei_ab_[aptei_index(p, q, r, s)];
}

double CustomIntegrals::aptei_bb(size_t p, size_t q, size_t r, size_t s) {
    return aphys_tei_bb_[aptei_index(p, q, r, s)];
}

ambit::Tensor CustomIntegrals::aptei_aa_block(const std::vector<size_t>& p,
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

ambit::Tensor CustomIntegrals::aptei_ab_block(const std::vector<size_t>& p,
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

ambit::Tensor CustomIntegrals::aptei_bb_block(const std::vector<size_t>& p,
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

void CustomIntegrals::set_tei(size_t p, size_t q, size_t r, size_t s, double value, bool alpha1,
                              bool alpha2) {
    size_t index = aptei_index(p, q, r, s);
    if (alpha1 == true and alpha2 == true)
        aphys_tei_aa_[index] = value;
    if (alpha1 == true and alpha2 == false)
        aphys_tei_ab_[index] = value;
    if (alpha1 == false and alpha2 == false)
        aphys_tei_bb_[index] = value;
}

void CustomIntegrals::gather_integrals() {
    // Copy the correlated part into one_electron_integrals_a/one_electron_integrals_b
    for (size_t p = 0; p < ncmo_; ++p) {
        for (size_t q = 0; q < ncmo_; ++q) {
            one_electron_integrals_a_[p * ncmo_ + q] =
                full_one_electron_integrals_a_[cmotomo_[p] * nmo_ + cmotomo_[q]];
            one_electron_integrals_b_[p * ncmo_ + q] =
                full_one_electron_integrals_b_[cmotomo_[p] * nmo_ + cmotomo_[q]];
        }
    }
    aphys_tei_aa_ = full_aphys_tei_aa_;
    aphys_tei_ab_ = full_aphys_tei_ab_;
    aphys_tei_bb_ = full_aphys_tei_bb_;
}

void CustomIntegrals::resort_integrals_after_freezing() {
    if (print_ > 0) {
        outfile->Printf("\n  Resorting integrals after freezing core.");
    }
    // Resort the four-index integrals
    resort_four(aphys_tei_aa_, cmotomo_);
    resort_four(aphys_tei_ab_, cmotomo_);
    resort_four(aphys_tei_bb_, cmotomo_);
}

void CustomIntegrals::resort_four(std::vector<double>& tei, std::vector<size_t>& map) {
    // Store the integrals in a temporary array
    size_t num_aptei_corr = ncmo_ * ncmo_ * ncmo_ * ncmo_;
    std::vector<double> temp_ints(num_aptei_corr, 0.0);
    for (size_t p = 0; p < ncmo_; ++p) {
        for (size_t q = 0; q < ncmo_; ++q) {
            for (size_t r = 0; r < ncmo_; ++r) {
                for (size_t s = 0; s < ncmo_; ++s) {
                    size_t pqrs_cmo = ncmo_ * ncmo_ * ncmo_ * p + ncmo_ * ncmo_ * q + ncmo_ * r + s;
                    size_t pqrs_mo =
                        nmo_ * nmo_ * nmo_ * map[p] + nmo_ * nmo_ * map[q] + nmo_ * map[r] + map[s];
                    temp_ints[pqrs_cmo] = tei[pqrs_mo];
                }
            }
        }
    }
    temp_ints.swap(tei);
}

void CustomIntegrals::compute_frozen_one_body_operator() {
    local_timer timer_frozen_one_body;

    auto nfrzcpi = mo_space_info_->dimension("FROZEN_DOCC");
    auto f = make_fock_inactive(psi::Dimension(nirrep_), nfrzcpi);
    auto Fock_a = std::get<0>(f);
    auto Fock_b = std::get<1>(f);
    frozen_core_energy_ = std::get<2>(f);

    // This loop grabs only the correlated part of the correction
    for (int h = 0, corr_offset = 0, full_offset = 0; h < nirrep_; h++) {
        for (int p = 0; p < ncmopi_[h]; ++p) {
            auto p_corr = p + corr_offset;
            auto p_full = cmotomo_[p + corr_offset] - full_offset;

            for (int q = 0; q < ncmopi_[h]; ++q) {
                auto q_corr = q + corr_offset;
                auto q_full = cmotomo_[q + corr_offset] - full_offset;

                one_electron_integrals_a_[p_corr * ncmo_ + q_corr] = Fock_a->get(h, p_full, q_full);
                one_electron_integrals_b_[p_corr * ncmo_ + q_corr] = Fock_b->get(h, p_full, q_full);
            }
        }

        full_offset += nmopi_[h];
        corr_offset += ncmopi_[h];
    }

    if (print_ > 0) {
        outfile->Printf("\n  Frozen-core energy        %20.15f a.u.", frozen_core_energy_);
        print_timing("frozen one-body operator", timer_frozen_one_body.get());
    }
    if (print_ > 2) {
        print_h1("One-body Hamiltonian elements dressed by frozen-core orbitals");
        Fock_a->set_name("Frozen One Body (alpha)");
        Fock_a->print();
        Fock_b->set_name("Frozen One Body (beta)");
        Fock_b->print();
    }
}

void CustomIntegrals::make_fock_matrix(ambit::Tensor Da, ambit::Tensor Db) {
    // build inactive Fock
    auto rdoccpi = mo_space_info_->dimension("INACTIVE_DOCC");
    auto fock_closed = make_fock_inactive(psi::Dimension(nirrep_), rdoccpi);

    // build active Fock
    auto fock_active = make_fock_active(Da, Db);

    // add them together
    fock_a_ = std::get<0>(fock_closed)->clone();
    fock_a_->add(std::get<0>(fock_active));
    fock_a_->set_name("Fock alpha");

    fock_b_ = std::get<1>(fock_closed)->clone();
    fock_b_->add(std::get<1>(fock_active));
    fock_b_->set_name("Fock beta");

    fock_b_->subtract(fock_a_);
    if (fock_b_->absmax() < 1.0e-7) { // threshold consistent with test_orbital_spin_restriction
        fock_b_ = fock_a_;
    } else {
        fock_b_->add(fock_a_);
    }
}

std::tuple<psi::SharedMatrix, psi::SharedMatrix, double>
CustomIntegrals::make_fock_inactive(psi::Dimension dim_start, psi::Dimension dim_end) {
    // Implementation Notes (spin-orbital)
    // F_{pq} = h_{pq} + \sum_{i}^{closed} <pi||qi>
    // e_closed = \sum_{i}^{closed} h_{ii} + 0.5 * \sum_{ij}^{closed} <ij||ij>

    auto dim = dim_end - dim_start;

    // we will use unrestricted formalism
    auto Fock_a = std::make_shared<psi::Matrix>("Fock_closed alpha", nmopi_, nmopi_);
    auto Fock_b = std::make_shared<psi::Matrix>("Fock_closed beta", nmopi_, nmopi_);

    // figure out closed-shell indices
    std::vector<int> closed_indices;
    for (int h2 = 0, offset2 = 0; h2 < nirrep_; ++h2) {
        for (int i = 0; i < dim[h2]; ++i) {
            auto ni = i + dim_start[h2] + offset2;
            closed_indices.push_back(ni);
        }
        offset2 += nmopi_[h2];
    }
    auto nclosed = closed_indices.size();

    auto nmo1 = nmo_;
    auto nmo2 = nmo1 * nmo1;
    auto nmo3 = nmo1 * nmo2;

    // compute inactive Fock
    for (int h = 0, offset = 0; h < nirrep_; ++h) {
        for (int p = 0; p < nmopi_[h]; ++p) {
            auto np = p + offset;
            for (int q = 0; q < nmopi_[h]; ++q) {
                auto nq = q + offset;

                double va = full_one_electron_integrals_a_[np * nmo_ + nq];
                double vb = full_one_electron_integrals_b_[np * nmo_ + nq];

#pragma omp parallel for reduction(+ : va, vb)
                for (size_t i = 0; i < nclosed; ++i) {
                    auto ni = closed_indices[i];

                    // Fock alpha: F_{pq} = h_{pq} + \sum_{i} <pi||qi> + \sum_{I} <pI||qI>
                    auto id_a = np * nmo3 + ni * nmo2 + nq * nmo1 + ni;
                    va += full_aphys_tei_aa_[id_a] + full_aphys_tei_ab_[id_a];

                    // Fock beta: F_{PQ} = h_{PQ} + \sum_{i} <Pi||Qi> + \sum_{I} <PI||QI>
                    auto id_b = ni * nmo3 + np * nmo2 + ni * nmo1 + nq;
                    vb += full_aphys_tei_bb_[id_b] + full_aphys_tei_ab_[id_b];
                }

                Fock_a->set(h, p, q, va);
                Fock_b->set(h, p, q, vb);
            }
        }
        offset += nmopi_[h];
    }

    // compute closed-shell energy
    double e_closed = 0.0;
    for (int h1 = 0, offset1 = 0; h1 < nirrep_; ++h1) {
        for (int i = 0; i < dim[h1]; ++i) {
            auto ni = i + dim_start[h1] + offset1;
            e_closed += full_one_electron_integrals_a_[ni * nmo_ + ni];
            e_closed += full_one_electron_integrals_b_[ni * nmo_ + ni];

#pragma omp parallel for reduction(+ : e_closed)
            for (size_t j = 0; j < nclosed; ++j) {
                auto nj = closed_indices[j];

                auto idx = ni * nmo3 + nj * nmo2 + ni * nmo1 + nj;
                e_closed += 0.5 * (full_aphys_tei_aa_[idx] + full_aphys_tei_bb_[idx]);
                e_closed += full_aphys_tei_ab_[idx];
            }
        }
        offset1 += nmopi_[h1];
    }

    return {Fock_a, Fock_b, e_closed};
};

std::tuple<psi::SharedMatrix, psi::SharedMatrix>
CustomIntegrals::make_fock_active(ambit::Tensor Da, ambit::Tensor Db) {
    // Implementation Notes (spin-orbital)
    // F_{pq} = \sum_{uv}^{active} <pu||qv> * gamma_{uv}

    auto nactv = mo_space_info_->size("ACTIVE");
    auto dim_actv = mo_space_info_->dimension("ACTIVE");

    if (Da.dims() != Db.dims()) {
        throw std::runtime_error("Different dimensions of alpha and beta 1RDM!");
    }
    if (nactv != Da.dim(0)) {
        throw std::runtime_error("Inconsistent number of active orbitals");
    }

    // to have a single maintained code, we translate densities to psi::SharedMatrix
    auto g1a = std::make_shared<psi::Matrix>("1RDM alpha", dim_actv, dim_actv);
    auto g1b = std::make_shared<psi::Matrix>("1RDM beta", dim_actv, dim_actv);

    auto& Da_data = Da.data();
    auto& Db_data = Db.data();

    for (int h = 0, offset = 0; h < nirrep_; ++h) {
        for (int u = 0; u < dim_actv[h]; ++u) {
            int nu = u + offset;
            for (int v = 0; v < dim_actv[h]; ++v) {
                int nv = v + offset;
                g1a->set(h, u, v, Da_data[nu * nactv + nv]);
                g1b->set(h, u, v, Db_data[nu * nactv + nv]);
            }
        }
        offset += dim_actv[h];
    }

    return make_fock_active_unrestricted(g1a, g1b);
};

psi::SharedMatrix CustomIntegrals::make_fock_active_restricted(psi::SharedMatrix D) {
    auto g1a = D->clone();
    g1a->scale(0.5);
    auto Ftuple = make_fock_active_unrestricted(g1a, g1a);
    return std::get<0>(Ftuple);
}

std::tuple<psi::SharedMatrix, psi::SharedMatrix>
CustomIntegrals::make_fock_active_unrestricted(psi::SharedMatrix g1a, psi::SharedMatrix g1b) {
    auto Fock_a = std::make_shared<psi::Matrix>("Fock_active alpha", nmopi_, nmopi_);
    auto Fock_b = std::make_shared<psi::Matrix>("Fock_active beta", nmopi_, nmopi_);

    auto abs_mo_actv = mo_space_info_->absolute_mo("ACTIVE");
    auto rel_mo_actv = mo_space_info_->relative_mo("ACTIVE");

    auto actv_dim = mo_space_info_->dimension("ACTIVE");

    // figure out symmetry allowed absolute index for active orbitals
    std::vector<std::tuple<size_t, size_t, size_t, size_t, size_t>> actv_indices_sym;
    for (int h = 0, offset = 0; h < nirrep_; ++h) {
        for (int u = 0; u < actv_dim[h]; ++u) {
            int nu = abs_mo_actv[u + offset];
            for (int v = 0; v < actv_dim[h]; ++v) {
                int nv = abs_mo_actv[v + offset];
                actv_indices_sym.push_back({h, u, v, nu, nv});
            }
        }
        offset += actv_dim[h];
    }
    auto actv_sym_size = actv_indices_sym.size();

    auto nmo1 = nmo_;
    auto nmo2 = nmo1 * nmo1;
    auto nmo3 = nmo1 * nmo2;

    // compute active Fock
    for (int h = 0, offset = 0; h < nirrep_; ++h) {
        for (int p = 0; p < nmopi_[h]; ++p) {
            auto np = p + offset;
            for (int q = 0; q < nmopi_[h]; ++q) {
                auto nq = q + offset;

                double va = 0.0;
                double vb = 0.0;

#pragma omp parallel for reduction(+ : va, vb)
                for (size_t i_sym = 0; i_sym < actv_sym_size; ++i_sym) {
                    size_t hactv, u, v, nu, nv;
                    std::tie(hactv, u, v, nu, nv) = actv_indices_sym[i_sym];

                    auto id_a = np * nmo3 + nu * nmo2 + nq * nmo1 + nv;
                    va += full_aphys_tei_aa_[id_a] * g1a->get(hactv, u, v);
                    va += full_aphys_tei_ab_[id_a] * g1b->get(hactv, u, v);

                    auto id_b = nu * nmo3 + np * nmo2 + nv * nmo1 + nq;
                    vb += full_aphys_tei_bb_[id_b] * g1b->get(hactv, u, v);
                    vb += full_aphys_tei_ab_[id_b] * g1a->get(hactv, u, v);
                }

                Fock_a->set(h, p, q, va);
                Fock_b->set(h, p, q, vb);
            }
        }
        offset += nmopi_[h];
    }

    return {Fock_a, Fock_b};
}

void CustomIntegrals::transform_one_electron_integrals() {
    // the first time we transform, we keep a copy of the original integrals
    if (original_full_one_electron_integrals_a_.size() == 0) {
        original_full_one_electron_integrals_a_ = full_one_electron_integrals_a_;
        original_full_one_electron_integrals_b_ = full_one_electron_integrals_b_;
    }

    // Grab the one-electron integrals from psi4's wave function object
    auto Ha = std::make_shared<psi::Matrix>(nmopi_, nmopi_);
    auto Hb = std::make_shared<psi::Matrix>(nmopi_, nmopi_);

    // Read the one-electron integrals (T + V)
    int offset = 0;
    for (int h = 0; h < nirrep_; ++h) {
        for (int p = 0; p < nmopi_[h]; ++p) {
            for (int q = 0; q < nmopi_[h]; ++q) {
                Ha->set(h, p, q,
                        original_full_one_electron_integrals_a_[(p + offset) * nmo_ + q + offset]);
                Hb->set(h, p, q,
                        original_full_one_electron_integrals_b_[(p + offset) * nmo_ + q + offset]);
            }
        }
        offset += nmopi_[h];
    }

    // transform the one-electron integrals
    Ha->transform(Ca_);
    Hb->transform(Cb_);

    OneBody_symm_ = Ha;

    // zero these vectors
    std::fill(full_one_electron_integrals_a_.begin(), full_one_electron_integrals_a_.end(), 0.0);
    std::fill(full_one_electron_integrals_b_.begin(), full_one_electron_integrals_b_.end(), 0.0);

    // Read the one-electron integrals (T + V, restricted)
    offset = 0;
    for (int h = 0; h < nirrep_; ++h) {
        for (int p = 0; p < nmopi_[h]; ++p) {
            for (int q = 0; q < nmopi_[h]; ++q) {
                full_one_electron_integrals_a_[(p + offset) * nmo_ + q + offset] = Ha->get(h, p, q);
                full_one_electron_integrals_b_[(p + offset) * nmo_ + q + offset] = Hb->get(h, p, q);
            }
        }
        offset += nmopi_[h];
    }
}

void CustomIntegrals::transform_two_electron_integrals() {
    if (not save_original_tei_) {
        original_V_aa_ = ambit::Tensor::build(tensor_type_, "V_aa", {nmo_, nmo_, nmo_, nmo_});
        original_V_ab_ = ambit::Tensor::build(tensor_type_, "V_ab", {nmo_, nmo_, nmo_, nmo_});
        original_V_bb_ = ambit::Tensor::build(tensor_type_, "V_bb", {nmo_, nmo_, nmo_, nmo_});

        original_V_aa_.data() = full_aphys_tei_aa_;
        original_V_ab_.data() = full_aphys_tei_ab_;
        original_V_bb_.data() = full_aphys_tei_bb_;

        save_original_tei_ = true;
    }

    auto Ca = ambit::Tensor::build(tensor_type_, "Ca", {nmo_, nmo_});
    auto Cb = ambit::Tensor::build(tensor_type_, "Cb", {nmo_, nmo_});

    auto& Ca_data = Ca.data();
    auto& Cb_data = Cb.data();

    int offset = 0;
    for (int h = 0; h < nirrep_; ++h) {
        for (int p = 0; p < nmopi_[h]; ++p) {
            for (int q = 0; q < nmopi_[h]; ++q) {
                int p_full = p + offset;
                int q_full = q + offset;
                Ca_data[p_full * nmo_ + q_full] = Ca_->get(h, p, q);
                Cb_data[p_full * nmo_ + q_full] = Cb_->get(h, p, q);
            }
        }
        offset += nmopi_[h];
    }

    auto T = ambit::Tensor::build(tensor_type_, "temp", {nmo_, nmo_, nmo_, nmo_});

    T("ijkl") = original_V_aa_("pqrs") * Ca("pi") * Ca("qj") * Ca("rk") * Ca("sl");
    full_aphys_tei_aa_ = T.data();

    T("ijkl") = original_V_ab_("pqrs") * Ca("pi") * Cb("qj") * Ca("rk") * Cb("sl");
    full_aphys_tei_ab_ = T.data();

    T("ijkl") = original_V_bb_("pqrs") * Cb("pi") * Cb("qj") * Cb("rk") * Cb("sl");
    full_aphys_tei_bb_ = T.data();
}

void CustomIntegrals::update_orbitals(std::shared_ptr<psi::Matrix> Ca,
                                      std::shared_ptr<psi::Matrix> Cb) {
    // 1. Copy orbitals and, if necessary, test they meet the spin restriction condition
    Ca_->copy(Ca);
    Cb_->copy(Cb);

    if (spin_restriction_ == IntegralSpinRestriction::Restricted) {
        if (not test_orbital_spin_restriction(Ca, Cb)) {
            Ca->print();
            Cb->print();
            auto msg = "CustomIntegrals::update_orbitals was passed two different sets of orbitals"
                       "\n  but the integral object assumes restricted orbitals";
            throw std::runtime_error(msg);
        }
    }

    // 2. Re-transform the integrals
    aptei_idx_ = nmo_;
    local_timer int_timer;
    outfile->Printf("\n  Integrals are about to be updated.");
    transform_one_electron_integrals();
    transform_two_electron_integrals();
    gather_integrals();
    freeze_core_orbitals();
    outfile->Printf("\n  Integrals update took %9.3f s.", int_timer.get());
}

// void CustomIntegrals::resort_integrals_after_freezing() {}

//    // Read the integrals from a file
//    std::string filename("INTDUMP");
//    std::ifstream file(filename);

//    if (not file.is_open()) {
//    }
//    std::string str((std::istreambuf_iterator<char>(file)),
//    std::istreambuf_iterator<char>());

//    std::vector<std::string> lines = split_string(str, "\n");

//    std::string open_tag("&FCI");
//    std::string close_tag("&END");

//    int nelec = 0;
//    int norb = 0;
//    int ms2 = 0;
//    std::vector<int> orbsym;
//    std::vector<double> two_electron_integrals_chemist;

//    bool parsing_section = false;
//    for (const auto& line : lines) {
//        outfile->Printf("\n%s", line.c_str());
//        if (line.find(close_tag) != std::string::npos) {
//            parsing_section = false;
//            // now we know how many orbitals are there and we can allocate memory for the
//            one- and
//            // two-electron integrals
//            custom_integrals_allocate(norb, orbsym);
//            two_electron_integrals_chemist.assign(num_tei_, 0.0);
//        } else if (line.find(open_tag) != std::string::npos) {
//            parsing_section = true;
//        } else {
//            if (parsing_section) {
//                std::vector<std::string> split_line = split_string(line, "=");
//                if (split_line[0] == "NORB") {
//                    split_line[1].pop_back();
//                    norb = stoi(split_line[1]);
//                }
//                if (split_line[0] == "NELEC") {
//                    split_line[1].pop_back();
//                    nelec = stoi(split_line[1]);
//                }
//                if (split_line[0] == "MS2") {
//                    split_line[1].pop_back();
//                    ms2 = stoi(split_line[1]);
//                }
//                if (split_line[0] == "ORBSYM") {
//                    split_line[1].pop_back();
//                    std::vector<std::string> vals = split_string(split_line[1], ",");
//                    for (const auto& val : vals) {
//                        orbsym.push_back(stoi(val));
//                    }
//                }
//            } else {
//                if (line.size() > 10) {
//                    std::vector<std::string> split_line = split_string(line, " ");
//                    double integral = stoi(split_line[0]);
//                    int p = stoi(split_line[1]);
//                    int q = stoi(split_line[2]);
//                    int r = stoi(split_line[3]);
//                    int s = stoi(split_line[4]);

//                    if (q == 0) {
//                        // orbital energies, skip them
//                    } else if ((r == 0) and (s == 0)) {
//                        // one-electron integrals
//                        full_one_electron_integrals_a_[(p - 1) * aptei_idx_ + q - 1] =
//                        integral; full_one_electron_integrals_b_[(p - 1) * aptei_idx_ + q
//                        - 1] = integral; full_one_electron_integrals_a_[(q - 1) *
//                        aptei_idx_ + p - 1] = integral; full_one_electron_integrals_b_[(q
//                        - 1) * aptei_idx_ + p - 1] = integral;
//                        one_electron_integrals_a_[(p - 1) * aptei_idx_ + q - 1] =
//                        integral; one_electron_integrals_b_[(p - 1) * aptei_idx_ + q - 1]
//                        = integral; one_electron_integrals_a_[(q - 1) * aptei_idx_ + p -
//                        1] = integral; one_electron_integrals_b_[(q - 1) * aptei_idx_ + p
//                        - 1] = integral;
//                    } else {
//                        // two-electron integrals
//                        two_electron_integrals_chemist[four(p, q, r, s)] = integral;
//                    }
//                }
//            }
//        }
//    }

//    // Store the integrals
//    for (size_t p = 0; p < nmo_; ++p) {
//        for (size_t q = 0; q < nmo_; ++q) {
//            for (size_t r = 0; r < nmo_; ++r) {
//                for (size_t s = 0; s < nmo_; ++s) {
//                    // <pq||rs> = <pq|rs> - <pq|sr> = (pr|qs) - (ps|qr)
//                    double direct = two_electron_integrals_chemist[INDEX4(p, r, q, s)];
//                    double exchange = two_electron_integrals_chemist[INDEX4(p, s, q, r)];
//                    size_t index = aptei_index(p, q, r, s);
//                    aphys_tei_aa[index] = direct - exchange;
//                    aphys_tei_ab[index] = direct;
//                    aphys_tei_bb[index] = direct - exchange;
//                }
//            }
//        }
//    }

//    std::string s(std::istreambuf_iterator<char>(file >> std::skipws),
//                   std::istreambuf_iterator<char>());

//    std::copy(std::istream_iterator<std::string>(file),
//              std::istream_iterator<std::string>(),
//              std::back_inserter(lines));

//    outfile->Printf("%s",s.c_str());

//    for (size_t p = 0; p < nmo_; ++p) {
//        for (size_t q = 0; q < nmo_; ++q) {
//            one_electron_integrals_a_[p * nmo_ + q] = 0.0;
//            one_electron_integrals_b_[p * nmo_ + q] = 0.0;
//        }
//    }

//    for (size_t pqrs = 0; pqrs < num_aptei_; ++pqrs)
//        aphys_tei_aa[pqrs] = 0.0;
//    for (size_t pqrs = 0; pqrs < num_aptei_; ++pqrs)
//        aphys_tei_ab[pqrs] = 0.0;
//    for (size_t pqrs = 0; pqrs < num_aptei_; ++pqrs)
//        aphys_tei_bb[pqrs] = 0.0;

//    // Store the integrals
//    for (size_t p = 0; p < nmo_; ++p) {
//        for (size_t q = 0; q < nmo_; ++q) {
//            for (size_t r = 0; r < nmo_; ++r) {
//                for (size_t s = 0; s < nmo_; ++s) {
//                    // <pq||rs> = <pq|rs> - <pq|sr> = (pr|qs) - (ps|qr)
//                    double direct = 0.0;
//                    double exchange = 0.0;
//                    size_t index = aptei_index(p, q, r, s);
//                    aphys_tei_aa[index] = direct - exchange;
//                    aphys_tei_ab[index] = direct;
//                    aphys_tei_bb[index] = direct - exchange;
//                }
//            }
//        }
//    }
} // namespace forte
