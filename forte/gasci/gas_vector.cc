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

#include "psi4/libpsi4util/process.h"
#include "psi4/libmints/matrix.h"
#include "psi4/libmints/molecule.h"

#include "helpers/helpers.h"
#include "helpers/printing.h"
#include "base_classes/mo_space_info.h"

#include "gas_vector.h"
#include "gas_string_lists.h"
#include "gas_string_address.h"

namespace forte {

// Global debug flag
bool debug_gas_vector = true;

// Wrapper function
template <typename Func> void debug(Func func) {
    if (debug_gas_vector) {
        func();
    }
}

std::shared_ptr<psi::Matrix> GASVector::CR;
std::shared_ptr<psi::Matrix> GASVector::CL;

double GASVector::hdiag_timer = 0.0;
double GASVector::h1_aa_timer = 0.0;
double GASVector::h1_bb_timer = 0.0;
double GASVector::h2_aaaa_timer = 0.0;
double GASVector::h2_aabb_timer = 0.0;
double GASVector::h2_bbbb_timer = 0.0;

void GASVector::allocate_temp_space(std::shared_ptr<GASStringLists> lists_, int print_) {
    size_t nirreps = lists_->nirrep();

    // if CR is already allocated (e.g., because we computed several roots) make sure
    // we do not allocate a matrix of smaller size. So let's find out the size of the current CR
    size_t current_size = CR ? CR->rowdim() : 0;

    // Find the largest size of the symmetry blocks
    size_t max_size = 0;
    for (size_t Ia_sym = 0; Ia_sym < nirreps; ++Ia_sym) {
        max_size = std::max(max_size, lists_->alfa_address()->strpcls(Ia_sym));
    }
    for (size_t Ib_sym = 0; Ib_sym < nirreps; ++Ib_sym) {
        max_size = std::max(max_size, lists_->beta_address()->strpcls(Ib_sym));
    }

    // Allocate the temporary arrays CR and CL with the largest block size
    if (max_size > current_size) {
        CR = std::make_shared<psi::Matrix>("CR", max_size, max_size);
        CL = std::make_shared<psi::Matrix>("CL", max_size, max_size);
        if (print_)
            psi::outfile->Printf("\n  Allocating memory for the Hamiltonian algorithm. "
                                 "Size: 2 x %zu x %zu.   Memory: %8.6f GB",
                                 max_size, max_size, to_gb(2 * max_size * max_size));
    }
}

void GASVector::release_temp_space() {}

std::shared_ptr<psi::Matrix> GASVector::get_CR() { return CR; }
std::shared_ptr<psi::Matrix> GASVector::get_CL() { return CL; }

GASVector::GASVector(std::shared_ptr<GASStringLists> lists)
    : symmetry_(lists->symmetry()), lists_(lists), alfa_address_(lists_->alfa_address()),
      beta_address_(lists_->beta_address()) {
    startup();
}

void GASVector::startup() {
    nirrep_ = lists_->nirrep();
    ncmo_ = lists_->ncmo();
    cmopi_ = lists_->cmopi();
    cmopi_offset_ = lists_->cmopi_offset();

    psi::outfile->Printf("\n  symmetry: %d", symmetry_);
    psi::outfile->Printf("\n  nirrep: %d", nirrep_);
    psi::outfile->Printf("\n  ncmo: %d", ncmo_);

    ndet_ = 0;
    for (const auto& [_, class_Ia, class_Ib] : lists_->determinant_classes()) {
        auto size_alfa = alfa_address_->strpcls(class_Ia);
        auto size_beta = beta_address_->strpcls(class_Ib);
        auto detpcls = size_alfa * size_beta;
        ndet_ += detpcls;
        detpcls_.push_back(detpcls);
    }

    psi::outfile->Printf("\n\n  Number of determinants: %d", ndet_);

    // Allocate the wave function
    for (const auto& [_, class_Ia, class_Ib] : lists_->determinant_classes()) {
        C_.push_back(std::make_shared<psi::Matrix>("C", alfa_address_->strpcls(class_Ia),
                                                   beta_address_->strpcls(class_Ib)));
    }
}

size_t GASVector::symmetry() const { return symmetry_; }

size_t GASVector::nirrep() const { return nirrep_; }

size_t GASVector::ncmo() const { return ncmo_; }

size_t GASVector::size() const { return ndet_; }

const std::vector<size_t>& GASVector::detpi() const { return detpi_; }

psi::Dimension GASVector::cmopi() const { return cmopi_; }

const std::vector<size_t>& GASVector::cmopi_offset() const { return cmopi_offset_; }

const std::shared_ptr<GASStringLists>& GASVector::lists() const { return lists_; }

void GASVector::print(double threshold) const {
    const_for_each_element([&](const size_t& n, const int& class_Ia, const int& class_Ib,
                               const size_t& Ia, const size_t& Ib, const double& c) {
        if (std::fabs(c) >= threshold) {
            Determinant I(lists_->alfa_str(class_Ia, Ia), lists_->beta_str(class_Ib, Ib));
            psi::outfile->Printf("\n  %+15.9f %s [%2d][%2d][%2d] (%d)", c,
                                 str(I, lists_->ncmo()).c_str(), class_Ia, static_cast<int>(Ia),
                                 static_cast<int>(Ib), static_cast<int>(n));
        }
    });
}

std::shared_ptr<StateVector> GASVector::as_state_vector() const {
    det_hash<double> state_vector;
    const_for_each_element([&](const size_t& /*n*/, const int& class_Ia, const int& class_Ib,
                               const size_t& Ia, const size_t& Ib, const double& c) {
        if (std::fabs(c) > 1.0e-12) {
            Determinant I(lists_->alfa_str(class_Ia, Ia), lists_->beta_str(class_Ib, Ib));
            state_vector[I] = c;
        }
    });
    return std::make_shared<StateVector>(state_vector);
}

void GASVector::copy(GASVector& wfn) {
    for (const auto& [n, _1, _2] : lists_->determinant_classes()) {
        C_[n]->copy(wfn.C_[n]);
    }
}

void GASVector::copy(std::shared_ptr<psi::Vector> vec) {
    for_each_index_element([&](const size_t& I, double& c) { c = vec->get(I); });
}

void GASVector::copy_to(std::shared_ptr<psi::Vector> vec) {
    const_for_each_index_element([&](const size_t& I, const double& c) { vec->set(I, c); });
}

void GASVector::set_to(double value) {
    for_each_index_element([&](const size_t& /*I*/, double& c) { c = value; });
}

void GASVector::set(std::vector<std::tuple<size_t, size_t, size_t, double>>& sparse_vec) {
    zero();
    for (const auto& [h, Ia, Ib, C] : sparse_vec) {
        C_[h]->set(Ia, Ib, C);
    }
}

double GASVector::dot(const GASVector& wfn) const {
    double dot = 0.0;
    for (const auto& [n, _1, _2] : lists_->determinant_classes()) {
        dot += C_[n]->vector_dot(wfn.C_[n]);
    }
    return (dot);
}

double GASVector::norm(double power) {
    double norm = dot(*this);
    return std::pow(norm, 1.0 / power);
}

void GASVector::normalize() {
    double factor = norm(2.0);
    for (auto& c : C_)
        c->scale(1.0 / factor);
}

void GASVector::zero() {
    for (auto& c : C_)
        c->zero();
}

// void GASVector::print_natural_orbitals(std::shared_ptr<MOSpaceInfo> mo_space_info,
//                                        std::shared_ptr<RDMs> rdms) {
//     print_h2("Natural Orbitals");
//     psi::Dimension active_dim = mo_space_info->dimension("ACTIVE");
//     auto nfdocc = mo_space_info->size("FROZEN_DOCC");

//     auto G1 = rdms->SF_G1();
//     auto& G1_data = G1.data();

//     auto opdm = std::make_shared<psi::Matrix>("OPDM", active_dim, active_dim);

//     int offset = 0;
//     for (int h = 0; h < nirrep_; h++) {
//         for (int u = 0; u < active_dim[h]; u++) {
//             for (int v = 0; v < active_dim[h]; v++) {
//                 double gamma_uv = G1_data[(u + offset) * ncmo_ + v + offset];
//                 opdm->set(h, u, v, gamma_uv);
//             }
//         }
//         offset += active_dim[h];
//     }

//     auto OCC = std::make_shared<psi::Vector>("Occupation numbers", active_dim);
//     auto NO = std::make_shared<psi::Matrix>("MO -> NO transformation", active_dim,
//     active_dim);

//     opdm->diagonalize(NO, OCC, psi::descending);
//     std::vector<std::pair<double, std::pair<int, int>>> vec_irrep_occupation;
//     for (int h = 0; h < nirrep_; h++) {
//         for (int u = 0; u < active_dim[h]; u++) {
//             auto irrep_occ = std::make_pair(OCC->get(h, u), std::make_pair(h, u + 1));
//             vec_irrep_occupation.push_back(irrep_occ);
//         }
//     }
//     std::sort(vec_irrep_occupation.begin(), vec_irrep_occupation.end(),
//               std::greater<std::pair<double, std::pair<int, int>>>());

//     size_t count = 0;
//     psi::outfile->Printf("\n    ");
//     for (auto vec : vec_irrep_occupation) {
//         psi::outfile->Printf(" %4d%-4s%11.6f  ", vec.second.second + nfdocc,
//                              mo_space_info->irrep_label(vec.second.first).c_str(),
//                              vec.first);
//         if (count++ % 3 == 2 && count != vec_irrep_occupation.size())
//             psi::outfile->Printf("\n    ");
//     }
//     psi::outfile->Printf("\n");
// }

double** GASVector::gather_C_block(std::shared_ptr<psi::Matrix> M, bool alfa,
                                   std::shared_ptr<StringAddress> alfa_address,
                                   std::shared_ptr<StringAddress> beta_address, int class_Ia,
                                   int class_Ib, bool zero) {
    // if alfa is true just return the pointer to the block
    int block_idx = lists_->string_class()->block_index(class_Ia, class_Ib);
    auto c = C(block_idx)->pointer();
    if (alfa) {
        if (zero)
            C(block_idx)->zero();
        return c;
    }
    // if alfa is false
    size_t maxIa = alfa_address->strpcls(class_Ia);
    size_t maxIb = beta_address->strpcls(class_Ib);
    auto m = M->pointer();
    if (zero) {
        for (size_t Ib = 0; Ib < maxIb; ++Ib)
            for (size_t Ia = 0; Ia < maxIa; ++Ia)
                m[Ib][Ia] = 0.0;
    } else {
        for (size_t Ia = 0; Ia < maxIa; ++Ia)
            for (size_t Ib = 0; Ib < maxIb; ++Ib)
                m[Ib][Ia] = c[Ia][Ib];
    }
    return m;
}

void GASVector::scatter_C_block(double** m, bool alfa, std::shared_ptr<StringAddress> alfa_address,
                                std::shared_ptr<StringAddress> beta_address, int class_Ia,
                                int class_Ib) {
    if (!alfa) {
        size_t maxIa = alfa_address->strpcls(class_Ia);
        size_t maxIb = beta_address->strpcls(class_Ib);

        int block_idx = lists_->string_class()->block_index(class_Ia, class_Ib);
        auto c = C(block_idx)->pointer();
        // Add m transposed to C
        for (size_t Ia = 0; Ia < maxIa; ++Ia)
            for (size_t Ib = 0; Ib < maxIb; ++Ib)
                c[Ia][Ib] += m[Ib][Ia];
    }
}

} // namespace forte
