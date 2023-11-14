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
#include "fci_vector.h"
#include "fci_string_lists.h"
#include "fci_string_address.h"

using namespace psi;

namespace forte {

std::shared_ptr<psi::Matrix> FCIVector::CR;
std::shared_ptr<psi::Matrix> FCIVector::CL;

double FCIVector::hdiag_timer = 0.0;
double FCIVector::h1_aa_timer = 0.0;
double FCIVector::h1_bb_timer = 0.0;
double FCIVector::h2_aaaa_timer = 0.0;
double FCIVector::h2_aabb_timer = 0.0;
double FCIVector::h2_bbbb_timer = 0.0;

void FCIVector::allocate_temp_space(std::shared_ptr<FCIStringLists> lists_, int print_) {
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
            outfile->Printf("\n  Allocating memory for the Hamiltonian algorithm. "
                            "Size: 2 x %zu x %zu.   Memory: %8.6f GB",
                            max_size, max_size, to_gb(2 * max_size * max_size));
    }
}

void FCIVector::release_temp_space() {}

std::shared_ptr<psi::Matrix> FCIVector::get_CR() { return CR; }
std::shared_ptr<psi::Matrix> FCIVector::get_CL() { return CL; }

FCIVector::FCIVector(std::shared_ptr<FCIStringLists> lists, size_t symmetry)
    : symmetry_(symmetry), lists_(lists), alfa_address_(lists_->alfa_address()),
      beta_address_(lists_->beta_address()) {
    startup();
}

void FCIVector::startup() {

    nirrep_ = lists_->nirrep();
    ncmo_ = lists_->ncmo();
    cmopi_ = lists_->cmopi();
    cmopi_offset_ = lists_->cmopi_offset();

    ndet_ = 0;
    for (int alfa_sym = 0; alfa_sym < nirrep_; ++alfa_sym) {
        int beta_sym = alfa_sym ^ symmetry_;
        size_t detpi = alfa_address_->strpcls(alfa_sym) * beta_address_->strpcls(beta_sym);
        ndet_ += detpi;
        detpi_.push_back(detpi);
    }

    // Allocate the symmetry blocks of the wave function
    for (int alfa_sym = 0; alfa_sym < nirrep_; ++alfa_sym) {
        int beta_sym = alfa_sym ^ symmetry_;
        //    outfile->Printf("\n\n  Block %d: allocate %d *
        //    %d",alfa_sym,(int)alfa_address_->strpcls(alfa_sym),(int)beta_address_->strpcls(beta_sym));
        C_.push_back(std::make_shared<psi::Matrix>("C", alfa_address_->strpcls(alfa_sym),
                                                   beta_address_->strpcls(beta_sym)));
    }
}

size_t FCIVector::symmetry() const { return symmetry_; }

size_t FCIVector::nirrep() const { return nirrep_; }

size_t FCIVector::ncmo() const { return ncmo_; }

size_t FCIVector::size() const { return ndet_; }

const std::vector<size_t>& FCIVector::detpi() const { return detpi_; }

psi::Dimension FCIVector::cmopi() const { return cmopi_; }

const std::vector<size_t>& FCIVector::cmopi_offset() const { return cmopi_offset_; }

const std::shared_ptr<FCIStringLists>& FCIVector::lists() const { return lists_; }

void FCIVector::copy(FCIVector& wfn) {
    for (int alfa_sym = 0; alfa_sym < nirrep_; ++alfa_sym) {
        C_[alfa_sym]->copy(wfn.C_[alfa_sym]);
    }
}

void FCIVector::copy(std::shared_ptr<psi::Vector> vec) {
    size_t I = 0;
    for (int alfa_sym = 0; alfa_sym < nirrep_; ++alfa_sym) {
        int beta_sym = alfa_sym ^ symmetry_;
        size_t maxIa = alfa_address_->strpcls(alfa_sym);
        size_t maxIb = beta_address_->strpcls(beta_sym);
        double** C_ha = C_[alfa_sym]->pointer();
        for (size_t Ia = 0; Ia < maxIa; ++Ia) {
            for (size_t Ib = 0; Ib < maxIb; ++Ib) {
                C_ha[Ia][Ib] = vec->get(I);
                I += 1;
            }
        }
    }
}

void FCIVector::copy_to(std::shared_ptr<psi::Vector> vec) {
    size_t I = 0;
    for (int alfa_sym = 0; alfa_sym < nirrep_; ++alfa_sym) {
        int beta_sym = alfa_sym ^ symmetry_;
        size_t maxIa = alfa_address_->strpcls(alfa_sym);
        size_t maxIb = beta_address_->strpcls(beta_sym);
        double** C_ha = C_[alfa_sym]->pointer();
        for (size_t Ia = 0; Ia < maxIa; ++Ia) {
            for (size_t Ib = 0; Ib < maxIb; ++Ib) {
                vec->set(I, C_ha[Ia][Ib]);
                I += 1;
            }
        }
    }
}

void FCIVector::set(std::vector<std::tuple<size_t, size_t, size_t, double>>& sparse_vec) {
    zero();
    for (const auto& [h, Ia, Ib, C] : sparse_vec) {
        C_[h]->set(Ia, Ib, C);
    }
}

void FCIVector::normalize() {
    double factor = norm(2.0);
    for (int alfa_sym = 0; alfa_sym < nirrep_; ++alfa_sym) {
        C_[alfa_sym]->scale(1.0 / factor);
    }
}

void FCIVector::zero() {
    for (auto C_h : C_) {
        C_h->zero();
    }
}

void FCIVector::print_natural_orbitals(std::shared_ptr<MOSpaceInfo> mo_space_info,
                                       std::shared_ptr<RDMs> rdms) {
    print_h2("Natural Orbitals");
    psi::Dimension active_dim = mo_space_info->dimension("ACTIVE");
    auto nfdocc = mo_space_info->size("FROZEN_DOCC");

    auto G1 = rdms->SF_G1();
    auto& G1_data = G1.data();

    auto opdm = std::make_shared<psi::Matrix>("OPDM", active_dim, active_dim);

    int offset = 0;
    for (int h = 0; h < nirrep_; h++) {
        for (int u = 0; u < active_dim[h]; u++) {
            for (int v = 0; v < active_dim[h]; v++) {
                double gamma_uv = G1_data[(u + offset) * ncmo_ + v + offset];
                opdm->set(h, u, v, gamma_uv);
            }
        }
        offset += active_dim[h];
    }

    auto OCC = std::make_shared<psi::Vector>("Occupation numbers", active_dim);
    auto NO = std::make_shared<psi::Matrix>("MO -> NO transformation", active_dim, active_dim);

    opdm->diagonalize(NO, OCC, descending);
    std::vector<std::pair<double, std::pair<int, int>>> vec_irrep_occupation;
    for (int h = 0; h < nirrep_; h++) {
        for (int u = 0; u < active_dim[h]; u++) {
            auto irrep_occ = std::make_pair(OCC->get(h, u), std::make_pair(h, u + 1));
            vec_irrep_occupation.push_back(irrep_occ);
        }
    }
    std::sort(vec_irrep_occupation.begin(), vec_irrep_occupation.end(),
              std::greater<std::pair<double, std::pair<int, int>>>());

    size_t count = 0;
    outfile->Printf("\n    ");
    for (auto vec : vec_irrep_occupation) {
        outfile->Printf(" %4d%-4s%11.6f  ", vec.second.second + nfdocc,
                        mo_space_info->irrep_label(vec.second.first).c_str(), vec.first);
        if (count++ % 3 == 2 && count != vec_irrep_occupation.size())
            outfile->Printf("\n    ");
    }
    outfile->Printf("\n");
}

double** gather_C_block(FCIVector& C, std::shared_ptr<psi::Matrix> M, bool alfa,
                        std::shared_ptr<FCIStringAddress> alfa_address,
                        std::shared_ptr<FCIStringAddress> beta_address, int ha, int hb, bool zero) {
    // if alfa is true just return the pointer to the block
    auto c = C.C(ha)->pointer();
    if (alfa) {
        if (zero)
            C.C(ha)->zero();
        return c;
    }
    // if alfa is false
    size_t maxIa = alfa_address->strpcls(ha);
    size_t maxIb = beta_address->strpcls(hb);
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

void scatter_C_block(FCIVector& C, double** m, bool alfa,
                     std::shared_ptr<FCIStringAddress> alfa_address,
                     std::shared_ptr<FCIStringAddress> beta_address, int ha, int hb) {
    if (!alfa) {
        size_t maxIa = alfa_address->strpcls(ha);
        size_t maxIb = beta_address->strpcls(hb);

        double** c = C.C(ha)->pointer();
        // Add m transposed to C
        for (size_t Ia = 0; Ia < maxIa; ++Ia)
            for (size_t Ib = 0; Ib < maxIb; ++Ib)
                c[Ia][Ib] += m[Ib][Ia];
    }
}

double FCIVector::norm(double power) {
    double norm = 0.0;
    for (int alfa_sym = 0; alfa_sym < nirrep_; ++alfa_sym) {
        int beta_sym = alfa_sym ^ symmetry_;
        size_t maxIa = alfa_address_->strpcls(alfa_sym);
        size_t maxIb = beta_address_->strpcls(beta_sym);
        double** C_ha = C_[alfa_sym]->pointer();
        for (size_t Ia = 0; Ia < maxIa; ++Ia) {
            for (size_t Ib = 0; Ib < maxIb; ++Ib) {
                norm += std::pow(std::fabs(C_ha[Ia][Ib]), power);
            }
        }
    }
    return std::pow(norm, 1.0 / power);
}

/**
 * Compute the dot product with another wave function
 */
double FCIVector::dot(FCIVector& wfn) {
    double dot = 0.0;
    for (int alfa_sym = 0; alfa_sym < nirrep_; ++alfa_sym) {
        dot += C_[alfa_sym]->vector_dot(wfn.C_[alfa_sym]);
    }
    return (dot);
}
double FCIVector::dot(std::shared_ptr<FCIVector>& wfn) {
    double dot = 0.0;
    for (int alfa_sym = 0; alfa_sym < nirrep_; ++alfa_sym) {
        dot += C_[alfa_sym]->vector_dot(wfn->C_[alfa_sym]);
    }
    return (dot);
}

/**
 * Print the non-zero contributions to the wave function
 */
void FCIVector::print() {
    // print the non-zero elements of the wave function
    size_t det = 0;
    for (int alfa_sym = 0; alfa_sym < nirrep_; ++alfa_sym) {
        int beta_sym = alfa_sym ^ symmetry_;
        double** C_ha = C_[alfa_sym]->pointer();
        for (size_t Ia = 0; Ia < alfa_address_->strpcls(alfa_sym); ++Ia) {
            for (size_t Ib = 0; Ib < beta_address_->strpcls(beta_sym); ++Ib) {
                if (std::fabs(C_ha[Ia][Ib]) > 1.0e-9) {
                    outfile->Printf("\n  %15.9f [%1d][%2d][%2d] (%d)", C_ha[Ia][Ib], alfa_sym,
                                    static_cast<int>(Ia), static_cast<int>(Ib),
                                    static_cast<int>(det));
                }
                ++det;
            }
        }
    }
}

} // namespace forte
