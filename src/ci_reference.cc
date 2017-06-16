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

#include "psi4/libmints/molecule.h"
#include "psi4/libmints/wavefunction.h"
#include "psi4/liboptions/liboptions.h"

#include "ci_reference.h"

#include <algorithm>

namespace psi {
namespace forte {

CI_Reference::CI_Reference(std::shared_ptr<Wavefunction> wfn, Options& options,
                           std::shared_ptr<MOSpaceInfo> mo_space_info, STLBitsetDeterminant det,
                           int multiplicity, double twice_ms, int symmetry)
    : wfn_(wfn), mo_space_info_(mo_space_info) {
    // Get the mutlilicity and twice M_s
    multiplicity_ = multiplicity;
    twice_ms_ = twice_ms;

    // State symmetry
    root_sym_ = symmetry;

    // Number of irreps
    nirrep_ = wfn_->nirrep();

    // Double and singly occupied MOs
    Dimension doccpi = wfn_->doccpi();
    Dimension soccpi = wfn_->soccpi();

    // Frozen DOCC + RDOCC
    size_t ninact = mo_space_info_->size("INACTIVE_DOCC");
    frzcpi_ = mo_space_info_->get_dimension("INACTIVE_DOCC");

    // Symmetry of each MO
    mo_symmetry_ = mo_space_info_->symmetry("ACTIVE");

    // Size of total active space
    nactpi_ = mo_space_info_->get_dimension("ACTIVE");

    // Size of subspace
    subspace_size_ = options.get_int("ACTIVE_GUESS_SIZE");

    // First determine number of alpha and beta electrons
    // Assume twice_ms =( Na - Nb )
    int nel = 0;
    for (int h = 0; h < nirrep_; ++h) {
        nel += 2 * doccpi[h] + soccpi[h];
    }

    nel -= 2 * ninact;

    nalpha_ = 0.5 * (nel + twice_ms_);
    nbeta_ = nel - nalpha_;

    outfile->Printf("\n  Number of active orbitals: %d", STLBitsetDeterminant::nmo_);
    outfile->Printf("\n  Number of active alpha electrons: %d", nalpha_);
    outfile->Printf("\n  Number of active beta electrons: %d", nbeta_);
    outfile->Printf("\n  Maximum reference space size: %zu", subspace_size_);
}

CI_Reference::~CI_Reference() {}

void CI_Reference::build_reference(std::vector<STLBitsetDeterminant>& ref_space) {
    int nact = mo_space_info_->size("ACTIVE");

    // Get the active mos
    auto active_mos = sym_labeled_orbitals("RHF");

    // The active subspace
    int na = twice_ms_;

    // The frozen subspace
    int nf = std::min(nalpha_, nbeta_);

    // Control when to add mos
    bool add_mo = true;
    bool reverse = false;
    bool nf_zero = (nf == 0) ? true : false;

    while (add_mo) {
        ref_space.clear();
        // Indices of subspace
        std::vector<int> active_subspace;

        // Compute number of subspace electrons
        int alpha_el = nalpha_ - nf;
        int beta_el = nbeta_ - nf;

        for (int i = nf, max_i = nf + na; i < max_i; ++i) {
            active_subspace.push_back(std::get<2>(active_mos[i]));
        }

        // Compute subspace vectors
        std::vector<bool> tmp_det_a(na, false);
        std::vector<bool> tmp_det_b(na, false);
        for (int i = 0; i < alpha_el; ++i) {
            tmp_det_a[i] = true;
        }
        for (int i = 0; i < beta_el; ++i) {
            tmp_det_b[i] = true;
        }
        // Make sure we start with the first permutation
        std::sort(begin(tmp_det_a), end(tmp_det_a));
        std::sort(begin(tmp_det_b), end(tmp_det_b));

        // Build the core det
        STLBitsetDeterminant core_det;
        for (int i = 0; i < nf; ++i) {
            core_det.set_alfa_bit(std::get<2>(active_mos[i]), true);
            core_det.set_beta_bit(std::get<2>(active_mos[i]), true);
        }
        // Generate all permutations, add the correct ones
        do {
            do {
                // Build determinant
                STLBitsetDeterminant det(core_det);
                int sym = 0;
                for (int p = 0; p < na; ++p) {
                    det.set_alfa_bit(active_subspace[p], tmp_det_a[p]);
                    det.set_beta_bit(active_subspace[p], tmp_det_b[p]);
                    if (tmp_det_a[p]) {
                        sym ^= mo_symmetry_[active_subspace[p]];
                    }
                    if (tmp_det_b[p]) {
                        sym ^= mo_symmetry_[active_subspace[p]];
                    }
                }
                // Check symmetry
                if (sym == root_sym_) {
                    ref_space.push_back(det);
                    //                    det.print();
                }

            } while (std::next_permutation(tmp_det_b.begin(), tmp_det_b.begin() + na));
        } while (std::next_permutation(tmp_det_a.begin(), tmp_det_a.begin() + na));
        //        outfile->Printf("\n na: %d, nf: %d, ref size: %zu", na, nf, ref_space.size());

        if (reverse and (ref_space.size() < subspace_size_)) {
            add_mo = false;
        }

        if (ref_space.size() < subspace_size_) {

            if (na == nact) {
                add_mo = false;
            }

            na += 2;
            nf -= 1;

            // No negative indices
            if (nf < 0) {
                nf = 0;
                nf_zero = true;
            }

            while ((na + nf) > nact) {
                na -= 1;
            }

        } else if (ref_space.size() > subspace_size_) {
            na -= 1;

            if (!nf_zero) {
                nf += 1;
            }
            reverse = true;
            //            outfile->Printf("  reverse = true");
        } else {
            add_mo = false;
        }
    }

    if (ref_space.size() == 0) {
        throw PSIEXCEPTION("Unable to generate CASCI space. Try increasing ACTIVE_GUESS_SIZE");
    }

    outfile->Printf("\n  Number of reference determinants: %zu", ref_space.size());
    outfile->Printf("\n  Reference generated from %d MOs", na);
}

std::vector<std::tuple<double, int, int>> CI_Reference::sym_labeled_orbitals(std::string type) {
    int nact = mo_space_info_->size("ACTIVE");

    std::vector<std::tuple<double, int, int>> labeled_orb;

    std::shared_ptr<Vector> epsilon_a = wfn_->epsilon_a();

    if (type == "RHF" or type == "ROHF" or type == "ALFA") {

        // Create a vector of orbital energy and index pairs
        std::vector<std::pair<double, int>> orb_e;
        int cumidx = 0;
        for (int h = 0; h < nirrep_; ++h) {
            for (int a = 0; a < nactpi_[h]; ++a) {
                orb_e.push_back(std::make_pair(epsilon_a->get(h, frzcpi_[h] + a), a + cumidx));
            }
            cumidx += nactpi_[h];
        }

        // Create a vector that stores the orbital energy, symmetry, and idx
        for (size_t a = 0; a < nact; ++a) {
            labeled_orb.push_back(
                std::make_tuple(orb_e[a].first, mo_symmetry_[a], orb_e[a].second));
        }
        // Order by energy, low to high
        std::sort(labeled_orb.begin(), labeled_orb.end());
    }
    if (type == "BETA") {
        // Create a vector of orbital energies and index pairs
        std::shared_ptr<Vector> epsilon_b = wfn_->epsilon_b();
        std::vector<std::pair<double, int>> orb_e;
        int cumidx = 0;
        for (int h = 0; h < nirrep_; ++h) {
            for (size_t a = 0, max = nactpi_[h]; a < max; ++a) {
                orb_e.push_back(std::make_pair(epsilon_b->get(h, frzcpi_[h] + a), a + cumidx));
            }
            cumidx += nactpi_[h];
        }

        // Create a vector that stores the orbital energy, sym, and idx
        for (size_t a = 0; a < nact; ++a) {
            labeled_orb.push_back(
                std::make_tuple(orb_e[a].first, mo_symmetry_[a], orb_e[a].second));
        }
        std::sort(labeled_orb.begin(), labeled_orb.end());
    }
    return labeled_orb;
}
}
} // End Namespaces
