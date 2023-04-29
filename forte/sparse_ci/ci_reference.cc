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

#include <numeric>

#include "psi4/psi4-dec.h"
#include "psi4/libpsi4util/PsiOutStream.h"
#include "psi4/libmints/vector.h"

#include "forte-def.h"
#include "ci_reference.h"
#include "base_classes/forte_options.h"
#include "helpers/helpers.h"
#include "helpers/printing.h"
#include "helpers/timer.h"

#include <algorithm>

using namespace psi;

namespace forte {

CI_Reference::CI_Reference(std::shared_ptr<SCFInfo> scf_info, std::shared_ptr<ForteOptions> options,
                           std::shared_ptr<MOSpaceInfo> mo_space_info,
                           std::shared_ptr<ActiveSpaceIntegrals> fci_ints, int multiplicity,
                           double twice_ms, int symmetry, StateInfo state_info)
    : scf_info_(scf_info), state_info_(state_info), mo_space_info_(mo_space_info),
      fci_ints_(fci_ints) {
    // Get the mutlilicity and twice M_s
    multiplicity_ = multiplicity;
    twice_ms_ = twice_ms;

    // State symmetry
    root_sym_ = symmetry;

    // Options
    options_ = options;

    // Double and singly occupied MOs
    // ONLY works if we have a Psi4 wave function!!!
    psi::Dimension doccpi = scf_info_->doccpi();
    psi::Dimension soccpi = scf_info_->soccpi();

    // Number of irreps
    nirrep_ = mo_space_info_->nirrep();

    // Frozen DOCC + RDOCC
    size_t ninact = mo_space_info_->size("INACTIVE_DOCC");
    frzcpi_ = mo_space_info_->dimension("INACTIVE_DOCC");

    // Symmetry of each MO
    mo_symmetry_ = mo_space_info_->symmetry("ACTIVE");

    // Size of total active space
    nact_ = mo_space_info_->size("ACTIVE");
    nactpi_ = mo_space_info_->dimension("ACTIVE");

    // Size of subspace
    subspace_size_ = options->get_int("ACTIVE_GUESS_SIZE");

    // Reference type
    ref_type_ = options->get_str("ACTIVE_REF_TYPE");

    nalpha_ = state_info_.na() - ninact;
    nbeta_ = state_info_.nb() - ninact;

    outfile->Printf("\n  Number of active orbitals: %d", nact_);
    outfile->Printf("\n  Number of active alpha electrons: %d", nalpha_);
    outfile->Printf("\n  Number of active beta electrons: %d", nbeta_);
}

CI_Reference::~CI_Reference() {}

void CI_Reference::build_reference(DeterminantHashVec& ref_space) {
    if (ref_type_ == "CAS") {
        outfile->Printf("\n  Maximum reference space size: %zu", subspace_size_);
        build_cas_reference(ref_space);
    } else if (ref_type_ == "GAS") {
        // Complete GAS
        build_gas_reference(ref_space);
    } else if (ref_type_ == "GAS_SINGLE") {
        // Low(est) energy one in GAS
        build_gas_single(ref_space);
    } else {
        build_ci_reference(ref_space);
    }
}

void CI_Reference::print_gas_scf_epsilon() {
    print_h2("GAS Orbital Energies from SCF");
    outfile->Printf("\n    GAS        Energy  Index");
    outfile->Printf("\n    ------------------------");

    auto epsilon_a = scf_info_->epsilon_a();
    for (int gas = 0; gas < 6; ++gas) {
        std::string space_name = "GAS" + std::to_string(gas + 1);
        auto abs_mos = mo_space_info_->absolute_mo(space_name);
        if (abs_mos.size() == 0)
            continue;

        auto rel_mos = mo_space_info_->pos_in_space(space_name, "ACTIVE");
        for (size_t i = 0, size = abs_mos.size(); i < size; ++i) {
            outfile->Printf("\n    %2d %14.8f  %5zu", gas + 1, epsilon_a->get(abs_mos[i]),
                            rel_mos[i]);
        }
    }
    outfile->Printf("\n    ------------------------");
}

std::pair<std::map<std::vector<int>, std::vector<std::pair<size_t, size_t>>>,
          std::map<std::vector<int>, std::vector<std::pair<size_t, size_t>>>>
CI_Reference::gas_single_criterion() {
    // return the bool vector for whether it is possible to excite one electron
    // from one GAS to another

    // adding a all 0 vector to the existing possible electron occupations
    std::vector<int> zero_occ(12, 0);
    size_t gas_config_number = gas_electrons_.size();
    gas_electrons_.push_back(zero_occ);

    //    outfile->Printf("%d gas_num", gas_num_);
    // for each possible occupation (gas_config), when adding one electron
    // from gas_count_1 to gas_count_2, is the result new possble GAS occupation
    // still within gas_electrons?
    std::map<std::vector<int>, std::vector<std::pair<size_t, size_t>>> alpha_criterion;
    std::map<std::vector<int>, std::vector<std::pair<size_t, size_t>>> beta_criterion;
    for (size_t gas_config = 0; gas_config < gas_config_number; ++gas_config) {
        auto gas_configuration = gas_electrons_[gas_config];
        std::vector<std::pair<size_t, size_t>> tmp_vectora;
        std::vector<std::pair<size_t, size_t>> tmp_vectorb;
        for (size_t gas_count_1 = 0; gas_count_1 < gas_num_; ++gas_count_1) {
            for (size_t gas_count_2 = 0; gas_count_2 < gas_num_; ++gas_count_2) {
                std::vector<int> gas_configuration_copya = gas_configuration;
                gas_configuration_copya.at(2 * gas_count_1) -= 1;
                gas_configuration_copya.at(2 * gas_count_2) += 1;
                if (std::find(gas_electrons_.begin(), gas_electrons_.end(),
                              gas_configuration_copya) != gas_electrons_.end()) {
                    tmp_vectora.push_back(std::make_pair(gas_count_1, gas_count_2));
                }
                std::vector<int> gas_configuration_copyb = gas_configuration;
                gas_configuration_copyb.at(2 * gas_count_1 + 1) -= 1;
                gas_configuration_copyb.at(2 * gas_count_2 + 1) += 1;
                if (std::find(gas_electrons_.begin(), gas_electrons_.end(),
                              gas_configuration_copyb) != gas_electrons_.end()) {
                    tmp_vectorb.push_back(std::make_pair(gas_count_1, gas_count_2));
                }
            }
        }
        alpha_criterion[gas_configuration] = tmp_vectora;
        beta_criterion[gas_configuration] = tmp_vectorb;
    }
    gas_electrons_.pop_back();
    return std::make_pair(alpha_criterion, beta_criterion);
}

std::tuple<std::map<std::vector<int>, std::vector<std::tuple<size_t, size_t, size_t, size_t>>>,
           std::map<std::vector<int>, std::vector<std::tuple<size_t, size_t, size_t, size_t>>>,
           std::map<std::vector<int>, std::vector<std::tuple<size_t, size_t, size_t, size_t>>>>
CI_Reference::gas_double_criterion() {
    // return the bool vector for whether it is possible to excite two electrons
    // from two GAS to other two GAS

    std::vector<int> zero_occ(12, 0);
    size_t gas_config_number = gas_electrons_.size();
    gas_electrons_.push_back(zero_occ);

    // for each possible occupation (gas_config), when adding two electrons
    // from gas_count_1 and gas_count_2 to gas_count_3 and gas_count4,
    // is the resulted new  GAS occupation still within gas_electrons?
    std::map<std::vector<int>, std::vector<std::tuple<size_t, size_t, size_t, size_t>>>
        aa_criterion;
    std::map<std::vector<int>, std::vector<std::tuple<size_t, size_t, size_t, size_t>>>
        bb_criterion;

    std::map<std::vector<int>, std::vector<std::tuple<size_t, size_t, size_t, size_t>>>
        ab_criterion;
    for (size_t gas_config = 0; gas_config < gas_config_number; ++gas_config) {
        auto gas_configuration = gas_electrons_[gas_config];
        std::vector<std::tuple<size_t, size_t, size_t, size_t>> tmp_vectoraa;
        std::vector<std::tuple<size_t, size_t, size_t, size_t>> tmp_vectorbb;
        std::vector<std::tuple<size_t, size_t, size_t, size_t>> tmp_vectorab;
        for (size_t gas_count_1 = 0; gas_count_1 < gas_num_; ++gas_count_1) {
            for (size_t gas_count_2 = gas_count_1; gas_count_2 < gas_num_; ++gas_count_2) {
                for (size_t gas_count_3 = 0; gas_count_3 < gas_num_; ++gas_count_3) {
                    for (size_t gas_count_4 = gas_count_3; gas_count_4 < gas_num_; ++gas_count_4) {
                        std::vector<int> gas_configuration_copyaa = gas_configuration;
                        gas_configuration_copyaa.at(2 * gas_count_1) -= 1;
                        gas_configuration_copyaa.at(2 * gas_count_2) -= 1;
                        gas_configuration_copyaa.at(2 * gas_count_3) += 1;
                        gas_configuration_copyaa.at(2 * gas_count_4) += 1;
                        if (std::find(gas_electrons_.begin(), gas_electrons_.end(),
                                      gas_configuration_copyaa) != gas_electrons_.end()) {
                            tmp_vectoraa.push_back(std::make_tuple(gas_count_1, gas_count_2,
                                                                   gas_count_3, gas_count_4));
                        }
                        std::vector<int> gas_configuration_copybb = gas_configuration;
                        gas_configuration_copybb.at(2 * gas_count_1 + 1) -= 1;
                        gas_configuration_copybb.at(2 * gas_count_2 + 1) -= 1;
                        gas_configuration_copybb.at(2 * gas_count_3 + 1) += 1;
                        gas_configuration_copybb.at(2 * gas_count_4 + 1) += 1;
                        if (std::find(gas_electrons_.begin(), gas_electrons_.end(),
                                      gas_configuration_copybb) != gas_electrons_.end()) {
                            tmp_vectorbb.push_back(std::make_tuple(gas_count_1, gas_count_2,
                                                                   gas_count_3, gas_count_4));
                        }
                    }
                }
            }
        }
        for (size_t gas_count_1 = 0; gas_count_1 < gas_num_; ++gas_count_1) {
            for (size_t gas_count_2 = 0; gas_count_2 < gas_num_; ++gas_count_2) {
                for (size_t gas_count_3 = 0; gas_count_3 < gas_num_; ++gas_count_3) {
                    for (size_t gas_count_4 = 0; gas_count_4 < gas_num_; ++gas_count_4) {
                        std::vector<int> gas_configuration_copyab = gas_configuration;
                        gas_configuration_copyab.at(2 * gas_count_1) -= 1;
                        gas_configuration_copyab.at(2 * gas_count_2 + 1) -= 1;
                        gas_configuration_copyab.at(2 * gas_count_3) += 1;
                        gas_configuration_copyab.at(2 * gas_count_4 + 1) += 1;
                        if (std::find(gas_electrons_.begin(), gas_electrons_.end(),
                                      gas_configuration_copyab) != gas_electrons_.end()) {
                            tmp_vectorab.push_back(std::make_tuple(gas_count_1, gas_count_2,
                                                                   gas_count_3, gas_count_4));
                        }
                    }
                }
            }
        }
        aa_criterion[gas_configuration] = tmp_vectoraa;
        bb_criterion[gas_configuration] = tmp_vectorbb;
        ab_criterion[gas_configuration] = tmp_vectorab;
    }
    gas_electrons_.pop_back();
    return std::make_tuple(aa_criterion, bb_criterion, ab_criterion);
}

void CI_Reference::build_ci_reference(DeterminantHashVec& ref_space, bool include_rhf) {
    // Special case. If there are no active orbitals return an empty determinant
    if (nact_ == 0) {
        Determinant det;
        ref_space.add(det);
        initial_det_ = det;
        return;
    }

    Determinant det(get_occupation());
    outfile->Printf("\n  %s", str(det, nact_).c_str());
    initial_det_ = det;

    if (include_rhf)
        ref_space.add(det);

    if ((ref_type_ == "CIS") or (ref_type_ == "CISD")) {
        std::vector<int> aocc = det.get_alfa_occ(nact_);
        std::vector<int> bocc = det.get_beta_occ(nact_);
        std::vector<int> avir = det.get_alfa_vir(nact_);
        std::vector<int> bvir = det.get_beta_vir(nact_);

        int noalpha = aocc.size();
        int nobeta = bocc.size();
        int nvalpha = avir.size();
        int nvbeta = bvir.size();

        for (int i = 0; i < noalpha; ++i) {
            int ii = aocc[i];
            for (int a = 0; a < nvalpha; ++a) {
                int aa = avir[a];
                if ((mo_symmetry_[ii] ^ mo_symmetry_[aa]) == 0) {
                    Determinant new_det(det);
                    new_det.set_alfa_bit(ii, false);
                    new_det.set_alfa_bit(aa, true);
                    ref_space.add(new_det);
                }
            }
        }

        for (int i = 0; i < nobeta; ++i) {
            int ii = bocc[i];
            for (int a = 0; a < nvbeta; ++a) {
                int aa = bvir[a];
                if ((mo_symmetry_[ii] ^ mo_symmetry_[aa]) == 0) {
                    Determinant new_det(det);
                    new_det.set_beta_bit(ii, false);
                    new_det.set_beta_bit(aa, true);
                    ref_space.add(new_det);
                }
            }
        }
    }

    if ((ref_type_ == "CID") or (ref_type_ == "CISD")) {
        std::vector<int> aocc = det.get_alfa_occ(nact_);
        std::vector<int> bocc = det.get_beta_occ(nact_);
        std::vector<int> avir = det.get_alfa_vir(nact_);
        std::vector<int> bvir = det.get_beta_vir(nact_);

        int noalpha = aocc.size();
        int nobeta = bocc.size();
        int nvalpha = avir.size();
        int nvbeta = bvir.size();

        for (int i = 0; i < noalpha; ++i) {
            int ii = aocc[i];
            for (int j = i + 1; j < noalpha; ++j) {
                int jj = aocc[j];
                for (int a = 0; a < nvalpha; ++a) {
                    int aa = avir[a];
                    for (int b = a + 1; b < nvalpha; ++b) {
                        int bb = avir[b];
                        if ((mo_symmetry_[ii] ^ mo_symmetry_[jj] ^ mo_symmetry_[aa] ^
                             mo_symmetry_[bb]) == 0) {
                            Determinant new_det(det);
                            new_det.set_alfa_bit(ii, false);
                            new_det.set_alfa_bit(jj, false);
                            new_det.set_alfa_bit(aa, true);
                            new_det.set_alfa_bit(bb, true);
                            ref_space.add(new_det);
                        }
                    }
                }
            }
        }
        // Then the alpha-beta
        for (int i = 0; i < noalpha; ++i) {
            int ii = aocc[i];
            for (int j = 0; j < nobeta; ++j) {
                int jj = bocc[j];
                for (int a = 0; a < nvalpha; ++a) {
                    int aa = avir[a];
                    for (int b = 0; b < nvbeta; ++b) {
                        int bb = bvir[b];
                        if ((mo_symmetry_[ii] ^ mo_symmetry_[jj] ^ mo_symmetry_[aa] ^
                             mo_symmetry_[bb]) == 0) {
                            Determinant new_det(det);
                            new_det.set_alfa_bit(ii, false);
                            new_det.set_beta_bit(jj, false);
                            new_det.set_alfa_bit(aa, true);
                            new_det.set_beta_bit(bb, true);
                            ref_space.add(new_det);
                        }
                    }
                }
            }
        }
        // Lastly the beta-beta
        for (int i = 0; i < nobeta; ++i) {
            int ii = bocc[i];
            for (int j = i + 1; j < nobeta; ++j) {
                int jj = bocc[j];
                for (int a = 0; a < nvbeta; ++a) {
                    int aa = bvir[a];
                    for (int b = a + 1; b < nvbeta; ++b) {
                        int bb = bvir[b];
                        if ((mo_symmetry_[ii] ^
                             (mo_symmetry_[jj] ^ (mo_symmetry_[aa] ^ mo_symmetry_[bb]))) == 0) {
                            Determinant new_det(det);
                            new_det.set_beta_bit(ii, false);
                            new_det.set_beta_bit(jj, false);
                            new_det.set_beta_bit(aa, true);
                            new_det.set_beta_bit(bb, true);
                            ref_space.add(new_det);
                        }
                    }
                }
            }
        }
    }
}

void CI_Reference::build_cas_reference(DeterminantHashVec& ref_space) {
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
        Determinant core_det;
        for (int i = 0; i < nf; ++i) {
            core_det.set_alfa_bit(std::get<2>(active_mos[i]), true);
            core_det.set_beta_bit(std::get<2>(active_mos[i]), true);
        }
        // Generate all permutations, add the correct ones
        do {
            do {
                // Build determinant
                Determinant det(core_det);
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
                    ref_space.add(det);
                    if (!ref_space.size())
                        initial_det_ = det;
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
        } else {
            add_mo = false;
        }
    }

    if (ref_space.size() == 0) {
        throw psi::PSIEXCEPTION("Unable to generate CASCI space. Try increasing ACTIVE_GUESS_SIZE");
    }

    outfile->Printf("\n  Number of reference determinants: %zu", ref_space.size());
    outfile->Printf("\n  Reference generated from %d MOs", na);
}

void CI_Reference::build_cas_reference_full(DeterminantHashVec& ref_space) {
    ref_space.clear();

    // build alpha and beta strings
    auto a_strings = build_occ_string(nact_, nalpha_, mo_symmetry_);
    auto b_strings = build_occ_string(nact_, nbeta_, mo_symmetry_);

    // construct determinants
    for (int ha = 0; ha != nirrep_; ++ha) {
        int hb = ha ^ root_sym_;
        for (size_t a = 0, a_size = a_strings[ha].size(); a < a_size; ++a) {
            for (size_t b = 0, b_size = b_strings[hb].size(); b < b_size; ++b) {
                ref_space.add(Determinant(a_strings[ha][a], b_strings[hb][b]));
                if (!ref_space.size())
                    initial_det_ = Determinant(a_strings[ha][a], b_strings[hb][b]);
            }
        }
    }
}

std::vector<std::vector<std::vector<bool>>>
CI_Reference::build_occ_string(size_t norb, size_t nele, const std::vector<int>& symmetry) {
    if (nele > norb) {
        throw psi::PSIEXCEPTION("Invalid number of electron / orbital to build occ string.");
    }

    std::vector<std::vector<std::vector<bool>>> out(nirrep_, std::vector<std::vector<bool>>());

    std::vector<bool> occ_tmp(norb, false);
    for (size_t i = norb - nele; i < norb; ++i)
        occ_tmp[i] = true;

    do {
        int sym = 0;
        for (size_t p = 0; p < norb; ++p) {
            if (occ_tmp[p])
                sym ^= symmetry[p];
        }
        out[sym].emplace_back(occ_tmp.begin(), occ_tmp.end());
    } while (std::next_permutation(occ_tmp.begin(), occ_tmp.begin() + norb));

    return out;
}

std::vector<std::vector<std::vector<bool>>>
CI_Reference::build_occ_string_subspace(size_t norb, size_t nele, const std::vector<int>& symmetry,
                                        size_t sub_orb, std::vector<size_t> eps_idx) {
    if (nele > norb) {
        throw psi::PSIEXCEPTION("Invalid number of electron / orbital to build occ string.");
    }

    std::vector<std::vector<std::vector<bool>>> out(nirrep_, std::vector<std::vector<bool>>());

    std::vector<bool> occ_tmp(norb, false);
    for (size_t i = 0; i < nele; ++i) {
        occ_tmp[i] = true;
    }

    // Orbitals in the subspace
    size_t start_index = nele > sub_orb ? nele - sub_orb : 0;
    size_t end_index = std::min(norb, (nele + sub_orb));

    do {
        int sym = 0;
        std::vector<bool> occ_eps_ordered(norb, false);
        for (size_t p = 0; p < norb; ++p) {
            if (occ_tmp[p]) {
                size_t pp = eps_idx[p];
                occ_eps_ordered[pp] = true;
                sym ^= symmetry[pp];
            }
        }
        out[sym].emplace_back(occ_eps_ordered.begin(), occ_eps_ordered.end());
    } while (std::prev_permutation(occ_tmp.begin() + start_index, occ_tmp.begin() + end_index));
    return out;
}

void CI_Reference::build_doci_reference(DeterminantHashVec& ref_space) {
    if (root_sym_ != 0) {
        outfile->Printf("\n  State must be totally symmetric for DOCI.");
        throw psi::PSIEXCEPTION("DOCI reference can only be under totally symmetric irrep.");
    }

    ref_space.clear();
    auto strings_per_irrep = build_occ_string(nact_, nalpha_, mo_symmetry_);

    // combine alpha and beta strings to form determinant
    for (int h = 0; h < nirrep_; ++h) {
        for (const auto& a : strings_per_irrep[h]) {
            ref_space.add(Determinant(a, a));
            if (!ref_space.size())
                initial_det_ = Determinant(a, a);
        }
    }
}

std::vector<std::vector<bool>>
CI_Reference::build_gas_occ_string(const std::vector<std::vector<std::vector<bool>>>& gas_strings,
                                   const std::vector<std::vector<size_t>>& rel_mos) {
    int ngas = gas_strings.size();
    if (ngas != static_cast<int>(rel_mos.size()))
        throw psi::PSIEXCEPTION("Inconsistent numbers of gas spaces");

    // compute the cartesian product of strings
    /* For example (see Python itertools product):
     * input:  [[gas1_occ1, gas1_occ2, gas1_occ3, ...], [gas2_occ1, gas2_occ2, ...], ...]
     * output: [[gas1_occ1, gas2_occ1, ...], [gas1_occ1, gas2_occ2, ...],
     *          [gas1_occ2, gas2_occ1, ...], [gas1_occ2, gas2_occ2, ...],
     *          [gas1_occ3, gas2_occ1, ...], [gas1_occ3, gas2_occ2, ...], ...]
     */
    auto product = math::cartesian_product(gas_strings);

    auto n_strings = product.size();
    std::vector<std::vector<bool>> out(n_strings);
    if (n_strings == 0)
        return out;

    // combine gas strings to a string of size nactv_orbs

    size_t n_threads = omp_get_num_threads();
    if (n_threads > n_strings)
        n_threads = n_strings;

#pragma omp parallel for num_threads(n_threads)
    for (size_t n = 0; n < n_strings; ++n) {

        const auto& strings = product[n];
        std::vector<bool> s(nact_, false);

        for (int g = 0; g < ngas; ++g) {
            const auto& rel = rel_mos[g];
            const auto& string = strings[g];
            for (int p = 0, psize = string.size(); p < psize; ++p) {
                if (string[p]) {
                    s[rel[p]] = true;
                }
            }
        }
        out[n] = s;
    }

    return out;
}

void CI_Reference::build_gas_single(DeterminantHashVec& ref_space) {
    // build the determinant from aufbau principle
    print_gas_scf_epsilon();
    get_gas_occupation();
    ref_space.clear();

    std::vector<std::vector<size_t>> rel_gas_mos;     // relative indices within active
    std::vector<std::vector<double>> rel_gas_eps(12); // ngas of SCF orbital energies
    std::vector<std::vector<size_t>> gas_eps_idx(
        12); // The index of orbitals in ascending order of orbital energies in each gas
    std::map<int, int> gas_nonzero_to_full;
    auto epsilon_a = scf_info_->epsilon_a();
    auto epsilon_b = scf_info_->epsilon_b();

    // Find the sorted_indexes of a vector
    auto argsort = [&](const std::vector<double>& v) {
        // initialize original index locations
        std::vector<size_t> idx(v.size());
        std::iota(idx.begin(), idx.end(), 0);

        // sort indexes based on comparing values in v
        // using std::stable_sort instead of std::sort
        // to avoid unnecessary index re-orderings
        // when v contains elements of equal values
        std::stable_sort(idx.begin(), idx.end(),
                         [&v](size_t i1, size_t i2) { return v[i1] < v[i2]; });

        return idx;
    };

    for (int gas = 0, gas_i = 0; gas < 6; ++gas) {
        std::string space_name = "GAS" + std::to_string(gas + 1);
        auto abs_mos = mo_space_info_->absolute_mo(space_name);

        auto norbs = abs_mos.size();

        std::vector<double> a_eps(norbs), b_eps(norbs);
        for (size_t i = 0; i < norbs; ++i) {
            a_eps[i] = epsilon_a->get(abs_mos[i]);
            b_eps[i] = epsilon_b->get(abs_mos[i]);
        }
        rel_gas_eps[2 * gas] = a_eps;
        rel_gas_eps[2 * gas + 1] = b_eps;
        gas_eps_idx[2 * gas] = argsort(a_eps);
        gas_eps_idx[2 * gas + 1] = argsort(b_eps);

        if (norbs == 0)
            continue;

        rel_gas_mos.push_back(mo_space_info_->pos_in_space(space_name, "ACTIVE"));

        gas_nonzero_to_full[gas_i] = gas;
        gas_i += 1;
    }

    int ngas = rel_gas_mos.size(); // number of nonzero-sized GAS

    // figure out symmetry product
    // e.g., [[gas1_h1, gas1_h2], [gas2_h1, gas2_h2]] ->
    // [[gas1_h1, gas2_h1], [gas1_h1, gas2_h2], [gas1_h2, gas2_h1], [gas1_h2, gas2_h2]]
    std::vector<std::vector<int>> irrep_pools(ngas);
    for (int i = 0; i < ngas; ++i) {
        std::vector<int> irrep(nirrep_);
        std::iota(irrep.begin(), irrep.end(), 0);
        irrep_pools[i] = irrep;
    }
    auto sym_product = math::cartesian_product(irrep_pools);

    // figure out aufbau occupation of a gas
    // in: nirrep of vector of occupation
    // out: nirrep of aufbau occupation
    auto aufbau_gas_occ = [&](std::vector<std::vector<std::vector<bool>>>& occ_strings,
                              const std::vector<double>& eps) {
        std::vector<std::vector<bool>> out(nirrep_);
        int norbs = eps.size();

        for (int h = 0; h < nirrep_; ++h) {
            const auto& occs = occ_strings[h];
            auto min_occ_h =
                std::min_element(occs.begin(), occs.end(),
                                 [&](const std::vector<bool>& lhs, const std::vector<bool>& rhs) {
                                     double e_lhs = 0.0, e_rhs = 0.0;
                                     for (int p = 0; p < norbs; ++p) {
                                         if (lhs[p])
                                             e_lhs += eps[p];
                                         if (rhs[p])
                                             e_rhs += eps[p];
                                     }
                                     return e_lhs < e_rhs;
                                 });
            if (min_occ_h != occs.end()) {
                out[h] = *min_occ_h;
            }
        }

        return out;
    };

    // figure out the aufbau occupation of size nactv
    // in: ngas of nirrep of aufbau occupation
    // out: nirrep of <energy, aufbau occupation>
    auto combine_aufbau_gas_occ = [&](const std::vector<std::vector<std::vector<bool>>>& gas_occs,
                                      bool beta) {
        std::vector<std::tuple<double, std::vector<bool>>> strings(nirrep_);

        auto product = math::cartesian_product(gas_occs);
        int shift = beta ? 1 : 0;

        for (size_t i = 0, isize = sym_product.size(); i < isize; ++i) {
            const auto& sym = sym_product[i];
            const auto& gas_occ = product[i];

            // check if all occupation is specified
            bool occ_ok = std::all_of(gas_occ.begin(), gas_occ.end(),
                                      [](const std::vector<bool>& x) { return x.size(); });
            if (occ_ok) {
                int h = 0; // symmetry of this product

                // combine to a big string
                std::vector<bool> big_string(nact_, false);
                double e = 0.0;

                for (int gas = 0; gas < ngas; ++gas) {
                    h ^= sym[gas];
                    const auto& occ = gas_occ[gas];
                    for (int p = 0, psize = occ.size(); p < psize; ++p) {
                        if (occ[p]) {
                            big_string[rel_gas_mos[gas][p]] = true;
                            e += rel_gas_eps[2 * gas_nonzero_to_full[gas] + shift][p];
                        }
                    }
                }
                strings[h] = {e, big_string};
            }
        }
        return strings;
    };

    // loop over all GAS configurations
    timer timer_gas("Build GAS determinants");
    print_h2("Building GAS Determinants");
    for (size_t config = 0, size = gas_electrons_.size(); config < size; ++config) {
        size_t max_sub_orb = 0;

        // Make sure only one ref is selected.
        if (ref_space.size()) {
            break;
        }

        // Calculate the maximum subspace size (HOMO-Max_sub_orb to LOMO+Max_sub_orb)
        // For each gas this is the maximum number between occupied and unoccupied orbitals
        for (int gas = 0; gas < 6; ++gas) {
            auto space_name = "GAS" + std::to_string(gas + 1);
            auto norb = mo_space_info_->size(space_name);
            if (norb == 0)
                continue;
            auto sym = mo_space_info_->symmetry(space_name);
            size_t max_gas_orb = 0;

            if (gas_electrons_[config][2 * gas] > gas_electrons_[config][2 * gas + 1]) {
                max_gas_orb = std::max(gas_electrons_[config][2 * gas],
                                       int(norb) - gas_electrons_[config][2 * gas + 1]);
            } else {
                max_gas_orb = std::max(gas_electrons_[config][2 * gas + 1],
                                       int(norb) - gas_electrons_[config][2 * gas]);
            }
            if (max_gas_orb > max_sub_orb) {
                max_sub_orb = max_gas_orb;
            }
        }

        size_t sub_orb = 0;
        // size of subspace (LUMO-sub_orb to HOMO+sub_orb)
        // Increase subspace till find a determinant within in a subspace

        do {
            // build alpha or beta strings (ngas of nirrep of aufbau occupation)
            std::vector<std::vector<std::vector<bool>>> a_tmp, b_tmp;
            for (int gas = 0; gas < 6; ++gas) {
                auto space_name = "GAS" + std::to_string(gas + 1);
                auto norb = mo_space_info_->size(space_name);
                if (norb == 0)
                    continue;
                auto sym = mo_space_info_->symmetry(space_name);

                // alpha aufbau string of each irrep
                auto strings = build_occ_string_subspace(norb, gas_electrons_[config][2 * gas], sym,
                                                         sub_orb, gas_eps_idx[2 * gas]);
                a_tmp.push_back(aufbau_gas_occ(strings, rel_gas_eps[2 * gas]));

                // beta aufbau string of each irrep
                strings = build_occ_string_subspace(norb, gas_electrons_[config][2 * gas + 1], sym,
                                                    sub_orb, gas_eps_idx[2 * gas + 1]);
                b_tmp.push_back(aufbau_gas_occ(strings, rel_gas_eps[2 * gas + 1]));
            }

            // combine to a string of size nactv <energy, occupation>
            auto a_strings = combine_aufbau_gas_occ(a_tmp, false);
            auto b_strings = combine_aufbau_gas_occ(b_tmp, true);

            // combine to determinant
            double e_min = 0.0;
            Determinant det_min;
            for (int h = 0; h < nirrep_; ++h) {
                const auto& a = a_strings[h];
                const auto& b = b_strings[h ^ root_sym_];

                double e = std::get<0>(a) + std::get<0>(b);
                double e_check = std::get<0>(a) * std::get<0>(b);
                Determinant det(std::get<1>(a), std::get<1>(b));

                if (e < e_min and (nalpha_ + nbeta_ - 2 * det.npair() + 1) >= multiplicity_ and
                    e_check != 0) {
                    det_min = det;
                    e_min = e;
                }
            }
            if (e_min != 0.0) {
                ref_space.add(det_min);
                initial_det_ = det_min;
                outfile->Printf("\n    Reference determinant: %s", str(det_min, nact_).c_str());
                break;
            }
            sub_orb++;
        } while ((sub_orb < max_sub_orb) && (ref_space.size() == 0));
    }
}

void CI_Reference::build_gas_reference(DeterminantHashVec& ref_space) {
    print_gas_scf_epsilon();
    get_gas_occupation();

    ref_space.clear();

    // relative indices within the active orbitals
    std::vector<std::vector<size_t>> rel_gas_mos;

    for (int gas = 0; gas < 6; ++gas) {
        std::string space_name = "GAS" + std::to_string(gas + 1);
        auto abs_mos = mo_space_info_->absolute_mo(space_name);
        if (abs_mos.size() == 0)
            continue;
        rel_gas_mos.push_back(mo_space_info_->pos_in_space(space_name, "ACTIVE"));
    }

    int ngas = rel_gas_mos.size(); // number of nonzero-sized GAS

    // figure out symmetry product
    // e.g., [[gas1_h1, gas1_h2], [gas2_h1, gas2_h2]] ->
    // [[gas1_h1, gas2_h1], [gas1_h1, gas2_h2], [gas1_h2, gas2_h1], [gas1_h2, gas2_h2]]
    std::vector<std::vector<int>> irrep_pools(ngas);
    for (int i = 0; i < ngas; ++i) {
        std::vector<int> irrep(nirrep_);
        std::iota(irrep.begin(), irrep.end(), 0);
        irrep_pools[i] = irrep;
    }
    auto sym_product = math::cartesian_product(irrep_pools);

    // dynamic programming for given GAS n/orbital/electron
    std::map<std::tuple<int, size_t, size_t>, std::vector<std::vector<std::vector<bool>>>>
        gasn_config_to_occ_string;

    // loop over all GAS configurations
    print_h2("Building GAS Determinants");
    outfile->Printf("\n    Config.  #Determinants     Time/s");
    outfile->Printf("\n    ---------------------------------");

    timer timer_gas("Build GAS determinants");
    for (size_t config = 0, size = gas_electrons_.size(); config < size; ++config) {
        local_timer lt;
        outfile->Printf("\n    %6d", config + 1);

        // build alpha or beta strings (ngas of nirrep of vector of occupation)
        std::vector<std::vector<std::vector<std::vector<bool>>>> a_tmp, b_tmp;

        for (int gas = 0; gas < 6; ++gas) {
            auto space_name = "GAS" + std::to_string(gas + 1);
            auto norb = mo_space_info_->size(space_name);
            if (norb == 0)
                continue;

            auto sym = mo_space_info_->symmetry(space_name);

            std::tuple<int, size_t, size_t> a_key{gas, norb, gas_electrons_[config][2 * gas]};
            if (gasn_config_to_occ_string.find(a_key) == gasn_config_to_occ_string.end()) {
                gasn_config_to_occ_string[a_key] =
                    build_occ_string(norb, gas_electrons_[config][2 * gas], sym);
            }
            a_tmp.push_back(gasn_config_to_occ_string[a_key]);

            std::tuple<int, size_t, size_t> b_key{gas, norb, gas_electrons_[config][2 * gas + 1]};
            if (gasn_config_to_occ_string.find(b_key) == gasn_config_to_occ_string.end()) {
                gasn_config_to_occ_string[b_key] =
                    build_occ_string(norb, gas_electrons_[config][2 * gas + 1], sym);
            }
            b_tmp.push_back(gasn_config_to_occ_string[b_key]);
        }

        // alpha and beta strings (nirrep of vector of occupations)
        std::vector<std::vector<std::vector<bool>>> a_strings(nirrep_), b_strings(nirrep_);

        // loop over symmetry product
        for (const auto& sym : sym_product) {
            int irrep = 0;
            std::vector<std::vector<std::vector<bool>>> a(ngas), b(ngas);

            for (int gas = 0; gas < ngas; ++gas) {
                int h = sym[gas];
                irrep ^= h;
                a[gas] = a_tmp[gas][h];
                b[gas] = b_tmp[gas][h];
            }

            // alpha
            auto strings_irrep = build_gas_occ_string(a, rel_gas_mos);
            std::move(strings_irrep.begin(), strings_irrep.end(),
                      std::back_inserter(a_strings[irrep]));

            // beta
            strings_irrep = build_gas_occ_string(b, rel_gas_mos);
            std::move(strings_irrep.begin(), strings_irrep.end(),
                      std::back_inserter(b_strings[irrep]));
        }

        // combine alpha and beta strings to form determinant
        size_t n = 0;
        for (int ha = 0; ha < nirrep_; ++ha) {
            int hb = root_sym_ ^ ha;
            for (const auto& a : a_strings[ha]) {
                for (const auto& b : b_strings[hb]) {
                    if (!ref_space.size())
                        initial_det_ = Determinant(a, b);
                    ref_space.add(Determinant(a, b));
                    n++;
                }
            }
        }

        outfile->Printf("  %14zu  %9.3e", n, lt.get());
    }

    outfile->Printf("\n    ---------------------------------");
    outfile->Printf("\n    Total:  %14zu  %9.3e", ref_space.size(), timer_gas.stop());
    outfile->Printf("\n    ---------------------------------");
    //    print_h2("GAS Determinants");
    //    for (const auto& det : ref_space) {
    //        outfile->Printf("\n    %s", str(det, nact_).c_str());
    //    }
}

std::vector<std::tuple<double, int, int>> CI_Reference::sym_labeled_orbitals(std::string type) {
    size_t nact = mo_space_info_->size("ACTIVE");

    std::vector<std::tuple<double, int, int>> labeled_orb;

    std::shared_ptr<Vector> epsilon_a = scf_info_->epsilon_a();

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
        std::shared_ptr<Vector> epsilon_b = scf_info_->epsilon_b();
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

void CI_Reference::get_gas_occupation() {
    gas_electrons_.clear();

    print_h2("Number of Electrons in GAS");
    outfile->Printf("\n    GAS  MAX  MIN");
    outfile->Printf("\n    -------------");

    // The vectors of maximum number of electrons, minimum number of electrons,
    // and the number of orbitals
    std::vector<int> gas_maxe;
    std::vector<int> gas_mine;
    std::vector<int> gas_orbital;
    gas_num_ = 0;
    for (size_t gas_count = 0; gas_count < 6; gas_count++) {
        std::string space = "GAS" + std::to_string(gas_count + 1);
        int orbital_maximum = mo_space_info_->size(space);
        gas_orbital.push_back(orbital_maximum);
        if (orbital_maximum) {
            outfile->Printf("\n    %3d", gas_count + 1);

            // define max_e_number to be the largest possible
            size_t max_e_number = std::min(orbital_maximum * 2, nalpha_ + nbeta_);
            // but if we can read its value, do so
            if (state_info_.gas_max().size() > gas_count) {
                // If the defined maximum number of electrons exceed number of orbitals,
                // redefine maximum number of elctrons
                max_e_number = std::min(state_info_.gas_max()[gas_count], max_e_number);
            }
            gas_maxe.push_back(max_e_number);

            // define min_e_number to be the smallest possible
            size_t min_e_number = 0;
            // but if we can read its value, do so
            if (state_info_.gas_min().size() > gas_count) {
                min_e_number = state_info_.gas_min()[gas_count];
            }
            gas_mine.push_back(min_e_number);

            outfile->Printf(" %4d %4d", max_e_number, min_e_number);
            gas_num_ = gas_num_ + 1;
        } else {
            gas_maxe.push_back(0);
            gas_mine.push_back(0);
        }
    }
    outfile->Printf("\n    -------------");

    print_h2("Possible Electron Occupations in GAS");
    outfile->Printf("\n    Config.");
    int ndash = 7;
    std::vector<std::string> gas_electron_name = {"GAS1_A", "GAS1_B", "GAS2_A", "GAS2_B",
                                                  "GAS3_A", "GAS3_B", "GAS4_A", "GAS4_B",
                                                  "GAS5_A", "GAS5_B", "GAS6_A", "GAS6_B"};
    for (size_t i = 0; i < 2 * gas_num_; i++) {
        std::string name = gas_electron_name.at(i).substr(3, 3);
        outfile->Printf("  %s", name.c_str());
        ndash += 5;
    }
    std::string dash(ndash, '-');
    outfile->Printf("\n    %s", dash.c_str());

    int n_config = 0;

    for (int gas6_na = std::max(0, gas_mine[5] - gas_orbital[5]);
         gas6_na <= std::min(gas_maxe[5], gas_orbital[5]); gas6_na++) {
        for (int gas6_nb = std::max(0, gas_mine[5] - gas6_na);
             gas6_nb <= std::min(gas_maxe[5] - gas6_na, gas_orbital[5]); gas6_nb++) {
            for (int gas5_na = std::max(0, gas_mine[4] - gas_orbital[4]);
                 gas5_na <= std::min(gas_maxe[4], gas_orbital[4]); gas5_na++) {
                for (int gas5_nb = std::max(0, gas_mine[4] - gas5_na);
                     gas5_nb <= std::min(gas_maxe[4] - gas5_na, gas_orbital[4]); gas5_nb++) {
                    for (int gas4_na = std::max(0, gas_mine[3] - gas_orbital[3]);
                         gas4_na <= std::min(gas_maxe[3], gas_orbital[3]); gas4_na++) {
                        for (int gas4_nb = std::max(0, gas_mine[3] - gas4_na);
                             gas4_nb <= std::min(gas_maxe[3] - gas4_na, gas_orbital[3]);
                             gas4_nb++) {
                            for (int gas3_na = std::max(0, gas_mine[2] - gas_orbital[2]);
                                 gas3_na <= std::min(gas_maxe[2], gas_orbital[2]); gas3_na++) {
                                for (int gas3_nb = std::max(0, gas_mine[2] - gas3_na);
                                     gas3_nb <= std::min(gas_maxe[2] - gas3_na, gas_orbital[2]);
                                     gas3_nb++) {
                                    for (int gas2_na = std::max(0, gas_mine[1] - gas_orbital[1]);
                                         gas2_na <= std::min(gas_maxe[1], gas_orbital[1]);
                                         gas2_na++) {
                                        for (int gas2_nb = std::max(0, gas_mine[1] - gas2_na);
                                             gas2_nb <=
                                             std::min(gas_maxe[1] - gas2_na, gas_orbital[1]);
                                             gas2_nb++) {
                                            int gas1_na = nalpha_ - gas2_na - gas3_na - gas4_na -
                                                          gas5_na - gas6_na;
                                            int gas1_nb = nbeta_ - gas2_nb - gas3_nb - gas4_nb -
                                                          gas5_nb - gas6_nb;
                                            int gas1_max = std::max(gas1_na, gas1_nb);
                                            int gas1_min = std::min(gas1_na, gas1_nb);
                                            int gas1_total = gas1_na + gas1_nb;
                                            if (gas1_total <= gas_maxe[0] and
                                                gas1_max <= gas_orbital[0] and gas1_min >= 0 and
                                                gas1_total >= gas_mine[0]) {
                                                std::vector<int> gas_configuration{
                                                    gas1_na, gas1_nb, gas2_na, gas2_nb,
                                                    gas3_na, gas3_nb, gas4_na, gas4_nb,
                                                    gas5_na, gas5_nb, gas6_na, gas6_nb};
                                                gas_electrons_.push_back(gas_configuration);

                                                outfile->Printf("\n    %6d ", ++n_config);
                                                for (size_t i = 0; i < 2 * gas_num_; i++) {
                                                    outfile->Printf(" %4d", gas_configuration[i]);
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    outfile->Printf("\n    %s", dash.c_str());
    outfile->Printf("\n    n_A/B: # of alpha/beta electrons in GASn");
}

std::vector<std::vector<int>> CI_Reference::gas_electrons() { return gas_electrons_; }

Determinant CI_Reference::get_occupation() {
    int nact = mo_space_info_->size("ACTIVE");
    Determinant det;

    // If there are no electrons return an empty determinant
    if (nalpha_ + nbeta_ == 0) {
        return det;
    }

    // nsym denotes the number of electrons needed to assign symmetry and
    // multiplicity
    int nsym = twice_ms_;
    int orb_sym = root_sym_;

    if (twice_ms_ == 0.0) {
        nsym = 2;
    }

    // Grab an ordered list of orbital energies, sym labels, and idxs
    std::vector<std::tuple<double, int, int>> labeled_orb_en;
    std::vector<std::tuple<double, int, int>> labeled_orb_en_alfa;
    std::vector<std::tuple<double, int, int>> labeled_orb_en_beta;

    // For a restricted reference
    labeled_orb_en = sym_labeled_orbitals("RHF");

    // Build initial reference determinant from restricted reference
    for (int i = 0; i < nalpha_; ++i) {
        det.set_alfa_bit(std::get<2>(labeled_orb_en[i]), true);
    }
    for (int i = 0; i < nbeta_; ++i) {
        det.set_beta_bit(std::get<2>(labeled_orb_en[i]), true);
    }

    // Loop over as many outer-shell electrons as needed to get correct sym
    for (int k = 1; k <= nsym;) {

        bool add = false;
        // Remove electron from highest energy docc
        det.set_alfa_bit(std::get<2>(labeled_orb_en[nalpha_ - k]), false);
        //        occupation[std::get<2>(labeled_orb_en[nalpha_ - k])] = 0;

        // Determine proper symmetry for new occupation
        // orb_sym = ms_;

        if (twice_ms_ == 0.0) {
            orb_sym = std::get<1>(labeled_orb_en[nalpha_ - 1]) ^ orb_sym;
        } else {
            for (int i = 1; i <= nsym; ++i) {
                orb_sym = std::get<1>(labeled_orb_en[nalpha_ - i]) ^ orb_sym;
            }
            orb_sym = std::get<1>(labeled_orb_en[nalpha_ - k]) ^ orb_sym;
        }

        // Add electron to lowest-energy orbital of proper symmetry
        // Loop from current occupation to max MO until correct orbital is
        // reached
        for (int i = std::max(nalpha_ - k, 0), maxi = nact; i < maxi; ++i) {
            if (orb_sym == std::get<1>(labeled_orb_en[i]) and
                det.get_alfa_bit(std::get<2>(labeled_orb_en[i])) != true) {
                det.set_alfa_bit(std::get<2>(labeled_orb_en[i]), true);
                //                occupation[std::get<2>(labeled_orb_en[i])] = true;
                add = true;
                break;
            } else {
                continue;
            }
        }
        // If a new occupation could not be created, put electron back and
        // remove a different one
        if (!add and (nalpha_ - k > 0)) {
            det.set_alfa_bit(std::get<2>(labeled_orb_en[nalpha_ - k]), true);
            //            occupation[] = 1;
            ++k;
        } else {
            break;
        }

    } // End loop over k
    return det;
}
} // namespace forte
