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
    psi::Dimension doccpi = scf_info_->doccpi();
    psi::Dimension soccpi = scf_info_->soccpi();

    // Number of irreps
    nirrep_ = doccpi.n();

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

    // First determine number of alpha and beta electrons
    // Assume twice_ms =( Na - Nb )
    int nel = 0;
    for (int h = 0; h < nirrep_; ++h) {
        nel += 2 * doccpi[h] + soccpi[h];
    }

    nel -= 2 * ninact;

    nalpha_ = 0.5 * (nel + twice_ms_);
    nbeta_ = nel - nalpha_;

    //    outfile->Printf("\n  Number of active orbitals: %d", Determinant::nmo_);
    outfile->Printf("\n  Number of active alpha electrons: %d", nalpha_);
    outfile->Printf("\n  Number of active beta electrons: %d", nbeta_);
    outfile->Printf("\n  Maximum reference space size: %zu", subspace_size_);
}

CI_Reference::~CI_Reference() {}

void CI_Reference::build_reference(std::vector<Determinant>& ref_space) {
    if (ref_type_ == "CAS") {
        build_cas_reference(ref_space);
    } else if (ref_type_ == "GAS") {
        // Complete GAS
        print_gas_scf_epsilon();
        get_gas_occupation();
        build_gas_reference(ref_space);
    } else if (ref_type_ == "GAS_SINGLE") {
        // Low(est) energy one in GAS
        print_gas_scf_epsilon();
        get_gas_occupation();
        build_gas_single(ref_space);
    } else {
        build_ci_reference(ref_space);
        outfile->Printf("\n  Building_reference.", subspace_size_);
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

void CI_Reference::build_ci_reference(std::vector<Determinant>& ref_space) {
    // Special case. If there are no active orbitals return an empty determinant
    if (nact_ == 0) {
        Determinant det;
        ref_space.push_back(det);
        return;
    }

    Determinant det(get_occupation());
    outfile->Printf("\n  %s", str(det, nact_).c_str());

    ref_space.push_back(det);

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
                    ref_space.push_back(new_det);
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
                    ref_space.push_back(new_det);
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
                            ref_space.push_back(new_det);
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
                            ref_space.push_back(new_det);
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
                            ref_space.push_back(new_det);
                        }
                    }
                }
            }
        }
    }
}

void CI_Reference::build_cas_reference(std::vector<Determinant>& ref_space) {
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
                for (size_t p = 0; p < na; ++p) {
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
        throw psi::PSIEXCEPTION("Unable to generate CASCI space. Try increasing ACTIVE_GUESS_SIZE");
    }

    outfile->Printf("\n  Number of reference determinants: %zu", ref_space.size());
    outfile->Printf("\n  Reference generated from %d MOs", na);
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

    size_t n_threads = omp_get_num_threads();
    if (n_threads > n_strings)
        n_threads = n_strings;

        // combine gas strings to a string of size nactv_orbs
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

void CI_Reference::build_gas_single(std::vector<Determinant>& ref_space) {

    // build one single low energy determinant in the gas space

    std::shared_ptr<Vector> epsilon_a = scf_info_->epsilon_a();

    // relative_mo of each GAS
    // Sort the aboslute_mo in the increasing order of energy for each GAS
    std::vector<std::vector<size_t>> relative_gas_mo;
    size_t total_act = 0;
    outfile->Printf("\n");
    outfile->Printf("\n  GAS Orbital Energies");
    outfile->Printf("\n  GAS   Energies    Orb ");
    std::vector<size_t> act_mo = mo_space_info_->absolute_mo("ACTIVE");
    std::map<int, int> re_ab_mo;
    for (size_t i = 0; i < act_mo.size(); i++) {
        re_ab_mo[act_mo[i]] = i;
    }
    for (size_t gas_count = 0; gas_count < 6; gas_count++) {
        const std::string space = "GAS" + std::to_string(gas_count + 1);
        ;
        std::vector<size_t> relative_mo_sorted;
        auto vec_mo_info = mo_space_info_->absolute_mo(space);
        std::vector<std::pair<double, int>> gas_orb_e;
        for (size_t i = 0; i < vec_mo_info.size(); ++i) {
            auto orb = vec_mo_info[i];
            gas_orb_e.push_back(std::make_pair(epsilon_a->get(orb), re_ab_mo[orb]));
        }
        total_act += gas_orb_e.size();
        std::sort(gas_orb_e.begin(), gas_orb_e.end());

        for (size_t i = 0; i < gas_orb_e.size(); i++) {
            auto act_orb = gas_orb_e[i].second;
            relative_mo_sorted.push_back(act_orb);
            outfile->Printf("\n   %d  %12.9f  %d ", gas_count + 1, gas_orb_e[i].first, act_orb);
        }
        relative_gas_mo.push_back(relative_mo_sorted);
    }

    // iterate over all possible gas occupations
    for (size_t i_config = 0; i_config < gas_electrons_.size(); ++i_config) {

        size_t gas1_na = gas_electrons_[i_config][0];
        size_t gas1_nb = gas_electrons_[i_config][1];
        size_t gas2_na = gas_electrons_[i_config][2];
        size_t gas2_nb = gas_electrons_[i_config][3];
        size_t gas3_na = gas_electrons_[i_config][4];
        size_t gas3_nb = gas_electrons_[i_config][5];
        size_t gas4_na = gas_electrons_[i_config][6];
        size_t gas4_nb = gas_electrons_[i_config][7];
        size_t gas5_na = gas_electrons_[i_config][8];
        size_t gas5_nb = gas_electrons_[i_config][9];
        size_t gas6_na = gas_electrons_[i_config][10];
        size_t gas6_nb = gas_electrons_[i_config][11];
        size_t gas1_size = relative_gas_mo[0].size();
        size_t gas2_size = relative_gas_mo[1].size();
        size_t gas3_size = relative_gas_mo[2].size();
        size_t gas4_size = relative_gas_mo[3].size();
        size_t gas5_size = relative_gas_mo[4].size();
        size_t gas6_size = relative_gas_mo[5].size();

        // create the occupation of orbitals
        std::vector<bool> tmp_det_gas1_a(gas1_size, false);
        std::vector<bool> tmp_det_gas1_b(gas1_size, false);
        std::vector<bool> tmp_det_gas2_a(gas2_size, false);
        std::vector<bool> tmp_det_gas2_b(gas2_size, false);
        std::vector<bool> tmp_det_gas3_a(gas3_size, false);
        std::vector<bool> tmp_det_gas3_b(gas3_size, false);
        std::vector<bool> tmp_det_gas4_a(gas4_size, false);
        std::vector<bool> tmp_det_gas4_b(gas4_size, false);
        std::vector<bool> tmp_det_gas5_a(gas5_size, false);
        std::vector<bool> tmp_det_gas5_b(gas5_size, false);
        std::vector<bool> tmp_det_gas6_a(gas6_size, false);
        std::vector<bool> tmp_det_gas6_b(gas6_size, false);

        if (gas1_size > 0) {
            for (size_t i = 0; i < gas1_na; ++i) {
                tmp_det_gas1_a[i] = true;
            }
            for (size_t i = 0; i < gas1_nb; ++i) {
                tmp_det_gas1_b[i] = true;
            }
        }
        if (gas2_size > 0) {
            for (size_t i = 0; i < gas2_na; ++i) {
                tmp_det_gas2_a[i] = true;
            }
            for (size_t i = 0; i < gas2_nb; ++i) {
                tmp_det_gas2_b[i] = true;
            }
        }
        if (gas3_size > 0) {
            for (size_t i = 0; i < gas3_na; ++i) {
                tmp_det_gas3_a[i] = true;
            }
            for (size_t i = 0; i < gas3_nb; ++i) {
                tmp_det_gas3_b[i] = true;
            }
        }
        if (gas4_size > 0) {
            for (size_t i = 0; i < gas4_na; ++i) {
                tmp_det_gas4_a[i] = true;
            }
            for (size_t i = 0; i < gas4_nb; ++i) {
                tmp_det_gas4_b[i] = true;
            }
        }
        if (gas5_size > 0) {
            for (size_t i = 0; i < gas5_na; ++i) {
                tmp_det_gas5_a[i] = true;
            }
            for (size_t i = 0; i < gas5_nb; ++i) {
                tmp_det_gas5_b[i] = true;
            }
        }
        if (gas6_size > 0) {
            for (size_t i = 0; i < gas6_na; ++i) {
                tmp_det_gas6_a[i] = true;
            }
            for (size_t i = 0; i < gas6_nb; ++i) {
                tmp_det_gas6_b[i] = true;
            }
        }

        // Sort
        std::sort(begin(tmp_det_gas1_a), end(tmp_det_gas1_a));
        std::sort(begin(tmp_det_gas1_b), end(tmp_det_gas1_b));
        std::sort(begin(tmp_det_gas2_a), end(tmp_det_gas2_a));
        std::sort(begin(tmp_det_gas2_b), end(tmp_det_gas2_b));
        std::sort(begin(tmp_det_gas3_a), end(tmp_det_gas3_a));
        std::sort(begin(tmp_det_gas3_b), end(tmp_det_gas3_b));
        std::sort(begin(tmp_det_gas4_a), end(tmp_det_gas4_a));
        std::sort(begin(tmp_det_gas4_b), end(tmp_det_gas4_b));
        std::sort(begin(tmp_det_gas5_a), end(tmp_det_gas5_a));
        std::sort(begin(tmp_det_gas5_b), end(tmp_det_gas5_b));
        std::sort(begin(tmp_det_gas6_a), end(tmp_det_gas6_a));
        std::sort(begin(tmp_det_gas6_b), end(tmp_det_gas6_b));

        // Save all permutations
        std::vector<std::vector<bool>> alldet_gas1_a;
        std::vector<std::vector<bool>> alldet_gas1_b;
        std::vector<std::vector<bool>> alldet_gas2_a;
        std::vector<std::vector<bool>> alldet_gas2_b;
        std::vector<std::vector<bool>> alldet_gas3_a;
        std::vector<std::vector<bool>> alldet_gas3_b;
        std::vector<std::vector<bool>> alldet_gas4_a;
        std::vector<std::vector<bool>> alldet_gas4_b;
        std::vector<std::vector<bool>> alldet_gas5_a;
        std::vector<std::vector<bool>> alldet_gas5_b;
        std::vector<std::vector<bool>> alldet_gas6_a;
        std::vector<std::vector<bool>> alldet_gas6_b;

        do {
            alldet_gas1_a.push_back(tmp_det_gas1_a);
        } while (std::next_permutation(tmp_det_gas1_a.begin(), tmp_det_gas1_a.begin() + gas1_size));
        do {
            alldet_gas1_b.push_back(tmp_det_gas1_b);
        } while (std::next_permutation(tmp_det_gas1_b.begin(), tmp_det_gas1_b.begin() + gas1_size));
        do {
            alldet_gas2_a.push_back(tmp_det_gas2_a);
        } while (std::next_permutation(tmp_det_gas2_a.begin(), tmp_det_gas2_a.begin() + gas2_size));
        do {
            alldet_gas2_b.push_back(tmp_det_gas2_b);
        } while (std::next_permutation(tmp_det_gas2_b.begin(), tmp_det_gas2_b.begin() + gas2_size));
        do {
            alldet_gas3_a.push_back(tmp_det_gas3_a);
        } while (std::next_permutation(tmp_det_gas3_a.begin(), tmp_det_gas3_a.begin() + gas3_size));
        do {
            alldet_gas3_b.push_back(tmp_det_gas3_b);
        } while (std::next_permutation(tmp_det_gas3_b.begin(), tmp_det_gas3_b.begin() + gas3_size));
        do {
            alldet_gas4_a.push_back(tmp_det_gas4_a);
        } while (std::next_permutation(tmp_det_gas4_a.begin(), tmp_det_gas4_a.begin() + gas4_size));
        do {
            alldet_gas4_b.push_back(tmp_det_gas4_b);
        } while (std::next_permutation(tmp_det_gas4_b.begin(), tmp_det_gas4_b.begin() + gas4_size));
        do {
            alldet_gas5_a.push_back(tmp_det_gas5_a);
        } while (std::next_permutation(tmp_det_gas5_a.begin(), tmp_det_gas5_a.begin() + gas5_size));
        do {
            alldet_gas5_b.push_back(tmp_det_gas5_b);
        } while (std::next_permutation(tmp_det_gas5_b.begin(), tmp_det_gas5_b.begin() + gas5_size));
        do {
            alldet_gas6_a.push_back(tmp_det_gas6_a);
        } while (std::next_permutation(tmp_det_gas6_a.begin(), tmp_det_gas6_a.begin() + gas6_size));
        do {
            alldet_gas6_b.push_back(tmp_det_gas6_b);
        } while (std::next_permutation(tmp_det_gas6_b.begin(), tmp_det_gas6_b.begin() + gas6_size));

        // Reverse for the low energy
        std::reverse(begin(alldet_gas1_a), end(alldet_gas1_a));
        std::reverse(begin(alldet_gas1_b), end(alldet_gas1_b));
        std::reverse(begin(alldet_gas2_a), end(alldet_gas2_a));
        std::reverse(begin(alldet_gas2_b), end(alldet_gas2_b));
        std::reverse(begin(alldet_gas3_a), end(alldet_gas3_a));
        std::reverse(begin(alldet_gas3_b), end(alldet_gas3_b));
        std::reverse(begin(alldet_gas4_a), end(alldet_gas4_a));
        std::reverse(begin(alldet_gas4_b), end(alldet_gas4_b));
        std::reverse(begin(alldet_gas5_a), end(alldet_gas5_a));
        std::reverse(begin(alldet_gas5_b), end(alldet_gas5_b));
        std::reverse(begin(alldet_gas6_a), end(alldet_gas6_a));
        std::reverse(begin(alldet_gas6_b), end(alldet_gas6_b));

        // Permutation of all the orbitals
        for (const auto& det_gas1_a : alldet_gas1_a) {
            for (const auto& det_gas1_b : alldet_gas1_b) {
                for (const auto& det_gas2_a : alldet_gas2_a) {
                    for (const auto& det_gas2_b : alldet_gas2_b) {
                        for (const auto& det_gas3_a : alldet_gas3_a) {
                            for (const auto& det_gas3_b : alldet_gas3_b) {
                                for (const auto& det_gas4_a : alldet_gas4_a) {
                                    for (const auto& det_gas4_b : alldet_gas4_b) {
                                        for (const auto& det_gas5_a : alldet_gas5_a) {
                                            for (const auto& det_gas5_b : alldet_gas5_b) {
                                                for (const auto& det_gas6_a : alldet_gas6_a) {
                                                    for (const auto& det_gas6_b : alldet_gas6_b) {
                                                        // Build determinant
                                                        // outfile->Printf(
                                                        //   "\n Possible
                                                        //   Configurations");
                                                        Determinant det;
                                                        int sym = 0;
                                                        for (size_t p = 0; p < gas1_size; ++p) {
                                                            det.set_alfa_bit(relative_gas_mo[0][p],
                                                                             det_gas1_a[p]);
                                                            det.set_beta_bit(relative_gas_mo[0][p],
                                                                             det_gas1_b[p]);
                                                            if (det_gas1_a[p]) {
                                                                sym ^= mo_symmetry_
                                                                    [relative_gas_mo[0][p]];
                                                            }
                                                            if (det_gas1_b[p]) {
                                                                sym ^= mo_symmetry_
                                                                    [relative_gas_mo[0][p]];
                                                            }
                                                        }
                                                        for (size_t p = 0; p < gas2_size; ++p) {
                                                            det.set_alfa_bit(relative_gas_mo[1][p],
                                                                             det_gas2_a[p]);
                                                            det.set_beta_bit(relative_gas_mo[1][p],
                                                                             det_gas2_b[p]);
                                                            if (det_gas2_a[p]) {
                                                                sym ^= mo_symmetry_
                                                                    [relative_gas_mo[1][p]];
                                                            }
                                                            if (det_gas2_b[p]) {
                                                                sym ^= mo_symmetry_
                                                                    [relative_gas_mo[1][p]];
                                                            }
                                                        }
                                                        for (size_t p = 0; p < gas3_size; ++p) {
                                                            det.set_alfa_bit(relative_gas_mo[2][p],
                                                                             det_gas3_a[p]);
                                                            det.set_beta_bit(relative_gas_mo[2][p],
                                                                             det_gas3_b[p]);
                                                            if (det_gas3_a[p]) {
                                                                sym ^= mo_symmetry_
                                                                    [relative_gas_mo[2][p]];
                                                            }
                                                            if (det_gas3_b[p]) {
                                                                sym ^= mo_symmetry_
                                                                    [relative_gas_mo[2][p]];
                                                            }
                                                        }
                                                        for (size_t p = 0; p < gas4_size; ++p) {
                                                            det.set_alfa_bit(relative_gas_mo[3][p],
                                                                             det_gas4_a[p]);
                                                            det.set_beta_bit(relative_gas_mo[3][p],
                                                                             det_gas4_b[p]);
                                                            if (det_gas4_a[p]) {
                                                                sym ^= mo_symmetry_
                                                                    [relative_gas_mo[3][p]];
                                                            }
                                                            if (det_gas4_b[p]) {
                                                                sym ^= mo_symmetry_
                                                                    [relative_gas_mo[3][p]];
                                                            }
                                                        }
                                                        for (size_t p = 0; p < gas5_size; ++p) {
                                                            det.set_alfa_bit(relative_gas_mo[4][p],
                                                                             det_gas5_a[p]);
                                                            det.set_beta_bit(relative_gas_mo[4][p],
                                                                             det_gas5_b[p]);
                                                            if (det_gas5_a[p]) {
                                                                sym ^= mo_symmetry_
                                                                    [relative_gas_mo[4][p]];
                                                            }
                                                            if (det_gas5_b[p]) {
                                                                sym ^= mo_symmetry_
                                                                    [relative_gas_mo[4][p]];
                                                            }
                                                        }
                                                        for (size_t p = 0; p < gas6_size; ++p) {
                                                            det.set_alfa_bit(relative_gas_mo[5][p],
                                                                             det_gas6_a[p]);
                                                            det.set_beta_bit(relative_gas_mo[5][p],
                                                                             det_gas6_b[p]);
                                                            if (det_gas6_a[p]) {
                                                                sym ^= mo_symmetry_
                                                                    [relative_gas_mo[5][p]];
                                                            }
                                                            if (det_gas6_b[p]) {
                                                                sym ^= mo_symmetry_
                                                                    [relative_gas_mo[5][p]];
                                                            }
                                                        }
                                                        int nunpair =
                                                            nalpha_ + nbeta_ - 2 * det.npair();
                                                        // Check symmetry and multiplicity
                                                        if (sym == root_sym_ &&
                                                            nunpair + 1 >= multiplicity_) {
                                                            ref_space.push_back(det);
                                                            //                                                            outfile->Printf("\n");
                                                            //                                                            outfile->Printf(
                                                            //                                                                "\n  Ref: %s",
                                                            //                                                                str(det, nact_).c_str());
                                                            return;
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
        }
    }
}

void CI_Reference::build_gas_reference(std::vector<Determinant>& ref_space) {
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

    // loop over all GAS configurations
    timer timer_gas("Build GAS determinants");
    print_h2("Building GAS Determinants");
    for (size_t config = 0, size = gas_electrons_.size(); config < size; ++config) {

        // build alpha or beta strings (ngas of nirrep of vector of occupation)
        std::vector<std::vector<std::vector<std::vector<bool>>>> a_tmp, b_tmp;

        for (int gas = 0; gas < 6; ++gas) {
            auto space_name = "GAS" + std::to_string(gas + 1);
            auto norb = mo_space_info_->size(space_name);
            if (norb == 0)
                continue;

            auto sym = mo_space_info_->symmetry(space_name);

            a_tmp.emplace_back(build_occ_string(norb, gas_electrons_[config][2 * gas], sym));
            b_tmp.emplace_back(build_occ_string(norb, gas_electrons_[config][2 * gas + 1], sym));
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
        for (int ha = 0; ha < nirrep_; ++ha) {
            int hb = root_sym_ ^ ha;
            for (const auto& a : a_strings[ha]) {
                for (const auto& b : b_strings[hb]) {
                    ref_space.emplace_back(a, b);
                }
            }
        }
    }
    timer_gas.stop();

    outfile->Printf("\n  Size of GAS reference: %zu", ref_space.size());
    print_h2("GAS Determinants");
    for (const auto& det : ref_space) {
        outfile->Printf("\n    %s", str(det, nact_).c_str());
    }
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
    // Calculate gas_info from mo_space_info_
    //    std::pair<size_t, std::map<std::string, SpaceInfo>> gas_info =
    //        mo_space_info_->make_gas_info(options_);
    //    gas_num_ = gas_info.first;
    //    general_active_spaces_ = gas_info.second;

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
            size_t max_e_number = state_info_.gas_max()[gas_count];
            // If the defined maximum number of electrons exceed number of orbitals,
            // redefine maximum number of elctrons
            if (max_e_number > std::min(orbital_maximum * 2, nalpha_ + nbeta_)) {
                // outfile->Printf("\n  The maximum number of electrons in %s"
                //                "\n  either exceeds the number of spin orbitals"
                //                "\n  or total number of electrons or undefined!",
                //               space.c_str());
                gas_maxe.push_back(std::min(orbital_maximum * 2, nalpha_ + nbeta_));
                outfile->Printf("\n  The maximum number of electrons in %s"
                                " is %d ",
                                space.c_str(), std::min(orbital_maximum * 2, nalpha_ + nbeta_));
            } else {
                outfile->Printf("\n  The maximum number of electrons in "
                                "%s is %d",
                                space.c_str(), max_e_number);
                gas_maxe.push_back(max_e_number);
            }
            int min_e_number = state_info_.gas_min()[gas_count];
            gas_mine.push_back(min_e_number);
            outfile->Printf("\n  The minimum number of electrons in "
                            "%s is %d",
                            space.c_str(), min_e_number);
            gas_num_ = gas_num_ + 1;
        } else {
            gas_maxe.push_back(0);
            gas_mine.push_back(0);
        }
    }
    outfile->Printf("\n  ");
    outfile->Printf("\n  Possible electron occupations in the GAS \n  ");
    std::vector<std::string> gas_electron_name = {"GAS1_A", "GAS1_B", "GAS2_A", "GAS2_B",
                                                  "GAS3_A", "GAS3_B", "GAS4_A", "GAS4_B",
                                                  "GAS5_A", "GAS5_B", "GAS6_A", "GAS6_B"};
    for (size_t i = 0; i < 2 * gas_num_; i++) {
        outfile->Printf("%s  ", gas_electron_name.at(i).c_str());
    }

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
                                                std::vector<int> gas_configuration = {
                                                    gas1_na, gas1_nb, gas2_na, gas2_nb,
                                                    gas3_na, gas3_nb, gas4_na, gas4_nb,
                                                    gas5_na, gas5_nb, gas6_na, gas6_nb};
                                                outfile->Printf("\n  ");
                                                for (size_t i = 0; i < 2 * gas_num_; i++) {
                                                    outfile->Printf("   %d    ",
                                                                    gas_configuration[i]);
                                                }
                                                gas_electrons_.push_back(gas_configuration);
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
