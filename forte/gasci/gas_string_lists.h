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

#pragma once

#include "psi4/libmints/dimension.h"

#include <map>
#include <vector>
#include <utility>

#include "helpers/timer.h"
#include "sparse_ci/determinant.h"
#include "fci/string_list_defs.h"
#include "gas_string_address.h"

namespace forte {

class StringAddress;
class MOSpaceInfo;

/**
 * @brief The GASStringLists class
 *
 * This class computes mappings between alpha/beta strings
 */
class GASStringLists {
  public:
    // ==> Constructor and Destructor <==

    /// @brief Constructor
    /// @param cmopi number of correlated MOs per irrep
    /// @param core_mo core MOs
    /// @param cmo_to_mo mapping from correlated MOs to MOs
    /// @param na number of alpha electrons
    /// @param nb number of beta electrons
    /// @param print print level
    /// @param gas_size size of the GAS spaces
    GASStringLists(std::shared_ptr<MOSpaceInfo> mo_space_info, size_t na, size_t nb, int symmetry,
                   int print, const std::vector<int> gas_min, const std::vector<int> gas_max);

    ~GASStringLists() {}

    // ==> Class Public Functions <==

    /// @return the number of alpha electrons
    size_t na() const { return na_; }

    /// @return the number of beta electrons
    size_t nb() const { return nb_; }

    /// @return the symmetry of the state
    int symmetry() const { return symmetry_; }

    /// @return the number of irreps
    int nirrep() const { return nirrep_; }

    /// @return the number of correlated MOs
    size_t ncmo() const { return ncmo_; }

    /// @return the number of correlated MOs per irrep
    psi::Dimension cmopi() const { return cmopi_; }

    /// @return the offset of correlated MOs per irrep
    std::vector<size_t> cmopi_offset() const { return cmopi_offset_; }

    /// @return the number of pairs per irrep
    size_t pairpi(int h) const { return pairpi_[h]; }

    /// @return the alpha string address object
    auto alfa_address() { return alfa_address_; }
    /// @return the beta string address object
    auto beta_address() { return beta_address_; }
    /// @return the alpha string address object for N - 1 electrons
    auto alfa_address_1h() { return alfa_address_1h_; }
    /// @return the beta string address object for N - 1 electrons
    auto beta_address_1h() { return beta_address_1h_; }
    /// @return the alpha string address object for N - 2 electrons
    auto alfa_address_2h() { return alfa_address_2h_; }
    /// @return the beta string address object for N - 2 electrons
    auto beta_address_2h() { return beta_address_2h_; }
    /// @return the alpha string address object for N - 3 electrons
    auto alfa_address_3h() { return alfa_address_3h_; }
    /// @return the beta string address object for N - 3 electrons
    auto beta_address_3h() { return beta_address_3h_; }

    /// @return the address of a determinant in the CI vector
    size_t determinant_address(const Determinant& d) const;
    /// @return the determinant corresponding to an address in the CI vector of a given symmetry
    Determinant determinant(size_t address, size_t symmetry) const;

    /// @return the alpha string list
    const auto& alfa_strings() const { return alfa_strings_; }
    /// @return the beta string list
    const auto& beta_strings() const { return beta_strings_; }
    /// @return the alpha string in irrep h and index I
    String alfa_str(size_t h, size_t I) const { return alfa_strings_[h][I]; }
    /// @return the beta string in irrep h and index I
    String beta_str(size_t h, size_t I) const { return beta_strings_[h][I]; }

    /// @return the alpha string classes
    const auto& alfa_string_classes() const { return string_class_->alfa_string_classes(); }
    /// @return the beta string classes
    const auto& beta_string_classes() const { return string_class_->beta_string_classes(); }
    /// @return the alpha/beta string classes
    const auto& string_classes() const { return string_class_->string_classes(); }

    /// @return the list of determinants with a given symmetry
    std::vector<Determinant> make_determinants() const;

    std::vector<StringSubstitution>& get_alfa_vo_list(size_t p, size_t q, int class_I, int class_J);
    std::vector<StringSubstitution>& get_beta_vo_list(size_t p, size_t q, int class_I, int class_J);

    std::vector<H1StringSubstitution>& get_alfa_1h_list(int h_I, size_t add_I, int h_J);
    std::vector<H1StringSubstitution>& get_beta_1h_list(int h_I, size_t add_I, int h_J);

    std::vector<H2StringSubstitution>& get_alfa_2h_list(int h_I, size_t add_I, int h_J);
    std::vector<H2StringSubstitution>& get_beta_2h_list(int h_I, size_t add_I, int h_J);

    std::vector<H3StringSubstitution>& get_alfa_3h_list(int h_I, size_t add_I, int h_J);
    std::vector<H3StringSubstitution>& get_beta_3h_list(int h_I, size_t add_I, int h_J);

    std::vector<u_int32_t>& get_alfa_oo_list(int pq_sym, size_t pq, int h);
    std::vector<u_int32_t>& get_beta_oo_list(int pq_sym, size_t pq, int h);

    std::vector<StringSubstitution>& get_alfa_vvoo_list(size_t p, size_t q, size_t r, size_t s,
                                                        int class_I, int class_J);
    std::vector<StringSubstitution>& get_beta_vvoo_list(size_t p, size_t q, size_t r, size_t s,
                                                        int class_I, int class_J);

    Pair get_pair_list(int h, int n) const { return pair_list_[h][n]; }

  private:
    // ==> Class Data <==

    /// The symmetry of the state
    const int symmetry_;
    /// The number of irreps
    const size_t nirrep_;
    /// The total number of correlated molecular orbitals
    const size_t ncmo_;
    /// The number of correlated molecular orbitals per irrep
    psi::Dimension cmopi_;
    /// The symmetry of the correlated molecular orbitals
    std::vector<int> cmo_sym_;
    /// The offset array for cmopi_
    std::vector<size_t> cmopi_offset_;
    /// The number of alpha electrons
    size_t na_;
    /// The number of beta electrons
    size_t nb_;
    /// The number of alpha strings
    size_t nas_;
    /// The number of beta strings
    size_t nbs_;
    /// The number of determinants
    size_t ndet_ = 0;
    /// The total number of orbital pairs per irrep
    std::vector<int> pairpi_;
    /// The offset array for pairpi
    std::vector<int> pair_offset_;
    /// The print level
    int print_ = 0;

    // GAS specific data

    std::vector<std::array<int, 6>> gas_alfa_occupations_;
    std::vector<std::array<int, 6>> gas_beta_occupations_;
    std::vector<std::array<int, 6>> gas_alfa_1h_occupations_;
    std::vector<std::array<int, 6>> gas_beta_1h_occupations_;
    std::vector<std::array<int, 6>> gas_alfa_2h_occupations_;
    std::vector<std::array<int, 6>> gas_beta_2h_occupations_;
    std::vector<std::array<int, 6>> gas_alfa_3h_occupations_;
    std::vector<std::array<int, 6>> gas_beta_3h_occupations_;

    std::vector<std::pair<size_t, size_t>> gas_occupations_;

    /// The number of GAS spaces used
    size_t ngas_spaces_;
    /// The position of the GAS spaces in the active space. For example, gas_mos_[0] gives the
    /// position of the MOs in the first GAS space in the active space
    std::vector<std::vector<size_t>> gas_mos_;
    /// The size of the GAS spaces (ignoring symmetry)
    std::vector<int> gas_size_;
    /// The minimum number of electrons in each GAS space
    std::vector<int> gas_min_;
    /// The maximum number of electrons in each GAS space
    std::vector<int> gas_max_;

    void recursive_gas_generation(const std::vector<int>& gas_mine,
                                  const std::vector<int>& gas_maxe, size_t gas_num, int& n_config,
                                  std::vector<int> gas_configuration, size_t gas_count);

    std::vector<std::array<int, 6>>
    generate_1h_occupations(const std::vector<std::array<int, 6>>& gas_occupations);

    // String lists
    std::shared_ptr<StringClass> string_class_;
    /// The alpha strings stored by class and address
    StringList alfa_strings_;
    /// The beta strings stored by class and address
    StringList beta_strings_;
    /// The pair string list
    PairList pair_list_;
    /// The VO string lists
    VOList2 alfa_vo_list;
    VOList2 beta_vo_list;
    /// The OO string lists
    OOList2 alfa_oo_list;
    OOList2 beta_oo_list;
    /// The VVOO string lists
    VVOOList2 alfa_vvoo_list;
    VVOOList2 beta_vvoo_list;
    /// The 1-hole lists
    H1List alfa_1h_list;
    H1List beta_1h_list;
    /// The 2-hole lists
    H2List alfa_2h_list;
    H2List beta_2h_list;
    /// The 3-hole lists
    H3List alfa_3h_list;
    H3List beta_3h_list;

    /// Addressers
    /// The alpha string address
    std::shared_ptr<StringAddress> alfa_address_;
    /// The beta string address
    std::shared_ptr<StringAddress> beta_address_;
    /// The alpha string graph for N - 1 electrons
    std::shared_ptr<StringAddress> alfa_address_1h_;
    /// The beta string graph for N - 1 electrons
    std::shared_ptr<StringAddress> beta_address_1h_;
    /// The alpha string graph for N - 2 electrons
    std::shared_ptr<StringAddress> alfa_address_2h_;
    /// The beta string graph for N - 2 electrons
    std::shared_ptr<StringAddress> beta_address_2h_;
    /// The alpha string graph for N - 3 electrons
    std::shared_ptr<StringAddress> alfa_address_3h_;
    /// The beta string graph for N - 3 electrons
    std::shared_ptr<StringAddress> beta_address_3h_;

    // ==> Class Functions <==

    /// Startup the class
    void startup(std::shared_ptr<MOSpaceInfo> mo_space_info);

    /// Make strings of for norb bits with ne of these set to 1 and (norb - ne) set to 0
    /// @return strings sorted according to their irrep
    StringList make_gas_strings(const std::vector<int>& gas_size,
                                const std::vector<std::array<int, 6>>& gas_occupations);

    void make_gas_strings_with_occupation(StringList& list, const std::vector<int>& gas_size,
                                          const std::array<int, 6>& gas_occupation);
    /// Make the pair list
    PairList make_pair_list();

    /// Make the VO list
    void make_vo_list(const StringList& strings, std::shared_ptr<StringAddress> addresser,
                      VOList2& list);
    void make_vo(const StringList& strings, std::shared_ptr<StringAddress> addresser, VOList2& list,
                 int p, int q);

    /// @brief Make the list of strings connected by a^{+}_p a^{+}_q a_q a_p
    void make_oo_list(const StringList& strings, std::shared_ptr<StringAddress> graph,
                      OOList2& list);

    /// @brief Make the list of strings connected by a^{+}_p a^{+}_q a_q a_p
    /// @param pq_sym symmetry of the pq pair
    /// @param pq relative pair index of the pq pair
    void make_oo(const StringList& strings, std::shared_ptr<StringAddress> address, OOList2& list,
                 int pq_sym, size_t pq);

    /// Make 1-hole lists (I -> a_p I = sgn J)
    void make_1h_list(const StringList& strings, std::shared_ptr<StringAddress> graph,
                      std::shared_ptr<StringAddress> graph_1h, H1List& list);
    /// Make 2-hole lists (I -> a_p a_q I = sgn J)
    void make_2h_list(const StringList& strings, std::shared_ptr<StringAddress> graph,
                      std::shared_ptr<StringAddress> graph_2h, H2List& list);
    /// Make 3-hole lists (I -> a_p a_q a_r I = sgn J)
    void make_3h_list(const StringList& strings, std::shared_ptr<StringAddress> graph,
                      std::shared_ptr<StringAddress> graph_3h, H3List& list);

    /// Make the VVOO list
    void make_vvoo_list(const StringList& strings, std::shared_ptr<StringAddress> graph,
                        VVOOList2& list);
    void make_vvoo(const StringList& strings, std::shared_ptr<StringAddress> graph, VVOOList2& list,
                   int p, int q, int r, int s);

    void get_gas_occupation();
};
} // namespace forte
