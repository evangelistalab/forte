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

#ifndef _string_lists_
#define _string_lists_

#include "psi4/libmints/dimension.h"

#include <map>
#include <vector>
#include <utility>

#include "binary_graph.hpp"
#include "helpers/timer.h"
#include "sparse_ci/determinant.h"

namespace forte {

/// A structure to store how the string J is connected to the string I and the corresponding sign
/// I -> sign J
/// The uint32_t will hold up to 4,294,967,296 elements (that should be enough)
struct StringSubstitution {
    const double sign;
    const uint32_t I;
    const uint32_t J;
    StringSubstitution(const double& sign_, const uint32_t& I_, const uint32_t& J_)
        : sign(sign_), I(I_), J(J_) {}
};

/// 1-hole string substitution
struct H1StringSubstitution {
    const int16_t sign;
    const int16_t p;
    const size_t J;
    H1StringSubstitution(int16_t sign_, int16_t p_, size_t J_) : sign(sign_), p(p_), J(J_) {}
};

/// 2-hole string substitution
struct H2StringSubstitution {
    const int16_t sign;
    const int16_t p;
    const int16_t q;
    size_t J;
    H2StringSubstitution(int16_t sign_, int16_t p_, int16_t q_, size_t J_)
        : sign(sign_), p(p_), q(q_), J(J_) {}
};

/// 3-hole string substitution
struct H3StringSubstitution {
    const int16_t sign;
    const int16_t p;
    const int16_t q;
    const int16_t r;
    const size_t J;
    H3StringSubstitution(int16_t sign_, int16_t p_, int16_t q_, int16_t r_, size_t J_)
        : sign(sign_), p(p_), q(q_), r(r_), J(J_) {}
};

using StringList = std::vector<std::vector<String>>;

/// Maps the integers (p,q,h) to list of strings connected by a^{+}_p a_q, where the string
/// I belongs to the irrep h
using VOList = std::map<std::tuple<size_t, size_t, int>, std::vector<StringSubstitution>>;

/// Maps the integers (p,q,r,s,h) to list of strings connected by a^{+}_p a^{+}_q a_s a_r, where the
/// string I belongs to the irrep h
using VVOOList =
    std::map<std::tuple<size_t, size_t, size_t, size_t, int>, std::vector<StringSubstitution>>;

/// Maps the integers (pq_sym, pq, h) to list of strings connected by a^{+}_p a^{+}_q a_q a_p where
/// the string I belongs to the irrep h
using OOList = std::map<std::tuple<int, size_t, int>, std::vector<StringSubstitution>>;

/// Maps the integers (h_J, add_J, h_I) to list of strings connected by a_p, where the string
/// I belongs to the irrep h_I and J belongs to the irrep h_J and add_J is the address of J
using H1List = std::map<std::tuple<int, size_t, int>, std::vector<H1StringSubstitution>>;

/// Maps the integers (h_J, add_J, h_I) to list of strings connected by a_p a_q, where the string
/// I belongs to the irrep h_I and J belongs to the irrep h_J and add_J is the address of J
using H2List = std::map<std::tuple<int, size_t, int>, std::vector<H2StringSubstitution>>;

/// Maps the integers (h_J, add_J, h_I) to list of strings connected by a_p a_q a_r, where the
/// string I belongs to the irrep h_I and J belongs to the irrep h_J and add_J is the address of J
using H3List = std::map<std::tuple<int, size_t, int>, std::vector<H3StringSubstitution>>;

using Pair = std::pair<int, int>;
using PairList = std::vector<std::vector<std::pair<int, int>>>;

class StringAddress;
class StringClass;

/**
 * @brief The StringLists class
 *
 * This class computes mappings between alpha/beta strings
 */
class StringLists {
  public:
    // ==> Constructor and Destructor <==

    /// @brief Constructor
    /// @param cmopi number of correlated MOs per irrep
    /// @param core_mo core MOs
    /// @param cmo_to_mo mapping from correlated MOs to MOs
    /// @param na number of alpha electrons
    /// @param nb number of beta electrons
    /// @param print print level
    StringLists(psi::Dimension cmopi, std::vector<size_t> core_mo, std::vector<size_t> cmo_to_mo,
                size_t na, size_t nb, int print);

    ~StringLists() {}

    // ==> Class Public Functions <==

    /// @return the number of alpha electrons
    size_t na() const { return na_; }

    /// @return the number of beta electrons
    size_t nb() const { return nb_; }

    /// @return the number of irreps
    int nirrep() const { return nirrep_; }

    /// @return the number of correlated MOs
    size_t ncmo() const { return ncmo_; }

    /// @return the mapping from correlated MOs to MOs
    std::vector<size_t> cmo_to_mo() const { return cmo_to_mo_; }

    /// @return the mapping from frozen MOs to correlated MOs
    std::vector<size_t> fomo_to_mo() const { return fomo_to_mo_; }

    /// @return the number of correlated MOs per irrep
    psi::Dimension cmopi() const { return cmopi_; }

    /// @return the offset of correlated MOs per irrep
    std::vector<size_t> cmopi_offset() const { return cmopi_offset_; }

    /// @return the number of pairs per irrep
    size_t pairpi(int h) const { return pairpi_[h]; }

    auto alfa_address() { return alfa_address_; }
    auto beta_address() { return beta_address_; }
    auto alfa_address_1h() { return alfa_address_1h_; }
    auto beta_address_1h() { return beta_address_1h_; }
    auto alfa_address_2h() { return alfa_address_2h_; }
    auto beta_address_2h() { return beta_address_2h_; }
    auto alfa_address_3h() { return alfa_address_3h_; }
    auto beta_address_3h() { return beta_address_3h_; }

    /// @return the alpha string list
    const auto& alfa_strings() const { return alfa_strings_; }
    /// @return the beta string list
    const auto& beta_strings() const { return beta_strings_; }
    /// @return the alpha string in irrep h and index I
    String alfa_str(size_t h, size_t I) const { return alfa_strings_[h][I]; }
    /// @return the beta string in irrep h and index I
    String beta_str(size_t h, size_t I) const { return beta_strings_[h][I]; }

    std::vector<Determinant> make_determinants(int symmetry) const;

    std::vector<StringSubstitution>& get_alfa_vo_list(size_t p, size_t q, int h);
    std::vector<StringSubstitution>& get_beta_vo_list(size_t p, size_t q, int h);

    std::vector<H1StringSubstitution>& get_alfa_1h_list(int h_I, size_t add_I, int h_J);
    std::vector<H1StringSubstitution>& get_beta_1h_list(int h_I, size_t add_I, int h_J);

    std::vector<H2StringSubstitution>& get_alfa_2h_list(int h_I, size_t add_I, int h_J);
    std::vector<H2StringSubstitution>& get_beta_2h_list(int h_I, size_t add_I, int h_J);

    std::vector<H3StringSubstitution>& get_alfa_3h_list(int h_I, size_t add_I, int h_J);
    std::vector<H3StringSubstitution>& get_beta_3h_list(int h_I, size_t add_I, int h_J);

    std::vector<StringSubstitution>& get_alfa_oo_list(int pq_sym, size_t pq, int h);
    std::vector<StringSubstitution>& get_beta_oo_list(int pq_sym, size_t pq, int h);

    std::vector<StringSubstitution>& get_alfa_vvoo_list(size_t p, size_t q, size_t r, size_t s,
                                                        int h);
    std::vector<StringSubstitution>& get_beta_vvoo_list(size_t p, size_t q, size_t r, size_t s,
                                                        int h);

    Pair get_pair_list(int h, int n) const { return pair_list_[h][n]; }

  private:
    // ==> Class Data <==

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
    /// The mapping between correlated molecular orbitals and all orbitals
    std::vector<size_t> cmo_to_mo_;
    /// The mapping between frozen occupied molecular orbitals and all orbitals
    std::vector<size_t> fomo_to_mo_;
    /// The number of alpha electrons
    size_t na_;
    /// The number of beta electrons
    size_t nb_;
    /// The number of alpha strings
    size_t nas_;
    /// The number of beta strings
    size_t nbs_;
    /// The total number of orbital pairs per irrep
    std::vector<int> pairpi_;
    /// The offset array for pairpi
    std::vector<int> pair_offset_;
    /// The print level
    int print_ = 0;

    // String lists
    std::shared_ptr<StringClass> string_class_;
    /// The alpha strings stored by irrep and address
    StringList alfa_strings_;
    /// The beta strings stored by irrep and address
    StringList beta_strings_;
    /// The pair string list
    PairList pair_list_;
    /// The VO string lists
    VOList alfa_vo_list;
    VOList beta_vo_list;
    /// The OO string lists
    OOList alfa_oo_list;
    OOList beta_oo_list;
    /// The VVOO string lists
    VVOOList alfa_vvoo_list;
    VVOOList beta_vvoo_list;
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
    void startup();

    /// Make strings of for norb bits with ne of these set to 1 and (norb - ne) set to 0
    /// @return strings sorted according to their irrep
    StringList make_fci_strings(const int norb, const int ne);

    /// Make the string list
    void make_strings(std::shared_ptr<StringAddress> graph, StringList& list);

    /// Make the pair list
    void make_pair_list(PairList& list);

    /// Make the VO list
    void make_vo_list(std::shared_ptr<StringAddress> graph, VOList& list);
    void make_vo(std::shared_ptr<StringAddress> graph, VOList& list, int p, int q);

    /// @brief Make the list of strings connected by a^{+}_p a^{+}_q a_q a_p
    void make_oo_list(std::shared_ptr<StringAddress> graph, OOList& list);

    /// @brief Make the list of strings connected by a^{+}_p a^{+}_q a_q a_p
    /// @param pq_sym symmetry of the pq pair
    /// @param pq relative pair index of the pq pair
    void make_oo(std::shared_ptr<StringAddress> address, OOList& list, int pq_sym, size_t pq);

    /// Make 1-hole lists (I -> a_p I = sgn J)
    void make_1h_list(std::shared_ptr<StringAddress> graph, std::shared_ptr<StringAddress> graph_1h,
                      H1List& list);
    /// Make 2-hole lists (I -> a_p a_q I = sgn J)
    void make_2h_list(std::shared_ptr<StringAddress> graph, std::shared_ptr<StringAddress> graph_2h,
                      H2List& list);
    /// Make 3-hole lists (I -> a_p a_q a_r I = sgn J)
    void make_3h_list(std::shared_ptr<StringAddress> graph, std::shared_ptr<StringAddress> graph_3h,
                      H3List& list);

    /// Make the VVOO list
    void make_vvoo_list(std::shared_ptr<StringAddress> graph, VVOOList& list);
    void make_vvoo(std::shared_ptr<StringAddress> graph, VVOOList& list, int p, int q, int r,
                   int s);
};
} // namespace forte

#endif // _string_lists_
