/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2019 by its authors (see COPYING, COPYING.LESSER, AUTHORS).
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

#ifndef _reference_h_
#define _reference_h_

#include <ambit/tensor.h>

namespace forte {

class ForteIntegrals;
class MOSpaceInfo;

/**
 * @class Reference
 *
 * @brief This class stores diagonal or transition reduced density matrices (RDMs) and
 *        reduced density cumulants (RDCs).
 *
 * The n-body reduced density matrix between two states |A> and |B> is defined as
 *
 *  RDM(p_1, p_2, ..., p_n, q_1, q_2, ..., q_n) = <A|  |B>
 *
 * This class is constructed by passing the RDMs up to a given rank n (n <= 3). For example,
 * to pass the one- and two-body RDMs a Reference object is initialized as
 *
 * auto ref = Reference(g1a, g1b, g2aa, g2ab, g2bb);
 *
 * From the RDMs it is possible to obtain the corresponding cumulants. For example,
 * the alpha-alpha 2-body density cumulants can be obtained by calling
 *
 * auto L2aa = ref.L2aa();
 *
 * @note Once passed in, the RDMs are assumed to be fixed and immutable.
 *
 */
class Reference {
  public:
    // ==> Class Constructors <==

    /// 0-rdm constructor
    Reference();
    /// @brief Construct a reference object with the 1-rdm
    Reference(ambit::Tensor g1a, ambit::Tensor g1b);
    /// @brief Construct a reference object with the 1- and 2-rdms
    Reference(ambit::Tensor g1a, ambit::Tensor g1b, ambit::Tensor g2aa, ambit::Tensor g2ab,
              ambit::Tensor g2bb);
    /// @brief Construct a reference object with the 1-, 2-, and 3-rdms
    Reference(ambit::Tensor g1a, ambit::Tensor g1b, ambit::Tensor g2aa, ambit::Tensor g2ab,
              ambit::Tensor g2bb, ambit::Tensor g3aaa, ambit::Tensor g3aab, ambit::Tensor g3abb,
              ambit::Tensor g3bbb);

    // ==> Class Interface <==

    // Reduced density matrices (RDMs)

    /// @return the alpha 1-RDM
    ambit::Tensor g1a() const { return g1a_; }
    /// @return the beta 1-RDM
    ambit::Tensor g1b() const { return g1b_; }
    /// @return the alpha-alpha 2-RDM
    ambit::Tensor g2aa() const { return g2aa_; }
    /// @return the alpha-beta 2-RDM
    ambit::Tensor g2ab() const { return g2ab_; }
    /// @return the beta-beta 2-RDM
    ambit::Tensor g2bb() const { return g2bb_; }
    /// @return the alpha-alpha-alpha 3-RDM
    ambit::Tensor g3aaa() const { return g3aaa_; }
    /// @return the alpha-alpha-beta 3-RDM
    ambit::Tensor g3aab() const { return g3aab_; }
    /// @return the alpha-beta-beta 3-RDM
    ambit::Tensor g3abb() const { return g3abb_; }
    /// @return the beta-beta-beta 3-RDM
    ambit::Tensor g3bbb() const { return g3bbb_; }

    /// @return the spin-free 2-RDM
    ambit::Tensor SFg2() const { return SFg2_; }

    // Reduced density cumulants (RDCs)

    /// @return the alpha-alpha 2-RDC
    ambit::Tensor L2aa();
    /// @return the alpha-beta 2-RDC
    ambit::Tensor L2ab();
    /// @return the beta-beta 2-RDC
    ambit::Tensor L2bb();
    /// @return the alpha-alpha-alpha 3-RDC
    ambit::Tensor L3aaa();
    /// @return the alpha-alpha-beta 3-RDC
    ambit::Tensor L3aab();
    /// @return the alpha-beta-beta 3-RDC
    ambit::Tensor L3abb();
    /// @return the beta-beta-beta 3-RDC
    ambit::Tensor L3bbb();

  protected:
    // ==> Class Data <==

    /// Maximum RDM/RDC rank stored by this object
    size_t max_rdm_ = 0;

    // Reduced density matrices

    /// The alpha 1-RDM
    ambit::Tensor g1a_;
    /// The beta 1-RDM
    ambit::Tensor g1b_;
    /// The alpha-alpha 2-RDM
    ambit::Tensor g2aa_;
    /// The alpha-beta 2-RDM
    ambit::Tensor g2ab_;
    /// The beta-beta 2-RDM
    ambit::Tensor g2bb_;
    /// The alpha-alpha-alpha 3-RDM
    ambit::Tensor g3aaa_;
    /// The alpha-alpha-beta 3-RDM
    ambit::Tensor g3aab_;
    /// The alpha-beta-beta 3-RDM
    ambit::Tensor g3abb_;
    /// The beta-beta-beta 3-RDM
    ambit::Tensor g3bbb_;

    /// The spin-free 2-RDM
    ambit::Tensor SFg2_;

    // Reduced density cumulants

    /// The alpha-alpha 2-RDC
    ambit::Tensor L2aa_;
    /// The alpha-beta 2-RDC
    ambit::Tensor L2ab_;
    /// The beta-beta 2-RDC
    ambit::Tensor L2bb_;
    /// The alpha-alpha-alpha 3-RDC
    ambit::Tensor L3aaa_;
    /// The alpha-alpha-beta 3-RDC
    ambit::Tensor L3aab_;
    /// The alpha-beta-beta 3-RDC
    ambit::Tensor L3abb_;
    /// The beta-beta-beta 3-RDC
    ambit::Tensor L3bbb_;

    /// Was L2aa built?
    bool have_L2aa_ = false;
    /// Was L2ab built?
    bool have_L2ab_ = false;
    /// Was L2bb built?
    bool have_L2bb_ = false;
    /// Was L3aaa built?
    bool have_L3aaa_ = false;
    /// Was L3aab built?
    bool have_L3aab_ = false;
    /// Was L3abb built?
    bool have_L3abb_ = false;
    /// Was L3bbb built?
    bool have_L3bbb_ = false;
};

/**
 * @brief make_cumulant_L2aa_in_place Make the alpha-alpha 2-body cumulant.
 * This function replaces the tensor passed in (L2aa) containing the aa 2-RDM
 * with the cumulant.
 * @param g1a the alpha 1-RDM
 * @param L2aa the alpha-alpha 2-RDM
 */
void make_cumulant_L2aa_in_place(const ambit::Tensor& L1a, ambit::Tensor& L2aa);

/**

 * @brief make_cumulant_L2ab_in_place Make the alpha-beta 2-body cumulant.
 * This function replaces the tensor passed in (L2ab) containing the ab 2-RDM
 * with the cumulant.
 * @param g1a the alpha 1-RDM
 * @param g1b the beta 1-RDM
 * @param L2ab The beta-beta 2-RDM
 */
void make_cumulant_L2ab_in_place(const ambit::Tensor& L1a, const ambit::Tensor& L1b,
                                 ambit::Tensor& L2ab);

/**
 * @brief make_cumulant_L2bb_in_place Make the beta-beta 2-body cumulant.
 * This function replaces the tensor passed in (L2bb) containing the bb 2-RDM
 * with the cumulant.
 * @param g1b the beta 1-RDM
 * @param L2bb the beta-beta 2-RDM
 */
void make_cumulant_L2bb_in_place(const ambit::Tensor& L1b, ambit::Tensor& L2bb);

/**
 * @brief make_cumulant_L3aaa_in_place Make the aaa 3-body cumulant.
 * This function replaces the tensor passed in (L3aaa) containing the aaa 3-RDM
 * with the cumulant.
 * @param g1a alpha 1-RDM
 * @param L2aa alpha-alpha 2-RDC
 * @param L3aaa alpha-alpha-alpha 2-RDM
 */
void make_cumulant_L3aaa_in_place(const ambit::Tensor& g1a, const ambit::Tensor& L2aa,
                                  ambit::Tensor& L3aaa);

/**
 * @brief make_cumulant_L3aab_in_place Make the aab 3-body cumulant.
 * This function replaces the tensor passed in (L3aab) containing the aab 3-RDM
 * with the cumulant.
 * @param g1a alpha 1-RDM
 * @param g1b beta 1-RDM
 * @param L2aa alpha-alpha 2-RDC
 * @param L2ab alpha-beta 2-RDC
 * @param L3aab alpha-alpha-beta 3-RDM
 */
void make_cumulant_L3aab_in_place(const ambit::Tensor& g1a, const ambit::Tensor& g1b,
                                  const ambit::Tensor& L2aa, const ambit::Tensor& L2ab,
                                  ambit::Tensor& L3aab);

/**
 * @brief make_cumulant_L3abb_in_place Make the abb 3-body cumulant.
 * This function replaces the tensor passed in (L3abb) containing the abb 3-RDM
 * with the cumulant.
 * @param g1a alpha 1-RDM
 * @param g1b beta 1-RDM
 * @param L2ab alpha-beta 2-RDC
 * @param L2bb beta-beta 2-RDC
 * @param L3abb alpha-beta-beta 3-RDM
 */
void make_cumulant_L3abb_in_place(const ambit::Tensor& g1a, const ambit::Tensor& g1b,
                                  const ambit::Tensor& L2ab, const ambit::Tensor& L2bb,
                                  ambit::Tensor& L3abb);

/**
 * @brief make_cumulant_L3bbb_in_place Make the bbb 3-body cumulant.
 * This function replaces the tensor passed in (L3bbb) containing the bbb 3-RDM
 * with the cumulant.
 * @param g1b beta 1-RDM
 * @param L2bb beta-beta 2-RDC
 * @param L3bbb beta-beta-beta 2-RDM
 */
void make_cumulant_L3bbb_in_place(const ambit::Tensor& g1b, const ambit::Tensor& L2bb,
                                  ambit::Tensor& L3bbb);

/**
 * @brief compute_Eref_from_reference Compute the energy of a wave function from its
 * density matrices stored in the Reference object
 * @param ref the reference object
 * @param ints the integrals
 * @param mo_space_info information about the orbital spaces
 * @param Enuc the nucleaer repulsion energy
 * @return the reference energy
 */
double compute_Eref_from_reference(Reference& ref, std::shared_ptr<ForteIntegrals> ints,
                                   std::shared_ptr<MOSpaceInfo> mo_space_info);
} // namespace forte

#endif // _reference_h_
