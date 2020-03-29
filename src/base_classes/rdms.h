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

#ifndef _rdms_h_
#define _rdms_h_

#include <vector>
#include <ambit/tensor.h>

namespace forte {

class ForteIntegrals;
class MOSpaceInfo;

/**
 * @class RDMs
 *
 * @brief This class stores diagonal or transition reduced density matrices (RDMs) and
 *        reduced density cumulants (RDCs).
 *
 * The n-body reduced density matrix between two states |A> and |B> is defined as
 *
 *  RDM(p_1, p_2, ..., p_n, q_1, q_2, ..., q_n) = <A| a+(p1) ... a+(pn) a(qn) ... a(q1)  |B>
 *
 * This class is constructed by passing the RDMs up to a given rank n (n <= 3). For example,
 * to pass the one- and two-body RDMs an object is initialized as
 *
 * ambit::Tensor g1a, g1b, g2aa, g2ab, g2bb;
 * // ...
 * // fill in g1a, g1b, ...
 * // ...
 * auto rdms = RDMs(g1a, g1b, g2aa, g2ab, g2bb);
 *
 * From the RDMs it is possible to obtain the corresponding cumulants by calling appropriate member
 * functions. For example,
 * the alpha-alpha 2-body density cumulant can be obtained by calling
 *
 * auto L2aa = rdms.L2aa();
 *
 * @note Once passed in, the RDMs are assumed to be fixed and immutable.
 *
 */
class RDMs {
  public:
    // ==> Class Constructors <==

    /// 0-rdm constructor
    RDMs();
    /// @brief Construct a RDMs object with the 1-rdm
    RDMs(ambit::Tensor g1a, ambit::Tensor g1b);
    /// @brief Construct a RDMs object with the 1- and 2-rdms
    RDMs(ambit::Tensor g1a, ambit::Tensor g1b, ambit::Tensor g2aa, ambit::Tensor g2ab,
         ambit::Tensor g2bb);
    /// @brief Construct a RDMs object with the 1-, 2-, and 3-rdms
    RDMs(ambit::Tensor g1a, ambit::Tensor g1b, ambit::Tensor g2aa, ambit::Tensor g2ab,
         ambit::Tensor g2bb, ambit::Tensor g3aaa, ambit::Tensor g3aab, ambit::Tensor g3abb,
         ambit::Tensor g3bbb);

    /// @brief Construct a RDMs object with the 1-rdm assuming ms averaging
    RDMs(bool ms_avg, ambit::Tensor g1a);
    /// @brief Construct a RDMs object with the 1- and 2-rdms ms averaging
    RDMs(bool ms_avg, ambit::Tensor g1a, ambit::Tensor g2ab);
    /// @brief Construct a RDMs object with the 1-, 2-, and 3-rdms assuming ms averaging
    RDMs(bool ms_avg, ambit::Tensor g1a, ambit::Tensor g2ab, ambit::Tensor g3aab);

    // ==> Class Interface <==

    // Reduced density matrices (RDMs)

    /// @return the alpha 1-RDM
    ambit::Tensor g1a() const { return g1a_; }
    /// @return the beta 1-RDM
    ambit::Tensor g1b();
    /// @return the alpha-alpha 2-RDM
    ambit::Tensor g2aa();
    /// @return the alpha-beta 2-RDM
    ambit::Tensor g2ab() const { return g2ab_; }
    /// @return the beta-beta 2-RDM
    ambit::Tensor g2bb();
    /// @return the alpha-alpha-alpha 3-RDM
    ambit::Tensor g3aaa();
    /// @return the alpha-alpha-beta 3-RDM
    ambit::Tensor g3aab() const { return g3aab_; }
    /// @return the alpha-beta-beta 3-RDM
    ambit::Tensor g3abb();
    /// @return the beta-beta-beta 3-RDM
    ambit::Tensor g3bbb();

    // Spin-free (spin-summed) RDMs

    /// @return the spin-free 2-RDM
    /// If ms is NOT averaged, G2 will be computed using the definition (see SF_g2_ below).
    /// If ms is averaged, G2 will be computed using only g2ab to avoid computing g2aa and g2bb.
    ambit::Tensor SFg2();

    // Reduced density cumulants

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

    // Spin-free (spin-summed) density cumulants

    /// @return the spin-free 1-cumulant
    ambit::Tensor SF_L1();
    /// @return the spin-free 2-cumulant
    ambit::Tensor SF_L2();
    /// @return the spin-free 3-cumulant
    ambit::Tensor SF_L3();

    // class variables

    size_t max_rdm_level() { return max_rdm_; }

    bool ms_avg() { return ms_avg_; }

  protected:
    // ==> Class Data <==

    /// Assume averaging over spin multiplets
    bool ms_avg_ = false;

    /// Maximum RDM/RDC rank stored by this object
    size_t max_rdm_ = 0;

    // Reduced density matrices

    /// Was g1b built?
    bool have_g1b_ = false;
    /// Was g2aa built?
    bool have_g2aa_ = false;
    /// Was g2bb built?
    bool have_g2bb_ = false;
    /// Was g3aaa built?
    bool have_g3aaa_ = false;
    /// Was g3abb built?
    bool have_g3abb_ = false;
    /// Was g3bbb built?
    bool have_g3bbb_ = false;

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

    /// Spin-free (spin-summed) 2-RDM defined as
    /// G2[pqrs] = g2aa[pqrs] + g2ab[pqrs] + g2ab[qpsr] + g2bb[pqrs]
    ambit::Tensor SF_g2_;

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

    /// Spin-free (spin-summed) 1-cumulant
    ambit::Tensor SF_L1_;
    /// Spin-free (spin-summed) 2-cumulant
    ambit::Tensor SF_L2_;
    /// Spin-free (spin-summed) 3-cumulant
    ambit::Tensor SF_L3_;

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

    /// Was SF_L1_ built?
    bool have_SF_L1_ = false;
    /// Was SF_L2_ built?
    bool have_SF_L2_ = false;
    /// Was SF_L3_ built?
    bool have_SF_L3_ = false;
};

/**
 * @brief make_g2_high_spin_case Make the alpha-alpha or beta-beta 2-RDM from alpha-beta 2-RDM.
 * This function returns the aa or bb 2-RDM using the ab 2-RDM assuming ms averaging.
 * @param g2ab the alpha-beta 2-RDM
 * @return the alpha-alpha or beta-beta 2-RDM
 */
ambit::Tensor make_g2_high_spin_case(const ambit::Tensor& g2ab);

/**
 * @brief make_g3_high_spin_case Make the aaa or bbb 3-RDM from aab 3-RDM.
 * This function returns the aaa or bbb 3-RDM using the aab 3-RDM assuming ms averaging.
 * @param g3aab the alpha-alpha-beta 3-RDM
 * @return the alpha-alpha-alpha or beta-beta-beta 3-RDM
 */
ambit::Tensor make_g3_high_spin_case(const ambit::Tensor& g3aab);

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
 * @brief compute_Eref_from_rdms Compute the energy of a wave function from its
 * density matrices stored in the RDMs object
 * @param ref the reference object
 * @param ints the integrals
 * @param mo_space_info information about the orbital spaces
 * @param Enuc the nucleaer repulsion energy
 * @return the reference energy
 */
double compute_Eref_from_rdms(RDMs& ref, std::shared_ptr<ForteIntegrals> ints,
                              std::shared_ptr<MOSpaceInfo> mo_space_info);
} // namespace forte

#endif // _reference_h_
