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

#include <vector>

#include "psi4/libmints/dimension.h"
#include "ambit/tensor.h"

#include "base_classes/rdms.h"

#define CAPRICCIO_USE_DAXPY 1

namespace psi {
class Matrix;
class Vector;
} // namespace psi

namespace forte {
class ActiveSpaceIntegrals;
class MOSpaceInfo;
class StringLists;
class StringAddress;
class RDMs;

class FCIVector {
  public:
    FCIVector(std::shared_ptr<StringLists> lists, size_t symmetry);

    /// @brief return the number of irreps
    size_t nirrep() const;
    /// @brief return the symmetry of this vector
    size_t symmetry() const;
    /// @brief return the number of correlated molecular orbitals
    size_t ncmo() const;
    /// @brief return the size of the CI basis
    size_t size() const;
    /// @brief return the number of determinants per irrep
    const std::vector<size_t>& detpi() const;

    /// @brief return the number of correlated molecular orbitals per irrep
    psi::Dimension cmopi() const;
    /// @brief return the offset array for cmopi
    const std::vector<size_t>& cmopi_offset() const;
    const std::shared_ptr<StringLists>& lists() const;

    /// @brief zero the vector
    void zero();
    /// @brief print the vector
    void print();

    /// copy the wave function object
    void copy(FCIVector& wfn);
    /// copy the coefficient from a Vector object
    void copy(std::shared_ptr<psi::Vector> vec);
    /// copy the wave function object
    void copy_to(std::shared_ptr<psi::Vector> vec);
    /// @brief set the vector from a list of tuples
    /// @param sparse_vec a list of tuples (irrep, Ia, Ib, C)
    void set(std::vector<std::tuple<size_t, size_t, size_t, double>>& sparse_vec);

    /// @brief compute the norm of the wave function
    /// @param power The power of the norm (default 2 = Frobenius norm)
    double norm(double power = 2.0);

    /// @brief normalize the wave function
    void normalize();

    /// @brief Compute the dot product of this wave functions with another
    /// @param wfn The wave function to dot with
    /// @return The dot product
    double dot(FCIVector& wfn);
    double dot(std::shared_ptr<FCIVector>& wfn);

    // return alfa_address_
    std::shared_ptr<StringAddress> alfa_address() { return alfa_address_; }
    // return beta_address_
    std::shared_ptr<StringAddress> beta_address() { return beta_address_; }

    std::shared_ptr<psi::Matrix>& C(int irrep) { return C_[irrep]; }

    // Operations on the wave function
    void Hamiltonian(FCIVector& result, std::shared_ptr<ActiveSpaceIntegrals> fci_ints);

    double energy_from_rdms(std::shared_ptr<ActiveSpaceIntegrals> fci_ints);

    /// @brief Test the RDMs
    /// @param Cl the left state
    /// @param Cr the right state
    /// @param type the type of RDMs to test
    /// @param rdms the RDMs object to test
    /// @param max_rdm_level the maximum RDM level to test
    static void test_rdms(FCIVector& Cl, FCIVector& Cr, int max_rdm_level, RDMsType type,
                          std::shared_ptr<RDMs> rdms);

    /// Compute the expectation value of the S^2 operator
    double compute_spin2();

    /// Print the natural_orbitals from FCIWFN
    /// Assume user specified active space
    void print_natural_orbitals(std::shared_ptr<MOSpaceInfo>);

    /// Return the elements with the largest absolute value
    /// This function returns the tuple (|C_I|,C_I,irrep,Ia,Ib)
    std::vector<std::tuple<double, double, size_t, size_t, size_t>>
    max_abs_elements(size_t num_dets);

    // Temporary memory allocation
    static void allocate_temp_space(std::shared_ptr<StringLists> lists_, int print_);
    static void release_temp_space();
    void set_print(int print) { print_ = print; }

    // ==> Class Static Functions <==
    static std::shared_ptr<RDMs> compute_rdms(FCIVector& C_left, FCIVector& C_right, int max_order,
                                              RDMsType type);

    /// Return the temporary matrix CR
    static std::shared_ptr<psi::Matrix> get_CR();
    /// Return the temporary matrix CL
    static std::shared_ptr<psi::Matrix> get_CL();

  private:
    // ==> Class Data <==

    /// The number of irreps
    int nirrep_;
    /// The symmetry of this vector
    size_t symmetry_;
    /// The total number of correlated molecular orbitals
    size_t ncmo_;
    /// The number of correlated molecular orbitals per irrep
    psi::Dimension cmopi_;
    /// The offset array for cmopi_
    std::vector<size_t> cmopi_offset_;
    /// The number of determinants
    size_t ndet_;
    /// The number of determinants per irrep
    std::vector<size_t> detpi_;
    /// The print level
    int print_ = 0;

    /// The string list
    std::shared_ptr<StringLists> lists_;
    /// The alpha string addressing object
    std::shared_ptr<StringAddress> alfa_address_;
    /// The beta string addressing object
    std::shared_ptr<StringAddress> beta_address_;
    /// Coefficient matrix stored in block-matrix form
    std::vector<std::shared_ptr<psi::Matrix>> C_;

    // ==> Class Static Data <==

    // Temporary matrix of size as large as the largest block of C. Used to store the right
    // coefficient vector
    static std::shared_ptr<psi::Matrix> CR;
    // Temporary matrix of size as large as the largest block of C. Used to store the left
    // coefficient vector
    static std::shared_ptr<psi::Matrix> CL;

    // Timers
    static double hdiag_timer;
    static double h1_aa_timer;
    static double h1_bb_timer;
    static double h2_aaaa_timer;
    static double h2_aabb_timer;
    static double h2_bbbb_timer;

    // ==> Class Public Functions <==

    void startup();
    void cleanup();

    // ==> Class Private Functions <==

    static size_t oei_index(size_t p, size_t q, size_t ncmo) { return ncmo * p + q; }
    static size_t tei_index(size_t p, size_t q, size_t r, size_t s, size_t ncmo) {
        return ncmo * ncmo * ncmo * p + ncmo * ncmo * q + ncmo * r + s;
    }
    static size_t six_index(size_t p, size_t q, size_t r, size_t s, size_t t, size_t u,
                            size_t ncmo) {
        return (ncmo * ncmo * ncmo * ncmo * ncmo * p + ncmo * ncmo * ncmo * ncmo * q +
                ncmo * ncmo * ncmo * r + ncmo * ncmo * s + ncmo * t + u);
    }

    void H0(FCIVector& result, std::shared_ptr<ActiveSpaceIntegrals> fci_ints);
    void H1(FCIVector& result, std::shared_ptr<ActiveSpaceIntegrals> fci_ints, bool alfa);
    void H2_aabb(FCIVector& result, std::shared_ptr<ActiveSpaceIntegrals> fci_ints);
    void H2_aaaa2(FCIVector& result, std::shared_ptr<ActiveSpaceIntegrals> fci_ints, bool alfa);

    // 1-RDM elements are stored in the format
    // <a^+_{pa} a^+_{qb} a_{sb} a_ra> -> rdm[oei_index(p,q)]

    /// Compute the matrix elements of the same 1-RDM <a^+_{p} a_{q}>
    static ambit::Tensor compute_1rdm_same_irrep(FCIVector& C_left, FCIVector& C_right, bool alfa);

    // 2-RDM elements are stored in the format
    // <a^+_{p} a^+_{q} a_{s} a_r> -> rdm[tei_index(p,q,r,s)]

    /// Compute the matrix elements of the same spin 2-RDM <a^+_p a^+_q a_s a_r> (with all
    /// indices alpha or beta)
    static ambit::Tensor compute_2rdm_aa_same_irrep(FCIVector& C_left, FCIVector& C_right,
                                                    bool alfa);
    /// Compute the matrix elements of the alpha-beta 2-RDM <a^+_{pa} a^+_{qb} a_{sb} a_{ra}>
    static ambit::Tensor compute_2rdm_ab_same_irrep(FCIVector& C_left, FCIVector& C_right);

    // 3-RDM elements are stored in the format
    // <a^+_p a^+_q a^+_r a_u a_t a_s> -> rdm[six_index(p,q,r,s,t,u)]

    /// Compute the matrix elements of the same spin 3-RDM <a^+_p a^+_q a_s a_r> (with all indices
    /// alpha or beta)
    static ambit::Tensor compute_3rdm_aaa_same_irrep(FCIVector& C_left, FCIVector& C_right,
                                                     bool alfa);
    /// Compute the matrix elements of the alpha-alpha-beta 3-RDM <a^+_{pa} a^+_{qa} a^+_{rb} a_{ub}
    /// a_{ta} a_{sa}>
    static ambit::Tensor compute_3rdm_aab_same_irrep(FCIVector& C_left, FCIVector& C_right);
    /// Compute the matrix elements of the alpha-beta-beta 3-RDM <a^+_{pa} a^+_{qb} a^+_{rb} a_{ub}
    /// a_{tb} a_{sa}>
    static ambit::Tensor compute_3rdm_abb_same_irrep(FCIVector& C_left, FCIVector& C_right);
};

void fill_C_block(FCIVector& C, double** m, bool alfa, std::shared_ptr<StringAddress> alfa_address,
                  std::shared_ptr<StringAddress> beta_address, int ha, int hb);

std::shared_ptr<RDMs> compute_transition_rdms(FCIVector& C_left, FCIVector& C_right,
                                              int max_rdm_level, RDMsType type);

ambit::Tensor compute_1rdm_different_irrep(FCIVector& C_left, FCIVector& C_right, bool alfa);

} // namespace forte
