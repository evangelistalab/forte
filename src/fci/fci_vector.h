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

#ifndef _fci_vector_
#define _fci_vector_

#include <vector>

#include "psi4/libmints/matrix.h"
#include "fci_integrals.h"
#include "../integrals/integrals.h"
#include "string_lists.h"

#define CAPRICCIO_USE_DAXPY 1
#define CAPRICCIO_USE_UNROLL 0

namespace psi {
namespace forte {

class FCIVector {
  public:
    FCIVector(std::shared_ptr<StringLists> lists, size_t symmetry);
    ~FCIVector();

    /// Print this vector
    void print();
    /// Zero this vector
    void zero();
    /// Return the size of the CI basis
    size_t size() const { return ndet_; }

    std::vector<SharedMatrix> C() {return C_;}

    /// Copy the wave function object
    void copy(FCIVector& wfn);
    /// Copy the coefficients from a Vector object
    void copy(SharedVector vec);
    /// Copy the coefficients to a Vector object
    void copy_to(SharedVector vec);

    /// Form the diagonal part of the Hamiltonian
    void form_H_diagonal(std::shared_ptr<FCIIntegrals> fci_ints);

    /// Set the value from a vector of (irrep, string a, string b, coefficient) tuples
    void set(std::vector<std::tuple<size_t, size_t, size_t, double>>& sparse_vec);

    /// Return the norm
    double norm(double power = 2.0);

    /// Normalize
    void normalize();

    /// Dot product with another vector
    double dot(FCIVector& wfn);
    double dot(std::shared_ptr<FCIVector>& wfn);

    /// Return the alpha one-particle density matrix
    std::vector<double>& opdm_a() { return opdm_a_; }
    /// Return the beta one-particle density matrix
    std::vector<double>& opdm_b() { return opdm_b_; }
    /// Return the alpha-alpha two-particle density matrix
    std::vector<double>& tpdm_aa() { return tpdm_aa_; }
    /// Return the alpha-beta two-particle density matrix
    std::vector<double>& tpdm_ab() { return tpdm_ab_; }
    /// Return the beta-beta two-particle density matrix
    std::vector<double>& tpdm_bb() { return tpdm_bb_; }
    /// Return the alpha-alpha-alpha three-particle density matrix
    std::vector<double>& tpdm_aaa() { return tpdm_aaa_; }
    /// Return the alpha-alpha-beta three-particle density matrix
    std::vector<double>& tpdm_aab() { return tpdm_aab_; }
    /// Return the alpha-beta-beta three-particle density matrix
    std::vector<double>& tpdm_abb() { return tpdm_abb_; }
    /// Return the beta-beta-beta three-particle density matrix
    std::vector<double>& tpdm_bbb() { return tpdm_bbb_; }

    // Appy the Hamiltonian to this vector
    void Hamiltonian(FCIVector& result, std::shared_ptr<FCIIntegrals> fci_ints,
                     RequiredLists required_lists);

    /// Compute the energy from the RDMs
    double energy_from_rdms(std::shared_ptr<FCIIntegrals> fci_ints);

    /// Compute the RDMs
    void compute_rdms(int max_order = 2);

    /// Test the RDMSs
    void rdm_test();

    /// Print the natural_orbitals from FCIWFN
    /// Assume user specifed active space
    void print_natural_orbitals(std::shared_ptr<MOSpaceInfo>);

    /// Return the elements with the smallest value
    /// This function returns the tuple (C_I,irrep,Ia,Ib)
    std::vector<std::tuple<double, size_t, size_t, size_t>> min_elements(size_t num_dets);

    /// Return the elements with the largest absolute value
    /// This function returns the tuple (|C_I|,C_I,irrep,Ia,Ib)
    std::vector<std::tuple<double, double, size_t, size_t, size_t>>
    max_abs_elements(size_t num_dets);

    /// Allocate temporary memory
    static void allocate_temp_space(std::shared_ptr<StringLists> lists_, int print_);
    /// Release temporary memory
    static void release_temp_space();

    /// Set the print level
    void set_print(int print) { print_ = print; }

  private:
    // ==> Class Data <==

    /// The number of irreps
    int nirrep_;
    /// The symmetry of this vector
    size_t symmetry_;
    /// The total number of correlated molecular orbitals
    size_t ncmo_;
    /// The number of correlated molecular orbitals per irrep
    Dimension cmopi_;
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
    /// The alpha string graph
    GraphPtr alfa_graph_;
    /// The beta string graph
    GraphPtr beta_graph_;
    /// Coefficient matrix stored in block-matrix form
    /// Each element of this vector is block of a given irrep
    std::vector<SharedMatrix> C_;

    std::vector<double> opdm_a_;
    std::vector<double> opdm_b_;
    std::vector<double> tpdm_aa_;
    std::vector<double> tpdm_ab_;
    std::vector<double> tpdm_bb_;
    std::vector<double> tpdm_aaa_;
    std::vector<double> tpdm_aab_;
    std::vector<double> tpdm_abb_;
    std::vector<double> tpdm_bbb_;

    // ==> Class Static Data <==

    static SharedMatrix C1;
    static SharedMatrix Y1;
    static size_t sizeC1;

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

    /// Compute the energy of a determinant
    double determinant_energy(bool*& Ia, bool*& Ib, int n, std::shared_ptr<FCIIntegrals> fci_ints);

    // ==> Class Private Functions <==

    size_t oei_index(size_t p, size_t q) const { return ncmo_ * p + q; }
    size_t tei_index(size_t p, size_t q, size_t r, size_t s) const {
        return ncmo_ * ncmo_ * ncmo_ * p + ncmo_ * ncmo_ * q + ncmo_ * r + s;
    }
    size_t six_index(size_t p, size_t q, size_t r, size_t s, size_t t, size_t u) const {
        return (ncmo_ * ncmo_ * ncmo_ * ncmo_ * ncmo_ * p + ncmo_ * ncmo_ * ncmo_ * ncmo_ * q +
                ncmo_ * ncmo_ * ncmo_ * r + ncmo_ * ncmo_ * s + ncmo_ * t + u);
    }

    void H0(FCIVector& result, std::shared_ptr<FCIIntegrals> fci_ints);
    void H1(FCIVector& result, std::shared_ptr<FCIIntegrals> fci_ints, bool alfa);
    void H2_aabb(FCIVector& result, std::shared_ptr<FCIIntegrals> fci_ints);
    void H2_aaaa2(FCIVector& result, std::shared_ptr<FCIIntegrals> fci_ints, bool alfa);

    void compute_1rdm(std::vector<double>& rdm, bool alfa);
    void compute_2rdm_aa(std::vector<double>& rdm, bool alfa);
    void compute_2rdm_ab(std::vector<double>& rdm);
    void compute_3rdm_aaa(std::vector<double>& rdm, bool alfa);
    void compute_3rdm_aab(std::vector<double>& rdm);
    void compute_3rdm_abb(std::vector<double>& rdm);
};
}
}

#endif // _fci_vector_
