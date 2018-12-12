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

#ifndef _sigma_vector_h_
#define _sigma_vector_h_

#include "helpers.h"
#include "helpers/timer.h"

#include "determinant_hashvector.h"
#include "fci/fci_integrals.h"
#include "operator.h"
#include "determinant.h"

#ifdef HAVE_MPI
#include <mpi.h>
#endif

namespace psi {
namespace forte {

/**
 * @brief The SigmaVector class
 * Base class for a sigma vector object.
 */
class SigmaVector {
  public:
    SigmaVector(size_t size) : size_(size){};

    size_t size() { return size_; }

    virtual void compute_sigma(SharedVector sigma, SharedVector b) = 0;
    //    virtual void compute_sigma(Matrix& sigma, Matrix& b, int nroot) = 0;
    virtual void get_diagonal(Vector& diag) = 0;
    virtual void add_bad_roots(std::vector<std::vector<std::pair<size_t, double>>>& bad_states) = 0;

  protected:
    /// The length of the C/sigma vector (number of determinants)
    size_t size_;
};

/**
 * @brief The SigmaVectorSparse class
 * Computes the sigma vector from a sparse Hamiltonian.
 */
class SigmaVectorSparse : public SigmaVector {
  public:
    SigmaVectorSparse(std::vector<std::pair<std::vector<size_t>, std::vector<double>>>& H,
                      std::shared_ptr<FCIIntegrals> fci_ints)
        : SigmaVector(H.size()), H_(H), fci_ints_(fci_ints){};

    void compute_sigma(SharedVector sigma, SharedVector b);
    //   void compute_sigma(Matrix& sigma, Matrix& b, int nroot) {}
    void get_diagonal(Vector& diag);
    void add_bad_roots(std::vector<std::vector<std::pair<size_t, double>>>& bad_states);

    std::vector<std::vector<std::pair<size_t, double>>> bad_states_;

  protected:
    std::vector<std::pair<std::vector<size_t>, std::vector<double>>>& H_;
    std::shared_ptr<FCIIntegrals> fci_ints_;
};

/**
 * @brief The SigmaVectorList class
 * Computes the sigma vector from a sparse Hamiltonian.
 */
class SigmaVectorList : public SigmaVector {
  public:
    SigmaVectorList(const std::vector<Determinant>& space, bool print_detail,
                    std::shared_ptr<FCIIntegrals> fci_ints);

    void compute_sigma(SharedVector sigma, SharedVector b);
    //  void compute_sigma(Matrix& sigma, Matrix& b, int nroot);
    void get_diagonal(Vector& diag);
    void get_hamiltonian(Matrix& H);
    std::vector<std::pair<std::vector<int>, std::vector<double>>> get_sparse_hamiltonian();
    void add_bad_roots(std::vector<std::vector<std::pair<size_t, double>>>& bad_states);

    std::vector<std::vector<std::pair<size_t, double>>> bad_states_;

  protected:
    const std::vector<Determinant>& space_;
    std::shared_ptr<FCIIntegrals> fci_ints_;

    // Create the list of a_p|N>
    std::vector<std::vector<std::pair<size_t, short>>> a_ann_list;
    std::vector<std::vector<std::pair<size_t, short>>> b_ann_list;
    // Create the list of a+_q |N-1>
    std::vector<std::vector<std::pair<size_t, short>>> a_cre_list;
    std::vector<std::vector<std::pair<size_t, short>>> b_cre_list;

    // Create the list of a_q a_p|N>
    std::vector<std::vector<std::tuple<size_t, short, short>>> aa_ann_list;
    std::vector<std::vector<std::tuple<size_t, short, short>>> ab_ann_list;
    std::vector<std::vector<std::tuple<size_t, short, short>>> bb_ann_list;
    // Create the list of a+_s a+_r |N-2>
    std::vector<std::vector<std::tuple<size_t, short, short>>> aa_cre_list;
    std::vector<std::vector<std::tuple<size_t, short, short>>> ab_cre_list;
    std::vector<std::vector<std::tuple<size_t, short, short>>> bb_cre_list;
    std::vector<double> diag_;

    bool print_details_ = true;
};

/* Uses ann/cre lists in sigma builds (Harrison and Zarrabian method) */
class SigmaVectorWfn1 : public SigmaVector {
  public:
    SigmaVectorWfn1(const DeterminantHashVec& space, WFNOperator& op,
                    std::shared_ptr<FCIIntegrals> fci_ints);

    void compute_sigma(SharedVector sigma, SharedVector b);
    //   void compute_sigma(Matrix& sigma, Matrix& b, int nroot) {}
    void get_diagonal(Vector& diag);
    void add_bad_roots(std::vector<std::vector<std::pair<size_t, double>>>& bad_states);
    std::vector<std::vector<std::pair<size_t, double>>> bad_states_;

  protected:
    // Create the list of a_p|N>
    std::vector<std::vector<std::pair<size_t, short>>>& a_ann_list_;
    std::vector<std::vector<std::pair<size_t, short>>>& b_ann_list_;
    // Create the list of a+_q |N-1>
    std::vector<std::vector<std::pair<size_t, short>>>& a_cre_list_;
    std::vector<std::vector<std::pair<size_t, short>>>& b_cre_list_;

    // Create the list of a_q a_p|N>
    std::vector<std::vector<std::tuple<size_t, short, short>>>& aa_ann_list_;
    std::vector<std::vector<std::tuple<size_t, short, short>>>& ab_ann_list_;
    std::vector<std::vector<std::tuple<size_t, short, short>>>& bb_ann_list_;
    // Create the list of a+_s a+_r |N-2>
    std::vector<std::vector<std::tuple<size_t, short, short>>>& aa_cre_list_;
    std::vector<std::vector<std::tuple<size_t, short, short>>>& ab_cre_list_;
    std::vector<std::vector<std::tuple<size_t, short, short>>>& bb_cre_list_;
    std::vector<double> diag_;
    const DeterminantHashVec& space_;
    std::shared_ptr<FCIIntegrals> fci_ints_;
};

/* Uses only cre lists, sparse sigma build */
class SigmaVectorWfn2 : public SigmaVector {
  public:
    SigmaVectorWfn2(const DeterminantHashVec& space, WFNOperator& op,
                    std::shared_ptr<FCIIntegrals> fci_ints);
    std::vector<std::vector<std::pair<size_t, short>>>& a_list_;
    std::vector<std::vector<std::pair<size_t, short>>>& b_list_;
    std::vector<std::vector<std::tuple<size_t, short, short>>>& aa_list_;
    std::vector<std::vector<std::tuple<size_t, short, short>>>& ab_list_;
    std::vector<std::vector<std::tuple<size_t, short, short>>>& bb_list_;

    void compute_sigma(SharedVector sigma, SharedVector b);
    // void compute_sigma(Matrix& sigma, Matrix& b, int nroot);
    void get_diagonal(Vector& diag);
    void add_bad_roots(std::vector<std::vector<std::pair<size_t, double>>>& bad_states_);

    std::vector<std::vector<std::pair<size_t, double>>> bad_states_;

  protected:
    bool print_;
    bool use_disk_ = false;

    const DeterminantHashVec& space_;
    // size_t noalfa_;
    // size_t nobeta_;

    std::vector<double> diag_;
    std::shared_ptr<FCIIntegrals> fci_ints_;
};
/* Uses only cre lists, DGEMM sigma build */
class SigmaVectorWfn3 : public SigmaVector {
  public:
    SigmaVectorWfn3(const DeterminantHashVec& space, WFNOperator& op,
                    std::shared_ptr<FCIIntegrals> fci_ints);
    std::vector<std::vector<std::pair<size_t, short>>>& a_list_;
    std::vector<std::vector<std::pair<size_t, short>>>& b_list_;
    std::vector<std::vector<std::tuple<size_t, short, short>>>& aa_list_;
    std::vector<std::vector<std::tuple<size_t, short, short>>>& ab_list_;
    std::vector<std::vector<std::tuple<size_t, short, short>>>& bb_list_;

    void compute_sigma(SharedVector sigma, SharedVector b);
    // void compute_sigma(Matrix& sigma, Matrix& b, int nroot);
    void get_diagonal(Vector& diag);
    void add_bad_roots(std::vector<std::vector<std::pair<size_t, double>>>& bad_states_);

    std::vector<std::vector<std::pair<size_t, double>>> bad_states_;

  protected:
    bool print_;
    bool use_disk_ = false;

    std::shared_ptr<FCIIntegrals> fci_ints_;
    const DeterminantHashVec& space_;
    // size_t noalfa_;
    // size_t nobeta_;

    std::vector<double> diag_;

    SharedMatrix aa_tei_;
    SharedMatrix ab_tei_;
    SharedMatrix bb_tei_;
};

#ifdef HAVE_MPI
class SigmaVectorMPI : public SigmaVector {
  public:
    SigmaVectorMPI(const DeterminantHashVec& space, WFNOperator& op,
                   std::shared_ptr<FCIIntegrals> fci_ints);

    void compute_sigma(SharedVector sigma, SharedVector b);
    void compute_sigma(Matrix& sigma, Matrix& b, int nroot);
    void get_diagonal(Vector& diag);
    void add_bad_roots(std::vector<std::vector<std::pair<size_t, double>>>& bad_states_);

    std::vector<std::vector<std::pair<size_t, double>>> bad_states_;

  protected:
    std::shared_ptr<FCIIntegrals> fci_ints_;
};
#endif
}
}

#endif // _sigma_vector_h_
