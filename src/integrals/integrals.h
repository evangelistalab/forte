/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2017 by its authors (see COPYING, COPYING.LESSER,
 * AUTHORS).
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

#ifndef _integrals_h_
#define _integrals_h_

#include <iostream>
#include <vector>

#include "psi4/libmints/typedefs.h"
#include "psi4/libmints/dimension.h"
#include "ambit/blocked_tensor.h"



class Tensor;
class Options;
class Matrix;
class Wavefunction;
class Dimension;

namespace forte {

class ForteOptions;
class MOSpaceInfo;

/**
 * @brief The IntegralSpinRestriction enum
 *
 * This is used to distinguish between restricted/unrestricted computations
 */
enum IntegralSpinRestriction { RestrictedMOs, UnrestrictedMOs };

/**
 * @brief The IntegralType enum
 *
 * This decides the type of integral used in a Forte computation
 */
enum IntegralType { Conventional, DF, Cholesky, DiskDF, DistDF, Own, Custom };

/**
 * @brief Add the options that control the integral class
 * @param foptions
 */
void set_INT_options(ForteOptions& foptions);

/**
 * @brief The ForteIntegrals class is a base class for transforming and storing MO integrals
 *
 * ForteIntegrals provides a common interface for reading one- and two-electron integrals
 * in the MO basis.
 * This class also takes care of removing frozen core and virtual orbitals (excluded from
 * any treatment of correlation energy) and forming the modified one-electron operator,
 * which includes contributions from doubly occupied frozen orbitals.
 *
 * One electron integrals include kinetic, nuclear potential, and frozen core potential and are
 * stored as
 *
 *     h_pq = <phi_p|h|phi_q>.
 *
 * Two electron integrals are returned as antisymmetrized integrals in physicist notation
 * (<pq||rs>), and are accessed as
 *
 *     aptei(p,q,r,s) = <pq||rs> = <pq|rs> - <pq|sr> = (pr|qs) - (ps|qr),
 *
 * where (pr|qs) is a two electron integral in chemist notation.
 */
class ForteIntegrals {
  public:
    // ==> Class Constructor and Destructor <==

    /**
     * @brief Class constructor
     * @param options The main options object
     * @param ref_wfn The reference wave function object
     * @param restricted Select a restricted or unrestricted transformation
     * @param mo_space_info The MOSpaceInfo object
     */
    ForteIntegrals(psi::Options& options, std::shared_ptr<Wavefunction> ref_wfn,
                   IntegralSpinRestriction restricted, std::shared_ptr<MOSpaceInfo> mo_space_info);

    /// Destructor
    virtual ~ForteIntegrals();

  public:
    // ==> Class Interface <==

    /// Return the total number of molecular orbitals (this number includes frozen MOs)
    size_t nmo() const { return nmo_; }

    /// Return the number of irreducible representations
    int nirrep() const { return nirrep_; }

    /// Return the number of frozen core orbitals per irrep
    psi::Dimension& frzcpi() { return frzcpi_; }
    /// Return the number of frozen virtual orbitals per irrep
    psi::Dimension& frzvpi() { return frzvpi_; }

    /// The number of correlated MOs per irrep (non frozen).  This is nmopi - nfzcpi - nfzvpi.
    psi::Dimension& ncmopi() { return ncmopi_; }

    /// Return the total number of correlated molecular orbitals (this number excludes frozen MOs)
    size_t ncmo() const { return ncmo_; }

    /// Set printing level
    void set_print(int print) { print_ = print; }

    /// Return the number of auxiliary functions
    virtual size_t nthree() const = 0;

    /// Return the frozen core energy
    double frozen_core_energy() { return frozen_core_energy_; }

    /// Scalar component of the Hamiltonian
    double scalar() const { return scalar_; }

    /// The alpha one-electron integrals
    double oei_a(size_t p, size_t q) { return one_electron_integrals_a_[p * aptei_idx_ + q]; }

    /// The beta one-electron integrals
    double oei_b(size_t p, size_t q) { return one_electron_integrals_b_[p * aptei_idx_ + q]; }

    /// Get the alpha fock matrix elements
    double get_fock_a(size_t p, size_t q) { return fock_matrix_a_[p * aptei_idx_ + q]; }

    /// Get the beta fock matrix elements
    double get_fock_b(size_t p, size_t q) { return fock_matrix_b_[p * aptei_idx_ + q]; }

    /// Get the alpha fock matrix in std::vector format
    std::vector<double> get_fock_a() const { return fock_matrix_a_; }

    /// Get the beta fock matrix in std::vector format
    std::vector<double> get_fock_b() const { return fock_matrix_b_; }

    /// Set the alpha fock matrix
    void set_fock_a(const std::vector<double>& fock_stl);

    /// Set the beta fock matrix
    void set_fock_b(const std::vector<double>& fock_stl);

    /// The antisymmetrixed alpha-alpha two-electron integrals in physicist
    /// notation <pq||rs>
    virtual double aptei_aa(size_t p, size_t q, size_t r, size_t s) = 0;

    /// The antisymmetrixed alpha-beta two-electron integrals in physicist
    /// notation <pq||rs>
    virtual double aptei_ab(size_t p, size_t q, size_t r, size_t s) = 0;
    /// The antisymmetrixed beta-beta two-electron integrals in physicist
    /// notation <pq||rs>
    virtual double aptei_bb(size_t p, size_t q, size_t r, size_t s) = 0;

    /// Grab a block of the integrals and return a tensor
    /// p, q, r, s correspond to the vector of indices you want for your tensor
    /// if p, q, r, s is equal to an array of all of the mos, then this will
    /// return a tensor of dimension nmo^4.
    virtual ambit::Tensor aptei_aa_block(const std::vector<size_t>& p, const std::vector<size_t>& q,
                                         const std::vector<size_t>& r,
                                         const std::vector<size_t>& s) = 0;
    /// Same as above but reads alpha-beta chunck
    virtual ambit::Tensor aptei_ab_block(const std::vector<size_t>& p, const std::vector<size_t>& q,
                                         const std::vector<size_t>& r,
                                         const std::vector<size_t>& s) = 0;
    /// The beta-beta integrals
    virtual ambit::Tensor aptei_bb_block(const std::vector<size_t>& p, const std::vector<size_t>& q,
                                         const std::vector<size_t>& r,
                                         const std::vector<size_t>& s) = 0;

    virtual ambit::Tensor three_integral_block(const std::vector<size_t>& A,
                                               const std::vector<size_t>& p,
                                               const std::vector<size_t>& q) = 0;
    /// This function is only used by DiskDF and it is used to go from a Apq->Aq
    /// tensor
    virtual ambit::Tensor three_integral_block_two_index(const std::vector<size_t>& A, size_t p,
                                                         const std::vector<size_t>& q) = 0;

    /// Make a Fock matrix computed with respect to a given determinant
    virtual void make_fock_matrix(std::shared_ptr<Matrix> gamma_a,
                                  std::shared_ptr<Matrix> gamma_b) = 0;

    /// Set the value of the scalar part of the Hamiltonian
    /// @param value the new value of the scalar part of the Hamiltonian
    void set_scalar(double value) { scalar_ = value; }

    /// Set the value of the one-electron integrals
    /// @param ints pointer to the integrals
    /// @param alpha the spin type of the integrals
    void set_oei(double** ints, bool alpha);

    /// Set the value of the one-electron integrals
    /// @param p the integral index
    /// @param q the integral index
    /// @param value the value of the integral
    /// @param alpha the spin type of the integrals
    void set_oei(size_t p, size_t q, double value, bool alpha);

    /// Set the value of the two-electron integrals
    virtual void set_tei(size_t p, size_t q, size_t r, size_t s, double value, bool alpha1,
                         bool alpha2) = 0;

    /// Update the integrals with a new set of MO coefficients
    virtual void retransform_integrals();
    /// Expert Option: just try and use three_integral
    virtual double** three_integral_pointer() = 0;

    /// Return the type of integral used
    IntegralType integral_type() { return integral_type_; }
    /// Return the one-body symmetry integrals
    std::shared_ptr<Matrix> OneBody_symm() { return OneBody_symm_; }
    /// Return the one-body AO integrals
    std::shared_ptr<Matrix> OneBodyAO() { return OneIntsAO_; }

    virtual int ga_handle() { return 0; }

    /// Print the details of the integral transformation
    void print_info();
    /// Print the one- and two-electron integrals to the output
    void print_ints();

    /// Obtain AO dipole integrals [X, Y, Z]
    /// Each direction is a psi::SharedMatrix of dimension nmo * nmo
    std::vector<std::shared_ptr<Matrix>> AOdipole_ints() { return AOdipole_ints_; }

    /**
     * Compute MO dipole integrals
     * @param alpha if true, compute MO dipole using Ca, else Cb
     * @param resort if true, MOdipole ints are sorted to Pitzer order, otherwise in C1 order
     * @return a vector of MOdipole ints in X, Y, Z order,
     *         each of which is a nmo by nmo psi::SharedMatrix
     */
    std::vector<std::shared_ptr<Matrix>> compute_MOdipole_ints(const bool& alpha = true,
                                                               const bool& resort = false);

  protected:
    // ==> Class data <==

    /// The options object
    psi::Options& options_;

    /// The Wavefunction object
    psi::SharedWavefunction wfn_;

    /// The integral_type
    IntegralType integral_type_;

    /// Are we doing a spin-restricted computation?
    IntegralSpinRestriction restricted_;

    /// Number of irreps
    int nirrep_;

    /// The number of MOs, including the ones that are frozen.
    size_t nmo_;

    /// The number of correlated MOs (excluding frozen).  This is nmo - nfzc - nfzv.
    size_t ncmo_;

    /// The mapping from correlated MO to full MO (frozen + correlated)
    std::vector<size_t> cmotomo_;

    /// The number of symmetrized AOs per irrep.
    psi::Dimension nsopi_;
    /// The number of MOs per irrep.
    psi::Dimension nmopi_;
    /// The number of frozen core MOs per irrep.
    psi::Dimension frzcpi_;
    /// The number of frozen unoccupied MOs per irrep.
    psi::Dimension frzvpi_;
    /// The number of correlated MOs per irrep (non frozen).  This is nmopi -
    /// nfzcpi - nfzvpi.
    psi::Dimension ncmopi_;

    /// The number of orbitals used in indexing routines (nmo or ncmo if core orbitals are frozen)
    /// The correct value is set by the integrals class
    size_t aptei_idx_;
    /// The number of symmetry orbitals
    size_t nso_;

    // OMP
    // Is OMP available?
#ifdef _OPENMP
    static const bool have_omp_ = true;
#else
    static const bool have_omp_ = false;
#endif
    /// The number of OMP threads
    int num_threads_;

    /// Number of two electron integrals in chemist notation (pq|rs)
    size_t num_tei_;

    /// The number of antisymmetrized two-electron integrals in physicist
    /// notation <pq||rs>
    size_t num_aptei_;

    /// Frozen-core energy
    double frozen_core_energy_;

    /// Scalar energy term
    double scalar_;

    /// The MOSpaceInfo object
    std::shared_ptr<MOSpaceInfo> mo_space_info_;

    /// Full one-electron integrals stored as a vector (includes frozen orbitals)
    std::vector<double> full_one_electron_integrals_a_;
    std::vector<double> full_one_electron_integrals_b_;

    /// One-electron integrals stored as a vector
    std::vector<double> one_electron_integrals_a_;
    std::vector<double> one_electron_integrals_b_;

    /// Fock matrix stored as a vector
    std::vector<double> fock_matrix_a_;
    std::vector<double> fock_matrix_b_;

    /// The type of tensor that ambit uses -> CoreTensor
    ambit::TensorType tensor_type_ = ambit::CoreTensor;
    /// How much memory each integral takes up
    double int_mem_;
    /// Control printing of timings
    int print_;
    /// The One Electron Integrals (T + V) in SO Basis
    std::shared_ptr<Matrix> OneBody_symm_;
    std::shared_ptr<Matrix> OneIntsAO_;

    /// AO dipole integrals
    std::vector<std::shared_ptr<Matrix>> AOdipole_ints_;
    /// Compute AO dipole integrals
    void build_AOdipole_ints();
    /// Compute MO dipole integrals
    std::vector<std::shared_ptr<psi::Matrix>>
    MOdipole_ints_helper(std::shared_ptr<psi::Matrix> Cao, psi::SharedVector epsilon, const bool& resort);

    // ==> Class private functions <==

    /// Class initializer
    void startup();

    /// Allocate the memory required to store the one-electron integrals and fock matrices
    void allocate();

    /// Transform the one-electron integrals
    void transform_one_electron_integrals();

    /// This function manages freezing core and virtual orbitals
    void freeze_core_orbitals();

    /// Compute the one-body operator modified by the frozen core orbitals
    void compute_frozen_one_body_operator();

    /// Function used to rotate MOs during contructor
    void rotate_mos();

    // ==> Class private virtual functions <==

    /// Computes/reads two-electron integrals (see CD/DF/Conventional classes for implementation)
    virtual void gather_integrals() = 0;

    /// Remove the doubly occupied and virtual orbitals and resort the rest so
    /// that we are left only with ncmo = nmo - nfzc - nfzv
    virtual void resort_integrals_after_freezing() = 0;
};

} // namespace forte


#endif // _integrals_h_
