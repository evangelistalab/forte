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

#define PAIRINDEX(i, j) ((i > j) ? (ioff[(i)] + (j)) : (ioff[(j)] + (i)))
#define four(i, j, k, l) PAIRINDEX(PAIRINDEX(i, j), PAIRINDEX(k, l))

#include <iostream>
#include <vector>

#include "ambit/blocked_tensor.h"
#include "psi4/libmints/matrix.h"
#include "psi4/liboptions/liboptions.h"
#include "psi4/libtrans/integraltransform.h"
#include "psi4/libpsi4util/PsiOutStream.h"
#include "psi4/lib3index/dfhelper.h"

namespace psi {

class Tensor;

namespace forte {

class ForteOptions;
class MOSpaceInfo;

/// This decides the type of transformation: resticted vs. unrestricted
enum IntegralSpinRestriction { RestrictedMOs, UnrestrictedMOs };

/// This decides the type of integral
/// The integrals implementation is in a cc file for each class.
/// DFIntegrals->df_integrals.cc
enum IntegralType { Conventional, DF, Cholesky, DiskDF, DistDF, Own, Custom };

/// Set integrals options
void set_INT_options(ForteOptions& foptions);

/**
 * @brief The ForteIntegrals class is a base class for transforming and storing MO integrals
 *
 * ForteIntegrals provides a common interface for reading one- and two-electron integrals
 * in the MO basis.
 * This class also takes care of removing frozen core and virtual orbitals (excluded from
 * any treatment of correlation energy) and forming the modified one-electron operator,
 * which includes contributions from doubly occupied frozen orbitals.
 */
class ForteIntegrals {
  public:
    // ==> Class Constructor and Destructor <==

    /**
     * Constructor
     * @param options The main options object
     * @param ref_wfn The reference wave function object
     * @param restricted Select a restricted or unrestricted transformation
     * @param mo_space_info The MOSpaceInfo object
     */
    ForteIntegrals(psi::Options& options, SharedWavefunction ref_wfn,
                   IntegralSpinRestriction restricted, std::shared_ptr<MOSpaceInfo> mo_space_info);

    /// Destructor
    virtual ~ForteIntegrals();

  public:
    // ==> Class Interface <==

    /// Return the total number of correlated molecular orbitals (this number excludes frozen MOs)
    size_t ncmo() const { return ncmo_; }

    /// Set printing level
    void set_print(int print) { print_ = print; }

    /// Return the number of auxiliary functions
    virtual size_t nthree() const {
        throw PSIEXCEPTION("WRONG INT_TYPE");
        return 1;
    }

    /// Return the frozen core energy
    double frozen_core_energy() { return frozen_core_energy_; }

    /// Scalar component of the Hamiltonian
    double scalar() const { return scalar_; }

    /// The alpha one-electron integrals
    double oei_a(size_t p, size_t q) { return one_electron_integrals_a[p * aptei_idx_ + q]; }

    /// The beta one-electron integrals
    double oei_b(size_t p, size_t q) { return one_electron_integrals_b[p * aptei_idx_ + q]; }

    /// Get the alpha fock matrix elements
    double get_fock_a(size_t p, size_t q) { return fock_matrix_a[p * aptei_idx_ + q]; }

    /// Get the beta fock matrix elements
    double get_fock_b(size_t p, size_t q) { return fock_matrix_b[p * aptei_idx_ + q]; }

    /// Get the alpha fock matrix in std::vector format
    std::vector<double> get_fock_a() const { return fock_matrix_a; }

    /// Get the beta fock matrix in std::vector format
    std::vector<double> get_fock_b() const { return fock_matrix_b; }

    /// Set the alpha fock matrix
    void set_fock_a(const std::vector<double>& fock_stl) {
        size_t fock_size = fock_stl.size();
        if (fock_size != ncmo_ * ncmo_) {
            throw PSIEXCEPTION("Cannot fill in fock_matrix_a because the vector is out-of-range.");
        } else {
            fock_matrix_a = fock_stl;
        }
    }

    /// Set the beta fock matrix
    void set_fock_b(const std::vector<double>& fock_stl) {
        size_t fock_size = fock_stl.size();
        if (fock_size != ncmo_ * ncmo_) {
            throw PSIEXCEPTION("Cannot fill in fock_matrix_b because the vector is out-of-range.");
        } else {
            fock_matrix_b = fock_stl;
        }
    }

    /// The alpha diagonal fock matrix integrals
    double diag_fock_a(size_t p) { return fock_matrix_a[p * aptei_idx_ + p]; }

    /// The beta diagonal fock matrix integrals
    double diag_fock_b(size_t p) { return fock_matrix_b[p * aptei_idx_ + p]; }

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
    /// return
    /// a tensor of nmo^4.
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

    virtual double three_integral(size_t A, size_t p, size_t q) = 0;

    virtual ambit::Tensor three_integral_block(const std::vector<size_t>& A,
                                               const std::vector<size_t>& p,
                                               const std::vector<size_t>& q) = 0;
    /// This function is only used by DiskDF and it is used to go from a Apq->Aq
    /// tensor
    virtual ambit::Tensor three_integral_block_two_index(const std::vector<size_t>& A, size_t p,
                                                         const std::vector<size_t>& q) = 0;

    /// Make a Fock matrix computed with respect to a given determinant
    virtual void make_fock_matrix(SharedMatrix gamma_a, SharedMatrix gamma_b) = 0;

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
    SharedMatrix OneBody_symm() { return OneBody_symm_; }
    /// Return the one-body AO integrals
    SharedMatrix OneBodyAO() { return OneIntsAO_; }

    virtual int ga_handle() { return 0; }

    /// Print the one- and two-electron integrals to the output
    void print_ints();

    /// Obtain AO dipole integrals [X, Y, Z]
    /// Each direction is a SharedMatrix of dimension nmo * nmo
    std::vector<SharedMatrix> AOdipole_ints() { return AOdipole_ints_; }

    /**
     * Compute MO dipole integrals
     * @param alpha if true, compute MO dipole using Ca, else Cb
     * @param resort if true, MOdipole ints are sorted to Pitzer order, otherwise in C1 order
     * @return a vector of MOdipole ints in X, Y, Z order,
     *         each of which is a nmo by nmo SharedMatrix
     */
    std::vector<SharedMatrix> compute_MOdipole_ints(const bool& alpha = true,
                                                    const bool& resort = false);

  protected:
    // ==> Class data <==

    /// The options object
    psi::Options& options_;

    /// The Wavefunction object
    SharedWavefunction wfn_;

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
    Dimension nsopi_;
    /// The number of MOs per irrep.
    Dimension nmopi_;
    /// The number of frozen core MOs per irrep.
    Dimension frzcpi_;
    /// The number of frozen unoccupied MOs per irrep.
    Dimension frzvpi_;
    /// The number of correlated MOs per irrep (non frozen).  This is nmopi -
    /// nfzcpi - nfzvpi.
    Dimension ncmopi_;

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

    /// Number of one electron integrals
    size_t num_oei;

    /// Number of two electron integrals in chemist notation (pq|rs)
    size_t num_tei;

    /// The number of antisymmetrized two-electron integrals in physicist
    /// notation <pq||rs>
    size_t num_aptei;

    /// Frozen-core energy
    double frozen_core_energy_;

    /// Scalar energy term
    double scalar_;

    /// The MOSpaceInfo object
    std::shared_ptr<MOSpaceInfo> mo_space_info_;

    /// Full one-electron integrals stored as a vector (includes frozen orbitals)
    std::vector<double> full_one_electron_integrals_a;
    std::vector<double> full_one_electron_integrals_b;

    /// One-electron integrals stored as a vector
    std::vector<double> one_electron_integrals_a;
    std::vector<double> one_electron_integrals_b;

    /// Fock matrix stored as a vector
    std::vector<double> fock_matrix_a;
    std::vector<double> fock_matrix_b;

    /// The B tensor
    // std::shared_ptr<psi::Tensor> B_;
    std::shared_ptr<DFHelper> df_;

    /// The type of tensor that ambit uses -> CoreTensor
    ambit::TensorType tensor_type_ = ambit::CoreTensor;
    /// How much memory each integral takes up
    double int_mem_;
    /// Control printing of timings
    int print_;
    /// The One Electron Integrals (T + V) in SO Basis
    SharedMatrix OneBody_symm_;
    SharedMatrix OneIntsAO_;

    /// AO dipole integrals
    std::vector<SharedMatrix> AOdipole_ints_;
    /// Compute AO dipole integrals
    void build_AOdipole_ints();
    /// Compute MO dipole integrals
    std::vector<SharedMatrix> MOdipole_ints_helper(SharedMatrix Cao, SharedVector epsilon,
                                                   const bool& resort);

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
    virtual void resort_three(std::shared_ptr<Matrix>&, std::vector<size_t>& map) = 0;
};

/**
 * @brief The ConventionalIntegrals class is an interface to calculate the
 * conventional integrals
 * Assumes storage of all tei and stores in core.
 */
class ConventionalIntegrals : public ForteIntegrals {
  public:
    /// Contructor of the class.  Calls std::shared_ptr<ForteIntegrals> ints
    /// constructor
    ConventionalIntegrals(psi::Options& options, SharedWavefunction ref_wfn,
                          IntegralSpinRestriction restricted,
                          std::shared_ptr<MOSpaceInfo> mo_space_info);
    virtual ~ConventionalIntegrals();

    /// Grabs the antisymmetriced TEI - assumes storage in aphy_tei_*
    virtual double aptei_aa(size_t p, size_t q, size_t r, size_t s);
    virtual double aptei_ab(size_t p, size_t q, size_t r, size_t s);
    virtual double aptei_bb(size_t p, size_t q, size_t r, size_t s);

    /// Grabs the antisymmetrized TEI - assumes storage of ambit tensor
    virtual ambit::Tensor aptei_aa_block(const std::vector<size_t>& p, const std::vector<size_t>& q,
                                         const std::vector<size_t>& r,
                                         const std::vector<size_t>& s);
    virtual ambit::Tensor aptei_ab_block(const std::vector<size_t>& p, const std::vector<size_t>& q,
                                         const std::vector<size_t>& r,
                                         const std::vector<size_t>& s);
    virtual ambit::Tensor aptei_bb_block(const std::vector<size_t>& p, const std::vector<size_t>& q,
                                         const std::vector<size_t>& r,
                                         const std::vector<size_t>& s);

    virtual double three_integral(size_t, size_t, size_t) {
        outfile->Printf("\n Oh no!, you tried to grab a ThreeIntegral but this "
                        "is not there!!");
        throw PSIEXCEPTION("INT_TYPE=DF/CHOLESKY to use ThreeIntegral");
    }
    virtual ambit::Tensor three_integral_block(const std::vector<size_t>&,
                                               const std::vector<size_t>&,
                                               const std::vector<size_t>&) {
        outfile->Printf("\n Oh no!, you tried to grab a ThreeIntegral but this "
                        "is not there!!");
        throw PSIEXCEPTION("INT_TYPE=DF/CHOLESKY to use ThreeIntegral");
    }
    virtual ambit::Tensor three_integral_block_two_index(const std::vector<size_t>&, size_t,
                                                         const std::vector<size_t>&) {
        outfile->Printf("\n Oh no! this isn't here");
        throw PSIEXCEPTION("INT_TYPE=DISKDF");
    }

    virtual double** three_integral_pointer() {
        outfile->Printf("\n Doh! There is no Three_integral here.  Use DF/CD");
        throw PSIEXCEPTION("INT_TYPE=DF/CHOLESKY to use ThreeIntegral!");
    }

    virtual void make_fock_matrix(SharedMatrix gamma_a, SharedMatrix gamma_b);

    virtual size_t nthree() const { throw PSIEXCEPTION("Wrong Int_Type"); }

  private:
    /// Transform the integrals
    void transform_integrals();

    void resort_four(std::vector<double>& tei, std::vector<size_t>& map);

    virtual void gather_integrals();
    virtual void resort_integrals_after_freezing();
    virtual void resort_three(std::shared_ptr<Matrix>&, std::vector<size_t>&) {}
    virtual void set_tei(size_t p, size_t q, size_t r, size_t s, double value, bool alpha1,
                         bool alpha2);

    /// An addressing function to retrieve the two-electron integrals
    size_t aptei_index(size_t p, size_t q, size_t r, size_t s) {
        return aptei_idx_ * aptei_idx_ * aptei_idx_ * p + aptei_idx_ * aptei_idx_ * q +
               aptei_idx_ * r + s;
    }

    /// The IntegralTransform object used by this class
    std::shared_ptr<IntegralTransform> integral_transform_;

    /// Two-electron integrals stored as a vector
    std::vector<double> aphys_tei_aa;
    std::vector<double> aphys_tei_ab;
    std::vector<double> aphys_tei_bb;
};

/// Classes written by Kevin Hannon
///
/**
 * @brief The CholeskyIntegrals:  An interface that computes the cholesky
 * integrals,
 * freezes the core, and creates fock matrices from determinant classes
 */
class CholeskyIntegrals : public ForteIntegrals {
  public:
    CholeskyIntegrals(psi::Options& options, SharedWavefunction ref_wfn,
                      IntegralSpinRestriction restricted,
                      std::shared_ptr<MOSpaceInfo> mo_space_info);
    virtual ~CholeskyIntegrals();
    /// aptei_x will grab antisymmetriced integrals and creates DF/CD integrals
    /// on the fly
    virtual double aptei_aa(size_t p, size_t q, size_t r, size_t s);
    virtual double aptei_ab(size_t p, size_t q, size_t r, size_t s);
    virtual double aptei_bb(size_t p, size_t q, size_t r, size_t s);

    /// Grabs the antisymmetrized TEI - assumes storage of ambit tensor
    virtual ambit::Tensor aptei_aa_block(const std::vector<size_t>& p, const std::vector<size_t>& q,
                                         const std::vector<size_t>& r,
                                         const std::vector<size_t>& s);
    virtual ambit::Tensor aptei_ab_block(const std::vector<size_t>& p, const std::vector<size_t>& q,
                                         const std::vector<size_t>& r,
                                         const std::vector<size_t>& s);
    virtual ambit::Tensor aptei_bb_block(const std::vector<size_t>& p, const std::vector<size_t>& q,
                                         const std::vector<size_t>& r,
                                         const std::vector<size_t>& s);

    virtual double three_integral(size_t A, size_t p, size_t q) {
        return ThreeIntegral_->get(p * aptei_idx_ + q, A);
    }
    virtual double** three_integral_pointer() { return ThreeIntegral_->pointer(); }
    virtual ambit::Tensor three_integral_block(const std::vector<size_t>& A,
                                               const std::vector<size_t>& p,
                                               const std::vector<size_t>& q);
    virtual ambit::Tensor three_integral_block_two_index(const std::vector<size_t>&, size_t,
                                                         const std::vector<size_t>&) {
        outfile->Printf("\n Oh no! this isn't here");
        throw PSIEXCEPTION("INT_TYPE=DISKDF");
    }
    /// Do not use this if you are using CD/DF integrals
    virtual void set_tei(size_t p, size_t q, size_t r, size_t s, double value, bool alpha1,
                         bool alpha2);

    virtual void make_fock_matrix(SharedMatrix gamma_a, SharedMatrix gamma_b);

    virtual size_t nthree() const { return nthree_; }
    SharedMatrix L_ao_;

  private:
    /// Computes Cholesky integrals
    virtual void gather_integrals();
    /// Allocates diagonal integrals
    virtual void resort_three(std::shared_ptr<Matrix>& threeint, std::vector<size_t>& map);
    virtual void resort_integrals_after_freezing();
    void transform_integrals();
    std::shared_ptr<Matrix> ThreeIntegral_;
    size_t nthree_ = 0;
};

/**
 * @brief The DFIntegrals class - interface to get DF integrals, freeze core and
 * resort,
 * make fock matrices, and grab information about the space
 */
class DFIntegrals : public ForteIntegrals {
  public:
    DFIntegrals(psi::Options& options, SharedWavefunction ref_wfn,
                IntegralSpinRestriction restricted, std::shared_ptr<MOSpaceInfo> mo_space_info);
    virtual double aptei_aa(size_t p, size_t q, size_t r, size_t s);
    virtual double aptei_ab(size_t p, size_t q, size_t r, size_t s);
    virtual double aptei_bb(size_t p, size_t q, size_t r, size_t s);

    /// Reads the antisymmetrized alpha-alpha chunck and returns an
    /// ambit::Tensor
    /// Grabs the antisymmetrized TEI - assumes storage of ambit tensor
    virtual ambit::Tensor aptei_aa_block(const std::vector<size_t>& p, const std::vector<size_t>& q,
                                         const std::vector<size_t>& r,
                                         const std::vector<size_t>& s);
    virtual ambit::Tensor aptei_ab_block(const std::vector<size_t>& p, const std::vector<size_t>& q,
                                         const std::vector<size_t>& r,
                                         const std::vector<size_t>& s);
    virtual ambit::Tensor aptei_bb_block(const std::vector<size_t>& p, const std::vector<size_t>& q,
                                         const std::vector<size_t>& r,
                                         const std::vector<size_t>& s);

    virtual double three_integral(size_t A, size_t p, size_t q) {
        return ThreeIntegral_->get(p * aptei_idx_ + q, A);
    }
    virtual ambit::Tensor three_integral_block(const std::vector<size_t>& A,
                                               const std::vector<size_t>& p,
                                               const std::vector<size_t>& q);
    virtual ambit::Tensor three_integral_block_two_index(const std::vector<size_t>&, size_t,
                                                         const std::vector<size_t>&) {
        outfile->Printf("\n Oh no! this isn't here");
        throw PSIEXCEPTION("INT_TYPE=DISKDF");
    }
    virtual double** three_integral_pointer() { return ThreeIntegral_->pointer(); }
    virtual void set_tei(size_t p, size_t q, size_t r, size_t s, double value, bool alpha1,
                         bool alpha2);
    virtual ~DFIntegrals();

    virtual void make_fock_matrix(SharedMatrix gamma_a, SharedMatrix gamma_b);

    virtual size_t nthree() const { return nthree_; }

  private:
    virtual void gather_integrals();
    virtual void resort_three(std::shared_ptr<Matrix>& threeint, std::vector<size_t>& map);
    virtual void resort_integrals_after_freezing();
    virtual void resort_four(double*&, std::vector<size_t>&) {}

    std::shared_ptr<Matrix> ThreeIntegral_;
    size_t nthree_ = 0;
};

/// A DiskDFIntegrals class for avoiding the storage of the ThreeIntegral tensor
/// Assumes that the DFIntegrals are stored in a binary file generated by
/// DF_Helper
/// Aptei_xy are extremely slow -> Try to use three_electron_block.  Much faster
/// Reading individual elements is slow
class DISKDFIntegrals : public ForteIntegrals {
  public:
    DISKDFIntegrals(psi::Options& options, SharedWavefunction ref_wfn,
                    IntegralSpinRestriction restricted, std::shared_ptr<MOSpaceInfo> mo_space_info);

    /// aptei_xy functions are slow.  try to use three_integral_block

    virtual double aptei_aa(size_t p, size_t q, size_t r, size_t s);
    virtual double aptei_ab(size_t p, size_t q, size_t r, size_t s);
    virtual double aptei_bb(size_t p, size_t q, size_t r, size_t s);

    /// Reads the antisymmetrized alpha-alpha chunck and returns an
    /// ambit::Tensor
    virtual ambit::Tensor aptei_aa_block(const std::vector<size_t>& p, const std::vector<size_t>& q,
                                         const std::vector<size_t>& r,
                                         const std::vector<size_t>& s);
    virtual ambit::Tensor aptei_ab_block(const std::vector<size_t>& p, const std::vector<size_t>& q,
                                         const std::vector<size_t>& r,
                                         const std::vector<size_t>& s);
    virtual ambit::Tensor aptei_bb_block(const std::vector<size_t>& p, const std::vector<size_t>& q,
                                         const std::vector<size_t>& r,
                                         const std::vector<size_t>& s);

    virtual double diag_aptei_aa(size_t p, size_t q);
    virtual double diag_aptei_ab(size_t p, size_t q);
    virtual double diag_aptei_bb(size_t p, size_t q);
    virtual double three_integral(size_t A, size_t p, size_t q);
    virtual double** three_integral_pointer() { return (ThreeIntegral_->pointer()); }
    /// Read a block of the DFIntegrals and return an Ambit tensor of size A by
    /// p by q
    virtual ambit::Tensor three_integral_block(const std::vector<size_t>& A,
                                               const std::vector<size_t>& p,
                                               const std::vector<size_t>& q);
    /// return ambit tensor of size A by q
    virtual ambit::Tensor three_integral_block_two_index(const std::vector<size_t>& A, size_t p,
                                                         const std::vector<size_t>& q);

    virtual void set_tei(size_t p, size_t q, size_t r, size_t s, double value, bool alpha1,
                         bool alpha2);
    virtual ~DISKDFIntegrals();

    virtual void make_fock_matrix(SharedMatrix gamma_a, SharedMatrix gamma_b);

    /// Make a Fock matrix computed with respect to a given determinant
    virtual size_t nthree() const { return nthree_; }

  private:
    virtual void gather_integrals();
    virtual void resort_three(std::shared_ptr<Matrix>& threeint, std::vector<size_t>& map);
    virtual void resort_integrals_after_freezing();

    std::shared_ptr<Matrix> ThreeIntegral_;
    size_t nthree_ = 0;
};

#ifdef HAVE_GA
class DistDFIntegrals : public ForteIntegrals {
  public:
    DistDFIntegrals(psi::Options& options, SharedWavefunction ref_wfn,
                    IntegralSpinRestriction restricted, std::shared_ptr<MOSpaceInfo> mo_space_info);

    virtual void retransform_integrals();
    /// aptei_xy functions are slow.  try to use three_integral_block

    virtual double aptei_aa(size_t /*p*/, size_t /*q*/, size_t /*r*/, size_t /*s*/) {}
    virtual double aptei_ab(size_t /*p*/, size_t /*q*/, size_t /*r*/, size_t /*s*/) {}
    virtual double aptei_bb(size_t /*p*/, size_t /*q*/, size_t /*r*/, size_t /*s*/) {}

    /// Reads the antisymmetrized alpha-alpha chunck and returns an
    /// ambit::Tensor
    virtual ambit::Tensor aptei_aa_block(const std::vector<size_t>& p, const std::vector<size_t>& q,
                                         const std::vector<size_t>& r,
                                         const std::vector<size_t>& s) {}
    virtual ambit::Tensor aptei_ab_block(const std::vector<size_t>& p, const std::vector<size_t>& q,
                                         const std::vector<size_t>& r,
                                         const std::vector<size_t>& s) {}
    virtual ambit::Tensor aptei_bb_block(const std::vector<size_t>& p, const std::vector<size_t>& q,
                                         const std::vector<size_t>& r,
                                         const std::vector<size_t>& s) {}

    virtual double diag_aptei_aa(size_t, size_t) {}
    virtual double diag_aptei_ab(size_t, size_t) {}
    virtual double diag_aptei_bb(size_t, size_t) {}
    virtual double three_integral(size_t, size_t, size_t) {}
    virtual double** three_integral_pointer() {
        throw PSIEXCEPTION("Integrals are distributed.  Pointer does not exist");
    }
    /// Read a block of the DFIntegrals and return an Ambit tensor of size A by
    /// p by q
    virtual ambit::Tensor three_integral_block(const std::vector<size_t>& /*A*/,
                                               const std::vector<size_t>& /*p*/,
                                               const std::vector<size_t>& /*q*/);
    /// return ambit tensor of size A by q
    virtual ambit::Tensor three_integral_block_two_index(const std::vector<size_t>& /*A*/,
                                                         size_t /*p*/,
                                                         const std::vector<size_t>& /*q*/) {}

    virtual void set_tei(size_t, size_t, size_t, size_t, double, bool, bool) {
        outfile->Printf("DistributedDF will not work with set_tei");
        throw PSIEXCEPTION("DistDF can not use set_tei");
    }
    virtual ~DistDFIntegrals();

    virtual void make_fock_matrix(SharedMatrix /*gamma_a*/, SharedMatrix /*gamma_b*/) {}

    /// Make a Fock matrix computed with respect to a given determinant
    virtual size_t nthree() const { return nthree_; }
    virtual int ga_handle() { return DistDF_ga_; }

  private:
    virtual void gather_integrals();
    virtual void resort_three(std::shared_ptr<Matrix>& /*threeint*/, std::vector<size_t>& /*map*/) {
    }
    virtual void resort_integrals_after_freezing() {}

    /// This is the handle for GA
    int DistDF_ga_;

    std::shared_ptr<Matrix> ThreeIntegral_;
    size_t nthree_ = 0;
    /// Assuming integrals are stored on disk
    /// Reads the block of integrals present for each process
    ambit::Tensor read_integral_chunk(std::shared_ptr<Tensor>& B, std::vector<int>& lo,
                                      std::vector<int>& hi);
    /// Distributes tensor according to naux dimension
    void create_dist_df();
    void test_distributed_integrals();
};
#endif

/// This class is used if the user wants to generate their own integrals for
/// their method.
/// This would be very useful for CI based methods (the integrals class is
/// wasteful and dumb for this area)
/// Also, I am putting this here if I(Kevin) ever get around to implementing
/// AO-DSRG-MRPT2
class OwnIntegrals : public ForteIntegrals {
  public:
    OwnIntegrals(psi::Options& options, SharedWavefunction ref_wfn,
                 IntegralSpinRestriction restricted, std::shared_ptr<MOSpaceInfo> mo_space_info);

    virtual void retransform_integrals() {}
    /// aptei_xy functions are slow.  try to use three_integral_block

    virtual double aptei_aa(size_t /*p*/, size_t /*q*/, size_t /*r*/, size_t /*s*/) { return 0.0; }
    virtual double aptei_ab(size_t /*p*/, size_t /*q*/, size_t /*r*/, size_t /*s*/) { return 0.0; }
    virtual double aptei_bb(size_t /*p*/, size_t /*q*/, size_t /*r*/, size_t /*s*/) { return 0.0; }

    /// Reads the antisymmetrized alpha-alpha chunck and returns an
    /// ambit::Tensor
    virtual ambit::Tensor aptei_aa_block(const std::vector<size_t>& /*p*/,
                                         const std::vector<size_t>& /*q*/,
                                         const std::vector<size_t>& /*r*/,
                                         const std::vector<size_t>& /*s*/) {
        return blank_tensor_;
    }
    virtual ambit::Tensor aptei_ab_block(const std::vector<size_t>& /*p*/,
                                         const std::vector<size_t>& /*q*/,
                                         const std::vector<size_t>& /*r*/,
                                         const std::vector<size_t>& /*s*/) {
        return blank_tensor_;
    }
    virtual ambit::Tensor aptei_bb_block(const std::vector<size_t>& /*p*/,
                                         const std::vector<size_t>& /*q*/,
                                         const std::vector<size_t>& /*r*/,
                                         const std::vector<size_t>& /*s*/) {
        return blank_tensor_;
    }

    virtual double diag_aptei_aa(size_t, size_t) { return 0.0; }
    virtual double diag_aptei_ab(size_t, size_t) { return 0.0; }
    virtual double diag_aptei_bb(size_t, size_t) { return 0.0; }
    virtual double three_integral(size_t, size_t, size_t) { return 0.0; }
    virtual double** three_integral_pointer() {
        throw PSIEXCEPTION("Integrals are distributed.  Pointer does not exist");
    }
    /// Read a block of the DFIntegrals and return an Ambit tensor of size A by
    /// p by q
    virtual ambit::Tensor three_integral_block(const std::vector<size_t>& /*A*/,
                                               const std::vector<size_t>& /*p*/,
                                               const std::vector<size_t>& /*q*/) {
        return blank_tensor_;
    }
    /// return ambit tensor of size A by q
    virtual ambit::Tensor three_integral_block_two_index(const std::vector<size_t>& /*A*/,
                                                         size_t /*p*/,
                                                         const std::vector<size_t>& /*q*/) {
        return blank_tensor_;
    }

    virtual void set_tei(size_t, size_t, size_t, size_t, double, bool, bool) {}
    virtual ~OwnIntegrals();

    virtual void make_fock_matrix(SharedMatrix /*gamma_a*/, SharedMatrix /*gamma_b*/) {}
    virtual size_t nthree() const { return 1; }

  private:
    virtual void gather_integrals() {}
    virtual void resort_three(std::shared_ptr<Matrix>& /*threeint*/, std::vector<size_t>& /*map*/) {
    }
    virtual void resort_integrals_after_freezing() {}
    ambit::Tensor blank_tensor_;
};

} // namespace forte
} // namespace psi

#endif // _integrals_h_
