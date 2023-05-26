/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2023 by its authors (see COPYING, COPYING.LESSER,
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

#include <vector>

#include "psi4/libfock/jk.h"
#include "psi4/libmints/dimension.h"
#include "ambit/blocked_tensor.h"

class Tensor;

namespace psi {
class Options;
class Matrix;
class Vector;
class Wavefunction;
class Dimension;
class BasisSet;
} // namespace psi

namespace forte {

class ForteOptions;
class MOSpaceInfo;

/**
 * @brief The IntegralSpinRestriction enum
 *
 * This is used to distinguish between restricted/unrestricted computations
 */
enum class IntegralSpinRestriction { Restricted, Unrestricted };

/**
 * @brief The IntegralType enum
 *
 * This decides the type of integral used in a Forte computation
 */
enum IntegralType { Conventional, DF, Cholesky, DiskDF, DistDF, Custom };

/**
 * @brief The order of three-index integrals when calling
 *
 * In Forte, the auxiliary index is by default the first index (i.e., Qpq).
 * However, pqQ order is more convenient for implementing DF-MRPT2.
 */
enum ThreeIntsBlockOrder { Qpq, pqQ };

/**
 * @brief The ForteIntegrals class is a base class for transforming and storing MO integrals
 *
 * ForteIntegrals provides a common interface for using one- and two-electron integrals
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
 *
 * There are several classes that derive from ForteIntegrals (Convetional, DF, ...) and these
 * are best created via the helper function
 *
 *     std::shared_ptr<ForteIntegrals> make_forte_integrals(...)
 *
 * defined in 'make_integrals.h'
 *
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
    ForteIntegrals(std::shared_ptr<ForteOptions> options,
                   std::shared_ptr<psi::Wavefunction> ref_wfn,
                   std::shared_ptr<MOSpaceInfo> mo_space_info, IntegralType integral_type,
                   IntegralSpinRestriction restricted);

    /**
     * @brief Class constructor
     * @param options The main options object
     * @param restricted Select a restricted or unrestricted transformation
     * @param mo_space_info The MOSpaceInfo object
     */
    ForteIntegrals(std::shared_ptr<ForteOptions> options,
                   std::shared_ptr<MOSpaceInfo> mo_space_info, IntegralType integral_type,
                   IntegralSpinRestriction restricted);

    /// Virtual destructor to enable deletion of a Derived* through a Base*
    virtual ~ForteIntegrals() = default;

    // ==> Class Interface <==

    /// Common initializer for all types of integrals
    void common_initialize();

    virtual void initialize() = 0;

    /// Skip integral transformation
    bool skip_build_;

    /// Return Ca
    std::shared_ptr<psi::Matrix> Ca() const;
    /// Return Cb
    std::shared_ptr<psi::Matrix> Cb() const;

    /// Return nuclear repulsion energy
    double nuclear_repulsion_energy() const;

    /// temporary solution for not having a Wavefunction
    std::shared_ptr<psi::Wavefunction> wfn();

    /// Return the Pis4 JK object
    std::shared_ptr<psi::JK> jk();

    /// Enum class for the status of Pis4 JK
    enum class JKStatus { empty, initialized, finalized };
    /// Return the status of Psi4 JK object
    JKStatus jk_status();
    /// Finalize Psi4 JK object
    void jk_finalize();

    // The number of symmetry-adapted orbitals
    // see https://github.com/psi4/psi4/wiki/OrbitalDimensions
    size_t nso() const;
    /// Return the number of symmetry-adapted orbitals per irrep
    const psi::Dimension& nsopi() const;

    /// Return the total number of molecular orbitals (this number includes frozen MOs)
    size_t nmo() const;

    /// Return the number of irreducible representations
    int nirrep() const;

    /// Return the number of frozen core orbitals per irrep
    const psi::Dimension& frzcpi() const;
    /// Return the number of frozen virtual orbitals per irrep
    const psi::Dimension& frzvpi() const;
    /// The number of correlated MOs per irrep (non frozen).  This is nmopi - nfzcpi - nfzvpi.
    const psi::Dimension& ncmopi() const;

    /// Return the total number of correlated molecular orbitals (this number excludes frozen MOs)
    size_t ncmo() const;

    /// Return the mapping from correlated MO to full MO (frozen + correlated)
    const std::vector<size_t>& cmotomo() const;

    /// Set printing level
    void set_print(int print);

    /// Return the number of auxiliary functions
    virtual size_t nthree() const;

    /// Return the frozen core energy
    double frozen_core_energy();

    /// Scalar component of the Hamiltonian
    double scalar() const;

    /// The alpha one-electron integrals
    double oei_a(size_t p, size_t q) const;

    /// The beta one-electron integrals
    double oei_b(size_t p, size_t q) const;

    /// Get the alpha Fock matrix element
    /// @param p The bra index
    /// @param q The ket index
    /// @param corr Whether indices p, q start counting from correlated orbitals
    /// @return the alpha Fock matrix element F_{pq}
    double get_fock_a(size_t p, size_t q, bool corr = true) const;
    /// Get the beta Fock matrix element
    /// @param p The bra index
    /// @param q The ket index
    /// @param corr Whether indices p, q start counting from correlated orbitals
    /// @return the beta Fock matrix element F_{pq}
    double get_fock_b(size_t p, size_t q, bool corr = true) const;

    /// Get the alpha Fock matrix in Psi4 matrix
    /// @param corr Whether to return only the part of correlated orbitals
    /// @return the alpha Fock matrix
    std::shared_ptr<psi::Matrix> get_fock_a(bool corr = true) const;
    /// Get the beta fock matrix in Psi4 matrix
    /// @param corr Whether to return only the part of correlated orbitals
    /// @return the beta Fock matrix
    std::shared_ptr<psi::Matrix> get_fock_b(bool corr = true) const;

    /// The antisymmetrixed alpha-alpha two-electron integrals in physicist notation <pq||rs>
    virtual double aptei_aa(size_t p, size_t q, size_t r, size_t s) = 0;
    /// The antisymmetrixed alpha-beta two-electron integrals in physicist notation <pq|rs>
    virtual double aptei_ab(size_t p, size_t q, size_t r, size_t s) = 0;
    /// The antisymmetrixed beta-beta two-electron integrals in physicist notation <pq||rs>
    virtual double aptei_bb(size_t p, size_t q, size_t r, size_t s) = 0;

    /// @return a tensor with a block of the alpha one-electron integrals
    ambit::Tensor oei_a_block(const std::vector<size_t>& p, const std::vector<size_t>& q);
    /// @return a tensor with a block of the beta one-electron integrals
    ambit::Tensor oei_b_block(const std::vector<size_t>& p, const std::vector<size_t>& q);

    /// Grab a block of the integrals and return a tensor
    /// p, q, r, s correspond to the vector of indices you want for your tensor
    /// if p, q, r, s is equal to an array of all of the mos, then this will
    /// @return a tensor with a block of the alpha-alpha antisymmetrized two-electron integrals
    virtual ambit::Tensor aptei_aa_block(const std::vector<size_t>& p, const std::vector<size_t>& q,
                                         const std::vector<size_t>& r,
                                         const std::vector<size_t>& s) = 0;
    /// @return a tensor with a block of the alpha-beta antisymmetrized two-electron integrals
    virtual ambit::Tensor aptei_ab_block(const std::vector<size_t>& p, const std::vector<size_t>& q,
                                         const std::vector<size_t>& r,
                                         const std::vector<size_t>& s) = 0;
    /// @return a tensor with a block of the beta-beta antisymmetrized two-electron integrals
    virtual ambit::Tensor aptei_bb_block(const std::vector<size_t>& p, const std::vector<size_t>& q,
                                         const std::vector<size_t>& r,
                                         const std::vector<size_t>& s) = 0;

    // Three-index integral functions (DF, Cholesky)
    virtual ambit::Tensor three_integral_block(const std::vector<size_t>&,
                                               const std::vector<size_t>&,
                                               const std::vector<size_t>&,
                                               ThreeIntsBlockOrder order = Qpq);

    /// This function is only used by DiskDF and it is used to go from a Apq->Aq tensor
    virtual ambit::Tensor three_integral_block_two_index(const std::vector<size_t>& A, size_t p,
                                                         const std::vector<size_t>&);

    /// Expert Option: just try and use three_integral
    virtual double** three_integral_pointer();

    /// Make the generalized Fock matrix (closed-shell + active)
    /// @param Da The alpha 1RDM (nactv x nactv, no symmetry) from RDMs class
    /// @param Db The beta 1RDM (nactv x nactv, no symmetry) from RDMs class
    virtual void make_fock_matrix(ambit::Tensor Da, ambit::Tensor Db) = 0;

    /// Make the closed-shell Fock matrix in MO basis (include frozen orbitals)
    /// @param dim_start Dimension for the starting index (per irrep) of closed-shell orbitals
    /// @param dim_end Dimension for the ending index (per irrep) of closed-shell orbitals
    /// @return alpha Fock, beta Fock, and closed-shell energy
    /// spin orbital equation:
    /// F_{pq} = h_{pq} + \sum_{i}^{closed} <pi||qi>
    /// e_closed = \sum_{i}^{closed} h_{ii} + 0.5 * \sum_{ij}^{closed} <ij||ij>
    virtual std::tuple<std::shared_ptr<psi::Matrix>, std::shared_ptr<psi::Matrix>, double>
    make_fock_inactive(psi::Dimension dim_start, psi::Dimension dim_end) = 0;

    /// Make the active Fock matrix in MO basis (include frozen orbitals)
    /// @param Da The alpha 1RDM (nactv x nactv, no symmetry) from RDMs class
    /// @param Db The beta 1RDM (nactv x nactv, no symmetry) from RDMs class
    /// @return alpha Fock, beta Fock
    /// spin orbital equation:
    /// F_{pq} = \sum_{uv}^{active} <pu||qv> * gamma_{uv}
    virtual std::tuple<std::shared_ptr<psi::Matrix>, std::shared_ptr<psi::Matrix>>
    make_fock_active(ambit::Tensor Da, ambit::Tensor Db) = 0;

    /// Make the active Fock matrix in MO basis (include frozen orbitals)
    /// @param D The spin-summed 1RDM in std::shared_ptr<psi::Matrix> form
    /// @return Fock matrix
    virtual std::shared_ptr<psi::Matrix>
    make_fock_active_restricted(std::shared_ptr<psi::Matrix> D) = 0;

    /// Make the active Fock matrix in MO basis (include frozen orbitals)
    /// @param Da The alpha 1RDM in std::shared_ptr<psi::Matrix> form
    /// @param Db The beta 1RDM in std::shared_ptr<psi::Matrix> form
    /// @return alpha Fock matrix, beta Fock matrix
    virtual std::tuple<std::shared_ptr<psi::Matrix>, std::shared_ptr<psi::Matrix>>
    make_fock_active_unrestricted(std::shared_ptr<psi::Matrix> Da,
                                  std::shared_ptr<psi::Matrix> Db) = 0;

    /// Set Fock matrix
    void set_fock_matrix(std::shared_ptr<psi::Matrix> fa, std::shared_ptr<psi::Matrix> fb);

    /// Set nuclear repulstion energy
    void set_nuclear_repulsion(double value);

    /// Set the value of the scalar part of the Hamiltonian
    /// @param value the new value of the scalar part of the Hamiltonian
    void set_scalar(double value);

    /// Set the value of the one-electron integrals (stored with no symmetry)
    /// @param oei_a vector of alpha one-electron integrals
    /// @param oei_b vector of beta one-electron integrals
    void set_oei_all(const std::vector<double>& oei_a, const std::vector<double>& oei_b);

    /// Set the value of the two-electron integrals (stored with no symmetry)
    /// @param tei_aa vector of antisymmetrized alpha-alpha two-electron integrals
    /// @param tei_ab vector of antisymmetrized alpha-alpha two-electron integrals
    /// @param tei_bb vector of antisymmetrized alpha-alpha two-electron integrals
    void set_tei_all(const std::vector<double>& tei_aa, const std::vector<double>& tei_ab,
                     const std::vector<double>& tei_bb);

    /// Set the value of the one-electron integrals
    /// @param p the integral index
    /// @param q the integral index
    /// @param value the value of the integral
    /// @param alpha the spin type of the integrals
    void set_oei(size_t p, size_t q, double value, bool alpha);

    /// Set the value of the two-electron integrals
    virtual void set_tei(size_t p, size_t q, size_t r, size_t s, double value, bool alpha1,
                         bool alpha2) = 0;

    /// Rotate the MO coefficients, update psi::Wavefunction, and re-transform integrals
    /// @param Ua the alpha unitary transformation matrix
    /// @param Ub the beta unitary transformation matrix
    /// @param re_transform re-transform integrals if true
    void rotate_orbitals(std::shared_ptr<psi::Matrix> Ua, std::shared_ptr<psi::Matrix> Ub,
                         bool re_transform = true);

    /// Copy these MO coeffs to class variables, update psi::Wavefunction, and re-transform
    /// integrals
    /// @param Ca the alpha MO coefficients
    /// @param Cb the beta MO coefficients
    /// @param re_transform re-transform integrals if true
    virtual void update_orbitals(std::shared_ptr<psi::Matrix> Ca, std::shared_ptr<psi::Matrix> Cb,
                                 bool re_transform = true);

    /// Make the orbital phase consistent when updating orbitals
    /// @param U the unitary transformation matrix so that C_new = C_old * U
    /// @param is_alpha target Ca if true else Cb
    /// @param debug print MO overlap and transformation matrix if true
    /// @return true if success
    bool fix_orbital_phases(std::shared_ptr<psi::Matrix> U, bool is_alpha, bool debug = false);

    /// Return the type of spin restriction enforced
    IntegralSpinRestriction spin_restriction() const;
    /// Return the type of integral used
    IntegralType integral_type() const;
    /// Return the one-body symmetry integrals
    std::shared_ptr<psi::Matrix> OneBody_symm() const;
    /// Return the one-body AO integrals
    std::shared_ptr<psi::Matrix> OneBodyAO() const;

    virtual int ga_handle();

    /// Print the details of the integral transformation
    void print_info();
    /// Print the one- and two-electron integrals to the output
    void print_ints();

    /// Orbital coefficients in AO x MO basis where MO is Pitzer order
    virtual std::shared_ptr<psi::Matrix> Ca_AO() const = 0;

    /// Obtain AO dipole integrals [X, Y, Z]
    /// Each direction is a std::shared_ptr<psi::Matrix> of dimension nao * nao
    std::vector<std::shared_ptr<psi::Matrix>> ao_dipole_ints() const;

    /// Obtain AO quadrupole integrals [XX, XY, XZ, YY, YZ, ZZ]
    std::vector<std::shared_ptr<psi::Matrix>> ao_quadrupole_ints() const;

    /// Compute MO dipole integrals (frozen orbitals included)
    /// @return a vector of MO dipole ints in X, Y, Z order,
    ///         each of which is a nmo x nmo std::shared_ptr<psi::Matrix> in Pitzer order
    virtual std::vector<std::shared_ptr<psi::Matrix>> mo_dipole_ints() const;

    /// Compute MO quadrupole integrals (frozen orbitals included)
    /// @return a vector of MO quadrupole ints in XX, XY, XZ, YY, YZ, ZZ order,
    ///         each of which is a nmo x nmo std::shared_ptr<psi::Matrix> in Pitzer order
    virtual std::vector<std::shared_ptr<psi::Matrix>> mo_quadrupole_ints() const;

  protected:
    // ==> Class data <==

    /// The options object
    std::shared_ptr<ForteOptions> options_;

    /// The MOSpaceInfo object
    std::shared_ptr<MOSpaceInfo> mo_space_info_;

    /// The Wavefunction object
    std::shared_ptr<psi::Wavefunction> wfn_;

    /// The integral_type
    IntegralType integral_type_;

    /// Are we doing a spin-restricted computation?
    IntegralSpinRestriction spin_restriction_;

    // Ca matrix from psi
    std::shared_ptr<psi::Matrix> Ca_;

    // Cb matrix from psi
    std::shared_ptr<psi::Matrix> Cb_;

    /// Number of irreps
    int nirrep_;

    /// The number of symmetry-adapted orbitals
    size_t nso_;
    /// The number of MOs, including the ones that are frozen.
    size_t nmo_;
    /// The number of correlated MOs (excluding frozen).  This is nmo - nfzc - nfzv.
    size_t ncmo_;

    /// The mapping from correlated MO to full MO (frozen + correlated)
    std::vector<size_t> cmotomo_;
    /// The mapping from full MO to irrep and relative indices
    std::vector<std::pair<size_t, size_t>> mo_to_relmo_;

    /// The number of symmetry-adapted orbitals per irrep.
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

    /// The number of orbitals used in indexing routines (nmo or ncmo if core orbitals are
    /// frozen) The correct value is set by the integrals class
    size_t aptei_idx_;

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

    // Nuclear repulsion energy
    double nucrep_ = 0.0;

    /// Frozen-core energy
    double frozen_core_energy_ = 0.0;

    /// Scalar energy term
    double scalar_energy_ = 0.0;

    /// Full one-electron integrals stored as a vector (includes frozen orbitals)
    std::vector<double> full_one_electron_integrals_a_;
    std::vector<double> full_one_electron_integrals_b_;

    /// One-electron integrals stored as a vector
    std::vector<double> one_electron_integrals_a_;
    std::vector<double> one_electron_integrals_b_;

    /// JK object from Psi4
    std::shared_ptr<psi::JK> JK_;

    /// Status of the JK object
    JKStatus JK_status_ = JKStatus::empty;

    /// Fock matrix (including frozen orbitals)
    std::shared_ptr<psi::Matrix> fock_a_;
    std::shared_ptr<psi::Matrix> fock_b_;

    /// Two-electron integrals stored as a vector with redundant elements (no permutational
    /// symmetry). These are addressed with the function aptei_index
    std::vector<double> aphys_tei_aa_;
    std::vector<double> aphys_tei_ab_;
    std::vector<double> aphys_tei_bb_;

    /// The type of tensor that ambit uses -> CoreTensor
    ambit::TensorType tensor_type_ = ambit::CoreTensor;

    /// How much memory each integral takes up
    double int_mem_;
    /// Control printing of timings
    int print_ = 1;

    /// The One Electron Integrals (T + V) in SO Basis
    std::shared_ptr<psi::Matrix> OneBody_symm_;

    /// AO dipole integrals
    std::vector<std::shared_ptr<psi::Matrix>> dipole_ints_ao_;
    /// AO quadrupole integrals
    std::vector<std::shared_ptr<psi::Matrix>> quadrupole_ints_ao_;
    /// Compute AO dipole and quadrupole integrals
    virtual void build_multipole_ints_ao();

    // ==> Class private functions <==

    /// Class initializer
    void startup();

    void read_information();

    void allocate();

    /// Test if two matrices are approximately identical
    bool test_orbital_spin_restriction(std::shared_ptr<psi::Matrix> A,
                                       std::shared_ptr<psi::Matrix> B) const;

    /// An addressing function to for two-electron integrals
    /// @return the address of the integral <pq|rs> or <pq||rs>
    size_t aptei_index(size_t p, size_t q, size_t r, size_t s) {
        return aptei_idx_ * aptei_idx_ * aptei_idx_ * p + aptei_idx_ * aptei_idx_ * q +
               aptei_idx_ * r + s;
    }

    void _undefined_function(const std::string& method) const;

    // ==> Class private virtual functions <==

    /// This function manages freezing core and virtual orbitals
    virtual void freeze_core_orbitals();

    /// Compute the one-body operator modified by the frozen core orbitals
    virtual void compute_frozen_one_body_operator();

    /// Function used to rotate MOs during contructor
    virtual void rotate_mos();

    /// Computes/reads two-electron integrals (see CD/DF/Conventional classes for
    /// implementation)
    virtual void gather_integrals() = 0;

    /// Remove the doubly occupied and virtual orbitals and resort the rest so
    /// that we are left only with ncmo = nmo - nfzc - nfzv
    virtual void resort_integrals_after_freezing() = 0;
};

/**
 * @brief Interface to integrals read from psi4
 */
class Psi4Integrals : public ForteIntegrals {
  public:
    Psi4Integrals(std::shared_ptr<ForteOptions> options, std::shared_ptr<psi::Wavefunction> ref_wfn,
                  std::shared_ptr<MOSpaceInfo> mo_space_info, IntegralType integral_type,
                  IntegralSpinRestriction restricted);

    /// Make the generalized Fock matrix using Psi4 JK object
    void make_fock_matrix(ambit::Tensor Da, ambit::Tensor Db) override;

    /// Make the closed-shell Fock matrix using Psi4 JK object
    std::tuple<std::shared_ptr<psi::Matrix>, std::shared_ptr<psi::Matrix>, double>
    make_fock_inactive(psi::Dimension dim_start, psi::Dimension dim_end) override;

    /// Make the active Fock matrix using Psi4 JK object
    std::tuple<std::shared_ptr<psi::Matrix>, std::shared_ptr<psi::Matrix>>
    make_fock_active(ambit::Tensor Da, ambit::Tensor Db) override;

    /// Make the active Fock matrix using restricted equation
    std::shared_ptr<psi::Matrix>
    make_fock_active_restricted(std::shared_ptr<psi::Matrix> D) override;

    /// Make the active Fock matrix using unrestricted equation
    std::tuple<std::shared_ptr<psi::Matrix>, std::shared_ptr<psi::Matrix>>
    make_fock_active_unrestricted(std::shared_ptr<psi::Matrix> Da,
                                  std::shared_ptr<psi::Matrix> Db) override;

    /// Orbital coefficients in AO x MO basis, where MO is in Pitzer order
    std::shared_ptr<psi::Matrix> Ca_AO() const override;

    /// Build and return MO dipole integrals (X, Y, Z) in Pitzer order
    std::vector<std::shared_ptr<psi::Matrix>> mo_dipole_ints() const override;

    /// Build and return MO quadrupole integrals (XX, XY, XZ, YY, YZ, ZZ) in Pitzer order
    std::vector<std::shared_ptr<psi::Matrix>> mo_quadrupole_ints() const override;

  private:
    void base_initialize_psi4();
    void setup_psi4_ints();
    void transform_one_electron_integrals();
    void compute_frozen_one_body_operator() override;
    void update_orbitals(std::shared_ptr<psi::Matrix> Ca, std::shared_ptr<psi::Matrix> Cb,
                         bool re_transform = true) override;
    void rotate_mos() override;

    /// Build AO dipole and quadrupole integrals
    void build_multipole_ints_ao() override;

    /// Make a shared pointer to a Psi4 JK object
    void make_psi4_JK();
    /// Call JK intialize
    void jk_initialize(double mem_percentage = 0.8, int print_level = 1);

    /// AO Fock control
    enum class FockAOStatus { none, inactive, generalized };
    FockAOStatus fock_ao_level_ = FockAOStatus::none;

  protected:
    void freeze_core_orbitals() override;

    // threshold for DF fitting condition (Psi4)
    double df_fitting_cutoff_;
    // threshold for Schwarz cutoff (Psi4)
    double schwarz_cutoff_;
};

} // namespace forte

#endif // _integrals_h_
