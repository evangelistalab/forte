#ifndef _integrals_h_
#define _integrals_h_

#define PAIRINDEX(i,j) ((i>j) ? (ioff[(i)]+(j)) : (ioff[(j)]+(i)))
#define four(i,j,k,l) PAIRINDEX(PAIRINDEX(i,j),PAIRINDEX(k,l))

#include <iostream>

#include <boost/shared_ptr.hpp>

#include <libmints/wavefunction.h>
#include <liboptions/liboptions.h>
#include <libtrans/integraltransform.h>

namespace psi{ namespace libadaptive{

/// This decides the type of transformation: resticted vs. unrestricted
enum IntegralSpinRestriction {RestrictedMOs,UnrestrictedMOs};
enum IntegralFrozenCore {RemoveFrozenMOs,KeepFrozenMOs};
enum IntegralType {ConventionalInts,DFInts,CholeskyInts};

/**
 * Integrals: transforms and stores the integrals in Pitzer ordering
 */
class ExplorerIntegrals{
public:
    // ==> Class Constructor and Destructor <==

    /**
     * Constructor
     * @param options The main options object
     * @param restricted Select a restricted or unrestricted transformation (RestrictedMOs = restricted, UnrestrictedMOs = unrestricted).
     * @param resort_frozen_core Determines if the the integral with frozen index are removed
     *        (RemoveFrozenMOs = remove and resort, KeepFrozenMOs = keep all the integrals).
     */
    ExplorerIntegrals(psi::Options &options,IntegralSpinRestriction restricted,IntegralFrozenCore resort_frozen_core);

    /// Destructor
    ~ExplorerIntegrals();

    // ==> Class Interface <==

    /// Return the total number of molecular orbitals (this number includes frozen MOs)
    size_t nmo() const {return nmo_;}

    /// Return the total number of correlated molecular orbitals (this number excludes frozen MOs)
    size_t ncmo() const {return ncmo_;}

    /// The number of correlated MOs per irrep (non frozen).  This is nmopi - nfzcpi - nfzvpi.
    Dimension& ncmopi() {return ncmopi_;}

    /// Return the frozen core energy
    double frozen_core_energy() const {return core_energy_;}

    /// Scalar component of the Hamiltonian
    double scalar() const {return scalar_;}

    /// The alpha one-electron integrals
    double oei_a(int p,int q) {return one_electron_integrals_a[p * aptei_idx_ + q];}

    /// The beta one-electron integrals
    double oei_b(int p,int q) {return one_electron_integrals_b[p * aptei_idx_ + q];}

    /// The diagonal fock matrix integrals
    double diag_fock_a(int p) {return fock_matrix_a[p * aptei_idx_ + p];}

    /// The diagonal fock matrix integrals
    double diag_fock_b(int p) {return fock_matrix_b[p * aptei_idx_ + p];}

    /// The antisymmetrixed alpha-alpha two-electron integrals in physicist notation <pq||rs>
    double aptei_aa(size_t p,size_t q,size_t r, size_t s) {return aphys_tei_aa[aptei_index(p,q,r,s)];}

    /// The antisymmetrixed alpha-beta two-electron integrals in physicist notation <pq||rs>
    double aptei_ab(size_t p,size_t q,size_t r, size_t s) {return aphys_tei_ab[aptei_index(p,q,r,s)];}

    /// The antisymmetrixed beta-beta two-electron integrals in physicist notation <pq||rs>
    double aptei_bb(size_t p,size_t q,size_t r, size_t s) {return aphys_tei_bb[aptei_index(p,q,r,s)];}

    /// The diagonal antisymmetrixed alpha-alpha two-electron integrals in physicist notation <pq||pq>

    double diag_aptei_aa(size_t p,size_t q) {return diagonal_aphys_tei_aa[p * aptei_idx_ + q];}

    /// The diagonal antisymmetrixed alpha-beta two-electron integrals in physicist notation <pq||pq>
    double diag_aptei_ab(size_t p,size_t q) {return diagonal_aphys_tei_ab[p * aptei_idx_ + q];}

    /// The diagonal antisymmetrixed beta-beta two-electron integrals in physicist notation <pq||pq>
    double diag_aptei_bb(size_t p,size_t q) {return diagonal_aphys_tei_bb[p * aptei_idx_ + q];}

    /// Make Fock matrix with respect to a given determinant
    void make_fock_matrix(bool* Ia, bool* Ib);

    /// Make the diagonal matrix elements of the Fock operator for a given set of occupation numbers
    void make_fock_diagonal(bool* Ia, bool* Ib,std::pair<std::vector<double>,std::vector<double> >& fock_diagonals);
    void make_alpha_fock_diagonal(bool* Ia, bool* Ib,std::vector<double>& fock_diagonals);
    void make_beta_fock_diagonal(bool* Ia, bool* Ib,std::vector<double>& fock_diagonals);

    /// Set the value of the scalar part of the Hamiltonian
    /// @param value the new value of the scalar part of the Hamiltonian
    void set_scalar(double value) {scalar_ = value;}

    /// Set the value of the one-electron integrals
    /// @param ints pointer to the integrals
    /// @param the spin type of the integrals
    void set_oei(double** ints,bool alpha);

    /// Set the value of the one-electron integrals
    /// @param the spin type of the integrals
    void set_oei(size_t p, size_t q,double value,bool alpha);

    /// Set the value of the two-electron integrals
    /// @param ints pointer to the integrals
    /// @param the spin type of the integrals
    void set_tei(double**** ints,bool alpha1,bool alpha2);

    /// Set the value of the two-electron integrals
    /// @param the spin type of the integrals
    void set_tei(size_t p, size_t q, size_t r,size_t s,double value,bool alpha1,bool alpha2);


    /// Update all integrals after providing one- and two-electron integrals
    /// via the functions set_oei and set_tei
    /// Sample use:
    ///     ExplorerIntegrals* ints = new ...
    ///
    ///     // One-electron integrals are updated
    ///     ints->set_oei(oei_aa,true);
    ///     ints->set_oei(oei_bb,false);
    ///     ints->update_integrals();
    void update_integrals();

    /// Update the integrals with a new set of MO coefficients
    void retransform_integrals();

    /// Return the number of frozen core orbitals per irrep
    Dimension& frzcpi() {return frzcpi_;}
    /// Return the number of frozen virtual orbitals per irrep
    Dimension& frzvpi() {return frzvpi_;}

    /// Compute df integrals
    void compute_df_integrals();
    /// Compute cholesky integrals
    void compute_chol_integrals();
    //const int* qt_pitzer_;
    std::vector<std::pair<std::vector<int>, std::vector<double> >  >chol_ints;
    std::vector<std::pair<std::vector<int>, std::vector<double> > > df_ints;
    std::vector<std::pair<std::vector<int>, std::vector<double> > > conv_ints;
    void debug_ints();

private:

    // ==> Class data <==

    /// The options object
    psi::Options& options_;

    /// Are we doing a spin-restricted computation?
    IntegralSpinRestriction restricted_;

    /// Do we have to resort the integrals to eliminate frozen orbitals?
    IntegralFrozenCore resort_frozen_core_;

    /// The IntegralTransform object used by this class
    IntegralTransform* ints_;

    /// Number of irreps
    size_t nirrep_;

    /// The number of MOs, including the ones that are frozen.
    size_t nmo_;

    /// The number of correlated MOs (excluding frozen).  This is nmo - nfzc - nfzv.
    size_t ncmo_;

    /// The number of MOs per irrep.
    Dimension nmopi_;
    /// The number of frozen core MOs per irrep.
    Dimension frzcpi_;
    /// The number of frozen unoccupied MOs per irrep.
    Dimension frzvpi_;
    /// The number of correlated MOs per irrep (non frozen).  This is nmopi - nfzcpi - nfzvpi.
    Dimension ncmopi_;

    size_t aptei_idx_;
    size_t nso_;

    /// Number of one electron integrals
    size_t num_oei;

    /// Number of two electron integrals in chemist notation (pq|rs)
    size_t num_tei;

    /// The number of antisymmetrized two-electron integrals in physicist notation <pq||rs>
    size_t num_aptei;

    /// Frozen-core energy
    double core_energy_;
    std::vector<int> pair_irrep_map;
    std::vector<int> pair_index_map;

    double scalar_;

    /// One-electron integrals stored as a vector
    double* one_electron_integrals_a;
    double* one_electron_integrals_b;

    /// Antisymmetrized two-electron integrals in physicist notation
    double* aphys_tei_aa;
    double* aphys_tei_ab;
    double* aphys_tei_bb;

    double* diagonal_aphys_tei_aa;
    double* diagonal_aphys_tei_ab;
    double* diagonal_aphys_tei_bb;

    double* fock_matrix_a;
    double* fock_matrix_b;

    // ==> Class private functions <==

    void startup();
    void cleanup();

    /// Allocate memory
    void allocate();

    /// Deallocate memory
    void deallocate();

    /// Transform the integrals
    void transform_integrals();
    void read_one_electron_integrals();
    void read_two_electron_integrals();
    void make_diagonal_integrals();

    /// This function manages freezing core and virtual orbitals
    void freeze_core_orbitals();

    /// Compute the frozen core energy
    void compute_frozen_core_energy();

    /// Compute the one-body operator modified by the frozen core orbitals
    void compute_frozen_one_body_operator();

    /// Remove the doubly occupied and virtual orbitals and resort the rest so that
    /// we are left only with ncmo = nmo - nfzc - nfzv
    void resort_integrals_after_freezing();

    void resort_two(double*& ints,std::vector<size_t>& map);
    void resort_four(double*& ints,std::vector<size_t>& map);

    /// Freeze the doubly occupied and virtual orbitals but do not resort the integrals
    void freeze_core_full();

    /// An addressing function to retrieve the two-electron integrals
    size_t aptei_index(size_t p,size_t q,size_t r,size_t s) {return aptei_idx_ * aptei_idx_ * aptei_idx_ * p + aptei_idx_ * aptei_idx_ * q + aptei_idx_ * r + s;}
};

}} // End Namespaces

#endif // _integrals_h_
