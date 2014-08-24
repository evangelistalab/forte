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

enum IntegralOrdering {Pitzer,MOGroup};

/**
 * Integrals: stores the integrals in Pitzer ordering
 */
class ExplorerIntegrals{
public:
    // ==> Class Constructor and Destructor <==

    ExplorerIntegrals(psi::Options &options,bool restricted);
    ~ExplorerIntegrals();

    // ==> Class Interface <==

    size_t nmo() const {return nmo_;}

    /// Return the frozen core energy
    double frozen_core_energy() const {return core_energy_;}

    /// Scalar component of the Hamiltonian
    double scalar() const {return scalar_;}

    /// The alpha one-electron integrals
    double oei_a(int p,int q) {return one_electron_integrals_a[p * nmo_ + q];}

    /// The beta one-electron integrals
    double oei_b(int p,int q) {return one_electron_integrals_b[p * nmo_ + q];}

    /// The diagonal part of the kinetic energy integrals
    double diag_rkei(int p) {return diagonal_kinetic_energy_integrals[p];}

    /// The diagonal one-electron integrals
    double diag_roei(int p) {return diagonal_one_electron_integrals_a[p];}

    /// The diagonal fock matrix integrals
    double diag_fock_a(int p) {return fock_matrix_a[p * nmo_ + p];}

    /// The diagonal fock matrix integrals
    double diag_fock_b(int p) {return fock_matrix_b[p * nmo_ + p];}

    /// The antisymmetrixed alpha-alpha two-electron integrals in physicist notation <pq||rs>
    double aptei_aa(size_t p,size_t q,size_t r, size_t s) {return aphys_tei_aa[aptei_index(p,q,r,s)];}

    /// The antisymmetrixed alpha-beta two-electron integrals in physicist notation <pq||rs>
    double aptei_ab(size_t p,size_t q,size_t r, size_t s) {return aphys_tei_ab[aptei_index(p,q,r,s)];}

    /// The antisymmetrixed beta-beta two-electron integrals in physicist notation <pq||rs>
    double aptei_bb(size_t p,size_t q,size_t r, size_t s) {return aphys_tei_bb[aptei_index(p,q,r,s)];}

    /// The diagonal antisymmetrixed alpha-alpha two-electron integrals in physicist notation <pq||pq>
    double diag_aptei_aa(size_t p,size_t q) {return diagonal_aphys_tei_aa[p * nmo_ + q];}

    /// The diagonal antisymmetrixed alpha-beta two-electron integrals in physicist notation <pq||pq>
    double diag_aptei_ab(size_t p,size_t q) {return diagonal_aphys_tei_ab[p * nmo_ + q];}

    /// The diagonal antisymmetrixed beta-beta two-electron integrals in physicist notation <pq||pq>
    double diag_aptei_bb(size_t p,size_t q) {return diagonal_aphys_tei_bb[p * nmo_ + q];}

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

private:
    // Class data
    psi::Options& options_;
    bool restricted_;
    IntegralTransform* ints_;
    size_t nirrep_;
    size_t nmo_;
    size_t nmo2_;
    size_t nmo3_;
    size_t nso_;
    size_t num_oei; // Number of one electron integrals
    /// Number of two electron integrals in chemist notation (pq|rs)
    size_t num_tei;
    /// The number of antisymmetrized two-electron integrals in physicist notation <pq||rs>
    size_t num_aptei;
    double core_energy_;  // Frozen-core energy
    std::vector<int> pair_irrep_map;
    std::vector<int> pair_index_map;

    // Class private functions
    void startup();
    void cleanup();
    void read_one_electron_integrals();
    void read_two_electron_integrals();
    void make_diagonal_integrals();
    void freeze_core();
    int pair_irrep(int p, int q) {return pair_irrep_map[p * nmo_ + q];}
    int pair_index(int p, int q) {return pair_index_map[p * nmo_ + q];}
    size_t aptei_index(size_t p,size_t q,size_t r,size_t s) {return nmo3_ * p + nmo2_ * q + nmo_ * r + s;}

    double scalar_;
    double* one_electron_integrals;
    double* one_electron_integrals_a;
    double* one_electron_integrals_b;
    double* diagonal_kinetic_energy_integrals;

    /// Antisymmetrized two-electron integrals in physicist notation
    double* aphys_tei_aa;
    double* aphys_tei_ab;
    double* aphys_tei_bb;

    double* diagonal_aphys_tei_aa;
    double* diagonal_aphys_tei_ab;
    double* diagonal_aphys_tei_bb;

    double* diagonal_one_electron_integrals;
    double* diagonal_one_electron_integrals_a;
    double* diagonal_one_electron_integrals_b;

    double* fock_matrix_a;
    double* fock_matrix_b;
};

}} // End Namespaces

#endif // _integrals_h_
