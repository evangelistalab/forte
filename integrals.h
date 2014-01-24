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
    // Class Constructor and Destructor
    ExplorerIntegrals(psi::Options &options);
    ~ExplorerIntegrals();

    // Class Interface
    /// Return the frozen core energy
    double frozen_core_energy() const {return core_energy_;}

//    /// The one-electron integrals
//    double roei(int p,int q) {return one_electron_integrals[p * nmo_ + q];}

    /// The alpha one-electron integrals
    double oei_a(int p,int q) {return one_electron_integrals_a[p * nmo_ + q];}

    /// The beta one-electron integrals
    double oei_b(int p,int q) {return one_electron_integrals_b[p * nmo_ + q];}

    /// The diagonal part of the kinetic energy integrals
    double diag_rkei(int p) {return diagonal_kinetic_energy_integrals[p];}

    /// The diagonal one-electron integrals
    double diag_roei(int p) {return diagonal_one_electron_integrals_a[p];}

//    /// The diagonal alpha one-electron integrals
//    double diag_oei_a(int p) {return diagonal_one_electron_integrals_a[p];}

//    /// The diagonal beta one-electron integrals
//    double diag_oei_b(int p) {return diagonal_one_electron_integrals_b[p];}

    /// The diagonal fock matrix integrals
    double diag_fock_a(int p) {return fock_matrix_a[p * nmo_ + p];}

    /// The diagonal fock matrix integrals
    double diag_fock_b(int p) {return fock_matrix_b[p * nmo_ + p];}

    /// The two-electron integrals in chemist notation (pq|rs)
//    double rtei(size_t p,size_t q,size_t r, size_t s) {return two_electron_integrals[INDEX4(p,q,r,s)];}

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

//    /// The diagonal two-electron integrals (Coulomb)
//    double diag_c_rtei(int p,int q) {return diagonal_c_integrals[p * nmo_ + q];}

//    /// The diagonal two-electron integrals (Coulomb + Exchange)
//    double diag_ce_rtei(int p, int q) {return diagonal_ce_integrals[p * nmo_ + q];}

    /// Make Fock matrix with respect to a given determinant
    void make_fock_matrix(bool* Ia, bool* Ib);

    /// Make the diagonal matrix elements of the Fock operator for a given set of occupation numbers
    void make_fock_diagonal(bool* Ia, bool* Ib,std::pair<std::vector<double>,std::vector<double> >& fock_diagonals);
    void make_alpha_fock_diagonal(bool* Ia, bool* Ib,std::vector<double>& fock_diagonals);
    void make_beta_fock_diagonal(bool* Ia, bool* Ib,std::vector<double>& fock_diagonals);

    /// Set the value of the one-electron integrals
    /// @param ints pointer to the integrals
    /// @param the spin type of the integrals
    void set_oei(double** ints,bool alpha);

    /// Set the value of the two-electron integrals
    /// @param ints pointer to the integrals
    /// @param the spin type of the integrals
    void set_tei(double**** ints,bool alpha1,bool alpha2);

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

    double* one_electron_integrals;
    double* one_electron_integrals_a;
    double* one_electron_integrals_b;
    double* diagonal_kinetic_energy_integrals;
    double* two_electron_integrals;

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
//    double* diagonal_c_integrals;
//    double* diagonal_c_integrals_aa;
//    double* diagonal_c_integrals_ab;
//    double* diagonal_c_integrals_bb;
//    double* diagonal_ce_integrals;
//    double* diagonal_ce_integrals_aa;
//    double* diagonal_ce_integrals_ab;
//    double* diagonal_ce_integrals_bb;
    double* fock_matrix_a;
    double* fock_matrix_b;
};


//  double get_alfa_h0_core_energy() const {return alfa_h0_core_energy;}
//  double get_beta_h0_core_energy() const {return beta_h0_core_energy;}

//  // Get functions for the integrals labeled in the all set
//  double get_oei_aa_all(int p, int q) {return oei_aa[all_to_mo[p]][all_to_mo[q]];}
//  double get_oei_bb_all(int p, int q) {return oei_bb[all_to_mo[p]][all_to_mo[q]];}
//  double get_h_aa_all(int p, int q) {return h_aa[all_to_mo[p]][all_to_mo[q]];}
//  double get_h_bb_all(int p, int q) {return h_bb[all_to_mo[p]][all_to_mo[q]];}
//  double get_fock_aa_all(int p, int q,int ref) {return f_aa[ref][all_to_mo[p]][all_to_mo[q]];}
//  double get_fock_bb_all(int p, int q,int ref) {return f_bb[ref][all_to_mo[p]][all_to_mo[q]];}
//  double get_tei_aaaa_all(int p, int q,int r, int s) {return tei_aaaa[four(all_to_mo[p],all_to_mo[q],all_to_mo[r],all_to_mo[s])];}
//  double get_tei_bbbb_all(int p, int q,int r, int s) {return tei_bbbb[four(all_to_mo[p],all_to_mo[q],all_to_mo[r],all_to_mo[s])];}
//  double get_tei_aabb_all(int p, int q,int r, int s) {return tei_aabb[PAIRINDEX(all_to_mo[p],all_to_mo[q])][PAIRINDEX(all_to_mo[r],all_to_mo[s])];}

//  // Get functions for the integrals labeled in the mos set
//  double get_oei_aa(int p, int q) {return oei_aa[p][q];}
//  double get_oei_bb(int p, int q) {return oei_bb[p][q];}
//  double get_h_aa(int p, int q) {return h_aa[p][q];}
//  double get_h_bb(int p, int q) {return h_bb[p][q];}
//  double get_tei_aaaa(int p, int q,int r, int s) {return tei_aaaa[four(p,q,r,s)];}
//  double get_tei_bbbb(int p, int q,int r, int s) {return tei_bbbb[four(p,q,r,s)];}
//  double get_tei_aabb(int p, int q,int r, int s) {return tei_aabb[PAIRINDEX(p,q)][PAIRINDEX(r,s)];}
//  double get_atei_aaaa(int p, int q,int r, int s) {return tei_aaaa[four(p,r,q,s)] - tei_aaaa[four(p,s,q,r)];}
//  double get_atei_bbbb(int p, int q,int r, int s) {return tei_bbbb[four(p,r,q,s)] - tei_bbbb[four(p,s,q,r)];}
//  double get_atei_abab(int p, int q,int r, int s) {return tei_aabb[PAIRINDEX(p,r)][PAIRINDEX(q,s)];}
//  double get_f_avg_aa(int p, int q) {return f_avg_aa[p][q];}
//  double get_f_avg_bb(int p, int q) {return f_avg_bb[p][q];}

//  double get_alfa_denominator(int* occ,int* vir,int n,int mu);
//  double get_beta_denominator(int* occ,int* vir,int n,int mu);
//  double get_alfa_denominator(const int* occvir,int n,int mu);
//  double get_beta_denominator(const int* occvir,int n,int mu);
//  double get_alfa_avg_denominator(const int* occvir,int n);
//  double get_beta_avg_denominator(const int* occvir,int n);
//  void   decompose_tei(double threshold);
//  void   decompose_tei2(double threshold);
//  void   make_f_avg(double** opdm_aa,double** opdm_bb);

//  double** get_alfa_fock_matrix(int mu) {return f_aa[mu];}
//  double** get_beta_fock_matrix(int mu) {return f_bb[mu];}
//private:

//  void read_oei();
//  void read_tei();
//  void freeze_core();
//  void make_h();
//  void make_f();
//  double aaaa(int p, int q,int r, int s) {return tei_aaaa[four(static_cast<size_t>(p),static_cast<size_t>(q),static_cast<size_t>(r),static_cast<size_t>(s))];}
//  double bbbb(int p, int q,int r, int s) {return tei_bbbb[four(static_cast<size_t>(p),static_cast<size_t>(q),static_cast<size_t>(r),static_cast<size_t>(s))];}
//  double aabb(int p, int q,int r, int s) {return tei_aabb[PAIRINDEX(static_cast<size_t>(p),static_cast<size_t>(q))][PAIRINDEX(static_cast<size_t>(r),static_cast<size_t>(s))];}
//  double bbaa(int p, int q,int r, int s) {return tei_aabb[PAIRINDEX(static_cast<size_t>(r),static_cast<size_t>(s))][PAIRINDEX(static_cast<size_t>(p),static_cast<size_t>(q))];}

//  int         nmo;       // Number of molecular orbitals (frozen + non-frozen)
//  size_t      noei;      // Number of packed one-electron integrals (p>=q)
//  size_t      ntei;      // Number of packed two-electron integrals (p>=q) >= (r=>s)
//  double**    oei_aa;    // Bare one-particle matrix elements of the Hamiltonian aa
//  double**    oei_bb;    // Bare one-particle matrix elements of the Hamiltonian bb
//  double**    h_aa;      // Effective one-particle matrix elements of the Hamiltonian aa
//  double**    h_bb;      // Effective one-particle matrix elements of the Hamiltonian bb
//  double***   f_aa;      // Fock matrix elements aa
//  double***   f_bb;      // Fock matrix elements bb
//  double**    f_avg_aa;  // Average fock matrix elements aa
//  double**    f_avg_bb;  // Average fock matrix elements bb
//  double*     tei_aaaa;  // Two-electron integrals in chemist notation aaaa
//  double*     tei_bbbb;  // Two-electron integrals in chemist notation bbbb
//  double**    tei_aabb;  // Two-electron integrals in chemist notation aabb
//  int*        all_to_mo; // Mapping from non-frozen orbitals to all orbitals in Pitzer ordering
//  double      alfa_h0_core_energy;
//  double      beta_h0_core_energy;
//  static const int ioffmax = 10000000;

}} // End Namespaces

#endif // _integrals_h_
