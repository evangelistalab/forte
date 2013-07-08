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
    double frozen_core_energy() const {return core_energy;}
    /// The one-electron integrals
    double roei(size_t p,size_t q) {return one_electron_integrals[p * nmo_ + q];}
    /// The two-electron integrals
    double rtei(size_t p,size_t q,size_t r, size_t s) {return two_electron_integrals[INDEX4(p,q,r,s)];}
private:
    // Class data
    psi::Options& options_;
    IntegralTransform* ints_;
    int nirrep_;
    int nmo_;
    size_t num_oei; // Number of one electron integrals
    size_t num_tei; // Number of two electron integrals
    double      core_energy;  // Frozen-core energy
    std::vector<int> pair_irrep_map;
    std::vector<int> pair_index_map;

    // Class private functions
    void startup();
    void cleanup();
    void read_one_electron_integrals();
    void read_two_electron_integrals();
    void freeze_core();
    int pair_irrep(int p, int q) {return pair_irrep_map[p * nmo_ + q];}
    int pair_index(int p, int q) {return pair_index_map[p * nmo_ + q];}
    double* one_electron_integrals;
    double* two_electron_integrals;
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
