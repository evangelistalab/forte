/*
 *@BEGIN LICENSE
 *
 * Libadaptive: an ab initio quantum chemistry software package
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with this program; if not, write to the Free Software Foundation, Inc.,
 * 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 *
 *@END LICENSE
 */

#ifndef _fci_vector_
#define _fci_vector_

#include <vector>
#include "boost/shared_ptr.hpp"

#include "libmints/matrix.h"
#include "integrals.h"
#include "string_lists.h"

#define CAPRICCIO_USE_DAXPY 1
#define CAPRICCIO_USE_UNROLL 0

namespace psi{ namespace forte{

/**
 * @brief The FCIIntegrals class stores integrals necessary for FCI calculations
 */
class FCIIntegrals
{
public:

    // ==> Class Constructors <==

    /// Constructor based on StringLists
    FCIIntegrals(std::shared_ptr<StringLists> lists, ForteIntegrals* ints);
    /// Constructor based on MOInfoSpace
    FCIIntegrals(ForteIntegrals* ints, std::shared_ptr<MOSpaceInfo> mospace_info);

    // ==> Class Interface <==

    /// Return the frozen core energy (contribution from FROZEN_DOCC)
    double frozen_core_energy() const {return frozen_core_energy_;}
    /// Return the scalar_energy energy (contribution from RESTRICTED_DOCC)
    double scalar_energy() const {return scalar_energy_;}

    /// Return the alpha effective one-electron integral
    double oei_a(size_t p,size_t q) const {return oei_a_[p * ncmo + q];}
    /// Return the beta effective one-electron integral
    double oei_b(size_t p,size_t q) const {return oei_b_[p * ncmo + q];}

    /// Return the alpha-alpha antisymmetrized two-electron integral <pq||rs>
    double tei_aa(size_t p,size_t q,size_t r,size_t s) const {return tei_aa_[tei_index(p,q,r,s)];}
    /// Return the alpha-beta two-electron integral <pq|rs>
    double tei_ab(size_t p,size_t q,size_t r,size_t s) const {return tei_ab_[tei_index(p,q,r,s)];}
    /// Return the beta-beta antisymmetrized two-electron integral <pq||rs>
    double tei_bb(size_t p,size_t q,size_t r,size_t s) const {return tei_bb_[tei_index(p,q,r,s)];}

    /// Return the alpha-alpha antisymmetrized two-electron integral <pq||pq>
    double diag_tei_aa(size_t p,size_t q) const {return diag_tei_aa_[p * ncmo + q];}
    /// Return the alpha-beta two-electron integral <pq|rs>
    double diag_tei_ab(size_t p,size_t q) const {return diag_tei_ab_[p * ncmo + q];}
    /// Return the beta-beta antisymmetrized two-electron integral <pq||rs>
    double diag_tei_bb(size_t p,size_t q) const {return diag_tei_bb_[p * ncmo + q];}
    IntegralType get_integral_type(){return integral_type_;}

private:

    // ==> Class Private Data <==

    size_t ncmo;
    /// The integral type
    IntegralType integral_type_;
    /// The frozen core energy
    double frozen_core_energy_;
    /// The scalar contribution to the energy
    double scalar_energy_;
    /// The alpha one-electron integrals
    std::vector<double> oei_a_;
    /// The beta one-electron integrals
    std::vector<double> oei_b_;
    /// The alpha-alpha antisymmetrized two-electron integrals in physicist notation
    std::vector<double> tei_aa_;
    /// The alpha-beta antisymmetrized two-electron integrals in physicist notation
    std::vector<double> tei_ab_;
    /// The beta-beta antisymmetrized two-electron integrals in physicist notation
    std::vector<double> tei_bb_;
    /// The diagonal alpha-alpha antisymmetrized two-electron integrals in physicist notation
    std::vector<double> diag_tei_aa_;
    /// The diagonal alpha-beta antisymmetrized two-electron integrals in physicist notation
    std::vector<double> diag_tei_ab_;
    /// The diagonal beta-beta antisymmetrized two-electron integrals in physicist notation
    std::vector<double> diag_tei_bb_;

    // ==> Class Private Functions <==

    inline size_t tei_index(size_t p, size_t q, size_t r, size_t s) const {return ncmo * ncmo * ncmo * p + ncmo * ncmo * q + ncmo * r + s;}
};


class FCIWfn{
public:
    FCIWfn(std::shared_ptr<StringLists> lists, size_t symmetry);
    ~FCIWfn();
    
//    // Simple operation
    void print();
    void zero();
    /// The size of the CI basis
    size_t size() const {return ndet_;}

    /// Copy the wave function object
    void copy(FCIWfn& wfn);
    /// Copy the coefficient from a Vector object
    void copy(SharedVector vec);
    /// Copy the wave function object
    void copy_to(SharedVector vec);

    /// Form the diagonal part of the Hamiltonian
    void form_H_diagonal(std::shared_ptr<FCIIntegrals> fci_ints);

//    double approximate_spin(double )

//    /// Initial guess
//    void initial_guess(FCIWfn& diag, size_t num_dets = 100);

////    void set_to(Determinant& det);
    void set(std::vector<std::tuple<size_t,size_t,size_t,double>>& sparse_vec);
//    double get(int n);
//    void plus_equal(double factor,FCIWfn& wfn);
//    void scale(double factor);
    double norm(double power = 2.0);
////    void normalize_wrt(Determinant& det);
    void normalize();
    double dot(FCIWfn& wfn);


    std::vector<double>& opdm_a() {return opdm_a_;}
    std::vector<double>& opdm_b() {return opdm_b_;}
    std::vector<double>& tpdm_aa() {return tpdm_aa_;}
    std::vector<double>& tpdm_ab() {return tpdm_ab_;}
    std::vector<double>& tpdm_bb() {return tpdm_bb_;}
    std::vector<double>& tpdm_aaa() {return tpdm_aaa_;}
    std::vector<double>& tpdm_aab() {return tpdm_aab_;}
    std::vector<double>& tpdm_abb() {return tpdm_abb_;}
    std::vector<double>& tpdm_bbb() {return tpdm_bbb_;}


//    void randomize();
////    double get_coefficient(Determinant& det);
//    double norm2();
//    double min_element();
//    double max_element();
//    std::vector<int> get_important(double alpha);
    
    // Operations on the wave function
    void Hamiltonian(FCIWfn& result, std::shared_ptr<FCIIntegrals> fci_ints, RequiredLists required_lists);
    
    double energy_from_rdms(std::shared_ptr<FCIIntegrals> fci_ints);

    void compute_rdms(int max_order = 2);
    void rdm_test();

    std::vector<std::tuple<double,size_t,size_t,size_t>> get_largest_contributions(size_t num_dets);
//    // FCIWfn update routines
//    void bendazzoli_update(double alpha,double E,FCIWfn& H,FCIWfn& R);
//    void davidson_update(double E,FCIWfn& H,FCIWfn& R);
//    void two_update(double alpha,double E,FCIWfn& H,FCIWfn& R);
    
//    void save(std::string filename = "wfn.dat");
//    void read(std::string filename = "wfn.dat");
        
    // Temporary memory allocation
    static void allocate_temp_space(std::shared_ptr<StringLists> lists_, size_t symmetry);
    static void release_temp_space();
//    void check_temp_space();
private:

    // ==> Class Data <==

    /// The number of irreps
    size_t nirrep_;
    /// The symmetry of this vector
    size_t symmetry_;
    /// The total number of correlated molecular orbitals
    size_t ncmo_;
    /// The number of correlated molecular orbitals per irrep
    Dimension  cmopi_;
    /// The offset array for cmopi_
    std::vector<size_t> cmopi_offset_;
//    /// The mapping between correlated molecular orbitals and all orbitals
//    std::vector<size_t> cmo_to_mo_;
    /// The number of determinants
    size_t ndet_;
    /// The number of determinants per irrep
    std::vector<size_t> detpi_;

    /// The string list
    std::shared_ptr<StringLists> lists_;
    // Graphs
    /// The alpha string graph
    GraphPtr  alfa_graph_;
    /// The beta string graph
    GraphPtr  beta_graph_;
    /// Coefficient matrix stored in block-matrix form
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
    static size_t   sizeC1;
//    static FCIWfn* tmp_wfn1;
//    static FCIWfn* tmp_wfn2;

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
    double determinant_energy(bool*& Ia,bool*& Ib,int n, std::shared_ptr<FCIIntegrals> fci_ints);


    // ==> Class Private Functions <==

    size_t oei_index(size_t p, size_t q) const {return ncmo_ * p + q;}
    size_t tei_index(size_t p, size_t q, size_t r, size_t s) const {return ncmo_ * ncmo_ * ncmo_ * p + ncmo_ * ncmo_ * q + ncmo_ * r + s;}
    size_t six_index(size_t p, size_t q, size_t r, size_t s, size_t t, size_t u) const {
        return (ncmo_ * ncmo_ * ncmo_ * ncmo_ * ncmo_ * p + ncmo_ * ncmo_ * ncmo_ * ncmo_ * q + ncmo_ * ncmo_ * ncmo_ * r + ncmo_ * ncmo_ * s + ncmo_ * t + u);
    }

//    double oei_aa(size_t p, size_t q) const {return fci_ints_->oei_a(ncmo_ * p + q);}
//    double oei_bb(size_t p, size_t q) const {return fci_ints_->oei_b(ncmo_ * p + q);}

//    double tei_aaaa(size_t p, size_t q, size_t r, size_t s) const {return fci_ints_->tei_aa(tei_index(p,q,r,s));}
//    double tei_aabb(size_t p, size_t q, size_t r, size_t s) const {return fci_ints_->tei_ab(tei_index(p,q,r,s));}
//    double tei_bbbb(size_t p, size_t q, size_t r, size_t s) const {return fci_ints_->tei_ab(tei_index(p,q,r,s));}

    void H0(FCIWfn& result, std::shared_ptr<FCIIntegrals> fci_ints);
    void H1(FCIWfn& result, std::shared_ptr<FCIIntegrals> fci_ints, bool alfa);
    void H2_aabb(FCIWfn& result, std::shared_ptr<FCIIntegrals> fci_ints);
    void H2_aaaa2(FCIWfn& result, std::shared_ptr<FCIIntegrals> fci_ints, bool alfa);

    void compute_1rdm(std::vector<double> &rdm, bool alfa);
    void compute_2rdm_aa(std::vector<double>& rdm, bool alfa);
    void compute_2rdm_ab(std::vector<double>& rdm);
    void compute_3rdm_aaa(std::vector<double>& rdm, bool alfa);
    void compute_3rdm_aab(std::vector<double>& rdm);
    void compute_3rdm_abb(std::vector<double>& rdm);
};

}}

#endif // _fci_vector_



////    DetAddress get_det_address(Determinant& det) {
////        int sym = alfa_graph_->sym(det.get_alfa_bits());
////        size_t alfa_string = alfa_graph_->rel_add(det.get_alfa_bits());
////        size_t beta_string = beta_graph_->rel_add(det.get_beta_bits());
////        return DetAddress(sym,alfa_string,beta_string);
////    };
