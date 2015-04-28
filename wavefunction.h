#ifndef _capriccio_wavefunction_h_
#define _capriccio_wavefunction_h_

#include <vector>
#include "boost/shared_ptr.hpp"

#include "libmints/matrix.h"
#include "integrals.h"
#include "string_lists.h"

#define CAPRICCIO_USE_DAXPY 1
#define CAPRICCIO_USE_UNROLL 0

namespace psi{ namespace libadaptive{

class FCIWfn{
public:
    FCIWfn(boost::shared_ptr<StringLists> lists, ExplorerIntegrals* ints,size_t symmetry);
    ~FCIWfn();
    
//    // Simple operation
    void print();
    void zero();


    /// Form the diagonal part of the Hamiltonian
    void form_H_diagonal();

    /// Initial guess
    void initial_guess(FCIWfn& diag, size_t num_dets = 100);

////    void set_to(Determinant& det);
//    void set_to(FCIWfn& wfn);
//    void set_to(int n);
//    double get(int n);
//    void plus_equal(double factor,FCIWfn& wfn);
//    void scale(double factor);
    double norm(double power = 2.0);
////    void normalize_wrt(Determinant& det);
    void normalize();
    double dot(FCIWfn& wfn);
//    void randomize();
////    double get_coefficient(Determinant& det);
//    double norm2();
//    double min_element();
//    double max_element();
//    std::vector<int> get_important(double alpha);
    
    // Operations on the wave function
    void Hamiltonian(FCIWfn& result,RequiredLists required_lists);
    
//    // FCIWfn update routines
//    void bendazzoli_update(double alpha,double E,FCIWfn& H,FCIWfn& R);
//    void davidson_update(double E,FCIWfn& H,FCIWfn& R);
//    void two_update(double alpha,double E,FCIWfn& H,FCIWfn& R);
    
//    void save(std::string filename = "wfn.dat");
//    void read(std::string filename = "wfn.dat");
        
    // Temporary memory allocation
    static void allocate_temp_space(boost::shared_ptr<StringLists> lists_, ExplorerIntegrals* ints_, size_t symmetry);
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
    /// The mapping between correlated molecular orbitals and all orbitals
    std::vector<size_t> cmo_to_mo_;
    /// The number of determinants per irrep
    std::vector<size_t> detpi_;
    
    boost::shared_ptr<StringLists> lists_;
    ExplorerIntegrals* ints_;
    
    // Graphs
    /// The alpha string graph
    GraphPtr  alfa_graph_;
    /// The beta string graph
    GraphPtr  beta_graph_;
    
    /// Coefficient matrix stored in block-matrix form
    std::vector<SharedMatrix> C_;
    

    // ==> Class Static Data <==

    static SharedMatrix C1;
    static SharedMatrix Y1;
    static size_t   sizeC1;
    static FCIWfn* tmp_wfn1;
    static FCIWfn* tmp_wfn2;

    // Timers
    static double hdiag_timer;
    static double h1_aa_timer;
    static double h1_bb_timer;
    static double h2_aaaa_timer;
    static double h2_aabb_timer;
    static double h2_bbbb_timer;

    // Integrals
    static bool integrals_are_set_;
    static std::vector<double> oei_a_;
    static std::vector<double> oei_b_;
    /// The alpha-alpha antisymmetrized two-electron integrals in physicist notation
    static std::vector<double> tei_aa_;
    /// The alpha-beta antisymmetrized two-electron integrals in physicist notation
    static std::vector<double> tei_ab_;
    /// The beta-beta antisymmetrized two-electron integrals in physicist notation
    static std::vector<double> tei_bb_;

    // ==> Class Public Functions <==

    void startup();
    void cleanup();

    /// Compute the energy of a determinant
    double determinant_energy(bool*& Ia,bool*& Ib,int n);


    // ==> Class Private Functions <==

    size_t tei_index(size_t p, size_t q, size_t r, size_t s) const {return ncmo_ * ncmo_ * ncmo_ * p + ncmo_ * ncmo_ * q + ncmo_ * r + s;}

    double oei_aa(size_t p, size_t q) const {return oei_a_[ncmo_ * p + q];}
    double oei_bb(size_t p, size_t q) const {return oei_b_[ncmo_ * p + q];}

    double tei_aaaa(size_t p, size_t q, size_t r, size_t s) const {return tei_aa_[tei_index(p,q,r,s)];}
    double tei_aabb(size_t p, size_t q, size_t r, size_t s) const {return tei_ab_[tei_index(p,q,r,s)];}
    double tei_bbbb(size_t p, size_t q, size_t r, size_t s) const {return tei_bb_[tei_index(p,q,r,s)];}

    void H0(FCIWfn& result);
    void H1(FCIWfn& result,bool alfa);
    void H2_aabb(FCIWfn& result);
    void H2_aaaa2(FCIWfn& result, bool alfa);
    //    void H2_aaaa(FCIWfn& result, bool alfa);
    //    void opdm(double** opdm,bool alfa);

//    void zero_block(int h);
//    void transpose_block(int h);

//    double string_energy(bool alfa,bool*& I,int n);
//    void   make_string_h0(bool alfa, GraphPtr graph,std::vector<double>& h0);

//    // Functions to access integrals
//    double oei_aa(int p,int q) {
//        return ints->oei_a(cmos_to_mos[p],cmos_to_mos[q]);}

//    double oei_bb(int p,int q) {
//        return ints->oei_b(cmos_to_mos[p],cmos_to_mos[q]);}

//    double h_aa(int p,int q) {
//        return ints->oei_a(cmos_to_mos[p],cmos_to_mos[q]);}

//    double h_bb(int p,int q) {
//        return ints->oei_b(cmos_to_mos[p],cmos_to_mos[q]);}

//    double tei_aaaa(int p,int q,int r,int s) {
//        return ints->aptei_aa(cmos_to_mos[p],cmos_to_mos[q],
//                                  cmos_to_mos[r],cmos_to_mos[s]);}

//    double tei_bbbb(int p,int q,int r,int s) {
//        return ints->aptei_bb(cmos_to_mos[p],cmos_to_mos[q],
//                                  cmos_to_mos[r],cmos_to_mos[s]);}

//    double tei_aabb(int p,int q,int r,int s) {
//        return ints->aptei_ab(cmos_to_mos[p],cmos_to_mos[q],
//                                  cmos_to_mos[r],cmos_to_mos[s]);}

////    DetAddress get_det_address(Determinant& det) {
////        int sym = alfa_graph_->sym(det.get_alfa_bits());
////        size_t alfa_string = alfa_graph_->rel_add(det.get_alfa_bits());
////        size_t beta_string = beta_graph_->rel_add(det.get_beta_bits());
////        return DetAddress(sym,alfa_string,beta_string);
////    };


};

}}

#endif // _capriccio_wavefunction_h_        
