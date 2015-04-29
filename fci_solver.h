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

#ifndef _fci_solver_h_
#define _fci_solver_h_

#include <libmints/wavefunction.h>
#include <liboptions/liboptions.h>
#include <physconst.h>

#include "integrals.h"
#include "string_lists.h"

namespace psi{ namespace libadaptive{

/**
 * @brief The FCI class
 * This class implements FCI
 */
class FCI : public Wavefunction
{
public:
    // ==> Class Constructor and Destructor <==

    /**
     * Constructor
     * @param wfn The main wavefunction object
     * @param options The main options object
     * @param ints A pointer to an allocated integral object
     */
    FCI(boost::shared_ptr<Wavefunction> wfn, Options &options, ExplorerIntegrals* ints);

    ~FCI() {}

    // ==> Class Interface <==

    /// Compute the energy
    virtual double compute_energy();

private:

    // ==> Class data <==

    // * Calculation data
    /// A reference to the options object
    Options& options_;   
    /// The molecular integrals
    ExplorerIntegrals* ints_;
//    /// The maximum number of threads
//    int num_threads_;
//    /// The wave function symmetry
//    int wavefunction_symmetry_;
//    /// The symmetry of each orbital in Pitzer ordering
//    std::vector<int> mo_symmetry_;
//    /// The number of correlated molecular orbitals
//    int ncmo_;
//    /// The number of correlated molecular orbitals per irrep
//    Dimension ncmopi_;
//    /// The nuclear repulsion energy
//    double nuclear_repulsion_energy_;

    // ==> Class functions <==

    /// All that happens before we compute the energy
    void startup();
};


/**
 * @brief The FCISolver class
 * This class performs Full CI calculations.
 */
class FCISolver
{
public:
    // ==> Class Constructor and Destructor <==

    FCISolver(std::vector<size_t> core_mo,std::vector<size_t> active_mo, size_t na, size_t nb, size_t symmetry, ExplorerIntegrals* ints);

    ~FCISolver() {}

    /// Compute the FCI energy
    double compute_energy();


private:

    // ==> Class Data <==

    // The orbitals frozen at the CI level
    std::vector<size_t> core_mo_;

    // The orbitals treated at the CI level
    std::vector<size_t> active_mo_;

    // A object that stores string information
    boost::shared_ptr<StringLists> lists_;

    /// The molecular integrals
    ExplorerIntegrals* ints_;

////    /// Use a OMP parallel algorithm?
////    bool parallel_;
////    /// Print details?
////    bool print_details_;

////    void cleanup();
////    void zero_block(int h);
////    void transpose_block(int h);

////    double string_energy(bool alfa,bool*& I,int n,boost::shared_ptr<Integrals>& ints);
////    double determinant_energy(bool*& Ia,bool*& Ib,int n,boost::shared_ptr<Integrals>& ints);
////    void   make_string_h0(bool alfa, GraphPtr graph,vector<double>& h0,boost::shared_ptr<Integrals>& ints);

////    // Functions to access integrals
////    double oei_aa(int p,int q) {
////        return ints->get_oei_aa(cmos_to_mos[p],cmos_to_mos[q]);}

////    double oei_bb(int p,int q) {
////        return ints->get_oei_bb(cmos_to_mos[p],cmos_to_mos[q]);}

////    double h_aa(int p,int q) {
////        return ints->get_oei_aa(cmos_to_mos[p],cmos_to_mos[q]);}

////    double h_bb(int p,int q) {
////        return ints->get_oei_bb(cmos_to_mos[p],cmos_to_mos[q]);}

////    double tei_aaaa(int p,int q,int r,int s) {
////        return ints->get_tei_aaaa(cmos_to_mos[p],cmos_to_mos[q],
////                                  cmos_to_mos[r],cmos_to_mos[s]);}

////    double tei_bbbb(int p,int q,int r,int s) {
////        return ints->get_tei_bbbb(cmos_to_mos[p],cmos_to_mos[q],
////                                  cmos_to_mos[r],cmos_to_mos[s]);}

////    double tei_aabb(int p,int q,int r,int s) {
////        return ints->get_tei_aabb(cmos_to_mos[p],cmos_to_mos[q],
////                                  cmos_to_mos[r],cmos_to_mos[s]);}

////    DetAddress get_det_address(Determinant& det) {
////        int sym = alfa_graph_->sym(det.get_alfa_bits());
////        size_t alfa_string = alfa_graph_->rel_add(det.get_alfa_bits());
////        size_t beta_string = beta_graph_->rel_add(det.get_beta_bits());
////        return DetAddress(sym,alfa_string,beta_string);
////    };

////    void H0(FCIWfn& result);
////    void H1(FCIWfn& result,bool alfa);
////    void H2_aabb(FCIWfn& result);
////    void H2_aaaa(FCIWfn& result, bool alfa);
////    void H2_aaaa2(FCIWfn& result, bool alfa);
////    void opdm(double** opdm,bool alfa);

    /// The number of irreps
    size_t nirrep_;
    /// The symmetry of the wave function
    size_t symmetry_;
    /// The number of alpha electrons
    size_t na_;
    /// The number of beta electrons
    size_t nb_;
    /// The number of roots
    size_t nroot_;
////    int ncmos;                     // # of correlated molecular orbitals
////    std::vector<int> cmos;         // # of correlated molecular orbitals per irrep
////    std::vector<int> cmos_offset;  // Offset array for non-frozen molecular orbitals
////    std::vector<int> cmos_to_mos;  // Mapping of the correlated MOs to the all ordering (all is used by the integral class)
////    std::vector<size_t> detpi;     // Determinants per irrep



////    GraphPtr alfa_graph;
////    GraphPtr beta_graph;

////    double*** coefficients;  // Coefficient matrix stored in block-matrix form
///
///

    // ==> Class functions <==

    /// All that happens before we compute the energy
    void startup();

////    static double** C1;
////    static double** Y1;
////    static size_t   sizeC1;
////    static FCIWfn* tmp_wfn1;
////    static FCIWfn* tmp_wfn2;
};

}}

#endif // _fci_solver_h_
