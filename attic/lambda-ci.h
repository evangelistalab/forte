/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2017 by its authors (see COPYING, COPYING.LESSER, AUTHORS).
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

#ifndef _lambda_ci_h_
#define _lambda_ci_h_

#include <fstream>

#include <liboptions/liboptions.h>
#include <libmints/vector.h>
#include <libmints/matrix.h>

#include "integrals.h"
#include "string_determinant.h"
#include "dynamic_bitset_determinant.h"

namespace psi{ namespace forte{

typedef std::vector<std::pair<double,std::vector<bool> > > half_string_list;
// Store the information for a list of strings in the format:
// [(E denominator,E determinant,string)]
typedef boost::tuple<double,double,std::vector<bool> > string_info;
typedef std::vector<string_info> string_list;
typedef std::vector<string_list> string_list_symm;

// Used to store the information of a determinant in the format:
// (det energy,exc_class_a,alpha string,exc_class_b,beta string)
typedef boost::tuple<double,int,int,int,int> det_info;

class LambdaCI
{
public:
    LambdaCI(Options &options, ForteIntegrals* ints);
    ~LambdaCI();

    int ncmo() {return ncmo_;}
    std::vector<double> Da() {return Da_;}
    std::vector<double> Db() {return Db_;}
private:
    /// A reference to the options object
    Options& options_;
    /// The molecular integrals required by Explorer
    ForteIntegrals* ints_;
    /// The number of irriducible representations
    int nirrep_;
    /// The wave function symmetry
    int wavefunction_symmetry_;
    /// The number of correlated molecular orbitals
    int ncmo_;
    /// The number of correlated alpha electrons
    int nalpha_;
    /// The number of correlated beta electrons
    int nbeta_;
//    /// The number of active alpha electrons
//    int nael_;
//    /// The number of active beta electrons
//    int nbel_;
//    /// The number of unoccupied active alpha MOs
//    int nauocc_;
//    /// The number of unoccupied active alpha MOs
//    int nbuocc_;
    /// The number of molecular orbitals per irrep
    Dimension ncmopi_;
    /// The number of frozen core MOs per irrep.
    Dimension frzcpi_;
    /// The number of frozen unoccupied MOs per irrep.
    Dimension frzvpi_;
    /// The number of restricted doubly occupied MOs per irrep.
    Dimension rdoccpi_;
    /// The number of restricted unoccupied MOs per irrep.
    Dimension ruoccpi_;
    /// The number of active MOs per irrep.
    Dimension actvpi_;
    /// The alpha occupation of the reference determinant
    Dimension nalphapi_ref_;
    /// The beta occupation of the reference determinant
    Dimension nbetapi_ref_;
    /// The lowest alpha molecular orbital
    Dimension minalphapi_;
    /// The lowest beta molecular orbital
    Dimension minbetapi_;
    /// The highest alpha molecular orbital
    Dimension maxalphapi_;
    /// The highest beta molecular orbital
    Dimension maxbetapi_;
    /// The maximum excitation level for the alpha electrons
    int maxnaex_;
    /// The maximum excitation level for the beta electrons
    int maxnbex_;
    /// The minimum excitation level
    int minnex_;
    /// The maximum excitation level
    int maxnex_;
    /// The symmetry of each orbital in Pitzer ordering
    std::vector<int> mo_symmetry_;
    /// The symmetry of each orbital in the qt ordering
    std::vector<int> mo_symmetry_qt_;
    /// The energy threshold (in Hartree) for the selection of the determinants
    double determinant_threshold_;
    /// The energy threshold (in Hartree) for prescreening the determinants
    double denominator_threshold_;
    /// The energy threshold (in Hartree) for the model space
    double space_m_threshold_;
    /// The energy threshold (in Hartree) for the intermediate space
    double space_i_threshold_;
    /// The threshold for screening matrix elements using the density matrix
    double t2_threshold_;
    /// Type of screening: mp denominators or excited determinants?
    bool mp_screening_;
    /// The energy of the reference determinant (includes the nuclear repulsion term)
    double ref_energy_;
    /// The minimum determinant energy (includes the nuclear repulsion term)
    double min_energy_;
    /// The maximum determinant energy (includes the nuclear repulsion term)
    double max_energy_;
    /// The alpha orbital energies
    SharedVector ref_eps_a_;
    /// The beta orbital energies
    SharedVector ref_eps_b_;
    /// The alpha orbital energies in qt order
    std::vector<double> epsilon_a_qt_;
    /// The beta orbital energies in qt order
    std::vector<double> epsilon_b_qt_;
    /// Mapping vector from QT (energy) to Pitzer (symmetry blocked)
    std::vector<int> qt_to_pitzer_;
    /// A vector that contains all the restricted doubly occupied MOs
    std::vector<int> rdocc;
    /// A vector that contains all the restricted unoccupied MOs
    std::vector<int> ruocc;
    /// The nuclear repulsion energy
    double nuclear_repulsion_energy_;
    /// The reference determinant
    StringDeterminant reference_determinant_;
    /// The determinant with minimum energy
    StringDeterminant min_energy_determinant_;
    /// The PT2 energy correction
    double pt2_energy_correction_;
    /// The PT2 energy correction
    std::vector<double> multistate_pt2_energy_correction_;
    /// The alpha occupation numbers
    std::vector<double> Da_;
    /// The beta occupation numbers
    std::vector<double> Db_;

    // // The strings are in QT format and are stored using the following structure:
    /// A list of alpha strings in QT format stored as [<string irrep>][<string index>](<string energy>,<string bits>)
    string_list_symm vec_astr_symm_;
    /// A list of alpha strings in QT format stored as [<string irrep>][<string index>](<string energy>,<string bits>)
    string_list_symm vec_bstr_symm_;
    /// A list of determinants in the format: [<determinant index>](determinant energy,alpha string exc class,alpha string index,beta string exc class,beta string index)
    std::vector<det_info> determinants_;

    /// An heuristic search for determinants with energy less than a given threshold (determinant_threshold_)
    void startup(Options& options);
    void read_info(Options& options);
    /// Do various stuff with the MOs
    void screen_mos();
    /// Find the determinants with energy below a given threshold
    void explore(Options &options);
    void explore_original(std::shared_ptr<ForteOptions> options);
    void explore_singles(std::shared_ptr<ForteOptions> options);
    void explore_singles_sequential(std::shared_ptr<ForteOptions> options);
    /// Diagonalize the Hamiltonian in the P space (model + intermediate space)
    void diagonalize_p_space(std::shared_ptr<ForteOptions> options);
    /// Diagonalize the Hamiltonian in the P space (model + intermediate space)
    void diagonalize_p_space_direct(std::shared_ptr<ForteOptions> options);
    /// Diagonalize the Hamiltonian in the P space including the Lowdin contribution to the energy
    void diagonalize_p_space_lowdin(std::shared_ptr<ForteOptions> options);
    /// Diagonalize the Hamiltonian in the main space and include only contributions relevant to each state
    void diagonalize_selected_space(std::shared_ptr<ForteOptions> options);
    /// Diagonalize the Hamiltonian using a renormalization procedure that selects determinants
    void diagonalize_renormalized_space(std::shared_ptr<ForteOptions> options);
    /// Diagonalize the Hamiltonian using a renormalization procedure that selects determinants and keeps only a fixed amount
    void diagonalize_renormalized_fixed_space(std::shared_ptr<ForteOptions> options);
    /// Print the results of a computation
    void print_results(SharedMatrix evecs,SharedVector evals,int nroots);
    void print_results_lambda_sd_ci(std::vector<StringDeterminant>& determinants,
                                    SharedMatrix evecs,
                                    SharedVector evals,
                                    int nroots);

    /// Build the Hamiltonian matrix
    SharedMatrix build_hamiltonian(Options &options);
    /// Build the Hamiltonian matrix in parallel
    SharedMatrix build_hamiltonian_parallel(Options& options);
    /// Build the Hamiltonian matrix in the model space in parallel
    SharedMatrix build_model_space_hamiltonian(Options& options);
    /// Build an Hamiltonian with determinants selected using a threshold
    SharedMatrix build_select_hamiltonian_roth(Options& options, SharedVector evals, SharedMatrix evecs);
    /// Build an Hamiltonian with determinants selected using a threshold and storing only the non-zero elements
    std::vector<std::vector<std::pair<int,double> > > build_hamiltonian_direct(Options& options);
    /// Smooth the Hamiltonian matrix in the intermediate space
    void smooth_hamiltonian(SharedMatrix H);
    /// Select the elements in the intermediate space according to the T2 coupling
    void select_important_hamiltonian(SharedMatrix H);
    /// Fold in the external configurations into the Hamiltonian
    void lowdin_hamiltonian(SharedMatrix H,double E);


    /// Diagonalize the a matrix using the Davidson-Liu method
    void davidson_liu(SharedMatrix H,SharedVector Eigenvalues,SharedMatrix Eigenvectors,int nroots);
    /// Diagonalize the a sparse matrix using the Davidson-Liu method
    bool davidson_liu_sparse(std::vector<std::vector<std::pair<int,double> > > H_sparse,SharedVector Eigenvalues,SharedMatrix Eigenvectors,int nroots);
    void examine_all(Options& options);
    /// Compute perturbative corrections to the energy
    void evaluate_perturbative_corrections(SharedVector evals,SharedMatrix evecs);

    // Lambda+SD-CI
    void lambda_mrcisd(std::shared_ptr<ForteOptions> options);
    // Lambda+S-CI
    void lambda_mrcis(std::shared_ptr<ForteOptions> options);
    /// A renormalized MRCISD
    void renormalized_mrcisd(std::shared_ptr<ForteOptions> options);
    void renormalized_mrcisd_simple(std::shared_ptr<ForteOptions> options);

    /// Iterative and adaptive MRCISD
    void iterative_adaptive_mrcisd(std::shared_ptr<ForteOptions> options);
    /// Iterative and adaptive MRCISD
    void iterative_adaptive_mrcisd_bitset(std::shared_ptr<ForteOptions> options);

    /// Factorized CI multireference wave functions
    void factorized_ci(std::shared_ptr<ForteOptions> options);


    // Functions for generating combination
    half_string_list compute_half_strings_screened(bool is_occ,int n,int k,std::vector<double>& weights,std::string label);
    string_list_symm compute_strings_screened(std::vector<double>& epsilon,int nocc,int nvir,int maxnex,bool alpha);
//    string_list_symm compute_strings_mp_screened(std::vector<double>& epsilon,int nocc,int maxnex);
//    string_list_symm compute_strings_energy_screened(std::vector<double>& epsilon,int nocc,int maxnex,bool alpha);
    bool next_bound_lex_combination(double max_sum,const std::vector<double>& a,bool* begin,bool* end);

    // Ancillary functions
    /// Computes the symmetry of a string with orbitals arranged in QT order
    int string_symmetry_qt(bool* I);
    /// Computes the symmetry of a string
    int string_symmetry(bool* I);
    /// Makes a bitmask containing ones all at the beginning or the end (used to compute permutations)
    void make_bitmask(bool*& vec,int n,int num1s,bool ones_first);
    double compute_denominator(bool is_occ,bool *begin, bool *end,std::vector<double>& epsilon);
    double compute_denominator2(bool is_occ, bool *begin, bool *end, std::vector<double>& epsilon);
    void write_determinant_energy(std::ofstream& os,bool* Ia, bool* Ib,double det_energy,double a_den_energy,double b_den_energy,int naex,int nbex);
    /// Maps an excitation level and irrep to an integer that represents the excitation class
    int excitation_class(int nex, int h);
};


bool not_bool(bool b);
unsigned long long choose(unsigned long long n, unsigned long long k);

}} // End Namespaces

#endif // _lambda_ci_h_
