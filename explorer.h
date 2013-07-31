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

#ifndef _explorer_h_
#define _explorer_h_

#include <fstream>

#include <liboptions/liboptions.h>
#include <libmints/vector.h>
#include <libmints/matrix.h>

#include "integrals.h"
#include "string_determinant.h"

namespace psi{ namespace libadaptive{

typedef std::vector<std::pair<double,std::vector<bool> > > half_string_list;
// Store the information for a list of strings in the format:
// [(E denominator,E determinant,string)]
typedef boost::tuple<double,double,std::vector<bool> > string_info;
typedef std::vector<string_info> string_list;
typedef std::vector<string_list> string_list_symm;

// Used to store the information of a determinant in the format:
// (det energy,exc_class_a,alpha string,exc_class_b,beta string)
typedef boost::tuple<double,int,int,int,int> det_info;

class Explorer
{
public:
    Explorer(Options &options, ExplorerIntegrals* ints);
    ~Explorer();
private:
    /// The number of irriducible representations
    int nirrep_;
    /// The wave function symmetry
    int wavefunction_symmetry_;
    /// The number of molecular orbitals
    int nmo_;
    /// The number of alpha electrons
    int nalpha_;
    /// The number of beta electrons
    int nbeta_;
    /// The number of molecular orbitals per irrep
    Dimension nmopi_;
    /// The number of alpha electrons per irrep
    Dimension nalphapi_;
    /// The number of beta electrons per irrep
    Dimension nbetapi_;
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
    /// The frozen core orbitals
    Dimension frzcpi_;
    /// The frozen virtual orbitals
    Dimension frzvpi_;
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
    /// Type of screening: mp denominators or excited determinants?
    bool mp_screening_;
    /// The energy of the reference determinant (includes the nuclear repulsion term)
    double ref_energy_;
    /// The minimum determinant energy (includes the nuclear repulsion term)
    double min_energy_;
    /// The maximum determinant energy (includes the nuclear repulsion term)
    double max_energy_;
    /// The alpha orbital energies
    SharedVector epsilon_a_;
    /// The beta orbital energies
    SharedVector epsilon_b_;
    /// The alpha orbital energies in qt order
    std::vector<double> epsilon_a_qt_;
    /// The beta orbital energies in qt order
    std::vector<double> epsilon_b_qt_;
    /// Mapping vector from QT to Pitzer
    std::vector<int> qt_to_pitzer_;
    /// A vector that contains all the frozen core
    std::vector<int> frzc_;
    /// A vector that contains all the frozen virtual
    std::vector<int> frzv_;
    /// The nuclear repulsion energy
    double nuclear_repulsion_energy_;
    /// The molecular integrals required by Explorer
    ExplorerIntegrals* ints_;
    /// The reference determinant
    StringDeterminant reference_determinant_;
    /// The determinant with minimum energy
    StringDeterminant min_energy_determinant_;


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
    /// Diagonalize the Hamiltonian built from a subset of the determinants
    void diagonalize(Options &options);
    /// Build the Hamiltonian matrix
    SharedMatrix build_hamiltonian(int ndets);
    void examine_all(Options& options);

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

#endif // _explorer_h_
