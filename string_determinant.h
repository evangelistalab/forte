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

#ifndef _string_determinant_h_
#define _string_determinant_h_

#include "integrals.h"

namespace psi{ namespace libadaptive{

/**
 * A class to store a Slater determinant.
 *
 * The determinant is represented by a pair of alpha/beta strings
 * that specify the occupation of each molecular orbital.
 *
 * |Det> = |Ia> x |Ib>
 *
 * Frozen core and virtual orbital are not included in the strings
 * and are assumed to be doubly occupied and empty, respectively.
 * The strings are represented using an array of bits, and the
 * following convention is used here:
 * true <-> 1
 * false <-> 0
 */
class StringDeterminant{
public:
    // Class Constructor and Destructor
    /// Construct an empty determinant
    StringDeterminant();
    /// Construct a vacuum determinant given the total number of MOs
    StringDeterminant(int nmo,bool print_det = false);
    /// Construct the determinant from an occupation vector that
    /// specifies the alpha and beta strings.  occupation = [Ia,Ib]
    StringDeterminant(std::vector<int> occupation,bool print_det = false);
    /// Construct the determinant from an occupation vector that
    /// specifies the alpha and beta strings.  occupation = [Ia,Ib]
    StringDeterminant(std::vector<bool> occupation,bool print_det = false);
    /// Copy constructor
    StringDeterminant(const StringDeterminant& det);
    /// Assignment operator
    StringDeterminant& operator=(const StringDeterminant& rhs);
    /// Destructor
    ~StringDeterminant();
    /// Returns the number of non-frozen molecular orbitals
    int nmo() {return nmo_;}
    /// Get a pointer to the alpha bits
    bool* get_alfa_bits() {return alfa_bits_;}
    /// Get a pointer to the beta bits
    bool* get_beta_bits() {return beta_bits_;}
    /// Specify the occupation numbers
    void set_bits(bool*& alfa_bits,bool*& beta_bits);
    /// Specify the occupation numbers
    void set_bits(std::vector<bool>& alfa_bits,std::vector<bool>& beta_bits);
    /// Print the Slater determinant
    void print();
    /// Compute the energy of a Slater determinant
    double energy();
    /// Compute the energy of a Slater determinant with respect to a given reference
    double excitation_energy(const StringDeterminant& reference);
    /// Compute the energy of a Slater determinant with respect to a given reference
    double excitation_ab_energy(const StringDeterminant& reference);
    /// Compute the excitation level of a Slater determiant with respect to a given reference
    int excitation_level(const StringDeterminant& reference);
    int excitation_level(const bool* Ia,const bool* Ib);
    /// Sets the pointer to the integral object
    static void set_ints(ExplorerIntegrals* ints) {ints_ = ints;}
private:
    // Functions
    /// Used to allocate the memory for the arrays
    void allocate();
    /// Used to deallocate the memory for the arrays
    void deallocate();

    // Data
    /// Number of non-frozen molecular orbitals
    int nmo_;
    /// The "charge" of the determinant if we are using it say in a FCIMC computation
    int charge_;
    /// The occupation vector for the alpha electrons (does not include the frozen orbitals)
    bool* alfa_bits_;
    /// The occupation vector for the beta electrons (does not include the frozen orbitals)
    bool* beta_bits_;

    // Static data
    /// A pointer to the integral object
    static ExplorerIntegrals* ints_;
    static double ahole_[20];
    static double bhole_[20];
    static double apart_[20];
    static double bpart_[20];
//    /// Compute the matrix element of the Hamiltonian between two Slater determinants
//    double SlaterRules(StringDeterminant& PhiA, StringDeterminant& PhiB);
//    /// Compute the one-particle density matrix contribution from a pair of determinants
//    double SlaterOPDM(StringDeterminant& PhiI, StringDeterminant& PhiJ,int p,bool spin_p,int q,bool spin_q);
//    /// Compute the two-particle density matrix contribution from a pair of determinants
//    double SlaterTPDM(StringDeterminant& PhiI, StringDeterminant& PhiJ,int p,bool spin_p,int q,bool spin_q,int r,bool spin_r,int s,bool spin_s);
//    /// Compute the three-particle density matrix contribution from a pair of determinants
//    double Slater3PDM(StringDeterminant& PhiI, StringDeterminant& PhiJ,int p,bool spin_p,int q,bool spin_q,int r,bool spin_r,int s,bool spin_s,int t,bool spin_t,int u,bool spin_u);
};

}} // End Namespaces

#endif // _string_determinant_h_
