///*
// *@BEGIN LICENSE
// *
// * Libadaptive: an ab initio quantum chemistry software package
// *
// * This program is free software; you can redistribute it and/or modify
// * it under the terms of the GNU General Public License as published by
// * the Free Software Foundation; either version 2 of the License, or
// * (at your option) any later version.
// *
// * This program is distributed in the hope that it will be useful,
// * but WITHOUT ANY WARRANTY; without even the implied warranty of
// * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// * GNU General Public License for more details.
// *
// * You should have received a copy of the GNU General Public License along
// * with this program; if not, write to the Free Software Foundation, Inc.,
// * 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
// *
// *@END LICENSE
// */

//#ifndef _string_determinant_h_
//#define _string_determinant_h_

//#include "integrals.h"
//#include "excitation_determinant.h"

//namespace psi{ namespace libadaptive{

///**
// * A class to store a Slater determinant.
// *
// * The determinant is represented by a pair of alpha/beta strings
// * that specify the occupation of each molecular orbital
// * (including frozen core and virtual orbitals).
// *
// * |Det> = |Ia> x |Ib>
// *
// * The strings are represented using an array of bits, and the
// * following convention is used here:
// * true <-> 1
// * false <-> 0
// */
//class SafeString{
//public:
//    // Class Constructor and Destructor
//    /// Construct an empty determinant
//    SafeString();
//    /// Construct a vacuum determinant given the total number of MOs
////    explicit SafeString(int nmo,bool print_det = false);
//    /// Construct the determinant from an occupation vector that
//    /// specifies the alpha and beta strings.  occupation = [Ia,Ib]
//    explicit SafeString(const std::vector<int>& occupation,bool print_det = false);
//    /// Construct the determinant from an occupation vector that
//    /// specifies the alpha and beta strings.  occupation = [Ia,Ib]
//    explicit SafeString(const std::vector<bool>& occupation,bool print_det = false);
//    /// Construct an excited determinant of a given reference
//    explicit SafeString(const SafeString& ref,const ExcitationDeterminant& ex);
//    /// Copy constructor
//    explicit SafeString(const SafeString& det);
//    /// Assignment operator
//    SafeString& operator=(const SafeString& rhs);
//    /// Destructor
//    ~SafeString();
//    /// Returns the number of non-frozen molecular orbitals
//    int nmo() {return nmo_;}
//    /// Get a pointer to the alpha bits
//    bool* get_alfa_bits() {return alfa_bits_;}
//    /// Get a pointer to the beta bits
//    bool* get_beta_bits() {return beta_bits_;}
//    /// Return the value of an alpha bit
//    bool get_alfa_bit(int n) {return alfa_bits_[n];}
//    /// Get a pointer to the beta bits
//    bool get_beta_bit(int n) {return beta_bits_[n];}
//    /// Return the value of an alpha bit
//    bool get_alfa_bit(int n) const {return alfa_bits_[n];}
//    /// Get a pointer to the beta bits
//    bool get_beta_bit(int n) const {return beta_bits_[n];}
//    /// Return the value of an alpha bit
//    void set_alfa_bit(int n, bool value) {alfa_bits_[n] = value;}
//    /// Get a pointer to the beta bits
//    void set_beta_bit(int n, bool value) {beta_bits_[n] = value;}
//    /// Specify the occupation numbers
//    void set_bits(bool*& alfa_bits,bool*& beta_bits);
//    /// Specify the occupation numbers
//    void set_bits(std::vector<bool>& alfa_bits,std::vector<bool>& beta_bits);
//    /// Print the Slater determinant
//    void print() const;
//    /// Compute the energy of a Slater determinant
//    double energy() const;
//    /// Compute the matrix element of the Hamiltonian between this determinant and a given one
//    double slater_rules(const SafeString& rhs) const;
//    /// Compute the excitation level of a Slater determiant with respect to a given reference
//    int excitation_level(const SafeString& reference);
//    int excitation_level(const bool* Ia,const bool* Ib);
//    /// Sets the pointer to the integral object
//    static void set_ints(ExplorerIntegrals* ints) {ints_ = ints;}
//private:
//    // Functions
//    /// Used to allocate the memory for the arrays
//    void allocate();
//    /// Used to deallocate the memory for the arrays
//    void deallocate();

//    // Data
//    /// Number of non-frozen molecular orbitals
//    int nmo_;
//    std::vector<bool> bits_;

//    // Static data
//    /// A pointer to the integral object
//    static ExplorerIntegrals* ints_;
//};

//}} // End Namespaces

//#endif // _string_determinant_h_
