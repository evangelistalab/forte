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

#ifndef _string_determinant_h_
#define _string_determinant_h_

#include "integrals.h"

namespace psi{ namespace forte{

/**
 * A class to store a Slater determinant.
 *
 * The determinant is represented by a pair of alpha/beta strings
 * that specify the occupation of each molecular orbital
 * (including frozen core and virtual orbitals).
 *
 * |Det> = |Ia> x |Ib>
 *
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
//    explicit StringDeterminant(int nmo,bool print_det = false);
    /// Construct the determinant from an occupation vector that
    /// specifies the alpha and beta strings.  occupation = [Ia,Ib]
    explicit StringDeterminant(const std::vector<int>& occupation,bool print_det = false);
    /// Construct the determinant from an occupation vector that
    /// specifies the alpha and beta strings.  occupation = [Ia,Ib]
    explicit StringDeterminant(const std::vector<bool>& occupation,bool print_det = false);
    /// Construct an excited determinant of a given reference
    /// Construct the determinant from two occupation vectors that
    /// specifies the alpha and beta strings.  occupation = [Ia,Ib]
    explicit StringDeterminant(const std::vector<bool>& occupation_a,const std::vector<bool>& occupation_b,bool print_det = false);
     /// Copy constructor
    StringDeterminant(const StringDeterminant& det);
    /// Copy constructor
    StringDeterminant(StringDeterminant& det);
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
    /// Get a pointer to the alpha bits
    std::vector<bool> get_alfa_bits_vector_bool() {
        std::vector<bool> result;
        for (int n = 0; n < nmo_; ++n){
            result.push_back(alfa_bits_[n]);
        }
        return result;
    }
    /// Get a pointer to the alpha bits
    std::vector<bool> get_beta_bits_vector_bool() {
        std::vector<bool> result;
        for (int n = 0; n < nmo_; ++n){
            result.push_back(beta_bits_[n]);
        }
        return result;
    }
    /// Get a pointer to the alpha bits
    const std::vector<bool> get_alfa_bits_vector_bool() const {
        std::vector<bool> result;
        for (int n = 0; n < nmo_; ++n){
            result.push_back(alfa_bits_[n]);
        }
        return result;
    }
    /// Get a pointer to the alpha bits
    const std::vector<bool> get_beta_bits_vector_bool() const {
        std::vector<bool> result;
        for (int n = 0; n < nmo_; ++n){
            result.push_back(beta_bits_[n]);
        }
        return result;
    }
    /// Return the value of an alpha bit
    bool get_alfa_bit(int n) {return alfa_bits_[n];}
    /// Get a pointer to the beta bits
    bool get_beta_bit(int n) {return beta_bits_[n];}
    /// Return the value of an alpha bit
    bool get_alfa_bit(int n) const {return alfa_bits_[n];}
    /// Get a pointer to the beta bits
    bool get_beta_bit(int n) const {return beta_bits_[n];}
    /// Return the value of an alpha bit
    void set_alfa_bit(int n, bool value) {alfa_bits_[n] = value;}
    /// Get a pointer to the beta bits
    void set_beta_bit(int n, bool value) {beta_bits_[n] = value;}
    /// Specify the occupation numbers
    void set_bits(bool*& alfa_bits,bool*& beta_bits);
    /// Specify the occupation numbers
    void set_bits(std::vector<bool>& alfa_bits,std::vector<bool>& beta_bits);
    /// Print the Slater determinant
    void print() const;
    /// Compute the energy of a Slater determinant
    double energy() const;
    /// Compute the one-electron contribution to the energy of a Slater determinant
    double one_electron_energy();
    /// Compute the kinetic energy of a Slater determinant
    double kinetic_energy();
    /// Compute the energy of a Slater determinant with respect to a given reference
    double excitation_energy(const StringDeterminant& reference);
//    /// Compute the energy of a Slater determinant with respect to a given reference
    double excitation_ab_energy(const StringDeterminant& reference);

    /// Compute the matrix element of the Hamiltonian between this determinant and a given one
    double slater_rules(const StringDeterminant& rhs) const;

    /// Compute the matrix element of the S^2 operator between this determinant and a given one
    double spin2(const StringDeterminant& rhs) const;

    /// Compute the cotribution of a determinant to the diagonal density matrix (number operator)
    void diag_opdm(std::vector<double>& Da,std::vector<double>& Db,double w);

    /// Compute the excitation level of a Slater determiant with respect to a given reference
    int excitation_level(const StringDeterminant& reference);
    int excitation_level(const bool* Ia,const bool* Ib);

    /// Sets the pointer to the integral object
    static void set_ints(std::shared_ptr<ForteIntegrals>  ints) {ints_ = ints;}
    /// Compute the matrix element of the Hamiltonian between two Slater determinants
    static double SlaterRules(const std::vector<bool>& Ia,const std::vector<bool>& Ib,const std::vector<bool>& Ja,const std::vector<bool>& Jb);
    /// Compute the matrix element of the S^2 operator between two Slater determinants
    static double Spin2(const std::vector<bool>& Ia,const std::vector<bool>& Ib,const std::vector<bool>& Ja,const std::vector<bool>& Jb);
    /// Compute the one-particle density matrix contribution from a pair of determinants
    static void SlaterOPDM(const std::vector<bool>& Ia,const std::vector<bool>& Ib,const std::vector<bool>& Ja,const std::vector<bool>& Jb,SharedMatrix Da,SharedMatrix Db,double w);
    /// Compute the diagonal of the one-particle density matrix contribution from a pair of determinants
    static void SlaterdiagOPDM(const std::vector<bool>& Ia,const std::vector<bool>& Ib,std::vector<double>& Da,std::vector<double>& Db,double w);

//    bool operator<(const StringDeterminant& rhs,const StringDeterminant& lhs){
//        for (int n = 0; n < 2 * rhs.nmo_; ++n){
//            if (rhs.alfa_bits_[n] and not lhs.alfa_bits_[n])
//                return false;
//            if (not rhs.alfa_bits_[n] and lhs.alfa_bits_[n])
//                return true;
//        }
//        return false;
//    }

    bool operator<(const StringDeterminant& lhs) const{
        for (int n = 0; n < 2 * nmo_; ++n){
            if (alfa_bits_[n] and (not lhs.alfa_bits_[n]))
                return false;
            if ((not alfa_bits_[n]) and lhs.alfa_bits_[n])
                return true;
        }
        return false;
    }

    bool operator==(const StringDeterminant& lhs) const{
        for (int n = 0; n < 2 * nmo_; ++n){
            if (alfa_bits_[n] != lhs.alfa_bits_[n])
                return false;
        }
        return true;
    }


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
    static std::shared_ptr<ForteIntegrals>  ints_;
    static double ahole_[20];
    static double bhole_[20];
    static double apart_[20];
    static double bpart_[20];
    /// Compute the matrix element of the Hamiltonian between two Slater determinants
    //double SlaterRules(StringDeterminant& PhiA, StringDeterminant& PhiB);
//    /// Compute the one-particle density matrix contribution from a pair of determinants
//    double SlaterOPDM(StringDeterminant& PhiI, StringDeterminant& PhiJ,int p,bool spin_p,int q,bool spin_q);
//    /// Compute the two-particle density matrix contribution from a pair of determinants
//    double SlaterTPDM(StringDeterminant& PhiI, StringDeterminant& PhiJ,int p,bool spin_p,int q,bool spin_q,int r,bool spin_r,int s,bool spin_s);
//    /// Compute the three-particle density matrix contribution from a pair of determinants
//    double Slater3PDM(StringDeterminant& PhiI, StringDeterminant& PhiJ,int p,bool spin_p,int q,bool spin_q,int r,bool spin_r,int s,bool spin_s,int t,bool spin_t,int u,bool spin_u);
};

}} // End Namespaces

#endif // _string_determinant_h_
