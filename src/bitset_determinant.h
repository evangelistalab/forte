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

#ifndef _bitset_determinant_h_
#define _bitset_determinant_h_

#include <unordered_map>

#define BOOST_DYNAMIC_BITSET_DONT_USE_FRIENDS
#include <boost/functional/hash.hpp>
#include "boost/dynamic_bitset.hpp"

#include "integrals.h"
#include "fci_vector.h"
#include "excitation_determinant.h"

namespace psi{ namespace forte{

/**
 * A class to store a Slater determinant using Boost's dynamic_bitset.
 *
 * The determinant is represented by a pair of alpha/beta strings
 * that specify the occupation of each molecular orbital
 * (excluding frozen core and virtual orbitals).
 *
 * |Det> = |Ia> x |Ib>
 *
 * The strings are represented using an array of bits, and the
 * following convention is used here:
 * true <-> 1
 * false <-> 0
 */
class BitsetDeterminant{
public:
    using bit_t = boost::dynamic_bitset<>;

	//test integrals
	void test_ints(){
		outfile->Printf("\n FC energy: %1.8f", fci_ints_->frozen_core_energy());
	}

    // Class Constructor and Destructor
    /// Construct an empty determinant
    BitsetDeterminant();

    /// Construct the determinant from an occupation vector that
    /// specifies the alpha and beta strings.  occupation = [Ia,Ib]
    explicit BitsetDeterminant(const std::vector<int>& occupation,bool print_det = false);

    /// Construct the determinant from an occupation vector that
    /// specifies the alpha and beta strings.  occupation = [Ia,Ib]
    explicit BitsetDeterminant(const std::vector<bool>& occupation,bool print_det = false);

    /// Construct an excited determinant of a given reference
    /// Construct the determinant from two occupation vectors that
    /// specifies the alpha and beta strings.  occupation = [Ia,Ib]
    explicit BitsetDeterminant(const std::vector<bool>& occupation_a,const std::vector<bool>& occupation_b,bool print_det = false);

    bool operator<(const BitsetDeterminant& lhs) const{
        if (alfa_bits_ > lhs.alfa_bits_) return false;
        if (alfa_bits_ < lhs.alfa_bits_) return true;
        return beta_bits_ < lhs.beta_bits_;
    }

    bool operator==(const BitsetDeterminant& lhs) const{
        return ((alfa_bits_ == lhs.alfa_bits_) and (beta_bits_ == lhs.beta_bits_));
    }

//    BitsetDeterminant& operator=(const BitsetDeterminant &rhs)
//    {
//       alfa_bits_ = rhs.alfa_bits_;
//       beta_bits_ = rhs.beta_bits_;
//       return *this;
//    }

    void copy(const BitsetDeterminant &rhs)
    {
        nmo_ = rhs.nmo_;
        alfa_bits_ = rhs.alfa_bits_;
        beta_bits_ = rhs.beta_bits_;
    }

    /// Get a pointer to the alpha bits
    const bit_t& alfa_bits() const {return alfa_bits_;}

    /// Get a pointer to the beta bits
    const bit_t& beta_bits() const {return beta_bits_;}
	/// Get the alpha bits
	std::vector<bool> get_alfa_bits_vector_bool(){
		std::vector<bool> result;
		for(int n = 0; n < nmo_;++n){
			result.push_back(alfa_bits_[n]);
		}
		return result;
	}
	// Get the beta bits
	std::vector<bool> get_beta_bits_vector_bool(){
		std::vector<bool> result;
		for( int n = 0; n < nmo_; ++n){
			result.push_back(beta_bits_[n]);	
		}
		return result;
	}
	/// Get the alpha bits
	const std::vector<bool> get_alfa_bits_vector_bool() const {
		std::vector<bool> result;
		for(int n = 0; n < nmo_;++n){
			result.push_back(alfa_bits_[n]);
		}
		return result;
	}
	// Get the beta bits
	const std::vector<bool> get_beta_bits_vector_bool() const {
		std::vector<bool> result;
		for( int n = 0; n < nmo_; ++n){
			result.push_back(beta_bits_[n]);	
		}
		return result;
	}

    /// Return the value of an alpha bit
    bool get_alfa_bit(int n) const {return alfa_bits_[n];}
    /// Return the value of a beta bit
    bool get_beta_bit(int n) const {return beta_bits_[n];}

    /// Set the value of an alpha bit
    void set_alfa_bit(int n, bool value) {alfa_bits_[n] = value;}
    /// Set the value of a beta bit
    void set_beta_bit(int n, bool value) {beta_bits_[n] = value;}

    /// Specify the occupation numbers
    void set_alfa_bits(const bit_t& alfa_bits) {alfa_bits_ = alfa_bits;}
    /// Specify the occupation numbers
    void set_beta_bits(const bit_t& beta_bits) {beta_bits_ = beta_bits;}

    /// Switch the alpha and beta occupations
    void spin_flip();

    /// Return a vector of occupied alpha orbitals
    std::vector<int> get_alfa_occ();
    /// Return a vector of occupied beta orbitals
    std::vector<int> get_beta_occ();
    /// Return a vector of virtual alpha orbitals
    std::vector<int> get_alfa_vir();
    /// Return a vector of virtual beta orbitals
    std::vector<int> get_beta_vir();


    /// Return a vector of occupied alpha orbitals
    std::vector<int> get_alfa_occ() const;
    /// Return a vector of occupied beta orbitals
    std::vector<int> get_beta_occ() const;
    /// Return a vector of virtual alpha orbitals
    std::vector<int> get_alfa_vir() const;
    /// Return a vector of virtual beta orbitals
    std::vector<int> get_beta_vir() const;

    /// Set the value of an alpha bit
    double create_alfa_bit(int n);
    /// Set the value of a beta bit
    double create_beta_bit(int n);
    /// Set the value of an alpha bit
    double destroy_alfa_bit(int n);
    /// Set the value of a beta bit
    double destroy_beta_bit(int n);

    /// Print the Slater determinant
    void print() const;
    /// Save the Slater determinant as a string
    std::string str() const;

    /// Compute the energy of a Slater determinant
    double energy() const;
    /// Compute the matrix element of the Hamiltonian between this determinant and a given one
    double slater_rules(const BitsetDeterminant& rhs) const;
    /// Compute the matrix element of the S^2 operator between this determinant and a given one
    double spin2(const BitsetDeterminant& rhs) const;

    /// Sets the pointer to the integral object
	static void set_ints(std::shared_ptr<FCIIntegrals> ints) {
		fci_ints_ = ints;
	}
	
	static void set_ints(ForteIntegrals* ints, std::shared_ptr<MOSpaceInfo> mo_space_info ){
		fci_ints_ = std::make_shared<FCIIntegrals>(ints,mo_space_info);
	}	

private:
    // Data
    /// Number of non-frozen molecular orbitals
    size_t nmo_;
public:
    /// The occupation vector for the alpha electrons (does not include the frozen orbitals)
    bit_t alfa_bits_;
    /// The occupation vector for the beta electrons (does not include the frozen orbitals)
    bit_t beta_bits_;

    // Static data
    /// A pointer to the integral object
    static std::shared_ptr<FCIIntegrals> fci_ints_;
    static double SlaterSign(const bit_t& I,int n);
};

typedef boost::shared_ptr<BitsetDeterminant> SharedBitsetDeterminant;

}} // End Namespaces

#endif // _bitset_determinant_h_
