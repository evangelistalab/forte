/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2017 by its authors (see LICENSE, AUTHORS).
 *
 * The copyrights for code used from other parties are included in
 * the corresponding files.
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see http://www.gnu.org/licenses/.
 *
 * @END LICENSE
 */

#ifndef _excitation_determinant_h_
#define _excitation_determinant_h_

#include <algorithm>
#include "integrals.h"

#define MAXEX 8

namespace psi{ namespace forte{

/**
 * A class to store a Slater determinant in the excitation representation.
 *
 * The determinant is represented by a pair of alpha/beta excitation strings
 * that specify the occupation with respect to a reference determinant.
 *
 * |Det> = Aa Ab |Ref> (Aa and Ab are strings of excitation operators)
 *
 */
class ExcitationDeterminant{
public:
    // Class Constructor and Destructor
    /// Construct an empty determinant
    ExcitationDeterminant();
    /// Copy constructor
    ExcitationDeterminant(const ExcitationDeterminant& det);
    /// Assignment operator
    ExcitationDeterminant& operator=(const ExcitationDeterminant& rhs);
    /// Destructor
    ~ExcitationDeterminant();

    /// Print the determinant
    void print();

    /// Add an alpha excitation
    void add_alpha_ex(short int i,short int a)
    {
        std::vector<short int>::iterator it = std::lower_bound(alpha_ops_.begin(), alpha_ops_.end(), i,std::greater<short int>()); // find proper position in descending order
        alpha_ops_.insert( it, i ); // insert before iterator it
        it = std::lower_bound(alpha_ops_.begin(), alpha_ops_.end(), a,std::greater<short int>()); // find proper position in descending order
        alpha_ops_.insert( it, a ); // insert before iterator it
        naex_ += 1;
    }
    /// Add a beta excitation
    void add_beta_ex(short int i,short int a)
    {
        std::vector<short int>::iterator it = std::lower_bound(beta_ops_.begin(), beta_ops_.end(), i,std::greater<short int>()); // find proper position in descending order
        beta_ops_.insert( it, i ); // insert before iterator it
        it = std::lower_bound(beta_ops_.begin(), beta_ops_.end(), a,std::greater<short int>()); // find proper position in descending order
        beta_ops_.insert( it, a ); // insert before iterator it
        nbex_ += 1;
    }

    std::vector<short int>& alpha_ops() {return alpha_ops_;}
    std::vector<short int>& beta_ops() {return beta_ops_;}

    /// Convert an excitation operator in Qt order to Pitzer order
    void to_pitzer(const std::vector<int>& qt_to_pitzer);

    /// Return the alpha excitation level
    int naex() {return naex_;}
    /// Return the beta excitation level
    int nbex() {return nbex_;}

    int aann(int n) const {return alpha_ops_[2 * n];}
    int acre(int n) const {return alpha_ops_[2 * n + 1];}
    int bann(int n) const {return beta_ops_[2 * n];}
    int bcre(int n) const {return beta_ops_[2 * n + 1];}
private:
    // Data
    /// The alpha excitation level
    int naex_;
    /// The beta excitation level
    int nbex_;
    /// The excitation operator stored in a compact form
    /// [0 ... MAXEX-1 | MAXEX ... 2 MAXEX-1 | 2 MAXEX ... 3 MAXEX-1 | 3 MAXEX ... 4 MAXEX - 1]
    ///   a destructor       a creator              b destructor             b creator
    std::vector<short int> alpha_ops_;
    std::vector<short int> beta_ops_;
    friend class StringDeterminant;
};

}} // End Namespaces

#endif // _excitation_determinant_h_



///**
// * A class to store a Slater determinant in the excitation representation.
// *
// * The determinant is represented by a pair of alpha/beta excitation strings
// * that specify the occupation with respect to a reference determinant.
// *
// * |Det> = Aa Ab |Ref> (Aa and Ab are strings of excitation operators)
// *
// */
//class ExcitationDeterminantSet{
//public:
//    // Class Constructor and Destructor
//    /// Construct an empty determinant
//    ExcitationDeterminantSet();
//    /// Copy constructor
//    ExcitationDeterminantSet(const ExcitationDeterminant& det);
//    /// Assignment operator
//    ExcitationDeterminantSet& operator=(const ExcitationDeterminantSet& rhs);
//    /// Destructor
//    ~ExcitationDeterminantSet();

//    /// Print the determinant
//    void print();

//    /// Add an alpha excitation
//    void add_alpha_ex(int i,int a) {alpha_ops_.push_back(i); alpha_ops_.push_back(a); naex_ += 1;}
//    /// Add an alpha excitation
//    void add_beta_ex(int i,int a) {beta_ops_.push_back(i); beta_ops_.push_back(a); nbex_ += 1;}

//    std::set<short int>& alpha_ann() {return alpha_ops_;}
//    std::vector<short int>& beta_ops() {return beta_ops_;}

//    /// Convert an excitation operator in Qt order to Pitzer order
//    void to_pitzer(const std::vector<int>& qt_to_pitzer);

//    /// Return the alpha excitation level
//    int naex() {return naex_;}
//    /// Return the beta excitation level
//    int nbex() {return nbex_;}

//    int aann(int n) const {return aann_.begin() + n;}
//    int acre(int n) const {return acre_.begin() + n;}
//    int bann(int n) const {return bann_.begin() + n;}
//    int bcre(int n) const {return bcre_.begin() + n;}

////    inline bool operator< (const ExcitationDeterminant& other) const{
////        if (naex_ == other.naex_){
////            if (nbex_ == other.nbex_){
////                if (alpha_ops_ == other.alpha_ops_){
////                    return beta_ops_ < other.beta_ops_;
////                }
////                return alpha_ops_ < other.alpha_ops_;
////            }
////            return nbex_ < other.nbex_;
////        }
////        return naex_ < other.naex_;
////    }
//private:
//    // Data
//    /// The alpha excitation level
//    int naex_;
//    /// The beta excitation level
//    int nbex_;
//    /// The excitation operator stored in a compact form
//    /// [0 ... MAXEX-1 | MAXEX ... 2 MAXEX-1 | 2 MAXEX ... 3 MAXEX-1 | 3 MAXEX ... 4 MAXEX - 1]
//    ///   a destructor       a creator              b destructor             b creator
//    std::set<short int> alpha_ann_;
//    std::set<short int> alpha_cre_;
//    std::set<short int> beta_ann_;
//    std::set<short int> beta_cre_;
//    friend class StringDeterminant;
//};
