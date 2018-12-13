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

#include <algorithm>
#include <cmath>

#include "psi4/psi4-dec.h"
#include "psi4/libpsi4util/PsiOutStream.h"

#include "helpers/mo_space_info.h"
#include "sq.h"

using namespace psi;

namespace psi {
namespace forte {

// Partitions of the number n
std::vector<std::vector<std::vector<int>>> partitions{
    {{0}},
    {{1}},
    {{1, 1}, {2}},
    {{1, 1, 1}, {2, 1}, {3}},
    {{1, 1, 1, 1}, {2, 1, 1}, {2, 2}, {3, 1}, {4}},
    {{1, 1, 1, 1, 1}, {2, 1, 1, 1}, {2, 2, 1}, {3, 1, 1}, {4, 1}, {5}},
    {{1, 1, 1, 1, 1, 1},
     {2, 1, 1, 1, 1},
     {2, 2, 1, 1},
     {2, 2, 2},
     {3, 1, 1, 1},
     {3, 2, 1},
     {3, 3},
     {4, 1, 1},
     {4, 2},
     {5, 1},
     {6}},
    {{1, 1, 1, 1, 1, 1, 1},
     {2, 1, 1, 1, 1, 1},
     {2, 2, 1, 1, 1},
     {2, 2, 2, 1},
     {3, 1, 1, 1, 1},
     {3, 2, 1, 1},
     {3, 2, 2},
     {3, 3, 1},
     {4, 1, 1, 1},
     {4, 2, 1},
     {4, 3},
     {5, 1, 1},
     {5, 2},
     {6, 1},
     {7}}};

// Partitions of the number 2n into even numbers
std::vector<std::vector<std::vector<int>>> even_partitions{
    {{0}},
    {{2}},
    {{2, 2}, {4}},
    {{2, 2, 2}, {4, 2}, {6}},
    {{2, 2, 2, 2}, {4, 2, 2}, {4, 4}, {6, 2}, {8}},
    {{2, 2, 2, 2, 2}, {4, 2, 2, 2}, {4, 4, 2}, {6, 2, 2}, {8, 2}, {10}},
    {{2, 2, 2, 2, 2, 2},
     {4, 2, 2, 2, 2},
     {4, 4, 2, 2},
     {4, 4, 4},
     {6, 2, 2, 2},
     {6, 4, 2},
     {6, 6},
     {8, 2, 2},
     {8, 4},
     {10, 2},
     {12}}};

// => Helper functions <=

/**
 * @brief permutation_sign
 * @param vec the permutation to test
 * @return a boolean: false = even permutation, true = odd permutation
 */
double permutation_sign(const std::vector<int>& vec) {
    // Quadratic algorithm to determine the sign of a permutation
    // From:
    // http://math.stackexchange.com/questions/65923/how-does-one-compute-the-sign-of-a-permutation
    int n = vec.size();
    int count = 0;
    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            if (vec[i] > vec[j])
                count++;
        }
    }
    return (count % 2 != 0) ? -1.0 : 1.0;
}

// => SqOperator class functions <=

SqOperator::SqOperator() {}

SqOperator::SqOperator(const std::vector<int>& cre, const std::vector<int>& ann)
    : cre_(cre), ann_(ann) {}

std::string SqOperator::str() const {
    std::string s("");
    if (cre_.size() + ann_.size() == 0)
        return s;
    std::vector<std::string> vec_cre, vec_ann;
    for (int c : cre_)
        vec_cre.push_back(std::to_string(c));
    for (int a : ann_)
        vec_ann.push_back(std::to_string(a));

    return "a^{" + to_string(vec_cre) + "}_{" + to_string(vec_ann) + "}";
}

bool SqOperator::operator<(const SqOperator& lhs) const {
    if (cre_ > lhs.cre_)
        return false;
    if (cre_ < lhs.cre_)
        return true;
    return ann_ < lhs.ann_;
}

bool SqOperator::operator==(const SqOperator& lhs) const {
    return ((cre_ == lhs.cre_) and (ann_ == lhs.ann_));
}

size_t SqOperator::hash() {
    // From:
    // http://stackoverflow.com/questions/20511347/a-good-hash-function-for-a-vector
    std::size_t h = 0;
    for (auto& i : cre_) {
        h ^= i + 0x9e3779b9 + (h << 6) + (h >> 2);
    }
    for (auto& i : ann_) {
        h ^= i + 0x9e3779b9 + (h << 6) + (h >> 2);
    }
    return h;
}

double SqOperator::sort(std::vector<int>& vec) {
    if (std::is_sorted(vec.begin(), vec.end())) {
        // test for repeated indices
        return (adjacent_find(vec.begin(), vec.end()) == vec.end()) ? 1.0 : 0.0;
    }
    double sign = permutation_sign(vec);
    std::sort(vec.begin(), vec.end());
    // test for repeated indices
    return (adjacent_find(vec.begin(), vec.end()) == vec.end()) ? sign : 0.0;
}

double SqOperator::sort() {
    double signc = sort(cre_);
    double signa = sort(ann_);
    return signc * signa;
}

void SqOperator::test_sort() {
    outfile->Printf("\nBefore sort: %s", str().c_str());
    double signc = sort(cre_);
    double signa = sort(ann_);
    double sign = signc * signa;
    outfile->Printf("\nAfter sort: %f %f %f %s", sign, signc, signa, str().c_str());
}

// => Operator class function <=

Operator::Operator() {}

void Operator::add(double value, const SqOperator& op) {
    // add only if has nonzero coefficient
    if (std::fabs(value) != 0.0) {
        SqOperator sorted_op = op;
        double sign = sorted_op.sort();
        ops_[sorted_op] += sign * value;
    }
}

std::string Operator::str() const {
    std::vector<std::string> vec_str;
    for (auto& c_op : ops_) {
        vec_str.push_back(std::to_string(c_op.second) + " " + c_op.first.str());
    }
    return to_string(vec_str, "\n ");
}

const op_hash& Operator::ops() { return ops_; }

// => WickTheorem class function <=

WickTheorem::WickTheorem() {}

Operator WickTheorem::evaluate(Operator& lhs, Operator& rhs) {
    Operator res;

    for (auto& opl : lhs.ops()) {
        for (auto& opr : rhs.ops()) {
            //            outfile->Printf("\n  Contracting:");
            //            outfile->Printf("\n  %+f x %+f { %s } { %s
            //            }",opl.second,opr.second,opl.first.str().c_str(),opr.first.str().c_str());
            contract(opl.first, opr.first, res);
        }
    }
    return res;
}

void WickTheorem::contract(const SqOperator& lhs, const SqOperator& rhs, Operator& res) {
    //    outfile->Printf("\n  Contracting:");
    //    outfile->Printf("\n  { %s } { %s
    //    }",lhs.str().c_str(),rhs.str().c_str());

    int ncl = lhs.ncre();
    int nal = lhs.nann();
    int ncr = rhs.ncre();
    int nar = rhs.nann();

    int max_contr = ncl + nal + ncr + nar;

    // Operator table
    std::map<std::vector<int>, Operator> op_table;

    // The product term
    //    outfile->Printf("\n  Contraction rank %d",0);
    std::pair<double, SqOperator> prod_term = simple_contract(lhs, rhs, {});
    res.add(prod_term.first, prod_term.second);

    for (int rank = 2; rank <= max_contr; rank += 2) {
        //        outfile->Printf("\n  Contraction rank %d",rank);
        for (auto& contractions : even_partitions[rank / 2]) {
            //            outfile->Printf("\n    Contraction: ",rank);
            for (int k = 0; k < contractions.size(); ++k) {
                //            for (auto& contraction : contractions){
                //                outfile->Printf(" %d",contractions[k]);
            }
        }
    }
}

std::pair<double, SqOperator>
WickTheorem::simple_contract(const SqOperator& lhs, const SqOperator& rhs,
                             const std::vector<std::pair<int, int>>& pattern) {
    // Compute the contraction of two operators with a k-legged contraction
    // specified by the pattern vector.  The pattern vector stores a list of
    // pairs (group,operator) that specifies where all the "legs" of the
    // contraction fall.
    // For example, pattern = {{0,2},{1,1},{2,0},{2,1}} corresponds to:
    //       _____________
    //      |       |   | |
    // {a b c d} {e f} {g h i j}
    //      ^       ^
    //    (0,2)   (1,1)

    // (lcre)(lann)(rcre)(rann)
    std::vector<std::vector<int>> op_groups{lhs.cre(), lhs.ann(), rhs.cre(), rhs.ann()};

    std::vector<int> lc = lhs.cre();
    std::vector<int> la = lhs.ann();
    std::vector<int> rc = rhs.cre();
    std::vector<int> ra = rhs.ann();

    // Stores the pairs (mo,creation) of operators that have been contracted
    std::vector<std::pair<int, bool>> contr_indices;

    // 1. determine the indices of the one-density/cumulant
    //   _____
    //  | | | |
    //  c f g h {a b d e i j}

    // Form a mask with all the operators to contract
    //       _____________
    //      |       |   | |
    // {a b c d} {e f} {g h i j}
    // {0 0 1 0} {0 1}  {1 1 0 0}
    std::vector<std::vector<bool>> op_mask;
    for (std::vector<int>& group : op_groups) {
        op_mask.push_back(std::vector<bool>(group.size(), false));
    }

    for (std::pair<int, int> leg : pattern) {
        int group = leg.first;
        int op = leg.second;
        contr_indices.push_back(std::make_pair(op_groups[group][op], group % 2 == 0));
        op_mask[group][op] = true;
    }

    // 2. determine the sign and remove operators
    //    for (int g = 0; g < 4; g++){
    //        for (int i : op_groups[g]){
    //            if (op_mask[g][i]){

    //            }else{

    //            }
    //        }
    //    }

    // Remove the contracted operators and compute sign
    double sign_contraction = 1.0;

    // TODO needs implementation

    // Parity of transposition (lcre)(lann)(rcre)(rann) ->
    // (lcre)(rcre)(lann)(rann)
    double sign = 1.0;
    sign *= ((la.size() * rc.size()) % 2 != 0) ? -1.0 : 1.0;

    lc.insert(lc.end(), rc.begin(), rc.end());
    la.insert(la.end(), ra.begin(), ra.end());

    SqOperator op(lc, la);
    sign *= op.sort();
    sign *= sign_contraction;

    return std::make_pair(sign, op);
}

// => SqTest class function <=

SqTest::SqTest() {
    //    SqOperator sqop1({6,5},{0,2,1});
    //    SqOperator sqop2({5,6},{0,1,2});
    //    SqOperator sqop3({5,6},{1,0,2});
    //    SqOperator sqop4({6,5},{2,0,1});

    //    sqop1.test_sort();
    //    sqop2.test_sort();
    //    sqop3.test_sort();
    //    sqop4.test_sort();

    SqOperator sqop1({4}, {0});
    SqOperator sqop2({5}, {0});
    SqOperator sqop3({4, 5}, {0, 1});
    SqOperator sqop4({5}, {1});

    Operator op;
    int no = 4;
    int nv = 10;
    for (int i = 0; i < no; ++i) {
        for (int j = 0; j < no; ++j) {
            for (int a = 0; a < nv; ++a) {
                for (int b = 0; b < nv; ++b) {
                    SqOperator ijab_op({i, j}, {no + a, no + b});
                    op.add(1.0, ijab_op);
                }
            }
        }
    }
    //    op.add(3.0,sqop1);
    //    op.add(5.0,sqop2);
    //    op.add(7.0,sqop3);
    //    op.add(11.0,sqop4);
    //    outfile->Printf("\n%s",op.str().c_str());

    WickTheorem wt;
    Operator op_op = wt.evaluate(op, op);
    outfile->Printf("\n%s", op_op.str().c_str());
}
}
}

// From:
// http://stackoverflow.com/questions/17554242/how-to-obtain-the-index-permutation-after-the-sorting
//    std::vector<int> index(vec.size(), 0);
//    std::iota(index.begin(),index.end(),0);
//    sort(index.begin(), index.end(),
//         [&](const int& a, const int& b) {
//        return (vec[a] < vec[b]);
//    }
//    );
