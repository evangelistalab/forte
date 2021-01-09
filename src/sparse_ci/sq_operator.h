/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2020 by its authors (see COPYING, COPYING.LESSER, AUTHORS).
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

#ifndef _sq_operator_h_
#define _sq_operator_h_

#include <vector>

#include "sparse_ci/determinant.h"

namespace forte {

class ActiveSpaceIntegrals;

/**
 * A data structure used to represent a second quantized operator string like
 *
 * ... op(2) op(1) op(0), where op(i) = a^\dagger_(orb_i, spin_i) or a_(orb_i, spin_i)
 *
 * like, for example
 *
 *  a^\dagger_{2,\alpha} a^\dagger_{3,\beta} a_{1,\beta} a_{0,\alpha}
 *
 * This operator is stored as
 *
 *  [(false,true,0),(false,false,1),(true,false,3),(true,true,2)]
 *
 * The data format is
 *
 *  [(creation_0, spin_0, orb_0), (creation_1, spin_1, orb_1), ...]
 *
 * where the operators are arranged as
 *
 * where
 *  creation_i  : bool (true = creation, false = annihilation)
 *  spin_i      : bool (true = alpha, false = beta)
 *  orb_i       : int  (the index of the mo)
 *
 */
using op_tuple_t = std::vector<std::tuple<bool, bool, int>>;

/**
 * @brief A class to represent a second quantized operator in normal ordered
 * form with respect to the true vacuum
 *
 *   a+_p1 a+_p2  ... a+_P1 a+_P2   ... a-_Q2 a-_Q1   ... a-_q2 a-_q1
 *   alpha creation  beta creation   alpha annihilation  beta annihilation
 *
 * The creation and annihilation operators are stored separately as bit arrays
 * using the Determinant class
 */
class SQOperator {
  public:
    SQOperator(double factor, const Determinant& cre, const Determinant& ann);
    SQOperator(const op_tuple_t& ops, double coefficient = 0.0);
    double factor() const;
    const Determinant& cre() const;
    const Determinant& ann() const;
    void set_factor(double& value);
    std::string str() const;
    std::string latex() const;

  private:
    double factor_;
    Determinant cre_;
    Determinant ann_;
};

/// This function converts a string to a single operator
std::vector<SQOperator> string_to_op_term(const std::string& str);

} // namespace forte

#endif // _sq_operator_h_
