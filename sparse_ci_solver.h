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

#ifndef _sparse_ci_h_
#define _sparse_ci_h_

#include "bitset_determinant.h"

namespace psi{ namespace libadaptive{

enum DiagonalizationMethod {Full,DavidsonLiuDense,DavidsonLiuSparse,DavidsonLiuString};

/**
 * @brief The SparseCISolver class
 * This class diagonalizes the Hamiltonian in a basis
 * of determinants.
 */
class SparseCISolver
{
public:    
    // ==> Class Constructor and Destructor <==

    /**
     * Constructor
     */
    SparseCISolver();

    /// Destructor
    ~SparseCISolver();

    // ==> Class Interface <==

    /**
     * Diagonalize the Hamiltonian in a basis of determinants
     * @param space The basis for the CI given as a vector of BitsetDeterminant objects
     * @param nroot The number of solutions to find
     * @param diag_method The diagonalization algorithm
     */
    double diagonalize_hamiltonian(std::vector<BitsetDeterminant>& space,
                                   int nroot,
                                   DiagonalizationMethod diag_method = DavidsonLiuSparse);
};

}}

#endif // _sparse_ci_h_
