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

#ifndef _wfn_operator_h_
#define _wfn_operator_h_

#include "stl_bitset_determinant.h"
#include "determinant_map.h"


namespace psi{ namespace forte{

/**
 * @brief A class to compute various expectation values, projections, 
 * and matrix elements of quantum mechanical operators on wavefunction objects.
 */

using wfn_hash = det_hash<double>;

class WFNOperator
{
public:

    /// Default constructor
    WFNOperator( std::shared_ptr<MOSpaceInfo> mo_space_info  );

    /// Empty constructor
    WFNOperator();

    /// Build the coupling lists for one-particle operators
    void op_lists( DeterminantMap& wfn );

    /// Build the coupling lists for two-particle operators
    void tp_lists( DeterminantMap& wfn );

    /*- Operators -*/
    
    /// Single excitations, a_p^(+) a_q|>
    void add_singles( DeterminantMap& wfn );

    /// Double excitations, a_p^(+) a_q^(+) a_r a_s|>
    void add_doubles( DeterminantMap& wfn );

    /// Compute total spin expectation value <|S^2|> 
    double s2( DeterminantMap& wfn );

    /// The alpha single-annihilation/creation list
    std::vector< std::vector< std::pair<size_t,short> >> a_ann_list_;
    std::vector< std::vector< std::pair<size_t,short> >> a_cre_list_;

    /// The beta single-annihilation/creation list
    std::vector< std::vector< std::pair<size_t,short> >> b_ann_list_;
    std::vector< std::vector< std::pair<size_t,short> >> b_cre_list_;

    /// The alpha-alpha double-annihilation/creation list
    std::vector< std::vector< std::tuple<size_t,short,short> >> aa_ann_list_;
    std::vector< std::vector< std::tuple<size_t,short,short> >> aa_cre_list_;

    /// The beta-beta single-annihilation/creation list
    std::vector< std::vector< std::tuple<size_t,short,short> >> bb_ann_list_;
    std::vector< std::vector< std::tuple<size_t,short,short> >> bb_cre_list_;

    /// The alfa-beta single-annihilation/creation list
    std::vector< std::vector< std::tuple<size_t,short,short> >> ab_ann_list_;
    std::vector< std::vector< std::tuple<size_t,short,short> >> ab_cre_list_;
protected:

    /// Initialize important variables on construction
    void startup();

    /// The MOSpaceInfo object
    std::shared_ptr<MOSpaceInfo> mo_space_info_;

    /// Active space symmetry
    std::vector<int> mo_symmetry_;

};

}}

#endif // _wfn_operator_h_
