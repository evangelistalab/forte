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

#include "psi4/libmints/molecule.h"
#include "psi4/libmints/wavefunction.h"
#include "psi4/liboptions/liboptions.h"

#include "stl_bitset_determinant.h"
#include "helpers.h"

namespace psi {
namespace forte {

class CI_Reference // : public Wavefunction
{
  protected:
    /// Reference energy = FCI_energy + frozen_core_energy + restricted_docc +
    /// nuclear_replusion
    double Eref_;

    SharedWavefunction wfn_;

    int multiplicity_;
    double ms_;

    int nalpha_;
    int nbeta_; 

    int root_sym_;

    int nirrep_;

    std::shared_ptr<MOSpaceInfo> mo_space_info_;

    Dimension mo_symmetry_;

    Dimension nactpi_;

    Dimension frzcpi_;

    std::vector<std::tuple<double, int, int>> sym_labeled_orbitals(std::string type);

  public:
    /// Default constructor
    CI_Reference(std::shared_ptr<Wavefunction> wfn, Options& options, std::shared_ptr<MOSpaceInfo> mo_space_info, STLBitsetDeterminant det, int multiplicity, double ms );

    /// Destructor
    ~CI_Reference();

    /// Build a reference
    void build_reference( std::vector<STLBitsetDeterminant>& ref_space );
    
};
}
} // End Namespaces

