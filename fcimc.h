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

#ifndef _fcimc_h_
#define _fcimc_h_

#include <fstream>

#include <liboptions/liboptions.h>
#include <libmints/vector.h>
#include <libmints/matrix.h>

#include "integrals.h"
#include "string_determinant.h"

namespace psi{ namespace libadaptive{

class FCIMC
{
public:
    FCIMC(Options &options, ExplorerIntegrals* ints);
    ~FCIMC();
private:
    /// The number of irriducible representations
    int nirrep_;
    /// The wave function symmetry
    int wavefunction_symmetry_;
    /// The number of molecular orbitals
    int nmo_;
    /// The number of alpha electrons
    int nalpha_;
    /// The number of beta electrons
    int nbeta_;
    /// The number of molecular orbitals per irrep
    Dimension nmopi_;
    /// The number of alpha electrons per irrep
    Dimension nalphapi_;
    /// The number of beta electrons per irrep
    Dimension nbetapi_;
    /// The alpha occupation of the reference determinant
    Dimension nalphapi_ref_;
    /// The beta occupation of the reference determinant
    Dimension nbetapi_ref_;
    /// The frozen core orbitals
    Dimension frzcpi_;
    /// The frozen virtual orbitals
    Dimension frzvpi_;
    /// The symmetry of each orbital in Pitzer ordering
    std::vector<int> mo_symmetry_;
    /// The symmetry of each orbital in the qt ordering
    std::vector<int> mo_symmetry_qt_;
    /// A vector that contains all the frozen core
    std::vector<int> frzc_;
    /// A vector that contains all the frozen virtual
    std::vector<int> frzv_;
    /// The nuclear repulsion energy
    double nuclear_repulsion_energy_;
    /// The molecular integrals required by fcimc
    ExplorerIntegrals* ints_;
    /// The reference determinant
    StringDeterminant reference_determinant_;
    /// The determinant with minimum energy
    StringDeterminant min_energy_determinant_;

    void startup(Options& options);
    void read_info(Options& options);
};

}} // End Namespaces

#endif // _fcimc_h_
