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

#ifndef _tdci_h_
#define _tdci_h_

#include <fstream>
#include <iomanip>

#include "psi4/libmints/wavefunction.h"
#include "psi4/liboptions/liboptions.h"
#include "psi4/physconst.h"

#include "../forte_options.h"
#include "../ci_rdm/ci_rdms.h"
#include "../ci_reference.h"
#include "../fci/fci_integrals.h"
#include "../mrpt2.h"
#include "../orbital-helper/unpaired_density.h"
#include "../determinant_hashvector.h"
#include "../reference.h"
#include "../sparse_ci/sparse_ci_solver.h"
#include "../sparse_ci/determinant.h"
#include "../orbital-helper/iao_builder.h"
#include "../orbital-helper/localize.h"
#include "aci.h"

#ifdef _OPENMP
#include <omp.h>
#else
#define omp_get_max_threads() 1
#define omp_get_thread_num() 0
#define omp_get_num_threads() 1
#endif

namespace psi {
namespace forte {

class Reference;

/// Set the ACI options
void set_TDACI_options(ForteOptions& foptions);

/**
 * @brief The TDACI class
 * This class implements an adaptive CI algorithm
 */
class TDACI : public Wavefunction {
  public:
    // ==> Class Constructor and Destructor <==

    /**
     * Constructor
     * @param ref_wfn The reference wavefunction object
     * @param options The main options object
     * @param ints A pointer to an allocated integral object
     * @param mo_space_info A pointer to the MOSpaceInfo object
     */
    TDACI(SharedWavefunction ref_wfn, Options& options, std::shared_ptr<ForteIntegrals> ints,
               std::shared_ptr<MOSpaceInfo> mo_space_info);

    /// Destructor
    ~TDACI();

    // ==> Class Interface <==

    /// Compute the energy
    double compute_energy();

  private:
    std::shared_ptr<ForteIntegrals> ints_;
    SharedWavefunction wfn_;      
    std::shared_ptr<MOSpaceInfo> mo_space_info_; 

    void annihilate_wfn( DeterminantHashVec& olddets, DeterminantHashVec& adets, int frz_orb );

    void renormalize_wfn( std::vector<double>& acoeffs );

    void save_matrix(SharedMatrix mat, std::string str);
    void save_vector(SharedVector vec, std::string str);
    void save_vector(std::vector<double>& vec, std::string str);
    void save_vector(std::vector<size_t>& vec, std::string str);
    void save_vector(std::vector<std::string>& vec, std::string str);

    void propogate_exact( SharedVector C0, SharedMatrix H);
    void propogate_cn( SharedVector C0, SharedMatrix H);
    void propogate_taylor1( SharedVector C0, SharedMatrix H);
    void propogate_taylor2( SharedVector C0, SharedMatrix H);
    void propogate_RK4( SharedVector C0, SharedMatrix H);
    

};

} // namespace forte
} // namespace psi

#endif // _tdaci_h_
