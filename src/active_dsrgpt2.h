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

#ifndef _active_dsrgpt2_h_
#define _active_dsrgpt2_h_

#include <string>
#include <vector>

#include "psi4/libmints/matrix.h"
#include "psi4/libmints/molecule.h"
#include "psi4/libmints/vector.h"
#include "psi4/libmints/wavefunction.h"
#include "psi4/liboptions/liboptions.h"
#include "psi4/libpsio/psio.h"
#include "psi4/libpsio/psio.hpp"
#include "psi4/libqt/qt.h"

#include "helpers.h"
#include "integrals/integrals.h"
#include "mrdsrg-spin-free/dsrg_mrpt2.h"
#include "mrdsrg-spin-free/dsrg_mrpt3.h"
#include "mrdsrg-spin-free/three_dsrg_mrpt2.h"
#include "reference.h"
#include "stl_bitset_determinant.h"

namespace psi {
namespace forte {

class FCI_MO;

struct Vector4 {
    double x, y, z, t;
};

class ACTIVE_DSRGPT2 : public Wavefunction {
  public:
    /**
     * @brief ACTIVE_DSRGPT2 Constructor
     * @param ref_wfn The reference wavefunction object
     * @param options PSI4 and FORTE options
     * @param ints ForteInegrals
     * @param mo_space_info MOSpaceInfo
     */
    ACTIVE_DSRGPT2(SharedWavefunction ref_wfn, Options& options,
                   std::shared_ptr<ForteIntegrals> ints,
                   std::shared_ptr<MOSpaceInfo> mo_space_info);

    /// Destructor
    ~ACTIVE_DSRGPT2();

    /// Compute energy
    double compute_energy();

  private:
    /// Basic Preparation
    void startup();

    /// Integrals
    std::shared_ptr<ForteIntegrals> ints_;

    /// MO space info
    std::shared_ptr<MOSpaceInfo> mo_space_info_;

    /// Name of the code
    std::string code_name_;

    /// Multiplicity
    int multiplicity_;

    /// Total number of roots
    int total_nroots_;

    /// Number of roots per irrep
    std::vector<int> nrootpi_;

    /// Irrep symbol
    std::vector<std::string> irrep_symbol_;

    /// The FCI_MO object
    std::shared_ptr<FCI_MO> fci_mo_;

    /// Reference type (CIS/CISD/COMPLETE)
    std::string ref_type_;

    /// Reference energies
    std::vector<std::vector<double>> ref_energies_;

    /// DSRGPT2 energies
    std::vector<std::vector<double>> pt_energies_;

    /// Singles (T1) percentage in VCISD
    std::vector<std::vector<double>> t1_percentage_;

    /// Dominant determinants
    std::vector<std::vector<STLBitsetDeterminant>> dominant_dets_;

    /// Compute the excitaion type based on ref_det
    std::string compute_ex_type(const STLBitsetDeterminant& det1,
                                const STLBitsetDeterminant& ref_det);

    /** Precompute all energies to
     *  1) determine excitation type
     *  2) obtain original orbital extent
     *  3) determine %T1 in CISD
     *  4) obtain unitary matrices that semicanonicalize the orbitals
     *  5) compute CIS or CISD oscillator strength
     */
    void precompute_energy();

    /// Unitary matrices that semicanonicalize orbitals of each state
    std::vector<std::vector<SharedMatrix>> Uaorbs_;
    std::vector<std::vector<SharedMatrix>> Uborbs_;

    /// Rotate to semicanonical orbitals and pass to this
    void rotate_orbs(SharedMatrix Ca0, SharedMatrix Cb0, SharedMatrix Ua, SharedMatrix Ub);

    /// Compute CIS/CISD transition dipole from root0 -> root1
    Vector4 compute_root_trans_dipole(const std::vector<SharedMatrix>& aodipole_ints,
                                      SharedMatrix sotoao, std::shared_ptr<FCIIntegrals> fci_ints,
                                      const std::vector<STLBitsetDeterminant>& p_space,
                                      SharedMatrix evecs, const int& root0, const int& root1);
    /// Compute CIS/CISD oscillator strength
    /// Only compute root_0 of eigen0 -> root_n of eigen0 or eigen1
    /// eigen0 and eigen1 are assumed to be different by default
    void compute_oscillator_strength(const int& irrep0, const int& irrep1,
                                     const std::vector<STLBitsetDeterminant>& p_space0,
                                     const std::vector<STLBitsetDeterminant>& p_space1,
                                     const std::vector<std::pair<SharedVector, double>>& eigen0,
                                     const std::vector<std::pair<SharedVector, double>>& eigen1,
                                     const bool& same = false);
    /// Active indices in C1 symmetry per irrep
    std::vector<std::vector<size_t>> actvIdxC1_;
    /// Oscillator strength
    std::map<std::string, Vector4> f_ref_;

    /// Print summary
    void print_summary();

    /// Orbital extents of original orbitals
    std::vector<double> orb_extents_;

    /// Flatten the structure of orbital extents in fci_mo and return a vector
    /// of <r^2>
    std::vector<double>
    flatten_fci_orbextents(const std::vector<std::vector<std::vector<double>>>& fci_orb_extents);
};
}
}

#endif // ACTIVE_DSRGPT2_H
