/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2021 by its authors (see COPYING, COPYING.LESSER, AUTHORS).
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

#ifdef ENABLE_UNTESTED_CODE

#ifndef _active_dsrgpt2_h_
#define _active_dsrgpt2_h_

#include <string>
#include <vector>
#include <sys/stat.h>

#include "psi4/libmints/matrix.h"
#include "psi4/libmints/molecule.h"
#include "psi4/libmints/vector.h"
#include "psi4/libmints/wavefunction.h"

#include "psi4/libpsio/psio.h"
#include "psi4/libpsio/psio.hpp"
#include "psi4/libqt/qt.h"
#include "ambit/blocked_tensor.h"
#include "ambit/tensor.h"

#include "base_classes/mo_space_info.h"
#include "integrals/integrals.h"
#include "base_classes/rdms.h"
#include "sparse_ci/determinant.h"
#include "master_mrdsrg.h"
#include "dsrg_mrpt2.h"
#include "three_dsrg_mrpt2.h"

namespace forte {

class FCI_MO;
class SCFInfo;
class ForteOptions;

struct Vector4 {
    double x, y, z, t;
};

class ACTIVE_DSRGPT2 {
  public:
    /**
     * @brief ACTIVE_DSRGPT2 Constructor
     * @param scf_info The SCFInfo
     * @param options The ForteOptions
     * @param ints ForteInegrals
     * @param mo_space_info MOSpaceInfo
     */
    ACTIVE_DSRGPT2(std::shared_ptr<SCFInfo> scf_info, std::shared_ptr<ForteOptions> options,
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

    /// SCF info
    std::shared_ptr<SCFInfo> scf_info_;

    /// ForteOptions
    std::shared_ptr<ForteOptions> foptions_;

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
    std::vector<std::vector<double>> pt2_energies_;

    /// Singles (T1) percentage in VCISD
    std::vector<std::vector<double>> t1_percentage_;

    /// Dominant determinants
    std::vector<std::vector<Determinant>> dominant_dets_;

    /// Compute the excitaion type based on ref_det
    std::string compute_ex_type(const Determinant& det1, const Determinant& ref_det);

    /// Compute unrelaxed DSRG-MRPT2 energy and return energy value
    double compute_dsrg_mrpt2_energy(std::shared_ptr<MASTER_DSRG>& dsrg, RDMs& reference);

    /// Set FCI_MO parameters
    void set_fcimo_params(int nroots, int root, int multiplicity);

    /// Rotate amplitudes
    void rotate_amp(psi::SharedMatrix Ua, psi::SharedMatrix Ub, ambit::BlockedTensor& T1,
                    ambit::BlockedTensor& T2);

    /// Rotate to semicanonical orbitals and pass to this
    void rotate_orbs(psi::SharedMatrix Ca0, psi::SharedMatrix Cb0, psi::SharedMatrix Ua,
                     psi::SharedMatrix Ub);

    /// Transform integrals using the orbital coefficients
    void transform_integrals(psi::SharedMatrix Ca0, psi::SharedMatrix Cb0);

    /// MO dipole integrals in C1 Pitzer ordering in the original basis
    std::vector<psi::SharedMatrix> modipole_ints_;

    /// Active indices in C1 symmetry per irrep
    std::vector<std::vector<size_t>> actvIdxC1_;
    /// Core indices in C1 symmetry per irrep
    std::vector<std::vector<size_t>> coreIdxC1_;
    /// Virtual indices in C1 symmetry per irrep
    std::vector<std::vector<size_t>> virtIdxC1_;

    /// Compute VCIS/VCISD transition dipole from root0 -> root1
    Vector4 compute_td_ref_root(std::shared_ptr<ActiveSpaceIntegrals> fci_ints,
                                const std::vector<Determinant>& p_space, psi::SharedMatrix evecs,
                                const int& root0, const int& root1);
    /// Compute VCIS/VCISD oscillator strength
    /// Only compute root_0 of eigen0 -> root_n of eigen0 or eigen1
    /// eigen0 and eigen1 are assumed to be different by default
    void compute_osc_ref(const int& irrep0, const int& irrep1,
                         const std::vector<Determinant>& p_space0,
                         const std::vector<Determinant>& p_space1,
                         const std::vector<std::pair<psi::SharedVector, double>>& eigen0,
                         const std::vector<std::pair<psi::SharedVector, double>>& eigen1);

    /// Transition dipole moment of reference in a.u.
    std::map<std::string, Vector4> tdipole_ref_;
    /// Oscillator strength of reference
    std::map<std::string, Vector4> f_ref_;

    /// Transition dipole moment of perturbation in a.u.
    std::map<std::string, Vector4> tdipole_pt2_;
    /// Oscillator strength of perturbation
    std::map<std::string, Vector4> f_pt2_;

    /// A uniform format for transition type
    std::string transition_type(const int& n0, const int& irrep0, const int& n1, const int& irrep1);

    /// Combine reference wavefunction of different symmetry
    psi::SharedMatrix combine_evecs(const int& h0, const int& h1);

    /// Store a copy of the ground-state determinants
    std::vector<Determinant> p_space_g_;
    /// Store a copy of all reference wavefunctions in the original basis
    std::vector<psi::SharedMatrix> ref_wfns_;

    /// Scalar term from T amplitudes de-normal-ordering of the ground state
    double Tde_g_;
    /// De-normal-ordered T1 amplitudes of the ground state
    ambit::BlockedTensor T1_g_;
    /// (De-normal-ordered) T2 amplitudes of the ground state
    ambit::BlockedTensor T2_g_;

    /// Compute the DSRG-PT2 oscillator strength
    void compute_osc_pt2(const int& irrep, const int& root, const double& Tde_x,
                         ambit::BlockedTensor& T1_x, ambit::BlockedTensor& T2_x);
    /**
     * Compute effective 1st-order transition densities
     * for transition dipole computaions in active_dsrgpt2.
     * @param T1 T1 amplitudes
     * @param T2 T2 amplitudes
     * @param TD1 one-body transition density of the reference
     * @param TD2 two-body transition density of the reference
     * @param TD3 three-body transition density of the reference
     * @param transpose Transpose the 0th-order transition density if true
     * @return the fully contracted term of T * TD
     *
     * The effective 1st-order trans. dens. is defined as follows:
     * for example: <X0|mu T1(G)|G0> = (mu)^a_u * (t)^v_a * (td)^u_v + ...
     *                               = (mu)^a_u * (td_eff)^u_a + ...
     *              where T1(G) is the de-normal-ordered singles of state G,
     *              and td_eff is defined as effective 1st-order transition density.
     */
    double compute_TDeff(ambit::BlockedTensor& T1, ambit::BlockedTensor& T2,
                         ambit::BlockedTensor& TD1, ambit::BlockedTensor& TD2,
                         ambit::BlockedTensor& TD3, ambit::BlockedTensor& TDeff,
                         const bool& transpose);

    /// Print summary
    void print_summary();
    /// Print oscillator strength and transition dipoles
    void print_osc();

    /// Format a double to string
    std::string format_double(const double& value, const int& width, const int& precision,
                              const bool& scientific = false);

    /// Orbital extents of original orbitals
    std::vector<double> orb_extents_;

    /// Flatten the structure of orbital extents in fci_mo and return a vector of <r^2>
    std::vector<double>
    flatten_fci_orbextents(const std::vector<std::vector<std::vector<double>>>& fci_orb_extents);

    // ==> debug functions for pt2 oscillator strength <==

    /**
     * IMPORTANT NOTE:
     *   1) All blocks of T should be stored
     *   2) Number of basis function should not exceed 128
     */

    /// transform the reference determinants of size nactive to size nmo with Pitzer ordering
    std::map<Determinant, double> p_space_actv_to_nmo(const std::vector<Determinant>& p_space,
                                                      psi::SharedVector wfn);

    /// generate excited determinants from the reference
    std::map<Determinant, double> excited_wfn_1st(const std::map<Determinant, double>& ref,
                                                  ambit::BlockedTensor& T1,
                                                  ambit::BlockedTensor& T2);

    /// compute pt2 oscillator strength using determinants
    void compute_osc_pt2_dets(const int& irrep, const int& root, const double& Tde_x,
                              ambit::BlockedTensor& T1_x, ambit::BlockedTensor& T2_x);

    /// generate singly excited determinants from the reference
    std::map<Determinant, double> excited_ref(const std::map<Determinant, double>& ref,
                                              const int& p, const int& q);

    /// compute overlap between two wavefunctions
    double compute_overlap(std::map<Determinant, double> wfn1, std::map<Determinant, double> wfn2);

    /// compute pt2 oscillator strength using determinants overlap
    void compute_osc_pt2_overlap(const int& irrep, const int& root, ambit::BlockedTensor& T1_x,
                                 ambit::BlockedTensor& T2_x);
};
} // namespace forte

#endif // ACTIVE_DSRGPT2_H
#endif
