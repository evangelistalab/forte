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

/* DISABLED DUE TO CHANGES IN THE HANDLING OF SCF IN PSI4
#ifndef ALTERNATIVESCASSCF_H
#define ALTERNATIVESCASSCF_H

#include "psi4/libmints/wavefunction.h"
#include "psi4/libmints/matrix.h"
#include <vector>
#include "psi4/psi4-dec.h"

#include "base_classes/mo_space_info.h"
#include "psi4/libscf_solver/rhf.h"


namespace forte {
/// CASSCF can be expensive so many researchers have come up with different
/// orbitals for CASSCF.
/// The first alternative, that will be implemented, is FT-HF based orbitals.
/// In FT-HF, the occupation numbers obey a fermi dirac distribution.
/// D_{uv} = n[i] * C_{ui}C_{vi}
/// Kevin first implemented this in a toy code, but this code was imported from
/// there.
/// RDMs
/// Ref 1. P. Slavcek and T. J. Martınez. J. Chem. Phys.132(23):234102, 2010.
/// Ref 2. A. D. Rabuck and G. E. Scuseria. J. Chem. Phys. 110(2):695–700, 1999.
///

class FiniteTemperatureHF : public scf::RHF {
  protected:
    /// Core Hamiltonian Matrix
    psi::SharedMatrix hMat_;
    /// The Overlap Matrix
    psi::SharedMatrix sMat_;
    /// The converged CMatrix from SCF
    psi::SharedMatrix CMatrix_;
    /// C = n_i * C_{mu, i}
    psi::SharedMatrix C_occ_folded_;
    /// Just the normal CMatrix
    psi::SharedMatrix C_occ_a_;
    /// A Vector of eigenvalues
    psi::SharedVector eps_;
    /// The active orbital energies (fractionally occupied orbitals)
    std::vector<std::pair<double, int>> active_orb_energy_;
    /// The Fermi-Dirac distribution for occupation
    std::vector<double> fermidirac_;
    /// The MOSpaceInfo object -> Tells active space and things
    std::shared_ptr<MOSpaceInfo> mo_space_info_;
    /// The options object
    Options options_;
    /// The psi::Dimension for restricted docc
    psi::Dimension rdocc_dim_;
    /// The psi::Dimension object for active
    psi::Dimension active_dim_;
    /// The psi::Dimension object for restricted_docc + active
    psi::Dimension rdocc_p_active_;

    /// General variables for use in SCF code
    /// The irrep
    int nirrep_;
    /// The number of basis functions
    int nbf_;
    /// The number of molecular orbitals
    size_t nmo_;
    /// The number of occupied orbitals
    int nocc_;
    /// The number of restricted_docc
    size_t rdocc_;
    /// The number of active
    size_t na_;
    /// The number of virtual orbitals
    size_t nuocc_;
    /// If FTHF< compute density via n_i C C^T but sum over evertything
    void frac_occupation();
    /// Perform a bisection method to solve for the Fermi Level
    double bisection(std::vector<double>&, double T);
    double occ_vec(std::vector<double>& bisect, double ef, double T);
    // The fermi level (n_i = N - solved using bisection method)
    double ef_ = 0.0;
    double scf_energy_ = 0.0;
    /// A function for computing the SCF iterations
    void scf_iteration(const std::shared_ptr<psi::Matrix> C_left);
    /// Function used to get all the SCF prelims
    void startup();
    /// Initialize the occupation vector
    /// restricted_docc -> 2
    /// active          -> f(e)
    /// restricted_uocc -> 0
    void initialize_occupation_vector(std::vector<double>& dirac);
    std::vector<std::pair<double, int>> get_active_orbital_energy();

    /// Use of Derived SCF calculations
    /// Compute the 2J - K
    /// This needs a algorithm that can handle assymetric densities
    virtual void form_G();
    /// This forms D from folded C and regular C.
    virtual void form_D();
    /// Whether or not to print debug stuff
    int debug_ = 0;

  public:
    FiniteTemperatureHF(psi::SharedWavefunction ref_wfn, Options& Options,
                        std::shared_ptr<MOSpaceInfo> mo_space);
    /// Get the SCF ENERGY for the complete iteration
    double get_scf_energy() { return scf_energy_; }
    std::shared_ptr<psi::Matrix> get_mo_coefficient() { return CMatrix_; }
    double compute_energy();
};
}
}
#endif // ALTERNATIVESCASSCF_H
*/
