#ifndef ALTERNATIVESCASSCF_H
#define ALTERNATIVESCASSCF_H

#include <libmints/wavefunction.h>
#include <libmints/matrix.h>
#include <vector>
#include <psi4-dec.h>
#include <liboptions/liboptions.h>
#include "helpers.h"
#include <libscf_solver/rhf.h>

namespace psi { namespace forte {
/// This class will implement various alteratives to CASSCF.
/// CASSCF can be expensive so many researchers have come up with different
/// orbitals for CASSCF.
/// The first alternative, that will be implemented, is FT-HF based orbitals.
/// In FT-HF, the occupation numbers obey a fermi dirac distribution.
/// D_{uv} = n[i] * C_{ui}C_{vi}
/// Kevin first implemented this in a toy code, but this code was imported from there.
/// References
/// Ref 1. P. Slavcek and T. J. Martınez. J. Chem. Phys.132(23):234102, 2010.
/// Ref 2. A. D. Rabuck and G. E. Scuseria. J. Chem. Phys. 110(2):695–700, 1999.
///
/// Implementation:
/// Grab the C Matrices from a previous scf computation.  Only allow the active orbitals to have variable occupation
/// This means that the occupation will be 2 and 0 for restricted_docc and restricted_uocc.
/// The total sum of the occupation needs to equal the number of electrons
/// This is enforced by the bisection method
/// I do not want to implement a full SCF code in here, so maybe, I will restrict iterations

class FiniteTemperatureHF : public scf::RHF
{
protected:
    ///Core Hamiltonian Matrix
    SharedMatrix hMat_;
    ///The Overlap Matrix
    SharedMatrix sMat_;
    ///The converged CMatrix from SCF
    SharedMatrix CMatrix_;
    ///C = n_i * C_{mu, i}
    SharedMatrix C_occ_folded_;
    /// Just the normal CMatrix
    SharedMatrix C_occ_a_;
    /// A Vector of eigenvalues
    SharedVector eps_;
    /// The active orbital energies (fractionally occupied orbitals)
    std::vector<std::pair<double, int> > active_orb_energy_;
    /// The Fermi-Dirac distribution for occupation
    std::vector<double> fermidirac_;
    /// The wavefunction object
    boost::shared_ptr<Wavefunction> wfn_;
    /// The MOSpaceInfo object -> Tells active space and things
    std::shared_ptr<MOSpaceInfo> mo_space_info_;
    ///The options object
    Options options_;
    /// The Dimension for restricted docc
    Dimension rdocc_dim_;
    /// The Dimension object for active
    Dimension active_dim_;
    /// The Dimension object for restricted_docc + active
    Dimension rdocc_p_active_;

    /// General variables for use in SCF code
    /// The irrep
    int nirrep_;
    /// The number of basis functions
    int nbf_;
    ///The number of molecular orbitals
    size_t nmo_;
    ///The number of occupied orbitals
    int nocc_;
    ///The number of restricted_docc
    size_t rdocc_;
    ///The number of active
    size_t na_;
    ///The number of virtual orbitals
    size_t nuocc_;
    ///If FTHF< compute density via n_i C C^T but sum over evertything
    void frac_occupation();
    ///Perform a bisection method to solve for the Fermi Level
    double bisection(std::vector<double>&, double T);
    double occ_vec(std::vector<double>& bisect, double ef, double T);
    //The fermi level (n_i = N - solved using bisection method)
    double ef_ = 0.0;
    double scf_energy_ = 0.0;
    /// A function for computing the SCF iterations
    void scf_iteration(const boost::shared_ptr<Matrix> C_left);
    /// Function used to get all the SCF prelims
    void startup();
    /// Initialize the occupation vector
    /// restricted_docc -> 2
    /// active          -> f(e)
    /// restricted_uocc -> 0
    void initialize_occupation_vector(std::vector<double>& dirac);
    std::vector<std::pair<double, int> > get_active_orbital_energy();

    /// Use of Derived SCF calculations
    /// Compute the 2J - K
    /// This needs a algorithm that can handle assymetric densities
    virtual void form_G();
    /// This forms D from folded C and regular C.
    virtual void form_D();
    /// Whether or not to print debug stuff
    int debug_ = 0;

public:
    FiniteTemperatureHF(boost::shared_ptr<Wavefunction> wfn, Options& Options, std::shared_ptr<MOSpaceInfo> mo_space);
    /// Get the SCF ENERGY for the complete iteration
    double get_scf_energy(){return scf_energy_;}
    boost::shared_ptr<Matrix> get_mo_coefficient(){return CMatrix_;}
    double compute_energy();





};
}}
#endif // ALTERNATIVESCASSCF_H
