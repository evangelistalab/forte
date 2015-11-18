#ifndef ALTERNATIVESCASSCF_H
#define ALTERNATIVESCASSCF_H

#include <libmints/wavefunction.h>
#include <libmints/matrix.h>
#include <vector>
#include <psi4-dec.h>
#include <liboptions/liboptions.h>
#include "helpers.h"

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

class FiniteTemperatureHF : public Wavefunction
{
protected:
    ///Core Hamiltonian Matrix
    SharedMatrix hMat_;
    ///The Overlap Matrix
    SharedMatrix sMat_;
    ///The converged CMatrix
    SharedMatrix CMatrix_;
    /// A Vector of eigenvalues
    std::vector<double> eps_;
    /// The Fermi-Dirac distribution for occupation
    std::vector<double> fermidirac_;
    /// The wavefunction object
    boost::shared_ptr<Wavefunction> wfn_;
    /// The MOSpaceInfo object -> Tells active space and things
    std::shared_ptr<MOSpaceInfo> mo_space_info_;
    ///The options object
    Options options_;


    /// General variables for use in SCF code
    int nbf_;
    ///The number of occupied orbitals
    int nocc_;
    ///If FTHF< compute density via n_i C C^T but sum over evertything
    boost::shared_ptr<Matrix> frac_occupation(SharedMatrix,int &iter, bool &t_done);
    ///Perform a bisection method to solve for the Fermi Level
    double bisection(std::vector<double>&, double T);
    double occ_vec(std::vector<double>& bisect, double ef, double T);
    //The fermi level (n_i = N - solved using bisection method)
    double ef_ = 0.0;
    double scf_energy_ = 0.0;
    /// A function for computing the SCF iterations
    void scf_iteration();
    /// Function used to get all the SCF prelims
    void startup();

public:
    FiniteTemperatureHF(boost::shared_ptr<Wavefunction> wfn, Options& Options, std::shared_ptr<MOSpaceInfo> mo_space);
    /// Get the SCF ENERGY for the complete iteration
    double get_scf_energy(){return scf_energy_;}
    boost::shared_ptr<Matrix> get_mo_coefficient(){return CMatrix_;}
    double compute_energy();





};
}}
#endif // ALTERNATIVESCASSCF_H
