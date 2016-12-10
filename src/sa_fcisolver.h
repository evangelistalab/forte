#ifndef SA_FCISOLVER_H
#define SA_FCISOLVER_H

#include "psi4/liboptions/liboptions.h"
#include "psi4/libmints/wavefunction.h"
#include "psi4/libmints/wavefunction.h"
#include "fci_solver.h"
#include "fci_vector.h"

namespace psi{ namespace forte{

/// SA_FCISolver seeks to call multiple instances of CAS-CI and combine all the RDMS and average them
class SA_FCISolver
{
public:

    SA_FCISolver(Options& options, std::shared_ptr<Wavefunction> wfn);

    /// E_{sa-casscf} = gamma_{avg} h_{pq} + Gamma_{avg} g_{pqrs}
    double compute_energy();

    Reference reference() { return sa_ref_; }

    void set_integral_pointer(std::shared_ptr<FCIIntegrals> fci_ints) { fci_ints_ = fci_ints; }

    void set_mo_space_info(std::shared_ptr<MOSpaceInfo> mo_space_info){ mo_space_info_ = mo_space_info; }

    void set_integrals(std::shared_ptr<ForteIntegrals> ints){ ints_ = ints; }

    std::vector<std::shared_ptr<FCIWfn>> StateAveragedCISolution(){ return SA_C_; }

private:
    /// Options from Psi4
    Options options_;
    /// The wavefunction object of Psi4
    std::shared_ptr<Wavefunction> wfn_;
    /// Integral objects (same for all SA computations)
    std::shared_ptr<ForteIntegrals> ints_;
    std::shared_ptr<FCIIntegrals> fci_ints_;
    /// MO space information of FORTE
    std::shared_ptr<MOSpaceInfo> mo_space_info_;

    /// A vector of the averaged FCI solutions
    std::vector<std::shared_ptr<FCIWfn>> SA_C_;

    /// The vector that contains states and weights information
    std::vector<std::tuple<int, int, int, std::vector<double>>> parsed_options_;

    /// The total number of states to be averaged
    int nstates_;

    /// The reference object in FORTE
    Reference sa_ref_;

    /// Read options and fill in parsed_options_
    void read_options();
};

}}

#endif // SA_FCISOLVER_H
