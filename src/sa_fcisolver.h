#ifndef SA_FCISOLVER_H
#define SA_FCISOLVER_H
#include <liboptions/liboptions.h>
#include <libmints/wavefunction.h>
#include "fci_solver.h"
#include "fci_vector.h"
namespace psi{ namespace forte{
/// SA_FCISolver seeks to call multiple instances of CAS-CI and combine all the RDMS and average them

class SA_FCISolver
{
public:
    SA_FCISolver(Options& options, boost::shared_ptr<Wavefunction> wfn);
    ///E_{sa-casscf} = gamma_{avg} h_{pq} + Gamma_{avg} g_{pqrs}
    double compute_energy();
    Reference reference()
    {
        return sa_ref_;
    }

    void set_integral_pointer(std::shared_ptr<FCIIntegrals> fci_ints)
    {
        fci_ints_ = fci_ints;
    }
    void set_mo_space_info(std::shared_ptr<MOSpaceInfo> mo_space_info)
    {
        mo_space_info_ = mo_space_info;
    }
    void set_integrals(std::shared_ptr<ForteIntegrals> ints)
    {
        ints_ = ints;
    }

private:
    void read_options();
    Options options_;
    boost::shared_ptr<Wavefunction> wfn_;
    ///The integrals object (same for all SA Computations)
    /// Generate this once and pass this information to FCISolver
    std::shared_ptr<FCIIntegrals> fci_ints_;
    std::shared_ptr<ForteIntegrals> ints_;
    std::shared_ptr<MOSpaceInfo> mo_space_info_;
    std::vector<std::tuple<int, int, int> > parsed_options_;
    /// The total number of states to be averaged in casscf
    int number_of_states_;
    Reference sa_ref_;
};

}}

#endif // SA_FCISOLVER_H
