#ifndef _run_dsrg_h_
#define _run_dsrg_h_

#include "forte_options.h"
#include "sci/fci_mo.h"
#include "fci/fci.h"
#include "mrdsrg-spin-integrated/dsrg_mrpt2.h"
#include "mrdsrg-spin-integrated/dsrg_mrpt3.h"
#include "mrdsrg-spin-integrated/master_mrdsrg.h"
#include "mrdsrg-spin-integrated/three_dsrg_mrpt2.h"
#include "mrdsrg-spin-integrated/mrdsrg.h"


namespace forte {

/// Set the DSRG options
void set_DSRG_options(ForteOptions& foptions);

/// Reference relaxation, relaxed dipoles, transition dipoles,
/// general sequence of running dsrg should be implemented in this class

//class RUN_DSRG {
//  public:
//    /**
//     * Constructor
//     * @param ref_wfn The reference wavefunction object
//     * @param options The main options object
//     * @param ints A pointer to an allocated integral object
//     * @param mo_space_info The MOSpaceInfo object
//     */
//    RUN_DSRG(Reference reference, psi::SharedWavefunction ref_wfn, psi::Options& options,
//             std::shared_ptr<ForteIntegrals> ints, std::shared_ptr<MOSpaceInfo> mo_space_info);

//    /// Compute DSRG energy
//    double compute_dsrg_energy();

//    /// Compute DSRG density
//    void compute_dsrg_density();

//protected:
//    /// Reference type (FCI for FCI_MO)
//    std::string ref_type_;


//};
}
}
#endif // RUN_DSRG_H
