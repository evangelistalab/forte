#ifndef _run_dsrg_h_
#define _run_dsrg_h_

#include "base_classes/forte_options.h"
#include "sci/fci_mo.h"
#include "mrdsrg-helper/dsrg_transformed.h"
#include "mrdsrg-spin-integrated/dsrg_mrpt2.h"
#include "mrdsrg-spin-integrated/dsrg_mrpt3.h"
#include "mrdsrg-spin-integrated/master_mrdsrg.h"
#include "mrdsrg-spin-integrated/three_dsrg_mrpt2.h"
#include "mrdsrg-spin-integrated/mrdsrg.h"
#include "mrdsrg-so/mrdsrg_so.h"
#include "mrdsrg-so/so-mrdsrg.h"
#include "mrdsrg-spin-adapted/dsrg_mrpt.h"

namespace forte {

/// Reference relaxation, relaxed dipoles, transition dipoles,
/// general sequence of running dsrg should be implemented in this class

// class RUN_DSRG {
//  public:
//    /**
//     * Constructor
//     * @param ref_wfn The reference wavefunction object
//     * @param options The main options object
//     * @param ints A pointer to an allocated integral object
//     * @param mo_space_info The MOSpaceInfo object
//     */
//    RUN_DSRG(RDMs rdms, psi::SharedWavefunction ref_wfn, std::shared_ptr<ForteOptions> options,
//             std::shared_ptr<ForteIntegrals> ints, std::shared_ptr<MOSpaceInfo> mo_space_info);

//    /// Compute DSRG energy
//    double compute_dsrg_energy();

//    /// Compute DSRG density
//    void compute_dsrg_density();

// protected:
//    /// RDMs type (FCI for FCI_MO)
//    std::string ref_type_;

//};

std::unique_ptr<MASTER_DSRG> make_dsrg_method(const std::string& method, RDMs rdms,
                                              std::shared_ptr<SCFInfo> scf_info,
                                              std::shared_ptr<ForteOptions> options,
                                              std::shared_ptr<ForteIntegrals> ints,
                                              std::shared_ptr<MOSpaceInfo> mo_space_info);

std::unique_ptr<MRDSRG_SO> make_dsrg_so_y(RDMs rdms, std::shared_ptr<SCFInfo> scf_info,
                                        std::shared_ptr<ForteOptions> options,
                                        std::shared_ptr<ForteIntegrals> ints,
                                        std::shared_ptr<MOSpaceInfo> mo_space_info);

std::unique_ptr<SOMRDSRG> make_dsrg_so_f(RDMs rdms, std::shared_ptr<SCFInfo> scf_info,
                                        std::shared_ptr<ForteOptions> options,
                                        std::shared_ptr<ForteIntegrals> ints,
                                        std::shared_ptr<MOSpaceInfo> mo_space_info);

std::unique_ptr<DSRG_MRPT> make_dsrg_spin_adapted(RDMs rdms, std::shared_ptr<SCFInfo> scf_info,
                                                  std::shared_ptr<ForteOptions> options,
                                                  std::shared_ptr<ForteIntegrals> ints,
                                                  std::shared_ptr<MOSpaceInfo> mo_space_info);

} // namespace forte

#endif // RUN_DSRG_H
