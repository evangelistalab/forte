#ifndef _dwms_mrpt2_h_
#define _dwms_mrpt2_h_

#include "psi4/liboptions/liboptions.h"

#include "../integrals/integrals.h"
#include "../helpers.h"

namespace psi {
namespace forte {

void set_DWMS_options(ForteOptions& foptions);

void compute_dwms_mrpt2_energy(SharedWavefunction ref_wfn, Options& options,
                               std::shared_ptr<ForteIntegrals> ints,
                               std::shared_ptr<MOSpaceInfo> mo_space_info);

//class DWMS_DSRGPT2 : public Wavefunction {
//  public:
//    /**
//     * @brief DWMS_DSRGPT2 Constructor
//     * @param ref_wfn The reference wavefunction object
//     * @param options PSI4 and FORTE options
//     * @param ints ForteInegrals
//     * @param mo_space_info MOSpaceInfo
//     */
//    DWMS_DSRGPT2(SharedWavefunction ref_wfn, Options& options, std::shared_ptr<ForteIntegrals> ints,
//                 std::shared_ptr<MOSpaceInfo> mo_space_info);

//    /// compute energy and return the ground state energy
//    double compute_energy();

//  private:
//    /// preparation
//    void startup();

//    /// gaussian cutoff for density reweighting
//    double zeta_;

//    /// irrep symbols
//    std::vector<std::string> irrep_symbol_;

//    /// multiplicity symbols
//    std::vector<std::string> multi_symbol_;

//    /// unitary matrices (in active space) from original to semicanonical
//    ambit::Tensor Ua_;
//    ambit::Tensor Ub_;

//    /// max rdm level
//    int max_rdm_level_;
//}
}
}
#endif // DWMS_MRPT2_H
