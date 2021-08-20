#ifndef _embedding_density_h_
#define _embedding_density_h_

#include <vector>

#include "base_classes/scf_info.h"
#include "base_classes/forte_options.h"
#include "base_classes/mo_space_info.h"
#include "base_classes/rdms.h"
#include "base_classes/state_info.h"
#include "base_classes/active_space_solver.h"
#include "base_classes/forte_options.h"
#include "casscf/casscf.h"
#include "integrals/active_space_integrals.h"

namespace forte {

/**
* @brief EMBEDDING_DENSITY generate density for the embedded fragment.
* The density is organized as a RDMs object.
* For RHF, we simply write the diagonal (i, i) elements to be 1.0, and 0.0 for other.
* CASCI and CASSCF densities are generated using Active_Space_Solver and CASSCF objects.
* Note that the rdms generated here depends on input mo_space_info, and
* for embedding codes, mo_spce_info_A should be passed in to build CASCI/CASSCF densities.
*
* Usage of this class:  Contructor will read the inputs, then use:
* rhf_rdms() to return RHF rdms.
* cas_rdms(mo_space_info_A) to build and return CASCI/CASSCF densities.
* cas_rdms(mo_space_info_AB) to build and return FCI densities.
* Example (build a CASCI density for A):
*
*     emb = EMBEDDING_DENSITY(state_weights_map, scf_info, mo_space_info_AB, ints, options)
*     rdms = emb.cas_rdms(mo_space_info_A)
*/

class EMBEDDING_DENSITY {

    // ==> Constructors <==

  public:
    EMBEDDING_DENSITY(const std::map<StateInfo, std::vector<double>>& state_weights_map,
                      std::shared_ptr<SCFInfo> scf_info, std::shared_ptr<MOSpaceInfo> mo_space_info,
                      std::shared_ptr<ForteIntegrals> ints, std::shared_ptr<ForteOptions> options);

    // ==> User's interface <==

    /// Return constructed close-shell RDMs (Diag: ....11110000....)
    RDMs rhf_rdms();
    /// Return the computed CAS RDMs
    RDMs cas_rdms(std::shared_ptr<MOSpaceInfo> mo_space_info_active);
    /// return active slices of the computed RDMs
    RDMs rdms_active_slice() const {return rdms_active_; }

  private:
    /// Set up options and inputs
    void start_up();
    /// List of states and weights
    std::map<StateInfo, std::vector<double>> state_weights_map_;
    /// Basic SCF info
    std::shared_ptr<SCFInfo> scf_info_;
    /// MO info of A + B
    std::shared_ptr<MOSpaceInfo> mo_space_info_;
    /// Active indices
    std::vector<size_t> mos_actv_o_;
    /// Integrals (A + B)
    std::shared_ptr<ForteIntegrals> ints_;
    /// Input options
    std::shared_ptr<ForteOptions> options_;
    /// Active slice of the RDMs
    RDMs rdms_active_;
};

} // namespace forte
#endif // EMBEDDING_DENSITY_H
