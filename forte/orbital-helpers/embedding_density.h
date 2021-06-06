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
class EMBEDDING_DENSITY {
  public:
    EMBEDDING_DENSITY(const std::map<StateInfo, std::vector<double>>& state_weights_map, 
                      std::shared_ptr<SCFInfo> scf_info, 
                      std::shared_ptr<MOSpaceInfo> mo_space_info, 
                      std::shared_ptr<ForteIntegrals> ints, 
                      std::shared_ptr<ForteOptions> options);

    RDMs rhf_rdms();

    RDMs cas_rdms(std::shared_ptr<MOSpaceInfo> mo_space_info_active);

  private:
    void start_up();

    std::map<StateInfo, std::vector<double>> state_weights_map_;

    std::shared_ptr<SCFInfo> scf_info_;

    std::shared_ptr<MOSpaceInfo> mo_space_info_;

    std::vector<size_t> mos_actv_o_;

    std::shared_ptr<ForteIntegrals> ints_;

    std::shared_ptr<ForteOptions> options_;
};

} // namespace forte
#endif // EMBEDDING_DENSITY_H
