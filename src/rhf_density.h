#ifndef _rhf_density_h_
#define _rhf_density_h_

#include <vector>

#include "base_classes/scf_info.h"
#include "base_classes/forte_options.h"
#include "base_classes/mo_space_info.h"
#include "base_classes/rdms.h"

namespace forte {
class RHF_DENSITY {
  public:
    RHF_DENSITY(std::shared_ptr<SCFInfo> scf_info, std::shared_ptr<MOSpaceInfo> mo_space_info);

    RDMs rhf_rdms();

  private:
    void start_up();

    std::shared_ptr<SCFInfo> scf_info_;
    std::shared_ptr<MOSpaceInfo> mo_space_info_;

    std::vector<size_t> mos_actv_o_;
};

} // namespace forte
#endif // RHF_DENSITY_H
