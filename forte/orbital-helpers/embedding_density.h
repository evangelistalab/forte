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

/**
* @brief EMBEDDING_DENSITY does an orbital optimization given an 1RDM, 2RDM, and the integrals
* Forms an orbital gradient:  g_{pq} = [ h_{pq} gamma_{pq} + Gamma_{pq}^{rs} <pq||rs> - A(p->q)]
* Here only form a diagonal hessian of orbitals: Look at Hohenstein J. Chem. Phys. 142, 224103.
* Diagonal Hessian only requires (pu|xy) integrals (many are built in JK library)
* Daniel Smith's CASSCF code in PSI4 was integral in debugging.
*
* Usage of this class:  Contructor will just set_up the basic values
* I learn best from examples:  Here is how I use it in the CASSCF code.
* Note:  If you are not freezing core, you do not need F_froze
*         OrbitalOptimizer orbital_optimizer(gamma1_,
                                           gamma2_,
                                           ints_->aptei_ab_block(nmo_abs_,
active_abs_, active_abs_, active_abs_) ,
                                           options_,
                                           mo_space_info_);
        orbital_optimizer.set_one_body(OneBody)
        orbital_optimizer.set_frozen_one_body(F_froze_);
        orbital_optimizer.set_no_symmetry_mo(Call_);
        orbital_optimizer.set_symmmetry_mo(Ca);

        orbital_optimizer.update()
        S = orbital_optimizer.approx_solve()
        C_new = orbital_optimizer.rotate(Ca, S) -> Right now, if this is an
iterative procedure, you should
        use the Ca that was previously updated, ie (C_new  =
Cold(exp(S_previous)) * exp(S))
*/

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
