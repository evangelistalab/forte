#include "base_classes/dynamic_correlation_solver.h"

namespace forte {
DynamicCorrelationSolver::DynamicCorrelationSolver(RDMs rdms,
                                                   std::shared_ptr<SCFInfo> scf_info,
                                                   std::shared_ptr<ForteOptions> options,
                                                   std::shared_ptr<ForteIntegrals> ints,
                                                   std::shared_ptr<MOSpaceInfo> mo_space_info)
    : ints_(ints), mo_space_info_(mo_space_info), rdms_(rdms), scf_info_(scf_info),
      foptions_(options) {
    startup();
}

void DynamicCorrelationSolver::startup() {
    Enuc_ = ints_->nuclear_repulsion_energy();
    Efrzc_ = ints_->frozen_core_energy();

    print_ = foptions_->get_int("PRINT");

    eri_df_ = false;
    ints_type_ = foptions_->get_str("INT_TYPE");
    if (ints_type_ == "CHOLESKY" || ints_type_ == "DF" || ints_type_ == "DISKDF") {
        eri_df_ = true;
    }

    diis_start_ = foptions_->get_int("DSRG_DIIS_START");
    diis_freq_ = foptions_->get_int("DSRG_DIIS_FREQ");
    diis_min_vec_ = foptions_->get_int("DSRG_DIIS_MIN_VEC");
    diis_max_vec_ = foptions_->get_int("DSRG_DIIS_MAX_VEC");
    if (diis_min_vec_ < 1) {
        diis_min_vec_ = 1;
    }
    if (diis_max_vec_ <= diis_min_vec_) {
        diis_max_vec_ = diis_min_vec_ + 4;
    }
    if (diis_freq_ < 1) {
        diis_freq_ = 1;
    }
}

std::shared_ptr<DynamicCorrelationSolver>
make_dynamic_correlation_solver(const std::string& /*type*/, std::shared_ptr<ForteOptions> /*options*/,
                                std::shared_ptr<ForteIntegrals> /*ints*/,
                                std::shared_ptr<MOSpaceInfo> /*mo_space_info*/) {
    // TODO fill and return objects!
    //    return DynamicCorrelationSolver();
    return std::shared_ptr<DynamicCorrelationSolver>();
}

} // namespace forte
