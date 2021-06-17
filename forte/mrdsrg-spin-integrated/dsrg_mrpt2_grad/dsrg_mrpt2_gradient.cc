/**
 * DSRG-MRPT2 gradient code by Shuhe Wang
 *
 * The computation procedure is listed as belows:
 * (1), Set MOs spaces;
 * (2), Set Tensors (F, H, V etc.);
 * (3), Solve the z-vector equations;
 * (4), Compute and write the Lagrangian;
 * (5), Write 1RDMs and 2RDMs coefficients;
 * (6), Back-transform the TPDM.
 */
#include <algorithm>
#include <map>
#include <vector>
#include <math.h>
#include <numeric>
#include <ctype.h>
#include <string>

#include "psi4/libpsi4util/PsiOutStream.h"
#include "psi4/libpsio/psio.h"
#include "psi4/libpsio/psio.hpp"
#include "psi4/libqt/qt.h"
#include "psi4/libmints/dipole.h"

#include "helpers/timer.h"
#include "ci_rdm/ci_rdms.h"
#include "boost/format.hpp"
#include "sci/fci_mo.h"
#include "fci/fci_solver.h"
#include "helpers/printing.h"
#include "../dsrg_mrpt2.h"

#include "psi4/libmints/factory.h"
#include "psi4/libiwl/iwl.hpp"
#include "psi4/libpsio/psio.hpp"

#include "psi4/libmints/matrix.h"
#include "psi4/libmints/wavefunction.h"
#include "psi4/psifiles.h"

#include "../master_mrdsrg.h"
#include "helpers/timer.h"

using namespace ambit;
using namespace psi;

namespace forte {

SharedMatrix DSRG_MRPT2::compute_gradient() {
    // NOTICE: compute the DSRG_MRPT2 gradient
    print_method_banner({"DSRG-MRPT2 Gradient", "Shuhe Wang"});
    set_global_variables();
    set_tensor();
    set_multiplier();
    write_lagrangian();
    write_1rdm_spin_dependent();
    write_2rdm_spin_dependent();
    tpdm_backtransform();

    outfile->Printf("\n    Computing Gradient .............................. Done\n");
    return std::make_shared<Matrix>("nullptr", 0, 0);
}

void DSRG_MRPT2::set_global_variables() {
    outfile->Printf("\n    Initializing Global Variables ................... ");
    nmo = mo_space_info_->size("CORRELATED");
    core_mos = mo_space_info_->corr_absolute_mo("RESTRICTED_DOCC");
    actv_mos = mo_space_info_->corr_absolute_mo("ACTIVE");
    virt_mos = mo_space_info_->corr_absolute_mo("RESTRICTED_UOCC");
    core_all = mo_space_info_->absolute_mo("RESTRICTED_DOCC");
    actv_all = mo_space_info_->absolute_mo("ACTIVE");
    virt_all = mo_space_info_->absolute_mo("RESTRICTED_UOCC");
    core_mos_relative = mo_space_info_->relative_mo("RESTRICTED_DOCC");
    actv_mos_relative = mo_space_info_->relative_mo("ACTIVE");
    virt_mos_relative = mo_space_info_->relative_mo("RESTRICTED_UOCC");
    irrep_vec = mo_space_info_->dimension("ALL");
    na = mo_space_info_->size("ACTIVE");
    ncore = mo_space_info_->size("RESTRICTED_DOCC");
    nvirt = mo_space_info_->size("RESTRICTED_UOCC");
    nirrep = mo_space_info_->nirrep();
    ndets = ci_vectors_[0].dims()[0];
    Alpha = 0.0;
    outfile->Printf("Done");
}


void DSRG_MRPT2::set_tensor() {
    outfile->Printf("\n    Initializing RDMs, DSRG Tensors and CI integrals. ");
    I = BTF_->build(CoreTensor, "identity matrix", {"cc", "CC", "aa", "AA", "vv", "VV"});
    I.iterate([&](const std::vector<size_t>& i, const std::vector<SpinType>&, double& value) {
        value = (i[0] == i[1]) ? 1.0 : 0.0;
    });

    x_ci = ambit::Tensor::build(ambit::CoreTensor, "solution of ci multipliers", {ndets});

    set_density();
    set_h();
    set_v();
    set_fock();
    set_dsrg_tensor();
    set_ci_ints();
    outfile->Printf("Done");
}

void DSRG_MRPT2::set_multiplier() {
    set_sigma();
    set_xi();
    set_tau();
    set_kappa();
    set_z();
    set_w();
}

} // namespace forte