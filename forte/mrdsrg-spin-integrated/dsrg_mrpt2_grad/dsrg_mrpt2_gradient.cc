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
#include "psi4/libpsi4util/PsiOutStream.h"
#include "psi4/libmints/matrix.h"
#include "sci/fci_mo.h"
#include "helpers/printing.h"
#include "../dsrg_mrpt2.h"
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
    if (eri_df_) {
        write_df_rdm();
    } else {
        write_2rdm_spin_dependent();
        tpdm_backtransform();
    }

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
    naux = aux_mos_.size();
    nirrep = mo_space_info_->nirrep();
    ndets = ci_vectors_[0].dims()[0];
    Alpha = 0.0;
    orb_label = {{'c', {"c"}},
                 {'C', {"C"}},
                 {'a', {"a"}},
                 {'A', {"A"}},
                 {'v', {"v"}},
                 {'V', {"V"}},
                 {'p', {"a", "v"}},
                 {'P', {"A", "V"}},
                 {'h', {"c", "a"}},
                 {'H', {"C", "A"}}};
    orb_size = {{"c", ncore}, {"a", na}, {"v", nvirt}, {"C", ncore}, {"A", na}, {"V", nvirt}};
    outfile->Printf("Done");
}

void DSRG_MRPT2::set_tensor() {
    outfile->Printf("\n    Initializing RDMs, DSRG Tensors and CI integrals. ");
    I = BTF_->build(CoreTensor, "identity matrix", {"cc", "CC", "aa", "AA", "vv", "VV"});
    I_ci = ambit::Tensor::build(ambit::CoreTensor, "identity", {ndets, ndets});
    one_vec = BTF_->build(CoreTensor, "vector with all components equal 1", {"c", "a", "v"});
    x_ci = ambit::Tensor::build(ambit::CoreTensor, "solution of ci multipliers", {ndets});

    I.iterate([&](const std::vector<size_t>& i, const std::vector<SpinType>&, double& value) {
        value = (i[0] == i[1]) ? 1.0 : 0.0;
    });
    one_vec.iterate([&](const std::vector<size_t>& i, const std::vector<SpinType>&, double& value) {
        value = 1.0;
    });
    I_ci.iterate(
        [&](const std::vector<size_t>& i, double& value) { value = (i[0] == i[1]) ? 1.0 : 0.0; });

    set_density();
    set_h();
    set_v();
    if (eri_df_) { set_j();}
    set_active_fock();
    set_dsrg_tensor();
    set_ci_ints();
    outfile->Printf("Done");
}

void DSRG_MRPT2::set_multiplier() {
    set_sigma_xi();
    set_tau();
    set_kappa();
    set_z();
    set_w();
}

} // namespace forte