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
#include "dsrg_mrpt2.h"

#include "psi4/libmints/factory.h"
#include "psi4/libiwl/iwl.hpp"
#include "psi4/libpsio/psio.hpp"

#include "gradient_tpdm/backtransform_tpdm.h"
#include "psi4/libmints/matrix.h"
#include "psi4/libmints/wavefunction.h"
#include "psi4/psifiles.h"

#include "master_mrdsrg.h"
#include "helpers/timer.h"


using namespace ambit;
using namespace psi;

namespace forte {

const bool PT2_TERM         = true;
const bool X1_TERM          = true;
const bool X2_TERM          = true;
const bool X3_TERM          = true;
const bool X4_TERM          = true;
const bool X5_TERM          = true;
const bool X6_TERM          = true;
const bool X7_TERM          = true;
const bool CORRELATION_TERM = true;

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

void DSRG_MRPT2::set_ci_ints() {
    cc = coupling_coefficients_;
    ci = ci_vectors_[0];

    cc1a_n = ambit::Tensor::build(ambit::CoreTensor, "cc1a_n", {ndets, na, na});
    cc1a_r = ambit::Tensor::build(ambit::CoreTensor, "cc1a_r", {ndets, na, na});
    cc1b_n = ambit::Tensor::build(ambit::CoreTensor, "cc1b_n", {ndets, na, na});
    cc1b_r = ambit::Tensor::build(ambit::CoreTensor, "cc1b_r", {ndets, na, na});

    cc2aa_n = ambit::Tensor::build(ambit::CoreTensor, "cc2aa_n", {ndets, na, na, na, na});
    cc2aa_r = ambit::Tensor::build(ambit::CoreTensor, "cc2aa_r", {ndets, na, na, na, na});
    cc2bb_n = ambit::Tensor::build(ambit::CoreTensor, "cc2bb_n", {ndets, na, na, na, na});
    cc2bb_r = ambit::Tensor::build(ambit::CoreTensor, "cc2bb_r", {ndets, na, na, na, na});
    cc2ab_n = ambit::Tensor::build(ambit::CoreTensor, "cc2ab_n", {ndets, na, na, na, na});
    cc2ab_r = ambit::Tensor::build(ambit::CoreTensor, "cc2ab_r", {ndets, na, na, na, na});

    cc3aaa_n = ambit::Tensor::build(ambit::CoreTensor, "cc3aaa_n", {ndets, na, na, na, na, na, na});
    cc3aaa_r = ambit::Tensor::build(ambit::CoreTensor, "cc3aaa_r", {ndets, na, na, na, na, na, na});
    cc3bbb_n = ambit::Tensor::build(ambit::CoreTensor, "cc3bbb_n", {ndets, na, na, na, na, na, na});
    cc3bbb_r = ambit::Tensor::build(ambit::CoreTensor, "cc3bbb_r", {ndets, na, na, na, na, na, na});
    cc3aab_n = ambit::Tensor::build(ambit::CoreTensor, "cc3aab_n", {ndets, na, na, na, na, na, na});
    cc3aab_r = ambit::Tensor::build(ambit::CoreTensor, "cc3aab_r", {ndets, na, na, na, na, na, na});
    cc3abb_n = ambit::Tensor::build(ambit::CoreTensor, "cc3abb_n", {ndets, na, na, na, na, na, na});
    cc3abb_r = ambit::Tensor::build(ambit::CoreTensor, "cc3abb_r", {ndets, na, na, na, na, na, na});

    cc1a_n("Kzw") = cc.cc1a()("KJzw") * ci("J");
    cc1a_r("Kzw") = cc.cc1a()("JKzw") * ci("J");
    cc1b_n("KZW") = cc.cc1b()("KJZW") * ci("J");
    cc1b_r("KZW") = cc.cc1b()("JKZW") * ci("J");

    cc2aa_n("Kuvxy") = cc.cc2aa()("KJuvxy") * ci("J");
    cc2aa_r("Kuvxy") = cc.cc2aa()("JKuvxy") * ci("J");
    cc2bb_n("KUVXY") = cc.cc2bb()("KJUVXY") * ci("J");
    cc2bb_r("KUVXY") = cc.cc2bb()("JKUVXY") * ci("J");
    cc2ab_n("KuVxY") = cc.cc2ab()("KJuVxY") * ci("J");
    cc2ab_r("KuVxY") = cc.cc2ab()("JKuVxY") * ci("J");

    cc3aaa_n("Kuvwxyz") = cc.cc3aaa()("KJuvwxyz") * ci("J");
    cc3aaa_r("Kuvwxyz") = cc.cc3aaa()("JKuvwxyz") * ci("J");
    cc3bbb_n("KUVWXYZ") = cc.cc3bbb()("KJUVWXYZ") * ci("J");
    cc3bbb_r("KUVWXYZ") = cc.cc3bbb()("JKUVWXYZ") * ci("J");
    cc3aab_n("KuvWxyZ") = cc.cc3aab()("KJuvWxyZ") * ci("J");
    cc3aab_r("KuvWxyZ") = cc.cc3aab()("JKuvWxyZ") * ci("J");
    cc3abb_n("KuVWxYZ") = cc.cc3abb()("KJuVWxYZ") * ci("J");
    cc3abb_r("KuVWxYZ") = cc.cc3abb()("JKuVWxYZ") * ci("J");

    dlamb_aa = ambit::Tensor::build(ambit::CoreTensor, "derivatives of Lambda2_ w.r.t. C_K alpha-alpha", {ndets, na, na, na, na});
    dlamb_bb = ambit::Tensor::build(ambit::CoreTensor, "derivatives of Lambda2_ w.r.t. C_K beta-beta", {ndets, na, na, na, na});
    dlamb_ab = ambit::Tensor::build(ambit::CoreTensor, "derivatives of Lambda2_ w.r.t. C_K alpha-beta", {ndets, na, na, na, na});

    // alpha-alpha
    dlamb_aa("Kxyuv") += cc2aa_n("Kxyuv");
    dlamb_aa("Kxyuv") += cc2aa_r("Kxyuv");

    dlamb_aa("Kxyuv") -= cc1a_n("Kxu") * Gamma1_.block("aa")("yv");
    dlamb_aa("Kxyuv") -= cc1a_n("Kux") * Gamma1_.block("aa")("yv");
    dlamb_aa("Kxyuv") -= cc1a_n("Kyv") * Gamma1_.block("aa")("xu");
    dlamb_aa("Kxyuv") -= cc1a_n("Kvy") * Gamma1_.block("aa")("xu");
    dlamb_aa("Kxyuv") += cc1a_n("Kxv") * Gamma1_.block("aa")("yu");
    dlamb_aa("Kxyuv") += cc1a_n("Kvx") * Gamma1_.block("aa")("yu");
    dlamb_aa("Kxyuv") += cc1a_n("Kyu") * Gamma1_.block("aa")("xv");
    dlamb_aa("Kxyuv") += cc1a_n("Kuy") * Gamma1_.block("aa")("xv");

    // beta-beta
    dlamb_bb("KXYUV") += cc2bb_n("KXYUV");
    dlamb_bb("KXYUV") += cc2bb_r("KXYUV");
    dlamb_bb("KXYUV") -= cc1b_n("KXU") * Gamma1_.block("AA")("YV");
    dlamb_bb("KXYUV") -= cc1b_n("KUX") * Gamma1_.block("AA")("YV");
    dlamb_bb("KXYUV") -= cc1b_n("KYV") * Gamma1_.block("AA")("XU");
    dlamb_bb("KXYUV") -= cc1b_n("KVY") * Gamma1_.block("AA")("XU");
    dlamb_bb("KXYUV") += cc1b_n("KXV") * Gamma1_.block("AA")("YU");
    dlamb_bb("KXYUV") += cc1b_n("KVX") * Gamma1_.block("AA")("YU");
    dlamb_bb("KXYUV") += cc1b_n("KYU") * Gamma1_.block("AA")("XV");
    dlamb_bb("KXYUV") += cc1b_n("KUY") * Gamma1_.block("AA")("XV");

    // alpha-beta
    dlamb_ab("KxYuV") += cc2ab_n("KxYuV");
    dlamb_ab("KxYuV") += cc2ab_r("KxYuV");
    dlamb_ab("KxYuV") -= cc1a_n("Kxu") * Gamma1_.block("AA")("YV");
    dlamb_ab("KxYuV") -= cc1a_n("Kux") * Gamma1_.block("AA")("YV");
    dlamb_ab("KxYuV") -= cc1b_n("KYV") * Gamma1_.block("aa")("xu");
    dlamb_ab("KxYuV") -= cc1b_n("KVY") * Gamma1_.block("aa")("xu");

    dlamb3_aaa = ambit::Tensor::build(ambit::CoreTensor, "derivatives of Lambda3_ w.r.t. C_K alpha-alpha-alpha", {ndets, na, na, na, na, na, na});
    dlamb3_bbb = ambit::Tensor::build(ambit::CoreTensor, "derivatives of Lambda3_ w.r.t. C_K beta-beta-beta", {ndets, na, na, na, na, na, na});
    dlamb3_aab = ambit::Tensor::build(ambit::CoreTensor, "derivatives of Lambda3_ w.r.t. C_K alpha-alpha-beta", {ndets, na, na, na, na, na, na});
    dlamb3_abb = ambit::Tensor::build(ambit::CoreTensor, "derivatives of Lambda3_ w.r.t. C_K alpha-beta-beta", {ndets, na, na, na, na, na, na});

    // alpha-alpha-alpha
    dlamb3_aaa("Kxyzuvw") += cc3aaa_n("Kxyzuvw");
    dlamb3_aaa("Kxyzuvw") += cc3aaa_r("Kxyzuvw");
    dlamb3_aaa("Kxyzuvw") -= 2.0 * cc1a_n("Kuz") * Gamma2_.block("aaaa")("xyvw");
    dlamb3_aaa("Kxyzuvw") -= 2.0 * cc1a_r("Kuz") * Gamma2_.block("aaaa")("xyvw");
    dlamb3_aaa("Kxyzuvw") -= 2.0 * cc2aa_n("Kxyvw") * Gamma1_.block("aa")("uz");
    dlamb3_aaa("Kxyzuvw") -= 2.0 * cc2aa_r("Kxyvw") * Gamma1_.block("aa")("uz");
    dlamb3_aaa("Kxyzuvw") -= cc1a_n("Kwz") * Gamma2_.block("aaaa")("xyuv");
    dlamb3_aaa("Kxyzuvw") -= cc1a_r("Kwz") * Gamma2_.block("aaaa")("xyuv");
    dlamb3_aaa("Kxyzuvw") -= cc2aa_n("Kxyuv") * Gamma1_.block("aa")("wz");
    dlamb3_aaa("Kxyzuvw") -= cc2aa_r("Kxyuv") * Gamma1_.block("aa")("wz");
    dlamb3_aaa("Kxyzuvw") += 4.0 * cc1a_n("Kxu") * Gamma2_.block("aaaa")("vwzy");
    dlamb3_aaa("Kxyzuvw") += 4.0 * cc1a_r("Kxu") * Gamma2_.block("aaaa")("vwzy");
    dlamb3_aaa("Kxyzuvw") += 4.0 * cc2aa_n("Kvwzy") * Gamma1_.block("aa")("xu");
    dlamb3_aaa("Kxyzuvw") += 4.0 * cc2aa_r("Kvwzy") * Gamma1_.block("aa")("xu");
    dlamb3_aaa("Kxyzuvw") += 2.0 * cc1a_n("Kxw") * Gamma2_.block("aaaa")("uvzy");
    dlamb3_aaa("Kxyzuvw") += 2.0 * cc1a_r("Kxw") * Gamma2_.block("aaaa")("uvzy");
    dlamb3_aaa("Kxyzuvw") += 2.0 * cc2aa_n("Kuvzy") * Gamma1_.block("aa")("xw");
    dlamb3_aaa("Kxyzuvw") += 2.0 * cc2aa_r("Kuvzy") * Gamma1_.block("aa")("xw");
    dlamb3_aaa("Kxyzuvw") += 8.0 * cc1a_n("Kuz") * Gamma1_.block("aa")("xv")* Gamma1_.block("aa")("yw");
    dlamb3_aaa("Kxyzuvw") += 8.0 * cc1a_r("Kuz") * Gamma1_.block("aa")("xv")* Gamma1_.block("aa")("yw");
    dlamb3_aaa("Kxyzuvw") += 8.0 * cc1a_n("Kxv") * Gamma1_.block("aa")("uz")* Gamma1_.block("aa")("yw");
    dlamb3_aaa("Kxyzuvw") += 8.0 * cc1a_r("Kxv") * Gamma1_.block("aa")("uz")* Gamma1_.block("aa")("yw");
    dlamb3_aaa("Kxyzuvw") += 8.0 * cc1a_n("Kyw") * Gamma1_.block("aa")("uz")* Gamma1_.block("aa")("xv");
    dlamb3_aaa("Kxyzuvw") += 8.0 * cc1a_r("Kyw") * Gamma1_.block("aa")("uz")* Gamma1_.block("aa")("xv");
    dlamb3_aaa("Kxyzuvw") += 4.0 * cc1a_n("Kwz") * Gamma1_.block("aa")("xu")* Gamma1_.block("aa")("yv");
    dlamb3_aaa("Kxyzuvw") += 4.0 * cc1a_r("Kwz") * Gamma1_.block("aa")("xu")* Gamma1_.block("aa")("yv");
    dlamb3_aaa("Kxyzuvw") += 4.0 * cc1a_n("Kxu") * Gamma1_.block("aa")("wz")* Gamma1_.block("aa")("yv");
    dlamb3_aaa("Kxyzuvw") += 4.0 * cc1a_r("Kxu") * Gamma1_.block("aa")("wz")* Gamma1_.block("aa")("yv");
    dlamb3_aaa("Kxyzuvw") += 4.0 * cc1a_n("Kyv") * Gamma1_.block("aa")("wz")* Gamma1_.block("aa")("xu");
    dlamb3_aaa("Kxyzuvw") += 4.0 * cc1a_r("Kyv") * Gamma1_.block("aa")("wz")* Gamma1_.block("aa")("xu");

    // beta-beta-beta
    dlamb3_bbb("KXYZUVW") += cc3bbb_n("KXYZUVW");
    dlamb3_bbb("KXYZUVW") += cc3bbb_r("KXYZUVW");
    dlamb3_bbb("KXYZUVW") -= 2.0 * cc1b_n("KUZ") * Gamma2_.block("AAAA")("XYVW");
    dlamb3_bbb("KXYZUVW") -= 2.0 * cc1b_r("KUZ") * Gamma2_.block("AAAA")("XYVW");
    dlamb3_bbb("KXYZUVW") -= 2.0 * cc2bb_n("KXYVW") * Gamma1_.block("AA")("UZ");
    dlamb3_bbb("KXYZUVW") -= 2.0 * cc2bb_r("KXYVW") * Gamma1_.block("AA")("UZ");
    dlamb3_bbb("KXYZUVW") -= cc1b_n("KWZ") * Gamma2_.block("AAAA")("XYUV");
    dlamb3_bbb("KXYZUVW") -= cc1b_r("KWZ") * Gamma2_.block("AAAA")("XYUV");
    dlamb3_bbb("KXYZUVW") -= cc2bb_n("KXYUV") * Gamma1_.block("AA")("WZ");
    dlamb3_bbb("KXYZUVW") -= cc2bb_r("KXYUV") * Gamma1_.block("AA")("WZ");
    dlamb3_bbb("KXYZUVW") += 4.0 * cc1b_n("KXU") * Gamma2_.block("AAAA")("VWZY");
    dlamb3_bbb("KXYZUVW") += 4.0 * cc1b_r("KXU") * Gamma2_.block("AAAA")("VWZY");
    dlamb3_bbb("KXYZUVW") += 4.0 * cc2bb_n("KVWZY") * Gamma1_.block("AA")("XU");
    dlamb3_bbb("KXYZUVW") += 4.0 * cc2bb_r("KVWZY") * Gamma1_.block("AA")("XU");
    dlamb3_bbb("KXYZUVW") += 2.0 * cc1b_n("KXW") * Gamma2_.block("AAAA")("UVZY");
    dlamb3_bbb("KXYZUVW") += 2.0 * cc1b_r("KXW") * Gamma2_.block("AAAA")("UVZY");
    dlamb3_bbb("KXYZUVW") += 2.0 * cc2bb_n("KUVZY") * Gamma1_.block("AA")("XW");
    dlamb3_bbb("KXYZUVW") += 2.0 * cc2bb_r("KUVZY") * Gamma1_.block("AA")("XW");
    dlamb3_bbb("KXYZUVW") += 8.0 * cc1b_n("KUZ") * Gamma1_.block("AA")("XV")* Gamma1_.block("AA")("YW");
    dlamb3_bbb("KXYZUVW") += 8.0 * cc1b_r("KUZ") * Gamma1_.block("AA")("XV")* Gamma1_.block("AA")("YW");
    dlamb3_bbb("KXYZUVW") += 8.0 * cc1b_n("KXV") * Gamma1_.block("AA")("UZ")* Gamma1_.block("AA")("YW");
    dlamb3_bbb("KXYZUVW") += 8.0 * cc1b_r("KXV") * Gamma1_.block("AA")("UZ")* Gamma1_.block("AA")("YW");
    dlamb3_bbb("KXYZUVW") += 8.0 * cc1b_n("KYW") * Gamma1_.block("AA")("UZ")* Gamma1_.block("AA")("XV");
    dlamb3_bbb("KXYZUVW") += 8.0 * cc1b_r("KYW") * Gamma1_.block("AA")("UZ")* Gamma1_.block("AA")("XV");
    dlamb3_bbb("KXYZUVW") += 4.0 * cc1b_n("KWZ") * Gamma1_.block("AA")("XU")* Gamma1_.block("AA")("YV");
    dlamb3_bbb("KXYZUVW") += 4.0 * cc1b_r("KWZ") * Gamma1_.block("AA")("XU")* Gamma1_.block("AA")("YV");
    dlamb3_bbb("KXYZUVW") += 4.0 * cc1b_n("KXU") * Gamma1_.block("AA")("WZ")* Gamma1_.block("AA")("YV");
    dlamb3_bbb("KXYZUVW") += 4.0 * cc1b_r("KXU") * Gamma1_.block("AA")("WZ")* Gamma1_.block("AA")("YV");
    dlamb3_bbb("KXYZUVW") += 4.0 * cc1b_n("KYV") * Gamma1_.block("AA")("WZ")* Gamma1_.block("AA")("XU");
    dlamb3_bbb("KXYZUVW") += 4.0 * cc1b_r("KYV") * Gamma1_.block("AA")("WZ")* Gamma1_.block("AA")("XU");

    // alpha-alpha-beta
    dlamb3_aab("KxyZuvW") += cc3aab_n("KxyZuvW");
    dlamb3_aab("KxyZuvW") += cc3aab_r("KxyZuvW");
    dlamb3_aab("KxyZuvW") -= cc1a_n("Kxu") * Lambda2_.block("aAaA")("yZvW");
    dlamb3_aab("KxyZuvW") -= cc1a_r("Kxu") * Lambda2_.block("aAaA")("yZvW");
    dlamb3_aab("KxyZuvW") -= Gamma1_.block("aa")("xu") * dlamb_ab("KyZvW");
    dlamb3_aab("KxyZuvW") += cc1a_n("Kxv") * Lambda2_.block("aAaA")("yZuW");
    dlamb3_aab("KxyZuvW") += cc1a_r("Kxv") * Lambda2_.block("aAaA")("yZuW");
    dlamb3_aab("KxyZuvW") += Gamma1_.block("aa")("xv") * dlamb_ab("KyZuW");
    dlamb3_aab("KxyZuvW") += cc1a_n("Kyu") * Lambda2_.block("aAaA")("xZvW");
    dlamb3_aab("KxyZuvW") += cc1a_r("Kyu") * Lambda2_.block("aAaA")("xZvW");
    dlamb3_aab("KxyZuvW") += Gamma1_.block("aa")("yu") * dlamb_ab("KxZvW");
    dlamb3_aab("KxyZuvW") -= cc1a_n("Kyv") * Lambda2_.block("aAaA")("xZuW");
    dlamb3_aab("KxyZuvW") -= cc1a_r("Kyv") * Lambda2_.block("aAaA")("xZuW");
    dlamb3_aab("KxyZuvW") -= Gamma1_.block("aa")("yv") * dlamb_ab("KxZuW");
    dlamb3_aab("KxyZuvW") -= cc1b_n("KZW") * Lambda2_.block("aaaa")("xyuv");
    dlamb3_aab("KxyZuvW") -= cc1b_r("KZW") * Lambda2_.block("aaaa")("xyuv");
    dlamb3_aab("KxyZuvW") -= Gamma1_.block("AA")("ZW") * dlamb_aa("Kxyuv");
    dlamb3_aab("KxyZuvW") -= cc1a_n("Kxu") * Gamma1_.block("aa")("yv") * Gamma1_.block("AA")("ZW");
    dlamb3_aab("KxyZuvW") -= cc1a_r("Kxu") * Gamma1_.block("aa")("yv") * Gamma1_.block("AA")("ZW");
    dlamb3_aab("KxyZuvW") -= cc1a_n("Kyv") * Gamma1_.block("aa")("xu") * Gamma1_.block("AA")("ZW");
    dlamb3_aab("KxyZuvW") -= cc1a_r("Kyv") * Gamma1_.block("aa")("xu") * Gamma1_.block("AA")("ZW");
    dlamb3_aab("KxyZuvW") -= cc1b_n("KZW") * Gamma1_.block("aa")("yv") * Gamma1_.block("aa")("xu");
    dlamb3_aab("KxyZuvW") -= cc1b_r("KZW") * Gamma1_.block("aa")("yv") * Gamma1_.block("aa")("xu");
    dlamb3_aab("KxyZuvW") += cc1a_n("Kxv") * Gamma1_.block("aa")("yu") * Gamma1_.block("AA")("ZW");
    dlamb3_aab("KxyZuvW") += cc1a_r("Kxv") * Gamma1_.block("aa")("yu") * Gamma1_.block("AA")("ZW");
    dlamb3_aab("KxyZuvW") += cc1a_n("Kyu") * Gamma1_.block("aa")("xv") * Gamma1_.block("AA")("ZW");
    dlamb3_aab("KxyZuvW") += cc1a_r("Kyu") * Gamma1_.block("aa")("xv") * Gamma1_.block("AA")("ZW");
    dlamb3_aab("KxyZuvW") += cc1b_n("KZW") * Gamma1_.block("aa")("yu") * Gamma1_.block("aa")("xv");
    dlamb3_aab("KxyZuvW") += cc1b_r("KZW") * Gamma1_.block("aa")("yu") * Gamma1_.block("aa")("xv");

    // alpha-beta-beta
    dlamb3_abb("KxYZuVW") += cc3abb_n("KxYZuVW");
    dlamb3_abb("KxYZuVW") += cc3abb_r("KxYZuVW");
    dlamb3_abb("KxYZuVW") -= cc1a_n("Kxu") * Lambda2_.block("AAAA")("YZVW");
    dlamb3_abb("KxYZuVW") -= cc1a_r("Kxu") * Lambda2_.block("AAAA")("YZVW");
    dlamb3_abb("KxYZuVW") -= Gamma1_.block("aa")("xu") * dlamb_bb("KYZVW");
    dlamb3_abb("KxYZuVW") -= cc1b_n("KYV") * Lambda2_.block("aAaA")("xZuW");
    dlamb3_abb("KxYZuVW") -= cc1b_r("KYV") * Lambda2_.block("aAaA")("xZuW");
    dlamb3_abb("KxYZuVW") -= Gamma1_.block("AA")("YV") * dlamb_ab("KxZuW");
    dlamb3_abb("KxYZuVW") += cc1b_n("KYW") * Lambda2_.block("aAaA")("xZuV");
    dlamb3_abb("KxYZuVW") += cc1b_r("KYW") * Lambda2_.block("aAaA")("xZuV");
    dlamb3_abb("KxYZuVW") += Gamma1_.block("AA")("YW") * dlamb_ab("KxZuV");
    dlamb3_abb("KxYZuVW") += cc1b_n("KZV") * Lambda2_.block("aAaA")("xYuW");
    dlamb3_abb("KxYZuVW") += cc1b_r("KZV") * Lambda2_.block("aAaA")("xYuW");
    dlamb3_abb("KxYZuVW") += Gamma1_.block("AA")("ZV") * dlamb_ab("KxYuW");
    dlamb3_abb("KxYZuVW") -= cc1b_n("KZW") * Lambda2_.block("aAaA")("xYuV");
    dlamb3_abb("KxYZuVW") -= cc1b_r("KZW") * Lambda2_.block("aAaA")("xYuV");
    dlamb3_abb("KxYZuVW") -= Gamma1_.block("AA")("ZW") * dlamb_ab("KxYuV");
    dlamb3_abb("KxYZuVW") -= cc1a_n("Kxu") * Gamma1_.block("AA")("YV") * Gamma1_.block("AA")("ZW");
    dlamb3_abb("KxYZuVW") -= cc1a_r("Kxu") * Gamma1_.block("AA")("YV") * Gamma1_.block("AA")("ZW");
    dlamb3_abb("KxYZuVW") -= cc1b_n("KYV") * Gamma1_.block("aa")("xu") * Gamma1_.block("AA")("ZW");
    dlamb3_abb("KxYZuVW") -= cc1b_r("KYV") * Gamma1_.block("aa")("xu") * Gamma1_.block("AA")("ZW");
    dlamb3_abb("KxYZuVW") -= cc1b_n("KZW") * Gamma1_.block("AA")("YV") * Gamma1_.block("aa")("xu");
    dlamb3_abb("KxYZuVW") -= cc1b_r("KZW") * Gamma1_.block("AA")("YV") * Gamma1_.block("aa")("xu");
    dlamb3_abb("KxYZuVW") += cc1a_n("Kxu") * Gamma1_.block("AA")("ZV") * Gamma1_.block("AA")("YW");
    dlamb3_abb("KxYZuVW") += cc1a_r("Kxu") * Gamma1_.block("AA")("ZV") * Gamma1_.block("AA")("YW");
    dlamb3_abb("KxYZuVW") += cc1b_n("KZV") * Gamma1_.block("aa")("xu") * Gamma1_.block("AA")("YW");
    dlamb3_abb("KxYZuVW") += cc1b_r("KZV") * Gamma1_.block("aa")("xu") * Gamma1_.block("AA")("YW");
    dlamb3_abb("KxYZuVW") += cc1b_n("KYW") * Gamma1_.block("AA")("ZV") * Gamma1_.block("aa")("xu");
    dlamb3_abb("KxYZuVW") += cc1b_r("KYW") * Gamma1_.block("AA")("ZV") * Gamma1_.block("aa")("xu");
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

void DSRG_MRPT2::set_density() {
    Gamma2_ = BTF_->build(CoreTensor, "Gamma2_", spin_cases({"aaaa"}));

    Gamma2_.block("aaaa")("pqrs") = rdms_.g2aa()("pqrs");
    Gamma2_.block("aAaA")("pqrs") = rdms_.g2ab()("pqrs");
    Gamma2_.block("AAAA")("pqrs") = rdms_.g2bb()("pqrs");
}

void DSRG_MRPT2::set_h() {
    H = BTF_->build(CoreTensor, "One-Electron Integral", spin_cases({"gg"}));
    H.iterate([&](const std::vector<size_t>& i, const std::vector<SpinType>& spin, double& value) {
        if (spin[0] == AlphaSpin) {
            value = ints_->oei_a(i[0], i[1]);
        } else {
            value = ints_->oei_b(i[0], i[1]);
        }
    });
}

void DSRG_MRPT2::set_v() {
    V = BTF_->build(CoreTensor, "Electron Repulsion Integral", 
        spin_cases({"gphh", "pghh", "ppgh", "pphg",
                    "gchc", "pghc", "pcgc", "pchg",
                    "gcpc", "hgpc", "hcgc", "hcpg", 
                    "gccc", "cgcc", "ccgc", "cccg", 
                    "gcvc", "vgvc", "vcgc", "vcvg",
                    "cgch", "gpch", "cpcg", "cpgh",
                    "cgcp", "ghcp", "chcg", "chgp", 
                    "cgcv", "gvcv", "cvcg", "cvgv"}));

    V.iterate([&](const std::vector<size_t>& i, const std::vector<SpinType>& spin, double& value) {
        if (spin[0] == AlphaSpin) {
            if (spin[1] == AlphaSpin) {
                value = ints_->aptei_aa(i[0], i[1], i[2], i[3]);
            } 
            else {
                value = ints_->aptei_ab(i[0], i[1], i[2], i[3]);
            }
        } 
        else if (spin[1] == BetaSpin) {
            value = ints_->aptei_bb(i[0], i[1], i[2], i[3]);
        }
    });

    V_N_Alpha = BTF_->build(CoreTensor, "normal Dimention-reduced Electron Repulsion Integral alpha", {"gg"});
    V_N_Beta = BTF_->build(CoreTensor, "normal Dimention-reduced Electron Repulsion Integral beta", {"gg"});
    V_R_Beta = BTF_->build(CoreTensor, "index-reversed Dimention-reduced Electron Repulsion Integral beta", {"GG"});
    V_all_Beta = BTF_->build(CoreTensor, "normal Dimention-reduced Electron Repulsion Integral all beta", {"GG"});

    // Summation of V["pmqm"] over index "m" or V["mpmq"] over index "m"
    V_N_Alpha["pq"] = V["pmqn"] * I["mn"];
    // Summation of V["pMqM"] over index "M"
    V_N_Beta["pq"] = V["pMqN"] * I["MN"];
    // Summation of V["mPmQ"] over index "m"
    V_R_Beta["PQ"] = V["mPnQ"] * I["mn"];
    // Summation of V["PMQM"] over index "M"
    V_all_Beta["PQ"] = V["PMQN"] * I["MN"];
}

void DSRG_MRPT2::set_fock() {
    F = BTF_->build(CoreTensor, "Fock Matrix", spin_cases({"gg"}));

    ints_->make_fock_matrix(Gamma1_.block("aa"), Gamma1_.block("AA"));

    F.iterate([&](const std::vector<size_t>& i, const std::vector<SpinType>& spin, double& value) {
        if (spin[0] == AlphaSpin) {
            value = ints_->get_fock_a(i[0], i[1]);
        } else {
            value = ints_->get_fock_b(i[0], i[1]);
        }
    });
}

void DSRG_MRPT2::set_dsrg_tensor() {
    Eeps1       = BTF_->build(CoreTensor, "e^[-s*(Delta1)^2]", spin_cases({"hp"}));
    Eeps1_m1    = BTF_->build(CoreTensor, "{1-e^[-s*(Delta1)^2]}/(Delta1)", spin_cases({"hp"}));
    Eeps1_m2    = BTF_->build(CoreTensor, "{1-e^[-s*(Delta1)^2]}/(Delta1)^2", spin_cases({"hp"}));
    Eeps2       = BTF_->build(CoreTensor, "e^[-s*(Delta2)^2]", spin_cases({"hhpp"}));
    Eeps2_p     = BTF_->build(CoreTensor, "1+e^[-s*(Delta2)^2]", spin_cases({"hhpp"}));
    Eeps2_m1    = BTF_->build(CoreTensor, "{1-e^[-s*(Delta2)^2]}/(Delta2)", spin_cases({"hhpp"}));
    Eeps2_m2    = BTF_->build(CoreTensor, "{1-e^[-s*(Delta2)^2]}/(Delta2)^2", spin_cases({"hhpp"}));
    Delta1      = BTF_->build(CoreTensor, "Delta1", spin_cases({"gg"}));
    Delta2      = BTF_->build(CoreTensor, "Delta2", spin_cases({"hhpp"}));
    DelGam1     = BTF_->build(CoreTensor, "Delta1 * Gamma1_", spin_cases({"aa"}));
    DelEeps1    = BTF_->build(CoreTensor, "Delta1 * Eeps1", spin_cases({"hp"}));
    T2OverDelta = BTF_->build(CoreTensor, "T2/Delta", spin_cases({"hhpp"}));

    Eeps1.iterate(
        [&](const std::vector<size_t>& i, const std::vector<SpinType>& spin, double& value) {
            if (spin[0] == AlphaSpin) { value = dsrg_source_->compute_renormalized(Fa_[i[0]] - Fa_[i[1]]);}
            else { value = dsrg_source_->compute_renormalized(Fb_[i[0]] - Fb_[i[1]]);}
        }
    );
    Delta1.iterate(
        [&](const std::vector<size_t>& i, const std::vector<SpinType>& spin, double& value) {
            if (spin[0] == AlphaSpin) { value = Fa_[i[0]] - Fa_[i[1]];}
            else { value = Fb_[i[0]] - Fb_[i[1]];}
        }
    );
    Delta2.iterate(
        [&](const std::vector<size_t>& i, const std::vector<SpinType>& spin, double& value) {
            if (spin[0] == AlphaSpin&& spin[1] == AlphaSpin) 
                { value = Fa_[i[0]] + Fa_[i[1]] - Fa_[i[2]] - Fa_[i[3]];}
            else if (spin[0] == BetaSpin&& spin[1] == BetaSpin)
                { value = Fb_[i[0]] + Fb_[i[1]] - Fb_[i[2]] - Fb_[i[3]];}
            else { value = Fa_[i[0]] + Fb_[i[1]] - Fa_[i[2]] - Fb_[i[3]];}
        }
    );
    Eeps2.iterate(
        [&](const std::vector<size_t>& i, const std::vector<SpinType>& spin, double& value) {
            if (spin[0] == AlphaSpin&& spin[1] == AlphaSpin) 
                { value = dsrg_source_->compute_renormalized(Fa_[i[0]] + Fa_[i[1]] - Fa_[i[2]] - Fa_[i[3]]);}
            else if (spin[0] == BetaSpin&& spin[1] == BetaSpin)
                { value = dsrg_source_->compute_renormalized(Fb_[i[0]] + Fb_[i[1]] - Fb_[i[2]] - Fb_[i[3]]);}
            else { value = dsrg_source_->compute_renormalized(Fa_[i[0]] + Fb_[i[1]] - Fa_[i[2]] - Fb_[i[3]]);}
        }
    );
    Eeps2_p.iterate(
        [&](const std::vector<size_t>& i, const std::vector<SpinType>& spin, double& value) {
            if (spin[0] == AlphaSpin&& spin[1] == AlphaSpin) 
                { value = 1.0 + dsrg_source_->compute_renormalized(Fa_[i[0]] + Fa_[i[1]] - Fa_[i[2]] - Fa_[i[3]]);}
            else if (spin[0] == BetaSpin&& spin[1] == BetaSpin)
                { value = 1.0 + dsrg_source_->compute_renormalized(Fb_[i[0]] + Fb_[i[1]] - Fb_[i[2]] - Fb_[i[3]]);}
            else { value = 1.0 + dsrg_source_->compute_renormalized(Fa_[i[0]] + Fb_[i[1]] - Fa_[i[2]] - Fb_[i[3]]);}
        }
    );
    Eeps2_m1.iterate(
        [&](const std::vector<size_t>& i, const std::vector<SpinType>& spin, double& value) {
            if (spin[0] == AlphaSpin&& spin[1] == AlphaSpin) 
                { value = dsrg_source_->compute_renormalized_denominator_deriv(Fa_[i[0]] + Fa_[i[1]] - Fa_[i[2]] - Fa_[i[3]], 1);}
            else if (spin[0] == BetaSpin&& spin[1] == BetaSpin)
                { value = dsrg_source_->compute_renormalized_denominator_deriv(Fb_[i[0]] + Fb_[i[1]] - Fb_[i[2]] - Fb_[i[3]], 1);}
            else { value = dsrg_source_->compute_renormalized_denominator_deriv(Fa_[i[0]] + Fb_[i[1]] - Fa_[i[2]] - Fb_[i[3]], 1);}
        }
    );
    Eeps2_m2.iterate(
        [&](const std::vector<size_t>& i, const std::vector<SpinType>& spin, double& value) {
            if (spin[0] == AlphaSpin&& spin[1] == AlphaSpin) 
                { value = dsrg_source_->compute_renormalized_denominator_deriv(Fa_[i[0]] + Fa_[i[1]] - Fa_[i[2]] - Fa_[i[3]], 2);}
            else if (spin[0] == BetaSpin&& spin[1] == BetaSpin)
                { value = dsrg_source_->compute_renormalized_denominator_deriv(Fb_[i[0]] + Fb_[i[1]] - Fb_[i[2]] - Fb_[i[3]], 2);}
            else { value = dsrg_source_->compute_renormalized_denominator_deriv(Fa_[i[0]] + Fb_[i[1]] - Fa_[i[2]] - Fb_[i[3]], 2);}
        }
    ); 

    Eeps1_m1.iterate(
        [&](const std::vector<size_t>& i, const std::vector<SpinType>& spin, double& value) {
            if (spin[0] == AlphaSpin) { value = dsrg_source_->compute_renormalized_denominator_deriv(Fa_[i[0]] - Fa_[i[1]], 1);}
            else { value = dsrg_source_->compute_renormalized_denominator_deriv(Fb_[i[0]] - Fb_[i[1]], 1);}
        }
    );

    Eeps1_m2.iterate(
        [&](const std::vector<size_t>& i, const std::vector<SpinType>& spin, double& value) {
            if (spin[0] == AlphaSpin) { value = dsrg_source_->compute_renormalized_denominator_deriv(Fa_[i[0]] - Fa_[i[1]], 2);}
            else { value = dsrg_source_->compute_renormalized_denominator_deriv(Fb_[i[0]] - Fb_[i[1]], 2);}
        }
    );
    
    // An intermediate tensor : T2 / Delta
    T2OverDelta["ijab"] += V["abij"] * Eeps2_m2["ijab"];
    T2OverDelta["iJaB"] += V["aBiJ"] * Eeps2_m2["iJaB"];   

    // Delta1 * Gamma1_
    DelGam1["xu"] = Delta1["xu"] * Gamma1_["xu"];
    DelGam1["XU"] = Delta1["XU"] * Gamma1_["XU"];

    // Delta1 * Eeps1
    DelEeps1["ia"] = Delta1["ia"] * Eeps1["ia"];
    DelEeps1["IA"] = Delta1["IA"] * Eeps1["IA"];
}

void DSRG_MRPT2::set_multiplier() {
    set_sigma();
    set_xi();
    set_tau();
    set_kappa();
    set_z();
    set_w();   
}

void DSRG_MRPT2::set_tau() {
    outfile->Printf("\n    Initializing Tau ................................ ");  
    Tau1 = BTF_->build(CoreTensor, "Tau1", spin_cases({"hhpp"}));
    Tau2 = BTF_->build(CoreTensor, "Tau2", spin_cases({"hhpp"}));
    // Tau * Delta
    // <[V, T2]> (C_2)^4
    if (PT2_TERM) {
        Tau2["mnef"] += 0.25 * V_["efmn"];
        Tau2["mneu"] += 0.25 * V_["evmn"] * Eta1_["uv"];
        Tau2["mnue"] += 0.25 * V_["vemn"] * Eta1_["uv"];
        Tau2["mnux"] += 0.25 * V_["vymn"] * Eta1_["uv"] * Eta1_["xy"];
        Tau2["mvef"] += 0.25 * V_["efmu"] * Gamma1_["uv"];
        Tau2["mvez"] += 0.25 * V_["ewmu"] * Gamma1_["uv"] * Eta1_["zw"];
        Tau2["mvze"] += 0.25 * V_["wemu"] * Gamma1_["uv"] * Eta1_["zw"];
        Tau2["mvzx"] += 0.25 * V_["wymu"] * Gamma1_["uv"] * Eta1_["zw"] * Eta1_["xy"];
        Tau2["vmef"] += 0.25 * V_["efum"] * Gamma1_["uv"];
        Tau2["vmez"] += 0.25 * V_["ewum"] * Gamma1_["uv"] * Eta1_["zw"];
        Tau2["vmze"] += 0.25 * V_["weum"] * Gamma1_["uv"] * Eta1_["zw"];
        Tau2["vmzx"] += 0.25 * V_["wyum"] * Gamma1_["uv"] * Eta1_["zw"] * Eta1_["xy"];
        Tau2["vyef"] += 0.25 * V_["efux"] * Gamma1_["uv"] * Gamma1_["xy"];
        Tau2["vyez"] += 0.25 * V_["ewux"] * Gamma1_["uv"] * Gamma1_["xy"] * Eta1_["zw"];
        Tau2["vyze"] += 0.25 * V_["weux"] * Gamma1_["uv"] * Gamma1_["xy"] * Eta1_["zw"];
        Tau2["v,y,z,u1"] += 0.25 * V_["w,a1,u,x"] * Gamma1_["uv"] * Gamma1_["xy"] * Eta1_["zw"] * Eta1_["u1,a1"];

        Tau2["MNEF"] += 0.25 * V_["EFMN"];
        Tau2["MNEU"] += 0.25 * V_["EVMN"] * Eta1_["UV"];
        Tau2["MNUE"] += 0.25 * V_["VEMN"] * Eta1_["UV"];
        Tau2["MNUX"] += 0.25 * V_["VYMN"] * Eta1_["UV"] * Eta1_["XY"];
        Tau2["MVEF"] += 0.25 * V_["EFMU"] * Gamma1_["UV"];
        Tau2["MVEZ"] += 0.25 * V_["EWMU"] * Gamma1_["UV"] * Eta1_["ZW"];
        Tau2["MVZE"] += 0.25 * V_["WEMU"] * Gamma1_["UV"] * Eta1_["ZW"];
        Tau2["MVZX"] += 0.25 * V_["WYMU"] * Gamma1_["UV"] * Eta1_["ZW"] * Eta1_["XY"];
        Tau2["VMEF"] += 0.25 * V_["EFUM"] * Gamma1_["UV"];
        Tau2["VMEZ"] += 0.25 * V_["EWUM"] * Gamma1_["UV"] * Eta1_["ZW"];
        Tau2["VMZE"] += 0.25 * V_["WEUM"] * Gamma1_["UV"] * Eta1_["ZW"];
        Tau2["VMZX"] += 0.25 * V_["WYUM"] * Gamma1_["UV"] * Eta1_["ZW"] * Eta1_["XY"];
        Tau2["VYEF"] += 0.25 * V_["EFUX"] * Gamma1_["UV"] * Gamma1_["XY"];
        Tau2["VYEZ"] += 0.25 * V_["EWUX"] * Gamma1_["UV"] * Gamma1_["XY"] * Eta1_["ZW"];
        Tau2["VYZE"] += 0.25 * V_["WEUX"] * Gamma1_["UV"] * Gamma1_["XY"] * Eta1_["ZW"];
        Tau2["V,Y,Z,U1"] += 0.25 * V_["W,A1,U,X"] * Gamma1_["UV"] * Gamma1_["XY"] * Eta1_["ZW"] * Eta1_["U1,A1"];

        Tau2["mNeF"] += 0.25 * V_["eFmN"];
        Tau2["mNeZ"] += 0.25 * V_["eWmN"] * Eta1_["ZW"];
        Tau2["mNzE"] += 0.25 * V_["wEmN"] * Eta1_["zw"];
        Tau2["mNzX"] += 0.25 * V_["wYmN"] * Eta1_["zw"] * Eta1_["XY"];
        Tau2["mVeF"] += 0.25 * V_["eFmU"] * Gamma1_["UV"];
        Tau2["mVeZ"] += 0.25 * V_["eWmU"] * Gamma1_["UV"] * Eta1_["ZW"];
        Tau2["mVzE"] += 0.25 * V_["wEmU"] * Gamma1_["UV"] * Eta1_["zw"];
        Tau2["mVzX"] += 0.25 * V_["wYmU"] * Gamma1_["UV"] * Eta1_["zw"] * Eta1_["XY"];
        Tau2["vMeF"] += 0.25 * V_["eFuM"] * Gamma1_["uv"];
        Tau2["vMeZ"] += 0.25 * V_["eWuM"] * Gamma1_["uv"] * Eta1_["ZW"];
        Tau2["vMzE"] += 0.25 * V_["wEuM"] * Gamma1_["uv"] * Eta1_["zw"];
        Tau2["vMzX"] += 0.25 * V_["wYuM"] * Gamma1_["uv"] * Eta1_["zw"] * Eta1_["XY"];
        Tau2["vYeF"] += 0.25 * V_["eFuX"] * Gamma1_["uv"] * Gamma1_["XY"];
        Tau2["vYeZ"] += 0.25 * V_["eWuX"] * Gamma1_["uv"] * Gamma1_["XY"] * Eta1_["ZW"];
        Tau2["vYzE"] += 0.25 * V_["wEuX"] * Gamma1_["uv"] * Gamma1_["XY"] * Eta1_["zw"];
        Tau2["v,Y,z,U1"] += 0.25 * V_["w,A1,u,X"] * Gamma1_["uv"] * Gamma1_["XY"] * Eta1_["zw"] * Eta1_["U1,A1"];
    }
    // <[V, T2]> C_4 (C_2)^2 PP
    if (X1_TERM) {
        Tau2["uvef"] += 0.125 * V_["efxy"] * Lambda2_["xyuv"];
        Tau2["uvez"] += 0.125 * V_["ewxy"] * Eta1_["zw"] * Lambda2_["xyuv"];
        Tau2["uvze"] += 0.125 * V_["wexy"] * Eta1_["zw"] * Lambda2_["xyuv"];
        Tau2["u,v,z,u1"] += 0.125 * V_["w,a1,x,y"] * Eta1_["zw"] * Eta1_["u1,a1"] * Lambda2_["xyuv"];
        Tau2["UVEF"] += 0.125 * V_["EFXY"] * Lambda2_["XYUV"];
        Tau2["UVEZ"] += 0.125 * V_["EWXY"] * Eta1_["ZW"] * Lambda2_["XYUV"];
        Tau2["UVZE"] += 0.125 * V_["WEXY"] * Eta1_["ZW"] * Lambda2_["XYUV"];
        Tau2["U,V,Z,U1"] += 0.125 * V_["W,A1,X,Y"] * Eta1_["ZW"] * Eta1_["U1,A1"] * Lambda2_["XYUV"];
        Tau2["uVeF"] += 0.250 * V_["eFxY"] * Lambda2_["xYuV"];
        Tau2["uVeZ"] += 0.250 * V_["eWxY"] * Eta1_["ZW"] * Lambda2_["xYuV"];
        Tau2["uVzE"] += 0.250 * V_["wExY"] * Eta1_["zw"] * Lambda2_["xYuV"];
        Tau2["u,V,z,U1"] += 0.250 * V_["w,A1,x,Y"] * Eta1_["zw"] * Eta1_["U1,A1"] * Lambda2_["xYuV"];
    }
    // <[V, T2]> C_4 (C_2)^2 HH
    if (X2_TERM) {
        Tau2["mnxy"] += 0.125 * V_["uvmn"] * Lambda2_["xyuv"];
        Tau2["mwxy"] += 0.125 * V_["uvmz"] * Gamma1_["zw"] * Lambda2_["xyuv"];
        Tau2["a1,m,x,y"] += 0.125 * V_["u,v,u1,m"] * Gamma1_["u1,a1"] * Lambda2_["xyuv"];
        Tau2["a1,w,x,y"] += 0.125 * V_["u,v,u1,z"] * Gamma1_["u1,a1"] * Gamma1_["zw"] * Lambda2_["xyuv"];

        Tau2["MNXY"] += 0.125 * V_["UVMN"] * Lambda2_["XYUV"];
        Tau2["MWXY"] += 0.125 * V_["UVMZ"] * Gamma1_["ZW"] * Lambda2_["XYUV"];
        Tau2["WMXY"] += 0.125 * V_["UVZM"] * Gamma1_["ZW"] * Lambda2_["XYUV"];
        Tau2["W,A1,X,Y"] += 0.125 * V_["U,V,Z,U1"] * Gamma1_["ZW"] * Gamma1_["U1,A1"] * Lambda2_["XYUV"];

        Tau2["mNxY"] += 0.250 * V_["uVmN"] * Lambda2_["xYuV"];
        Tau2["mWxY"] += 0.250 * V_["uVmZ"] * Gamma1_["ZW"] * Lambda2_["xYuV"];
        Tau2["wMxY"] += 0.250 * V_["uVzM"] * Gamma1_["zw"] * Lambda2_["xYuV"];
        Tau2["w,A1,x,Y"] += 0.250 * V_["u,V,z,U1"] * Gamma1_["zw"] * Gamma1_["U1,A1"] * Lambda2_["xYuV"];
    }
    // <[V, T2]> C_4 (C_2)^2 PH
    if (X3_TERM) {
        Tau2["muye"] -= 0.25 * V_["vemx"] * Lambda2_["xyuv"];
        Tau2["muyz"] -= 0.25 * V_["vwmx"] * Eta1_["zw"] * Lambda2_["xyuv"];
        Tau2["zuye"] -= 0.25 * V_["vewx"] * Gamma1_["zw"] * Lambda2_["xyuv"];
        Tau2["z,u,y,u1"] -= 0.25 * V_["v,a1,w,x"] * Gamma1_["zw"] * Eta1_["u1,a1"] * Lambda2_["xyuv"];
        Tau2["muye"] -= 0.25 * V_["eVmX"] * Lambda2_["yXuV"];
        Tau2["muyz"] -= 0.25 * V_["wVmX"] * Eta1_["zw"] * Lambda2_["yXuV"];
        Tau2["zuye"] -= 0.25 * V_["eVwX"] * Gamma1_["zw"] * Lambda2_["yXuV"];
        Tau2["z,u,y,u1"] -= 0.25 * V_["a1,V,w,X"] * Gamma1_["zw"] * Eta1_["u1,a1"] * Lambda2_["yXuV"];
        Tau2["MUYE"] -= 0.25 * V_["VEMX"] * Lambda2_["XYUV"];
        Tau2["MUYZ"] -= 0.25 * V_["VWMX"] * Eta1_["ZW"] * Lambda2_["XYUV"];
        Tau2["ZUYE"] -= 0.25 * V_["VEWX"] * Gamma1_["ZW"] * Lambda2_["XYUV"];
        Tau2["Z,U,Y,U1"] -= 0.25 * V_["V,A1,W,X"] * Gamma1_["ZW"] * Eta1_["U1,A1"] * Lambda2_["XYUV"];
        Tau2["MUYE"] -= 0.25 * V_["vExM"] * Lambda2_["xYvU"];
        Tau2["MUYZ"] -= 0.25 * V_["vWxM"] * Eta1_["ZW"] * Lambda2_["xYvU"];
        Tau2["ZUYE"] -= 0.25 * V_["vExW"] * Gamma1_["ZW"] * Lambda2_["xYvU"];
        Tau2["Z,U,Y,U1"] -= 0.25 * V_["v,A1,x,W"] * Gamma1_["ZW"] * Eta1_["U1,A1"] * Lambda2_["xYvU"];
        Tau2["mUyE"] -= 0.25 * V_["vEmX"] * Lambda2_["yXvU"];
        Tau2["mUyZ"] -= 0.25 * V_["vWmX"] * Eta1_["ZW"] * Lambda2_["yXvU"];
        Tau2["zUyE"] -= 0.25 * V_["vEwX"] * Gamma1_["zw"] * Lambda2_["yXvU"];
        Tau2["z,U,y,U1"] -= 0.25 * V_["v,A1,w,X"] * Gamma1_["zw"] * Eta1_["U1,A1"] * Lambda2_["yXvU"];

        Tau2["umye"] -= 0.25 * V_["vexm"] * Lambda2_["xyuv"];
        Tau2["umyz"] -= 0.25 * V_["vwxm"] * Eta1_["zw"] * Lambda2_["xyuv"];
        Tau2["uzye"] -= 0.25 * V_["vexw"] * Gamma1_["zw"] * Lambda2_["xyuv"];
        Tau2["u,z,y,u1"] -= 0.25 * V_["v,a1,x,w"] * Gamma1_["zw"] * Eta1_["u1,a1"] * Lambda2_["xyuv"];
        Tau2["umye"] += 0.25 * V_["eVmX"] * Lambda2_["yXuV"];
        Tau2["umyz"] += 0.25 * V_["wVmX"] * Eta1_["zw"] * Lambda2_["yXuV"];
        Tau2["uzye"] += 0.25 * V_["eVwX"] * Gamma1_["zw"] * Lambda2_["yXuV"];
        Tau2["u,z,y,u1"] += 0.25 * V_["a1,V,w,X"] * Gamma1_["zw"] * Eta1_["u1,a1"] * Lambda2_["yXuV"];
        Tau2["UMYE"] -= 0.25 * V_["VEXM"] * Lambda2_["XYUV"];
        Tau2["UMYZ"] -= 0.25 * V_["VWXM"] * Eta1_["ZW"] * Lambda2_["XYUV"];
        Tau2["UZYE"] -= 0.25 * V_["VEXW"] * Gamma1_["ZW"] * Lambda2_["XYUV"];
        Tau2["U,Z,Y,U1"] -= 0.25 * V_["V,A1,X,W"] * Gamma1_["ZW"] * Eta1_["U1,A1"] * Lambda2_["XYUV"];
        Tau2["UMYE"] += 0.25 * V_["vExM"] * Lambda2_["xYvU"];
        Tau2["UMYZ"] += 0.25 * V_["vWxM"] * Eta1_["ZW"] * Lambda2_["xYvU"];
        Tau2["UZYE"] += 0.25 * V_["vExW"] * Gamma1_["ZW"] * Lambda2_["xYvU"];
        Tau2["U,Z,Y,U1"] += 0.25 * V_["v,A1,x,W"] * Gamma1_["ZW"] * Eta1_["U1,A1"] * Lambda2_["xYvU"];
        Tau2["uMyE"] -= 0.25 * V_["vExM"] * Lambda2_["xyuv"];
        Tau2["uMyZ"] -= 0.25 * V_["vWxM"] * Eta1_["ZW"] * Lambda2_["xyuv"];
        Tau2["uZyE"] -= 0.25 * V_["vExW"] * Gamma1_["ZW"] * Lambda2_["xyuv"];
        Tau2["u,Z,y,U1"] -= 0.25 * V_["v,A1,x,W"] * Gamma1_["ZW"] * Eta1_["U1,A1"] * Lambda2_["xyuv"];
        Tau2["uMyE"] += 0.25 * V_["VEXM"] * Lambda2_["yXuV"];
        Tau2["uMyZ"] += 0.25 * V_["VWXM"] * Eta1_["ZW"] * Lambda2_["yXuV"];
        Tau2["uZyE"] += 0.25 * V_["VEXW"] * Gamma1_["ZW"] * Lambda2_["yXuV"];
        Tau2["u,Z,y,U1"] += 0.25 * V_["V,A1,X,W"] * Gamma1_["ZW"] * Eta1_["U1,A1"] * Lambda2_["yXuV"];

        Tau2["umey"] -= 0.25 * V_["evxm"] * Lambda2_["xyuv"];
        Tau2["umzy"] -= 0.25 * V_["wvxm"] * Eta1_["zw"] * Lambda2_["xyuv"];
        Tau2["uzey"] -= 0.25 * V_["evxw"] * Gamma1_["zw"] * Lambda2_["xyuv"];
        Tau2["u,z,u1,y"] -= 0.25 * V_["a1,v,x,w"] * Gamma1_["zw"] * Eta1_["u1,a1"] * Lambda2_["xyuv"];
        Tau2["umey"] -= 0.25 * V_["eVmX"] * Lambda2_["yXuV"];
        Tau2["umzy"] -= 0.25 * V_["wVmX"] * Eta1_["zw"] * Lambda2_["yXuV"];
        Tau2["uzey"] -= 0.25 * V_["eVwX"] * Gamma1_["zw"] * Lambda2_["yXuV"];
        Tau2["u,z,u1,y"] -= 0.25 * V_["a1,V,w,X"] * Gamma1_["zw"] * Eta1_["u1,a1"] * Lambda2_["yXuV"];
        Tau2["UMEY"] -= 0.25 * V_["EVXM"] * Lambda2_["XYUV"];
        Tau2["UMZY"] -= 0.25 * V_["WVXM"] * Eta1_["ZW"] * Lambda2_["XYUV"];
        Tau2["UZEY"] -= 0.25 * V_["EVXW"] * Gamma1_["ZW"] * Lambda2_["XYUV"];
        Tau2["U,Z,U1,Y"] -= 0.25 * V_["A1,V,X,W"] * Gamma1_["ZW"] * Eta1_["U1,A1"] * Lambda2_["XYUV"];
        Tau2["UMEY"] -= 0.25 * V_["vExM"] * Lambda2_["xYvU"];
        Tau2["UMZY"] -= 0.25 * V_["vWxM"] * Eta1_["ZW"] * Lambda2_["xYvU"];
        Tau2["UZEY"] -= 0.25 * V_["vExW"] * Gamma1_["ZW"] * Lambda2_["xYvU"];
        Tau2["U,Z,U1,Y"] -= 0.25 * V_["v,A1,x,W"] * Gamma1_["ZW"] * Eta1_["U1,A1"] * Lambda2_["xYvU"];
        Tau2["uMeY"] -= 0.25 * V_["eVxM"] * Lambda2_["xYuV"];
        Tau2["uMzY"] -= 0.25 * V_["wVxM"] * Eta1_["zw"] * Lambda2_["xYuV"];
        Tau2["uZeY"] -= 0.25 * V_["eVxW"] * Gamma1_["ZW"] * Lambda2_["xYuV"];
        Tau2["u,Z,u1,Y"] -= 0.25 * V_["a1,V,x,W"] * Gamma1_["ZW"] * Eta1_["u1,a1"] * Lambda2_["xYuV"];

        Tau2["muey"] -= 0.25 * V_["evmx"] * Lambda2_["xyuv"];
        Tau2["muzy"] -= 0.25 * V_["wvmx"] * Eta1_["zw"] * Lambda2_["xyuv"];
        Tau2["zuey"] -= 0.25 * V_["evwx"] * Gamma1_["zw"] * Lambda2_["xyuv"];
        Tau2["z,u,u1,y"] -= 0.25 * V_["a1,v,w,x"] * Gamma1_["zw"] * Eta1_["u1,a1"] * Lambda2_["xyuv"];
        Tau2["muey"] += 0.25 * V_["eVmX"] * Lambda2_["yXuV"];
        Tau2["muzy"] += 0.25 * V_["wVmX"] * Eta1_["zw"] * Lambda2_["yXuV"];
        Tau2["zuey"] += 0.25 * V_["eVwX"] * Gamma1_["zw"] * Lambda2_["yXuV"];
        Tau2["z,u,u1,y"] += 0.25 * V_["a1,V,w,X"] * Gamma1_["zw"] * Eta1_["u1,a1"] * Lambda2_["yXuV"];
        Tau2["MUEY"] -= 0.25 * V_["EVMX"] * Lambda2_["XYUV"];
        Tau2["MUZY"] -= 0.25 * V_["WVMX"] * Eta1_["ZW"] * Lambda2_["XYUV"];
        Tau2["ZUEY"] -= 0.25 * V_["EVWX"] * Gamma1_["ZW"] * Lambda2_["XYUV"];
        Tau2["Z,U,U1,Y"] -= 0.25 * V_["A1,V,W,X"] * Gamma1_["ZW"] * Eta1_["U1,A1"] * Lambda2_["XYUV"];
        Tau2["MUEY"] += 0.25 * V_["vExM"] * Lambda2_["xYvU"];
        Tau2["MUZY"] += 0.25 * V_["vWxM"] * Eta1_["ZW"] * Lambda2_["xYvU"];
        Tau2["ZUEY"] += 0.25 * V_["vExW"] * Gamma1_["ZW"] * Lambda2_["xYvU"];
        Tau2["Z,U,U1,Y"] += 0.25 * V_["v,A1,x,W"] * Gamma1_["ZW"] * Eta1_["U1,A1"] * Lambda2_["xYvU"];
        Tau2["mUeY"] -= 0.25 * V_["eVmX"] * Lambda2_["XYUV"];
        Tau2["mUzY"] -= 0.25 * V_["wVmX"] * Eta1_["zw"] * Lambda2_["XYUV"];
        Tau2["zUeY"] -= 0.25 * V_["eVwX"] * Gamma1_["zw"] * Lambda2_["XYUV"];
        Tau2["z,U,u1,Y"] -= 0.25 * V_["a1,V,w,X"] * Gamma1_["zw"] * Eta1_["u1,a1"] * Lambda2_["XYUV"];
        Tau2["mUeY"] += 0.25 * V_["evmx"] * Lambda2_["xYvU"];
        Tau2["mUzY"] += 0.25 * V_["wvmx"] * Eta1_["zw"] * Lambda2_["xYvU"];
        Tau2["zUeY"] += 0.25 * V_["evwx"] * Gamma1_["zw"] * Lambda2_["xYvU"];
        Tau2["z,U,u1,Y"] += 0.25 * V_["a1,v,w,x"] * Gamma1_["zw"] * Eta1_["u1,a1"] * Lambda2_["xYvU"];
    }
    // <[V, T2]> C_6 C_2
    if (X4_TERM) {
        Tau2.block("caaa")("mwxy") += 0.125 * V_.block("aaca")("uvmz") * rdms_.L3aaa()("xyzuvw");
        Tau2.block("caaa")("mwxy") -= 0.250 * V_.block("aAcA")("uVmZ") * rdms_.L3aab()("xyZuwV");
        Tau2.block("CAAA")("MWXY") += 0.125 * V_.block("AACA")("UVMZ") * rdms_.L3bbb()("XYZUVW");
        Tau2.block("CAAA")("MWXY") -= 0.250 * V_.block("aAaC")("uVzM") * rdms_.L3abb()("zXYuVW");
        Tau2.block("cAaA")("mWxY") -= 0.125 * V_.block("aaca")("uvmz") * rdms_.L3aab()("xzYuvW");
        Tau2.block("cAaA")("mWxY") += 0.250 * V_.block("aAcA")("uVmZ") * rdms_.L3abb()("xYZuVW");

        Tau2.block("acaa")("wmxy") += 0.125 * V_.block("aaac")("uvzm") * rdms_.L3aaa()("xyzuvw");
        Tau2.block("acaa")("wmxy") += 0.250 * V_.block("aAcA")("uVmZ") * rdms_.L3aab()("xyZuwV");
        Tau2.block("ACAA")("WMXY") += 0.125 * V_.block("AAAC")("UVZM") * rdms_.L3bbb()("XYZUVW");
        Tau2.block("ACAA")("WMXY") += 0.250 * V_.block("aAaC")("uVzM") * rdms_.L3abb()("zXYuVW");
        Tau2.block("aCaA")("wMxY") += 0.125 * V_.block("AAAC")("UVZM") * rdms_.L3abb()("xYZwUV");
        Tau2.block("aCaA")("wMxY") += 0.250 * V_.block("aAaC")("uVzM") * rdms_.L3aab()("xzYuwV");
 
        Tau2.block("aava")("xyew") -= 0.125 * V_.block("vaaa")("ezuv") * rdms_.L3aaa()("xyzuvw");
        Tau2.block("aava")("xyew") += 0.250 * V_.block("vAaA")("eZuV") * rdms_.L3aab()("xyZuwV");
        Tau2.block("AAVA")("XYEW") -= 0.125 * V_.block("VAAA")("EZUV") * rdms_.L3bbb()("XYZUVW");
        Tau2.block("AAVA")("XYEW") += 0.250 * V_.block("aVaA")("zEuV") * rdms_.L3abb()("zXYuVW");
        Tau2.block("aAvA")("xYeW") += 0.125 * V_.block("vaaa")("ezuv") * rdms_.L3aab()("xzYuvW");
        Tau2.block("aAvA")("xYeW") -= 0.250 * V_.block("vAaA")("eZuV") * rdms_.L3abb()("xYZuVW");

        Tau2.block("aaav")("xywe") -= 0.125 * V_.block("avaa")("zeuv") * rdms_.L3aaa()("xyzuvw");
        Tau2.block("aaav")("xywe") -= 0.250 * V_.block("vAaA")("eZuV") * rdms_.L3aab()("xyZuwV");
        Tau2.block("AAAV")("XYWE") -= 0.125 * V_.block("AVAA")("ZEUV") * rdms_.L3bbb()("XYZUVW");
        Tau2.block("AAAV")("XYWE") -= 0.250 * V_.block("aVaA")("zEuV") * rdms_.L3abb()("zXYuVW");
        Tau2.block("aAaV")("xYwE") -= 0.125 * V_.block("AVAA")("ZEUV") * rdms_.L3abb()("xYZwUV");
        Tau2.block("aAaV")("xYwE") -= 0.250 * V_.block("aVaA")("zEuV") * rdms_.L3aab()("xzYuwV");
    }
    if (CORRELATION_TERM) {
        Tau2["iuax"] += 0.25 * DelGam1["xu"] * Sigma2["ia"];
        Tau2["IUAX"] += 0.25 * DelGam1["XU"] * Sigma2["IA"];
        Tau2["iUaX"] += 0.25 * DelGam1["XU"] * Sigma2["ia"];

        Tau2["iuxa"] -= 0.25 * DelGam1["xu"] * Sigma2["ia"];
        Tau2["IUXA"] -= 0.25 * DelGam1["XU"] * Sigma2["IA"];

        Tau2["uixa"] += 0.25 * DelGam1["xu"] * Sigma2["ia"];
        Tau2["UIXA"] += 0.25 * DelGam1["XU"] * Sigma2["IA"];
        Tau2["uIxA"] += 0.25 * DelGam1["xu"] * Sigma2["IA"];

        Tau2["uiax"] -= 0.25 * DelGam1["xu"] * Sigma2["ia"];
        Tau2["UIAX"] -= 0.25 * DelGam1["XU"] * Sigma2["IA"];
    }
    // <[F, T2]>
    if (X5_TERM) {
        Tau2["uvey"] += 0.125 * F_["ex"] * Lambda2_["xyuv"];
        Tau2["UVEY"] += 0.125 * F_["EX"] * Lambda2_["XYUV"];
        Tau2["uVeY"] += 0.125 * F_["ex"] * Lambda2_["xYuV"];

        Tau2["vuye"] += 0.125 * F_["ex"] * Lambda2_["yxvu"];
        Tau2["VUYE"] += 0.125 * F_["EX"] * Lambda2_["YXVU"];
        Tau2["vUyE"] += 0.125 * F_["EX"] * Lambda2_["yXvU"];

        Tau2["uvye"] += 0.125 * F_["ex"] * Lambda2_["yxuv"];
        Tau2["UVYE"] += 0.125 * F_["EX"] * Lambda2_["YXUV"];
        Tau2["uVyE"] += 0.125 * F_["EX"] * Lambda2_["yXuV"];

        Tau2["vuey"] += 0.125 * F_["ex"] * Lambda2_["xyvu"];
        Tau2["VUEY"] += 0.125 * F_["EX"] * Lambda2_["XYVU"];
        Tau2["vUeY"] += 0.125 * F_["ex"] * Lambda2_["xYvU"];
    
        Tau2["umxy"] -= 0.125 * F_["vm"] * Lambda2_["xyuv"];
        Tau2["UMXY"] -= 0.125 * F_["VM"] * Lambda2_["XYUV"];
        Tau2["uMxY"] -= 0.125 * F_["VM"] * Lambda2_["xYuV"];

        Tau2["muyx"] -= 0.125 * F_["vm"] * Lambda2_["yxvu"];
        Tau2["MUYX"] -= 0.125 * F_["VM"] * Lambda2_["YXVU"];
        Tau2["mUyX"] -= 0.125 * F_["vm"] * Lambda2_["yXvU"];

        Tau2["umyx"] -= 0.125 * F_["vm"] * Lambda2_["yxuv"];
        Tau2["UMYX"] -= 0.125 * F_["VM"] * Lambda2_["YXUV"];
        Tau2["uMyX"] -= 0.125 * F_["VM"] * Lambda2_["yXuV"];

        Tau2["muxy"] -= 0.125 * F_["vm"] * Lambda2_["xyvu"];
        Tau2["MUXY"] -= 0.125 * F_["VM"] * Lambda2_["XYVU"];
        Tau2["mUxY"] -= 0.125 * F_["vm"] * Lambda2_["xYvU"];
    }

    if (CORRELATION_TERM) {
        Tau2["iuax"] += 0.25 * DelGam1["xu"] * Xi3["ia"];
        Tau2["IUAX"] += 0.25 * DelGam1["XU"] * Xi3["IA"];
        Tau2["iUaX"] += 0.25 * DelGam1["XU"] * Xi3["ia"];

        Tau2["iuxa"] -= 0.25 * DelGam1["xu"] * Xi3["ia"];
        Tau2["IUXA"] -= 0.25 * DelGam1["XU"] * Xi3["IA"];

        Tau2["uixa"] += 0.25 * DelGam1["xu"] * Xi3["ia"];
        Tau2["UIXA"] += 0.25 * DelGam1["XU"] * Xi3["IA"];
        Tau2["uIxA"] += 0.25 * DelGam1["xu"] * Xi3["IA"];

        Tau2["uiax"] -= 0.25 * DelGam1["xu"] * Xi3["ia"];
        Tau2["UIAX"] -= 0.25 * DelGam1["XU"] * Xi3["IA"];
    }

    // NOTICE: remove the internal parts based on the DSRG theories
    Tau2.block("aaaa").zero();
    Tau2.block("aAaA").zero();
    Tau2.block("AAAA").zero();

    // Tau * [1 - e^(-s * Delta^2)]
    Tau1["ijab"] = Tau2["ijab"] * Eeps2_m1["ijab"];
    Tau1["IJAB"] = Tau2["IJAB"] * Eeps2_m1["IJAB"];
    Tau1["iJaB"] = Tau2["iJaB"] * Eeps2_m1["iJaB"];

    outfile->Printf("Done");
}

void DSRG_MRPT2::set_sigma() {
    outfile->Printf("\n    Initializing Sigma .............................. ");
    Sigma = BTF_->build(CoreTensor, "Sigma", spin_cases({"hp"}));    
    Sigma1 = BTF_->build(CoreTensor, "Sigma * DelEeps1", spin_cases({"hp"}));    
    Sigma2 = BTF_->build(CoreTensor, "Sigma * Eeps1", spin_cases({"hp"}));    
    Sigma3 = BTF_->build(CoreTensor, "Sigma * (1 + Eeps1)", spin_cases({"hp"}));

    if (X5_TERM) {
        Sigma["xe"] += 0.5 * T2_["uvey"] * Lambda2_["xyuv"];
        Sigma["xe"] += T2_["uVeY"] * Lambda2_["xYuV"];
        Sigma["XE"] += 0.5 * T2_["UVEY"] * Lambda2_["XYUV"];
        Sigma["XE"] += T2_["uVyE"] * Lambda2_["yXuV"];
    
        Sigma["mv"] -= 0.5 * T2_["umxy"] * Lambda2_["xyuv"];
        Sigma["mv"] -= T2_["mUxY"] * Lambda2_["xYvU"];
        Sigma["MV"] -= 0.5 * T2_["UMXY"] * Lambda2_["XYUV"];
        Sigma["MV"] -= T2_["uMxY"] * Lambda2_["xYuV"];
    }

    if (X7_TERM) {
        Sigma["me"] += T1_["me"];
        Sigma["mv"] += T1_["mu"] * Eta1_["uv"];
        Sigma["ve"] += T1_["ue"] * Gamma1_["uv"];

        Sigma["ME"] += T1_["ME"];
        Sigma["MV"] += T1_["MU"] * Eta1_["UV"];
        Sigma["VE"] += T1_["UE"] * Gamma1_["UV"];
    }

    Sigma1["ia"] = Sigma["ia"] * DelEeps1["ia"];
    Sigma1["IA"] = Sigma["IA"] * DelEeps1["IA"];

    Sigma2["ia"] = Sigma["ia"] * Eeps1["ia"];
    Sigma2["IA"] = Sigma["IA"] * Eeps1["IA"];

    Sigma3["ia"] = Sigma["ia"];
    Sigma3["ia"] += Sigma2["ia"];
    Sigma3["IA"] = Sigma["IA"];
    Sigma3["IA"] += Sigma2["IA"];

    outfile->Printf("Done");
}

void DSRG_MRPT2::set_xi() {
    outfile->Printf("\n    Initializing Xi ................................. ");
    Xi = BTF_->build(CoreTensor, "Xi", spin_cases({"hp"}));    
    Xi1 = BTF_->build(CoreTensor, "Xi * Eeps1_m2", spin_cases({"hp"}));  
    Xi2 = BTF_->build(CoreTensor, "Xi * Eeps1", spin_cases({"hp"}));    
    Xi3 = BTF_->build(CoreTensor, "Xi * Eeps1_m1", spin_cases({"hp"}));   

    if (X6_TERM) {
        Xi["ue"] += 0.5 * V_["evxy"] * Lambda2_["xyuv"];
        Xi["ue"] += V_["eVxY"] * Lambda2_["xYuV"];
        Xi["UE"] += 0.5 * V_["EVXY"] * Lambda2_["XYUV"];
        Xi["UE"] += V_["vExY"] * Lambda2_["xYvU"];
    
        Xi["mx"] -= 0.5 * V_["uvmy"] * Lambda2_["xyuv"];
        Xi["mx"] -= V_["uVmY"] * Lambda2_["xYuV"];
        Xi["MX"] -= 0.5 * V_["UVMY"] * Lambda2_["XYUV"];
        Xi["MX"] -= V_["uVyM"] * Lambda2_["yXuV"];
    }

    if (X7_TERM) {
        Xi["me"] += F_["em"];
        Xi["mu"] += F_["vm"] * Eta1_["uv"];
        Xi["ue"] += F_["ev"] * Gamma1_["uv"];

        Xi["ME"] += F_["EM"];
        Xi["MU"] += F_["VM"] * Eta1_["UV"];
        Xi["UE"] += F_["EV"] * Gamma1_["UV"];
    }

    Xi1["ia"] = Xi["ia"] * Eeps1_m2["ia"];
    Xi1["IA"] = Xi["IA"] * Eeps1_m2["IA"];

    Xi2["ia"] = Xi["ia"] * Eeps1["ia"];
    Xi2["IA"] = Xi["IA"] * Eeps1["IA"];

    Xi3["ia"] = Xi["ia"] * Eeps1_m1["ia"];
    Xi3["IA"] = Xi["IA"] * Eeps1_m1["IA"];

    outfile->Printf("Done");
}

void DSRG_MRPT2::set_kappa() {
    outfile->Printf("\n    Initializing Kappa .............................. ");
    Kappa = BTF_->build(CoreTensor, "Kappa", spin_cases({"hhpp"}));  
    // <[V, T2]> (C_2)^4
    if (PT2_TERM) {
        Kappa["mnef"] += 0.25 * T2_["mnef"];
        Kappa["mnev"] += 0.25 * T2_["mneu"] * Eta1_["uv"];
        Kappa["mnve"] += 0.25 * T2_["mnue"] * Eta1_["uv"];
        Kappa["mnvy"] += 0.25 * T2_["mnux"] * Eta1_["uv"] * Eta1_["xy"];
        Kappa["muef"] += 0.25 * T2_["mvef"] * Gamma1_["uv"];
        Kappa["muew"] += 0.25 * T2_["mvez"] * Gamma1_["uv"] * Eta1_["zw"];
        Kappa["muwe"] += 0.25 * T2_["mvze"] * Gamma1_["uv"] * Eta1_["zw"];
        Kappa["muwy"] += 0.25 * T2_["mvzx"] * Gamma1_["uv"] * Eta1_["zw"] * Eta1_["xy"];
        Kappa["umef"] += 0.25 * T2_["vmef"] * Gamma1_["uv"];
        Kappa["umew"] += 0.25 * T2_["vmez"] * Gamma1_["uv"] * Eta1_["zw"];
        Kappa["umwe"] += 0.25 * T2_["vmze"] * Gamma1_["uv"] * Eta1_["zw"];
        Kappa["umwy"] += 0.25 * T2_["vmzx"] * Gamma1_["uv"] * Eta1_["zw"] * Eta1_["xy"];
        Kappa["uxef"] += 0.25 * T2_["vyef"] * Gamma1_["uv"] * Gamma1_["xy"];
        Kappa["uxew"] += 0.25 * T2_["vyez"] * Gamma1_["uv"] * Gamma1_["xy"] * Eta1_["zw"];
        Kappa["uxwe"] += 0.25 * T2_["vyze"] * Gamma1_["uv"] * Gamma1_["xy"] * Eta1_["zw"];

        Kappa["MNEF"] += 0.25 * T2_["MNEF"];
        Kappa["MNEV"] += 0.25 * T2_["MNEU"] * Eta1_["UV"];
        Kappa["MNVE"] += 0.25 * T2_["MNUE"] * Eta1_["UV"];
        Kappa["MNVY"] += 0.25 * T2_["MNUX"] * Eta1_["UV"] * Eta1_["XY"];
        Kappa["MUEF"] += 0.25 * T2_["MVEF"] * Gamma1_["UV"];
        Kappa["MUEW"] += 0.25 * T2_["MVEZ"] * Gamma1_["UV"] * Eta1_["ZW"];
        Kappa["MUWE"] += 0.25 * T2_["MVZE"] * Gamma1_["UV"] * Eta1_["ZW"];
        Kappa["MUWY"] += 0.25 * T2_["MVZX"] * Gamma1_["UV"] * Eta1_["ZW"] * Eta1_["XY"];
        Kappa["UMEF"] += 0.25 * T2_["VMEF"] * Gamma1_["UV"];
        Kappa["UMEW"] += 0.25 * T2_["VMEZ"] * Gamma1_["UV"] * Eta1_["ZW"];
        Kappa["UMWE"] += 0.25 * T2_["VMZE"] * Gamma1_["UV"] * Eta1_["ZW"];
        Kappa["UMWY"] += 0.25 * T2_["VMZX"] * Gamma1_["UV"] * Eta1_["ZW"] * Eta1_["XY"];
        Kappa["UXEF"] += 0.25 * T2_["VYEF"] * Gamma1_["UV"] * Gamma1_["XY"];
        Kappa["UXEW"] += 0.25 * T2_["VYEZ"] * Gamma1_["UV"] * Gamma1_["XY"] * Eta1_["ZW"];
        Kappa["UXWE"] += 0.25 * T2_["VYZE"] * Gamma1_["UV"] * Gamma1_["XY"] * Eta1_["ZW"];

        Kappa["mNeF"] += 0.25 * T2_["mNeF"];
        Kappa["mNeV"] += 0.25 * T2_["mNeU"] * Eta1_["UV"];
        Kappa["mNvE"] += 0.25 * T2_["mNuE"] * Eta1_["uv"];
        Kappa["mNvY"] += 0.25 * T2_["mNuX"] * Eta1_["uv"] * Eta1_["XY"];
        Kappa["mUeF"] += 0.25 * T2_["mVeF"] * Gamma1_["UV"];
        Kappa["mUeW"] += 0.25 * T2_["mVeZ"] * Gamma1_["UV"] * Eta1_["ZW"];
        Kappa["mUwE"] += 0.25 * T2_["mVzE"] * Gamma1_["UV"] * Eta1_["zw"];
        Kappa["mUwY"] += 0.25 * T2_["mVzX"] * Gamma1_["UV"] * Eta1_["zw"] * Eta1_["XY"];
        Kappa["uMeF"] += 0.25 * T2_["vMeF"] * Gamma1_["uv"];
        Kappa["uMeW"] += 0.25 * T2_["vMeZ"] * Gamma1_["uv"] * Eta1_["ZW"];
        Kappa["uMwE"] += 0.25 * T2_["vMzE"] * Gamma1_["uv"] * Eta1_["zw"];
        Kappa["uMwY"] += 0.25 * T2_["vMzX"] * Gamma1_["uv"] * Eta1_["zw"] * Eta1_["XY"];
        Kappa["uXeF"] += 0.25 * T2_["vYeF"] * Gamma1_["uv"] * Gamma1_["XY"];
        Kappa["uXeW"] += 0.25 * T2_["vYeZ"] * Gamma1_["uv"] * Gamma1_["XY"] * Eta1_["ZW"];
        Kappa["uXwE"] += 0.25 * T2_["vYzE"] * Gamma1_["uv"] * Gamma1_["XY"] * Eta1_["zw"];
    }
    // <[V, T2]> C_4 (C_2)^2 PP
    if (X1_TERM) {
        Kappa["xyef"] += 0.125 * T2_["uvef"] * Lambda2_["xyuv"];
        Kappa["xyew"] += 0.125 * T2_["uvez"] * Eta1_["zw"] * Lambda2_["xyuv"];
        Kappa["xywe"] += 0.125 * T2_["uvze"] * Eta1_["zw"] * Lambda2_["xyuv"];

        Kappa["XYEF"] += 0.125 * T2_["UVEF"] * Lambda2_["XYUV"];
        Kappa["XYEW"] += 0.125 * T2_["UVEZ"] * Eta1_["ZW"] * Lambda2_["XYUV"];
        Kappa["XYWE"] += 0.125 * T2_["UVZE"] * Eta1_["ZW"] * Lambda2_["XYUV"];

        Kappa["xYeF"] += 0.250 * T2_["uVeF"] * Lambda2_["xYuV"];
        Kappa["xYeW"] += 0.250 * T2_["uVeZ"] * Eta1_["ZW"] * Lambda2_["xYuV"];
        Kappa["xYwE"] += 0.250 * T2_["uVzE"] * Eta1_["zw"] * Lambda2_["xYuV"];
    }
    // <[V, T2]> C_4 (C_2)^2 HH
    if (X2_TERM) {
        Kappa["mnuv"] += 0.125 * T2_["mnxy"] * Lambda2_["xyuv"];
        Kappa["mzuv"] += 0.125 * T2_["mwxy"] * Gamma1_["zw"] * Lambda2_["xyuv"];
        Kappa["zmuv"] += 0.125 * T2_["wmxy"] * Gamma1_["zw"] * Lambda2_["xyuv"];

        Kappa["MNUV"] += 0.125 * T2_["MNXY"] * Lambda2_["XYUV"];
        Kappa["MZUV"] += 0.125 * T2_["MWXY"] * Gamma1_["ZW"] * Lambda2_["XYUV"];
        Kappa["ZMUV"] += 0.125 * T2_["WMXY"] * Gamma1_["ZW"] * Lambda2_["XYUV"];

        Kappa["mNuV"] += 0.250 * T2_["mNxY"] * Lambda2_["xYuV"];
        Kappa["mZuV"] += 0.250 * T2_["mWxY"] * Gamma1_["ZW"] * Lambda2_["xYuV"];
        Kappa["zMuV"] += 0.250 * T2_["wMxY"] * Gamma1_["zw"] * Lambda2_["xYuV"];
    }
    // <[V, T2]> C_4 (C_2)^2 PH
    if (X3_TERM) {
        Kappa["mxve"] -= 0.25 * T2_["muye"] * Lambda2_["xyuv"];
        Kappa["mxvw"] -= 0.25 * T2_["muyz"] * Eta1_["zw"] * Lambda2_["xyuv"];
        Kappa["wxve"] -= 0.25 * T2_["zuye"] * Gamma1_["zw"] * Lambda2_["xyuv"];
        Kappa["mxve"] -= 0.25 * T2_["mUeY"] * Lambda2_["xYvU"];
        Kappa["mxvw"] -= 0.25 * T2_["mUzY"] * Eta1_["zw"] * Lambda2_["xYvU"];
        Kappa["wxve"] -= 0.25 * T2_["zUeY"] * Gamma1_["zw"] * Lambda2_["xYvU"];
        Kappa["MXVE"] -= 0.25 * T2_["MUYE"] * Lambda2_["XYUV"];
        Kappa["MXVW"] -= 0.25 * T2_["MUYZ"] * Eta1_["ZW"] * Lambda2_["XYUV"];
        Kappa["WXVE"] -= 0.25 * T2_["ZUYE"] * Gamma1_["ZW"] * Lambda2_["XYUV"];
        Kappa["MXVE"] -= 0.25 * T2_["uMyE"] * Lambda2_["yXuV"];
        Kappa["MXVW"] -= 0.25 * T2_["uMyZ"] * Eta1_["ZW"] * Lambda2_["yXuV"];
        Kappa["WXVE"] -= 0.25 * T2_["uZyE"] * Gamma1_["ZW"] * Lambda2_["yXuV"];
        Kappa["mXvE"] -= 0.25 * T2_["mUyE"] * Lambda2_["yXvU"];
        Kappa["mXvW"] -= 0.25 * T2_["mUyZ"] * Eta1_["ZW"] * Lambda2_["yXvU"];
        Kappa["wXvE"] -= 0.25 * T2_["zUyE"] * Gamma1_["zw"] * Lambda2_["yXvU"];

        Kappa["xmve"] -= 0.25 * T2_["umye"] * Lambda2_["xyuv"];
        Kappa["xmvw"] -= 0.25 * T2_["umyz"] * Eta1_["zw"] * Lambda2_["xyuv"];
        Kappa["xwve"] -= 0.25 * T2_["uzye"] * Gamma1_["zw"] * Lambda2_["xyuv"];
        Kappa["xmve"] += 0.25 * T2_["mUeY"] * Lambda2_["xYvU"];
        Kappa["xmvw"] += 0.25 * T2_["mUzY"] * Eta1_["zw"] * Lambda2_["xYvU"];
        Kappa["xwve"] += 0.25 * T2_["zUeY"] * Gamma1_["zw"] * Lambda2_["xYvU"];
        Kappa["XMVE"] -= 0.25 * T2_["UMYE"] * Lambda2_["XYUV"];
        Kappa["XMVW"] -= 0.25 * T2_["UMYZ"] * Eta1_["ZW"] * Lambda2_["XYUV"];
        Kappa["XWVE"] -= 0.25 * T2_["UZYE"] * Gamma1_["ZW"] * Lambda2_["XYUV"];
        Kappa["XMVE"] += 0.25 * T2_["uMyE"] * Lambda2_["yXuV"];
        Kappa["XMVW"] += 0.25 * T2_["uMyZ"] * Eta1_["ZW"] * Lambda2_["yXuV"];
        Kappa["XWVE"] += 0.25 * T2_["uZyE"] * Gamma1_["ZW"] * Lambda2_["yXuV"];
        Kappa["xMvE"] -= 0.25 * T2_["uMyE"] * Lambda2_["xyuv"];
        Kappa["xMvW"] -= 0.25 * T2_["uMyZ"] * Eta1_["ZW"] * Lambda2_["xyuv"];
        Kappa["xWvE"] -= 0.25 * T2_["uZyE"] * Gamma1_["ZW"] * Lambda2_["xyuv"];
        Kappa["xMvE"] += 0.25 * T2_["UMYE"] * Lambda2_["xYvU"];
        Kappa["xMvW"] += 0.25 * T2_["UMYZ"] * Eta1_["ZW"] * Lambda2_["xYvU"];
        Kappa["xWvE"] += 0.25 * T2_["UZYE"] * Gamma1_["ZW"] * Lambda2_["xYvU"];

        Kappa["xmev"] -= 0.25 * T2_["umey"] * Lambda2_["xyuv"];
        Kappa["xmwv"] -= 0.25 * T2_["umzy"] * Eta1_["zw"] * Lambda2_["xyuv"];
        Kappa["xwev"] -= 0.25 * T2_["uzey"] * Gamma1_["zw"] * Lambda2_["xyuv"];
        Kappa["xmev"] -= 0.25 * T2_["mUeY"] * Lambda2_["xYvU"];
        Kappa["xmwv"] -= 0.25 * T2_["mUzY"] * Eta1_["zw"] * Lambda2_["xYvU"];
        Kappa["xwev"] -= 0.25 * T2_["zUeY"] * Gamma1_["zw"] * Lambda2_["xYvU"];
        Kappa["XMEV"] -= 0.25 * T2_["UMEY"] * Lambda2_["XYUV"];
        Kappa["XMWV"] -= 0.25 * T2_["UMZY"] * Eta1_["ZW"] * Lambda2_["XYUV"];
        Kappa["XWEV"] -= 0.25 * T2_["UZEY"] * Gamma1_["ZW"] * Lambda2_["XYUV"];
        Kappa["XMEV"] -= 0.25 * T2_["uMyE"] * Lambda2_["yXuV"];
        Kappa["XMWV"] -= 0.25 * T2_["uMyZ"] * Eta1_["ZW"] * Lambda2_["yXuV"];
        Kappa["XWEV"] -= 0.25 * T2_["uZyE"] * Gamma1_["ZW"] * Lambda2_["yXuV"];
        Kappa["xMeV"] -= 0.25 * T2_["uMeY"] * Lambda2_["xYuV"];
        Kappa["xMwV"] -= 0.25 * T2_["uMzY"] * Eta1_["zw"] * Lambda2_["xYuV"];
        Kappa["xWeV"] -= 0.25 * T2_["uZeY"] * Gamma1_["ZW"] * Lambda2_["xYuV"];


        Kappa["mxev"] -= 0.25 * T2_["muey"] * Lambda2_["xyuv"];
        Kappa["mxwv"] -= 0.25 * T2_["muzy"] * Eta1_["zw"] * Lambda2_["xyuv"];
        Kappa["wxev"] -= 0.25 * T2_["zuey"] * Gamma1_["zw"] * Lambda2_["xyuv"];
        Kappa["mxev"] += 0.25 * T2_["mUeY"] * Lambda2_["xYvU"];
        Kappa["mxwv"] += 0.25 * T2_["mUzY"] * Eta1_["zw"] * Lambda2_["xYvU"];
        Kappa["wxev"] += 0.25 * T2_["zUeY"] * Gamma1_["zw"] * Lambda2_["xYvU"];
        Kappa["MXEV"] -= 0.25 * T2_["MUEY"] * Lambda2_["XYUV"];
        Kappa["MXWV"] -= 0.25 * T2_["MUZY"] * Eta1_["ZW"] * Lambda2_["XYUV"];
        Kappa["WXEV"] -= 0.25 * T2_["ZUEY"] * Gamma1_["ZW"] * Lambda2_["XYUV"];
        Kappa["MXEV"] += 0.25 * T2_["uMyE"] * Lambda2_["yXuV"];
        Kappa["MXWV"] += 0.25 * T2_["uMyZ"] * Eta1_["ZW"] * Lambda2_["yXuV"];
        Kappa["WXEV"] += 0.25 * T2_["uZyE"] * Gamma1_["ZW"] * Lambda2_["yXuV"];
        Kappa["mXeV"] -= 0.25 * T2_["mUeY"] * Lambda2_["XYUV"];
        Kappa["mXwV"] -= 0.25 * T2_["mUzY"] * Eta1_["zw"] * Lambda2_["XYUV"];
        Kappa["wXeV"] -= 0.25 * T2_["zUeY"] * Gamma1_["zw"] * Lambda2_["XYUV"];
        Kappa["mXeV"] += 0.25 * T2_["muey"] * Lambda2_["yXuV"];
        Kappa["mXwV"] += 0.25 * T2_["muzy"] * Eta1_["zw"] * Lambda2_["yXuV"];
        Kappa["wXeV"] += 0.25 * T2_["zuey"] * Gamma1_["zw"] * Lambda2_["yXuV"];
    }
    // <[V, T2]> C_6 C_2
    if (X4_TERM) {
        Kappa.block("caaa")("mzuv") += 0.125 * T2_.block("caaa")("mwxy") * rdms_.L3aaa()("xyzuvw");
        Kappa.block("caaa")("mzuv") -= 0.250 * T2_.block("cAaA")("mWxY") * rdms_.L3aab()("xzYuvW");
        Kappa.block("CAAA")("MZUV") += 0.125 * T2_.block("CAAA")("MWXY") * rdms_.L3bbb()("XYZUVW");
        Kappa.block("CAAA")("MZUV") -= 0.250 * T2_.block("aCaA")("wMxY") * rdms_.L3abb()("xYZwUV");
        Kappa.block("cAaA")("mZuV") -= 0.125 * T2_.block("caaa")("mwxy") * rdms_.L3aab()("xyZuwV");
        Kappa.block("cAaA")("mZuV") += 0.250 * T2_.block("cAaA")("mWxY") * rdms_.L3abb()("xYZuVW");

        Kappa.block("acaa")("zmuv") += 0.125 * T2_.block("acaa")("wmxy") * rdms_.L3aaa()("xyzuvw");
        Kappa.block("acaa")("zmuv") += 0.250 * T2_.block("cAaA")("mWxY") * rdms_.L3aab()("xzYuvW");
        Kappa.block("ACAA")("ZMUV") += 0.125 * T2_.block("ACAA")("WMXY") * rdms_.L3bbb()("XYZUVW");
        Kappa.block("ACAA")("ZMUV") += 0.250 * T2_.block("aCaA")("wMxY") * rdms_.L3abb()("xYZwUV");
        Kappa.block("aCaA")("zMuV") += 0.125 * T2_.block("ACAA")("WMXY") * rdms_.L3abb()("zXYuVW");
        Kappa.block("aCaA")("zMuV") += 0.250 * T2_.block("aCaA")("wMxY") * rdms_.L3aab()("xzYuwV");

        Kappa.block("aava")("uvez") -= 0.125 * T2_.block("aava")("xyew") * rdms_.L3aaa()("xyzuvw");
        Kappa.block("aava")("uvez") += 0.250 * T2_.block("aAvA")("xYeW") * rdms_.L3aab()("xzYuvW");
        Kappa.block("AAVA")("UVEZ") -= 0.125 * T2_.block("AAVA")("XYEW") * rdms_.L3bbb()("XYZUVW");
        Kappa.block("AAVA")("UVEZ") += 0.250 * T2_.block("aAaV")("xYwE") * rdms_.L3abb()("xYZwUV");
        Kappa.block("aAvA")("uVeZ") += 0.125 * T2_.block("aava")("xyew") * rdms_.L3aab()("xyZuwV");
        Kappa.block("aAvA")("uVeZ") -= 0.250 * T2_.block("aAvA")("xYeW") * rdms_.L3abb()("xYZuVW");

        Kappa.block("aaav")("uvze") -= 0.125 * T2_.block("aaav")("xywe") * rdms_.L3aaa()("xyzuvw");
        Kappa.block("aaav")("uvze") -= 0.250 * T2_.block("aAvA")("xYeW") * rdms_.L3aab()("xzYuvW");
        Kappa.block("AAAV")("UVZE") -= 0.125 * T2_.block("AAAV")("XYWE") * rdms_.L3bbb()("XYZUVW");
        Kappa.block("AAAV")("UVZE") -= 0.250 * T2_.block("aAaV")("xYwE") * rdms_.L3abb()("xYZwUV");
        Kappa.block("aAaV")("uVzE") -= 0.125 * T2_.block("AAAV")("XYWE") * rdms_.L3abb()("zXYuVW");
        Kappa.block("aAaV")("uVzE") -= 0.250 * T2_.block("aAaV")("xYwE") * rdms_.L3aab()("xzYuwV");
    }
    // <[V, T1]>
    if (X6_TERM) {
        Kappa["xyev"] += 0.25 * T1_["ue"] * Lambda2_["xyuv"];
        Kappa["XYEV"] += 0.25 * T1_["UE"] * Lambda2_["XYUV"];
        Kappa["xYeV"] += 0.25 * T1_["ue"] * Lambda2_["xYuV"];

        Kappa["yxve"] += 0.25 * T1_["ue"] * Lambda2_["yxvu"]; 
        Kappa["YXVE"] += 0.25 * T1_["UE"] * Lambda2_["YXVU"]; 
        Kappa["yXvE"] += 0.25 * T1_["UE"] * Lambda2_["yXvU"]; 
    
        Kappa["myuv"] -= 0.25 * T1_["mx"] * Lambda2_["xyuv"];
        Kappa["MYUV"] -= 0.25 * T1_["MX"] * Lambda2_["XYUV"];
        Kappa["mYuV"] -= 0.25 * T1_["mx"] * Lambda2_["xYuV"];

        Kappa["ymvu"] -= 0.25 * T1_["mx"] * Lambda2_["yxvu"];
        Kappa["YMVU"] -= 0.25 * T1_["MX"] * Lambda2_["YXVU"];
        Kappa["yMvU"] -= 0.25 * T1_["MX"] * Lambda2_["yXvU"];
    }

    outfile->Printf("Done");
}

void DSRG_MRPT2::set_z() {
    Z = BTF_->build(CoreTensor, "Z Matrix", spin_cases({"gg"}));
    outfile->Printf("\n    Initializing Diagonal Entries of Z .............. ");
    set_z_cc();  
    set_z_vv();
    set_z_aa_diag();
    outfile->Printf("Done");
    // LAPACK solver
    solve_z();
}

void DSRG_MRPT2::set_w() {
    outfile->Printf("\n    Solving Entries of W ............................ ");
    W = BTF_->build(CoreTensor, "Lagrangian", spin_cases({"gg"}));
    BlockedTensor temp = BTF_->build(CoreTensor, "temporal tensor", spin_cases({"hhpp"}));

    //NOTICE: w for {virtual-general}
    if (CORRELATION_TERM) {
        W["pe"] += 0.5 * Sigma3["ie"] * F["ip"];
    }

    if (CORRELATION_TERM) {
        W["pe"] += 0.5 * Xi3["ie"] * F["ip"];
    }

    if (CORRELATION_TERM) {
        W["pe"] += Tau1["ijeb"] * V["pbij"];
        W["pe"] += 2.0 * Tau1["iJeB"] * V["pBiJ"];

        temp["kled"] += Kappa["kled"] * Eeps2_p["kled"];
        temp["kLeD"] += Kappa["kLeD"] * Eeps2_p["kLeD"];
        W["pe"] += temp["kled"] * V["pdkl"];
        W["pe"] += 2.0 * temp["kLeD"] * V["pDkL"];
        temp.zero();
    }
    W["pe"] += Z["e,m1"] * F["m1,p"];
    W["pe"] += Z["eu"] * H["vp"] * Gamma1_["uv"];
    W["pe"] += Z["eu"] * V_N_Alpha["v,p"] * Gamma1_["uv"];
    W["pe"] += Z["eu"] * V_N_Beta["v,p"] * Gamma1_["uv"];
    W["pe"] += 0.5 * Z["eu"] * V["xypv"] * Gamma2_["uvxy"];
    W["pe"] += Z["eu"] * V["xYpV"] * Gamma2_["uVxY"];
    W["pe"] += Z["e,f1"] * F["f1,p"];
    W["ei"] = W["ie"];

    //NOTICE: w for {core-hole}
    if (CORRELATION_TERM) {
        W["jm"] += 0.5 * Sigma3["ma"] * F["ja"];
        W["jm"] += 0.5 * Sigma3["ia"] * V["amij"];
        W["jm"] += 0.5 * Sigma3["IA"] * V["mAjI"];
        W["jm"] += 0.5 * Sigma3["ia"] * V["ajim"];
        W["jm"] += 0.5 * Sigma3["IA"] * V["jAmI"];
    }

    if (CORRELATION_TERM) {
        W["jm"] += 0.5 * Xi3["ma"] * F["ja"];
        W["jm"] += 0.5 * Xi3["ia"] * V["amij"];
        W["jm"] += 0.5 * Xi3["IA"] * V["mAjI"];
        W["jm"] += 0.5 * Xi3["ia"] * V["ajim"];
        W["jm"] += 0.5 * Xi3["IA"] * V["jAmI"];
    }

    if (CORRELATION_TERM) {
        W["im"] += Tau1["mjab"] * V["abij"];
        W["im"] += 2.0 * Tau1["mJaB"] * V["aBiJ"];

        temp["mlcd"] += Kappa["mlcd"] * Eeps2_p["mlcd"];
        temp["mLcD"] += Kappa["mLcD"] * Eeps2_p["mLcD"];
        W["im"] += temp["mlcd"] * V["cdil"];
        W["im"] += 2.0 * temp["mLcD"] * V["cDiL"];
        temp.zero();
    }
    W["im"] += Z["e1,m"] * F["i,e1"];
    W["im"] += Z["e1,m1"] * V["e1,i,m1,m"];
    W["im"] += Z["E1,M1"] * V["i,E1,m,M1"];
    W["im"] += Z["e1,m1"] * V["e1,m,m1,i"];
    W["im"] += Z["E1,M1"] * V["m,E1,i,M1"];
    W["im"] += Z["mu"] * F["ui"];
    W["im"] -= Z["mu"] * H["vi"] * Gamma1_["uv"];
    W["im"] -= Z["mu"] * V_N_Alpha["vi"] * Gamma1_["uv"];
    W["im"] -= Z["mu"] * V_N_Beta["vi"] * Gamma1_["uv"];
    W["im"] -= 0.5 * Z["mu"] * V["xyiv"] * Gamma2_["uvxy"];
    W["im"] -= Z["mu"] * V["xYiV"] * Gamma2_["uVxY"];
    W["im"] += Z["n1,u"] * V["u,i,n1,m"];
    W["im"] += Z["N1,U"] * V["i,U,m,N1"];
    W["im"] += Z["n1,u"] * V["u,m,n1,i"];
    W["im"] += Z["N1,U"] * V["m,U,i,N1"];
    W["im"] -= Z["n1,u"] * Gamma1_["uv"] * V["v,i,n1,m"];
    W["im"] -= Z["N1,U"] * Gamma1_["UV"] * V["i,V,m,N1"];
    W["im"] -= Z["n1,u"] * Gamma1_["uv"] * V["v,m,n1,i"];
    W["im"] -= Z["N1,U"] * Gamma1_["UV"] * V["m,V,i,N1"];
    W["im"] += Z["e1,u"] * Gamma1_["uv"] * V["e1,i,v,m"];
    W["im"] += Z["E1,U"] * Gamma1_["UV"] * V["i,E1,m,V"];
    W["im"] += Z["e1,u"] * Gamma1_["uv"] * V["e1,m,v,i"];
    W["im"] += Z["E1,U"] * Gamma1_["UV"] * V["m,E1,i,V"];
    W["im"] += Z["m,n1"] * F["n1,i"];
    W["im"] += Z["m1,n1"] * V["n1,i,m1,m"];
    W["im"] += Z["M1,N1"] * V["i,N1,m,M1"];
    W["im"] += Z["uv"] * V["vium"];
    W["im"] += Z["UV"] * V["iVmU"];
    W["im"] += Z["e1,f"] * V["f,i,e1,m"];
    W["im"] += Z["E1,F"] * V["i,F,m,E1"];

    // CI contribution
    W.block("cc")("nm") += 0.5 * V.block("acac")("umvn") * x_ci("I") * cc1a_n("Iuv");
    W.block("cc")("nm") += 0.5 * V.block("acac")("umvn") * x_ci("J") * cc1a_r("Juv");
    W.block("cc")("nm") += 0.5 * V.block("cAcA")("mUnV") * x_ci("I") * cc1b_n("IUV");
    W.block("cc")("nm") += 0.5 * V.block("cAcA")("mUnV") * x_ci("J") * cc1b_r("JUV");

    W.block("ac")("xm") += 0.5 * V.block("acaa")("umvx") * x_ci("I") * cc1a_n("Iuv");
    W.block("ac")("xm") += 0.5 * V.block("acaa")("umvx") * x_ci("J") * cc1a_r("Juv");
    W.block("ac")("xm") += 0.5 * V.block("cAaA")("mUxV") * x_ci("I") * cc1b_n("IUV");
    W.block("ac")("xm") += 0.5 * V.block("cAaA")("mUxV") * x_ci("J") * cc1b_r("JUV");
    W["mu"] = W["um"];

    //NOTICE: w for {active-active}
    if (CORRELATION_TERM) {
        W["zw"] += 0.5 * Sigma3["wa"] * F["za"];
        W["zw"] += 0.5 * Sigma3["iw"] * F["iz"];
        W["zw"] += 0.5 * Sigma3["ia"] * V["aziv"] * Gamma1_["wv"];
        W["zw"] += 0.5 * Sigma3["IA"] * V["zAvI"] * Gamma1_["wv"];
        W["zw"] += 0.5 * Sigma3["ia"] * V["auiz"] * Gamma1_["uw"];
        W["zw"] += 0.5 * Sigma3["IA"] * V["uAzI"] * Gamma1_["uw"];
    }

    if (CORRELATION_TERM) {
        W["zw"] += 0.5 * Xi3["wa"] * F["za"];
        W["zw"] += 0.5 * Xi3["iw"] * F["iz"];
        W["zw"] += 0.5 * Xi3["ia"] * V["aziv"] * Gamma1_["wv"];
        W["zw"] += 0.5 * Xi3["IA"] * V["zAvI"] * Gamma1_["wv"];
        W["zw"] += 0.5 * Xi3["ia"] * V["auiz"] * Gamma1_["uw"];
        W["zw"] += 0.5 * Xi3["IA"] * V["uAzI"] * Gamma1_["uw"];
    }

    if (CORRELATION_TERM) {
        W["zw"] += Tau1["ijwb"] * V["zbij"];
        W["zw"] += 2.0 * Tau1["iJwB"] * V["zBiJ"];

        temp["klwd"] += Kappa["klwd"] * Eeps2_p["klwd"];
        temp["kLwD"] += Kappa["kLwD"] * Eeps2_p["kLwD"];
        W["zw"] += temp["klwd"] * V["zdkl"];
        W["zw"] += 2.0 * temp["kLwD"] * V["zDkL"];
        temp.zero();

        W["zw"] += Tau1["wjab"] * V["abzj"];
        W["zw"] += 2.0 * Tau1["wJaB"] * V["aBzJ"];

        temp["wlcd"] += Kappa["wlcd"] * Eeps2_p["wlcd"];
        temp["wLcD"] += Kappa["wLcD"] * Eeps2_p["wLcD"];
        W["zw"] += temp["wlcd"] * V["cdzl"];
        W["zw"] += 2.0 * temp["wLcD"] * V["cDzL"];
        temp.zero(); 
    }
    W["zw"] += Z["e1,m1"] * V["e1,u,m1,z"] * Gamma1_["uw"]; 
    W["zw"] += Z["E1,M1"] * V["u,E1,z,M1"] * Gamma1_["uw"]; 
    W["zw"] += Z["e1,m1"] * V["e1,z,m1,u"] * Gamma1_["uw"]; 
    W["zw"] += Z["E1,M1"] * V["z,E1,u,M1"] * Gamma1_["uw"];
    W["zw"] += Z["n1,w"] * F["z,n1"];
    W["zw"] += Z["n1,u"] * V["u,v,n1,z"] * Gamma1_["wv"];
    W["zw"] += Z["N1,U"] * V["v,U,z,N1"] * Gamma1_["wv"];
    W["zw"] += Z["n1,u"] * V["u,z,n1,v"] * Gamma1_["wv"];
    W["zw"] += Z["N1,U"] * V["z,U,v,N1"] * Gamma1_["wv"];   
    W["zw"] -= Z["n1,u"] * H["z,n1"] * Gamma1_["uw"];
    W["zw"] -= Z["n1,u"] * V_N_Alpha["z,n1"] * Gamma1_["uw"];
    W["zw"] -= Z["n1,u"] * V_N_Beta["z,n1"] * Gamma1_["uw"];
    W["zw"] -= 0.5 * Z["n1,u"] * V["x,y,n1,z"] * Gamma2_["u,w,x,y"];
    W["zw"] -= Z["N1,U"] * V["y,X,z,N1"] * Gamma2_["w,U,y,X"];
    W["zw"] -= Z["n1,u"] * V["z,y,n1,v"] * Gamma2_["u,v,w,y"];
    W["zw"] -= Z["n1,u"] * V["z,Y,n1,V"] * Gamma2_["u,V,w,Y"];
    W["zw"] -= Z["N1,U"] * V["z,Y,v,N1"] * Gamma2_["v,U,w,Y"];
    W["zw"] += Z["e1,u"] * H["z,e1"] * Gamma1_["uw"];
    W["zw"] += Z["e1,u"] * V_N_Alpha["z,e1"] * Gamma1_["uw"];
    W["zw"] += Z["e1,u"] * V_N_Beta["z,e1"] * Gamma1_["uw"];
    W["zw"] += 0.5 * Z["e1,u"] * V["e1,z,x,y"] * Gamma2_["u,w,x,y"];
    W["zw"] += Z["E1,U"] * V["z,E1,y,X"] * Gamma2_["w,U,y,X"];
    W["zw"] += Z["e1,u"] * V["e1,v,z,y"] * Gamma2_["u,v,w,y"];
    W["zw"] += Z["e1,u"] * V["e1,V,z,Y"] * Gamma2_["u,V,w,Y"];
    W["zw"] += Z["E1,U"] * V["v,E1,z,Y"] * Gamma2_["v,U,w,Y"];
    W["zw"] += Z["m1,n1"] * V["n1,v,m1,z"] * Gamma1_["wv"];
    W["zw"] += Z["M1,N1"] * V["v,N1,z,M1"] * Gamma1_["wv"];
    W["zw"] += Z["e1,f1"] * V["f1,v,e1,z"] * Gamma1_["wv"];
    W["zw"] += Z["E1,F1"] * V["v,F1,z,E1"] * Gamma1_["wv"];
    W["zw"] += Z["u1,a1"] * V["a1,v,u1,z"] * Gamma1_["wv"];
    W["zw"] += Z["U1,A1"] * V["v,A1,z,U1"] * Gamma1_["wv"];
    W["zw"] += Z["wv"] * F["vz"];

    W.block("aa")("zw") += 0.25 * x_ci("I") * H.block("aa")("vz") * cc1a_n("Iwv");
    W.block("aa")("zw") += 0.25 * x_ci("J") * H.block("aa")("vz") * cc1a_r("Jwv");
    W.block("aa")("zw") += 0.25 * x_ci("I") * H.block("aa")("zu") * cc1a_n("Iuw");
    W.block("aa")("zw") += 0.25 * x_ci("J") * H.block("aa")("zu") * cc1a_r("Juw");

    W.block("aa")("zw") += 0.25 * V_N_Alpha.block("aa")("uz") * cc1a_n("Iuw") * x_ci("I"); 
    W.block("aa")("zw") += 0.25 * V_N_Alpha.block("aa")("uz") * cc1a_r("Juw") * x_ci("J"); 
    W.block("aa")("zw") += 0.25 * V_N_Beta.block("aa")("uz") * cc1a_n("Iuw") * x_ci("I"); 
    W.block("aa")("zw") += 0.25 * V_N_Beta.block("aa")("uz") * cc1a_r("Juw") * x_ci("J"); 
    W.block("aa")("zw") += 0.25 * V_N_Alpha.block("aa")("zv") * cc1a_n("Iwv") * x_ci("I"); 
    W.block("aa")("zw") += 0.25 * V_N_Alpha.block("aa")("zv") * cc1a_r("Jwv") * x_ci("J"); 
    W.block("aa")("zw") += 0.25 * V_N_Beta.block("aa")("zv") * cc1a_n("Iwv") * x_ci("I"); 
    W.block("aa")("zw") += 0.25 * V_N_Beta.block("aa")("zv") * cc1a_r("Jwv") * x_ci("J"); 

    W.block("aa")("zw") += 0.0625 * x_ci("I") * V.block("aaaa")("zvxy") * cc2aa_n("Iwvxy");
    W.block("aa")("zw") += 0.1250 * x_ci("I") * V.block("aAaA")("zVxY") * cc2ab_n("IwVxY");
    W.block("aa")("zw") += 0.0625 * x_ci("I") * V.block("aaaa")("uzxy") * cc2aa_n("Iuwxy");
    W.block("aa")("zw") += 0.1250 * x_ci("I") * V.block("aAaA")("zUxY") * cc2ab_n("IwUxY");
    W.block("aa")("zw") += 0.0625 * x_ci("I") * V.block("aaaa")("uvzy") * cc2aa_n("Iuvwy");
    W.block("aa")("zw") += 0.1250 * x_ci("I") * V.block("aAaA")("uVzY") * cc2ab_n("IuVwY");
    W.block("aa")("zw") += 0.0625 * x_ci("I") * V.block("aaaa")("uvxz") * cc2aa_n("Iuvxw");
    W.block("aa")("zw") += 0.1250 * x_ci("I") * V.block("aAaA")("uVzX") * cc2ab_n("IuVwX");

    W.block("aa")("zw") += 0.0625 * x_ci("J") * V.block("aaaa")("zvxy") * cc2aa_r("Jwvxy");
    W.block("aa")("zw") += 0.1250 * x_ci("J") * V.block("aAaA")("zVxY") * cc2ab_r("JwVxY");
    W.block("aa")("zw") += 0.0625 * x_ci("J") * V.block("aaaa")("uzxy") * cc2aa_r("Juwxy");
    W.block("aa")("zw") += 0.1250 * x_ci("J") * V.block("aAaA")("zUxY") * cc2ab_r("JwUxY");
    W.block("aa")("zw") += 0.0625 * x_ci("J") * V.block("aaaa")("uvzy") * cc2aa_r("Juvwy");
    W.block("aa")("zw") += 0.1250 * x_ci("J") * V.block("aAaA")("uVzY") * cc2ab_r("JuVwY");
    W.block("aa")("zw") += 0.0625 * x_ci("J") * V.block("aaaa")("uvxz") * cc2aa_r("Juvxw");
    W.block("aa")("zw") += 0.1250 * x_ci("J") * V.block("aAaA")("uVzX") * cc2ab_r("JuVwX");

    // CASSCF reference
    BlockedTensor temp1 = BTF_->build(CoreTensor, "temporal tensor 1", spin_cases({"gg"}));

    W["mp"] += F["mp"];
    temp1["vp"] = H["vp"];
    temp1["vp"] += V_N_Alpha["vp"];
    temp1["vp"] += V_N_Beta["vp"];
    W["up"] += temp1["vp"] * Gamma1_["uv"];
    W["up"] += 0.5 * V["xypv"] * Gamma2_["uvxy"];
    W["up"] += V["xYpV"] * Gamma2_["uVxY"];

    // Copy alpha-alpha to beta-beta 
    (W.block("CC")).iterate([&](const std::vector<size_t>& i, double& value) {
        value = W.block("cc").data()[i[0] * ncore + i[1]];
    });
    (W.block("AA")).iterate([&](const std::vector<size_t>& i, double& value) {
        value = W.block("aa").data()[i[0] * na + i[1]];
    });
    (W.block("VV")).iterate([&](const std::vector<size_t>& i, double& value) {
        value = W.block("vv").data()[i[0] * nvirt + i[1]];
    });
    (W.block("CV")).iterate([&](const std::vector<size_t>& i, double& value) {
        value = W.block("cv").data()[i[0] * nvirt + i[1]];
    });
    (W.block("VC")).iterate([&](const std::vector<size_t>& i, double& value) {
        value = W.block("vc").data()[i[0] * ncore + i[1]];
    });
    (W.block("CA")).iterate([&](const std::vector<size_t>& i, double& value) {
        value = W.block("ca").data()[i[0] * na + i[1]];
    });
    (W.block("AC")).iterate([&](const std::vector<size_t>& i, double& value) {
        value = W.block("ac").data()[i[0] * ncore + i[1]];
    });
    (W.block("AV")).iterate([&](const std::vector<size_t>& i, double& value) {
        value = W.block("av").data()[i[0] * nvirt + i[1]];
    });
    (W.block("VA")).iterate([&](const std::vector<size_t>& i, double& value) {
        value = W.block("va").data()[i[0] * na + i[1]];
    });
    outfile->Printf("Done");
}


void DSRG_MRPT2::set_z_cc() {   
    BlockedTensor val1 = BTF_->build(CoreTensor, "val1", {"c"});
    BlockedTensor temp = BTF_->build(CoreTensor, "temporal tensor", spin_cases({"hhpp"}));
    BlockedTensor temp_1 = BTF_->build(CoreTensor, "temporal tensor_1", spin_cases({"hhpp"}));

    // core-core diagonal entries
    if (CORRELATION_TERM) {
        val1["m"] += -2 * s_ * Sigma1["ma"] * F["ma"];
        val1["m"] += -2 * s_ * DelGam1["xu"] * T2_["muax"] * Sigma1["ma"];
        val1["m"] += -2 * s_ * DelGam1["XU"] * T2_["mUaX"] * Sigma1["ma"];
    }
    if (CORRELATION_TERM) {
        val1["m"] -= Xi1["ma"] * F["ma"];
        val1["m"] += 2 * s_ * Xi2["ma"] * F["ma"];
        val1["m"] -= Xi1["ma"] * T2_["muax"] * DelGam1["xu"];
        val1["m"] -= Xi1["ma"] * T2_["mUaX"] * DelGam1["XU"];
        val1["m"] += 2 * s_ * Xi2["ma"] * T2_["muax"] * DelGam1["xu"];
        val1["m"] += 2 * s_ * Xi2["ma"] * T2_["mUaX"] * DelGam1["XU"];
    }
    if (CORRELATION_TERM) {
        temp["mjab"] += V["abmj"] * Eeps2["mjab"];
        temp["mJaB"] += V["aBmJ"] * Eeps2["mJaB"];
        val1["m"] += 4.0 * s_ * Tau2["mjab"] * temp["mjab"]; 
        val1["m"] += 8.0 * s_ * Tau2["mJaB"] * temp["mJaB"]; 
        temp.zero();

        val1["m"] -= 2.0 * T2OverDelta["mjab"] * Tau2["mjab"];
        val1["m"] -= 4.0 * T2OverDelta["mJaB"] * Tau2["mJaB"];

        temp["mlcd"] += V["cdml"] * Eeps2["mlcd"];
        temp["mLcD"] += V["cDmL"] * Eeps2["mLcD"];
        temp_1["mlcd"] += Kappa["mlcd"] * Delta2["mlcd"];
        temp_1["mLcD"] += Kappa["mLcD"] * Delta2["mLcD"];
        val1["m"] -= 4.0 * s_ * temp["mlcd"] * temp_1["mlcd"];
        val1["m"] -= 8.0 * s_ * temp["mLcD"] * temp_1["mLcD"];
        temp.zero();
        temp_1.zero();
    }
    BlockedTensor zmn = BTF_->build(CoreTensor, "z{mn} normal", {"cc"});
    // core-core block entries within normal conditions
    if (CORRELATION_TERM) {
        zmn["mn"] += 0.5 * Sigma3["na"] * F["ma"];
        zmn["mn"] -= 0.5 * Sigma3["ma"] * F["na"];
    }
    if (CORRELATION_TERM) {
        zmn["mn"] += 0.5 * Xi3["na"] * F["ma"];
        zmn["mn"] -= 0.5 * Xi3["ma"] * F["na"];
    }
    if (CORRELATION_TERM) {
        zmn["mn"] += Tau1["njab"] * V["abmj"];
        zmn["mn"] += 2.0 * Tau1["nJaB"] * V["aBmJ"];

        temp["nlcd"] += Kappa["nlcd"] * Eeps2_p["nlcd"];
        temp["nLcD"] += Kappa["nLcD"] * Eeps2_p["nLcD"];
        zmn["mn"] += temp["nlcd"] * V["cdml"] ; 
        zmn["mn"] += 2.0 * temp["nLcD"] * V["cDmL"];
        temp.zero(); 

        zmn["mn"] -= Tau1["mjab"] * V["abnj"];
        zmn["mn"] -= 2.0 * Tau1["mJaB"] * V["aBnJ"];

        temp["mlcd"] += Kappa["mlcd"] * Eeps2_p["mlcd"];
        temp["mLcD"] += Kappa["mLcD"] * Eeps2_p["mLcD"];
        zmn["mn"] -= temp["mlcd"] * V["cdnl"]; 
        zmn["mn"] -= 2.0 * temp["mLcD"] * V["cDnL"]; 
        temp.zero();
    }

    for (const std::string& block : {"cc", "CC"}) {
        (Z.block(block)).iterate([&](const std::vector<size_t>& i, double& value) {
            if (i[0] == i[1]) { value = val1.block("c").data()[i[0]];}
            else {
                auto dmt = Delta1.block("cc").data()[i[1] * ncore + i[0]];
                if (std::fabs(dmt) > 1e-12) { value = zmn.block("cc").data()[i[0] * ncore + i[1]] / dmt;}
            }       
        });
    }  
}

void DSRG_MRPT2::set_z_vv() {
    BlockedTensor val2 = BTF_->build(CoreTensor, "val2", {"v"});
    BlockedTensor temp = BTF_->build(CoreTensor, "temporal tensor", spin_cases({"hhpp"}));
    BlockedTensor temp_1 = BTF_->build(CoreTensor, "temporal tensor_1", spin_cases({"hhpp"}));

    // virtual-virtual diagonal entries
    if (CORRELATION_TERM) {
        val2["e"] += 2 * s_ * Sigma1["ie"] * F["ie"];
        val2["e"] += 2 * s_ * DelGam1["xu"] * T2_["iuex"] * Sigma1["ie"];
        val2["e"] += 2 * s_ * DelGam1["XU"] * T2_["iUeX"] * Sigma1["ie"];
    }

    if (CORRELATION_TERM) {
        val2["e"] += Xi1["ie"] * F["ie"];
        val2["e"] -= 2 * s_ * Xi2["ie"] * F["ie"];
        val2["e"] += Xi1["ie"] * T2_["iuex"] * DelGam1["xu"];
        val2["e"] += Xi1["ie"] * T2_["iUeX"] * DelGam1["XU"];
        val2["e"] -= 2 * s_ * Xi2["ie"] * T2_["iuex"] * DelGam1["xu"];
        val2["e"] -= 2 * s_ * Xi2["ie"] * T2_["iUeX"] * DelGam1["XU"];
    }

    if (CORRELATION_TERM) {
        temp["ijeb"] += V["ebij"] * Eeps2["ijeb"];
        temp["iJeB"] += V["eBiJ"] * Eeps2["iJeB"];
        val2["e"] -= 4.0 * s_ * Tau2["ijeb"] * temp["ijeb"]; 
        val2["e"] -= 8.0 * s_ * Tau2["iJeB"] * temp["iJeB"]; 
        temp.zero();

        val2["e"] += 2.0 * T2OverDelta["ijeb"] * Tau2["ijeb"];
        val2["e"] += 4.0 * T2OverDelta["iJeB"] * Tau2["iJeB"];

        temp["kled"] += V["edkl"] * Eeps2["kled"];
        temp["kLeD"] += V["eDkL"] * Eeps2["kLeD"];
        temp_1["kled"] += Kappa["kled"] * Delta2["kled"];
        temp_1["kLeD"] += Kappa["kLeD"] * Delta2["kLeD"];
        val2["e"] += 4.0 * s_ * temp["kled"] * temp_1["kled"];
        val2["e"] += 8.0 * s_ * temp["kLeD"] * temp_1["kLeD"];
        temp.zero();
        temp_1.zero();
    }
   
    BlockedTensor zef = BTF_->build(CoreTensor, "z{ef} normal", {"vv"});
    // virtual-virtual block entries within normal conditions
    if (CORRELATION_TERM) {
        zef["ef"] += 0.5 * Sigma3["if"] * F["ie"];
        zef["ef"] -= 0.5 * Sigma3["ie"] * F["if"];
    }

    if (CORRELATION_TERM) {
        zef["ef"] += 0.5 * Xi3["if"] * F["ie"];
        zef["ef"] -= 0.5 * Xi3["ie"] * F["if"];
    }

    if (CORRELATION_TERM) {
        zef["ef"] += Tau1["ijfb"] * V["ebij"];
        zef["ef"] += 2.0 * Tau1["iJfB"] * V["eBiJ"];

        temp["klfd"] += Kappa["klfd"] * Eeps2_p["klfd"];
        temp["kLfD"] += Kappa["kLfD"] * Eeps2_p["kLfD"];
        zef["ef"] += temp["klfd"] * V["edkl"] ; 
        zef["ef"] += 2.0 * temp["kLfD"] * V["eDkL"];
        temp.zero(); 

        zef["ef"] -= Tau1["ijeb"] * V["fbij"];
        zef["ef"] -= 2.0 * Tau1["iJeB"] * V["fBiJ"];

        temp["kled"] += Kappa["kled"] * Eeps2_p["kled"];
        temp["kLeD"] += Kappa["kLeD"] * Eeps2_p["kLeD"];
        zef["ef"] -= temp["kled"] * V["fdkl"]; 
        zef["ef"] -= 2.0 * temp["kLeD"] * V["fDkL"]; 
        temp.zero();
    }

    for (const std::string& block : {"vv", "VV"}) {
        (Z.block(block)).iterate([&](const std::vector<size_t>& i, double& value) {
            if (i[0] == i[1]) { value = val2.block("v").data()[i[0]];}
            else {
                auto dmt = Delta1.block("vv").data()[i[1] * nvirt + i[0]];
                if (std::fabs(dmt) > 1e-12) { value = zef.block("vv").data()[i[0] * nvirt + i[1]] / dmt;}
            }       
        });
    }  
}

void DSRG_MRPT2::set_z_aa_diag() {
    BlockedTensor val3 = BTF_->build(CoreTensor, "val3", {"a"});
    BlockedTensor temp = BTF_->build(CoreTensor, "temporal tensor", spin_cases({"hhpp"}));
    BlockedTensor temp_1 = BTF_->build(CoreTensor, "temporal tensor_1", spin_cases({"hhpp"}));

    // active-active diagonal entries
    if (CORRELATION_TERM) {
        val3["w"] += -2 * s_ * Sigma1["wa"] * F["wa"];
        val3["w"] += -2 * s_ * DelGam1["xu"] * T2_["wuax"] * Sigma1["wa"];
        val3["w"] += -2 * s_ * DelGam1["XU"] * T2_["wUaX"] * Sigma1["wa"];
        val3["w"] +=  2 * s_ * Sigma1["iw"] * F["iw"];
        val3["w"] +=  2 * s_ * DelGam1["xu"] * T2_["iuwx"] * Sigma1["iw"];
        val3["w"] +=  2 * s_ * DelGam1["XU"] * T2_["iUwX"] * Sigma1["iw"];

        val3["w"] += Sigma2["ia"] * T2_["iuaw"] * Gamma1_["wu"];
        val3["w"] += Sigma2["IA"] * T2_["uIwA"] * Gamma1_["wu"];
        val3["w"] -= Sigma2["ia"] * T2_["iwax"] * Gamma1_["xw"];
        val3["w"] -= Sigma2["IA"] * T2_["wIxA"] * Gamma1_["xw"];
    }

    if (CORRELATION_TERM) {
        val3["w"] -= Xi1["wa"] * F["wa"];
        val3["w"] += 2 * s_ * Xi2["wa"] * F["wa"];
        val3["w"] -= Xi1["wa"] * T2_["wuax"] * DelGam1["xu"];
        val3["w"] -= Xi1["wa"] * T2_["wUaX"] * DelGam1["XU"];
        val3["w"] += 2 * s_ * Xi2["wa"] * T2_["wuax"] * DelGam1["xu"];
        val3["w"] += 2 * s_ * Xi2["wa"] * T2_["wUaX"] * DelGam1["XU"];

        val3["w"] += Xi1["iw"] * F["iw"];
        val3["w"] -= 2 * s_ * Xi2["iw"] * F["iw"];
        val3["w"] += Xi1["iw"] * T2_["iuwx"] * DelGam1["xu"];
        val3["w"] += Xi1["iw"] * T2_["iUwX"] * DelGam1["XU"];
        val3["w"] -= 2 * s_ * Xi2["iw"] * T2_["iuwx"] * DelGam1["xu"];
        val3["w"] -= 2 * s_ * Xi2["iw"] * T2_["iUwX"] * DelGam1["XU"];

        val3["w"] += Xi3["ia"] * T2_["iuaw"] * Gamma1_["wu"];
        val3["w"] += Xi3["IA"] * T2_["uIwA"] * Gamma1_["wu"];
        val3["w"] -= Xi3["ia"] * T2_["iwax"] * Gamma1_["xw"];
        val3["w"] -= Xi3["IA"] * T2_["wIxA"] * Gamma1_["xw"];
    }

    if (CORRELATION_TERM) {
        temp["ujab"] += V["abuj"] * Eeps2["ujab"];
        temp["uJaB"] += V["aBuJ"] * Eeps2["uJaB"];
        val3["u"] += 4.0 * s_ * Tau2["ujab"] * temp["ujab"]; 
        val3["u"] += 8.0 * s_ * Tau2["uJaB"] * temp["uJaB"]; 
        temp.zero();

        val3["u"] -= 2.0 * T2OverDelta["ujab"] * Tau2["ujab"];
        val3["u"] -= 4.0 * T2OverDelta["uJaB"] * Tau2["uJaB"];

        temp["ulcd"] += V["cdul"] * Eeps2["ulcd"];
        temp["uLcD"] += V["cDuL"] * Eeps2["uLcD"];
        temp_1["ulcd"] += Kappa["ulcd"] * Delta2["ulcd"];
        temp_1["uLcD"] += Kappa["uLcD"] * Delta2["uLcD"];
        val3["u"] -= 4.0 * s_ * temp["ulcd"] * temp_1["ulcd"];
        val3["u"] -= 8.0 * s_ * temp["uLcD"] * temp_1["uLcD"];
        temp.zero();
        temp_1.zero();

        temp["ijub"] += V["ubij"] * Eeps2["ijub"];
        temp["iJuB"] += V["uBiJ"] * Eeps2["iJuB"];
        val3["u"] -= 4.0 * s_ * Tau2["ijub"] * temp["ijub"]; 
        val3["u"] -= 8.0 * s_ * Tau2["iJuB"] * temp["iJuB"]; 
        temp.zero();

        val3["u"] += 2.0 * T2OverDelta["ijub"] * Tau2["ijub"];
        val3["u"] += 4.0 * T2OverDelta["iJuB"] * Tau2["iJuB"];

        temp["klud"] += V["udkl"] * Eeps2["klud"];
        temp["kLuD"] += V["uDkL"] * Eeps2["kLuD"];
        temp_1["klud"] += Kappa["klud"] * Delta2["klud"];
        temp_1["kLuD"] += Kappa["kLuD"] * Delta2["kLuD"];
        val3["u"] += 4.0 * s_ * temp["klud"] * temp_1["klud"];
        val3["u"] += 8.0 * s_ * temp["kLuD"] * temp_1["kLuD"];
        temp.zero();
        temp_1.zero();
    }
  
    for (const std::string& block : {"aa", "AA"}) {
        (Z.block(block)).iterate([&](const std::vector<size_t>& i, double& value) {
            if (i[0] == i[1]) { value = val3.block("a").data()[i[0]];}
        });
    } 
}

void DSRG_MRPT2::set_b() {
    outfile->Printf("\n    Initializing b of the Linear System ............. ");
    Z_b = BTF_->build(CoreTensor, "b(AX=b)", spin_cases({"gg"}));
    //NOTICE: constant b for z{core-virtual}
    BlockedTensor temp = BTF_->build(CoreTensor, "temporal tensor", spin_cases({"hhpp"}));

    if (CORRELATION_TERM) {
        Z_b["em"] += 0.5 * Sigma3["ma"] * F["ea"];
        Z_b["em"] += 0.5 * Sigma3["ia"] * V["ieam"];
        Z_b["em"] += 0.5 * Sigma3["IA"] * V["eImA"];
        Z_b["em"] += 0.5 * Sigma3["ia"] * V["aeim"];
        Z_b["em"] += 0.5 * Sigma3["IA"] * V["eAmI"];
        Z_b["em"] -= 0.5 * Sigma3["ie"] * F["im"];
    }

    if (CORRELATION_TERM) {
        Z_b["em"] += 0.5 * Xi3["ma"] * F["ea"];
        Z_b["em"] += 0.5 * Xi3["ia"] * V["ieam"];
        Z_b["em"] += 0.5 * Xi3["IA"] * V["eImA"];
        Z_b["em"] += 0.5 * Xi3["ia"] * V["aeim"];
        Z_b["em"] += 0.5 * Xi3["IA"] * V["eAmI"];
        Z_b["em"] -= 0.5 * Xi3["ie"] * F["im"];
    }

    if (CORRELATION_TERM) {
        Z_b["em"] += Tau1["mjab"] * V["abej"];
        Z_b["em"] += 2.0 * Tau1["mJaB"] * V["aBeJ"];

        temp["mlcd"] += Kappa["mlcd"] * Eeps2_p["mlcd"];
        temp["mLcD"] += Kappa["mLcD"] * Eeps2_p["mLcD"];
        Z_b["em"] += temp["mlcd"] * V["cdel"];
        Z_b["em"] += 2.0 * temp["mLcD"] * V["cDeL"];
        temp.zero();

        Z_b["em"] -= Tau1["ijeb"] * V["mbij"];
        Z_b["em"] -= 2.0 * Tau1["iJeB"] * V["mBiJ"];

        temp["kled"] += Kappa["kled"] * Eeps2_p["kled"];
        temp["kLeD"] += Kappa["kLeD"] * Eeps2_p["kLeD"];
        Z_b["em"] -= temp["kled"] * V["mdkl"];
        Z_b["em"] -= 2.0 * temp["kLeD"] * V["mDkL"];
        temp.zero();
    }
    Z_b["em"] += Z["m1,n1"] * V["n1,e,m1,m"];
    Z_b["em"] += Z["M1,N1"] * V["e,N1,m,M1"];

    Z_b["em"] += Z["e1,f"] * V["f,e,e1,m"];
    Z_b["em"] += Z["E1,F"] * V["e,F,m,E1"];

    //NOTICE: constant b for z{active-active}
    if (CORRELATION_TERM) {
        Z_b["wz"] += 0.5 * Sigma3["za"] * F["wa"];
        Z_b["wz"] += 0.5 * Sigma3["iz"] * F["iw"];
        Z_b["wz"] += 0.5 * Sigma3["ia"] * V["awiv"] * Gamma1_["zv"];
        Z_b["wz"] += 0.5 * Sigma3["IA"] * V["wAvI"] * Gamma1_["zv"];
        Z_b["wz"] += 0.5 * Sigma3["ia"] * V["auiw"] * Gamma1_["uz"];
        Z_b["wz"] += 0.5 * Sigma3["IA"] * V["uAwI"] * Gamma1_["uz"];

        Z_b["wz"] -= 0.5 * Sigma3["wa"] * F["za"];
        Z_b["wz"] -= 0.5 * Sigma3["iw"] * F["iz"];
        Z_b["wz"] -= 0.5 * Sigma3["ia"] * V["aziv"] * Gamma1_["wv"];
        Z_b["wz"] -= 0.5 * Sigma3["IA"] * V["zAvI"] * Gamma1_["wv"];
        Z_b["wz"] -= 0.5 * Sigma3["ia"] * V["auiz"] * Gamma1_["uw"];
        Z_b["wz"] -= 0.5 * Sigma3["IA"] * V["uAzI"] * Gamma1_["uw"];
    }

    if (CORRELATION_TERM) {
        Z_b["wz"] += 0.5 * Xi3["za"] * F["wa"];
        Z_b["wz"] += 0.5 * Xi3["iz"] * F["iw"];
        Z_b["wz"] += 0.5 * Xi3["ia"] * V["awiv"] * Gamma1_["zv"];
        Z_b["wz"] += 0.5 * Xi3["IA"] * V["wAvI"] * Gamma1_["zv"];
        Z_b["wz"] += 0.5 * Xi3["ia"] * V["auiw"] * Gamma1_["uz"];
        Z_b["wz"] += 0.5 * Xi3["IA"] * V["uAwI"] * Gamma1_["uz"];

        Z_b["wz"] -= 0.5 * Xi3["wa"] * F["za"];
        Z_b["wz"] -= 0.5 * Xi3["iw"] * F["iz"];
        Z_b["wz"] -= 0.5 * Xi3["ia"] * V["aziv"] * Gamma1_["wv"];
        Z_b["wz"] -= 0.5 * Xi3["IA"] * V["zAvI"] * Gamma1_["wv"];
        Z_b["wz"] -= 0.5 * Xi3["ia"] * V["auiz"] * Gamma1_["uw"];
        Z_b["wz"] -= 0.5 * Xi3["IA"] * V["uAzI"] * Gamma1_["uw"];
    }

    if (CORRELATION_TERM) {
        Z_b["wz"] += Tau1["ijzb"] * V["wbij"];
        Z_b["wz"] += 2.0 * Tau1["iJzB"] * V["wBiJ"];

        temp["klzd"] += Kappa["klzd"] * Eeps2_p["klzd"];
        temp["kLzD"] += Kappa["kLzD"] * Eeps2_p["kLzD"];
        Z_b["wz"] += temp["klzd"] * V["wdkl"];
        Z_b["wz"] += 2.0 * temp["kLzD"] * V["wDkL"];
        temp.zero();

        Z_b["wz"] += Tau1["zjab"] * V["abwj"];
        Z_b["wz"] += 2.0 * Tau1["zJaB"] * V["aBwJ"];

        temp["zlcd"] += Kappa["zlcd"] * Eeps2_p["zlcd"];
        temp["zLcD"] += Kappa["zLcD"] * Eeps2_p["zLcD"];
        Z_b["wz"] += temp["zlcd"] * V["cdwl"];
        Z_b["wz"] += 2.0 * temp["zLcD"] * V["cDwL"];
        temp.zero();

        Z_b["wz"] -= Tau1["ijwb"] * V["zbij"];
        Z_b["wz"] -= 2.0 * Tau1["iJwB"] * V["zBiJ"];

        temp["klwd"] += Kappa["klwd"] * Eeps2_p["klwd"];
        temp["kLwD"] += Kappa["kLwD"] * Eeps2_p["kLwD"];
        Z_b["wz"] -= temp["klwd"] * V["zdkl"];
        Z_b["wz"] -= 2.0 * temp["kLwD"] * V["zDkL"];
        temp.zero();

        Z_b["wz"] -= Tau1["wjab"] * V["abzj"];
        Z_b["wz"] -= 2.0 * Tau1["wJaB"] * V["aBzJ"];

        temp["wlcd"] += Kappa["wlcd"] * Eeps2_p["wlcd"];
        temp["wLcD"] += Kappa["wLcD"] * Eeps2_p["wLcD"];
        Z_b["wz"] -= temp["wlcd"] * V["cdzl"];
        Z_b["wz"] -= 2.0 * temp["wLcD"] * V["cDzL"];
        temp.zero();
    }
    Z_b["wz"] += Z["m1,n1"] * V["n1,v,m1,w"] * Gamma1_["zv"];
    Z_b["wz"] += Z["M1,N1"] * V["v,N1,w,M1"] * Gamma1_["zv"];

    Z_b["wz"] += Z["e1,f1"] * V["f1,v,e1,w"] * Gamma1_["zv"];
    Z_b["wz"] += Z["E1,F1"] * V["v,F1,w,E1"] * Gamma1_["zv"];

    Z_b["wz"] -= Z["m1,n1"] * V["n1,v,m1,z"] * Gamma1_["wv"];
    Z_b["wz"] -= Z["M1,N1"] * V["v,N1,z,M1"] * Gamma1_["wv"];

    Z_b["wz"] -= Z["e1,f1"] * V["f1,v,e1,z"] * Gamma1_["wv"];
    Z_b["wz"] -= Z["E1,F1"] * V["v,F1,z,E1"] * Gamma1_["wv"];

    //NOTICE: constant b for z{virtual-active}
    if (CORRELATION_TERM) {
        Z_b["ew"] += 0.5 * Sigma3["wa"] * F["ea"];
        Z_b["ew"] += 0.5 * Sigma3["iw"] * F["ie"];
        Z_b["ew"] += 0.5 * Sigma3["ia"] * V["aeiv"] * Gamma1_["wv"];
        Z_b["ew"] += 0.5 * Sigma3["IA"] * V["eAvI"] * Gamma1_["wv"];
        Z_b["ew"] += 0.5 * Sigma3["ia"] * V["auie"] * Gamma1_["uw"];
        Z_b["ew"] += 0.5 * Sigma3["IA"] * V["uAeI"] * Gamma1_["uw"];
        Z_b["ew"] -= 0.5 * Sigma3["ie"] * F["iw"];
    }

    if (CORRELATION_TERM) {
        Z_b["ew"] += 0.5 * Xi3["wa"] * F["ea"];
        Z_b["ew"] += 0.5 * Xi3["iw"] * F["ie"];
        Z_b["ew"] += 0.5 * Xi3["ia"] * V["aeiv"] * Gamma1_["wv"];
        Z_b["ew"] += 0.5 * Xi3["IA"] * V["eAvI"] * Gamma1_["wv"];
        Z_b["ew"] += 0.5 * Xi3["ia"] * V["auie"] * Gamma1_["uw"];
        Z_b["ew"] += 0.5 * Xi3["IA"] * V["uAeI"] * Gamma1_["uw"];
        Z_b["ew"] -= 0.5 * Xi3["ie"] * F["iw"];
    }

    if (CORRELATION_TERM) {
        Z_b["ew"] += Tau1["ijwb"] * V["ebij"];
        Z_b["ew"] += 2.0 * Tau1["iJwB"] * V["eBiJ"];

        temp["klwd"] += Kappa["klwd"] * Eeps2_p["klwd"];
        temp["kLwD"] += Kappa["kLwD"] * Eeps2_p["kLwD"];
        Z_b["ew"] += temp["klwd"] * V["edkl"];
        Z_b["ew"] += 2.0 * temp["kLwD"] * V["eDkL"];
        temp.zero();

        Z_b["ew"] += Tau1["wjab"] * V["abej"];
        Z_b["ew"] += 2.0 * Tau1["wJaB"] * V["aBeJ"];

        temp["wlcd"] += Kappa["wlcd"] * Eeps2_p["wlcd"];
        temp["wLcD"] += Kappa["wLcD"] * Eeps2_p["wLcD"];
        Z_b["ew"] += temp["wlcd"] * V["cdel"];
        Z_b["ew"] += 2.0 * temp["wLcD"] * V["cDeL"];
        temp.zero();

        Z_b["ew"] -= Tau1["ijeb"] * V["wbij"];
        Z_b["ew"] -= 2.0 * Tau1["iJeB"] * V["wBiJ"];

        temp["kled"] += Kappa["kled"] * Eeps2_p["kled"];
        temp["kLeD"] += Kappa["kLeD"] * Eeps2_p["kLeD"];
        Z_b["ew"] -= temp["kled"] * V["wdkl"];
        Z_b["ew"] -= 2.0 * temp["kLeD"] * V["wDkL"];
        temp.zero();
    }
    Z_b["ew"] -= Z["e,f1"] * F["f1,w"];
    Z_b["ew"] += Z["m1,n1"] * V["m1,e,n1,v"] * Gamma1_["wv"];
    Z_b["ew"] += Z["M1,N1"] * V["e,M1,v,N1"] * Gamma1_["wv"];
    Z_b["ew"] += Z["e1,f1"] * V["e1,e,f1,v"] * Gamma1_["wv"];
    Z_b["ew"] += Z["E1,F1"] * V["e,E1,v,F1"] * Gamma1_["wv"];

    //NOTICE: constant b for z{core-active}
    if (CORRELATION_TERM) {
        Z_b["mw"] += 0.5 * Sigma3["wa"] * F["ma"];
        Z_b["mw"] += 0.5 * Sigma3["iw"] * F["im"];
        Z_b["mw"] += 0.5 * Sigma3["ia"] * V["amiv"] * Gamma1_["wv"];
        Z_b["mw"] += 0.5 * Sigma3["IA"] * V["mAvI"] * Gamma1_["wv"];
        Z_b["mw"] += 0.5 * Sigma3["ia"] * V["auim"] * Gamma1_["uw"];
        Z_b["mw"] += 0.5 * Sigma3["IA"] * V["uAmI"] * Gamma1_["uw"];

        Z_b["mw"] -= 0.5 * Sigma3["ma"] * F["wa"];
        Z_b["mw"] -= 0.5 * Sigma3["ia"] * V["amiw"];
        Z_b["mw"] -= 0.5 * Sigma3["IA"] * V["mAwI"];
        Z_b["mw"] -= 0.5 * Sigma3["ia"] * V["awim"];
        Z_b["mw"] -= 0.5 * Sigma3["IA"] * V["wAmI"];
    }

    if (CORRELATION_TERM) {
        Z_b["mw"] += 0.5 * Xi3["wa"] * F["ma"];
        Z_b["mw"] += 0.5 * Xi3["iw"] * F["im"];
        Z_b["mw"] += 0.5 * Xi3["ia"] * V["amiv"] * Gamma1_["wv"];
        Z_b["mw"] += 0.5 * Xi3["IA"] * V["mAvI"] * Gamma1_["wv"];
        Z_b["mw"] += 0.5 * Xi3["ia"] * V["auim"] * Gamma1_["uw"];
        Z_b["mw"] += 0.5 * Xi3["IA"] * V["uAmI"] * Gamma1_["uw"];

        Z_b["mw"] -= 0.5 * Xi3["ma"] * F["wa"];
        Z_b["mw"] -= 0.5 * Xi3["ia"] * V["amiw"];
        Z_b["mw"] -= 0.5 * Xi3["IA"] * V["mAwI"];
        Z_b["mw"] -= 0.5 * Xi3["ia"] * V["awim"];
        Z_b["mw"] -= 0.5 * Xi3["IA"] * V["wAmI"];
    }

    if (CORRELATION_TERM) {
        Z_b["mw"] += Tau1["ijwb"] * V["mbij"];
        Z_b["mw"] += 2.0 * Tau1["iJwB"] * V["mBiJ"];

        temp["klwd"] += Kappa["klwd"] * Eeps2_p["klwd"];
        temp["kLwD"] += Kappa["kLwD"] * Eeps2_p["kLwD"];
        Z_b["mw"] += temp["klwd"] * V["mdkl"];
        Z_b["mw"] += 2.0 * temp["kLwD"] * V["mDkL"];
        temp.zero();

        Z_b["mw"] += Tau1["wjab"] * V["abmj"];
        Z_b["mw"] += 2.0 * Tau1["wJaB"] * V["aBmJ"];

        temp["wlcd"] += Kappa["wlcd"] * Eeps2_p["wlcd"];
        temp["wLcD"] += Kappa["wLcD"] * Eeps2_p["wLcD"];
        Z_b["mw"] += temp["wlcd"] * V["cdml"];
        Z_b["mw"] += 2.0 * temp["wLcD"] * V["cDmL"];
        temp.zero();

        Z_b["mw"] -= Tau1["mjab"] * V["abwj"];
        Z_b["mw"] -= 2.0 * Tau1["mJaB"] * V["aBwJ"];

        temp["mlcd"] += Kappa["mlcd"] * Eeps2_p["mlcd"];
        temp["mLcD"] += Kappa["mLcD"] * Eeps2_p["mLcD"];
        Z_b["mw"] -= temp["mlcd"] * V["cdwl"];
        Z_b["mw"] -= 2.0 * temp["mLcD"] * V["cDwL"];
        temp.zero();
    }
    Z_b["mw"] += Z["m1,n1"] * V["n1,v,m1,m"] * Gamma1_["wv"];
    Z_b["mw"] += Z["M1,N1"] * V["v,N1,m,M1"] * Gamma1_["wv"];

    Z_b["mw"] += Z["e1,f1"] * V["f1,v,e1,m"] * Gamma1_["wv"];
    Z_b["mw"] += Z["E1,F1"] * V["v,F1,m,E1"] * Gamma1_["wv"];

    Z_b["mw"] -= Z["m,n1"] * F["n1,w"];

    Z_b["mw"] -= Z["m1,n1"] * V["n1,w,m1,m"];
    Z_b["mw"] -= Z["M1,N1"] * V["w,N1,m,M1"];

    Z_b["mw"] -= Z["e1,f"] * V["f,w,e1,m"];
    Z_b["mw"] -= Z["E1,F"] * V["w,F,m,E1"];
}

void DSRG_MRPT2::solve_z() {
    set_b();

    int dim_vc = nvirt * ncore,
        dim_ca = ncore * na,
        dim_va = nvirt * na,
        dim_aa = na * (na - 1) / 2,
        dim_ci = ndets;
        // dim_ci = 0;
    int dim = dim_vc + dim_ca + dim_va + dim_aa + dim_ci;
    int N=dim;
    int NRHS=1, LDA=N,LDB=N;
    int n = N, nrhs = NRHS, lda = LDA, ldb = LDB;
    std::vector<int> ipiv(N);

    std::vector<double> A(dim * dim);
    std::vector<double> b(dim);

    std::map<string,int> preidx = {
        {"vc", 0}, {"VC", 0}, {"ca", dim_vc}, {"CA", dim_vc},
        {"va", dim_vc + dim_ca}, {"VA", dim_vc + dim_ca},
        {"aa", dim_vc + dim_ca + dim_va}, {"AA", dim_vc + dim_ca + dim_va},
        {"ci", dim_vc + dim_ca + dim_va + dim_aa} 
    };

    std::map<string,int> block_dim = {
        {"vc", ncore}, {"VC", ncore}, {"ca", na}, {"CA", na},
        {"va", na}, {"VA", na}, {"aa", 0}, {"AA", 0}
    };

    //NOTICE:b
    BlockedTensor temp2 = BTF_->build(CoreTensor, "temporal tensor 2", {"gg"});
    BlockedTensor temp3 = BTF_->build(CoreTensor, "temporal tensor 3", {"aa","AA"});

    temp3["uv"] += Z["uv"] * I["uv"];
    temp3["UV"] += Z["UV"] * I["UV"];

    // VIRTUAL-CORE
    temp2["em"] += Z_b["em"];
    temp2["em"] += temp3["uv"] * V["veum"];
    temp2["em"] += temp3["UV"] * V["eVmU"];

    // CORE-ACTIVE
    temp2["mw"] += Z_b["mw"];
    temp2["mw"] += temp3["u1,a1"] * V["a1,v,u1,m"] * Gamma1_["wv"];
    temp2["mw"] += temp3["U1,A1"] * V["v,A1,m,U1"] * Gamma1_["wv"];
    temp2["mw"] += temp3["wv"] * F["vm"];
    temp2["mw"] -= temp3["uv"] * V["vwum"];
    temp2["mw"] -= temp3["UV"] * V["wVmU"];

    // VIRTUAL-ACTIVE
    temp2["ew"] += Z_b["ew"];
    temp2["ew"] += temp3["u1,a1"] * V["u1,e,a1,v"] * Gamma1_["wv"];
    temp2["ew"] += temp3["U1,A1"] * V["e,U1,v,A1"] * Gamma1_["wv"];
    temp2["ew"] += temp3["wv"] * F["ve"];

    // ACTIVE-ACTIVE
    temp2["wz"] += Z_b["wz"];
    temp2["wz"] += temp3["a1,u1"] * V["u1,v,a1,w"] * Gamma1_["zv"];
    temp2["wz"] += temp3["A1,U1"] * V["v,U1,w,A1"] * Gamma1_["zv"];
    temp2["wz"] -= temp3["a1,u1"] * V["u1,v,a1,z"] * Gamma1_["wv"];
    temp2["wz"] -= temp3["A1,U1"] * V["v,U1,z,A1"] * Gamma1_["wv"];

    for (const std::string& block : {"vc", "ca", "va", "aa"}) {
        (temp2.block(block)).iterate([&](const std::vector<size_t>& i, double& value) {
            if (block != "aa") {
                int index = preidx[block] + i[0] * block_dim[block] + i[1];
                b.at(index) = value;
            }
            else if (block == "aa" && i[0] > i[1]) {
                int index = preidx[block] + i[0] * (i[0] - 1) / 2 + i[1];
                b.at(index) = value;
            }    
        });
    } 

    outfile->Printf("Done");
    outfile->Printf("\n    Initializing A of the Linear System ............. ");

    //NOTICE: Linear system A
    BlockedTensor temp1 = BTF_->build(CoreTensor, "temporal tensor 1", 
        {"vcvc","vcca", "vcva", "vcaa", 
         "cavc","caca", "cava", "caaa",
         "vavc","vaca", "vava", "vaaa",
         "aavc","aaca", "aava", "aaaa",
         "vcVC","vcCA", "vcVA", "vcAA", 
         "caVC","caCA", "caVA", "caAA",
         "vaVC","vaCA", "vaVA", "vaAA",
         "aaVC","aaCA", "aaVA", "aaAA",});

    // VIRTUAL-CORE
    temp1["e,m,e1,m1"] += Delta1["m1,e1"] * I["e1,e"] * I["m1,m"];

    temp1["e,m,e1,m1"] -= V["e1,e,m1,m"];
    temp1["e,m,E1,M1"] -= V["e,E1,m,M1"];
    temp1["e,m,e1,m1"] -= V["m1,e,e1,m"];
    temp1["e,m,E1,M1"] -= V["e,M1,m,E1"];

    temp1["e,m,m1,u"] -= F["ue"] * I["m,m1"];

    temp1["e,m,n1,u"] -= V["u,e,n1,m"];
    temp1["e,m,N1,U"] -= V["e,U,m,N1"];
    temp1["e,m,n1,u"] -= V["n1,e,u,m"];
    temp1["e,m,N1,U"] -= V["e,N1,m,U"];

    temp1["e,m,n1,u"] += Gamma1_["uv"] * V["v,e,n1,m"];
    temp1["e,m,N1,U"] += Gamma1_["UV"] * V["e,V,m,N1"];
    temp1["e,m,n1,u"] += Gamma1_["uv"] * V["n1,e,v,m"];
    temp1["e,m,N1,U"] += Gamma1_["UV"] * V["e,N1,m,V"];

    temp1["e,m,e1,u"] -= Gamma1_["uv"] * V["e1,e,v,m"];
    temp1["e,m,E1,U"] -= Gamma1_["UV"] * V["e,E1,m,V"];
    temp1["e,m,e1,u"] -= Gamma1_["uv"] * V["v,e,e1,m"];
    temp1["e,m,E1,U"] -= Gamma1_["UV"] * V["e,V,m,E1"];

    temp1["e,m,e1,u"] += F["um"] * I["e,e1"];

    temp1["e,m,u,v"] -= V["veum"];
    temp1["e,m,U,V"] -= V["eVmU"];

    // CORE-ACTIVE
    temp1["m,w,e1,m1"] += F["w,e1"] * I["m,m1"];
    temp1["m,w,e1,m1"] += V["e1,w,m1,m"];
    temp1["m,w,E1,M1"] += V["w,E1,m,M1"];
    temp1["m,w,e1,m1"] += V["e1,m,m1,w"];
    temp1["m,w,E1,M1"] += V["m,E1,w,M1"];

    temp1["m,w,m1,u"] += F["uw"] * I["m1,m"];
    temp1["m,w,m1,u"] -= H["vw"] * Gamma1_["uv"] * I["m1,m"];
    temp1["m,w,m1,u"] -= V["v,n1,w,n"] * Gamma1_["uv"] * I["n,n1"] * I["m,m1"];
    temp1["m,w,m1,u"] -= V["v,N1,w,N"] * Gamma1_["uv"] * I["N,N1"] * I["m,m1"];
    temp1["m,w,m1,u"] -= 0.5 * V["xywv"] * Gamma2_["uvxy"] * I["m,m1"];
    temp1["m,w,m1,u"] -= V["xYwV"] * Gamma2_["uVxY"] * I["m,m1"];

    temp1["m,w,n1,u"] += V["u,w,n1,m"];
    temp1["m,w,N1,U"] += V["w,U,m,N1"];
    temp1["m,w,n1,u"] += V["u,m,n1,w"];
    temp1["m,w,N1,U"] += V["m,U,w,N1"];

    temp1["m,w,n1,u"] -= Gamma1_["uv"] * V["v,w,n1,m"];
    temp1["m,w,N1,U"] -= Gamma1_["UV"] * V["w,V,m,N1"];
    temp1["m,w,n1,u"] -= Gamma1_["uv"] * V["v,m,n1,w"];
    temp1["m,w,N1,U"] -= Gamma1_["UV"] * V["m,V,w,N1"];

    temp1["m,w,e1,u"] += Gamma1_["uv"] * V["e1,w,v,m"];
    temp1["m,w,E1,U"] += Gamma1_["UV"] * V["w,E1,m,V"];
    temp1["m,w,e1,u"] += Gamma1_["uv"] * V["e1,m,v,w"];
    temp1["m,w,E1,U"] += Gamma1_["UV"] * V["m,E1,w,V"];

    temp1["m,w,u,v"] += V["vwum"];
    temp1["m,w,U,V"] += V["wVmU"];

    temp1["m,w,e1,m1"] -= V["e1,u,m1,m"] * Gamma1_["uw"];
    temp1["m,w,E1,M1"] -= V["u,E1,m,M1"] * Gamma1_["uw"];
    temp1["m,w,e1,m1"] -= V["e1,m,m1,u"] * Gamma1_["uw"];
    temp1["m,w,E1,M1"] -= V["m,E1,u,M1"] * Gamma1_["uw"];

    temp1["m,w,n1,w1"] -= F["m,n1"] * I["w,w1"];

    temp1["m,w,n1,u"] -= V["u,v,n1,m"] * Gamma1_["wv"];
    temp1["m,w,N1,U"] -= V["v,U,m,N1"] * Gamma1_["wv"];
    temp1["m,w,n1,u"] -= V["u,m,n1,v"] * Gamma1_["wv"];
    temp1["m,w,N1,U"] -= V["m,U,v,N1"] * Gamma1_["wv"];

    temp1["m,w,n1,u"] += H["m,n1"] * Gamma1_["uw"];
    temp1["m,w,n1,u"] += V["m,m2,n1,n2"] * Gamma1_["uw"] * I["m2,n2"];
    temp1["m,w,n1,u"] += V["m,M2,n1,N2"] * Gamma1_["uw"] * I["M2,N2"];
    temp1["m,w,n1,u"] += 0.5 * V["x,y,n1,m"] * Gamma2_["u,w,x,y"];
    temp1["m,w,N1,U"] += V["y,X,m,N1"] * Gamma2_["w,U,y,X"];
    temp1["m,w,n1,u"] += V["m,y,n1,v"] * Gamma2_["u,v,w,y"];
    temp1["m,w,n1,u"] += V["m,Y,n1,V"] * Gamma2_["u,V,w,Y"];
    temp1["m,w,N1,U"] += V["m,Y,v,N1"] * Gamma2_["v,U,w,Y"];

    temp1["m,w,e1,u"] -= H["m,e1"] * Gamma1_["uw"];
    temp1["m,w,e1,u"] -= V["e1,n2,m,m2"] * Gamma1_["uw"] * I["m2,n2"];
    temp1["m,w,e1,u"] -= V["e1,N2,m,M2"] * Gamma1_["uw"] * I["M2,N2"];
    temp1["m,w,e1,u"] -= 0.5 * V["e1,m,x,y"] * Gamma2_["u,w,x,y"];
    temp1["m,w,E1,U"] -= V["m,E1,y,X"] * Gamma2_["w,U,y,X"];
    temp1["m,w,e1,u"] -= V["e1,v,m,y"] * Gamma2_["u,v,w,y"];
    temp1["m,w,e1,u"] -= V["e1,V,m,Y"] * Gamma2_["u,V,w,Y"];
    temp1["m,w,E1,U"] -= V["v,E1,m,Y"] * Gamma2_["v,U,w,Y"];

    temp1["m,w,u1,a1"] -= V["a1,v,u1,m"] * Gamma1_["wv"];
    temp1["m,w,U1,A1"] -= V["v,A1,m,U1"] * Gamma1_["wv"];

    temp1["m,w,w1,v"] -= F["vm"] * I["w,w1"];

    // VIRTUAL-ACTIVE
    temp1["e,w,e1,m1"] += F["m1,w"] * I["e,e1"];

    temp1["e,w,e1,u"] += H["vw"] * Gamma1_["uv"] * I["e,e1"];
    temp1["e,w,e1,u"] += V["v,m,w,m1"] * Gamma1_["uv"] * I["m,m1"] * I["e,e1"];
    temp1["e,w,e1,u"] += V["v,M,w,M1"] * Gamma1_["uv"] * I["M,M1"] * I["e,e1"];
    temp1["e,w,e1,u"] += 0.5 * V["xywv"] * Gamma2_["uvxy"] * I["e,e1"];
    temp1["e,w,e1,u"] += V["xYwV"] * Gamma2_["uVxY"] * I["e,e1"];

    temp1["e,w,e1,m1"] -= V["e1,u,m1,e"] * Gamma1_["uw"];
    temp1["e,w,E1,M1"] -= V["u,E1,e,M1"] * Gamma1_["uw"];
    temp1["e,w,e1,m1"] -= V["e1,e,m1,u"] * Gamma1_["uw"];
    temp1["e,w,E1,M1"] -= V["e,E1,u,M1"] * Gamma1_["uw"];

    temp1["e,w,n1,u"] -= V["u,v,n1,e"] * Gamma1_["wv"];
    temp1["e,w,N1,U"] -= V["v,U,e,N1"] * Gamma1_["wv"];
    temp1["e,w,n1,u"] -= V["u,e,n1,v"] * Gamma1_["wv"];
    temp1["e,w,N1,U"] -= V["e,U,v,N1"] * Gamma1_["wv"];

    temp1["e,w,n1,u"] += H["e,n1"] * Gamma1_["uw"];
    temp1["e,w,n1,u"] += V["e,m,n1,m1"] * Gamma1_["uw"] * I["m,m1"];
    temp1["e,w,n1,u"] += V["e,M,n1,M1"] * Gamma1_["uw"] * I["M,M1"];
    temp1["e,w,n1,u"] += 0.5 * V["x,y,n1,e"] * Gamma2_["u,w,x,y"];
    temp1["e,w,N1,U"] += V["y,X,e,N1"] * Gamma2_["w,U,y,X"];
    temp1["e,w,n1,u"] += V["e,y,n1,v"] * Gamma2_["u,v,w,y"];
    temp1["e,w,n1,u"] += V["e,Y,n1,V"] * Gamma2_["u,V,w,Y"];
    temp1["e,w,N1,U"] += V["e,Y,v,N1"] * Gamma2_["v,U,w,Y"];

    temp1["e,w,u1,a1"] -= V["u1,e,a1,v"] * Gamma1_["wv"];
    temp1["e,w,U1,A1"] -= V["e,U1,v,A1"] * Gamma1_["wv"];

    temp1["e,w,w1,v"] -= F["ve"] * I["w,w1"];

    temp1["e,w,e1,u"] -= H["e,e1"] * Gamma1_["uw"];
    temp1["e,w,e1,u"] -= V["e,m,e1,m1"] * Gamma1_["uw"] * I["m,m1"];
    temp1["e,w,e1,u"] -= V["e,M,e1,M1"] * Gamma1_["uw"] * I["M,M1"];
    temp1["e,w,e1,u"] -= 0.5 * V["e1,e,x,y"] * Gamma2_["u,w,x,y"];
    temp1["e,w,E1,U"] -= V["e,E1,y,X"] * Gamma2_["w,U,y,X"];
    temp1["e,w,e1,u"] -= V["e,y,e1,v"] * Gamma2_["u,v,w,y"];
    temp1["e,w,e1,u"] -= V["e,Y,e1,V"] * Gamma2_["u,V,w,Y"];
    temp1["e,w,E1,U"] -= V["e,Y,v,E1"] * Gamma2_["v,U,w,Y"];

    // ACTIVE-ACTIVE
    temp1["w,z,w1,z1"] += Delta1["zw"] * I["w,w1"] * I["z,z1"];

    temp1["w,z,e1,m1"] -= V["e1,u,m1,w"] * Gamma1_["uz"];
    temp1["w,z,E1,M1"] -= V["u,E1,w,M1"] * Gamma1_["uz"];
    temp1["w,z,e1,m1"] -= V["e1,w,m1,u"] * Gamma1_["uz"];
    temp1["w,z,E1,M1"] -= V["w,E1,u,M1"] * Gamma1_["uz"];

    temp1["w,z,e1,m1"] += V["e1,u,m1,z"] * Gamma1_["uw"];
    temp1["w,z,E1,M1"] += V["u,E1,z,M1"] * Gamma1_["uw"];
    temp1["w,z,e1,m1"] += V["e1,z,m1,u"] * Gamma1_["uw"];
    temp1["w,z,E1,M1"] += V["z,E1,u,M1"] * Gamma1_["uw"];

    temp1["w,z,n1,z1"] -= F["w,n1"] * I["z,z1"];
    temp1["w,z,n1,w1"] += F["z,n1"] * I["w,w1"];

    temp1["w,z,n1,u"] -= V["u,v,n1,w"] * Gamma1_["zv"];
    temp1["w,z,N1,U"] -= V["v,U,w,N1"] * Gamma1_["zv"];
    temp1["w,z,n1,u"] -= V["u,w,n1,v"] * Gamma1_["zv"];
    temp1["w,z,N1,U"] -= V["w,U,v,N1"] * Gamma1_["zv"];

    temp1["w,z,n1,u"] += V["u,v,n1,z"] * Gamma1_["wv"];
    temp1["w,z,N1,U"] += V["v,U,z,N1"] * Gamma1_["wv"];
    temp1["w,z,n1,u"] += V["u,z,n1,v"] * Gamma1_["wv"];
    temp1["w,z,N1,U"] += V["z,U,v,N1"] * Gamma1_["wv"];

    temp1["w,z,n1,u"] += H["w,n1"] * Gamma1_["uz"];
    temp1["w,z,n1,u"] += V["w,m1,n1,m"] * Gamma1_["uz"] * I["m1,m"];
    temp1["w,z,n1,u"] += V["w,M1,n1,M"] * Gamma1_["uz"] * I["M1,M"];
    temp1["w,z,n1,u"] += 0.5 * V["x,y,n1,w"] * Gamma2_["u,z,x,y"];
    temp1["w,z,N1,U"] += V["y,X,w,N1"] * Gamma2_["z,U,y,X"];
    temp1["w,z,n1,u"] += V["w,y,n1,v"] * Gamma2_["u,v,z,y"];
    temp1["w,z,n1,u"] += V["w,Y,n1,V"] * Gamma2_["u,V,z,Y"];
    temp1["w,z,N1,U"] += V["w,Y,v,N1"] * Gamma2_["v,U,z,Y"];

    temp1["w,z,n1,u"] -= H["z,n1"] * Gamma1_["uw"];
    temp1["w,z,n1,u"] -= V["z,m1,n1,m"] * Gamma1_["uw"] * I["m1,m"];
    temp1["w,z,n1,u"] -= V["z,M1,n1,M"] * Gamma1_["uw"] * I["M1,M"];
    temp1["w,z,n1,u"] -= 0.5 * V["x,y,n1,z"] * Gamma2_["u,w,x,y"];
    temp1["w,z,N1,U"] -= V["y,X,z,N1"] * Gamma2_["w,U,y,X"];
    temp1["w,z,n1,u"] -= V["z,y,n1,v"] * Gamma2_["u,v,w,y"];
    temp1["w,z,n1,u"] -= V["z,Y,n1,V"] * Gamma2_["u,V,w,Y"];
    temp1["w,z,N1,U"] -= V["z,Y,v,N1"] * Gamma2_["v,U,w,Y"];

    temp1["w,z,e1,u"] -= H["w,e1"] * Gamma1_["uz"];
    temp1["w,z,e1,u"] -= V["e1,m,w,m1"] * Gamma1_["uz"] * I["m1,m"];
    temp1["w,z,e1,u"] -= V["e1,M,w,M1"] * Gamma1_["uz"] * I["M1,M"];
    temp1["w,z,e1,u"] -= 0.5 * V["e1,w,x,y"] * Gamma2_["u,z,x,y"];
    temp1["w,z,E1,U"] -= V["w,E1,y,X"] * Gamma2_["z,U,y,X"];
    temp1["w,z,e1,u"] -= V["e1,v,w,y"] * Gamma2_["u,v,z,y"];
    temp1["w,z,e1,u"] -= V["e1,V,w,Y"] * Gamma2_["u,V,z,Y"];
    temp1["w,z,E1,U"] -= V["v,E1,w,Y"] * Gamma2_["v,U,z,Y"];
    
    temp1["w,z,e1,u"] += H["z,e1"] * Gamma1_["uw"];
    temp1["w,z,e1,u"] += V["e1,m,z,m1"] * Gamma1_["uw"] * I["m1,m"];
    temp1["w,z,e1,u"] += V["e1,M,z,M1"] * Gamma1_["uw"] * I["M1,M"];
    temp1["w,z,e1,u"] += 0.5 * V["e1,z,x,y"] * Gamma2_["u,w,x,y"];
    temp1["w,z,E1,U"] += V["z,E1,y,X"] * Gamma2_["w,U,y,X"];
    temp1["w,z,e1,u"] += V["e1,v,z,y"] * Gamma2_["u,v,w,y"];
    temp1["w,z,e1,u"] += V["e1,V,z,Y"] * Gamma2_["u,V,w,Y"];
    temp1["w,z,E1,U"] += V["v,E1,z,Y"] * Gamma2_["v,U,w,Y"];

    temp1["w,z,u1,a1"] -= V["a1,v,u1,w"] * Gamma1_["zv"];
    temp1["w,z,U1,A1"] -= V["v,A1,w,U1"] * Gamma1_["zv"];

    temp1["w,z,u1,a1"] += V["a1,v,u1,z"] * Gamma1_["wv"];
    temp1["w,z,U1,A1"] += V["v,A1,z,U1"] * Gamma1_["wv"];

    for (const std::string& row : {"vc","ca","va","aa"}) {
        int idx1 = block_dim[row];
        int pre1 = preidx[row];

        for (const std::string& col : {"vc","VC","ca","CA","va","VA","aa","AA"}) {
            int idx2 = block_dim[col];
            int pre2 = preidx[col];
            if (row != "aa" && col != "aa" && col != "AA") {
                (temp1.block(row + col)).iterate([&](const std::vector<size_t>& i, double& value) {
                    int index = (pre1 + i[0] * idx1 + i[1]) 
                                 + dim * (pre2 + i[2] * idx2 + i[3]);
                    A.at(index) += value;
                });
            }
            else if (row == "aa" && col != "aa" && col != "AA") {       
                (temp1.block(row + col)).iterate([&](const std::vector<size_t>& i, double& value) {
                    if (i[0] > i[1] ) {
                        int index = (pre1 + i[0] * (i[0] - 1) / 2 + i[1])
                                     + dim * (pre2 + i[2] * idx2 + i[3]); 
                        A.at(index) += value;    
                    }
                });
            }
            else if (row != "aa" && (col == "aa" || col == "AA")) {
                (temp1.block(row + col)).iterate([&](const std::vector<size_t>& i, double& value) {
                    int i2 = i[2] > i[3]? i[2]: i[3],
                        i3 = i[2] > i[3]? i[3]: i[2];
                    if (i2 != i3 ) {
                        int index = (pre1 + i[0] * idx1 + i[1]) 
                                     + dim * (pre2 + i2 * (i2 - 1) / 2 + i3);
                        A.at(index) += value;    
                    }
                });   
            }
            else {
                (temp1.block(row + col)).iterate([&](const std::vector<size_t>& i, double& value) {
                    int i2 = i[2] > i[3]? i[2]: i[3],
                        i3 = i[2] > i[3]? i[3]: i[2];
                    if (i[0] > i[1] && i2 != i3) {
                        int index = (pre1 + i[0] * (i[0] - 1) / 2 + i[1])
                                     + dim * (pre2 + i2 * (i2 - 1) / 2 + i3);
                        A.at(index) += value;    
                    }       
                });
            }
        }
    } 

    // CI contribution
    auto ci_vc = ambit::Tensor::build(ambit::CoreTensor, "ci contribution to z{vc}", {ndets, nvirt, ncore});
    auto ci_ca = ambit::Tensor::build(ambit::CoreTensor, "ci contribution to z{ca}", {ndets, ncore, na});
    auto ci_va = ambit::Tensor::build(ambit::CoreTensor, "ci contribution to z{va}", {ndets, nvirt, na});
    auto ci_aa = ambit::Tensor::build(ambit::CoreTensor, "ci contribution to z{aa}", {ndets, na, na});

    // CI contribution to Z{VC}
    ci_vc("Iem") -= H.block("vc")("em") * ci("I");
    ci_vc("Iem") -= V_N_Alpha.block("cv")("me") * ci("I");
    ci_vc("Iem") -= V_N_Beta.block("cv")("me") * ci("I");
    ci_vc("Iem") -= 0.5 * V.block("avac")("veum") * cc1a_n("Iuv");
    ci_vc("Iem") -= 0.5 * V.block("vAcA")("eVmU") * cc1b_n("IUV");
    ci_vc("Iem") -= 0.5 * V.block("avac")("veum") * cc1a_r("Iuv");
    ci_vc("Iem") -= 0.5 * V.block("vAcA")("eVmU") * cc1b_r("IUV");

    // CI contribution to Z{CA}
    ci_ca("Imw") -= 0.25 * H.block("ac")("vm") * cc1a_n("Iwv");
    ci_ca("Imw") -= 0.25 * H.block("ca")("mu") * cc1a_n("Iuw");
    ci_ca("Imw") -= 0.25 * H.block("ac")("vm") * cc1a_r("Iwv");
    ci_ca("Imw") -= 0.25 * H.block("ca")("mu") * cc1a_r("Iuw");
    ci_ca("Imw") -= 0.25 * V_N_Alpha.block("ac")("um") * cc1a_n("Iuw");
    ci_ca("Imw") -= 0.25 * V_N_Beta.block("ac")("um")  * cc1a_n("Iuw");
    ci_ca("Imw") -= 0.25 * V_N_Alpha.block("ac")("um") * cc1a_r("Iuw");
    ci_ca("Imw") -= 0.25 * V_N_Beta.block("ac")("um")  * cc1a_r("Iuw");
    ci_ca("Imw") -= 0.25 * V_N_Alpha.block("ca")("mv") * cc1a_n("Iwv"); 
    ci_ca("Imw") -= 0.25 * V_N_Beta.block("ca")("mv")  * cc1a_n("Iwv"); 
    ci_ca("Imw") -= 0.25 * V_N_Alpha.block("ca")("mv") * cc1a_r("Iwv"); 
    ci_ca("Imw") -= 0.25 * V_N_Beta.block("ca")("mv")  * cc1a_r("Iwv"); 
    ci_ca("Imw") -= 0.0625 * V.block("aaca")("xymv") * cc2aa_n("Iwvxy");
    ci_ca("Imw") -= 0.1250 * V.block("aAcA")("xYmV") * cc2ab_n("IwVxY");
    ci_ca("Imw") -= 0.0625 * V.block("aaac")("xyum") * cc2aa_n("Iuwxy");
    ci_ca("Imw") -= 0.1250 * V.block("aAcA")("xYmU") * cc2ab_n("IwUxY");
    ci_ca("Imw") -= 0.0625 * V.block("aaca")("uvmy") * cc2aa_n("Iuvwy");
    ci_ca("Imw") -= 0.1250 * V.block("aAcA")("uVmY") * cc2ab_n("IuVwY");
    ci_ca("Imw") -= 0.0625 * V.block("aaac")("uvxm") * cc2aa_n("Iuvxw");
    ci_ca("Imw") -= 0.1250 * V.block("aAcA")("uVmX") * cc2ab_n("IuVwX");
    ci_ca("Jmw") -= 0.0625 * V.block("aaca")("xymv") * cc2aa_r("Jwvxy");
    ci_ca("Jmw") -= 0.1250 * V.block("aAcA")("xYmV") * cc2ab_r("JwVxY");
    ci_ca("Jmw") -= 0.0625 * V.block("aaac")("xyum") * cc2aa_r("Juwxy");
    ci_ca("Jmw") -= 0.1250 * V.block("aAcA")("xYmU") * cc2ab_r("JwUxY");
    ci_ca("Jmw") -= 0.0625 * V.block("aaca")("uvmy") * cc2aa_r("Juvwy");
    ci_ca("Jmw") -= 0.1250 * V.block("aAcA")("uVmY") * cc2ab_r("JuVwY");
    ci_ca("Jmw") -= 0.0625 * V.block("aaac")("uvxm") * cc2aa_r("Juvxw");
    ci_ca("Jmw") -= 0.1250 * V.block("aAcA")("uVmX") * cc2ab_r("JuVwX");

    ci_ca("Imw") += H.block("ac")("wm") * ci("I");
    ci_ca("Imw") += V_N_Alpha.block("ca")("mw") * ci("I");
    ci_ca("Imw") += V_N_Beta.block("ca")("mw") * ci("I");
    ci_ca("Imw") += 0.5 * V.block("aaac")("vwum") * cc1a_n("Iuv");
    ci_ca("Imw") += 0.5 * V.block("aAcA")("wVmU") * cc1b_n("IUV");
    ci_ca("Imw") += 0.5 * V.block("aaac")("vwum") * cc1a_r("Iuv");
    ci_ca("Imw") += 0.5 * V.block("aAcA")("wVmU") * cc1b_r("IUV");

    // CI contribution to Z{VA}
    ci_va("Iew") -= 0.25 * H.block("av")("ve") * cc1a_n("Iwv");
    ci_va("Iew") -= 0.25 * H.block("va")("eu") * cc1a_n("Iuw");
    ci_va("Iew") -= 0.25 * H.block("av")("ve") * cc1a_r("Iwv");
    ci_va("Iew") -= 0.25 * H.block("va")("eu") * cc1a_r("Iuw");

    ci_va("Iew") -= 0.25 * V_N_Alpha.block("av")("ue") * cc1a_n("Iuw");
    ci_va("Iew") -= 0.25 * V_N_Beta.block("av")("ue")  * cc1a_n("Iuw");
    ci_va("Iew") -= 0.25 * V_N_Alpha.block("va")("ev") * cc1a_n("Iwv");
    ci_va("Iew") -= 0.25 * V_N_Beta.block("va")("ev")  * cc1a_n("Iwv");
    ci_va("Iew") -= 0.25 * V_N_Alpha.block("av")("ue") * cc1a_r("Iuw");
    ci_va("Iew") -= 0.25 * V_N_Beta.block("av")("ue")  * cc1a_r("Iuw");
    ci_va("Iew") -= 0.25 * V_N_Alpha.block("va")("ev") * cc1a_r("Iwv");
    ci_va("Iew") -= 0.25 * V_N_Beta.block("va")("ev")  * cc1a_r("Iwv");

    ci_va("Iew") -= 0.0625 * V.block("vaaa")("evxy") * cc2aa_n("Iwvxy");
    ci_va("Iew") -= 0.1250 * V.block("vAaA")("eVxY") * cc2ab_n("IwVxY");
    ci_va("Iew") -= 0.0625 * V.block("avaa")("uexy") * cc2aa_n("Iuwxy");
    ci_va("Iew") -= 0.1250 * V.block("vAaA")("eUxY") * cc2ab_n("IwUxY");
    ci_va("Iew") -= 0.0625 * V.block("vaaa")("eyuv") * cc2aa_n("Iuvwy");
    ci_va("Iew") -= 0.1250 * V.block("vAaA")("eYuV") * cc2ab_n("IuVwY");
    ci_va("Iew") -= 0.0625 * V.block("avaa")("xeuv") * cc2aa_n("Iuvxw");
    ci_va("Iew") -= 0.1250 * V.block("vAaA")("eXuV") * cc2ab_n("IuVwX");

    ci_va("Jew") -= 0.0625 * V.block("vaaa")("evxy") * cc2aa_r("Jwvxy");
    ci_va("Jew") -= 0.1250 * V.block("vAaA")("eVxY") * cc2ab_r("JwVxY");
    ci_va("Jew") -= 0.0625 * V.block("avaa")("uexy") * cc2aa_r("Juwxy");
    ci_va("Jew") -= 0.1250 * V.block("vAaA")("eUxY") * cc2ab_r("JwUxY");
    ci_va("Jew") -= 0.0625 * V.block("vaaa")("eyuv") * cc2aa_r("Juvwy");
    ci_va("Jew") -= 0.1250 * V.block("vAaA")("eYuV") * cc2ab_r("JuVwY");
    ci_va("Jew") -= 0.0625 * V.block("avaa")("xeuv") * cc2aa_r("Juvxw");
    ci_va("Jew") -= 0.1250 * V.block("vAaA")("eXuV") * cc2ab_r("JuVwX");

    // CI contribution to Z{AA}
    ci_aa("Iwz") -= 0.25 * H.block("aa")("vw") * cc1a_n("Izv");
    ci_aa("Iwz") -= 0.25 * H.block("aa")("wu") * cc1a_n("Iuz");
    ci_aa("Iwz") += 0.25 * H.block("aa")("vz") * cc1a_n("Iwv");
    ci_aa("Iwz") += 0.25 * H.block("aa")("zu") * cc1a_n("Iuw");
    ci_aa("Iwz") -= 0.25 * H.block("aa")("vw") * cc1a_r("Izv");
    ci_aa("Iwz") -= 0.25 * H.block("aa")("wu") * cc1a_r("Iuz");
    ci_aa("Iwz") += 0.25 * H.block("aa")("vz") * cc1a_r("Iwv");
    ci_aa("Iwz") += 0.25 * H.block("aa")("zu") * cc1a_r("Iuw");

    ci_aa("Iwz") -= 0.25 * V_N_Alpha.block("aa")("uw") * cc1a_n("Iuz");
    ci_aa("Iwz") -= 0.25 * V_N_Beta.block("aa")("uw")  * cc1a_n("Iuz");
    ci_aa("Iwz") -= 0.25 * V_N_Alpha.block("aa")("wv") * cc1a_n("Izv");
    ci_aa("Iwz") -= 0.25 * V_N_Beta.block("aa")("wv")  * cc1a_n("Izv");
    ci_aa("Iwz") -= 0.25 * V_N_Alpha.block("aa")("uw") * cc1a_r("Iuz");
    ci_aa("Iwz") -= 0.25 * V_N_Beta.block("aa")("uw")  * cc1a_r("Iuz");
    ci_aa("Iwz") -= 0.25 * V_N_Alpha.block("aa")("wv") * cc1a_r("Izv");
    ci_aa("Iwz") -= 0.25 * V_N_Beta.block("aa")("wv")  * cc1a_r("Izv");

    ci_aa("Iwz") += 0.25 * V_N_Alpha.block("aa")("uz") * cc1a_n("Iuw");
    ci_aa("Iwz") += 0.25 * V_N_Beta.block("aa")("uz")  * cc1a_n("Iuw");
    ci_aa("Iwz") += 0.25 * V_N_Alpha.block("aa")("zv") * cc1a_n("Iwv");
    ci_aa("Iwz") += 0.25 * V_N_Beta.block("aa")("zv")  * cc1a_n("Iwv");
    ci_aa("Iwz") += 0.25 * V_N_Alpha.block("aa")("uz") * cc1a_r("Iuw");
    ci_aa("Iwz") += 0.25 * V_N_Beta.block("aa")("uz")  * cc1a_r("Iuw");
    ci_aa("Iwz") += 0.25 * V_N_Alpha.block("aa")("zv") * cc1a_r("Iwv");
    ci_aa("Iwz") += 0.25 * V_N_Beta.block("aa")("zv")  * cc1a_r("Iwv");

    ci_aa("Iwz") -= 0.0625 * V.block("aaaa")("wvxy") * cc2aa_n("Izvxy");
    ci_aa("Iwz") -= 0.1250 * V.block("aAaA")("wVxY") * cc2ab_n("IzVxY");
    ci_aa("Iwz") -= 0.0625 * V.block("aaaa")("uwxy") * cc2aa_n("Iuzxy");
    ci_aa("Iwz") -= 0.1250 * V.block("aAaA")("wUxY") * cc2ab_n("IzUxY");
    ci_aa("Iwz") -= 0.0625 * V.block("aaaa")("uvwy") * cc2aa_n("Iuvzy");
    ci_aa("Iwz") -= 0.1250 * V.block("aAaA")("uVwY") * cc2ab_n("IuVzY");
    ci_aa("Iwz") -= 0.0625 * V.block("aaaa")("uvxw") * cc2aa_n("Iuvxz");
    ci_aa("Iwz") -= 0.1250 * V.block("aAaA")("uVwX") * cc2ab_n("IuVzX");
    ci_aa("Iwz") += 0.0625 * V.block("aaaa")("zvxy") * cc2aa_n("Iwvxy");
    ci_aa("Iwz") += 0.1250 * V.block("aAaA")("zVxY") * cc2ab_n("IwVxY");
    ci_aa("Iwz") += 0.0625 * V.block("aaaa")("uzxy") * cc2aa_n("Iuwxy");
    ci_aa("Iwz") += 0.1250 * V.block("aAaA")("zUxY") * cc2ab_n("IwUxY");
    ci_aa("Iwz") += 0.0625 * V.block("aaaa")("uvzy") * cc2aa_n("Iuvwy");
    ci_aa("Iwz") += 0.1250 * V.block("aAaA")("uVzY") * cc2ab_n("IuVwY");
    ci_aa("Iwz") += 0.0625 * V.block("aaaa")("uvxz") * cc2aa_n("Iuvxw");
    ci_aa("Iwz") += 0.1250 * V.block("aAaA")("uVzX") * cc2ab_n("IuVwX");

    ci_aa("Jwz") -= 0.0625 * V.block("aaaa")("wvxy") * cc2aa_r("Jzvxy");
    ci_aa("Jwz") -= 0.1250 * V.block("aAaA")("wVxY") * cc2ab_r("JzVxY");
    ci_aa("Jwz") -= 0.0625 * V.block("aaaa")("uwxy") * cc2aa_r("Juzxy");
    ci_aa("Jwz") -= 0.1250 * V.block("aAaA")("wUxY") * cc2ab_r("JzUxY");
    ci_aa("Jwz") -= 0.0625 * V.block("aaaa")("uvwy") * cc2aa_r("Juvzy");
    ci_aa("Jwz") -= 0.1250 * V.block("aAaA")("uVwY") * cc2ab_r("JuVzY");
    ci_aa("Jwz") -= 0.0625 * V.block("aaaa")("uvxw") * cc2aa_r("Juvxz");
    ci_aa("Jwz") -= 0.1250 * V.block("aAaA")("uVwX") * cc2ab_r("JuVzX");
    ci_aa("Jwz") += 0.0625 * V.block("aaaa")("zvxy") * cc2aa_r("Jwvxy");
    ci_aa("Jwz") += 0.1250 * V.block("aAaA")("zVxY") * cc2ab_r("JwVxY");
    ci_aa("Jwz") += 0.0625 * V.block("aaaa")("uzxy") * cc2aa_r("Juwxy");
    ci_aa("Jwz") += 0.1250 * V.block("aAaA")("zUxY") * cc2ab_r("JwUxY");
    ci_aa("Jwz") += 0.0625 * V.block("aaaa")("uvzy") * cc2aa_r("Juvwy");
    ci_aa("Jwz") += 0.1250 * V.block("aAaA")("uVzY") * cc2ab_r("JuVwY");
    ci_aa("Jwz") += 0.0625 * V.block("aaaa")("uvxz") * cc2aa_r("Juvxw");
    ci_aa("Jwz") += 0.1250 * V.block("aAaA")("uVzX") * cc2ab_r("JuVwX");

    for (const std::string& row : {"vc","ca","va","aa"}) {
        int idx1 = block_dim[row];
        int pre1 = preidx[row];
        auto temp_ci = ci_vc;
        if (row == "ca") temp_ci = ci_ca;
        else if (row == "va") temp_ci = ci_va;
        else if (row == "aa") temp_ci = ci_aa;

        for (const std::string& col : {"ci"}) {
            int pre2 = preidx[col];
            if (row != "aa") {
                (temp_ci).iterate([&](const std::vector<size_t>& i, double& value) {
                    int index = (pre1 + i[1] * idx1 + i[2]) 
                                 + dim * (pre2 + i[0]);
                    A.at(index) += value;
                });
            }
            else if (row == "aa") {       
                (temp_ci).iterate([&](const std::vector<size_t>& i, double& value) {
                    if (i[1] > i[2] ) {
                        int index = (pre1 + i[1] * (i[1] - 1) / 2 + i[2])
                                     + dim * (pre2 + i[0]); 
                        A.at(index) += value;    
                    }
                });
            }
        }
    } 

    auto b_ck  = ambit::Tensor::build(ambit::CoreTensor, "ci equations b part", {ndets});
    b_ck("K") -= H.block("aa")("vu") * cc1a_n("Kuv");
    b_ck("K") -= H.block("AA")("VU") * cc1b_n("KUV");
    b_ck("K") -= H.block("aa")("vu") * cc1a_r("Kuv");
    b_ck("K") -= H.block("AA")("VU") * cc1b_r("KUV");

    b_ck("K") -= V_N_Alpha.block("aa")("vu") * cc1a_n("Kuv");
    b_ck("K") -= V_N_Beta.block("aa")("vu")  * cc1a_n("Kuv");
    b_ck("K") -= V_N_Alpha.block("aa")("vu") * cc1a_r("Kuv");
    b_ck("K") -= V_N_Beta.block("aa")("vu")  * cc1a_r("Kuv");

    b_ck("K") -= V_all_Beta.block("AA")("VU") * cc1b_n("KUV");
    b_ck("K") -= V_R_Beta.block("AA")("VU")   * cc1b_n("KUV");
    b_ck("K") -= V_all_Beta.block("AA")("VU") * cc1b_r("KUV");
    b_ck("K") -= V_R_Beta.block("AA")("VU")   * cc1b_r("KUV");

    b_ck("K") -= 0.5 * V.block("aaaa")("xyuv") * cc2aa_n("Kuvxy");
    b_ck("K") -= 0.5 * V.block("AAAA")("XYUV") * cc2bb_n("KUVXY");
    b_ck("K") -= V.block("aAaA")("xYuV") * cc2ab_n("KuVxY");
    b_ck("J") -= V.block("aAaA")("xYuV") * cc2ab_r("JuVxY");

    // Alpha
    Alpha  = 0.0;
    Alpha += H["vu"] * Gamma1_["uv"];
    Alpha += H["VU"] * Gamma1_["UV"];
    Alpha += V_N_Alpha["v,u"] * Gamma1_["uv"];
    Alpha += V_N_Beta["v,u"] * Gamma1_["uv"];
    Alpha += V["V,M,U,M1"] * Gamma1_["UV"] * I["M,M1"];
    Alpha += V["m,V,m1,U"] * Gamma1_["UV"] * I["m,m1"];
    Alpha += 0.25 * V["xyuv"] * Gamma2_["uvxy"];
    Alpha += 0.25 * V["XYUV"] * Gamma2_["UVXY"];
    Alpha += V["xYuV"] * Gamma2_["uVxY"];

    BlockedTensor temp = BTF_->build(CoreTensor, "temporal tensor", {"aa","AA"});

    if (PT2_TERM) {
        temp["uv"] += 0.25 * T2_["vmef"] * V_["efum"];
        temp["uv"] += 0.25 * T2_["vmez"] * V_["ewum"] * Eta1_["zw"];
        temp["uv"] += 0.25 * T2_["vmze"] * V_["weum"] * Eta1_["zw"];
        temp["uv"] += 0.25 * T2_["vmzx"] * V_["wyum"] * Eta1_["zw"] * Eta1_["xy"];
        temp["uv"] += 0.25 * T2_["vwef"] * V_["efuz"] * Gamma1_["zw"];
        temp["uv"] += 0.25 * T2_["vwex"] * V_["eyuz"] * Gamma1_["zw"] * Eta1_["xy"];
        temp["uv"] += 0.25 * T2_["vwxe"] * V_["yeuz"] * Gamma1_["zw"] * Eta1_["xy"];
        temp["uv"] += 0.50 * T2_["vMeF"] * V_["eFuM"];
        temp["uv"] += 0.50 * T2_["vMeZ"] * V_["eWuM"] * Eta1_["ZW"];
        temp["uv"] += 0.50 * T2_["vMzE"] * V_["wEuM"] * Eta1_["zw"];
        temp["uv"] += 0.50 * T2_["vMzX"] * V_["wYuM"] * Eta1_["zw"] * Eta1_["XY"];
        temp["uv"] += 0.50 * T2_["vWeF"] * V_["eFuZ"] * Gamma1_["ZW"];
        temp["uv"] += 0.50 * T2_["vWeX"] * V_["eYuZ"] * Gamma1_["ZW"] * Eta1_["XY"];
        temp["uv"] += 0.50 * T2_["vWxE"] * V_["yEuZ"] * Gamma1_["ZW"] * Eta1_["xy"];
        temp["UV"] += 0.25 * T2_["VMEF"] * V_["EFUM"];
        temp["UV"] += 0.25 * T2_["VMEZ"] * V_["EWUM"] * Eta1_["ZW"];
        temp["UV"] += 0.25 * T2_["VMZE"] * V_["WEUM"] * Eta1_["ZW"];
        temp["UV"] += 0.25 * T2_["VMZX"] * V_["WYUM"] * Eta1_["ZW"] * Eta1_["XY"];
        temp["UV"] += 0.25 * T2_["VWEF"] * V_["EFUZ"] * Gamma1_["ZW"];
        temp["UV"] += 0.25 * T2_["VWEX"] * V_["EYUZ"] * Gamma1_["ZW"] * Eta1_["XY"];
        temp["UV"] += 0.25 * T2_["VWXE"] * V_["YEUZ"] * Gamma1_["ZW"] * Eta1_["XY"];
        temp["UV"] += 0.50 * T2_["mVeF"] * V_["eFmU"];
        temp["UV"] += 0.50 * T2_["mVeZ"] * V_["eWmU"] * Eta1_["ZW"];
        temp["UV"] += 0.50 * T2_["mVzE"] * V_["wEmU"] * Eta1_["zw"];
        temp["UV"] += 0.50 * T2_["mVzX"] * V_["wYmU"] * Eta1_["zw"] * Eta1_["XY"];
        temp["UV"] += 0.50 * T2_["wVeF"] * V_["eFzU"] * Gamma1_["zw"];
        temp["UV"] += 0.50 * T2_["wVeX"] * V_["eYzU"] * Gamma1_["zw"] * Eta1_["XY"];
        temp["UV"] += 0.50 * T2_["wVxE"] * V_["yEzU"] * Gamma1_["zw"] * Eta1_["xy"];

        temp["uv"] += 0.25 * T2_["umef"] * V_["efvm"]; 
        temp["uv"] += 0.25 * T2_["umez"] * V_["ewvm"] * Eta1_["zw"]; 
        temp["uv"] += 0.25 * T2_["umze"] * V_["wevm"] * Eta1_["zw"]; 
        temp["uv"] += 0.25 * T2_["umzx"] * V_["wyvm"] * Eta1_["zw"] * Eta1_["xy"]; 
        temp["uv"] += 0.25 * T2_["uwef"] * V_["efvz"] * Gamma1_["zw"]; 
        temp["uv"] += 0.25 * T2_["uwex"] * V_["eyvz"] * Gamma1_["zw"] * Eta1_["xy"]; 
        temp["uv"] += 0.25 * T2_["uwxe"] * V_["yevz"] * Gamma1_["zw"] * Eta1_["xy"]; 
        temp["uv"] += 0.50 * T2_["uMeF"] * V_["eFvM"];
        temp["uv"] += 0.50 * T2_["uMeZ"] * V_["eWvM"] * Eta1_["ZW"];
        temp["uv"] += 0.50 * T2_["uMzE"] * V_["wEvM"] * Eta1_["zw"];
        temp["uv"] += 0.50 * T2_["uMzX"] * V_["wYvM"] * Eta1_["zw"] * Eta1_["XY"];
        temp["uv"] += 0.50 * T2_["uWeF"] * V_["eFvZ"] * Gamma1_["ZW"];
        temp["uv"] += 0.50 * T2_["uWeX"] * V_["eYvZ"] * Gamma1_["ZW"] * Eta1_["XY"];
        temp["uv"] += 0.50 * T2_["uWxE"] * V_["yEvZ"] * Gamma1_["ZW"] * Eta1_["xy"];
        temp["UV"] += 0.25 * T2_["UMEF"] * V_["EFVM"];
        temp["UV"] += 0.25 * T2_["UMEZ"] * V_["EWVM"] * Eta1_["ZW"];
        temp["UV"] += 0.25 * T2_["UMZE"] * V_["WEVM"] * Eta1_["ZW"];
        temp["UV"] += 0.25 * T2_["UMZX"] * V_["WYVM"] * Eta1_["ZW"] * Eta1_["XY"];
        temp["UV"] += 0.25 * T2_["UWEF"] * V_["EFVZ"] * Gamma1_["ZW"];
        temp["UV"] += 0.25 * T2_["UWEX"] * V_["EYVZ"] * Gamma1_["ZW"] * Eta1_["XY"];
        temp["UV"] += 0.25 * T2_["UWXE"] * V_["YEVZ"] * Gamma1_["ZW"] * Eta1_["XY"];
        temp["UV"] += 0.50 * T2_["mUeF"] * V_["eFmV"];
        temp["UV"] += 0.50 * T2_["mUeZ"] * V_["eWmV"] * Eta1_["ZW"];
        temp["UV"] += 0.50 * T2_["mUzE"] * V_["wEmV"] * Eta1_["zw"];
        temp["UV"] += 0.50 * T2_["mUzX"] * V_["wYmV"] * Eta1_["zw"] * Eta1_["XY"];
        temp["UV"] += 0.50 * T2_["wUeF"] * V_["eFzV"] * Gamma1_["zw"];
        temp["UV"] += 0.50 * T2_["wUeX"] * V_["eYzV"] * Gamma1_["zw"] * Eta1_["XY"];
        temp["UV"] += 0.50 * T2_["wUxE"] * V_["yEzV"] * Gamma1_["zw"] * Eta1_["xy"];


        temp["uv"] += 0.25 * T2_["mvef"] * V_["efmu"];
        temp["uv"] += 0.25 * T2_["mvez"] * V_["ewmu"] * Eta1_["zw"];
        temp["uv"] += 0.25 * T2_["mvze"] * V_["wemu"] * Eta1_["zw"];
        temp["uv"] += 0.25 * T2_["mvzx"] * V_["wymu"] * Eta1_["zw"] * Eta1_["xy"];
        temp["uv"] += 0.25 * T2_["wvef"] * V_["efzu"] * Gamma1_["zw"];
        temp["uv"] += 0.25 * T2_["wvex"] * V_["eyzu"] * Gamma1_["zw"] * Eta1_["xy"];
        temp["uv"] += 0.25 * T2_["wvxe"] * V_["yezu"] * Gamma1_["zw"] * Eta1_["xy"];
        temp["uv"] += 0.50 * T2_["vMeF"] * V_["eFuM"];
        temp["uv"] += 0.50 * T2_["vMeZ"] * V_["eWuM"] * Eta1_["ZW"];
        temp["uv"] += 0.50 * T2_["vMzE"] * V_["wEuM"] * Eta1_["zw"];
        temp["uv"] += 0.50 * T2_["vMzX"] * V_["wYuM"] * Eta1_["zw"] * Eta1_["XY"];
        temp["uv"] += 0.50 * T2_["vWeF"] * V_["eFuZ"] * Gamma1_["ZW"];
        temp["uv"] += 0.50 * T2_["vWeX"] * V_["eYuZ"] * Gamma1_["ZW"] * Eta1_["XY"];
        temp["uv"] += 0.50 * T2_["vWxE"] * V_["yEuZ"] * Gamma1_["ZW"] * Eta1_["xy"];
        temp["UV"] += 0.25 * T2_["MVEF"] * V_["EFMU"];
        temp["UV"] += 0.25 * T2_["MVEZ"] * V_["EWMU"] * Eta1_["ZW"];
        temp["UV"] += 0.25 * T2_["MVZE"] * V_["WEMU"] * Eta1_["ZW"];
        temp["UV"] += 0.25 * T2_["MVZX"] * V_["WYMU"] * Eta1_["ZW"] * Eta1_["XY"];
        temp["UV"] += 0.25 * T2_["WVEF"] * V_["EFZU"] * Gamma1_["ZW"];
        temp["UV"] += 0.25 * T2_["WVEX"] * V_["EYZU"] * Gamma1_["ZW"] * Eta1_["XY"];
        temp["UV"] += 0.25 * T2_["WVXE"] * V_["YEZU"] * Gamma1_["ZW"] * Eta1_["XY"];
        temp["UV"] += 0.50 * T2_["mVeF"] * V_["eFmU"];
        temp["UV"] += 0.50 * T2_["mVeZ"] * V_["eWmU"] * Eta1_["ZW"];
        temp["UV"] += 0.50 * T2_["mVzE"] * V_["wEmU"] * Eta1_["zw"];
        temp["UV"] += 0.50 * T2_["mVzX"] * V_["wYmU"] * Eta1_["zw"] * Eta1_["XY"];
        temp["UV"] += 0.50 * T2_["wVeF"] * V_["eFzU"] * Gamma1_["zw"];
        temp["UV"] += 0.50 * T2_["wVeX"] * V_["eYzU"] * Gamma1_["zw"] * Eta1_["XY"];
        temp["UV"] += 0.50 * T2_["wVxE"] * V_["yEzU"] * Gamma1_["zw"] * Eta1_["xy"];


        temp["uv"] += 0.25 * T2_["muef"] * V_["efmv"];
        temp["uv"] += 0.25 * T2_["muez"] * V_["ewmv"] * Eta1_["zw"];
        temp["uv"] += 0.25 * T2_["muze"] * V_["wemv"] * Eta1_["zw"];
        temp["uv"] += 0.25 * T2_["muzx"] * V_["wymv"] * Eta1_["zw"] * Eta1_["xy"];
        temp["uv"] += 0.25 * T2_["wuef"] * V_["efzv"] * Gamma1_["zw"];
        temp["uv"] += 0.25 * T2_["wuex"] * V_["eyzv"] * Gamma1_["zw"] * Eta1_["xy"];
        temp["uv"] += 0.25 * T2_["wuxe"] * V_["yezv"] * Gamma1_["zw"] * Eta1_["xy"];
        temp["uv"] += 0.50 * T2_["uMeF"] * V_["eFvM"];
        temp["uv"] += 0.50 * T2_["uMeZ"] * V_["eWvM"] * Eta1_["ZW"];
        temp["uv"] += 0.50 * T2_["uMzE"] * V_["wEvM"] * Eta1_["zw"];
        temp["uv"] += 0.50 * T2_["uMzX"] * V_["wYvM"] * Eta1_["zw"] * Eta1_["XY"];
        temp["uv"] += 0.50 * T2_["uWeF"] * V_["eFvZ"] * Gamma1_["ZW"];
        temp["uv"] += 0.50 * T2_["uWeX"] * V_["eYvZ"] * Gamma1_["ZW"] * Eta1_["XY"];
        temp["uv"] += 0.50 * T2_["uWxE"] * V_["yEvZ"] * Gamma1_["ZW"] * Eta1_["xy"];
        temp["UV"] += 0.25 * T2_["MUEF"] * V_["EFMV"];
        temp["UV"] += 0.25 * T2_["MUEZ"] * V_["EWMV"] * Eta1_["ZW"];
        temp["UV"] += 0.25 * T2_["MUZE"] * V_["WEMV"] * Eta1_["ZW"];
        temp["UV"] += 0.25 * T2_["MUZX"] * V_["WYMV"] * Eta1_["ZW"] * Eta1_["XY"];
        temp["UV"] += 0.25 * T2_["WUEF"] * V_["EFZV"] * Gamma1_["ZW"];
        temp["UV"] += 0.25 * T2_["WUEX"] * V_["EYZV"] * Gamma1_["ZW"] * Eta1_["XY"];
        temp["UV"] += 0.25 * T2_["WUXE"] * V_["YEZV"] * Gamma1_["ZW"] * Eta1_["XY"];
        temp["UV"] += 0.50 * T2_["mUeF"] * V_["eFmV"];
        temp["UV"] += 0.50 * T2_["mUeZ"] * V_["eWmV"] * Eta1_["ZW"];
        temp["UV"] += 0.50 * T2_["mUzE"] * V_["wEmV"] * Eta1_["zw"];
        temp["UV"] += 0.50 * T2_["mUzX"] * V_["wYmV"] * Eta1_["zw"] * Eta1_["XY"];
        temp["UV"] += 0.50 * T2_["wUeF"] * V_["eFzV"] * Gamma1_["zw"];
        temp["UV"] += 0.50 * T2_["wUeX"] * V_["eYzV"] * Gamma1_["zw"] * Eta1_["XY"];
        temp["UV"] += 0.50 * T2_["wUxE"] * V_["yEzV"] * Gamma1_["zw"] * Eta1_["xy"];


        temp["uv"] -= 0.25 * T2_["mnue"] * V_["vemn"];
        temp["uv"] -= 0.25 * T2_["mnuz"] * V_["vwmn"] * Eta1_["zw"];
        temp["uv"] -= 0.25 * T2_["mwue"] * V_["vemz"] * Gamma1_["zw"];
        temp["uv"] -= 0.25 * T2_["mwux"] * V_["vymz"] * Gamma1_["zw"] * Eta1_["xy"];
        temp["uv"] -= 0.25 * T2_["wmue"] * V_["vezm"] * Gamma1_["zw"];
        temp["uv"] -= 0.25 * T2_["wmux"] * V_["vyzm"] * Gamma1_["zw"] * Eta1_["xy"];
        temp["uv"] -= 0.25 * T2_["wyue"] * V_["vezx"] * Gamma1_["zw"] * Gamma1_["xy"];
        temp["uv"] -= 0.50 * T2_["mNuE"] * V_["vEmN"];
        temp["uv"] -= 0.50 * T2_["mNuZ"] * V_["vWmN"] * Eta1_["ZW"];
        temp["uv"] -= 0.50 * T2_["mWuE"] * V_["vEmZ"] * Gamma1_["ZW"];
        temp["uv"] -= 0.50 * T2_["mWuX"] * V_["vYmZ"] * Gamma1_["ZW"] * Eta1_["XY"];
        temp["uv"] -= 0.50 * T2_["wMuE"] * V_["vEzM"] * Gamma1_["zw"];
        temp["uv"] -= 0.50 * T2_["wMuX"] * V_["vYzM"] * Gamma1_["zw"] * Eta1_["XY"];
        temp["uv"] -= 0.50 * T2_["wYuE"] * V_["vEzX"] * Gamma1_["zw"] * Gamma1_["XY"];
        temp["UV"] -= 0.25 * T2_["MNUE"] * V_["VEMN"];
        temp["UV"] -= 0.25 * T2_["MNUZ"] * V_["VWMN"] * Eta1_["ZW"];
        temp["UV"] -= 0.25 * T2_["MWUE"] * V_["VEMZ"] * Gamma1_["ZW"];
        temp["UV"] -= 0.25 * T2_["MWUX"] * V_["VYMZ"] * Gamma1_["ZW"] * Eta1_["XY"];
        temp["UV"] -= 0.25 * T2_["WMUE"] * V_["VEZM"] * Gamma1_["ZW"];
        temp["UV"] -= 0.25 * T2_["WMUX"] * V_["VYZM"] * Gamma1_["ZW"] * Eta1_["XY"];
        temp["UV"] -= 0.25 * T2_["WYUE"] * V_["VEZX"] * Gamma1_["ZW"] * Gamma1_["XY"];
        temp["UV"] -= 0.50 * T2_["mNeU"] * V_["eVmN"];
        temp["UV"] -= 0.50 * T2_["mNzU"] * V_["wVmN"] * Eta1_["zw"];
        temp["UV"] -= 0.50 * T2_["mWeU"] * V_["eVmZ"] * Gamma1_["ZW"];
        temp["UV"] -= 0.50 * T2_["mWxU"] * V_["yVmZ"] * Gamma1_["ZW"] * Eta1_["xy"];
        temp["UV"] -= 0.50 * T2_["wMeU"] * V_["eVzM"] * Gamma1_["zw"];
        temp["UV"] -= 0.50 * T2_["wMxU"] * V_["yVzM"] * Gamma1_["zw"] * Eta1_["xy"];
        temp["UV"] -= 0.50 * T2_["wYeU"] * V_["eVzX"] * Gamma1_["zw"] * Gamma1_["XY"];


        temp["uv"] -= 0.25 * T2_["mnve"] * V_["uemn"];
        temp["uv"] -= 0.25 * T2_["mnvz"] * V_["uwmn"] * Eta1_["zw"];
        temp["uv"] -= 0.25 * T2_["mwve"] * V_["uemz"] * Gamma1_["zw"];
        temp["uv"] -= 0.25 * T2_["mwvx"] * V_["uymz"] * Gamma1_["zw"] * Eta1_["xy"];
        temp["uv"] -= 0.25 * T2_["wmve"] * V_["uezm"] * Gamma1_["zw"];
        temp["uv"] -= 0.25 * T2_["wmvx"] * V_["uyzm"] * Gamma1_["zw"] * Eta1_["xy"];
        temp["uv"] -= 0.25 * T2_["wyve"] * V_["uezx"] * Gamma1_["zw"] * Gamma1_["xy"];
        temp["uv"] -= 0.50 * T2_["mNvE"] * V_["uEmN"];
        temp["uv"] -= 0.50 * T2_["mNvZ"] * V_["uWmN"] * Eta1_["ZW"];
        temp["uv"] -= 0.50 * T2_["mWvE"] * V_["uEmZ"] * Gamma1_["ZW"];
        temp["uv"] -= 0.50 * T2_["mWvX"] * V_["uYmZ"] * Gamma1_["ZW"] * Eta1_["XY"];
        temp["uv"] -= 0.50 * T2_["wMvE"] * V_["uEzM"] * Gamma1_["zw"];
        temp["uv"] -= 0.50 * T2_["wMvX"] * V_["uYzM"] * Gamma1_["zw"] * Eta1_["XY"];
        temp["uv"] -= 0.50 * T2_["wYvE"] * V_["uEzX"] * Gamma1_["zw"] * Gamma1_["XY"];
        temp["UV"] -= 0.25 * T2_["MNVE"] * V_["UEMN"];
        temp["UV"] -= 0.25 * T2_["MNVZ"] * V_["UWMN"] * Eta1_["ZW"];
        temp["UV"] -= 0.25 * T2_["MWVE"] * V_["UEMZ"] * Gamma1_["ZW"];
        temp["UV"] -= 0.25 * T2_["MWVX"] * V_["UYMZ"] * Gamma1_["ZW"] * Eta1_["XY"];
        temp["UV"] -= 0.25 * T2_["WMVE"] * V_["UEZM"] * Gamma1_["ZW"];
        temp["UV"] -= 0.25 * T2_["WMVX"] * V_["UYZM"] * Gamma1_["ZW"] * Eta1_["XY"];
        temp["UV"] -= 0.25 * T2_["WYVE"] * V_["UEZX"] * Gamma1_["ZW"] * Gamma1_["XY"];
        temp["UV"] -= 0.50 * T2_["mNeV"] * V_["eUmN"];
        temp["UV"] -= 0.50 * T2_["mNzV"] * V_["wUmN"] * Eta1_["zw"];
        temp["UV"] -= 0.50 * T2_["mWeV"] * V_["eUmZ"] * Gamma1_["ZW"];
        temp["UV"] -= 0.50 * T2_["mWxV"] * V_["yUmZ"] * Gamma1_["ZW"] * Eta1_["xy"];
        temp["UV"] -= 0.50 * T2_["wMeV"] * V_["eUzM"] * Gamma1_["zw"];
        temp["UV"] -= 0.50 * T2_["wMxV"] * V_["yUzM"] * Gamma1_["zw"] * Eta1_["xy"];
        temp["UV"] -= 0.50 * T2_["wYeV"] * V_["eUzX"] * Gamma1_["zw"] * Gamma1_["XY"];


        temp["uv"] -= 0.25 * T2_["mneu"] * V_["evmn"];
        temp["uv"] -= 0.25 * T2_["mnzu"] * V_["wvmn"] * Eta1_["zw"];
        temp["uv"] -= 0.25 * T2_["mweu"] * V_["evmz"] * Gamma1_["zw"];
        temp["uv"] -= 0.25 * T2_["mwxu"] * V_["yvmz"] * Gamma1_["zw"] * Eta1_["xy"];
        temp["uv"] -= 0.25 * T2_["wmeu"] * V_["evzm"] * Gamma1_["zw"];
        temp["uv"] -= 0.25 * T2_["wmxu"] * V_["yvzm"] * Gamma1_["zw"] * Eta1_["xy"];
        temp["uv"] -= 0.25 * T2_["wyeu"] * V_["evzx"] * Gamma1_["zw"] * Gamma1_["xy"];
        temp["uv"] -= 0.50 * T2_["mNuE"] * V_["vEmN"];
        temp["uv"] -= 0.50 * T2_["mNuZ"] * V_["vWmN"] * Eta1_["ZW"];
        temp["uv"] -= 0.50 * T2_["mWuE"] * V_["vEmZ"] * Gamma1_["ZW"];
        temp["uv"] -= 0.50 * T2_["mWuX"] * V_["vYmZ"] * Gamma1_["ZW"] * Eta1_["XY"];
        temp["uv"] -= 0.50 * T2_["wMuE"] * V_["vEzM"] * Gamma1_["zw"];
        temp["uv"] -= 0.50 * T2_["wMuX"] * V_["vYzM"] * Gamma1_["zw"] * Eta1_["XY"];
        temp["uv"] -= 0.50 * T2_["wYuE"] * V_["vEzX"] * Gamma1_["zw"] * Gamma1_["XY"];
        temp["UV"] -= 0.25 * T2_["MNEU"] * V_["EVMN"];
        temp["UV"] -= 0.25 * T2_["MNZU"] * V_["WVMN"] * Eta1_["ZW"];
        temp["UV"] -= 0.25 * T2_["MWEU"] * V_["EVMZ"] * Gamma1_["ZW"];
        temp["UV"] -= 0.25 * T2_["MWXU"] * V_["YVMZ"] * Gamma1_["ZW"] * Eta1_["XY"];
        temp["UV"] -= 0.25 * T2_["WMEU"] * V_["EVZM"] * Gamma1_["ZW"];
        temp["UV"] -= 0.25 * T2_["WMXU"] * V_["YVZM"] * Gamma1_["ZW"] * Eta1_["XY"];
        temp["UV"] -= 0.25 * T2_["WYEU"] * V_["EVZX"] * Gamma1_["ZW"] * Gamma1_["XY"];
        temp["UV"] -= 0.50 * T2_["mNeU"] * V_["eVmN"];
        temp["UV"] -= 0.50 * T2_["mNzU"] * V_["wVmN"] * Eta1_["zw"];
        temp["UV"] -= 0.50 * T2_["mWeU"] * V_["eVmZ"] * Gamma1_["ZW"];
        temp["UV"] -= 0.50 * T2_["mWxU"] * V_["yVmZ"] * Gamma1_["ZW"] * Eta1_["xy"];
        temp["UV"] -= 0.50 * T2_["wMeU"] * V_["eVzM"] * Gamma1_["zw"];
        temp["UV"] -= 0.50 * T2_["wMxU"] * V_["yVzM"] * Gamma1_["zw"] * Eta1_["xy"];
        temp["UV"] -= 0.50 * T2_["wYeU"] * V_["eVzX"] * Gamma1_["zw"] * Gamma1_["XY"];


        temp["uv"] -= 0.25 * T2_["mnev"] * V_["eumn"];
        temp["uv"] -= 0.25 * T2_["mnzv"] * V_["wumn"] * Eta1_["zw"];
        temp["uv"] -= 0.25 * T2_["mwev"] * V_["eumz"] * Gamma1_["zw"];
        temp["uv"] -= 0.25 * T2_["mwxv"] * V_["yumz"] * Gamma1_["zw"] * Eta1_["xy"];
        temp["uv"] -= 0.25 * T2_["wmev"] * V_["euzm"] * Gamma1_["zw"];
        temp["uv"] -= 0.25 * T2_["wmxv"] * V_["yuzm"] * Gamma1_["zw"] * Eta1_["xy"];
        temp["uv"] -= 0.25 * T2_["wyev"] * V_["euzx"] * Gamma1_["zw"] * Gamma1_["xy"];
        temp["uv"] -= 0.50 * T2_["mNvE"] * V_["uEmN"];
        temp["uv"] -= 0.50 * T2_["mNvZ"] * V_["uWmN"] * Eta1_["ZW"];
        temp["uv"] -= 0.50 * T2_["mWvE"] * V_["uEmZ"] * Gamma1_["ZW"];
        temp["uv"] -= 0.50 * T2_["mWvX"] * V_["uYmZ"] * Gamma1_["ZW"] * Eta1_["XY"];
        temp["uv"] -= 0.50 * T2_["wMvE"] * V_["uEzM"] * Gamma1_["zw"];
        temp["uv"] -= 0.50 * T2_["wMvX"] * V_["uYzM"] * Gamma1_["zw"] * Eta1_["XY"];
        temp["uv"] -= 0.50 * T2_["wYvE"] * V_["uEzX"] * Gamma1_["zw"] * Gamma1_["XY"];
        temp["UV"] -= 0.25 * T2_["MNEV"] * V_["EUMN"];
        temp["UV"] -= 0.25 * T2_["MNZV"] * V_["WUMN"] * Eta1_["ZW"];
        temp["UV"] -= 0.25 * T2_["MWEV"] * V_["EUMZ"] * Gamma1_["ZW"];
        temp["UV"] -= 0.25 * T2_["MWXV"] * V_["YUMZ"] * Gamma1_["ZW"] * Eta1_["XY"];
        temp["UV"] -= 0.25 * T2_["WMEV"] * V_["EUZM"] * Gamma1_["ZW"];
        temp["UV"] -= 0.25 * T2_["WMXV"] * V_["YUZM"] * Gamma1_["ZW"] * Eta1_["XY"];
        temp["UV"] -= 0.25 * T2_["WYEV"] * V_["EUZX"] * Gamma1_["ZW"] * Gamma1_["XY"];
        temp["UV"] -= 0.50 * T2_["mNeV"] * V_["eUmN"];
        temp["UV"] -= 0.50 * T2_["mNzV"] * V_["wUmN"] * Eta1_["zw"];
        temp["UV"] -= 0.50 * T2_["mWeV"] * V_["eUmZ"] * Gamma1_["ZW"];
        temp["UV"] -= 0.50 * T2_["mWxV"] * V_["yUmZ"] * Gamma1_["ZW"] * Eta1_["xy"];
        temp["UV"] -= 0.50 * T2_["wMeV"] * V_["eUzM"] * Gamma1_["zw"];
        temp["UV"] -= 0.50 * T2_["wMxV"] * V_["yUzM"] * Gamma1_["zw"] * Eta1_["xy"];
        temp["UV"] -= 0.50 * T2_["wYeV"] * V_["eUzX"] * Gamma1_["zw"] * Gamma1_["XY"];
    }

    if (X1_TERM) {
        temp["zw"] -= 0.125 * T2_["uvwe"] * V_["zexy"] * Lambda2_["xyuv"];
        temp["zw"] -= 0.500 * T2_["uVwE"] * V_["zExY"] * Lambda2_["xYuV"];
        temp["ZW"] -= 0.125 * T2_["UVWE"] * V_["ZEXY"] * Lambda2_["XYUV"];
        temp["ZW"] -= 0.500 * T2_["uVeW"] * V_["eZxY"] * Lambda2_["xYuV"];

        temp["zw"] -= 0.125 * T2_["uvze"] * V_["wexy"] * Lambda2_["xyuv"];
        temp["zw"] -= 0.500 * T2_["uVzE"] * V_["wExY"] * Lambda2_["xYuV"];
        temp["ZW"] -= 0.125 * T2_["UVZE"] * V_["WEXY"] * Lambda2_["XYUV"];
        temp["ZW"] -= 0.500 * T2_["uVeZ"] * V_["eWxY"] * Lambda2_["xYuV"];

        temp["zw"] -= 0.125 * T2_["uvew"] * V_["ezxy"] * Lambda2_["xyuv"];
        temp["zw"] -= 0.500 * T2_["uVwE"] * V_["zExY"] * Lambda2_["xYuV"];
        temp["ZW"] -= 0.125 * T2_["UVEW"] * V_["EZXY"] * Lambda2_["XYUV"];
        temp["ZW"] -= 0.500 * T2_["uVeW"] * V_["eZxY"] * Lambda2_["xYuV"];

        temp["zw"] -= 0.125 * T2_["uvez"] * V_["ewxy"] * Lambda2_["xyuv"];
        temp["zw"] -= 0.500 * T2_["uVzE"] * V_["wExY"] * Lambda2_["xYuV"];
        temp["ZW"] -= 0.125 * T2_["UVEZ"] * V_["EWXY"] * Lambda2_["XYUV"];
        temp["ZW"] -= 0.500 * T2_["uVeZ"] * V_["eWxY"] * Lambda2_["xYuV"];
    }

    if (X2_TERM) {
        temp["zw"] += 0.125 * T2_["wmxy"] * V_["uvzm"] * Lambda2_["xyuv"];
        temp["zw"] += 0.500 * T2_["wMxY"] * V_["uVzM"] * Lambda2_["xYuV"];
        temp["ZW"] += 0.125 * T2_["WMXY"] * V_["UVZM"] * Lambda2_["XYUV"];
        temp["ZW"] += 0.500 * T2_["mWxY"] * V_["uVmZ"] * Lambda2_["xYuV"];

        temp["zw"] += 0.125 * T2_["zmxy"] * V_["uvwm"] * Lambda2_["xyuv"];
        temp["zw"] += 0.500 * T2_["zMxY"] * V_["uVwM"] * Lambda2_["xYuV"];
        temp["ZW"] += 0.125 * T2_["ZMXY"] * V_["UVWM"] * Lambda2_["XYUV"];
        temp["ZW"] += 0.500 * T2_["mZxY"] * V_["uVmW"] * Lambda2_["xYuV"];

        temp["zw"] += 0.125 * T2_["mwxy"] * V_["uvmz"] * Lambda2_["xyuv"];
        temp["zw"] += 0.500 * T2_["wMxY"] * V_["uVzM"] * Lambda2_["xYuV"];
        temp["ZW"] += 0.125 * T2_["MWXY"] * V_["UVMZ"] * Lambda2_["XYUV"];
        temp["ZW"] += 0.500 * T2_["mWxY"] * V_["uVmZ"] * Lambda2_["xYuV"];

        temp["zw"] += 0.125 * T2_["mzxy"] * V_["uvmw"] * Lambda2_["xyuv"];
        temp["zw"] += 0.500 * T2_["zMxY"] * V_["uVwM"] * Lambda2_["xYuV"];
        temp["ZW"] += 0.125 * T2_["MZXY"] * V_["UVMW"] * Lambda2_["XYUV"];
        temp["ZW"] += 0.500 * T2_["mZxY"] * V_["uVmW"] * Lambda2_["xYuV"]; 
    }

    if (X3_TERM) {
        temp["zw"] -= V_["vezx"] * T2_["wuye"] * Lambda2_["xyuv"];
        temp["zw"] -= V_["vezx"] * T2_["wUeY"] * Lambda2_["xYvU"];
        temp["zw"] -= V_["vEzX"] * T2_["wUyE"] * Lambda2_["yXvU"];
        temp["zw"] -= V_["eVzX"] * T2_["wUeY"] * Lambda2_["XYUV"];
        temp["zw"] -= V_["eVzX"] * T2_["wuye"] * Lambda2_["yXuV"];
        temp["ZW"] -= V_["eVxZ"] * T2_["uWeY"] * Lambda2_["xYuV"];
        temp["ZW"] -= V_["vExZ"] * T2_["uWyE"] * Lambda2_["xyuv"];
        temp["ZW"] -= V_["vExZ"] * T2_["WUYE"] * Lambda2_["xYvU"];
        temp["ZW"] -= V_["VEZX"] * T2_["WUYE"] * Lambda2_["XYUV"];
        temp["ZW"] -= V_["VEZX"] * T2_["uWyE"] * Lambda2_["yXuV"];

        temp["wz"] -= V_["vewx"] * T2_["zuye"] * Lambda2_["xyuv"];
        temp["wz"] -= V_["vewx"] * T2_["zUeY"] * Lambda2_["xYvU"];
        temp["wz"] -= V_["vEwX"] * T2_["zUyE"] * Lambda2_["yXvU"];
        temp["wz"] -= V_["eVwX"] * T2_["zUeY"] * Lambda2_["XYUV"];
        temp["wz"] -= V_["eVwX"] * T2_["zuye"] * Lambda2_["yXuV"];
        temp["WZ"] -= V_["eVxW"] * T2_["uZeY"] * Lambda2_["xYuV"];
        temp["WZ"] -= V_["vExW"] * T2_["uZyE"] * Lambda2_["xyuv"];
        temp["WZ"] -= V_["vExW"] * T2_["ZUYE"] * Lambda2_["xYvU"];
        temp["WZ"] -= V_["VEWX"] * T2_["ZUYE"] * Lambda2_["XYUV"];
        temp["WZ"] -= V_["VEWX"] * T2_["uZyE"] * Lambda2_["yXuV"];

        temp["zw"] += V_["vwmx"] * T2_["muyz"] * Lambda2_["xyuv"];
        temp["zw"] += V_["vwmx"] * T2_["mUzY"] * Lambda2_["xYvU"];
        temp["ZW"] += V_["vWmX"] * T2_["mUyZ"] * Lambda2_["yXvU"];
        temp["zw"] += V_["wVmX"] * T2_["mUzY"] * Lambda2_["XYUV"];
        temp["zw"] += V_["wVmX"] * T2_["muyz"] * Lambda2_["yXuV"];
        temp["zw"] += V_["wVxM"] * T2_["uMzY"] * Lambda2_["xYuV"];
        temp["ZW"] += V_["vWxM"] * T2_["uMyZ"] * Lambda2_["xyuv"];
        temp["ZW"] += V_["vWxM"] * T2_["MUYZ"] * Lambda2_["xYvU"];
        temp["ZW"] += V_["VWMX"] * T2_["MUYZ"] * Lambda2_["XYUV"];
        temp["ZW"] += V_["VWMX"] * T2_["uMyZ"] * Lambda2_["yXuV"];

        temp["zw"] += V_["vzmx"] * T2_["muyw"] * Lambda2_["xyuv"];
        temp["zw"] += V_["vzmx"] * T2_["mUwY"] * Lambda2_["xYvU"];
        temp["ZW"] += V_["vZmX"] * T2_["mUyW"] * Lambda2_["yXvU"];
        temp["zw"] += V_["zVmX"] * T2_["mUwY"] * Lambda2_["XYUV"];
        temp["zw"] += V_["zVmX"] * T2_["muyw"] * Lambda2_["yXuV"];
        temp["zw"] += V_["zVxM"] * T2_["uMwY"] * Lambda2_["xYuV"];
        temp["ZW"] += V_["vZxM"] * T2_["uMyW"] * Lambda2_["xyuv"];
        temp["ZW"] += V_["vZxM"] * T2_["MUYW"] * Lambda2_["xYvU"];
        temp["ZW"] += V_["VZMX"] * T2_["MUYW"] * Lambda2_["XYUV"];
        temp["ZW"] += V_["VZMX"] * T2_["uMyW"] * Lambda2_["yXuV"];
    }

    if (CORRELATION_TERM) {
        temp["vu"] += Sigma3["ia"] * V["auiv"];
        temp["vu"] += Sigma3["IA"] * V["uAvI"];
        temp["VU"] += Sigma3["IA"] * V["AUIV"];
        temp["VU"] += Sigma3["ia"] * V["aUiV"];

        temp["vu"] += Sigma3["ia"] * V["aviu"];
        temp["vu"] += Sigma3["IA"] * V["vAuI"];
        temp["VU"] += Sigma3["IA"] * V["AVIU"];
        temp["VU"] += Sigma3["ia"] * V["aViU"];

        temp["xu"] += Sigma2["ia"] * Delta1["xu"] * T2_["iuax"];
        temp["xu"] += Sigma2["IA"] * Delta1["xu"] * T2_["uIxA"];
        temp["XU"] += Sigma2["IA"] * Delta1["XU"] * T2_["IUAX"];
        temp["XU"] += Sigma2["ia"] * Delta1["XU"] * T2_["iUaX"];

        temp["xu"] += Sigma2["ia"] * Delta1["ux"] * T2_["ixau"];
        temp["xu"] += Sigma2["IA"] * Delta1["ux"] * T2_["xIuA"];
        temp["XU"] += Sigma2["IA"] * Delta1["UX"] * T2_["IXAU"];
        temp["XU"] += Sigma2["ia"] * Delta1["UX"] * T2_["iXaU"];
    }

    if (CORRELATION_TERM) {
        temp["vu"] += Xi3["ia"] * V["auiv"];
        temp["vu"] += Xi3["IA"] * V["uAvI"];
        temp["VU"] += Xi3["IA"] * V["AUIV"];
        temp["VU"] += Xi3["ia"] * V["aUiV"];

        temp["vu"] += Xi3["ia"] * V["aviu"];
        temp["vu"] += Xi3["IA"] * V["vAuI"];
        temp["VU"] += Xi3["IA"] * V["AVIU"];
        temp["VU"] += Xi3["ia"] * V["aViU"];

        temp["xu"] += Xi3["ia"] * Delta1["xu"] * T2_["iuax"];
        temp["xu"] += Xi3["IA"] * Delta1["xu"] * T2_["uIxA"];
        temp["XU"] += Xi3["IA"] * Delta1["XU"] * T2_["IUAX"];
        temp["XU"] += Xi3["ia"] * Delta1["XU"] * T2_["iUaX"];

        temp["xu"] += Xi3["ia"] * Delta1["ux"] * T2_["ixau"];
        temp["xu"] += Xi3["IA"] * Delta1["ux"] * T2_["xIuA"];
        temp["XU"] += Xi3["IA"] * Delta1["UX"] * T2_["IXAU"];
        temp["XU"] += Xi3["ia"] * Delta1["UX"] * T2_["iXaU"];  
    }

    if (X7_TERM) {
        temp["uv"] += F_["ev"] * T1_["ue"];
        temp["UV"] += F_["EV"] * T1_["UE"];

        temp["uv"] += F_["eu"] * T1_["ve"];
        temp["UV"] += F_["EU"] * T1_["VE"];

        temp["uv"] -= F_["vm"] * T1_["mu"];
        temp["UV"] -= F_["VM"] * T1_["MU"];

        temp["uv"] -= F_["um"] * T1_["mv"];
        temp["UV"] -= F_["UM"] * T1_["MV"];
    }

    b_ck("K") -= 0.5 * temp.block("aa")("uv") * cc1a_n("Kuv"); 
    b_ck("K") -= 0.5 * temp.block("AA")("UV") * cc1b_n("KUV");
    b_ck("K") -= 0.5 * temp.block("aa")("uv") * cc1a_r("Kuv"); 
    b_ck("K") -= 0.5 * temp.block("AA")("UV") * cc1b_r("KUV");
    Alpha += 0.5 * temp["uv"] * Gamma1_["uv"]; 
    Alpha += 0.5 * temp["UV"] * Gamma1_["UV"];
    temp.zero();

    BlockedTensor temp4 = BTF_->build(CoreTensor, "temporal tensor 4", {"aaaa","AAAA","aAaA"});

    if (X1_TERM) {
        temp4["uvxy"] += 0.125 * V_["efxy"] * T2_["uvef"];
        temp4["uvxy"] += 0.125 * V_["ewxy"] * T2_["uvez"] * Eta1_["zw"];
        temp4["uvxy"] += 0.125 * V_["wexy"] * T2_["uvze"] * Eta1_["zw"];

        temp4["UVXY"] += 0.125 * V_["EFXY"] * T2_["UVEF"];
        temp4["UVXY"] += 0.125 * V_["EWXY"] * T2_["UVEZ"] * Eta1_["ZW"];
        temp4["UVXY"] += 0.125 * V_["WEXY"] * T2_["UVZE"] * Eta1_["ZW"];

        temp4["uVxY"] += V_["eFxY"] * T2_["uVeF"];
        temp4["uVxY"] += V_["eWxY"] * T2_["uVeZ"] * Eta1_["ZW"];
        temp4["uVxY"] += V_["wExY"] * T2_["uVzE"] * Eta1_["zw"];
    }

    if (X2_TERM) {
        temp4["uvxy"] += 0.125 * V_["uvmn"] * T2_["mnxy"];
        temp4["uvxy"] += 0.125 * V_["uvmz"] * T2_["mwxy"] * Gamma1_["zw"];
        temp4["uvxy"] += 0.125 * V_["uvzm"] * T2_["wmxy"] * Gamma1_["zw"];

        temp4["UVXY"] += 0.125 * V_["UVMN"] * T2_["MNXY"];
        temp4["UVXY"] += 0.125 * V_["UVMZ"] * T2_["MWXY"] * Gamma1_["ZW"];
        temp4["UVXY"] += 0.125 * V_["UVZM"] * T2_["WMXY"] * Gamma1_["ZW"];

        temp4["uVxY"] += V_["uVmN"] * T2_["mNxY"];
        temp4["uVxY"] += V_["uVmZ"] * T2_["mWxY"] * Gamma1_["ZW"];
        temp4["uVxY"] += V_["uVzM"] * T2_["wMxY"] * Gamma1_["zw"];
    }

    if (X3_TERM) {
        temp4["xyuv"] -= V_["vemx"] * T2_["muye"];
        temp4["xyuv"] -= V_["vwmx"] * T2_["muyz"] * Eta1_["zw"];
        temp4["xyuv"] -= V_["vezx"] * T2_["wuye"] * Gamma1_["zw"];
        temp4["xyuv"] -= V_["vExM"] * T2_["uMyE"];
        temp4["xyuv"] -= V_["vWxM"] * T2_["uMyZ"] * Eta1_["ZW"];
        temp4["xyuv"] -= V_["vExZ"] * T2_["uWyE"] * Gamma1_["ZW"];
        temp4["XYUV"] -= V_["eVmX"] * T2_["mUeY"];
        temp4["XYUV"] -= V_["wVmX"] * T2_["mUzY"] * Eta1_["zw"];
        temp4["XYUV"] -= V_["eVzX"] * T2_["wUeY"] * Gamma1_["zw"];
        temp4["XYUV"] -= V_["VEMX"] * T2_["MUYE"];
        temp4["XYUV"] -= V_["VWMX"] * T2_["MUYZ"] * Eta1_["ZW"];
        temp4["XYUV"] -= V_["VEZX"] * T2_["WUYE"] * Gamma1_["ZW"];
        temp4["xYvU"] -= V_["vemx"] * T2_["mUeY"];
        temp4["xYvU"] -= V_["vwmx"] * T2_["mUzY"] * Eta1_["zw"];
        temp4["xYvU"] -= V_["vezx"] * T2_["wUeY"] * Gamma1_["zw"];
        temp4["xYvU"] -= V_["vExM"] * T2_["MUYE"];
        temp4["xYvU"] -= V_["vWxM"] * T2_["MUYZ"] * Eta1_["ZW"];
        temp4["xYvU"] -= V_["vExZ"] * T2_["WUYE"] * Gamma1_["ZW"];
        temp4["yXuV"] -= V_["eVmX"] * T2_["muye"];
        temp4["yXuV"] -= V_["wVmX"] * T2_["muyz"] * Eta1_["zw"];
        temp4["yXuV"] -= V_["eVzX"] * T2_["wuye"] * Gamma1_["zw"];
        temp4["yXuV"] -= V_["VEMX"] * T2_["uMyE"];
        temp4["yXuV"] -= V_["VWMX"] * T2_["uMyZ"] * Eta1_["ZW"];
        temp4["yXuV"] -= V_["VEZX"] * T2_["uWyE"] * Gamma1_["ZW"];
        temp4["yXvU"] -= V_["vEmX"] * T2_["mUyE"];
        temp4["yXvU"] -= V_["vWmX"] * T2_["mUyZ"] * Eta1_["ZW"];
        temp4["yXvU"] -= V_["vEzX"] * T2_["wUyE"] * Gamma1_["zw"];
        temp4["xYuV"] -= V_["eVxM"] * T2_["uMeY"];
        temp4["xYuV"] -= V_["wVxM"] * T2_["uMzY"] * Eta1_["zw"];
        temp4["xYuV"] -= V_["eVxZ"] * T2_["uWeY"] * Gamma1_["ZW"];
    }

    if (X5_TERM) {
        temp4["uvxy"] += 0.5 * F_["ex"] * T2_["uvey"];
        temp4["UVXY"] += 0.5 * F_["EX"] * T2_["UVEY"];
        temp4["xYuV"] += F_["ex"] * T2_["uVeY"];
        temp4["yXuV"] += F_["EX"] * T2_["uVyE"];
        temp4["uvxy"] -= 0.5 * F_["vm"] * T2_["umxy"];
        temp4["UVXY"] -= 0.5 * F_["VM"] * T2_["UMXY"];
        temp4["xYvU"] -= F_["vm"] * T2_["mUxY"];
        temp4["xYuV"] -= F_["VM"] * T2_["uMxY"];
    }

    if (X6_TERM) {
        temp4["uvxy"] += 0.5 * T1_["ue"] * V_["evxy"];
        temp4["UVXY"] += 0.5 * T1_["UE"] * V_["EVXY"];
        temp4["xYuV"] += T1_["ue"] * V_["eVxY"];
        temp4["xYvU"] += T1_["UE"] * V_["vExY"];
        temp4["uvxy"] -= 0.5 * T1_["mx"] * V_["uvmy"];
        temp4["UVXY"] -= 0.5 * T1_["MX"] * V_["UVMY"];
        temp4["xYuV"] -= T1_["mx"] * V_["uVmY"];
        temp4["yXuV"] -= T1_["MX"] * V_["uVyM"];
    }

    b_ck("K") -= temp4.block("aaaa")("uvxy") * dlamb_aa("Kxyuv");
    b_ck("K") -= temp4.block("AAAA")("UVXY") * dlamb_bb("KXYUV");
    b_ck("K") -= temp4.block("aAaA")("uVxY") * dlamb_ab("KxYuV");
    Alpha += 2 * temp4["uvxy"] * Lambda2_["xyuv"];
    Alpha += 2 * temp4["UVXY"] * Lambda2_["XYUV"];
    Alpha += 2 * temp4["uVxY"] * Lambda2_["xYuV"];
    Alpha -= temp4["uvxy"] * Gamma2_["xyuv"];
    Alpha -= temp4["UVXY"] * Gamma2_["XYUV"];
    Alpha -= temp4["uVxY"] * Gamma2_["xYuV"];
    temp4.zero();

    if (X4_TERM) {
        b_ck("K") -= 0.25 * V_.block("aaca")("uvmz") * T2_.block("caaa")("mwxy") * dlamb3_aaa("Kxyzuvw");
        b_ck("K") -= 0.25 * V_.block("AACA")("UVMZ") * T2_.block("CAAA")("MWXY") * dlamb3_bbb("KXYZUVW");
        b_ck("K") += 0.50 * V_.block("aaca")("uvmy") * T2_.block("cAaA")("mWxZ") * dlamb3_aab("KxyZuvW"); 
        b_ck("K") += 0.50 * V_.block("aAcA")("uWmZ") * T2_.block("caaa")("mvxy") * dlamb3_aab("KxyZuvW"); 
        b_ck("K") -= 1.00 * V_.block("aAaC")("uWyM") * T2_.block("aCaA")("vMxZ") * dlamb3_aab("KxyZuvW"); 
        b_ck("K") += 0.50 * V_.block("AACA")("VWMZ") * T2_.block("aCaA")("uMxY") * dlamb3_abb("KxYZuVW");
        b_ck("K") += 0.50 * V_.block("aAaC")("uVxM") * T2_.block("CAAA")("MWYZ") * dlamb3_abb("KxYZuVW");
        b_ck("K") -= 1.00 * V_.block("aAcA")("uVmZ") * T2_.block("cAaA")("mWxY") * dlamb3_abb("KxYZuVW");  
        b_ck("K") += 0.25 * V_.block("vaaa")("ewxy") * T2_.block("aava")("uvez") * dlamb3_aaa("Kxyzuvw");
        b_ck("K") += 0.25 * V_.block("VAAA")("EWXY") * T2_.block("AAVA")("UVEZ") * dlamb3_bbb("KXYZUVW");
        b_ck("K") -= 0.50 * V_.block("vAaA")("eWxZ") * T2_.block("aava")("uvey") * dlamb3_aab("KxyZuvW");
        b_ck("K") += 0.50 * V_.block("avaa")("vexy") * T2_.block("aAvA")("uWeZ") * dlamb3_aab("KxyZuvW");
        b_ck("K") += 1.00 * V_.block("aVaA")("vExZ") * T2_.block("aAaV")("uWyE") * dlamb3_aab("KxyZuvW");
        b_ck("K") -= 0.50 * V_.block("aVaA")("uExY") * T2_.block("AAVA")("VWEZ") * dlamb3_abb("KxYZuVW");
        b_ck("K") += 0.50 * V_.block("AVAA")("WEYZ") * T2_.block("aAaV")("uVxE") * dlamb3_abb("KxYZuVW");
        b_ck("K") += 1.00 * V_.block("vAaA")("eWxY") * T2_.block("aAvA")("uVeZ") * dlamb3_abb("KxYZuVW");
    }

    if (X4_TERM) {
        Alpha += 0.25 * V_.block("aaca")("uvmz") * T2_.block("caaa")("mwxy") * rdms_.g3aaa()("xyzuvw");
        Alpha += 0.25 * V_.block("AACA")("UVMZ") * T2_.block("CAAA")("MWXY") * rdms_.g3bbb()("XYZUVW");
        Alpha -= 0.50 * V_.block("aaca")("uvmy") * T2_.block("cAaA")("mWxZ") * rdms_.g3aab()("xyZuvW"); 
        Alpha -= 0.50 * V_.block("aAcA")("uWmZ") * T2_.block("caaa")("mvxy") * rdms_.g3aab()("xyZuvW"); 
        Alpha += 1.00 * V_.block("aAaC")("uWyM") * T2_.block("aCaA")("vMxZ") * rdms_.g3aab()("xyZuvW"); 
        Alpha -= 0.50 * V_.block("AACA")("VWMZ") * T2_.block("aCaA")("uMxY") * rdms_.g3abb()("xYZuVW");
        Alpha -= 0.50 * V_.block("aAaC")("uVxM") * T2_.block("CAAA")("MWYZ") * rdms_.g3abb()("xYZuVW");
        Alpha += 1.00 * V_.block("aAcA")("uVmZ") * T2_.block("cAaA")("mWxY") * rdms_.g3abb()("xYZuVW");

        Alpha -= 1.0 * V_["uvmz"] * T2_["mwxy"] * Gamma1_["uz"] * Gamma2_["xyvw"];
        Alpha -= 2.0 * V_["uvmz"] * T2_["mWxY"] * Gamma1_["uz"] * Gamma2_["xYvW"];
        Alpha += 1.0 * V_["uVzM"] * T2_["MWXY"] * Gamma1_["uz"] * Gamma2_["XYVW"];
        Alpha += 2.0 * V_["uVzM"] * T2_["wMxY"] * Gamma1_["uz"] * Gamma2_["xYwV"];
        Alpha -= 1.0 * V_["UVMZ"] * T2_["MWXY"] * Gamma1_["UZ"] * Gamma2_["XYVW"];
        Alpha -= 2.0 * V_["UVMZ"] * T2_["wMxY"] * Gamma1_["UZ"] * Gamma2_["xYwV"];  
        Alpha += 1.0 * V_["vUmZ"] * T2_["mwxy"] * Gamma1_["UZ"] * Gamma2_["xyvw"];
        Alpha += 2.0 * V_["vUmZ"] * T2_["mWxY"] * Gamma1_["UZ"] * Gamma2_["xYvW"];

        Alpha -= 0.5 * V_["uvmz"] * T2_["mwxy"] * Gamma1_["wz"] * Gamma2_["xyuv"];
        Alpha -= 2.0 * V_["uVzM"] * T2_["wMxY"] * Gamma1_["wz"] * Gamma2_["xYuV"];
        Alpha -= 0.5 * V_["UVMZ"] * T2_["MWXY"] * Gamma1_["WZ"] * Gamma2_["XYUV"];
        Alpha -= 2.0 * V_["uVmZ"] * T2_["mWxY"] * Gamma1_["WZ"] * Gamma2_["xYuV"];

        Alpha += 2.0 * V_["uvmz"] * T2_["mwxy"] * Gamma1_["xu"] * Gamma2_["vwzy"];
        Alpha += 2.0 * V_["uVmZ"] * T2_["mwxy"] * Gamma1_["xu"] * Gamma2_["wVyZ"];
        Alpha += 2.0 * V_["uvmz"] * T2_["mWxY"] * Gamma1_["xu"] * Gamma2_["vWzY"];
        Alpha += 2.0 * V_["uVmZ"] * T2_["mWxY"] * Gamma1_["xu"] * Gamma2_["VWZY"];
        Alpha -= 2.0 * V_["uVzM"] * T2_["wMxY"] * Gamma1_["xu"] * Gamma2_["wVzY"];

        Alpha += 2.0 * V_["UVMZ"] * T2_["MWXY"] * Gamma1_["XU"] * Gamma2_["VWZY"];
        Alpha += 2.0 * V_["vUzM"] * T2_["MWXY"] * Gamma1_["XU"] * Gamma2_["vWzY"];
        Alpha += 2.0 * V_["UVMZ"] * T2_["wMyX"] * Gamma1_["XU"] * Gamma2_["wVyZ"];
        Alpha += 2.0 * V_["vUzM"] * T2_["wMyX"] * Gamma1_["XU"] * Gamma2_["vwzy"];
        Alpha -= 2.0 * V_["vUmZ"] * T2_["mWyX"] * Gamma1_["XU"] * Gamma2_["vWyZ"];

        Alpha += 1.0 * V_["uvmz"] * T2_["mwxy"] * Gamma1_["xw"] * Gamma2_["uvzy"];
        Alpha -= 2.0 * V_["uVmZ"] * T2_["mwxy"] * Gamma1_["xw"] * Gamma2_["uVyZ"];
        Alpha -= 1.0 * V_["UVMZ"] * T2_["wMxY"] * Gamma1_["xw"] * Gamma2_["UVZY"];
        Alpha += 2.0 * V_["uVzM"] * T2_["wMxY"] * Gamma1_["xw"] * Gamma2_["uVzY"];

        Alpha -= 1.0 * V_["uvmz"] * T2_["mWyX"] * Gamma1_["XW"] * Gamma2_["uvzy"];
        Alpha += 2.0 * V_["uVmZ"] * T2_["mWyX"] * Gamma1_["XW"] * Gamma2_["uVyZ"];
        Alpha += 1.0 * V_["UVMZ"] * T2_["MWXY"] * Gamma1_["XW"] * Gamma2_["UVZY"];
        Alpha -= 2.0 * V_["uVzM"] * T2_["MWXY"] * Gamma1_["XW"] * Gamma2_["uVzY"];

        Alpha += 6 * V_["uvmz"] * T2_["mwxy"] * Gamma1_["uz"] * Gamma1_["xv"] * Gamma1_["yw"];
        Alpha += 6 * V_["uvmz"] * T2_["mWxY"] * Gamma1_["uz"] * Gamma1_["xv"] * Gamma1_["YW"];
        Alpha -= 6 * V_["vUmZ"] * T2_["mwxy"] * Gamma1_["UZ"] * Gamma1_["xv"] * Gamma1_["yw"];
        Alpha -= 6 * V_["vUmZ"] * T2_["mWxY"] * Gamma1_["UZ"] * Gamma1_["xv"] * Gamma1_["YW"];

        Alpha += 6 * V_["UVMZ"] * T2_["MWXY"] * Gamma1_["UZ"] * Gamma1_["XV"] * Gamma1_["YW"];
        Alpha += 6 * V_["UVMZ"] * T2_["wMyX"] * Gamma1_["UZ"] * Gamma1_["XV"] * Gamma1_["yw"];
        Alpha -= 6 * V_["uVzM"] * T2_["MWXY"] * Gamma1_["uz"] * Gamma1_["XV"] * Gamma1_["YW"];
        Alpha -= 6 * V_["uVzM"] * T2_["wMyX"] * Gamma1_["uz"] * Gamma1_["XV"] * Gamma1_["yw"];

        Alpha += 3.0 * V_["uvmz"] * T2_["mwxy"] * Gamma1_["wz"] * Gamma1_["xu"] * Gamma1_["yv"];
        Alpha += 6.0 * V_["uVmZ"] * T2_["mWxY"] * Gamma1_["WZ"] * Gamma1_["xu"] * Gamma1_["YV"];
        Alpha += 3.0 * V_["UVMZ"] * T2_["MWXY"] * Gamma1_["WZ"] * Gamma1_["XU"] * Gamma1_["YV"];
        Alpha += 6.0 * V_["uVzM"] * T2_["wMxY"] * Gamma1_["wz"] * Gamma1_["xu"] * Gamma1_["YV"];

        Alpha -= 0.25 * V_.block("vaaa")("ewxy") * T2_.block("aava")("uvez") * rdms_.g3aaa()("xyzuvw");
        Alpha -= 0.25 * V_.block("VAAA")("EWXY") * T2_.block("AAVA")("UVEZ") * rdms_.g3bbb()("XYZUVW");
        Alpha += 0.50 * V_.block("vAaA")("eWxZ") * T2_.block("aava")("uvey") * rdms_.g3aab()("xyZuvW");
        Alpha -= 0.50 * V_.block("avaa")("vexy") * T2_.block("aAvA")("uWeZ") * rdms_.g3aab()("xyZuvW");
        Alpha -= 1.00 * V_.block("aVaA")("vExZ") * T2_.block("aAaV")("uWyE") * rdms_.g3aab()("xyZuvW");
        Alpha += 0.50 * V_.block("aVaA")("uExY") * T2_.block("AAVA")("VWEZ") * rdms_.g3abb()("xYZuVW");
        Alpha -= 0.50 * V_.block("AVAA")("WEYZ") * T2_.block("aAaV")("uVxE") * rdms_.g3abb()("xYZuVW");
        Alpha -= 1.00 * V_.block("vAaA")("eWxY") * T2_.block("aAvA")("uVeZ") * rdms_.g3abb()("xYZuVW");

        Alpha += 1.0 * V_["ezuv"] * T2_["xyew"] * Gamma1_["uz"] * Gamma2_["xyvw"];
        Alpha += 2.0 * V_["ezuv"] * T2_["xYeW"] * Gamma1_["uz"] * Gamma2_["xYvW"];
        Alpha -= 1.0 * V_["zEuV"] * T2_["XYEW"] * Gamma1_["uz"] * Gamma2_["XYVW"];
        Alpha -= 2.0 * V_["zEuV"] * T2_["xYwE"] * Gamma1_["uz"] * Gamma2_["xYwV"];
        Alpha += 1.0 * V_["EZUV"] * T2_["XYEW"] * Gamma1_["UZ"] * Gamma2_["XYVW"];
        Alpha += 2.0 * V_["EZUV"] * T2_["xYwE"] * Gamma1_["UZ"] * Gamma2_["xYwV"];  
        Alpha -= 1.0 * V_["eZvU"] * T2_["xyew"] * Gamma1_["UZ"] * Gamma2_["xyvw"];
        Alpha -= 2.0 * V_["eZvU"] * T2_["xYeW"] * Gamma1_["UZ"] * Gamma2_["xYvW"];

        Alpha += 0.5 * V_["ezuv"] * T2_["xyew"] * Gamma1_["wz"] * Gamma2_["xyuv"];
        Alpha += 2.0 * V_["zEuV"] * T2_["xYwE"] * Gamma1_["wz"] * Gamma2_["xYuV"];
        Alpha += 0.5 * V_["EZUV"] * T2_["XYEW"] * Gamma1_["WZ"] * Gamma2_["XYUV"];
        Alpha += 2.0 * V_["eZuV"] * T2_["xYeW"] * Gamma1_["WZ"] * Gamma2_["xYuV"];

        Alpha -= 2.0 * V_["ezuv"] * T2_["xyew"] * Gamma1_["xu"] * Gamma2_["vwzy"];
        Alpha -= 2.0 * V_["eZuV"] * T2_["xyew"] * Gamma1_["xu"] * Gamma2_["wVyZ"];
        Alpha -= 2.0 * V_["ezuv"] * T2_["xYeW"] * Gamma1_["xu"] * Gamma2_["vWzY"];
        Alpha -= 2.0 * V_["eZuV"] * T2_["xYeW"] * Gamma1_["xu"] * Gamma2_["VWZY"];
        Alpha += 2.0 * V_["zEuV"] * T2_["xYwE"] * Gamma1_["xu"] * Gamma2_["wVzY"];
        Alpha -= 2.0 * V_["EZUV"] * T2_["XYEW"] * Gamma1_["XU"] * Gamma2_["VWZY"];
        Alpha -= 2.0 * V_["zEvU"] * T2_["XYEW"] * Gamma1_["XU"] * Gamma2_["vWzY"];
        Alpha -= 2.0 * V_["EZUV"] * T2_["yXwE"] * Gamma1_["XU"] * Gamma2_["wVyZ"];
        Alpha -= 2.0 * V_["zEvU"] * T2_["yXwE"] * Gamma1_["XU"] * Gamma2_["vwzy"];
        Alpha += 2.0 * V_["eZvU"] * T2_["yXeW"] * Gamma1_["XU"] * Gamma2_["vWyZ"];

        Alpha -= 1.0 * V_["ezuv"] * T2_["xyew"] * Gamma1_["xw"] * Gamma2_["uvzy"];
        Alpha += 2.0 * V_["eZuV"] * T2_["xyew"] * Gamma1_["xw"] * Gamma2_["uVyZ"];
        Alpha += 1.0 * V_["EZUV"] * T2_["xYwE"] * Gamma1_["xw"] * Gamma2_["UVZY"];
        Alpha -= 2.0 * V_["zEuV"] * T2_["xYwE"] * Gamma1_["xw"] * Gamma2_["uVzY"];
        Alpha += 1.0 * V_["ezuv"] * T2_["yXeW"] * Gamma1_["XW"] * Gamma2_["uvzy"];
        Alpha -= 2.0 * V_["eZuV"] * T2_["yXeW"] * Gamma1_["XW"] * Gamma2_["uVyZ"];
        Alpha -= 1.0 * V_["EZUV"] * T2_["XYEW"] * Gamma1_["XW"] * Gamma2_["UVZY"];
        Alpha += 2.0 * V_["zEuV"] * T2_["XYEW"] * Gamma1_["XW"] * Gamma2_["uVzY"];

        Alpha -= 6 * V_["ezuv"] * T2_["xyew"] * Gamma1_["uz"] * Gamma1_["xv"] * Gamma1_["yw"];
        Alpha -= 6 * V_["ezuv"] * T2_["xYeW"] * Gamma1_["uz"] * Gamma1_["xv"] * Gamma1_["YW"];
        Alpha += 6 * V_["eZvU"] * T2_["xyew"] * Gamma1_["UZ"] * Gamma1_["xv"] * Gamma1_["yw"];
        Alpha += 6 * V_["eZvU"] * T2_["xYeW"] * Gamma1_["UZ"] * Gamma1_["xv"] * Gamma1_["YW"];
        Alpha -= 6 * V_["EZUV"] * T2_["XYEW"] * Gamma1_["UZ"] * Gamma1_["XV"] * Gamma1_["YW"];
        Alpha -= 6 * V_["EZUV"] * T2_["yXwE"] * Gamma1_["UZ"] * Gamma1_["XV"] * Gamma1_["yw"];
        Alpha += 6 * V_["zEuV"] * T2_["XYEW"] * Gamma1_["uz"] * Gamma1_["XV"] * Gamma1_["YW"];
        Alpha += 6 * V_["zEuV"] * T2_["yXwE"] * Gamma1_["uz"] * Gamma1_["XV"] * Gamma1_["yw"];

        Alpha -= 3.0 * V_["ezuv"] * T2_["xyew"] * Gamma1_["wz"] * Gamma1_["xu"] * Gamma1_["yv"];
        Alpha -= 6.0 * V_["eZuV"] * T2_["xYeW"] * Gamma1_["WZ"] * Gamma1_["xu"] * Gamma1_["YV"];
        Alpha -= 3.0 * V_["EZUV"] * T2_["XYEW"] * Gamma1_["WZ"] * Gamma1_["XU"] * Gamma1_["YV"];
        Alpha -= 6.0 * V_["zEuV"] * T2_["xYwE"] * Gamma1_["wz"] * Gamma1_["xu"] * Gamma1_["YV"];
    }

    Alpha += Z["mn"] * V["m,a1,n,u1"] * Gamma1_["u1,a1"];
    Alpha += Z["mn"] * V["m,A1,n,U1"] * Gamma1_["U1,A1"];
    Alpha += Z["MN"] * V["M,A1,N,U1"] * Gamma1_["U1,A1"];
    Alpha += Z["MN"] * V["a1,M,u1,N"] * Gamma1_["u1,a1"];

    Alpha += Z["ef"] * V["e,a1,f,u1"] * Gamma1_["u1,a1"];
    Alpha += Z["ef"] * V["e,A1,f,U1"] * Gamma1_["U1,A1"];
    Alpha += Z["EF"] * V["E,A1,F,U1"] * Gamma1_["U1,A1"];
    Alpha += Z["EF"] * V["a1,E,u1,F"] * Gamma1_["u1,a1"];

    Alpha += temp3["uv"] * V["u,a1,v,u1"] * Gamma1_["u1,a1"];
    Alpha += temp3["uv"] * V["u,A1,v,U1"] * Gamma1_["U1,A1"];
    Alpha += temp3["UV"] * V["U,A1,V,U1"] * Gamma1_["U1,A1"];
    Alpha += temp3["UV"] * V["a1,U,u1,V"] * Gamma1_["u1,a1"];

    b_ck("K") += 2 * Alpha * ci("K");

    b_ck("K") -= 2 * temp3.block("aa")("xy") * V.block("aaaa")("xvyu") * cc1a_n("Kuv"); 
    b_ck("K") -= 2 * temp3.block("aa")("xy") * V.block("aAaA")("xVyU") * cc1b_n("KUV"); 
    b_ck("K") -= 2 * temp3.block("AA")("XY") * V.block("AAAA")("XVYU") * cc1b_n("KUV"); 
    b_ck("K") -= 2 * temp3.block("AA")("XY") * V.block("aAaA")("vXuY") * cc1a_n("Kuv"); 

    temp["uv"] += 2 * Z["mn"] * V["mvnu"];
    temp["uv"] += 2 * Z["MN"] * V["vMuN"];
    temp["uv"] += 2 * Z["ef"] * V["evfu"];
    temp["uv"] += 2 * Z["EF"] * V["vEuF"];
    temp["UV"] += 2 * Z["MN"] * V["MVNU"];
    temp["UV"] += 2 * Z["mn"] * V["mVnU"];
    temp["UV"] += 2 * Z["EF"] * V["EVFU"];
    temp["UV"] += 2 * Z["ef"] * V["eVfU"];

    b_ck("K") -= temp.block("aa")("uv") * cc1a_n("Kuv"); 
    b_ck("K") -= temp.block("AA")("UV") * cc1b_n("KUV"); 

    for (const std::string& block : {"ci"}) {
        (b_ck).iterate([&](const std::vector<size_t>& i, double& value) {
            int index = preidx[block] + i[0];
            b.at(index) = value;
        });
    } 

    auto ck_vc_a = ambit::Tensor::build(ambit::CoreTensor, "ci equations z{vc} alpha part", {ndets, nvirt, ncore});
    auto ck_ca_a = ambit::Tensor::build(ambit::CoreTensor, "ci equations z{ca} alpha part", {ndets, ncore, na});
    auto ck_va_a = ambit::Tensor::build(ambit::CoreTensor, "ci equations z{va} alpha part", {ndets, nvirt, na});
    auto ck_aa_a = ambit::Tensor::build(ambit::CoreTensor, "ci equations z{aa} alpha part", {ndets, na, na});

    auto ck_vc_b = ambit::Tensor::build(ambit::CoreTensor, "ci equations z{vc} beta part", {ndets, nvirt, ncore});
    auto ck_ca_b = ambit::Tensor::build(ambit::CoreTensor, "ci equations z{ca} beta part", {ndets, ncore, na});
    auto ck_va_b = ambit::Tensor::build(ambit::CoreTensor, "ci equations z{va} beta part", {ndets, nvirt, na});
    auto ck_aa_b = ambit::Tensor::build(ambit::CoreTensor, "ci equations z{aa} beta part", {ndets, na, na});

    // virtual-core
    ck_vc_a("Kem") += 2 * V.block("vaca")("eumv") * cc1a_n("Kuv");
    ck_vc_a("Kem") += 2 * V.block("vAcA")("eUmV") * cc1b_n("KUV");
    ck_vc_a("Kem") += 2 * V.block("vaca")("eumv") * cc1a_r("Kuv");
    ck_vc_a("Kem") += 2 * V.block("vAcA")("eUmV") * cc1b_r("KUV");

    ck_vc_b("KEM") += 2 * V.block("VACA")("EUMV") * cc1b_n("KUV");
    ck_vc_b("KEM") += 2 * V.block("aVaC")("uEvM") * cc1a_n("Kuv");
    ck_vc_b("KEM") += 2 * V.block("VACA")("EUMV") * cc1b_r("KUV");
    ck_vc_b("KEM") += 2 * V.block("aVaC")("uEvM") * cc1a_r("Kuv");

    // contribution from Alpha
    ck_vc_a("Kem") += -4 * ci("K") * V.block("vaca")("exmy") * Gamma1_.block("aa")("xy");
    ck_vc_a("Kem") += -4 * ci("K") * V.block("vAcA")("eXmY") * Gamma1_.block("AA")("XY");
    ck_vc_b("KEM") += -4 * ci("K") * V.block("VACA")("EXMY") * Gamma1_.block("AA")("XY");
    ck_vc_b("KEM") += -4 * ci("K") * V.block("aVaC")("xEyM") * Gamma1_.block("aa")("xy");

    // core-active
    ck_ca_a("Knu") += 2 * V.block("aaca")("uynx") * cc1a_n("Kxy");
    ck_ca_a("Knu") += 2 * V.block("aAcA")("uYnX") * cc1b_n("KXY");
    ck_ca_a("Knu") -= 2 * H.block("ac")("vn") * cc1a_n("Kuv");
    ck_ca_a("Knu") -= 2 * V_N_Alpha.block("ac")("vn") * cc1a_n("Kuv");
    ck_ca_a("Knu") -= 2 * V_N_Beta.block("ac")("vn")  * cc1a_n("Kuv");
    ck_ca_a("Knu") -= V.block("aaca")("xynv") * cc2aa_n("Kuvxy");
    ck_ca_a("Knu") -= V.block("aAcA")("xYnV") * cc2ab_n("KuVxY");
    ck_ca_a("Knu") -= V.block("aAcA")("xYnV") * cc2ab_n("KuVxY");
    ck_ca_a("Knu") += 2 * V.block("aaca")("uynx") * cc1a_r("Kxy");
    ck_ca_a("Knu") += 2 * V.block("aAcA")("uYnX") * cc1b_r("KXY");
    ck_ca_a("Knu") -= 2 * H.block("ac")("vn") * cc1a_r("Kuv");
    ck_ca_a("Knu") -= 2 * V_N_Alpha.block("ac")("vn") * cc1a_r("Kuv");
    ck_ca_a("Knu") -= 2 * V_N_Beta.block("ac")("vn")  * cc1a_r("Kuv");
    ck_ca_a("Knu") -= V.block("aaca")("xynv") * cc2aa_r("Kuvxy");
    ck_ca_a("Knu") -= V.block("aAcA")("xYnV") * cc2ab_r("KuVxY");
    ck_ca_a("Knu") -= V.block("aAcA")("xYnV") * cc2ab_r("KuVxY");

    // NOTICE beta
    ck_ca_b("KNU") += 2 * V.block("AACA")("UYNX") * cc1b_n("KXY");
    ck_ca_b("KNU") += 2 * V.block("aAaC")("yUxN") * cc1a_n("Kxy");
    ck_ca_b("KNU") -= 2 * H.block("AC")("VN") * cc1b_n("KUV");
    ck_ca_b("KNU") -= 2 * V_all_Beta.block("AC")("VN") * cc1b_n("KUV");
    ck_ca_b("KNU") -= 2 * V_R_Beta.block("AC")("VN")   * cc1b_n("KUV");
    ck_ca_b("KNU") -= V.block("AACA")("XYNV") * cc2bb_n("KUVXY");
    ck_ca_b("KNU") -= 2 * V.block("aAaC")("xYvN") * cc2ab_n("KvUxY");
    ck_ca_b("KNU") += 2 * V.block("AACA")("UYNX") * cc1b_r("KXY");
    ck_ca_b("KNU") += 2 * V.block("aAaC")("yUxN") * cc1a_r("Kxy");
    ck_ca_b("KNU") -= 2 * H.block("AC")("VN") * cc1b_r("KUV");
    ck_ca_b("KNU") -= 2 * V_all_Beta.block("AC")("VN") * cc1b_r("KUV");
    ck_ca_b("KNU") -= 2 * V_R_Beta.block("AC")("VN")   * cc1b_r("KUV");
    ck_ca_b("KNU") -= V.block("AACA")("XYNV") * cc2bb_r("KUVXY");
    ck_ca_b("KNU") -= 2 * V.block("aAaC")("xYvN") * cc2ab_r("KvUxY");

    // contribution from Alpha
    ck_ca_a("Knu") += -4 * ci("K") * V.block("aaca")("uynx") * Gamma1_.block("aa")("xy");
    ck_ca_a("Knu") += -4 * ci("K") * V.block("aAcA")("uYnX") * Gamma1_.block("AA")("XY");
    ck_ca_a("Knu") +=  4 * ci("K") * H.block("ac")("vn") * Gamma1_.block("aa")("uv");
    ck_ca_a("Knu") +=  4 * ci("K") * V_N_Alpha.block("ac")("vn") * Gamma1_.block("aa")("uv");
    ck_ca_a("Knu") +=  4 * ci("K") * V_N_Beta.block("ac")("vn") * Gamma1_.block("aa")("uv");
    ck_ca_a("Knu") +=  2 * ci("K") * V.block("aaca")("xynv") * Gamma2_.block("aaaa")("uvxy");
    ck_ca_a("Knu") +=  4 * ci("K") * V.block("aAcA")("xYnV") * Gamma2_.block("aAaA")("uVxY");

    // NOTICE beta
    ck_ca_b("KNU") += -4 * ci("K") * V.block("AACA")("UYNX") * Gamma1_.block("AA")("XY");
    ck_ca_b("KNU") += -4 * ci("K") * V.block("aAaC")("yUxN") * Gamma1_.block("aa")("xy");
    ck_ca_b("KNU") +=  4 * ci("K") * H.block("AC")("VN") * Gamma1_.block("AA")("UV");
    ck_ca_b("KNU") +=  4 * ci("K") * V_all_Beta.block("AC")("VN") * Gamma1_.block("AA")("UV");
    ck_ca_b("KNU") +=  4 * ci("K") * V_R_Beta.block("AC")("VN") * Gamma1_.block("AA")("UV");
    ck_ca_b("KNU") +=  2 * ci("K") * V.block("AACA")("XYNV") * Gamma2_.block("AAAA")("UVXY");
    ck_ca_b("KNU") +=  4 * ci("K") * V.block("aAaC")("xYvN") * Gamma2_.block("aAaA")("vUxY");

    // virtual-active
    ck_va_a("Keu") += 2 * H.block("av")("ve") * cc1a_n("Kuv");
    ck_va_a("Keu") += 2 * V_N_Alpha.block("av")("ve") * cc1a_n("Kuv");
    ck_va_a("Keu") += 2 * V_N_Beta.block("av")("ve")  * cc1a_n("Kuv");
    ck_va_a("Keu") += V.block("vaaa")("evxy") * cc2aa_n("Kuvxy");
    ck_va_a("Keu") += 2 * V.block("vAaA")("eVxY") * cc2ab_n("KuVxY");
    ck_va_a("Keu") += 2 * H.block("av")("ve") * cc1a_r("Kuv");
    ck_va_a("Keu") += 2 * V_N_Alpha.block("av")("ve") * cc1a_r("Kuv");
    ck_va_a("Keu") += 2 * V_N_Beta.block("av")("ve")  * cc1a_r("Kuv");
    ck_va_a("Keu") += V.block("vaaa")("evxy") * cc2aa_r("Kuvxy");
    ck_va_a("Keu") += 2 * V.block("vAaA")("eVxY") * cc2ab_r("KuVxY");

    // NOTICE beta
    ck_va_b("KEU") += 2 * H.block("AV")("VE") * cc1b_n("KUV");
    ck_va_b("KEU") += 2 * V_all_Beta.block("AV")("VE") * cc1b_n("KUV");
    ck_va_b("KEU") += 2 * V_R_Beta.block("AV")("VE")   * cc1b_n("KUV");
    ck_va_b("KEU") += V.block("VAAA")("EVXY") * cc2bb_n("KUVXY");
    ck_va_b("KEU") += 2 * V.block("aVaA")("vExY") * cc2ab_n("KvUxY");
    ck_va_b("KEU") += 2 * H.block("AV")("VE") * cc1b_r("KUV");
    ck_va_b("KEU") += 2 * V_all_Beta.block("AV")("VE") * cc1b_r("KUV");
    ck_va_b("KEU") += 2 * V_R_Beta.block("AV")("VE")   * cc1b_r("KUV");
    ck_va_b("KEU") += V.block("VAAA")("EVXY") * cc2bb_r("KUVXY");
    ck_va_b("KEU") += 2 * V.block("aVaA")("vExY") * cc2ab_r("KvUxY");

    /// contribution from Alpha
    ck_va_a("Keu") += -4 * ci("K") * H.block("av")("ve") * Gamma1_.block("aa")("uv");
    ck_va_a("Keu") += -4 * ci("K") * V_N_Alpha.block("av")("ve") * Gamma1_.block("aa")("uv");
    ck_va_a("Keu") += -4 * ci("K") * V_N_Beta.block("av")("ve") * Gamma1_.block("aa")("uv");
    ck_va_a("Keu") += -2 * ci("K") * V.block("vaaa")("evxy") * Gamma2_.block("aaaa")("uvxy");
    ck_va_a("Keu") += -4 * ci("K") * V.block("vAaA")("eVxY") * Gamma2_.block("aAaA")("uVxY");

    // NOTICE beta
    ck_va_b("KEU") += -4 * ci("K") * H.block("AV")("VE") * Gamma1_.block("AA")("UV");
    ck_va_b("KEU") += -4 * ci("K") * V_all_Beta.block("AV")("VE") * Gamma1_.block("AA")("UV");
    ck_va_b("KEU") += -4 * ci("K") * V_R_Beta.block("AV")("VE") * Gamma1_.block("AA")("UV");
    ck_va_b("KEU") += -2 * ci("K") * V.block("VAAA")("EVXY") * Gamma2_.block("AAAA")("UVXY");
    ck_va_b("KEU") += -4 * ci("K") * V.block("aVaA")("vExY") * Gamma2_.block("aAaA")("vUxY");

    // active-active
    ck_aa_a("Kuv") += V.block("aaaa")("uyvx") * cc1a_n("Kxy");
    ck_aa_a("Kuv") += V.block("aAaA")("uYvX") * cc1b_n("KXY");
    ck_aa_a("Kuv") += V.block("aaaa")("uyvx") * cc1a_r("Kxy");
    ck_aa_a("Kuv") += V.block("aAaA")("uYvX") * cc1b_r("KXY");

    // NOTICE beta
    ck_aa_b("KUV") += V.block("AAAA")("UYVX") * cc1b_n("KXY");
    ck_aa_b("KUV") += V.block("aAaA")("yUxV") * cc1a_n("Kxy");
    ck_aa_b("KUV") += V.block("AAAA")("UYVX") * cc1b_r("KXY");
    ck_aa_b("KUV") += V.block("aAaA")("yUxV") * cc1a_r("Kxy");

    /// contribution from Alpha
    ck_aa_a("Kuv") += -2 * ci("K") * V.block("aaaa")("uyvx") * Gamma1_.block("aa")("xy");
    ck_aa_a("Kuv") += -2 * ci("K") * V.block("aAaA")("uYvX") * Gamma1_.block("AA")("XY");

    // NOTICE beta
    ck_aa_b("KUV") += -2 * ci("K") * V.block("AAAA")("UYVX") * Gamma1_.block("AA")("XY");
    ck_aa_b("KUV") += -2 * ci("K") * V.block("aAaA")("yUxV") * Gamma1_.block("aa")("xy");

    // CI equations' contribution to A
    for (const std::string& row : {"ci"}) {
        int pre1 = preidx[row];

        for (const std::string& col : {"vc","ca","va","aa"}) {
            int idx2 = block_dim[col];
            int pre2 = preidx[col];
            auto temp_ci = ck_vc_a;

            if (col == "ca") temp_ci = ck_ca_a;
            else if (col == "va") temp_ci = ck_va_a;
            else if (col == "aa") temp_ci = ck_aa_a;

            if (col != "aa") {
                (temp_ci).iterate([&](const std::vector<size_t>& i, double& value) {
                    int index = (pre1 + i[0]) + dim * (pre2 + i[1] * idx2 + i[2]);
                    A.at(index) += value;
                });
            }

            else if (col == "aa") {
                (temp_ci).iterate([&](const std::vector<size_t>& i, double& value) {
                    int i1 = i[1] > i[2]? i[1]: i[2],
                        i2 = i[1] > i[2]? i[2]: i[1];
                    if (i1 != i2 ) {
                        int index = (pre1 + i[0]) + dim * (pre2 + i1 * (i1 - 1) / 2 + i2);
                        A.at(index) += value;    
                    }
                });   
            }
        }
    } 

    for (const std::string& row : {"ci"}) {
        int pre1 = preidx[row];

        for (const std::string& col : {"VC","CA","VA","AA"}) {
            int idx2 = block_dim[col];
            int pre2 = preidx[col];
            auto temp_ci = ck_vc_b;

            if (col == "CA") temp_ci = ck_ca_b;
            else if (col == "VA") temp_ci = ck_va_b;
            else if (col == "AA") temp_ci = ck_aa_b;

            if (col != "AA") {
                (temp_ci).iterate([&](const std::vector<size_t>& i, double& value) {
                    int index = (pre1 + i[0]) + dim * (pre2 + i[1] * idx2 + i[2]);
                    A.at(index) += value;
                });
            }

            else if (col == "AA") {
                (temp_ci).iterate([&](const std::vector<size_t>& i, double& value) {
                    int i1 = i[1] > i[2]? i[1]: i[2],
                        i2 = i[1] > i[2]? i[2]: i[1];
                    if (i1 != i2 ) {
                        int index = (pre1 + i[0]) + dim * (pre2 + i1 * (i1 - 1) / 2 + i2);
                        A.at(index) += value;    
                    }
                });   
            }
        }
    } 

    auto ck_ci = ambit::Tensor::build(ambit::CoreTensor, "ci equations ci multiplier part", {ndets, ndets});
    auto I_ci = ambit::Tensor::build(ambit::CoreTensor, "identity", {ndets, ndets});

    I_ci.iterate([&](const std::vector<size_t>& i, double& value) {
        value = (i[0] == i[1]) ? 1.0 : 0.0;
    });
    

    ck_ci("KI") += H.block("cc")("mn") * I.block("cc")("mn") * I_ci("KI");
    ck_ci("KI") += H.block("CC")("MN") * I.block("CC")("MN") * I_ci("KI");
    ck_ci("KI") += cc.cc1a()("KIuv") * H.block("aa")("uv");
    ck_ci("KI") += cc.cc1b()("KIUV") * H.block("AA")("UV");

    ck_ci("KI") += 0.5 * V_N_Alpha["m,m1"] * I["m,m1"] * I_ci("KI");
    ck_ci("KI") += 0.5 * V_all_Beta["M,M1"] * I["M,M1"] * I_ci("KI");
    ck_ci("KI") +=       V_N_Beta["m,m1"] * I["m,m1"] * I_ci("KI");

    ck_ci("KI") += cc.cc1a()("KIuv") * V_N_Alpha.block("aa")("uv");
    ck_ci("KI") += cc.cc1b()("KIUV") * V_all_Beta.block("AA")("UV");
    
    ck_ci("KI") += cc.cc1a()("KIuv") * V_N_Beta.block("aa")("uv");
    ck_ci("KI") += cc.cc1b()("KIUV") * V_R_Beta.block("AA")("UV");

    ck_ci("KI") += 0.25 * cc.cc2aa()("KIuvxy") * V.block("aaaa")("uvxy");
    ck_ci("KI") += 0.25 * cc.cc2bb()("KIUVXY") * V.block("AAAA")("UVXY");
    ck_ci("KI") += 0.50 * cc.cc2ab()("KIuVxY") * V.block("aAaA")("uVxY");
    ck_ci("KI") += 0.50 * cc.cc2ab()("IKuVxY") * V.block("aAaA")("uVxY");

    ck_ci("KI") -= (Eref_ - Enuc_ - Efrzc_) * I_ci("KI");

    // NOTICE: QR decomposition with column pivoting 
    int dim2 = ndets;
    int n2 = dim2, lda2 = dim2;
    int lwork = 3 * n2 + 3;
    std::vector<int> jpvt(n2);
    std::vector<double> tau(dim2);
    std::vector<double> work(lwork);
    std::vector<double> A2(dim2 * dim2);

    

    for (const std::string& row : {"ci"}) {
        for (const std::string& col : {"ci"}) {
            (ck_ci).iterate([&](const std::vector<size_t>& i, double& value) {
                int index = i[1] + dim2 * i[0];
                A2.at(index) = value;
            });
        }
    } 

    C_DGEQP3(n2, n2, &A2[0], lda2, &jpvt[0], &tau[0], &work[0], lwork);

    const int ROW2DEL = jpvt[ndets - 1] - 1;

    for (const std::string& row : {"ci"}) {
        int pre1 = preidx[row];

        for (const std::string& col : {"ci"}) {
            int pre2 = preidx[col];

            (ck_ci).iterate([&](const std::vector<size_t>& i, double& value) {
                if (i[0] != ROW2DEL) {
                    int index = (pre1 + i[0]) + dim * (pre2 + i[1]);
                    A.at(index) += value;
                }
            });

            (ci).iterate([&](const std::vector<size_t>& i, double& value) {
                int index = (pre1 + ROW2DEL) + dim * (pre2 + i[0]);
                A.at(index) += value;
            });

            for(int j = 0; j < pre2; j++) 
                A.at((pre1 + ROW2DEL) + dim * j) = 0.0;
            b.at(pre1 + ROW2DEL) = 0.0;
        }
    } 

    outfile->Printf("Done");
    outfile->Printf("\n    Solving Off-diagonal Entries of Z ............... ");
    int info;

    C_DGESV( n, nrhs, &A[0], lda, &ipiv[0], &b[0], ldb);

    for (const std::string& block : {"vc","ca","va","aa"}) {
        int pre = preidx[block],
            idx = block_dim[block];
        if (block != "aa") {
            (Z.block(block)).iterate([&](const std::vector<size_t>& i, double& value) {
                int index = pre + i[0] * idx + i[1];
                value = b.at(index);
            });
        }
        else {
            (Z.block(block)).iterate([&](const std::vector<size_t>& i, double& value) {
                int i0 = i[0] > i[1]? i[0]: i[1],
                    i1 = i[0] > i[1]? i[1]: i[0];
                if (i0 != i1) {
                    int index = pre + i0 * (i0 - 1) / 2 + i1;
                    value = b.at(index);
                }
            });
        }
    }

    for (const std::string& block : {"ci"}) {
        int pre = preidx[block];
        (x_ci).iterate([&](const std::vector<size_t>& i, double& value) {
            int index = pre + i[0];
            value = b.at(index);
        });
    } 

    Z["me"] = Z["em"];
    Z["wm"] = Z["mw"];
    Z["we"] = Z["ew"];

    // Beta part
    for (const std::string& block : {"VC"}) {
        (Z.block(block)).iterate([&](const std::vector<size_t>& i, double& value) {
            value = Z.block("vc").data()[i[0] * ncore + i[1]];
        });
    }
    for (const std::string& block : {"CA"}) {
        (Z.block(block)).iterate([&](const std::vector<size_t>& i, double& value) {
            value = Z.block("ca").data()[i[0] * na + i[1]];
        });
    } 
    for (const std::string& block : {"VA"}) {
        (Z.block(block)).iterate([&](const std::vector<size_t>& i, double& value) {
            value = Z.block("va").data()[i[0] * na + i[1]];
        });
    } 

    for (const std::string& block : {"AA"}) {
        (Z.block(block)).iterate([&](const std::vector<size_t>& i, double& value) {
            value = Z.block("aa").data()[i[0] * na + i[1]];
        });
    } 

    Z["ME"] = Z["EM"];
    Z["WM"] = Z["MW"];
    Z["WE"] = Z["EW"];

    outfile->Printf("Done");
}

void DSRG_MRPT2::tpdm_backtransform() {
    // Backtransform the TPDM
    // NOTICE: This function also appears in the CASSCF gradient code thus can be refined in the future!!
    
    std::vector<std::shared_ptr<psi::MOSpace>> spaces;
    spaces.push_back(psi::MOSpace::all);
    std::shared_ptr<TPDMBackTransform> transform =
        std::shared_ptr<TPDMBackTransform>(new TPDMBackTransform(
            ints_->wfn(), spaces,
            IntegralTransform::TransformationType::Unrestricted, // Transformation type
            IntegralTransform::OutputType::DPDOnly,              // Output buffer
            IntegralTransform::MOOrdering::QTOrder,              // MO ordering
            IntegralTransform::FrozenOrbitals::None));           // Frozen orbitals?
    transform->backtransform_density();
    transform.reset();

    outfile->Printf("\n    TPDM Backtransformation ......................... Done");
}

void DSRG_MRPT2::write_lagrangian() {
    // NOTICE: write the Lagrangian
    outfile->Printf("\n    Writing Lagrangian .............................. ");

    SharedMatrix L(new Matrix("Lagrangian", nirrep, irrep_vec, irrep_vec));

    for (const std::string& block : {"cc", "CC", "aa", "AA", "ca", "ac", "CA", "AC",
                    "vv", "VV", "av", "cv", "va", "vc", "AV", "CV", "VA", "VC"}) {
        std::vector<std::vector<std::pair<unsigned long, unsigned long>,
                                std::allocator<std::pair<unsigned long, unsigned long>>>>
            spin_pair;
        for (size_t idx : {0, 1}) {
            auto spin = std::tolower(block.at(idx));
            if (spin == 'c') {
                spin_pair.push_back(core_mos_relative);
            } else if (spin == 'a') {
                spin_pair.push_back(actv_mos_relative);
            }
            else if (spin == 'v') {
                spin_pair.push_back(virt_mos_relative);
            }
        }

        (W.block(block)).iterate([&](const std::vector<size_t>& i, double& value) {
            if (spin_pair[0][i[0]].first == spin_pair[1][i[1]].first) {
                L->add(spin_pair[0][i[0]].first, spin_pair[0][i[0]].second,
                       spin_pair[1][i[1]].second, value);
            }
        });

    }

    L->back_transform(ints_->Ca());
    ints_->wfn()->set_lagrangian(SharedMatrix(new Matrix("Lagrangian", nirrep, irrep_vec, irrep_vec)));
    ints_->wfn()->Lagrangian()->copy(L);

    outfile->Printf("Done");
}

/**
 * Write spin_dependent one-RDMs coefficients.
 *
 * We force "Da == Db". This function needs be changed if such constraint is revoked.
 */
void DSRG_MRPT2::write_1rdm_spin_dependent() {
    // NOTICE: write spin_dependent one-RDMs coefficients. 
    outfile->Printf("\n    Writing 1RDM Coefficients ....................... ");
    SharedMatrix D1(new Matrix("1rdm coefficients contribution", nirrep, irrep_vec, irrep_vec));

    (Z.block("vc")).iterate([&](const std::vector<size_t>& i, double& value) {
        if (virt_mos_relative[i[0]].first == core_mos_relative[i[1]].first) {
            D1->set(virt_mos_relative[i[0]].first, virt_mos_relative[i[0]].second,
                core_mos_relative[i[1]].second, value);
            D1->set(virt_mos_relative[i[0]].first, core_mos_relative[i[1]].second,
                virt_mos_relative[i[0]].second, value);
        }
    });

    BlockedTensor temp = BTF_->build(CoreTensor, "temporal tensor", {"ca"});
    temp["nu"] = Z["un"];
    temp["nv"] -= Z["un"] * Gamma1_["uv"];

    (temp.block("ca")).iterate([&](const std::vector<size_t>& i, double& value) {
        if (core_mos_relative[i[0]].first == actv_mos_relative[i[1]].first) {
            D1->set(core_mos_relative[i[0]].first, core_mos_relative[i[0]].second,
                actv_mos_relative[i[1]].second, value);
            D1->set(core_mos_relative[i[0]].first, actv_mos_relative[i[1]].second,
                core_mos_relative[i[0]].second, value);
        }
    });

    temp = BTF_->build(CoreTensor, "temporal tensor", {"va"});
    temp["ev"] = Z["eu"] * Gamma1_["uv"];

    (temp.block("va")).iterate([&](const std::vector<size_t>& i, double& value) {
        if (virt_mos_relative[i[0]].first == actv_mos_relative[i[1]].first) {
            D1->set(virt_mos_relative[i[0]].first, virt_mos_relative[i[0]].second,
                actv_mos_relative[i[1]].second, value);
            D1->set(virt_mos_relative[i[0]].first, actv_mos_relative[i[1]].second,
                virt_mos_relative[i[0]].second, value);
        }
    });

    (Z.block("cc")).iterate([&](const std::vector<size_t>& i, double& value) {
        if (core_mos_relative[i[0]].first == core_mos_relative[i[1]].first) {
            D1->set(core_mos_relative[i[0]].first, core_mos_relative[i[0]].second,
                core_mos_relative[i[1]].second, value);
        }
    });

    (Z.block("aa")).iterate([&](const std::vector<size_t>& i, double& value) {
        if (actv_mos_relative[i[0]].first == actv_mos_relative[i[1]].first) {
            D1->set(actv_mos_relative[i[0]].first, actv_mos_relative[i[0]].second,
                actv_mos_relative[i[1]].second, value);
        }
    });

    (Z.block("vv")).iterate([&](const std::vector<size_t>& i, double& value) {
        if (virt_mos_relative[i[0]].first == virt_mos_relative[i[1]].first) {
            D1->set(virt_mos_relative[i[0]].first, virt_mos_relative[i[0]].second,
                virt_mos_relative[i[1]].second, value);
        }
    });

    // <[F, T2]>
    (Sigma3.block("ca")).iterate([&](const std::vector<size_t>& i, double& value) {
        if (core_mos_relative[i[0]].first == actv_mos_relative[i[1]].first) {
            D1->add(core_mos_relative[i[0]].first, core_mos_relative[i[0]].second,
                actv_mos_relative[i[1]].second, 0.5 * value);
            D1->add(core_mos_relative[i[0]].first, actv_mos_relative[i[1]].second,
                core_mos_relative[i[0]].second, 0.5 * value);
        }
    });

    (Sigma3.block("cv")).iterate([&](const std::vector<size_t>& i, double& value) {
        if (core_mos_relative[i[0]].first == virt_mos_relative[i[1]].first) {
            D1->add(core_mos_relative[i[0]].first, core_mos_relative[i[0]].second,
                virt_mos_relative[i[1]].second, 0.5 * value);
            D1->add(core_mos_relative[i[0]].first, virt_mos_relative[i[1]].second,
                core_mos_relative[i[0]].second, 0.5 * value);
        }
    });

    (Sigma3.block("av")).iterate([&](const std::vector<size_t>& i, double& value) {
        if (actv_mos_relative[i[0]].first == virt_mos_relative[i[1]].first) {
            D1->add(actv_mos_relative[i[0]].first, actv_mos_relative[i[0]].second,
                virt_mos_relative[i[1]].second, 0.5 * value);
            D1->add(actv_mos_relative[i[0]].first, virt_mos_relative[i[1]].second,
                actv_mos_relative[i[0]].second, 0.5 * value);
        }
    });

    // <[V, T1]>
    (Xi3.block("ca")).iterate([&](const std::vector<size_t>& i, double& value) {
        if (core_mos_relative[i[0]].first == actv_mos_relative[i[1]].first) {
            D1->add(core_mos_relative[i[0]].first, core_mos_relative[i[0]].second,
                actv_mos_relative[i[1]].second, 0.5 * value);
            D1->add(core_mos_relative[i[0]].first, actv_mos_relative[i[1]].second,
                core_mos_relative[i[0]].second, 0.5 * value);
        }
    });

    (Xi3.block("cv")).iterate([&](const std::vector<size_t>& i, double& value) {
        if (core_mos_relative[i[0]].first == virt_mos_relative[i[1]].first) {
            D1->add(core_mos_relative[i[0]].first, core_mos_relative[i[0]].second,
                virt_mos_relative[i[1]].second, 0.5 * value);
            D1->add(core_mos_relative[i[0]].first, virt_mos_relative[i[1]].second,
                core_mos_relative[i[0]].second, 0.5 * value);
        }
    });

    (Xi3.block("av")).iterate([&](const std::vector<size_t>& i, double& value) {
        if (actv_mos_relative[i[0]].first == virt_mos_relative[i[1]].first) {
            D1->add(actv_mos_relative[i[0]].first, actv_mos_relative[i[0]].second,
                virt_mos_relative[i[1]].second, 0.5 * value);
            D1->add(actv_mos_relative[i[0]].first, virt_mos_relative[i[1]].second,
                actv_mos_relative[i[0]].second, 0.5 * value);
        }
    });

    // CASSCF reference
    for (size_t i = 0, size_c = core_mos_relative.size(); i < size_c; ++i) {
        D1->add(core_mos_relative[i].first, core_mos_relative[i].second,
                core_mos_relative[i].second, 1.0);
    }


    (Gamma1_.block("aa")).iterate([&](const std::vector<size_t>& i, double& value) {
        if (actv_mos_relative[i[0]].first == actv_mos_relative[i[1]].first) {
            D1->add(actv_mos_relative[i[0]].first, actv_mos_relative[i[0]].second,
                    actv_mos_relative[i[1]].second, value);
        }
    });

    // CI contribution
    auto tp = ambit::Tensor::build(ambit::CoreTensor, "temporal tensor", {na, na});

    tp("uv") = 0.5  * x_ci("I") * cc1a_n("Iuv");
    tp("uv") += 0.5 * x_ci("J") * cc1a_r("Juv");

    (tp).iterate([&](const std::vector<size_t>& i, double& value) {
        if (actv_mos_relative[i[0]].first == actv_mos_relative[i[1]].first) {
            D1->add(actv_mos_relative[i[0]].first, actv_mos_relative[i[0]].second,
                    actv_mos_relative[i[1]].second, value);
        }
    });


    D1->back_transform(ints_->Ca());
    ints_->wfn()->Da()->copy(D1);
    ints_->wfn()->Db()->copy(D1);

    outfile->Printf("Done");
}

/**
 * Write spin_dependent two-RDMs coefficients using IWL.
 *
 * Coefficients in d2aa and d2bb need be multiplied with additional 1/2!
 * Specifically:
 * If you have v_aa as coefficients before 2-RDMs_alpha_alpha, v_bb before
 * 2-RDMs_beta_beta and v_bb before 2-RDMs_alpha_beta, you need to write
 * 0.5 * v_aa, 0.5 * v_bb and v_ab into the IWL file instead of using
 * the original coefficients v_aa, v_bb and v_ab.
 */
void DSRG_MRPT2::write_2rdm_spin_dependent() {
    // NOTICE: write spin_dependent two-RDMs coefficients using IWL
    outfile->Printf("\n    Writing 2RDM Coefficients ....................... ");
    auto psio_ = _default_psio_lib_;
    IWL d2aa(psio_.get(), PSIF_MO_AA_TPDM, 1.0e-14, 0, 0);
    IWL d2ab(psio_.get(), PSIF_MO_AB_TPDM, 1.0e-14, 0, 0);
    IWL d2bb(psio_.get(), PSIF_MO_BB_TPDM, 1.0e-14, 0, 0);

    BlockedTensor temp = BTF_->build(CoreTensor, "temporal tensor", {"vc", "VC"});
    // <[F, T2]>
    temp["em"] += 0.5 * Sigma3["me"];
    temp["EM"] += 0.5 * Sigma3["ME"];
    // <[V, T1]>
    temp["em"] += 0.5 * Xi3["me"];
    temp["EM"] += 0.5 * Xi3["ME"];

    for (size_t i = 0, size_c = core_all.size(); i < size_c; ++i) {
        auto m = core_all[i];
        for (size_t a = 0, size_v = virt_all.size(); a < size_v; ++a) {
            auto e = virt_all[a];
            auto idx = a * ncore + i;
            auto z_a = Z.block("vc").data()[idx] + temp.block("vc").data()[idx];
            auto z_b = Z.block("VC").data()[idx] + temp.block("VC").data()[idx];
            for (size_t j = 0; j < size_c; ++j) {
                auto m1 = core_all[j];
                
                d2aa.write_value(m, e, m1, m1, z_a, 0, "outfile", 0);
                d2bb.write_value(m, e, m1, m1, z_b, 0, "outfile", 0);
                d2aa.write_value(m, m1, m1, e, -z_a, 0, "outfile", 0);
                d2bb.write_value(m, m1, m1, e, -z_b, 0, "outfile", 0); 
                d2ab.write_value(m, e, m1, m1, 2.0 * (z_a + z_b), 0, "outfile", 0);
            }
        }
    }

    temp = BTF_->build(CoreTensor, "temporal tensor", {"ac", "AC"});
    temp["un"] = Z["un"];
    temp["un"] -= Z["vn"] * Gamma1_["uv"];
    temp["UN"] = Z["UN"];
    temp["UN"] -= Z["VN"] * Gamma1_["UV"];
    // <[F, T2]>
    temp["un"] += 0.5 * Sigma3["nu"];
    temp["UN"] += 0.5 * Sigma3["NU"];
    // <[V, T1]>
    temp["un"] += 0.5 * Xi3["nu"];
    temp["UN"] += 0.5 * Xi3["NU"];

    for (size_t i = 0, size_c = core_all.size(); i < size_c; ++i) {
        auto n = core_all[i];
        for (size_t a = 0, size_a = actv_all.size(); a < size_a; ++a) {
            auto u = actv_all[a];
            auto idx = a * ncore + i;
            auto z_a = temp.block("ac").data()[idx];
            auto z_b = temp.block("AC").data()[idx];
            for (size_t j = 0; j < size_c; ++j) {
                auto m1 = core_all[j];              
                if (n != m1) {
                    d2aa.write_value(u, n, m1, m1, z_a, 0, "outfile", 0);
                    d2bb.write_value(u, n, m1, m1, z_b, 0, "outfile", 0);
                    d2aa.write_value(u, m1, m1, n, -z_a, 0, "outfile", 0);
                    d2bb.write_value(u, m1, m1, n, -z_b, 0, "outfile", 0);
                }
                d2ab.write_value(u, n, m1, m1, 2.0 * (z_a + z_b), 0, "outfile", 0);
            }
        }
    }

    temp = BTF_->build(CoreTensor, "temporal tensor", {"va", "VA"});
    temp["ev"] = Z["eu"] * Gamma1_["uv"];
    temp["EV"] = Z["EU"] * Gamma1_["UV"];
    // <[F, T2]>
    temp["ev"] += 0.5 * Sigma3["ve"];
    temp["EV"] += 0.5 * Sigma3["VE"];
    // <[V, T1]>
    temp["ev"] += 0.5 * Xi3["ve"];
    temp["EV"] += 0.5 * Xi3["VE"];

    for (size_t i = 0, size_a = actv_all.size(); i < size_a; ++i) {
        auto v = actv_all[i];
        for (size_t a = 0, size_v = virt_all.size(); a < size_v; ++a) {
            auto e = virt_all[a];
            auto idx = a * na + i;
            auto z_a = temp.block("va").data()[idx];
            auto z_b = temp.block("VA").data()[idx];
            for (size_t j = 0, size_c = core_all.size(); j < size_c; ++j) {
                auto m1 = core_all[j];
                
                d2aa.write_value(v, e, m1, m1, z_a, 0, "outfile", 0);
                d2bb.write_value(v, e, m1, m1, z_b, 0, "outfile", 0);
                d2aa.write_value(v, m1, m1, e, -z_a, 0, "outfile", 0);
                d2bb.write_value(v, m1, m1, e, -z_b, 0, "outfile", 0);  
                d2ab.write_value(v, e, m1, m1, 2.0 * (z_a + z_b), 0, "outfile", 0);
            }
        }
    }

    for (size_t i = 0, size_c = core_all.size(); i < size_c; ++i) {
        auto n = core_all[i];
        for (size_t k = 0; k < size_c; ++k) {
            auto m = core_all[k];
            auto idx = k * ncore + i;
            auto z_a = Z.block("cc").data()[idx];
            auto z_b = Z.block("CC").data()[idx];
            for (size_t j = 0; j < size_c; ++j) {
                auto m1 = core_all[j];
                double a1 = 0.5 * z_a, v2 = 0.5 * z_b, v3 = z_a + z_b;

                if (m == n) {
                    a1 += 0.25;
                    v2 += 0.25;
                    v3 += 1.00;
                }

                if (m != m1) {
                    d2aa.write_value(n, m, m1, m1,  a1, 0, "outfile", 0);
                    d2bb.write_value(n, m, m1, m1,  v2, 0, "outfile", 0);
                    d2aa.write_value(n, m1, m1, m, -a1, 0, "outfile", 0);
                    d2bb.write_value(n, m1, m1, m, -v2, 0, "outfile", 0);
                }  
                d2ab.write_value(n, m, m1, m1, v3, 0, "outfile", 0);
            }
        }
    }

    auto ci_g1_a = ambit::Tensor::build(ambit::CoreTensor, "effective alpha gamma tensor", {na, na});
    auto ci_g1_b = ambit::Tensor::build(ambit::CoreTensor, "effective beta gamma tensor", {na, na});

    ci_g1_a("uv") += 0.5 * x_ci("I") * cc1a_n("Iuv");
    ci_g1_a("uv") += 0.5 * x_ci("J") * cc1a_r("Juv");
    ci_g1_b("UV") += 0.5 * x_ci("I") * cc1b_n("IUV");
    ci_g1_b("UV") += 0.5 * x_ci("J") * cc1b_r("JUV");

    for (size_t i = 0, size_a = actv_all.size(); i < size_a; ++i) {
        auto v = actv_all[i];
        for (size_t k = 0; k < size_a; ++k) {
            auto u = actv_all[k];
            auto idx = k * na + i;
            auto z_a = Z.block("aa").data()[idx];
            auto z_b = Z.block("AA").data()[idx];
            auto gamma_a = Gamma1_.block("aa").data()[idx];
            auto gamma_b = Gamma1_.block("AA").data()[idx];
            auto ci_gamma_a = ci_g1_a.data()[idx];
            auto ci_gamma_b = ci_g1_b.data()[idx];
            auto a1 = z_a + gamma_a + ci_gamma_a;
            auto v2 = z_b + gamma_b + ci_gamma_b;

            for (size_t j = 0, size_c = core_all.size(); j < size_c; ++j) {
                auto m1 = core_all[j];
                
                d2aa.write_value(v, u, m1, m1, 0.5 * a1, 0, "outfile", 0);
                d2bb.write_value(v, u, m1, m1, 0.5 * v2, 0, "outfile", 0);
                d2aa.write_value(v, m1, m1, u, -0.5 * a1, 0, "outfile", 0);
                d2bb.write_value(v, m1, m1, u, -0.5 * v2, 0, "outfile", 0);   
                d2ab.write_value(v, u, m1, m1, (a1 + v2), 0, "outfile", 0);
            }
        }
    }

    for (size_t i = 0, size_v = virt_all.size(); i < size_v; ++i) {
        auto f = virt_all[i];
        for (size_t k = 0; k < size_v; ++k) {
            auto e = virt_all[k];
            auto idx = k * nvirt + i;
            auto z_a = Z.block("vv").data()[idx];
            auto z_b = Z.block("VV").data()[idx];
            for (size_t j = 0, size_c = core_all.size(); j < size_c; ++j) {
                auto m1 = core_all[j];
                
                d2aa.write_value(f, e, m1, m1, 0.5 * z_a, 0, "outfile", 0);
                d2bb.write_value(f, e, m1, m1, 0.5 * z_b, 0, "outfile", 0);
                d2aa.write_value(f, m1, m1, e, -0.5 * z_a, 0, "outfile", 0);
                d2bb.write_value(f, m1, m1, e, -0.5 * z_b, 0, "outfile", 0);    
                d2ab.write_value(f, e, m1, m1, (z_a + z_b), 0, "outfile", 0);
            }
        }
    }

    for (size_t i = 0, size_c = core_all.size(); i < size_c; ++i) {
        auto n = core_all[i];
        for (size_t j = 0; j < size_c; ++j) {
            auto m = core_all[j];
            auto idx = j * ncore + i;
            auto z_a = Z.block("cc").data()[idx];
            auto z_b = Z.block("CC").data()[idx];
            for (size_t k = 0, size_a = actv_all.size(); k < size_a; ++k) {
                auto a1 = actv_all[k];
                for (size_t l = 0; l < size_a; ++l) {
                    auto u1 = actv_all[l];
                    auto idx1 = l * na + k;
                    auto g_a = Gamma1_.block("aa").data()[idx1];
                    auto g_b = Gamma1_.block("AA").data()[idx1];
                
                    d2aa.write_value(n, m, a1, u1, 0.5 * z_a * g_a, 0, "outfile", 0);
                    d2bb.write_value(n, m, a1, u1, 0.5 * z_b * g_b, 0, "outfile", 0);
                    d2aa.write_value(n, u1, a1, m, -0.5 * z_a * g_a, 0, "outfile", 0);
                    d2bb.write_value(n, u1, a1, m, -0.5 * z_b * g_b, 0, "outfile", 0);
                    d2ab.write_value(n, m, a1, u1, (z_a * g_b + z_b * g_a), 0, "outfile", 0);
                }
            }
        }
    }

    for (size_t i = 0, size_v = virt_all.size(); i < size_v; ++i) {
        auto f = virt_all[i];
        for (size_t j = 0; j < size_v; ++j) {
            auto e = virt_all[j];
            auto idx = j * nvirt + i;
            auto z_a = Z.block("vv").data()[idx];
            auto z_b = Z.block("VV").data()[idx];
            for (size_t k = 0, size_a = actv_all.size(); k < size_a; ++k) {
                auto a1 = actv_all[k];
                for (size_t l = 0; l < size_a; ++l) {
                    auto u1 = actv_all[l];
                    auto idx1 = l * na + k;
                    auto g_a = Gamma1_.block("aa").data()[idx1];
                    auto g_b = Gamma1_.block("AA").data()[idx1];
                
                    d2aa.write_value(f, e, a1, u1, 0.5 * z_a * g_a, 0, "outfile", 0);
                    d2bb.write_value(f, e, a1, u1, 0.5 * z_b * g_b, 0, "outfile", 0);
                    d2aa.write_value(f, u1, a1, e, -0.5 * z_a * g_a, 0, "outfile", 0);
                    d2bb.write_value(f, u1, a1, e, -0.5 * z_b * g_b, 0, "outfile", 0);
                    d2ab.write_value(f, e, a1, u1, (z_a * g_b + z_b * g_a), 0, "outfile", 0);
                }
            }
        }
    }

    // terms with overlap
    temp = BTF_->build(CoreTensor, "temporal tensor", {"pphh", "PPHH", "pPhH"});
    BlockedTensor temp2 = BTF_->build(CoreTensor, "temporal tensor 2", {"phph","phPH"});

    if (CORRELATION_TERM) {
        temp["abij"] += Tau1["ijab"];
        temp["ABIJ"] += Tau1["IJAB"];
        temp["aBiJ"] += Tau1["iJaB"];

        temp["cdkl"] += Kappa["klcd"] * Eeps2_p["klcd"];
        temp["CDKL"] += Kappa["KLCD"] * Eeps2_p["KLCD"];
        temp["cDkL"] += Kappa["kLcD"] * Eeps2_p["kLcD"];
    }

    temp["xynv"] -= Z["un"] * Gamma2_["uvxy"]; 
    temp["XYNV"] -= Z["UN"] * Gamma2_["UVXY"]; 
    temp["xYnV"] -= Z["un"] * Gamma2_["uVxY"]; 

    temp["evxy"] += Z["eu"] * Gamma2_["uvxy"];
    temp["EVXY"] += Z["EU"] * Gamma2_["UVXY"];
    temp["eVxY"] += Z["eu"] * Gamma2_["uVxY"];

    // CASSCF reference
    temp["xyuv"] += 0.25 * Gamma2_["uvxy"];
    temp["XYUV"] += 0.25 * Gamma2_["UVXY"];
    temp["xYuV"] += 0.25 * Gamma2_["uVxY"];

    // CI contribution
    temp.block("aaaa")("xyuv") += 0.5 * 0.25 * cc2aa_n("Iuvxy") * x_ci("I");
    temp.block("AAAA")("XYUV") += 0.5 * 0.25 * cc2bb_n("IUVXY") * x_ci("I");
    temp.block("aAaA")("xYuV") += 0.5 * 0.25 * cc2ab_n("IuVxY") * x_ci("I");

    temp.block("aaaa")("xyuv") += 0.5 * 0.25 * cc2aa_r("Juvxy") * x_ci("J");
    temp.block("AAAA")("XYUV") += 0.5 * 0.25 * cc2bb_r("JUVXY") * x_ci("J");
    temp.block("aAaA")("xYuV") += 0.5 * 0.25 * cc2ab_r("JuVxY") * x_ci("J");

    // all-alpha and all-beta
    temp2["ckdl"] += temp["cdkl"];
    temp2["cldk"] -= temp["cdkl"];
    // alpha-beta
    temp2["ckDL"] += 2.0 * temp["cDkL"];
    temp2["clDK"] += 2.0 * temp["cDlK"];
    temp.zero();

    temp["eumv"] += 2.0 * Z["em"] * Gamma1_["uv"];
    temp["EUMV"] += 2.0 * Z["EM"] * Gamma1_["UV"];
    temp["eUmV"] += 2.0 * Z["em"] * Gamma1_["UV"];

    temp["u,a1,n,u1"] += 2.0 * Z["un"] * Gamma1_["u1,a1"]; 
    temp["U,A1,N,U1"] += 2.0 * Z["UN"] * Gamma1_["U1,A1"];
    temp["u,A1,n,U1"] += 2.0 * Z["un"] * Gamma1_["U1,A1"]; 

    temp["v,a1,u,u1"] += Z["uv"] * Gamma1_["u1,a1"];
    temp["V,A1,U,U1"] += Z["UV"] * Gamma1_["U1,A1"];
    temp["v,A1,u,U1"] += Z["uv"] * Gamma1_["U1,A1"];

    // <[F, T2]>
    if (X5_TERM||X7_TERM) {
        temp["aviu"] += Sigma3["ia"] * Gamma1_["uv"];
        temp["AVIU"] += Sigma3["IA"] * Gamma1_["UV"];
        temp["aViU"] += Sigma3["ia"] * Gamma1_["UV"];
    }
    // <[V, T1]>
    if (X6_TERM||X7_TERM) {
        temp["aviu"] += Xi3["ia"] * Gamma1_["uv"];
        temp["AVIU"] += Xi3["IA"] * Gamma1_["UV"];
        temp["aViU"] += Xi3["ia"] * Gamma1_["UV"];
    }

    // all-alpha and all-beta
    temp2["ckdl"] += temp["cdkl"];
    temp2["cldk"] -= temp["cdkl"];
    // alpha-beta
    temp2["ckDL"] += 2.0 * temp["cDkL"];

    temp2.iterate([&](const std::vector<size_t>& i, const std::vector<SpinType>& spin, double& value) {
        if (std::fabs(value) > 1e-12) {
            if (spin[2] == AlphaSpin) {
                d2aa.write_value(i[0], i[1], i[2], i[3], 0.5 * value, 0, "outfile", 0);          
                d2bb.write_value(i[0], i[1], i[2], i[3], 0.5 * value, 0, "outfile", 0);          
            }
            else {
                d2ab.write_value(i[0], i[1], i[2], i[3], value, 0, "outfile", 0); 
            }
        }
    }); 

    d2aa.flush(1);
    d2bb.flush(1);
    d2ab.flush(1);

    d2aa.set_keep_flag(1);
    d2bb.set_keep_flag(1);
    d2ab.set_keep_flag(1);

    d2aa.close();
    d2bb.close();
    d2ab.close();

    outfile->Printf("Done");
}

} 