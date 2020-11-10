/**
 * DSRG-MRPT2 gradient code by Shuhe Wang
 *
 * The computation procedure is listed as belows:
 * (1), Set MOs spaces;
 * (2), Set Tensors (F, H, V etc.);
 * (3), Compute and write the Lagrangian;
 * (4), Write 1RDMs and 2RDMs coefficients;
 * (5), Back-transform the TPDM.
 */
#include <algorithm>
#include <map>
#include <vector>
#include <math.h>
#include <numeric>
#include <ctype.h>
#include <string>

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
    nmo_ = mo_space_info_->size("CORRELATED");
    s = dsrg_source_->get_s();
    core_mos_ = mo_space_info_->get_corr_abs_mo("RESTRICTED_DOCC");
    actv_mos_ = mo_space_info_->get_corr_abs_mo("ACTIVE");
    virt_mos_ = mo_space_info_->get_corr_abs_mo("RESTRICTED_UOCC");
    core_all_ = mo_space_info_->get_absolute_mo("RESTRICTED_DOCC");
    actv_all_ = mo_space_info_->get_absolute_mo("ACTIVE");
    virt_all_ = mo_space_info_->get_absolute_mo("RESTRICTED_UOCC");
    core_mos_relative = mo_space_info_->get_relative_mo("RESTRICTED_DOCC");
    actv_mos_relative = mo_space_info_->get_relative_mo("ACTIVE");
    virt_mos_relative = mo_space_info_->get_relative_mo("RESTRICTED_UOCC");
    irrep_vec = mo_space_info_->get_dimension("ALL");
    na_ = mo_space_info_->size("ACTIVE");
    ncore_ = mo_space_info_->size("RESTRICTED_DOCC");
    nvirt_ = mo_space_info_->size("RESTRICTED_UOCC");
    nirrep_ = mo_space_info_->nirrep();
    ndets = ci_vectors_[0].dims()[0];
    Alpha = 0.0;
    outfile->Printf("Done");
}

void DSRG_MRPT2::set_ci_ints() {
    cc = coupling_coefficients_;
    ci = ci_vectors_[0];

    cc1a_n = ambit::Tensor::build(ambit::CoreTensor, "cc1a_n", {ndets, na_, na_});
    cc1a_r = ambit::Tensor::build(ambit::CoreTensor, "cc1a_r", {ndets, na_, na_});
    cc1b_n = ambit::Tensor::build(ambit::CoreTensor, "cc1b_n", {ndets, na_, na_});
    cc1b_r = ambit::Tensor::build(ambit::CoreTensor, "cc1b_r", {ndets, na_, na_});

    cc2aa_n = ambit::Tensor::build(ambit::CoreTensor, "cc2aa_n", {ndets, na_, na_, na_, na_});
    cc2aa_r = ambit::Tensor::build(ambit::CoreTensor, "cc2aa_r", {ndets, na_, na_, na_, na_});
    cc2bb_n = ambit::Tensor::build(ambit::CoreTensor, "cc2bb_n", {ndets, na_, na_, na_, na_});
    cc2bb_r = ambit::Tensor::build(ambit::CoreTensor, "cc2bb_r", {ndets, na_, na_, na_, na_});
    cc2ab_n = ambit::Tensor::build(ambit::CoreTensor, "cc2ab_n", {ndets, na_, na_, na_, na_});
    cc2ab_r = ambit::Tensor::build(ambit::CoreTensor, "cc2ab_r", {ndets, na_, na_, na_, na_});

    cc3aaa_n = ambit::Tensor::build(ambit::CoreTensor, "cc3aaa_n", {ndets, na_, na_, na_, na_, na_, na_});
    cc3aaa_r = ambit::Tensor::build(ambit::CoreTensor, "cc3aaa_r", {ndets, na_, na_, na_, na_, na_, na_});
    cc3bbb_n = ambit::Tensor::build(ambit::CoreTensor, "cc3bbb_n", {ndets, na_, na_, na_, na_, na_, na_});
    cc3bbb_r = ambit::Tensor::build(ambit::CoreTensor, "cc3bbb_r", {ndets, na_, na_, na_, na_, na_, na_});
    cc3aab_n = ambit::Tensor::build(ambit::CoreTensor, "cc3aab_n", {ndets, na_, na_, na_, na_, na_, na_});
    cc3aab_r = ambit::Tensor::build(ambit::CoreTensor, "cc3aab_r", {ndets, na_, na_, na_, na_, na_, na_});
    cc3abb_n = ambit::Tensor::build(ambit::CoreTensor, "cc3abb_n", {ndets, na_, na_, na_, na_, na_, na_});
    cc3abb_r = ambit::Tensor::build(ambit::CoreTensor, "cc3abb_r", {ndets, na_, na_, na_, na_, na_, na_});

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

    dlamb_aa = ambit::Tensor::build(ambit::CoreTensor, "derivatives of Lambda2_ w.r.t. C_K alpha-alpha", {ndets, na_, na_, na_, na_});
    dlamb_bb = ambit::Tensor::build(ambit::CoreTensor, "derivatives of Lambda2_ w.r.t. C_K beta-beta", {ndets, na_, na_, na_, na_});
    dlamb_ab = ambit::Tensor::build(ambit::CoreTensor, "derivatives of Lambda2_ w.r.t. C_K alpha-beta", {ndets, na_, na_, na_, na_});

    // alpha-alpha
    dlamb_aa("Kxyuv") += cc2aa_n("Kxyuv");
    dlamb_aa("Kxyuv") += cc2aa_r("Kxyuv");
    dlamb_aa("Kxyuv") -= cc1a_n("Kxu") * Gamma1.block("aa")("yv");
    dlamb_aa("Kxyuv") -= cc1a_n("Kux") * Gamma1.block("aa")("yv");
    dlamb_aa("Kxyuv") -= cc1a_n("Kyv") * Gamma1.block("aa")("xu");
    dlamb_aa("Kxyuv") -= cc1a_n("Kvy") * Gamma1.block("aa")("xu");
    dlamb_aa("Kxyuv") += cc1a_n("Kxv") * Gamma1.block("aa")("yu");
    dlamb_aa("Kxyuv") += cc1a_n("Kvx") * Gamma1.block("aa")("yu");
    dlamb_aa("Kxyuv") += cc1a_n("Kyu") * Gamma1.block("aa")("xv");
    dlamb_aa("Kxyuv") += cc1a_n("Kuy") * Gamma1.block("aa")("xv");

    // beta-beta
    dlamb_bb("KXYUV") += cc2bb_n("KXYUV");
    dlamb_bb("KXYUV") += cc2bb_r("KXYUV");
    dlamb_bb("KXYUV") -= cc1b_n("KXU") * Gamma1.block("AA")("YV");
    dlamb_bb("KXYUV") -= cc1b_n("KUX") * Gamma1.block("AA")("YV");
    dlamb_bb("KXYUV") -= cc1b_n("KYV") * Gamma1.block("AA")("XU");
    dlamb_bb("KXYUV") -= cc1b_n("KVY") * Gamma1.block("AA")("XU");
    dlamb_bb("KXYUV") += cc1b_n("KXV") * Gamma1.block("AA")("YU");
    dlamb_bb("KXYUV") += cc1b_n("KVX") * Gamma1.block("AA")("YU");
    dlamb_bb("KXYUV") += cc1b_n("KYU") * Gamma1.block("AA")("XV");
    dlamb_bb("KXYUV") += cc1b_n("KUY") * Gamma1.block("AA")("XV");

    // alpha-beta
    dlamb_ab("KxYuV") += cc2ab_n("KxYuV");
    dlamb_ab("KxYuV") += cc2ab_r("KxYuV");
    dlamb_ab("KxYuV") -= cc1a_n("Kxu") * Gamma1.block("AA")("YV");
    dlamb_ab("KxYuV") -= cc1a_n("Kux") * Gamma1.block("AA")("YV");
    dlamb_ab("KxYuV") -= cc1b_n("KYV") * Gamma1.block("aa")("xu");
    dlamb_ab("KxYuV") -= cc1b_n("KVY") * Gamma1.block("aa")("xu");

    dlamb3_aaa = ambit::Tensor::build(ambit::CoreTensor, "derivatives of Lambda3_ w.r.t. C_K alpha-alpha-alpha", {ndets, na_, na_, na_, na_, na_, na_});
    dlamb3_bbb = ambit::Tensor::build(ambit::CoreTensor, "derivatives of Lambda3_ w.r.t. C_K beta-beta-beta", {ndets, na_, na_, na_, na_, na_, na_});
    dlamb3_aab = ambit::Tensor::build(ambit::CoreTensor, "derivatives of Lambda3_ w.r.t. C_K alpha-alpha-beta", {ndets, na_, na_, na_, na_, na_, na_});
    dlamb3_abb = ambit::Tensor::build(ambit::CoreTensor, "derivatives of Lambda3_ w.r.t. C_K alpha-beta-beta", {ndets, na_, na_, na_, na_, na_, na_});

    // alpha-alpha-alpha
    dlamb3_aaa("Kxyzuvw") += cc3aaa_n("Kxyzuvw");
    dlamb3_aaa("Kxyzuvw") += cc3aaa_r("Kxyzuvw");
    dlamb3_aaa("Kxyzuvw") -= 2.0 * cc1a_n("Kuz") * Gamma2.block("aaaa")("xyvw");
    dlamb3_aaa("Kxyzuvw") -= 2.0 * cc1a_r("Kuz") * Gamma2.block("aaaa")("xyvw");
    dlamb3_aaa("Kxyzuvw") -= 2.0 * cc2aa_n("Kxyvw") * Gamma1.block("aa")("uz");
    dlamb3_aaa("Kxyzuvw") -= 2.0 * cc2aa_r("Kxyvw") * Gamma1.block("aa")("uz");
    dlamb3_aaa("Kxyzuvw") -= cc1a_n("Kwz") * Gamma2.block("aaaa")("xyuv");
    dlamb3_aaa("Kxyzuvw") -= cc1a_r("Kwz") * Gamma2.block("aaaa")("xyuv");
    dlamb3_aaa("Kxyzuvw") -= cc2aa_n("Kxyuv") * Gamma1.block("aa")("wz");
    dlamb3_aaa("Kxyzuvw") -= cc2aa_r("Kxyuv") * Gamma1.block("aa")("wz");
    dlamb3_aaa("Kxyzuvw") += 4.0 * cc1a_n("Kxu") * Gamma2.block("aaaa")("vwzy");
    dlamb3_aaa("Kxyzuvw") += 4.0 * cc1a_r("Kxu") * Gamma2.block("aaaa")("vwzy");
    dlamb3_aaa("Kxyzuvw") += 4.0 * cc2aa_n("Kvwzy") * Gamma1.block("aa")("xu");
    dlamb3_aaa("Kxyzuvw") += 4.0 * cc2aa_r("Kvwzy") * Gamma1.block("aa")("xu");
    dlamb3_aaa("Kxyzuvw") += 2.0 * cc1a_n("Kxw") * Gamma2.block("aaaa")("uvzy");
    dlamb3_aaa("Kxyzuvw") += 2.0 * cc1a_r("Kxw") * Gamma2.block("aaaa")("uvzy");
    dlamb3_aaa("Kxyzuvw") += 2.0 * cc2aa_n("Kuvzy") * Gamma1.block("aa")("xw");
    dlamb3_aaa("Kxyzuvw") += 2.0 * cc2aa_r("Kuvzy") * Gamma1.block("aa")("xw");
    dlamb3_aaa("Kxyzuvw") += 8.0 * cc1a_n("Kuz") * Gamma1.block("aa")("xv")* Gamma1.block("aa")("yw");
    dlamb3_aaa("Kxyzuvw") += 8.0 * cc1a_r("Kuz") * Gamma1.block("aa")("xv")* Gamma1.block("aa")("yw");
    dlamb3_aaa("Kxyzuvw") += 8.0 * cc1a_n("Kxv") * Gamma1.block("aa")("uz")* Gamma1.block("aa")("yw");
    dlamb3_aaa("Kxyzuvw") += 8.0 * cc1a_r("Kxv") * Gamma1.block("aa")("uz")* Gamma1.block("aa")("yw");
    dlamb3_aaa("Kxyzuvw") += 8.0 * cc1a_n("Kyw") * Gamma1.block("aa")("uz")* Gamma1.block("aa")("xv");
    dlamb3_aaa("Kxyzuvw") += 8.0 * cc1a_r("Kyw") * Gamma1.block("aa")("uz")* Gamma1.block("aa")("xv");
    dlamb3_aaa("Kxyzuvw") += 4.0 * cc1a_n("Kwz") * Gamma1.block("aa")("xu")* Gamma1.block("aa")("yv");
    dlamb3_aaa("Kxyzuvw") += 4.0 * cc1a_r("Kwz") * Gamma1.block("aa")("xu")* Gamma1.block("aa")("yv");
    dlamb3_aaa("Kxyzuvw") += 4.0 * cc1a_n("Kxu") * Gamma1.block("aa")("wz")* Gamma1.block("aa")("yv");
    dlamb3_aaa("Kxyzuvw") += 4.0 * cc1a_r("Kxu") * Gamma1.block("aa")("wz")* Gamma1.block("aa")("yv");
    dlamb3_aaa("Kxyzuvw") += 4.0 * cc1a_n("Kyv") * Gamma1.block("aa")("wz")* Gamma1.block("aa")("xu");
    dlamb3_aaa("Kxyzuvw") += 4.0 * cc1a_r("Kyv") * Gamma1.block("aa")("wz")* Gamma1.block("aa")("xu");

    // beta-beta-beta
    dlamb3_bbb("KXYZUVW") += cc3bbb_n("KXYZUVW");
    dlamb3_bbb("KXYZUVW") += cc3bbb_r("KXYZUVW");
    dlamb3_bbb("KXYZUVW") -= 2.0 * cc1b_n("KUZ") * Gamma2.block("AAAA")("XYVW");
    dlamb3_bbb("KXYZUVW") -= 2.0 * cc1b_r("KUZ") * Gamma2.block("AAAA")("XYVW");
    dlamb3_bbb("KXYZUVW") -= 2.0 * cc2bb_n("KXYVW") * Gamma1.block("AA")("UZ");
    dlamb3_bbb("KXYZUVW") -= 2.0 * cc2bb_r("KXYVW") * Gamma1.block("AA")("UZ");
    dlamb3_bbb("KXYZUVW") -= cc1b_n("KWZ") * Gamma2.block("AAAA")("XYUV");
    dlamb3_bbb("KXYZUVW") -= cc1b_r("KWZ") * Gamma2.block("AAAA")("XYUV");
    dlamb3_bbb("KXYZUVW") -= cc2bb_n("KXYUV") * Gamma1.block("AA")("WZ");
    dlamb3_bbb("KXYZUVW") -= cc2bb_r("KXYUV") * Gamma1.block("AA")("WZ");
    dlamb3_bbb("KXYZUVW") += 4.0 * cc1b_n("KXU") * Gamma2.block("AAAA")("VWZY");
    dlamb3_bbb("KXYZUVW") += 4.0 * cc1b_r("KXU") * Gamma2.block("AAAA")("VWZY");
    dlamb3_bbb("KXYZUVW") += 4.0 * cc2bb_n("KVWZY") * Gamma1.block("AA")("XU");
    dlamb3_bbb("KXYZUVW") += 4.0 * cc2bb_r("KVWZY") * Gamma1.block("AA")("XU");
    dlamb3_bbb("KXYZUVW") += 2.0 * cc1b_n("KXW") * Gamma2.block("AAAA")("UVZY");
    dlamb3_bbb("KXYZUVW") += 2.0 * cc1b_r("KXW") * Gamma2.block("AAAA")("UVZY");
    dlamb3_bbb("KXYZUVW") += 2.0 * cc2bb_n("KUVZY") * Gamma1.block("AA")("XW");
    dlamb3_bbb("KXYZUVW") += 2.0 * cc2bb_r("KUVZY") * Gamma1.block("AA")("XW");
    dlamb3_bbb("KXYZUVW") += 8.0 * cc1b_n("KUZ") * Gamma1.block("AA")("XV")* Gamma1.block("AA")("YW");
    dlamb3_bbb("KXYZUVW") += 8.0 * cc1b_r("KUZ") * Gamma1.block("AA")("XV")* Gamma1.block("AA")("YW");
    dlamb3_bbb("KXYZUVW") += 8.0 * cc1b_n("KXV") * Gamma1.block("AA")("UZ")* Gamma1.block("AA")("YW");
    dlamb3_bbb("KXYZUVW") += 8.0 * cc1b_r("KXV") * Gamma1.block("AA")("UZ")* Gamma1.block("AA")("YW");
    dlamb3_bbb("KXYZUVW") += 8.0 * cc1b_n("KYW") * Gamma1.block("AA")("UZ")* Gamma1.block("AA")("XV");
    dlamb3_bbb("KXYZUVW") += 8.0 * cc1b_r("KYW") * Gamma1.block("AA")("UZ")* Gamma1.block("AA")("XV");
    dlamb3_bbb("KXYZUVW") += 4.0 * cc1b_n("KWZ") * Gamma1.block("AA")("XU")* Gamma1.block("AA")("YV");
    dlamb3_bbb("KXYZUVW") += 4.0 * cc1b_r("KWZ") * Gamma1.block("AA")("XU")* Gamma1.block("AA")("YV");
    dlamb3_bbb("KXYZUVW") += 4.0 * cc1b_n("KXU") * Gamma1.block("AA")("WZ")* Gamma1.block("AA")("YV");
    dlamb3_bbb("KXYZUVW") += 4.0 * cc1b_r("KXU") * Gamma1.block("AA")("WZ")* Gamma1.block("AA")("YV");
    dlamb3_bbb("KXYZUVW") += 4.0 * cc1b_n("KYV") * Gamma1.block("AA")("WZ")* Gamma1.block("AA")("XU");
    dlamb3_bbb("KXYZUVW") += 4.0 * cc1b_r("KYV") * Gamma1.block("AA")("WZ")* Gamma1.block("AA")("XU");

    // alpha-alpha-beta
    dlamb3_aab("KxyZuvW") += cc3aab_n("KxyZuvW");
    dlamb3_aab("KxyZuvW") += cc3aab_r("KxyZuvW");
    dlamb3_aab("KxyZuvW") -= cc1a_n("Kxu") * Lambda2_.block("aAaA")("yZvW");
    dlamb3_aab("KxyZuvW") -= cc1a_r("Kxu") * Lambda2_.block("aAaA")("yZvW");
    dlamb3_aab("KxyZuvW") -= Gamma1.block("aa")("xu") * dlamb_ab("KyZvW");
    dlamb3_aab("KxyZuvW") += cc1a_n("Kxv") * Lambda2_.block("aAaA")("yZuW");
    dlamb3_aab("KxyZuvW") += cc1a_r("Kxv") * Lambda2_.block("aAaA")("yZuW");
    dlamb3_aab("KxyZuvW") += Gamma1.block("aa")("xv") * dlamb_ab("KyZuW");
    dlamb3_aab("KxyZuvW") += cc1a_n("Kyu") * Lambda2_.block("aAaA")("xZvW");
    dlamb3_aab("KxyZuvW") += cc1a_r("Kyu") * Lambda2_.block("aAaA")("xZvW");
    dlamb3_aab("KxyZuvW") += Gamma1.block("aa")("yu") * dlamb_ab("KxZvW");
    dlamb3_aab("KxyZuvW") -= cc1a_n("Kyv") * Lambda2_.block("aAaA")("xZuW");
    dlamb3_aab("KxyZuvW") -= cc1a_r("Kyv") * Lambda2_.block("aAaA")("xZuW");
    dlamb3_aab("KxyZuvW") -= Gamma1.block("aa")("yv") * dlamb_ab("KxZuW");
    dlamb3_aab("KxyZuvW") -= cc1b_n("KZW") * Lambda2_.block("aaaa")("xyuv");
    dlamb3_aab("KxyZuvW") -= cc1b_r("KZW") * Lambda2_.block("aaaa")("xyuv");
    dlamb3_aab("KxyZuvW") -= Gamma1.block("AA")("ZW") * dlamb_aa("Kxyuv");
    dlamb3_aab("KxyZuvW") -= cc1a_n("Kxu") * Gamma1.block("aa")("yv") * Gamma1.block("AA")("ZW");
    dlamb3_aab("KxyZuvW") -= cc1a_r("Kxu") * Gamma1.block("aa")("yv") * Gamma1.block("AA")("ZW");
    dlamb3_aab("KxyZuvW") -= cc1a_n("Kyv") * Gamma1.block("aa")("xu") * Gamma1.block("AA")("ZW");
    dlamb3_aab("KxyZuvW") -= cc1a_r("Kyv") * Gamma1.block("aa")("xu") * Gamma1.block("AA")("ZW");
    dlamb3_aab("KxyZuvW") -= cc1b_n("KZW") * Gamma1.block("aa")("yv") * Gamma1.block("aa")("xu");
    dlamb3_aab("KxyZuvW") -= cc1b_r("KZW") * Gamma1.block("aa")("yv") * Gamma1.block("aa")("xu");
    dlamb3_aab("KxyZuvW") += cc1a_n("Kxv") * Gamma1.block("aa")("yu") * Gamma1.block("AA")("ZW");
    dlamb3_aab("KxyZuvW") += cc1a_r("Kxv") * Gamma1.block("aa")("yu") * Gamma1.block("AA")("ZW");
    dlamb3_aab("KxyZuvW") += cc1a_n("Kyu") * Gamma1.block("aa")("xv") * Gamma1.block("AA")("ZW");
    dlamb3_aab("KxyZuvW") += cc1a_r("Kyu") * Gamma1.block("aa")("xv") * Gamma1.block("AA")("ZW");
    dlamb3_aab("KxyZuvW") += cc1b_n("KZW") * Gamma1.block("aa")("yu") * Gamma1.block("aa")("xv");
    dlamb3_aab("KxyZuvW") += cc1b_r("KZW") * Gamma1.block("aa")("yu") * Gamma1.block("aa")("xv");

    // alpha-beta-beta
    dlamb3_abb("KxYZuVW") += cc3abb_n("KxYZuVW");
    dlamb3_abb("KxYZuVW") += cc3abb_r("KxYZuVW");
    dlamb3_abb("KxYZuVW") -= cc1a_n("Kxu") * Lambda2_.block("AAAA")("YZVW");
    dlamb3_abb("KxYZuVW") -= cc1a_r("Kxu") * Lambda2_.block("AAAA")("YZVW");
    dlamb3_abb("KxYZuVW") -= Gamma1.block("aa")("xu") * dlamb_bb("KYZVW");
    dlamb3_abb("KxYZuVW") -= cc1b_n("KYV") * Lambda2_.block("aAaA")("xZuW");
    dlamb3_abb("KxYZuVW") -= cc1b_r("KYV") * Lambda2_.block("aAaA")("xZuW");
    dlamb3_abb("KxYZuVW") -= Gamma1.block("AA")("YV") * dlamb_ab("KxZuW");
    dlamb3_abb("KxYZuVW") += cc1b_n("KYW") * Lambda2_.block("aAaA")("xZuV");
    dlamb3_abb("KxYZuVW") += cc1b_r("KYW") * Lambda2_.block("aAaA")("xZuV");
    dlamb3_abb("KxYZuVW") += Gamma1.block("AA")("YW") * dlamb_ab("KxZuV");
    dlamb3_abb("KxYZuVW") += cc1b_n("KZV") * Lambda2_.block("aAaA")("xYuW");
    dlamb3_abb("KxYZuVW") += cc1b_r("KZV") * Lambda2_.block("aAaA")("xYuW");
    dlamb3_abb("KxYZuVW") += Gamma1.block("AA")("ZV") * dlamb_ab("KxYuW");
    dlamb3_abb("KxYZuVW") -= cc1b_n("KZW") * Lambda2_.block("aAaA")("xYuV");
    dlamb3_abb("KxYZuVW") -= cc1b_r("KZW") * Lambda2_.block("aAaA")("xYuV");
    dlamb3_abb("KxYZuVW") -= Gamma1.block("AA")("ZW") * dlamb_ab("KxYuV");
    dlamb3_abb("KxYZuVW") -= cc1a_n("Kxu") * Gamma1.block("AA")("YV") * Gamma1.block("AA")("ZW");
    dlamb3_abb("KxYZuVW") -= cc1a_r("Kxu") * Gamma1.block("AA")("YV") * Gamma1.block("AA")("ZW");
    dlamb3_abb("KxYZuVW") -= cc1b_n("KYV") * Gamma1.block("aa")("xu") * Gamma1.block("AA")("ZW");
    dlamb3_abb("KxYZuVW") -= cc1b_r("KYV") * Gamma1.block("aa")("xu") * Gamma1.block("AA")("ZW");
    dlamb3_abb("KxYZuVW") -= cc1b_n("KZW") * Gamma1.block("AA")("YV") * Gamma1.block("aa")("xu");
    dlamb3_abb("KxYZuVW") -= cc1b_r("KZW") * Gamma1.block("AA")("YV") * Gamma1.block("aa")("xu");
    dlamb3_abb("KxYZuVW") += cc1a_n("Kxu") * Gamma1.block("AA")("ZV") * Gamma1.block("AA")("YW");
    dlamb3_abb("KxYZuVW") += cc1a_r("Kxu") * Gamma1.block("AA")("ZV") * Gamma1.block("AA")("YW");
    dlamb3_abb("KxYZuVW") += cc1b_n("KZV") * Gamma1.block("aa")("xu") * Gamma1.block("AA")("YW");
    dlamb3_abb("KxYZuVW") += cc1b_r("KZV") * Gamma1.block("aa")("xu") * Gamma1.block("AA")("YW");
    dlamb3_abb("KxYZuVW") += cc1b_n("KYW") * Gamma1.block("AA")("ZV") * Gamma1.block("aa")("xu");
    dlamb3_abb("KxYZuVW") += cc1b_r("KYW") * Gamma1.block("AA")("ZV") * Gamma1.block("aa")("xu");
}

void DSRG_MRPT2::set_tensor() {
    outfile->Printf("\n    Initializing RDMs, DSRG Tensors and CI integrals. ");
    I = BTF_->build(CoreTensor, "identity matrix", spin_cases({"gg"}));
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
    Gamma1 = BTF_->build(CoreTensor, "Gamma1", spin_cases({"gg"}));
    Gamma2 = BTF_->build(CoreTensor, "Gamma2", spin_cases({"aaaa"}));
    Eta1 = BTF_->build(CoreTensor, "Eta1", spin_cases({"gg"}));

    Gamma1.block("aa")("pq") = rdms_.g1a()("pq");
    Gamma1.block("AA")("pq") = rdms_.g1b()("pq");

    for (const std::string& block : {"aa", "AA", "vv", "VV"}) {
        (Eta1.block(block)).iterate([&](const std::vector<size_t>& i, double& value) {
            value = (i[0] == i[1]) ? 1.0 : 0.0;
        });
    }

    for (const std::string& block : {"cc", "CC"}) {
        (Gamma1.block(block)).iterate([&](const std::vector<size_t>& i, double& value) {
            value = (i[0] == i[1]) ? 1.0 : 0.0;
        });
    }

    Eta1["uv"] -= Gamma1["uv"];
    Eta1["UV"] -= Gamma1["UV"];

    // 2-body density
    Gamma2.block("aaaa")("pqrs") = rdms_.g2aa()("pqrs");
    Gamma2.block("aAaA")("pqrs") = rdms_.g2ab()("pqrs");
    Gamma2.block("AAAA")("pqrs") = rdms_.g2bb()("pqrs");
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
    V = BTF_->build(CoreTensor, "Electron Repulsion Integral", spin_cases({"gggg"}));
    V.iterate([&](const std::vector<size_t>& i, const std::vector<SpinType>& spin, double& value) {
        if (spin[0] == AlphaSpin) {
            if (spin[1] == AlphaSpin) {
                value = ints_->aptei_aa(i[0], i[1], i[2], i[3]);
            } else {
                value = ints_->aptei_ab(i[0], i[1], i[2], i[3]);
            }
        } else if (spin[1] == BetaSpin) {
            value = ints_->aptei_bb(i[0], i[1], i[2], i[3]);
        }
    });

    V_N_Alpha = BTF_->build(CoreTensor, "normal Dimention-reduced Electron Repulsion Integral alpha", {"gg"});
    V_N_Beta = BTF_->build(CoreTensor, "normal Dimention-reduced Electron Repulsion Integral beta", {"gg"});
    V_R_Alpha = BTF_->build(CoreTensor, "index-reversed Dimention-reduced Electron Repulsion Integral alpha", {"gg"});
    V_R_Beta = BTF_->build(CoreTensor, "index-reversed Dimention-reduced Electron Repulsion Integral beta", {"GG"});
    V_all_Beta = BTF_->build(CoreTensor, "normal Dimention-reduced Electron Repulsion Integral all beta", {"GG"});

    // Summation of V["pmqm"] over index "m"
    V_N_Alpha["pq"] = V["pmqn"] * I["mn"];
    // Summation of V["pMqM"] over index "M"
    V_N_Beta["pq"] = V["pMqN"] * I["MN"];
    // Summation of V["mpmq"] over index "m"
    V_R_Alpha["pq"] = V["mpnq"] * I["mn"];
    // Summation of V["mPmQ"] over index "m"
    V_R_Beta["PQ"] = V["mPnQ"] * I["mn"];
    // Summation of V["PMQM"] over index "M"
    V_all_Beta["PQ"] = V["PMQN"] * I["MN"];
}

void DSRG_MRPT2::set_fock() {
    F = BTF_->build(CoreTensor, "Fock Matrix", spin_cases({"gg"}));
    psi::SharedMatrix D1a(new psi::Matrix("D1a", nmo_, nmo_));
    psi::SharedMatrix D1b(new psi::Matrix("D1b", nmo_, nmo_));

    // Fill core-core blocks
    for (size_t m = 0, ncore = core_mos_.size(); m < ncore; m++) {
        D1a->set(core_mos_[m], core_mos_[m], 1.0);
        D1b->set(core_mos_[m], core_mos_[m], 1.0);
    }

    // Fill active-active blocks
    Gamma1.block("aa").citerate([&](const std::vector<size_t>& i, const double& value) {
        D1a->set(actv_mos_[i[0]], actv_mos_[i[1]], value);
    });
    Gamma1.block("AA").citerate([&](const std::vector<size_t>& i, const double& value) {
        D1b->set(actv_mos_[i[0]], actv_mos_[i[1]], value);
    });

    // Make Fock matrices
    ints_->make_fock_matrix(D1a, D1b);
    F.iterate([&](const std::vector<size_t>& i, const std::vector<SpinType>& spin, double& value) {
        if (spin[0] == AlphaSpin) {
            value = ints_->get_fock_a(i[0], i[1]);
        } else {
            value = ints_->get_fock_b(i[0], i[1]);
        }
    });
}

void DSRG_MRPT2::set_dsrg_tensor() {
    Eeps1 = BTF_->build(CoreTensor, "e^[-s*(Delta1)^2]", spin_cases({"hp"}));
    Eeps1_m1 = BTF_->build(CoreTensor, "{1-e^[-s*(Delta1)^2]}/(Delta1)", spin_cases({"hp"}));
    Eeps1_m2 = BTF_->build(CoreTensor, "{1-e^[-s*(Delta1)^2]}/(Delta1)^2", spin_cases({"hp"}));
    Eeps2 = BTF_->build(CoreTensor, "e^[-s*(Delta2)^2]", spin_cases({"hhpp"}));
    Eeps2_p = BTF_->build(CoreTensor, "1+e^[-s*(Delta2)^2]", spin_cases({"hhpp"}));
    Eeps2_m1 = BTF_->build(CoreTensor, "{1-e^[-s*(Delta2)^2]}/(Delta2)", spin_cases({"hhpp"}));
    Eeps2_m2 = BTF_->build(CoreTensor, "{1-e^[-s*(Delta2)^2]}/(Delta2)^2", spin_cases({"hhpp"}));
    Delta1 = BTF_->build(CoreTensor, "Delta1", spin_cases({"gg"}));
    Delta2 = BTF_->build(CoreTensor, "Delta2", spin_cases({"hhpp"}));
    DelGam1 = BTF_->build(CoreTensor, "Delta1 * Gamma1", spin_cases({"aa"}));
    DelEeps1 = BTF_->build(CoreTensor, "Delta1 * Eeps1", spin_cases({"hp"}));
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
                { value = dsrg_source_->compute_denominator(Fa_[i[0]] + Fa_[i[1]] - Fa_[i[2]] - Fa_[i[3]], 1);}
            else if (spin[0] == BetaSpin&& spin[1] == BetaSpin)
                { value = dsrg_source_->compute_denominator(Fb_[i[0]] + Fb_[i[1]] - Fb_[i[2]] - Fb_[i[3]], 1);}
            else { value = dsrg_source_->compute_denominator(Fa_[i[0]] + Fb_[i[1]] - Fa_[i[2]] - Fb_[i[3]], 1);}
        }
    );
    Eeps2_m2.iterate(
        [&](const std::vector<size_t>& i, const std::vector<SpinType>& spin, double& value) {
            if (spin[0] == AlphaSpin&& spin[1] == AlphaSpin) 
                { value = dsrg_source_->compute_denominator(Fa_[i[0]] + Fa_[i[1]] - Fa_[i[2]] - Fa_[i[3]], 2);}
            else if (spin[0] == BetaSpin&& spin[1] == BetaSpin)
                { value = dsrg_source_->compute_denominator(Fb_[i[0]] + Fb_[i[1]] - Fb_[i[2]] - Fb_[i[3]], 2);}
            else { value = dsrg_source_->compute_denominator(Fa_[i[0]] + Fb_[i[1]] - Fa_[i[2]] - Fb_[i[3]], 2);}
        }
    ); 

    Eeps1_m1.iterate(
        [&](const std::vector<size_t>& i, const std::vector<SpinType>& spin, double& value) {
            if (spin[0] == AlphaSpin) { value = dsrg_source_->compute_denominator(Fa_[i[0]] - Fa_[i[1]], 1);}
            else { value = dsrg_source_->compute_denominator(Fb_[i[0]] - Fb_[i[1]], 1);}
        }
    );

    Eeps1_m2.iterate(
        [&](const std::vector<size_t>& i, const std::vector<SpinType>& spin, double& value) {
            if (spin[0] == AlphaSpin) { value = dsrg_source_->compute_denominator(Fa_[i[0]] - Fa_[i[1]], 2);}
            else { value = dsrg_source_->compute_denominator(Fb_[i[0]] - Fb_[i[1]], 2);}
        }
    );
    
    // An intermediate tensor : T2 / Delta
    T2OverDelta["ijab"] += V["ijab"] * Eeps2_m2["ijab"];
    T2OverDelta["iJaB"] += V["iJaB"] * Eeps2_m2["iJaB"];   

    // Delta1 * Gamma1
    DelGam1["xu"] = Delta1["xu"] * Gamma1["xu"];
    DelGam1["XU"] = Delta1["XU"] * Gamma1["XU"];

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
        Tau2["ijab"] += 0.25 * V_["cdkl"] * Gamma1["ki"] * Gamma1["lj"] * Eta1["ac"] * Eta1["bd"];
        Tau2["IJAB"] += 0.25 * V_["CDKL"] * Gamma1["KI"] * Gamma1["LJ"] * Eta1["AC"] * Eta1["BD"];
        Tau2["iJaB"] += 0.25 * V_["cDkL"] * Gamma1["ki"] * Gamma1["LJ"] * Eta1["ac"] * Eta1["BD"];
    }
    // <[V, T2]> C_4 (C_2)^2 PP
    if (X1_TERM) {
        Tau2["uvab"] += 0.125 * V_["cdxy"] * Eta1["ac"] * Eta1["bd"] * Lambda2_["xyuv"];
        Tau2["UVAB"] += 0.125 * V_["CDXY"] * Eta1["AC"] * Eta1["BD"] * Lambda2_["XYUV"];
        Tau2["uVaB"] += 0.250 * V_["cDxY"] * Eta1["ac"] * Eta1["BD"] * Lambda2_["xYuV"];
    }
    // <[V, T2]> C_4 (C_2)^2 HH
    if (X2_TERM) {
        Tau2["ijxy"] += 0.125 * V_["uvkl"] * Gamma1["ki"] * Gamma1["lj"] * Lambda2_["xyuv"];
        Tau2["IJXY"] += 0.125 * V_["UVKL"] * Gamma1["KI"] * Gamma1["LJ"] * Lambda2_["XYUV"];
        Tau2["iJxY"] += 0.250 * V_["uVkL"] * Gamma1["ki"] * Gamma1["LJ"] * Lambda2_["xYuV"];
    }
    // <[V, T2]> C_4 (C_2)^2 PH
    if (X3_TERM) {
        Tau2["iuya"] -= 0.25 * V_["vbjx"] * Gamma1["ij"] * Eta1["ab"] * Lambda2_["xyuv"];
        Tau2["iuya"] -= 0.25 * V_["bVjX"] * Gamma1["ij"] * Eta1["ab"] * Lambda2_["yXuV"];
        Tau2["IUYA"] -= 0.25 * V_["VBJX"] * Gamma1["IJ"] * Eta1["AB"] * Lambda2_["XYUV"];
        Tau2["IUYA"] -= 0.25 * V_["vBxJ"] * Gamma1["IJ"] * Eta1["AB"] * Lambda2_["xYvU"];
        Tau2["iUyA"] -= 0.25 * V_["vBjX"] * Gamma1["ij"] * Eta1["AB"] * Lambda2_["yXvU"];

        Tau2["uiya"] -= 0.25 * V_["vbxj"] * Gamma1["ij"] * Eta1["ab"] * Lambda2_["xyuv"];
        Tau2["uiya"] += 0.25 * V_["bVjX"] * Gamma1["ij"] * Eta1["ab"] * Lambda2_["yXuV"];
        Tau2["UIYA"] -= 0.25 * V_["VBXJ"] * Gamma1["IJ"] * Eta1["AB"] * Lambda2_["XYUV"];
        Tau2["UIYA"] += 0.25 * V_["vBxJ"] * Gamma1["IJ"] * Eta1["AB"] * Lambda2_["xYvU"];
        Tau2["uIyA"] -= 0.25 * V_["vBxJ"] * Gamma1["IJ"] * Eta1["AB"] * Lambda2_["xyuv"];
        Tau2["uIyA"] += 0.25 * V_["VBXJ"] * Gamma1["IJ"] * Eta1["AB"] * Lambda2_["yXuV"];

        Tau2["uiay"] -= 0.25 * V_["bvxj"] * Gamma1["ij"] * Eta1["ab"] * Lambda2_["xyuv"];
        Tau2["uiay"] -= 0.25 * V_["bVjX"] * Gamma1["ij"] * Eta1["ab"] * Lambda2_["yXuV"];
        Tau2["UIAY"] -= 0.25 * V_["BVXJ"] * Gamma1["IJ"] * Eta1["AB"] * Lambda2_["XYUV"];
        Tau2["UIAY"] -= 0.25 * V_["vBxJ"] * Gamma1["IJ"] * Eta1["AB"] * Lambda2_["xYvU"];
        Tau2["uIaY"] -= 0.25 * V_["bVxJ"] * Gamma1["IJ"] * Eta1["ab"] * Lambda2_["xYuV"];

        Tau2["iuay"] -= 0.25 * V_["bvjx"] * Gamma1["ij"] * Eta1["ab"] * Lambda2_["xyuv"];
        Tau2["iuay"] += 0.25 * V_["bVjX"] * Gamma1["ij"] * Eta1["ab"] * Lambda2_["yXuV"];
        Tau2["IUAY"] -= 0.25 * V_["BVJX"] * Gamma1["IJ"] * Eta1["AB"] * Lambda2_["XYUV"];
        Tau2["IUAY"] += 0.25 * V_["vBxJ"] * Gamma1["IJ"] * Eta1["AB"] * Lambda2_["xYvU"];
        Tau2["iUaY"] -= 0.25 * V_["bVjX"] * Gamma1["ij"] * Eta1["ab"] * Lambda2_["XYUV"];
        Tau2["iUaY"] += 0.25 * V_["bvjx"] * Gamma1["ij"] * Eta1["ab"] * Lambda2_["xYvU"];
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
        Sigma["jb"] += T1_["ia"] * Gamma1["ij"] * Eta1["ab"];
        Sigma["JB"] += T1_["IA"] * Gamma1["IJ"] * Eta1["AB"];
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
        Xi["ia"] += F_["bj"] * Gamma1["ij"] * Eta1["ab"];
        Xi["IA"] += F_["BJ"] * Gamma1["IJ"] * Eta1["AB"];
    }
    Xi.block("aa").zero();
    Xi.block("AA").zero();

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
        Kappa["klcd"] += 0.25 * T2_["ijab"] * Gamma1["ki"] * Gamma1["lj"] * Eta1["ac"] * Eta1["bd"];
        Kappa["KLCD"] += 0.25 * T2_["IJAB"] * Gamma1["KI"] * Gamma1["LJ"] * Eta1["AC"] * Eta1["BD"];
        Kappa["kLcD"] += 0.25 * T2_["iJaB"] * Gamma1["ki"] * Gamma1["LJ"] * Eta1["ac"] * Eta1["BD"];
    }
    // <[V, T2]> C_4 (C_2)^2 PP
    if (X1_TERM) {
        Kappa["xycd"] += 0.125 * T2_["uvab"] * Eta1["ac"] * Eta1["bd"] * Lambda2_["xyuv"];
        Kappa["XYCD"] += 0.125 * T2_["UVAB"] * Eta1["AC"] * Eta1["BD"] * Lambda2_["XYUV"];
        Kappa["xYcD"] += 0.250 * T2_["uVaB"] * Eta1["ac"] * Eta1["BD"] * Lambda2_["xYuV"];
    }
    // <[V, T2]> C_4 (C_2)^2 HH
    if (X2_TERM) {
        Kappa["kluv"] += 0.125 * T2_["ijxy"] * Gamma1["ki"] * Gamma1["lj"] * Lambda2_["xyuv"];
        Kappa["KLUV"] += 0.125 * T2_["IJXY"] * Gamma1["KI"] * Gamma1["LJ"] * Lambda2_["XYUV"];
        Kappa["kLuV"] += 0.250 * T2_["iJxY"] * Gamma1["ki"] * Gamma1["LJ"] * Lambda2_["xYuV"];
    }
    // <[V, T2]> C_4 (C_2)^2 PH
    if (X3_TERM) {
        Kappa["jxvb"] -= 0.25 * T2_["iuya"] * Gamma1["ij"] * Eta1["ab"] * Lambda2_["xyuv"];
        Kappa["jxvb"] -= 0.25 * T2_["iUaY"] * Gamma1["ij"] * Eta1["ab"] * Lambda2_["xYvU"];
        Kappa["JXVB"] -= 0.25 * T2_["IUYA"] * Gamma1["IJ"] * Eta1["AB"] * Lambda2_["XYUV"];
        Kappa["JXVB"] -= 0.25 * T2_["uIyA"] * Gamma1["IJ"] * Eta1["AB"] * Lambda2_["yXuV"];
        Kappa["jXvB"] -= 0.25 * T2_["iUyA"] * Gamma1["ij"] * Eta1["AB"] * Lambda2_["yXvU"];

        Kappa["xjvb"] -= 0.25 * T2_["uiya"] * Gamma1["ij"] * Eta1["ab"] * Lambda2_["xyuv"];
        Kappa["xjvb"] += 0.25 * T2_["iUaY"] * Gamma1["ij"] * Eta1["ab"] * Lambda2_["xYvU"];
        Kappa["XJVB"] -= 0.25 * T2_["UIYA"] * Gamma1["IJ"] * Eta1["AB"] * Lambda2_["XYUV"];
        Kappa["XJVB"] += 0.25 * T2_["uIyA"] * Gamma1["IJ"] * Eta1["AB"] * Lambda2_["yXuV"];
        Kappa["xJvB"] -= 0.25 * T2_["uIyA"] * Gamma1["IJ"] * Eta1["AB"] * Lambda2_["xyuv"];
        Kappa["xJvB"] += 0.25 * T2_["UIYA"] * Gamma1["IJ"] * Eta1["AB"] * Lambda2_["xYvU"];

        Kappa["xjbv"] -= 0.25 * T2_["uiay"] * Gamma1["ij"] * Eta1["ab"] * Lambda2_["xyuv"];
        Kappa["xjbv"] -= 0.25 * T2_["iUaY"] * Gamma1["ij"] * Eta1["ab"] * Lambda2_["xYvU"];
        Kappa["XJBV"] -= 0.25 * T2_["UIAY"] * Gamma1["IJ"] * Eta1["AB"] * Lambda2_["XYUV"];
        Kappa["XJBV"] -= 0.25 * T2_["uIyA"] * Gamma1["IJ"] * Eta1["AB"] * Lambda2_["yXuV"];
        Kappa["xJbV"] -= 0.25 * T2_["uIaY"] * Gamma1["IJ"] * Eta1["ab"] * Lambda2_["xYuV"];

        Kappa["jxbv"] -= 0.25 * T2_["iuay"] * Gamma1["ij"] * Eta1["ab"] * Lambda2_["xyuv"];
        Kappa["jxbv"] += 0.25 * T2_["iUaY"] * Gamma1["ij"] * Eta1["ab"] * Lambda2_["xYvU"];
        Kappa["JXBV"] -= 0.25 * T2_["IUAY"] * Gamma1["IJ"] * Eta1["AB"] * Lambda2_["XYUV"];
        Kappa["JXBV"] += 0.25 * T2_["uIyA"] * Gamma1["IJ"] * Eta1["AB"] * Lambda2_["yXuV"];
        Kappa["jXbV"] -= 0.25 * T2_["iUaY"] * Gamma1["ij"] * Eta1["ab"] * Lambda2_["XYUV"];
        Kappa["jXbV"] += 0.25 * T2_["iuay"] * Gamma1["ij"] * Eta1["ab"] * Lambda2_["yXuV"];
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
    W_ = BTF_->build(CoreTensor, "Lagrangian", spin_cases({"gg"}));
    BlockedTensor temp = BTF_->build(CoreTensor, "temporal tensor", spin_cases({"hhpp"}));

    //NOTICE: w for {virtual-general}
    if (CORRELATION_TERM) {
        W_["pe"] += 0.5 * Sigma3["ie"] * F["ip"];
    }

    if (CORRELATION_TERM) {
        W_["pe"] += 0.5 * Xi3["ie"] * F["ip"];
    }

    if (CORRELATION_TERM) {
        W_["pe"] += Tau1["ijeb"] * V["ijpb"];
        W_["pe"] += 2.0 * Tau1["iJeB"] * V["iJpB"];

        temp["kled"] += Kappa["kled"] * Eeps2_p["kled"];
        temp["kLeD"] += Kappa["kLeD"] * Eeps2_p["kLeD"];
        W_["pe"] += temp["kled"] * V["klpd"];
        W_["pe"] += 2.0 * temp["kLeD"] * V["kLpD"];
        temp.zero();
    }
    W_["pe"] += Z["e,m1"] * F["m1,p"];
    W_["pe"] += Z["eu"] * H["vp"] * Gamma1["uv"];
    W_["pe"] += Z["eu"] * V_N_Alpha["v,p"] * Gamma1["uv"];
    W_["pe"] += Z["eu"] * V_N_Beta["v,p"] * Gamma1["uv"];
    W_["pe"] += 0.5 * Z["eu"] * V["xypv"] * Gamma2["uvxy"];
    W_["pe"] += Z["eu"] * V["xYpV"] * Gamma2["uVxY"];
    W_["pe"] += Z["e,f1"] * F["f1,p"];
    W_["ei"] = W_["ie"];

    //NOTICE: w for {core-hole}
    if (CORRELATION_TERM) {
        W_["jm"] += 0.5 * Sigma3["ma"] * F["ja"];
        W_["jm"] += 0.5 * Sigma3["ia"] * V["ijam"];
        W_["jm"] += 0.5 * Sigma3["IA"] * V["jImA"];
        W_["jm"] += 0.5 * Sigma3["ia"] * V["imaj"];
        W_["jm"] += 0.5 * Sigma3["IA"] * V["mIjA"];
    }

    if (CORRELATION_TERM) {
        W_["jm"] += 0.5 * Xi3["ma"] * F["ja"];
        W_["jm"] += 0.5 * Xi3["ia"] * V["ijam"];
        W_["jm"] += 0.5 * Xi3["IA"] * V["jImA"];
        W_["jm"] += 0.5 * Xi3["ia"] * V["imaj"];
        W_["jm"] += 0.5 * Xi3["IA"] * V["mIjA"];
    }

    if (CORRELATION_TERM) {
        W_["im"] += Tau1["mjab"] * V["ijab"];
        W_["im"] += 2.0 * Tau1["mJaB"] * V["iJaB"];

        temp["mlcd"] += Kappa["mlcd"] * Eeps2_p["mlcd"];
        temp["mLcD"] += Kappa["mLcD"] * Eeps2_p["mLcD"];
        W_["im"] += temp["mlcd"] * V["ilcd"];
        W_["im"] += 2.0 * temp["mLcD"] * V["iLcD"];
        temp.zero();
    }
    W_["im"] += Z["e1,m"] * F["i,e1"];
    W_["im"] += Z["e1,m1"] * V["m1,m,e1,i"];
    W_["im"] += Z["E1,M1"] * V["m,M1,i,E1"];
    W_["im"] += Z["e1,m1"] * V["m1,i,e1,m"];
    W_["im"] += Z["E1,M1"] * V["i,M1,m,E1"];
    W_["im"] += Z["mu"] * F["ui"];
    W_["im"] -= Z["mu"] * H["vi"] * Gamma1["uv"];
    W_["im"] -= Z["mu"] * V_N_Alpha["vi"] * Gamma1["uv"];
    W_["im"] -= Z["mu"] * V_N_Beta["vi"] * Gamma1["uv"];
    W_["im"] -= 0.5 * Z["mu"] * V["xyiv"] * Gamma2["uvxy"];
    W_["im"] -= Z["mu"] * V["xYiV"] * Gamma2["uVxY"];
    W_["im"] += Z["n1,u"] * V["u,i,n1,m"];
    W_["im"] += Z["N1,U"] * V["i,U,m,N1"];
    W_["im"] += Z["n1,u"] * V["u,m,n1,i"];
    W_["im"] += Z["N1,U"] * V["m,U,i,N1"];
    W_["im"] -= Z["n1,u"] * Gamma1["uv"] * V["v,i,n1,m"];
    W_["im"] -= Z["N1,U"] * Gamma1["UV"] * V["i,V,m,N1"];
    W_["im"] -= Z["n1,u"] * Gamma1["uv"] * V["v,m,n1,i"];
    W_["im"] -= Z["N1,U"] * Gamma1["UV"] * V["m,V,i,N1"];
    W_["im"] += Z["e1,u"] * Gamma1["uv"] * V["v,m,e1,i"];
    W_["im"] += Z["E1,U"] * Gamma1["UV"] * V["m,V,i,E1"];
    W_["im"] += Z["e1,u"] * Gamma1["uv"] * V["v,i,e1,m"];
    W_["im"] += Z["E1,U"] * Gamma1["UV"] * V["i,V,m,E1"];
    W_["im"] += Z["m,n1"] * F["n1,i"];
    W_["im"] += Z["m1,n1"] * V["n1,i,m1,m"];
    W_["im"] += Z["M1,N1"] * V["i,N1,m,M1"];
    W_["im"] += Z["uv"] * V["vium"];
    W_["im"] += Z["UV"] * V["iVmU"];
    W_["im"] += Z["e1,f"] * V["f,i,e1,m"];
    W_["im"] += Z["E1,F"] * V["i,F,m,E1"];

    // CI contribution
    W_.block("cc")("nm") += 0.5 * V.block("acac")("umvn") * x_ci("I") * cc1a_n("Iuv");
    W_.block("cc")("nm") += 0.5 * V.block("acac")("umvn") * x_ci("J") * cc1a_r("Juv");
    W_.block("cc")("nm") += 0.5 * V.block("cAcA")("mUnV") * x_ci("I") * cc1b_n("IUV");
    W_.block("cc")("nm") += 0.5 * V.block("cAcA")("mUnV") * x_ci("J") * cc1b_r("JUV");

    W_.block("ac")("xm") += 0.5 * V.block("acaa")("umvx") * x_ci("I") * cc1a_n("Iuv");
    W_.block("ac")("xm") += 0.5 * V.block("acaa")("umvx") * x_ci("J") * cc1a_r("Juv");
    W_.block("ac")("xm") += 0.5 * V.block("cAaA")("mUxV") * x_ci("I") * cc1b_n("IUV");
    W_.block("ac")("xm") += 0.5 * V.block("cAaA")("mUxV") * x_ci("J") * cc1b_r("JUV");
    W_["mu"] = W_["um"];

    //NOTICE: w for {active-active}
    if (CORRELATION_TERM) {
        W_["zw"] += 0.5 * Sigma3["wa"] * F["za"];
        W_["zw"] += 0.5 * Sigma3["iw"] * F["iz"];
        W_["zw"] += 0.5 * Sigma3["ia"] * V["ivaz"] * Gamma1["wv"];
        W_["zw"] += 0.5 * Sigma3["IA"] * V["vIzA"] * Gamma1["wv"];
        W_["zw"] += 0.5 * Sigma3["ia"] * V["izau"] * Gamma1["uw"];
        W_["zw"] += 0.5 * Sigma3["IA"] * V["zIuA"] * Gamma1["uw"];
    }

    if (CORRELATION_TERM) {
        W_["zw"] += 0.5 * Xi3["wa"] * F["za"];
        W_["zw"] += 0.5 * Xi3["iw"] * F["iz"];
        W_["zw"] += 0.5 * Xi3["ia"] * V["ivaz"] * Gamma1["wv"];
        W_["zw"] += 0.5 * Xi3["IA"] * V["vIzA"] * Gamma1["wv"];
        W_["zw"] += 0.5 * Xi3["ia"] * V["izau"] * Gamma1["uw"];
        W_["zw"] += 0.5 * Xi3["IA"] * V["zIuA"] * Gamma1["uw"];
    }

    if (CORRELATION_TERM) {
        W_["zw"] += Tau1["ijwb"] * V["ijzb"];
        W_["zw"] += 2.0 * Tau1["iJwB"] * V["iJzB"];

        temp["klwd"] += Kappa["klwd"] * Eeps2_p["klwd"];
        temp["kLwD"] += Kappa["kLwD"] * Eeps2_p["kLwD"];
        W_["zw"] += temp["klwd"] * V["klzd"];
        W_["zw"] += 2.0 * temp["kLwD"] * V["kLzD"];
        temp.zero();

        W_["zw"] += Tau1["wjab"] * V["zjab"];
        W_["zw"] += 2.0 * Tau1["wJaB"] * V["zJaB"];

        temp["wlcd"] += Kappa["wlcd"] * Eeps2_p["wlcd"];
        temp["wLcD"] += Kappa["wLcD"] * Eeps2_p["wLcD"];
        W_["zw"] += temp["wlcd"] * V["zlcd"];
        W_["zw"] += 2.0 * temp["wLcD"] * V["zLcD"];
        temp.zero(); 
    }
    W_["zw"] += Z["e1,m1"] * V["m1,z,e1,u"] * Gamma1["uw"]; 
    W_["zw"] += Z["E1,M1"] * V["z,M1,u,E1"] * Gamma1["uw"]; 
    W_["zw"] += Z["e1,m1"] * V["m1,u,e1,z"] * Gamma1["uw"]; 
    W_["zw"] += Z["E1,M1"] * V["u,M1,z,E1"] * Gamma1["uw"];
    W_["zw"] += Z["n1,w"] * F["z,n1"];
    W_["zw"] += Z["n1,u"] * V["u,v,n1,z"] * Gamma1["wv"];
    W_["zw"] += Z["N1,U"] * V["v,U,z,N1"] * Gamma1["wv"];
    W_["zw"] += Z["n1,u"] * V["u,z,n1,v"] * Gamma1["wv"];
    W_["zw"] += Z["N1,U"] * V["z,U,v,N1"] * Gamma1["wv"];   
    W_["zw"] -= Z["n1,u"] * H["z,n1"] * Gamma1["uw"];
    W_["zw"] -= Z["n1,u"] * V_N_Alpha["z,n1"] * Gamma1["uw"];
    W_["zw"] -= Z["n1,u"] * V_N_Beta["z,n1"] * Gamma1["uw"];
    W_["zw"] -= 0.5 * Z["n1,u"] * V["x,y,n1,z"] * Gamma2["u,w,x,y"];
    W_["zw"] -= Z["N1,U"] * V["y,X,z,N1"] * Gamma2["w,U,y,X"];
    W_["zw"] -= Z["n1,u"] * V["z,y,n1,v"] * Gamma2["u,v,w,y"];
    W_["zw"] -= Z["n1,u"] * V["z,Y,n1,V"] * Gamma2["u,V,w,Y"];
    W_["zw"] -= Z["N1,U"] * V["z,Y,v,N1"] * Gamma2["v,U,w,Y"];
    W_["zw"] += Z["e1,u"] * H["z,e1"] * Gamma1["uw"];
    W_["zw"] += Z["e1,u"] * V_N_Alpha["z,e1"] * Gamma1["uw"];
    W_["zw"] += Z["e1,u"] * V_N_Beta["z,e1"] * Gamma1["uw"];
    W_["zw"] += 0.5 * Z["e1,u"] * V["x,y,e1,z"] * Gamma2["u,w,x,y"];
    W_["zw"] += Z["E1,U"] * V["y,X,z,E1"] * Gamma2["w,U,y,X"];
    W_["zw"] += Z["e1,u"] * V["z,y,e1,v"] * Gamma2["u,v,w,y"];
    W_["zw"] += Z["e1,u"] * V["z,Y,e1,V"] * Gamma2["u,V,w,Y"];
    W_["zw"] += Z["E1,U"] * V["z,Y,v,E1"] * Gamma2["v,U,w,Y"];
    W_["zw"] += Z["m1,n1"] * V["n1,v,m1,z"] * Gamma1["wv"];
    W_["zw"] += Z["M1,N1"] * V["v,N1,z,M1"] * Gamma1["wv"];
    W_["zw"] += Z["e1,f1"] * V["f1,v,e1,z"] * Gamma1["wv"];
    W_["zw"] += Z["E1,F1"] * V["v,F1,z,E1"] * Gamma1["wv"];
    W_["zw"] += Z["u1,v1"] * V["v1,v,u1,z"] * Gamma1["wv"];
    W_["zw"] += Z["U1,V1"] * V["v,V1,z,U1"] * Gamma1["wv"];
    W_["zw"] += Z["wv"] * F["vz"];

    W_.block("aa")("zw") += 0.25 * x_ci("I") * H.block("aa")("vz") * cc1a_n("Iwv");
    W_.block("aa")("zw") += 0.25 * x_ci("J") * H.block("aa")("vz") * cc1a_r("Jwv");
    W_.block("aa")("zw") += 0.25 * x_ci("I") * H.block("aa")("zu") * cc1a_n("Iuw");
    W_.block("aa")("zw") += 0.25 * x_ci("J") * H.block("aa")("zu") * cc1a_r("Juw");

    W_.block("aa")("zw") += 0.25 * V_N_Alpha.block("aa")("uz") * cc1a_n("Iuw") * x_ci("I"); 
    W_.block("aa")("zw") += 0.25 * V_N_Alpha.block("aa")("uz") * cc1a_r("Juw") * x_ci("J"); 
    W_.block("aa")("zw") += 0.25 * V_N_Beta.block("aa")("uz") * cc1a_n("Iuw") * x_ci("I"); 
    W_.block("aa")("zw") += 0.25 * V_N_Beta.block("aa")("uz") * cc1a_r("Juw") * x_ci("J"); 
    W_.block("aa")("zw") += 0.25 * V_N_Alpha.block("aa")("zv") * cc1a_n("Iwv") * x_ci("I"); 
    W_.block("aa")("zw") += 0.25 * V_N_Alpha.block("aa")("zv") * cc1a_r("Jwv") * x_ci("J"); 
    W_.block("aa")("zw") += 0.25 * V_N_Beta.block("aa")("zv") * cc1a_n("Iwv") * x_ci("I"); 
    W_.block("aa")("zw") += 0.25 * V_N_Beta.block("aa")("zv") * cc1a_r("Jwv") * x_ci("J"); 

    W_.block("aa")("zw") += 0.0625 * x_ci("I") * V.block("aaaa")("zvxy") * cc2aa_n("Iwvxy");
    W_.block("aa")("zw") += 0.1250 * x_ci("I") * V.block("aAaA")("zVxY") * cc2ab_n("IwVxY");
    W_.block("aa")("zw") += 0.0625 * x_ci("I") * V.block("aaaa")("uzxy") * cc2aa_n("Iuwxy");
    W_.block("aa")("zw") += 0.1250 * x_ci("I") * V.block("aAaA")("zUxY") * cc2ab_n("IwUxY");
    W_.block("aa")("zw") += 0.0625 * x_ci("I") * V.block("aaaa")("uvzy") * cc2aa_n("Iuvwy");
    W_.block("aa")("zw") += 0.1250 * x_ci("I") * V.block("aAaA")("uVzY") * cc2ab_n("IuVwY");
    W_.block("aa")("zw") += 0.0625 * x_ci("I") * V.block("aaaa")("uvxz") * cc2aa_n("Iuvxw");
    W_.block("aa")("zw") += 0.1250 * x_ci("I") * V.block("aAaA")("uVzX") * cc2ab_n("IuVwX");

    W_.block("aa")("zw") += 0.0625 * x_ci("J") * V.block("aaaa")("zvxy") * cc2aa_r("Jwvxy");
    W_.block("aa")("zw") += 0.1250 * x_ci("J") * V.block("aAaA")("zVxY") * cc2ab_r("JwVxY");
    W_.block("aa")("zw") += 0.0625 * x_ci("J") * V.block("aaaa")("uzxy") * cc2aa_r("Juwxy");
    W_.block("aa")("zw") += 0.1250 * x_ci("J") * V.block("aAaA")("zUxY") * cc2ab_r("JwUxY");
    W_.block("aa")("zw") += 0.0625 * x_ci("J") * V.block("aaaa")("uvzy") * cc2aa_r("Juvwy");
    W_.block("aa")("zw") += 0.1250 * x_ci("J") * V.block("aAaA")("uVzY") * cc2ab_r("JuVwY");
    W_.block("aa")("zw") += 0.0625 * x_ci("J") * V.block("aaaa")("uvxz") * cc2aa_r("Juvxw");
    W_.block("aa")("zw") += 0.1250 * x_ci("J") * V.block("aAaA")("uVzX") * cc2ab_r("JuVwX");

    // CASSCF reference
    BlockedTensor temp1 = BTF_->build(CoreTensor, "temporal tensor 1", spin_cases({"gg"}));

    W_["mp"] += F["mp"];
    temp1["vp"] = H["vp"];
    temp1["vp"] += V_N_Alpha["vp"];
    temp1["vp"] += V_N_Beta["vp"];
    W_["up"] += temp1["vp"] * Gamma1["uv"];
    W_["up"] += 0.5 * V["xypv"] * Gamma2["uvxy"];
    W_["up"] += V["xYpV"] * Gamma2["uVxY"];

    // Copy alpha-alpha to beta-beta 
    (W_.block("CC")).iterate([&](const std::vector<size_t>& i, double& value) {
        value = W_.block("cc").data()[i[0] * ncore_ + i[1]];
    });
    (W_.block("AA")).iterate([&](const std::vector<size_t>& i, double& value) {
        value = W_.block("aa").data()[i[0] * na_ + i[1]];
    });
    (W_.block("VV")).iterate([&](const std::vector<size_t>& i, double& value) {
        value = W_.block("vv").data()[i[0] * nvirt_ + i[1]];
    });
    (W_.block("CV")).iterate([&](const std::vector<size_t>& i, double& value) {
        value = W_.block("cv").data()[i[0] * nvirt_ + i[1]];
    });
    (W_.block("VC")).iterate([&](const std::vector<size_t>& i, double& value) {
        value = W_.block("vc").data()[i[0] * ncore_ + i[1]];
    });
    (W_.block("CA")).iterate([&](const std::vector<size_t>& i, double& value) {
        value = W_.block("ca").data()[i[0] * na_ + i[1]];
    });
    (W_.block("AC")).iterate([&](const std::vector<size_t>& i, double& value) {
        value = W_.block("ac").data()[i[0] * ncore_ + i[1]];
    });
    (W_.block("AV")).iterate([&](const std::vector<size_t>& i, double& value) {
        value = W_.block("av").data()[i[0] * nvirt_ + i[1]];
    });
    (W_.block("VA")).iterate([&](const std::vector<size_t>& i, double& value) {
        value = W_.block("va").data()[i[0] * na_ + i[1]];
    });
    outfile->Printf("Done");
}

void DSRG_MRPT2::change_zmn_degenerate(BlockedTensor& temp1, 
        BlockedTensor& temp2, BlockedTensor& zmn_d, double coeff) {
    BlockedTensor temp3 = BTF_->build(CoreTensor, "temporal tensor 3", spin_cases({"gggg"}));
    BlockedTensor temp4 = BTF_->build(CoreTensor, "temporal tensor 4", spin_cases({"gggg"}));

    temp3["abml"] = temp1["abmj"] * Gamma1["lj"];
    temp4["adml"] = temp3["abml"] * Eta1["bd"];
    temp3.zero();
    temp3["cdml"] = temp4["adml"] * Eta1["ac"];
    zmn_d["mn"] += coeff * temp3["cdml"] * temp2["cdmnl"];
    temp3.zero();
    temp4.zero();

    temp3["aBmL"] = temp1["aBmJ"] * Gamma1["LJ"];
    temp4["aDmL"] = temp3["aBmL"] * Eta1["BD"];
    temp3.zero();
    temp3["cDmL"] = temp4["aDmL"] * Eta1["ac"];
    zmn_d["mn"] += 2.0 * coeff * temp3["cDmL"] * temp2["cDmnL"];

    temp1.zero();
    temp2.zero();
}

void DSRG_MRPT2::set_z_cc() {   
    BlockedTensor val1 = BTF_->build(CoreTensor, "val1", {"c"});
    BlockedTensor temp = BTF_->build(CoreTensor, "temporal tensor", spin_cases({"hhpp"}));
    BlockedTensor temp_1 = BTF_->build(CoreTensor, "temporal tensor_1", spin_cases({"hhpp"}));

    // core-core diagonal entries
    if (CORRELATION_TERM) {
        val1["m"] += -2 * s * Sigma1["ma"] * F["ma"];
        val1["m"] += -2 * s * DelGam1["xu"] * T2_["muax"] * Sigma1["ma"];
        val1["m"] += -2 * s * DelGam1["XU"] * T2_["mUaX"] * Sigma1["ma"];
    }

    if (CORRELATION_TERM) {
        val1["m"] -= Xi1["ma"] * F["ma"];
        val1["m"] += 2 * s * Xi2["ma"] * F["ma"];
        val1["m"] -= Xi1["ma"] * T2_["muax"] * DelGam1["xu"];
        val1["m"] -= Xi1["ma"] * T2_["mUaX"] * DelGam1["XU"];
        val1["m"] += 2 * s * Xi2["ma"] * T2_["muax"] * DelGam1["xu"];
        val1["m"] += 2 * s * Xi2["ma"] * T2_["mUaX"] * DelGam1["XU"];
    }

    if (CORRELATION_TERM) {
        temp["mjab"] += V["mjab"] * Eeps2["mjab"];
        temp["mJaB"] += V["mJaB"] * Eeps2["mJaB"];
        val1["m"] += 4.0 * s * Tau2["mjab"] * temp["mjab"]; 
        val1["m"] += 8.0 * s * Tau2["mJaB"] * temp["mJaB"]; 
        temp.zero();

        val1["m"] -= 2.0 * T2OverDelta["mjab"] * Tau2["mjab"];
        val1["m"] -= 4.0 * T2OverDelta["mJaB"] * Tau2["mJaB"];

        temp["mlcd"] += V["mlcd"] * Eeps2["mlcd"];
        temp["mLcD"] += V["mLcD"] * Eeps2["mLcD"];
        temp_1["mlcd"] += Kappa["mlcd"] * Delta2["mlcd"];
        temp_1["mLcD"] += Kappa["mLcD"] * Delta2["mLcD"];
        val1["m"] -= 4.0 * s * temp["mlcd"] * temp_1["mlcd"];
        val1["m"] -= 8.0 * s * temp["mLcD"] * temp_1["mLcD"];
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
        zmn["mn"] += Tau1["njab"] * V["mjab"];
        zmn["mn"] += 2.0 * Tau1["nJaB"] * V["mJaB"];

        temp["nlcd"] += Kappa["nlcd"] * Eeps2_p["nlcd"];
        temp["nLcD"] += Kappa["nLcD"] * Eeps2_p["nLcD"];
        zmn["mn"] += temp["nlcd"] * V["mlcd"] ; 
        zmn["mn"] += 2.0 * temp["nLcD"] * V["mLcD"];
        temp.zero(); 

        zmn["mn"] -= Tau1["mjab"] * V["njab"];
        zmn["mn"] -= 2.0 * Tau1["mJaB"] * V["nJaB"];

        temp["mlcd"] += Kappa["mlcd"] * Eeps2_p["mlcd"];
        temp["mLcD"] += Kappa["mLcD"] * Eeps2_p["mLcD"];
        zmn["mn"] -= temp["mlcd"] * V["nlcd"]; 
        zmn["mn"] -= 2.0 * temp["mLcD"] * V["nLcD"]; 
        temp.zero();
    }

    BlockedTensor zmn_d = BTF_->build(CoreTensor, "z{mn} degenerate orbital case", {"cc"});
    BlockedTensor temp1 = BTF_->build(CoreTensor, "temporal tensor 1", {"ppch", "pPcH"});
    BlockedTensor temp2 = BTF_->build(CoreTensor, "temporal tensor 2", {"ppcch", "pPccH"});
    // core-core block entries within degenerate conditions
    if (CORRELATION_TERM) {
        // Alpha
        temp1["abmj"] = V["abmj"] * Delta2["mjab"] * Eeps2["mjab"];
        temp1["aBmJ"] = V["aBmJ"] * Delta2["mJaB"] * Eeps2["mJaB"];
        temp2["cdmnl"] = V["cdnl"] * Eeps2_m1["mlcd"];
        temp2["cDmnL"] = V["cDnL"] * Eeps2_m1["mLcD"];
        change_zmn_degenerate(temp1, temp2, zmn_d, -0.5 * s);

        temp1["abmj"] = V["abmj"] * Eeps2_m1["mjab"];
        temp1["aBmJ"] = V["aBmJ"] * Eeps2_m1["mJaB"];
        temp2["cdmnl"] = V["cdnl"] * Delta2["mlcd"] * Eeps2["mlcd"];
        temp2["cDmnL"] = V["cDnL"] * Delta2["mLcD"] * Eeps2["mLcD"];
        change_zmn_degenerate(temp1, temp2, zmn_d, -0.5 * s);

        temp1["abmj"] = V["abmj"] * Eeps2_p["mjab"];
        temp1["aBmJ"] = V["aBmJ"] * Eeps2_p["mJaB"];
        temp2["cdmnl"] = V["cdnl"] * Eeps2["mlcd"];
        temp2["cDmnL"] = V["cDnL"] * Eeps2["mLcD"];
        change_zmn_degenerate(temp1, temp2, zmn_d, 0.5 * s);

        temp1["abmj"] = V["abmj"] * Eeps2["mjab"];
        temp1["aBmJ"] = V["aBmJ"] * Eeps2["mJaB"];
        temp2["cdmnl"] = V["cdnl"] * Eeps2_p["mlcd"];
        temp2["cDmnL"] = V["cDnL"] * Eeps2_p["mLcD"];
        change_zmn_degenerate(temp1, temp2, zmn_d, 0.5 * s);

        temp1["abmj"] = V["abmj"] * Eeps2_p["mjab"];
        temp1["aBmJ"] = V["aBmJ"] * Eeps2_p["mJaB"];
        temp2["cdmnl"] = V["cdnl"] * Eeps2_m2["mlcd"];
        temp2["cDmnL"] = V["cDnL"] * Eeps2_m2["mLcD"];
        change_zmn_degenerate(temp1, temp2, zmn_d, -0.25);

        temp1["abmj"] = V["abmj"] * Eeps2_m2["mjab"];
        temp1["aBmJ"] = V["aBmJ"] * Eeps2_m2["mJaB"];
        temp2["cdmnl"] = V["cdnl"] * Eeps2_p["mlcd"];
        temp2["cDmnL"] = V["cDnL"] * Eeps2_p["mLcD"];
        change_zmn_degenerate(temp1, temp2, zmn_d, -0.25);
    }

    for (const std::string& block : {"cc", "CC"}) {
        (Z.block(block)).iterate([&](const std::vector<size_t>& i, double& value) {
            if (i[0] == i[1]) { value = val1.block("c").data()[i[0]];}
            else {
                auto dmt = Delta1.block("cc").data()[i[1] * ncore_ + i[0]];
                if (std::fabs(dmt) > 1e-12) { value = zmn.block("cc").data()[i[0] * ncore_ + i[1]] / dmt;}
                else { value = zmn_d.block("cc").data()[i[0] * ncore_ + i[1]];}
            }       
        });
    }  
}

void DSRG_MRPT2::change_zef_degenerate(BlockedTensor& temp1, 
        BlockedTensor& temp2, BlockedTensor& zef_d, double coeff) {
    BlockedTensor temp3 = BTF_->build(CoreTensor, "temporal tensor 3", spin_cases({"gggg"}));
    BlockedTensor temp4 = BTF_->build(CoreTensor, "temporal tensor 4", spin_cases({"gggg"}));

    temp3["fdkj"] = temp1["fdkl"] * Gamma1["lj"];
    temp4["fdij"] = temp3["fdkj"] * Gamma1["ki"];
    temp3.zero();
    temp3["fbij"] = temp4["fdij"] * Eta1["bd"];
    zef_d["ef"] += coeff * temp3["fbij"] * temp2["efbij"];
    temp3.zero();
    temp4.zero();

    temp3["fDkJ"] = temp1["fDkL"] * Gamma1["LJ"];
    temp4["fDiJ"] = temp3["fDkJ"] * Gamma1["ki"];
    temp3.zero();
    temp3["fBiJ"] = temp4["fDiJ"] * Eta1["BD"];
    zef_d["ef"] += 2.0 * coeff * temp3["fBiJ"] * temp2["efBiJ"];

    temp1.zero();
    temp2.zero();
}

void DSRG_MRPT2::set_z_vv() {
    BlockedTensor val2 = BTF_->build(CoreTensor, "val2", {"v"});
    BlockedTensor temp = BTF_->build(CoreTensor, "temporal tensor", spin_cases({"hhpp"}));
    BlockedTensor temp_1 = BTF_->build(CoreTensor, "temporal tensor_1", spin_cases({"hhpp"}));

    // virtual-virtual diagonal entries
    if (CORRELATION_TERM) {
        val2["e"] += 2 * s * Sigma1["ie"] * F["ie"];
        val2["e"] += 2 * s * DelGam1["xu"] * T2_["iuex"] * Sigma1["ie"];
        val2["e"] += 2 * s * DelGam1["XU"] * T2_["iUeX"] * Sigma1["ie"];
    }

    if (CORRELATION_TERM) {
        val2["e"] += Xi1["ie"] * F["ie"];
        val2["e"] -= 2 * s * Xi2["ie"] * F["ie"];
        val2["e"] += Xi1["ie"] * T2_["iuex"] * DelGam1["xu"];
        val2["e"] += Xi1["ie"] * T2_["iUeX"] * DelGam1["XU"];
        val2["e"] -= 2 * s * Xi2["ie"] * T2_["iuex"] * DelGam1["xu"];
        val2["e"] -= 2 * s * Xi2["ie"] * T2_["iUeX"] * DelGam1["XU"];
    }

    if (CORRELATION_TERM) {
        temp["ijeb"] += V["ijeb"] * Eeps2["ijeb"];
        temp["iJeB"] += V["iJeB"] * Eeps2["iJeB"];
        val2["e"] -= 4.0 * s * Tau2["ijeb"] * temp["ijeb"]; 
        val2["e"] -= 8.0 * s * Tau2["iJeB"] * temp["iJeB"]; 
        temp.zero();

        val2["e"] += 2.0 * T2OverDelta["ijeb"] * Tau2["ijeb"];
        val2["e"] += 4.0 * T2OverDelta["iJeB"] * Tau2["iJeB"];

        temp["kled"] += V["kled"] * Eeps2["kled"];
        temp["kLeD"] += V["kLeD"] * Eeps2["kLeD"];
        temp_1["kled"] += Kappa["kled"] * Delta2["kled"];
        temp_1["kLeD"] += Kappa["kLeD"] * Delta2["kLeD"];
        val2["e"] += 4.0 * s * temp["kled"] * temp_1["kled"];
        val2["e"] += 8.0 * s * temp["kLeD"] * temp_1["kLeD"];
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
        zef["ef"] += Tau1["ijfb"] * V["ijeb"];
        zef["ef"] += 2.0 * Tau1["iJfB"] * V["iJeB"];

        temp["klfd"] += Kappa["klfd"] * Eeps2_p["klfd"];
        temp["kLfD"] += Kappa["kLfD"] * Eeps2_p["kLfD"];
        zef["ef"] += temp["klfd"] * V["kled"] ; 
        zef["ef"] += 2.0 * temp["kLfD"] * V["kLeD"];
        temp.zero(); 

        zef["ef"] -= Tau1["ijeb"] * V["ijfb"];
        zef["ef"] -= 2.0 * Tau1["iJeB"] * V["iJfB"];

        temp["kled"] += Kappa["kled"] * Eeps2_p["kled"];
        temp["kLeD"] += Kappa["kLeD"] * Eeps2_p["kLeD"];
        zef["ef"] -= temp["kled"] * V["klfd"]; 
        zef["ef"] -= 2.0 * temp["kLeD"] * V["kLfD"]; 
        temp.zero();
    }

    BlockedTensor zef_d = BTF_->build(CoreTensor, "z{ef} degenerate orbital case", {"vv"});
    BlockedTensor temp1 = BTF_->build(CoreTensor, "temporal tensor 1", {"vphh", "vPhH"});
    BlockedTensor temp2 = BTF_->build(CoreTensor, "temporal tensor 2", {"vvphh", "vvPhH"});
    // virtual-virtual block entries within degenerate conditions
    if (CORRELATION_TERM) {
        temp1["fdkl"] = V["fdkl"] * Eeps2_m1["klfd"];
        temp1["fDkL"] = V["fDkL"] * Eeps2_m1["kLfD"];
        temp2["efbij"] = V["ebij"] * Delta2["ijfb"] * Eeps2["ijfb"];
        temp2["efBiJ"] = V["eBiJ"] * Delta2["iJfB"] * Eeps2["iJfB"];
        change_zef_degenerate(temp1, temp2, zef_d, 0.5 * s);

        temp1["fdkl"] = V["fdkl"] * Delta2["klfd"] * Eeps2["klfd"];
        temp1["fDkL"] = V["fDkL"] * Delta2["kLfD"] * Eeps2["kLfD"];
        temp2["efbij"] = V["ebij"] * Eeps2_m1["ijfb"];
        temp2["efBiJ"] = V["eBiJ"] * Eeps2_m1["iJfB"];
        change_zef_degenerate(temp1, temp2, zef_d, 0.5 * s);

        temp1["fdkl"] = V["fdkl"] * Eeps2_p["klfd"];
        temp1["fDkL"] = V["fDkL"] * Eeps2_p["kLfD"];
        temp2["efbij"] = V["ebij"] * Eeps2["ijfb"];
        temp2["efBiJ"] = V["eBiJ"] * Eeps2["iJfB"];
        change_zef_degenerate(temp1, temp2, zef_d, -0.5 * s);

        temp1["fdkl"] = V["fdkl"] * Eeps2["klfd"];
        temp1["fDkL"] = V["fDkL"] * Eeps2["kLfD"];
        temp2["efbij"] = V["ebij"] * Eeps2_p["ijfb"];
        temp2["efBiJ"] = V["eBiJ"] * Eeps2_p["iJfB"];
        change_zef_degenerate(temp1, temp2, zef_d, -0.5 * s);

        temp1["fdkl"] = V["fdkl"] * Eeps2_p["klfd"];
        temp1["fDkL"] = V["fDkL"] * Eeps2_p["kLfD"];
        temp2["efbij"] = V["ebij"] * Eeps2_m2["ijfb"];
        temp2["efBiJ"] = V["eBiJ"] * Eeps2_m2["iJfB"];
        change_zef_degenerate(temp1, temp2, zef_d, 0.25);

        temp1["fdkl"] = V["fdkl"] * Eeps2_m2["klfd"];
        temp1["fDkL"] = V["fDkL"] * Eeps2_m2["kLfD"];
        temp2["efbij"] = V["ebij"] * Eeps2_p["ijfb"];
        temp2["efBiJ"] = V["eBiJ"] * Eeps2_p["iJfB"];
        change_zef_degenerate(temp1, temp2, zef_d, 0.25);
    }

    for (const std::string& block : {"vv", "VV"}) {
        (Z.block(block)).iterate([&](const std::vector<size_t>& i, double& value) {
            if (i[0] == i[1]) { value = val2.block("v").data()[i[0]];}
            else {
                auto dmt = Delta1.block("vv").data()[i[1] * nvirt_ + i[0]];
                if (std::fabs(dmt) > 1e-12) { value = zef.block("vv").data()[i[0] * nvirt_ + i[1]] / dmt;}
                else { value = zef_d.block("vv").data()[i[0] * nvirt_ + i[1]];}
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
        val3["w"] += -2 * s * Sigma1["wa"] * F["wa"];
        val3["w"] += -2 * s * DelGam1["xu"] * T2_["wuax"] * Sigma1["wa"];
        val3["w"] += -2 * s * DelGam1["XU"] * T2_["wUaX"] * Sigma1["wa"];
        val3["w"] +=  2 * s * Sigma1["iw"] * F["iw"];
        val3["w"] +=  2 * s * DelGam1["xu"] * T2_["iuwx"] * Sigma1["iw"];
        val3["w"] +=  2 * s * DelGam1["XU"] * T2_["iUwX"] * Sigma1["iw"];

        val3["w"] += Sigma2["ia"] * T2_["iuaw"] * Gamma1["wu"];
        val3["w"] += Sigma2["IA"] * T2_["uIwA"] * Gamma1["wu"];
        val3["w"] -= Sigma2["ia"] * T2_["iwax"] * Gamma1["xw"];
        val3["w"] -= Sigma2["IA"] * T2_["wIxA"] * Gamma1["xw"];
    }

    if (CORRELATION_TERM) {
        val3["w"] -= Xi1["wa"] * F["wa"];
        val3["w"] += 2 * s * Xi2["wa"] * F["wa"];
        val3["w"] -= Xi1["wa"] * T2_["wuax"] * DelGam1["xu"];
        val3["w"] -= Xi1["wa"] * T2_["wUaX"] * DelGam1["XU"];
        val3["w"] += 2 * s * Xi2["wa"] * T2_["wuax"] * DelGam1["xu"];
        val3["w"] += 2 * s * Xi2["wa"] * T2_["wUaX"] * DelGam1["XU"];

        val3["w"] += Xi1["iw"] * F["iw"];
        val3["w"] -= 2 * s * Xi2["iw"] * F["iw"];
        val3["w"] += Xi1["iw"] * T2_["iuwx"] * DelGam1["xu"];
        val3["w"] += Xi1["iw"] * T2_["iUwX"] * DelGam1["XU"];
        val3["w"] -= 2 * s * Xi2["iw"] * T2_["iuwx"] * DelGam1["xu"];
        val3["w"] -= 2 * s * Xi2["iw"] * T2_["iUwX"] * DelGam1["XU"];

        val3["w"] += Xi3["ia"] * T2_["iuaw"] * Gamma1["wu"];
        val3["w"] += Xi3["IA"] * T2_["uIwA"] * Gamma1["wu"];
        val3["w"] -= Xi3["ia"] * T2_["iwax"] * Gamma1["xw"];
        val3["w"] -= Xi3["IA"] * T2_["wIxA"] * Gamma1["xw"];
    }

    if (CORRELATION_TERM) {
        temp["ujab"] += V["ujab"] * Eeps2["ujab"];
        temp["uJaB"] += V["uJaB"] * Eeps2["uJaB"];
        val3["u"] += 4.0 * s * Tau2["ujab"] * temp["ujab"]; 
        val3["u"] += 8.0 * s * Tau2["uJaB"] * temp["uJaB"]; 
        temp.zero();

        val3["u"] -= 2.0 * T2OverDelta["ujab"] * Tau2["ujab"];
        val3["u"] -= 4.0 * T2OverDelta["uJaB"] * Tau2["uJaB"];

        temp["ulcd"] += V["ulcd"] * Eeps2["ulcd"];
        temp["uLcD"] += V["uLcD"] * Eeps2["uLcD"];
        temp_1["ulcd"] += Kappa["ulcd"] * Delta2["ulcd"];
        temp_1["uLcD"] += Kappa["uLcD"] * Delta2["uLcD"];
        val3["u"] -= 4.0 * s * temp["ulcd"] * temp_1["ulcd"];
        val3["u"] -= 8.0 * s * temp["uLcD"] * temp_1["uLcD"];
        temp.zero();
        temp_1.zero();

        temp["ijub"] += V["ijub"] * Eeps2["ijub"];
        temp["iJuB"] += V["iJuB"] * Eeps2["iJuB"];
        val3["u"] -= 4.0 * s * Tau2["ijub"] * temp["ijub"]; 
        val3["u"] -= 8.0 * s * Tau2["iJuB"] * temp["iJuB"]; 
        temp.zero();

        val3["u"] += 2.0 * T2OverDelta["ijub"] * Tau2["ijub"];
        val3["u"] += 4.0 * T2OverDelta["iJuB"] * Tau2["iJuB"];

        temp["klud"] += V["klud"] * Eeps2["klud"];
        temp["kLuD"] += V["kLuD"] * Eeps2["kLuD"];
        temp_1["klud"] += Kappa["klud"] * Delta2["klud"];
        temp_1["kLuD"] += Kappa["kLuD"] * Delta2["kLuD"];
        val3["u"] += 4.0 * s * temp["klud"] * temp_1["klud"];
        val3["u"] += 8.0 * s * temp["kLuD"] * temp_1["kLuD"];
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
        Z_b["em"] += 0.5 * Sigma3["ia"] * V["imae"];
        Z_b["em"] += 0.5 * Sigma3["IA"] * V["mIeA"];
        Z_b["em"] -= 0.5 * Sigma3["ie"] * F["im"];
    }

    if (CORRELATION_TERM) {
        Z_b["em"] += 0.5 * Xi3["ma"] * F["ea"];
        Z_b["em"] += 0.5 * Xi3["ia"] * V["ieam"];
        Z_b["em"] += 0.5 * Xi3["IA"] * V["eImA"];
        Z_b["em"] += 0.5 * Xi3["ia"] * V["imae"];
        Z_b["em"] += 0.5 * Xi3["IA"] * V["mIeA"];
        Z_b["em"] -= 0.5 * Xi3["ie"] * F["im"];
    }

    if (CORRELATION_TERM) {
        Z_b["em"] += Tau1["mjab"] * V["ejab"];
        Z_b["em"] += 2.0 * Tau1["mJaB"] * V["eJaB"];

        temp["mlcd"] += Kappa["mlcd"] * Eeps2_p["mlcd"];
        temp["mLcD"] += Kappa["mLcD"] * Eeps2_p["mLcD"];
        Z_b["em"] += temp["mlcd"] * V["elcd"];
        Z_b["em"] += 2.0 * temp["mLcD"] * V["eLcD"];
        temp.zero();

        Z_b["em"] -= Tau1["ijeb"] * V["ijmb"];
        Z_b["em"] -= 2.0 * Tau1["iJeB"] * V["iJmB"];

        temp["kled"] += Kappa["kled"] * Eeps2_p["kled"];
        temp["kLeD"] += Kappa["kLeD"] * Eeps2_p["kLeD"];
        Z_b["em"] -= temp["kled"] * V["klmd"];
        Z_b["em"] -= 2.0 * temp["kLeD"] * V["kLmD"];
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
        Z_b["wz"] += 0.5 * Sigma3["ia"] * V["ivaw"] * Gamma1["zv"];
        Z_b["wz"] += 0.5 * Sigma3["IA"] * V["vIwA"] * Gamma1["zv"];
        Z_b["wz"] += 0.5 * Sigma3["ia"] * V["iwau"] * Gamma1["uz"];
        Z_b["wz"] += 0.5 * Sigma3["IA"] * V["wIuA"] * Gamma1["uz"];

        Z_b["wz"] -= 0.5 * Sigma3["wa"] * F["za"];
        Z_b["wz"] -= 0.5 * Sigma3["iw"] * F["iz"];
        Z_b["wz"] -= 0.5 * Sigma3["ia"] * V["ivaz"] * Gamma1["wv"];
        Z_b["wz"] -= 0.5 * Sigma3["IA"] * V["vIzA"] * Gamma1["wv"];
        Z_b["wz"] -= 0.5 * Sigma3["ia"] * V["izau"] * Gamma1["uw"];
        Z_b["wz"] -= 0.5 * Sigma3["IA"] * V["zIuA"] * Gamma1["uw"];
    }

    if (CORRELATION_TERM) {
        Z_b["wz"] += 0.5 * Xi3["za"] * F["wa"];
        Z_b["wz"] += 0.5 * Xi3["iz"] * F["iw"];
        Z_b["wz"] += 0.5 * Xi3["ia"] * V["ivaw"] * Gamma1["zv"];
        Z_b["wz"] += 0.5 * Xi3["IA"] * V["vIwA"] * Gamma1["zv"];
        Z_b["wz"] += 0.5 * Xi3["ia"] * V["iwau"] * Gamma1["uz"];
        Z_b["wz"] += 0.5 * Xi3["IA"] * V["wIuA"] * Gamma1["uz"];

        Z_b["wz"] -= 0.5 * Xi3["wa"] * F["za"];
        Z_b["wz"] -= 0.5 * Xi3["iw"] * F["iz"];
        Z_b["wz"] -= 0.5 * Xi3["ia"] * V["ivaz"] * Gamma1["wv"];
        Z_b["wz"] -= 0.5 * Xi3["IA"] * V["vIzA"] * Gamma1["wv"];
        Z_b["wz"] -= 0.5 * Xi3["ia"] * V["izau"] * Gamma1["uw"];
        Z_b["wz"] -= 0.5 * Xi3["IA"] * V["zIuA"] * Gamma1["uw"];
    }

    if (CORRELATION_TERM) {
        Z_b["wz"] += Tau1["ijzb"] * V["ijwb"];
        Z_b["wz"] += 2.0 * Tau1["iJzB"] * V["iJwB"];

        temp["klzd"] += Kappa["klzd"] * Eeps2_p["klzd"];
        temp["kLzD"] += Kappa["kLzD"] * Eeps2_p["kLzD"];
        Z_b["wz"] += temp["klzd"] * V["klwd"];
        Z_b["wz"] += 2.0 * temp["kLzD"] * V["kLwD"];
        temp.zero();

        Z_b["wz"] += Tau1["zjab"] * V["wjab"];
        Z_b["wz"] += 2.0 * Tau1["zJaB"] * V["wJaB"];

        temp["zlcd"] += Kappa["zlcd"] * Eeps2_p["zlcd"];
        temp["zLcD"] += Kappa["zLcD"] * Eeps2_p["zLcD"];
        Z_b["wz"] += temp["zlcd"] * V["wlcd"];
        Z_b["wz"] += 2.0 * temp["zLcD"] * V["wLcD"];
        temp.zero();

        Z_b["wz"] -= Tau1["ijwb"] * V["ijzb"];
        Z_b["wz"] -= 2.0 * Tau1["iJwB"] * V["iJzB"];

        temp["klwd"] += Kappa["klwd"] * Eeps2_p["klwd"];
        temp["kLwD"] += Kappa["kLwD"] * Eeps2_p["kLwD"];
        Z_b["wz"] -= temp["klwd"] * V["klzd"];
        Z_b["wz"] -= 2.0 * temp["kLwD"] * V["kLzD"];
        temp.zero();

        Z_b["wz"] -= Tau1["wjab"] * V["zjab"];
        Z_b["wz"] -= 2.0 * Tau1["wJaB"] * V["zJaB"];

        temp["wlcd"] += Kappa["wlcd"] * Eeps2_p["wlcd"];
        temp["wLcD"] += Kappa["wLcD"] * Eeps2_p["wLcD"];
        Z_b["wz"] -= temp["wlcd"] * V["zlcd"];
        Z_b["wz"] -= 2.0 * temp["wLcD"] * V["zLcD"];
        temp.zero();
    }
    Z_b["wz"] += Z["m1,n1"] * V["n1,v,m1,w"] * Gamma1["zv"];
    Z_b["wz"] += Z["M1,N1"] * V["v,N1,w,M1"] * Gamma1["zv"];

    Z_b["wz"] += Z["e1,f1"] * V["f1,v,e1,w"] * Gamma1["zv"];
    Z_b["wz"] += Z["E1,F1"] * V["v,F1,w,E1"] * Gamma1["zv"];

    Z_b["wz"] -= Z["m1,n1"] * V["n1,v,m1,z"] * Gamma1["wv"];
    Z_b["wz"] -= Z["M1,N1"] * V["v,N1,z,M1"] * Gamma1["wv"];

    Z_b["wz"] -= Z["e1,f1"] * V["f1,v,e1,z"] * Gamma1["wv"];
    Z_b["wz"] -= Z["E1,F1"] * V["v,F1,z,E1"] * Gamma1["wv"];

    //NOTICE: constant b for z{virtual-active}
    if (CORRELATION_TERM) {
        Z_b["ew"] += 0.5 * Sigma3["wa"] * F["ea"];
        Z_b["ew"] += 0.5 * Sigma3["iw"] * F["ie"];
        Z_b["ew"] += 0.5 * Sigma3["ia"] * V["ivae"] * Gamma1["wv"];
        Z_b["ew"] += 0.5 * Sigma3["IA"] * V["vIeA"] * Gamma1["wv"];
        Z_b["ew"] += 0.5 * Sigma3["ia"] * V["ieau"] * Gamma1["uw"];
        Z_b["ew"] += 0.5 * Sigma3["IA"] * V["eIuA"] * Gamma1["uw"];
        Z_b["ew"] -= 0.5 * Sigma3["ie"] * F["iw"];
    }

    if (CORRELATION_TERM) {
        Z_b["ew"] += 0.5 * Xi3["wa"] * F["ea"];
        Z_b["ew"] += 0.5 * Xi3["iw"] * F["ie"];
        Z_b["ew"] += 0.5 * Xi3["ia"] * V["ivae"] * Gamma1["wv"];
        Z_b["ew"] += 0.5 * Xi3["IA"] * V["vIeA"] * Gamma1["wv"];
        Z_b["ew"] += 0.5 * Xi3["ia"] * V["ieau"] * Gamma1["uw"];
        Z_b["ew"] += 0.5 * Xi3["IA"] * V["eIuA"] * Gamma1["uw"];
        Z_b["ew"] -= 0.5 * Xi3["ie"] * F["iw"];
    }

    if (CORRELATION_TERM) {
        Z_b["ew"] += Tau1["ijwb"] * V["ijeb"];
        Z_b["ew"] += 2.0 * Tau1["iJwB"] * V["iJeB"];

        temp["klwd"] += Kappa["klwd"] * Eeps2_p["klwd"];
        temp["kLwD"] += Kappa["kLwD"] * Eeps2_p["kLwD"];
        Z_b["ew"] += temp["klwd"] * V["kled"];
        Z_b["ew"] += 2.0 * temp["kLwD"] * V["kLeD"];
        temp.zero();

        Z_b["ew"] += Tau1["wjab"] * V["ejab"];
        Z_b["ew"] += 2.0 * Tau1["wJaB"] * V["eJaB"];

        temp["wlcd"] += Kappa["wlcd"] * Eeps2_p["wlcd"];
        temp["wLcD"] += Kappa["wLcD"] * Eeps2_p["wLcD"];
        Z_b["ew"] += temp["wlcd"] * V["elcd"];
        Z_b["ew"] += 2.0 * temp["wLcD"] * V["eLcD"];
        temp.zero();

        Z_b["ew"] -= Tau1["ijeb"] * V["ijwb"];
        Z_b["ew"] -= 2.0 * Tau1["iJeB"] * V["iJwB"];

        temp["kled"] += Kappa["kled"] * Eeps2_p["kled"];
        temp["kLeD"] += Kappa["kLeD"] * Eeps2_p["kLeD"];
        Z_b["ew"] -= temp["kled"] * V["klwd"];
        Z_b["ew"] -= 2.0 * temp["kLeD"] * V["kLwD"];
        temp.zero();
    }
    Z_b["ew"] -= Z["e,f1"] * F["f1,w"];
    Z_b["ew"] += Z["m1,n1"] * V["n1,v,m1,e"] * Gamma1["wv"];
    Z_b["ew"] += Z["M1,N1"] * V["v,N1,e,M1"] * Gamma1["wv"];
    Z_b["ew"] += Z["e1,f1"] * V["f1,v,e1,e"] * Gamma1["wv"];
    Z_b["ew"] += Z["E1,F1"] * V["v,F1,e,E1"] * Gamma1["wv"];

    //NOTICE: constant b for z{core-active}
    if (CORRELATION_TERM) {
        Z_b["mw"] += 0.5 * Sigma3["wa"] * F["ma"];
        Z_b["mw"] += 0.5 * Sigma3["iw"] * F["im"];
        Z_b["mw"] += 0.5 * Sigma3["ia"] * V["ivam"] * Gamma1["wv"];
        Z_b["mw"] += 0.5 * Sigma3["IA"] * V["vImA"] * Gamma1["wv"];
        Z_b["mw"] += 0.5 * Sigma3["ia"] * V["imau"] * Gamma1["uw"];
        Z_b["mw"] += 0.5 * Sigma3["IA"] * V["mIuA"] * Gamma1["uw"];

        Z_b["mw"] -= 0.5 * Sigma3["ma"] * F["wa"];
        Z_b["mw"] -= 0.5 * Sigma3["ia"] * V["iwam"];
        Z_b["mw"] -= 0.5 * Sigma3["IA"] * V["wImA"];
        Z_b["mw"] -= 0.5 * Sigma3["ia"] * V["imaw"];
        Z_b["mw"] -= 0.5 * Sigma3["IA"] * V["mIwA"];
    }

    if (CORRELATION_TERM) {
        Z_b["mw"] += 0.5 * Xi3["wa"] * F["ma"];
        Z_b["mw"] += 0.5 * Xi3["iw"] * F["im"];
        Z_b["mw"] += 0.5 * Xi3["ia"] * V["ivam"] * Gamma1["wv"];
        Z_b["mw"] += 0.5 * Xi3["IA"] * V["vImA"] * Gamma1["wv"];
        Z_b["mw"] += 0.5 * Xi3["ia"] * V["imau"] * Gamma1["uw"];
        Z_b["mw"] += 0.5 * Xi3["IA"] * V["mIuA"] * Gamma1["uw"];

        Z_b["mw"] -= 0.5 * Xi3["ma"] * F["wa"];
        Z_b["mw"] -= 0.5 * Xi3["ia"] * V["iwam"];
        Z_b["mw"] -= 0.5 * Xi3["IA"] * V["wImA"];
        Z_b["mw"] -= 0.5 * Xi3["ia"] * V["imaw"];
        Z_b["mw"] -= 0.5 * Xi3["IA"] * V["mIwA"];
    }

    if (CORRELATION_TERM) {
        Z_b["mw"] += Tau1["ijwb"] * V["ijmb"];
        Z_b["mw"] += 2.0 * Tau1["iJwB"] * V["iJmB"];

        temp["klwd"] += Kappa["klwd"] * Eeps2_p["klwd"];
        temp["kLwD"] += Kappa["kLwD"] * Eeps2_p["kLwD"];
        Z_b["mw"] += temp["klwd"] * V["klmd"];
        Z_b["mw"] += 2.0 * temp["kLwD"] * V["kLmD"];
        temp.zero();

        Z_b["mw"] += Tau1["wjab"] * V["mjab"];
        Z_b["mw"] += 2.0 * Tau1["wJaB"] * V["mJaB"];

        temp["wlcd"] += Kappa["wlcd"] * Eeps2_p["wlcd"];
        temp["wLcD"] += Kappa["wLcD"] * Eeps2_p["wLcD"];
        Z_b["mw"] += temp["wlcd"] * V["mlcd"];
        Z_b["mw"] += 2.0 * temp["wLcD"] * V["mLcD"];
        temp.zero();

        Z_b["mw"] -= Tau1["mjab"] * V["wjab"];
        Z_b["mw"] -= 2.0 * Tau1["mJaB"] * V["wJaB"];

        temp["mlcd"] += Kappa["mlcd"] * Eeps2_p["mlcd"];
        temp["mLcD"] += Kappa["mLcD"] * Eeps2_p["mLcD"];
        Z_b["mw"] -= temp["mlcd"] * V["wlcd"];
        Z_b["mw"] -= 2.0 * temp["mLcD"] * V["wLcD"];
        temp.zero();
    }
    Z_b["mw"] += Z["m1,n1"] * V["n1,v,m1,m"] * Gamma1["wv"];
    Z_b["mw"] += Z["M1,N1"] * V["v,N1,m,M1"] * Gamma1["wv"];

    Z_b["mw"] += Z["e1,f1"] * V["f1,v,e1,m"] * Gamma1["wv"];
    Z_b["mw"] += Z["E1,F1"] * V["v,F1,m,E1"] * Gamma1["wv"];

    Z_b["mw"] -= Z["m,n1"] * F["n1,w"];

    Z_b["mw"] -= Z["m1,n1"] * V["n1,w,m1,m"];
    Z_b["mw"] -= Z["M1,N1"] * V["w,N1,m,M1"];

    Z_b["mw"] -= Z["e1,f"] * V["f,w,e1,m"];
    Z_b["mw"] -= Z["E1,F"] * V["w,F,m,E1"];
}

void DSRG_MRPT2::solve_z() {
    set_b();

    int dim_vc = nvirt_ * ncore_,
        dim_ca = ncore_ * na_,
        dim_va = nvirt_ * na_,
        dim_aa = na_ * (na_ - 1) / 2,
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
        {"vc", ncore_}, {"VC", ncore_}, {"ca", na_}, {"CA", na_},
        {"va", na_}, {"VA", na_}, {"aa", 0}, {"AA", 0}
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
    temp2["mw"] += temp3["u1,v1"] * V["v1,v,u1,m"] * Gamma1["w,v"];
    temp2["mw"] += temp3["U1,V1"] * V["v,V1,m,U1"] * Gamma1["w,v"];
    temp2["mw"] += temp3["wv"] * F["vm"];
    temp2["mw"] -= temp3["uv"] * V["vwum"];
    temp2["mw"] -= temp3["UV"] * V["wVmU"];

    // VIRTUAL-ACTIVE
    temp2["ew"] += Z_b["ew"];
    temp2["ew"] += temp3["u1,v1"] * V["v1,v,u1,e"] * Gamma1["wv"];
    temp2["ew"] += temp3["U1,V1"] * V["v,V1,e,U1"] * Gamma1["wv"];
    temp2["ew"] += temp3["wv"] * F["ve"];

    // ACTIVE-ACTIVE
    temp2["wz"] += Z_b["wz"];
    temp2["wz"] += temp3["v1,u1"] * V["u1,v,v1,w"] * Gamma1["zv"];
    temp2["wz"] += temp3["V1,U1"] * V["v,U1,w,V1"] * Gamma1["zv"];
    temp2["wz"] -= temp3["v1,u1"] * V["u1,v,v1,z"] * Gamma1["wv"];
    temp2["wz"] -= temp3["V1,U1"] * V["v,U1,z,V1"] * Gamma1["wv"];

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

    //NOTICE:A
    BlockedTensor temp1 = BTF_->build(CoreTensor, "temporal tensor 1", {"gggg","ggGG"});

    // VIRTUAL-CORE
    temp1["e,m,e1,m1"] += Delta1["m1,e1"] * I["e1,e"] * I["m1,m"];

    temp1["e,m,e1,m1"] -= V["m1,m,e1,e"];
    temp1["e,m,E1,M1"] -= V["m,M1,e,E1"];
    temp1["e,m,e1,m1"] -= V["m1,e,e1,m"];
    temp1["e,m,E1,M1"] -= V["e,M1,m,E1"];

    temp1["e,m,m1,u"] -= F["ue"] * I["m,m1"];

    temp1["e,m,n1,u"] -= V["u,e,n1,m"];
    temp1["e,m,N1,U"] -= V["e,U,m,N1"];
    temp1["e,m,n1,u"] -= V["u,m,n1,e"];
    temp1["e,m,N1,U"] -= V["m,U,e,N1"];

    temp1["e,m,n1,u"] += Gamma1["uv"] * V["v,e,n1,m"];
    temp1["e,m,N1,U"] += Gamma1["UV"] * V["e,V,m,N1"];
    temp1["e,m,n1,u"] += Gamma1["uv"] * V["v,m,n1,e"];
    temp1["e,m,N1,U"] += Gamma1["UV"] * V["m,V,e,N1"];

    temp1["e,m,e1,u"] -= Gamma1["uv"] * V["v,m,e1,e"];
    temp1["e,m,E1,U"] -= Gamma1["UV"] * V["m,V,e,E1"];
    temp1["e,m,e1,u"] -= Gamma1["uv"] * V["v,e,e1,m"];
    temp1["e,m,E1,U"] -= Gamma1["UV"] * V["e,V,m,E1"];

    temp1["e,m,e1,u"] += F["um"] * I["e,e1"];

    temp1["e,m,u,v"] -= V["veum"];
    temp1["e,m,U,V"] -= V["eVmU"];

    // CORE-ACTIVE
    temp1["m,w,e1,m1"] += F["w,e1"] * I["m,m1"];
    temp1["m,w,e1,m1"] += V["m1,m,e1,w"];
    temp1["m,w,E1,M1"] += V["m,M1,w,E1"];
    temp1["m,w,e1,m1"] += V["m1,w,e1,m"];
    temp1["m,w,E1,M1"] += V["w,M1,m,E1"];

    temp1["m,w,m1,u"] += F["uw"] * I["m1,m"];
    temp1["m,w,m1,u"] -= H["vw"] * Gamma1["uv"] * I["m1,m"];
    temp1["m,w,m1,u"] -= V["v,n1,w,n"] * Gamma1["uv"] * I["n,n1"] * I["m,m1"];
    temp1["m,w,m1,u"] -= V["v,N1,w,N"] * Gamma1["uv"] * I["N,N1"] * I["m,m1"];
    temp1["m,w,m1,u"] -= 0.5 * V["xywv"] * Gamma2["uvxy"] * I["m,m1"];
    temp1["m,w,m1,u"] -= V["xYwV"] * Gamma2["uVxY"] * I["m,m1"];

    temp1["m,w,n1,u"] += V["u,w,n1,m"];
    temp1["m,w,N1,U"] += V["w,U,m,N1"];
    temp1["m,w,n1,u"] += V["u,m,n1,w"];
    temp1["m,w,N1,U"] += V["m,U,w,N1"];

    temp1["m,w,n1,u"] -= Gamma1["uv"] * V["v,w,n1,m"];
    temp1["m,w,N1,U"] -= Gamma1["UV"] * V["w,V,m,N1"];
    temp1["m,w,n1,u"] -= Gamma1["uv"] * V["v,m,n1,w"];
    temp1["m,w,N1,U"] -= Gamma1["UV"] * V["m,V,w,N1"];

    temp1["m,w,e1,u"] += Gamma1["uv"] * V["v,m,e1,w"];
    temp1["m,w,E1,U"] += Gamma1["UV"] * V["m,V,w,E1"];
    temp1["m,w,e1,u"] += Gamma1["uv"] * V["v,w,e1,m"];
    temp1["m,w,E1,U"] += Gamma1["UV"] * V["w,V,m,E1"];

    temp1["m,w,u,v"] += V["vwum"];
    temp1["m,w,U,V"] += V["wVmU"];

    temp1["m,w,e1,m1"] -= V["m1,m,e1,u"] * Gamma1["uw"];
    temp1["m,w,E1,M1"] -= V["m,M1,u,E1"] * Gamma1["uw"];
    temp1["m,w,e1,m1"] -= V["m1,u,e1,m"] * Gamma1["uw"];
    temp1["m,w,E1,M1"] -= V["u,M1,m,E1"] * Gamma1["uw"];

    temp1["m,w,n1,w1"] -= F["m,n1"] * I["w,w1"];

    temp1["m,w,n1,u"] -= V["u,v,n1,m"] * Gamma1["wv"];
    temp1["m,w,N1,U"] -= V["v,U,m,N1"] * Gamma1["wv"];
    temp1["m,w,n1,u"] -= V["u,m,n1,v"] * Gamma1["wv"];
    temp1["m,w,N1,U"] -= V["m,U,v,N1"] * Gamma1["wv"];

    temp1["m,w,n1,u"] += H["m,n1"] * Gamma1["uw"];
    temp1["m,w,n1,u"] += V["m,m2,n1,n2"] * Gamma1["uw"] * I["m2,n2"];
    temp1["m,w,n1,u"] += V["m,M2,n1,N2"] * Gamma1["uw"] * I["M2,N2"];
    temp1["m,w,n1,u"] += 0.5 * V["x,y,n1,m"] * Gamma2["u,w,x,y"];
    temp1["m,w,N1,U"] += V["y,X,m,N1"] * Gamma2["w,U,y,X"];
    temp1["m,w,n1,u"] += V["m,y,n1,v"] * Gamma2["u,v,w,y"];
    temp1["m,w,n1,u"] += V["m,Y,n1,V"] * Gamma2["u,V,w,Y"];
    temp1["m,w,N1,U"] += V["m,Y,v,N1"] * Gamma2["v,U,w,Y"];

    temp1["m,w,e1,u"] -= H["m,e1"] * Gamma1["uw"];
    temp1["m,w,e1,u"] -= V["m,m2,e1,n2"] * Gamma1["uw"] * I["m2,n2"];
    temp1["m,w,e1,u"] -= V["m,M2,e1,N2"] * Gamma1["uw"] * I["M2,N2"];
    temp1["m,w,e1,u"] -= 0.5 * V["x,y,e1,m"] * Gamma2["u,w,x,y"];
    temp1["m,w,E1,U"] -= V["y,X,m,E1"] * Gamma2["w,U,y,X"];
    temp1["m,w,e1,u"] -= V["m,y,e1,v"] * Gamma2["u,v,w,y"];
    temp1["m,w,e1,u"] -= V["m,Y,e1,V"] * Gamma2["u,V,w,Y"];
    temp1["m,w,E1,U"] -= V["m,Y,v,E1"] * Gamma2["v,U,w,Y"];

    temp1["m,w,u1,v1"] -= V["v1,v,u1,m"] * Gamma1["wv"];
    temp1["m,w,U1,V1"] -= V["v,V1,m,U1"] * Gamma1["wv"];

    temp1["m,w,w1,v"] -= F["vm"] * I["w,w1"];

    // VIRTUAL-ACTIVE
    temp1["e,w,e1,m1"] += F["m1,w"] * I["e,e1"];

    temp1["e,w,e1,u"] += H["vw"] * Gamma1["uv"] * I["e,e1"];
    temp1["e,w,e1,u"] += V["v,m,w,m1"] * Gamma1["uv"] * I["m,m1"] * I["e,e1"];
    temp1["e,w,e1,u"] += V["v,M,w,M1"] * Gamma1["uv"] * I["M,M1"] * I["e,e1"];
    temp1["e,w,e1,u"] += 0.5 * V["xywv"] * Gamma2["uvxy"] * I["e,e1"];
    temp1["e,w,e1,u"] += V["xYwV"] * Gamma2["uVxY"] * I["e,e1"];

    temp1["e,w,e1,m1"] -= V["m1,e,e1,u"] * Gamma1["uw"];
    temp1["e,w,E1,M1"] -= V["e,M1,u,E1"] * Gamma1["uw"];
    temp1["e,w,e1,m1"] -= V["m1,u,e1,e"] * Gamma1["uw"];
    temp1["e,w,E1,M1"] -= V["u,M1,e,E1"] * Gamma1["uw"];

    temp1["e,w,n1,u"] -= V["u,v,n1,e"] * Gamma1["wv"];
    temp1["e,w,N1,U"] -= V["v,U,e,N1"] * Gamma1["wv"];
    temp1["e,w,n1,u"] -= V["u,e,n1,v"] * Gamma1["wv"];
    temp1["e,w,N1,U"] -= V["e,U,v,N1"] * Gamma1["wv"];

    temp1["e,w,n1,u"] += H["e,n1"] * Gamma1["uw"];
    temp1["e,w,n1,u"] += V["e,m,n1,m1"] * Gamma1["uw"] * I["m,m1"];
    temp1["e,w,n1,u"] += V["e,M,n1,M1"] * Gamma1["uw"] * I["M,M1"];
    temp1["e,w,n1,u"] += 0.5 * V["x,y,n1,e"] * Gamma2["u,w,x,y"];
    temp1["e,w,N1,U"] += V["y,X,e,N1"] * Gamma2["w,U,y,X"];
    temp1["e,w,n1,u"] += V["e,y,n1,v"] * Gamma2["u,v,w,y"];
    temp1["e,w,n1,u"] += V["e,Y,n1,V"] * Gamma2["u,V,w,Y"];
    temp1["e,w,N1,U"] += V["e,Y,v,N1"] * Gamma2["v,U,w,Y"];

    temp1["e,w,u1,v1"] -= V["v1,v,u1,e"] * Gamma1["wv"];
    temp1["e,w,U1,V1"] -= V["v,V1,e,U1"] * Gamma1["wv"];

    temp1["e,w,w1,v"] -= F["ve"] * I["w,w1"];

    temp1["e,w,e1,u"] -= H["e,e1"] * Gamma1["uw"];
    temp1["e,w,e1,u"] -= V["e,m,e1,m1"] * Gamma1["uw"] * I["m,m1"];
    temp1["e,w,e1,u"] -= V["e,M,e1,M1"] * Gamma1["uw"] * I["M,M1"];
    temp1["e,w,e1,u"] -= 0.5 * V["x,y,e1,e"] * Gamma2["u,w,x,y"];
    temp1["e,w,E1,U"] -= V["y,X,e,E1"] * Gamma2["w,U,y,X"];
    temp1["e,w,e1,u"] -= V["e,y,e1,v"] * Gamma2["u,v,w,y"];
    temp1["e,w,e1,u"] -= V["e,Y,e1,V"] * Gamma2["u,V,w,Y"];
    temp1["e,w,E1,U"] -= V["e,Y,v,E1"] * Gamma2["v,U,w,Y"];

    // ACTIVE-ACTIVE
    temp1["w,z,w1,z1"] += Delta1["zw"] * I["w,w1"] * I["z,z1"];

    temp1["w,z,e1,m1"] -= V["m1,w,e1,u"] * Gamma1["uz"];
    temp1["w,z,E1,M1"] -= V["w,M1,u,E1"] * Gamma1["uz"];
    temp1["w,z,e1,m1"] -= V["m1,u,e1,w"] * Gamma1["uz"];
    temp1["w,z,E1,M1"] -= V["u,M1,w,E1"] * Gamma1["uz"];

    temp1["w,z,e1,m1"] += V["m1,z,e1,u"] * Gamma1["uw"];
    temp1["w,z,E1,M1"] += V["z,M1,u,E1"] * Gamma1["uw"];
    temp1["w,z,e1,m1"] += V["m1,u,e1,z"] * Gamma1["uw"];
    temp1["w,z,E1,M1"] += V["u,M1,z,E1"] * Gamma1["uw"];

    temp1["w,z,n1,z1"] -= F["w,n1"] * I["z,z1"];
    temp1["w,z,n1,w1"] += F["z,n1"] * I["w,w1"];

    temp1["w,z,n1,u"] -= V["u,v,n1,w"] * Gamma1["zv"];
    temp1["w,z,N1,U"] -= V["v,U,w,N1"] * Gamma1["zv"];
    temp1["w,z,n1,u"] -= V["u,w,n1,v"] * Gamma1["zv"];
    temp1["w,z,N1,U"] -= V["w,U,v,N1"] * Gamma1["zv"];

    temp1["w,z,n1,u"] += V["u,v,n1,z"] * Gamma1["wv"];
    temp1["w,z,N1,U"] += V["v,U,z,N1"] * Gamma1["wv"];
    temp1["w,z,n1,u"] += V["u,z,n1,v"] * Gamma1["wv"];
    temp1["w,z,N1,U"] += V["z,U,v,N1"] * Gamma1["wv"];

    temp1["w,z,n1,u"] += H["w,n1"] * Gamma1["uz"];
    temp1["w,z,n1,u"] += V["w,m1,n1,m"] * Gamma1["uz"] * I["m1,m"];
    temp1["w,z,n1,u"] += V["w,M1,n1,M"] * Gamma1["uz"] * I["M1,M"];
    temp1["w,z,n1,u"] += 0.5 * V["x,y,n1,w"] * Gamma2["u,z,x,y"];
    temp1["w,z,N1,U"] += V["y,X,w,N1"] * Gamma2["z,U,y,X"];
    temp1["w,z,n1,u"] += V["w,y,n1,v"] * Gamma2["u,v,z,y"];
    temp1["w,z,n1,u"] += V["w,Y,n1,V"] * Gamma2["u,V,z,Y"];
    temp1["w,z,N1,U"] += V["w,Y,v,N1"] * Gamma2["v,U,z,Y"];

    temp1["w,z,n1,u"] -= H["z,n1"] * Gamma1["uw"];
    temp1["w,z,n1,u"] -= V["z,m1,n1,m"] * Gamma1["uw"] * I["m1,m"];
    temp1["w,z,n1,u"] -= V["z,M1,n1,M"] * Gamma1["uw"] * I["M1,M"];
    temp1["w,z,n1,u"] -= 0.5 * V["x,y,n1,z"] * Gamma2["u,w,x,y"];
    temp1["w,z,N1,U"] -= V["y,X,z,N1"] * Gamma2["w,U,y,X"];
    temp1["w,z,n1,u"] -= V["z,y,n1,v"] * Gamma2["u,v,w,y"];
    temp1["w,z,n1,u"] -= V["z,Y,n1,V"] * Gamma2["u,V,w,Y"];
    temp1["w,z,N1,U"] -= V["z,Y,v,N1"] * Gamma2["v,U,w,Y"];

    temp1["w,z,e1,u"] -= H["w,e1"] * Gamma1["uz"];
    temp1["w,z,e1,u"] -= V["w,m1,e1,m"] * Gamma1["uz"] * I["m1,m"];
    temp1["w,z,e1,u"] -= V["w,M1,e1,M"] * Gamma1["uz"] * I["M1,M"];
    temp1["w,z,e1,u"] -= 0.5 * V["x,y,e1,w"] * Gamma2["u,z,x,y"];
    temp1["w,z,E1,U"] -= V["y,X,w,E1"] * Gamma2["z,U,y,X"];
    temp1["w,z,e1,u"] -= V["w,y,e1,v"] * Gamma2["u,v,z,y"];
    temp1["w,z,e1,u"] -= V["w,Y,e1,V"] * Gamma2["u,V,z,Y"];
    temp1["w,z,E1,U"] -= V["w,Y,v,E1"] * Gamma2["v,U,z,Y"];
    
    temp1["w,z,e1,u"] += H["z,e1"] * Gamma1["uw"];
    temp1["w,z,e1,u"] += V["z,m1,e1,m"] * Gamma1["uw"] * I["m1,m"];
    temp1["w,z,e1,u"] += V["z,M1,e1,M"] * Gamma1["uw"] * I["M1,M"];
    temp1["w,z,e1,u"] += 0.5 * V["x,y,e1,z"] * Gamma2["u,w,x,y"];
    temp1["w,z,E1,U"] += V["y,X,z,E1"] * Gamma2["w,U,y,X"];
    temp1["w,z,e1,u"] += V["z,y,e1,v"] * Gamma2["u,v,w,y"];
    temp1["w,z,e1,u"] += V["z,Y,e1,V"] * Gamma2["u,V,w,Y"];
    temp1["w,z,E1,U"] += V["z,Y,v,E1"] * Gamma2["v,U,w,Y"];

    temp1["w,z,u1,v1"] -= V["v1,v,u1,w"] * Gamma1["zv"];
    temp1["w,z,U1,V1"] -= V["v,V1,w,U1"] * Gamma1["zv"];

    temp1["w,z,u1,v1"] += V["v1,v,u1,z"] * Gamma1["wv"];
    temp1["w,z,U1,V1"] += V["v,V1,z,U1"] * Gamma1["wv"];

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
    auto ci_vc = ambit::Tensor::build(ambit::CoreTensor, "ci contribution to z{vc}", {ndets, nvirt_, ncore_});
    auto ci_ca = ambit::Tensor::build(ambit::CoreTensor, "ci contribution to z{ca}", {ndets, ncore_, na_});
    auto ci_va = ambit::Tensor::build(ambit::CoreTensor, "ci contribution to z{va}", {ndets, nvirt_, na_});
    auto ci_aa = ambit::Tensor::build(ambit::CoreTensor, "ci contribution to z{aa}", {ndets, na_, na_});

    // CI contribution to Z{VC}
    ci_vc("Iem") -= H.block("vc")("em") * ci("I");
    ci_vc("Iem") -= V_N_Alpha.block("cv")("me") * ci("I");
    ci_vc("Iem") -= V_N_Beta.block("cv")("me") * ci("I");
    ci_vc("Iem") -= 0.5 * V.block("acav")("umve") * cc1a_n("Iuv");
    ci_vc("Iem") -= 0.5 * V.block("cAvA")("mUeV") * cc1b_n("IUV");
    ci_vc("Iem") -= 0.5 * V.block("acav")("umve") * cc1a_r("Iuv");
    ci_vc("Iem") -= 0.5 * V.block("cAvA")("mUeV") * cc1b_r("IUV");

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
    ci_ca("Imw") -= 0.0625 * V.block("caaa")("mvxy") * cc2aa_n("Iwvxy");
    ci_ca("Imw") -= 0.1250 * V.block("cAaA")("mVxY") * cc2ab_n("IwVxY");
    ci_ca("Imw") -= 0.0625 * V.block("acaa")("umxy") * cc2aa_n("Iuwxy");
    ci_ca("Imw") -= 0.1250 * V.block("cAaA")("mUxY") * cc2ab_n("IwUxY");
    ci_ca("Imw") -= 0.0625 * V.block("aaca")("uvmy") * cc2aa_n("Iuvwy");
    ci_ca("Imw") -= 0.1250 * V.block("aAcA")("uVmY") * cc2ab_n("IuVwY");
    ci_ca("Imw") -= 0.0625 * V.block("aaac")("uvxm") * cc2aa_n("Iuvxw");
    ci_ca("Imw") -= 0.1250 * V.block("aAcA")("uVmX") * cc2ab_n("IuVwX");
    ci_ca("Jmw") -= 0.0625 * V.block("caaa")("mvxy") * cc2aa_r("Jwvxy");
    ci_ca("Jmw") -= 0.1250 * V.block("cAaA")("mVxY") * cc2ab_r("JwVxY");
    ci_ca("Jmw") -= 0.0625 * V.block("acaa")("umxy") * cc2aa_r("Juwxy");
    ci_ca("Jmw") -= 0.1250 * V.block("cAaA")("mUxY") * cc2ab_r("JwUxY");
    ci_ca("Jmw") -= 0.0625 * V.block("aaca")("uvmy") * cc2aa_r("Juvwy");
    ci_ca("Jmw") -= 0.1250 * V.block("aAcA")("uVmY") * cc2ab_r("JuVwY");
    ci_ca("Jmw") -= 0.0625 * V.block("aaac")("uvxm") * cc2aa_r("Juvxw");
    ci_ca("Jmw") -= 0.1250 * V.block("aAcA")("uVmX") * cc2ab_r("JuVwX");

    ci_ca("Imw") += H.block("ac")("wm") * ci("I");
    ci_ca("Imw") += V_N_Alpha.block("ca")("mw") * ci("I");
    ci_ca("Imw") += V_N_Beta.block("ca")("mw") * ci("I");
    ci_ca("Imw") += 0.5 * V.block("acaa")("umvw") * cc1a_n("Iuv");
    ci_ca("Imw") += 0.5 * V.block("cAaA")("mUwV") * cc1b_n("IUV");
    ci_ca("Imw") += 0.5 * V.block("acaa")("umvw") * cc1a_r("Iuv");
    ci_ca("Imw") += 0.5 * V.block("cAaA")("mUwV") * cc1b_r("IUV");

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
    ci_va("Iew") -= 0.0625 * V.block("aava")("uvey") * cc2aa_n("Iuvwy");
    ci_va("Iew") -= 0.1250 * V.block("aAvA")("uVeY") * cc2ab_n("IuVwY");
    ci_va("Iew") -= 0.0625 * V.block("aaav")("uvxe") * cc2aa_n("Iuvxw");
    ci_va("Iew") -= 0.1250 * V.block("aAvA")("uVeX") * cc2ab_n("IuVwX");

    ci_va("Jew") -= 0.0625 * V.block("vaaa")("evxy") * cc2aa_r("Jwvxy");
    ci_va("Jew") -= 0.1250 * V.block("vAaA")("eVxY") * cc2ab_r("JwVxY");
    ci_va("Jew") -= 0.0625 * V.block("avaa")("uexy") * cc2aa_r("Juwxy");
    ci_va("Jew") -= 0.1250 * V.block("vAaA")("eUxY") * cc2ab_r("JwUxY");
    ci_va("Jew") -= 0.0625 * V.block("aava")("uvey") * cc2aa_r("Juvwy");
    ci_va("Jew") -= 0.1250 * V.block("aAvA")("uVeY") * cc2ab_r("JuVwY");
    ci_va("Jew") -= 0.0625 * V.block("aaav")("uvxe") * cc2aa_r("Juvxw");
    ci_va("Jew") -= 0.1250 * V.block("aAvA")("uVeX") * cc2ab_r("JuVwX");

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
    Alpha = 0.0;

    Alpha += H["vu"] * Gamma1["uv"];
    Alpha += H["VU"] * Gamma1["UV"];
    Alpha += V_N_Alpha["v,u"] * Gamma1["uv"];
    Alpha += V_N_Beta["v,u"] * Gamma1["uv"];
    Alpha += V["V,M,U,M1"] * Gamma1["UV"] * I["M,M1"];
    Alpha += V["m,V,m1,U"] * Gamma1["UV"] * I["m,m1"];
    Alpha += 0.25 * V["xyuv"] * Gamma2["uvxy"];
    Alpha += 0.25 * V["XYUV"] * Gamma2["UVXY"];
    Alpha += V["xYuV"] * Gamma2["uVxY"];

    BlockedTensor temp = BTF_->build(CoreTensor, "temporal tensor", {"aa","AA"});

    if (PT2_TERM) {
        temp["uv"] += 0.25 * T2_["vjab"] * V_["cdul"] * Gamma1["lj"] * Eta1["ac"] * Eta1["bd"];
        temp["uv"] += 0.50 * T2_["vJaB"] * V_["cDuL"] * Gamma1["LJ"] * Eta1["ac"] * Eta1["BD"];
        temp["UV"] += 0.25 * T2_["VJAB"] * V_["CDUL"] * Gamma1["LJ"] * Eta1["AC"] * Eta1["BD"];
        temp["UV"] += 0.50 * T2_["jVaB"] * V_["cDlU"] * Gamma1["lj"] * Eta1["ac"] * Eta1["BD"];

        temp["uv"] += 0.25 * T2_["ujab"] * V_["cdvl"] * Gamma1["lj"] * Eta1["ac"] * Eta1["bd"];
        temp["uv"] += 0.50 * T2_["uJaB"] * V_["cDvL"] * Gamma1["LJ"] * Eta1["ac"] * Eta1["BD"];
        temp["UV"] += 0.25 * T2_["UJAB"] * V_["CDVL"] * Gamma1["LJ"] * Eta1["AC"] * Eta1["BD"];
        temp["UV"] += 0.50 * T2_["jUaB"] * V_["cDlV"] * Gamma1["lj"] * Eta1["ac"] * Eta1["BD"];

        temp["uv"] += 0.25 * T2_["ivab"] * V_["cdku"] * Gamma1["ki"] * Eta1["ac"] * Eta1["bd"];
        temp["uv"] += 0.50 * T2_["vIaB"] * V_["cDuK"] * Gamma1["KI"] * Eta1["ac"] * Eta1["BD"];
        temp["UV"] += 0.25 * T2_["IVAB"] * V_["CDKU"] * Gamma1["KI"] * Eta1["AC"] * Eta1["BD"];
        temp["UV"] += 0.50 * T2_["iVaB"] * V_["cDkU"] * Gamma1["ki"] * Eta1["ac"] * Eta1["BD"];

        temp["uv"] += 0.25 * T2_["iuab"] * V_["cdkv"] * Gamma1["ki"] * Eta1["ac"] * Eta1["bd"];
        temp["uv"] += 0.50 * T2_["uIaB"] * V_["cDvK"] * Gamma1["KI"] * Eta1["ac"] * Eta1["BD"];
        temp["UV"] += 0.25 * T2_["IUAB"] * V_["CDKV"] * Gamma1["KI"] * Eta1["AC"] * Eta1["BD"];
        temp["UV"] += 0.50 * T2_["iUaB"] * V_["cDkV"] * Gamma1["ki"] * Eta1["ac"] * Eta1["BD"];

        temp["uv"] -= 0.25 * T2_["ijub"] * V_["vdkl"] * Gamma1["ki"] * Gamma1["lj"] * Eta1["bd"];
        temp["uv"] -= 0.50 * T2_["iJuB"] * V_["vDkL"] * Gamma1["ki"] * Gamma1["LJ"] * Eta1["BD"];
        temp["UV"] -= 0.25 * T2_["IJUB"] * V_["VDKL"] * Gamma1["KI"] * Gamma1["LJ"] * Eta1["BD"];
        temp["UV"] -= 0.50 * T2_["iJbU"] * V_["dVkL"] * Gamma1["ki"] * Gamma1["LJ"] * Eta1["bd"];

        temp["uv"] -= 0.25 * T2_["ijvb"] * V_["udkl"] * Gamma1["ki"] * Gamma1["lj"] * Eta1["bd"];
        temp["uv"] -= 0.50 * T2_["iJvB"] * V_["uDkL"] * Gamma1["ki"] * Gamma1["LJ"] * Eta1["BD"];
        temp["UV"] -= 0.25 * T2_["IJVB"] * V_["UDKL"] * Gamma1["KI"] * Gamma1["LJ"] * Eta1["BD"];
        temp["UV"] -= 0.50 * T2_["iJbV"] * V_["dUkL"] * Gamma1["ki"] * Gamma1["LJ"] * Eta1["bd"];

        temp["uv"] -= 0.25 * T2_["ijau"] * V_["cvkl"] * Gamma1["ki"] * Gamma1["lj"] * Eta1["ac"];
        temp["uv"] -= 0.50 * T2_["iJuA"] * V_["vCkL"] * Gamma1["ki"] * Gamma1["LJ"] * Eta1["AC"];
        temp["UV"] -= 0.25 * T2_["IJAU"] * V_["CVKL"] * Gamma1["KI"] * Gamma1["LJ"] * Eta1["AC"];
        temp["UV"] -= 0.50 * T2_["iJaU"] * V_["cVkL"] * Gamma1["ki"] * Gamma1["LJ"] * Eta1["ac"];

        temp["uv"] -= 0.25 * T2_["ijav"] * V_["cukl"] * Gamma1["ki"] * Gamma1["lj"] * Eta1["ac"];
        temp["uv"] -= 0.50 * T2_["iJvA"] * V_["uCkL"] * Gamma1["ki"] * Gamma1["LJ"] * Eta1["AC"];
        temp["UV"] -= 0.25 * T2_["IJAV"] * V_["CUKL"] * Gamma1["KI"] * Gamma1["LJ"] * Eta1["AC"];
        temp["UV"] -= 0.50 * T2_["iJaV"] * V_["cUkL"] * Gamma1["ki"] * Gamma1["LJ"] * Eta1["ac"];
    }

    if (X1_TERM) {
        temp["zw"] -= 0.125 * T2_["uvwb"] * V_["zdxy"] * Eta1["bd"] * Lambda2_["xyuv"];
        temp["zw"] -= 0.500 * T2_["uVwB"] * V_["zDxY"] * Eta1["BD"] * Lambda2_["xYuV"];
        temp["ZW"] -= 0.125 * T2_["UVWB"] * V_["ZDXY"] * Eta1["BD"] * Lambda2_["XYUV"];
        temp["ZW"] -= 0.500 * T2_["uVbW"] * V_["dZxY"] * Eta1["bd"] * Lambda2_["xYuV"];

        temp["zw"] -= 0.125 * T2_["uvzb"] * V_["wdxy"] * Eta1["bd"] * Lambda2_["xyuv"];
        temp["zw"] -= 0.500 * T2_["uVzB"] * V_["wDxY"] * Eta1["BD"] * Lambda2_["xYuV"];
        temp["ZW"] -= 0.125 * T2_["UVZB"] * V_["WDXY"] * Eta1["BD"] * Lambda2_["XYUV"];
        temp["ZW"] -= 0.500 * T2_["uVbZ"] * V_["dWxY"] * Eta1["bd"] * Lambda2_["xYuV"];

        temp["zw"] -= 0.125 * T2_["uvaw"] * V_["czxy"] * Eta1["ac"] * Lambda2_["xyuv"];
        temp["zw"] -= 0.500 * T2_["uVwA"] * V_["zCxY"] * Eta1["AC"] * Lambda2_["xYuV"];
        temp["ZW"] -= 0.125 * T2_["UVAW"] * V_["CZXY"] * Eta1["AC"] * Lambda2_["XYUV"];
        temp["ZW"] -= 0.500 * T2_["uVaW"] * V_["cZxY"] * Eta1["ac"] * Lambda2_["xYuV"];

        temp["zw"] -= 0.125 * T2_["uvaz"] * V_["cwxy"] * Eta1["ac"] * Lambda2_["xyuv"];
        temp["zw"] -= 0.500 * T2_["uVzA"] * V_["wCxY"] * Eta1["AC"] * Lambda2_["xYuV"];
        temp["ZW"] -= 0.125 * T2_["UVAZ"] * V_["CWXY"] * Eta1["AC"] * Lambda2_["XYUV"];
        temp["ZW"] -= 0.500 * T2_["uVaZ"] * V_["cWxY"] * Eta1["ac"] * Lambda2_["xYuV"];
    }

    if (X2_TERM) {
        temp["zw"] += 0.125 * T2_["wjxy"] * V_["uvzl"] * Gamma1["lj"] * Lambda2_["xyuv"];
        temp["zw"] += 0.500 * T2_["wJxY"] * V_["uVzL"] * Gamma1["LJ"] * Lambda2_["xYuV"];
        temp["ZW"] += 0.125 * T2_["WJXY"] * V_["UVZL"] * Gamma1["LJ"] * Lambda2_["XYUV"];
        temp["ZW"] += 0.500 * T2_["jWxY"] * V_["uVlZ"] * Gamma1["lj"] * Lambda2_["xYuV"];

        temp["zw"] += 0.125 * T2_["zjxy"] * V_["uvwl"] * Gamma1["lj"] * Lambda2_["xyuv"];
        temp["zw"] += 0.500 * T2_["zJxY"] * V_["uVwL"] * Gamma1["LJ"] * Lambda2_["xYuV"];
        temp["ZW"] += 0.125 * T2_["ZJXY"] * V_["UVWL"] * Gamma1["LJ"] * Lambda2_["XYUV"];
        temp["ZW"] += 0.500 * T2_["jZxY"] * V_["uVlW"] * Gamma1["lj"] * Lambda2_["xYuV"];

        temp["zw"] += 0.125 * T2_["iwxy"] * V_["uvkz"] * Gamma1["ki"] * Lambda2_["xyuv"];
        temp["zw"] += 0.500 * T2_["wIxY"] * V_["uVzK"] * Gamma1["KI"] * Lambda2_["xYuV"];
        temp["ZW"] += 0.125 * T2_["IWXY"] * V_["UVKZ"] * Gamma1["KI"] * Lambda2_["XYUV"];
        temp["ZW"] += 0.500 * T2_["iWxY"] * V_["uVkZ"] * Gamma1["ki"] * Lambda2_["xYuV"];

        temp["zw"] += 0.125 * T2_["izxy"] * V_["uvkw"] * Gamma1["ki"] * Lambda2_["xyuv"];
        temp["zw"] += 0.500 * T2_["zIxY"] * V_["uVwK"] * Gamma1["KI"] * Lambda2_["xYuV"];
        temp["ZW"] += 0.125 * T2_["IZXY"] * V_["UVKW"] * Gamma1["KI"] * Lambda2_["XYUV"];
        temp["ZW"] += 0.500 * T2_["iZxY"] * V_["uVkW"] * Gamma1["ki"] * Lambda2_["xYuV"]; 
    }

    if (X3_TERM) {
        temp["zw"] -= V_["vbzx"] * T2_["wuya"] * Eta1["ab"] * Lambda2_["xyuv"];
        temp["zw"] -= V_["vbzx"] * T2_["wUaY"] * Eta1["ab"] * Lambda2_["xYvU"];
        temp["zw"] -= V_["vBzX"] * T2_["wUyA"] * Eta1["AB"] * Lambda2_["yXvU"];
        temp["zw"] -= V_["bVzX"] * T2_["wUaY"] * Eta1["ab"] * Lambda2_["XYUV"];
        temp["zw"] -= V_["bVzX"] * T2_["wuya"] * Eta1["ab"] * Lambda2_["yXuV"];
        temp["ZW"] -= V_["bVxZ"] * T2_["uWaY"] * Eta1["ab"] * Lambda2_["xYuV"];
        temp["ZW"] -= V_["vBxZ"] * T2_["uWyA"] * Eta1["AB"] * Lambda2_["xyuv"];
        temp["ZW"] -= V_["vBxZ"] * T2_["WUYA"] * Eta1["AB"] * Lambda2_["xYvU"];
        temp["ZW"] -= V_["VBZX"] * T2_["WUYA"] * Eta1["AB"] * Lambda2_["XYUV"];
        temp["ZW"] -= V_["VBZX"] * T2_["uWyA"] * Eta1["AB"] * Lambda2_["yXuV"];

        temp["wz"] -= V_["vbwx"] * T2_["zuya"] * Eta1["ab"] * Lambda2_["xyuv"];
        temp["wz"] -= V_["vbwx"] * T2_["zUaY"] * Eta1["ab"] * Lambda2_["xYvU"];
        temp["wz"] -= V_["vBwX"] * T2_["zUyA"] * Eta1["AB"] * Lambda2_["yXvU"];
        temp["wz"] -= V_["bVwX"] * T2_["zUaY"] * Eta1["ab"] * Lambda2_["XYUV"];
        temp["wz"] -= V_["bVwX"] * T2_["zuya"] * Eta1["ab"] * Lambda2_["yXuV"];
        temp["WZ"] -= V_["bVxW"] * T2_["uZaY"] * Eta1["ab"] * Lambda2_["xYuV"];
        temp["WZ"] -= V_["vBxW"] * T2_["uZyA"] * Eta1["AB"] * Lambda2_["xyuv"];
        temp["WZ"] -= V_["vBxW"] * T2_["ZUYA"] * Eta1["AB"] * Lambda2_["xYvU"];
        temp["WZ"] -= V_["VBWX"] * T2_["ZUYA"] * Eta1["AB"] * Lambda2_["XYUV"];
        temp["WZ"] -= V_["VBWX"] * T2_["uZyA"] * Eta1["AB"] * Lambda2_["yXuV"];

        temp["zw"] += V_["vwjx"] * T2_["iuyz"] * Gamma1["ji"] * Lambda2_["xyuv"];
        temp["zw"] += V_["vwjx"] * T2_["iUzY"] * Gamma1["ji"] * Lambda2_["xYvU"];
        temp["ZW"] += V_["vWjX"] * T2_["iUyZ"] * Gamma1["ji"] * Lambda2_["yXvU"];
        temp["zw"] += V_["wVjX"] * T2_["iUzY"] * Gamma1["ji"] * Lambda2_["XYUV"];
        temp["zw"] += V_["wVjX"] * T2_["iuyz"] * Gamma1["ji"] * Lambda2_["yXuV"];
        temp["zw"] += V_["wVxJ"] * T2_["uIzY"] * Gamma1["JI"] * Lambda2_["xYuV"];
        temp["ZW"] += V_["vWxJ"] * T2_["uIyZ"] * Gamma1["JI"] * Lambda2_["xyuv"];
        temp["ZW"] += V_["vWxJ"] * T2_["IUYZ"] * Gamma1["JI"] * Lambda2_["xYvU"];
        temp["ZW"] += V_["VWJX"] * T2_["IUYZ"] * Gamma1["JI"] * Lambda2_["XYUV"];
        temp["ZW"] += V_["VWJX"] * T2_["uIyZ"] * Gamma1["JI"] * Lambda2_["yXuV"];

        temp["zw"] += V_["vzjx"] * T2_["iuyw"] * Gamma1["ji"] * Lambda2_["xyuv"];
        temp["zw"] += V_["vzjx"] * T2_["iUwY"] * Gamma1["ji"] * Lambda2_["xYvU"];
        temp["ZW"] += V_["vZjX"] * T2_["iUyW"] * Gamma1["ji"] * Lambda2_["yXvU"];
        temp["zw"] += V_["zVjX"] * T2_["iUwY"] * Gamma1["ji"] * Lambda2_["XYUV"];
        temp["zw"] += V_["zVjX"] * T2_["iuyw"] * Gamma1["ji"] * Lambda2_["yXuV"];
        temp["zw"] += V_["zVxJ"] * T2_["uIwY"] * Gamma1["JI"] * Lambda2_["xYuV"];
        temp["ZW"] += V_["vZxJ"] * T2_["uIyW"] * Gamma1["JI"] * Lambda2_["xyuv"];
        temp["ZW"] += V_["vZxJ"] * T2_["IUYW"] * Gamma1["JI"] * Lambda2_["xYvU"];
        temp["ZW"] += V_["VZJX"] * T2_["IUYW"] * Gamma1["JI"] * Lambda2_["XYUV"];
        temp["ZW"] += V_["VZJX"] * T2_["uIyW"] * Gamma1["JI"] * Lambda2_["yXuV"];
    }

    if (CORRELATION_TERM) {
        temp["vu"] += Sigma3["ia"] * V["ivau"];
        temp["vu"] += Sigma3["IA"] * V["vIuA"];
        temp["VU"] += Sigma3["IA"] * V["IVAU"];
        temp["VU"] += Sigma3["ia"] * V["iVaU"];

        temp["vu"] += Sigma3["ia"] * V["iuav"];
        temp["vu"] += Sigma3["IA"] * V["uIvA"];
        temp["VU"] += Sigma3["IA"] * V["IUAV"];
        temp["VU"] += Sigma3["ia"] * V["iUaV"];

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
        temp["vu"] += Xi3["ia"] * V["ivau"];
        temp["vu"] += Xi3["IA"] * V["vIuA"];
        temp["VU"] += Xi3["IA"] * V["IVAU"];
        temp["VU"] += Xi3["ia"] * V["iVaU"];

        temp["vu"] += Xi3["ia"] * V["iuav"];
        temp["vu"] += Xi3["IA"] * V["uIvA"];
        temp["VU"] += Xi3["IA"] * V["IUAV"];
        temp["VU"] += Xi3["ia"] * V["iUaV"];

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
        temp["uv"] += F_["bv"] * T1_["ua"] * Eta1["ab"];
        temp["UV"] += F_["BV"] * T1_["UA"] * Eta1["AB"];

        temp["uv"] += F_["bu"] * T1_["va"] * Eta1["ab"];
        temp["UV"] += F_["BU"] * T1_["VA"] * Eta1["AB"];

        temp["uv"] -= F_["vj"] * T1_["iu"] * Gamma1["ij"];
        temp["UV"] -= F_["VJ"] * T1_["IU"] * Gamma1["IJ"];

        temp["uv"] -= F_["uj"] * T1_["iv"] * Gamma1["ij"];
        temp["UV"] -= F_["UJ"] * T1_["IV"] * Gamma1["IJ"];
    }

    b_ck("K") -= 0.5 * temp.block("aa")("uv") * cc1a_n("Kuv"); 
    b_ck("K") -= 0.5 * temp.block("AA")("UV") * cc1b_n("KUV");
    b_ck("K") -= 0.5 * temp.block("aa")("uv") * cc1a_r("Kuv"); 
    b_ck("K") -= 0.5 * temp.block("AA")("UV") * cc1b_r("KUV");
    Alpha += 0.5 * temp["uv"] * Gamma1["uv"]; 
    Alpha += 0.5 * temp["UV"] * Gamma1["UV"];
    temp.zero();

    BlockedTensor temp4 = BTF_->build(CoreTensor, "temporal tensor 4", {"aaaa","AAAA","aAaA"});

    if (X1_TERM) {
        temp4["uvxy"] += 0.125 * V_["cdxy"] * T2_["uvab"] * Eta1["ac"] * Eta1["bd"];
        temp4["UVXY"] += 0.125 * V_["CDXY"] * T2_["UVAB"] * Eta1["AC"] * Eta1["BD"];
        temp4["uVxY"] += V_["cDxY"] * T2_["uVaB"] * Eta1["ac"] * Eta1["BD"];
    }

    if (X2_TERM) {
        temp4["uvxy"] += 0.125 * V_["uvkl"] * T2_["ijxy"] * Gamma1["ki"] * Gamma1["lj"];
        temp4["UVXY"] += 0.125 * V_["UVKL"] * T2_["IJXY"] * Gamma1["KI"] * Gamma1["LJ"];
        temp4["uVxY"] += V_["uVkL"] * T2_["iJxY"] * Gamma1["ki"] * Gamma1["LJ"];
    }

    if (X3_TERM) {
        temp4["xyuv"] -= V_["vbjx"] * T2_["iuya"] * Gamma1["ji"] * Eta1["ab"];
        temp4["xyuv"] -= V_["vBxJ"] * T2_["uIyA"] * Gamma1["JI"] * Eta1["AB"];   
        temp4["XYUV"] -= V_["bVjX"] * T2_["iUaY"] * Gamma1["ji"] * Eta1["ab"];
        temp4["XYUV"] -= V_["VBJX"] * T2_["IUYA"] * Gamma1["JI"] * Eta1["AB"];
        temp4["xYvU"] -= V_["vbjx"] * T2_["iUaY"] * Gamma1["ji"] * Eta1["ab"];
        temp4["xYvU"] -= V_["vBxJ"] * T2_["IUYA"] * Gamma1["JI"] * Eta1["AB"];
        temp4["yXuV"] -= V_["bVjX"] * T2_["iuya"] * Gamma1["ji"] * Eta1["ab"];
        temp4["yXuV"] -= V_["VBJX"] * T2_["uIyA"] * Gamma1["JI"] * Eta1["AB"];
        temp4["yXvU"] -= V_["vBjX"] * T2_["iUyA"] * Gamma1["ji"] * Eta1["AB"];
        temp4["xYuV"] -= V_["bVxJ"] * T2_["uIaY"] * Gamma1["JI"] * Eta1["ab"];
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
    Alpha -= temp4["uvxy"] * Gamma2["xyuv"];
    Alpha -= temp4["UVXY"] * Gamma2["XYUV"];
    Alpha -= temp4["uVxY"] * Gamma2["xYuV"];
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

        Alpha -= 1.0 * V_["uvmz"] * T2_["mwxy"] * Gamma1["uz"] * Gamma2["xyvw"];
        Alpha -= 2.0 * V_["uvmz"] * T2_["mWxY"] * Gamma1["uz"] * Gamma2["xYvW"];
        Alpha += 1.0 * V_["uVzM"] * T2_["MWXY"] * Gamma1["uz"] * Gamma2["XYVW"];
        Alpha += 2.0 * V_["uVzM"] * T2_["wMxY"] * Gamma1["uz"] * Gamma2["xYwV"];
        Alpha -= 1.0 * V_["UVMZ"] * T2_["MWXY"] * Gamma1["UZ"] * Gamma2["XYVW"];
        Alpha -= 2.0 * V_["UVMZ"] * T2_["wMxY"] * Gamma1["UZ"] * Gamma2["xYwV"];  
        Alpha += 1.0 * V_["vUmZ"] * T2_["mwxy"] * Gamma1["UZ"] * Gamma2["xyvw"];
        Alpha += 2.0 * V_["vUmZ"] * T2_["mWxY"] * Gamma1["UZ"] * Gamma2["xYvW"];

        Alpha -= 0.5 * V_["uvmz"] * T2_["mwxy"] * Gamma1["wz"] * Gamma2["xyuv"];
        Alpha -= 2.0 * V_["uVzM"] * T2_["wMxY"] * Gamma1["wz"] * Gamma2["xYuV"];
        Alpha -= 0.5 * V_["UVMZ"] * T2_["MWXY"] * Gamma1["WZ"] * Gamma2["XYUV"];
        Alpha -= 2.0 * V_["uVmZ"] * T2_["mWxY"] * Gamma1["WZ"] * Gamma2["xYuV"];

        Alpha += 2.0 * V_["uvmz"] * T2_["mwxy"] * Gamma1["xu"] * Gamma2["vwzy"];
        Alpha += 2.0 * V_["uVmZ"] * T2_["mwxy"] * Gamma1["xu"] * Gamma2["wVyZ"];
        Alpha += 2.0 * V_["uvmz"] * T2_["mWxY"] * Gamma1["xu"] * Gamma2["vWzY"];
        Alpha += 2.0 * V_["uVmZ"] * T2_["mWxY"] * Gamma1["xu"] * Gamma2["VWZY"];
        Alpha -= 2.0 * V_["uVzM"] * T2_["wMxY"] * Gamma1["xu"] * Gamma2["wVzY"];

        Alpha += 2.0 * V_["UVMZ"] * T2_["MWXY"] * Gamma1["XU"] * Gamma2["VWZY"];
        Alpha += 2.0 * V_["vUzM"] * T2_["MWXY"] * Gamma1["XU"] * Gamma2["vWzY"];
        Alpha += 2.0 * V_["UVMZ"] * T2_["wMyX"] * Gamma1["XU"] * Gamma2["wVyZ"];
        Alpha += 2.0 * V_["vUzM"] * T2_["wMyX"] * Gamma1["XU"] * Gamma2["vwzy"];
        Alpha -= 2.0 * V_["vUmZ"] * T2_["mWyX"] * Gamma1["XU"] * Gamma2["vWyZ"];

        Alpha += 1.0 * V_["uvmz"] * T2_["mwxy"] * Gamma1["xw"] * Gamma2["uvzy"];
        Alpha -= 2.0 * V_["uVmZ"] * T2_["mwxy"] * Gamma1["xw"] * Gamma2["uVyZ"];
        Alpha -= 1.0 * V_["UVMZ"] * T2_["wMxY"] * Gamma1["xw"] * Gamma2["UVZY"];
        Alpha += 2.0 * V_["uVzM"] * T2_["wMxY"] * Gamma1["xw"] * Gamma2["uVzY"];

        Alpha -= 1.0 * V_["uvmz"] * T2_["mWyX"] * Gamma1["XW"] * Gamma2["uvzy"];
        Alpha += 2.0 * V_["uVmZ"] * T2_["mWyX"] * Gamma1["XW"] * Gamma2["uVyZ"];
        Alpha += 1.0 * V_["UVMZ"] * T2_["MWXY"] * Gamma1["XW"] * Gamma2["UVZY"];
        Alpha -= 2.0 * V_["uVzM"] * T2_["MWXY"] * Gamma1["XW"] * Gamma2["uVzY"];

        Alpha += 6 * V_["uvmz"] * T2_["mwxy"] * Gamma1["uz"] * Gamma1["xv"] * Gamma1["yw"];
        Alpha += 6 * V_["uvmz"] * T2_["mWxY"] * Gamma1["uz"] * Gamma1["xv"] * Gamma1["YW"];
        Alpha -= 6 * V_["vUmZ"] * T2_["mwxy"] * Gamma1["UZ"] * Gamma1["xv"] * Gamma1["yw"];
        Alpha -= 6 * V_["vUmZ"] * T2_["mWxY"] * Gamma1["UZ"] * Gamma1["xv"] * Gamma1["YW"];

        Alpha += 6 * V_["UVMZ"] * T2_["MWXY"] * Gamma1["UZ"] * Gamma1["XV"] * Gamma1["YW"];
        Alpha += 6 * V_["UVMZ"] * T2_["wMyX"] * Gamma1["UZ"] * Gamma1["XV"] * Gamma1["yw"];
        Alpha -= 6 * V_["uVzM"] * T2_["MWXY"] * Gamma1["uz"] * Gamma1["XV"] * Gamma1["YW"];
        Alpha -= 6 * V_["uVzM"] * T2_["wMyX"] * Gamma1["uz"] * Gamma1["XV"] * Gamma1["yw"];

        Alpha += 3.0 * V_["uvmz"] * T2_["mwxy"] * Gamma1["wz"] * Gamma1["xu"] * Gamma1["yv"];
        Alpha += 6.0 * V_["uVmZ"] * T2_["mWxY"] * Gamma1["WZ"] * Gamma1["xu"] * Gamma1["YV"];
        Alpha += 3.0 * V_["UVMZ"] * T2_["MWXY"] * Gamma1["WZ"] * Gamma1["XU"] * Gamma1["YV"];
        Alpha += 6.0 * V_["uVzM"] * T2_["wMxY"] * Gamma1["wz"] * Gamma1["xu"] * Gamma1["YV"];

        Alpha -= 0.25 * V_.block("vaaa")("ewxy") * T2_.block("aava")("uvez") * rdms_.g3aaa()("xyzuvw");
        Alpha -= 0.25 * V_.block("VAAA")("EWXY") * T2_.block("AAVA")("UVEZ") * rdms_.g3bbb()("XYZUVW");
        Alpha += 0.50 * V_.block("vAaA")("eWxZ") * T2_.block("aava")("uvey") * rdms_.g3aab()("xyZuvW");
        Alpha -= 0.50 * V_.block("avaa")("vexy") * T2_.block("aAvA")("uWeZ") * rdms_.g3aab()("xyZuvW");
        Alpha -= 1.00 * V_.block("aVaA")("vExZ") * T2_.block("aAaV")("uWyE") * rdms_.g3aab()("xyZuvW");
        Alpha += 0.50 * V_.block("aVaA")("uExY") * T2_.block("AAVA")("VWEZ") * rdms_.g3abb()("xYZuVW");
        Alpha -= 0.50 * V_.block("AVAA")("WEYZ") * T2_.block("aAaV")("uVxE") * rdms_.g3abb()("xYZuVW");
        Alpha -= 1.00 * V_.block("vAaA")("eWxY") * T2_.block("aAvA")("uVeZ") * rdms_.g3abb()("xYZuVW");

        Alpha += 1.0 * V_["ezuv"] * T2_["xyew"] * Gamma1["uz"] * Gamma2["xyvw"];
        Alpha += 2.0 * V_["ezuv"] * T2_["xYeW"] * Gamma1["uz"] * Gamma2["xYvW"];
        Alpha -= 1.0 * V_["zEuV"] * T2_["XYEW"] * Gamma1["uz"] * Gamma2["XYVW"];
        Alpha -= 2.0 * V_["zEuV"] * T2_["xYwE"] * Gamma1["uz"] * Gamma2["xYwV"];
        Alpha += 1.0 * V_["EZUV"] * T2_["XYEW"] * Gamma1["UZ"] * Gamma2["XYVW"];
        Alpha += 2.0 * V_["EZUV"] * T2_["xYwE"] * Gamma1["UZ"] * Gamma2["xYwV"];  
        Alpha -= 1.0 * V_["eZvU"] * T2_["xyew"] * Gamma1["UZ"] * Gamma2["xyvw"];
        Alpha -= 2.0 * V_["eZvU"] * T2_["xYeW"] * Gamma1["UZ"] * Gamma2["xYvW"];

        Alpha += 0.5 * V_["ezuv"] * T2_["xyew"] * Gamma1["wz"] * Gamma2["xyuv"];
        Alpha += 2.0 * V_["zEuV"] * T2_["xYwE"] * Gamma1["wz"] * Gamma2["xYuV"];
        Alpha += 0.5 * V_["EZUV"] * T2_["XYEW"] * Gamma1["WZ"] * Gamma2["XYUV"];
        Alpha += 2.0 * V_["eZuV"] * T2_["xYeW"] * Gamma1["WZ"] * Gamma2["xYuV"];

        Alpha -= 2.0 * V_["ezuv"] * T2_["xyew"] * Gamma1["xu"] * Gamma2["vwzy"];
        Alpha -= 2.0 * V_["eZuV"] * T2_["xyew"] * Gamma1["xu"] * Gamma2["wVyZ"];
        Alpha -= 2.0 * V_["ezuv"] * T2_["xYeW"] * Gamma1["xu"] * Gamma2["vWzY"];
        Alpha -= 2.0 * V_["eZuV"] * T2_["xYeW"] * Gamma1["xu"] * Gamma2["VWZY"];
        Alpha += 2.0 * V_["zEuV"] * T2_["xYwE"] * Gamma1["xu"] * Gamma2["wVzY"];
        Alpha -= 2.0 * V_["EZUV"] * T2_["XYEW"] * Gamma1["XU"] * Gamma2["VWZY"];
        Alpha -= 2.0 * V_["zEvU"] * T2_["XYEW"] * Gamma1["XU"] * Gamma2["vWzY"];
        Alpha -= 2.0 * V_["EZUV"] * T2_["yXwE"] * Gamma1["XU"] * Gamma2["wVyZ"];
        Alpha -= 2.0 * V_["zEvU"] * T2_["yXwE"] * Gamma1["XU"] * Gamma2["vwzy"];
        Alpha += 2.0 * V_["eZvU"] * T2_["yXeW"] * Gamma1["XU"] * Gamma2["vWyZ"];

        Alpha -= 1.0 * V_["ezuv"] * T2_["xyew"] * Gamma1["xw"] * Gamma2["uvzy"];
        Alpha += 2.0 * V_["eZuV"] * T2_["xyew"] * Gamma1["xw"] * Gamma2["uVyZ"];
        Alpha += 1.0 * V_["EZUV"] * T2_["xYwE"] * Gamma1["xw"] * Gamma2["UVZY"];
        Alpha -= 2.0 * V_["zEuV"] * T2_["xYwE"] * Gamma1["xw"] * Gamma2["uVzY"];
        Alpha += 1.0 * V_["ezuv"] * T2_["yXeW"] * Gamma1["XW"] * Gamma2["uvzy"];
        Alpha -= 2.0 * V_["eZuV"] * T2_["yXeW"] * Gamma1["XW"] * Gamma2["uVyZ"];
        Alpha -= 1.0 * V_["EZUV"] * T2_["XYEW"] * Gamma1["XW"] * Gamma2["UVZY"];
        Alpha += 2.0 * V_["zEuV"] * T2_["XYEW"] * Gamma1["XW"] * Gamma2["uVzY"];

        Alpha -= 6 * V_["ezuv"] * T2_["xyew"] * Gamma1["uz"] * Gamma1["xv"] * Gamma1["yw"];
        Alpha -= 6 * V_["ezuv"] * T2_["xYeW"] * Gamma1["uz"] * Gamma1["xv"] * Gamma1["YW"];
        Alpha += 6 * V_["eZvU"] * T2_["xyew"] * Gamma1["UZ"] * Gamma1["xv"] * Gamma1["yw"];
        Alpha += 6 * V_["eZvU"] * T2_["xYeW"] * Gamma1["UZ"] * Gamma1["xv"] * Gamma1["YW"];
        Alpha -= 6 * V_["EZUV"] * T2_["XYEW"] * Gamma1["UZ"] * Gamma1["XV"] * Gamma1["YW"];
        Alpha -= 6 * V_["EZUV"] * T2_["yXwE"] * Gamma1["UZ"] * Gamma1["XV"] * Gamma1["yw"];
        Alpha += 6 * V_["zEuV"] * T2_["XYEW"] * Gamma1["uz"] * Gamma1["XV"] * Gamma1["YW"];
        Alpha += 6 * V_["zEuV"] * T2_["yXwE"] * Gamma1["uz"] * Gamma1["XV"] * Gamma1["yw"];

        Alpha -= 3.0 * V_["ezuv"] * T2_["xyew"] * Gamma1["wz"] * Gamma1["xu"] * Gamma1["yv"];
        Alpha -= 6.0 * V_["eZuV"] * T2_["xYeW"] * Gamma1["WZ"] * Gamma1["xu"] * Gamma1["YV"];
        Alpha -= 3.0 * V_["EZUV"] * T2_["XYEW"] * Gamma1["WZ"] * Gamma1["XU"] * Gamma1["YV"];
        Alpha -= 6.0 * V_["zEuV"] * T2_["xYwE"] * Gamma1["wz"] * Gamma1["xu"] * Gamma1["YV"];
    }

    Alpha += Z["mn"] * V["m,v1,n,u1"] * Gamma1["u1,v1"];
    Alpha += Z["mn"] * V["m,V1,n,U1"] * Gamma1["U1,V1"];
    Alpha += Z["MN"] * V["M,V1,N,U1"] * Gamma1["U1,V1"];
    Alpha += Z["MN"] * V["v1,M,u1,N"] * Gamma1["u1,v1"];

    Alpha += Z["ef"] * V["e,v1,f,u1"] * Gamma1["u1,v1"];
    Alpha += Z["ef"] * V["e,V1,f,U1"] * Gamma1["U1,V1"];
    Alpha += Z["EF"] * V["E,V1,F,U1"] * Gamma1["U1,V1"];
    Alpha += Z["EF"] * V["v1,E,u1,F"] * Gamma1["u1,v1"];

    Alpha += temp3["uv"] * V["u,v1,v,u1"] * Gamma1["u1,v1"];
    Alpha += temp3["uv"] * V["u,V1,v,U1"] * Gamma1["U1,V1"];
    Alpha += temp3["UV"] * V["U,V1,V,U1"] * Gamma1["U1,V1"];
    Alpha += temp3["UV"] * V["v1,U,u1,V"] * Gamma1["u1,v1"];

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

    auto ck_vc_a = ambit::Tensor::build(ambit::CoreTensor, "ci equations z{vc} alpha part", {ndets, nvirt_, ncore_});
    auto ck_ca_a = ambit::Tensor::build(ambit::CoreTensor, "ci equations z{ca} alpha part", {ndets, ncore_, na_});
    auto ck_va_a = ambit::Tensor::build(ambit::CoreTensor, "ci equations z{va} alpha part", {ndets, nvirt_, na_});
    auto ck_aa_a = ambit::Tensor::build(ambit::CoreTensor, "ci equations z{aa} alpha part", {ndets, na_, na_});

    auto ck_vc_b = ambit::Tensor::build(ambit::CoreTensor, "ci equations z{vc} beta part", {ndets, nvirt_, ncore_});
    auto ck_ca_b = ambit::Tensor::build(ambit::CoreTensor, "ci equations z{ca} beta part", {ndets, ncore_, na_});
    auto ck_va_b = ambit::Tensor::build(ambit::CoreTensor, "ci equations z{va} beta part", {ndets, nvirt_, na_});
    auto ck_aa_b = ambit::Tensor::build(ambit::CoreTensor, "ci equations z{aa} beta part", {ndets, na_, na_});

    // virtual-core
    ck_vc_a("Kem") += 2 * V.block("cava")("mveu") * cc1a_n("Kuv");
    ck_vc_a("Kem") += 2 * V.block("cAvA")("mVeU") * cc1b_n("KUV");
    ck_vc_a("Kem") += 2 * V.block("cava")("mveu") * cc1a_r("Kuv");
    ck_vc_a("Kem") += 2 * V.block("cAvA")("mVeU") * cc1b_r("KUV");

    ck_vc_b("KEM") += 2 * V.block("CAVA")("MVEU") * cc1b_n("KUV");
    ck_vc_b("KEM") += 2 * V.block("aCaV")("vMuE") * cc1a_n("Kuv");
    ck_vc_b("KEM") += 2 * V.block("CAVA")("MVEU") * cc1b_r("KUV");
    ck_vc_b("KEM") += 2 * V.block("aCaV")("vMuE") * cc1a_r("Kuv");

    // contribution from Alpha
    ck_vc_a("Kem") += -4 * ci("K") * V.block("cava")("myex") * Gamma1.block("aa")("xy");
    ck_vc_a("Kem") += -4 * ci("K") * V.block("cAvA")("mYeX") * Gamma1.block("AA")("XY");

    ck_vc_b("KEM") += -4 * ci("K") * V.block("CAVA")("MYEX") * Gamma1.block("AA")("XY");
    ck_vc_b("KEM") += -4 * ci("K") * V.block("aCaV")("yMxE") * Gamma1.block("aa")("xy");

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
    ck_ca_a("Knu") += -4 * ci("K") * V.block("aaca")("uynx") * Gamma1.block("aa")("xy");
    ck_ca_a("Knu") += -4 * ci("K") * V.block("aAcA")("uYnX") * Gamma1.block("AA")("XY");
    ck_ca_a("Knu") +=  4 * ci("K") * H.block("ac")("vn") * Gamma1.block("aa")("uv");
    ck_ca_a("Knu") +=  4 * ci("K") * V_N_Alpha.block("ac")("vn") * Gamma1.block("aa")("uv");
    ck_ca_a("Knu") +=  4 * ci("K") * V_N_Beta.block("ac")("vn") * Gamma1.block("aa")("uv");
    ck_ca_a("Knu") +=  2 * ci("K") * V.block("aaca")("xynv") * Gamma2.block("aaaa")("uvxy");
    ck_ca_a("Knu") +=  4 * ci("K") * V.block("aAcA")("xYnV") * Gamma2.block("aAaA")("uVxY");

    // NOTICE beta
    ck_ca_b("KNU") += -4 * ci("K") * V.block("AACA")("UYNX") * Gamma1.block("AA")("XY");
    ck_ca_b("KNU") += -4 * ci("K") * V.block("aAaC")("yUxN") * Gamma1.block("aa")("xy");
    ck_ca_b("KNU") +=  4 * ci("K") * H.block("AC")("VN") * Gamma1.block("AA")("UV");
    ck_ca_b("KNU") +=  4 * ci("K") * V_all_Beta.block("AC")("VN") * Gamma1.block("AA")("UV");
    ck_ca_b("KNU") +=  4 * ci("K") * V_R_Beta.block("AC")("VN") * Gamma1.block("AA")("UV");
    ck_ca_b("KNU") +=  2 * ci("K") * V.block("AACA")("XYNV") * Gamma2.block("AAAA")("UVXY");
    ck_ca_b("KNU") +=  4 * ci("K") * V.block("aAaC")("xYvN") * Gamma2.block("aAaA")("vUxY");

    // virtual-active
    ck_va_a("Keu") += 2 * H.block("av")("ve") * cc1a_n("Kuv");
    ck_va_a("Keu") += 2 * V_N_Alpha.block("av")("ve") * cc1a_n("Kuv");
    ck_va_a("Keu") += 2 * V_N_Beta.block("av")("ve")  * cc1a_n("Kuv");
    ck_va_a("Keu") += V.block("aava")("xyev") * cc2aa_n("Kuvxy");
    ck_va_a("Keu") += 2 * V.block("aAvA")("xYeV") * cc2ab_n("KuVxY");
    ck_va_a("Keu") += 2 * H.block("av")("ve") * cc1a_r("Kuv");
    ck_va_a("Keu") += 2 * V_N_Alpha.block("av")("ve") * cc1a_r("Kuv");
    ck_va_a("Keu") += 2 * V_N_Beta.block("av")("ve")  * cc1a_r("Kuv");
    ck_va_a("Keu") += V.block("aava")("xyev") * cc2aa_r("Kuvxy");
    ck_va_a("Keu") += 2 * V.block("aAvA")("xYeV") * cc2ab_r("KuVxY");

    // NOTICE beta
    ck_va_b("KEU") += 2 * H.block("AV")("VE") * cc1b_n("KUV");
    ck_va_b("KEU") += 2 * V_all_Beta.block("AV")("VE") * cc1b_n("KUV");
    ck_va_b("KEU") += 2 * V_R_Beta.block("AV")("VE")   * cc1b_n("KUV");
    ck_va_b("KEU") += V.block("AAVA")("XYEV") * cc2bb_n("KUVXY");
    ck_va_b("KEU") += 2 * V.block("aAaV")("xYvE") * cc2ab_n("KvUxY");
    ck_va_b("KEU") += 2 * H.block("AV")("VE") * cc1b_r("KUV");
    ck_va_b("KEU") += 2 * V_all_Beta.block("AV")("VE") * cc1b_r("KUV");
    ck_va_b("KEU") += 2 * V_R_Beta.block("AV")("VE")   * cc1b_r("KUV");
    ck_va_b("KEU") += V.block("AAVA")("XYEV") * cc2bb_r("KUVXY");
    ck_va_b("KEU") += 2 * V.block("aAaV")("xYvE") * cc2ab_r("KvUxY");

    /// contribution from Alpha
    ck_va_a("Keu") += -4 * ci("K") * H.block("av")("ve") * Gamma1.block("aa")("uv");
    ck_va_a("Keu") += -4 * ci("K") * V_N_Alpha.block("av")("ve") * Gamma1.block("aa")("uv");
    ck_va_a("Keu") += -4 * ci("K") * V_N_Beta.block("av")("ve") * Gamma1.block("aa")("uv");
    ck_va_a("Keu") += -2 * ci("K") * V.block("aava")("xyev") * Gamma2.block("aaaa")("uvxy");
    ck_va_a("Keu") += -4 * ci("K") * V.block("aAvA")("xYeV") * Gamma2.block("aAaA")("uVxY");

    // NOTICE beta
    ck_va_b("KEU") += -4 * ci("K") * H.block("AV")("VE") * Gamma1.block("AA")("UV");
    ck_va_b("KEU") += -4 * ci("K") * V_all_Beta.block("AV")("VE") * Gamma1.block("AA")("UV");
    ck_va_b("KEU") += -4 * ci("K") * V_R_Beta.block("AV")("VE") * Gamma1.block("AA")("UV");
    ck_va_b("KEU") += -2 * ci("K") * V.block("AAVA")("XYEV") * Gamma2.block("AAAA")("UVXY");
    ck_va_b("KEU") += -4 * ci("K") * V.block("aAaV")("xYvE") * Gamma2.block("aAaA")("vUxY");

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
    ck_aa_a("Kuv") += -2 * ci("K") * V.block("aaaa")("uyvx") * Gamma1.block("aa")("xy");
    ck_aa_a("Kuv") += -2 * ci("K") * V.block("aAaA")("uYvX") * Gamma1.block("AA")("XY");

    // NOTICE beta
    ck_aa_b("KUV") += -2 * ci("K") * V.block("AAAA")("UYVX") * Gamma1.block("AA")("XY");
    ck_aa_b("KUV") += -2 * ci("K") * V.block("aAaA")("yUxV") * Gamma1.block("aa")("xy");

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
    auto one_ci = ambit::Tensor::build(ambit::CoreTensor, "one", {ndets, ndets});

    I_ci.iterate([&](const std::vector<size_t>& i, double& value) {
        value = (i[0] == i[1]) ? 1.0 : 0.0;
    });
    one_ci.iterate([&](const std::vector<size_t>& i, double& value) {
        value = 1.0;
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
            value = Z.block("vc").data()[i[0] * ncore_ + i[1]];
        });
    }
    for (const std::string& block : {"CA"}) {
        (Z.block(block)).iterate([&](const std::vector<size_t>& i, double& value) {
            value = Z.block("ca").data()[i[0] * na_ + i[1]];
        });
    } 
    for (const std::string& block : {"VA"}) {
        (Z.block(block)).iterate([&](const std::vector<size_t>& i, double& value) {
            value = Z.block("va").data()[i[0] * na_ + i[1]];
        });
    } 

    for (const std::string& block : {"AA"}) {
        (Z.block(block)).iterate([&](const std::vector<size_t>& i, double& value) {
            value = Z.block("aa").data()[i[0] * na_ + i[1]];
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

    SharedMatrix L(new Matrix("Lagrangian", nirrep_, irrep_vec, irrep_vec));

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

        (W_.block(block)).iterate([&](const std::vector<size_t>& i, double& value) {
            if (spin_pair[0][i[0]].first == spin_pair[1][i[1]].first) {
                L->add(spin_pair[0][i[0]].first, spin_pair[0][i[0]].second,
                       spin_pair[1][i[1]].second, value);
            }
        });
    }

    L->back_transform(ints_->Ca());
    ints_->wfn()->set_Lagrangian(SharedMatrix(new Matrix("Lagrangian", nirrep_, irrep_vec, irrep_vec)));
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
    SharedMatrix D1(new Matrix("1rdm coefficients contribution", nirrep_, irrep_vec, irrep_vec));

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
    temp["nv"] -= Z["un"] * Gamma1["uv"];

    (temp.block("ca")).iterate([&](const std::vector<size_t>& i, double& value) {
        if (core_mos_relative[i[0]].first == actv_mos_relative[i[1]].first) {
            D1->set(core_mos_relative[i[0]].first, core_mos_relative[i[0]].second,
                actv_mos_relative[i[1]].second, value);
            D1->set(core_mos_relative[i[0]].first, actv_mos_relative[i[1]].second,
                core_mos_relative[i[0]].second, value);
        }
    });

    temp = BTF_->build(CoreTensor, "temporal tensor", {"va"});
    temp["ev"] = Z["eu"] * Gamma1["uv"];

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

    (Sigma3.block("aa")).iterate([&](const std::vector<size_t>& i, double& value) {
        if (actv_mos_relative[i[0]].first == actv_mos_relative[i[1]].first) {
            D1->add(actv_mos_relative[i[0]].first, actv_mos_relative[i[0]].second,
                actv_mos_relative[i[1]].second, 0.5 * value);
            D1->add(actv_mos_relative[i[0]].first, actv_mos_relative[i[1]].second,
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

    (Xi3.block("aa")).iterate([&](const std::vector<size_t>& i, double& value) {
        if (actv_mos_relative[i[0]].first == actv_mos_relative[i[1]].first) {
            D1->add(actv_mos_relative[i[0]].first, actv_mos_relative[i[0]].second,
                actv_mos_relative[i[1]].second, 0.5 * value);
            D1->add(actv_mos_relative[i[0]].first, actv_mos_relative[i[1]].second,
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
    auto tp = ambit::Tensor::build(ambit::CoreTensor, "temporal tensor", {na_, na_});

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
    // TODO: write spin_dependent two-RDMs coefficients using IWL
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

    for (size_t i = 0, size_c = core_all_.size(); i < size_c; ++i) {
        auto m = core_all_[i];
        for (size_t a = 0, size_v = virt_all_.size(); a < size_v; ++a) {
            auto e = virt_all_[a];
            auto idx = a * ncore_ + i;
            auto z_a = Z.block("vc").data()[idx] + temp.block("vc").data()[idx];
            auto z_b = Z.block("VC").data()[idx] + temp.block("VC").data()[idx];
            for (size_t j = 0; j < size_c; ++j) {
                auto m1 = core_all_[j];
                
                d2aa.write_value(m, e, m1, m1, z_a, 0, "NULL", 0);
                d2bb.write_value(m, e, m1, m1, z_b, 0, "NULL", 0);
                d2aa.write_value(m, m1, m1, e, -z_a, 0, "NULL", 0);
                d2bb.write_value(m, m1, m1, e, -z_b, 0, "NULL", 0); 
                d2ab.write_value(m, e, m1, m1, 2.0 * (z_a + z_b), 0, "NULL", 0);
            }
        }
    }

    temp = BTF_->build(CoreTensor, "temporal tensor", {"ac", "AC"});
    temp["un"] = Z["un"];
    temp["un"] -= Z["vn"] * Gamma1["uv"];
    temp["UN"] = Z["UN"];
    temp["UN"] -= Z["VN"] * Gamma1["UV"];
    // <[F, T2]>
    temp["un"] += 0.5 * Sigma3["nu"];
    temp["UN"] += 0.5 * Sigma3["NU"];
    // <[V, T1]>
    temp["un"] += 0.5 * Xi3["nu"];
    temp["UN"] += 0.5 * Xi3["NU"];

    for (size_t i = 0, size_c = core_all_.size(); i < size_c; ++i) {
        auto n = core_all_[i];
        for (size_t a = 0, size_a = actv_all_.size(); a < size_a; ++a) {
            auto u = actv_all_[a];
            auto idx = a * ncore_ + i;
            auto z_a = temp.block("ac").data()[idx];
            auto z_b = temp.block("AC").data()[idx];
            for (size_t j = 0; j < size_c; ++j) {
                auto m1 = core_all_[j];              
                if (n != m1) {
                    d2aa.write_value(u, n, m1, m1, z_a, 0, "NULL", 0);
                    d2bb.write_value(u, n, m1, m1, z_b, 0, "NULL", 0);
                    d2aa.write_value(u, m1, m1, n, -z_a, 0, "NULL", 0);
                    d2bb.write_value(u, m1, m1, n, -z_b, 0, "NULL", 0);
                }
                d2ab.write_value(u, n, m1, m1, 2.0 * (z_a + z_b), 0, "NULL", 0);
            }
        }
    }

    temp = BTF_->build(CoreTensor, "temporal tensor", {"va", "VA"});
    temp["ev"] = Z["eu"] * Gamma1["uv"];
    temp["EV"] = Z["EU"] * Gamma1["UV"];
    // <[F, T2]>
    temp["ev"] += 0.5 * Sigma3["ve"];
    temp["EV"] += 0.5 * Sigma3["VE"];
    // <[V, T1]>
    temp["ev"] += 0.5 * Xi3["ve"];
    temp["EV"] += 0.5 * Xi3["VE"];

    for (size_t i = 0, size_a = actv_all_.size(); i < size_a; ++i) {
        auto v = actv_all_[i];
        for (size_t a = 0, size_v = virt_all_.size(); a < size_v; ++a) {
            auto e = virt_all_[a];
            auto idx = a * na_ + i;
            auto z_a = temp.block("va").data()[idx];
            auto z_b = temp.block("VA").data()[idx];
            for (size_t j = 0, size_c = core_all_.size(); j < size_c; ++j) {
                auto m1 = core_all_[j];
                
                d2aa.write_value(v, e, m1, m1, z_a, 0, "NULL", 0);
                d2bb.write_value(v, e, m1, m1, z_b, 0, "NULL", 0);
                d2aa.write_value(v, m1, m1, e, -z_a, 0, "NULL", 0);
                d2bb.write_value(v, m1, m1, e, -z_b, 0, "NULL", 0);  
                d2ab.write_value(v, e, m1, m1, 2.0 * (z_a + z_b), 0, "NULL", 0);
            }
        }
    }

    for (size_t i = 0, size_c = core_all_.size(); i < size_c; ++i) {
        auto n = core_all_[i];
        for (size_t k = 0; k < size_c; ++k) {
            auto m = core_all_[k];
            auto idx = k * ncore_ + i;
            auto z_a = Z.block("cc").data()[idx];
            auto z_b = Z.block("CC").data()[idx];
            for (size_t j = 0; j < size_c; ++j) {
                auto m1 = core_all_[j];
                double v1 = 0.5 * z_a, v2 = 0.5 * z_b, v3 = z_a + z_b;

                if (m == n) {
                    v1 += 0.25;
                    v2 += 0.25;
                    v3 += 1.00;
                }

                if (m != m1) {
                    d2aa.write_value(n, m, m1, m1,  v1, 0, "NULL", 0);
                    d2bb.write_value(n, m, m1, m1,  v2, 0, "NULL", 0);
                    d2aa.write_value(n, m1, m1, m, -v1, 0, "NULL", 0);
                    d2bb.write_value(n, m1, m1, m, -v2, 0, "NULL", 0);
                }  
                d2ab.write_value(n, m, m1, m1, v3, 0, "NULL", 0);
            }
        }
    }

    auto ci_g1_a = ambit::Tensor::build(ambit::CoreTensor, "effective alpha gamma tensor", {na_, na_});
    auto ci_g1_b = ambit::Tensor::build(ambit::CoreTensor, "effective beta gamma tensor", {na_, na_});

    ci_g1_a("uv") += 0.5 * x_ci("I") * cc1a_n("Iuv");
    ci_g1_a("uv") += 0.5 * x_ci("J") * cc1a_r("Juv");
    ci_g1_b("UV") += 0.5 * x_ci("I") * cc1b_n("IUV");
    ci_g1_b("UV") += 0.5 * x_ci("J") * cc1b_r("JUV");

    temp = BTF_->build(CoreTensor, "temporal tensor", {"aa", "AA"});
    // <[F, T2]>
    temp["uv"] += Sigma3["vu"];
    temp["UV"] += Sigma3["VU"];
    // <[V, T1]>
    temp["uv"] += Xi3["vu"];
    temp["UV"] += Xi3["VU"];

    for (size_t i = 0, size_a = actv_all_.size(); i < size_a; ++i) {
        auto v = actv_all_[i];
        for (size_t k = 0; k < size_a; ++k) {
            auto u = actv_all_[k];
            auto idx = k * na_ + i;
            auto z_a = Z.block("aa").data()[idx];
            auto z_b = Z.block("AA").data()[idx];
            auto gamma_a = Gamma1_.block("aa").data()[idx];
            auto gamma_b = Gamma1_.block("AA").data()[idx];
            auto ci_gamma_a = ci_g1_a.data()[idx];
            auto ci_gamma_b = ci_g1_b.data()[idx];
            auto sigxi_a = temp.block("aa").data()[idx];
            auto sigxi_b = temp.block("AA").data()[idx];
            auto v1 = z_a + gamma_a + ci_gamma_a + sigxi_a;
            auto v2 = z_b + gamma_b + ci_gamma_b + sigxi_b;

            for (size_t j = 0, size_c = core_all_.size(); j < size_c; ++j) {
                auto m1 = core_all_[j];
                
                d2aa.write_value(v, u, m1, m1, 0.5 * v1, 0, "NULL", 0);
                d2bb.write_value(v, u, m1, m1, 0.5 * v2, 0, "NULL", 0);
                d2aa.write_value(v, m1, m1, u, -0.5 * v1, 0, "NULL", 0);
                d2bb.write_value(v, m1, m1, u, -0.5 * v2, 0, "NULL", 0);   
                d2ab.write_value(v, u, m1, m1, (v1 + v2), 0, "NULL", 0);
            }
        }
    }

    for (size_t i = 0, size_v = virt_all_.size(); i < size_v; ++i) {
        auto f = virt_all_[i];
        for (size_t k = 0; k < size_v; ++k) {
            auto e = virt_all_[k];
            auto idx = k * nvirt_ + i;
            auto z_a = Z.block("vv").data()[idx];
            auto z_b = Z.block("VV").data()[idx];
            for (size_t j = 0, size_c = core_all_.size(); j < size_c; ++j) {
                auto m1 = core_all_[j];
                
                d2aa.write_value(f, e, m1, m1, 0.5 * z_a, 0, "NULL", 0);
                d2bb.write_value(f, e, m1, m1, 0.5 * z_b, 0, "NULL", 0);
                d2aa.write_value(f, m1, m1, e, -0.5 * z_a, 0, "NULL", 0);
                d2bb.write_value(f, m1, m1, e, -0.5 * z_b, 0, "NULL", 0);    
                d2ab.write_value(f, e, m1, m1, (z_a + z_b), 0, "NULL", 0);
            }
        }
    }

    for (size_t i = 0, size_c = core_all_.size(); i < size_c; ++i) {
        auto n = core_all_[i];
        for (size_t j = 0; j < size_c; ++j) {
            auto m = core_all_[j];
            auto idx = j * ncore_ + i;
            auto z_a = Z.block("cc").data()[idx];
            auto z_b = Z.block("CC").data()[idx];
            for (size_t k = 0, size_a = actv_all_.size(); k < size_a; ++k) {
                auto v1 = actv_all_[k];
                for (size_t l = 0; l < size_a; ++l) {
                    auto u1 = actv_all_[l];
                    auto idx1 = l * na_ + k;
                    auto g_a = Gamma1.block("aa").data()[idx1];
                    auto g_b = Gamma1.block("AA").data()[idx1];
                
                    d2aa.write_value(n, m, v1, u1, 0.5 * z_a * g_a, 0, "NULL", 0);
                    d2bb.write_value(n, m, v1, u1, 0.5 * z_b * g_b, 0, "NULL", 0);
                    d2aa.write_value(n, u1, v1, m, -0.5 * z_a * g_a, 0, "NULL", 0);
                    d2bb.write_value(n, u1, v1, m, -0.5 * z_b * g_b, 0, "NULL", 0);
                    d2ab.write_value(n, m, v1, u1, (z_a * g_b + z_b * g_a), 0, "NULL", 0);
                }
            }
        }
    }

    for (size_t i = 0, size_v = virt_all_.size(); i < size_v; ++i) {
        auto f = virt_all_[i];
        for (size_t j = 0; j < size_v; ++j) {
            auto e = virt_all_[j];
            auto idx = j * nvirt_ + i;
            auto z_a = Z.block("vv").data()[idx];
            auto z_b = Z.block("VV").data()[idx];
            for (size_t k = 0, size_a = actv_all_.size(); k < size_a; ++k) {
                auto v1 = actv_all_[k];
                for (size_t l = 0; l < size_a; ++l) {
                    auto u1 = actv_all_[l];
                    auto idx1 = l * na_ + k;
                    auto g_a = Gamma1.block("aa").data()[idx1];
                    auto g_b = Gamma1.block("AA").data()[idx1];
                
                    d2aa.write_value(f, e, v1, u1, 0.5 * z_a * g_a, 0, "NULL", 0);
                    d2bb.write_value(f, e, v1, u1, 0.5 * z_b * g_b, 0, "NULL", 0);
                    d2aa.write_value(f, u1, v1, e, -0.5 * z_a * g_a, 0, "NULL", 0);
                    d2bb.write_value(f, u1, v1, e, -0.5 * z_b * g_b, 0, "NULL", 0);
                    d2ab.write_value(f, e, v1, u1, (z_a * g_b + z_b * g_a), 0, "NULL", 0);
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

    temp["xynv"] -= Z["un"] * Gamma2["uvxy"]; 
    temp["XYNV"] -= Z["UN"] * Gamma2["UVXY"]; 
    temp["xYnV"] -= Z["un"] * Gamma2["uVxY"]; 

    temp["evxy"] += Z["eu"] * Gamma2["uvxy"];
    temp["EVXY"] += Z["EU"] * Gamma2["UVXY"];
    temp["eVxY"] += Z["eu"] * Gamma2["uVxY"];

    // CASSCF reference
    temp["xyuv"] += 0.25 * Gamma2["uvxy"];
    temp["XYUV"] += 0.25 * Gamma2["UVXY"];
    temp["xYuV"] += 0.25 * Gamma2["uVxY"];

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

    temp["eumv"] += 2.0 * Z["em"] * Gamma1["uv"];
    temp["EUMV"] += 2.0 * Z["EM"] * Gamma1["UV"];
    temp["eUmV"] += 2.0 * Z["em"] * Gamma1["UV"];

    temp["u,v1,n,u1"] += 2.0 * Z["un"] * Gamma1["u1,v1"]; 
    temp["U,V1,N,U1"] += 2.0 * Z["UN"] * Gamma1["U1,V1"];
    temp["u,V1,n,U1"] += 2.0 * Z["un"] * Gamma1["U1,V1"]; 

    temp["v,v1,u,u1"] += Z["uv"] * Gamma1["u1,v1"];
    temp["V,V1,U,U1"] += Z["UV"] * Gamma1["U1,V1"];
    temp["v,V1,u,U1"] += Z["uv"] * Gamma1["U1,V1"];

    // <[F, T2]>
    if (X5_TERM||X7_TERM) {
        temp["aviu"] += Sigma3["ia"] * Gamma1["uv"];
        temp["AVIU"] += Sigma3["IA"] * Gamma1["UV"];
        temp["aViU"] += Sigma3["ia"] * Gamma1["UV"];
    }
    // <[V, T1]>
    if (X6_TERM||X7_TERM) {
        temp["aviu"] += Xi3["ia"] * Gamma1["uv"];
        temp["AVIU"] += Xi3["IA"] * Gamma1["UV"];
        temp["aViU"] += Xi3["ia"] * Gamma1["UV"];
    }

    // all-alpha and all-beta
    temp2["ckdl"] += temp["cdkl"];
    temp2["cldk"] -= temp["cdkl"];
    // alpha-beta
    temp2["ckDL"] += 2.0 * temp["cDkL"];

    temp2.iterate([&](const std::vector<size_t>& i, const std::vector<SpinType>& spin, double& value) {
        if (std::fabs(value) > 1e-12) {
            if (spin[2] == AlphaSpin) {
                d2aa.write_value(i[0], i[1], i[2], i[3], 0.5 * value, 0, "NULL", 0);          
                d2bb.write_value(i[0], i[1], i[2], i[3], 0.5 * value, 0, "NULL", 0);          
            }
            else {
                d2ab.write_value(i[0], i[1], i[2], i[3], value, 0, "NULL", 0); 
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