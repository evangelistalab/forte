/**
 * Set CI-related and DSRG tensors.
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

    dlamb_aa =
        ambit::Tensor::build(ambit::CoreTensor, "derivatives of Lambda2_ w.r.t. C_K alpha-alpha",
                             {ndets, na, na, na, na});
    dlamb_bb = ambit::Tensor::build(
        ambit::CoreTensor, "derivatives of Lambda2_ w.r.t. C_K beta-beta", {ndets, na, na, na, na});
    dlamb_ab =
        ambit::Tensor::build(ambit::CoreTensor, "derivatives of Lambda2_ w.r.t. C_K alpha-beta",
                             {ndets, na, na, na, na});

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

    dlamb3_aaa = ambit::Tensor::build(ambit::CoreTensor,
                                      "derivatives of Lambda3_ w.r.t. C_K alpha-alpha-alpha",
                                      {ndets, na, na, na, na, na, na});
    dlamb3_bbb =
        ambit::Tensor::build(ambit::CoreTensor, "derivatives of Lambda3_ w.r.t. C_K beta-beta-beta",
                             {ndets, na, na, na, na, na, na});
    dlamb3_aab = ambit::Tensor::build(ambit::CoreTensor,
                                      "derivatives of Lambda3_ w.r.t. C_K alpha-alpha-beta",
                                      {ndets, na, na, na, na, na, na});
    dlamb3_abb = ambit::Tensor::build(ambit::CoreTensor,
                                      "derivatives of Lambda3_ w.r.t. C_K alpha-beta-beta",
                                      {ndets, na, na, na, na, na, na});

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
    dlamb3_aaa("Kxyzuvw") +=
        8.0 * cc1a_n("Kuz") * Gamma1_.block("aa")("xv") * Gamma1_.block("aa")("yw");
    dlamb3_aaa("Kxyzuvw") +=
        8.0 * cc1a_r("Kuz") * Gamma1_.block("aa")("xv") * Gamma1_.block("aa")("yw");
    dlamb3_aaa("Kxyzuvw") +=
        8.0 * cc1a_n("Kxv") * Gamma1_.block("aa")("uz") * Gamma1_.block("aa")("yw");
    dlamb3_aaa("Kxyzuvw") +=
        8.0 * cc1a_r("Kxv") * Gamma1_.block("aa")("uz") * Gamma1_.block("aa")("yw");
    dlamb3_aaa("Kxyzuvw") +=
        8.0 * cc1a_n("Kyw") * Gamma1_.block("aa")("uz") * Gamma1_.block("aa")("xv");
    dlamb3_aaa("Kxyzuvw") +=
        8.0 * cc1a_r("Kyw") * Gamma1_.block("aa")("uz") * Gamma1_.block("aa")("xv");
    dlamb3_aaa("Kxyzuvw") +=
        4.0 * cc1a_n("Kwz") * Gamma1_.block("aa")("xu") * Gamma1_.block("aa")("yv");
    dlamb3_aaa("Kxyzuvw") +=
        4.0 * cc1a_r("Kwz") * Gamma1_.block("aa")("xu") * Gamma1_.block("aa")("yv");
    dlamb3_aaa("Kxyzuvw") +=
        4.0 * cc1a_n("Kxu") * Gamma1_.block("aa")("wz") * Gamma1_.block("aa")("yv");
    dlamb3_aaa("Kxyzuvw") +=
        4.0 * cc1a_r("Kxu") * Gamma1_.block("aa")("wz") * Gamma1_.block("aa")("yv");
    dlamb3_aaa("Kxyzuvw") +=
        4.0 * cc1a_n("Kyv") * Gamma1_.block("aa")("wz") * Gamma1_.block("aa")("xu");
    dlamb3_aaa("Kxyzuvw") +=
        4.0 * cc1a_r("Kyv") * Gamma1_.block("aa")("wz") * Gamma1_.block("aa")("xu");

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
    dlamb3_bbb("KXYZUVW") +=
        8.0 * cc1b_n("KUZ") * Gamma1_.block("AA")("XV") * Gamma1_.block("AA")("YW");
    dlamb3_bbb("KXYZUVW") +=
        8.0 * cc1b_r("KUZ") * Gamma1_.block("AA")("XV") * Gamma1_.block("AA")("YW");
    dlamb3_bbb("KXYZUVW") +=
        8.0 * cc1b_n("KXV") * Gamma1_.block("AA")("UZ") * Gamma1_.block("AA")("YW");
    dlamb3_bbb("KXYZUVW") +=
        8.0 * cc1b_r("KXV") * Gamma1_.block("AA")("UZ") * Gamma1_.block("AA")("YW");
    dlamb3_bbb("KXYZUVW") +=
        8.0 * cc1b_n("KYW") * Gamma1_.block("AA")("UZ") * Gamma1_.block("AA")("XV");
    dlamb3_bbb("KXYZUVW") +=
        8.0 * cc1b_r("KYW") * Gamma1_.block("AA")("UZ") * Gamma1_.block("AA")("XV");
    dlamb3_bbb("KXYZUVW") +=
        4.0 * cc1b_n("KWZ") * Gamma1_.block("AA")("XU") * Gamma1_.block("AA")("YV");
    dlamb3_bbb("KXYZUVW") +=
        4.0 * cc1b_r("KWZ") * Gamma1_.block("AA")("XU") * Gamma1_.block("AA")("YV");
    dlamb3_bbb("KXYZUVW") +=
        4.0 * cc1b_n("KXU") * Gamma1_.block("AA")("WZ") * Gamma1_.block("AA")("YV");
    dlamb3_bbb("KXYZUVW") +=
        4.0 * cc1b_r("KXU") * Gamma1_.block("AA")("WZ") * Gamma1_.block("AA")("YV");
    dlamb3_bbb("KXYZUVW") +=
        4.0 * cc1b_n("KYV") * Gamma1_.block("AA")("WZ") * Gamma1_.block("AA")("XU");
    dlamb3_bbb("KXYZUVW") +=
        4.0 * cc1b_r("KYV") * Gamma1_.block("AA")("WZ") * Gamma1_.block("AA")("XU");

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
                    spin_cases({"gphh", "pghh", "ppgh", "pphg", "gchc", "pghc", "pcgc", "pchg",
                                "gcpc", "hgpc", "hcgc", "hcpg", "gccc", "cgcc", "ccgc", "cccg",
                                "gcvc", "vgvc", "vcgc", "vcvg", "cgch", "gpch", "cpcg", "cpgh",
                                "cgcp", "ghcp", "chcg", "chgp", "cgcv", "gvcv", "cvcg", "cvgv"}));

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

    V_N_Alpha = BTF_->build(CoreTensor,
                            "normal Dimention-reduced Electron Repulsion Integral alpha", {"gg"});
    V_N_Beta = BTF_->build(CoreTensor, "normal Dimention-reduced Electron Repulsion Integral beta",
                           {"gg"});
    V_R_Beta = BTF_->build(
        CoreTensor, "index-reversed Dimention-reduced Electron Repulsion Integral beta", {"GG"});
    V_all_Beta = BTF_->build(
        CoreTensor, "normal Dimention-reduced Electron Repulsion Integral all beta", {"GG"});

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
    Eeps1 = BTF_->build(CoreTensor, "e^[-s*(Delta1)^2]", spin_cases({"hp"}));
    Eeps1_m1 = BTF_->build(CoreTensor, "{1-e^[-s*(Delta1)^2]}/(Delta1)", spin_cases({"hp"}));
    Eeps1_m2 = BTF_->build(CoreTensor, "{1-e^[-s*(Delta1)^2]}/(Delta1)^2", spin_cases({"hp"}));
    Eeps2 = BTF_->build(CoreTensor, "e^[-s*(Delta2)^2]", spin_cases({"hhpp"}));
    Eeps2_p = BTF_->build(CoreTensor, "1+e^[-s*(Delta2)^2]", spin_cases({"hhpp"}));
    Eeps2_m1 = BTF_->build(CoreTensor, "{1-e^[-s*(Delta2)^2]}/(Delta2)", spin_cases({"hhpp"}));
    Eeps2_m2 = BTF_->build(CoreTensor, "{1-e^[-s*(Delta2)^2]}/(Delta2)^2", spin_cases({"hhpp"}));
    Delta1 = BTF_->build(CoreTensor, "Delta1", spin_cases({"gg"}));
    Delta2 = BTF_->build(CoreTensor, "Delta2", spin_cases({"hhpp"}));
    DelGam1 = BTF_->build(CoreTensor, "Delta1 * Gamma1_", spin_cases({"aa"}));
    DelEeps1 = BTF_->build(CoreTensor, "Delta1 * Eeps1", spin_cases({"hp"}));
    T2OverDelta = BTF_->build(CoreTensor, "T2/Delta", spin_cases({"hhpp"}));

    Eeps1.iterate(
        [&](const std::vector<size_t>& i, const std::vector<SpinType>& spin, double& value) {
            if (spin[0] == AlphaSpin) {
                value = dsrg_source_->compute_renormalized(Fa_[i[0]] - Fa_[i[1]]);
            } else {
                value = dsrg_source_->compute_renormalized(Fb_[i[0]] - Fb_[i[1]]);
            }
        });
    Delta1.iterate(
        [&](const std::vector<size_t>& i, const std::vector<SpinType>& spin, double& value) {
            if (spin[0] == AlphaSpin) {
                value = Fa_[i[0]] - Fa_[i[1]];
            } else {
                value = Fb_[i[0]] - Fb_[i[1]];
            }
        });
    Delta2.iterate(
        [&](const std::vector<size_t>& i, const std::vector<SpinType>& spin, double& value) {
            if (spin[0] == AlphaSpin && spin[1] == AlphaSpin) {
                value = Fa_[i[0]] + Fa_[i[1]] - Fa_[i[2]] - Fa_[i[3]];
            } else if (spin[0] == BetaSpin && spin[1] == BetaSpin) {
                value = Fb_[i[0]] + Fb_[i[1]] - Fb_[i[2]] - Fb_[i[3]];
            } else {
                value = Fa_[i[0]] + Fb_[i[1]] - Fa_[i[2]] - Fb_[i[3]];
            }
        });
    Eeps2.iterate([&](const std::vector<size_t>& i, const std::vector<SpinType>& spin,
                      double& value) {
        if (spin[0] == AlphaSpin && spin[1] == AlphaSpin) {
            value =
                dsrg_source_->compute_renormalized(Fa_[i[0]] + Fa_[i[1]] - Fa_[i[2]] - Fa_[i[3]]);
        } else if (spin[0] == BetaSpin && spin[1] == BetaSpin) {
            value =
                dsrg_source_->compute_renormalized(Fb_[i[0]] + Fb_[i[1]] - Fb_[i[2]] - Fb_[i[3]]);
        } else {
            value =
                dsrg_source_->compute_renormalized(Fa_[i[0]] + Fb_[i[1]] - Fa_[i[2]] - Fb_[i[3]]);
        }
    });
    Eeps2_p.iterate(
        [&](const std::vector<size_t>& i, const std::vector<SpinType>& spin, double& value) {
            if (spin[0] == AlphaSpin && spin[1] == AlphaSpin) {
                value = 1.0 + dsrg_source_->compute_renormalized(Fa_[i[0]] + Fa_[i[1]] - Fa_[i[2]] -
                                                                 Fa_[i[3]]);
            } else if (spin[0] == BetaSpin && spin[1] == BetaSpin) {
                value = 1.0 + dsrg_source_->compute_renormalized(Fb_[i[0]] + Fb_[i[1]] - Fb_[i[2]] -
                                                                 Fb_[i[3]]);
            } else {
                value = 1.0 + dsrg_source_->compute_renormalized(Fa_[i[0]] + Fb_[i[1]] - Fa_[i[2]] -
                                                                 Fb_[i[3]]);
            }
        });
    Eeps2_m1.iterate(
        [&](const std::vector<size_t>& i, const std::vector<SpinType>& spin, double& value) {
            if (spin[0] == AlphaSpin && spin[1] == AlphaSpin) {
                value = dsrg_source_->compute_renormalized_denominator(Fa_[i[0]] + Fa_[i[1]] -
                                                                       Fa_[i[2]] - Fa_[i[3]]);
            } else if (spin[0] == BetaSpin && spin[1] == BetaSpin) {
                value = dsrg_source_->compute_renormalized_denominator(Fb_[i[0]] + Fb_[i[1]] -
                                                                       Fb_[i[2]] - Fb_[i[3]]);
            } else {
                value = dsrg_source_->compute_renormalized_denominator(Fa_[i[0]] + Fb_[i[1]] -
                                                                       Fa_[i[2]] - Fb_[i[3]]);
            }
        });
    Eeps2_m2.iterate(
        [&](const std::vector<size_t>& i, const std::vector<SpinType>& spin, double& value) {
            if (spin[0] == AlphaSpin && spin[1] == AlphaSpin) {
                value = dsrg_source_->compute_regularized_denominator_derivR(Fa_[i[0]] + Fa_[i[1]] -
                                                                             Fa_[i[2]] - Fa_[i[3]]);
            } else if (spin[0] == BetaSpin && spin[1] == BetaSpin) {
                value = dsrg_source_->compute_regularized_denominator_derivR(Fb_[i[0]] + Fb_[i[1]] -
                                                                             Fb_[i[2]] - Fb_[i[3]]);
            } else {
                value = dsrg_source_->compute_regularized_denominator_derivR(Fa_[i[0]] + Fb_[i[1]] -
                                                                             Fa_[i[2]] - Fb_[i[3]]);
            }
        });

    Eeps1_m1.iterate(
        [&](const std::vector<size_t>& i, const std::vector<SpinType>& spin, double& value) {
            if (spin[0] == AlphaSpin) {
                value = dsrg_source_->compute_renormalized_denominator(Fa_[i[0]] - Fa_[i[1]]);
            } else {
                value = dsrg_source_->compute_renormalized_denominator(Fb_[i[0]] - Fb_[i[1]]);
            }
        });

    Eeps1_m2.iterate(
        [&](const std::vector<size_t>& i, const std::vector<SpinType>& spin, double& value) {
            if (spin[0] == AlphaSpin) {
                value = dsrg_source_->compute_regularized_denominator_derivR(Fa_[i[0]] - Fa_[i[1]]);
            } else {
                value = dsrg_source_->compute_regularized_denominator_derivR(Fb_[i[0]] - Fb_[i[1]]);
            }
        });

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

} // namespace forte