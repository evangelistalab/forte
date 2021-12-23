/**
 * Write 1- and 2-RDMs and back-transform the TPDM.
 */
#include "psi4/libpsi4util/PsiOutStream.h"
#include "psi4/libiwl/iwl.hpp"
#include "psi4/libmints/matrix.h"
#include "psi4/libmints/wavefunction.h"
#include "psi4/psifiles.h"
#include "../dsrg_mrpt2.h"
#include "helpers/timer.h"
#include "gradient_tpdm/backtransform_tpdm.h"

using namespace ambit;
using namespace psi;

namespace forte {

void DSRG_MRPT2::write_lagrangian() {
    // NOTICE: write the Lagrangian
    outfile->Printf("\n    Writing EWDM (Lagrangian) ....................... ");

    SharedMatrix L(new Matrix("Lagrangian", nirrep, irrep_vec, irrep_vec));

    for (const std::string& block : {"cc", "CC", "aa", "AA", "ca", "ac", "CA", "AC", "vv", "VV",
                                     "av", "cv", "va", "vc", "AV", "CV", "VA", "VC"}) {
        std::vector<std::vector<std::pair<unsigned long, unsigned long>,
                                std::allocator<std::pair<unsigned long, unsigned long>>>>
            spin_pair;
        for (size_t idx : {0, 1}) {
            auto spin = std::tolower(block.at(idx));
            if (spin == 'c') {
                spin_pair.push_back(core_mos_relative);
            } else if (spin == 'a') {
                spin_pair.push_back(actv_mos_relative);
            } else if (spin == 'v') {
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
    ints_->wfn()->set_lagrangian(
        SharedMatrix(new Matrix("Lagrangian", nirrep, irrep_vec, irrep_vec)));
    ints_->wfn()->Lagrangian()->copy(L);

    outfile->Printf("Done");
}

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

    // <[F, T2]> and <[V, T1]>
    (sigma3_xi3.block("ca")).iterate([&](const std::vector<size_t>& i, double& value) {
        if (core_mos_relative[i[0]].first == actv_mos_relative[i[1]].first) {
            D1->add(core_mos_relative[i[0]].first, core_mos_relative[i[0]].second,
                    actv_mos_relative[i[1]].second, 0.5 * value);
            D1->add(core_mos_relative[i[0]].first, actv_mos_relative[i[1]].second,
                    core_mos_relative[i[0]].second, 0.5 * value);
        }
    });

    (sigma3_xi3.block("cv")).iterate([&](const std::vector<size_t>& i, double& value) {
        if (core_mos_relative[i[0]].first == virt_mos_relative[i[1]].first) {
            D1->add(core_mos_relative[i[0]].first, core_mos_relative[i[0]].second,
                    virt_mos_relative[i[1]].second, 0.5 * value);
            D1->add(core_mos_relative[i[0]].first, virt_mos_relative[i[1]].second,
                    core_mos_relative[i[0]].second, 0.5 * value);
        }
    });

    (sigma3_xi3.block("av")).iterate([&](const std::vector<size_t>& i, double& value) {
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

    tp("uv") = 0.5 * Gamma1_tilde.block("aa")("uv");

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

void DSRG_MRPT2::write_2rdm_spin_dependent() {
    // NOTICE: write spin_dependent two-RDMs coefficients using IWL
    outfile->Printf("\n    Writing 2RDM Coefficients ....................... ");
    auto psio_ = _default_psio_lib_;
    IWL d2aa(psio_.get(), PSIF_MO_AA_TPDM, 1.0e-14, 0, 0);
    IWL d2ab(psio_.get(), PSIF_MO_AB_TPDM, 1.0e-14, 0, 0);
    IWL d2bb(psio_.get(), PSIF_MO_BB_TPDM, 1.0e-14, 0, 0);

    BlockedTensor temp = BTF_->build(CoreTensor, "temporal tensor", {"vc", "VC"});
    // <[F, T2]> and <[V, T1]>
    temp["em"] += 0.5 * sigma3_xi3["me"];
    temp["EM"] += 0.5 * sigma3_xi3["ME"];   

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
    // <[F, T2]> and <[V, T1]>
    temp["un"] += 0.5 * sigma3_xi3["nu"];
    temp["UN"] += 0.5 * sigma3_xi3["NU"];

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
    // <[F, T2]> and <[V, T1]>
    temp["ev"] += 0.5 * sigma3_xi3["ve"];
    temp["EV"] += 0.5 * sigma3_xi3["VE"];

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
                    d2aa.write_value(n, m, m1, m1, a1, 0, "outfile", 0);
                    d2bb.write_value(n, m, m1, m1, v2, 0, "outfile", 0);
                    d2aa.write_value(n, m1, m1, m, -a1, 0, "outfile", 0);
                    d2bb.write_value(n, m1, m1, m, -v2, 0, "outfile", 0);
                }
                d2ab.write_value(n, m, m1, m1, v3, 0, "outfile", 0);
            }
        }
    }

    auto ci_g1_a =
        ambit::Tensor::build(ambit::CoreTensor, "effective alpha gamma tensor", {na, na});
    auto ci_g1_b = ambit::Tensor::build(ambit::CoreTensor, "effective beta gamma tensor", {na, na});

    ci_g1_a("uv") += 0.5 * Gamma1_tilde.block("aa")("uv");
    ci_g1_b("UV") += 0.5 * Gamma1_tilde.block("AA")("UV");

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
    BlockedTensor temp2 = BTF_->build(CoreTensor, "temporal tensor 2", {"phph", "phPH"});

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
    temp["xyuv"] += 0.125 * Gamma2_tilde["uvxy"];
    temp["XYUV"] += 0.125 * Gamma2_tilde["UVXY"];
    temp["xYuV"] += 0.125 * Gamma2_tilde["uVxY"];

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

    // <[F, T2]> and <[V, T1]>
    if (X5_TERM || X6_TERM || X7_TERM) {
        temp["aviu"] += sigma3_xi3["ia"] * Gamma1_["uv"];
        temp["AVIU"] += sigma3_xi3["IA"] * Gamma1_["UV"];
        temp["aViU"] += sigma3_xi3["ia"] * Gamma1_["UV"];
    }

    // all-alpha and all-beta
    temp2["ckdl"] += temp["cdkl"];
    temp2["cldk"] -= temp["cdkl"];
    // alpha-beta
    temp2["ckDL"] += 2.0 * temp["cDkL"];

    temp2.iterate(
        [&](const std::vector<size_t>& i, const std::vector<SpinType>& spin, double& value) {
            if (std::fabs(value) > 1e-12) {
                if (spin[2] == AlphaSpin) {
                    d2aa.write_value(i[0], i[1], i[2], i[3], 0.5 * value, 0, "outfile", 0);
                    d2bb.write_value(i[0], i[1], i[2], i[3], 0.5 * value, 0, "outfile", 0);
                } else {
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

void DSRG_MRPT2::tpdm_backtransform() {
    // Backtransform the TPDM

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

} // namespace forte