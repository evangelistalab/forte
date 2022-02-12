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

    auto blocklabel = {"cc", "CC", "aa", "AA", "ca", "ac", "CA", "AC", "vv", "VV",
                       "av", "cv", "va", "vc", "AV", "CV", "VA", "VC"};

    for (const std::string& block : blocklabel) {
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

    // terms contracted with V["abij"]
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

void DSRG_MRPT2::write_df_rdm() {
    BlockedTensor df_2rdm = BTF_->build(tensor_type_, "df_2rdm", {"LL"});
    BlockedTensor df_3rdm = BTF_->build(tensor_type_, "df_3rdm", {"Lgg", "LGG"});

    // density terms contracted with V["abij"]
    BlockedTensor dvabij = BTF_->build(CoreTensor, "density of V['abij']", {"pphh", "PPHH", "pPhH"});

    // if (CORRELATION_TERM) {
    //     dvabij["abij"] += Tau1["ijab"];
    //     dvabij["ABIJ"] += Tau1["IJAB"];
    //     dvabij["aBiJ"] += Tau1["iJaB"];

    //     dvabij["cdkl"] += Kappa["klcd"] * Eeps2_p["klcd"];
    //     dvabij["CDKL"] += Kappa["KLCD"] * Eeps2_p["KLCD"];
    //     dvabij["cDkL"] += Kappa["kLcD"] * Eeps2_p["kLcD"];
    // }

    // dvabij["xynv"] -= Z["un"] * Gamma2_["uvxy"];
    // dvabij["XYNV"] -= Z["UN"] * Gamma2_["UVXY"];
    // dvabij["xYnV"] -= Z["un"] * Gamma2_["uVxY"];

    // dvabij["evxy"] += Z["eu"] * Gamma2_["uvxy"];
    // dvabij["EVXY"] += Z["EU"] * Gamma2_["UVXY"];
    // dvabij["eVxY"] += Z["eu"] * Gamma2_["uVxY"];

    // // CASSCF reference
    // dvabij["xyuv"] += 0.25 * Gamma2_["uvxy"];
    // dvabij["XYUV"] += 0.25 * Gamma2_["UVXY"];
    // dvabij["xYuV"] += 0.25 * Gamma2_["uVxY"];

    // // CI contribution
    // dvabij["xyuv"] += 0.125 * Gamma2_tilde["uvxy"];
    // dvabij["XYUV"] += 0.125 * Gamma2_tilde["UVXY"];
    // dvabij["xYuV"] += 0.125 * Gamma2_tilde["uVxY"];

    // dvabij["eumv"] += 2.0 * Z["em"] * Gamma1_["uv"];
    // dvabij["EUMV"] += 2.0 * Z["EM"] * Gamma1_["UV"];
    // dvabij["eUmV"] += 2.0 * Z["em"] * Gamma1_["UV"];

    // dvabij["u,a1,n,u1"] += 2.0 * Z["un"] * Gamma1_["u1,a1"];
    // dvabij["U,A1,N,U1"] += 2.0 * Z["UN"] * Gamma1_["U1,A1"];
    // dvabij["u,A1,n,U1"] += 2.0 * Z["un"] * Gamma1_["U1,A1"];

    // dvabij["v,a1,u,u1"] += Z["uv"] * Gamma1_["u1,a1"];
    // dvabij["V,A1,U,U1"] += Z["UV"] * Gamma1_["U1,A1"];
    // dvabij["v,A1,u,U1"] += Z["uv"] * Gamma1_["U1,A1"];

    // // <[F, T2]> and <[V, T1]>
    // if (X5_TERM || X6_TERM || X7_TERM) {
    //     dvabij["aviu"] += sigma3_xi3["ia"] * Gamma1_["uv"];
    //     dvabij["AVIU"] += sigma3_xi3["IA"] * Gamma1_["UV"];
    //     dvabij["aViU"] += sigma3_xi3["ia"] * Gamma1_["UV"];
    // }

    // df_2rdm["R!,S!"] +=       B["A!,i,a"] * Jm12["A!,R!"] * B["B!,j,b"] * Jm12["B!,S!"] * dvabij["abij"];
    // df_2rdm["R!,S!"] +=       B["A!,I,A"] * Jm12["A!,R!"] * B["B!,J,B"] * Jm12["B!,S!"] * dvabij["ABIJ"];
    // df_2rdm["R!,S!"] += 4.0 * B["A!,i,a"] * Jm12["A!,R!"] * B["B!,J,B"] * Jm12["B!,S!"] * dvabij["aBiJ"];
    // ! this line can be optimized as "df_2rdm_ab["S!,R!"] -= df_2rdm_ab["R!,S!"]" in the future
    // df_2rdm["R!,S!"] += 2.0 * B["A!,I,A"] * Jm12["A!,R!"] * B["B!,j,b"] * Jm12["B!,S!"] * dvabij["bAjI"];



    // df_3rdm["Q!,a,i"] += Jm12["Q!,R!"] * B["R!,b,j"] * dvabij["abij"];
    // df_3rdm["Q!,a,i"] += 2.0 * Jm12["Q!,R!"] * B["R!,B,J"] * dvabij["aBiJ"];
    // df_3rdm["Q!,A,I"] += 2.0 * Jm12["Q!,R!"] * B["R!,b,j"] * dvabij["bAjI"];
    // df_3rdm["Q!,A,I"] += Jm12["Q!,R!"] * B["R!,B,J"] * dvabij["ABIJ"];

    // df_3rdm["Q!,i,a"] += Jm12["Q!,R!"] * B["R!,j,b"] * dvabij["abij"];
    // df_3rdm["Q!,i,a"] += 2.0 * Jm12["Q!,R!"] * B["R!,J,B"] * dvabij["aBiJ"];
    // df_3rdm["Q!,I,A"] += 2.0 * Jm12["Q!,R!"] * B["R!,j,b"] * dvabij["bAjI"];
    // df_3rdm["Q!,I,A"] += Jm12["Q!,R!"] * B["R!,J,B"] * dvabij["ABIJ"];











    // df_3rdm["Q!,a,i"] += Jm12["Q!,R!"] * B["R!,b,j"] * dvabij["abij"];
    // df_3rdm["Q!,a,i"] += Jm12["Q!,R!"] * B["R!,B,J"] * dvabij["aBiJ"];
    // df_3rdm["Q!,A,I"] += Jm12["Q!,R!"] * B["R!,b,j"] * dvabij["bAjI"];
    // df_3rdm["Q!,A,I"] += Jm12["Q!,R!"] * B["R!,B,J"] * dvabij["ABIJ"];

    // df_3rdm["R!,b,j"] += Jm12["Q!,R!"] * B["Q!,a,i"] * dvabij["abij"];
    // df_3rdm["R!,B,J"] += Jm12["Q!,R!"] * B["Q!,a,i"] * dvabij["aBiJ"];
    // df_3rdm["R!,b,j"] += Jm12["Q!,R!"] * B["Q!,A,I"] * dvabij["bAjI"];
    // df_3rdm["R!,B,J"] += Jm12["Q!,R!"] * B["Q!,A,I"] * dvabij["ABIJ"];



    /************************************************************************************************/

    // BlockedTensor temp = BTF_->build(CoreTensor, "temporal tensor", {"gg", "GG"});
    // // <[F, T2]> and <[V, T1]>
    // temp["em"] += sigma3_xi3["me"];
    // temp["em"] += 2.0 * Z["em"];
    // temp["EM"] += sigma3_xi3["ME"]; 
    // temp["EM"] += 2.0 * Z["EM"];  

    // temp["un"] += 2.0 * Z["un"];
    // temp["un"] -= 2.0 * Z["vn"] * Gamma1_["uv"];
    // temp["UN"] += 2.0 * Z["UN"];
    // temp["UN"] -= 2.0 * Z["VN"] * Gamma1_["UV"];
    // // <[F, T2]> and <[V, T1]>
    // temp["un"] += sigma3_xi3["nu"];
    // temp["UN"] += sigma3_xi3["NU"];

    // temp["ev"] += 2.0 * Z["eu"] * Gamma1_["uv"];
    // temp["EV"] += 2.0 * Z["EU"] * Gamma1_["UV"];
    // // <[F, T2]> and <[V, T1]>
    // temp["ev"] += sigma3_xi3["ve"];
    // temp["EV"] += sigma3_xi3["VE"];

    // temp["mn"] += Z["mn"];
    // temp["MN"] += Z["MN"];

    // temp["uv"] += Z["uv"];
    // temp["uv"] += Gamma1_["uv"];
    // temp["uv"] += 0.5 * Gamma1_tilde["uv"];
    // temp["UV"] += Z["UV"];
    // temp["UV"] += Gamma1_["UV"];
    // temp["UV"] += 0.5 * Gamma1_tilde["UV"];

    // temp["ef"] += Z["ef"];
    // temp["EF"] += Z["EF"];

    // df_2rdm["R!,S!"] += B["A!,p,q"] * Jm12["A!,R!"] * B["B!,m,n"] * I["mn"] * Jm12["B!,S!"] * temp["pq"];
    // df_2rdm["R!,S!"] += B["A!,P,Q"] * Jm12["A!,R!"] * B["B!,M,N"] * I["MN"] * Jm12["B!,S!"] * temp["PQ"];
    // df_2rdm["R!,S!"] += B["A!,p,q"] * Jm12["A!,R!"] * B["B!,M,N"] * I["MN"] * Jm12["B!,S!"] * temp["pq"];
    // df_2rdm["R!,S!"] += B["A!,P,Q"] * Jm12["A!,R!"] * B["B!,m,n"] * I["mn"] * Jm12["B!,S!"] * temp["PQ"];



    // df_3rdm["Q!,p,r"] += Jm12["Q!,R!"] * B["R!,m,n"] * I["mn"] * temp["pr"];
    // df_3rdm["Q!,p,r"] += Jm12["Q!,R!"] * B["R!,M,N"] * I["MN"] * temp["pr"];
    // df_3rdm["Q!,P,R"] += Jm12["Q!,R!"] * B["R!,m,n"] * I["mn"] * temp["PR"];
    // df_3rdm["Q!,P,R"] += Jm12["Q!,R!"] * B["R!,M,N"] * I["MN"] * temp["PR"];

    // df_3rdm["R!,m,n"] += Jm12["Q!,R!"] * B["Q!,p,r"] * I["mn"] * temp["pr"];
    // df_3rdm["R!,M,N"] += Jm12["Q!,R!"] * B["Q!,p,r"] * I["MN"] * temp["pr"];
    // df_3rdm["R!,m,n"] += Jm12["Q!,R!"] * B["Q!,P,R"] * I["mn"] * temp["PR"];
    // df_3rdm["R!,M,N"] += Jm12["Q!,R!"] * B["Q!,P,R"] * I["MN"] * temp["PR"];



    /*************** NOTICE remember to remove ***************/

    // df_2rdm["R!,S!"] += B["A!,u,v"] * Jm12["A!,R!"] * B["B!,m,n"] * I["mn"] * Jm12["B!,S!"] * Gamma1_["uv"];
    // df_2rdm["R!,S!"] += B["A!,U,V"] * Jm12["A!,R!"] * B["B!,M,N"] * I["MN"] * Jm12["B!,S!"] * Gamma1_["UV"];
    // df_2rdm["R!,S!"] += B["A!,u,v"] * Jm12["A!,R!"] * B["B!,M,N"] * I["MN"] * Jm12["B!,S!"] * Gamma1_["uv"];
    // df_2rdm["R!,S!"] += B["A!,U,V"] * Jm12["A!,R!"] * B["B!,m,n"] * I["mn"] * Jm12["B!,S!"] * Gamma1_["UV"];
    // df_2rdm["R!,S!"] -= B["A!,u,m"] * Jm12["A!,R!"] * B["B!,m,v"] * Jm12["B!,S!"] * Gamma1_["uv"];
    // df_2rdm["R!,S!"] -= B["A!,U,M"] * Jm12["A!,R!"] * B["B!,M,V"] * Jm12["B!,S!"] * Gamma1_["UV"];


    // df_3rdm["Q!,u,v"] += Jm12["Q!,R!"] * B["R!,m,n"] * I["mn"] * Gamma1_["uv"];
    // df_3rdm["Q!,u,v"] += Jm12["Q!,R!"] * B["R!,M,N"] * I["MN"] * Gamma1_["uv"];
    // df_3rdm["Q!,U,V"] += Jm12["Q!,R!"] * B["R!,m,n"] * I["mn"] * Gamma1_["UV"];
    // df_3rdm["Q!,U,V"] += Jm12["Q!,R!"] * B["R!,M,N"] * I["MN"] * Gamma1_["UV"];
    // df_3rdm["Q!,u,m"] -= Jm12["Q!,R!"] * B["R!,m,v"] * Gamma1_["uv"];
    // df_3rdm["Q!,U,M"] -= Jm12["Q!,R!"] * B["R!,M,V"] * Gamma1_["UV"];



    // df_3rdm["R!,m,n"] += Jm12["Q!,R!"] * B["Q!,u,v"] * I["mn"] * Gamma1_["uv"];
    // df_3rdm["R!,M,N"] += Jm12["Q!,R!"] * B["Q!,u,v"] * I["MN"] * Gamma1_["uv"];
    // df_3rdm["R!,m,n"] += Jm12["Q!,R!"] * B["Q!,U,V"] * I["mn"] * Gamma1_["UV"];
    // df_3rdm["R!,M,N"] += Jm12["Q!,R!"] * B["Q!,U,V"] * I["MN"] * Gamma1_["UV"];
    // df_3rdm["R!,m,v"] -= Jm12["Q!,R!"] * B["Q!,u,m"] * Gamma1_["uv"];
    // df_3rdm["R!,M,V"] -= Jm12["Q!,R!"] * B["Q!,U,M"] * Gamma1_["UV"];








    /***************NOTICE remember to remove***************/



    /**************************** CASSCF reference ****************************/
    // Coulomb part
    df_2rdm["R!,S!"] += 0.5 * B["A!,m1,n1"] * I["m1,n1"] * Jm12["A!,R!"] * B["B!,m,n"] * I["mn"] * Jm12["B!,S!"];
    df_2rdm["R!,S!"] += 0.5 * B["A!,M1,N1"] * I["M1,N1"] * Jm12["A!,R!"] * B["B!,M,N"] * I["MN"] * Jm12["B!,S!"];
    df_2rdm["R!,S!"] += 0.5 * B["A!,m1,n1"] * I["m1,n1"] * Jm12["A!,R!"] * B["B!,M,N"] * I["MN"] * Jm12["B!,S!"];
    df_2rdm["R!,S!"] += 0.5 * B["A!,M1,N1"] * I["M1,N1"] * Jm12["A!,R!"] * B["B!,m,n"] * I["mn"] * Jm12["B!,S!"];
    // Exchange part
    df_2rdm["R!,S!"] -= 0.5 * B["A!,m1,n1"] * I["m,n1"] * Jm12["A!,R!"] * B["B!,m,n"] * I["m1,n"] * Jm12["B!,S!"];
    df_2rdm["R!,S!"] -= 0.5 * B["A!,M1,N1"] * I["M,N1"] * Jm12["A!,R!"] * B["B!,M,N"] * I["M1,N"] * Jm12["B!,S!"];

    // Coulomb part
    df_3rdm["Q!,m1,n1"] += Jm12["Q!,P!"] * B["P!,m,n"] * I["mn"] * I["m1,n1"];
    df_3rdm["Q!,m1,n1"] += Jm12["Q!,P!"] * B["P!,M,N"] * I["MN"] * I["m1,n1"];
    df_3rdm["Q!,M1,N1"] += Jm12["Q!,P!"] * B["P!,m,n"] * I["mn"] * I["M1,N1"];
    df_3rdm["Q!,M1,N1"] += Jm12["Q!,P!"] * B["P!,M,N"] * I["MN"] * I["M1,N1"];
    // Exchange part
    df_3rdm["Q!,m,n"]   -= Jm12["Q!,P!"] * B["P!,n,m"];
    df_3rdm["Q!,M,N"]   -= Jm12["Q!,P!"] * B["P!,N,M"];




    // // CASSCF reference
    // df_2rdm["R!,S!"] += Ppq["A!,m1,n1"] * I["m1,n1"] * Jm12["A!,R!"] * Ppq["B!,m,n"] * I["mn"] * Jm12["B!,S!"];
    // df_2rdm["R!,S!"] += Ppq["A!,M1,N1"] * I["M1,N1"] * Jm12["A!,R!"] * Ppq["B!,M,N"] * I["MN"] * Jm12["B!,S!"];
    // df_2rdm["R!,S!"] += Ppq["A!,m1,n1"] * I["m1,n1"] * Jm12["A!,R!"] * Ppq["B!,M,N"] * I["MN"] * Jm12["B!,S!"];
    // df_2rdm["R!,S!"] += Ppq["A!,M1,N1"] * I["M1,N1"] * Jm12["A!,R!"] * Ppq["B!,m,n"] * I["mn"] * Jm12["B!,S!"];

    // // 0.5 V["mnmn"] = 2(mm|nn) - (mn|nm)

    // df_3rdm["Q!,m1,n1"] += 2.0 * Jm12["Q!,R!"] * Ppq["R!,m,n"] * I["mn"] * I["m1,n1"];
    // df_3rdm["Q!,m1,n1"] += 2.0 * Jm12["Q!,R!"] * Ppq["R!,M,N"] * I["MN"] * I["m1,n1"];
    // df_3rdm["Q!,M1,N1"] += 2.0 * Jm12["Q!,R!"] * Ppq["R!,m,n"] * I["mn"] * I["M1,N1"];
    // df_3rdm["Q!,M1,N1"] += 2.0 * Jm12["Q!,R!"] * Ppq["R!,M,N"] * I["MN"] * I["M1,N1"];


    // // residue terms
    // df_2rdm["R!,S!"] += B["A!,m,n"] * Jm12["A!,R!"] * B["B!,u,v"] * Jm12["B!,S!"] * Z["mn"] * Gamma1_["uv"];
    // df_2rdm["R!,S!"] += B["A!,M,N"] * Jm12["A!,R!"] * B["B!,U,V"] * Jm12["B!,S!"] * Z["MN"] * Gamma1_["UV"];
    // df_2rdm["R!,S!"] += B["A!,m,n"] * Jm12["A!,R!"] * B["B!,U,V"] * Jm12["B!,S!"] * Z["mn"] * Gamma1_["UV"];
    // df_2rdm["R!,S!"] += B["A!,M,N"] * Jm12["A!,R!"] * B["B!,u,v"] * Jm12["B!,S!"] * Z["MN"] * Gamma1_["uv"];

    // df_2rdm["R!,S!"] += B["A!,e,f"] * Jm12["A!,R!"] * B["B!,u,v"] * Jm12["B!,S!"] * Z["ef"] * Gamma1_["uv"];
    // df_2rdm["R!,S!"] += B["A!,E,F"] * Jm12["A!,R!"] * B["B!,U,V"] * Jm12["B!,S!"] * Z["EF"] * Gamma1_["UV"];
    // df_2rdm["R!,S!"] += B["A!,e,f"] * Jm12["A!,R!"] * B["B!,U,V"] * Jm12["B!,S!"] * Z["ef"] * Gamma1_["UV"];
    // df_2rdm["R!,S!"] += B["A!,E,F"] * Jm12["A!,R!"] * B["B!,u,v"] * Jm12["B!,S!"] * Z["EF"] * Gamma1_["uv"];



    // df_3rdm["Q!,m,n"] += Jm12["Q!,R!"] * B["R!,u,v"] * Z["mn"] * Gamma1_["uv"];
    // df_3rdm["Q!,m,n"] += Jm12["Q!,R!"] * B["R!,U,V"] * Z["mn"] * Gamma1_["UV"];
    // df_3rdm["Q!,M,N"] += Jm12["Q!,R!"] * B["R!,u,v"] * Z["MN"] * Gamma1_["uv"];
    // df_3rdm["Q!,M,N"] += Jm12["Q!,R!"] * B["R!,U,V"] * Z["MN"] * Gamma1_["UV"];

    // df_3rdm["R!,u,v"] += Jm12["Q!,R!"] * B["Q!,m,n"] * Z["mn"] * Gamma1_["uv"];
    // df_3rdm["R!,U,V"] += Jm12["Q!,R!"] * B["Q!,m,n"] * Z["mn"] * Gamma1_["UV"];
    // df_3rdm["R!,u,v"] += Jm12["Q!,R!"] * B["Q!,M,N"] * Z["MN"] * Gamma1_["uv"];
    // df_3rdm["R!,U,V"] += Jm12["Q!,R!"] * B["Q!,M,N"] * Z["MN"] * Gamma1_["UV"];

    // df_3rdm["Q!,e,f"] += Jm12["Q!,R!"] * B["R!,u,v"] * Z["ef"] * Gamma1_["uv"];
    // df_3rdm["Q!,e,f"] += Jm12["Q!,R!"] * B["R!,U,V"] * Z["ef"] * Gamma1_["UV"];
    // df_3rdm["Q!,E,F"] += Jm12["Q!,R!"] * B["R!,u,v"] * Z["EF"] * Gamma1_["uv"];
    // df_3rdm["Q!,E,F"] += Jm12["Q!,R!"] * B["R!,U,V"] * Z["EF"] * Gamma1_["UV"];

    // df_3rdm["R!,u,v"] += Jm12["Q!,R!"] * B["Q!,e,f"] * Z["ef"] * Gamma1_["uv"];
    // df_3rdm["R!,U,V"] += Jm12["Q!,R!"] * B["Q!,e,f"] * Z["ef"] * Gamma1_["UV"];
    // df_3rdm["R!,u,v"] += Jm12["Q!,R!"] * B["Q!,E,F"] * Z["EF"] * Gamma1_["uv"];
    // df_3rdm["R!,U,V"] += Jm12["Q!,R!"] * B["Q!,E,F"] * Z["EF"] * Gamma1_["UV"];

    /******************************* Backtransform (P|pq) to (P|\mu \nu) *******************************/

    int nso = ints_->wfn()->nso();
    int nmo_matsize = nmo * nmo;
    int ao_matsize  = nso * nso;
    std::map<char, std::vector<std::pair<unsigned long, unsigned long>,
                std::allocator<std::pair<unsigned long, unsigned long>>>> idxmap;
    std::map<string, std::pair<SharedMatrix, SharedMatrix>> slicemap;
    std::map<string, int> stride_size;
    std::map<char, int> orbital_size;
    SharedMatrix M(new Matrix("backtransformed df_3rdm", naux, ao_matsize));
    idxmap = {{'c', core_mos_relative},
              {'a', actv_mos_relative},
              {'v', virt_mos_relative}};

    stride_size = {{"ca", ncore * na},    {"ac", na * ncore},
                   {"cv", ncore * nvirt}, {"vc", nvirt * ncore},
                   {"av", na * nvirt},    {"va", nvirt * na},
                   {"cc", ncore * ncore}, {"vv", nvirt * nvirt}, {"aa", na * na}};

    orbital_size = {{'c', ncore}, {'a', na}, {'v', nvirt}};
    auto blocklabels = {"cc", "aa", "ca", "ac", "vv", "av", "cv", "va", "vc"};

    std::map<char, int> pre_idx;
    pre_idx = {{'c', 0},
               {'a', ncore},
               {'v', ncore + na}};

    SharedMatrix temp_mat(new Matrix("temp_mat", nirrep, irrep_vec, irrep_vec));

    auto temp_mat_MO = std::make_shared<Matrix>("MO temp matrix", nmo, nmo);

    for(int aux_idx = 0; aux_idx < naux; ++aux_idx) {
        temp_mat_MO->zero();

        for (const std::string& block : blocklabels) {
            auto dfblk = "L" + block;
            auto stride = stride_size[block];
            const auto& block_data = df_3rdm.block(dfblk).data();
            auto label1 = block[0];
            auto label2 = block[1];
            int rowsize = orbital_size[label1];
            int colsize = orbital_size[label2];

            for (int i = 0; i < rowsize; ++i) {
                for (int j = 0; j < colsize; ++j) {
                    auto val  = block_data[aux_idx * stride + i * colsize + j];   
                    auto idx1 = pre_idx[label1] + i;
                    auto idx2 = pre_idx[label2] + j;     
                    temp_mat_MO->add(idx1, idx2, val);
                }
            }
        }

        auto Ca = ints_->Ca();
        // Copy Ca to a matrix without symmetry blocking
        auto Cat = std::make_shared<Matrix>("Ca temp matrix", nso, nmo);

        int offset = 0;
        std::vector<int> sum_nmopi(nirrep, 0);
        for (int irp = 1; irp < nirrep; ++irp) {
            sum_nmopi[irp] = sum_nmopi[irp-1] + ints_->wfn()->nmopi()[irp-1];
        }

        for (const char& label : {'c', 'a', 'v'}) {
            auto space = idxmap[label];
            for (auto irp_i : space) {
                auto irp = irp_i.first;
                auto index = irp_i.second;
                auto nsopi = ints_->wfn()->nsopi()[irp];
                auto nmopi = ints_->wfn()->nmopi()[irp];
                for (int i = 0; i < nsopi; ++i) { 
                    auto val = Ca->get(irp, i, index);
                    Cat->set(sum_nmopi[irp] + i, offset, val);
                }
                offset += 1;
            }
        }

        temp_mat_MO->back_transform(Cat);

        auto aotoso = std::make_shared<Matrix>("aotoso", nso, nso);

        int offset_col = 0;
        for(int irp = 0; irp < nirrep; ++irp) {
            auto nmopi = ints_->wfn()->nmopi()[irp];
            for(int i = 0; i < nso; ++i) {
                for(int j = 0; j < nmopi; ++j) {
                    aotoso->set(i, offset_col+j, ints_->wfn()->aotoso()->get(irp, i, j));
                }   
            }
            offset_col += nmopi;
        }

        temp_mat_MO->transform(aotoso->transpose());

        for(int i = 0; i < nso; ++i) {
            for (int j = 0; j < nso; ++j) {
                auto val = temp_mat_MO->get(i, j);
                M->set(aux_idx, i * nso + j, val);
            }
        }
    }

    // assume "alpha == beta"
    M->scale(2.0);

    auto psio_ = _default_psio_lib_;

    M->set_name("3-Center Reference Density");
    M->save(psio_, PSIF_AO_TPDM, Matrix::SaveType::ThreeIndexLowerTriangle);
    M->zero();
    M->set_name("3-Center Correlation Density");
    M->save(psio_, PSIF_AO_TPDM, Matrix::SaveType::ThreeIndexLowerTriangle);

    SharedMatrix N(new Matrix("metric derivative density", naux, naux));

    (df_2rdm.block("LL")).iterate([&](const std::vector<size_t>& i, double& value) {
        N->set(i[0], i[1], value);
    });
    N->set_name("Metric Reference Density");
    N->save(psio_, PSIF_AO_TPDM, Matrix::SaveType::LowerTriangle);
    N->zero();
    N->set_name("Metric Correlation Density");
    N->save(psio_, PSIF_AO_TPDM, Matrix::SaveType::LowerTriangle);
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