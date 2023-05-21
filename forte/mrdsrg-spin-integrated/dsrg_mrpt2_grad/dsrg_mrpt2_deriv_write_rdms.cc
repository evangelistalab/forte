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
#include "psi4/lib3index/3index.h"
#include "psi4/libmints/mintshelper.h"
#include "base_classes/mo_space_info.h"
#include "psi4/libmints/molecule.h"
#include "psi4/libmints/vector.h"

using namespace ambit;
using namespace psi;

namespace forte {

void DSRG_MRPT2::write_lagrangian() {
    // NOTICE: write the Lagrangian
    outfile->Printf("\n    Writing EWDM (Lagrangian) ....................... ");

    auto L = std::make_shared<psi::Matrix>("Lagrangian", nirrep, irrep_vec, irrep_vec);

    auto blocklabel = {"cc", "CC", "aa", "AA", "ca", "ac", "CA", "AC", "vv",
                       "VV", "av", "cv", "va", "vc", "AV", "CV", "VA", "VC"};

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
        std::make_shared<psi::Matrix>("Lagrangian", nirrep, irrep_vec, irrep_vec));
    ints_->wfn()->lagrangian()->copy(L);

    outfile->Printf("Done");
}

void DSRG_MRPT2::write_1rdm_spin_dependent() {
    // NOTICE: write spin_dependent one-RDMs coefficients.
    outfile->Printf("\n    Writing 1RDM Coefficients ....................... ");

    auto blocklabel = {"cc", "aa", "vv", "ca", "ac", "av", "va", "vc", "cv"};
    std::map<char, std::vector<std::pair<unsigned long, unsigned long>,
                               std::allocator<std::pair<unsigned long, unsigned long>>>>
        idxmap_re;
    idxmap_re = {{'c', core_mos_relative}, {'a', actv_mos_relative}, {'v', virt_mos_relative}};
    auto D1 = std::make_shared<psi::Matrix>("1rdm coefficients contribution", nirrep, irrep_vec,
                                            irrep_vec);
    BlockedTensor D1_temp = BTF_->build(CoreTensor, "D1_temp", {"gg"}, true);

    D1_temp["em"] += Z["em"];
    D1_temp["em"] += 0.5 * sigma3_xi3["me"];
    D1_temp["me"] += Z["em"];
    D1_temp["me"] += 0.5 * sigma3_xi3["me"];
    D1_temp["nu"] += Z["un"];
    D1_temp["nu"] += 0.5 * sigma3_xi3["nu"];
    D1_temp["nv"] -= Z["un"] * Gamma1_["uv"];
    D1_temp["un"] += Z["un"];
    D1_temp["un"] += 0.5 * sigma3_xi3["nu"];
    D1_temp["vn"] -= Z["un"] * Gamma1_["uv"];
    D1_temp["ev"] += Z["eu"] * Gamma1_["uv"];
    D1_temp["ev"] += 0.5 * sigma3_xi3["ve"];
    D1_temp["ve"] += Z["eu"] * Gamma1_["uv"];
    D1_temp["ve"] += 0.5 * sigma3_xi3["ve"];
    D1_temp["mn"] += I["mn"];
    D1_temp["mn"] += Z["mn"];
    D1_temp["uv"] += Gamma1_["uv"];
    D1_temp["uv"] += 0.5 * Gamma1_tilde["uv"];
    D1_temp["uv"] += Z["uv"];
    D1_temp["ef"] += Z["ef"];
    for (const std::string& block : blocklabel) {
        auto label1 = block[0];
        auto label2 = block[1];
        (D1_temp.block(block)).iterate([&](const std::vector<size_t>& i, double& value) {
            if (idxmap_re[label1][i[0]].first == idxmap_re[label2][i[1]].first) {
                D1->add(idxmap_re[label1][i[0]].first, idxmap_re[label1][i[0]].second,
                        idxmap_re[label2][i[1]].second, value);
            }
        });
    }
    D1->back_transform(ints_->Ca());
    ints_->wfn()->Da()->copy(D1);
    ints_->wfn()->Db()->copy(D1);
    outfile->Printf("Done");

    // DSRG-MRPT2 dipole moment
    auto mo_dipole_ints = ints_->mo_dipole_ints(); // just take alpha spin
    std::map<char, std::vector<size_t>> idxmap_abs;
    idxmap_abs = {{'c', core_all_}, {'a', actv_all_}, {'v', virt_all_}};
    std::vector<double> dipole(4, 0.0);
    Vector3 dm_nuc =
        psi::Process::environment.molecule()->nuclear_dipole(psi::Vector3(0.0, 0.0, 0.0));

    for (int i = 0; i < 3; ++i) {
        auto dm_ints = mo_dipole_ints[i];
        BlockedTensor dipole_ints = BTF_->build(CoreTensor, "dipole_ints", {"gg"}, true);
        for (const std::string& block : blocklabel) {
            auto label1 = block[0];
            auto label2 = block[1];
            (dipole_ints.block(block)).iterate([&](const std::vector<size_t>& i, double& value) {
                value = dm_ints->get(idxmap_abs[label1][i[0]], idxmap_abs[label2][i[1]]);
            });
        }
        dipole[i] = 2.0 * D1_temp["pq"] * dipole_ints["pq"];
        dipole[i] += dm_nuc[i];
        dipole[3] += dipole[i] * dipole[i];
    }
    dipole[3] = std::sqrt(dipole[3]);
    if (eri_df_) {
        outfile->Printf("\n\n    DF-DSRG-MRPT2 dipole moment:");
    } else {
        outfile->Printf("\n    DSRG-MRPT2 dipole moment:");
    }
    outfile->Printf("\n      X: %10.6f  Y: %10.6f  Z: %10.6f  Total: %10.6f\n", dipole[0],
                    dipole[1], dipole[2], dipole[3]);
}

void DSRG_MRPT2::write_2rdm_spin_dependent() {
    // NOTICE: write spin_dependent two-RDMs coefficients using IWL
    outfile->Printf("\n    Writing 2RDM Coefficients ....................... ");
    auto psio_ = _default_psio_lib_;
    IWL d2aa(psio_.get(), PSIF_MO_AA_TPDM, 1.0e-14, 0, 0);
    IWL d2ab(psio_.get(), PSIF_MO_AB_TPDM, 1.0e-14, 0, 0);
    IWL d2bb(psio_.get(), PSIF_MO_BB_TPDM, 1.0e-14, 0, 0);

    BlockedTensor temp = BTF_->build(CoreTensor, "temporal tensor", {"vc", "VC"}, true);
    // <[F, T2]> and <[V, T1]>
    temp["em"] += 0.5 * sigma3_xi3["me"];
    temp["EM"] += 0.5 * sigma3_xi3["ME"];

    for (size_t i = 0, size_c = core_all_.size(); i < size_c; ++i) {
        auto m = core_all_[i];
        for (size_t a = 0, size_v = virt_all_.size(); a < size_v; ++a) {
            auto e = virt_all_[a];
            auto idx = a * ncore + i;
            auto z_a = Z.block("vc").data()[idx] + temp.block("vc").data()[idx];
            auto z_b = Z.block("VC").data()[idx] + temp.block("VC").data()[idx];
            for (size_t j = 0; j < size_c; ++j) {
                auto m1 = core_all_[j];

                d2aa.write_value(m, e, m1, m1, z_a, 0, "outfile", 0);
                d2bb.write_value(m, e, m1, m1, z_b, 0, "outfile", 0);
                d2aa.write_value(m, m1, m1, e, -z_a, 0, "outfile", 0);
                d2bb.write_value(m, m1, m1, e, -z_b, 0, "outfile", 0);
                d2ab.write_value(m, e, m1, m1, 2.0 * (z_a + z_b), 0, "outfile", 0);
            }
        }
    }

    temp = BTF_->build(CoreTensor, "temporal tensor", {"ac", "AC"}, true);
    temp["un"] = Z["un"];
    temp["un"] -= Z["vn"] * Gamma1_["uv"];
    temp["UN"] = Z["UN"];
    temp["UN"] -= Z["VN"] * Gamma1_["UV"];
    // <[F, T2]> and <[V, T1]>
    temp["un"] += 0.5 * sigma3_xi3["nu"];
    temp["UN"] += 0.5 * sigma3_xi3["NU"];

    for (size_t i = 0, size_c = core_all_.size(); i < size_c; ++i) {
        auto n = core_all_[i];
        for (size_t a = 0, size_a = actv_all_.size(); a < size_a; ++a) {
            auto u = actv_all_[a];
            auto idx = a * ncore + i;
            auto z_a = temp.block("ac").data()[idx];
            auto z_b = temp.block("AC").data()[idx];
            for (size_t j = 0; j < size_c; ++j) {
                auto m1 = core_all_[j];
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

    temp = BTF_->build(CoreTensor, "temporal tensor", {"va", "VA"}, true);
    temp["ev"] = Z["eu"] * Gamma1_["uv"];
    temp["EV"] = Z["EU"] * Gamma1_["UV"];
    // <[F, T2]> and <[V, T1]>
    temp["ev"] += 0.5 * sigma3_xi3["ve"];
    temp["EV"] += 0.5 * sigma3_xi3["VE"];

    for (size_t i = 0, size_a = actv_all_.size(); i < size_a; ++i) {
        auto v = actv_all_[i];
        for (size_t a = 0, size_v = virt_all_.size(); a < size_v; ++a) {
            auto e = virt_all_[a];
            auto idx = a * na + i;
            auto z_a = temp.block("va").data()[idx];
            auto z_b = temp.block("VA").data()[idx];
            for (size_t j = 0, size_c = core_all_.size(); j < size_c; ++j) {
                auto m1 = core_all_[j];

                d2aa.write_value(v, e, m1, m1, z_a, 0, "outfile", 0);
                d2bb.write_value(v, e, m1, m1, z_b, 0, "outfile", 0);
                d2aa.write_value(v, m1, m1, e, -z_a, 0, "outfile", 0);
                d2bb.write_value(v, m1, m1, e, -z_b, 0, "outfile", 0);
                d2ab.write_value(v, e, m1, m1, 2.0 * (z_a + z_b), 0, "outfile", 0);
            }
        }
    }

    for (size_t i = 0, size_c = core_all_.size(); i < size_c; ++i) {
        auto n = core_all_[i];
        for (size_t k = 0; k < size_c; ++k) {
            auto m = core_all_[k];
            auto idx = k * ncore + i;
            auto z_a = Z.block("cc").data()[idx];
            auto z_b = Z.block("CC").data()[idx];
            for (size_t j = 0; j < size_c; ++j) {
                auto m1 = core_all_[j];
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

    for (size_t i = 0, size_a = actv_all_.size(); i < size_a; ++i) {
        auto v = actv_all_[i];
        for (size_t k = 0; k < size_a; ++k) {
            auto u = actv_all_[k];
            auto idx = k * na + i;
            auto z_a = Z.block("aa").data()[idx];
            auto z_b = Z.block("AA").data()[idx];
            auto gamma_a = Gamma1_.block("aa").data()[idx];
            auto gamma_b = Gamma1_.block("AA").data()[idx];
            auto ci_gamma_a = ci_g1_a.data()[idx];
            auto ci_gamma_b = ci_g1_b.data()[idx];
            auto a1 = z_a + gamma_a + ci_gamma_a;
            auto v2 = z_b + gamma_b + ci_gamma_b;

            for (size_t j = 0, size_c = core_all_.size(); j < size_c; ++j) {
                auto m1 = core_all_[j];

                d2aa.write_value(v, u, m1, m1, 0.5 * a1, 0, "outfile", 0);
                d2bb.write_value(v, u, m1, m1, 0.5 * v2, 0, "outfile", 0);
                d2aa.write_value(v, m1, m1, u, -0.5 * a1, 0, "outfile", 0);
                d2bb.write_value(v, m1, m1, u, -0.5 * v2, 0, "outfile", 0);
                d2ab.write_value(v, u, m1, m1, (a1 + v2), 0, "outfile", 0);
            }
        }
    }

    for (size_t i = 0, size_v = virt_all_.size(); i < size_v; ++i) {
        auto f = virt_all_[i];
        for (size_t k = 0; k < size_v; ++k) {
            auto e = virt_all_[k];
            auto idx = k * nvirt + i;
            auto z_a = Z.block("vv").data()[idx];
            auto z_b = Z.block("VV").data()[idx];
            for (size_t j = 0, size_c = core_all_.size(); j < size_c; ++j) {
                auto m1 = core_all_[j];

                d2aa.write_value(f, e, m1, m1, 0.5 * z_a, 0, "outfile", 0);
                d2bb.write_value(f, e, m1, m1, 0.5 * z_b, 0, "outfile", 0);
                d2aa.write_value(f, m1, m1, e, -0.5 * z_a, 0, "outfile", 0);
                d2bb.write_value(f, m1, m1, e, -0.5 * z_b, 0, "outfile", 0);
                d2ab.write_value(f, e, m1, m1, (z_a + z_b), 0, "outfile", 0);
            }
        }
    }

    for (size_t i = 0, size_c = core_all_.size(); i < size_c; ++i) {
        auto n = core_all_[i];
        for (size_t j = 0; j < size_c; ++j) {
            auto m = core_all_[j];
            auto idx = j * ncore + i;
            auto z_a = Z.block("cc").data()[idx];
            auto z_b = Z.block("CC").data()[idx];
            for (size_t k = 0, size_a = actv_all_.size(); k < size_a; ++k) {
                auto a1 = actv_all_[k];
                for (size_t l = 0; l < size_a; ++l) {
                    auto u1 = actv_all_[l];
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

    for (size_t i = 0, size_v = virt_all_.size(); i < size_v; ++i) {
        auto f = virt_all_[i];
        for (size_t j = 0; j < size_v; ++j) {
            auto e = virt_all_[j];
            auto idx = j * nvirt + i;
            auto z_a = Z.block("vv").data()[idx];
            auto z_b = Z.block("VV").data()[idx];
            for (size_t k = 0, size_a = actv_all_.size(); k < size_a; ++k) {
                auto a1 = actv_all_[k];
                for (size_t l = 0; l < size_a; ++l) {
                    auto u1 = actv_all_[l];
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
    temp = BTF_->build(CoreTensor, "temporal tensor", {"pphh", "pPhH"}, true);
    BlockedTensor temp2 = BTF_->build(CoreTensor, "temporal tensor 2", {"phph", "phPH"}, true);

    if (CORRELATION_TERM) {
        {
            BlockedTensor Eeps2_m1 =
                BTF_->build(CoreTensor, "{1-e^[-s*(Delta2)^2]}/(Delta2)", {"hhpp", "hHpP"}, true);
            Eeps2_m1.iterate([&](const std::vector<size_t>& i,
                                 const std::vector<SpinType>& /*spin*/, double& value) {
                value = dsrg_source_->compute_renormalized_denominator(Fa_[i[0]] + Fa_[i[1]] -
                                                                       Fa_[i[2]] - Fa_[i[3]]);
            });
            temp["abij"] += Tau2["ijab"] * Eeps2_m1["ijab"];
            temp["aBiJ"] += Tau2["iJaB"] * Eeps2_m1["iJaB"];
        }
        {
            BlockedTensor Eeps2_p =
                BTF_->build(CoreTensor, "1+e^[-s*(Delta2)^2]", {"hhpp", "hHpP"}, true);
            Eeps2_p.iterate([&](const std::vector<size_t>& i, const std::vector<SpinType>& /*spin*/,
                                double& value) {
                value = 1.0 + dsrg_source_->compute_renormalized(Fa_[i[0]] + Fa_[i[1]] - Fa_[i[2]] -
                                                                 Fa_[i[3]]);
            });
            temp["cdkl"] += Kappa["klcd"] * Eeps2_p["klcd"];
            temp["cDkL"] += Kappa["kLcD"] * Eeps2_p["kLcD"];
        }
    }

    temp["xynv"] -= Z["un"] * Gamma2_["uvxy"];
    temp["xYnV"] -= Z["un"] * Gamma2_["uVxY"];

    temp["evxy"] += Z["eu"] * Gamma2_["uvxy"];
    temp["eVxY"] += Z["eu"] * Gamma2_["uVxY"];

    // CASSCF reference
    temp["xyuv"] += 0.25 * Gamma2_["uvxy"];
    temp["xYuV"] += 0.25 * Gamma2_["uVxY"];

    // CI contribution
    temp["xyuv"] += 0.125 * Gamma2_tilde["uvxy"];
    temp["xYuV"] += 0.125 * Gamma2_tilde["uVxY"];

    // all-alpha and all-beta
    temp2["ckdl"] += temp["cdkl"];
    temp2["cldk"] -= temp["cdkl"];
    // alpha-beta
    temp2["ckDL"] += 2.0 * temp["cDkL"];
    temp2["clDK"] += 2.0 * temp["cDlK"];
    temp.zero();

    temp["eumv"] += 2.0 * Z["em"] * Gamma1_["uv"];
    temp["eUmV"] += 2.0 * Z["em"] * Gamma1_["UV"];

    temp["u,a1,n,u1"] += 2.0 * Z["un"] * Gamma1_["u1,a1"];
    temp["u,A1,n,U1"] += 2.0 * Z["un"] * Gamma1_["U1,A1"];

    temp["v,a1,u,u1"] += Z["uv"] * Gamma1_["u1,a1"];
    temp["v,A1,u,U1"] += Z["uv"] * Gamma1_["U1,A1"];

    // <[F, T2]> and <[V, T1]>
    if (X5_TERM || X6_TERM || X7_TERM) {
        temp["aviu"] += sigma3_xi3["ia"] * Gamma1_["uv"];
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
    /**
     * Initializing the DF metric J^(-1/2).
     */
    BlockedTensor Jm12 = BTF_->build(tensor_type_, "Jm12", {"LL"}, true);
    std::shared_ptr<BasisSet> auxiliary_ = ints_->wfn()->get_basisset("DF_BASIS_MP2");
    auto metric = std::make_shared<FittingMetric>(auxiliary_, true);
    // "form_eig_inverse()" genererates J^(-1/2); "form_full_eig_inverse()" genererates J^(-1)
    metric->form_eig_inverse(Process::environment.options.get_double("DF_FITTING_CONDITION"));
    auto J = metric->get_metric();
    (Jm12.block("LL")).iterate([&](const std::vector<size_t>& i, double& value) {
        value = J->get(i[0], i[1]);
    });

    BlockedTensor df_3rdm_temp = BTF_->build(tensor_type_, "df_3rdm_temp", {"Lgg"}, true);
    {
        // density terms contracted with V["abij"]
        BlockedTensor dvabij =
            BTF_->build(CoreTensor, "density of V['abij']", {"pphh", "pPhH"}, true);

        if (CORRELATION_TERM) {
            {
                BlockedTensor Eeps2_m1 =
                    BTF_->build(CoreTensor, "{1-e^[-s*(Delta2)^2]}/(Delta2)", {"hhpp"}, true);
                Eeps2_m1.iterate([&](const std::vector<size_t>& i,
                                     const std::vector<SpinType>& /*spin*/, double& value) {
                    value = dsrg_source_->compute_renormalized_denominator(Fa_[i[0]] + Fa_[i[1]] -
                                                                           Fa_[i[2]] - Fa_[i[3]]);
                });
                dvabij["abij"] += Tau2["ijab"] * Eeps2_m1["ijab"];
            }
            {
                BlockedTensor Eeps2_m1 =
                    BTF_->build(CoreTensor, "{1-e^[-s*(Delta2)^2]}/(Delta2)", {"hHpP"}, true);
                Eeps2_m1.iterate([&](const std::vector<size_t>& i,
                                     const std::vector<SpinType>& /*spin*/, double& value) {
                    value = dsrg_source_->compute_renormalized_denominator(Fa_[i[0]] + Fa_[i[1]] -
                                                                           Fa_[i[2]] - Fa_[i[3]]);
                });
                dvabij["aBiJ"] += Tau2["iJaB"] * Eeps2_m1["iJaB"];
            }
            {
                BlockedTensor Eeps2_p =
                    BTF_->build(CoreTensor, "1+e^[-s*(Delta2)^2]", {"hhpp"}, true);
                Eeps2_p.iterate([&](const std::vector<size_t>& i,
                                    const std::vector<SpinType>& /*spin*/, double& value) {
                    value = 1.0 + dsrg_source_->compute_renormalized(Fa_[i[0]] + Fa_[i[1]] -
                                                                     Fa_[i[2]] - Fa_[i[3]]);
                });
                dvabij["cdkl"] += Kappa["klcd"] * Eeps2_p["klcd"];
            }
            {
                BlockedTensor Eeps2_p =
                    BTF_->build(CoreTensor, "1+e^[-s*(Delta2)^2]", {"hHpP"}, true);
                Eeps2_p.iterate([&](const std::vector<size_t>& i,
                                    const std::vector<SpinType>& /*spin*/, double& value) {
                    value = 1.0 + dsrg_source_->compute_renormalized(Fa_[i[0]] + Fa_[i[1]] -
                                                                     Fa_[i[2]] - Fa_[i[3]]);
                });
                dvabij["cDkL"] += Kappa["kLcD"] * Eeps2_p["kLcD"];
            }
        }
        // CASSCF reference
        dvabij["xyuv"] += 0.25 * Gamma2_["uvxy"];
        dvabij["xYuV"] += 0.25 * Gamma2_["uVxY"];

        // CI contribution
        dvabij["xyuv"] += 0.125 * Gamma2_tilde["uvxy"];
        dvabij["xYuV"] += 0.125 * Gamma2_tilde["uVxY"];

        df_3rdm_temp["Q!,a,i"] += 4.0 * Jm12["Q!,R!"] * B["R!,b,j"] * dvabij["abij"];
        df_3rdm_temp["Q!,a,i"] += 4.0 * Jm12["Q!,R!"] * B["R!,B,J"] * dvabij["aBiJ"];
    }

    // - Z["un"] * Gamma2_["uvxy"]
    // - Z["UN"] * Gamma2_["UVXY"]
    // - Z["un"] * Gamma2_["uVxY"]
    // Coulomb part
    df_3rdm_temp["Q!,x,n"] -= Jm12["Q!,R!"] * B["R!,y,v"] * Z["un"] * Gamma2_["uvxy"];
    df_3rdm_temp["Q!,x,n"] -= Jm12["Q!,R!"] * B["R!,Y,V"] * Z["un"] * Gamma2_["uVxY"];

    df_3rdm_temp["R!,y,v"] -= Jm12["Q!,R!"] * B["Q!,x,n"] * Z["un"] * Gamma2_["uvxy"];
    df_3rdm_temp["R!,y,v"] -= Jm12["Q!,R!"] * B["Q!,X,N"] * Z["UN"] * Gamma2_["vUyX"];

    // Exchange part
    df_3rdm_temp["Q!,y,n"] += Jm12["Q!,R!"] * B["R!,x,v"] * Z["un"] * Gamma2_["uvxy"];
    df_3rdm_temp["Q!,y,n"] -= Jm12["Q!,R!"] * B["R!,X,V"] * Z["un"] * Gamma2_["uVyX"];

    df_3rdm_temp["R!,x,v"] += Jm12["Q!,R!"] * B["Q!,y,n"] * Z["un"] * Gamma2_["uvxy"];
    df_3rdm_temp["R!,x,v"] -= Jm12["Q!,R!"] * B["Q!,Y,N"] * Z["UN"] * Gamma2_["vUxY"];

    // Z["eu"] * Gamma2_["uvxy"];
    // Z["EU"] * Gamma2_["UVXY"];
    // Z["eu"] * Gamma2_["uVxY"];
    // Coulomb part
    df_3rdm_temp["Q!,x,e"] += Jm12["Q!,R!"] * B["R!,y,v"] * Z["ue"] * Gamma2_["uvxy"];
    df_3rdm_temp["Q!,x,e"] += Jm12["Q!,R!"] * B["R!,Y,V"] * Z["ue"] * Gamma2_["uVxY"];

    df_3rdm_temp["R!,y,v"] += Jm12["Q!,R!"] * B["Q!,x,e"] * Z["ue"] * Gamma2_["uvxy"];
    df_3rdm_temp["R!,y,v"] += Jm12["Q!,R!"] * B["Q!,X,E"] * Z["UE"] * Gamma2_["vUyX"];

    // Exchange part
    df_3rdm_temp["Q!,y,e"] -= Jm12["Q!,R!"] * B["R!,x,v"] * Z["ue"] * Gamma2_["uvxy"];
    df_3rdm_temp["Q!,y,e"] += Jm12["Q!,R!"] * B["R!,X,V"] * Z["ue"] * Gamma2_["uVyX"];

    df_3rdm_temp["R!,x,v"] -= Jm12["Q!,R!"] * B["Q!,y,e"] * Z["ue"] * Gamma2_["uvxy"];
    df_3rdm_temp["R!,x,v"] += Jm12["Q!,R!"] * B["Q!,Y,E"] * Z["UE"] * Gamma2_["vUxY"];

    // 2.0 * Z["em"] * Gamma1_["uv"]
    // 2.0 * Z["EM"] * Gamma1_["UV"]
    // 2.0 * Z["em"] * Gamma1_["UV"]
    // Coulomb part
    df_3rdm_temp["Q!,e,m"] += 4.0 * Jm12["Q!,R!"] * B["R!,u,v"] * Z["em"] * Gamma1_["uv"];
    df_3rdm_temp["R!,u,v"] += 4.0 * Jm12["Q!,R!"] * B["Q!,e,m"] * Z["em"] * Gamma1_["uv"];
    // Exchange part
    df_3rdm_temp["Q!,e,v"] -= 2.0 * Jm12["Q!,R!"] * B["R!,u,m"] * Z["em"] * Gamma1_["uv"];
    df_3rdm_temp["R!,u,m"] -= 2.0 * Jm12["Q!,R!"] * B["Q!,e,v"] * Z["em"] * Gamma1_["uv"];

    // 2.0 * Z["xm"] * Gamma1_["uv"]
    // 2.0 * Z["XM"] * Gamma1_["UV"]
    // 2.0 * Z["xm"] * Gamma1_["UV"]
    // Coulomb part
    df_3rdm_temp["Q!,x,m"] += 4.0 * Jm12["Q!,R!"] * B["R!,u,v"] * Z["xm"] * Gamma1_["uv"];
    df_3rdm_temp["R!,u,v"] += 4.0 * Jm12["Q!,R!"] * B["Q!,x,m"] * Z["xm"] * Gamma1_["uv"];
    // Exchange part
    df_3rdm_temp["Q!,x,v"] -= 2.0 * Jm12["Q!,R!"] * B["R!,u,m"] * Z["xm"] * Gamma1_["uv"];
    df_3rdm_temp["R!,u,m"] -= 2.0 * Jm12["Q!,R!"] * B["Q!,x,v"] * Z["xm"] * Gamma1_["uv"];

    // Z["xy"] * Gamma1_["uv"]
    // Z["XY"] * Gamma1_["UV"]
    // Z["xy"] * Gamma1_["UV"]
    // Coulomb part
    df_3rdm_temp["Q!,x,y"] += 2.0 * Jm12["Q!,R!"] * B["R!,u,v"] * Z["xy"] * Gamma1_["uv"];
    df_3rdm_temp["R!,u,v"] += 2.0 * Jm12["Q!,R!"] * B["Q!,x,y"] * Z["xy"] * Gamma1_["uv"];
    // Exchange part
    df_3rdm_temp["Q!,x,v"] -= Jm12["Q!,R!"] * B["R!,u,y"] * Z["xy"] * Gamma1_["uv"];
    df_3rdm_temp["R!,u,y"] -= Jm12["Q!,R!"] * B["Q!,x,v"] * Z["xy"] * Gamma1_["uv"];

    // <[F, T2]> and <[V, T1]>
    // if X5_TERM || X6_TERM || X7_TERM is true
    // sigma3_xi3["ia"] * Gamma1_["uv"]
    // sigma3_xi3["IA"] * Gamma1_["UV"]
    // sigma3_xi3["ia"] * Gamma1_["UV"]
    // Coulomb part
    df_3rdm_temp["Q!,i,a"] += 2.0 * Jm12["Q!,R!"] * B["R!,u,v"] * sigma3_xi3["ia"] * Gamma1_["uv"];
    df_3rdm_temp["R!,u,v"] += 2.0 * Jm12["Q!,R!"] * B["Q!,i,a"] * sigma3_xi3["ia"] * Gamma1_["uv"];
    // Exchange part
    df_3rdm_temp["Q!,i,v"] -= Jm12["Q!,R!"] * B["R!,u,a"] * sigma3_xi3["ia"] * Gamma1_["uv"];
    df_3rdm_temp["R!,u,a"] -= Jm12["Q!,R!"] * B["Q!,i,v"] * sigma3_xi3["ia"] * Gamma1_["uv"];

    /************************************************************************************************/

    BlockedTensor temp = BTF_->build(CoreTensor, "temporal tensor", {"gg"}, true);
    // <[F, T2]> and <[V, T1]>
    temp["em"] += sigma3_xi3["me"];
    temp["em"] += 2.0 * Z["em"];

    temp["un"] += 2.0 * Z["un"];
    temp["un"] -= 2.0 * Z["vn"] * Gamma1_["uv"];
    // <[F, T2]> and <[V, T1]>
    temp["un"] += sigma3_xi3["nu"];

    temp["ev"] += 2.0 * Z["eu"] * Gamma1_["uv"];
    // <[F, T2]> and <[V, T1]>
    temp["ev"] += sigma3_xi3["ve"];

    temp["mn"] += Z["mn"];

    temp["uv"] += Z["uv"];
    temp["uv"] += Gamma1_["uv"];
    temp["uv"] += 0.5 * Gamma1_tilde["uv"];

    temp["ef"] += Z["ef"];

    // Coulomb part
    df_3rdm_temp["Q!,p,r"] += 2.0 * Jm12["Q!,R!"] * B["R!,m,n"] * I["mn"] * temp["pr"];
    df_3rdm_temp["Q!,m,n"] += 2.0 * Jm12["Q!,R!"] * B["R!,p,r"] * I["mn"] * temp["pr"];

    // Exchange part
    df_3rdm_temp["Q!,p,m"] -= Jm12["Q!,R!"] * B["R!,m,r"] * temp["pr"];
    df_3rdm_temp["Q!,m,r"] -= Jm12["Q!,R!"] * B["R!,p,m"] * temp["pr"];

    /**************************** CASSCF energy term : 0.5 * V["mnmn"] ****************************/
    // Coulomb part
    df_3rdm_temp["Q!,m1,n1"] += 2.0 * Jm12["Q!,P!"] * B["P!,m,n"] * I["mn"] * I["m1,n1"];
    // Exchange part
    df_3rdm_temp["Q!,m,n"] -= Jm12["Q!,P!"] * B["P!,n,m"];

    // residue terms
    df_3rdm_temp["Q!,m,n"] += 2.0 * Jm12["Q!,R!"] * B["R!,u,v"] * Z["mn"] * Gamma1_["uv"];
    df_3rdm_temp["Q!,m,v"] -= Jm12["Q!,R!"] * B["R!,u,n"] * Z["mn"] * Gamma1_["uv"];

    df_3rdm_temp["R!,u,v"] += 2.0 * Jm12["Q!,R!"] * B["Q!,m,n"] * Z["mn"] * Gamma1_["uv"];
    df_3rdm_temp["R!,u,n"] -= Jm12["Q!,R!"] * B["Q!,m,v"] * Z["mn"] * Gamma1_["uv"];

    df_3rdm_temp["Q!,e,f"] += 2.0 * Jm12["Q!,R!"] * B["R!,u,v"] * Z["ef"] * Gamma1_["uv"];
    df_3rdm_temp["Q!,e,v"] -= Jm12["Q!,R!"] * B["R!,u,f"] * Z["ef"] * Gamma1_["uv"];

    df_3rdm_temp["R!,u,v"] += 2.0 * Jm12["Q!,R!"] * B["Q!,e,f"] * Z["ef"] * Gamma1_["uv"];
    df_3rdm_temp["R!,u,f"] -= Jm12["Q!,R!"] * B["Q!,e,v"] * Z["ef"] * Gamma1_["uv"];

    BlockedTensor df_3rdm = BTF_->build(tensor_type_, "df_3rdm", {"Lgg"}, true);
    df_3rdm["Q!,p,q"] += 0.5 * df_3rdm_temp["Q!,p,q"];
    df_3rdm["Q!,q,p"] += 0.5 * df_3rdm_temp["Q!,p,q"];

    /********************** Backtransform (P|pq) to (P|\mu \nu) **********************/

    int nso = ints_->wfn()->nso();
    int ao_matsize = nso * nso;
    std::map<char, std::vector<std::pair<unsigned long, unsigned long>,
                               std::allocator<std::pair<unsigned long, unsigned long>>>>
        idxmap;
    std::map<string, std::pair<SharedMatrix, SharedMatrix>> slicemap;
    std::map<string, int> stride_size;
    std::map<char, int> orbital_size;
    auto M = std::make_shared<psi::Matrix>("backtransformed df_3rdm", naux, ao_matsize);
    idxmap = {{'c', core_mos_relative}, {'a', actv_mos_relative}, {'v', virt_mos_relative}};

    stride_size = {{"ca", ncore * na},    {"ac", na * ncore},    {"cv", ncore * nvirt},
                   {"vc", nvirt * ncore}, {"av", na * nvirt},    {"va", nvirt * na},
                   {"cc", ncore * ncore}, {"vv", nvirt * nvirt}, {"aa", na * na}};

    orbital_size = {{'c', ncore}, {'a', na}, {'v', nvirt}};
    auto blocklabels = {"cc", "aa", "ca", "ac", "vv", "av", "cv", "va", "vc"};

    std::map<char, int> pre_idx;
    pre_idx = {{'c', 0}, {'a', ncore}, {'v', ncore + na}};

    auto temp_mat_MO = std::make_shared<psi::Matrix>("MO temp matrix", nmo, nmo);

    auto aotoso = std::make_shared<psi::Matrix>("aotoso", nso, nso);

    size_t offset_col = 0;
    for (size_t irp = 0; irp < nirrep; ++irp) {
        auto nmopi = ints_->wfn()->nmopi()[irp];
        for (int i = 0; i < nso; ++i) {
            for (int j = 0; j < nmopi; ++j) {
                aotoso->set(i, offset_col + j, ints_->wfn()->aotoso()->get(irp, i, j));
            }
        }
        offset_col += nmopi;
    }

    auto Ca = ints_->Ca();
    // Copy Ca to a matrix without symmetry blocking
    auto Cat = std::make_shared<psi::Matrix>("Ca temp matrix", nso, nmo);

    std::vector<int> sum_nsopi(nirrep, 0);
    for (size_t irp = 1; irp < nirrep; ++irp) {
        sum_nsopi[irp] = sum_nsopi[irp - 1] + ints_->wfn()->nsopi()[irp - 1];
    }

    size_t offset = 0;
    for (const char& label : {'c', 'a', 'v'}) {
        auto space = idxmap[label];
        for (auto irp_i : space) {
            auto irp = irp_i.first;
            auto index = irp_i.second;
            auto nsopi = ints_->wfn()->nsopi()[irp];
            for (int i = 0; i < nsopi; ++i) {
                auto val = Ca->get(irp, i, index);
                Cat->set(sum_nsopi[irp] + i, offset, val);
            }
            offset += 1;
        }
    }

    for (size_t aux_idx = 0; aux_idx < naux; ++aux_idx) {
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
                    auto val = block_data[aux_idx * stride + i * colsize + j];
                    auto idx1 = pre_idx[label1] + i;
                    auto idx2 = pre_idx[label2] + j;
                    temp_mat_MO->set(idx1, idx2, val);
                }
            }
        }

        temp_mat_MO->back_transform(Cat);
        temp_mat_MO->transform(aotoso->transpose());

        for (int i = 0; i < nso; ++i) {
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

    auto N = std::make_shared<psi::Matrix>("metric derivative density", naux, naux);

    // Using restricted orbitals, thus B["A!,P,Q"] * Jm12["A!,R!"] * df_3rdm["S!,P,Q"]
    // equals B["A!,p,q"] * Jm12["A!,R!"] * df_3rdm["S!,p,q"], yielding a factor 2
    BlockedTensor df_2rdm = BTF_->build(tensor_type_, "df_2rdm", {"LL"}, true);
    df_2rdm["R!,S!"] += 0.5 * B["A!,p,q"] * Jm12["A!,R!"] * df_3rdm["S!,p,q"];
    df_2rdm["S!,R!"] += 0.5 * B["A!,p,q"] * Jm12["A!,R!"] * df_3rdm["S!,p,q"];

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
