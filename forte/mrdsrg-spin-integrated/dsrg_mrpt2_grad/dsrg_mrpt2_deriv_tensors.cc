/**
 * Set CI-related and DSRG tensors.
 */
#include "psi4/libpsi4util/PsiOutStream.h"
#include "../dsrg_mrpt2.h"
#include "helpers/timer.h"

using namespace ambit;
using namespace psi;

namespace forte {

void DSRG_MRPT2::set_ci_ints() {
    cc = coupling_coefficients_;
    ci = ci_vectors_[0];

    // X_K <K|p^+ q|0> + <0|p^+ q|K> X_K
    Gamma1_tilde = BTF_->build(CoreTensor, "Gamma1_tilde", spin_cases({"aa"}));
    // X_K <K|p^+ q^+ r s|0> + <0|p^+ q^+ r s|K> X_K
    Gamma2_tilde = BTF_->build(CoreTensor, "Gamma2_tilde", spin_cases({"aaaa"}));

    cc2aa = ambit::Tensor::build(ambit::CoreTensor, "<K|p^+ q^+ r s|0> + <0|p^+ q^+ r s|K>",
                                 {ndets, na, na, na, na});
    cc2bb = ambit::Tensor::build(ambit::CoreTensor, "<K|P^+ Q^+ R S|0> + <0|P^+ Q^+ R S|K>",
                                 {ndets, na, na, na, na});
    cc2ab = ambit::Tensor::build(ambit::CoreTensor, "<K|p^+ Q^+ r S|0> + <0|p^+ Q^+ r S|K>",
                                 {ndets, na, na, na, na});
    cc3aaa =
        ambit::Tensor::build(ambit::CoreTensor, "<K|o^+ p^+ q^+ r s t|0> + <0|o^+ p^+ q^+ r s t|K>",
                             {ndets, na, na, na, na, na, na});
    cc3bbb =
        ambit::Tensor::build(ambit::CoreTensor, "<K|O^+ P^+ Q^+ R S T|0> + <0|O^+ P^+ Q^+ R S T|K>",
                             {ndets, na, na, na, na, na, na});
    cc3aab =
        ambit::Tensor::build(ambit::CoreTensor, "<K|o^+ p^+ Q^+ r s T|0> + <0|o^+ p^+ Q^+ r s T|K>",
                             {ndets, na, na, na, na, na, na});
    cc3abb =
        ambit::Tensor::build(ambit::CoreTensor, "<K|o^+ P^+ Q^+ r S T|0> + <0|o^+ P^+ Q^+ r S T|K>",
                             {ndets, na, na, na, na, na, na});

    cc2aa("Kuvxy") += cc.cc2aa()("KJuvxy") * ci("J");
    cc2aa("Kuvxy") += cc.cc2aa()("KJxyuv") * ci("J");
    cc2bb("KUVXY") += cc.cc2bb()("KJUVXY") * ci("J");
    cc2bb("KUVXY") += cc.cc2bb()("KJXYUV") * ci("J");
    cc2ab("KuVxY") += cc.cc2ab()("KJuVxY") * ci("J");
    cc2ab("KuVxY") += cc.cc2ab()("KJxYuV") * ci("J");
    cc3aaa("Kuvwxyz") += cc.cc3aaa()("KJuvwxyz") * ci("J");
    cc3aaa("Kuvwxyz") += cc.cc3aaa()("KJxyzuvw") * ci("J");
    cc3bbb("KUVWXYZ") += cc.cc3bbb()("KJUVWXYZ") * ci("J");
    cc3bbb("KUVWXYZ") += cc.cc3bbb()("KJXYZUVW") * ci("J");
    cc3aab("KuvWxyZ") += cc.cc3aab()("KJuvWxyZ") * ci("J");
    cc3aab("KuvWxyZ") += cc.cc3aab()("KJxyZuvW") * ci("J");
    cc3abb("KuVWxYZ") += cc.cc3abb()("KJuVWxYZ") * ci("J");
    cc3abb("KuVWxYZ") += cc.cc3abb()("KJxYZuVW") * ci("J");
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

    V_sumA_Alpha = BTF_->build(
        CoreTensor, "normal Dimension-reduced Electron Repulsion Integral alpha", {"gg"});
    V_sumB_Alpha = BTF_->build(CoreTensor,
                               "normal Dimension-reduced Electron Repulsion Integral beta", {"gg"});
    V_sumA_Beta = BTF_->build(
        CoreTensor, "index-reversed Dimension-reduced Electron Repulsion Integral beta", {"GG"});
    V_sumB_Beta = BTF_->build(
        CoreTensor, "normal Dimension-reduced Electron Repulsion Integral all beta", {"GG"});

    // Summation of V["pmqm"] over index "m" or V["mpmq"] over index "m"
    V_sumA_Alpha["pq"] = V["pmqn"] * I["mn"];
    // Summation of V["pMqM"] over index "M"
    V_sumB_Alpha["pq"] = V["pMqN"] * I["MN"];
    // Summation of V["mPmQ"] over index "m"
    V_sumA_Beta["PQ"] = V["mPnQ"] * I["mn"];
    // Summation of V["PMQM"] over index "M"
    V_sumB_Beta["PQ"] = V["PMQN"] * I["MN"];
}

void DSRG_MRPT2::set_active_fock() {
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