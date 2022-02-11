/**
 * Set CI-related and DSRG tensors.
 */
#include "psi4/libpsi4util/PsiOutStream.h"
#include "../dsrg_mrpt2.h"
#include "helpers/timer.h"



// #include "corr_grad.h"

#include "psi4/libqt/qt.h"
#include "psi4/lib3index/3index.h"
#include "psi4/libpsio/psio.hpp"
#include "psi4/libpsio/psio.h"
#include "psi4/psifiles.h"
#include "psi4/libpsi4util/PsiOutStream.h"
#include "psi4/libpsi4util/process.h"
#include "psi4/liboptions/liboptions.h"

#include "psi4/libmints/basisset.h"
#include "psi4/libmints/matrix.h"
#include "psi4/libmints/mintshelper.h"
#include "psi4/libmints/molecule.h"
#include "psi4/libmints/twobody.h"
#include "psi4/libmints/integral.h"
#include "psi4/libmints/vector.h"

#include "/Users/shuhe.wang/src/psi4/psi4/src/psi4/dfmp2/corr_grad.h"

#include "psi4/lib3index/dftensor.h"
#include "psi4/psi4-dec.h"
#include "psi4/physconst.h"
#include "psi4/psifiles.h"

#include "psi4/lib3index/3index.h"
#include "psi4/libfock/apps.h"
#include "psi4/libfock/jk.h"
#include "psi4/libmints/basisset.h"
#include "psi4/libmints/extern.h"
#include "psi4/libmints/integral.h"
#include "psi4/libmints/matrix.h"
#include "psi4/libmints/mintshelper.h"
#include "psi4/libmints/molecule.h"
#include "psi4/libmints/oeprop.h"
#include "psi4/libmints/twobody.h"
#include "psi4/libmints/vector.h"
#include "psi4/libpsi4util/PsiOutStream.h"
#include "psi4/libpsi4util/process.h"
#include "psi4/libpsio/psio.h"
#include "psi4/libpsio/psio.hpp"
#include "psi4/libqt/qt.h"

#include "psi4/libmints/wavefunction.h"
#include <map>

using namespace ambit;
using namespace psi;

namespace forte {

void DSRG_MRPT2::set_ci_ints() {
    ci = ci_vectors_[0];

    // X_K <K|p^+ q|0> + <0|p^+ q|K> X_K
    Gamma1_tilde = BTF_->build(CoreTensor, "Gamma1_tilde", spin_cases({"aa"}));
    // X_K <K|p^+ q^+ r s|0> + <0|p^+ q^+ r s|K> X_K
    Gamma2_tilde = BTF_->build(CoreTensor, "Gamma2_tilde", spin_cases({"aaaa"}));
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

void DSRG_MRPT2::set_j() {  
    Jm12 = BTF_->build(tensor_type_, "Jm12", {"LL"});
    std::shared_ptr<BasisSet> auxiliary_ = ints_->wfn()->get_basisset("DF_BASIS_MP2");
    auto metric = std::make_shared<FittingMetric>(auxiliary_, true);
    // "form_eig_inverse()" genererates J^(-1/2); "form_full_eig_inverse()" genererates J^(-1)
    metric->form_full_eig_inverse(Process::environment.options.get_double("DF_FITTING_CONDITION"));
    auto J = metric->get_metric();

    (Jm12.block("LL")).iterate([&](const std::vector<size_t>& i, double& value) {
        value = J->get(i[0], i[1]);
    });



    Ppq = BTF_->build(tensor_type_, "Ppq", {"Lgg", "LGG"});
    auto blocklabels = {"cc", "aa", "vv", "ac", "ca", "va", "av", "vc", "cv",
                        "CC", "AA", "VV", "AC", "CA", "VA", "AV", "VC", "CV"};

    std::map<char, int> pre_idx;
    pre_idx = {{'c', 0},
               {'a', ncore},
               {'v', ncore + na},
               {'C', 0},
               {'A', ncore},
               {'V', ncore + na}};


    auto primary = ints_->wfn()->basisset();
    std::shared_ptr<BasisSet> zero_bas(BasisSet::zero_ao_basis_set());
    auto mints = std::make_shared<MintsHelper>(primary, Process::environment.options);

    auto Pmn = mints->ao_eri(auxiliary_, zero_bas, primary, primary);
    int ao_dim = ints_->wfn()->nso();

    auto Pmn_mn = std::make_shared<Matrix>("Pmn mn block", ao_dim, ao_dim);
 
    auto Ppq_mat = std::make_shared<Matrix>("Ppq Matrix", naux, nmo * nmo);

    for(int aux_idx = 0; aux_idx < naux; ++aux_idx) {
        Pmn_mn->zero();

        for (int i = 0; i < ao_dim; ++i) {
            for (int j = 0; j < ao_dim; ++j) {
                auto val = Pmn->get(aux_idx, i * ao_dim + j);
                Pmn_mn->set(i, j, val);
            }
        }

        Pmn_mn->transform(ints_->wfn()->Ca_subset("AO"));

        for (int i = 0; i < nmo; ++i) {
            for (int j = 0; j < nmo; ++j) {
                auto val = Pmn_mn->get(i, j);
                Ppq_mat->set(aux_idx, i * nmo + j, val);
            }
        }
    }


    for (const std::string& block : blocklabels) {
        auto dfblk = "L" + block;

        auto label1 = block[0];
        auto label2 = block[1];

        (Ppq.block(dfblk)).iterate([&](const std::vector<size_t>& i, double& value) {
            auto val = Ppq_mat->get(i[0], (i[1] + pre_idx[label1]) * nmo + i[2] + pre_idx[label2]);
            value = val;
        });
    }

















}

void DSRG_MRPT2::set_v() {   
    if (eri_df_) {
        B = BTF_->build(tensor_type_, "B", {"Lgg", "LGG"});
        V = BTF_->build(CoreTensor, "Electron Repulsion Integral", spin_cases({"pphh"}));
        for (const std::string& block : B.block_labels()) {
            std::vector<size_t> iaux = label_to_spacemo_[block[0]];
            std::vector<size_t> ip   = label_to_spacemo_[block[1]];
            std::vector<size_t> ih   = label_to_spacemo_[block[2]];
            ambit::Tensor Bblock = ints_->three_integral_block(iaux, ip, ih);
            B.block(block).copy(Bblock);
        }
        V["abij"] =  B["gai"] * B["gbj"];
        V["abij"] -= B["gaj"] * B["gbi"];
        V["aBiJ"] =  B["gai"] * B["gBJ"];
        V["ABIJ"] =  B["gAI"] * B["gBJ"];
        V["ABIJ"] -= B["gAJ"] * B["gBI"];
    } else {
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
    }

    V_sumA_Alpha = BTF_->build(
        CoreTensor, "normal Dimension-reduced Electron Repulsion Integral alpha", {"gg"});
    V_sumB_Alpha = BTF_->build(CoreTensor,
                               "normal Dimension-reduced Electron Repulsion Integral beta", {"gg"});
    V_sumA_Beta = BTF_->build(
        CoreTensor, "index-reversed Dimension-reduced Electron Repulsion Integral beta", {"GG"});
    V_sumB_Beta = BTF_->build(
        CoreTensor, "normal Dimension-reduced Electron Repulsion Integral all beta", {"GG"});

    if (eri_df_) {
        // Summation of V["pmqm"] over index "m" or V["mpmq"] over index "m"
        V_sumA_Alpha["pq"]  = B["gpq"] * B["gmn"] * I["mn"];
        V_sumA_Alpha["pq"] -= B["gpm"] * B["gmq"];
        // Summation of V["pMqM"] over index "M"
        V_sumB_Alpha["pq"] = B["gpq"] * B["gMN"] * I["MN"];
        // Summation of V["mPmQ"] over index "m"
        V_sumA_Beta["PQ"] =  B["gmn"] * B["gPQ"] * I["mn"];
        // Summation of V["PMQM"] over index "M"
        V_sumB_Beta["PQ"] =  B["gPQ"] * B["gMN"] * I["MN"];
        V_sumB_Beta["PQ"] -= B["gPM"] * B["gMQ"];
    } else {
        // Summation of V["pmqm"] over index "m" or V["mpmq"] over index "m"
        V_sumA_Alpha["pq"] = V["pmqn"] * I["mn"];
        // Summation of V["pMqM"] over index "M"
        V_sumB_Alpha["pq"] = V["pMqN"] * I["MN"];
        // Summation of V["mPmQ"] over index "m"
        V_sumA_Beta["PQ"] = V["mPnQ"] * I["mn"];
        // Summation of V["PMQM"] over index "M"
        V_sumB_Beta["PQ"] = V["PMQN"] * I["MN"];
    }
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