/**
 * CASSCF gradient code by Shuhe Wang
 *
 * The computation procedure is listed as belows:
 * (1), Set MOs spaces;
 * (2), Set Tensors (F, H, V etc.);
 * (3), Compute and write the Lagrangian;
 * (4), Write 1RDMs and 2RDMs coefficients;
 * (5), Back-transform the TPDM.
 */
#include "casscf/casscf.h"
#include "gradient_tpdm/backtransform_tpdm.h"
#include "psi4/libmints/matrix.h"
#include "psi4/libmints/wavefunction.h"
#include "psi4/psifiles.h"
#include "helpers/printing.h"
#include "sci/aci.h"
#include "base_classes/active_space_solver.h"
#include "psi4/libmints/factory.h"
#include "psi4/libiwl/iwl.hpp"
#include "psi4/libpsio/psio.hpp"

using namespace ambit;
using namespace psi;
namespace forte {
/**
 * Initialize global variables.
 */
void CASSCF::set_all_variables() {
    // Set MOs containers.
    core_mos_ = mo_space_info_->corr_absolute_mo("RESTRICTED_DOCC");
    actv_mos_ = mo_space_info_->corr_absolute_mo("ACTIVE");
    virt_mos_ = mo_space_info_->corr_absolute_mo("RESTRICTED_UOCC");
    core_all_ = mo_space_info_->absolute_mo("RESTRICTED_DOCC");
    actv_all_ = mo_space_info_->absolute_mo("ACTIVE");
    core_mos_relative = mo_space_info_->get_relative_mo("RESTRICTED_DOCC");
    actv_mos_relative = mo_space_info_->get_relative_mo("ACTIVE");
    irrep_vec = mo_space_info_->dimension("ALL");

    // Set MO spaces.
    set_ambit_space();

    // Initialize tensors.
    Gamma1_ = BTF_->build(CoreTensor, "Gamma1", spin_cases({"aa"}));
    Gamma2_ = BTF_->build(CoreTensor, "Gamma2", spin_cases({"aaaa"}));
    H_ = BTF_->build(CoreTensor, "One-Electron Integral", spin_cases({"gg"}));
    V_ = BTF_->build(CoreTensor, "Electron Repulsion Integral", spin_cases({"gggg"}));
    F_ = BTF_->build(CoreTensor, "Fock Matrix", spin_cases({"gg"}));
    W_ = BTF_->build(CoreTensor, "Lagrangian", spin_cases({"gg"}));

    // Set tensors.
    set_tensor();
}

/**
 * Set ambit MO spaces.
 */
void CASSCF::set_ambit_space() {
    outfile->Printf("\n    Setting Ambit MO Space .......................... ");

    BlockedTensor::reset_mo_spaces();
    BlockedTensor::set_expert_mode(true);
    BTF_ = std::make_shared<BlockedTensorFactory>();
    std::string acore_label_ = "c";
    std::string aactv_label_ = "a";
    std::string avirt_label_ = "v";
    std::string bcore_label_ = "C";
    std::string bactv_label_ = "A";
    std::string bvirt_label_ = "V";

    // Add Ambit index labels.
    BTF_->add_mo_space(acore_label_, "m, n", core_mos_, AlphaSpin);
    BTF_->add_mo_space(bcore_label_, "M, N", core_mos_, BetaSpin);
    BTF_->add_mo_space(aactv_label_, "u, v, w, x, y, z", actv_mos_, AlphaSpin);
    BTF_->add_mo_space(bactv_label_, "U, V, W, X, Y, Z", actv_mos_, BetaSpin);
    BTF_->add_mo_space(avirt_label_, "e, f", virt_mos_, AlphaSpin);
    BTF_->add_mo_space(bvirt_label_, "E, F", virt_mos_, BetaSpin);

    // Define composite spaces.
    BTF_->add_composite_mo_space("h", "i, j, k, l", {acore_label_, aactv_label_});
    BTF_->add_composite_mo_space("H", "I, J, K, L", {bcore_label_, bactv_label_});
    BTF_->add_composite_mo_space("p", "a, b, c, d", {aactv_label_, avirt_label_});
    BTF_->add_composite_mo_space("P", "A, B, C, D", {bactv_label_, bvirt_label_});
    BTF_->add_composite_mo_space("g", "p, q, r, s", {acore_label_, aactv_label_, avirt_label_});
    BTF_->add_composite_mo_space("G", "P, Q, R, S", {bcore_label_, bactv_label_, bvirt_label_});

    outfile->Printf("Done");
}

/**
 * Set one-body and two-body densities.
 */
void CASSCF::set_density() {
    outfile->Printf("\n    Setting One- and Two-Body Density ............... ");

    // 1-body density
    Gamma1_.block("aa")("pq") = cas_ref_.g1a()("pq");
    Gamma1_.block("AA")("pq") = cas_ref_.g1b()("pq");

    // 2-body density
    Gamma2_.block("aaaa")("pqrs") = cas_ref_.g2aa()("pqrs");
    Gamma2_.block("aAaA")("pqrs") = cas_ref_.g2ab()("pqrs");
    Gamma2_.block("AAAA")("pqrs") = cas_ref_.g2bb()("pqrs");

    outfile->Printf("Done");
}

/**
 * Set one-electron integrals.
 */
void CASSCF::set_h() {
    outfile->Printf("\n    Setting One-Electron Integral ................... ");

    H_.iterate([&](const std::vector<size_t>& i, const std::vector<SpinType>& spin, double& value) {
        if (spin[0] == AlphaSpin) {
            value = ints_->oei_a(i[0], i[1]);
        } else {
            value = ints_->oei_b(i[0], i[1]);
        }
    });

    outfile->Printf("Done");
}

/**
 * Set two electron repulsion integrals.
 */
void CASSCF::set_v() {
    outfile->Printf("\n    Setting Two-Electron Integral ................... ");

    V_.iterate([&](const std::vector<size_t>& i, const std::vector<SpinType>& spin, double& value) {
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

    outfile->Printf("Done");
}

/**
 * Set Fock matrices.
 */
void CASSCF::set_fock() {
    outfile->Printf("\n    Setting Fock matrix ............................. ");

    psi::SharedMatrix D1a(new psi::Matrix("D1a", nmo_, nmo_));
    psi::SharedMatrix D1b(new psi::Matrix("D1b", nmo_, nmo_));

    // Fill core-core blocks
    for (size_t m = 0, ncore = core_mos_.size(); m < ncore; m++) {
        D1a->set(core_mos_[m], core_mos_[m], 1.0);
        D1b->set(core_mos_[m], core_mos_[m], 1.0);
    }

    // Fill active-active blocks
    Gamma1_.block("aa").citerate([&](const std::vector<size_t>& i, const double& value) {
        D1a->set(actv_mos_[i[0]], actv_mos_[i[1]], value);
    });
    Gamma1_.block("AA").citerate([&](const std::vector<size_t>& i, const double& value) {
        D1b->set(actv_mos_[i[0]], actv_mos_[i[1]], value);
    });

    // Make Fock matrices
    ints_->make_fock_matrix(D1a, D1b);
    F_.iterate([&](const std::vector<size_t>& i, const std::vector<SpinType>& spin, double& value) {
        if (spin[0] == AlphaSpin) {
            value = ints_->get_fock_a(i[0], i[1]);
        } else {
            value = ints_->get_fock_b(i[0], i[1]);
        }
    });

    outfile->Printf("Done");
}

/**
 * Initialize and set the Lagrangian.
 */
void CASSCF::set_lagrangian() {
    // Create a temporal container and an identity matrix
    BlockedTensor temp = BTF_->build(CoreTensor, "temporal tensor", spin_cases({"gg"}));
    BlockedTensor I = BTF_->build(CoreTensor, "identity matrix", spin_cases({"gg"}));
    I.iterate([&](const std::vector<size_t>& i, const std::vector<SpinType>&, double& value) {
        value = (i[0] == i[1]) ? 1.0 : 0.0;
    });

    // Set core-core and core-active block entries of Lagrangian.
    // Alpha.
    W_["mp"] = F_["mp"];
    // Beta.
    W_["MP"] = F_["MP"];

    // Set active-active block entries of Lagrangian.
    // Alpha.
    temp["vp"] = H_["vp"];
    temp["vp"] += V_["vmpn"] * I["mn"];
    temp["vp"] += V_["vMpN"] * I["MN"];
    W_["up"] += temp["vp"] * Gamma1_["uv"];
    W_["up"] += 0.5 * V_["xypv"] * Gamma2_["uvxy"];
    W_["up"] += V_["xYpV"] * Gamma2_["uVxY"];
    // Beta.
    temp["VP"] = H_["VP"];
    temp["VP"] += V_["mVnP"] * I["mn"];
    temp["VP"] += V_["VMPN"] * I["MN"];
    W_["UP"] += temp["VP"] * Gamma1_["UV"];
    W_["UP"] += 0.5 * V_["XYPV"] * Gamma2_["UVXY"];
    W_["UP"] += V_["yXvP"] * Gamma2_["vUyX"];
    // No need to set the rest symmetric blocks since they are 0
}

/**
 * Set densities, one-electron integrals, two-electron integrals and Fock matrices.
 */
void CASSCF::set_tensor() {
    set_density();
    set_h();
    set_v();
    set_fock();
}

/**
 * The procedure of TPDM back-transformation
 */
void CASSCF::tpdm_backtransform() {
    // This line of code is to deceive Psi4 and avoid computing scf gradient
    // Remove once TravisCI is updated
     ints_->wfn()->set_reference_wavefunction(ints_->wfn());

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

/**
 * The procedure of computing gradients
 */
SharedMatrix CASSCF::compute_gradient() {
    print_method_banner({"Complete Active Space Self Consistent Field Gradient", "Shuhe Wang"});
    set_all_variables();
    write_lagrangian();
    write_1rdm_spin_dependent();
    write_2rdm_spin_dependent();
    tpdm_backtransform();

    outfile->Printf("\n    Computing Gradient .............................. Done\n");
    return std::make_shared<Matrix>("nullptr", 0, 0);
}

/**
 * Write spin_dependent one-RDMs coefficients.
 *
 * We force "Da == Db". This function needs be changed if such constraint is revoked.
 */
void CASSCF::write_1rdm_spin_dependent() {
    outfile->Printf("\n    Writing 1RDM Coefficients ....................... ");

    SharedMatrix D1(new Matrix("1rdm coefficients contribution", nirrep_, irrep_vec, irrep_vec));

    for (size_t i = 0, size_c = core_mos_relative.size(); i < size_c; ++i) {
        D1->set(core_mos_relative[i].first, core_mos_relative[i].second,
                core_mos_relative[i].second, 1.0);
    }

    (Gamma1_.block("aa")).iterate([&](const std::vector<size_t>& i, double& value) {
        if (actv_mos_relative[i[0]].first == actv_mos_relative[i[1]].first) {
            D1->set(actv_mos_relative[i[0]].first, actv_mos_relative[i[0]].second,
                    actv_mos_relative[i[1]].second, value);
        }
    });

    D1->back_transform(ints_->Ca());
    ints_->wfn()->Da()->copy(D1);
    ints_->wfn()->Db()->copy(D1);

    outfile->Printf("Done");
}

/**
 * Write the Lagrangian.
 *
 * This function needs to be changed if frozen approximation is considered!
 */
void CASSCF::write_lagrangian() {
    outfile->Printf("\n    Writing Lagrangian .............................. ");

    set_lagrangian();
    SharedMatrix L(new Matrix("Lagrangian", nirrep_, irrep_vec, irrep_vec));

    for (const std::string& block : {"cc", "CC", "aa", "AA", "ca", "ac", "CA", "AC"}) {
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
        }

        (W_.block(block)).iterate([&](const std::vector<size_t>& i, double& value) {
            if (spin_pair[0][i[0]].first == spin_pair[1][i[1]].first) {
                L->add(spin_pair[0][i[0]].first, spin_pair[0][i[0]].second,
                       spin_pair[1][i[1]].second, value);
            }
        });
    }
    L->back_transform(ints_->Ca());
    ints_->wfn()->Lagrangian()->copy(L);

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
void CASSCF::write_2rdm_spin_dependent() {
    outfile->Printf("\n    Writing 2RDM Coefficients ....................... ");

    auto psio_ = _default_psio_lib_;
    IWL d2aa(psio_.get(), PSIF_MO_AA_TPDM, 1.0e-14, 0, 0);
    IWL d2ab(psio_.get(), PSIF_MO_AB_TPDM, 1.0e-14, 0, 0);
    IWL d2bb(psio_.get(), PSIF_MO_BB_TPDM, 1.0e-14, 0, 0);

    for (size_t i = 0, size_c = core_all_.size(); i < size_c; ++i) {
        auto m = core_all_[i];
        for (size_t j = 0; j < size_c; ++j) {
            auto n = core_all_[j];
            if (m != n) {
                d2aa.write_value(m, m, n, n, 0.25, 0, "NULL", 0);
                d2bb.write_value(m, m, n, n, 0.25, 0, "NULL", 0);
                d2aa.write_value(m, n, n, m, -0.25, 0, "NULL", 0);
                d2bb.write_value(m, n, n, m, -0.25, 0, "NULL", 0);
            }
            d2ab.write_value(m, m, n, n, 1.00, 0, "NULL", 0);
        }
    }

    for (size_t i = 0, size_c = core_all_.size(), size_a = actv_all_.size(); i < size_a; ++i) {
        auto u = actv_all_[i];
        for (size_t j = 0; j < size_a; ++j) {
            auto v = actv_all_[j];
            auto idx = i * na_ + j;
            auto gamma_a = Gamma1_.block("aa").data()[idx];
            auto gamma_b = Gamma1_.block("AA").data()[idx];

            for (size_t k = 0; k < size_c; ++k) {
                auto m = core_all_[k];
                d2aa.write_value(v, u, m, m, 0.5 * gamma_a, 0, "NULL", 0);
                d2bb.write_value(v, u, m, m, 0.5 * gamma_b, 0, "NULL", 0);
                d2aa.write_value(v, m, m, u, -0.5 * gamma_a, 0, "NULL", 0);
                d2bb.write_value(v, m, m, u, -0.5 * gamma_b, 0, "NULL", 0);
                d2ab.write_value(v, u, m, m, (gamma_a + gamma_b), 0, "NULL", 0);
            }
        }
    }

    for (size_t i = 0, size_a = actv_all_.size(); i < size_a; ++i) {
        auto u = actv_all_[i];
        for (size_t j = 0; j < size_a; ++j) {
            auto v = actv_all_[j];
            for (size_t k = 0; k < size_a; ++k) {
                auto x = actv_all_[k];
                for (size_t l = 0; l < size_a; ++l) {
                    auto y = actv_all_[l];
                    auto idx = i * na_ * na_ * na_ + j * na_ * na_ + k * na_ + l;
                    auto gamma_aa = Gamma2_.block("aaaa").data()[idx];
                    auto gamma_bb = Gamma2_.block("AAAA").data()[idx];
                    auto gamma_ab = Gamma2_.block("aAaA").data()[idx];
                    d2aa.write_value(x, u, y, v, 0.25 * gamma_aa, 0, "NULL", 0);
                    d2bb.write_value(x, u, y, v, 0.25 * gamma_bb, 0, "NULL", 0);
                    d2ab.write_value(x, u, y, v, gamma_ab, 0, "NULL", 0);
                }
            }
        }
    }

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

} // namespace forte
