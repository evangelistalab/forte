/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2020 by its authors (see COPYING, COPYING.LESSER, AUTHORS).
 *
 * The copyrights for code used from other parties are included in
 * the corresponding files.
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program.  If not, see http://www.gnu.org/licenses/.
 *
 * @END LICENSE
 */

#include "casscf/casscf_hess.h"
#include "integrals/integrals.h"
#include "base_classes/rdms.h"

#include "psi4/libfock/jk.h"
#include "psi4/libmints/matrix.h"
#include "psi4/libmints/wavefunction.h"
#include "psi4/libqt/qt.h"
#include "psi4/psifiles.h"
#include "psi4/libmints/basisset.h"

#include "helpers/printing.h"
#include "helpers/helpers.h"
#include "helpers/timer.h"
#include "sci/aci.h"
#include "fci/fci_solver.h"
#include "base_classes/active_space_solver.h"

#include "sci/fci_mo.h"
#include "orbital-helpers/mp2_nos.h"
#include "orbital-helpers/orbitaloptimizer.h"
#include "orbital-helpers/semi_canonicalize.h"

#include "psi4/libdiis/diisentry.h"
#include "psi4/libdiis/diismanager.h"
#include "psi4/libmints/factory.h"

using namespace ambit;

namespace forte {

CASSCF_HESS::CASSCF_HESS(const std::map<StateInfo, std::vector<double>>& state_weights_map,
                         std::shared_ptr<SCFInfo> scf_info, std::shared_ptr<ForteOptions> options,
                         std::shared_ptr<MOSpaceInfo> mo_space_info,
                         std::shared_ptr<ForteIntegrals> ints)
    : state_weights_map_(state_weights_map), scf_info_(scf_info), options_(options),
      mo_space_info_(mo_space_info), ints_(ints) {
    startup();
}

void CASSCF_HESS::setup_ambit() {
    BlockedTensor::reset_mo_spaces();
    BlockedTensor::set_expert_mode(true);

    BlockedTensor::add_mo_space("c", "ij", rdocc_mos_, NoSpin);
    BlockedTensor::add_mo_space("a", "tuvwyxz", actv_mos_, NoSpin);
    BlockedTensor::add_mo_space("v", "ab", ruocc_mos_, NoSpin);

    label_to_mos_["c"] = rdocc_mos_;
    label_to_mos_["a"] = actv_mos_;
    label_to_mos_["v"] = ruocc_mos_;

    BlockedTensor::add_composite_mo_space("o", "kl", {"c", "a"});
    BlockedTensor::add_composite_mo_space("g", "pqrs", {"c", "a", "v"});

    I_ = BlockedTensor::build(CoreTensor, "Identity", {"gg"});
    I_.iterate([&](const std::vector<size_t>& i, const std::vector<SpinType>&, double& value) {
        if (i[0] == i[1])
            value = 1.0;
    });
}

void CASSCF_HESS::read_options() {
    print_ = options_->get_int("PRINT");
    debug_print_ = options_->get_bool("CASSCF_DEBUG_PRINTING");

    maxiter_macro_ = options_->get_int("CASSCF_MAXITER");

    ci_type_ = options_->get_str("CASSCF_CI_SOLVER");

    // ignore rotation between certain pairs
}

void CASSCF_HESS::setup_mo_spaces() {
    rdocc_mos_ = mo_space_info_->corr_absolute_mo("RESTRICTED_DOCC");
    actv_mos_ = mo_space_info_->corr_absolute_mo("ACTIVE");
    ruocc_mos_ = mo_space_info_->corr_absolute_mo("RESTRICTED_UOCC");

    // Not sure the following still needed or not
    nsopi_ = scf_info_->nsopi();
    nirrep_ = mo_space_info_->nirrep();

    core_mos_abs_ = mo_space_info_->absolute_mo("RESTRICTED_DOCC");
    actv_mos_abs_ = mo_space_info_->absolute_mo("ACTIVE");
    core_mos_rel_ = mo_space_info_->get_relative_mo("RESTRICTED_DOCC");
    actv_mos_rel_ = mo_space_info_->get_relative_mo("ACTIVE");
    virt_mos_rel_ = mo_space_info_->get_relative_mo("RESTRICTED_UOCC");

    frozen_docc_dim_ = mo_space_info_->dimension("FROZEN_DOCC");
    restricted_docc_dim_ = mo_space_info_->dimension("RESTRICTED_DOCC");
    active_dim_ = mo_space_info_->dimension("ACTIVE");
    restricted_uocc_dim_ = mo_space_info_->dimension("RESTRICTED_UOCC");
    inactive_docc_dim_ = mo_space_info_->dimension("INACTIVE_DOCC");
    corr_dim_ = mo_space_info_->dimension("CORRELATED");
    nmo_dim_ = mo_space_info_->dimension("ALL");

    corr_mos_ = mo_space_info_->corr_absolute_mo("CORRELATED");

    ncmo_ = mo_space_info_->size("CORRELATED");
    nmo_ = mo_space_info_->size("ALL");
    nrdocc_ = rdocc_mos_.size();
    nruocc_ = ruocc_mos_.size();
    nfdocc_ = mo_space_info_->size("FROZEN_DOCC");
    nactv_ = mo_space_info_->size("ACTIVE");
}

void CASSCF_HESS::startup() {
    read_options();

    setup_mo_spaces();

    setup_ambit();

    Enuc_ = ints_->nuclear_repulsion_energy();

    build_ints();
}

void CASSCF_HESS::build_ints() {
    oei_ = BlockedTensor::build(CoreTensor, "OEI", {"gg"});
    oei_.iterate([&](const std::vector<size_t>& i, const std::vector<SpinType>&, double& value) {
        value = ints_->oei_a(i[0], i[1]);
    });
    outfile->Printf("\n FUCK1");

    tei_ = BlockedTensor::build(CoreTensor, "TEI", {"ggoo", "oovo", "vovo"});
    for (const std::string& block : tei_.block_labels()) {
        std::vector<size_t> i0 = label_to_mos_[block.substr(0, 1)];
        std::vector<size_t> i1 = label_to_mos_[block.substr(1, 1)];
        std::vector<size_t> i2 = label_to_mos_[block.substr(2, 1)];
        std::vector<size_t> i3 = label_to_mos_[block.substr(3, 1)];

        auto Vblock = ints_->aptei_ab_block(i0, i2, i1, i3);
        tei_.block(block)("pqrs") = Vblock("prqs");
    }
    outfile->Printf("\n FUCK2");

    build_L4();
    outfile->Printf("\n FUCK3");
}

void CASSCF_HESS::compute_closed_shell() {
    Fc_ = BlockedTensor::build(CoreTensor, "Fc", {"gg"});
    Fc_["rs"] = oei_["rs"];
    Fc_["rs"] += 2.0 * tei_["rsij"] * I_["ij"];
    Fc_["rs"] -= tei_["rijs"] * I_["ij"];

    Ec_ = oei_["ij"] * I_["ij"] + Fc_["ij"] * I_["ij"];
}

void CASSCF_HESS::compute_init_macro_energy() {
    compute_closed_shell();

    E0_ = Ec_ + Enuc_;
    E0_ += Fc_["tu"] * d1_["tu"];
    E0_ += 0.5 * tei_["tuvw"] * d2_["tuvw"];
}

void CASSCF_HESS::build_L4() {
    L4_ = BlockedTensor::build(CoreTensor, "L4", {"ocgg"});
    L4_["kjrs"] = 4.0 * tei_["rksj"];
    L4_["kjrs"] -= tei_["skrj"];
    L4_["kjrs"] -= tei_["rskj"];
}

void CASSCF_HESS::build_fock() {
    F_ = BlockedTensor::build(CoreTensor, "Fock", {"gg"});
    F_["rs"] = Fc_["rs"];

    F_["rs"] += d1_["tu"] * tei_["rstu"];
    F_["rs"] -= 0.5 * d1_["tu"] * tei_["rtsu"];
}

double CASSCF_HESS::compute_2nd_o() {
    double E2 = 0.0;

    // build A
    A_ = BlockedTensor::build(CoreTensor, "A", {"gg"});
    A_["ri"] += 2.0 * F_["ri"];
    A_["ru"] += Fc_["rt"] * d1_["tu"];
    A_["ru"] += tei_["rtvw"] * d2_["tuvw"];

    // build G
    G4_ = BlockedTensor::build(CoreTensor, "G4", {"oogg"});
    G4_["ijrs"] += 2.0 * L4_["ijrs"];
    G4_["ijrs"] += 2.0 * F_["rs"] * I_["ij"];

    G4_["tjrs"] += L4_["vjrs"] * d1_["tv"];
    G4_["jtsr"] += L4_["vjrs"] * d1_["tv"];

    G4_["turs"] += Fc_["rs"] * d1_["tu"];
    G4_["turs"] += tei_["rsvw"] * d2_["tuvw"];
    G4_["turs"] += 2.0 * tei_["rvsw"] * d2_["tvuw"];

    // build B
    B_ = BlockedTensor::build(CoreTensor, "B", {"gg"});
    B_["rk"] += A_["rk"];
    B_["rk"] += G4_["klrs"] * T_["sl"];

    // compute 2nd-order energy for fixed CI coefficients
    E2 = E0_;
    E2 += 2.0 * T_["rk"] * A_["rk"];
    E2 += 2.0 * T_["rk"] * G4_["klrs"] * T_["sl"];

    return E2;
}

void CASSCF_HESS::build_2nd_actv_ints() {
    Ec2_ = Ec_;
    Ec2_ += 4.0 * Fc_["ri"] * T_["ri"];
    Ec2_ += 2.0 * Fc_["rs"] * I_["ij"] * T_["ri"] * T_["sj"];
    Ec2_ += 2.0 * L4_["ijrs"] * T_["ri"] * T_["sj"];

    Fc2_ = BlockedTensor::build(CoreTensor, "Fc2", {"aa"});
    Fc2_["tu"] += Fc_["rs"] * U_["rt"] * U_["su"];

    Fc2_["tu"] += 2.0 * tei_["rstu"] * U_["ri"] * U_["si"];
    Fc2_["tu"] -= 2.0 * tei_["ijtu"] * I_["ij"];

    Fc2_["tu"] -= tei_["rtsu"] * U_["ri"] * U_["si"];
    Fc2_["tu"] += tei_["itju"] * I_["ij"];

    Fc2_["tu"] += L4_["uirs"] * T_["rt"] * T_["si"];
    Fc2_["tu"] += L4_["tirs"] * T_["ru"] * T_["si"];

    Vab2_ = BlockedTensor::build(CoreTensor, "Vab2", {"aaaa"});
    Vab2_["tuvw"] -= tei_["tuvw"];

    Vab2_["tuvw"] += tei_["rsvw"] * U_["rt"] * U_["su"];
    Vab2_["tuvw"] += tei_["rstu"] * U_["rv"] * U_["sw"];

    Vab2_["tuvw"] += tei_["rtsv"] * T_["ru"] * T_["sw"];
    Vab2_["tuvw"] += tei_["rusv"] * T_["rt"] * T_["sw"];
    Vab2_["tuvw"] += tei_["rtsw"] * T_["ru"] * T_["sv"];
    Vab2_["tuvw"] += tei_["rusw"] * T_["rt"] * T_["sv"];
}

double CASSCF_HESS::compute_2nd_ci() {
    // prepare active integrals
    std::vector<int> actv_sym = mo_space_info_->symmetry("ACTIVE");
    auto fci_ints = std::make_shared<ActiveSpaceIntegrals>(ints_, actv_mos_, actv_sym, rdocc_mos_);

    auto aa = ambit::Tensor::build(ambit::CoreTensor, "AA", {nactv_, nactv_, nactv_, nactv_});
    auto ab = ambit::Tensor::build(ambit::CoreTensor, "AB", {nactv_, nactv_, nactv_, nactv_});

    ab("tvuw") = Vab2_.block("aaaa")("tuvw");
    outfile->Printf("\n ab norm %.15f", ab.norm(2));
    aa("tuvw") = ab("tuvw") - ab("tuwv");
    outfile->Printf("\n aa norm %.15f", aa.norm(2));

    fci_ints->set_active_integrals(aa, ab, aa);

    auto oei = Fc2_.block("aa").data();
    fci_ints->set_restricted_one_body_operator(oei, oei);

    fci_ints->set_scalar_energy(Ec2_);

    // diagonalize
    auto state_map = to_state_nroots_map(state_weights_map_);
    auto active_space_solver = make_active_space_solver(ci_type_, state_map, scf_info_,
                                                        mo_space_info_, fci_ints, options_);
    active_space_solver->set_print(print_);
    const auto state_energies_map = active_space_solver->compute_energy();
    rdms_ = active_space_solver->compute_average_rdms(state_weights_map_, 2);
    E_casscf_ = compute_average_state_energy(state_energies_map, state_weights_map_);

    return E_casscf_;
}

void CASSCF_HESS::build_density() {
    d1_ = BlockedTensor::build(CoreTensor, "D1", {"aa"});
    d2_ = BlockedTensor::build(CoreTensor, "D2", {"aaaa"});

    d1_.block("aa")("uv") = rdms_.g1a()("uv") + rdms_.g1b()("uv");

    auto g2 = rdms_.SFg2();
    d2_.block("aaaa")("tuvw") = 0.5 * g2("tvuw");
    d2_.block("aaaa")("tuvw") += 0.5 * g2("uvtw");
}

void CASSCF_HESS::orbital_step() {
    // build Atilde and Gtilde
    Atilde_ = BlockedTensor::build(CoreTensor, "Atilde", {"gg"});
    Atilde_["rk"] = B_["sk"] * U_["sr"];

    Gtilde_ = BlockedTensor::build(CoreTensor, "Gtilde", {"oogg"});
    Gtilde_["klrs"] = G4_["klpq"] * U_["pr"] * U_["qs"];

    // form gradient and hessian
    auto grad = BlockedTensor::build(CoreTensor, "grad", {"go"});
    grad["rk"] = Atilde_["rk"];
    grad["rk"] -= Atilde_["kr"];

    auto hess = BlockedTensor::build(CoreTensor, "hess", {"gogo"});
    hess["rksl"] += Gtilde_["klrs"];
    hess["rksl"] -= 0.5 * I_["kl"] * Atilde_["rs"];
    hess["rksl"] -= 0.5 * I_["kl"] * Atilde_["sr"];

    hess["rksl"] -= Gtilde_["rlks"];
    hess["rksl"] += 0.5 * I_["rl"] * Atilde_["ks"];
    hess["rksl"] += 0.5 * I_["rl"] * Atilde_["sk"];

    hess["rksl"] -= Gtilde_["ksrl"];
    hess["rksl"] += 0.5 * I_["ks"] * Atilde_["rl"];
    hess["rksl"] += 0.5 * I_["ks"] * Atilde_["lr"];

    hess["rksl"] += Gtilde_["rskl"];
    hess["rksl"] -= 0.5 * I_["rs"] * Atilde_["kl"];
    hess["rksl"] -= 0.5 * I_["rs"] * Atilde_["lk"];

    // form the augmented Hessian matrix
    auto ac = nactv_ * nrdocc_;
    auto vc = nruocc_ * nrdocc_;
    auto va = nruocc_ * nactv_;
    auto dim = ac + vc + va + 1;

    std::map<std::string, size_t> shifts;
    shifts["ac"] = 1;
    shifts["vc"] = ac + 1;
    shifts["va"] = ac + vc + 1;

    std::map<char, size_t> sizes;
    sizes['c'] = nrdocc_;
    sizes['a'] = nactv_;
    sizes['v'] = nruocc_;

    auto ah = std::make_shared<psi::Matrix>("AH", dim, dim);
    for (const std::string& block1 : {"ac"}) {
        size_t dim1 = sizes[block1[0]];
        size_t dim2 = sizes[block1[1]];
        size_t shift1 = shifts[block1];

        auto data1 = grad.block(block1).data();

        for (size_t p = 0; p < dim1; ++p) {
            for (size_t q = 0; q < dim2; ++q) {
                size_t idx = p * dim2 + q;
                double v = data1[idx];
                ah->set(idx + shift1, 0, v);
                ah->set(0, idx + shift1, v);
            }
        }

        for (const std::string& block2 : {"ac"}) {
            size_t dim3 = sizes[block2[0]];
            size_t dim4 = sizes[block2[1]];
            size_t shift2 = shifts[block2];

            auto data2 = hess.block(block1 + block2).data();

            for (size_t p = 0; p < dim1; ++p) {
                for (size_t q = 0; q < dim2; ++q) {
                    for (size_t r = 0; r < dim3; ++r) {
                        for (size_t s = 0; s < dim4; ++s) {
                            size_t idx = p * dim2 * dim3 * dim4 + q * dim3 * dim4 + r * dim4 + s;
                            double v = data2[idx];
                            ah->set(p * dim2 + q + shift1, r * dim4 + s + shift2, v);
                        }
                    }
                }
            }
        }
    }

    ah->print();
}

double CASSCF_HESS::compute_energy() {
    // init U and T
    U_ = BlockedTensor::build(CoreTensor, "U", {"gg"});
    U_.iterate([&](const std::vector<size_t>& i, const std::vector<SpinType>&, double& value) {
        if (i[0] == i[1])
            value = 1.0;
    });

    T_ = BlockedTensor::build(CoreTensor, "T", {"gg"});
    T_["pq"] = U_["pq"] - I_["pq"];

    compute_closed_shell();
    outfile->Printf("\n !!Ec = %20.15f", Ec_);
    build_2nd_actv_ints();
    outfile->Printf("\n DONE 2nd actv");
    compute_2nd_ci();
    build_density();
    compute_init_macro_energy();
    outfile->Printf("\n !!E0 = %20.15f", E0_);
    build_L4();
    build_fock();
    double E2 = compute_2nd_o();
    outfile->Printf("\n !!E2 = %20.15f", E2);
    orbital_step();

    return 0.0;
    //    if (nactv_ == 0) {
    //        outfile->Printf("\n\n\n Please set the active space");
    //        throw psi::PSIEXCEPTION(" The active space is zero.  Set the active space");
    //    } else if (nactv_ == ncmo_) {
    //        outfile->Printf("\n Your about to do an all active CASSCF");
    //        throw psi::PSIEXCEPTION("The active space is all the MOs.  Orbitals don't "
    //                                "matter at this point");
    //    }

    //    // Provide a nice summary at the end for iterations
    //    std::vector<int> iter_con;

    //    // Frozen-core C Matrix is never rotated
    //    if (nfdocc_ > 0) {
    //        F_frozen_core_ = set_frozen_core_orbitals();
    //    }

    //    // DIIS options
    //    bool do_diis = options_->get_bool("CASSCF_DO_DIIS");
    //    int diis_freq = options_->get_int("CASSCF_DIIS_FREQ");
    //    int diis_start = options_->get_int("CASSCF_DIIS_START");
    //    int diis_max_vec = options_->get_int("CASSCF_DIIS_MAX_VEC");
    //    double diis_gradient_norm = options_->get_double("CASSCF_DIIS_NORM");

    //    // CI update options
    //    bool ci_step = options_->get_bool("CASSCF_CI_STEP");
    //    int casscf_freq = options_->get_int("CASSCF_CI_FREQ");

    //    double rotation_max_value = options_->get_double("CASSCF_MAX_ROTATION");

    //    psi::Dimension nhole_dim = mo_space_info_->dimension("GENERALIZED HOLE");
    //    psi::Dimension npart_dim = mo_space_info_->dimension("GENERALIZED PARTICLE");
    //    psi::SharedMatrix S(new psi::Matrix("Orbital Rotation", nirrep_, nhole_dim, npart_dim));
    //    psi::SharedMatrix Sstep;

    //    // Setup the DIIS manager
    //    auto diis_manager = std::make_shared<DIISManager>(
    //        diis_max_vec, "MCSCF DIIS", DIISManager::OldestAdded, DIISManager::InCore);
    //    diis_manager->set_error_vector_size(1, DIISEntry::Matrix, S.get());
    //    diis_manager->set_vector_size(1, DIISEntry::Matrix, S.get());

    //    int diis_count = 0;

    //    E_casscf_ = 0.0;
    //    double E_casscf_old = 0.0, Ediff = 0.0;
    //    std::shared_ptr<psi::Matrix> C_start = ints_->Ca()->clone();
    //    double econv = options_->get_double("CASSCF_E_CONVERGENCE");
    //    double gconv = options_->get_double("CASSCF_G_CONVERGENCE");

    //    psi::SharedMatrix Ca = ints_->Ca();

    //    print_h2("CASSCF Iteration");
    //    outfile->Printf("\n  iter    ||g||           Delta_E            E_CASSCF CONV_TYPE");

    //    for (int iter = 1; iter <= maxiter_macro_; iter++) {
    //        local_timer casscf_total_iter;

    //        local_timer trans_ints_timer;
    //        tei_gaaa_ = transform_integrals(Ca);
    //        if (print_ > 0) {
    //            outfile->Printf("\n\n  Transform Integrals takes %8.8f s.",
    //            trans_ints_timer.get());
    //        }
    //        iter_con.push_back(iter);

    //        // Perform a CASCI
    //        E_casscf_old = E_casscf_;
    //        if (print_ > 0) {
    //            std::string ci_type = options_->get_str("CASSCF_CI_SOLVER");
    //            outfile->Printf("\n\n  Performing a CAS with %s", ci_type.c_str());
    //        }
    //        local_timer cas_timer;
    //        // Perform a DMRG-CI, ACI, FCI inside an active space
    //        // If casscf_freq is on, do the CI step every casscf_iteration
    //        // for example, every 5 iterations, run a CASCI.
    //        if (ci_step) {
    //            if ((iter < casscf_freq) || (iter % casscf_freq) == 0) {
    //                diagonalize_hamiltonian();
    //            }
    //        } else {
    //            diagonalize_hamiltonian();
    //        }
    //        if (print_ > 0) {
    //            outfile->Printf("\n\n CAS took %8.6f seconds.", cas_timer.get());
    //        }

    //        CASSCFOrbitalOptimizer orbital_optimizer(gamma1_, gamma2_, tei_gaaa_, options_,
    //                                                 mo_space_info_);

    //        orbital_optimizer.set_scf_info(scf_info_);
    //        orbital_optimizer.set_frozen_one_body(F_frozen_core_);
    //        orbital_optimizer.set_symmmetry_mo(Ca);
    //        orbital_optimizer.one_body(Hcore_->clone());
    //        if (print_ > 0) {
    //            orbital_optimizer.set_print_timings(true);
    //        }
    //        orbital_optimizer.set_jk(JK_);
    //        orbital_optimizer.update();
    //        double g_norm = orbital_optimizer.orbital_gradient_norm();

    //        Ediff = E_casscf_ - E_casscf_old;
    //        if (iter > 1 && std::fabs(Ediff) < econv && g_norm < gconv) {

    //            outfile->Printf("\n  %4d   %10.12f   %10.12f   %10.12f  %10.6f s", iter, g_norm,
    //            Ediff,
    //                            E_casscf_, casscf_total_iter.get());

    //            outfile->Printf("\n\n  A miracle has come to pass: CASSCF iterations have
    //            converged."); break;
    //        }

    //        Sstep = orbital_optimizer.approx_solve();

    //        // Max rotation
    //        double maxS = Sstep->absmax();
    //        if (maxS > rotation_max_value) {
    //            Sstep->scale(rotation_max_value / maxS);
    //        }

    //        // Add step to overall rotation
    //        S->add(Sstep);

    //        // TODO:  Add options controlled.  Iteration and g_norm
    //        if (do_diis and (iter >= diis_start or g_norm < diis_gradient_norm)) {
    //            diis_manager->add_entry(2, Sstep.get(), S.get());
    //            diis_count++;
    //        }

    //        if (do_diis and iter > diis_start and (diis_count % diis_freq == 0)) {
    //            diis_manager->extrapolate(1, S.get());
    //        }
    //        psi::SharedMatrix Cp = orbital_optimizer.rotate_orbitals(C_start, S);

    //        // update MO coefficients
    //        Ca->copy(Cp);

    //        std::string diis_start_label = "";
    //        if (do_diis and (iter > diis_start or g_norm < diis_gradient_norm)) {
    //            diis_start_label = "DIIS";
    //        }
    //        outfile->Printf("\n %4d %14.12f %18.12f %18.12f %6.1f s  %4s ~", iter, g_norm, Ediff,
    //                        E_casscf_, casscf_total_iter.get(), diis_start_label.c_str());
    //    }

    //    diis_manager->delete_diis_file();
    //    diis_manager.reset();

    //    // if(casscf_debug_print_)
    //    //{
    //    //    overlap_orbitals(this->Ca(), C_start);
    //    //}
    //    //    if (options_->get_bool("MONITOR_SA_SOLUTION")) {
    //    //        overlap_coefficients();
    //    //    }

    //    outfile->Printf("\n\n @ Final CASSCF Energy = %20.15f\n", E_casscf_);
    //    if (iter_con.size() == size_t(maxiter_macro_) && maxiter_macro_ > 1) {
    //        outfile->Printf("\n CASSCF did not converge");
    //        throw psi::PSIEXCEPTION("CASSCF did not converge.");
    //    }

    //    // semicanonicalize orbitals
    //    auto U = semicanonicalize(Ca);
    //    auto Ca_semi = linalg::doublet(Ca, U, false, false);
    //    Ca_semi->set_name(Ca->name());

    //    // restransform integrals if derivatives are needed
    //    if (options_->get_str("DERTYPE") == "FIRST") {
    //        print_h2("Final Integral Transformation for CASSCF Gradients");
    //        ints_->update_orbitals(Ca_semi, Ca_semi);

    //        // diagonalize the Hamiltonian one last time
    //        diagonalize_hamiltonian();
    //    } else {
    //        ints_->wfn()->Ca()->copy(Ca_semi);
    //    }

    //    psi::Process::environment.globals["CURRENT ENERGY"] = E_casscf_;
    //    psi::Process::environment.globals["CASSCF_ENERGY"] = E_casscf_;

    //    return E_casscf_;
}

void CASSCF_HESS::diagonalize_hamiltonian() {
    // perform a CAS-CI with the active given in the input
    std::shared_ptr<ActiveSpaceIntegrals> fci_ints = get_ci_integrals();

    std::string casscf_ci_type = options_->get_str("CASSCF_CI_SOLVER");

    auto state_map = to_state_nroots_map(state_weights_map_);
    auto active_space_solver = make_active_space_solver(casscf_ci_type, state_map, scf_info_,
                                                        mo_space_info_, fci_ints, options_);
    active_space_solver->set_print(print_);
    const auto state_energies_map = active_space_solver->compute_energy();
    rdms_ = active_space_solver->compute_average_rdms(state_weights_map_, 2);
    E_casscf_ = compute_average_state_energy(state_energies_map, state_weights_map_);

    // Compute 1-RDM
    gamma1_ = rdms_.g1a().clone();
    gamma1_("ij") += rdms_.g1b()("ij");

    // Compute 2-RDM
    gamma2_ = rdms_.SFg2();
}

std::shared_ptr<psi::Matrix> CASSCF_HESS::set_frozen_core_orbitals() {
    auto Ca = ints_->Ca();
    auto C_core = std::make_shared<psi::Matrix>("C_core", nirrep_, nsopi_, frozen_docc_dim_);

    // Need to get the frozen block of the C matrix
    for (size_t h = 0; h < nirrep_; h++) {
        for (int i = 0; i < frozen_docc_dim_[h]; i++) {
            C_core->set_column(h, i, Ca->get_column(h, i));
        }
    }

    JK_->set_do_K(true);
    std::vector<std::shared_ptr<psi::Matrix>>& Cl = JK_->C_left();
    std::vector<std::shared_ptr<psi::Matrix>>& Cr = JK_->C_right();

    Cl.clear();
    Cr.clear();
    Cl.push_back(C_core); // Cr is the same as Cl

    JK_->compute();

    psi::SharedMatrix F_core = JK_->J()[0];
    psi::SharedMatrix K_core = JK_->K()[0];

    F_core->scale(2.0);
    F_core->subtract(K_core);

    return F_core;
}

ambit::Tensor CASSCF_HESS::transform_integrals(std::shared_ptr<psi::Matrix> Ca) {
    // This function will do an integral transformation using the JK builder,
    // and return the integrals of type <px|uy> = (pu|xy).
    // This was borrowed from Kevin Hannon's IntegralTransform Plugin.

    // Transform C matrix to C1 symmetry
    size_t nso = scf_info_->nso();
    psi::SharedMatrix aotoso = ints_->aotoso();
    auto Ca_nosym = std::make_shared<psi::Matrix>(nso, nmo_);

    // Transform from the SO to the AO basis for the C matrix.
    // just transfroms the C_{mu_ao i} -> C_{mu_so i}
    local_timer CSO2AO;
    for (size_t h = 0, index = 0; h < nirrep_; ++h) {
        for (int i = 0; i < nmo_dim_[h]; ++i) {
            size_t nao = nso;
            size_t nso = nsopi_[h];

            if (!nso)
                continue;

            C_DGEMV('N', nao, nso, 1.0, aotoso->pointer(h)[0], nso, &Ca->pointer(h)[0][i],
                    nmo_dim_[h], 0.0, &Ca_nosym->pointer()[0][index], nmo_dim_.sum());

            index += 1;
        }
    }
    if (print_ > 1) {
        outfile->Printf("\n  CSO2AO takes %8.4f s.", CSO2AO.get());
    }

    // set up the active part of the C matrix
    auto Cact = std::make_shared<psi::Matrix>("Cact", nso, nactv_);
    std::vector<std::shared_ptr<psi::Matrix>> Cact_vec(nactv_);
    for (size_t x = 0; x < nactv_; x++) {
        psi::SharedVector Ca_nosym_vec = Ca_nosym->get_column(0, actv_mos_abs_[x]);
        Cact->set_column(0, x, Ca_nosym_vec);

        std::string name = "Cact slice " + std::to_string(x);
        auto temp = std::make_shared<psi::Matrix>(name, nso, 1);
        temp->set_column(0, 0, Ca_nosym_vec);
        Cact_vec[x] = temp;
    }

    // The following type of integrals are needed:
    // (pu|xy) = C_{Mp}^T C_{Nu} C_{Rx}^T C_{Sy} (MN|RS)
    //         = C_{Mp}^T C_{Nu} J_{MN}^{xy}
    //         = C_{Mp}^T J_{MN}^{xy} C_{Nu}

    JK_->set_memory(psi::Process::environment.get_memory() * 0.85);
    JK_->set_do_K(false);
    std::vector<psi::SharedMatrix>& Cl = JK_->C_left();
    std::vector<psi::SharedMatrix>& Cr = JK_->C_right();
    Cl.clear();
    Cr.clear();

    for (size_t x = 0; x < nactv_; ++x) {
        for (size_t y = x; y < nactv_; ++y) {
            Cl.push_back(Cact_vec[x]);
            Cr.push_back(Cact_vec[y]);
        }
    }

    JK_->compute();

    auto half_trans = std::make_shared<psi::Matrix>("Trans", nmo_, nactv_);
    auto corr_mos_abs = mo_space_info_->absolute_mo("CORRELATED");

    std::vector<size_t> ints_dim{ncmo_, nactv_, nactv_, nactv_};
    auto ints = ambit::Tensor::build(ambit::CoreTensor, "gaaa ints", ints_dim);
    std::vector<double>& ints_data = ints.data();
    size_t nactv2 = nactv_ * nactv_;
    size_t nactv3 = nactv2 * nactv_;

    for (size_t x = 0, shift = 0; x < nactv_; ++x) {
        shift += x;
        for (size_t y = x; y < nactv_; ++y) {
            std::shared_ptr<psi::Matrix> J = JK_->J()[x * nactv_ + y - shift];
            half_trans = psi::linalg::triplet(Ca_nosym, J, Cact, true, false, false);

            for (size_t p = 0; p < ncmo_; ++p) {
                for (size_t u = 0; u < nactv_; ++u) {
                    double value = half_trans->get(corr_mos_abs[p], u);
                    ints_data[corr_mos_[p] * nactv3 + x * nactv2 + u * nactv_ + y] = value;
                    ints_data[corr_mos_[p] * nactv3 + y * nactv2 + u * nactv_ + x] = value;
                }
            }
        }
    }

    return ints;
}

std::shared_ptr<ActiveSpaceIntegrals> CASSCF_HESS::get_ci_integrals() {
    std::vector<int> actv_sym = mo_space_info_->symmetry("ACTIVE");
    auto fci_ints = std::make_shared<ActiveSpaceIntegrals>(ints_, actv_mos_, actv_sym, rdocc_mos_);

    if (!(options_->get_bool("RESTRICTED_DOCC_JK"))) {
        fci_ints->set_active_integrals_and_restricted_docc();
    } else {
        auto active_aa = ambit::Tensor::build(ambit::CoreTensor, "ActiveIntegralsAA",
                                              {nactv_, nactv_, nactv_, nactv_});
        auto active_ab = ambit::Tensor::build(ambit::CoreTensor, "ActiveIntegralsAB",
                                              {nactv_, nactv_, nactv_, nactv_});
        const std::vector<double>& tei_paaa_data = tei_gaaa_.data();

        size_t nactv2 = nactv_ * nactv_;
        size_t nactv3 = nactv2 * nactv_;

        active_ab.iterate([&](const std::vector<size_t>& i, double& value) {
            value = tei_paaa_data[actv_mos_[i[0]] * nactv3 + i[1] * nactv2 + i[2] * nactv_ + i[3]];
        });

        active_aa.copy(active_ab);
        active_aa("u,v,x,y") -= active_ab("u, v, y, x");

        fci_ints->set_active_integrals(active_aa, active_ab, active_aa);
        if (debug_print_) {
            outfile->Printf("\n\n  tei_active_aa: %8.8f, tei_active_ab: %8.8f", active_aa.norm(2),
                            active_ab.norm(2));
        }

        auto oei = compute_restricted_docc_operator();
        fci_ints->set_restricted_one_body_operator(oei, oei);
        fci_ints->set_scalar_energy(scalar_energy_);
    }

    return fci_ints;
}

std::vector<double> CASSCF_HESS::compute_restricted_docc_operator() {
    auto Ca = ints_->Ca();

    double Edocc = 0.0;                         // energy from restricted docc
    double Efrzc = ints_->frozen_core_energy(); // energy from frozen docc
    scalar_energy_ = ints_->scalar();           // scalar energy from the Integral class

    // bare one-electron integrals
    std::shared_ptr<psi::Matrix> Hcore = Hcore_->clone();
    Hcore->transform(Ca);

    // one-electron integrals dressed by inactive orbitals
    std::vector<double> oei(nactv_ * nactv_, 0.0);
    // one-electron integrals in SharedMatrix format, set to MO Hcore by default
    std::shared_ptr<psi::Matrix> oei_shared_matrix = Hcore;

    // special case when there is no inactive docc
    if (nrdocc_ + nfdocc_ != 0) {
        // compute inactive Fock
        auto Fdocc = build_fock_inactive(Ca);
        oei_shared_matrix = Fdocc;

        // compute energy from inactive docc
        for (size_t h = 0; h < nirrep_; h++) {
            for (int rd = 0; rd < inactive_docc_dim_[h]; rd++) {
                Edocc += Hcore->get(h, rd, rd) + Fdocc->get(h, rd, rd);
            }
        }

        // Edocc includes frozen-core energy and should be subtracted
        scalar_energy_ += Edocc - Efrzc;
    }

    // fill in oei data
    for (size_t u = 0; u < nactv_; ++u) {
        size_t h = actv_mos_rel_[u].first;   // irrep
        size_t nu = actv_mos_rel_[u].second; // index

        for (size_t v = 0; v < nactv_; ++v) {
            if (actv_mos_rel_[v].first != h)
                continue;

            size_t nv = actv_mos_rel_[v].second;

            oei[u * nactv_ + v] = oei_shared_matrix->get(h, nu, nv);
        }
    }

    if (debug_print_) {
        for (size_t u = 0; u < nactv_; u++) {
            for (size_t v = 0; v < nactv_; v++) {
                outfile->Printf("\n  oei(%d, %d) = %8.8f", u, v, oei[u * nactv_ + v]);
            }
        }

        outfile->Printf("\n Frozen Core Energy = %8.8f", Efrzc);
        outfile->Printf("\n Restricted Energy = %8.8f", Edocc - Efrzc);
        outfile->Printf("\n Scalar Energy = %8.8f", scalar_energy_);
    }

    return oei;
}

std::shared_ptr<psi::Matrix> CASSCF_HESS::semicanonicalize(std::shared_ptr<psi::Matrix> Ca) {
    print_h2("Semi-canonicalize CASSCF Orbitals");

    // build averaged Fock matrix
    outfile->Printf("\n    Building Fock matrix  ...");
    auto F = build_fock(Ca);
    outfile->Printf(" Done.");

    // unitary rotation matrix for output
    auto U = std::make_shared<psi::Matrix>("U_CAS_SEMI", nmo_dim_, nmo_dim_);
    U->identity();

    // diagonalize three sub-blocks (restricted_docc, active, restricted_uocc)
    std::vector<psi::Dimension> mos_dim{restricted_docc_dim_, active_dim_, restricted_uocc_dim_};
    std::vector<psi::Dimension> mos_offsets{frozen_docc_dim_, inactive_docc_dim_,
                                            inactive_docc_dim_ + active_dim_};
    for (int i = 0; i < 3; ++i) {
        outfile->Printf("\n    Diagonalizing block " + std::to_string(i) + " ...");

        auto dim = mos_dim[i];
        auto offset_dim = mos_offsets[i];

        auto Fsub = std::make_shared<psi::Matrix>("Fsub_" + std::to_string(i), dim, dim);

        for (size_t h = 0; h < nirrep_; ++h) {
            for (int p = 0; p < dim[h]; ++p) {
                size_t np = p + offset_dim[h];
                for (int q = 0; q < dim[h]; ++q) {
                    size_t nq = q + offset_dim[h];
                    Fsub->set(h, p, q, F->get(h, np, nq));
                }
            }
        }

        // test off-diagonal elements to decide if need to diagonalize this block
        auto Fsub_od = Fsub->clone();
        Fsub_od->zero_diagonal();

        double Fsub_max = Fsub_od->absmax();
        double Fsub_norm = std::sqrt(Fsub_od->sum_of_squares());

        double threshold_max = 0.1 * options_->get_double("CASSCF_G_CONVERGENCE");
        if (ints_->integral_type() == Cholesky) {
            double cd_tlr = options_->get_double("CHOLESKY_TOLERANCE");
            threshold_max = (threshold_max < 0.5 * cd_tlr) ? 0.5 * cd_tlr : threshold_max;
        }
        double threshold_rms = std::sqrt(dim.sum() * (dim.sum() - 1) / 2.0) * threshold_max;

        // diagonalize
        if (Fsub_max > threshold_max or Fsub_norm > threshold_rms) {
            auto Usub = std::make_shared<psi::Matrix>("Usub_" + std::to_string(i), dim, dim);
            auto Esub = std::make_shared<psi::Vector>("Esub_" + std::to_string(i), dim);
            Fsub->diagonalize(Usub, Esub);

            // fill in data
            for (size_t h = 0; h < nirrep_; ++h) {
                for (int p = 0; p < dim[h]; ++p) {
                    size_t np = p + offset_dim[h];
                    for (int q = 0; q < dim[h]; ++q) {
                        size_t nq = q + offset_dim[h];
                        U->set(h, np, nq, Usub->get(h, p, q));
                    }
                }
            }
        } // end if need to diagonalize

        outfile->Printf(" Done.");
    } // end sub blocl

    return U;
}

std::shared_ptr<psi::Matrix> CASSCF_HESS::build_fock(std::shared_ptr<psi::Matrix> Ca) {
    auto F_i = build_fock_inactive(Ca);
    auto F_a = build_fock_active(Ca);

    auto Fock = F_i->clone();
    Fock->add(F_a);
    Fock->set_name("Fock");
    return Fock;
}

std::shared_ptr<psi::Matrix> CASSCF_HESS::build_fock_inactive(std::shared_ptr<psi::Matrix> Ca) {
    // Implementation Notes (in AO basis)
    // F_frozen = D_{uv}^{frozen} * (2 * (uv|rs) - (us|rv))
    // F_restricted = D_{uv}^{restricted} * (2 * (uv|rs) - (us|rv))
    // F_inactive = Hcore + F_frozen + F_restricted
    // D_{uv}^{frozen} = \sum_{i}^{frozen} C_{ui} * C_{vi}
    // D_{uv}^{restricted} = \sum_{i}^{restricted} C_{ui} * C_{vi}

    // grab part of Ca for inactive docc
    auto Cdocc = std::make_shared<psi::Matrix>("C_INACTIVE", nirrep_, nsopi_, inactive_docc_dim_);
    for (size_t h = 0; h < nirrep_; h++) {
        for (int i = 0; i < inactive_docc_dim_[h]; i++) {
            Cdocc->set_column(h, i, Ca->get_column(h, i));
        }
    }

    // JK build
    std::vector<std::shared_ptr<psi::Matrix>>& Cl = JK_->C_left();
    std::vector<std::shared_ptr<psi::Matrix>>& Cr = JK_->C_right();

    JK_->set_do_K(true);
    Cl.clear();
    Cr.clear();
    Cl.push_back(Cdocc); // Cr is the same as Cl
    JK_->compute();

    std::shared_ptr<psi::Matrix> Fdocc = (JK_->J()[0])->clone();
    Fdocc->scale(2.0);
    Fdocc->subtract(JK_->K()[0]);

    Fdocc->add(Hcore_);
    Fdocc->transform(Ca);
    Fdocc->set_name("Fock_inactive");

    if (debug_print_) {
        Fdocc->print();
    }

    return Fdocc;
}

std::shared_ptr<psi::Matrix> CASSCF_HESS::build_fock_active(std::shared_ptr<psi::Matrix> Ca) {
    // Implementation Notes (in AO basis)
    // F_active = D_{uv}^{active} * ( (uv|rs) - 0.5 * (us|rv) )
    // D_{uv}^{active} = \sum_{xy}^{active} C_{ux} * C_{vy} * Gamma1_{xy}

    // grab part of Ca for active docc
    auto Cactv = std::make_shared<psi::Matrix>("C_ACTIVE", nirrep_, nsopi_, active_dim_);
    for (size_t h = 0; h < nirrep_; h++) {
        for (int i = 0; i < active_dim_[h]; i++) {
            int offset = frozen_docc_dim_[h] + restricted_docc_dim_[h];
            Cactv->set_column(h, i, Ca->get_column(h, i + offset));
        }
    }

    // put one-density to SharedMatrix form
    auto Gamma1 = std::make_shared<psi::Matrix>("Gamma1", active_dim_, active_dim_);
    const auto& gamma1_data = gamma1_.data();

    for (size_t h = 0, offset = 0; h < nirrep_; ++h) {
        for (int u = 0; u < active_dim_[h]; ++u) {
            size_t nu = u + offset;
            for (int v = 0; v < active_dim_[h]; ++v) {
                size_t nv = v + offset;
                Gamma1->set(h, u, v, gamma1_data[nu * nactv_ + nv]);
            }
        }
        offset += active_dim_[h];
    }

    // dress Cactv by one-density, which will the C_right for JK
    auto Cactv_dressed = linalg::doublet(Cactv, Gamma1, false, false);

    // JK build
    std::vector<std::shared_ptr<psi::Matrix>>& Cl = JK_->C_left();
    std::vector<std::shared_ptr<psi::Matrix>>& Cr = JK_->C_right();

    JK_->set_do_K(true);
    Cl.clear();
    Cr.clear();
    Cl.push_back(Cactv);
    Cr.push_back(Cactv_dressed);
    JK_->compute();

    std::shared_ptr<psi::Matrix> Factv = (JK_->K()[0])->clone();
    Factv->scale(-0.5);
    Factv->add(JK_->J()[0]);

    // transform to MO
    Factv->transform(Ca);
    Factv->set_name("Fock_active");

    if (debug_print_) {
        Factv->print();
    }

    return Factv;
}

void CASSCF_HESS::overlap_orbitals(const psi::SharedMatrix& C_old, const psi::SharedMatrix& C_new) {
    psi::SharedMatrix S_orbitals(
        new psi::Matrix("Overlap", scf_info_->nsopi(), scf_info_->nsopi()));
    psi::SharedMatrix S_basis = ints_->wfn()->S();
    S_orbitals = psi::linalg::triplet(C_old, S_basis, C_new, true, false, false);
    S_orbitals->set_name("C^T S C (Overlap)");
    for (size_t h = 0; h < nirrep_; h++) {
        for (int i = 0; i < S_basis->rowspi(h); i++) {
            if (std::fabs(S_basis->get(h, i, i) - 1.0000000) > 1e-6) {
                //    S_basis->get_row(h, i)->print();
            }
        }
    }
}

void CASSCF_HESS::write_orbitals_molden() {
    psi::SharedVector occ_vector(new psi::Vector(nirrep_, corr_dim_));
    view_modified_orbitals(ints_->wfn(), ints_->Ca(), scf_info_->epsilon_a(), occ_vector);
}

// void CASSCF_HESS::overlap_coefficients() {
//    outfile->Printf("\n iter  Overlap_{i-1} Overlap_{i}");
//    for (size_t iter = 1; iter < CISolutions_.size(); ++iter) {
//        for (size_t cisoln = 0; cisoln < CISolutions_[iter].size(); cisoln++) {
//            for (size_t j = 0; j < CISolutions_[iter].size(); j++) {
//                if (std::fabs(CISolutions_[0][cisoln]->dot(CISolutions_[iter][j])) > 0.90) {
//                    outfile->Printf("\n %d:%d %d:%d %8.8f", 0, cisoln, iter, j,
//                                    CISolutions_[0][cisoln]->dot(CISolutions_[iter][j]));
//                }
//            }
//        }
//        outfile->Printf("\n");
//    }
//}

// std::vector<RDMs> CASSCF_HESS::rdms(const std::vector<std::pair<size_t, size_t>>& /*root_list*/,
//                               int /*max_rdm_level*/) {
//    // TODO (York): this does not seem the correct thing to do.
//    std::vector<RDMs> refs;
//    refs.push_back(cas_ref_);
//    return refs;
//}

// std::vector<RDMs>
// CASSCF_HESS::transition_rdms(const std::vector<std::pair<size_t, size_t>>& /*root_list*/,
//                        std::shared_ptr<ActiveSpaceMethod> /*method2*/, int /*max_rdm_level*/) {
//    std::vector<RDMs> refs;
//    throw std::runtime_error("FCISolver::transition_rdms is not implemented!");
//    return refs;
//}

} // namespace forte
