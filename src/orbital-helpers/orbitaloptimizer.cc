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

#include "ambit/blocked_tensor.h"


#include "psi4/lib3index/cholesky.h"
#include "psi4/libfock/jk.h"
#include "psi4/libmints/matrix.h"
#include "psi4/psifiles.h"
#include "psi4/libpsi4util/PsiOutStream.h"

#include "base_classes/forte_options.h"
#include "fci/fci_solver.h"
#include "base_classes/mo_space_info.h"
#include "helpers/helpers.h"
#include "integrals/integrals.h"
#include "helpers/timer.h"
#include "orbitaloptimizer.h"

#include "base_classes/rdms.h"
#include "base_classes/scf_info.h"

using namespace psi;

namespace forte {

OrbitalOptimizer::OrbitalOptimizer() {}

OrbitalOptimizer::OrbitalOptimizer(ambit::Tensor Gamma1, ambit::Tensor Gamma2,
                                   ambit::Tensor two_body_ab, std::shared_ptr<ForteOptions> options,
                                   std::shared_ptr<MOSpaceInfo> mo_space_info)
    : gamma1_(Gamma1), gamma2_(Gamma2), integral_(two_body_ab), mo_space_info_(mo_space_info),
      options_(options) {}
void OrbitalOptimizer::update() {
    startup();
    fill_shared_density_matrices();
    /// F^{I}_{pq} = h_{pq} + 2 (pq | kk) - (pk |qk)
    /// This is done using JK builder: F^{I}_{pq} = h_{pq} + C^{T}[2J - K]C
    /// Built in the AO basis for efficiency
    local_timer overall_update;

    local_timer fock_core;
    form_fock_intermediates();
    if (timings_) {
        outfile->Printf("\n\n FormFockIntermediates took %8.8f s.", fock_core.get());
    }

    local_timer orbital_grad;
    orbital_gradient();
    if (timings_) {
        outfile->Printf("\n\n FormOrbitalGradient took %8.8f s.", orbital_grad.get());
    }

    local_timer diag_hess;
    diagonal_hessian();
    if (timings_) {
        outfile->Printf("\n\n FormDiagHessian took %8.8f s.", diag_hess.get());
    }

    if (timings_) {
        outfile->Printf("\n\n Update function takes %8.8f s", overall_update.get());
    }
}

void OrbitalOptimizer::startup() {
    frozen_docc_dim_ = mo_space_info_->dimension("FROZEN_DOCC");
    restricted_docc_dim_ = mo_space_info_->dimension("RESTRICTED_DOCC");
    active_dim_ = mo_space_info_->dimension("ACTIVE");
    restricted_uocc_dim_ = mo_space_info_->dimension("RESTRICTED_UOCC");
    inactive_docc_dim_ = mo_space_info_->dimension("INACTIVE_DOCC");
    nmopi_ = mo_space_info_->dimension("CORRELATED");

    frozen_docc_abs_ = mo_space_info_->corr_absolute_mo("FROZEN_DOCC");
    restricted_docc_abs_ = mo_space_info_->corr_absolute_mo("RESTRICTED_DOCC");
    active_abs_ = mo_space_info_->corr_absolute_mo("ACTIVE");
    restricted_uocc_abs_ = mo_space_info_->corr_absolute_mo("RESTRICTED_UOCC");
    inactive_docc_abs_ = mo_space_info_->corr_absolute_mo("INACTIVE_DOCC");
    nmo_abs_ = mo_space_info_->corr_absolute_mo("CORRELATED");
    if (frozen_docc_abs_.size() && !(options_->get_bool("OPTIMIZE_FROZEN_CORE"))) {
        casscf_freeze_core_ = true;
    } else {
        casscf_freeze_core_ = false;
    }
    if (options_->get_bool("OPTIMIZE_FROZEN_CORE")) {
        throw psi::PSIEXCEPTION("CASSCF can not handle optimization of frozen core, yet.");
    }

    nmo_ = mo_space_info_->size("CORRELATED");
    all_nmo_ = mo_space_info_->size("ALL");

    nrdocc_ = restricted_docc_abs_.size();
    nfrozen_ = frozen_docc_abs_.size();
    na_ = active_abs_.size();
    nvir_ = restricted_uocc_abs_.size();
    casscf_debug_print_ = options_->get_bool("CASSCF_DEBUG_PRINTING");
    nirrep_ = mo_space_info_->nirrep();
    nsopi_ = scf_info_->nsopi();

    if (options_->get_str("CASSCF_CI_SOLVER") == "FCI") {
        cas_ = true;
    } else if (options_->get_str("CASSCF_CI_SOLVER") == "CAS") {
        if (options_->get_str("FCIMO_ACTV_TYPE") != "COMPLETE") {
            cas_ = false;
        } else {
            cas_ = true;
        }
    } else if (options_->get_str("CASSCF_CI_SOLVER") == "ACI") {
        if (options_->get_double("SIGMA") == 0.0) {
            cas_ = true;
        } else {
            cas_ = true;
        }
    } else if (options_->get_str("CASSCF_CI_SOLVER") == "DMRG") {
        cas_ = true;
    } else {
        outfile->Printf("\n\n Please set your CASSCF_CI_SOLVER to either FCI, CAS, ACI, or DMRG");
        outfile->Printf("\n\n You set your CASSCF_CI_SOLVER to %s.",
                        options_->get_str("CASSCF_CI_SOLVER").c_str());
        throw psi::PSIEXCEPTION("You did not specify your CASSCF_CI_SOLVER correctly.");
    }
    cas_ = true;
}
void OrbitalOptimizer::orbital_gradient() {
    // std::vector<size_t> nmo_array =
    // mo_space_info_->corr_absolute_mo("CORRELATED");
    /// From Y_{pt} = F_{pu}^{core} * Gamma_{tu}
    ambit::Tensor Y = ambit::Tensor::build(ambit::CoreTensor, "Y", {nmo_, na_});
    ambit::Tensor F_pu = ambit::Tensor::build(ambit::CoreTensor, "F_pu", {nmo_, na_});
    if (nrdocc_ > 0 or nfrozen_ > 0) {
        F_pu.iterate([&](const std::vector<size_t>& i, double& value) {
            value = F_core_->get(nmo_abs_[i[0]], active_abs_[i[1]]);
        });
    }
    Y("p,t") = F_pu("p,u") * gamma1_("t, u");

    psi::SharedMatrix Y_m(new psi::Matrix("Y_m", nmo_, na_));

    Y.iterate([&](const std::vector<size_t>& i, double& value) {
        Y_m->set(nmo_abs_[i[0]], i[1], value);
    });
    Y_ = Y_m;
    Y_->set_name("F * gamma");
    if (casscf_debug_print_) {
        Y_->print();
    }

    // Form Z (pu | v w) * Gamma2(tuvw)
    // One thing I am not sure about for Gamma2->how to get spin free RDM from
    // spin based RDM
    // gamma1 = gamma1a + gamma1b;
    // gamma2 = gamma2aa + gamma2ab + gamma2ba + gamma2bb
    /// lambda2 = gamma1*gamma1

    // std::vector<size_t> na_array = mo_space_info_->corr_absolute_mo("ACTIVE");
    /// SInce the integrals class assumes that the indices are relative,
    /// pass the relative indices to the integrals code.
    ambit::Tensor Z = ambit::Tensor::build(ambit::CoreTensor, "Z", {nmo_, na_});

    //(pu | x y) -> <px | uy> * gamma2_{"t, u, x, y"
    Z("p, t") = integral_("p,u,x,y") * gamma2_("t, u, x, y");
    if (casscf_debug_print_) {
        outfile->Printf("\n\n integral_: %8.8f  gamma2_: %8.8f", integral_.norm(2),
                        gamma2_.norm(2));
    }

    psi::SharedMatrix Zm(new psi::Matrix("Zm", nmo_, na_));
    Z.iterate([&](const std::vector<size_t>& i, double& value) { Zm->set(i[0], i[1], value); });

    Z_ = Zm;
    Z_->set_name("g * rdm2");
    if (casscf_debug_print_) {
        Z_->print();
    }
    // g_ia = 4F_core + 2F_act
    // g_ta = 2Y + 4Z
    // g_it = 4F_core + 2 F_act - 2Y - 4Z;

    // GOTCHA:  Z and T are of size nmo by na
    // The absolute MO should not be used to access elements of Z, Y, or Gamma
    // since these are of 0....na_ arrays
    // auto occ_array = mo_space_info_->corr_absolute_mo("RESTRICTED_DOCC");
    // auto virt_array = mo_space_info_->corr_absolute_mo("RESTRICTED_UOCC");
    // auto active_array = mo_space_info_->corr_absolute_mo("ACTIVE");

    size_t nhole = nrdocc_ + na_;
    size_t npart = na_ + nvir_;
    psi::SharedMatrix Orb_grad(new psi::Matrix("G_pq", nhole, npart));
    Orb_grad->set_name("CASSCF Gradient");

    auto generalized_hole_abs = mo_space_info_->corr_absolute_mo("GENERALIZED HOLE");
    auto generalized_part_abs = mo_space_info_->corr_absolute_mo("GENERALIZED PARTICLE");
    psi::Dimension general_hole_dim = mo_space_info_->dimension("GENERALIZED HOLE");
    psi::Dimension general_part_dim = mo_space_info_->dimension("GENERALIZED PARTICLE");
    auto generalized_hole_rel = mo_space_info_->get_relative_mo("GENERALIZED HOLE");
    auto generalized_part_rel = mo_space_info_->get_relative_mo("GENERALIZED PARTICLE");
    if (casscf_debug_print_) {
        outfile->Printf("Generalized_hole_abs\n");
        for (auto gha : generalized_hole_abs) {
            outfile->Printf(" %d", gha);
        }
        outfile->Printf("\n Generalized_part_abs\n");
        for (auto gpa : generalized_part_abs) {
            outfile->Printf(" %d", gpa);
        }
        outfile->Printf("Generalized_hole_rel\n");
        for (auto ghr : generalized_hole_rel) {
            outfile->Printf(" (%d, %d) ", ghr.first, ghr.second);
        }
        outfile->Printf("Generalized_part_rel\n");
        for (auto gpr : generalized_part_rel) {
            outfile->Printf(" (%d, %d) ", gpr.first, gpr.second);
        }
    }
    int offset_hole = 0;
    int offset_part = 0;
    std::vector<size_t> hole_offset_vector(nirrep_);
    std::vector<size_t> particle_offset_vector(nirrep_);
    for (size_t h = 0; h < nirrep_; h++) {
        hole_offset_vector[h] = offset_hole;
        particle_offset_vector[h] = offset_part;
        offset_hole += general_hole_dim[h];
        offset_part += general_part_dim[h];
    }

    /// Create a map that takes absolute index (in nmo indexing) and returns a
    /// value that corresponds to either hole or particle
    /// IE: 0, 1, 2, 9 for hole where 2 and 9 are active
    /// This map returns 0, 1, 2, 3
    for (size_t i = 0; i < nhole; i++) {
        int sym_irrep = generalized_hole_rel[i].first;
        // nhole_map_[generalized_hole_abs[i]] = (generalized_hole_rel[i].second
        // + restricted_docc_dim_[generalized_hole_rel[i].first]);
        /// generalized_hole_rel[i].second tells where the entry is in with
        /// respect to the irrep
        /// hole_offset is an offset that accounts for the previous irreps
        nhole_map_[generalized_hole_abs[i]] = generalized_hole_rel[i].second +
                                              hole_offset_vector[sym_irrep] -
                                              frozen_docc_dim_[sym_irrep];
    }
    for (size_t a = 0; a < npart; a++) {
        int sym_irrep = generalized_part_rel[a].first;
        /// generalized_part_rel[a].second tells where the entry is in with
        /// respect to the irrep
        /// part_offset_vector is an offset that accounts for the previous
        /// irreps
        npart_map_[generalized_part_abs[a]] =
            (generalized_part_rel[a].second + particle_offset_vector[sym_irrep] -
             restricted_docc_dim_[sym_irrep]) -
            frozen_docc_dim_[sym_irrep];
    }

    if (casscf_debug_print_) {
        for (auto hole : nhole_map_) {
            outfile->Printf("\n nhole_map[%d] = %d", hole.first, hole.second);
        }
        for (auto part : npart_map_) {
            outfile->Printf("\n npart_map[%d] = %d", part.first, part.second);
        }
    }

    /// Some wierdness going on
    /// Since G is nhole by npart
    /// I need to make sure that the matrix is ordered in pitzer ordering

    /// The offset allows me to place the correct values with pitzer ordering

    /// The Fock matrix is computed as stated in Pg. 622 of Helgaker book
    psi::SharedMatrix Fock(new psi::Matrix("Generalized Fock", nmo_, nmo_));
    for (size_t q = 0; q < nmo_; q++) {
        int q_o = nmo_abs_[q];
        for (size_t i = 0; i < nrdocc_; i++) {
            int i_iq = restricted_docc_abs_[i];
            double value = 2.0 * (F_core_->get(q_o, i_iq) + F_act_->get(q_o, i_iq));
            Fock->set(i_iq, q_o, value);
        }
        for (size_t v = 0; v < na_; v++) {
            int act_o = active_abs_[v];
            double value_vn = Y_->get(q_o, v) + Z_->get(q_o, v);
            Fock->set(act_o, q_o, value_vn);
        }
        for (size_t a = 0; a < nvir_; a++) {
            int a_o = restricted_uocc_abs_[a];
            Fock->set(a_o, q_o, 0.0);
        }
    }
    if (casscf_debug_print_) {
        Fock->print();
    }
    psi::SharedMatrix Orb_grad_Fock(new psi::Matrix("G_pq", nhole, npart));
    for (size_t h = 0; h < nhole; h++) {
        for (size_t p = 0; p < npart; p++) {
            size_t h_act = generalized_hole_abs[h];
            size_t p_act = generalized_part_abs[p];
            double fock_value = Fock->get(h_act, p_act) - Fock->get(p_act, h_act);
            Orb_grad_Fock->set(nhole_map_[h_act], npart_map_[p_act], 2.0 * fock_value);
        }
    }
    for (size_t ui = 0; ui < na_; ui++) {
        for (size_t v = 0; v < na_; v++) {
            size_t u = active_abs_[ui];
            size_t vo = active_abs_[v];
            if (vo == u) {
                Orb_grad_Fock->set(nhole_map_[vo], npart_map_[u], 0.0);
            }
        }
    }
    if (cas_) {
        zero_redunant(Orb_grad_Fock);
    }

    if (casscf_debug_print_) {
        Orb_grad_Fock->print();
    }

    g_ = Orb_grad_Fock;
    g_->set_name("CASSCF_GRADIENT");
}
void OrbitalOptimizer::diagonal_hessian() {
    size_t nhole = nrdocc_ + na_;
    size_t npart = na_ + nvir_;
    psi::SharedMatrix D(new psi::Matrix("D_pq", nhole, npart));
    D->set_name("Diagonal Hessian");

    for (size_t ii = 0; ii < nrdocc_; ii++) {
        for (size_t ai = 0; ai < nvir_; ai++) {
            size_t a = restricted_uocc_abs_[ai];
            size_t i = restricted_docc_abs_[ii];
            // double value_ia = F_core_->get(a,a) * 4.0 + 2 * F_act_->get(a,a);
            // value_ia -= 4.0 * F_core_->get(i,i)  - 2 * F_act_->get(i,i);
            double value_ia = (F_core_->get(a, a) * 4.0 + 4.0 * F_act_->get(a, a));
            value_ia -= (4.0 * F_core_->get(i, i) + 4.0 * F_act_->get(i, i));
            D->set(nhole_map_[i], npart_map_[a], value_ia);
        }
    }
    for (size_t ai = 0; ai < nvir_; ai++) {
        for (size_t ti = 0; ti < na_; ti++) {
            size_t a = restricted_uocc_abs_[ai];
            size_t t = active_abs_[ti];
            // double value_ta = 2.0 * gamma1M_->get(ti,ti) * F_core_->get(a,a);
            // value_ta += gamma1M_->get(ti,ti) * F_act_->get(a,a);
            // value_ta -= 2*Y_->get(t,ti) + 4.0 *Z_->get(t,ti);
            double value_ta = 2.0 * gamma1M_->get(ti, ti) * F_core_->get(a, a);
            value_ta += 2.0 * gamma1M_->get(ti, ti) * F_act_->get(a, a);
            value_ta -= (2.0 * Y_->get(t, ti) + 2.0 * Z_->get(t, ti));
            D->set(nhole_map_[t], npart_map_[a], value_ta);
        }
    }
    for (size_t ii = 0; ii < nrdocc_; ii++) {
        for (size_t ti = 0; ti < na_; ti++) {
            size_t i = restricted_docc_abs_[ii];
            size_t t = active_abs_[ti];
            double value_it = 4.0 * F_core_->get(t, t);
            value_it += 4.0 * F_act_->get(t, t);
            value_it += 2.0 * gamma1M_->get(ti, ti) * F_core_->get(i, i);
            value_it += 2.0 * gamma1M_->get(ti, ti) * F_act_->get(i, i);
            value_it -= (4.0 * F_core_->get(i, i) + 4.0 * F_act_->get(i, i));
            value_it -= (2.0 * Y_->get(t, ti) + 2.0 * Z_->get(t, ti));
            D->set(nhole_map_[i], npart_map_[t], value_it);
        }
    }

    for (size_t u = 0; u < na_; u++) {
        for (size_t v = 0; v < na_; v++) {
            size_t uo = active_abs_[u];
            size_t vo = active_abs_[v];
            D->set(nhole_map_[uo], npart_map_[vo], 1.0);
        }
    }
    d_ = D;
    if (casscf_debug_print_) {
        d_->print();
    }
}
psi::SharedMatrix OrbitalOptimizer::approx_solve() {
    psi::Dimension nhole_dim = restricted_docc_dim_ + active_dim_;
    psi::Dimension nvirt_dim = restricted_uocc_dim_ + active_dim_;

    psi::SharedMatrix G_grad(new psi::Matrix("GradientSym", nhole_dim, nvirt_dim));
    psi::SharedMatrix D_grad(new psi::Matrix("HessianSym", nhole_dim, nvirt_dim));

    int offset_hole = 0;
    int offset_part = 0;
    for (size_t h = 0; h < nirrep_; h++) {
        for (int i = 0; i < nhole_dim[h]; i++) {
            int ioff = i + offset_hole;
            for (int a = 0; a < nvirt_dim[h]; a++) {
                int aoff = a + offset_part;
                G_grad->set(h, i, a, g_->get(ioff, aoff));
                D_grad->set(h, i, a, d_->get(ioff, aoff));
            }
        }
        offset_hole += nhole_dim[h];
        offset_part += nvirt_dim[h];
    }
    psi::SharedMatrix S_tmp = G_grad->clone();
    // S_tmp->apply_denominator(D_grad);
    for (size_t h = 0; h < nirrep_; h++) {
        for (int p = 0; p < S_tmp->rowspi(h); p++) {
            for (int q = 0; q < S_tmp->colspi(h); q++) {
                // if(std::fabs(D_grad->get(h, p, q)) > 1e-12)
                //{
                S_tmp->set(h, p, q, G_grad->get(h, p, q) / D_grad->get(h, p, q));
                //}
                // else{
                //    S_tmp->set(h, p, q, 0.0);
                //    outfile->Printf("\n Warning: D_grad(%d, %d, %d) is NAN",
                //    h, p, q);
                //}
            }
        }
    }
    // psi::SharedMatrix S_tmp_AH = AugmentedHessianSolve();
    for (size_t h = 0; h < nirrep_; h++) {
        for (int u = 0; u < active_dim_[h]; u++) {
            for (int v = 0; v < active_dim_[h]; v++) {
                S_tmp->set(h, restricted_docc_dim_[h] + u, v, 0.0);
            }
        }
    }

    if (casscf_debug_print_) {
        G_grad->print();
        D_grad->print();
        S_tmp->set_name("g / d");
        S_tmp->print();
    }
    return S_tmp;
}
psi::SharedMatrix OrbitalOptimizer::AugmentedHessianSolve() {
    size_t nhole = mo_space_info_->size("GENERALIZED HOLE");
    size_t npart = mo_space_info_->size("GENERALIZED PARTICLE");

    psi::SharedMatrix AugmentedHessian(
        new psi::Matrix("Augmented Hessian", nhole + npart + 1, nhole + npart + 1));
    for (size_t hol = 0; hol < nhole; hol++) {
        for (size_t part = 0; part < npart; part++) {
            AugmentedHessian->set(hol, part, d_->get(hol, part));
        }
    }
    psi::SharedMatrix g_transpose = g_->transpose();
    for (size_t hol = 0; hol < nhole; hol++) {
        for (size_t part = 0; part < npart; part++) {
            AugmentedHessian->set(part, hol + npart, g_transpose->get(part, hol));
        }
    }
    for (size_t hol = 0; hol < nhole; hol++) {
        for (size_t part = 0; part < npart; part++) {
            AugmentedHessian->set(part + nhole, hol, g_->get(hol, part));
        }
    }
    // for(int hol = 0; hol < nhole; hol++){
    //    for(int part = nhole; part < (nhole + npart + 1); part++){
    //        AugmentedHessian->set(hol, part, g_->transpose()->get(npart - part
    //        , hol));
    //    }
    //}
    // AugmentedHessian->print();
    // C_DCOPY(nhole * npart, g_->pointer()[0], 1,
    // &(AugmentedHessian->pointer()[0][nhole * npart]), 1);
    // C_DCOPY(nhole * npart, g_->transpose()->pointer()[0], 1,
    // &(AugmentedHessian->pointer()[nhole * npart][0]), 1);

    // AugmentedHessian->set(nhole * npart, nhole * npart, 0.0);
    // AugmentedHessian->print();
    psi::SharedMatrix HessianEvec(
        new psi::Matrix("HessianEvec", nhole + npart + 1, nhole + npart + 1));
    psi::SharedVector HessianEval(new Vector("HessianEval", nhole + npart + 1));
    AugmentedHessian->diagonalize(HessianEvec, HessianEval);
    HessianEvec->print();
    // psi::SharedMatrix S_AH(new psi::Matrix("AugmentedHessianLowestEigenvalue", nhole,
    // npart));
    // if(casscf_debug_print_)
    //{
    //    AugmentedHessian->print();
    //    HessianEval->print();
    //}
    return HessianEvec;
}

psi::SharedMatrix OrbitalOptimizer::rotate_orbitals(psi::SharedMatrix C, psi::SharedMatrix S) {
    psi::Dimension nhole_dim = mo_space_info_->dimension("GENERALIZED HOLE");
    psi::Dimension nvirt_dim = mo_space_info_->dimension("GENERALIZED PARTICLE");
    /// Clone the C matrix
    psi::SharedMatrix C_rot(C->clone());
    psi::SharedMatrix S_mat(S->clone());
    psi::SharedMatrix S_sym(new psi::Matrix("Exp(K)", mo_space_info_->nirrep(),
                                            mo_space_info_->dimension("ALL"),
                                            mo_space_info_->dimension("ALL")));
    int offset_hole = 0;
    int offset_part = 0;
    for (size_t h = 0; h < nirrep_; h++) {
        for (int i = 0; i < frozen_docc_dim_[h]; i++) {
            S_sym->set(h, i, i, 1.0);
        }
        for (int i = 0; i < nhole_dim[h]; i++) {
            for (int a = std::max(restricted_docc_dim_[h], i); a < nmopi_[h]; a++) {
                int ioff = i + frozen_docc_dim_[h];
                int aoff = a + frozen_docc_dim_[h];
                S_sym->set(h, ioff, aoff, S_mat->get(h, i, a - restricted_docc_dim_[h]));
                S_sym->set(h, aoff, ioff, -1.0 * S_mat->get(h, i, a - restricted_docc_dim_[h]));
            }
        }
        offset_hole += nhole_dim[h];
        offset_part += nvirt_dim[h];
    }
    psi::SharedMatrix S_exp = matrix_exp(S_sym);
    for (size_t h = 0; h < nirrep_; h++) {
        for (int f = 0; f < frozen_docc_dim_[h]; f++) {
            S_exp->set(h, f, f, 1.0);
        }
    }

    C_rot = psi::linalg::doublet(C, S_exp);
    C_rot->set_name("ROTATED_ORBITAL");
    S_sym->set_name("Orbital Rotation (S = exp(x))");
    if (casscf_debug_print_) {
        C_rot->print();
        S_exp->print();
    }
    return C_rot;
}

void OrbitalOptimizer::fill_shared_density_matrices() {
    psi::SharedMatrix gamma_spin_free(new psi::Matrix("Gamma", na_, na_));
    gamma1_.iterate([&](const std::vector<size_t>& i, double& value) {
        gamma_spin_free->set(i[0], i[1], value);
    });
    gamma1M_ = gamma_spin_free;

    psi::SharedMatrix gamma2_matrix(new psi::Matrix("Gamma2", na_ * na_, na_ * na_));
    gamma2_.iterate([&](const std::vector<size_t>& i, double& value) {
        gamma2_matrix->set(i[0] * na_ + i[1], i[2] * na_ + i[3], value);
    });
    gamma2M_ = gamma2_matrix;
    if (casscf_debug_print_) {
        gamma1M_->set_name("Spin-free 1 RDM");
        gamma1M_->print();
        gamma2M_->set_name("Spin-free 2 RDM");
        gamma2M_->print();
    }
}
std::shared_ptr<psi::Matrix> OrbitalOptimizer::make_c_sym_aware(psi::SharedMatrix aotoso) {
    /// Step 1: Obtain guess MO coefficients C_{mup}
    /// Since I want to use these in a symmetry aware basis,
    /// I will move the C matrix into a Pfitzer ordering

    psi::Dimension nmopi = mo_space_info_->dimension("ALL");

    /// I want a C matrix in the C1 basis but symmetry aware
    size_t nso = scf_info_->nso();
    nirrep_ = mo_space_info_->nirrep();
    psi::SharedMatrix Call(new psi::Matrix(nso, nmopi.sum()));

    // Transform from the SO to the AO basis for the C matrix.
    // just transfroms the C_{mu_ao i} -> C_{mu_so i}
    for (size_t h = 0, index = 0; h < nirrep_; ++h) {
        for (int i = 0; i < nmopi[h]; ++i) {
            size_t nao = nso;
            size_t nso = nsopi_[h];

            if (!nso)
                continue;

            C_DGEMV('N', nao, nso, 1.0, aotoso->pointer(h)[0], nso, &Ca_sym_->pointer(h)[0][i],
                    nmopi[h], 0.0, &Call->pointer()[0][index], nmopi.sum());

            index += 1;
        }
    }

    return Call;
}
psi::SharedMatrix OrbitalOptimizer::matrix_exp(const psi::SharedMatrix& unitary) {
    psi::SharedMatrix U(unitary->clone());
    if (false) {
        U->expm();
    } else {
        // Build exp(U) = 1 + U + 1/2 U U + 1/6 U U U

        for (size_t h = 0; h < nirrep_; h++) {
            if (!U->rowspi()[h])
                continue;
            double** Up = U->pointer(h);
            for (int i = 0; i < (U->colspi()[h]); i++) {
                Up[i][i] += 1.0;
            }
        }
        U->gemm(false, false, 0.5, unitary, unitary, 1.0);

        psi::SharedMatrix tmp_third = psi::linalg::triplet(unitary, unitary, unitary);
        tmp_third->scale(1.0 / 6.0);
        U->add(tmp_third);
        tmp_third.reset();

        // We did not fully exponentiate the matrix, need to orthogonalize
        U->schmidt();
    }
    return U;
}
void OrbitalOptimizer::zero_redunant(psi::SharedMatrix& matrix) {
    for (size_t u = 0; u < na_; u++) {
        for (size_t v = 0; v < na_; v++) {
            size_t uo = active_abs_[u];
            size_t vo = active_abs_[v];
            matrix->set(nhole_map_[uo], npart_map_[vo], 0.0);
        }
    }
}
CASSCFOrbitalOptimizer::CASSCFOrbitalOptimizer(ambit::Tensor Gamma1, ambit::Tensor Gamma2,
                                               ambit::Tensor two_body_ab,
                                               std::shared_ptr<ForteOptions> options,
                                               std::shared_ptr<MOSpaceInfo> mo_space_info)
    : OrbitalOptimizer(Gamma1, Gamma2, two_body_ab, options, mo_space_info) {}
CASSCFOrbitalOptimizer::~CASSCFOrbitalOptimizer() {}

void CASSCFOrbitalOptimizer::form_fock_intermediates() {
    /// Get the CoreHamiltonian in AO basis

    if (Ca_sym_ == nullptr) {
        outfile->Printf("\n\n Please give your OrbitalOptimize an Orbital");
        throw psi::PSIEXCEPTION("Please set CMatrix before you call orbital rotation casscf");
    }
    if (H_ == nullptr) {
        outfile->Printf("\n\n Please set the OneBody operator");
        throw psi::PSIEXCEPTION("Please set H before you call orbital rotation casscf");
    }
    H_->transform(Ca_sym_);
    if (casscf_debug_print_) {
        H_->set_name("CORR_HAMIL");
        H_->print();
    }

    /// Creating a C_core and C_active matrices
    psi::SharedMatrix C_active(new psi::Matrix("C_active", nirrep_, nsopi_, active_dim_));
    psi::SharedMatrix C_core(new psi::Matrix("C_core", nirrep_, nsopi_, restricted_docc_dim_));

    // Need to get the inactive block of the C matrix
    psi::SharedMatrix F_core_c1(new psi::Matrix("F_core_no_sym", nmo_, nmo_));
    psi::SharedMatrix F_core(new psi::Matrix("InactiveTemp1", nirrep_, nsopi_, nsopi_));
    F_core_c1->zero();

    for (size_t h = 0; h < nirrep_; h++) {
        for (int i = 0; i < active_dim_[h]; i++) {
            C_active->set_column(
                h, i, Ca_sym_->get_column(h, i + frozen_docc_dim_[h] + restricted_docc_dim_[h]));
        }
    }
    psi::SharedMatrix C_active_ao(new psi::Matrix("C_active", nirrep_, nsopi_, nsopi_));
    psi::SharedMatrix gamma1_sym(new psi::Matrix("gamma1_sym", nirrep_, active_dim_, active_dim_));
    size_t offset_active = 0;
    for (size_t h = 0; h < nirrep_; h++) {
        for (int u = 0; u < active_dim_[h]; u++) {
            size_t uoff = u + offset_active;
            for (int v = 0; v < active_dim_[h]; v++) {
                size_t voff = v + offset_active;
                gamma1_sym->set(h, u, v, gamma1M_->get(uoff, voff));
            }
        }
        offset_active += active_dim_[h];
    }
    /// Back transform 1-RDM to AO basis
    C_active_ao = psi::linalg::triplet(C_active, gamma1_sym, C_active, false, false, true);
    if (casscf_debug_print_)
        C_active_ao->print();
    // std::shared_ptr<JK> JK_fock = JK::build_JK(wfn_->basisset(),options_ );
    // JK_fock->set_memory(psi::Process::environment.get_memory() * 0.8);
    // JK_fock->set_cutoff(options_->get_double("INTEGRAL_SCREENING"));
    // JK_fock->initialize();
    // JK_->set_allow_desymmetrization(true);
    JK_->set_do_K(true);
    std::vector<std::shared_ptr<psi::Matrix>>& Cl = JK_->C_left();
    std::vector<std::shared_ptr<psi::Matrix>>& Cr = JK_->C_right();

    /// Since this is CASSCF this will always be an active fock matrix
    psi::SharedMatrix Identity(new psi::Matrix("I", nirrep_, nsopi_, nsopi_));
    Identity->identity();
    Cl.clear();
    Cr.clear();
    Cl.push_back(C_active_ao);
    Cr.push_back(Identity);

    /// If there is no restricted_docc, there is no C_core
    if (restricted_docc_dim_.sum() > 0) {
        for (size_t h = 0; h < nirrep_; h++) {
            for (int i = 0; i < restricted_docc_dim_[h]; i++) {
                C_core->set_column(h, i, Ca_sym_->get_column(h, i + frozen_docc_dim_[h]));
            }
        }

        if (casscf_debug_print_) {
            C_core->print();
        }

        Cl.push_back(C_core);
        Cr.push_back(C_core);
    }

    JK_->compute();

    psi::SharedMatrix J_act = JK_->J()[0];
    psi::SharedMatrix K_act = JK_->K()[0];
    psi::SharedMatrix F_act = J_act->clone();
    K_act->scale(0.5);
    F_act->subtract(K_act);
    F_act->transform(Ca_sym_);

    if (restricted_docc_dim_.sum() > 0) {
        psi::SharedMatrix J_core = JK_->J()[1];
        psi::SharedMatrix K_core = JK_->K()[1];
        J_core->scale(2.0);
        F_core = J_core->clone();
        F_core->subtract(K_core);
    }

    /// If there are frozen orbitals, need to add
    /// FrozenCore Fock matrix to inactive block
    if (casscf_freeze_core_) {
        F_core->add(F_froze_);
    }

    F_core->transform(Ca_sym_);
    // F_core->set_name("TRANSFORM BUG?");
    // F_core->print();
    // psi::SharedMatrix F_core_triplet = psi::linalg::triplet(Ca_sym_, F_core_tmp,
    // Ca_sym_, true, false, false);
    // F_core_triplet->set_name("TripletTransform");
    // F_core_triplet->print();
    F_core->add(H_);

    int offset = 0;
    for (size_t h = 0; h < nirrep_; h++) {
        for (int p = 0; p < nmopi_[h]; p++) {
            for (int q = 0; q < nmopi_[h]; q++) {
                F_core_c1->set(p + offset, q + offset,
                               F_core->get(h, p + frozen_docc_dim_[h], q + frozen_docc_dim_[h]));
            }
        }
        offset += nmopi_[h];
    }
    psi::SharedMatrix F_active_c1(new psi::Matrix("F_act", nmo_, nmo_));
    int offset_nofroze = 0;
    int offset_froze = 0;
    psi::Dimension no_frozen_dim = mo_space_info_->dimension("ALL");

    for (size_t h = 0; h < nirrep_; h++) {
        int froze = frozen_docc_dim_[h];
        for (int p = 0; p < nmopi_[h]; p++) {
            for (int q = 0; q < nmopi_[h]; q++) {
                F_active_c1->set(p + offset_froze, q + offset_froze,
                                 F_act->get(h, p + froze, q + froze));
            }
        }
        offset_froze += nmopi_[h];
        offset_nofroze += no_frozen_dim[h];
    }
    if (casscf_debug_print_) {
        F_active_c1->print();
    }
    F_act_ = F_active_c1;
    F_core_ = F_core_c1;
    if (casscf_debug_print_) {
        F_core_->set_name("INACTIVE_FOCK");
        F_core_->print();
        F_act_->set_name("ACTIVE_FOCK");
        F_act_->print();
    }
}
PostCASSCFOrbitalOptimizer::PostCASSCFOrbitalOptimizer(ambit::Tensor Gamma1, ambit::Tensor Gamma2,
                                                       ambit::Tensor two_body_ab,
                                                       std::shared_ptr<ForteOptions> options,
                                                       std::shared_ptr<MOSpaceInfo> mo_space_info)
    : OrbitalOptimizer(Gamma1, Gamma2, two_body_ab, options, mo_space_info) {}
PostCASSCFOrbitalOptimizer::~PostCASSCFOrbitalOptimizer() {}
void PostCASSCFOrbitalOptimizer::form_fock_intermediates() {
    psi::SharedMatrix F_core_c1(new psi::Matrix("F_core_no_sym", nmo_, nmo_));
    psi::SharedMatrix F_active_c1(new psi::Matrix("F_active_no_sym", nmo_, nmo_));
    ambit::Tensor F_core = ambit::Tensor::build(ambit::CoreTensor, "F_core", {nmo_, nmo_});
    ambit::Tensor F_active = ambit::Tensor::build(ambit::CoreTensor, "F_active", {nmo_, nmo_});
    H_->transform(Ca_sym_);
    if (casscf_debug_print_) {
        H_->set_name("CORR_HAMIL");
        H_->print();
    }

    /// Form F_core and F_active using user provided integrals
    F_core("p, q") = 2.0 * pq_mm_("pqmm") - pm_qm_("pmqm");
    F_active("p, q") = gamma1_("u, v") * (pq_uv_("pquv"));
    F_active("p, q") -= 0.5 * gamma1_("u, v") * (pu_qv_("pquv"));
    F_core_c1 = tensor_to_matrix(F_core);
    F_active_c1 = tensor_to_matrix(F_active);
    F_core_ = F_core_c1;
    F_act_ = F_core_c1;
}
} // namespace forte
