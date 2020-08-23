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

#include "psi4/psi4-dec.h"
#include "psi4/psifiles.h"
#include "psi4/libfock/jk.h"
#include "psi4/libqt/qt.h"
#include "psi4/libmints/molecule.h"
#include "psi4/libmints/pointgrp.h"
#include "psi4/libmints/matrix.h"
#include "psi4/libmints/wavefunction.h"
#include "psi4/libmints/basisset.h"
#include "psi4/libmints/vector.h"
#include "psi4/libpsi4util/PsiOutStream.h"
#include "psi4/libpsi4util/process.h"
#include "psi4/libiwl/iwl.hpp"
#include "psi4/libpsio/psio.hpp"

#include "helpers/printing.h"
#include "helpers/lbfgs/lbfgs.h"
#include "helpers/timer.h"
#include "integrals/integrals.h"
#include "integrals/active_space_integrals.h"
#include "base_classes/rdms.h"

#include "gradient_tpdm/backtransform_tpdm.h"
#include "casscf/casscf_orb_grad.h"

using namespace psi;
using namespace ambit;

namespace forte {

void CASSCF_ORB_GRAD::compute_nuclear_gradient() {
    // check availability
    if (mo_space_info_->size("FROZEN_UOCC"))
        throw std::runtime_error("Frozen-virtual not implemented in MCSCF gradient!");

    if (nfrzc_) {
        outfile->Printf("\n  Warning: MCSCF gradient detected frozen-core orbitals.");
        outfile->Printf(" They are ASSUMED from Hartree-Fock!");

        auto nuoccpi = nmopi_ - ints_->wfn()->doccpi();
        Zfc_ = std::make_shared<psi::Matrix>("Z_fc", nfrzcpi_, nuoccpi);
        Wfc_ = std::make_shared<psi::Matrix>("W_fc", nfrzcpi_, nuoccpi);

        solve_Zfc();
        build_Wfc();
    }

    // back-transform densities
    auto L = Lagrangian();
    L->back_transform(C_);
    L->set_name("Lagrangian AO Back-Transformed");
    ints_->wfn()->Lagrangian()->copy(L);

    // 1-RDM
    auto D1 = opdm();
    D1->scale(0.5);
    D1->back_transform(C_);
    D1->set_name("D1a AO Back-Transformed");
    ints_->wfn()->Da()->copy(D1);
    ints_->wfn()->Db()->copy(D1);

    // 2-RDM
    dump_tpdm_iwl();

    std::vector<std::shared_ptr<psi::MOSpace>> spaces{psi::MOSpace::all};
    auto transform = std::make_shared<psi::TPDMBackTransform>(
        ints_->wfn(), spaces,
        psi::IntegralTransform::TransformationType::Restricted, // Transformation type
        psi::IntegralTransform::OutputType::DPDOnly,            // Output buffer
        psi::IntegralTransform::MOOrdering::PitzerOrder,        // MO ordering (does not matter)
        psi::IntegralTransform::FrozenOrbitals::None);          // Frozen orbitals
    transform->set_print(debug_print_ ? 5 : print_);
    transform->backtransform_density();
}

void CASSCF_ORB_GRAD::solve_Zfc() {
    // build Z independent part

    // start iteration
    bool converged = false;
    int iter = 1;
    int maxiter = options_->get_int("CPMCSCF_MAXITER");
    SharedMatrix Zold = Zfc_->clone();

    outfile->Printf("\n\n  ==> Solving MCSCF Z Vector <==\n");
    outfile->Printf("\n    *======================*");
    outfile->Printf("\n    * Iter    Delta Z norm *");
    outfile->Printf("\n    *----------------------*");

    while (iter <= maxiter) {
        // Zfc_ = build_Lfc();

        // add independent part
        // Zfc_->add(...);

        // divide by HF orbital energies


        Zold->subtract(Zfc_);
        double Zrms = Zold->rms();
        outfile->Printf("\n    * %4d    %12.4e *", iter, Zrms);
        if (Zrms < g_conv_) {
            converged = true;
            break;
        }
        Zold->copy(Zfc_);
        iter++;
    }
    outfile->Printf("\n    *======================*");

    if (!converged) {
        auto msg = "CP-MCSCF did not converge in " + std::to_string(maxiter) + " iterations.";
        throw std::runtime_error(msg);
    }
}

SharedMatrix CASSCF_ORB_GRAD::build_Lfc() {
    // L_{Ic} = C_RJ C_Sd Z_Jd [4 * (RS|MN) - (RN|MS) - (RM|NS)] C_MI C_Nc
    // AO: M,N,R,S; frozen-core: I,J; unoccupied in HF: c,d
    auto Lfc_so = std::make_shared<psi::Matrix>("L_fc SO", nsopi_, nsopi_);

    // return psi::linalg::triplet(Cfc, Lfc_so, Cu, true, false, false);
    return Lfc_so;
}

psi::SharedMatrix CASSCF_ORB_GRAD::Lagrangian() {
    auto S = ints_->wfn()->S();
    auto I = linalg::triplet(C_, S, C_, true, false, false);
    I->print();

    // format A matrix
    auto L = std::make_shared<psi::Matrix>("Lagrangian (MO)", nmopi_, nmopi_);

    A_.citerate(
        [&](const std::vector<size_t>& i, const std::vector<SpinType>&, const double& value) {
            auto irrep_index_pair1 = mos_rel_[i[0]];
            auto irrep_index_pair2 = mos_rel_[i[1]];

            int h1 = irrep_index_pair1.first;

            if (h1 == irrep_index_pair2.first) {
                auto p = irrep_index_pair1.second;
                auto q = irrep_index_pair2.second;
                L->set(h1, p, q, value);
            }
        });

    return L;
}

psi::SharedMatrix CASSCF_ORB_GRAD::opdm() {
    auto D1 = std::make_shared<psi::Matrix>("OPDM (MO)", nmopi_, nmopi_);

    // inactive docc part
    for (int h = 0; h < nirrep_; ++h) {
        for (int i = 0; i < ndoccpi_[h]; ++i) {
            D1->set(h, i, i, 2.0);
        }
    }

    // active part
    for (int h = 0; h < nirrep_; ++h) {
        auto offset = ndoccpi_[h];
        for (int u = 0; u < nactvpi_[h]; ++u) {
            auto nu = u + offset;
            for (int v = u; v < nactvpi_[h]; ++v) {
                auto nv = v + offset;
                D1->set(h, nu, nv, rdm1_->get(h, u, v));
                D1->set(h, nv, nu, rdm1_->get(h, v, u));
            }
        }
    }

    return D1;
}

void CASSCF_ORB_GRAD::dump_tpdm_iwl() {
    auto psio = _default_psio_lib_;
    IWL d2(psio.get(), PSIF_MO_TPDM, 1.0e-15, 0, 0);
    std::string name = "outfile";
    int print = debug_print_ ? 1 : 0;

    // inactive docc part
    auto docc_mos = mo_space_info_->absolute_mo("INACTIVE_DOCC");
    for (int i = 0, ndocc = docc_mos.size(); i < ndocc; ++i) {
        auto ni = docc_mos[i];
        d2.write_value(ni, ni, ni, ni, 1.0, print, name, 0);
        for (int j = 0; j < i; ++j) {
            auto nj = docc_mos[j];
            d2.write_value(ni, ni, nj, nj, 2.0, print, name, 0);
            d2.write_value(nj, nj, ni, ni, 2.0, print, name, 0);
            d2.write_value(ni, nj, nj, ni, -1.0, print, name, 0);
            d2.write_value(nj, ni, ni, nj, -1.0, print, name, 0);
        }
    }

    // 1-rdm part
    for (int h = 0, offset = 0; h < nirrep_; ++h) {
        for (int u = 0; u < nactvpi_[h]; ++u) {
            auto nu = u + offset + ndoccpi_[h];
            for (int v = u; v < nactvpi_[h]; ++v) {
                auto nv = v + offset + ndoccpi_[h];

                double d_uv = rdm1_->get(h, u, v);
                double d_vu = rdm1_->get(h, v, u);

                for (int i = 0, ndocc = docc_mos.size(); i < ndocc; ++i) {
                    auto ni = docc_mos[i];

                    d2.write_value(nu, nv, ni, ni, 2.0 * d_uv, print, name, 0);
                    d2.write_value(nu, ni, ni, nv, -d_uv, print, name, 0);

                    if (u != v) {
                        d2.write_value(nv, nu, ni, ni, 2.0 * d_vu, print, name, 0);
                        d2.write_value(nv, ni, ni, nu, -d_vu, print, name, 0);
                    }
                }
            }
        }
        offset += nmopi_[h];
    }

    // 2-rdm part
    auto& d2_data = D2_.block("aaaa").data();
    auto na2 = nactv_ * nactv_;
    auto na3 = nactv_ * na2;

    for (size_t u = 0; u < nactv_; ++u) {
        auto nu = actv_mos_[u];
        for (size_t v = 0; v < nactv_; ++v) {
            auto nv = actv_mos_[v];
            for (size_t x = 0; x < nactv_; ++x) {
                auto nx = actv_mos_[x];
                for (size_t y = 0; y < nactv_; ++y) {
                    auto ny = actv_mos_[y];

                    double value = d2_data[u * na3 + v * na2 + x * nactv_ + y];
                    d2.write_value(nu, nv, nx, ny, 0.5 * value, print, name, 0);
                }
            }
        }
    }

    d2.flush(1);
    d2.set_keep_flag(1);
    d2.close();
}
} // namespace forte
