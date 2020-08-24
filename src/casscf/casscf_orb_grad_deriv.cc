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
        // TODO: throw warning for high spin?

        outfile->Printf("\n  Warning: MCSCF gradient detected frozen-core orbitals.");
        outfile->Printf(" They are ASSUMED from Hartree-Fock!");
        print_h2("Solving CP-MCSCF Equations for Frozen-Core Approximation");

        hf_ndoccpi_ = ints_->wfn()->doccpi();
        hf_nuoccpi_ = nmopi_ - ints_->wfn()->doccpi();

        hf_docc_mos_.clear();
        hf_uocc_mos_.clear();
        for (int h = 0, offset = 0; h < nirrep_; ++h) {
            for (int i = 0; i < hf_ndoccpi_[h]; ++i) {
                hf_docc_mos_.push_back(i + offset);
            }
            for (int a = 0; a < hf_nuoccpi_[h]; ++a) {
                hf_uocc_mos_.push_back(a + hf_ndoccpi_[h] + offset);
            }
            offset += nmopi_[h];
        }

        epsilon_ = ints_->wfn()->epsilon_a();

        Zfc_ = std::make_shared<psi::Matrix>("Z_fc", nfrzcpi_, hf_nuoccpi_);
        Wfc_ = std::make_shared<psi::Matrix>("W_fc", nfrzcpi_, hf_nuoccpi_);

        solve_Zfc();
        Zfc_->print();

        build_Wfc();
        Wfc_->print();
    }

    // back-transform densities
    auto L = Lagrangian();
    L->print();
    L->back_transform(C_);
    L->set_name("Lagrangian AO Back-Transformed");
    ints_->wfn()->Lagrangian()->copy(L);

    // 1-RDM
    auto D1 = opdm();
    D1->scale(0.5);
    D1->print();
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
    auto Zfc_fixed = build_Zfc_fixed();

    // start iteration
    bool converged = false;
    int iter = 1;
    int maxiter = options_->get_int("CPMCSCF_MAXITER");
    SharedMatrix Zold = Zfc_->clone();

    outfile->Printf("\n    *======================*");
    outfile->Printf("\n    * Iter     Delta Z rms *");
    outfile->Printf("\n    *----------------------*");

    while (iter <= maxiter) {
        // compute r.h.s. of CP-MCSCF equation
        Zfc_ = build_Lfc();
        Zfc_->add(Zfc_fixed);

        // divide by HF orbital energies
        for (int h = 0; h < nirrep_; ++h) {
            for (int I = 0; I < nfrzcpi_[h]; ++I) {
                double e_I = epsilon_->get(h, I);
                for (int a = 0, offset = hf_ndoccpi_[h]; a < hf_nuoccpi_[h]; ++a) {
                    double e_a = epsilon_->get(h, a + offset);
                    double v = Zfc_->get(h, I, a) / (e_I - e_a);
                    Zfc_->set(h, I, a, v);
                }
            }
        }

        // test convergence
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
        outfile->Printf("\n  Error: %s", msg.c_str());
        outfile->Printf("\n  Please check if the specified frozen-core orbitals make sense.");
        throw std::runtime_error(msg);
    }
}

SharedMatrix CASSCF_ORB_GRAD::build_Zfc_fixed() {
    auto rhs = std::make_shared<psi::Matrix>("Z_fc Fixed", nfrzcpi_, hf_nuoccpi_);

    // fill in data
    for (int h = 0, offset_I = 0, offset_a = 0; h < nirrep_; ++h) {
        offset_a += hf_ndoccpi_[h];

        for (int I = 0; I < nfrzcpi_[h]; ++I) {
            // space label and index of I in ambit Tensor
            std::string space_I = mos_rel_space_[I + offset_I].first;
            size_t n_I = mos_rel_space_[I + offset_I].second;
            size_t size_I = label_to_mos_[space_I].size();

            for (int a = 0; a < hf_nuoccpi_[h]; ++a) {
                // space label and index of a in ambit Tensor
                std::string space_a = mos_rel_space_[a + offset_a].first;
                size_t n_a = mos_rel_space_[a + offset_a].second;
                size_t size_a = label_to_mos_[space_a].size();

                // A_{aI}
                double value = A_.block(space_a + space_I).data()[n_a * size_I + n_I];

                // A_{Ia}
                value -= A_.block(space_I + space_a).data()[n_I * size_a + n_a];

                // set value
                rhs->set(h, I, a, 2.0 * value);
            }
        }

        offset_I += nmopi_[h];
        offset_a += hf_nuoccpi_[h];
    }

    return rhs;
}

SharedMatrix CASSCF_ORB_GRAD::build_Lfc() {
    // L_{Ic} = C_RJ C_Sd Z_Jd [4 * (RS|MN) - (RN|MS) - (RM|NS)] C_MI C_Nc
    // AO: M,N,R,S; frozen-core: I,J; HF UOCC: c,d
    auto Lfc_so = std::make_shared<psi::Matrix>("L_fc SO", nsopi_, nsopi_);

    // grab the frozen core part of Ca
    auto Cfrzc = std::make_shared<psi::Matrix>("C_FRZC", nirrep_, nsopi_, nfrzcpi_);
    for (int h = 0; h < nirrep_; ++h) {
        for (int i = 0; i < nfrzcpi_[h]; ++i) {
            Cfrzc->set_column(h, i, C_->get_column(h, i));
        }
    }

    // grab the unoccupied part of Ca
    auto Cuocc = std::make_shared<psi::Matrix>("C_UOCC", nirrep_, nsopi_, hf_nuoccpi_);
    for (int h = 0; h < nirrep_; ++h) {
        for (int i = 0, offset = hf_ndoccpi_[h]; i < hf_nuoccpi_[h]; ++i) {
            Cuocc->set_column(h, i, C_->get_column(h, i + offset));
        }
    }

    // dress Cuocc by Z_Jd
    auto Cuocc_dressed = psi::linalg::doublet(Cuocc, Zfc_, false, true);

    // JK build
    std::vector<std::shared_ptr<psi::Matrix>>& Cl = JK_->C_left();
    std::vector<std::shared_ptr<psi::Matrix>>& Cr = JK_->C_right();
    Cl.clear();
    Cr.clear();
    JK_->set_do_K(true);

    Cl.push_back(Cfrzc);
    Cr.push_back(Cuocc_dressed);
    JK_->compute();

    // copy data
    Lfc_so->copy(JK_->J()[0]);
    Lfc_so->scale(4.0);
    Lfc_so->subtract(JK_->K()[0]);

    // transform (4J - K) to MO
    auto Lfc = psi::linalg::triplet(Cfrzc, Lfc_so, Cuocc, true, false, false);

    // subtract the other K like term
    auto K2 = psi::linalg::triplet(Cuocc, JK_->K()[0], Cfrzc, true, false, false);
    Lfc->subtract(K2->transpose());

    return Lfc;
}

void CASSCF_ORB_GRAD::build_Wfc() {
    for (int h = 0, offset_I = 0, offset_a = 0; h < nirrep_; ++h) {
        offset_a += hf_ndoccpi_[h];

        for (int I = 0; I < nfrzcpi_[h]; ++I) {
            double epsilon_I = epsilon_->get(h, I);

            // space label and index of I in ambit Tensor
            std::string space_I = mos_rel_space_[I + offset_I].first;
            size_t n_I = mos_rel_space_[I + offset_I].second;

            for (int a = 0; a < hf_nuoccpi_[h]; ++a) {
                // space label and index of a in ambit Tensor
                std::string space_a = mos_rel_space_[a + offset_a].first;
                size_t n_a = mos_rel_space_[a + offset_a].second;
                size_t size_a = label_to_mos_[space_a].size();

                double value = 0.5 * epsilon_I * Zfc_->get(h, I, a);

                // add A_{Ia} part if a is active index
                if (space_a == "a")
                    value += A_.block(space_I + space_a).data()[n_I * size_a + n_a];

                Wfc_->set(h, I, a, value);
            }
        }

        offset_I += nmopi_[h];
        offset_a += hf_nuoccpi_[h];
    }
}

psi::SharedMatrix CASSCF_ORB_GRAD::Lagrangian() {
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

    // add frozen-core response part
    if (nfrzc_) {
        for (int h = 0; h < nirrep_; ++h) {
            for (int I = 0; I < nfrzcpi_[h]; ++I) {
                for (int a = 0, offset_a = hf_ndoccpi_[h]; a < hf_nuoccpi_[h]; ++a) {
                    double w = Wfc_->get(h, I, a);
                    L->add(h, I, a + offset_a, w);
                    L->add(h, a + offset_a, I, w);
                }
            }
        }
    }

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

    // frozen-core response part
    if (nfrzc_) {
        for (int h = 0; h < nirrep_; ++h) {
            for (int I = 0; I < nfrzcpi_[h]; ++I) {
                for (int a = 0, offset_a = hf_ndoccpi_[h]; a < hf_nuoccpi_[h]; ++a) {
                    double w = 0.5 * Zfc_->get(h, I, a);
                    D1->set(h, I, a + offset_a, w);
                    D1->add(h, a + offset_a, I, w);
                }
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

    // 1-rdm part from frozen-core response
    if (nfrzc_) {
        for (int h = 0, offset = 0; h < nirrep_; ++h) {
            for (int I = 0; I < nfrzcpi_[h]; ++I) {
                auto nI = I + offset;
                for (int a = 0; a < hf_nuoccpi_[h]; ++a) {
                    auto na = a + offset + hf_ndoccpi_[h];

                    double z_Ia = Zfc_->get(h, I, a);

                    for (const int ni: hf_docc_mos_) {
                        d2.write_value(nI, na, ni, ni, z_Ia, print, name, 0);
                        d2.write_value(nI, ni, ni, na, -0.5 * z_Ia, print, name, 0);
                        d2.write_value(na, nI, ni, ni, z_Ia, print, name, 0);
                        d2.write_value(na, ni, ni, nI, -0.5 * z_Ia, print, name, 0);
                    }
                }
            }
            offset += nmopi_[h];
        }
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
