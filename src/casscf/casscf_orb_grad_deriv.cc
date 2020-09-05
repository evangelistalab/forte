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
#include "psi4/libdpd/dpd.h"

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
    // format A to SharedMatrix
    format_A_matrix();

    is_frozen_orbs_ = nfrzc_ or mo_space_info_->size("FROZEN_UOCC");

    if (is_frozen_orbs_) {
        // TODO: terminate for high spin?

        outfile->Printf("\n  Warning: MCSCF gradient code detected frozen orbitals.");
        outfile->Printf("\n  They are ASSUMED from well-converged Hartree-Fock of Psi4!");

        print_h2("Solving CPSCF Equations for MCSCF Gradient with Frozen Orbitals");

        // set up HF orbitals
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

        // transform A matrix to HF basis
        Am_ = psi::linalg::triplet(U_, Am_, U_, false, false, true);
        Am_->set_name("A (SCF)");
        Am_->print();

        // grab Hartree-Fock orbital energies from Psi4
        epsilon_ = ints_->wfn()->epsilon_a();

        // solve CPSCF equations
        solve_cpscf();
        solve_cpscf_jcp();
    }

    // back-transform Lagrangian
    compute_Lagrangian();

    // back-transform 1-RDM
    compute_opdm_ao();

    // dump 2-RDM to disk
    dump_tpdm_iwl();

    // back-transform 2-RDM
    std::vector<std::shared_ptr<psi::MOSpace>> spaces{psi::MOSpace::all};
    auto transform = std::make_shared<psi::TPDMBackTransform>(
        ints_->wfn(), spaces,
        psi::IntegralTransform::TransformationType::Restricted, // Transformation type
        psi::IntegralTransform::OutputType::DPDOnly,            // Output buffer
        psi::IntegralTransform::MOOrdering::PitzerOrder,        // MO ordering (does not matter)
        psi::IntegralTransform::FrozenOrbitals::None);          // Frozen orbitals
    if (is_frozen_orbs_) {
        transform->set_Ca_additional(C0_);
    }
    transform->set_print(debug_print_ ? 5 : print_);
    transform->backtransform_density();
}

void CASSCF_ORB_GRAD::format_A_matrix() {
    Am_ = std::make_shared<psi::Matrix>("A (MCSCF)", nmopi_, nmopi_);

    A_.citerate(
        [&](const std::vector<size_t>& i, const std::vector<SpinType>&, const double& value) {
            auto irrep_index_pair1 = mos_rel_[i[0]];
            auto irrep_index_pair2 = mos_rel_[i[1]];

            int h1 = irrep_index_pair1.first;

            if (h1 == irrep_index_pair2.first) {
                auto p = irrep_index_pair1.second;
                auto q = irrep_index_pair2.second;
                Am_->set(h1, p, q, value);
            }
        });
}

void CASSCF_ORB_GRAD::build_mixed_fock() {
    /* Mixed Fock matrix is defined as:
     *   F_{pq} = h_{pq} + sum_{i} [2(pq|ii) - (pi|iq)]
     *   p,q: MCSCF indices; i: HF index
     * The constraint is F_{Ip} = 0 for p != I
     */

    // grab part of Ca for inactive docc
    auto Cdocc = std::make_shared<psi::Matrix>("C_INACTIVE", nirrep_, nsopi_, hf_ndoccpi_);
    for (int h = 0; h < nirrep_; h++) {
        for (int i = 0; i < hf_ndoccpi_[h]; i++) {
            Cdocc->set_column(h, i, C0_->get_column(h, i));
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

    auto J = JK_->J()[0];
    J->scale(2.0);
    J->subtract(JK_->K()[0]);
    J->add(H_ao_);

    Fock_mixed_->copy(J);
    Fock_mixed_->transform(C_);

    Fock_mixed_->print();
}

void CASSCF_ORB_GRAD::solve_cpscf() {
    // compute orbital gradient in Hartree-Fock basis
    auto G = Am_->clone();
    G->subtract(Am_->transpose());
    G->set_name("A - A^T");

    Z_ = std::make_shared<psi::Matrix>("Z", nmopi_, nmopi_);

    // occupied-occupied part
    for (int h = 0; h < nirrep_; ++h) {
        for (int i = 0; i < hf_ndoccpi_[h]; ++i) {
            for (int j = i + 1; j < hf_ndoccpi_[h]; ++j) {
                double value = G->get(h, i, j) / (epsilon_->get(h, j) - epsilon_->get(h, i));
                Z_->set(h, i, j, value);
                Z_->set(h, j, i, value);
            }
        }
    }

    // virtual-virtual part
    for (int h = 0; h < nirrep_; ++h) {
        for (int a = hf_ndoccpi_[h]; a < nmopi_[h]; ++a) {
            for (int b = a + 1; b < nmopi_[h]; ++b) {
                double value = G->get(h, a, b) / (epsilon_->get(h, b) - epsilon_->get(h, a));
                Z_->set(h, a, b, value);
                Z_->set(h, b, a, value);
            }
        }
    }

    // TODO: solve this linear system using L-BFGS

    // start iteration
    bool converged = false;
    int iter = 1;
    int maxiter = options_->get_int("CPMCSCF_MAXITER");
    SharedMatrix Zold = Z_->clone();

    outfile->Printf("\n    *======================*");
    outfile->Printf("\n    * Iter     Delta Z rms *");
    outfile->Printf("\n    *----------------------*");

    while (iter <= maxiter) {
        auto L = contract_Z_Lsuper();

        for (int h = 0; h < nirrep_; ++h) {
            for (int i = 0; i < hf_ndoccpi_[h]; ++i) {
                for (int a = hf_ndoccpi_[h]; a < nmopi_[h]; ++a) {
                    double value = (G->get(h, i, a) - 0.5 * L->get(h, a, i)) /
                                   (epsilon_->get(h, a) - epsilon_->get(h, i));
                    Z_->set(h, i, a, value);
                    Z_->set(h, a, i, value);
                }
            }
        }

        // test convergence
        Zold->subtract(Z_);
        double Zrms = Zold->rms();
        outfile->Printf("\n    * %4d    %12.4e *", iter, Zrms);
        if (Zrms < g_conv_) {
            converged = true;
            break;
        }
        Zold->copy(Z_);
        iter++;
    }
    outfile->Printf("\n    *======================*");

    if (!converged) {
        auto msg = "CP-SCF did not converge in " + std::to_string(maxiter) + " iterations.";
        outfile->Printf("\n  Error: %s", msg.c_str());
        outfile->Printf("\n  Please check if the specified frozen orbitals make sense.");
        throw std::runtime_error(msg);
    }

    Z_->print();
}

void CASSCF_ORB_GRAD::solve_cpscf_jcp() {
    // prepare some blocks of Ca
    auto Cdocc = std::make_shared<psi::Matrix>("Cdocc_hf", nsopi_, hf_ndoccpi_);
    auto Cuocc = std::make_shared<psi::Matrix>("Cuocc_hf", nsopi_, hf_nuoccpi_);
    auto Cfrzc = std::make_shared<psi::Matrix>("Cfrzc", nsopi_, nfrzcpi_);
    auto Cvalc = std::make_shared<psi::Matrix>("Cvalc", nsopi_, hf_ndoccpi_ - nfrzcpi_);

    for (int h = 0; h < nirrep_; ++h) {
        // frozen core
        for (int i = 0; i < nfrzcpi_[h]; ++i) {
            Cfrzc->set_column(h, i, C0_->get_column(h, i));
        }

        // valence
        for (int k = 0; k < hf_ndoccpi_[h] - nfrzcpi_[h]; ++k) {
            int nk = k + nfrzcpi_[h];
            Cvalc->set_column(h, k, C0_->get_column(h, nk));
        }

        // docc
        for (int p = 0; p < hf_ndoccpi_[h]; ++p) {
            Cdocc->set_column(h, p, C0_->get_column(h, p));
        }

        // virtual
        for (int a = 0; a < hf_nuoccpi_[h]; ++a) {
            int na = a + hf_ndoccpi_[h];
            Cuocc->set_column(h, a, C0_->get_column(h, na));
        }
    }
    Cfrzc->print();
    Cvalc->print();
    Cdocc->print();
    Cuocc->print();

    auto Atilde = Am_->clone();
    Atilde->subtract(Am_->transpose());
    Atilde->set_name("A - A^T (SCF)");
    Atilde->print();

    auto R = std::make_shared<psi::Matrix>("R", hf_nuoccpi_, hf_ndoccpi_);
    for (int h = 0; h < nirrep_; ++h) {
        for (int p = 0; p < hf_ndoccpi_[h]; ++p) {
            for (int a = 0; a < hf_nuoccpi_[h]; ++a) {
                R->set(h, a, p, Atilde->get(h, a + hf_ndoccpi_[h], p));
            }
        }
    }
    R->print();

    auto As = std::make_shared<psi::Matrix>("As", hf_ndoccpi_ - nfrzcpi_, nfrzcpi_);
    for (int h = 0; h < nirrep_; ++h) {
        for (int i = 0; i < nfrzcpi_[h]; ++i) {
            for (int k = 0; k < hf_ndoccpi_[h] - nfrzcpi_[h]; ++k) {
                int nk = k + nfrzcpi_[h];
                double v = Atilde->get(h, nk, i) / (epsilon_->get(h, nk) - epsilon_->get(h, i));
                As->set(h, k, i, v);
            }
        }
    }
    As->print();

    // contract with L: sum_{ik} As_{ki} L_{ki,ap}
    auto Cdressed = psi::linalg::doublet(Cvalc, As, false, false);

    build_JK_fock(Cdressed, Cfrzc);
    auto J = JK_->J()[0];
    J->scale(4.0);
    J->subtract(JK_->K()[0]);
    J->subtract((JK_->K()[0])->transpose());

    auto R1 = psi::linalg::triplet(Cuocc, J, Cdocc, true, false, false);
    R->subtract(R1);

    // solve Z_{ap}
    auto Z = std::make_shared<psi::Matrix>("Z", hf_nuoccpi_, hf_ndoccpi_);
    auto Zold = Z->clone();

    bool converged = false;
    int iter = 1;
    int maxiter = options_->get_int("CPMCSCF_MAXITER");

    outfile->Printf("\n    *======================*");
    outfile->Printf("\n    * Iter     Delta Z rms *");
    outfile->Printf("\n    *----------------------*");

    while (iter <= maxiter) {
        auto RHS = R->clone();

        auto Cdressed = psi::linalg::doublet(Cuocc, Z, false, false);

        build_JK_fock(Cdressed, Cdocc);
        auto J = JK_->J()[0];
        J->scale(4.0);
        J->subtract(JK_->K()[0]);
        J->subtract((JK_->K()[0])->transpose());

        auto LZ = linalg::triplet(Cuocc, J, Cdocc, true, false, false);

        RHS->subtract(LZ);

        for (int h = 0; h < nirrep_; ++h) {
            for (int p = 0; p < hf_ndoccpi_[h]; ++p) {
                for (int a = 0; a < hf_nuoccpi_[h]; ++a) {
                    double value = RHS->get(h, a, p);
                    Z->set(h, a, p,
                           value / (epsilon_->get(h, a + hf_ndoccpi_[h]) - epsilon_->get(h, p)));
                }
            }
        }

        // test convergence
        Zold->subtract(Z);
        double Zrms = Zold->rms();
        outfile->Printf("\n    * %4d    %12.4e *", iter, Zrms);
        if (Zrms < g_conv_) {
            converged = true;
            break;
        }
        Zold->copy(Z);
        iter++;
    }
    outfile->Printf("\n    *======================*");

    if (!converged) {
        auto msg = "CP-SCF did not converge in " + std::to_string(maxiter) + " iterations.";
        outfile->Printf("\n  Error: %s", msg.c_str());
        outfile->Printf("\n  Please check if the specified frozen orbitals make sense.");
        throw std::runtime_error(msg);
    }

    Z->print();

    // compute E_{mu nu}
    auto E = linalg::triplet(Cuocc, Z, Cdocc, false, false, true);
    E->add(linalg::triplet(Cvalc, As, Cfrzc, false, false, true));
    E->print();

    // AO 1RDM from CASSCF
    auto D1 = std::make_shared<psi::Matrix>("OPDM", nmopi_, nmopi_);

    // inactive docc part
    for (int h = 0; h < nirrep_; ++h) {
        for (int i = 0; i < ndoccpi_[h]; ++i) {
            D1->set(h, i, i, 1.0);
        }
    }

    // active part
    for (int h = 0; h < nirrep_; ++h) {
        auto offset = ndoccpi_[h];
        for (int u = 0; u < nactvpi_[h]; ++u) {
            auto nu = u + offset;
            for (int v = u; v < nactvpi_[h]; ++v) {
                auto nv = v + offset;
                D1->set(h, nu, nv, 0.5 * rdm1_->get(h, u, v));
                D1->set(h, nv, nu, 0.5 * rdm1_->get(h, v, u));
            }
        }
    }

    D1 = psi::linalg::triplet(C_, D1, C_, false, false, true);
    D1->set_name("D1a AO Back-Transformed");
    D1->subtract(E);

    //    // push to Psi4
    //    ints_->wfn()->Da()->copy(D1);
    //    ints_->wfn()->Db()->copy(D1);

    // compute X_{mu nu}
    auto Zrp = std::make_shared<psi::Matrix>("Z_rp", nmopi_, hf_ndoccpi_);
    for (int h = 0; h < nirrep_; ++h) {
        // valence/frozen-core
        for (int i = 0; i < nfrzcpi_[h]; ++i) {
            for (int k = 0; k < hf_ndoccpi_[h] - nfrzcpi_[h]; ++k) {
                int nk = k + nfrzcpi_[h];
                Zrp->set(h, nk, i, As->get(h, k, i));
            }
        }
        // vitual/occupied
        for (int p = 0; p < hf_ndoccpi_[h]; ++p) {
            for (int a = 0; a < hf_nuoccpi_[h]; ++a) {
                int na = a + hf_ndoccpi_[h];
                Zrp->set(h, na, p, Z->get(h, a, p));
            }
        }
    }
    Zrp->print();

    Cdressed = psi::linalg::doublet(C0_, Zrp, false, false);

    build_JK_fock(Cdressed, Cdocc);
    J = JK_->J()[0];
    J->scale(4.0);
    J->subtract(JK_->K()[0]);
    J->subtract((JK_->K()[0])->transpose());

    auto X = psi::linalg::triplet(C0_, J, Cdocc, true, false, false);
    X->set_name("X MO");
    X->print();

    auto Zrp_s = Zrp->clone();
    Zrp_s->set_name("scaled Z_rp");
    for (int h = 0; h < nirrep_; ++h) {
        for (int r = 0; r < nmopi_[h]; ++r) {
            for (int p = 0; p < hf_ndoccpi_[h]; ++p) {
                double v = Zrp->get(h, r, p) * (epsilon_->get(h, r) + epsilon_->get(h, p));
                Zrp_s->set(h, r, p, v);
            }
        }
    }
    Zrp_s->print();

    X->add(Zrp_s);
    X->print();

    auto Xao = psi::linalg::triplet(C0_, X, Cdocc, false, false, true);
    Xao->set_name("X AO");
    Xao->print();
}

SharedMatrix CASSCF_ORB_GRAD::build_Zfc_fixed() {
    auto rhs = std::make_shared<psi::Matrix>("Z_fc Fixed", nfrzcpi_, nmopi_);

    // fill in data
    for (int h = 0, offset = 0, offset_a = 0; h < nirrep_; ++h) {
        offset_a += hf_ndoccpi_[h];

        for (int I = 0; I < nfrzcpi_[h]; ++I) {
            // space label and index of I in ambit Tensor
            std::string space_I = mos_rel_space_[I + offset].first;
            size_t n_I = mos_rel_space_[I + offset].second;
            size_t size_I = label_to_mos_[space_I].size();

            for (int p = 0; p < nmopi_[h]; ++p) {
                // space label and index of p in ambit Tensor
                std::string space_p = mos_rel_space_[p + offset].first;
                size_t n_p = mos_rel_space_[p + offset].second;
                size_t size_p = label_to_mos_[space_p].size();

                // A_{pI}
                double value = A_.block(space_p + space_I).data()[n_p * size_I + n_I];

                // A_{Ip}
                value -= A_.block(space_I + space_p).data()[n_I * size_p + n_p];

                // set value
                rhs->set(h, I, p, 2.0 * value);
            }

            //            for (int a = 0; a < hf_nuoccpi_[h]; ++a) {
            //                // space label and index of a in ambit Tensor
            //                std::string space_a = mos_rel_space_[a + offset_a].first;
            //                size_t n_a = mos_rel_space_[a + offset_a].second;
            //                size_t size_a = label_to_mos_[space_a].size();

            //                // A_{aI}
            //                double value = A_.block(space_a + space_I).data()[n_a * size_I + n_I];

            //                // A_{Ia}
            //                value -= A_.block(space_I + space_a).data()[n_I * size_a + n_a];

            //                // set value
            //                rhs->set(h, I, a, 2.0 * value);
            //            }
        }

        offset += nmopi_[h];
        //        offset_a += hf_nuoccpi_[h];
    }

    return rhs;
}

SharedMatrix CASSCF_ORB_GRAD::contract_Z_Lsuper() {
    // compute sum_{pq} Z_{pq} L_{pq,rs} using Hartree-Fock orbitals
    // super matrix L_{pq,rs} = 4 * (pq|rs) - (pr|sq) - (ps|rq)

    // C dressed by Z_pq
    auto Cdressed = psi::linalg::doublet(C0_, Z_, false, false);

    // JK build
    build_JK_fock(C0_, Cdressed);

    auto J = JK_->J()[0];
    J->scale(4.0);
    J->subtract(JK_->K()[0]);
    J->subtract((JK_->K()[0])->transpose());

    // transform to MO and return
    return psi::linalg::triplet(C0_, J, C0_, true, false, false);
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

void CASSCF_ORB_GRAD::compute_Lagrangian() {
    if (not is_frozen_orbs_) {
        // A matrix is equivalent to W when there is no frozen orbitals
        W_ = Am_->clone();

        W_->print();
        // transform to AO
        W_->back_transform(C_);
        W_->print();
    } else {
        Z_->print();
        W_ = std::make_shared<psi::Matrix>("Lagrangian", nmopi_, nmopi_);

        auto L = contract_Z_Lsuper();
        L->set_name("Z * Lsuper");
        L->print();

        // occupied-occupied part
        for (int h = 0; h < nirrep_; ++h) {
            for (int i = 0; i < hf_ndoccpi_[h]; ++i) {
                for (int j = i; j < hf_ndoccpi_[h]; ++j) {
                    double value = Am_->get(h, i, j) + Am_->get(h, j, i);
                    value += Z_->get(h, i, j) * (epsilon_->get(h, i) + epsilon_->get(h, j));
                    value += 0.5 * (L->get(h, i, j) + L->get(h, j, i));
                    W_->set(h, i, j, 0.5 * value);
                    W_->set(h, j, i, 0.5 * value);
                }
            }
        }

        // virtual-virtual part
        for (int h = 0; h < nirrep_; ++h) {
            for (int a = hf_ndoccpi_[h]; a < nmopi_[h]; ++a) {
                for (int b = a; b < nmopi_[h]; ++b) {
                    double value = Am_->get(h, a, b) + Am_->get(h, b, a);
                    value += Z_->get(h, a, b) * (epsilon_->get(h, a) + epsilon_->get(h, b));
                    W_->set(h, a, b, 0.5 * value);
                    W_->set(h, b, a, 0.5 * value);
                }
            }
        }

        // occupied-virtual part
        for (int h = 0; h < nirrep_; ++h) {
            for (int i = 0; i < hf_ndoccpi_[h]; ++i) {
                for (int a = hf_ndoccpi_[h]; a < nmopi_[h]; ++a) {
                    double value = Am_->get(h, i, a);
                    value += Z_->get(h, i, a) * epsilon_->get(h, i);
                    W_->set(h, i, a, value);
                    W_->set(h, a, i, value);

//                    double v = Am_->get(h, i, a) + Am_->get(h, a, i);
                    double v = 0.0;
                    v += Z_->get(h, i, a) * (epsilon_->get(h, i) + epsilon_->get(h, a));
                    v += 0.5 * L->get(h, a, i);
                    v *= 0.5;
                    outfile->Printf("\n  h = %d, i = %2d, a = %2d, value = %20.15f, v = %20.15f, diff = %.15f",
                                    h, i, a, value, v, value-v);
                }
            }
        }
        W_->print();

        // transform to AO
        W_->back_transform(C0_);
        W_->print();

        auto W = psi::linalg::triplet(C_, Am_, C_, false, false, true);
        W->set_name("Lagrangian PAPER");

        auto T = std::make_shared<psi::Matrix>("X SCF", hf_ndoccpi_, nmopi_);
        for (int h = 0; h < nirrep_; ++h) {
            for (int p = 0; p < hf_ndoccpi_[h]; ++p) {
                for (int r = 0; r < nmopi_[h]; ++r) {
                    double v = (epsilon_->get(h, p) + epsilon_->get(h, r)) * Z_->get(h, r, p);
                    T->set(h, p, r, v);
                }
            }
        }
        T->print();
        auto Csub = std::make_shared<psi::Matrix>("C0 docc", nsopi_, hf_ndoccpi_);
        for (int h = 0; h < nirrep_; ++h) {
            for (int i = 0; i < hf_ndoccpi_[h]; ++i) {
                Csub->set_column(h, i, C0_->get_column(h, i));
            }
        }
        W->add(psi::linalg::triplet(Csub, T, C0_, false, false, true));
        W->print();
    }

    // transform to AO and push to Psi4 Wavefunction
    ints_->wfn()->Lagrangian()->copy(W_);
    ints_->wfn()->Lagrangian()->set_name("Lagrangian AO Back-Transformed");
}

void CASSCF_ORB_GRAD::compute_opdm_ao() {
    auto D1 = std::make_shared<psi::Matrix>("OPDM", nmopi_, nmopi_);

    // inactive docc part
    for (int h = 0; h < nirrep_; ++h) {
        for (int i = 0; i < ndoccpi_[h]; ++i) {
            D1->set(h, i, i, 1.0);
        }
    }

    // active part
    for (int h = 0; h < nirrep_; ++h) {
        auto offset = ndoccpi_[h];
        for (int u = 0; u < nactvpi_[h]; ++u) {
            auto nu = u + offset;
            for (int v = u; v < nactvpi_[h]; ++v) {
                auto nv = v + offset;
                D1->set(h, nu, nv, 0.5 * rdm1_->get(h, u, v));
                D1->set(h, nv, nu, 0.5 * rdm1_->get(h, v, u));
            }
        }
    }

    D1 = psi::linalg::triplet(C_, D1, C_, false, false, true);
    D1->set_name("D1a AO Back-Transformed");

    // add Z vector due to frozen-core response
    if (is_frozen_orbs_) {
        auto T = psi::linalg::triplet(C0_, Z_, C0_, false, false, true);
        T->scale(0.5);
        T->print();
        D1->add(T);
    }

    // push to Psi4
    ints_->wfn()->Da()->copy(D1);
    ints_->wfn()->Db()->copy(D1);
}

void CASSCF_ORB_GRAD::dump_tpdm_iwl() {
    // if we solve response for frozen orbitals
    if (is_frozen_orbs_) {
        dump_tpdm_iwl_hf();
    } else {
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
}

void CASSCF_ORB_GRAD::dump_tpdm_iwl_hf() {
    auto psio = _default_psio_lib_;
    IWL d2(psio.get(), PSIF_MO_TPDM, 1.0e-15, 0, 0);
    std::string name = "outfile";
    int print = debug_print_ ? 1 : 0;

    //    auto psio = _default_psio_lib_;
    //    IWL d2(psio.get(), PSIF_MO_BB_TPDM, 1.0e-15, 0, 0); // use PSIF_MO_BB_TPDM
    //    std::string name = "outfile";
    //    int print = debug_print_ ? 1 : 0;

    //    auto frzv_start_pi = nmopi_ - mo_space_info_->dimension("FROZEN_UOCC");

    //    for (int h = 0, offset = 0; h < nirrep_; ++h) {
    //        for (int j = 0, ndocc = hf_docc_mos_.size(); j < ndocc; ++j) {
    //            int nj = hf_docc_mos_[j];

    //            // frozen-core/active-docc
    //            for (int I = 0; I < nfrzcpi_[h]; ++I) {
    //                int nI = I + offset;
    //                for (int i = nfrzcpi_[h]; i < hf_ndoccpi_[h]; ++i) {
    //                    int ni = i + offset;

    //                    double z = Z_->get(h, I, i);

    //                    if (nj == ni) {
    //                        d2.write_value(nI, ni, ni, ni, z, print, name, 0);
    //                        d2.write_value(ni, nI, ni, ni, 2.0 * z, print, name, 0);
    //                        d2.write_value(ni, ni, ni, nI, -z, print, name, 0);
    //                    } else if (nj == nI) {
    //                        d2.write_value(ni, nI, nI, nI, z, print, name, 0);
    //                        d2.write_value(nI, ni, nI, nI, 2.0 * z, print, name, 0);
    //                        d2.write_value(nI, nI, nI, ni, -z, print, name, 0);
    //                    } else {
    //                        d2.write_value(nI, ni, nj, nj, 2.0 * z, print, name, 0);
    //                        d2.write_value(nI, nj, nj, ni, -z, print, name, 0);

    //                        d2.write_value(ni, nI, nj, nj, 2.0 * z, print, name, 0);
    //                        d2.write_value(ni, nj, nj, nI, -z, print, name, 0);
    //                    }
    //                }
    //            }

    //            // frozen-virtual/active-virtual
    //            for (int A = frzv_start_pi[h]; A < nmopi_[h]; ++A) {
    //                int nA = A + offset;
    //                for (int a = hf_ndoccpi_[h]; a < frzv_start_pi[h]; ++a) {
    //                    int na = a + offset;

    //                    double z = Z_->get(h, A, a);

    //                    d2.write_value(nA, na, nj, nj, 2.0 * z, print, name, 0);
    //                    d2.write_value(nA, nj, nj, na, -z, print, name, 0);

    //                    d2.write_value(na, nA, nj, nj, 2.0 * z, print, name, 0);
    //                    d2.write_value(na, nj, nj, nA, -z, print, name, 0);
    //                }
    //            }

    //            // docc/virtual
    //            for (int i = 0; i < hf_ndoccpi_[h]; ++i) {
    //                int ni = i + offset;
    //                for (int a = hf_ndoccpi_[h]; a < nmopi_[h]; ++a) {
    //                    int na = a + offset;

    //                    double z = Z_->get(h, i, a);

    //                    if (nj == ni) {
    //                        d2.write_value(na, ni, ni, ni, z, print, name, 0);
    //                        d2.write_value(ni, na, ni, ni, 2.0 * z, print, name, 0);
    //                        d2.write_value(ni, ni, ni, na, -z, print, name, 0);
    //                    } else {
    //                        d2.write_value(ni, na, nj, nj, 2.0 * z, print, name, 0);
    //                        d2.write_value(ni, nj, nj, na, -z, print, name, 0);

    //                        d2.write_value(na, ni, nj, nj, 2.0 * z, print, name, 0);
    //                        d2.write_value(na, nj, nj, ni, -z, print, name, 0);
    //                    }
    //                }
    //            }
    //        }
    //        offset += nmopi_[h];
    //    }

    d2.flush(1);
    d2.set_keep_flag(1);
    d2.close();
}
} // namespace forte
