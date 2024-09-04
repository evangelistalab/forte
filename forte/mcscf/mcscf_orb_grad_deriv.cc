/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2024 by its authors (see COPYING, COPYING.LESSER, AUTHORS).
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

#include <numeric>

#include "psi4/libqt/qt.h"
#include "psi4/lib3index/3index.h"
#include "psi4/libiwl/iwl.hpp"
#include "psi4/libmints/wavefunction.h"
#include "psi4/libpsi4util/PsiOutStream.h"
#include "psi4/psifiles.h"

#include "helpers/helpers.h"
#include "helpers/printing.h"
#include "helpers/lbfgs/lbfgs.h"

#include "gradient_tpdm/backtransform_tpdm.h"
#include "mcscf/mcscf_orb_grad.h"
#include "mcscf/cpscf.h"

using namespace psi;
using namespace ambit;

namespace forte {

void MCSCF_ORB_GRAD::compute_nuclear_gradient() {
    print_h2("MCSCF Gradient");

    // format A to SharedMatrix
    Am_ = std::make_shared<psi::Matrix>("A (MCSCF)", nmopi_, nmopi_);
    fill_A_matrix_data(A_);

    is_frozen_orbs_ = nfrzc_ or label_to_mos_["u"].size();

    if (is_frozen_orbs_) {
        // see J. Chem. Phys. 94, 6708-6715 (1991)

        // setup MOs and sanity check
        setup_grad_frozen();

        // compute frozen part of the A matrix
        build_Am_frozen();

        // transform A matrix to HF basis
        Am_ = psi::linalg::triplet(U_, Am_, U_, false, false, true);
        Am_->set_name("A (SCF)");

        // solve CPSCF equations
        solve_cpscf();
    }

    print_h2("Prepare AO Densities");

    // back-transform Lagrangian
    compute_Lagrangian();

    // back-transform 1-RDM
    compute_opdm_ao();

    if (tei_alg_ == TEIALG::DF) {
        dump_tpdm_df();
    } else {
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
    psi::outfile->Printf("\n\n");
}

void MCSCF_ORB_GRAD::JK_build(std::shared_ptr<psi::Matrix> Cl, std::shared_ptr<psi::Matrix> Cr) {
    /* JK build for Fock-like term
     * J: sum_{rs} D_{rs} (rs|PQ) = sum_{rsRS} D_{rs} C_{Rr} C_{Ss} (RS|PQ)
     * K: sum_{rs} D_{rs} (rQ|Ps) = sum_{rsRS} D_{rs} C_{Rr} C_{Ss} (RQ|PS)
     *
     * Cl_{Rs} = sum_{r} D_{rs} C_{Rr}
     * Cr_{Ss} = C_{Ss}
     *
     * r,s: MO indices; P,Q,R,S: AO indices; C: MO coefficients; D: any matrix
     */
    JK_->set_do_K(true);
    std::vector<std::shared_ptr<psi::Matrix>>& Cls = JK_->C_left();
    std::vector<std::shared_ptr<psi::Matrix>>& Crs = JK_->C_right();
    Cls.clear();
    Crs.clear();
    Cls.push_back(Cl);
    Crs.push_back(Cr);
    JK_->compute();
}

std::shared_ptr<psi::Matrix> MCSCF_ORB_GRAD::C_subset(const std::string& name,
                                                      std::shared_ptr<psi::Matrix> C,
                                                      psi::Dimension dim_start,
                                                      psi::Dimension dim_end) {
    auto dim = dim_end - dim_start;
    auto Csub = std::make_shared<psi::Matrix>(name, nsopi_, dim);

    for (int h = 0; h < nirrep_; ++h) {
        for (int p = 0, offset = dim_start[h]; p < dim[h]; ++p) {
            Csub->set_column(h, p, C->get_column(h, p + offset));
        }
    }

    return Csub;
}

void MCSCF_ORB_GRAD::fill_A_matrix_data(ambit::BlockedTensor A) {
    A.citerate(
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

void MCSCF_ORB_GRAD::setup_grad_frozen() {
    // terminate for high spin because ROHF CP-SCF is not implemented
    if (ints_->wfn()->soccpi().sum())
        throw std::runtime_error("MCSCF gradient with frozen orbitals only works for singlet!");

    // set up HF orbitals
    hf_ndoccpi_ = ints_->wfn()->doccpi();
    hf_nuoccpi_ = nmopi_ - ints_->wfn()->doccpi();

    std::vector<size_t> adocc_mos, auocc_mos;
    hf_docc_mos_.clear();
    hf_uocc_mos_.clear();
    for (int h = 0, offset = 0; h < nirrep_; ++h) {
        for (int i = 0; i < hf_ndoccpi_[h]; ++i) {
            hf_docc_mos_.push_back(i + offset);
            if (i >= nfrzcpi_[h])
                adocc_mos.push_back(i + offset);
        }
        for (int a = 0; a < hf_nuoccpi_[h]; ++a) {
            hf_uocc_mos_.push_back(a + hf_ndoccpi_[h] + offset);
            if (a < (hf_nuoccpi_[h] - nfrzvpi_[h]))
                auocc_mos.push_back(a + hf_ndoccpi_[h] + offset);
        }
        offset += nmopi_[h];
    }
    label_to_mos_["m"] = adocc_mos;
    label_to_mos_["e"] = auocc_mos;

    BlockedTensor::add_mo_space("m", "m,n", adocc_mos, NoSpin);
    BlockedTensor::add_mo_space("e", "e,f", auocc_mos, NoSpin);
    BlockedTensor::add_composite_mo_space("O", "k,l", {"f", "m"});
    BlockedTensor::add_composite_mo_space("V", "c,d", {"e", "u"});
    BlockedTensor::add_composite_mo_space("H", "g,h", {"f", "m", "e", "u"});

    // build HF Fock matrix, assuming MCSCF initial orbitals are from HF
    auto Cdocc = C_subset("C_DOCC", C0_, psi::Dimension(nirrep_), hf_ndoccpi_);

    JK_build(Cdocc, Cdocc);
    auto J = JK_->J()[0];
    J->scale(2.0);
    J->subtract(JK_->K()[0]);
    J->add(ints_->wfn()->H());

    auto F = psi::linalg::triplet(C0_, J, C0_, true, false, false);
    F->set_name("Fock MO (SCF)");

    // grab HF orbital energy
    epsilon_ = std::make_shared<psi::Vector>("Orbital Energies (SCF)", nmopi_);

    for (int h = 0; h < nirrep_; ++h) {
        for (int p = 0; p < nmopi_[h]; ++p) {
            double value = F->get(h, p, p);
            epsilon_->set(h, p, value);
            F->set(h, p, p, 0.0);
        }
    }

    // test off-diagonal elements of Fock matrix
    if (F->rms() > 5.0 * options_->get_double("D_CONVERGENCE")) {
        outfile->Printf("\n  Error: MCSCF initial orbitals are not from Hartree-Fock!");
        outfile->Printf("\n  MCSCF gradient with frozen orbitals will not work.");
        throw std::runtime_error("MCSCF grad error: initial orbitals NOT from Hartree-Fock!");
    }
}

void MCSCF_ORB_GRAD::build_Am_frozen() {
    // fill in A_{ri}
    for (int h = 0; h < nirrep_; ++h) {
        for (int I = 0; I < nfrzcpi_[h]; ++I) {
            for (int p = 0; p < nmopi_[h]; ++p) {
                Am_->set(h, p, I, 2.0 * Fock_->get(h, p, I));
            }
        }
        for (int k = nfrzcpi_[h]; k < ndoccpi_[h]; ++k) {
            for (int I = 0; I < nfrzcpi_[h]; ++I) {
                Am_->set(h, I, k, 2.0 * Fock_->get(h, I, k));
            }
            for (int B = nmopi_[h] - nfrzvpi_[h]; B < nmopi_[h]; ++B) {
                Am_->set(h, B, k, 2.0 * Fock_->get(h, B, k));
            }
        }
    }

    // compute A_{ru}
    auto At = ambit::BlockedTensor::build(CoreTensor, "A", {"Fa"});
    At["Mu"] = Fc_["Mt"] * D1_["tu"];
    At["Mu"] += V_["Mtvw"] * D2_["tuvw"];

    fill_A_matrix_data(At);
}

void MCSCF_ORB_GRAD::solve_cpscf() {
    // compute orbital gradient in Hartree-Fock basis
    auto G = Am_->clone();
    G->subtract(Am_->transpose());
    G->set_name("A - A^T");

    Z_ = std::make_shared<psi::Matrix>("Z", nmopi_, nmopi_);

    // occupied-occupied part of Z
    // only set frozen-docc/active-docc part, the rest is treated as zero due to MCSCF
    for (int h = 0; h < nirrep_; ++h) {
        for (int i = 0; i < nfrzcpi_[h]; ++i) {
            for (int j = nfrzcpi_[h]; j < hf_ndoccpi_[h]; ++j) {
                double value = G->get(h, i, j) / (epsilon_->get(h, j) - epsilon_->get(h, i));
                Z_->set(h, i, j, value);
                Z_->set(h, j, i, value);
            }
        }
    }

    // virtual-virtual part of Z
    // only set frozen-uocc/active-uocc part, the rest is treated as zero due to MCSCF
    for (int h = 0; h < nirrep_; ++h) {
        int frzv_start = nmopi_[h] - nfrzvpi_[h];
        for (int a = frzv_start; a < nmopi_[h]; ++a) {
            for (int b = hf_ndoccpi_[h]; b < frzv_start; ++b) {
                double value = G->get(h, b, a) / (epsilon_->get(h, a) - epsilon_->get(h, b));
                Z_->set(h, a, b, value);
                Z_->set(h, b, a, value);
            }
        }
    }

    // prepare r.h.s. of CPSCF equation (Z_{ai} independent part)
    // b_{ai} = G_{ia} - sum_{Ik} Z_{Ik} L_{Ik,ia} - sum_{Ac} Z_{Ac} L_{Ac,ia}
    // i: DOCC; I: frozen core; k: active docc; a: UOCC; A: frozen virutal; c: active virtual
    auto b = std::make_shared<psi::Matrix>("CPSCF b", hf_nuoccpi_, hf_ndoccpi_);

    // grab G_{ia}
    Slice Sdocc(psi::Dimension(nirrep_), hf_ndoccpi_);
    Slice Suocc(hf_ndoccpi_, nmopi_);
    b->copy(G->get_block(Sdocc, Suocc)->transpose());

    // grab part of orbital coefficients
    auto dim0 = psi::Dimension(nirrep_);
    auto dim1 = nmopi_ - nfrzvpi_;
    auto Cdocc = C_subset("C_DOCC", C0_, dim0, hf_ndoccpi_);
    auto Cfrzc = C_subset("C_FRZC", C0_, dim0, nfrzcpi_);
    auto Cactc = C_subset("C_ACTC", C0_, nfrzcpi_, hf_ndoccpi_);
    auto Cuocc = C_subset("C_UOCC", C0_, hf_ndoccpi_, nmopi_);
    auto Cactv = C_subset("C_ACTV", C0_, hf_ndoccpi_, dim1);
    auto Cfrzv = C_subset("C_FRZV", C0_, dim1, nmopi_);

    // contract Z_{Ik} with supermatrix L_{Ik,ia}
    auto Z_Ik = Z_->get_block(Slice(dim0, nfrzcpi_), Slice(nfrzcpi_, hf_ndoccpi_));
    b->subtract(contract_RB_Z(Z_Ik, Cfrzc, Cactc, Cuocc, Cdocc));

    // contract Z_{Ac} with supermatrix L_{Ac,ia}
    auto Z_Ac = Z_->get_block(Slice(dim1, nmopi_), Slice(hf_ndoccpi_, dim1));
    b->subtract(contract_RB_Z(Z_Ac, Cfrzv, Cactv, Cuocc, Cdocc));

    if (debug_print_) {
        G->print();
        b->print();
    }

    // grab occupied and virtual part of orbital energies
    auto edocc = std::make_shared<Vector>(epsilon_->get_block(Sdocc));
    auto euocc = std::make_shared<Vector>(epsilon_->get_block(Suocc));

    // solve CPSCF equation
    CPSCF_SOLVER cpscf_solver(options_, JK_, C0_, b, edocc, euocc);
    bool converged = cpscf_solver.solve();
    auto Z_ai = cpscf_solver.x();

    // copy results to Z
    Z_->set_block(Suocc, Sdocc, Z_ai);
    Z_->set_block(Sdocc, Suocc, Z_ai->transpose());

    if (not converged) {
        auto msg = "CP-SCF did not converge!";
        outfile->Printf("\n  Error: %s", msg);
        outfile->Printf("\n  Please check if the specified frozen orbitals make sense.");
        throw std::runtime_error(msg);
    }

    if (debug_print_)
        Z_->print();
}

SharedMatrix MCSCF_ORB_GRAD::contract_RB_Z(std::shared_ptr<psi::Matrix> Z,
                                           std::shared_ptr<psi::Matrix> C_Zrow,
                                           std::shared_ptr<psi::Matrix> C_Zcol,
                                           std::shared_ptr<psi::Matrix> C_row,
                                           std::shared_ptr<psi::Matrix> C_col) {
    // compute sum_{pq} Z_{pq} L_{pq,rs} using Hartree-Fock orbitals
    // Roothaan-Bagus supermatrix L_{pq,rs} = 4 * (pq|rs) - (pr|sq) - (ps|rq)
    // Express contraction in AO basis:
    // sum_{pq} sum_{PQRS} CZrow_{Pp} Z_{pq} CZcol_{Qq} L_{PQ,RS} Crow_{Rr} Ccol_{Ss}

    // C dressed by Z_pq
    auto Cdressed = psi::linalg::doublet(C_Zrow, Z, false, false);

    // JK build
    JK_build(Cdressed, C_Zcol);

    auto J = JK_->J()[0];
    J->scale(4.0);
    J->subtract(JK_->K()[0]);
    J->subtract((JK_->K()[0])->transpose());

    // transform to MO and return
    return psi::linalg::triplet(C_row, J, C_col, true, false, false);
}

void MCSCF_ORB_GRAD::compute_Lagrangian() {
    psi::outfile->Printf("\n    Computing AO Lagrangian ......");
    auto W = std::make_shared<psi::Matrix>();

    if (not is_frozen_orbs_) {
        // The A matrix is equivalent to W when there are no frozen orbitals
        // We can directly back transform the A matrix
        W = psi::linalg::triplet(C_, Am_, C_, false, false, true);
    } else {
        // If there are frozen orbitals, we have solved the CP-SCF equation in HF basis
        auto Wmo = std::make_shared<psi::Matrix>("Lagrangian MO (SCF)", nmopi_, nmopi_);

        // occupied-occupied part
        auto Cdocc = C_subset("C_DOCC", C0_, psi::Dimension(nirrep_), hf_ndoccpi_);
        auto L = contract_RB_Z(Z_, C0_, C0_, Cdocc, Cdocc);
        for (int h = 0; h < nirrep_; ++h) {
            for (int i = 0; i < hf_ndoccpi_[h]; ++i) {
                for (int j = i; j < hf_ndoccpi_[h]; ++j) {
                    double value = Am_->get(h, i, j) + Am_->get(h, j, i);
                    value += Z_->get(h, i, j) * (epsilon_->get(h, i) + epsilon_->get(h, j));
                    value += 0.5 * (L->get(h, i, j) + L->get(h, j, i));
                    Wmo->set(h, i, j, 0.5 * value);
                    Wmo->set(h, j, i, 0.5 * value);
                }
            }
        }

        // virtual-virtual part
        for (int h = 0; h < nirrep_; ++h) {
            for (int a = hf_ndoccpi_[h]; a < nmopi_[h]; ++a) {
                for (int b = a; b < nmopi_[h]; ++b) {
                    double value = Am_->get(h, a, b) + Am_->get(h, b, a);
                    value += Z_->get(h, a, b) * (epsilon_->get(h, a) + epsilon_->get(h, b));
                    Wmo->set(h, a, b, 0.5 * value);
                    Wmo->set(h, b, a, 0.5 * value);
                }
            }
        }

        // occupied-virtual part
        for (int h = 0; h < nirrep_; ++h) {
            for (int i = 0; i < hf_ndoccpi_[h]; ++i) {
                for (int a = hf_ndoccpi_[h]; a < nmopi_[h]; ++a) {
                    double value = Am_->get(h, i, a);
                    value += Z_->get(h, i, a) * epsilon_->get(h, i);
                    Wmo->set(h, i, a, value);
                    Wmo->set(h, a, i, value);
                }
            }
        }

        if (debug_print_)
            Wmo->print();

        // transform to AO
        W = psi::linalg::triplet(C0_, Wmo, C0_, false, false, true);
    }

    W->set_name("Lagrangian AO Back-Transformed");
    if (debug_print_)
        W->print();

    // transform to AO and push to Psi4 Wavefunction
    ints_->wfn()->set_lagrangian(W);
    psi::outfile->Printf(" Done.");
}

void MCSCF_ORB_GRAD::compute_opdm_ao() {
    psi::outfile->Printf("\n    Computing AO OPDM ............");
    auto D1a = std::make_shared<psi::Matrix>("D1a", nmopi_, nmopi_);

    // inactive docc part
    for (int h = 0; h < nirrep_; ++h) {
        for (int i = 0; i < ndoccpi_[h]; ++i) {
            D1a->set(h, i, i, 1.0);
        }
    }

    // active part
    for (int h = 0; h < nirrep_; ++h) {
        auto offset = ndoccpi_[h];
        for (int u = 0; u < nactvpi_[h]; ++u) {
            auto nu = u + offset;
            for (int v = u; v < nactvpi_[h]; ++v) {
                auto nv = v + offset;
                D1a->set(h, nu, nv, 0.5 * rdm1_->get(h, u, v));
                D1a->set(h, nv, nu, 0.5 * rdm1_->get(h, v, u));
            }
        }
    }

    D1a = psi::linalg::triplet(C_, D1a, C_, false, false, true);
    D1a->set_name("D1a AO Back-Transformed");

    // add Z vector due to response of frozen orbitals
    if (is_frozen_orbs_) {
        auto T = psi::linalg::triplet(C0_, Z_, C0_, false, false, true);
        T->scale(0.5);
        D1a->add(T);
    }

    // push to Psi4
    ints_->wfn()->Da()->copy(D1a);
    ints_->wfn()->Db()->copy(D1a);
    psi::outfile->Printf(" Done.");
}

void MCSCF_ORB_GRAD::dump_tpdm_df() {
    psi::outfile->Printf("\n    Dumping DF AO TPDM to disk ...");
    auto naux = df_helper_->get_naux();

    // grab (P|Q)^{-1/2}
    auto bs_aux = ints_->wfn()->get_basisset("DF_BASIS_SCF");
    auto metric = std::make_shared<FittingMetric>(bs_aux, true);
    metric->form_eig_inverse(options_->get_double("DF_FITTING_CONDITION"));
    auto Jm12 = metric->get_metric();

    // DFHelper to get B^{P}_{pq} = (P|Q)^{-1/2} (Q|pq) integrals in MO basis
    auto C_nosym = ints_->Ca_SO2AO(C_);

    auto Cactv = std::make_shared<psi::Matrix>("Cactv", nso_, nactv_);
    for (size_t x = 0; x < nactv_; ++x) {
        Cactv->set_column(0, x, C_nosym->get_column(0, actv_mos_[x]));
    }

    auto docc_mos = mo_space_info_->absolute_mo("INACTIVE_DOCC");
    auto ndocc = docc_mos.size();
    auto Cdocc = std::make_shared<psi::Matrix>("Cdocc", nso_, ndocc);
    for (size_t i = 0; i < ndocc; ++i) {
        Cdocc->set_column(0, i, C_nosym->get_column(0, docc_mos[i]));
    }

    df_helper_->add_space("DOCC", Cdocc);
    df_helper_->add_space("ACTV", Cactv);
    df_helper_->add_transformation("A", "DOCC", "DOCC", "Qpq");
    df_helper_->add_transformation("B", "DOCC", "ACTV", "Qpq");
    df_helper_->add_transformation("C", "ACTV", "ACTV", "Qpq");
    df_helper_->transform();

    // compute C^{P}_{pq} = (P|Q)^{-1/2} B^{Q}_{pq} integrals in ambit form
    // notes on C_DGEMM
    // LDA = transA ? rowA : colA; LDB = transB ? rowB : colB; LDC = colC;
    auto A = df_helper_->get_tensor("A");
    auto C3oo = ambit::Tensor::build(CoreTensor, "C3oo", {naux, ndocc, ndocc});
    C_DGEMM('N', 'N', naux, ndocc * ndocc, naux, 1.0, Jm12->get_pointer(), naux, A->get_pointer(),
            ndocc * ndocc, 0.0, C3oo.data().data(), ndocc * ndocc);

    A = df_helper_->get_tensor("B");
    auto C3oa = ambit::Tensor::build(CoreTensor, "C3oa", {naux, ndocc, nactv_});
    C_DGEMM('N', 'N', naux, ndocc * nactv_, naux, 1.0, Jm12->get_pointer(), naux, A->get_pointer(),
            ndocc * nactv_, 0.0, C3oa.data().data(), ndocc * nactv_);

    A = df_helper_->get_tensor("C");
    auto C3aa = ambit::Tensor::build(CoreTensor, "C3aa", {naux, nactv_, nactv_});
    C_DGEMM('N', 'N', naux, nactv_ * nactv_, naux, 1.0, Jm12->get_pointer(), naux, A->get_pointer(),
            nactv_ * nactv_, 0.0, C3aa.data().data(), nactv_ * nactv_);

    df_helper_->clear_all();

    auto Ioo = ambit::Tensor::build(CoreTensor, "Ioo", {ndocc, ndocc});
    for (size_t i = 0; i < ndocc; ++i) {
        Ioo.data()[i * ndocc + i] = 1.0;
    }

    // symmetrized 3-index 2-RDM [(pq|Q) contribution]
    auto D3oo = ambit::Tensor::build(CoreTensor, "D3oo", {naux, ndocc, ndocc});
    D3oo("Qmn") += 2.0 * C3oo("Qij") * Ioo("ij") * Ioo("mn");
    D3oo("Qmn") -= C3oo("Qnm");
    D3oo("Qmn") += C3aa("Quv") * D1_.block("aa")("uv") * Ioo("mn");

    auto D3oa = ambit::Tensor::build(CoreTensor, "D3oa", {naux, ndocc, nactv_});
    D3oa("Qmu") -= 0.5 * C3oa("Qmv") * D1_.block("aa")("uv");

    auto D3aa = ambit::Tensor::build(CoreTensor, "D3aa", {naux, nactv_, nactv_});
    D3aa("Quv") += C3oo("Qmn") * Ioo("mn") * D1_.block("aa")("uv");
    D3aa("Quv") += 0.5 * C3aa("Qxy") * D2_.block("aaaa")("uvxy");

    auto d3 = std::make_shared<psi::Matrix>("d3xss", naux, nso_ * nso_);
    auto d3_so = std::make_shared<psi::Matrix>("d3_so", nso_, nso_);
    auto oo = std::make_shared<psi::Matrix>("d3_mo oo", ndocc, ndocc);
    auto oo_half = std::make_shared<psi::Matrix>("oo half trans", nso_, ndocc);
    auto oa = std::make_shared<psi::Matrix>("d3_mo oa", ndocc, nactv_);
    auto oa_half = std::make_shared<psi::Matrix>("oa half trans", nso_, nactv_);
    auto aa = std::make_shared<psi::Matrix>("d3_mo aa", nactv_, nactv_);
    auto aa_half = std::make_shared<psi::Matrix>("aa half trans", nso_, nactv_);

    // back-transform 3-index 2-RDM to AO basis
    for (size_t Q = 0; Q < naux; ++Q) {
        d3_so->zero();

        // oo block
        C_DCOPY(ndocc * ndocc, D3oo.data().data() + Q * ndocc * ndocc, 1, oo->get_pointer(), 1);
        C_DGEMM('N', 'N', nso_, ndocc, ndocc, 1.0, Cdocc->get_pointer(), ndocc, oo->get_pointer(),
                ndocc, 0.0, oo_half->get_pointer(), ndocc);
        C_DGEMM('N', 'T', nso_, nso_, ndocc, 1.0, oo_half->get_pointer(), ndocc,
                Cdocc->get_pointer(), ndocc, 1.0, d3_so->get_pointer(), nso_);

        // oa block
        C_DCOPY(ndocc * nactv_, D3oa.data().data() + Q * ndocc * nactv_, 1, oa->get_pointer(), 1);
        C_DGEMM('N', 'N', nso_, nactv_, ndocc, 1.0, Cdocc->get_pointer(), ndocc, oa->get_pointer(),
                nactv_, 0.0, oa_half->get_pointer(), nactv_);
        C_DGEMM('N', 'T', nso_, nso_, nactv_, 1.0, oa_half->get_pointer(), nactv_,
                Cactv->get_pointer(), nactv_, 1.0, d3_so->get_pointer(), nso_);

        // ao block
        matrix_transpose_in_place(oa_half->get_pointer(), nso_, nactv_);
        // oa_half is now nactv_ x nso_ after transposition
        C_DGEMM('N', 'N', nso_, nso_, nactv_, 1.0, Cactv->get_pointer(), nactv_,
                oa_half->get_pointer(), nso_, 1.0, d3_so->get_pointer(), nso_);

        // aa block
        C_DCOPY(nactv_ * nactv_, D3aa.data().data() + Q * nactv_ * nactv_, 1, aa->get_pointer(), 1);
        C_DGEMM('N', 'N', nso_, nactv_, nactv_, 1.0, Cactv->get_pointer(), nactv_,
                aa->get_pointer(), nactv_, 0.0, aa_half->get_pointer(), nactv_);
        C_DGEMM('N', 'T', nso_, nso_, nactv_, 1.0, aa_half->get_pointer(), nactv_,
                Cactv->get_pointer(), nactv_, 1.0, d3_so->get_pointer(), nso_);

        C_DCOPY(nso_ * nso_, d3_so->get_pointer(), 1, d3->get_pointer() + Q * nso_ * nso_, 1);
    }

    // 2-index 2-RDM
    auto D2xx = ambit::Tensor::build(CoreTensor, "D2xx", {naux, naux});
    D2xx("RQ") += C3oo("Rmn") * D3oo("Qmn");
    D2xx("RQ") += 2.0 * C3oa("Rmu") * D3oa("Qmu");
    D2xx("RQ") += C3aa("Ruv") * D3aa("Quv");

    auto d2 = std::make_shared<psi::Matrix>("d2xx", naux, naux);
    C_DCOPY(naux * naux, D2xx.data().data(), 1, d2->get_pointer(), 1);

    // contributions from frozen orbitals
    if (is_frozen_orbs_)
        dump_tpdm_df_hf(Jm12, d3, d2);

    // dump to psio
    d3->scale(2.0); // consider (Q|pq) contribution
    d3->set_name("3-Center Reference Density");
    d3->save(_default_psio_lib_, PSIF_AO_TPDM, Matrix::SaveType::ThreeIndexLowerTriangle);
    d3->zero();
    d3->set_name("3-Center Correlation Density");
    d3->save(_default_psio_lib_, PSIF_AO_TPDM, Matrix::SaveType::ThreeIndexLowerTriangle);

    d2->set_name("Metric Reference Density");
    d2->save(_default_psio_lib_, PSIF_AO_TPDM, Matrix::SaveType::LowerTriangle);
    d2->zero();
    d2->set_name("Metric Correlation Density");
    d2->save(_default_psio_lib_, PSIF_AO_TPDM, Matrix::SaveType::LowerTriangle);

    psi::outfile->Printf(" Done.");
}

void MCSCF_ORB_GRAD::dump_tpdm_df_hf(std::shared_ptr<psi::Matrix> Jm12,
                                     std::shared_ptr<psi::Matrix> d3,
                                     std::shared_ptr<psi::Matrix> d2) {
    /**
     * Implementation notes
     * contribution to total Lagrangian: z_{pq} * [2 * (pq|ii) - (pi|iq)]
     * D3^{Q}_{ii} <- z_{pq} * C3^{Q}_{pq}           term 1
     * D3^{Q}_{pq} <- z_{pq} * C3^{Q}_{ii}           term 2
     * D3^{Q}_{pi} <- -0.5 * z_{pq} * C3^{Q}_{iq}    term 3
     * D3^{Q}_{iq} <- -0.5 * z_{pq} * C3^{Q}_{pi}    term 4
     * will be scaled by 2 at the end of dump_tpdm_df()
     *
     * D2^{PQ} = z_{pq} * ( 2 * C^{P}_{pq} * C^{Q}_{ii} - C^{P}_{pi} * C^{Q}_{iq} )
     */
    auto naux = df_helper_->get_naux();
    std::vector<size_t> aux_mos(naux);
    std::iota(aux_mos.begin(), aux_mos.end(), 0);
    BlockedTensor::add_mo_space("L", "K,L", aux_mos, NoSpin);

    auto nf = label_to_mos_["f"].size();
    auto nm = label_to_mos_["m"].size();
    auto nso2 = nso_ * nso_;

    // HF orbital coefficients without symmetry
    auto C_nosym = ints_->Ca_SO2AO(C0_);
    auto Csub = [&](const std::vector<size_t>& p_mos) {
        auto np = p_mos.size();
        auto Cp = std::make_shared<psi::Matrix>("Csub", nso_, np);
        for (size_t p = 0; p < np; ++p) {
            Cp->set_column(0, p, C_nosym->get_column(0, p_mos[p]));
        }
        return Cp;
    };

    ///////////////////////////////////////////////////////////////

    auto nd = hf_docc_mos_.size();
    auto no = nd - nfrzc_;
    auto Zt = ambit::Tensor::build(CoreTensor, "Zt", {nmo_, nmo_});
    Zt.iterate([&](const std::vector<size_t>& i, double& value) {
        auto [hp, ip] = mos_rel_[i[0]];
        auto [hq, iq] = mos_rel_[i[1]];
        if (hp == hq) {
            value = Z_->get(hp, ip, iq);
        }
    });

    auto It = ambit::Tensor::build(CoreTensor, "I", {nd, nd});
    It.iterate([&](const std::vector<size_t>& i, double& value) {
        if (i[0] == i[1])
            value = 1.0;
    });

    auto Cdocc = Csub(hf_docc_mos_);
    df_helper_->add_space("ALL", C_nosym);
    df_helper_->add_space("DOCC", Cdocc);
    df_helper_->add_transformation("A", "ALL", "ALL", "Qpq");
    df_helper_->add_transformation("B", "DOCC", "ALL", "Qpq");
    df_helper_->add_transformation("C", "DOCC", "DOCC", "Qpq");
    df_helper_->transform();

    auto A = df_helper_->get_tensor("A");
    auto C3pq = ambit::Tensor::build(CoreTensor, "C3pq", {naux, nmo_, nmo_});
    C_DGEMM('N', 'N', naux, nmo_ * nmo_, naux, 1.0, Jm12->get_pointer(), naux, A->get_pointer(),
            nmo_ * nmo_, 0.0, C3pq.data().data(), nmo_ * nmo_);

    A = df_helper_->get_tensor("B");
    auto C3ip = ambit::Tensor::build(CoreTensor, "C3ip", {naux, nd, nmo_});
    C_DGEMM('N', 'N', naux, nd * nmo_, naux, 1.0, Jm12->get_pointer(), naux, A->get_pointer(),
            nd * nmo_, 0.0, C3ip.data().data(), nd * nmo_);

    A = df_helper_->get_tensor("C");
    auto C3ij = ambit::Tensor::build(CoreTensor, "C3ij", {naux, nd, nd});
    C_DGEMM('N', 'N', naux, nd * nd, naux, 1.0, Jm12->get_pointer(), naux, A->get_pointer(),
            nd * nd, 0.0, C3ij.data().data(), nd * nd);

    df_helper_->clear_all();

    auto D3pq = ambit::Tensor::build(CoreTensor, "D3pq", {naux, nmo_, nmo_});
    auto D3ip = ambit::Tensor::build(CoreTensor, "D3ip", {naux, nd, nmo_});
    auto D3pi = ambit::Tensor::build(CoreTensor, "D3pi", {naux, nmo_, nd});
    auto D3ij = ambit::Tensor::build(CoreTensor, "D3ij", {naux, nd, nd});
    D3ij("Qij") += C3pq("Qpq") * Zt("pq") * It("ij");
    D3pq("Qpq") += Zt("pq") * C3ij("Qij") * It("ij");
    D3ip("Qiq") -= 0.5 * C3ip("Qip") * Zt("pq");
    D3pi("Qpi") -= 0.5 * C3ip("Qiq") * Zt("pq");

    auto D3pq_copy = D3pq.clone();
    for (size_t Q = 0; Q < naux; ++Q) {
        for (size_t i = 0; i < nd; ++i) {
            size_t ni = hf_docc_mos_[i];
            for (size_t j = 0; j < nd; ++j) {
                size_t nj = hf_docc_mos_[j];
                D3pq_copy.data()[Q * nmo_ * nmo_ + ni * nmo_ + nj] +=
                    D3ij.data()[Q * nd * nd + i * nd + j];
            }
            for (size_t p = 0; p < nmo_; ++p) {
                D3pq_copy.data()[Q * nmo_ * nmo_ + ni * nmo_ + p] +=
                    D3ip.data()[Q * nd * nmo_ + i * nmo_ + p];
                D3pq_copy.data()[Q * nmo_ * nmo_ + p * nmo_ + ni] +=
                    D3pi.data()[Q * nd * nmo_ + p * nd + i];
            }
        }
    }
    // D3pq_copy.print();

    auto D2RS_tmp = ambit::Tensor::build(CoreTensor, "D2RS tmp", {naux, naux});
    D2RS_tmp("RS") += 2.0 * Zt("pq") * C3pq("Rpq") * C3ij("Sij") * It("ij");
    D2RS_tmp("RS") -= Zt("pq") * C3ip("Rip") * C3ip("Siq");
    auto D2RS = ambit::Tensor::build(CoreTensor, "D2RS", {naux, naux});
    D2RS("RS") += 0.5 * D2RS_tmp("RS");
    D2RS("RS") += 0.5 * D2RS_tmp("SR");

    // D2RS("RS") = D2RS_tmp("RS");

    // D2RS.print();

    auto Ct = ambit::Tensor::build(CoreTensor, "Ct", {nso_, nmo_});
    Ct.iterate(
        [&](const std::vector<size_t>& i, double& value) { value = C_nosym->get(i[0], i[1]); });
    auto Cd = ambit::Tensor::build(CoreTensor, "Ct", {nso_, nd});
    Cd.iterate(
        [&](const std::vector<size_t>& i, double& value) { value = Cdocc->get(i[0], i[1]); });

    auto D3ao = ambit::Tensor::build(CoreTensor, "D3ao", {naux, nso_, nso_});
    D3ao("QMN") += D3ij("Qij") * Cd("Mi") * Cd("Nj");
    D3ao("QMN") += D3pq("Qpq") * Ct("Mp") * Ct("Nq");
    D3ao("QMN") += D3ip("Qip") * Cd("Mi") * Ct("Np");
    D3ao("QMN") += D3ip("Qip") * Cd("Ni") * Ct("Mp");

    auto D3ao_copy = D3ao.clone();
    D3ao_copy("QMN") = D3pq_copy("Qpq") * Ct("Mp") * Ct("Nq");
    D3ao_copy("QMN") -= D3ao("QMN");
    psi::outfile->Printf("\n QMN DIFF = %20.15f", D3ao_copy.norm());

    // D3ao.print();

    auto d2tmp = std::make_shared<psi::Matrix>("d2tmp", naux, naux);
    C_DCOPY(naux * naux, D2RS.data().data(), 1, d2tmp->get_pointer(), 1);
    auto d3tmp = std::make_shared<psi::Matrix>("d3tmp", naux, nso_ * nso_);
    C_DCOPY(naux * nso_ * nso_, D3ao.data().data(), 1, d3tmp->get_pointer(), 1);

    // d2->add(d2tmp);
    // d3->add(d3tmp);

    ///////////////////////////////////////////////////////////////

    // Z matrix without symmetry
    auto Z = ambit::BlockedTensor::build(CoreTensor, "Z", {"fm", "OV", "eu"});
    for (const std::string& block : Z.block_labels()) {
        const auto& p_mos = label_to_mos_[block.substr(0, 1)];
        const auto& q_mos = label_to_mos_[block.substr(1, 1)];
        Z.block(block).iterate([&](const std::vector<size_t>& i, double& value) {
            auto [hp, ip] = mos_rel_[p_mos[i[0]]];
            auto [hq, iq] = mos_rel_[q_mos[i[1]]];
            if (hp == hq)
                value = Z_->get(hp, ip, iq);
        });
    }
    // auto Zsub = [&](const std::vector<size_t>& p_mos, const std::vector<size_t>& q_mos) {
    //     auto np = p_mos.size();
    //     auto nq = q_mos.size();
    //     auto Z = ambit::Tensor::build(CoreTensor, "Zsub", {np, nq});
    //     Z.iterate([&](const std::vector<size_t>& i, double& value) {
    //         auto [hp, ip] = mos_rel_[p_mos[i[0]]];
    //         auto [hq, iq] = mos_rel_[q_mos[i[1]]];
    //         if (hp == hq)
    //             value = Z_->get(hp, ip, iq);
    //     });
    //     return Z;
    // };

    // 3-index integrals C^{P}_{pq} = (P|Q)^{-1/2} B^{Q}_{pq}
    auto C3 = ambit::BlockedTensor::build(CoreTensor, "C3", {"LOH", "Leu"});
    for (const std::string& block : C3.block_labels()) {
        auto p_mos = label_to_mos_[block.substr(1, 1)];
        auto q_mos = label_to_mos_[block.substr(2, 1)];
        auto np = p_mos.size();
        auto nq = q_mos.size();
        auto Cp = Csub(p_mos);
        auto Cq = Csub(q_mos);
        df_helper_->add_space("p", Cp);
        df_helper_->add_space("q", Cq);
        df_helper_->add_transformation("B", "p", "q", "Qpq");
        df_helper_->transform();
        auto B = df_helper_->get_tensor("B");
        double* C3ptr = C3.block(block).data().data();
        C_DGEMM('N', 'N', naux, np * nq, naux, 1.0, Jm12->get_pointer(), naux, B->get_pointer(),
                np * nq, 0.0, C3ptr, np * nq);
        df_helper_->clear_all();
    }

    // identity matrix
    auto I = ambit::BlockedTensor::build(CoreTensor, "I", {"OO"});
    I.iterate([&](const std::vector<size_t>& i, const std::vector<SpinType>&, double& value) {
        if (i[0] == i[1])
            value = 1.0;
    });

    // 3-index 2-RDM
    auto D3 = ambit::BlockedTensor::build(CoreTensor, "D3", {"LfH", "Lmm", "LmV", "Leu"});

    D3["Lkl"] += 2.0 * Z["gh"] * C3["Lgh"] * I["kl"];
    D3["Lgh"] += Z["gh"] * C3["Lkl"] * I["kl"];

    // Z occ/vir contribution
    D3["Lkl"] -= 0.5 * C3["Llc"] * Z["kc"];
    D3["Llk"] -= 0.5 * C3["Llc"] * Z["kc"];

    D3["Lkc"] -= 0.5 * C3["Lkl"] * Z["lc"];

    // Z frozen-core/active occ contribution
    D3["LIk"] -= 0.5 * C3["Lkm"] * Z["Im"];
    D3["Lmk"] -= 0.5 * C3["LIk"] * Z["Im"];
    D3["LkI"] -= 0.5 * C3["Lkm"] * Z["Im"];
    D3["Lkm"] -= 0.5 * C3["LIk"] * Z["Im"];

    // Z active vir/frozen-vir contribution
    D3["LkA"] -= 0.5 * C3["Lke"] * Z["eA"];
    D3["Lke"] -= 0.5 * C3["LkA"] * Z["eA"];

    // D3["Lkh"] -= 0.5 * C3["Lkg"] * Z["gh"];
    // D3["Lkh"] -= 0.5 * C3["LIk"] * Z["Ih"];
    // D3["Lgk"] -= 0.5 * C3["Lkh"] * Z["gh"];
    // D3ip("Qiq") -= 0.5 * C3ip("Qip") * Zt("pq");
    // D3pi("Qpi") -= 0.5 * C3ip("Qiq") * Zt("pq");

    // D3.print();

    // back-transform 3-index 2-RDMs
    auto d3ptr = d3->get_pointer();
    for (const std::string& block : D3.block_labels()) {
        auto bp = block.substr(1, 1);
        auto bq = block.substr(2, 1);
        auto np = label_to_mos_[bp].size();
        auto nq = label_to_mos_[bq].size();
        auto npq = np * nq;

        auto Cp = Csub(label_to_mos_[bp]);
        auto Cq = (bp == bq ? Cp : Csub(label_to_mos_[bq]));
        auto half_trans = std::make_shared<psi::Matrix>("half trans " + block, nso_, nq);
        double* D3_ptr = D3.block(block).data().data();
        double* Cp_ptr = Cp->get_pointer();
        double* Cq_ptr = Cq->get_pointer();
        double* ht_ptr = half_trans->get_pointer();

        for (size_t Q = 0; Q < naux; ++Q) {
            C_DGEMM('N', 'N', nso_, nq, np, 1.0, Cp_ptr, np, D3_ptr + Q * npq, nq, 0.0, ht_ptr, nq);
            C_DGEMM('N', 'T', nso_, nso_, nq, 1.0, ht_ptr, nq, Cq_ptr, nq, 1.0, d3ptr + Q * nso2,
                    nso_);
            if (bp != bq) {
                matrix_transpose_in_place(ht_ptr, nso_, nq);
                C_DGEMM('N', 'N', nso_, nso_, nq, 1.0, Cq_ptr, nq, ht_ptr, nso_, 1.0,
                        d3ptr + Q * nso2, nso_);
            }
        }
    }

    // 2-index 2-RDM
    auto d2ptr = d2->get_pointer();
    for (const std::string& block : D3.block_labels()) {
        double* D3ptr = D3.block(block).data().data();
        double* C3ptr = C3.block(block).data().data();
        auto factor = (block.substr(1, 1) == block.substr(2, 1) ? 1.0 : 2.0);
        auto k = C3.block(block).dim(1) * C3.block(block).dim(2);
        C_DGEMM('N', 'T', naux, naux, k, factor, C3ptr, k, D3ptr, k, 1.0, d2ptr, naux);
    }

    // // C^{P}_{pq} = (P|Q)^{-1/2} B^{Q}_{pq} integrals in ambit form
    // auto C3ints_helper = [&](const std::vector<size_t>& p_mos, const std::vector<size_t>& q_mos)
    // {
    //     auto Cp = Csub(p_mos);
    //     auto Cq = Csub(q_mos);
    //     df_helper_->add_space("p", Cp);
    //     df_helper_->add_space("q", Cq);
    //     df_helper_->add_transformation("B", "p", "q", "Qpq");
    //     df_helper_->transform();
    //     // auto C3 = psi::linalg::doublet(Jm12, df_helper_->get_tensor("B"));
    //     auto np = p_mos.size();
    //     auto nq = q_mos.size();
    //     auto C3 = ambit::Tensor::build(CoreTensor, "C3sub", {naux, np, nq});
    //     auto B = df_helper_->get_tensor("B");
    //     C_DGEMM('N', 'N', naux, np * nq, naux, 1.0, Jm12->get_pointer(), naux, B->get_pointer(),
    //             np * nq, 0.0, C3.data().data(), np * nq);
    //     df_helper_->clear_all();
    //     return C3;
    // };

    // // precompute sum_{i} C^{Q}_{ii}
    // auto C3oo = C3ints_helper(hf_docc_mos_, hf_docc_mos_);
    // auto C3Q = ambit::Tensor::build(CoreTensor, "C3Q", {naux});
    // auto Ioo = ambit::Tensor::build(CoreTensor, "Ioo", {nd, nd});
    // for (size_t i = 0; i < nd; ++i) {
    //     Ioo.data()[i * nd + i] = 1.0;
    // }
    // C3Q("Q") = C3oo("Qij") * Ioo("ij");

    // // intermediate of sum_{pq} C^{Q}_{pq} Z_{pq}
    // auto D3Q = ambit::Tensor::build(CoreTensor, "D3Q", {naux});

    // // 3-index 2-RDMs from Z occupied - virtual block
    // auto Zov = Zsub(hf_docc_mos_, hf_uocc_mos_);
    // auto C3ov = C3ints_helper(hf_docc_mos_, hf_uocc_mos_);
    // auto D3oo = ambit::Tensor::build(CoreTensor, "D3oo", C3oo.dims());
    // auto D3ov = ambit::Tensor::build(CoreTensor, "D3ov", C3ov.dims());
    // D3Q("Q") += 2.0 * C3ov("Qia") * Zov("ia");
    // D3ov("Qia") += C3Q("Q") * Zov("ia");
    // D3oo("Qji") -= 0.5 * C3ov("Qib") * Zov("jb");
    // D3oo("Qij") -= 0.5 * C3ov("Qib") * Zov("jb");
    // D3ov("Qib") -= 0.5 * C3oo("Qij") * Zov("jb");

    // // 3-index 2-RDMs from Z frozen docc - active docc block
    // auto Zfm = Zsub(label_to_mos_["f"], label_to_mos_["m"]);
    // auto C3fm = C3ints_helper(label_to_mos_["f"], label_to_mos_["m"]);
    // auto D3fm = ambit::Tensor::build(CoreTensor, "D3fm", C3fm.dims());
    // D3Q("Q") += 2.0 * C3fm("QIk") * Zfm("Ik");
    // D3fm("QIk") += C3Q("Q") * Zfm("QIk");

    // auto Zue = Zsub(label_to_mos_["u"], label_to_mos_["e"]); // frozen uocc - active uocc

    // D3oo("Qij") += 2.0 * C3ov("Qkb") * Zov("kb") * Ioo("ij");

    // // 3-index 2-RDMs
    // // std::map<std::string, std::shared_ptr<psi::Matrix>> D3map;
    // std::map<std::string, ambit::Tensor> D3map;

    // // compute C^{P}_{pq} = (P|Q)^{-1/2} B^{Q}_{pq} integrals in ambit form
    // // std::map<std::string, std::shared_ptr<psi::Matrix>> C3ints;
    // std::map<std::string, ambit::Tensor> C3ints;

    // for (const std::string& block : {"ff", "fm", "fe", "fu", "mm", "me", "mu", "eu"}) {
    //     auto p_mos = label_to_mos_[block.substr(0, 1)];
    //     auto q_mos = label_to_mos_[block.substr(1, 1)];
    //     auto np = p_mos.size();
    //     auto nq = q_mos.size();
    //     if (np == 0 or nq == 0)
    //         continue;

    //     C3ints[block] = C3ints_helper(p_mos, q_mos);
    //     C3ints[block].set_name("C3" + block);
    //     // D3map[block] = std::make_shared<psi::Matrix>("D3" + block, naux, np * nq);
    //     D3map[block] = ambit::Tensor::build(CoreTensor, "D3" + block, {naux, np, nq});
    // }

    // // precompute sum_{i} C^{Q}_{ii}
    // // std::vector<double> C3Q(naux, 0.0);
    // // for (const std::string& block : {"f", "m"}) {
    // //     const auto& C3 = C3ints[block + block];
    // //     for (size_t Q = 0; Q < naux; ++Q) {
    // //         for (size_t i = 0, size = label_to_mos_[block].size(); i < size; ++i) {
    // //             C3Q[Q] += C3->get(i, i);
    // //         }
    // //     }
    // // }

    // // compute 3-index 2-RDMs
    // // for terms 1 and 2, we loop over upper triangle blocks of Z
    // // for terms 3 and 4, we loop over upper triangle blocks of D3

    // // precompute sum_{i} C^{Q}_{ii}
    // auto C3Q = ambit::Tensor::build(CoreTensor, "C3Q", {naux});
    // for (const std::string& block : {"f", "m"}) {
    //     const auto& C3 = C3ints[block + block];
    //     auto I = ambit::Tensor::build(CoreTensor, "I" + block + block, {C3.dim(1), C3.dim(2)});
    //     I.iterate([&](const std::vector<size_t>& i, double& value) {
    //         if (i[0] == i[1])
    //             value = 1.0;
    //     });
    //     C3Q("Q") += C3("Qij") * I("ij");
    // }

    // // DOCC diagonal contribution of 3-index 2-RDMs
    // auto D3Q = ambit::Tensor::build(CoreTensor, "D3Q", {naux});

    // // loop over nonzero Z blocks, upper triangle
    // for (const std::string& block : {"fm", "fe", "fu", "me", "mu", "eu"}) {
    //     auto bp = block.substr(0, 1);
    //     auto bq = block.substr(1, 1);
    //     auto np = label_to_mos_[bp].size();
    //     auto nq = label_to_mos_[bq].size();
    //     if (np == 0 or nq == 0)
    //         continue;

    //     // auto npq = np * nq;
    //     // auto bfp = "f" + bp, bmp = "m" + bp, bpm = bp + "m";
    //     // auto bfq = "f" + bq, bmq = "m" + bq;
    //     // auto nfp = nf * np, nmp = nm * np;
    //     // auto nfq = nf * nq, nmq = nm * nq;

    //     auto Zpq = Zsub(label_to_mos_[bp], label_to_mos_[bq]);

    //     // D3^{Q}_{ii} <- C3^{Q}_{pq} * ( z_{pq} + z_{qp} ) for p < q
    //     D3Q("Q") += 2.0 * C3ints[block]("Qpq") * Zpq("pq");

    //     // D3^{Q}_{pq} <- z_{pq} * C3^{Q}_{ii}
    //     D3map[block]("Qpq") += Zpq("pq") * C3Q("Q");

    //     // // D3^{Q}_{iq} <- -0.5 * C3^{Q}_{ip} * ( z_{pq} + z_{qp} ) for p < q
    //     // D3map[bfq]("Qiq") -= 0.5 * C3ints[bfp]("Qip") * Zpq("pq");
    //     // D3map[bfp]("Qip") -= (bp == "f" ? 1 : 0.5) * C3ints[bfq]("Qiq") * Zpq("pq");

    //     // // special case for frozen-core i index ("mf", "ef", "uf" blocks are not available)
    //     // if (bp == "f") {
    //     //     D3map[bmq]("Qiq") -= 0.5 * C3ints[bpm]("Qpi") * Zpq("pq");
    //     //     D3map[bpm]("Qpi") -= 0.5 * C3ints[bmq]("Qiq") * Zpq("pq");

    //     // } else {
    //     //     D3map[bmq]("Qiq") -= 0.5 * C3ints[bmp]("Qip") * Zpq("pq");
    //     //     D3map[bmp]("Qip") -= 0.5 * C3ints[bmq]("Qiq") * Zpq("pq");
    //     // }

    //     // // D3^{Q}_{pi} <- -0.5 * C3^{Q}_{iq} * ( z_{pq} + z_{qp} ) for p < q
    // }
    // for (const std::string& block : {"f", "m"}) {
    //     const auto& D3 = D3map[block + block];
    //     auto I = ambit::Tensor::build(CoreTensor, "I" + block + block, {D3.dim(1), D3.dim(2)});
    //     I.iterate([&](const std::vector<size_t>& i, double& value) {
    //         if (i[0] == i[1])
    //             value = 1.0;
    //     });
    //     D3("Qij") += D3Q("Q") * I("ij");
    // }

    // for (const std::string& block : {"ff", "fm", "fe", "fu", "mm", "me", "mu"}) {
    //     if (D3map.find(block) == D3map.end())
    //         continue;
    //     auto bi = block.substr(0, 1);
    //     auto bq = block.substr(1, 1);

    //     auto factor = (bi == bq ? 1.0 : 0.5);

    //     for (const std::string& bp : {"f", "m", "e", "u"}) {
    //         auto bip = bi + bp;
    //         std::string ip = "ip";
    //         if (C3ints.find(bip) == C3ints.end()) {
    //             bip = bp + bi;
    //             ip = "pi";
    //         }
    //         D3map[block]("Qiq") -= factor * C3ints[bip]("Q" + ip) * Zblocks[bp + bq]("pq");
    //     }

    //     if (block == "fm") {
    //         for (const std::string& bp : {"f", "m", "e", "u"}) {
    //             auto bqp = bq + bp;
    //             std::string qp = "qp";
    //             if (C3ints.find(bqp) == C3ints.end()) {
    //                 bqp = bp + bq;
    //                 qp = "pq";
    //             }
    //             D3map[block]("Qiq") -= 0.5 * C3ints[bqp]("Q" + qp) * Zblocks[bp + bi]("pi");
    //         }
    //     }
    // }

    // for (const auto& pair : D3map) {
    //     const auto& [block, D3] = pair;
    //     D3.print();
    // }

    // // back-transform 3-index 2-RDMs
    // auto d3_tmp = std::make_shared<psi::Matrix>("d3_tmp", naux, nso_ * nso_);
    // double* d3ptr = d3_tmp->get_pointer();
    // for (auto& block_matrix : D3map) {
    //     auto& [block, D3] = block_matrix;
    //     auto bp = block.substr(0, 1);
    //     auto bq = block.substr(1, 1);
    //     auto np = label_to_mos_[bp].size();
    //     auto nq = label_to_mos_[bq].size();
    //     auto npq = np * nq;

    //     auto Cp = Csub(label_to_mos_[bp]);
    //     auto Cq = (bp == bq ? Cp : Csub(label_to_mos_[bq]));
    //     auto half_trans = std::make_shared<psi::Matrix>("half trans " + block, nso_, nq);
    //     double* D3_ptr = D3.data().data();
    //     double* Cp_ptr = Cp->get_pointer();
    //     double* Cq_ptr = Cq->get_pointer();
    //     double* ht_ptr = half_trans->get_pointer();

    //     for (size_t Q = 0; Q < naux; ++Q) {
    //         C_DGEMM('N', 'N', nso_, nq, np, 1.0, Cp_ptr, np, D3_ptr + Q * npq, nq, 0.0, ht_ptr,
    //         nq); C_DGEMM('N', 'T', nso_, nso_, nq, 1.0, ht_ptr, nq, Cq_ptr, nq, 1.0, d3ptr + Q *
    //         nso2,
    //                 nso_);
    //         if (bp != bq) {
    //             matrix_transpose_in_place(ht_ptr, nso_, nq);
    //             C_DGEMM('N', 'N', nso_, nso_, nq, 1.0, Cq_ptr, nq, ht_ptr, nso_, 1.0,
    //                     d3ptr + Q * nso2, nso_);
    //         }
    //     }
    // }

    // d3_tmp->subtract(d3tmp);
    // d3_tmp->print();
    // psi::outfile->Printf("\n D3 diff = %20.15f", d3_tmp->rms());

    // for (std::string pq : {"mf", "eu", "me"}) {
    //     auto p_mos = label_to_mos_[pq.substr(0, 1)];
    //     auto q_mos = label_to_mos_[pq.substr(1, 1)];
    //     auto np = p_mos.size();
    //     auto nq = q_mos.size();
    //     if (np == 0 or nq == 0)
    //         continue;

    //     std::string ip = "i" + pq.substr(0, 1);
    //     std::string iq = "i" + pq.substr(1, 1);
    //     auto pq_stride = np * nq;
    //     auto ip_stride = ni * np;
    //     auto iq_stride = ni * nq;

    //     // prepare C3ints and D3
    //     for (std::string block : {pq, std::string("ii"), ip, iq}) {
    //         if (C3ints.find(block) == C3ints.end()) {
    //             auto mos1 = label_to_mos_[block.substr(0, 1)];
    //             auto mos2 = label_to_mos_[block.substr(1, 1)];
    //             C3ints[block] = C3ints_helper(mos1, mos2);
    //             D3map[block] =
    //                 std::make_shared<psi::Matrix>("D3" + block, naux, mos1.size() * mos2.size());
    //             if (block == "ii") {
    //                 for (size_t Q = 0; Q < naux; ++Q) {
    //                     for (size_t i = 0; i < ni; ++i) {
    //                         C3Q[Q] += C3ints["ii"]->get(Q, i * ni + i);
    //                     }
    //                 }
    //             }
    //         }
    //     }

    //     // prepare Z_{pq}
    //     auto z = std::make_shared<psi::Matrix>("z_pq", np, nq);
    //     for (size_t p = 0; p < np; ++p) {
    //         for (size_t q = 0; q < nq; ++q) {
    //             z->set(p, q, Z->get(p_mos[p], q_mos[q]));
    //         }
    //     }
    //     double* z_ptr = z->get_pointer();

    //     for (size_t Q = 0; Q < naux; ++Q) {
    //         auto gii = C_DDOT(np * nq, z_ptr, 1, C3ints[pq]->get_pointer() + Q * pq_stride, 1);
    //         for (size_t i = 0; i < ni; ++i) {
    //             D3map["ii"]->add(i, i, gii);
    //         }

    //         C_DAXPY(np * nq, C3Q[Q], z_ptr, 1, D3map[pq]->get_pointer() + Q * pq_stride, 1);

    //         C_DGEMM('N', 'N', ni, nq, np, -0.5, C3ints[ip]->get_pointer() + Q * ip_stride, np,
    //                 z_ptr, nq, 1.0, D3map[iq]->get_pointer() + Q * iq_stride, nq);

    //         C_DGEMM('N', 'N', ni, np, nq, -0.5, C3ints[iq]->get_pointer() + Q * iq_stride, nq,
    //                 z_ptr, np, 1.0, D3map[ip]->get_pointer() + Q * ip_stride, np);
    //     }
    // }

    // // 2-index 2-RDM
    // for (const auto& kv : D3map) {
    //     auto [block, D3] = kv;
    //     outfile->Printf("\n D3 block %s", block.c_str());
    //     auto C3 = C3ints[block];
    //     double factor = (block.substr(0, 1) == block.substr(1, 1) ? 1.0 : 2.0);
    //     auto k = C3->ncol();
    //     C_DGEMM('N', 'T', naux, naux, k, factor, C3->get_pointer(), k, D3->get_pointer(), k, 1.0,
    //             d2->get_pointer(), naux);
    // }
}

void MCSCF_ORB_GRAD::dump_tpdm_iwl() {
    psi::outfile->Printf("\n    Dumping MO TPDM to disk ......");
    auto psio = _default_psio_lib_;
    IWL d2(psio.get(), PSIF_FORTE_MO_TPDM, 1.0e-15, 0, 0);
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

    if (is_frozen_orbs_)
        dump_tpdm_iwl_hf();
    psi::outfile->Printf(" Done.");
}

void MCSCF_ORB_GRAD::dump_tpdm_iwl_hf() {
    auto psio = _default_psio_lib_;
    IWL d2(psio.get(), PSIF_FORTE_MO_TPDM2, 1.0e-15, 0, 0);
    std::string name = "outfile";
    int print = debug_print_ ? 1 : 0;

    auto frzv_start_pi = nmopi_ - mo_space_info_->dimension("FROZEN_UOCC");

    for (int h = 0, offset = 0; h < nirrep_; ++h) {
        for (int j = 0, ndocc = hf_docc_mos_.size(); j < ndocc; ++j) {
            int nj = hf_docc_mos_[j];

            // frozen-core/active-docc
            for (int I = 0; I < nfrzcpi_[h]; ++I) {
                int nI = I + offset;
                for (int i = nfrzcpi_[h]; i < hf_ndoccpi_[h]; ++i) {
                    int ni = i + offset;

                    double z = Z_->get(h, I, i);

                    if (nj == ni) {
                        d2.write_value(nI, ni, ni, ni, z, print, name, 0);
                        d2.write_value(ni, nI, ni, ni, 2.0 * z, print, name, 0);
                        d2.write_value(ni, ni, ni, nI, -z, print, name, 0);
                    } else if (nj == nI) {
                        d2.write_value(ni, nI, nI, nI, z, print, name, 0);
                        d2.write_value(nI, ni, nI, nI, 2.0 * z, print, name, 0);
                        d2.write_value(nI, nI, nI, ni, -z, print, name, 0);
                    } else {
                        d2.write_value(nI, ni, nj, nj, 2.0 * z, print, name, 0);
                        d2.write_value(nI, nj, nj, ni, -z, print, name, 0);

                        d2.write_value(ni, nI, nj, nj, 2.0 * z, print, name, 0);
                        d2.write_value(ni, nj, nj, nI, -z, print, name, 0);
                    }
                }
            }

            // frozen-virtual/active-virtual
            for (int A = frzv_start_pi[h]; A < nmopi_[h]; ++A) {
                int nA = A + offset;
                for (int a = hf_ndoccpi_[h]; a < frzv_start_pi[h]; ++a) {
                    int na = a + offset;

                    double z = Z_->get(h, A, a);

                    d2.write_value(nA, na, nj, nj, 2.0 * z, print, name, 0);
                    d2.write_value(nA, nj, nj, na, -z, print, name, 0);

                    d2.write_value(na, nA, nj, nj, 2.0 * z, print, name, 0);
                    d2.write_value(na, nj, nj, nA, -z, print, name, 0);
                }
            }

            // docc/virtual
            for (int i = 0; i < hf_ndoccpi_[h]; ++i) {
                int ni = i + offset;
                for (int a = hf_ndoccpi_[h]; a < nmopi_[h]; ++a) {
                    int na = a + offset;

                    double z = Z_->get(h, i, a);

                    if (nj == ni) {
                        d2.write_value(na, ni, ni, ni, z, print, name, 0);
                        d2.write_value(ni, na, ni, ni, 2.0 * z, print, name, 0);
                        d2.write_value(ni, ni, ni, na, -z, print, name, 0);
                    } else {
                        d2.write_value(ni, na, nj, nj, 2.0 * z, print, name, 0);
                        d2.write_value(ni, nj, nj, na, -z, print, name, 0);

                        d2.write_value(na, ni, nj, nj, 2.0 * z, print, name, 0);
                        d2.write_value(na, nj, nj, ni, -z, print, name, 0);
                    }
                }
            }
        }
        offset += nmopi_[h];
    }

    d2.flush(1);
    d2.set_keep_flag(1);
    d2.close();
}
} // namespace forte
