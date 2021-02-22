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

#include "psi4/libiwl/iwl.hpp"
#include "psi4/libmints/wavefunction.h"
#include "psi4/libpsi4util/PsiOutStream.h"
#include "psi4/psifiles.h"

#include "helpers/printing.h"
#include "helpers/lbfgs/lbfgs.h"

#include "gradient_tpdm/backtransform_tpdm.h"
#include "casscf/casscf_orb_grad.h"
#include "casscf/cpscf.h"

using namespace psi;
using namespace ambit;

namespace forte {

void CASSCF_ORB_GRAD::compute_nuclear_gradient() {
    print_h2("MCSCF Gradient");

    // format A to SharedMatrix
    Am_ = std::make_shared<psi::Matrix>("A (MCSCF)", nmopi_, nmopi_);
    fill_A_matrix_data(A_);

    is_frozen_orbs_ = nfrzc_ or mo_space_info_->size("FROZEN_UOCC");

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

void CASSCF_ORB_GRAD::JK_build(psi::SharedMatrix Cl, psi::SharedMatrix Cr) {
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

psi::SharedMatrix CASSCF_ORB_GRAD::C_subset(const std::string& name, psi::SharedMatrix C,
                                            psi::Dimension dim_start, psi::Dimension dim_end) {
    auto dim = dim_end - dim_start;
    auto Csub = std::make_shared<psi::Matrix>(name, nsopi_, dim);

    for (int h = 0; h < nirrep_; ++h) {
        for (int p = 0, offset = dim_start[h]; p < dim[h]; ++p) {
            Csub->set_column(h, p, C->get_column(h, p + offset));
        }
    }

    return Csub;
}

void CASSCF_ORB_GRAD::fill_A_matrix_data(ambit::BlockedTensor A) {
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

void CASSCF_ORB_GRAD::setup_grad_frozen() {
    // terminate for high spin because ROHF CP-SCF is not implemented
    if (ints_->wfn()->soccpi().sum())
        throw std::runtime_error("MCSCF gradient with frozen orbitals only works for singlet!");

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

void CASSCF_ORB_GRAD::build_Am_frozen() {
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

void CASSCF_ORB_GRAD::solve_cpscf() {
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
    auto edocc = epsilon_->get_block(Sdocc);
    auto euocc = epsilon_->get_block(Suocc);

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

SharedMatrix CASSCF_ORB_GRAD::contract_RB_Z(psi::SharedMatrix Z, psi::SharedMatrix C_Zrow,
                                            psi::SharedMatrix C_Zcol, psi::SharedMatrix C_row,
                                            psi::SharedMatrix C_col) {
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

void CASSCF_ORB_GRAD::compute_Lagrangian() {
    psi::outfile->Printf("\n    Computing AO Lagrangian ...");
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
    ints_->wfn()->lagrangian()->copy(W);
    psi::outfile->Printf(" Done.");
}

void CASSCF_ORB_GRAD::compute_opdm_ao() {
    psi::outfile->Printf("\n    Computing AO OPDM .........");
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

void CASSCF_ORB_GRAD::dump_tpdm_iwl() {
    psi::outfile->Printf("\n    Dumping MO TPDM to disk ...");
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

void CASSCF_ORB_GRAD::dump_tpdm_iwl_hf() {
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
