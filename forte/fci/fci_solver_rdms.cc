/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2023 by its authors (see COPYING, COPYING.LESSER, AUTHORS).
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

#include "helpers/printing.h"

#include "sparse_ci/determinant.h"
#include "sparse_ci/ci_spin_adaptation.h"
#include "string_lists.h"
#include "fci_vector.h"

#include "fci_solver.h"

namespace forte {

std::vector<std::shared_ptr<RDMs>>
FCISolver::rdms(const std::vector<std::pair<size_t, size_t>>& root_list, int max_rdm_level,
                RDMsType type) {
    if (not C_) {
        throw std::runtime_error("FCIVector is not assigned. Cannot compute RDMs.");
    }

    if (max_rdm_level <= 0) {
        auto nroots = root_list.size();
        if (type == RDMsType::spin_dependent) {
            return std::vector<std::shared_ptr<RDMs>>(nroots,
                                                      std::make_shared<RDMsSpinDependent>());
        } else {
            return std::vector<std::shared_ptr<RDMs>>(nroots, std::make_shared<RDMsSpinFree>());
        }
    }

    // check if we have to compute off-diagonal RDMs
    bool has_offdiag = false;
    for (auto& roots : root_list) {
        if (roots.first != roots.second) {
            has_offdiag = true;
            break;
        }
    }
    auto C_left = C_;
    auto C_right = C_;
    if (has_offdiag) {
        C_left = std::make_shared<FCIVector>(lists_, symmetry_);
    }

    std::vector<std::shared_ptr<RDMs>> refs;
    // loop over all the pairs of references
    for (const auto& [root1, root2] : root_list) {
        compute_rdms_root(root1, root2, max_rdm_level, C_left, C_right);

        size_t nact = active_dim_.sum();
        size_t nact2 = nact * nact;
        size_t nact3 = nact2 * nact;
        size_t nact4 = nact3 * nact;
        size_t nact5 = nact4 * nact;

        ambit::Tensor g1a, g1b;
        ambit::Tensor g2aa, g2ab, g2bb;
        ambit::Tensor g3aaa, g3aab, g3abb, g3bbb;

        // TODO: the following needs clean-up/optimization for spin-free RDMs
        // TODO: put RDMs directly as ambit Tensor in FCIVector?

        if (max_rdm_level >= 1) {
            // One-particle density matrices in the active space
            std::vector<double>& opdm_a = C_->opdm_a();
            std::vector<double>& opdm_b = C_->opdm_b();
            g1a = ambit::Tensor::build(ambit::CoreTensor, "g1a", {nact, nact});
            g1b = ambit::Tensor::build(ambit::CoreTensor, "g1b", {nact, nact});
            if (na_ >= 1) {
                g1a.iterate([&](const std::vector<size_t>& i, double& value) {
                    value = opdm_a[i[0] * nact + i[1]];
                });
            }
            if (nb_ >= 1) {
                g1b.iterate([&](const std::vector<size_t>& i, double& value) {
                    value = opdm_b[i[0] * nact + i[1]];
                });
            }
        }

        if (max_rdm_level >= 2) {
            // Two-particle density matrices in the active space
            g2aa = ambit::Tensor::build(ambit::CoreTensor, "g2aa", {nact, nact, nact, nact});
            g2ab = ambit::Tensor::build(ambit::CoreTensor, "g2ab", {nact, nact, nact, nact});
            g2bb = ambit::Tensor::build(ambit::CoreTensor, "g2bb", {nact, nact, nact, nact});

            if (na_ >= 2) {
                std::vector<double>& tpdm_aa = C_->tpdm_aa();
                g2aa.iterate([&](const std::vector<size_t>& i, double& value) {
                    value = tpdm_aa[i[0] * nact3 + i[1] * nact2 + i[2] * nact + i[3]];
                });
            }
            if ((na_ >= 1) and (nb_ >= 1)) {
                std::vector<double>& tpdm_ab = C_->tpdm_ab();
                g2ab.iterate([&](const std::vector<size_t>& i, double& value) {
                    value = tpdm_ab[i[0] * nact3 + i[1] * nact2 + i[2] * nact + i[3]];
                });
            }
            if (nb_ >= 2) {
                std::vector<double>& tpdm_bb = C_->tpdm_bb();
                g2bb.iterate([&](const std::vector<size_t>& i, double& value) {
                    value = tpdm_bb[i[0] * nact3 + i[1] * nact2 + i[2] * nact + i[3]];
                });
            }
        }

        if (max_rdm_level >= 3) {
            // Three-particle density matrices in the active space
            g3aaa = ambit::Tensor::build(ambit::CoreTensor, "g3aaa",
                                         {nact, nact, nact, nact, nact, nact});
            g3aab = ambit::Tensor::build(ambit::CoreTensor, "g3aab",
                                         {nact, nact, nact, nact, nact, nact});
            g3abb = ambit::Tensor::build(ambit::CoreTensor, "g3abb",
                                         {nact, nact, nact, nact, nact, nact});
            g3bbb = ambit::Tensor::build(ambit::CoreTensor, "g3bbb",
                                         {nact, nact, nact, nact, nact, nact});
            if (na_ >= 3) {
                std::vector<double>& tpdm_aaa = C_->tpdm_aaa();
                g3aaa.iterate([&](const std::vector<size_t>& i, double& value) {
                    value = tpdm_aaa[i[0] * nact5 + i[1] * nact4 + i[2] * nact3 + i[3] * nact2 +
                                     i[4] * nact + i[5]];
                });
            }
            if ((na_ >= 2) and (nb_ >= 1)) {
                std::vector<double>& tpdm_aab = C_->tpdm_aab();
                g3aab.iterate([&](const std::vector<size_t>& i, double& value) {
                    value = tpdm_aab[i[0] * nact5 + i[1] * nact4 + i[2] * nact3 + i[3] * nact2 +
                                     i[4] * nact + i[5]];
                });
            }
            if ((na_ >= 1) and (nb_ >= 2)) {
                std::vector<double>& tpdm_abb = C_->tpdm_abb();
                g3abb.iterate([&](const std::vector<size_t>& i, double& value) {
                    value = tpdm_abb[i[0] * nact5 + i[1] * nact4 + i[2] * nact3 + i[3] * nact2 +
                                     i[4] * nact + i[5]];
                });
            }
            if (nb_ >= 3) {
                std::vector<double>& tpdm_bbb = C_->tpdm_bbb();
                g3bbb.iterate([&](const std::vector<size_t>& i, double& value) {
                    value = tpdm_bbb[i[0] * nact5 + i[1] * nact4 + i[2] * nact3 + i[3] * nact2 +
                                     i[4] * nact + i[5]];
                });
            }
        }

        if (type == RDMsType::spin_dependent) {
            if (max_rdm_level == 1) {
                refs.emplace_back(std::make_shared<RDMsSpinDependent>(g1a, g1b));
            }
            if (max_rdm_level == 2) {
                refs.emplace_back(std::make_shared<RDMsSpinDependent>(g1a, g1b, g2aa, g2ab, g2bb));
            }
            if (max_rdm_level == 3) {
                refs.emplace_back(std::make_shared<RDMsSpinDependent>(g1a, g1b, g2aa, g2ab, g2bb,
                                                                      g3aaa, g3aab, g3abb, g3bbb));
            }
        } else {
            g1a("pq") += g1b("pq");
            if (max_rdm_level > 1) {
                g2aa("pqrs") += g2ab("pqrs") + g2ab("qpsr");
                g2aa("pqrs") += g2bb("pqrs");
            }
            if (max_rdm_level > 2) {
                g3aaa("pqrstu") += g3aab("pqrstu") + g3aab("prqsut") + g3aab("qrptus");
                g3aaa("pqrstu") += g3abb("pqrstu") + g3abb("qprtsu") + g3abb("rpqust");
                g3aaa("pqrstu") += g3bbb("pqrstu");
            }
            if (max_rdm_level == 1)
                refs.emplace_back(std::make_shared<RDMsSpinFree>(g1a));
            if (max_rdm_level == 2)
                refs.emplace_back(std::make_shared<RDMsSpinFree>(g1a, g2aa));
            if (max_rdm_level == 3)
                refs.emplace_back(std::make_shared<RDMsSpinFree>(g1a, g2aa, g3aaa));
        }
    }
    return refs;
}

void FCISolver::compute_rdms_root(size_t root_left, size_t root_right,
                                  std::shared_ptr<FCIVector> C_left,
                                  std::shared_ptr<FCIVector> C_right, max_rdm_level) {
    // make sure the root is valid
    if ((root1 >= nroot_) or (root2 >= nroot_))) {
            std::string error = "Cannot compute RDMs of root " + std::to_string(root1) +
                                "(0-based) because nroot = " + std::to_string(nroot_);
            throw std::runtime_error(error);
        }

    std::shared_ptr<psi::Vector> cl, cr;
    if (spin_adapt_) {
        cl = std::make_shared<psi::Vector>(spin_adapter_->ndet());
        spin_adapter_->csf_C_to_det_C(eigen_vecs_->get_row(0, root_left), cl);
    } else {
        cl = eigen_vecs_->get_row(0, root_left);
    }
    C_left->copy(cl);

    if (root1 != root2) {
        if (spin_adapt_) {
            cr = std::make_shared<psi::Vector>(spin_adapter_->ndet());
            spin_adapter_->csf_C_to_det_C(eigen_vecs_->get_row(0, root_right), cr);
        } else {
            cr = eigen_vecs_->get_row(0, root_right);
        }
        C_right->copy(cr);
    } else {
        C_right = C_left;
    }

    if (print_) {
        std::string title_rdm = "Computing RDMs for Root Pair " + std::to_string(root_left) +
                                " and " + std::to_string(root_right);
        print_h2(title_rdm);
    }
    C_left->compute_rdms(C_right, max_rdm_level);

    // Optionally, test the RDMs
    if (test_rdms_) {
        C_->rdm_test();
    }

    // Print the NO if energy converged
    if (print_no_ || print_ > 0) {
        C_->print_natural_orbitals(mo_space_info_);
    }
}

} // namespace forte
