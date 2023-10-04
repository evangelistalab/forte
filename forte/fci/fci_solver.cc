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

#include <numeric>

#include "psi4/libpsi4util/process.h"

#include "integrals/active_space_integrals.h"
#include "sparse_ci/ci_spin_adaptation.h"
#include "helpers/iterative_solvers.h"

#include "fci_solver.h"
#include "fci_vector.h"
#include "string_lists.h"
#include "helpers/printing.h"
#include "fci/string_address.h"

#ifdef HAVE_GA
#include <ga.h>
#include <macdecls.h>
#endif

#include "psi4/psi4-dec.h"

using namespace psi;

namespace forte {

class MOSpaceInfo;

FCISolver::FCISolver(StateInfo state, size_t nroot, std::shared_ptr<MOSpaceInfo> mo_space_info,
                     std::shared_ptr<ActiveSpaceIntegrals> as_ints)
    : ActiveSpaceMethod(state, nroot, mo_space_info, as_ints),
      active_dim_(mo_space_info->dimension("ACTIVE")), nirrep_(as_ints->ints()->nirrep()),
      symmetry_(state.irrep()) {
    // TODO: read this info from the base class
    na_ = state.na() - core_mo_.size() - mo_space_info->size("FROZEN_DOCC");
    nb_ = state.nb() - core_mo_.size() - mo_space_info->size("FROZEN_DOCC");
}

void FCISolver::set_maxiter_davidson(int value) { maxiter_davidson_ = value; }

void FCISolver::set_ndets_per_guess_state(size_t value) { ndets_per_guess_ = value; }

void FCISolver::set_guess_per_root(int value) { guess_per_root_ = value; }

void FCISolver::set_collapse_per_root(int value) { collapse_per_root_ = value; }

void FCISolver::set_subspace_per_root(int value) { subspace_per_root_ = value; }

void FCISolver::set_spin_adapt(bool value) { spin_adapt_ = value; }

void FCISolver::set_spin_adapt_full_preconditioner(bool value) {
    spin_adapt_full_preconditioner_ = value;
}

void FCISolver::set_test_rdms(bool value) { test_rdms_ = value; }

void FCISolver::set_print_no(bool value) { print_no_ = value; }

std::shared_ptr<FCIVector> FCISolver::get_FCIWFN() { return C_; }

std::shared_ptr<psi::Matrix> FCISolver::evecs() { return eigen_vecs_; }

std::shared_ptr<StringLists> FCISolver::lists() { return lists_; }

int FCISolver::symmetry() { return symmetry_; }

void FCISolver::startup() {
    // Create the string lists
    lists_ = std::make_shared<StringLists>(active_dim_, core_mo_, active_mo_, na_, nb_, print_);

    size_t ndfci = 0;
    for (int h = 0; h < nirrep_; ++h) {
        size_t nastr = lists_->alfa_address()->strpi(h);
        size_t nbstr = lists_->beta_address()->strpi(h ^ symmetry_);
        ndfci += nastr * nbstr;
    }

    // Create the spin adapter
    if (spin_adapt_) {
        spin_adapter_ = std::make_shared<SpinAdapter>(state().multiplicity() - 1,
                                                      state().twice_ms(), lists_->ncmo());
        dets_ = lists_->make_determinants(symmetry_);
        spin_adapter_->prepare_couplings(dets_);
    }

    if (print_) {
        table_printer printer;
        printer.add_int_data({{"Number of determinants", ndfci},
                              {"Symmetry", symmetry_},
                              {"Multiplicity", state().multiplicity()},
                              {"Number of roots", nroot_},
                              {"Target root", root_}});
        printer.add_bool_data({{"Spin adapt", spin_adapt_}});
        std::string table = printer.get_table("FCI Solver");
        outfile->Printf("%s", table.c_str());
    }
}

void FCISolver::set_options(std::shared_ptr<ForteOptions> options) {
    set_e_convergence(options->get_double("E_CONVERGENCE"));
    set_r_convergence(options->get_double("R_CONVERGENCE"));
    set_spin_adapt(options->get_bool("CI_SPIN_ADAPT"));
    set_spin_adapt_full_preconditioner(options->get_bool("CI_SPIN_ADAPT_FULL_PRECONDITIONER"));
    set_test_rdms(options->get_bool("FCI_TEST_RDMS"));

    set_root(options->get_int("ROOT"));

    set_guess_per_root(options->get_int("DL_GUESS_PER_ROOT"));
    set_ndets_per_guess_state(options->get_int("DL_DETS_PER_GUESS"));
    set_collapse_per_root(options->get_int("DL_COLLAPSE_PER_ROOT"));
    set_subspace_per_root(options->get_int("DL_SUBSPACE_PER_ROOT"));
    set_maxiter_davidson(options->get_int("DL_MAXITER"));
    // set_save_dl_vectors(options->get_bool("DL_SAVE_VECTORS"));

    set_print(options->get_int("PRINT"));
}

/*
 * See Appendix A in J. Comput. Chem. 2001 vol. 22 (13) pp. 1574-1589
 */
double FCISolver::compute_energy() {
    local_timer t;
    startup();

    FCIVector::allocate_temp_space(lists_, print_);

    FCIVector Hdiag(lists_, symmetry_);
    C_ = std::make_shared<FCIVector>(lists_, symmetry_);
    FCIVector HC(lists_, symmetry_);
    C_->set_print(print_);

    // Compute the size of the determinant space and the basis used by the Davidson solver
    size_t det_size = Hdiag.size();
    size_t basis_size = spin_adapt_ ? spin_adapter_->ncsf() : det_size;

    // Create the vectors that stores the b and sigma vectors in the determinant basis
    auto b = std::make_shared<psi::Vector>("b", det_size);
    auto sigma = std::make_shared<psi::Vector>("sigma", det_size);

    // Optionally create the vectors that stores the b and sigma vectors in the CSF basis
    auto b_basis = b;
    auto sigma_basis = sigma;

    if (spin_adapt_) {
        b_basis = std::make_shared<psi::Vector>("b", basis_size);
        sigma_basis = std::make_shared<psi::Vector>("sigma", basis_size);
    }

    // Create the Davidson solver and set the options
    DavidsonLiuSolver dls(basis_size, nroot_);
    dls.set_e_convergence(e_convergence_);
    dls.set_r_convergence(r_convergence_);
    dls.set_print_level(print_);
    dls.set_collapse_per_root(collapse_per_root_);
    dls.set_subspace_per_root(subspace_per_root_);

    // determine the number of guess vectors
    const size_t num_guess_states = std::min(guess_per_root_ * nroot_, basis_size);

    // Form the diagonal of the Hamiltonian and the initial guess
    if (spin_adapt_) {
        auto Hdiag_vec = form_Hdiag_csf(as_ints_, spin_adapter_);
        dls.startup(Hdiag_vec);
        initial_guess_csf(Hdiag_vec, num_guess_states, dls);
    } else {
        Hdiag.form_H_diagonal(as_ints_);
        Hdiag.copy_to(sigma);
        dls.startup(sigma);
        initial_guess_det(Hdiag, num_guess_states, as_ints_, dls);
    }

    // Set a variable to track the convergence of the solver
    SolverStatus converged = SolverStatus::NotConverged;

    if (print_) {
        outfile->Printf("\n\n  ==> Diagonalizing Hamiltonian <==\n");
        outfile->Printf("\n  Energy   convergence: %.2e", dls.get_e_convergence());
        outfile->Printf("\n  Residual convergence: %.2e", dls.get_r_convergence());
        outfile->Printf("\n  -----------------------------------------------------");
        outfile->Printf("\n    Iter.      Avg. Energy       Delta_E     Res. Norm");
        outfile->Printf("\n  -----------------------------------------------------");
    }

    double old_avg_energy = 0.0;
    int real_cycle = 1;
    for (int cycle = 0; cycle < maxiter_davidson_; ++cycle) {
        bool add_sigma = true;
        do {
            // get the next b vector and compute the sigma vector
            dls.get_b(b_basis);
            if (spin_adapt_) {
                // Compute sigma in the CSF basis and convert it to the determinant basis
                spin_adapter_->csf_C_to_det_C(b_basis, b);
                C_->copy(b);
                C_->Hamiltonian(HC, as_ints_);
                HC.copy_to(sigma);
                spin_adapter_->det_C_to_csf_C(sigma, sigma_basis);
            } else {
                // Compute sigma in the determinant basis
                C_->copy(b_basis);
                C_->Hamiltonian(HC, as_ints_);
                HC.copy_to(sigma_basis);
            }
            add_sigma = dls.add_sigma(sigma_basis);
        } while (add_sigma);

        converged = dls.update();

        if (converged != SolverStatus::Collapse) {
            // compute the average energy
            double avg_energy = 0.0;
            for (size_t r = 0; r < nroot_; ++r) {
                avg_energy += dls.eigenvalues()->get(r);
            }
            avg_energy /= static_cast<double>(nroot_);

            // compute the average residual
            auto r = dls.residuals();
            double avg_residual =
                std::accumulate(r.begin(), r.end(), 0.0) / static_cast<double>(nroot_);

            if (print_) {
                outfile->Printf("\n    %3d  %20.12f  %+.3e  %+.3e", real_cycle, avg_energy,
                                avg_energy - old_avg_energy, avg_residual);
            }
            old_avg_energy = avg_energy;
            real_cycle++;
        }

        if (converged == SolverStatus::Converged)
            break;
    }

    if (print_) {
        outfile->Printf("\n  -----------------------------------------------------");
        if (converged == SolverStatus::Converged) {
            outfile->Printf("\n  The Davidson-Liu algorithm converged in %d iterations.",
                            real_cycle);
        }
    }

    if (converged == SolverStatus::NotConverged) {
        outfile->Printf("\n  FCI did not converge!");
        throw std::runtime_error("FCI did not converge. Try increasing DL_MAXITER.");
    }

    // Copy eigenvalues and eigenvectors from the Davidson-Liu solver
    evals_ = dls.eigenvalues();
    energies_ = std::vector<double>(nroot_, 0.0);
    spin2_ = std::vector<double>(nroot_, 0.0);
    for (size_t r = 0; r < nroot_; r++) {
        energies_[r] = evals_->get(r);
        b_basis = dls.eigenvector(r);
        if (spin_adapt_) {
            spin_adapter_->csf_C_to_det_C(b_basis, b);
        } else {
            b = b_basis;
        }
        C_->copy(b);
        spin2_[r] = C_->compute_spin2();
    }
    eigen_vecs_ = dls.eigenvectors();

    // Print determinants
    if (print_) {
        print_solutions(num_guess_states, b, b_basis, dls);
    }

    // Optionally, test the RDMs
    if (test_rdms_) {
        test_rdms(b, b_basis, dls);
    }

    energy_ = dls.eigenvalues()->get(root_);
    psi::Process::environment.globals["CURRENT ENERGY"] = energy_;
    psi::Process::environment.globals["FCI ENERGY"] = energy_;

    return energy_;
}

void FCISolver::compute_rdms_root(size_t root1, size_t /*root2*/, int max_rdm_level) {
    // make sure a compute_energy is called before this
    if (C_) {
        if (root1 >= nroot_) {
            std::string error = "Cannot compute RDMs of root " + std::to_string(root1) +
                                "(0-based) because nroot = " + std::to_string(nroot_);
            throw psi::PSIEXCEPTION(error);
        }

        std::shared_ptr<psi::Vector> evec(eigen_vecs_->get_row(0, root1));
        std::shared_ptr<psi::Vector> b;
        if (spin_adapt_) {
            b = std::make_shared<psi::Vector>(spin_adapter_->ndet());
            spin_adapter_->csf_C_to_det_C(evec, b);
        } else {
            b = evec;
        }
        C_->copy(b);
        if (print_) {
            std::string title_rdm = "Computing RDMs for Root No. " + std::to_string(root1);
            print_h2(title_rdm);
        }
        C_->compute_rdms(max_rdm_level);

        // Optionally, test the RDMs
        if (test_rdms_) {
            C_->rdm_test();
        }

        // Print the NO if energy converged
        if (print_no_ || print_ > 0) {
            C_->print_natural_orbitals(mo_space_info_);
        }
    } else {
        throw psi::PSIEXCEPTION("FCIVector is not assigned. Cannot compute RDMs.");
    }
}

void FCISolver::print_solutions(size_t guess_size, std::shared_ptr<psi::Vector> b,
                                std::shared_ptr<psi::Vector> b_basis, DavidsonLiuSolver& dls) {
    for (size_t r = 0; r < nroot_; ++r) {
        outfile->Printf("\n\n  ==> Root No. %d <==\n", r);

        b_basis = dls.eigenvector(r);
        if (spin_adapt_) {
            spin_adapter_->csf_C_to_det_C(b_basis, b);
        } else {
            b = b_basis;
        }
        C_->copy(b);
        std::vector<std::tuple<double, double, size_t, size_t, size_t>> dets_config =
            C_->max_abs_elements(guess_size);

        for (auto& det_config : dets_config) {
            double ci_abs, ci;
            size_t h, add_Ia, add_Ib;
            std::tie(ci_abs, ci, h, add_Ia, add_Ib) = det_config;

            if (ci_abs < 0.1)
                continue;

            auto Ia_v = lists_->alfa_str(h, add_Ia);
            auto Ib_v = lists_->beta_str(h ^ symmetry_, add_Ib);

            outfile->Printf("\n    ");
            size_t offset = 0;
            for (int h = 0; h < nirrep_; ++h) {
                for (int k = 0; k < active_dim_[h]; ++k) {
                    size_t i = k + offset;
                    bool a = Ia_v[i];
                    bool b = Ib_v[i];
                    if (a == b) {
                        outfile->Printf("%d", a ? 2 : 0);
                    } else {
                        outfile->Printf("%c", a ? 'a' : 'b');
                    }
                }
                if (active_dim_[h] != 0)
                    outfile->Printf(" ");
                offset += active_dim_[h];
            }
            outfile->Printf("%15.8f", ci);
        }

        double root_energy = dls.eigenvalues()->get(r);

        outfile->Printf("\n\n    Total Energy: %20.12f, <S^2>: %8.6f", root_energy, spin2_[r]);
    }
}

void FCISolver::test_rdms(std::shared_ptr<psi::Vector> b, std::shared_ptr<psi::Vector> b_basis,
                          DavidsonLiuSolver& dls) {
    b_basis = dls.eigenvector(root_);
    if (spin_adapt_) {
        spin_adapter_->csf_C_to_det_C(b_basis, b);
    } else {
        b = b_basis;
    }
    C_->copy(b);
    if (print_) {
        std::string title_rdm = "Computing RDMs for Root No. " + std::to_string(root_);
        print_h2(title_rdm);
    }
    C_->compute_rdms(3);
    C_->rdm_test();
}

std::shared_ptr<psi::Matrix> FCISolver::ci_wave_functions() {
    if (eigen_vecs_ == nullptr)
        return std::make_shared<psi::Matrix>();

    auto evecs = std::make_shared<psi::Matrix>("FCI Eigenvectors", eigen_vecs_->ncol(), nroot_);
    for (int i = 0, size = static_cast<int>(nroot_); i < size; ++i) {
        evecs->set_column(0, i, eigen_vecs_->get_row(0, i));
    }
    return evecs;
}

std::vector<std::shared_ptr<RDMs>>
FCISolver::rdms(const std::vector<std::pair<size_t, size_t>>& root_list, int max_rdm_level,
                RDMsType type) {
    if (max_rdm_level <= 0) {
        auto nroots = root_list.size();
        if (type == RDMsType::spin_dependent) {
            return std::vector<std::shared_ptr<RDMs>>(nroots,
                                                      std::make_shared<RDMsSpinDependent>());
        } else {
            return std::vector<std::shared_ptr<RDMs>>(nroots, std::make_shared<RDMsSpinFree>());
        }
    }

    std::vector<std::shared_ptr<RDMs>> refs;

    // loop over all the pairs of references
    for (auto& roots : root_list) {

        if (roots.first != roots.second) {
            throw psi::PSIEXCEPTION(
                "FCISolver::rdms(): cannot compute transition RDMs within the same symmetry.");
        }

        compute_rdms_root(roots.first, roots.second, max_rdm_level);

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

std::vector<std::shared_ptr<RDMs>>
FCISolver::transition_rdms(const std::vector<std::pair<size_t, size_t>>& /*root_list*/,
                           std::shared_ptr<ActiveSpaceMethod> /*method2*/, int /*max_rdm_level*/,
                           RDMsType /*rdm_type*/) {
    std::vector<std::shared_ptr<RDMs>> refs;
    throw std::runtime_error("FCISolver::transition_rdms is not implemented!");
    return refs;
}
} // namespace forte
