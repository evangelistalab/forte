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

#include "psi4/libpsi4util/process.h"
#include "psi4/libmints/molecule.h"

#include "boost/format.hpp"

#include "base_classes/rdms.h"
#include "base_classes/forte_options.h"
#include "base_classes/mo_space_info.h"

#include "integrals/active_space_integrals.h"
#include "sparse_ci/determinant.h"
#include "sparse_ci/determinant_functions.hpp"
#include "helpers/iterative_solvers.h"

#include "fci_solver.h"
#include "fci_vector.h"
#include "string_lists.h"
#include "helpers/helpers.h"
#include "helpers/printing.h"

#ifdef HAVE_GA
#include <ga.h>
#include <macdecls.h>
#endif

#include "psi4/psi4-dec.h"

using namespace psi;

int fci_debug_level = 4;

namespace forte {

class MOSpaceInfo;

FCISolver::FCISolver(StateInfo state, size_t nroot, std::shared_ptr<MOSpaceInfo> mo_space_info,
                     std::shared_ptr<ActiveSpaceIntegrals> as_ints)
    : ActiveSpaceMethod(state, nroot, mo_space_info, as_ints),
      active_dim_(mo_space_info->dimension("ACTIVE")), nirrep_(as_ints->ints()->nirrep()),
      symmetry_(state.irrep()), multiplicity_(state.multiplicity()) {
    // TODO: read this info from the base class
    na_ = state.na() - core_mo_.size() - mo_space_info->size("FROZEN_DOCC");
    nb_ = state.nb() - core_mo_.size() - mo_space_info->size("FROZEN_DOCC");
}

void FCISolver::set_ntrial_per_root(int value) { ntrial_per_root_ = value; }

void FCISolver::set_fci_iterations(int value) { fci_iterations_ = value; }

void FCISolver::set_collapse_per_root(int value) { collapse_per_root_ = value; }

void FCISolver::set_subspace_per_root(int value) { subspace_per_root_ = value; }

void FCISolver::startup() {
    // Create the string lists
    lists_ = std::shared_ptr<StringLists>(
        new StringLists(twoSubstituitionVVOO, active_dim_, core_mo_, active_mo_, na_, nb_, print_));

    size_t ndfci = 0;
    for (int h = 0; h < nirrep_; ++h) {
        size_t nastr = lists_->alfa_graph()->strpi(h);
        size_t nbstr = lists_->beta_graph()->strpi(h ^ symmetry_);
        ndfci += nastr * nbstr;
    }
    if (print_) {
        // Print a summary of options
        std::vector<std::pair<std::string, int>> calculation_info{
            {"Number of determinants", ndfci},
            {"Symmetry", symmetry_},
            {"Multiplicity", multiplicity_},
            {"Number of roots", nroot_},
            {"Target root", root_},
            {"Trial vectors per root", ntrial_per_root_}};

        // Print some information
        outfile->Printf("\n\n  ==> FCI Solver <==\n\n");
        for (auto& str_dim : calculation_info) {
            outfile->Printf("    %-39s %10d\n", str_dim.first.c_str(), str_dim.second);
        }
    }
}

void FCISolver::set_options(std::shared_ptr<ForteOptions> options) {
    set_root(options->get_int("ROOT"));
    set_test_rdms(options->get_bool("FCI_TEST_RDMS"));
    set_fci_iterations(options->get_int("FCI_MAXITER"));
    set_collapse_per_root(options->get_int("DL_COLLAPSE_PER_ROOT"));
    set_subspace_per_root(options->get_int("DL_SUBSPACE_PER_ROOT"));
    set_print(options->get_int("PRINT"));
    set_ntrial_per_root(options->get_int("NTRIAL_PER_ROOT"));
    set_print(options->get_int("PRINT"));
    set_e_convergence(options->get_double("E_CONVERGENCE"));
    set_r_convergence(options->get_double("R_CONVERGENCE"));
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

    size_t fci_size = Hdiag.size();
    Hdiag.form_H_diagonal(as_ints_);

    psi::SharedVector b(new Vector("b", fci_size));
    psi::SharedVector sigma(new Vector("sigma", fci_size));

    Hdiag.copy_to(sigma);

    DavidsonLiuSolver dls(fci_size, nroot_);
    dls.set_e_convergence(e_convergence_);
    dls.set_r_convergence(r_convergence_);
    dls.set_print_level(print_);
    dls.set_collapse_per_root(collapse_per_root_);
    dls.set_subspace_per_root(subspace_per_root_);
    dls.startup(sigma);

    size_t guess_size = dls.collapse_size();
    auto guess = initial_guess(Hdiag, guess_size, as_ints_);

    std::vector<int> guess_list;
    for (size_t g = 0; g < guess.size(); ++g) {
        if (guess[g].first == multiplicity_)
            guess_list.push_back(g);
    }

    // number of guess to be used
    size_t nguess = std::min(guess_list.size(), guess_size);

    if (nguess == 0) {
        throw psi::PSIEXCEPTION("\n\n  Found zero FCI guesses with the requested "
                                "multiplicity.\n\n");
    }

    for (size_t n = 0; n < nguess; ++n) {
        HC.set(guess[guess_list[n]].second);
        HC.copy_to(sigma);
        dls.add_guess(sigma);
    }

    // Prepare a list of bad roots to project out and pass them to the solver
    std::vector<std::vector<std::pair<size_t, double>>> bad_roots;
    int gr = 0;
    for (auto& g : guess) {
        if (g.first != multiplicity_) {
            if (print_ > 0) {
                outfile->Printf("\n  Projecting out root %d", gr);
            }
            HC.set(g.second);
            HC.copy_to(sigma);
            std::vector<std::pair<size_t, double>> bad_root;
            for (size_t I = 0; I < fci_size; ++I) {
                if (std::fabs(sigma->get(I)) > 1.0e-12) {
                    bad_root.push_back(std::make_pair(I, sigma->get(I)));
                }
            }
            bad_roots.push_back(bad_root);
        }
        gr += 1;
    }
    dls.set_project_out(bad_roots);

    SolverStatus converged = SolverStatus::NotConverged;

    if (print_) {
        outfile->Printf("\n\n  ==> Diagonalizing Hamiltonian <==\n");
        outfile->Printf("\n  -----------------------------------------------------");
        outfile->Printf("\n    Iter.      Avg. Energy       Delta_E     Res. Norm");
        outfile->Printf("\n  -----------------------------------------------------");
    }

    double old_avg_energy = 0.0;
    int real_cycle = 1;
    for (int cycle = 0; cycle < fci_iterations_; ++cycle) {
        bool add_sigma = true;
        do {
            dls.get_b(b);
            C_->copy(b);
            C_->Hamiltonian(HC, as_ints_);
            HC.copy_to(sigma);
            add_sigma = dls.add_sigma(sigma);
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
        throw psi::PSIEXCEPTION("FCI did not converge. Try increasing FCI_ITERATIONS.");
    }

    // Compute final eigenvectors
    dls.get_results();

    // Copy eigen values and eigen vectors
    evals_ = dls.eigenvalues();
    energies_ = std::vector<double>(nroot_, 0.0);
    for (size_t r = 0; r < nroot_; r++) {
        energies_[r] = evals_->get(r);
    }
    eigen_vecs_ = dls.eigenvectors();

    // Print determinants
    if (print_) {
        for (size_t r = 0; r < nroot_; ++r) {
            outfile->Printf("\n\n  ==> Root No. %d <==\n", r);

            C_->copy(dls.eigenvector(r));
            std::vector<std::tuple<double, double, size_t, size_t, size_t>> dets_config =
                C_->max_abs_elements(guess_size * ntrial_per_root_);
            // psi::Dimension nactvpi = mo_space_info_->dimension("ACTIVE");

            for (auto& det_config : dets_config) {
                double ci_abs, ci;
                size_t h, add_Ia, add_Ib;
                std::tie(ci_abs, ci, h, add_Ia, add_Ib) = det_config;

                if (ci_abs < 0.1)
                    continue;

                std::bitset<Determinant::nbits_half> Ia_v = lists_->alfa_str(h, add_Ia);
                std::bitset<Determinant::nbits_half> Ib_v = lists_->beta_str(h ^ symmetry_, add_Ib);

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

            outfile->Printf("\n\n    Total Energy: %25.15f", root_energy);
        }
    }

    //    // Optionally, test the RDMs
    if (test_rdms_) {
        C_->copy(dls.eigenvector(root_));
        if (print_) {
            std::string title_rdm = "Computing RDMs for Root No. " + std::to_string(root_);
            print_h2(title_rdm);
        }
        C_->compute_rdms(3);
        C_->rdm_test();
    }

    //    // Print the NO if energy converged
    //    if (print_no_ || print_ > 0) {
    //        C_->print_natural_orbitals(mo_space_info_);
    //    }

    energy_ = dls.eigenvalues()->get(root_);
    psi::Process::environment.globals["CURRENT ENERGY"] = energy_;
    psi::Process::environment.globals["FCI ENERGY"] = energy_;

    return energy_;
}

void FCISolver::compute_rdms_root(int root1, int root2, int max_rdm_level) {
    // make sure a compute_energy is called before this
    if (C_) {
        if (root1 >= nroot_) {
            std::string error = "Cannot compute RDMs of root " + std::to_string(root1) +
                                "(0-based) because nroot = " + std::to_string(nroot_);
            throw psi::PSIEXCEPTION(error);
        }

        psi::SharedVector evec(eigen_vecs_->get_row(0, root1));
        C_->copy(evec);
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

std::vector<std::pair<int, std::vector<std::tuple<size_t, size_t, size_t, double>>>>
FCISolver::initial_guess(FCIVector& diag, size_t n,
                         std::shared_ptr<ActiveSpaceIntegrals> fci_ints) {
    local_timer t;

    double nuclear_repulsion_energy =
        psi::Process::environment.molecule()->nuclear_repulsion_energy({{0, 0, 0}});
    double scalar_energy = fci_ints->scalar_energy();

    size_t ntrial = n * ntrial_per_root_;

    // Get the list of most important determinants
    std::vector<std::tuple<double, size_t, size_t, size_t>> dets = diag.min_elements(ntrial);

    size_t num_dets = dets.size();

    std::vector<Determinant> bsdets;

    // Build the full determinants
    size_t nact = active_mo_.size();
    for (auto det : dets) {
        double e;
        size_t h, add_Ia, add_Ib;
        std::tie(e, h, add_Ia, add_Ib) = det;
        std::bitset<Determinant::nbits_half> Ia_v = lists_->alfa_str(h, add_Ia);
        std::bitset<Determinant::nbits_half> Ib_v = lists_->beta_str(h ^ symmetry_, add_Ib);

        std::vector<bool> Ia(nact, false);
        std::vector<bool> Ib(nact, false);

        for (size_t i = 0; i < nact; ++i) {
            if (Ia_v[i])
                Ia[i] = true;
            if (Ib_v[i])
                Ib[i] = true;
        }
        Determinant bsdet(Ia, Ib);
        bsdets.push_back(bsdet);
    }

    // Make sure that the spin space is complete
    //    Determinant det(nact);
    enforce_spin_completeness(bsdets, nact);
    if (bsdets.size() > num_dets) {
        bool* Ia = new bool[nact];
        bool* Ib = new bool[nact];
        size_t nnew_dets = bsdets.size() - num_dets;
        if (print_ > 0) {
            outfile->Printf("\n  Initial guess space is incomplete.\n  Adding "
                            "%d determinant(s).",
                            nnew_dets);
        }
        for (size_t i = 0; i < nnew_dets; ++i) {
            // Find the address of a determinant
            size_t h, add_Ia, add_Ib;
            for (size_t j = 0; j < nact; ++j) {
                Ia[j] = bsdets[num_dets + i].get_alfa_bit(j);
                Ib[j] = bsdets[num_dets + i].get_beta_bit(j);
            }
            h = lists_->alfa_graph()->sym(Ia);
            add_Ia = lists_->alfa_graph()->rel_add(Ia);
            add_Ib = lists_->beta_graph()->rel_add(Ib);
            std::tuple<double, size_t, size_t, size_t> d(0.0, h, add_Ia, add_Ib);
            dets.push_back(d);
        }
        delete[] Ia;
        delete[] Ib;
    }
    num_dets = dets.size();

    Matrix H("H", num_dets, num_dets);
    Matrix evecs("Evecs", num_dets, num_dets);
    Vector evals("Evals", num_dets);

    for (size_t I = 0; I < num_dets; ++I) {
        for (size_t J = I; J < num_dets; ++J) {
            double HIJ = fci_ints->slater_rules(bsdets[I], bsdets[J]);
            if (I == J)
                HIJ += scalar_energy;
            H.set(I, J, HIJ);
            H.set(J, I, HIJ);
        }
    }

    H.diagonalize(evecs, evals);

    std::vector<std::pair<int, std::vector<std::tuple<size_t, size_t, size_t, double>>>> guess;

    std::vector<std::string> s2_labels(
        {"singlet", "doublet", "triplet", "quartet", "quintet", "sextet", "septet", "octet",
         "nonet",   "decaet",  "11-et",   "12-et",   "13-et",   "14-et",  "15-et",  "16-et",
         "17-et",   "18-et",   "19-et",   "20-et",   "21-et",   "22-et",  "23-et",  "24-et"});
    std::vector<std::string> table;

    for (size_t r = 0; r < num_dets; ++r) {
        double energy = evals.get(r) + nuclear_repulsion_energy;
        double norm = 0.0;
        double S2 = 0.0;
        for (size_t I = 0; I < num_dets; ++I) {
            for (size_t J = 0; J < num_dets; ++J) {
                const double S2IJ = spin2(bsdets[I], bsdets[J]);
                S2 += evecs.get(I, r) * evecs.get(J, r) * S2IJ;
            }
            norm += std::pow(evecs.get(I, r), 2.0);
        }
        S2 /= norm;
        double S = std::fabs(0.5 * (std::sqrt(1.0 + 4.0 * S2) - 1.0));
        int SS = std::round(S * 2.0);
        int state_multp = SS + 1;
        std::string state_label = s2_labels[SS];
        table.push_back(boost::str(boost::format("    %3d  %20.12f  %.3f  %s") % r % energy %
                                   std::fabs(S2) % state_label.c_str()));
        // Save states of the desired multiplicity
        std::vector<std::tuple<size_t, size_t, size_t, double>> solution;
        for (size_t I = 0; I < num_dets; ++I) {
            auto det = dets[I];
            double e;
            size_t h, add_Ia, add_Ib;
            std::tie(e, h, add_Ia, add_Ib) = det;
            solution.push_back(std::make_tuple(h, add_Ia, add_Ib, evecs.get(I, r)));
        }
        guess.push_back(std::make_pair(state_multp, solution));
    }
    if (print_) {
        print_h2("FCI Initial Guess");
        outfile->Printf("\n  ---------------------------------------------");
        outfile->Printf("\n    Root            Energy     <S^2>   Spin");
        outfile->Printf("\n  ---------------------------------------------");
        outfile->Printf("\n%s", to_string(table, "\n").c_str());
        outfile->Printf("\n  ---------------------------------------------");
        outfile->Printf("\n  Timing for initial guess  = %10.3f s\n", t.get());
    }

    return guess;
}

std::vector<RDMs> FCISolver::rdms(const std::vector<std::pair<size_t, size_t>>& root_list,
                                  int max_rdm_level) {
    std::vector<RDMs> refs;
    if (max_rdm_level <= 0)
        return refs;

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
        if (max_rdm_level == 1) {
            refs.emplace_back(g1a, g1b);
        }
        if (max_rdm_level == 2) {
            refs.emplace_back(g1a, g1b, g2aa, g2ab, g2bb);
        }
        if (max_rdm_level == 3) {
            refs.emplace_back(g1a, g1b, g2aa, g2ab, g2bb, g3aaa, g3aab, g3abb, g3bbb);
        }
    }
    return refs;
}

std::vector<RDMs>
FCISolver::transition_rdms(const std::vector<std::pair<size_t, size_t>>& root_list,
                           std::shared_ptr<ActiveSpaceMethod> method2, int max_rdm_level) {
    std::vector<RDMs> refs;
    throw std::runtime_error("FCISolver::transition_rdms is not implemented!");
    return refs;
}
} // namespace forte
