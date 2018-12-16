/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2017 by its authors (see COPYING, COPYING.LESSER, AUTHORS).
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

#include "sparse_ci/determinant.h"
#include "helpers/iterative_solvers.h"

#include "fci_solver.h"
#include "helpers/helpers.h"

#ifdef HAVE_GA
#include <ga.h>
#include <macdecls.h>
#endif

#include "psi4/psi4-dec.h"

using namespace psi;

int fci_debug_level = 4;


namespace forte {

class MOSpaceInfo;

FCISolver::FCISolver(Dimension active_dim, std::vector<size_t> core_mo,
                     std::vector<size_t> active_mo, size_t na, size_t nb, size_t multiplicity,
                     size_t symmetry, std::shared_ptr<ForteIntegrals> ints,
                     std::shared_ptr<MOSpaceInfo> mo_space_info, size_t ntrial_per_root, int print,
                     Options& options)
    : active_dim_(active_dim), core_mo_(core_mo), active_mo_(active_mo), ints_(ints),
      nirrep_(active_dim.n()), symmetry_(symmetry), na_(na), nb_(nb), multiplicity_(multiplicity),
      nroot_(0), ntrial_per_root_(ntrial_per_root), print_(print), mo_space_info_(mo_space_info),
      options_(options) {
    nroot_ = options_.get_int("NROOT");
    startup();
}

FCISolver::FCISolver(Dimension active_dim, std::vector<size_t> core_mo,
                     std::vector<size_t> active_mo, size_t na, size_t nb, size_t multiplicity,
                     size_t symmetry, std::shared_ptr<ForteIntegrals> ints,
                     std::shared_ptr<MOSpaceInfo> mo_space_info, Options& options)
    : active_dim_(active_dim), core_mo_(core_mo), active_mo_(active_mo), ints_(ints),
      nirrep_(active_dim.n()), symmetry_(symmetry), na_(na), nb_(nb), multiplicity_(multiplicity),
      nroot_(0), mo_space_info_(mo_space_info), options_(options) {
    ntrial_per_root_ = options_.get_int("NTRIAL_PER_ROOT");
    print_ = options_.get_int("PRINT");
    startup();
}

void FCISolver::set_max_rdm_level(int value) { max_rdm_level_ = value; }

void FCISolver::set_nroot(int value) { nroot_ = value; }

void FCISolver::set_root(int value) { root_ = value; }

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

/*
 * See Appendix A in J. Comput. Chem. 2001 vol. 22 (13) pp. 1574-1589
 */
double FCISolver::compute_energy() {
    local_timer t;

    double nuclear_repulsion_energy =
        Process::environment.molecule()->nuclear_repulsion_energy({{0, 0, 0}});
    std::shared_ptr<FCIIntegrals> fci_ints;
    if (!provide_integrals_and_restricted_docc_) {
        fci_ints = std::make_shared<FCIIntegrals>(ints_, active_mo_, core_mo_);
        ambit::Tensor tei_active_aa =
            ints_->aptei_aa_block(active_mo_, active_mo_, active_mo_, active_mo_);
        ambit::Tensor tei_active_ab =
            ints_->aptei_ab_block(active_mo_, active_mo_, active_mo_, active_mo_);
        ambit::Tensor tei_active_bb =
            ints_->aptei_bb_block(active_mo_, active_mo_, active_mo_, active_mo_);
        fci_ints->set_active_integrals(tei_active_aa, tei_active_ab, tei_active_bb);
        fci_ints->compute_restricted_one_body_operator();
    } else {
        if (fci_ints_ == nullptr) {
            outfile->Printf("\n You said you would specify integrals and restricted_docc");
            throw PSIEXCEPTION("Need to set the fci_ints in your code");
        } else {
            fci_ints = fci_ints_;
        }
    }

    FCIWfn::allocate_temp_space(lists_, print_);

    FCIWfn Hdiag(lists_, symmetry_);
    C_ = std::make_shared<FCIWfn>(lists_, symmetry_);
    FCIWfn HC(lists_, symmetry_);
    C_->set_print(print_);

    size_t fci_size = Hdiag.size();
    Hdiag.form_H_diagonal(fci_ints);

    SharedVector b(new Vector("b", fci_size));
    SharedVector sigma(new Vector("sigma", fci_size));

    Hdiag.copy_to(sigma);

    DavidsonLiuSolver dls(fci_size, nroot_);
    dls.set_e_convergence(options_.get_double("E_CONVERGENCE"));
    dls.set_print_level(print_);
    dls.set_collapse_per_root(collapse_per_root_);
    dls.set_subspace_per_root(subspace_per_root_);
    dls.startup(sigma);

    size_t guess_size = dls.collapse_size();
    auto guess = initial_guess(Hdiag, guess_size, fci_ints);

    std::vector<int> guess_list;
    for (size_t g = 0; g < guess.size(); ++g) {
        if (guess[g].first == multiplicity_)
            guess_list.push_back(g);
    }

    // number of guess to be used
    size_t nguess = std::min(guess_list.size(), guess_size);

    if (nguess == 0) {
        throw PSIEXCEPTION("\n\n  Found zero FCI guesses with the requested "
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
        outfile->Printf("\n  ==> Diagonalizing Hamiltonian <==\n");
        outfile->Printf("\n  ----------------------------------------");
        outfile->Printf("\n    Iter.      Avg. Energy       Delta_E");
        outfile->Printf("\n  ----------------------------------------");
    }

    double old_avg_energy = 0.0;
    int real_cycle = 1;
    for (int cycle = 0; cycle < fci_iterations_; ++cycle) {
        bool add_sigma = true;
        do {
            dls.get_b(b);
            C_->copy(b);
            C_->Hamiltonian(HC, fci_ints, twoSubstituitionVVOO);
            HC.copy_to(sigma);
            add_sigma = dls.add_sigma(sigma);
        } while (add_sigma);

        converged = dls.update();

        if (converged != SolverStatus::Collapse) {
            double avg_energy = 0.0;
            for (int r = 0; r < nroot_; ++r) {
                avg_energy += dls.eigenvalues()->get(r) + nuclear_repulsion_energy;
            }
            avg_energy /= static_cast<double>(nroot_);
            if (print_) {
                outfile->Printf("\n    %3d  %20.12f  %+.3e", real_cycle, avg_energy,
                                avg_energy - old_avg_energy);
            }
            old_avg_energy = avg_energy;
            real_cycle++;
        }

        if (converged == SolverStatus::Converged)
            break;
    }

    if (print_) {
        outfile->Printf("\n  ----------------------------------------");
        if (converged == SolverStatus::Converged) {
            outfile->Printf("\n  The Davidson-Liu algorithm converged in %d iterations.",
                            real_cycle);
        }
    }

    if (converged == SolverStatus::NotConverged) {
        outfile->Printf("\n  FCI did not converge!");
        throw PSIEXCEPTION("FCI did not converge. Try increasing FCI_ITERATIONS.");
    }

    // Compute final eigenvectors
    dls.get_results();

    // Copy eigen values and eigen vectors
    eigen_vals_ = dls.eigenvalues();
    eigen_vecs_ = dls.eigenvectors();

    // Print determinants
    if (print_) {
        for (int r = 0; r < nroot_; ++r) {
            outfile->Printf("\n\n  ==> Root No. %d <==\n", r);

            C_->copy(dls.eigenvector(r));
            std::vector<std::tuple<double, double, size_t, size_t, size_t>> dets_config =
                C_->max_abs_elements(guess_size * ntrial_per_root_);
            Dimension nactvpi = mo_space_info_->get_dimension("ACTIVE");

            for (auto& det_config : dets_config) {
                double ci_abs, ci;
                size_t h, add_Ia, add_Ib;
                std::tie(ci_abs, ci, h, add_Ia, add_Ib) = det_config;

                if (ci_abs < 0.1)
                    continue;

                std::bitset<Determinant::num_str_bits> Ia_v = lists_->alfa_str(h, add_Ia);
                std::bitset<Determinant::num_str_bits> Ib_v =
                    lists_->beta_str(h ^ symmetry_, add_Ib);

                outfile->Printf("\n    ");
                size_t offset = 0;
                for (int h = 0; h < nirrep_; ++h) {
                    for (int k = 0; k < nactvpi[h]; ++k) {
                        size_t i = k + offset;
                        bool a = Ia_v[i];
                        bool b = Ib_v[i];
                        if (a == b) {
                            outfile->Printf("%d", a ? 2 : 0);
                        } else {
                            outfile->Printf("%c", a ? 'a' : 'b');
                        }
                    }
                    if (nactvpi[h] != 0)
                        outfile->Printf(" ");
                    offset += nactvpi[h];
                }
                outfile->Printf("%15.8f", ci);
            }

            double root_energy = dls.eigenvalues()->get(r) + nuclear_repulsion_energy;
            outfile->Printf("\n\n    Total Energy: %25.15f", root_energy);
        }
    }

    // Compute the RDMs
    compute_rdms_root(root_);
    //    C_->copy(dls.eigenvector(root_));
    //    if (print_) {
    //        std::string title_rdm = "Computing RDMs for Root No. " + std::to_string(root_);
    //        print_h2(title_rdm);
    //    }
    //    C_->compute_rdms(max_rdm_level_);

    if (print_ > 1 && max_rdm_level_ > 1) {
        C_->energy_from_rdms(fci_ints);
    }

    //    // Optionally, test the RDMs
    //    if (test_rdms_) {
    //        C_->rdm_test();
    //    }

    //    // Print the NO if energy converged
    //    if (print_no_ || print_ > 0) {
    //        C_->print_natural_orbitals(mo_space_info_);
    //    }

    energy_ = dls.eigenvalues()->get(root_) + nuclear_repulsion_energy;

    return energy_;
}

void FCISolver::compute_rdms_root(int root) {
    // make sure a compute_energy is called before this
    if (C_) {
        if (root >= nroot_) {
            std::string error = "Cannot compute RDMs of root " + std::to_string(root) +
                                "(0-based) because nroot = " + std::to_string(nroot_);
            throw PSIEXCEPTION(error);
        }

        SharedVector evec(eigen_vecs_->get_row(0, root));
        C_->copy(evec);
        if (print_) {
            std::string title_rdm = "Computing RDMs for Root No. " + std::to_string(root);
            print_h2(title_rdm);
        }
        C_->compute_rdms(max_rdm_level_);

        // Optionally, test the RDMs
        if (test_rdms_) {
            C_->rdm_test();
        }

        // Print the NO if energy converged
        if (print_no_ || print_ > 0) {
            C_->print_natural_orbitals(mo_space_info_);
        }
    } else {
        throw PSIEXCEPTION("FCIWfn is not assigned. Cannot compute RDMs.");
    }
}

std::vector<std::pair<int, std::vector<std::tuple<size_t, size_t, size_t, double>>>>
FCISolver::initial_guess(FCIWfn& diag, size_t n, std::shared_ptr<FCIIntegrals> fci_ints) {
    local_timer t;

    double nuclear_repulsion_energy =
        Process::environment.molecule()->nuclear_repulsion_energy({{0, 0, 0}});
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
        std::bitset<Determinant::num_str_bits> Ia_v = lists_->alfa_str(h, add_Ia);
        std::bitset<Determinant::num_str_bits> Ib_v = lists_->beta_str(h ^ symmetry_, add_Ib);

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

Reference FCISolver::reference() {
    size_t nact = active_dim_.sum();
    size_t nact2 = nact * nact;
    size_t nact3 = nact2 * nact;
    size_t nact4 = nact3 * nact;
    size_t nact5 = nact4 * nact;

    Reference fci_ref;
    fci_ref.set_Eref(energy_);

    if (max_rdm_level_ >= 1) {
        // One-particle density matrices in the active space
        std::vector<double>& opdm_a = C_->opdm_a();
        std::vector<double>& opdm_b = C_->opdm_b();
        ambit::Tensor L1a = ambit::Tensor::build(ambit::CoreTensor, "L1a", {nact, nact});
        ambit::Tensor L1b = ambit::Tensor::build(ambit::CoreTensor, "L1b", {nact, nact});
        if (na_ >= 1) {
            L1a.iterate([&](const std::vector<size_t>& i, double& value) {
                value = opdm_a[i[0] * nact + i[1]];
            });
        }
        if (nb_ >= 1) {
            L1b.iterate([&](const std::vector<size_t>& i, double& value) {
                value = opdm_b[i[0] * nact + i[1]];
            });
        }
        fci_ref.set_L1a(L1a);
        fci_ref.set_L1b(L1b);

        if (max_rdm_level_ >= 2) {
            // Two-particle density matrices in the active space
            ambit::Tensor L2aa =
                ambit::Tensor::build(ambit::CoreTensor, "L2aa", {nact, nact, nact, nact});
            ambit::Tensor L2ab =
                ambit::Tensor::build(ambit::CoreTensor, "L2ab", {nact, nact, nact, nact});
            ambit::Tensor L2bb =
                ambit::Tensor::build(ambit::CoreTensor, "L2bb", {nact, nact, nact, nact});
            ambit::Tensor g2aa =
                ambit::Tensor::build(ambit::CoreTensor, "L2aa", {nact, nact, nact, nact});
            ambit::Tensor g2ab =
                ambit::Tensor::build(ambit::CoreTensor, "L2ab", {nact, nact, nact, nact});
            ambit::Tensor g2bb =
                ambit::Tensor::build(ambit::CoreTensor, "L2bb", {nact, nact, nact, nact});

            if (na_ >= 2) {
                std::vector<double>& tpdm_aa = C_->tpdm_aa();
                L2aa.iterate([&](const std::vector<size_t>& i, double& value) {
                    value = tpdm_aa[i[0] * nact3 + i[1] * nact2 + i[2] * nact + i[3]];
                });
            }
            if ((na_ >= 1) and (nb_ >= 1)) {
                std::vector<double>& tpdm_ab = C_->tpdm_ab();
                L2ab.iterate([&](const std::vector<size_t>& i, double& value) {
                    value = tpdm_ab[i[0] * nact3 + i[1] * nact2 + i[2] * nact + i[3]];
                });
            }
            if (nb_ >= 2) {
                std::vector<double>& tpdm_bb = C_->tpdm_bb();
                L2bb.iterate([&](const std::vector<size_t>& i, double& value) {
                    value = tpdm_bb[i[0] * nact3 + i[1] * nact2 + i[2] * nact + i[3]];
                });
            }
            g2aa.copy(L2aa);
            g2ab.copy(L2ab);
            g2bb.copy(L2bb);

            fci_ref.set_g2aa(g2aa);
            fci_ref.set_g2ab(g2ab);
            fci_ref.set_g2bb(g2bb);

            // Convert the 2-RDMs to 2-RCMs
            L2aa("pqrs") -= L1a("pr") * L1a("qs");
            L2aa("pqrs") += L1a("ps") * L1a("qr");

            L2ab("pqrs") -= L1a("pr") * L1b("qs");

            L2bb("pqrs") -= L1b("pr") * L1b("qs");
            L2bb("pqrs") += L1b("ps") * L1b("qr");

            fci_ref.set_L2aa(L2aa);
            fci_ref.set_L2ab(L2ab);
            fci_ref.set_L2bb(L2bb);

            if (max_rdm_level_ >= 3) {
                // Three-particle density matrices in the active space
                ambit::Tensor L3aaa = ambit::Tensor::build(ambit::CoreTensor, "L3aaa",
                                                           {nact, nact, nact, nact, nact, nact});
                ambit::Tensor L3aab = ambit::Tensor::build(ambit::CoreTensor, "L3aab",
                                                           {nact, nact, nact, nact, nact, nact});
                ambit::Tensor L3abb = ambit::Tensor::build(ambit::CoreTensor, "L3abb",
                                                           {nact, nact, nact, nact, nact, nact});
                ambit::Tensor L3bbb = ambit::Tensor::build(ambit::CoreTensor, "L3bbb",
                                                           {nact, nact, nact, nact, nact, nact});
                if (na_ >= 3) {
                    std::vector<double>& tpdm_aaa = C_->tpdm_aaa();
                    L3aaa.iterate([&](const std::vector<size_t>& i, double& value) {
                        value = tpdm_aaa[i[0] * nact5 + i[1] * nact4 + i[2] * nact3 + i[3] * nact2 +
                                         i[4] * nact + i[5]];
                    });
                }
                if ((na_ >= 2) and (nb_ >= 1)) {
                    std::vector<double>& tpdm_aab = C_->tpdm_aab();
                    L3aab.iterate([&](const std::vector<size_t>& i, double& value) {
                        value = tpdm_aab[i[0] * nact5 + i[1] * nact4 + i[2] * nact3 + i[3] * nact2 +
                                         i[4] * nact + i[5]];
                    });
                }
                if ((na_ >= 1) and (nb_ >= 2)) {
                    std::vector<double>& tpdm_abb = C_->tpdm_abb();
                    L3abb.iterate([&](const std::vector<size_t>& i, double& value) {
                        value = tpdm_abb[i[0] * nact5 + i[1] * nact4 + i[2] * nact3 + i[3] * nact2 +
                                         i[4] * nact + i[5]];
                    });
                }
                if (nb_ >= 3) {
                    std::vector<double>& tpdm_bbb = C_->tpdm_bbb();
                    L3bbb.iterate([&](const std::vector<size_t>& i, double& value) {
                        value = tpdm_bbb[i[0] * nact5 + i[1] * nact4 + i[2] * nact3 + i[3] * nact2 +
                                         i[4] * nact + i[5]];
                    });
                }

                // Convert the 3-RDMs to 3-RCMs
                L3aaa("pqrstu") -= L1a("ps") * L2aa("qrtu");
                L3aaa("pqrstu") += L1a("pt") * L2aa("qrsu");
                L3aaa("pqrstu") += L1a("pu") * L2aa("qrts");

                L3aaa("pqrstu") -= L1a("qt") * L2aa("prsu");
                L3aaa("pqrstu") += L1a("qs") * L2aa("prtu");
                L3aaa("pqrstu") += L1a("qu") * L2aa("prst");

                L3aaa("pqrstu") -= L1a("ru") * L2aa("pqst");
                L3aaa("pqrstu") += L1a("rs") * L2aa("pqut");
                L3aaa("pqrstu") += L1a("rt") * L2aa("pqsu");

                L3aaa("pqrstu") -= L1a("ps") * L1a("qt") * L1a("ru");
                L3aaa("pqrstu") -= L1a("pt") * L1a("qu") * L1a("rs");
                L3aaa("pqrstu") -= L1a("pu") * L1a("qs") * L1a("rt");

                L3aaa("pqrstu") += L1a("ps") * L1a("qu") * L1a("rt");
                L3aaa("pqrstu") += L1a("pu") * L1a("qt") * L1a("rs");
                L3aaa("pqrstu") += L1a("pt") * L1a("qs") * L1a("ru");

                L3aab("pqRstU") -= L1a("ps") * L2ab("qRtU");
                L3aab("pqRstU") += L1a("pt") * L2ab("qRsU");

                L3aab("pqRstU") -= L1a("qt") * L2ab("pRsU");
                L3aab("pqRstU") += L1a("qs") * L2ab("pRtU");

                L3aab("pqRstU") -= L1b("RU") * L2aa("pqst");

                L3aab("pqRstU") -= L1a("ps") * L1a("qt") * L1b("RU");
                L3aab("pqRstU") += L1a("pt") * L1a("qs") * L1b("RU");

                L3abb("pQRsTU") -= L1a("ps") * L2bb("QRTU");

                L3abb("pQRsTU") -= L1b("QT") * L2ab("pRsU");
                L3abb("pQRsTU") += L1b("QU") * L2ab("pRsT");

                L3abb("pQRsTU") -= L1b("RU") * L2ab("pQsT");
                L3abb("pQRsTU") += L1b("RT") * L2ab("pQsU");

                L3abb("pQRsTU") -= L1a("ps") * L1b("QT") * L1b("RU");
                L3abb("pQRsTU") += L1a("ps") * L1b("QU") * L1b("RT");

                L3bbb("pqrstu") -= L1b("ps") * L2bb("qrtu");
                L3bbb("pqrstu") += L1b("pt") * L2bb("qrsu");
                L3bbb("pqrstu") += L1b("pu") * L2bb("qrts");

                L3bbb("pqrstu") -= L1b("qt") * L2bb("prsu");
                L3bbb("pqrstu") += L1b("qs") * L2bb("prtu");
                L3bbb("pqrstu") += L1b("qu") * L2bb("prst");

                L3bbb("pqrstu") -= L1b("ru") * L2bb("pqst");
                L3bbb("pqrstu") += L1b("rs") * L2bb("pqut");
                L3bbb("pqrstu") += L1b("rt") * L2bb("pqsu");

                L3bbb("pqrstu") -= L1b("ps") * L1b("qt") * L1b("ru");
                L3bbb("pqrstu") -= L1b("pt") * L1b("qu") * L1b("rs");
                L3bbb("pqrstu") -= L1b("pu") * L1b("qs") * L1b("rt");

                L3bbb("pqrstu") += L1b("ps") * L1b("qu") * L1b("rt");
                L3bbb("pqrstu") += L1b("pu") * L1b("qt") * L1b("rs");
                L3bbb("pqrstu") += L1b("pt") * L1b("qs") * L1b("ru");

                fci_ref.set_L3aaa(L3aaa);
                fci_ref.set_L3aab(L3aab);
                fci_ref.set_L3abb(L3abb);
                fci_ref.set_L3bbb(L3bbb);

                if (print_ > 1)
                    for (auto L1 : {L1a, L1b}) {
                        outfile->Printf("\n\n** %s **", L1.name().c_str());
                        L1.iterate([&](const std::vector<size_t>& i, double& value) {
                            if (std::fabs(value) > 1.0e-15)
                                outfile->Printf("\n  Lambda [%3lu][%3lu] = %18.15lf", i[0], i[1],
                                                value);
                        });
                    }

                if (print_ > 2)
                    for (auto L2 : {L2aa, L2ab, L2bb}) {
                        outfile->Printf("\n\n** %s **", L2.name().c_str());
                        L2.iterate([&](const std::vector<size_t>& i, double& value) {
                            if (std::fabs(value) > 1.0e-15)
                                outfile->Printf("\n  Lambda "
                                                "[%3lu][%3lu][%3lu][%3lu] = "
                                                "%18.15lf",
                                                i[0], i[1], i[2], i[3], value);
                        });
                    }

                if (print_ > 3)
                    for (auto L3 : {L3aaa, L3aab, L3abb, L3bbb}) {
                        outfile->Printf("\n\n** %s **", L3.name().c_str());
                        L3.iterate([&](const std::vector<size_t>& i, double& value) {
                            if (std::fabs(value) > 1.0e-15)
                                outfile->Printf("\n  Lambda "
                                                "[%3lu][%3lu][%3lu][%3lu][%"
                                                "3lu][%3lu] = %18.15lf",
                                                i[0], i[1], i[2], i[3], i[4], i[5], value);
                        });
                    }
            }
        }
    }

    return fci_ref;
}
} // namespace forte

