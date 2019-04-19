/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2019 by its authors (see COPYING, COPYING.LESSER, AUTHORS).
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

#include <fstream>
#include <iomanip>
#include <cmath>

#include "psi4/libpsi4util/PsiOutStream.h"
#include "psi4/physconst.h"
#include "excited_state_solver.h"
#include "sci/sci.h"
#include "base_classes/mo_space_info.h"
#include "sparse_ci/operator.h"
#include "helpers/helpers.h"
#include "helpers/printing.h"
#include "ci_rdm/ci_rdms.h"
namespace forte {
ExcitedStateSolver::ExcitedStateSolver(StateInfo state, size_t nroot,
                                       std::shared_ptr<MOSpaceInfo> mo_space_info,
                                       std::shared_ptr<ActiveSpaceIntegrals> as_ints,
                                       std::unique_ptr<SelectedCIMethod> sci)
    : ActiveSpaceMethod(state, nroot, mo_space_info, as_ints), sci_(std::move(sci)) {
    nact_ = mo_space_info_->size("ACTIVE");
}

void ExcitedStateSolver::set_options(std::shared_ptr<ForteOptions> options) {
    // TODO: This shouldn't come from options
    root_ = options->get_int("ROOT");

    full_pt2_ = options->get_bool("FULL_MRPT2");

    // TODO: move all ACI_* options to SCI_* and update register_forte_options.py
    ex_alg_ = options->get_str("SCI_EXCITED_ALGORITHM");
    core_ex_ = options->get_bool("SCI_CORE_EX");
    if (options->has_changed("ACI_QUIET_MODE")) {
        quiet_mode_ = options->get_bool("ACI_QUIET_MODE");
    }
    direct_rdms_ = options->get_bool("SCI_DIRECT_RDMS");
    test_rdms_ = options->get_bool("SCI_TEST_RDMS");
    save_final_wfn_ = options->get_bool("SCI_SAVE_FINAL_WFN");
    first_iter_roots_ = options->get_bool("SCI_FIRST_ITER_ROOTS");
    sparse_solver_ = std::make_shared<SparseCISolver>(as_ints_);
    sparse_solver_->set_parallel(true);
    sparse_solver_->set_force_diag(options->get_bool("FORCE_DIAG_METHOD"));
    sparse_solver_->set_e_convergence(options->get_double("E_CONVERGENCE"));
    sparse_solver_->set_maxiter_davidson(options->get_int("DL_MAXITER"));
    sparse_solver_->set_spin_project(options->get_bool("SCI_PROJECT_OUT_SPIN_CONTAMINANTS"));
    sparse_solver_->set_spin_project_full(options->get_bool("SCI_PROJECT_OUT_SPIN_CONTAMINANTS"));
    sparse_solver_->set_guess_dimension(options->get_int("DL_GUESS_SIZE"));
    sparse_solver_->set_num_vecs(options->get_int("N_GUESS_VEC"));
    sparse_solver_->set_sigma_method(options->get_str("SIGMA_BUILD_TYPE"));
    sparse_solver_->set_max_memory(options->get_int("SIGMA_VECTOR_MAX_MEMORY"));

    sci_->set_options(options);
}

void ExcitedStateSolver::print_info() {

    // Print a summary
    std::vector<std::pair<std::string, int>> calculation_info{
        {"Multiplicity", state_.multiplicity()},
        {"Symmetry", state_.irrep()},
        {"Number of roots", nroot_}};

    std::vector<std::pair<std::string, std::string>> calculation_info_string{
        {"Ms", get_ms_string(state_.twice_ms())}, {"Excited Algorithm", ex_alg_}};

    // Print some information
    psi::outfile->Printf("\n  ==> Calculation Information <==\n");
    psi::outfile->Printf("\n  %s", std::string(65, '-').c_str());
    for (auto& str_dim : calculation_info) {
        psi::outfile->Printf("\n    %-40s %-5d", str_dim.first.c_str(), str_dim.second);
    }
    for (auto& str_dim : calculation_info_string) {
        psi::outfile->Printf("\n    %-40s %s", str_dim.first.c_str(), str_dim.second.c_str());
    }
    psi::outfile->Printf("\n  %s", std::string(65, '-').c_str());
}

double ExcitedStateSolver::compute_energy() {
    timer energy_timer("ExcitedStateSolver:Energy");

    print_method_banner(
        {"Selected Configuration Interaction Excited States",
         "written by Jeffrey B. Schriber, Tianyuan Zhang and Francesco A. Evangelista"});
    psi::outfile->Printf("\n  ==> Reference Information <==\n");
    print_info();
    if (!quiet_mode_) {
        psi::outfile->Printf("\n  Using %d threads", omp_get_max_threads());
    }

    // Compute wavefunction and energy
    size_t dim;
    int nrun = 1;
    bool multi_state = false;

    if (core_ex_) {
        ex_alg_ = "ROOT_ORTHOGONALIZE";
    }

    if (ex_alg_ == "ROOT_COMBINE" or ex_alg_ == "MULTISTATE" or ex_alg_ == "ROOT_ORTHOGONALIZE") {
        nrun = nroot_;
        multi_state = true;
    }

    int ref_root = root_;

    DeterminantHashVec full_space;
    std::vector<size_t> sizes(nroot_);
    psi::SharedVector energies(new psi::Vector(nroot_));
    std::vector<double> pt2_energies(nroot_);

    // The eigenvalues and eigenvectors
    DeterminantHashVec PQ_space;
    psi::SharedMatrix PQ_evecs;
    psi::SharedVector PQ_evals;

    for (int i = 0; i < nrun; ++i) {
        if (!quiet_mode_)
            psi::outfile->Printf("\n  Computing wavefunction for root %d", i);

        if (multi_state) {
            root_ = i;
            ref_root = i;
        }

        if (core_ex_ and (i > 0)) {
            ref_root = i - 1;
        }

        size_t nroot_method = nroot_;

        if (multi_state and ref_root == 0 and !first_iter_roots_) {
            nroot_method = 1;
        }

        sci_->set_method_variables(ex_alg_, nroot_method, root_, old_roots_);

        sci_->compute_energy();

        PQ_space = sci_->get_PQ_space();
        PQ_evecs = sci_->get_PQ_evecs();
        PQ_evals = sci_->get_PQ_evals();
        ref_root = sci_->get_ref_root();

        if (ex_alg_ == "ROOT_COMBINE") {
            sizes[i] = PQ_space.size();
            if (!quiet_mode_)
                psi::outfile->Printf("\n  Combining determinant spaces");
            // Combine selected determinants into total space
            full_space.merge(PQ_space);
            PQ_space.clear();
        } else if ((ex_alg_ == "ROOT_ORTHOGONALIZE")) { // and i != (nrun - 1))
            // orthogonalize
            save_old_root(PQ_space, PQ_evecs, i, ref_root);
            energies->set(i, PQ_evals->get(0));
            pt2_energies[i] = sci_->get_multistate_pt2_energy_correction()[0];
        } else if ((ex_alg_ == "MULTISTATE")) {
            // orthogonalize
            save_old_root(PQ_space, PQ_evecs, i, ref_root);
        }
        if (ex_alg_ == "ROOT_ORTHOGONALIZE" and (nroot_ > 1)) {
            root_ = i;
        }
    }
    op_ = sci_->get_op();
    multistate_pt2_energy_correction_ = sci_->get_multistate_pt2_energy_correction();

    dim = PQ_space.size();
    final_wfn_.copy(PQ_space);
    PQ_space.clear();

    int froot = root_;
    if (ex_alg_ == "ROOT_ORTHOGONALIZE") {
        froot = nroot_ - 1;
        multistate_pt2_energy_correction_ = pt2_energies;
        PQ_evals = energies;
    }

    std::vector<int> mo_symmetry = mo_space_info_->symmetry("ACTIVE");
    WFNOperator op_c(mo_symmetry, as_ints_);
    if (ex_alg_ == "ROOT_COMBINE") {
        psi::outfile->Printf("\n\n  ==> Diagonalizing Final Space <==");
        dim = full_space.size();

        for (size_t n = 0; n < nroot_; ++n) {
            psi::outfile->Printf("\n  Determinants for root %d: %zu", n, sizes[n]);
        }

        psi::outfile->Printf("\n  Size of combined space: %zu", dim);

        if (diag_method_ != Dynamic) {
            op_c.build_strings(full_space);
            op_c.op_lists(full_space);
            op_c.tp_lists(full_space);
        }
        sparse_solver_->diagonalize_hamiltonian_map(full_space, op_c, PQ_evals, PQ_evecs, nroot_,
                                                    state_.multiplicity(), diag_method_);
    }

    if (ex_alg_ == "MULTISTATE") {
        local_timer multi;
        compute_multistate(PQ_evals);
        psi::outfile->Printf("\n  Time spent computing multistate solution: %1.5f s", multi.get());
    }

    if (ex_alg_ == "ROOT_COMBINE") {
        print_final(full_space, PQ_evecs, PQ_evals, sci_->get_cycle());
    } else if (ex_alg_ == "ROOT_ORTHOGONALIZE" and nroot_ > 1) {
        print_final(final_wfn_, PQ_evecs, energies, sci_->get_cycle());
    } else {
        print_final(final_wfn_, PQ_evecs, PQ_evals, sci_->get_cycle());
    }
    evecs_ = PQ_evecs;

    double root_energy = PQ_evals->get(froot) + as_ints_->ints()->nuclear_repulsion_energy() +
                         as_ints_->scalar_energy();
    double root_energy_pt2 = root_energy + multistate_pt2_energy_correction_[froot];

    psi::Process::environment.globals["CURRENT ENERGY"] = root_energy;
    psi::Process::environment.globals["ACI ENERGY"] = root_energy;
    psi::Process::environment.globals["ACI+PT2 ENERGY"] = root_energy_pt2;

    // Save final wave function to a file
    if (save_final_wfn_) {
        psi::outfile->Printf("\n  Saving final wave function for root %d", root_);
        wfn_to_file(final_wfn_, PQ_evecs, root_);
    }

    //    psi::outfile->Printf("\n\n  %s: %f s", "Adaptive-CI ran in ", aci_elapse.get());
    psi::outfile->Printf("\n\n  %s: %d", "Saving information for root", root_);

    // Set active space method evals

    energies_.resize(nroot_, 0.0);
    for (size_t n = 0; n < nroot_; ++n) {
        energies_[n] = PQ_evals->get(n) + as_ints_->ints()->nuclear_repulsion_energy() +
                       as_ints_->scalar_energy();
    }

    return PQ_evals->get(root_) + as_ints_->ints()->nuclear_repulsion_energy() +
           as_ints_->scalar_energy();
}

void ExcitedStateSolver::compute_multistate(psi::SharedVector& PQ_evals) {
    psi::outfile->Printf("\n  Computing multistate solution");
    int nroot = old_roots_.size();

    // Form the overlap matrix

    psi::SharedMatrix S(new psi::Matrix(nroot, nroot));
    S->identity();
    for (int A = 0; A < nroot; ++A) {
        std::vector<std::pair<Determinant, double>>& stateA = old_roots_[A];
        size_t ndetA = stateA.size();
        for (int B = 0; B < nroot; ++B) {
            if (A == B)
                continue;
            std::vector<std::pair<Determinant, double>>& stateB = old_roots_[B];
            size_t ndetB = stateB.size();
            double overlap = 0.0;

            for (size_t I = 0; I < ndetA; ++I) {
                Determinant& detA = stateA[I].first;
                for (size_t J = 0; J < ndetB; ++J) {
                    Determinant& detB = stateB[J].first;
                    if (detA == detB) {
                        overlap += stateA[I].second * stateB[J].second;
                    }
                }
            }
            S->set(A, B, overlap);
        }
    }
    // Diagonalize the overlap
    psi::SharedMatrix Sevecs(new psi::Matrix(nroot, nroot));
    psi::SharedVector Sevals(new psi::Vector(nroot));
    S->diagonalize(Sevecs, Sevals);

    // Form symmetric orthogonalization matrix

    psi::SharedMatrix Strans(new psi::Matrix(nroot, nroot));
    psi::SharedMatrix Sint(new psi::Matrix(nroot, nroot));
    psi::SharedMatrix Diag(new psi::Matrix(nroot, nroot));
    Diag->identity();
    for (int n = 0; n < nroot; ++n) {
        Diag->set(n, n, 1.0 / sqrt(Sevals->get(n)));
    }

    Sint->gemm(false, true, 1.0, Diag, Sevecs, 1.0);
    Strans->gemm(false, false, 1.0, Sevecs, Sint, 1.0);

    // Form the Hamiltonian

    psi::SharedMatrix H(new psi::Matrix(nroot, nroot));

#pragma omp parallel for
    for (int A = 0; A < nroot; ++A) {
        std::vector<std::pair<Determinant, double>>& stateA = old_roots_[A];
        size_t ndetA = stateA.size();
        for (int B = A; B < nroot; ++B) {
            std::vector<std::pair<Determinant, double>>& stateB = old_roots_[B];
            size_t ndetB = stateB.size();
            double HIJ = 0.0;
            for (size_t I = 0; I < ndetA; ++I) {
                Determinant& detA = stateA[I].first;
                for (size_t J = 0; J < ndetB; ++J) {
                    Determinant& detB = stateB[J].first;
                    HIJ += as_ints_->slater_rules(detA, detB) * stateA[I].second * stateB[J].second;
                }
            }
            H->set(A, B, HIJ);
            H->set(B, A, HIJ);
        }
    }
    //    H->print();
    H->transform(Strans);

    psi::SharedMatrix Hevecs(new psi::Matrix(nroot, nroot));
    psi::SharedVector Hevals(new psi::Vector(nroot));

    H->diagonalize(Hevecs, Hevals);

    for (int n = 0; n < nroot; ++n) {
        PQ_evals->set(n, Hevals->get(n)); // + nuclear_repulsion_energy_ +
                                          // as_ints_->scalar_energy());
    }

    //    PQ_evals->print();
}

void ExcitedStateSolver::print_final(DeterminantHashVec& dets, psi::SharedMatrix& PQ_evecs,
                                     psi::SharedVector& PQ_evals, size_t cycle) {
    size_t dim = dets.size();
    // Print a summary
    psi::outfile->Printf("\n\n  ==> Excited state solver summary <==\n");

    psi::outfile->Printf("\n  Iterations required:                         %zu", cycle);
    psi::outfile->Printf("\n  Dimension of optimized determinant space:    %zu\n", dim);

    for (size_t i = 0; i < nroot_; ++i) {
        double abs_energy = PQ_evals->get(i) + as_ints_->ints()->nuclear_repulsion_energy() +
                            as_ints_->scalar_energy();
        double exc_energy = pc_hartree2ev * (PQ_evals->get(i) - PQ_evals->get(0));

        if( full_pt2_ ){
            psi::outfile->Printf("\n  * Selected-CI Energy Root %3d             = %.12f Eh = %8.4f eV", i,
                                 abs_energy, exc_energy);
            psi::outfile->Printf("\n  * Selected-CI Energy Root %3d + full EPT2 = %.12f Eh = %8.4f eV", i,
                                 abs_energy + multistate_pt2_energy_correction_[i],
                                 exc_energy +
                                     pc_hartree2ev * (multistate_pt2_energy_correction_[i] -
                                                      multistate_pt2_energy_correction_[0]));
        } else {
            psi::outfile->Printf("\n  * Selected-CI Energy Root %3d        = %.12f Eh = %8.4f eV", i,
                                 abs_energy, exc_energy);
            psi::outfile->Printf("\n  * Selected-CI Energy Root %3d + EPT2 = %.12f Eh = %8.4f eV", i,
                                 abs_energy + multistate_pt2_energy_correction_[i],
                                 exc_energy +
                                     pc_hartree2ev * (multistate_pt2_energy_correction_[i] -
                                                      multistate_pt2_energy_correction_[0]));
        }
    }

    if ((ex_alg_ != "ROOT_ORTHOGONALIZE") or (nroot_ == 1)) {
        psi::outfile->Printf("\n\n  ==> Wavefunction Information <==");

        print_wfn(dets, op_, PQ_evecs, nroot_);
    }
}

void ExcitedStateSolver::print_wfn(DeterminantHashVec& space, WFNOperator& op,
                                   psi::SharedMatrix evecs, int nroot) {
    std::string state_label;
    std::vector<std::string> s2_labels({"singlet", "doublet", "triplet", "quartet", "quintet",
                                        "sextet", "septet", "octet", "nonet", "decatet"});

    std::vector<std::pair<double, double>> spins = compute_spin(space, op, evecs, nroot);

    //    std::vector<std::pair<double, double>> root_spin_vec;

    for (int n = 0; n < nroot; ++n) {
        DeterminantHashVec tmp;
        std::vector<double> tmp_evecs;

        psi::outfile->Printf("\n\n  Most important contributions to root %3d:", n);

        size_t max_dets = std::min(10, evecs->nrow());
        tmp.subspace(space, evecs, tmp_evecs, max_dets, n);

        for (size_t I = 0; I < max_dets; ++I) {
            psi::outfile->Printf("\n  %3zu  %9.6f %.9f  %10zu %s", I, tmp_evecs[I],
                                 tmp_evecs[I] * tmp_evecs[I], space.get_idx(tmp.get_det(I)),
                                 tmp.get_det(I).str(nact_).c_str());
        }
        state_label = s2_labels[std::round(spins[n].first * 2.0)];
        psi::outfile->Printf("\n\n  Spin state for root %zu: S^2 = %5.6f, S = %5.3f, %s", n,
                             spins[n].first, spins[n].second, state_label.c_str());
    }
}

void ExcitedStateSolver::wfn_to_file(DeterminantHashVec& det_space, psi::SharedMatrix evecs,
                                     int root) {

    std::ofstream final_wfn;
    final_wfn.open("sci_final_wfn_" + std::to_string(root) + ".txt");
    const det_hashvec& detmap = det_space.wfn_hash();
    for (size_t I = 0, maxI = detmap.size(); I < maxI; ++I) {
        final_wfn << std::scientific << std::setw(20) << std::setprecision(11)
                  << evecs->get(I, root) << " \t " << detmap[I].str(nact_).c_str() << std::endl;
    }
    final_wfn.close();
}

std::vector<std::pair<double, double>> ExcitedStateSolver::compute_spin(DeterminantHashVec& space,
                                                                        WFNOperator& op,
                                                                        psi::SharedMatrix evecs,
                                                                        int nroot) {
    // WFNOperator op(mo_symmetry_);

    // op.build_strings(space);
    // op.op_lists(space);
    // op.tp_lists(space);

    std::vector<std::pair<double, double>> spin_vec(nroot);
    if (sparse_solver_->sigma_method_ == "HZ") {
        op.clear_op_s_lists();
        op.clear_tp_s_lists();
        op.build_strings(space);
        op.op_lists(space);
        op.tp_lists(space);
    }

    if (diag_method_ == Dynamic) {
        for (size_t n = 0; n < nroot_; ++n) {
            double S2 = op.s2_direct(space, evecs, n);
            double S = std::fabs(0.5 * (std::sqrt(1.0 + 4.0 * S2) - 1.0));
            spin_vec[n] = std::make_pair(S, S2);
        }
    } else {
        for (size_t n = 0; n < nroot_; ++n) {
            double S2 = op.s2(space, evecs, n);
            double S = std::fabs(0.5 * (std::sqrt(1.0 + 4.0 * S2) - 1.0));
            spin_vec[n] = std::make_pair(S, S2);
        }
    }
    return spin_vec;
}

double ExcitedStateSolver::compute_spin_contamination(DeterminantHashVec& space, WFNOperator& op,
                                                      psi::SharedMatrix evecs, int nroot) {
    auto spins = compute_spin(space, op, evecs, nroot);
    double spin_contam = 0.0;
    for (int n = 0; n < nroot; ++n) {
        spin_contam += spins[n].second;
    }
    spin_contam /= static_cast<double>(nroot);
    spin_contam -= (0.25 * (state_.multiplicity() * state_.multiplicity() - 1.0));

    return spin_contam;
}

std::vector<RDMs> ExcitedStateSolver::rdms(const std::vector<std::pair<size_t, size_t>>& root_list,
                                           int max_rdm_level) {

    std::vector<RDMs> refs;

    for (const auto& root_pair : root_list) {

        refs.push_back(compute_rdms(as_ints_, final_wfn_, op_, evecs_, root_pair.first,
                                    root_pair.second, max_rdm_level));
    }
    return refs;
}

std::vector<RDMs>
ExcitedStateSolver::transition_rdms(const std::vector<std::pair<size_t, size_t>>& root_list,
                                    std::shared_ptr<ActiveSpaceMethod> method2, int max_rdm_level) {
    std::vector<RDMs> refs;
    throw std::runtime_error("ExcitedStateSolver::transition_rdms is not implemented!");
    return refs;
}

RDMs ExcitedStateSolver::compute_rdms(std::shared_ptr<ActiveSpaceIntegrals> fci_ints,
                                      DeterminantHashVec& dets, WFNOperator& op,
                                      psi::SharedMatrix& PQ_evecs, int root1, int root2,
                                      int max_rdm_level) {

    if (!direct_rdms_) {
        op.clear_op_s_lists();
        op.clear_tp_s_lists();
        if (diag_method_ == Dynamic) {
            op.build_strings(dets);
        }
        op.op_s_lists(dets);
        op.tp_s_lists(dets);

        if (max_rdm_level >= 3) {
            psi::outfile->Printf("\n  Computing 3-list...    ");
            local_timer l3;
            op_.three_s_lists(final_wfn_);
            psi::outfile->Printf(" done (%1.5f s)", l3.get());
        }
    }

    CI_RDMS ci_rdms(dets, fci_ints, PQ_evecs, root1, root2);

    ci_rdms.set_max_rdm(max_rdm_level);

    ambit::Tensor ordm_a;
    ambit::Tensor ordm_b;
    ambit::Tensor trdm_aa;
    ambit::Tensor trdm_ab;
    ambit::Tensor trdm_bb;
    ambit::Tensor trdm_aaa;
    ambit::Tensor trdm_aab;
    ambit::Tensor trdm_abb;
    ambit::Tensor trdm_bbb;

    if (direct_rdms_) {
        // TODO: Implemente order-by-order version of direct algorithm
        ordm_a = ambit::Tensor::build(ambit::CoreTensor, "g1a", {nact_, nact_});
        ordm_b = ambit::Tensor::build(ambit::CoreTensor, "g1b", {nact_, nact_});

        trdm_aa = ambit::Tensor::build(ambit::CoreTensor, "g2aa", {nact_, nact_, nact_, nact_});
        trdm_ab = ambit::Tensor::build(ambit::CoreTensor, "g2ab", {nact_, nact_, nact_, nact_});
        trdm_bb = ambit::Tensor::build(ambit::CoreTensor, "g2bb", {nact_, nact_, nact_, nact_});

        trdm_aaa = ambit::Tensor::build(ambit::CoreTensor, "g2aaa",
                                        {nact_, nact_, nact_, nact_, nact_, nact_});
        trdm_aab = ambit::Tensor::build(ambit::CoreTensor, "g2aab",
                                        {nact_, nact_, nact_, nact_, nact_, nact_});
        trdm_abb = ambit::Tensor::build(ambit::CoreTensor, "g2abb",
                                        {nact_, nact_, nact_, nact_, nact_, nact_});
        trdm_bbb = ambit::Tensor::build(ambit::CoreTensor, "g2bbb",
                                        {nact_, nact_, nact_, nact_, nact_, nact_});

        ci_rdms.compute_rdms_dynamic(ordm_a.data(), ordm_b.data(), trdm_aa.data(), trdm_ab.data(),
                                     trdm_bb.data(), trdm_aaa.data(), trdm_aab.data(),
                                     trdm_abb.data(), trdm_bbb.data());
        //        print_nos();
    } else {
        if (max_rdm_level >= 1) {
            local_timer one_r;
            ordm_a = ambit::Tensor::build(ambit::CoreTensor, "g1a", {nact_, nact_});
            ordm_b = ambit::Tensor::build(ambit::CoreTensor, "g1b", {nact_, nact_});

            ci_rdms.compute_1rdm(ordm_a.data(), ordm_b.data(), op);
            psi::outfile->Printf("\n  1-RDM  took %2.6f s (determinant)", one_r.get());

            //            if (options_->get_bool("ACI_PRINT_NO")) {
            //                print_nos();
            //            }
        }
        if (max_rdm_level >= 2) {
            local_timer two_r;
            trdm_aa = ambit::Tensor::build(ambit::CoreTensor, "g2aa", {nact_, nact_, nact_, nact_});
            trdm_ab = ambit::Tensor::build(ambit::CoreTensor, "g2ab", {nact_, nact_, nact_, nact_});
            trdm_bb = ambit::Tensor::build(ambit::CoreTensor, "g2bb", {nact_, nact_, nact_, nact_});

            ci_rdms.compute_2rdm(trdm_aa.data(), trdm_ab.data(), trdm_bb.data(), op);
            psi::outfile->Printf("\n  2-RDMS took %2.6f s (determinant)", two_r.get());
        }
        if (max_rdm_level >= 3) {
            local_timer tr;
            trdm_aaa = ambit::Tensor::build(ambit::CoreTensor, "g2aaa",
                                            {nact_, nact_, nact_, nact_, nact_, nact_});
            trdm_aab = ambit::Tensor::build(ambit::CoreTensor, "g2aab",
                                            {nact_, nact_, nact_, nact_, nact_, nact_});
            trdm_abb = ambit::Tensor::build(ambit::CoreTensor, "g2abb",
                                            {nact_, nact_, nact_, nact_, nact_, nact_});
            trdm_bbb = ambit::Tensor::build(ambit::CoreTensor, "g2bbb",
                                            {nact_, nact_, nact_, nact_, nact_, nact_});

            ci_rdms.compute_3rdm(trdm_aaa.data(), trdm_aab.data(), trdm_abb.data(), trdm_bbb.data(),
                                 op);
            psi::outfile->Printf("\n  3-RDMs took %2.6f s (determinant)", tr.get());
        }
    }
    if (test_rdms_) {
        ci_rdms.rdm_test(ordm_a.data(), ordm_b.data(), trdm_aa.data(), trdm_bb.data(),
                         trdm_ab.data(), trdm_aaa.data(), trdm_aab.data(), trdm_abb.data(),
                         trdm_bbb.data());
    }

    if (max_rdm_level == 1) {
        return RDMs(ordm_a, ordm_b);
    }
    if (max_rdm_level == 2) {
        return RDMs(ordm_a, ordm_b, trdm_aa, trdm_ab, trdm_bb);
    }

    return RDMs(ordm_a, ordm_b, trdm_aa, trdm_ab, trdm_bb, trdm_aaa, trdm_aab, trdm_abb, trdm_bbb);
}

void ExcitedStateSolver::save_old_root(DeterminantHashVec& dets, psi::SharedMatrix& PQ_evecs,
                                       int root, int ref_root) {
    std::vector<std::pair<Determinant, double>> vec;

    if (!quiet_mode_ and nroot_ > 0) {
        psi::outfile->Printf("\n  Saving root %d, ref_root is %d", root, ref_root);
    }
    const det_hashvec& detmap = dets.wfn_hash();
    for (size_t i = 0, max_i = detmap.size(); i < max_i; ++i) {
        vec.push_back(std::make_pair(detmap[i], PQ_evecs->get(i, ref_root)));
    }
    old_roots_.push_back(vec);
    if (!quiet_mode_ and nroot_ > 0) {
        psi::outfile->Printf("\n  Number of old roots: %zu", old_roots_.size());
    }
}

void ExcitedStateSolver::set_excitation_algorithm(std::string ex_alg) { ex_alg_ = ex_alg; }

void ExcitedStateSolver::set_core_excitation(bool core_ex) { core_ex_ = core_ex; }

void ExcitedStateSolver::set_quiet(bool quiet) { quiet_mode_ = quiet; }
}
