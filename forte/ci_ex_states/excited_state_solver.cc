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

#include <fstream>
#include <iomanip>
#include <cmath>
#include <sstream>

#include "psi4/libpsi4util/process.h"
#include "psi4/libpsi4util/PsiOutStream.h"
#include "psi4/physconst.h"

#include "base_classes/mo_space_info.h"
#include "sci/sci.h"
#include "sparse_ci/determinant_substitution_lists.h"
#include "helpers/helpers.h"
#include "helpers/printing.h"
#include "helpers/timer.h"
#include "ci_rdm/ci_rdms.h"

#include "excited_state_solver.h"

#ifdef _OPENMP
#include <omp.h>
#else
#define omp_get_max_threads() 1
#define omp_get_thread_num() 0
#define omp_get_num_threads() 1
#endif

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

    // set a default
    if ((nroot_ > 1) and (ex_alg_ == "NONE")) {
        ex_alg_ = "ROOT_ORTHOGONALIZE";
    }

    core_ex_ = options->get_bool("SCI_CORE_EX");
    quiet_ = options->get_bool("SCI_QUIET_MODE");
    direct_rdms_ = options->get_bool("SCI_DIRECT_RDMS");
    test_rdms_ = options->get_bool("SCI_TEST_RDMS");
    save_final_wfn_ = options->get_bool("SCI_SAVE_FINAL_WFN");
    first_iter_roots_ = options->get_bool("SCI_FIRST_ITER_ROOTS");
    transition_dipole_ = options->get_bool("TRANSITION_DIPOLES");
    sparse_solver_ = std::make_shared<SparseCISolver>();
    sparse_solver_->set_parallel(true);
    sparse_solver_->set_force_diag(options->get_bool("FORCE_DIAG_METHOD"));
    sparse_solver_->set_e_convergence(options->get_double("E_CONVERGENCE"));
    sparse_solver_->set_maxiter_davidson(options->get_int("DL_MAXITER"));
    sparse_solver_->set_spin_project(options->get_bool("SCI_PROJECT_OUT_SPIN_CONTAMINANTS"));
    sparse_solver_->set_spin_project_full(options->get_bool("SCI_PROJECT_OUT_SPIN_CONTAMINANTS"));
    sparse_solver_->set_guess_dimension(options->get_int("DL_GUESS_SIZE"));
    sparse_solver_->set_num_vecs(options->get_int("N_GUESS_VEC"));
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
         "written by Jeffrey B. Schriber, Tianyuan Zhang, and Francesco A. Evangelista"});
    print_info();
    if (!quiet_) {
        psi::outfile->Printf("\n  Using %d thread(s)", omp_get_max_threads());
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
        if (!quiet_)
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
            if (!quiet_)
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
    //    op_ = sci_->get_op();
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

    std::shared_ptr<DeterminantSubstitutionLists> op_c =
        std::make_shared<DeterminantSubstitutionLists>(as_ints_);

    if (ex_alg_ == "ROOT_COMBINE") {
        psi::outfile->Printf("\n\n  ==> Diagonalizing Final Space <==");
        dim = full_space.size();

        for (size_t n = 0; n < nroot_; ++n) {
            psi::outfile->Printf("\n  Determinants for root %d: %zu", n, sizes[n]);
        }

        psi::outfile->Printf("\n  Size of combined space: %zu", dim);

        size_t max_memory = sci_->max_memory();

        auto sigma_type = sci_->sigma_vector_type();
        auto sigma_vector = make_sigma_vector(full_space, as_ints_, max_memory, sigma_type);
        std::tie(PQ_evals, PQ_evecs) = sparse_solver_->diagonalize_hamiltonian(
            full_space, sigma_vector, nroot_, state_.multiplicity());
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

    // Dump wavefunction when transition dipole is calculated
    if (transition_dipole_) {
        dump_wave_function(wfn_filename_);
    }

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
    spin2_ = sci_->get_PQ_spin2();

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

void ExcitedStateSolver::dump_wave_function(const std::string& filename) {
    std::ofstream file(filename);
    file << "# sCI: " << state_.str() << std::endl;
    file << final_wfn_.size() << " " << nroot_ << std::endl;
    for (size_t I = 0, Isize = final_wfn_.size(); I < Isize; ++I) {
        std::string det_str = str(final_wfn_.get_det(I), nact_);
        file << det_str;
        for (size_t n = 0; n < nroot_; ++n) {
            file << ", " << std::scientific << std::setprecision(12) << evecs_->get(I, n);
        }
        file << std::endl;
    }
    file.close();
}

std::tuple<size_t, std::vector<Determinant>, psi::SharedMatrix>
ExcitedStateSolver::read_wave_function(const std::string& filename) {
    std::string line;
    std::ifstream file(filename);

    if (not file.is_open()) {
        psi::outfile->Printf("\n  sCI Error: Failed to open wave function file!");
        return {0, std::vector<Determinant>(), std::make_shared<psi::Matrix>()};
    }

    // read first line
    std::getline(file, line);
    if (line.find("sCI") == std::string::npos) {
        psi::outfile->Printf("\n  sCI Error: Wave function file not from a previous sCI!");
        std::runtime_error("Failed read wave function: file not generated from sCI.");
    }

    // read second line for number of determinants and number of roots
    std::getline(file, line);
    size_t ndets, nroots;
    std::stringstream ss;
    ss << line;
    ss >> ndets >> nroots;

    std::vector<Determinant> det_space;
    det_space.reserve(ndets);
    auto evecs = std::make_shared<psi::Matrix>("evecs " + filename, ndets, nroots);

    size_t norbs = 0; // number of active orbitals
    size_t I = 0;     // index to keep track of determinant
    std::string delimiter = ", ";

    while (std::getline(file, line)) {
        // get the determinant, format in file: e.g., |220ab002>
        size_t next = line.find(delimiter);
        auto det_str = line.substr(0, next);
        norbs = det_str.size() - 2;

        // form determinant
        String Ia, Ib;
        for (size_t i = 0; i < norbs; ++i) {
            char x = det_str[i + 1];
            if (x == '2' or x == '+') {
                Ia[i] = true;
            }
            if (x == '2' or x == '-') {
                Ib[i] = true;
            }
        }
        det_space.emplace_back(Ia, Ib);

        size_t last = next + 1, n = 0;
        while ((next = line.find(delimiter, last)) != std::string::npos) {
            evecs->set(I, n, std::stod(line.substr(last, next - last)));
            n++;
            last = next + 1;
        }
        evecs->set(I, n, std::stod(line.substr(last)));

        I++;
    }

    return {norbs, det_space, evecs};
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

        if (full_pt2_) {
            psi::outfile->Printf(
                "\n  * Selected-CI Energy Root %3d             = %.12f Eh = %8.4f eV", i,
                abs_energy, exc_energy);
            psi::outfile->Printf(
                "\n  * Selected-CI Energy Root %3d + full EPT2 = %.12f Eh = %8.4f eV", i,
                abs_energy + multistate_pt2_energy_correction_[i],
                exc_energy + pc_hartree2ev * (multistate_pt2_energy_correction_[i] -
                                              multistate_pt2_energy_correction_[0]));
        } else {
            psi::outfile->Printf("\n  * Selected-CI Energy Root %3d        = %.12f Eh = %8.4f eV",
                                 i, abs_energy, exc_energy);
            psi::outfile->Printf("\n  * Selected-CI Energy Root %3d + EPT2 = %.12f Eh = %8.4f eV",
                                 i, abs_energy + multistate_pt2_energy_correction_[i],
                                 exc_energy +
                                     pc_hartree2ev * (multistate_pt2_energy_correction_[i] -
                                                      multistate_pt2_energy_correction_[0]));
        }
    }

    if ((ex_alg_ != "ROOT_ORTHOGONALIZE") or (nroot_ == 1)) {
        psi::outfile->Printf("\n\n  ==> Wavefunction Information <==");

        print_wfn(dets, PQ_evecs, nroot_);
    }
}

void ExcitedStateSolver::print_wfn(DeterminantHashVec& space, std::shared_ptr<psi::Matrix> evecs,
                                   int nroot) {
    std::string state_label;
    std::vector<std::string> s2_labels({"singlet", "doublet", "triplet", "quartet", "quintet",
                                        "sextet", "septet", "octet", "nonet", "decatet"});

    //    std::vector<std::pair<double, double>> spins = compute_spin(space, op, evecs, nroot);

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
                                 str(tmp.get_det(I), nact_).c_str());
        }
        //        state_label = s2_labels[std::round(spins[n].first * 2.0)];
        //        psi::outfile->Printf("\n\n  Spin state for root %zu: S^2 = %5.6f, S = %5.3f,
        //        %s", n,
        //                             spins[n].first, spins[n].second, state_label.c_str());
    }
}

void ExcitedStateSolver::wfn_to_file(DeterminantHashVec& det_space, psi::SharedMatrix evecs,
                                     int root) {

    std::ofstream final_wfn;
    final_wfn.open("sci_final_wfn_" + std::to_string(root) + ".txt");
    const det_hashvec& detmap = det_space.wfn_hash();
    for (size_t I = 0, maxI = detmap.size(); I < maxI; ++I) {
        final_wfn << std::scientific << std::setw(20) << std::setprecision(11)
                  << evecs->get(I, root) << " \t " << str(detmap[I], nact_).c_str() << std::endl;
    }
    final_wfn.close();
}

std::vector<std::shared_ptr<RDMs>>
ExcitedStateSolver::rdms(const std::vector<std::pair<size_t, size_t>>& root_list, int max_rdm_level,
                         RDMsType rdm_type) {

    std::vector<std::shared_ptr<RDMs>> refs;

    for (const auto& root_pair : root_list) {
        refs.push_back(compute_rdms(as_ints_, final_wfn_, evecs_, root_pair.first, root_pair.second,
                                    max_rdm_level, rdm_type));
    }
    return refs;
}

std::vector<std::shared_ptr<RDMs>>
ExcitedStateSolver::transition_rdms(const std::vector<std::pair<size_t, size_t>>& root_list,
                                    std::shared_ptr<ActiveSpaceMethod> method2, int max_rdm_level,
                                    RDMsType rdm_type) {
    if (max_rdm_level > 3 || max_rdm_level < 1) {
        throw std::runtime_error("Invalid max_rdm_level, required 1 <= max_rdm_level <= 3.");
    }

    // read wave function from method2
    size_t norbs2;
    std::vector<Determinant> dets2;
    psi::SharedMatrix evecs2;
    std::tie(norbs2, dets2, evecs2) = method2->read_wave_function(method2->wfn_filename());

    if (norbs2 != size_t(nact_)) {
        throw std::runtime_error("sCI Error: Inconsistent number of active orbitals");
    }

    size_t nroot2 = evecs2->coldim();

    // combine with current set of determinants
    DeterminantHashVec dets(dets2);
    for (const auto& det : final_wfn_) {
        if (not dets.has_det(det))
            dets.add(det);
    }

    // fill in eigen vectors
    size_t ndets = dets.size();
    size_t nroots = nroot_ + nroot2;
    auto evecs = std::make_shared<psi::Matrix>("evecs combined", ndets, nroots);

    for (const auto& det : final_wfn_) {
        for (size_t n = 0; n < nroot_; ++n) {
            evecs->set(dets[det], n, evecs_->get(final_wfn_[det], n));
        }
    }

    for (size_t I = 0, size = dets2.size(); I < size; ++I) {
        const auto& det = dets2[I];
        for (size_t n = 0; n < nroot2; ++n) {
            evecs->set(dets[det], n + nroot_, evecs2->get(I, n));
        }
    }

    std::vector<size_t> dim2(2, nact_);
    std::vector<size_t> dim4(4, nact_);
    std::vector<size_t> dim6(6, nact_);

    // loop over roots and compute the transition RDMs
    std::vector<std::shared_ptr<RDMs>> rdms;
    for (const auto& roots_pair : root_list) {
        size_t root1 = roots_pair.first;
        size_t root2 = roots_pair.second + nroot_;

        CI_RDMS ci_rdms(dets, as_ints_, evecs, root1, root2);
        ci_rdms.set_print(false);

        ambit::Tensor a, b, aa, ab, bb, aaa, aab, abb, bbb;
        ambit::Tensor d1, d2, d3;

        if (rdm_type == RDMsType::spin_dependent) {
            // compute 1-RDM
            a = ambit::Tensor::build(ambit::CoreTensor, "TD1a", dim2);
            b = ambit::Tensor::build(ambit::CoreTensor, "TD1b", dim2);
            ci_rdms.compute_1rdm_op(a.data(), b.data());

            if (max_rdm_level > 1) {
                // compute 2-RDM
                ci_rdms.set_print(true);
                aa = ambit::Tensor::build(ambit::CoreTensor, "TD2aa", dim4);
                ab = ambit::Tensor::build(ambit::CoreTensor, "TD2ab", dim4);
                bb = ambit::Tensor::build(ambit::CoreTensor, "TD2bb", dim4);
                ci_rdms.compute_2rdm_op(aa.data(), ab.data(), bb.data());
            }
            if (max_rdm_level > 2) {
                // compute 3-RDM
                aaa = ambit::Tensor::build(ambit::CoreTensor, "TD3aaa", dim6);
                aab = ambit::Tensor::build(ambit::CoreTensor, "TD3aab", dim6);
                abb = ambit::Tensor::build(ambit::CoreTensor, "TD3abb", dim6);
                bbb = ambit::Tensor::build(ambit::CoreTensor, "TD3bbb", dim6);
                ci_rdms.compute_3rdm_op(aaa.data(), aab.data(), abb.data(), bbb.data());
            }

            if (max_rdm_level == 1) {
                rdms.emplace_back(std::make_shared<RDMsSpinDependent>(a, b));
            } else if (max_rdm_level == 2) {
                rdms.emplace_back(std::make_shared<RDMsSpinDependent>(a, b, aa, ab, bb));
            } else {
                rdms.emplace_back(
                    std::make_shared<RDMsSpinDependent>(a, b, aa, ab, bb, aaa, aab, abb, bbb));
            }
        } else {
            // compute 1-RDM
            d1 = ambit::Tensor::build(ambit::CoreTensor, "TD1", dim2);
            ci_rdms.compute_1rdm_sf_op(d1.data());

            if (max_rdm_level > 1) {
                // compute 2-RDM
                ci_rdms.set_print(true);
                d2 = ambit::Tensor::build(ambit::CoreTensor, "TD2", dim4);
                ci_rdms.compute_2rdm_sf_op(d2.data());
            }
            if (max_rdm_level > 2) {
                // compute 3-RDM
                d3 = ambit::Tensor::build(ambit::CoreTensor, "TD3", dim6);
                ci_rdms.compute_3rdm_sf_op(d3.data());
            }

            if (max_rdm_level == 1) {
                rdms.emplace_back(std::make_shared<RDMsSpinFree>(d1));
            } else if (max_rdm_level == 2) {
                rdms.emplace_back(std::make_shared<RDMsSpinFree>(d1, d2));
            } else {
                rdms.emplace_back(std::make_shared<RDMsSpinFree>(d1, d2, d3));
            }
        }
    }

    return rdms;
}

std::shared_ptr<RDMs>
ExcitedStateSolver::compute_rdms(std::shared_ptr<ActiveSpaceIntegrals> fci_ints,
                                 DeterminantHashVec& dets, psi::SharedMatrix& PQ_evecs, int root1,
                                 int root2, int max_rdm_level, RDMsType rdm_type) {

    // TODO: this code might be OBSOLETE (Francesco)
    if (!direct_rdms_) {
        auto op = std::make_shared<DeterminantSubstitutionLists>(fci_ints);
        if (sci_->sigma_vector_type() == SigmaVectorType::Dynamic) {
            op->build_strings(dets);
        }
        op->op_s_lists(dets);

        //        if (max_rdm_level >= 2) {
        op->tp_s_lists(dets);
        //        }

        if (max_rdm_level >= 3) {
            psi::outfile->Printf("\n  Computing 3-list...    ");
            local_timer l3;
            op->three_s_lists(final_wfn_);
            psi::outfile->Printf(" done (%1.5f s)", l3.get());
        }
    }

    CI_RDMS ci_rdms(dets, fci_ints, PQ_evecs, root1, root2);

    ci_rdms.set_max_rdm(max_rdm_level);

    ambit::Tensor ordm_a, ordm_b;
    ambit::Tensor trdm_aa, trdm_ab, trdm_bb;
    ambit::Tensor trdm_aaa, trdm_aab, trdm_abb, trdm_bbb;
    ambit::Tensor G1, G2, G3;

    std::vector<size_t> dim2{nact_, nact_};
    std::vector<size_t> dim4{nact_, nact_, nact_, nact_};
    std::vector<size_t> dim6{nact_, nact_, nact_, nact_, nact_, nact_};

    if (direct_rdms_) {
        // TODO: Implement order-by-order version of direct algorithm
        if (rdm_type == RDMsType::spin_dependent) {
            ordm_a = ambit::Tensor::build(ambit::CoreTensor, "g1a", dim2);
            ordm_b = ambit::Tensor::build(ambit::CoreTensor, "g1b", dim2);

            trdm_aa = ambit::Tensor::build(ambit::CoreTensor, "g2aa", dim4);
            trdm_ab = ambit::Tensor::build(ambit::CoreTensor, "g2ab", dim4);
            trdm_bb = ambit::Tensor::build(ambit::CoreTensor, "g2bb", dim4);

            trdm_aaa = ambit::Tensor::build(ambit::CoreTensor, "g3aaa", dim6);
            trdm_aab = ambit::Tensor::build(ambit::CoreTensor, "g3aab", dim6);
            trdm_abb = ambit::Tensor::build(ambit::CoreTensor, "g3abb", dim6);
            trdm_bbb = ambit::Tensor::build(ambit::CoreTensor, "g3bbb", dim6);

            ci_rdms.compute_rdms_dynamic(ordm_a.data(), ordm_b.data(), trdm_aa.data(),
                                         trdm_ab.data(), trdm_bb.data(), trdm_aaa.data(),
                                         trdm_aab.data(), trdm_abb.data(), trdm_bbb.data());
            //        print_nos();
        } else {
            G1 = ambit::Tensor::build(ambit::CoreTensor, "G1", dim2);
            G2 = ambit::Tensor::build(ambit::CoreTensor, "G2", dim4);
            G3 = ambit::Tensor::build(ambit::CoreTensor, "G3", dim6);
            ci_rdms.compute_rdms_dynamic_sf(G1.data(), G2.data(), G3.data());
        }
    } else {
        if (rdm_type == RDMsType::spin_dependent) {
            if (max_rdm_level >= 1) {
                local_timer one_r;
                ordm_a = ambit::Tensor::build(ambit::CoreTensor, "g1a", dim2);
                ordm_b = ambit::Tensor::build(ambit::CoreTensor, "g1b", dim2);

                ci_rdms.compute_1rdm_op(ordm_a.data(), ordm_b.data());
                psi::outfile->Printf("\n  1-RDM  took %2.6f s (determinant)", one_r.get());

                //            if (options_->get_bool("ACI_PRINT_NO")) {
                //                print_nos();
                //            }
            }
            if (max_rdm_level >= 2) {
                local_timer two_r;
                trdm_aa = ambit::Tensor::build(ambit::CoreTensor, "g2aa", dim4);
                trdm_ab = ambit::Tensor::build(ambit::CoreTensor, "g2ab", dim4);
                trdm_bb = ambit::Tensor::build(ambit::CoreTensor, "g2bb", dim4);

                ci_rdms.compute_2rdm_op(trdm_aa.data(), trdm_ab.data(), trdm_bb.data());
                psi::outfile->Printf("\n  2-RDMS took %2.6f s (determinant)", two_r.get());
            }
            if (max_rdm_level >= 3) {
                local_timer tr;
                trdm_aaa = ambit::Tensor::build(ambit::CoreTensor, "g3aaa", dim6);
                trdm_aab = ambit::Tensor::build(ambit::CoreTensor, "g3aab", dim6);
                trdm_abb = ambit::Tensor::build(ambit::CoreTensor, "g3abb", dim6);
                trdm_bbb = ambit::Tensor::build(ambit::CoreTensor, "g3bbb", dim6);

                ci_rdms.compute_3rdm_op(trdm_aaa.data(), trdm_aab.data(), trdm_abb.data(),
                                        trdm_bbb.data());
                psi::outfile->Printf("\n  3-RDMs took %2.6f s (determinant)", tr.get());
            }
        } else {
            if (max_rdm_level >= 1) {
                local_timer one_r;
                G1 = ambit::Tensor::build(ambit::CoreTensor, "G1", dim2);
                ci_rdms.compute_1rdm_sf_op(G1.data());
                psi::outfile->Printf("\n  1-RDM  took %2.6f s (determinant)", one_r.get());
            }
            if (max_rdm_level >= 2) {
                local_timer two_r;
                G2 = ambit::Tensor::build(ambit::CoreTensor, "G2", dim4);
                ci_rdms.compute_2rdm_sf_op(G2.data());
                psi::outfile->Printf("\n  2-RDMS took %2.6f s (determinant)", two_r.get());
            }
            if (max_rdm_level >= 3) {
                local_timer three_r;
                G3 = ambit::Tensor::build(ambit::CoreTensor, "G3", dim6);
                ci_rdms.compute_3rdm_sf_op(G3.data());
                psi::outfile->Printf("\n  3-RDMS took %2.6f s (determinant)", three_r.get());
            }
        }
    }
    if (test_rdms_ and rdm_type == RDMsType::spin_dependent) {
        ci_rdms.rdm_test(ordm_a.data(), ordm_b.data(), trdm_aa.data(), trdm_bb.data(),
                         trdm_ab.data(), trdm_aaa.data(), trdm_aab.data(), trdm_abb.data(),
                         trdm_bbb.data());
    }

    std::shared_ptr<RDMs> out;
    if (max_rdm_level == 1) {
        if (rdm_type == RDMsType::spin_dependent)
            out = std::make_shared<RDMsSpinDependent>(ordm_a, ordm_b);
        else
            out = std::make_shared<RDMsSpinFree>(G1);
    } else if (max_rdm_level == 2) {
        if (rdm_type == RDMsType::spin_dependent)
            out = std::make_shared<RDMsSpinDependent>(ordm_a, ordm_b, trdm_aa, trdm_ab, trdm_bb);
        else
            out = std::make_shared<RDMsSpinFree>(G1, G2);
    } else {
        if (rdm_type == RDMsType::spin_dependent)
            out = std::make_shared<RDMsSpinDependent>(ordm_a, ordm_b, trdm_aa, trdm_ab, trdm_bb,
                                                      trdm_aaa, trdm_aab, trdm_abb, trdm_bbb);
        else
            out = std::make_shared<RDMsSpinFree>(G1, G2, G3);
    }
    return out;
}

void ExcitedStateSolver::save_old_root(DeterminantHashVec& dets, psi::SharedMatrix& PQ_evecs,
                                       int root, int ref_root) {
    std::vector<std::pair<Determinant, double>> vec;

    if (!quiet_ and nroot_ > 0) {
        psi::outfile->Printf("\n  Saving root %d, ref_root is %d", root, ref_root);
    }
    const det_hashvec& detmap = dets.wfn_hash();
    for (size_t i = 0, max_i = detmap.size(); i < max_i; ++i) {
        vec.push_back(std::make_pair(detmap[i], PQ_evecs->get(i, ref_root)));
    }
    old_roots_.push_back(vec);
    if (!quiet_ and nroot_ > 0) {
        psi::outfile->Printf("\n  Number of old roots: %zu", old_roots_.size());
    }
}

ExcitedStateSolver::~ExcitedStateSolver() {
    // remove wave function file if calculated transition dipole
    if (transition_dipole_) {
        if (std::remove(wfn_filename_.c_str()) != 0) {
            psi::outfile->Printf("\n  sCI wave function %s not available.", state_.str().c_str());
            std::perror("Error when deleting sCI wave function. See output file.");
        }
    }
}

void ExcitedStateSolver::set_excitation_algorithm(std::string ex_alg) { ex_alg_ = ex_alg; }

void ExcitedStateSolver::set_core_excitation(bool core_ex) { core_ex_ = core_ex; }
} // namespace forte
