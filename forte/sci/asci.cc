/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2022 by its authors (see COPYING, COPYING.LESSER,
 * AUTHORS).
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

#include <algorithm>

#include "psi4/physconst.h"
#include "psi4/libpsi4util/PsiOutStream.h"
#include "psi4/libmints/molecule.h"

#include "base_classes/forte_options.h"
#include "base_classes/scf_info.h"
#include "helpers/printing.h"
#include "helpers/helpers.h"
#include "ci_rdm/ci_rdms.h"
#include "sparse_ci/ci_reference.h"

#include "mrpt2.h"
#include "asci.h"

#ifdef _OPENMP
#include <omp.h>
#else
#define omp_get_max_threads() 1
#define omp_get_thread_num() 0
#define omp_get_num_threads() 1
#endif

using namespace psi;

namespace forte {

bool pairCompDescend(const std::pair<double, Determinant> E1,
                     const std::pair<double, Determinant> E2) {
    return E1.first > E2.first;
}

ASCI::ASCI(StateInfo state, size_t nroot, std::shared_ptr<SCFInfo> scf_info,
           std::shared_ptr<ForteOptions> options, std::shared_ptr<MOSpaceInfo> mo_space_info,
           std::shared_ptr<ActiveSpaceIntegrals> as_ints)
    : SelectedCIMethod(state, nroot, scf_info, options, mo_space_info, as_ints) {
    startup();
}

ASCI::~ASCI() {}

void ASCI::set_fci_ints(std::shared_ptr<ActiveSpaceIntegrals> fci_ints) {
    as_ints_ = fci_ints;
    nuclear_repulsion_energy_ = as_ints_->ints()->nuclear_repulsion_energy();
    set_ints_ = true;
}

void ASCI::pre_iter_preparation() {
    outfile->Printf("\n  Using %d threads", omp_get_max_threads());

    CI_Reference ref(scf_info_, options_, mo_space_info_, as_ints_, multiplicity_, twice_ms_,
                     wavefunction_symmetry_, state_);
    ref.build_reference(initial_reference_);
    P_space_ = initial_reference_;

    if (quiet_mode_) {
        sparse_solver_->set_print_details(false);
    }

    sparse_solver_->set_parallel(true);
    sparse_solver_->set_force_diag(options_->get_bool("FORCE_DIAG_METHOD"));
    sparse_solver_->set_e_convergence(options_->get_double("E_CONVERGENCE"));
    sparse_solver_->set_r_convergence(options_->get_double("R_CONVERGENCE"));
    sparse_solver_->set_maxiter_davidson(options_->get_int("DL_MAXITER"));
    sparse_solver_->set_spin_project_full(options_->get_bool("SPIN_PROJECT_FULL"));
    sparse_solver_->set_spin_project(options_->get_bool("SCI_PROJECT_OUT_SPIN_CONTAMINANTS"));
    sparse_solver_->set_guess_dimension(options_->get_int("DL_GUESS_SIZE"));
    sparse_solver_->set_num_vecs(options_->get_int("N_GUESS_VEC"));
}

void ASCI::startup() {
    // Build the reference determinant and compute its energy
    CI_Reference ref(scf_info_, options_, mo_space_info_, as_ints_, multiplicity_, twice_ms_,
                     wavefunction_symmetry_, state_);
    ref.build_reference(initial_reference_);

    // Read options
    nroot_ = options_->get_int("NROOT");

    // Decide when to compute coupling lists
    build_lists_ = true;
    // The Dynamic algorithm does not need lists
    if (sigma_vector_type_ == SigmaVectorType::Dynamic) {
        build_lists_ = false;
    }

    t_det_ = options_->get_int("ASCI_TDET");
    c_det_ = options_->get_int("ASCI_CDET");
}

void ASCI::print_info() {

    print_method_banner({"ASCI", "written by Jeffrey B. Schriber and Francesco A. Evangelista"});
    outfile->Printf("\n  ==> Reference Information <==\n");
    outfile->Printf("\n  There are %d frozen orbitals.", nfrzc_);
    outfile->Printf("\n  There are %zu active orbitals.\n", nact_);
    // Print a summary
    std::vector<std::pair<std::string, int>> calculation_info{{"Multiplicity", multiplicity_},
                                                              {"Symmetry", wavefunction_symmetry_},
                                                              {"Number of roots", nroot_},
                                                              {"CDet", c_det_},
                                                              {"TDet", t_det_}};

    std::vector<std::pair<std::string, double>> calculation_info_double{
        {"Convergence threshold", options_->get_double("ASCI_E_CONVERGENCE")}};

    std::vector<std::pair<std::string, std::string>> calculation_info_string{
        {"Ms", get_ms_string(twice_ms_)},
        {"Diagonalization algorithm", options_->get_str("DIAG_ALGORITHM")}};

    // Print some information
    outfile->Printf("\n  ==> Calculation Information <==\n");
    outfile->Printf("\n  %s", std::string(65, '-').c_str());
    for (auto& str_dim : calculation_info) {
        outfile->Printf("\n    %-40s %-5d", str_dim.first.c_str(), str_dim.second);
    }
    for (auto& str_dim : calculation_info_double) {
        outfile->Printf("\n    %-40s %8.2e", str_dim.first.c_str(), str_dim.second);
    }
    for (auto& str_dim : calculation_info_string) {
        outfile->Printf("\n    %-40s %s", str_dim.first.c_str(), str_dim.second.c_str());
    }
    outfile->Printf("\n  %s", std::string(65, '-').c_str());
}

void ASCI::diagonalize_P_space() {
    cycle_time_.reset();
    // Step 1. Diagonalize the Hamiltonian in the P space
    num_ref_roots_ = std::min(nroot_, P_space_.size());
    std::string cycle_h = "Cycle " + std::to_string(cycle_);

    follow_ = false;
    if (ex_alg_ == "ROOT_COMBINE" or ex_alg_ == "MULTISTATE" or ex_alg_ == "ROOT_ORTHOGONALIZE") {

        follow_ = true;
    }

    if (!quiet_mode_) {
        print_h2(cycle_h);
        outfile->Printf("\n  Initial P space dimension: %zu", P_space_.size());
    }

    sparse_solver_->manual_guess(false);
    local_timer diag;

    auto sigma_vector = make_sigma_vector(P_space_, as_ints_, max_memory_, sigma_vector_type_);
    std::tie(P_evals_, P_evecs_) = sparse_solver_->diagonalize_hamiltonian(
        P_space_, sigma_vector, num_ref_roots_, multiplicity_);

    if (!quiet_mode_)
        outfile->Printf("\n  Time spent diagonalizing H:   %1.6f s", diag.get());

    // Save ground state energy
    P_energies_.push_back(P_evals_->get(0));

    // Update the reference root if root following
    if (follow_ and num_ref_roots_ > 1 and (cycle_ >= pre_iter_) and cycle_ > 0) {
        ref_root_ = root_follow(P_ref_, P_ref_evecs_, P_space_, P_evecs_, num_ref_roots_);
    }

    // Print the energy
    if (!quiet_mode_) {
        outfile->Printf("\n");
        for (int i = 0; i < num_ref_roots_; ++i) {
            double abs_energy =
                P_evals_->get(i) + nuclear_repulsion_energy_ + as_ints_->scalar_energy();
            double exc_energy = pc_hartree2ev * (P_evals_->get(i) - P_evals_->get(0));
            outfile->Printf("\n    P-space  CI Energy Root %3d        = "
                            "%.12f Eh = %8.4f eV",
                            i, abs_energy, exc_energy);
        }
        outfile->Printf("\n");
    }

    if (!quiet_mode_ and options_->get_bool("ACI_PRINT_REFS"))
        print_wfn(P_space_, P_evecs_, num_ref_roots_);
}

void ASCI::find_q_space() {
    timer find_q("ASCI:Build Model Space");
    local_timer build;
    det_hash<double> V_hash;
    get_excited_determinants_sr(P_evecs_, P_space_, V_hash);

    // This will contain all the determinants
    PQ_space_.clear();
    // Add the P-space determinants and zero the hash
    const det_hashvec& detmap = P_space_.wfn_hash();
    for (det_hashvec::iterator it = detmap.begin(), endit = detmap.end(); it != endit; ++it) {
        V_hash.erase(*it);
    }
    //  PQ_space.swap(P_space);

    outfile->Printf("\n  %s: %zu determinants", "psi::Dimension of the Ref + SD space",
                    V_hash.size());
    outfile->Printf("\n  %s: %f s\n", "Time spent building the external space (default)",
                    build.get());

    local_timer screen;
    // Compute criteria for all dets, store them all
    Determinant zero_det; // <- xsize (nact_);
    std::vector<std::pair<double, Determinant>> F_space(V_hash.size(),
                                                        std::make_pair(0.0, zero_det));

    local_timer build_sort;
    size_t N = 0;
    if (options_->get_str("SCI_EXCITED_ALGORITHM") == "AVERAGE") {
        for (const auto& I : V_hash) {
            double criteria = 0.0;
            for (size_t n = 0; n < nroot_; ++n) {
                double delta = as_ints_->energy(I.first) - P_evals_->get(n);
                double V = I.second;

                criteria += (V / delta);
            }
            criteria /= nroot_;
            F_space[N] = std::make_pair(std::fabs(criteria), I.first);

            N++;
        }
    } else {
        for (const auto& I : V_hash) {
            double delta = as_ints_->energy(I.first) - P_evals_->get(0);
            double V = I.second;

            double criteria = V / delta;
            F_space[N] = std::make_pair(std::fabs(criteria), I.first);

            N++;
        }
    }
    for (const auto& I : detmap) {
        F_space.push_back(std::make_pair(std::fabs(P_evecs_->get(P_space_.get_idx(I), 0)), I));
    }
    outfile->Printf("\n  Time spent building sorting list: %1.6f", build_sort.get());

    local_timer sorter;
    std::sort(F_space.begin(), F_space.end(), pairCompDescend);
    outfile->Printf("\n  Time spent sorting: %1.6f", sorter.get());
    local_timer select;

    size_t maxI = std::min(t_det_, int(F_space.size()));
    for (size_t I = 0; I < maxI; ++I) {
        auto& pair = F_space[I];
        PQ_space_.add(pair.second);
    }
    outfile->Printf("\n  Time spent selecting: %1.6f", select.get());
    outfile->Printf("\n  %s: %zu determinants", "psi::Dimension of the P + Q space",
                    PQ_space_.size());
    outfile->Printf("\n  %s: %f s", "Time spent screening the model space", screen.get());
}

bool ASCI::check_convergence() {
    int nroot = PQ_evals_->dim();

    if (energy_history_.size() == 0) {
        std::vector<double> new_energies;
        double state_n_energy = PQ_evals_->get(0) + nuclear_repulsion_energy_;
        new_energies.push_back(state_n_energy);
        energy_history_.push_back(new_energies);
        return false;
    }

    double old_avg_energy = 0.0;
    double new_avg_energy = 0.0;

    std::vector<double> new_energies;
    std::vector<double> old_energies = energy_history_[energy_history_.size() - 1];
    double state_n_energy = PQ_evals_->get(0) + nuclear_repulsion_energy_;
    new_energies.push_back(state_n_energy);
    new_avg_energy += state_n_energy;
    old_avg_energy += old_energies[0];

    old_avg_energy /= static_cast<double>(nroot);
    new_avg_energy /= static_cast<double>(nroot);

    energy_history_.push_back(new_energies);

    // Check for convergence
    return (std::fabs(new_avg_energy - old_avg_energy) <
            options_->get_double("ASCI_E_CONVERGENCE"));
}

void ASCI::prune_PQ_to_P() {

    // Select the new reference space using the sorted CI coefficients
    P_space_.clear();

    // Create a vector that stores the absolute value of the CI coefficients
    std::vector<std::pair<double, Determinant>> dm_det_list;
    // for (size_t I = 0, max = PQ_space.size(); I < max; ++I){
    const det_hashvec& detmap = PQ_space_.wfn_hash();
    for (size_t i = 0, max_i = detmap.size(); i < max_i; ++i) {
        double criteria = std::fabs(PQ_evecs_->get(i, 0));
        dm_det_list.push_back(std::make_pair(criteria, detmap[i]));
    }

    // Decide which determinants will go in pruned_space
    // Include all determinants such that
    // sum_I |C_I|^2 < tau_p, where the sum runs over all the excluded
    // determinants
    // Sort the CI coefficients in ascending order
    std::sort(dm_det_list.begin(), dm_det_list.end(), pairCompDescend);
    size_t Imax = std::min(c_det_, int(dm_det_list.size()));

    for (size_t I = 0; I < Imax; ++I) {
        P_space_.add(dm_det_list[I].second);
    }
}

void ASCI::print_nos() {
    print_h2("NATURAL ORBITALS");

    CI_RDMS ci_rdm(PQ_space_, as_ints_, PQ_evecs_, 0, 0);
    ci_rdm.set_max_rdm(1);
    std::vector<double> ordm_a_v;
    std::vector<double> ordm_b_v;
    ci_rdm.compute_1rdm_op(ordm_a_v, ordm_b_v);

    std::shared_ptr<psi::Matrix> opdm_a(new psi::Matrix("OPDM_A", nirrep_, nactpi_, nactpi_));
    std::shared_ptr<psi::Matrix> opdm_b(new psi::Matrix("OPDM_B", nirrep_, nactpi_, nactpi_));

    int offset = 0;
    for (size_t h = 0; h < nirrep_; h++) {
        for (int u = 0; u < nactpi_[h]; u++) {
            for (int v = 0; v < nactpi_[h]; v++) {
                opdm_a->set(h, u, v, ordm_a_v[(u + offset) * nact_ + v + offset]);
                opdm_b->set(h, u, v, ordm_b_v[(u + offset) * nact_ + v + offset]);
            }
        }
        offset += nactpi_[h];
    }
    psi::SharedVector OCC_A(new Vector("ALPHA OCCUPATION", nirrep_, nactpi_));
    psi::SharedVector OCC_B(new Vector("BETA OCCUPATION", nirrep_, nactpi_));
    psi::SharedMatrix NO_A(new psi::Matrix(nirrep_, nactpi_, nactpi_));
    psi::SharedMatrix NO_B(new psi::Matrix(nirrep_, nactpi_, nactpi_));

    opdm_a->diagonalize(NO_A, OCC_A, descending);
    opdm_b->diagonalize(NO_B, OCC_B, descending);

    // std::ofstream file;
    // file.open("nos.txt",std::ios_base::app);
    std::vector<std::pair<double, std::pair<int, int>>> vec_irrep_occupation;
    for (size_t h = 0; h < nirrep_; h++) {
        for (int u = 0; u < nactpi_[h]; u++) {
            auto irrep_occ =
                std::make_pair(OCC_A->get(h, u) + OCC_B->get(h, u), std::make_pair(h, u + 1));
            vec_irrep_occupation.push_back(irrep_occ);
            //          file << OCC_A->get(h, u) + OCC_B->get(h, u) << "  ";
        }
    }
    // file << endl;
    // file.close();

    std::sort(vec_irrep_occupation.begin(), vec_irrep_occupation.end(),
              std::greater<std::pair<double, std::pair<int, int>>>());

    size_t count = 0;
    outfile->Printf("\n    ");
    for (auto vec : vec_irrep_occupation) {
        outfile->Printf(" %4d%-4s%11.6f  ", vec.second.second,
                        mo_space_info_->irrep_label(vec.second.first).c_str(), vec.first);
        if (count++ % 3 == 2 && count != vec_irrep_occupation.size())
            outfile->Printf("\n    ");
    }
    outfile->Printf("\n\n");
}

void ASCI::get_excited_determinants_sr(psi::SharedMatrix evecs, DeterminantHashVec& P_space,
                                       det_hash<double>& V_hash) {
    local_timer build;
    size_t max_P = P_space.size();
    const det_hashvec& P_dets = P_space.wfn_hash();
    double screen_thresh_ = options_->get_double("ASCI_PRESCREEN_THRESHOLD");

// Loop over reference determinants
#pragma omp parallel
    {
        size_t num_thread = omp_get_num_threads();
        size_t tid = omp_get_thread_num();
        size_t bin_size = max_P / num_thread;
        bin_size += (tid < (max_P % num_thread)) ? 1 : 0;
        size_t start_idx =
            (tid < (max_P % num_thread))
                ? tid * bin_size
                : (max_P % num_thread) * (bin_size + 1) + (tid - (max_P % num_thread)) * bin_size;
        size_t end_idx = start_idx + bin_size;

        det_hash<double> V_hash_t;
        for (size_t P = start_idx; P < end_idx; ++P) {
            const Determinant& det(P_dets[P]);
            double Cp = evecs->get(P, 0);

            std::vector<int> aocc = det.get_alfa_occ(nact_); // TODO check size
            std::vector<int> bocc = det.get_beta_occ(nact_); // TODO check size
            std::vector<int> avir = det.get_alfa_vir(nact_); // TODO check size
            std::vector<int> bvir = det.get_beta_vir(nact_); // TODO check size

            int noalpha = aocc.size();
            int nobeta = bocc.size();
            int nvalpha = avir.size();
            int nvbeta = bvir.size();
            Determinant new_det(det);
            // Generate alpha excitations
            for (int i = 0; i < noalpha; ++i) {
                int ii = aocc[i];
                for (int a = 0; a < nvalpha; ++a) {
                    int aa = avir[a];
                    if ((mo_symmetry_[ii] ^ mo_symmetry_[aa]) == 0) {
                        double HIJ = as_ints_->slater_rules_single_alpha(det, ii, aa) * Cp;
                        if (std::abs(HIJ) >= screen_thresh_) {
                            new_det = det;
                            new_det.set_alfa_bit(ii, false);
                            new_det.set_alfa_bit(aa, true);
                            V_hash_t[new_det] += HIJ;
                        }
                    }
                }
            }
            // Generate beta excitations
            for (int i = 0; i < nobeta; ++i) {
                int ii = bocc[i];
                for (int a = 0; a < nvbeta; ++a) {
                    int aa = bvir[a];
                    if ((mo_symmetry_[ii] ^ mo_symmetry_[aa]) == 0) {
                        double HIJ = as_ints_->slater_rules_single_beta(det, ii, aa) * Cp;
                        if (std::abs(HIJ) >= screen_thresh_) {
                            new_det = det;
                            new_det.set_beta_bit(ii, false);
                            new_det.set_beta_bit(aa, true);
                            V_hash_t[new_det] += HIJ;
                        }
                    }
                }
            }
            // Generate aa excitations
            for (int i = 0; i < noalpha; ++i) {
                int ii = aocc[i];
                for (int j = i + 1; j < noalpha; ++j) {
                    int jj = aocc[j];
                    for (int a = 0; a < nvalpha; ++a) {
                        int aa = avir[a];
                        for (int b = a + 1; b < nvalpha; ++b) {
                            int bb = avir[b];
                            if ((mo_symmetry_[ii] ^ mo_symmetry_[jj] ^ mo_symmetry_[aa] ^
                                 mo_symmetry_[bb]) == 0) {
                                double HIJ = as_ints_->tei_aa(ii, jj, aa, bb) * Cp;
                                if (std::abs(HIJ) >= screen_thresh_) {
                                    new_det = det;
                                    HIJ *= new_det.double_excitation_aa(ii, jj, aa, bb);
                                    V_hash_t[new_det] += HIJ;
                                }
                            }
                        }
                    }
                }
            }
            // Generate ab excitations
            for (int i = 0; i < noalpha; ++i) {
                int ii = aocc[i];
                for (int j = 0; j < nobeta; ++j) {
                    int jj = bocc[j];
                    for (int a = 0; a < nvalpha; ++a) {
                        int aa = avir[a];
                        for (int b = 0; b < nvbeta; ++b) {
                            int bb = bvir[b];
                            if ((mo_symmetry_[ii] ^ mo_symmetry_[jj] ^ mo_symmetry_[aa] ^
                                 mo_symmetry_[bb]) == 0) {
                                double HIJ = as_ints_->tei_ab(ii, jj, aa, bb) * Cp;
                                if (std::abs(HIJ) >= screen_thresh_) {
                                    new_det = det;
                                    HIJ *= new_det.double_excitation_ab(ii, jj, aa, bb);
                                    V_hash_t[new_det] += HIJ;
                                }
                            }
                        }
                    }
                }
            }

            // Generate bb excitations
            for (int i = 0; i < nobeta; ++i) {
                int ii = bocc[i];
                for (int j = i + 1; j < nobeta; ++j) {
                    int jj = bocc[j];
                    for (int a = 0; a < nvbeta; ++a) {
                        int aa = bvir[a];
                        for (int b = a + 1; b < nvbeta; ++b) {
                            int bb = bvir[b];
                            if ((mo_symmetry_[ii] ^ mo_symmetry_[jj] ^ mo_symmetry_[aa] ^
                                 mo_symmetry_[bb]) == 0) {
                                double HIJ = as_ints_->tei_bb(ii, jj, aa, bb) * Cp;
                                if (std::abs(HIJ) >= screen_thresh_) {
                                    new_det = det;
                                    HIJ *= new_det.double_excitation_bb(ii, jj, aa, bb);
                                    V_hash_t[new_det] += HIJ;
                                }
                            }
                        }
                    }
                }
            }
        }
        if (tid == 0)
            outfile->Printf("\n  Time spent forming F space: %20.6f", build.get());
        local_timer merge_t;
#pragma omp critical
        {
            for (auto& pair : V_hash_t) {
                const Determinant& det = pair.first;
                V_hash[det] += pair.second;
            }
        }
        if (tid == 0)
            outfile->Printf("\n  Time spent merging thread F spaces: %20.6f", merge_t.get());
    } // Close threads
}

DeterminantHashVec ASCI::get_PQ_space() { return PQ_space_; }

psi::SharedMatrix ASCI::get_PQ_evecs() { return PQ_evecs_; }
psi::SharedVector ASCI::get_PQ_evals() { return PQ_evals_; }

// std::shared_ptr<WFNOperator> ASCI::get_op() { return op_; }

void ASCI::set_method_variables(
    std::string ex_alg, size_t nroot_method, size_t root,
    const std::vector<std::vector<std::pair<Determinant, double>>>& old_roots) {
    ex_alg_ = ex_alg;
    nroot_ = nroot_method;
    root_ = root;
    ref_root_ = root;
    old_roots_ = old_roots;
}

size_t ASCI::get_ref_root() { return ref_root_; }

std::vector<double> ASCI::get_multistate_pt2_energy_correction() {
    multistate_pt2_energy_correction_.resize(nroot_, 0.0);
    return multistate_pt2_energy_correction_;
}

int ASCI::root_follow(DeterminantHashVec& P_ref, std::vector<double>& P_ref_evecs,
                      DeterminantHashVec& P_space, psi::SharedMatrix P_evecs, int num_ref_roots) {
    int ndets = P_space.size();
    int max_dim = std::min(ndets, 1000);
    //    int max_dim = ndets;
    int new_root;
    double old_overlap = 0.0;
    DeterminantHashVec P_int;
    std::vector<double> P_int_evecs;

    int max_overlap = std::min(int(P_space.size()), num_ref_roots);

    for (int n = 0; n < max_overlap; ++n) {
        if (!quiet_mode_)
            outfile->Printf("\n\n  Computing overlap for root %d", n);
        double new_overlap = P_ref.overlap(P_ref_evecs, P_space, P_evecs, n);

        new_overlap = std::fabs(new_overlap);
        if (!quiet_mode_) {
            outfile->Printf("\n  Root %d has overlap %f", n, new_overlap);
        }
        // If the overlap is larger, set it as the new root and reference, for
        // now
        if (new_overlap > old_overlap) {

            if (!quiet_mode_) {
                outfile->Printf("\n  Saving reference for root %d", n);
            }
            // Save most important subspace
            new_root = n;
            P_int.subspace(P_space, P_evecs, P_int_evecs, max_dim, n);
            old_overlap = new_overlap;
        }
    }

    // Update the reference P_ref

    P_ref.clear();
    P_ref = P_int;

    P_ref_evecs = P_int_evecs;

    outfile->Printf("\n  Setting reference root to: %d", new_root);

    return new_root;
}

void ASCI::diagonalize_PQ_space() {
    // Step 3. Diagonalize the Hamiltonian in the P + Q space
    local_timer diag_pq;

    auto sigma_vector = make_sigma_vector(PQ_space_, as_ints_, max_memory_, sigma_vector_type_);
    std::tie(PQ_evals_, PQ_evecs_) = sparse_solver_->diagonalize_hamiltonian(
        PQ_space_, sigma_vector, num_ref_roots_, multiplicity_);

    if (!quiet_mode_)
        outfile->Printf("\n  Total time spent diagonalizing H:   %1.6f s", diag_pq.get());

    // Save the solutions for the next iteration
    //        old_dets.clear();
    //        old_dets = PQ_space_;
    //        old_evecs = PQ_evecs->clone();

    if (!quiet_mode_) {
        // Print the energy
        outfile->Printf("\n");
        for (int i = 0; i < num_ref_roots_; ++i) {
            double abs_energy =
                PQ_evals_->get(i) + nuclear_repulsion_energy_ + as_ints_->scalar_energy();
            double exc_energy = pc_hartree2ev * (PQ_evals_->get(i) - PQ_evals_->get(0));
            outfile->Printf("\n    PQ-space CI Energy Root %3d        = "
                            "%.12f Eh = %8.4f eV",
                            i, abs_energy, exc_energy);
        }
        outfile->Printf("\n");
    }

    num_ref_roots_ = std::min(nroot_, PQ_space_.size());

    // If doing root-following, grab the initial root
    if (follow_ and ((pre_iter_ == 0 and cycle_ == 0) or cycle_ == (pre_iter_ - 1))) {
        size_t dim = std::min(static_cast<int>(PQ_space_.size()), 1000);
        P_ref_.subspace(PQ_space_, PQ_evecs_, P_ref_evecs_, dim, ref_root_);
    }

    // if( follow and num_ref_roots > 0 and (cycle >= (pre_iter_ - 1)) ){
    if (follow_ and (num_ref_roots_ > 1) and (cycle_ >= pre_iter_)) {
        ref_root_ = root_follow(P_ref_, P_ref_evecs_, PQ_space_, PQ_evecs_, num_ref_roots_);
    }
    print_wfn(PQ_space_, PQ_evecs_, nroot_);
}

void ASCI::post_iter_process() { print_nos(); }

} // namespace forte
