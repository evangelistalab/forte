#include <algorithm>

#include "psi4/libpsi4util/PsiOutStream.h"
#include "psi4/libpsi4util/process.h"
#include "psi4/libmints/molecule.h"
#include "psi4/libmints/integral.h"
#include "psi4/libmints/basisset.h"
#include "psi4/libpsio/psio.hpp"

#include "helpers/timer.h"
#include "helpers/printing.h"
#include "sparse_ci/ci_reference.h"
#include "detci.h"

using namespace psi;
using namespace ambit;

namespace forte {
DETCI::DETCI(StateInfo state, size_t nroot, std::shared_ptr<SCFInfo> scf_info,
             std::shared_ptr<ForteOptions> options, std::shared_ptr<MOSpaceInfo> mo_space_info,
             std::shared_ptr<ActiveSpaceIntegrals> as_ints)
    : ActiveSpaceMethod(state, nroot, mo_space_info, as_ints), scf_info_(scf_info),
      options_(options) {
    if (not quiet_) {
        print_method_banner({"Determinant-Based Configuration Interaction", "Chenyang Li"});
    }

    startup();
}

void DETCI::startup() {
    nirrep_ = mo_space_info_->nirrep();
    nactv_ = mo_space_info_->size("ACTIVE");
    actv_dim_ = mo_space_info_->dimension("ACTIVE");

    multiplicity_ = state_.multiplicity();
    twice_ms_ = state_.twice_ms();
    wfn_irrep_ = state_.irrep();

    state_label_ = state_.multiplicity_label() + " (Ms = " + get_ms_string(twice_ms_) + ") " +
                   state_.irrep_label();

    set_options(options_);

    ci_ref_ = std::make_shared<CI_Reference>(scf_info_, options_, mo_space_info_, as_ints_,
                                             multiplicity_, twice_ms_, wfn_irrep_, state_);
}

void DETCI::set_options(std::shared_ptr<ForteOptions> options) {
    actv_space_type_ = options->get_str("ACTIVE_REF_TYPE");

    e_conv_ = options->get_double("E_CONVERGENCE");
    r_conv_ = options->get_double("R_CONVERGENCE");
    ci_print_threshold_ = options->get_double("FCIMO_PRINT_CIVEC");

    auto dl_maxiter = options->get_int("DL_MAXITER");
    auto de_maxiter = options->get_int("MAXITER");
    maxiter_ = dl_maxiter > de_maxiter ? dl_maxiter : de_maxiter;

    dl_guess_size_ = options->get_int("DL_GUESS_SIZE");

    ncollapse_per_root_ = options->get_int("DL_COLLAPSE_PER_ROOT");
    nsubspace_per_root_ = options->get_int("DL_SUBSPACE_PER_ROOT");

    sigma_vector_type_ = string_to_sigma_vector_type(options->get_str("DIAG_ALGORITHM"));
    sigma_max_memory_ = options_->get_int("SIGMA_VECTOR_MAX_MEMORY");
}

double DETCI::compute_energy() {
    // build determinants
    build_determinant_space();

    // diagonalize Hamiltonian
    diagoanlize_hamiltonian();

    // compute 1RDMs
    compute_1rdms();

    // print CI vectors
    if (not quiet_) {
        print_ci_wfn();
    }

    // compute dipole momemts

    // save wave functions

    // push to psi4 environment
    double energy = energies_[root_];
    psi::Process::environment.globals["CURRENT ENERGY"] = energy;
    psi::Process::environment.globals["DETCI ENERGY"] = energy;
    return energy;
}

void DETCI::build_determinant_space() {
    p_space_.clear();

    if (actv_space_type_ == "GAS") {
        ci_ref_->build_gas_reference(p_space_);
    } else if (actv_space_type_ == "DOCI") {
        ci_ref_->build_doci_reference(p_space_);
    } else {
        ci_ref_->build_cas_reference_full(p_space_);
    }

    auto size = p_space_.size();

    if (size == 0) {
        outfile->Printf("\n  There is no determinant matching the conditions!");
        outfile->Printf("\n  Please check the input (symmetry, multiplicity, etc.)!");
        throw psi::PSIEXCEPTION("No determinant matching the conditions!");
    }

    if (print_ > 2) {
        print_h2("Determinants");
        for (const auto& det : p_space_) {
            outfile->Printf("\n  %s", str(det, nactv_).c_str());
        }
    }
    if (not quiet_) {
        outfile->Printf("\n  Number of determinants (%s): %zu", actv_space_type_.c_str(), size);
    }

    if (size < 100) {
        sigma_vector_type_ = SigmaVectorType::Full;
    }
}

void DETCI::diagoanlize_hamiltonian() {
    timer tdiag("Diagonalize CI Hamiltonian");
    energies_ = std::vector<double>(nroot_);

    print_h2("Diagonalizing Hamiltonian " + state_label_);

    auto solver = prepare_ci_solver();

    DeterminantHashVec detmap(p_space_);
    auto sigma_vector =
        make_sigma_vector(detmap, as_ints_, sigma_max_memory_, sigma_vector_type_);
    std::tie(evals_, evecs_) =
        solver->diagonalize_hamiltonian(detmap, sigma_vector, nroot_, multiplicity_);

    // add energy offset
    double energy_offset = as_ints_->scalar_energy() + as_ints_->nuclear_repulsion_energy();
    for (size_t i = 0; i < nroot_; ++i) {
        evals_->add(i, energy_offset);
        energies_[i] = evals_->get(i);
    }

    outfile->Printf("\n\n  Done diagonalizing Hamiltonian, %.3e seconds.", tdiag.stop());
}

std::shared_ptr<SparseCISolver> DETCI::prepare_ci_solver() {
    auto solver = std::make_shared<SparseCISolver>();
    solver->set_parallel(true);
    solver->set_spin_project(true);
    solver->set_print_details(not quiet_);

    solver->set_e_convergence(e_conv_);
    solver->set_r_convergence(r_conv_);
    solver->set_maxiter_davidson(maxiter_);

    solver->set_ncollapse_per_root(ncollapse_per_root_);
    solver->set_nsubspace_per_root(nsubspace_per_root_);

    // TODO check the format of initial_guess_
    solver->set_guess_dimension(dl_guess_size_);
    if (initial_guess_.size() == p_space_.size()) {
        solver->set_initial_guess(initial_guess_);
    }

    // TODO: set projected roots
    if (projected_roots_.size() != 0) {
        solver->set_root_project(true);
        solver->add_bad_states(projected_roots_);
    }

    return solver;
}

void DETCI::compute_1rdms() {
    opdm_a_.resize(nroot_);
    opdm_b_.resize(nroot_);

    for (size_t N = 0; N < nroot_; ++N) {
        CI_RDMS ci_rdms(as_ints_, p_space_, evecs_, N, N);
        std::vector<double> opdm_a, opdm_b;
        ci_rdms.compute_1rdm(opdm_a, opdm_b);

        std::string name = "Root " + std::to_string(N) + " of " + state_label_;
        auto da = std::make_shared<psi::Matrix>("D1a " + name, actv_dim_, actv_dim_);
        auto db = std::make_shared<psi::Matrix>("D1b " + name, actv_dim_, actv_dim_);

        for (int h = 0, offset = 0; h < nirrep_; ++h) {
            for (int u = 0; u < actv_dim_[h]; ++u) {
                int nu = u + offset;
                for (int v = 0; v < actv_dim_[h]; ++v) {
                    da->set(h, u, v, opdm_a[nu * nactv_ + v + offset]);
                    db->set(h, u, v, opdm_b[nu * nactv_ + v + offset]);
                }
            }
            offset += actv_dim_[h];
        }

        opdm_a_[N] = da;
        opdm_b_[N] = db;
    }
}

void DETCI::print_ci_wfn() {
    print_h2("CI Vectors & Occupation Number for " + state_label_);

    outfile->Printf("\n  Important determinants with coefficients |C| >= %.3e\n",
                    ci_print_threshold_);

    for (size_t N = 0; N < nroot_; ++N) {
        outfile->Printf("\n  ---- Root No. %d ----\n", N);

        if (nactv_) {
            // select coefficients greater than threshold
            std::vector<int> id;
            for (size_t i = 0, size = p_space_.size(); i < size; ++i) {
                if (std::fabs(evecs_->get(i, N)) >= ci_print_threshold_)
                    id.push_back(i);
            }

            // sort the selected coefficients
            std::sort(id.begin(), id.end(), [&](int i, int j) {
                return std::fabs(evecs_->get(i, N)) > std::fabs(evecs_->get(j, N));
            });

            // print CI coefficients
            outfile->Printf("\n    ");
            int dash_size = 16;
            for (int h = 0; h < nirrep_; ++h) {
                if (actv_dim_[h] == 0)
                    continue;
                auto label = mo_space_info_->irrep_label(h);
                size_t padding_size = actv_dim_[h] > label.size() ? actv_dim_[h] - label.size() : 0;
                std::string padding(padding_size, ' ');
                outfile->Printf(" %s%s", padding.c_str(), label.c_str());
                dash_size += 1 + padding_size + label.size();
            }
            std::string dash = std::string(dash_size, '-');
            outfile->Printf("%16s\n    %s", "Coefficients", dash.c_str());

            for (size_t i = 0, size = id.size(); i < size; ++i) {
                auto det = p_space_[id[i]];
                double ci = evecs_->get(id[i], N);

                outfile->Printf("\n    ");
                for (int h = 0, offset = 0; h < nirrep_; ++h) {
                    if (actv_dim_[h] == 0)
                        continue;
                    std::string label = mo_space_info_->irrep_label(h);
                    size_t padding_size =
                        actv_dim_[h] < label.size() ? label.size() - actv_dim_[h] : 0;
                    outfile->Printf(" %s", std::string(padding_size, ' ').c_str());
                    for (int k = 0; k < actv_dim_[h]; ++k) {
                        auto nk = k + offset;
                        bool a = det.get_alfa_bit(nk);
                        bool b = det.get_beta_bit(nk);
                        if (a == b) {
                            outfile->Printf("%d", a == 1 ? 2 : 0);
                        } else {
                            outfile->Printf("%c", a == 1 ? 'a' : 'b');
                        }
                    }
                    offset += actv_dim_[h];
                }
                outfile->Printf(" %15.10f", ci);
            }
            outfile->Printf("\n    %s", dash.c_str());

            // sort orbital occupations
            std::vector<std::pair<double, std::pair<int, int>>> vec_irrep_occupation;
            for (int h = 0; h < nirrep_; ++h) {
                for (int u = 0; u < actv_dim_[h]; ++u) {
                    double occ = opdm_a_[N]->get(h, u, u) + opdm_b_[N]->get(h, u, u);
                    vec_irrep_occupation.push_back({occ, {h, u + 1}});
                }
            }
            std::sort(vec_irrep_occupation.begin(), vec_irrep_occupation.end(),
                      std::greater<std::pair<double, std::pair<int, int>>>());

            // print orbital occupations
            outfile->Printf("\n\n    Occupation Numbers:");
            for (int i = 0, size = vec_irrep_occupation.size(); i < size; ++i) {
                if (i % 4 == 0) {
                    outfile->Printf("\n    ");
                }
                const auto& vec = vec_irrep_occupation[i];
                outfile->Printf(" %4d%-3s %11.8f", vec.second.second,
                                mo_space_info_->irrep_label(vec.second.first).c_str(), vec.first);
            }
        }

        outfile->Printf("\n\n    Total Energy:  %.15f", evals_->get(N));
    }
}

std::vector<RDMs> DETCI::rdms(const std::vector<std::pair<size_t, size_t>>& root_list,
                              int max_rdm_level) {
    if (max_rdm_level > 3 || max_rdm_level < 1) {
        throw psi::PSIEXCEPTION("Invalid max_rdm_level, required 1 <= max_rdm_level <= 3.");
    }

    std::vector<RDMs> rdms;

    for (auto& root_pair : root_list) {
        auto root1 = root_pair.first;
        auto root2 = root_pair.second;

        auto D1 = compute_trans_1rdms_sosd(root1, root2);

        if (max_rdm_level == 1) {
            rdms.emplace_back(D1[0], D1[1]);
        } else {
            auto D2 = compute_trans_2rdms_sosd(root1, root2);

            if (max_rdm_level == 2) {
                rdms.emplace_back(D1[0], D1[1], D2[0], D2[1], D2[2]);
            } else {
                auto D3 = compute_trans_3rdms_sosd(root1, root2);
                rdms.emplace_back(D1[0], D1[1], D2[0], D2[1], D2[2], D3[0], D3[1], D3[2], D3[3]);
            }
        }
    }

    return rdms;
}

std::vector<ambit::Tensor> DETCI::compute_trans_1rdms_sosd(int root1, int root2) {
    auto a = ambit::Tensor::build(CoreTensor, "D1a", std::vector<size_t>(2, nactv_));
    auto b = ambit::Tensor::build(CoreTensor, "D1b", std::vector<size_t>(2, nactv_));
    auto& a_data = a.data();
    auto& b_data = b.data();

    if (root1 == root2) { // we have computed this, just fill in the data
        for (int h = 0, offset = 0; h < nirrep_; ++h) {
            for (int u = 0; u < actv_dim_[h]; ++u) {
                int nu = u + offset;
                for (int v = 0; v < actv_dim_[h]; ++v) {
                    a_data[nu * nactv_ + v + offset] = opdm_a_[root1]->get(h, u, v);
                    b_data[nu * nactv_ + v + offset] = opdm_b_[root1]->get(h, u, v);
                }
            }
            offset += actv_dim_[h];
        }
    } else {
        CI_RDMS ci_rdms(as_ints_, p_space_, evecs_, root1, root2);
        ci_rdms.compute_1rdm(a_data, b_data);
    }

    return {a, b};
}

std::vector<ambit::Tensor> DETCI::compute_trans_2rdms_sosd(int root1, int root2) {
    auto aa = ambit::Tensor::build(CoreTensor, "D2aa", std::vector<size_t>(4, nactv_));
    auto ab = ambit::Tensor::build(CoreTensor, "D2ab", std::vector<size_t>(4, nactv_));
    auto bb = ambit::Tensor::build(CoreTensor, "D2bb", std::vector<size_t>(4, nactv_));
    auto& aa_data = aa.data();
    auto& ab_data = ab.data();
    auto& bb_data = bb.data();

    CI_RDMS ci_rdms(as_ints_, p_space_, evecs_, root1, root2);
    ci_rdms.compute_2rdm(aa_data, ab_data, bb_data);

    return {aa, ab, bb};
}

std::vector<ambit::Tensor> DETCI::compute_trans_3rdms_sosd(int root1, int root2) {
    auto aaa = ambit::Tensor::build(CoreTensor, "D3aaa", std::vector<size_t>(6, nactv_));
    auto aab = ambit::Tensor::build(CoreTensor, "D3aab", std::vector<size_t>(6, nactv_));
    auto abb = ambit::Tensor::build(CoreTensor, "D3abb", std::vector<size_t>(6, nactv_));
    auto bbb = ambit::Tensor::build(CoreTensor, "D3bbb", std::vector<size_t>(6, nactv_));
    auto& aaa_data = aaa.data();
    auto& aab_data = aab.data();
    auto& abb_data = abb.data();
    auto& bbb_data = bbb.data();

    if (options_->get_str("THREEPDC") == "MK") {
        CI_RDMS ci_rdms(as_ints_, p_space_, evecs_, root1, root2);
        ci_rdms.compute_3rdm(aaa_data, aab_data, abb_data, bbb_data);
    }

    return {aaa, aab, abb, bbb};
}

std::vector<RDMs> DETCI::transition_rdms(const std::vector<std::pair<size_t, size_t>>& root_list,
                                         std::shared_ptr<ActiveSpaceMethod> method2,
                                         int max_rdm_level) {}

} // namespace forte
