#include <algorithm>
#include <cstdio>
#include <fstream>
#include <iomanip>

#include "psi4/libpsi4util/PsiOutStream.h"
#include "psi4/libpsi4util/process.h"
#include "psi4/libmints/molecule.h"
#include "psi4/libmints/integral.h"
#include "psi4/libmints/basisset.h"
#include "psi4/libpsio/psio.hpp"

#include "base_classes/rdms.h"
#include "helpers/timer.h"
#include "helpers/printing.h"
#include "helpers/string_algorithms.h"
#include "sparse_ci/ci_reference.h"
#include "sparse_ci/determinant_substitution_lists.h"
#include "detci.h"

using namespace psi;
using namespace ambit;

namespace forte {
DETCI::DETCI(StateInfo state, size_t nroot, std::shared_ptr<SCFInfo> scf_info,
             std::shared_ptr<ForteOptions> options, std::shared_ptr<MOSpaceInfo> mo_space_info,
             std::shared_ptr<ActiveSpaceIntegrals> as_ints)
    : ActiveSpaceMethod(state, nroot, mo_space_info, as_ints), scf_info_(scf_info),
      options_(options) {
    startup();
}

void DETCI::startup() {
    nirrep_ = static_cast<int>(mo_space_info_->nirrep());
    nactv_ = static_cast<int>(mo_space_info_->size("ACTIVE"));
    actv_dim_ = mo_space_info_->dimension("ACTIVE");

    multiplicity_ = state_.multiplicity();
    twice_ms_ = state_.twice_ms();
    wfn_irrep_ = state_.irrep();

    state_label_ = state_.multiplicity_label() + " (Ms = " + get_ms_string(twice_ms_) + ") " +
                   state_.irrep_label();

    sparse_ci_solver_ = std::make_unique<SparseCISolver>();

    actv_space_type_ = options_->get_str("ACTIVE_REF_TYPE");
    exclude_hf_in_cid_ = options_->get_bool("DETCI_CISD_NO_HF");
    build_determinant_space();

    sigma_max_memory_ = options_->get_int("SIGMA_VECTOR_MAX_MEMORY");
    sigma_vector_type_ = string_to_sigma_vector_type(options_->get_str("DIAG_ALGORITHM"));
    if (p_space_.size() < 500 and (not options_->get_bool("FORCE_DIAG_METHOD")))
        sigma_vector_type_ = SigmaVectorType::Full;
    sigma_vector_ = make_sigma_vector(p_space_, as_ints_, sigma_max_memory_, sigma_vector_type_);
    if (sigma_vector_type_ == SigmaVectorType::SparseList)
        sub_lists_ = sigma_vector_->substitution_lists();
    else {
        sub_lists_ = std::make_shared<DeterminantSubstitutionLists>(as_ints_);
        sub_lists_->build_strings(p_space_);
        sub_lists_->op_s_lists(p_space_);
        if (sigma_vector_type_ == SigmaVectorType::Full) {
            sub_lists_->tp_s_lists(p_space_);
        }
    }
}

void DETCI::set_options(std::shared_ptr<ForteOptions> options) {
    e_convergence_ = options->get_double("E_CONVERGENCE");
    r_convergence_ = options->get_double("R_CONVERGENCE");
    ci_print_threshold_ = options->get_double("DETCI_PRINT_CIVEC");

    auto dl_maxiter = options->get_int("DL_MAXITER");
    auto de_maxiter = options->get_int("MAXITER");
    maxiter_ = dl_maxiter > de_maxiter ? dl_maxiter : de_maxiter;

    dl_guess_size_ = options->get_int("DL_GUESS_SIZE");

    ncollapse_per_root_ = options->get_int("DL_COLLAPSE_PER_ROOT");
    nsubspace_per_root_ = options->get_int("DL_SUBSPACE_PER_ROOT");

    read_wfn_guess_ = options_->get_bool("READ_ACTIVE_WFN_GUESS");
    dump_wfn_ = options_->get_bool("DUMP_ACTIVE_WFN");
}

DETCI::~DETCI() {
    // save wave functions
    if (dump_wfn_ and !wfn_filename_.empty())
        dump_wave_function(wfn_filename_);

    // // remove wave function file
    // if (not dump_wfn_) {
    //     if (!wfn_filename_.empty() and std::remove(wfn_filename_.c_str()) != 0) {
    //         outfile->Printf("\n  DETCI wave function %s not available.", state_.str().c_str());
    //         std::perror("Error when deleting DETCI wave function. See output file.");
    //     }
    // }
}

double DETCI::compute_energy() {
    print_h2("General Determinant-Based CI Solver");

    // diagonalize Hamiltonian
    diagonalize_hamiltonian();

    // compute 1RDMs
    compute_1rdms();

    // print CI vectors
    if (not quiet_) {
        print_ci_wfn();
    }

    // push to psi4 environment
    double energy = energies_[root_];
    psi::Process::environment.globals["CURRENT ENERGY"] = energy;
    psi::Process::environment.globals["DETCI ENERGY"] = energy;
    return energy;
}

void DETCI::build_determinant_space() {
    CI_Reference ci_ref(scf_info_, options_, mo_space_info_, as_ints_, multiplicity_, twice_ms_,
                        wfn_irrep_, state_);
    if (actv_space_type_ == "GAS") {
        ci_ref.build_gas_reference(p_space_);
    } else if (actv_space_type_ == "DOCI") {
        ci_ref.build_doci_reference(p_space_);
    } else if (actv_space_type_ == "CAS") {
        ci_ref.build_cas_reference_full(p_space_);
    } else {
        ci_ref.build_ci_reference(p_space_, !exclude_hf_in_cid_);
    }

    auto size = p_space_.size();
    if (size == 0) {
        outfile->Printf("\n  No determinant found that matches the state requested!");
        outfile->Printf("\n  Please check the input (symmetry, multiplicity, etc.)!");
        throw std::runtime_error(
            "No determinant found that matches the state requested!\n  Please check the "
            "input (symmetry and multiplicity of the root, etc.)!");
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
}

void DETCI::diagonalize_hamiltonian() {
    timer tdiag("Diagonalize CI Hamiltonian");
    energies_ = std::vector<double>(nroot_);

    print_h2("Diagonalizing Hamiltonian " + state_label_);

    set_sparse_ci_solver();

    sigma_vector_->set_active_space_ints(as_ints_);

    std::tie(evals_, evecs_) =
        sparse_ci_solver_->diagonalize_hamiltonian(p_space_, sigma_vector_, nroot_, multiplicity_);

    // add energy offset
    double energy_offset = as_ints_->scalar_energy() + as_ints_->nuclear_repulsion_energy();
    for (size_t i = 0; i < nroot_; ++i) {
        evals_->add(i, energy_offset);
        energies_[i] = evals_->get(i);
    }

    // spin
    spin2_ = sparse_ci_solver_->spin();

    outfile->Printf("\n\n  Done diagonalizing Hamiltonian, %.3e seconds.", tdiag.stop());
}

void DETCI::set_sparse_ci_solver() {
    sparse_ci_solver_->set_parallel(true);
    sparse_ci_solver_->set_spin_project(true);
    sparse_ci_solver_->set_print_details(not quiet_);

    sparse_ci_solver_->set_e_convergence(e_convergence_);
    sparse_ci_solver_->set_r_convergence(r_convergence_);
    sparse_ci_solver_->set_maxiter_davidson(maxiter_);
    sparse_ci_solver_->set_die_if_not_converged(die_if_not_converged_);
    sparse_ci_solver_->set_restart(restart_);

    sparse_ci_solver_->set_ncollapse_per_root(ncollapse_per_root_);
    sparse_ci_solver_->set_nsubspace_per_root(nsubspace_per_root_);

    if (read_wfn_guess_) {
        outfile->Printf("\n  Reading wave function from disk as initial guess:");
        std::string status = read_initial_guess(wfn_filename_) ? "Success" : "Failed";
        outfile->Printf(" %s!", status.c_str());
    }

    sparse_ci_solver_->set_guess_dimension(dl_guess_size_);
    if (not initial_guess_.empty()) {
        sparse_ci_solver_->set_initial_guess(initial_guess_);
    }

    if (not projected_roots_.empty()) {
        sparse_ci_solver_->set_root_project(true);
        sparse_ci_solver_->add_bad_states(projected_roots_);
    }
}

void DETCI::compute_1rdms() {
    opdm_a_.resize(nroot_);
    opdm_b_.resize(nroot_);

    for (size_t N = 0; N < nroot_; ++N) {
        CI_RDMS ci_rdms(p_space_, as_ints_, evecs_, N, N);
        std::vector<double> opdm_a, opdm_b;
        ci_rdms.set_print(print_ci_rdms_);
        ci_rdms.compute_1rdm_op(opdm_a, opdm_b, sub_lists_);

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

    outfile->Printf("\n  Important determinants with coefficients |C| >= %.3e",
                    ci_print_threshold_);

    for (size_t N = 0; N < nroot_; ++N) {
        outfile->Printf("\n\n  ---- Root No. %d ----\n", N);

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
                size_t nactv_h = actv_dim_[h];
                size_t padding_size = nactv_h > label.size() ? nactv_h - label.size() : 0;
                std::string padding(padding_size, ' ');
                outfile->Printf(" %s%s", padding.c_str(), label.c_str());
                dash_size += 1 + padding_size + label.size();
            }
            std::string dash = std::string(dash_size, '-');
            outfile->Printf("%16s\n    %s", "Coefficients", dash.c_str());

            for (const int& i : id) {
                auto det = p_space_.get_det(i);
                double ci = evecs_->get(i, N);

                outfile->Printf("\n    ");
                for (int h = 0, offset = 0; h < nirrep_; ++h) {
                    if (actv_dim_[h] == 0)
                        continue;
                    std::string label = mo_space_info_->irrep_label(h);
                    size_t nactv_h = actv_dim_[h];
                    size_t padding_size = nactv_h < label.size() ? label.size() - nactv_h : 0;
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
            for (int i = 0, size = static_cast<int>(vec_irrep_occupation.size()); i < size; ++i) {
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

void DETCI::dump_wave_function(const std::string& filename) {
    timer t_dump("Dump DETCI WFN");
    std::ofstream file(filename);
    file << "# DETCI: " << state_.str() << '\n';
    file << p_space_.size() << " " << nroot_ << '\n';
    file << std::scientific << std::setprecision(10);
    for (size_t I = 0, Isize = p_space_.size(); I < Isize; ++I) {
        file << str(p_space_.get_det(I), nactv_);
        for (size_t n = 0; n < nroot_; ++n) {
            file << "," << std::setw(18) << evecs_->get(I, n);
        }
        file << '\n';
    }
    file.close();
}

std::tuple<size_t, std::vector<Determinant>, psi::SharedMatrix>
DETCI::read_wave_function(const std::string& filename) {
    timer t_read("Read DETCI WFN");
    std::string line;
    std::ifstream file(filename);

    if (not file.is_open()) {
        outfile->Printf("\n  DETCI Error: Failed to open wave function file!");
        return {0, std::vector<Determinant>(), std::make_shared<psi::Matrix>()};
    }

    // read first line
    std::getline(file, line);
    if (line.find("DETCI") == std::string::npos) {
        outfile->Printf("\n  DETCI Error: Wave function file not from a previous DETCI!");
        throw std::runtime_error("Failed read wave function: file not generated from DETCI.");
    }

    // read second line for number of determinants and number of roots
    std::getline(file, line);
    size_t ndets, nroots;
    stringstream ss;
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
        while ((next = line.find(delimiter, last)) != string::npos) {
            evecs->set(I, n, std::stod(line.substr(last, next - last)));
            n++;
            last = next + 1;
        }
        evecs->set(I, n, std::stod(line.substr(last)));

        I++;
    }

    return {norbs, det_space, evecs};
}

bool DETCI::read_initial_guess(const std::string& filename) {
    // read wave function from file
    size_t norbs;
    std::vector<Determinant> dets;
    SharedMatrix evecs;
    std::tie(norbs, dets, evecs) = read_wave_function(filename);

    // failed to open the file or empty file
    auto ndets = dets.size();
    if (ndets == 0)
        return false;

    // inconsistent number of active orbitals
    if (norbs != size_t(nactv_))
        return false;

    // test the number of roots from file are larger or equal than current
    size_t nroots = evecs->coldim();
    if (nroots < nroot_)
        return false;

    // make sure the determinants are in p_space_
    std::vector<double> norms(nroots, 0.0);
    std::vector<size_t> indices;
    for (size_t I = 0; I < ndets; ++I) {
        const auto& det = dets[I];
        if (not p_space_.has_det(det))
            continue;

        for (size_t n = 0; n < nroots; ++n) {
            double c = evecs->get(I, n);
            norms[n] += c * c;
        }

        indices.push_back(I);
    }

    // translate to initial_guess_ format
    initial_guess_.clear();

    for (size_t n = 0; n < nroots; ++n) {
        std::vector<std::pair<size_t, double>> tmp;
        tmp.reserve(indices.size());
        for (const size_t I : indices) {
            const auto& det = dets[I];
            tmp.emplace_back(p_space_[det], evecs->get(I, n) / norms[n]);
        }
        if (not tmp.empty())
            initial_guess_.push_back(tmp);
    }

    return true;
}

std::vector<std::shared_ptr<RDMs>>
DETCI::rdms(const std::vector<std::pair<size_t, size_t>>& root_list, int max_rdm_level,
            RDMsType rdm_type) {
    if (max_rdm_level > 3 || max_rdm_level < 1) {
        throw std::runtime_error("Invalid max_rdm_level, required 1 <= max_rdm_level <= 3.");
    }

    std::vector<std::shared_ptr<RDMs>> rdms;

    // special case for 1-RDMs
    if (max_rdm_level == 1) {
        for (const auto& [root1, root2] : root_list) {
            auto D1 = compute_trans_1rdms_sosd(root1, root2);
            if (rdm_type == RDMsType::spin_dependent) {
                rdms.emplace_back(std::make_shared<RDMsSpinDependent>(D1[0], D1[1]));
            } else {
                auto opdm = D1[0].clone();
                opdm.set_name("D1");
                opdm("pq") += D1[1]("pq");
                rdms.emplace_back(std::make_shared<RDMsSpinFree>(opdm));
            }
        }
        return rdms;
    }

    // special case for dynamic algorithm
    if (sigma_vector_type_ == SigmaVectorType::Dynamic) {
        for (const auto& [root1, root2] : root_list) {
            CI_RDMS ci_rdms(p_space_, as_ints_, evecs_, root1, root2);
            ci_rdms.set_print(print_ci_rdms_);
            rdms.push_back(compute_trans_rdms_sosd_dynamic(ci_rdms, max_rdm_level, rdm_type));
        }
        return rdms;
    }

    // for 2- and 3-RDMs
    for (const auto& [root1, root2] : root_list) {
        auto D1 = compute_trans_1rdms_sosd(root1, root2);

        if (rdm_type == RDMsType::spin_dependent) {
            auto D2 = compute_trans_2rdms_sosd(root1, root2);

            std::vector<ambit::Tensor> D3;
            if (max_rdm_level > 2)
                D3 = compute_trans_3rdms_sosd(root1, root2);

            if (max_rdm_level == 2) {
                rdms.emplace_back(
                    std::make_shared<RDMsSpinDependent>(D1[0], D1[1], D2[0], D2[1], D2[2]));
            } else {
                rdms.emplace_back(std::make_shared<RDMsSpinDependent>(
                    D1[0], D1[1], D2[0], D2[1], D2[2], D3[0], D3[1], D3[2], D3[3]));
            }
        } else {
            CI_RDMS ci_rdms(p_space_, as_ints_, evecs_, root1, root2);
            ci_rdms.set_print(print_ci_rdms_);

            auto G1 = ambit::Tensor::build(CoreTensor, "D1", std::vector<size_t>(2, nactv_));
            G1("pq") += D1[0]("pq");
            G1("pq") += D1[1]("pq");

            auto D2 = ambit::Tensor::build(CoreTensor, "D2", std::vector<size_t>(4, nactv_));
            ci_rdms.compute_2rdm_sf_op(D2.data(), sub_lists_);

            ambit::Tensor D3;
            if (max_rdm_level > 2) {
                D3 = ambit::Tensor::build(CoreTensor, "D3", std::vector<size_t>(6, nactv_));
                ci_rdms.compute_3rdm_sf_op(D3.data());
            }

            if (max_rdm_level == 2) {
                rdms.emplace_back(std::make_shared<RDMsSpinFree>(G1, D2));
            } else {
                rdms.emplace_back(std::make_shared<RDMsSpinFree>(G1, D2, D3));
            }
        }
    }

    return rdms;
}

std::shared_ptr<RDMs> DETCI::compute_trans_rdms_sosd_dynamic(CI_RDMS& ci_rdms, int max_rdm_level,
                                                             RDMsType rdm_type) {
    if (rdm_type == RDMsType::spin_dependent) {
        auto a = ambit::Tensor::build(CoreTensor, "D1a", std::vector<size_t>(2, nactv_));
        auto b = ambit::Tensor::build(CoreTensor, "D1b", std::vector<size_t>(2, nactv_));
        auto& a_data = a.data();
        auto& b_data = b.data();

        auto aa = ambit::Tensor::build(CoreTensor, "D2aa", std::vector<size_t>(4, nactv_));
        auto ab = ambit::Tensor::build(CoreTensor, "D2ab", std::vector<size_t>(4, nactv_));
        auto bb = ambit::Tensor::build(CoreTensor, "D2bb", std::vector<size_t>(4, nactv_));
        auto& aa_data = aa.data();
        auto& ab_data = ab.data();
        auto& bb_data = bb.data();

        if (max_rdm_level < 3) {
            ci_rdms.compute_rdms_dynamic(a_data, b_data, aa_data, ab_data, bb_data);
            return std::make_shared<RDMsSpinDependent>(a, b, aa, ab, bb);
        } else {
            auto aaa = ambit::Tensor::build(CoreTensor, "D3aaa", std::vector<size_t>(6, nactv_));
            auto aab = ambit::Tensor::build(CoreTensor, "D3aab", std::vector<size_t>(6, nactv_));
            auto abb = ambit::Tensor::build(CoreTensor, "D3abb", std::vector<size_t>(6, nactv_));
            auto bbb = ambit::Tensor::build(CoreTensor, "D3bbb", std::vector<size_t>(6, nactv_));
            auto& aaa_data = aaa.data();
            auto& aab_data = aab.data();
            auto& abb_data = abb.data();
            auto& bbb_data = bbb.data();
            ci_rdms.compute_rdms_dynamic(a_data, b_data, aa_data, ab_data, bb_data, aaa_data,
                                         aab_data, abb_data, bbb_data);
            return std::make_shared<RDMsSpinDependent>(a, b, aa, ab, bb, aaa, aab, abb, bbb);
        }
    } else {
        auto opdm = ambit::Tensor::build(CoreTensor, "D1", std::vector<size_t>(2, nactv_));
        auto& opdm_data = opdm.data();

        auto tpdm = ambit::Tensor::build(CoreTensor, "D2", std::vector<size_t>(4, nactv_));
        auto& tpdm_data = tpdm.data();

        if (max_rdm_level < 3) {
            ci_rdms.compute_rdms_dynamic_sf(opdm_data, tpdm_data);
            return std::make_shared<RDMsSpinFree>(opdm, tpdm);
        } else {
            auto t3pdm = ambit::Tensor::build(CoreTensor, "D3", std::vector<size_t>(6, nactv_));
            auto& t3pdm_data = t3pdm.data();
            ci_rdms.compute_rdms_dynamic_sf(opdm_data, tpdm_data, t3pdm_data);
            return std::make_shared<RDMsSpinFree>(opdm, tpdm, t3pdm);
        }
    }
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
        CI_RDMS ci_rdms(p_space_, as_ints_, evecs_, root1, root2);
        ci_rdms.set_print(print_ci_rdms_);
        ci_rdms.compute_1rdm_op(a_data, b_data, sub_lists_);
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

    CI_RDMS ci_rdms(p_space_, as_ints_, evecs_, root1, root2);
    ci_rdms.set_print(print_ci_rdms_);
    ci_rdms.compute_2rdm_op(aa_data, ab_data, bb_data, sub_lists_);

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
        CI_RDMS ci_rdms(p_space_, as_ints_, evecs_, root1, root2);
        ci_rdms.set_print(print_ci_rdms_);
        ci_rdms.compute_3rdm_op(aaa_data, aab_data, abb_data, bbb_data);
    }

    return {aaa, aab, abb, bbb};
}

std::vector<std::shared_ptr<RDMs>>
DETCI::transition_rdms(const std::vector<std::pair<size_t, size_t>>& root_list,
                       std::shared_ptr<ActiveSpaceMethod> method2, int max_rdm_level,
                       RDMsType rdm_type) {
    if (max_rdm_level > 3 || max_rdm_level < 1) {
        throw std::runtime_error("Invalid max_rdm_level, required 1 <= max_rdm_level <= 3.");
    }

    auto ci_other = std::dynamic_pointer_cast<DETCI>(method2);
    if (ci_other->actv_dim() != actv_dim_) {
        throw std::runtime_error("DETCI Error: Inconsistent number of active orbitals");
    }

    auto evecs2 = ci_other->ci_wave_functions();
    auto dets2 = ci_other->determinants();

    // combine with current set of determinants
    DeterminantHashVec dets(dets2);
    for (const auto& det : p_space_) {
        if (not dets.has_det(det))
            dets.add(det);
    }

    // fill in eigen vectors
    size_t ndets = dets.size();
    size_t nroot2 = evecs2->coldim();
    size_t nroots = nroot_ + nroot2;
    auto evecs = std::make_shared<psi::Matrix>("evecs combined", ndets, nroots);

    for (const auto& det : p_space_) {
        for (size_t n = 0; n < nroot_; ++n) {
            evecs->set(dets[det], n, evecs_->get(p_space_[det], n));
        }
    }
    for (const auto& det : dets2) {
        for (size_t n = 0; n < nroot2; ++n) {
            evecs->set(dets[det], n + nroot_, evecs2->get(dets[det], n));
        }
    }

    // loop over roots and compute the transition RDMs
    std::vector<std::shared_ptr<RDMs>> rdms;

    // // special case for dynamic algorithm
    // if (sigma_vector_type_ == SigmaVectorType::Dynamic) {
    //     for (const auto& roots_pair : root_list) {
    //         size_t root1 = roots_pair.first;
    //         size_t root2 = roots_pair.second + nroot_;
    //         CI_RDMS ci_rdms(dets, as_ints_, evecs, root1, root2);
    //         ci_rdms.set_print(false);
    //         rdms.push_back(compute_trans_rdms_sosd_dynamic(ci_rdms, max_rdm_level, rdm_type));
    //     }
    //     return rdms;
    // }

    for (const auto& roots_pair : root_list) {
        size_t root1 = roots_pair.first;
        size_t root2 = roots_pair.second + nroot_;

        CI_RDMS ci_rdms(dets, as_ints_, evecs, root1, root2);
        ci_rdms.set_print(false);

        if (rdm_type == RDMsType::spin_dependent) {
            // compute 1-RDM
            auto a = ambit::Tensor::build(CoreTensor, "TD1a", std::vector<size_t>(2, nactv_));
            auto b = ambit::Tensor::build(CoreTensor, "TD1b", std::vector<size_t>(2, nactv_));
            auto& a_data = a.data();
            auto& b_data = b.data();

            ci_rdms.compute_1rdm_op(a_data, b_data);

            if (max_rdm_level == 1) {
                rdms.emplace_back(std::make_shared<RDMsSpinDependent>(a, b));
            } else {
                // compute 2-RDM
                std::vector<size_t> dim4(4, nactv_);
                auto aa = ambit::Tensor::build(CoreTensor, "TD2aa", dim4);
                auto ab = ambit::Tensor::build(CoreTensor, "TD2ab", dim4);
                auto bb = ambit::Tensor::build(CoreTensor, "TD2bb", dim4);
                auto& aa_data = aa.data();
                auto& ab_data = ab.data();
                auto& bb_data = bb.data();

                ci_rdms.compute_2rdm_op(aa_data, ab_data, bb_data);

                if (max_rdm_level == 2) {
                    rdms.emplace_back(std::make_shared<RDMsSpinDependent>(a, b, aa, ab, bb));
                } else {
                    // compute 3-RDM
                    std::vector<size_t> dim6(6, nactv_);
                    auto aaa = ambit::Tensor::build(CoreTensor, "TD3aaa", dim6);
                    auto aab = ambit::Tensor::build(CoreTensor, "TD3aab", dim6);
                    auto abb = ambit::Tensor::build(CoreTensor, "TD3abb", dim6);
                    auto bbb = ambit::Tensor::build(CoreTensor, "TD3bbb", dim6);
                    auto& aaa_data = aaa.data();
                    auto& aab_data = aab.data();
                    auto& abb_data = abb.data();
                    auto& bbb_data = bbb.data();

                    ci_rdms.compute_3rdm_op(aaa_data, aab_data, abb_data, bbb_data);

                    rdms.emplace_back(
                        std::make_shared<RDMsSpinDependent>(a, b, aa, ab, bb, aaa, aab, abb, bbb));
                }
            }
        } else {
            // compute 1-RDM
            auto D1 = ambit::Tensor::build(CoreTensor, "TD1", std::vector<size_t>(2, nactv_));
            ci_rdms.compute_1rdm_sf_op(D1.data());

            if (max_rdm_level == 1) {
                rdms.emplace_back(std::make_shared<RDMsSpinFree>(D1));
            } else {
                // compute 2-RDM
                std::vector<size_t> dim4(4, nactv_);
                auto D2 = ambit::Tensor::build(CoreTensor, "TD2", dim4);
                ci_rdms.compute_2rdm_sf_op(D2.data());

                if (max_rdm_level == 2) {
                    rdms.emplace_back(std::make_shared<RDMsSpinFree>(D1, D2));
                } else {
                    // compute 3-RDM
                    std::vector<size_t> dim6(6, nactv_);
                    auto D3 = ambit::Tensor::build(CoreTensor, "TD3", dim6);
                    ci_rdms.compute_3rdm_sf_op(D3.data());

                    rdms.emplace_back(std::make_shared<RDMsSpinFree>(D1, D2, D3));
                }
            }
        }
    }

    return rdms;
}

} // namespace forte
