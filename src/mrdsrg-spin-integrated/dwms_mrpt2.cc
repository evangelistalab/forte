#include <tuple>
#include <sstream>

#include "psi4/libmints/molecule.h"
#include "psi4/libmints/dimension.h"

#include "dwms_mrpt2.h"
#include "master_mrdsrg.h"
#include "../fci_mo.h"
#include "../semi_canonicalize.h"

namespace psi {
namespace forte {

void set_DWMS_options(ForteOptions& foptions) {
    /*- Automatic weight switching -*/
    foptions.add_double("DWMS_ZETA", 0.0, "Gaussian width cutoff for the density weights");

    /*- DWMS algorithms
     *  - DWMS-0: weights computed using SA-CASCI energies, non-orthogonal final solutions
     *  - DWMS-1: weights computed using SA-CASCI energies, orthogonal final solutions
     *  - DWMS-AVG0: weights computed using SA-DSRG-PT2 energies, non-orthogonal final solutions
     *  - DWMS-AVG1: weights computed using SA-DSRG-PT2 energies, orthogonal final solutions -*/
    foptions.add_str("DWMS_ALGORITHM", "DWMS-0", {"DWMS-0", "DWMS-1", "DWMS-AVG0", "DWMS-AVG1"},
                     "DWMS algorithms");
}

DWMS_DSRGPT2::DWMS_DSRGPT2(SharedWavefunction ref_wfn, Options& options,
                           std::shared_ptr<ForteIntegrals> ints,
                           std::shared_ptr<MOSpaceInfo> mo_space_info)
    : Wavefunction(options), ints_(ints), mo_space_info_(mo_space_info) {
    reference_wavefunction_ = ref_wfn;

    print_method_banner({"Dynamically Weighted Multi-State DSRG-PT2", "Chenyang Li"});
    startup();
}

DWMS_DSRGPT2::~DWMS_DSRGPT2() {}

void DWMS_DSRGPT2::startup() {
    print_h2("DWMS-DSRG-PT2 Options");
    zeta_ = options_.get_double("DWMS_ZETA");
    algorithm_ = options_.get_str("DWMS_ALGORITHM");
    outfile->Printf("\n    DWMS ZETA          %10.2e", zeta_);
    outfile->Printf("\n    DWMS ALGORITHM     %10s", algorithm_.c_str());

    print_note();

    CharacterTable ct = Process::environment.molecule()->point_group()->char_table();
    int nirrep = reference_wavefunction_->nirrep();
    irrep_symbol_.resize(nirrep);
    for (int h = 0; h < nirrep; ++h) {
        irrep_symbol_[h] = std::string(ct.gamma(h).symbol());
    }

    multi_symbol_ = std::vector<std::string>{"Singlet", "Doublet", "Triplet", "Quartet", "Quintet",
                                             "Sextet",  "Septet",  "Octet",   "Nonet",   "Decaet"};

    size_t na = mo_space_info_->get_dimension("ACTIVE").sum();
    Ua_ = ambit::Tensor::build(CoreTensor, "Uactv a", {na, na});
    Ub_ = ambit::Tensor::build(CoreTensor, "Uactv b", {na, na});
    for (size_t u = 0; u < na; ++u) {
        Ua_.data()[u * na + u] = 1.0;
        Ub_.data()[u * na + u] = 1.0;
    }

    max_rdm_level_ = (options_.get_str("THREEPDC") == "ZERO") ? 2 : 3;

    std::string actv_type = options_.get_str("FCIMO_ACTV_TYPE");
    actv_vci_ = actv_type == "CIS" || actv_type == "CISD";

    std::string int_type = options_.get_str("INT_TYPE");
    eri_df_ = (int_type == "CHOLESKY") || (int_type == "DF") || (int_type == "DISKDF");
}

void DWMS_DSRGPT2::print_note() {
    print_h2("Implementation Note on " + algorithm_);

    bool avg = algorithm_.find("AVG") != std::string::npos;

    outfile->Printf("\n  - Perform SA-CASCI using user-defined weights.");
    if (!avg) {
        outfile->Printf(" (*)");
    }

    if (algorithm_ != "DWMS-0") {
        outfile->Printf("\n  - Perform SA-DSRG-PT2/SA-CASCI using user-defined weights.");
        if (avg) {
            outfile->Printf(" (*)");
        }
    }

    outfile->Printf("\n  - Loop over averaged states:");
    outfile->Printf("\n      - Compute new density weights using energies of (*).");
    outfile->Printf("\n      - Semicanonicalize orbitals.");
    outfile->Printf("\n      - Form DSRG-PT2 transformed Hamiltonian.");
    outfile->Printf("\n      - Perform CASCI on DSRG-PT2 Hamiltonian. (#)");
    outfile->Printf("\n  - Test orthogonality for CI vectors of (#) between states.");
    outfile->Printf("\n  - Collect and print excitation energies.\n");

    outfile->Printf("\n  Additional note on step (#):");
    if (algorithm_.find("0") != std::string::npos) {
        outfile->Printf("\n  - CI vectors will NOT be orthogonal between states.");
    } else {
        outfile->Printf("\n  - Initial guess from SA-DSRG-PT2/SA-CASCI for Davidson-Liu solver.");
        outfile->Printf("\n  - CI vectors of previous roots are projected out from current root.");
        outfile->Printf("\n  - CI vectors should be orthogonal between states.");
    }
}

std::shared_ptr<FCIIntegrals> DWMS_DSRGPT2::compute_dsrg_pt2(std::shared_ptr<FCI_MO> fci_mo,
                                                             Reference& reference) {
    // semi-canonicalize orbitals
    SemiCanonical semi(reference_wavefunction_, ints_, mo_space_info_);
    if (actv_vci_) {
        semi.set_actv_dims(fci_mo->actv_docc(), fci_mo->actv_virt());
    }
    semi.semicanonicalize(reference, max_rdm_level_);
    Ua_ = semi.Ua_t();
    Ub_ = semi.Ub_t();

    // compute dsrg-pt2 energy
    std::shared_ptr<MASTER_DSRG> dsrg_pt2;
    if (eri_df_) {
        dsrg_pt2 = std::make_shared<THREE_DSRG_MRPT2>(reference, reference_wavefunction_, options_,
                                                      ints_, mo_space_info_);
    } else {
        dsrg_pt2 = std::make_shared<DSRG_MRPT2>(reference, reference_wavefunction_, options_, ints_,
                                                mo_space_info_);
    }
    dsrg_pt2->set_Uactv(Ua_, Ub_);

    if (actv_vci_) {
        dsrg_pt2->set_actv_occ(fci_mo->actv_occ());
        dsrg_pt2->set_actv_uocc(fci_mo->actv_uocc());
    }

    dsrg_pt2->compute_energy();
    auto fci_ints = dsrg_pt2->compute_Heff();

    // rotate integrals back to original
    semi.back_transform_ints();

    return fci_ints;
}

std::shared_ptr<FCI_MO> DWMS_DSRGPT2::precompute_energy() {
    // perform SA-CASCI using user-defined weights
    auto fci_mo =
        std::make_shared<FCI_MO>(reference_wavefunction_, options_, ints_, mo_space_info_);
    fci_mo->compute_energy();

    auto sa_info = fci_mo->sa_info();
    int nentry = sa_info.size();

    // save SA-CASCI energies
    Eref_0_.resize(nentry);

    for (int n = 0; n < nentry; ++n) {
        int nroots;
        std::tie(std::ignore, std::ignore, nroots, std::ignore) = sa_info[n];
        for (int i = 0; i < nroots; ++i) {
            Eref_0_[n].push_back(fci_mo->eigens()[n][i].second);
        }
    }

    // perform SA-DSRG-PT2 if needed
    if (algorithm_ != "DWMS-0") {
        // need to save a copy of sa eigen vectors for DWMS-1
        std::vector<std::vector<std::pair<SharedVector, double>>> eigens_old;
        if (algorithm_ == "DWMS-1") {
            eigens_old = fci_mo->eigens();
        }

        Reference reference = fci_mo->reference(max_rdm_level_);

        auto fci_ints = compute_dsrg_pt2(fci_mo, reference);

        print_h2("Diagonalize SA-DSRG-PT2 Active Hamiltonian");
        fci_mo->set_fci_int(fci_ints);
        fci_mo->set_localize_actv(false);
        fci_mo->compute_energy();
        auto eigens = fci_mo->eigens();

        // save SA-DSRG-PT2 energies and CI vectors
        Ept2_0_.resize(nentry);
        initial_guesses_.resize(nentry);

        for (int n = 0; n < nentry; ++n) {
            int nroots;
            std::tie(std::ignore, std::ignore, nroots, std::ignore) = sa_info[n];
            for (int i = 0; i < nroots; ++i) {
                initial_guesses_[n].push_back(eigens[n][i].first);
                Ept2_0_[n].push_back(eigens[n][i].second);
            }
        }

        // use original eigens for DWMS-1
        if (algorithm_ == "DWMS-1") {
            fci_mo->set_eigens(eigens_old);
        }
    }

    return fci_mo;
}

double DWMS_DSRGPT2::compute_energy() {
    // perform SA-CASCI and SA-DSRG-PT2 using user-defined weights
    std::shared_ptr<FCI_MO> fci_mo = precompute_energy();
    auto sa_info = fci_mo->sa_info();
    int nentry = sa_info.size();

    // the final DWMS-DSRG-PT2 energies
    Ept2_.resize(nentry);

    // loop over averaged states
    for (int n = 0; n < nentry; ++n) {
        int multi, irrep, nroots;
        std::tie(irrep, multi, nroots, std::ignore) = sa_info[n];
        Ept2_[n].resize(nroots);

        // save the re-diagonalized eigen vectors
        std::vector<SharedVector> evecs_new;
        evecs_new.resize(nroots);

        // save the previous projected roots
        std::vector<std::vector<std::pair<size_t, double>>> projected_roots;

        for (int i = 0; i < nroots; ++i) {
            print_current_title(multi, irrep, i);

            // compute new weights
            std::vector<std::vector<double>>& Erefs = Eref_0_;
            if (algorithm_.find("AVG") != std::string::npos) {
                Erefs = Ept2_0_;
            }
            auto sa_info_new = compute_dwms_weights(sa_info, n, i, Erefs);
            print_sa_info("Original State Averaging Summary", sa_info);
            print_sa_info("Reweighted State Averaging Summary", sa_info_new);

            // compute Reference
            fci_mo->set_sa_info(sa_info_new);
            Reference reference = fci_mo->reference(max_rdm_level_);

            double Enuc = Process::environment.molecule()->nuclear_repulsion_energy(
                reference_wavefunction_->get_dipole_field_strength());
            reference.update_Eref(ints_, mo_space_info_, Enuc);

            // compute DSRG-PT2 transformed Hamiltonian
            auto fci_ints = compute_dsrg_pt2(fci_mo, reference);

            // prepare for diagonalizing the DSRG-PT2 active Hamiltonian
            print_h2("Diagonalize DWMS-DSRG-PT2 Active Hamiltonian");
            fci_mo->set_fci_int(fci_ints);
            fci_mo->set_root_sym(irrep);

            if (algorithm_ == "DWMS-1" || algorithm_ == "DWMS-AVG1") {
                fci_mo->set_nroots(nroots - i);
                fci_mo->set_root(0);

                // project out previous DWMS-DSRG-PT2 roots
                if (i != 0) {
                    outfile->Printf("\n\n  Project out previous DWMS-DSRG-PT2 roots.\n");
                }
                fci_mo->project_roots(projected_roots);

                // set initial guess to help SparseCISolver convergence
                std::vector<std::pair<size_t, double>> guess;
                for (size_t I = 0, nI = initial_guesses_[n][i]->dim(); I < nI; ++I) {
                    guess.push_back(std::make_pair(I, initial_guesses_[n][i]->get(I)));
                }
                fci_mo->set_initial_guess(guess);
            } else {
                fci_mo->set_nroots(nroots);
                fci_mo->set_root(i);
            }

            // finally diagonalize DSRG-PT2 active Hamiltonian
            Ept2_[n][i] = fci_mo->compute_ss_energy();

            // since Heff is rotated to the original basis, we can safely store the CI vectors
            if (algorithm_ == "DWMS-1" || algorithm_ == "DWMS-AVG1") {
                evecs_new[i] = fci_mo->eigen()[0].first;

                // append current CI vectors to the projection list
                std::vector<std::pair<size_t, double>> projection;
                for (size_t I = 0, nI = evecs_new[i]->dim(); I < nI; ++I) {
                    projection.push_back(std::make_pair(I, evecs_new[i]->get(I)));
                }
                projected_roots.push_back(projection);
            } else {
                evecs_new[i] = fci_mo->eigen()[i].first;
            }
        }

        // form and print overlap
        std::string Sname = "Overlap of " + multi_symbol_[multi - 1] + " " + irrep_symbol_[irrep];
        print_overlap(evecs_new, Sname);
    }

    // print energy summary
    outfile->Printf("  ==> Dynamically Weighted Multi-State DSRG-PT2 Energy Summary <==\n");
    print_energy_list("SA-CASCI Energy", Eref_0_, sa_info);
    if (algorithm_ != "DWMS-0") {
        print_energy_list("SA-DSRG-PT2 Energy", Ept2_0_, sa_info);
    }
    print_energy_list("DWMS-DSRGPT2 Energy", Ept2_, sa_info, true);

    return Ept2_[0][0];
}

std::vector<std::tuple<int, int, int, std::vector<double>>> DWMS_DSRGPT2::compute_dwms_weights(
    const std::vector<std::tuple<int, int, int, std::vector<double>>>& sa_info, int entry, int root,
    const std::vector<std::vector<double>>& energy) {

    int nentry = sa_info.size();
    if (nentry != energy.size()) {
        throw PSIEXCEPTION("Mismatching sizes between energy list and sa_info list");
    }

    // new weights
    std::vector<std::vector<double>> new_weights;
    new_weights.resize(nentry);

    // new weights for state alpha:
    // w_i = exp(-zeta * (E_alpha - E_i)^2) / sum_j exp(-zeta * (E_alpha - E_j)^2)

    double Ealpha = energy[entry][root];
    double wsum = 0.0;

    for (int n = 0; n < nentry; ++n) {
        int irrep, multi, nroots;
        std::vector<double> weights;
        std::tie(irrep, multi, nroots, weights) = sa_info[n];

        new_weights[n].resize(nroots);
        for (int i = 0; i < nroots; ++i) {
            double Ediff = energy[n][i] - Ealpha;
            double gaussian = std::exp(-zeta_ * Ediff * Ediff);
            new_weights[n][i] = gaussian;
            wsum += gaussian;
        }
    }

    for (int n = 0; n < nentry; ++n) {
        for (int i = 0, nroots = new_weights[n].size(); i < nroots; ++i) {
            new_weights[n][i] /= wsum;
        }
    }

    // form new sa_info
    std::vector<std::tuple<int, int, int, std::vector<double>>> out;
    out.resize(nentry);
    for (int n = 0; n < nentry; ++n) {
        int irrep, multi, nroots;
        std::tie(irrep, multi, nroots, std::ignore) = sa_info[n];

        out[n] = std::make_tuple(irrep, multi, nroots, new_weights[n]);
    }

    return out;
}

void DWMS_DSRGPT2::print_current_title(int multi, int irrep, int root) {
    std::stringstream current_job;
    current_job << "Current Job: " << multi_symbol_[multi - 1] << " " << irrep_symbol_[irrep]
                << ", root " << root;
    size_t title_size = current_job.str().size();
    outfile->Printf("\n\n  %s", std::string(title_size, '=').c_str());
    outfile->Printf("\n  %s", current_job.str().c_str());
    outfile->Printf("\n  %s\n", std::string(title_size, '=').c_str());
}

void DWMS_DSRGPT2::print_sa_info(
    const std::string& name,
    const std::vector<std::tuple<int, int, int, std::vector<double>>>& sa_info) {

    print_h2(name);

    auto longer = [](const std::tuple<int, int, int, std::vector<double>>& A,
                     const std::tuple<int, int, int, std::vector<double>>& B) {
        return std::get<2>(A) < std::get<2>(B);
    };
    int max_nroots = std::get<2>(*std::max_element(sa_info.begin(), sa_info.end(), longer));

    int max_line = 0;
    if (max_nroots == 1) {
        max_line = 7;
    } else {
        max_line = 6 * (max_nroots > 5 ? 5 : max_nroots);
    }

    int ltotal = 6 + 2 + 6 + 2 + 6 + 2 + max_line;
    std::string blank(max_line - 7, ' ');
    std::string dash(ltotal, '-');
    outfile->Printf("\n    Irrep.  Multi.  Nroots  %sWeights", blank.c_str());
    outfile->Printf("\n    %s", dash.c_str());

    for (int n = 0, nentry = sa_info.size(); n < nentry; ++n) {
        int irrep, multi, nroots;
        std::vector<double> weights;
        std::tie(irrep, multi, nroots, weights) = sa_info[n];

        std::stringstream ss_w;
        for (int i = 0, nw = weights.size(); i < nw; ++i) {
            ss_w << " " << std::fixed << std::setprecision(3) << weights[i];
            if (i % 5 == 0 && i != 0) {
                ss_w << std::endl << std::string(24, ' ');
            }
        }

        std::ostringstream oss;
        oss << std::setw(4) << std::right << irrep_symbol_[irrep] << "    " << std::setw(4)
            << std::right << multi << "    " << std::setw(4) << std::right << nroots << "    "
            << std::setw(max_line) << ss_w.str();
        outfile->Printf("\n    %s", oss.str().c_str());
    }
    outfile->Printf("\n    %s", dash.c_str());
}

void DWMS_DSRGPT2::print_overlap(const std::vector<SharedVector>& evecs, const std::string& Sname) {
    print_h2(Sname);
    outfile->Printf("\n");

    int nroots = evecs.size();
    SharedMatrix S(new Matrix("S", nroots, nroots));

    for (int i = 0; i < nroots; ++i) {
        for (int j = i; j < nroots; ++j) {
            double Sij = evecs[i]->vector_dot(evecs[j]);
            S->set(i, j, Sij);
            S->set(j, i, Sij);
        }
    }

    S->print();
}

void DWMS_DSRGPT2::print_energy_list(
    std::string name, const std::vector<std::vector<double>>& energy,
    const std::vector<std::tuple<int, int, int, std::vector<double>>>& sa_info, bool pass_process) {

    if (sa_info.size() != energy.size()) {
        throw PSIEXCEPTION("Mismatching sizes between energy list and sa_info list");
    }

    outfile->Printf("\n    Multi.  Irrep.  No.    %20s", name.c_str());
    std::string dash(43, '-');
    outfile->Printf("\n    %s", dash.c_str());

    for (int n = 0, counter = 0, nentry = energy.size(); n < nentry; ++n) {
        int irrep, multi, nroots;
        std::tie(irrep, multi, nroots, std::ignore) = sa_info[n];

        for (int i = 0; i < nroots; ++i) {
            outfile->Printf("\n     %3d     %3s    %2d   %22.14f", multi,
                            irrep_symbol_[irrep].c_str(), i, energy[n][i]);
            if (pass_process) {
                Process::environment.globals["ENERGY ROOT " + std::to_string(counter)] =
                    energy[n][i];
            }
            counter += 1;
        }
        outfile->Printf("\n    %s", dash.c_str());
    }
    outfile->Printf("\n");
}
}
}
