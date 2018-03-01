#include <tuple>
#include <sstream>

#include "psi4/libmints/molecule.h"

#include "dwms_mrpt2.h"
#include "master_mrdsrg.h"
#include "../fci_mo.h"
#include "../semi_canonicalize.h"

namespace psi {
namespace forte {

void set_DWMS_options(ForteOptions& foptions) {
    /**
     * Weights of state α:
     *    Wi = exp(-ζ * (Eα - Ei)^2) / sum_j exp(-ζ * (Eα - Ej)^2)
     *
     * Energies (Eα, Ei, Ej) can be SA-CASCI or SA-DSRG-PT2 energies.
     */

    /*- Automatic weight switching -*/
    foptions.add_double("DWMS_ZETA", 0.0, "Gaussian width cutoff for the density weights");

    /*- Using what energies to compute the weight
     * CAS: use SA-CASCI energies
     * DSRG-PT2: use SA-DSRG-PT2 energies -*/
    foptions.add_str("DWMS_ENERGY", "CAS", {"CAS", "DSRG-PT2"},
                     "Energies to compute dynamic weights");

    /*- Using what CI vectors to perform multi-state computation
     * CAS: use SA-CASCI eigenvectors
     * DSRG-PT2: use SA-DSRG-PT2/CASCI eigenvectors -*/
    foptions.add_str("DWMS_CI", "CAS", {"CAS", "DSRG-PT2"},
                     "CI vectors to compute dynamic weights");

    /*- DWMS algorithms
     *  - MS: multi-state (single-state single-reference)
     *  - XMS: extended multi-state (single-state single-reference)
     *
     * To Be Deprecated:
     *  - DWMS-0: weights from SA-CASCI energies, non-orthogonal final solutions
     *  - DWMS-1: weights from SA-CASCI energies, orthogonal final solutions
     *  - DWMS-AVG0: weights from SA-DSRG-PT2 energies, non-orthogonal final solutions
     *  - DWMS-AVG1: weights from SA-DSRG-PT2 energies, orthogonal final solutions -*/
    foptions.add_str("DWMS_ALGORITHM", "DWMS-0",
                     {"MS", "XMS", "SA", "DWMS-0", "DWMS-1", "DWMS-AVG0", "DWMS-AVG1"},
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
    zeta_ = options_.get_double("DWMS_ZETA");
    algorithm_ = options_.get_str("DWMS_ALGORITHM");
    dwms_e_ = options_.get_str("DWMS_ENERGY");
    dwms_ci_ = options_.get_str("DWMS_CI");

    print_h2("DWMS-DSRG-PT2 Options");
    outfile->Printf("\n    DWMS ZETA          %10.2e", zeta_);
    outfile->Printf("\n    DWMS ENERGY        %10s", dwms_e_.c_str());
    outfile->Printf("\n    DWMS CI VECTORS    %10s", dwms_ci_.c_str());
    outfile->Printf("\n    DWMS ALGORITHM     %10s", algorithm_.c_str());

    if (zeta_ < 0.0) {
        throw PSIEXCEPTION("DWMS_ZETA should be a value greater or equal than 0.0!");
    }

    if (algorithm_ == "MS" || algorithm_ == "XMS") {
        print_note_xms();
    } else {
        print_note();
    }

    CharacterTable ct = Process::environment.molecule()->point_group()->char_table();
    int nirrep = reference_wavefunction_->nirrep();
    irrep_symbol_.resize(nirrep);
    for (int h = 0; h < nirrep; ++h) {
        irrep_symbol_[h] = std::string(ct.gamma(h).symbol());
    }

    multi_symbol_ = std::vector<std::string>{"Singlet", "Doublet", "Triplet", "Quartet", "Quintet",
                                             "Sextet",  "Septet",  "Octet",   "Nonet",   "Decaet"};

    size_t na = mo_space_info_->size("ACTIVE");
    Ua_ = ambit::Tensor::build(CoreTensor, "Uactv a", {na, na});
    Ub_ = ambit::Tensor::build(CoreTensor, "Uactv b", {na, na});
    for (size_t u = 0; u < na; ++u) {
        Ua_.data()[u * na + u] = 1.0;
        Ub_.data()[u * na + u] = 1.0;
    }

    max_rdm_level_ = (options_.get_str("THREEPDC") == "ZERO") ? 2 : 3;

    std::string actv_type = options_.get_str("FCIMO_ACTV_TYPE");
    actv_vci_ = actv_type == "CIS" || actv_type == "CISD";

    IntegralType int_type = ints_->integral_type();
    eri_df_ = (int_type == Cholesky) || (int_type == DF) || (int_type == DiskDF);
}

void DWMS_DSRGPT2::print_note_xms() {
    print_h2("Implementation Note on " + algorithm_);

    outfile->Printf("\n  Use SA-%s energies to compute weights.", dwms_e_.c_str());
    outfile->Printf("\n  Use SA-%s CI vectors to do %s.\n", dwms_ci_.c_str(), algorithm_.c_str());

    outfile->Printf("\n  - Perform SA-CASCI using user-defined weights.");

    if (dwms_ci_ == "DSRG-PT2" || dwms_e_ == "DSRG-PT2") {
        outfile->Printf("\n  - Perform SA-DSRG-PT2/CASCI using user-defined weights.");
    }

    if (algorithm_ == "XMS") {
        outfile->Printf("\n  - Perform XMS rotation to SA-%s CI vectors.", dwms_ci_.c_str());
    }

    outfile->Printf("\n  - Loop over symmetry entries:");
    outfile->Printf("\n    - Initialize effective Hamiltonian Heff.");
    outfile->Printf("\n    - Loop over averaged states:");
    outfile->Printf("\n      - Compute new density weights if ζ is nonzero.");
    outfile->Printf("\n      - Semicanonicalize orbitals.");
    outfile->Printf("\n      - Put SS-DSRG-PT2 energy to diagonal of Heff.");
    outfile->Printf("\n      - Compute couplings of Heff:");
    outfile->Printf("\n        - De-normal-order cluster amplitudes.");
    outfile->Printf("\n        - Compute 1, 2, 3 transition densities in semicanonical basis.");
    outfile->Printf("\n        - <A|Heff|B> <- 0.5 * <A|Heff(B)|B>.");
    outfile->Printf("\n  - Collect and print energies.\n");
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
    auto fci_ints = dsrg_pt2->compute_Heff_actv();

    // rotate integrals back to original
    semi.back_transform_ints();

    return fci_ints;
}

std::shared_ptr<FCI_MO> DWMS_DSRGPT2::precompute_energy() {
    // perform SA-CASCI using user-defined weights
    auto fci_mo =
        std::make_shared<FCI_MO>(reference_wavefunction_, options_, ints_, mo_space_info_);
    fci_mo->compute_energy();
    fci_ints_ = fci_mo->fci_ints();

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
    if (dwms_ci_ == "DSRG-PT2" || dwms_e_ == "DSRG-PT2") {
        // save a copy of sa eigen vectors when necessary
        std::vector<std::vector<std::pair<SharedVector, double>>> eigens_old;
        if (dwms_e_ == "DSRG-PT2" && dwms_ci_ == "CAS") {
            eigens_old = fci_mo->eigens();
        }

        Reference reference = fci_mo->reference(max_rdm_level_);

        fci_ints_ = compute_dsrg_pt2(fci_mo, reference);

        print_h2("Diagonalize SA-DSRG-PT2 Active Hamiltonian");
        fci_mo->set_fci_int(fci_ints_);
        fci_mo->set_localize_actv(false);
        fci_mo->compute_energy();
        auto eigens = fci_mo->eigens();

        // save SA-DSRG-PT2 energies and CI vectors
        Ept2_0_.resize(nentry);

        for (int n = 0; n < nentry; ++n) {
            int nroots;
            std::tie(std::ignore, std::ignore, nroots, std::ignore) = sa_info[n];
            for (int i = 0; i < nroots; ++i) {
                Ept2_0_[n].push_back(eigens[n][i].second);
            }
        }

        // use original eigens when necessary
        if (dwms_e_ == "DSRG-PT2" && dwms_ci_ == "CAS") {
            fci_mo->set_eigens(eigens_old);
        }
    }

    return fci_mo;
}

double DWMS_DSRGPT2::compute_energy() {
    if (algorithm_ == "MS" || algorithm_ == "XMS") {
        return compute_dwms_energy();
    } else if (algorithm_ == "SA") {
        return compute_dwsa_energy();
    } else {
        return compute_dwms_energy_old();
    }
}

double DWMS_DSRGPT2::compute_dwsa_energy() {
    // perform SA-CASCI or SA-DSRG-PT2 if necessary
    std::shared_ptr<FCI_MO> fci_mo = precompute_energy();
    auto sa_info = fci_mo->sa_info();
    int nentry = sa_info.size();
    bool do_hbar3 = options_.get_bool("FORM_HBAR3");

    // prepare the final DWMS-DSRG-PT2 energies
    Ept2_.resize(nentry);

    // save a copy of original orbitals
    SharedMatrix Ca_copy(reference_wavefunction_->Ca()->clone());
    SharedMatrix Cb_copy(reference_wavefunction_->Cb()->clone());

    // loop over symmetry entries
    for (int n = 0; n < nentry; ++n) {
        int multi, irrep, nroots;
        std::tie(irrep, multi, nroots, std::ignore) = sa_info[n];
        Ept2_[n].resize(nroots);

        // print current status
        std::string entry_name = multi_symbol_[multi - 1] + " " + irrep_symbol_[irrep];
        std::string entry_title = "Build Effective Hamiltonian of " + entry_name;
        size_t entry_title_size = entry_title.size();
        outfile->Printf("\n\n  %s", std::string(entry_title_size, '=').c_str());
        outfile->Printf("\n  %s", entry_title.c_str());
        outfile->Printf("\n  %s\n", std::string(entry_title_size, '=').c_str());

        // prepare Heff
        SharedMatrix Heff(new Matrix("Heff " + entry_name, nroots, nroots));
        SharedMatrix Heff_sym(new Matrix("Symmetrized Heff " + entry_name, nroots, nroots));

        // loop over states of current symmetry
        for (int M = 0; M < nroots; ++M) {
            print_h2("Compute DSRG-MRPT2 Energy of Root " + std::to_string(M));

            // compute new weights
            std::vector<std::vector<double>>& Erefs = Eref_0_;
            if (dwms_e_ == "DSRG-PT2") {
                Erefs = Ept2_0_;
            }
            auto sa_info_new = compute_dwms_weights(sa_info, n, M, Erefs);
            print_sa_info("Original State Averaging Summary", sa_info);
            print_sa_info("Reweighted State Averaging Summary", sa_info_new);

            // compute Reference
            std::vector<double> weights;
            std::tie(std::ignore, std::ignore, std::ignore, weights) = sa_info_new[n];
            fci_mo->set_sa_info(sa_info_new);
            Reference reference = fci_mo->reference(max_rdm_level_);

            // update MK vacuum energy
            double Enuc = Process::environment.molecule()->nuclear_repulsion_energy(
                reference_wavefunction_->get_dipole_field_strength());
            reference.update_Eref(ints_, mo_space_info_, Enuc);

            // DSRG-PT2 pointer
            std::shared_ptr<MASTER_DSRG> dsrg_pt2;

            // canonicalize orbitals if density fitting
            // because three-dsrg-mrpt2 assumes semicanonical orbitals
            if (eri_df_) {
                SemiCanonical semi(reference_wavefunction_, ints_, mo_space_info_);
                semi.semicanonicalize(reference, max_rdm_level_);
                Ua_ = semi.Ua_t();
                Ub_ = semi.Ub_t();

                dsrg_pt2 = std::make_shared<THREE_DSRG_MRPT2>(reference, reference_wavefunction_,
                                                              options_, ints_, mo_space_info_);
                dsrg_pt2->set_Uactv(Ua_, Ub_);
            } else {
                dsrg_pt2 = std::make_shared<DSRG_MRPT2>(reference, reference_wavefunction_,
                                                        options_, ints_, mo_space_info_);
            }

            // compute state-specific DSRG-MRPT2 energy
            dsrg_pt2->compute_energy();

            // compute 2nd-order efffective Hamiltonian for the couplings
            auto fci_ints = dsrg_pt2->compute_Heff_actv();

//            outfile->Printf("\n!!!! H1a norm   %25.15f", dsrg_pt2->Hbar(1)[0].norm());
//            outfile->Printf("\n!!!! H1b norm   %25.15f", dsrg_pt2->Hbar(1)[1].norm());
//            outfile->Printf("\n!!!! H2aa norm  %25.15f", dsrg_pt2->Hbar(2)[0].norm());
//            outfile->Printf("\n!!!! H2ab norm  %25.15f", dsrg_pt2->Hbar(2)[1].norm());
//            outfile->Printf("\n!!!! H2bb norm  %25.15f", dsrg_pt2->Hbar(2)[2].norm());
//            outfile->Printf("\n!!!! H3aaa norm %25.15f", dsrg_pt2->Hbar(3)[0].norm());
//            outfile->Printf("\n!!!! H3aab norm %25.15f", dsrg_pt2->Hbar(3)[1].norm());
//            outfile->Printf("\n!!!! H3abb norm %25.15f", dsrg_pt2->Hbar(3)[2].norm());
//            outfile->Printf("\n!!!! H3bbb norm %25.15f", dsrg_pt2->Hbar(3)[3].norm());

            for (int N = 0; N < nroots; ++N) {
                std::string msg = "densities";
                if (M == N) {
                    msg = "transition densities";
                }

                // compute transition densities
                outfile->Printf("\n  Compute %s.", msg.c_str());
                Reference TrD;
                if (do_hbar3) {
                    TrD = fci_mo->compute_trans_density(M, N, true, n, 3, false);
                } else {
                    TrD = fci_mo->compute_trans_density(M, N, true, n, 2, false);
                }

                outfile->Printf("\n  Contract %s with Heff.", msg.c_str());
                double coupling = 0.0;

                auto Hbar_vec = dsrg_pt2->Hbar(1);
                coupling += Hbar_vec[0]("vu") * TrD.L1a()("uv");
                coupling += Hbar_vec[1]("vu") * TrD.L1b()("uv");

                Hbar_vec = dsrg_pt2->Hbar(2);
                coupling += 0.25 * Hbar_vec[0]("xyuv") * TrD.g2aa()("uvxy");
                coupling += Hbar_vec[1]("xYuV") * TrD.g2ab()("uVxY");
                coupling += 0.25 * Hbar_vec[2]("XYUV") * TrD.g2bb()("UVXY");

                if (do_hbar3) {
                    Hbar_vec = dsrg_pt2->Hbar(3);
                    coupling += (1.0 / 36) * Hbar_vec[0]("uvwxyz") * TrD.g3aaa()("xyzuvw");
                    coupling += 0.25 * Hbar_vec[1]("uvwxyz") * TrD.g3aab()("xyzuvw");
                    coupling += 0.25 * Hbar_vec[2]("uvwxyz") * TrD.g3abb()("xyzuvw");
                    coupling += (1.0 / 36) * Hbar_vec[3]("uvwxyz") * TrD.g3bbb()("xyzuvw");
                }

                if (M == N) {
                    double shift = ints_->frozen_core_energy() + Enuc + fci_ints->scalar_energy();
                    Heff->set(M, M, coupling + shift);
                    Heff_sym->set(M, M, coupling + shift);
                } else {
                    Heff->set(N, M, coupling);
                    Heff_sym->add(N, M, 0.5 * coupling);
                    Heff_sym->add(M, N, 0.5 * coupling);
                }
            }

            // rotate basis back to original if density fitting
            if (eri_df_) {
                print_h2("Back Transformation of Semicanonical Integrals");
                SharedMatrix Ca = reference_wavefunction_->Ca();
                SharedMatrix Cb = reference_wavefunction_->Cb();
                Ca->copy(Ca_copy);
                Cb->copy(Cb_copy);
                ints_->retransform_integrals();
            }
        }

        // print effective Hamiltonian
        print_h2("Effective Hamiltonian Summary");
        outfile->Printf("\n");

        Heff->print();
        Heff_sym->print();

        // diagonalize Heff and print eigen vectors
        SharedMatrix U(new Matrix("U of Heff (Symmetrized)", nroots, nroots));
        SharedVector Ems(new Vector("MS Energies", nroots));
        Heff_sym->diagonalize(U, Ems);
        U->eivprint(Ems);

        // set Ept2_ value
        for (int i = 0; i < nroots; ++i) {
            Ept2_[n][i] = Ems->get(i);
        }
    }

    // print energy summary
    outfile->Printf("  ==> Dynamically Weighted Multi-State DSRG-PT2 Energy Summary <==\n");
    print_energy_list("SA-CASCI Energy", Eref_0_, sa_info);
    if (dwms_ci_ == "DSRG-PT2" || dwms_e_ == "DSRG-PT2") {
        print_energy_list("SA-DSRG-PT2 Energy", Ept2_0_, sa_info);
    }
    print_energy_list("DWSA-DSRGP-T2 Energy", Ept2_, sa_info, true);

    return Ept2_[0][0];
}

double DWMS_DSRGPT2::compute_dwms_energy() {
    // perform SA-CASCI or SA-DSRG-PT2 if necessary
    std::shared_ptr<FCI_MO> fci_mo = precompute_energy();
    auto sa_info = fci_mo->sa_info();
    int nentry = sa_info.size();

    // XMS rotation for all symmetries (same orbital basis)
    if (algorithm_ == "XMS") {
        fci_mo->xms_rotate_civecs();
    }

    // prepare the final DWMS-DSRG-PT2 energies
    Ept2_.resize(nentry);

    // save a copy of original orbitals
    SharedMatrix Ca_copy(reference_wavefunction_->Ca()->clone());
    SharedMatrix Cb_copy(reference_wavefunction_->Cb()->clone());

    // prepare Reference
    Reference reference;

    // if zeta = 0, reference and amplitudes will be the same for all states
    // we should only compute DSRG once, from which we obtain the "effective Hamiltonian"
    // then loop over states to obtain the couplings
    double zero_threshold = 1.0e-8;
    if (zeta_ < zero_threshold) {
        outfile->Printf("\n  DWMS_ZETA = 0.0! Equal weights will be used for all states.");
    }

    // loop over symmetry entries
    for (int n = 0; n < nentry; ++n) {
        int multi, irrep, nroots;
        std::tie(irrep, multi, nroots, std::ignore) = sa_info[n];
        Ept2_[n].resize(nroots);

        // print current status
        std::string entry_name = multi_symbol_[multi - 1] + " " + irrep_symbol_[irrep];
        std::string entry_title = "Build Effective Hamiltonian of " + entry_name;
        size_t entry_title_size = entry_title.size();
        outfile->Printf("\n\n  %s", std::string(entry_title_size, '=').c_str());
        outfile->Printf("\n  %s", entry_title.c_str());
        outfile->Printf("\n  %s\n", std::string(entry_title_size, '=').c_str());

        // prepare Heff
        SharedMatrix Heff(new Matrix("Heff " + entry_name, nroots, nroots));
        SharedMatrix Heff_sym(new Matrix("Symmetrized Heff " + entry_name, nroots, nroots));

        // loop over states of current symmetry
        for (int M = 0; M < nroots; ++M) {
            print_h2("Compute DSRG-MRPT2 Energy of Root " + std::to_string(M));

            // compute new weights
            std::vector<std::vector<double>>& Erefs = Eref_0_;
            if (dwms_e_ == "DSRG-PT2") {
                Erefs = Ept2_0_;
            }
            auto sa_info_new = compute_dwms_weights(sa_info, n, M, Erefs);
            print_sa_info("Original State Averaging Summary", sa_info);
            print_sa_info("Reweighted State Averaging Summary", sa_info_new);

            // compute Reference
            // use state-specific Reference if current weight is close to 1.0
            std::vector<double> weights;
            std::tie(std::ignore, std::ignore, std::ignore, weights) = sa_info_new[n];
            if ((1.0 - weights[M]) < zero_threshold) {
                reference = fci_mo->compute_trans_density(M, M, true, n, max_rdm_level_, true);
            } else {
                fci_mo->set_sa_info(sa_info_new);
                reference = fci_mo->reference(max_rdm_level_);
            }

            // update MK vacuum energy
            double Enuc = Process::environment.molecule()->nuclear_repulsion_energy(
                reference_wavefunction_->get_dipole_field_strength());
            reference.update_Eref(ints_, mo_space_info_, Enuc);

            // DSRG-PT2 pointer
            std::shared_ptr<MASTER_DSRG> dsrg_pt2;

            // canonicalize orbitals if density fitting
            // because three-dsrg-mrpt2 assumes semicanonical orbitals
            if (eri_df_) {
                SemiCanonical semi(reference_wavefunction_, ints_, mo_space_info_);
                semi.semicanonicalize(reference, max_rdm_level_);
                Ua_ = semi.Ua_t();
                Ub_ = semi.Ub_t();

                dsrg_pt2 = std::make_shared<THREE_DSRG_MRPT2>(reference, reference_wavefunction_,
                                                              options_, ints_, mo_space_info_);
            } else {
                dsrg_pt2 = std::make_shared<DSRG_MRPT2>(reference, reference_wavefunction_,
                                                        options_, ints_, mo_space_info_);
            }

            // compute state-specific DSRG-MRPT2 energy
            double E = dsrg_pt2->compute_energy();
            //            Heff->set(M, M, E);
            //            Heff_sym->set(M, M, E);

            // compute 2nd-order efffective Hamiltonian for the couplings
            print_h2("Compute couplings of 2nd-order effective Hamiltonian");

            outfile->Printf("\n  Compute 2nd-order Heff = H + H * T(root %d).", M);
            ambit::Tensor H1a, H1b, H2aa, H2ab, H2bb, H3aaa, H3aab, H3abb, H3bbb;
            double H0;
            dsrg_pt2->compute_Heff_2nd_coupling(H0, H1a, H1b, H2aa, H2ab, H2bb, H3aaa, H3aab, H3abb,
                                                H3bbb);
//            outfile->Printf("\n!!!! H1a norm   %25.15f", H1a.norm());
//            outfile->Printf("\n!!!! H1b norm   %25.15f", H1b.norm());
//            outfile->Printf("\n!!!! H2aa norm  %25.15f", H2aa.norm());
//            outfile->Printf("\n!!!! H2ab norm  %25.15f", H2ab.norm());
//            outfile->Printf("\n!!!! H2bb norm  %25.15f", H2bb.norm());
//            outfile->Printf("\n!!!! H3aaa norm %25.15f", H3aaa.norm());
//            outfile->Printf("\n!!!! H3aab norm %25.15f", H3aab.norm());
//            outfile->Printf("\n!!!! H3abb norm %25.15f", H3abb.norm());
//            outfile->Printf("\n!!!! H3bbb norm %25.15f", H3bbb.norm());

            // need to rotate these Heff to original basis if density fitting
            // because CI vectors are obtained in the original orbital basis
            if (eri_df_) {
                ambit::Tensor temp;

                temp = H1a.clone();
                temp("rs") = Ua_("rp") * H1a("pq") * Ua_("sq");
                H1a("pq") = temp("pq");

                temp.set_name(H1b.name());
                temp("rs") = Ub_("rp") * H1b("pq") * Ub_("sq");
                H1b("pq") = temp("pq");

                temp = H2aa.clone();
                temp("pqrs") = Ua_("pa") * Ua_("qb") * H2aa("abcd") * Ua_("rc") * Ua_("sd");
                H2aa("pqrs") = temp("pqrs");

                temp.set_name(H2ab.name());
                temp("pqrs") = Ua_("pa") * Ub_("qb") * H2ab("abcd") * Ua_("rc") * Ub_("sd");
                H2ab("pqrs") = temp("pqrs");

                temp.set_name(H2bb.name());
                temp("pqrs") = Ub_("pa") * Ub_("qb") * H2bb("abcd") * Ub_("rc") * Ub_("sd");
                H2bb("pqrs") = temp("pqrs");

                temp = H3aaa.clone();
                temp("pqrsto") = Ua_("pa") * Ua_("qb") * Ua_("rc") * H3aaa("abcdef") * Ua_("sd") *
                                 Ua_("te") * Ua_("of");
                H3aaa("pqrsto") = temp("pqrsto");

                temp.set_name(H3aab.name());
                temp("pqrsto") = Ua_("pa") * Ua_("qb") * Ub_("rc") * H3aab("abcdef") * Ua_("sd") *
                                 Ua_("te") * Ub_("of");
                H3aab("pqrsto") = temp("pqrsto");

                temp.set_name(H3abb.name());
                temp("pqrsto") = Ua_("pa") * Ub_("qb") * Ub_("rc") * H3abb("abcdef") * Ua_("sd") *
                                 Ub_("te") * Ub_("of");
                H3abb("pqrsto") = temp("pqrsto");

                temp.set_name(H3bbb.name());
                temp("pqrsto") = Ub_("pa") * Ub_("qb") * Ub_("rc") * H3bbb("abcdef") * Ub_("sd") *
                                 Ub_("te") * Ub_("of");
                H3bbb("pqrsto") = temp("pqrsto");
            }

            for (int N = 0; N < nroots; ++N) {
                //                if (M == N) {
                //                    continue;
                //                }

                // compute transition densities
                outfile->Printf("\n  Compute transition densities.");
                Reference TrD = fci_mo->compute_trans_density(M, N, true, n, 3, false);

                outfile->Printf("\n  Contract transition densities with Heff.");
                double coupling = 0.0;
                if (M == N) {
                    coupling = ints_->frozen_core_energy() + Enuc + H0;
                }
                coupling += H1a("vu") * TrD.L1a()("uv");
                coupling += H1b("vu") * TrD.L1b()("uv");

                coupling += 0.25 * H2aa("xyuv") * TrD.g2aa()("uvxy");
                coupling += H2ab("xYuV") * TrD.g2ab()("uVxY");
                coupling += 0.25 * H2bb("XYUV") * TrD.g2bb()("UVXY");

                coupling += 1.0 / 36.0 * H3aaa("uvwxyz") * TrD.g3aaa()("xyzuvw");
                coupling += 0.25 * H3aab("uvwxyz") * TrD.g3aab()("xyzuvw");
                coupling += 0.25 * H3abb("uvwxyz") * TrD.g3abb()("xyzuvw");
                coupling += 1.0 / 36.0 * H3bbb("uvwxyz") * TrD.g3bbb()("xyzuvw");

                Heff->set(N, M, coupling);
                Heff_sym->add(N, M, 0.5 * coupling);
                Heff_sym->add(M, N, 0.5 * coupling);
            }

            // rotate basis back to original if density fitting
            if (eri_df_) {
                print_h2("Back Transformation of Semicanonical Integrals");
                SharedMatrix Ca = reference_wavefunction_->Ca();
                SharedMatrix Cb = reference_wavefunction_->Cb();
                Ca->copy(Ca_copy);
                Cb->copy(Cb_copy);
                ints_->retransform_integrals();
            }
        }

        // print effective Hamiltonian
        print_h2("Effective Hamiltonian Summary");
        outfile->Printf("\n");

        Heff->print();
        Heff_sym->print();

        // diagonalize Heff and print eigen vectors
        SharedMatrix U(new Matrix("U of Heff (Symmetrized)", nroots, nroots));
        SharedVector Ems(new Vector("MS Energies", nroots));
        Heff_sym->diagonalize(U, Ems);
        U->eivprint(Ems);

        // set Ept2_ value
        for (int i = 0; i < nroots; ++i) {
            Ept2_[n][i] = Ems->get(i);
        }
    }

    // print energy summary
    outfile->Printf("  ==> Dynamically Weighted Multi-State DSRG-PT2 Energy Summary <==\n");
    print_energy_list("SA-CASCI Energy", Eref_0_, sa_info);
    if (dwms_ci_ == "DSRG-PT2" || dwms_e_ == "DSRG-PT2") {
        print_energy_list("SA-DSRG-PT2 Energy", Ept2_0_, sa_info);
    }
    print_energy_list("DWMS-DSRGP-T2 Energy", Ept2_, sa_info, true);

    return Ept2_[0][0];
}

void DWMS_DSRGPT2::compute_Fock_actv(const det_vec& p_space, SharedMatrix civecs, ambit::Tensor Fa,
                                     ambit::Tensor Fb) {
    // make sure the size match
    size_t nroots = p_space.size();
    if (nroots != civecs->ncol()) {
        throw PSIEXCEPTION("Inconsistent number of roots in p_space and civecs.");
    }

    // compute state-averaged density
    outfile->Printf("\n    Compute SA density matrix with equal weights 1/%d.", nroots);

    auto actv_mos = mo_space_info_->get_corr_abs_mo("ACTIVE");
    auto core_mos = mo_space_info_->get_corr_abs_mo("RESTRICTED_DOCC");

    size_t nc = mo_space_info_->size("RESTRICTED_DOCC");
    size_t na = mo_space_info_->size("ACTIVE");
    std::vector<double> sa_opdm_a(na * na, 0.0);
    std::vector<double> sa_opdm_b(na * na, 0.0);

    for (int M = 0; M < nroots; ++M) {
        CI_RDMS ci_rdms(options_, fci_ints_, p_space, civecs, M, M);
        std::vector<double> opdm_a, opdm_b;
        ci_rdms.compute_1rdm(opdm_a, opdm_b);

        std::transform(sa_opdm_a.begin(), sa_opdm_a.end(), opdm_a.begin(), sa_opdm_a.begin(),
                       std::plus<double>());
        std::transform(sa_opdm_b.begin(), sa_opdm_b.end(), opdm_b.begin(), sa_opdm_b.begin(),
                       std::plus<double>());
    }
    std::for_each(sa_opdm_a.begin(), sa_opdm_a.end(), [&](double& v) { v /= nroots; });
    std::for_each(sa_opdm_b.begin(), sa_opdm_b.end(), [&](double& v) { v /= nroots; });

    ambit::Tensor Da = ambit::Tensor::build(CoreTensor, "Da", {na, na});
    Da.data() = std::move(sa_opdm_a);

    ambit::Tensor Db = ambit::Tensor::build(CoreTensor, "Db", {na, na});
    Db.data() = std::move(sa_opdm_b);

    // form Fock matrix within active space
    Fa = ambit::Tensor::build(CoreTensor, "Fa", {na, na});
    Fb = ambit::Tensor::build(CoreTensor, "Fb", {na, na});

    Fa.iterate([&](const std::vector<size_t>& i, double& value) {
        size_t nu = actv_mos[i[0]];
        size_t nv = actv_mos[i[1]];
        value = ints_->oei_a(nu, nv);
    });

    Fb.iterate([&](const std::vector<size_t>& i, double& value) {
        size_t nu = actv_mos[i[0]];
        size_t nv = actv_mos[i[1]];
        value = ints_->oei_b(nu, nv);
    });

    ambit::Tensor I = ambit::Tensor::build(CoreTensor, "Identity", {nc, nc});
    for (int m = 0; m < nc; ++m) {
        I.data()[m * nc + m] = 1.0;
    }

    ambit::Tensor V;
    V = ints_->aptei_aa_block(actv_mos, core_mos, actv_mos, core_mos);
    Fa("uv") += V("umvn") * I("mn");

    V = ints_->aptei_ab_block(actv_mos, core_mos, actv_mos, core_mos);
    Fa("uv") += V("umvn") * I("mn");

    V = ints_->aptei_ab_block(core_mos, actv_mos, core_mos, actv_mos);
    Fb("uv") += V("munv") * I("mn");

    V = ints_->aptei_bb_block(actv_mos, core_mos, actv_mos, core_mos);
    Fb("uv") += V("umvn") * I("mn");

    V = ints_->aptei_aa_block(actv_mos, actv_mos, actv_mos, actv_mos);
    Fa("uv") += V("uxvy") * Da("xy");

    V = ints_->aptei_ab_block(actv_mos, actv_mos, actv_mos, actv_mos);
    Fa("uv") += V("uxvy") * Db("xy");
    Fb("uv") += V("xuyv") * Da("xy");

    V = ints_->aptei_bb_block(actv_mos, actv_mos, actv_mos, actv_mos);
    Fb("uv") += V("uxvy") * Db("xy");
}

SharedMatrix DWMS_DSRGPT2::xms_rotate_civecs(const det_vec& p_space, SharedMatrix civecs,
                                             ambit::Tensor Fa, ambit::Tensor Fb) {
    int nroots = civecs->ncol();
    outfile->Printf("\n    Build Fock matrix <M|F|N>.");
    SharedMatrix Fock(new Matrix("Fock <M|F|N>", nroots, nroots));

    size_t na = mo_space_info_->size("ACTIVE");

    for (int M = 0; M < nroots; ++M) {
        for (int N = M; N < nroots; ++N) {

            // compute transition density
            std::vector<double> opdm_a, opdm_b;
            CI_RDMS ci_rdms(options_, fci_ints_, p_space, civecs, M, N);
            ci_rdms.compute_1rdm(opdm_a, opdm_b);

            // put rdms in tensor format
            ambit::Tensor Da = ambit::Tensor::build(CoreTensor, "Da", {na, na});
            ambit::Tensor Db = ambit::Tensor::build(CoreTensor, "Da", {na, na});
            Da.data() = std::move(opdm_a);
            Db.data() = std::move(opdm_b);

            // compute Fock elements
            double F_MN = 0.0;
            F_MN += Da("uv") * Fa("vu");
            F_MN += Db("UV") * Fb("VU");
            Fock->set(M, N, F_MN);
            if (M != N) {
                Fock->set(N, M, F_MN);
            }
        }
    }
    Fock->print();

    // diagonalize Fock
    SharedMatrix Fevec(new Matrix("Fock Evec", nroots, nroots));
    SharedVector Feval(new Vector("Fock Eval", nroots));
    Fock->diagonalize(Fevec, Feval);
    Fevec->eivprint(Feval);

    // Rotate ci vecs
    SharedMatrix rcivecs(civecs->clone());
    rcivecs->zero();
    rcivecs->gemm(false, false, 1.0, civecs, Fevec, 0.0);

    return rcivecs;
}

Reference DWMS_DSRGPT2::compute_Reference(CI_RDMS& ci_rdms, bool do_cumulant) {
    size_t na = mo_space_info_->size("ACTIVE");

    std::string job_type = "RDM";
    if (do_cumulant) {
        job_type = "PDC";
    }

    Reference ref;
    std::vector<double> opdm_a, opdm_b;
    std::vector<double> tpdm_aa, tpdm_ab, tpdm_bb;
    std::vector<double> tpdm_aaa, tpdm_aab, tpdm_abb, tpdm_bbb;

    // 1-RDM
    if (max_rdm_level_ >= 1) {
        ForteTimer timer;
        outfile->Printf("\n  Computing 1-%ss ... ", job_type.c_str());

        ci_rdms.compute_1rdm(opdm_a, opdm_b);
        ref.set_G1(opdm_a, opdm_b, na);

        outfile->Printf("Done. Timing %15.6f s\n", timer.elapsed());
    }

    // 2-RDM
    if (max_rdm_level_ >= 2) {
        ForteTimer timer;
        outfile->Printf("\n  Computing 2-%ss ... ", job_type.c_str());

        ci_rdms.compute_2rdm(tpdm_aa, tpdm_ab, tpdm_bb);
        ref.set_G2(tpdm_aa, tpdm_ab, tpdm_bb, na, do_cumulant);

        outfile->Printf("Done. Timing %15.6f s\n", timer.elapsed());
    }

    // 3-RDM
    if (max_rdm_level_ >= 3) {
        ForteTimer timer;
        outfile->Printf("\n  Computing 3-ss ... ", job_type.c_str());

        ci_rdms.compute_3rdm(tpdm_aaa, tpdm_aab, tpdm_abb, tpdm_bbb);
        ref.set_G3(tpdm_aaa, tpdm_aab, tpdm_abb, tpdm_bbb, na, do_cumulant);

        outfile->Printf("Done. Timing %15.6f s\n", timer.elapsed());
    }

    return ref;
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

std::shared_ptr<FCI_MO> DWMS_DSRGPT2::precompute_energy_old() {
    // perform SA-CASCI using user-defined weights
    auto fci_mo =
        std::make_shared<FCI_MO>(reference_wavefunction_, options_, ints_, mo_space_info_);
    fci_mo->compute_energy();
    fci_ints_ = fci_mo->fci_ints();

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

        fci_ints_ = compute_dsrg_pt2(fci_mo, reference);

        print_h2("Diagonalize SA-DSRG-PT2 Active Hamiltonian");
        fci_mo->set_fci_int(fci_ints_);
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

double DWMS_DSRGPT2::compute_dwms_energy_old() {
    // perform SA-CASCI and SA-DSRG-PT2 using user-defined weights
    std::shared_ptr<FCI_MO> fci_mo = precompute_energy_old();
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
}
}
