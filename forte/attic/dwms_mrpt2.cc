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

#include <tuple>
#include <sstream>
#include <iomanip>

#include "psi4/libpsi4util/PsiOutStream.h"
#include "psi4/libpsi4util/process.h"
#include "psi4/libmints/molecule.h"
#include "psi4/libmints/dimension.h"

#include "dwms_mrpt2.h"
#include "master_mrdsrg.h"
#include "dsrg_mrpt2.h"
#include "dsrg_mrpt3.h"
#include "three_dsrg_mrpt2.h"
#include "orbital-helpers/semi_canonicalize.h"
#include "helpers/disk_io.h"
#include "helpers/printing.h"

namespace forte {

DWMS_DSRGPT2::DWMS_DSRGPT2(std::shared_ptr<SCFInfo> scf_info, std::shared_ptr<ForteOptions> options,
                           std::shared_ptr<ForteIntegrals> ints,
                           std::shared_ptr<MOSpaceInfo> mo_space_info)
    : ints_(ints), mo_space_info_(mo_space_info), scf_info_(scf_info), foptions_(options) {
    print_method_banner({"Dynamically Weighted Driven Similarity Renormalization Group",
                         "Multi-State Perturbation Theory", "Chenyang Li"});
    startup();
}

DWMS_DSRGPT2::~DWMS_DSRGPT2() {}

void DWMS_DSRGPT2::startup() {
    read_options();
    test_options();
    print_options();

    print_impl_note();

    Enuc_ = ints_->nuclear_repulsion_energy();

    irrep_symbol_ = mo_space_info_->irrep_labels();

    multi_symbol_ = std::vector<std::string>{"Singlet", "Doublet", "Triplet", "Quartet", "Quintet",
                                             "Sextet",  "Septet",  "Octet",   "Nonet",   "Decaet"};

    size_t na = mo_space_info_->size("ACTIVE");
    Ua_ = ambit::Tensor::build(ambit::CoreTensor, "Uactv a", {na, na});
    Ub_ = ambit::Tensor::build(ambit::CoreTensor, "Uactv b", {na, na});
    for (size_t u = 0; u < na; ++u) {
        Ua_.data()[u * na + u] = 1.0;
        Ub_.data()[u * na + u] = 1.0;
    }

    Ca_copy_ = ints_->Ca()->clone();
    Cb_copy_ = ints_->Cb()->clone();
}

void DWMS_DSRGPT2::read_options() {
    dwms_corrlv_ = foptions_->get_str("DWMS_CORRLV");
    zeta_ = foptions_->get_double("DWMS_ZETA");
    algorithm_ = foptions_->get_str("DWMS_ALGORITHM");
    dwms_ref_ = foptions_->get_str("DWMS_REFERENCE");
    do_delta_amp_ = foptions_->get_bool("DWMS_DELTA_AMP");
    dwms_iterate_ = foptions_->get_bool("DWMS_ITERATE");
    dwms_maxiter_ = foptions_->get_int("DWMS_MAXITER");
    dwms_e_convergence_ = foptions_->get_double("DWMS_E_CONVERGENCE");

    do_hbar3_ = foptions_->get_bool("FORM_HBAR3");
    max_hbar_level_ = do_hbar3_ ? 3 : 2;
    max_rdm_level_ = (foptions_->get_str("THREEPDC") == "ZERO") ? 2 : 3;

    IntegralType int_type = ints_->integral_type();
    eri_df_ = (int_type == Cholesky) || (int_type == DF) || (int_type == DiskDF);
}

void DWMS_DSRGPT2::print_options() {
    print_h2("DWMS-DSRG Options");
    outfile->Printf("\n    CORRELATION LEVEL  %10s", dwms_corrlv_.c_str());
    outfile->Printf("\n    ZETA VALUE         %10.2e", zeta_);
    outfile->Printf("\n    REFERENCE          %10s", dwms_ref_.c_str());
    outfile->Printf("\n    ALGORITHM          %10s", algorithm_.c_str());

    if (algorithm_ == "SA" || algorithm_ == "XSA") {
        std::string damps = do_delta_amp_ ? "TRUE" : "FALSE";
        outfile->Printf("\n    DO DELTA AMPS      %10s", damps.c_str());
    }

    if (dwms_iterate_) {
        outfile->Printf("\n    MAX ITERATION      %10d", dwms_maxiter_);
        outfile->Printf("\n    E_CONVERGENCE      %10.2e", dwms_e_convergence_);
    }
}

void DWMS_DSRGPT2::test_options() {
    if (zeta_ < 0.0) {
        throw psi::PSIEXCEPTION("DWMS_ZETA should be a value greater or equal than 0.0!");
    }

    std::string actv_type = foptions_->get_str("FCIMO_ACTV_TYPE");
    if (actv_type == "CIS" || actv_type == "CISD") {
        throw psi::PSIEXCEPTION("VCIS and VCISD are not supported for DWMS-DSRG-PT yet!");
    }

    if (do_hbar3_ && (dwms_ref_ == "PT3" || dwms_corrlv_ == "PT3")) {
        throw psi::PSIEXCEPTION("DSRG-MRPT3 does not support FORM_HBAR3 yet!");
    }

    if (dwms_corrlv_ == "PT3" && (algorithm_ == "MS" || algorithm_ == "XMS")) {
        throw psi::PSIEXCEPTION("DWMS-DSRG-PT3 does not support MS or XMS algorithm yet!");
    }

    if (do_delta_amp_) {
        if (eri_df_ && dwms_corrlv_ == "PT2") {
            throw psi::PSIEXCEPTION("DF-DSRG-MRPT2 does not support DWMS_DELTA_AMP = TRUE!");
        }
        if (dwms_corrlv_ == "PT3") {
            throw psi::PSIEXCEPTION("DSRG-MRPT3 does not support DWMS_DELTA_AMP = TRUE!");
        }
        if (!do_hbar3_) {
            throw psi::PSIEXCEPTION("3-body terms should be included when DWMS_DELTA_AMP = TRUE!");
        }
    }
}

void DWMS_DSRGPT2::print_impl_note() {
    print_h2("Implementation Note on " + algorithm_);
    if (algorithm_ == "SA" || algorithm_ == "XSA") {
        print_impl_note_sa();
    } else if (algorithm_ == "MS" || algorithm_ == "XMS") {
        print_impl_note_ms();
    } else {
        print_impl_note_sH();
    }
}

void DWMS_DSRGPT2::print_impl_note_sa() {
    outfile->Printf("\n  - Perform CASCI using user-defined weights.");

    if (dwms_ref_ != "CASCI") {
        outfile->Printf("\n  - Perform SA-DSRG-%s using user-defined weights.", dwms_ref_.c_str());
    }

    if (algorithm_ == "XSA") {
        outfile->Printf("\n  - Perform XMS rotation on reference CI vectors.");
    }

    outfile->Printf("\n  - Loop over symmetry entries:");
    outfile->Printf("\n    - Initialize effective Hamiltonian Heff.");
    outfile->Printf("\n    - Loop over averaged states:");
    outfile->Printf("\n      - Compute new density weights if ζ is nonzero.");
    outfile->Printf("\n      - Semicanonicalize orbitals.");
    outfile->Printf("\n      - Put SS-DSRG-PT2 energy to diagonal of Heff.");
    outfile->Printf("\n      - Compute couplings of Heff:");
    outfile->Printf("\n        - De-normal-order Hbar.");
    outfile->Printf("\n        - Compute 1, 2, 3 transition densities.");
    outfile->Printf("\n        - <A|Heff|B> <- 0.5 * <A|Hbar(B)|B>.");
    outfile->Printf("\n  - Collect and print energies.\n");
}

void DWMS_DSRGPT2::print_impl_note_ms() {
    outfile->Printf("\n  - Perform CASCI using user-defined weights.");

    if (dwms_ref_ != "CASCI") {
        outfile->Printf("\n  - Perform SA-DSRG-%s using user-defined weights.", dwms_ref_.c_str());
    }

    if (algorithm_ == "XMS") {
        outfile->Printf("\n  - Perform XMS rotation on reference CI vectors.");
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

void DWMS_DSRGPT2::print_impl_note_sH() {
    bool avg = algorithm_.find("AVG") != std::string::npos;

    outfile->Printf("\n  - Perform CASCI using user-defined weights.");
    if (!avg) {
        outfile->Printf(" (*)");
    }

    if (algorithm_ != "DWMS-0") {
        outfile->Printf("\n  - Perform SA-DSRG-PT2 using user-defined weights.");
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
        outfile->Printf("\n  - Initial guess from SA-DSRG-PT2/CASCI for Davidson-Liu solver.");
        outfile->Printf("\n  - CI vectors of previous roots are projected out from current root.");
        outfile->Printf("\n  - CI vectors should be orthogonal between states.");
    }
}

double DWMS_DSRGPT2::compute_energy() {
    // perform CASCI or SA-DSRG-PT2/3
    std::shared_ptr<FCI_MO> fci_mo = precompute_energy();

    // diagonalize Fock matrix in contracted CI basis
    if (algorithm_ == "XSA" || algorithm_ == "XMS") {
        fci_mo->xms_rotate_civecs();
    }

    // compute energy
    do_semi_ = false;
    std::string name = "DW-DSRG-" + algorithm_;
    if (algorithm_ == "MS" || algorithm_ == "XMS") {
        compute_dwms_energy(fci_mo);
    } else if (algorithm_ == "SA" || algorithm_ == "XSA") {
        if (dwms_iterate_) {
            compute_dwsa_energy_iterate(fci_mo);
        } else {
            compute_dwsa_energy(fci_mo);
        }
    } else {
        name = "DWMS(sH)-DSRG-";
        compute_dwms_energy_separated_H(fci_mo);
    }

    // print energy summary
    print_h2("Dynamically Weighted Driven Similarity Renormalization Group Energy Summary");
    auto sa_info = fci_mo->sa_info();
    print_energy_list("CASCI", Eref_0_, sa_info);
    if (dwms_ref_ != "CASCI") {
        print_energy_list("SA-DSRG-" + dwms_ref_, Ept_0_, sa_info);
    }
    print_energy_list(name + dwms_corrlv_, Ept_, sa_info, true);

    return Ept_[0][0];
}

std::shared_ptr<FCI_MO> DWMS_DSRGPT2::precompute_energy() {
    // perform CASCI using user-defined weights
    auto fci_mo = std::make_shared<FCI_MO>(scf_info_, foptions_, ints_, mo_space_info_);
    fci_mo->compute_energy();
    auto eigens = fci_mo->eigens();
    fci_ints_ = fci_mo->fci_ints();

    auto sa_info = fci_mo->sa_info();
    int nentry = sa_info.size();

    std::vector<int> nrootspi(nentry);
    for (int n = 0; n < nentry; ++n) {
        int nroots;
        std::tie(std::ignore, std::ignore, nroots, std::ignore) = sa_info[n];
        nrootspi[n] = nroots;
    }

    // save CASCI energies
    Eref_0_.resize(nentry);
    for (int n = 0; n < nentry; ++n) {
        for (int i = 0, nroots = nrootspi[n]; i < nroots; ++i) {
            Eref_0_[n].push_back(eigens[n][i].second);
        }
    }

    // perform SA-DSRG-PT2/3 if needed
    if (dwms_ref_ != "CASCI") {
        auto rdms = fci_mo->reference(max_rdm_level_); // TODO: bug here? (Francesco)

        std::shared_ptr<MASTER_DSRG> dsrg_pt;
        fci_ints_ = compute_dsrg_pt(dsrg_pt, rdms, dwms_ref_);

        Ept_0_.resize(nentry);

        if (dwms_ref_ == "PT2D") {
            BlockedTensor oei =
                ambit::BlockedTensor::build(ambit::CoreTensor, "oei", spin_cases({"aa"}));
            oei.block("aa").data() = fci_ints_->oei_a_vector();
            oei.block("AA").data() = fci_ints_->oei_b_vector();

            BlockedTensor tei =
                ambit::BlockedTensor::build(ambit::CoreTensor, "tei", spin_cases({"aaaa"}));
            tei.block("aaaa").data() = fci_ints_->tei_aa_vector();
            tei.block("aAaA").data() = fci_ints_->tei_ab_vector();
            tei.block("AAAA").data() = fci_ints_->tei_bb_vector();

            BlockedTensor D1 =
                ambit::BlockedTensor::build(ambit::CoreTensor, "D1", spin_cases({"aa"}));
            BlockedTensor D2 =
                ambit::BlockedTensor::build(ambit::CoreTensor, "D2", spin_cases({"aaaa"}));

            for (int n = 0; n < nentry; ++n) {
                int multi, irrep, nroots;
                std::tie(irrep, multi, nroots, std::ignore) = sa_info[n];

                for (int A = 0; A < nroots; ++A) {
                    double value = 0.0;

                    auto filenames = fci_mo->density_filenames_generator(1, irrep, multi, A, A);
                    bool files_exist = fci_mo->check_density_files_fcimo(1, irrep, multi, A, A);
                    if (files_exist) {
                        read_disk_vector_double(filenames[0], D1.block("aa").data());
                        read_disk_vector_double(filenames[1], D1.block("AA").data());
                    }
                    value += oei["uv"] * D1["uv"];
                    value += oei["UV"] * D1["UV"];

                    filenames = fci_mo->density_filenames_generator(2, irrep, multi, A, A);
                    files_exist = fci_mo->check_density_files_fcimo(2, irrep, multi, A, A);
                    if (files_exist) {
                        read_disk_vector_double(filenames[0], D2.block("aaaa").data());
                        read_disk_vector_double(filenames[1], D2.block("aAaA").data());
                        read_disk_vector_double(filenames[2], D2.block("AAAA").data());
                    }
                    value += 0.25 * tei["uvxy"] * D2["xyuv"];
                    value += 0.25 * tei["UVXY"] * D2["XYUV"];
                    value += tei["uVxY"] * D2["xYuV"];

                    outfile->Printf("\n  SA-DSRG-PT2c Diagonal Hbar (Active) %20.15f", value);
                    Ept_0_[n].push_back(value);
                }
            }
        } else {
            std::stringstream title;
            title << "Diagonalize SA-DSRG-" << dwms_ref_ << " Active Hamiltonian";
            print_h2(title.str());
            fci_mo->set_fci_int(fci_ints_);
            fci_mo->set_localize_actv(false);
            fci_mo->compute_energy();
            eigens = fci_mo->eigens();
            fci_mo->set_safe_to_read_density_files(false);

            // save SA-DSRG-PT2/3 energies
            Ept_0_.resize(nentry);
            for (int n = 0; n < nentry; ++n) {
                for (int i = 0, nroots = nrootspi[n]; i < nroots; ++i) {
                    Ept_0_[n].push_back(eigens[n][i].second);
                }
            }
        }
    }

    // save CI vectors if orthogonalized separated diagonalizations
    if (algorithm_ == "SH-1") {
        initial_guesses_.resize(nentry);
        for (int n = 0; n < nentry; ++n) {
            for (int i = 0, nroots = nrootspi[n]; i < nroots; ++i) {
                initial_guesses_[n].push_back(eigens[n][i].first);
            }
        }
    }

    return fci_mo;
}

std::shared_ptr<ActiveSpaceIntegrals>
DWMS_DSRGPT2::compute_dsrg_pt(std::shared_ptr<MASTER_DSRG>& dsrg_pt, std::shared_ptr<RDMs> rdms,
                              std::string level) {
    // use semicanonical orbitals only for THREE-DSRG-MRPT2
    do_semi_ = (level.find("PT2") != std::string::npos) && eri_df_;

    // compute dsrg-pt2/3 energy
    if (do_semi_) {
        SemiCanonical semi(mo_space_info_, ints_, foptions_);
        semi.semicanonicalize(rdms, max_rdm_level_);
        Ua_ = semi.Ua_t();
        Ub_ = semi.Ub_t();

        dsrg_pt =
            std::make_shared<THREE_DSRG_MRPT2>(rdms, scf_info_, foptions_, ints_, mo_space_info_);
        dsrg_pt->set_Uactv(Ua_, Ub_);
    } else {
        if (level == "PT3") {
            dsrg_pt =
                std::make_shared<DSRG_MRPT3>(rdms, scf_info_, foptions_, ints_, mo_space_info_);
        } else {
            dsrg_pt =
                std::make_shared<DSRG_MRPT2>(rdms, scf_info_, foptions_, ints_, mo_space_info_);
        }
    }

    dsrg_pt->compute_energy();
    auto fci_ints = dsrg_pt->compute_Heff_actv();

    return fci_ints;
}

std::shared_ptr<ActiveSpaceIntegrals>
DWMS_DSRGPT2::compute_macro_dsrg_pt(std::shared_ptr<MASTER_DSRG>& dsrg_pt,
                                    std::shared_ptr<FCI_MO> fci_mo, int entry, int root) {
    auto sa_info = fci_mo->sa_info();

    // compute new weights
    std::vector<std::vector<double>> Erefs = (dwms_ref_ == "CASCI") ? Eref_0_ : Ept_0_;
    if (dwms_iterate_ && dwms_niter_ != 1) {
        Erefs = Ept_;
    }
    auto sa_info_new = compute_dwms_weights(sa_info, entry, root, Erefs);
    print_sa_info("Original State Averaging Summary", sa_info);
    print_sa_info("Reweighted State Averaging Summary", sa_info_new);

    // compute RDMs
    fci_mo->set_sa_info(sa_info_new);
    auto rdms = fci_mo->reference(max_rdm_level_); // TODO: bug here? (Francesco)

    // update MK vacuum energy
    // double new_Eref = compute_Eref_from_rdms(reference, ints_, mo_space_info_, Enuc_);
    // rdms.set_Eref(new_Eref); // TODO: ?why do this here this way?

    // compute DSRG-PT2/3 energies and Hbar
    return compute_dsrg_pt(dsrg_pt, rdms, dwms_corrlv_);
}

void DWMS_DSRGPT2::compute_dwsa_energy_iterate(std::shared_ptr<FCI_MO>& fci_mo) {
    auto sa_info = fci_mo->sa_info();
    int nentry = sa_info.size();
    std::vector<std::vector<double>> Eold;
    for (int i = 0; i < nentry; ++i) {
        Eold.push_back(std::vector<double>(Eref_0_[i].size(), 0.0));
    }

    bool converged = true, failed = true;
    std::string dash(70, '-');
    do {
        print_h2("Start DWMS Iteration " + std::to_string(dwms_niter_));
        compute_dwsa_energy(fci_mo);

        print_h2("DWMS Iteration " + std::to_string(dwms_niter_) + " Summary");
        outfile->Printf("\n    Multi.  Irrep.  No. %18s  %18s  %10s", "E new", "E old", "E diff");
        outfile->Printf("\n    %s", dash.c_str());

        for (int n = 0, counter = 0, nentry = Ept_.size(); n < nentry; ++n) {
            int irrep, multi, nroots;
            std::tie(irrep, multi, nroots, std::ignore) = sa_info[n];

            for (int i = 0; i < nroots; ++i) {
                double diff = Ept_[n][i] - Eold[n][i];
                if (fabs(diff) > dwms_e_convergence_) {
                    converged = false;
                }

                outfile->Printf("\n     %3d     %3s    %2d  %18.10f  %18.10f  %10.2e", multi,
                                irrep_symbol_[irrep].c_str(), i, Ept_[n][i], Eold[n][i], diff);

                std::stringstream ss;
                ss << "ITER " << dwms_niter_ << " ENERGY ROOT " << counter;
                psi::Process::environment.globals[ss.str()] = Ept_[n][i];
                counter += 1;
            }
            outfile->Printf("\n    %s", dash.c_str());
        }
        outfile->Printf("\n");

        if (converged) {
            failed = false;
            break;
        } else {
            converged = true;
        }

        Eold = Ept_;
    } while (dwms_niter_++ < dwms_maxiter_);

    if (converged && (!failed)) {
        outfile->Printf("\n  !! DWMS iterations converged in %d runs !!", dwms_niter_);
    } else {
        outfile->Printf("\n  !! DWMS iterations did not converge in %d runs !!", dwms_maxiter_);

        print_h2("Dynamically Weighted Driven Similarity Renormalization Group Energy Summary");
        auto sa_info = fci_mo->sa_info();
        print_energy_list("CASCI", Eref_0_, sa_info);
        if (dwms_ref_ != "CASCI") {
            print_energy_list("SA-DSRG-" + dwms_ref_, Ept_0_, sa_info);
        }
        print_energy_list("DW-DSRG" + dwms_corrlv_, Ept_, sa_info, true);

        throw psi::PSIEXCEPTION("DWMS iterations did not converge!");
    }
}

void DWMS_DSRGPT2::compute_dwsa_energy(std::shared_ptr<FCI_MO>& fci_mo) {
    // prepare the final DWMS-DSRG-PT2/3 energies
    auto sa_info = fci_mo->sa_info();
    int nentry = sa_info.size();
    Ept_.resize(nentry);

    // eigens for iterative procedure
    std::vector<std::vector<std::pair<std::shared_ptr<psi::Vector>, double>>> eigens_new;
    // old eigens
    auto eigens = fci_mo->eigens();

    std::shared_ptr<MASTER_DSRG> dsrg_pt;
    std::shared_ptr<ActiveSpaceIntegrals> fci_ints;

    // if zeta == 0, just transform Hamiltonian once
    if (zeta_ == 0.0) {
        print_h2("Important Note of DW-DSRG");
        outfile->Printf("\n  DWMS_ZETA is detected to be 0.0.");
        outfile->Printf("\n  The bare Hamiltonian will be transformed ONLY once.");

        fci_ints = compute_macro_dsrg_pt(dsrg_pt, fci_mo, 0, 0);
    }

    // loop over symmetry entries
    for (int n = 0; n < nentry; ++n) {
        int multi, irrep, nroots;
        std::tie(irrep, multi, nroots, std::ignore) = sa_info[n];
        Ept_[n].resize(nroots);

        // print current status
        std::string entry_name = multi_symbol_[multi - 1] + " " + irrep_symbol_[irrep];
        std::string entry_title = "Build Effective Hamiltonian of " + entry_name;
        print_title(entry_title);

        // prepare Heff
        auto Heff = std::make_shared<psi::Matrix>("Heff " + entry_name, nroots, nroots);

        // vector of T1, T2, and summed 1st-order Hbar
        std::vector<ambit::BlockedTensor> T1s, T2s, RH1s, RH2s;

        // loop over states of current symmetry
        for (int M = 0; M < nroots; ++M) {

            // transform bare Hamiltonian for each root
            if (zeta_ != 0.0) {
                print_h2("Compute DSRG-MR" + dwms_corrlv_ + " Energy of Root " + std::to_string(M));
                if (do_semi_) {
                    transform_ints0();
                }
                fci_ints = compute_macro_dsrg_pt(dsrg_pt, fci_mo, n, M);

                if (do_delta_amp_) {
                    double temp = 0.0;
                    T1s.push_back(dsrg_pt->get_T1deGNO(temp));
                    T2s.push_back(dsrg_pt->get_T2());
                    RH1s.push_back(dsrg_pt->get_RH1deGNO());
                    RH2s.push_back(dsrg_pt->get_RH2());
                }
            }

            print_h2("Compute Elements of Effective Hamiltonian");
            for (int N = 0; N < nroots; ++N) {
                std::string msg = (M == N) ? "densities" : "transition densities";
                outfile->Printf("\n  Compute %s of %d->%d.", msg.c_str(), M, N);

                int MM = M, NN = N;
                if (N < M) {
                    MM = N, NN = M;
                    outfile->Printf(" Use Hermiticity of Hbar.");
                }

                // compute transition densities
                std::shared_ptr<RDMs> TrD;
                TrD = fci_mo->transition_reference(MM, NN, true, n, max_hbar_level_, false);

                outfile->Printf("\n  Contract %s with Heff.", msg.c_str());
                double coupling = 0.0;

                auto Hbar_vec = dsrg_pt->Hbar(1);
                coupling += contract_Heff_1TrDM(Hbar_vec[0], Hbar_vec[1], TrD, false);

                Hbar_vec = dsrg_pt->Hbar(2);
                coupling += contract_Heff_2TrDM(Hbar_vec[0], Hbar_vec[1], Hbar_vec[2], TrD, false);

                if (do_hbar3_) {
                    Hbar_vec = dsrg_pt->Hbar(3);
                    coupling += contract_Heff_3TrDM(Hbar_vec[0], Hbar_vec[1], Hbar_vec[2],
                                                    Hbar_vec[3], TrD, false);
                }

                if (M == N) {
                    double shift = ints_->frozen_core_energy() + Enuc_ + fci_ints->scalar_energy();
                    Heff->set(M, M, coupling + shift);
                } else {
                    Heff->add(N, M, 0.5 * coupling);
                    Heff->add(M, N, 0.5 * coupling);
                }
            }
        }

        // print effective Hamiltonian
        print_h2("Effective Hamiltonian Summary");
        outfile->Printf("\n");
        Heff->print();

        // diagonalize Heff and print eigen vectors
        auto U = std::make_shared<psi::Matrix>("U of Heff", nroots, nroots);
        auto Ems = std::make_shared<psi::Vector>("MS Energies", nroots);
        Heff->diagonalize(U, Ems);
        U->eivprint(Ems);

        // corrections to Heff from delta amplitudes
        if (do_delta_amp_ && do_hbar3_) {
            outfile->Printf("\n  Compute corrections of Heff due to delta amplitudes.");
            for (int M = 0; M < nroots; ++M) {
                auto& T1M = T1s[M];
                auto& T2M = T2s[M];
                for (int N = M + 1; N < nroots; ++N) {
                    auto TrD = fci_mo->transition_reference(M, N, true, n, max_hbar_level_, false);

                    T1M["ia"] -= T1s[N]["ia"];
                    T1M["IA"] -= T1s[N]["IA"];
                    T2M["ijab"] -= T2s[N]["ijab"];
                    T2M["iJaB"] -= T2s[N]["iJaB"];
                    T2M["IJAB"] -= T2s[N]["IJAB"];

                    double coupling = 0.0;
                    dsrgHeff H = dsrg_pt->commutator_HT_noGNO(RH1s[N], RH2s[N], T1M, T2M);
                    coupling += contract_Heff_1TrDM(H.H1a, H.H1b, TrD, false);
                    coupling += contract_Heff_2TrDM(H.H2aa, H.H2ab, H.H2bb, TrD, false);
                    coupling += contract_Heff_3TrDM(H.H3aaa, H.H3aab, H.H3abb, H.H3bbb, TrD, false);

                    H = dsrg_pt->commutator_HT_noGNO(RH1s[M], RH2s[M], T1M, T2M);
                    coupling -= contract_Heff_1TrDM(H.H1a, H.H1b, TrD, true);
                    coupling -= contract_Heff_2TrDM(H.H2aa, H.H2ab, H.H2bb, TrD, true);
                    coupling -= contract_Heff_3TrDM(H.H3aaa, H.H3aab, H.H3abb, H.H3bbb, TrD, true);

                    Heff->add(N, M, 0.5 * coupling);
                    Heff->add(M, N, 0.5 * coupling);

                    T1M["ia"] += T1s[N]["ia"];
                    T1M["IA"] += T1s[N]["IA"];
                    T2M["ijab"] += T2s[N]["ijab"];
                    T2M["iJaB"] += T2s[N]["iJaB"];
                    T2M["IJAB"] += T2s[N]["IJAB"];
                }
            }

            print_h2("Delta Amplitudes Corrected Effective Hamiltonian");
            outfile->Printf("\n");
            Heff->print();
            Heff->diagonalize(U, Ems);
            U->eivprint(Ems);
        }

        if (dwms_iterate_ && dwms_niter_ < dwms_maxiter_) {
            eigens_new.push_back(compute_new_eigen(eigens[n], Ems, U));
        }

        // set Ept_ value
        for (int i = 0; i < nroots; ++i) {
            Ept_[n][i] = Ems->get(i);
        }
    }

    // set new eigens to fci_mo
    if (eigens_new.size() != 0) {
        fci_mo->set_eigens(eigens_new);
    }
}

std::vector<std::pair<std::shared_ptr<psi::Vector>, double>> DWMS_DSRGPT2::compute_new_eigen(
    const std::vector<std::pair<std::shared_ptr<psi::Vector>, double>>& old_eigen,
    std::shared_ptr<psi::Vector> new_vals, std::shared_ptr<psi::Matrix> new_vecs) {
    int nroots = new_vals->dim();
    int ndets = (old_eigen[0].first)->dim();
    std::vector<std::pair<std::shared_ptr<psi::Vector>, double>> out;
    out.reserve(nroots);

    for (int i = 0; i < nroots; ++i) {
        auto vec = std::make_shared<Vector>("New Eigen Vector State " + std::to_string(i), ndets);
        auto vec_root = new_vecs->get_column(0, i);
        for (int j = 0; j < nroots; ++j) {
            auto temp = (old_eigen[j].first)->clone();
            temp.scale(vec_root->get(j));
            vec->add(temp);
        }
        out.emplace_back(vec, new_vals->get(i));
    }

    return out;
}

void DWMS_DSRGPT2::compute_dwms_energy(std::shared_ptr<FCI_MO>& fci_mo) {
    // prepare the final DWMS-DSRG-PT2 energies
    auto sa_info = fci_mo->sa_info();
    int nentry = sa_info.size();
    Ept_.resize(nentry);

    std::shared_ptr<MASTER_DSRG> dsrg_pt2;
    std::shared_ptr<ActiveSpaceIntegrals> fci_ints;

    // if zeta == 0, just transform Hamiltonian once
    if (zeta_ == 0.0) {
        print_h2("Important Note of DW-DSRG");
        outfile->Printf("\n  DWMS_ZETA is detected to be 0.0.");
        outfile->Printf("\n  The bare Hamiltonian will be transformed ONLY once.");

        fci_ints = compute_macro_dsrg_pt(dsrg_pt2, fci_mo, 0, 0);
    }

    // loop over symmetry entries
    for (int n = 0; n < nentry; ++n) {
        int multi, irrep, nroots;
        std::tie(irrep, multi, nroots, std::ignore) = sa_info[n];
        Ept_[n].resize(nroots);

        // print current status
        std::string entry_name = multi_symbol_[multi - 1] + " " + irrep_symbol_[irrep];
        std::string entry_title = "Build Effective Hamiltonian of " + entry_name;
        print_title(entry_title);

        // prepare Heff
        auto Heff = std::make_shared<psi::Matrix>("Heff " + entry_name, nroots, nroots);
        auto Heff_sym =
            std::make_shared<psi::Matrix>("Symmetrized Heff " + entry_name, nroots, nroots);

        // loop over states of current symmetry
        for (int M = 0; M < nroots; ++M) {

            // transform bare Hamiltonian for each root
            if (zeta_ != 0.0) {
                print_h2("Compute DSRG-MR" + dwms_corrlv_ + " Energy of Root " + std::to_string(M));
                if (do_semi_) {
                    transform_ints0();
                }
                fci_ints = compute_macro_dsrg_pt(dsrg_pt2, fci_mo, n, M);
            }

            // compute 2nd-order efffective Hamiltonian for the couplings
            print_h2("Compute couplings of 2nd-order effective Hamiltonian");

            outfile->Printf("\n  Compute 2nd-order Heff = H + H * T(root %d).", M);
            double H0 = 0.0;
            ambit::Tensor H1a, H1b, H2aa, H2ab, H2bb, H3aaa, H3aab, H3abb, H3bbb;
            dsrg_pt2->compute_Heff_2nd_coupling(H0, H1a, H1b, H2aa, H2ab, H2bb, H3aaa, H3aab, H3abb,
                                                H3bbb);

            // rotate Heff to original basis if DF
            // no need to do this if not DF since "invariant" form is used in DSRG-MRPT2/3
            if (do_semi_) {
                outfile->Printf("\n  Transform Heff_2nd to original basis.");
                rotate_H1(H1a, H1b);
                rotate_H2(H2aa, H2ab, H2bb);
                rotate_H3(H3aaa, H3aab, H3abb, H3bbb);
            }

            for (int N = 0; N < nroots; ++N) {
                std::string msg = (M == N) ? "densities" : "transition densities";

                // compute transition densities
                outfile->Printf("\n  Compute %s.", msg.c_str());
                auto TrD = (M <= N) ? fci_mo->transition_reference(M, N, true, n, 3, false)
                                    : fci_mo->transition_reference(N, M, true, n, 3, false);
                bool transpose = (M <= N) ? false : true;

                outfile->Printf("\n  Contract %s with Heff.", msg.c_str());
                double coupling = 0.0;

                coupling += contract_Heff_1TrDM(H1a, H1b, TrD, transpose);
                coupling += contract_Heff_2TrDM(H2aa, H2ab, H2bb, TrD, transpose);
                coupling += contract_Heff_3TrDM(H3aaa, H3aab, H3abb, H3bbb, TrD, transpose);

                if (M == N) {
                    double Ediag = fci_ints->scalar_energy();

                    auto Hbar_vec = dsrg_pt2->Hbar(1);
                    Ediag += contract_Heff_1TrDM(Hbar_vec[0], Hbar_vec[1], TrD, false);

                    Hbar_vec = dsrg_pt2->Hbar(2);
                    Ediag += contract_Heff_2TrDM(Hbar_vec[0], Hbar_vec[1], Hbar_vec[2], TrD, false);

                    if (do_hbar3_) {
                        Hbar_vec = dsrg_pt2->Hbar(3);
                        Ediag += contract_Heff_3TrDM(Hbar_vec[0], Hbar_vec[1], Hbar_vec[2],
                                                     Hbar_vec[3], TrD, false);

                        if (!do_semi_) {
                            double Ediff = Ediag - H0 - coupling;
                            outfile->Printf("\n\n  Energy difference of root %d", M);
                            outfile->Printf("\n    Real 2nd-order energy:   %20.15f", Ediag);
                            outfile->Printf("\n    Pseudo 2nd-order energy: %20.15f",
                                            H0 + coupling);
                            outfile->Printf("\n    Energy difference:       %20.15f", Ediff);
                        }
                    }

                    double shift = ints_->frozen_core_energy() + Enuc_;
                    Heff->set(M, M, Ediag + shift);
                    Heff_sym->set(M, M, Ediag + shift);
                } else {
                    Heff->set(N, M, coupling);
                    Heff_sym->add(N, M, 0.5 * coupling);
                    Heff_sym->add(M, N, 0.5 * coupling);
                }
            }

            if (zeta_ != 0.0) {
                dsrg_pt2 = nullptr;
            }
        }

        // print effective Hamiltonian
        print_h2("Effective Hamiltonian Summary");
        outfile->Printf("\n");

        Heff->print();
        Heff_sym->print();

        // diagonalize Heff and print eigen vectors
        auto U = std::make_shared<psi::Matrix>("U of Heff (Symmetrized)", nroots, nroots);
        auto Ems = std::make_shared<psi::Vector>("MS Energies", nroots);
        Heff_sym->diagonalize(U, Ems);
        U->eivprint(Ems);

        // set Ept2_ value
        for (int i = 0; i < nroots; ++i) {
            Ept_[n][i] = Ems->get(i);
        }
    }
}

void DWMS_DSRGPT2::rotate_H1(ambit::Tensor& H1a, ambit::Tensor& H1b) {
    ambit::Tensor temp = H1a.clone();
    H1a("rs") = Ua_("rp") * temp("pq") * Ua_("sq");

    temp("pq") = H1b("pq");
    H1b("rs") = Ub_("rp") * temp("pq") * Ub_("sq");
}

void DWMS_DSRGPT2::rotate_H2(ambit::Tensor& H2aa, ambit::Tensor& H2ab, ambit::Tensor& H2bb) {
    ambit::Tensor temp = H2aa.clone();
    H2aa("pqrs") = Ua_("pa") * Ua_("qb") * temp("abcd") * Ua_("rc") * Ua_("sd");

    temp("pqrs") = H2ab("pqrs");
    H2ab("pqrs") = Ua_("pa") * Ub_("qb") * temp("abcd") * Ua_("rc") * Ub_("sd");

    temp("pqrs") = H2bb("pqrs");
    H2bb("pqrs") = Ub_("pa") * Ub_("qb") * temp("abcd") * Ub_("rc") * Ub_("sd");
}

void DWMS_DSRGPT2::rotate_H3(ambit::Tensor& H3aaa, ambit::Tensor& H3aab, ambit::Tensor& H3abb,
                             ambit::Tensor& H3bbb) {
    ambit::Tensor temp = H3aaa.clone();
    H3aaa("pqrsto") =
        Ua_("pa") * Ua_("qb") * Ua_("rc") * temp("abcdef") * Ua_("sd") * Ua_("te") * Ua_("of");

    temp("pqrsto") = H3aab("pqrsto");
    H3aab("pqrsto") =
        Ua_("pa") * Ua_("qb") * Ub_("rc") * temp("abcdef") * Ua_("sd") * Ua_("te") * Ub_("of");

    temp("pqrsto") = H3abb("pqrsto");
    H3abb("pqrsto") =
        Ua_("pa") * Ub_("qb") * Ub_("rc") * temp("abcdef") * Ua_("sd") * Ub_("te") * Ub_("of");

    temp("pqrsto") = H3bbb("pqrsto");
    H3bbb("pqrsto") =
        Ub_("pa") * Ub_("qb") * Ub_("rc") * temp("abcdef") * Ub_("sd") * Ub_("te") * Ub_("of");
}

double DWMS_DSRGPT2::contract_Heff_1TrDM(ambit::Tensor& H1a, ambit::Tensor& H1b,
                                         std::shared_ptr<RDMs> TrD, bool transpose) {
    double coupling = 0.0;
    std::string indices = transpose ? "vu" : "uv";

    coupling += H1a("vu") * TrD->g1a()(indices);
    coupling += H1b("vu") * TrD->g1b()(indices);

    return coupling;
}

double DWMS_DSRGPT2::contract_Heff_2TrDM(ambit::Tensor& H2aa, ambit::Tensor& H2ab,
                                         ambit::Tensor& H2bb, std::shared_ptr<RDMs> TrD,
                                         bool transpose) {
    double coupling = 0.0;
    std::string indices = transpose ? "uvxy" : "xyuv";

    coupling += 0.25 * H2aa("uvxy") * TrD->g2aa()(indices);
    coupling += H2ab("uvxy") * TrD->g2ab()(indices);
    coupling += 0.25 * H2bb("uvxy") * TrD->g2bb()(indices);

    return coupling;
}

double DWMS_DSRGPT2::contract_Heff_3TrDM(ambit::Tensor& H3aaa, ambit::Tensor& H3aab,
                                         ambit::Tensor& H3abb, ambit::Tensor& H3bbb,
                                         std::shared_ptr<RDMs> TrD, bool transpose) {
    double coupling = 0.0;
    std::string indices = transpose ? "uvwxyz" : "xyzuvw";

    coupling += 1.0 / 36.0 * H3aaa("uvwxyz") * TrD->g3aaa()(indices);
    coupling += 0.25 * H3aab("uvwxyz") * TrD->g3aab()(indices);
    coupling += 0.25 * H3abb("uvwxyz") * TrD->g3abb()(indices);
    coupling += 1.0 / 36.0 * H3bbb("uvwxyz") * TrD->g3bbb()(indices);

    return coupling;
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
        throw psi::PSIEXCEPTION("Mismatching sizes between energy list and sa_info list");
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
                psi::Process::environment.globals["ENERGY ROOT " + std::to_string(counter)] =
                    energy[n][i];
            }
            counter += 1;
        }
        outfile->Printf("\n    %s", dash.c_str());
    }
    outfile->Printf("\n");
}

void DWMS_DSRGPT2::compute_dwms_energy_separated_H(std::shared_ptr<FCI_MO>& fci_mo) {
    // the final DWMS-DSRG-PT2 energies
    auto sa_info = fci_mo->sa_info();
    int nentry = sa_info.size();
    Ept_.resize(nentry);

    // loop over averaged states
    for (int n = 0; n < nentry; ++n) {
        int multi, irrep, nroots;
        std::tie(irrep, multi, nroots, std::ignore) = sa_info[n];
        Ept_[n].resize(nroots);

        // save the re-diagonalized eigen vectors
        std::vector<std::shared_ptr<psi::Vector>> evecs_new;
        evecs_new.resize(nroots);

        // save the previous projected roots
        std::vector<std::vector<std::pair<size_t, double>>> projected_roots;

        for (int i = 0; i < nroots; ++i) {
            // print current title
            std::stringstream current_job;
            current_job << "Current Job: " << multi_symbol_[multi - 1] << " "
                        << irrep_symbol_[irrep] << ", root " << i;
            print_title(current_job.str());

            // compute DSRG-PT2 transformed Hamiltonian
            if (do_semi_) {
                transform_ints0();
            }
            std::shared_ptr<MASTER_DSRG> dsrg_pt;
            auto fci_ints = compute_macro_dsrg_pt(dsrg_pt, fci_mo, n, i);

            // diagonalize the DSRG active Hamiltonian for each root
            print_h2("Diagonalize DWMS(sH)-DSRG Active Hamiltonian");
            fci_mo->set_fci_int(fci_ints);
            fci_mo->set_root_sym(irrep);

            if (algorithm_ == "SH-1") {
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
                fci_mo->set_initial_guess({guess});

                // diagonalize DSRG-PT2 active Hamiltonian
                Ept_[n][i] = fci_mo->compute_ss_energies()[0]; // bc set root = 0

                // since Heff is rotated to the original basis, we can safely store the CI vectors
                evecs_new[i] = fci_mo->eigen()[0].first;

                // append current CI vectors to the projection list
                std::vector<std::pair<size_t, double>> projection;
                for (size_t I = 0, nI = evecs_new[i]->dim(); I < nI; ++I) {
                    projection.push_back(std::make_pair(I, evecs_new[i]->get(I)));
                }
                projected_roots.push_back(projection);
            } else {
                fci_mo->set_nroots(nroots);
                fci_mo->set_root(i);
                Ept_[n] = fci_mo->compute_ss_energies();

                // since Heff is rotated to the original basis, we can safely store the CI vectors
                evecs_new[i] = fci_mo->eigen()[i].first;
            }
        }

        // form and print overlap
        std::string Sname = "Overlap of " + multi_symbol_[multi - 1] + " " + irrep_symbol_[irrep];
        print_overlap(evecs_new, Sname);
    }
}

std::vector<std::tuple<int, int, int, std::vector<double>>> DWMS_DSRGPT2::compute_dwms_weights(
    const std::vector<std::tuple<int, int, int, std::vector<double>>>& sa_info, int entry, int root,
    const std::vector<std::vector<double>>& energy) {

    size_t nentry = sa_info.size();
    if (nentry != energy.size()) {
        throw psi::PSIEXCEPTION("Mismatching sizes between energy list and sa_info list");
    }

    // new weights
    std::vector<std::vector<double>> new_weights;
    new_weights.resize(nentry);

    // new weights for state alpha:
    // w_i = exp(-zeta * (E_alpha - E_i)^2) / sum_j exp(-zeta * (E_alpha - E_j)^2)

    double Ealpha = energy[entry][root];
    double wsum = 0.0;

    for (size_t n = 0; n < nentry; ++n) {
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

    for (size_t n = 0; n < nentry; ++n) {
        for (int i = 0, nroots = new_weights[n].size(); i < nroots; ++i) {
            new_weights[n][i] /= wsum;
        }
    }

    // form new sa_info
    std::vector<std::tuple<int, int, int, std::vector<double>>> out;
    out.resize(nentry);
    for (size_t n = 0; n < nentry; ++n) {
        int irrep, multi, nroots;
        std::tie(irrep, multi, nroots, std::ignore) = sa_info[n];

        out[n] = std::make_tuple(irrep, multi, nroots, new_weights[n]);
    }

    return out;
}

void DWMS_DSRGPT2::transform_ints0() {
    print_h2("Transformation Integrals Back to Original");
    ints_->update_orbitals(Ca_copy_, Cb_copy_);
}

void DWMS_DSRGPT2::print_title(const std::string& title) {
    size_t title_size = title.size();
    outfile->Printf("\n\n  %s", std::string(title_size, '=').c_str());
    outfile->Printf("\n  %s", title.c_str());
    outfile->Printf("\n  %s\n", std::string(title_size, '=').c_str());
}

void DWMS_DSRGPT2::print_overlap(const std::vector<std::shared_ptr<psi::Vector>>& evecs,
                                 const std::string& Sname) {
    print_h2(Sname);
    outfile->Printf("\n");

    int nroots = evecs.size();
    psi::Matrix S("S", nroots, nroots);

    for (int i = 0; i < nroots; ++i) {
        for (int j = i; j < nroots; ++j) {
            double Sij = evecs[i]->vector_dot(*evecs[j]);
            S.set(i, j, Sij);
            S.set(j, i, Sij);
        }
    }

    S.print();
}
} // namespace forte
