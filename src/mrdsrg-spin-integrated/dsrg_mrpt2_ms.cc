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

#include <iomanip>

#include "../ci_rdm/ci_rdms.h"
#include "../fci/fci_solver.h"
#include "../fci_mo.h"
#include "dsrg_mrpt2.h"

namespace psi {
namespace forte {

double DSRG_MRPT2::compute_energy_multi_state() {
    // throw a waring if states with different symmetry
    int nentry = eigens_.size();
    if (nentry > 1) {
        outfile->Printf(
            "\n\n  Warning: States with different symmetry are found in the list of AVG_STATES.");
        outfile->Printf("\n           Each symmetry will be considered separately here.");
    }

    // get character table
    CharacterTable ct = Process::environment.molecule()->point_group()->char_table();
    std::vector<std::string> irrep_symbol;
    for (int h = 0; h < this->nirrep(); ++h) {
        irrep_symbol.push_back(std::string(ct.gamma(h).symbol()));
    }

    // multi-state calculation
    std::vector<std::vector<double>> Edsrg_ms;

    if (multi_state_algorithm_.find("SA") != std::string::npos) {
        Edsrg_ms = compute_energy_sa();
    } else {
        Edsrg_ms = compute_energy_xms();
    }

    // energy summuary
    print_h2("Multi-State DSRG-MRPT2 Energy Summary");

    outfile->Printf("\n    Multi.  Irrep.  No.    DSRG-MRPT2 Energy");
    std::string dash(41, '-');
    outfile->Printf("\n    %s", dash.c_str());

    for (int n = 0, counter = 0; n < nentry; ++n) {
        int irrep = options_["AVG_STATE"][n][0].to_integer();
        int multi = options_["AVG_STATE"][n][1].to_integer();
        int nstates = options_["AVG_STATE"][n][2].to_integer();

        for (int i = 0; i < nstates; ++i) {
            outfile->Printf("\n     %3d     %3s    %2d   %20.12f", multi,
                            irrep_symbol[irrep].c_str(), i, Edsrg_ms[n][i]);
            Process::environment.globals["ENERGY ROOT " + std::to_string(counter)] = Edsrg_ms[n][i];
            ++counter;
        }
        outfile->Printf("\n    %s", dash.c_str());
    }

    Process::environment.globals["CURRENT ENERGY"] = Edsrg_ms[0][0];
    return Edsrg_ms[0][0];
}

std::vector<std::vector<double>> DSRG_MRPT2::compute_energy_sa() {
    // compute DSRG-MRPT2 energy using SA densities
    compute_energy();

    // get character table
    CharacterTable ct = Process::environment.molecule()->point_group()->char_table();
    std::vector<std::string> irrep_symbol;
    for (int h = 0; h < this->nirrep(); ++h) {
        irrep_symbol.push_back(std::string(ct.gamma(h).symbol()));
    }

    // multiplicity table
    std::vector<std::string> multi_label{
        "Singlet", "Doublet", "Triplet", "Quartet", "Quintet", "Sextet", "Septet", "Octet",
        "Nonet",   "Decaet",  "11-et",   "12-et",   "13-et",   "14-et",  "15-et",  "16-et",
        "17-et",   "18-et",   "19-et",   "20-et",   "21-et",   "22-et",  "23-et",  "24-et"};

    // obtain the all-active DSRG transformed Hamiltonian
    auto fci_ints = compute_Heff_actv();

    // get effective one-electron integral (DSRG transformed)
    BlockedTensor oei = BTF_->build(tensor_type_, "temp1", spin_cases({"aa"}));
    oei.block("aa").data() = fci_ints->oei_a_vector();
    oei.block("AA").data() = fci_ints->oei_b_vector();

    // loop over entries of AVG_STATE
    int nentry = eigens_.size();
    std::vector<std::vector<double>> Edsrg_sa(nentry, std::vector<double>());

    // call FCI_MO if SA_FULL and CAS_TYPE == CAS
    if (multi_state_algorithm_ == "SA_FULL" && options_.get_str("CAS_TYPE") == "CAS") {
        FCI_MO fci_mo(reference_wavefunction_, options_, ints_, mo_space_info_, fci_ints);
        fci_mo.set_localize_actv(false);
        fci_mo.compute_energy();
        auto eigens = fci_mo.eigens();
        for (int n = 0; n < nentry; ++n) {
            auto eigen = eigens[n];
            int ni = eigen.size();
            for (int i = 0; i < ni; ++i) {
                Edsrg_sa[n].push_back(eigen[i].second);
            }
        }

        if (do_dm_) {
            // de-normal-order DSRG dipole integrals
            for (int z = 0; z < 3; ++z) {
                std::string name = "Dipole " + dm_dirs_[z] + " Integrals";
                if (options_.get_bool("FORM_MBAR3")) {
                    deGNO_ints(name, Mbar0_[z], Mbar1_[z], Mbar2_[z], Mbar3_[z]);
                    rotate_ints_semi_to_origin(name, Mbar1_[z], Mbar2_[z], Mbar3_[z]);
                } else {
                    deGNO_ints(name, Mbar0_[z], Mbar1_[z], Mbar2_[z]);
                    rotate_ints_semi_to_origin(name, Mbar1_[z], Mbar2_[z]);
                }
            }

            // compute permanent dipoles
            std::map<std::string, std::vector<double>> dm_relax;
            if (options_.get_bool("FORM_MBAR3")) {
                dm_relax = fci_mo.compute_ref_relaxed_dm(Mbar0_, Mbar1_, Mbar2_, Mbar3_);
            } else {
                dm_relax = fci_mo.compute_ref_relaxed_dm(Mbar0_, Mbar1_, Mbar2_);
            }

            print_h2("SA-DSRG-PT2 Dipole Moment (in a.u.) Summary");
            outfile->Printf("\n    %14s  %10s  %10s  %10s", "State", "X", "Y", "Z");
            std::string dash(50, '-');
            outfile->Printf("\n    %s", dash.c_str());
            for (const auto& p : dm_relax) {
                std::stringstream ss;
                ss << std::setw(14) << p.first;
                for (int i = 0; i < 3; ++i) {
                    ss << "  " << std::setw(10) << std::fixed << std::right << std::setprecision(6)
                       << p.second[i] + dm_nuc_[i];
                }
                outfile->Printf("\n    %s", ss.str().c_str());
            }
            outfile->Printf("\n    %s", dash.c_str());

            // oscillator strength
            std::map<std::string, std::vector<double>> osc;
            if (options_.get_bool("FORM_MBAR3")) {
                osc = fci_mo.compute_ref_relaxed_osc(Mbar1_, Mbar2_, Mbar3_);
            } else {
                osc = fci_mo.compute_ref_relaxed_osc(Mbar1_, Mbar2_);
            }

            print_h2("SA-DSRG-PT2 Oscillator Strength (in a.u.) Summary");
            outfile->Printf("\n    %32s  %10s  %10s  %10s  %10s", "State", "X", "Y", "Z", "Total");
            dash = std::string(80, '-');
            outfile->Printf("\n    %s", dash.c_str());
            for (const auto& p : osc) {
                std::stringstream ss;
                ss << std::setw(32) << p.first;
                double total = 0.0;
                for (int i = 0; i < 3; ++i) {
                    ss << "  " << std::setw(10) << std::fixed << std::right << std::setprecision(6)
                       << p.second[i];
                    total += p.second[i];
                }
                ss << "  " << std::setw(10) << std::fixed << std::right << std::setprecision(6)
                   << total;
                outfile->Printf("\n    %s", ss.str().c_str());
            }
            outfile->Printf("\n    %s", dash.c_str());
        }
    } else {

        for (int n = 0; n < nentry; ++n) {
            int irrep = options_["AVG_STATE"][n][0].to_integer();
            int multi = options_["AVG_STATE"][n][1].to_integer();
            int nstates = options_["AVG_STATE"][n][2].to_integer();
            std::vector<psi::forte::Determinant> p_space = p_spaces_[n];

            // print current symmetry
            std::stringstream ss;
            ss << "Diagonalize Effective Hamiltonian (" << multi_label[multi - 1] << " "
               << irrep_symbol[irrep] << ")";
            print_h2(ss.str());

            // diagonalize which the second-order effective Hamiltonian
            // FULL: CASCI using determinants
            // AVG_STATES: H_AB = <A|H|B> where A and B are SA-CAS states
            if (options_.get_str("DSRG_MULTI_STATE") == "SA_FULL") {

                outfile->Printf("\n    Use string FCI code.");

                // prepare FCISolver
                int charge = Process::environment.molecule()->molecular_charge();
                if (options_["CHARGE"].has_changed()) {
                    charge = options_.get_int("CHARGE");
                }
                auto nelec = 0;
                int natom = Process::environment.molecule()->natom();
                for (int i = 0; i < natom; ++i) {
                    nelec += Process::environment.molecule()->fZ(i);
                }
                nelec -= charge;
                int ms = (multi + 1) % 2;
                auto nelec_actv =
                    nelec - 2 * mo_space_info_->size("FROZEN_DOCC") - 2 * core_mos_.size();
                auto na = (nelec_actv + ms) / 2;
                auto nb = nelec_actv - na;

                Dimension active_dim = mo_space_info_->get_dimension("ACTIVE");
                int ntrial_per_root = options_.get_int("NTRIAL_PER_ROOT");

                FCISolver fcisolver(active_dim, core_mos_, actv_mos_, na, nb, multi, irrep, ints_,
                                    mo_space_info_, ntrial_per_root, print_, options_);
                fcisolver.set_max_rdm_level(1);
                fcisolver.set_nroot(nstates);
                fcisolver.set_root(nstates - 1);
                fcisolver.set_fci_iterations(options_.get_int("FCI_MAXITER"));
                fcisolver.set_collapse_per_root(options_.get_int("DL_COLLAPSE_PER_ROOT"));
                fcisolver.set_subspace_per_root(options_.get_int("DL_SUBSPACE_PER_ROOT"));
                fcisolver.set_integral_pointer(fci_ints);

                // compute energy and fill in results
                fcisolver.compute_energy();
                SharedVector Ems = fcisolver.eigen_vals();
                for (int i = 0; i < nstates; ++i) {
                    Edsrg_sa[n].push_back(Ems->get(i) + Enuc_);
                }

            } else {

                outfile->Printf("\n    Use the sub-space of CASCI.");

                int dim = (eigens_[n][0].first)->dim();
                size_t eigen_size = eigens_[n].size();
                SharedMatrix evecs(new Matrix("evecs", dim, eigen_size));
                for (int i = 0; i < eigen_size; ++i) {
                    evecs->set_column(0, i, (eigens_[n][i]).first);
                }

                SharedMatrix Heff(
                    new Matrix("Heff " + multi_label[multi - 1] + " " + irrep_symbol[irrep],
                               nstates, nstates));
                for (int A = 0; A < nstates; ++A) {
                    for (int B = A; B < nstates; ++B) {

                        // compute rdms
                        CI_RDMS ci_rdms(fci_ints, p_space, evecs, A, B);

                        // since Hbar is rotated to the original basis
                        // there is no need to rotate RDMs
                        BlockedTensor D1 =
                            BTF_->build(tensor_type_, "D1", spin_cases({"aa"}), true);
                        ci_rdms.compute_1rdm(D1.block("aa").data(), D1.block("AA").data());

                        BlockedTensor D2 =
                            BTF_->build(tensor_type_, "D2", spin_cases({"aaaa"}), true);
                        ci_rdms.compute_2rdm(D2.block("aaaa").data(), D2.block("aAaA").data(), D2.block("AAAA").data());

                        double H_AB = 0.0;
                        H_AB += oei["uv"] * D1["uv"];
                        H_AB += oei["UV"] * D1["UV"];

                        H_AB += 0.25 * Hbar2_["uvxy"] * D2["xyuv"];
                        H_AB += 0.25 * Hbar2_["UVXY"] * D2["XYUV"];
                        H_AB += Hbar2_["uVxY"] * D2["xYuV"];

                        if (options_.get_bool("FORM_HBAR3")) {
                            BlockedTensor D3 =
                                BTF_->build(tensor_type_, "D3", spin_cases({"aaaaaa"}), true);
                            ci_rdms.compute_3rdm(D3.block("aaaaaa").data(), D3.block("aaAaaA").data(),
                                                 D3.block("aAAaAA").data(), D3.block("AAAAAA").data());

                            H_AB += (1.0 / 36) * Hbar3_["xyzuvw"] * D3["uvwxyz"];
                            H_AB += (1.0 / 36) * Hbar3_["XYZUVW"] * D3["UVWXYZ"];
                            H_AB += 0.25 * Hbar3_["xyZuvW"] * D3["uvWxyZ"];
                            H_AB += 0.25 * Hbar3_["xYZuVW"] * D3["uVWxYZ"];
                        }

                        if (A == B) {
                            H_AB += Efrzc_ + fci_ints->scalar_energy() + Enuc_;
                            Heff->set(A, B, H_AB);
                        } else {
                            Heff->set(A, B, H_AB);
                            Heff->set(B, A, H_AB);
                        }
                    }
                } // end forming effective Hamiltonian

                print_h2("Effective Hamiltonian Summary");
                outfile->Printf("\n");
                Heff->print();
                SharedMatrix U(new Matrix("U of Heff", nstates, nstates));
                SharedVector Ems(new Vector("MS Energies", nstates));
                Heff->diagonalize(U, Ems);
                U->eivprint(Ems);

                // fill in Edsrg_sa
                for (int i = 0; i < nstates; ++i) {
                    Edsrg_sa[n].push_back(Ems->get(i));
                }
            } // end if DSRG_AVG_DIAG

        } // end looping averaged states
    }

    return Edsrg_sa;
}

std::vector<std::vector<double>> DSRG_MRPT2::compute_energy_xms() {
    // get character table
    CharacterTable ct = Process::environment.molecule()->point_group()->char_table();
    std::vector<std::string> irrep_symbol;
    for (int h = 0; h < this->nirrep(); ++h) {
        irrep_symbol.push_back(std::string(ct.gamma(h).symbol()));
    }

    // multiplicity table
    std::vector<std::string> multi_label{
        "Singlet", "Doublet", "Triplet", "Quartet", "Quintet", "Sextet", "Septet", "Octet",
        "Nonet",   "Decaet",  "11-et",   "12-et",   "13-et",   "14-et",  "15-et",  "16-et",
        "17-et",   "18-et",   "19-et",   "20-et",   "21-et",   "22-et",  "23-et",  "24-et"};

    // prepare FCI integrals (a fake one)
    std::shared_ptr<FCIIntegrals> fci_ints =
        std::make_shared<FCIIntegrals>(ints_, actv_mos_, core_mos_);
    //    ambit::Tensor actv_aa = ints_->aptei_aa_block(aactv_mos_, aactv_mos_,
    //    aactv_mos_, aactv_mos_);
    //    ambit::Tensor actv_ab = ints_->aptei_ab_block(aactv_mos_, aactv_mos_,
    //    aactv_mos_, aactv_mos_);
    //    ambit::Tensor actv_bb = ints_->aptei_bb_block(aactv_mos_, aactv_mos_,
    //    aactv_mos_, aactv_mos_);
    //    fci_ints->set_active_integrals(actv_aa, actv_ab, actv_bb);
    //    fci_ints->compute_restricted_one_body_operator();

    // allocate space for one-electron integrals
    Hoei_ = BTF_->build(tensor_type_, "OEI", spin_cases({"ph", "cc"}));

    // obtain zeroth-order states
    int nentry = eigens_.size();
    std::vector<std::vector<double>> Edsrg_ms(nentry, std::vector<double>());

    for (int n = 0; n < nentry; ++n) {
        int irrep = options_["AVG_STATE"][n][0].to_integer();
        int multi = options_["AVG_STATE"][n][1].to_integer();
        int nstates = eigens_[n].size();
        std::vector<psi::forte::Determinant> p_space = p_spaces_[n];

        // print current status
        std::stringstream ss;
        ss << multi_label[multi - 1] << " " << irrep_symbol[irrep];
        print_h2("Build Effective Hamiltonian (" + ss.str() + ")");
        outfile->Printf("\n");

        // fill in ci vectors
        int dim = (eigens_[n][0].first)->dim();
        SharedMatrix civecs(new Matrix("ci vecs", dim, nstates));
        for (int i = 0; i < nstates; ++i) {
            civecs->set_column(0, i, (eigens_[n][i]).first);
        }

        // XMS rotaion if needed
        if (options_.get_str("DSRG_MULTI_STATE") == "XMS") {
            if (nentry > 1) {
                // recompute state-averaged density
                outfile->Printf("\n    Recompute SA density matrix of %s with equal weights.",
                                ss.str().c_str());
                size_t na = actv_mos_.size();
                size_t na2 = na * na;
                std::vector<double> sa_opdm_a(na2, 0.0);
                std::vector<double> sa_opdm_b(na2, 0.0);

                for (int M = 0; M < nstates; ++M) {
                    CI_RDMS ci_rdms(fci_ints, p_space, civecs, M, M);
                    std::vector<double> opdm_a(na2, 0.0);
                    std::vector<double> opdm_b(na2, 0.0);
                    ci_rdms.compute_1rdm(opdm_a, opdm_b);

                    std::transform(sa_opdm_a.begin(), sa_opdm_a.end(), opdm_a.begin(),
                                   sa_opdm_a.begin(), std::plus<double>());
                    std::transform(sa_opdm_b.begin(), sa_opdm_b.end(), opdm_b.begin(),
                                   sa_opdm_b.begin(), std::plus<double>());
                }
                std::for_each(sa_opdm_a.begin(), sa_opdm_a.end(), [&](double& v) { v /= nstates; });
                std::for_each(sa_opdm_b.begin(), sa_opdm_b.end(), [&](double& v) { v /= nstates; });

                rotate_1rdm(sa_opdm_a, sa_opdm_b);
                Gamma1_.block("aa").data() = std::move(sa_opdm_a);
                Gamma1_.block("AA").data() = std::move(sa_opdm_b);

                // rebuild Fock matrix
                build_fock();
            }

            // XMS rotation
            civecs = xms_rotation(fci_ints, p_space, civecs, irrep);
        }

        // prepare Heff
        SharedMatrix Heff(new Matrix("Heff " + multi_label[multi - 1] + " " + irrep_symbol[irrep],
                                     nstates, nstates));
        SharedMatrix Heff_sym(
            new Matrix("Heff (Symmetrized) " + multi_label[multi - 1] + " " + irrep_symbol[irrep],
                       nstates, nstates));

        // loop over states
        for (int M = 0; M < nstates; ++M) {

            print_h2("Compute DSRG-MRPT2 Energy of State " + std::to_string(M));

            // compute the densities
            compute_cumulants(fci_ints, p_space, civecs, M, M, irrep);

            // compute Fock
            build_fock();

            // reset one-electron integral
            build_oei();

            // recompute reference energy
            Eref_ = compute_reference_energy(Hoei_, F_, V_);

            // compute DSRG-MRPT2 energy
            double Ept2 = compute_energy();

            // set diagonal elements of Heff
            Heff->set(M, M, Ept2);
            Heff_sym->set(M, M, Ept2);

            // reset two-electron integrals because it is renormalized
            build_ints();

            // set Hoei to effective oei for coupling computations
            build_eff_oei();

            // build effective singles resulting from de-normal-ordering
            build_T1eff_deGNO();

            // compute couplings between states
            print_h2("Compute Couplings with State " + std::to_string(M));
            for (int N = 0; N < nstates; ++N) {
                if (N == M) {
                    continue;
                } else {
                    // compute transition densities
                    compute_densities(fci_ints, p_space, civecs, M, N, irrep);

                    // compute coupling of <N|H|M>
                    std::stringstream ss;
                    if (N > M) {
                        ss << "<" << N << "|H|" << M << ">";
                        double c1 = compute_ms_1st_coupling(ss.str());
                        Heff->add(M, N, c1);
                        Heff->add(N, M, c1);
                        Heff_sym->add(M, N, c1);
                        Heff_sym->add(N, M, c1);
                    }

                    // compute coupling of <N|HT|M>
                    ss.str(std::string());
                    ss.clear();
                    ss << "<" << N << "|HT|" << M << ">";
                    double c2 = compute_ms_2nd_coupling(ss.str());
                    Heff->add(N, M, c2);
                    Heff_sym->add(N, M, 0.5 * c2);
                    Heff_sym->add(M, N, 0.5 * c2);
                }
            }
        }

        print_h2("Effective Hamiltonian Summary");
        outfile->Printf("\n");

        Heff->print();
        Heff_sym->print();

        SharedMatrix U(new Matrix("U of Heff (Symmetrized)", nstates, nstates));
        SharedVector Ems(new Vector("MS Energies", nstates));
        Heff_sym->diagonalize(U, Ems);
        U->eivprint(Ems);

        // fill in Edsrg_ms
        for (int i = 0; i < nstates; ++i) {
            Edsrg_ms[n].push_back(Ems->get(i));
        }
    }

    return Edsrg_ms;
}

void DSRG_MRPT2::build_oei() {
    Hoei_.iterate(
        [&](const std::vector<size_t>& i, const std::vector<SpinType>& spin, double& value) {
            if (spin[0] == AlphaSpin) {
                value = ints_->oei_a(i[0], i[1]);
            } else {
                value = ints_->oei_b(i[0], i[1]);
            }
        });
}

void DSRG_MRPT2::build_eff_oei() {
    for (const auto& block : Hoei_.block_labels()) {
        // lowercase: alpha spin
        if (islower(block[0])) {
            Hoei_.block(block).iterate([&](const std::vector<size_t>& i, double& value) {
                size_t np = label_to_spacemo_[block[0]][i[0]];
                size_t nq = label_to_spacemo_[block[1]][i[1]];
                value = ints_->oei_a(np, nq);

                for (const size_t& nm : core_mos_) {
                    value += ints_->aptei_aa(np, nm, nq, nm);
                    value += ints_->aptei_ab(np, nm, nq, nm);
                }
            });
        } else {
            Hoei_.block(block).iterate([&](const std::vector<size_t>& i, double& value) {
                size_t np = label_to_spacemo_[block[0]][i[0]];
                size_t nq = label_to_spacemo_[block[1]][i[1]];
                value = ints_->oei_b(np, nq);

                for (const size_t& nm : core_mos_) {
                    value += ints_->aptei_bb(np, nm, nq, nm);
                    value += ints_->aptei_ab(nm, np, nm, nq);
                }
            });
        }
    }
}

SharedMatrix DSRG_MRPT2::xms_rotation(std::shared_ptr<FCIIntegrals> fci_ints,
                                      std::vector<psi::forte::Determinant>& p_space,
                                      SharedMatrix civecs, const int& irrep) {
    print_h2("Perform XMS Rotation to Reference States");
    outfile->Printf("\n");

    // build Fock matrix
    int nstates = civecs->ncol();
    SharedMatrix Fock(new Matrix("Fock", nstates, nstates));

    for (int M = 0; M < nstates; ++M) {
        for (int N = M; N < nstates; ++N) {

            // compute transition density
            CI_RDMS ci_rdms(fci_ints, p_space, civecs, M, N);

            std::vector<double> opdm_a, opdm_b;
            ci_rdms.compute_1rdm(opdm_a, opdm_b);
            rotate_1rdm(opdm_a, opdm_b);

            // put rdms in tensor format
            BlockedTensor D1 = BTF_->build(tensor_type_, "D1", spin_cases({"aa"}), true);
            D1.block("aa").data() = std::move(opdm_a);
            D1.block("AA").data() = std::move(opdm_b);

            // compute Fock elements
            double F_MN = 0.0;
            F_MN += D1["uv"] * F_["vu"];
            F_MN += D1["UV"] * F_["VU"];
            Fock->set(M, N, F_MN);
            if (M != N) {
                Fock->set(N, M, F_MN);
            }
        }
    }
    Fock->print();

    // diagonalize Fock
    SharedMatrix Fevec(new Matrix("Fock Evec", nstates, nstates));
    SharedVector Feval(new Vector("Fock Eval", nstates));
    Fock->diagonalize(Fevec, Feval);
    //    Fevec->print();
    Fevec->eivprint(Feval);

    // Rotate ci vecs
    SharedMatrix rcivecs(civecs->clone());
    rcivecs->zero();
    rcivecs->gemm(false, false, 1.0, civecs, Fevec, 0.0);

    return rcivecs;
}

double DSRG_MRPT2::Tamp_deGNO() {
    // de-normal-order T1
    build_T1eff_deGNO();

    double out = 0.0;
    if (internal_amp_) {
        // the scalar term of amplitudes when de-normal-ordering
        out -= T1_["uv"] * Gamma1_["vu"];
        out -= T1_["UV"] * Gamma1_["VU"];

        out -= 0.25 * T2_["xyuv"] * Lambda2_["uvxy"];
        out -= 0.25 * T2_["XYUV"] * Lambda2_["UVXY"];
        out -= T2_["xYuV"] * Lambda2_["uVxY"];

        out += 0.5 * T2_["xyuv"] * Gamma1_["ux"] * Gamma1_["vy"];
        out += 0.5 * T2_["XYUV"] * Gamma1_["UX"] * Gamma1_["VY"];
        out += T2_["xYuV"] * Gamma1_["ux"] * Gamma1_["VY"];
    }

    return out;
}

void DSRG_MRPT2::build_T1eff_deGNO() {
    T1eff_ = BTF_->build(tensor_type_, "Effective T1 from de-GNO", spin_cases({"hp"}));

    T1eff_["ia"] = T1_["ia"];
    T1eff_["IA"] = T1_["IA"];

    T1eff_["ia"] -= T2_["iuav"] * Gamma1_["vu"];
    T1eff_["ia"] -= T2_["iUaV"] * Gamma1_["VU"];
    T1eff_["IA"] -= T2_["uIvA"] * Gamma1_["vu"];
    T1eff_["IA"] -= T2_["IUAV"] * Gamma1_["VU"];
}

double DSRG_MRPT2::compute_ms_1st_coupling(const std::string& name) {
    Timer timer;
    std::string str = "Computing coupling of " + name;
    outfile->Printf("\n    %-40s ...", str.c_str());

    double coupling = 0.0;
    coupling += Hoei_["uv"] * Gamma1_["vu"];
    coupling += Hoei_["UV"] * Gamma1_["VU"];

    coupling += 0.25 * V_["xyuv"] * Lambda2_["uvxy"];
    coupling += 0.25 * V_["XYUV"] * Lambda2_["UVXY"];
    coupling += V_["xYuV"] * Lambda2_["uVxY"];

    outfile->Printf("  Done. Timing %15.6f s", timer.get());
    return coupling;
}

double DSRG_MRPT2::compute_ms_2nd_coupling(const std::string& name) {
    Timer timer;
    std::string str = "Computing coupling of " + name;
    outfile->Printf("\n    %-40s ...", str.c_str());

    double coupling = 0.0;

    // H1 contract with D1
    BlockedTensor H1 = BTF_->build(tensor_type_, "Heff1_2nd", spin_cases({"aa"}));
    H1["vu"] += Hoei_["eu"] * T1eff_["ve"];
    H1["VU"] += Hoei_["EU"] * T1eff_["VE"];

    H1["vu"] -= Hoei_["vm"] * T1eff_["mu"];
    H1["VU"] -= Hoei_["VM"] * T1eff_["MU"];

    H1["vu"] += V_["avmu"] * T1eff_["ma"];
    H1["vu"] += V_["vAuM"] * T1eff_["MA"];
    H1["VU"] += V_["aVmU"] * T1eff_["ma"];
    H1["VU"] += V_["AVMU"] * T1eff_["MA"];

    H1["vu"] += Hoei_["am"] * T2_["vmua"];
    H1["vu"] += Hoei_["AM"] * T2_["vMuA"];
    H1["VU"] += Hoei_["am"] * T2_["mVaU"];
    H1["VU"] += Hoei_["AM"] * T2_["VMUA"];

    H1["vu"] += 0.5 * V_["abum"] * T2_["vmab"];
    H1["vu"] += V_["aBuM"] * T2_["vMaB"];
    H1["VU"] += V_["aBmU"] * T2_["mVaB"];
    H1["VU"] += 0.5 * V_["ABUM"] * T2_["VMAB"];

    H1["vu"] -= 0.5 * V_["avmn"] * T2_["mnau"];
    H1["vu"] -= V_["vAmN"] * T2_["mNuA"];
    H1["VU"] -= V_["aVmN"] * T2_["mNaU"];
    H1["VU"] -= 0.5 * V_["AVMN"] * T2_["MNAU"];

    coupling += H1["vu"] * Gamma1_["uv"];
    coupling += H1["VU"] * Gamma1_["UV"];

    // H2 contract with D2
    BlockedTensor H2 = BTF_->build(tensor_type_, "Heff2_2nd", spin_cases({"aaaa"}));
    BlockedTensor temp = BTF_->build(tensor_type_, "temp", {"aaaa", "AAAA"}, true);
    temp["xyuv"] = V_["eyuv"] * T1eff_["xe"];
    temp["XYUV"] = V_["EYUV"] * T1eff_["XE"];

    H2["xyuv"] += temp["xyuv"];
    H2["XYUV"] += temp["XYUV"];
    H2["xyuv"] -= temp["yxuv"];
    H2["XYUV"] -= temp["YXUV"];

    H2["xYuV"] += V_["eYuV"] * T1eff_["xe"];
    H2["xYuV"] += V_["xEuV"] * T1eff_["YE"];

    temp["xyuv"] = V_["xymv"] * T1eff_["mu"];
    temp["XYUV"] = V_["XYMV"] * T1eff_["MU"];

    H2["xyuv"] -= temp["xyuv"];
    H2["XYUV"] -= temp["XYUV"];
    H2["xyuv"] += temp["xyvu"];
    H2["XYUV"] += temp["XYVU"];

    H2["xYuV"] -= V_["xYmV"] * T1eff_["mu"];
    H2["xYuV"] -= V_["xYuM"] * T1eff_["MV"];

    temp["xyuv"] = Hoei_["eu"] * T2_["xyev"];
    temp["XYUV"] = Hoei_["EU"] * T2_["XYEV"];

    H2["xyuv"] += temp["xyuv"];
    H2["XYUV"] += temp["XYUV"];
    H2["xyuv"] -= temp["xyvu"];
    H2["XYUV"] -= temp["XYVU"];

    H2["xYuV"] += Hoei_["eu"] * T2_["xYeV"];
    H2["xYuV"] += Hoei_["EV"] * T2_["xYuE"];

    temp["xyuv"] = Hoei_["xm"] * T2_["myuv"];
    temp["XYUV"] = Hoei_["XM"] * T2_["MYUV"];

    H2["xyuv"] -= temp["xyuv"];
    H2["XYUV"] -= temp["XYUV"];
    H2["xyuv"] += temp["yxuv"];
    H2["XYUV"] += temp["YXUV"];

    H2["xYuV"] -= Hoei_["xm"] * T2_["mYuV"];
    H2["xYuV"] -= Hoei_["YM"] * T2_["xMuV"];

    H2["xyuv"] += 0.5 * V_["abuv"] * T2_["xyab"];
    H2["xYuV"] += V_["aBuV"] * T2_["xYaB"];
    H2["XYUV"] += 0.5 * V_["ABUV"] * T2_["XYAB"];

    H2["xyuv"] -= 0.5 * V_["xyij"] * T2_["ijuv"];
    H2["xYuV"] -= V_["xYiJ"] * T2_["iJuV"];
    H2["XYUV"] -= 0.5 * V_["XYIJ"] * T2_["IJUV"];

    H2["xyuv"] += V_["xyim"] * T2_["imuv"];
    H2["xYuV"] += V_["xYiM"] * T2_["iMuV"];
    H2["xYuV"] += V_["xYmI"] * T2_["mIuV"];
    H2["XYUV"] += V_["XYIM"] * T2_["IMUV"];

    temp["xyuv"] = V_["ayum"] * T2_["xmav"];
    temp["xyuv"] += V_["yAuM"] * T2_["xMvA"];
    temp["XYUV"] = V_["aYmU"] * T2_["mXaV"];
    temp["XYUV"] += V_["AYUM"] * T2_["XMAV"];

    H2["xyuv"] -= temp["xyuv"];
    H2["XYUV"] -= temp["XYUV"];
    H2["xyuv"] += temp["yxuv"];
    H2["XYUV"] += temp["YXUV"];
    H2["xyuv"] += temp["xyvu"];
    H2["XYUV"] += temp["XYVU"];
    H2["xyuv"] -= temp["yxvu"];
    H2["XYUV"] -= temp["YXVU"];

    H2["xYuV"] -= V_["aYuM"] * T2_["xMaV"];
    H2["xYuV"] += V_["xaum"] * T2_["mYaV"];
    H2["xYuV"] += V_["xAuM"] * T2_["MYAV"];
    H2["xYuV"] += V_["aYmV"] * T2_["xmua"];
    H2["xYuV"] += V_["AYMV"] * T2_["xMuA"];
    H2["xYuV"] -= V_["xAmV"] * T2_["mYuA"];

    coupling += 0.25 * H2["xyuv"] * Lambda2_["uvxy"];
    coupling += H2["xYuV"] * Lambda2_["uVxY"];
    coupling += 0.25 * H2["XYUV"] * Lambda2_["UVXY"];

    // H3 contract with D3
    BlockedTensor H3 = BTF_->build(tensor_type_, "Heff3_2nd", spin_cases({"aaaaaa"}));
    H2_T2_C3(V_, T2_, 1.0, H3, true);

    coupling += 1.0 / 36.0 * H3.block("aaaaaa")("uvwxyz") * reference_.L3aaa()("xyzuvw");
    coupling += 1.0 / 36.0 * H3.block("AAAAAA")("UVWXYZ") * reference_.L3bbb()("XYZUVW");
    coupling += 0.25 * H3.block("aaAaaA")("uvWxyZ") * reference_.L3aab()("xyZuvW");
    coupling += 0.25 * H3.block("aAAaAA")("uVWxYZ") * reference_.L3abb()("xYZuVW");

    outfile->Printf("  Done. Timing %15.6f s", timer.get());
    return coupling;
}

void DSRG_MRPT2::compute_Heff_2nd_coupling(double& H0, ambit::Tensor& H1a, ambit::Tensor& H1b,
                                           ambit::Tensor& H2aa, ambit::Tensor& H2ab,
                                           ambit::Tensor& H2bb, ambit::Tensor& H3aaa,
                                           ambit::Tensor& H3aab, ambit::Tensor& H3abb,
                                           ambit::Tensor& H3bbb) {
    // reset Tensors
    BlockedTensor H1 = BTF_->build(tensor_type_, "Heff1_2nd", spin_cases({"aa"}));
    H1a = H1.block("aa");
    H1b = H1.block("AA");

    BlockedTensor H2 = BTF_->build(tensor_type_, "Heff2_2nd", spin_cases({"aaaa"}));
    H2aa = H2.block("aaaa");
    H2ab = H2.block("aAaA");
    H2bb = H2.block("AAAA");

    BlockedTensor H3 = BTF_->build(tensor_type_, "Heff3_2nd", spin_cases({"aaaaaa"}));
    H3aaa = H3.block("aaaaaa");
    H3aab = H3.block("aaAaaA");
    H3abb = H3.block("aAAaAA");
    H3bbb = H3.block("AAAAAA");

    // de-normal-order amplitudes
    BlockedTensor T1eff = deGNO_Tamp(T1_, T2_, Gamma1_);

    // reset APTEI because it is renormalized
    build_ints();

    // "effective" one-electron integrals: hbar^p_q = h^p_q + sum_m v^{mp}_{mq}
    Hoei_ = BTF_->build(tensor_type_, "OEI", spin_cases({"ph"}));
    build_eff_oei();

    // set H1 and H2 to bare Hamiltonian
    H1["uv"] = Hoei_["uv"];
    H1["UV"] = Hoei_["UV"];

    H2["uvxy"] = V_["uvxy"];
    H2["uVxY"] = V_["uVxY"];
    H2["UVXY"] = V_["UVXY"];

    // scalar term from bare Hamiltonian
    H0 = 0.0;
    size_t ncore = core_mos_.size();
    for (int m = 0; m < ncore; ++m) {
        size_t nm = core_mos_[m];
        H0 += ints_->oei_a(nm, nm);
        H0 += ints_->oei_b(nm, nm);

        for (int n = 0; n < ncore; ++n) {
            size_t nn = core_mos_[n];
            H0 += 0.5 * ints_->aptei_aa(nm, nn, nm, nn);
            H0 += 0.5 * ints_->aptei_bb(nm, nn, nm, nn);
            H0 += ints_->aptei_ab(nm, nn, nm, nn);
        }
    }

    // scalar from [H, T]
    H0 += Hoei_["am"] * T1eff["ma"];
    H0 += Hoei_["AM"] * T1eff["MA"];

    H0 += 0.25 * V_["abmn"] * T2_["mnab"];
    H0 += V_["aBmN"] * T2_["mNaB"];
    H0 += 0.25 * V_["ABMN"] * T2_["MNAB"];

    // 1-body
    H1["vu"] += Hoei_["eu"] * T1eff["ve"];
    H1["VU"] += Hoei_["EU"] * T1eff["VE"];

    H1["vu"] -= Hoei_["vm"] * T1eff["mu"];
    H1["VU"] -= Hoei_["VM"] * T1eff["MU"];

    H1["vu"] += V_["avmu"] * T1eff["ma"];
    H1["vu"] += V_["vAuM"] * T1eff["MA"];
    H1["VU"] += V_["aVmU"] * T1eff["ma"];
    H1["VU"] += V_["AVMU"] * T1eff["MA"];

    H1["vu"] += Hoei_["am"] * T2_["vmua"];
    H1["vu"] += Hoei_["AM"] * T2_["vMuA"];
    H1["VU"] += Hoei_["am"] * T2_["mVaU"];
    H1["VU"] += Hoei_["AM"] * T2_["VMUA"];

    H1["vu"] += 0.5 * V_["abum"] * T2_["vmab"];
    H1["vu"] += V_["aBuM"] * T2_["vMaB"];
    H1["VU"] += V_["aBmU"] * T2_["mVaB"];
    H1["VU"] += 0.5 * V_["ABUM"] * T2_["VMAB"];

    H1["vu"] -= 0.5 * V_["avmn"] * T2_["mnau"];
    H1["vu"] -= V_["vAmN"] * T2_["mNuA"];
    H1["VU"] -= V_["aVmN"] * T2_["mNaU"];
    H1["VU"] -= 0.5 * V_["AVMN"] * T2_["MNAU"];

    // 2-body
    BlockedTensor temp = BTF_->build(tensor_type_, "temp", {"aaaa", "AAAA"});

    temp["xyuv"] = V_["eyuv"] * T1eff["xe"];
    temp["XYUV"] = V_["EYUV"] * T1eff["XE"];

    H2["xyuv"] += temp["xyuv"];
    H2["XYUV"] += temp["XYUV"];
    H2["xyuv"] -= temp["yxuv"];
    H2["XYUV"] -= temp["YXUV"];

    H2["xYuV"] += V_["eYuV"] * T1eff["xe"];
    H2["xYuV"] += V_["xEuV"] * T1eff["YE"];

    temp["xyuv"] = V_["xymv"] * T1eff["mu"];
    temp["XYUV"] = V_["XYMV"] * T1eff["MU"];

    H2["xyuv"] -= temp["xyuv"];
    H2["XYUV"] -= temp["XYUV"];
    H2["xyuv"] += temp["xyvu"];
    H2["XYUV"] += temp["XYVU"];

    H2["xYuV"] -= V_["xYmV"] * T1eff["mu"];
    H2["xYuV"] -= V_["xYuM"] * T1eff["MV"];

    temp["xyuv"] = Hoei_["eu"] * T2_["xyev"];
    temp["XYUV"] = Hoei_["EU"] * T2_["XYEV"];

    H2["xyuv"] += temp["xyuv"];
    H2["XYUV"] += temp["XYUV"];
    H2["xyuv"] -= temp["xyvu"];
    H2["XYUV"] -= temp["XYVU"];

    H2["xYuV"] += Hoei_["eu"] * T2_["xYeV"];
    H2["xYuV"] += Hoei_["EV"] * T2_["xYuE"];

    temp["xyuv"] = Hoei_["xm"] * T2_["myuv"];
    temp["XYUV"] = Hoei_["XM"] * T2_["MYUV"];

    H2["xyuv"] -= temp["xyuv"];
    H2["XYUV"] -= temp["XYUV"];
    H2["xyuv"] += temp["yxuv"];
    H2["XYUV"] += temp["YXUV"];

    H2["xYuV"] -= Hoei_["xm"] * T2_["mYuV"];
    H2["xYuV"] -= Hoei_["YM"] * T2_["xMuV"];

    H2["xyuv"] += 0.5 * V_["abuv"] * T2_["xyab"];
    H2["xYuV"] += V_["aBuV"] * T2_["xYaB"];
    H2["XYUV"] += 0.5 * V_["ABUV"] * T2_["XYAB"];

    H2["xyuv"] -= 0.5 * V_["xyij"] * T2_["ijuv"];
    H2["xYuV"] -= V_["xYiJ"] * T2_["iJuV"];
    H2["XYUV"] -= 0.5 * V_["XYIJ"] * T2_["IJUV"];

    H2["xyuv"] += V_["xyim"] * T2_["imuv"];
    H2["xYuV"] += V_["xYiM"] * T2_["iMuV"];
    H2["xYuV"] += V_["xYmI"] * T2_["mIuV"];
    H2["XYUV"] += V_["XYIM"] * T2_["IMUV"];

    temp["xyuv"] = V_["ayum"] * T2_["xmav"];
    temp["xyuv"] += V_["yAuM"] * T2_["xMvA"];
    temp["XYUV"] = V_["aYmU"] * T2_["mXaV"];
    temp["XYUV"] += V_["AYUM"] * T2_["XMAV"];

    H2["xyuv"] -= temp["xyuv"];
    H2["XYUV"] -= temp["XYUV"];
    H2["xyuv"] += temp["yxuv"];
    H2["XYUV"] += temp["YXUV"];
    H2["xyuv"] += temp["xyvu"];
    H2["XYUV"] += temp["XYVU"];
    H2["xyuv"] -= temp["yxvu"];
    H2["XYUV"] -= temp["YXVU"];

    H2["xYuV"] -= V_["aYuM"] * T2_["xMaV"];
    H2["xYuV"] += V_["xaum"] * T2_["mYaV"];
    H2["xYuV"] += V_["xAuM"] * T2_["MYAV"];
    H2["xYuV"] += V_["aYmV"] * T2_["xmua"];
    H2["xYuV"] += V_["AYMV"] * T2_["xMuA"];
    H2["xYuV"] -= V_["xAmV"] * T2_["mYuA"];

    // 3-body
    H2_T2_C3(V_, T2_, 1.0, H3, true);
}

void DSRG_MRPT2::compute_cumulants(std::shared_ptr<FCIIntegrals> fci_ints,
                                   std::vector<Determinant>& p_space, SharedMatrix evecs,
                                   const int& root1, const int& root2, const int& irrep) {
    CI_RDMS ci_rdms(fci_ints, p_space, evecs, root1, root2);

    // 1 cumulant
    std::vector<double> opdm_a, opdm_b;
    ci_rdms.compute_1rdm(opdm_a, opdm_b);
    rotate_1rdm(opdm_a, opdm_b);

    ambit::Tensor L1a = Gamma1_.block("aa");
    ambit::Tensor L1b = Gamma1_.block("AA");

    L1a.data() = std::move(opdm_a);
    L1b.data() = std::move(opdm_b);

    (Eta1_.block("aa")).iterate([&](const std::vector<size_t>& i, double& value) {
        value = i[0] == i[1] ? 1.0 : 0.0;
    });
    (Eta1_.block("AA")).iterate([&](const std::vector<size_t>& i, double& value) {
        value = i[0] == i[1] ? 1.0 : 0.0;
    });
    Eta1_.block("aa")("pq") -= Gamma1_.block("aa")("pq");
    Eta1_.block("AA")("pq") -= Gamma1_.block("AA")("pq");

    // 2 cumulant
    std::vector<double> tpdm_aa, tpdm_ab, tpdm_bb;
    ci_rdms.compute_2rdm(tpdm_aa, tpdm_ab, tpdm_bb);
    rotate_2rdm(tpdm_aa, tpdm_ab, tpdm_bb);

    ambit::Tensor L2aa = Lambda2_.block("aaaa");
    ambit::Tensor L2ab = Lambda2_.block("aAaA");
    ambit::Tensor L2bb = Lambda2_.block("AAAA");

    L2aa.data() = std::move(tpdm_aa);
    L2ab.data() = std::move(tpdm_ab);
    L2bb.data() = std::move(tpdm_bb);

    L2aa("pqrs") -= L1a("pr") * L1a("qs");
    L2aa("pqrs") += L1a("ps") * L1a("qr");

    L2bb("pqrs") -= L1b("pr") * L1b("qs");
    L2bb("pqrs") += L1b("ps") * L1b("qr");

    L2ab("pqrs") -= L1a("pr") * L1b("qs");

    // 3 cumulant
    if (options_.get_str("THREEPDC") != "ZERO") {
        std::vector<double> tpdm_aaa, tpdm_aab, tpdm_abb, tpdm_bbb;
        ci_rdms.compute_3rdm(tpdm_aaa, tpdm_aab, tpdm_abb, tpdm_bbb);
        rotate_3rdm(tpdm_aaa, tpdm_aab, tpdm_abb, tpdm_bbb);

        ambit::Tensor L3aaa = reference_.L3aaa();
        ambit::Tensor L3aab = reference_.L3aab();
        ambit::Tensor L3abb = reference_.L3abb();
        ambit::Tensor L3bbb = reference_.L3bbb();
        //        ambit::Tensor L3aaa = Lambda3_.block("aaaaaa");
        //        ambit::Tensor L3aab = Lambda3_.block("aaAaaA");
        //        ambit::Tensor L3abb = Lambda3_.block("aAAaAA");
        //        ambit::Tensor L3bbb = Lambda3_.block("AAAAAA");

        L3aaa.data() = std::move(tpdm_aaa);
        L3aab.data() = std::move(tpdm_aab);
        L3abb.data() = std::move(tpdm_abb);
        L3bbb.data() = std::move(tpdm_bbb);

        // - step 1: aaa
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

        // - step 2: aab
        L3aab("pqRstU") -= L1a("ps") * L2ab("qRtU");
        L3aab("pqRstU") += L1a("pt") * L2ab("qRsU");

        L3aab("pqRstU") -= L1a("qt") * L2ab("pRsU");
        L3aab("pqRstU") += L1a("qs") * L2ab("pRtU");

        L3aab("pqRstU") -= L1b("RU") * L2aa("pqst");

        L3aab("pqRstU") -= L1a("ps") * L1a("qt") * L1b("RU");
        L3aab("pqRstU") += L1a("pt") * L1a("qs") * L1b("RU");

        // - step 3: abb
        L3abb("pQRsTU") -= L1a("ps") * L2bb("QRTU");

        L3abb("pQRsTU") -= L1b("QT") * L2ab("pRsU");
        L3abb("pQRsTU") += L1b("QU") * L2ab("pRsT");

        L3abb("pQRsTU") -= L1b("RU") * L2ab("pQsT");
        L3abb("pQRsTU") += L1b("RT") * L2ab("pQsU");

        L3abb("pQRsTU") -= L1a("ps") * L1b("QT") * L1b("RU");
        L3abb("pQRsTU") += L1a("ps") * L1b("QU") * L1b("RT");

        // - step 4: bbb
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
    }
}

void DSRG_MRPT2::compute_densities(std::shared_ptr<FCIIntegrals> fci_ints,
                                   std::vector<psi::forte::Determinant>& p_space,
                                   SharedMatrix evecs, const int& root1, const int& root2,
                                   const int& irrep) {
    CI_RDMS ci_rdms(fci_ints, p_space, evecs, root1, root2);

    // 1 density
    std::vector<double> opdm_a, opdm_b;
    ci_rdms.compute_1rdm(opdm_a, opdm_b);
    rotate_1rdm(opdm_a, opdm_b);

    Gamma1_.block("aa").data() = std::move(opdm_a);
    Gamma1_.block("AA").data() = std::move(opdm_b);

    //    (Eta1_.block("aa")).iterate([&](const std::vector<size_t>& i,double&
    //    value){
    //        value = i[0] == i[1] ? 1.0 : 0.0;
    //    });
    //    (Eta1_.block("AA")).iterate([&](const std::vector<size_t>& i,double&
    //    value){
    //        value = i[0] == i[1] ? 1.0 : 0.0;
    //    });
    //    Eta1_.block("aa")("pq") -= Gamma1_.block("aa")("pq");
    //    Eta1_.block("AA")("pq") -= Gamma1_.block("AA")("pq");

    // 2 density
    std::vector<double> tpdm_aa, tpdm_ab, tpdm_bb;
    ci_rdms.compute_2rdm(tpdm_aa, tpdm_ab, tpdm_bb);
    rotate_2rdm(tpdm_aa, tpdm_ab, tpdm_bb);

    Lambda2_.block("aaaa").data() = std::move(tpdm_aa);
    Lambda2_.block("aAaA").data() = std::move(tpdm_ab);
    Lambda2_.block("AAAA").data() = std::move(tpdm_bb);

    // 3 density
    std::vector<double> tpdm_aaa, tpdm_aab, tpdm_abb, tpdm_bbb;
    ci_rdms.compute_3rdm(tpdm_aaa, tpdm_aab, tpdm_abb, tpdm_bbb);
    rotate_3rdm(tpdm_aaa, tpdm_aab, tpdm_abb, tpdm_bbb);

    ambit::Tensor L3aaa = reference_.L3aaa();
    ambit::Tensor L3aab = reference_.L3aab();
    ambit::Tensor L3abb = reference_.L3abb();
    ambit::Tensor L3bbb = reference_.L3bbb();
    //        ambit::Tensor L3aaa = Lambda3_.block("aaaaaa");
    //        ambit::Tensor L3aab = Lambda3_.block("aaAaaA");
    //        ambit::Tensor L3abb = Lambda3_.block("aAAaAA");
    //        ambit::Tensor L3bbb = Lambda3_.block("AAAAAA");

    L3aaa.data() = std::move(tpdm_aaa);
    L3aab.data() = std::move(tpdm_aab);
    L3abb.data() = std::move(tpdm_abb);
    L3bbb.data() = std::move(tpdm_bbb);
}

void DSRG_MRPT2::rotate_1rdm(std::vector<double>& opdm_a, std::vector<double>& opdm_b) {
    size_t na = actv_mos_.size();
    if ((na * na != opdm_a.size()) || (na * na != opdm_b.size())) {
        throw PSIEXCEPTION("Cannot rotate 1RDM in DSRG_MRPT2.");
    }

    ambit::Tensor Ua = Uactv_.block("aa");
    ambit::Tensor Ub = Uactv_.block("AA");

    ambit::Tensor temp = ambit::Tensor::build(tensor_type_, "temp", {na, na});
    ambit::Tensor D = ambit::Tensor::build(tensor_type_, "D1", {na, na});

    temp.data() = std::move(opdm_a);
    D("pq") = Ua("ap") * temp("ab") * Ua("bq");
    opdm_a = D.data();

    temp.data() = std::move(opdm_b);
    D("PQ") = Ub("AP") * temp("AB") * Ub("BQ");
    opdm_b = D.data();
}

void DSRG_MRPT2::rotate_2rdm(std::vector<double>& tpdm_aa, std::vector<double>& tpdm_ab,
                             std::vector<double>& tpdm_bb) {
    size_t na = actv_mos_.size();
    size_t na4 = na * na * na * na;
    if ((na4 != tpdm_aa.size()) || (na4 != tpdm_ab.size()) || (na4 != tpdm_bb.size())) {
        throw PSIEXCEPTION("Cannot rotate 2RDM in DSRG_MRPT2.");
    }

    ambit::Tensor Ua = Uactv_.block("aa");
    ambit::Tensor Ub = Uactv_.block("AA");

    ambit::Tensor temp = ambit::Tensor::build(tensor_type_, "temp", {na, na, na, na});
    ambit::Tensor D = ambit::Tensor::build(tensor_type_, "D2", {na, na, na, na});

    temp.data() = std::move(tpdm_aa);
    D("pqrs") = Ua("ap") * Ua("bq") * temp("abcd") * Ua("cr") * Ua("ds");
    tpdm_aa = D.data();

    temp.data() = std::move(tpdm_ab);
    D("pQrS") = Ua("ap") * Ub("BQ") * temp("aBcD") * Ua("cr") * Ub("DS");
    tpdm_ab = D.data();

    temp.data() = std::move(tpdm_bb);
    D("PQRS") = Ub("AP") * Ub("BQ") * temp("ABCD") * Ub("CR") * Ub("DS");
    tpdm_bb = D.data();
}

void DSRG_MRPT2::rotate_3rdm(std::vector<double>& tpdm_aaa, std::vector<double>& tpdm_aab,
                             std::vector<double>& tpdm_abb, std::vector<double>& tpdm_bbb) {
    size_t na = actv_mos_.size();
    size_t na6 = na * na * na * na * na * na;
    if ((na6 != tpdm_aaa.size()) || (na6 != tpdm_aab.size()) || (na6 != tpdm_abb.size()) ||
        (na6 != tpdm_bbb.size())) {
        throw PSIEXCEPTION("Cannot rotate 3RDM in DSRG_MRPT2.");
    }

    ambit::Tensor Ua = Uactv_.block("aa");
    ambit::Tensor Ub = Uactv_.block("AA");

    ambit::Tensor temp = ambit::Tensor::build(tensor_type_, "temp", {na, na, na, na, na, na});
    ambit::Tensor D = ambit::Tensor::build(tensor_type_, "D3", {na, na, na, na, na, na});

    temp.data() = std::move(tpdm_aaa);
    D("pqrstu") = Ua("ap") * Ua("bq") * Ua("cr") * temp("abcijk") * Ua("is") * Ua("jt") * Ua("ku");
    tpdm_aaa = D.data();

    temp.data() = std::move(tpdm_aab);
    D("pqRstU") = Ua("ap") * Ua("bq") * Ub("CR") * temp("abCijK") * Ua("is") * Ua("jt") * Ub("KU");
    tpdm_aab = D.data();

    temp.data() = std::move(tpdm_abb);
    D("pQRsTU") = Ua("ap") * Ub("BQ") * Ub("CR") * temp("aBCiJK") * Ua("is") * Ub("JT") * Ub("KU");
    tpdm_abb = D.data();

    temp.data() = std::move(tpdm_bbb);
    D("PQRSTU") = Ub("AP") * Ub("BQ") * Ub("CR") * temp("ABCIJK") * Ub("IS") * Ub("JT") * Ub("KU");
    tpdm_bbb = D.data();
}
}
}
