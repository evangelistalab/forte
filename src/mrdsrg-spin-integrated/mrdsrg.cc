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

#include <algorithm>
#include <cmath>
#include <map>
#include <vector>

#include "psi4/libmints/molecule.h"

#include "base_classes/active_space_solver.h"
#include "fci/fci_solver.h"
#include "sci/fci_mo.h"
#include "helpers/printing.h"
#include "orbital-helpers/semi_canonicalize.h"
#include "boost/format.hpp"
#include "orbital-helpers/mp2_nos.h"
#include "mrdsrg.h"

using namespace psi;

namespace forte {

MRDSRG::MRDSRG(RDMs rdms, std::shared_ptr<SCFInfo> scf_info,
               std::shared_ptr<ForteOptions> options, std::shared_ptr<ForteIntegrals> ints,
               std::shared_ptr<MOSpaceInfo> mo_space_info)
    : MASTER_DSRG(rdms, scf_info, options, ints, mo_space_info) {

    print_method_banner({"Multireference Driven Similarity Renormalization Group", "Chenyang Li"});
    outfile->Printf("\n  Additional contributions by: Tianyuan Zhang");

    read_options();
    startup();
    print_options();
}

MRDSRG::~MRDSRG() { cleanup(); }

void MRDSRG::cleanup() {}

void MRDSRG::read_options() {
    dsrg_trans_type_ = foptions_->get_str("DSRG_TRANS_TYPE");
    if (dsrg_trans_type_ != "UNITARY") {
        std::stringstream ss;
        ss << "DSRG transformation type (" << dsrg_trans_type_
           << ") is not implemented yet. Please change to UNITARY";
        throw psi::PSIEXCEPTION(ss.str());
    }

    corrlv_string_ = foptions_->get_str("CORR_LEVEL");
    std::vector<std::string> available{"PT2", "PT3", "LDSRG2", "LDSRG2_QC", "LSRG2", "SRG_PT2"};
    if (std::find(available.begin(), available.end(), corrlv_string_) == available.end()) {
        outfile->Printf("\n  Warning: CORR_LEVEL option %s is not implemented.",
                        corrlv_string_.c_str());
        outfile->Printf("\n  Changed CORR_LEVEL option to PT2");
        corrlv_string_ = "PT2";

        warnings_.push_back(std::make_tuple("Unsupported CORR_LEVEL", "Change to PT2",
                                            "Change options in input.dat"));
    }

    sequential_Hbar_ = foptions_->get_bool("DSRG_HBAR_SEQ");
    nivo_ = foptions_->get_bool("DSRG_NIVO");
}

void MRDSRG::startup() {
    // prepare integrals
    H_ = BTF_->build(tensor_type_, "H", spin_cases({"gg"}));

    // if density fitted
    if (eri_df_) {
        B_ = BTF_->build(tensor_type_, "B 3-idx", {"Lgg", "LGG"});
    } else {
        V_ = BTF_->build(tensor_type_, "V", spin_cases({"gggg"}));
    }
    build_ints();

    // print norm and max of 2- and 3-cumulants
    print_cumulant_summary();

    // copy Fock matrix from master_dsrg
    F_ = BTF_->build(tensor_type_, "Fock", spin_cases({"gg"}));
    F_["pq"] = Fock_["pq"];
    F_["PQ"] = Fock_["PQ"];
    Fa_ = Fdiag_a_;
    Fb_ = Fdiag_b_;

    // auto adjusted s_
    s_ = make_s_smart();

    // test semi-canonical
    semi_canonical_ = check_semi_orbs();

    if (!semi_canonical_) {
        outfile->Printf("\n    Orbital invariant formalism will be employed for MR-DSRG.");
        U_ = ambit::BlockedTensor::build(tensor_type_, "U", spin_cases({"gg"}));
        std::vector<std::vector<double>> eigens = diagonalize_Fock_diagblocks(U_);
        Fa_ = eigens[0];
        Fb_ = eigens[1];
    }
}

void MRDSRG::print_options() {
    // fill in information
    std::vector<std::pair<std::string, int>> calculation_info{
        {"ntamp", ntamp_},
        {"diis_min_vecs", foptions_->get_int("DIIS_MIN_VECS")},
        {"diis_max_vecs", foptions_->get_int("DIIS_MAX_VECS")}};

    std::vector<std::pair<std::string, double>> calculation_info_double{
        {"flow parameter", s_},
        {"taylor expansion threshold", pow(10.0, -double(taylor_threshold_))},
        {"intruder_tamp", intruder_tamp_}};

    std::vector<std::pair<std::string, std::string>> calculation_info_string{
        {"corr_level", corrlv_string_},
        {"int_type", ints_type_},
        {"source operator", source_},
        {"smart_dsrg_s", foptions_->get_str("SMART_DSRG_S")},
        {"reference relaxation", relax_ref_},
        {"dsrg transformation type", dsrg_trans_type_},
        {"core virtual source type", foptions_->get_str("CCVV_SOURCE")}};

    auto true_false_string = [](bool x) {
        if (x) {
            return std::string("TRUE");
        } else {
            return std::string("FALSE");
        }
    };
    calculation_info_string.push_back(
        {"sequential dsrg transformation", true_false_string(sequential_Hbar_)});
    calculation_info_string.push_back(
        {"omit blocks of >= 3 virtual indices", true_false_string(nivo_)});

    // print some information
    print_h2("Calculation Information");
    for (auto& str_dim : calculation_info) {
        outfile->Printf("\n    %-35s %15d", str_dim.first.c_str(), str_dim.second);
    }
    for (auto& str_dim : calculation_info_double) {
        outfile->Printf("\n    %-35s %15.3e", str_dim.first.c_str(), str_dim.second);
    }
    for (auto& str_dim : calculation_info_string) {
        outfile->Printf("\n    %-35s %15s", str_dim.first.c_str(), str_dim.second.c_str());
    }
    outfile->Printf("\n");
}

void MRDSRG::build_ints() {
    // prepare one-electron integrals
    H_.iterate([&](const std::vector<size_t>& i, const std::vector<SpinType>& spin, double& value) {
        if (spin[0] == AlphaSpin)
            value = ints_->oei_a(i[0], i[1]);
        else
            value = ints_->oei_b(i[0], i[1]);
    });

    // prepare two-electron integrals or three-index B
    if (eri_df_) {
        fill_three_index_ints(B_);

        //        B_.iterate([&](const std::vector<size_t>& i, const std::vector<SpinType>&, double&
        //        value) {
        ////            value = ints_->three_integral(i[0], i[1], i[2]);
        //        });
    } else {
        V_.iterate(
            [&](const std::vector<size_t>& i, const std::vector<SpinType>& spin, double& value) {
                if ((spin[0] == AlphaSpin) and (spin[1] == AlphaSpin))
                    value = ints_->aptei_aa(i[0], i[1], i[2], i[3]);
                if ((spin[0] == AlphaSpin) and (spin[1] == BetaSpin))
                    value = ints_->aptei_ab(i[0], i[1], i[2], i[3]);
                if ((spin[0] == BetaSpin) and (spin[1] == BetaSpin))
                    value = ints_->aptei_bb(i[0], i[1], i[2], i[3]);
            });
    }
}

void MRDSRG::build_density() {
    // directly call function of MASTER_DSRG
    fill_density();

    // check cumulants
    print_cumulant_summary();
}

void MRDSRG::build_fock(BlockedTensor& H, BlockedTensor& V) {
    // the core-core density is an identity matrix
    BlockedTensor D1c = BTF_->build(tensor_type_, "Gamma1 core", spin_cases({"cc"}));
    for (size_t m = 0, nc = core_mos_.size(); m < nc; ++m) {
        D1c.block("cc").data()[m * nc + m] = 1.0;
        D1c.block("CC").data()[m * nc + m] = 1.0;
    }

    // build Fock matrix
    F_["pq"] = H["pq"];
    F_["pq"] += V["pnqm"] * D1c["mn"];
    F_["pq"] += V["pNqM"] * D1c["MN"];
    F_["pq"] += V["pvqu"] * Gamma1_["uv"];
    F_["pq"] += V["pVqU"] * Gamma1_["UV"];

    F_["PQ"] = H["PQ"];
    F_["PQ"] += V["nPmQ"] * D1c["mn"];
    F_["PQ"] += V["PNQM"] * D1c["MN"];
    F_["PQ"] += V["vPuQ"] * Gamma1_["uv"];
    F_["PQ"] += V["PVQU"] * Gamma1_["UV"];

    // obtain diagonal elements of Fock matrix
    size_t ncmo_ = mo_space_info_->size("CORRELATED");
    Fa_ = std::vector<double>(ncmo_);
    Fb_ = std::vector<double>(ncmo_);
    F_.iterate([&](const std::vector<size_t>& i, const std::vector<SpinType>& spin, double& value) {
        if (spin[0] == AlphaSpin and (i[0] == i[1])) {
            Fa_[i[0]] = value;
        }
        if (spin[0] == BetaSpin and (i[0] == i[1])) {
            Fb_[i[0]] = value;
        }
    });

    // set F_ to Fock_ in master_dsrg because check_semi_orbs use Fock_
    Fock_["pq"] = F_["pq"];
    Fock_["PQ"] = F_["PQ"];
}

void MRDSRG::build_fock_df(BlockedTensor& H, BlockedTensor& B) {
    // the core-core density is an identity matrix
    BlockedTensor D1c = BTF_->build(tensor_type_, "Gamma1 core", spin_cases({"cc"}));
    for (size_t m = 0, nc = core_mos_.size(); m < nc; ++m) {
        D1c.block("cc").data()[m * nc + m] = 1.0;
        D1c.block("CC").data()[m * nc + m] = 1.0;
    }

    // build Fock matrix
    F_["pq"] = H["pq"];
    F_["PQ"] = H["PQ"];

    BlockedTensor temp = BTF_->build(tensor_type_, "B temp", {"L"});
    temp["g"] = B["gmn"] * D1c["mn"];
    temp["g"] += B["guv"] * Gamma1_["uv"];
    F_["pq"] += temp["g"] * B["gpq"];
    F_["PQ"] += temp["g"] * B["gPQ"];

    temp["g"] = B["gMN"] * D1c["MN"];
    temp["g"] += B["gUV"] * Gamma1_["UV"];
    F_["pq"] += temp["g"] * B["gpq"];
    F_["PQ"] += temp["g"] * B["gPQ"];

    // exchange
    F_["pq"] -= B["gpn"] * B["gmq"] * D1c["mn"];
    F_["pq"] -= B["gpv"] * B["guq"] * Gamma1_["uv"];

    F_["PQ"] -= B["gPN"] * B["gMQ"] * D1c["MN"];
    F_["PQ"] -= B["gPV"] * B["gUQ"] * Gamma1_["UV"];

    // obtain diagonal elements of Fock matrix
    size_t ncmo_ = mo_space_info_->size("CORRELATED");
    Fa_ = std::vector<double>(ncmo_);
    Fb_ = std::vector<double>(ncmo_);
    F_.iterate([&](const std::vector<size_t>& i, const std::vector<SpinType>& spin, double& value) {
        if (spin[0] == AlphaSpin and (i[0] == i[1])) {
            Fa_[i[0]] = value;
        }
        if (spin[0] == BetaSpin and (i[0] == i[1])) {
            Fb_[i[0]] = value;
        }
    });

    // set F_ to Fock_ in master_dsrg because check_semi_orbs use Fock_
    Fock_["pq"] = F_["pq"];
    Fock_["PQ"] = F_["PQ"];
}

double MRDSRG::compute_energy() {
    // guess amplitudes when necessary
    bool initialize_T = true;
    if (corrlv_string_ == "LSRG2" || corrlv_string_ == "SRG_PT2") {
        initialize_T = false;
    }

    if (initialize_T) {
        // build initial amplitudes
        print_h2("Build Initial Amplitude from DSRG-MRPT2");
        T1_ = BTF_->build(tensor_type_, "T1 Amplitudes", spin_cases({"hp"}));
        T2_ = BTF_->build(tensor_type_, "T2 Amplitudes", spin_cases({"hhpp"}));
        if (eri_df_) {
            guess_t_df(B_, T2_, F_, T1_);
        } else {
            guess_t(V_, T2_, F_, T1_);
        }

        // check initial amplitudes
        analyze_amplitudes("First-Order", T1_, T2_);
    }

    // get reference energy
    double Etotal = Eref_;

    // compute energy
    switch (corrlevelmap[corrlv_string_]) {
    case CORR_LV::LDSRG2: {
        Etotal += compute_energy_ldsrg2();
        break;
    }
    case CORR_LV::LDSRG2_QC: {
        Etotal += compute_energy_ldsrg2_qc();
        break;
    }
    case CORR_LV::LDSRG2_P3: {
        break;
    }
    case CORR_LV::QDSRG2: {
        break;
    }
    case CORR_LV::QDSRG2_P3: {
        break;
    }
    case CORR_LV::LSRG2: {
        Etotal += compute_energy_lsrg2();
        break;
    }
    case CORR_LV::SRG_PT2: {
        Etotal += compute_energy_srgpt2();
        break;
    }
    case CORR_LV::PT3: {
        Etotal += compute_energy_pt3();
        break;
    }
    default: {
        Etotal += compute_energy_pt2();
    }
    }

    return Etotal;
}
/*
double MRDSRG::compute_energy_relaxed() {
    // reference relaxation
    double Edsrg = 0.0, Erelax = 0.0;
    std::string cas_type = foptions_->get_str("CAS_TYPE");

    size_t nroot = foptions_->get_int("NROOT");

    if (relax_ref_ == "ONCE") {
        // compute energy with fixed ref.
        Edsrg = compute_energy();

        // compute de-normal-ordered all-active DSRG transformed Hamiltonian
        auto fci_ints = compute_Heff_actv();

        if (cas_type == "CAS") {
            FCI_MO fci_mo(scf_info_, foptions_, ints_, mo_space_info_, fci_ints);
            fci_mo.set_localize_actv(false);
            Erelax = fci_mo.compute_energy();
        } else if (cas_type == "ACI") {
            auto state = make_state_info_from_psi_wfn(ints_->wfn());
            AdaptiveCI aci(state, nroot, scf_info_, foptions_, mo_space_info_,
                           fci_ints); // ints_->wfn() is implicitly converted to StateInfo
            aci.set_fci_ints(fci_ints);
            if (foptions_->has_changed("ACI_RELAX_SIGMA")) {
                aci.update_sigma();
            }
            Erelax = aci.compute_energy();

        } else {
            auto state = make_state_info_from_psi_wfn(ints_->wfn());
            auto fci = make_active_space_method("FCI", state, nroot, scf_info_, mo_space_info_,
                                                ints_, foptions_);
            fci->set_max_rdm_level(1);
            fci->set_active_space_integrals(fci_ints);
            Erelax = fci->compute_energy();
        }

        // printing
        print_h2("MRDSRG Energy Summary");
        outfile->Printf("\n    %-30s = %22.15f", "MRDSRG Total Energy (fixed)", Edsrg);
        outfile->Printf("\n    %-30s = %22.15f", "MRDSRG Total Energy (relaxed)", Erelax);
        outfile->Printf("\n");

        psi::Process::environment.globals["UNRELAXED ENERGY"] = Edsrg;
        psi::Process::environment.globals["PARTIALLY RELAXED ENERGY"] = Erelax;

    } else if (relax_ref_ == "ITERATE" || relax_ref_ == "TWICE") {

        int max_rdm_level = foptions_->get_str("THREEPDC") == "ZERO" ? 2 : 3;
        SemiCanonical semiorb(foptions_, ints_, mo_space_info_, true);

        // iteration variables
        int cycle = 0, maxiter = foptions_->get_int("MAXITER_RELAX_REF");
        double e_conv = foptions_->get_double("RELAX_E_CONVERGENCE");
        std::vector<double> Edsrg_vec, Erelax_vec;
        std::vector<double> Edelta_dsrg_vec, Edelta_relax_vec;
        bool converged = false, failed = false;

        // start iteration
        do {
            std::string relax_title = "MR-DSRG RDMs Relaxation Iter. " + std::to_string(cycle);
            print_h2(relax_title);

            // compute dsrg energy
            double Etemp = Edsrg;
            Edsrg = compute_energy();
            Edsrg_vec.push_back(Edsrg);
            double Edelta_dsrg = Edsrg - Etemp;
            Edelta_dsrg_vec.push_back(Edelta_dsrg);

            // compute de-normal-ordered all-active DSRG transformed Hamiltonian
            auto as_ints = compute_Heff_actv();

            /// NOTE: For consistant CI coefficients, compute_Heff will rotate Hbar to the basis
            /// before semicanonicalization!
            /// This means we need to transform the new reference to the old semicanonical basis.
            ///
            /// ints_ -- semicanonical basis
            /// fci_ints -- original basis

            // diagonalize the Hamiltonian using fci_ints
            Etemp = Erelax;
            // auto fci_ints = make_active_space_ints(mo_space_info_, ints_, "ACTIVE",
            // {{"RESTRICTED_DOCC"}});
            auto state_weights_list = make_state_weights_list(foptions_, ints_->wfn());
            auto ci = make_active_space_solver(cas_type, state_weights_list, scf_info_,
                                               mo_space_info_, as_ints, foptions_);
            ci->set_max_rdm_level(3);
            const auto& state_energies_list = ci->compute_energy();
            double average_energy =
                compute_average_state_energy(state_energies_list, state_weights_list);
            Erelax = average_energy;
            reference_ = ci->reference();

            outfile->Printf("\n  The following reference rotation will make the new reference and "
                            "integrals in the same basis.");
            ambit::Tensor Ua = Uactv_.block("aa"), Ub = Uactv_.block("AA");
            semiorb.transform_reference(Ua, Ub, reference_, max_rdm_level);

            Erelax_vec.push_back(Erelax);
            double Edelta_relax = Erelax - Etemp;
            Edelta_relax_vec.push_back(Edelta_relax);

            // semicanonicalize orbitals
            if (foptions_->get_bool("SEMI_CANONICAL")) {
                print_h2("Semicanonicalize Orbitals");

                // use semicanonicalize class
                semiorb.semicanonicalize(reference_);
                Uactv_.block("aa")("pq") = semiorb.Ua_t()("pq");
                Uactv_.block("AA")("pq") = semiorb.Ub_t()("pq");

                // refill H_, V_, and B_ from ForteIntegrals
                build_ints();

                // refill densities
                build_density();

                // build Fock matrix
                if (!eri_df_) {
                    build_fock(H_, V_);
                } else {
                    build_fock_df(H_, B_);
                }
            } else {
                // refill density
                build_density();

                // build Fock matrix
                if (!eri_df_) {
                    build_fock(H_, V_);
                } else {
                    build_fock_df(H_, B_);
                }

                // check semi-canonicalization
                semi_canonical_ = check_semi_orbs();

                if (!semi_canonical_) {
                    outfile->Printf(
                        "\n    Orbital invariant formalism will be employed for MR-DSRG.");
                    U_ = ambit::BlockedTensor::build(tensor_type_, "U", spin_cases({"gg"}));
                    std::vector<std::vector<double>> eigens = diagonalize_Fock_diagblocks(U_);
                    Fa_ = eigens[0];
                    Fb_ = eigens[1];
                }
            }

            // recompute reference energy (rebuild MK vacuum)
            if (eri_df_) {
                Eref_ = compute_reference_energy_df(H_, F_, B_);
            } else {
                Eref_ = compute_reference_energy(H_, F_, V_);
            }

            outfile->Printf("\n\n  Updated reference energy E0 = %23.15f", Eref_);

            // test convergence
            if (std::fabs(Edelta_dsrg) < e_conv && std::fabs(Edelta_relax) < e_conv) {
                converged = true;
                psi::Process::environment.globals["FULLY RELAXED ENERGY"] = Erelax;
            }
            if (cycle > maxiter) {
                outfile->Printf("\n\n    The reference relaxation does not "
                                "converge in %d iterations! Quitting.\n",
                                maxiter);
                converged = true;
                failed = true;
            }
            ++cycle;

            // terminate peacefully if relaxed twice
            if (relax_ref_ == "TWICE" && cycle == 2) {
                converged = true;
                failed = false;
            }
        } while (!converged);

        print_h2("MRDSRG RDMs Relaxation Summary");
        std::string indent(4, ' ');
        std::string dash(71, '-');
        std::string title;
        title += indent + str(boost::format("%5c  %=31s  %=31s\n") % ' ' % "Fixed Ref. (a.u.)" %
                              "Relaxed Ref. (a.u.)");
        title += indent + std::string(7, ' ') + std::string(31, '-') + "  " + std::string(31, '-') +
                 "\n";
        title += indent + str(boost::format("%5s  %=20s %=10s  %=20s %=10s\n") % "Iter." %
                              "Total Energy" % "Delta" % "Total Energy" % "Delta");
        title += indent + dash;
        outfile->Printf("\n%s", title.c_str());
        for (int n = 0; n != cycle; ++n) {
            outfile->Printf("\n    %5d  %20.12f %10.3e  %20.12f %10.3e", n, Edsrg_vec[n],
                            Edelta_dsrg_vec[n], Erelax_vec[n], Edelta_relax_vec[n]);
        }
        outfile->Printf("\n    %s", dash.c_str());
        outfile->Printf("\n    %-30s = %23.15f", "MRDSRG Total Energy", Edsrg);
        outfile->Printf("\n    %-30s = %23.15f", "MRDSRG Total Energy (relaxed)", Erelax);
        outfile->Printf("\n");

        if (failed) {
            throw psi::PSIEXCEPTION("RDMs relaxation process does not converge.");
        }

        // set energies to psi4 environment
        psi::Process::environment.globals["UNRELAXED ENERGY"] = Edsrg_vec[0];
        psi::Process::environment.globals["PARTIALLY RELAXED ENERGY"] = Erelax_vec[0];
        if (cycle > 1) {
            psi::Process::environment.globals["RELAXED ENERGY"] = Edsrg_vec[1];
        }
    }

    psi::Process::environment.globals["CURRENT ENERGY"] = Erelax;
    return Erelax;
}
*/

[[deprecated]] double MRDSRG::compute_energy_sa() {
//    int nentry = eigens_.size();
//    std::vector<std::vector<std::vector<double>>> Edsrg_vec;
//    SemiCanonical semiorb(mo_space_info_, ints_, foptions_, true);
//    int max_rdm_level = foptions_->get_str("THREEPDC") == "ZERO" ? 2 : 3;

//    // iteration variables
//    double Edsrg_sa = 0.0, Erelax_sa = 0.0;
//    int cycle = 0, maxiter = foptions_->get_int("MAXITER_RELAX_REF");
//    double e_conv = foptions_->get_double("RELAX_E_CONVERGENCE");
//    std::vector<double> Edsrg_sa_vec, Erelax_sa_vec;
//    std::vector<double> Edelta_dsrg_sa_vec, Edelta_relax_sa_vec;
//    bool converged = false, failed = false;

//    // start iteration
//    do {
//        // print
//        outfile->Printf("\n  ==> SA-MR-DSRG CI Iter. %d <==", cycle);

//        // compute dsrg energy
//        double Etemp = Edsrg_sa;
//        Edsrg_sa = compute_energy();
//        Edsrg_sa_vec.push_back(Edsrg_sa);
//        double Edelta_dsrg = Edsrg_sa - Etemp;
//        Edelta_dsrg_sa_vec.push_back(Edelta_dsrg);

//        // compute de-normal-ordered all-active DSRG transformed Hamiltonian
//        auto fci_ints = compute_Heff_actv();

//        // diagonalize the Hamiltonian
//        auto fci_mo =
//            std::make_shared<FCI_MO>(scf_info_, foptions_, ints_, mo_space_info_, fci_ints);
//        Etemp = Erelax_sa;
//        fci_mo->set_localize_actv(false);
//        fci_mo->set_max_rdm_level(max_rdm_level);
//        Erelax_sa = fci_mo->compute_energy();
//        Erelax_sa_vec.push_back(Erelax_sa);
//        double Edelta_relax = Erelax_sa - Etemp;
//        Edelta_relax_sa_vec.push_back(Edelta_relax);

//        // copy energy
//        std::vector<std::vector<double>> Evec(nentry, std::vector<double>());
//        for (int n = 0; n < nentry; ++n) {
//            int nstates = (foptions_->psi_options())["AVG_STATE"][n][2].to_integer();

//            for (int i = 0; i < nstates; ++i) {
//                Evec[n].push_back(fci_mo->eigens()[n][i].second);
//            }

//            Edsrg_vec.push_back(Evec);
//        }

//        // obtain new reference
//        std::vector<std::pair<size_t, size_t>> roots; // unused for SA
//        rdms_ = fci_mo->reference(roots)[0];

//        outfile->Printf("\n  The following reference rotation will make the new reference and "
//                        "integrals in the same basis.");
//        ambit::Tensor Ua = Uactv_.block("aa"), Ub = Uactv_.block("AA");
//        semiorb.transform_rdms(Ua, Ub, rdms_, max_rdm_level);

//        // semicanonicalize orbitals
//        if (foptions_->get_bool("SEMI_CANONICAL")) {
//            print_h2("Semicanonicalize Orbitals");

//            // use semicanonicalize class
//            semiorb.semicanonicalize(rdms_);
//            Uactv_.block("aa")("pq") = semiorb.Ua_t()("pq");
//            Uactv_.block("AA")("pq") = semiorb.Ub_t()("pq");

//            // refill H_, V_, and B_ from ForteIntegrals
//            build_ints();

//            // refill densities
//            build_density();

//            // build Fock matrix
//            if (!eri_df_) {
//                build_fock(H_, V_);
//            } else {
//                build_fock_df(H_, B_);
//            }
//        } else {
//            // refill density
//            build_density();

//            // build Fock matrix
//            if (!eri_df_) {
//                build_fock(H_, V_);
//            } else {
//                build_fock_df(H_, B_);
//            }

//            // check semi-canonicalization
//            semi_canonical_ = check_semi_orbs();

//            if (!semi_canonical_) {
//                outfile->Printf("\n    Orbital invariant formalism will be employed for MR-DSRG.");
//                U_ = ambit::BlockedTensor::build(tensor_type_, "U", spin_cases({"gg"}));
//                std::vector<std::vector<double>> eigens = diagonalize_Fock_diagblocks(U_);
//                Fa_ = eigens[0];
//                Fb_ = eigens[1];
//            }
//        }

//        // recompute reference energy (rebuild MK vacuum)
//        if (eri_df_) {
//            Eref_ = compute_reference_energy_df(H_, F_, B_);
//        } else {
//            Eref_ = compute_reference_energy(H_, F_, V_);
//        }

//        // test convergence
//        if (std::fabs(Edelta_dsrg) < e_conv && std::fabs(Edelta_relax) < e_conv) {
//            converged = true;
//        }
//        if (cycle > maxiter) {
//            outfile->Printf("\n\n    The reference relaxation does not "
//                            "converge in %d iterations! Quitting.\n",
//                            maxiter);
//            converged = true;
//            failed = true;
//        }
//        ++cycle;
//    } while (!converged);

//    print_h2("State-Average MR-DSRG Energy Summary");

//    outfile->Printf("\n    state-averaged energy summary (not useful)");
//    std::string indent(4, ' ');
//    std::string dash(71, '-');
//    std::string title;
//    title += indent + str(boost::format("%5c  %=31s  %=31s\n") % ' ' % "Fixed Ref. (a.u.)" %
//                          "Relaxed Ref. (a.u.)");
//    title +=
//        indent + std::string(7, ' ') + std::string(31, '-') + "  " + std::string(31, '-') + "\n";
//    title += indent + str(boost::format("%5s  %=20s %=10s  %=20s %=10s\n") % "Iter." %
//                          "Total Energy" % "Delta" % "Total Energy" % "Delta");
//    title += indent + dash;
//    outfile->Printf("\n%s", title.c_str());
//    for (int n = 0; n != cycle; ++n) {
//        outfile->Printf("\n    %5d  %20.12f %10.3e  %20.12f %10.3e", n, Edsrg_sa_vec[n],
//                        Edelta_dsrg_sa_vec[n], Erelax_sa_vec[n], Edelta_relax_sa_vec[n]);
//    }
//    outfile->Printf("\n    %s", dash.c_str());

//    // get character table
//    CharacterTable ct = psi::Process::environment.molecule()->point_group()->char_table();
//    std::vector<std::string> irrep_symbol;
//    for (int h = 0, nirrep = mo_space_info_->nirrep(); h < nirrep; ++h) {
//        irrep_symbol.push_back(std::string(ct.gamma(h).symbol()));
//    }

//    // frist step results
//    print_h2("One-Step Results");
//    outfile->Printf("\n    Multi.  Irrep.  No.    MR-DSRG Energy");
//    std::string dash1(41, '-');
//    outfile->Printf("\n    %s", dash1.c_str());
//    for (int n = 0; n < nentry; ++n) {
//        int irrep = (foptions_->psi_options())["AVG_STATE"][n][0].to_integer();
//        int multi = (foptions_->psi_options())["AVG_STATE"][n][1].to_integer();
//        int nstates = (foptions_->psi_options())["AVG_STATE"][n][2].to_integer();

//        for (int i = 0; i < nstates; ++i) {
//            outfile->Printf("\n     %3d     %3s    %2d   %20.12f", multi,
//                            irrep_symbol[irrep].c_str(), i, Edsrg_vec[0][n][i]);
//        }
//        outfile->Printf("\n    %s", dash1.c_str());
//    }

//    // final step results
//    print_h2("Final-Step Results");
//    outfile->Printf("\n    Multi.  Irrep.  No.    MR-DSRG Energy");
//    outfile->Printf("\n    %s", dash1.c_str());
//    auto& Esa = Edsrg_vec[Edsrg_vec.size() - 1];
//    for (int n = 0, counter = 0; n < nentry; ++n) {
//        int irrep = (foptions_->psi_options())["AVG_STATE"][n][0].to_integer();
//        int multi = (foptions_->psi_options())["AVG_STATE"][n][1].to_integer();
//        int nstates = (foptions_->psi_options())["AVG_STATE"][n][2].to_integer();

//        for (int i = 0; i < nstates; ++i) {
//            outfile->Printf("\n     %3d     %3s    %2d   %20.12f*", multi,
//                            irrep_symbol[irrep].c_str(), i, Esa[n][i]);
//            psi::Process::environment.globals["ENERGY ROOT " + std::to_string(counter)] = Esa[n][i];
//            ++counter;
//        }
//        outfile->Printf("\n    %s", dash1.c_str());
//    }

//    if (failed) {
//        throw psi::PSIEXCEPTION("RDMs relaxation process does not converge.");
//    }

//    psi::Process::environment.globals["CURRENT ENERGY"] = Erelax_sa;
//    return Erelax_sa;
    return 0;
}

// void MRDSRG::transfer_integrals() {
//    // printing
//    print_h2("De-Normal-Order the DSRG Transformed Hamiltonian");

//    // compute scalar term (all active only)
//    local_timer t_scalar;
//    std::string str = "Computing the scalar term   ...";
//    outfile->Printf("\n    %-35s", str.c_str());
//    double scalar0 =
//        Eref_ + Hbar0_ -
//        molecule_->nuclear_repulsion_energy(reference_wavefunction_->get_dipole_field_strength())
//        -
//        ints_->frozen_core_energy();

//    // scalar from Hbar1
//    double scalar1 = 0.0;
//    scalar1 -= Hbar1_["vu"] * Gamma1_["uv"];
//    scalar1 -= Hbar1_["VU"] * Gamma1_["UV"];

//    // scalar from Hbar2
//    double scalar2 = 0.0;
//    scalar2 += 0.5 * Gamma1_["uv"] * Hbar2_["vyux"] * Gamma1_["xy"];
//    scalar2 += 0.5 * Gamma1_["UV"] * Hbar2_["VYUX"] * Gamma1_["XY"];
//    scalar2 += Gamma1_["uv"] * Hbar2_["vYuX"] * Gamma1_["XY"];

//    scalar2 -= 0.25 * Hbar2_["xyuv"] * Lambda2_["uvxy"];
//    scalar2 -= 0.25 * Hbar2_["XYUV"] * Lambda2_["UVXY"];
//    scalar2 -= Hbar2_["xYuV"] * Lambda2_["uVxY"];

//    double scalar = scalar0 + scalar1 + scalar2;
//    outfile->Printf("  Done. Timing %10.3f s", t_scalar.get());

//    // compute one-body term
//    local_timer t_one;
//    str = "Computing the one-body term ...";
//    outfile->Printf("\n    %-35s", str.c_str());
//    BlockedTensor temp1 = BTF_->build(tensor_type_, "temp1", spin_cases({"aa"}));
//    temp1["uv"] = Hbar1_["uv"];
//    temp1["UV"] = Hbar1_["UV"];
//    temp1["uv"] -= Hbar2_["uxvy"] * Gamma1_["yx"];
//    temp1["uv"] -= Hbar2_["uXvY"] * Gamma1_["YX"];
//    temp1["UV"] -= Hbar2_["xUyV"] * Gamma1_["yx"];
//    temp1["UV"] -= Hbar2_["UXVY"] * Gamma1_["YX"];
//    outfile->Printf("  Done. Timing %10.3f s", t_one.get());

//    //    // compute hole contributions using bare Hamiltonian
//    //    double h0 = 0.0;
//    //    BlockedTensor h1 =
//    //    BTF_->build(tensor_type_,"hole1",spin_cases({"cc"}));
//    //    h1["mn"] += V_["mvnu"] * Gamma1_["uv"];
//    //    h1["mn"] += V_["mVnU"] * Gamma1_["UV"];
//    //    h1["MN"] += V_["vMuN"] * Gamma1_["uv"];
//    //    h1["MN"] += V_["MVNU"] * Gamma1_["UV"];

//    //    // - 1) -sum_{m} H^{m}_{m} + \sum_{muv} H^{mv}_{mu} * L^{u}_{v}
//    //    for(const std::string block: {"cc", "CC"}){
//    //        F_.block(block).citerate([&](const std::vector<size_t>& i,const
//    //        double& value){
//    //            if(i[0] == i[1]){
//    //                h0 -= value;
//    //            }
//    //        });

//    //        h1.block(block).citerate([&](const std::vector<size_t>& i,const
//    //        double& value){
//    //            if(i[0] == i[1]){
//    //                h0 += value;
//    //            }
//    //        });
//    //    }

//    //    // - 2) 0.5 * sum_{mn} H^{mn}_{mn}
//    //    for(const std::string block: {"cccc", "CCCC"}){
//    //        V_.block(block).citerate([&](const std::vector<size_t>& i,const
//    //        double& value){
//    //            if(i[0] == i[2] && i[1] == i[3]){
//    //                h0 += 0.5 * value;
//    //            }
//    //        });
//    //    }

//    //    V_.block("cCcC").citerate([&](const std::vector<size_t>& i,const
//    //    double& value){
//    //        if(i[0] == i[2] && i[1] == i[3]){
//    //            h0 += value;
//    //        }
//    //    });

//    //    scalar += h0;

//    //    // - 3) -sum_{m} H^{qm}_{pm}
//    //    for(const std::string block: {"cc", "CC"}){
//    //        (h1.block(block)).iterate([&](const std::vector<size_t>& i,double&
//    //        value){
//    //            value = i[0] == i[1] ? 1.0 : 0.0;
//    //        });
//    //    }

//    //    temp1["uv"] -= V_["umvn"] * h1["nm"];
//    //    temp1["uv"] -= V_["uMvN"] * h1["NM"];
//    //    temp1["UV"] -= V_["mUnV"] * h1["nm"];
//    //    temp1["UV"] -= V_["UMVN"] * h1["NM"];

//    // update integrals
//    local_timer t_int;
//    str = "Updating integrals          ...";
//    outfile->Printf("\n    %-35s", str.c_str());
//    ints_->set_scalar(scalar);

//    //   a) zero hole integrals
//    std::vector<size_t> hole_mos = core_mos_;
//    hole_mos.insert(hole_mos.end(), actv_mos_.begin(), actv_mos_.end());
//    for (const size_t& i : hole_mos) {
//        for (const size_t& j : hole_mos) {
//            ints_->set_oei(i, j, 0.0, true);
//            ints_->set_oei(i, j, 0.0, false);
//            for (const size_t& k : hole_mos) {
//                for (const size_t& l : hole_mos) {
//                    ints_->set_tei(i, j, k, l, 0.0, true, true);
//                    ints_->set_tei(i, j, k, l, 0.0, true, false);
//                    ints_->set_tei(i, j, k, l, 0.0, false, false);
//                }
//            }
//        }
//    }

//    //   b) copy all active part
//    temp1.citerate(
//        [&](const std::vector<size_t>& i, const std::vector<SpinType>& spin, const double& value)
//        {
//            if (spin[0] == AlphaSpin) {
//                ints_->set_oei(i[0], i[1], value, true);
//            } else {
//                ints_->set_oei(i[0], i[1], value, false);
//            }
//        });
//    O1_["uv"] = temp1["uv"];
//    O1_["UV"] = temp1["UV"];

//    BlockedTensor temp2 = BTF_->build(tensor_type_, "temp2", spin_cases({"aaaa"}));
//    temp2["uvxy"] = Hbar2_["uvxy"];
//    temp2["uVxY"] = Hbar2_["uVxY"];
//    temp2["UVXY"] = Hbar2_["UVXY"];
//    temp2.citerate(
//        [&](const std::vector<size_t>& i, const std::vector<SpinType>& spin, const double& value)
//        {
//            if ((spin[0] == AlphaSpin) && (spin[1] == AlphaSpin)) {
//                ints_->set_tei(i[0], i[1], i[2], i[3], value, true, true);
//            } else if ((spin[0] == AlphaSpin) && (spin[1] == BetaSpin)) {
//                ints_->set_tei(i[0], i[1], i[2], i[3], value, true, false);
//            } else if ((spin[0] == BetaSpin) && (spin[1] == BetaSpin)) {
//                ints_->set_tei(i[0], i[1], i[2], i[3], value, false, false);
//            }
//        });
//    outfile->Printf("  Done. Timing %10.3f s", t_int.get());

//    // print scalar
//    double scalar_include_fc = scalar + ints_->frozen_core_energy();
//    print_h2("Scalar of the DSRG Hamiltonian (WRT True Vacuum)");
//    outfile->Printf("\n    %-30s = %22.15f", "Scalar0", scalar0);
//    outfile->Printf("\n    %-30s = %22.15f", "Scalar1", scalar1);
//    outfile->Printf("\n    %-30s = %22.15f", "Scalar2", scalar2);
//    outfile->Printf("\n    %-30s = %22.15f", "Total Scalar W/O Frozen-Core", scalar);
//    outfile->Printf("\n    %-30s = %22.15f", "Total Scalar W/  Frozen-Core", scalar_include_fc);

//    // test if de-normal-ordering is correct
//    print_h2("Test De-Normal-Ordered Hamiltonian");
//    double Etest =
//        scalar_include_fc +
//        molecule_->nuclear_repulsion_energy(reference_wavefunction_->get_dipole_field_strength());

//    double Etest1 = 0.0;
//    Etest1 += temp1["uv"] * Gamma1_["vu"];
//    Etest1 += temp1["UV"] * Gamma1_["VU"];

//    Etest1 += Hbar1_["uv"] * Gamma1_["vu"];
//    Etest1 += Hbar1_["UV"] * Gamma1_["VU"];
//    Etest1 *= 0.5;

//    //    for(const std::string block: {"cc","CC"}){
//    //        F_.block(block).citerate([&](const std::vector<size_t>& i,const
//    //        double& value){
//    //            if (i[0] == i[1]){
//    //                Etest1 += 0.5 * value;
//    //            }
//    //        });
//    //        H_.block(block).citerate([&](const std::vector<size_t>& i,const
//    //        double& value){
//    //            if (i[0] == i[1]){
//    //                Etest1 += 0.5 * value;
//    //            }
//    //        });
//    //    }

//    double Etest2 = 0.0;
//    Etest2 += 0.25 * Hbar2_["uvxy"] * Lambda2_["xyuv"];
//    Etest2 += 0.25 * Hbar2_["UVXY"] * Lambda2_["XYUV"];
//    Etest2 += Hbar2_["uVxY"] * Lambda2_["xYuV"];

//    Etest += Etest1 + Etest2;
//    outfile->Printf("\n    %-30s = %22.15f", "One-Body Energy (after)", Etest1);
//    outfile->Printf("\n    %-30s = %22.15f", "Two-Body Energy (after)", Etest2);
//    outfile->Printf("\n    %-30s = %22.15f", "Total Energy (after)", Etest);
//    outfile->Printf("\n    %-30s = %22.15f", "Total Energy (before)", Eref_ + Hbar0_);

//    if (std::fabs(Etest - Eref_ - Hbar0_) > 100.0 * foptions_->get_double("E_CONVERGENCE")) {
//        throw psi::PSIEXCEPTION("De-normal-odering failed.");
//    } else {
//    //    ints_->update_integrals(false); <- this should not be here
//    }
//}

void MRDSRG::reset_ints(BlockedTensor& H, BlockedTensor& V) {
    ints_->set_scalar(0.0);
    H.citerate(
        [&](const std::vector<size_t>& i, const std::vector<SpinType>& spin, const double& value) {
            if (spin[0] == AlphaSpin) {
                ints_->set_oei(i[0], i[1], value, true);
            } else {
                ints_->set_oei(i[0], i[1], value, false);
            }
        });
    V.citerate(
        [&](const std::vector<size_t>& i, const std::vector<SpinType>& spin, const double& value) {
            if ((spin[0] == AlphaSpin) && (spin[1] == AlphaSpin)) {
                ints_->set_tei(i[0], i[1], i[2], i[3], value, true, true);
            } else if ((spin[0] == AlphaSpin) && (spin[1] == BetaSpin)) {
                ints_->set_tei(i[0], i[1], i[2], i[3], value, true, false);
            } else if ((spin[0] == BetaSpin) && (spin[1] == BetaSpin)) {
                ints_->set_tei(i[0], i[1], i[2], i[3], value, false, false);
            }
        });
}

std::vector<std::vector<double>> MRDSRG::diagonalize_Fock_diagblocks(BlockedTensor& U) {
    // diagonal blocks identifiers (C-A-V ordering)
    std::vector<std::string> blocks = diag_one_labels();

    // map MO space label to its psi::Dimension
    std::map<std::string, psi::Dimension> MOlabel_to_dimension;
    MOlabel_to_dimension[acore_label_] = mo_space_info_->get_dimension("RESTRICTED_DOCC");
    MOlabel_to_dimension[aactv_label_] = mo_space_info_->get_dimension("ACTIVE");
    MOlabel_to_dimension[avirt_label_] = mo_space_info_->get_dimension("RESTRICTED_UOCC");

    // eigen values to be returned
    size_t ncmo = mo_space_info_->size("CORRELATED");
    psi::Dimension corr = mo_space_info_->get_dimension("CORRELATED");
    std::vector<double> eigenvalues_a(ncmo, 0.0);
    std::vector<double> eigenvalues_b(ncmo, 0.0);

    // map MO space label to its offset psi::Dimension
    std::map<std::string, psi::Dimension> MOlabel_to_offset_dimension;
    int nirrep = corr.n();
    MOlabel_to_offset_dimension[acore_label_] = psi::Dimension(std::vector<int>(nirrep, 0));
    MOlabel_to_offset_dimension[aactv_label_] = mo_space_info_->get_dimension("RESTRICTED_DOCC");
    MOlabel_to_offset_dimension[avirt_label_] =
        mo_space_info_->get_dimension("RESTRICTED_DOCC") + mo_space_info_->get_dimension("ACTIVE");

    // figure out index
    auto fill_eigen = [&](std::string block_label, int irrep, std::vector<double> values) {
        int h = irrep;
        size_t idx_begin = 0;
        while ((--h) >= 0)
            idx_begin += corr[h];

        std::string label(1, tolower(block_label[0]));
        idx_begin += MOlabel_to_offset_dimension[label][irrep];

        bool spin_alpha = islower(block_label[0]);
        size_t nvalues = values.size();
        if (spin_alpha) {
            for (size_t i = 0; i < nvalues; ++i) {
                eigenvalues_a[i + idx_begin] = values[i];
            }
        } else {
            for (size_t i = 0; i < nvalues; ++i) {
                eigenvalues_b[i + idx_begin] = values[i];
            }
        }
    };

    // diagonalize diagonal blocks
    for (const auto& block : blocks) {
        size_t dim = F_.block(block).dim(0);
        if (dim == 0) {
            continue;
        } else {
            std::string label(1, tolower(block[0]));
            psi::Dimension space = MOlabel_to_dimension[label];
            int nirrep = space.n();

            // separate Fock with irrep
            for (int h = 0; h < nirrep; ++h) {
                size_t h_dim = space[h];
                ambit::Tensor U_h;
                if (h_dim == 0) {
                    continue;
                } else if (h_dim == 1) {
                    U_h = ambit::Tensor::build(tensor_type_, "U_h", std::vector<size_t>(2, h_dim));
                    U_h.data()[0] = 1.0;
                    ambit::Tensor F_block =
                        ambit::Tensor::build(tensor_type_, "F_block", F_.block(block).dims());
                    F_block.data() = F_.block(block).data();
                    ambit::Tensor T_h = separate_tensor(F_block, space, h);
                    fill_eigen(block, h, T_h.data());
                } else {
                    ambit::Tensor F_block =
                        ambit::Tensor::build(tensor_type_, "F_block", F_.block(block).dims());
                    F_block.data() = F_.block(block).data();
                    ambit::Tensor T_h = separate_tensor(F_block, space, h);
                    auto Feigen = T_h.syev(AscendingEigenvalue);
                    U_h = ambit::Tensor::build(tensor_type_, "U_h", std::vector<size_t>(2, h_dim));
                    U_h("pq") = Feigen["eigenvectors"]("pq");
                    fill_eigen(block, h, Feigen["eigenvalues"].data());
                }
                ambit::Tensor U_out = U.block(block);
                combine_tensor(U_out, U_h, space, h);
            }
        }
    }
    return {eigenvalues_a, eigenvalues_b};
}

ambit::Tensor MRDSRG::separate_tensor(ambit::Tensor& tens, const psi::Dimension& irrep,
                                      const int& h) {
    // test tens and irrep
    int tens_dim = static_cast<int>(tens.dim(0));
    if (tens_dim != irrep.sum() || tens_dim != static_cast<int>(tens.dim(1))) {
        throw psi::PSIEXCEPTION("Wrong dimension for the to-be-separated ambit Tensor.");
    }
    if (h >= irrep.n()) {
        throw psi::PSIEXCEPTION("Ask for wrong irrep.");
    }

    // from relative (blocks) to absolute (big tensor) index
    auto rel_to_abs = [&](size_t i, size_t j, size_t offset) {
        return (i + offset) * tens_dim + (j + offset);
    };

    // compute offset
    size_t offset = 0, h_dim = irrep[h];
    int h_local = h;
    while ((--h_local) >= 0)
        offset += irrep[h_local];

    // fill in values
    ambit::Tensor T_h = ambit::Tensor::build(tensor_type_, "T_h", std::vector<size_t>(2, h_dim));
    for (size_t i = 0; i < h_dim; ++i) {
        for (size_t j = 0; j < h_dim; ++j) {
            size_t abs_idx = rel_to_abs(i, j, offset);
            T_h.data()[i * h_dim + j] = tens.data()[abs_idx];
        }
    }

    return T_h;
}

void MRDSRG::combine_tensor(ambit::Tensor& tens, ambit::Tensor& tens_h, const psi::Dimension& irrep,
                            const int& h) {
    // test tens and irrep
    if (h >= irrep.n()) {
        throw psi::PSIEXCEPTION("Ask for wrong irrep.");
    }
    size_t tens_h_dim = tens_h.dim(0), h_dim = irrep[h];
    if (tens_h_dim != h_dim || tens_h_dim != tens_h.dim(1)) {
        throw psi::PSIEXCEPTION("Wrong dimension for the to-be-combined ambit Tensor.");
    }

    // from relative (blocks) to absolute (big tensor) index
    size_t tens_dim = tens.dim(0);
    auto rel_to_abs = [&](size_t i, size_t j, size_t offset) {
        return (i + offset) * tens_dim + (j + offset);
    };

    // compute offset
    size_t offset = 0;
    int h_local = h;
    while ((--h_local) >= 0)
        offset += irrep[h_local];

    // fill in values
    for (size_t i = 0; i < h_dim; ++i) {
        for (size_t j = 0; j < h_dim; ++j) {
            size_t abs_idx = rel_to_abs(i, j, offset);
            tens.data()[abs_idx] = tens_h.data()[i * h_dim + j];
        }
    }
}

void MRDSRG::print_cumulant_summary() {
    print_h2("Density Cumulant Summary");

    // 2-body
    std::vector<double> maxes, norms;

    for (const std::string& block : {"aaaa", "aAaA", "AAAA"}) {
        maxes.push_back(Lambda2_.block(block).norm(0));
        norms.push_back(Lambda2_.block(block).norm(2));
    }

    std::string dash(8 + 13 * 3, '-');
    outfile->Printf("\n    %-8s %12s %12s %12s", "2-body", "AA", "AB", "BB");
    outfile->Printf("\n    %s", dash.c_str());
    outfile->Printf("\n    %-8s %12.6f %12.6f %12.6f", "max", maxes[0], maxes[1], maxes[2]);
    outfile->Printf("\n    %-8s %12.6f %12.6f %12.6f", "norm", norms[0], norms[1], norms[2]);
    outfile->Printf("\n    %s", dash.c_str());

    // 3-body
    maxes.clear();
    maxes.push_back(rdms_.L3aaa().norm(0));
    maxes.push_back(rdms_.L3aab().norm(0));
    maxes.push_back(rdms_.L3abb().norm(0));
    maxes.push_back(rdms_.L3bbb().norm(0));

    norms.clear();
    norms.push_back(rdms_.L3aaa().norm(2));
    norms.push_back(rdms_.L3aab().norm(2));
    norms.push_back(rdms_.L3abb().norm(2));
    norms.push_back(rdms_.L3bbb().norm(2));

    dash = std::string(8 + 13 * 4, '-');
    outfile->Printf("\n    %-8s %12s %12s %12s %12s", "3-body", "AAA", "AAB", "ABB", "BBB");
    outfile->Printf("\n    %s", dash.c_str());
    outfile->Printf("\n    %-8s %12.6f %12.6f %12.6f %12.6f", "max", maxes[0], maxes[1], maxes[2],
                    maxes[3]);
    outfile->Printf("\n    %-8s %12.6f %12.6f %12.6f %12.6f", "norm", norms[0], norms[1], norms[2],
                    norms[3]);
    outfile->Printf("\n    %s", dash.c_str());

    //    check_density(Lambda2_, "2-body");
    //    if (foptions_->get_str("THREEPDC") != "ZERO") {
    //        check_density(Lambda3_, "3-body");
    //    }
}

void MRDSRG::check_density(BlockedTensor& D, const std::string& name) {
    int rank_half = D.rank() / 2;
    std::vector<std::string> labels;
    std::vector<double> maxes, norms;
    std::vector<std::string> blocks = D.block_labels();
    for (const auto& block : blocks) {
        std::string spin_label;
        std::vector<int> idx;
        for (int i = 0; i < rank_half; ++i) {
            idx.emplace_back(i);
        }
        for (const auto& i : idx) {
            if (islower(block[i])) {
                spin_label += "A";
            } else {
                spin_label += "B";
            }
        }
        labels.emplace_back(spin_label);

        double D_norm = 0.0, D_max = 0.0;
        D.block(block).citerate([&](const std::vector<size_t>&, const double& value) {
            double abs_value = std::fabs(value);
            if (abs_value > 1.0e-15) {
                if (abs_value > D_max)
                    D_max = value;
                D_norm += value * value;
            }
        });
        maxes.emplace_back(D_max);
        norms.emplace_back(std::sqrt(D_norm));
    }

    int n = labels.size();
    std::string sep(10 + 13 * n, '-');
    std::string indent = "\n    ";
    std::string output = indent + str(boost::format("%-10s") % name);
    for (int i = 0; i < n; ++i)
        output += str(boost::format(" %12s") % labels[i]);
    output += indent + sep;

    output += indent + str(boost::format("%-10s") % "max");
    for (int i = 0; i < n; ++i)
        output += str(boost::format(" %12.6f") % maxes[i]);
    output += indent + str(boost::format("%-10s") % "norm");
    for (int i = 0; i < n; ++i)
        output += str(boost::format(" %12.6f") % norms[i]);
    output += indent + sep;
    outfile->Printf("%s", output.c_str());
}
} // namespace forte
