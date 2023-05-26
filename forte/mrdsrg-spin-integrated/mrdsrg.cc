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

#include <algorithm>
#include <cmath>
#include <map>
#include <vector>

#include "boost/format.hpp"

#include "psi4/libmints/molecule.h"
#include "psi4/libpsi4util/process.h"
#include "psi4/libpsi4util/PsiOutStream.h"
#include "psi4/libmints/molecule.h"

#include "base_classes/active_space_solver.h"
#include "fci/fci_solver.h"
#include "helpers/printing.h"
#include "orbital-helpers/semi_canonicalize.h"
#include "orbital-helpers/mp2_nos.h"
#include "mrdsrg.h"

using namespace psi;

namespace forte {

MRDSRG::MRDSRG(std::shared_ptr<RDMs> rdms, std::shared_ptr<SCFInfo> scf_info,
               std::shared_ptr<ForteOptions> options, std::shared_ptr<ForteIntegrals> ints,
               std::shared_ptr<MOSpaceInfo> mo_space_info)
    : MASTER_DSRG(rdms, scf_info, options, ints, mo_space_info) {

    print_method_banner({"Multireference Driven Similarity Renormalization Group",
                         "written by Chenyang Li and Tianyuan Zhang"});

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

    pt2_h0th_ = foptions_->get_str("DSRG_PT2_H0TH");
    if (pt2_h0th_ != "FFULL" and pt2_h0th_ != "FDIAG_VACTV" and pt2_h0th_ != "FDIAG_VDIAG") {
        pt2_h0th_ = "FDIAG";
    }

    restart_amps_ = foptions_->get_bool("DSRG_RESTART_AMPS");
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

    // set up file name prefix
    restart_file_prefix_ = psi::PSIOManager::shared_object()->get_default_path() + "forte." +
                           std::to_string(getpid()) + "." +
                           psi::Process::environment.molecule()->name();
    t1_file_chk_.clear();
    t2_file_chk_.clear();
    if (restart_amps_ and (relax_ref_ != "NONE") and
        corrlv_string_.find("DSRG") != std::string::npos) {
        t1_file_chk_ = restart_file_prefix_ + ".mrdsrg.spin.t1.bin";
        t2_file_chk_ = restart_file_prefix_ + ".mrdsrg.spin.t2.bin";
    }

    t1_file_cwd_ = "forte.mrdsrg.spin.t1.bin";
    t2_file_cwd_ = "forte.mrdsrg.spin.t2.bin";
}

void MRDSRG::print_options() {
    // fill in information
    std::vector<std::pair<std::string, int>> calculation_info_int{
        {"Number of T amplitudes", ntamp_},
        {"DIIS start", diis_start_},
        {"Min DIIS vectors", diis_min_vec_},
        {"Max DIIS vectors", diis_max_vec_},
        {"DIIS extrapolating freq", diis_freq_}};

    std::vector<std::pair<std::string, double>> calculation_info_double{
        {"Flow parameter", s_},
        {"Taylor expansion threshold", pow(10.0, -double(taylor_threshold_))},
        {"Intruder amplitudes threshold", intruder_tamp_}};

    std::vector<std::pair<std::string, std::string>> calculation_info_string{
        {"Correlation level", corrlv_string_},
        {"Integral type", ints_type_},
        {"Source operator", source_},
        {"Adaptive DSRG flow type", foptions_->get_str("SMART_DSRG_S")},
        {"Reference relaxation", relax_ref_},
        {"DSRG transformation type", dsrg_trans_type_},
        {"Core-Virtual source type", foptions_->get_str("CCVV_SOURCE")},
        {"T1 amplitudes initial guess", foptions_->get_str("DSRG_T1_AMPS_GUESS")}};

    if (corrlv_string_ == "PT2") {
        calculation_info_string.emplace_back("PT2 0-order Hamiltonian", pt2_h0th_);
    }

    std::vector<std::pair<std::string, bool>> calculation_info_bool{
        {"Restart amplitudes", restart_amps_},
        {"Sequential DSRG transformation", sequential_Hbar_},
        {"Omit blocks of >= 3 virtual indices", nivo_},
        {"Read amplitudes from current dir", read_amps_cwd_},
        {"Write amplitudes to current dir", dump_amps_cwd_}};

    // print information
    print_selected_options("Calculation Information", calculation_info_string,
                           calculation_info_bool, calculation_info_double, calculation_info_int);
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
        T1_ = BTF_->build(tensor_type_, "T1 Amplitudes", spin_cases({"hp"}));
        T2_ = BTF_->build(tensor_type_, "T2 Amplitudes", spin_cases({"hhpp"}));
        if (eri_df_) {
            guess_t_df(B_, T2_, F_, T1_);
        } else {
            guess_t(V_, T2_, F_, T1_);
        }
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

void MRDSRG::print_cumulant_summary() {
    print_h2("Density Cumulant Summary");

    // 2-body
    std::vector<double> maxes, norms;

    for (const std::string block : {"aaaa", "aAaA", "AAAA"}) {
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
    maxes.push_back(L3aaa_.norm(0));
    maxes.push_back(L3aab_.norm(0));
    maxes.push_back(L3abb_.norm(0));
    maxes.push_back(L3bbb_.norm(0));

    norms.clear();
    norms.push_back(L3aaa_.norm(2));
    norms.push_back(L3aab_.norm(2));
    norms.push_back(L3abb_.norm(2));
    norms.push_back(L3bbb_.norm(2));

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

double MRDSRG::Hbar1od_norm(const std::vector<std::string>& blocks) {
    double norm = 0.0;

    for (auto& block : blocks) {
        double norm_block = Hbar1_.block(block).norm();
        norm += 2.0 * norm_block * norm_block;
    }
    norm = std::sqrt(norm);

    return norm;
}

double MRDSRG::Hbar2od_norm(const std::vector<std::string>& blocks) {
    double norm = 0.0;

    for (auto& block : blocks) {
        double norm_block = Hbar2_.block(block).norm();
        norm += 2.0 * norm_block * norm_block;
    }
    norm = std::sqrt(norm);

    return norm;
}

} // namespace forte
