/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2020 by its authors (see COPYING, COPYING.LESSER, AUTHORS).
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
#include "sci/fci_mo.h"
#include "helpers/printing.h"
#include "orbital-helpers/semi_canonicalize.h"
#include "orbital-helpers/mp2_nos.h"
#include "mrdsrg.h"

using namespace psi;

namespace forte {

MRDSRG::MRDSRG(RDMs rdms, std::shared_ptr<SCFInfo> scf_info, std::shared_ptr<ForteOptions> options,
               std::shared_ptr<ForteIntegrals> ints, std::shared_ptr<MOSpaceInfo> mo_space_info)
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
}

void MRDSRG::print_options() {
    // fill in information
    std::vector<std::pair<std::string, int>> calculation_info{
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
        {"Core-Virtual source type", foptions_->get_str("CCVV_SOURCE")}};

    if (corrlv_string_ == "PT2") {
        calculation_info_string.push_back({"PT2 0-order Hamiltonian", pt2_h0th_});
    }

    auto true_false_string = [](bool x) {
        if (x) {
            return std::string("TRUE");
        } else {
            return std::string("FALSE");
        }
    };
    calculation_info_string.push_back(
        {"Restart amplitudes", true_false_string(restart_amps_)});
    calculation_info_string.push_back(
        {"Sequential DSRG transformation", true_false_string(sequential_Hbar_)});
    calculation_info_string.push_back(
        {"Omit blocks of >= 3 virtual indices", true_false_string(nivo_)});
    calculation_info_string.push_back(
        {"Read amplitudes from current dir", true_false_string(read_amps_cwd_)});
    calculation_info_string.push_back(
        {"Write amplitudes to current dir", true_false_string(dump_amps_cwd_)});

    // print some information
    print_h2("Calculation Information");
    for (auto& str_dim : calculation_info) {
        outfile->Printf("\n    %-40s %15d", str_dim.first.c_str(), str_dim.second);
    }
    for (auto& str_dim : calculation_info_double) {
        outfile->Printf("\n    %-40s %15.3e", str_dim.first.c_str(), str_dim.second);
    }
    for (auto& str_dim : calculation_info_string) {
        outfile->Printf("\n    %-40s %15s", str_dim.first.c_str(), str_dim.second.c_str());
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
    MOlabel_to_dimension[acore_label_] = mo_space_info_->dimension("RESTRICTED_DOCC");
    MOlabel_to_dimension[aactv_label_] = mo_space_info_->dimension("ACTIVE");
    MOlabel_to_dimension[avirt_label_] = mo_space_info_->dimension("RESTRICTED_UOCC");

    // eigen values to be returned
    size_t ncmo = mo_space_info_->size("CORRELATED");
    psi::Dimension corr = mo_space_info_->dimension("CORRELATED");
    std::vector<double> eigenvalues_a(ncmo, 0.0);
    std::vector<double> eigenvalues_b(ncmo, 0.0);

    // map MO space label to its offset psi::Dimension
    std::map<std::string, psi::Dimension> MOlabel_to_offset_dimension;
    int nirrep = corr.n();
    MOlabel_to_offset_dimension[acore_label_] = psi::Dimension(std::vector<int>(nirrep, 0));
    MOlabel_to_offset_dimension[aactv_label_] = mo_space_info_->dimension("RESTRICTED_DOCC");
    MOlabel_to_offset_dimension[avirt_label_] =
        mo_space_info_->dimension("RESTRICTED_DOCC") + mo_space_info_->dimension("ACTIVE");

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
