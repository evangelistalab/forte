/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2024 by its authors (see COPYING, COPYING.LESSER, AUTHORS).
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

#include <unistd.h>
#include <algorithm>

#include "psi4/libmints/molecule.h"
#include "psi4/libpsi4util/process.h"
#include "psi4/libpsi4util/PsiOutStream.h"
#include "psi4/libpsio/psio.hpp"

#include "helpers/printing.h"
#include "helpers/timer.h"
#include "sa_mrdsrg.h"

using namespace psi;

namespace forte {

SA_MRDSRG::SA_MRDSRG(std::shared_ptr<RDMs> rdms, std::shared_ptr<SCFInfo> scf_info,
                     std::shared_ptr<ForteOptions> options, std::shared_ptr<ForteIntegrals> ints,
                     std::shared_ptr<MOSpaceInfo> mo_space_info)
    : SADSRG(rdms, scf_info, options, ints, mo_space_info) {
    read_options();
    print_options();
    check_memory();
    startup();
}

void SA_MRDSRG::read_options() {
    corrlv_string_ = foptions_->get_str("CORR_LEVEL");
    std::vector<std::string> available{"LDSRG2", "LDSRG2_QC"};
    if (std::find(available.begin(), available.end(), corrlv_string_) == available.end()) {
        outfile->Printf("\n  Warning: CORR_LEVEL option %s is not implemented.",
                        corrlv_string_.c_str());
        outfile->Printf("\n  Changed CORR_LEVEL option to LDSRG2_QC");

        corrlv_string_ = "LDSRG2_QC";
        warnings_.push_back(std::make_tuple("Unsupported CORR_LEVEL", "Change to LDSRG2_QC",
                                            "Change options in input.dat"));
    }

    sequential_Hbar_ = foptions_->get_bool("DSRG_HBAR_SEQ");
    nivo_ = foptions_->get_bool("DSRG_NIVO");

    rsc_ncomm_ = foptions_->get_int("DSRG_RSC_NCOMM");
    rsc_conv_ = foptions_->get_double("DSRG_RSC_THRESHOLD");

    maxiter_ = foptions_->get_int("MAXITER");
    e_conv_ = foptions_->get_double("E_CONVERGENCE");
    r_conv_ = foptions_->get_double("R_CONVERGENCE");

    restart_amps_ = foptions_->get_bool("DSRG_RESTART_AMPS");
    t1_guess_ = foptions_->get_str("DSRG_T1_AMPS_GUESS");
}

void SA_MRDSRG::startup() {
    // prepare integrals
    build_ints();

    // link F_ with Fock_ of SADSRG
    F_ = Fock_;

    // test semi-canonical
    if (!semi_canonical_) {
        outfile->Printf("\n  Orbital invariant formalism will be employed for MR-DSRG.");
        U_ = ambit::BlockedTensor::build(tensor_type_, "U", {"cc", "aa", "vv"});
        Fdiag_ = diagonalize_Fock_diagblocks(U_);
    }

    // determine file names
    chk_filename_prefix_ = psi::PSIOManager::shared_object()->get_default_path() + "forte." +
                           std::to_string(getpid()) + "." +
                           psi::Process::environment.molecule()->name();
    t1_file_chk_.clear();
    t2_file_chk_.clear();
    if (restart_amps_ and (relax_ref_ != "NONE")) {
        t1_file_chk_ = chk_filename_prefix_ + ".mrdsrg.adapted.t1.bin";
        t2_file_chk_ = chk_filename_prefix_ + ".mrdsrg.adapted.t2.bin";
    }

    t1_file_cwd_ = "forte.mrdsrg.adapted.t1.bin";
    t2_file_cwd_ = "forte.mrdsrg.adapted.t2.bin";
}

void SA_MRDSRG::print_options() {
    // fill in information
    table_printer printer;
    printer.add_int_data({{"Max number of iterations", maxiter_},
                          {"Max nested commutators", rsc_ncomm_},
                          {"DIIS start", diis_start_},
                          {"Min DIIS vectors", diis_min_vec_},
                          {"Max DIIS vectors", diis_max_vec_},
                          {"DIIS extrapolating freq", diis_freq_},
                          {"Number of amplitudes for printing", ntamp_}});
    printer.add_double_data({{"Flow parameter", s_},
                             {"Energy convergence threshold", e_conv_},
                             {"Residual convergence threshold", r_conv_},
                             {"Recursive single commutator threshold", rsc_conv_},
                             {"Taylor expansion threshold", pow(10.0, -double(taylor_threshold_))},
                             {"Intruder amplitudes threshold", intruder_tamp_}});
    printer.add_bool_data({{"Restart amplitudes", restart_amps_},
                           {"Sequential DSRG transformation", sequential_Hbar_},
                           {"Omit blocks of >= 3 virtual indices", nivo_},
                           {"Read amplitudes from current dir", read_amps_cwd_},
                           {"Write amplitudes to current dir", dump_amps_cwd_}});

    std::vector<std::pair<std::string, std::string>> calculation_info_string{
        {"Correlation level", corrlv_string_},
        {"Integral type", ints_type_},
        {"Source operator", source_},
        {"Reference relaxation", relax_ref_},
        {"3RDM algorithm", L3_algorithm_},
        {"Core-Virtual source type", ccvv_source_},
        {"T1 amplitudes initial guess", t1_guess_}};

    if (internal_amp_ != "NONE") {
        calculation_info_string.emplace_back("Internal amplitudes levels", internal_amp_);
        calculation_info_string.emplace_back("Internal amplitudes selection", internal_amp_select_);
    }

    // print information
    printer.add_string_data(calculation_info_string);
    std::string table = printer.get_table("Calculation Information");
    psi::outfile->Printf("%s", table.c_str());
}

void SA_MRDSRG::check_memory() {
    if (eri_df_) {
        dsrg_mem_.add_entry("1-electron and 3-index integrals", {"gg", "Lgg"});
    } else {
        dsrg_mem_.add_entry("1- and 2-electron integrals", {"gg", "gggg"});
    }

    dsrg_mem_.add_entry("T1 cluster amplitudes and residuals", {"hp"}, 2);
    dsrg_mem_.add_entry("T2 cluster amplitudes and residuals", {"hhpp"}, 3); // T2, S2, DT2

    if (corrlv_string_ == "LDSRG2_QC") {
        dsrg_mem_.add_entry("1- and 2-body Hbar", {"hhpp", "hp"});
        dsrg_mem_.add_entry("1- and 2-body intermediates", {"gg", "gggg", "hhpp"});
    } else {
        dsrg_mem_.add_entry("1-body Hbar and intermediates", {"gg"}, 3);
        if (nivo_) {
            dsrg_mem_.add_entry("2-body Hbar and intermediates", nivo_labels(), 3);
        } else {
            dsrg_mem_.add_entry("2-body Hbar and intermediates", {"gggg"}, 3);
        }

        if (sequential_Hbar_) {
            size_t mem_seq =
                eri_df_ ? dsrg_mem_.compute_memory({"Lgg"}) : dsrg_mem_.compute_memory({"gggg"});
            dsrg_mem_.add_entry("Local intermediates for sequential Hbar", mem_seq, false);
        }
    }

    // intermediates used in actual commutator computation
    size_t mem_comm = dsrg_mem_.compute_memory({"hhpp", "ahpp", "hhhp"}) * 2;
    if ((!eri_df_) and (!nivo_)) {
        mem_comm = std::max(mem_comm, dsrg_mem_.compute_memory({"ppph"}));
    }
    dsrg_mem_.add_entry("Local intermediates for commutators", mem_comm, false);

    dsrg_mem_.print("MR-DSRG (" + corrlv_string_ + ")");
}

void SA_MRDSRG::build_ints() {
    // prepare one-electron integrals
    H_ = BTF_->build(tensor_type_, "H", {"gg"});
    H_.iterate([&](const std::vector<size_t>& i, const std::vector<SpinType>&, double& value) {
        value = ints_->oei_a(i[0], i[1]);
    });

    // prepare two-electron integrals or three-index B
    if (eri_df_) {
        B_ = BTF_->build(tensor_type_, "B 3-idx", {"Lgg"});
        fill_three_index_ints(B_);
    } else {
        V_ = BTF_->build(tensor_type_, "V", {"gggg"});

        for (const std::string& block : V_.block_labels()) {
            auto mo_to_index = BTF_->get_mo_to_index();

            std::vector<size_t> i0 = mo_to_index[block.substr(0, 1)];
            std::vector<size_t> i1 = mo_to_index[block.substr(1, 1)];
            std::vector<size_t> i2 = mo_to_index[block.substr(2, 1)];
            std::vector<size_t> i3 = mo_to_index[block.substr(3, 1)];

            auto Vblock = ints_->aptei_ab_block(i0, i1, i2, i3);
            V_.block(block).copy(Vblock);
        }
    }
}

double SA_MRDSRG::compute_energy() {
    // build initial amplitudes
    T1_ = BTF_->build(tensor_type_, "T1 Amplitudes", {"hp"});
    T2_ = BTF_->build(tensor_type_, "T2 Amplitudes", {"hhpp"});
    guess_t(V_, T2_, F_, T1_, B_);

    // get reference energy
    double Etotal = Eref_;

    // compute energy
    Etotal += compute_energy_ldsrg2();
    //    switch (corrlevelmap[corrlv_string_]) {
    //    case CORR_LV::LDSRG2: {
    //        Etotal += compute_energy_ldsrg2();
    //        break;
    //    }
    //    default: { Etotal += compute_energy_ldsrg2_qc(); }
    //    }

    return Etotal;
}

double SA_MRDSRG::Hbar_od_norm(const int& n, const std::vector<std::string>& blocks) {
    double norm = 0.0;

    auto T = (n == 1) ? Hbar1_ : Hbar2_;

    for (auto& block : blocks) {
        double norm_block = T.block(block).norm();
        norm += 2.0 * norm_block * norm_block;
    }
    norm = std::sqrt(norm);

    return norm;
}

void SA_MRDSRG::transform_one_body(const std::vector<ambit::BlockedTensor>& oetens,
                                   const std::vector<int>& max_levels) {
    print_h2("Transform One-Electron Operators");

    if (corrlv_string_ == "LDSRG2_QC")
        throw std::runtime_error(
            "Not available for LDSRG2_QC: Try LDSRG2 with DSRG_RSC_NCOMM = 2.");

    int n_tensors = oetens.size();
    Mbar0_ = std::vector<double>(n_tensors, 0.0);
    Mbar1_.resize(n_tensors);
    Mbar2_.resize(n_tensors);
    for (int i = 0; i < n_tensors; ++i) {
        Mbar1_[i] = BTF_->build(tensor_type_, oetens[i].name() + "1", {"aa"});
        if (max_levels[i] > 1)
            Mbar2_[i] = BTF_->build(tensor_type_, oetens[i].name() + "2", {"aaaa"});
    }

    auto max_body = *std::max_element(max_levels.begin(), max_levels.end());
    if (max_body > 1) {
        DT2_["ijab"] = 2.0 * T2_["ijab"] - T2_["ijba"];
    }

    for (int i = 0; i < n_tensors; ++i) {
        local_timer t_local;
        const auto& M = oetens[i];
        print_contents("Transforming " + M.name());
        compute_mbar_ldsrg2(M, max_levels[i], i);
        print_done(t_local.get());
    }
}

} // namespace forte
