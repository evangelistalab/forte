/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2020 by its authors (see COPYING, COPYING.LESSER,
 * AUTHORS).
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
#include <cctype>
#include <map>
#include <memory>
#include <vector>

#include "psi4/libpsi4util/process.h"
#include "psi4/libdiis/diismanager.h"
#include "psi4/libmints/molecule.h"
#include "psi4/libqt/qt.h"

#include "helpers/printing.h"
#include "sa_mrpt2.h"

using namespace psi;

namespace forte {

SA_MRPT2::SA_MRPT2(RDMs rdms, std::shared_ptr<SCFInfo> scf_info,
                   std::shared_ptr<ForteOptions> options, std::shared_ptr<ForteIntegrals> ints,
                   std::shared_ptr<MOSpaceInfo> mo_space_info)
    : SADSRG(rdms, scf_info, options, ints, mo_space_info) {

    print_method_banner({"MR-DSRG Second-Order Perturbation Theory"});
    startup();
    read_options();
    print_options();
}

void SA_MRPT2::startup() {
    // test semi-canonical
    semi_canonical_ = check_semi_orbs();

    if (!semi_canonical_) {
        if (ints_type_ == "DISKDF") {
            outfile->Printf("\n  Orbitals are not semicanonicalized: ");
            outfile->Printf("NOT OK for DSRG-MRPT2 with DISKDF integrals.");
            throw std::runtime_error("Orbitals are not semicanonicalized for DSRG-MRPT2. Quit.");
        }

        outfile->Printf("\n    Orbital invariant formalism will be employed for MR-DSRG.");
        U_ = ambit::BlockedTensor::build(tensor_type_, "U", {"gg"});
        Fdiag_ = diagonalize_Fock_diagblocks(U_);
    }

    // link F_ with Fock_ of SADSRG
    F_ = Fock_;

    // prepare integrals
    build_ints();
}

void SA_MRPT2::read_options() {
    internal_amp_ = foptions_->get_str("INTERNAL_AMP");
    internal_amp_select_ = foptions_->get_str("INTERNAL_AMP_SELECT");
}

void SA_MRPT2::print_options() {
    // Print a summary
    std::vector<std::pair<std::string, int>> calculation_info_int{
        {"Number of amplitudes for printing", ntamp_}};

    std::vector<std::pair<std::string, double>> calculation_info_double{
        {"Flow parameter", s_},
        {"Taylor expansion threshold", std::pow(10.0, -double(taylor_threshold_))},
        {"Intruder amplitudes threshold", intruder_tamp_}};

    if (ints_type_ == "CHOLESKY") {
        auto cholesky_threshold = foptions_->get_double("CHOLESKY_TOLERANCE");
        calculation_info_double.push_back({"Cholesky tolerance", cholesky_threshold});
    }

    std::vector<std::pair<std::string, std::string>> calculation_info_string{
        {"Integral type", ints_type_},
        {"Source operator", source_},
        {"Core-Virtual source type", ccvv_source_},
        {"Reference relaxation", relax_ref_},
        {"Internal amplitudes", internal_amp_}};

    if (multi_state_) {
        calculation_info_string.push_back({"State type", "multiple state"});
        calculation_info_string.push_back({"Multi-state type", multi_state_algorithm_});
    } else {
        calculation_info_string.push_back({"State type", "state specific"});
    }

    if (internal_amp_ != "NONE") {
        calculation_info_string.push_back({"Internal amplitudes selection", internal_amp_select_});
    }

    // Print some information
    print_h2("Calculation Information");
    for (auto& str_dim : calculation_info_int) {
        outfile->Printf("\n    %-40s %15d", str_dim.first.c_str(), str_dim.second);
    }
    for (auto& str_dim : calculation_info_double) {
        outfile->Printf("\n    %-40s %15.3e", str_dim.first.c_str(), str_dim.second);
    }
    for (auto& str_dim : calculation_info_string) {
        outfile->Printf("\n    %-40s %15s", str_dim.first.c_str(), str_dim.second.c_str());
    }
}

void SA_MRPT2::build_ints() {
    // prepare one-electron integrals
    H_ = BTF_->build(tensor_type_, "H", {"gg"});
    H_.iterate([&](const std::vector<size_t>& i, const std::vector<SpinType>&, double& value) {
        value = ints_->oei_a(i[0], i[1]);
    });

    // prepare two-electron integrals or three-index B
    if (eri_df_) {
        if (ints_type_ != "DISKDF") {
            B_ = BTF_->build(tensor_type_, "B 3-idx", {"Lph"});
            fill_three_index_ints(B_);
        }
    } else {
        V_ = BTF_->build(tensor_type_, "V", {"pphh"});

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

double SA_MRPT2::compute_energy() {
    // build amplitudes

    // compute energy
    double Ecorr = 0.0;

    return Ecorr;
}

void SA_MRPT2::compute_t2_minimal() {}

void SA_MRPT2::compute_t1() {}

void SA_MRPT2::compute_hbar() {}
} // namespace forte
