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

#include "methodbase.h"

#include "helpers/mo_space_info.h"
#include "psi4/libpsio/psio.h"
#include "psi4/libpsio/psio.hpp"

namespace psi {
namespace forte {

using namespace ambit;

MethodBase::MethodBase(SharedWavefunction ref_wfn, Options& options,
                       std::shared_ptr<ForteIntegrals> ints,
                       std::shared_ptr<MOSpaceInfo> mo_space_info)
    : Wavefunction(options), ints_(ints), tensor_type_(CoreTensor), mo_space_info_(mo_space_info) {
    // Copy the wavefunction information
    shallow_copy(ref_wfn);
    ref_wfn_ = ref_wfn;
    //    Tensor::set_print_level(debug_);
    startup();
}

MethodBase::~MethodBase() { cleanup(); }

void MethodBase::startup() {
    E0_ = reference_energy();

    BlockedTensor::set_expert_mode(true);

    size_t ncmo_ = mo_space_info_->size("CORRELATED");
    Dimension ncmopi_ = mo_space_info_->dimension("CORRELATED");
    frzcpi_ = mo_space_info_->dimension("FROZEN_DOCC");

    Dimension corr_docc(doccpi_);
    corr_docc -= frzcpi_;

    for (int h = 0, p = 0; h < nirrep_; ++h) {
        for (int i = 0; i < corr_docc[h]; ++i, ++p) {
            a_occ_mos.push_back(p);
            b_occ_mos.push_back(p);
        }
        for (int i = 0; i < soccpi_[h]; ++i, ++p) {
            a_occ_mos.push_back(p);
            b_vir_mos.push_back(p);
        }
        for (int a = 0; a < ncmopi_[h] - corr_docc[h] - soccpi_[h]; ++a, ++p) {
            a_vir_mos.push_back(p);
            b_vir_mos.push_back(p);
        }
    }

    for (size_t p = 0; p < a_occ_mos.size(); ++p)
        mos_to_aocc[a_occ_mos[p]] = p;
    for (size_t p = 0; p < b_occ_mos.size(); ++p)
        mos_to_bocc[b_occ_mos[p]] = p;
    for (size_t p = 0; p < a_vir_mos.size(); ++p)
        mos_to_avir[a_vir_mos[p]] = p;
    for (size_t p = 0; p < b_vir_mos.size(); ++p)
        mos_to_bvir[b_vir_mos[p]] = p;

    BlockedTensor::add_mo_space("o", "ijklmn", a_occ_mos, AlphaSpin);
    BlockedTensor::add_mo_space("O", "IJKLMN", b_occ_mos, BetaSpin);
    BlockedTensor::add_mo_space("v", "abcdef", a_vir_mos, AlphaSpin);
    BlockedTensor::add_mo_space("V", "ABCDEF", b_vir_mos, BetaSpin);
    BlockedTensor::add_composite_mo_space("i", "pqrstuvwxyz", {"o", "v"});
    BlockedTensor::add_composite_mo_space("I", "PQRSTUVWXYZ", {"O", "V"});

    H = BlockedTensor::build(tensor_type_, "H", spin_cases({"ii"}));
    G1 = BlockedTensor::build(tensor_type_, "G1", spin_cases({"oo"}));
    CG1 = BlockedTensor::build(tensor_type_, "CG1", spin_cases({"vv"}));
    F = BlockedTensor::build(tensor_type_, "F", spin_cases({"ii"}));
    V = BlockedTensor::build(tensor_type_, "V", spin_cases({"iiii"}));
    InvD1 = BlockedTensor::build(tensor_type_, "Inverse D1", spin_cases({"ov"}));
    InvD2 = BlockedTensor::build(tensor_type_, "Inverse D2", spin_cases({"oovv"}));

    // Fill in the one-electron operator (H)
    H.iterate([&](const std::vector<size_t>& i, const std::vector<SpinType>& spin, double& value) {
        if (spin[0] == AlphaSpin)
            value = ints_->oei_a(i[0], i[1]);
        else
            value = ints_->oei_b(i[0], i[1]);
    });

    // Fill in the two-electron operator (V)
    V.iterate([&](const std::vector<size_t>& i, const std::vector<SpinType>& spin, double& value) {
        if ((spin[0] == AlphaSpin) and (spin[1] == AlphaSpin))
            value = ints_->aptei_aa(i[0], i[1], i[2], i[3]);
        if ((spin[0] == AlphaSpin) and (spin[1] == BetaSpin))
            value = ints_->aptei_ab(i[0], i[1], i[2], i[3]);
        if ((spin[0] == BetaSpin) and (spin[1] == BetaSpin))
            value = ints_->aptei_bb(i[0], i[1], i[2], i[3]);
    });

    G1.iterate([&](const std::vector<size_t>& i, const std::vector<SpinType>& spin, double& value) {
        value = i[0] == i[1] ? 1.0 : 0.0;
    });

    CG1.iterate([&](const std::vector<size_t>& i, const std::vector<SpinType>& spin,
                    double& value) { value = i[0] == i[1] ? 1.0 : 0.0; });

    // Form the Fock matrix
    F["pq"] = H["pq"];
    F["pq"] += V["prqs"] * G1["sr"];
    F["pq"] += V["pRqS"] * G1["SR"];

    F["PQ"] += H["PQ"];
    F["PQ"] += V["rPsQ"] * G1["sr"];
    F["PQ"] += V["PRQS"] * G1["SR"];

    //    if (print_ > 2){
    //        G1.print();
    //        CG1.print();
    //        H.print();
    //        F.print();
    //    }

    std::vector<double> Fa(ncmo_);
    std::vector<double> Fb(ncmo_);

    F.iterate([&](const std::vector<size_t>& i, const std::vector<SpinType>& spin, double& value) {
        if (spin[0] == AlphaSpin and (i[0] == i[1])) {
            Fa[i[0]] = value;
        }
        if (spin[0] == BetaSpin and (i[0] == i[1])) {
            Fb[i[0]] = value;
        }
    });

    InvD1.iterate(
        [&](const std::vector<size_t>& i, const std::vector<SpinType>& spin, double& value) {
            if (spin[0] == AlphaSpin) {
                value = 1.0 / (Fa[i[0]] - Fa[i[1]]);
            } else if (spin[0] == BetaSpin) {
                value = 1.0 / (Fb[i[0]] - Fb[i[1]]);
            }
        });

    InvD2.iterate(
        [&](const std::vector<size_t>& i, const std::vector<SpinType>& spin, double& value) {
            if ((spin[0] == AlphaSpin) and (spin[1] == AlphaSpin)) {
                value = 1.0 / (Fa[i[0]] + Fa[i[1]] - Fa[i[2]] - Fa[i[3]]);
            } else if ((spin[0] == AlphaSpin) and (spin[1] == BetaSpin)) {
                value = 1.0 / (Fa[i[0]] + Fb[i[1]] - Fa[i[2]] - Fb[i[3]]);
            } else if ((spin[0] == BetaSpin) and (spin[1] == BetaSpin)) {
                value = 1.0 / (Fb[i[0]] + Fb[i[1]] - Fb[i[2]] - Fb[i[3]]);
            }
        });
}

void MethodBase::cleanup() { BlockedTensor::set_expert_mode(false); }
}
} // End Namespaces
