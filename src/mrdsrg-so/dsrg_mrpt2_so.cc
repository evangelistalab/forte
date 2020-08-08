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
#include <map>
#include <vector>

#include "psi4/libpsi4util/process.h"
#include "psi4/libmints/molecule.h"

#include "base_classes/mo_space_info.h"
#include "helpers/printing.h"
#include "helpers/timer.h"
#include "dsrg_mrpt2_so.h"

using namespace psi;

namespace forte {

DSRG_MRPT2_SO::DSRG_MRPT2_SO(RDMs rdms, std::shared_ptr<SCFInfo> scf_info,
                             std::shared_ptr<ForteOptions> options,
                             std::shared_ptr<ForteIntegrals> ints,
                             std::shared_ptr<MOSpaceInfo> mo_space_info)
    : DynamicCorrelationSolver(rdms, scf_info, options, ints, mo_space_info),
      BTF_(new BlockedTensorFactory()), tensor_type_(ambit::CoreTensor) {
    print_method_banner({"Spin-Orbital DSRG-MRPT2", "Chenyang Li"});
    startup();
    print_summary();
}

std::shared_ptr<ActiveSpaceIntegrals> DSRG_MRPT2_SO::compute_Heff_actv() {
    throw std::runtime_error(
        "Computing active-space Hamiltonian is not yet implemented for spin-orbital code.");
}

void DSRG_MRPT2_SO::startup() {
    s_ = foptions_->get_double("DSRG_S");
    taylor_threshold_ = foptions_->get_int("TAYLOR_THRESHOLD");
    source_ = foptions_->get_str("SOURCE");
    if (source_ == "LABS") {
        dsrg_source_ = std::make_shared<LABS_SOURCE>(s_, taylor_threshold_);
    } else if (source_ == "DYSON") {
        dsrg_source_ = std::make_shared<DYSON_SOURCE>(s_, taylor_threshold_);
    } else {
        dsrg_source_ = std::make_shared<STD_SOURCE>(s_, taylor_threshold_);
    }

    // orbital spaces
    acore_sos_ = mo_space_info_->corr_absolute_mo("RESTRICTED_DOCC");
    aactv_sos_ = mo_space_info_->corr_absolute_mo("ACTIVE");
    avirt_sos_ = mo_space_info_->corr_absolute_mo("RESTRICTED_UOCC");

    // put all beta behind alpha
    size_t mo_shift = mo_space_info_->size("CORRELATED");

    for (size_t idx : acore_sos_)
        bcore_sos_.push_back(idx + mo_shift);
    for (size_t idx : aactv_sos_)
        bactv_sos_.push_back(idx + mo_shift);
    for (size_t idx : avirt_sos_)
        bvirt_sos_.push_back(idx + mo_shift);

    // Form the maps from active relative indices to absolute SO indices
    std::map<size_t, size_t> aactvrel_to_soabs;
    std::map<size_t, size_t> bactvrel_to_soabs;
    for (size_t i = 0; i < aactv_sos_.size(); ++i)
        aactvrel_to_soabs[i] = aactv_sos_[i];
    for (size_t i = 0; i < bactv_sos_.size(); ++i)
        bactvrel_to_soabs[i] = bactv_sos_[i];

    // spin orbital indices
    core_sos_ = acore_sos_;
    actv_sos_ = aactv_sos_;
    virt_sos_ = avirt_sos_;
    core_sos_.insert(core_sos_.end(), bcore_sos_.begin(), bcore_sos_.end());
    actv_sos_.insert(actv_sos_.end(), bactv_sos_.begin(), bactv_sos_.end());
    virt_sos_.insert(virt_sos_.end(), bvirt_sos_.begin(), bvirt_sos_.end());

    // size of each spin orbital space
    nc_ = core_sos_.size();
    na_ = actv_sos_.size();
    nv_ = virt_sos_.size();
    nh_ = na_ + nc_;
    np_ = na_ + nv_;
    nso_ = nh_ + nv_;
    size_t nmo = nso_ / 2;
    size_t na_mo = na_ / 2;

    BlockedTensor::reset_mo_spaces();
    BlockedTensor::set_expert_mode(true);

    BTF_->add_mo_space("c", "mn", core_sos_, NoSpin);
    BTF_->add_mo_space("a", "uvwxyz", actv_sos_, NoSpin);
    BTF_->add_mo_space("v", "ef", virt_sos_, NoSpin);

    BTF_->add_composite_mo_space("h", "ijkl", {"c", "a"});
    BTF_->add_composite_mo_space("p", "abcd", {"a", "v"});
    BTF_->add_composite_mo_space("g", "pqrs", {"c", "a", "v"});

    // prepare one-electron integrals
    H_ = BTF_->build(tensor_type_, "H", {"gg"});
    H_.iterate([&](const std::vector<size_t>& i, const std::vector<SpinType>&, double& value) {
        if (i[0] < nmo && i[1] < nmo) {
            value = ints_->oei_a(i[0], i[1]);
        }
        if (i[0] >= nmo && i[1] >= nmo) {
            value = ints_->oei_b(i[0] - nmo, i[1] - nmo);
        }
    });

    // prepare two-electron integrals
    V_ = BTF_->build(tensor_type_, "V", {"gggg"});
    V_.iterate([&](const std::vector<size_t>& i, const std::vector<SpinType>&, double& value) {
        bool spin0 = i[0] < nmo;
        bool spin1 = i[1] < nmo;
        bool spin2 = i[2] < nmo;
        bool spin3 = i[3] < nmo;
        if (spin0 && spin1 && spin2 && spin3) {
            value = ints_->aptei_aa(i[0], i[1], i[2], i[3]);
        }
        if ((!spin0) && (!spin1) && (!spin2) && (!spin3)) {
            value = ints_->aptei_bb(i[0] - nmo, i[1] - nmo, i[2] - nmo, i[3] - nmo);
        }
        if (spin0 && (!spin1) && spin2 && (!spin3)) {
            value = ints_->aptei_ab(i[0], i[1] - nmo, i[2], i[3] - nmo);
        }
        if (spin1 && (!spin0) && spin3 && (!spin2)) {
            value = ints_->aptei_ab(i[1], i[0] - nmo, i[3], i[2] - nmo);
        }
        if (spin0 && (!spin1) && spin3 && (!spin2)) {
            value = -ints_->aptei_ab(i[0], i[1] - nmo, i[3], i[2] - nmo);
        }
        if (spin1 && (!spin0) && spin2 && (!spin3)) {
            value = -ints_->aptei_ab(i[1], i[0] - nmo, i[2], i[3] - nmo);
        }
    });

    // prepare density matrices
    D1_ = BTF_->build(tensor_type_, "D1", {"hh"});
    C1_ = BTF_->build(tensor_type_, "C1", {"pp"});
    (D1_.block("cc")).iterate([&](const std::vector<size_t>& i, double& value) {
        value = (i[0] == i[1] ? 1.0 : 0.0);
    });
    C1_.iterate([&](const std::vector<size_t>& i, const std::vector<SpinType>&, double& value) {
        value = (i[0] == i[1] ? 1.0 : 0.0);
    });
    (rdms_.g1a()).citerate([&](const std::vector<size_t>& i, const double& value) {
        size_t index = i[0] * na_ + i[1];
        (D1_.block("aa")).data()[index] = value;
        (C1_.block("aa")).data()[index] -= value;
    });
    (rdms_.g1b()).citerate([&](const std::vector<size_t>& i, const double& value) {
        size_t index = (i[0] + na_mo) * na_ + (i[1] + na_mo);
        (D1_.block("aa")).data()[index] = value;
        (C1_.block("aa")).data()[index] -= value;
    });

    auto myPow = [&](size_t x, size_t p) {
        size_t i = 1;
        for (size_t j = 1; j <= p; j++)
            i *= x;
        return i;
    };

    // prepare two-body density cumulant
    D2_ = BTF_->build(tensor_type_, "D2", {"aaaa"});
    (rdms_.g2aa()).citerate([&](const std::vector<size_t>& i, const double& value) {
        if (std::fabs(value) > 1.0e-15) {
            size_t index = 0;
            for (int m = 0; m < 4; ++m) {
                index += i[m] * myPow(na_, 3 - m);
            }
            (D2_.block("aaaa")).data()[index] = value;
        }
    });
    (rdms_.g2bb()).citerate([&](const std::vector<size_t>& i, const double& value) {
        if (std::fabs(value) > 1.0e-15) {
            size_t index = 0;
            for (int m = 0; m < 4; ++m) {
                index += (i[m] + na_mo) * myPow(na_, 3 - m);
            }
            (D2_.block("aaaa")).data()[index] = value;
        }
    });
    (rdms_.g2ab()).citerate([&](const std::vector<size_t>& i, const double& value) {
        if (std::fabs(value) > 1.0e-15) {
            size_t i0 = i[0];
            size_t i1 = i[1] + na_mo;
            size_t i2 = i[2];
            size_t i3 = i[3] + na_mo;
            size_t index = 0;
            index = i0 * myPow(na_, 3) + i1 * myPow(na_, 2) + i2 * myPow(na_, 1) + i3;
            (D2_.block("aaaa")).data()[index] = value;
            index = i1 * myPow(na_, 3) + i0 * myPow(na_, 2) + i3 * myPow(na_, 1) + i2;
            (D2_.block("aaaa")).data()[index] = value;
            index = i1 * myPow(na_, 3) + i0 * myPow(na_, 2) + i2 * myPow(na_, 1) + i3;
            (D2_.block("aaaa")).data()[index] = -value;
            index = i0 * myPow(na_, 3) + i1 * myPow(na_, 2) + i3 * myPow(na_, 1) + i2;
            (D2_.block("aaaa")).data()[index] = -value;
        }
    });

    // build Fock matrix (initial guess of one-body Hamiltonian)
    F_ = BTF_->build(tensor_type_, "Fock", {"gg"});
    F_["pq"] = H_["pq"];
    F_["pq"] += V_["pjqi"] * D1_["ij"];

    // obtain diagonal elements of Fock matrix
    Fd_ = std::vector<double>(nso_);
    F_.citerate(
        [&](const std::vector<size_t>& i, const std::vector<SpinType>&, const double& value) {
            if (i[0] == i[1]) {
                Fd_[i[0]] = value;
            }
        });

    Fc_ = BTF_->build(tensor_type_, "Fock inactive", {"gg"});
    Fc_["pq"] = H_["pq"];
    Fc_["pq"] += V_["pnqm"] * D1_["mn"];
}

void DSRG_MRPT2_SO::print_summary() {
    // Print a summary
    std::vector<std::pair<std::string, int>> calculation_info{};

    std::vector<std::pair<std::string, double>> calculation_info_double{
        {"flow parameter", s_},
        {"taylor expansion threshold", pow(10.0, -double(taylor_threshold_))}};

    std::vector<std::pair<std::string, std::string>> calculation_info_string{
        {"int_type", foptions_->get_str("INT_TYPE")}, {"source operator", source_}};

    // Print some information
    outfile->Printf("\n\n  ==> Calculation Information <==\n");
    for (auto& str_dim : calculation_info) {
        outfile->Printf("\n    %-39s %10d", str_dim.first.c_str(), str_dim.second);
    }
    for (auto& str_dim : calculation_info_double) {
        outfile->Printf("\n    %-39s %10.3e", str_dim.first.c_str(), str_dim.second);
    }
    for (auto& str_dim : calculation_info_string) {
        outfile->Printf("\n    %-39s %10s", str_dim.first.c_str(), str_dim.second.c_str());
    }
}

void DSRG_MRPT2_SO::compute_t2() {
    local_timer timer;
    std::string str = "Computing T2 amplitudes     ...";
    outfile->Printf("\n    %-35s", str.c_str());

    T2_["ijab"] = V_["ijab"];

    T2_.iterate([&](const std::vector<size_t>& i, const std::vector<SpinType>&, double& value) {
        value *= dsrg_source_->compute_renormalized_denominator(Fd_[i[0]] + Fd_[i[1]] - Fd_[i[2]] -
                                                                Fd_[i[3]]);
    });

    // zero internal amplitudes
    if (T2_.is_block("aaaa")) {
        T2_.block("aaaa").zero();
    }

    outfile->Printf("  Done. Timing %10.3f s", timer.get());
}

void DSRG_MRPT2_SO::compute_m2() {
    local_timer timer;
    std::string str = "Computing M2 integrals     ...";
    outfile->Printf("\n    %-35s", str.c_str());

    M2_["ijab"] = V_["ijab"];

    M2_.iterate([&](const std::vector<size_t>& i, const std::vector<SpinType>&, double& value) {
        value *=
            1.0 + dsrg_source_->compute_renormalized(Fd_[i[0]] + Fd_[i[1]] - Fd_[i[2]] - Fd_[i[3]]);
    });

    // zero internal amplitudes
    if (M2_.is_block("aaaa")) {
        M2_.block("aaaa").zero();
    }

    outfile->Printf("  Done. Timing %10.3f s", timer.get());
}

double DSRG_MRPT2_SO::compute_reference_energy() {
    double Eout = 0.0;

    H_.block("cc").citerate([&](const std::vector<size_t>& i, const double& value) {
        if (i[0] == i[1])
            Eout += value;
    });

    V_.block("cccc").citerate([&](const std::vector<size_t>& i, const double& value) {
        if (i[0] == i[2] and i[1] == i[3])
            Eout += 0.5 * value;
    });

    Eout += Fc_["uv"] * D1_["uv"];

    Eout += 0.25 * V_["uvxy"] * D2_["xyuv"];

    return Eout;
}

void DSRG_MRPT2_SO::compute_m2_bar() {
    Mbar2_["cdkl"] = T2_["ijab"] * D1_["ik"] * D1_["jl"] * C1_["ac"] * C1_["bd"];
}

double DSRG_MRPT2_SO::compute_correlation_energy() { return 0.25 * M2_["klcd"] * Mbar2_["cdkl"]; }

double DSRG_MRPT2_SO::compute_energy() {
    T2_ = BTF_->build(tensor_type_, "T2", {"hhpv", "hhva", "hcaa", "caaa"});
    compute_t2();

    M2_ = BTF_->build(tensor_type_, "M2", {"hhpv", "hhva", "hcaa", "caaa"});
    compute_m2();

    Mbar2_ = BTF_->build(tensor_type_, "Mbar2", {"pvhh", "vahh", "aahc", "aaca"});
    compute_m2_bar();

    // compute the reference energy
    double Eref = compute_reference_energy() + Enuc_ + Efrzc_;

    // compute the correlation energy
    double Ecorr = compute_correlation_energy();

    double Etotal = Eref + Ecorr;

    // print
    print_h2("DSRG-MRPT2 (MP2-like term only) Energy");
    outfile->Printf("\n    Nuclear repulsion energy: %20.15f", Enuc_);
    outfile->Printf("\n    Frozen-core energy:       %20.15f", Efrzc_);
    outfile->Printf("\n    Reference energy:         %20.15f", Eref);
    outfile->Printf("\n    Correlation energy:       %20.15f", Ecorr);
    outfile->Printf("\n    Total energy:             %20.15f", Etotal);

    compute_gradients();

    return Etotal;
}

void DSRG_MRPT2_SO::compute_m2_double_bar() {
    Mdbar2_["abij"] = Mbar2_["abij"];

    Mdbar2_.iterate([&](const std::vector<size_t>& i, const std::vector<SpinType>&, double& value) {
        value *=
            1.0 + dsrg_source_->compute_renormalized(Fd_[i[0]] + Fd_[i[1]] - Fd_[i[2]] - Fd_[i[3]]);
    });
}

void DSRG_MRPT2_SO::compute_t2_bar() {
    Tbar2_["ijab"] = M2_["klcd"] * D1_["ik"] * D1_["jl"] * C1_["ac"] * C1_["bd"];
}

void DSRG_MRPT2_SO::compute_t2_double_bar() {
    Tdbar2_["ijab"] = Tbar2_["ijab"];

    Tdbar2_.iterate([&](const std::vector<size_t>& i, const std::vector<SpinType>&, double& value) {
        value *= dsrg_source_->compute_renormalized_denominator(Fd_[i[0]] + Fd_[i[1]] - Fd_[i[2]] -
                                                                Fd_[i[3]]);
    });
}

void DSRG_MRPT2_SO::compute_gradients() {
    Tbar2_ = BTF_->build(tensor_type_, "Tbar2", {"hhpv", "hhva", "hcaa", "caaa"});
    compute_t2_bar();

    Tdbar2_ = BTF_->build(tensor_type_, "Tdbar2", {"hhpv", "hhva", "hcaa", "caaa"});
    compute_t2_double_bar();

    Mdbar2_ = BTF_->build(tensor_type_, "Mdbar2", {"pvhh", "vahh", "aahc", "aaca"});
    compute_m2_double_bar();

    compute_orb_grad();
}

void DSRG_MRPT2_SO::compute_orb_grad() {
    z_ = BTF_->build(tensor_type_, "z", {"cc", "aa", "vv"});
    Z_ = BTF_->build(tensor_type_, "z", {"ca", "ac", "av", "va", "cv", "vc"});

    compute_z_diag();
}

void DSRG_MRPT2_SO::compute_z_diag() {
    auto z = BTF_->build(tensor_type_, "z", {"g"}, true);

    auto V = BTF_->build(tensor_type_, "V temp", {"pphh"}, true);

    V["abij"] = V_["abij"];
    V.iterate([&](const std::vector<size_t>& i, const std::vector<SpinType>&, double& value) {
        double d = Fd_[i[0]] + Fd_[i[1]] - Fd_[i[2]] - Fd_[i[3]];
        double scale = 2 * s_ * dsrg_source_->compute_renormalized(d);
        scale -= dsrg_source_->compute_renormalized_denominator2(d);
        value *= scale;
    });

    z["i"] += V["abij"] * Tbar2_["ijab"];
    z["a"] -= V["abij"] * Tbar2_["ijab"];

    V = BTF_->build(tensor_type_, "V temp", {"hhpp"}, true);
    V["ijab"] = V_["ijab"];
    V.iterate([&](const std::vector<size_t>& i, const std::vector<SpinType>&, double& value) {
        double d = Fd_[i[0]] + Fd_[i[1]] - Fd_[i[2]] - Fd_[i[3]];
        value *= d * dsrg_source_->compute_renormalized(d);
    });

    z["i"] -= 2 * s_ * Mbar2_["abij"] * V["ijab"];
    z["a"] += 2 * s_ * Mbar2_["abij"] * V["ijab"];

    for (std::string block: {"cc", "aa", "vv"}) {
        z_.block(block).iterate([&](const std::vector<size_t>& i, double& value) {
            if (i[0] == i[1]) {
                value = z.block(block.substr(0, 1)).data()[i[0]];
            }
        });
    }
    z_.print();
}

} // namespace forte
