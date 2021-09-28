/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2021 by its authors (see COPYING, COPYING.LESSER, AUTHORS).
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
#include "mrdsrg_so.h"

using namespace psi;

namespace forte {

MRDSRG_SO::MRDSRG_SO(RDMs rdms, std::shared_ptr<SCFInfo> scf_info,
                     std::shared_ptr<ForteOptions> options, std::shared_ptr<ForteIntegrals> ints,
                     std::shared_ptr<MOSpaceInfo> mo_space_info)
    : DynamicCorrelationSolver(rdms, scf_info, options, ints, mo_space_info),
      BTF_(new BlockedTensorFactory()), tensor_type_(ambit::CoreTensor) {
    print_method_banner(
        {"SO-Based Multireference Driven Similarity Renormalization Group", "Chenyang Li"});
    startup();
    print_summary();
}

MRDSRG_SO::~MRDSRG_SO() {}

std::shared_ptr<ActiveSpaceIntegrals> MRDSRG_SO::compute_Heff_actv() {
    throw psi::PSIEXCEPTION(
        "Computing active-space Hamiltonian is not yet implemented for spin-orbital code.");

    return std::make_shared<ActiveSpaceIntegrals>(
        ints_, mo_space_info_->corr_absolute_mo("ACTIVE"), mo_space_info_->symmetry("ACTIVE"),
        mo_space_info_->corr_absolute_mo("RESTRICTED_DOCC"));

    //    // de-normal-order DSRG transformed Hamiltonian
    //    double Edsrg = Eref_ + Hbar0_;
    //    deGNO_ints("Hamiltonian", Edsrg, Hbar1_, Hbar2_);
    //    rotate_ints_semi_to_origin("Hamiltonian", Hbar1_, Hbar2_);

    //    // create FCIIntegral shared_ptr
    //    std::shared_ptr<ActiveSpaceIntegrals> fci_ints =
    //        std::make_shared<ActiveSpaceIntegrals>(ints_, actv_mos_, core_mos_);
    //    fci_ints->set_active_integrals(Hbar2_.block("aaaa"), Hbar2_.block("aAaA"),
    //                                   Hbar2_.block("AAAA"));
    //    fci_ints->set_restricted_one_body_operator(Hbar1_.block("aa").data(),
    //                                               Hbar1_.block("AA").data());
    //    fci_ints->set_scalar_energy(Edsrg - Enuc_ - Efrzc_);

    //    return fci_ints;
}

void MRDSRG_SO::startup() {
    Eref = compute_Eref_from_rdms(rdms_, ints_, mo_space_info_);

    frozen_core_energy = ints_->frozen_core_energy();

    s_ = foptions_->get_double("DSRG_S");
    if (s_ < 0) {
        outfile->Printf("\n  S parameter for DSRG must >= 0!");
        throw psi::PSIEXCEPTION("S parameter for DSRG must >= 0!");
    }
    taylor_threshold_ = foptions_->get_int("TAYLOR_THRESHOLD");
    if (taylor_threshold_ <= 0) {
        outfile->Printf("\n  Threshold for Taylor expansion must be an integer "
                        "greater than 0!");
        throw psi::PSIEXCEPTION("Threshold for Taylor expansion must be an integer "
                                "greater than 0!");
    }
    taylor_order_ = int(0.5 * (15.0 / taylor_threshold_ + 1)) + 1;

    source_ = foptions_->get_str("SOURCE");

    ntamp_ = foptions_->get_int("NTAMP");
    intruder_tamp_ = foptions_->get_double("INTRUDER_TAMP");

    // orbital spaces
    acore_sos = mo_space_info_->corr_absolute_mo("RESTRICTED_DOCC");
    aactv_sos = mo_space_info_->corr_absolute_mo("ACTIVE");
    avirt_sos = mo_space_info_->corr_absolute_mo("RESTRICTED_UOCC");

    // put all beta behind alpha
    size_t mo_shift = mo_space_info_->size("RESTRICTED_DOCC") + mo_space_info_->size("ACTIVE") +
                      mo_space_info_->size("RESTRICTED_UOCC");

    for (size_t idx : acore_sos)
        bcore_sos.push_back(idx + mo_shift);
    for (size_t idx : aactv_sos)
        bactv_sos.push_back(idx + mo_shift);
    for (size_t idx : avirt_sos)
        bvirt_sos.push_back(idx + mo_shift);

    // Form the maps from active relative indices to absolute SO indices
    std::map<size_t, size_t> aactvrel_to_soabs;
    std::map<size_t, size_t> bactvrel_to_soabs;
    for (size_t i = 0; i < aactv_sos.size(); ++i)
        aactvrel_to_soabs[i] = aactv_sos[i];
    for (size_t i = 0; i < bactv_sos.size(); ++i)
        bactvrel_to_soabs[i] = bactv_sos[i];

    // spin orbital indices
    core_sos = acore_sos;
    actv_sos = aactv_sos;
    virt_sos = avirt_sos;
    core_sos.insert(core_sos.end(), bcore_sos.begin(), bcore_sos.end());
    actv_sos.insert(actv_sos.end(), bactv_sos.begin(), bactv_sos.end());
    virt_sos.insert(virt_sos.end(), bvirt_sos.begin(), bvirt_sos.end());

    // size of each spin orbital space
    nc_ = core_sos.size();
    na_ = actv_sos.size();
    nv_ = virt_sos.size();
    nh_ = na_ + nc_;
    np_ = na_ + nv_;
    nso_ = nh_ + nv_;
    size_t nmo = nso_ / 2;
    size_t na_mo = na_ / 2;

    BTF_->add_mo_space("c", "mn", core_sos, AlphaSpin);
    BTF_->add_mo_space("a", "uvwxyz", actv_sos, AlphaSpin);
    BTF_->add_mo_space("v", "ef", virt_sos, AlphaSpin);

    BTF_->add_composite_mo_space("h", "ijkl", {"c", "a"});
    BTF_->add_composite_mo_space("p", "abcd", {"a", "v"});
    BTF_->add_composite_mo_space("g", "pqrsto", {"c", "a", "v"});

    // prepare one-electron integrals
    H = BTF_->build(tensor_type_, "H", {"gg"});
    H.iterate([&](const std::vector<size_t>& i, const std::vector<SpinType>&, double& value) {
        if (i[0] < nmo && i[1] < nmo) {
            value = ints_->oei_a(i[0], i[1]);
        }
        if (i[0] >= nmo && i[1] >= nmo) {
            value = ints_->oei_b(i[0] - nmo, i[1] - nmo);
        }
    });

    // prepare two-electron integrals
    V = BTF_->build(tensor_type_, "V", {"gggg"});
    V.iterate([&](const std::vector<size_t>& i, const std::vector<SpinType>&, double& value) {
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
    Gamma1 = BTF_->build(tensor_type_, "Gamma1", {"hh"});
    Eta1 = BTF_->build(tensor_type_, "Eta1", {"pp"});
    (Gamma1.block("cc")).iterate([&](const std::vector<size_t>& i, double& value) {
        value = (i[0] == i[1] ? 1.0 : 0.0);
    });
    Eta1.iterate([&](const std::vector<size_t>& i, const std::vector<SpinType>&, double& value) {
        value = (i[0] == i[1] ? 1.0 : 0.0);
    });
    (rdms_.g1a()).citerate([&](const std::vector<size_t>& i, const double& value) {
        size_t index = i[0] * na_ + i[1];
        (Gamma1.block("aa")).data()[index] = value;
        (Eta1.block("aa")).data()[index] -= value;
    });
    (rdms_.g1b()).citerate([&](const std::vector<size_t>& i, const double& value) {
        size_t index = (i[0] + na_mo) * na_ + (i[1] + na_mo);
        (Gamma1.block("aa")).data()[index] = value;
        (Eta1.block("aa")).data()[index] -= value;
    });

    // prepare two-body density cumulant
    Lambda2 = BTF_->build(tensor_type_, "Lambda2", {"aaaa"});
    (rdms_.L2aa()).citerate([&](const std::vector<size_t>& i, const double& value) {
        if (std::fabs(value) > 1.0e-15) {
            size_t index = 0;
            for (int m = 0; m < 4; ++m) {
                index += i[m] * myPow(na_, 3 - m);
            }
            (Lambda2.block("aaaa")).data()[index] = value;
        }
    });
    (rdms_.L2bb()).citerate([&](const std::vector<size_t>& i, const double& value) {
        if (std::fabs(value) > 1.0e-15) {
            size_t index = 0;
            for (int m = 0; m < 4; ++m) {
                index += (i[m] + na_mo) * myPow(na_, 3 - m);
            }
            (Lambda2.block("aaaa")).data()[index] = value;
        }
    });
    (rdms_.L2ab()).citerate([&](const std::vector<size_t>& i, const double& value) {
        if (std::fabs(value) > 1.0e-15) {
            size_t i0 = i[0];
            size_t i1 = i[1] + na_mo;
            size_t i2 = i[2];
            size_t i3 = i[3] + na_mo;
            size_t index = 0;
            index = i0 * myPow(na_, 3) + i1 * myPow(na_, 2) + i2 * myPow(na_, 1) + i3;
            (Lambda2.block("aaaa")).data()[index] = value;
            index = i1 * myPow(na_, 3) + i0 * myPow(na_, 2) + i3 * myPow(na_, 1) + i2;
            (Lambda2.block("aaaa")).data()[index] = value;
            index = i1 * myPow(na_, 3) + i0 * myPow(na_, 2) + i2 * myPow(na_, 1) + i3;
            (Lambda2.block("aaaa")).data()[index] = -value;
            index = i0 * myPow(na_, 3) + i1 * myPow(na_, 2) + i3 * myPow(na_, 1) + i2;
            (Lambda2.block("aaaa")).data()[index] = -value;
        }
    });
    outfile->Printf("\n    Norm of L2: %12.8f.", Lambda2.norm());

    // prepare three-body density cumulant
    if (foptions_->get_str("THREEPDC") != "ZERO") {
        Lambda3 = BTF_->build(tensor_type_, "Lambda3", {"aaaaaa"});
        (rdms_.L3aaa()).citerate([&](const std::vector<size_t>& i, const double& value) {
            if (std::fabs(value) > 1.0e-15) {
                size_t index = 0;
                for (int m = 0; m < 6; ++m) {
                    index += i[m] * myPow(na_, 5 - m);
                }
                (Lambda3.block("aaaaaa")).data()[index] = value;
            }
        });
        (rdms_.L3bbb()).citerate([&](const std::vector<size_t>& i, const double& value) {
            if (std::fabs(value) > 1.0e-15) {
                size_t index = 0;
                for (int m = 0; m < 6; ++m) {
                    index += (i[m] + na_mo) * myPow(na_, 5 - m);
                }
                (Lambda3.block("aaaaaa")).data()[index] = value;
            }
        });
        (rdms_.L3aab()).citerate([&](const std::vector<size_t>& i, const double& value) {
            if (std::fabs(value) > 1.0e-15) {
                // original: a[0]a[1]b[2]; permutation: a[0]b[2]a[1] (-1),
                // b[2]a[0]a[1] (+1)
                std::vector<size_t> upper(3);
                std::vector<std::vector<size_t>> uppers;
                upper[0] = i[0];
                upper[1] = i[1];
                upper[2] = i[2] + na_mo;
                uppers.push_back(upper);
                upper[0] = i[0];
                upper[2] = i[1];
                upper[1] = i[2] + na_mo;
                uppers.push_back(upper);
                upper[1] = i[0];
                upper[2] = i[1];
                upper[0] = i[2] + na_mo;
                uppers.push_back(upper);
                std::vector<size_t> lower(3);
                std::vector<std::vector<size_t>> lowers;
                lower[0] = i[3];
                lower[1] = i[4];
                lower[2] = i[5] + na_mo;
                lowers.push_back(lower);
                lower[0] = i[3];
                lower[2] = i[4];
                lower[1] = i[5] + na_mo;
                lowers.push_back(lower);
                lower[1] = i[3];
                lower[2] = i[4];
                lower[0] = i[5] + na_mo;
                lowers.push_back(lower);

                for (int m = 0; m < 3; ++m) {
                    std::vector<size_t> u = uppers[m];
                    size_t iu = 0;
                    for (int mi = 0; mi < 3; ++mi)
                        iu += u[mi] * myPow(na_, 5 - mi);
                    for (int n = 0; n < 3; ++n) {
                        std::vector<size_t> l = lowers[n];
                        size_t index = iu;
                        for (int ni = 0; ni < 3; ++ni)
                            index += l[ni] * myPow(na_, 2 - ni);
                        (Lambda3.block("aaaaaa")).data()[index] = value * pow(-1.0, m + n);
                    }
                }
            }
        });
        (rdms_.L3abb()).citerate([&](const std::vector<size_t>& i, const double& value) {
            if (std::fabs(value) > 1.0e-15) {
                // original: a[0]b[1]b[2]; permutation: b[1]a[0]b[2] (-1),
                // b[1]b[2]a[0] (+1)
                std::vector<size_t> upper(3);
                std::vector<std::vector<size_t>> uppers;
                upper[0] = i[0];
                upper[1] = i[1] + na_mo;
                upper[2] = i[2] + na_mo;
                uppers.push_back(upper);
                upper[1] = i[0];
                upper[0] = i[1] + na_mo;
                upper[2] = i[2] + na_mo;
                uppers.push_back(upper);
                upper[2] = i[0];
                upper[0] = i[1] + na_mo;
                upper[1] = i[2] + na_mo;
                uppers.push_back(upper);
                std::vector<size_t> lower(3);
                std::vector<std::vector<size_t>> lowers;
                lower[0] = i[3];
                lower[1] = i[4] + na_mo;
                lower[2] = i[5] + na_mo;
                lowers.push_back(lower);
                lower[1] = i[3];
                lower[0] = i[4] + na_mo;
                lower[2] = i[5] + na_mo;
                lowers.push_back(lower);
                lower[2] = i[3];
                lower[0] = i[4] + na_mo;
                lower[1] = i[5] + na_mo;
                lowers.push_back(lower);

                for (int m = 0; m < 3; ++m) {
                    std::vector<size_t> u = uppers[m];
                    size_t iu = 0;
                    for (int mi = 0; mi < 3; ++mi)
                        iu += u[mi] * myPow(na_, 5 - mi);
                    for (int n = 0; n < 3; ++n) {
                        std::vector<size_t> l = lowers[n];
                        size_t index = iu;
                        for (int ni = 0; ni < 3; ++ni)
                            index += l[ni] * myPow(na_, 2 - ni);
                        (Lambda3.block("aaaaaa")).data()[index] = value * pow(-1.0, m + n);
                    }
                }
            }
        });
        outfile->Printf("\n    Norm of L3: %12.8f.", Lambda3.norm());
    }

    // build Fock matrix (initial guess of one-body Hamiltonian)
    F = BTF_->build(tensor_type_, "Fock", {"gg"});
    F["pq"] = H["pq"];
    F["pq"] += V["pjqi"] * Gamma1["ij"];

    // obtain diagonal elements of Fock matrix
    Fd = std::vector<double>(nso_);
    F.citerate(
        [&](const std::vector<size_t>& i, const std::vector<SpinType>&, const double& value) {
            if (i[0] == i[1]) {
                Fd[i[0]] = value;
            }
        });
}

void MRDSRG_SO::print_summary() {
    // Print a summary
    std::vector<std::pair<std::string, int>> calculation_info{{"ntamp", ntamp_}};

    std::vector<std::pair<std::string, double>> calculation_info_double{
        {"flow parameter", s_},
        {"taylor expansion threshold", pow(10.0, -double(taylor_threshold_))},
        {"intruder_tamp", intruder_tamp_}};

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

void MRDSRG_SO::guess_t2() {
    local_timer timer;
    std::string str = "Computing T2 amplitudes     ...";
    outfile->Printf("\n    %-35s", str.c_str());

    T2["ijab"] = V["ijab"];

    T2.iterate([&](const std::vector<size_t>& i, const std::vector<SpinType>&, double& value) {
        value *= renormalized_denominator(Fd[i[0]] + Fd[i[1]] - Fd[i[2]] - Fd[i[3]]);
    });

    // zero internal amplitudes
    T2.block("aaaa").zero();

    // norm and max
    T2max = 0.0, T2norm = T2.norm();
    T2.citerate([&](const std::vector<size_t>&, const std::vector<SpinType>&, const double& value) {
        if (std::fabs(value) > std::fabs(T2max))
            T2max = value;
    });

    outfile->Printf("  Done. Timing %10.3f s", timer.get());
}

void MRDSRG_SO::guess_t1() {
    local_timer timer;
    std::string str = "Computing T1 amplitudes     ...";
    outfile->Printf("\n    %-35s", str.c_str());

    //    BlockedTensor temp = BTF->build(tensor_type_,"temp",{"aa"});
    //    temp["xu"] = Gamma1["xu"];
    //    temp.iterate([&](const std::vector<size_t>& i, const
    //    std::vector<SpinType>& spin, double& value){
    //        value *= Fd[i[0]] - Fd[i[1]];
    //    });

    T1["ia"] = F["ia"];
    //    T1["ia"] += temp["xu"] * T2["iuax"];
    T1.iterate([&](const std::vector<size_t>& i, const std::vector<SpinType>&, double& value) {
        value *= renormalized_denominator(Fd[i[0]] - Fd[i[1]]);
    });

    // zero internal amplitudes
    T1.block("aa").zero();

    // norm and max
    T1max = 0.0, T1norm = T1.norm();
    T1.citerate([&](const std::vector<size_t>&, const std::vector<SpinType>&, const double& value) {
        if (std::fabs(value) > std::fabs(T1max))
            T1max = value;
    });

    outfile->Printf("  Done. Timing %10.3f s", timer.get());
}

double MRDSRG_SO::renormalized_denominator(double D) {
    double Z = std::sqrt(s_) * D;
    if (std::fabs(Z) < std::pow(0.1, taylor_threshold_)) {
        return Taylor_Exp(Z, taylor_order_) * std::sqrt(s_);
    } else {
        return (1.0 - std::exp(-s_ * std::pow(D, 2.0))) / D;
    }
}

void MRDSRG_SO::update_t2() {
    BlockedTensor R2 = ambit::BlockedTensor::build(tensor_type_, "R2", {"hhpp"});
    R2["ijab"] = T2["ijab"];
    R2.iterate([&](const std::vector<size_t>& i, const std::vector<SpinType>&, double& value) {
        value *= (Fd[i[0]] + Fd[i[1]] - Fd[i[2]] - Fd[i[3]]);
    });
    R2["ijab"] += Hbar2["ijab"];
    R2.iterate([&](const std::vector<size_t>& i, const std::vector<SpinType>&, double& value) {
        value *= renormalized_denominator(Fd[i[0]] + Fd[i[1]] - Fd[i[2]] - Fd[i[3]]);
    });

    // zero internal amplitudes
    R2.block("aaaa").zero();

    BlockedTensor D2 = ambit::BlockedTensor::build(tensor_type_, "DT2", {"hhpp"});
    D2["ijab"] = R2["ijab"] - T2["ijab"];
    rms_t2 = D2.norm();

    T2["ijab"] = R2["ijab"];

    // norm and max
    T2max = 0.0, T2norm = T2.norm();
    T2.citerate([&](const std::vector<size_t>&, const std::vector<SpinType>&, const double& value) {
        if (std::fabs(value) > std::fabs(T2max))
            T2max = value;
    });
}

void MRDSRG_SO::update_t1() {
    BlockedTensor R1 = ambit::BlockedTensor::build(tensor_type_, "R1", {"hp"});
    R1["ia"] = T1["ia"];
    R1.iterate([&](const std::vector<size_t>& i, const std::vector<SpinType>&, double& value) {
        value *= (Fd[i[0]] - Fd[i[1]]);
    });
    R1["ia"] += Hbar1["ia"];
    R1.iterate([&](const std::vector<size_t>& i, const std::vector<SpinType>&, double& value) {
        value *= renormalized_denominator(Fd[i[0]] - Fd[i[1]]);
    });

    // zero internal amplitudes
    R1.block("aa").zero();

    BlockedTensor D1 = ambit::BlockedTensor::build(tensor_type_, "DT1", {"hp"});
    D1["ia"] = R1["ia"] - T1["ia"];
    rms_t1 = D1.norm();

    T1["ia"] = R1["ia"];

    // norm and max
    T1max = 0.0, T1norm = T1.norm();
    T1.citerate([&](const std::vector<size_t>&, const std::vector<SpinType>&, const double& value) {
        if (std::fabs(value) > std::fabs(T1max))
            T1max = value;
    });
}

double MRDSRG_SO::compute_energy() {

    // copy initial one-body Hamiltonian to Hbar1
    Hbar1 = BTF_->build(tensor_type_, "Hbar1", {"gg"});
    Hbar1["pq"] = F["pq"];

    // copy initial two-body Hamiltonian to Hbar2
    Hbar2 = BTF_->build(tensor_type_, "Hbar2", {"gggg"});
    Hbar2["pqrs"] = V["pqrs"];

    // build initial amplitudes
    outfile->Printf("\n\n  ==> Build Initial Amplitude from DSRG-MRPT2 <==\n");
    T1 = BTF_->build(tensor_type_, "T1 Amplitudes", {"hp"});
    T2 = BTF_->build(tensor_type_, "T2 Amplitudes", {"hhpp"});
    guess_t2();
    guess_t1();

    // iteration variables
    double Etotal = Eref;
    bool converged = false;
    int cycle = 0;

    // start iteration
    outfile->Printf("\n\n  ==> Start Iterations <==\n");
    outfile->Printf("\n    "
                    "----------------------------------------------------------"
                    "----------------------------------------");
    outfile->Printf("\n           Cycle     Energy (a.u.)     Delta(E)  "
                    "|Hbar1|_N  |Hbar2|_N    |T1|    |T2|  max(T1) max(T2)");
    outfile->Printf("\n    "
                    "----------------------------------------------------------"
                    "----------------------------------------");

    do {
        // compute hbar
        if (foptions_->get_str("CORR_LEVEL") == "QDSRG2") {
            compute_qhbar();
        } else {
            // single-commutator by default
            compute_hbar();
        }
        double Edelta = Eref + Hbar0 - Etotal;
        Etotal = Eref + Hbar0;

        // norm of non-diagonal Hbar
        double Hbar1Nnorm = 0.0, Hbar2Nnorm = 0.0;
        for (const std::string& block : Hbar1.block_labels()) {
            bool idx0 = block[0] == 'c' || block[0] == 'a';
            bool idx1 = block[1] == 'a' || block[0] == 'v';
            if (idx0 && idx1) {
                if (block != "aa") {
                    Hbar1Nnorm += 2 * pow(Hbar1.block(block).norm(), 2.0);
                } else {
                    Hbar1Nnorm += pow(Hbar1.block(block).norm(), 2.0);
                }
            }
        }
        Hbar1Nnorm = sqrt(Hbar1Nnorm);

        for (const std::string& block : Hbar2.block_labels()) {
            bool idx0 = block[0] == 'c' || block[0] == 'a';
            bool idx1 = block[0] == 'c' || block[0] == 'a';
            bool idx2 = block[1] == 'a' || block[0] == 'v';
            bool idx3 = block[1] == 'a' || block[0] == 'v';
            if (idx0 && idx1 && idx2 && idx3) {
                if (block != "aaaa") {
                    Hbar2Nnorm += 2 * pow(Hbar2.block(block).norm(), 2.0);
                } else {
                    Hbar2Nnorm += pow(Hbar2.block(block).norm(), 2.0);
                }
            }
        }
        Hbar2Nnorm = sqrt(Hbar2Nnorm);

        outfile->Printf("\n      @CT %4d %20.12f %11.3e %10.3e %10.3e %7.4f "
                        "%7.4f %7.4f %7.4f",
                        cycle, Etotal, Edelta, Hbar1Nnorm, Hbar2Nnorm, T1norm, T2norm, T1max,
                        T2max);

        // update amplitudes
        update_t2();
        update_t1();

        // test convergence
        double rms = rms_t1 > rms_t2 ? rms_t1 : rms_t2;
        if (std::fabs(Edelta) < foptions_->get_double("E_CONVERGENCE") &&
            rms < foptions_->get_double("R_CONVERGENCE")) {
            converged = true;
        }
        if (cycle > foptions_->get_int("MAXITER")) {
            outfile->Printf("\n\n\tThe calculation did not converge in %d "
                            "cycles\n\tQuitting.\n",
                            foptions_->get_int("MAXITER"));
            converged = true;
        }

        ++cycle;
    } while (!converged);

    outfile->Printf("\n    "
                    "----------------------------------------------------------"
                    "----------------------------------------");
    outfile->Printf("\n\n\n    MR-DSRG(2) correlation energy      = %25.15f", Etotal - Eref);
    outfile->Printf("\n  * MR-DSRG(2) total energy            = %25.15f\n", Etotal);

    psi::Process::environment.globals["CURRENT ENERGY"] = Etotal;

    return Etotal;
}

void MRDSRG_SO::compute_hbar() {

    //    outfile->Printf("\n\n  Computing the similarity-transformed
    //    Hamiltonian");
    //    outfile->Printf("\n
    //    -----------------------------------------------------------------");
    //    outfile->Printf("\n  nComm           C0                 |C1| |C2|" );
    //    outfile->Printf("\n
    //    -----------------------------------------------------------------");

    // copy initial one-body Hamiltonian
    Hbar0 = 0.0;
    Hbar1["pq"] = F["pq"];
    Hbar2["pqrs"] = V["pqrs"];

    BlockedTensor O1 = ambit::BlockedTensor::build(tensor_type_, "O1", {"gg"});
    BlockedTensor O2 = ambit::BlockedTensor::build(tensor_type_, "O2", {"gggg"});
    O1["pq"] = F["pq"];
    O2["pqrs"] = V["pqrs"];

    //    outfile->Printf("\n  %2d %20.12f %20e
    //    %20e",0,Hbar0,Hbar1.norm(),Hbar2.norm());

    // iterator variables
    int maxn = foptions_->get_int("DSRG_RSC_NCOMM");
    double ct_threshold = foptions_->get_double("DSRG_RSC_THRESHOLD");
    BlockedTensor C1 = ambit::BlockedTensor::build(tensor_type_, "C1", {"gg"});
    BlockedTensor C2 = ambit::BlockedTensor::build(tensor_type_, "C2", {"gggg"});

    // compute Hbar recursively
    for (int n = 1; n <= maxn; ++n) {
        // prefactor before n-nested commutator
        double factor = 1.0 / n;

        // Compute the commutator C = 1/n [O,T]
        double C0 = 0.0;
        C1.zero();
        C2.zero();

        // zero-body
        H1_T1_C0(O1, T1, factor, C0);
        H1_T2_C0(O1, T2, factor, C0);
        H2_T1_C0(O2, T1, factor, C0);
        H2_T2_C0(O2, T2, factor, C0);

        // one-body
        H1_T1_C1(O1, T1, factor, C1);
        H1_T2_C1(O1, T2, factor, C1);
        H2_T1_C1(O2, T1, factor, C1);
        H2_T2_C1(O2, T2, factor, C1);

        // two-body
        H1_T2_C2(O1, T2, factor, C2);
        H2_T1_C2(O2, T1, factor, C2);
        H2_T2_C2(O2, T2, factor, C2);

        //        outfile->Printf("\n   H0  = %20.12f", C0);
        //        outfile->Printf("\n  |H1| = %20.12f", C1.norm(1));
        //        outfile->Printf("\n  |H2| = %20.12f", C2.norm(1));
        //        outfile->Printf("\n  --------------------------------");

        // [H, A] = [H, T] + [H, T]^dagger
        C0 *= 2.0;
        O1["pq"] = C1["pq"];
        C1["pq"] += O1["qp"];
        O2["pqrs"] = C2["pqrs"];
        C2["pqrs"] += O2["rspq"];

        // Hbar += C
        Hbar0 += C0;
        Hbar1["pq"] += C1["pq"];
        Hbar2["pqrs"] += C2["pqrs"];

        // copy C to O for next level commutator
        O1["pq"] = C1["pq"];
        O2["pqrs"] = C2["pqrs"];

        // test convergence of C
        double norm_C1 = C1.norm();
        double norm_C2 = C2.norm();
        //        outfile->Printf("\n  %2d %20.12f %20e
        //        %20e",n,C0,norm_C1,norm_C2);
        if (std::sqrt(norm_C2 * norm_C2 + norm_C1 * norm_C1) < ct_threshold) {
            break;
        }
    }
    //    outfile->Printf("\n
    //    -----------------------------------------------------------------");
}

void MRDSRG_SO::compute_qhbar() {

    //    outfile->Printf("\n\n  Computing the similarity-transformed
    //    Hamiltonian");
    //    outfile->Printf("\n
    //    -----------------------------------------------------------------");
    //    outfile->Printf("\n  nComm           C0                 |C1| |C2|" );
    //    outfile->Printf("\n
    //    -----------------------------------------------------------------");

    // copy initial one-body Hamiltonian
    Hbar0 = 0.0;
    Hbar1["pq"] = F["pq"];
    Hbar2["pqrs"] = V["pqrs"];

    BlockedTensor O1 = ambit::BlockedTensor::build(tensor_type_, "O1", {"gg"});
    BlockedTensor O2 = ambit::BlockedTensor::build(tensor_type_, "O2", {"gggg"});
    BlockedTensor O3 = ambit::BlockedTensor::build(tensor_type_, "O3", {"gggggg"});
    O1["pq"] = F["pq"];
    O2["pqrs"] = V["pqrs"];

    //    outfile->Printf("\n  %2d %20.12f %20e
    //    %20e",0,Hbar0,Hbar1.norm(),Hbar2.norm());

    // iterator variables
    int maxn = foptions_->get_int("DSRG_RSC_NCOMM");
    double ct_threshold = foptions_->get_double("DSRG_RSC_THRESHOLD");
    BlockedTensor C1 = ambit::BlockedTensor::build(tensor_type_, "C1", {"gg"});
    BlockedTensor C2 = ambit::BlockedTensor::build(tensor_type_, "C2", {"gggg"});
    BlockedTensor C3 = ambit::BlockedTensor::build(tensor_type_, "C3", {"gggggg"});

    // compute Hbar recursively
    for (int n = 1; n <= maxn; ++n) {
        // prefactor before n-nested commutator
        double factor = 1.0 / n;

        // Compute the commutator C = 1/n [O,T]
        double C0 = 0.0;
        C1.zero();
        C2.zero();

        // zero-body
        H1_T1_C0(O1, T1, factor, C0);
        H1_T2_C0(O1, T2, factor, C0);
        H2_T1_C0(O2, T1, factor, C0);
        H2_T2_C0(O2, T2, factor, C0);

        // one-body
        H1_T1_C1(O1, T1, factor, C1);
        H1_T2_C1(O1, T2, factor, C1);
        H2_T1_C1(O2, T1, factor, C1);
        H2_T2_C1(O2, T2, factor, C1);

        // two-body
        H1_T2_C2(O1, T2, factor, C2);
        H2_T1_C2(O2, T1, factor, C2);
        H2_T2_C2(O2, T2, factor, C2);

        // three-body if odd
        if (n % 2 == 1) {
            C3.zero();
            H2_T2_C3(O2, T2, factor, C3);
            O3["pqrsto"] = C3["pqrsto"];
            O3["pqrsto"] += C3["stopqr"];
        }

        // compute three-body contrinution if even
        if (n % 2 == 0) {
            H3_T1_C1(O3, T1, factor, C1);
            H3_T1_C2(O3, T1, factor, C2);
            H3_T2_C1(O3, T2, factor, C1);
            H3_T2_C2(O3, T2, factor, C2);
        }

        //        outfile->Printf("\n   H0  = %20.12f", C0);
        //        outfile->Printf("\n  |H1| = %20.12f", C1.norm(1));
        //        outfile->Printf("\n  |H2| = %20.12f", C2.norm(1));
        //        outfile->Printf("\n  --------------------------------");

        // [H, A] = [H, T] + [H, T]^dagger
        C0 *= 2.0;
        O1["pq"] = C1["pq"];
        C1["pq"] += O1["qp"];
        O2["pqrs"] = C2["pqrs"];
        C2["pqrs"] += O2["rspq"];

        // Hbar += C
        Hbar0 += C0;
        Hbar1["pq"] += C1["pq"];
        Hbar2["pqrs"] += C2["pqrs"];

        // copy C to O for next level commutator
        O1["pq"] = C1["pq"];
        O2["pqrs"] = C2["pqrs"];

        // test convergence of C
        double norm_C1 = C1.norm();
        double norm_C2 = C2.norm();
        //        outfile->Printf("\n  %2d %20.12f %20e
        //        %20e",n,C0,norm_C1,norm_C2);
        if (std::sqrt(norm_C2 * norm_C2 + norm_C1 * norm_C1) < ct_threshold) {
            break;
        }
    }
    //    outfile->Printf("\n
    //    -----------------------------------------------------------------");
}

void MRDSRG_SO::H1_T1_C0(BlockedTensor& H1, BlockedTensor& T1, const double& alpha, double& C0) {
    //    local_timer timer;
    //    std::string str = "Computing [Hbar1, T1] -> C0 ...";
    //    outfile->Printf("\n    %-35s", str.c_str());

    double E = 0.0;
    BlockedTensor temp;
    temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"hp"});
    temp["jb"] = T1["ia"] * Eta1["ab"] * Gamma1["ji"];
    E += temp["jb"] * H1["bj"];
    E *= alpha;
    C0 += E;
    //    outfile->Printf("  Done. Timing %10.3f s; Energy = %14.10f Eh",
    //    timer.get(), E);
}

void MRDSRG_SO::H2_T1_C0(BlockedTensor& H2, BlockedTensor& T1, const double& alpha, double& C0) {
    //    local_timer timer;
    //    std::string str = "Computing [Hbar2, T1] -> C0 ...";
    //    outfile->Printf("\n    %-35s", str.c_str());

    double E = 0.0;
    BlockedTensor temp;
    temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"aaaa"});
    temp["uvxy"] += H2["evxy"] * T1["ue"];
    temp["uvxy"] -= H2["uvmy"] * T1["mx"];
    E += 0.5 * temp["uvxy"] * Lambda2["xyuv"];
    E *= alpha;
    C0 += E;
    //    outfile->Printf("  Done. Timing %10.3f s; Energy = %14.10f Eh",
    //    timer.get(), E);
}

void MRDSRG_SO::H1_T2_C0(BlockedTensor& H1, BlockedTensor& T2, const double& alpha, double& C0) {
    //    local_timer timer;
    //    std::string str = "Computing [Hbar1, T2] -> C0 ...";
    //    outfile->Printf("\n    %-35s", str.c_str());

    double E = 0.0;
    BlockedTensor temp;
    temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"aaaa"});
    temp["uvxy"] += H1["ex"] * T2["uvey"];
    temp["uvxy"] -= H1["vm"] * T2["umxy"];
    E += 0.5 * temp["uvxy"] * Lambda2["xyuv"];
    E *= alpha;
    C0 += E;
    //    outfile->Printf("  Done. Timing %10.3f s; Energy = %14.10f Eh",
    //    timer.get(), E);
}

void MRDSRG_SO::H2_T2_C0(BlockedTensor& H2, BlockedTensor& T2, const double& alpha, double& C0) {
    //    local_timer timer;
    //    std::string str = "Computing [Hbar2, T2] -> C0 ...";
    //    outfile->Printf("\n    %-35s", str.c_str());

    // <[Hbar2, T2]> (C_2)^4
    double E = 0.25 * H2["efmn"] * T2["mnef"];

    BlockedTensor temp1 = ambit::BlockedTensor::build(tensor_type_, "temp1", {"hhaa"});
    BlockedTensor temp2 = ambit::BlockedTensor::build(tensor_type_, "temp2", {"hhaa"});
    temp1["klux"] = T2["ijux"] * Gamma1["ki"] * Gamma1["lj"];
    temp2["klvy"] = temp1["klux"] * Eta1["uv"] * Eta1["xy"];
    E += 0.25 * H2["vykl"] * temp2["klvy"];

    temp1 = ambit::BlockedTensor::build(tensor_type_, "temp1", {"hhav"});
    temp2 = ambit::BlockedTensor::build(tensor_type_, "temp2", {"hhav"});
    temp1["klue"] = T2["ijue"] * Gamma1["ki"] * Gamma1["lj"];
    temp2["klve"] = temp1["klue"] * Eta1["uv"];
    E += 0.5 * H2["vekl"] * temp2["klve"];

    temp2 = ambit::BlockedTensor::build(tensor_type_, "temp2", {"aaaa"});
    temp2["yvxu"] -= H2["fexu"] * T2["yvef"];
    E += 0.25 * temp2["yvxu"] * Gamma1["xy"] * Gamma1["uv"];

    temp2 = ambit::BlockedTensor::build(tensor_type_, "temp2", {"aa"});
    temp2["vu"] -= H2["femu"] * T2["mvef"];
    E += 0.5 * temp2["vu"] * Gamma1["uv"];

    // <[Hbar2, T2]> C_4 (C_2)^2 HH
    temp1 = ambit::BlockedTensor::build(tensor_type_, "temp1", {"aahh"});
    temp2 = ambit::BlockedTensor::build(tensor_type_, "temp2", {"aaaa"});
    temp1["uvij"] = H2["uvkl"] * Gamma1["ki"] * Gamma1["lj"];
    temp2["uvxy"] += 0.125 * temp1["uvij"] * T2["ijxy"];

    // <[Hbar2, T2]> C_4 (C_2)^2 PP
    temp1 = ambit::BlockedTensor::build(tensor_type_, "temp1", {"aapp"});
    temp1["uvcd"] = T2["uvab"] * Eta1["ac"] * Eta1["bd"];
    temp2["uvxy"] += 0.125 * temp1["uvcd"] * H2["cdxy"];

    // <[Hbar2, T2]> C_4 (C_2)^2 PH
    temp1 = ambit::BlockedTensor::build(tensor_type_, "temp1", {"hapa"});
    temp1["juby"] = T2["iuay"] * Gamma1["ji"] * Eta1["ab"];
    temp2["uvxy"] += H2["vbjx"] * temp1["juby"];
    E += temp2["uvxy"] * Lambda2["xyuv"];

    // <[Hbar2, T2]> C_6 C_2
    if (foptions_->get_str("THREEPDC") != "ZERO") {
        temp1 = ambit::BlockedTensor::build(tensor_type_, "temp", {"aaaaaa"});
        temp1["uvwxyz"] += H2["uviz"] * T2["iwxy"];
        temp1["uvwxyz"] += H2["waxy"] * T2["uvaz"];
        E += 0.25 * temp1["uvwxyz"] * Lambda3["xyzuvw"];
    }

    E *= alpha;
    C0 += E;
    //    outfile->Printf("  Done. Timing %10.3f s; Energy = %14.10f Eh",
    //    timer.get(), E);
}

void MRDSRG_SO::H1_T1_C1(BlockedTensor& H1, BlockedTensor& T1, const double& alpha,
                         BlockedTensor& C1) {
    //    local_timer timer;
    //    std::string str = "Computing [Hbar1, T1] -> C1 ...";
    //    outfile->Printf("\n    %-35s", str.c_str());

    C1["ip"] += alpha * H1["ap"] * T1["ia"];
    C1["pa"] -= alpha * H1["pi"] * T1["ia"];

    //    outfile->Printf("  Done. Timing %10.3f s", timer.get());
}

void MRDSRG_SO::H2_T1_C1(BlockedTensor& H2, BlockedTensor& T1, const double& alpha,
                         BlockedTensor& C1) {
    //    local_timer timer;
    //    std::string str = "Computing [Hbar2, T1] -> C1 ...";
    //    outfile->Printf("\n    %-35s", str.c_str());

    C1["qp"] += alpha * T1["ia"] * H2["qapj"] * Gamma1["ji"];
    C1["qp"] -= alpha * T1["mu"] * H2["qvpm"] * Gamma1["uv"];

    //    outfile->Printf("  Done. Timing %10.3f s", timer.get());
}

void MRDSRG_SO::H1_T2_C1(BlockedTensor& H1, BlockedTensor& T2, const double& alpha,
                         BlockedTensor& C1) {
    //    local_timer timer;
    //    std::string str = "Computing [Hbar1, T2] -> C1 ...";
    //    outfile->Printf("\n    %-35s", str.c_str());

    C1["ia"] += alpha * T2["ijab"] * H1["bk"] * Gamma1["kj"];
    C1["ia"] -= alpha * T2["ijau"] * H1["vj"] * Gamma1["uv"];

    //    outfile->Printf("  Done. Timing %10.3f s", timer.get());
}

void MRDSRG_SO::H2_T2_C1(BlockedTensor& H2, BlockedTensor& T2, const double& alpha,
                         BlockedTensor& C1) {
    //    local_timer timer;
    //    std::string str = "Computing [Hbar2, T2] -> C1 ...";
    //    outfile->Printf("\n    %-35s", str.c_str());

    // [Hbar2, T2] (C_2)^3 -> C1
    BlockedTensor temp1 = ambit::BlockedTensor::build(tensor_type_, "temp1", {"hhgh"});
    temp1["ijrk"] = H2["abrk"] * T2["ijab"];
    C1["ir"] += 0.5 * alpha * temp1["ijrk"] * Gamma1["kj"];

    temp1 = ambit::BlockedTensor::build(tensor_type_, "temp1", {"hhaa"});
    temp1["ijvy"] = T2["ijux"] * Gamma1["uv"] * Gamma1["xy"];
    C1["ir"] += 0.5 * alpha * temp1["ijvy"] * H2["vyrj"];

    temp1 = ambit::BlockedTensor::build(tensor_type_, "temp1", {"hhap"});
    temp1["ikvb"] = T2["ijub"] * Gamma1["kj"] * Gamma1["uv"];
    C1["ir"] -= alpha * temp1["ikvb"] * H2["vbrk"];

    temp1 = ambit::BlockedTensor::build(tensor_type_, "temp1", {"hhpa"});
    temp1["ijav"] = T2["ijau"] * Gamma1["uv"];
    C1["pa"] -= 0.5 * alpha * temp1["ijav"] * H2["pvij"];

    temp1 = ambit::BlockedTensor::build(tensor_type_, "temp1", {"hhpp"});
    temp1["klab"] = T2["ijab"] * Gamma1["ki"] * Gamma1["lj"];
    C1["pa"] -= 0.5 * alpha * temp1["klab"] * H2["pbkl"];

    temp1 = ambit::BlockedTensor::build(tensor_type_, "temp1", {"hhpa"});
    temp1["ikav"] = T2["ijau"] * Gamma1["uv"] * Gamma1["kj"];
    C1["pa"] += alpha * temp1["ikav"] * H2["pvik"];

    // [Hbar2, T2] C_4 C_2 2:2 -> C1
    temp1 = ambit::BlockedTensor::build(tensor_type_, "temp1", {"hhaa"});
    temp1["ijuv"] = T2["ijxy"] * Lambda2["xyuv"];
    C1["ir"] += 0.25 * alpha * temp1["ijuv"] * H2["uvrj"];

    temp1 = ambit::BlockedTensor::build(tensor_type_, "temp1", {"aapp"});
    temp1["xyab"] = T2["uvab"] * Lambda2["xyuv"];
    C1["pa"] -= 0.25 * alpha * temp1["xyab"] * H2["pbxy"];

    temp1 = ambit::BlockedTensor::build(tensor_type_, "temp1", {"hapa"});
    temp1["iuax"] = T2["iyav"] * Lambda2["uvxy"];
    C1["ir"] += alpha * temp1["iuax"] * H2["axru"];
    C1["pa"] -= alpha * temp1["iuax"] * H2["pxiu"];

    // [Hbar2, T2] C_4 C_2 1:3 -> C1
    temp1 = ambit::BlockedTensor::build(tensor_type_, "temp1", {"pa"});
    temp1["au"] = H2["avxy"] * Lambda2["xyuv"];
    C1["jb"] += 0.5 * alpha * temp1["au"] * T2["ujab"];

    temp1 = ambit::BlockedTensor::build(tensor_type_, "temp1", {"ah"});
    temp1["ui"] = H2["xyiv"] * Lambda2["uvxy"];
    C1["jb"] -= 0.5 * alpha * temp1["ui"] * T2["ijub"];

    temp1 = ambit::BlockedTensor::build(tensor_type_, "temp1", {"av"});
    temp1["xe"] = T2["uvey"] * Lambda2["xyuv"];
    C1["qs"] += 0.5 * alpha * temp1["xe"] * H2["eqxs"];

    temp1 = ambit::BlockedTensor::build(tensor_type_, "temp1", {"ca"});
    temp1["mx"] = T2["myuv"] * Lambda2["uvxy"];
    C1["qs"] -= 0.5 * alpha * temp1["mx"] * H2["xqms"];

    //    outfile->Printf("  Done. Timing %10.3f s", timer.get());
}

void MRDSRG_SO::H1_T2_C2(BlockedTensor& H1, BlockedTensor& T2, const double& alpha,
                         BlockedTensor& C2) {
    //    local_timer timer;
    //    std::string str = "Computing [Hbar1, T2] -> C2 ...";
    //    outfile->Printf("\n    %-35s", str.c_str());

    C2["ijpb"] += alpha * T2["ijab"] * H1["ap"];
    C2["ijap"] += alpha * T2["ijab"] * H1["bp"];
    C2["qjab"] -= alpha * T2["ijab"] * H1["qi"];
    C2["iqab"] -= alpha * T2["ijab"] * H1["qj"];

    //    outfile->Printf("  Done. Timing %10.3f s", timer.get());
}

void MRDSRG_SO::H2_T1_C2(BlockedTensor& H2, BlockedTensor& T1, const double& alpha,
                         BlockedTensor& C2) {
    //    local_timer timer;
    //    std::string str = "Computing [Hbar2, T1] -> C2 ...";
    //    outfile->Printf("\n    %-35s", str.c_str());

    C2["irpq"] += alpha * T1["ia"] * H2["arpq"];
    C2["ripq"] += alpha * T1["ia"] * H2["rapq"];
    C2["rsaq"] -= alpha * T1["ia"] * H2["rsiq"];
    C2["rspa"] -= alpha * T1["ia"] * H2["rspi"];

    //    outfile->Printf("  Done. Timing %10.3f s", timer.get());
}

void MRDSRG_SO::H2_T2_C2(BlockedTensor& H2, BlockedTensor& T2, const double& alpha,
                         BlockedTensor& C2) {
    //    local_timer timer;
    //    std::string str = "Computing [Hbar2, T2] -> C2 ...";
    //    outfile->Printf("\n    %-35s", str.c_str());

    // particle-particle contractions
    C2["ijrs"] += 0.5 * alpha * H2["abrs"] * T2["ijab"];

    BlockedTensor temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"hhpa"});
    temp["ijby"] = T2["ijbx"] * Gamma1["xy"];
    C2["ijrs"] -= alpha * temp["ijby"] * H2["byrs"];

    // hole-hole contractions
    C2["pqab"] += 0.5 * alpha * H2["pqij"] * T2["ijab"];

    temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"ahpp"});
    temp["xjab"] = T2["yjab"] * Eta1["xy"];
    C2["pqab"] -= alpha * temp["xjab"] * H2["pqxj"];

    // particle-hole contractions
    temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"hhpp"});
    temp["kjab"] = T2["ijab"] * Gamma1["ki"];
    C2["qjsb"] += alpha * temp["kjab"] * H2["aqks"];
    C2["qjas"] += alpha * temp["kjab"] * H2["bqks"];

    C2["iqsb"] -= alpha * temp["kiab"] * H2["aqks"];
    C2["iqas"] -= alpha * temp["kiab"] * H2["bqks"];

    temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"hhap"});
    temp["ijvb"] = T2["ijub"] * Gamma1["uv"];
    C2["qjsb"] -= alpha * temp["ijvb"] * H2["vqis"];
    C2["iqsb"] -= alpha * temp["ijvb"] * H2["vqjs"];

    C2["qjas"] += alpha * temp["ijva"] * H2["vqis"];
    C2["iqas"] += alpha * temp["ijva"] * H2["vqjs"];

    //    outfile->Printf("  Done. Timing %10.3f s", timer.get());
}

void MRDSRG_SO::H2_T2_C3(BlockedTensor& H2, BlockedTensor& T2, const double& alpha,
                         BlockedTensor& C3) {
    BlockedTensor temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"gggggg"});
    temp["rsjabq"] -= H2["rsqi"] * T2["ijab"];
    C3["rsjabq"] += alpha * temp["rsjabq"];
    C3["rjsabq"] -= alpha * temp["rsjabq"];
    C3["jrsabq"] += alpha * temp["rsjabq"];
    C3["rsjaqb"] -= alpha * temp["rsjabq"];
    C3["rjsaqb"] += alpha * temp["rsjabq"];
    C3["jrsaqb"] -= alpha * temp["rsjabq"];
    C3["rsjqab"] += alpha * temp["rsjabq"];
    C3["rjsqab"] -= alpha * temp["rsjabq"];
    C3["jrsqab"] += alpha * temp["rsjabq"];

    temp.zero();
    temp["ijspqb"] += H2["aspq"] * T2["ijba"];
    C3["ijspqb"] += alpha * temp["ijspqb"];
    C3["isjpqb"] -= alpha * temp["ijspqb"];
    C3["sijpqb"] += alpha * temp["ijspqb"];
    C3["ijspbq"] -= alpha * temp["ijspqb"];
    C3["isjpbq"] += alpha * temp["ijspqb"];
    C3["sijpbq"] -= alpha * temp["ijspqb"];
    C3["ijsbpq"] += alpha * temp["ijspqb"];
    C3["isjbpq"] -= alpha * temp["ijspqb"];
    C3["sijbpq"] += alpha * temp["ijspqb"];
}

void MRDSRG_SO::H3_T1_C1(BlockedTensor& H3, BlockedTensor& T1, const double& alpha,
                         BlockedTensor& C1) {
    BlockedTensor temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"gaagaa"});
    temp["qxypuv"] += 0.5 * H3["qeypuv"] * T1["xe"];
    temp["qxypuv"] -= 0.5 * H3["qxypmv"] * T1["mu"];

    C1["qp"] += alpha * temp["qxypuv"] * Lambda2["uvxy"];
}

void MRDSRG_SO::H3_T1_C2(BlockedTensor& H3, BlockedTensor& T1, const double& alpha,
                         BlockedTensor& C2) {
    C2["toqr"] += alpha * H3["atojqr"] * T1["ia"] * Gamma1["ji"];
    C2["toqr"] -= alpha * H3["ytomqr"] * T1["mx"] * Gamma1["xy"];
}

void MRDSRG_SO::H3_T2_C1(BlockedTensor& H3, BlockedTensor& T2, const double& alpha,
                         BlockedTensor& C1) {
    // (6:2)
    BlockedTensor temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"pa"});
    temp["ax"] += (1.0 / 12.0) * H3["ayzuvw"] * Lambda3["uvwxyz"];
    C1["jb"] += alpha * temp["ax"] * T2["xjab"];

    temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"ah"});
    temp["ui"] += (1.0 / 12.0) * H3["xyzivw"] * Lambda3["uvwxyz"];
    C1["jb"] -= alpha * temp["ui"] * T2["ijub"];

    // (5:3)
    temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"ppga"});
    temp["abpx"] += 0.25 * H3["abypuv"] * Lambda2["uvxy"];
    C1["ip"] += alpha * temp["abpx"] * T2["ixab"];

    temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"paga"});
    temp["azpx"] += 0.5 * H3["azypuv"] * Lambda2["uvxy"];
    C1["ip"] -= alpha * T2["ixaw"] * Gamma1["wz"] * temp["azpx"];

    temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"gahh"});
    temp["suij"] += 0.25 * H3["sxyijv"] * Lambda2["uvxy"];
    C1["sa"] += alpha * temp["suij"] * T2["ijau"];

    temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"gaha"});
    temp["suiw"] += 0.5 * H3["sxyiwv"] * Lambda2["uvxy"];
    C1["sa"] -= alpha * temp["suiw"] * Eta1["wz"] * T2["izau"];

    temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"aagh"});
    temp["uzpj"] += 0.5 * H3["xyzpjv"] * Lambda2["uvxy"];
    C1["ip"] -= alpha * T2["ijuw"] * Gamma1["wz"] * temp["uzpj"];

    temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"pagc"});
    temp["bupm"] += 0.5 * H3["xybpmv"] * Lambda2["uvxy"];
    C1["ip"] += alpha * T2["imub"] * temp["bupm"];

    temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"paga"});
    temp["bupw"] += 0.5 * H3["xybpwv"] * Lambda2["uvxy"];
    C1["ip"] += alpha * T2["izub"] * temp["bupw"] * Gamma1["wz"];

    temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"gpaa"});
    temp["sbwx"] += 0.5 * H3["sbyuvw"] * Lambda2["uvxy"];
    C1["sa"] -= alpha * temp["sbwx"] * Eta1["wz"] * T2["xzab"];

    temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"gvha"});
    temp["sejx"] += 0.5 * H3["seyjuv"] * Lambda2["uvxy"];
    C1["sa"] += alpha * temp["sejx"] * T2["xjae"];

    temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"gaha"});
    temp["szjx"] += 0.5 * H3["szyujv"] * Lambda2["uvxy"];
    C1["sa"] += alpha * temp["szjx"] * T2["xjaw"] * Eta1["wz"];

    temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"aagh"});
    temp["uvpj"] += (1.0 / 12.0) * H3["xyzpjw"] * Lambda3["uvwxyz"];
    C1["ip"] += alpha * T2["ijuv"] * temp["uvpj"];

    temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"paga"});
    temp["aupx"] += 0.25 * H3["ayzpvw"] * Lambda3["uvwxyz"];
    C1["ip"] += alpha * T2["ixau"] * temp["aupx"];

    temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"gpaa"});
    temp["sbxy"] += (1.0 / 12.0) * H3["sbzuvw"] * Lambda3["uvwxyz"];
    C1["sa"] -= alpha * temp["sbxy"] * T2["xyab"];

    temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"gaha"});
    temp["suix"] += 0.25 * H3["syzivw"] * Lambda3["uvwxyz"];
    C1["sa"] -= alpha * temp["suix"] * T2["ixau"];

    // (4:4)
    temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"hhpp"});
    temp["klab"] += 0.25 * T2["ijab"] * Gamma1["ki"] * Gamma1["lj"];
    temp["klav"] -= 0.5 * T2["ijau"] * Gamma1["ki"] * Gamma1["uv"] * Gamma1["lj"];
    C1["or"] += alpha * H3["aboklr"] * temp["klab"];

    temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"hhaa"});
    temp["ijvy"] -= 0.25 * T2["ijux"] * Gamma1["uv"] * Gamma1["xy"];
    temp["ikvy"] += 0.5 * T2["ijux"] * Gamma1["xy"] * Gamma1["kj"] * Gamma1["uv"];
    C1["or"] += alpha * H3["vyoijr"] * temp["ijvy"];

    temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"aapp"});
    temp["uvab"] += 0.125 * T2["xyab"] * Lambda2["uvxy"];
    C1["or"] += alpha * H3["abouvr"] * temp["uvab"];

    temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"aava"});
    temp["uvez"] += 0.25 * T2["xyew"] * Gamma1["wz"] * Lambda2["uvxy"];
    C1["or"] -= alpha * H3["ezouvr"] * temp["uvez"];

    temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"hhaa"});
    temp["ijxy"] += 0.125 * T2["ijuv"] * Lambda2["uvxy"];
    C1["or"] += alpha * H3["xyoijr"] * temp["ijxy"];

    temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"caaa"});
    temp["mwxy"] += 0.25 * T2["mzuv"] * Lambda2["uvxy"] * Eta1["wz"];
    C1["or"] -= alpha * H3["xyomwr"] * temp["mwxy"];

    temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"hapa"});
    temp["iuax"] += T2["iyav"] * Lambda2["uvxy"];
    C1["or"] += alpha * H3["axoiur"] * temp["iuax"];

    temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"aava"});
    temp["wuex"] -= T2["zyev"] * Eta1["wz"] * Lambda2["uvxy"];
    temp["uvex"] += 0.25 * T2["yzew"] * Lambda3["uvwxyz"];
    C1["or"] += alpha * H3["exowur"] * temp["wuex"];

    temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"caaa"});
    temp["muzx"] -= T2["mywv"] * Lambda2["uvxy"] * Gamma1["wz"];
    temp["muyz"] += 0.25 * T2["mxvw"] * Lambda3["uvwxyz"];
    C1["or"] += alpha * H3["zxomur"] * temp["muzx"];
}

void MRDSRG_SO::H3_T2_C2(BlockedTensor& H3, BlockedTensor& T2, const double& alpha,
                         BlockedTensor& C2) {
    // (4:2)
    BlockedTensor temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"hhapaa"});
    BlockedTensor temp2 = ambit::BlockedTensor::build(tensor_type_, "temp2", {"ghpg"});
    temp["ijuaxy"] += 0.5 * T2["ijav"] * Lambda2["uvxy"];
    C2["ijpq"] += alpha * temp["ijuaxy"] * H3["axypqu"];
    temp2["ojar"] += alpha * temp["ijvaxy"] * H3["xyoivr"];

    temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"haappa"});
    temp["iuvabx"] += 0.5 * T2["iyab"] * Lambda2["uvxy"];
    C2["stab"] -= alpha * temp["iuvabx"] * H3["stxiuv"];
    temp2["ojar"] -= alpha * temp["juvaby"] * H3["byouvr"];

    C2["ojar"] += temp2["ojar"];
    C2["ojra"] -= temp2["ojar"];
    C2["joar"] -= temp2["ojar"];
    C2["jora"] += temp2["ojar"];

    // (3:3)
    temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"hhpp"});
    temp2 = ambit::BlockedTensor::build(tensor_type_, "temp2", {"gggp"});
    temp["ijcb"] += 0.5 * T2["ijab"] * Eta1["ac"];
    temp2["torb"] += alpha * temp["ijcb"] * H3["ctorij"];

    temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"aapp"});
    temp["uxab"] += 0.5 * T2["vyab"] * Eta1["uv"] * Eta1["xy"];
    temp2["torb"] += alpha * temp["uxab"] * H3["atorux"];

    temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"happ"});
    temp["iucb"] += T2["ivab"] * Eta1["ac"] * Eta1["uv"];
    temp2["torb"] -= alpha * temp["iucb"] * H3["ctoriu"];

    C2["torb"] += temp2["torb"];
    C2["tobr"] -= temp2["torb"];

    temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"hhpp"});
    temp2 = ambit::BlockedTensor::build(tensor_type_, "temp2", {"ghgg"});
    temp["kjab"] += 0.5 * T2["ijab"] * Gamma1["ki"];
    temp2["ojqr"] -= alpha * temp["kjab"] * H3["abokqr"];

    temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"hhaa"});
    temp["ijvy"] += 0.5 * T2["ijux"] * Gamma1["uv"] * Gamma1["xy"];
    temp2["ojqr"] -= alpha * temp["ijvy"] * H3["vyoiqr"];

    temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"hhpa"});
    temp["kjav"] += T2["ijau"] * Gamma1["uv"] * Gamma1["ki"];
    temp2["ojqr"] += alpha * temp["kjav"] * H3["avokqr"];

    C2["ojqr"] += temp2["ojqr"];
    C2["joqr"] -= temp2["ojqr"];

    temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"aapp"});
    temp2 = ambit::BlockedTensor::build(tensor_type_, "temp2", {"ggpg"});
    temp["uvab"] += 0.25 * T2["xyab"] * Lambda2["uvxy"];
    temp2["rsaq"] += alpha * temp["uvab"] * H3["brsuvq"];

    temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"hapa"});
    temp["iuax"] += T2["iyav"] * Lambda2["uvxy"];
    temp2["rsaq"] -= alpha * temp["iuax"] * H3["rsxiqu"];

    C2["rsaq"] += temp2["rsaq"];
    C2["rsqa"] -= temp2["rsaq"];

    temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"hhaa"});
    temp2 = ambit::BlockedTensor::build(tensor_type_, "temp2", {"hggg"});
    temp["ijxy"] += 0.25 * T2["ijuv"] * Lambda2["uvxy"];
    temp2["ispq"] -= alpha * temp["ijxy"] * H3["xysjpq"];

    temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"hapa"});
    temp["iuax"] += T2["iyav"] * Lambda2["uvxy"];
    temp2["ispq"] += alpha * temp["iuax"] * H3["asxpqu"];

    C2["ispq"] += temp2["ispq"];
    C2["sipq"] -= temp2["ispq"];

    // (2:4)
    temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"av"});
    temp["ue"] += 0.5 * T2["xyev"] * Lambda2["uvxy"];
    C2["toqr"] += alpha * temp["ue"] * H3["etouqr"];

    temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"ca"});
    temp["mx"] += 0.5 * T2["myuv"] * Lambda2["uvxy"];
    C2["toqr"] -= alpha * temp["mx"] * H3["xtomqr"];
}
} // namespace forte
