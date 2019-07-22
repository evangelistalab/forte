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
        {"Spin-Orbital Multireference Driven Similarity Renormalization Group", "Chenyang Li"});
    startup();
    print_summary();
}

MRDSRG_SO::~MRDSRG_SO() {}

std::shared_ptr<ActiveSpaceIntegrals> MRDSRG_SO::compute_Heff_actv() {
    throw psi::PSIEXCEPTION(
        "Computing active-space Hamiltonian is not yet implemented for spin-orbital code.");

    return std::make_shared<ActiveSpaceIntegrals>(
        ints_, mo_space_info_->get_corr_abs_mo("ACTIVE"),
        mo_space_info_->get_corr_abs_mo("RESTRICTED_DOCC"));

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
    BlockedTensor::reset_mo_spaces();
    BlockedTensor::set_expert_mode(true);

    Eref = compute_Eref_from_rdms(rdms_, ints_, mo_space_info_);
    frozen_core_energy = ints_->frozen_core_energy();

    do_t3_ = foptions_->get_str("CORR_LEVEL").find("DSRG3") != std::string::npos;
    ldsrg3_ddca_ = foptions_->get_bool("LDSRG3_DDCA");
    ncomm_3body_ = foptions_->get_int("LDSRG3_NCOMM_3BODY");
    if (ncomm_3body_ > 2 or ncomm_3body_ <= 0) {
        ncomm_3body_ = foptions_->get_int("DSRG_RSC_NCOMM");
    }
    ldsrg3_level_ = 3;
    if (foptions_->get_str("CORR_LEVEL") == "LDSRG3_2")
        ldsrg3_level_ = 2;
    if (foptions_->get_str("CORR_LEVEL") == "LDSRG3_1")
        ldsrg3_level_ = 1;

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
    acore_sos = mo_space_info_->get_corr_abs_mo("RESTRICTED_DOCC");
    aactv_sos = mo_space_info_->get_corr_abs_mo("ACTIVE");
    avirt_sos = mo_space_info_->get_corr_abs_mo("RESTRICTED_UOCC");

    // put all beta behind alpha
    size_t mo_shift = mo_space_info_->size("CORRELATED");

    for (size_t idx : acore_sos)
        bcore_sos.push_back(idx + mo_shift);
    for (size_t idx : aactv_sos)
        bactv_sos.push_back(idx + mo_shift);
    for (size_t idx : avirt_sos)
        bvirt_sos.push_back(idx + mo_shift);

    // Form the maps from active relative indices to absolute SO indices
    std::map<size_t, size_t> aactvrel_to_soabs;
    std::map<size_t, size_t> bactvrel_to_soabs;
    for (size_t i = 0, nactv = aactv_sos.size(); i < nactv; ++i) {
        aactvrel_to_soabs[i] = aactv_sos[i];
        bactvrel_to_soabs[i] = bactv_sos[i];
    }

    // spin orbital indices
    core_sos_ = acore_sos;
    actv_sos_ = aactv_sos;
    virt_sos_ = avirt_sos;
    core_sos_.insert(core_sos_.end(), bcore_sos.begin(), bcore_sos.end());
    actv_sos_.insert(actv_sos_.end(), bactv_sos.begin(), bactv_sos.end());
    virt_sos_.insert(virt_sos_.end(), bvirt_sos.begin(), bvirt_sos.end());

    // size of each spin orbital space
    nc_ = core_sos_.size();
    na_ = actv_sos_.size();
    nv_ = virt_sos_.size();
    nh_ = na_ + nc_;
    np_ = na_ + nv_;
    nso_ = nh_ + nv_;
    size_t nmo = nso_ / 2;
    size_t na_mo = na_ / 2;

    BTF_->add_mo_space("c", "m,n,c0,c1,c2,c3,c4,c5,c6,c7,c8,c9", core_sos_, NoSpin);
    BTF_->add_mo_space("a", "u,v,w,x,y,z,a0,a1,a2,a3,a4,a5,a6,a7,a8,a9", actv_sos_, NoSpin);
    BTF_->add_mo_space("v", "e,f,v0,v1,v2,v3,v4,v5,v6,v7,v8,v9", virt_sos_, NoSpin);

    BTF_->add_composite_mo_space("h", "i,j,k,l,h0,h1,h2,h3,h4,h5,h6,h7,h8,h9", {"c", "a"});
    BTF_->add_composite_mo_space("p", "a,b,c,d,p0,p1,p2,p3,p4,p5,p6,p7,p8,p9", {"a", "v"});
    BTF_->add_composite_mo_space("g", "p,q,r,s,t,o,g0,g1,g2,g3,g4,g5,g6,g7,g8,g9", {"c", "a", "v"});

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
    L1 = BTF_->build(tensor_type_, "Gamma1", {"hh"});
    Eta1 = BTF_->build(tensor_type_, "Eta1", {"pp"});
    (L1.block("cc")).iterate([&](const std::vector<size_t>& i, double& value) {
        value = (i[0] == i[1] ? 1.0 : 0.0);
    });
    Eta1.iterate([&](const std::vector<size_t>& i, const std::vector<SpinType>&, double& value) {
        value = (i[0] == i[1] ? 1.0 : 0.0);
    });
    (rdms_.g1a()).citerate([&](const std::vector<size_t>& i, const double& value) {
        size_t index = i[0] * na_ + i[1];
        (L1.block("aa")).data()[index] = value;
        (Eta1.block("aa")).data()[index] -= value;
    });
    (rdms_.g1b()).citerate([&](const std::vector<size_t>& i, const double& value) {
        size_t index = (i[0] + na_mo) * na_ + (i[1] + na_mo);
        (L1.block("aa")).data()[index] = value;
        (Eta1.block("aa")).data()[index] -= value;
    });

    // prepare two-body density cumulant
    L2 = BTF_->build(tensor_type_, "Lambda2", {"aaaa"});
    (rdms_.L2aa()).citerate([&](const std::vector<size_t>& i, const double& value) {
        if (std::fabs(value) > 1.0e-15) {
            size_t index = 0;
            for (int m = 0; m < 4; ++m) {
                index += i[m] * myPow(na_, 3 - m);
            }
            (L2.block("aaaa")).data()[index] = value;
        }
    });
    (rdms_.L2bb()).citerate([&](const std::vector<size_t>& i, const double& value) {
        if (std::fabs(value) > 1.0e-15) {
            size_t index = 0;
            for (int m = 0; m < 4; ++m) {
                index += (i[m] + na_mo) * myPow(na_, 3 - m);
            }
            (L2.block("aaaa")).data()[index] = value;
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
            (L2.block("aaaa")).data()[index] = value;
            index = i1 * myPow(na_, 3) + i0 * myPow(na_, 2) + i3 * myPow(na_, 1) + i2;
            (L2.block("aaaa")).data()[index] = value;
            index = i1 * myPow(na_, 3) + i0 * myPow(na_, 2) + i2 * myPow(na_, 1) + i3;
            (L2.block("aaaa")).data()[index] = -value;
            index = i0 * myPow(na_, 3) + i1 * myPow(na_, 2) + i3 * myPow(na_, 1) + i2;
            (L2.block("aaaa")).data()[index] = -value;
        }
    });
    outfile->Printf("\n    Norm of L2: %12.8f.", L2.norm());

    // prepare three-body density cumulant
    if (foptions_->get_str("THREEPDC") != "ZERO") {
        L3 = BTF_->build(tensor_type_, "Lambda3", {"aaaaaa"});
        (rdms_.L3aaa()).citerate([&](const std::vector<size_t>& i, const double& value) {
            if (std::fabs(value) > 1.0e-15) {
                size_t index = 0;
                for (int m = 0; m < 6; ++m) {
                    index += i[m] * myPow(na_, 5 - m);
                }
                (L3.block("aaaaaa")).data()[index] = value;
            }
        });
        (rdms_.L3bbb()).citerate([&](const std::vector<size_t>& i, const double& value) {
            if (std::fabs(value) > 1.0e-15) {
                size_t index = 0;
                for (int m = 0; m < 6; ++m) {
                    index += (i[m] + na_mo) * myPow(na_, 5 - m);
                }
                (L3.block("aaaaaa")).data()[index] = value;
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
                        (L3.block("aaaaaa")).data()[index] = value * pow(-1.0, m + n);
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
                        (L3.block("aaaaaa")).data()[index] = value * pow(-1.0, m + n);
                    }
                }
            }
        });
        outfile->Printf("\n    Norm of L3: %12.8f.", L3.norm());
    }

    // build Fock matrix (initial guess of one-body Hamiltonian)
    F = BTF_->build(tensor_type_, "Fock", {"gg"});
    F["pq"] = H["pq"];
    F["pq"] += V["pjqi"] * L1["ij"];

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
        {"correlation level", foptions_->get_str("CORR_LEVEL")},
        {"int_type", foptions_->get_str("INT_TYPE")},
        {"source operator", source_}};

    if (do_t3_) {
        calculation_info.push_back({"LDSRG3_NCOMM_3BODY", ncomm_3body_});
        calculation_info_string.push_back({"LDSRG_DDCA", ldsrg3_ddca_ ? "TRUE" : "FALSE"});
    }

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

    // use simple single-reference guess
    T1["ia"] = F["ia"];
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

void MRDSRG_SO::guess_t3() {
//    local_timer timer;
//    std::string str = "Computing T3 amplitudes     ...";
//    outfile->Printf("\n    %-35s", str.c_str());

//    ambit::BlockedTensor C3 = ambit::BlockedTensor::build(tensor_type_, "C3", {"hhhppp"});
//    auto temp = ambit::BlockedTensor::build(CoreTensor, "temp", {"hhhppp"});
//    temp["g2,c0,c1,g0,g1,v0"] += -1.0 * V["g2,v1,g0,g1"] * T2["c0,c1,v0,v1"];
//    C3["c0,c1,g2,g0,g1,v0"] += temp["g2,c0,c1,g0,g1,v0"];
//    C3["c0,g2,c1,g0,g1,v0"] -= temp["g2,c0,c1,g0,g1,v0"];
//    C3["g2,c0,c1,g0,g1,v0"] += temp["g2,c0,c1,g0,g1,v0"];
//    C3["c0,c1,g2,g0,v0,g1"] -= temp["g2,c0,c1,g0,g1,v0"];
//    C3["c0,g2,c1,g0,v0,g1"] += temp["g2,c0,c1,g0,g1,v0"];
//    C3["g2,c0,c1,g0,v0,g1"] -= temp["g2,c0,c1,g0,g1,v0"];
//    C3["c0,c1,g2,v0,g0,g1"] += temp["g2,c0,c1,g0,g1,v0"];
//    C3["c0,g2,c1,v0,g0,g1"] -= temp["g2,c0,c1,g0,g1,v0"];
//    C3["g2,c0,c1,v0,g0,g1"] += temp["g2,c0,c1,g0,g1,v0"];

//    temp.zero();
//    temp["g1,g2,c0,g0,v0,v1"] += 1.0 * V["g1,g2,g0,c1"] * T2["c0,c1,v0,v1"];
//    C3["c0,g1,g2,g0,v0,v1"] += temp["g1,g2,c0,g0,v0,v1"];
//    C3["g1,c0,g2,g0,v0,v1"] -= temp["g1,g2,c0,g0,v0,v1"];
//    C3["g1,g2,c0,g0,v0,v1"] += temp["g1,g2,c0,g0,v0,v1"];
//    C3["c0,g1,g2,v0,g0,v1"] -= temp["g1,g2,c0,g0,v0,v1"];
//    C3["g1,c0,g2,v0,g0,v1"] += temp["g1,g2,c0,g0,v0,v1"];
//    C3["g1,g2,c0,v0,g0,v1"] -= temp["g1,g2,c0,g0,v0,v1"];
//    C3["c0,g1,g2,v0,v1,g0"] += temp["g1,g2,c0,g0,v0,v1"];
//    C3["g1,c0,g2,v0,v1,g0"] -= temp["g1,g2,c0,g0,v0,v1"];
//    C3["g1,g2,c0,v0,v1,g0"] += temp["g1,g2,c0,g0,v0,v1"];

//    T3["ijkabc"] = C3["ijkabc"];
//    T3["ijkabc"] += C3["abcijk"];

//    T3.iterate([&](const std::vector<size_t>& i, const std::vector<SpinType>&, double& value) {
//        value *= renormalized_denominator(Fd[i[0]] + Fd[i[1]] + Fd[i[2]] - Fd[i[3]] - Fd[i[4]] -
//                Fd[i[5]]);
//    });

//    // zero internal amplitudes
//    T3.block("aaaaaa").zero();

//    // norm and max
//    T3max = T3.norm(0), T3norm = T3.norm();

//    outfile->Printf("  Done. Timing %10.3f s", timer.get());
}

double MRDSRG_SO::renormalized_denominator(double D) {
    double Z = std::sqrt(s_) * D;
    if (std::fabs(Z) < std::pow(0.1, taylor_threshold_)) {
        return Taylor_Exp(Z, taylor_order_) * std::sqrt(s_);
    } else {
        return (1.0 - std::exp(-s_ * D * D)) / D;
    }
}

void MRDSRG_SO::update_t2() {
    // compute DT2 = Hbar2 * (1 - exp(-s * D * D)) / D - T2 * exp(-s * D * D)
    BlockedTensor DT2 = ambit::BlockedTensor::build(tensor_type_, "DT2", {"hhpp"});
    DT2["ijab"] = Hbar2["ijab"];
    DT2.iterate([&](const std::vector<size_t>& i, const std::vector<SpinType>&, double& value) {
        value *= renormalized_denominator(Fd[i[0]] + Fd[i[1]] - Fd[i[2]] - Fd[i[3]]);
    });

    // copy T2 to Hbar2
    Hbar2["ijab"] = T2["ijab"];

    // scale T2 by exp(-s * D * D)
    T2.iterate([&](const std::vector<size_t>& i, const std::vector<SpinType>&, double& value) {
        double delta = Fd[i[0]] + Fd[i[1]] - Fd[i[2]] - Fd[i[3]];
        value *= std::exp(-s_ * delta * delta);
    });

    DT2["ijab"] -= T2["ijab"];
    DT2.block("aaaa").zero();
    rms_t2 = DT2.norm();

    T2["ijab"] = Hbar2["ijab"] + DT2["ijab"];

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

void MRDSRG_SO::update_t3() {
    // compute DT3 = Hbar3 * (1 - exp(-s * D * D)) / D - T3 * exp(-s * D * D)
    BlockedTensor DT3 = ambit::BlockedTensor::build(tensor_type_, "DT3", {"hhhppp"});
    DT3["ijkabc"] = Hbar3["ijkabc"];
    DT3.iterate([&](const std::vector<size_t>& i, const std::vector<SpinType>&, double& value) {
        value *= renormalized_denominator(Fd[i[0]] + Fd[i[1]] + Fd[i[2]] - Fd[i[3]] - Fd[i[4]] -
                Fd[i[5]]);
    });

    // copy T3 to Hbar3
    Hbar3["ijkabc"] = T3["ijkabc"];

    // scale T3 by exp(-s * D * D)
    T3.iterate([&](const std::vector<size_t>& i, const std::vector<SpinType>&, double& value) {
        double delta = Fd[i[0]] + Fd[i[1]] + Fd[i[2]] - Fd[i[3]] - Fd[i[4]] - Fd[i[5]];
        value *= std::exp(-s_ * delta * delta);
    });

    DT3["ijkabc"] -= T3["ijkabc"];
    DT3.block("aaaaaa").zero();
    rms_t3 = DT3.norm();

    T3["ijkabc"] = Hbar3["ijkabc"] + DT3["ijkabc"];

    // norm and max
    T3max = T3.norm(0), T3norm = T3.norm();
}

std::vector<std::string> MRDSRG_SO::sr_ldsrg3_ddca_blocks() {
    return {"ccccvv","cccvcv","cccvvc","cccvvv",
        "ccvccv","ccvcvc","ccvcvv","ccvvcc",
        "ccvvcv","ccvvvc","ccvvvv","cvcccv",
        "cvccvc","cvccvv","cvcvcc","cvcvcv",
        "cvcvvc","cvcvvv","cvvccc","cvvccv",
        "cvvcvc","cvvcvv","cvvvcc","cvvvcv",
        "cvvvvc","vccccv","vcccvc","vcccvv",
        "vccvcc","vccvcv","vccvvc","vccvvv",
        "vcvccc","vcvccv","vcvcvc","vcvcvv",
        "vcvvcc","vcvvcv","vcvvvc","vvcccc",
        "vvcccv","vvccvc","vvccvv","vvcvcc",
        "vvcvcv","vvcvvc","vvvccc","vvvccv",
        "vvvcvc","vvvvcc"};
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

    bool store_H3 = (ncomm_3body_ == foptions_->get_int("DSRG_RSC_NCOMM"));
    if (do_t3_) {
        if (store_H3) {
            if (ldsrg3_ddca_) {
                Hbar3 = BTF_->build(tensor_type_, "Hbar3", sr_ldsrg3_ddca_blocks());
            } else {
                Hbar3 = BTF_->build(tensor_type_, "Hbar3", {"gggggg"});
            }
        }
        T3 = BTF_->build(tensor_type_, "T3 Amplitudes", {"hhhppp"});
        guess_t3();
    }

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
                    "|Hbar1|_N  |Hbar2|_N    |T1|    |T2|    |T3|  max(T1) max(T2) max(T3)");
    outfile->Printf("\n    "
                    "----------------------------------------------------------"
                    "----------------------------------------");

    do {
        // compute hbar
        if (foptions_->get_str("CORR_LEVEL") == "QDSRG2") {
            compute_qhbar();
        } else {
            // single-commutator by default
            compute_lhbar();
        }
        double Edelta = Eref + Hbar0 - Etotal;
        Etotal = Eref + Hbar0;

        // norm of non-diagonal Hbar
        BlockedTensor temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"hv", "ca"});
        temp["ia"] = Hbar1["ia"];
        double Hbar1Nnorm = temp.norm();

        temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"hhpv", "hhva", "hcaa", "ccaa"});
        temp["ijab"] = Hbar2["ijab"];
        double Hbar2Nnorm = temp.norm();
        for (const std::string block: temp.block_labels()) {
            temp.block(block).reset();
        }

        outfile->Printf("\n      @CT %4d %20.12f %11.3e %10.3e %10.3e %7.4f "
                        "%7.4f %7.4f %7.4f %7.4f %7.4f",
                        cycle, Etotal, Edelta, Hbar1Nnorm, Hbar2Nnorm, T1norm, T2norm,
                        T3norm, T1max, T2max, T3max);

        // update amplitudes
        update_t2();
        update_t1();
        if (do_t3_) {
            if (store_H3) {
                update_t3();
            } else {
                direct_t3();
            }
        }

        // test convergence
        double rms = std::max(std::max(rms_t1, rms_t2), rms_t3);
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
    outfile->Printf("\n\n\n    %s Energy Summary", foptions_->get_str("CORR_LEVEL").c_str());
    outfile->Printf("\n    Correlation energy      = %25.15f", Etotal - Eref);
    outfile->Printf("\n  * Total energy            = %25.15f\n", Etotal);

    psi::Process::environment.globals["CURRENT ENERGY"] = Etotal;

    return Etotal;
}

void MRDSRG_SO::compute_lhbar() {

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
    double C0 = 0.0;
    BlockedTensor C1 = ambit::BlockedTensor::build(tensor_type_, "C1", {"gg"});
    BlockedTensor C2 = ambit::BlockedTensor::build(tensor_type_, "C2", {"gggg"});

    BlockedTensor O3, C3;
    bool store_H3 = (ncomm_3body_ == foptions_->get_int("DSRG_RSC_NCOMM"));
    if (do_t3_ and store_H3) {
        Hbar3.zero();
        if (ldsrg3_ddca_) {
            std::vector<std::string> blocks = sr_ldsrg3_ddca_blocks();
            O3 = ambit::BlockedTensor::build(tensor_type_, "O3", blocks);
            C3 = ambit::BlockedTensor::build(tensor_type_, "C3", blocks);
        } else {
            O3 = ambit::BlockedTensor::build(tensor_type_, "O3", {"gggggg"});
            C3 = ambit::BlockedTensor::build(tensor_type_, "C3", {"gggggg"});
        }
    }

    // compute Hbar recursively
    for (int n = 1; n <= maxn; ++n) {
        // prefactor before n-nested commutator
        double factor = 1.0 / n;

        if (do_t3_) {
            timer_on("3-body [H, A]");
            if (na_ == 0) {
                if (store_H3) {
                    comm_H_A_3_sr(factor, O1, O2, O3, T1, T2, T3, C0, C1, C2, C3);
                } else {
                    comm_H_A_3_sr_2(factor, O1, O2, T1, T2, T3, C0, C1, C2);
                }
            } else {
                comm_H_A_3(factor, O1, O2, O3, T1, T2, T3, C0, C1, C2, C3);
            }
            timer_off("3-body [H, A]");
        } else {
            comm_H_A_2(factor, O1, O2, T1, T2, C0, C1, C2);
        }

        // add to Hbar
        Hbar0 += C0;
        Hbar1["pq"] += C1["pq"];
        Hbar2["pqrs"] += C2["pqrs"];

        // copy C to O for next level commutator
        O1["pq"] = C1["pq"];
        O2["pqrs"] = C2["pqrs"];

        if (!store_H3 and (n == 2 or n == 3)) {
            if (n == 2 and ncomm_3body_ >= 1) {
                comm2_l3(F, V, T1, T2, T3, C0, C1, C2);
            }

            if (n == 3 and ncomm_3body_ >= 2) {
                if (ldsrg3_level_ == 3) {
                    comm3_q3_lv3(F, V, T1, T2, T3, C0, C1, C2);
                } else if (ldsrg3_level_ == 2) {
                    comm3_q3_lv2(F, V, T1, T2, T3, C0, C1, C2);
                } else {
                    comm3_q3_lv1(F, V, T1, T2, T3, C0, C1, C2);
                }
            }

            // add to Hbar
            Hbar0 += C0;
            Hbar1["pq"] += C1["pq"];
            Hbar2["pqrs"] += C2["pqrs"];

            // add C to O for next level commutator
            O1["pq"] += C1["pq"];
            O2["pqrs"] += C2["pqrs"];
        }

        // test convergence of C
        double norm_C1 = C1.norm();
        double norm_C2 = C2.norm();
        double norm_C3 = 0.0;
        if (do_t3_ and store_H3) {
            Hbar3["g0,g1,g2,g3,g4,g5"] += C3["g0,g1,g2,g3,g4,g5"];
            O3["g0,g1,g2,g3,g4,g5"] = C3["g0,g1,g2,g3,g4,g5"];
            norm_C3 = C3.norm();
        }

        if (std::sqrt(norm_C2 * norm_C2 + norm_C1 * norm_C1 + norm_C3 * norm_C3) < ct_threshold) {
            break;
        }

//        // Compute the commutator C = 1/n [O,T]
//        double C0 = 0.0;
//        C1.zero();
//        C2.zero();

//        // zero-body
//        H1_T1_C0(O1, T1, factor, C0);
//        H1_T2_C0(O1, T2, factor, C0);
//        H2_T1_C0(O2, T1, factor, C0);
//        H2_T2_C0(O2, T2, factor, C0);

//        // one-body
//        H1_T1_C1(O1, T1, factor, C1);
//        H1_T2_C1(O1, T2, factor, C1);
//        H2_T1_C1(O2, T1, factor, C1);
//        H2_T2_C1(O2, T2, factor, C1);

//        // two-body
//        H1_T2_C2(O1, T2, factor, C2);
//        H2_T1_C2(O2, T1, factor, C2);
//        H2_T2_C2(O2, T2, factor, C2);

//        //        outfile->Printf("\n   H0  = %20.12f", C0);
//        //        outfile->Printf("\n  |H1| = %20.12f", C1.norm(1));
//        //        outfile->Printf("\n  |H2| = %20.12f", C2.norm(1));
//        //        outfile->Printf("\n  --------------------------------");

//        // [H, A] = [H, T] + [H, T]^dagger
//        C0 *= 2.0;
//        O1["pq"] = C1["pq"];
//        C1["pq"] += O1["qp"];
//        O2["pqrs"] = C2["pqrs"];
//        C2["pqrs"] += O2["rspq"];

//        // Hbar += C
//        Hbar0 += C0;
//        Hbar1["pq"] += C1["pq"];
//        Hbar2["pqrs"] += C2["pqrs"];

//        // copy C to O for next level commutator
//        O1["pq"] = C1["pq"];
//        O2["pqrs"] = C2["pqrs"];

//        // test convergence of C
//        double norm_C1 = C1.norm();
//        double norm_C2 = C2.norm();
//        //        outfile->Printf("\n  %2d %20.12f %20e %20e",n,C0,norm_C1,norm_C2);
//        if (std::sqrt(norm_C2 * norm_C2 + norm_C1 * norm_C1) < ct_threshold) {
//            break;
//        }
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
} // namespace forte
