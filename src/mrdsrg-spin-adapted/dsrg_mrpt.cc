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
#include <numeric>
#include <vector>

#include "psi4/libpsi4util/process.h"
#include "boost/format.hpp"
#include "helpers/printing.h"
#include "dsrg_mrpt.h"

using namespace psi;

namespace forte {

DSRG_MRPT::DSRG_MRPT(RDMs rdms, std::shared_ptr<SCFInfo> scf_info,
                     std::shared_ptr<ForteOptions> options, std::shared_ptr<ForteIntegrals> ints,
                     std::shared_ptr<MOSpaceInfo> mo_space_info)
    : DynamicCorrelationSolver(rdms, scf_info, options, ints, mo_space_info),
      tensor_type_(ambit::CoreTensor) {
    print_method_banner({"Spin-Adapted 2nd-order DSRG-MRPT", "Chenyang Li"});
    print_citation();
    read_options();
    print_options();
    startup();
}

DSRG_MRPT::~DSRG_MRPT() { cleanup(); }

void DSRG_MRPT::cleanup() { dsrg_time_.print_comm_time(); }

void DSRG_MRPT::read_options() {
    print_ = foptions_->get_int("PRINT");

    corr_lv_ = foptions_->get_str("CORR_LEVEL");
    if (corr_lv_ != "PT2" && corr_lv_ != "PT3") {
        outfile->Printf("\n  Warning: CORR_LEVEL option \"%s\" is not "
                        "available in DSRG_MRPT. Changed to PT2.",
                        corr_lv_.c_str());
        corr_lv_ = "PT2";
    }

    ref_relax_ = foptions_->get_str("RELAX_REF");
    if (ref_relax_ != "NONE" && ref_relax_ != "ONCE") {
        outfile->Printf("\n  Warning: RELAX_REF option \"%s\" is not available "
                        "in DSRG_MRPT. Changed to ONCE",
                        ref_relax_.c_str());
        ref_relax_ = "ONCE";
    }

    s_ = foptions_->get_double("DSRG_S");
    if (s_ < 0) {
        outfile->Printf("\n  Error: S parameter for DSRG must >= 0!");
        throw psi::PSIEXCEPTION("S parameter for DSRG must >= 0!");
    }
    taylor_threshold_ = foptions_->get_int("TAYLOR_THRESHOLD");
    if (taylor_threshold_ <= 0) {
        outfile->Printf("\n  Error: Threshold for Taylor expansion must be an "
                        "integer greater than 0!");
        throw psi::PSIEXCEPTION("Threshold for Taylor expansion must be an integer "
                                "greater than 0!");
    }

    ntamp_ = foptions_->get_int("NTAMP");
    intruder_tamp_ = foptions_->get_double("INTRUDER_TAMP");

    source_ = foptions_->get_str("SOURCE");
    if (source_ != "STANDARD" && source_ != "LABS" && source_ != "DYSON") {
        outfile->Printf("\n  Warning: SOURCE option \"%s\" is not implemented "
                        "in DSRG_MRPT. Changed to STANDARD.",
                        source_.c_str());
        source_ = "STANDARD";
    }
    if (source_ == "STANDARD") {
        dsrg_source_ = std::make_shared<STD_SOURCE>(s_, taylor_threshold_);
    } else if (source_ == "LABS") {
        dsrg_source_ = std::make_shared<LABS_SOURCE>(s_, taylor_threshold_);
    } else if (source_ == "DYSON") {
        dsrg_source_ = std::make_shared<DYSON_SOURCE>(s_, taylor_threshold_);
    }

    ccvv_source_ = foptions_->get_str("CCVV_SOURCE");
}

void DSRG_MRPT::print_options() {
    // fill in information
    std::vector<std::pair<std::string, size_t>> calculation_info{{"ntamp", ntamp_}};

    std::vector<std::pair<std::string, double>> calculation_info_double{
        {"flow parameter", s_},
        {"taylor expansion threshold", pow(10.0, -double(taylor_threshold_))},
        {"intruder_tamp", intruder_tamp_}};

    std::vector<std::pair<std::string, std::string>> calculation_info_string{
        {"corr_level", corr_lv_},
        {"int_type", foptions_->get_str("INT_TYPE")},
        {"source operator", source_},
        {"reference relaxation", ref_relax_},
        {"core virtual source type", ccvv_source_}};

    // print some information
    print_h2("Calculation Information");
    for (auto& str_dim : calculation_info) {
        outfile->Printf("\n    %-35s %15zu", str_dim.first.c_str(), str_dim.second);
    }
    for (auto& str_dim : calculation_info_double) {
        outfile->Printf("\n    %-35s %15.3e", str_dim.first.c_str(), str_dim.second);
    }
    for (auto& str_dim : calculation_info_string) {
        outfile->Printf("\n    %-35s %15s", str_dim.first.c_str(), str_dim.second.c_str());
    }
    outfile->Printf("\n");
}

void DSRG_MRPT::startup() {
    // frozen-core energy
    frozen_core_energy_ = ints_->frozen_core_energy();

    // reference energy
    Eref_ = compute_Eref_from_rdms(rdms_, ints_, mo_space_info_);

    // orbital spaces
    core_mos_ = mo_space_info_->corr_absolute_mo("RESTRICTED_DOCC");
    actv_mos_ = mo_space_info_->corr_absolute_mo("ACTIVE");
    virt_mos_ = mo_space_info_->corr_absolute_mo("RESTRICTED_UOCC");

    // define space labels
    ambit::BlockedTensor::reset_mo_spaces();
    ambit::BlockedTensor::add_mo_space("c", "mn", core_mos_, NoSpin);
    ambit::BlockedTensor::add_mo_space("a", "uvwxyz", actv_mos_, NoSpin);
    ambit::BlockedTensor::add_mo_space("v", "ef", virt_mos_, NoSpin);

    // map space labels to mo spaces
    label_to_spacemo_['c'] = core_mos_;
    label_to_spacemo_['a'] = actv_mos_;
    label_to_spacemo_['v'] = virt_mos_;

    // define composite spaces
    ambit::BlockedTensor::add_composite_mo_space("h", "ijkl", {"c", "a"});
    ambit::BlockedTensor::add_composite_mo_space("p", "abcd", {"a", "v"});
    ambit::BlockedTensor::add_composite_mo_space("g", "pqrs", {"c", "a", "v"});

    // test memory
    test_memory(mo_space_info_->size("RESTRICTED_DOCC"), mo_space_info_->size("ACTIVE"),
                mo_space_info_->size("RESTRICTED_UOCC"));

    // prepare density matrix and cumulants
    L1_ = ambit::BlockedTensor::build(tensor_type_, "OPDC", {"aa"});
    Eta1_ = ambit::BlockedTensor::build(tensor_type_, "Eta1", {"aa"});
    L2_ = ambit::BlockedTensor::build(tensor_type_, "T2PDC", {"aaaa"});
    if (foptions_->get_str("THREEPDC") != "ZERO") {
        L3_ = ambit::BlockedTensor::build(tensor_type_, "T3PDC", {"aaaaaa"});
    }
    build_density();

    // prepare integrals
    // H does not contain ac, vc, va blocks
    // V does not contain ccvv or aaaa blocks
    H_ = ambit::BlockedTensor::build(tensor_type_, "H", {"cg", "ap", "vv"});
    std::vector<std::string> od_blocks = od_two_labels();
    std::vector<std::string> throw_blocks{"ccvv", "cavv", "acvv", "ccav", "ccva"};
    od_blocks.erase(std::remove_if(od_blocks.begin(), od_blocks.end(),
                                   [&](std::string i) -> bool {
                                       return std::find(throw_blocks.begin(), throw_blocks.end(),
                                                        i) != throw_blocks.end();
                                   }),
                    od_blocks.end());
    //    od_blocks.erase(std::remove(od_blocks.begin(), od_blocks.end(),
    //    "ccvv"), od_blocks.end());
    V_ = ambit::BlockedTensor::build(tensor_type_, "V", od_blocks);
    build_ints();

    // build Fock matrix
    F_ = ambit::BlockedTensor::build(tensor_type_, "Fock", {"cg", "ap", "vv"});
    build_fock();

    // test semicanonical
    semi_canonical_ = check_semicanonical();

    // initialize T amplitudes
    T1_ = ambit::BlockedTensor::build(tensor_type_, "T1", {"hv", "ca"});
    T2_ = ambit::BlockedTensor::build(tensor_type_, "T2", od_blocks);
    compute_T_1st(V_, T2_, F_, T1_);

    // initialize timer for commutator
    dsrg_time_ = DSRG_TIME();
}

double DSRG_MRPT::compute_energy() {
    // get reference energy
    double Etotal = Eref_;

    // compute energy
    if (corr_lv_ == "PT2") {
        Etotal += compute_energy_pt2();
    } else if (corr_lv_ == "PT3") {
        //        Etotal += compute_energy_pt3(); // TODO: throw if you reach here
    }

    psi::Process::environment.globals["CURRENT ENERGY"] = Etotal;
    return Etotal;
}

std::shared_ptr<ActiveSpaceIntegrals> DSRG_MRPT::compute_Heff_actv() {
    throw psi::PSIEXCEPTION(
        "Computing active-space Hamiltonian is not yet implemented for spin-adapted code.");

    return std::make_shared<ActiveSpaceIntegrals>(
        ints_, mo_space_info_->corr_absolute_mo("ACTIVE"), mo_space_info_->symmetry("ACTIVE"),
        mo_space_info_->corr_absolute_mo("RESTRICTED_DOCC"));
}

void DSRG_MRPT::build_ints() {
    // fill two-eletron integrals
    V_.iterate([&](const std::vector<size_t>& i, const std::vector<SpinType>&, double& value) {
        value = ints_->aptei_ab(i[0], i[1], i[2], i[3]);
    });

    // fill one-electron integrals
    H_.iterate([&](const std::vector<size_t>& i, const std::vector<SpinType>&, double& value) {
        value = ints_->oei_a(i[0], i[1]);
    });
}

void DSRG_MRPT::build_density() {
//    // test OPDC
//    ambit::Tensor diff = ambit::Tensor::build(tensor_type_, "diff_L1", rdms_.g1a().dims());
//    diff("pq") = rdms_.g1a()("pq") - rdms_.g1b()("pq");
//    if (diff.norm() > 1.0e-8) {
//        outfile->Printf("\n  Error: one-particle density cumulant cannot be spin-adapted!");
//        outfile->Printf("\n  |L1a - L1b| = %20.15f  <== This should be 0.0.", diff.norm());
//        throw psi::PSIEXCEPTION("One-particle density cumulant cannot be spin-adapted!");
//    }

    // fill spin-summed OPDC
    ambit::Tensor L1aa = L1_.block("aa");
//    L1aa("pq") = rdms_.g1a()("pq") + rdms_.g1b()("pq");
    L1aa("pq") = rdms_.SF_L1()("pq");

    ambit::Tensor E1aa = Eta1_.block("aa");
    E1aa.iterate(
        [&](const std::vector<size_t>& i, double& value) { value = i[0] == i[1] ? 2.0 : 0.0; });
    E1aa("pq") -= L1aa("pq");

//    // test T2PDC
//    diff = ambit::Tensor::build(tensor_type_, "diff_L2", rdms_.L2aa().dims());
//    diff("pqrs") = rdms_.L2aa()("pqrs") - rdms_.L2ab()("pqrs") + rdms_.L2ab()("pqsr");
//    if (diff.norm() > 1.0e-8) {
//        outfile->Printf("\n  Error: two-particle density cumulant cannot be spin-adapted!");
//        outfile->Printf("\n  |L2[pqrs] - (L2[pQrS] - L2[pQsR])| = %20.15f  <== "
//                        "This should be 0.0.",
//                        diff.norm());
//        throw psi::PSIEXCEPTION("Two-particle density cumulant cannot be spin-adapted!");
//    }

    // fill spin-summed T2PDC
    ambit::Tensor L2aa = L2_.block("aaaa");
//    L2aa("pqrs") += 4.0 * rdms_.L2ab()("pqrs");
//    L2aa("pqrs") -= 2.0 * rdms_.L2ab()("pqsr");
    L2aa("pqrs") = rdms_.SF_L2()("pqrs");

    // T3PDC
    if (foptions_->get_str("THREEPDC") != "ZERO") {
//        // test spin adaptation
//        diff = ambit::Tensor::build(tensor_type_, "diff_L3", rdms_.L3aaa().dims());
//        diff("pqrstu") +=
//            rdms_.L3aab()("pqrstu") - rdms_.L3aab()("pqrsut") + rdms_.L3aab()("pqrtus");
//        diff("pqrstu") -=
//            rdms_.L3aab()("prqstu") - rdms_.L3aab()("prqsut") + rdms_.L3aab()("prqtus");
//        diff("pqrstu") +=
//            rdms_.L3aab()("qrpstu") - rdms_.L3aab()("qrpsut") + rdms_.L3aab()("qrptus");
//        diff.scale(1.0 / 3.0);
//        diff("pqrstu") -= rdms_.L3aaa()("pqrstu");
//        if (diff.norm() > 1.0e-8) {
//            outfile->Printf("\n  Error: three-particle density cumulant cannot "
//                            "be spin-adapted!");
//            outfile->Printf("\n  |L3aaa - 1/3 * P(L3aab)| = %20.15f  <== This "
//                            "should be 0.0.",
//                            diff.norm());
//            throw psi::PSIEXCEPTION("Three-particle density cumulant cannot be spin-adapted!");
//        }

        // fill spin-summed T3PDC
        ambit::Tensor L3aaa = L3_.block("aaaaaa");
//        L3aaa("pqrstu") += rdms_.L3aaa()("pqrstu");
//        L3aaa("pqrstu") += rdms_.L3aab()("pqrstu");
//        L3aaa("pqrstu") += rdms_.L3aab()("prqsut");
//        L3aaa("pqrstu") += rdms_.L3aab()("qrptus");
//        L3aaa.scale(2.0);
        L3aaa("pqrstu") = rdms_.SF_L3()("pqrstu");
    }
}

void DSRG_MRPT::build_fock() {
    // copy F to T1
    for (const auto& block : H_.block_labels()) {
        F_.block(block)("pq") = H_.block(block)("pq");
    }

    // extra work here for not storing ccvv of V
    F_.iterate([&](const std::vector<size_t>& i, const std::vector<SpinType>&, double& value) {
        for (const size_t& m : core_mos_) {
            value += 2.0 * ints_->aptei_ab(i[0], m, i[1], m);
            value -= ints_->aptei_ab(i[0], m, m, i[1]);
        }
    });

    ambit::BlockedTensor VFock =
        ambit::BlockedTensor::build(tensor_type_, "VFock", {"caga", "aapa", "vava"});
    VFock.iterate([&](const std::vector<size_t>& i, const std::vector<SpinType>&, double& value) {
        value = ints_->aptei_ab(i[0], i[1], i[2], i[3]);
    });
    F_["mq"] += L1_["uv"] * VFock["mvqu"];
    F_["xa"] += L1_["uv"] * VFock["xvau"];
    F_["ef"] += L1_["uv"] * VFock["evfu"];

    VFock = ambit::BlockedTensor::build(tensor_type_, "VFock", {"caag", "aaap", "vaav"});
    VFock.iterate([&](const std::vector<size_t>& i, const std::vector<SpinType>&, double& value) {
        value = ints_->aptei_ab(i[0], i[1], i[2], i[3]);
    });
    F_["mq"] -= 0.5 * L1_["uv"] * VFock["mvuq"];
    F_["xa"] -= 0.5 * L1_["uv"] * VFock["xvua"];
    F_["ef"] -= 0.5 * L1_["uv"] * VFock["evuf"];

    // obtain diagonal elements of Fock matrix
    size_t ncmo = mo_space_info_->size("CORRELATED");
    Fdiag_ = std::vector<double>(ncmo);
    F_.citerate(
        [&](const std::vector<size_t>& i, const std::vector<SpinType>&, const double& value) {
            if (i[0] == i[1]) {
                Fdiag_[i[0]] = value;
            }
        });
}

bool DSRG_MRPT::check_semicanonical() {
    print_h2("Checking Orbitals");
    outfile->Printf("\n    Checking if orbitals are semi-canonicalized ...");
    std::vector<double> Fd_od_max;
    std::vector<double> Fd_od_norm;
    ambit::BlockedTensor diff =
        ambit::BlockedTensor::build(tensor_type_, "F - H0", {"cc", "aa", "vv"});
    diff["mn"] = F_["mn"];
    diff["uv"] = F_["uv"];
    diff["ef"] = F_["ef"];
    diff.iterate([&](const std::vector<size_t>& i, const std::vector<SpinType>&, double& value) {
        if (i[0] == i[1]) {
            value -= Fdiag_[i[0]];
        }
    });

    // compute norm and max values
    for (auto& block : {"cc", "aa", "vv"}) {
        ambit::Tensor diff_block = diff.block(block);
        Fd_od_norm.emplace_back(diff_block.norm(1));
        Fd_od_max.emplace_back(diff_block.norm(0));
    }

    double Fd_od_sum = std::accumulate(Fd_od_norm.begin(), Fd_od_norm.end(), 0.0);
    double threshold = 10.0 * foptions_->get_double("D_CONVERGENCE");
    bool semi = false;
    if (Fd_od_sum > threshold) {
        outfile->Printf("     NO.\n");
        std::string sep(4 + 16 * 3, '-');
        outfile->Printf("\n    Warning: orbitals are not semi-canonicalized!");
        outfile->Printf("\n    Off-diagonal elements of the core, active, "
                        "virtual blocks of Fock matrix");
        outfile->Printf("\n         %15s %15s %15s", "core", "active", "virtual");
        outfile->Printf("\n    %s", sep.c_str());
        outfile->Printf("\n    Max  %15.10f %15.10f %15.10f", Fd_od_max[0], Fd_od_max[1],
                        Fd_od_max[2]);
        outfile->Printf("\n    Norm %15.10f %15.10f %15.10f", Fd_od_norm[0], Fd_od_norm[1],
                        Fd_od_norm[2]);
        outfile->Printf("\n    %s\n", sep.c_str());
        outfile->Printf("\n    Warning: these elements above will be ignored!");
        outfile->Printf("\n    The DSRG-MRPT energy will make sense only when "
                        "these elements are small (< 1.0e-6).");
    } else {
        outfile->Printf("     OK.");
        outfile->Printf("\n    Orbitals are semi-canonicalized.");
        semi = true;
    }
    return semi;
}

void DSRG_MRPT::test_memory(const size_t& c, const size_t& a, const size_t& v) {
    size_t h = c + a;
    size_t p = a + v;
    size_t g = c + a + v;

    size_t gg = g * g;
    size_t hp = h * p;
    size_t hh = h * h;
    size_t cc = c * c;
    size_t aa = a * a;
    size_t vv = v * v;
    size_t hp_small = hp - aa;
    size_t gg_half = gg - p * c - v * a;

    size_t aaaa = aa * aa;
    size_t hhhh = hh * hh;
    size_t hhpp = hp * hp - aa * aa;
    size_t ccvv = cc * vv;
    size_t aavv = a * a * vv;
    size_t hhpp_small = hhpp - ccvv - 2 * c * a * vv - 2 * a * v * cc;
    if (ccvv < aavv) {
        ccvv = aavv;
    }

    // We store H[gg*], F[gg*], T1[hp*], V[hhpp_], T2[hhpp_], L1[aa], LH1[aa],
    // L2[aaaa], L3[aaaaaa]
    // gg* does not include ac, vc, va blocks.
    // hp* does not include aa block.
    // hhpp_ depends on if ccvv can be stored or not

    size_t total = psi::Process::environment.get_memory();
    size_t required;        // number of elements
    long long int leftover; // in bytes
    if (ref_relax_ == "NONE") {
        required = 2 * gg_half + hp_small + 2 * hhpp_small + 2 * aa + aaaa;
        leftover = total - sizeof(double) * required;
    } else {
        required = 2 * gg_half + hp_small + 2 * hhpp_small + hh + hhhh + 2 * aa + aaaa;
        leftover = total - sizeof(double) * required;
    }

    // consider L3
    if (foptions_->get_str("THREEPDC") != "ZERO") {
        required += static_cast<size_t>(aaaa * aa);
        leftover -= static_cast<size_t>(aaaa * aa * sizeof(double));
    }

    // compute number of batches
    if (leftover > 0) {
        nbatch_ = static_cast<int>((sizeof(double) * ccvv * 2) / leftover) + 1;
        required += static_cast<size_t>(2 * ccvv / nbatch_);
        leftover -= static_cast<size_t>(2 * ccvv * sizeof(double) / nbatch_);
    }

    std::map<std::string, double> to_XB;
    to_XB["B"] = 1.0;
    to_XB["KB"] = 1000.0; // use 1000.0 for safety
    to_XB["MB"] = 1000000.0;
    to_XB["GB"] = 1000000000.0;
    to_XB["TB"] = 1000000000000.0;

    // convert to appropriate unit
    auto converter = [&](size_t numel, bool mem = false) {
        size_t bytes;
        if (mem)
            bytes = numel;
        else
            bytes = numel * sizeof(double);
        std::string out;
        for (auto& XB : to_XB) {
            double xb = bytes / XB.second;
            if (xb >= 0.1 && xb < 100.0) {
                out = str(boost::format("%5.1f") % xb);
                out += (XB.first == "B" ? "  " : " ") + XB.first;
                break;
            }
        }
        return out;
    };

    std::vector<std::pair<std::string, std::string>> mem_summary{
        {"one-body (H, F half)", converter(2 * gg_half)},
        {"two-body (V small)", converter(hhpp_small)},
        {"T1 amp (no aa)", converter(hp_small)},
        {"T2 amp (small)", converter(hhpp_small)},
        {"V, T2 (ccvv)", converter(2 * ccvv)},
        {"L1, L2", converter(2 * aa + aaaa)},
        {"Hbar1 (relax ref.)", converter(hh)},
        {"Hbar2 (relax ref.)", converter(hhhh)},
        {"intermediate", "not decided"},
        {"memory required", converter(required)},
        {"memory available", converter(total, true)},
        {"memory leftover", converter(leftover, true)}};
    if (foptions_->get_str("THREEPDC") != "ZERO") {
        mem_summary.insert(mem_summary.begin() + 6, {"L3", converter(aa * aaaa)});
    }

    print_h2("Memory Summary");
    for (auto& str_dim : mem_summary) {
        outfile->Printf("\n    %-35s %15s", str_dim.first.c_str(), str_dim.second.c_str());
    }
    outfile->Printf("\n  Note: Two-index quantities: ONLY upper triangle (hp, cc, vv).");
    outfile->Printf("\n  Four-index quantities: NO aaaa, ccvv, cavv, acvv, "
                    "ccav, ccva blocks.");

    if (leftover < 0) {
        outfile->Printf("\n  Error: Not enough memory! Need %s of memory.",
                        converter(required).c_str());
        throw psi::PSIEXCEPTION("Not enough memory!");
    }

    if (nbatch_ > 1) {
        outfile->Printf("\n  Warning: not enough memory to store two ccvv terms.");
        outfile->Printf("\n  They will be computed using %3d batches.", nbatch_);
    }
}

std::vector<std::string> DSRG_MRPT::od_one_labels() {
    // according to HP
    std::vector<std::string> blocks1;
    for (const std::string& h : {"c", "a"}) {
        for (const std::string& p : {"a", "v"}) {
            if (h != "a" || p != "a") {
                blocks1.emplace_back(h + p);
            }
        }
    }
    return blocks1;
}

std::vector<std::string> DSRG_MRPT::od_two_labels() {
    // according to HHPP
    std::vector<std::string> blocks2;
    for (const std::string& h1 : {"c", "a"}) {
        for (const std::string& h2 : {"c", "a"}) {
            for (const std::string& p1 : {"a", "v"}) {
                for (const std::string& p2 : {"a", "v"}) {
                    blocks2.emplace_back(h1 + h2 + p1 + p2);
                }
            }
        }
    }
    blocks2.erase(std::remove(blocks2.begin(), blocks2.end(), "aaaa"), blocks2.end());
    return blocks2;
}

void DSRG_MRPT::print_citation() {
    print_h2("References");
    std::vector<std::pair<std::string, std::string>> papers{
        {"DSRG-MRPT2", "J. Chem. Theory Comput. 2015, 11, 2097."},
        {"DSRG-MRPT3", "J. Chem. Phys. (in preparation)"}};

    for (auto& str_dim : papers) {
        outfile->Printf("\n    %-20s %-60s", str_dim.first.c_str(), str_dim.second.c_str());
    }
    outfile->Printf("\n");
}
} // namespace forte
