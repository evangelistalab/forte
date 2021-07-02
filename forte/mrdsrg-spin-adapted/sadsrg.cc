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

#include <numeric>

#include "psi4/libmints/matrix.h"
#include "psi4/libmints/molecule.h"
#include "psi4/libpsi4util/process.h"
#include "psi4/libpsi4util/PsiOutStream.h"
#include "psi4/libmints/vector.h"
#include "psi4/libpsio/psio.hpp"

#include "forte-def.h"
#include "helpers/helpers.h"
#include "helpers/printing.h"
#include "helpers/timer.h"

#include "sadsrg.h"

using namespace psi;

namespace forte {

SADSRG::SADSRG(RDMs rdms, std::shared_ptr<SCFInfo> scf_info, std::shared_ptr<ForteOptions> options,
               std::shared_ptr<ForteIntegrals> ints, std::shared_ptr<MOSpaceInfo> mo_space_info)
    : DynamicCorrelationSolver(rdms, scf_info, options, ints, mo_space_info),
      BTF_(new BlockedTensorFactory()), tensor_type_(ambit::CoreTensor) {
    n_threads_ = omp_get_max_threads();
    std::string thread_title =
        std::to_string(n_threads_) + (n_threads_ > 1 ? " OMP threads" : " OMP thread");
    print_method_banner({"Spin-Adapted Multireference Driven Similarity Renormalization Group",
                         "written by Chenyang Li", thread_title});
    outfile->Printf("\n  Disclaimer:");
    outfile->Printf("\n    The spin-adapted DSRG code is largely adopted from the spin-integrated "
                    "code developed by");
    outfile->Printf(
        "\n    Chenyang Li, Kevin P. Hannon, Tianyuan Zhang, and Francesco A. Evangelista.");
    startup();
}

SADSRG::~SADSRG() {
    dsrg_time_.print_comm_time();

    if (warnings_.size() != 0) {
        print_h2("DSRG Warnings");

        outfile->Printf("\n  %32s  %32s  %32s", "Description", "This Run", "Solution");
        outfile->Printf("\n  %s", std::string(100, '-').c_str());
        for (auto& x : warnings_) {
            outfile->Printf("\n  %32s  %32s  %32s", std::get<0>(x), std::get<1>(x), std::get<2>(x));
        }
        outfile->Printf("\n  %s", std::string(100, '-').c_str());
        outfile->Printf("\n\n");
    }
}

void SADSRG::startup() {
    print_h2("Multireference Driven Similarity Renormalization Group");

    // build Fock and cleanup JK in ForteIntegrals
    build_fock_from_ints();

    // read options
    read_options();

    // read orbital spaces
    read_MOSpaceInfo();

    // set Ambit MO space labels
    set_ambit_MOSpace();

    // initialize timer for commutator
    dsrg_time_ = DSRG_TIME();

    // set memory variables
    check_init_memory();

    // prepare density matrix and cumulants
    init_density();

    // initialize Fock matrix
    init_fock();

    // recompute reference energy from ForteIntegral
    compute_reference_energy_from_ints();

    // general printing for all derived classes
    print_cumulant_summary();

    // initialize Uactv_ to identity
    Uactv_ = BTF_->build(tensor_type_, "Uactv", {"aa"});
    Uactv_.iterate([&](const std::vector<size_t>& i, const std::vector<SpinType>&, double& value) {
        if (i[0] == i[1]) {
            value = 1.0;
        }
    });

    // check if using semicanonical orbitals
    semi_canonical_ = check_semi_orbs();
}

void SADSRG::build_fock_from_ints() {
    local_timer lt;
    print_contents("Computing Fock matrix and cleaning JK");
    ints_->make_fock_matrix(rdms_.g1a(), rdms_.g1b());
    ints_->jk_finalize();
    print_done(lt.get());
}

void SADSRG::read_options() {
    local_timer lt;
    print_contents("Reading DSRG options");

    auto throw_error = [&](const std::string& message) -> void {
        outfile->Printf("\n  %s", message.c_str());
        throw psi::PSIEXCEPTION(message);
    };

    s_ = foptions_->get_double("DSRG_S");
    if (s_ < 0) {
        throw_error("S parameter for DSRG must >= 0!");
    }
    taylor_threshold_ = foptions_->get_int("TAYLOR_THRESHOLD");
    if (taylor_threshold_ <= 0) {
        throw_error("Threshold for Taylor expansion must be an integer greater than 0!");
    }

    source_ = foptions_->get_str("SOURCE");
    if (source_ != "STANDARD" && source_ != "LABS" && source_ != "DYSON") {
        outfile->Printf("\n  Warning: SOURCE option %s is not implemented.", source_.c_str());
        outfile->Printf("\n  Changed SOURCE option to STANDARD");
        source_ = "STANDARD";

        warnings_.push_back(std::make_tuple("Unsupported SOURCE", "Change to STANDARD",
                                            "Change options in input.dat"));
    }
    if (source_ == "STANDARD") {
        dsrg_source_ = std::make_shared<STD_SOURCE>(s_, taylor_threshold_);
    } else if (source_ == "LABS") {
        dsrg_source_ = std::make_shared<LABS_SOURCE>(s_, taylor_threshold_);
    } else if (source_ == "DYSON") {
        dsrg_source_ = std::make_shared<DYSON_SOURCE>(s_, taylor_threshold_);
    }
    ccvv_source_ = foptions_->get_str("CCVV_SOURCE");

    do_cu3_ = foptions_->get_str("THREEPDC") != "ZERO";

    ntamp_ = foptions_->get_int("NTAMP");
    intruder_tamp_ = foptions_->get_double("INTRUDER_TAMP");

    internal_amp_ = foptions_->get_str("INTERNAL_AMP");
    internal_amp_select_ = foptions_->get_str("INTERNAL_AMP_SELECT");

    dump_amps_cwd_ = foptions_->get_bool("DSRG_DUMP_AMPS");
    read_amps_cwd_ = foptions_->get_bool("DSRG_READ_AMPS");

    relax_ref_ = foptions_->get_str("RELAX_REF");

    multi_state_ = foptions_->get_gen_list("AVG_STATE").size() != 0;
    multi_state_algorithm_ = foptions_->get_str("DSRG_MULTI_STATE");

    print_done(lt.get());
}

void SADSRG::read_MOSpaceInfo() {
    core_mos_ = mo_space_info_->corr_absolute_mo("RESTRICTED_DOCC");
    actv_mos_ = mo_space_info_->corr_absolute_mo("ACTIVE");
    virt_mos_ = mo_space_info_->corr_absolute_mo("RESTRICTED_UOCC");
    actv_mos_sym_ = mo_space_info_->symmetry("ACTIVE");

    if (eri_df_) {
        aux_mos_ = std::vector<size_t>(ints_->nthree());
        std::iota(aux_mos_.begin(), aux_mos_.end(), 0);
    }
}

void SADSRG::set_ambit_MOSpace() {
    local_timer lt;
    print_contents("Setting ambit MO space");
    BlockedTensor::reset_mo_spaces();
    BlockedTensor::set_expert_mode(true);

    // define space labels
    core_label_ = "c";
    actv_label_ = "a";
    virt_label_ = "v";

    // add Ambit index labels
    BTF_->add_mo_space(core_label_, "m,n,c0,c1,c2,c3,c4,c5,c6,c7,c8,c9", core_mos_, NoSpin);
    BTF_->add_mo_space(actv_label_, "u,v,w,x,y,z,a0,a1,a2,a3,a4,a5,a6,a7,a8,a9", actv_mos_, NoSpin);
    BTF_->add_mo_space(virt_label_, "e,f,v0,v1,v2,v3,v4,v5,v6,v7,v8,v9", virt_mos_, NoSpin);

    // map space labels to mo spaces
    label_to_spacemo_[core_label_[0]] = core_mos_;
    label_to_spacemo_[actv_label_[0]] = actv_mos_;
    label_to_spacemo_[virt_label_[0]] = virt_mos_;

    // define composite spaces
    BTF_->add_composite_mo_space("h", "i,j,k,l,h0,h1,h2,h3,h4,h5,h6,h7,h8,h9",
                                 {core_label_, actv_label_});
    BTF_->add_composite_mo_space("p", "a,b,c,d,p0,p1,p2,p3,p4,p5,p6,p7,p8,p9",
                                 {actv_label_, virt_label_});
    BTF_->add_composite_mo_space("g", "p,q,r,s,t,o,g0,g1,g2,g3,g4,g5,g6,g7,g8,g9",
                                 {core_label_, actv_label_, virt_label_});

    // if DF/CD
    if (eri_df_) {
        aux_label_ = "L";
        BTF_->add_mo_space(aux_label_, "g", aux_mos_, NoSpin);
        label_to_spacemo_[aux_label_[0]] = aux_mos_;
    }

    print_done(lt.get());
}

void SADSRG::check_init_memory() {
    mem_sys_ = psi::Process::environment.get_memory();
    int64_t mem_left = mem_sys_ * 0.9;
    if (ints_->integral_type() != DiskDF and ints_->integral_type() != Cholesky) {
        mem_left -= ints_->jk()->memory_estimate() * sizeof(double);
    }

    // integrals already stored by the ForteIntegrals
    size_t n_ele = 0;
    auto ng = mo_space_info_->size("CORRELATED");
    if (eri_df_) {
        if (ints_->integral_type() != DiskDF) {
            auto nQ = aux_mos_.size();
            n_ele = nQ * ng * ng;
        }
    } else {
        n_ele = ng * ng * ng * ng;
    }

    // densities already stored by RDMs
    auto na = actv_mos_.size();
    n_ele += na * na + na * na * na * na;
    if (do_cu3_) {
        n_ele += na * na * na * na * na * na;
    }

    mem_left -= n_ele * sizeof(double);
    if (mem_left < 0) {
        std::stringstream ss;
        ss.precision(2);
        ss << "Not enough memory to run spin-adapted DSRG." << std::endl;
        auto pair = to_xb(-1.2 * mem_left, sizeof(double));
        ss << "Need at least " << std::fixed << pair.first << " " << pair.second
           << " more to pass the pre-DSRG memory check." << std::endl;
        ss << "Try DiskDF integrals?" << std::endl;
        outfile->Printf("\n%s", ss.str().c_str());
        throw psi::PSIEXCEPTION(ss.str());
    }

    // prepare DSRG_MEM
    std::map<char, size_t> label_to_size;
    label_to_size['c'] = core_mos_.size();
    label_to_size['a'] = na;
    label_to_size['v'] = virt_mos_.size();
    label_to_size['h'] = label_to_size['c'] + label_to_size['a'];
    label_to_size['p'] = label_to_size['v'] + label_to_size['a'];
    label_to_size['g'] = label_to_size['c'] + label_to_size['p'];
    if (eri_df_) {
        label_to_size['L'] = aux_mos_.size();
    }
    dsrg_mem_.set_mem_avai(mem_left);
    dsrg_mem_.set_label_to_size(label_to_size);

    dsrg_mem_.add_print_entry("Memory assigned by the user", mem_sys_);
    dsrg_mem_.add_print_entry("Memory available for MR-DSRG", mem_left);
    dsrg_mem_.add_entry("Generalized Fock matrix", {"g", "gg"});
    if (do_cu3_) {
        dsrg_mem_.add_entry("1-, 2-, and 3-density cumulants", {"aa", "aa", "aaaa", "aaaaaa"});
    } else {
        dsrg_mem_.add_entry("1- and 2-density cumulants", {"aa", "aa", "aaaa"});
    }
}

void SADSRG::init_density() {
    local_timer lt;
    print_contents("Initializing density cumulants");
    Eta1_ = BTF_->build(tensor_type_, "Eta1", {"aa"});
    L1_ = BTF_->build(tensor_type_, "L1", {"aa"});
    L2_ = BTF_->build(tensor_type_, "L2", {"aaaa"});
    fill_density();
    print_done(lt.get());
}

void SADSRG::fill_density() {
    // 1-particle density (make a copy)
    ambit::Tensor L1a = L1_.block("aa");
    L1a("pq") = rdms_.SF_L1()("pq");

    // 1-hole density
    ambit::Tensor E1a = Eta1_.block("aa");
    E1a.iterate(
        [&](const std::vector<size_t>& i, double& value) { value = i[0] == i[1] ? 2.0 : 0.0; });
    E1a("pq") -= L1a("pq");

    // 2-body density cumulants (make a copy)
    ambit::Tensor L2aa = L2_.block("aaaa");
    L2aa("pqrs") = rdms_.SF_L2()("pqrs");
}

void SADSRG::init_fock() {
    local_timer lt;
    print_contents("Filling Fock matrix from ForteIntegrals");
    Fock_ = BTF_->build(tensor_type_, "Fock", {"gg"});
    Fock_.iterate([&](const std::vector<size_t>& i, const std::vector<SpinType>&, double& value) {
        value = ints_->get_fock_a(i[0], i[1]);
    });
    fill_Fdiag(Fock_, Fdiag_);
    print_done(lt.get());
}

void SADSRG::fill_Fdiag(BlockedTensor& F, std::vector<double>& Fa) {
    size_t ncmo = mo_space_info_->size("CORRELATED");
    Fa.resize(ncmo);

    F.iterate([&](const std::vector<size_t>& i, const std::vector<SpinType>&, double& value) {
        if (i[0] == i[1]) {
            Fa[i[0]] = value;
        }
    });
}

double SADSRG::compute_reference_energy_from_ints() {
    BlockedTensor H = BTF_->build(tensor_type_, "OEI", {"cc", "aa"}, true);
    H.iterate([&](const std::vector<size_t>& i, const std::vector<SpinType>&, double& value) {
        value = ints_->oei_a(i[0], i[1]);
    });

    BlockedTensor V = BTF_->build(tensor_type_, "APEI", {"aaaa"}, true);
    V.block("aaaa")("prqs") =
        ints_->aptei_ab_block(actv_mos_, actv_mos_, actv_mos_, actv_mos_)("prqs");

    Eref_ = compute_reference_energy(H, Fock_, V);
    psi::Process::environment.globals["DSRG REFERENCE ENERGY"] = Eref_;
    return Eref_;
}

double SADSRG::compute_reference_energy(BlockedTensor H, BlockedTensor F, BlockedTensor V) {
    /// H: bare OEI; F: Fock; V: bare APTEI.
    /// Spin-orbital expression:
    /// E = 0.5 * ( H["ij"] + F["ij"] ) * L1["ji"] + 0.25 * V["xyuv"] * L2["uvxy"]
    /// Spin-adapted expression:
    /// E = 0.5 * ( H["ij"] + F["ij"] ) * L1["ji"] + 0.5 * V["xyuv"] * L2["uvxy"]
    /// Note that L1_mn = 2.0 * δ_mn now

    size_t ncore = core_mos_.size();
    double E = Efrzc_ + Enuc_;

    auto& H_cc = H.block("cc").data();
    auto& F_cc = F.block("cc").data();
    for (size_t m = 0; m < ncore; ++m) {
        E += H_cc[m * ncore + m];
        E += F_cc[m * ncore + m];
    }

    E += 0.5 * H["uv"] * L1_["vu"];
    E += 0.5 * F["uv"] * L1_["vu"];

    E += 0.5 * V["uvxy"] * L2_["xyuv"];

    return E;
}

void SADSRG::fill_three_index_ints(ambit::BlockedTensor B) {
    const auto& block_labels = B.block_labels();
    for (const std::string& string_block : block_labels) {
        auto mo_to_index = BTF_->get_mo_to_index();
        std::vector<size_t> first_index = mo_to_index[string_block.substr(0, 1)];
        std::vector<size_t> second_index = mo_to_index[string_block.substr(1, 1)];
        std::vector<size_t> third_index = mo_to_index[string_block.substr(2, 1)];
        ambit::Tensor block = ints_->three_integral_block(first_index, second_index, third_index);
        B.block(string_block).copy(block);
    }
}

std::shared_ptr<ActiveSpaceIntegrals> SADSRG::compute_Heff_actv() {
    // de-normal-order DSRG transformed Hamiltonian
    double Edsrg = Eref_ + Hbar0_;
    if (foptions_->get_bool("FORM_HBAR3")) {
        deGNO_ints("Hamiltonian", Edsrg, Hbar1_, Hbar2_, Hbar3_);
        rotate_ints_semi_to_origin("Hamiltonian", Hbar1_, Hbar2_, Hbar3_);
    } else {
        deGNO_ints("Hamiltonian", Edsrg, Hbar1_, Hbar2_);
        rotate_ints_semi_to_origin("Hamiltonian", Hbar1_, Hbar2_);
    }

    // create FCIIntegral shared_ptr
    auto fci_ints =
        std::make_shared<ActiveSpaceIntegrals>(ints_, actv_mos_, actv_mos_sym_, core_mos_);
    fci_ints->set_scalar_energy(Edsrg - Enuc_ - Efrzc_);
    fci_ints->set_restricted_one_body_operator(Hbar1_.block("aa").data(),
                                               Hbar1_.block("aa").data());

    auto Hbar2aa = Hbar2_.block("aaaa").clone();
    Hbar2aa("pqrs") -= Hbar2_.block("aaaa")("pqsr");
    fci_ints->set_active_integrals(Hbar2aa, Hbar2_.block("aaaa"), Hbar2aa);

    return fci_ints;
}

void SADSRG::deGNO_ints(const std::string& name, double& H0, BlockedTensor& H1, BlockedTensor& H2) {
    print_h2("De-Normal-Order DSRG Transformed " + name);

    // compute scalar
    local_timer t0;
    print_contents("Computing the scalar term");

    // build a temp["pqrs"] = 2 * H2["pqrs"] - H2["pqsr"]
    auto temp = H2.block("aaaa").clone();
    temp.scale(2.0);
    temp("pqrs") -= H2.block("aaaa")("pqsr");

    // scalar from H1
    double scalar1 = 0.0;
    scalar1 -= H1["vu"] * L1_["uv"];

    // scalar from H2
    double scalar2 = 0.0;
    ambit::Tensor L1a = L1_.block("aa");
    scalar2 += 0.25 * L1a("uv") * temp("vyux") * L1a("xy");

    scalar2 -= 0.5 * H2["xyuv"] * L2_["uvxy"];

    H0 += scalar1 + scalar2;
    print_done(t0.get());

    // compute 1-body term
    local_timer t1;
    print_contents("Computing the 1-body term");

    H1.block("aa")("uv") -= 0.5 * temp("uxvy") * L1a("yx");
    print_done(t1.get());
}

void SADSRG::deGNO_ints(const std::string& name, double& H0, BlockedTensor& H1, BlockedTensor& H2,
                        BlockedTensor& /*H3*/) {
    throw psi::PSIEXCEPTION("Not yet implemented when forming Hbar3.");

    print_h2("De-Normal-Order DSRG Transformed " + name);

    // build a temp["pqrs"] = 2 * H2["pqrs"] - H2["pqsr"]
    auto temp = H2.block("aaaa").clone();
    temp.scale(2.0);
    temp("pqrs") -= H2.block("aaaa")("pqsr");

    // compute scalar
    local_timer t0;
    print_contents("Computing the scalar term");

    // scalar from H1
    double scalar1 = 0.0;
    scalar1 -= H1["vu"] * L1_["uv"];

    // scalar from H2
    double scalar2 = 0.0;
    ambit::Tensor L1a = L1_.block("aa");
    scalar2 += 0.25 * L1a("uv") * temp("vyux") * L1a("xy");

    scalar2 -= 0.5 * H2["xyuv"] * L2_["uvxy"];

    // scalar from H3
    /**
     * Different equations are obtained depending on how H3 is handled.
     * If H3 is equivalent to aab spin case, then for L3 term becomes
     * scalar3 -= 1/36 * H3["pqrstu"] * (2 * A - B - C - 3 * D),
     * where A = L3["stupqr"], B = L3["stuqrp"], C = L3["sturpq"], D = L3["stuqpr"].
     */
    double scalar3 = 0.0;
    //    scalar3 -= (1.0 / 36.0) * H3.block("aaaaaa")("xyzuvw") * rdms_.L3aaa()("xyzuvw");
    //    scalar3 -= (1.0 / 36.0) * H3.block("AAAAAA")("XYZUVW") * rdms_.L3bbb()("XYZUVW");
    //    scalar3 -= 0.25 * H3.block("aaAaaA")("xyZuvW") * rdms_.L3aab()("xyZuvW");
    //    scalar3 -= 0.25 * H3.block("aAAaAA")("xYZuVW") * rdms_.L3abb()("xYZuVW");

    //    // TODO: form one-body intermediate for scalar and 1-body
    //    scalar3 += 0.25 * H3["xyzuvw"] * Lambda2_["uvxy"] * Gamma1_["wz"];
    //    scalar3 += 0.25 * H3["XYZUVW"] * Lambda2_["UVXY"] * Gamma1_["WZ"];
    //    scalar3 += 0.25 * H3["xyZuvW"] * Lambda2_["uvxy"] * Gamma1_["WZ"];
    //    scalar3 += H3["xzYuwV"] * Lambda2_["uVxY"] * Gamma1_["wz"];
    //    scalar3 += 0.25 * H3["zXYwUV"] * Lambda2_["UVXY"] * Gamma1_["wz"];
    //    scalar3 += H3["xZYuWV"] * Lambda2_["uVxY"] * Gamma1_["WZ"];

    //    scalar3 -= (1.0 / 6.0) * H3["xyzuvw"] * Gamma1_["ux"] * Gamma1_["vy"] * Gamma1_["wz"];
    //    scalar3 -= (1.0 / 6.0) * H3["XYZUVW"] * Gamma1_["UX"] * Gamma1_["VY"] * Gamma1_["WZ"];
    //    scalar3 -= 0.5 * H3["xyZuvW"] * Gamma1_["ux"] * Gamma1_["vy"] * Gamma1_["WZ"];
    //    scalar3 -= 0.5 * H3["xYZuVW"] * Gamma1_["ux"] * Gamma1_["VY"] * Gamma1_["WZ"];

    H0 += scalar1 + scalar2 + scalar3;
    print_done(t0.get());

    // compute 1-body term
    local_timer t1;
    print_contents("Computing the 1-body term");

    // 1-body from H2
    H1.block("aa")("uv") -= 0.5 * temp("uxvy") * L1a("yx");

    // 1-body from H3
    //    H1["uv"] += 0.5 * H3["uyzvxw"] * Gamma1_["xy"] * Gamma1_["wz"];
    //    H1["uv"] += 0.5 * H3["uYZvXW"] * Gamma1_["XY"] * Gamma1_["WZ"];
    //    H1["uv"] += H3["uyZvxW"] * Gamma1_["xy"] * Gamma1_["WZ"];

    //    H1["UV"] += 0.5 * H3["UYZVXW"] * Gamma1_["XY"] * Gamma1_["WZ"];
    //    H1["UV"] += 0.5 * H3["yzUxwV"] * Gamma1_["xy"] * Gamma1_["wz"];
    //    H1["UV"] += H3["yUZxVW"] * Gamma1_["xy"] * Gamma1_["WZ"];

    //    H1["uv"] -= 0.25 * H3["uxyvwz"] * Lambda2_["wzxy"];
    //    H1["uv"] -= 0.25 * H3["uXYvWZ"] * Lambda2_["WZXY"];
    //    H1["uv"] -= H3["uxYvwZ"] * Lambda2_["wZxY"];

    //    H1["UV"] -= 0.25 * H3["UXYVWZ"] * Lambda2_["WZXY"];
    //    H1["UV"] -= 0.25 * H3["xyUwzV"] * Lambda2_["wzxy"];
    //    H1["UV"] -= H3["xUYwVZ"] * Lambda2_["wZxY"];
    print_done(t1.get());

    // compute 2-body term
    local_timer t2;
    print_contents("Computing the 2-body term");
    //    H2["xyuv"] -= H3["xyzuvw"] * Gamma1_["wz"];
    //    H2["xyuv"] -= H3["xyZuvW"] * Gamma1_["WZ"];
    //    H2["xYuV"] -= H3["xYZuVW"] * Gamma1_["WZ"];
    //    H2["xYuV"] -= H3["xzYuwV"] * Gamma1_["wz"];
    //    H2["XYUV"] -= H3["XYZUVW"] * Gamma1_["WZ"];
    //    H2["XYUV"] -= H3["zXYwUV"] * Gamma1_["wz"];
    print_done(t2.get());
}

ambit::BlockedTensor SADSRG::deGNO_Tamp(BlockedTensor& T1, BlockedTensor& T2, BlockedTensor& D1) {
    BlockedTensor T1eff = BTF_->build(tensor_type_, "T1eff from de-GNO", {"hp"});

    T1eff["ia"] = T1["ia"];

    T1eff["ia"] -= 2.0 * T2["iuav"] * D1["vu"];
    T1eff["ia"] += T2["iuva"] * D1["vu"];

    return T1eff;
}

void SADSRG::set_Uactv(ambit::Tensor& U) {
    Uactv_ = BTF_->build(tensor_type_, "Uactv", {"aa"});
    Uactv_.block("aa")("pq") = U("pq");
}

void SADSRG::rotate_ints_semi_to_origin(const std::string& name, BlockedTensor& H1,
                                        BlockedTensor& H2) {
    print_h2("Rotate DSRG Transformed " + name + " back to Original Basis");
    ambit::Tensor temp;
    ambit::Tensor Ua = Uactv_.block("aa");

    local_timer timer;
    print_contents("Rotating 1-body term to original basis");
    temp = H1.block("aa").clone(tensor_type_);
    H1.block("aa")("pq") = Ua("pu") * temp("uv") * Ua("qv");
    print_done(timer.get());

    timer.reset();
    print_contents("Rotating 2-body term to original basis");
    temp = H2.block("aaaa").clone(tensor_type_);
    H2.block("aaaa")("pqrs") = Ua("pa") * Ua("qb") * temp("abcd") * Ua("rc") * Ua("sd");
    print_done(timer.get());
}

void SADSRG::rotate_ints_semi_to_origin(const std::string& name, BlockedTensor& H1,
                                        BlockedTensor& H2, BlockedTensor& H3) {
    print_h2("Rotate DSRG Transformed " + name + " back to Original Basis");
    ambit::Tensor temp;
    ambit::Tensor Ua = Uactv_.block("aa");

    local_timer timer;
    print_contents("Rotating 1-body term to original basis");
    temp = H1.block("aa").clone(tensor_type_);
    H1.block("aa")("pq") = Ua("pu") * temp("uv") * Ua("qv");
    print_done(timer.get());

    timer.reset();
    print_contents("Rotating 2-body term to original basis");
    temp = H2.block("aaaa").clone(tensor_type_);
    H2.block("aaaa")("pqrs") = Ua("pa") * Ua("qb") * temp("abcd") * Ua("rc") * Ua("sd");
    print_done(timer.get());

    timer.reset();
    print_contents("Rotating 3-body term to original basis");
    temp = H3.block("aaaaaa").clone(tensor_type_);
    H3.block("aaaaaa")("pqrstu") =
        Ua("pa") * Ua("qb") * Ua("rc") * temp("abcijk") * Ua("si") * Ua("tj") * Ua("uk");
    print_done(timer.get());
}

bool SADSRG::check_semi_orbs() {
    print_h2("Checking Semicanonical Orbitals");
    semi_checked_results_.clear();

    BlockedTensor Fd = BTF_->build(tensor_type_, "Fd", {"cc", "aa", "vv"});
    Fd["pq"] = Fock_["pq"];

    Fd.iterate([&](const std::vector<size_t>& i, const std::vector<SpinType>&, double& value) {
        if (i[0] == i[1]) {
            value = 0.0;
        }
    });

    bool semi = true;
    double e_conv = foptions_->get_double("E_CONVERGENCE");
    double cd = foptions_->get_double("CHOLESKY_TOLERANCE");
    e_conv = cd < e_conv ? e_conv : cd * 0.1;
    e_conv = e_conv < 1.0e-12 ? 1.0e-12 : e_conv;
    double threshold_max = 10.0 * e_conv;

    std::vector<std::tuple<std::string, double, double>> Fcheck;
    for (const auto& block : {"cc", "vv"}) {
        double fmax = Fd.block(block).norm(0);
        double fmean = Fd.block(block).norm(1);
        fmean /= Fd.block(block).numel() > 2 ? Fd.block(block).numel() : 1.0;

        std::string space = (block[0] == 'c') ? "RESTRICTED_DOCC" : "RESTRICTED_UOCC";
        semi_checked_results_[space] = fmax <= threshold_max and fmean <= e_conv;

        space = (block[0] == 'c') ? "CORE" : "VIRTUAL";
        Fcheck.emplace_back(space, fmax, fmean);
    }

    auto nactv = actv_mos_.size();
    for (const std::string& space : mo_space_info_->space_names()) {
        if (space.find("GAS") == std::string::npos or mo_space_info_->size(space) == 0)
            continue;

        auto rel_indices = mo_space_info_->pos_in_space(space, "ACTIVE");
        auto size = rel_indices.size();
        double fmax = 0.0, fmean = 0.0;

        for (size_t p = 0; p < size; ++p) {
            auto np = rel_indices[p];
            for (size_t q = p + 1; q < size; ++q) {
                auto nq = rel_indices[q];
                double v = std::fabs(Fd.block("aa").data()[np * nactv + nq]);
                if (v > fmax)
                    fmax = v;
                fmean += v;
            }
        }
        fmean /= size * size * 0.5; // roughly correct

        semi_checked_results_[space] = fmax <= threshold_max and fmean <= e_conv;

        Fcheck.emplace_back(space, fmax, fmean);
    }

    std::string dash(8 + 32, '-');
    outfile->Printf("\n    %-8s %15s %15s", "Block", "Max", "Mean");
    outfile->Printf("\n    %s", dash.c_str());
    for (const auto& Ftuple : Fcheck) {
        std::string space;
        double fmax, fmean;
        std::tie(space, fmax, fmean) = Ftuple;
        outfile->Printf("\n    %-8s %15.10f %15.10f", space.c_str(), fmax, fmean);
    }
    outfile->Printf("\n    %s", dash.c_str());

    for (const auto& pair : semi_checked_results_) {
        if (pair.second == false) {
            semi = false;
            break;
        }
    }

    if (semi) {
        outfile->Printf("\n    Orbitals are semi-canonicalized.");
    } else {
        outfile->Printf("\n    Warning! Orbitals are not semi-canonicalized!");
    }

    return semi;
}

void SADSRG::print_cumulant_summary() {
    print_h2("Density Cumulant Summary");

    std::vector<double> maxes(2), norms(2);

    maxes[0] = L2_.norm(0);
    norms[0] = L2_.norm(2);

    if (do_cu3_) {
        maxes[1] = rdms_.SF_L3().norm(0);
        norms[1] = rdms_.SF_L3().norm(2);
    } else {
        maxes[1] = 0.0;
        norms[1] = 0.0;
    }

    std::string dash(6 + 13 * 2, '-');
    outfile->Printf("\n    %-6s %12s %12s", "", "2-cumulant", "3-cumulant");
    outfile->Printf("\n    %s", dash.c_str());
    outfile->Printf("\n    %-6s %12.6f %12.6f", "max", maxes[0], maxes[1]);
    outfile->Printf("\n    %-6s %12.6f %12.6f", "2-norm", norms[0], norms[1]);
    outfile->Printf("\n    %s", dash.c_str());
}

std::vector<double> SADSRG::diagonalize_Fock_diagblocks(BlockedTensor& U) {
    // set U to identity and output diagonal Fock
    U.iterate([&](const std::vector<size_t>& i, const std::vector<SpinType>&, double& value) {
        if (i[0] == i[1]) {
            value = 1.0;
        }
    });
    std::vector<double> Fdiag(Fdiag_);

    // loop each correlated elementary space
    int nirrep = mo_space_info_->nirrep();

    auto elementary_spaces = mo_space_info_->composite_space_names()["CORRELATED"];
    for (const std::string& space : elementary_spaces) {
        if (mo_space_info_->size(space) == 0 or semi_checked_results_[space])
            continue;

        std::string block, composite_space;
        if (space.find("DOCC") != std::string::npos) {
            block = core_label_ + core_label_;
            composite_space = space;
        } else if (space.find("UOCC") != std::string::npos) {
            block = virt_label_ + virt_label_;
            composite_space = space;
        } else {
            block = actv_label_ + actv_label_;
            composite_space = "ACTIVE";
        }

        auto& Fdata = Fock_.block(block).data();
        auto indices = mo_space_info_->pos_in_space(space, composite_space);
        auto composite_size = mo_space_info_->size(composite_space);

        auto dim = mo_space_info_->dimension(space);
        auto Fd = std::make_shared<psi::Matrix>("F " + space, dim, dim);

        for (int h = 0, offset = 0; h < nirrep; ++h) {
            for (int p = 0; p < dim[h]; ++p) {
                auto np = indices[p + offset];
                for (int q = p; q < dim[h]; ++q) {
                    auto nq = indices[q + offset];
                    double v = Fdata[np * composite_size + nq];
                    Fd->set(h, p, q, v);
                    Fd->set(h, q, p, v);
                }
            }
            offset += dim[h];
        }

        auto Usub = std::make_shared<psi::Matrix>("U " + space, dim, dim);
        auto evals = std::make_shared<psi::Vector>("evals " + space, dim);
        Fd->diagonalize(Usub, evals);

        auto& Udata = U.block(block).data();
        auto corr_abs_indices = mo_space_info_->corr_absolute_mo(space);
        for (int h = 0, offset = 0; h < nirrep; ++h) {
            for (int p = 0; p < dim[h]; ++p) {
                auto np = indices[p + offset];
                for (int q = 0; q < dim[h]; ++q) {
                    auto nq = indices[q + offset];
                    Udata[nq * composite_size + np] = Usub->get(h, p, q); // row: new, column: old
                }
                Fdiag[corr_abs_indices[p + offset]] = evals->get(h, p);
            }
            offset += dim[h];
        }
    }

    return Fdiag;
}

void SADSRG::print_contents(const std::string& str, size_t size) {
    if (str.size() + 4 > size)
        size = str.size() + 4;
    std::string padding(size - str.size() - 1, '.');
    outfile->Printf("\n    %s %s", str.c_str(), padding.c_str());
}

void SADSRG::print_done(double t) { outfile->Printf(" Done. Timing %10.3f s", t); }
} // namespace forte
