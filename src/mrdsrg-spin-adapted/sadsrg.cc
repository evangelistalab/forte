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

#include <numeric>

#include "psi4/libpsi4util/process.h"
#include "psi4/libpsi4util/PsiOutStream.h"

#include "forte-def.h"
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
    Eref_ = compute_reference_energy_from_ints(ints_);
    psi::Process::environment.globals["DSRG REFERENCE ENERGY"] = Eref_;

    // initialize Uactv_ to identity
    Uactv_ = BTF_->build(tensor_type_, "Uactv", {"aa"});
    Uactv_.iterate([&](const std::vector<size_t>& i, const std::vector<SpinType>&, double& value) {
        if (i[0] == i[1]) {
            value = 1.0;
        }
    });

    // general printing for all derived classes
    print_cumulant_summary();

    // check if using semicanonical orbitals
    semi_canonical_ = check_semi_orbs();
}

void SADSRG::read_options() {
    outfile->Printf("\n    Reading DSRG options ............................ ");

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

    relax_ref_ = foptions_->get_str("RELAX_REF");

    multi_state_ = foptions_->get_gen_list("AVG_STATE").size() != 0;
    multi_state_algorithm_ = foptions_->get_str("DSRG_MULTI_STATE");

    outfile->Printf("Done");
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
    outfile->Printf("\n    Setting ambit MO space .......................... ");
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

    outfile->Printf("Done");
}

void SADSRG::check_init_memory() {
    mem_sys_ = psi::Process::environment.get_memory();
    auto mem_left = static_cast<int64_t>(0.98 * mem_sys_);

    // integrals already stored by the ForteIntegrals
    size_t n_ele = 0;
    auto ng = mo_space_info_->size("CORRELATED");
    if (eri_df_) {
        if (ints_type_ != "DISKDF") {
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
        throw psi::PSIEXCEPTION("Not enough memory to run FORTE.");
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
    outfile->Printf("\n    Preparing tensors for density cumulants ......... ");
    Eta1_ = BTF_->build(tensor_type_, "Eta1", {"aa"});
    L1_ = BTF_->build(tensor_type_, "L1", {"aa"});
    L2_ = BTF_->build(tensor_type_, "L2", {"aaaa"});
    fill_density();
    outfile->Printf("Done");
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
    outfile->Printf("\n    Building Fock matrix ............................ ");
    build_fock_from_ints(ints_, Fock_);
    fill_Fdiag(Fock_, Fdiag_);
    outfile->Printf("Done");
}

void SADSRG::build_fock_from_ints(std::shared_ptr<ForteIntegrals> ints, BlockedTensor& F) {
    size_t ncmo = mo_space_info_->size("CORRELATED");
    F = BTF_->build(tensor_type_, "Fock", {"gg"});

    // for convenience, directly call make_fock_matrix in ForteIntegral
    psi::SharedMatrix D1a(new psi::Matrix("D1a", ncmo, ncmo));
    for (size_t m = 0, ncore = core_mos_.size(); m < ncore; m++) {
        D1a->set(core_mos_[m], core_mos_[m], 1.0);
    }

    L1_.block("aa").citerate([&](const std::vector<size_t>& i, const double& value) {
        D1a->set(actv_mos_[i[0]], actv_mos_[i[1]], 0.5 * value);
    });

    ints->make_fock_matrix(D1a, D1a);

    F.iterate([&](const std::vector<size_t>& i, const std::vector<SpinType>&, double& value) {
        value = ints->get_fock_a(i[0], i[1]);
    });
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

double SADSRG::compute_reference_energy_from_ints(std::shared_ptr<ForteIntegrals> ints) {
    BlockedTensor H = BTF_->build(tensor_type_, "OEI", {"cc", "aa"}, true);
    H.iterate([&](const std::vector<size_t>& i, const std::vector<SpinType>&, double& value) {
        value = ints->oei_a(i[0], i[1]);
    });

    BlockedTensor V = BTF_->build(tensor_type_, "APEI", {"aaaa"}, true);
    V.block("aaaa")("prqs") =
        ints->aptei_ab_block(actv_mos_, actv_mos_, actv_mos_, actv_mos_)("prqs");

    // form Fock in batches
    // in cc block, only the diagonal elements are useful
    BlockedTensor F = BTF_->build(tensor_type_, "Fock pruned", {"cc", "aa"}, true);
    F["ij"] = H["ij"];
    F["uv"] += V["uyvx"] * L1_["xy"];
    F["uv"] -= 0.5 * V["uyxv"] * L1_["xy"];

    // an identity tensor of shape 1 * nc * 1 * nc for F["mm"] <- sum_{n} V["mnmn"]
    size_t nc = core_mos_.size();
    std::vector<size_t> Idims{1, nc, 1, nc};
    ambit::Tensor I = ambit::Tensor::build(tensor_type_, "I", Idims);
    I.iterate([&](const std::vector<size_t>& i, double& value) {
        if (i[1] == i[3]) {
            value = 1.0;
        }
    });

    // an temp tensor of shape 1 * 1
    ambit::Tensor O = ambit::Tensor::build(tensor_type_, "ONE", std::vector<size_t>{1, 1});
    O.data()[0] = 1.0;

    ambit::Tensor Vtemp;
    for (size_t m = 0; m < nc; ++m) {
        size_t nm = core_mos_[m];
        Vtemp = ints->aptei_ab_block({nm}, core_mos_, {nm}, core_mos_);
        F.block("cc").data()[m * nc + m] += 2.0 * Vtemp("pqrs") * I("pqrs");

        Vtemp = ints->aptei_ab_block({nm}, core_mos_, core_mos_, {nm});
        F.block("cc").data()[m * nc + m] -= Vtemp("pqrs") * I("pqsr");
    }

    Vtemp = ints->aptei_ab_block(core_mos_, actv_mos_, core_mos_, actv_mos_);
    F.block("cc")("pq") += Vtemp("prqs") * L1_.block("aa")("sr");
    F.block("aa")("pq") += 2.0 * Vtemp("rpsq") * I("isjr") * O("ij");

    Vtemp = ints->aptei_ab_block(core_mos_, actv_mos_, actv_mos_, core_mos_);
    F.block("cc")("pq") -= 0.5 * Vtemp("prsq") * L1_.block("aa")("sr");
    F.block("aa")("pq") -= Vtemp("rpqs") * I("isjr") * O("ij");

    return compute_reference_energy(H, F, V);
}

double SADSRG::compute_reference_energy(BlockedTensor H, BlockedTensor F, BlockedTensor V) {
    /// H: bare OEI; F: Fock; V: bare APTEI.
    /// Spin-orbital expression:
    /// E = 0.5 * ( H["ij"] + F["ij"] ) * L1["ji"] + 0.25 * V["xyuv"] * L2["uvxy"]
    /// Spin-adapted expression:
    /// E = 0.5 * ( H["ij"] + F["ij"] ) * L1["ji"] + 0.5 * V["xyuv"] * L2["uvxy"]
    /// Note that L1_mn = 2.0 * δ_mn now

    double E = Efrzc_ + Enuc_;

    for (size_t m = 0, nc = core_mos_.size(); m < nc; ++m) {
        E += H.block("cc").data()[m * nc + m];
        E += F.block("cc").data()[m * nc + m];
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
    outfile->Printf("\n    %-40s ... ", "Computing the scalar term");

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
    outfile->Printf("Done. Timing %8.3f s", t0.get());

    // compute 1-body term
    local_timer t1;
    outfile->Printf("\n    %-40s ... ", "Computing the 1-body term");

    H1.block("aa")("uv") -= 0.5 * temp("uxvy") * L1a("yx");
    outfile->Printf("Done. Timing %8.3f s", t1.get());
}

void SADSRG::deGNO_ints(const std::string& name, double& H0, BlockedTensor& H1, BlockedTensor& H2,
                        BlockedTensor& H3) {
    throw psi::PSIEXCEPTION("Not yet implemented when forming Hbar3.");

    print_h2("De-Normal-Order DSRG Transformed " + name);

    // build a temp["pqrs"] = 2 * H2["pqrs"] - H2["pqsr"]
    auto temp = H2.block("aaaa").clone();
    temp.scale(2.0);
    temp("pqrs") -= H2.block("aaaa")("pqsr");

    // compute scalar
    local_timer t0;
    outfile->Printf("\n    %-40s ... ", "Computing the scalar term");

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
    outfile->Printf("Done. Timing %8.3f s", t0.get());

    // compute 1-body term
    local_timer t1;
    outfile->Printf("\n    %-40s ... ", "Computing the 1-body term");

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
    outfile->Printf("Done. Timing %8.3f s", t1.get());

    // compute 2-body term
    local_timer t2;
    outfile->Printf("\n    %-40s ... ", "Computing the 2-body term");
    //    H2["xyuv"] -= H3["xyzuvw"] * Gamma1_["wz"];
    //    H2["xyuv"] -= H3["xyZuvW"] * Gamma1_["WZ"];
    //    H2["xYuV"] -= H3["xYZuVW"] * Gamma1_["WZ"];
    //    H2["xYuV"] -= H3["xzYuwV"] * Gamma1_["wz"];
    //    H2["XYUV"] -= H3["XYZUVW"] * Gamma1_["WZ"];
    //    H2["XYUV"] -= H3["zXYwUV"] * Gamma1_["wz"];
    outfile->Printf("Done. Timing %8.3f s", t2.get());
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
    outfile->Printf("\n    %-40s ... ", "Rotating 1-body term to original basis");
    temp = H1.block("aa").clone(tensor_type_);
    H1.block("aa")("pq") = Ua("pu") * temp("uv") * Ua("qv");
    outfile->Printf("Done. Timing %8.3f s", timer.get());

    local_timer timer2;
    outfile->Printf("\n    %-40s ... ", "Rotating 2-body term to original basis");
    temp = H2.block("aaaa").clone(tensor_type_);
    H2.block("aaaa")("pqrs") = Ua("pa") * Ua("qb") * temp("abcd") * Ua("rc") * Ua("sd");
    outfile->Printf("Done. Timing %8.3f s", timer2.get());
}

void SADSRG::rotate_ints_semi_to_origin(const std::string& name, BlockedTensor& H1,
                                        BlockedTensor& H2, BlockedTensor& H3) {
    print_h2("Rotate DSRG Transformed " + name + " back to Original Basis");
    ambit::Tensor temp;
    ambit::Tensor Ua = Uactv_.block("aa");

    local_timer timer;
    outfile->Printf("\n    %-40s ... ", "Rotating 1-body term to original basis");
    temp = H1.block("aa").clone(tensor_type_);
    H1.block("aa")("pq") = Ua("pu") * temp("uv") * Ua("qv");
    outfile->Printf("Done. Timing %8.3f s", timer.get());

    local_timer timer2;
    outfile->Printf("\n    %-40s ... ", "Rotating 2-body term to original basis");
    temp = H2.block("aaaa").clone(tensor_type_);
    H2.block("aaaa")("pqrs") = Ua("pa") * Ua("qb") * temp("abcd") * Ua("rc") * Ua("sd");
    outfile->Printf("Done. Timing %8.3f s", timer2.get());

    local_timer timer3;
    outfile->Printf("\n    %-40s ... ", "Rotating 3-body term to original basis");
    temp = H3.block("aaaaaa").clone(tensor_type_);
    H3.block("aaaaaa")("pqrstu") =
        Ua("pa") * Ua("qb") * Ua("rc") * temp("abcijk") * Ua("si") * Ua("tj") * Ua("uk");
    outfile->Printf("Done. Timing %8.3f s", timer3.get());
}

bool SADSRG::check_semi_orbs() {
    print_h2("Checking Semicanonical Orbitals");

    std::string actv_type = foptions_->get_str("FCIMO_ACTV_TYPE");
    if (actv_type == "CIS" || actv_type == "CISD") {
        std::string job_type = foptions_->get_str("CORRELATION_SOLVER");
        bool fci_mo = foptions_->get_str("ACTIVE_SPACE_SOLVER") == "CAS";
        if ((job_type == "MRDSRG" || job_type == "DSRG-MRPT3") && fci_mo) {
            std::stringstream ss;
            ss << "Unsupported FCIMO_ACTV_TYPE for " << job_type << " code.";
            throw psi::PSIEXCEPTION(ss.str());
        }

        outfile->Printf("\n    Incomplete active space %s is detected.", actv_type.c_str());
        outfile->Printf("\n    Please make sure Semicanonical class has been called.");
        outfile->Printf("\n    Abort checking semicanonical orbitals.");
        return true;
    }

    BlockedTensor Fd = BTF_->build(tensor_type_, "Fd", {"cc", "aa", "vv"});
    Fd["pq"] = Fock_["pq"];

    Fd.iterate([&](const std::vector<size_t>& i, const std::vector<SpinType>&, double& value) {
        if (i[0] == i[1]) {
            value = 0.0;
        }
    });

    bool semi = true;
    std::vector<double> Fmax, Fnorm;
    double e_conv = foptions_->get_double("E_CONVERGENCE");
    double cd = foptions_->get_double("CHOLESKY_TOLERANCE");
    e_conv = cd < e_conv ? e_conv : cd * 0.1;
    e_conv = e_conv < 1.0e-12 ? 1.0e-12 : e_conv;
    double threshold_max = 10.0 * e_conv;
    for (const auto& block : {"cc", "aa", "vv"}) {
        double fmax = Fd.block(block).norm(0);
        double fnorm = Fd.block(block).norm(1);
        Fmax.emplace_back(fmax);
        Fnorm.emplace_back(fnorm);

        if (fmax > threshold_max) {
            semi = false;
        }
        if (fnorm > Fd.block(block).numel() * e_conv) {
            semi = false;
        }
    }

    std::string dash(7 + 47, '-');
    outfile->Printf("\n    Fock core, active, virtual blocks (Fij, i != j)");
    outfile->Printf("\n    %6s %15s %15s %15s", "", "core", "active", "virtual");
    outfile->Printf("\n    %s", dash.c_str());
    outfile->Printf("\n    %6s %15.10f %15.10f %15.10f", "max", Fmax[0], Fmax[1], Fmax[2]);
    outfile->Printf("\n    %6s %15.10f %15.10f %15.10f", "1-norm", Fnorm[0], Fnorm[1], Fnorm[2]);
    outfile->Printf("\n    %s\n", dash.c_str());

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
    outfile->Printf("\n    %-6s %12.6f %12.6f", "norm", norms[0], norms[1]);
    outfile->Printf("\n    %s", dash.c_str());
}

std::vector<double> SADSRG::diagonalize_Fock_diagblocks(BlockedTensor& U) {
    // map MO space label to its psi::Dimension
    std::map<std::string, psi::Dimension> MOlabel_to_dimension;
    MOlabel_to_dimension[core_label_] = mo_space_info_->dimension("RESTRICTED_DOCC");
    MOlabel_to_dimension[actv_label_] = mo_space_info_->dimension("ACTIVE");
    MOlabel_to_dimension[virt_label_] = mo_space_info_->dimension("RESTRICTED_UOCC");

    // eigen values to be returned
    size_t ncmo = mo_space_info_->size("CORRELATED");
    psi::Dimension corr = mo_space_info_->dimension("CORRELATED");
    std::vector<double> eigenvalues(ncmo, 0.0);

    // map MO space label to its offset psi::Dimension
    std::map<std::string, psi::Dimension> MOlabel_to_offset_dimension;
    int nirrep = corr.n();
    MOlabel_to_offset_dimension[core_label_] = psi::Dimension(std::vector<int>(nirrep, 0));
    MOlabel_to_offset_dimension[actv_label_] = mo_space_info_->dimension("RESTRICTED_DOCC");
    MOlabel_to_offset_dimension[virt_label_] =
        mo_space_info_->dimension("RESTRICTED_DOCC") + mo_space_info_->dimension("ACTIVE");

    // figure out index
    auto fill_eigen = [&](std::string block_label, int irrep, std::vector<double> values) {
        int h = irrep;
        size_t idx_begin = 0;
        while ((--h) >= 0)
            idx_begin += corr[h];

        std::string label(1, tolower(block_label[0]));
        idx_begin += MOlabel_to_offset_dimension[label][irrep];

        size_t nvalues = values.size();
        for (size_t i = 0; i < nvalues; ++i) {
            eigenvalues[i + idx_begin] = values[i];
        }
    };

    // diagonalize diagonal blocks (C-A-V ordering)
    for (const auto& block : diag_one_labels()) {
        auto dims = Fock_.block(block).dims();
        size_t dim = dims[0];

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
                } else {
                    auto F_block = Fock_.block(block).clone();
                    auto F_h = separate_tensor(F_block, space, h);
                    U_h = ambit::Tensor::build(tensor_type_, "U_h", std::vector<size_t>(2, h_dim));
                    if (h_dim == 1) {
                        U_h.data()[0] = 1.0;
                        fill_eigen(block, h, F_h.data());
                    } else {
                        auto Feigen = F_h.syev(AscendingEigenvalue);
                        U_h("pq") = Feigen["eigenvectors"]("pq");
                        fill_eigen(block, h, Feigen["eigenvalues"].data());
                    }
                }

                ambit::Tensor U_out = U.block(block);
                combine_tensor(U_out, U_h, space, h);
            }
        }
    }
    return eigenvalues;
}

ambit::Tensor SADSRG::separate_tensor(ambit::Tensor& tens, const psi::Dimension& irrep,
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
    auto T_h = ambit::Tensor::build(tensor_type_, "T_h", std::vector<size_t>(2, h_dim));
    for (size_t i = 0; i < h_dim; ++i) {
        for (size_t j = 0; j < h_dim; ++j) {
            size_t abs_idx = rel_to_abs(i, j, offset);
            T_h.data()[i * h_dim + j] = tens.data()[abs_idx];
        }
    }

    return T_h;
}

void SADSRG::combine_tensor(ambit::Tensor& tens, ambit::Tensor& tens_h, const psi::Dimension& irrep,
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

void SADSRG::print_options_info(
    const std::string& title,
    const std::vector<std::pair<std::string, std::string>>& calculation_info_string,
    const std::vector<std::pair<std::string, double>>& calculation_info_double,
    const std::vector<std::pair<std::string, int>>& calculation_info_int) {
    print_h2(title);
    for (auto& str_dim : calculation_info_string) {
        outfile->Printf("\n    %-40s %15s", str_dim.first.c_str(), str_dim.second.c_str());
    }
    for (auto& str_dim : calculation_info_double) {
        outfile->Printf("\n    %-40s %15.3e", str_dim.first.c_str(), str_dim.second);
    }
    for (auto& str_dim : calculation_info_int) {
        outfile->Printf("\n    %-40s %15d", str_dim.first.c_str(), str_dim.second);
    }
    outfile->Printf("\n");
}
} // namespace forte
