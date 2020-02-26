#include <numeric>

#include "psi4/libpsi4util/process.h"
#include "psi4/libmints/molecule.h"
#include "psi4/libmints/dipole.h"

#include "helpers/printing.h"
#include "helpers/timer.h"

#include "sadsrg.h"

using namespace psi;

namespace forte {

SADSRG::SADSRG(RDMs rdms, std::shared_ptr<SCFInfo> scf_info, std::shared_ptr<ForteOptions> options,
               std::shared_ptr<ForteIntegrals> ints, std::shared_ptr<MOSpaceInfo> mo_space_info)
    : DynamicCorrelationSolver(rdms, scf_info, options, ints, mo_space_info),
      BTF_(new BlockedTensorFactory()), tensor_type_(ambit::CoreTensor) {
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
}

void SADSRG::read_options() {
    outfile->Printf("\n    Reading DSRG options ............................ ");

    auto throw_error = [&](const std::string& message) -> void {
        outfile->Printf("\n  %s", message.c_str());
        throw psi::PSIEXCEPTION(message);
    };

    print_ = foptions_->get_int("PRINT");

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

    ntamp_ = foptions_->get_int("NTAMP");
    intruder_tamp_ = foptions_->get_double("INTRUDER_TAMP");

    relax_ref_ = foptions_->get_str("RELAX_REF");

    eri_df_ = false;
    ints_type_ = foptions_->get_str("INT_TYPE");
    if (ints_type_ == "CHOLESKY" || ints_type_ == "DF" || ints_type_ == "DISKDF") {
        eri_df_ = true;
    }

    multi_state_ = foptions_->get_gen_list("AVG_STATE").size() != 0;
    multi_state_algorithm_ = foptions_->get_str("DSRG_MULTI_STATE");

    diis_start_ = foptions_->get_int("DSRG_DIIS_START");
    diis_freq_ = foptions_->get_int("DSRG_DIIS_FREQ");
    diis_min_vec_ = foptions_->get_int("DSRG_DIIS_MIN_VEC");
    diis_max_vec_ = foptions_->get_int("DSRG_DIIS_MAX_VEC");
    if (diis_min_vec_ < 1) {
        diis_min_vec_ = 1;
    }
    if (diis_max_vec_ <= diis_min_vec_) {
        diis_max_vec_ = diis_min_vec_ + 4;
    }
    if (diis_freq_ < 1) {
        diis_freq_ = 1;
    }

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

void SADSRG::fill_three_index_ints(ambit::BlockedTensor T) {
    const auto& block_labels = T.block_labels();
    for (const std::string& string_block : block_labels) {
        auto mo_to_index = BTF_->get_mo_to_index();
        std::vector<size_t> first_index = mo_to_index[string_block.substr(0, 1)];
        std::vector<size_t> second_index = mo_to_index[string_block.substr(1, 1)];
        std::vector<size_t> third_index = mo_to_index[string_block.substr(2, 1)];
        ambit::Tensor block = ints_->three_integral_block(first_index, second_index, third_index);
        T.block(string_block).copy(block);
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
    BlockedTensor T1eff = BTF_->build(tensor_type_, "T1eff from de-GNO", spin_cases({"hp"}));

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

    std::string dash(2 + 45, '-');
    outfile->Printf("\n    Abs. max of Fock core, active, virtual blocks (Fij, i != j)");
    outfile->Printf("\n    %15s %15s %15s", "core", "active", "virtual");
    outfile->Printf("\n    %s", dash.c_str());
    outfile->Printf("\n    %15.10f %15.10f %15.10f", Fmax[0], Fmax[1], Fmax[2]);
    outfile->Printf("\n    %s\n", dash.c_str());

    outfile->Printf("\n    1-Norm of Fock core, active, virtual blocks (Fij, i != j)");
    outfile->Printf("\n    %15s %15s %15s", "core", "active", "virtual");
    outfile->Printf("\n    %s", dash.c_str());
    outfile->Printf("\n    %15.10f %15.10f %15.10f", Fnorm[0], Fnorm[1], Fnorm[2]);
    outfile->Printf("\n    %s\n", dash.c_str());

    if (semi) {
        outfile->Printf("\n    Orbitals are semi-canonicalized.");
    } else {
        outfile->Printf("\n    Warning! Orbitals are not semi-canonicalized!");
    }

    return semi;
}

std::vector<std::string> SADSRG::diag_one_labels() {
    std::vector<std::string> labels;
    for (const std::string& p : {core_label_, actv_label_, virt_label_}) {
        labels.push_back(p + p);
    }
    return labels;
}

std::vector<std::string> SADSRG::od_one_labels_hp() {
    std::vector<std::string> labels;
    for (const std::string& p : {core_label_, actv_label_}) {
        for (const std::string& q : {actv_label_, virt_label_}) {
            if (p == actv_label_ && q == actv_label_) {
                continue;
            }
            labels.push_back(p + q);
        }
    }
    return labels;
}

std::vector<std::string> SADSRG::od_one_labels_ph() {
    std::vector<std::string> blocks1(od_one_labels_hp());
    for (auto& block : blocks1) {
        std::swap(block[0], block[1]);
    }
    return blocks1;
}

std::vector<std::string> SADSRG::od_one_labels() {
    std::vector<std::string> labels(od_one_labels_hp());
    std::vector<std::string> temp(od_one_labels_ph());
    labels.insert(std::end(labels), std::begin(temp), std::end(temp));
    return labels;
}

std::vector<std::string> SADSRG::od_two_labels_hhpp() {
    std::vector<std::string> labels;
    for (const std::string& p : {core_label_, actv_label_}) {
        for (const std::string& q : {core_label_, actv_label_}) {
            for (const std::string& r : {actv_label_, virt_label_}) {
                for (const std::string& s : {actv_label_, virt_label_}) {
                    if (p == actv_label_ && q == actv_label_ && r == actv_label_ &&
                        s == actv_label_) {
                        continue;
                    }
                    labels.push_back(p + q + r + s);
                }
            }
        }
    }
    return labels;
}

std::vector<std::string> SADSRG::od_two_labels_pphh() {
    std::vector<std::string> labels(od_two_labels_hhpp());
    for (auto& block : labels) {
        std::swap(block[0], block[2]);
        std::swap(block[1], block[3]);
    }
    return labels;
}

std::vector<std::string> SADSRG::od_two_labels() {
    std::vector<std::string> labels(od_two_labels_hhpp());
    std::vector<std::string> temp(od_two_labels_pphh());
    labels.insert(std::end(labels), std::begin(temp), std::end(temp));
    return labels;
}

std::vector<std::string> SADSRG::diag_two_labels() {
    std::vector<std::string> general{core_label_, actv_label_, virt_label_};

    std::vector<std::string> all;
    for (const std::string& p : general) {
        for (const std::string& q : general) {
            for (const std::string& r : general) {
                for (const std::string& s : general) {
                    all.push_back(p + q + r + s);
                }
            }
        }
    }

    std::vector<std::string> od(od_two_labels());
    std::sort(od.begin(), od.end());
    std::sort(all.begin(), all.end());

    std::vector<std::string> labels;
    std::set_symmetric_difference(all.begin(), all.end(), od.begin(), od.end(),
                                  std::back_inserter(labels));

    return labels;
}

std::vector<std::string> SADSRG::re_two_labels() {
    std::vector<std::vector<std::string>> half_labels{
        {core_label_ + core_label_},
        {actv_label_ + actv_label_},
        {virt_label_ + virt_label_},
        {core_label_ + actv_label_, actv_label_ + core_label_},
        {core_label_ + virt_label_, virt_label_ + core_label_},
        {actv_label_ + virt_label_, virt_label_ + actv_label_}};

    std::vector<std::string> labels;
    for (const auto& half : half_labels) {
        for (const std::string& half1 : half) {
            for (const std::string& half2 : half) {
                labels.push_back(half1 + half2);
            }
        }
    }

    return labels;
}

 void SADSRG::H1_T1_C0(BlockedTensor& H1, BlockedTensor& T1, const double& alpha, double& C0) {
    local_timer timer;

    double E = 0.0;
    E += 2.0 * H1["ma"] * T1["ma"];
    auto temp = ambit::BlockedTensor::build(tensor_type_, "Temp110", {"aa"});
    temp["uv"] += H1["ve"] * T1["ue"];
    temp["uv"] -= H1["mu"] * T1["mv"];
    E += L1_["vu"] * temp["uv"];

    E *= alpha;
    C0 += E;

    if (print_ > 2) {
        outfile->Printf("\n    Time for [H1, T1] -> C0 : %12.3f", timer.get());
    }
    dsrg_time_.add("110", timer.get());
}

 void SADSRG::H1_T2_C0(BlockedTensor& H1, BlockedTensor& T2, const double& alpha, double& C0) {
    local_timer timer;

    double E = 0.0;
    auto temp = ambit::BlockedTensor::build(tensor_type_, "Temp120", {"aaaa"});
    temp["uvxy"] += H1["xe"] * T2["uvey"];
    temp["uvxy"] -= H1["mv"] * T2["umxy"];
    temp["uvxy"] += H1["ye"] * T2["uvxe"];
    temp["uvxy"] -= H1["mu"] * T2["mvxy"];
    E += 0.5 * L2_["xyuv"] * temp["uvxy"];

    E *= alpha;
    C0 += E;

    if (print_ > 2) {
        outfile->Printf("\n    Time for [H1, T2] -> C0 : %12.3f", timer.get());
    }
    dsrg_time_.add("120", timer.get());
}

 void SADSRG::H2_T1_C0(BlockedTensor& H2, BlockedTensor& T1, const double& alpha, double& C0) {
    local_timer timer;

    double E = 0.0;
    auto temp = ambit::BlockedTensor::build(tensor_type_, "Temp120", {"aaaa"});
    temp["uvxy"] += H2["xyev"] * T1["ue"];
    temp["uvxy"] -= H2["myuv"] * T1["mx"];
    temp["uvxy"] += H2["xyue"] * T1["ve"];
    temp["uvxy"] -= H2["xmuv"] * T1["my"];
    E += 0.5 * L2_["xyuv"] * temp["uvxy"];

    E *= alpha;
    C0 += E;

    if (print_ > 2) {
        outfile->Printf("\n    Time for [H2, T1] -> C0 : %12.3f", timer.get());
    }
    dsrg_time_.add("210", timer.get());
}

// void SADSRG::H2_T2_C0(BlockedTensor& H2, BlockedTensor& T2, const double& alpha, double& C0) {
//    local_timer timer;

//    // <[Hbar2, T2]> (C_2)^4
//    double E = H2["eFmN"] * T2["mNeF"];
//    E += 0.25 * H2["efmn"] * T2["mnef"];
//    E += 0.25 * H2["EFMN"] * T2["MNEF"];

//    BlockedTensor temp = ambit::BlockedTensor::build(tensor_type_, "temp", spin_cases({"aa"}));
//    temp["vu"] += 0.5 * H2["efmu"] * T2["mvef"];
//    temp["vu"] += H2["fEuM"] * T2["vMfE"];
//    temp["VU"] += 0.5 * H2["EFMU"] * T2["MVEF"];
//    temp["VU"] += H2["eFmU"] * T2["mVeF"];
//    E += temp["vu"] * Gamma1_["uv"];
//    E += temp["VU"] * Gamma1_["UV"];

//    temp.zero();
//    temp["vu"] += 0.5 * H2["vemn"] * T2["mnue"];
//    temp["vu"] += H2["vEmN"] * T2["mNuE"];
//    temp["VU"] += 0.5 * H2["VEMN"] * T2["MNUE"];
//    temp["VU"] += H2["eVnM"] * T2["nMeU"];
//    E += temp["vu"] * Eta1_["uv"];
//    E += temp["VU"] * Eta1_["UV"];

//    temp = ambit::BlockedTensor::build(tensor_type_, "temp", spin_cases({"aaaa"}));
//    temp["yvxu"] += H2["efxu"] * T2["yvef"];
//    temp["yVxU"] += H2["eFxU"] * T2["yVeF"];
//    temp["YVXU"] += H2["EFXU"] * T2["YVEF"];
//    E += 0.25 * temp["yvxu"] * Gamma1_["xy"] * Gamma1_["uv"];
//    E += temp["yVxU"] * Gamma1_["UV"] * Gamma1_["xy"];
//    E += 0.25 * temp["YVXU"] * Gamma1_["XY"] * Gamma1_["UV"];

//    temp.zero();
//    temp["vyux"] += H2["vymn"] * T2["mnux"];
//    temp["vYuX"] += H2["vYmN"] * T2["mNuX"];
//    temp["VYUX"] += H2["VYMN"] * T2["MNUX"];
//    E += 0.25 * temp["vyux"] * Eta1_["uv"] * Eta1_["xy"];
//    E += temp["vYuX"] * Eta1_["uv"] * Eta1_["XY"];
//    E += 0.25 * temp["VYUX"] * Eta1_["UV"] * Eta1_["XY"];

//    temp.zero();
//    temp["vyux"] += H2["vemx"] * T2["myue"];
//    temp["vyux"] += H2["vExM"] * T2["yMuE"];
//    temp["VYUX"] += H2["eVmX"] * T2["mYeU"];
//    temp["VYUX"] += H2["VEXM"] * T2["YMUE"];
//    E += temp["vyux"] * Gamma1_["xy"] * Eta1_["uv"];
//    E += temp["VYUX"] * Gamma1_["XY"] * Eta1_["UV"];
//    temp["yVxU"] = H2["eVxM"] * T2["yMeU"];
//    E += temp["yVxU"] * Gamma1_["xy"] * Eta1_["UV"];
//    temp["vYuX"] = H2["vEmX"] * T2["mYuE"];
//    E += temp["vYuX"] * Gamma1_["XY"] * Eta1_["uv"];

//    temp.zero();
//    temp["yvxu"] += 0.5 * Gamma1_["wz"] * H2["vexw"] * T2["yzue"];
//    temp["yvxu"] += Gamma1_["WZ"] * H2["vExW"] * T2["yZuE"];
//    temp["yvxu"] += 0.5 * Eta1_["wz"] * T2["myuw"] * H2["vzmx"];
//    temp["yvxu"] += Eta1_["WZ"] * T2["yMuW"] * H2["vZxM"];
//    E += temp["yvxu"] * Gamma1_["xy"] * Eta1_["uv"];

//    temp["YVXU"] += 0.5 * Gamma1_["WZ"] * H2["VEXW"] * T2["YZUE"];
//    temp["YVXU"] += Gamma1_["wz"] * H2["eVwX"] * T2["zYeU"];
//    temp["YVXU"] += 0.5 * Eta1_["WZ"] * T2["MYUW"] * H2["VZMX"];
//    temp["YVXU"] += Eta1_["wz"] * H2["zVmX"] * T2["mYwU"];
//    E += temp["YVXU"] * Gamma1_["XY"] * Eta1_["UV"];

//    // <[Hbar2, T2]> C_4 (C_2)^2 HH -- combined with PH
//    temp = ambit::BlockedTensor::build(tensor_type_, "temp", spin_cases({"aaaa"}));
//    temp["uvxy"] += 0.125 * H2["uvmn"] * T2["mnxy"];
//    temp["uvxy"] += 0.25 * Gamma1_["wz"] * H2["uvmw"] * T2["mzxy"];
//    temp["uVxY"] += H2["uVmN"] * T2["mNxY"];
//    temp["uVxY"] += Gamma1_["wz"] * T2["zMxY"] * H2["uVwM"];
//    temp["uVxY"] += Gamma1_["WZ"] * H2["uVmW"] * T2["mZxY"];
//    temp["UVXY"] += 0.125 * H2["UVMN"] * T2["MNXY"];
//    temp["UVXY"] += 0.25 * Gamma1_["WZ"] * H2["UVMW"] * T2["MZXY"];

//    // <[Hbar2, T2]> C_4 (C_2)^2 PP -- combined with PH
//    temp["uvxy"] += 0.125 * H2["efxy"] * T2["uvef"];
//    temp["uvxy"] += 0.25 * Eta1_["wz"] * T2["uvew"] * H2["ezxy"];
//    temp["uVxY"] += H2["eFxY"] * T2["uVeF"];
//    temp["uVxY"] += Eta1_["wz"] * H2["zExY"] * T2["uVwE"];
//    temp["uVxY"] += Eta1_["WZ"] * T2["uVeW"] * H2["eZxY"];
//    temp["UVXY"] += 0.125 * H2["EFXY"] * T2["UVEF"];
//    temp["UVXY"] += 0.25 * Eta1_["WZ"] * T2["UVEW"] * H2["EZXY"];

//    // <[Hbar2, T2]> C_4 (C_2)^2 PH
//    temp["uvxy"] += H2["eumx"] * T2["mvey"];
//    temp["uvxy"] += H2["uExM"] * T2["vMyE"];
//    temp["uvxy"] += Gamma1_["wz"] * T2["zvey"] * H2["euwx"];
//    temp["uvxy"] += Gamma1_["WZ"] * H2["uExW"] * T2["vZyE"];
//    temp["uvxy"] += Eta1_["zw"] * H2["wumx"] * T2["mvzy"];
//    temp["uvxy"] += Eta1_["ZW"] * T2["vMyZ"] * H2["uWxM"];
//    E += temp["uvxy"] * Lambda2_["xyuv"];

//    temp["UVXY"] += H2["eUmX"] * T2["mVeY"];
//    temp["UVXY"] += H2["EUMX"] * T2["MVEY"];
//    temp["UVXY"] += Gamma1_["wz"] * T2["zVeY"] * H2["eUwX"];
//    temp["UVXY"] += Gamma1_["WZ"] * T2["ZVEY"] * H2["EUWX"];
//    temp["UVXY"] += Eta1_["zw"] * H2["wUmX"] * T2["mVzY"];
//    temp["UVXY"] += Eta1_["ZW"] * H2["WUMX"] * T2["MVZY"];
//    E += temp["UVXY"] * Lambda2_["XYUV"];

//    temp["uVxY"] += H2["uexm"] * T2["mVeY"];
//    temp["uVxY"] += H2["uExM"] * T2["MVEY"];
//    temp["uVxY"] -= H2["eVxM"] * T2["uMeY"];
//    temp["uVxY"] -= H2["uEmY"] * T2["mVxE"];
//    temp["uVxY"] += H2["eVmY"] * T2["umxe"];
//    temp["uVxY"] += H2["EVMY"] * T2["uMxE"];

//    temp["uVxY"] += Gamma1_["wz"] * T2["zVeY"] * H2["uexw"];
//    temp["uVxY"] += Gamma1_["WZ"] * T2["ZVEY"] * H2["uExW"];
//    temp["uVxY"] -= Gamma1_["WZ"] * H2["eVxW"] * T2["uZeY"];
//    temp["uVxY"] -= Gamma1_["wz"] * T2["zVxE"] * H2["uEwY"];
//    temp["uVxY"] += Gamma1_["wz"] * T2["zuex"] * H2["eVwY"];
//    temp["uVxY"] -= Gamma1_["WZ"] * H2["EVYW"] * T2["uZxE"];

//    temp["uVxY"] += Eta1_["zw"] * H2["wumx"] * T2["mVzY"];
//    temp["uVxY"] += Eta1_["ZW"] * T2["VMYZ"] * H2["uWxM"];
//    temp["uVxY"] -= Eta1_["zw"] * H2["wVxM"] * T2["uMzY"];
//    temp["uVxY"] -= Eta1_["ZW"] * T2["mVxZ"] * H2["uWmY"];
//    temp["uVxY"] += Eta1_["zw"] * T2["umxz"] * H2["wVmY"];
//    temp["uVxY"] += Eta1_["ZW"] * H2["WVMY"] * T2["uMxZ"];
//    E += temp["uVxY"] * Lambda2_["xYuV"];

//    // <[Hbar2, T2]> C_6 C_2
//    if (foptions_->get_str("THREEPDC") != "ZERO") {
//        temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"aaaaaa"});
//        temp["uvwxyz"] += H2["uviz"] * T2["iwxy"];
//        temp["uvwxyz"] += H2["waxy"] * T2["uvaz"];
//        E += 0.25 * temp.block("aaaaaa")("uvwxyz") * rdms_.L3aaa()("xyzuvw");

//        temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"AAAAAA"});
//        temp["UVWXYZ"] += H2["UVIZ"] * T2["IWXY"];
//        temp["UVWXYZ"] += H2["WAXY"] * T2["UVAZ"];
//        E += 0.25 * temp.block("AAAAAA")("UVWXYZ") * rdms_.L3bbb()("XYZUVW");

//        temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"aaAaaA"});
//        temp["uvWxyZ"] -= H2["uviy"] * T2["iWxZ"];
//        temp["uvWxyZ"] -= H2["uWiZ"] * T2["ivxy"];
//        temp["uvWxyZ"] += 2.0 * H2["uWyI"] * T2["vIxZ"];

//        temp["uvWxyZ"] += H2["aWxZ"] * T2["uvay"];
//        temp["uvWxyZ"] -= H2["vaxy"] * T2["uWaZ"];
//        temp["uvWxyZ"] -= 2.0 * H2["vAxZ"] * T2["uWyA"];
//        E += 0.5 * temp.block("aaAaaA")("uvWxyZ") * rdms_.L3aab()("xyZuvW");

//        temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"aAAaAA"});
//        temp["uVWxYZ"] -= H2["VWIZ"] * T2["uIxY"];
//        temp["uVWxYZ"] -= H2["uVxI"] * T2["IWYZ"];
//        temp["uVWxYZ"] += 2.0 * H2["uViZ"] * T2["iWxY"];

//        temp["uVWxYZ"] += H2["uAxY"] * T2["VWAZ"];
//        temp["uVWxYZ"] -= H2["WAYZ"] * T2["uVxA"];
//        temp["uVWxYZ"] -= 2.0 * H2["aWxY"] * T2["uVaZ"];
//        E += 0.5 * temp.block("aAAaAA")("uVWxYZ") * rdms_.L3abb()("xYZuVW");
//    }

//    // multiply prefactor and copy to C0
//    E *= alpha;
//    C0 += E;

//    if (print_ > 2) {
//        outfile->Printf("\n    Time for [H2, T2] -> C0 : %12.3f", timer.get());
//    }
//    dsrg_time_.add("220", timer.get());
//}

// void SADSRG::H1_T1_C1(BlockedTensor& H1, BlockedTensor& T1, const double& alpha,
//                      BlockedTensor& C1) {
//    local_timer timer;

//    C1["ip"] += alpha * H1["ap"] * T1["ia"];
//    C1["qa"] -= alpha * T1["ia"] * H1["qi"];
//    C1["IP"] += alpha * H1["AP"] * T1["IA"];
//    C1["QA"] -= alpha * T1["IA"] * H1["QI"];

//    if (print_ > 2) {
//        outfile->Printf("\n    Time for [H1, T1] -> C1 : %12.3f", timer.get());
//    }
//    dsrg_time_.add("111", timer.get());
//}

// void SADSRG::H1_T2_C1(BlockedTensor& H1, BlockedTensor& T2, const double& alpha,
//                      BlockedTensor& C1) {
//    local_timer timer;

//    C1["ia"] += alpha * H1["bm"] * T2["imab"];
//    C1["ia"] += alpha * H1["bu"] * Gamma1_["uv"] * T2["ivab"];
//    C1["ia"] -= alpha * H1["vj"] * Gamma1_["uv"] * T2["ijau"];
//    C1["ia"] += alpha * H1["BM"] * T2["iMaB"];
//    C1["ia"] += alpha * H1["BU"] * Gamma1_["UV"] * T2["iVaB"];
//    C1["ia"] -= alpha * H1["VJ"] * Gamma1_["UV"] * T2["iJaU"];

//    C1["IA"] += alpha * H1["bm"] * T2["mIbA"];
//    C1["IA"] += alpha * H1["bu"] * Gamma1_["uv"] * T2["vIbA"];
//    C1["IA"] -= alpha * H1["vj"] * Gamma1_["uv"] * T2["jIuA"];
//    C1["IA"] += alpha * H1["BM"] * T2["IMAB"];
//    C1["IA"] += alpha * H1["BU"] * Gamma1_["UV"] * T2["IVAB"];
//    C1["IA"] -= alpha * H1["VJ"] * Gamma1_["UV"] * T2["IJAU"];

//    if (print_ > 2) {
//        outfile->Printf("\n    Time for [H1, T2] -> C1 : %12.3f", timer.get());
//    }
//    dsrg_time_.add("121", timer.get());
//}

// void SADSRG::H2_T1_C1(BlockedTensor& H2, BlockedTensor& T1, const double& alpha,
//                      BlockedTensor& C1) {
//    local_timer timer;

//    C1["qp"] += alpha * T1["ma"] * H2["qapm"];
//    C1["qp"] += alpha * T1["xe"] * Gamma1_["yx"] * H2["qepy"];
//    C1["qp"] -= alpha * T1["mu"] * Gamma1_["uv"] * H2["qvpm"];
//    C1["qp"] += alpha * T1["MA"] * H2["qApM"];
//    C1["qp"] += alpha * T1["XE"] * Gamma1_["YX"] * H2["qEpY"];
//    C1["qp"] -= alpha * T1["MU"] * Gamma1_["UV"] * H2["qVpM"];

//    C1["QP"] += alpha * T1["ma"] * H2["aQmP"];
//    C1["QP"] += alpha * T1["xe"] * Gamma1_["yx"] * H2["eQyP"];
//    C1["QP"] -= alpha * T1["mu"] * Gamma1_["uv"] * H2["vQmP"];
//    C1["QP"] += alpha * T1["MA"] * H2["QAPM"];
//    C1["QP"] += alpha * T1["XE"] * Gamma1_["YX"] * H2["QEPY"];
//    C1["QP"] -= alpha * T1["MU"] * Gamma1_["UV"] * H2["QVPM"];

//    if (print_ > 2) {
//        outfile->Printf("\n    Time for [H2, T1] -> C1 : %12.3f", timer.get());
//    }
//    dsrg_time_.add("211", timer.get());
//}

// void SADSRG::H2_T2_C1(BlockedTensor& H2, BlockedTensor& T2, const double& alpha,
//                      BlockedTensor& C1) {
//    local_timer timer;
//    BlockedTensor temp;

//    /// max intermediate: a * a * p * p

//    // [Hbar2, T2] (C_2)^3 -> C1 particle contractions
//    C1["ir"] += 0.5 * alpha * H2["abrm"] * T2["imab"];
//    C1["ir"] += alpha * H2["aBrM"] * T2["iMaB"];
//    C1["IR"] += 0.5 * alpha * H2["ABRM"] * T2["IMAB"];
//    C1["IR"] += alpha * H2["aBmR"] * T2["mIaB"];

//    C1["ir"] += 0.5 * alpha * Gamma1_["uv"] * T2["ivab"] * H2["abru"];
//    C1["ir"] += alpha * Gamma1_["UV"] * T2["iVaB"] * H2["aBrU"];
//    C1["IR"] += 0.5 * alpha * Gamma1_["UV"] * T2["IVAB"] * H2["ABRU"];
//    C1["IR"] += alpha * Gamma1_["uv"] * T2["vIaB"] * H2["aBuR"];

//    C1["ir"] += 0.5 * alpha * T2["ijux"] * Gamma1_["xy"] * Gamma1_["uv"] * H2["vyrj"];
//    C1["IR"] += 0.5 * alpha * T2["IJUX"] * Gamma1_["XY"] * Gamma1_["UV"] * H2["VYRJ"];
//    temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"hHaA"});
//    temp["iJvY"] = T2["iJuX"] * Gamma1_["XY"] * Gamma1_["uv"];
//    C1["ir"] += alpha * temp["iJvY"] * H2["vYrJ"];
//    C1["IR"] += alpha * temp["jIvY"] * H2["vYjR"];

//    C1["ir"] -= alpha * Gamma1_["uv"] * T2["imub"] * H2["vbrm"];
//    C1["ir"] -= alpha * Gamma1_["uv"] * T2["iMuB"] * H2["vBrM"];
//    C1["ir"] -= alpha * Gamma1_["UV"] * T2["iMbU"] * H2["bVrM"];
//    C1["IR"] -= alpha * Gamma1_["UV"] * T2["IMUB"] * H2["VBRM"];
//    C1["IR"] -= alpha * Gamma1_["UV"] * T2["mIbU"] * H2["bVmR"];
//    C1["IR"] -= alpha * Gamma1_["uv"] * T2["mIuB"] * H2["vBmR"];

//    C1["ir"] -= alpha * T2["iyub"] * Gamma1_["uv"] * Gamma1_["xy"] * H2["vbrx"];
//    C1["ir"] -= alpha * T2["iYuB"] * Gamma1_["uv"] * Gamma1_["XY"] * H2["vBrX"];
//    C1["ir"] -= alpha * T2["iYbU"] * Gamma1_["XY"] * Gamma1_["UV"] * H2["bVrX"];
//    C1["IR"] -= alpha * T2["IYUB"] * Gamma1_["UV"] * Gamma1_["XY"] * H2["VBRX"];
//    C1["IR"] -= alpha * T2["yIuB"] * Gamma1_["uv"] * Gamma1_["xy"] * H2["vBxR"];
//    C1["IR"] -= alpha * T2["yIbU"] * Gamma1_["UV"] * Gamma1_["xy"] * H2["bVxR"];

//    // [Hbar2, T2] (C_2)^3 -> C1 hole contractions
//    C1["pa"] -= 0.5 * alpha * H2["peij"] * T2["ijae"];
//    C1["pa"] -= alpha * H2["pEiJ"] * T2["iJaE"];
//    C1["PA"] -= 0.5 * alpha * H2["PEIJ"] * T2["IJAE"];
//    C1["PA"] -= alpha * H2["ePiJ"] * T2["iJeA"];

//    C1["pa"] -= 0.5 * alpha * Eta1_["uv"] * T2["ijau"] * H2["pvij"];
//    C1["pa"] -= alpha * Eta1_["UV"] * T2["iJaU"] * H2["pViJ"];
//    C1["PA"] -= 0.5 * alpha * Eta1_["UV"] * T2["IJAU"] * H2["PVIJ"];
//    C1["PA"] -= alpha * Eta1_["uv"] * T2["iJuA"] * H2["vPiJ"];

//    C1["pa"] -= 0.5 * alpha * T2["vyab"] * Eta1_["uv"] * Eta1_["xy"] * H2["pbux"];
//    C1["PA"] -= 0.5 * alpha * T2["VYAB"] * Eta1_["UV"] * Eta1_["XY"] * H2["PBUX"];
//    temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"aApP"});
//    temp["uXaB"] = T2["vYaB"] * Eta1_["uv"] * Eta1_["XY"];
//    C1["pa"] -= alpha * H2["pBuX"] * temp["uXaB"];
//    C1["PA"] -= alpha * H2["bPuX"] * temp["uXbA"];

//    C1["pa"] += alpha * Eta1_["uv"] * T2["vjae"] * H2["peuj"];
//    C1["pa"] += alpha * Eta1_["uv"] * T2["vJaE"] * H2["pEuJ"];
//    C1["pa"] += alpha * Eta1_["UV"] * T2["jVaE"] * H2["pEjU"];
//    C1["PA"] += alpha * Eta1_["UV"] * T2["VJAE"] * H2["PEUJ"];
//    C1["PA"] += alpha * Eta1_["uv"] * T2["vJeA"] * H2["ePuJ"];
//    C1["PA"] += alpha * Eta1_["UV"] * T2["jVeA"] * H2["ePjU"];

//    C1["pa"] += alpha * T2["vjax"] * Eta1_["uv"] * Eta1_["xy"] * H2["pyuj"];
//    C1["pa"] += alpha * T2["vJaX"] * Eta1_["uv"] * Eta1_["XY"] * H2["pYuJ"];
//    C1["pa"] += alpha * T2["jVaX"] * Eta1_["XY"] * Eta1_["UV"] * H2["pYjU"];
//    C1["PA"] += alpha * T2["VJAX"] * Eta1_["UV"] * Eta1_["XY"] * H2["PYUJ"];
//    C1["PA"] += alpha * T2["vJxA"] * Eta1_["uv"] * Eta1_["xy"] * H2["yPuJ"];
//    C1["PA"] += alpha * T2["jVxA"] * Eta1_["UV"] * Eta1_["xy"] * H2["yPjU"];

//    // [Hbar2, T2] C_4 C_2 2:2 -> C1
//    C1["ir"] += 0.25 * alpha * T2["ijxy"] * Lambda2_["xyuv"] * H2["uvrj"];
//    C1["IR"] += 0.25 * alpha * T2["IJXY"] * Lambda2_["XYUV"] * H2["UVRJ"];
//    temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"hHaA"});
//    temp["iJuV"] = T2["iJxY"] * Lambda2_["xYuV"];
//    C1["ir"] += alpha * H2["uVrJ"] * temp["iJuV"];
//    C1["IR"] += alpha * H2["uVjR"] * temp["jIuV"];

//    C1["pa"] -= 0.25 * alpha * Lambda2_["xyuv"] * T2["uvab"] * H2["pbxy"];
//    C1["PA"] -= 0.25 * alpha * Lambda2_["XYUV"] * T2["UVAB"] * H2["PBXY"];
//    temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"aApP"});
//    temp["xYaB"] = T2["uVaB"] * Lambda2_["xYuV"];
//    C1["pa"] -= alpha * H2["pBxY"] * temp["xYaB"];
//    C1["PA"] -= alpha * H2["bPxY"] * temp["xYbA"];

//    C1["ir"] -= alpha * Lambda2_["yXuV"] * T2["iVyA"] * H2["uArX"];
//    C1["IR"] -= alpha * Lambda2_["xYvU"] * T2["vIaY"] * H2["aUxR"];
//    C1["pa"] += alpha * Lambda2_["xYvU"] * T2["vIaY"] * H2["pUxI"];
//    C1["PA"] += alpha * Lambda2_["yXuV"] * T2["iVyA"] * H2["uPiX"];

//    temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"hapa"});
//    temp["ixau"] += Lambda2_["xyuv"] * T2["ivay"];
//    temp["ixau"] += Lambda2_["xYuV"] * T2["iVaY"];
//    C1["ir"] += alpha * temp["ixau"] * H2["aurx"];
//    C1["pa"] -= alpha * H2["puix"] * temp["ixau"];
//    temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"hApA"});
//    temp["iXaU"] += Lambda2_["XYUV"] * T2["iVaY"];
//    temp["iXaU"] += Lambda2_["yXvU"] * T2["ivay"];
//    C1["ir"] += alpha * temp["iXaU"] * H2["aUrX"];
//    C1["pa"] -= alpha * H2["pUiX"] * temp["iXaU"];
//    temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"aHaP"});
//    temp["xIuA"] += Lambda2_["xyuv"] * T2["vIyA"];
//    temp["xIuA"] += Lambda2_["xYuV"] * T2["VIYA"];
//    C1["IR"] += alpha * temp["xIuA"] * H2["uAxR"];
//    C1["PA"] -= alpha * H2["uPxI"] * temp["xIuA"];
//    temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"HAPA"});
//    temp["IXAU"] += Lambda2_["XYUV"] * T2["IVAY"];
//    temp["IXAU"] += Lambda2_["yXvU"] * T2["vIyA"];
//    C1["IR"] += alpha * temp["IXAU"] * H2["AURX"];
//    C1["PA"] -= alpha * H2["PUIX"] * temp["IXAU"];

//    // [Hbar2, T2] C_4 C_2 1:3 -> C1
//    temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"pa"});
//    temp["au"] += 0.5 * Lambda2_["xyuv"] * H2["avxy"];
//    temp["au"] += Lambda2_["xYuV"] * H2["aVxY"];
//    C1["jb"] += alpha * temp["au"] * T2["ujab"];
//    C1["JB"] += alpha * temp["au"] * T2["uJaB"];
//    temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"PA"});
//    temp["AU"] += 0.5 * Lambda2_["XYUV"] * H2["AVXY"];
//    temp["AU"] += Lambda2_["xYvU"] * H2["vAxY"];
//    C1["jb"] += alpha * temp["AU"] * T2["jUbA"];
//    C1["JB"] += alpha * temp["AU"] * T2["UJAB"];

//    temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"ah"});
//    temp["xi"] += 0.5 * Lambda2_["xyuv"] * H2["uviy"];
//    temp["xi"] += Lambda2_["xYuV"] * H2["uViY"];
//    C1["jb"] -= alpha * temp["xi"] * T2["ijxb"];
//    C1["JB"] -= alpha * temp["xi"] * T2["iJxB"];
//    temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"AH"});
//    temp["XI"] += 0.5 * Lambda2_["XYUV"] * H2["UVIY"];
//    temp["XI"] += Lambda2_["yXvU"] * H2["vUyI"];
//    C1["jb"] -= alpha * temp["XI"] * T2["jIbX"];
//    C1["JB"] -= alpha * temp["XI"] * T2["IJXB"];

//    temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"av"});
//    temp["xe"] += 0.5 * T2["uvey"] * Lambda2_["xyuv"];
//    temp["xe"] += T2["uVeY"] * Lambda2_["xYuV"];
//    C1["qs"] += alpha * temp["xe"] * H2["eqxs"];
//    C1["QS"] += alpha * temp["xe"] * H2["eQxS"];
//    temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"AV"});
//    temp["XE"] += 0.5 * T2["UVEY"] * Lambda2_["XYUV"];
//    temp["XE"] += T2["uVyE"] * Lambda2_["yXuV"];
//    C1["qs"] += alpha * temp["XE"] * H2["qEsX"];
//    C1["QS"] += alpha * temp["XE"] * H2["EQXS"];

//    temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"ca"});
//    temp["mu"] += 0.5 * T2["mvxy"] * Lambda2_["xyuv"];
//    temp["mu"] += T2["mVxY"] * Lambda2_["xYuV"];
//    C1["qs"] -= alpha * temp["mu"] * H2["uqms"];
//    C1["QS"] -= alpha * temp["mu"] * H2["uQmS"];
//    temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"CA"});
//    temp["MU"] += 0.5 * T2["MVXY"] * Lambda2_["XYUV"];
//    temp["MU"] += T2["vMxY"] * Lambda2_["xYvU"];
//    C1["qs"] -= alpha * temp["MU"] * H2["qUsM"];
//    C1["QS"] -= alpha * temp["MU"] * H2["UQMS"];

//    if (print_ > 2) {
//        outfile->Printf("\n    Time for [H2, T2] -> C1 : %12.3f", timer.get());
//    }
//    dsrg_time_.add("221", timer.get());
//}

// void SADSRG::H1_T2_C2(BlockedTensor& H1, BlockedTensor& T2, const double& alpha,
//                      BlockedTensor& C2) {
//    local_timer timer;

//    C2["ijpb"] += alpha * T2["ijab"] * H1["ap"];
//    C2["ijap"] += alpha * T2["ijab"] * H1["bp"];
//    C2["qjab"] -= alpha * T2["ijab"] * H1["qi"];
//    C2["iqab"] -= alpha * T2["ijab"] * H1["qj"];

//    C2["iJpB"] += alpha * T2["iJaB"] * H1["ap"];
//    C2["iJaP"] += alpha * T2["iJaB"] * H1["BP"];
//    C2["qJaB"] -= alpha * T2["iJaB"] * H1["qi"];
//    C2["iQaB"] -= alpha * T2["iJaB"] * H1["QJ"];

//    C2["IJPB"] += alpha * T2["IJAB"] * H1["AP"];
//    C2["IJAP"] += alpha * T2["IJAB"] * H1["BP"];
//    C2["QJAB"] -= alpha * T2["IJAB"] * H1["QI"];
//    C2["IQAB"] -= alpha * T2["IJAB"] * H1["QJ"];

//    if (print_ > 2) {
//        outfile->Printf("\n    Time for [H1, T2] -> C2 : %12.3f", timer.get());
//    }
//    dsrg_time_.add("122", timer.get());
//}

// void SADSRG::H2_T1_C2(BlockedTensor& H2, BlockedTensor& T1, const double& alpha,
//                      BlockedTensor& C2) {
//    local_timer timer;

//    C2["irpq"] += alpha * T1["ia"] * H2["arpq"];
//    C2["ripq"] += alpha * T1["ia"] * H2["rapq"];
//    C2["rsaq"] -= alpha * T1["ia"] * H2["rsiq"];
//    C2["rspa"] -= alpha * T1["ia"] * H2["rspi"];

//    C2["iRpQ"] += alpha * T1["ia"] * H2["aRpQ"];
//    C2["rIpQ"] += alpha * T1["IA"] * H2["rApQ"];
//    C2["rSaQ"] -= alpha * T1["ia"] * H2["rSiQ"];
//    C2["rSpA"] -= alpha * T1["IA"] * H2["rSpI"];

//    C2["IRPQ"] += alpha * T1["IA"] * H2["ARPQ"];
//    C2["RIPQ"] += alpha * T1["IA"] * H2["RAPQ"];
//    C2["RSAQ"] -= alpha * T1["IA"] * H2["RSIQ"];
//    C2["RSPA"] -= alpha * T1["IA"] * H2["RSPI"];

//    if (print_ > 2) {
//        outfile->Printf("\n    Time for [H2, T1] -> C2 : %12.3f", timer.get());
//    }
//    dsrg_time_.add("212", timer.get());
//}

// void SADSRG::H2_T2_C2(BlockedTensor& H2, BlockedTensor& T2, const double& alpha,
//                      BlockedTensor& C2) {
//    local_timer timer;

//    /// max intermediate: g * g * p * p

//    // particle-particle contractions
//    C2["ijrs"] += 0.5 * alpha * H2["abrs"] * T2["ijab"];
//    C2["iJrS"] += alpha * H2["aBrS"] * T2["iJaB"];
//    C2["IJRS"] += 0.5 * alpha * H2["ABRS"] * T2["IJAB"];

//    C2["ijrs"] -= alpha * Gamma1_["xy"] * T2["ijxb"] * H2["ybrs"];
//    C2["iJrS"] -= alpha * Gamma1_["xy"] * T2["iJxB"] * H2["yBrS"];
//    C2["iJrS"] -= alpha * Gamma1_["XY"] * T2["iJbX"] * H2["bYrS"];
//    C2["IJRS"] -= alpha * Gamma1_["XY"] * T2["IJXB"] * H2["YBRS"];

//    // hole-hole contractions
//    C2["pqab"] += 0.5 * alpha * H2["pqij"] * T2["ijab"];
//    C2["pQaB"] += alpha * H2["pQiJ"] * T2["iJaB"];
//    C2["PQAB"] += 0.5 * alpha * H2["PQIJ"] * T2["IJAB"];

//    C2["pqab"] -= alpha * Eta1_["xy"] * T2["yjab"] * H2["pqxj"];
//    C2["pQaB"] -= alpha * Eta1_["xy"] * T2["yJaB"] * H2["pQxJ"];
//    C2["pQaB"] -= alpha * Eta1_["XY"] * T2["jYaB"] * H2["pQjX"];
//    C2["PQAB"] -= alpha * Eta1_["XY"] * T2["YJAB"] * H2["PQXJ"];

//    // hole-particle contractions
//    // figure out useful blocks of temp (assume symmetric C2 blocks, if cavv exists => acvv
//    exists) std::vector<std::string> C2_blocks(C2.block_labels()); std::sort(C2_blocks.begin(),
//    C2_blocks.end()); std::vector<std::string> temp_blocks; for (const std::string& p : {"c", "a",
//    "v"}) {
//        for (const std::string& q : {"c", "a"}) {
//            for (const std::string& r : {"c", "a", "v"}) {
//                for (const std::string& s : {"a", "v"}) {
//                    temp_blocks.emplace_back(p + q + r + s);
//                }
//            }
//        }
//    }
//    std::sort(temp_blocks.begin(), temp_blocks.end());
//    std::vector<std::string> blocks;
//    std::set_intersection(temp_blocks.begin(), temp_blocks.end(), C2_blocks.begin(),
//                          C2_blocks.end(), std::back_inserter(blocks));
//    BlockedTensor temp;
//    if (blocks.size() != 0) {
//        temp = ambit::BlockedTensor::build(tensor_type_, "temp", blocks);
//        temp["qjsb"] += alpha * H2["aqms"] * T2["mjab"];
//        temp["qjsb"] += alpha * H2["qAsM"] * T2["jMbA"];
//        temp["qjsb"] += alpha * Gamma1_["xy"] * T2["yjab"] * H2["aqxs"];
//        temp["qjsb"] += alpha * Gamma1_["XY"] * T2["jYbA"] * H2["qAsX"];
//        temp["qjsb"] -= alpha * Gamma1_["xy"] * T2["ijxb"] * H2["yqis"];
//        temp["qjsb"] -= alpha * Gamma1_["XY"] * T2["jIbX"] * H2["qYsI"];
//        C2["qjsb"] += temp["qjsb"];
//        C2["jqsb"] -= temp["qjsb"];
//        C2["qjbs"] -= temp["qjsb"];
//        C2["jqbs"] += temp["qjsb"];
//    }

//    // figure out useful blocks of temp (assume symmetric C2 blocks, if cavv exists => acvv
//    exists) temp_blocks.clear(); for (const std::string& p : {"C", "A", "V"}) {
//        for (const std::string& q : {"C", "A"}) {
//            for (const std::string& r : {"C", "A", "V"}) {
//                for (const std::string& s : {"A", "V"}) {
//                    temp_blocks.emplace_back(p + q + r + s);
//                }
//            }
//        }
//    }
//    std::sort(temp_blocks.begin(), temp_blocks.end());
//    blocks.clear();
//    std::set_intersection(temp_blocks.begin(), temp_blocks.end(), C2_blocks.begin(),
//                          C2_blocks.end(), std::back_inserter(blocks));
//    if (blocks.size() != 0) {
//        temp = ambit::BlockedTensor::build(tensor_type_, "temp", blocks);
//        temp["QJSB"] += alpha * H2["AQMS"] * T2["MJAB"];
//        temp["QJSB"] += alpha * H2["aQmS"] * T2["mJaB"];
//        temp["QJSB"] += alpha * Gamma1_["XY"] * T2["YJAB"] * H2["AQXS"];
//        temp["QJSB"] += alpha * Gamma1_["xy"] * T2["yJaB"] * H2["aQxS"];
//        temp["QJSB"] -= alpha * Gamma1_["XY"] * T2["IJXB"] * H2["YQIS"];
//        temp["QJSB"] -= alpha * Gamma1_["xy"] * T2["iJxB"] * H2["yQiS"];
//        C2["QJSB"] += temp["QJSB"];
//        C2["JQSB"] -= temp["QJSB"];
//        C2["QJBS"] -= temp["QJSB"];
//        C2["JQBS"] += temp["QJSB"];
//    }

//    C2["qJsB"] += alpha * H2["aqms"] * T2["mJaB"];
//    C2["qJsB"] += alpha * H2["qAsM"] * T2["MJAB"];
//    C2["qJsB"] += alpha * Gamma1_["xy"] * T2["yJaB"] * H2["aqxs"];
//    C2["qJsB"] += alpha * Gamma1_["XY"] * T2["YJAB"] * H2["qAsX"];
//    C2["qJsB"] -= alpha * Gamma1_["xy"] * T2["iJxB"] * H2["yqis"];
//    C2["qJsB"] -= alpha * Gamma1_["XY"] * T2["IJXB"] * H2["qYsI"];

//    C2["iQsB"] -= alpha * T2["iMaB"] * H2["aQsM"];
//    C2["iQsB"] -= alpha * Gamma1_["XY"] * T2["iYaB"] * H2["aQsX"];
//    C2["iQsB"] += alpha * Gamma1_["xy"] * T2["iJxB"] * H2["yQsJ"];

//    C2["qJaS"] -= alpha * T2["mJaB"] * H2["qBmS"];
//    C2["qJaS"] -= alpha * Gamma1_["xy"] * T2["yJaB"] * H2["qBxS"];
//    C2["qJaS"] += alpha * Gamma1_["XY"] * T2["iJaX"] * H2["qYiS"];

//    C2["iQaS"] += alpha * T2["imab"] * H2["bQmS"];
//    C2["iQaS"] += alpha * T2["iMaB"] * H2["BQMS"];
//    C2["iQaS"] += alpha * Gamma1_["xy"] * T2["iyab"] * H2["bQxS"];
//    C2["iQaS"] += alpha * Gamma1_["XY"] * T2["iYaB"] * H2["BQXS"];
//    C2["iQaS"] -= alpha * Gamma1_["xy"] * T2["ijax"] * H2["yQjS"];
//    C2["iQaS"] -= alpha * Gamma1_["XY"] * T2["iJaX"] * H2["YQJS"];

//    if (print_ > 2) {
//        outfile->Printf("\n    Time for [H2, T2] -> C2 : %12.3f", timer.get());
//    }
//    dsrg_time_.add("222", timer.get());
//}

// void SADSRG::H2_T2_C3(BlockedTensor& H2, BlockedTensor& T2, const double& alpha, BlockedTensor&
// C3,
//                      const bool& active_only) {
//    local_timer timer;

//    /// Potentially be as large as p * p * h * g * g * g

//    BlockedTensor temp;

//    // aaa and bbb
//    if (active_only) {
//        temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"aaaaaa"});
//        temp["rsjabq"] -= alpha * H2["rsqi"] * T2["ijab"];
//        temp["ijspqb"] += alpha * H2["aspq"] * T2["ijba"];
//        C3["xyzuvw"] += temp["xyzuvw"];
//        C3["zxyuvw"] += temp["xyzuvw"];
//        C3["xzyuvw"] -= temp["xyzuvw"];
//        C3["xyzwuv"] += temp["xyzuvw"];
//        C3["zxywuv"] += temp["xyzuvw"];
//        C3["xzywuv"] -= temp["xyzuvw"];
//        C3["xyzuwv"] -= temp["xyzuvw"];
//        C3["zxyuwv"] -= temp["xyzuvw"];
//        C3["xzyuwv"] += temp["xyzuvw"];

//        temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"AAAAAA"});
//        temp["RSJABQ"] -= alpha * H2["RSQI"] * T2["IJAB"];
//        temp["IJSPQB"] += alpha * H2["ASPQ"] * T2["IJBA"];
//        C3["XYZUVW"] += temp["XYZUVW"];
//        C3["ZXYUVW"] += temp["XYZUVW"];
//        C3["XZYUVW"] -= temp["XYZUVW"];
//        C3["XYZWUV"] += temp["XYZUVW"];
//        C3["ZXYWUV"] += temp["XYZUVW"];
//        C3["XZYWUV"] -= temp["XYZUVW"];
//        C3["XYZUWV"] -= temp["XYZUVW"];
//        C3["ZXYUWV"] -= temp["XYZUVW"];
//        C3["XZYUWV"] += temp["XYZUVW"];
//    } else {
//        temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"gghppg"});
//        temp["rsjabq"] -= H2["rsqi"] * T2["ijab"];
//        C3["rsjabq"] += alpha * temp["rsjabq"];
//        C3["rjsabq"] -= alpha * temp["rsjabq"];
//        C3["jrsabq"] += alpha * temp["rsjabq"];
//        C3["rsjaqb"] -= alpha * temp["rsjabq"];
//        C3["rjsaqb"] += alpha * temp["rsjabq"];
//        C3["jrsaqb"] -= alpha * temp["rsjabq"];
//        C3["rsjqab"] += alpha * temp["rsjabq"];
//        C3["rjsqab"] -= alpha * temp["rsjabq"];
//        C3["jrsqab"] += alpha * temp["rsjabq"];

//        temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"hhgggp"});
//        temp["ijspqb"] += H2["aspq"] * T2["ijba"];
//        C3["ijspqb"] += alpha * temp["ijspqb"];
//        C3["isjpqb"] -= alpha * temp["ijspqb"];
//        C3["sijpqb"] += alpha * temp["ijspqb"];
//        C3["ijspbq"] -= alpha * temp["ijspqb"];
//        C3["isjpbq"] += alpha * temp["ijspqb"];
//        C3["sijpbq"] -= alpha * temp["ijspqb"];
//        C3["ijsbpq"] += alpha * temp["ijspqb"];
//        C3["isjbpq"] -= alpha * temp["ijspqb"];
//        C3["sijbpq"] += alpha * temp["ijspqb"];

//        temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"GGHPPG"});
//        temp["RSJABQ"] -= H2["RSQI"] * T2["IJAB"];
//        C3["RSJABQ"] += alpha * temp["RSJABQ"];
//        C3["RJSABQ"] -= alpha * temp["RSJABQ"];
//        C3["JRSABQ"] += alpha * temp["RSJABQ"];
//        C3["RSJAQB"] -= alpha * temp["RSJABQ"];
//        C3["RJSAQB"] += alpha * temp["RSJABQ"];
//        C3["JRSAQB"] -= alpha * temp["RSJABQ"];
//        C3["RSJQAB"] += alpha * temp["RSJABQ"];
//        C3["RJSQAB"] -= alpha * temp["RSJABQ"];
//        C3["JRSQAB"] += alpha * temp["RSJABQ"];

//        temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"HHGGGP"});
//        temp["IJSPQB"] += H2["ASPQ"] * T2["IJBA"];
//        C3["IJSPQB"] += alpha * temp["IJSPQB"];
//        C3["ISJPQB"] -= alpha * temp["IJSPQB"];
//        C3["SIJPQB"] += alpha * temp["IJSPQB"];
//        C3["IJSPBQ"] -= alpha * temp["IJSPQB"];
//        C3["ISJPBQ"] += alpha * temp["IJSPQB"];
//        C3["SIJPBQ"] -= alpha * temp["IJSPQB"];
//        C3["IJSBPQ"] += alpha * temp["IJSPQB"];
//        C3["ISJBPQ"] -= alpha * temp["IJSPQB"];
//        C3["SIJBPQ"] += alpha * temp["IJSPQB"];
//    }

//    // aab hole contraction
//    C3["rjSabQ"] -= alpha * H2["rSiQ"] * T2["ijab"];
//    C3["jrSabQ"] += alpha * H2["rSiQ"] * T2["ijab"];

//    C3["rsJaqB"] += alpha * H2["rsqi"] * T2["iJaB"];
//    C3["rsJqaB"] -= alpha * H2["rsqi"] * T2["iJaB"];

//    if (active_only) {
//        temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"aAaaAa"});
//    } else {
//        temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"gGhpPg"});
//    }
//    temp["rSjaBq"] += H2["rSqI"] * T2["jIaB"];
//    C3["rjSaqB"] += alpha * temp["rSjaBq"];
//    C3["jrSaqB"] -= alpha * temp["rSjaBq"];
//    C3["rjSqaB"] -= alpha * temp["rSjaBq"];
//    C3["jrSqaB"] += alpha * temp["rSjaBq"];

//    // aab particle contraction
//    C3["isJpqB"] += alpha * H2["aspq"] * T2["iJaB"];
//    C3["siJpqB"] -= alpha * H2["aspq"] * T2["iJaB"];

//    C3["ijSpbQ"] -= alpha * H2["aSpQ"] * T2["ijba"];
//    C3["ijSbpQ"] += alpha * H2["aSpQ"] * T2["ijba"];

//    if (active_only) {
//        temp.zero();
//    } else {
//        temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"hHggGp"});
//    }
//    temp["iJspQb"] -= H2["sApQ"] * T2["iJbA"];
//    C3["isJpbQ"] += alpha * temp["iJspQb"];
//    C3["siJpbQ"] -= alpha * temp["iJspQb"];
//    C3["isJbpQ"] -= alpha * temp["iJspQb"];
//    C3["siJbpQ"] += alpha * temp["iJspQb"];

//    // abb hole contraction
//    if (active_only) {
//        temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"aAAaAA"});
//    } else {
//        temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"gGHpPG"});
//    }
//    temp["rSJaBQ"] += H2["rSiQ"] * T2["iJaB"];
//    C3["rSJaBQ"] += alpha * temp["rSJaBQ"];
//    C3["rJSaBQ"] -= alpha * temp["rSJaBQ"];
//    C3["rSJaQB"] -= alpha * temp["rSJaBQ"];
//    C3["rJSaQB"] += alpha * temp["rSJaBQ"];

//    C3["jRSaBQ"] += alpha * H2["RSQI"] * T2["jIaB"];
//    C3["jRSaQB"] -= alpha * H2["RSQI"] * T2["jIaB"];

//    C3["rSJqAB"] -= alpha * H2["rSqI"] * T2["IJAB"];
//    C3["rJSqAB"] += alpha * H2["rSqI"] * T2["IJAB"];

//    // abb particle contraction
//    if (active_only) {
//        temp.zero();
//    } else {
//        temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"hHGgGP"});
//    }
//    temp["iJSpQB"] -= H2["aSpQ"] * T2["iJaB"];
//    C3["iJSpQB"] += alpha * temp["iJSpQB"];
//    C3["iSJpQB"] -= alpha * temp["iJSpQB"];
//    C3["iJSpBQ"] -= alpha * temp["iJSpQB"];
//    C3["iSJpBQ"] += alpha * temp["iJSpQB"];

//    C3["sIJpQB"] -= alpha * H2["sApQ"] * T2["IJBA"];
//    C3["sIJpBQ"] += alpha * H2["sApQ"] * T2["IJBA"];

//    C3["iJSbPQ"] += alpha * H2["ASPQ"] * T2["iJbA"];
//    C3["iSJbPQ"] -= alpha * H2["ASPQ"] * T2["iJbA"];

//    if (print_ > 2) {
//        outfile->Printf("\n    Time for [H2, T2] -> C3 : %12.3f", timer.get());
//    }
//    dsrg_time_.add("223", timer.get());
//}

// dsrgHeff SADSRG::commutator_HT_noGNO(ambit::BlockedTensor H1, ambit::BlockedTensor H2,
//                                     ambit::BlockedTensor T1, ambit::BlockedTensor T2) {
//    dsrgHeff Heff;

//    Heff.H1 = BTF_->build(tensor_type_, "[H,T]1", spin_cases({"aa"}));
//    Heff.H1a = Heff.H1.block("aa");
//    Heff.H1b = Heff.H1.block("AA");

//    Heff.H2 = BTF_->build(tensor_type_, "[H,T]2", spin_cases({"aaaa"}));
//    Heff.H2aa = Heff.H2.block("aaaa");
//    Heff.H2ab = Heff.H2.block("aAaA");
//    Heff.H2bb = Heff.H2.block("AAAA");

//    Heff.H3 = BTF_->build(tensor_type_, "[H,T]3", spin_cases({"aaaaaa"}));
//    Heff.H3aaa = Heff.H3.block("aaaaaa");
//    Heff.H3aab = Heff.H3.block("aaAaaA");
//    Heff.H3abb = Heff.H3.block("aAAaAA");
//    Heff.H3bbb = Heff.H3.block("AAAAAA");

//    // scalar
//    double& H0 = Heff.H0;
//    H0 += H1["am"] * T1["ma"];
//    H0 += H1["AM"] * T1["MA"];

//    H0 += 0.25 * H2["abmn"] * T2["mnab"];
//    H0 += H2["aBmN"] * T2["mNaB"];
//    H0 += 0.25 * H2["ABMN"] * T2["MNAB"];

//    // 1-body
//    ambit::BlockedTensor& C1 = Heff.H1;
//    C1["vu"] += H1["eu"] * T1["ve"];
//    C1["VU"] += H1["EU"] * T1["VE"];

//    C1["vu"] -= H1["vm"] * T1["mu"];
//    C1["VU"] -= H1["VM"] * T1["MU"];

//    C1["vu"] += H2["avmu"] * T1["ma"];
//    C1["vu"] += H2["vAuM"] * T1["MA"];
//    C1["VU"] += H2["aVmU"] * T1["ma"];
//    C1["VU"] += H2["AVMU"] * T1["MA"];

//    C1["vu"] += H1["am"] * T2["vmua"];
//    C1["vu"] += H1["AM"] * T2["vMuA"];
//    C1["VU"] += H1["am"] * T2["mVaU"];
//    C1["VU"] += H1["AM"] * T2["VMUA"];

//    C1["vu"] += 0.5 * H2["abum"] * T2["vmab"];
//    C1["vu"] += H2["aBuM"] * T2["vMaB"];
//    C1["VU"] += H2["aBmU"] * T2["mVaB"];
//    C1["VU"] += 0.5 * H2["ABUM"] * T2["VMAB"];

//    C1["vu"] -= 0.5 * H2["avmn"] * T2["mnau"];
//    C1["vu"] -= H2["vAmN"] * T2["mNuA"];
//    C1["VU"] -= H2["aVmN"] * T2["mNaU"];
//    C1["VU"] -= 0.5 * H2["AVMN"] * T2["MNAU"];

//    // 2-body
//    ambit::BlockedTensor& C2 = Heff.H2;
//    BlockedTensor temp = BTF_->build(tensor_type_, "temp", {"aaaa", "AAAA"});

//    temp["xyuv"] = H2["eyuv"] * T1["xe"];
//    temp["XYUV"] = H2["EYUV"] * T1["XE"];

//    C2["xyuv"] += temp["xyuv"];
//    C2["XYUV"] += temp["XYUV"];
//    C2["xyuv"] -= temp["yxuv"];
//    C2["XYUV"] -= temp["YXUV"];

//    C2["xYuV"] += H2["eYuV"] * T1["xe"];
//    C2["xYuV"] += H2["xEuV"] * T1["YE"];

//    temp["xyuv"] = H2["xymv"] * T1["mu"];
//    temp["XYUV"] = H2["XYMV"] * T1["MU"];

//    C2["xyuv"] -= temp["xyuv"];
//    C2["XYUV"] -= temp["XYUV"];
//    C2["xyuv"] += temp["xyvu"];
//    C2["XYUV"] += temp["XYVU"];

//    C2["xYuV"] -= H2["xYmV"] * T1["mu"];
//    C2["xYuV"] -= H2["xYuM"] * T1["MV"];

//    temp["xyuv"] = H1["eu"] * T2["xyev"];
//    temp["XYUV"] = H1["EU"] * T2["XYEV"];

//    C2["xyuv"] += temp["xyuv"];
//    C2["XYUV"] += temp["XYUV"];
//    C2["xyuv"] -= temp["xyvu"];
//    C2["XYUV"] -= temp["XYVU"];

//    C2["xYuV"] += H1["eu"] * T2["xYeV"];
//    C2["xYuV"] += H1["EV"] * T2["xYuE"];

//    temp["xyuv"] = H1["xm"] * T2["myuv"];
//    temp["XYUV"] = H1["XM"] * T2["MYUV"];

//    C2["xyuv"] -= temp["xyuv"];
//    C2["XYUV"] -= temp["XYUV"];
//    C2["xyuv"] += temp["yxuv"];
//    C2["XYUV"] += temp["YXUV"];

//    C2["xYuV"] -= H1["xm"] * T2["mYuV"];
//    C2["xYuV"] -= H1["YM"] * T2["xMuV"];

//    C2["xyuv"] += 0.5 * H2["abuv"] * T2["xyab"];
//    C2["xYuV"] += H2["aBuV"] * T2["xYaB"];
//    C2["XYUV"] += 0.5 * H2["ABUV"] * T2["XYAB"];

//    C2["xyuv"] -= 0.5 * H2["xyij"] * T2["ijuv"];
//    C2["xYuV"] -= H2["xYiJ"] * T2["iJuV"];
//    C2["XYUV"] -= 0.5 * H2["XYIJ"] * T2["IJUV"];

//    C2["xyuv"] += H2["xyim"] * T2["imuv"];
//    C2["xYuV"] += H2["xYiM"] * T2["iMuV"];
//    C2["xYuV"] += H2["xYmI"] * T2["mIuV"];
//    C2["XYUV"] += H2["XYIM"] * T2["IMUV"];

//    temp["xyuv"] = H2["ayum"] * T2["xmav"];
//    temp["xyuv"] += H2["yAuM"] * T2["xMvA"];
//    temp["XYUV"] = H2["aYmU"] * T2["mXaV"];
//    temp["XYUV"] += H2["AYUM"] * T2["XMAV"];

//    C2["xyuv"] -= temp["xyuv"];
//    C2["XYUV"] -= temp["XYUV"];
//    C2["xyuv"] += temp["yxuv"];
//    C2["XYUV"] += temp["YXUV"];
//    C2["xyuv"] += temp["xyvu"];
//    C2["XYUV"] += temp["XYVU"];
//    C2["xyuv"] -= temp["yxvu"];
//    C2["XYUV"] -= temp["YXVU"];

//    C2["xYuV"] -= H2["aYuM"] * T2["xMaV"];
//    C2["xYuV"] += H2["xaum"] * T2["mYaV"];
//    C2["xYuV"] += H2["xAuM"] * T2["MYAV"];
//    C2["xYuV"] += H2["aYmV"] * T2["xmua"];
//    C2["xYuV"] += H2["AYMV"] * T2["xMuA"];
//    C2["xYuV"] -= H2["xAmV"] * T2["mYuA"];

//    // 3-body
//    H2_T2_C3(H2, T2, 1.0, Heff.H3, true);

//    return Heff;
//}
} // namespace forte