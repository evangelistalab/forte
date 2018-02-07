#include <numeric>

#include "psi4/libpsi4util/process.h"
#include "psi4/libmints/molecule.h"
#include "psi4/libmints/dipole.h"

#include "master_mrdsrg.h"

namespace psi {
namespace forte {

MASTER_DSRG::MASTER_DSRG(Reference reference, SharedWavefunction ref_wfn, Options& options,
                         std::shared_ptr<ForteIntegrals> ints,
                         std::shared_ptr<MOSpaceInfo> mo_space_info)
    : DynamicCorrelationSolver(reference, ref_wfn, options, ints, mo_space_info),
      tensor_type_(ambit::CoreTensor), BTF_(new BlockedTensorFactory(options)) {
    reference_wavefunction_ = ref_wfn;
    startup();
}

MASTER_DSRG::~MASTER_DSRG() {
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

void MASTER_DSRG::startup() {
    print_h2("Multireference Driven Similarity Renormalization Group");

    // read options
    read_options();

    // read orbital spaces
    read_MOSpaceInfo();

    // set Ambit MO space labels
    set_ambit_MOSpace();

    // read commonly used energies
    Eref_ = reference_.get_Eref();
    Enuc_ = Process::environment.molecule()->nuclear_repulsion_energy(
        reference_wavefunction_->get_dipole_field_strength());
    Efrzc_ = ints_->frozen_core_energy();

    // initialize timer for commutator
    dsrg_time_ = DSRG_TIME();

    // prepare density matrix and cumulants
    init_density();

    // initialize Fock matrix
    init_fock();

    // setup bare dipole tensors and compute reference dipoles
    if (do_dm_) {
        init_dm_ints();
    }

    // recompute reference energy from ForteIntegral and check consistency with Reference
    check_init_reference_energy();

    // initialize Uactv_ to identity
    Uactv_ = BTF_->build(tensor_type_, "Uactv", spin_cases({"aa"}));
    Uactv_.iterate([&](const std::vector<size_t>& i, const std::vector<SpinType>&, double& value) {
        if (i[0] == i[1]) {
            value = 1.0;
        }
    });
}

void MASTER_DSRG::read_options() {
    outfile->Printf("\n    Reading DSRG options ............................ ");

    auto throw_error = [&](const std::string& message) -> void {
        outfile->Printf("\n  %s", message.c_str());
        throw PSIEXCEPTION(message);
    };

    print_ = options_.get_int("PRINT");

    s_ = options_.get_double("DSRG_S");
    if (s_ < 0) {
        throw_error("S parameter for DSRG must >= 0!");
    }
    taylor_threshold_ = options_.get_int("TAYLOR_THRESHOLD");
    if (taylor_threshold_ <= 0) {
        throw_error("Threshold for Taylor expansion must be an integer greater than 0!");
    }

    source_ = options_.get_str("SOURCE");
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

    ntamp_ = options_.get_int("NTAMP");
    intruder_tamp_ = options_.get_double("INTRUDER_TAMP");

    relax_ref_ = options_.get_str("RELAX_REF");

    eri_df_ = false;
    ints_type_ = options_.get_str("INT_TYPE");
    if (ints_type_ == "CHOLESKY" || ints_type_ == "DF" || ints_type_ == "DISKDF") {
        eri_df_ = true;
    }

    multi_state_ = options_["AVG_STATE"].size() != 0;
    multi_state_algorithm_ = options_.get_str("DSRG_MULTI_STATE");

    do_dm_ = options_.get_bool("DSRG_DIPOLE");
    if (multi_state_ && do_dm_) {
        if (multi_state_algorithm_ != "SA_FULL") {
            do_dm_ = false;
        }
    }

    outfile->Printf("Done");
}

void MASTER_DSRG::read_MOSpaceInfo() {
    core_mos_ = mo_space_info_->get_corr_abs_mo("RESTRICTED_DOCC");
    actv_mos_ = mo_space_info_->get_corr_abs_mo("ACTIVE");
    virt_mos_ = mo_space_info_->get_corr_abs_mo("RESTRICTED_UOCC");

    if (eri_df_) {
        aux_mos_ = std::vector<size_t>(ints_->nthree());
        std::iota(aux_mos_.begin(), aux_mos_.end(), 0);
    }
}

void MASTER_DSRG::set_ambit_MOSpace() {
    outfile->Printf("\n    Setting ambit MO space .......................... ");
    BlockedTensor::reset_mo_spaces();
    BlockedTensor::set_expert_mode(true);

    // define space labels
    acore_label_ = "c";
    aactv_label_ = "a";
    avirt_label_ = "v";
    bcore_label_ = "C";
    bactv_label_ = "A";
    bvirt_label_ = "V";

    // add Ambit index labels
    BTF_->add_mo_space(acore_label_, "mn", core_mos_, AlphaSpin);
    BTF_->add_mo_space(bcore_label_, "MN", core_mos_, BetaSpin);
    BTF_->add_mo_space(aactv_label_, "uvwxyz123", actv_mos_, AlphaSpin);
    BTF_->add_mo_space(bactv_label_, "UVWXYZ!@#", actv_mos_, BetaSpin);
    BTF_->add_mo_space(avirt_label_, "ef", virt_mos_, AlphaSpin);
    BTF_->add_mo_space(bvirt_label_, "EF", virt_mos_, BetaSpin);

    // map space labels to mo spaces
    label_to_spacemo_[acore_label_[0]] = core_mos_;
    label_to_spacemo_[bcore_label_[0]] = core_mos_;
    label_to_spacemo_[aactv_label_[0]] = actv_mos_;
    label_to_spacemo_[bactv_label_[0]] = actv_mos_;
    label_to_spacemo_[avirt_label_[0]] = virt_mos_;
    label_to_spacemo_[bvirt_label_[0]] = virt_mos_;

    // define composite spaces
    BTF_->add_composite_mo_space("h", "ijkl", {acore_label_, aactv_label_});
    BTF_->add_composite_mo_space("H", "IJKL", {bcore_label_, bactv_label_});
    BTF_->add_composite_mo_space("p", "abcd", {aactv_label_, avirt_label_});
    BTF_->add_composite_mo_space("P", "ABCD", {bactv_label_, bvirt_label_});
    BTF_->add_composite_mo_space("g", "pqrsto456", {acore_label_, aactv_label_, avirt_label_});
    BTF_->add_composite_mo_space("G", "PQRSTO789", {bcore_label_, bactv_label_, bvirt_label_});

    // if DF/CD
    if (eri_df_) {
        aux_label_ = "L";
        BTF_->add_mo_space(aux_label_, "g", aux_mos_, NoSpin);
        label_to_spacemo_[aux_label_[0]] = aux_mos_;
    }

    outfile->Printf("Done");
}

void MASTER_DSRG::init_density() {
    outfile->Printf("\n    Preparing tensors for density cumulants ......... ");
    Eta1_ = BTF_->build(tensor_type_, "Eta1", spin_cases({"aa"}));
    Gamma1_ = BTF_->build(tensor_type_, "Gamma1", spin_cases({"aa"}));
    Lambda2_ = BTF_->build(tensor_type_, "Lambda2", spin_cases({"aaaa"}));
    fill_density();
    outfile->Printf("Done");
}

void MASTER_DSRG::fill_density() {
    // 1-particle density (make a copy)
    Gamma1_.block("aa")("pq") = reference_.L1a()("pq");
    Gamma1_.block("AA")("pq") = reference_.L1a()("pq");

    // 1-hole density
    for (const std::string& block : {"aa", "AA"}) {
        (Eta1_.block(block)).iterate([&](const std::vector<size_t>& i, double& value) {
            value = (i[0] == i[1]) ? 1.0 : 0.0;
        });
    }
    Eta1_["uv"] -= Gamma1_["uv"];
    Eta1_["UV"] -= Gamma1_["UV"];

    // 2-body density cumulants (make a copy)
    Lambda2_.block("aaaa")("pqrs") = reference_.L2aa()("pqrs");
    Lambda2_.block("aAaA")("pqrs") = reference_.L2ab()("pqrs");
    Lambda2_.block("AAAA")("pqrs") = reference_.L2bb()("pqrs");
}

void MASTER_DSRG::init_fock() {
    outfile->Printf("\n    Building Fock matrix ............................ ");
    build_fock_from_ints(ints_, Fock_);
    fill_Fdiag(Fock_, Fdiag_a_, Fdiag_b_);
    outfile->Printf("Done");
}

void MASTER_DSRG::build_fock_from_ints(std::shared_ptr<ForteIntegrals> ints, BlockedTensor& F) {
    size_t ncmo = mo_space_info_->size("CORRELATED");
    F = BTF_->build(tensor_type_, "Fock", spin_cases({"gg"}));

    // for convenience, directly call make_fock_matrix in ForteIntegral

    SharedMatrix D1a(new Matrix("D1a", ncmo, ncmo));
    SharedMatrix D1b(new Matrix("D1b", ncmo, ncmo));
    for (size_t m = 0, ncore = core_mos_.size(); m < ncore; m++) {
        D1a->set(core_mos_[m], core_mos_[m], 1.0);
        D1b->set(core_mos_[m], core_mos_[m], 1.0);
    }

    Gamma1_.block("aa").citerate([&](const std::vector<size_t>& i, const double& value) {
        D1a->set(actv_mos_[i[0]], actv_mos_[i[1]], value);
    });
    Gamma1_.block("AA").citerate([&](const std::vector<size_t>& i, const double& value) {
        D1b->set(actv_mos_[i[0]], actv_mos_[i[1]], value);
    });

    ints->make_fock_matrix(D1a, D1b);

    F.iterate([&](const std::vector<size_t>& i, const std::vector<SpinType>& spin, double& value) {
        if (spin[0] == AlphaSpin) {
            value = ints_->get_fock_a(i[0], i[1]);
        } else {
            value = ints_->get_fock_b(i[0], i[1]);
        }
    });
}

void MASTER_DSRG::fill_Fdiag(BlockedTensor& F, std::vector<double>& Fa, std::vector<double>& Fb) {
    size_t ncmo = mo_space_info_->size("CORRELATED");
    Fa.resize(ncmo);
    Fb.resize(ncmo);

    F.iterate([&](const std::vector<size_t>& i, const std::vector<SpinType>& spin, double& value) {
        if ((spin[0] == AlphaSpin) && (i[0] == i[1])) {
            Fa[i[0]] = value;
        } else if ((spin[0] == BetaSpin) && (i[0] == i[1])) {
            Fb[i[0]] = value;
        }
    });
}

void MASTER_DSRG::check_init_reference_energy() {
    outfile->Printf("\n    Checking reference energy ....................... ");
    double E = compute_reference_energy_from_ints(ints_);
    outfile->Printf("Done");

    double econv = options_.get_double("E_CONVERGENCE");
    econv = econv < 1.0e-12 ? 1.0e-12 : econv;
    if (fabs(E - Eref_) > 10.0 * econv) {
        outfile->Printf("\n    Warning! Inconsistent reference energy!");
        outfile->Printf("\n    Read from Reference class:            %.12f", Eref_);
        outfile->Printf("\n    Recomputed using Reference densities: %.12f", E);
        outfile->Printf("\n    Reference energy (MK vacuum) is set to recomputed value.");

        warnings_.push_back(std::make_tuple("Inconsistent ref. energy", "Use recomputed value",
                                            "A bug? Post an issue."));
        Eref_ = E;
    }
}

double MASTER_DSRG::compute_reference_energy_from_ints(std::shared_ptr<ForteIntegrals> ints) {
    BlockedTensor H = BTF_->build(tensor_type_, "OEI", spin_cases({"cc", "aa"}), true);
    H.iterate([&](const std::vector<size_t>& i, const std::vector<SpinType>& spin, double& value) {
        if (spin[0] == AlphaSpin) {
            value = ints->oei_a(i[0], i[1]);
        } else {
            value = ints->oei_b(i[0], i[1]);
        }
    });

    BlockedTensor V = BTF_->build(tensor_type_, "APEI", spin_cases({"aaaa"}), true);
    V.block("aaaa")("prqs") =
        ints->aptei_aa_block(actv_mos_, actv_mos_, actv_mos_, actv_mos_)("prqs");
    V.block("aAaA")("prqs") =
        ints->aptei_ab_block(actv_mos_, actv_mos_, actv_mos_, actv_mos_)("prqs");
    V.block("AAAA")("prqs") =
        ints->aptei_bb_block(actv_mos_, actv_mos_, actv_mos_, actv_mos_)("prqs");

    // form Fock in batches
    // in cc block, only the diagonal elements are useful
    BlockedTensor F = BTF_->build(tensor_type_, "Fock pruned", spin_cases({"cc", "aa"}), true);
    F["ij"] = H["ij"];
    F["IJ"] = H["IJ"];
    F["uv"] += V["uxvy"] * Gamma1_["xy"];
    F["uv"] += V["uXvY"] * Gamma1_["XY"];
    F["UV"] += V["xUyV"] * Gamma1_["xy"];
    F["UV"] += V["UXVY"] * Gamma1_["XY"];

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
        Vtemp = ints->aptei_aa_block({nm}, core_mos_, {nm}, core_mos_);
        F.block("cc").data()[m * nc + m] += Vtemp("pqrs") * I("pqrs");

        Vtemp = ints->aptei_ab_block({nm}, core_mos_, {nm}, core_mos_);
        F.block("cc").data()[m * nc + m] += Vtemp("pqrs") * I("pqrs");

        Vtemp = ints->aptei_ab_block(core_mos_, {nm}, core_mos_, {nm});
        F.block("CC").data()[m * nc + m] += Vtemp("pqrs") * I("qpsr");

        Vtemp = ints->aptei_bb_block({nm}, core_mos_, {nm}, core_mos_);
        F.block("CC").data()[m * nc + m] += Vtemp("pqrs") * I("pqrs");
    }

    Vtemp = ints->aptei_aa_block(core_mos_, actv_mos_, core_mos_, actv_mos_);
    F.block("cc")("pq") += Vtemp("prqs") * Gamma1_.block("aa")("rs");
    F.block("aa")("pq") += Vtemp("rpsq") * I("irjs") * O("ij");

    Vtemp = ints->aptei_ab_block(core_mos_, actv_mos_, core_mos_, actv_mos_);
    F.block("cc")("pq") += Vtemp("prqs") * Gamma1_.block("AA")("rs");
    F.block("AA")("pq") += Vtemp("rpsq") * I("irjs") * O("ij");

    Vtemp = ints->aptei_ab_block(actv_mos_, core_mos_, actv_mos_, core_mos_);
    F.block("CC")("pq") += Vtemp("rpsq") * Gamma1_.block("aa")("rs");
    F.block("aa")("pq") += Vtemp("prqs") * I("irjs") * O("ij");

    Vtemp = ints->aptei_bb_block(core_mos_, actv_mos_, core_mos_, actv_mos_);
    F.block("CC")("pq") += Vtemp("prqs") * Gamma1_.block("AA")("rs");
    F.block("AA")("pq") += Vtemp("rpsq") * I("irjs") * O("ij");

    return compute_reference_energy(H, F, V);
}

double MASTER_DSRG::compute_reference_energy(BlockedTensor H, BlockedTensor F, BlockedTensor V) {
    /// H: bare OEI; F: Fock; V: bare APTEI.
    /// E = 0.5 * ( H["ij"] + F["ij"] ) * L1["ji"] + 0.25 * V["xyuv"] * L2["uvxy"]

    double E = Efrzc_ + Enuc_;

    for (const std::string block : {"cc", "CC"}) {
        for (size_t m = 0, nc = core_mos_.size(); m < nc; ++m) {
            E += 0.5 * H.block(block).data()[m * nc + m];
            E += 0.5 * F.block(block).data()[m * nc + m];
        }
    }

    E += 0.5 * H["uv"] * Gamma1_["vu"];
    E += 0.5 * H["UV"] * Gamma1_["VU"];
    E += 0.5 * F["uv"] * Gamma1_["vu"];
    E += 0.5 * F["UV"] * Gamma1_["VU"];

    E += 0.25 * V["uvxy"] * Lambda2_["xyuv"];
    E += 0.25 * V["UVXY"] * Lambda2_["XYUV"];
    E += V["uVxY"] * Lambda2_["xYuV"];

    return E;
}

double MASTER_DSRG::compute_reference_energy_df(BlockedTensor H, BlockedTensor F, BlockedTensor B) {
    /// H: bare OEI; F: Fock; V: bare APTEI; B: DF three-index
    /// E = 0.5 * ( H["ij"] + F["ij"] ) * L1["ji"] + 0.25 * V["xyuv"] * L2["uvxy"]
    /// V["pqrs"] = <pq||rs> = (pr|qs) - (ps|qr); (pr|qs) ~= B["Lpr"] * B["Lqs"]

    double E = Efrzc_ + Enuc_;

    for (const std::string block : {"cc", "CC"}) {
        for (size_t m = 0, nc = core_mos_.size(); m < nc; ++m) {
            E += 0.5 * H.block(block).data()[m * nc + m];
            E += 0.5 * F.block(block).data()[m * nc + m];
        }
    }

    E += 0.5 * H["uv"] * Gamma1_["vu"];
    E += 0.5 * H["UV"] * Gamma1_["VU"];
    E += 0.5 * F["uv"] * Gamma1_["vu"];
    E += 0.5 * F["UV"] * Gamma1_["VU"];

    E += 0.25 * B["gux"] * B["gvy"] * Lambda2_["xyuv"];
    E -= 0.25 * B["guy"] * B["gvx"] * Lambda2_["xyuv"];
    E += 0.25 * B["gUX"] * B["gVY"] * Lambda2_["XYUV"];
    E -= 0.25 * B["gUY"] * B["gVX"] * Lambda2_["XYUV"];
    E += B["gux"] * B["gVY"] * Lambda2_["xYuV"];

    return E;
}

void MASTER_DSRG::init_dm_ints() {
    outfile->Printf("\n    Preparing ambit tensors for dipole moments ...... ");
    dm_.clear();
    dm_nuc_ = std::vector<double>(3, 0.0);
    SharedVector dm_nuc =
        DipoleInt::nuclear_contribution(Process::environment.molecule(), Vector3(0.0, 0.0, 0.0));
    for (int i = 0; i < 3; ++i) {
        dm_nuc_[i] = dm_nuc->get(i);
        BlockedTensor dm_i = BTF_->build(tensor_type_, "Dipole " + dm_dirs_[i], spin_cases({"gg"}));
        dm_.emplace_back(dm_i);
    }

    std::vector<SharedMatrix> dm_a = ints_->compute_MOdipole_ints(true, true);
    std::vector<SharedMatrix> dm_b = ints_->compute_MOdipole_ints(false, true);
    fill_MOdm(dm_a, dm_b);
    compute_dm_ref();

    // prepare transformed dipole integrals
    if (multi_state_ || (relax_ref_ != "NONE")) {
        Mbar0_ = std::vector<double>(3, 0.0);
        Mbar1_.clear();
        Mbar2_.clear();
        Mbar3_.clear();
        for (int i = 0; i < 3; ++i) {
            BlockedTensor Mbar1 =
                BTF_->build(tensor_type_, "DSRG DM1 " + dm_dirs_[i], spin_cases({"aa"}));
            Mbar1_.emplace_back(Mbar1);
            BlockedTensor Mbar2 =
                BTF_->build(tensor_type_, "DSRG DM2 " + dm_dirs_[i], spin_cases({"aaaa"}));
            Mbar2_.emplace_back(Mbar2);
            if (options_.get_bool("FORM_MBAR3")) {
                BlockedTensor Mbar3 =
                    BTF_->build(tensor_type_, "DSRG DM3 " + dm_dirs_[i], spin_cases({"aaaaaa"}));
                Mbar3_.emplace_back(Mbar3);
            }
        }
    }

    outfile->Printf("Done");
}

void MASTER_DSRG::fill_MOdm(std::vector<SharedMatrix>& dm_a, std::vector<SharedMatrix>& dm_b) {
    // consider frozen-core part
    dm_frzc_ = std::vector<double>(3, 0.0);
    std::vector<size_t> frzc_mos = mo_space_info_->get_absolute_mo("FROZEN_DOCC");
    for (int z = 0; z < 3; ++z) {
        double dipole = 0.0;
        for (const auto& p : frzc_mos) {
            dipole += dm_a[z]->get(p, p);
            dipole += dm_b[z]->get(p, p);
        }
        dm_frzc_[z] = dipole;
    }

    // find out correspondance between ncmo and nmo
    std::vector<size_t> cmo_to_mo;
    Dimension frzcpi = mo_space_info_->get_dimension("FROZEN_DOCC");
    Dimension frzvpi = mo_space_info_->get_dimension("FROZEN_UOCC");
    Dimension ncmopi = mo_space_info_->get_dimension("CORRELATED");
    for (int h = 0, p = 0; h < nirrep_; ++h) {
        p += frzcpi[h];
        for (int r = 0; r < ncmopi[h]; ++r) {
            cmo_to_mo.push_back((size_t)p);
            ++p;
        }
        p += frzvpi[h];
    }

    // fill in dipole integrals to dm_
    for (int z = 0; z < 3; ++z) {
        dm_[z].iterate(
            [&](const std::vector<size_t>& i, const std::vector<SpinType>& spin, double& value) {
                if (spin[0] == AlphaSpin)
                    value = dm_a[z]->get(cmo_to_mo[i[0]], cmo_to_mo[i[1]]);
                if (spin[0] == BetaSpin)
                    value = dm_b[z]->get(cmo_to_mo[i[0]], cmo_to_mo[i[1]]);
            });
    }
}

void MASTER_DSRG::compute_dm_ref() {
    dm_ref_ = std::vector<double>(3, 0.0);
    do_dm_dirs_.clear();
    for (int z = 0; z < 3; ++z) {
        double dipole = dm_frzc_[z];
        for (const std::string& block : {"cc", "CC"}) {
            dm_[z].block(block).citerate([&](const std::vector<size_t>& i, const double& value) {
                if (i[0] == i[1]) {
                    dipole += value;
                }
            });
        }
        dipole += dm_[z]["uv"] * Gamma1_["uv"];
        dipole += dm_[z]["UV"] * Gamma1_["UV"];
        dm_ref_[z] = dipole;

        do_dm_dirs_.push_back(std::fabs(dipole) > 1.0e-15 ? true : false);
    }
}

std::shared_ptr<FCIIntegrals> MASTER_DSRG::compute_Heff_actv() {
    // de-normal-order DSRG transformed Hamiltonian
    double Edsrg = Eref_ + Hbar0_;
    if (options_.get_bool("FORM_HBAR3")) {
        deGNO_ints("Hamiltonian", Edsrg, Hbar1_, Hbar2_, Hbar3_);
        rotate_ints_semi_to_origin("Hamiltonian", Hbar1_, Hbar2_, Hbar3_);
    } else {
        deGNO_ints("Hamiltonian", Edsrg, Hbar1_, Hbar2_);
        rotate_ints_semi_to_origin("Hamiltonian", Hbar1_, Hbar2_);
    }

    // create FCIIntegral shared_ptr
    std::shared_ptr<FCIIntegrals> fci_ints =
        std::make_shared<FCIIntegrals>(ints_, actv_mos_, core_mos_);
    fci_ints->set_active_integrals(Hbar2_.block("aaaa"), Hbar2_.block("aAaA"),
                                   Hbar2_.block("AAAA"));
    fci_ints->set_restricted_one_body_operator(Hbar1_.block("aa").data(),
                                               Hbar1_.block("AA").data());
    fci_ints->set_scalar_energy(Edsrg - Enuc_ - Efrzc_);

    return fci_ints;
}

void MASTER_DSRG::deGNO_ints(const std::string& name, double& H0, BlockedTensor& H1,
                             BlockedTensor& H2) {
    print_h2("De-Normal-Order DSRG Transformed " + name);

    // compute scalar
    ForteTimer t0;
    outfile->Printf("\n    %-40s ... ", "Computing the scalar term");

    // scalar from H1
    double scalar1 = 0.0;
    scalar1 -= H1["vu"] * Gamma1_["uv"];
    scalar1 -= H1["VU"] * Gamma1_["UV"];

    // scalar from H2
    double scalar2 = 0.0;
    scalar2 += 0.5 * Gamma1_["uv"] * H2["vyux"] * Gamma1_["xy"];
    scalar2 += 0.5 * Gamma1_["UV"] * H2["VYUX"] * Gamma1_["XY"];
    scalar2 += Gamma1_["uv"] * H2["vYuX"] * Gamma1_["XY"];

    scalar2 -= 0.25 * H2["xyuv"] * Lambda2_["uvxy"];
    scalar2 -= 0.25 * H2["XYUV"] * Lambda2_["UVXY"];
    scalar2 -= H2["xYuV"] * Lambda2_["uVxY"];

    H0 += scalar1 + scalar2;
    outfile->Printf("Done. Timing %8.3f s", t0.elapsed());

    // compute 1-body term
    ForteTimer t1;
    outfile->Printf("\n    %-40s ... ", "Computing the 1-body term");

    H1["uv"] -= H2["uxvy"] * Gamma1_["yx"];
    H1["uv"] -= H2["uXvY"] * Gamma1_["YX"];
    H1["UV"] -= H2["xUyV"] * Gamma1_["yx"];
    H1["UV"] -= H2["UXVY"] * Gamma1_["YX"];
    outfile->Printf("Done. Timing %8.3f s", t1.elapsed());
}

void MASTER_DSRG::deGNO_ints(const std::string& name, double& H0, BlockedTensor& H1,
                             BlockedTensor& H2, BlockedTensor& H3) {
    print_h2("De-Normal-Order DSRG Transformed " + name);

    // compute scalar
    ForteTimer t0;
    outfile->Printf("\n    %-40s ... ", "Computing the scalar term");

    // scalar from H1
    double scalar1 = 0.0;
    scalar1 -= H1["vu"] * Gamma1_["uv"];
    scalar1 -= H1["VU"] * Gamma1_["UV"];

    // scalar from H2
    double scalar2 = 0.0;
    scalar2 += 0.5 * Gamma1_["uv"] * H2["vyux"] * Gamma1_["xy"];
    scalar2 += 0.5 * Gamma1_["UV"] * H2["VYUX"] * Gamma1_["XY"];
    scalar2 += Gamma1_["uv"] * H2["vYuX"] * Gamma1_["XY"];

    scalar2 -= 0.25 * H2["xyuv"] * Lambda2_["uvxy"];
    scalar2 -= 0.25 * H2["XYUV"] * Lambda2_["UVXY"];
    scalar2 -= H2["xYuV"] * Lambda2_["uVxY"];

    // scalar from H3
    double scalar3 = 0.0;
    scalar3 -= (1.0 / 36.0) * H3.block("aaaaaa")("xyzuvw") * reference_.L3aaa()("xyzuvw");
    scalar3 -= (1.0 / 36.0) * H3.block("AAAAAA")("XYZUVW") * reference_.L3bbb()("XYZUVW");
    scalar3 -= 0.25 * H3.block("aaAaaA")("xyZuvW") * reference_.L3aab()("xyZuvW");
    scalar3 -= 0.25 * H3.block("aAAaAA")("xYZuVW") * reference_.L3abb()("xYZuVW");

    // TODO: form one-body intermediate for scalar and 1-body
    scalar3 += 0.25 * H3["xyzuvw"] * Lambda2_["uvxy"] * Gamma1_["wz"];
    scalar3 += 0.25 * H3["XYZUVW"] * Lambda2_["UVXY"] * Gamma1_["WZ"];
    scalar3 += 0.25 * H3["xyZuvW"] * Lambda2_["uvxy"] * Gamma1_["WZ"];
    scalar3 += H3["xzYuwV"] * Lambda2_["uVxY"] * Gamma1_["wz"];
    scalar3 += 0.25 * H3["zXYwUV"] * Lambda2_["UVXY"] * Gamma1_["wz"];
    scalar3 += H3["xZYuWV"] * Lambda2_["uVxY"] * Gamma1_["WZ"];

    scalar3 -= (1.0 / 6.0) * H3["xyzuvw"] * Gamma1_["ux"] * Gamma1_["vy"] * Gamma1_["wz"];
    scalar3 -= (1.0 / 6.0) * H3["XYZUVW"] * Gamma1_["UX"] * Gamma1_["VY"] * Gamma1_["WZ"];
    scalar3 -= 0.5 * H3["xyZuvW"] * Gamma1_["ux"] * Gamma1_["vy"] * Gamma1_["WZ"];
    scalar3 -= 0.5 * H3["xYZuVW"] * Gamma1_["ux"] * Gamma1_["VY"] * Gamma1_["WZ"];

    H0 += scalar1 + scalar2 + scalar3;
    outfile->Printf("Done. Timing %8.3f s", t0.elapsed());

    // compute 1-body term
    ForteTimer t1;
    outfile->Printf("\n    %-40s ... ", "Computing the 1-body term");

    // 1-body from H2
    H1["uv"] -= H2["uxvy"] * Gamma1_["yx"];
    H1["uv"] -= H2["uXvY"] * Gamma1_["YX"];
    H1["UV"] -= H2["xUyV"] * Gamma1_["yx"];
    H1["UV"] -= H2["UXVY"] * Gamma1_["YX"];

    // 1-body from H3
    H1["uv"] += 0.5 * H3["uyzvxw"] * Gamma1_["xy"] * Gamma1_["wz"];
    H1["uv"] += 0.5 * H3["uYZvXW"] * Gamma1_["XY"] * Gamma1_["WZ"];
    H1["uv"] += H3["uyZvxW"] * Gamma1_["xy"] * Gamma1_["WZ"];

    H1["UV"] += 0.5 * H3["UYZVXW"] * Gamma1_["XY"] * Gamma1_["WZ"];
    H1["UV"] += 0.5 * H3["yzUxwV"] * Gamma1_["xy"] * Gamma1_["wz"];
    H1["UV"] += H3["yUZxVW"] * Gamma1_["xy"] * Gamma1_["WZ"];

    H1["uv"] -= 0.25 * H3["uxyvwz"] * Lambda2_["wzxy"];
    H1["uv"] -= 0.25 * H3["uXYvWZ"] * Lambda2_["WZXY"];
    H1["uv"] -= H3["uxYvwZ"] * Lambda2_["wZxY"];

    H1["UV"] -= 0.25 * H3["UXYVWZ"] * Lambda2_["WZXY"];
    H1["UV"] -= 0.25 * H3["xyUwzV"] * Lambda2_["wzxy"];
    H1["UV"] -= H3["xUYwVZ"] * Lambda2_["wZxY"];
    outfile->Printf("Done. Timing %8.3f s", t1.elapsed());

    // compute 2-body term
    ForteTimer t2;
    outfile->Printf("\n    %-40s ... ", "Computing the 2-body term");
    H2["xyuv"] -= H3["xyzuvw"] * Gamma1_["wz"];
    H2["xyuv"] -= H3["xyZuvW"] * Gamma1_["WZ"];
    H2["xYuV"] -= H3["xYZuVW"] * Gamma1_["WZ"];
    H2["xYuV"] -= H3["xzYuwV"] * Gamma1_["wz"];
    H2["XYUV"] -= H3["XYZUVW"] * Gamma1_["WZ"];
    H2["XYUV"] -= H3["zXYwUV"] * Gamma1_["wz"];
    outfile->Printf("Done. Timing %8.3f s", t2.elapsed());
}

ambit::BlockedTensor MASTER_DSRG::deGNO_Tamp(BlockedTensor& T1, BlockedTensor& T2,
                                             BlockedTensor& D1) {
    BlockedTensor T1eff = BTF_->build(tensor_type_, "T1eff from de-GNO", spin_cases({"hp"}));

    T1eff["ia"] = T1["ia"];
    T1eff["IA"] = T1["IA"];

    T1eff["ia"] -= T2["iuav"] * D1["vu"];
    T1eff["ia"] -= T2["iUaV"] * D1["VU"];
    T1eff["IA"] -= T2["uIvA"] * D1["vu"];
    T1eff["IA"] -= T2["IUAV"] * D1["VU"];

    return T1eff;
}

void MASTER_DSRG::rotate_ints_semi_to_origin(const std::string& name, BlockedTensor& H1,
                                             BlockedTensor& H2) {

    print_h2("Rotate DSRG Transformed " + name + " back to Original Basis");
    ambit::Tensor temp;
    ambit::Tensor Ua = Uactv_.block("aa");
    ambit::Tensor Ub = Uactv_.block("AA");

    ForteTimer timer;
    outfile->Printf("\n    %-40s ... ", "Rotating 1-body term to original basis");
    temp = H1.block("aa").clone(tensor_type_);
    H1.block("aa")("pq") = Ua("pu") * temp("uv") * Ua("qv");

    temp = H1.block("AA").clone(tensor_type_);
    H1.block("AA")("PQ") = Ub("PU") * temp("UV") * Ub("QV");
    outfile->Printf("Done. Timing %8.3f s", timer.elapsed());

    timer.reset();
    outfile->Printf("\n    %-40s ... ", "Rotating 2-body term to original basis");
    temp = H2.block("aaaa").clone(tensor_type_);
    H2.block("aaaa")("pqrs") = Ua("pa") * Ua("qb") * temp("abcd") * Ua("rc") * Ua("sd");

    temp = H2.block("aAaA").clone(tensor_type_);
    H2.block("aAaA")("pQrS") = Ua("pa") * Ub("QB") * temp("aBcD") * Ua("rc") * Ub("SD");

    temp = H2.block("AAAA").clone(tensor_type_);
    H2.block("AAAA")("PQRS") = Ub("PA") * Ub("QB") * temp("ABCD") * Ub("RC") * Ub("SD");
    outfile->Printf("Done. Timing %8.3f s", timer.elapsed());
}

void MASTER_DSRG::rotate_ints_semi_to_origin(const std::string& name, BlockedTensor& H1,
                                             BlockedTensor& H2, BlockedTensor& H3) {
    print_h2("Rotate DSRG Transformed " + name + " back to Original Basis");
    ambit::Tensor temp;
    ambit::Tensor Ua = Uactv_.block("aa");
    ambit::Tensor Ub = Uactv_.block("AA");

    ForteTimer timer;
    outfile->Printf("\n    %-40s ... ", "Rotating 1-body term to original basis");
    temp = H1.block("aa").clone(tensor_type_);
    H1.block("aa")("pq") = Ua("pu") * temp("uv") * Ua("qv");

    temp = H1.block("AA").clone(tensor_type_);
    H1.block("AA")("PQ") = Ub("PU") * temp("UV") * Ub("QV");
    outfile->Printf("Done. Timing %8.3f s", timer.elapsed());

    timer.reset();
    outfile->Printf("\n    %-40s ... ", "Rotating 2-body term to original basis");
    temp = H2.block("aaaa").clone(tensor_type_);
    H2.block("aaaa")("pqrs") = Ua("pa") * Ua("qb") * temp("abcd") * Ua("rc") * Ua("sd");

    temp = H2.block("aAaA").clone(tensor_type_);
    H2.block("aAaA")("pQrS") = Ua("pa") * Ub("QB") * temp("aBcD") * Ua("rc") * Ub("SD");

    temp = H2.block("AAAA").clone(tensor_type_);
    H2.block("AAAA")("PQRS") = Ub("PA") * Ub("QB") * temp("ABCD") * Ub("RC") * Ub("SD");
    outfile->Printf("Done. Timing %8.3f s", timer.elapsed());

    timer.reset();
    outfile->Printf("\n    %-40s ... ", "Rotating 3-body to original basis");
    temp = H3.block("aaaaaa").clone(tensor_type_);
    H3.block("aaaaaa")("pqrstu") =
        Ua("pa") * Ua("qb") * Ua("rc") * temp("abcijk") * Ua("si") * Ua("tj") * Ua("uk");

    temp = H3.block("aaAaaA").clone(tensor_type_);
    H3.block("aaAaaA")("pqrstu") =
        Ua("pa") * Ua("qb") * Ub("rc") * temp("abcijk") * Ua("si") * Ua("tj") * Ub("uk");

    temp = H3.block("aAAaAA").clone(tensor_type_);
    H3.block("aAAaAA")("pqrstu") =
        Ua("pa") * Ub("qb") * Ub("rc") * temp("abcijk") * Ua("si") * Ub("tj") * Ub("uk");

    temp = H3.block("AAAAAA").clone(tensor_type_);
    H3.block("AAAAAA")("pqrstu") =
        Ub("pa") * Ub("qb") * Ub("rc") * temp("abcijk") * Ub("si") * Ub("tj") * Ub("uk");

    outfile->Printf("Done. Timing %8.3f s", timer.elapsed());
}

std::vector<ambit::Tensor> MASTER_DSRG::Hbar(int n) {
    std::vector<ambit::Tensor> out;
    if (n == 1) {
        out = {Hbar1_.block("aa"), Hbar1_.block("AA")};
    } else if (n == 2) {
        out = {Hbar2_.block("aaaa"), Hbar2_.block("aAaA"), Hbar2_.block("AAAA")};
    } else if (n == 3) {
        if (options_.get_bool("FORM_HBAR3")) {
            out = {Hbar3_.block("aaaaaa"), Hbar3_.block("aaAaaA"), Hbar3_.block("aAAaAA"),
                   Hbar3_.block("AAAAAA")};
        } else {
            throw PSIEXCEPTION("Hbar3 is not formed. Check your code.");
        }
    } else {
        throw PSIEXCEPTION("Only 1, 2, and 3 Hbar are in Tensor format.");
    }
    return out;
}

void MASTER_DSRG::H1_T1_C0(BlockedTensor& H1, BlockedTensor& T1, const double& alpha, double& C0) {
    ForteTimer timer;

    double E = 0.0;
    E += H1["em"] * T1["me"];
    E += H1["ex"] * T1["ye"] * Gamma1_["xy"];
    E += H1["xm"] * T1["my"] * Eta1_["yx"];

    E += H1["EM"] * T1["ME"];
    E += H1["EX"] * T1["YE"] * Gamma1_["XY"];
    E += H1["XM"] * T1["MY"] * Eta1_["YX"];

    E *= alpha;
    C0 += E;

    if (print_ > 2) {
        outfile->Printf("\n    Time for [H1, T1] -> C0 : %12.3f", timer.elapsed());
    }
    dsrg_time_.add("110", timer.elapsed());
}

void MASTER_DSRG::H1_T2_C0(BlockedTensor& H1, BlockedTensor& T2, const double& alpha, double& C0) {
    ForteTimer timer;
    BlockedTensor temp;
    double E = 0.0;

    temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"aaaa"});
    temp["uvxy"] += H1["ex"] * T2["uvey"];
    temp["uvxy"] -= H1["vm"] * T2["umxy"];
    E += 0.5 * temp["uvxy"] * Lambda2_["xyuv"];

    temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"AAAA"});
    temp["UVXY"] += H1["EX"] * T2["UVEY"];
    temp["UVXY"] -= H1["VM"] * T2["UMXY"];
    E += 0.5 * temp["UVXY"] * Lambda2_["XYUV"];

    temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"aAaA"});
    temp["uVxY"] += H1["ex"] * T2["uVeY"];
    temp["uVxY"] += H1["EY"] * T2["uVxE"];
    temp["uVxY"] -= H1["VM"] * T2["uMxY"];
    temp["uVxY"] -= H1["um"] * T2["mVxY"];
    E += temp["uVxY"] * Lambda2_["xYuV"];

    E *= alpha;
    C0 += E;

    if (print_ > 2) {
        outfile->Printf("\n    Time for [H1, T2] -> C0 : %12.3f", timer.elapsed());
    }
    dsrg_time_.add("120", timer.elapsed());
}

void MASTER_DSRG::H2_T1_C0(BlockedTensor& H2, BlockedTensor& T1, const double& alpha, double& C0) {
    ForteTimer timer;
    BlockedTensor temp;
    double E = 0.0;

    temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"aaaa"});
    temp["uvxy"] += H2["evxy"] * T1["ue"];
    temp["uvxy"] -= H2["uvmy"] * T1["mx"];
    E += 0.5 * temp["uvxy"] * Lambda2_["xyuv"];

    temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"AAAA"});
    temp["UVXY"] += H2["EVXY"] * T1["UE"];
    temp["UVXY"] -= H2["UVMY"] * T1["MX"];
    E += 0.5 * temp["UVXY"] * Lambda2_["XYUV"];

    temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"aAaA"});
    temp["uVxY"] += H2["eVxY"] * T1["ue"];
    temp["uVxY"] += H2["uExY"] * T1["VE"];
    temp["uVxY"] -= H2["uVmY"] * T1["mx"];
    temp["uVxY"] -= H2["uVxM"] * T1["MY"];
    E += temp["uVxY"] * Lambda2_["xYuV"];

    E *= alpha;
    C0 += E;

    if (print_ > 2) {
        outfile->Printf("\n    Time for [H2, T1] -> C0 : %12.3f", timer.elapsed());
    }
    dsrg_time_.add("210", timer.elapsed());
}

void MASTER_DSRG::H2_T2_C0(BlockedTensor& H2, BlockedTensor& T2, const double& alpha, double& C0) {
    ForteTimer timer;

    // <[Hbar2, T2]> (C_2)^4
    double E = H2["eFmN"] * T2["mNeF"];
    E += 0.25 * H2["efmn"] * T2["mnef"];
    E += 0.25 * H2["EFMN"] * T2["MNEF"];

    BlockedTensor temp = ambit::BlockedTensor::build(tensor_type_, "temp", spin_cases({"aa"}));
    temp["vu"] += 0.5 * H2["efmu"] * T2["mvef"];
    temp["vu"] += H2["fEuM"] * T2["vMfE"];
    temp["VU"] += 0.5 * H2["EFMU"] * T2["MVEF"];
    temp["VU"] += H2["eFmU"] * T2["mVeF"];
    E += temp["vu"] * Gamma1_["uv"];
    E += temp["VU"] * Gamma1_["UV"];

    temp.zero();
    temp["vu"] += 0.5 * H2["vemn"] * T2["mnue"];
    temp["vu"] += H2["vEmN"] * T2["mNuE"];
    temp["VU"] += 0.5 * H2["VEMN"] * T2["MNUE"];
    temp["VU"] += H2["eVnM"] * T2["nMeU"];
    E += temp["vu"] * Eta1_["uv"];
    E += temp["VU"] * Eta1_["UV"];

    temp = ambit::BlockedTensor::build(tensor_type_, "temp", spin_cases({"aaaa"}));
    temp["yvxu"] += H2["efxu"] * T2["yvef"];
    temp["yVxU"] += H2["eFxU"] * T2["yVeF"];
    temp["YVXU"] += H2["EFXU"] * T2["YVEF"];
    E += 0.25 * temp["yvxu"] * Gamma1_["xy"] * Gamma1_["uv"];
    E += temp["yVxU"] * Gamma1_["UV"] * Gamma1_["xy"];
    E += 0.25 * temp["YVXU"] * Gamma1_["XY"] * Gamma1_["UV"];

    temp.zero();
    temp["vyux"] += H2["vymn"] * T2["mnux"];
    temp["vYuX"] += H2["vYmN"] * T2["mNuX"];
    temp["VYUX"] += H2["VYMN"] * T2["MNUX"];
    E += 0.25 * temp["vyux"] * Eta1_["uv"] * Eta1_["xy"];
    E += temp["vYuX"] * Eta1_["uv"] * Eta1_["XY"];
    E += 0.25 * temp["VYUX"] * Eta1_["UV"] * Eta1_["XY"];

    temp.zero();
    temp["vyux"] += H2["vemx"] * T2["myue"];
    temp["vyux"] += H2["vExM"] * T2["yMuE"];
    temp["VYUX"] += H2["eVmX"] * T2["mYeU"];
    temp["VYUX"] += H2["VEXM"] * T2["YMUE"];
    E += temp["vyux"] * Gamma1_["xy"] * Eta1_["uv"];
    E += temp["VYUX"] * Gamma1_["XY"] * Eta1_["UV"];
    temp["yVxU"] = H2["eVxM"] * T2["yMeU"];
    E += temp["yVxU"] * Gamma1_["xy"] * Eta1_["UV"];
    temp["vYuX"] = H2["vEmX"] * T2["mYuE"];
    E += temp["vYuX"] * Gamma1_["XY"] * Eta1_["uv"];

    temp.zero();
    temp["yvxu"] += 0.5 * Gamma1_["wz"] * H2["vexw"] * T2["yzue"];
    temp["yvxu"] += Gamma1_["WZ"] * H2["vExW"] * T2["yZuE"];
    temp["yvxu"] += 0.5 * Eta1_["wz"] * T2["myuw"] * H2["vzmx"];
    temp["yvxu"] += Eta1_["WZ"] * T2["yMuW"] * H2["vZxM"];
    E += temp["yvxu"] * Gamma1_["xy"] * Eta1_["uv"];

    temp["YVXU"] += 0.5 * Gamma1_["WZ"] * H2["VEXW"] * T2["YZUE"];
    temp["YVXU"] += Gamma1_["wz"] * H2["eVwX"] * T2["zYeU"];
    temp["YVXU"] += 0.5 * Eta1_["WZ"] * T2["MYUW"] * H2["VZMX"];
    temp["YVXU"] += Eta1_["wz"] * H2["zVmX"] * T2["mYwU"];
    E += temp["YVXU"] * Gamma1_["XY"] * Eta1_["UV"];

    // <[Hbar2, T2]> C_4 (C_2)^2 HH -- combined with PH
    temp = ambit::BlockedTensor::build(tensor_type_, "temp", spin_cases({"aaaa"}));
    temp["uvxy"] += 0.125 * H2["uvmn"] * T2["mnxy"];
    temp["uvxy"] += 0.25 * Gamma1_["wz"] * H2["uvmw"] * T2["mzxy"];
    temp["uVxY"] += H2["uVmN"] * T2["mNxY"];
    temp["uVxY"] += Gamma1_["wz"] * T2["zMxY"] * H2["uVwM"];
    temp["uVxY"] += Gamma1_["WZ"] * H2["uVmW"] * T2["mZxY"];
    temp["UVXY"] += 0.125 * H2["UVMN"] * T2["MNXY"];
    temp["UVXY"] += 0.25 * Gamma1_["WZ"] * H2["UVMW"] * T2["MZXY"];

    // <[Hbar2, T2]> C_4 (C_2)^2 PP -- combined with PH
    temp["uvxy"] += 0.125 * H2["efxy"] * T2["uvef"];
    temp["uvxy"] += 0.25 * Eta1_["wz"] * T2["uvew"] * H2["ezxy"];
    temp["uVxY"] += H2["eFxY"] * T2["uVeF"];
    temp["uVxY"] += Eta1_["wz"] * H2["zExY"] * T2["uVwE"];
    temp["uVxY"] += Eta1_["WZ"] * T2["uVeW"] * H2["eZxY"];
    temp["UVXY"] += 0.125 * H2["EFXY"] * T2["UVEF"];
    temp["UVXY"] += 0.25 * Eta1_["WZ"] * T2["UVEW"] * H2["EZXY"];

    // <[Hbar2, T2]> C_4 (C_2)^2 PH
    temp["uvxy"] += H2["eumx"] * T2["mvey"];
    temp["uvxy"] += H2["uExM"] * T2["vMyE"];
    temp["uvxy"] += Gamma1_["wz"] * T2["zvey"] * H2["euwx"];
    temp["uvxy"] += Gamma1_["WZ"] * H2["uExW"] * T2["vZyE"];
    temp["uvxy"] += Eta1_["zw"] * H2["wumx"] * T2["mvzy"];
    temp["uvxy"] += Eta1_["ZW"] * T2["vMyZ"] * H2["uWxM"];
    E += temp["uvxy"] * Lambda2_["xyuv"];

    temp["UVXY"] += H2["eUmX"] * T2["mVeY"];
    temp["UVXY"] += H2["EUMX"] * T2["MVEY"];
    temp["UVXY"] += Gamma1_["wz"] * T2["zVeY"] * H2["eUwX"];
    temp["UVXY"] += Gamma1_["WZ"] * T2["ZVEY"] * H2["EUWX"];
    temp["UVXY"] += Eta1_["zw"] * H2["wUmX"] * T2["mVzY"];
    temp["UVXY"] += Eta1_["ZW"] * H2["WUMX"] * T2["MVZY"];
    E += temp["UVXY"] * Lambda2_["XYUV"];

    temp["uVxY"] += H2["uexm"] * T2["mVeY"];
    temp["uVxY"] += H2["uExM"] * T2["MVEY"];
    temp["uVxY"] -= H2["eVxM"] * T2["uMeY"];
    temp["uVxY"] -= H2["uEmY"] * T2["mVxE"];
    temp["uVxY"] += H2["eVmY"] * T2["umxe"];
    temp["uVxY"] += H2["EVMY"] * T2["uMxE"];

    temp["uVxY"] += Gamma1_["wz"] * T2["zVeY"] * H2["uexw"];
    temp["uVxY"] += Gamma1_["WZ"] * T2["ZVEY"] * H2["uExW"];
    temp["uVxY"] -= Gamma1_["WZ"] * H2["eVxW"] * T2["uZeY"];
    temp["uVxY"] -= Gamma1_["wz"] * T2["zVxE"] * H2["uEwY"];
    temp["uVxY"] += Gamma1_["wz"] * T2["zuex"] * H2["eVwY"];
    temp["uVxY"] -= Gamma1_["WZ"] * H2["EVYW"] * T2["uZxE"];

    temp["uVxY"] += Eta1_["zw"] * H2["wumx"] * T2["mVzY"];
    temp["uVxY"] += Eta1_["ZW"] * T2["VMYZ"] * H2["uWxM"];
    temp["uVxY"] -= Eta1_["zw"] * H2["wVxM"] * T2["uMzY"];
    temp["uVxY"] -= Eta1_["ZW"] * T2["mVxZ"] * H2["uWmY"];
    temp["uVxY"] += Eta1_["zw"] * T2["umxz"] * H2["wVmY"];
    temp["uVxY"] += Eta1_["ZW"] * H2["WVMY"] * T2["uMxZ"];
    E += temp["uVxY"] * Lambda2_["xYuV"];

    // <[Hbar2, T2]> C_6 C_2
    if (options_.get_str("THREEPDC") != "ZERO") {
        temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"aaaaaa"});
        temp["uvwxyz"] += H2["uviz"] * T2["iwxy"];
        temp["uvwxyz"] += H2["waxy"] * T2["uvaz"];
        E += 0.25 * temp.block("aaaaaa")("uvwxyz") * reference_.L3aaa()("xyzuvw");

        temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"AAAAAA"});
        temp["UVWXYZ"] += H2["UVIZ"] * T2["IWXY"];
        temp["UVWXYZ"] += H2["WAXY"] * T2["UVAZ"];
        E += 0.25 * temp.block("AAAAAA")("UVWXYZ") * reference_.L3bbb()("XYZUVW");

        temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"aaAaaA"});
        temp["uvWxyZ"] -= H2["uviy"] * T2["iWxZ"];
        temp["uvWxyZ"] -= H2["uWiZ"] * T2["ivxy"];
        temp["uvWxyZ"] += 2.0 * H2["uWyI"] * T2["vIxZ"];

        temp["uvWxyZ"] += H2["aWxZ"] * T2["uvay"];
        temp["uvWxyZ"] -= H2["vaxy"] * T2["uWaZ"];
        temp["uvWxyZ"] -= 2.0 * H2["vAxZ"] * T2["uWyA"];
        E += 0.5 * temp.block("aaAaaA")("uvWxyZ") * reference_.L3aab()("xyZuvW");

        temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"aAAaAA"});
        temp["uVWxYZ"] -= H2["VWIZ"] * T2["uIxY"];
        temp["uVWxYZ"] -= H2["uVxI"] * T2["IWYZ"];
        temp["uVWxYZ"] += 2.0 * H2["uViZ"] * T2["iWxY"];

        temp["uVWxYZ"] += H2["uAxY"] * T2["VWAZ"];
        temp["uVWxYZ"] -= H2["WAYZ"] * T2["uVxA"];
        temp["uVWxYZ"] -= 2.0 * H2["aWxY"] * T2["uVaZ"];
        E += 0.5 * temp.block("aAAaAA")("uVWxYZ") * reference_.L3abb()("xYZuVW");
    }

    // multiply prefactor and copy to C0
    E *= alpha;
    C0 += E;

    if (print_ > 2) {
        outfile->Printf("\n    Time for [H2, T2] -> C0 : %12.3f", timer.elapsed());
    }
    dsrg_time_.add("220", timer.elapsed());
}

void MASTER_DSRG::H1_T1_C1(BlockedTensor& H1, BlockedTensor& T1, const double& alpha,
                           BlockedTensor& C1) {
    ForteTimer timer;

    C1["ip"] += alpha * H1["ap"] * T1["ia"];
    C1["qa"] -= alpha * T1["ia"] * H1["qi"];
    C1["IP"] += alpha * H1["AP"] * T1["IA"];
    C1["QA"] -= alpha * T1["IA"] * H1["QI"];

    if (print_ > 2) {
        outfile->Printf("\n    Time for [H1, T1] -> C1 : %12.3f", timer.elapsed());
    }
    dsrg_time_.add("111", timer.elapsed());
}

// void MASTER_DSRG::H1_T1_C1aa(BlockedTensor& H1, BlockedTensor& T1, const double& alpha,
//                             BlockedTensor& C1) {
//    Timer timer;

//    C1["uv"] += alpha * H1["av"] * T1["ua"];
//    C1["uv"] -= alpha * T1["iv"] * H1["ui"];
//    C1["UV"] += alpha * H1["AV"] * T1["UA"];
//    C1["UV"] -= alpha * T1["IV"] * H1["UI"];

//    if (print_ > 2) {
//        outfile->Printf("\n    Time for [H1, T1] -> C1aa : %12.3f", timer.get());
//    }
//    dsrg_time_.add("111", timer.get());
//}

// void MASTER_DSRG::H1_T1_C1ph(BlockedTensor& H1, BlockedTensor& T1, const double& alpha,
//                             BlockedTensor& C1) {
//    Timer timer;

//    C1["ui"] += alpha * H1["ai"] * T1["ua"];
//    C1["au"] -= alpha * T1["iu"] * H1["ai"];
//    C1["UI"] += alpha * H1["AI"] * T1["UA"];
//    C1["AU"] -= alpha * T1["IU"] * H1["AI"];

//    if (print_ > 2) {
//        outfile->Printf("\n    Time for [H1, T1] -> C1ph : %12.3f", timer.get());
//    }
//    dsrg_time_.add("111", timer.get());
//}

// void MASTER_DSRG::H1_T1_C1hp(BlockedTensor& H1, BlockedTensor& T1, const double& alpha,
//                             BlockedTensor& C1) {
//    Timer timer;

//    C1["ib"] += alpha * H1["ab"] * T1["ia"];
//    C1["ja"] -= alpha * T1["ia"] * H1["ji"];
//    C1["IB"] += alpha * H1["AB"] * T1["IA"];
//    C1["JA"] -= alpha * T1["IA"] * H1["JI"];

//    if (print_ > 2) {
//        outfile->Printf("\n    Time for [H1, T1] -> C1hp : %12.3f", timer.get());
//    }
//    dsrg_time_.add("111", timer.get());
//}

void MASTER_DSRG::H1_T2_C1(BlockedTensor& H1, BlockedTensor& T2, const double& alpha,
                           BlockedTensor& C1) {
    ForteTimer timer;

    C1["ia"] += alpha * H1["bm"] * T2["imab"];
    C1["ia"] += alpha * H1["bu"] * Gamma1_["uv"] * T2["ivab"];
    C1["ia"] -= alpha * H1["vj"] * Gamma1_["uv"] * T2["ijau"];
    C1["ia"] += alpha * H1["BM"] * T2["iMaB"];
    C1["ia"] += alpha * H1["BU"] * Gamma1_["UV"] * T2["iVaB"];
    C1["ia"] -= alpha * H1["VJ"] * Gamma1_["UV"] * T2["iJaU"];

    C1["IA"] += alpha * H1["bm"] * T2["mIbA"];
    C1["IA"] += alpha * H1["bu"] * Gamma1_["uv"] * T2["vIbA"];
    C1["IA"] -= alpha * H1["vj"] * Gamma1_["uv"] * T2["jIuA"];
    C1["IA"] += alpha * H1["BM"] * T2["IMAB"];
    C1["IA"] += alpha * H1["BU"] * Gamma1_["UV"] * T2["IVAB"];
    C1["IA"] -= alpha * H1["VJ"] * Gamma1_["UV"] * T2["IJAU"];

    if (print_ > 2) {
        outfile->Printf("\n    Time for [H1, T2] -> C1 : %12.3f", timer.elapsed());
    }
    dsrg_time_.add("121", timer.elapsed());
}

// void MASTER_DSRG::H1_T2_C1aa(BlockedTensor& H1, BlockedTensor& T2, const double& alpha,
//                             BlockedTensor& C1) {
//    Timer timer;

//    C1["xy"] += alpha * H1["bm"] * T2["xmyb"];
//    C1["xy"] += alpha * H1["bu"] * Gamma1_["uv"] * T2["xvyb"];
//    C1["xy"] -= alpha * H1["vj"] * Gamma1_["uv"] * T2["xjyu"];
//    C1["xy"] += alpha * H1["BM"] * T2["xMyB"];
//    C1["xy"] += alpha * H1["BU"] * Gamma1_["UV"] * T2["xVyB"];
//    C1["xy"] -= alpha * H1["VJ"] * Gamma1_["UV"] * T2["xJyU"];

//    C1["XY"] += alpha * H1["bm"] * T2["mXbY"];
//    C1["XY"] += alpha * H1["bu"] * Gamma1_["uv"] * T2["vXbY"];
//    C1["XY"] -= alpha * H1["vj"] * Gamma1_["uv"] * T2["jXuY"];
//    C1["XY"] += alpha * H1["BM"] * T2["XMYB"];
//    C1["XY"] += alpha * H1["BU"] * Gamma1_["UV"] * T2["XVYB"];
//    C1["XY"] -= alpha * H1["VJ"] * Gamma1_["UV"] * T2["XJYU"];

//    if (print_ > 2) {
//        outfile->Printf("\n    Time for [H1, T2] -> C1aa : %12.3f", timer.get());
//    }
//    dsrg_time_.add("121", timer.get());
//}

// void MASTER_DSRG::H1_T2_C1ph(BlockedTensor& H1, BlockedTensor& T2, const double& alpha,
//                             BlockedTensor& C1) {
//    H1_T2_C1aa(H1, T2, alpha, C1);
//}

// void MASTER_DSRG::H1_T2_C1hp(BlockedTensor& H1, BlockedTensor& T2, const double& alpha,
//                             BlockedTensor& C1) {
//    H1_T2_C1(H1, T2, alpha, C1);
//}

void MASTER_DSRG::H2_T1_C1(BlockedTensor& H2, BlockedTensor& T1, const double& alpha,
                           BlockedTensor& C1) {
    ForteTimer timer;

    C1["qp"] += alpha * T1["ma"] * H2["qapm"];
    C1["qp"] += alpha * T1["xe"] * Gamma1_["yx"] * H2["qepy"];
    C1["qp"] -= alpha * T1["mu"] * Gamma1_["uv"] * H2["qvpm"];
    C1["qp"] += alpha * T1["MA"] * H2["qApM"];
    C1["qp"] += alpha * T1["XE"] * Gamma1_["YX"] * H2["qEpY"];
    C1["qp"] -= alpha * T1["MU"] * Gamma1_["UV"] * H2["qVpM"];

    C1["QP"] += alpha * T1["ma"] * H2["aQmP"];
    C1["QP"] += alpha * T1["xe"] * Gamma1_["yx"] * H2["eQyP"];
    C1["QP"] -= alpha * T1["mu"] * Gamma1_["uv"] * H2["vQmP"];
    C1["QP"] += alpha * T1["MA"] * H2["QAPM"];
    C1["QP"] += alpha * T1["XE"] * Gamma1_["YX"] * H2["QEPY"];
    C1["QP"] -= alpha * T1["MU"] * Gamma1_["UV"] * H2["QVPM"];

    if (print_ > 2) {
        outfile->Printf("\n    Time for [H2, T1] -> C1 : %12.3f", timer.elapsed());
    }
    dsrg_time_.add("211", timer.elapsed());
}

// void MASTER_DSRG::H2_T1_C1aa(BlockedTensor& H2, BlockedTensor& T1, const double& alpha,
//                             BlockedTensor& C1) {
//    Timer timer;

//    C1["xy"] += alpha * T1["ma"] * H2["xaym"];
//    C1["xy"] += alpha * T1["ue"] * Gamma1_["vu"] * H2["xeyv"];
//    C1["xy"] -= alpha * T1["mu"] * Gamma1_["uv"] * H2["xvym"];
//    C1["xy"] += alpha * T1["MA"] * H2["xAyM"];
//    C1["xy"] += alpha * T1["UE"] * Gamma1_["VU"] * H2["xEyV"];
//    C1["xy"] -= alpha * T1["MU"] * Gamma1_["UV"] * H2["xVyM"];

//    C1["XY"] += alpha * T1["ma"] * H2["aXmY"];
//    C1["XY"] += alpha * T1["ue"] * Gamma1_["vu"] * H2["eXvY"];
//    C1["XY"] -= alpha * T1["mu"] * Gamma1_["uv"] * H2["vXmY"];
//    C1["XY"] += alpha * T1["MA"] * H2["XAYM"];
//    C1["XY"] += alpha * T1["UE"] * Gamma1_["VU"] * H2["XEYV"];
//    C1["XY"] -= alpha * T1["MU"] * Gamma1_["UV"] * H2["XVYM"];

//    if (print_ > 2) {
//        outfile->Printf("\n    Time for [H2, T1] -> C1aa : %12.3f", timer.get());
//    }
//    dsrg_time_.add("211", timer.get());
//}

// void MASTER_DSRG::H2_T1_C1hp(BlockedTensor& H2, BlockedTensor& T1, const double& alpha,
//                             BlockedTensor& C1) {
//    Timer timer;

//    C1["kc"] += alpha * T1["ma"] * H2["kacm"];
//    C1["kc"] += alpha * T1["xe"] * Gamma1_["yx"] * H2["kecy"];
//    C1["kc"] -= alpha * T1["mu"] * Gamma1_["uv"] * H2["kvcm"];
//    C1["kc"] += alpha * T1["MA"] * H2["kAcM"];
//    C1["kc"] += alpha * T1["XE"] * Gamma1_["YX"] * H2["kEcY"];
//    C1["kc"] -= alpha * T1["MU"] * Gamma1_["UV"] * H2["kVcM"];

//    C1["KC"] += alpha * T1["ma"] * H2["aKmC"];
//    C1["KC"] += alpha * T1["xe"] * Gamma1_["yx"] * H2["eKyC"];
//    C1["KC"] -= alpha * T1["mu"] * Gamma1_["uv"] * H2["vKmC"];
//    C1["KC"] += alpha * T1["MA"] * H2["KACM"];
//    C1["KC"] += alpha * T1["XE"] * Gamma1_["YX"] * H2["KECY"];
//    C1["KC"] -= alpha * T1["MU"] * Gamma1_["UV"] * H2["KVCM"];

//    if (print_ > 2) {
//        outfile->Printf("\n    Time for [H2, T1] -> C1hp : %12.3f", timer.get());
//    }
//    dsrg_time_.add("211", timer.get());
//}

// void MASTER_DSRG::H2_T1_C1ph(BlockedTensor& H2, BlockedTensor& T1, const double& alpha,
//                           BlockedTensor& C1) {
//    Timer timer;

//    C1["ck"] += alpha * T1["ma"] * H2["cakm"];
//    C1["ck"] += alpha * T1["xe"] * Gamma1_["yx"] * H2["ceky"];
//    C1["ck"] -= alpha * T1["mu"] * Gamma1_["uv"] * H2["cvkm"];
//    C1["ck"] += alpha * T1["MA"] * H2["cAkM"];
//    C1["ck"] += alpha * T1["XE"] * Gamma1_["YX"] * H2["cEkY"];
//    C1["ck"] -= alpha * T1["MU"] * Gamma1_["UV"] * H2["cVkM"];

//    C1["CK"] += alpha * T1["ma"] * H2["aCmK"];
//    C1["CK"] += alpha * T1["xe"] * Gamma1_["yx"] * H2["eCyK"];
//    C1["CK"] -= alpha * T1["mu"] * Gamma1_["uv"] * H2["vCmK"];
//    C1["CK"] += alpha * T1["MA"] * H2["CAKM"];
//    C1["CK"] += alpha * T1["XE"] * Gamma1_["YX"] * H2["CEKY"];
//    C1["CK"] -= alpha * T1["MU"] * Gamma1_["UV"] * H2["CVKM"];

//    if (print_ > 2) {
//        outfile->Printf("\n    Time for [H2, T1] -> C1ph : %12.3f", timer.get());
//    }
//    dsrg_time_.add("211", timer.get());
//}

void MASTER_DSRG::H2_T2_C1(BlockedTensor& H2, BlockedTensor& T2, const double& alpha,
                           BlockedTensor& C1) {
    ForteTimer timer;
    BlockedTensor temp;

    /// max intermediate: a * a * p * p

    // [Hbar2, T2] (C_2)^3 -> C1 particle contractions
    C1["ir"] += 0.5 * alpha * H2["abrm"] * T2["imab"];
    C1["ir"] += alpha * H2["aBrM"] * T2["iMaB"];
    C1["IR"] += 0.5 * alpha * H2["ABRM"] * T2["IMAB"];
    C1["IR"] += alpha * H2["aBmR"] * T2["mIaB"];

    C1["ir"] += 0.5 * alpha * Gamma1_["uv"] * T2["ivab"] * H2["abru"];
    C1["ir"] += alpha * Gamma1_["UV"] * T2["iVaB"] * H2["aBrU"];
    C1["IR"] += 0.5 * alpha * Gamma1_["UV"] * T2["IVAB"] * H2["ABRU"];
    C1["IR"] += alpha * Gamma1_["uv"] * T2["vIaB"] * H2["aBuR"];

    C1["ir"] += 0.5 * alpha * T2["ijux"] * Gamma1_["xy"] * Gamma1_["uv"] * H2["vyrj"];
    C1["IR"] += 0.5 * alpha * T2["IJUX"] * Gamma1_["XY"] * Gamma1_["UV"] * H2["VYRJ"];
    temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"hHaA"});
    temp["iJvY"] = T2["iJuX"] * Gamma1_["XY"] * Gamma1_["uv"];
    C1["ir"] += alpha * temp["iJvY"] * H2["vYrJ"];
    C1["IR"] += alpha * temp["jIvY"] * H2["vYjR"];

    C1["ir"] -= alpha * Gamma1_["uv"] * T2["imub"] * H2["vbrm"];
    C1["ir"] -= alpha * Gamma1_["uv"] * T2["iMuB"] * H2["vBrM"];
    C1["ir"] -= alpha * Gamma1_["UV"] * T2["iMbU"] * H2["bVrM"];
    C1["IR"] -= alpha * Gamma1_["UV"] * T2["IMUB"] * H2["VBRM"];
    C1["IR"] -= alpha * Gamma1_["UV"] * T2["mIbU"] * H2["bVmR"];
    C1["IR"] -= alpha * Gamma1_["uv"] * T2["mIuB"] * H2["vBmR"];

    C1["ir"] -= alpha * T2["iyub"] * Gamma1_["uv"] * Gamma1_["xy"] * H2["vbrx"];
    C1["ir"] -= alpha * T2["iYuB"] * Gamma1_["uv"] * Gamma1_["XY"] * H2["vBrX"];
    C1["ir"] -= alpha * T2["iYbU"] * Gamma1_["XY"] * Gamma1_["UV"] * H2["bVrX"];
    C1["IR"] -= alpha * T2["IYUB"] * Gamma1_["UV"] * Gamma1_["XY"] * H2["VBRX"];
    C1["IR"] -= alpha * T2["yIuB"] * Gamma1_["uv"] * Gamma1_["xy"] * H2["vBxR"];
    C1["IR"] -= alpha * T2["yIbU"] * Gamma1_["UV"] * Gamma1_["xy"] * H2["bVxR"];

    // [Hbar2, T2] (C_2)^3 -> C1 hole contractions
    C1["pa"] -= 0.5 * alpha * H2["peij"] * T2["ijae"];
    C1["pa"] -= alpha * H2["pEiJ"] * T2["iJaE"];
    C1["PA"] -= 0.5 * alpha * H2["PEIJ"] * T2["IJAE"];
    C1["PA"] -= alpha * H2["ePiJ"] * T2["iJeA"];

    C1["pa"] -= 0.5 * alpha * Eta1_["uv"] * T2["ijau"] * H2["pvij"];
    C1["pa"] -= alpha * Eta1_["UV"] * T2["iJaU"] * H2["pViJ"];
    C1["PA"] -= 0.5 * alpha * Eta1_["UV"] * T2["IJAU"] * H2["PVIJ"];
    C1["PA"] -= alpha * Eta1_["uv"] * T2["iJuA"] * H2["vPiJ"];

    C1["pa"] -= 0.5 * alpha * T2["vyab"] * Eta1_["uv"] * Eta1_["xy"] * H2["pbux"];
    C1["PA"] -= 0.5 * alpha * T2["VYAB"] * Eta1_["UV"] * Eta1_["XY"] * H2["PBUX"];
    temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"aApP"});
    temp["uXaB"] = T2["vYaB"] * Eta1_["uv"] * Eta1_["XY"];
    C1["pa"] -= alpha * H2["pBuX"] * temp["uXaB"];
    C1["PA"] -= alpha * H2["bPuX"] * temp["uXbA"];

    C1["pa"] += alpha * Eta1_["uv"] * T2["vjae"] * H2["peuj"];
    C1["pa"] += alpha * Eta1_["uv"] * T2["vJaE"] * H2["pEuJ"];
    C1["pa"] += alpha * Eta1_["UV"] * T2["jVaE"] * H2["pEjU"];
    C1["PA"] += alpha * Eta1_["UV"] * T2["VJAE"] * H2["PEUJ"];
    C1["PA"] += alpha * Eta1_["uv"] * T2["vJeA"] * H2["ePuJ"];
    C1["PA"] += alpha * Eta1_["UV"] * T2["jVeA"] * H2["ePjU"];

    C1["pa"] += alpha * T2["vjax"] * Eta1_["uv"] * Eta1_["xy"] * H2["pyuj"];
    C1["pa"] += alpha * T2["vJaX"] * Eta1_["uv"] * Eta1_["XY"] * H2["pYuJ"];
    C1["pa"] += alpha * T2["jVaX"] * Eta1_["XY"] * Eta1_["UV"] * H2["pYjU"];
    C1["PA"] += alpha * T2["VJAX"] * Eta1_["UV"] * Eta1_["XY"] * H2["PYUJ"];
    C1["PA"] += alpha * T2["vJxA"] * Eta1_["uv"] * Eta1_["xy"] * H2["yPuJ"];
    C1["PA"] += alpha * T2["jVxA"] * Eta1_["UV"] * Eta1_["xy"] * H2["yPjU"];

    // [Hbar2, T2] C_4 C_2 2:2 -> C1
    C1["ir"] += 0.25 * alpha * T2["ijxy"] * Lambda2_["xyuv"] * H2["uvrj"];
    C1["IR"] += 0.25 * alpha * T2["IJXY"] * Lambda2_["XYUV"] * H2["UVRJ"];
    temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"hHaA"});
    temp["iJuV"] = T2["iJxY"] * Lambda2_["xYuV"];
    C1["ir"] += alpha * H2["uVrJ"] * temp["iJuV"];
    C1["IR"] += alpha * H2["uVjR"] * temp["jIuV"];

    C1["pa"] -= 0.25 * alpha * Lambda2_["xyuv"] * T2["uvab"] * H2["pbxy"];
    C1["PA"] -= 0.25 * alpha * Lambda2_["XYUV"] * T2["UVAB"] * H2["PBXY"];
    temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"aApP"});
    temp["xYaB"] = T2["uVaB"] * Lambda2_["xYuV"];
    C1["pa"] -= alpha * H2["pBxY"] * temp["xYaB"];
    C1["PA"] -= alpha * H2["bPxY"] * temp["xYbA"];

    C1["ir"] -= alpha * Lambda2_["yXuV"] * T2["iVyA"] * H2["uArX"];
    C1["IR"] -= alpha * Lambda2_["xYvU"] * T2["vIaY"] * H2["aUxR"];
    C1["pa"] += alpha * Lambda2_["xYvU"] * T2["vIaY"] * H2["pUxI"];
    C1["PA"] += alpha * Lambda2_["yXuV"] * T2["iVyA"] * H2["uPiX"];

    temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"hapa"});
    temp["ixau"] += Lambda2_["xyuv"] * T2["ivay"];
    temp["ixau"] += Lambda2_["xYuV"] * T2["iVaY"];
    C1["ir"] += alpha * temp["ixau"] * H2["aurx"];
    C1["pa"] -= alpha * H2["puix"] * temp["ixau"];
    temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"hApA"});
    temp["iXaU"] += Lambda2_["XYUV"] * T2["iVaY"];
    temp["iXaU"] += Lambda2_["yXvU"] * T2["ivay"];
    C1["ir"] += alpha * temp["iXaU"] * H2["aUrX"];
    C1["pa"] -= alpha * H2["pUiX"] * temp["iXaU"];
    temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"aHaP"});
    temp["xIuA"] += Lambda2_["xyuv"] * T2["vIyA"];
    temp["xIuA"] += Lambda2_["xYuV"] * T2["VIYA"];
    C1["IR"] += alpha * temp["xIuA"] * H2["uAxR"];
    C1["PA"] -= alpha * H2["uPxI"] * temp["xIuA"];
    temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"HAPA"});
    temp["IXAU"] += Lambda2_["XYUV"] * T2["IVAY"];
    temp["IXAU"] += Lambda2_["yXvU"] * T2["vIyA"];
    C1["IR"] += alpha * temp["IXAU"] * H2["AURX"];
    C1["PA"] -= alpha * H2["PUIX"] * temp["IXAU"];

    // [Hbar2, T2] C_4 C_2 1:3 -> C1
    temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"pa"});
    temp["au"] += 0.5 * Lambda2_["xyuv"] * H2["avxy"];
    temp["au"] += Lambda2_["xYuV"] * H2["aVxY"];
    C1["jb"] += alpha * temp["au"] * T2["ujab"];
    C1["JB"] += alpha * temp["au"] * T2["uJaB"];
    temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"PA"});
    temp["AU"] += 0.5 * Lambda2_["XYUV"] * H2["AVXY"];
    temp["AU"] += Lambda2_["xYvU"] * H2["vAxY"];
    C1["jb"] += alpha * temp["AU"] * T2["jUbA"];
    C1["JB"] += alpha * temp["AU"] * T2["UJAB"];

    temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"ah"});
    temp["xi"] += 0.5 * Lambda2_["xyuv"] * H2["uviy"];
    temp["xi"] += Lambda2_["xYuV"] * H2["uViY"];
    C1["jb"] -= alpha * temp["xi"] * T2["ijxb"];
    C1["JB"] -= alpha * temp["xi"] * T2["iJxB"];
    temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"AH"});
    temp["XI"] += 0.5 * Lambda2_["XYUV"] * H2["UVIY"];
    temp["XI"] += Lambda2_["yXvU"] * H2["vUyI"];
    C1["jb"] -= alpha * temp["XI"] * T2["jIbX"];
    C1["JB"] -= alpha * temp["XI"] * T2["IJXB"];

    temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"av"});
    temp["xe"] += 0.5 * T2["uvey"] * Lambda2_["xyuv"];
    temp["xe"] += T2["uVeY"] * Lambda2_["xYuV"];
    C1["qs"] += alpha * temp["xe"] * H2["eqxs"];
    C1["QS"] += alpha * temp["xe"] * H2["eQxS"];
    temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"AV"});
    temp["XE"] += 0.5 * T2["UVEY"] * Lambda2_["XYUV"];
    temp["XE"] += T2["uVyE"] * Lambda2_["yXuV"];
    C1["qs"] += alpha * temp["XE"] * H2["qEsX"];
    C1["QS"] += alpha * temp["XE"] * H2["EQXS"];

    temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"ca"});
    temp["mu"] += 0.5 * T2["mvxy"] * Lambda2_["xyuv"];
    temp["mu"] += T2["mVxY"] * Lambda2_["xYuV"];
    C1["qs"] -= alpha * temp["mu"] * H2["uqms"];
    C1["QS"] -= alpha * temp["mu"] * H2["uQmS"];
    temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"CA"});
    temp["MU"] += 0.5 * T2["MVXY"] * Lambda2_["XYUV"];
    temp["MU"] += T2["vMxY"] * Lambda2_["xYvU"];
    C1["qs"] -= alpha * temp["MU"] * H2["qUsM"];
    C1["QS"] -= alpha * temp["MU"] * H2["UQMS"];

    if (print_ > 2) {
        outfile->Printf("\n    Time for [H2, T2] -> C1 : %12.3f", timer.elapsed());
    }
    dsrg_time_.add("221", timer.elapsed());
}

void MASTER_DSRG::H1_T2_C2(BlockedTensor& H1, BlockedTensor& T2, const double& alpha,
                           BlockedTensor& C2) {
    ForteTimer timer;

    C2["ijpb"] += alpha * T2["ijab"] * H1["ap"];
    C2["ijap"] += alpha * T2["ijab"] * H1["bp"];
    C2["qjab"] -= alpha * T2["ijab"] * H1["qi"];
    C2["iqab"] -= alpha * T2["ijab"] * H1["qj"];

    C2["iJpB"] += alpha * T2["iJaB"] * H1["ap"];
    C2["iJaP"] += alpha * T2["iJaB"] * H1["BP"];
    C2["qJaB"] -= alpha * T2["iJaB"] * H1["qi"];
    C2["iQaB"] -= alpha * T2["iJaB"] * H1["QJ"];

    C2["IJPB"] += alpha * T2["IJAB"] * H1["AP"];
    C2["IJAP"] += alpha * T2["IJAB"] * H1["BP"];
    C2["QJAB"] -= alpha * T2["IJAB"] * H1["QI"];
    C2["IQAB"] -= alpha * T2["IJAB"] * H1["QJ"];

    if (print_ > 2) {
        outfile->Printf("\n    Time for [H1, T2] -> C2 : %12.3f", timer.elapsed());
    }
    dsrg_time_.add("122", timer.elapsed());
}

void MASTER_DSRG::H2_T1_C2(BlockedTensor& H2, BlockedTensor& T1, const double& alpha,
                           BlockedTensor& C2) {
    ForteTimer timer;

    C2["irpq"] += alpha * T1["ia"] * H2["arpq"];
    C2["ripq"] += alpha * T1["ia"] * H2["rapq"];
    C2["rsaq"] -= alpha * T1["ia"] * H2["rsiq"];
    C2["rspa"] -= alpha * T1["ia"] * H2["rspi"];

    C2["iRpQ"] += alpha * T1["ia"] * H2["aRpQ"];
    C2["rIpQ"] += alpha * T1["IA"] * H2["rApQ"];
    C2["rSaQ"] -= alpha * T1["ia"] * H2["rSiQ"];
    C2["rSpA"] -= alpha * T1["IA"] * H2["rSpI"];

    C2["IRPQ"] += alpha * T1["IA"] * H2["ARPQ"];
    C2["RIPQ"] += alpha * T1["IA"] * H2["RAPQ"];
    C2["RSAQ"] -= alpha * T1["IA"] * H2["RSIQ"];
    C2["RSPA"] -= alpha * T1["IA"] * H2["RSPI"];

    if (print_ > 2) {
        outfile->Printf("\n    Time for [H2, T1] -> C2 : %12.3f", timer.elapsed());
    }
    dsrg_time_.add("212", timer.elapsed());
}

void MASTER_DSRG::H2_T2_C2(BlockedTensor& H2, BlockedTensor& T2, const double& alpha,
                           BlockedTensor& C2) {
    ForteTimer timer;

    /// max intermediate: g * g * p * p

    // particle-particle contractions
    C2["ijrs"] += 0.5 * alpha * H2["abrs"] * T2["ijab"];
    C2["iJrS"] += alpha * H2["aBrS"] * T2["iJaB"];
    C2["IJRS"] += 0.5 * alpha * H2["ABRS"] * T2["IJAB"];

    C2["ijrs"] -= alpha * Gamma1_["xy"] * T2["ijxb"] * H2["ybrs"];
    C2["iJrS"] -= alpha * Gamma1_["xy"] * T2["iJxB"] * H2["yBrS"];
    C2["iJrS"] -= alpha * Gamma1_["XY"] * T2["iJbX"] * H2["bYrS"];
    C2["IJRS"] -= alpha * Gamma1_["XY"] * T2["IJXB"] * H2["YBRS"];

    // hole-hole contractions
    C2["pqab"] += 0.5 * alpha * H2["pqij"] * T2["ijab"];
    C2["pQaB"] += alpha * H2["pQiJ"] * T2["iJaB"];
    C2["PQAB"] += 0.5 * alpha * H2["PQIJ"] * T2["IJAB"];

    C2["pqab"] -= alpha * Eta1_["xy"] * T2["yjab"] * H2["pqxj"];
    C2["pQaB"] -= alpha * Eta1_["xy"] * T2["yJaB"] * H2["pQxJ"];
    C2["pQaB"] -= alpha * Eta1_["XY"] * T2["jYaB"] * H2["pQjX"];
    C2["PQAB"] -= alpha * Eta1_["XY"] * T2["YJAB"] * H2["PQXJ"];

    // hole-particle contractions
    // figure out useful blocks of temp (assume symmetric C2 blocks, if cavv exists => acvv exists)
    std::vector<std::string> C2_blocks(C2.block_labels());
    std::sort(C2_blocks.begin(), C2_blocks.end());
    std::vector<std::string> temp_blocks;
    for (const std::string& p : {"c", "a", "v"}) {
        for (const std::string& q : {"c", "a"}) {
            for (const std::string& r : {"c", "a", "v"}) {
                for (const std::string& s : {"a", "v"}) {
                    temp_blocks.emplace_back(p + q + r + s);
                }
            }
        }
    }
    std::sort(temp_blocks.begin(), temp_blocks.end());
    std::vector<std::string> blocks;
    std::set_intersection(temp_blocks.begin(), temp_blocks.end(), C2_blocks.begin(),
                          C2_blocks.end(), std::back_inserter(blocks));
    BlockedTensor temp;
    if (blocks.size() != 0) {
        temp = ambit::BlockedTensor::build(tensor_type_, "temp", blocks);
        temp["qjsb"] += alpha * H2["aqms"] * T2["mjab"];
        temp["qjsb"] += alpha * H2["qAsM"] * T2["jMbA"];
        temp["qjsb"] += alpha * Gamma1_["xy"] * T2["yjab"] * H2["aqxs"];
        temp["qjsb"] += alpha * Gamma1_["XY"] * T2["jYbA"] * H2["qAsX"];
        temp["qjsb"] -= alpha * Gamma1_["xy"] * T2["ijxb"] * H2["yqis"];
        temp["qjsb"] -= alpha * Gamma1_["XY"] * T2["jIbX"] * H2["qYsI"];
        C2["qjsb"] += temp["qjsb"];
        C2["jqsb"] -= temp["qjsb"];
        C2["qjbs"] -= temp["qjsb"];
        C2["jqbs"] += temp["qjsb"];
    }

    // figure out useful blocks of temp (assume symmetric C2 blocks, if cavv exists => acvv exists)
    temp_blocks.clear();
    for (const std::string& p : {"C", "A", "V"}) {
        for (const std::string& q : {"C", "A"}) {
            for (const std::string& r : {"C", "A", "V"}) {
                for (const std::string& s : {"A", "V"}) {
                    temp_blocks.emplace_back(p + q + r + s);
                }
            }
        }
    }
    std::sort(temp_blocks.begin(), temp_blocks.end());
    blocks.clear();
    std::set_intersection(temp_blocks.begin(), temp_blocks.end(), C2_blocks.begin(),
                          C2_blocks.end(), std::back_inserter(blocks));
    if (blocks.size() != 0) {
        temp = ambit::BlockedTensor::build(tensor_type_, "temp", blocks);
        temp["QJSB"] += alpha * H2["AQMS"] * T2["MJAB"];
        temp["QJSB"] += alpha * H2["aQmS"] * T2["mJaB"];
        temp["QJSB"] += alpha * Gamma1_["XY"] * T2["YJAB"] * H2["AQXS"];
        temp["QJSB"] += alpha * Gamma1_["xy"] * T2["yJaB"] * H2["aQxS"];
        temp["QJSB"] -= alpha * Gamma1_["XY"] * T2["IJXB"] * H2["YQIS"];
        temp["QJSB"] -= alpha * Gamma1_["xy"] * T2["iJxB"] * H2["yQiS"];
        C2["QJSB"] += temp["QJSB"];
        C2["JQSB"] -= temp["QJSB"];
        C2["QJBS"] -= temp["QJSB"];
        C2["JQBS"] += temp["QJSB"];
    }

    C2["qJsB"] += alpha * H2["aqms"] * T2["mJaB"];
    C2["qJsB"] += alpha * H2["qAsM"] * T2["MJAB"];
    C2["qJsB"] += alpha * Gamma1_["xy"] * T2["yJaB"] * H2["aqxs"];
    C2["qJsB"] += alpha * Gamma1_["XY"] * T2["YJAB"] * H2["qAsX"];
    C2["qJsB"] -= alpha * Gamma1_["xy"] * T2["iJxB"] * H2["yqis"];
    C2["qJsB"] -= alpha * Gamma1_["XY"] * T2["IJXB"] * H2["qYsI"];

    C2["iQsB"] -= alpha * T2["iMaB"] * H2["aQsM"];
    C2["iQsB"] -= alpha * Gamma1_["XY"] * T2["iYaB"] * H2["aQsX"];
    C2["iQsB"] += alpha * Gamma1_["xy"] * T2["iJxB"] * H2["yQsJ"];

    C2["qJaS"] -= alpha * T2["mJaB"] * H2["qBmS"];
    C2["qJaS"] -= alpha * Gamma1_["xy"] * T2["yJaB"] * H2["qBxS"];
    C2["qJaS"] += alpha * Gamma1_["XY"] * T2["iJaX"] * H2["qYiS"];

    C2["iQaS"] += alpha * T2["imab"] * H2["bQmS"];
    C2["iQaS"] += alpha * T2["iMaB"] * H2["BQMS"];
    C2["iQaS"] += alpha * Gamma1_["xy"] * T2["iyab"] * H2["bQxS"];
    C2["iQaS"] += alpha * Gamma1_["XY"] * T2["iYaB"] * H2["BQXS"];
    C2["iQaS"] -= alpha * Gamma1_["xy"] * T2["ijax"] * H2["yQjS"];
    C2["iQaS"] -= alpha * Gamma1_["XY"] * T2["iJaX"] * H2["YQJS"];

    if (print_ > 2) {
        outfile->Printf("\n    Time for [H2, T2] -> C2 : %12.3f", timer.elapsed());
    }
    dsrg_time_.add("222", timer.elapsed());
}

void MASTER_DSRG::H2_T2_C3(BlockedTensor& H2, BlockedTensor& T2, const double& alpha,
                           BlockedTensor& C3, const bool& active_only) {
    ForteTimer timer;

    /// Potentially be as large as p * p * h * g * g * g

    BlockedTensor temp;

    // aaa and bbb
    if (active_only) {
        temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"aaaaaa"});
        temp["rsjabq"] -= alpha * H2["rsqi"] * T2["ijab"];
        temp["ijspqb"] += alpha * H2["aspq"] * T2["ijba"];
        C3["xyzuvw"] += temp["xyzuvw"];
        C3["zxyuvw"] += temp["xyzuvw"];
        C3["xzyuvw"] -= temp["xyzuvw"];
        C3["xyzwuv"] += temp["xyzuvw"];
        C3["zxywuv"] += temp["xyzuvw"];
        C3["xzywuv"] -= temp["xyzuvw"];
        C3["xyzuwv"] -= temp["xyzuvw"];
        C3["zxyuwv"] -= temp["xyzuvw"];
        C3["xzyuwv"] += temp["xyzuvw"];

        temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"AAAAAA"});
        temp["RSJABQ"] -= alpha * H2["RSQI"] * T2["IJAB"];
        temp["IJSPQB"] += alpha * H2["ASPQ"] * T2["IJBA"];
        C3["XYZUVW"] += temp["XYZUVW"];
        C3["ZXYUVW"] += temp["XYZUVW"];
        C3["XZYUVW"] -= temp["XYZUVW"];
        C3["XYZWUV"] += temp["XYZUVW"];
        C3["ZXYWUV"] += temp["XYZUVW"];
        C3["XZYWUV"] -= temp["XYZUVW"];
        C3["XYZUWV"] -= temp["XYZUVW"];
        C3["ZXYUWV"] -= temp["XYZUVW"];
        C3["XZYUWV"] += temp["XYZUVW"];
    } else {
        temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"gghppg"});
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

        temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"hhgggp"});
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

        temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"GGHPPG"});
        temp["RSJABQ"] -= H2["RSQI"] * T2["IJAB"];
        C3["RSJABQ"] += alpha * temp["RSJABQ"];
        C3["RJSABQ"] -= alpha * temp["RSJABQ"];
        C3["JRSABQ"] += alpha * temp["RSJABQ"];
        C3["RSJAQB"] -= alpha * temp["RSJABQ"];
        C3["RJSAQB"] += alpha * temp["RSJABQ"];
        C3["JRSAQB"] -= alpha * temp["RSJABQ"];
        C3["RSJQAB"] += alpha * temp["RSJABQ"];
        C3["RJSQAB"] -= alpha * temp["RSJABQ"];
        C3["JRSQAB"] += alpha * temp["RSJABQ"];

        temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"HHGGGP"});
        temp["IJSPQB"] += H2["ASPQ"] * T2["IJBA"];
        C3["IJSPQB"] += alpha * temp["IJSPQB"];
        C3["ISJPQB"] -= alpha * temp["IJSPQB"];
        C3["SIJPQB"] += alpha * temp["IJSPQB"];
        C3["IJSPBQ"] -= alpha * temp["IJSPQB"];
        C3["ISJPBQ"] += alpha * temp["IJSPQB"];
        C3["SIJPBQ"] -= alpha * temp["IJSPQB"];
        C3["IJSBPQ"] += alpha * temp["IJSPQB"];
        C3["ISJBPQ"] -= alpha * temp["IJSPQB"];
        C3["SIJBPQ"] += alpha * temp["IJSPQB"];
    }

    // aab hole contraction
    C3["rjSabQ"] -= alpha * H2["rSiQ"] * T2["ijab"];
    C3["jrSabQ"] += alpha * H2["rSiQ"] * T2["ijab"];

    C3["rsJaqB"] += alpha * H2["rsqi"] * T2["iJaB"];
    C3["rsJqaB"] -= alpha * H2["rsqi"] * T2["iJaB"];

    if (active_only) {
        temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"aAaaAa"});
    } else {
        temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"gGhpPg"});
    }
    temp["rSjaBq"] += H2["rSqI"] * T2["jIaB"];
    C3["rjSaqB"] += alpha * temp["rSjaBq"];
    C3["jrSaqB"] -= alpha * temp["rSjaBq"];
    C3["rjSqaB"] -= alpha * temp["rSjaBq"];
    C3["jrSqaB"] += alpha * temp["rSjaBq"];

    // aab particle contraction
    C3["isJpqB"] += alpha * H2["aspq"] * T2["iJaB"];
    C3["siJpqB"] -= alpha * H2["aspq"] * T2["iJaB"];

    C3["ijSpbQ"] -= alpha * H2["aSpQ"] * T2["ijba"];
    C3["ijSbpQ"] += alpha * H2["aSpQ"] * T2["ijba"];

    if (active_only) {
        temp.zero();
    } else {
        temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"hHggGp"});
    }
    temp["iJspQb"] -= H2["sApQ"] * T2["iJbA"];
    C3["isJpbQ"] += alpha * temp["iJspQb"];
    C3["siJpbQ"] -= alpha * temp["iJspQb"];
    C3["isJbpQ"] -= alpha * temp["iJspQb"];
    C3["siJbpQ"] += alpha * temp["iJspQb"];

    // abb hole contraction
    if (active_only) {
        temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"aAAaAA"});
    } else {
        temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"gGHpPG"});
    }
    temp["rSJaBQ"] += H2["rSiQ"] * T2["iJaB"];
    C3["rSJaBQ"] += alpha * temp["rSJaBQ"];
    C3["rJSaBQ"] -= alpha * temp["rSJaBQ"];
    C3["rSJaQB"] -= alpha * temp["rSJaBQ"];
    C3["rJSaQB"] += alpha * temp["rSJaBQ"];

    C3["jRSaBQ"] += alpha * H2["RSQI"] * T2["jIaB"];
    C3["jRSaQB"] -= alpha * H2["RSQI"] * T2["jIaB"];

    C3["rSJqAB"] -= alpha * H2["rSqI"] * T2["IJAB"];
    C3["rJSqAB"] += alpha * H2["rSqI"] * T2["IJAB"];

    // abb particle contraction
    if (active_only) {
        temp.zero();
    } else {
        temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"hHGgGP"});
    }
    temp["iJSpQB"] -= H2["aSpQ"] * T2["iJaB"];
    C3["iJSpQB"] += alpha * temp["iJSpQB"];
    C3["iSJpQB"] -= alpha * temp["iJSpQB"];
    C3["iJSpBQ"] -= alpha * temp["iJSpQB"];
    C3["iSJpBQ"] += alpha * temp["iJSpQB"];

    C3["sIJpQB"] -= alpha * H2["sApQ"] * T2["IJBA"];
    C3["sIJpBQ"] += alpha * H2["sApQ"] * T2["IJBA"];

    C3["iJSbPQ"] += alpha * H2["ASPQ"] * T2["iJbA"];
    C3["iSJbPQ"] -= alpha * H2["ASPQ"] * T2["iJbA"];

    if (print_ > 2) {
        outfile->Printf("\n    Time for [H2, T2] -> C3 : %12.3f", timer.elapsed());
    }
    dsrg_time_.add("223", timer.elapsed());
}

bool MASTER_DSRG::check_semi_orbs() {
    print_h2("Checking Semicanonical Orbitals");

    BlockedTensor Fd = BTF_->build(tensor_type_, "Fd", spin_cases({"cc", "aa", "vv"}));
    Fd["pq"] = Fock_["pq"];
    Fd["PQ"] = Fock_["PQ"];

    Fd.iterate([&](const std::vector<size_t>& i, const std::vector<SpinType>&, double& value) {
        if (i[0] == i[1]) {
            value = 0.0;
        }
    });

    bool semi = true;
    std::vector<double> Fmax, Fnorm;
    double e_conv = options_.get_double("E_CONVERGENCE");
    double cd = options_.get_double("CHOLESKY_TOLERANCE");
    e_conv = cd < e_conv ? e_conv : cd;
    e_conv = e_conv < 1.0e-12 ? 1.0e-12 : e_conv;
    double threshold_max = 10.0 * e_conv;
    for (const auto& block : {"cc", "aa", "vv", "CC", "AA", "VV"}) {
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

    std::string dash(2 + 16 * 3, '-');
    outfile->Printf("\n    Abs. max of Fock core, active, virtual blocks (Fij, i != j)");
    outfile->Printf("\n       %15s %15s %15s", "core", "active", "virtual");
    outfile->Printf("\n    %s", dash.c_str());
    outfile->Printf("\n    Fα %15.10f %15.10f %15.10f", Fmax[0], Fmax[1], Fmax[2]);
    outfile->Printf("\n    Fβ %15.10f %15.10f %15.10f", Fmax[3], Fmax[4], Fmax[5]);
    outfile->Printf("\n    %s\n", dash.c_str());

    outfile->Printf("\n    1-Norm of Fock core, active, virtual blocks (Fij, i != j)");
    outfile->Printf("\n       %15s %15s %15s", "core", "active", "virtual");
    outfile->Printf("\n    %s", dash.c_str());
    outfile->Printf("\n    Fα %15.10f %15.10f %15.10f", Fnorm[0], Fnorm[1], Fnorm[2]);
    outfile->Printf("\n    Fβ %15.10f %15.10f %15.10f", Fnorm[3], Fnorm[4], Fnorm[5]);
    outfile->Printf("\n    %s\n", dash.c_str());

    if (semi) {
        outfile->Printf("\n    Orbitals are semi-canonicalized.");
    } else {
        outfile->Printf("\n    Warning! Orbitals are not semi-canonicalized!");
        outfile->Printf("\n    Energy is reliable about to the same digit as max(|Fij|, i != j).");
    }

    return semi;
}

std::vector<std::string> MASTER_DSRG::diag_one_labels() {
    std::vector<std::string> labels;
    for (const std::string& p : {acore_label_, aactv_label_, avirt_label_}) {
        labels.push_back(p + p);
    }
    for (const std::string& p : {bcore_label_, bactv_label_, bvirt_label_}) {
        labels.push_back(p + p);
    }
    return labels;
}

std::vector<std::string> MASTER_DSRG::od_one_labels_hp() {
    std::vector<std::string> labels;
    for (const std::string& p : {acore_label_, aactv_label_}) {
        for (const std::string& q : {aactv_label_, avirt_label_}) {
            if (p == aactv_label_ && q == aactv_label_) {
                continue;
            }
            labels.push_back(p + q);
        }
    }

    for (const std::string& p : {bcore_label_, bactv_label_}) {
        for (const std::string& q : {bactv_label_, bvirt_label_}) {
            if (p == bactv_label_ && q == bactv_label_) {
                continue;
            }
            labels.push_back(p + q);
        }
    }
    return labels;
}

std::vector<std::string> MASTER_DSRG::od_one_labels_ph() {
    std::vector<std::string> blocks1(od_one_labels_hp());
    for (auto& block : blocks1) {
        std::swap(block[0], block[1]);
    }
    return blocks1;
}

std::vector<std::string> MASTER_DSRG::od_one_labels() {
    std::vector<std::string> labels(od_one_labels_hp());
    std::vector<std::string> temp(od_one_labels_ph());
    labels.insert(std::end(labels), std::begin(temp), std::end(temp));
    return labels;
}

std::vector<std::string> MASTER_DSRG::od_two_labels_hhpp() {
    std::map<char, std::vector<std::string>> space_map;
    space_map['h'] = {acore_label_, aactv_label_};
    space_map['H'] = {bcore_label_, bactv_label_};
    space_map['p'] = {aactv_label_, avirt_label_};
    space_map['P'] = {bactv_label_, bvirt_label_};

    auto hhpp_spin = [&](const std::string& block, const std::string& actv) {
        std::vector<std::string> out;
        for (const std::string& p : space_map[block[0]]) {
            for (const std::string& q : space_map[block[1]]) {
                for (const std::string& r : space_map[block[2]]) {
                    for (const std::string& s : space_map[block[3]]) {
                        out.push_back(p + q + r + s);
                    }
                }
            }
        }
        out.erase(std::remove(out.begin(), out.end(), actv), out.end());
        return out;
    };

    std::string aaaa = aactv_label_ + aactv_label_ + aactv_label_ + aactv_label_;
    std::string aAaA = aactv_label_ + bactv_label_ + aactv_label_ + bactv_label_;
    std::string AAAA = bactv_label_ + bactv_label_ + bactv_label_ + bactv_label_;

    std::vector<std::string> labels(hhpp_spin("hhpp", aaaa));
    std::vector<std::string> labels_ab(hhpp_spin("hHpP", aAaA));
    std::vector<std::string> labels_bb(hhpp_spin("HHPP", AAAA));

    labels.insert(std::end(labels), std::begin(labels_ab), std::end(labels_ab));
    labels.insert(std::end(labels), std::begin(labels_bb), std::end(labels_bb));

    return labels;
}

std::vector<std::string> MASTER_DSRG::od_two_labels_pphh() {
    std::vector<std::string> labels(od_two_labels_hhpp());
    for (auto& block : labels) {
        std::swap(block[0], block[2]);
        std::swap(block[1], block[3]);
    }
    return labels;
}

std::vector<std::string> MASTER_DSRG::od_two_labels() {
    std::vector<std::string> labels(od_two_labels_hhpp());
    std::vector<std::string> temp(od_two_labels_pphh());
    labels.insert(std::end(labels), std::begin(temp), std::end(temp));
    return labels;
}

std::vector<std::string> MASTER_DSRG::diag_two_labels() {
    std::map<char, std::vector<std::string>> general;
    general['a'] = {acore_label_, aactv_label_, avirt_label_};
    general['b'] = {bcore_label_, bactv_label_, bvirt_label_};

    auto all_spin = [&](const std::string& block) {
        std::vector<std::string> out;
        for (const std::string& p : general[block[0]]) {
            for (const std::string& q : general[block[1]]) {
                for (const std::string& r : general[block[2]]) {
                    for (const std::string& s : general[block[3]]) {
                        out.push_back(p + q + r + s);
                    }
                }
            }
        }
        return out;
    };

    std::vector<std::string> all(all_spin("aaaa"));
    std::vector<std::string> all_ab(all_spin("abab"));
    std::vector<std::string> all_bb(all_spin("bbbb"));

    all.insert(std::end(all), std::begin(all_ab), std::end(all_ab));
    all.insert(std::end(all), std::begin(all_bb), std::end(all_bb));

    std::vector<std::string> od(od_two_labels());
    std::sort(od.begin(), od.end());
    std::sort(all.begin(), all.end());

    std::vector<std::string> labels;
    std::set_symmetric_difference(all.begin(), all.end(), od.begin(), od.end(),
                                  std::back_inserter(labels));

    return labels;
}

std::vector<std::string> MASTER_DSRG::re_two_labels() {
    std::map<char, std::string> core, actv, virt;
    core['a'] = acore_label_;
    core['b'] = bcore_label_;
    actv['a'] = aactv_label_;
    actv['b'] = bactv_label_;
    virt['a'] = avirt_label_;
    virt['b'] = bvirt_label_;

    auto half_spin = [&](const std::string& spin) {
        const char& p = spin[0];
        const char& q = spin[1];
        std::vector<std::vector<std::string>> out{{core[p] + core[q]},
                                                  {actv[p] + actv[q]},
                                                  {virt[p] + virt[q]},
                                                  {core[p] + actv[q], actv[p] + core[q]},
                                                  {core[p] + virt[q], virt[p] + core[q]},
                                                  {actv[p] + virt[q], virt[p] + actv[q]}};
        return out;
    };
    std::vector<std::vector<std::string>> half_aa = half_spin("aa");
    std::vector<std::vector<std::string>> half_ab = half_spin("ab");
    std::vector<std::vector<std::string>> half_bb = half_spin("bb");

    std::vector<std::string> labels;
    for (const auto& halfs : {half_aa, half_ab, half_bb}) {
        for (const auto& half : halfs) {
            for (const std::string& half1 : half) {
                for (const std::string& half2 : half) {
                    labels.emplace_back(half1 + half2);
                }
            }
        }
    }

    return labels;
}
}
}
