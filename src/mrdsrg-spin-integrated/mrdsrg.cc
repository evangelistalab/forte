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

#include <algorithm>
#include <cmath>
#include <map>
#include <vector>

#include "psi4/libmints/molecule.h"

#include "../fci/fci.h"
#include "../fci_mo.h"
#include "../helpers.h"
#include "../semi_canonicalize.h"
#include "../mini-boost/boost/format.hpp"
#include "../mp2_nos.h"
#include "mrdsrg.h"

namespace psi {
namespace forte {

MRDSRG::MRDSRG(Reference reference, SharedWavefunction ref_wfn, Options& options,
               std::shared_ptr<ForteIntegrals> ints, std::shared_ptr<MOSpaceInfo> mo_space_info)
    : Wavefunction(options), reference_(reference), ints_(ints), mo_space_info_(mo_space_info),
      BTF_(new BlockedTensorFactory(options)), tensor_type_(CoreTensor) {
    shallow_copy(ref_wfn);
    reference_wavefunction_ = ref_wfn;

    print_method_banner({"Multireference Driven Similarity Renormalization Group", "Chenyang Li"});
    read_options();
    print_options();
    startup();
}

MRDSRG::~MRDSRG() { cleanup(); }

void MRDSRG::cleanup() { dsrg_time_.print_comm_time(); }

void MRDSRG::read_options() {

    print_ = options_.get_int("PRINT");

    s_ = options_.get_double("DSRG_S");
    if (s_ < 0) {
        outfile->Printf("\n  S parameter for DSRG must >= 0!");
        throw PSIEXCEPTION("S parameter for DSRG must >= 0!");
    }
    taylor_threshold_ = options_.get_int("TAYLOR_THRESHOLD");
    if (taylor_threshold_ <= 0) {
        outfile->Printf("\n  Threshold for Taylor expansion must be an integer "
                        "greater than 0!");
        throw PSIEXCEPTION("Threshold for Taylor expansion must be an integer "
                           "greater than 0!");
    }

    source_ = options_.get_str("SOURCE");
    if (source_ != "STANDARD" && source_ != "LABS" && source_ != "DYSON") {
        outfile->Printf("\n  Warning: SOURCE option \"%s\" is not implemented "
                        "in MRDSRG. Changed to STANDARD.",
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

    ntamp_ = options_.get_int("NTAMP");
    intruder_tamp_ = options_.get_double("INTRUDER_TAMP");
}

void MRDSRG::startup() {
    // frozen-core energy
    frozen_core_energy_ = ints_->frozen_core_energy();

    // reference energy
    Eref_ = reference_.get_Eref();

    // density fitted ERI?
    eri_df_ = false;
    std::string int_type = options_.get_str("INT_TYPE");
    if (int_type == "CHOLESKY" || int_type == "DF" || int_type == "DISKDF") {
        eri_df_ = true;
    }

    // orbital spaces
    BlockedTensor::reset_mo_spaces();
    acore_mos_ = mo_space_info_->get_corr_abs_mo("RESTRICTED_DOCC");
    bcore_mos_ = mo_space_info_->get_corr_abs_mo("RESTRICTED_DOCC");
    aactv_mos_ = mo_space_info_->get_corr_abs_mo("ACTIVE");
    bactv_mos_ = mo_space_info_->get_corr_abs_mo("ACTIVE");
    avirt_mos_ = mo_space_info_->get_corr_abs_mo("RESTRICTED_UOCC");
    bvirt_mos_ = mo_space_info_->get_corr_abs_mo("RESTRICTED_UOCC");

    // define space labels
    acore_label_ = "c";
    aactv_label_ = "a";
    avirt_label_ = "v";
    bcore_label_ = "C";
    bactv_label_ = "A";
    bvirt_label_ = "V";
    BTF_->add_mo_space(acore_label_, "mn", acore_mos_, AlphaSpin);
    BTF_->add_mo_space(bcore_label_, "MN", bcore_mos_, BetaSpin);
    BTF_->add_mo_space(aactv_label_, "uvwxyz", aactv_mos_, AlphaSpin);
    BTF_->add_mo_space(bactv_label_, "UVWXYZ", bactv_mos_, BetaSpin);
    BTF_->add_mo_space(avirt_label_, "ef", avirt_mos_, AlphaSpin);
    BTF_->add_mo_space(bvirt_label_, "EF", bvirt_mos_, BetaSpin);

    // map space labels to mo spaces
    label_to_spacemo_[acore_label_[0]] = acore_mos_;
    label_to_spacemo_[bcore_label_[0]] = bcore_mos_;
    label_to_spacemo_[aactv_label_[0]] = aactv_mos_;
    label_to_spacemo_[bactv_label_[0]] = bactv_mos_;
    label_to_spacemo_[avirt_label_[0]] = avirt_mos_;
    label_to_spacemo_[bvirt_label_[0]] = bvirt_mos_;

    // define composite spaces
    BTF_->add_composite_mo_space("h", "ijkl", {acore_label_, aactv_label_});
    BTF_->add_composite_mo_space("H", "IJKL", {bcore_label_, bactv_label_});
    BTF_->add_composite_mo_space("p", "abcd", {aactv_label_, avirt_label_});
    BTF_->add_composite_mo_space("P", "ABCD", {bactv_label_, bvirt_label_});
    BTF_->add_composite_mo_space("g", "pqrsto", {acore_label_, aactv_label_, avirt_label_});
    BTF_->add_composite_mo_space("G", "PQRSTO", {bcore_label_, bactv_label_, bvirt_label_});

    // prepare integrals
    H_ = BTF_->build(tensor_type_, "H", spin_cases({"gg"}));
    V_ = BTF_->build(tensor_type_, "V", spin_cases({"gggg"}));
    build_ints();

    // prepare density matrix and cumulants
    // TODO: future code will store only active Gamma1 and Eta1
    Eta1_ = BTF_->build(tensor_type_, "Eta1", spin_cases({"pp"}));
    Gamma1_ = BTF_->build(tensor_type_, "Gamma1", spin_cases({"hh"}));
    Lambda2_ = BTF_->build(tensor_type_, "Lambda2", spin_cases({"aaaa"}));
    if (options_.get_str("THREEPDC") != "ZERO") {
        Lambda3_ = BTF_->build(tensor_type_, "Lambda3", spin_cases({"aaaaaa"}));
    }
    build_density();

    // build Fock matrix
    F_ = BTF_->build(tensor_type_, "Fock", spin_cases({"gg"}));
    build_fock(H_, V_);

    // auto adjusted s_
    s_ = make_s_smart();

    // test semi-canonical
    print_h2("Checking Orbitals");
    H0th_ = BTF_->build(tensor_type_, "Zeroth-order H", diag_one_labels());
    semi_canonical_ = check_semicanonical();

    if (!semi_canonical_) {
        outfile->Printf("\n    MR-DSRG will be computed in an arbitrary basis. "
                        "Orbital invariant formalism is employed.");
        outfile->Printf("\n    We recommend using semi-canonical for all "
                        "denominator-based source operator.");
        if (options_.get_str("RELAX_REF") != "NONE") {
            outfile->Printf("\n\n    Currently, only RELAX_REF = NONE is "
                            "available for orbital invariant formalism.");
            throw PSIEXCEPTION("Orbital invariant formalism is not implemented "
                               "for the RELAX_REF option.");
        } else {
            U_ = ambit::BlockedTensor::build(tensor_type_, "U", spin_cases({"gg"}));
            std::vector<std::vector<double>> eigens = diagonalize_Fock_diagblocks(U_);
            Fa_ = eigens[0];
            Fb_ = eigens[1];
        }
    } else {
        outfile->Printf("\n    Orbitals are semi-canonicalized.\n");
    }

    // initialize timer for commutator
    dsrg_time_ = DSRG_TIME();
}

void MRDSRG::build_ints() {
    // prepare one-electron integrals
    H_.iterate([&](const std::vector<size_t>& i, const std::vector<SpinType>& spin, double& value) {
        if (spin[0] == AlphaSpin)
            value = ints_->oei_a(i[0], i[1]);
        else
            value = ints_->oei_b(i[0], i[1]);
    });

    // prepare two-electron integrals
    if (eri_df_) {
        V_["pqrs"] = B_["gpr"] * B_["gqs"];
        V_["pqrs"] -= B_["gps"] * B_["gqr"];

        V_["pQrS"] = B_["gpr"] * B_["gQS"];

        V_["PQRS"] = B_["gPR"] * B_["gQS"];
        V_["PQRS"] -= B_["gPS"] * B_["gQR"];
    } else {
        V_.iterate(
            [&](const std::vector<size_t>& i, const std::vector<SpinType>& spin, double& value) {
                if ((spin[0] == AlphaSpin) and (spin[1] == AlphaSpin))
                    value = ints_->aptei_aa(i[0], i[1], i[2], i[3]);
                if ((spin[0] == AlphaSpin) and (spin[1] == BetaSpin))
                    value = ints_->aptei_ab(i[0], i[1], i[2], i[3]);
                if ((spin[0] == BetaSpin) and (spin[1] == BetaSpin))
                    value = ints_->aptei_bb(i[0], i[1], i[2], i[3]);
            });
    }
}

void MRDSRG::build_density() {
    // prepare density matrices
    (Gamma1_.block("cc")).iterate([&](const std::vector<size_t>& i, double& value) {
        value = i[0] == i[1] ? 1.0 : 0.0;
    });
    (Gamma1_.block("CC")).iterate([&](const std::vector<size_t>& i, double& value) {
        value = i[0] == i[1] ? 1.0 : 0.0;
    });
    (Eta1_.block("aa")).iterate([&](const std::vector<size_t>& i, double& value) {
        value = i[0] == i[1] ? 1.0 : 0.0;
    });
    (Eta1_.block("AA")).iterate([&](const std::vector<size_t>& i, double& value) {
        value = i[0] == i[1] ? 1.0 : 0.0;
    });
    (Eta1_.block("vv")).iterate([&](const std::vector<size_t>& i, double& value) {
        value = i[0] == i[1] ? 1.0 : 0.0;
    });
    (Eta1_.block("VV")).iterate([&](const std::vector<size_t>& i, double& value) {
        value = i[0] == i[1] ? 1.0 : 0.0;
    });
    //    // symmetrize beta spin
    //    outfile->Printf("\n  Warning: I am forcing density Db = Da to avoid spin "
    //                    "symmetry breaking.");
    //    outfile->Printf("\n  If this is not desired, go to mrdsrg.cc "
    //                    "build_density() around line 190.");
    Gamma1_.block("aa")("pq") = reference_.L1a()("pq");
    Gamma1_.block("AA")("pq") = reference_.L1b()("pq");
    Eta1_.block("aa")("pq") -= reference_.L1a()("pq");
    Eta1_.block("AA")("pq") -= reference_.L1b()("pq");

    //    ambit::Tensor Diff =
    //    ambit::Tensor::build(tensor_type_,"Diff",reference_.L1a().dims());
    //    Diff.data() = reference_.L1a().data();
    //    Diff("pq") -= reference_.L1b()("pq");
    //    outfile->Printf("\n  L1a diff Here !!!!");
    //    Diff.citerate([&](const std::vector<size_t>& i,const double& value){
    //        if(value != 0.0){
    //            outfile->Printf("\n  [%zu][%zu] = %20.15f",i[0],i[1],value);
    //        }
    //    });

    // prepare two-body density cumulants
    ambit::Tensor Lambda2_aa = Lambda2_.block("aaaa");
    ambit::Tensor Lambda2_aA = Lambda2_.block("aAaA");
    ambit::Tensor Lambda2_AA = Lambda2_.block("AAAA");
    Lambda2_aa("pqrs") = reference_.L2aa()("pqrs");
    Lambda2_aA("pqrs") = reference_.L2ab()("pqrs");
    Lambda2_AA("pqrs") = reference_.L2bb()("pqrs");

    // prepare three-body density cumulants
    if (options_.get_str("THREEPDC") != "ZERO") {
        ambit::Tensor Lambda3_aaa = Lambda3_.block("aaaaaa");
        ambit::Tensor Lambda3_aaA = Lambda3_.block("aaAaaA");
        ambit::Tensor Lambda3_aAA = Lambda3_.block("aAAaAA");
        ambit::Tensor Lambda3_AAA = Lambda3_.block("AAAAAA");
        Lambda3_aaa("pqrstu") = reference_.L3aaa()("pqrstu");
        Lambda3_aaA("pqrstu") = reference_.L3aab()("pqrstu");
        Lambda3_aAA("pqrstu") = reference_.L3abb()("pqrstu");
        Lambda3_AAA("pqrstu") = reference_.L3bbb()("pqrstu");
    }

    // check cumulants
    print_cumulant_summary();
}

void MRDSRG::build_fock(BlockedTensor& H, BlockedTensor& V) {
    // build Fock matrix
    F_["pq"] = H["pq"];
    F_["pq"] += V["pjqi"] * Gamma1_["ij"];
    F_["pq"] += V["pJqI"] * Gamma1_["IJ"];
    F_["PQ"] = H["PQ"];
    F_["PQ"] += V["jPiQ"] * Gamma1_["ij"];
    F_["PQ"] += V["PJQI"] * Gamma1_["IJ"];

    // obtain diagonal elements of Fock matrix
    size_t ncmo_ = mo_space_info_->size("CORRELATED");
    Fa_ = std::vector<double>(ncmo_);
    Fb_ = std::vector<double>(ncmo_);
    F_.iterate([&](const std::vector<size_t>& i, const std::vector<SpinType>& spin, double& value) {
        if (spin[0] == AlphaSpin and (i[0] == i[1])) {
            Fa_[i[0]] = value;
        }
        if (spin[0] == BetaSpin and (i[0] == i[1])) {
            Fb_[i[0]] = value;
        }
    });
}

void MRDSRG::print_options() {
    // fill in information
    std::vector<std::pair<std::string, int>> calculation_info{
        {"ntamp", ntamp_},
        {"diis_min_vecs", options_.get_int("DIIS_MIN_VECS")},
        {"diis_max_vecs", options_.get_int("DIIS_MAX_VECS")}};

    std::vector<std::pair<std::string, double>> calculation_info_double{
        {"flow parameter", s_},
        {"taylor expansion threshold", pow(10.0, -double(taylor_threshold_))},
        {"intruder_tamp", intruder_tamp_}};

    std::vector<std::pair<std::string, std::string>> calculation_info_string{
        {"corr_level", options_.get_str("CORR_LEVEL")},
        {"int_type", options_.get_str("INT_TYPE")},
        {"source operator", source_},
        {"smart_dsrg_s", options_.get_str("SMART_DSRG_S")},
        {"reference relaxation", options_.get_str("RELAX_REF")},
        {"dsrg transformation type", options_.get_str("DSRG_TRANS_TYPE")},
        {"core virtual source type", options_.get_str("CCVV_SOURCE")}};

    // print some information
    print_h2("Calculation Information");
    for (auto& str_dim : calculation_info) {
        outfile->Printf("\n    %-35s %15d", str_dim.first.c_str(), str_dim.second);
    }
    for (auto& str_dim : calculation_info_double) {
        outfile->Printf("\n    %-35s %15.3e", str_dim.first.c_str(), str_dim.second);
    }
    for (auto& str_dim : calculation_info_string) {
        outfile->Printf("\n    %-35s %15s", str_dim.first.c_str(), str_dim.second.c_str());
    }
    outfile->Printf("\n");
}

double MRDSRG::compute_energy() {
    // guess amplitudes when necessary
    bool initialize_T = true;
    std::string corrlv_string = options_.get_str("CORR_LEVEL");
    if (corrlv_string == "LSRG2" || corrlv_string == "SRG_PT2") {
        initialize_T = false;
    }

    if (initialize_T) {
        // build initial amplitudes
        print_h2("Build Initial Amplitude from DSRG-MRPT2");
        T1_ = BTF_->build(tensor_type_, "T1 Amplitudes", spin_cases({"hp"}));
        T2_ = BTF_->build(tensor_type_, "T2 Amplitudes", spin_cases({"hhpp"}));
        guess_t(V_, T2_, F_, T1_);

        // check initial amplitudes
        analyze_amplitudes("First-Order", T1_, T2_);
    }

    // get reference energy
    double Etotal = Eref_;

    // compute energy
    switch (corrlevelmap[corrlv_string]) {
    case CORR_LV::LDSRG2: {
        Etotal += compute_energy_ldsrg2();
        break;
    }
    case CORR_LV::LDSRG2_QC: {
        Etotal += compute_energy_ldsrg2_qc();
        break;
    }
    case CORR_LV::LDSRG2_P3: {
        break;
    }
    case CORR_LV::QDSRG2: {
        break;
    }
    case CORR_LV::QDSRG2_P3: {
        break;
    }
    case CORR_LV::LSRG2: {
        Etotal += compute_energy_lsrg2();
        break;
    }
    case CORR_LV::SRG_PT2: {
        Etotal += compute_energy_srgpt2();
        break;
    }
    case CORR_LV::PT3: {
        Etotal += compute_energy_pt3();
        break;
    }
    default: { Etotal += compute_energy_pt2(); }
    }

    Process::environment.globals["CURRENT ENERGY"] = Etotal;
    return Etotal;
}

double MRDSRG::compute_energy_relaxed() {
    //    // setup for FCISolver
    //    Dimension active_dim = mo_space_info_->get_dimension("ACTIVE");
    //    int charge = Process::environment.molecule()->molecular_charge();
    //    if (options_["CHARGE"].has_changed()) {
    //        charge = options_.get_int("CHARGE");
    //    }
    //    auto nelec = 0;
    //    int natom = Process::environment.molecule()->natom();
    //    for (int i = 0; i < natom; ++i) {
    //        nelec += Process::environment.molecule()->fZ(i);
    //    }
    //    nelec -= charge;
    //    int multi = Process::environment.molecule()->multiplicity();
    //    if (options_["MULTIPLICITY"].has_changed()) {
    //        multi = options_.get_int("MULTIPLICITY");
    //    }
    //    int twice_ms = (multi + 1) % 2;
    //    if (options_["MS"].has_changed()) {
    //        twice_ms = std::round(2.0 * options_.get_double("MS"));
    //    }
    //    auto nelec_actv = nelec - 2 * mo_space_info_->size("FROZEN_DOCC") - 2 * acore_mos_.size();
    //    auto na = (nelec_actv + twice_ms) / 2;
    //    auto nb = nelec_actv - na;

    // reference relaxation
    double Edsrg = 0.0, Erelax = 0.0;
    std::string relax_algorithm = options_.get_str("RELAX_REF");
    std::string cas_type = options_.get_str("CAS_TYPE");

    if (relax_algorithm == "ONCE") {
        // compute energy with fixed ref.
        Edsrg = compute_energy();

        // transfer integrals
        transfer_integrals();

        if (cas_type == "CAS") {
            FCI_MO fci_mo(reference_wavefunction_, options_, ints_, mo_space_info_);
            fci_mo.set_localize_actv(false);
            Erelax = fci_mo.compute_energy();
        } else {
            FCI fci(reference_wavefunction_, options_, ints_, mo_space_info_);
            Erelax = fci.compute_energy();
        }

        // printing
        print_h2("MRDSRG Energy Summary");
        outfile->Printf("\n    %-30s = %22.15f", "MRDSRG Total Energy (fixed)", Edsrg);
        outfile->Printf("\n    %-30s = %22.15f", "MRDSRG Total Energy (relaxed)", Erelax);
        outfile->Printf("\n");
    } else if (relax_algorithm == "ITERATE" || relax_algorithm == "TWICE") {
        // iteration variables
        int cycle = 0, maxiter = options_.get_int("MAXITER_RELAX_REF");
        double e_conv = options_.get_double("E_CONVERGENCE");
        std::vector<double> Edsrg_vec, Erelax_vec;
        std::vector<double> Edelta_dsrg_vec, Edelta_relax_vec;
        bool converged = false, failed = false;
        SemiCanonical semiorb(reference_wavefunction_, ints_, mo_space_info_, true);

        // start iteration
        do {
            // print
            std::string relax_title = "MR-DSRG Ref. Relaxation Iter. " + std::to_string(cycle);
            print_h2(relax_title);

            // compute dsrg energy
            double Etemp = Edsrg;
            Edsrg = compute_energy();
            Edsrg_vec.push_back(Edsrg);
            double Edelta_dsrg = Edsrg - Etemp;
            Edelta_dsrg_vec.push_back(Edelta_dsrg);

            // transfer integrals (O1, Hbar2)
            transfer_integrals();

            // diagonalize the Hamiltonian
            if (cas_type == "CAS") {
                FCI_MO fci_mo(reference_wavefunction_, options_, ints_, mo_space_info_);
                fci_mo.set_localize_actv(false);
                Erelax = fci_mo.compute_energy();

                // obtain new reference
                reference_ = fci_mo.reference();
            } else {
                FCI fci(reference_wavefunction_, options_, ints_, mo_space_info_);
                if (options_.get_str("THREEPDC") == "ZERO") {
                    fci.set_max_rdm_level(2);
                } else {
                    fci.set_max_rdm_level(3);
                }
                Erelax = fci.compute_energy();

                // obtain new reference
                reference_ = fci.reference();

                //                FCISolver fcisolver(active_dim, acore_mos_, aactv_mos_, na, nb,
                //                multi,
                //                                    options_.get_int("ROOT_SYM"), ints_,
                //                                    mo_space_info_, options_);
                //                Etemp = Erelax;
                //                fcisolver.set_max_rdm_level(3);
                //                fcisolver.set_nroot(options_.get_int("NROOT"));
                //                fcisolver.set_root(options_.get_int("ROOT"));
                //                fcisolver.set_fci_iterations(options_.get_int("FCI_MAXITER"));
                //                fcisolver.set_collapse_per_root(options_.get_int("DL_COLLAPSE_PER_ROOT"));
                //                fcisolver.set_subspace_per_root(options_.get_int("DL_SUBSPACE_PER_ROOT"));
                //                Erelax = fcisolver.compute_energy();
            }

            Erelax_vec.push_back(Erelax);
            double Edelta_relax = Erelax - Etemp;
            Edelta_relax_vec.push_back(Edelta_relax);

            //            // obtain new reference
            //            reference_ = fcisolver.reference();

            // refill densities
            build_density();

            // build the new Fock matrix
            build_fock(H_, V_);

            // check orbitals
            print_h2("Orbital Check");
            bool semi = check_semicanonical();

            // semicanonicalize orbitals
            if (options_.get_bool("SEMI_CANONICAL") && !semi) {
                print_h2("Semicanonicalize Orbitals");

                // reset forte integrals
                reset_ints(H_, V_);

                // use semicanonicalize class
                semiorb.semicanonicalize(reference_);

                // refill transformed integrals
                build_ints();

                // refill densities
                build_density();

                // rebuild Fock matrix
                build_fock(H_, V_);
            }

            // test convergence
            if (std::fabs(Edelta_dsrg) < e_conv && std::fabs(Edelta_relax) < e_conv) {
                converged = true;
            }
            if (cycle > maxiter) {
                outfile->Printf("\n\n    The reference relaxation does not "
                                "converge in %d iterations! Quitting.\n",
                                maxiter);
                converged = true;
                failed = true;
            }
            ++cycle;

            // terminate peacefully if relaxed twice
            if (relax_algorithm == "TWICE" && cycle == 2) {
                converged = true;
                failed = false;
            }
        } while (!converged);

        print_h2("MRDSRG Reference Relaxation Summary");
        std::string indent(4, ' ');
        std::string dash(71, '-');
        std::string title;
        title += indent + str(boost::format("%5c  %=31s  %=31s\n") % ' ' % "Fixed Ref. (a.u.)" %
                              "Relaxed Ref. (a.u.)");
        title += indent + std::string(7, ' ') + std::string(31, '-') + "  " + std::string(31, '-') +
                 "\n";
        title += indent + str(boost::format("%5s  %=20s %=10s  %=20s %=10s\n") % "Iter." %
                              "Total Energy" % "Delta" % "Total Energy" % "Delta");
        title += indent + dash;
        outfile->Printf("\n%s", title.c_str());
        for (int n = 0; n != cycle; ++n) {
            outfile->Printf("\n    %5d  %20.12f %10.3e  %20.12f %10.3e", n, Edsrg_vec[n],
                            Edelta_dsrg_vec[n], Erelax_vec[n], Edelta_relax_vec[n]);
        }
        outfile->Printf("\n    %s", dash.c_str());
        outfile->Printf("\n    %-30s = %23.15f", "MRDSRG Total Energy", Edsrg);
        outfile->Printf("\n    %-30s = %23.15f", "MRDSRG Total Energy (relaxed)", Erelax);
        outfile->Printf("\n");

        if (failed) {
            throw PSIEXCEPTION("Reference relaxation process does not converge.");
        }

        // set energies to psi4 environment
        Process::environment.globals["UNRELAXED ENERGY"] = Edsrg_vec[0];
        Process::environment.globals["PARTIALLY RELAXED ENERGY"] = Erelax_vec[0];
        if (cycle > 1) {
            Process::environment.globals["RELAXED ENERGY"] = Edsrg_vec[1];
        }
    }

    Process::environment.globals["CURRENT ENERGY"] = Erelax;
    return Erelax;
}

double MRDSRG::compute_energy_sa() {
    //    // compute MR-DSRG energy
    //    compute_energy();

    //    // transfer integrals
    //    transfer_integrals();

    //    // prepare FCI integrals
    //    std::shared_ptr<FCIIntegrals> fci_ints =
    //    std::make_shared<FCIIntegrals>(ints_, aactv_mos_, acore_mos_);
    //    fci_ints->set_active_integrals(Hbar2_.block("aaaa"),
    //    Hbar2_.block("aAaA"), Hbar2_.block("AAAA"));
    //    fci_ints->compute_restricted_one_body_operator();

    //    // get character table
    //    CharacterTable ct =
    //    Process::environment.molecule()->point_group()->char_table();
    //    std::vector<std::string> irrep_symbol;
    //    for(int h = 0; h < this->nirrep(); ++h){
    //        irrep_symbol.push_back(std::string(ct.gamma(h).symbol()));
    //    }

    //    // multiplicity table
    //    std::vector<std::string>
    //    multi_label{"Singlet","Doublet","Triplet","Quartet","Quintet","Sextet","Septet","Octet",
    //                                  "Nonet","Decaet","11-et","12-et","13-et","14-et","15-et","16-et","17-et","18-et",
    //                                  "19-et","20-et","21-et","22-et","23-et","24-et"};

    //    // size of 1rdm and 2rdm
    //    size_t na1 = mo_space_info_->size("ACTIVE");
    //    size_t na2 = na1 * na1;
    //    size_t na3 = na1 * na2;

    //    // get effective one-electron integral (DSRG transformed)
    //    BlockedTensor oei =
    //    BTF_->build(tensor_type_,"temp1",spin_cases({"aa"}));
    //    oei.block("aa").data() = fci_ints->oei_a_vector();
    //    oei.block("AA").data() = fci_ints->oei_b_vector();

    //    // get nuclear repulsion energy
    //    std::shared_ptr<Molecule> molecule = Process::environment.molecule();
    //    double Enuc = molecule->nuclear_repulsion_energy();

    //    // loop over entries of AVG_STATE
    //    print_h2("Diagonalize Effective Hamiltonian");
    //    outfile->Printf("\n");
    //    int nentry = eigens_.size();
    //    std::vector<std::vector<double>> Edsrg_sa (nentry, std::vector<double>
    //    ());

    //    for(int n = 0; n < nentry; ++n){
    //        int irrep = options_["AVG_STATE"][n][0].to_integer();
    //        int multi = options_["AVG_STATE"][n][1].to_integer();
    //        int nstates = options_["AVG_STATE"][n][2].to_integer();

    //        // diagonalize which the second-order effective Hamiltonian
    //        // FULL: CASCI using determinants
    //        // AVG_STATES: H_AB = <A|H|B> where A and B are SA-CAS states
    //        if(options_.get_str("DSRG_SA_HEFF") == "FULL") {

    //            outfile->Printf("    Use string FCI code.");

    //            // prepare FCISolver
    //            int charge =
    //            Process::environment.molecule()->molecular_charge();
    //            if(options_["CHARGE"].has_changed()){
    //                charge = options_.get_int("CHARGE");
    //            }
    //            auto nelec = 0;
    //            int natom = Process::environment.molecule()->natom();
    //            for(int i = 0; i < natom; ++i){
    //                nelec += Process::environment.molecule()->fZ(i);
    //            }
    //            nelec -= charge;
    //            int ms = (multi + 1) % 2;
    //            auto nelec_actv = nelec - 2 *
    //            mo_space_info_->size("FROZEN_DOCC") - 2 * acore_mos_.size();
    //            auto na = (nelec_actv + ms) / 2;
    //            auto nb =  nelec_actv - na;

    //            Dimension active_dim =
    //            mo_space_info_->get_dimension("ACTIVE");
    //            int ntrial_per_root = options_.get_int("NTRIAL_PER_ROOT");

    //            FCISolver
    //            fcisolver(active_dim,acore_mos_,aactv_mos_,na,nb,multi,irrep,
    //                                ints_,mo_space_info_,ntrial_per_root,print_,options_);
    //            fcisolver.set_max_rdm_level(1);
    //            fcisolver.set_nroot(nstates);
    //            fcisolver.set_root(nstates - 1);
    //            fcisolver.set_fci_iterations(options_.get_int("FCI_MAXITER"));
    //            fcisolver.set_collapse_per_root(options_.get_int("DL_COLLAPSE_PER_ROOT"));
    //            fcisolver.set_subspace_per_root(options_.get_int("DL_SUBSPACE_PER_ROOT"));

    //            // compute energy and fill in results
    //            fcisolver.compute_energy();
    //            SharedVector Ems = fcisolver.eigen_vals();
    //            for(int i = 0; i < nstates; ++i){
    //                Edsrg_sa[n].push_back(Ems->get(i) + Enuc);
    //            }

    //        } else {

    //            int dim = (eigens_[n][0].first)->dim();
    //            SharedMatrix evecs (new Matrix("evecs",dim,dim));
    //            for(int i = 0; i < eigens_[n].size(); ++i){
    //                evecs->set_column(0,i,(eigens_[n][i]).first);
    //            }

    //            SharedMatrix Heff (new Matrix("Heff " + multi_label[multi - 1]
    //                               + " " + irrep_symbol[irrep], nstates,
    //                               nstates));
    //            for(int A = 0; A < nstates; ++A){
    //                for(int B = A; B < nstates; ++B){

    //                    // compute rdms
    //                    CI_RDMS ci_rdms
    //                    (options_,fci_ints,p_space_,evecs,A,B);
    //                    ci_rdms.set_symmetry(irrep);

    //                    std::vector<double> opdm_a (na2, 0.0);
    //                    std::vector<double> opdm_b (na2, 0.0);
    //                    ci_rdms.compute_1rdm(opdm_a,opdm_b);

    //                    std::vector<double> tpdm_aa (na3, 0.0);
    //                    std::vector<double> tpdm_ab (na3, 0.0);
    //                    std::vector<double> tpdm_bb (na3, 0.0);
    //                    ci_rdms.compute_2rdm(tpdm_aa,tpdm_ab,tpdm_bb);

    //                    // put rdms in tensor format
    //                    BlockedTensor D1 =
    //                    BTF_->build(tensor_type_,"D1",spin_cases({"aa"}),true);
    //                    D1.block("aa").data() = opdm_a;
    //                    D1.block("AA").data() = opdm_b;

    //                    BlockedTensor D2 =
    //                    BTF_->build(tensor_type_,"D2",spin_cases({"aaaa"}),true);
    //                    D2.block("aaaa").data() = tpdm_aa;
    //                    D2.block("aAaA").data() = tpdm_ab;
    //                    D2.block("AAAA").data() = tpdm_bb;

    //                    double H_AB = 0.0;
    //                    H_AB += oei["uv"] * D1["uv"];
    //                    H_AB += oei["UV"] * D1["UV"];
    //                    H_AB += 0.25 * Hbar2_["uvxy"] * D2["xyuv"];
    //                    H_AB += 0.25 * Hbar2_["UVXY"] * D2["XYUV"];
    //                    H_AB += Hbar2_["uVxY"] * D2["xYuV"];

    //                    if(A == B){
    //                        H_AB += ints_->frozen_core_energy() +
    //                        fci_ints->scalar_energy() + Enuc;
    //                        Heff->set(A,B,H_AB);
    //                    } else {
    //                        Heff->set(A,B,H_AB);
    //                        Heff->set(B,A,H_AB);
    //                    }

    //                }
    //            } // end forming effective Hamiltonian

    //            Heff->print();

    //            SharedMatrix U (new Matrix("U of Heff", nstates, nstates));
    //            SharedVector Ems (new Vector("MS Energies", nstates));
    //            Heff->diagonalize(U, Ems);

    //            // fill in Edsrg_sa
    //            for(int i = 0; i < nstates; ++i){
    //                Edsrg_sa[n].push_back(Ems->get(i));
    //            }
    //        } // end if DSRG_AVG_DIAG

    //    } // end looping averaged states

    //    // energy summuary
    //    print_h2("State-Average MR-DSRG Energy Summary");

    //    outfile->Printf("\n    Multi.  Irrep.  No.    MR-DSRG Energy");
    //    std::string dash(41, '-');
    //    outfile->Printf("\n    %s", dash.c_str());

    //    for(int n = 0; n < nentry; ++n){
    //        int irrep = options_["AVG_STATE"][n][0].to_integer();
    //        int multi = options_["AVG_STATE"][n][1].to_integer();
    //        int nstates = options_["AVG_STATE"][n][2].to_integer();

    //        for(int i = 0; i < nstates; ++i){
    //            outfile->Printf("\n     %3d     %3s    %2d   %20.12f",
    //                            multi, irrep_symbol[irrep].c_str(), i,
    //                            Edsrg_sa[n][i]);
    //        }
    //        outfile->Printf("\n    %s", dash.c_str());
    //    }

    //    Process::environment.globals["CURRENT ENERGY"] = Edsrg_sa[0][0];
    //    return Edsrg_sa[0][0];

    // iteration variables
    double Edsrg_sa = 0.0, Erelax_sa = 0.0;
    int cycle = 0, maxiter = options_.get_int("MAXITER_RELAX_REF");
    double e_conv = options_.get_double("E_CONVERGENCE");
    std::vector<double> Edsrg_sa_vec, Erelax_sa_vec;
    std::vector<double> Edelta_dsrg_sa_vec, Edelta_relax_sa_vec;
    bool converged = false, failed = false;
    int nentry = eigens_.size();
    std::vector<std::vector<std::vector<double>>> Edsrg_vec;
    SemiCanonical semiorb(reference_wavefunction_, ints_, mo_space_info_, true);

    // start iteration
    do {
        // print
        outfile->Printf("\n  ==> SA-MR-DSRG CI Iter. %d <==", cycle);

        // compute dsrg energy
        double Etemp = Edsrg_sa;
        Edsrg_sa = compute_energy();
        Edsrg_sa_vec.push_back(Edsrg_sa);
        double Edelta_dsrg = Edsrg_sa - Etemp;
        Edelta_dsrg_sa_vec.push_back(Edelta_dsrg);

        // transfer integrals (O1, Hbar2)
        transfer_integrals();

        // diagonalize the Hamiltonian
        auto fci_mo =
            std::make_shared<FCI_MO>(reference_wavefunction_, options_, ints_, mo_space_info_);
        Etemp = Erelax_sa;
        fci_mo->set_localize_actv(false);
        Erelax_sa = fci_mo->compute_energy();
        Erelax_sa_vec.push_back(Erelax_sa);
        double Edelta_relax = Erelax_sa - Etemp;
        Edelta_relax_sa_vec.push_back(Edelta_relax);

        // copy energy
        std::vector<std::vector<double>> Evec(nentry, std::vector<double>());
        for (int n = 0; n < nentry; ++n) {
            int nstates = options_["AVG_STATE"][n][2].to_integer();

            for (int i = 0; i < nstates; ++i) {
                Evec[n].push_back(fci_mo->eigens()[n][i].second);
            }

            Edsrg_vec.push_back(Evec);
        }

        // obtain new reference
        reference_ = fci_mo->reference();

        // refill densities
        build_density();

        // build the new Fock matrix
        build_fock(H_, V_);

        // check orbitals
        print_h2("Orbital Check");
        bool semi = check_semicanonical();

        // semicanonicalize orbitals
        if (options_.get_bool("SEMI_CANONICAL") && !semi) {
            print_h2("Semicanonicalize Orbitals");

            // reset forte integrals
            reset_ints(H_, V_);

            // use semicanonicalize class
            semiorb.semicanonicalize(reference_);

            // refill transformed integrals
            build_ints();

            // refill densities
            build_density();

            // rebuild Fock matrix
            build_fock(H_, V_);
        }

        // test convergence
        if (std::fabs(Edelta_dsrg) < e_conv && std::fabs(Edelta_relax) < e_conv) {
            converged = true;
        }
        if (cycle > maxiter) {
            outfile->Printf("\n\n    The reference relaxation does not "
                            "converge in %d iterations! Quitting.\n",
                            maxiter);
            converged = true;
            failed = true;
        }
        ++cycle;
    } while (!converged);

    print_h2("State-Average MR-DSRG Energy Summary");

    outfile->Printf("\n    state-averaged energy summary (not useful)");
    std::string indent(4, ' ');
    std::string dash(71, '-');
    std::string title;
    title += indent + str(boost::format("%5c  %=31s  %=31s\n") % ' ' % "Fixed Ref. (a.u.)" %
                          "Relaxed Ref. (a.u.)");
    title +=
        indent + std::string(7, ' ') + std::string(31, '-') + "  " + std::string(31, '-') + "\n";
    title += indent + str(boost::format("%5s  %=20s %=10s  %=20s %=10s\n") % "Iter." %
                          "Total Energy" % "Delta" % "Total Energy" % "Delta");
    title += indent + dash;
    outfile->Printf("\n%s", title.c_str());
    for (int n = 0; n != cycle; ++n) {
        outfile->Printf("\n    %5d  %20.12f %10.3e  %20.12f %10.3e", n, Edsrg_sa_vec[n],
                        Edelta_dsrg_sa_vec[n], Erelax_sa_vec[n], Edelta_relax_sa_vec[n]);
    }
    outfile->Printf("\n    %s", dash.c_str());

    // get character table
    CharacterTable ct = Process::environment.molecule()->point_group()->char_table();
    std::vector<std::string> irrep_symbol;
    for (int h = 0; h < this->nirrep(); ++h) {
        irrep_symbol.push_back(std::string(ct.gamma(h).symbol()));
    }

    // frist step results
    print_h2("One-Step Results");
    outfile->Printf("\n    Multi.  Irrep.  No.    MR-DSRG Energy");
    std::string dash1(41, '-');
    outfile->Printf("\n    %s", dash1.c_str());
    for (int n = 0; n < nentry; ++n) {
        int irrep = options_["AVG_STATE"][n][0].to_integer();
        int multi = options_["AVG_STATE"][n][1].to_integer();
        int nstates = options_["AVG_STATE"][n][2].to_integer();

        for (int i = 0; i < nstates; ++i) {
            outfile->Printf("\n     %3d     %3s    %2d   %20.12f", multi,
                            irrep_symbol[irrep].c_str(), i, Edsrg_vec[0][n][i]);
        }
        outfile->Printf("\n    %s", dash1.c_str());
    }

    // final step results
    print_h2("Final-Step Results");
    outfile->Printf("\n    Multi.  Irrep.  No.    MR-DSRG Energy");
    outfile->Printf("\n    %s", dash1.c_str());
    for (int n = 0; n < nentry; ++n) {
        int irrep = options_["AVG_STATE"][n][0].to_integer();
        int multi = options_["AVG_STATE"][n][1].to_integer();
        int nstates = options_["AVG_STATE"][n][2].to_integer();

        for (int i = 0; i < nstates; ++i) {
            outfile->Printf("\n     %3d     %3s    %2d   %20.12f*", multi,
                            irrep_symbol[irrep].c_str(), i, Edsrg_vec[Edsrg_vec.size() - 1][n][i]);
        }
        outfile->Printf("\n    %s", dash1.c_str());
    }

    if (failed) {
        throw PSIEXCEPTION("Reference relaxation process does not converge.");
    }

    Process::environment.globals["CURRENT ENERGY"] = Erelax_sa;
    return Erelax_sa;
}

void MRDSRG::transfer_integrals() {
    // printing
    print_h2("De-Normal-Order the DSRG Transformed Hamiltonian");

    // compute scalar term (all active only)
    Timer t_scalar;
    std::string str = "Computing the scalar term   ...";
    outfile->Printf("\n    %-35s", str.c_str());
    double scalar0 =
        Eref_ + Hbar0_ - molecule_->nuclear_repulsion_energy() - ints_->frozen_core_energy();

    // scalar from Hbar1
    double scalar1 = 0.0;
    scalar1 -= Hbar1_["vu"] * Gamma1_["uv"];
    scalar1 -= Hbar1_["VU"] * Gamma1_["UV"];

    // scalar from Hbar2
    double scalar2 = 0.0;
    scalar2 += 0.5 * Gamma1_["uv"] * Hbar2_["vyux"] * Gamma1_["xy"];
    scalar2 += 0.5 * Gamma1_["UV"] * Hbar2_["VYUX"] * Gamma1_["XY"];
    scalar2 += Gamma1_["uv"] * Hbar2_["vYuX"] * Gamma1_["XY"];

    scalar2 -= 0.25 * Hbar2_["xyuv"] * Lambda2_["uvxy"];
    scalar2 -= 0.25 * Hbar2_["XYUV"] * Lambda2_["UVXY"];
    scalar2 -= Hbar2_["xYuV"] * Lambda2_["uVxY"];

    double scalar = scalar0 + scalar1 + scalar2;
    outfile->Printf("  Done. Timing %10.3f s", t_scalar.get());

    // compute one-body term
    Timer t_one;
    str = "Computing the one-body term ...";
    outfile->Printf("\n    %-35s", str.c_str());
    BlockedTensor temp1 = BTF_->build(tensor_type_, "temp1", spin_cases({"aa"}));
    temp1["uv"] = Hbar1_["uv"];
    temp1["UV"] = Hbar1_["UV"];
    temp1["uv"] -= Hbar2_["uxvy"] * Gamma1_["yx"];
    temp1["uv"] -= Hbar2_["uXvY"] * Gamma1_["YX"];
    temp1["UV"] -= Hbar2_["xUyV"] * Gamma1_["yx"];
    temp1["UV"] -= Hbar2_["UXVY"] * Gamma1_["YX"];
    outfile->Printf("  Done. Timing %10.3f s", t_one.get());

    //    // compute hole contributions using bare Hamiltonian
    //    double h0 = 0.0;
    //    BlockedTensor h1 =
    //    BTF_->build(tensor_type_,"hole1",spin_cases({"cc"}));
    //    h1["mn"] += V_["mvnu"] * Gamma1_["uv"];
    //    h1["mn"] += V_["mVnU"] * Gamma1_["UV"];
    //    h1["MN"] += V_["vMuN"] * Gamma1_["uv"];
    //    h1["MN"] += V_["MVNU"] * Gamma1_["UV"];

    //    // - 1) -sum_{m} H^{m}_{m} + \sum_{muv} H^{mv}_{mu} * L^{u}_{v}
    //    for(const std::string block: {"cc", "CC"}){
    //        F_.block(block).citerate([&](const std::vector<size_t>& i,const
    //        double& value){
    //            if(i[0] == i[1]){
    //                h0 -= value;
    //            }
    //        });

    //        h1.block(block).citerate([&](const std::vector<size_t>& i,const
    //        double& value){
    //            if(i[0] == i[1]){
    //                h0 += value;
    //            }
    //        });
    //    }

    //    // - 2) 0.5 * sum_{mn} H^{mn}_{mn}
    //    for(const std::string block: {"cccc", "CCCC"}){
    //        V_.block(block).citerate([&](const std::vector<size_t>& i,const
    //        double& value){
    //            if(i[0] == i[2] && i[1] == i[3]){
    //                h0 += 0.5 * value;
    //            }
    //        });
    //    }

    //    V_.block("cCcC").citerate([&](const std::vector<size_t>& i,const
    //    double& value){
    //        if(i[0] == i[2] && i[1] == i[3]){
    //            h0 += value;
    //        }
    //    });

    //    scalar += h0;

    //    // - 3) -sum_{m} H^{qm}_{pm}
    //    for(const std::string block: {"cc", "CC"}){
    //        (h1.block(block)).iterate([&](const std::vector<size_t>& i,double&
    //        value){
    //            value = i[0] == i[1] ? 1.0 : 0.0;
    //        });
    //    }

    //    temp1["uv"] -= V_["umvn"] * h1["nm"];
    //    temp1["uv"] -= V_["uMvN"] * h1["NM"];
    //    temp1["UV"] -= V_["mUnV"] * h1["nm"];
    //    temp1["UV"] -= V_["UMVN"] * h1["NM"];

    // update integrals
    Timer t_int;
    str = "Updating integrals          ...";
    outfile->Printf("\n    %-35s", str.c_str());
    ints_->set_scalar(scalar);

    //   a) zero hole integrals
    std::vector<size_t> hole_mos = acore_mos_;
    hole_mos.insert(hole_mos.end(), aactv_mos_.begin(), aactv_mos_.end());
    for (const size_t& i : hole_mos) {
        for (const size_t& j : hole_mos) {
            ints_->set_oei(i, j, 0.0, true);
            ints_->set_oei(i, j, 0.0, false);
            for (const size_t& k : hole_mos) {
                for (const size_t& l : hole_mos) {
                    ints_->set_tei(i, j, k, l, 0.0, true, true);
                    ints_->set_tei(i, j, k, l, 0.0, true, false);
                    ints_->set_tei(i, j, k, l, 0.0, false, false);
                }
            }
        }
    }

    //   b) copy all active part
    temp1.citerate(
        [&](const std::vector<size_t>& i, const std::vector<SpinType>& spin, const double& value) {
            if (spin[0] == AlphaSpin) {
                ints_->set_oei(i[0], i[1], value, true);
            } else {
                ints_->set_oei(i[0], i[1], value, false);
            }
        });
    O1_["uv"] = temp1["uv"];
    O1_["UV"] = temp1["UV"];

    BlockedTensor temp2 = BTF_->build(tensor_type_, "temp2", spin_cases({"aaaa"}));
    temp2["uvxy"] = Hbar2_["uvxy"];
    temp2["uVxY"] = Hbar2_["uVxY"];
    temp2["UVXY"] = Hbar2_["UVXY"];
    temp2.citerate(
        [&](const std::vector<size_t>& i, const std::vector<SpinType>& spin, const double& value) {
            if ((spin[0] == AlphaSpin) && (spin[1] == AlphaSpin)) {
                ints_->set_tei(i[0], i[1], i[2], i[3], value, true, true);
            } else if ((spin[0] == AlphaSpin) && (spin[1] == BetaSpin)) {
                ints_->set_tei(i[0], i[1], i[2], i[3], value, true, false);
            } else if ((spin[0] == BetaSpin) && (spin[1] == BetaSpin)) {
                ints_->set_tei(i[0], i[1], i[2], i[3], value, false, false);
            }
        });
    outfile->Printf("  Done. Timing %10.3f s", t_int.get());

    // print scalar
    double scalar_include_fc = scalar + ints_->frozen_core_energy();
    print_h2("Scalar of the DSRG Hamiltonian (WRT True Vacuum)");
    outfile->Printf("\n    %-30s = %22.15f", "Scalar0", scalar0);
    outfile->Printf("\n    %-30s = %22.15f", "Scalar1", scalar1);
    outfile->Printf("\n    %-30s = %22.15f", "Scalar2", scalar2);
    outfile->Printf("\n    %-30s = %22.15f", "Total Scalar W/O Frozen-Core", scalar);
    outfile->Printf("\n    %-30s = %22.15f", "Total Scalar W/  Frozen-Core", scalar_include_fc);

    // test if de-normal-ordering is correct
    print_h2("Test De-Normal-Ordered Hamiltonian");
    double Etest = scalar_include_fc + molecule_->nuclear_repulsion_energy();

    double Etest1 = 0.0;
    Etest1 += temp1["uv"] * Gamma1_["vu"];
    Etest1 += temp1["UV"] * Gamma1_["VU"];

    Etest1 += Hbar1_["uv"] * Gamma1_["vu"];
    Etest1 += Hbar1_["UV"] * Gamma1_["VU"];
    Etest1 *= 0.5;

    //    for(const std::string block: {"cc","CC"}){
    //        F_.block(block).citerate([&](const std::vector<size_t>& i,const
    //        double& value){
    //            if (i[0] == i[1]){
    //                Etest1 += 0.5 * value;
    //            }
    //        });
    //        H_.block(block).citerate([&](const std::vector<size_t>& i,const
    //        double& value){
    //            if (i[0] == i[1]){
    //                Etest1 += 0.5 * value;
    //            }
    //        });
    //    }

    double Etest2 = 0.0;
    Etest2 += 0.25 * Hbar2_["uvxy"] * Lambda2_["xyuv"];
    Etest2 += 0.25 * Hbar2_["UVXY"] * Lambda2_["XYUV"];
    Etest2 += Hbar2_["uVxY"] * Lambda2_["xYuV"];

    Etest += Etest1 + Etest2;
    outfile->Printf("\n    %-30s = %22.15f", "One-Body Energy (after)", Etest1);
    outfile->Printf("\n    %-30s = %22.15f", "Two-Body Energy (after)", Etest2);
    outfile->Printf("\n    %-30s = %22.15f", "Total Energy (after)", Etest);
    outfile->Printf("\n    %-30s = %22.15f", "Total Energy (before)", Eref_ + Hbar0_);

    if (std::fabs(Etest - Eref_ - Hbar0_) > 100.0 * options_.get_double("E_CONVERGENCE")) {
        throw PSIEXCEPTION("De-normal-odering failed.");
    } else {
        ints_->update_integrals(false);
    }
}

void MRDSRG::reset_ints(BlockedTensor& H, BlockedTensor& V) {
    ints_->set_scalar(0.0);
    H.citerate(
        [&](const std::vector<size_t>& i, const std::vector<SpinType>& spin, const double& value) {
            if (spin[0] == AlphaSpin) {
                ints_->set_oei(i[0], i[1], value, true);
            } else {
                ints_->set_oei(i[0], i[1], value, false);
            }
        });
    V.citerate(
        [&](const std::vector<size_t>& i, const std::vector<SpinType>& spin, const double& value) {
            if ((spin[0] == AlphaSpin) && (spin[1] == AlphaSpin)) {
                ints_->set_tei(i[0], i[1], i[2], i[3], value, true, true);
            } else if ((spin[0] == AlphaSpin) && (spin[1] == BetaSpin)) {
                ints_->set_tei(i[0], i[1], i[2], i[3], value, true, false);
            } else if ((spin[0] == BetaSpin) && (spin[1] == BetaSpin)) {
                ints_->set_tei(i[0], i[1], i[2], i[3], value, false, false);
            }
        });
    ints_->update_integrals(false);
}

std::vector<std::vector<double>> MRDSRG::diagonalize_Fock_diagblocks(BlockedTensor& U) {
    // diagonal blocks identifiers (C-A-V ordering)
    std::vector<std::string> blocks = diag_one_labels();

    // map MO space label to its Dimension
    std::map<std::string, Dimension> MOlabel_to_dimension;
    MOlabel_to_dimension[acore_label_] = mo_space_info_->get_dimension("RESTRICTED_DOCC");
    MOlabel_to_dimension[aactv_label_] = mo_space_info_->get_dimension("ACTIVE");
    MOlabel_to_dimension[avirt_label_] = mo_space_info_->get_dimension("RESTRICTED_UOCC");

    // eigen values to be returned
    size_t ncmo = mo_space_info_->size("CORRELATED");
    Dimension corr = mo_space_info_->get_dimension("CORRELATED");
    std::vector<double> eigenvalues_a(ncmo, 0.0);
    std::vector<double> eigenvalues_b(ncmo, 0.0);

    // map MO space label to its offset Dimension
    std::map<std::string, Dimension> MOlabel_to_offset_dimension;
    int nirrep = corr.n();
    MOlabel_to_offset_dimension[acore_label_] = Dimension(std::vector<int>(nirrep, 0));
    MOlabel_to_offset_dimension[aactv_label_] = mo_space_info_->get_dimension("RESTRICTED_DOCC");
    MOlabel_to_offset_dimension[avirt_label_] =
        mo_space_info_->get_dimension("RESTRICTED_DOCC") + mo_space_info_->get_dimension("ACTIVE");

    // figure out index
    auto fill_eigen = [&](std::string block_label, int irrep, std::vector<double> values) {
        int h = irrep;
        size_t idx_begin = 0;
        while ((--h) >= 0)
            idx_begin += corr[h];

        std::string label(1, tolower(block_label[0]));
        idx_begin += MOlabel_to_offset_dimension[label][irrep];

        bool spin_alpha = islower(block_label[0]);
        size_t nvalues = values.size();
        if (spin_alpha) {
            for (size_t i = 0; i < nvalues; ++i) {
                eigenvalues_a[i + idx_begin] = values[i];
            }
        } else {
            for (size_t i = 0; i < nvalues; ++i) {
                eigenvalues_b[i + idx_begin] = values[i];
            }
        }
    };

    // diagonalize diagonal blocks
    for (const auto& block : blocks) {
        size_t dim = F_.block(block).dim(0);
        if (dim == 0) {
            continue;
        } else {
            std::string label(1, tolower(block[0]));
            Dimension space = MOlabel_to_dimension[label];
            int nirrep = space.n();

            // separate Fock with irrep
            for (int h = 0; h < nirrep; ++h) {
                size_t h_dim = space[h];
                ambit::Tensor U_h;
                if (h_dim == 0) {
                    continue;
                } else if (h_dim == 1) {
                    U_h = ambit::Tensor::build(tensor_type_, "U_h", std::vector<size_t>(2, h_dim));
                    U_h.data()[0] = 1.0;
                    ambit::Tensor F_block =
                        ambit::Tensor::build(tensor_type_, "F_block", F_.block(block).dims());
                    F_block.data() = F_.block(block).data();
                    ambit::Tensor T_h = separate_tensor(F_block, space, h);
                    fill_eigen(block, h, T_h.data());
                } else {
                    ambit::Tensor F_block =
                        ambit::Tensor::build(tensor_type_, "F_block", F_.block(block).dims());
                    F_block.data() = F_.block(block).data();
                    ambit::Tensor T_h = separate_tensor(F_block, space, h);
                    auto Feigen = T_h.syev(AscendingEigenvalue);
                    U_h = ambit::Tensor::build(tensor_type_, "U_h", std::vector<size_t>(2, h_dim));
                    U_h("pq") = Feigen["eigenvectors"]("pq");
                    fill_eigen(block, h, Feigen["eigenvalues"].data());
                }
                ambit::Tensor U_out = U.block(block);
                combine_tensor(U_out, U_h, space, h);
            }
        }
    }
    return {eigenvalues_a, eigenvalues_b};
}

ambit::Tensor MRDSRG::separate_tensor(ambit::Tensor& tens, const Dimension& irrep, const int& h) {
    // test tens and irrep
    int tens_dim = static_cast<int>(tens.dim(0));
    if (tens_dim != irrep.sum() || tens_dim != tens.dim(1)) {
        throw PSIEXCEPTION("Wrong dimension for the to-be-separated ambit Tensor.");
    }
    if (h >= irrep.n()) {
        throw PSIEXCEPTION("Ask for wrong irrep.");
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
    ambit::Tensor T_h = ambit::Tensor::build(tensor_type_, "T_h", std::vector<size_t>(2, h_dim));
    for (size_t i = 0; i < h_dim; ++i) {
        for (size_t j = 0; j < h_dim; ++j) {
            size_t abs_idx = rel_to_abs(i, j, offset);
            T_h.data()[i * h_dim + j] = tens.data()[abs_idx];
        }
    }

    return T_h;
}

void MRDSRG::combine_tensor(ambit::Tensor& tens, ambit::Tensor& tens_h, const Dimension& irrep,
                            const int& h) {
    // test tens and irrep
    if (h >= irrep.n()) {
        throw PSIEXCEPTION("Ask for wrong irrep.");
    }
    size_t tens_h_dim = tens_h.dim(0), h_dim = irrep[h];
    if (tens_h_dim != h_dim || tens_h_dim != tens_h.dim(1)) {
        throw PSIEXCEPTION("Wrong dimension for the to-be-combined ambit Tensor.");
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

void MRDSRG::print_cumulant_summary() {
    print_h2("Density Cumulant Summary");
    check_density(Lambda2_, "2-body");
    if (options_.get_str("THREEPDC") != "ZERO") {
        check_density(Lambda3_, "3-body");
    }
}

void MRDSRG::check_density(BlockedTensor& D, const std::string& name) {
    int rank_half = D.rank() / 2;
    std::vector<std::string> labels;
    std::vector<double> maxes, norms;
    std::vector<std::string> blocks = D.block_labels();
    for (const auto& block : blocks) {
        std::string spin_label;
        std::vector<int> idx;
        for (int i = 0; i < rank_half; ++i) {
            idx.emplace_back(i);
        }
        for (const auto& i : idx) {
            if (islower(block[i])) {
                spin_label += "A";
            } else {
                spin_label += "B";
            }
        }
        labels.emplace_back(spin_label);

        double D_norm = 0.0, D_max = 0.0;
        D.block(block).citerate([&](const std::vector<size_t>&, const double& value) {
            double abs_value = std::fabs(value);
            if (abs_value > 1.0e-15) {
                if (abs_value > D_max)
                    D_max = value;
                D_norm += value * value;
            }
        });
        maxes.emplace_back(D_max);
        norms.emplace_back(std::sqrt(D_norm));
    }

    int n = labels.size();
    std::string sep(10 + 13 * n, '-');
    std::string indent = "\n    ";
    std::string output = indent + str(boost::format("%-10s") % name);
    for (int i = 0; i < n; ++i)
        output += str(boost::format(" %12s") % labels[i]);
    output += indent + sep;

    output += indent + str(boost::format("%-10s") % "max");
    for (int i = 0; i < n; ++i)
        output += str(boost::format(" %12.6f") % maxes[i]);
    output += indent + str(boost::format("%-10s") % "norm");
    for (int i = 0; i < n; ++i)
        output += str(boost::format(" %12.6f") % norms[i]);
    output += indent + sep;
    outfile->Printf("%s", output.c_str());
}

std::vector<std::string> MRDSRG::diag_one_labels() {
    // C-A-V ordering
    std::vector<std::string> labels{acore_label_ + acore_label_, aactv_label_ + aactv_label_,
                                    avirt_label_ + avirt_label_, bcore_label_ + bcore_label_,
                                    bactv_label_ + bactv_label_, bvirt_label_ + bvirt_label_};
    return labels;
}

std::vector<std::string> MRDSRG::re_two_labels() {
    std::vector<std::string> labels;
    std::vector<std::string> labels_aa{acore_label_ + acore_label_ + acore_label_ + acore_label_,
                                       aactv_label_ + aactv_label_ + aactv_label_ + aactv_label_,
                                       avirt_label_ + avirt_label_ + avirt_label_ + avirt_label_};
    std::vector<std::string> mixed{acore_label_ + aactv_label_, acore_label_ + avirt_label_,
                                   aactv_label_ + avirt_label_};
    for (const std::string& half : mixed) {
        std::string reverse(half);
        std::reverse(reverse.begin(), reverse.end());
        labels_aa.push_back(half + half);
        labels_aa.push_back(half + reverse);
        labels_aa.push_back(reverse + half);
        labels_aa.push_back(reverse + reverse);
    }
    for (const std::string& label : labels_aa) {
        std::string aa(label);
        std::string ab(label);
        std::string bb(label);

        for (const int& idx : {1, 3}) {
            ab[idx] = std::toupper(ab[idx]);
            bb[idx] = std::toupper(bb[idx]);
        }
        for (const int& idx : {0, 2}) {
            bb[idx] = std::toupper(bb[idx]);
        }

        labels.push_back(aa);
        labels.push_back(ab);
        labels.push_back(bb);
    }
    //    std::vector<std::string> labels (V_.block_labels());
    //    std::vector<std::string> od_labels (od_two_labels());
    //    labels.erase(std::remove_if(labels.begin(), labels.end(),
    //                                 [&](std::string i) {return
    //                                 std::find(od_labels.begin(),
    //                                 od_labels.end(), i) !=
    //                                 od_labels.end();}),
    //            labels.end());
    return labels;
}

std::vector<std::string> MRDSRG::diag_two_labels() {
    std::vector<std::string> general{acore_label_, aactv_label_, avirt_label_};
    std::vector<std::string> hole{acore_label_, aactv_label_};
    std::vector<std::string> particle{aactv_label_, avirt_label_};

    std::vector<std::string> d_aa;
    for (const std::string& p : general) {
        for (const std::string& q : general) {
            for (const std::string& r : general) {
                for (const std::string& s : general) {
                    d_aa.push_back(p + q + r + s);
                }
            }
        }
    }

    for (const std::string& p : hole) {
        for (const std::string& q : hole) {
            for (const std::string& r : particle) {
                for (const std::string& s : particle) {
                    if (p == aactv_label_ && q == aactv_label_ && r == aactv_label_ &&
                        s == aactv_label_) {
                        continue;
                    }

                    std::vector<std::string> od_aa{p + q + r + s, r + s + p + q};
                    d_aa.erase(std::remove_if(d_aa.begin(), d_aa.end(),
                                              [&](std::string i) {
                                                  return std::find(od_aa.begin(), od_aa.end(), i) !=
                                                         od_aa.end();
                                              }),
                               d_aa.end());
                }
            }
        }
    }

    std::vector<std::string> labels;
    for (const std::string& label : d_aa) {
        std::string aa(label);
        std::string ab(label);
        std::string bb(label);

        for (const int& idx : {1, 3}) {
            ab[idx] = std::toupper(ab[idx]);
            bb[idx] = std::toupper(bb[idx]);
        }
        for (const int& idx : {0, 2}) {
            bb[idx] = std::toupper(bb[idx]);
        }

        labels.push_back(aa);
        labels.push_back(ab);
        labels.push_back(bb);
    }

    return labels;
}

std::vector<std::string> MRDSRG::od_one_labels_hp() {
    std::vector<std::string> labels_a;
    for (const std::string& p : {acore_label_, aactv_label_}) {
        for (const std::string& q : {aactv_label_, avirt_label_}) {
            if (p == aactv_label_ && q == aactv_label_) {
                continue;
            }
            labels_a.push_back(p + q);
        }
    }

    std::vector<std::string> blocks1;
    for (const std::string& label : labels_a) {
        std::string a(label);
        std::string b(label);

        for (const int& idx : {0, 1}) {
            b[idx] = std::toupper(b[idx]);
        }

        blocks1.push_back(a);
        blocks1.push_back(b);
    }
    return blocks1;
}

std::vector<std::string> MRDSRG::od_one_labels_ph() {
    std::vector<std::string> blocks1(od_one_labels_hp());
    for (auto& block : blocks1) {
        std::swap(block[0], block[1]);
    }
    return blocks1;
}

std::vector<std::string> MRDSRG::od_one_labels() {
    std::vector<std::string> blocks1(od_one_labels_hp());
    std::vector<std::string> temp(blocks1);
    for (auto& block : temp) {
        std::swap(block[0], block[1]);
    }
    blocks1.insert(std::end(blocks1), std::begin(temp), std::end(temp));
    return blocks1;
}

std::vector<std::string> MRDSRG::od_two_labels_hhpp() {
    std::vector<std::string> labels_aa;
    for (const std::string& p : {acore_label_, aactv_label_}) {
        for (const std::string& q : {acore_label_, aactv_label_}) {
            for (const std::string& r : {aactv_label_, avirt_label_}) {
                for (const std::string& s : {aactv_label_, avirt_label_}) {
                    if (p == aactv_label_ && q == aactv_label_ && r == aactv_label_ &&
                        s == aactv_label_) {
                        continue;
                    }
                    labels_aa.push_back(p + q + r + s);
                }
            }
        }
    }

    std::vector<std::string> blocks2;
    for (const std::string& label : labels_aa) {
        std::string aa(label);
        std::string ab(label);
        std::string bb(label);

        for (const int& idx : {1, 3}) {
            ab[idx] = std::toupper(ab[idx]);
            bb[idx] = std::toupper(bb[idx]);
        }
        for (const int& idx : {0, 2}) {
            bb[idx] = std::toupper(bb[idx]);
        }

        blocks2.push_back(aa);
        blocks2.push_back(ab);
        blocks2.push_back(bb);
    }

    return blocks2;
}

std::vector<std::string> MRDSRG::od_two_labels_pphh() {
    std::vector<std::string> blocks2(od_two_labels_hhpp());
    for (auto& block : blocks2) {
        std::swap(block[0], block[2]);
        std::swap(block[1], block[3]);
    }
    return blocks2;
}

std::vector<std::string> MRDSRG::od_two_labels() {
    std::vector<std::string> blocks2(od_two_labels_hhpp());
    std::vector<std::string> temp(blocks2);
    for (auto& block : temp) {
        std::swap(block[0], block[2]);
        std::swap(block[1], block[3]);
    }
    blocks2.insert(std::end(blocks2), std::begin(temp), std::end(temp));
    return blocks2;
}
}
}
