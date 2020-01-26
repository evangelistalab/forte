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
#include "base_classes/scf_info.h"
#include "helpers/helpers.h"
#include "helpers/printing.h"
#include "helpers/timer.h"
#include "cc.h"

using namespace psi;

namespace forte {

std::unique_ptr<CC_SO> make_cc_so(RDMs rdms, std::shared_ptr<SCFInfo> scf_info,
                                  std::shared_ptr<ForteOptions> options,
                                  std::shared_ptr<ForteIntegrals> ints,
                                  std::shared_ptr<MOSpaceInfo> mo_space_info) {
    return std::make_unique<CC_SO>(rdms, scf_info, options, ints, mo_space_info);
}

CC_SO::CC_SO(RDMs rdms, std::shared_ptr<SCFInfo> scf_info, std::shared_ptr<ForteOptions> options,
             std::shared_ptr<ForteIntegrals> ints, std::shared_ptr<MOSpaceInfo> mo_space_info)
    : DynamicCorrelationSolver(rdms, scf_info, options, ints, mo_space_info),
      BTF_(new BlockedTensorFactory()), tensor_type_(ambit::CoreTensor) {
    print_method_banner({"Spin-Orbital Coupled Cluster Using Generated Equations", "Chenyang Li"});
    startup();
    print_summary();
}

CC_SO::~CC_SO() {}

std::shared_ptr<ActiveSpaceIntegrals> CC_SO::compute_Heff_actv() {
    throw psi::PSIEXCEPTION(
        "Computing active-space Hamiltonian is not yet implemented for spin-orbital code.");
}

void CC_SO::startup() {
    BlockedTensor::reset_mo_spaces();
    BlockedTensor::set_expert_mode(true);

    Eref_ = compute_Eref_from_rdms(rdms_, ints_, mo_space_info_);
    Enuc_ = ints_->nuclear_repulsion_energy();
    Efrzc_ = ints_->frozen_core_energy();

    corr_level_ = foptions_->get_str("CC_LEVEL");
    do_triples_ = corr_level_.find("CCSDT") != std::string::npos or
                  corr_level_.find("CC3") != std::string::npos;
    trotter_level_ = foptions_->get_int("CCSD_TROTTER_LEVEL");
    trotter_sym_ = foptions_->get_bool("CCSD_TROTTER_SYMM");

    e_convergence_ = foptions_->get_double("E_CONVERGENCE");
    r_convergence_ = foptions_->get_double("R_CONVERGENCE");
    maxiter_ = foptions_->get_int("MAXITER");

    ntamp_ = foptions_->get_int("NTAMP");

    fink_order_ = foptions_->get_int("LUCCSDT_FINK_ORDER");
    if (fink_order_ < 4) {
        fink_order_ = 4;
    }
    if (fink_order_ > 8) {
        fink_order_ = 8;
    }

    // test for SOCC
    auto socc = scf_info_->soccpi();
    auto actv_dim = mo_space_info_->get_dimension("ACTIVE");
    if (socc != actv_dim) {
        throw PSIEXCEPTION("Inconsistent dimension for singly occupied orbitals.");
    }

    size_t nsocc = mo_space_info_->size("ACTIVE");
    size_t twice_ms = std::round(2.0 * foptions_->get_double("MS"));
    if (nsocc != twice_ms) {
        throw PSIEXCEPTION("Not high-spin configuration. Please change Ms.");
    }

    // orbital spaces
    acore_sos_ = mo_space_info_->get_corr_abs_mo("GENERALIZED HOLE");
    avirt_sos_ = mo_space_info_->get_corr_abs_mo("RESTRICTED_UOCC");

    // put all beta behind alpha
    size_t mo_shift = mo_space_info_->size("CORRELATED");
    for (size_t idx : mo_space_info_->get_corr_abs_mo("RESTRICTED_DOCC")) {
        bcore_sos_.push_back(idx + mo_shift);
    }
    for (size_t idx : mo_space_info_->get_corr_abs_mo("GENERALIZED PARTICLE")) {
        bvirt_sos_.push_back(idx + mo_shift);
    }

    // spin orbital indices
    core_sos_ = acore_sos_;
    virt_sos_ = avirt_sos_;
    core_sos_.insert(core_sos_.end(), bcore_sos_.begin(), bcore_sos_.end());
    virt_sos_.insert(virt_sos_.end(), bvirt_sos_.begin(), bvirt_sos_.end());

    // size of each spin orbital space
    nc_ = core_sos_.size();
    nv_ = virt_sos_.size();
    nso_ = nc_ + nv_;
    nmo_ = nso_ / 2;

    BTF_->add_mo_space("c", "i,j,k,l,m,n,c0,c1,c2,c3,c4,c5,c6,c7,c8,c9", core_sos_, NoSpin);
    BTF_->add_mo_space("v", "a,b,c,d,e,f,v0,v1,v2,v3,v4,v5,v6,v7,v8,v9", virt_sos_, NoSpin);

    BTF_->add_composite_mo_space("g", "p,q,r,s,t,o,g0,g1,g2,g3,g4,g5,g6,g7,g8,g9", {"c", "v"});

    // prepare one-electron integrals
    H_ = BTF_->build(tensor_type_, "H", {"gg"});
    H_.iterate([&](const std::vector<size_t>& i, const std::vector<SpinType>&, double& value) {
        if (i[0] < nmo_ && i[1] < nmo_) {
            value = ints_->oei_a(i[0], i[1]);
        }
        if (i[0] >= nmo_ && i[1] >= nmo_) {
            value = ints_->oei_b(i[0] - nmo_, i[1] - nmo_);
        }
    });

    // prepare two-electron integrals
    V_ = BTF_->build(tensor_type_, "V", {"gggg"});
    V_.iterate([&](const std::vector<size_t>& i, const std::vector<SpinType>&, double& value) {
        bool spin0 = i[0] < nmo_;
        bool spin1 = i[1] < nmo_;
        bool spin2 = i[2] < nmo_;
        bool spin3 = i[3] < nmo_;
        if (spin0 && spin1 && spin2 && spin3) {
            value = ints_->aptei_aa(i[0], i[1], i[2], i[3]);
        }
        if ((!spin0) && (!spin1) && (!spin2) && (!spin3)) {
            value = ints_->aptei_bb(i[0] - nmo_, i[1] - nmo_, i[2] - nmo_, i[3] - nmo_);
        }
        if (spin0 && (!spin1) && spin2 && (!spin3)) {
            value = ints_->aptei_ab(i[0], i[1] - nmo_, i[2], i[3] - nmo_);
        }
        if (spin1 && (!spin0) && spin3 && (!spin2)) {
            value = ints_->aptei_ab(i[1], i[0] - nmo_, i[3], i[2] - nmo_);
        }
        if (spin0 && (!spin1) && spin3 && (!spin2)) {
            value = -ints_->aptei_ab(i[0], i[1] - nmo_, i[3], i[2] - nmo_);
        }
        if (spin1 && (!spin0) && spin2 && (!spin3)) {
            value = -ints_->aptei_ab(i[1], i[0] - nmo_, i[2], i[3] - nmo_);
        }
    });

    // build Fock matrix (initial guess of one-body Hamiltonian)
    F_ = BTF_->build(tensor_type_, "Fock", {"gg"});
    F_["pq"] = H_["pq"];

    auto K1 = BTF_->build(tensor_type_, "Kronecker delta", {"cc"});
    (K1.block("cc")).iterate([&](const std::vector<size_t>& i, double& value) {
        value = (i[0] == i[1] ? 1.0 : 0.0);
    });
    F_["pq"] += V_["pjqi"] * K1["ij"];

    // obtain diagonal elements of Fock matrix
    Fd_ = std::vector<double>(nso_);
    F_.citerate(
        [&](const std::vector<size_t>& i, const std::vector<SpinType>&, const double& value) {
            if (i[0] == i[1]) {
                Fd_[i[0]] = value;
            }
        });

    // print orbital energies
    size_t nc_a = acore_sos_.size();
    size_t nc_b = bcore_sos_.size();
    print_h2("Orbital Energies");
    outfile->Printf("\n     MO     Alpha           Beta");
    outfile->Printf("\n    ---------------------------------");
    for (size_t i = 0; i < nmo_; ++i) {
        outfile->Printf("\n    %3zu %11.6f(%d) %11.6f(%d)", i + 1, Fd_[i], i < nc_a, Fd_[i + nmo_],
                        i < nc_b);
    }
    outfile->Printf("\n    ---------------------------------");
}

void CC_SO::print_summary() {
    // Print a summary
    std::vector<std::pair<std::string, int>> calculation_info_int{
        {"Max Iteration", maxiter_}, {"Number of Printed T Amplitudes", ntamp_}};

    std::vector<std::pair<std::string, double>> calculation_info_double{
        {"Energy Convergence", e_convergence_}, {"Residue Convergence", r_convergence_}};

    std::vector<std::pair<std::string, std::string>> calculation_info_string{
        {"Correlation Level", corr_level_}, {"Integral Type", foptions_->get_str("INT_TYPE")}};

    if (do_triples_) {
        calculation_info_int.push_back({"Perturbation Order (Fink)", fink_order_});
    }

    // Print some information
    outfile->Printf("\n\n  ==> Calculation Information <==\n");
    for (auto& str_dim : calculation_info_string) {
        outfile->Printf("\n    %-35s %20s", str_dim.first.c_str(), str_dim.second.c_str());
    }
    for (auto& str_dim : calculation_info_double) {
        outfile->Printf("\n    %-35s %20.3e", str_dim.first.c_str(), str_dim.second);
    }
    for (auto& str_dim : calculation_info_int) {
        outfile->Printf("\n    %-35s %20d", str_dim.first.c_str(), str_dim.second);
    }
}

void CC_SO::guess_t2() {
    local_timer timer;
    std::string str = "Computing T2 amplitudes     ...";
    outfile->Printf("\n    %-35s", str.c_str());

    T2_["ijab"] = V_["ijab"];

    T2_.iterate([&](const std::vector<size_t>& i, const std::vector<SpinType>&, double& value) {
        value *= 1.0 / (Fd_[i[0]] + Fd_[i[1]] - Fd_[i[2]] - Fd_[i[3]]);
    });

    // norm and max
    T2max_ = 0.0, T2norm_ = T2_.norm();
    T2_.citerate(
        [&](const std::vector<size_t>&, const std::vector<SpinType>&, const double& value) {
            if (std::fabs(value) > std::fabs(T2max_))
                T2max_ = value;
        });

    outfile->Printf("  Done. Timing %10.3f s", timer.get());
}

void CC_SO::guess_t1() {
    local_timer timer;
    std::string str = "Computing T1 amplitudes     ...";
    outfile->Printf("\n    %-35s", str.c_str());

    // use simple single-reference guess
    T1_["ia"] = F_["ia"];
    T1_.iterate([&](const std::vector<size_t>& i, const std::vector<SpinType>&, double& value) {
        value *= 1.0 / (Fd_[i[0]] - Fd_[i[1]]);
    });

    // norm and max
    T1max_ = 0.0, T1norm_ = T1_.norm();
    T1_.citerate(
        [&](const std::vector<size_t>&, const std::vector<SpinType>&, const double& value) {
            if (std::fabs(value) > std::fabs(T1max_))
                T1max_ = value;
        });

    outfile->Printf("  Done. Timing %10.3f s", timer.get());
}

void CC_SO::guess_t3() {
    local_timer timer;
    std::string str = "Computing T3 amplitudes     ...";
    outfile->Printf("\n    %-35s", str.c_str());

    ambit::BlockedTensor C3 = ambit::BlockedTensor::build(tensor_type_, "C3", {"cccvvv"});
    auto temp = ambit::BlockedTensor::build(CoreTensor, "temp", {"cccvvv"});
    temp["g2,c0,c1,g0,g1,v0"] += -1.0 * V_["g2,v1,g0,g1"] * T2_["c0,c1,v0,v1"];
    C3["c0,c1,g2,g0,g1,v0"] += temp["g2,c0,c1,g0,g1,v0"];
    C3["c0,g2,c1,g0,g1,v0"] -= temp["g2,c0,c1,g0,g1,v0"];
    C3["g2,c0,c1,g0,g1,v0"] += temp["g2,c0,c1,g0,g1,v0"];
    C3["c0,c1,g2,g0,v0,g1"] -= temp["g2,c0,c1,g0,g1,v0"];
    C3["c0,g2,c1,g0,v0,g1"] += temp["g2,c0,c1,g0,g1,v0"];
    C3["g2,c0,c1,g0,v0,g1"] -= temp["g2,c0,c1,g0,g1,v0"];
    C3["c0,c1,g2,v0,g0,g1"] += temp["g2,c0,c1,g0,g1,v0"];
    C3["c0,g2,c1,v0,g0,g1"] -= temp["g2,c0,c1,g0,g1,v0"];
    C3["g2,c0,c1,v0,g0,g1"] += temp["g2,c0,c1,g0,g1,v0"];

    temp.zero();
    temp["g1,g2,c0,g0,v0,v1"] += 1.0 * V_["g1,g2,g0,c1"] * T2_["c0,c1,v0,v1"];
    C3["c0,g1,g2,g0,v0,v1"] += temp["g1,g2,c0,g0,v0,v1"];
    C3["g1,c0,g2,g0,v0,v1"] -= temp["g1,g2,c0,g0,v0,v1"];
    C3["g1,g2,c0,g0,v0,v1"] += temp["g1,g2,c0,g0,v0,v1"];
    C3["c0,g1,g2,v0,g0,v1"] -= temp["g1,g2,c0,g0,v0,v1"];
    C3["g1,c0,g2,v0,g0,v1"] += temp["g1,g2,c0,g0,v0,v1"];
    C3["g1,g2,c0,v0,g0,v1"] -= temp["g1,g2,c0,g0,v0,v1"];
    C3["c0,g1,g2,v0,v1,g0"] += temp["g1,g2,c0,g0,v0,v1"];
    C3["g1,c0,g2,v0,v1,g0"] -= temp["g1,g2,c0,g0,v0,v1"];
    C3["g1,g2,c0,v0,v1,g0"] += temp["g1,g2,c0,g0,v0,v1"];

    T3_["ijkabc"] = C3["ijkabc"];
    T3_["ijkabc"] += C3["abcijk"];

    T3_.iterate([&](const std::vector<size_t>& i, const std::vector<SpinType>&, double& value) {
        value *= 1.0 / (Fd_[i[0]] + Fd_[i[1]] + Fd_[i[2]] - Fd_[i[3]] - Fd_[i[4]] - Fd_[i[5]]);
    });

    // norm and max
    T3max_ = T3_.norm(0), T3norm_ = T3_.norm();

    outfile->Printf("  Done. Timing %10.3f s", timer.get());
}

void CC_SO::update_t2() {
    // compute DT2 = Hbar2 / D
    BlockedTensor DT2 = ambit::BlockedTensor::build(tensor_type_, "DT2", {"ccvv"});
    DT2["ijab"] = Hbar2_["ijab"];
    DT2.iterate([&](const std::vector<size_t>& i, const std::vector<SpinType>&, double& value) {
        value *= 1.0 / (Fd_[i[0]] + Fd_[i[1]] - Fd_[i[2]] - Fd_[i[3]]);
    });

    rms_t2_ = DT2.norm();

    T2_["ijab"] += DT2["ijab"];

    // norm and max
    T2max_ = T2_.norm(0), T2norm_ = T2_.norm();
}

void CC_SO::update_t1() {
    // compute DT1 = Hbar1 / D
    BlockedTensor DT1 = ambit::BlockedTensor::build(tensor_type_, "DT1", {"cv"});
    DT1["ia"] = Hbar1_["ia"];
    DT1.iterate([&](const std::vector<size_t>& i, const std::vector<SpinType>&, double& value) {
        value *= 1.0 / (Fd_[i[0]] - Fd_[i[1]]);
    });

    rms_t1_ = DT1.norm();

    T1_["ia"] += DT1["ia"];

    // norm and max
    T1max_ = T1_.norm(0), T1norm_ = T1_.norm();
}

void CC_SO::update_t3() {
    // compute DT3 = Hbar3 / D
    BlockedTensor DT3 = ambit::BlockedTensor::build(tensor_type_, "DT3", {"cccvvv"});
    DT3["ijkabc"] = Hbar3_["ijkabc"];
    DT3.iterate([&](const std::vector<size_t>& i, const std::vector<SpinType>&, double& value) {
        value *= 1.0 / (Fd_[i[0]] + Fd_[i[1]] + Fd_[i[2]] - Fd_[i[3]] - Fd_[i[4]] - Fd_[i[5]]);
    });

    rms_t3_ = DT3.norm();

    T3_["ijkabc"] += DT3["ijkabc"];

    // norm and max
    T3max_ = T3_.norm(0), T3norm_ = T3_.norm();
}

double CC_SO::compute_energy() {
    if (corr_level_ == "CCSD" or corr_level_ == "CCSD_TROTTER" or corr_level_ == "CCSDT" or
        corr_level_ == "CCSDT_1A" or corr_level_ == "CCSDT_1B" or corr_level_ == "UCC3" or
        corr_level_ == "VUCCSD5") {
        Hbar1_ = BTF_->build(tensor_type_, "Hbar1", {"cv"});
        Hbar2_ = BTF_->build(tensor_type_, "Hbar2", {"ccvv"});
    } else {
        throw PSIEXCEPTION("Not Implemented yet!");
        Hbar1_ = BTF_->build(tensor_type_, "Hbar1", {"gg"});
        Hbar2_ = BTF_->build(tensor_type_, "Hbar2", {"gggg"});
    }

    //    // initialize Hbar with bare Hamiltonian
    //    Hbar0_ = 0.0;
    //    Hbar1_["pq"] = F_["pq"];
    //    Hbar2_["pqrs"] = V_["pqrs"];

    // build initial amplitudes
    print_h2("Build Initial Cluster Amplitudes");
    T1_ = BTF_->build(tensor_type_, "T1 Amplitudes", {"cv"});
    T2_ = BTF_->build(tensor_type_, "T2 Amplitudes", {"ccvv"});
    guess_t2();
    guess_t1();

    if (do_triples_) {
        if (corr_level_ == "CCSDT" or corr_level_ == "CCSDT_1A" or corr_level_ == "CCSDT_1B" or
            corr_level_ == "UCC3") {
            Hbar3_ = BTF_->build(tensor_type_, "Hbar3", {"cccvvv"});
        } else {
            Hbar3_ = BTF_->build(tensor_type_, "Hbar3", {"gggggg"});
        }

        T3_ = BTF_->build(tensor_type_, "T3 Amplitudes", {"cccvvv"});
        //        guess_t3();
    }

    // iteration variables
    double Etotal = Eref_;
    bool converged = false;

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

    for (int cycle = 1; cycle <= maxiter_; ++cycle) {
        if (corr_level_ == "CCSD") {
            compute_ccsd_amp(F_, V_, T1_, T2_, Hbar0_, Hbar1_, Hbar2_);
        } else if (corr_level_ == "CCSD_TROTTER") {
            compute_ccsd_trotter(F_, V_, T1_, T2_, Hbar0_, Hbar1_, Hbar2_);
        } else if (corr_level_ == "CCSDT") {
            compute_ccsdt_amp(F_, V_, T1_, T2_, T3_, Hbar0_, Hbar1_, Hbar2_, Hbar3_);
        } else if (corr_level_ == "CCSDT_1A" or corr_level_ == "CCSDT_1B") {
            compute_ccsdt1_amp(F_, V_, T1_, T2_, T3_, Hbar0_, Hbar1_, Hbar2_, Hbar3_);
        } else if (corr_level_ == "UCC3") {
            double Eeff = 0.0;
            ambit::BlockedTensor Frot = BTF_->build(tensor_type_, "Frot", {"gg"});
            ambit::BlockedTensor Vrot = BTF_->build(tensor_type_, "Vrot", {"gggg"});
            rotate_hamiltonian(Eeff, Frot, Vrot);
            //            outfile->Printf("\n  Eeff = %.15f", Eeff);

            compute_ucc3_amp(Frot, Vrot, T2_, T3_, Hbar0_, Hbar1_, Hbar2_, Hbar3_);

            Hbar0_ += Eeff;
        } else if (corr_level_ == "VUCCSD5") {
            double Eeff = 0.0;
            ambit::BlockedTensor Frot = BTF_->build(tensor_type_, "Frot", {"gg"});
            ambit::BlockedTensor Vrot = BTF_->build(tensor_type_, "Vrot", {"gggg"});
            rotate_hamiltonian(Eeff, Frot, Vrot);
            compute_uccsd5_amp(Frot, Vrot, T2_, Hbar0_, Hbar1_, Hbar2_);
            Hbar0_ += Eeff;
        }

        double Edelta = Eref_ + Hbar0_ - Etotal;
        Etotal = Eref_ + Hbar0_;

        // norm of non-diagonal Hbar
        BlockedTensor temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"cv"});
        temp["ia"] = Hbar1_["ia"];
        double Hbar1Nnorm = temp.norm();

        temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"ccvv"});
        temp["ijab"] = Hbar2_["ijab"];
        double Hbar2Nnorm = temp.norm();
        for (const std::string block : temp.block_labels()) {
            temp.block(block).reset();
        }

        outfile->Printf("\n      @CC %4d %20.12f %11.3e %10.3e %10.3e %7.4f "
                        "%7.4f %7.4f %7.4f %7.4f %7.4f",
                        cycle, Etotal, Edelta, Hbar1Nnorm, Hbar2Nnorm, T1norm_, T2norm_, T3norm_,
                        T1max_, T2max_, T3max_);

        update_t2();
        update_t1();
        if (do_triples_) {
            update_t3();
        }

        // test convergence
        double rms = std::max(std::max(rms_t1_, rms_t2_), rms_t3_);
        if (std::fabs(Edelta) < e_convergence_ && rms < r_convergence_) {
            converged = true;
        }

        if (converged) {
            break;
        }
    }

    outfile->Printf("\n    "
                    "----------------------------------------------------------"
                    "----------------------------------------");
    outfile->Printf("\n\n\n    %s Energy Summary", corr_level_.c_str());
    outfile->Printf("\n    Correlation energy      = %25.15f", Etotal - Eref_);
    outfile->Printf("\n  * Total energy            = %25.15f\n", Etotal);

    if (not converged) {
        throw PSIEXCEPTION("CC_SO computation did not converged.");
    }

    psi::Process::environment.globals["CURRENT ENERGY"] = Etotal;

    return Etotal;
}

void CC_SO::rotate_hamiltonian(double& Eeff, BlockedTensor& Fnew, BlockedTensor& Vnew) {
    ambit::BlockedTensor A1 = BTF_->build(tensor_type_, "A1 Amplitudes", {"gg"});
    A1["ia"] = T1_["ia"];
    A1["ai"] -= T1_["ia"];

    psi::SharedMatrix A1_m(new psi::Matrix("A1", nso_, nso_));
    A1.iterate([&](const std::vector<size_t>& i, const std::vector<SpinType>&, double& value) {
        A1_m->set(i[0], i[1], value);
    });

    // >=3 is required for high energy convergence
    A1_m->expm(3);

    ambit::BlockedTensor U1 = BTF_->build(tensor_type_, "Transformer", {"gg"});
    U1.iterate([&](const std::vector<size_t>& i, const std::vector<SpinType>&, double& value) {
        value = A1_m->get(i[0], i[1]);
    });

    // Recompute Hbar0 (ref. energy + T1 correlation), Fnew (Fock), and Vnew (aptei)
    // E = 0.5 * ( H["ji"] + F["ji] ) * D1["ij"]

    Fnew["rs"] = U1["rp"] * H_["pq"] * U1["sq"];

    Eeff = 0.0;
    Fnew.block("cc").iterate([&](const std::vector<size_t>& i, double& value) {
        if (i[0] == i[1]) {
            Eeff += 0.5 * value;
        }
    });

    Vnew["g0,g1,g2,g3"] = U1["g0,g4"] * U1["g1,g5"] * V_["g4,g5,g6,g7"] * U1["g2,g6"] * U1["g3,g7"];

    auto K1 = BTF_->build(tensor_type_, "Kronecker delta", {"cc"});
    (K1.block("cc")).iterate([&](const std::vector<size_t>& i, double& value) {
        value = (i[0] == i[1] ? 1.0 : 0.0);
    });
    Fnew["pq"] += Vnew["pjqi"] * K1["ij"];

    // compute fully contracted term from T1
    Fnew.block("cc").iterate([&](const std::vector<size_t>& i, double& value) {
        if (i[0] == i[1]) {
            Eeff += 0.5 * value;
        }
    });

    Eeff += Efrzc_ + Enuc_ - Eref_;
}

// void CC_SO::compute_lhbar() {

//    //    outfile->Printf("\n\n  Computing the similarity-transformed
//    //    Hamiltonian");
//    //    outfile->Printf("\n
//    //    -----------------------------------------------------------------");
//    //    outfile->Printf("\n  nComm           C0                 |C1| |C2|" );
//    //    outfile->Printf("\n
//    //    -----------------------------------------------------------------");

//    // copy initial one-body Hamiltonian
//    Hbar0_ = 0.0;
//    Hbar1_["pq"] = F["pq"];
//    Hbar2_["pqrs"] = V["pqrs"];

//    BlockedTensor O1 = ambit::BlockedTensor::build(tensor_type_, "O1", {"gg"});
//    BlockedTensor O2 = ambit::BlockedTensor::build(tensor_type_, "O2", {"gggg"});
//    O1["pq"] = F["pq"];
//    O2["pqrs"] = V["pqrs"];

//    //    outfile->Printf("\n  %2d %20.12f %20e
//    //    %20e",0,Hbar0,Hbar1.norm(),Hbar2.norm());

//    // iterator variables
//    int maxn = foptions_->get_int("DSRG_RSC_NCOMM");
//    double ct_threshold = foptions_->get_double("DSRG_RSC_THRESHOLD");
//    double C0 = 0.0;
//    BlockedTensor C1 = ambit::BlockedTensor::build(tensor_type_, "C1", {"gg"});
//    BlockedTensor C2 = ambit::BlockedTensor::build(tensor_type_, "C2", {"gggg"});

//    BlockedTensor O3, C3;
////    bool store_H3 = (ncomm_3body_ == foptions_->get_int("DSRG_RSC_NCOMM"));
//    bool store_H3 = true;
//    if (do_t3_ and store_H3) {
//        Hbar3_.zero();
//        if (ldsrg3_ddca_) {
//            std::vector<std::string> blocks = sr_ldsrg3_ddca_blocks();
//            O3 = ambit::BlockedTensor::build(tensor_type_, "O3", blocks);
//            C3 = ambit::BlockedTensor::build(tensor_type_, "C3", blocks);
//        } else {
//            O3 = ambit::BlockedTensor::build(tensor_type_, "O3", {"gggggg"});
//            C3 = ambit::BlockedTensor::build(tensor_type_, "C3", {"gggggg"});
//        }
//    }

//    // compute Hbar recursively
//    for (int n = 1; n <= maxn; ++n) {
//        // prefactor before n-nested commutator
//        double factor = 1.0 / n;

//        if (do_t3_) {
//            timer_on("3-body [H, A]");
//            if (na_ == 0) {
//                comm_H_A_3_sr_fink(factor, O1, O2, O3, T1_, T2_, T3_, C0, C1, C2, C3);
//                if (n > ncomm_3body_) {
//                    C3.zero();
//                }
////                if (store_H3) {
////                    comm_H_A_3_sr(factor, O1, O2, O3, T1, T2, T3, C0, C1, C2, C3);
////                } else {
////                    comm_H_A_3_sr_2(factor, O1, O2, T1, T2, T3, C0, C1, C2);
////                }
//            } else {
//                comm_H_A_3(factor, O1, O2, O3, T1_, T2_, T3_, C0, C1, C2, C3);
//            }
//            timer_off("3-body [H, A]");
//        } else {
//            comm_H_A_2(factor, O1, O2, T1_, T2_, C0, C1, C2);
//        }

//        // add to Hbar
//        Hbar0_ += C0;
//        Hbar1_["pq"] += C1["pq"];
//        Hbar2_["pqrs"] += C2["pqrs"];

//        // copy C to O for next level commutator
//        O1["pq"] = C1["pq"];
//        O2["pqrs"] = C2["pqrs"];

////        if (!store_H3 and (n == 2 or n == 3)) {
////            if (n == 2 and ncomm_3body_ >= 1) {
////                comm2_l3(F, V, T1, T2, T3, C0, C1, C2);
////            }

////            if (n == 3 and ncomm_3body_ >= 2) {
////                if (ldsrg3_level_ == 3) {
////                    comm3_q3_lv3(F, V, T1, T2, T3, C0, C1, C2);
////                } else if (ldsrg3_level_ == 2) {
////                    comm3_q3_lv2(F, V, T1, T2, T3, C0, C1, C2);
////                } else {
////                    comm3_q3_lv1(F, V, T1, T2, T3, C0, C1, C2);
////                }
////            }

////            // add to Hbar
////            Hbar0 += C0;
////            Hbar1["pq"] += C1["pq"];
////            Hbar2["pqrs"] += C2["pqrs"];

////            // add C to O for next level commutator
////            O1["pq"] += C1["pq"];
////            O2["pqrs"] += C2["pqrs"];
////        }

//        // test convergence of C
//        double norm_C1 = C1.norm();
//        double norm_C2 = C2.norm();
//        double norm_C3 = 0.0;
//        if (do_t3_ and store_H3) {
//            Hbar3_["g0,g1,g2,g3,g4,g5"] += C3["g0,g1,g2,g3,g4,g5"];
//            O3["g0,g1,g2,g3,g4,g5"] = C3["g0,g1,g2,g3,g4,g5"];
//            norm_C3 = C3.norm();
//        }

//        if (std::sqrt(norm_C2 * norm_C2 + norm_C1 * norm_C1 + norm_C3 * norm_C3) < ct_threshold) {
//            break;
//        }

////        // Compute the commutator C = 1/n [O,T]
////        double C0 = 0.0;
////        C1.zero();
////        C2.zero();

////        // zero-body
////        H1_T1_C0(O1, T1, factor, C0);
////        H1_T2_C0(O1, T2, factor, C0);
////        H2_T1_C0(O2, T1, factor, C0);
////        H2_T2_C0(O2, T2, factor, C0);

////        // one-body
////        H1_T1_C1(O1, T1, factor, C1);
////        H1_T2_C1(O1, T2, factor, C1);
////        H2_T1_C1(O2, T1, factor, C1);
////        H2_T2_C1(O2, T2, factor, C1);

////        // two-body
////        H1_T2_C2(O1, T2, factor, C2);
////        H2_T1_C2(O2, T1, factor, C2);
////        H2_T2_C2(O2, T2, factor, C2);

////        //        outfile->Printf("\n   H0  = %20.12f", C0);
////        //        outfile->Printf("\n  |H1| = %20.12f", C1.norm(1));
////        //        outfile->Printf("\n  |H2| = %20.12f", C2.norm(1));
////        //        outfile->Printf("\n  --------------------------------");

////        // [H, A] = [H, T] + [H, T]^dagger
////        C0 *= 2.0;
////        O1["pq"] = C1["pq"];
////        C1["pq"] += O1["qp"];
////        O2["pqrs"] = C2["pqrs"];
////        C2["pqrs"] += O2["rspq"];

////        // Hbar += C
////        Hbar0 += C0;
////        Hbar1["pq"] += C1["pq"];
////        Hbar2["pqrs"] += C2["pqrs"];

////        // copy C to O for next level commutator
////        O1["pq"] = C1["pq"];
////        O2["pqrs"] = C2["pqrs"];

////        // test convergence of C
////        double norm_C1 = C1.norm();
////        double norm_C2 = C2.norm();
////        //        outfile->Printf("\n  %2d %20.12f %20e %20e",n,C0,norm_C1,norm_C2);
////        if (std::sqrt(norm_C2 * norm_C2 + norm_C1 * norm_C1) < ct_threshold) {
////            break;
////        }
//    }

//    //    outfile->Printf("\n
//    //    -----------------------------------------------------------------");
//}

// void CC_SO::compute_qhbar() {

//    //    outfile->Printf("\n\n  Computing the similarity-transformed
//    //    Hamiltonian");
//    //    outfile->Printf("\n
//    //    -----------------------------------------------------------------");
//    //    outfile->Printf("\n  nComm           C0                 |C1| |C2|" );
//    //    outfile->Printf("\n
//    //    -----------------------------------------------------------------");

//    // copy initial one-body Hamiltonian
//    Hbar0_ = 0.0;
//    Hbar1_["pq"] = F["pq"];
//    Hbar2_["pqrs"] = V["pqrs"];

//    BlockedTensor O1 = ambit::BlockedTensor::build(tensor_type_, "O1", {"gg"});
//    BlockedTensor O2 = ambit::BlockedTensor::build(tensor_type_, "O2", {"gggg"});
//    BlockedTensor O3 = ambit::BlockedTensor::build(tensor_type_, "O3", {"gggggg"});
//    O1["pq"] = F["pq"];
//    O2["pqrs"] = V["pqrs"];

//    //    outfile->Printf("\n  %2d %20.12f %20e
//    //    %20e",0,Hbar0,Hbar1.norm(),Hbar2.norm());

//    // iterator variables
//    int maxn = foptions_->get_int("DSRG_RSC_NCOMM");
//    double ct_threshold = foptions_->get_double("DSRG_RSC_THRESHOLD");
//    BlockedTensor C1 = ambit::BlockedTensor::build(tensor_type_, "C1", {"gg"});
//    BlockedTensor C2 = ambit::BlockedTensor::build(tensor_type_, "C2", {"gggg"});
//    BlockedTensor C3 = ambit::BlockedTensor::build(tensor_type_, "C3", {"gggggg"});

//    // compute Hbar recursively
//    for (int n = 1; n <= maxn; ++n) {
//        // prefactor before n-nested commutator
//        double factor = 1.0 / n;

//        // Compute the commutator C = 1/n [O,T]
//        double C0 = 0.0;
//        C1.zero();
//        C2.zero();

//        // zero-body
//        H1_T1_C0(O1, T1_, factor, C0);
//        H1_T2_C0(O1, T2_, factor, C0);
//        H2_T1_C0(O2, T1_, factor, C0);
//        H2_T2_C0(O2, T2_, factor, C0);

//        // one-body
//        H1_T1_C1(O1, T1_, factor, C1);
//        H1_T2_C1(O1, T2_, factor, C1);
//        H2_T1_C1(O2, T1_, factor, C1);
//        H2_T2_C1(O2, T2_, factor, C1);

//        // two-body
//        H1_T2_C2(O1, T2_, factor, C2);
//        H2_T1_C2(O2, T1_, factor, C2);
//        H2_T2_C2(O2, T2_, factor, C2);

//        // three-body if odd
//        if (n % 2 == 1) {
//            C3.zero();
//            H2_T2_C3(O2, T2_, factor, C3);
//            O3["pqrsto"] = C3["pqrsto"];
//            O3["pqrsto"] += C3["stopqr"];
//        }

//        // compute three-body contrinution if even
//        if (n % 2 == 0) {
//            H3_T1_C1(O3, T1_, factor, C1);
//            H3_T1_C2(O3, T1_, factor, C2);
//            H3_T2_C1(O3, T2_, factor, C1);
//            H3_T2_C2(O3, T2_, factor, C2);
//        }

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
//        Hbar0_ += C0;
//        Hbar1_["pq"] += C1["pq"];
//        Hbar2_["pqrs"] += C2["pqrs"];

//        // copy C to O for next level commutator
//        O1["pq"] = C1["pq"];
//        O2["pqrs"] = C2["pqrs"];

//        // test convergence of C
//        double norm_C1 = C1.norm();
//        double norm_C2 = C2.norm();
//        //        outfile->Printf("\n  %2d %20.12f %20e
//        //        %20e",n,C0,norm_C1,norm_C2);
//        if (std::sqrt(norm_C2 * norm_C2 + norm_C1 * norm_C1) < ct_threshold) {
//            break;
//        }
//    }
//    //    outfile->Printf("\n
//    //    -----------------------------------------------------------------");
//}
} // namespace forte
