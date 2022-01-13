/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2022 by its authors (see COPYING, COPYING.LESSER, AUTHORS).
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

#ifdef HAVE_CHEMPS2

#include <cstdlib>
#include <sys/stat.h>
#include <iostream>
#include <fstream>
#include <filesystem>

#include "psi4/libdpd/dpd.h"
#include "psi4/libmints/factory.h"
#include "psi4/libmints/molecule.h"
#include "psi4/libmints/pointgrp.h"
#include "psi4/libmints/mintshelper.h"
#include "psi4/libmints/typedefs.h"
#include "psi4/libmints/basisset.h"
#include "psi4/libpsio/psio.hpp"

#include "base_classes/mo_space_info.h"
#include "integrals/active_space_integrals.h"
#include "helpers/helpers.h"
#include "helpers/printing.h"
#include "helpers/timer.h"

#include "dmrgsolver.h"

namespace fs = std::filesystem;
using namespace psi;

namespace forte {

DMRGSolver::DMRGSolver(StateInfo state, size_t nroot, std::shared_ptr<SCFInfo> scf_info,
                       std::shared_ptr<ForteOptions> options,
                       std::shared_ptr<MOSpaceInfo> mo_space_info,
                       std::shared_ptr<ActiveSpaceIntegrals> as_ints)
    : ActiveSpaceMethod(state, nroot, mo_space_info, as_ints), scf_info_(scf_info),
      options_(options) {
    print_method_banner({"Density Matrix Renormalization Group", "Sebastian Wouters"});
    startup();
}

void DMRGSolver::startup() {
    nirrep_ = static_cast<int>(mo_space_info_->nirrep());
    multiplicity_ = state_.multiplicity();
    wfn_irrep_ = state_.irrep();

    nactv_ = static_cast<int>(mo_space_info_->size("ACTIVE"));
    auto nelecs = state_.na() + state_.nb();
    nelecs_actv_ = static_cast<int>(nelecs - 2 * mo_space_info_->size("INACTIVE_DOCC"));

    pg_number_ = 0; // C1 symmetry by default
    auto pg = mo_space_info_->point_group_label();
    std::transform(pg.begin(), pg.end(), pg.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    for (int i = 0; i < 8; ++i) {
        if (pg == CheMPS2::Irreps::getGroupName(i)) {
            pg_number_ = i;
            break;
        }
    }

    std::stringstream ss;
    ss << multiplicity_ << state_.irrep_label() << "_CAS" << nactv_;
    state_label_ = ss.str();
}

void DMRGSolver::set_options(std::shared_ptr<ForteOptions> options) {
    read_wfn_guess_ = options_->get_bool("READ_ACTIVE_WFN_GUESS");
    dump_wfn_ = options_->get_bool("DUMP_ACTIVE_WFN");

    dmrg_sweep_states_ = options->get_int_list("DMRG_SWEEP_STATES");
    dmrg_sweep_e_convergence_ = options->get_double_list("DMRG_SWEEP_ENERGY_CONV");
    dmrg_sweep_max_sweeps_ = options->get_int_list("DMRG_SWEEP_MAX_SWEEPS");
    dmrg_noise_prefactors_ = options->get_double_list("DMRG_SWEEP_NOISE_PREFAC");
    dmrg_davidson_rtol_ = options->get_double_list("DMRG_SWEEP_DVDSON_RTOL");
    dmrg_print_corr_ = options->get_bool("DMRG_PRINT_CORR");

    // sanity check
    auto nstates = dmrg_sweep_states_.size();
    if (nstates == 0) {
        throw std::runtime_error("Please set option DMRG_SWEEP_STATES (integer array)!");
    }
    std::map<std::string, size_t> sweep_sizes;
    sweep_sizes["DMRG_SWEEP_ENERGY_CONV (double array)"] = dmrg_sweep_e_convergence_.size();
    sweep_sizes["DMRG_SWEEP_MAX_SWEEPS (integer array)"] = dmrg_sweep_max_sweeps_.size();
    sweep_sizes["DMRG_SWEEP_NOISE_PREFAC (double array)"] = dmrg_noise_prefactors_.size();
    sweep_sizes["DMRG_SWEEP_DVDSON_RTOL (double array)"] = dmrg_davidson_rtol_.size();
    for (const auto& pair : sweep_sizes) {
        if (pair.second == 0) {
            throw std::runtime_error("Please set option " + pair.first + "!");
        }
        if (pair.second != nstates) {
            std::stringstream ss;
            ss << "Inconsistent number of elements in " << pair.first << " and DMRG_SWEEP_STATES";
            throw std::runtime_error(ss.str());
        }
    }
    for (size_t cnt = 0; cnt < nstates; cnt++) {
        if (dmrg_sweep_states_[cnt] < 2) {
            throw std::runtime_error("DMRG_SWEEP_STATES entries should be >= 1!");
        }
        if (dmrg_sweep_e_convergence_[cnt] <= 0.0) {
            throw std::runtime_error("DMRG_SWEEP_ENERGY_CONV entries should be positive!");
        }
        if (dmrg_sweep_max_sweeps_[cnt] < 1) {
            throw std::runtime_error("DMRG_SWEEP_MAX_SWEEPS entries should be positive!");
        }
        if (dmrg_davidson_rtol_[cnt] <= 0.0) {
            throw std::runtime_error("DMRG_SWEEP_DVDSON_RTOL entries should be positive!");
        }
    }
}

double DMRGSolver::compute_energy() {
    std::ofstream capturing;
    std::streambuf* cout_buffer;
    std::string chemps2filename = "output.chemps2.tmp";
    capturing.open(chemps2filename.c_str(), std::ios::trunc); // truncate
    cout_buffer = std::cout.rdbuf(capturing.rdbuf());

    auto n_sweep_states = static_cast<int>(dmrg_sweep_states_.size());

    // Create a CheMPS2::ConvergenceSchem
    CheMPS2::Initialize::Init();
    auto conv_scheme = std::make_unique<CheMPS2::ConvergenceScheme>(n_sweep_states);
    for (int cnt = 0; cnt < n_sweep_states; cnt++) {
        conv_scheme->set_instruction(cnt, dmrg_sweep_states_[cnt], dmrg_sweep_e_convergence_[cnt],
                                     dmrg_sweep_max_sweeps_[cnt], dmrg_noise_prefactors_[cnt],
                                     dmrg_davidson_rtol_[cnt]);
    }

    // Create the CheMPS2::Hamiltonian --> fill later
    auto actv_irreps = mo_space_info_->symmetry("ACTIVE");
    auto Hamiltonian =
        std::make_unique<CheMPS2::Hamiltonian>(nactv_, pg_number_, actv_irreps.data());

    // Create the CheMPS2::Problem
    auto prob = std::make_unique<CheMPS2::Problem>(Hamiltonian.get(), multiplicity_ - 1,
                                                   nelecs_actv_, wfn_irrep_);
    if (not prob->checkConsistency()) {
        throw std::runtime_error("No Hilbert state vector compatible with all symmetry sectors!");
    }
    prob->SetupReorderD2h();

    // Fill Hamiltonian
    Hamiltonian->setEconst(as_ints_->scalar_energy() + as_ints_->nuclear_repulsion_energy());

    for (int p = 0; p < nactv_; ++p) {
        for (int q = 0; q < nactv_; ++q) {
            if (actv_irreps[p] == actv_irreps[q]) {
                Hamiltonian->setTmat(p, q, as_ints_->oei_a(p, q));
            }
            for (int r = 0; r < nactv_; ++r) {
                for (int s = 0; s < nactv_; ++s) {
                    if ((actv_irreps[p] ^ actv_irreps[q]) == (actv_irreps[r] ^ actv_irreps[s])) {
                        Hamiltonian->setVmat(p, q, r, s, as_ints_->tei_ab(p, q, r, s));
                    }
                }
            }
        }
    }

    // Save CheMPS2 default MPS file names
    mps_files_.clear();
    for (size_t root = 0; root < nroot_; ++root) {
        std::stringstream ss;
        ss << CheMPS2::DMRG_MPS_storage_prefix << root << ".h5"; // default name by CheMPS2
        mps_files_.emplace_back(ss.str());
    }

    // Path for MPS files
    auto cwd = fs::current_path();
    auto dump_path = cwd / fs::path(state_label_);
    if (dump_wfn_) {
        if (not fs::exists(dump_path))
            fs::create_directory(dump_path);
        std::cout << "MPS files will be dumped to " << dump_path << '\n';
    }
    if (read_wfn_guess_) {
        if (fs::exists(dump_path)) {
            for (const std::string& name : mps_files_) {
                auto pf = fs::path(name);
                if (fs::exists(dump_path / pf))
                    fs::copy_file(dump_path / pf, cwd / pf);
            }
        }
    }

    // Start DMRG sweeps
    const std::string tmp_path = PSIOManager::shared_object()->get_default_path();
    solver_ = std::make_unique<CheMPS2::DMRG>(prob.get(), conv_scheme.get(), dump_wfn_ or read_wfn_guess_, tmp_path);

    energies_.clear();
    opdms_.clear();
    tpdms_.clear();

    for (size_t root = 0; root < nroot_; ++root) {
        if (root > 0)
            solver_->newExcitation(std::fabs(energies_[root - 1]));
        std::cout << "\n  ==> Computing Energy for " << state_.str_minimum() << " <==\n\n";
        energies_.push_back(solver_->Solve());
        if (dump_wfn_) {
            auto pf = fs::path(mps_files_[root]);
            if (fs::exists(pf)) {
                fs::rename(pf, dump_path / pf);
            }
        }
        solver_->calc_rdms_and_correlations(false, false); // compute 2RDM by default
        push_back_rdms();
        if (dmrg_print_corr_) {
            solver_->getCorrelations()->Print();
        }
        if (root == 0 and nroot_ > 1) {
            solver_->activateExcitations(static_cast<int>(nroot_ - 1));
        }
    }

    // clean up copied MPS files
    if (read_wfn_guess_) {
        for (const std::string& name : mps_files_) {
            auto pf = fs::path(name);
            if (fs::exists(pf))
                fs::remove(pf);
        }
    }

    std::cout.rdbuf(cout_buffer);
    capturing.close();
    std::ifstream copying;
    copying.open(chemps2filename, std::ios::in); // read only
    if (copying.is_open()) {
        std::string line;
        while (getline(copying, line)) {
            (*outfile->stream()) << line << std::endl;
        }
        copying.close();
    }
    system(("rm " + chemps2filename).c_str());

    // Push to psi4 environment
    double energy = energies_[root_];
    psi::Process::environment.globals["CURRENT ENERGY"] = energy;
    psi::Process::environment.globals["DMRG ENERGY"] = energy;
    return energy;
}

std::vector<RDMs> DMRGSolver::rdms(const std::vector<std::pair<size_t, size_t>>& root_list,
                                   int max_rdm_level) {
    if (solver_ == nullptr) {
        throw std::runtime_error("Please first run DSRGSolver::compute_energy!");
    }

    // test root list
    std::unordered_set<size_t> roots;
    for (const auto& pair : root_list) {
        auto root1 = pair.first;
        if (root1 != pair.second) {
            throw std::runtime_error(
                "Different roots for bra and ket! Transition RDMs not available in DMRG!");
        }
        if (root1 >= nroot_) {
            throw std::runtime_error("Root number out of range!");
        }
        roots.insert(root1);
    }

    std::vector<RDMs> rdms;
    bool do_3rdm = max_rdm_level > 2;
    bool disk_3rdm = mo_space_info_->size("ACTIVE") >= 30;

    std::ofstream capturing;
    std::streambuf* cout_buffer;
    std::string chemps2filename = "output.chemps2.rdms.tmp";
    capturing.open(chemps2filename.c_str(), std::ios::trunc); // truncate
    cout_buffer = std::cout.rdbuf(capturing.rdbuf());

    for (size_t root = 0; root < nroot_; ++root) {
        if (root > 0)
            solver_->newExcitation(std::fabs(energies_[root - 1]));
        //        solver_->Solve(); // TODO: need to test excited states
        if (roots.find(root) != roots.end()) {
            solver_->calc_rdms_and_correlations(do_3rdm, disk_3rdm);
            rdms.push_back(fill_current_rdms(do_3rdm));
        }
        if (root == 0 and nroot_ > 1) {
            solver_->activateExcitations(static_cast<int>(nroot_ - 1));
        }
    }

    std::cout.rdbuf(cout_buffer);
    capturing.close();
    std::ifstream copying;
    copying.open(chemps2filename, std::ios::in); // read only
    if (copying.is_open()) {
        std::string line;
        while (getline(copying, line)) {
            (*outfile->stream()) << line << std::endl;
        }
        copying.close();
    }
    system(("rm " + chemps2filename).c_str());

    return rdms;
}

void DMRGSolver::push_back_rdms() {
    std::vector<size_t> dim2(2, nactv_);
    std::vector<size_t> dim4(4, nactv_);
    auto g1 = ambit::Tensor::build(ambit::CoreTensor, "DMRG G1", dim2);
    auto g2 = ambit::Tensor::build(ambit::CoreTensor, "DMRG G2", dim4);
    CheMPS2::CASSCF::copy2DMover(solver_->get2DM(), nactv_, g2.data().data());
    CheMPS2::CASSCF::setDMRG1DM(nelecs_actv_, nactv_, g1.data().data(), g2.data().data());
    opdms_.push_back(g1);
    tpdms_.push_back(g2);
}

RDMs DMRGSolver::fill_current_rdms(const bool do_3rdm) {
    std::vector<size_t> dim2(2, nactv_);
    std::vector<size_t> dim4(4, nactv_);
    auto g1 = ambit::Tensor::build(ambit::CoreTensor, "DMRG G1", dim2);
    auto g2 = ambit::Tensor::build(ambit::CoreTensor, "DMRG G2", dim4);

    CheMPS2::CASSCF::copy2DMover(solver_->get2DM(), nactv_, g2.data().data());
    CheMPS2::CASSCF::setDMRG1DM(nelecs_actv_, nactv_, g1.data().data(), g2.data().data());

    g1.scale(0.5);
    auto g2ab = g2.clone();
    g2ab("pqrs") += g2("pqrs") - g2("pqsr");
    g2ab.scale(1.0 / 6.0);

    if (do_3rdm) {
        std::vector<size_t> dim6(6, nactv_);
        auto g3 = ambit::Tensor::build(ambit::CoreTensor, "DMRG G3", dim6);
        solver_->get3DM()->fill_ham_index(1.0, false, g3.data().data(), 0, nactv_);

        auto g3aab = g3.clone();
        g3aab("pqrstu") -= g3("pqrtus") + g3("pqrust") + 2.0 * g3("pqrtsu");
        g3aab.scale(1.0 / 12.0);
        return {true, g1, g2ab, g3aab};
    }

    return {true, g1, g2ab};
}

std::vector<RDMs> DMRGSolver::transition_rdms(const std::vector<std::pair<size_t, size_t>>&,
                                              std::shared_ptr<ActiveSpaceMethod>, int) {
    throw std::runtime_error("DMRGSolver::transition_rdms is not available in CheMPS2!");
}

// void DMRGSolver::compute_reference(double* one_rdm, double* two_rdm, double* three_rdm,
//                                    CheMPS2::DMRGSCFindices* iHandler) {
//     // if(options_->get_int("MULTIPLICITY") != 1 &&
//     // options_->get_int("DMRG_WFN_MULTP") != 1)
//     //{
//     //    outfile->Printf("\n\n Spinadapted formalism requires spin-averaged
//     //    quantitities");
//     //    throw psi::PSIEXCEPTION("You need to spin averaged things");
//     //}
//     RDMs dmrg_ref;
//     size_t na = mo_space_info_->size("ACTIVE");
//     ambit::Tensor gamma1_a = ambit::Tensor::build(ambit::CoreTensor, "gamma1_a", {na, na});
//     ambit::Tensor gamma2_dmrg =
//         ambit::Tensor::build(ambit::CoreTensor, "Gamma2_DMRG", {na, na, na, na});
//     ambit::Tensor gamma2_aa =
//         ambit::Tensor::build(ambit::CoreTensor, "gamma2_aa", {na, na, na, na});
//     ambit::Tensor gamma2_ab =
//         ambit::Tensor::build(ambit::CoreTensor, "gamma2_ab", {na, na, na, na});
//
//     const int nOrbDMRG = iHandler->getDMRGcumulative(iHandler->getNirreps());
//     std::vector<double>& gamma1_data = gamma1_a.data();
//     for (int irrep = 0; irrep < iHandler->getNirreps(); irrep++) {
//         const int shift = iHandler->getDMRGcumulative(irrep);
//         for (int orb1 = 0; orb1 < iHandler->getNDMRG(irrep); orb1++) {
//             for (int orb2 = orb1; orb2 < iHandler->getNDMRG(irrep); orb2++) {
//                 const double value = one_rdm[shift + orb1 + nOrbDMRG * (shift + orb2)];
//                 gamma1_data[shift + orb1 + nOrbDMRG * (shift + orb2)] = 0.5 * value;
//                 gamma1_data[shift + orb2 + nOrbDMRG * (shift + orb1)] = 0.5 * value;
//             }
//         }
//     }
//     /// Gamma_a = 1_RDM / 2
//     /// Gamma_b = 1_RDM / 2
//     dmrg_ref.set_L1a(gamma1_a);
//     dmrg_ref.set_L1b(gamma1_a);
//     /// Form 2_rdms
//     {
//         gamma2_dmrg.iterate([&](const std::vector<size_t>& i, double& value) {
//             value = two_rdm[i[0] * na * na * na + i[1] * na * na + i[2] * na + i[3]];
//         });
//         if (spin_free_rdm_) {
//             dmrg_ref.set_SFg2(gamma2_dmrg);
//         }
//         /// gamma2_aa = 1 / 6 * (Gamma2(pqrs) - Gamma2(pqsr))
//         // gamma2_aa.copy(gamma2_dmrg);
//         gamma2_aa("p, q, r, s") = gamma2_dmrg("p, q, r, s") - gamma2_dmrg("p, q, s, r");
//         gamma2_aa.scale(1.0 / 6.0);
//
//         gamma2_ab("p, q, r, s") = (2.0 * gamma2_dmrg("p, q, r, s") + gamma2_dmrg("p, q, s, r"));
//         gamma2_ab.scale(1.0 / 6.0);
//         dmrg_ref.set_g2aa(gamma2_aa);
//         dmrg_ref.set_g2bb(gamma2_aa);
//         dmrg_ref.set_g2ab(gamma2_ab);
//         ambit::Tensor cumulant2_aa =
//             ambit::Tensor::build(ambit::CoreTensor, "Cumulant2_aa", {na, na, na, na});
//         ambit::Tensor cumulant2_ab =
//             ambit::Tensor::build(ambit::CoreTensor, "Cumulant2_ab", {na, na, na, na});
//         cumulant2_aa.copy(gamma2_aa);
//         cumulant2_aa("pqrs") -= gamma1_a("pr") * gamma1_a("qs");
//         cumulant2_aa("pqrs") += gamma1_a("ps") * gamma1_a("qr");
//
//         cumulant2_ab.copy(gamma2_ab);
//         cumulant2_ab("pqrs") -= gamma1_a("pr") * gamma1_a("qs");
//         dmrg_ref.set_L2aa(cumulant2_aa);
//         dmrg_ref.set_L2ab(cumulant2_ab);
//         dmrg_ref.set_L2bb(cumulant2_aa);
//     }
//     // if((options_->get_str("THREEPDC") != "ZERO") &&
//     // (options_->get_str("JOB_TYPE") == "DSRG-MRPT2" or
//     // options_->get_str("JOB_TYPE") == "THREE-DSRG-MRPT2"))
//     if (max_rdm_ > 2 && !disk_3_rdm_) {
//         ambit::Tensor gamma3_dmrg =
//             ambit::Tensor::build(ambit::CoreTensor, "Gamma3_DMRG", {na, na, na, na, na, na});
//         ambit::Tensor gamma3_aaa =
//             ambit::Tensor::build(ambit::CoreTensor, "Gamma3_aaa", {na, na, na, na, na, na});
//         ambit::Tensor gamma3_aab =
//             ambit::Tensor::build(ambit::CoreTensor, "Gamma3_aab", {na, na, na, na, na, na});
//         ambit::Tensor gamma3_abb =
//             ambit::Tensor::build(ambit::CoreTensor, "Gamma2_abb", {na, na, na, na, na, na});
//         gamma3_dmrg.iterate([&](const std::vector<size_t>& i, double& value) {
//             value = three_rdm[i[0] * na * na * na * na * na + i[1] * na * na * na * na +
//                               i[2] * na * na * na + i[3] * na * na + i[4] * na + i[5]];
//         });
//         gamma3_aaa("p, q, r, s, t, u") = gamma3_dmrg("p, q, r, s, t, u") +
//                                          gamma3_dmrg("p, q, r, t, u, s") +
//                                          gamma3_dmrg("p, q, r, u, s, t");
//         gamma3_aaa.scale(1.0 / 12.0);
//         gamma3_aab("p, q, r, s, t, u") =
//             (gamma3_dmrg("p, q, r, s, t, u") - gamma3_dmrg("p, q, r, t, u, s") -
//              gamma3_dmrg("p, q, r, u, s, t") - 2.0 * gamma3_dmrg("p, q, r, t, s, u"));
//         gamma3_aab.scale(1.0 / 12.0);
//         // gamma3_abb("p, q, r, s, t, u") = (-gamma3_dmrg("p, q, r, s, t, u") -
//         // gamma3_dmrg("p, q, r, t, u, s") - gamma3_dmrg("p, q, r, u, s, t") -
//         // 2.0 * gamma3_dmrg("p, q, r, s, u, t"));
//         // gamma3_abb.scale(1.0 / 12.0);
//         ambit::Tensor L1a = dmrg_ref.g1a();
//         ambit::Tensor L1b = dmrg_ref.g1b();
//         ambit::Tensor L2aa = dmrg_ref.L2aa();
//         ambit::Tensor L2ab = dmrg_ref.L2ab();
//         // ambit::Tensor L2bb = dmrg_ref.L2bb();
//         // Convert the 3-RDMs to 3-RCMs
//         gamma3_aaa("pqrstu") -= L1a("ps") * L2aa("qrtu");
//         gamma3_aaa("pqrstu") += L1a("pt") * L2aa("qrsu");
//         gamma3_aaa("pqrstu") += L1a("pu") * L2aa("qrts");
//
//         gamma3_aaa("pqrstu") -= L1a("qt") * L2aa("prsu");
//         gamma3_aaa("pqrstu") += L1a("qs") * L2aa("prtu");
//         gamma3_aaa("pqrstu") += L1a("qu") * L2aa("prst");
//
//         gamma3_aaa("pqrstu") -= L1a("ru") * L2aa("pqst");
//         gamma3_aaa("pqrstu") += L1a("rs") * L2aa("pqut");
//         gamma3_aaa("pqrstu") += L1a("rt") * L2aa("pqsu");
//
//         gamma3_aaa("pqrstu") -= L1a("ps") * L1a("qt") * L1a("ru");
//         gamma3_aaa("pqrstu") -= L1a("pt") * L1a("qu") * L1a("rs");
//         gamma3_aaa("pqrstu") -= L1a("pu") * L1a("qs") * L1a("rt");
//
//         gamma3_aaa("pqrstu") += L1a("ps") * L1a("qu") * L1a("rt");
//         gamma3_aaa("pqrstu") += L1a("pu") * L1a("qt") * L1a("rs");
//         gamma3_aaa("pqrstu") += L1a("pt") * L1a("qs") * L1a("ru");
//
//         gamma3_aab("pqRstU") -= L1a("ps") * L2ab("qRtU");
//         gamma3_aab("pqRstU") += L1a("pt") * L2ab("qRsU");
//
//         gamma3_aab("pqRstU") -= L1a("qt") * L2ab("pRsU");
//         gamma3_aab("pqRstU") += L1a("qs") * L2ab("pRtU");
//
//         gamma3_aab("pqRstU") -= L1b("RU") * L2aa("pqst");
//
//         gamma3_aab("pqRstU") -= L1a("ps") * L1a("qt") * L1b("RU");
//         gamma3_aab("pqRstU") += L1a("pt") * L1a("qs") * L1b("RU");
//
//         /// KPH found notes from York's paper useful in deriving this
//         /// relationship
//         gamma3_abb("p, q, r, s, t, u") = gamma3_aab("q,r,p,t,u,s");
//
//         dmrg_ref.set_L3aaa(gamma3_aaa);
//         dmrg_ref.set_L3aab(gamma3_aab);
//         dmrg_ref.set_L3abb(gamma3_abb);
//         dmrg_ref.set_L3bbb(gamma3_aaa);
//     }
//     dmrg_ref_ = dmrg_ref;
// }
// void DMRGSolver::compute_energy() {
//     const int wfn_irrep = options_->get_int("ROOT_SYM");
//     const int wfn_multp = options_->get_int("MULTIPLICITY");
//     std::vector<int> dmrg_states = options_->get_int_list("DMRG_STATES");
//     const int ndmrg_states = dmrg_states.size();
//     std::vector<double> dmrg_econv = options_->get_double_list("DMRG_ECONV");
//     const int ndmrg_econv = dmrg_econv.size();
//     std::vector<int> dmrg_maxsweeps = options_->get_int_list("DMRG_MAXSWEEPS");
//     const int ndmrg_maxsweeps = dmrg_maxsweeps.size();
//     std::vector<double> dmrg_noiseprefactors = options_->get_double_list("DMRG_NOISEPREFACTORS");
//     const int ndmrg_noiseprefactors = dmrg_noiseprefactors.size();
//     const bool dmrg_print_corr = options_->get_bool("DMRG_PRINT_CORR");
//     psi::Dimension frozen_docc = mo_space_info_->dimension("INACTIVE_DOCC");
//     psi::Dimension active = mo_space_info_->dimension("ACTIVE");
//     psi::Dimension virtual_orbs = mo_space_info_->dimension("RESTRICTED_UOCC");
//     const double dmrgscf_convergence = options_->get_double("D_CONVERGENCE");
//     const bool dmrgscf_store_unit = options_->get_bool("DMRG_STORE_UNIT");
//     const bool dmrgscf_do_diis = options_->get_bool("DMRG_DO_DIIS");
//     const double dmrgscf_diis_branch = options_->get_double("DMRG_DIIS_BRANCH");
//     const bool dmrgscf_store_diis = options_->get_bool("DMRG_STORE_DIIS");
//     const int dmrgscf_max_iter = options_->get_int("DMRGSCF_MAX_ITER");
//     const int dmrgscf_which_root = options_->get_int("DMRG_WHICH_ROOT");
//     const bool dmrgscf_state_avg = options_->get_bool("DMRG_AVG_STATES");
//     const string dmrgscf_active_space = options_->get_str("DMRG_ACTIVE_SPACE");
//     const bool dmrgscf_loc_random = options_->get_bool("DMRG_LOC_RANDOM");
//     std::vector<double> dmrg_davidson_tol = options_->get_double_list("DMRG_DAVIDSON_RTOL");
//     const int ndmrg_davidson_tol = dmrg_davidson_tol.size();
//     const int dmrgscf_num_vec_diis = CheMPS2::DMRGSCF_numDIISvecs;
//     const std::string unitaryname =
//         psi::get_writer_file_prefix(ints_->wfn()->name()) + ".unitary.h5";
//     const std::string diisname = psi::get_writer_file_prefix(ints_->wfn()->name()) + ".DIIS.h5";
//     bool three_pdm = false;
//     if (options_->get_str("JOB_TYPE") == "DSRG-MRPT2" or
//         options_->get_str("JOB_TYPE") == "THREE-DSRG-MRPT2") {
//         if (options_->get_str("THREEPDC") != "ZERO")
//             three_pdm = true;
//     }
//
//     /*   Check if the input is consistent   */
//
//     const int SyGroup = chemps2_groupnumber(ints_->wfn()->molecule()->sym_label());
//     const int nmo = mo_space_info_->size("ALL");
//     const int nirrep = mo_space_info_->nirrep();
//     psi::Dimension orbspi = mo_space_info_->dimension("ALL");
//
//     if (wfn_irrep < 0) {
//         throw psi::PSIEXCEPTION("Option ROOT_SYM (integer) may not be smaller than zero!");
//     }
//     if (wfn_multp < 1) {
//         throw psi::PSIEXCEPTION("Option MULTIPLICTY (integer) should be larger or "
//                                 "equal to one: WFN_MULTP = (2S+1) >= 1 !");
//     }
//     if (ndmrg_states == 0) {
//         throw psi::PSIEXCEPTION("Option DMRG_STATES (integer array) should be set!");
//     }
//     if (ndmrg_econv == 0) {
//         throw psi::PSIEXCEPTION("Option DMRG_ECONV (double array) should be set!");
//     }
//     if (ndmrg_maxsweeps == 0) {
//         throw psi::PSIEXCEPTION("Option DMRG_MAXSWEEPS (integer array) should be set!");
//     }
//     if (ndmrg_noiseprefactors == 0) {
//         throw psi::PSIEXCEPTION("Option DMRG_NOISEPREFACTORS (double array) should be set!");
//     }
//     if (ndmrg_states != ndmrg_econv) {
//         throw psi::PSIEXCEPTION("Options DMRG_STATES (integer array) and DMRG_ECONV "
//                                 "(double array) should contain the same number of "
//                                 "elements!");
//     }
//     if (ndmrg_states != ndmrg_maxsweeps) {
//         throw psi::PSIEXCEPTION("Options DMRG_STATES (integer array) and "
//                                 "DMRG_MAXSWEEPS (integer array) should contain the "
//                                 "same number of elements!");
//     }
//     if (ndmrg_states != ndmrg_noiseprefactors) {
//         throw psi::PSIEXCEPTION("Options DMRG_STATES (integer array) and "
//                                 "DMRG_NOISEPREFACTORS (double array) should contain "
//                                 "the same number of elements!");
//     }
//
//     for (int cnt = 0; cnt < ndmrg_states; cnt++) {
//         if (dmrg_states[cnt] < 2) {
//             throw psi::PSIEXCEPTION("Entries in DMRG_STATES (integer array) should "
//                                     "be larger than 1!");
//         }
//     }
//     if (dmrgscf_convergence <= 0.0) {
//         throw psi::PSIEXCEPTION("Option D_CONVERGENCE (double) must be larger than zero!");
//     }
//     if (dmrgscf_diis_branch <= 0.0) {
//         throw psi::PSIEXCEPTION("Option DMRG_DIIS_BRANCH (double) must be larger than zero!");
//     }
//     if (dmrgscf_max_iter < 1) {
//         throw psi::PSIEXCEPTION("Option DMRG_MAX_ITER (integer) must be larger than zero!");
//     }
//     if (dmrgscf_which_root < 1) {
//         throw psi::PSIEXCEPTION("Option DMRG_WHICH_ROOT (integer) must be larger than zero!");
//     }
//     /*   Create a CheMPS2::ConvergenceScheme   */
//     CheMPS2::Initialize::Init();
//     std::shared_ptr<CheMPS2::ConvergenceScheme> OptScheme =
//         std::make_shared<CheMPS2::ConvergenceScheme>(ndmrg_states);
//     for (int cnt = 0; cnt < ndmrg_states; cnt++) {
//         if (ndmrg_davidson_tol != ndmrg_states)
//             OptScheme->setInstruction(cnt, dmrg_states[cnt], dmrg_econv[cnt],
//             dmrg_maxsweeps[cnt],
//                                       dmrg_noiseprefactors[cnt]);
//         else {
//             OptScheme->set_instruction(cnt, dmrg_states[cnt], dmrg_econv[cnt],
//             dmrg_maxsweeps[cnt],
//                                        dmrg_noiseprefactors[cnt], dmrg_davidson_tol[cnt]);
//         }
//     }
//     // CheMPS2::DMRGSCFindices * iHandler = new ChemP
//     std::shared_ptr<CheMPS2::DMRGSCFindices> iHandler =
//         std::make_shared<CheMPS2::DMRGSCFindices>(nmo, SyGroup, frozen_docc, active,
//         virtual_orbs);
//     int nElectrons = state_.na() + state_.nb();
//     outfile->Printf("\n  nElectrons  = %d", nElectrons);
//
//     // Number of electrons in the active space
//     int nDMRGelectrons = nElectrons;
//     for (int cnt = 0; cnt < nirrep; cnt++) {
//         nDMRGelectrons -= 2 * frozen_docc[cnt];
//     }
//     outfile->Printf("\n  El. active = %d", nDMRGelectrons);
//
//     // Create the CheMPS2::Hamiltonian --> fill later
//     const size_t nOrbDMRG = mo_space_info_->size("ACTIVE");
//     int* orbitalIrreps = new int[nOrbDMRG];
//     int counterFillOrbitalIrreps = 0;
//     for (int h = 0; h < nirrep; h++) {
//         for (int cnt = 0; cnt < active[h];
//              cnt++) { // Only the active space is treated with DMRG-SCF!
//
//             orbitalIrreps[counterFillOrbitalIrreps] = h;
//             counterFillOrbitalIrreps++;
//         }
//     }
//     std::shared_ptr<CheMPS2::Hamiltonian> Ham =
//         std::make_shared<CheMPS2::Hamiltonian>(nOrbDMRG, SyGroup, orbitalIrreps);
//     // fill_integrals(Ham);
//
//     std::shared_ptr<CheMPS2::Problem> Prob =
//         std::make_shared<CheMPS2::Problem>(Ham.get(), wfn_multp - 1, nDMRGelectrons, wfn_irrep);
//
//     if (!(Prob->checkConsistency())) {
//         throw psi::PSIEXCEPTION("CheMPS2::Problem : No Hilbert state vector "
//                                 "compatible with all symmetry sectors!");
//     }
//     Prob->SetupReorderD2h();
//
//     /// If one does not provide integrals when they call solver, compute them
//     /// yourself
//     if (!use_user_integrals_) {
//         std::vector<size_t> active_array = mo_space_info_->corr_absolute_mo("ACTIVE");
//         active_integrals_ =
//             ints_->aptei_ab_block(active_array, active_array, active_array, active_array);
//         /// SCF_TYPE CD tends to be slow.  Avoid it and use integral class
//         if (options_->get_str("SCF_TYPE") != "CD") {
//             local_timer one_body_timer;
//             one_body_integrals_ = one_body_operator();
//             outfile->Printf("\n OneBody integrals (though one_body_operator) takes %6.5f s",
//                             one_body_timer.get());
//         } else {
//             local_timer one_body_fci_ints;
//             std::shared_ptr<ActiveSpaceIntegrals> fci_ints =
//             std::make_shared<ActiveSpaceIntegrals>(
//                 ints_, mo_space_info_->corr_absolute_mo("ACTIVE"),
//                 mo_space_info_->corr_absolute_mo("RESTRICTED_DOCC"));
//             fci_ints->set_active_integrals_and_restricted_docc();
//             one_body_integrals_ = fci_ints->oei_a_vector();
//             scalar_energy_ = fci_ints->scalar_energy();
//             scalar_energy_ += ints_->nuclear_repulsion_energy() + ints_->frozen_core_energy();
//             outfile->Printf("\n OneBody integrals (fci_ints) takes %6.5f s",
//                             one_body_fci_ints.get());
//         }
//     }
//     active_integrals_.iterate([&](const std::vector<size_t>& i, double& value) {
//         if (CheMPS2::Irreps::directProd(orbitalIrreps[i[0]], orbitalIrreps[i[1]]) ==
//             CheMPS2::Irreps::directProd(orbitalIrreps[i[2]], orbitalIrreps[i[3]])) {
//             Ham->setVmat(i[0], i[1], i[2], i[3], value);
//         };
//     });
//     // int shift = iHandler->getDMRGcumulative(h);
//     int shift = 0;
//
//     for (int h = 0; h < iHandler->getNirreps(); h++) {
//         const int NOCC = frozen_docc[h];
//         for (int orb1 = 0; orb1 < active[h]; orb1++) {
//             for (int orb2 = orb1; orb2 < active[h]; orb2++) {
//                 Ham->setTmat(shift + orb1, shift + orb2,
//                              one_body_integrals_[(shift + orb1) * nOrbDMRG + (orb2 + shift)]);
//             }
//         }
//         shift += active[h];
//     }
//     Ham->setEconst(scalar_energy_);
//
//     double Energy = 1e-8;
//     /// DMRG can compute the 2DM pretty simply.  Always compute it
//     double* DMRG1DM = new double[nOrbDMRG * nOrbDMRG];
//     double* DMRG2DM = new double[nOrbDMRG * nOrbDMRG * nOrbDMRG * nOrbDMRG];
//     if (nOrbDMRG > 30) {
//         outfile->Printf("\n  Using a disk based 3 rdm storage");
//         disk_3_rdm_ = true;
//     }
//     double* DMRG3DM;
//     if (max_rdm_ > 2 && !disk_3_rdm_)
//         DMRG3DM = new double[nOrbDMRG * nOrbDMRG * nOrbDMRG * nOrbDMRG * nOrbDMRG * nOrbDMRG];
//
//     std::memset(DMRG1DM, 0.0, sizeof(double) * nOrbDMRG * nOrbDMRG);
//     std::memset(DMRG2DM, 0.0, sizeof(double) * nOrbDMRG * nOrbDMRG * nOrbDMRG * nOrbDMRG);
//     if (max_rdm_ > 2 && !disk_3_rdm_)
//         std::memset(DMRG3DM, 0.0,
//                     sizeof(double) * nOrbDMRG * nOrbDMRG * nOrbDMRG * nOrbDMRG * nOrbDMRG *
//                         nOrbDMRG);
//
//     std::shared_ptr<CheMPS2::DMRG> DMRGCI =
//         std::make_shared<CheMPS2::DMRG>(Prob.get(), OptScheme.get());
//
//     for (int state = 0; state < dmrgscf_which_root; state++) {
//
//         if (state > 0) {
//             DMRGCI->newExcitation(std::fabs(Energy));
//         }
//         local_timer DMRGSolve;
//         Energy = DMRGCI->Solve();
//         outfile->Printf("\n  Overall DMRG Solver took %6.5f s.", DMRGSolve.get());
//         local_timer DMRGRDMs;
//
//         DMRGCI->calc_rdms_and_correlations(max_rdm_ > 2 ? true : false, disk_3_rdm_);
//         outfile->Printf("\n  Overall DMRG RDM computation took %6.5f s.", DMRGRDMs.get());
//         outfile->Printf("\n  @DMRG Energy = %8.12f", Energy);
//         psi::Process::environment.globals["CURRENT ENERGY"] = Energy;
//         // if(dmrgscf_state_avg)
//         //{
//         //    DMRGCI->calc_rdms_and_correlations(max_rdm_ > 2 ? true : false);
//         //    CheMPS2::CASSCF::copy2DMover( DMRGCI->get2DM(), nOrbDMRG,
//         //    DMRG2DM);
//         //}
//
//         // if((state == 0) && (dmrgscf_which_root > 1)) {
//         // DMRGCI->activateExcitations( dmrgscf_which_root-1);}
//         //    {
//
//         //        DMRGCI->calc_rdms_and_correlations(max_rdm_ > 2 ? true :
//         //        false);
//         //        CheMPS2::CASSCF::copy2DMover( DMRGCI->get2DM(), nOrbDMRG,
//         //        DMRG2DM);
//         //    }
//         //}
//         // if( !(dmrgscf_state_avg) ) {
//         //    DMRGCI->calc_rdms_and_correlations(max_rdm_ > 2 ? true : false);
//     }
//     if (dmrg_print_corr) {
//         DMRGCI->getCorrelations()->Print();
//     }
//     // cout.rdbuf(cout_buffer);
//     // capturing.close();
//     // std::ifstream copying;
//     // copying.open( chemps2filename , ios::in ); // read only
//     // if (copying.is_open()){
//     //    string line;
//     //    while( getline( copying, line ) ){ (*outfile) << line << endl; }
//     //    copying.close();
//     //}
//     // system(("rm " + chemps2filename).c_str());
//
//     CheMPS2::CASSCF::copy2DMover(DMRGCI->get2DM(), nOrbDMRG, DMRG2DM);
//     CheMPS2::CASSCF::setDMRG1DM(nDMRGelectrons, nOrbDMRG, DMRG1DM, DMRG2DM);
//     if (max_rdm_ > 2 && !disk_3_rdm_) {
//         DMRGCI->get3DM()->fill_ham_index(1.0, false, DMRG3DM, 0, nOrbDMRG);
//     }
//
//     compute_reference(DMRG1DM, DMRG2DM, DMRG3DM, iHandler.get());
//     if (options_->get_bool("PRINT_NO")) {
//         print_natural_orbitals(DMRG1DM);
//     }
//     dmrg_ref_.set_Eref(Energy);
//
//     delete[] DMRG1DM;
//     delete[] DMRG2DM;
//     delete[] orbitalIrreps;
//     if (max_rdm_ > 2) {
//         delete[] DMRG3DM;
//     }
// }
// int DMRGSolver::chemps2_groupnumber(const string SymmLabel) {
//
//     int SyGroup = 0;
//     bool stopFindGN = false;
//     const int magic_number_max_groups_chemps2 = 8;
//     do {
//         if (SymmLabel.compare(CheMPS2::Irreps::getGroupName(SyGroup)) == 0) {
//             stopFindGN = true;
//         } else {
//             SyGroup++;
//         }
//     } while ((!stopFindGN) && (SyGroup < magic_number_max_groups_chemps2));
//
//     outfile->Printf("\n  Psi4 symmetry group was found to be <%s>.", SymmLabel.c_str());
//     if (SyGroup >= magic_number_max_groups_chemps2) {
//         outfile->Printf("\n  CheMPS2 did not recognize this symmetry group name. "
//                         "CheMPS2 only knows:");
//         for (int cnt = 0; cnt < magic_number_max_groups_chemps2; cnt++) {
//             outfile->Printf("\n    <%s>", (CheMPS2::Irreps::getGroupName(cnt)).c_str());
//         }
//         throw psi::PSIEXCEPTION("CheMPS2 did not recognize the symmetry group name!");
//     }
//     return SyGroup;
// }
// std::vector<double> DMRGSolver::one_body_operator() {
//     ///
//     psi::Dimension restricted_docc_dim = mo_space_info_->dimension("INACTIVE_DOCC");
//     psi::Dimension nsopi = scf_info_->nsopi();
//     int nirrep = ints_->nirrep();
//     psi::Dimension nmopi = mo_space_info_->dimension("ALL");
//
//     psi::SharedMatrix Cdocc(new psi::Matrix("C_RESTRICTED", nirrep, nsopi, restricted_docc_dim));
//     psi::SharedMatrix Ca = ints_->Ca();
//     for (int h = 0; h < nirrep; h++) {
//         for (int i = 0; i < restricted_docc_dim[h]; i++) {
//             Cdocc->set_column(h, i, Ca->get_column(h, i));
//         }
//     }
//     /// F_frozen = D_{uv}^{frozen} * (2<uv|rs> - <ur | vs>)
//     /// F_restricted = D_{uv}^{restricted} * (2<uv|rs> - <ur | vs>)
//     /// F_inactive = F_frozen + F_restricted + H_{pq}^{core}
//     /// D_{uv}^{frozen} = \sum_{i = 0}^{frozen}C_{ui} * C_{vi}
//     /// D_{uv}^{inactive} = \sum_{i = 0}^{inactive}C_{ui} * C_{vi}
//     /// This section of code computes the fock matrix for the
//     /// INACTIVE_DOCC("RESTRICTED_DOCC")
//
//     std::shared_ptr<psi::JK> JK_inactive = psi::JK::build_JK(
//         ints_->basisset(), psi::BasisSet::zero_ao_basis_set(), ints_->wfn()->options());
//
//     JK_inactive->set_memory(psi::Process::environment.get_memory() * 0.8);
//     JK_inactive->initialize();
//
//     std::vector<std::shared_ptr<psi::Matrix>>& Cl = JK_inactive->C_left();
//     Cl.clear();
//     Cl.push_back(Cdocc);
//     JK_inactive->compute();
//     psi::SharedMatrix J_restricted = JK_inactive->J()[0];
//     psi::SharedMatrix K_restricted = JK_inactive->K()[0];
//
//     J_restricted->scale(2.0);
//     psi::SharedMatrix F_restricted = J_restricted->clone();
//     F_restricted->subtract(K_restricted);
//
//     /// Just create the OneInt integrals from scratch
//     std::shared_ptr<psi::PSIO> psio_ = psi::PSIO::shared_object();
//     psi::SharedMatrix T =
//         psi::SharedMatrix(ints_->wfn()->matrix_factory()->create_matrix(PSIF_SO_T));
//     psi::SharedMatrix V =
//         psi::SharedMatrix(ints_->wfn()->matrix_factory()->create_matrix(PSIF_SO_V));
//     psi::SharedMatrix OneInt = T;
//     OneInt->zero();
//
//     //    T->load(psio_, PSIF_OEI);
//     //    V->load(psio_, PSIF_OEI);
//     //    psi::SharedMatrix Hcore_ = ints_->wfn()->matrix_factory()->create_shared_matrix("Core
//     //    Hamiltonian"); Hcore_->add(T); Hcore_->add(V);
//     Hcore_ = psi::SharedMatrix(ints_->wfn()->H()->clone());
//
//     psi::SharedMatrix Hcore(Hcore_->clone());
//     F_restricted->add(Hcore);
//     F_restricted->transform(Ca);
//     Hcore->transform(Ca);
//
//     size_t all_nmo = mo_space_info_->size("ALL");
//     psi::SharedMatrix F_restric_c1(new psi::Matrix("F_restricted", all_nmo, all_nmo));
//     size_t offset = 0;
//     for (int h = 0; h < nirrep; h++) {
//         for (int p = 0; p < nmopi[h]; p++) {
//             for (int q = 0; q < nmopi[h]; q++) {
//                 F_restric_c1->set(p + offset, q + offset, F_restricted->get(h, p, q));
//             }
//         }
//         offset += nmopi[h];
//     }
//     size_t na_ = mo_space_info_->size("ACTIVE");
//     size_t nmo2 = na_ * na_;
//     std::vector<double> oei_a(nmo2);
//     std::vector<double> oei_b(nmo2);
//     bool casscf_debug_print_ = options_->get_bool("CASSCF_DEBUG_PRINTING");
//
//     auto absolute_active = mo_space_info_->absolute_mo("ACTIVE");
//     for (size_t u = 0; u < na_; u++) {
//         for (size_t v = 0; v < na_; v++) {
//             double value = F_restric_c1->get(absolute_active[u], absolute_active[v]);
//             // double h_value = H->get(absolute_active[u], absolute_active[v]);
//             oei_a[u * na_ + v] = value;
//             // oei_b[u * na_ + v ] = value;
//             if (casscf_debug_print_)
//                 outfile->Printf("\n oei(%d, %d) = %8.8f", u, v, value);
//         }
//     }
//     psi::Dimension restricted_docc = mo_space_info_->dimension("INACTIVE_DOCC");
//     double E_restricted = ints_->nuclear_repulsion_energy();
//     for (int h = 0; h < nirrep; h++) {
//         for (int rd = 0; rd < restricted_docc[h]; rd++) {
//             E_restricted += Hcore->get(h, rd, rd) + F_restricted->get(h, rd, rd);
//         }
//     }
//     /// Since F^{INACTIVE} includes frozen_core in fock build, the energy
//     /// contribution includes frozen_core_energy
//     scalar_energy_ = 0.000;
//     if (casscf_debug_print_) {
//         outfile->Printf("\n Frozen Core Energy = %8.8f", ints_->frozen_core_energy());
//         outfile->Printf("\n Restricted Energy = %8.8f", E_restricted -
//         ints_->frozen_core_energy()); outfile->Printf("\n Scalar Energy = %8.8f",
//                         ints_->scalar() + E_restricted - ints_->frozen_core_energy());
//     }
//     scalar_energy_ = E_restricted;
//     return oei_a;
// }
// void DMRGSolver::print_natural_orbitals(double* opdm) {
//     print_h2("NATURAL ORBITALS");
//     psi::Dimension active_dim = mo_space_info_->dimension("ACTIVE");
//     int nirrep = ints_->nirrep();
//     size_t na_ = mo_space_info_->size("ACTIVE");
//
//     std::shared_ptr<psi::Matrix> opdm_a(new psi::Matrix("OPDM_A", nirrep, active_dim,
//     active_dim));
//
//     int offset = 0;
//     for (int h = 0; h < nirrep; h++) {
//         for (int u = 0; u < active_dim[h]; u++) {
//             for (int v = 0; v < active_dim[h]; v++) {
//                 opdm_a->set(h, u, v, opdm[(u + offset) * na_ + v + offset]);
//             }
//         }
//         offset += active_dim[h];
//     }
//     psi::SharedVector OCC_A(new Vector("ALPHA OCCUPATION", nirrep, active_dim));
//     psi::SharedMatrix NO_A(new psi::Matrix(nirrep, active_dim, active_dim));
//
//     opdm_a->diagonalize(NO_A, OCC_A, descending);
//     std::vector<std::pair<double, std::pair<int, int>>> vec_irrep_occupation;
//     for (int h = 0; h < nirrep; h++) {
//         for (int u = 0; u < active_dim[h]; u++) {
//             auto irrep_occ = std::make_pair(OCC_A->get(h, u), std::make_pair(h, u + 1));
//             vec_irrep_occupation.push_back(irrep_occ);
//         }
//     }
//     std::sort(vec_irrep_occupation.begin(), vec_irrep_occupation.end(),
//               std::greater<std::pair<double, std::pair<int, int>>>());
//
//     int count = 0;
//     outfile->Printf("\n    ");
//     for (auto vec : vec_irrep_occupation) {
//         outfile->Printf(" %4d%-4s%11.6f  ", vec.second.second,
//                         mo_space_info->irrep_label(vec.second.first).c_str(), vec.first);
//         if (count++ % 3 == 2 && count != vec_irrep_occupation.size())
//             outfile->Printf("\n    ");
//     }
//     outfile->Printf("\n\n");
// }
} // namespace forte

#endif // #ifdef HAVE_CHEMPS2
