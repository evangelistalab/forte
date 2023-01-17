/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2023 by its authors (see COPYING, COPYING.LESSER, AUTHORS).
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
#include <iostream>
#include <fstream>
#include <filesystem>

#include "chemps2/CASSCF.h"
#include "chemps2/Initialize.h"
#include "chemps2/Irreps.h"
#include "chemps2/Problem.h"

#include "psi4/libpsi4util/process.h"
#include "psi4/libpsio/psio.hpp"

#include "base_classes/mo_space_info.h"
#include "integrals/active_space_integrals.h"
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

DMRGSolver::~DMRGSolver() {
    // move MPS files to mps_files_path_
    move_mps_files(true);
    // delete directory for checkpoint files
    fs::remove_all(tmp_path_);
}

void DMRGSolver::move_mps_files(bool from_cwd_to_folder) {
    if (from_cwd_to_folder) {
        for (const std::string& name : mps_files_) {
            auto pf = fs::path(name);
            if (fs::exists(pf))
                fs::rename(pf, mps_files_path_ / pf);
        }
    } else {
        auto cwd = fs::current_path();
        for (const std::string& name : mps_files_) {
            auto pf = mps_files_path_ / fs::path(name);
            if (fs::exists(pf))
                fs::rename(pf, cwd / fs::path(name));
        }
    }
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

    // Save CheMPS2 default MPS file names
    for (size_t root = 0; root < nroot_; ++root) {
        std::stringstream ss;
        ss << CheMPS2::DMRG_MPS_storage_prefix << root << ".h5"; // default name by CheMPS2
        mps_files_.emplace_back(ss.str());
    }

    auto cwd = fs::current_path();
    mps_files_path_ = cwd / fs::path(state_label_);

    tmp_path_ = fs::path(PSIOManager::shared_object()->get_default_path()) / fs::path(state_label_);
    if (not fs::exists(tmp_path_))
        fs::create_directory(tmp_path_);
}

void DMRGSolver::set_options(std::shared_ptr<ForteOptions> options) {
    read_wfn_guess_ = options_->get_bool("READ_ACTIVE_WFN_GUESS");
    dump_wfn_ = true; // always dump MPS to disk
    //    dump_wfn_ = options_->get_bool("DUMP_ACTIVE_WFN");

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
    timer t("DMRG Solver Compute Energy");

    std::ofstream capturing;
    std::streambuf* cout_buffer;
    std::string chemps2filename = "output.chemps2.tmp";
    capturing.open(chemps2filename.c_str(), std::ios::trunc); // truncate
    cout_buffer = std::cout.rdbuf(capturing.rdbuf());

    timer t_init("DMRG Initialization");
    auto n_sweep_states = static_cast<int>(dmrg_sweep_states_.size());
    CheMPS2::Initialize::Init();

    // Create a CheMPS2::ConvergenceSchem
    conv_scheme_ = std::make_unique<CheMPS2::ConvergenceScheme>(n_sweep_states);
    for (int cnt = 0; cnt < n_sweep_states; cnt++) {
        conv_scheme_->set_instruction(cnt, dmrg_sweep_states_[cnt], dmrg_sweep_e_convergence_[cnt],
                                      dmrg_sweep_max_sweeps_[cnt], dmrg_noise_prefactors_[cnt],
                                      dmrg_davidson_rtol_[cnt]);
    }

    // Create the CheMPS2::Hamiltonian
    auto actv_irreps = mo_space_info_->symmetry("ACTIVE");
    hamiltonian_ = std::make_unique<CheMPS2::Hamiltonian>(nactv_, pg_number_, actv_irreps.data());

    double energy_core = as_ints_->frozen_core_energy() + as_ints_->scalar_energy();
    hamiltonian_->setEconst(energy_core + as_ints_->nuclear_repulsion_energy());

    for (int p = 0; p < nactv_; ++p) {
        for (int q = 0; q < nactv_; ++q) {
            if (actv_irreps[p] == actv_irreps[q]) {
                hamiltonian_->setTmat(p, q, as_ints_->oei_a(p, q));
            }
            for (int r = 0; r < nactv_; ++r) {
                for (int s = 0; s < nactv_; ++s) {
                    if ((actv_irreps[p] ^ actv_irreps[q]) == (actv_irreps[r] ^ actv_irreps[s])) {
                        hamiltonian_->setVmat(p, q, r, s, as_ints_->tei_ab(p, q, r, s));
                    }
                }
            }
        }
    }

    // Create the CheMPS2::Problem
    auto prob = std::make_unique<CheMPS2::Problem>(hamiltonian_.get(), multiplicity_ - 1,
                                                   nelecs_actv_, wfn_irrep_);
    if (not prob->checkConsistency()) {
        throw std::runtime_error("No Hilbert state vector compatible with all symmetry sectors!");
    }
    prob->SetupReorderD2h();
    t_init.stop();

    // Path for MPS files
    auto cwd = fs::current_path();
    if (dump_wfn_) {
        if (not fs::exists(mps_files_path_))
            fs::create_directory(mps_files_path_);
        std::cout << "MPS files will be dumped to " << mps_files_path_ << '\n';
    }
    if (read_wfn_guess_) {
        // try to read current path, then try to read mps_files_path_ (i.e., copy to cwd)
        for (const std::string& name : mps_files_) {
            auto pf = fs::path(name);
            if ((not fs::exists(cwd / pf)) and fs::exists(mps_files_path_ / pf))
                fs::copy_file(mps_files_path_ / pf, cwd / pf);
        }
    }

    // Start DMRG sweeps
    auto solver = std::make_shared<CheMPS2::DMRG>(prob.get(), conv_scheme_.get(), true, tmp_path_);
    energies_.clear();

    timer t_compt("DMRG Energy");
    for (size_t root = 0; root < nroot_; ++root) {
        if (root > 0)
            solver->newExcitation(std::fabs(energies_[root - 1]));
        std::cout << "\n  ==> Computing Energy for " << state_.str_minimum() << " <==\n\n";
        timer t_root("DMRG Energy Root " + std::to_string(root));
        energies_.push_back(solver->Solve());
        t_root.stop();
        if (dmrg_print_corr_) {
            solver->calc_rdms_and_correlations(false, false); // need RDMs
            solver->getCorrelations()->Print();
        }
        if (root == 0 and nroot_ > 1) {
            solver->activateExcitations(static_cast<int>(nroot_ - 1));
        }
    }
    t_compt.stop();

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

    // move MPS files from CWD to the folder (to prevent load problems of CheMPS2)
    move_mps_files(true);

    // Push to psi4 environment
    double energy = energies_[root_];
    psi::Process::environment.globals["CURRENT ENERGY"] = energy;
    psi::Process::environment.globals["DMRG ENERGY"] = energy;
    return energy;
}

std::vector<std::shared_ptr<RDMs>>
DMRGSolver::rdms(const std::vector<std::pair<size_t, size_t>>& root_list, int max_rdm_level,
                 RDMsType rdm_type) {
    timer t("DMRG Solver Compute RDMs");

    if (max_rdm_level < 1)
        return std::vector<std::shared_ptr<RDMs>>(root_list.size());

    // make sure the MPS files are available
    move_mps_files(false);
    for (const std::string& name : mps_files_) {
        if (not fs::exists(name)) {
            outfile->Printf("\n  File does not exist: %s", name.c_str());
            outfile->Printf("\n  Please first run DMRGSolver::compute_energy!");
            throw std::runtime_error("Please first run DMRGSolver::compute_energy!");
        }
    }

    // check root list
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

    std::vector<std::shared_ptr<RDMs>> rdms;
    bool do_3rdm = max_rdm_level > 2;
    bool disk_3rdm = mo_space_info_->size("ACTIVE") >= 30;

    timer t_init("DMRG RDMs Initialization");

    std::ofstream capturing;
    std::streambuf* cout_buffer;
    std::string chemps2filename = "output.chemps2.rdms.tmp";
    capturing.open(chemps2filename.c_str(), std::ios::trunc); // truncate
    cout_buffer = std::cout.rdbuf(capturing.rdbuf());

    // need a new solver to read the MPS files in CWD
    const std::string tmp_path = PSIOManager::shared_object()->get_default_path();
    auto prob = std::make_unique<CheMPS2::Problem>(hamiltonian_.get(), multiplicity_ - 1,
                                                   nelecs_actv_, wfn_irrep_);
    prob->SetupReorderD2h();
    auto solver = std::make_shared<CheMPS2::DMRG>(prob.get(), conv_scheme_.get(), true, tmp_path);
    t_init.stop();

    timer t_compt("DMRG RDMs Computation");
    for (size_t root = 0; root < nroot_; ++root) {
        if (root > 0)
            solver->newExcitation(std::fabs(energies_[root - 1]));
        if (roots.find(root) != roots.end()) {
            timer t_root("DMRG RDMs Root " + std::to_string(root));
            solver->calc_rdms_and_correlations(do_3rdm, disk_3rdm);
            rdms.push_back(fill_current_rdms(solver, do_3rdm, rdm_type));
        }
        if (root == 0 and nroot_ > 1) {
            solver->activateExcitations(static_cast<int>(nroot_ - 1));
        }
    }
    t_compt.stop();

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

    // move MPS files from CWD to the folder
    move_mps_files(true);

    return rdms;
}

std::shared_ptr<RDMs> DMRGSolver::fill_current_rdms(std::shared_ptr<CheMPS2::DMRG> solver,
                                                    const bool do_3rdm, RDMsType rdm_type) {
    std::vector<size_t> dim2(2, nactv_);
    std::vector<size_t> dim4(4, nactv_);
    auto g1 = ambit::Tensor::build(ambit::CoreTensor, "DMRG G1", dim2);
    auto g2 = ambit::Tensor::build(ambit::CoreTensor, "DMRG G2", dim4);

    CheMPS2::CASSCF::copy2DMover(solver->get2DM(), nactv_, g2.data().data());
    CheMPS2::CASSCF::setDMRG1DM(nelecs_actv_, nactv_, g1.data().data(), g2.data().data());

    ambit::Tensor g3;
    if (do_3rdm) {
        std::vector<size_t> dim6(6, nactv_);
        g3 = ambit::Tensor::build(ambit::CoreTensor, "DMRG G3", dim6);
        solver->get3DM()->fill_ham_index(1.0, false, g3.data().data(), 0, nactv_);
    }

    if (rdm_type == RDMsType::spin_free) {
        if (do_3rdm)
            return std::make_shared<RDMsSpinFree>(g1, g2, g3);
        else
            return std::make_shared<RDMsSpinFree>(g1, g2);
    }

    auto g1a = RDMs::sf1_to_sd1(g1);
    auto g2aa = RDMs::sf2_to_sd2aa(g2);
    auto g2ab = RDMs::sf2_to_sd2ab(g2);
    if (not do_3rdm)
        return std::make_shared<RDMsSpinDependent>(g1a, g1a, g2aa, g2ab, g2aa);

    auto g3aaa = RDMs::sf3_to_sd3aaa(g3);
    auto g3aab = RDMs::sf3_to_sd3aab(g3);
    auto g3abb = RDMs::sf3_to_sd3abb(g3);
    return std::make_shared<RDMsSpinDependent>(g1a, g1a, g2aa, g2ab, g2aa, g3aaa, g3aab, g3abb,
                                               g3aaa);
}

std::vector<std::shared_ptr<RDMs>>
DMRGSolver::transition_rdms(const std::vector<std::pair<size_t, size_t>>&,
                            std::shared_ptr<ActiveSpaceMethod>, int, RDMsType) {
    throw std::runtime_error("DMRGSolver::transition_rdms is not available in CheMPS2!");
}

} // namespace forte
#endif // #ifdef HAVE_CHEMPS2
