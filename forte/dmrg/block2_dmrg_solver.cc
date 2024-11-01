/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2024 by its authors (see COPYING, COPYING.LESSER, AUTHORS).
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

#ifdef HAVE_BLOCK2

#include "psi4/libmints/molecule.h"
#include "psi4/libpsi4util/process.h"
#include "psi4/libpsio/psio.hpp"
#include "psi4/libmints/dimension.h"

namespace psi {
class Matrix;
class Vector;
} // namespace psi

#include "helpers/printing.h"
#include "helpers/timer.h"
#include "integrals/active_space_integrals.h"
#include "block2_dmrg_solver.h"

#include "block2.hpp"
#include <vector>
#include <memory>
#include <cmath>
#include <array>
#include <string>
#include <functional>
#include <filesystem>
#include <set>

namespace fs = std::filesystem;

namespace forte {

struct Block2DMRGSolverImpl {
    std::shared_ptr<block2::DMRGDriver<block2::SU2, double>> driver_su2_ = nullptr;
    std::shared_ptr<block2::DMRGDriver<block2::SZ, double>> driver_sz_ = nullptr;
    bool is_spin_adapted_;
    std::string scratch_;
    size_t stack_mem_;
    Block2DMRGSolverImpl(bool is_spin_adapted, size_t stack_mem, std::string scratch)
        : is_spin_adapted_(is_spin_adapted), scratch_(scratch), stack_mem_(stack_mem) {
        if (is_spin_adapted_)
            driver_su2_ =
                std::make_shared<block2::DMRGDriver<block2::SU2, double>>(stack_mem, scratch);
        else
            driver_sz_ =
                std::make_shared<block2::DMRGDriver<block2::SZ, double>>(stack_mem, scratch);
    }
    void reset() {
        if (is_spin_adapted_) {
            // this line should not be deleted
            driver_su2_ = nullptr;
            driver_su2_ =
                std::make_shared<block2::DMRGDriver<block2::SU2, double>>(stack_mem_, scratch_);
        } else {
            // this line should not be deleted
            driver_sz_ = nullptr;
            driver_sz_ =
                std::make_shared<block2::DMRGDriver<block2::SZ, double>>(stack_mem_, scratch_);
        }
    }
    void initialize_system(int n_sites, int n_elec, int spin, int pg_irrep,
                           const vector<int>& actv_irreps, bool singlet_embedding = true,
                           int heis_twos = -1, int heis_twosz = 0) const {
        if (is_spin_adapted_) {
            std::vector<typename block2::SU2::pg_t> orb_sym;
            orb_sym.reserve(actv_irreps.size());
            for (auto& ir : actv_irreps)
                orb_sym.push_back(static_cast<typename block2::SU2::pg_t>(ir));
            driver_su2_->initialize_system(n_sites, n_elec, abs(spin), pg_irrep, orb_sym, heis_twos,
                                           heis_twosz, singlet_embedding);
        } else {
            std::vector<typename block2::SZ::pg_t> orb_sym;
            orb_sym.reserve(actv_irreps.size());
            for (auto& ir : actv_irreps)
                orb_sym.push_back(static_cast<typename block2::SZ::pg_t>(ir));
            driver_sz_->initialize_system(n_sites, n_elec, spin, pg_irrep, orb_sym, heis_twos,
                                          heis_twosz, singlet_embedding);
        }
    }
    std::shared_ptr<block2::GeneralFCIDUMP<double>> expr_builder() const {
        return is_spin_adapted_ ? driver_su2_->expr_builder() : driver_sz_->expr_builder();
    }
    std::shared_ptr<void>
    get_mpo(std::shared_ptr<block2::GeneralFCIDUMP<double>> expr, int iprint, double cutoff = 1E-14,
            block2::MPOAlgorithmTypes algo_type = block2::MPOAlgorithmTypes::FastBipartite) const {
        return is_spin_adapted_ ? std::static_pointer_cast<void>(
                                      driver_su2_->get_mpo(expr, iprint, cutoff, algo_type))
                                : std::static_pointer_cast<void>(
                                      driver_sz_->get_mpo(expr, iprint, cutoff, algo_type));
    }
    std::shared_ptr<void> get_random_mps(const std::string& tag, int bond_dim = 500, int center = 0,
                                         int dot = 2, int nroots = 1,
                                         const std::vector<double>& occs = vector<double>(),
                                         bool full_fci = true) const {
        return is_spin_adapted_ ? std::static_pointer_cast<void>(driver_su2_->get_random_mps(
                                      tag, bond_dim, center, dot, block2::SU2(block2::SU2::invalid),
                                      nroots, occs, full_fci))
                                : std::static_pointer_cast<void>(driver_sz_->get_random_mps(
                                      tag, bond_dim, center, dot, block2::SZ(block2::SZ::invalid),
                                      nroots, occs, full_fci));
    }
    std::shared_ptr<void> load_mps(const std::string& tag, int nroots = 1) const {
        return is_spin_adapted_ ? std::static_pointer_cast<void>(driver_su2_->load_mps(tag, nroots))
                                : std::static_pointer_cast<void>(driver_sz_->load_mps(tag, nroots));
    }
    std::vector<double> dmrg(std::shared_ptr<void> mpo, std::shared_ptr<void> ket,
                             int n_sweeps = 10, double tol = 1E-8,
                             std::vector<int> bond_dims = vector<int>(),
                             std::vector<double> noises = std::vector<double>(),
                             std::vector<double> thrds = std::vector<double>(), int iprint = 2,
                             double cutoff = 1E-20, int dav_max_iter = 4000) const {
        std::vector<block2::ubond_t> bond_dims_(bond_dims.size());
        for (size_t i = 0; i < bond_dims.size(); i++)
            bond_dims_[i] = static_cast<block2::ubond_t>(bond_dims[i]);
        if (is_spin_adapted_) {
            auto mpo_ = std::static_pointer_cast<block2::MPO<block2::SU2, double>>(mpo);
            auto ket_ = std::static_pointer_cast<block2::MPS<block2::SU2, double>>(ket);
            return driver_su2_->dmrg(mpo_, ket_, n_sweeps, tol, bond_dims_, noises, thrds, iprint,
                                     cutoff, dav_max_iter);
        } else {
            auto mpo_ = std::static_pointer_cast<block2::MPO<block2::SZ, double>>(mpo);
            auto ket_ = std::static_pointer_cast<block2::MPS<block2::SZ, double>>(ket);
            return driver_sz_->dmrg(mpo_, ket_, n_sweeps, tol, bond_dims_, noises, thrds, iprint,
                                    cutoff, dav_max_iter);
        }
    }
    std::vector<std::shared_ptr<block2::GTensor<double>>>
    get_npdm(const std::vector<std::string>& exprs, std::shared_ptr<void> ket,
             std::shared_ptr<void> bra, int site_type = 0, int iprint = 0,
             block2::ExpectationAlgorithmTypes algo_type =
                 block2::ExpectationAlgorithmTypes::SymbolFree |
                 block2::ExpectationAlgorithmTypes::Compressed,
             int max_bond_dim = -1) const {
        if (is_spin_adapted_) {
            auto ket_ = std::static_pointer_cast<block2::MPS<block2::SU2, double>>(ket);
            auto bra_ = std::static_pointer_cast<block2::MPS<block2::SU2, double>>(bra);
            return driver_su2_->get_npdm(exprs, ket_, bra_, site_type, algo_type, iprint, 1.0e-16,
                                         true, max_bond_dim);
        } else {
            auto ket_ = std::static_pointer_cast<block2::MPS<block2::SZ, double>>(ket);
            auto bra_ = std::static_pointer_cast<block2::MPS<block2::SZ, double>>(bra);
            return driver_sz_->get_npdm(exprs, ket_, bra_, site_type, algo_type, iprint, 1.0e-16,
                                        true, max_bond_dim);
        }
    }
    std::shared_ptr<void> split_mps(std::shared_ptr<void> ket, int nroot, int iroot,
                                    const std::string& tag) const {
        if (is_spin_adapted_) {
            auto ket_ = std::static_pointer_cast<block2::MPS<block2::SU2, double>>(ket);
            return nroot == 1 ? ket_ : driver_su2_->split_mps(ket_, iroot, tag);
        } else {
            auto ket_ = std::static_pointer_cast<block2::MPS<block2::SZ, double>>(ket);
            return nroot == 1 ? ket_ : driver_sz_->split_mps(ket_, iroot, tag);
        }
    }
    double get_spin_square(std::shared_ptr<void> ket, int nroot, int iroot, int iprint = 0) const {
        if (is_spin_adapted_) {
            auto ket_ = std::static_pointer_cast<block2::MPS<block2::SU2, double>>(ket);
            auto iket_ = split_mps(ket_, nroot, iroot, ket_->info->tag + "@TMP-SS");
            ket_ = std::static_pointer_cast<block2::MPS<block2::SU2, double>>(iket_);
            auto mpo_ = driver_su2_->get_spin_square_mpo(iprint);
            return driver_su2_->expectation(ket_, mpo_, ket_, iprint);
        } else {
            auto ket_ = std::static_pointer_cast<block2::MPS<block2::SZ, double>>(ket);
            auto iket_ = split_mps(ket_, nroot, iroot, ket_->info->tag + "@TMP-SS");
            ket_ = std::static_pointer_cast<block2::MPS<block2::SZ, double>>(iket_);
            auto mpo_ = driver_sz_->get_spin_square_mpo(iprint);
            return driver_sz_->expectation(ket_, mpo_, ket_, iprint);
        }
    }
};

struct Block2ScratchManager {
    std::set<std::string> scratch_folders;
    Block2ScratchManager() {}
    ~Block2ScratchManager() {
        // remove scratch files when just before the program quits
        // TODO may cause problem for MPI
        const std::vector<std::string> file_name_patterns = std::vector<std::string>{
            ".NPDM.FRAG.", ".MPS.",         ".MPS.INFO.", ".MMPS.", ".MMPS.INFO.",
            ".MMPS-WFN.",  "-mps_info.bin", "@TMP.",      "@TMP-",  "DSRG-"};
        for (auto& path : scratch_folders)
            if (block2::Parsing::path_exists(path)) {
                for (const auto& file : fs::directory_iterator(path))
                    for (auto& name : file_name_patterns)
                        if (file.path().filename().string().find(name) != std::string::npos)
                            fs::remove(file.path());
                if (fs::is_empty(path))
                    fs::remove(path);
            }
    }
    static Block2ScratchManager& manager() {
        static Block2ScratchManager _manager;
        return _manager;
    }
};

Block2DMRGSolver::Block2DMRGSolver(StateInfo state, size_t nroot, std::shared_ptr<SCFInfo> scf_info,
                                   std::shared_ptr<ForteOptions> options,
                                   std::shared_ptr<MOSpaceInfo> mo_space_info,
                                   std::shared_ptr<ActiveSpaceIntegrals> as_ints)
    : ActiveSpaceMethod(state, nroot, mo_space_info, as_ints), scf_info_(scf_info),
      options_(options) {
    bool is_spin_adapted =
        as_ints->ints()->spin_restriction() == IntegralSpinRestriction::Restricted &&
        options_->get_bool("BLOCK2_SPIN_ADAPTED");
    std::string scratch = psi::PSIOManager::shared_object()->get_default_path() + "forte." +
                          std::to_string(getpid()) + ".block2." +
                          std::to_string(mo_space_info_->size("ACTIVE"));
    Block2ScratchManager::manager().scratch_folders.insert(scratch);
    size_t stack_mem =
        static_cast<size_t>(options_->get_double("BLOCK2_STACK_MEM") * 1024 * 1024 * 1024);
    impl_ = make_shared<Block2DMRGSolverImpl>(is_spin_adapted, stack_mem, scratch);
    na_ = state.na() - mo_space_info->size("INACTIVE_DOCC");
    nb_ = state.nb() - mo_space_info->size("INACTIVE_DOCC");
    maxiter_ = options_->get_int("BLOCK2_N_TOTAL_SWEEPS");
}

Block2DMRGSolver::~Block2DMRGSolver() {
    if (impl_ == nullptr || !block2::Parsing::path_exists(impl_->scratch_))
        return;
    const std::vector<std::string> file_name_patterns =
        std::vector<std::string>{".NPDM.FRAG.", "@TMP.", "@TMP-"};
    for (const auto& file : fs::directory_iterator(impl_->scratch_))
        for (auto& name : file_name_patterns)
            if (file.path().filename().string().find(name) != std::string::npos)
                fs::remove(file.path());
    if (fs::is_empty(impl_->scratch_))
        fs::remove(impl_->scratch_);
}

double Block2DMRGSolver::compute_energy() {
    if (print_ > PrintLevel::Quiet) {
        print_method_banner({impl_->is_spin_adapted_ ? "block2 DMRG (spin-adapted)"
                                                     : "block2 DMRG (non-spin-adapted)",
                             "by Huanchen Zhai", "Jan 14-16, 2023"});
        psi::outfile->Printf(
            "\n    Reference: H. Zhai & G. K. Chan J. Chem. Phys. 2021, 154, 224116.");
    }

    timer t("BLOCK2 Solver Compute Energy");

    impl_->reset();

    // system initialization
    bool singlet_embedding = dmrg_options_->get_bool("BLOCK2_SINGLET_EMBEDDING");
    std::vector<int> actv_irreps = mo_space_info_->symmetry("ACTIVE");
    int n_sites = static_cast<int>(mo_space_info_->size("ACTIVE"));
    int n_elec = na_ + nb_;
    int spin = state_.multiplicity() - 1;
    int pg_irrep = state_.irrep();
    impl_->initialize_system(n_sites, n_elec, spin, pg_irrep, actv_irreps, singlet_embedding);

    // set up integrals
    std::shared_ptr<block2::GeneralFCIDUMP<double>> r = impl_->expr_builder();
    r->const_e += as_ints_->frozen_core_energy() + as_ints_->scalar_energy() +
                  as_ints_->nuclear_repulsion_energy();
    const double integral_cutoff = dmrg_options_->get_double("BLOCK2_INTERGRAL_CUTOFF");
    const uint16_t n = n_sites;
    const std::vector<int> tei_shape = std::vector<int>{n, n, n, n};
    const std::vector<size_t> tei_strides =
        std::vector<size_t>{(size_t)n * n * n, (size_t)n * n, 1, (size_t)n};
    const std::vector<size_t> tei_ba_strides =
        std::vector<size_t>{(size_t)n * n, (size_t)n * n * n, (size_t)n, 1};
    const std::vector<int> oei_shape = std::vector<int>{n, n};
    const std::vector<size_t> oei_strides = std::vector<size_t>{(size_t)n, 1};

    if (impl_->is_spin_adapted_) {
        // (.+.) indicates spin coupling
        // (.+.)0 = singlet
        // (.+.)1 = doublet
        r->exprs.push_back("((C+(C+D)0)1+D)0");
        r->exprs.push_back("(C+D)0");
        r->add_sum_term(as_ints_->tei_ab_vector().data(), as_ints_->tei_ab_vector().size(),
                        tei_shape, tei_strides, integral_cutoff, 1.0, actv_irreps);
        r->add_sum_term(as_ints_->oei_a_vector().data(), as_ints_->oei_a_vector().size(), oei_shape,
                        oei_strides, integral_cutoff, sqrt(2.0), actv_irreps);
    } else {
        // cd = create/destroy alpha
        // CD = create/destroy beta
        r->exprs.push_back("ccdd"), r->exprs.push_back("cCDd");
        r->exprs.push_back("CcdD"), r->exprs.push_back("CCDD");
        r->exprs.push_back("cd"), r->exprs.push_back("CD");
        r->add_sum_term(as_ints_->tei_aa_vector().data(), as_ints_->tei_aa_vector().size(),
                        tei_shape, tei_strides, integral_cutoff, 0.25, actv_irreps);
        r->add_sum_term(as_ints_->tei_ab_vector().data(), as_ints_->tei_ab_vector().size(),
                        tei_shape, tei_strides, integral_cutoff, 0.5, actv_irreps);
        r->add_sum_term(as_ints_->tei_ab_vector().data(), as_ints_->tei_ab_vector().size(),
                        tei_shape, tei_ba_strides, integral_cutoff, 0.5, actv_irreps);
        r->add_sum_term(as_ints_->tei_bb_vector().data(), as_ints_->tei_bb_vector().size(),
                        tei_shape, tei_strides, integral_cutoff, 0.25, actv_irreps);
        r->add_sum_term(as_ints_->oei_a_vector().data(), as_ints_->oei_a_vector().size(), oei_shape,
                        oei_strides, integral_cutoff, 1.0, actv_irreps);
        r->add_sum_term(as_ints_->oei_b_vector().data(), as_ints_->oei_b_vector().size(), oei_shape,
                        oei_strides, integral_cutoff, 1.0, actv_irreps);
    }

    r = r->adjust_order();

    // dmrg sweep settings
    double e_conv = dmrg_options_->get_double("BLOCK2_SWEEP_ENERGY_CONV");
    double mps_cutoff = dmrg_options_->get_double("BLOCK2_CUTOFF");
    std::vector<int> sweep_n_sweeps = dmrg_options_->get_int_list("BLOCK2_SWEEP_N_SWEEPS");
    std::vector<int> sweep_bond_dims = dmrg_options_->get_int_list("BLOCK2_SWEEP_BOND_DIMS");
    std::vector<double> sweep_noises = dmrg_options_->get_double_list("BLOCK2_SWEEP_NOISES");
    std::vector<double> sweep_davidson_tols =
        dmrg_options_->get_double_list("BLOCK2_SWEEP_DAVIDSON_TOLS");
    int dmrg_verbose = dmrg_options_->get_int("BLOCK2_VERBOSE");

    size_t n_schedule =
        max(sweep_n_sweeps.size(),
            max(sweep_bond_dims.size(), max(sweep_noises.size(), sweep_davidson_tols.size())));
    while (sweep_davidson_tols.size() < n_schedule)
        sweep_davidson_tols.push_back(e_conv == 0.0 ? 1E-9 : e_conv * 10);
    while (sweep_noises.size() < n_schedule)
        if (sweep_noises.size() == n_schedule - 1)
            sweep_noises.push_back(0.0);
        else
            sweep_noises.push_back(sweep_davidson_tols[sweep_noises.size()] * 10);
    while (sweep_bond_dims.size() < n_schedule)
        if (sweep_bond_dims.size() == 0)
            sweep_bond_dims.push_back(500);
        else
            sweep_bond_dims.push_back(sweep_bond_dims.back());
    while (sweep_n_sweeps.size() < n_schedule)
        if (n_schedule == 1)
            sweep_n_sweeps.push_back(maxiter_ == 0 ? 10 : maxiter_);
        else
            sweep_n_sweeps.push_back(6);

    // expand sweep schedule
    int n_total_cnt = 0;
    std::vector<int> bond_dims;
    std::vector<double> noises, davidson_tols;
    for (size_t i = 0; i < n_schedule; i++) {
        n_total_cnt += sweep_n_sweeps[i];
        for (int j = 0; j < sweep_n_sweeps[i]; j++) {
            bond_dims.push_back(sweep_bond_dims[i]);
            noises.push_back(sweep_noises[i]);
            davidson_tols.push_back(sweep_davidson_tols[i]);
        }
    }

    if (maxiter_ == 0)
        maxiter_ = n_total_cnt;

    bool read_initial_guess = options_->get_bool("READ_ACTIVE_WFN_GUESS");

    if (read_initial_guess) {
        fs::path dir_from{"block2.o" + std::to_string(mo_space_info_->size("ACTIVE")) + "." +
                          state_.str_short()};
        fs::path dir_to{impl_->scratch_};
        // copy from current directory for fresh run
        bool fresh = true;
        for (const auto& file : fs::directory_iterator(dir_to)) {
            if (file.path().filename().string().find("KET") != std::string::npos and
                file.path().filename().string().find(state_.str_short()) != std::string::npos) {
                fresh = false;
                break;
            }
        }
        if (fresh) {
            if (fs::exists(dir_from)) {
                for (const auto& file : fs::directory_iterator(dir_from)) {
                    if (file.path().filename().string().find("KET") != std::string::npos and
                        file.path().filename().string().find(state_.str_short()) !=
                            std::string::npos and
                        (!fs::exists(dir_to / file.path().filename()))) {
                        psi::outfile->Printf("\n  Copy %s to scratch directory",
                                             file.path().filename().c_str());
                        fs::copy_file(file.path(), dir_to / file.path().filename(),
                                      fs::copy_options::overwrite_existing);
                    }
                }
            } else {
                read_initial_guess = false;
            }
        }
    }

    // print sweep schedule
    if (print_ >= PrintLevel::Default) {
        print_h2("DMRG Settings");
        psi::outfile->Printf("\n    N sweeps       = ");
        for (auto& x : sweep_n_sweeps)
            psi::outfile->Printf("%10d", x);
        psi::outfile->Printf("\n    Bond dims      = ");
        for (auto& x : sweep_bond_dims)
            psi::outfile->Printf("%10d", x);
        psi::outfile->Printf("\n    Noises         = ");
        for (auto& x : sweep_noises)
            psi::outfile->Printf("%10.2E", x);
        psi::outfile->Printf("\n    Davidson tols  = ");
        for (auto& x : sweep_davidson_tols)
            psi::outfile->Printf("%10.2E", x);
        psi::outfile->Printf("\n    N total sweeps = %10d", maxiter_);
        psi::outfile->Printf("\n    E convergence  = %10.2E", e_conv);
        psi::outfile->Printf("\n    Cutoff         = %10.2E", mps_cutoff);
        psi::outfile->Printf("\n    Initial guess  = %10s", (read_initial_guess ? "load" : "new"));
        psi::outfile->Printf("\n    Verbosity      = %10d", dmrg_verbose);
        psi::outfile->Printf("\n    Stack memory   = %10s",
                             block2::Parsing::to_size_string(impl_->stack_mem_).c_str());
        psi::outfile->Printf("\n    Scratch        = " + impl_->scratch_);
        psi::outfile->Printf("\n");
    }

    // get occupation numbers of orbitals
    double occ_shift = dmrg_options_->get_double("BLOCK2_INITIAL_GUESS");
    std::vector<double> occs;
    if (!read_initial_guess and occ_shift >= 0.0) {
        auto occs_read = dmrg_options_->get_double_list("BLOCK2_INITIAL_GUESS_OCC");
        bool occs_good = (occs_read.size() == (size_t)n_sites);
        for (auto n : occs_read) {
            if (n < 0.0 or n > 2.0) {
                occs_good = false;
                break;
            }
        }
        if (occs_good) {
            occs = occs_read;
        } else {
            occs.resize(n_sites, 0);
            // by default, we use Hartree-Fock occupations as initial guess
            if (print_ >= PrintLevel::Default) {
                if (occs_read.size() != 0)
                    psi::outfile->Printf(
                        "\n    Input BLOCK2_INITIAL_GUESS_OCC seems wrong. Please check!");
                psi::outfile->Printf("\n    Use Hartree-Fock occupancy as initial guess.");
            }
            auto docc_dim = scf_info_->doccpi();
            auto socc_dim = scf_info_->soccpi();
            auto actv_dim = mo_space_info_->dimension("ACTIVE");
            auto adocc_dim = docc_dim - mo_space_info_->dimension("INACTIVE_DOCC");
            for (int h = 0, nirrep = mo_space_info_->nirrep(), shift = 0; h < nirrep; ++h) {
                for (int i = 0; i < adocc_dim[h]; ++i) {
                    occs[i + shift] = 2.0;
                }
                for (int i = 0; i < socc_dim[h]; ++i) {
                    occs[i + shift + adocc_dim[h]] = 1.0;
                }
                shift += actv_dim[h];
            }
        }
        // apply occ shift
        for (int i = 0; i < n_sites; i++) {
            if (occs[i] < 0.75)
                occs[i] += occ_shift;
            else if (occs[i] > 1.25)
                occs[i] -= occ_shift;
        }
        // print occ guess
        if (print_ >= PrintLevel::Default) {
            psi::outfile->Printf("\n    Na = %4d Nb = %4d", na_, nb_);
            psi::outfile->Printf("\n    Occupation numbers for initial guess (with shift %.3f)",
                                 occ_shift);
            auto actv_dim = mo_space_info_->dimension("ACTIVE");
            for (int h = 0, nirrep = mo_space_info_->nirrep(), shift = 0; h < nirrep; ++h) {
                if (actv_dim[h] != 0)
                    psi::outfile->Printf("\n    %4s", mo_space_info_->irrep_label(h).c_str());
                for (int i = 0; i < actv_dim[h]; ++i) {
                    if (i % 10 == 0 and i != 0)
                        psi::outfile->Printf("\n        ");
                    psi::outfile->Printf(" %5.3f", occs[i + shift]);
                }
                shift += actv_dim[h];
            }
            psi::outfile->Printf("\n");
        }
    }

    // initialize mps
    std::string ket_tag = "KET@" + state_.str_short();
    auto ket = read_initial_guess
                   ? impl_->load_mps(ket_tag, nroot_)
                   : impl_->get_random_mps(ket_tag, bond_dims.size() != 0 ? bond_dims[0] : 500, 0,
                                           2, nroot_, occs);

    // do dmrg
    auto mpo = impl_->get_mpo(r, dmrg_verbose);
    energies_ = impl_->dmrg(mpo, ket, maxiter_, e_conv, bond_dims, noises, davidson_tols,
                            dmrg_verbose, mps_cutoff);

    // compute <S^2>
    spin2_ = std::vector<double>(nroot_, 0.0);
    for (size_t ir = 0; ir < nroot_; ir++) {
        if (impl_->is_spin_adapted_)
            // spin2_[ir] = spin * (spin + 2) / 4.0;
            spin2_[ir] = impl_->get_spin_square(ket, nroot_, ir);
        else
            spin2_[ir] = impl_->get_spin_square(ket, nroot_, ir);
    }

    double energy = energies_[root_];
    psi::Process::environment.globals["CURRENT ENERGY"] = energy;
    psi::Process::environment.globals["DMRG ENERGY"] = energy;
    return energy;
}

std::vector<std::shared_ptr<RDMs>>
Block2DMRGSolver::rdms(const std::vector<std::pair<size_t, size_t>>& root_list, int max_rdm_level,
                       RDMsType type) {
    return transition_rdms(root_list, make_shared<Block2DMRGSolver>(*this), max_rdm_level, type);
}

std::vector<std::shared_ptr<RDMs>>
Block2DMRGSolver::transition_rdms(const std::vector<std::pair<size_t, size_t>>& root_list,
                                  std::shared_ptr<ActiveSpaceMethod> method2, int max_rdm_level,
                                  RDMsType type) {

    timer t("BLOCK2 Solver Compute RDMs");

    if (max_rdm_level < 1)
        return std::vector<std::shared_ptr<RDMs>>(root_list.size());

    // system initialization
    bool singlet_embedding = dmrg_options_->get_bool("BLOCK2_SINGLET_EMBEDDING");
    std::vector<int> actv_irreps = mo_space_info_->symmetry("ACTIVE");
    int n_sites = static_cast<int>(mo_space_info_->size("ACTIVE"));
    int n_elec = na_ + nb_;
    int spin = state_.multiplicity() - 1;
    int pg_irrep = state_.irrep();
    impl_->initialize_system(n_sites, n_elec, spin, pg_irrep, actv_irreps, singlet_embedding);

    std::vector<std::shared_ptr<RDMs>> rdms;
    int dmrg_verbose = dmrg_options_->get_int("BLOCK2_VERBOSE");

    // main loop
    for (size_t ir = 0; ir < root_list.size(); ir++) {

        const size_t iroot = root_list[ir].first, jroot = root_list[ir].second;

        // construct operator strings
        std::vector<std::string> exprs;
        if (impl_->is_spin_adapted_) {
            exprs.push_back("(C+D)0");
            for (int il = 2; il <= max_rdm_level; il++)
                exprs.push_back("((C+" + exprs.back() + ")1+D)0");
        } else {
            std::vector<std::string> i_exprs = std::vector<std::string>{"cd", "CD"};
            exprs.insert(exprs.end(), i_exprs.begin(), i_exprs.end());
            for (int il = 2; il <= max_rdm_level; il++) {
                std::vector<std::string> j_exprs;
                for (auto& x : i_exprs)
                    j_exprs.push_back("c" + x + "d");
                j_exprs.push_back("C" + i_exprs.back() + "D");
                i_exprs = j_exprs;
                exprs.insert(exprs.end(), i_exprs.begin(), i_exprs.end());
            }
        }

        // loading MPSs from disk
        // both are called "KET" in the original solver
        std::string bra_tag_sa = "KET@" + state().str_short();
        std::string ket_tag_sa = "KET@" + method2->state().str_short();
        std::string bra_tag =
            "KET@" + block2::Parsing::to_string(iroot) + "@" + state().str_short() + "@TMP";
        std::string ket_tag = "KET@" + block2::Parsing::to_string(jroot) + "@" +
                              method2->state().str_short() + "@TMP";

        if (print_ >= PrintLevel::Default) {
            if (bra_tag == ket_tag)
                print_h2("Computing RDMs for Root No. " + block2::Parsing::to_string(iroot));
            else
                print_h2("Computing Transition RDMs for Root No. " +
                         block2::Parsing::to_string(iroot) + " <-- No. " +
                         block2::Parsing::to_string(jroot));
        }

        std::shared_ptr<void> bra, ket;
        bra = impl_->split_mps(impl_->load_mps(bra_tag_sa, this->nroot()), this->nroot(), iroot,
                               bra_tag);
        if (bra_tag == ket_tag)
            ket = bra;
        else
            ket = impl_->split_mps(impl_->load_mps(ket_tag_sa, method2->nroot()), method2->nroot(),
                                   jroot, ket_tag);

        // compute all rdms
        int bond_dim = 500;
        auto sweep_bond_dims = dmrg_options_->get_int_list("BLOCK2_SWEEP_BOND_DIMS");
        if (sweep_bond_dims.size() != 0)
            bond_dim = sweep_bond_dims.back();
        auto rdm_algo_type = ((block2::ExpectationAlgorithmTypes::SymbolFree) |
                              (block2::ExpectationAlgorithmTypes::Compressed));
        if (dmrg_options_->get_bool("BLOCK2_RDM_LOW_MEM_ALG")) {
            rdm_algo_type = ((block2::ExpectationAlgorithmTypes::SymbolFree) |
                             (block2::ExpectationAlgorithmTypes::Compressed) |
                             (block2::ExpectationAlgorithmTypes::LowMem));
            bond_dim *= 2;
        } else {
            bond_dim = -1; // block2 default: exact
        }
        std::vector<std::shared_ptr<block2::GTensor<double>>> npdms =
            impl_->get_npdm(exprs, ket, bra, 0, dmrg_verbose, rdm_algo_type, bond_dim);

        // currently no need to add the interface for 4rdm
        assert(max_rdm_level <= 3);

        // copy to forte rdm container
        if (impl_->is_spin_adapted_) {
            std::vector<ambit::Tensor> tensors;
            for (int i = 1; i <= max_rdm_level; i++) {
                ambit::Tensor raw_rdm = ambit::Tensor::build(
                    ambit::CoreTensor, "DMRG G" + block2::Parsing::to_string(i),
                    std::vector<size_t>(i * 2, n_sites));
                assert(npdms[i - 1]->size() == raw_rdm.numel());
                std::memcpy(raw_rdm.data().data(), npdms[i - 1]->data->data(),
                            sizeof(double) * npdms[i - 1]->size());
                npdms[i - 1] = nullptr;
                ambit::Tensor sf_rdm = raw_rdm.clone();
                if (i == 1)
                    sf_rdm("ia") = raw_rdm("ia");
                else if (i == 2)
                    sf_rdm("ijab") = raw_rdm("ijba");
                else if (i == 3)
                    sf_rdm("ijkabc") = raw_rdm("ijkcba");
                raw_rdm.reset();
                if (type == RDMsType::spin_free)
                    tensors.push_back(sf_rdm);
                else {
                    if (i == 1) {
                        ambit::Tensor sd1 = RDMs::sf1_to_sd1(sf_rdm);
                        tensors.push_back(sd1);
                        tensors.push_back(sd1);
                    } else if (i == 2) {
                        ambit::Tensor sd2aa = RDMs::sf2_to_sd2aa(sf_rdm);
                        ambit::Tensor sd2ab = RDMs::sf2_to_sd2ab(sf_rdm);
                        tensors.push_back(sd2aa);
                        tensors.push_back(sd2ab);
                        tensors.push_back(sd2aa);
                    } else if (i == 3) {
                        ambit::Tensor sd3aaa = RDMs::sf3_to_sd3aaa(sf_rdm);
                        ambit::Tensor sd3aab = RDMs::sf3_to_sd3aab(sf_rdm);
                        ambit::Tensor sd3abb = RDMs::sf3_to_sd3abb(sf_rdm);
                        tensors.push_back(sd3aaa);
                        tensors.push_back(sd3aab);
                        tensors.push_back(sd3abb);
                        tensors.push_back(sd3aaa);
                    }
                    sf_rdm.reset();
                }
            }
            if (type == RDMsType::spin_free) {
                if (max_rdm_level == 1)
                    rdms.push_back(std::make_shared<RDMsSpinFree>(tensors[0]));
                else if (max_rdm_level == 2)
                    rdms.push_back(std::make_shared<RDMsSpinFree>(tensors[0], tensors[1]));
                else if (max_rdm_level == 3)
                    rdms.push_back(
                        std::make_shared<RDMsSpinFree>(tensors[0], tensors[1], tensors[2]));
            } else {
                if (max_rdm_level == 1)
                    rdms.push_back(std::make_shared<RDMsSpinDependent>(tensors[0], tensors[1]));
                else if (max_rdm_level == 2)
                    rdms.push_back(std::make_shared<RDMsSpinDependent>(
                        tensors[0], tensors[1], tensors[2], tensors[3], tensors[4]));
                else if (max_rdm_level == 3)
                    rdms.push_back(std::make_shared<RDMsSpinDependent>(
                        tensors[0], tensors[1], tensors[2], tensors[3], tensors[4], tensors[5],
                        tensors[6], tensors[7], tensors[8]));
            }
        } else {
            std::vector<ambit::Tensor> tensors;
            for (int i = 1, ix = 0; i <= max_rdm_level; i++) {
                std::vector<ambit::Tensor> sd_rdms(i + 1);
                for (int j = 0; j < i + 1; j++) {
                    ambit::Tensor raw_rdm =
                        ambit::Tensor::build(ambit::CoreTensor,
                                             "DMRG S" + block2::Parsing::to_string(i) + "." +
                                                 block2::Parsing::to_string(j),
                                             std::vector<size_t>(i * 2, n_sites));
                    assert(npdms[ix + j]->size() == raw_rdm.numel());
                    std::memcpy(raw_rdm.data().data(), npdms[ix + j]->data->data(),
                                sizeof(double) * npdms[ix + j]->size());
                    npdms[ix + j] = nullptr;
                    ambit::Tensor sd_rdm = raw_rdm.clone();
                    if (i == 1)
                        sd_rdm("ia") = raw_rdm("ia");
                    else if (i == 2)
                        sd_rdm("ijab") = raw_rdm("ijba");
                    else if (i == 3)
                        sd_rdm("ijkabc") = raw_rdm("ijkcba");
                    raw_rdm.reset();
                    sd_rdms[j] = sd_rdm;
                }
                for (auto& rdm : sd_rdms)
                    tensors.push_back(rdm);
                ix += i + 1;
            }
            if (max_rdm_level == 1)
                rdms.push_back(std::make_shared<RDMsSpinDependent>(tensors[0], tensors[1]));
            else if (max_rdm_level == 2)
                rdms.push_back(std::make_shared<RDMsSpinDependent>(
                    tensors[0], tensors[1], tensors[2], tensors[3], tensors[4]));
            else if (max_rdm_level == 3)
                rdms.push_back(std::make_shared<RDMsSpinDependent>(
                    tensors[0], tensors[1], tensors[2], tensors[3], tensors[4], tensors[5],
                    tensors[6], tensors[7], tensors[8]));
            if (type == RDMsType::spin_free) {
                if (max_rdm_level == 1)
                    rdms.back() = std::make_shared<RDMsSpinFree>(rdms.back()->SF_G1());
                else if (max_rdm_level == 2)
                    rdms.back() =
                        std::make_shared<RDMsSpinFree>(rdms.back()->SF_G1(), rdms.back()->SF_G2());
                else if (max_rdm_level == 3)
                    rdms.back() = std::make_shared<RDMsSpinFree>(
                        rdms.back()->SF_G1(), rdms.back()->SF_G2(), rdms.back()->SF_G3());
            }
        }

        if (print_ >= PrintLevel::Default && bra_tag == ket_tag)
            print_natural_orbitals(mo_space_info_, rdms.back());
    }

    return rdms;
}

std::vector<double> Block2DMRGSolver::compute_complementary_H2caa_overlap(
    const std::vector<size_t>& roots, ambit::Tensor Tbra, ambit::Tensor Tket,
    const std::vector<int>& p_syms, const std::string& name, bool load) {
    /// |bra_{pσ}> = \sum_{uvw} Tbra_{pwuv} \sum_{σ1} w^+_{σ1} v_{σ1} u_{σ} |Ψ>
    /// |ket_{pσ}> = \sum_{uvw} Tket_{pwuv} \sum_{σ1} w^+_{σ1} v_{σ1} u_{σ} |Ψ>
    /// energy <- \sum_{p} \sum_{σ} <bra_{pσ}|ket_{pσ}>

    auto dims_bra = Tbra.dims();
    auto dims_ket = Tket.dims();
    auto nactv = mo_space_info_->size("ACTIVE");
    auto np = p_syms.size();

    if (dims_bra.size() != 4)
        throw std::runtime_error("Invalid Tensor for bra: Dimension must be 4!");
    if (dims_ket.size() != 4)
        throw std::runtime_error("Invalid Tensor for ket: Dimension must be 4!");
    if ((dims_bra[1] != nactv) or (dims_bra[2] != nactv) or (dims_bra[3] != nactv))
        throw std::runtime_error("Invalid Tensor for bra: Inconsistent active indices");
    if ((dims_ket[1] != nactv) or (dims_ket[2] != nactv) or (dims_ket[3] != nactv))
        throw std::runtime_error("Invalid Tensor for ket: Inconsistent active indices");
    if (dims_bra[0] != dims_ket[0] or dims_bra[0] != np)
        throw std::runtime_error("Invalid non-active indices for bra and ket Tensors");

    auto nroots = roots.size();
    std::vector<double> out(nroots, 0.0);

    auto& Tbra_data = Tbra.data();
    auto& Tket_data = Tket.data();

    auto na1 = static_cast<int>(nactv);
    auto na2 = na1 * na1;
    auto na3 = na2 * na1;
    const std::vector<int> tshape{na1, na1, na1};
    const std::vector<size_t> tstride{nactv * nactv, 1, nactv};

    // read bond dimention
    int bond_dim = 500;
    auto sweep_bond_dims = dmrg_options_->get_int_list("BLOCK2_SWEEP_BOND_DIMS");
    if (sweep_bond_dims.size() != 0)
        bond_dim = sweep_bond_dims.back();
    std::vector<block2::ubond_t> ket0_bond_dims(1, bond_dim);
    std::vector<block2::ubond_t> bra_bond_dims(
        1, dmrg_options_->get_int("DSRG_3RDM_BLOCK2_CPS_BOND_DIMENSION"));
    std::vector<double> noises{0.0};

    // system initialization
    auto integral_cutoff = dmrg_options_->get_double("BLOCK2_INTERGRAL_CUTOFF");
    bool singlet_embedding = dmrg_options_->get_bool("BLOCK2_SINGLET_EMBEDDING");
    auto actv_irreps = mo_space_info_->symmetry("ACTIVE");
    int n_sites = static_cast<int>(mo_space_info_->size("ACTIVE"));
    int n_elec = na_ + nb_;
    int spin = state_.multiplicity() - 1;
    int pg_irrep = state_.irrep();
    impl_->initialize_system(n_sites, n_elec, spin, pg_irrep, actv_irreps, singlet_embedding);

    int dmrg_verbose = dmrg_options_->get_int("BLOCK2_VERBOSE");

    for (size_t ir = 0; ir < nroots; ++ir) {
        double value = 0.0;

        // loading MPSs from disk
        std::string ket_tag_sa = "KET@" + state().str_short();
        std::string ket_tag =
            "KET@" + block2::Parsing::to_string(ir) + "@" + state().str_short() + "@TMP";
        auto ket = impl_->split_mps(impl_->load_mps(ket_tag_sa, this->nroot()), this->nroot(), ir,
                                    ket_tag);

        if (impl_->is_spin_adapted_) {
            auto ket0 = std::static_pointer_cast<block2::MPS<block2::SU2, double>>(ket);
            // auto bond_dim = ket0->info->get_max_bond_dimension();
            auto bond_dim = ket0_bond_dims[0];

            for (size_t p = 0; p < np; ++p) {
                auto bra_expr = impl_->expr_builder();
                bra_expr->exprs.push_back("((C+D)0+D)1");
                bra_expr->add_sum_term(Tbra_data.data() + p * na3, na3, tshape, tstride,
                                       integral_cutoff, 2.0, actv_irreps, {}, p_syms[p]);
                bra_expr = bra_expr->adjust_order();
                if (bra_expr->exprs.size() == 0)
                    continue;

                if (print_ > PrintLevel::Default)
                    psi::outfile->Printf("\n orbital %2zu", p);

                auto bmpo = std::static_pointer_cast<block2::MPO<block2::SU2, double>>(
                    impl_->get_mpo(bra_expr, dmrg_verbose));
                auto bq = bmpo->op->q_label + ket0->info->target;
                auto bra_left_vacuum = ket0->info->left_dims_fci[0]->quanta[0] + bmpo->left_vacuum;

                auto ket_expr = impl_->expr_builder();
                ket_expr->exprs.push_back("((C+D)0+D)1");
                ket_expr->add_sum_term(Tket_data.data() + p * na3, na3, tshape, tstride,
                                       integral_cutoff, 2.0, actv_irreps, {}, p_syms[p]);
                ket_expr = ket_expr->adjust_order();
                if (ket_expr->exprs.size() == 0)
                    continue;

                auto kmpo = std::static_pointer_cast<block2::MPO<block2::SU2, double>>(
                    impl_->get_mpo(ket_expr, dmrg_verbose));

                auto vacuum = impl_->driver_su2_->vacuum;

                for (int j = 0; j < bra_left_vacuum.count(); ++j) {
                    if (print_ > PrintLevel::Default)
                        psi::outfile->Printf("  j = %2d", j);

                    auto tag = ket0->info->tag + "@BRA." + name + std::to_string(p) + "." +
                               std::to_string(j);
                    std::shared_ptr<block2::MPS<block2::SU2, double>> bra = nullptr;
                    auto binfo = std::make_shared<block2::MPSInfo<block2::SU2>>(
                        na1, vacuum, bq, impl_->driver_su2_->ghamil->basis);

                    bra = std::make_shared<block2::MPS<block2::SU2, double>>(na1, ket0->center,
                                                                             ket0->dot);
                    if (load) {
                        binfo->load_data(impl_->scratch_ + "/" + tag + "-mps_info.bin");
                        binfo->tag = tag;
                        binfo->load_mutable();
                        binfo->set_bond_dimension_fci(bra_left_vacuum[j], vacuum);
                        binfo->set_bond_dimension(bond_dim);
                        binfo->bond_dim = bond_dim;
                        if (binfo->get_max_bond_dimension() == 0)
                            continue;
                        bra->initialize(binfo);
                        bra->load_data();
                        bra->load_mutable();
                    } else {
                        binfo->tag = tag;
                        binfo->set_bond_dimension_fci(bra_left_vacuum[j], vacuum);
                        binfo->set_bond_dimension(bond_dim);
                        binfo->bond_dim = bond_dim;

                        if (binfo->get_max_bond_dimension() == 0)
                            continue;

                        bra->initialize(binfo);
                        bra->random_canonicalize();
                        bra->tensors[bra->center]->normalize();
                        bra->save_mutable();
                        binfo->save_mutable();
                        binfo->save_data(impl_->scratch_ + "/" + tag + "-mps_info.bin");
                        bra->save_data();

                        auto bref = ket0->deep_copy("DSRG-BRA@TMP");
                        auto bme = std::make_shared<
                            block2::MovingEnvironment<block2::SU2, double, double>>(bmpo, bra, bref,
                                                                                    "DSRG-CPS1");
                        bme->delayed_contraction = block2::OpNamesSet::normal_ops();
                        bme->cached_contraction = true;
                        bme->init_environments(true);

                        auto bcps = std::make_shared<block2::Linear<block2::SU2, double, double>>(
                            bme, bra_bond_dims, std::vector<block2::ubond_t>{bref->info->bond_dim},
                            noises);
                        bcps->iprint = 2;
                        bcps->noise_type = block2::NoiseTypes::ReducedPerturbative;
                        bcps->eq_type = block2::EquationTypes::PerturbativeCompression;
                        bcps->solve(2 * maxiter_, bra->center == 0, 1.0e-8);
                        if (bra->center != ket0->center)
                            bcps->solve(1, ket0->center != 0);
                    }

                    auto pvalue = impl_->driver_su2_->expectation(bra, kmpo, ket0, true, bond_dim);

                    if (print_ > PrintLevel::Default)
                        psi::outfile->Printf(" pvalue = %20.15f", pvalue);
                    value += pvalue;
                }
            }
        } else {
            auto ket0 = std::static_pointer_cast<block2::MPS<block2::SZ, double>>(ket);

            for (size_t p = 0; p < np; ++p) {
                if (print_ > PrintLevel::Default)
                    psi::outfile->Printf("\n orbital %zu", p);
                for (size_t sigma = 0; sigma != 2; ++sigma) {
                    auto bra_expr = impl_->expr_builder();
                    if (sigma == 0) {
                        bra_expr->exprs.push_back("cdd");
                        bra_expr->exprs.push_back("CDd");
                    } else {
                        bra_expr->exprs.push_back("cdD");
                        bra_expr->exprs.push_back("CDD");
                    }
                    bra_expr->add_sum_term(Tbra_data.data() + p * na3, na3, tshape, tstride,
                                           integral_cutoff, 1.0, actv_irreps, {}, p_syms[p]);
                    bra_expr->add_sum_term(Tbra_data.data() + p * na3, na3, tshape, tstride,
                                           integral_cutoff, 1.0, actv_irreps, {}, p_syms[p]);
                    bra_expr = bra_expr->adjust_order();
                    if (bra_expr->exprs.size() == 0)
                        continue;

                    auto bmpo = std::static_pointer_cast<block2::MPO<block2::SZ, double>>(
                        impl_->get_mpo(bra_expr, dmrg_verbose));
                    auto bq = bmpo->op->q_label + ket0->info->target;
                    std::string btag = "BRA." + std::to_string(p);
                    btag += (sigma == 0 ? ".A" : ".B");
                    auto bra =
                        impl_->driver_sz_->get_random_mps(btag, bond_dim, ket0->center, 2, bq);

                    auto bme = make_shared<block2::MovingEnvironment<block2::SZ, double, double>>(
                        bmpo, bra, ket0, "MULT");
                    bme->delayed_contraction = block2::OpNamesSet::normal_ops();
                    bme->cached_contraction = true;
                    bme->init_environments(true);

                    auto bcps = std::make_shared<block2::Linear<block2::SZ, double, double>>(
                        bme, bra_bond_dims, ket0_bond_dims, noises);
                    bcps->solve(maxiter_, true, 1.0e-8);

                    auto ket_expr = impl_->expr_builder();
                    if (sigma == 0) {
                        ket_expr->exprs.push_back("cdd");
                        ket_expr->exprs.push_back("CDd");
                    } else {
                        ket_expr->exprs.push_back("cdD");
                        ket_expr->exprs.push_back("CDD");
                    }
                    ket_expr->add_sum_term(Tket_data.data() + p * na3, na3, tshape, tstride,
                                           integral_cutoff, 1.0, actv_irreps, {}, p_syms[p]);
                    ket_expr->add_sum_term(Tket_data.data() + p * na3, na3, tshape, tstride,
                                           integral_cutoff, 1.0, actv_irreps, {}, p_syms[p]);
                    ket_expr = ket_expr->adjust_order();

                    auto kmpo = std::static_pointer_cast<block2::MPO<block2::SZ, double>>(
                        impl_->get_mpo(ket_expr, dmrg_verbose));
                    if (ket_expr->exprs.size() == 0)
                        continue;

                    auto pvalue =
                        impl_->driver_sz_->expectation(bra, kmpo, ket0, true, 2 * bond_dim);
                    if (print_ > PrintLevel::Default)
                        psi::outfile->Printf(" %s = %20.15f", sigma == 0 ? "alpha" : "beta",
                                             pvalue);
                    value += pvalue;
                }
            }
        }
        out[ir] = value;
    }
    return out;
}

void Block2DMRGSolver::set_options(std::shared_ptr<ForteOptions> options) {
    dmrg_options_ = options;
}

void Block2DMRGSolver::print_natural_orbitals(std::shared_ptr<MOSpaceInfo> mo_space_info,
                                              std::shared_ptr<RDMs> rdms) {
    if (print_ >= PrintLevel::Default)
        print_h2("NATURAL ORBITALS");
    psi::Dimension active_dim = mo_space_info->dimension("ACTIVE");
    int n_sites = static_cast<int>(mo_space_info_->size("ACTIVE"));
    auto opdm = std::make_shared<psi::Matrix>(new psi::Matrix("OPDM", active_dim, active_dim));
    int nirrep = static_cast<int>(mo_space_info->nirrep());

    ambit::Tensor sf_opdm = rdms->SF_G1();

    int offset = 0;
    for (int h = 0; h < nirrep; h++) {
        for (int u = 0; u < active_dim[h]; u++)
            for (int v = 0; v < active_dim[h]; v++)
                opdm->set(h, u, v, sf_opdm.data()[(u + offset) * n_sites + v + offset]);
        offset += active_dim[h];
    }

    auto OCC = std::make_shared<psi::Vector>("Occupation numbers", active_dim);
    auto NO = std::make_shared<psi::Matrix>("MO -> NO transformation", active_dim, active_dim);

    opdm->diagonalize(NO, OCC, psi::descending);
    std::vector<std::pair<double, std::pair<int, int>>> vec_irrep_occupation;
    for (int h = 0; h < nirrep; h++) {
        for (int u = 0; u < active_dim[h]; u++) {
            auto irrep_occ = std::make_pair(OCC->get(h, u), std::make_pair(h, u + 1));
            vec_irrep_occupation.push_back(irrep_occ);
        }
    }
    std::sort(vec_irrep_occupation.begin(), vec_irrep_occupation.end(),
              std::greater<std::pair<double, std::pair<int, int>>>());

    size_t count = 0;
    psi::outfile->Printf("\n    ");
    for (auto vec : vec_irrep_occupation) {
        psi::outfile->Printf(" %4d%-4s%11.6f  ", vec.second.second,
                             mo_space_info->irrep_label(vec.second.first).c_str(), vec.first);
        if (count++ % 3 == 2 && count != vec_irrep_occupation.size())
            psi::outfile->Printf("\n    ");
    }
    psi::outfile->Printf("\n");
}

void Block2DMRGSolver::dump_wave_function(const std::string&) {
    // create a folder for the wave function
    fs::path folder{"block2.o" + std::to_string(mo_space_info_->size("ACTIVE")) + "." +
                    state_.str_short()};
    fs::create_directory(folder);
    // copy KET files to the folder
    for (const auto& file : fs::directory_iterator(impl_->scratch_)) {
        if (file.path().filename().string().find("KET") != std::string::npos and
            file.path().filename().string().find(state_.str_short()) != std::string::npos) {
            fs::copy_file(file.path(), folder / file.path().filename(),
                          fs::copy_options::overwrite_existing);
        }
    }
}
} // namespace forte

#endif // HAVE_BLOCK2
