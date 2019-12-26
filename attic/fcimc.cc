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

#include <cmath>

#include "psi4/libpsi4util/process.h"
#include "psi4/libmints/molecule.h"
#include "psi4/libmints/wavefunction.h"
#include "psi4/libpsio/psio.hpp"

#include "psi4/libqt/qt.h"

#include "fci/fci_vector.h"
#include "fcimc.h"
#include "helpers/mo_space_info.h"
#include "helpers/helpers.h"

using namespace psi;

namespace psi {
namespace forte {

std::default_random_engine generator_;
std::uniform_real_distribution<double> distribution_real_(0.0, 1.0);
std::uniform_int_distribution<size_t> distribution_int_;
auto rand_real = std::bind(distribution_real_, generator_);
auto rand_int = std::bind(distribution_int_, generator_);

std::pair<size_t, size_t> generate_ind_random_pair(size_t range) {
    size_t r1 = rand_int() % range;
    size_t r2 = rand_int() % (range - 1);
    if (r2 >= r1)
        r2++;
    return std::make_pair(r1, r2);
}

#ifdef _OPENMP
#include <omp.h>
bool FCIQMC::have_omp_ = true;
#else
#define omp_get_max_threads() 1
#define omp_get_thread_num() 0
bool FCIQMC::have_omp_ = false;
#endif

FCIQMC::FCIQMC(SharedWavefunction ref_wfn, Options& options, std::shared_ptr<ForteIntegrals> ints,
               std::shared_ptr<MOSpaceInfo> mo_space_info)
    : Wavefunction(options), ints_(ints), mo_space_info_(mo_space_info)
// fciInts_(ints, mo_space_info)
{
    shallow_copy(ref_wfn);
    reference_wavefunction_ = ref_wfn;
    startup();
}

FCIQMC::~FCIQMC() {}
// Initialize static pointer to ints
std::shared_ptr<FCIIntegrals> FCIQMC::fci_ints_ = nullptr;

void FCIQMC::startup() {
    // Connect the integrals to the determinant class
    fci_ints_ = std::make_shared<FCIIntegrals>(ints_, mo_space_info_->corr_absolute_mo("ACTIVE"),
                                               mo_space_info_->corr_absolute_mo("RESTRICTED_DOCC"));

    auto active_mo = mo_space_info_->corr_absolute_mo("ACTIVE");
    ambit::Tensor tei_active_aa = ints_->aptei_aa_block(active_mo, active_mo, active_mo, active_mo);
    ambit::Tensor tei_active_ab = ints_->aptei_ab_block(active_mo, active_mo, active_mo, active_mo);
    ambit::Tensor tei_active_bb = ints_->aptei_bb_block(active_mo, active_mo, active_mo, active_mo);
    fci_ints_->set_active_integrals(tei_active_aa, tei_active_ab, tei_active_bb);
    fci_ints_->compute_restricted_one_body_operator();

    // The number of correlated molecular orbitals
    ncmo_ = mo_space_info_->corr_absolute_mo("ACTIVE").size();
    ncmopi_ = mo_space_info_->dimension("ACTIVE");

    // Overwrite the frozen orbitals arrays
    frzcpi_ = mo_space_info_->dimension("FROZEN_DOCC");
    frzvpi_ = mo_space_info_->dimension("FROZEN_UOCC");

    // Create the array with mo symmetry
    for (int h = 0; h < nirrep_; ++h) {
        if (h == 0) {
            cume_ncmopi_.push_back(0);
        } else {
            cume_ncmopi_.push_back(cume_ncmopi_[h - 1] + ncmopi_[h - 1]);
        }
        for (int p = 0; p < ncmopi_[h]; ++p) {
            mo_symmetry_.push_back(h);
        }
    }
    cume_ncmopi_.push_back(ncmo_);

    // compute the number of irrep combination catagories per alpha beta
    // combination
    cume_excit_irrep_[0] = nirrep_ * nirrep_ * nirrep_;
    cume_excit_irrep_[1] = cume_excit_irrep_[0] + nirrep_ * nirrep_;
    cume_excit_irrep_[2] = cume_excit_irrep_[1] + nirrep_ * nirrep_ * (nirrep_ - 1) / 2;
    cume_excit_irrep_[3] = cume_excit_irrep_[2] + nirrep_ * nirrep_;
    cume_excit_irrep_[4] = cume_excit_irrep_[3] + nirrep_ * nirrep_ * (nirrep_ - 1) / 2;
    cume_excit_irrep_[5] = cume_excit_irrep_[4] + nirrep_;
    cume_excit_irrep_[6] = cume_excit_irrep_[5] + nirrep_;

    nuclear_repulsion_energy_ =
        molecule_->nuclear_repulsion_energy(reference_wavefunction_->get_dipole_field_strength());

    wavefunction_symmetry_ = 0;
    if (options_["ROOT_SYM"].has_changed()) {
        wavefunction_symmetry_ = options_.get_int("ROOT_SYM");
    }

    // Build the reference determinant and compute its energy
    std::vector<bool> occupation(2 * ncmo_, false);
    int cumidx = 0;
    for (int h = 0; h < nirrep_; ++h) {
        for (int i = 0; i < doccpi_[h] - frzcpi_[h]; ++i) {
            occupation[i + cumidx] = true;
            occupation[ncmo_ + i + cumidx] = true;
        }
        for (int i = 0; i < soccpi_[h]; ++i) {
            occupation[i + cumidx] = true;
        }
        cumidx += ncmopi_[h];
    }
    Determinant reference_determinant(occupation);
    reference_ = reference_determinant;

    outfile->Printf("\n  The reference determinant is:\n");
    outfile->Printf("\n  %s", reference_.str().c_str());

    time_step_ = options_.get_double("TAU");
    maxiter_ = options_.get_int("MAXITER");
    start_num_walkers_ = options_.get_double("START_NUM_WALKERS");
    do_shift_ = options_.get_bool("USE_SHIFT");
    shift_freq_ = options_.get_int("SHIFT_FREQ");
    shift_damp_ = options_.get_double("SHIFT_DAMP");
    shift_num_walkers_ = options_.get_double("SHIFT_NUM_WALKERS");
    death_parent_only_ = options_.get_bool("DEATH_PARENT_ONLY");
    energy_estimate_freq_ = options_.get_int("VAR_ENERGY_ESTIMATE_FREQ");
    print_freq_ = options_.get_int("PRINT_FREQ");
    //    outfile->Printf("\nPrint Freq is %d", print_freq_);
    use_initiator_ = options_.get_bool("USE_INITIATOR");
    initiator_na_ = options_.get_double("INITIATOR_NA");
    if (options_.get_str("SPAWN_TYPE") == "RANDOM") {
        spawn_type_ = random;
    } else if (options_.get_str("SPAWN_TYPE") == "ALL") {
        spawn_type_ = all;
    } else if (options_.get_str("SPAWN_TYPE") == "GROUND_AND_RANDOM") {
        spawn_type_ = ground_and_random;
    }
    // Read options
    //    nroot_ = options_.get_int("NROOT");
    //    spawning_threshold_ = options_.get_double("SPAWNING_THRESHOLD");
    //    initial_guess_spawning_threshold_ =
    //    options_.get_double("GUESS_SPAWNING_THRESHOLD");

    //    e_convergence_ = options_.get_double("E_CONVERGENCE");
    //    energy_estimate_threshold_ =
    //    options_.get_double("ENERGY_ESTIMATE_THRESHOLD");

    //    adaptive_beta_ = options_.get_bool("ADAPTIVE_BETA");
    //    fast_variational_estimate_ = options_.get_bool("FAST_EVAR");

    //    do_simple_prescreening_ = options_.get_bool("SIMPLE_PRESCREENING");
    //    do_dynamic_prescreening_ = options_.get_bool("DYNAMIC_PRESCREENING");

    //    if (options_.get_str("PROPAGATOR") == "LINEAR"){
    //        propagator_ = LinearPropagator;
    //        propagator_description_ = "Linear";
    //    }else if (options_.get_str("PROPAGATOR") == "QUADRATIC"){
    //        propagator_ = QuadraticPropagator;
    //        propagator_description_ = "Quadratic";
    //    }else if (options_.get_str("PROPAGATOR") == "CUBIC"){
    //        propagator_ = CubicPropagator;
    //        propagator_description_ = "Cubic";
    //    }else if (options_.get_str("PROPAGATOR") == "QUARTIC"){
    //        propagator_ = QuarticPropagator;
    //        propagator_description_ = "Quartic";
    //    }else if (options_.get_str("PROPAGATOR") == "POWER"){
    //        propagator_ = PowerPropagator;
    //        propagator_description_ = "Power";
    //    }else if (options_.get_str("PROPAGATOR") == "POSITIVE"){
    //        propagator_ = PositivePropagator;
    //        propagator_description_ = "Positive";
    //    }
    shift_ = 0.0;
    nWalkers_ = 0.0;
    num_threads_ = omp_get_max_threads();
}

void FCIQMC::print_info() {
    // Print a summary
    std::vector<std::pair<std::string, int>> calculation_info{
        {"Symmetry", wavefunction_symmetry_},
        //        {"Number of roots",nroot_},
        //        {"Root used for properties",options_.get_int("ROOT")},
        {"Maximum number of steps", maxiter_},
        {"Var. Energy estimation frequency", energy_estimate_freq_},
        {"Print info frequency", print_freq_},
        {"Number of threads", num_threads_}};

    std::vector<std::pair<std::string, double>> calculation_info_double{
        {"Time step (beta)", time_step_},
        {"start_num_walkers", start_num_walkers_},
        {"shift_num_walkers", shift_num_walkers_},
        {"initiator_na", initiator_na_}
        //        {"Spawning threshold",spawning_threshold_},
        //        {"Initial guess spawning
        //        threshold",initial_guess_spawning_threshold_},
        //        {"Convergence threshold",e_convergence_},
        //        {"Prescreening tollerance
        //        factor",prescreening_tollerance_factor_},
        //        {"Energy estimate tollerance",energy_estimate_threshold_}
    };
    std::vector<std::pair<std::string, std::string>> calculation_info_string{
        //        {"Propagator type",propagator_description_},
        //        {"Adaptive time step",adaptive_beta_ ? "YES" : "NO"},
        //        {"Shift the energy",do_shift_ ? "YES" : "NO"},
        //        {"Prescreen spawning",do_simple_prescreening_ ? "YES" : "NO"},
        //        {"Dynamic prescreening",do_dynamic_prescreening_ ? "YES" :
        //        "NO"},
        //        {"Fast variational estimate",fast_variational_estimate_ ?
        //        "YES" : "NO"},
        {"Using shift", do_shift_ ? "YES" : "NO"},
        {"Using initiator", use_initiator_ ? "YES" : "NO"},
        {"death parent only", death_parent_only_ ? "YES" : "NO"},
        {"spawn type", options_.get_str("SPAWN_TYPE")},
        {"Using OpenMP", have_omp_ ? "YES" : "NO"},
    };

    // Print some information
    outfile->Printf("\n\n  ==> Calculation Information <==\n");
    for (auto& str_dim : calculation_info) {
        outfile->Printf("\n    %-39s %10d", str_dim.first.c_str(), str_dim.second);
    }
    for (auto& str_dim : calculation_info_double) {
        outfile->Printf("\n    %-39s %10.3e", str_dim.first.c_str(), str_dim.second);
    }
    for (auto& str_dim : calculation_info_string) {
        outfile->Printf("\n    %-39s %10s", str_dim.first.c_str(), str_dim.second.c_str());
    }
}

double FCIQMC::compute_energy() {
    timer_on("FCIQMC:Energy");
    outfile->Printf("\n\n\t  ---------------------------------------------------------");
    outfile->Printf("\n\t      Full Configuration Interaction QUANTUM MONTE-CARLO");
    outfile->Printf("\n\t         by Tianyuan Zhang and Francesco A. Evangelista");
    outfile->Printf("\n\t                    %4d thread(s) %s", num_threads_,
                    have_omp_ ? "(OMP)" : "");
    outfile->Printf("\n\t  ---------------------------------------------------------");

    // Print a summary of the options
    print_info();

    //    // Build the reference determinant and compute its energy
    //    std::vector<bool> occupation_a(ncmo_,false);
    //    std::vector<bool> occupation_b(ncmo_,false);
    //    int cumidx = 0;
    //    for (int h = 0; h < nirrep_; ++h){
    //        for (int i = 0; i < doccpi_[h]; ++i){
    //            occupation_a[i + cumidx] = true;
    //            occupation_b[i + cumidx] = true;
    //        }
    //        for (int i = 0; i < soccpi_[h]; ++i){
    //            occupation_a[i + doccpi_[h] + cumidx] = true;
    //        }
    //        cumidx += nmopi_[h];
    //    }
    outfile->Printf("\n  %s", reference_.str().c_str());

    Ehf_ = fci_ints_->energy(reference_);

    std::tie(nsa_, nsb_, ndaa_, ndab_, ndbb_) = compute_pgen(reference_);
    sumgen_ = nsa_ + nsb_ + ndaa_ + ndab_ + ndbb_;
    cume_sumgen_[0] = nsa_;
    cume_sumgen_[1] = nsa_ + nsb_;
    cume_sumgen_[2] = nsa_ + nsb_ + ndab_;
    cume_sumgen_[3] = nsa_ + nsb_ + ndab_ + ndaa_;
    cume_sumgen_[4] = sumgen_;

    double nre =
        molecule_->nuclear_repulsion_energy(reference_wavefunction_->get_dipole_field_strength());
    outfile->Printf("\nnuclear_repulsion_energy:%lf, Ehf:%lf", nre, Ehf_);

    // Create the initial walker population
    std::map<Determinant, double> walkers;
    walkers[reference_] = start_num_walkers_;

    bool shift_flag = false;
    double pre_nWalker = 0;
    size_t pre_iter = 0;

    std::vector<double> shifts;
    std::vector<double> Eprojs;

    outfile->Printf("\n\n  ==> FCIQMC Iterations <==");

    outfile->Printf("\n\n "
                    "----------------------------------------------------------"
                    "----------------------------------------------------------"
                    "-----------------------------------------------------");
    outfile->Printf("\n   Steps  Beta/Eh   Nwalkers      Ndets      Proj. "
                    "Energy/Eh   Avg.Proj.Energy/Eh             Shift/Eh       "
                    " Avg. Shift/Eh    Shifted Energy/Eh  Avg.Shifted "
                    "Energy/Eh");
    outfile->Printf("\n "
                    "----------------------------------------------------------"
                    "----------------------------------------------------------"
                    "-----------------------------------------------------");

    for (iter_ = 1; iter_ <= maxiter_; ++iter_) {
        if (!shift_flag && do_shift_ && nWalkers_ > shift_num_walkers_) {
            shift_flag = true;
            pre_nWalker = nWalkers_;
            pre_iter = iter_;
        }

        if (do_shift_ && shift_flag && iter_ % shift_freq_ == 0) {
            adjust_shift(pre_nWalker, pre_iter);
            pre_nWalker = nWalkers_;
            pre_iter = iter_;
            shifts.push_back(shift_);
        }

        std::map<Determinant, double> new_walkers;

        // Step #1.  Spawning
        timer_on("FCIQMC:Spawn");

        spawn(walkers, new_walkers);
        //        spawn_generative(walkers,new_walkers);
        timer_off("FCIQMC:Spawn");
        //        outfile->Printf("\nRef walkers: %f after Spawn",
        //        new_walkers[reference_]);

        if (death_parent_only_) {
            // Step #2.  Death/Clone
            timer_on("FCIQMC:Death_Clone");
            death_clone(walkers, shift_);
            timer_off("FCIQMC:Death_Clone");

            //            outfile->Printf("\nRef walkers: %f after Death/Clone",
            //            walkers[reference_]);

            // Step #3.  Merge parents and spawned
            timer_on("FCIQMC:Merge");
            merge(walkers, new_walkers);
            timer_off("FCIQMC:Merge");

            //            outfile->Printf("\nRef walkers: %f after merge",
            //            walkers[reference_]);
        } else {
            // Step #3.  Merge parents and spawned
            timer_on("FCIQMC:Merge");
            merge(walkers, new_walkers);
            timer_off("FCIQMC:Merge");

            //            outfile->Printf("\nRef walkers: %f after merge",
            //            walkers[reference_]);

            // Step #2.  Death/Clone
            timer_on("FCIQMC:Death_Clone");
            death_clone(walkers, shift_);
            timer_off("FCIQMC:Death_Clone");

            //            outfile->Printf("\nRef walkers: %f after Death/Clone",
            //            walkers[reference_]);
        }

        // Step #3.  annihilation
        timer_on("FCIQMC:Annihilation");
        annihilate(walkers, new_walkers);
        timer_off("FCIQMC:Annihilation");

        // calculate iter info
        count_walkers(walkers);
        Eprojs.push_back(compute_proj_energy(reference_, walkers));

        if (iter_ % print_freq_ == 0) {
            //            compute_var_energy(walkers);
            compute_avg_Eproj(Eprojs);
            compute_avg_shift(shifts);
            print_iter_info(iter_);
        }
    }

    outfile->Printf("\n "
                    "----------------------------------------------------------"
                    "----------------------------------------------------------"
                    "-----------------------------------------------------");
    compute_avg_Eproj(Eprojs);
    compute_avg_shift(shifts);
    compute_err_Eproj(Eprojs);
    compute_err_shift(shifts);

    outfile->Printf("\n\n  ==> Post-Iterations <==\n");
    outfile->Printf("\n  * FCIQMC Avg projective Energy     = %.12f Eh    "
                    "Error Estimation = %.12f Eh",
                    AvgEproj_, ErrEproj_);
    outfile->Printf("\n  * FCIQMC Avg shifted Energy        = %.12f Eh    "
                    "Error Estimation = %.12f Eh",
                    AvgShift_ + Ehf_ + nuclear_repulsion_energy_, ErrShift_);
    outfile->Printf("\n\n  * Numbef of walkers                = %zu", (size_t)nWalkers_);
    outfile->Printf("\n\n  * Size of CI space                 = %zu", (size_t)nDets_);
    //    outfile->Printf("\n  * Spawning events/iteration          =
    //    %zu",nspawned_);
    //    outfile->Printf("\n  * Determinants that do not spawn     =
    //    %zu",nzerospawn_);

    //    outfile->Printf("\n\n  %s: %f s","Adaptive Path-Integral CI (bitset)
    //    ran in ",t_apici.get());

    compute_var_energy(walkers);
    outfile->Printf("\n\n  * FCIQMC Variational Energy        = %.12f Eh", Evar_);

    //    outfile->Printf("\n\nFCIQMC calculation ended with:");
    //    outfile->Printf("\nFinal iter:");
    ////    print_iter_info(--iter_);
    //    outfile->Printf("\nProjectional Energy info:");
    ////    print_Eproj_info(Eprojs);
    //    if (do_shift_) {
    //        outfile->Printf("\nShift info:");
    ////        print_shift_info(shifts);
    //    }
    timer_off("FCIQMC:Energy");
    Process::environment.globals["CURRENT ENERGY"] = Evar_;
    return Evar_;
}

void FCIQMC::adjust_shift(double pre_nWalker, size_t pre_iter) {
    //    shift_ = shift_ -
    //    (shift_damp_/((iter_-pre_iter)*time_step_))*std::log(nWalkers_/shift_num_walkers_);
    shift_ = shift_ -
             (shift_damp_ / ((iter_ - pre_iter) * time_step_)) * std::log(nWalkers_ / pre_nWalker);
    //    outfile->Printf("\niter=%d,pre_iter=%d,nWalkers=%.0f,
    //    pre_nWalkers=%.0f, Shift adjusted to %.12lf",iter_, pre_iter,
    //    nWalkers_, pre_nWalker,shift_);
}

void FCIQMC::spawn_generative(walker_map& walkers, walker_map& new_walkers) {
    for (auto& det_coef : walkers) {
        const Determinant& det = det_coef.first;
        //        det.print();
        double coef = det_coef.second;
        size_t nid = std::round(std::fabs(coef));
        if (use_initiator_ && nid < std::round(initiator_na_))
            continue;
        const std::vector<int> aocc = det.get_alfa_occ(ncmo_);
        const std::vector<int> bocc = det.get_beta_occ(ncmo_);
        const std::vector<int> avir = det.get_alfa_vir(ncmo_);
        const std::vector<int> bvir = det.get_beta_vir(ncmo_);
        switch (spawn_type_) {
        case random:
            for (size_t detW = 0; detW < nid; ++detW) {
                Determinant new_det(det);
                bool successFlag = false;
                //                do {
                size_t rand_ext = rand_int() % sumgen_;
                //                outfile->Printf("\nreached here 1: %d %d %d %d
                //                %d, rand: %d", cume_sumgen_[0],
                //                cume_sumgen_[1], cume_sumgen_[2],
                //                cume_sumgen_[3], cume_sumgen_[4], rand_ext);
                if (rand_ext < cume_sumgen_[2]) {
                    if (rand_ext >= cume_sumgen_[1]) {
                        //                        outfile->Printf("\nreached
                        //                        here 2");
                        successFlag =
                            detDoubleMixSpinRandomExcitation(new_det, aocc, bocc, avir, bvir);
                    } else {
                        if (rand_ext < cume_sumgen_[0]) {
                            successFlag = detSingleRandomExcitation(new_det, aocc, avir, true);
                        } else {
                            successFlag = detSingleRandomExcitation(new_det, bocc, bvir, false);
                        }
                    }

                } else {
                    if (rand_ext < cume_sumgen_[3]) {
                        successFlag = detDoubleSoloSpinRandomExcitation(new_det, aocc, avir, true);
                    } else {
                        successFlag = detDoubleSoloSpinRandomExcitation(new_det, bocc, bvir, false);
                    }
                }
                //                } while(!successFlag);

                //                outfile->Printf("\nreached here 4, excited
                //                det:");
                //                new_det.print();
                if (!successFlag)
                    continue;
                double HIJ = fci_ints_->slater_rules(new_det, det);
                double pspawn = time_step_ * std::fabs(HIJ) * double(sumgen_);
                //                double pspawn = time_step_ * std::fabs(HIJ) *
                //                double(sumgen_)/nirrep_;

                int pspawn_floor = std::floor(pspawn);
                if (rand_real() < pspawn - double(pspawn_floor)) {
                    pspawn_floor++;
                }
                int nspawn = coef * HIJ > 0 ? -pspawn_floor : pspawn_floor;
                if (nspawn != 0) {
                    new_walkers[new_det] += double(nspawn);
                }

                //                outfile->Printf("\nreached here 5");
            }

            break;
        case all:
        case ground_and_random:
        default:
            break;
        }
    }
}

void FCIQMC::spawn(walker_map& walkers, walker_map& new_walkers) {
    for (auto& det_coef : walkers) {
        const Determinant& det = det_coef.first;
        double coef = det_coef.second;
        size_t nid = std::round(std::fabs(coef));
        if (use_initiator_ && nid < std::round(initiator_na_))
            continue;
        //        size_t nsa,nsb,ndaa,ndab,ndbb;
        //        std::tuple<size_t,size_t,size_t,size_t,size_t> pgen =
        //        compute_pgen(det);
        //        std::tie (nsa,nsb,ndaa,ndab,ndbb) = pgen;

        //        size_t sumgen = nsa + nsb + ndaa + ndbb + ndab;
        std::vector<std::tuple<size_t, size_t>> singleExcitations;
        std::vector<std::tuple<size_t, size_t, size_t, size_t>> doubleExcitations;
        size_t sumSingle = 0, sumDouble = 0;
        size_t sumgen = 0;

        ObtCount obtCount;
        std::vector<size_t> excitationDivides;
        std::vector<std::tuple<int, int, int, int>> excitationType;

        //        outfile->Printf("\nspawn_type_:%d", spawn_type_);

        switch (spawn_type_) {
        case random:
            timer_on("FCIQMC:Compute_excitations");
            sumgen =
                compute_irrep_divided_excitations(det, excitationDivides, excitationType, obtCount);
            //            outfile->Printf("\nreached sumgen! sumgen= %d",
            //            sumgen);

            timer_off("FCIQMC:Compute_excitations");

            for (size_t detW = 0; detW < nid; ++detW) {
                Determinant new_det(det);
                // Select a random number within the range of allowed
                // determinants
                //            singleWalkerSpawn(new_det,det,pgen,sumgen);

                size_t rand_ext = rand_int() % sumgen;
                //                if (rand_ext<sumDouble)
                //                    detDoubleExcitation(new_det,
                //                    doubleExcitations[rand_ext]);
                //                else
                //                    detSingleExcitation(new_det,
                //                    singleExcitations[rand_ext-sumDouble]);
                //                outfile->Printf("\nreached here!");

                detExcitation(new_det, rand_ext, excitationDivides, excitationType, obtCount);

                //                outfile->Printf("\nreached here 2!");

                double HIJ = fci_ints_->slater_rules(new_det, det);
                double pspawn = time_step_ * std::fabs(HIJ) * double(sumgen);
                int pspawn_floor = std::floor(pspawn);
                if (rand_real() < pspawn - double(pspawn_floor)) {
                    pspawn_floor++;
                }

                int nspawn = coef * HIJ > 0 ? -pspawn_floor : pspawn_floor;

                // TODO: check
                if (nspawn != 0) {
                    new_walkers[new_det] += double(nspawn);
                }

                //            outfile->Printf("\n  Determinant %d:",count);
                //            det.print();
                //            outfile->Printf(" spawned %d
                //            (%f):",nspawn,pspawn);
                //            new_det.print();
                //            if (count == 15){
                //                outfile->Printf("\n  Determinant %d spawn
                //                detail:",count);
                //                outfile->Printf("\n  Random: %zu",rand_ext);
                //                outfile->Printf("\n  NS %zu, ND
                //                %zu",sumSingle,sumDouble);
                //                size_t ii,aa,jj,bb;
                //                std::tie (ii,aa,jj,bb) =
                //                doubleExcitations[rand_ext];
                //                outfile->Printf("\n  Ext: %zu %zu %zu
                //                %zu",ii,aa,jj,bb);
                //                std::vector<std::tuple<size_t,size_t>>::iterator
                //                t1 ;
                //                for(t1=singleExcitations.begin();
                //                t1!=singleExcitations.end(); t1++){
                //                    size_t ii,aa;
                //                    std::tie (ii,aa) = *t1;
                //                    outfile->Printf("\n %zu %zu",ii,aa);
                //                }
                //                std::vector<std::tuple<size_t,size_t,size_t,size_t>>::iterator
                //                t2 ;
                //                for(t2=doubleExcitations.begin();
                //                t2!=doubleExcitations.end(); t2++){
                //                    size_t ii,aa,jj,bb;
                //                    std::tie (ii,aa,jj,bb) = *t2;
                //                    outfile->Printf("\n %zu %zu %zu
                //                    %zu",ii,aa,jj,bb);
                //                }
                //            }
            }
            break;
        case all:
            timer_on("FCIQMC:Compute_excitations");
            //            std::vector<std::tuple<size_t,size_t>>
            //            singleExcitations;
            //            std::vector<std::tuple<size_t,size_t,size_t,size_t>>
            //            doubleExcitations;
            compute_excitations(det, singleExcitations, doubleExcitations);
            sumSingle = singleExcitations.size();
            sumDouble = doubleExcitations.size();
            sumgen = sumSingle + sumDouble;
            timer_off("FCIQMC:Compute_excitations");
            for (size_t gen = 0; gen < sumgen; ++gen) {
                Determinant new_det(det);
                if (gen < sumDouble)
                    detDoubleExcitation(new_det, doubleExcitations[gen]);
                else
                    detSingleExcitation(new_det, singleExcitations[gen - sumDouble]);

                double HIJ = fci_ints_->slater_rules(new_det, det);
                double pspawn = time_step_ * std::fabs(HIJ) * nid;
                int pspawn_floor = std::floor(pspawn);
                if (rand_real() < pspawn - double(pspawn_floor)) {
                    pspawn_floor++;
                }

                //                outfile->Printf("\nall excitation called.");

                int nspawn = coef * HIJ > 0 ? -pspawn_floor : pspawn_floor;

                // TODO: check
                if (nspawn != 0) {
                    new_walkers[new_det] += double(nspawn);
                }
            }
            break;
        case ground_and_random:
            timer_on("FCIQMC:Compute_excitations");
            sumgen =
                compute_irrep_divided_excitations(det, excitationDivides, excitationType, obtCount);

            timer_off("FCIQMC:Compute_excitations");

            for (size_t detW = 0; detW < nid; ++detW) {
                Determinant new_det(det);
                size_t rand_ext = rand_int() % sumgen;
                detExcitation(new_det, rand_ext, excitationDivides, excitationType, obtCount);
                if (new_det == reference_) {
                    double HIJ = fci_ints_->slater_rules(new_det, det);
                    double pspawn = time_step_ * std::fabs(HIJ) * double(sumgen);
                    int pspawn_floor = std::floor(pspawn);
                    if (rand_real() < pspawn - double(pspawn_floor)) {
                        pspawn_floor++;
                    }
                    int nspawn = coef * HIJ > 0 ? -pspawn_floor : pspawn_floor;

                    // TODO: check
                    if (nspawn != 0) {
                        new_walkers[new_det] += double(nspawn);
                    }
                } else {
                    double HIJ = fci_ints_->slater_rules(new_det, det);
                    double pspawn = time_step_ * std::fabs(HIJ) * double(sumgen - 1);
                    int pspawn_floor = std::floor(pspawn);
                    if (rand_real() < pspawn - double(pspawn_floor)) {
                        pspawn_floor++;
                    }
                    int nspawn = coef * HIJ > 0 ? -pspawn_floor : pspawn_floor;

                    // TODO: check
                    if (nspawn != 0) {
                        new_walkers[new_det] += double(nspawn);
                    }
                    HIJ = fci_ints_->slater_rules(reference_, det);
                    pspawn = time_step_ * std::fabs(HIJ);
                    pspawn_floor = std::floor(pspawn);
                    if (rand_real() < pspawn - double(pspawn_floor)) {
                        pspawn_floor++;
                    }
                    nspawn = coef * HIJ > 0 ? -pspawn_floor : pspawn_floor;

                    // TODO: check
                    if (nspawn != 0) {
                        new_walkers[reference_] += double(nspawn);
                    }
                }
            }
            break;
        default:
            break;
        }
    }
}

void FCIQMC::singleWalkerSpawn(Determinant& new_det, const Determinant& det,
                               std::tuple<size_t, size_t, size_t, size_t, size_t> pgen,
                               size_t sumgen) {

    size_t nsa, nsb, ndaa, ndab, ndbb;
    std::tie(nsa, nsb, ndaa, ndab, ndbb) = pgen;
    size_t rand_class = rand_int() % sumgen;
    if (rand_class < ndab) {
        const std::vector<int> aocc = det.get_alfa_occ(ncmo_);
        const std::vector<int> avir = det.get_alfa_vir(ncmo_);
        const std::vector<int> bocc = det.get_beta_occ(ncmo_);
        const std::vector<int> bvir = det.get_beta_vir(ncmo_);
        size_t i = aocc[rand_int() % aocc.size()];
        size_t j = bocc[rand_int() % bocc.size()];
        size_t a = avir[rand_int() % avir.size()];
        size_t b = bvir[rand_int() % bvir.size()];
        new_det.set_alfa_bit(i, false);
        new_det.set_alfa_bit(a, true);
        new_det.set_beta_bit(j, false);
        new_det.set_beta_bit(b, true);
    } else if (rand_class < ndab + ndaa) {
        const std::vector<int> aocc = det.get_alfa_occ(ncmo_);
        const std::vector<int> avir = det.get_alfa_vir(ncmo_);
        std::pair<size_t, size_t> ij = generate_ind_random_pair(aocc.size());
        std::pair<size_t, size_t> ab = generate_ind_random_pair(avir.size());
        size_t i = aocc[ij.first];
        size_t j = aocc[ij.second];
        size_t a = avir[ab.first];
        size_t b = avir[ab.second];
        new_det.set_alfa_bit(i, false);
        new_det.set_alfa_bit(j, false);
        new_det.set_alfa_bit(a, true);
        new_det.set_alfa_bit(b, true);
    } else if (rand_class < ndab + ndaa + ndbb) {
        const std::vector<int> bocc = det.get_beta_occ(ncmo_);
        const std::vector<int> bvir = det.get_beta_vir(ncmo_);
        std::pair<size_t, size_t> ij = generate_ind_random_pair(bocc.size());
        std::pair<size_t, size_t> ab = generate_ind_random_pair(bvir.size());
        size_t i = bocc[ij.first];
        size_t j = bocc[ij.second];
        size_t a = bvir[ab.first];
        size_t b = bvir[ab.second];
        new_det.set_beta_bit(i, false);
        new_det.set_beta_bit(j, false);
        new_det.set_beta_bit(a, true);
        new_det.set_beta_bit(b, true);
    } else if (rand_class < ndab + ndaa + ndbb + nsa) {
        const std::vector<int> aocc = det.get_alfa_occ(ncmo_);
        const std::vector<int> avir = det.get_alfa_vir(ncmo_);
        size_t i = aocc[rand_int() % aocc.size()];
        size_t a = avir[rand_int() % avir.size()];
        new_det.set_alfa_bit(i, false);
        new_det.set_alfa_bit(a, true);
    } else {
        const std::vector<int> bocc = det.get_beta_occ(ncmo_);
        const std::vector<int> bvir = det.get_beta_vir(ncmo_);
        size_t i = bocc[rand_int() % bocc.size()];
        size_t a = bvir[rand_int() % bvir.size()];
        new_det.set_beta_bit(i, false);
        new_det.set_beta_bit(a, true);
    }
}

// Step #2.  Death/Clone
void FCIQMC::death_clone(walker_map& walkers, double shift) {
    for (auto& det_coef : walkers) {
        const Determinant& det = det_coef.first;
        double coef = det_coef.second;
        double HII = fci_ints_->energy(det);
        double pDeathClone = time_step_ * (HII - Ehf_ - shift);

        int pDC_trunc = std::trunc(pDeathClone);

        size_t nid = std::round(std::fabs(coef));
        double signCoef = coef >= 0.0 ? 1.0 : -1.0;
        for (size_t detW = 0; detW < nid; ++detW) {

            if (rand_real() > std::fabs(pDeathClone - pDC_trunc)) {
                coef -= signCoef * pDC_trunc;
            } else {
                coef -= signCoef * (pDC_trunc + (pDeathClone >= 0.0 ? 1.0 : -1.0));
            }
        }
        //        det.print();
        //        outfile->Printf("\nold coef=%lf, new coef=%lf,
        //        pDC=%lf",det_coef.second, coef, pDeathClone);

        walkers[det] = coef;

        //        if (pDeathClone<0)
        //            detClone(walkers, det, coef, pDeathClone);
        //        else
        //            detDeath(walkers, det, coef, pDeathClone);
    }
}

void FCIQMC::detClone(walker_map& walkers, const Determinant& det, double coef,
                      double pDeathClone) {
    int pDC_trunc = std::trunc(pDeathClone);
    size_t nid = std::round(std::fabs(coef));
    int cloneCount = 0;
    for (size_t detW = 0; detW < nid; ++detW) {
        if (rand_real() < double(pDC_trunc) - pDeathClone) {
            cloneCount += pDC_trunc - 1;
        } else {
            cloneCount += pDC_trunc;
        }
    }
    if (cloneCount < 0) {
        double signFlag = coef > 0.0 ? -1.0 : 1.0;
        walkers[det] += signFlag * cloneCount;
        //        outfile->Printf("\n  Determinant:");
        //        det.print();
        //        outfile->Printf("%f dets cloned %d:",nid,signFlag*cloneCount);
    } else {
        //        outfile->Printf("\n  Determinant:");
        //        det.print();
        //        outfile->Printf("dets did not clone");
    }
}

void FCIQMC::detDeath(walker_map& walkers, const Determinant& det, double coef,
                      double pDeathClone) {
    int pDC_trunc = std::trunc(pDeathClone);
    size_t nid = std::round(std::fabs(coef));
    int deathCount = 0;
    for (size_t detW = 0; detW < nid; ++detW) {
        if (rand_real() < pDeathClone - double(pDC_trunc)) {
            deathCount += pDC_trunc + 1;
        } else {
            deathCount += pDC_trunc;
        }
    }
    if (deathCount < 0) {
        double signFlag = coef > 0.0 ? -1.0 : 1.0;
        walkers[det] += signFlag * deathCount;
        //        outfile->Printf("\n  Determinant:");
        //        det.print();
        //        outfile->Printf("%f dets died %d:",nid,signFlag*deathCount);
    } else {
        //        outfile->Printf("\n  Determinant:");
        //        det.print();
        //        outfile->Printf("dets did not die");
    }
}

// Step #3.  Merge
void FCIQMC::merge(walker_map& walkers, walker_map& new_walkers) {
    std::vector<Determinant> removeDets;
    for (auto& det_coef : walkers) {
        const Determinant& det = det_coef.first;
        if (new_walkers.count(det)) {
            walkers[det] += new_walkers[det];
            new_walkers.erase(det);
        }
        if (int(std::round(walkers[det])) == 0) {
            removeDets.push_back(det);
        }
    }
    for (auto& det_coef : new_walkers) {
        const Determinant& det = det_coef.first;
        walkers[det] = det_coef.second;
        if (int(std::round(walkers[det])) == 0) {
            removeDets.push_back(det);
        }
    }
    for (auto& det : removeDets) {
        walkers.erase(det);
    }
}

// Step #3.  Annihilation
void FCIQMC::annihilate(walker_map& walkers, walker_map& new_walkers) {}

std::tuple<size_t, size_t, size_t, size_t, size_t> FCIQMC::compute_pgen_C1(const Determinant& det) {
    const std::vector<int> aocc = det.get_alfa_occ(ncmo_);
    const std::vector<int> bocc = det.get_beta_occ(ncmo_);
    const std::vector<int> avir = det.get_alfa_vir(ncmo_);
    const std::vector<int> bvir = det.get_beta_vir(ncmo_);

    int noalpha = aocc.size();
    int nobeta = bocc.size();
    int nvalpha = avir.size();
    int nvbeta = bvir.size();

    size_t nsa = noalpha * nvalpha;
    size_t nsb = nobeta * nvbeta;
    size_t ndaa = noalpha * (noalpha - 1) * nvalpha * (nvalpha - 1) / 4;
    size_t ndbb = nobeta * (nobeta - 1) * nvbeta * (nvbeta - 1) / 4;
    size_t ndab = noalpha * nobeta * nvalpha * nvbeta;
    return std::make_tuple(nsa, nsb, ndaa, ndab, ndbb);
}

std::tuple<size_t, size_t, size_t, size_t, size_t> FCIQMC::compute_pgen(const Determinant& det) {
    const std::vector<int> aocc = det.get_alfa_occ(ncmo_);
    const std::vector<int> bocc = det.get_beta_occ(ncmo_);
    const std::vector<int> avir = det.get_alfa_vir(ncmo_);
    const std::vector<int> bvir = det.get_beta_vir(ncmo_);

    int noalpha = aocc.size();
    int nobeta = bocc.size();
    int nvalpha = avir.size();
    int nvbeta = bvir.size();

    size_t nsa = 0;
    for (int i = 0; i < noalpha; ++i) {
        int ii = aocc[i];
        for (int a = 0; a < nvalpha; ++a) {
            int aa = avir[a];
            if ((mo_symmetry_[ii] ^ mo_symmetry_[aa]) == wavefunction_symmetry_) {
                nsa++;
            }
        }
    }
    size_t nsb = 0;
    for (int i = 0; i < nobeta; ++i) {
        int ii = bocc[i];
        for (int a = 0; a < nvbeta; ++a) {
            int aa = bvir[a];
            if ((mo_symmetry_[ii] ^ mo_symmetry_[aa]) == wavefunction_symmetry_) {
                nsb++;
            }
        }
    }

    size_t ndaa = 0;
    for (int i = 0; i < noalpha; ++i) {
        int ii = aocc[i];
        for (int j = i + 1; j < noalpha; ++j) {
            int jj = aocc[j];
            for (int a = 0; a < nvalpha; ++a) {
                int aa = avir[a];
                for (int b = a + 1; b < nvalpha; ++b) {
                    int bb = avir[b];
                    if ((mo_symmetry_[ii] ^ mo_symmetry_[jj] ^ mo_symmetry_[aa] ^
                         mo_symmetry_[bb]) == wavefunction_symmetry_) {
                        ndaa++;
                    }
                }
            }
        }
    }

    size_t ndab = 0;
    for (int i = 0; i < noalpha; ++i) {
        int ii = aocc[i];
        for (int j = 0; j < nobeta; ++j) {
            int jj = bocc[j];
            for (int a = 0; a < nvalpha; ++a) {
                int aa = avir[a];
                for (int b = 0; b < nvbeta; ++b) {
                    int bb = bvir[b];
                    if ((mo_symmetry_[ii] ^ mo_symmetry_[jj] ^ mo_symmetry_[aa] ^
                         mo_symmetry_[bb]) == wavefunction_symmetry_) {
                        ndab++;
                    }
                }
            }
        }
    }

    size_t ndbb = 0;
    for (int i = 0; i < nobeta; ++i) {
        int ii = bocc[i];
        for (int j = i + 1; j < nobeta; ++j) {
            int jj = bocc[j];
            for (int a = 0; a < nvbeta; ++a) {
                int aa = bvir[a];
                for (int b = a + 1; b < nvbeta; ++b) {
                    int bb = bvir[b];
                    if ((mo_symmetry_[ii] ^
                         (mo_symmetry_[jj] ^ (mo_symmetry_[aa] ^ mo_symmetry_[bb]))) ==
                        wavefunction_symmetry_) {
                        ndbb++;
                    }
                }
            }
        }
    }
    return std::make_tuple(nsa, nsb, ndaa, ndab, ndbb);
}

void FCIQMC::compute_excitations(
    const Determinant& det, std::vector<std::tuple<size_t, size_t>>& singleExcitations,
    std::vector<std::tuple<size_t, size_t, size_t, size_t>>& doubleExcitations) {
    compute_single_excitations(det, singleExcitations);
    compute_double_excitations(det, doubleExcitations);
}

void FCIQMC::compute_single_excitations(
    const Determinant& det, std::vector<std::tuple<size_t, size_t>>& singleExcitations) {
    const std::vector<int> aocc = det.get_alfa_occ(ncmo_);
    const std::vector<int> bocc = det.get_beta_occ(ncmo_);
    const std::vector<int> avir = det.get_alfa_vir(ncmo_);
    const std::vector<int> bvir = det.get_beta_vir(ncmo_);

    //    det.print();

    //    for (size_t ii : aocc){
    //        outfile->Printf("\n  aocc = %d",ii);
    //    }

    //    for (size_t aa : avir){
    //        outfile->Printf("\n  avir = %d",aa);
    //    }
    int noalpha = aocc.size();
    int nobeta = bocc.size();
    int nvalpha = avir.size();
    int nvbeta = bvir.size();

    for (int i = 0; i < noalpha; ++i) {
        int ii = aocc[i];
        for (int a = 0; a < nvalpha; ++a) {
            int aa = avir[a];
            if ((mo_symmetry_[ii] ^ mo_symmetry_[aa]) == 0) {
                singleExcitations.push_back(std::make_tuple(ii, aa));
            }
        }
    }
    for (int i = 0; i < nobeta; ++i) {
        int ii = bocc[i];
        for (int a = 0; a < nvbeta; ++a) {
            int aa = bvir[a];
            if ((mo_symmetry_[ii] ^ mo_symmetry_[aa]) == 0) {
                singleExcitations.push_back(std::make_tuple(ii + ncmo_, aa + ncmo_));
            }
        }
    }
}

void FCIQMC::compute_double_excitations(
    const Determinant& det,
    std::vector<std::tuple<size_t, size_t, size_t, size_t>>& doubleExcitations) {
    const std::vector<int> aocc = det.get_alfa_occ(ncmo_);
    const std::vector<int> bocc = det.get_beta_occ(ncmo_);
    const std::vector<int> avir = det.get_alfa_vir(ncmo_);
    const std::vector<int> bvir = det.get_beta_vir(ncmo_);

    int noalpha = aocc.size();
    int nobeta = bocc.size();
    int nvalpha = avir.size();
    int nvbeta = bvir.size();

    for (int i = 0; i < noalpha; ++i) {
        int ii = aocc[i];
        for (int j = i + 1; j < noalpha; ++j) {
            int jj = aocc[j];
            for (int a = 0; a < nvalpha; ++a) {
                int aa = avir[a];
                for (int b = a + 1; b < nvalpha; ++b) {
                    int bb = avir[b];
                    if ((mo_symmetry_[ii] ^ mo_symmetry_[jj] ^ mo_symmetry_[aa] ^
                         mo_symmetry_[bb]) == 0) {
                        doubleExcitations.push_back(std::make_tuple(ii, aa, jj, bb));
                    }
                }
            }
        }
    }

    for (int i = 0; i < noalpha; ++i) {
        int ii = aocc[i];
        for (int j = 0; j < nobeta; ++j) {
            int jj = bocc[j];
            for (int a = 0; a < nvalpha; ++a) {
                int aa = avir[a];
                for (int b = 0; b < nvbeta; ++b) {
                    int bb = bvir[b];
                    if ((mo_symmetry_[ii] ^ mo_symmetry_[jj] ^ mo_symmetry_[aa] ^
                         mo_symmetry_[bb]) == 0) {
                        doubleExcitations.push_back(
                            std::make_tuple(ii, aa, jj + ncmo_, bb + ncmo_));
                    }
                }
            }
        }
    }

    for (int i = 0; i < nobeta; ++i) {
        int ii = bocc[i];
        for (int j = i + 1; j < nobeta; ++j) {
            int jj = bocc[j];
            for (int a = 0; a < nvbeta; ++a) {
                int aa = bvir[a];
                for (int b = a + 1; b < nvbeta; ++b) {
                    int bb = bvir[b];
                    if ((mo_symmetry_[ii] ^ mo_symmetry_[jj] ^ mo_symmetry_[aa] ^
                         mo_symmetry_[bb]) == 0) {
                        doubleExcitations.push_back(
                            std::make_tuple(ii + ncmo_, aa + ncmo_, jj + ncmo_, bb + ncmo_));
                    }
                }
            }
        }
    }
}

/**
 * @brief FCIQMC::compute_irrep_divided_excitations
 * compute the number of excitations in each irrep combination catagory
 * @param det
 * determinants to be excited
 * @param excitationDivides
 * A division of excitations by the cumulative number of excitations
 * @param excitationType
 * The type of excitation by tuple of three irrep index
 * @return
 * total number of excitations
 */
size_t FCIQMC::compute_irrep_divided_excitations(
    const Determinant& det, std::vector<size_t>& excitationDivides,
    std::vector<std::tuple<int, int, int, int>>& excitationType, ObtCount& obtCount) {
    const std::vector<int> aocc = det.get_alfa_occ(ncmo_);
    const std::vector<int> bocc = det.get_beta_occ(ncmo_);
    const std::vector<int> avir = det.get_alfa_vir(ncmo_);
    const std::vector<int> bvir = det.get_beta_vir(ncmo_);
    size_t totalExcitation = 0;
    obtCount.naocc.clear();
    obtCount.nbocc.clear();
    obtCount.navir.clear();
    obtCount.nbvir.clear();
    for (int i = 0; i < nirrep_; i++) {
        obtCount.naocc.push_back(0);
        obtCount.nbocc.push_back(0);
        obtCount.navir.push_back(0);
        obtCount.nbvir.push_back(0);
    }
    excitationDivides.clear();
    excitationType.clear();
    for (int i = 0; i < aocc.size(); i++) {
        obtCount.naocc[mo_symmetry_[aocc[i]]]++;
    }
    for (int i = 0; i < bocc.size(); i++) {
        obtCount.nbocc[mo_symmetry_[bocc[i]]]++;
    }
    for (int i = 0; i < avir.size(); i++) {
        obtCount.navir[mo_symmetry_[avir[i]]]++;
    }
    for (int i = 0; i < bvir.size(); i++) {
        obtCount.nbvir[mo_symmetry_[bvir[i]]]++;
    }
    for (int i = 0; i < nirrep_; i++) {
        for (int j = 0; j < nirrep_; j++) {
            for (int k = 0; k < nirrep_; k++) {
                int l = i ^ j ^ k;
                totalExcitation +=
                    obtCount.naocc[i] * obtCount.nbocc[j] * obtCount.navir[k] * obtCount.nbvir[l];
                excitationDivides.push_back(totalExcitation);
                excitationType.push_back(std::make_tuple(0, i, j, k));
            }
        }
    }
    for (int i = 0; i < nirrep_; i++) {
        for (int j = 0; j < nirrep_; j++) {
            totalExcitation += obtCount.naocc[i] * (obtCount.naocc[i] - 1) * obtCount.navir[j] *
                               (obtCount.navir[j] - 1) / 4;
            excitationDivides.push_back(totalExcitation);
            excitationType.push_back(std::make_tuple(1, i, i, j));
        }
    }
    for (int i = 0; i < nirrep_; i++) {
        for (int j = i + 1; j < nirrep_; j++) {
            for (int k = 0; k < nirrep_; k++) {
                int l = i ^ j ^ k;
                if (l > k) {
                    totalExcitation += obtCount.naocc[i] * obtCount.naocc[j] * obtCount.navir[k] *
                                       obtCount.navir[l];
                    excitationDivides.push_back(totalExcitation);
                    excitationType.push_back(std::make_tuple(2, i, j, k));
                }
            }
        }
    }
    for (int i = 0; i < nirrep_; i++) {
        for (int j = 0; j < nirrep_; j++) {
            totalExcitation += obtCount.nbocc[i] * (obtCount.nbocc[i] - 1) * obtCount.nbvir[j] *
                               (obtCount.nbvir[j] - 1) / 4;
            excitationDivides.push_back(totalExcitation);
            excitationType.push_back(std::make_tuple(3, i, i, j));
        }
    }
    for (int i = 0; i < nirrep_; i++) {
        for (int j = i + 1; j < nirrep_; j++) {
            for (int k = 0; k < nirrep_; k++) {
                int l = i ^ j ^ k;
                if (l > k) {
                    totalExcitation += obtCount.nbocc[i] * obtCount.nbocc[j] * obtCount.nbvir[k] *
                                       obtCount.nbvir[l];
                    excitationDivides.push_back(totalExcitation);
                    excitationType.push_back(std::make_tuple(4, i, j, k));
                }
            }
        }
    }
    for (int i = 0; i < nirrep_; i++) {
        totalExcitation += obtCount.naocc[i] * obtCount.navir[i];
        excitationDivides.push_back(totalExcitation);
        excitationType.push_back(std::make_tuple(5, i, i, -1));
    }
    for (int i = 0; i < nirrep_; i++) {
        totalExcitation += obtCount.nbocc[i] * obtCount.nbvir[i];
        excitationDivides.push_back(totalExcitation);
        excitationType.push_back(std::make_tuple(6, i, i, -1));
    }
    return totalExcitation;
}

bool FCIQMC::detSingleRandomExcitation(Determinant& new_det, const std::vector<int>& occ,
                                       const std::vector<int>& vir, bool isAlpha) {
    int o = 0, v = 0;
    int randO = 0, randV = 0;
    //    do {
    //        if (count > 2*nirrep_) return false;
    //        count++;
    randO = rand_int() % occ.size();
    randV = rand_int() % vir.size();
    o = occ[randO];
    v = vir[randV];
    //    }while (mo_symmetry_[o] != mo_symmetry_[v]);
    if (mo_symmetry_[o] == mo_symmetry_[v]) {
        if (isAlpha) {
            new_det.set_alfa_bit(o, false);
            new_det.set_alfa_bit(v, true);
        } else {
            new_det.set_beta_bit(o, false);
            new_det.set_beta_bit(v, true);
        }
        return true;
    }
    return false;
}

bool FCIQMC::detDoubleSoloSpinRandomExcitation(Determinant& new_det, const std::vector<int>& occ,
                                               const std::vector<int>& vir, bool isAlpha) {
    int o1 = 0, o2 = 0, v1 = 0, v2 = 0;
    int randO1 = 0, randO2 = 0, randV1 = 0, randV2 = 0;
    //    int count = 0;
    //    do {
    //        if (count > 2*nirrep_) return false;
    //        count++;
    randO1 = rand_int() % occ.size();
    randO2 = rand_int() % (occ.size() - 1);
    if (randO2 >= randO1)
        randO2++;
    randV1 = rand_int() % vir.size();
    randV2 = rand_int() % (vir.size() - 1);
    if (randV2 >= randV1)
        randV2++;
    o1 = occ[randO1];
    o2 = occ[randO2];
    v1 = vir[randV1];
    v2 = vir[randV2];
    //    }while (mo_symmetry_[o1]^mo_symmetry_[o2]^mo_symmetry_[v1] !=
    //    mo_symmetry_[v2]);
    if (mo_symmetry_[o1] ^ mo_symmetry_[o2] ^ mo_symmetry_[v1] == mo_symmetry_[v2]) {
        if (isAlpha) {
            new_det.set_alfa_bit(o1, false);
            new_det.set_alfa_bit(v1, true);
            new_det.set_alfa_bit(o2, false);
            new_det.set_alfa_bit(v2, true);
        } else {
            new_det.set_beta_bit(o1, false);
            new_det.set_beta_bit(v1, true);
            new_det.set_beta_bit(o2, false);
            new_det.set_beta_bit(v2, true);
        }
        return true;
    }
    return false;
}

bool FCIQMC::detDoubleMixSpinRandomExcitation(Determinant& new_det, const std::vector<int>& aocc,
                                              const std::vector<int>& bocc,
                                              const std::vector<int>& avir,
                                              const std::vector<int>& bvir) {
    int o1 = 0, o2 = 0, v1 = 0, v2 = 0;
    int randO1 = 0, randO2 = 0, randV1 = 0, randV2 = 0;
    //    int count = 0;
    //    do {
    //        if (count > 2*nirrep_) return false;
    //        count++;
    randO1 = rand_int() % aocc.size();
    randO2 = rand_int() % bocc.size();
    randV1 = rand_int() % avir.size();
    randV2 = rand_int() % bvir.size();
    o1 = aocc[randO1];
    o2 = bocc[randO2];
    v1 = avir[randV1];
    v2 = bvir[randV2];
    //    }while (mo_symmetry_[o1]^mo_symmetry_[o2]^mo_symmetry_[v1] !=
    //    mo_symmetry_[v2]);

    //    outfile->Printf("\nreached here 3 o1%dv1%do2%dv2%d",o1,v1,o2,v2);
    //    new_det.print();
    if (mo_symmetry_[o1] ^ mo_symmetry_[o2] ^ mo_symmetry_[v1] == mo_symmetry_[v2]) {
        new_det.set_alfa_bit(o1, false);
        new_det.set_alfa_bit(v1, true);
        new_det.set_beta_bit(o2, false);
        new_det.set_beta_bit(v2, true);
        return true;
    }
    return false;
}

void FCIQMC::detExcitation(Determinant& new_det, size_t rand_ext,
                           std::vector<size_t>& excitationDivides,
                           std::vector<std::tuple<int, int, int, int>>& excitationType,
                           ObtCount& obtCount) {
    size_t begin = 0, end = excitationDivides.size() - 1;
    size_t middle = (begin + end) / 2;
    rand_ext++;
    while (begin + 1 < end) {
        if (rand_ext < excitationDivides[middle]) {
            end = middle;
        } else {
            begin = middle;
        }
        middle = (begin + end) / 2;
        //        outfile->Printf("\nreached here 3! middle %d, begin %d, end
        //        %d", middle, begin, end);
    }

    //    outfile->Printf("\nreached here 3! end %d, rand_ext %d", end,
    //    rand_ext);
    while (end > 0 && excitationDivides[end - 1] >= rand_ext) {
        end--;
    }

    if (rand_ext == 0) {
        while (excitationDivides[end] == 0) {
            end++;
        }
    }

    //    outfile->Printf("\nreached here 3! end %d, cume_excit_irrep: %d %d %d
    //    %d %d %d %d", end,
    //                    cume_excit_irrep_[0], cume_excit_irrep_[1],
    //                    cume_excit_irrep_[2], cume_excit_irrep_[3],
    //                    cume_excit_irrep_[4], cume_excit_irrep_[5],
    //                    cume_excit_irrep_[6]);
    //    outfile->Printf("\nreached here 3! excitationDivides: 1090: %d %d %d
    //    %d", excitationDivides[1090], excitationDivides[1091],
    //    excitationDivides[1092], excitationDivides[1093]);

    int t, i, j, k, l, randI, randJ, randK, randL, obtI, obtJ, obtK, obtL;
    std::tie(t, i, j, k) = excitationType[end];

    switch (t) {
    case 0:
        l = i ^ j ^ k;
        randI = rand_int() % obtCount.naocc[i];
        randJ = rand_int() % obtCount.nbocc[j];
        randK = rand_int() % obtCount.navir[k];
        randL = rand_int() % obtCount.nbvir[l];

        for (obtI = cume_ncmopi_[i]; obtI < cume_ncmopi_[i + 1]; obtI++) {
            if (new_det.get_alfa_bit(obtI) == true) {
                if (randI == 0) {
                    break;
                } else {
                    randI--;
                }
            }
        }

        for (obtJ = cume_ncmopi_[j]; obtJ < cume_ncmopi_[j + 1]; obtJ++) {
            if (new_det.get_beta_bit(obtJ) == true) {
                if (randJ == 0) {
                    break;
                } else {
                    randJ--;
                }
            }
        }

        for (obtK = cume_ncmopi_[k]; obtK < cume_ncmopi_[k + 1]; obtK++) {
            if (new_det.get_alfa_bit(obtK) == false) {
                if (randK == 0) {
                    break;
                } else {
                    randK--;
                }
            }
        }

        for (obtL = cume_ncmopi_[l]; obtL < cume_ncmopi_[l + 1]; obtL++) {
            if (new_det.get_beta_bit(obtL) == false) {
                if (randL == 0) {
                    break;
                } else {
                    randL--;
                }
            }
        }
        if (obtI == cume_ncmopi_[i + 1] || obtJ == cume_ncmopi_[j + 1] ||
            obtK == cume_ncmopi_[k + 1] || obtL == cume_ncmopi_[l + 1]) {
            outfile->Printf("\nError in det excitation type 0: orbital out of bond");
        }

        //        if (new_det.get_alfa_bit(cume_ncmopi_[i]+randI) == false){
        //            outfile->Printf("\nline 1001: Error in det excitation type
        //            0:\n sym: i %d j %d k %d l %d\nrand: i %d j %d k %d l %d",
        //            i,j,k,l,randI,randJ,randK,randL);
        //            new_det.print();
        //        }
        //        if (new_det.get_beta_bit(cume_ncmopi_[j]+randJ) == false){
        //            outfile->Printf("\nline 1005: Error in det excitation type
        //            0:\n sym: i %d j %d k %d l %d\nrand: i %d j %d k %d l %d",
        //            i,j,k,l,randI,randJ,randK,randL);
        //            new_det.print();
        //        }
        //        if (new_det.get_alfa_bit(cume_ncmopi_[k]+randK) == true){
        //            outfile->Printf("\nline 1009: Error in det excitation type
        //            0:\n sym: i %d j %d k %d l %d\nrand: i %d j %d k %d l %d",
        //            i,j,k,l,randI,randJ,randK,randL);
        //            new_det.print();
        //        }
        //        if (new_det.get_beta_bit(cume_ncmopi_[l]+randL) == true){
        //            outfile->Printf("\nline 1013: Error in det excitation type
        //            0:\n sym: i %d j %d k %d l %d\nrand: i %d j %d k %d l %d",
        //            i,j,k,l,randI,randJ,randK,randL);
        //            new_det.print();
        //        }
        new_det.set_alfa_bit(obtI, false);
        new_det.set_beta_bit(obtJ, false);
        new_det.set_alfa_bit(obtK, true);
        new_det.set_beta_bit(obtL, true);
        break;
    case 1:
        randI = rand_int() % obtCount.naocc[i];
        randJ = rand_int() % (obtCount.naocc[i] - 1);
        if (randJ >= randI)
            randJ++;
        else
            std::swap(randI, randJ);
        randK = rand_int() % obtCount.navir[k];
        randL = rand_int() % (obtCount.navir[k] - 1);
        if (randL >= randK)
            randL++;
        else
            std::swap(randK, randL);
        obtI = -1;
        for (obtJ = cume_ncmopi_[i]; obtJ < cume_ncmopi_[i + 1]; obtJ++) {
            if (new_det.get_alfa_bit(obtJ) == true) {
                if (randI == 0) {
                    obtI = obtJ;
                    randI = -1;
                } else if (randI > 0) {
                    randI--;
                }
                if (randJ == 0) {
                    break;
                } else {
                    randJ--;
                }
            }
        }
        obtK = -1;
        for (obtL = cume_ncmopi_[k]; obtL < cume_ncmopi_[k + 1]; obtL++) {
            if (new_det.get_alfa_bit(obtL) == false) {
                if (randK == 0) {
                    obtK = obtL;
                    randK = -1;
                } else if (randK > 0) {
                    randK--;
                }
                if (randL == 0) {
                    break;
                } else {
                    randL--;
                }
            }
        }
        if (obtI == -1 || obtJ == cume_ncmopi_[i + 1] || obtK == -1 ||
            obtL == cume_ncmopi_[k + 1]) {
            outfile->Printf("\nError in det excitation type 1: orbital out of bond");
        }

        //        if (new_det.get_alfa_bit(cume_ncmopi_[i]+randI) == false){
        //            outfile->Printf("\nline 1029: Error in det excitation type
        //            1:\n sym: i %d j %d k %d l %d\nrand: i %d j %d k %d l %d",
        //            i,j,k,l,randI,randJ,randK,randL);
        //            new_det.print();
        //        }
        //        if (new_det.get_alfa_bit(cume_ncmopi_[i]+randJ) == false){
        //            outfile->Printf("\nline 1033: Error in det excitation type
        //            1:\n sym: i %d j %d k %d l %d\nrand: i %d j %d k %d l %d",
        //            i,j,k,l,randI,randJ,randK,randL);
        //            new_det.print();
        //        }
        //        if (new_det.get_alfa_bit(cume_ncmopi_[k]+randK) == true){
        //            outfile->Printf("\nline 1037: Error in det excitation type
        //            1:\n sym: i %d j %d k %d l %d\nrand: i %d j %d k %d l %d",
        //            i,j,k,l,randI,randJ,randK,randL);
        //            new_det.print();
        //        }
        //        if (new_det.get_alfa_bit(cume_ncmopi_[k]+randL) == true){
        //            outfile->Printf("\nline 1041: Error in det excitation type
        //            1:\n sym: i %d j %d k %d l %d\nrand: i %d j %d k %d l %d",
        //            i,j,k,l,randI,randJ,randK,randL);
        //            new_det.print();
        //        }
        new_det.set_alfa_bit(obtI, false);
        new_det.set_alfa_bit(obtJ, false);
        new_det.set_alfa_bit(obtK, true);
        new_det.set_alfa_bit(obtL, true);
        break;
    case 2:
        l = i ^ j ^ k;
        randI = rand_int() % obtCount.naocc[i];
        randJ = rand_int() % obtCount.naocc[j];
        randK = rand_int() % obtCount.navir[k];
        randL = rand_int() % obtCount.navir[l];

        for (obtI = cume_ncmopi_[i]; obtI < cume_ncmopi_[i + 1]; obtI++) {
            if (new_det.get_alfa_bit(obtI) == true) {
                if (randI == 0) {
                    break;
                } else {
                    randI--;
                }
            }
        }

        for (obtJ = cume_ncmopi_[j]; obtJ < cume_ncmopi_[j + 1]; obtJ++) {
            if (new_det.get_alfa_bit(obtJ) == true) {
                if (randJ == 0) {
                    break;
                } else {
                    randJ--;
                }
            }
        }

        for (obtK = cume_ncmopi_[k]; obtK < cume_ncmopi_[k + 1]; obtK++) {
            if (new_det.get_alfa_bit(obtK) == false) {
                if (randK == 0) {
                    break;
                } else {
                    randK--;
                }
            }
        }

        for (obtL = cume_ncmopi_[l]; obtL < cume_ncmopi_[l + 1]; obtL++) {
            if (new_det.get_alfa_bit(obtL) == false) {
                if (randL == 0) {
                    break;
                } else {
                    randL--;
                }
            }
        }
        if (obtI == cume_ncmopi_[i + 1] || obtJ == cume_ncmopi_[j + 1] ||
            obtK == cume_ncmopi_[k + 1] || obtL == cume_ncmopi_[l + 1]) {
            outfile->Printf("\nError in det excitation type 2: orbital out of bond");
        }
        //        if (new_det.get_alfa_bit(cume_ncmopi_[i]+randI) == false){
        //            outfile->Printf("\nline 1056: Error in det excitation type
        //            2:\n sym: i %d j %d k %d l %d\nrand: i %d j %d k %d l %d",
        //            i,j,k,l,randI,randJ,randK,randL);
        //            new_det.print();
        //        }
        //        if (new_det.get_alfa_bit(cume_ncmopi_[j]+randJ) == false){
        //            outfile->Printf("\nline 1060: Error in det excitation type
        //            2:\n sym: i %d j %d k %d l %d\nrand: i %d j %d k %d l %d",
        //            i,j,k,l,randI,randJ,randK,randL);
        //            new_det.print();
        //        }
        //        if (new_det.get_alfa_bit(cume_ncmopi_[k]+randK) == true){
        //            outfile->Printf("\nline 1064: Error in det excitation type
        //            2:\n sym: i %d j %d k %d l %d\nrand: i %d j %d k %d l %d",
        //            i,j,k,l,randI,randJ,randK,randL);
        //            new_det.print();
        //        }
        //        if (new_det.get_alfa_bit(cume_ncmopi_[l]+randL) == true){
        //            outfile->Printf("\nline 1068: Error in det excitation type
        //            2:\n sym: i %d j %d k %d l %d\nrand: i %d j %d k %d l %d",
        //            i,j,k,l,randI,randJ,randK,randL);
        //            new_det.print();
        //        }
        new_det.set_alfa_bit(obtI, false);
        new_det.set_alfa_bit(obtJ, false);
        new_det.set_alfa_bit(obtK, true);
        new_det.set_alfa_bit(obtL, true);
        break;
    case 3:
        randI = rand_int() % obtCount.nbocc[i];
        randJ = rand_int() % (obtCount.nbocc[i] - 1);
        if (randJ >= randI)
            randJ++;
        else
            std::swap(randI, randJ);
        randK = rand_int() % obtCount.nbvir[k];
        randL = rand_int() % (obtCount.nbvir[k] - 1);
        if (randL >= randK)
            randL++;
        else
            std::swap(randK, randL);
        obtI = -1;
        for (obtJ = cume_ncmopi_[i]; obtJ < cume_ncmopi_[i + 1]; obtJ++) {
            if (new_det.get_beta_bit(obtJ) == true) {
                if (randI == 0) {
                    obtI = obtJ;
                    randI = -1;
                } else if (randI > 0) {
                    randI--;
                }
                if (randJ == 0) {
                    break;
                } else {
                    randJ--;
                }
            }
        }
        obtK = -1;
        for (obtL = cume_ncmopi_[k]; obtL < cume_ncmopi_[k + 1]; obtL++) {
            if (new_det.get_beta_bit(obtL) == false) {
                if (randK == 0) {
                    obtK = obtL;
                    randK = -1;
                } else if (randK > 0) {
                    randK--;
                }
                if (randL == 0) {
                    break;
                } else {
                    randL--;
                }
            }
        }
        if (obtI == -1 || obtJ == cume_ncmopi_[i + 1] || obtK == -1 ||
            obtL == cume_ncmopi_[k + 1]) {
            outfile->Printf("\nError in det excitation type 3: orbital out of bond");
        }
        //        if (new_det.get_beta_bit(cume_ncmopi_[i]+randI) == false){
        //            outfile->Printf("\nline 1084: Error in det excitation type
        //            3:\n sym: i %d j %d k %d l %d\nrand: i %d j %d k %d l %d",
        //            i,j,k,l,randI,randJ,randK,randL);
        //            new_det.print();
        //        }
        //        if (new_det.get_beta_bit(cume_ncmopi_[i]+randJ) == false){
        //            outfile->Printf("\nline 1088: Error in det excitation type
        //            3:\n sym: i %d j %d k %d l %d\nrand: i %d j %d k %d l %d",
        //            i,j,k,l,randI,randJ,randK,randL);
        //            new_det.print();
        //        }
        //        if (new_det.get_beta_bit(cume_ncmopi_[k]+randK) == true){
        //            outfile->Printf("\nline 1092: Error in det excitation type
        //            3:\n sym: i %d j %d k %d l %d\nrand: i %d j %d k %d l %d",
        //            i,j,k,l,randI,randJ,randK,randL);
        //            new_det.print();
        //        }
        //        if (new_det.get_beta_bit(cume_ncmopi_[k]+randL) == true){
        //            outfile->Printf("\nline 1096: Error in det excitation type
        //            3:\n sym: i %d j %d k %d l %d\nrand: i %d j %d k %d l %d",
        //            i,j,k,l,randI,randJ,randK,randL);
        //            new_det.print();
        //        }
        new_det.set_beta_bit(obtI, false);
        new_det.set_beta_bit(obtJ, false);
        new_det.set_beta_bit(obtK, true);
        new_det.set_beta_bit(obtL, true);
        break;
    case 4:
        l = i ^ j ^ k;
        randI = rand_int() % obtCount.nbocc[i];
        randJ = rand_int() % obtCount.nbocc[j];
        randK = rand_int() % obtCount.nbvir[k];
        randL = rand_int() % obtCount.nbvir[l];

        for (obtI = cume_ncmopi_[i]; obtI < cume_ncmopi_[i + 1]; obtI++) {
            if (new_det.get_beta_bit(obtI) == true) {
                if (randI == 0) {
                    break;
                } else {
                    randI--;
                }
            }
        }

        for (obtJ = cume_ncmopi_[j]; obtJ < cume_ncmopi_[j + 1]; obtJ++) {
            if (new_det.get_beta_bit(obtJ) == true) {
                if (randJ == 0) {
                    break;
                } else {
                    randJ--;
                }
            }
        }

        for (obtK = cume_ncmopi_[k]; obtK < cume_ncmopi_[k + 1]; obtK++) {
            if (new_det.get_beta_bit(obtK) == false) {
                if (randK == 0) {
                    break;
                } else {
                    randK--;
                }
            }
        }

        for (obtL = cume_ncmopi_[l]; obtL < cume_ncmopi_[l + 1]; obtL++) {
            if (new_det.get_beta_bit(obtL) == false) {
                if (randL == 0) {
                    break;
                } else {
                    randL--;
                }
            }
        }
        if (obtI == cume_ncmopi_[i + 1] || obtJ == cume_ncmopi_[j + 1] ||
            obtK == cume_ncmopi_[k + 1] || obtL == cume_ncmopi_[l + 1]) {
            outfile->Printf("\nError in det excitation type 4: orbital out of bond");
        }
        //        if (new_det.get_beta_bit(cume_ncmopi_[i]+randI) == false){
        //            outfile->Printf("\nline 1111: Error in det excitation type
        //            4:\n sym: i %d j %d k %d l %d\nrand: i %d j %d k %d l %d",
        //            i,j,k,l,randI,randJ,randK,randL);
        //            new_det.print();
        //        }
        //        if (new_det.get_beta_bit(cume_ncmopi_[j]+randJ) == false){
        //            outfile->Printf("\nline 1115: Error in det excitation type
        //            4:\n sym: i %d j %d k %d l %d\nrand: i %d j %d k %d l %d",
        //            i,j,k,l,randI,randJ,randK,randL);
        //            new_det.print();
        //        }
        //        if (new_det.get_beta_bit(cume_ncmopi_[k]+randK) == true){
        //            outfile->Printf("\nline 1119: Error in det excitation type
        //            4:\n sym: i %d j %d k %d l %d\nrand: i %d j %d k %d l %d",
        //            i,j,k,l,randI,randJ,randK,randL);
        //            new_det.print();
        //        }
        //        if (new_det.get_beta_bit(cume_ncmopi_[l]+randL) == true){
        //            outfile->Printf("\nline 1123: Error in det excitation type
        //            4:\n sym: i %d j %d k %d l %d\nrand: i %d j %d k %d l %d",
        //            i,j,k,l,randI,randJ,randK,randL);
        //            new_det.print();
        //        }
        new_det.set_beta_bit(obtI, false);
        new_det.set_beta_bit(obtJ, false);
        new_det.set_beta_bit(obtK, true);
        new_det.set_beta_bit(obtL, true);
        break;
    case 5:
        randI = rand_int() % obtCount.naocc[i];
        randJ = rand_int() % obtCount.navir[i];
        obtI = -1;
        obtJ = -1;
        for (obtK = cume_ncmopi_[i]; obtK < cume_ncmopi_[i + 1]; obtK++) {
            if (new_det.get_alfa_bit(obtK) == true) {
                if (randI == 0) {
                    obtI = obtK;
                    randI = -1;
                } else if (randI > 0) {
                    randI--;
                }
            } else {
                if (randJ == 0) {
                    obtJ = obtK;
                    randJ = -1;
                } else if (randJ > 0) {
                    randJ--;
                }
            }
        }
        if (obtI == -1 || obtJ == -1) {
            outfile->Printf("\nError in det excitation type 5: orbital out of bond");
        }
        //        if (new_det.get_alfa_bit(cume_ncmopi_[i]+randI) == false){
        //            outfile->Printf("\nline 1135: Error in det excitation type
        //            5:\n sym: i %d j %d k %d l %d\nrand: i %d j %d k %d l %d",
        //            i,j,k,l,randI,randJ,randK,randL);
        //            new_det.print();
        //        }
        //        if (new_det.get_alfa_bit(cume_ncmopi_[j]+randJ) == true){
        //            outfile->Printf("\nline 1139: Error in det excitation type
        //            5:\n sym: i %d j %d k %d l %d\nrand: i %d j %d k %d l %d",
        //            i,j,k,l,randI,randJ,randK,randL);
        //            new_det.print();
        //        }
        new_det.set_alfa_bit(obtI, false);
        new_det.set_alfa_bit(obtJ, true);
        if (k != -1) {
            outfile->Printf("Error in spawning excitation:k is not -1 for single ext");
        }
        break;
    case 6:
        randI = rand_int() % obtCount.nbocc[i];
        randJ = rand_int() % obtCount.nbvir[j];
        obtI = -1;
        obtJ = -1;
        for (obtK = cume_ncmopi_[i]; obtK < cume_ncmopi_[i + 1]; obtK++) {
            if (new_det.get_beta_bit(obtK) == true) {
                if (randI == 0) {
                    obtI = obtK;
                    randI = -1;
                } else if (randI > 0) {
                    randI--;
                }
            } else {
                if (randJ == 0) {
                    obtJ = obtK;
                    randJ = -1;
                } else if (randJ > 0) {
                    randJ--;
                }
            }
        }
        if (obtI == -1 || obtJ == -1) {
            outfile->Printf("\nError in det excitation type 5: orbital out of bond");
        }
        //        if (new_det.get_beta_bit(cume_ncmopi_[i]+randI) == false){
        //            outfile->Printf("\nline 1135: Error in det excitation type
        //            5:\n sym: i %d j %d k %d l %d\nrand: i %d j %d k %d l %d",
        //            i,j,k,l,randI,randJ,randK,randL);
        //            new_det.print();
        //        }
        //        if (new_det.get_beta_bit(cume_ncmopi_[j]+randJ) == true){
        //            outfile->Printf("\nline 1139: Error in det excitation type
        //            5:\n sym: i %d j %d k %d l %d\nrand: i %d j %d k %d l %d",
        //            i,j,k,l,randI,randJ,randK,randL);
        //            new_det.print();
        //        }
        new_det.set_beta_bit(obtI, false);
        new_det.set_beta_bit(obtJ, true);
        if (k != -1) {
            outfile->Printf("Error in spawning excitation:k is not -1 for single ext");
        }
        break;
    default:
        outfile->Printf("Error in spawning excitation:end of choice");
        break;
    }

    //    outfile->Printf("\nreached here 5! i %d j %d k %d", i,j,k);

    //    if (end<cume_excit_irrep_[0]) {
    //        l=i^j^k;
    //        randI = rand_int()%obtCount.naocc[i];
    //        randJ = rand_int()%obtCount.nbocc[j];
    //        randK = rand_int()%obtCount.navir[k];
    //        randL = rand_int()%obtCount.nbvir[l];
    //        if (new_det.get_alfa_bit(cume_ncmopi_[i]+randI) == false){
    //            outfile->Printf("\nline 996: Error in det excitation:\n sym: i
    //            %d j %d k %d l %d\nrand: i %d j %d k %d l %d",
    //            i,j,k,l,randI,randJ,randK,randL);
    //            new_det.print();
    //        }
    //        new_det.set_alfa_bit(cume_ncmopi_[i]+randI,false);
    //        new_det.set_beta_bit(cume_ncmopi_[j]+randJ,false);
    //        new_det.set_alfa_bit(cume_ncmopi_[k]+randK,true);
    //        new_det.set_beta_bit(cume_ncmopi_[l]+randL,true);
    //        return;
    //    }else if (end<cume_excit_irrep_[1]) {
    //        randI = rand_int()%obtCount.naocc[i];
    //        randJ = rand_int()%(obtCount.naocc[i]-1);
    //        if (randJ >= randI) randJ++;
    //        randK = rand_int()%obtCount.navir[k];
    //        randL = rand_int()%(obtCount.navir[k]-1);
    //        if (randL >= randK) randL++;
    //        new_det.set_alfa_bit(cume_ncmopi_[i]+randI,false);
    //        new_det.set_alfa_bit(cume_ncmopi_[i]+randJ,false);
    //        new_det.set_alfa_bit(cume_ncmopi_[k]+randK,true);
    //        new_det.set_alfa_bit(cume_ncmopi_[k]+randL,true);
    //        return;
    //    }else if (end<cume_excit_irrep_[2]) {
    //        l=i^j^k;
    //        randI = rand_int()%obtCount.naocc[i];
    //        randJ = rand_int()%obtCount.naocc[j];
    //        randK = rand_int()%obtCount.navir[k];
    //        randL = rand_int()%obtCount.navir[l];
    //        new_det.set_alfa_bit(cume_ncmopi_[i]+randI,false);
    //        new_det.set_alfa_bit(cume_ncmopi_[j]+randJ,false);
    //        new_det.set_alfa_bit(cume_ncmopi_[k]+randK,true);
    //        new_det.set_alfa_bit(cume_ncmopi_[l]+randL,true);
    //        return;
    //    }else if (end<cume_excit_irrep_[3]) {
    //        randI = rand_int()%obtCount.nbocc[i];
    //        randJ = rand_int()%(obtCount.nbocc[i]-1);
    //        if (randJ >= randI) randJ++;
    //        randK = rand_int()%obtCount.nbvir[k];
    //        randL = rand_int()%(obtCount.nbvir[k]-1);
    //        if (randL >= randK) randL++;
    //        new_det.set_beta_bit(cume_ncmopi_[i]+randI,false);
    //        new_det.set_beta_bit(cume_ncmopi_[i]+randJ,false);
    //        new_det.set_beta_bit(cume_ncmopi_[k]+randK,true);
    //        new_det.set_beta_bit(cume_ncmopi_[k]+randL,true);
    //        return;
    //    }else if (end<cume_excit_irrep_[4]) {
    //        l=i^j^k;
    //        randI = rand_int()%obtCount.nbocc[i];
    //        randJ = rand_int()%obtCount.nbocc[j];
    //        randK = rand_int()%obtCount.nbvir[k];
    //        randL = rand_int()%obtCount.nbvir[l];
    //        new_det.set_beta_bit(cume_ncmopi_[i]+randI,false);
    //        new_det.set_beta_bit(cume_ncmopi_[j]+randJ,false);
    //        new_det.set_beta_bit(cume_ncmopi_[k]+randK,true);
    //        new_det.set_beta_bit(cume_ncmopi_[l]+randL,true);
    //        return;
    //    }else if (end<cume_excit_irrep_[5]) {
    //        randI = rand_int()%obtCount.naocc[i];
    //        randJ = rand_int()%obtCount.navir[j];
    //        new_det.set_alfa_bit(cume_ncmopi_[i]+randI,false);
    //        new_det.set_alfa_bit(cume_ncmopi_[j]+randJ,true);
    //        if (k != -1) {
    //            outfile->Printf("Error in spawning excitation:k is not -1 for
    //            single ext");
    //        }
    //        return;
    //    }else if (end<cume_excit_irrep_[6]) {
    //        randI = rand_int()%obtCount.nbocc[i];
    //        randJ = rand_int()%obtCount.nbvir[j];
    //        new_det.set_beta_bit(cume_ncmopi_[i]+randI,false);
    //        new_det.set_beta_bit(cume_ncmopi_[j]+randJ,true);
    //        if (k != -1) {
    //            outfile->Printf("Error in spawning excitation:k is not -1 for
    //            single ext");
    //        }
    //        return;
    //    }
    //    outfile->Printf("Error in spawning excitation:end of choice");
}

void FCIQMC::detSingleExcitation(Determinant& new_det, std::tuple<size_t, size_t>& rand_ext) {
    size_t ii, aa;
    std::tie(ii, aa) = rand_ext;
    if (ii < ncmo_) {
        new_det.set_alfa_bit(ii, false);
        new_det.set_alfa_bit(aa, true);
    } else {
        new_det.set_beta_bit(ii - ncmo_, false);
        new_det.set_beta_bit(aa - ncmo_, true);
    }
}

void FCIQMC::detDoubleExcitation(Determinant& new_det,
                                 std::tuple<size_t, size_t, size_t, size_t>& rand_ext) {
    size_t ii, aa, jj, bb;
    std::tie(ii, aa, jj, bb) = rand_ext;
    if (ii >= ncmo_) {
        new_det.set_beta_bit(ii - ncmo_, false);
        new_det.set_beta_bit(jj - ncmo_, false);
        new_det.set_beta_bit(aa - ncmo_, true);
        new_det.set_beta_bit(bb - ncmo_, true);
    } else if (jj >= ncmo_) {
        new_det.set_alfa_bit(ii, false);
        new_det.set_alfa_bit(aa, true);
        new_det.set_beta_bit(jj - ncmo_, false);
        new_det.set_beta_bit(bb - ncmo_, true);
    } else {
        new_det.set_alfa_bit(ii, false);
        new_det.set_alfa_bit(jj, false);
        new_det.set_alfa_bit(aa, true);
        new_det.set_alfa_bit(bb, true);
    }
}

double FCIQMC::count_walkers(walker_map& walkers) {
    timer_on("FCIQMC:CountWalker");
    double countWalkers = 0;
    for (auto walker : walkers) {
        double Cwalker = walker.second;
        countWalkers += std::fabs(Cwalker);
    }
    timer_off("FCIQMC:CountWalker");
    nDets_ = walkers.size();
    nWalkers_ = countWalkers;
    return countWalkers;
}

double FCIQMC::compute_proj_energy(Determinant& ref, walker_map& walkers) {
    timer_on("FCIQMC:Calc_Eproj");
    double Cref = walkers[ref];
    Eproj_ = nuclear_repulsion_energy_;
    for (auto walker : walkers) {
        const Determinant& det = walker.first;
        double Cwalker = walker.second;
        Eproj_ += fci_ints_->slater_rules(ref, det) * Cwalker / Cref;
    }
    timer_off("FCIQMC:Calc_Eproj");
    return Eproj_;
}

double FCIQMC::compute_var_energy(walker_map& walkers) {
    timer_on("FCIQMC:Calc_Evar");
    Evar_ = 0;
    double mod2 = 0.0;
    for (auto walker : walkers) {
        const Determinant& det = walker.first;
        double Cwalker = walker.second;
        mod2 += Cwalker * Cwalker;
        for (auto walker2 : walkers) {
            const Determinant& det2 = walker2.first;
            double Cwalker2 = walker2.second;
            Evar_ += fci_ints_->slater_rules(det, det2) * Cwalker * Cwalker2;
        }
        //        det.print();
        //        outfile->Printf("\nCwalker: %lf",Cwalker);
    }
    Evar_ /= mod2;
    Evar_ += nuclear_repulsion_energy_;
    timer_off("FCIQMC:Calc_Evar");
    return Evar_;
}

void FCIQMC::print_iter_info(size_t iter) {
    outfile->Printf("\n%8zu %8.4lf %10zu %10zu %20.12lf %20.12lf %20.12lf "
                    "%20.12lf %20.12lf %20.12lf",
                    iter, iter * time_step_, (size_t)nWalkers_, (size_t)nDets_, Eproj_, AvgEproj_,
                    shift_, AvgShift_, shift_ + Ehf_ + nuclear_repulsion_energy_,
                    AvgShift_ + Ehf_ + nuclear_repulsion_energy_);
    //    outfile->Printf("\niter:%zu ended with %zu dets",iter,
    //    (size_t)nDets_);
    //    if (countWalkers)
    //        outfile->Printf(", %lf walkers", nWalkers_);
    //    if (calcEproj)
    //        outfile->Printf(", proj E=%.12lf", Eproj_);
    //    if (calcEvar)
    //        outfile->Printf(", var E=%.12lf", Evar_);
    //    if (do_shift_)
    //        outfile->Printf(", shift+Ehf=%.12lf",
    //        shift_+Ehf_+nuclear_repulsion_energy_);
}

void FCIQMC::compute_avg_Eproj(std::vector<double> Eprojs) {
    double sum = 0.0;
    for (size_t iter = iter_ * 3 / 4; iter < iter_; iter++) {
        sum += Eprojs[iter];
        //        outfile->Printf("\n%d\t: %.12lf",iter, Eprojs[iter]);
    }

    AvgEproj_ = sum / (iter_ - iter_ * 3 / 4);
    //    outfile->Printf("\nAverage Eproj=%.12lf", sum);
}

void FCIQMC::compute_err_Eproj(std::vector<double> Eprojs) {
    double sum = 0.0;
    for (size_t iter = iter_ * 3 / 4; iter < iter_; iter++) {
        double diff = Eprojs[iter] - AvgEproj_;
        sum += diff * diff;
        //        outfile->Printf("\n%d\t: %.12lf",iter, Eprojs[iter]);
    }

    ErrEproj_ = std::sqrt(sum / (iter_ - iter_ * 3 / 4 - 1));
    //    outfile->Printf("\nAverage Eproj=%.12lf", sum);
}

void FCIQMC::compute_avg_shift(std::vector<double> shifts) {
    double sum = 0.0;
    for (double shift : shifts) {
        sum += shift;
    }
    AvgShift_ = sum / shifts.size();
    //    outfile->Printf("\nAverage shift=%.12lf, shift+Ehf=%.12lf", sum,
    //    sum+Ehf_+nuclear_repulsion_energy_);
}

void FCIQMC::compute_err_shift(std::vector<double> shifts) {
    double sum = 0.0;
    for (double shift : shifts) {
        double diff = shift - AvgShift_;
        sum += diff * diff;
    }
    ErrShift_ = sqrt(sum / (shifts.size() - 1));
    //    outfile->Printf("\nAverage shift=%.12lf, shift+Ehf=%.12lf", sum,
    //    sum+Ehf_+nuclear_repulsion_energy_);
}
}
} // EndNamepaces
