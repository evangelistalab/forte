/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2017 by its authors (see COPYING, COPYING.LESSER,
 * AUTHORS).
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
#include <numeric>
#include <string>
#include <vector>
#ifdef HAVE_MPI
#include <mpi.h>
#endif
#ifdef HAVE_GA
#include <ga.h>
#include <macdecls.h>
#else
#define GA_Nnodes() 1
#define GA_Nodeid() 0
#endif

#include "psi4/lib3index/dftensor.h"
#include "psi4/libmints/matrix.h"
#include "psi4/libmints/molecule.h"
#include "psi4/libmints/vector.h"
#include "psi4/libpsio/psio.h"
#include "psi4/libpsio/psio.hpp"
#include "psi4/libqt/qt.h"

#include "../ao_helper.h"
#include "../blockedtensorfactory.h"
#include "../fci/fci_solver.h"
#include "../fci/fci_vector.h"
#include "../fci_mo.h"
#include "three_dsrg_mrpt2.h"

using namespace ambit;

namespace psi {
namespace forte {

#ifdef _OPENMP
#include <omp.h>
bool THREE_DSRG_MRPT2::have_omp_ = true;
#else
#define omp_get_max_threads() 1
#define omp_get_thread_num() 0
bool THREE_DSRG_MRPT2::have_omp_ = false;
#endif

THREE_DSRG_MRPT2::THREE_DSRG_MRPT2(Reference reference, SharedWavefunction ref_wfn,
                                   Options& options, std::shared_ptr<ForteIntegrals> ints,
                                   std::shared_ptr<MOSpaceInfo> mo_space_info)
    : Wavefunction(options), reference_(reference), ints_(ints), tensor_type_(ambit::CoreTensor),
      BTF_(new BlockedTensorFactory(options)), mo_space_info_(mo_space_info) {
    shallow_copy(ref_wfn);
    reference_wavefunction_ = ref_wfn;

    ambit::BlockedTensor::reset_mo_spaces();

    num_threads_ = omp_get_max_threads();
    /// Get processor number
    int nproc = 1;
    int my_proc = 0;
#ifdef HAVE_MPI
    nproc = MPI::COMM_WORLD.Get_size();
    my_proc = MPI::COMM_WORLD.Get_rank();
#endif

    std::string title_thread = std::to_string(num_threads_) + " thread";
    if (num_threads_ > 1) {
        title_thread += "s";
    }
    if (have_omp_) {
        title_thread += " (OMP)";
    }
    if (nproc > 1) {
        title_thread += " and " + std::to_string(nproc) + " Process";
    }

    print_method_banner({"DF/CD - Driven Similarity Renormalization Group MBPT2",
                         "Kevin Hannon and Chenyang (York) Li", title_thread});
    outfile->Printf("\n    References:");
    outfile->Printf("\n      u-DSRG-MRPT2:    J. Chem. Theory Comput. 2015, 11, 2097.");
    outfile->Printf("\n      DF/CD-DSRG-MRPT2:  J. Chem. Phys. 2016, 144, 204111.");
    outfile->Printf("\n      (pr-)DSRG-MRPT2: J. Chem. Phys. 2017, 146, 124132.");
    outfile->Printf("\n");

    if (options_.get_bool("MEMORY_SUMMARY")) {
        BTF_->print_memory_info();
    }
    ref_type_ = options_.get_str("REFERENCE");
    outfile->Printf("\n  Reference = %s", ref_type_.c_str());
    if (options_.get_bool("THREE_MRPT2_TIMINGS")) {
        detail_time_ = true;
    }

    // printf("\n P%d about to enter startup", my_proc);
    // GA_Sync();
    startup();
    if (my_proc == 0)
        print_summary();
}

THREE_DSRG_MRPT2::~THREE_DSRG_MRPT2() { cleanup(); }

void THREE_DSRG_MRPT2::startup() {
    int nproc = 1;
    int my_proc = 0;
#ifdef HAVE_MPI
    nproc = MPI::COMM_WORLD.Get_size();
    my_proc = MPI::COMM_WORLD.Get_rank();
#endif

    if (my_proc == 0) {
        frozen_core_energy_ = ints_->frozen_core_energy();
        Eref_ = reference_.get_Eref();
        outfile->Printf("\n  Reference Energy = %.15f", Eref_);
    }

    ncmopi_ = mo_space_info_->get_dimension("CORRELATED");
    ncmo_ = mo_space_info_->size("CORRELATED");

    s_ = options_.get_double("DSRG_S");
    if (s_ < 0) {
        outfile->Printf("\n  S parameter for DSRG must >= 0!");
        exit(1);
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
                        "in DSRG-MRPT2. Changed to STANDARD.",
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

    // initialize timer for commutator
    dsrg_time_ = DSRG_TIME();

    // include internal amplitudes or not
    internal_amp_ = options_.get_str("INTERNAL_AMP") != "NONE";

    rdoccpi_ = mo_space_info_->get_dimension("RESTRICTED_DOCC");
    actvpi_ = mo_space_info_->get_dimension("ACTIVE");
    ruoccpi_ = mo_space_info_->get_dimension("RESTRICTED_UOCC");

    acore_mos_ = mo_space_info_->get_corr_abs_mo("RESTRICTED_DOCC");
    bcore_mos_ = mo_space_info_->get_corr_abs_mo("RESTRICTED_DOCC");
    aactv_mos_ = mo_space_info_->get_corr_abs_mo("ACTIVE");
    bactv_mos_ = mo_space_info_->get_corr_abs_mo("ACTIVE");
    avirt_mos_ = mo_space_info_->get_corr_abs_mo("RESTRICTED_UOCC");
    bvirt_mos_ = mo_space_info_->get_corr_abs_mo("RESTRICTED_UOCC");

    BlockedTensor::set_expert_mode(true);

    BTF_->add_mo_space("c", "m,n,µ,π", acore_mos_, AlphaSpin);
    BTF_->add_mo_space("C", "M,N,Ω,∏", bcore_mos_, BetaSpin);

    core_ = acore_mos_.size();

    BTF_->add_mo_space("a", "uvwxyz123", aactv_mos_, AlphaSpin);
    BTF_->add_mo_space("A", "UVWXYZ!@#", bactv_mos_, BetaSpin);
    active_ = aactv_mos_.size();

    BTF_->add_mo_space("v", "e,f,ε,φ", avirt_mos_, AlphaSpin);
    BTF_->add_mo_space("V", "E,F,Ƒ,Ǝ", bvirt_mos_, BetaSpin);
    virtual_ = avirt_mos_.size();

    BTF_->add_composite_mo_space("h", "ijkl", {"c", "a"});
    BTF_->add_composite_mo_space("H", "IJKL", {"C", "A"});

    BTF_->add_composite_mo_space("p", "abcd", {"a", "v"});
    BTF_->add_composite_mo_space("P", "ABCD", {"A", "V"});

    BTF_->add_composite_mo_space("g", "pqrs", {"c", "a", "v"});
    BTF_->add_composite_mo_space("G", "PQRS", {"C", "A", "V"});

    // These two blocks of functions create a Blocked tensor
    std::vector<std::string> hhpp_no_cv = BTF_->generate_indices("cav", "hhpp");
    no_hhpp_ = hhpp_no_cv;

    if (my_proc == 0)
        nthree_ = ints_->nthree();
    Timer naux_bcast;
#ifdef HAVE_MPI
    MPI_Bcast(&nthree_, 1, MPI_INT, 0, MPI_COMM_WORLD);
// printf("\n P%d took %8.8f s to broadcast %d size", my_proc, naux_bcast.get(),
// nthree_);
#endif

    if (my_proc == 0) {
        std::vector<size_t> nauxpi(nthree_);
        std::iota(nauxpi.begin(), nauxpi.end(), 0);
        BTF_->add_mo_space("d", "g", nauxpi, NoSpin);

        H_ = BTF_->build(tensor_type_, "H", spin_cases({"gg"}));

        Gamma1_ = BTF_->build(tensor_type_, "Gamma1_", spin_cases({"hh"}));
        Eta1_ = BTF_->build(tensor_type_, "Eta1_", spin_cases({"pp"}));
        F_ = BTF_->build(tensor_type_, "Fock", spin_cases({"gg"}));
        F_no_renorm_ = BTF_->build(tensor_type_, "Fock", spin_cases({"gg"}));
        Delta1_ = BTF_->build(tensor_type_, "Delta1_", spin_cases({"aa"}));

        RDelta1_ = BTF_->build(tensor_type_, "RDelta1_", spin_cases({"hp"}));

        T1_ = BTF_->build(tensor_type_, "T1 Amplitudes", spin_cases({"hp"}));

        RExp1_ = BTF_->build(tensor_type_, "RExp1", spin_cases({"hp"}));

        H_.iterate(
            [&](const std::vector<size_t>& i, const std::vector<SpinType>& spin, double& value) {
                if (spin[0] == AlphaSpin)
                    value = ints_->oei_a(i[0], i[1]);
                else
                    value = ints_->oei_b(i[0], i[1]);
            });

        ambit::Tensor Gamma1_cc = Gamma1_.block("cc");
        ambit::Tensor Gamma1_aa = Gamma1_.block("aa");
        ambit::Tensor Gamma1_CC = Gamma1_.block("CC");
        ambit::Tensor Gamma1_AA = Gamma1_.block("AA");

        ambit::Tensor Eta1_aa = Eta1_.block("aa");
        ambit::Tensor Eta1_vv = Eta1_.block("vv");
        ambit::Tensor Eta1_AA = Eta1_.block("AA");
        ambit::Tensor Eta1_VV = Eta1_.block("VV");

        Gamma1_cc.iterate(
            [&](const std::vector<size_t>& i, double& value) { value = i[0] == i[1] ? 1.0 : 0.0; });
        Gamma1_CC.iterate(
            [&](const std::vector<size_t>& i, double& value) { value = i[0] == i[1] ? 1.0 : 0.0; });

        Eta1_aa.iterate(
            [&](const std::vector<size_t>& i, double& value) { value = i[0] == i[1] ? 1.0 : 0.0; });
        Eta1_AA.iterate(
            [&](const std::vector<size_t>& i, double& value) { value = i[0] == i[1] ? 1.0 : 0.0; });

        Eta1_vv.iterate(
            [&](const std::vector<size_t>& i, double& value) { value = i[0] == i[1] ? 1.0 : 0.0; });
        Eta1_VV.iterate(
            [&](const std::vector<size_t>& i, double& value) { value = i[0] == i[1] ? 1.0 : 0.0; });

        Gamma1_aa("pq") = reference_.L1a()("pq");
        Gamma1_AA("pq") = reference_.L1b()("pq");

        Eta1_aa("pq") -= reference_.L1a()("pq");
        Eta1_AA("pq") -= reference_.L1b()("pq");

        // printf("\n Settingup reference shit begin P%d", my_proc);
        // Compute the fock matrix from the reference.  Make sure fock matrix is
        // updated in integrals class.
        std::shared_ptr<Matrix> Gamma1_matrixA(new Matrix("Gamma1_RDM", ncmo_, ncmo_));
        std::shared_ptr<Matrix> Gamma1_matrixB(new Matrix("Gamma1_RDM", ncmo_, ncmo_));
        for (size_t m = 0; m < core_; m++) {
            Gamma1_matrixA->set(acore_mos_[m], acore_mos_[m], 1.0);
            Gamma1_matrixB->set(bcore_mos_[m], bcore_mos_[m], 1.0);
        }
        Gamma1_aa.iterate([&](const std::vector<size_t>& i, double& value) {
            Gamma1_matrixA->set(aactv_mos_[i[0]], aactv_mos_[i[1]], value);
        });

        Gamma1_AA.iterate([&](const std::vector<size_t>& i, double& value) {
            Gamma1_matrixB->set(bactv_mos_[i[0]], bactv_mos_[i[1]], value);
        });
        ints_->make_fock_matrix(Gamma1_matrixA, Gamma1_matrixB);

        F_.iterate(
            [&](const std::vector<size_t>& i, const std::vector<SpinType>& spin, double& value) {
                if (spin[0] == AlphaSpin) {
                    value = ints_->get_fock_a(i[0], i[1]);
                } else if (spin[0] == BetaSpin) {
                    value = ints_->get_fock_b(i[0], i[1]);
                }
            });
        F_no_renorm_["pq"] = F_["pq"];
        F_no_renorm_["PQ"] = F_["PQ"];

        Fa_.resize(ncmo_);
        Fb_.resize(ncmo_);

        for (size_t p = 0; p < ncmo_; p++) {
            Fa_[p] = ints_->get_fock_a(p, p);
            Fb_[p] = ints_->get_fock_b(p, p);
        }
    }

    // Prepare Hbar
    bool relaxRef =
        (options_.get_str("RELAX_REF") != "NONE") || (options_["AVG_STATE"].size() != 0);
    std::string relax_ref = options_.get_str("RELAX_REF");
    if (relax_ref != "NONE" && relax_ref != "ONCE") {
        if (relax_ref != "ONCE") {
            outfile->Printf("\n  Warning: RELAX_REF option \"%s\" is not "
                            "supported. Change to ONCE",
                            relax_ref.c_str());
            relax_ref = "ONCE";
        }
    }

    if (relaxRef) {
        if (my_proc == 0) {
            Hbar1_ = BTF_->build(tensor_type_, "One-body Hbar", spin_cases({"aa"}));
            Hbar2_ = BTF_->build(tensor_type_, "Two-body Hbar", spin_cases({"aaaa"}));
            Hbar1_["uv"] = F_["uv"];
            Hbar1_["UV"] = F_["UV"];
        }
    }

    print_ = options_.get_int("PRINT");

    if (my_proc == 0) {
        if (print_ > 1) {
            Gamma1_.print(stdout);
            Eta1_.print(stdout);
            F_.print(stdout);
            H_.print(stdout);
        }

        Delta1_.iterate(
            [&](const std::vector<size_t>& i, const std::vector<SpinType>& spin, double& value) {
                if (spin[0] == AlphaSpin) {
                    value = Fa_[i[0]] - Fa_[i[1]];
                } else if (spin[0] == BetaSpin) {
                    value = Fb_[i[0]] - Fb_[i[1]];
                }
            });

        RDelta1_.iterate(
            [&](const std::vector<size_t>& i, const std::vector<SpinType>& spin, double& value) {
                if (spin[0] == AlphaSpin) {
                    value = dsrg_source_->compute_renormalized_denominator(Fa_[i[0]] - Fa_[i[1]]);
                } else if (spin[0] == BetaSpin) {
                    value = dsrg_source_->compute_renormalized_denominator(Fb_[i[0]] - Fb_[i[1]]);
                }
            });

        // Fill out Lambda2_ and Lambda3_
        Lambda2_ = BTF_->build(tensor_type_, "Lambda2_", spin_cases({"aaaa"}));
        ambit::Tensor Lambda2_aa = Lambda2_.block("aaaa");
        ambit::Tensor Lambda2_aA = Lambda2_.block("aAaA");
        ambit::Tensor Lambda2_AA = Lambda2_.block("AAAA");
        Lambda2_aa("pqrs") = reference_.L2aa()("pqrs");
        Lambda2_aA("pqrs") = reference_.L2ab()("pqrs");
        Lambda2_AA("pqrs") = reference_.L2bb()("pqrs");

        // Prepare exponential tensors for effective Fock matrix and integrals
        RExp1_.iterate(
            [&](const std::vector<size_t>& i, const std::vector<SpinType>& spin, double& value) {
                if (spin[0] == AlphaSpin) {
                    value = dsrg_source_->compute_renormalized(Fa_[i[0]] - Fa_[i[1]]);
                } else if (spin[0] == BetaSpin) {
                    value = dsrg_source_->compute_renormalized(Fb_[i[0]] - Fb_[i[1]]);
                }
            });

        print_ = options_.get_int("PRINT");
        if (print_ > 1) {
            Gamma1_.print(stdout);
            Eta1_.print(stdout);
            F_.print(stdout);
            H_.print(stdout);
        }
        if (print_ > 2) {
            Lambda2_.print(stdout);
        }
    }
    integral_type_ = ints_->integral_type();
    // GA_Sync();
    // printf("\n P%d integral_type", my_proc);

    if (integral_type_ != DiskDF) {
        if (my_proc == 0) {
            std::vector<std::string> list_of_pphh_V = BTF_->generate_indices("vac", "pphh");
            V_ = BTF_->build(tensor_type_, "V_", BTF_->spin_cases_avoid(list_of_pphh_V, 1));
            T2_ = BTF_->build(tensor_type_, "T2 Amplitudes", BTF_->spin_cases_avoid(no_hhpp_, 1));
            ThreeIntegral_ = BTF_->build(tensor_type_, "ThreeInt", {"dph", "dPH"});

            std::vector<std::string> ThreeInt_block = ThreeIntegral_.block_labels();

            std::map<std::string, std::vector<size_t>> mo_to_index = BTF_->get_mo_to_index();

            for (std::string& string_block : ThreeInt_block) {
                std::string pos1(1, string_block[0]);
                std::string pos2(1, string_block[1]);
                std::string pos3(1, string_block[2]);

                std::vector<size_t> first_index = mo_to_index[pos1];
                std::vector<size_t> second_index = mo_to_index[pos2];
                std::vector<size_t> third_index = mo_to_index[pos3];

                ambit::Tensor ThreeIntegral_block =
                    ints_->three_integral_block(first_index, second_index, third_index);
                ThreeIntegral_.block(string_block).copy(ThreeIntegral_block);
            }
            V_["abij"] = ThreeIntegral_["gai"] * ThreeIntegral_["gbj"];
            V_["abij"] -= ThreeIntegral_["gaj"] * ThreeIntegral_["gbi"];

            V_["aBiJ"] = ThreeIntegral_["gai"] * ThreeIntegral_["gBJ"];

            V_["ABIJ"] = ThreeIntegral_["gAI"] * ThreeIntegral_["gBJ"];
            V_["ABIJ"] -= ThreeIntegral_["gAJ"] * ThreeIntegral_["gBI"];
        }
    }
}

void THREE_DSRG_MRPT2::print_summary() {
    // Print a summary
    std::vector<std::pair<std::string, int>> calculation_info;

    std::vector<std::pair<std::string, double>> calculation_info_double{
        {"Flow parameter", s_},
        {"Cholesky Tolerance", options_.get_double("CHOLESKY_TOLERANCE")},
        {"Taylor expansion threshold", std::pow(10.0, -double(taylor_threshold_))}};

    std::vector<std::pair<std::string, std::string>> calculation_info_string{
        {"int_type", options_.get_str("INT_TYPE")},
        {"ccvv_algorithm", options_.get_str("ccvv_algorithm")},
        {"ccvv_source", options_.get_str("CCVV_SOURCE")}};

    // Print some information
    print_h2("Calculation Information");
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

void THREE_DSRG_MRPT2::cleanup() { dsrg_time_.print_comm_time(); }

double THREE_DSRG_MRPT2::compute_energy() {
    Timer ComputeEnergy;
    int my_proc = 0;
    int nproc = 1;
#ifdef HAVE_MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &my_proc);
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);
#endif

    // check semi-canonical orbitals
    if (my_proc == 0)
        semi_canonical_ = check_semicanonical();
    if (!semi_canonical_) {
        outfile->Printf("\n    Warning: DF/CD-DSRG-MRPT2 only takes semi-canonical orbitals. "
                        "The code will keep running.");

        //            U_ =
        //            ambit::BlockedTensor::build(tensor_type_,"U",spin_cases({"gg"}));
        //            std::vector<std::vector<double>> eigens =
        //            diagonalize_Fock_diagblocks(U_);
        //            Fa_ = eigens[0];
        //            Fb_ = eigens[1];
    }

    // Compute T2 and T1
    if (integral_type_ != DiskDF) {
        compute_t2();
    }
    if (integral_type_ != DiskDF && my_proc == 0) {
        renormalize_V();
    }
    if (integral_type_ == DiskDF && my_proc == 0) {
        size_t memory_cost = nmo_ * nmo_ * nmo_ * active_ * 16;
        bool exceed_memory = memory_cost < Process::environment.get_memory();
        exceed_memory = false;

        std::vector<std::string> list_of_pphh_V = BTF_->generate_indices("vac", "pphh");
        std::string str = "Computing T2";
        // outfile->Printf("\n    %-37s ...", str.c_str());
        Timer T2timer;

        // If exceed memory, use diskbased algorithm
        // for all terms with <= 1 active idex
        // If not, just compute V in the beginning
        if (!exceed_memory) {
            T2_ = compute_T2_minimal(BTF_->spin_cases_avoid(no_hhpp_, 2));
        } else {
            T2_ = compute_T2_minimal(BTF_->spin_cases_avoid(no_hhpp_, 1));
        }
        outfile->Printf("      %-37s ...Done. Timing %15.6f s", str.c_str(), T2timer.get());

        std::string strV = "Computing V and Renormalizing";
        outfile->Printf("\n    %-37s ...", strV.c_str());
        Timer Vtimer;
        if (!exceed_memory) {
            V_ = compute_V_minimal(BTF_->spin_cases_avoid(list_of_pphh_V, 2));
        } else {
            V_ = compute_V_minimal(BTF_->spin_cases_avoid(list_of_pphh_V, 1));
        }
        outfile->Printf("...Done. Timing %15.6f s", Vtimer.get());
    }
    if (my_proc == 0) {
        compute_t1();
        check_t1();
    }

    // Compute effective integrals
    if (my_proc == 0)
        renormalize_F();
    if (print_ > 1 && my_proc == 0)
        F_.print(stdout); // The actv-actv block is different but OK.
    if (print_ > 1 && my_proc == 0)
        F_.print(stdout);
    if (print_ > 2 && my_proc == 0) {
        T1_.print(stdout);
    }

    // Compute DSRG-MRPT2 correlation energy
    double Etemp = 0.0;
    double EVT2 = 0.0;
    double Ecorr = 0.0;
    double Etotal = 0.0;
    std::vector<std::pair<std::string, double>> energy;
    if (my_proc == 0) {
        energy.push_back({"E0 (reference)", Eref_});

        Etemp = E_FT1();
        Ecorr += Etemp;
        energy.push_back({"<[F, T1]>", Etemp});

        Etemp = E_FT2();
        Ecorr += Etemp;
        energy.push_back({"<[F, T2]>", Etemp});

        Etemp = E_VT1();
        Ecorr += Etemp;
        energy.push_back({"<[V, T1]>", Etemp});
    }

    // printf("\n P%d about to enter E_VT2_2", my_proc);
    // GA_Sync();
    Etemp = E_VT2_2();
    if (my_proc == 0) {
        EVT2 += Etemp;
        energy.push_back({"<[V, T2]> (C_2)^4", Etemp});
        Etemp = E_VT2_4HH();
        EVT2 += Etemp;
        energy.push_back({"<[V, T2]> C_4 (C_2)^2 HH", Etemp});

        Etemp = E_VT2_4PP();
        EVT2 += Etemp;
        energy.push_back({"<[V, T2]> C_4 (C_2)^2 PP", Etemp});

        Etemp = E_VT2_4PH();
        EVT2 += Etemp;
        energy.push_back({"<[V, T2]> C_4 (C_2)^2 PH", Etemp});

        Etemp = E_VT2_6();
        EVT2 += Etemp;
        energy.push_back({"<[V, T2]> C_6 C_2", Etemp});

        Ecorr += EVT2;
        Etotal = Ecorr + Eref_;
        energy.push_back({"<[V, T2]>", EVT2});
        energy.push_back({"DSRG-MRPT2 correlation energy", Ecorr});
        energy.push_back({"DSRG-MRPT2 total energy", Etotal});

        // Analyze T1 and T2
        check_t1();
        energy.push_back({"max(T1)", T1max_});
        energy.push_back({"||T1||", T1norm_});

        // Print energy summary
        print_h2("DSRG-MRPT2 Energy Summary");
        for (auto& str_dim : energy)
            outfile->Printf("\n    %-30s = %22.15f", str_dim.first.c_str(), str_dim.second);
    }

    Process::environment.globals["CURRENT ENERGY"] = Etotal;

    if (my_proc == 0) {
        if (options_.get_bool("PRINT_DENOM2")) {
            std::ofstream myfile;
            myfile.open("DENOM.txt");
            ambit::BlockedTensor Delta2 =
                BTF_->build(tensor_type_, "Delta1_", {"cavv", "ccvv", "ccva"});
            Delta2.iterate([&](const std::vector<size_t>& i, const std::vector<SpinType>& spin,
                               double& value) {
                if (spin[0] == AlphaSpin) {
                    value = 1 / (Fa_[i[0]] + Fa_[i[1]] - Fa_[i[2]] - Fa_[i[3]]);
                }
            });
            ambit::Tensor Delta2ccvv = Delta2.block("ccvv");
            ambit::Tensor Delta2cavv = Delta2.block("cavv");
            ambit::Tensor Delta2ccva = Delta2.block("ccva");
            myfile << "ccvv DELTA2\n";
            int count = 0;
            Delta2ccvv.iterate([&](const std::vector<size_t>&, double& value) {
                myfile << count << "  " << value << "\n";
            });
            myfile << "cavv DELTA2\n";

            count = 0;
            Delta2cavv.iterate([&](const std::vector<size_t>&, double& value) {
                myfile << count << "  " << value << "\n";
            });
            myfile << "ccva DELTA2\n";

            count = 0;
            Delta2ccva.iterate([&](const std::vector<size_t>&, double& value) {
                myfile << count << "  " << value << "\n";
            });
        }
    }
// if(my_proc == 0)
//{
//    Hbar0_ = Etotal - Eref_;
//    if(options_.get_str("RELAX_REF") != "NONE")
//    {
//        relax_reference_once();
//    }
//}
#ifdef HAVE_MPI
    MPI_Bcast(&Etotal, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
#endif
    Process::environment.globals["CURRENT ENERGY"] = Etotal;

    if (my_proc == 0)
        Hbar0_ = Etotal - Eref_;
#ifdef HAVE_MPI
    MPI_Bcast(&Hbar0_, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
#endif

    // use relaxation code to do SA_FULL
    bool relaxRef =
        (options_.get_str("RELAX_REF") != "NONE") || (options_["AVG_STATE"].size() != 0);
    if (relaxRef) {
        if (my_proc == 0)
            relax_reference_once();
    }

    return Etotal;
}

double THREE_DSRG_MRPT2::compute_ref() {
    double E = 0.0;

    E = 0.5 * H_["ij"] * Gamma1_["ij"];
    E += 0.5 * F_["ij"] * Gamma1_["ij"];
    E += 0.5 * H_["IJ"] * Gamma1_["IJ"];
    E += 0.5 * F_["IJ"] * Gamma1_["IJ"];

    E += 0.25 * V_["uvxy"] * Lambda2_["uvxy"];
    E += 0.25 * V_["UVXY"] * Lambda2_["UVXY"];
    E += V_["uVxY"] * Lambda2_["uVxY"];

    std::shared_ptr<Molecule> molecule = Process::environment.molecule();
    double Enuc = molecule->nuclear_repulsion_energy();

    return E + frozen_core_energy_ + Enuc;
}
void THREE_DSRG_MRPT2::compute_t2() {
    int my_rank = 0;
#ifdef HAVE_MPI
    my_rank = MPI::COMM_WORLD.Get_rank();
#endif
    if (my_rank == 0) {
        std::string str = "Computing T2";
        outfile->Printf("\n    %-37s ...", str.c_str());
        Timer timer;

        T2_["ijab"] = V_["abij"];
        T2_["iJaB"] = V_["aBiJ"];
        T2_["IJAB"] = V_["ABIJ"];
        T2_.iterate(
            [&](const std::vector<size_t>& i, const std::vector<SpinType>& spin, double& value) {
                if (spin[0] == AlphaSpin && spin[1] == AlphaSpin) {
                    value *= dsrg_source_->compute_renormalized_denominator(Fa_[i[0]] + Fa_[i[1]] -
                                                                            Fa_[i[2]] - Fa_[i[3]]);
                } else if (spin[0] == BetaSpin && spin[1] == BetaSpin) {
                    value *= dsrg_source_->compute_renormalized_denominator(Fb_[i[0]] + Fb_[i[1]] -
                                                                            Fb_[i[2]] - Fb_[i[3]]);
                } else {
                    value *= dsrg_source_->compute_renormalized_denominator(Fa_[i[0]] + Fb_[i[1]] -
                                                                            Fa_[i[2]] - Fb_[i[3]]);
                }
            });

        // internal amplitudes (AA->AA)
        std::string internal_amp = options_.get_str("INTERNAL_AMP");
        std::string internal_amp_select = options_.get_str("INTERNAL_AMP_SELECT");
        if (internal_amp.find("DOUBLES") != string::npos) {
            size_t nactv1 = mo_space_info_->size("ACTIVE");
            size_t nactv2 = nactv1 * nactv1;
            size_t nactv3 = nactv2 * nactv1;
            size_t nactv_occ = actv_occ_mos_.size();
            size_t nactv_uocc = actv_uocc_mos_.size();

            if (internal_amp_select == "ALL") {
                for (size_t i = 0; i < nactv1; ++i) {
                    for (size_t j = 0; j < nactv1; ++j) {
                        size_t c = i * nactv1 + j;

                        for (size_t a = 0; a < nactv1; ++a) {
                            for (size_t b = 0; b < nactv1; ++b) {
                                size_t v = a * nactv1 + b;

                                if (c >= v) {
                                    size_t idx = i * nactv3 + j * nactv2 + a * nactv1 + b;
                                    for (const std::string& block : {"aaaa", "aAaA", "AAAA"}) {
                                        T2_.block(block).data()[idx] = 0.0;
                                    }
                                }
                            }
                        }
                    }
                }
            } else if (internal_amp_select == "OOVV") {
                for (const std::string& block : {"aaaa", "aAaA", "AAAA"}) {
                    // copy original data
                    std::vector<double> data(T2_.block(block).data());

                    T2_.block(block).zero();
                    for (size_t I = 0; I < nactv_occ; ++I) {
                        for (size_t J = 0; J < nactv_occ; ++J) {
                            for (size_t A = 0; A < nactv_uocc; ++A) {
                                for (size_t B = 0; B < nactv_uocc; ++B) {
                                    size_t idx = actv_occ_mos_[I] * nactv3 +
                                                 actv_occ_mos_[J] * nactv2 +
                                                 actv_uocc_mos_[A] * nactv1 + actv_uocc_mos_[B];
                                    T2_.block(block).data()[idx] = data[idx];
                                }
                            }
                        }
                    }
                }
            } else {
                for (const std::string& block : {"aaaa", "aAaA", "AAAA"}) {
                    // copy original data
                    std::vector<double> data(T2_.block(block).data());
                    T2_.block(block).zero();

                    // OO->VV
                    for (size_t I = 0; I < nactv_occ; ++I) {
                        for (size_t J = 0; J < nactv_occ; ++J) {
                            for (size_t A = 0; A < nactv_uocc; ++A) {
                                for (size_t B = 0; B < nactv_uocc; ++B) {
                                    size_t idx = actv_occ_mos_[I] * nactv3 +
                                                 actv_occ_mos_[J] * nactv2 +
                                                 actv_uocc_mos_[A] * nactv1 + actv_uocc_mos_[B];
                                    T2_.block(block).data()[idx] = data[idx];
                                }
                            }
                        }
                    }

                    // OO->OV, OO->VO
                    for (size_t I = 0; I < nactv_occ; ++I) {
                        for (size_t J = 0; J < nactv_occ; ++J) {
                            for (size_t K = 0; K < nactv_occ; ++K) {
                                for (size_t A = 0; A < nactv_uocc; ++A) {
                                    size_t idx = actv_occ_mos_[I] * nactv3 +
                                                 actv_occ_mos_[J] * nactv2 +
                                                 actv_occ_mos_[K] * nactv1 + actv_uocc_mos_[A];
                                    T2_.block(block).data()[idx] = data[idx];

                                    idx = actv_occ_mos_[I] * nactv3 + actv_occ_mos_[J] * nactv2 +
                                          actv_uocc_mos_[A] * nactv1 + actv_occ_mos_[K];
                                    T2_.block(block).data()[idx] = data[idx];
                                }
                            }
                        }
                    }

                    // OV->VV, VO->VV
                    for (size_t I = 0; I < nactv_occ; ++I) {
                        for (size_t A = 0; A < nactv_uocc; ++A) {
                            for (size_t B = 0; B < nactv_uocc; ++B) {
                                for (size_t C = 0; C < nactv_uocc; ++C) {
                                    size_t idx = actv_occ_mos_[I] * nactv3 +
                                                 actv_uocc_mos_[A] * nactv2 +
                                                 actv_uocc_mos_[B] * nactv1 + actv_uocc_mos_[C];
                                    T2_.block(block).data()[idx] = data[idx];

                                    idx = actv_uocc_mos_[A] * nactv3 + actv_occ_mos_[I] * nactv2 +
                                          actv_uocc_mos_[B] * nactv1 + actv_uocc_mos_[C];
                                    T2_.block(block).data()[idx] = data[idx];
                                }
                            }
                        }
                    }
                }
            }

        } else {
            T2_.block("aaaa").zero();
            T2_.block("aAaA").zero();
            T2_.block("AAAA").zero();
        }

        outfile->Printf("...Done. Timing %15.6f s", timer.get());
    }
}

ambit::BlockedTensor
THREE_DSRG_MRPT2::compute_T2_minimal(const std::vector<std::string>& t2_spaces) {
    ambit::BlockedTensor T2min;

    T2min = BTF_->build(tensor_type_, "T2min", t2_spaces, true);
    Timer timer_b_min;
    ambit::BlockedTensor ThreeInt = compute_B_minimal(t2_spaces);
    if (detail_time_)
        outfile->Printf("\n Took %8.4f s to compute_B_minimal", timer_b_min.get());
    Timer v_t2;
    T2min["ijab"] = (ThreeInt["gia"] * ThreeInt["gjb"]);
    T2min["ijab"] -= (ThreeInt["gib"] * ThreeInt["gja"]);
    T2min["IJAB"] = (ThreeInt["gIA"] * ThreeInt["gJB"]);
    T2min["IJAB"] -= (ThreeInt["gIB"] * ThreeInt["gJA"]);
    T2min["iJaB"] = (ThreeInt["gia"] * ThreeInt["gJB"]);
    if (detail_time_)
        outfile->Printf("\n Took %8.4f s to compute T2 from B", v_t2.get());

    Timer t2_iterate;
    T2min.iterate(
        [&](const std::vector<size_t>& i, const std::vector<SpinType>& spin, double& value) {
            if (spin[0] == AlphaSpin && spin[1] == AlphaSpin) {
                value *= dsrg_source_->compute_renormalized_denominator(Fa_[i[0]] + Fa_[i[1]] -
                                                                        Fa_[i[2]] - Fa_[i[3]]);
            } else if (spin[0] == BetaSpin && spin[1] == BetaSpin) {
                value *= dsrg_source_->compute_renormalized_denominator(Fb_[i[0]] + Fb_[i[1]] -
                                                                        Fb_[i[2]] - Fb_[i[3]]);
            } else {
                value *= dsrg_source_->compute_renormalized_denominator(Fa_[i[0]] + Fb_[i[1]] -
                                                                        Fa_[i[2]] - Fb_[i[3]]);
            }
        });
    if (detail_time_)
        outfile->Printf("\n T2 iteration takes %8.4f s", t2_iterate.get());

    // internal amplitudes (AA->AA)
    std::string internal_amp = options_.get_str("INTERNAL_AMP");
    std::string internal_amp_select = options_.get_str("INTERNAL_AMP_SELECT");

    for (const std::string& block : {"aaaa", "aAaA", "AAAA"}) {
        if (std::find(t2_spaces.begin(), t2_spaces.end(), block) != t2_spaces.end()) {

            if (internal_amp.find("DOUBLES") != string::npos) {
                size_t nactv1 = mo_space_info_->size("ACTIVE");
                size_t nactv2 = nactv1 * nactv1;
                size_t nactv3 = nactv2 * nactv1;
                size_t nactv_occ = actv_occ_mos_.size();
                size_t nactv_uocc = actv_uocc_mos_.size();

                if (internal_amp_select == "ALL") {
                    for (size_t i = 0; i < nactv1; ++i) {
                        for (size_t j = 0; j < nactv1; ++j) {
                            size_t c = i * nactv1 + j;

                            for (size_t a = 0; a < nactv1; ++a) {
                                for (size_t b = 0; b < nactv1; ++b) {
                                    size_t v = a * nactv1 + b;

                                    if (c >= v) {
                                        size_t idx = i * nactv3 + j * nactv2 + a * nactv1 + b;
                                        T2min.block(block).data()[idx] = 0.0;
                                    }
                                }
                            }
                        }
                    }
                } else if (internal_amp_select == "OOVV") {
                    // copy original data
                    std::vector<double> data(T2min.block(block).data());

                    T2min.block(block).zero();
                    for (size_t I = 0; I < nactv_occ; ++I) {
                        for (size_t J = 0; J < nactv_occ; ++J) {
                            for (size_t A = 0; A < nactv_uocc; ++A) {
                                for (size_t B = 0; B < nactv_uocc; ++B) {
                                    size_t idx = actv_occ_mos_[I] * nactv3 +
                                                 actv_occ_mos_[J] * nactv2 +
                                                 actv_uocc_mos_[A] * nactv1 + actv_uocc_mos_[B];
                                    T2min.block(block).data()[idx] = data[idx];
                                }
                            }
                        }
                    }
                } else {
                    // copy original data
                    std::vector<double> data(T2min.block(block).data());
                    T2min.block(block).zero();

                    // OO->VV
                    for (size_t I = 0; I < nactv_occ; ++I) {
                        for (size_t J = 0; J < nactv_occ; ++J) {
                            for (size_t A = 0; A < nactv_uocc; ++A) {
                                for (size_t B = 0; B < nactv_uocc; ++B) {
                                    size_t idx = actv_occ_mos_[I] * nactv3 +
                                                 actv_occ_mos_[J] * nactv2 +
                                                 actv_uocc_mos_[A] * nactv1 + actv_uocc_mos_[B];
                                    T2min.block(block).data()[idx] = data[idx];
                                }
                            }
                        }
                    }

                    // OO->OV, OO->VO
                    for (size_t I = 0; I < nactv_occ; ++I) {
                        for (size_t J = 0; J < nactv_occ; ++J) {
                            for (size_t K = 0; K < nactv_occ; ++K) {
                                for (size_t A = 0; A < nactv_uocc; ++A) {
                                    size_t idx = actv_occ_mos_[I] * nactv3 +
                                                 actv_occ_mos_[J] * nactv2 +
                                                 actv_occ_mos_[K] * nactv1 + actv_uocc_mos_[A];
                                    T2min.block(block).data()[idx] = data[idx];

                                    idx = actv_occ_mos_[I] * nactv3 + actv_occ_mos_[J] * nactv2 +
                                          actv_uocc_mos_[A] * nactv1 + actv_occ_mos_[K];
                                    T2min.block(block).data()[idx] = data[idx];
                                }
                            }
                        }
                    }

                    // OV->VV, VO->VV
                    for (size_t I = 0; I < nactv_occ; ++I) {
                        for (size_t A = 0; A < nactv_uocc; ++A) {
                            for (size_t B = 0; B < nactv_uocc; ++B) {
                                for (size_t C = 0; C < nactv_uocc; ++C) {
                                    size_t idx = actv_occ_mos_[I] * nactv3 +
                                                 actv_uocc_mos_[A] * nactv2 +
                                                 actv_uocc_mos_[B] * nactv1 + actv_uocc_mos_[C];
                                    T2min.block(block).data()[idx] = data[idx];

                                    idx = actv_uocc_mos_[A] * nactv3 + actv_occ_mos_[I] * nactv2 +
                                          actv_uocc_mos_[B] * nactv1 + actv_uocc_mos_[C];
                                    T2min.block(block).data()[idx] = data[idx];
                                }
                            }
                        }
                    }
                } // end internal selection

            } else {
                T2min.block(block).zero();
            } // end internal
        }     // end block existence test
    }         // end block labels loop

    return T2min;
}

ambit::BlockedTensor THREE_DSRG_MRPT2::compute_V_minimal(const std::vector<std::string>& spaces,
                                                         bool renormalize) {
    ambit::BlockedTensor Vmin = BTF_->build(tensor_type_, "Vmin", spaces, true);
    ambit::BlockedTensor ThreeInt;
    Timer computeB;
    ThreeInt = compute_B_minimal(spaces);
    if (detail_time_) {
        outfile->Printf("\n  Compute B minimal takes %8.6f s", computeB.get());
    }
    Timer ComputeV;
    Vmin["abij"] = ThreeInt["gai"] * ThreeInt["gbj"];
    Vmin["abij"] -= ThreeInt["gaj"] * ThreeInt["gbi"];
    Vmin["ABIJ"] = ThreeInt["gAI"] * ThreeInt["gBJ"];
    Vmin["ABIJ"] -= ThreeInt["gAJ"] * ThreeInt["gBI"];
    Vmin["aBiJ"] = ThreeInt["gai"] * ThreeInt["gBJ"];
    if (detail_time_) {
        outfile->Printf("\n  Compute V from B takes %8.6f s", ComputeV.get());
    }

    if (renormalize) {
        Timer RenormV;
        Vmin.iterate(
            [&](const std::vector<size_t>& i, const std::vector<SpinType>& spin, double& value) {
                if ((spin[0] == AlphaSpin) and (spin[1] == AlphaSpin)) {
                    value = (value +
                             value *
                                 dsrg_source_->compute_renormalized(Fa_[i[0]] + Fa_[i[1]] -
                                                                    Fa_[i[2]] - Fa_[i[3]]));
                } else if ((spin[0] == AlphaSpin) and (spin[1] == BetaSpin)) {
                    value = (value +
                             value *
                                 dsrg_source_->compute_renormalized(Fa_[i[0]] + Fb_[i[1]] -
                                                                    Fa_[i[2]] - Fb_[i[3]]));
                } else if ((spin[0] == BetaSpin) and (spin[1] == BetaSpin)) {
                    value = (value +
                             value *
                                 dsrg_source_->compute_renormalized(Fb_[i[0]] + Fb_[i[1]] -
                                                                    Fb_[i[2]] - Fb_[i[3]]));
                }
            });
        if (detail_time_) {
            outfile->Printf("\n  RenormalizeV takes %8.6f s.", RenormV.get());
        }
    }
    return Vmin;
}

ambit::BlockedTensor THREE_DSRG_MRPT2::compute_B_minimal(const std::vector<std::string>& spaces) {
    std::vector<size_t> nauxpi(nthree_);
    std::iota(nauxpi.begin(), nauxpi.end(), 0);

    // BlockedTensor::add_mo_space("@","$",nauxpi,NoSpin);
    // BlockedTensor::add_mo_space("d","g",nauxpi,NoSpin);
    std::vector<std::string> ThreeIntegral_labels;
    for (const auto& label : spaces) {
        std::string left_threeint;
        std::string right_threeint;
        left_threeint += "d";
        right_threeint += "d";

        // Since aAaA-> (aa)(AA) -> ThreeInt
        if (std::islower(label[0]) && std::isupper(label[1]) && std::islower(label[2]) &&
            std::isupper(label[3])) {
            left_threeint += label[0];
            left_threeint += label[2];

            if (std::find(ThreeIntegral_labels.begin(), ThreeIntegral_labels.end(),
                          left_threeint) == ThreeIntegral_labels.end()) {
                ThreeIntegral_labels.push_back(left_threeint);
            }

            right_threeint += label[1];
            right_threeint += label[3];
            if (std::find(ThreeIntegral_labels.begin(), ThreeIntegral_labels.end(),
                          right_threeint) == ThreeIntegral_labels.end()) {
                ThreeIntegral_labels.push_back(right_threeint);
            }

        }
        // Since acac -> (aa)(cc) - (ac)(ac)
        else if (std::islower(label[0]) && std::islower(label[1]) && std::islower(label[2]) &&
                 std::islower(label[3])) {
            // Declare a string for the Kexchange part
            std::string left_threeintK;
            std::string right_threeintK;

            // Next section of code is standard J-like term
            left_threeint += label[0];
            left_threeint += label[2];

            if (std::find(ThreeIntegral_labels.begin(), ThreeIntegral_labels.end(),
                          left_threeint) == ThreeIntegral_labels.end()) {
                ThreeIntegral_labels.push_back(left_threeint);
            }
            right_threeint += label[1];
            right_threeint += label[3];
            if (std::find(ThreeIntegral_labels.begin(), ThreeIntegral_labels.end(),
                          right_threeint) == ThreeIntegral_labels.end()) {
                ThreeIntegral_labels.push_back(right_threeint);
            }

            // Add the exchange part of ThreeInt
            left_threeintK += "d";
            left_threeintK += label[0];
            left_threeintK += label[3];
            ;
            if (std::find(ThreeIntegral_labels.begin(), ThreeIntegral_labels.end(),
                          left_threeintK) == ThreeIntegral_labels.end()) {
                ThreeIntegral_labels.push_back(left_threeintK);
            }
            right_threeintK += "d";
            right_threeintK += label[1];
            right_threeintK += label[2];
            ;
            if (std::find(ThreeIntegral_labels.begin(), ThreeIntegral_labels.end(),
                          right_threeintK) == ThreeIntegral_labels.end()) {
                ThreeIntegral_labels.push_back(right_threeintK);
            }

        } else if (std::isupper(label[0]) && std::isupper(label[1]) && std::isupper(label[2]) &&
                   std::isupper(label[3])) {
            // Declare a string for the Kexchange part
            std::string left_threeintK;
            std::string right_threeintK;

            // Next section of code is standard J-like term
            left_threeint += label[0];
            left_threeint += label[2];

            if (std::find(ThreeIntegral_labels.begin(), ThreeIntegral_labels.end(),
                          left_threeint) == ThreeIntegral_labels.end()) {
                ThreeIntegral_labels.push_back(left_threeint);
            }
            right_threeint += label[1];
            right_threeint += label[3];
            if (std::find(ThreeIntegral_labels.begin(), ThreeIntegral_labels.end(),
                          right_threeint) == ThreeIntegral_labels.end()) {
                ThreeIntegral_labels.push_back(right_threeint);
            }

            // Add the exchange part of ThreeInt
            left_threeintK += "d";
            left_threeintK += label[0];
            left_threeintK += label[3];
            ;
            if (std::find(ThreeIntegral_labels.begin(), ThreeIntegral_labels.end(),
                          left_threeintK) == ThreeIntegral_labels.end()) {
                ThreeIntegral_labels.push_back(left_threeintK);
            }
            right_threeintK += "d";
            right_threeintK += label[1];
            right_threeintK += label[2];
            ;
            if (std::find(ThreeIntegral_labels.begin(), ThreeIntegral_labels.end(),
                          right_threeintK) == ThreeIntegral_labels.end()) {
                ThreeIntegral_labels.push_back(right_threeintK);
            }
        }
    }

    ambit::BlockedTensor ThreeInt =
        BTF_->build(tensor_type_, "ThreeIntMin", ThreeIntegral_labels, true);

    std::vector<std::string> ThreeInt_block = ThreeInt.block_labels();

    std::map<std::string, std::vector<size_t>> mo_to_index = BTF_->get_mo_to_index();

    for (std::string& string_block : ThreeInt_block) {
        std::string pos1(1, string_block[0]);
        std::string pos2(1, string_block[1]);
        std::string pos3(1, string_block[2]);

        std::vector<size_t> first_index = mo_to_index[pos1];
        std::vector<size_t> second_index = mo_to_index[pos2];
        std::vector<size_t> third_index = mo_to_index[pos3];

        ambit::Tensor ThreeIntegral_block =
            ints_->three_integral_block(first_index, second_index, third_index);
        ThreeInt.block(string_block).copy(ThreeIntegral_block);
    }

    return ThreeInt;
}

void THREE_DSRG_MRPT2::compute_t1() {
    // A temporary tensor to use for the building of T1
    // Francesco's library does not handle repeating indices between 3 different
    // terms, so need to form an intermediate
    // via a pointwise multiplcation
    std::string str = "Computing T1";
    outfile->Printf("\n    %-37s ...", str.c_str());
    Timer timer;
    BlockedTensor temp;
    temp = BTF_->build(tensor_type_, "temp", spin_cases({"aa"}), true);
    temp["xu"] = Gamma1_["xu"] * Delta1_["xu"];
    temp["XU"] = Gamma1_["XU"] * Delta1_["XU"];

    // Form the T1 amplitudes

    BlockedTensor N = BTF_->build(tensor_type_, "N", spin_cases({"hp"}));

    N["ia"] = F_["ia"];
    N["ia"] += temp["xu"] * T2_["iuax"];
    N["ia"] += temp["XU"] * T2_["iUaX"];
    T1_["ia"] = N["ia"] * RDelta1_["ia"];

    N["IA"] = F_["IA"];
    N["IA"] += temp["xu"] * T2_["uIxA"];
    N["IA"] += temp["XU"] * T2_["IUAX"];
    T1_["IA"] = N["IA"] * RDelta1_["IA"];

    // internal amplitudes (A->A)
    std::string internal_amp = options_.get_str("INTERNAL_AMP");
    std::string internal_amp_select = options_.get_str("INTERNAL_AMP_SELECT");
    if (internal_amp.find("SINGLES") != std::string::npos) {
        size_t nactv = mo_space_info_->size("ACTIVE");

        // zero half internals to avoid double counting
        for (size_t i = 0; i < nactv; ++i) {
            for (size_t a = 0; a < nactv; ++a) {
                if (i >= a) {
                    size_t idx = i * nactv + a;
                    for (const std::string& block : {"aa", "AA"}) {
                        T1_.block(block).data()[idx] = 0.0;
                    }
                }
            }
        }

        if (internal_amp_select != "ALL") {
            size_t nactv_occ = actv_occ_mos_.size();
            size_t nactv_uocc = actv_uocc_mos_.size();

            // zero O->O internals
            for (size_t I = 0; I < nactv_occ; ++I) {
                for (size_t J = 0; J < nactv_occ; ++J) {
                    size_t idx = actv_occ_mos_[I] * nactv + actv_occ_mos_[J];
                    for (const std::string& block : {"aa", "AA"}) {
                        T1_.block(block).data()[idx] = 0.0;
                    }
                }
            }

            // zero V->V internals
            for (size_t A = 0; A < nactv_uocc; ++A) {
                for (size_t B = 0; B < nactv_uocc; ++B) {
                    size_t idx = actv_uocc_mos_[A] * nactv + actv_uocc_mos_[B];
                    for (const std::string& block : {"aa", "AA"}) {
                        T1_.block(block).data()[idx] = 0.0;
                    }
                }
            }
        }
    } else {
        T1_.block("AA").zero();
        T1_.block("aa").zero();
    }

    outfile->Printf("...Done. Timing %15.6f s", timer.get());
}
void THREE_DSRG_MRPT2::check_t1() {
    // norm and maximum of T1 amplitudes
    T1norm_ = T1_.norm();
    T1max_ = 0.0;
    T1_.iterate([&](const std::vector<size_t>&, const std::vector<SpinType>&, double& value) {
        T1max_ = T1max_ > std::fabs(value) ? T1max_ : std::fabs(value);
    });
}

void THREE_DSRG_MRPT2::renormalize_V() {
    Timer timer;
    std::string str = "Renormalizing V";
    outfile->Printf("\n    %-37s ...", str.c_str());

    V_.iterate([&](const std::vector<size_t>& i, const std::vector<SpinType>& spin, double& value) {
        if ((spin[0] == AlphaSpin) and (spin[1] == AlphaSpin)) {
            value = (value +
                     value *
                         dsrg_source_->compute_renormalized(Fa_[i[0]] + Fa_[i[1]] - Fa_[i[2]] -
                                                            Fa_[i[3]]));
        } else if ((spin[0] == AlphaSpin) and (spin[1] == BetaSpin)) {
            value = (value +
                     value *
                         dsrg_source_->compute_renormalized(Fa_[i[0]] + Fb_[i[1]] - Fa_[i[2]] -
                                                            Fb_[i[3]]));
        } else if ((spin[0] == BetaSpin) and (spin[1] == BetaSpin)) {
            value = (value +
                     value *
                         dsrg_source_->compute_renormalized(Fb_[i[0]] + Fb_[i[1]] - Fb_[i[2]] -
                                                            Fb_[i[3]]));
        }
    });

    outfile->Printf("...Done. Timing %15.6f s", timer.get());
}

void THREE_DSRG_MRPT2::renormalize_F() {
    Timer timer;

    std::string str = "Renormalizing F";
    outfile->Printf("\n    %-37s ...", str.c_str());

    BlockedTensor temp_aa = BTF_->build(tensor_type_, "temp_aa", spin_cases({"aa"}), true);
    temp_aa["xu"] = Gamma1_["xu"] * Delta1_["xu"];
    temp_aa["XU"] = Gamma1_["XU"] * Delta1_["XU"];

    BlockedTensor temp1 = BTF_->build(tensor_type_, "temp1", spin_cases({"hp"}));
    BlockedTensor temp2 = BTF_->build(tensor_type_, "temp2", spin_cases({"hp"}));

    temp1["ia"] += temp_aa["xu"] * T2_["iuax"];
    temp1["ia"] += temp_aa["XU"] * T2_["iUaX"];
    temp2["ia"] += F_["ia"] * RExp1_["ia"];
    temp2["ia"] += temp1["ia"] * RExp1_["ia"];

    temp1["IA"] += temp_aa["xu"] * T2_["uIxA"];
    temp1["IA"] += temp_aa["XU"] * T2_["IUAX"];
    temp2["IA"] += F_["IA"] * RExp1_["IA"];
    temp2["IA"] += temp1["IA"] * RExp1_["IA"];

    F_["ia"] += temp2["ia"];
    F_["IA"] += temp2["IA"];

    // prevent double counting within the active
    F_["em"] += temp2["me"];
    F_["eu"] += temp2["ue"];
    F_["um"] += temp2["mu"];
    F_["EM"] += temp2["ME"];
    F_["EU"] += temp2["UE"];
    F_["UM"] += temp2["MU"];
    outfile->Printf("...Done. Timing %15.6f s", timer.get());
}

double THREE_DSRG_MRPT2::E_FT1() {
    Timer timer;
    std::string str = "Computing <[F, T1]>";
    outfile->Printf("\n    %-37s ...", str.c_str());

    double E = 0.0;
    E += F_["em"] * T1_["me"];
    E += F_["ex"] * T1_["ye"] * Gamma1_["xy"];
    E += F_["xm"] * T1_["my"] * Eta1_["yx"];

    E += F_["EM"] * T1_["ME"];
    E += F_["EX"] * T1_["YE"] * Gamma1_["XY"];
    E += F_["XM"] * T1_["MY"] * Eta1_["YX"];

    if (internal_amp_) {
        E += F_["xv"] * T1_["ux"] * Gamma1_["vu"];
        E -= F_["yu"] * T1_["ux"] * Gamma1_["xy"];

        E += F_["XV"] * T1_["UX"] * Gamma1_["VU"];
        E -= F_["YU"] * T1_["UX"] * Gamma1_["XY"];
    }

    outfile->Printf("...Done. Timing %15.6f s", timer.get());
    dsrg_time_.add("110", timer.get());
    return E;
}

double THREE_DSRG_MRPT2::E_VT1() {
    Timer timer;
    std::string str = "Computing <[V, T1]>";
    outfile->Printf("\n    %-37s ...", str.c_str());

    double E = 0.0;
    BlockedTensor temp = BTF_->build(tensor_type_, "temp", spin_cases({"aaaa"}), true);

    temp["uvxy"] += V_["evxy"] * T1_["ue"];
    temp["uvxy"] -= V_["uvmy"] * T1_["mx"];

    temp["UVXY"] += V_["EVXY"] * T1_["UE"];
    temp["UVXY"] -= V_["UVMY"] * T1_["MX"];

    temp["uVxY"] += V_["eVxY"] * T1_["ue"];
    temp["uVxY"] += V_["uExY"] * T1_["VE"];
    temp["uVxY"] -= V_["uVmY"] * T1_["mx"];
    temp["uVxY"] -= V_["uVxM"] * T1_["MY"];

    E += 0.5 * temp["uvxy"] * Lambda2_["xyuv"];
    E += 0.5 * temp["UVXY"] * Lambda2_["XYUV"];
    E += temp["uVxY"] * Lambda2_["xYuV"];

    if (internal_amp_) {
        temp.zero();

        temp["uvxy"] += V_["wvxy"] * T1_["uw"];
        temp["uvxy"] -= V_["uvwy"] * T1_["wx"];

        temp["UVXY"] += V_["WVXY"] * T1_["UW"];
        temp["UVXY"] -= V_["UVWY"] * T1_["WX"];

        temp["uVxY"] += V_["wVxY"] * T1_["uw"];
        temp["uVxY"] += V_["uWxY"] * T1_["VW"];
        temp["uVxY"] -= V_["uVwY"] * T1_["wx"];
        temp["uVxY"] -= V_["uVxW"] * T1_["WY"];

        E += 0.5 * temp["uvxy"] * Lambda2_["xyuv"];
        E += 0.5 * temp["UVXY"] * Lambda2_["XYUV"];
        E += temp["uVxY"] * Lambda2_["xYuV"];
    }

    outfile->Printf("...Done. Timing %15.6f s", timer.get());
    dsrg_time_.add("210", timer.get());
    return E;
}

double THREE_DSRG_MRPT2::E_FT2() {
    Timer timer;
    std::string str = "Computing <[F, T2]>";
    outfile->Printf("\n    %-37s ...", str.c_str());

    double E = 0.0;
    BlockedTensor temp = BTF_->build(tensor_type_, "temp", spin_cases({"aaaa"}), true);

    temp["uvxy"] += F_["xe"] * T2_["uvey"];
    temp["uvxy"] -= F_["mv"] * T2_["umxy"];

    temp["UVXY"] += F_["XE"] * T2_["UVEY"];
    temp["UVXY"] -= F_["MV"] * T2_["UMXY"];

    temp["uVxY"] += F_["xe"] * T2_["uVeY"];
    temp["uVxY"] += F_["YE"] * T2_["uVxE"];
    temp["uVxY"] -= F_["MV"] * T2_["uMxY"];
    temp["uVxY"] -= F_["mu"] * T2_["mVxY"];

    E += 0.5 * temp["uvxy"] * Lambda2_["xyuv"];
    E += 0.5 * temp["UVXY"] * Lambda2_["XYUV"];
    E += temp["uVxY"] * Lambda2_["xYuV"];

    if (internal_amp_) {
        temp.zero();

        temp["uvxy"] += F_["wx"] * T2_["uvwy"];
        temp["uvxy"] -= F_["vw"] * T2_["uwxy"];

        temp["UVXY"] += F_["WX"] * T2_["UVWY"];
        temp["UVXY"] -= F_["VW"] * T2_["UWXY"];

        temp["uVxY"] += F_["wx"] * T2_["uVwY"];
        temp["uVxY"] += F_["WY"] * T2_["uVxW"];
        temp["uVxY"] -= F_["VW"] * T2_["uWxY"];
        temp["uVxY"] -= F_["uw"] * T2_["wVxY"];

        E += 0.5 * temp["uvxy"] * Lambda2_["xyuv"];
        E += 0.5 * temp["UVXY"] * Lambda2_["XYUV"];
        E += temp["uVxY"] * Lambda2_["xYuV"];
    }

    outfile->Printf("...Done. Timing %15.6f s", timer.get());
    dsrg_time_.add("120", timer.get());
    return E;
}

double THREE_DSRG_MRPT2::E_VT2_2() {
    double E = 0.0;
    int my_proc = 0;
#ifdef HAVE_MPI
    my_proc = MPI::COMM_WORLD.Get_rank();
#endif
    ambit::BlockedTensor temp = BTF_->build(tensor_type_, "temp", {"aa", "AA"});
    Timer timer;
    if (my_proc == 0) {
        std::string str = "Computing <[V, T2]> (C_2)^4 (no ccvv)";
        outfile->Printf("\n    %-36s ...", str.c_str());
        // TODO: Implement these without storing V and/or T2 by using blocking
        if ((integral_type_ != DiskDF)) {
            temp.zero();
            temp["vu"] += 0.5 * V_["efmu"] * T2_["mvef"];
            temp["vu"] += V_["fEuM"] * T2_["vMfE"];
            temp["VU"] += 0.5 * V_["EFMU"] * T2_["MVEF"];
            temp["VU"] += V_["eFmU"] * T2_["mVeF"];
            E += temp["vu"] * Gamma1_["uv"];
            E += temp["VU"] * Gamma1_["UV"];
            // outfile->Printf("\n E = V^{ef}_{mu} * T_{ef}^{mv}: %8.6f", E);
            temp.zero();
            temp["vu"] += 0.5 * V_["vemn"] * T2_["mnue"];
            temp["vu"] += V_["vEmN"] * T2_["mNuE"];
            temp["VU"] += 0.5 * V_["VEMN"] * T2_["MNUE"];
            temp["VU"] += V_["eVnM"] * T2_["nMeU"];
            E += temp["vu"] * Eta1_["uv"];
            E += temp["VU"] * Eta1_["UV"];
            // outfile->Printf("\n E = V^{ve}_{mn} * T_{ue}^{mn}: %8.6f", E);
        } else {
            E += E_VT2_2_one_active();
        }
        /// These terms all have two active indices -> I will assume these can
        /// be store in core.

        temp = BTF_->build(tensor_type_, "temp", spin_cases({"aaaa"}), true);
        temp["yvxu"] += V_["efxu"] * T2_["yvef"];
        temp["yVxU"] += V_["eFxU"] * T2_["yVeF"];
        temp["YVXU"] += V_["EFXU"] * T2_["YVEF"];
        E += 0.25 * temp["yvxu"] * Gamma1_["xy"] * Gamma1_["uv"];
        E += temp["yVxU"] * Gamma1_["UV"] * Gamma1_["xy"];
        E += 0.25 * temp["YVXU"] * Gamma1_["XY"] * Gamma1_["UV"];
        // outfile->Printf("\n V_{xu}^{ef} * T2_{ef}^{yv} * G1 * G1: %8.6f", E);

        temp.zero();
        temp["vyux"] += V_["vymn"] * T2_["mnux"];
        temp["vYuX"] += V_["vYmN"] * T2_["mNuX"];
        temp["VYUX"] += V_["VYMN"] * T2_["MNUX"];
        E += 0.25 * temp["vyux"] * Eta1_["uv"] * Eta1_["xy"];
        E += temp["vYuX"] * Eta1_["uv"] * Eta1_["XY"];
        E += 0.25 * temp["VYUX"] * Eta1_["UV"] * Eta1_["XY"];
        // outfile->Printf("\n V_{vy}^{ux} * T2_{ef}^{yv} * E1 * E1: %8.6f", E);

        temp.zero();
        temp["vyux"] += V_["vemx"] * T2_["myue"];
        temp["vyux"] += V_["vExM"] * T2_["yMuE"];
        temp["VYUX"] += V_["eVmX"] * T2_["mYeU"];
        temp["VYUX"] += V_["VEXM"] * T2_["YMUE"];
        E += temp["vyux"] * Gamma1_["xy"] * Eta1_["uv"];
        E += temp["VYUX"] * Gamma1_["XY"] * Eta1_["UV"];
        temp["yVxU"] = V_["eVxM"] * T2_["yMeU"];
        E += temp["yVxU"] * Gamma1_["xy"] * Eta1_["UV"];
        temp["vYuX"] = V_["vEmX"] * T2_["mYuE"];
        E += temp["vYuX"] * Gamma1_["XY"] * Eta1_["uv"];
        // outfile->Printf("\n V_{ve}^{mx} * T2_{ue}^{my} * G1 * E1: %8.6f", E);

        temp.zero();
        temp["yvxu"] += 0.5 * Gamma1_["wz"] * V_["vexw"] * T2_["yzue"];
        temp["yvxu"] += Gamma1_["WZ"] * V_["vExW"] * T2_["yZuE"];
        temp["yvxu"] += 0.5 * Eta1_["wz"] * T2_["myuw"] * V_["vzmx"];
        temp["yvxu"] += Eta1_["WZ"] * T2_["yMuW"] * V_["vZxM"];
        E += temp["yvxu"] * Gamma1_["xy"] * Eta1_["uv"];
        // outfile->Printf("\n V_{ve}^{xw} * T2_{ue}^{yz} * G1 * E1: %8.6f", E);

        temp["YVXU"] += 0.5 * Gamma1_["WZ"] * V_["VEXW"] * T2_["YZUE"];
        temp["YVXU"] += Gamma1_["wz"] * V_["eVwX"] * T2_["zYeU"];
        temp["YVXU"] += 0.5 * Eta1_["WZ"] * T2_["MYUW"] * V_["VZMX"];
        temp["YVXU"] += Eta1_["wz"] * V_["zVmX"] * T2_["mYwU"];
        E += temp["YVXU"] * Gamma1_["XY"] * Eta1_["UV"];
        // outfile->Printf("\n V_{VE}^{XW} * T2_{UE}^{YZ} * G1 * E1: %8.6f", E);

        // Calculates all but ccvv, cCvV, and CCVV energies
        outfile->Printf("...Done. Timing %15.6f s", timer.get());
    }
    // TODO: Implement these without storing V and/or T2 by using blocking
    // ambit::BlockedTensor temp = BTF_->build(tensor_type_, "temp",{"aa",
    // "AA"});

    // if(integral_type_ != DiskDF)
    //{
    //    temp.zero();
    //    temp["vu"] += 0.5 * V_["efmu"] * T2_["mvef"];
    //    temp["vu"] += V_["fEuM"] * T2_["vMfE"];
    //    temp["VU"] += 0.5 * V_["EFMU"] * T2_["MVEF"];
    //    temp["VU"] += V_["eFmU"] * T2_["mVeF"];
    //    E += temp["vu"] * Gamma1_["uv"];
    //    E += temp["VU"] * Gamma1_["UV"];
    //    //outfile->Printf("\n  E = V^{ef}_{mu} * T_{ef}^{mv}: %8.6f", E);
    //    temp.zero();
    //    temp["vu"] += 0.5 * V_["vemn"] * T2_["mnue"];
    //    temp["vu"] += V_["vEmN"] * T2_["mNuE"];
    //    temp["VU"] += 0.5 * V_["VEMN"] * T2_["MNUE"];
    //    temp["VU"] += V_["eVnM"] * T2_["nMeU"];
    //    E += temp["vu"] * Eta1_["uv"];
    //    E += temp["VU"] * Eta1_["UV"];
    //    //outfile->Printf("\n  E = V^{ve}_{mn} * T_{ue}^{mn}: %8.6f", E);
    //}
    // else
    //{
    //    E += E_VT2_2_one_active();
    //}
    //// These terms all have two active indices -> I will assume these can be
    /// store in core.

    // temp = BTF_->build(tensor_type_,"temp",spin_cases({"aaaa"}), true);
    // temp["yvxu"] += V_["efxu"] * T2_["yvef"];
    // temp["yVxU"] += V_["eFxU"] * T2_["yVeF"];
    // temp["YVXU"] += V_["EFXU"] * T2_["YVEF"];
    // E += 0.25 * temp["yvxu"] * Gamma1_["xy"] * Gamma1_["uv"];
    // E += temp["yVxU"] * Gamma1_["UV"] * Gamma1_["xy"];
    // E += 0.25 * temp["YVXU"] * Gamma1_["XY"] * Gamma1_["UV"];
    ////outfile->Printf("\n  V_{xu}^{ef} * T2_{ef}^{yv} * G1 * G1: %8.6f", E);

    // temp.zero();
    // temp["vyux"] += V_["vymn"] * T2_["mnux"];
    // temp["vYuX"] += V_["vYmN"] * T2_["mNuX"];
    // temp["VYUX"] += V_["VYMN"] * T2_["MNUX"];
    // E += 0.25 * temp["vyux"] * Eta1_["uv"] * Eta1_["xy"];
    // E += temp["vYuX"] * Eta1_["uv"] * Eta1_["XY"];
    // E += 0.25 * temp["VYUX"] * Eta1_["UV"] * Eta1_["XY"];
    ////outfile->Printf("\n  V_{vy}^{ux} * T2_{ef}^{yv} * E1 * E1: %8.6f", E);

    // temp.zero();
    // temp["vyux"] += V_["vemx"] * T2_["myue"];
    // temp["vyux"] += V_["vExM"] * T2_["yMuE"];
    // temp["VYUX"] += V_["eVmX"] * T2_["mYeU"];
    // temp["VYUX"] += V_["VEXM"] * T2_["YMUE"];
    // E += temp["vyux"] * Gamma1_["xy"] * Eta1_["uv"];
    // E += temp["VYUX"] * Gamma1_["XY"] * Eta1_["UV"];
    // temp["yVxU"] = V_["eVxM"] * T2_["yMeU"];
    // E += temp["yVxU"] * Gamma1_["xy"] * Eta1_["UV"];
    // temp["vYuX"] = V_["vEmX"] * T2_["mYuE"];
    // E += temp["vYuX"] * Gamma1_["XY"] * Eta1_["uv"];
    ////outfile->Printf("\n  V_{ve}^{mx} * T2_{ue}^{my} * G1 * E1: %8.6f", E);

    // temp.zero();
    // temp["yvxu"] += 0.5 * Gamma1_["wz"] * V_["vexw"] * T2_["yzue"];
    // temp["yvxu"] += Gamma1_["WZ"] * V_["vExW"] * T2_["yZuE"];
    // temp["yvxu"] += 0.5 * Eta1_["wz"] * T2_["myuw"] * V_["vzmx"];
    // temp["yvxu"] += Eta1_["WZ"] * T2_["yMuW"] * V_["vZxM"];
    // E += temp["yvxu"] * Gamma1_["xy"] * Eta1_["uv"];
    ////outfile->Printf("\n  V_{ve}^{xw} * T2_{ue}^{yz} * G1 * E1: %8.6f", E);

    // temp["YVXU"] += 0.5 * Gamma1_["WZ"] * V_["VEXW"] * T2_["YZUE"];
    // temp["YVXU"] += Gamma1_["wz"] * V_["eVwX"] * T2_["zYeU"];
    // temp["YVXU"] += 0.5 * Eta1_["WZ"] * T2_["MYUW"] * V_["VZMX"];
    // temp["YVXU"] += Eta1_["wz"] * V_["zVmX"] * T2_["mYwU"];
    // E += temp["YVXU"] * Gamma1_["XY"] * Eta1_["UV"];
    ////outfile->Printf("\n  V_{VE}^{XW} * T2_{UE}^{YZ} * G1 * E1: %8.6f", E);

    ////Calculates all but ccvv, cCvV, and CCVV energies
    double Eccvv = 0.0;

    Timer ccvv_timer;
    // TODO:  Make this smarter and automatically switch to right algorithm for
    // size
    // Small size -> use core algorithm
    // Large size -> use fly_ambit

    if (options_.get_str("ccvv_algorithm") == "CORE") {
        if (my_proc == 0)
            Eccvv = E_VT2_2_core();
    } else if (options_.get_str("ccvv_algorithm") == "FLY_LOOP") {
        if (my_proc == 0)
            Eccvv = E_VT2_2_fly_openmp();
    } else if (options_.get_str("ccvv_algorithm") == "FLY_AMBIT") {
        if (my_proc == 0)
            Eccvv = E_VT2_2_ambit();
    } else if (options_.get_str("ccvv_algorithm") == "BATCH_CORE") {
        if (my_proc == 0)
            Eccvv = E_VT2_2_batch_core();
    } else if (options_.get_str("ccvv_algorithm") == "BATCH_CORE_GA") {
#ifdef HAVE_MPI
        Eccvv = E_VT2_2_batch_core_ga();
#endif
    } else if (options_.get_str("CCVV_ALGORITHM") == "BATCH_CORE_REP") {
#ifdef HAVE_MPI
        Eccvv = E_VT2_2_batch_core_rep();
#endif
    } else if (options_.get_str("CCVV_ALGORITHM") == "BATCH_CORE_MPI") {
#ifdef HAVE_MPI
        Eccvv = E_VT2_2_batch_core_mpi();
#endif
    } else if (options_.get_str("CCVV_ALGORITHM") == "BATCH_VIRTUAL") {
        if (my_proc == 0)
            Eccvv = E_VT2_2_batch_virtual();
    } else if (options_.get_str("CCVV_ALGORITHM") == "BATCH_VIRTUAL_GA") {
#ifdef HAVE_MPI
        Eccvv = E_VT2_2_batch_virtual_ga();
#endif
    } else if (options_.get_str("CCVV_ALGORITHM") == "BATCH_VIRTUAL_REP") {
#ifdef HAVE_MPI
        Eccvv = E_VT2_2_batch_virtual_rep();
#endif
    } else if (options_.get_str("CCVV_ALGORITHM") == "BATCH_VIRTUAL_MPI") {
#ifdef HAVE_MPI
        Eccvv = E_VT2_2_batch_virtual_mpi();
#endif
    } else {
        outfile->Printf("\n Specify a correct algorithm string");
        throw PSIEXCEPTION("Specify either CORE FLY_LOOP FLY_AMBIT BATCH_CORE "
                           "BATCH_VIRTUAL BATCH_CORE_MPI BATCH_VIRTUAL_MPI or "
                           "other algorihm");
    }
    if (options_.get_bool("AO_DSRG_MRPT2")) {
        double Eccvv_ao = E_VT2_2_AO_Slow();
        Eccvv = Eccvv_ao;
        outfile->Printf("\n Eccvv_ao: %8.10f", Eccvv_ao);
    }
    outfile->Printf("\n Eccvv: %8.10f", Eccvv);
    std::string strccvv = "Computing <[V, T2]> (C_2)^4 ccvv";
    outfile->Printf("\n    %-37s ...", strccvv.c_str());
    outfile->Printf("...Done. Timing %15.6f s", ccvv_timer.get());
    double all_e = 0.0;
    if (my_proc == 0)
        all_e = E + Eccvv;
#ifdef HAVE_MPI
    MPI_Bcast(&all_e, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
#endif

    if (internal_amp_ && my_proc == 0) {
        temp.zero();
        temp["uvxy"] += 0.25 * V_["uvwz"] * Gamma1_["wx"] * Gamma1_["zy"];
        temp["uVxY"] += V_["uVwZ"] * Gamma1_["wx"] * Gamma1_["ZY"];
        temp["UVXY"] += 0.25 * V_["UVWZ"] * Gamma1_["WX"] * Gamma1_["ZY"];

        temp["uvxy"] -= 0.25 * V_["wzxy"] * Gamma1_["uw"] * Gamma1_["vz"];
        temp["uVxY"] -= V_["wZxY"] * Gamma1_["uw"] * Gamma1_["VZ"];
        temp["UVXY"] -= 0.25 * V_["WZXY"] * Gamma1_["UW"] * Gamma1_["VZ"];

        temp["uvxy"] -= 0.5 * V_["u1wz"] * Gamma1_["v1"] * Gamma1_["wx"] * Gamma1_["zy"];
        temp["uVxY"] -= V_["u!wZ"] * Gamma1_["V!"] * Gamma1_["wx"] * Gamma1_["ZY"];
        temp["uVxY"] -= V_["1VwZ"] * Gamma1_["u1"] * Gamma1_["wx"] * Gamma1_["ZY"];
        temp["UVXY"] -= 0.5 * V_["U!WZ"] * Gamma1_["V!"] * Gamma1_["WX"] * Gamma1_["ZY"];

        temp["uvxy"] += 0.5 * V_["wzx1"] * Gamma1_["uw"] * Gamma1_["vz"] * Gamma1_["1y"];
        temp["uVxY"] += V_["wZx!"] * Gamma1_["uw"] * Gamma1_["VZ"] * Gamma1_["!Y"];
        temp["uVxY"] += V_["wZ1Y"] * Gamma1_["uw"] * Gamma1_["VZ"] * Gamma1_["1x"];
        temp["UVXY"] += 0.5 * V_["WZX!"] * Gamma1_["UW"] * Gamma1_["VZ"] * Gamma1_["!Y"];

        E += temp["uvxy"] * T2_["xyuv"];
        E += temp["uVxY"] * T2_["xYuV"];
        E += temp["UVXY"] * T2_["XYUV"];
    }

    dsrg_time_.add("220", timer.get());
    return (E + Eccvv);
}

double THREE_DSRG_MRPT2::E_VT2_4HH() {
    Timer timer;
    std::string str = "Computing <[V, T2]> 4HH";
    outfile->Printf("\n    %-37s ...", str.c_str());

    double E = 0.0;
    BlockedTensor temp = BTF_->build(tensor_type_, "temp", spin_cases({"aaaa"}), true);

    temp["uvxy"] += 0.125 * V_["uvmn"] * T2_["mnxy"];
    temp["uvxy"] += 0.25 * Gamma1_["wz"] * V_["uvmw"] * T2_["mzxy"];
    temp["uVxY"] += V_["uVmN"] * T2_["mNxY"];
    temp["uVxY"] += Gamma1_["wz"] * T2_["zMxY"] * V_["uVwM"];
    temp["uVxY"] += Gamma1_["WZ"] * V_["uVmW"] * T2_["mZxY"];
    temp["UVXY"] += 0.125 * V_["UVMN"] * T2_["MNXY"];
    temp["UVXY"] += 0.25 * Gamma1_["WZ"] * V_["UVMW"] * T2_["MZXY"];

    E += Lambda2_["xyuv"] * temp["uvxy"];
    E += Lambda2_["xYuV"] * temp["uVxY"];
    E += Lambda2_["XYUV"] * temp["UVXY"];

    if (internal_amp_) {
        temp.zero();
        temp["uvxy"] -= 0.125 * V_["uvwz"] * T2_["wzxy"];
        temp["uVxY"] -= V_["uVwZ"] * T2_["wZxY"];
        temp["UVXY"] -= 0.125 * V_["UVWZ"] * T2_["WZXY"];

        temp["uvxy"] += 0.25 * V_["uv1w"] * T2_["1zxy"] * Gamma1_["wz"];
        temp["uVxY"] += V_["uV1W"] * T2_["1ZxY"] * Gamma1_["WZ"];
        temp["uVxY"] += V_["uVw!"] * T2_["z!xY"] * Gamma1_["wz"];
        temp["UVXY"] += 0.25 * V_["UV!W"] * T2_["!ZXY"] * Gamma1_["WZ"];

        E += Lambda2_["xyuv"] * temp["uvxy"];
        E += Lambda2_["XYUV"] * temp["UVXY"];
        E += Lambda2_["xYuV"] * temp["uVxY"];
    }

    outfile->Printf("...Done. Timing %15.6f s", timer.get());
    dsrg_time_.add("220", timer.get());
    return E;
}

double THREE_DSRG_MRPT2::E_VT2_4PP() {
    Timer timer;
    std::string str = "Computing <V, T2]> 4PP";
    outfile->Printf("\n    %-37s ...", str.c_str());

    double E = 0.0;
    BlockedTensor temp = BTF_->build(tensor_type_, "temp", spin_cases({"aaaa"}), true);

    temp["uvxy"] += 0.125 * V_["efxy"] * T2_["uvef"];
    temp["uvxy"] += 0.25 * Eta1_["wz"] * T2_["uvew"] * V_["ezxy"];
    temp["uVxY"] += V_["eFxY"] * T2_["uVeF"];
    temp["uVxY"] += Eta1_["wz"] * V_["zExY"] * T2_["uVwE"];
    temp["uVxY"] += Eta1_["WZ"] * T2_["uVeW"] * V_["eZxY"];
    temp["UVXY"] += 0.125 * V_["EFXY"] * T2_["UVEF"];
    temp["UVXY"] += 0.25 * Eta1_["WZ"] * T2_["UVEW"] * V_["EZXY"];

    E += Lambda2_["xyuv"] * temp["uvxy"];
    E += Lambda2_["xYuV"] * temp["uVxY"];
    E += Lambda2_["XYUV"] * temp["UVXY"];

    if (internal_amp_) {
        temp.zero();
        temp["uvxy"] += 0.125 * V_["wzxy"] * T2_["uvwz"];
        temp["uVxY"] += V_["wZxY"] * T2_["uVwZ"];
        temp["UVXY"] += 0.125 * V_["WZXY"] * T2_["UVWZ"];

        temp["uvxy"] -= 0.25 * V_["1zxy"] * T2_["uv1w"] * Gamma1_["wz"];
        temp["uVxY"] -= V_["1ZxY"] * T2_["uV1W"] * Gamma1_["WZ"];
        temp["uVxY"] -= V_["z!xY"] * T2_["uVw!"] * Gamma1_["wz"];
        temp["UVXY"] -= 0.25 * V_["!ZXY"] * T2_["UV!W"] * Gamma1_["WZ"];

        E += Lambda2_["xyuv"] * temp["uvxy"];
        E += Lambda2_["xYuV"] * temp["uVxY"];
        E += Lambda2_["XYUV"] * temp["UVXY"];
    }

    outfile->Printf("...Done. Timing %15.6f s", timer.get());
    dsrg_time_.add("220", timer.get());
    return E;
}

double THREE_DSRG_MRPT2::E_VT2_4PH() {
    Timer timer;
    std::string str = "Computing [V, T2] 4PH";
    outfile->Printf("\n    %-37s ...", str.c_str());

    //    double E = 0.0;
    //    BlockedTensor temp1;
    //    BlockedTensor temp2;
    //    temp1 = BTF_->build(tensor_type_,"temp1",{"hapa", "HAPA", "hApA",
    //    "ahap", "AHAP", "aHaP", "aHpA", "hAaP"});
    //    temp2 = BTF_->build(tensor_type_,"temp2", spin_cases({"aaaa"}));

    //    temp1["juby"]  =  T2_["iuay"] * Gamma1_["ji"] * Eta1_["ab"];
    //    temp2["uvxy"] +=  V_["vbjx"] * temp1["juby"];

    //    temp1["uJyB"]  =  T2_["uIyA"] * Gamma1_["JI"] * Eta1_["AB"];
    //    temp2["uvxy"] -=  V_["vBxJ"] * temp1["uJyB"];
    //    E += temp2["uvxy"] * Lambda2_["xyuv"];

    //    temp1["JUBY"]  = T2_["IUAY"] * Gamma1_["IJ"] * Eta1_["AB"];
    //    temp2["UVXY"] += V_["VBJX"] * temp1["JUBY"];

    //    temp1["jUbY"]  = T2_["iUaY"] * Gamma1_["ji"] * Eta1_["ab"];
    //    temp2["UVXY"] -= V_["bVjX"] * temp1["jUbY"];
    //    E += temp2["UVXY"] * Lambda2_["XYUV"];

    //    temp1["jVbY"]  = T2_["iVaY"] * Gamma1_["ji"] * Eta1_["ab"];
    //    temp2["uVxY"] -= V_["ubjx"] * temp1["jVbY"];

    //    temp1["JVBY"]  = T2_["IVAY"] * Gamma1_["JI"] * Eta1_["AB"];
    //    temp2["uVxY"] += V_["uBxJ"] * temp1["JVBY"];

    //    temp1["jubx"]  = T2_["iuax"] * Gamma1_["ji"] * Eta1_["ab"];
    //    temp2["uVxY"] += V_["bVjY"] * temp1["jubx"];

    //    temp1["uJxB"]  = T2_["uIxA"] * Gamma1_["JI"] * Eta1_["AB"];
    //    temp2["uVxY"] -= V_["VBJY"] * temp1["uJxB"];

    //    temp1["uJbY"]  = T2_["uIaY"] * Gamma1_["JI"] * Eta1_["ab"];
    //    temp2["uVxY"] -= V_["bVxJ"] * temp1["uJbY"];

    //    temp1["jVxB"]  = T2_["iVxA"] * Gamma1_["ji"] * Eta1_["AB"];
    //    temp2["uVxY"] -= V_["uBjY"] * temp1["jVxB"];
    //    E += temp2["uVxY"] * Lambda2_["xYuV"];

    double E = 0.0;
    BlockedTensor temp = BTF_->build(tensor_type_, "temp", spin_cases({"aaaa"}), true);

    temp["uvxy"] += V_["eumx"] * T2_["mvey"];
    temp["uvxy"] += V_["uExM"] * T2_["vMyE"];
    temp["uvxy"] += Gamma1_["wz"] * T2_["zvey"] * V_["euwx"];
    temp["uvxy"] += Gamma1_["WZ"] * V_["uExW"] * T2_["vZyE"];
    temp["uvxy"] += Eta1_["zw"] * V_["wumx"] * T2_["mvzy"];
    temp["uvxy"] += Eta1_["ZW"] * T2_["vMyZ"] * V_["uWxM"];
    E += temp["uvxy"] * Lambda2_["xyuv"];

    temp["UVXY"] += V_["eUmX"] * T2_["mVeY"];
    temp["UVXY"] += V_["EUMX"] * T2_["MVEY"];
    temp["UVXY"] += Gamma1_["wz"] * T2_["zVeY"] * V_["eUwX"];
    temp["UVXY"] += Gamma1_["WZ"] * T2_["ZVEY"] * V_["EUWX"];
    temp["UVXY"] += Eta1_["zw"] * V_["wUmX"] * T2_["mVzY"];
    temp["UVXY"] += Eta1_["ZW"] * V_["WUMX"] * T2_["MVZY"];
    E += temp["UVXY"] * Lambda2_["XYUV"];

    temp["uVxY"] += V_["uexm"] * T2_["mVeY"];
    temp["uVxY"] += V_["uExM"] * T2_["MVEY"];
    temp["uVxY"] -= V_["eVxM"] * T2_["uMeY"];
    temp["uVxY"] -= V_["uEmY"] * T2_["mVxE"];
    temp["uVxY"] += V_["eVmY"] * T2_["umxe"];
    temp["uVxY"] += V_["EVMY"] * T2_["uMxE"];

    temp["uVxY"] += Gamma1_["wz"] * T2_["zVeY"] * V_["uexw"];
    temp["uVxY"] += Gamma1_["WZ"] * T2_["ZVEY"] * V_["uExW"];
    temp["uVxY"] -= Gamma1_["WZ"] * V_["eVxW"] * T2_["uZeY"];
    temp["uVxY"] -= Gamma1_["wz"] * T2_["zVxE"] * V_["uEwY"];
    temp["uVxY"] += Gamma1_["wz"] * T2_["zuex"] * V_["eVwY"];
    temp["uVxY"] -= Gamma1_["WZ"] * V_["EVYW"] * T2_["uZxE"];

    temp["uVxY"] += Eta1_["zw"] * V_["wumx"] * T2_["mVzY"];
    temp["uVxY"] += Eta1_["ZW"] * T2_["VMYZ"] * V_["uWxM"];
    temp["uVxY"] -= Eta1_["zw"] * V_["wVxM"] * T2_["uMzY"];
    temp["uVxY"] -= Eta1_["ZW"] * T2_["mVxZ"] * V_["uWmY"];
    temp["uVxY"] += Eta1_["zw"] * T2_["umxz"] * V_["wVmY"];
    temp["uVxY"] += Eta1_["ZW"] * V_["WVMY"] * T2_["uMxZ"];
    E += temp["uVxY"] * Lambda2_["xYuV"];

    if (internal_amp_) {
        temp.zero();
        temp["uvxy"] -= V_["v1xw"] * T2_["zu1y"] * Gamma1_["wz"];
        temp["uvxy"] -= V_["v!xW"] * T2_["uZy!"] * Gamma1_["WZ"];
        temp["uvxy"] += V_["vzx1"] * T2_["1uwy"] * Gamma1_["wz"];
        temp["uvxy"] += V_["vZx!"] * T2_["u!yW"] * Gamma1_["WZ"];
        E += temp["uvxy"] * Lambda2_["xyuv"];

        temp["UVXY"] -= V_["V!XW"] * T2_["ZU!Y"] * Gamma1_["WZ"];
        temp["UVXY"] -= V_["1VwX"] * T2_["zU1Y"] * Gamma1_["wz"];
        temp["UVXY"] += V_["VZX!"] * T2_["!UWY"] * Gamma1_["WZ"];
        temp["UVXY"] += V_["zV1X"] * T2_["1UwY"] * Gamma1_["wz"];
        E += temp["UVXY"] * Lambda2_["XYUV"];

        temp["uVxY"] -= V_["1VxW"] * T2_["uZ1Y"] * Gamma1_["WZ"];
        temp["uVxY"] -= V_["u!wY"] * T2_["zVx!"] * Gamma1_["wz"];
        temp["uVxY"] += V_["u1xw"] * T2_["zV1Y"] * Gamma1_["wz"];
        temp["uVxY"] += V_["u!xW"] * T2_["ZV!Y"] * Gamma1_["WZ"];
        temp["uVxY"] += V_["1VwY"] * T2_["zu1x"] * Gamma1_["wz"];
        temp["uVxY"] += V_["!VWY"] * T2_["uZx!"] * Gamma1_["WZ"];

        temp["uVxY"] += V_["zVx!"] * T2_["u!wY"] * Gamma1_["wz"];
        temp["uVxY"] += V_["uZ1Y"] * T2_["1VxW"] * Gamma1_["WZ"];
        temp["uVxY"] -= V_["uzx1"] * T2_["1VwY"] * Gamma1_["wz"];
        temp["uVxY"] -= V_["uZx!"] * T2_["!VWY"] * Gamma1_["WZ"];
        temp["uVxY"] -= V_["zV1Y"] * T2_["1uwx"] * Gamma1_["wz"];
        temp["uVxY"] -= V_["ZV!Y"] * T2_["u!xW"] * Gamma1_["WZ"];
        E += temp["uVxY"] * Lambda2_["xYuV"];
    }

    outfile->Printf("...Done. Timing %15.6f s", timer.get());
    dsrg_time_.add("220", timer.get());
    return E;
}

double THREE_DSRG_MRPT2::E_VT2_6() {
    Timer timer;
    std::string str = "Computing [V, T2] λ3";
    outfile->Printf("\n    %-37s  ...", str.c_str());
    double E = 0.0;

    if (options_.get_str("THREEPDC") != "ZERO") {
        if (options_.get_str("THREEPDC_ALGORITHM") == "CORE") {

            /* Note: internal amplitudes are included already
                     because we use complex indices "i" and "a" */

            // aaa
            BlockedTensor temp = BTF_->build(tensor_type_, "temp", {"aaaaaa"});
            temp["uvwxyz"] += V_["uviz"] * T2_["iwxy"];
            temp["uvwxyz"] += V_["waxy"] * T2_["uvaz"];

            E += 0.25 * temp.block("aaaaaa")("uvwxyz") * reference_.L3aaa()("xyzuvw");

            // bbb
            temp = BTF_->build(tensor_type_, "temp", {"AAAAAA"});
            temp["UVWXYZ"] += V_["UVIZ"] * T2_["IWXY"];
            temp["UVWXYZ"] += V_["WAXY"] * T2_["UVAZ"];

            E += 0.25 * temp.block("AAAAAA")("UVWXYZ") * reference_.L3bbb()("XYZUVW");

            // aab
            temp = BTF_->build(tensor_type_, "temp", {"aaAaaA"});
            temp["uvWxyZ"] -= V_["uviy"] * T2_["iWxZ"];       //  aaAaaA from hole
            temp["uvWxyZ"] -= V_["uWiZ"] * T2_["ivxy"];       //  aaAaaA from hole
            temp["uvWxyZ"] += 2.0 * V_["uWyI"] * T2_["vIxZ"]; //  aaAaaA from hole

            temp["uvWxyZ"] += V_["aWxZ"] * T2_["uvay"];       //  aaAaaA from particle
            temp["uvWxyZ"] -= V_["vaxy"] * T2_["uWaZ"];       //  aaAaaA from particle
            temp["uvWxyZ"] -= 2.0 * V_["vAxZ"] * T2_["uWyA"]; //  aaAaaA from particle

            E += 0.50 * temp.block("aaAaaA")("uvWxyZ") * reference_.L3aab()("xyZuvW");

            // abb
            temp = BTF_->build(tensor_type_, "temp", {"aAAaAA"});
            temp["uVWxYZ"] -= V_["VWIZ"] * T2_["uIxY"];       //  aAAaAA from hole
            temp["uVWxYZ"] -= V_["uVxI"] * T2_["IWYZ"];       //  aAAaAA from hole
            temp["uVWxYZ"] += 2.0 * V_["uViZ"] * T2_["iWxY"]; //  aAAaAA from hole

            temp["uVWxYZ"] += V_["uAxY"] * T2_["VWAZ"];       //  aAAaAA from particle
            temp["uVWxYZ"] -= V_["WAYZ"] * T2_["uVxA"];       //  aAAaAA from particle
            temp["uVWxYZ"] -= 2.0 * V_["aWxY"] * T2_["uVaZ"]; //  aAAaAA from particle

            E += 0.50 * temp.block("aAAaAA")("uVWxYZ") * reference_.L3abb()("xYZuVW");

        } else if (options_.get_str("THREEPDC_ALGORITHM") == "BATCH") {

            outfile->Printf("\n  Temporarily disabled by York.");

            /** TODO:
             * 3-cumulant files: handled by reference codes (FCI, CAS)
             * 3-cumulant files: cirdm code should NOT return full spin cases
             * temp tensor should also be written to files
            **/

            //            BlockedTensor Lambda3 = BTF_->build(tensor_type_, "Lambda3_",
            //            spin_cases({"aaaaaa"}));

            //            ambit::Tensor Lambda3_aaa = Lambda3.block("aaaaaa");
            //            ambit::Tensor Lambda3_aaA = Lambda3.block("aaAaaA");
            //            ambit::Tensor Lambda3_aAA = Lambda3.block("aAAaAA");
            //            ambit::Tensor Lambda3_AAA = Lambda3.block("AAAAAA");
            //            Lambda3_aaa("pqrstu") = reference_.L3aaa()("pqrstu");
            //            Lambda3_aaA("pqrstu") = reference_.L3aab()("pqrstu");
            //            Lambda3_aAA("pqrstu") = reference_.L3abb()("pqrstu");
            //            Lambda3_AAA("pqrstu") = reference_.L3bbb()("pqrstu");

            //            if (print_ > 3){
            //                Lambda3.print(stdout);
            //            }

            //            Lambda3_aaa("pqrstu") = reference_.L3aaa()("pqrstu");
            //            Lambda3_aaA("pqrstu") = reference_.L3aab()("pqrstu");
            //            Lambda3_aAA("pqrstu") = reference_.L3abb()("pqrstu");
            //            Lambda3_AAA("pqrstu") = reference_.L3bbb()("pqrstu");
            //            size_t size = Lambda3_aaa.data().size();
            //            std::string path = PSIOManager::shared_object()->get_default_path();
            //            FILE* fl3aaa = fopen((path + "forte.l3aaa.bin").c_str(), "w+");
            //            FILE* fl3aAA = fopen((path + "forte.l3aAA.bin").c_str(), "w+");
            //            FILE* fl3aaA = fopen((path + "forte.l3aaA.bin").c_str(), "w+");
            //            FILE* fl3AAA = fopen((path + "forte.l3AAA.bin").c_str(), "w+");

            //            fwrite(&Lambda3_aaa.data()[0], sizeof(double), size, fl3aaa);
            //            fwrite(&Lambda3_aAA.data()[0], sizeof(double), size, fl3aAA);
            //            fwrite(&Lambda3_aaA.data()[0], sizeof(double), size, fl3aaA);
            //            fwrite(&Lambda3_AAA.data()[0], sizeof(double), size, fl3AAA);

            //            temp["uvwxyz"] += V_["uviz"] * T2_["iwxy"];
            //            temp["uvwxyz"] += V_["waxy"] * T2_["uvaz"]; //  aaaaaa from particle
            //            temp["UVWXYZ"] += V_["UVIZ"] * T2_["IWXY"]; //  AAAAAA from hole
            //            temp["UVWXYZ"] += V_["WAXY"] * T2_["UVAZ"]; //  AAAAAA from particle
            //            // E += 0.25 * temp["uvwxyz"] * Lambda3["xyzuvw"];
            //            // E += 0.25 * temp["UVWXYZ"] * Lambda3["XYZUVW"];

            //            temp["uvWxyZ"] -= V_["uviy"] * T2_["iWxZ"]; //  aaAaaA from hole
            //            temp["uvWxyZ"] -= V_["uWiZ"] * T2_["ivxy"]; //  aaAaaA from hole
            //            temp["uvWxyZ"] += V_["uWyI"] * T2_["vIxZ"]; //  aaAaaA from hole
            //            temp["uvWxyZ"] += V_["uWyI"] * T2_["vIxZ"]; //  aaAaaA from hole

            //            temp["uvWxyZ"] += V_["aWxZ"] * T2_["uvay"]; //  aaAaaA from particle
            //            temp["uvWxyZ"] -= V_["vaxy"] * T2_["uWaZ"]; //  aaAaaA from particle
            //            temp["uvWxyZ"] -= V_["vAxZ"] * T2_["uWyA"]; //  aaAaaA from particle
            //            temp["uvWxyZ"] -= V_["vAxZ"] * T2_["uWyA"]; //  aaAaaA from particle

            //            E += 0.50 * temp["uvWxyZ"] * Lambda3["xyZuvW"];

            //            temp["uVWxYZ"] -= V_["VWIZ"] * T2_["uIxY"]; //  aAAaAA from hole
            //            temp["uVWxYZ"] -= V_["uVxI"] * T2_["IWYZ"]; //  aAAaAA from hole
            //            temp["uVWxYZ"] += V_["uViZ"] * T2_["iWxY"]; //  aAAaAA from hole
            //            temp["uVWxYZ"] += V_["uViZ"] * T2_["iWxY"]; //  aAAaAA from hole

            //            temp["uVWxYZ"] += V_["uAxY"] * T2_["VWAZ"]; //  aAAaAA from particle
            //            temp["uVWxYZ"] -= V_["WAYZ"] * T2_["uVxA"]; //  aAAaAA from particle
            //            temp["uVWxYZ"] -= V_["aWxY"] * T2_["uVaZ"]; //  aAAaAA from particle
            //            temp["uVWxYZ"] -= V_["aWxY"] * T2_["uVaZ"]; //  aAAaAA from particle

            //            // E += 0.5 * temp["uVWxYZ"] * Lambda3["xYZuVW"];
            //            double Econtrib = 0.5 * temp["uVWxYZ"] * Lambda3["xYZuVW"];
            //            outfile->Printf("\n  Econtrib: %8.8f", Econtrib);
            //            outfile->Printf("\n  L3aAANorm: %8.8f",
            //                            Lambda3.block("aAAaAA").norm(2.0) *
            //                                Lambda3.block("aAAaAA").norm(2.0));
            //            outfile->Printf("\n  temp: %8.8f",
            //                            temp.block("aAAaAA").norm(2.0) *
            //                                temp.block("aAAaAA").norm(2.0));
            //            ambit::Tensor temp_uVWz = ambit::Tensor::build(
            //                tensor_type_, "VWxz", {active_, active_, active_, active_});
            //            std::vector<double>& temp_uVWz_data = temp.block("aAAaAA").data();
            //            ambit::Tensor L3_ZuVW = ambit::Tensor::build(
            //                tensor_type_, "L3Slice", {active_, active_, active_, active_});
            //            size_t active2 = active_ * active_;
            //            size_t active3 = active2 * active_;
            //            size_t active4 = active3 * active_;
            //            size_t active5 = active4 * active_;
            //            double normTemp = 0.0;
            //            double normCumulant = 0.0;
            //            double Econtrib2 = 0.0;
            //            for (size_t x = 0; x < active_; x++) {
            //                for (size_t y = 0; y < active_; y++) {

            //                    BlockedTensor V_wa =
            //                        BTF_->build(tensor_type_, "V_wa", {"ah", "AH"}, true);
            //                    BlockedTensor T_iw =
            //                        BTF_->build(tensor_type_, "T_iw", {"ha", "HA"}, true);

            //                    BlockedTensor temp_uvwz =
            //                        BTF_->build(tensor_type_, "T_uvwz", {"AAAA", "aaaa"});
            //                    BlockedTensor L3_zuvw =
            //                        BTF_->build(tensor_type_, "L3_zuvw", {"AAAA", "aaaa"});
            //                    temp_uvwz["uvwz"] += V_["uviz"] * T_iw["iw"];
            //                    temp_uvwz["uvwz"] += V_wa["wa"] * T2_["uvaz"];
            //                    temp_uvwz["UVWZ"] += T_iw["IW"] * V_["UVIZ"];
            //                    temp_uvwz["uvwz"] += V_wa["WA"] * T2_["UVAZ"];

            //                    fseek(fl3aaa, (x * active5 + y * active4) * sizeof(double),
            //                          SEEK_SET);
            //                    fread(&(L3_zuvw.block("aaaa").data()[0]), sizeof(double),
            //                          active4, fl3aaa);
            //                    fseek(fl3AAA, (x * active5 + y * active4) * sizeof(double),
            //                          SEEK_SET);
            //                    fread(&(L3_zuvw.block("AAAA").data()[0]), sizeof(double),
            //                          active4, fl3AAA);
            //                    E += 0.25 * temp_uvwz["uvwz"] * L3_zuvw["zuvw"];
            //                    E += 0.25 * temp_uvwz["UVWZ"] * L3_zuvw["ZUVW"];
            //                }
            //            }
            //            outfile->Printf("\n  Econtrib2: %8.8f", Econtrib2);
            //            outfile->Printf("\n  Temp: %8.8f Cumulant: %8.8f", normTemp,
            //                            normCumulant);
        }
    }

    outfile->Printf("...Done. Timing %15.6f s", timer.get());
    return E;
}

double THREE_DSRG_MRPT2::E_VT2_2_fly_openmp() {
    double Eflyalpha = 0.0;
    double Eflybeta = 0.0;
    double Eflymixed = 0.0;
    double Efly = 0.0;
#pragma omp parallel for num_threads(num_threads_) schedule(dynamic)                               \
    reduction(+ : Eflyalpha, Eflybeta, Eflymixed)
    for (size_t mind = 0; mind < core_; mind++) {
        for (size_t nind = 0; nind < core_; nind++) {
            for (size_t eind = 0; eind < virtual_; eind++) {
                for (size_t find = 0; find < virtual_; find++) {
                    // These are used because active is not partitioned as
                    // simple as
                    // core orbs -- active orbs -- virtual
                    // This also takes in account symmetry labeled
                    size_t m = acore_mos_[mind];
                    size_t n = acore_mos_[nind];
                    size_t e = avirt_mos_[eind];
                    size_t f = bvirt_mos_[find];
                    size_t mb = bcore_mos_[mind];
                    size_t nb = bcore_mos_[nind];
                    size_t eb = bvirt_mos_[eind];
                    size_t fb = bvirt_mos_[find];
                    double vmnefalpha = 0.0;

                    double vmnefalphaR = 0.0;
                    double vmnefbeta = 0.0;
                    double vmnefalphaC = 0.0;
                    double vmnefalphaE = 0.0;
                    double vmnefbetaC = 0.0;
                    double vmnefbetaE = 0.0;
                    double vmnefbetaR = 0.0;
                    double vmnefmixed = 0.0;
                    double vmnefmixedC = 0.0;
                    double vmnefmixedR = 0.0;
                    double t2alpha = 0.0;
                    double t2mixed = 0.0;
                    double t2beta = 0.0;
                    vmnefalphaC =
                        C_DDOT(nthree_, &(ints_->three_integral_pointer()[0][m * ncmo_ + e]), 1,
                               &(ints_->three_integral_pointer()[0][n * ncmo_ + f]), 1);
                    vmnefalphaE =
                        C_DDOT(nthree_, &(ints_->three_integral_pointer()[0][m * ncmo_ + f]), 1,
                               &(ints_->three_integral_pointer()[0][n * ncmo_ + e]), 1);
                    vmnefbetaC =
                        C_DDOT(nthree_, &(ints_->three_integral_pointer()[0][mb * ncmo_ + eb]), 1,
                               &(ints_->three_integral_pointer()[0][nb * ncmo_ + fb]), 1);
                    vmnefbetaE =
                        C_DDOT(nthree_, &(ints_->three_integral_pointer()[0][mb * ncmo_ + fb]), 1,
                               &(ints_->three_integral_pointer()[0][nb * ncmo_ + eb]), 1);
                    vmnefmixedC =
                        C_DDOT(nthree_, &(ints_->three_integral_pointer()[0][m * ncmo_ + eb]), 1,
                               &(ints_->three_integral_pointer()[0][n * ncmo_ + fb]), 1);

                    vmnefalpha = vmnefalphaC - vmnefalphaE;
                    vmnefbeta = vmnefbetaC - vmnefbetaE;
                    vmnefmixed = vmnefmixedC;

                    t2alpha = vmnefalpha *
                              dsrg_source_->compute_renormalized_denominator(Fa_[m] + Fa_[n] -
                                                                             Fa_[e] - Fa_[f]);
                    t2beta = vmnefbeta *
                             dsrg_source_->compute_renormalized_denominator(Fb_[m] + Fb_[n] -
                                                                            Fb_[e] - Fb_[f]);
                    t2mixed = vmnefmixed *
                              dsrg_source_->compute_renormalized_denominator(Fa_[m] + Fb_[n] -
                                                                             Fa_[e] - Fb_[f]);

                    vmnefalphaR = vmnefalpha;
                    vmnefbetaR = vmnefbeta;
                    vmnefmixedR = vmnefmixed;
                    vmnefalphaR +=
                        vmnefalpha *
                        dsrg_source_->compute_renormalized(Fa_[m] + Fa_[n] - Fa_[e] - Fa_[f]);
                    vmnefbetaR +=
                        vmnefbeta *
                        dsrg_source_->compute_renormalized(Fb_[m] + Fb_[n] - Fb_[e] - Fb_[f]);
                    vmnefmixedR +=
                        vmnefmixed *
                        dsrg_source_->compute_renormalized(Fa_[m] + Fb_[n] - Fa_[e] - Fb_[f]);

                    Eflyalpha += 0.25 * vmnefalphaR * t2alpha;
                    Eflybeta += 0.25 * vmnefbetaR * t2beta;
                    Eflymixed += vmnefmixedR * t2mixed;
                }
            }
        }
    }
    Efly = Eflyalpha + Eflybeta + Eflymixed;

    return Efly;
}
double THREE_DSRG_MRPT2::E_VT2_2_ambit() {
    // Compute <[V, T2]> (C_2)^4 ccvv term; (me|nf) = B(L|me) * B(L|nf)
    // For a given m and n, form Bm(L|e) and Bn(L|f)
    // Bef(ef) = Bm(L|e) * Bn(L|f)
    size_t dim = nthree_ * virtual_;
    std::vector<size_t> naux(nthree_);
    std::iota(naux.begin(), naux.end(), 0);

    std::vector<size_t> virt_mos = mo_space_info_->get_corr_abs_mo("RESTRICTED_UOCC");

    double Ealpha = 0.0;
    double Ebeta = 0.0;
    double Emixed = 0.0;
    int nthread = 1;
#ifdef _OPENMP
    nthread = omp_get_max_threads();
#endif
    /// This block of code assumes that ThreeIntegral are not stored as a member
    /// variable.  Requires the reading from aptei_block which makes code
    /// general for all, but makes it slow for DiskDF.

    if (integral_type_ == DiskDF) {
        std::vector<ambit::Tensor> BefVec;
        std::vector<ambit::Tensor> BefJKVec;
        std::vector<ambit::Tensor> RDVec;
        std::vector<ambit::Tensor> BmaVec;
        std::vector<ambit::Tensor> BnaVec;
        std::vector<ambit::Tensor> BmbVec;
        std::vector<ambit::Tensor> BnbVec;

        for (int i = 0; i < nthread; i++) {
            BmaVec.push_back(ambit::Tensor::build(tensor_type_, "Bma", {nthree_, virtual_}));
            BnaVec.push_back(ambit::Tensor::build(tensor_type_, "Bna", {nthree_, virtual_}));
            BmbVec.push_back(ambit::Tensor::build(tensor_type_, "Bmb", {nthree_, virtual_}));
            BnbVec.push_back(ambit::Tensor::build(tensor_type_, "Bnb", {nthree_, virtual_}));
            BefVec.push_back(ambit::Tensor::build(tensor_type_, "Bef", {virtual_, virtual_}));
            BefJKVec.push_back(ambit::Tensor::build(tensor_type_, "BefJK", {virtual_, virtual_}));
            RDVec.push_back(ambit::Tensor::build(tensor_type_, "RDVec", {virtual_, virtual_}));
        }
        bool ao_dsrg_check = options_.get_bool("AO_DSRG_MRPT2");

#pragma omp parallel for num_threads(num_threads_) reduction(+ : Ealpha, Ebeta, Emixed)
        for (size_t m = 0; m < core_; ++m) {

            int thread = 0;
#ifdef _OPENMP
            thread = omp_get_thread_num();
#endif

            size_t ma = acore_mos_[m];
            size_t mb = bcore_mos_[m];
#pragma omp critical
            {
                BmaVec[thread] = ints_->three_integral_block_two_index(naux, ma, virt_mos);
                BmbVec[thread] = ints_->three_integral_block_two_index(naux, ma, virt_mos);
            }
            for (size_t n = m; n < core_; ++n) {
                size_t na = acore_mos_[n];
                size_t nb = bcore_mos_[n];
#pragma omp critical
                {
                    BnaVec[thread] = ints_->three_integral_block_two_index(naux, na, virt_mos);
                    BnbVec[thread] = ints_->three_integral_block_two_index(naux, na, virt_mos);
                }
                double factor = (m < n) ? 2.0 : 1.0;

                // alpha-aplha
                BefVec[thread].zero();
                BefJKVec[thread].zero();
                RDVec[thread].zero();

                BefVec[thread]("ef") = BmaVec[thread]("ge") * BnaVec[thread]("gf");
                BefJKVec[thread]("ef") = BefVec[thread]("ef") * BefVec[thread]("ef");
                BefJKVec[thread]("ef") -= BefVec[thread]("ef") * BefVec[thread]("fe");
                RDVec[thread].iterate([&](const std::vector<size_t>& i, double& value) {
                    double D = Fa_[ma] + Fa_[na] - Fa_[avirt_mos_[i[0]]] - Fa_[avirt_mos_[i[1]]];
                    if (ao_dsrg_check)
                        value = 1.0 / D;
                    else {
                        value = dsrg_source_->compute_renormalized_denominator(D) *
                                (1.0 + dsrg_source_->compute_renormalized(D));
                        ;
                    }
                });
                Ealpha += factor * 1.0 * BefJKVec[thread]("ef") * RDVec[thread]("ef");

                BefVec[thread].zero();
                BefJKVec[thread].zero();
                RDVec[thread].zero();

                // beta-beta
                //                BefVec[thread]("EF") = BmbVec[thread]("gE") *
                //                BnbVec[thread]("gF");
                //                BefJKVec[thread]("EF")  = BefVec[thread]("EF")
                //                * BefVec[thread]("EF");
                //                BefJKVec[thread]("EF") -= BefVec[thread]("EF")
                //                * BefVec[thread]("FE");
                //                RDVec[thread].iterate([&](const
                //                std::vector<size_t>& i,double& value){
                //                    double D = Fb_[mb] + Fb_[nb] -
                //                    Fb_[bvirt_mos_[i[0]]] -
                //                    Fb_[bvirt_mos_[i[1]]];
                //                    value =
                //                    dsrg_source_->compute_renormalized_denominator(D)
                //                    * (1.0 +
                //                    dsrg_source_->compute_renormalized(D));});
                //                Ebeta += 0.5 * BefJKVec[thread]("EF") *
                //                RDVec[thread]("EF");

                // alpha-beta
                BefVec[thread].zero();
                BefJKVec[thread].zero();

                BefVec[thread]("eF") = BmaVec[thread]("ge") * BnbVec[thread]("gF");
                BefJKVec[thread]("eF") = BefVec[thread]("eF") * BefVec[thread]("eF");
                RDVec[thread].iterate([&](const std::vector<size_t>& i, double& value) {
                    double D = Fa_[ma] + Fa_[na] - Fa_[avirt_mos_[i[0]]] - Fa_[avirt_mos_[i[1]]];
                    if (ao_dsrg_check)
                        value = 1.0 / D;
                    else {
                        value = dsrg_source_->compute_renormalized_denominator(D) *
                                (1.0 + dsrg_source_->compute_renormalized(D));
                        ;
                    }
                });
                Emixed += factor * BefJKVec[thread]("eF") * RDVec[thread]("eF");
            }
        }
    }
    // This block of code runs with DF and assumes that ThreeIntegral_ is
    // created in startup.  Will fail for systems around 800 or 900 BF
    else {
        ambit::Tensor Ba = ambit::Tensor::build(tensor_type_, "Ba", {core_, nthree_, virtual_});
        ambit::Tensor Bb = ambit::Tensor::build(tensor_type_, "Bb", {core_, nthree_, virtual_});
        Ba("mge") = (ThreeIntegral_.block("dvc"))("gem");
        Bb("MgE") = (ThreeIntegral_.block("dvc"))("gEM");

        std::vector<ambit::Tensor> BmaVec;
        std::vector<ambit::Tensor> BnaVec;
        std::vector<ambit::Tensor> BmbVec;
        std::vector<ambit::Tensor> BnbVec;
        std::vector<ambit::Tensor> BefVec;
        std::vector<ambit::Tensor> BefJKVec;
        std::vector<ambit::Tensor> RDVec;
        for (int i = 0; i < nthread; i++) {
            BmaVec.push_back(ambit::Tensor::build(tensor_type_, "Bma", {nthree_, virtual_}));
            BnaVec.push_back(ambit::Tensor::build(tensor_type_, "Bna", {nthree_, virtual_}));
            BmbVec.push_back(ambit::Tensor::build(tensor_type_, "Bmb", {nthree_, virtual_}));
            BnbVec.push_back(ambit::Tensor::build(tensor_type_, "Bnb", {nthree_, virtual_}));
            BefVec.push_back(ambit::Tensor::build(tensor_type_, "Bef", {virtual_, virtual_}));
            BefJKVec.push_back(ambit::Tensor::build(tensor_type_, "BefJK", {virtual_, virtual_}));
            RDVec.push_back(ambit::Tensor::build(tensor_type_, "RD", {virtual_, virtual_}));
        }
        bool ao_dsrg_check = options_.get_bool("AO_DSRG_MRPT2");
#pragma omp parallel for num_threads(num_threads_) schedule(dynamic)                               \
    reduction(+ : Ealpha, Ebeta, Emixed) shared(Ba, Bb)

        for (size_t m = 0; m < core_; ++m) {
            int thread = 0;
#ifdef _OPENMP
            thread = omp_get_thread_num();
#endif
            size_t ma = acore_mos_[m];
            size_t mb = bcore_mos_[m];

            std::copy(&Ba.data()[m * dim], &Ba.data()[m * dim + dim],
                      BmaVec[thread].data().begin());
            // std::copy(&Bb.data()[m * dim], &Bb.data()[m * dim + dim],
            // BmbVec[thread].data().begin());
            std::copy(&Ba.data()[m * dim], &Ba.data()[m * dim + dim],
                      BmbVec[thread].data().begin());

            for (size_t n = m; n < core_; ++n) {
                size_t na = acore_mos_[n];
                size_t nb = bcore_mos_[n];

                std::copy(&Ba.data()[n * dim], &Ba.data()[n * dim + dim],
                          BnaVec[thread].data().begin());
                // std::copy(&Bb.data()[n * dim], &Bb.data()[n * dim + dim],
                // BnbVec[thread].data().begin());
                std::copy(&Ba.data()[n * dim], &Ba.data()[n * dim + dim],
                          BnbVec[thread].data().begin());

                double factor = (m < n) ? 2.0 : 1.0;

                // alpha-aplha
                BefVec[thread]("ef") = BmaVec[thread]("ge") * BnaVec[thread]("gf");
                BefJKVec[thread]("ef") = BefVec[thread]("ef") * BefVec[thread]("ef");
                BefJKVec[thread]("ef") -= BefVec[thread]("ef") * BefVec[thread]("fe");
                RDVec[thread].iterate([&](const std::vector<size_t>& i, double& value) {
                    double D = Fa_[ma] + Fa_[na] - Fa_[avirt_mos_[i[0]]] - Fa_[avirt_mos_[i[1]]];
                    if (ao_dsrg_check)
                        value = (1.0 / D);
                    else
                        value = dsrg_source_->compute_renormalized_denominator(D) *
                                (1.0 + dsrg_source_->compute_renormalized(D));
                });
                Ealpha += factor * 1.0 * BefJKVec[thread]("ef") * RDVec[thread]("ef");

                // beta-beta
                //                BefVec[thread]("EF") = BmbVec[thread]("gE") *
                //                BnbVec[thread]("gF");
                //                BefJKVec[thread]("EF")  = BefVec[thread]("EF")
                //                * BefVec[thread]("EF");
                //                BefJKVec[thread]("EF") -= BefVec[thread]("EF")
                //                * BefVec[thread]("FE");
                //                RDVec[thread].iterate([&](const
                //                std::vector<size_t>& i,double& value){
                //                    double D = Fb_[mb] + Fb_[nb] -
                //                    Fb_[bvirt_mos_[i[0]]] -
                //                    Fb_[bvirt_mos_[i[1]]];
                //                    value =
                //                    dsrg_source_->compute_renormalized_denominator(D)
                //                    * (1.0 +
                //                    dsrg_source_->compute_renormalized(D));});
                //                Ebeta += 0.5 * BefJKVec[thread]("EF") *
                //                RDVec[thread]("EF");

                // alpha-beta
                BefVec[thread]("eF") = BmaVec[thread]("ge") * BnbVec[thread]("gF");
                BefJKVec[thread]("eF") = BefVec[thread]("eF") * BefVec[thread]("eF");
                RDVec[thread].iterate([&](const std::vector<size_t>& i, double& value) {
                    double D = Fa_[ma] + Fb_[nb] - Fa_[avirt_mos_[i[0]]] - Fb_[bvirt_mos_[i[1]]];
                    if (ao_dsrg_check)
                        value = (1.0 / D);
                    else
                        value = dsrg_source_->compute_renormalized_denominator(D) *
                                (1.0 + dsrg_source_->compute_renormalized(D));
                });
                Emixed += factor * BefJKVec[thread]("eF") * RDVec[thread]("eF");
            }
        }
    }

    return (Ealpha + Ebeta + Emixed);
}
double THREE_DSRG_MRPT2::E_VT2_2_batch_core() {
    bool debug_print = options_.get_bool("DSRG_MRPT2_DEBUG");
    double Ealpha = 0.0;
    double Emixed = 0.0;
    double Ebeta = 0.0;
    // Compute <[V, T2]> (C_2)^4 ccvv term; (me|nf) = B(L|me) * B(L|nf)
    // For a given m and n, form Bm(L|e) and Bn(L|f)
    // Bef(ef) = Bm(L|e) * Bn(L|f)
    outfile->Printf("\n  Computing V_T2_2 in batch algorithm\n");
    outfile->Printf("\n  Batching algorithm is going over m and n");
    size_t dim = nthree_ * virtual_;
    int nthread = 1;
#ifdef _OPENMP
    nthread = omp_get_max_threads();
#endif

    // Step 1:  Figure out the largest chunk of B_{me}^{Q} and B_{nf}^{Q} can be
    // stored in core.
    outfile->Printf("\n\n====Blocking information==========\n");
    size_t int_mem_int = (nthree_ * core_ * virtual_) * sizeof(double);
    size_t memory_input = Process::environment.get_memory() * 0.75;
    size_t num_block = int_mem_int / memory_input < 1 ? 1 : int_mem_int / memory_input;

    if (options_.get_int("CCVV_BATCH_NUMBER") != -1) {
        num_block = options_.get_int("CCVV_BATCH_NUMBER");
    }
    size_t block_size = core_ / num_block;

    if (block_size < 1) {
        outfile->Printf("\n\n  Block size is FUBAR.");
        outfile->Printf("\n  Block size is %d", block_size);
        throw PSIEXCEPTION("Block size is either 0 or negative.  Fix this problem");
    }
    if (num_block > core_) {
        outfile->Printf("\n  Number of blocks can not be larger than core_");
        throw PSIEXCEPTION("Number of blocks is larger than core.  Fix "
                           "num_block or check source code");
    }

    if (num_block >= 1) {
        outfile->Printf("\n  %lu / %lu = %lu", int_mem_int, memory_input,
                        int_mem_int / memory_input);
        outfile->Printf("\n  Block_size = %lu num_block = %lu", block_size, num_block);
    }

    std::vector<size_t> virt_mos = mo_space_info_->get_corr_abs_mo("RESTRICTED_UOCC");
    std::vector<size_t> naux(nthree_);
    std::iota(naux.begin(), naux.end(), 0);

    // Race condition if each thread access ambit tensors
    // Force each thread to have its own copy of matrices (memory NQ * V)
    std::vector<ambit::Tensor> BefVec;
    std::vector<ambit::Tensor> BefJKVec;
    std::vector<ambit::Tensor> RDVec;
    std::vector<ambit::Tensor> BmaVec;
    std::vector<ambit::Tensor> BnaVec;
    std::vector<ambit::Tensor> BmbVec;
    std::vector<ambit::Tensor> BnbVec;

    for (int i = 0; i < nthread; i++) {
        BmaVec.push_back(ambit::Tensor::build(tensor_type_, "Bma", {nthree_, virtual_}));
        BnaVec.push_back(ambit::Tensor::build(tensor_type_, "Bna", {nthree_, virtual_}));
        BmbVec.push_back(ambit::Tensor::build(tensor_type_, "Bmb", {nthree_, virtual_}));
        BnbVec.push_back(ambit::Tensor::build(tensor_type_, "Bnb", {nthree_, virtual_}));
        BefVec.push_back(ambit::Tensor::build(tensor_type_, "Bef", {virtual_, virtual_}));
        BefJKVec.push_back(ambit::Tensor::build(tensor_type_, "BefJK", {virtual_, virtual_}));
        RDVec.push_back(ambit::Tensor::build(tensor_type_, "RDVec", {virtual_, virtual_}));
    }

    // Step 2:  Loop over memory allowed blocks of m and n
    // Get batch sizes and create vectors of mblock length
    for (size_t m_blocks = 0; m_blocks < num_block; m_blocks++) {
        std::vector<size_t> m_batch;
        // If core_ goes into num_block equally, all blocks are equal
        if (core_ % num_block == 0) {
            // Fill the mbatch from block_begin to block_end
            // This is done so I can pass a block to IntegralsAPI to read a
            // chunk
            m_batch.resize(block_size);
            // copy used to get correct indices for B.
            std::copy(acore_mos_.begin() + (m_blocks * block_size),
                      acore_mos_.begin() + ((m_blocks + 1) * block_size), m_batch.begin());
        } else {
            // If last_block is shorter or long, fill the rest
            size_t gimp_block_size =
                m_blocks == (num_block - 1) ? block_size + core_ % num_block : block_size;
            m_batch.resize(gimp_block_size);
            // std::iota(m_batch.begin(), m_batch.end(), m_blocks * (core_ /
            // num_block));
            std::copy(acore_mos_.begin() + (m_blocks)*block_size,
                      acore_mos_.begin() + (m_blocks)*block_size + gimp_block_size,
                      m_batch.begin());
        }

        ambit::Tensor B = ints_->three_integral_block(naux, m_batch, virt_mos);
        ambit::Tensor BmQe =
            ambit::Tensor::build(tensor_type_, "BmQE", {m_batch.size(), nthree_, virtual_});
        BmQe("mQe") = B("Qme");
        B.reset();

        if (debug_print) {
            outfile->Printf("\n  BmQe norm: %8.8f", BmQe.norm(2.0));
            outfile->Printf("\n  m_block: %d", m_blocks);
            int count = 0;
            for (auto mb : m_batch) {
                outfile->Printf("m_batch[%d] =  %d ", count, mb);
                count++;
            }
            outfile->Printf("\n  Core indice list");
            for (auto coremo : acore_mos_) {
                outfile->Printf(" %d ", coremo);
            }
        }

        for (size_t n_blocks = 0; n_blocks <= m_blocks; n_blocks++) {
            std::vector<size_t> n_batch;
            // If core_ goes into num_block equally, all blocks are equal
            if (core_ % num_block == 0) {
                // Fill the mbatch from block_begin to block_end
                // This is done so I can pass a block to IntegralsAPI to read a
                // chunk
                n_batch.resize(block_size);
                std::copy(acore_mos_.begin() + n_blocks * block_size,
                          acore_mos_.begin() + ((n_blocks + 1) * block_size), n_batch.begin());
            } else {
                // If last_block is longer, block_size + remainder
                size_t gimp_block_size =
                    n_blocks == (num_block - 1) ? block_size + core_ % num_block : block_size;
                n_batch.resize(gimp_block_size);
                std::copy(acore_mos_.begin() + (n_blocks)*block_size,
                          acore_mos_.begin() + (n_blocks * block_size) + gimp_block_size,
                          n_batch.begin());
            }
            ambit::Tensor BnQf =
                ambit::Tensor::build(tensor_type_, "BnQf", {n_batch.size(), nthree_, virtual_});
            if (n_blocks == m_blocks) {
                BnQf.copy(BmQe);
            } else {
                ambit::Tensor B = ints_->three_integral_block(naux, n_batch, virt_mos);
                BnQf("mQe") = B("Qme");
                B.reset();
            }
            if (debug_print) {
                outfile->Printf("\n  BnQf norm: %8.8f", BnQf.norm(2.0));
                outfile->Printf("\n  m_block: %d", m_blocks);
                int count = 0;
                for (auto nb : n_batch) {
                    outfile->Printf("n_batch[%d] =  %d ", count, nb);
                    count++;
                }
            }
            size_t m_size = m_batch.size();
            size_t n_size = n_batch.size();
            Timer Core_Loop;
#pragma omp parallel for schedule(runtime) reduction(+ : Ealpha, Emixed)
            for (size_t mn = 0; mn < m_size * n_size; ++mn) {
                int thread = 0;
                size_t m = mn / n_size + m_batch[0];
                size_t n = mn % n_size + n_batch[0];
                if (n > m)
                    continue;
                double factor = (m == n ? 1.0 : 2.0);
#ifdef _OPENMP
                thread = omp_get_thread_num();
#endif
                // Since loop over mn is collapsed, need to use fancy offset
                // tricks
                // m_in_loop = mn / n_size -> corresponds to m increment (m++)
                // n_in_loop = mn % n_size -> corresponds to n increment (n++)
                // m_batch[m_in_loop] corresponds to the absolute index
                size_t m_in_loop = mn / n_size;
                size_t n_in_loop = mn % n_size;
                size_t ma = m_batch[m_in_loop];
                size_t mb = m_batch[m_in_loop];

                size_t na = n_batch[n_in_loop];
                size_t nb = n_batch[n_in_loop];

                std::copy(BmQe.data().begin() + (m_in_loop)*dim,
                          BmQe.data().begin() + (m_in_loop)*dim + dim,
                          BmaVec[thread].data().begin());

                std::copy(BnQf.data().begin() + (mn % n_size) * dim,
                          BnQf.data().begin() + (n_in_loop)*dim + dim,
                          BnaVec[thread].data().begin());
                std::copy(BnQf.data().begin() + (mn % n_size) * dim,
                          BnQf.data().begin() + (n_in_loop)*dim + dim,
                          BnbVec[thread].data().begin());

                // alpha-aplha
                BefVec[thread]("ef") = BmaVec[thread]("ge") * BnaVec[thread]("gf");
                BefJKVec[thread]("ef") = BefVec[thread]("ef") * BefVec[thread]("ef");
                BefJKVec[thread]("ef") -= BefVec[thread]("ef") * BefVec[thread]("fe");
                RDVec[thread].iterate([&](const std::vector<size_t>& i, double& value) {
                    double D = Fa_[ma] + Fa_[na] - Fa_[avirt_mos_[i[0]]] - Fa_[avirt_mos_[i[1]]];
                    value = dsrg_source_->compute_renormalized_denominator(D) *
                            (1.0 + dsrg_source_->compute_renormalized(D));
                });
                Ealpha += factor * 1.0 * BefJKVec[thread]("ef") * RDVec[thread]("ef");

                // beta-beta
                //                BefVec[thread]("EF") = BmbVec[thread]("gE") *
                //                BnbVec[thread]("gF");
                //                BefJKVec[thread]("EF")  = BefVec[thread]("EF")
                //                * BefVec[thread]("EF");
                //                BefJKVec[thread]("EF") -= BefVec[thread]("EF")
                //                * BefVec[thread]("FE");
                //                RDVec[thread].iterate([&](const
                //                std::vector<size_t>& i,double& value){
                //                    double D = Fb_[mb] + Fb_[nb] -
                //                    Fb_[bvirt_mos_[i[0]]] -
                //                    Fb_[bvirt_mos_[i[1]]];
                //                    value =
                //                    dsrg_source_->compute_renormalized_denominator(D)
                //                    * (1.0 +
                //                    dsrg_source_->compute_renormalized(D));});
                //                Ebeta += 0.5 * BefJKVec[thread]("EF") *
                //                RDVec[thread]("EF");

                // alpha-beta
                BefVec[thread]("eF") = BmaVec[thread]("ge") * BnbVec[thread]("gF");
                BefJKVec[thread]("eF") = BefVec[thread]("eF") * BefVec[thread]("eF");
                RDVec[thread].iterate([&](const std::vector<size_t>& i, double& value) {
                    double D = Fa_[ma] + Fb_[nb] - Fa_[avirt_mos_[i[0]]] - Fb_[bvirt_mos_[i[1]]];
                    value = dsrg_source_->compute_renormalized_denominator(D) *
                            (1.0 + dsrg_source_->compute_renormalized(D));
                });
                Emixed += factor * BefJKVec[thread]("eF") * RDVec[thread]("eF");
                if (debug_print) {
                    outfile->Printf("\n  m_size: %d n_size: %d m: %d n:%d", m_size, n_size, m, n);
                    outfile->Printf("\n  m: %d n:%d Ealpha = %8.8f Emixed = "
                                    "%8.8f Sum = %8.8f",
                                    m, n, Ealpha, Emixed, Ealpha + Emixed);
                }
            }
            outfile->Printf("\n Batch_core loop per Mbatch: %d and Nbatch: %d takes %8.8f",
                            m_blocks, n_blocks, Core_Loop.get());
        }
    }
    // return (Ealpha + Ebeta + Emixed);
    return (Ealpha + Ebeta + Emixed);
}
double THREE_DSRG_MRPT2::E_VT2_2_AO_Slow() {
    /// E_{DSRG} -> Conventional basis

    /// V_{me}^{nf} = [(me | nf) - (mf | ne)] * [1 + exp(Delta_{me}^{nf}]
    /// V_{me}^{NF} = [(me | NF) * [1 + exp(Delta_{me}^{NF}]
    /// V_{ME}^{NF} = [(ME | NF) * [1 + exp(Delta_{ME}^{NF}]

    /// T_{nf}^{me} = [(me | nf) - (mf | ne)] * [1 - exp(Delta_{me}^{nf}] /
    /// Delta_{me}^{nf}[(
    /// T_{NF}^{me} = [(me | NF)] * [1 - exp(Delta_{me}^{NF}] /
    /// Delta_{me}^{NF}[(
    /// T_{NF}^{ME} = [(ME | NF) - (MF | NE)] * [1 - exp(Delta_{ME}^{NF}] /
    /// Delta_{ME}^{NF}[(

    /// E_{menf} = V_{me}^{nf} * T_{nf}^{me}
    /// E_{meNF} = V_{me}^{NF} * T_{NF}^{me}
    /// E_{MENF} = V_{ME}^{NF} * T_{NF}^{ME}

    /// E_{AO-DSRG-MRPT2}

    /// The first step for the derivation of the AO method:
    /// R(s)_{me}^{nf} = 0 -> decouple the ccvv term (ie similar to MP2)
    /// V_{me}^{nf} = (me | nf) - (mf | ne)
    /// V_{me}^{NF} = (me | NF)
    /// V_{ME}^{NF} = (ME | NF) - (MF | NE)

    /// T_{nf}^{me} = [(me | nf) - (mf | ne)] * [1 / Delta_{me}^{nf}]
    /// T_{NF}^{me} = [(me | NF)] * [1  / Delta_{me}^{NF}]
    /// T_{NF}^{ME} = [(ME | NF) - (MF | NE)] * [1 / Delta_{ME}^{NF}[(

    /// If this is an AO method, need to actually use AO integrals
    /// PO = C_{mu i} C_{nu i} Pi_{i}^{\alpha}
    /// PV = C_{mu a} C_{nu a} Pi_{a}^{\alpha}
    /// (me | nf) = PO_{\mu m} * PV_{\nu e} (\mu \nu | \rho \sigma) * PO_{n \rho
    /// } * PV_{\sigma f}

    double Ealpha = 0.0;
    double Emixed = 0.0;
    double Ebeta = 0.0;
    SharedMatrix Cwfn = reference_wavefunction_->Ca();
    if (Cwfn->nirrep() != 1)
        throw PSIEXCEPTION("AO-DSRGMPT2 does not work with symmetry");

    /// Create the AtomicOrbitalHelper Class
    SharedVector epsilon_rdocc(new Vector("EPS_RDOCC", core_));
    SharedVector epsilon_virtual(new Vector("EPS_VIRTUAL", virtual_));
    int core_count = 0;
    for (auto m : acore_mos_) {
        epsilon_rdocc->set(core_count, Fa_[m]);
        core_count++;
    }
    int virtual_count = 0;
    for (auto e : avirt_mos_) {
        epsilon_virtual->set(virtual_count, Fa_[e]);
        virtual_count++;
    }
    epsilon_rdocc->print();
    epsilon_virtual->print();

    AtomicOrbitalHelper ao_helper(Cwfn, epsilon_rdocc, epsilon_virtual, 1e-6, active_);
    std::shared_ptr<BasisSet> primary = reference_wavefunction_->basisset();
    std::shared_ptr<BasisSet> auxiliary = reference_wavefunction_->get_basisset("DF_BASIS_MP2");

    ao_helper.Compute_AO_Screen(primary);
    ao_helper.Estimate_TransAO_Screen(primary, auxiliary);
    size_t weights = ao_helper.Weights();
    SharedMatrix AO_Screen = ao_helper.AO_Screen();
    SharedMatrix TransAO_Screen = ao_helper.TransAO_Screen();
    SharedMatrix Occupied_Density = ao_helper.POcc();
    SharedMatrix Virtual_Density = ao_helper.PVir();
    Occupied_Density->print();
    Virtual_Density->print();
    size_t nmo = static_cast<size_t>(nmo_);

    ambit::Tensor POcc = ambit::Tensor::build(tensor_type_, "POcc", {weights, nmo, nmo});
    ambit::Tensor PVir = ambit::Tensor::build(tensor_type_, "Pvir", {weights, nmo, nmo});
    ambit::Tensor AO_Full = ambit::Tensor::build(tensor_type_, "Qso", {nmo, nmo, nmo, nmo});
    ambit::Tensor DF_AO = ambit::Tensor::build(tensor_type_, "Qso", {nthree_, nmo, nmo});
    ambit::Tensor DF_LTAO = ambit::Tensor::build(tensor_type_, "Qso", {weights, nthree_, nmo, nmo});
    ambit::Tensor Full_LTAO =
        ambit::Tensor::build(tensor_type_, "Qso", {weights, nmo, nmo, nmo, nmo});
    ambit::Tensor E_weight_alpha = ambit::Tensor::build(tensor_type_, "Ew", {weights});
    ambit::Tensor E_weight_mixed = ambit::Tensor::build(tensor_type_, "Ew", {weights});
    // ambit::Tensor E_weight_alpha = ambit::Tensor::build(tensor_type_, "Ew",
    // {weights});
    DFTensor df_tensor(primary, auxiliary, Cwfn, core_, virtual_);
    SharedMatrix Qso = df_tensor.Qso();
    DF_AO.iterate([&](const std::vector<size_t>& i, double& value) {
        value = Qso->get(i[0], i[1] * nmo + i[2]);
    });
    POcc.iterate([&](const std::vector<size_t>& i, double& value) {
        value = Occupied_Density->get(i[0], i[1] * nmo + i[2]);
    });
    PVir.iterate([&](const std::vector<size_t>& i, double& value) {
        value = Virtual_Density->get(i[0], i[1] * nmo + i[2]);
    });

    DF_LTAO("w,Q,m,e") = DF_AO("Q, mu, nu") * POcc("w, mu, m") * PVir("w, nu, e");
    Full_LTAO("w, m, e, n, f") = DF_LTAO("w, Q, m, e") * DF_LTAO("w, Q, n, f");
    AO_Full("m, e, n, f") = DF_AO("Q, m, e") * DF_AO("Q, n, f");
    E_weight_mixed("w") = Full_LTAO("w, m, e, n, f") * AO_Full("m, e, n, f");

    // AO_Full.zero();
    // Full_LTAO.zero();
    AO_Full("m, e, n, f") = DF_AO("Q, m, e") * DF_AO("Q, n, f");
    AO_Full("m, e, n, f") -= DF_AO("Q, m, f") * DF_AO("Q, n, e");
    Full_LTAO("w, m, e, n, f") = DF_LTAO("w, Q, m, e") * DF_LTAO("w, Q, n, f");
    Full_LTAO("w, m, e, n, f") -= DF_LTAO("w, Q, m, f") * DF_LTAO("w, Q, n, e");
    E_weight_alpha("w") = Full_LTAO("w,m,e,n,f") * AO_Full("m, e, n, f");
    for (int w = 0; w < weights; w++) {
        Ealpha -= E_weight_alpha.data()[w];
        Ebeta -= E_weight_alpha.data()[w];
        Emixed -= E_weight_mixed.data()[w];
    }

    return (0.25 * Ealpha + 0.25 * Ebeta + Emixed);
}
double THREE_DSRG_MRPT2::E_VT2_2_batch_virtual() {
    bool debug_print = options_.get_bool("DSRG_MRPT2_DEBUG");
    double Ealpha = 0.0;
    double Emixed = 0.0;
    double Ebeta = 0.0;
    // Compute <[V, T2]> (C_2)^4 ccvv term; (me|nf) = B(L|me) * B(L|nf)
    // For a given e and f, form Be(L|m) and Bf(L|n)
    // Bef(mn) = Be(L|m) * Bf(L|n)
    outfile->Printf("\n  Computing V_T2_2 in batch algorithm\n");
    outfile->Printf("\n  Batching algorithm is going over e and f");
    size_t dim = nthree_ * core_;
    int nthread = 1;
#ifdef _OPENMP
    nthread = omp_get_max_threads();
#endif

    // Step 1:  Figure out the largest chunk of B_{me}^{Q} and B_{nf}^{Q} can be
    // stored in core.
    outfile->Printf("\n\n====Blocking information==========\n");
    size_t int_mem_int = (nthree_ * core_ * virtual_) * sizeof(double);
    size_t memory_input = Process::environment.get_memory() * 0.75;
    size_t num_block = int_mem_int / memory_input < 1 ? 1 : int_mem_int / memory_input;

    if (options_.get_int("CCVV_BATCH_NUMBER") != -1) {
        num_block = options_.get_int("CCVV_BATCH_NUMBER");
    }
    size_t block_size = virtual_ / num_block;

    if (block_size < 1) {
        outfile->Printf("\n\n  Block size is FUBAR.");
        outfile->Printf("\n  Block size is %d", block_size);
        throw PSIEXCEPTION("Block size is either 0 or negative.  Fix this problem");
    }
    if (num_block > virtual_) {
        outfile->Printf("\n  Number of blocks can not be larger than core_");
        throw PSIEXCEPTION("Number of blocks is larger than core.  Fix "
                           "num_block or check source code");
    }

    if (num_block >= 1) {
        outfile->Printf("\n  %lu / %lu = %lu", int_mem_int, memory_input,
                        int_mem_int / memory_input);
        outfile->Printf("\n  Block_size = %lu num_block = %lu", block_size, num_block);
    }

    std::vector<size_t> virt_mos = mo_space_info_->get_corr_abs_mo("RESTRICTED_UOCC");
    std::vector<size_t> naux(nthree_);
    std::iota(naux.begin(), naux.end(), 0);

    // Race condition if each thread access ambit tensors
    // Force each thread to have its own copy of matrices (memory NQ * V)
    std::vector<ambit::Tensor> BmnVec;
    std::vector<ambit::Tensor> BmnJKVec;
    std::vector<ambit::Tensor> RDVec;
    std::vector<ambit::Tensor> BmaVec;
    std::vector<ambit::Tensor> BnaVec;
    std::vector<ambit::Tensor> BmbVec;
    std::vector<ambit::Tensor> BnbVec;

    for (int i = 0; i < nthread; i++) {
        BmaVec.push_back(ambit::Tensor::build(tensor_type_, "Bma", {nthree_, core_}));
        BnaVec.push_back(ambit::Tensor::build(tensor_type_, "Bna", {nthree_, core_}));
        BmbVec.push_back(ambit::Tensor::build(tensor_type_, "Bmb", {nthree_, core_}));
        BnbVec.push_back(ambit::Tensor::build(tensor_type_, "Bnb", {nthree_, core_}));
        BmnVec.push_back(ambit::Tensor::build(tensor_type_, "Bmn", {core_, core_}));
        BmnJKVec.push_back(ambit::Tensor::build(tensor_type_, "BmnJK", {core_, core_}));
        RDVec.push_back(ambit::Tensor::build(tensor_type_, "RDVec", {core_, core_}));
    }

    // Step 2:  Loop over memory allowed blocks of m and n
    // Get batch sizes and create vectors of mblock length
    for (size_t e_blocks = 0; e_blocks < num_block; e_blocks++) {
        std::vector<size_t> e_batch;
        // If core_ goes into num_block equally, all blocks are equal
        if (virtual_ % num_block == 0) {
            // Fill the mbatch from block_begin to block_end
            // This is done so I can pass a block to IntegralsAPI to read a
            // chunk
            e_batch.resize(block_size);
            // copy used to get correct indices for B.
            std::copy(virt_mos.begin() + (e_blocks * block_size),
                      virt_mos.begin() + ((e_blocks + 1) * block_size), e_batch.begin());
        } else {
            // If last_block is shorter or long, fill the rest
            size_t gimp_block_size =
                e_blocks == (num_block - 1) ? block_size + virtual_ % num_block : block_size;
            e_batch.resize(gimp_block_size);
            // std::iota(m_batch.begin(), m_batch.end(), m_blocks * (core_ /
            // num_block));
            std::copy(virt_mos.begin() + (e_blocks)*block_size,
                      virt_mos.begin() + (e_blocks)*block_size + gimp_block_size, e_batch.begin());
        }

        ambit::Tensor B = ints_->three_integral_block(naux, e_batch, acore_mos_);
        ambit::Tensor BeQm =
            ambit::Tensor::build(tensor_type_, "BmQE", {e_batch.size(), nthree_, core_});
        BeQm("eQm") = B("Qem");
        B.reset();

        if (debug_print) {
            outfile->Printf("\n  BeQm norm: %8.8f", BeQm.norm(2.0));
            outfile->Printf("\n  e_block: %d", e_blocks);
            int count = 0;
            for (auto e : e_batch) {
                outfile->Printf("e_batch[%d] =  %d ", count, e);
                count++;
            }
            outfile->Printf("\n  Virtual index list");
            for (auto virtualmo : virt_mos) {
                outfile->Printf(" %d ", virtualmo);
            }
        }

        for (size_t f_blocks = 0; f_blocks <= e_blocks; f_blocks++) {
            std::vector<size_t> f_batch;
            // If core_ goes into num_block equally, all blocks are equal
            if (virtual_ % num_block == 0) {
                // Fill the mbatch from block_begin to block_end
                // This is done so I can pass a block to IntegralsAPI to read a
                // chunk
                f_batch.resize(block_size);
                std::copy(virt_mos.begin() + f_blocks * block_size,
                          virt_mos.begin() + ((f_blocks + 1) * block_size), f_batch.begin());
            } else {
                // If last_block is longer, block_size + remainder
                size_t gimp_block_size =
                    f_blocks == (num_block - 1) ? block_size + virtual_ % num_block : block_size;
                f_batch.resize(gimp_block_size);
                std::copy(virt_mos.begin() + (f_blocks)*block_size,
                          virt_mos.begin() + (f_blocks * block_size) + gimp_block_size,
                          f_batch.begin());
            }
            ambit::Tensor BfQn =
                ambit::Tensor::build(tensor_type_, "BnQf", {f_batch.size(), nthree_, core_});
            if (f_blocks == e_blocks) {
                BfQn.copy(BeQm);
            } else {
                ambit::Tensor B = ints_->three_integral_block(naux, f_batch, acore_mos_);
                BfQn("eQm") = B("Qem");
                B.reset();
            }
            if (debug_print) {
                outfile->Printf("\n  BfQn norm: %8.8f", BfQn.norm(2.0));
                outfile->Printf("\n  f_block: %d", f_blocks);
                int count = 0;
                for (auto nf : f_batch) {
                    outfile->Printf("f_batch[%d] =  %d ", count, nf);
                    count++;
                }
            }
            size_t e_size = e_batch.size();
            size_t f_size = f_batch.size();
            Timer Virtual_loop;
#pragma omp parallel for schedule(runtime) reduction(+ : Ealpha, Emixed)
            for (size_t ef = 0; ef < e_size * f_size; ++ef) {
                int thread = 0;
                size_t e = ef / e_size + e_batch[0];
                size_t f = ef % f_size + f_batch[0];
                if (f > e)
                    continue;
                double factor = (e == f ? 1.0 : 2.0);
#ifdef _OPENMP
                thread = omp_get_thread_num();
#endif
                // Since loop over mn is collapsed, need to use fancy offset
                // tricks
                // m_in_loop = mn / n_size -> corresponds to m increment (m++)
                // n_in_loop = mn % n_size -> corresponds to n increment (n++)
                // m_batch[m_in_loop] corresponds to the absolute index
                size_t e_in_loop = ef / f_size;
                size_t f_in_loop = ef % f_size;
                size_t ea = e_batch[e_in_loop];
                size_t eb = e_batch[e_in_loop];

                size_t fa = f_batch[f_in_loop];
                size_t fb = f_batch[f_in_loop];

                std::copy(BeQm.data().begin() + (e_in_loop)*dim,
                          BeQm.data().begin() + (e_in_loop)*dim + dim,
                          BmaVec[thread].data().begin());

                std::copy(BfQn.data().begin() + f_in_loop * dim,
                          BfQn.data().begin() + (f_in_loop)*dim + dim,
                          BnaVec[thread].data().begin());
                std::copy(BfQn.data().begin() + f_in_loop * dim,
                          BfQn.data().begin() + (f_in_loop)*dim + dim,
                          BnbVec[thread].data().begin());

                // alpha-aplha
                BmnVec[thread]("mn") = BmaVec[thread]("gm") * BnaVec[thread]("gn");
                BmnJKVec[thread]("mn") = BmnVec[thread]("mn") * BmnVec[thread]("mn");
                BmnJKVec[thread]("mn") -= BmnVec[thread]("mn") * BmnVec[thread]("nm");
                RDVec[thread].iterate([&](const std::vector<size_t>& i, double& value) {
                    double D = Fa_[acore_mos_[i[0]]] + Fa_[acore_mos_[i[1]]] - Fa_[ea] - Fa_[fa];
                    value = dsrg_source_->compute_renormalized_denominator(D) *
                            (1.0 + dsrg_source_->compute_renormalized(D));
                });
                Ealpha += factor * 1.0 * BmnJKVec[thread]("mn") * RDVec[thread]("mn");

                // alpha-beta
                BmnVec[thread]("mN") = BmaVec[thread]("gm") * BnbVec[thread]("gN");
                BmnJKVec[thread]("mN") = BmnVec[thread]("mN") * BmnVec[thread]("mN");
                RDVec[thread].iterate([&](const std::vector<size_t>& i, double& value) {
                    double D = Fa_[acore_mos_[i[0]]] + Fa_[acore_mos_[i[1]]] - Fa_[ea] - Fa_[fb];
                    value = dsrg_source_->compute_renormalized_denominator(D) *
                            (1.0 + dsrg_source_->compute_renormalized(D));
                });
                Emixed += factor * BmnJKVec[thread]("mN") * RDVec[thread]("mN");
                if (debug_print) {
                    outfile->Printf("\n  e_size: %d f_size: %d e: %d f:%d", e_size, f_size, e, f);
                    outfile->Printf("\n  e: %d f:%d Ealpha = %8.8f Emixed = "
                                    "%8.8f Sum = %8.8f",
                                    e, f, Ealpha, Emixed, Ealpha + Emixed);
                }
            }
            if (debug_print)
                outfile->Printf("\n Virtual loop OpenMP timing for e_batch: %d "
                                "and f_batch: %d takes %8.8f",
                                e_blocks, f_blocks, Virtual_loop.get());
        }
    }
    // return (Ealpha + Ebeta + Emixed);
    return (Ealpha + Ebeta + Emixed);
}
double THREE_DSRG_MRPT2::E_VT2_2_core() {
    double E2_core = 0.0;
    BlockedTensor T2ccvv = BTF_->build(tensor_type_, "T2ccvv", spin_cases({"ccvv"}));
    BlockedTensor v = BTF_->build(tensor_type_, "Vccvv", spin_cases({"ccvv"}));

    BlockedTensor ThreeIntegral = BTF_->build(tensor_type_, "ThreeInt", {"dph", "dPH"});
    ThreeIntegral.iterate([&](const std::vector<size_t>& i, const std::vector<SpinType>&,
                              double& value) { value = ints_->three_integral(i[0], i[1], i[2]); });

    v("mnef") = ThreeIntegral("gem") * ThreeIntegral("gfn");
    v("mnef") -= ThreeIntegral("gfm") * ThreeIntegral("gen");
    v("MNEF") = ThreeIntegral("gEM") * ThreeIntegral("gFN");
    v("MNEF") -= ThreeIntegral("gFM") * ThreeIntegral("gEN");
    v("mNeF") = ThreeIntegral("gem") * ThreeIntegral("gFN");

    if (options_.get_str("CCVV_SOURCE") == "NORMAL") {
        BlockedTensor RD2_ccvv = BTF_->build(tensor_type_, "RDelta2ccvv", spin_cases({"ccvv"}));
        BlockedTensor RExp2ccvv = BTF_->build(tensor_type_, "RExp2ccvv", spin_cases({"ccvv"}));
        RD2_ccvv.iterate(
            [&](const std::vector<size_t>& i, const std::vector<SpinType>& spin, double& value) {
                if ((spin[0] == AlphaSpin) and (spin[1] == AlphaSpin)) {
                    value = dsrg_source_->compute_renormalized_denominator(Fa_[i[0]] + Fa_[i[1]] -
                                                                           Fa_[i[2]] - Fa_[i[3]]);
                } else if ((spin[0] == AlphaSpin) and (spin[1] == BetaSpin)) {
                    value = dsrg_source_->compute_renormalized_denominator(Fa_[i[0]] + Fb_[i[1]] -
                                                                           Fa_[i[2]] - Fb_[i[3]]);
                } else if ((spin[0] == BetaSpin) and (spin[1] == BetaSpin)) {
                    value = dsrg_source_->compute_renormalized_denominator(Fb_[i[0]] + Fb_[i[1]] -
                                                                           Fb_[i[2]] - Fb_[i[3]]);
                }
            });
        RExp2ccvv.iterate(
            [&](const std::vector<size_t>& i, const std::vector<SpinType>& spin, double& value) {
                if ((spin[0] == AlphaSpin) and (spin[1] == AlphaSpin)) {
                    value = dsrg_source_->compute_renormalized(Fa_[i[0]] + Fa_[i[1]] - Fa_[i[2]] -
                                                               Fa_[i[3]]);
                } else if ((spin[0] == AlphaSpin) and (spin[1] == BetaSpin)) {
                    value = dsrg_source_->compute_renormalized(Fa_[i[0]] + Fb_[i[1]] - Fa_[i[2]] -
                                                               Fb_[i[3]]);
                } else if ((spin[0] == BetaSpin) and (spin[1] == BetaSpin)) {
                    value = dsrg_source_->compute_renormalized(Fb_[i[0]] + Fb_[i[1]] - Fb_[i[2]] -
                                                               Fb_[i[3]]);
                }
            });
        BlockedTensor Rv = BTF_->build(tensor_type_, "ReV", spin_cases({"ccvv"}));
        Rv("mnef") = v("mnef");
        Rv("mNeF") = v("mNeF");
        Rv("MNEF") = v("MNEF");
        Rv("mnef") += v("mnef") * RExp2ccvv("mnef");
        Rv("MNEF") += v("MNEF") * RExp2ccvv("MNEF");
        Rv("mNeF") += v("mNeF") * RExp2ccvv("mNeF");

        T2ccvv["MNEF"] = V_["MNEF"] * RD2_ccvv["MNEF"];
        T2ccvv["mnef"] = V_["mnef"] * RD2_ccvv["mnef"];
        T2ccvv["mNeF"] = V_["mNeF"] * RD2_ccvv["mNeF"];
        E2_core += 0.25 * T2ccvv["mnef"] * Rv["mnef"];
        E2_core += 0.25 * T2ccvv["MNEF"] * Rv["MNEF"];
        E2_core += T2ccvv["mNeF"] * Rv["mNeF"];
    } else if (options_.get_str("CCVV_SOURCE") == "ZERO") {
        BlockedTensor Denom = BTF_->build(tensor_type_, "Mp2Denom", spin_cases({"ccvv"}));
        Denom.iterate(
            [&](const std::vector<size_t>& i, const std::vector<SpinType>& spin, double& value) {
                if ((spin[0] == AlphaSpin) and (spin[1] == AlphaSpin)) {
                    value = 1.0 / (Fa_[i[0]] + Fa_[i[1]] - Fa_[i[2]] - Fa_[i[3]]);
                } else if ((spin[0] == AlphaSpin) and (spin[1] == BetaSpin)) {
                    value = 1.0 / (Fa_[i[0]] + Fb_[i[1]] - Fa_[i[2]] - Fb_[i[3]]);
                } else if ((spin[0] == BetaSpin) and (spin[1] == BetaSpin)) {
                    value = 1.0 / (Fb_[i[0]] + Fb_[i[1]] - Fb_[i[2]] - Fb_[i[3]]);
                }
            });
        T2ccvv["MNEF"] = V_["MNEF"] * Denom["MNEF"];
        T2ccvv["mnef"] = V_["mnef"] * Denom["mnef"];
        T2ccvv["mNeF"] = V_["mNeF"] * Denom["mNeF"];

        E2_core += 0.25 * T2ccvv["mnef"] * V_["mnef"];
        E2_core += 0.25 * T2ccvv["MNEF"] * V_["MNEF"];
        E2_core += T2ccvv["mNeF"] * V_["mNeF"];
    }

    return E2_core;
}
double THREE_DSRG_MRPT2::E_VT2_2_one_active() {
    double Eccva = 0;
    double Eacvv = 0;
    int nthread = 1;
    int thread = 0;
#ifdef _OPENMP
    nthread = omp_get_max_threads();
    thread = omp_get_thread_num();
#endif
    /// This block of code assumes that ThreeIntegral are not stored as a member
    /// variable.  Requires the reading from aptei_block which makes code
    std::vector<size_t> naux(nthree_);
    std::iota(naux.begin(), naux.end(), 0);
    ambit::Tensor Gamma1_aa = Gamma1_.block("aa");
    ambit::Tensor Gamma1_AA = Gamma1_.block("AA");

    std::vector<ambit::Tensor> Bm_Qe;
    std::vector<ambit::Tensor> Bm_Qf;

    std::vector<ambit::Tensor> Vefu;
    std::vector<ambit::Tensor> Tefv;
    std::vector<ambit::Tensor> tempTaa;
    std::vector<ambit::Tensor> tempTAA;

    Timer ccvaTimer;
    for (int thread = 0; thread < nthread; thread++) {
        Bm_Qe.push_back(ambit::Tensor::build(tensor_type_, "BemQ", {nthree_, virtual_}));
        Bm_Qf.push_back(ambit::Tensor::build(tensor_type_, "Bmq", {nthree_, virtual_}));

        Vefu.push_back(ambit::Tensor::build(tensor_type_, "muJK", {virtual_, virtual_, active_}));
        Tefv.push_back(ambit::Tensor::build(tensor_type_, "T2", {virtual_, virtual_, active_}));

        tempTaa.push_back(ambit::Tensor::build(tensor_type_, "TEMPaa", {active_, active_}));
        tempTAA.push_back(ambit::Tensor::build(tensor_type_, "TEMPAA", {active_, active_}));
    }
    // ambit::Tensor BemQ = ints_->three_integral_block(naux,  acore_mos_,
    // avirt_mos_);
    // ambit::Tensor BeuQ = ints_->three_integral_block(naux,  aactv_mos_,
    // avirt_mos_);

    // Loop over e and f to compute V

    ambit::Tensor BeuQ = ints_->three_integral_block(naux, avirt_mos_, aactv_mos_);

// std::vector<double>& BemQ_data = BemQ.data();

// I think this loop is typically too small to allow efficient use of
// OpenMP.  Should probably test this assumption.
#pragma omp parallel for num_threads(num_threads_)
    for (size_t m = 0; m < core_; m++) {
        int thread = 0;
#ifdef _OPENMP
        thread = omp_get_thread_num();
#endif
        size_t ma = acore_mos_[m];

// V[efu]_m = B_{em}^Q * B_{fu}^Q - B_{eu}^Q B_{fm}^Q
// V[efu]_m = V[efmu] + V[efmu] * exp[efmu]
// T2["mvef"] = V["mvef"] * D["mvef"]
// temp["uv"] = V * T2
#pragma omp critical
        { Bm_Qe[thread] = ints_->three_integral_block_two_index(naux, ma, avirt_mos_); }

        Vefu[thread]("e, f, u") = Bm_Qe[thread]("Q, e") * BeuQ("Q, f, u");
        Vefu[thread]("e, f, u") -= BeuQ("Q, e, u") * Bm_Qe[thread]("Q, f");

        // E = V["efmu"] (1 + Exp(-s * D^{ef}_{mu}) * V^{mv}_{ef} *
        // Denom^{mv}_{ef}
        Tefv[thread].data() = Vefu[thread].data();

        std::vector<double>& T_mv_data = Tefv[thread].data();
        Vefu[thread].iterate([&](const std::vector<size_t>& i, double& value) {
            double Exp =
                Fa_[avirt_mos_[i[0]]] + Fa_[avirt_mos_[i[1]]] - Fa_[aactv_mos_[i[2]]] - Fa_[ma];
            double D = -1.0 * (Fa_[avirt_mos_[i[0]]] + Fa_[avirt_mos_[i[1]]] -
                               Fa_[aactv_mos_[i[2]]] - Fa_[ma]);
            value = value + value * dsrg_source_->compute_renormalized(Exp);
            T_mv_data[i[0] * virtual_ * active_ + i[1] * active_ + i[2]] *=
                dsrg_source_->compute_renormalized_denominator(D);
        });

        //        T_mv[thread].iterate([&](const std::vector<size_t>& i,double&
        //        value){
        //            double D = Fa_[aactv_mos_[i[1]]] + Fa_[acore_mos_[i[0]]] -
        //            Fa_[ea] - Fa_[fa];
        //            value = value *
        //            dsrg_source_->compute_renormalized_denominator(D);});

        tempTaa[thread]("u,v") += 0.5 * Vefu[thread]("e, f, u") * Tefv[thread]("e, f, v");
        Vefu[thread].zero();
        Tefv[thread].zero();

        Vefu[thread].zero();
        Vefu[thread]("e, f, u") = Bm_Qe[thread]("Q, e") * BeuQ("Q, f, u");

        // E = V["efmu"] (1 + Exp(-s * D^{ef}_{mu}) * V^{mv}_{ef} *
        // Denom^{mv}_{ef}
        Tefv[thread].data() = Vefu[thread].data();
        T_mv_data = Tefv[thread].data();
        T_mv_data = Tefv[thread].data();
        Vefu[thread].iterate([&](const std::vector<size_t>& i, double& value) {
            double Exp =
                Fa_[avirt_mos_[i[0]]] + Fb_[avirt_mos_[i[1]]] - Fa_[aactv_mos_[i[2]]] - Fb_[ma];
            double D = -1.0 * (Fa_[avirt_mos_[i[0]]] + Fb_[avirt_mos_[i[1]]] -
                               Fa_[aactv_mos_[i[2]]] - Fb_[ma]);
            value = value + value * dsrg_source_->compute_renormalized(Exp);
            T_mv_data[i[0] * virtual_ * active_ + i[1] * active_ + i[2]] *=
                dsrg_source_->compute_renormalized_denominator(D);
        });

        //        T_mv[thread].iterate([&](const std::vector<size_t>& i,double&
        //        value){
        //            double D = Fa_[aactv_mos_[i[1]]] + Fa_[acore_mos_[i[0]]] -
        //            Fa_[ea] - Fa_[fa];
        //            value = value *
        //            dsrg_source_->compute_renormalized_denominator(D);});

        tempTAA[thread]("vu") += Vefu[thread]("e, f, u") * Tefv[thread]("e,f, v");
        tempTaa[thread]("vu") += Vefu[thread]("e,f, u") * Tefv[thread]("e,f, v");
        Vefu[thread].zero();
        Tefv[thread].zero();

        Vefu[thread]("e, f, u") = Bm_Qe[thread]("Q, e") * BeuQ("Q, f, u");
        Vefu[thread]("e, f, u") -= BeuQ("Q, e, u") * Bm_Qe[thread]("Q, f");

        // E = V["efmu"] (1 + Exp(-s * D^{ef}_{mu}) * V^{mv}_{ef} *
        // Denom^{mv}_{ef}

        Tefv[thread].data() = Vefu[thread].data();
        T_mv_data = Tefv[thread].data();

        T_mv_data = Tefv[thread].data();
        Vefu[thread].iterate([&](const std::vector<size_t>& i, double& value) {
            double Exp =
                Fa_[bvirt_mos_[i[0]]] + Fb_[bvirt_mos_[i[1]]] - Fb_[bactv_mos_[i[2]]] - Fb_[ma];
            double D = -1.0 * (Fa_[bvirt_mos_[i[0]]] + Fa_[bvirt_mos_[i[1]]] -
                               Fb_[bactv_mos_[i[2]]] - Fb_[ma]);
            value = value + value * dsrg_source_->compute_renormalized(Exp);
            T_mv_data[i[0] * virtual_ * active_ + i[1] * active_ + i[2]] *=
                dsrg_source_->compute_renormalized_denominator(D);
        });

        tempTaa[thread]("u,v") += 0.5 * Vefu[thread]("e, f, u") * Tefv[thread]("e, f, v");
    }

    ambit::Tensor tempTAA_all =
        ambit::Tensor::build(tensor_type_, "tempTAA_all", {active_, active_});
    ambit::Tensor tempTaa_all =
        ambit::Tensor::build(tensor_type_, "tempTaa_all", {active_, active_});
    for (int thread = 0; thread < nthread; thread++) {
        tempTAA_all("v, u") += tempTAA[thread]("v, u");
        tempTaa_all("v, u") += tempTaa[thread]("v, u");
    }

    Eacvv += tempTAA_all("v,u") * Gamma1_AA("v,u");
    Eacvv += tempTaa_all("v,u") * Gamma1_aa("v,u");

    if (print_ > 0) {
        outfile->Printf("\n\n  CAVV computation takes %8.8f", ccvaTimer.get());
    }

    std::vector<ambit::Tensor> Bm_vQ;
    std::vector<ambit::Tensor> Bn_eQ;
    std::vector<ambit::Tensor> Bm_eQ;
    std::vector<ambit::Tensor> Bn_vQ;

    std::vector<ambit::Tensor> V_eu;
    std::vector<ambit::Tensor> T_ev;
    std::vector<ambit::Tensor> tempTaa_e;
    std::vector<ambit::Tensor> tempTAA_e;

    ambit::Tensor BmvQ = ints_->three_integral_block(naux, acore_mos_, aactv_mos_);
    ambit::Tensor BmvQ_swapped =
        ambit::Tensor::build(tensor_type_, "Bm_vQ", {core_, nthree_, active_});
    BmvQ_swapped("m, Q, u") = BmvQ("Q, m, u");
    Timer cavvTimer;
    for (int thread = 0; thread < nthread; thread++) {
        Bm_vQ.push_back(ambit::Tensor::build(tensor_type_, "BemQ", {nthree_, active_}));
        Bn_eQ.push_back(ambit::Tensor::build(tensor_type_, "Bf_uQ", {nthree_, virtual_}));
        Bm_eQ.push_back(ambit::Tensor::build(tensor_type_, "Bmq", {nthree_, virtual_}));
        Bn_vQ.push_back(ambit::Tensor::build(tensor_type_, "Bmq", {nthree_, active_}));

        V_eu.push_back(ambit::Tensor::build(tensor_type_, "muJK", {virtual_, active_}));
        T_ev.push_back(ambit::Tensor::build(tensor_type_, "T2", {virtual_, active_}));

        tempTaa_e.push_back(ambit::Tensor::build(tensor_type_, "TEMPaa", {active_, active_}));
        tempTAA_e.push_back(ambit::Tensor::build(tensor_type_, "TEMPAA", {active_, active_}));
    }
    ambit::Tensor Eta1_aa = Eta1_.block("aa");
    ambit::Tensor Eta1_AA = Eta1_.block("AA");

#pragma omp parallel for num_threads(num_threads_)
    for (size_t m = 0; m < core_; ++m) {
        size_t ma = acore_mos_[m];
        size_t mb = bcore_mos_[m];
        int thread = 0;
#ifdef _OPENMP
        thread = omp_get_thread_num();
#endif

#pragma omp critical
        { Bm_eQ[thread] = ints_->three_integral_block_two_index(naux, ma, avirt_mos_); }
        std::copy(&BmvQ_swapped.data()[m * nthree_ * active_],
                  &BmvQ_swapped.data()[m * nthree_ * active_ + nthree_ * active_],
                  Bm_vQ[thread].data().begin());

        for (size_t n = 0; n < core_; ++n) {
            // alpha-aplha
            size_t na = acore_mos_[n];
            size_t nb = bcore_mos_[n];

            std::copy(&BmvQ_swapped.data()[n * nthree_ * active_],
                      &BmvQ_swapped.data()[n * nthree_ * active_ + nthree_ * active_],
                      Bn_vQ[thread].data().begin());
//    Bn_vQ[thread].iterate([&](const std::vector<size_t>& i,double& value){
//        value = BmvQ_data[i[0] * core_ * active_ + n * active_ + i[1] ];
//    });
#pragma omp critical
            { Bn_eQ[thread] = ints_->three_integral_block_two_index(naux, na, avirt_mos_); }

            // B_{mv}^{Q} * B_{ne}^{Q} - B_{me}^Q * B_{nv}
            V_eu[thread]("e, u") = Bm_vQ[thread]("Q, u") * Bn_eQ[thread]("Q, e");
            V_eu[thread]("e, u") -= Bm_eQ[thread]("Q, e") * Bn_vQ[thread]("Q, u");
            // E = V["efmu"] (1 + Exp(-s * D^{ef}_{mu}) * V^{mv}_{ef} *
            // Denom^{mv}_{ef}
            T_ev[thread].data() = V_eu[thread].data();

            V_eu[thread].iterate([&](const std::vector<size_t>& i, double& value) {
                double Exp = Fa_[aactv_mos_[i[1]]] + Fa_[avirt_mos_[i[0]]] - Fa_[ma] - Fa_[na];
                value = value + value * dsrg_source_->compute_renormalized(Exp);
                double D = Fa_[ma] + Fa_[na] - Fa_[aactv_mos_[i[1]]] - Fa_[avirt_mos_[i[0]]];
                T_ev[thread].data()[i[0] * active_ + i[1]] *=
                    dsrg_source_->compute_renormalized_denominator(D);
                ;
            });

            tempTaa_e[thread]("u,v") += 0.5 * V_eu[thread]("e,u") * T_ev[thread]("e,v");
            V_eu[thread].zero();
            T_ev[thread].zero();

            // alpha-beta
            // temp["vu"] += V_["vEmN"] * T2_["mNuE"];
            //
            V_eu[thread]("E,u") = Bm_vQ[thread]("Q, u") * Bn_eQ[thread]("Q, E");
            T_ev[thread].data() = V_eu[thread].data();
            V_eu[thread].iterate([&](const std::vector<size_t>& i, double& value) {
                double Exp = Fa_[aactv_mos_[i[1]]] + Fb_[bvirt_mos_[i[0]]] - Fa_[ma] - Fb_[nb];
                value = value + value * dsrg_source_->compute_renormalized(Exp);
                double D = Fa_[ma] + Fb_[nb] - Fa_[aactv_mos_[i[1]]] - Fb_[bvirt_mos_[i[0]]];
                T_ev[thread].data()[i[0] * active_ + i[1]] *=
                    dsrg_source_->compute_renormalized_denominator(D);
                ;
            });

            tempTAA_e[thread]("vu") += V_eu[thread]("M,v") * T_ev[thread]("M,u");
            tempTaa_e[thread]("vu") += V_eu[thread]("M,v") * T_ev[thread]("M, u");

            // beta-beta
            V_eu[thread].zero();
            T_ev[thread].zero();
            V_eu[thread]("E,U") = Bm_vQ[thread]("Q, U") * Bn_eQ[thread]("Q,E");
            V_eu[thread]("E,U") -= Bm_eQ[thread]("Q, E") * Bn_vQ[thread]("Q, U");
            T_ev[thread].data() = V_eu[thread].data();

            V_eu[thread].iterate([&](const std::vector<size_t>& i, double& value) {
                double Exp = Fb_[mb] + Fb_[nb] - Fb_[bactv_mos_[i[1]]] - Fb_[bvirt_mos_[i[0]]];
                value = value + value * dsrg_source_->compute_renormalized(Exp);
                double D = Fb_[mb] + Fb_[nb] - Fb_[bactv_mos_[i[1]]] - Fb_[avirt_mos_[i[0]]];
                T_ev[thread].data()[i[0] * active_ + i[1]] *=
                    dsrg_source_->compute_renormalized_denominator(D);
                ;
            });

            tempTAA_e[thread]("v,u") += 0.5 * V_eu[thread]("M,v") * T_ev[thread]("M,u");
            V_eu[thread].zero();
            T_ev[thread].zero();
        }
    }

    tempTAA_all = ambit::Tensor::build(tensor_type_, "tempTAA_all", {active_, active_});
    tempTaa_all = ambit::Tensor::build(tensor_type_, "tempTaa_all", {active_, active_});
    for (int thread = 0; thread < nthread; thread++) {
        tempTAA_all("u, v") += tempTAA_e[thread]("u,v");
        tempTaa_all("u, v") += tempTaa_e[thread]("u,v");
    }
    Eccva += tempTaa_all("vu") * Eta1_aa("uv");
    Eccva += tempTAA_all("VU") * Eta1_AA("UV");
    if (print_ > 0) {
        outfile->Printf("\n\n  CCVA takes %8.8f", cavvTimer.get());
    }

    return (Eacvv + Eccva);
}

void THREE_DSRG_MRPT2::relax_reference_once() {
    // Time to relax this reference!
    BlockedTensor T2all = BTF_->build(tensor_type_, "T2all", spin_cases({"hhpp"}));
    BlockedTensor Vint = BTF_->build(tensor_type_, "AllV", spin_cases({"pphh"}));
    BlockedTensor ThreeInt = compute_B_minimal(Vint.block_labels());
    Vint["pqrs"] = ThreeInt["gpr"] * ThreeInt["gqs"];
    Vint["pqrs"] -= ThreeInt["gps"] * ThreeInt["gqr"];
    Vint["PQRS"] = ThreeInt["gPR"] * ThreeInt["gQS"];
    Vint["PQRS"] -= ThreeInt["gPS"] * ThreeInt["gQR"];
    Vint["qPsR"] = ThreeInt["gPR"] * ThreeInt["gqs"];

    Hbar2_["uvxy"] = Vint["uvxy"];
    Hbar2_["uVxY"] = Vint["uVxY"];
    Hbar2_["UVXY"] = Vint["UVXY"];

    T2all["ijab"] = Vint["abij"];
    T2all["IJAB"] = Vint["ABIJ"];
    T2all["iJaB"] = Vint["aBiJ"];

    T2all.iterate(
        [&](const std::vector<size_t>& i, const std::vector<SpinType>& spin, double& value) {
            if (spin[0] == AlphaSpin && spin[1] == AlphaSpin) {
                value *= dsrg_source_->compute_renormalized_denominator(Fa_[i[0]] + Fa_[i[1]] -
                                                                        Fa_[i[2]] - Fa_[i[3]]);
            } else if (spin[0] == BetaSpin && spin[1] == BetaSpin) {
                value *= dsrg_source_->compute_renormalized_denominator(Fb_[i[0]] + Fb_[i[1]] -
                                                                        Fb_[i[2]] - Fb_[i[3]]);
            } else {
                value *= dsrg_source_->compute_renormalized_denominator(Fa_[i[0]] + Fb_[i[1]] -
                                                                        Fa_[i[2]] - Fb_[i[3]]);
            }
        });

    if (!options_.get_bool("INTERNAL_AMP")) {
        T2all.block("aaaa").zero();
        T2all.block("AAAA").zero();
        T2all.block("aAaA").zero();
    }

    Vint.iterate(
        [&](const std::vector<size_t>& i, const std::vector<SpinType>& spin, double& value) {
            if (std::fabs(value) > 1.0e-12) {
                if ((spin[0] == AlphaSpin) and (spin[1] == AlphaSpin)) {
                    value *= 1.0 +
                             dsrg_source_->compute_renormalized(Fa_[i[0]] + Fa_[i[1]] - Fa_[i[2]] -
                                                                Fa_[i[3]]);
                } else if ((spin[0] == AlphaSpin) and (spin[1] == BetaSpin)) {
                    value *= 1.0 +
                             dsrg_source_->compute_renormalized(Fa_[i[0]] + Fb_[i[1]] - Fa_[i[2]] -
                                                                Fb_[i[3]]);
                } else if ((spin[0] == BetaSpin) and (spin[1] == BetaSpin)) {
                    value *= 1.0 +
                             dsrg_source_->compute_renormalized(Fb_[i[0]] + Fb_[i[1]] - Fb_[i[2]] -
                                                                Fb_[i[3]]);
                }
            } else {
                value = 0.0;
            }
        });

    BlockedTensor C1 = BTF_->build(tensor_type_, "C1", spin_cases({"aa"}));
    BlockedTensor C2 = BTF_->build(tensor_type_, "C2", spin_cases({"aaaa"}));
    H1_T1_C1(F_, T1_, 0.5, C1);
    H1_T2_C1(F_, T2all, 0.5, C1);
    H2_T1_C1(Vint, T1_, 0.5, C1);
    H2_T2_C1(Vint, T2all, 0.5, C1);
    H1_T2_C2(F_, T2all, 0.5, C2);
    H2_T1_C2(Vint, T1_, 0.5, C2);
    H2_T2_C2(Vint, T2all, 0.5, C2);

    Hbar1_["uv"] += C1["uv"];
    Hbar1_["uv"] += C1["vu"];
    Hbar1_["UV"] += C1["UV"];
    Hbar1_["UV"] += C1["VU"];
    Hbar2_["uvxy"] += C2["uvxy"];
    Hbar2_["uvxy"] += C2["xyuv"];
    Hbar2_["uVxY"] += C2["uVxY"];
    Hbar2_["uVxY"] += C2["xYuV"];
    Hbar2_["UVXY"] += C2["UVXY"];
    Hbar2_["UVXY"] += C2["XYUV"];

    de_normal_order();

    std::vector<double> E_relaxes = relaxed_energy();

    if (options_["AVG_STATE"].size() == 0) {
        double Erelax = E_relaxes[0];

        // printing
        print_h2("CD/DF DSRG-MRPT2 Energy Summary");
        outfile->Printf("\n    %-37s = %22.15f", "CD/DF DSRG-MRPT2 Total Energy (fixed)  ",
                        Hbar0_ + Eref_);
        outfile->Printf("\n    %-37s = %22.15f", "CD/DF DSRG-MRPT2 Total Energy (relaxed)", Erelax);

        Process::environment.globals["CURRENT ENERGY"] = Erelax;
    } else {
        // get character table
        CharacterTable ct = Process::environment.molecule()->point_group()->char_table();
        std::vector<std::string> irrep_symbol;
        for (int h = 0; h < this->nirrep(); ++h) {
            irrep_symbol.push_back(std::string(ct.gamma(h).symbol()));
        }

        // energy summuary
        print_h2("DF/CD SA-DSRG-PT2 Energy Summary");

        outfile->Printf("\n    Multi.  Irrep.  No.    DSRG-MRPT2 Energy");
        std::string dash(41, '-');
        outfile->Printf("\n    %s", dash.c_str());

        int offset = 0;
        int nentry = options_["AVG_STATE"].size();
        for (int n = 0; n < nentry; ++n) {
            int irrep = options_["AVG_STATE"][n][0].to_integer();
            int multi = options_["AVG_STATE"][n][1].to_integer();
            int nstates = options_["AVG_STATE"][n][2].to_integer();

            for (int i = 0; i < nstates; ++i) {
                outfile->Printf("\n     %3d     %3s    %2d   %20.12f", multi,
                                irrep_symbol[irrep].c_str(), i, E_relaxes[i + offset]);
            }
            outfile->Printf("\n    %s", dash.c_str());

            offset += nstates;
        }

        Process::environment.globals["CURRENT ENERGY"] = E_relaxes[0];
    }
}

std::vector<double> THREE_DSRG_MRPT2::relaxed_energy() {

    // reference relaxation
    std::vector<double> Erelax;

    // setup FCIIntegrals manually
    std::shared_ptr<FCIIntegrals> fci_ints =
        std::make_shared<FCIIntegrals>(ints_, aactv_mos_, acore_mos_);
    fci_ints->set_active_integrals(Hbar2_.block("aaaa"), Hbar2_.block("aAaA"),
                                   Hbar2_.block("AAAA"));
    fci_ints->set_restricted_one_body_operator(aone_eff_, bone_eff_);
    fci_ints->set_scalar_energy(ints_->scalar());

    // check CAS_TYPE to decide diagonalization code
    if (options_.get_str("CAS_TYPE") == "CAS") {

        FCI_MO fci_mo(reference_wavefunction_, options_, ints_, mo_space_info_);
        fci_mo.set_fci_int(fci_ints);

        // test state specific or state average
        if (options_["AVG_STATE"].size() == 0) {
            Erelax.push_back(fci_mo.compute_ss_energy());
        } else {
            fci_mo.compute_sa_energy();

            std::vector<std::vector<std::pair<SharedVector, double>>> eigens = fci_mo.eigens();
            size_t nentry = eigens.size();
            for (size_t n = 0; n < nentry; ++n) {
                std::vector<std::pair<SharedVector, double>> eigen = eigens[n];
                size_t ni = eigen.size();
                for (size_t i = 0; i < ni; ++i) {
                    Erelax.push_back(eigen[i].second);
                }
            }
        }

    } else {

        // common (SS and SA) setup of FCISolver
        int ntrial_per_root = options_.get_int("NTRIAL_PER_ROOT");
        Dimension active_dim = mo_space_info_->get_dimension("ACTIVE");
        std::shared_ptr<Molecule> molecule = Process::environment.molecule();
        double Enuc = molecule->nuclear_repulsion_energy();
        int charge = molecule->molecular_charge();
        if (options_["CHARGE"].has_changed()) {
            charge = options_.get_int("CHARGE");
        }
        auto nelec = 0;
        int natom = molecule->natom();
        for (int i = 0; i < natom; ++i) {
            nelec += molecule->fZ(i);
        }
        nelec -= charge;

        // if state specific, read from fci_root and fci_nroot
        if (options_["AVG_STATE"].size() == 0) {
            // setup for FCISolver
            int multi = Process::environment.molecule()->multiplicity();
            if (options_["MULTIPLICITY"].has_changed()) {
                multi = options_.get_int("MULTIPLICITY");
            }
            int twice_ms = (multi + 1) % 2;
            if (options_["MS"].has_changed()) {
                twice_ms = std::round(2.0 * options_.get_double("MS"));
            }
            auto nelec_actv =
                nelec - 2 * mo_space_info_->size("FROZEN_DOCC") - 2 * acore_mos_.size();
            auto na = (nelec_actv + twice_ms) / 2;
            auto nb = nelec_actv - na;

            // diagonalize the Hamiltonian
            FCISolver fcisolver(active_dim, acore_mos_, aactv_mos_, na, nb, multi,
                                options_.get_int("ROOT_SYM"), ints_, mo_space_info_,
                                ntrial_per_root, print_, options_);
            fcisolver.set_max_rdm_level(1);
            fcisolver.set_nroot(options_.get_int("FCI_NROOT"));
            fcisolver.set_root(options_.get_int("FCI_ROOT"));
            fcisolver.set_test_rdms(options_.get_bool("FCI_TEST_RDMS"));
            fcisolver.set_fci_iterations(options_.get_int("FCI_MAXITER"));
            fcisolver.set_collapse_per_root(options_.get_int("DL_COLLAPSE_PER_ROOT"));
            fcisolver.set_subspace_per_root(options_.get_int("DL_SUBSPACE_PER_ROOT"));

            // set integrals manually
            fcisolver.use_user_integrals_and_restricted_docc(true);
            fcisolver.set_integral_pointer(fci_ints);

            Erelax.push_back(fcisolver.compute_energy());
        } else {
            int nentry = options_["AVG_STATE"].size();

            for (int n = 0; n < nentry; ++n) {
                int irrep = options_["AVG_STATE"][n][0].to_integer();
                int multi = options_["AVG_STATE"][n][1].to_integer();
                int nstates = options_["AVG_STATE"][n][2].to_integer();

                // prepare FCISolver
                int ms = (multi + 1) % 2;
                auto nelec_actv =
                    nelec - 2 * mo_space_info_->size("FROZEN_DOCC") - 2 * acore_mos_.size();
                auto na = (nelec_actv + ms) / 2;
                auto nb = nelec_actv - na;

                FCISolver fcisolver(active_dim, acore_mos_, aactv_mos_, na, nb, multi, irrep, ints_,
                                    mo_space_info_, ntrial_per_root, print_, options_);
                fcisolver.set_max_rdm_level(1);
                fcisolver.set_nroot(nstates);
                fcisolver.set_root(nstates - 1);
                fcisolver.set_fci_iterations(options_.get_int("FCI_MAXITER"));
                fcisolver.set_collapse_per_root(options_.get_int("DL_COLLAPSE_PER_ROOT"));
                fcisolver.set_subspace_per_root(options_.get_int("DL_SUBSPACE_PER_ROOT"));

                // compute energy and fill in results
                fcisolver.compute_energy();
                SharedVector Ems = fcisolver.eigen_vals();
                for (int i = 0; i < nstates; ++i) {
                    Erelax.push_back(Ems->get(i) + Enuc);
                }
            }
        }
    }

    return Erelax;
}

void THREE_DSRG_MRPT2::H1_T1_C1(BlockedTensor& H1, BlockedTensor& T1, const double& alpha,
                                BlockedTensor& C1) {
    Timer timer;

    C1["ip"] += alpha * H1["ap"] * T1["ia"];
    C1["qa"] -= alpha * T1["ia"] * H1["qi"];

    C1["IP"] += alpha * H1["AP"] * T1["IA"];
    C1["QA"] -= alpha * T1["IA"] * H1["QI"];

    if (print_ > 2) {
        outfile->Printf("\n    Time for [H1, T1] -> C1 : %12.3f", timer.get());
    }
    dsrg_time_.add("111", timer.get());
}

void THREE_DSRG_MRPT2::H1_T2_C1(BlockedTensor& H1, BlockedTensor& T2, const double& alpha,
                                BlockedTensor& C1) {
    Timer timer;

    C1["ia"] += alpha * H1["bm"] * T2["imab"];
    C1["ia"] += alpha * H1["bu"] * T2["ivab"] * Gamma1_["uv"];
    C1["ia"] -= alpha * H1["vj"] * T2["ijau"] * Gamma1_["uv"];
    C1["ia"] += alpha * H1["BM"] * T2["iMaB"];
    C1["ia"] += alpha * H1["BU"] * T2["iVaB"] * Gamma1_["UV"];
    C1["ia"] -= alpha * H1["VJ"] * T2["iJaU"] * Gamma1_["UV"];

    C1["IA"] += alpha * H1["bm"] * T2["mIbA"];
    C1["IA"] += alpha * H1["bu"] * Gamma1_["uv"] * T2["vIbA"];
    C1["IA"] -= alpha * H1["vj"] * T2["jIuA"] * Gamma1_["uv"];
    C1["IA"] += alpha * H1["BM"] * T2["IMAB"];
    C1["IA"] += alpha * H1["BU"] * T2["IVAB"] * Gamma1_["UV"];
    C1["IA"] -= alpha * H1["VJ"] * T2["IJAU"] * Gamma1_["UV"];

    if (print_ > 2) {
        outfile->Printf("\n    Time for [H1, T2] -> C1 : %12.3f", timer.get());
    }
    dsrg_time_.add("121", timer.get());
}

void THREE_DSRG_MRPT2::H2_T1_C1(BlockedTensor& H2, BlockedTensor& T1, const double& alpha,
                                BlockedTensor& C1) {
    Timer timer;

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
        outfile->Printf("\n    Time for [H2, T1] -> C1 : %12.3f", timer.get());
    }
    dsrg_time_.add("211", timer.get());
}

void THREE_DSRG_MRPT2::H2_T2_C1(BlockedTensor& H2, BlockedTensor& T2, const double& alpha,
                                BlockedTensor& C1) {
    Timer timer;
    BlockedTensor temp;

    // [Hbar2, T2] (C_2)^3 -> C1 particle contractions
    C1["ir"] += 0.5 * alpha * H2["abrm"] * T2["imab"];
    C1["ir"] += alpha * H2["aBrM"] * T2["iMaB"];
    C1["IR"] += 0.5 * alpha * H2["ABRM"] * T2["IMAB"];
    C1["IR"] += alpha * H2["aBmR"] * T2["mIaB"];

    C1["ir"] += 0.5 * alpha * Gamma1_["uv"] * H2["abru"] * T2["ivab"];
    C1["ir"] += alpha * Gamma1_["UV"] * H2["aBrU"] * T2["iVaB"];
    C1["IR"] += 0.5 * alpha * Gamma1_["UV"] * H2["ABRU"] * T2["IVAB"];
    C1["IR"] += alpha * Gamma1_["uv"] * H2["aBuR"] * T2["vIaB"];

    C1["ir"] += 0.5 * alpha * T2["ijux"] * Gamma1_["xy"] * Gamma1_["uv"] * H2["vyrj"];
    C1["IR"] += 0.5 * alpha * T2["IJUX"] * Gamma1_["XY"] * Gamma1_["UV"] * H2["VYRJ"];
    temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"hHaA"});
    temp["iJvY"] = T2["iJuX"] * Gamma1_["XY"] * Gamma1_["uv"];
    C1["ir"] += alpha * temp["iJvY"] * H2["vYrJ"];
    C1["IR"] += alpha * temp["jIvY"] * H2["vYjR"];

    C1["ir"] -= alpha * Gamma1_["uv"] * H2["vbrm"] * T2["imub"];
    C1["ir"] -= alpha * Gamma1_["uv"] * H2["vBrM"] * T2["iMuB"];
    C1["ir"] -= alpha * Gamma1_["UV"] * T2["iMbU"] * H2["bVrM"];
    C1["IR"] -= alpha * Gamma1_["UV"] * H2["VBRM"] * T2["IMUB"];
    C1["IR"] -= alpha * Gamma1_["UV"] * H2["bVmR"] * T2["mIbU"];
    C1["IR"] -= alpha * Gamma1_["uv"] * H2["vBmR"] * T2["mIuB"];

    C1["ir"] -= alpha * H2["vbrx"] * Gamma1_["uv"] * Gamma1_["xy"] * T2["iyub"];
    C1["ir"] -= alpha * H2["vBrX"] * Gamma1_["uv"] * Gamma1_["XY"] * T2["iYuB"];
    C1["ir"] -= alpha * H2["bVrX"] * Gamma1_["XY"] * Gamma1_["UV"] * T2["iYbU"];
    C1["IR"] -= alpha * H2["VBRX"] * Gamma1_["UV"] * Gamma1_["XY"] * T2["IYUB"];
    C1["IR"] -= alpha * H2["vBxR"] * Gamma1_["uv"] * Gamma1_["xy"] * T2["yIuB"];
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
    C1["pa"] += alpha * Eta1_["UV"] * H2["pEjU"] * T2["jVaE"];
    C1["PA"] += alpha * Eta1_["UV"] * T2["VJAE"] * H2["PEUJ"];
    C1["PA"] += alpha * Eta1_["uv"] * T2["vJeA"] * H2["ePuJ"];
    C1["PA"] += alpha * Eta1_["UV"] * H2["ePjU"] * T2["jVeA"];

    C1["pa"] += alpha * T2["vjax"] * Eta1_["uv"] * Eta1_["xy"] * H2["pyuj"];
    C1["pa"] += alpha * T2["vJaX"] * Eta1_["uv"] * Eta1_["XY"] * H2["pYuJ"];
    C1["pa"] += alpha * T2["jVaX"] * Eta1_["XY"] * Eta1_["UV"] * H2["pYjU"];
    C1["PA"] += alpha * T2["VJAX"] * Eta1_["UV"] * Eta1_["XY"] * H2["PYUJ"];
    C1["PA"] += alpha * T2["vJxA"] * Eta1_["uv"] * Eta1_["xy"] * H2["yPuJ"];
    C1["PA"] += alpha * H2["yPjU"] * Eta1_["UV"] * Eta1_["xy"] * T2["jVxA"];

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
        outfile->Printf("\n    Time for [H2, T2] -> C1 : %12.3f", timer.get());
    }
    dsrg_time_.add("221", timer.get());
}

void THREE_DSRG_MRPT2::H1_T2_C2(BlockedTensor& H1, BlockedTensor& T2, const double& alpha,
                                BlockedTensor& C2) {
    Timer timer;

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
        outfile->Printf("\n    Time for [H1, T2] -> C2 : %12.3f", timer.get());
    }
    dsrg_time_.add("122", timer.get());
}

void THREE_DSRG_MRPT2::H2_T1_C2(BlockedTensor& H2, BlockedTensor& T1, const double& alpha,
                                BlockedTensor& C2) {
    Timer timer;

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
        outfile->Printf("\n    Time for [H2, T1] -> C2 : %12.3f", timer.get());
    }
    dsrg_time_.add("212", timer.get());
}

void THREE_DSRG_MRPT2::H2_T2_C2(BlockedTensor& H2, BlockedTensor& T2, const double& alpha,
                                BlockedTensor& C2) {
    Timer timer;

    // particle-particle contractions
    C2["ijrs"] += 0.5 * alpha * H2["abrs"] * T2["ijab"];
    C2["iJrS"] += alpha * H2["aBrS"] * T2["iJaB"];
    C2["IJRS"] += 0.5 * alpha * H2["ABRS"] * T2["IJAB"];

    C2["ijrs"] -= alpha * Gamma1_["xy"] * H2["ybrs"] * T2["ijxb"];
    C2["iJrS"] -= alpha * Gamma1_["xy"] * H2["yBrS"] * T2["iJxB"];
    C2["iJrS"] -= alpha * Gamma1_["XY"] * T2["iJbX"] * H2["bYrS"];
    C2["IJRS"] -= alpha * Gamma1_["XY"] * H2["YBRS"] * T2["IJXB"];

    // hole-hole contractions
    C2["pqab"] += 0.5 * alpha * H2["pqij"] * T2["ijab"];
    C2["pQaB"] += alpha * H2["pQiJ"] * T2["iJaB"];
    C2["PQAB"] += 0.5 * alpha * H2["PQIJ"] * T2["IJAB"];

    C2["pqab"] -= alpha * Eta1_["xy"] * T2["yjab"] * H2["pqxj"];
    C2["pQaB"] -= alpha * Eta1_["xy"] * T2["yJaB"] * H2["pQxJ"];
    C2["pQaB"] -= alpha * Eta1_["XY"] * H2["pQjX"] * T2["jYaB"];
    C2["PQAB"] -= alpha * Eta1_["XY"] * T2["YJAB"] * H2["PQXJ"];

    // hole-particle contractions
    BlockedTensor temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"ghgp"});
    temp["qjsb"] += alpha * H2["aqms"] * T2["mjab"];
    temp["qjsb"] += alpha * H2["qAsM"] * T2["jMbA"];
    temp["qjsb"] += alpha * Gamma1_["xy"] * T2["yjab"] * H2["aqxs"];
    temp["qjsb"] += alpha * Gamma1_["XY"] * T2["jYbA"] * H2["qAsX"];
    temp["qjsb"] -= alpha * Gamma1_["xy"] * H2["yqis"] * T2["ijxb"];
    temp["qjsb"] -= alpha * Gamma1_["XY"] * H2["qYsI"] * T2["jIbX"];
    C2["qjsb"] += temp["qjsb"];
    C2["jqsb"] -= temp["qjsb"];
    C2["qjbs"] -= temp["qjsb"];
    C2["jqbs"] += temp["qjsb"];

    temp = ambit::BlockedTensor::build(tensor_type_, "temp", {"GHGP"});
    temp["QJSB"] += alpha * H2["AQMS"] * T2["MJAB"];
    temp["QJSB"] += alpha * H2["aQmS"] * T2["mJaB"];
    temp["QJSB"] += alpha * Gamma1_["XY"] * T2["YJAB"] * H2["AQXS"];
    temp["QJSB"] += alpha * Gamma1_["xy"] * T2["yJaB"] * H2["aQxS"];
    temp["QJSB"] -= alpha * Gamma1_["XY"] * H2["YQIS"] * T2["IJXB"];
    temp["QJSB"] -= alpha * Gamma1_["xy"] * H2["yQiS"] * T2["iJxB"];
    C2["QJSB"] += temp["QJSB"];
    C2["JQSB"] -= temp["QJSB"];
    C2["QJBS"] -= temp["QJSB"];
    C2["JQBS"] += temp["QJSB"];

    C2["qJsB"] += alpha * H2["aqms"] * T2["mJaB"];
    C2["qJsB"] += alpha * H2["qAsM"] * T2["MJAB"];
    C2["qJsB"] += alpha * Gamma1_["xy"] * T2["yJaB"] * H2["aqxs"];
    C2["qJsB"] += alpha * Gamma1_["XY"] * T2["YJAB"] * H2["qAsX"];
    C2["qJsB"] -= alpha * Gamma1_["xy"] * H2["yqis"] * T2["iJxB"];
    C2["qJsB"] -= alpha * Gamma1_["XY"] * H2["qYsI"] * T2["IJXB"];

    C2["iQsB"] -= alpha * T2["iMaB"] * H2["aQsM"];
    C2["iQsB"] -= alpha * Gamma1_["XY"] * T2["iYaB"] * H2["aQsX"];
    C2["iQsB"] += alpha * Gamma1_["xy"] * H2["yQsJ"] * T2["iJxB"];

    C2["qJaS"] -= alpha * T2["mJaB"] * H2["qBmS"];
    C2["qJaS"] -= alpha * Gamma1_["xy"] * T2["yJaB"] * H2["qBxS"];
    C2["qJaS"] += alpha * Gamma1_["XY"] * H2["qYiS"] * T2["iJaX"];

    C2["iQaS"] += alpha * T2["imab"] * H2["bQmS"];
    C2["iQaS"] += alpha * T2["iMaB"] * H2["BQMS"];
    C2["iQaS"] += alpha * Gamma1_["xy"] * T2["iyab"] * H2["bQxS"];
    C2["iQaS"] += alpha * Gamma1_["XY"] * T2["iYaB"] * H2["BQXS"];
    C2["iQaS"] -= alpha * Gamma1_["xy"] * H2["yQjS"] * T2["ijax"];
    C2["iQaS"] -= alpha * Gamma1_["XY"] * H2["YQJS"] * T2["iJaX"];

    if (print_ > 2) {
        outfile->Printf("\n    Time for [H2, T2] -> C2 : %12.3f", timer.get());
    }
    dsrg_time_.add("222", timer.get());
}

void THREE_DSRG_MRPT2::de_normal_order() {
    // printing
    print_h2("De-Normal-Order the DSRG Transformed Hamiltonian");

    // compute scalar term
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
    aone_eff_ = temp1.block("aa").data();
    bone_eff_ = temp1.block("AA").data();
    outfile->Printf("  Done. Timing %10.3f s", t_one.get());
    ints_->set_scalar(scalar);

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

    double Etest2 = 0.0;
    Etest2 += 0.25 * Hbar2_["uvxy"] * Lambda2_["xyuv"];
    Etest2 += 0.25 * Hbar2_["UVXY"] * Lambda2_["XYUV"];
    Etest2 += Hbar2_["uVxY"] * Lambda2_["xYuV"];

    Etest += Etest1 + Etest2;
    outfile->Printf("\n    %-35s = %22.15f", "One-Body Energy (after)", Etest1);
    outfile->Printf("\n    %-35s = %22.15f", "Two-Body Energy (after)", Etest2);
    outfile->Printf("\n    %-35s = %22.15f", "Total Energy (after)", Etest);
    outfile->Printf("\n    %-35s = %22.15f", "Total Energy (before)", Eref_ + Hbar0_);

    if (std::fabs(Etest - Eref_ - Hbar0_) > 100.0 * options_.get_double("E_CONVERGENCE")) {
        throw PSIEXCEPTION("De-normal-odering failed.");
    }
}

bool THREE_DSRG_MRPT2::check_semicanonical() {
    print_h2("Checking Orbitals");

    // zero diagonal elements
    F_.iterate([&](const std::vector<size_t>& i, const std::vector<SpinType>& spin, double& value) {
        if (spin[0] == AlphaSpin and (i[0] == i[1])) {
            value = 0.0;
        }
        if (spin[0] == BetaSpin and (i[0] == i[1])) {
            value = 0.0;
        }
    });

    // off diagonal elements of diagonal blocks
    double Foff_sum = 0.0;
    std::vector<double> Foff;
    for (const auto& block : {"cc", "aa", "vv", "CC", "AA", "VV"}) {
        double value = F_.block(block).norm();
        Foff.emplace_back(value);
        Foff_sum += value;
    }

    // add diagonal elements back
    F_.iterate([&](const std::vector<size_t>& i, const std::vector<SpinType>& spin, double& value) {
        if (spin[0] == AlphaSpin and (i[0] == i[1])) {
            value = Fa_[i[0]];
        }
        if (spin[0] == BetaSpin and (i[0] == i[1])) {
            value = Fb_[i[0]];
        }
    });

    bool semi = false;
    double threshold = 0.1 * std::sqrt(options_.get_double("E_CONVERGENCE"));
    if (Foff_sum > threshold) {
        std::string sep(2 + 16 * 3, '-');
        outfile->Printf("\n    Warning! Orbitals are not semi-canonicalized!");
        outfile->Printf("\n    Off-Diagonal norms of the core, active, virtual "
                        "blocks of Fock matrix");
        outfile->Printf("\n       %15s %15s %15s", "core", "active", "virtual");
        outfile->Printf("\n    %s", sep.c_str());
        outfile->Printf("\n    Fa %15.10f %15.10f %15.10f", Foff[0], Foff[1], Foff[2]);
        outfile->Printf("\n    Fb %15.10f %15.10f %15.10f", Foff[3], Foff[4], Foff[5]);
        outfile->Printf("\n    %s\n", sep.c_str());

        outfile->Printf("\n    DSRG energy is reliable roughly to the same "
                        "digit as max(|F_ij|, i != j), F: Fock diag. blocks.");
    } else {
        outfile->Printf("\n    Orbitals are semi-canonicalized.");
        semi = true;
    }

    if (ignore_semicanonical_ && Foff_sum > threshold) {
        std::string actv_type = options_.get_str("FCIMO_ACTV_TYPE");
        if (actv_type == "CIS" || actv_type == "CISD") {
            outfile->Printf("\n    It is OK for Fock (active) not being diagonal because %s "
                            "active space is incomplete.",
                            actv_type.c_str());
            outfile->Printf("\n    Please inspect if the Fock diag. blocks (C, AH, AP, V) "
                            "are diagonal or not in the prior CI step.");

        } else {
            outfile->Printf("\n    Warning: ignore testing of semi-canonical orbitals.");
            outfile->Printf("\n    Please inspect if the Fock diag. blocks (C, A, V) are "
                            "diagonal or not.");
        }

        semi = true;
    }
    outfile->Printf("\n");

    return semi;
}

std::vector<std::vector<double>> THREE_DSRG_MRPT2::diagonalize_Fock_diagblocks(BlockedTensor& U) {
    // diagonal blocks identifiers (C-A-V ordering)
    std::vector<std::string> blocks{"cc", "aa", "vv", "CC", "AA", "VV"};

    // map MO space label to its Dimension
    std::map<std::string, Dimension> MOlabel_to_dimension;
    MOlabel_to_dimension["c"] = mo_space_info_->get_dimension("RESTRICTED_DOCC");
    MOlabel_to_dimension["a"] = mo_space_info_->get_dimension("ACTIVE");
    MOlabel_to_dimension["v"] = mo_space_info_->get_dimension("RESTRICTED_UOCC");

    // eigen values to be returned
    size_t ncmo = mo_space_info_->size("CORRELATED");
    Dimension corr = mo_space_info_->get_dimension("CORRELATED");
    std::vector<double> eigenvalues_a(ncmo, 0.0);
    std::vector<double> eigenvalues_b(ncmo, 0.0);

    // map MO space label to its offset Dimension
    std::map<std::string, Dimension> MOlabel_to_offset_dimension;
    int nirrep = corr.n();
    MOlabel_to_offset_dimension["c"] = Dimension(std::vector<int>(nirrep, 0));
    MOlabel_to_offset_dimension["a"] = mo_space_info_->get_dimension("RESTRICTED_DOCC");
    MOlabel_to_offset_dimension["v"] =
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

ambit::Tensor THREE_DSRG_MRPT2::separate_tensor(ambit::Tensor& tens, const Dimension& irrep,
                                                const int& h) {
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

void THREE_DSRG_MRPT2::combine_tensor(ambit::Tensor& tens, ambit::Tensor& tens_h,
                                      const Dimension& irrep, const int& h) {
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

double THREE_DSRG_MRPT2::Tamp_deGNO() {
    // de-normal-order T1
    T1eff_ = BTF_->build(tensor_type_, "Effective T1 from de-GNO", spin_cases({"hp"}));

    T1eff_["ia"] = T1_["ia"];
    T1eff_["IA"] = T1_["IA"];

    T1eff_["ia"] -= T2_["iuav"] * Gamma1_["vu"];
    T1eff_["ia"] -= T2_["iUaV"] * Gamma1_["VU"];
    T1eff_["IA"] -= T2_["uIvA"] * Gamma1_["vu"];
    T1eff_["IA"] -= T2_["IUAV"] * Gamma1_["VU"];

    double out = 0.0;
    if (internal_amp_) {
        // the scalar term of amplitudes when de-normal-ordering
        out -= T1_["uv"] * Gamma1_["vu"];
        out -= T1_["UV"] * Gamma1_["VU"];

        out -= 0.25 * T2_["xyuv"] * Lambda2_["uvxy"];
        out -= 0.25 * T2_["XYUV"] * Lambda2_["UVXY"];
        out -= T2_["xYuV"] * Lambda2_["uVxY"];

        out += 0.5 * T2_["xyuv"] * Gamma1_["ux"] * Gamma1_["vy"];
        out += 0.5 * T2_["XYUV"] * Gamma1_["UX"] * Gamma1_["VY"];
        out += T2_["xYuV"] * Gamma1_["ux"] * Gamma1_["VY"];
    }

    return out;
}

ambit::BlockedTensor THREE_DSRG_MRPT2::get_T1(const std::vector<std::string>& blocks) {
    for (const std::string& block : blocks) {
        if (!T1_.is_block(block)) {
            std::string error = "Error from T1(blocks): cannot find block " + block;
            throw PSIEXCEPTION(error);
        }
    }
    ambit::BlockedTensor out = ambit::BlockedTensor::build(tensor_type_, "T1 selected", blocks);
    out["ia"] = T1_["ia"];
    out["IA"] = T1_["IA"];
    return out;
}

ambit::BlockedTensor THREE_DSRG_MRPT2::get_T1deGNO(const std::vector<std::string>& blocks) {
    for (const std::string& block : blocks) {
        if (!T1eff_.is_block(block)) {
            std::string error = "Error from T1deGNO(blocks): cannot find block " + block;
            throw PSIEXCEPTION(error);
        }
    }
    ambit::BlockedTensor out =
        ambit::BlockedTensor::build(tensor_type_, "T1deGNO selected", blocks);
    out["ia"] = T1eff_["ia"];
    out["IA"] = T1eff_["IA"];
    return out;
}

ambit::BlockedTensor THREE_DSRG_MRPT2::get_T2(const std::vector<std::string>& blocks) {
    for (const std::string& block : blocks) {
        if (!T2_.is_block(block)) {
            std::string error = "Error from T2(blocks): cannot find block " + block;
            throw PSIEXCEPTION(error);
        }
    }
    ambit::BlockedTensor out = ambit::BlockedTensor::build(tensor_type_, "T2 selected", blocks);
    out["ijab"] = T2_["ijab"];
    out["iJaB"] = T2_["iJaB"];
    out["IJAB"] = T2_["IJAB"];
    return out;
}

void THREE_DSRG_MRPT2::rotate_amp(SharedMatrix Ua, SharedMatrix Ub, const bool& transpose,
                                  const bool& t1eff) {
    ambit::BlockedTensor U = BTF_->build(tensor_type_, "Uorb", spin_cases({"gg"}));

    std::map<char, std::vector<std::pair<size_t, size_t>>> space_to_relmo;
    space_to_relmo['c'] = mo_space_info_->get_relative_mo("RESTRICTED_DOCC");
    space_to_relmo['a'] = mo_space_info_->get_relative_mo("ACTIVE");
    space_to_relmo['v'] = mo_space_info_->get_relative_mo("RESTRICTED_UOCC");

    // alpha
    for (const std::string& block : {"cc", "aa", "vv"}) {
        char space = block[0];

        U.block(block).iterate([&](const std::vector<size_t>& i, double& value) {
            std::pair<size_t, size_t> p0 = space_to_relmo[space][i[0]];
            std::pair<size_t, size_t> p1 = space_to_relmo[space][i[1]];
            size_t h0 = p0.first, h1 = p1.first;
            size_t i0 = p0.second, i1 = p1.second;

            if (h0 == h1) {
                if (transpose) {
                    value = Ua->get(h0, i1, i0);
                } else {
                    value = Ua->get(h0, i0, i1);
                }
            }
        });
    }

    // beta
    for (const std::string& block : {"CC", "AA", "VV"}) {
        char space = tolower(block[0]);

        U.block(block).iterate([&](const std::vector<size_t>& i, double& value) {
            std::pair<size_t, size_t> p0 = space_to_relmo[space][i[0]];
            std::pair<size_t, size_t> p1 = space_to_relmo[space][i[1]];
            size_t h0 = p0.first, h1 = p1.first;
            size_t i0 = p0.second, i1 = p1.second;

            if (h0 == h1) {
                if (transpose) {
                    value = Ub->get(h0, i1, i0);
                } else {
                    value = Ub->get(h0, i0, i1);
                }
            }
        });
    }

    // rotate amplitudes
    BlockedTensor temp = ambit::BlockedTensor::build(tensor_type_, "Temp T2", spin_cases({"hhpp"}));
    temp["klab"] = U["ik"] * U["jl"] * T2_["ijab"];
    temp["kLaB"] = U["ik"] * U["JL"] * T2_["iJaB"];
    temp["KLAB"] = U["IK"] * U["JL"] * T2_["IJAB"];
    T2_["ijcd"] = temp["ijab"] * U["bd"] * U["ac"];
    T2_["iJcD"] = temp["iJaB"] * U["BD"] * U["ac"];
    T2_["IJCD"] = temp["IJAB"] * U["BD"] * U["AC"];

    temp = ambit::BlockedTensor::build(tensor_type_, "Temp T1", spin_cases({"hp"}));
    temp["jb"] = U["ij"] * T1_["ia"] * U["ab"];
    temp["JB"] = U["IJ"] * T1_["IA"] * U["AB"];
    T1_["ia"] = temp["ia"];
    T1_["IA"] = temp["IA"];

    if (t1eff) {
        temp["jb"] = U["ij"] * T1eff_["ia"] * U["ab"];
        temp["JB"] = U["IJ"] * T1eff_["IA"] * U["AB"];
        T1eff_["ia"] = temp["ia"];
        T1eff_["IA"] = temp["IA"];
    }
}
}
} // End Namespaces
