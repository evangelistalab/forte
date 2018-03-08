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
#include <sstream>
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
#include "../aci/aci.h"
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
    : MASTER_DSRG(reference, ref_wfn, options, ints, mo_space_info) {

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

    print_method_banner({"Density Fitted / Cholesky Decomposed",
                         "MR-DSRG Second-Order Perturbation Theory",
                         "Kevin Hannon and Chenyang (York) Li", title_thread});
    outfile->Printf("\n    References:");
    outfile->Printf("\n      u-DSRG-MRPT2:      J. Chem. Theory Comput. 2015, 11, 2097.");
    outfile->Printf("\n      DF/CD-DSRG-MRPT2:  J. Chem. Phys. 2016, 144, 204111.");
    outfile->Printf("\n      (pr-)DSRG-MRPT2:   J. Chem. Phys. 2017, 146, 124132.");
    outfile->Printf("\n");

    if (options_.get_bool("MEMORY_SUMMARY")) {
        BTF_->print_memory_info();
    }

    // printf("\n P%d about to enter startup", my_proc);
    // GA_Sync();
    startup();
    if (my_proc == 0)
        print_options_summary();
}

THREE_DSRG_MRPT2::~THREE_DSRG_MRPT2() { cleanup(); }

void THREE_DSRG_MRPT2::startup() {
    int nproc = 1;
    int my_proc = 0;
#ifdef HAVE_MPI
    nproc = MPI::COMM_WORLD.Get_size();
    my_proc = MPI::COMM_WORLD.Get_rank();
#endif

    integral_type_ = ints_->integral_type();
    // GA_Sync();
    // printf("\n P%d integral_type", my_proc);

    ref_type_ = options_.get_str("REFERENCE");
    detail_time_ = options_.get_bool("THREE_MRPT2_TIMINGS");

    ncmopi_ = mo_space_info_->get_dimension("CORRELATED");
    ncmo_ = mo_space_info_->size("CORRELATED");

    // include internal amplitudes or not
    internal_amp_ = options_.get_str("INTERNAL_AMP") != "NONE";
    internal_amp_select_ = options_.get_str("INTERNAL_AMP_SELECT");

    // ignore semicanonical test
    std::string actv_type = options_.get_str("FCIMO_ACTV_TYPE");
    if (actv_type != "COMPLETE" && actv_type != "DOCI") {
        ignore_semicanonical_ = true;
    }

    rdoccpi_ = mo_space_info_->get_dimension("RESTRICTED_DOCC");
    actvpi_ = mo_space_info_->get_dimension("ACTIVE");
    ruoccpi_ = mo_space_info_->get_dimension("RESTRICTED_UOCC");

    ncore_ = core_mos_.size();
    nactive_ = actv_mos_.size();
    nvirtual_ = virt_mos_.size();

    // change options if not available
    if (relax_ref_ != "NONE" && relax_ref_ != "ONCE") {
        outfile->Printf("\n  Warning: RELAX_REF option \"%s\" is not supported. Change to ONCE.",
                        relax_ref_.c_str());
        relax_ref_ = "ONCE";

        warnings_.push_back(std::make_tuple("Unsupported RELAX_REF", "Change to ONCE",
                                            "Change options in input.dat"));
    }

    if (multi_state_ && multi_state_algorithm_ != "SA_FULL") {
        outfile->Printf("\n    Warning: %s is not supported in THREE-DSRG-MRPT2 at present.",
                        multi_state_algorithm_.c_str());
        outfile->Printf("\n             Set DSRG_MULTI_STATE back to default SA_FULL.");
        multi_state_algorithm_ = "SA_FULL";

        warnings_.push_back(std::make_tuple("Unsupported DSRG_MULTI_STATE", "Change to SA_FULL",
                                            "Change options in input.dat"));
    }

    // These two blocks of functions create a Blocked tensor
    // The block labels can be found in master_dsrg.cc
    std::vector<std::string> hhpp_no_cv = BTF_->generate_indices("cav", "hhpp");
    no_hhpp_ = hhpp_no_cv;

    if (my_proc == 0)
        nthree_ = ints_->nthree();

//    Timer naux_bcast;
#ifdef HAVE_MPI
    MPI_Bcast(&nthree_, 1, MPI_INT, 0, MPI_COMM_WORLD);
// printf("\n P%d took %8.8f s to broadcast %d size", my_proc, naux_bcast.get(),
// nthree_);
#endif

    if (my_proc == 0) {
        // bare 1e part of Hamiltonian
        H_ = BTF_->build(tensor_type_, "H", spin_cases({"gg"}));
        H_.iterate(
            [&](const std::vector<size_t>& i, const std::vector<SpinType>& spin, double& value) {
                if (spin[0] == AlphaSpin)
                    value = ints_->oei_a(i[0], i[1]);
                else
                    value = ints_->oei_b(i[0], i[1]);
            });

        // make a copy of Fock from MASTER_DSRG
        // as such, there is always a copy of non-renormalized Fock matrix
        F_ = BTF_->build(tensor_type_, "Fock", spin_cases({"gg"}));
        F_["pq"] = Fock_["pq"];
        F_["PQ"] = Fock_["PQ"];
        Fa_ = Fdiag_a_;
        Fb_ = Fdiag_b_;

        if (print_ > 1) {
            Gamma1_.print(stdout);
            Eta1_.print(stdout);
            F_.print(stdout);
            H_.print(stdout);
        }

        // some 1-body tensors
        Delta1_ = BTF_->build(tensor_type_, "Delta1_", spin_cases({"aa"}));
        RDelta1_ = BTF_->build(tensor_type_, "RDelta1_", spin_cases({"hp"}));
        RExp1_ = BTF_->build(tensor_type_, "RExp1", spin_cases({"hp"}));

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

        RExp1_.iterate(
            [&](const std::vector<size_t>& i, const std::vector<SpinType>& spin, double& value) {
                if (spin[0] == AlphaSpin) {
                    value = dsrg_source_->compute_renormalized(Fa_[i[0]] - Fa_[i[1]]);
                } else if (spin[0] == BetaSpin) {
                    value = dsrg_source_->compute_renormalized(Fb_[i[0]] - Fb_[i[1]]);
                }
            });

        // allocate memory for T1
        T1_ = BTF_->build(tensor_type_, "T1 Amplitudes", spin_cases({"hp"}));

        // allocate memory for T2 and V
        if (integral_type_ != DiskDF) {
            std::vector<std::string> list_of_pphh_V = BTF_->generate_indices("vac", "pphh");
            V_ = BTF_->build(tensor_type_, "V_", BTF_->spin_cases_avoid(list_of_pphh_V, 1));
            T2_ = BTF_->build(tensor_type_, "T2 Amplitudes", BTF_->spin_cases_avoid(no_hhpp_, 1));
            ThreeIntegral_ = BTF_->build(tensor_type_, "ThreeInt", {"Lph", "LPH"});

            std::vector<std::string> ThreeInt_block = ThreeIntegral_.block_labels();

            for (std::string& string_block : ThreeInt_block) {
                std::vector<size_t> first_index = label_to_spacemo_[string_block[0]];
                std::vector<size_t> second_index = label_to_spacemo_[string_block[1]];
                std::vector<size_t> third_index = label_to_spacemo_[string_block[2]];

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

        // Prepare Hbar
        if (relax_ref_ != "NONE" || multi_state_) {
            Hbar1_ = BTF_->build(tensor_type_, "One-body Hbar", spin_cases({"aa"}));
            Hbar2_ = BTF_->build(tensor_type_, "Two-body Hbar", spin_cases({"aaaa"}));
            Hbar1_["uv"] = F_["uv"];
            Hbar1_["UV"] = F_["UV"];

            if (options_.get_bool("FORM_HBAR3")) {
                Hbar3_ = BTF_->build(tensor_type_, "3-body Hbar", spin_cases({"aaaaaa"}));
            }
        }
    }
}

void THREE_DSRG_MRPT2::print_options_summary() {
    // Print a summary
    std::vector<std::pair<std::string, int>> calculation_info_int;

    std::vector<std::pair<std::string, double>> calculation_info_double{
        {"Flow parameter", s_},
        {"Taylor expansion threshold", std::pow(10.0, -double(taylor_threshold_))},
        {"Cholesky tolerance", options_.get_double("CHOLESKY_TOLERANCE")}};

    std::vector<std::pair<std::string, std::string>> calculation_info_string{
        {"Psi4 ref_type", ref_type_},
        {"Integral type", ints_type_},
        {"Source operator", source_},
        {"CCVV algorithm", options_.get_str("CCVV_ALGORITHM")},
        {"CCVV source", options_.get_str("CCVV_SOURCE")},
        {"Reference relaxation", relax_ref_}};

    if (multi_state_) {
        calculation_info_string.push_back({"State type", "MULTI-STATE"});
        calculation_info_string.push_back({"Multi-state type", multi_state_algorithm_});
    } else {
        calculation_info_string.push_back({"State type", "STATE-SPECIFIC"});
    }

    if (internal_amp_) {
        calculation_info_string.push_back({"Internal_amp", options_.get_str("INTERNAL_AMP")});
        calculation_info_string.push_back({"Internal_amp_select", internal_amp_select_});
    }

    if (options_.get_bool("FORM_HBAR3")) {
        calculation_info_string.push_back({"form Hbar3", "TRUE"});
    } else {
        calculation_info_string.push_back({"form Hbar3", "FALSE"});
    }

    // Print some information
    print_h2("Calculation Information");
    for (auto& str_dim : calculation_info_int) {
        outfile->Printf("\n    %-40s %15d", str_dim.first.c_str(), str_dim.second);
    }
    for (auto& str_dim : calculation_info_double) {
        outfile->Printf("\n    %-40s %15.3e", str_dim.first.c_str(), str_dim.second);
    }
    for (auto& str_dim : calculation_info_string) {
        outfile->Printf("\n    %-40s %15s", str_dim.first.c_str(), str_dim.second.c_str());
    }
}

void THREE_DSRG_MRPT2::cleanup() {}

double THREE_DSRG_MRPT2::compute_energy() {
    Timer ComputeEnergy;
    int my_proc = 0;
    int nproc = 1;
#ifdef HAVE_MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &my_proc);
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);
#endif

    if (my_proc == 0) {
        // check semi-canonical orbitals
        semi_canonical_ = check_semicanonical();
        if (!semi_canonical_) {
            outfile->Printf("\n    Warning: DF/CD-DSRG-MRPT2 only takes semi-canonical orbitals. "
                            "The code will keep running.");

            warnings_.push_back(std::make_tuple("Semicanonical orbital test",
                                                "Assume Semicanonical orbital",
                                                "Change options in input.dat"));
        }

        print_h2("Computing DSRG-MRPT2 Energy");
        outfile->Printf("\n  Reference Energy = %.15f", Eref_);

        // compute T2 and renormalize V
        if (integral_type_ != DiskDF) {
            compute_t2();
            renormalize_V();
        } else {
            // we only store V and T2 with at least two active indices
            outfile->Printf("\n    %-40s ...", "Computing minimal T2");
            ForteTimer T2timer;
            T2_ = compute_T2_minimal(BTF_->spin_cases_avoid(no_hhpp_, 2));
            outfile->Printf("... Done. Timing %15.6f s", T2timer.elapsed());

            outfile->Printf("\n    %-40s ...", "Renormalizing minimal V");
            ForteTimer Vtimer;
            std::vector<std::string> list_of_pphh_V = BTF_->generate_indices("vac", "pphh");
            V_ = compute_V_minimal(BTF_->spin_cases_avoid(list_of_pphh_V, 2));
            outfile->Printf("... Done. Timing %15.6f s", Vtimer.elapsed());
        }

        // compute T1
        compute_t1();

        // renormalize F
        renormalize_F();

        if (print_ > 1) {
            F_.print();
        }
        if (print_ > 2) {
            T1_.print();
        }
    }

    // Compute DSRG-MRPT2 correlation energy, special treatment for ccvv term
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
    }

    // printf("\n P%d about to enter E_VT2_2", my_proc);
    // GA_Sync();
    Etemp = E_VT2_2();

    if (my_proc == 0) {
        EVT2 += Etemp;
        energy.push_back({"<[V, T2]> (C_2)^4", Etemp});

        Ecorr += EVT2;
        Etotal = Ecorr + Eref_;
        Hbar0_ = Ecorr;
        energy.push_back({"<[V, T2]>", EVT2});
        energy.push_back({"DSRG-MRPT2 correlation energy", Ecorr});
        energy.push_back({"DSRG-MRPT2 total energy", Etotal});

        // Analyze T1 and T2
        check_t1();
        energy.push_back({"max(T1)", T1max_});
        energy.push_back({"||T1||", T1norm_});

        // Print energy summary
        print_h2("DSRG-MRPT2 (DF/CD) Energy Summary");
        for (auto& str_dim : energy)
            outfile->Printf("\n    %-30s = %22.15f", str_dim.first.c_str(), str_dim.second);
    }

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

#ifdef HAVE_MPI
    MPI_Bcast(&Etotal, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&Hbar0_, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
#endif

    Process::environment.globals["UNRELAXED ENERGY"] = Etotal;
    Process::environment.globals["CURRENT ENERGY"] = Etotal;

    // use relaxation code to do SA_FULL
    if (relax_ref_ != "NONE" || multi_state_) {
        if (my_proc == 0) {
            form_Hbar();
        }
    }

    return Etotal;
}

double THREE_DSRG_MRPT2::compute_ref() {
    double E = 0.0;

    for (const std::string block : {"cc", "CC"}) {
        F_.block(block).iterate([&](const std::vector<size_t>& i, double& value) {
            if (i[0] == i[1]) {
                E += 0.5 * value;
            }
        });
        H_.block(block).iterate([&](const std::vector<size_t>& i, double& value) {
            if (i[0] == i[1]) {
                E += 0.5 * value;
            }
        });
    }

    E = 0.5 * H_["uv"] * Gamma1_["vu"];
    E += 0.5 * F_["uv"] * Gamma1_["vu"];
    E += 0.5 * H_["UV"] * Gamma1_["VU"];
    E += 0.5 * F_["UV"] * Gamma1_["VU"];

    E += 0.25 * V_["uvxy"] * Lambda2_["uvxy"];
    E += 0.25 * V_["UVXY"] * Lambda2_["UVXY"];
    E += V_["uVxY"] * Lambda2_["uVxY"];

    return E + Efrzc_ + Enuc_;
}

void THREE_DSRG_MRPT2::compute_t2() {
    outfile->Printf("\n    %-40s ...", "Computing T2");
    ForteTimer timer;

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
    if (internal_amp.find("DOUBLES") != string::npos) {
        size_t nactv1 = mo_space_info_->size("ACTIVE");
        size_t nactv2 = nactv1 * nactv1;
        size_t nactv3 = nactv2 * nactv1;
        size_t nactv_occ = actv_occ_mos_.size();
        size_t nactv_uocc = actv_uocc_mos_.size();

        if (internal_amp_select_ == "ALL") {
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
        } else if (internal_amp_select_ == "OOVV") {
            for (const std::string& block : {"aaaa", "aAaA", "AAAA"}) {
                // copy original data
                std::vector<double> data(T2_.block(block).data());

                T2_.block(block).zero();
                for (size_t I = 0; I < nactv_occ; ++I) {
                    for (size_t J = 0; J < nactv_occ; ++J) {
                        for (size_t A = 0; A < nactv_uocc; ++A) {
                            for (size_t B = 0; B < nactv_uocc; ++B) {
                                size_t idx = actv_occ_mos_[I] * nactv3 + actv_occ_mos_[J] * nactv2 +
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
                                size_t idx = actv_occ_mos_[I] * nactv3 + actv_occ_mos_[J] * nactv2 +
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
                                size_t idx = actv_occ_mos_[I] * nactv3 + actv_occ_mos_[J] * nactv2 +
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

    outfile->Printf("... Done. Timing %15.6f s", timer.elapsed());
}

ambit::BlockedTensor
THREE_DSRG_MRPT2::compute_T2_minimal(const std::vector<std::string>& t2_spaces) {
    ambit::BlockedTensor T2min;

    T2min = BTF_->build(tensor_type_, "T2min", t2_spaces, true);
    ForteTimer timer_b_min;
    ambit::BlockedTensor ThreeInt = compute_B_minimal(t2_spaces);
    if (detail_time_)
        outfile->Printf("\n Took %8.4f s to compute_B_minimal", timer_b_min.elapsed());
    ForteTimer v_t2;
    T2min["ijab"] = (ThreeInt["gia"] * ThreeInt["gjb"]);
    T2min["ijab"] -= (ThreeInt["gib"] * ThreeInt["gja"]);
    T2min["IJAB"] = (ThreeInt["gIA"] * ThreeInt["gJB"]);
    T2min["IJAB"] -= (ThreeInt["gIB"] * ThreeInt["gJA"]);
    T2min["iJaB"] = (ThreeInt["gia"] * ThreeInt["gJB"]);
    if (detail_time_)
        outfile->Printf("\n Took %8.4f s to compute T2 from B", v_t2.elapsed());

    ForteTimer t2_iterate;
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
        outfile->Printf("\n T2 iteration takes %8.4f s", t2_iterate.elapsed());

    // internal amplitudes (AA->AA)
    std::string internal_amp = options_.get_str("INTERNAL_AMP");

    for (const std::string& block : {"aaaa", "aAaA", "AAAA"}) {
        if (std::find(t2_spaces.begin(), t2_spaces.end(), block) != t2_spaces.end()) {

            if (internal_amp.find("DOUBLES") != string::npos) {
                size_t nactv1 = mo_space_info_->size("ACTIVE");
                size_t nactv2 = nactv1 * nactv1;
                size_t nactv3 = nactv2 * nactv1;
                size_t nactv_occ = actv_occ_mos_.size();
                size_t nactv_uocc = actv_uocc_mos_.size();

                if (internal_amp_select_ == "ALL") {
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
                } else if (internal_amp_select_ == "OOVV") {
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
    ForteTimer computeB;
    ThreeInt = compute_B_minimal(spaces);
    if (detail_time_) {
        outfile->Printf("\n  Compute B minimal takes %8.6f s", computeB.elapsed());
    }
    ForteTimer ComputeV;
    Vmin["abij"] = ThreeInt["gai"] * ThreeInt["gbj"];
    Vmin["abij"] -= ThreeInt["gaj"] * ThreeInt["gbi"];
    Vmin["ABIJ"] = ThreeInt["gAI"] * ThreeInt["gBJ"];
    Vmin["ABIJ"] -= ThreeInt["gAJ"] * ThreeInt["gBI"];
    Vmin["aBiJ"] = ThreeInt["gai"] * ThreeInt["gBJ"];
    if (detail_time_) {
        outfile->Printf("\n  Compute V from B takes %8.6f s", ComputeV.elapsed());
    }

    if (renormalize) {
        ForteTimer RenormV;
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
            outfile->Printf("\n  RenormalizeV takes %8.6f s.", RenormV.elapsed());
        }
    }
    return Vmin;
}

ambit::BlockedTensor THREE_DSRG_MRPT2::compute_B_minimal(const std::vector<std::string>& spaces) {
    /**
     * @param spaces Space labels for antisymmetrized rank-4 tensors
     * <pq||rs> = <pq|rs> - <pq|sr>
     *          = (pr|qs) - (ps|qr)
     *          = (L|pr) * (L|qs) - (L|ps) * (L|qr)
     *            - J0 -   - J1 -   - K0 -   - K1 -
     *            ------ J ------   ------ K ------
     */

    std::vector<std::string> ThreeIntegral_labels;
    for (const auto& label : spaces) {
        // for all spin cases, J term is a must
        std::string J0("L");
        std::string J1("L");

        J0 += label[0];
        J0 += label[2];

        J1 += label[1];
        J1 += label[3];

        if (std::find(ThreeIntegral_labels.begin(), ThreeIntegral_labels.end(), J0) ==
            ThreeIntegral_labels.end()) {
            ThreeIntegral_labels.push_back(J0);
        }
        if (std::find(ThreeIntegral_labels.begin(), ThreeIntegral_labels.end(), J1) ==
            ThreeIntegral_labels.end()) {
            ThreeIntegral_labels.push_back(J1);
        }

        // for aa or bb spin cases, K term should be considered
        bool aa = std::islower(label[0]) && std::islower(label[1]);
        bool bb = std::isupper(label[0]) && std::isupper(label[1]);
        if (aa || bb) {
            std::string K0("L");
            std::string K1("L");

            K0 += label[0];
            K0 += label[3];

            K1 += label[1];
            K1 += label[2];

            if (std::find(ThreeIntegral_labels.begin(), ThreeIntegral_labels.end(), K0) ==
                ThreeIntegral_labels.end()) {
                ThreeIntegral_labels.push_back(K0);
            }
            if (std::find(ThreeIntegral_labels.begin(), ThreeIntegral_labels.end(), K1) ==
                ThreeIntegral_labels.end()) {
                ThreeIntegral_labels.push_back(K1);
            }
        }
    }

    ambit::BlockedTensor ThreeInt =
        BTF_->build(tensor_type_, "ThreeIntMin", ThreeIntegral_labels, true);

    std::vector<std::string> ThreeInt_block = ThreeInt.block_labels();

    for (std::string& string_block : ThreeInt_block) {
        std::vector<size_t> first_index = label_to_spacemo_[string_block[0]];
        std::vector<size_t> second_index = label_to_spacemo_[string_block[1]];
        std::vector<size_t> third_index = label_to_spacemo_[string_block[2]];

        ambit::Tensor ThreeIntegral_block =
            ints_->three_integral_block(first_index, second_index, third_index);
        ThreeInt.block(string_block).copy(ThreeIntegral_block);
    }

    return ThreeInt;
}

void THREE_DSRG_MRPT2::compute_t1() {
    outfile->Printf("\n    %-40s ...", "Computing T1");
    ForteTimer timer;
    BlockedTensor temp = BTF_->build(tensor_type_, "temp", spin_cases({"aa"}), true);
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

    outfile->Printf("... Done. Timing %15.6f s", timer.elapsed());
}

void THREE_DSRG_MRPT2::check_t1() {
    // norm and maximum of T1 amplitudes
    T1norm_ = T1_.norm();
    T1max_ = T1_.norm(0);
}

void THREE_DSRG_MRPT2::renormalize_V() {
    ForteTimer timer;
    outfile->Printf("\n    %-40s ...", "Renormalizing V");

    V_.iterate([&](const std::vector<size_t>& i, const std::vector<SpinType>& spin, double& value) {
        if ((spin[0] == AlphaSpin) and (spin[1] == AlphaSpin)) {
            value *=
                1.0 +
                dsrg_source_->compute_renormalized(Fa_[i[0]] + Fa_[i[1]] - Fa_[i[2]] - Fa_[i[3]]);
        } else if ((spin[0] == AlphaSpin) and (spin[1] == BetaSpin)) {
            value *=
                1.0 +
                dsrg_source_->compute_renormalized(Fa_[i[0]] + Fb_[i[1]] - Fa_[i[2]] - Fb_[i[3]]);
        } else if ((spin[0] == BetaSpin) and (spin[1] == BetaSpin)) {
            value *=
                1.0 +
                dsrg_source_->compute_renormalized(Fb_[i[0]] + Fb_[i[1]] - Fb_[i[2]] - Fb_[i[3]]);
        }
    });

    outfile->Printf("... Done. Timing %15.6f s", timer.elapsed());
}

void THREE_DSRG_MRPT2::renormalize_F() {
    ForteTimer timer;
    outfile->Printf("\n    %-40s ...", "Renormalizing F");

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
    outfile->Printf("... Done. Timing %15.6f s", timer.elapsed());
}

double THREE_DSRG_MRPT2::E_FT1() {
    ForteTimer timer;
    outfile->Printf("\n    %-40s ...", "Computing <[F, T1]>");

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

    outfile->Printf("... Done. Timing %15.6f s", timer.elapsed());
    dsrg_time_.add("110", timer.elapsed());
    return E;
}

double THREE_DSRG_MRPT2::E_VT1() {
    ForteTimer timer;
    outfile->Printf("\n    %-40s ...", "Computing <[V, T1]>");

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

    outfile->Printf("... Done. Timing %15.6f s", timer.elapsed());
    dsrg_time_.add("210", timer.elapsed());
    return E;
}

double THREE_DSRG_MRPT2::E_FT2() {
    ForteTimer timer;
    outfile->Printf("\n    %-40s ...", "Computing <[F, T2]>");

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

    outfile->Printf("... Done. Timing %15.6f s", timer.elapsed());
    dsrg_time_.add("120", timer.elapsed());
    return E;
}

double THREE_DSRG_MRPT2::E_VT2_2() {
    double E = 0.0;
    int my_proc = 0;
#ifdef HAVE_MPI
    my_proc = MPI::COMM_WORLD.Get_rank();
#endif
    ambit::BlockedTensor temp = BTF_->build(tensor_type_, "temp", {"aa", "AA"});
    ForteTimer timer;
    if (my_proc == 0) {
        outfile->Printf("\n    %-40s ...", "Computing <[V, T2]> (C_2)^4 (no ccvv)");
        // TODO: Implement these without storing V and/or T2 by using blocking
        if (integral_type_ != DiskDF) {
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

        /// These terms all have at least two active indices (assume can be store in core).
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
        outfile->Printf("... Done. Timing %15.6f s", timer.elapsed());
    }

    // Calculates all but ccvv, cCvV, and CCVV energies
    double Eccvv = 0.0;
    std::string ccvv_algorithm = options_.get_str("ccvv_algorithm");
    ForteTimer ccvv_timer;
    if (my_proc == 0) {
        outfile->Printf("\n    %-40s ...", "Computing <[V, T2]> (C_2)^4 ccvv");
    }

    // TODO: Make this smarter and automatically switch to right algorithm for size
    // Small size -> use core algorithm
    // Large size -> use fly_ambit
    if (ccvv_algorithm == "CORE") {
        if (my_proc == 0)
            Eccvv = E_VT2_2_core();
    } else if (ccvv_algorithm == "FLY_LOOP") {
        if (my_proc == 0)
            Eccvv = E_VT2_2_fly_openmp();
    } else if (ccvv_algorithm == "FLY_AMBIT") {
        if (my_proc == 0)
            Eccvv = E_VT2_2_ambit();
    } else if (ccvv_algorithm == "BATCH_CORE") {
        if (my_proc == 0)
            Eccvv = E_VT2_2_batch_core();
    } else if (ccvv_algorithm == "BATCH_CORE_GA") {
#ifdef HAVE_MPI
        Eccvv = E_VT2_2_batch_core_ga();
#endif
    } else if (ccvv_algorithm == "BATCH_CORE_REP") {
#ifdef HAVE_MPI
        Eccvv = E_VT2_2_batch_core_rep();
#endif
    } else if (ccvv_algorithm == "BATCH_CORE_MPI") {
#ifdef HAVE_MPI
        Eccvv = E_VT2_2_batch_core_mpi();
#endif
    } else if (ccvv_algorithm == "BATCH_VIRTUAL") {
        if (my_proc == 0)
            Eccvv = E_VT2_2_batch_virtual();
    } else if (ccvv_algorithm == "BATCH_VIRTUAL_GA") {
#ifdef HAVE_MPI
        Eccvv = E_VT2_2_batch_virtual_ga();
#endif
    } else if (ccvv_algorithm == "BATCH_VIRTUAL_REP") {
#ifdef HAVE_MPI
        Eccvv = E_VT2_2_batch_virtual_rep();
#endif
    } else if (ccvv_algorithm == "BATCH_VIRTUAL_MPI") {
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
        outfile->Printf("\n  Eccvv_ao: %8.10f", Eccvv_ao);
    }

    if (my_proc == 0) {
        outfile->Printf("... Done. Timing %15.6f s", ccvv_timer.elapsed());
        outfile->Printf("\n  Eccvv: %8.10f", Eccvv);
    }

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

    dsrg_time_.add("220", timer.elapsed());
    return (E + Eccvv);
}

double THREE_DSRG_MRPT2::E_VT2_4HH() {
    ForteTimer timer;
    outfile->Printf("\n    %-40s ...", "Computing <[V, T2]> 4HH");

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

    outfile->Printf("... Done. Timing %15.6f s", timer.elapsed());
    dsrg_time_.add("220", timer.elapsed());
    return E;
}

double THREE_DSRG_MRPT2::E_VT2_4PP() {
    ForteTimer timer;
    outfile->Printf("\n    %-40s ...", "Computing <V, T2]> 4PP");

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

    outfile->Printf("... Done. Timing %15.6f s", timer.elapsed());
    dsrg_time_.add("220", timer.elapsed());
    return E;
}

double THREE_DSRG_MRPT2::E_VT2_4PH() {
    ForteTimer timer;
    outfile->Printf("\n    %-40s ...", "Computing [V, T2] 4PH");

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

    outfile->Printf("... Done. Timing %15.6f s", timer.elapsed());
    dsrg_time_.add("220", timer.elapsed());
    return E;
}

double THREE_DSRG_MRPT2::E_VT2_6() {
    ForteTimer timer;
    outfile->Printf("\n    %-40s  ...", "Computing [V, T2] λ3");
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
            temp["uvWxyZ"] -= V_["uviy"] * T2_["iWxZ"];
            temp["uvWxyZ"] -= V_["uWiZ"] * T2_["ivxy"];
            temp["uvWxyZ"] += 2.0 * V_["uWyI"] * T2_["vIxZ"];

            temp["uvWxyZ"] += V_["aWxZ"] * T2_["uvay"];
            temp["uvWxyZ"] -= V_["vaxy"] * T2_["uWaZ"];
            temp["uvWxyZ"] -= 2.0 * V_["vAxZ"] * T2_["uWyA"];

            E += 0.50 * temp.block("aaAaaA")("uvWxyZ") * reference_.L3aab()("xyZuvW");

            // abb
            temp = BTF_->build(tensor_type_, "temp", {"aAAaAA"});
            temp["uVWxYZ"] -= V_["VWIZ"] * T2_["uIxY"];
            temp["uVWxYZ"] -= V_["uVxI"] * T2_["IWYZ"];
            temp["uVWxYZ"] += 2.0 * V_["uViZ"] * T2_["iWxY"];

            temp["uVWxYZ"] += V_["uAxY"] * T2_["VWAZ"];
            temp["uVWxYZ"] -= V_["WAYZ"] * T2_["uVxA"];
            temp["uVWxYZ"] -= 2.0 * V_["aWxY"] * T2_["uVaZ"];

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

    outfile->Printf("... Done. Timing %15.6f s", timer.elapsed());
    return E;
}

double THREE_DSRG_MRPT2::E_VT2_2_fly_openmp() {
    double Eflyalpha = 0.0;
    double Eflybeta = 0.0;
    double Eflymixed = 0.0;
    double Efly = 0.0;
#pragma omp parallel for num_threads(num_threads_) schedule(dynamic)                               \
    reduction(+ : Eflyalpha, Eflybeta, Eflymixed)
    for (size_t mind = 0; mind < ncore_; mind++) {
        for (size_t nind = 0; nind < ncore_; nind++) {
            for (size_t eind = 0; eind < nvirtual_; eind++) {
                for (size_t find = 0; find < nvirtual_; find++) {
                    // These are used because active is not partitioned as simple as
                    // core orbs -- active orbs -- virtual
                    // This also takes in account symmetry labeled
                    size_t m = core_mos_[mind];
                    size_t n = core_mos_[nind];
                    size_t e = virt_mos_[eind];
                    size_t f = virt_mos_[find];
                    size_t mb = core_mos_[mind];
                    size_t nb = core_mos_[nind];
                    size_t eb = virt_mos_[eind];
                    size_t fb = virt_mos_[find];
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
    /**
     * Compute <[V, T2]> (C_2)^4 ccvv term
     * E = 0.25 * <mn||ef> * <mn||ef> * [1 - e^(-2 * s * D)] / D
     *   = 0.5 * [(me|nf)(me|nf) - (me|nf)(mf|ne)] * [1 - e^(-s * Daa)] / Daa * [1 + e^(-s * Daa)]
     *             __ __  __ __     __ __  __ __
     *   + 0.5 * [(me|nf)(me|nf) - (me|nf)(mf|ne)] * [1 - e^(-s * Dbb)] / Dbb * [1 + e^(-s * Dbb)]
     *         __     __
     *   + (me|nf)(me|nf) * [1 - e^(-s * Dab)] / Dab * [1 + e^(-s * Dab)]
     *
     * Ignoring spin cases:
     * E = (me|nf) * [2 * (me|nf) - (mf|ne)] * [1 - e^(-2 * s * D)] / D
     *
     * Batching: for a given m and n, form B(ef) = Bm(L|e) * Bn(L|f)
     *
     * TODO:
     *   This function needs clean up.
     *   It seems Kevin wants to ignore the beta spin, but UMP2 equations are used.
     */

    size_t dim = nthree_ * nvirtual_;

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
            BmaVec.push_back(ambit::Tensor::build(tensor_type_, "Bma", {nthree_, nvirtual_}));
            BnaVec.push_back(ambit::Tensor::build(tensor_type_, "Bna", {nthree_, nvirtual_}));
            BmbVec.push_back(ambit::Tensor::build(tensor_type_, "Bmb", {nthree_, nvirtual_}));
            BnbVec.push_back(ambit::Tensor::build(tensor_type_, "Bnb", {nthree_, nvirtual_}));
            BefVec.push_back(ambit::Tensor::build(tensor_type_, "Bef", {nvirtual_, nvirtual_}));
            BefJKVec.push_back(ambit::Tensor::build(tensor_type_, "BefJK", {nvirtual_, nvirtual_}));
            RDVec.push_back(ambit::Tensor::build(tensor_type_, "RDVec", {nvirtual_, nvirtual_}));
        }
        bool ao_dsrg_check = options_.get_bool("AO_DSRG_MRPT2");

#pragma omp parallel for num_threads(num_threads_) reduction(+ : Ealpha, Ebeta, Emixed)
        for (size_t m = 0; m < ncore_; ++m) {

            int thread = 0;
#ifdef _OPENMP
            thread = omp_get_thread_num();
#endif

            size_t ma = core_mos_[m];
#pragma omp critical
            {
                BmaVec[thread] = ints_->three_integral_block_two_index(aux_mos_, ma, virt_mos_);
                BmbVec[thread] = ints_->three_integral_block_two_index(aux_mos_, ma, virt_mos_);
            }
            for (size_t n = m; n < ncore_; ++n) {
                size_t na = core_mos_[n];
#pragma omp critical
                {
                    BnaVec[thread] = ints_->three_integral_block_two_index(aux_mos_, na, virt_mos_);
                    BnbVec[thread] = ints_->three_integral_block_two_index(aux_mos_, na, virt_mos_);
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
                    double D = Fa_[ma] + Fa_[na] - Fa_[virt_mos_[i[0]]] - Fa_[virt_mos_[i[1]]];
                    if (ao_dsrg_check)
                        value = 1.0 / D;
                    else {
                        value = dsrg_source_->compute_renormalized_denominator(D) *
                                (1.0 + dsrg_source_->compute_renormalized(D));
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
                    double D = Fa_[ma] + Fa_[na] - Fa_[virt_mos_[i[0]]] - Fa_[virt_mos_[i[1]]];
                    if (ao_dsrg_check)
                        value = 1.0 / D;
                    else {
                        value = dsrg_source_->compute_renormalized_denominator(D) *
                                (1.0 + dsrg_source_->compute_renormalized(D));
                    }
                });
                Emixed += factor * BefJKVec[thread]("eF") * RDVec[thread]("eF");
            }
        }
    }
    // This block of code runs with DF and assumes that ThreeIntegral_ is
    // created in startup.  Will fail for systems around 800 or 900 BF
    /// We should change integral class such that only the useful part of B is stored
    else {
        ambit::Tensor Ba = ambit::Tensor::build(tensor_type_, "Ba", {ncore_, nthree_, nvirtual_});
        ambit::Tensor Bb = ambit::Tensor::build(tensor_type_, "Bb", {ncore_, nthree_, nvirtual_});
        Ba("mge") = (ThreeIntegral_.block("Lvc"))("gem");
        Bb("MgE") = (ThreeIntegral_.block("Lvc"))("gEM");

        std::vector<ambit::Tensor> BmaVec;
        std::vector<ambit::Tensor> BnaVec;
        std::vector<ambit::Tensor> BmbVec;
        std::vector<ambit::Tensor> BnbVec;
        std::vector<ambit::Tensor> BefVec;
        std::vector<ambit::Tensor> BefJKVec;
        std::vector<ambit::Tensor> RDVec;
        for (int i = 0; i < nthread; i++) {
            BmaVec.push_back(ambit::Tensor::build(tensor_type_, "Bma", {nthree_, nvirtual_}));
            BnaVec.push_back(ambit::Tensor::build(tensor_type_, "Bna", {nthree_, nvirtual_}));
            BmbVec.push_back(ambit::Tensor::build(tensor_type_, "Bmb", {nthree_, nvirtual_}));
            BnbVec.push_back(ambit::Tensor::build(tensor_type_, "Bnb", {nthree_, nvirtual_}));
            BefVec.push_back(ambit::Tensor::build(tensor_type_, "Bef", {nvirtual_, nvirtual_}));
            BefJKVec.push_back(ambit::Tensor::build(tensor_type_, "BefJK", {nvirtual_, nvirtual_}));
            RDVec.push_back(ambit::Tensor::build(tensor_type_, "RD", {nvirtual_, nvirtual_}));
        }
        bool ao_dsrg_check = options_.get_bool("AO_DSRG_MRPT2");
#pragma omp parallel for num_threads(num_threads_) schedule(dynamic)                               \
    reduction(+ : Ealpha, Ebeta, Emixed) shared(Ba, Bb)

        for (size_t m = 0; m < ncore_; ++m) {
            int thread = 0;
#ifdef _OPENMP
            thread = omp_get_thread_num();
#endif
            size_t ma = core_mos_[m];

            std::copy(&Ba.data()[m * dim], &Ba.data()[m * dim + dim],
                      BmaVec[thread].data().begin());
            // std::copy(&Bb.data()[m * dim], &Bb.data()[m * dim + dim],
            // BmbVec[thread].data().begin());
            std::copy(&Ba.data()[m * dim], &Ba.data()[m * dim + dim],
                      BmbVec[thread].data().begin());

            for (size_t n = m; n < ncore_; ++n) {
                size_t na = core_mos_[n];
                size_t nb = core_mos_[n];

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
                    double D = Fa_[ma] + Fa_[na] - Fa_[virt_mos_[i[0]]] - Fa_[virt_mos_[i[1]]];
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
                    double D = Fa_[ma] + Fb_[nb] - Fa_[virt_mos_[i[0]]] - Fb_[virt_mos_[i[1]]];
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
    size_t dim = nthree_ * nvirtual_;
    int nthread = 1;
#ifdef _OPENMP
    nthread = omp_get_max_threads();
#endif

    // Step 1:  Figure out the largest chunk of B_{me}^{Q} and B_{nf}^{Q} can be
    // stored in core.
    outfile->Printf("\n\n====Blocking information==========\n");
    size_t int_mem_int = (nthree_ * ncore_ * nvirtual_) * sizeof(double);
    size_t memory_input = Process::environment.get_memory() * 0.75;
    size_t num_block = int_mem_int / memory_input < 1 ? 1 : int_mem_int / memory_input;

    if (options_.get_int("CCVV_BATCH_NUMBER") != -1) {
        num_block = options_.get_int("CCVV_BATCH_NUMBER");
    }
    size_t block_size = ncore_ / num_block;

    if (block_size < 1) {
        outfile->Printf("\n\n  Block size is FUBAR.");
        outfile->Printf("\n  Block size is %d", block_size);
        throw PSIEXCEPTION("Block size is either 0 or negative.  Fix this problem");
    }
    if (num_block > ncore_) {
        outfile->Printf("\n  Number of blocks can not be larger than core_");
        throw PSIEXCEPTION("Number of blocks is larger than core.  Fix "
                           "num_block or check source code");
    }

    if (num_block >= 1) {
        outfile->Printf("\n  %lu / %lu = %lu", int_mem_int, memory_input,
                        int_mem_int / memory_input);
        outfile->Printf("\n  Block_size = %lu num_block = %lu", block_size, num_block);
    }

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
        BmaVec.push_back(ambit::Tensor::build(tensor_type_, "Bma", {nthree_, nvirtual_}));
        BnaVec.push_back(ambit::Tensor::build(tensor_type_, "Bna", {nthree_, nvirtual_}));
        BmbVec.push_back(ambit::Tensor::build(tensor_type_, "Bmb", {nthree_, nvirtual_}));
        BnbVec.push_back(ambit::Tensor::build(tensor_type_, "Bnb", {nthree_, nvirtual_}));
        BefVec.push_back(ambit::Tensor::build(tensor_type_, "Bef", {nvirtual_, nvirtual_}));
        BefJKVec.push_back(ambit::Tensor::build(tensor_type_, "BefJK", {nvirtual_, nvirtual_}));
        RDVec.push_back(ambit::Tensor::build(tensor_type_, "RDVec", {nvirtual_, nvirtual_}));
    }

    // Step 2:  Loop over memory allowed blocks of m and n
    // Get batch sizes and create vectors of mblock length
    for (size_t m_blocks = 0; m_blocks < num_block; m_blocks++) {
        std::vector<size_t> m_batch;
        // If core_ goes into num_block equally, all blocks are equal
        if (ncore_ % num_block == 0) {
            // Fill the mbatch from block_begin to block_end
            // This is done so I can pass a block to IntegralsAPI to read a
            // chunk
            m_batch.resize(block_size);
            // copy used to get correct indices for B.
            std::copy(core_mos_.begin() + (m_blocks * block_size),
                      core_mos_.begin() + ((m_blocks + 1) * block_size), m_batch.begin());
        } else {
            // If last_block is shorter or long, fill the rest
            size_t gimp_block_size =
                m_blocks == (num_block - 1) ? block_size + ncore_ % num_block : block_size;
            m_batch.resize(gimp_block_size);
            // std::iota(m_batch.begin(), m_batch.end(), m_blocks * (core_ /
            // num_block));
            std::copy(core_mos_.begin() + (m_blocks)*block_size,
                      core_mos_.begin() + (m_blocks)*block_size + gimp_block_size, m_batch.begin());
        }

        ambit::Tensor B = ints_->three_integral_block(aux_mos_, m_batch, virt_mos_);
        ambit::Tensor BmQe =
            ambit::Tensor::build(tensor_type_, "BmQE", {m_batch.size(), nthree_, nvirtual_});
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
            for (auto coremo : core_mos_) {
                outfile->Printf(" %d ", coremo);
            }
        }

        for (size_t n_blocks = 0; n_blocks <= m_blocks; n_blocks++) {
            std::vector<size_t> n_batch;
            // If core_ goes into num_block equally, all blocks are equal
            if (ncore_ % num_block == 0) {
                // Fill the mbatch from block_begin to block_end
                // This is done so I can pass a block to IntegralsAPI to read a chunk
                n_batch.resize(block_size);
                std::copy(core_mos_.begin() + n_blocks * block_size,
                          core_mos_.begin() + ((n_blocks + 1) * block_size), n_batch.begin());
            } else {
                // If last_block is longer, block_size + remainder
                size_t gimp_block_size =
                    n_blocks == (num_block - 1) ? block_size + ncore_ % num_block : block_size;
                n_batch.resize(gimp_block_size);
                std::copy(core_mos_.begin() + (n_blocks)*block_size,
                          core_mos_.begin() + (n_blocks * block_size) + gimp_block_size,
                          n_batch.begin());
            }
            ambit::Tensor BnQf =
                ambit::Tensor::build(tensor_type_, "BnQf", {n_batch.size(), nthree_, nvirtual_});
            if (n_blocks == m_blocks) {
                BnQf.copy(BmQe);
            } else {
                ambit::Tensor B = ints_->three_integral_block(aux_mos_, n_batch, virt_mos_);
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
            ForteTimer Core_Loop;
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
                    double D = Fa_[ma] + Fa_[na] - Fa_[virt_mos_[i[0]]] - Fa_[virt_mos_[i[1]]];
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
                    double D = Fa_[ma] + Fb_[nb] - Fa_[virt_mos_[i[0]]] - Fb_[virt_mos_[i[1]]];
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
                            m_blocks, n_blocks, Core_Loop.elapsed());
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
    SharedVector epsilon_rdocc(new Vector("EPS_RDOCC", ncore_));
    SharedVector epsilon_virtual(new Vector("EPS_VIRTUAL", nvirtual_));
    int core_count = 0;
    for (auto m : core_mos_) {
        epsilon_rdocc->set(core_count, Fa_[m]);
        core_count++;
    }
    int virtual_count = 0;
    for (auto e : virt_mos_) {
        epsilon_virtual->set(virtual_count, Fa_[e]);
        virtual_count++;
    }
    epsilon_rdocc->print();
    epsilon_virtual->print();

    AtomicOrbitalHelper ao_helper(Cwfn, epsilon_rdocc, epsilon_virtual, 1e-6, nactive_);
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
    DFTensor df_tensor(primary, auxiliary, Cwfn, ncore_, nvirtual_);
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
    size_t dim = nthree_ * ncore_;
    int nthread = 1;
#ifdef _OPENMP
    nthread = omp_get_max_threads();
#endif

    // Step 1:  Figure out the largest chunk of B_{me}^{Q} and B_{nf}^{Q} can be
    // stored in core.
    outfile->Printf("\n\n====Blocking information==========\n");
    size_t int_mem_int = (nthree_ * ncore_ * nvirtual_) * sizeof(double);
    size_t memory_input = Process::environment.get_memory() * 0.75;
    size_t num_block = int_mem_int / memory_input < 1 ? 1 : int_mem_int / memory_input;

    if (options_.get_int("CCVV_BATCH_NUMBER") != -1) {
        num_block = options_.get_int("CCVV_BATCH_NUMBER");
    }
    size_t block_size = nvirtual_ / num_block;

    if (block_size < 1) {
        outfile->Printf("\n\n  Block size is FUBAR.");
        outfile->Printf("\n  Block size is %d", block_size);
        throw PSIEXCEPTION("Block size is either 0 or negative.  Fix this problem");
    }
    if (num_block > nvirtual_) {
        outfile->Printf("\n  Number of blocks can not be larger than core_");
        throw PSIEXCEPTION("Number of blocks is larger than core.  Fix "
                           "num_block or check source code");
    }

    if (num_block >= 1) {
        outfile->Printf("\n  %lu / %lu = %lu", int_mem_int, memory_input,
                        int_mem_int / memory_input);
        outfile->Printf("\n  Block_size = %lu num_block = %lu", block_size, num_block);
    }

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
        BmaVec.push_back(ambit::Tensor::build(tensor_type_, "Bma", {nthree_, ncore_}));
        BnaVec.push_back(ambit::Tensor::build(tensor_type_, "Bna", {nthree_, ncore_}));
        BmbVec.push_back(ambit::Tensor::build(tensor_type_, "Bmb", {nthree_, ncore_}));
        BnbVec.push_back(ambit::Tensor::build(tensor_type_, "Bnb", {nthree_, ncore_}));
        BmnVec.push_back(ambit::Tensor::build(tensor_type_, "Bmn", {ncore_, ncore_}));
        BmnJKVec.push_back(ambit::Tensor::build(tensor_type_, "BmnJK", {ncore_, ncore_}));
        RDVec.push_back(ambit::Tensor::build(tensor_type_, "RDVec", {ncore_, ncore_}));
    }

    // Step 2:  Loop over memory allowed blocks of m and n
    // Get batch sizes and create vectors of mblock length
    for (size_t e_blocks = 0; e_blocks < num_block; e_blocks++) {
        std::vector<size_t> e_batch;
        // If core_ goes into num_block equally, all blocks are equal
        if (nvirtual_ % num_block == 0) {
            // Fill the mbatch from block_begin to block_end
            // This is done so I can pass a block to IntegralsAPI to read a
            // chunk
            e_batch.resize(block_size);
            // copy used to get correct indices for B.
            std::copy(virt_mos_.begin() + (e_blocks * block_size),
                      virt_mos_.begin() + ((e_blocks + 1) * block_size), e_batch.begin());
        } else {
            // If last_block is shorter or long, fill the rest
            size_t gimp_block_size =
                e_blocks == (num_block - 1) ? block_size + nvirtual_ % num_block : block_size;
            e_batch.resize(gimp_block_size);
            // std::iota(m_batch.begin(), m_batch.end(), m_blocks * (core_ /
            // num_block));
            std::copy(virt_mos_.begin() + (e_blocks)*block_size,
                      virt_mos_.begin() + (e_blocks)*block_size + gimp_block_size, e_batch.begin());
        }

        ambit::Tensor B = ints_->three_integral_block(aux_mos_, e_batch, core_mos_);
        ambit::Tensor BeQm =
            ambit::Tensor::build(tensor_type_, "BmQE", {e_batch.size(), nthree_, ncore_});
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
            for (auto virtualmo : virt_mos_) {
                outfile->Printf(" %d ", virtualmo);
            }
        }

        for (size_t f_blocks = 0; f_blocks <= e_blocks; f_blocks++) {
            std::vector<size_t> f_batch;
            // If core_ goes into num_block equally, all blocks are equal
            if (nvirtual_ % num_block == 0) {
                // Fill the mbatch from block_begin to block_end
                // This is done so I can pass a block to IntegralsAPI to read a
                // chunk
                f_batch.resize(block_size);
                std::copy(virt_mos_.begin() + f_blocks * block_size,
                          virt_mos_.begin() + ((f_blocks + 1) * block_size), f_batch.begin());
            } else {
                // If last_block is longer, block_size + remainder
                size_t gimp_block_size =
                    f_blocks == (num_block - 1) ? block_size + nvirtual_ % num_block : block_size;
                f_batch.resize(gimp_block_size);
                std::copy(virt_mos_.begin() + (f_blocks)*block_size,
                          virt_mos_.begin() + (f_blocks * block_size) + gimp_block_size,
                          f_batch.begin());
            }
            ambit::Tensor BfQn =
                ambit::Tensor::build(tensor_type_, "BnQf", {f_batch.size(), nthree_, ncore_});
            if (f_blocks == e_blocks) {
                BfQn.copy(BeQm);
            } else {
                ambit::Tensor B = ints_->three_integral_block(aux_mos_, f_batch, core_mos_);
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
            ForteTimer Virtual_loop;
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
                // Since loop over mn is collapsed, need to use fancy offset tricks
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
                    double D = Fa_[core_mos_[i[0]]] + Fa_[core_mos_[i[1]]] - Fa_[ea] - Fa_[fa];
                    value = dsrg_source_->compute_renormalized_denominator(D) *
                            (1.0 + dsrg_source_->compute_renormalized(D));
                });
                Ealpha += factor * 1.0 * BmnJKVec[thread]("mn") * RDVec[thread]("mn");

                // alpha-beta
                BmnVec[thread]("mN") = BmaVec[thread]("gm") * BnbVec[thread]("gN");
                BmnJKVec[thread]("mN") = BmnVec[thread]("mN") * BmnVec[thread]("mN");
                RDVec[thread].iterate([&](const std::vector<size_t>& i, double& value) {
                    double D = Fa_[core_mos_[i[0]]] + Fa_[core_mos_[i[1]]] - Fa_[ea] - Fa_[fb];
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
                                e_blocks, f_blocks, Virtual_loop.elapsed());
        }
    }
    // return (Ealpha + Ebeta + Emixed);
    return (Ealpha + Ebeta + Emixed);
}
double THREE_DSRG_MRPT2::E_VT2_2_core() {
    double E2_core = 0.0;
    BlockedTensor T2ccvv = BTF_->build(tensor_type_, "T2ccvv", spin_cases({"ccvv"}));
    BlockedTensor v = BTF_->build(tensor_type_, "Vccvv", spin_cases({"ccvv"}));

    BlockedTensor ThreeIntegral = BTF_->build(tensor_type_, "ThreeInt", {"Lph", "LPH"});
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

    /// This block of code assumes that ThreeIntegral are not stored as a member variable.
    /// Requires the reading from aptei_block which makes code
    ambit::Tensor Gamma1_aa = Gamma1_.block("aa");
    ambit::Tensor Gamma1_AA = Gamma1_.block("AA");

    std::vector<ambit::Tensor> Bm_Qe;
    std::vector<ambit::Tensor> Bm_Qf;

    std::vector<ambit::Tensor> Vefu;
    std::vector<ambit::Tensor> Tefv;
    std::vector<ambit::Tensor> tempTaa;
    std::vector<ambit::Tensor> tempTAA;

    ForteTimer ccvaTimer;
    for (int thread = 0; thread < nthread; thread++) {
        Bm_Qe.push_back(ambit::Tensor::build(tensor_type_, "BemQ", {nthree_, nvirtual_}));
        Bm_Qf.push_back(ambit::Tensor::build(tensor_type_, "Bmq", {nthree_, nvirtual_}));

        Vefu.push_back(
            ambit::Tensor::build(tensor_type_, "muJK", {nvirtual_, nvirtual_, nactive_}));
        Tefv.push_back(ambit::Tensor::build(tensor_type_, "T2", {nvirtual_, nvirtual_, nactive_}));

        tempTaa.push_back(ambit::Tensor::build(tensor_type_, "TEMPaa", {nactive_, nactive_}));
        tempTAA.push_back(ambit::Tensor::build(tensor_type_, "TEMPAA", {nactive_, nactive_}));
    }
    // ambit::Tensor BemQ = ints_->three_integral_block(naux,  acore_mos_,
    // avirt_mos_);
    // ambit::Tensor BeuQ = ints_->three_integral_block(naux,  aactv_mos_,
    // avirt_mos_);

    // Loop over e and f to compute V

    ambit::Tensor BeuQ = ints_->three_integral_block(aux_mos_, virt_mos_, actv_mos_);

// std::vector<double>& BemQ_data = BemQ.data();

// I think this loop is typically too small to allow efficient use of
// OpenMP.  Should probably test this assumption.
#pragma omp parallel for num_threads(num_threads_)
    for (size_t m = 0; m < ncore_; m++) {
        int thread = 0;
#ifdef _OPENMP
        thread = omp_get_thread_num();
#endif
        size_t ma = core_mos_[m];

// V[efu]_m = B_{em}^Q * B_{fu}^Q - B_{eu}^Q B_{fm}^Q
// V[efu]_m = V[efmu] + V[efmu] * exp[efmu]
// T2["mvef"] = V["mvef"] * D["mvef"]
// temp["uv"] = V * T2
#pragma omp critical
        { Bm_Qe[thread] = ints_->three_integral_block_two_index(aux_mos_, ma, virt_mos_); }

        Vefu[thread]("e, f, u") = Bm_Qe[thread]("Q, e") * BeuQ("Q, f, u");
        Vefu[thread]("e, f, u") -= BeuQ("Q, e, u") * Bm_Qe[thread]("Q, f");

        // E = V["efmu"] (1 + Exp(-s * D^{ef}_{mu}) * V^{mv}_{ef} *
        // Denom^{mv}_{ef}
        Tefv[thread].data() = Vefu[thread].data();

        std::vector<double>& T_mv_data = Tefv[thread].data();
        Vefu[thread].iterate([&](const std::vector<size_t>& i, double& value) {
            double Exp =
                Fa_[virt_mos_[i[0]]] + Fa_[virt_mos_[i[1]]] - Fa_[actv_mos_[i[2]]] - Fa_[ma];
            double D = -1.0 * (Fa_[virt_mos_[i[0]]] + Fa_[virt_mos_[i[1]]] - Fa_[actv_mos_[i[2]]] -
                               Fa_[ma]);
            value = value + value * dsrg_source_->compute_renormalized(Exp);
            T_mv_data[i[0] * nvirtual_ * nactive_ + i[1] * nactive_ + i[2]] *=
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
                Fa_[virt_mos_[i[0]]] + Fb_[virt_mos_[i[1]]] - Fa_[actv_mos_[i[2]]] - Fb_[ma];
            double D = -1.0 * (Fa_[virt_mos_[i[0]]] + Fb_[virt_mos_[i[1]]] - Fa_[actv_mos_[i[2]]] -
                               Fb_[ma]);
            value = value + value * dsrg_source_->compute_renormalized(Exp);
            T_mv_data[i[0] * nvirtual_ * nactive_ + i[1] * nactive_ + i[2]] *=
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
                Fa_[virt_mos_[i[0]]] + Fb_[virt_mos_[i[1]]] - Fb_[actv_mos_[i[2]]] - Fb_[ma];
            double D = -1.0 * (Fa_[virt_mos_[i[0]]] + Fa_[virt_mos_[i[1]]] - Fb_[actv_mos_[i[2]]] -
                               Fb_[ma]);
            value = value + value * dsrg_source_->compute_renormalized(Exp);
            T_mv_data[i[0] * nvirtual_ * nactive_ + i[1] * nactive_ + i[2]] *=
                dsrg_source_->compute_renormalized_denominator(D);
        });

        tempTaa[thread]("u,v") += 0.5 * Vefu[thread]("e, f, u") * Tefv[thread]("e, f, v");
    }

    ambit::Tensor tempTAA_all =
        ambit::Tensor::build(tensor_type_, "tempTAA_all", {nactive_, nactive_});
    ambit::Tensor tempTaa_all =
        ambit::Tensor::build(tensor_type_, "tempTaa_all", {nactive_, nactive_});
    for (int thread = 0; thread < nthread; thread++) {
        tempTAA_all("v, u") += tempTAA[thread]("v, u");
        tempTaa_all("v, u") += tempTaa[thread]("v, u");
    }

    Eacvv += tempTAA_all("v,u") * Gamma1_AA("v,u");
    Eacvv += tempTaa_all("v,u") * Gamma1_aa("v,u");

    if (print_ > 0) {
        outfile->Printf("\n\n  CAVV computation takes %8.8f", ccvaTimer.elapsed());
    }

    std::vector<ambit::Tensor> Bm_vQ;
    std::vector<ambit::Tensor> Bn_eQ;
    std::vector<ambit::Tensor> Bm_eQ;
    std::vector<ambit::Tensor> Bn_vQ;

    std::vector<ambit::Tensor> V_eu;
    std::vector<ambit::Tensor> T_ev;
    std::vector<ambit::Tensor> tempTaa_e;
    std::vector<ambit::Tensor> tempTAA_e;

    ambit::Tensor BmvQ = ints_->three_integral_block(aux_mos_, core_mos_, actv_mos_);
    ambit::Tensor BmvQ_swapped =
        ambit::Tensor::build(tensor_type_, "Bm_vQ", {ncore_, nthree_, nactive_});
    BmvQ_swapped("m, Q, u") = BmvQ("Q, m, u");
    ForteTimer cavvTimer;
    for (int thread = 0; thread < nthread; thread++) {
        Bm_vQ.push_back(ambit::Tensor::build(tensor_type_, "BemQ", {nthree_, nactive_}));
        Bn_eQ.push_back(ambit::Tensor::build(tensor_type_, "Bf_uQ", {nthree_, nvirtual_}));
        Bm_eQ.push_back(ambit::Tensor::build(tensor_type_, "Bmq", {nthree_, nvirtual_}));
        Bn_vQ.push_back(ambit::Tensor::build(tensor_type_, "Bmq", {nthree_, nactive_}));

        V_eu.push_back(ambit::Tensor::build(tensor_type_, "muJK", {nvirtual_, nactive_}));
        T_ev.push_back(ambit::Tensor::build(tensor_type_, "T2", {nvirtual_, nactive_}));

        tempTaa_e.push_back(ambit::Tensor::build(tensor_type_, "TEMPaa", {nactive_, nactive_}));
        tempTAA_e.push_back(ambit::Tensor::build(tensor_type_, "TEMPAA", {nactive_, nactive_}));
    }
    ambit::Tensor Eta1_aa = Eta1_.block("aa");
    ambit::Tensor Eta1_AA = Eta1_.block("AA");

#pragma omp parallel for num_threads(num_threads_)
    for (size_t m = 0; m < ncore_; ++m) {
        size_t ma = core_mos_[m];
        size_t mb = core_mos_[m];
        int thread = 0;
#ifdef _OPENMP
        thread = omp_get_thread_num();
#endif

#pragma omp critical
        { Bm_eQ[thread] = ints_->three_integral_block_two_index(aux_mos_, ma, virt_mos_); }
        std::copy(&BmvQ_swapped.data()[m * nthree_ * nactive_],
                  &BmvQ_swapped.data()[m * nthree_ * nactive_ + nthree_ * nactive_],
                  Bm_vQ[thread].data().begin());

        for (size_t n = 0; n < ncore_; ++n) {
            // alpha-aplha
            size_t na = core_mos_[n];
            size_t nb = core_mos_[n];

            std::copy(&BmvQ_swapped.data()[n * nthree_ * nactive_],
                      &BmvQ_swapped.data()[n * nthree_ * nactive_ + nthree_ * nactive_],
                      Bn_vQ[thread].data().begin());
//    Bn_vQ[thread].iterate([&](const std::vector<size_t>& i,double& value){
//        value = BmvQ_data[i[0] * core_ * active_ + n * active_ + i[1] ];
//    });
#pragma omp critical
            { Bn_eQ[thread] = ints_->three_integral_block_two_index(aux_mos_, na, virt_mos_); }

            // B_{mv}^{Q} * B_{ne}^{Q} - B_{me}^Q * B_{nv}
            V_eu[thread]("e, u") = Bm_vQ[thread]("Q, u") * Bn_eQ[thread]("Q, e");
            V_eu[thread]("e, u") -= Bm_eQ[thread]("Q, e") * Bn_vQ[thread]("Q, u");
            // E = V["efmu"] (1 + Exp(-s * D^{ef}_{mu}) * V^{mv}_{ef} *
            // Denom^{mv}_{ef}
            T_ev[thread].data() = V_eu[thread].data();

            V_eu[thread].iterate([&](const std::vector<size_t>& i, double& value) {
                double Exp = Fa_[actv_mos_[i[1]]] + Fa_[virt_mos_[i[0]]] - Fa_[ma] - Fa_[na];
                value = value + value * dsrg_source_->compute_renormalized(Exp);
                double D = Fa_[ma] + Fa_[na] - Fa_[actv_mos_[i[1]]] - Fa_[virt_mos_[i[0]]];
                T_ev[thread].data()[i[0] * nactive_ + i[1]] *=
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
                double Exp = Fa_[actv_mos_[i[1]]] + Fb_[virt_mos_[i[0]]] - Fa_[ma] - Fb_[nb];
                value = value + value * dsrg_source_->compute_renormalized(Exp);
                double D = Fa_[ma] + Fb_[nb] - Fa_[actv_mos_[i[1]]] - Fb_[virt_mos_[i[0]]];
                T_ev[thread].data()[i[0] * nactive_ + i[1]] *=
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
                double Exp = Fb_[mb] + Fb_[nb] - Fb_[actv_mos_[i[1]]] - Fb_[virt_mos_[i[0]]];
                value = value + value * dsrg_source_->compute_renormalized(Exp);
                double D = Fb_[mb] + Fb_[nb] - Fb_[actv_mos_[i[1]]] - Fb_[virt_mos_[i[0]]];
                T_ev[thread].data()[i[0] * nactive_ + i[1]] *=
                    dsrg_source_->compute_renormalized_denominator(D);
                ;
            });

            tempTAA_e[thread]("v,u") += 0.5 * V_eu[thread]("M,v") * T_ev[thread]("M,u");
            V_eu[thread].zero();
            T_ev[thread].zero();
        }
    }

    tempTAA_all = ambit::Tensor::build(tensor_type_, "tempTAA_all", {nactive_, nactive_});
    tempTaa_all = ambit::Tensor::build(tensor_type_, "tempTaa_all", {nactive_, nactive_});
    for (int thread = 0; thread < nthread; thread++) {
        tempTAA_all("u, v") += tempTAA_e[thread]("u,v");
        tempTaa_all("u, v") += tempTaa_e[thread]("u,v");
    }
    Eccva += tempTaa_all("vu") * Eta1_aa("uv");
    Eccva += tempTAA_all("VU") * Eta1_AA("UV");
    if (print_ > 0) {
        outfile->Printf("\n\n  CCVA takes %8.8f", cavvTimer.elapsed());
    }

    return (Eacvv + Eccva);
}

void THREE_DSRG_MRPT2::form_Hbar() {
    /// Note: if internal amplitudes are included, this function will NOT be correct.
    print_h2("Form DSRG-PT2 Transformed Hamiltonian");

    // initialize Hbar2 (Hbar1 is initialized in the end of startup function)
    ForteTimer timer;
    outfile->Printf("\n    %-40s ... ", "Initalizing Hbar");
    BlockedTensor Bactv = compute_B_minimal({"aaaa", "aAaA", "AAAA"});
    Hbar2_["pqrs"] = Bactv["gpr"] * Bactv["gqs"];
    Hbar2_["pqrs"] -= Bactv["gps"] * Bactv["gqr"];
    Hbar2_["PQRS"] = Bactv["gPR"] * Bactv["gQS"];
    Hbar2_["PQRS"] -= Bactv["gPS"] * Bactv["gQR"];
    Hbar2_["qPsR"] = Bactv["gPR"] * Bactv["gqs"];
    outfile->Printf("Done. Timing: %10.3f s.", timer.elapsed());

    /**
     * Implementation Notes
     *
     * 1. If DiskDF is NOT used, we already have all integrals to perform reference relaxation.
     * 2. If DiskDF is used, we will use the stored V and T2 first.
     *    These integrals contain at least two active indices.
     *    In fact, the only two other blocks of V that need to be considered are vvac and avcc.
     *    We shall implement these two blocks manually.
     * IMPORTANT: I (York) use spin adapted form for the last two cases!!!
     **/

    timer.reset();
    outfile->Printf("\n    %-40s ... ", "Computing all-active Hbar");
    BlockedTensor C1 = BTF_->build(tensor_type_, "C1", spin_cases({"aa"}));
    BlockedTensor C2 = BTF_->build(tensor_type_, "C2", spin_cases({"aaaa"}));
    H1_T1_C1(F_, T1_, 0.5, C1);
    H1_T2_C1(F_, T2_, 0.5, C1);
    H2_T1_C1(V_, T1_, 0.5, C1);
    H2_T2_C1(V_, T2_, 0.5, C1);
    H1_T2_C2(F_, T2_, 0.5, C2);
    H2_T1_C2(V_, T1_, 0.5, C2);
    H2_T2_C2(V_, T2_, 0.5, C2);

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
    outfile->Printf("Done. Timing: %10.3f s.", timer.elapsed());

    if (integral_type_ == DiskDF) {
        /**
         * AVCC and VACC Blocks of APTEI
         *
         * Spin integrated form:
         * [V, T]["zw"] <- -0.5 * V'["zemn"] * T["mnwe"] - V'["zEmN"] * T["mNwE"]
         * [V, T]["ZW"] <- -0.5 * V'["ZEMN"] * T["MNWE"] - V'["eZmN"] * T["mNeW"]
         * where V' = V * (1 + exp(-s * D * D)) and T = V * (1 - exp(-s * D * D)) / D.
         * V is expressed using DF, for example, V["zEmN"] = B(L|zm) * B(L|en).
         *
         * Assume spin Ms = 0 => V["zemn"] = V["ZEMN"] = V["zEmN"] - V["zEnM"].
         * Then it is easy to show:
         * [V, T]["zw"] <- -2 * V''(zn|em) * T''(nw|me) + V''(zn|em) * T''(mw|ne)  (*)
         * [V, T]["ZW"] <- -2 * V''(zn|em) * T''(nw|me) + V''(zn|em) * T''(mw|ne)  (*)
         * where V'' and T'' are NOT antisymmetrized, e.g., V''(zn|em) = V'["zEnM"].
         *
         * We will implement the spin adapted form, i.e., Eq. (*),
         * because B is formed using only Ca (thus Cb is ignored throughout).
         *
         * 1. We will batch over w and z, assuming two tensors of size c * c * v can be stored.
         *    This assumption is almost always reasonable, for example, v = 2500, c = 500.
         * 2. In the code, B1 refers to B(L|zm) for a fixed index "z";
         *    B2 refers to B(L|mw) (used to form T'') for a fixed index "w".
         *    Since L * v * c can potentially large, we load B(L|en) in batches of e.
         *    B3 refers to a vector (size = num_threads) of B(L|en) for a fixed index "e".
         * 3. For convenience, we store T'' for a given "w" as {nvirtual_, ncore_, ncore_}.
         * 4. Remember to include Hermitian adjoint of [V, T] to Hbar1!
         **/

        timer.reset();
        outfile->Printf("\n    %-40s ... ", "Computing DISKDF Hbar C");

        size_t nc2 = ncore_ * ncore_;
        ambit::Tensor V, T, B1, B2;
        V = ambit::Tensor::build(tensor_type_, "V_z", {nvirtual_, ncore_, ncore_});
        T = ambit::Tensor::build(tensor_type_, "T_w", {nvirtual_, ncore_, ncore_});

        int nthread = 1;
#ifdef _OPENMP
        nthread = num_threads_;
#endif

        std::vector<ambit::Tensor> B3, temp;
        for (int thread = 0; thread < nthread; ++thread) {
            B3.push_back(ambit::Tensor());
            temp.push_back(ambit::Tensor::build(tensor_type_, "V_ez", {ncore_, ncore_}));
        }

        /// => start from here <=
        for (size_t z = 0; z < nactive_; ++z) {
            size_t nz = actv_mos_[z];
            double Fz = Fa_[nz];

            /// V (alpha-beta equivalent) for a given "z"
            B1 = ints_->three_integral_block_two_index(aux_mos_, nz, core_mos_);

#pragma omp parallel for
            for (size_t e = 0; e < nvirtual_; ++e) {
                size_t ne = virt_mos_[e];
                int thread = 0; // SUPER IMPORTANT!!!
#ifdef _OPENMP
                thread = omp_get_thread_num();
#endif
#pragma omp critical
                { B3[thread] = ints_->three_integral_block_two_index(aux_mos_, ne, core_mos_); }

                // compute V for a given "z" and "e"
                temp[thread]("nm") = B1("gn") * B3[thread]("gm");

                // copy temp to V
                size_t offset = e * nc2;
                std::copy(temp[thread].data().begin(), temp[thread].data().end(),
                          V.data().begin() + offset);
            }

            // scale V := V * (1 + e^{-s * D * D})
            // TODO: test if this needs to be parallelized
            V.iterate([&](const std::vector<size_t>& i, double& value) {
                double D = Fa_[virt_mos_[i[0]]] + Fz - Fa_[core_mos_[i[1]]] - Fa_[core_mos_[i[2]]];
                value += value * dsrg_source_->compute_renormalized(D);
            });

            /// => loop active index for T <=
            for (size_t w = 0; w < nactive_; ++w) {
                size_t nw = actv_mos_[w];
                double Fw = Fa_[nw];

                /// T (alpha-beta equivalent) for a given "w"
                B2 = ints_->three_integral_block_two_index(aux_mos_, nw, core_mos_);

#pragma omp parallel for
                for (size_t e = 0; e < nvirtual_; ++e) {
                    size_t ne = virt_mos_[e];
                    int thread = 0; // SUPER IMPORTANT!!!
#ifdef _OPENMP
                    thread = omp_get_thread_num();
#endif
#pragma omp critical
                    { B3[thread] = ints_->three_integral_block_two_index(aux_mos_, ne, core_mos_); }

                    // compute V for T for a given "w" and "e"
                    temp[thread]("nm") = B2("gn") * B3[thread]("gm");

                    // copy temp to T
                    size_t offset = e * nc2;
                    std::copy(temp[thread].data().begin(), temp[thread].data().end(),
                              T.data().begin() + offset);
                }

                // scale T := V * (1 - e^{-s * D * D}) / D
                // TODO: test if this needs to be parallelized
                T.iterate([&](const std::vector<size_t>& i, double& value) {
                    double D =
                        Fa_[core_mos_[i[1]]] + Fa_[core_mos_[i[2]]] - Fw - Fa_[virt_mos_[i[0]]];
                    value *= dsrg_source_->compute_renormalized_denominator(D);
                });

                /// contract V and T
                double hbar = 0.0;
                hbar += V("emn") * T("emn");       // J
                hbar -= 0.5 * V("emn") * T("enm"); // K
                Hbar1_.block("aa").data()[w * nactive_ + z] -= hbar;
                Hbar1_.block("aa").data()[z * nactive_ + w] -= hbar;
                Hbar1_.block("AA").data()[w * nactive_ + z] -= hbar;
                Hbar1_.block("AA").data()[z * nactive_ + w] -= hbar;
            }
        }
        outfile->Printf("Done. Timing: %10.3f s.", timer.elapsed());

        /**
         * VVAC and VVCA Blocks of APTEI
         *
         * Spin integrated form (similar to previous block):
         * [V, T]["zw"] <- 0.5 * <ef||wm>' * T<zm||ef> + <eF||wM>' * T<eM||eF>
         * [V, T]["ZW"] <- 0.5 * <EF||WM>' * T<ZM||EF> + <eF||mW>' * T<mZ||eF>
         * where <ef||wm>' = <ef||wm> * (1 + exp(-s * D * D)),
         * and T<zm||ef> = <zm||ef> * (1 - exp(-s * D * D)) / D.
         * Note, <eF||wM>' = <eF|wM>' because of the spin.
         *
         * Assume spin Ms = 0 => <ef||wm> = <EF||WM> = <eF|wM> - <eF|mW> = (ew|fm) - (em|fw).
         * Then it is easy to show:
         * [V, T]["zw"] <- 2 * (ew|fm)'' * T(ze|mf)'' - (ew|fm)'' * T(zf|me)''  (*)
         * [V, T]["ZW"] <- 2 * (ew|fm)'' * T(ze|mf)'' - (ew|fm)'' * T(zf|me)''  (*)
         * where (ew|fm)'' and T(ze|mf)'' are NOT antisymmetrized, e.g., (ew|fm)'' = <eF|wM>'.
         *
         * We will implement the spin adapted form, i.e., Eq. (*),
         * because B is formed using only Ca (thus Cb is ignored throughout).
         *
         * 1. We will batch over w and z, assuming two tensors of size v * v * c can be stored.
         *    This assumption is usually reasonable, for example, v = 2000, c = 500.
         * 2. In the code, B1 refers to B(L|ew) for a fixed index "w";
         *    B2 refers to B(L|ze) (used to form T'') for a fixed index "z".
         *    Since L * v * c can potentially large, we load B(L|fm) in batches of m:
         *    B3 refers to a vector (size = num_threads) of B(L|fm) for a fixed index "m".
         * 3. For convenience, we store T'' for a given "z" as {ncore_, nvirtual_, nvirtual_}.
         * 4. Remember to include Hermitian adjoint of [V, T] to Hbar1!
         **/

        timer.reset();
        outfile->Printf("\n    %-40s ... ", "Computing DISKDF Hbar V");

        size_t nv2 = nvirtual_ * nvirtual_;
        V = ambit::Tensor::build(tensor_type_, "V_w", {ncore_, nvirtual_, nvirtual_});
        T = ambit::Tensor::build(tensor_type_, "T_z", {ncore_, nvirtual_, nvirtual_});

        for (int thread = 0; thread < nthread; ++thread) {
            temp[thread] = ambit::Tensor::build(tensor_type_, "V_wm", {nvirtual_, nvirtual_});
        }

        for (size_t w = 0; w < nactive_; ++w) {
            size_t nw = actv_mos_[w];
            double Fw = Fa_[nw];

            /// compute (ew|fm) = B(L|ew) * B(L|fm) for a given "w"
            B1 = ints_->three_integral_block_two_index(aux_mos_, nw, virt_mos_);

#pragma omp parallel for
            for (size_t m = 0; m < ncore_; ++m) {
                size_t nm = core_mos_[m];
                int thread = 0; // SUPER IMPORTANT!!!
#ifdef _OPENMP
                thread = omp_get_thread_num();
#endif
#pragma omp critical
                { B3[thread] = ints_->three_integral_block_two_index(aux_mos_, nm, virt_mos_); }

                // compute V for a given "w" and "m"
                temp[thread]("ef") = B1("ge") * B3[thread]("gf");

                // copy temp to V
                size_t offset = m * nv2;
                std::copy(temp[thread].data().begin(), temp[thread].data().end(),
                          V.data().begin() + offset);
            }

            // scale V := V * (1 + e^{-s * D * D})
            // TODO: test if this needs to be parallelized
            V.iterate([&](const std::vector<size_t>& i, double& value) {
                double D = Fa_[virt_mos_[i[1]]] + Fa_[virt_mos_[i[2]]] - Fa_[core_mos_[i[0]]] - Fw;
                value += value * dsrg_source_->compute_renormalized(D);
            });

            for (size_t z = 0; z < nactive_; ++z) {
                size_t nz = actv_mos_[z];
                double Fz = Fa_[nz];

                /// compute (ze|mf) = B(L|ze) * B(L|mf) for T for a given "z"
                B2 = ints_->three_integral_block_two_index(aux_mos_, nz, virt_mos_);

#pragma omp parallel for
                for (size_t m = 0; m < ncore_; ++m) {
                    size_t nm = core_mos_[m];
                    int thread = 0; // SUPER IMPORTANT!!!
#ifdef _OPENMP
                    thread = omp_get_thread_num();
#endif
#pragma omp critical
                    { B3[thread] = ints_->three_integral_block_two_index(aux_mos_, nm, virt_mos_); }

                    // compute T for a given "z" and "m"
                    temp[thread]("ef") = B2("ge") * B3[thread]("gf");

                    // copy temp to T
                    size_t offset = m * nv2;
                    std::copy(temp[thread].data().begin(), temp[thread].data().end(),
                              T.data().begin() + offset);
                }

                // scale T := V * (1 - e^{-s * D * D}) / D
                // TODO: test if this needs to be parallelized
                T.iterate([&](const std::vector<size_t>& i, double& value) {
                    double D =
                        Fa_[core_mos_[i[0]]] + Fz - Fa_[virt_mos_[i[1]]] - Fa_[virt_mos_[i[2]]];
                    value *= dsrg_source_->compute_renormalized_denominator(D);
                });

                /// contract V and T
                double hbar = 0.0;
                hbar += V("mef") * T("mef");       // J
                hbar -= 0.5 * V("mef") * T("mfe"); // K
                Hbar1_.block("aa").data()[w * nactive_ + z] += hbar;
                Hbar1_.block("aa").data()[z * nactive_ + w] += hbar;
                Hbar1_.block("AA").data()[w * nactive_ + z] += hbar;
                Hbar1_.block("AA").data()[z * nactive_ + w] += hbar;
            }
        }
        outfile->Printf("Done. Timing: %10.3f s.", timer.elapsed());
    }

    if (options_.get_bool("FORM_HBAR3")) {
        BlockedTensor C3 = BTF_->build(tensor_type_, "C3", spin_cases({"aaaaaa"}));
        H2_T2_C3(V_, T2_, 0.5, C3, true);

        Hbar3_["uvwxyz"] += C3["uvwxyz"];
        Hbar3_["uvwxyz"] += C3["xyzuvw"];
        Hbar3_["uvWxyZ"] += C3["uvWxyZ"];
        Hbar3_["uvWxyZ"] += C3["xyZuvW"];
        Hbar3_["uVWxYZ"] += C3["uVWxYZ"];
        Hbar3_["uVWxYZ"] += C3["xYZuVW"];
        Hbar3_["UVWXYZ"] += C3["UVWXYZ"];
        Hbar3_["UVWXYZ"] += C3["XYZUVW"];
    }
}

void THREE_DSRG_MRPT2::relax_reference_once() {

    auto fci_ints = compute_Heff_actv();

    std::vector<double> E_relaxed = relaxed_energy(fci_ints);

    if (options_["AVG_STATE"].size() == 0) {
        double Erelax = E_relaxed[0];

        // printing
        print_h2("CD/DF DSRG-MRPT2 Energy Summary");
        outfile->Printf("\n    %-37s = %22.15f", "CD/DF DSRG-MRPT2 Total Energy (fixed)  ",
                        Hbar0_ + Eref_);
        outfile->Printf("\n    %-37s = %22.15f", "CD/DF DSRG-MRPT2 Total Energy (relaxed)", Erelax);

        Process::environment.globals["PARTIALLY RELAXED ENERGY"] = Erelax;
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

        int nentry = options_["AVG_STATE"].size();
        for (int n = 0, offset = 0; n < nentry; ++n) {
            int irrep = options_["AVG_STATE"][n][0].to_integer();
            int multi = options_["AVG_STATE"][n][1].to_integer();
            int nstates = options_["AVG_STATE"][n][2].to_integer();

            for (int i = 0; i < nstates; ++i) {
                outfile->Printf("\n     %3d     %3s    %2d   %20.12f", multi,
                                irrep_symbol[irrep].c_str(), i, E_relaxed[i + offset]);
            }
            outfile->Printf("\n    %s", dash.c_str());

            offset += nstates;
        }

        Process::environment.globals["CURRENT ENERGY"] = E_relaxed[0];
    }
}

std::vector<double> THREE_DSRG_MRPT2::relaxed_energy(std::shared_ptr<FCIIntegrals> fci_ints) {

    // reference relaxation
    std::vector<double> Erelax;

    // check CAS_TYPE to decide diagonalization code
    if (options_.get_str("CAS_TYPE") == "CAS") {

        FCI_MO fci_mo(reference_wavefunction_, options_, ints_, mo_space_info_, fci_ints);
        fci_mo.set_localize_actv(false);
        double Eci = fci_mo.compute_energy();

        // test state specific or state average
        if (!multi_state_) {
            Erelax.push_back(Eci);
        } else {
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

    } else if (options_.get_str("CAS_TYPE") == "ACI") {

        // Only do ground state ACI for now
        AdaptiveCI aci(reference_wavefunction_, options_, ints_, mo_space_info_);
        aci.set_fci_ints(fci_ints);
        if (options_["ACI_RELAX_SIGMA"].has_changed()) {
            aci.update_sigma();
        }

        double relaxed_aci_en = aci.compute_energy();
        Erelax.push_back(relaxed_aci_en);

        // Compute relaxed NOs
        if( options_.get_bool("ACI_NO") ){
            aci.compute_nos();
        }

    } else {

        // common (SS and SA) setup of FCISolver
        int ntrial_per_root = options_.get_int("NTRIAL_PER_ROOT");
        Dimension active_dim = mo_space_info_->get_dimension("ACTIVE");
        std::shared_ptr<Molecule> molecule = Process::environment.molecule();
        double Enuc = molecule->nuclear_repulsion_energy(
            reference_wavefunction_->get_dipole_field_strength());
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
                nelec - 2 * mo_space_info_->size("FROZEN_DOCC") - 2 * core_mos_.size();
            auto na = (nelec_actv + twice_ms) / 2;
            auto nb = nelec_actv - na;

            // diagonalize the Hamiltonian
            FCISolver fcisolver(active_dim, core_mos_, actv_mos_, na, nb, multi,
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
                    nelec - 2 * mo_space_info_->size("FROZEN_DOCC") - 2 * core_mos_.size();
                auto na = (nelec_actv + ms) / 2;
                auto nb = nelec_actv - na;

                FCISolver fcisolver(active_dim, core_mos_, actv_mos_, na, nb, multi, irrep, ints_,
                                    mo_space_info_, ntrial_per_root, print_, options_);
                fcisolver.set_max_rdm_level(1);
                fcisolver.set_nroot(nstates);
                fcisolver.set_root(nstates - 1);
                fcisolver.set_fci_iterations(options_.get_int("FCI_MAXITER"));
                fcisolver.set_collapse_per_root(options_.get_int("DL_COLLAPSE_PER_ROOT"));
                fcisolver.set_subspace_per_root(options_.get_int("DL_SUBSPACE_PER_ROOT"));

                // set integrals manually
                fcisolver.use_user_integrals_and_restricted_docc(true);
                fcisolver.set_integral_pointer(fci_ints);

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

void THREE_DSRG_MRPT2::compute_Heff_2nd_coupling(double& H0, ambit::Tensor& H1a, ambit::Tensor& H1b,
                                                 ambit::Tensor& H2aa, ambit::Tensor& H2ab,
                                                 ambit::Tensor& H2bb, ambit::Tensor& H3aaa,
                                                 ambit::Tensor& H3aab, ambit::Tensor& H3abb,
                                                 ambit::Tensor& H3bbb) {
    // reset Tensors
    BlockedTensor H1 = BTF_->build(tensor_type_, "Heff1_2nd", spin_cases({"aa"}));
    H1a = H1.block("aa");
    H1b = H1.block("AA");

    BlockedTensor H2 = BTF_->build(tensor_type_, "Heff2_2nd", spin_cases({"aaaa"}));
    H2aa = H2.block("aaaa");
    H2ab = H2.block("aAaA");
    H2bb = H2.block("AAAA");

    BlockedTensor H3 = BTF_->build(tensor_type_, "Heff3_2nd", spin_cases({"aaaaaa"}));
    H3aaa = H3.block("aaaaaa");
    H3aab = H3.block("aaAaaA");
    H3abb = H3.block("aAAaAA");
    H3bbb = H3.block("AAAAAA");

    // de-normal-order amplitudes
    BlockedTensor T1eff = deGNO_Tamp(T1_, T2_, Gamma1_);

    // reset APTEI because it is renormalized
    std::vector<std::string> list_of_pphh_V = BTF_->generate_indices("vac", "pphh");
    if (integral_type_ != DiskDF) {
        V_["abij"] = ThreeIntegral_["gai"] * ThreeIntegral_["gbj"];
        V_["abij"] -= ThreeIntegral_["gaj"] * ThreeIntegral_["gbi"];

        V_["aBiJ"] = ThreeIntegral_["gai"] * ThreeIntegral_["gBJ"];

        V_["ABIJ"] = ThreeIntegral_["gAI"] * ThreeIntegral_["gBJ"];
        V_["ABIJ"] -= ThreeIntegral_["gAJ"] * ThreeIntegral_["gBI"];
    } else {
        V_ = compute_V_minimal(BTF_->spin_cases_avoid(list_of_pphh_V, 2), false);
    }

    // "effective" one-electron integrals: hbar^p_q = h^p_q + sum_m v^{mp}_{mq}
    BlockedTensor Hoei = BTF_->build(tensor_type_, "OEI", spin_cases({"ph"}));
    Hoei["pq"] = H_["pq"];
    Hoei["PQ"] = H_["PQ"];

    // add core contribution to Hoei
    ambit::Tensor temp, I;
    size_t nc = core_mos_.size();
    I = ambit::Tensor::build(tensor_type_, "I", std::vector<size_t>{nc, nc});
    I.iterate([&](const std::vector<size_t>& i, double& value) {
        if (i[0] == i[1]) {
            value = 1.0;
        }
    });

    temp = ints_->aptei_aa_block(core_mos_, actv_mos_, core_mos_, actv_mos_);
    Hoei.block("aa")("pq") += temp("mpnq") * I("mn");

    temp = ints_->aptei_ab_block(actv_mos_, core_mos_, actv_mos_, core_mos_);
    Hoei.block("aa")("pq") += temp("pmqn") * I("mn");

    temp = ints_->aptei_ab_block(core_mos_, actv_mos_, core_mos_, actv_mos_);
    Hoei.block("AA")("pq") += temp("mpnq") * I("mn");

    temp = ints_->aptei_bb_block(core_mos_, actv_mos_, core_mos_, actv_mos_);
    Hoei.block("AA")("pq") += temp("mpnq") * I("mn");

    I = ambit::Tensor::build(tensor_type_, "I", std::vector<size_t>{1, 1});
    I.data()[0] = 1.0;
    for (const size_t& m : core_mos_) {
        for (const std::string& block : {"ac", "va", "vc"}) {
            std::string block_beta =
                std::string(1, toupper(block[0])) + std::string(1, toupper(block[1]));
            std::vector<size_t>& mos1 = label_to_spacemo_[block[0]];
            std::vector<size_t>& mos2 = label_to_spacemo_[block[1]];

            temp = ints_->aptei_aa_block({m}, mos1, {m}, mos2);
            Hoei.block(block)("pq") += temp("mpnq") * I("mn");

            temp = ints_->aptei_ab_block(mos1, {m}, mos2, {m});
            Hoei.block(block)("pq") += temp("pmqn") * I("mn");

            temp = ints_->aptei_ab_block({m}, mos1, {m}, mos2);
            Hoei.block(block_beta)("pq") += temp("mpnq") * I("mn");

            temp = ints_->aptei_bb_block({m}, mos1, {m}, mos2);
            Hoei.block(block_beta)("pq") += temp("mpnq") * I("mn");
        }
    }

    // 1-body
    H1["uv"] = Hoei["uv"];
    H1["UV"] = Hoei["UV"];

    H1["vu"] += Hoei["eu"] * T1eff["ve"];
    H1["VU"] += Hoei["EU"] * T1eff["VE"];

    H1["vu"] -= Hoei["vm"] * T1eff["mu"];
    H1["VU"] -= Hoei["VM"] * T1eff["MU"];

    H1["vu"] += V_["avmu"] * T1eff["ma"];
    H1["vu"] += V_["vAuM"] * T1eff["MA"];
    H1["VU"] += V_["aVmU"] * T1eff["ma"];
    H1["VU"] += V_["AVMU"] * T1eff["MA"];

    H1["vu"] += Hoei["am"] * T2_["mvau"];
    H1["vu"] += Hoei["AM"] * T2_["vMuA"];
    H1["VU"] += Hoei["am"] * T2_["mVaU"];
    H1["VU"] += Hoei["AM"] * T2_["MVAU"];

    H1["vu"] += 0.5 * V_["abum"] * T2_["vmab"];
    H1["vu"] += V_["aBuM"] * T2_["vMaB"];
    H1["VU"] += 0.5 * V_["ABUM"] * T2_["VMAB"];
    H1["VU"] += V_["aBmU"] * T2_["mVaB"];

    H1["vu"] -= 0.5 * V_["avmn"] * T2_["mnau"];
    H1["vu"] -= V_["vAmN"] * T2_["mNuA"];
    H1["VU"] -= 0.5 * V_["AVMN"] * T2_["MNAU"];
    H1["VU"] -= V_["aVmN"] * T2_["mNaU"];

    if (integral_type_ == DiskDF) {
    }

    // 2-body
    H2["uvxy"] = 0.25 * V_["uvxy"];
    H2["uVxY"] = V_["uVxY"];
    H2["UVXY"] = 0.25 * V_["UVXY"];

    H2["xyuv"] += 0.5 * V_["eyuv"] * T1eff["xe"];
    H2["xYuV"] += V_["eYuV"] * T1eff["xe"];
    H2["xYuV"] += V_["xEuV"] * T1eff["YE"];
    H2["XYUV"] += 0.5 * V_["EYUV"] * T1eff["XE"];

    H2["xyuv"] += 0.5 * V_["xyvm"] * T1eff["mu"];
    H2["xYuV"] -= V_["xYmV"] * T1eff["mu"];
    H2["xYuV"] -= V_["xYuM"] * T1eff["MV"];
    H2["XYUV"] += 0.5 * V_["XYVM"] * T1eff["MU"];

    H2["xyuv"] += 0.5 * Hoei["eu"] * T2_["xyev"];
    H2["xYuV"] += Hoei["eu"] * T2_["xYeV"];
    H2["xYuV"] += Hoei["EV"] * T2_["xYuE"];
    H2["XYUV"] += 0.5 * Hoei["EU"] * T2_["XYEV"];

    H2["xyuv"] -= 0.5 * Hoei["xm"] * T2_["myuv"];
    H2["xYuV"] -= Hoei["xm"] * T2_["mYuV"];
    H2["xYuV"] -= Hoei["YM"] * T2_["xMuV"];
    H2["XYUV"] -= 0.5 * Hoei["XM"] * T2_["MYUV"];

    H2["xyuv"] -= V_["aymu"] * T2_["mxav"];
    H2["xyuv"] -= V_["yAuM"] * T2_["xMvA"];
    H2["xYuV"] -= V_["aYuM"] * T2_["xMaV"];
    H2["xYuV"] -= V_["xAmV"] * T2_["mYuA"];
    H2["xYuV"] += V_["axmu"] * T2_["mYaV"];
    H2["xYuV"] += V_["xAuM"] * T2_["MYAV"];
    H2["xYuV"] += V_["aYmV"] * T2_["mxau"];
    H2["xYuV"] += V_["AYMV"] * T2_["xMuA"];
    H2["XYUV"] -= V_["aYmU"] * T2_["mXaV"];
    H2["XYUV"] -= V_["AYMU"] * T2_["XMAV"];

    H2["xyuv"] += 0.125 * V_["xymn"] * T2_["mnuv"];
    H2["xYuV"] += V_["xYmN"] * T2_["mNuV"];
    H2["XYUV"] += 0.125 * V_["XYMN"] * T2_["MNUV"];

    H2["xyuv"] += 0.125 * V_["abuv"] * T2_["xyab"];
    H2["xYuV"] += V_["aBuV"] * T2_["xYaB"];
    H2["XYUV"] += 0.125 * V_["ABUV"] * T2_["XYAB"];

    // temp contract with D3
    H3["xyzuvw"] += 0.25 * V_["yzmu"] * T2_["mxvw"];
    H3["xyzuvw"] -= 0.25 * V_["ezuv"] * T2_["xyew"];

    H3["XYZUVW"] += 0.25 * V_["YZMU"] * T2_["MXVW"];
    H3["XYZUVW"] -= 0.25 * V_["EZUV"] * T2_["XYEW"];

    H3["xyZuvW"] += 0.5 * V_["yZmW"] * T2_["mxuv"];
    H3["xyZuvW"] += 0.5 * V_["xymu"] * T2_["mZvW"];
    H3["xyZuvW"] += V_["yZuM"] * T2_["xMvW"];

    H3["xyZuvW"] += 0.5 * V_["eZuW"] * T2_["xyev"];
    H3["xyZuvW"] += 0.5 * V_["eyuv"] * T2_["xZeW"];
    H3["xyZuvW"] -= V_["yEuW"] * T2_["xZvE"];

    H3["xYZuVW"] += 0.5 * V_["YZMV"] * T2_["xMuW"];
    H3["xYZuVW"] += 0.5 * V_["xZuM"] * T2_["MYVW"];
    H3["xYZuVW"] += V_["xZmV"] * T2_["mYuW"];

    H3["xYZuVW"] += 0.5 * V_["EZVW"] * T2_["xYuE"];
    H3["xYZuVW"] += 0.5 * V_["xEuV"] * T2_["YZEW"];
    H3["xYZuVW"] -= V_["eZuV"] * T2_["xYeW"];
}

void THREE_DSRG_MRPT2::de_normal_order() {
    // printing
    print_h2("De-Normal-Order the DSRG Transformed Hamiltonian");

    // compute scalar term
    ForteTimer t_scalar;
    std::string str = "Computing the scalar term   ...";
    outfile->Printf("\n    %-35s", str.c_str());
    double scalar0 = Eref_ + Hbar0_ - Enuc_ - Efrzc_;

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
    outfile->Printf("  Done. Timing %10.3f s", t_scalar.elapsed());

    // compute one-body term
    ForteTimer t_one;
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
    outfile->Printf("  Done. Timing %10.3f s", t_one.elapsed());
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
    double Etest =
        scalar_include_fc +
        molecule_->nuclear_repulsion_energy(reference_wavefunction_->get_dipole_field_strength());

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
    bool semi = check_semi_orbs();
    if (ignore_semicanonical_) {
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

            warnings_.push_back(std::make_tuple("Semicanonical orbital test", "Ignore test results",
                                                "Post an issue for advice"));
        }
        outfile->Printf("\n");
        semi = true;
    }

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
