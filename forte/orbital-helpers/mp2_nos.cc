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
#include <memory>

#include "ambit/blocked_tensor.h"

#include "psi4/libpsi4util/process.h"
#include "psi4/libpsio/psio.h"
#include "psi4/libpsio/psio.hpp"
#include "psi4/libmints/matrix.h"
#include "psi4/libmints/vector.h"
#include "psi4/libpsi4util/PsiOutStream.h"

#include "helpers/helpers.h"
#include "helpers/blockedtensorfactory.h"
#include "helpers/printing.h"
#include "helpers/timer.h"
#include "mp2_nos.h"

#ifdef _OPENMP
#include <omp.h>
#else
#define omp_get_max_threads() 1
#define omp_get_thread_num() 0
#define omp_get_num_threads() 1
#endif

using namespace ambit;

using namespace psi;

namespace forte {

MP2_NOS::MP2_NOS(std::shared_ptr<SCFInfo> scf_info, std::shared_ptr<ForteOptions> options,
                 std::shared_ptr<ForteIntegrals> ints, std::shared_ptr<MOSpaceInfo> mo_space_info)
    : OrbitalTransform(ints, mo_space_info), scf_info_(scf_info), options_(options) {}

psi::SharedMatrix MP2_NOS::get_Ua() { return Ua_; }
psi::SharedMatrix MP2_NOS::get_Ub() { return Ub_; }

void MP2_NOS::compute_transformation() {
    print_method_banner(
        {"Second-Order Moller-Plesset Natural Orbitals", "written by Francesco A. Evangelista"});

    // memory in bytes
    memory_ = psi::Process::environment.get_memory() * 0.9;
    outfile->Printf("\n    Memory in MB                  %8.1f", memory_ / 1024. / 1024.);
    outfile->Printf("\n    Number of threads             %8d", omp_get_max_threads());

    BlockedTensor::set_expert_mode(true);
    BlockedTensor::reset_mo_spaces();

    a_occ_mos_.clear();
    b_occ_mos_.clear();
    a_vir_mos_.clear();
    b_vir_mos_.clear();

    psi::Dimension ncmopi = mo_space_info_->dimension("CORRELATED");
    psi::Dimension frzcpi = mo_space_info_->dimension("FROZEN_DOCC");
    psi::Dimension frzvpi = mo_space_info_->dimension("FROZEN_UOCC");

    psi::Dimension nmopi = mo_space_info_->dimension("ALL");
    psi::Dimension doccpi = scf_info_->doccpi();
    psi::Dimension soccpi = scf_info_->soccpi();

    psi::Dimension corr_docc(doccpi);
    corr_docc -= frzcpi;

    psi::Dimension aoccpi = corr_docc + scf_info_->soccpi();
    psi::Dimension boccpi = corr_docc;
    psi::Dimension avirpi = ncmopi - aoccpi;
    psi::Dimension bvirpi = ncmopi - boccpi;

    int nirrep = ints_->nirrep();

    for (int h = 0, p = 0; h < nirrep; ++h) {
        for (int i = 0; i < corr_docc[h]; ++i, ++p) {
            a_occ_mos_.push_back(p);
            b_occ_mos_.push_back(p);
        }
        for (int i = 0; i < soccpi[h]; ++i, ++p) {
            a_occ_mos_.push_back(p);
            b_vir_mos_.push_back(p);
        }
        for (int a = 0; a < ncmopi[h] - corr_docc[h] - soccpi[h]; ++a, ++p) {
            a_vir_mos_.push_back(p);
            b_vir_mos_.push_back(p);
        }
    }

    naocc_ = a_occ_mos_.size();
    nbocc_ = b_occ_mos_.size();
    navir_ = a_vir_mos_.size();
    nbvir_ = b_vir_mos_.size();
    outfile->Printf("\n    Number of α occupied orbitals %8zu", naocc_);
    outfile->Printf("\n    Number of β occupied orbitals %8zu", nbocc_);
    outfile->Printf("\n    Number of α virtual orbitals  %8zu", navir_);
    outfile->Printf("\n    Number of β virtual orbitals  %8zu", nbvir_);

    // directly read orbital energies from psi4
    Fa_.clear();
    Fb_.clear();

    for (int h = 0; h < nirrep; ++h) {
        for (int i = 0; i < ncmopi[h]; ++i) {
            Fa_.push_back(scf_info_->epsilon_a()->get(h, i + frzcpi[h]));
            Fb_.push_back(scf_info_->epsilon_b()->get(h, i + frzcpi[h]));
        }
    }

    BlockedTensor::add_mo_space("o", "ijklmn", a_occ_mos_, AlphaSpin);
    BlockedTensor::add_mo_space("O", "IJKLMN", b_occ_mos_, BetaSpin);
    BlockedTensor::add_mo_space("v", "abcdef", a_vir_mos_, AlphaSpin);
    BlockedTensor::add_mo_space("V", "ABCDEF", b_vir_mos_, BetaSpin);
    BlockedTensor::add_composite_mo_space("i", "pqrstuvwxyz", {"o", "v"});
    BlockedTensor::add_composite_mo_space("I", "PQRSTUVWXYZ", {"O", "V"});

    BlockedTensor D1;
    auto ints_type = options_->get_str("INT_TYPE");
    if (ints_type == "DF" or ints_type == "DISKDF" or ints_type == "CHOLESKY") {
        naux_ = ints_->nthree();
        aux_mos_ = std::vector<size_t>(naux_);
        std::iota(aux_mos_.begin(), aux_mos_.end(), 0);
        BlockedTensor::add_mo_space("L", "g", aux_mos_, NoSpin);

        D1 = build_1rdm_df();
    } else {
        D1 = build_1rdm_conv();
    }

    // Copy the density matrix to matrix objects
    auto D1oo = tensor_to_matrix(D1.block("oo"), aoccpi);
    auto D1OO = tensor_to_matrix(D1.block("OO"), boccpi);
    auto D1vv = tensor_to_matrix(D1.block("vv"), avirpi);
    auto D1VV = tensor_to_matrix(D1.block("VV"), bvirpi);

    psi::Process::environment.arrays["MP2 1RDM OO ALPHA"] = D1oo;
    psi::Process::environment.arrays["MP2 1RDM VV ALPHA"] = D1vv;
    psi::Process::environment.arrays["MP2 1RDM OO BETA"] = D1OO;
    psi::Process::environment.arrays["MP2 1RDM VV BETA"] = D1VV;

    Matrix D1oo_evecs("D1oo_evecs", aoccpi, aoccpi);
    Matrix D1OO_evecs("D1OO_evecs", boccpi, boccpi);
    Matrix D1vv_evecs("D1vv_evecs", avirpi, avirpi);
    Matrix D1VV_evecs("D1VV_evecs", bvirpi, bvirpi);

    Vector D1oo_evals("D1oo_evals", aoccpi);
    Vector D1OO_evals("D1OO_evals", boccpi);
    Vector D1vv_evals("D1vv_evals", avirpi);
    Vector D1VV_evals("D1VV_evals", bvirpi);

    D1oo->diagonalize(D1oo_evecs, D1oo_evals, descending);
    D1vv->diagonalize(D1vv_evecs, D1vv_evals, descending);
    D1OO->diagonalize(D1OO_evecs, D1OO_evals, descending);
    D1VV->diagonalize(D1VV_evecs, D1VV_evals, descending);

    // Print natural orbitals
    if (options_->get_bool("NAT_ORBS_PRINT")) {
        D1oo_evals.print();
        D1vv_evals.print();
        D1OO_evals.print();
        D1VV_evals.print();
    }

    // This will suggest a restricted_docc and an active
    // Does not take in account frozen_docc
    if (options_->get_bool("NAT_ACT")) {
        std::vector<size_t> restricted_docc(nirrep);
        std::vector<size_t> active(nirrep);
        double occupied = options_->get_double("MP2NO_OCC_THRESHOLD");
        double virtual_orb = options_->get_double("MP2NO_VIR_THRESHOLD");
        outfile->Printf("\n Suggested Active Space \n");
        outfile->Printf("\n Occupied orbitals with an occupation less than %6.4f are active",
                        occupied);
        outfile->Printf("\n Virtual orbitals with an occupation greater than %6.4f are active",
                        virtual_orb);
        outfile->Printf("\n Remember, these are suggestions  :-)!\n");
        for (int h = 0; h < nirrep; ++h) {
            size_t restricted_docc_number = 0;
            size_t active_number = 0;
            for (int i = 0; i < aoccpi[h]; ++i) {
                if (D1oo_evals.get(h, i) < occupied) {
                    active_number++;
                    outfile->Printf("\n Irrep %d orbital %4d occupation: %8.6f Active occupied", h,
                                    i, D1oo_evals.get(h, i));
                } else {
                    restricted_docc_number++;
                }
            }
            for (int a = 0; a < avirpi[h]; ++a) {
                if (D1vv_evals.get(h, a) > virtual_orb) {
                    active_number++;
                    outfile->Printf("\n Irrep %d orbital %4d occupation: %8.6f Active virtual", h,
                                    a, D1vv_evals.get(h, a));
                }
            }
            active[h] = active_number;
            restricted_docc[h] = restricted_docc_number;
        }
        outfile->Printf("\n By occupation analysis, the restricted_docc should be\n");
        outfile->Printf("\n Restricted_docc = [ ");
        for (auto& rocc : restricted_docc) {
            outfile->Printf("%zu ", rocc);
        }
        outfile->Printf("]\n");
        outfile->Printf("\n By occupation analysis, the active orbitals should be\n");
        outfile->Printf("\n Active = [ ");
        for (auto& ract : active) {
            outfile->Printf("%zu ", ract);
        }
        outfile->Printf("]\n");
    }

    auto Ua = std::make_shared<psi::Matrix>("Ua", nmopi, nmopi);
    Ua->identity();

    Slice slice_occ_a(frzcpi, aoccpi + frzcpi);
    Slice slice_vir_a(aoccpi + frzcpi, ncmopi + frzcpi);
    Ua->set_block(slice_occ_a, D1oo_evecs);
    Ua->set_block(slice_vir_a, D1vv_evecs);

    auto Ub = std::make_shared<psi::Matrix>("Ub", nmopi, nmopi);
    Ub->identity();

    Slice slice_occ_b(frzcpi, boccpi + frzcpi);
    Slice slice_vir_b(boccpi + frzcpi, ncmopi + frzcpi);
    Ub->set_block(slice_occ_b, D1OO_evecs);
    Ub->set_block(slice_vir_b, D1VV_evecs);

    // Retransform the integrals in the new basis
    // TODO: this class should read this information (ints_->spin_restriction()) early and compute
    // only one set of MOs

    auto spin_restriction = ints_->spin_restriction();

    Ua_ = std::make_shared<psi::Matrix>("Ua", nmopi, nmopi);
    Ub_ = std::make_shared<psi::Matrix>("Ub", nmopi, nmopi);

    Ua_->copy(Ua->clone());
    if (spin_restriction == IntegralSpinRestriction::Restricted) {
        Ub_->copy(Ua->clone());
    } else {
        Ub_->copy(Ub->clone());
    }

    BlockedTensor::set_expert_mode(false);
    // Erase all mo_space information
    BlockedTensor::reset_mo_spaces();
}

ambit::BlockedTensor MP2_NOS::build_1rdm_conv() {
    // we assume restricted canonical orbitals!

    // build fock matrix
    BlockedTensor F = BlockedTensor::build(CoreTensor, "F", spin_cases({"ii"}));
    F.iterate([&](const std::vector<size_t>& i, const std::vector<SpinType>& spin, double& value) {
        if (spin[0] == AlphaSpin) {
            value = ints_->oei_a(i[0], i[1]);
            for (size_t a : a_occ_mos_) {
                value += ints_->aptei_aa(i[0], a, i[1], a);
            }
            for (size_t b : b_occ_mos_) {
                value += ints_->aptei_ab(i[0], b, i[1], b);
            }
        } else {
            value = ints_->oei_b(i[0], i[1]);
            for (size_t a : a_occ_mos_) {
                value += ints_->aptei_ab(a, i[0], a, i[1]);
            }
            for (size_t b : b_occ_mos_) {
                value += ints_->aptei_bb(i[0], b, i[1], b);
            }
        }
    });

    // TODO: test fock matrix

    // store diagonal fock matrix elements
    size_t ncmo = mo_space_info_->size("CORRELATED");
    Fa_ = std::vector<double>(ncmo);
    Fb_ = std::vector<double>(ncmo);

    F.iterate([&](const std::vector<size_t>& i, const std::vector<SpinType>& spin, double& value) {
        if (i[0] == i[1]) {
            if (spin[0] == AlphaSpin)
                Fa_[i[0]] = value;
            else
                Fb_[i[0]] = value;
        }
    });

    // fill anti-symmetrized two-electron integrals
    BlockedTensor V = BlockedTensor::build(CoreTensor, "V", spin_cases({"oovv"}));
    V.iterate([&](const std::vector<size_t>& i, const std::vector<SpinType>& spin, double& value) {
        if ((spin[0] == AlphaSpin) and (spin[1] == AlphaSpin))
            value = ints_->aptei_aa(i[0], i[1], i[2], i[3]);
        if ((spin[0] == AlphaSpin) and (spin[1] == BetaSpin))
            value = ints_->aptei_ab(i[0], i[1], i[2], i[3]);
        if ((spin[0] == BetaSpin) and (spin[1] == BetaSpin))
            value = ints_->aptei_bb(i[0], i[1], i[2], i[3]);
    });

    // build T2 amplitudes
    BlockedTensor T2 = BlockedTensor::build(CoreTensor, "T2", spin_cases({"oovv"}));
    T2["ijab"] = V["ijab"];
    T2["iJaB"] = V["iJaB"];
    T2["IJAB"] = V["IJAB"];

    T2.iterate([&](const std::vector<size_t>& i, const std::vector<SpinType>& spin, double& value) {
        if ((spin[0] == AlphaSpin) and (spin[1] == AlphaSpin))
            value /= Fa_[i[0]] + Fa_[i[1]] - Fa_[i[2]] - Fa_[i[3]];
        if ((spin[0] == AlphaSpin) and (spin[1] == BetaSpin))
            value /= Fa_[i[0]] + Fb_[i[1]] - Fa_[i[2]] - Fb_[i[3]];
        if ((spin[0] == BetaSpin) and (spin[1] == BetaSpin))
            value /= Fb_[i[0]] + Fb_[i[1]] - Fb_[i[2]] - Fb_[i[3]];
    });

    // compute MP2 energy
    double Eaa = 0.25 * T2["ijab"] * V["ijab"];
    double Eab = T2["iJaB"] * V["iJaB"];
    double Ebb = 0.25 * T2["IJAB"] * V["IJAB"];

    double mp2_corr_energy = Eaa + Eab + Ebb;
    double ref_energy = scf_info_->reference_energy();
    outfile->Printf("\n\n    SCF energy                            = %20.15f", ref_energy);
    outfile->Printf("\n    MP2 correlation energy (aa)           = %20.15f", Eaa);
    outfile->Printf("\n    MP2 correlation energy (ab)           = %20.15f", Eab);
    outfile->Printf("\n    MP2 correlation energy (bb)           = %20.15f", Ebb);
    outfile->Printf("\n    MP2 correlation energy                = %20.15f", mp2_corr_energy);
    outfile->Printf("\n  * MP2 total energy                      = %20.15f\n\n",
                    ref_energy + mp2_corr_energy);
    psi::Process::environment.globals["MP2 CORRELATION ENERGY"] = mp2_corr_energy;

    // build 1RDM
    BlockedTensor D1 = BlockedTensor::build(CoreTensor, "D1", spin_cases({"oo", "vv"}));
    D1.block("oo").iterate(
        [&](const std::vector<size_t>& i, double& value) { value = i[0] == i[1] ? 1.0 : 0.0; });
    D1.block("OO").iterate(
        [&](const std::vector<size_t>& i, double& value) { value = i[0] == i[1] ? 1.0 : 0.0; });

    D1["ab"] += 0.5 * T2["ijbc"] * T2["ijac"];
    D1["ab"] += 1.0 * T2["iJbC"] * T2["iJaC"];

    D1["AB"] += 0.5 * T2["IJCB"] * T2["IJCA"];
    D1["AB"] += 1.0 * T2["iJcB"] * T2["iJcA"];

    D1["ij"] -= 0.5 * T2["ikab"] * T2["jkab"];
    D1["ij"] -= 1.0 * T2["iKaB"] * T2["jKaB"];

    D1["IJ"] -= 0.5 * T2["IKAB"] * T2["JKAB"];
    D1["IJ"] -= 1.0 * T2["kIaB"] * T2["kJaB"];

    return D1;
}

ambit::BlockedTensor MP2_NOS::build_1rdm_df() {
    // 1RDM to be returned
    BlockedTensor D1 = BlockedTensor::build(CoreTensor, "D1", spin_cases({"oo", "vv"}));
    D1.block("oo").iterate(
        [&](const std::vector<size_t>& i, double& value) { value = i[0] == i[1] ? 1.0 : 0.0; });
    D1.block("OO").iterate(
        [&](const std::vector<size_t>& i, double& value) { value = i[0] == i[1] ? 1.0 : 0.0; });

    if (ints_->spin_restriction() == IntegralSpinRestriction::Restricted) {
        compute_df_rmp2_1rdm_vv(D1);
        compute_df_rmp2_1rdm_oo(D1);
    } else {
        compute_df_ump2_1rdm_vv(D1);
        compute_df_ump2_1rdm_oo(D1);
    }

    return D1;
}

void MP2_NOS::compute_df_ump2_1rdm_vv(ambit::BlockedTensor& D1) {
    timer tvv("DF-UMP2 1RDM VV");
    int nthreads = omp_get_max_threads();

    auto na_Qv = naux_ * navir_;
    auto nb_Qv = naux_ * nbvir_;

    // batches of occupied indices
    std::vector<std::vector<size_t>> batch_aocc, batch_bocc;

    // test memory
    size_t memory_min = 4 * nthreads * nbvir_ * nbvir_;
    if ((memory_min + na_Qv + nb_Qv) * sizeof(double) > memory_) {
        outfile->Printf("\n  Error: Not enough memory for DF-UMP2 (VV).");
        outfile->Printf(" Need at least %zu Bytes more!",
                        (memory_min + na_Qv + nb_Qv) * sizeof(double) - memory_);
        throw std::runtime_error("Not enough memory to run DF-MP2. Please check output.");
    } else {
        size_t max_occ = (memory_ / sizeof(double) - memory_min) / (na_Qv + nb_Qv);
        if (max_occ < naocc_) {
            outfile->Printf("\n -> DF-UMP2 VV to be run in batches: max occ size = %zu", max_occ);
        } else {
            max_occ = naocc_;
        }

        batch_aocc = split_vector(a_occ_mos_, max_occ);
        batch_bocc = split_vector(b_occ_mos_, max_occ > nbocc_ ? nbocc_ : max_occ);
    }
    auto nbatches_alfa = batch_aocc.size();
    auto nbatches_beta = batch_bocc.size();

    // temp tensors for each thread
    double e_aa = 0.0, e_ab = 0.0, e_bb = 0.0;
    std::vector<ambit::Tensor> Da(nthreads), Db(nthreads);
    std::vector<ambit::Tensor> Jab(nthreads), JKab(nthreads);

    // alpha-alpha spin
    for (int i = 0; i < nthreads; ++i) {
        Da[i] = ambit::Tensor::build(CoreTensor, "Da vv", {navir_, navir_});
        Db[i] = ambit::Tensor::build(CoreTensor, "Db vv", {nbvir_, nbvir_});
        Jab[i] = ambit::Tensor::build(CoreTensor, "Jvv aa", {navir_, navir_});
        JKab[i] = ambit::Tensor::build(CoreTensor, "JKvv aa", {navir_, navir_});
    }

    for (size_t i_batch = 0; i_batch < nbatches_alfa; ++i_batch) {
        const auto& i_batch_occ_mos = batch_aocc[i_batch];
        auto i_naocc = i_batch_occ_mos.size();
        auto Bi = ambit::Tensor::build(ambit::CoreTensor, "Ba", {i_naocc, navir_, naux_});
        Bi("iag") = ints_->three_integral_block(aux_mos_, i_batch_occ_mos, a_vir_mos_)("gia");
        auto& Bi_vec = Bi.data();

        for (size_t j_batch = i_batch; j_batch < nbatches_alfa; ++j_batch) {
            const auto& j_batch_occ_mos = batch_aocc[j_batch];
            auto j_naocc = j_batch_occ_mos.size();
            ambit::Tensor Bj;
            if (j_batch == i_batch) {
                Bj = Bi;
            } else {
                Bj = ambit::Tensor::build(ambit::CoreTensor, "Bb", {j_naocc, navir_, naux_});
                Bj("iag") =
                    ints_->three_integral_block(aux_mos_, j_batch_occ_mos, a_vir_mos_)("gia");
            }
            auto& Bj_vec = Bj.data();

            // index pairs of i and j
            std::vector<std::pair<size_t, size_t>> ij_pairs;
            if (i_batch == j_batch) {
                for (size_t i = 0; i < i_naocc; ++i) {
                    for (size_t j = i + 1; j < j_naocc; ++j) {
                        ij_pairs.emplace_back(i, j);
                    }
                }
            } else {
                for (size_t i = 0; i < i_naocc; ++i) {
                    for (size_t j = 0; j < j_naocc; ++j) {
                        ij_pairs.emplace_back(i, j);
                    }
                }
            }
            size_t ij_pairs_size = ij_pairs.size();

// TODO: On adding default(none) shared(...), we observe symbol not found error on some OSX builds. Investigate and restore.
//#pragma omp parallel for default(none) shared(i_batch_occ_mos, j_batch_occ_mos, ij_pairs_size, ij_pairs, Bi_vec, Bj_vec, na_Qv, Jab, JKab, Da) reduction(+ : e_aa)
#pragma omp parallel for reduction(+ : e_aa)
            for (size_t p = 0; p < ij_pairs_size; ++p) {
                int thread = omp_get_thread_num();

                size_t i = ij_pairs[p].first;
                size_t j = ij_pairs[p].second;

                auto fock_i = Fa_[i_batch_occ_mos[i]];
                auto fock_j = Fa_[j_batch_occ_mos[j]];

                double* Bia_ptr = &Bi_vec[i * na_Qv];
                double* Bjb_ptr = &Bj_vec[j * na_Qv];

                // compute (ia|jb) for given indices i and j
                C_DGEMM('N', 'T', navir_, navir_, naux_, 1.0, Bia_ptr, naux_, Bjb_ptr, naux_, 0.0,
                        Jab[thread].data().data(), navir_);

                JKab[thread]("pq") = Jab[thread]("pq") - Jab[thread]("qp");
                Jab[thread].iterate([&](const std::vector<size_t>& i, double& value) {
                    value /= fock_i + fock_j - Fa_[a_vir_mos_[i[0]]] - Fa_[a_vir_mos_[i[1]]];
                });

                e_aa += Jab[thread]("pq") * JKab[thread]("pq");

                JKab[thread]("pq") = Jab[thread]("pq") - Jab[thread]("qp");
                Da[thread]("ab") += Jab[thread]("ac") * JKab[thread]("bc");
                Da[thread]("ab") += Jab[thread]("ca") * JKab[thread]("cb");
            }
        }
    }

    // alpha-beta spin
    for (int i = 0; i < nthreads; ++i) {
        Jab[i] = ambit::Tensor::build(CoreTensor, "Jvv ab", {navir_, nbvir_});
        JKab[i] = ambit::Tensor::build(CoreTensor, "JKvv ab", {navir_, nbvir_});
    }

    for (size_t i_batch = 0; i_batch < nbatches_alfa; ++i_batch) {
        const auto& i_batch_occ_mos = batch_aocc[i_batch];
        auto i_naocc = i_batch_occ_mos.size();
        auto Bi = ambit::Tensor::build(ambit::CoreTensor, "Ba", {i_naocc, navir_, naux_});
        Bi("iag") = ints_->three_integral_block(aux_mos_, i_batch_occ_mos, a_vir_mos_)("gia");
        auto& Bi_vec = Bi.data();

        for (size_t j_batch = 0; j_batch < nbatches_beta; ++j_batch) {
            const auto& j_batch_occ_mos = batch_bocc[j_batch];
            auto j_nbocc = j_batch_occ_mos.size();
            auto Bj = ambit::Tensor::build(ambit::CoreTensor, "Bb", {j_nbocc, nbvir_, naux_});
            Bj("iag") = ints_->three_integral_block(aux_mos_, j_batch_occ_mos, b_vir_mos_)("gia");
            auto& Bj_vec = Bj.data();

//#pragma omp parallel for default(none) shared(i_batch_occ_mos, i_naocc, Bi_vec, j_batch_occ_mos, j_nbocc, Bj_vec, na_Qv, nb_Qv, Jab, JKab, Da, Db) reduction(+ : e_ab)
#pragma omp parallel for reduction(+ : e_ab)
            for (size_t p = 0; p < i_naocc * j_nbocc; ++p) {
                int thread = omp_get_thread_num();
                size_t i = p / j_nbocc;
                size_t j = p % j_nbocc;

                auto fock_i = Fa_[i_batch_occ_mos[i]];
                auto fock_j = Fb_[j_batch_occ_mos[j]];

                double* Bia_ptr = &Bi_vec[i * na_Qv];
                double* Bjb_ptr = &Bj_vec[j * nb_Qv];

                // compute (ia|jb) for given indices i and j
                C_DGEMM('N', 'T', navir_, nbvir_, naux_, 1.0, Bia_ptr, naux_, Bjb_ptr, naux_, 0.0,
                        Jab[thread].data().data(), navir_);

                JKab[thread]("pq") = Jab[thread]("pq");
                Jab[thread].iterate([&](const std::vector<size_t>& i, double& value) {
                    value /= fock_i + fock_j - Fa_[a_vir_mos_[i[0]]] - Fb_[b_vir_mos_[i[1]]];
                });

                e_ab += Jab[thread]("pq") * JKab[thread]("pq");

                Da[thread]("ab") += Jab[thread]("ac") * Jab[thread]("bc");
                Db[thread]("ab") += Jab[thread]("ca") * Jab[thread]("cb");
            }
        }
    }

    // beta-beta spin
    for (int i = 0; i < nthreads; ++i) {
        Jab[i] = ambit::Tensor::build(CoreTensor, "Jvv bb", {nbvir_, nbvir_});
        JKab[i] = ambit::Tensor::build(CoreTensor, "JKvv bb", {nbvir_, nbvir_});
    }

    for (size_t i_batch = 0; i_batch < nbatches_beta; ++i_batch) {
        const auto& i_batch_occ_mos = batch_bocc[i_batch];
        auto i_nbocc = i_batch_occ_mos.size();
        auto Bi = ambit::Tensor::build(ambit::CoreTensor, "Ba", {i_nbocc, nbvir_, naux_});
        Bi("iag") = ints_->three_integral_block(aux_mos_, i_batch_occ_mos, b_vir_mos_)("gia");
        auto& Bi_vec = Bi.data();

        for (size_t j_batch = i_batch; j_batch < nbatches_beta; ++j_batch) {
            const auto& j_batch_occ_mos = batch_bocc[j_batch];
            auto j_nbocc = j_batch_occ_mos.size();
            ambit::Tensor Bj;
            if (j_batch == i_batch) {
                Bj = Bi;
            } else {
                Bj = ambit::Tensor::build(ambit::CoreTensor, "Bb", {j_nbocc, nbvir_, naux_});
                Bj("iag") =
                    ints_->three_integral_block(aux_mos_, j_batch_occ_mos, b_vir_mos_)("gia");
            }
            auto& Bj_vec = Bj.data();

            // index pairs of i and j
            std::vector<std::pair<size_t, size_t>> ij_pairs;
            if (j_batch == i_batch) {
                for (size_t i = 0; i < i_nbocc; ++i) {
                    for (size_t j = i + 1; j < j_nbocc; ++j) {
                        ij_pairs.emplace_back(i, j);
                    }
                }
            } else {
                for (size_t i = 0; i < i_nbocc; ++i) {
                    for (size_t j = 0; j < j_nbocc; ++j) {
                        ij_pairs.emplace_back(i, j);
                    }
                }
            }
            size_t ij_pairs_size = ij_pairs.size();

//#pragma omp parallel for default(none) shared(i_batch_occ_mos, j_batch_occ_mos, ij_pairs_size, ij_pairs, Bi_vec, Bj_vec, nb_Qv, Jab, JKab, Db) reduction(+ : e_bb)
#pragma omp parallel for reduction(+ : e_bb)
            for (size_t p = 0; p < ij_pairs_size; ++p) {
                int thread = omp_get_thread_num();

                size_t i = ij_pairs[p].first;
                size_t j = ij_pairs[p].second;

                auto fock_i = Fb_[i_batch_occ_mos[i]];
                auto fock_j = Fb_[j_batch_occ_mos[j]];

                double* Bia_ptr = &Bi_vec[i * nb_Qv];
                double* Bjb_ptr = &Bj_vec[j * nb_Qv];

                // compute (ia|jb) for given indices i and j
                C_DGEMM('N', 'T', nbvir_, nbvir_, naux_, 1.0, Bia_ptr, naux_, Bjb_ptr, naux_, 0.0,
                        Jab[thread].data().data(), nbvir_);

                JKab[thread]("pq") = Jab[thread]("pq") - Jab[thread]("qp");
                Jab[thread].iterate([&](const std::vector<size_t>& i, double& value) {
                    value /= fock_i + fock_j - Fb_[b_vir_mos_[i[0]]] - Fb_[b_vir_mos_[i[1]]];
                });

                e_bb += Jab[thread]("pq") * JKab[thread]("pq");

                JKab[thread]("pq") = Jab[thread]("pq") - Jab[thread]("qp");
                Db[thread]("ab") += Jab[thread]("ac") * JKab[thread]("bc");
                Db[thread]("ab") += Jab[thread]("ca") * JKab[thread]("cb");
            }
        }
    }

    // print energy
    double e_corr = e_aa + e_ab + e_bb;
    double e_ref = scf_info_->reference_energy();
    outfile->Printf("\n\n    SCF energy                            = %20.15f", e_ref);
    outfile->Printf("\n    MP2 correlation energy (aa)           = %20.15f", e_aa);
    outfile->Printf("\n    MP2 correlation energy (ab)           = %20.15f", e_ab);
    outfile->Printf("\n    MP2 correlation energy (bb)           = %20.15f", e_bb);
    outfile->Printf("\n    MP2 correlation energy                = %20.15f", e_corr);
    outfile->Printf("\n  * MP2 total energy                      = %20.15f\n", e_ref + e_corr);
    psi::Process::environment.globals["MP2 CORRELATION ENERGY"] = e_corr;

    // add Dvv contributions to D1
    for (int i = 0; i < nthreads; ++i) {
        D1.block("vv")("pq") += Da[i]("pq");
        D1.block("VV")("pq") += Db[i]("pq");
    }
}

void MP2_NOS::compute_df_ump2_1rdm_oo(ambit::BlockedTensor& D1) {
    timer tvv("DF-UMP2 1RDM OO");
    int nthreads = omp_get_max_threads();

    auto na_Qo = naux_ * naocc_;
    auto nb_Qo = naux_ * nbocc_;

    // batches of virtual indices
    std::vector<std::vector<size_t>> batch_avir, batch_bvir;

    // test memory
    size_t memory_min = 4 * nthreads * naocc_ * naocc_;
    if ((memory_min + na_Qo + nb_Qo) * sizeof(double) > memory_) {
        outfile->Printf("\n  Error: Not enough memory for DF-UMP2 (OO).");
        outfile->Printf(" Need at least %zu Bytes more!",
                        (memory_min + na_Qo + nb_Qo) * sizeof(double) - memory_);
        throw std::runtime_error("Not enough memory to run DF-MP2. Please check output.");
    } else {
        size_t max_vir = (memory_ / sizeof(double) - memory_min) / (na_Qo + nb_Qo);
        if (max_vir < navir_) {
            outfile->Printf("\n  -> DF-UMP2 OO to be run in batches: max vir size = %zu", max_vir);
        } else {
            max_vir = navir_;
        }

        batch_avir = split_vector(a_vir_mos_, max_vir);
        batch_bvir = split_vector(b_vir_mos_, max_vir > nbvir_ ? nbvir_ : max_vir);
    }
    auto nbatches_alfa = batch_avir.size();
    auto nbatches_beta = batch_bvir.size();

    // temp tensors for each thread
    std::vector<ambit::Tensor> Da(nthreads), Db(nthreads);
    std::vector<ambit::Tensor> Jmn(nthreads), JKmn(nthreads);

    // alpha-alpha spin
    for (int i = 0; i < nthreads; ++i) {
        Da[i] = ambit::Tensor::build(CoreTensor, "Da oo", {naocc_, naocc_});
        Db[i] = ambit::Tensor::build(CoreTensor, "Db oo", {nbocc_, nbocc_});
        Jmn[i] = ambit::Tensor::build(CoreTensor, "Joo aa", {naocc_, naocc_});
        JKmn[i] = ambit::Tensor::build(CoreTensor, "JKoo aa", {naocc_, naocc_});
    }

    for (size_t c_batch = 0; c_batch < nbatches_alfa; ++c_batch) {
        const auto& c_batch_vir_mos = batch_avir[c_batch];
        auto c_navir = c_batch_vir_mos.size();
        auto Bc = ambit::Tensor::build(ambit::CoreTensor, "Bi", {c_navir, naocc_, naux_});
        Bc("aig") = ints_->three_integral_block(aux_mos_, c_batch_vir_mos, a_occ_mos_)("gai");
        auto& Bc_vec = Bc.data();

        for (size_t d_batch = c_batch; d_batch < nbatches_alfa; ++d_batch) {
            const auto& d_batch_vir_mos = batch_avir[d_batch];
            auto d_navir = d_batch_vir_mos.size();
            ambit::Tensor Bd;
            if (d_batch == c_batch) {
                Bd = Bc;
            } else {
                Bd = ambit::Tensor::build(ambit::CoreTensor, "Bj", {d_navir, naocc_, naux_});
                Bd("aig") =
                    ints_->three_integral_block(aux_mos_, d_batch_vir_mos, a_occ_mos_)("gai");
            }
            auto& Bd_vec = Bd.data();

            // index pairs for c and d
            std::vector<std::pair<size_t, size_t>> cd_pairs;
            if (d_batch == c_batch) {
                for (size_t c = 0; c < c_navir; ++c) {
                    for (size_t d = c + 1; d < d_navir; ++d) {
                        cd_pairs.emplace_back(c, d);
                    }
                }
            } else {
                for (size_t c = 0; c < c_navir; ++c) {
                    for (size_t d = 0; d < d_navir; ++d) {
                        cd_pairs.emplace_back(c, d);
                    }
                }
            }
            size_t cd_pairs_size = cd_pairs.size();

//#pragma omp parallel for default(none) shared(c_batch_vir_mos, d_batch_vir_mos, cd_pairs_size,     \
//                                              cd_pairs, Bc_vec, Bd_vec, na_Qo, Jmn, JKmn, Da)
#pragma omp parallel for
            for (size_t p = 0; p < cd_pairs_size; ++p) {
                int thread = omp_get_thread_num();

                auto c = cd_pairs[p].first;
                auto d = cd_pairs[p].second;

                auto fock_c = Fa_[c_batch_vir_mos[c]];
                auto fock_d = Fa_[d_batch_vir_mos[d]];

                double* Bci_ptr = &Bc_vec[c * na_Qo];
                double* Bdj_ptr = &Bd_vec[d * na_Qo];

                // compute (ci|dj) for given indices c and d
                C_DGEMM('N', 'T', naocc_, naocc_, naux_, 1.0, Bci_ptr, naux_, Bdj_ptr, naux_, 0.0,
                        Jmn[thread].data().data(), naocc_);

                Jmn[thread].iterate([&](const std::vector<size_t>& i, double& value) {
                    value /= Fa_[a_occ_mos_[i[0]]] + Fa_[a_occ_mos_[i[1]]] - fock_c - fock_d;
                });

                JKmn[thread]("pq") = Jmn[thread]("pq") - Jmn[thread]("qp");
                Da[thread]("ij") -= Jmn[thread]("ik") * JKmn[thread]("jk");
                Da[thread]("ij") -= Jmn[thread]("ki") * JKmn[thread]("kj");
            }
        }
    }

    // alpha-beta spin
    for (int i = 0; i < nthreads; ++i) {
        Jmn[i] = ambit::Tensor::build(CoreTensor, "Joo ab", {naocc_, nbocc_});
    }

    for (size_t c_batch = 0; c_batch < nbatches_alfa; ++c_batch) {
        const auto& c_batch_vir_mos = batch_avir[c_batch];
        auto c_navir = c_batch_vir_mos.size();
        auto Bc = ambit::Tensor::build(ambit::CoreTensor, "Bi", {c_navir, naocc_, naux_});
        Bc("aig") = ints_->three_integral_block(aux_mos_, c_batch_vir_mos, a_occ_mos_)("gai");
        auto& Bc_vec = Bc.data();

        for (size_t d_batch = 0; d_batch < nbatches_beta; ++d_batch) {
            const auto& d_batch_vir_mos = batch_bvir[d_batch];
            auto d_nbvir = d_batch_vir_mos.size();
            auto Bd = ambit::Tensor::build(ambit::CoreTensor, "Bj", {d_nbvir, nbocc_, naux_});
            Bd("aig") = ints_->three_integral_block(aux_mos_, d_batch_vir_mos, b_occ_mos_)("gai");
            auto& Bd_vec = Bd.data();

//#pragma omp parallel for default(none) shared(c_batch_vir_mos, c_navir, d_batch_vir_mos, d_nbvir,  \
//                                              Bc_vec, Bd_vec, na_Qo, nb_Qo, Jmn, Da, Db)
#pragma omp parallel for
            for (size_t p = 0; p < c_navir * d_nbvir; ++p) {
                int thread = omp_get_thread_num();
                size_t c = p / d_nbvir;
                size_t d = p % d_nbvir;

                auto fock_c = Fa_[c_batch_vir_mos[c]];
                auto fock_d = Fb_[d_batch_vir_mos[d]];

                double* Bci_ptr = &Bc_vec[c * na_Qo];
                double* Bdj_ptr = &Bd_vec[d * nb_Qo];

                // compute (ci|dj) for given indices c and d
                C_DGEMM('N', 'T', naocc_, nbocc_, naux_, 1.0, Bci_ptr, naux_, Bdj_ptr, naux_, 0.0,
                        Jmn[thread].data().data(), naocc_);

                Jmn[thread].iterate([&](const std::vector<size_t>& i, double& value) {
                    value /= Fa_[a_occ_mos_[i[0]]] + Fb_[b_occ_mos_[i[1]]] - fock_c - fock_d;
                });

                Da[thread]("ij") -= Jmn[thread]("ik") * Jmn[thread]("jk");
                Db[thread]("ij") -= Jmn[thread]("ki") * Jmn[thread]("kj");
            }
        }
    }

    // beta-beta spin
    for (int i = 0; i < nthreads; ++i) {
        Jmn[i] = ambit::Tensor::build(CoreTensor, "Joo bb", {nbocc_, nbocc_});
        JKmn[i] = ambit::Tensor::build(CoreTensor, "JKoo bb", {nbocc_, nbocc_});
    }

    for (size_t c_batch = 0; c_batch < nbatches_beta; ++c_batch) {
        const auto& c_batch_vir_mos = batch_bvir[c_batch];
        auto c_nbvir = c_batch_vir_mos.size();
        auto Bc = ambit::Tensor::build(ambit::CoreTensor, "Bi", {c_nbvir, nbocc_, naux_});
        Bc("aig") = ints_->three_integral_block(aux_mos_, c_batch_vir_mos, b_occ_mos_)("gai");
        auto& Bc_vec = Bc.data();

        for (size_t d_batch = c_batch; d_batch < nbatches_beta; ++d_batch) {
            const auto& d_batch_vir_mos = batch_bvir[d_batch];
            auto d_nbvir = d_batch_vir_mos.size();
            ambit::Tensor Bd;
            if (d_batch == c_batch) {
                Bd = Bc;
            } else {
                Bd = ambit::Tensor::build(ambit::CoreTensor, "Bj", {d_nbvir, nbocc_, naux_});
                Bd("aig") =
                    ints_->three_integral_block(aux_mos_, d_batch_vir_mos, b_occ_mos_)("gai");
            }
            auto& Bd_vec = Bd.data();

            // index pairs for c and d
            std::vector<std::pair<size_t, size_t>> cd_pairs;
            if (d_batch == c_batch) {
                for (size_t c = 0; c < c_nbvir; ++c) {
                    for (size_t d = c + 1; d < d_nbvir; ++d) {
                        cd_pairs.emplace_back(c, d);
                    }
                }
            } else {
                for (size_t c = 0; c < c_nbvir; ++c) {
                    for (size_t d = 0; d < d_nbvir; ++d) {
                        cd_pairs.emplace_back(c, d);
                    }
                }
            }
            size_t cd_pairs_size = cd_pairs.size();

//#pragma omp parallel for default(none) shared(c_batch_vir_mos, d_batch_vir_mos, cd_pairs_size,     \
//                                              cd_pairs, Bc_vec, Bd_vec, nb_Qo, Jmn, JKmn, Db)
#pragma omp parallel for
            for (size_t p = 0; p < cd_pairs_size; ++p) {
                int thread = omp_get_thread_num();

                auto c = cd_pairs[p].first;
                auto d = cd_pairs[p].second;

                auto fock_c = Fb_[c_batch_vir_mos[c]];
                auto fock_d = Fb_[d_batch_vir_mos[d]];

                double* Bci_ptr = &Bc_vec[c * nb_Qo];
                double* Bdj_ptr = &Bd_vec[d * nb_Qo];

                // compute (ci|dj) for given indices c and d
                C_DGEMM('N', 'T', nbocc_, nbocc_, naux_, 1.0, Bci_ptr, naux_, Bdj_ptr, naux_, 0.0,
                        Jmn[thread].data().data(), nbocc_);

                Jmn[thread].iterate([&](const std::vector<size_t>& i, double& value) {
                    value /= Fb_[b_occ_mos_[i[0]]] + Fb_[b_occ_mos_[i[1]]] - fock_c - fock_d;
                });

                JKmn[thread]("pq") = Jmn[thread]("pq") - Jmn[thread]("qp");
                Db[thread]("ij") -= Jmn[thread]("ik") * JKmn[thread]("jk");
                Db[thread]("ij") -= Jmn[thread]("ki") * JKmn[thread]("kj");
            }
        }
    }

    // add Doo contributions to D1
    for (int i = 0; i < nthreads; ++i) {
        D1.block("oo")("pq") += Da[i]("pq");
        D1.block("OO")("pq") += Db[i]("pq");
    }
}

void MP2_NOS::compute_df_rmp2_1rdm_vv(ambit::BlockedTensor& D1) {
    timer tvv("DF-RMP2 1RDM VV");
    int nthreads = omp_get_max_threads();
    auto n_Qv = naux_ * navir_;

    // batches of occupied indices
    std::vector<std::vector<size_t>> batch_occ;

    // test memory
    size_t memory_min = 3 * nthreads * navir_ * navir_;
    if ((memory_min + n_Qv) * sizeof(double) > memory_) {
        outfile->Printf("\n  Error: Not enough memory for DF-RMP2 (VV).");
        outfile->Printf(" Need at least %zu Bytes more!",
                        (memory_min + n_Qv) * sizeof(double) - memory_);
        throw std::runtime_error("Not enough memory to run DF-MP2. Please check output.");
    } else {
        size_t max_occ = (memory_ / sizeof(double) - memory_min) / n_Qv;
        if (max_occ < naocc_) {
            outfile->Printf("\n -> DF-RMP2 VV to be run in batches: max occ size = %zu", max_occ);
        } else {
            max_occ = naocc_;
        }
        batch_occ = split_vector(a_occ_mos_, max_occ);
    }
    auto nbatches = batch_occ.size();

    // temp tensors for each thread
    double e_corr = 0.0;
    std::vector<ambit::Tensor> Da(nthreads);
    std::vector<ambit::Tensor> Jab(nthreads), JKab(nthreads);

    for (int i = 0; i < nthreads; ++i) {
        Da[i] = ambit::Tensor::build(CoreTensor, "Da vv", {navir_, navir_});
        Jab[i] = ambit::Tensor::build(CoreTensor, "Jvv aa", {navir_, navir_});
        JKab[i] = ambit::Tensor::build(CoreTensor, "JKvv aa", {navir_, navir_});
    }

    for (size_t i_batch = 0; i_batch < nbatches; ++i_batch) {
        const auto& i_batch_occ_mos = batch_occ[i_batch];
        auto i_nocc = i_batch_occ_mos.size();
        auto Bi = ambit::Tensor::build(ambit::CoreTensor, "Ba", {i_nocc, navir_, naux_});
        Bi("iag") = ints_->three_integral_block(aux_mos_, i_batch_occ_mos, a_vir_mos_)("gia");
        auto& Bi_vec = Bi.data();

        for (size_t j_batch = i_batch; j_batch < nbatches; ++j_batch) {
            const auto& j_batch_occ_mos = batch_occ[j_batch];
            auto j_nocc = j_batch_occ_mos.size();
            ambit::Tensor Bj;
            if (j_batch == i_batch) {
                Bj = Bi;
            } else {
                Bj = ambit::Tensor::build(ambit::CoreTensor, "Bb", {j_nocc, navir_, naux_});
                Bj("iag") =
                    ints_->three_integral_block(aux_mos_, j_batch_occ_mos, a_vir_mos_)("gia");
            }
            auto& Bj_vec = Bj.data();

            // index pairs of i and j
            std::vector<std::pair<size_t, size_t>> ij_pairs;
            if (i_batch == j_batch) {
                for (size_t i = 0; i < i_nocc; ++i) {
                    for (size_t j = i; j < j_nocc; ++j) {
                        ij_pairs.emplace_back(i, j);
                    }
                }
            } else {
                for (size_t i = 0; i < i_nocc; ++i) {
                    for (size_t j = 0; j < j_nocc; ++j) {
                        ij_pairs.emplace_back(i, j);
                    }
                }
            }
            size_t ij_pairs_size = ij_pairs.size();

//#pragma omp parallel for default(none) shared(i_batch_occ_mos, j_batch_occ_mos, ij_pairs_size, ij_pairs, Bi_vec, Bj_vec, n_Qv, Jab, JKab, Da) reduction(+ : e_corr)
#pragma omp parallel for reduction(+ : e_corr)
            for (size_t p = 0; p < ij_pairs_size; ++p) {
                int thread = omp_get_thread_num();

                size_t i = ij_pairs[p].first;
                size_t j = ij_pairs[p].second;

                auto fock_i = Fa_[i_batch_occ_mos[i]];
                auto fock_j = Fa_[j_batch_occ_mos[j]];

                double* Bia_ptr = &Bi_vec[i * n_Qv];
                double* Bjb_ptr = &Bj_vec[j * n_Qv];

                // compute (ia|jb) for given indices i and j
                C_DGEMM('N', 'T', navir_, navir_, naux_, 1.0, Bia_ptr, naux_, Bjb_ptr, naux_, 0.0,
                        Jab[thread].data().data(), navir_);

                JKab[thread]("pq") = 2.0 * Jab[thread]("pq") - Jab[thread]("qp");
                Jab[thread].iterate([&](const std::vector<size_t>& i, double& value) {
                    value /= fock_i + fock_j - Fa_[a_vir_mos_[i[0]]] - Fa_[a_vir_mos_[i[1]]];
                });
                auto factor = (i_batch_occ_mos[i] == j_batch_occ_mos[j]) ? 1.0 : 2.0;
                e_corr += factor * Jab[thread]("pq") * JKab[thread]("pq");

                JKab[thread]("pq") = 2.0 * Jab[thread]("pq") - Jab[thread]("qp");
                Da[thread]("ab") += 0.5 * factor * Jab[thread]("ac") * JKab[thread]("bc");
                Da[thread]("ab") += 0.5 * factor * Jab[thread]("ca") * JKab[thread]("cb");
            }
        }
    }

    // print energy
    double e_ref = scf_info_->reference_energy();
    outfile->Printf("\n\n    SCF energy                            = %20.15f", e_ref);
    outfile->Printf("\n    MP2 correlation energy                = %20.15f", e_corr);
    outfile->Printf("\n  * MP2 total energy                      = %20.15f\n", e_ref + e_corr);
    psi::Process::environment.globals["MP2 CORRELATION ENERGY"] = e_corr;

    // add Dvv contributions to D1
    for (int i = 0; i < nthreads; ++i) {
        D1.block("vv")("pq") += Da[i]("pq");
        D1.block("VV")("pq") += Da[i]("pq");
    }
}

void MP2_NOS::compute_df_rmp2_1rdm_oo(ambit::BlockedTensor& D1) {
    timer tvv("DF-RMP2 1RDM OO");
    int nthreads = omp_get_max_threads();
    auto n_Qo = naux_ * naocc_;

    // batches of virtual indices
    std::vector<std::vector<size_t>> batch_vir;

    // test memory
    size_t memory_min = 3 * nthreads * naocc_ * naocc_;
    if ((memory_min + n_Qo) * sizeof(double) > memory_) {
        outfile->Printf("\n  Error: Not enough memory for DF-RMP2 (OO).");
        outfile->Printf(" Need at least %zu Bytes more!",
                        (memory_min + n_Qo) * sizeof(double) - memory_);
        throw std::runtime_error("Not enough memory to run DF-MP2. Please check output.");
    } else {
        size_t max_vir = (memory_ / sizeof(double) - memory_min) / n_Qo;
        if (max_vir < navir_) {
            outfile->Printf("\n -> DF-RMP2 OO to be run in batches: max occ size = %zu", max_vir);
        } else {
            max_vir = navir_;
        }
        batch_vir = split_vector(a_vir_mos_, max_vir);
    }
    auto nbatches = batch_vir.size();

    // temp tensors for each thread
    std::vector<ambit::Tensor> Da(nthreads);
    std::vector<ambit::Tensor> Jmn(nthreads), JKmn(nthreads);

    for (int i = 0; i < nthreads; ++i) {
        Da[i] = ambit::Tensor::build(CoreTensor, "Da oo", {naocc_, naocc_});
        Jmn[i] = ambit::Tensor::build(CoreTensor, "Jvv aa", {naocc_, naocc_});
        JKmn[i] = ambit::Tensor::build(CoreTensor, "JKvv aa", {naocc_, naocc_});
    }

    for (size_t c_batch = 0; c_batch < nbatches; ++c_batch) {
        const auto& c_batch_vir_mos = batch_vir[c_batch];
        auto c_nvir = c_batch_vir_mos.size();
        auto Bc = ambit::Tensor::build(ambit::CoreTensor, "Bi", {c_nvir, naocc_, naux_});
        Bc("aig") = ints_->three_integral_block(aux_mos_, c_batch_vir_mos, a_occ_mos_)("gai");
        auto& Bc_vec = Bc.data();

        for (size_t d_batch = c_batch; d_batch < nbatches; ++d_batch) {
            const auto& d_batch_vir_mos = batch_vir[d_batch];
            auto d_nvir = d_batch_vir_mos.size();
            ambit::Tensor Bd;
            if (d_batch == c_batch) {
                Bd = Bc;
            } else {
                Bd = ambit::Tensor::build(ambit::CoreTensor, "Bj", {d_nvir, naocc_, naux_});
                Bd("aig") =
                    ints_->three_integral_block(aux_mos_, d_batch_vir_mos, a_occ_mos_)("gai");
            }
            auto& Bd_vec = Bd.data();

            // index pairs for c and d
            std::vector<std::pair<size_t, size_t>> cd_pairs;
            if (d_batch == c_batch) {
                for (size_t c = 0; c < c_nvir; ++c) {
                    for (size_t d = c; d < d_nvir; ++d) {
                        cd_pairs.emplace_back(c, d);
                    }
                }
            } else {
                for (size_t c = 0; c < c_nvir; ++c) {
                    for (size_t d = 0; d < d_nvir; ++d) {
                        cd_pairs.emplace_back(c, d);
                    }
                }
            }
            size_t cd_pairs_size = cd_pairs.size();

//#pragma omp parallel for default(none) shared(c_batch_vir_mos, d_batch_vir_mos, cd_pairs_size,     \
//                                              cd_pairs, Bc_vec, Bd_vec, n_Qo, Jmn, JKmn, Da)
#pragma omp parallel for
            for (size_t p = 0; p < cd_pairs_size; ++p) {
                int thread = omp_get_thread_num();

                auto c = cd_pairs[p].first;
                auto d = cd_pairs[p].second;

                auto fock_c = Fa_[c_batch_vir_mos[c]];
                auto fock_d = Fa_[d_batch_vir_mos[d]];

                double* Bci_ptr = &Bc_vec[c * n_Qo];
                double* Bdj_ptr = &Bd_vec[d * n_Qo];

                // compute (ci|dj) for given indices c and d
                C_DGEMM('N', 'T', naocc_, naocc_, naux_, 1.0, Bci_ptr, naux_, Bdj_ptr, naux_, 0.0,
                        Jmn[thread].data().data(), naocc_);

                Jmn[thread].iterate([&](const std::vector<size_t>& i, double& value) {
                    value /= Fa_[a_occ_mos_[i[0]]] + Fa_[a_occ_mos_[i[1]]] - fock_c - fock_d;
                });
                JKmn[thread]("pq") = 2.0 * Jmn[thread]("pq") - Jmn[thread]("qp");

                auto factor = (c_batch_vir_mos[c] == d_batch_vir_mos[d]) ? 0.5 : 1.0;
                Da[thread]("ij") -= factor * Jmn[thread]("ik") * JKmn[thread]("jk");
                Da[thread]("ij") -= factor * Jmn[thread]("ki") * JKmn[thread]("kj");
            }
        }
    }

    // add Doo contributions to D1
    for (int i = 0; i < nthreads; ++i) {
        D1.block("oo")("pq") += Da[i]("pq");
        D1.block("OO")("pq") += Da[i]("pq");
    }
}

} // namespace forte
