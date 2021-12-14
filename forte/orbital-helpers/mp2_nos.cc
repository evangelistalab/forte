/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2021 by its authors (see COPYING, COPYING.LESSER, AUTHORS).
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
#include "mp2_nos.h"

#ifdef _OPENMP
#include <omp.h>
#else
bool MP2_NOS::have_omp_ = false;
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

    psi::Dimension ncmopi_ = mo_space_info_->dimension("CORRELATED");
    psi::Dimension frzcpi = mo_space_info_->dimension("FROZEN_DOCC");
    psi::Dimension frzvpi = mo_space_info_->dimension("FROZEN_UOCC");

    psi::Dimension nmopi = mo_space_info_->dimension("ALL");
    psi::Dimension doccpi = scf_info_->doccpi();
    psi::Dimension soccpi = scf_info_->soccpi();

    psi::Dimension corr_docc(doccpi);
    corr_docc -= frzcpi;

    psi::Dimension aoccpi = corr_docc + scf_info_->soccpi();
    psi::Dimension boccpi = corr_docc;
    psi::Dimension avirpi = ncmopi_ - aoccpi;
    psi::Dimension bvirpi = ncmopi_ - boccpi;

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
        for (int a = 0; a < ncmopi_[h] - corr_docc[h] - soccpi[h]; ++a, ++p) {
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

    Fa_.clear();
    Fb_.clear();

    for (int h = 0; h < nirrep; ++h) {
        for (int i = 0; i < ncmopi_[h]; ++i) {
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

        BlockedTensor D1pp = BlockedTensor::build(CoreTensor, "D1", spin_cases({"oo", "vv"}));
        D1pp.block("oo").iterate(
            [&](const std::vector<size_t>& i, double& value) { value = i[0] == i[1] ? 1.0 : 0.0; });
        D1pp.block("OO").iterate(
            [&](const std::vector<size_t>& i, double& value) { value = i[0] == i[1] ? 1.0 : 0.0; });

        compute_df_ump2_1rdm_vv(D1pp);
        auto D1t = build_1rdm_df();
        D1 = build_1rdm_conv();

        D1pp.block("vv")("pq") -= D1.block("vv")("pq");
        D1pp.block("VV")("pq") -= D1.block("VV")("pq");
        outfile->Printf("\n  Dvv diff = %20.15f, DVV diff = %20.15f", D1pp.block("vv").norm(0),
                        D1pp.block("VV").norm(0));

        D1t["pq"] -= D1["pq"];
        D1t["PQ"] -= D1["PQ"];
        outfile->Printf("\n  Dvv diff = %20.15f, DVV diff = %20.15f", D1t.block("vv").norm(0),
                        D1t.block("VV").norm(0));
        outfile->Printf("\n  Doo diff = %20.15f, DOO diff = %20.15f", D1t.block("oo").norm(0),
                        D1t.block("OO").norm(0));
    } else {
        D1 = build_1rdm_conv();
    }

    // Copy the density matrix to matrix objects
    auto D1oo = tensor_to_matrix(D1.block("oo"), aoccpi);
    auto D1OO = tensor_to_matrix(D1.block("OO"), boccpi);
    auto D1vv = tensor_to_matrix(D1.block("vv"), avirpi);
    auto D1VV = tensor_to_matrix(D1.block("VV"), bvirpi);

    Matrix D1oo_evecs("D1oo_evecs", aoccpi, aoccpi);
    Matrix D1OO_evecs("D1OO_evecs", boccpi, boccpi);
    Matrix D1vv_evecs("D1vv_evecs", avirpi, avirpi);
    Matrix D1VV_evecs("D1VV_evecs", bvirpi, bvirpi);

    Vector D1oo_evals("D1oo_evals", aoccpi);
    Vector D1OO_evals("D1OO_evals", boccpi);
    Vector D1vv_evals("D1vv_evals", avirpi);
    Vector D1VV_evals("D1VV_evals", bvirpi);

    D1oo->diagonalize(D1oo_evecs, D1oo_evals);
    D1vv->diagonalize(D1vv_evecs, D1vv_evals);
    D1OO->diagonalize(D1OO_evecs, D1OO_evals);
    D1VV->diagonalize(D1VV_evecs, D1VV_evals);

    // Print natural orbitals
    if (options_->get_bool("NAT_ORBS_PRINT"))

    {
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
        double occupied = options_->get_double("OCC_NATURAL");
        double virtual_orb = options_->get_double("VIRT_NATURAL");
        outfile->Printf("\n Suggested Active Space \n");
        outfile->Printf("\n Occupied orbitals with an occupation less than "
                        "%6.4f are active",
                        occupied);
        outfile->Printf("\n Virtual orbitals with an occupation greater than "
                        "%6.4f are active",
                        virtual_orb);
        outfile->Printf("\n Remember, these are suggestions  :-)!\n");
        for (int h = 0; h < nirrep; ++h) {
            size_t restricted_docc_number = 0;
            size_t active_number = 0;
            for (int i = 0; i < aoccpi[h]; ++i) {
                if (D1oo_evals.get(h, i) < occupied) {
                    active_number++;
                    outfile->Printf("\n In %u, orbital occupation %u = %8.6f "
                                    "Active occupied",
                                    h, i, D1oo_evals.get(h, i));
                    active[h] = active_number;
                } else if (D1oo_evals.get(h, i) >= occupied) {
                    restricted_docc_number++;
                    outfile->Printf("\n In %u, orbital occupation %u = %8.6f  RDOCC", h, i,
                                    D1oo_evals.get(h, i));
                    restricted_docc[h] = restricted_docc_number;
                }
            }
            for (int a = 0; a < avirpi[h]; ++a) {
                if (D1vv_evals.get(h, a) > virtual_orb) {
                    active_number++;
                    active[h] = active_number;
                    outfile->Printf("\n In %u, orbital occupation %u = %8.6f "
                                    "Active virtual",
                                    h, a, D1vv_evals.get(h, a));
                }
            }
        }
        outfile->Printf("\n By occupation analysis, your restricted docc should be\n");
        outfile->Printf("\n Restricted_docc = [");
        for (auto& rocc : restricted_docc) {
            outfile->Printf("%u, ", rocc);
        }
        outfile->Printf("]\n");
        outfile->Printf("\n By occupation analysis, active space should be \n");
        outfile->Printf("\n Active = [");
        for (auto& ract : active) {
            outfile->Printf("%u, ", ract);
        }
        outfile->Printf("]\n");
    }

    std::shared_ptr<psi::Matrix> Ua = std::make_shared<psi::Matrix>("Ua", nmopi, nmopi);
    // Patch together the transformation matrices
    for (int h = 0; h < nirrep; ++h) {
        size_t irrep_offset = 0;

        // Frozen core orbitals are unchanged
        for (int p = 0; p < frzcpi[h]; ++p) {
            Ua->set(h, p, p, 1.0);
        }
        irrep_offset += frzcpi[h];

        // Occupied alpha
        for (int p = 0; p < aoccpi[h]; ++p) {
            for (int q = 0; q < aoccpi[h]; ++q) {
                double value = D1oo_evecs.get(h, p, q);
                Ua->set(h, p + irrep_offset, q + irrep_offset, value);
            }
        }
        irrep_offset += aoccpi[h];

        // Virtual alpha
        for (int p = 0; p < avirpi[h]; ++p) {
            for (int q = 0; q < avirpi[h]; ++q) {
                double value = D1vv_evecs.get(h, p, q);
                Ua->set(h, p + irrep_offset, q + irrep_offset, value);
            }
        }
        irrep_offset += avirpi[h];

        // Frozen virtual orbitals are unchanged
        for (int p = 0; p < frzvpi[h]; ++p) {
            Ua->set(h, p + irrep_offset, p + irrep_offset, 1.0);
        }
    }

    std::shared_ptr<psi::Matrix> Ub = std::make_shared<psi::Matrix>("Ub", nmopi, nmopi);
    // Patch together the transformation matrices
    for (int h = 0; h < nirrep; ++h) {
        size_t irrep_offset = 0;

        // Frozen core orbitals are unchanged
        for (int p = 0; p < frzcpi[h]; ++p) {
            Ub->set(h, p, p, 1.0);
        }
        irrep_offset += frzcpi[h];

        // Occupied alpha
        for (int p = 0; p < boccpi[h]; ++p) {
            for (int q = 0; q < boccpi[h]; ++q) {
                double value = D1OO_evecs.get(h, p, q);
                Ub->set(h, p + irrep_offset, q + irrep_offset, value);
            }
        }
        irrep_offset += boccpi[h];

        // Virtual alpha
        for (int p = 0; p < bvirpi[h]; ++p) {
            for (int q = 0; q < bvirpi[h]; ++q) {
                double value = D1VV_evecs.get(h, p, q);
                Ub->set(h, p + irrep_offset, q + irrep_offset, value);
            }
        }
        irrep_offset += bvirpi[h];

        // Frozen virtual orbitals are unchanged
        for (int p = 0; p < frzvpi[h]; ++p) {
            Ub->set(h, p + irrep_offset, p + irrep_offset, 1.0);
        }
    }

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

    // number of orbitals
    auto naocc = a_occ_mos_.size();
    auto nbocc = b_occ_mos_.size();
    auto navir = a_vir_mos_.size();
    auto nbvir = b_vir_mos_.size();
    auto naux = aux_mos_.size();

    auto na_Qv = naux * navir;
    auto nb_Qv = naux * nbvir;
    auto na_Qo = naux * naocc;
    auto nb_Qo = naux * nbocc;

    // memory in bytes
    size_t memory = psi::Process::environment.get_memory() * 0.9;
    outfile->Printf("\n  Memory:            %zu MB", memory / 1024 / 1024);

    int nthreads = omp_get_max_threads();
    outfile->Printf("\n  Number of threads: %d", nthreads);

    // just assume we can store 3-index integrals in memory
    auto Ba = ambit::Tensor::build(ambit::CoreTensor, "Ba", {naocc, navir, naux});
    Ba("iag") = ints_->three_integral_block(aux_mos_, a_occ_mos_, a_vir_mos_)("gia");
    auto& Ba_data = Ba.data();

    auto Bb = ambit::Tensor::build(ambit::CoreTensor, "Bb", {nbocc, nbvir, naux});
    Bb("iag") = ints_->three_integral_block(aux_mos_, b_occ_mos_, b_vir_mos_)("gia");
    auto& Bb_data = Bb.data();

    // compute MP2 energy
    double e_aa = 0.0, e_ab = 0.0, e_bb = 0.0;
    std::vector<ambit::Tensor> Jab(nthreads), JKab(nthreads);
    std::vector<ambit::Tensor> Da(nthreads), Db(nthreads);

    // aa
    for (int i = 0; i < nthreads; ++i) {
        Jab[i] = ambit::Tensor::build(CoreTensor, "Jab", {navir, navir});
        JKab[i] = ambit::Tensor::build(CoreTensor, "JKab", {navir, navir});
        Da[i] = ambit::Tensor::build(CoreTensor, "Da", {navir, navir});
        Db[i] = ambit::Tensor::build(CoreTensor, "Db", {nbvir, nbvir});
    }

#pragma omp parallel for collapse(2) default(none) shared(Ba_data, naocc, navir, naux, na_Qv, Jab, JKab, Da) reduction(+ : e_aa)
    for (size_t i = 0; i < naocc; ++i) {
        double fock_i = Fa_[a_occ_mos_[i]];

        // grab data for index i
        double* Bia_ptr = &Ba_data[i * na_Qv];

        int thread = omp_get_thread_num();

        for (size_t j = 0; j < naocc; ++j) {
            double fock_j = Fa_[a_occ_mos_[j]];

            // grab data for index j
            double* Bjb_ptr = &Ba_data[j * na_Qv];

            // compute (ia|jb) for given indices i and j
            double* Vab_ptr = Jab[thread].data().data();
            C_DGEMM('N', 'T', navir, navir, naux, 1.0, Bia_ptr, naux, Bjb_ptr, naux, 0.0, Vab_ptr,
                    navir);

            JKab[thread]("pq") = Jab[thread]("pq") - Jab[thread]("qp");
            Jab[thread].iterate([&](const std::vector<size_t>& i, double& value) {
                value /= fock_i + fock_j - Fa_[a_vir_mos_[i[0]]] - Fa_[a_vir_mos_[i[1]]];
            });

            e_aa += 0.5 * Jab[thread]("pq") * JKab[thread]("pq");

            JKab[thread]("pq") = Jab[thread]("pq") - Jab[thread]("qp");
            Da[thread]("ab") += Jab[thread]("ac") * JKab[thread]("bc");
        }
    }

    // ab
    for (int i = 0; i < nthreads; ++i) {
        Jab[i] = ambit::Tensor::build(CoreTensor, "Jab", {navir, nbvir});
        JKab[i] = ambit::Tensor::build(CoreTensor, "JKab", {navir, nbvir});
    }

#pragma omp parallel for collapse(2) default(none) shared(Ba_data, Bb_data, naocc, navir, na_Qv, nbocc, nbvir, nb_Qv, naux, Jab, JKab, Da, Db) reduction(+ : e_ab)
    for (size_t i = 0; i < naocc; ++i) {
        double fock_i = Fa_[a_occ_mos_[i]];

        // grab data for index i
        double* Bia_ptr = &Ba_data[i * na_Qv];

        int thread = omp_get_thread_num();

        for (size_t j = 0; j < nbocc; ++j) {
            double fock_j = Fb_[b_occ_mos_[j]];

            // grab data for index j
            double* Bjb_ptr = &Bb_data[j * nb_Qv];

            // compute (ia|jb) = Bi(aQ) * Bj(bQ) for given indices i and j
            double* Vab_ptr = Jab[thread].data().data();
            C_DGEMM('N', 'T', navir, nbvir, naux, 1.0, Bia_ptr, naux, Bjb_ptr, naux, 0.0, Vab_ptr,
                    navir);

            JKab[thread]("pq") = Jab[thread]("pq");
            Jab[thread].iterate([&](const std::vector<size_t>& i, double& value) {
                value /= fock_i + fock_j - Fa_[a_vir_mos_[i[0]]] - Fb_[b_vir_mos_[i[1]]];
            });

            e_ab += Jab[thread]("pq") * JKab[thread]("pq");

            Da[thread]("ab") += Jab[thread]("ac") * Jab[thread]("bc");
            Db[thread]("ab") += Jab[thread]("ca") * Jab[thread]("cb");
        }
    }

    // bb
    for (int i = 0; i < nthreads; ++i) {
        Jab[i] = ambit::Tensor::build(CoreTensor, "Jab", {nbvir, nbvir});
        JKab[i] = ambit::Tensor::build(CoreTensor, "JKab", {nbvir, nbvir});
    }

#pragma omp parallel for collapse(2) default(none) shared(Bb_data, nbocc, nbvir, naux, nb_Qv, Jab, JKab, Db) reduction(+ : e_bb)
    for (size_t i = 0; i < nbocc; ++i) {
        double fock_i = Fb_[b_occ_mos_[i]];

        // grab data for index i
        double* Bia_ptr = &Bb_data[i * nb_Qv];

        int thread = omp_get_thread_num();

        for (size_t j = 0; j < nbocc; ++j) {
            double fock_j = Fb_[b_occ_mos_[j]];

            // grab data for index j
            double* Bjb_ptr = &Bb_data[j * nb_Qv];

            // compute (ia|jb) for given indices i and j
            double* Vab_ptr = Jab[thread].data().data();
            C_DGEMM('N', 'T', nbvir, nbvir, naux, 1.0, Bia_ptr, naux, Bjb_ptr, naux, 0.0, Vab_ptr,
                    nbvir);

            JKab[thread]("pq") = Jab[thread]("pq") - Jab[thread]("qp");
            Jab[thread].iterate([&](const std::vector<size_t>& i, double& value) {
                value /= fock_i + fock_j - Fb_[b_vir_mos_[i[0]]] - Fb_[b_vir_mos_[i[1]]];
            });

            e_bb += 0.5 * Jab[thread]("pq") * JKab[thread]("pq");

            JKab[thread]("pq") = Jab[thread]("pq") - Jab[thread]("qp");
            Db[thread]("ab") += Jab[thread]("ac") * JKab[thread]("bc");
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
    outfile->Printf("\n  * MP2 total energy                      = %20.15f\n\n", e_ref + e_corr);

    // add Dvv contributions to D1
    for (int i = 0; i < nthreads; ++i) {
        D1.block("vv")("pq") += Da[i]("pq");
        D1.block("VV")("pq") += Db[i]("pq");
    }

    // compute Doo contributions
    Ba = ambit::Tensor::build(ambit::CoreTensor, "Ba", {navir, naocc, naux});
    Ba("aig") = ints_->three_integral_block(aux_mos_, a_vir_mos_, a_occ_mos_)("gai");
    Ba_data = Ba.data();

    Bb = ambit::Tensor::build(ambit::CoreTensor, "Bb", {nbvir, nbocc, naux});
    Bb("aig") = ints_->three_integral_block(aux_mos_, b_vir_mos_, b_occ_mos_)("gai");
    Bb_data = Bb.data();

    // aa
    for (int i = 0; i < nthreads; ++i) {
        Jab[i] = ambit::Tensor::build(CoreTensor, "J_ij", {naocc, naocc});
        JKab[i] = ambit::Tensor::build(CoreTensor, "JK_ij", {naocc, naocc});
        Da[i] = ambit::Tensor::build(CoreTensor, "Da", {naocc, naocc});
        Db[i] = ambit::Tensor::build(CoreTensor, "Db", {naocc, naocc});
    }

#pragma omp parallel for collapse(2) default(none)                                                 \
    shared(Ba_data, naocc, navir, naux, na_Qo, Jab, JKab, Da)
    for (size_t a = 0; a < navir; ++a) {
        double fock_a = Fa_[a_vir_mos_[a]];

        // grab data for index a
        double* Bai_ptr = &Ba_data[a * na_Qo];

        int thread = omp_get_thread_num();

        for (size_t b = 0; b < navir; ++b) {
            double fock_b = Fa_[a_vir_mos_[b]];

            // grab data for index b
            double* Bbj_ptr = &Ba_data[b * na_Qo];

            // compute (ia|jb) for given indices a and b
            double* Vij_ptr = Jab[thread].data().data();
            C_DGEMM('N', 'T', naocc, naocc, naux, 1.0, Bai_ptr, naux, Bbj_ptr, naux, 0.0, Vij_ptr,
                    naocc);

            Jab[thread].iterate([&](const std::vector<size_t>& i, double& value) {
                value /= Fa_[a_occ_mos_[i[0]]] + Fa_[a_occ_mos_[i[1]]] - fock_a - fock_b;
            });

            JKab[thread]("pq") = Jab[thread]("pq") - Jab[thread]("qp");
            Da[thread]("ij") -= Jab[thread]("ik") * JKab[thread]("jk");
        }
    }

    // ab
    for (int i = 0; i < nthreads; ++i) {
        Jab[i] = ambit::Tensor::build(CoreTensor, "J_ij", {naocc, nbocc});
        JKab[i] = ambit::Tensor::build(CoreTensor, "JKij", {naocc, nbocc});
    }

#pragma omp parallel for collapse(2) default(none)                                                 \
    shared(Ba_data, Bb_data, naocc, navir, na_Qo, nbocc, nbvir, nb_Qo, naux, Jab, JKab, Da, Db)
    for (size_t a = 0; a < navir; ++a) {
        double fock_a = Fa_[a_vir_mos_[a]];

        // grab data for index a
        double* Bai_ptr = &Ba_data[a * na_Qo];

        int thread = omp_get_thread_num();

        for (size_t b = 0; b < nbvir; ++b) {
            double fock_b = Fb_[b_vir_mos_[b]];

            // grab data for index b
            double* Bbj_ptr = &Bb_data[b * nb_Qo];

            // compute (ia|jb) = Ba(iQ) * Bb(jQ) for given indices a and b
            double* Vij_ptr = Jab[thread].data().data();
            C_DGEMM('N', 'T', naocc, nbocc, naux, 1.0, Bai_ptr, naux, Bbj_ptr, naux, 0.0, Vij_ptr,
                    naocc);

            Jab[thread].iterate([&](const std::vector<size_t>& i, double& value) {
                value /= Fa_[a_occ_mos_[i[0]]] + Fb_[b_occ_mos_[i[1]]] - fock_a - fock_b;
            });

            Da[thread]("ij") -= Jab[thread]("ik") * Jab[thread]("jk");
            Db[thread]("ij") -= Jab[thread]("ki") * Jab[thread]("kj");
        }
    }

    // bb
    for (int i = 0; i < nthreads; ++i) {
        Jab[i] = ambit::Tensor::build(CoreTensor, "Jab", {nbocc, nbocc});
        JKab[i] = ambit::Tensor::build(CoreTensor, "JKab", {nbocc, nbocc});
    }

#pragma omp parallel for collapse(2) default(none)                                                 \
    shared(Bb_data, nbocc, nbvir, naux, nb_Qo, Jab, JKab, Db)
    for (size_t a = 0; a < nbvir; ++a) {
        double fock_a = Fb_[b_vir_mos_[a]];

        // grab data for index a
        double* Bai_ptr = &Bb_data[a * nb_Qo];

        int thread = omp_get_thread_num();

        for (size_t b = 0; b < nbvir; ++b) {
            double fock_b = Fb_[b_vir_mos_[b]];

            // grab data for index b
            double* Bbj_ptr = &Bb_data[b * nb_Qo];

            // compute (ia|jb) for given indices a and b
            double* Vij_ptr = Jab[thread].data().data();
            C_DGEMM('N', 'T', nbocc, nbocc, naux, 1.0, Bai_ptr, naux, Bbj_ptr, naux, 0.0, Vij_ptr,
                    nbocc);

            Jab[thread].iterate([&](const std::vector<size_t>& i, double& value) {
                value /= Fb_[b_occ_mos_[i[0]]] + Fb_[b_occ_mos_[i[1]]] - fock_a - fock_b;
            });

            JKab[thread]("pq") = Jab[thread]("pq") - Jab[thread]("qp");
            Db[thread]("ij") -= Jab[thread]("ik") * JKab[thread]("jk");
        }
    }

    // add Doo contributions to D1
    for (int i = 0; i < nthreads; ++i) {
        D1.block("oo")("pq") += Da[i]("pq");
        D1.block("OO")("pq") += Db[i]("pq");
    }

    return D1;
}

void MP2_NOS::compute_df_ump2_1rdm_vv(ambit::BlockedTensor& D1) {
    int nthreads = omp_get_max_threads();

    auto na_Qv = naux_ * navir_;
    auto nb_Qv = naux_ * nbvir_;

    // batches of occupied indices
    std::vector<std::vector<size_t>> batch_aocc, batch_bocc;

    // test memory
    size_t memory_min = 4 * nthreads * nbvir_ * nbvir_;
    if ((memory_min + na_Qv + nb_Qv) * sizeof(double) > memory_) {
        outfile->Printf("\n  Error: Not enough memory for DF-UMP.");
        outfile->Printf(" Need at least %zu Bytes more!",
                        (memory_min + na_Qv + nb_Qv) * sizeof(double) - memory_);
        throw std::runtime_error("Not enough memory to run DF-MP2. Please check output.");
    } else {
        size_t max_occ = (memory_ / sizeof(double) - memory_min) / (na_Qv + nb_Qv);
        if (max_occ < naocc_) {
            outfile->Printf("\n  -> DF-UMP2 VV to be run in batches: max occ size = %zu", max_occ);
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
        Da[i] = ambit::Tensor::build(CoreTensor, "Da", {navir_, navir_});
        Db[i] = ambit::Tensor::build(CoreTensor, "Db", {nbvir_, nbvir_});
        Jab[i] = ambit::Tensor::build(CoreTensor, "Jab", {navir_, navir_});
        JKab[i] = ambit::Tensor::build(CoreTensor, "JKab", {navir_, navir_});
    }

    for (size_t i_batch = 0; i_batch < nbatches_alfa; ++i_batch) {
        const auto& i_batch_occ_mos = batch_aocc[i_batch];
        auto i_naocc = i_batch_occ_mos.size();
        auto Bi = ambit::Tensor::build(ambit::CoreTensor, "Ba", {i_naocc, navir_, naux_});
        Bi("iag") = ints_->three_integral_block(aux_mos_, i_batch_occ_mos, a_vir_mos_)("gia");
        auto& Bi_vec = Bi.data();

        // same batch
        std::vector<std::pair<size_t, size_t>> ij_pairs;
        for (size_t i = 0; i < i_naocc; ++i) {
            for (size_t j = i; j < i_naocc; ++j) {
                ij_pairs.emplace_back(i, j);
            }
        }
        size_t ij_pairs_size = ij_pairs.size();

#pragma omp parallel for default(none) shared(i_batch_occ_mos, ij_pairs_size, ij_pairs, Bi_vec, na_Qv, Jab, JKab, Da) reduction(+ : e_aa)
        for (size_t p = 0; p < ij_pairs_size; ++p) {
            int thread = omp_get_thread_num();

            size_t i = ij_pairs[p].first;
            size_t j = ij_pairs[p].second;

            auto fock_i = Fa_[i_batch_occ_mos[i]];
            auto fock_j = Fa_[i_batch_occ_mos[j]];

            double* Bia_ptr = &Bi_vec[i * na_Qv];
            double* Bjb_ptr = &Bi_vec[j * na_Qv];

            // compute (ia|jb) for given indices i and j
            C_DGEMM('N', 'T', navir_, navir_, naux_, 1.0, Bia_ptr, naux_, Bjb_ptr, naux_, 0.0,
                    Jab[thread].data().data(), navir_);

            JKab[thread]("pq") = Jab[thread]("pq") - Jab[thread]("qp");
            Jab[thread].iterate([&](const std::vector<size_t>& i, double& value) {
                value /= fock_i + fock_j - Fa_[a_vir_mos_[i[0]]] - Fa_[a_vir_mos_[i[1]]];
            });

            auto factor = (i == j) ? 0.5 : 1.0;
            e_aa += factor * Jab[thread]("pq") * JKab[thread]("pq");

            JKab[thread]("pq") = Jab[thread]("pq") - Jab[thread]("qp");
            Da[thread]("ab") += Jab[thread]("ac") * JKab[thread]("bc");
            Da[thread]("ab") += Jab[thread]("ca") * JKab[thread]("cb");
        }

        // different batch
        for (size_t j_batch = i_batch + 1; j_batch < nbatches_alfa; ++j_batch) {
            const auto& j_batch_occ_mos = batch_aocc[j_batch];
            auto j_naocc = j_batch_occ_mos.size();
            auto Bj = ambit::Tensor::build(ambit::CoreTensor, "Bb", {j_naocc, navir_, naux_});
            Bj("iag") = ints_->three_integral_block(aux_mos_, j_batch_occ_mos, a_vir_mos_)("gia");
            auto& Bj_vec = Bj.data();

#pragma omp parallel for collapse(2) default(none) shared(i_batch_occ_mos, i_naocc, Bi_vec, j_batch_occ_mos, j_naocc, Bj_vec, na_Qv, Jab, JKab, Da) reduction(+ : e_aa)
            for (size_t i = 0; i < i_naocc; ++i) {
                int thread = omp_get_thread_num();
                auto fock_i = Fa_[i_batch_occ_mos[i]];
                double* Bia_ptr = &Bi_vec[i * na_Qv];

                for (size_t j = 0; j < j_naocc; ++j) {
                    auto fock_j = Fa_[j_batch_occ_mos[j]];
                    double* Bjb_ptr = &Bj_vec[j * na_Qv];

                    // compute (ia|jb) for given indices i and j
                    C_DGEMM('N', 'T', navir_, navir_, naux_, 1.0, Bia_ptr, naux_, Bjb_ptr, naux_,
                            0.0, Jab[thread].data().data(), navir_);

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
    }

    // alpha-beta spin
    for (int i = 0; i < nthreads; ++i) {
        Jab[i] = ambit::Tensor::build(CoreTensor, "Jab", {navir_, nbvir_});
        JKab[i] = ambit::Tensor::build(CoreTensor, "JKab", {navir_, nbvir_});
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

#pragma omp parallel for collapse(2) default(none) shared(i_batch_occ_mos, i_naocc, Bi_vec, j_batch_occ_mos, j_nbocc, Bj_vec, na_Qv, nb_Qv, Jab, JKab, Da, Db) reduction(+ : e_ab)
            for (size_t i = 0; i < i_naocc; ++i) {
                int thread = omp_get_thread_num();
                auto fock_i = Fa_[i_batch_occ_mos[i]];
                double* Bia_ptr = &Bi_vec[i * na_Qv];

                for (size_t j = 0; j < j_nbocc; ++j) {
                    auto fock_j = Fb_[j_batch_occ_mos[j]];
                    double* Bjb_ptr = &Bj_vec[j * nb_Qv];

                    // compute (ia|jb) for given indices i and j
                    C_DGEMM('N', 'T', navir_, nbvir_, naux_, 1.0, Bia_ptr, naux_, Bjb_ptr, naux_,
                            0.0, Jab[thread].data().data(), navir_);

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
    }

    // beta-beta spin
    for (int i = 0; i < nthreads; ++i) {
        Jab[i] = ambit::Tensor::build(CoreTensor, "Jab", {nbvir_, nbvir_});
        JKab[i] = ambit::Tensor::build(CoreTensor, "JKab", {nbvir_, nbvir_});
    }

    for (size_t i_batch = 0; i_batch < nbatches_beta; ++i_batch) {
        const auto& i_batch_occ_mos = batch_bocc[i_batch];
        auto i_nbocc = i_batch_occ_mos.size();
        auto Bi = ambit::Tensor::build(ambit::CoreTensor, "Ba", {i_nbocc, nbvir_, naux_});
        Bi("iag") = ints_->three_integral_block(aux_mos_, i_batch_occ_mos, b_vir_mos_)("gia");
        auto& Bi_vec = Bi.data();

        // same batch
        std::vector<std::pair<size_t, size_t>> ij_pairs;
        for (size_t i = 0; i < i_nbocc; ++i) {
            for (size_t j = i; j < i_nbocc; ++j) {
                ij_pairs.emplace_back(i, j);
            }
        }
        size_t ij_pairs_size = ij_pairs.size();

#pragma omp parallel for default(none) shared(i_batch_occ_mos, ij_pairs_size, ij_pairs, Bi_vec, nb_Qv, Jab, JKab, Db) reduction(+ : e_bb)
        for (size_t p = 0; p < ij_pairs_size; ++p) {
            int thread = omp_get_thread_num();

            size_t i = ij_pairs[p].first;
            size_t j = ij_pairs[p].second;

            auto fock_i = Fb_[i_batch_occ_mos[i]];
            auto fock_j = Fb_[i_batch_occ_mos[j]];

            double* Bia_ptr = &Bi_vec[i * nb_Qv];
            double* Bjb_ptr = &Bi_vec[j * nb_Qv];

            // compute (ia|jb) for given indices i and j
            C_DGEMM('N', 'T', nbvir_, nbvir_, naux_, 1.0, Bia_ptr, naux_, Bjb_ptr, naux_, 0.0,
                    Jab[thread].data().data(), nbvir_);

            JKab[thread]("pq") = Jab[thread]("pq") - Jab[thread]("qp");
            Jab[thread].iterate([&](const std::vector<size_t>& i, double& value) {
                value /= fock_i + fock_j - Fb_[b_vir_mos_[i[0]]] - Fb_[b_vir_mos_[i[1]]];
            });

            auto factor = (i == j) ? 0.5 : 1.0;
            e_bb += factor * Jab[thread]("pq") * JKab[thread]("pq");

            JKab[thread]("pq") = Jab[thread]("pq") - Jab[thread]("qp");
            Db[thread]("ab") += Jab[thread]("ac") * JKab[thread]("bc");
            Db[thread]("ab") += Jab[thread]("ca") * JKab[thread]("cb");
        }

        // different batch
        for (size_t j_batch = i_batch + 1; j_batch < nbatches_beta; ++j_batch) {
            const auto& j_batch_occ_mos = batch_bocc[j_batch];
            auto j_nbocc = j_batch_occ_mos.size();
            auto Bj = ambit::Tensor::build(ambit::CoreTensor, "Bb", {j_nbocc, nbvir_, naux_});
            Bj("iag") = ints_->three_integral_block(aux_mos_, j_batch_occ_mos, b_vir_mos_)("gia");
            auto& Bj_vec = Bj.data();

#pragma omp parallel for collapse(2) default(none) shared(i_batch_occ_mos, i_nbocc, Bi_vec, j_batch_occ_mos, j_nbocc, Bj_vec, nb_Qv, Jab, JKab, Db) reduction(+ : e_bb)
            for (size_t i = 0; i < i_nbocc; ++i) {
                int thread = omp_get_thread_num();
                auto fock_i = Fb_[i_batch_occ_mos[i]];
                double* Bia_ptr = &Bi_vec[i * nb_Qv];

                for (size_t j = 0; j < j_nbocc; ++j) {
                    auto fock_j = Fb_[j_batch_occ_mos[j]];
                    double* Bjb_ptr = &Bj_vec[j * nb_Qv];

                    // compute (ia|jb) for given indices i and j
                    C_DGEMM('N', 'T', nbvir_, nbvir_, naux_, 1.0, Bia_ptr, naux_, Bjb_ptr, naux_,
                            0.0, Jab[thread].data().data(), nbvir_);

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
    }

    // print energy
    double e_corr = e_aa + e_ab + e_bb;
    double e_ref = scf_info_->reference_energy();
    outfile->Printf("\n\n    SCF energy                            = %20.15f", e_ref);
    outfile->Printf("\n    MP2 correlation energy (aa)           = %20.15f", e_aa);
    outfile->Printf("\n    MP2 correlation energy (ab)           = %20.15f", e_ab);
    outfile->Printf("\n    MP2 correlation energy (bb)           = %20.15f", e_bb);
    outfile->Printf("\n    MP2 correlation energy                = %20.15f", e_corr);
    outfile->Printf("\n  * MP2 total energy                      = %20.15f\n\n", e_ref + e_corr);

    // add Dvv contributions to D1
    for (int i = 0; i < nthreads; ++i) {
        D1.block("vv")("pq") += Da[i]("pq");
        D1.block("VV")("pq") += Db[i]("pq");
    }
}

void MP2_NOS::compute_df_ump2_1rdm_oo(ambit::BlockedTensor& D1) {}

void MP2_NOS::compute_df_rmp2_1rdm_vv(ambit::BlockedTensor& D1) {}

void MP2_NOS::compute_df_rmp2_1rdm_oo(ambit::BlockedTensor& D1) {}

} // namespace forte
