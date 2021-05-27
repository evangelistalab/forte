/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2020 by its authors (see COPYING, COPYING.LESSER, AUTHORS).
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

#include "ambit/blocked_tensor.h"

#include "psi4/libpsio/psio.h"
#include "psi4/libpsio/psio.hpp"

#include "psi4/libmints/matrix.h"
#include "psi4/libmints/vector.h"

#include "psi4/libpsi4util/PsiOutStream.h"
#include "helpers/helpers.h"

#include "helpers/blockedtensorfactory.h"
#include "helpers/printing.h"
#include "mp2_nos.h"

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

    BlockedTensor::set_expert_mode(true);
    BlockedTensor::reset_mo_spaces();

    /// List of alpha occupied MOs
    std::vector<size_t> a_occ_mos;
    /// List of beta occupied MOs
    std::vector<size_t> b_occ_mos;
    /// List of alpha virtual MOs
    std::vector<size_t> a_vir_mos;
    /// List of beta virtual MOs
    std::vector<size_t> b_vir_mos;

    /// Map from all the MOs to the alpha occupied
    std::map<size_t, size_t> mos_to_aocc;
    /// Map from all the MOs to the beta occupied
    std::map<size_t, size_t> mos_to_bocc;
    /// Map from all the MOs to the alpha virtual
    std::map<size_t, size_t> mos_to_avir;
    /// Map from all the MOs to the beta virtual
    std::map<size_t, size_t> mos_to_bvir;

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
            a_occ_mos.push_back(p);
            b_occ_mos.push_back(p);
        }
        for (int i = 0; i < soccpi[h]; ++i, ++p) {
            a_occ_mos.push_back(p);
            b_vir_mos.push_back(p);
        }
        for (int a = 0; a < ncmopi_[h] - corr_docc[h] - soccpi[h]; ++a, ++p) {
            a_vir_mos.push_back(p);
            b_vir_mos.push_back(p);
        }
    }
    // f r (size_t p = 0; p < a_occ_mos.size(); ++p) mos_to_aocc[a_occ_mos[p]] =
    // p;
    // for (size_t p = 0; p < b_occ_mos.size(); ++p) mos_to_bocc[b_occ_mos[p]] =
    // p;
    // for (size_t p = 0; p < a_vir_mos.size(); ++p) mos_to_avir[a_vir_mos[p]] =
    // p;
    // for (size_t p = 0; p < b_vir_mos.size(); ++p) mos_to_bvir[b_vir_mos[p]] =
    // p;

    BlockedTensor::add_mo_space("o", "ijklmn", a_occ_mos, AlphaSpin);
    BlockedTensor::add_mo_space("O", "IJKLMN", b_occ_mos, BetaSpin);
    BlockedTensor::add_mo_space("v", "abcdef", a_vir_mos, AlphaSpin);
    BlockedTensor::add_mo_space("V", "ABCDEF", b_vir_mos, BetaSpin);
    BlockedTensor::add_composite_mo_space("i", "pqrstuvwxyz", {"o", "v"});
    BlockedTensor::add_composite_mo_space("I", "PQRSTUVWXYZ", {"O", "V"});

    BlockedTensor G1 = BlockedTensor::build(CoreTensor, "G1", spin_cases({"oo"}));
    BlockedTensor D1 = BlockedTensor::build(CoreTensor, "D1", spin_cases({"oo", "vv"}));
    BlockedTensor H = BlockedTensor::build(CoreTensor, "H", spin_cases({"ii"}));
    BlockedTensor F = BlockedTensor::build(CoreTensor, "F", spin_cases({"ii"}));
    BlockedTensor V = BlockedTensor::build(CoreTensor, "V", spin_cases({"iiii"}));
    BlockedTensor T2 = BlockedTensor::build(CoreTensor, "T2", spin_cases({"oovv"}));
    BlockedTensor InvD2 = BlockedTensor::build(CoreTensor, "Inverse D2", spin_cases({"oovv"}));

    // Fill in the one-electron operator (H)
    H.iterate([&](const std::vector<size_t>& i, const std::vector<SpinType>& spin, double& value) {
        if (spin[0] == AlphaSpin)
            value = ints_->oei_a(i[0], i[1]);
        else
            value = ints_->oei_b(i[0], i[1]);
    });

    // Fill in the two-electron operator (V)
    V.iterate([&](const std::vector<size_t>& i, const std::vector<SpinType>& spin, double& value) {
        if ((spin[0] == AlphaSpin) and (spin[1] == AlphaSpin))
            value = ints_->aptei_aa(i[0], i[1], i[2], i[3]);
        if ((spin[0] == AlphaSpin) and (spin[1] == BetaSpin))
            value = ints_->aptei_ab(i[0], i[1], i[2], i[3]);
        if ((spin[0] == BetaSpin) and (spin[1] == BetaSpin))
            value = ints_->aptei_bb(i[0], i[1], i[2], i[3]);
    });

    H.iterate([&](const std::vector<size_t>& i, const std::vector<SpinType>& spin, double& value) {
        if (spin[0] == AlphaSpin)
            value = ints_->oei_a(i[0], i[1]);
        else
            value = ints_->oei_b(i[0], i[1]);
    });

    G1.iterate([&](const std::vector<size_t>& i, const std::vector<SpinType>&, double& value) {
        value = i[0] == i[1] ? 1.0 : 0.0;
    });

    D1.block("oo").iterate(
        [&](const std::vector<size_t>& i, double& value) { value = i[0] == i[1] ? 1.0 : 0.0; });

    D1.block("OO").iterate(
        [&](const std::vector<size_t>& i, double& value) { value = i[0] == i[1] ? 1.0 : 0.0; });

    // Form the Fock matrix
    F["ij"] = H["ij"];
    F["ab"] = H["ab"];
    F["pq"] += V["prqs"] * G1["sr"];
    F["pq"] += V["pRqS"] * G1["SR"];

    F["IJ"] += H["IJ"];
    F["AB"] += H["AB"];
    F["PQ"] += V["rPsQ"] * G1["sr"];
    F["PQ"] += V["PRQS"] * G1["SR"];

    size_t ncmo_ = mo_space_info_->size("CORRELATED");
    std::vector<double> Fa(ncmo_);
    std::vector<double> Fb(ncmo_);

    F.iterate([&](const std::vector<size_t>& i, const std::vector<SpinType>& spin, double& value) {
        if (spin[0] == AlphaSpin and (i[0] == i[1])) {
            Fa[i[0]] = value;
        }
        if (spin[0] == BetaSpin and (i[0] == i[1])) {
            Fb[i[0]] = value;
        }
    });

    InvD2.iterate(
        [&](const std::vector<size_t>& i, const std::vector<SpinType>& spin, double& value) {
            if ((spin[0] == AlphaSpin) and (spin[1] == AlphaSpin)) {
                value = 1.0 / (Fa[i[0]] + Fa[i[1]] - Fa[i[2]] - Fa[i[3]]);
            } else if ((spin[0] == AlphaSpin) and (spin[1] == BetaSpin)) {
                value = 1.0 / (Fa[i[0]] + Fb[i[1]] - Fa[i[2]] - Fb[i[3]]);
            } else if ((spin[0] == BetaSpin) and (spin[1] == BetaSpin)) {
                value = 1.0 / (Fb[i[0]] + Fb[i[1]] - Fb[i[2]] - Fb[i[3]]);
            }
        });

    T2["ijab"] = V["ijab"] * InvD2["ijab"];
    T2["iJaB"] = V["iJaB"] * InvD2["iJaB"];
    T2["IJAB"] = V["IJAB"] * InvD2["IJAB"];

    double Eaa = 0.25 * T2["ijab"] * V["ijab"];
    double Eab = T2["iJaB"] * V["iJaB"];
    double Ebb = 0.25 * T2["IJAB"] * V["IJAB"];

    double mp2_correlation_energy = Eaa + Eab + Ebb;
    double ref_energy = scf_info_->reference_energy();
    outfile->Printf("\n\n    SCF energy                            = %20.15f", ref_energy);
    outfile->Printf("\n    MP2 correlation energy                = %20.15f",
                    mp2_correlation_energy);
    outfile->Printf("\n  * MP2 total energy                      = %20.15f\n\n",
                    ref_energy + mp2_correlation_energy);

    D1["ab"] += 0.5 * T2["ijbc"] * T2["ijac"];
    D1["ab"] += 1.0 * T2["iJbC"] * T2["iJaC"];

    D1["AB"] += 0.5 * T2["IJCB"] * T2["IJCA"];
    D1["AB"] += 1.0 * T2["iJcB"] * T2["iJcA"];

    D1["ij"] -= 0.5 * T2["ikab"] * T2["jkab"];
    D1["ij"] -= 1.0 * T2["iKaB"] * T2["jKaB"];

    D1["IJ"] -= 0.5 * T2["IKAB"] * T2["JKAB"];
    D1["IJ"] -= 1.0 * T2["kIaB"] * T2["kJaB"];

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
    // This will suggested a restricted_docc and a active
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

    Ua_.reset(new psi::Matrix("Ua", nmopi, nmopi));
    Ub_.reset(new psi::Matrix("Ub", nmopi, nmopi));

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
} // namespace forte
