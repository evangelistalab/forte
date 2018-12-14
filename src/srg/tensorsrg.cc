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

#include <exception>

#include "boost/numeric/odeint.hpp"

#include "psi4/libmints/molecule.h"
#include "psi4/libmints/wavefunction.h"

#include "psi4/libpsi4util/PsiOutStream.h"
#include "psi4/psi4-dec.h"

#include "helpers/mo_space_info.h"
#include "srg/tensorsrg.h"

using namespace psi;
using namespace ambit;

namespace psi {
namespace forte {

TensorSRG::TensorSRG(std::shared_ptr<Wavefunction> wfn, Options& options,
                     std::shared_ptr<ForteIntegrals> ints,
                     std::shared_ptr<MOSpaceInfo> mo_space_info)
    : MethodBase(wfn, options, ints, mo_space_info) {
    startup();
}

TensorSRG::~TensorSRG() { cleanup(); }

void TensorSRG::startup() {
    outfile->Printf("\n\n      --------------------------------------");
    outfile->Printf("\n          Similarity Renormalization Group");
    outfile->Printf("\n                tensor-based code");
    outfile->Printf("\n");
    outfile->Printf("\n                Version 0.2.2");
    outfile->Printf("\n");
    outfile->Printf("\n       written by Francesco A. Evangelista");
    outfile->Printf("\n      --------------------------------------\n");
    outfile->Printf("\n      Debug level = %d", debug_);
    outfile->Printf("\n      Print level = %d\n", print_);

    //    BlockedTensor::print_mo_spaces();

    dsrg_power_ = options_.get_double("DSRG_POWER");

    outfile->Printf("\n      Energy convergence = %e\n", options_.get_double("E_CONVERGENCE"));

    S1 = BlockedTensor::build(tensor_type_, "S1", spin_cases({"ov"}));
    S2 = BlockedTensor::build(tensor_type_, "S2", spin_cases({"oovv"}));
    DS1 = BlockedTensor::build(tensor_type_, "S1", spin_cases({"ov"}));
    DS2 = BlockedTensor::build(tensor_type_, "S2", spin_cases({"oovv"}));
    R1 = BlockedTensor::build(tensor_type_, "R1", spin_cases({"ov"}));
    R2 = BlockedTensor::build(tensor_type_, "R2", spin_cases({"oovv"}));
    Hbar1 = BlockedTensor::build(tensor_type_, "Hbar1", spin_cases({"ii"}));
    Hbar2 = BlockedTensor::build(tensor_type_, "Hbar2", spin_cases({"iiii"}));
    eta1 = BlockedTensor::build(tensor_type_, "eta1", spin_cases({"ii"}));
    eta2 = BlockedTensor::build(tensor_type_, "eta2", spin_cases({"iiii"}));
    g1 = BlockedTensor::build(tensor_type_, "g1", spin_cases({"ii"}));
    g2 = BlockedTensor::build(tensor_type_, "g2", spin_cases({"iiii"}));
    O1 = BlockedTensor::build(tensor_type_, "O1", spin_cases({"ii"}));
    O2 = BlockedTensor::build(tensor_type_, "O2", spin_cases({"iiii"}));
    C1 = BlockedTensor::build(tensor_type_, "C1", spin_cases({"ii"}));
    C2 = BlockedTensor::build(tensor_type_, "C2", spin_cases({"iiii"}));
    D1 = BlockedTensor::build(tensor_type_, "D1", spin_cases({"ov"}));
    D2 = BlockedTensor::build(tensor_type_, "D2", spin_cases({"oovv"}));
    RInvD1 = BlockedTensor::build(tensor_type_, "InvRD1", spin_cases({"ov"}));
    RInvD2 = BlockedTensor::build(tensor_type_, "InvRD2", spin_cases({"oovv"}));
    I_ioiv = BlockedTensor::build(tensor_type_, "I_ioiv", spin_cases({"ioiv"}));

    size_t ncmo_ = mo_space_info_->size("CORRELATED");
    std::vector<double> Fa(ncmo_);
    std::vector<double> Fb(ncmo_);

    F.iterate([&](const std::vector<size_t>& i, const std::vector<SpinType>& spin, double& value) {
        if (spin[0] == AlphaSpin and (i[0] == i[1]))
            Fa[i[0]] = value;
        if (spin[0] == BetaSpin and (i[0] == i[1]))
            Fb[i[0]] = value;
    });

    double srg_s = options_.get_double("DSRG_S");

    D1.iterate([&](const std::vector<size_t>& i, const std::vector<SpinType>& spin, double& value) {
        if (spin[0] == AlphaSpin) {
            value = Fa[i[0]] - Fa[i[1]];
        } else if (spin[0] == BetaSpin) {
            value = Fb[i[0]] - Fb[i[1]];
        }
    });

    D2.iterate([&](const std::vector<size_t>& i, const std::vector<SpinType>& spin, double& value) {
        if ((spin[0] == AlphaSpin) and (spin[1] == AlphaSpin)) {
            value = Fa[i[0]] + Fa[i[1]] - Fa[i[2]] - Fa[i[3]];
        } else if ((spin[0] == AlphaSpin) and (spin[1] == BetaSpin)) {
            value = Fa[i[0]] + Fb[i[1]] - Fa[i[2]] - Fb[i[3]];
        } else if ((spin[0] == BetaSpin) and (spin[1] == BetaSpin)) {
            value = Fb[i[0]] + Fb[i[1]] - Fb[i[2]] - Fb[i[3]];
        }
    });

    RInvD1.iterate(
        [&](const std::vector<size_t>& i, const std::vector<SpinType>& spin, double& value) {
            if (spin[0] == AlphaSpin) {
                value = one_minus_exp_div_x(srg_s, Fa[i[0]] - Fa[i[1]], dsrg_power_);
            } else if (spin[0] == BetaSpin) {
                value = one_minus_exp_div_x(srg_s, Fb[i[0]] - Fb[i[1]], dsrg_power_);
            }
        });

    RInvD2.iterate([&](const std::vector<size_t>& i, const std::vector<SpinType>& spin,
                       double& value) {
        if ((spin[0] == AlphaSpin) and (spin[1] == AlphaSpin)) {
            value =
                one_minus_exp_div_x(srg_s, Fa[i[0]] + Fa[i[1]] - Fa[i[2]] - Fa[i[3]], dsrg_power_);
        } else if ((spin[0] == AlphaSpin) and (spin[1] == BetaSpin)) {
            value =
                one_minus_exp_div_x(srg_s, Fa[i[0]] + Fb[i[1]] - Fa[i[2]] - Fb[i[3]], dsrg_power_);
        } else if ((spin[0] == BetaSpin) and (spin[1] == BetaSpin)) {
            value =
                one_minus_exp_div_x(srg_s, Fb[i[0]] + Fb[i[1]] - Fb[i[2]] - Fb[i[3]], dsrg_power_);
        }
    });
}

void TensorSRG::cleanup() { print_timings(); }

double TensorSRG::compute_mp2_guess() {
    S2["ijab"] = V["ijab"] * InvD2["ijab"];
    S2["iJaB"] = V["iJaB"] * InvD2["iJaB"];
    S2["IJAB"] = V["IJAB"] * InvD2["IJAB"];

    double Eaa = 0.25 * S2["ijab"] * V["ijab"];
    double Eab = S2["iJaB"] * V["iJaB"];
    double Ebb = 0.25 * S2["IJAB"] * V["IJAB"];

    double mp2_correlation_energy = Eaa + Eab + Ebb;
    double ref_energy = reference_energy();
    outfile->Printf("\n\n    SCF energy                            = %20.15f", ref_energy);
    outfile->Printf("\n    SRG-PT2 correlation energy            = %20.15f",
                    mp2_correlation_energy);
    outfile->Printf("\n  * SRG-PT2 total energy                  = %20.15f\n",
                    ref_energy + mp2_correlation_energy);

    //    outfile->Printf("\n\n    SCF energy                            =
    //    %20.15f",E0_);
    //    outfile->Printf("\n\n    SCF energy                            =
    //    %20.15f",E0_);
    //    outfile->Printf("\n    MP2 correlation energy                =
    //    %20.15f",mp2_correlation_energy);
    //    outfile->Printf("\n  * MP2 total energy                      =
    //    %20.15f\n",E0_ + mp2_correlation_energy);
    return E0_ + mp2_correlation_energy;
}

double TensorSRG::compute_mp2_guess_driven_srg() {

    S2.zero();
    S2["ijab"] = V["ijab"] * RInvD2["ijab"];
    S2["iJaB"] = V["iJaB"] * RInvD2["iJaB"];
    S2["IJAB"] = V["IJAB"] * RInvD2["IJAB"];

    double Eaa = 0.25 * S2["ijab"] * V["ijab"];
    double Eab = S2["iJaB"] * V["iJaB"];
    double Ebb = 0.25 * S2["IJAB"] * V["IJAB"];

    double mp2_correlation_energy = Eaa + Eab + Ebb;
    double ref_energy = reference_energy();
    outfile->Printf("\n\n    SCF energy                            = %20.15f", ref_energy);
    outfile->Printf("\n    SRG-PT2 correlation energy            = %20.15f",
                    mp2_correlation_energy);
    outfile->Printf("\n  * SRG-PT2 total energy                  = %20.15f\n",
                    ref_energy + mp2_correlation_energy);
    return ref_energy + mp2_correlation_energy;
}

double one_minus_exp_div_x(double s, double x, double power) {
    return ((1.0 - std::exp(-s * std::pow(x, power))) / x);
}

double TensorSRG::compute_energy() {
    compute_ct_energy();
    //    if(options_.get_str("SRG_MODE") == "SRG"){
    //        compute_srg_energy();
    //    }else if(options_.get_str("SRG_MODE") == "CT"){
    //        return compute_ct_energy();
    //    }else if(options_.get_str("SRG_MODE") == "DSRG"){
    ////        compute_ct_energy();
    //    }
    return 0.0;
}

void TensorSRG::transfer_integrals() {
    // Scalar term
    double scalar0 = E0_ + Hbar0;
    double scalar1 = 0.0;
    double scalar2 = 0.0;

    Hbar1.block("oo").citerate([&](const std::vector<size_t>& i, const double& value) {
        if (i[0] == i[1])
            scalar1 -= value;
    });

    Hbar1.block("OO").citerate([&](const std::vector<size_t>& i, const double& value) {
        if (i[0] == i[1])
            scalar1 -= value;
    });

    Hbar2.block("oooo").citerate([&](const std::vector<size_t>& i, const double& value) {
        if ((i[0] == i[2]) and (i[1] == i[3]))
            scalar2 += 0.5 * value;
    });

    Hbar2.block("oOoO").citerate([&](const std::vector<size_t>& i, const double& value) {
        if ((i[0] == i[2]) and (i[1] == i[3]))
            scalar2 += value;
    });

    Hbar2.block("OOOO").citerate([&](const std::vector<size_t>& i, const double& value) {
        if ((i[0] == i[2]) and (i[1] == i[3]))
            scalar2 += 0.5 * value;
    });

    double scalar = scalar0 + scalar1 + scalar2 -
                    molecule_->nuclear_repulsion_energy(ref_wfn_->get_dipole_field_strength());
    outfile->Printf("\n  The Hamiltonian electronic scalar term (normal "
                    "ordered wrt the true vacuum");
    outfile->Printf("\n  E0 = %20.12f", scalar);

    O1["pq"] = Hbar1["pq"];
    O1["PQ"] = Hbar1["PQ"];
    O1["pq"] -= Hbar2["prqs"] * G1["rs"];
    O1["pq"] -= Hbar2["pRqS"] * G1["RS"];
    O1["PQ"] -= Hbar2["rPsQ"] * G1["rs"];
    O1["PQ"] -= Hbar2["PRQS"] * G1["RS"];

    outfile->Printf("\n  Updating all the integrals");
    ints_->set_scalar(scalar);
    O1.citerate(
        [&](const std::vector<size_t>& i, const std::vector<SpinType>& spin, const double& value) {
            if ((spin[0] == AlphaSpin) and (spin[1] == AlphaSpin)) {
                ints_->set_oei(i[0], i[1], value, true);
            }
            if ((spin[0] == BetaSpin) and (spin[1] == BetaSpin)) {
                ints_->set_oei(i[0], i[1], value, false);
            }
        });
    Hbar2.citerate(
        [&](const std::vector<size_t>& i, const std::vector<SpinType>& spin, const double& value) {
            if ((spin[0] == AlphaSpin) and (spin[1] == AlphaSpin) and (spin[2] == AlphaSpin) and
                (spin[3] == AlphaSpin)) {
                ints_->set_tei(i[0], i[1], i[2], i[3], value, true, true);
            }
            if ((spin[0] == AlphaSpin) and (spin[1] == BetaSpin) and (spin[2] == AlphaSpin) and
                (spin[3] == BetaSpin)) {
                ints_->set_tei(i[0], i[1], i[2], i[3], value, true, false);
            }
            if ((spin[0] == BetaSpin) and (spin[1] == BetaSpin) and (spin[2] == BetaSpin) and
                (spin[3] == BetaSpin)) {
                ints_->set_tei(i[0], i[1], i[2], i[3], value, false, false);
            }
        });

    // As a test compute the current CT energy
    double Esth = scalar;
    O1.block("oo").citerate([&](const std::vector<size_t>& i, const double& value) {
        if (i[0] == i[1])
            Esth += value;
    });
    O1.block("OO").citerate([&](const std::vector<size_t>& i, const double& value) {
        if (i[0] == i[1])
            Esth += value;
    });
    Hbar2.block("oooo").citerate([&](const std::vector<size_t>& i, const double& value) {
        if (i[0] == i[1])
            Esth += 0.5 * value;
    });
    Hbar2.block("oOoO").citerate([&](const std::vector<size_t>& i, const double& value) {
        if (i[0] == i[1])
            Esth += value;
    });
    Hbar2.block("OOOO").citerate([&](const std::vector<size_t>& i, const double& value) {
        if (i[0] == i[1])
            Esth += 0.5 * value;
    });

    outfile->Printf(
        "\n  <H> = %24.12f",
        Esth + molecule()->nuclear_repulsion_energy(ref_wfn_->get_dipole_field_strength()));
}
}
} // EndNamespaces
