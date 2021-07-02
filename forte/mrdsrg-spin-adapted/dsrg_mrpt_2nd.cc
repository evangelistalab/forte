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

#include <utility>

#include "psi4/libpsi4util/PsiOutStream.h"
#include "helpers/timer.h"
#include "helpers/printing.h"

#include "dsrg_mrpt.h"

using namespace psi;

namespace forte {

double DSRG_MRPT::compute_energy_pt2() {
    local_timer DSRG_energy;
    print_h2("Computing DSRG-MRPT2 Energy");

    // Compute effective integrals
    renormalize_V_E2nd();
    renormalize_F_E2nd();

    // Compute DSRG-MRPT2 correlation energy
    double Etemp = 0.0;
    double Ecorr = 0.0;
    double Etotal = 0.0;
    std::vector<std::pair<std::string, double>> energy;
    energy.push_back({"E0 (reference)", Eref_});

    H1_T1_C0(F_, T1_, 1.0, Ecorr);
    energy.push_back({"<[F, T1]>", Ecorr - Etemp});
    Etemp = Ecorr;

    H1_T2_C0(F_, T2_, 1.0, Ecorr);
    energy.push_back({"<[F, T2]>", Ecorr - Etemp});
    Etemp = Ecorr;

    H2_T1_C0(V_, T1_, 1.0, Ecorr);
    energy.push_back({"<[V, T1]>", Ecorr - Etemp});
    Etemp = Ecorr;

    H2_T2_C0_L1(V_, T2_, 1.0, Ecorr, false);
    energy.push_back({"<[V, T2]> L1", Ecorr - Etemp});
    Etemp = Ecorr;

    H2_T2_C0_L2(V_, T2_, 1.0, Ecorr);
    energy.push_back({"<[V, T2]> L2", Ecorr - Etemp});
    Etemp = Ecorr;

    H2_T2_C0_L3(V_, T2_, 1.0, Ecorr);
    energy.push_back({"<[V, T2]> L3", Ecorr - Etemp});
    Etemp = Ecorr;

    Etotal = Ecorr + Eref_;
    energy.push_back({"DSRG-MRPT2 correlation energy", Ecorr});
    energy.push_back({"DSRG-MRPT2 total energy", Etotal});

    // Print energy summary
    print_h2("DSRG-MRPT2 Energy Summary");
    for (const auto& str_dim : energy) {
        outfile->Printf("\n    %-30s = %23.15f", std::get<0>(str_dim).c_str(),
                        std::get<1>(str_dim));
    }

    outfile->Printf("\n\n  DSRG-MRPT2 energy took %10.3f s.", DSRG_energy.get());
    return Ecorr;
}

void DSRG_MRPT::renormalize_V_E2nd() {
    local_timer timer;
    std::string str = "Renormalizing two-electron integrals";
    outfile->Printf("\n    %-40s ...", str.c_str());

    // Note: we directly modify the two-electron integrals.
    BT_scaled_by_Rplus1(V_);

    outfile->Printf("  Done. Timing %10.3f s", timer.get());
}

void DSRG_MRPT::renormalize_F_E2nd() {
    local_timer timer;
    std::string str = "Renormalizing the Fock matrix";
    outfile->Printf("\n    %-40s ...", str.c_str());

    // Note: we directly modify the Fock matrix and keep the aa block unchanged.
    // Same strategy when forming T1 amplitudes.
    ambit::BlockedTensor F1st = ambit::BlockedTensor::build(tensor_type_, "Temp", {"hv", "ca"});
    for (const auto& block : F1st.block_labels()) {
        F1st.block(block)("pq") = F_.block(block)("pq");
    }

    // temp BlockedTensor for contraction between L1 and T2
    ambit::BlockedTensor temp = ambit::BlockedTensor::build(tensor_type_, "Temp", {"aa"});
    temp["xu"] = 0.5 * L1_["xu"];

    temp.iterate([&](const std::vector<size_t>& i, const std::vector<SpinType>&, double& value) {
        value *= Fdiag_[i[0]] - Fdiag_[i[1]];
    });

    F1st["ie"] += 2.0 * T2_["iuex"] * temp["xu"];
    F1st["ie"] -= T2_["iuxe"] * temp["xu"];
    F1st["my"] += 2.0 * T2_["muyx"] * temp["xu"];
    F1st["my"] -= T2_["muxy"] * temp["xu"];

    // scale F1st by R
    F1st.iterate([&](const std::vector<size_t>& i, const std::vector<SpinType>&, double& value) {
        if (std::fabs(value) > 1.0e-15) {
            value *= dsrg_source_->compute_renormalized(Fdiag_[i[0]] - Fdiag_[i[1]]);
        } else {
            value = 0.0; // ignore all noise
        }
    });

    // add F1st to F
    for (const auto& block : F1st.block_labels()) {
        F_.block(block)("pq") += F1st.block(block)("pq");
    }

    outfile->Printf("  Done. Timing %10.3f s", timer.get());
}

void DSRG_MRPT::BT_scaled_by_D(BlockedTensor& BT) {
    if (BT.rank() == 4) {
        BT.iterate([&](const std::vector<size_t>& i, const std::vector<SpinType>&, double& value) {
            if (std::fabs(value) > 1.0e-15) {
                value *= 1.0 / (Fdiag_[i[0]] + Fdiag_[i[1]] - Fdiag_[i[2]] - Fdiag_[i[3]]);
            } else {
                value = 0.0; // ignore all noise
            }
        });
    } else if (BT.rank() == 2) {
        BT.iterate([&](const std::vector<size_t>& i, const std::vector<SpinType>&, double& value) {
            if (std::fabs(value) > 1.0e-15) {
                value *= 1.0 / (Fdiag_[i[0]] - Fdiag_[i[1]]);
            } else {
                value = 0.0; // ignore all noise
            }
        });
    } else {
        outfile->Printf("\n  Wrong rank when using function BT_scaled_by_D for "
                        "BlockedTensor %s",
                        BT.name().c_str());
        throw psi::PSIEXCEPTION("Wrong rank when using function BT_scaled_by_D!");
    }
}

void DSRG_MRPT::BT_scaled_by_Rplus1(BlockedTensor& BT) {
    if (BT.rank() == 4) {
        BT.iterate([&](const std::vector<size_t>& i, const std::vector<SpinType>&, double& value) {
            if (std::fabs(value) > 1.0e-15) {
                value *= 1.0 +
                         dsrg_source_->compute_renormalized(Fdiag_[i[0]] + Fdiag_[i[1]] -
                                                            Fdiag_[i[2]] - Fdiag_[i[3]]);
            } else {
                value = 0.0; // ignore all noise
            }
        });
    } else if (BT.rank() == 2) {
        BT.iterate([&](const std::vector<size_t>& i, const std::vector<SpinType>&, double& value) {
            if (std::fabs(value) > 1.0e-15) {
                value *= 1.0 + dsrg_source_->compute_renormalized(Fdiag_[i[0]] - Fdiag_[i[1]]);
            } else {
                value = 0.0; // ignore all noise
            }
        });
    } else {
        outfile->Printf("\n  Wrong rank when using function BT_scaled_by_R for "
                        "BlockedTensor %s",
                        BT.name().c_str());
        throw psi::PSIEXCEPTION("Wrong rank when using function BT_scaled_by_R!");
    }
}

void DSRG_MRPT::BT_scaled_by_RD(BlockedTensor& BT) {
    if (BT.rank() == 4) {
        BT.iterate([&](const std::vector<size_t>& i, const std::vector<SpinType>&, double& value) {
            if (std::fabs(value) > 1.0e-15) {
                value *= dsrg_source_->compute_renormalized_denominator(
                    Fdiag_[i[0]] + Fdiag_[i[1]] - Fdiag_[i[2]] - Fdiag_[i[3]]);
            } else {
                value = 0.0; // ignore all noise
            }
        });
    } else if (BT.rank() == 2) {
        BT.iterate([&](const std::vector<size_t>& i, const std::vector<SpinType>&, double& value) {
            if (std::fabs(value) > 1.0e-15) {
                value *=
                    dsrg_source_->compute_renormalized_denominator(Fdiag_[i[0]] - Fdiag_[i[1]]);
            } else {
                value = 0.0; // ignore all noise
            }
        });
    } else {
        outfile->Printf("\n  Wrong rank when using function BT_scaled_by_RD "
                        "for BlockedTensor %s",
                        BT.name().c_str());
        throw psi::PSIEXCEPTION("Wrong rank when using function BT_scaled_by_RD!");
    }
}
} // namespace forte

