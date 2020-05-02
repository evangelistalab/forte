/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2020 by its authors (see COPYING, COPYING.LESSER,
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

#include "psi4/libpsi4util/PsiOutStream.h"

#include "helpers/timer.h"
#include "helpers/printing.h"
#include "sa_mrpt3.h"

using namespace psi;

namespace forte {

SA_MRPT3::SA_MRPT3(RDMs rdms, std::shared_ptr<SCFInfo> scf_info,
                   std::shared_ptr<ForteOptions> options, std::shared_ptr<ForteIntegrals> ints,
                   std::shared_ptr<MOSpaceInfo> mo_space_info)
    : SA_DSRGPT(rdms, scf_info, options, ints, mo_space_info) {

    print_method_banner({"MR-DSRG Third-Order Perturbation Theory"});
    read_options();
    print_options();
    check_memory();
    startup();
}

void SA_MRPT3::startup() {
    // test semi-canonical
    if (!semi_canonical_) {
        outfile->Printf("\n  Orbital invariant formalism will be employed for DSRG-MRPT3.");
        U_ = ambit::BlockedTensor::build(tensor_type_, "U", {"gg"});
        Fdiag_ = diagonalize_Fock_diagblocks(U_);
    }

    // prepare integrals
    build_ints();

    // initialize tensors for amplitudes
    init_amps();
}

void SA_MRPT3::build_ints() {
    timer t("Initialize integrals");

    // prepare two-electron integrals or three-index B
    V_ = BTF_->build(tensor_type_, "V", {"pphh"});
    if (eri_df_) {
        B_ = BTF_->build(tensor_type_, "B 3-idx", {"Lgg"});
        fill_three_index_ints(B_);
        V_["pqrs"] = B_["gpr"] * B_["gqs"];
    } else {
        for (const std::string& block : V_.block_labels()) {
            auto mo_to_index = BTF_->get_mo_to_index();
            std::vector<size_t> i0 = mo_to_index[block.substr(0, 1)];
            std::vector<size_t> i1 = mo_to_index[block.substr(1, 1)];
            std::vector<size_t> i2 = mo_to_index[block.substr(2, 1)];
            std::vector<size_t> i3 = mo_to_index[block.substr(3, 1)];
            auto Vblock = ints_->aptei_ab_block(i0, i1, i2, i3);
            V_.block(block).copy(Vblock);
        }
    }

    // prepare Hbar
    if (form_Hbar_) {
        Hbar1_ = BTF_->build(tensor_type_, "1-body Hbar", {"aa"});
        Hbar2_ = BTF_->build(tensor_type_, "2-body Hbar", {"aaaa"});
        Hbar1_["uv"] = F_["uv"];
        Hbar2_["uvxy"] = V_["uvxy"];
    }

    t.stop();
}

void SA_MRPT3::init_amps() {
    timer t("Initialize T1 and T2");
    T1_ = BTF_->build(tensor_type_, "T1 Amplitudes", {"hp"});
    T2_ = BTF_->build(tensor_type_, "T2 Amplitudes", {"hhpp"});
    S2_ = BTF_->build(tensor_type_, "S2 Amplitudes", {"hhpp"});
    t.stop();
}

void SA_MRPT3::check_memory() {
    dsrg_mem_.add_entry("2-electron (4-index) integrals", {"pphh"});
    if (eri_df_) {
        dsrg_mem_.add_entry("3-index integrals", {"Lgg"});
    }

    dsrg_mem_.add_entry("T1 and T2 cluster amplitudes", {"hp", "hhpp", "hhpp"});

    if (form_Hbar_) {
        dsrg_mem_.add_entry("1- and 2-body Hbar", {"aa", "aaaa"});
    }

    dsrg_mem_.add_entry("Global 1- and 2-body intermediates", {"hp", "hhpp", "hhpp"});

    auto local1 = dsrg_mem_.compute_memory({"hp", "hhpp"}, 3);
    dsrg_mem_.add_entry("Local intermediates (energy part 1)", local1, false);

    auto local2 = dsrg_mem_.compute_memory({"ph", "pphh"});
    if (!eri_df_) {
        local2 += dsrg_mem_.compute_memory({"gggg"});
    }
    dsrg_mem_.add_entry("Local intermediates (energy part 2)", local2, false);

    auto local_comm = dsrg_mem_.compute_memory({"pphh"});
    dsrg_mem_.add_entry("Local intermediates for commutators", local_comm, false);

    dsrg_mem_.print("DSRG-MRPT3");
}

double SA_MRPT3::compute_energy() {
    // build amplitudes
    compute_t2_full();
    compute_t1();
    analyze_amplitudes("First-Order", T1_, T2_);

    // compute energy, order matters!!!
    double Ept3_1 = compute_energy_pt3_1();
    double Ept2 = compute_energy_pt2();
    double Ept3_2 = compute_energy_pt3_2(); // put 2nd-order amps to T_
    double Ept3_3 = compute_energy_pt3_3();
    double Ept3 = Ept3_1 + Ept3_2 + Ept3_3;
    Hbar0_ = Ept3 + Ept2;
    double Etotal = Hbar0_ + Eref_;

    // printing
    std::vector<std::pair<std::string, double>> energy;
    energy.push_back({"Reference energy", Eref_});
    energy.push_back({"2nd-order correlation energy", Ept2});
    energy.push_back({"3rd-order correlation energy part 1", Ept3_1});
    energy.push_back({"3rd-order correlation energy part 2", Ept3_2});
    energy.push_back({"3rd-order correlation energy part 3", Ept3_3});
    energy.push_back({"3rd-order correlation energy", Ept3});
    energy.push_back({"DSRG-MRPT3 correlation energy", Hbar0_});
    energy.push_back({"DSRG-MRPT3 total energy", Etotal});

    print_h2("DSRG-MRPT3 Energy Summary");
    for (const auto& str_dim : energy) {
        outfile->Printf("\n    %-40s = %22.15f", str_dim.first.c_str(), str_dim.second);
    }
    outfile->Printf("\n\n    Notes:");
    outfile->Printf("\n      3rd-order energy part 1: -1.0 / 12.0 * [[[H0th, "
                    "A1st], A1st], A1st]");
    outfile->Printf("\n      3rd-order energy part 2: 0.5 * [H1st + Hbar1st, A2nd]");
    outfile->Printf("\n      3rd-order energy part 3: 0.5 * [Hbar2nd, A1st]");
    outfile->Printf("\n      Hbar1st = H1st + [H0th, A1st]");
    outfile->Printf("\n      Hbar2nd = 0.5 * [H1st + Hbar1st, A1st] + [H0th, A2nd]");


    return Etotal;
}

double SA_MRPT3::compute_energy_pt2() {
    print_h2("Computing 2nd-Order Correlation Energy");

    // Compute effective integrals
    renormalize_integrals(true);

    // Compute DSRG-MRPT2 correlation energy
    double Ept2 = 0.0;
    local_timer t1;
    std::string str = "Computing 2nd-order energy";
    outfile->Printf("\n    %-40s ...", str.c_str());
    H1_T1_C0(F_, T1_, 1.0, Ept2);
    H1_T2_C0(F_, T2_, 1.0, Ept2);
    H2_T1_C0(V_, T1_, 1.0, Ept2);
    H2_T2_C0(V_, T2_, S2_, 1.0, Ept2);
    outfile->Printf("  Done. Timing %10.3f s", t1.get());

    // relax reference
    if (form_Hbar_) {
        local_timer t2;
        str = "Computing integrals for ref. relaxation";
        outfile->Printf("\n    %-40s ...", str.c_str());

        H_A_Ca(F_, V_, T1_, T2_, S2_, 0.5, Hbar1_, Hbar2_);

        outfile->Printf("  Done. Timing %10.3f s", t2.get());
    }

    return Ept2;
}

double SA_MRPT3::compute_energy_pt3_1() {
    print_h2("Computing 3rd-Order Energy Contribution (1/3)");

    local_timer t1;
    std::string str = "Computing 3rd-order energy (1/3)";
    outfile->Printf("\n    %-40s ...", str.c_str());

    // 1- and 2-body -[[H0th,A1st],A1st]
    O1_ = BTF_->build(tensor_type_, "O1 PT3 1/3", od_one_labels_ph());
    O2_ = BTF_->build(tensor_type_, "O2 PT3 1/3", od_two_labels_pphh());

    // declare other tensors
    BlockedTensor C1, C2, temp1, temp2;

    // compute -[H0th,A1st] = Delta * T and save to C1 and C2
    C1 = BTF_->build(tensor_type_, "C1", od_one_labels());
    C2 = BTF_->build(tensor_type_, "C2", od_two_labels());
    H1_T1_C1(F0th_, T1_, -1.0, C1);
    H1_T2_C1(F0th_, T2_, -1.0, C1);
    H1_T2_C2(F0th_, T2_, -1.0, C2);

    C1["ai"] += C1["ia"];
    C2["abij"] += C2["ijab"];

    // compute -[[H0th,A1st],A1st]
    // - Step 1: ph and pphh part
    H1_T1_C1(C1, T1_, 1.0, O1_);
    H1_T2_C1(C1, T2_, 1.0, O1_);
    H2_T1_C1(C2, T1_, 1.0, O1_);
    H2_T2_C1(C2, T2_, S2_, 1.0, O1_);
    H1_T2_C2(C1, T2_, 1.0, O2_);
    H2_T1_C2(C2, T1_, 1.0, O2_);
    H2_T2_C2(C2, T2_, S2_, 1.0, O2_);

    // - Step 2: hp and hhpp part
    temp1 = BTF_->build(tensor_type_, "temp1 pt3 1/3", od_one_labels_hp());
    temp2 = BTF_->build(tensor_type_, "temp2 pt3 1/3", od_two_labels_hhpp());
    H1_T1_C1(C1, T1_, 1.0, temp1);
    H1_T2_C1(C1, T2_, 1.0, temp1);
    H2_T1_C1(C2, T1_, 1.0, temp1);
    H2_T2_C1(C2, T2_, S2_, 1.0, temp1);
    H1_T2_C2(C1, T2_, 1.0, temp2);
    H2_T1_C2(C2, T1_, 1.0, temp2);
    H2_T2_C2(C2, T2_, S2_, 1.0, temp2);

    // - Step 3: add hp and hhpp to O1 and O2
    O1_["ai"] += temp1["ia"];
    O2_["abij"] += temp2["ijab"];

    // compute -1.0 / 12.0 * [[[H0th,A1st],A1st],A1st]
    double Ereturn = 0.0;
    double factor = 1.0 / 6.0;
    H1_T1_C0(O1_, T1_, factor, Ereturn);
    H1_T2_C0(O1_, T2_, factor, Ereturn);
    H2_T1_C0(O2_, T1_, factor, Ereturn);
    H2_T2_C0(O2_, T2_, S2_, factor, Ereturn);

    outfile->Printf("  Done. Timing %10.3f s", t1.get());

    if (form_Hbar_) {
        local_timer t2;
        str = "Computing integrals for ref. relaxation";
        outfile->Printf("\n    %-40s ...", str.c_str());

        factor = 1.0 / 12.0;
        H_A_Ca(O1_, O2_, T1_, T2_, S2_, factor, Hbar1_, Hbar2_);

        outfile->Printf("  Done. Timing %10.3f s", t2.get());
    }

    return Ereturn;
}

double SA_MRPT3::compute_energy_pt3_2() {
    print_h2("Computing 3rd-Order Energy Contribution (2/3)");

    local_timer t1;
    std::string str = "Preparing 2nd-order amplitudes";
    outfile->Printf("\n    %-40s ...", str.c_str());

    // compute 2nd-order amplitudes
    // Step 1: compute 0.5 * [H1st + Hbar1st, A1st] = [H1st, A1st] + 0.5 * [[H0th, A1st], A1st]
    //     a) keep a copy of H1st + Hbar1st
    auto X1 = BTF_->build(tensor_type_, "O1 pt3 2/3", od_one_labels_ph());
    auto X2 = BTF_->build(tensor_type_, "O2 pt3 2/3", od_two_labels_pphh());
    X1["ai"] = F_["ai"];
    X2["abij"] = V_["abij"];

    //     b) scale -[[H0th, A1st], A1st] by -0.5, computed in compute_energy_pt3_1
    O1_.scale(-0.5);
    O2_.scale(-0.5);

    //     c) prepare V and F
    F_.zero();
    V_.zero();
    F_["ai"] = O1_["ai"];
    V_["abij"] = O2_["abij"];

    //     d) compute contraction from one-body term (first-order bare Fock)
    H1_T1_C1(F1st_, T1_, 1.0, F_);
    H1_T2_C1(F1st_, T2_, 1.0, F_);
    H1_T2_C2(F1st_, T2_, 1.0, V_);

    O1_ = BTF_->build(tensor_type_, "HP2 pt3 2/3", od_one_labels_hp());
    O2_ = BTF_->build(tensor_type_, "HP2 pt3 2/3", od_two_labels_hhpp());
    H1_T1_C1(F1st_, T1_, 1.0, O1_);
    H1_T2_C1(F1st_, T2_, 1.0, O1_);
    H1_T2_C2(F1st_, T2_, 1.0, O2_);

    F_["ai"] += O1_["ia"];
    V_["abij"] += O2_["ijab"];

    //     e) compute contraction from two-body term
    if (eri_df_) {
        // pphh part
        V_T1_C1_DF(B_, T1_, 1.0, F_);
        V_T2_C1_DF(B_, T2_, S2_, 1.0, F_);
        V_T1_C2_DF(B_, T1_, 1.0, V_);
        V_T2_C2_DF(B_, T2_, S2_, 1.0, V_);

        // hhpp part
        O1_.zero();
        O2_.zero();
        V_T1_C1_DF(B_, T1_, 1.0, O1_);
        V_T2_C1_DF(B_, T2_, S2_, 1.0, O1_);
        V_T1_C2_DF(B_, T1_, 1.0, O2_);
        V_T2_C2_DF(B_, T2_, S2_, 1.0, O2_);
    } else {
        auto C2 = BTF_->build(tensor_type_, "C2 pt3 2/3", {"gggg"});
        for (const std::string& block : C2.block_labels()) {
            auto mo_to_index = BTF_->get_mo_to_index();
            std::vector<size_t> i0 = mo_to_index[block.substr(0, 1)];
            std::vector<size_t> i1 = mo_to_index[block.substr(1, 1)];
            std::vector<size_t> i2 = mo_to_index[block.substr(2, 1)];
            std::vector<size_t> i3 = mo_to_index[block.substr(3, 1)];
            auto Vblock = ints_->aptei_ab_block(i0, i1, i2, i3);
            C2.block(block).copy(Vblock);
        }

        // pphh part
        H2_T1_C1(C2, T1_, 1.0, F_);
        H2_T2_C1(C2, T2_, S2_, 1.0, F_);
        H2_T1_C2(C2, T1_, 1.0, V_);
        H2_T2_C2(C2, T2_, S2_, 1.0, V_);

        // hhpp part
        O1_.zero();
        O2_.zero();
        H2_T1_C1(C2, T1_, 1.0, O1_);
        H2_T2_C1(C2, T2_, S2_, 1.0, O1_);
        H2_T1_C2(C2, T1_, 1.0, O2_);
        H2_T2_C2(C2, T2_, S2_, 1.0, O2_);
    }
    F_["ai"] += O1_["ia"];
    V_["abij"] += O2_["ijab"];
    outfile->Printf("  Done. Timing %10.3f s", t1.get());

    // Step 2: compute amplitdes
    //     a) save 1st-order amplitudes for later use
    O1_.set_name("T1 1st");
    O2_.set_name("T2 1st");
    O1_["ia"] = T1_["ia"];
    O2_["ijab"] = T2_["ijab"];

    //     b) compute 2nd-order amplitdes
    compute_t2_full();
    compute_t1();

    // compute energy from 0.5 * [[H1st + Hbar1st, A1st], A2nd]
    local_timer t2;
    str = "Computing 3rd-order energy (2/3)";
    outfile->Printf("\n    %-40s ...", str.c_str());
    double Ereturn = 0.0;
    H1_T1_C0(X1, T1_, 1.0, Ereturn);
    H1_T2_C0(X1, T2_, 1.0, Ereturn);
    H2_T1_C0(X2, T1_, 1.0, Ereturn);
    H2_T2_C0(X2, T2_, S2_, 1.0, Ereturn);
    outfile->Printf("  Done. Timing %10.3f s", t2.get());

    if (form_Hbar_) {
        local_timer t3;
        str = "Computing integrals for ref. relaxation";
        outfile->Printf("\n    %-40s ...", str.c_str());

        H_A_Ca(X1, X2, T1_, T2_, S2_, 0.5, Hbar1_, Hbar2_);

        outfile->Printf("  Done. Timing %10.3f s", t3.get());
    }

    // analyze amplitudes
    analyze_amplitudes("Second-Order", T1_, T2_);

    return Ereturn;
}

double SA_MRPT3::compute_energy_pt3_3() {
    print_h2("Computing 3rd-Order Energy Contribution (3/3)");

    // scale F and V by exponential delta
    renormalize_integrals(false);

    // reset S2 to first order
    S2_["ijab"] = 2.0 * O2_["ijab"] - O2_["ijba"];

    // compute energy of 0.5 * [Hbar2nd, A1st]
    double Ereturn = 0.0;
    local_timer t1;
    std::string str = "Computing 3rd-order energy (3/3)";
    outfile->Printf("\n    %-40s ...", str.c_str());
    H1_T1_C0(F_, O1_, 1.0, Ereturn);
    H1_T2_C0(F_, O2_, 1.0, Ereturn);
    H2_T1_C0(V_, O1_, 1.0, Ereturn);
    H2_T2_C0(V_, O2_, S2_, 1.0, Ereturn);
    outfile->Printf("  Done. Timing %10.3f s", t1.get());

    // relax reference
    if (form_Hbar_) {
        local_timer t2;
        str = "Computing integrals for ref. relaxation";
        outfile->Printf("\n    %-40s ...", str.c_str());

        H_A_Ca(F_, V_, O1_, O2_, S2_, 0.5, Hbar1_, Hbar2_);

        outfile->Printf("  Done. Timing %10.3f s", t2.get());
    }

    return Ereturn;
}
} // namespace forte
