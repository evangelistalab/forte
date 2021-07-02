/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2021 by its authors (see COPYING, COPYING.LESSER,
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

#include <chrono>

#include "psi4/libpsi4util/PsiOutStream.h"

#include "base_classes/mo_space_info.h"
#include "boost/format.hpp"
#include "boost/numeric/odeint.hpp"
#include "mrdsrg.h"

using namespace boost::numeric::odeint;

using namespace psi;

namespace forte {

void MRSRG_ODEInt::operator()(const odeint_state_type& x, odeint_state_type& dxdt, const double) {
    auto t_start = std::chrono::high_resolution_clock::now();

    // a bunch of references to simplify the typing
    double& Hbar0 = mrdsrg_obj_.Hbar0_;
    ambit::BlockedTensor& Hbar1 = mrdsrg_obj_.Hbar1_;
    ambit::BlockedTensor& Hbar2 = mrdsrg_obj_.Hbar2_;
    ambit::BlockedTensor& C1 = mrdsrg_obj_.C1_;
    ambit::BlockedTensor& C2 = mrdsrg_obj_.C2_;
    ambit::BlockedTensor& O1 = mrdsrg_obj_.O1_;
    ambit::BlockedTensor& O2 = mrdsrg_obj_.O2_;
    ambit::BlockedTensor& T1 = mrdsrg_obj_.T1_;
    ambit::BlockedTensor& T2 = mrdsrg_obj_.T2_;

    // Step 1: read from x
    size_t nelement = 1;
    C1.iterate([&](const std::vector<size_t>&, const std::vector<SpinType>&, double& value) {
        value = x[nelement];
        ++nelement;
    });
    C2.iterate([&](const std::vector<size_t>&, const std::vector<SpinType>&, double& value) {
        value = x[nelement];
        ++nelement;
    });

    // Step 2: compute the flow generator

    //     a) O1_ and O2_ are the diagonal part
    for (const auto& block : mrdsrg_obj_.diag_one_labels()) {
        O1.block(block)("pq") = C1.block(block)("pq");
    }
    for (const auto& block : mrdsrg_obj_.diag_two_labels()) {
        O2.block(block)("pqrs") = C2.block(block)("pqrs");
    }

    //     b) T1_ and T2_ are the non-diagonal part
    for (const auto& block : mrdsrg_obj_.od_one_labels()) {
        T1.block(block)("pq") = C1.block(block)("pq");
    }
    for (const auto& block : mrdsrg_obj_.od_two_labels()) {
        T2.block(block)("pqrs") = C2.block(block)("pqrs");
    }

    //     c) compute [Hd, Hod]
    Hbar1.zero();
    mrdsrg_obj_.H1_G1_C1(O1, T1, 1.0, Hbar1);
    mrdsrg_obj_.H1_G2_C1(O1, T2, 1.0, Hbar1);
    mrdsrg_obj_.H1_G2_C1(T1, O2, -1.0, Hbar1);
    mrdsrg_obj_.H2_G2_C1(O2, T2, 1.0, Hbar1);

    Hbar2.zero();
    mrdsrg_obj_.H1_G2_C2(O1, T2, 1.0, Hbar2);
    mrdsrg_obj_.H1_G2_C2(T1, O2, -1.0, Hbar2);
    mrdsrg_obj_.H2_G2_C2(O2, T2, 1.0, Hbar2);

    //     d) copy Hbar1_ and Hbar2_ to T1_ and T2_, respectively
    T1["pq"] = Hbar1["pq"];
    T1["PQ"] = Hbar1["PQ"];

    T2["pqrs"] = Hbar2["pqrs"];
    T2["pQrS"] = Hbar2["pQrS"];
    T2["PQRS"] = Hbar2["PQRS"];

    // Step 3: compute d[H(s)] / d(s) = -[H(s), eta(s)]

    Hbar0 = 0.0;
    mrdsrg_obj_.H1_G1_C0(C1, T1, -1.0, Hbar0);
    mrdsrg_obj_.H1_G2_C0(C1, T2, -1.0, Hbar0);
    mrdsrg_obj_.H1_G2_C0(T1, C2, 1.0, Hbar0);
    mrdsrg_obj_.H2_G2_C0(C2, T2, -1.0, Hbar0);

    Hbar1.zero();
    mrdsrg_obj_.H1_G1_C1(C1, T1, -1.0, Hbar1);
    mrdsrg_obj_.H1_G2_C1(C1, T2, -1.0, Hbar1);
    mrdsrg_obj_.H1_G2_C1(T1, C2, 1.0, Hbar1);
    mrdsrg_obj_.H2_G2_C1(C2, T2, -1.0, Hbar1);

    Hbar2.zero();
    mrdsrg_obj_.H1_G2_C2(C1, T2, -1.0, Hbar2);
    mrdsrg_obj_.H1_G2_C2(T1, C2, 1.0, Hbar2);
    mrdsrg_obj_.H2_G2_C2(C2, T2, -1.0, Hbar2);

    // Step 4: set values for the rhs of the ODE
    dxdt[0] = Hbar0;

    nelement = 1;
    Hbar1.iterate([&](const std::vector<size_t>&, const std::vector<SpinType>&, double& value) {
        dxdt[nelement] = value;
        ++nelement;
    });
    Hbar2.iterate([&](const std::vector<size_t>&, const std::vector<SpinType>&, double& value) {
        dxdt[nelement] = value;
        ++nelement;
    });

    auto t_end = std::chrono::high_resolution_clock::now();
    auto t_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start).count();
    mrdsrg_obj_.srg_time_ += t_ms / 1000.0;
}

void MRSRG_Print::operator()(const odeint_state_type& x, const double t) {
    double Ediff;
    size_t size = energies_.size();
    if (size == 0) {
        Ediff = 0.0;
    } else {
        Ediff = x[0] - energies_.back();
    }
    energies_.push_back(x[0]);
    ++size;

    // compute norms of off-diagonal Hbar
    double Hbar1od = mrdsrg_obj_.Hbar1od_norm(mrdsrg_obj_.od_one_labels_hp());
    double Hbar2od = mrdsrg_obj_.Hbar2od_norm(mrdsrg_obj_.od_two_labels_hhpp());

    // print
    outfile->Printf("\n    %5zu  %10.5f  %16.12f %10.3e  %10.3e %10.3e  %8.3f", size, t, x[0],
                    Ediff, Hbar1od, Hbar2od, mrdsrg_obj_.srg_time_);
    mrdsrg_obj_.srg_time_ = 0.0;
    mrdsrg_obj_.Hbar0_ = x[0];
}

double MRDSRG::compute_energy_lsrg2() {
    // print title
    outfile->Printf("\n\n  ==> Computing MR-LSRG(2) Energy <==\n");
    outfile->Printf("\n    Reference:");
    outfile->Printf("\n      J. Chem. Phys. 2016 (in preparation)\n");
    if (foptions_->get_str("THREEPDC") == "ZERO") {
        outfile->Printf("\n    Skip Lambda3 contributions in [O2, T2].");
    }

    double start_time = 0.0;
    double end_time = foptions_->get_double("DSRG_S");
    if (end_time > 1000.0) {
        end_time = 1000.0;
        outfile->Printf("\n    Set max s to 1000.");
    }

    double initial_step = foptions_->get_double("SRG_DT");
    std::string srg_odeint = foptions_->get_str("SRG_ODEINT");
    outfile->Printf("\n    Max s:             %10.6f", end_time);
    outfile->Printf("\n    ODE algorithm:     %10s", srg_odeint.c_str());
    outfile->Printf("\n    Initial time step: %10.6f", initial_step);
    outfile->Printf("\n");

    std::string title;
    std::string indent(4, ' ');
    std::string dash(79, '-');
    title += indent + str(boost::format("%5c  %10c  %=27s  %=21s  %=8s\n") % ' ' % ' ' %
                          "Energy (a.u.)" % "Non-Diagonal Norm" % " ");
    title += indent + std::string(19, ' ') + std::string(27, '-') + "  " + std::string(21, '-') +
             "  " + std::string(8, ' ') + "\n";
    title += indent + str(boost::format("%5s  %=10s  %=16s %=10s  %=10s %=10s  %=8s\n") % "Iter." %
                          "s" % "Corr." % "Delta" % "Hbar1" % "Hbar2" % "Time (s)");
    title += indent + dash;
    outfile->Printf("\n%s", title.c_str());

    // initialize tensors
    Hbar1_ = BTF_->build(tensor_type_, "Hbar1", spin_cases({"gg"}));
    Hbar2_ = BTF_->build(tensor_type_, "Hbar2", spin_cases({"gggg"}));
    C1_ = BTF_->build(tensor_type_, "C1", spin_cases({"gg"}));
    C2_ = BTF_->build(tensor_type_, "C2", spin_cases({"gggg"}));
    O1_ = BTF_->build(tensor_type_, "O1", diag_one_labels());
    O2_ = BTF_->build(tensor_type_, "O2", diag_two_labels());
    T1_ = BTF_->build(tensor_type_, "T1", od_one_labels());
    T2_ = BTF_->build(tensor_type_, "T2", od_two_labels());
    BlockedTensor::set_expert_mode(true);

    // set up ODE initial conditions
    odeint_state_type x;
    Hbar0_ = 0.0;
    x.push_back(Hbar0_);

    F_.iterate([&](const std::vector<size_t>&, const std::vector<SpinType>&, double& value) {
        x.push_back(value);
    });
    V_.iterate([&](const std::vector<size_t>&, const std::vector<SpinType>&, double& value) {
        x.push_back(value);
    });

    double absolute_error = foptions_->get_double("SRG_ODEINT_ABSERR");
    double relative_error = foptions_->get_double("SRG_ODEINT_RELERR");
    srg_time_ = 0.0;
    MRSRG_ODEInt mrsrg_flow_computer(*this);
    MRSRG_Print mrsrg_printer(*this);

    // start iterations
    if (srg_odeint == "FEHLBERG78") {
        integrate_adaptive(make_controlled(absolute_error, relative_error,
                                           runge_kutta_fehlberg78<odeint_state_type>()),
                           mrsrg_flow_computer, x, start_time, end_time, initial_step,
                           mrsrg_printer);
    } else if (srg_odeint == "CASHKARP") {
        integrate_adaptive(make_controlled(absolute_error, relative_error,
                                           runge_kutta_cash_karp54<odeint_state_type>()),
                           mrsrg_flow_computer, x, start_time, end_time, initial_step,
                           mrsrg_printer);
    } else if (srg_odeint == "DOPRI5") {
        integrate_adaptive(make_controlled(absolute_error, relative_error,
                                           runge_kutta_dopri5<odeint_state_type>()),
                           mrsrg_flow_computer, x, start_time, end_time, initial_step,
                           mrsrg_printer);
    }

    // print summary
    outfile->Printf("\n    %s", dash.c_str());
    outfile->Printf("\n\n  ==> MR-LSRG(2) Energy Summary <==\n");
    std::vector<std::pair<std::string, double>> energy;
    energy.push_back({"E0 (reference)", Eref_});
    energy.push_back({"MR-LSRG(2) correlation energy", Hbar0_});
    energy.push_back({"MR-LSRG(2) total energy", Eref_ + Hbar0_});
    for (auto& str_dim : energy) {
        outfile->Printf("\n    %-30s = %23.15f", str_dim.first.c_str(), str_dim.second);
    }

    return Hbar0_;
}

void SRGPT2_ODEInt::operator()(const odeint_state_type& x, odeint_state_type& dxdt, const double) {
    auto t_start = std::chrono::high_resolution_clock::now();

    // a bunch of references to simplify the typing
    double& Hbar0 = mrdsrg_obj_.Hbar0_;
    ambit::BlockedTensor& Hbar1 = mrdsrg_obj_.Hbar1_;
    ambit::BlockedTensor& Hbar2 = mrdsrg_obj_.Hbar2_;
    ambit::BlockedTensor& O1 = mrdsrg_obj_.O1_;
    ambit::BlockedTensor& O2 = mrdsrg_obj_.O2_;
    ambit::BlockedTensor& T1 = mrdsrg_obj_.T1_;
    ambit::BlockedTensor& T2 = mrdsrg_obj_.T2_;

    // Step 1: read from x
    size_t nelement = 1;
    Hbar1.iterate([&](const std::vector<size_t>&, const std::vector<SpinType>&, double& value) {
        value = x[nelement];
        ++nelement;
    });
    Hbar2.iterate([&](const std::vector<size_t>&, const std::vector<SpinType>&, double& value) {
        value = x[nelement];
        ++nelement;
    });

    // Step 2: compute first-order eta
    T1.zero();
    T2.zero();
    mrdsrg_obj_.H1_G1_C1(O1, Hbar1, 1.0, T1);
    mrdsrg_obj_.H1_G2_C1(O1, Hbar2, 1.0, T1);
    mrdsrg_obj_.H1_G2_C2(O1, Hbar2, 1.0, T2);

    if (Hzero_ == "FDIAG_VDIAG" || Hzero_ == "FDIAG_VACTV") {
        mrdsrg_obj_.H1_G2_C1(Hbar1, O2, -1.0, T1);
        mrdsrg_obj_.H2_G2_C1(O2, Hbar2, 1.0, T1);

        mrdsrg_obj_.H1_G2_C2(Hbar1, O2, -1.0, T2);
        mrdsrg_obj_.H2_G2_C2(O2, Hbar2, 1.0, T2);
    }

    // Step 3: compute first-order d[H(s)] / d(s) = [eta(s), H(s)]
    Hbar1.zero();
    Hbar2.zero();
    mrdsrg_obj_.H1_G1_C1(T1, O1, 1.0, Hbar1);
    mrdsrg_obj_.H1_G2_C1(O1, T2, -1.0, Hbar1);
    mrdsrg_obj_.H1_G2_C2(O1, T2, -1.0, Hbar2);

    if (Hzero_ == "FDIAG_VDIAG" || Hzero_ == "FDIAG_VACTV") {
        mrdsrg_obj_.H1_G2_C1(T1, O2, 1.0, Hbar1);
        mrdsrg_obj_.H2_G2_C1(T2, O2, 1.0, Hbar1);

        mrdsrg_obj_.H1_G2_C2(T1, O2, 1.0, Hbar2);
        mrdsrg_obj_.H2_G2_C2(T2, O2, 1.0, Hbar2);
    }

    // Step 4: set values for the rhs of the ODE
    nelement = 1;
    Hbar1.iterate([&](const std::vector<size_t>&, const std::vector<SpinType>&, double& value) {
        dxdt[nelement] = value;
        ++nelement;
    });
    Hbar2.iterate([&](const std::vector<size_t>&, const std::vector<SpinType>&, double& value) {
        dxdt[nelement] = value;
        ++nelement;
    });

    // Step 5: compute second-order energy
    //      a) need to reset Hbar to first-order Hamiltonian
    nelement = 1;
    Hbar1.iterate([&](const std::vector<size_t>&, const std::vector<SpinType>&, double& value) {
        value = x[nelement];
        ++nelement;
    });
    Hbar2.iterate([&](const std::vector<size_t>&, const std::vector<SpinType>&, double& value) {
        value = x[nelement];
        ++nelement;
    });

    //      b) compute 2nd-order energy
    Hbar0 = 0.0;
    mrdsrg_obj_.H1_G1_C0(T1, Hbar1, 1.0, Hbar0);
    mrdsrg_obj_.H1_G2_C0(T1, Hbar2, 1.0, Hbar0);
    mrdsrg_obj_.H1_G2_C0(Hbar1, T2, -1.0, Hbar0);
    mrdsrg_obj_.H2_G2_C0(T2, Hbar2, 1.0, Hbar0);
    dxdt[0] = Hbar0;

    // Step 6: if relax reference
    if (relax_ref_) {
        ambit::BlockedTensor& C1 = mrdsrg_obj_.C1_;
        ambit::BlockedTensor& C2 = mrdsrg_obj_.C2_;

        C1.zero();
        mrdsrg_obj_.H1_G1_C1(T1, Hbar1, 1.0, C1);
        mrdsrg_obj_.H1_G2_C1(T1, Hbar2, 1.0, C1);
        mrdsrg_obj_.H1_G2_C1(Hbar1, T2, -1.0, C1);
        mrdsrg_obj_.H2_G2_C1(T2, Hbar2, 1.0, C1);

        C2.zero();
        mrdsrg_obj_.H1_G2_C2(T1, Hbar2, 1.0, C2);
        mrdsrg_obj_.H1_G2_C2(Hbar1, T2, -1.0, C2);
        mrdsrg_obj_.H2_G2_C2(T2, Hbar2, 1.0, C2);

        C1.iterate([&](const std::vector<size_t>&, const std::vector<SpinType>&, double& value) {
            dxdt[nelement] = value;
            ++nelement;
        });
        C2.iterate([&](const std::vector<size_t>&, const std::vector<SpinType>&, double& value) {
            dxdt[nelement] = value;
            ++nelement;
        });
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    auto t_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start).count();
    mrdsrg_obj_.srg_time_ += t_ms / 1000.0;
}

double MRDSRG::compute_energy_srgpt2() {
    // print title
    outfile->Printf("\n\n  ==> Computing SRG-MRPT2 Energy <==\n");
    outfile->Printf("\n    Reference:");
    outfile->Printf("\n      J. Chem. Phys. 2016 (in preparation)\n");
    if (foptions_->get_str("THREEPDC") == "ZERO") {
        outfile->Printf("\n    Skip Lambda3 contributions in [O2, T2].");
    }

    double start_time = 0.0;
    double end_time = foptions_->get_double("DSRG_S");
    if (end_time > 1000.0) {
        end_time = 1000.0;
        outfile->Printf("\n    Set max s to 1000.");
    }

    double initial_step = foptions_->get_double("SRG_DT");
    std::string srg_odeint = foptions_->get_str("SRG_ODEINT");
    outfile->Printf("\n    Max s:             %10.6f", end_time);
    outfile->Printf("\n    ODE algorithm:     %10s", srg_odeint.c_str());
    outfile->Printf("\n    Initial time step: %10.6f", initial_step);
    outfile->Printf("\n");

    std::string title;
    std::string indent(4, ' ');
    std::string dash(79, '-');
    title += indent + str(boost::format("%5c  %10c  %=27s  %=21s  %=8s\n") % ' ' % ' ' %
                          "Energy (a.u.)" % "Non-Diagonal Norm" % " ");
    title += indent + std::string(19, ' ') + std::string(27, '-') + "  " + std::string(21, '-') +
             "  " + std::string(8, ' ') + "\n";
    title += indent + str(boost::format("%5s  %=10s  %=16s %=10s  %=10s %=10s  %=8s\n") % "Iter." %
                          "s" % "Corr." % "Delta" % "Hbar1" % "Hbar2" % "Time (s)");
    title += indent + dash;
    outfile->Printf("\n%s", title.c_str());

    // some options
    std::string Hzero = foptions_->get_str("DSRG_PT2_H0TH");
    bool multi_state = foptions_->get_gen_list("AVG_STATE").size() != 0;

    bool relax_ref = foptions_->get_str("RELAX_REF") != "NONE" || multi_state;

    // initialize tensors
    BlockedTensor::set_expert_mode(true);
    T1_ = BTF_->build(tensor_type_, "T1", od_one_labels());       // one-body flow generator
    T2_ = BTF_->build(tensor_type_, "T2", od_two_labels());       // two-body flow generator
    Hbar1_ = BTF_->build(tensor_type_, "Hbar1", od_one_labels()); // one-body 1st-order Hamiltonian
    Hbar2_ = BTF_->build(tensor_type_, "Hbar2", od_two_labels()); // two-body 1st-order Hamiltonian

    // include active part (2nd-order) if relax reference
    if (relax_ref) {
        C1_ = BTF_->build(tensor_type_, "C1", spin_cases({"aa"}));
        C2_ = BTF_->build(tensor_type_, "C2", spin_cases({"aaaa"}));
    }

    // prepare zeroth-order Hamiltonian
    O1_ = BTF_->build(tensor_type_, "O1", diag_one_labels());
    O1_["pq"] = F_["pq"];
    O1_["PQ"] = F_["PQ"];

    if (Hzero == "FDIAG_VDIAG") {
        O2_ = BTF_->build(tensor_type_, "O2", re_two_labels());
        O2_["pqrs"] = V_["pqrs"];
        O2_["pQrS"] = V_["pQrS"];
        O2_["PQRS"] = V_["PQRS"];
    } else if (Hzero == "FDIAG_VACTV") {
        O2_ = BTF_->build(tensor_type_, "O2", spin_cases({"aaaa"}));
        O2_["pqrs"] = V_["pqrs"];
        O2_["pQrS"] = V_["pQrS"];
        O2_["PQRS"] = V_["PQRS"];
    }

    // set up ODE initial conditions
    odeint_state_type x;
    Hbar0_ = 0.0;
    x.push_back(Hbar0_);

    // note that Hbar contains only non-diagonal part
    // so it is safe to do the following
    Hbar1_["pq"] = F_["pq"];
    Hbar1_["PQ"] = F_["PQ"];

    Hbar2_["pqrs"] = V_["pqrs"];
    Hbar2_["pQrS"] = V_["pQrS"];
    Hbar2_["PQRS"] = V_["PQRS"];

    Hbar1_.iterate([&](const std::vector<size_t>&, const std::vector<SpinType>&, double& value) {
        x.push_back(value);
    });
    Hbar2_.iterate([&](const std::vector<size_t>&, const std::vector<SpinType>&, double& value) {
        x.push_back(value);
    });

    if (relax_ref) {
        C1_.iterate([&](const std::vector<size_t>&, const std::vector<SpinType>&, double& value) {
            x.push_back(value);
        });
        C2_.iterate([&](const std::vector<size_t>&, const std::vector<SpinType>&, double& value) {
            x.push_back(value);
        });
    }

    double absolute_error = foptions_->get_double("SRG_ODEINT_ABSERR");
    double relative_error = foptions_->get_double("SRG_ODEINT_RELERR");
    srg_time_ = 0.0;
    SRGPT2_ODEInt mrsrg_flow_computer(*this, Hzero, relax_ref);
    MRSRG_Print mrsrg_printer(*this);

    // start iterations
    if (srg_odeint == "FEHLBERG78") {
        integrate_adaptive(make_controlled(absolute_error, relative_error,
                                           runge_kutta_fehlberg78<odeint_state_type>()),
                           mrsrg_flow_computer, x, start_time, end_time, initial_step,
                           mrsrg_printer);
    } else if (srg_odeint == "CASHKARP") {
        integrate_adaptive(make_controlled(absolute_error, relative_error,
                                           runge_kutta_cash_karp54<odeint_state_type>()),
                           mrsrg_flow_computer, x, start_time, end_time, initial_step,
                           mrsrg_printer);
    } else if (srg_odeint == "DOPRI5") {
        integrate_adaptive(make_controlled(absolute_error, relative_error,
                                           runge_kutta_dopri5<odeint_state_type>()),
                           mrsrg_flow_computer, x, start_time, end_time, initial_step,
                           mrsrg_printer);
    }

    // print summary
    outfile->Printf("\n    %s", dash.c_str());
    outfile->Printf("\n\n  ==> SRG-MRPT2 Energy Summary <==\n");
    std::vector<std::pair<std::string, double>> energy;
    energy.push_back({"E0 (reference)", Eref_});
    energy.push_back({"SRG-MRPT2 correlation energy", Hbar0_});
    energy.push_back({"SRG-MRPT2 total energy", Eref_ + Hbar0_});
    for (auto& str_dim : energy) {
        outfile->Printf("\n    %-30s = %23.15f", str_dim.first.c_str(), str_dim.second);
    }

    // set up all active Hbar
    if (relax_ref) {
        // a) reset Hbar to active only
        Hbar1_ = BTF_->build(tensor_type_, "C1", spin_cases({"aa"}));
        Hbar2_ = BTF_->build(tensor_type_, "C2", spin_cases({"aaaa"}));

        // b) copy C to Hbar
        Hbar1_["uv"] = F_["uv"];
        Hbar1_["UV"] = F_["UV"];
        Hbar1_["uv"] += C1_["uv"];
        Hbar1_["UV"] += C1_["UV"];

        Hbar2_["uvxy"] = V_["uvxy"];
        Hbar2_["uVxY"] = V_["uVxY"];
        Hbar2_["UVXY"] = V_["UVXY"];
        Hbar2_["uvxy"] += C2_["uvxy"];
        Hbar2_["uVxY"] += C2_["uVxY"];
        Hbar2_["UVXY"] += C2_["UVXY"];
    }

    return Hbar0_;
}
} // namespace forte
