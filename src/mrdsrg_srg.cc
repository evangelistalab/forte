#include <chrono>
#include <boost/timer.hpp>
#include <boost/format.hpp>
#include <boost/numeric/odeint.hpp>

#include "helpers.h"
#include "mrdsrg.h"

using namespace boost::numeric::odeint;

namespace psi{ namespace forte{

void MRSRG_ODEInt::operator() (const odeint_state_type& x, odeint_state_type& dxdt, const double t){
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
    C1.iterate([&](const std::vector<size_t>&, const std::vector<SpinType>&, double& value){
        value = x[nelement];
        ++nelement;
    });
    C2.iterate([&](const std::vector<size_t>&, const std::vector<SpinType>&, double& value){
        value = x[nelement];
        ++nelement;
    });

    // Step 2: compute the flow generator

    //     a) O1_ and O2_ are the diagonal part
    for(const auto& block: mrdsrg_obj_.diag_one_labels()){
        O1.block(block)("pq") = C1.block(block)("pq");
    }
    for(const auto& block: mrdsrg_obj_.diag_two_labels()){
        O2.block(block)("pqrs") = C2.block(block)("pqrs");
    }

    //     b) T1_ and T2_ are the non-diagonal part
    T1["ia"] = C1["ia"];
    T1["IA"] = C1["IA"];
    T1.block("aa").zero();
    T1.block("AA").zero();

    T2["ijab"] = C2["ijab"];
    T2["iJaB"] = C2["iJaB"];
    T2["IJAB"] = C2["IJAB"];
    T2.block("aaaa").zero();
    T2.block("aAaA").zero();
    T2.block("AAAA").zero();

    //     c) compute [Hd, Hod], need to turn on ambit expert mode
    Hbar1.zero();
    mrdsrg_obj_.H1_T1_C1(O1,T1,1.0,Hbar1);
    mrdsrg_obj_.H1_T2_C1(O1,T2,1.0,Hbar1);
    mrdsrg_obj_.H2_T1_C1(O2,T1,1.0,Hbar1);
    mrdsrg_obj_.H2_T2_C1(O2,T2,1.0,Hbar1);

    Hbar2.zero();
    mrdsrg_obj_.H1_T2_C2(O1,T2,1.0,Hbar2);
    mrdsrg_obj_.H2_T1_C2(O2,T1,1.0,Hbar2);
    mrdsrg_obj_.H2_T2_C2(O2,T2,1.0,Hbar2);

    //     d) copy Hbar1_ and Hbar2_ to T1_ and T2_, respectively
    T1["ia"] = Hbar1["ia"];
    T1["IA"] = Hbar1["IA"];

    T2["ijab"] = Hbar2["ijab"];
    T2["iJaB"] = Hbar2["iJaB"];
    T2["IJAB"] = Hbar2["IJAB"];

    // Step 3: compute d[H(s)] / d(s) = -[H(s), eta(s)]

    //     a) compute -[H(s), eta(s)], where eta(s) contains only hp (or hhpp) blocks
    Hbar0 = 0.0;
    mrdsrg_obj_.H1_T1_C0(C1,T1,-1.0,Hbar0);
    mrdsrg_obj_.H1_T2_C0(C1,T2,-1.0,Hbar0);
    mrdsrg_obj_.H2_T1_C0(C2,T1,-1.0,Hbar0);
    mrdsrg_obj_.H2_T2_C0(C2,T2,-1.0,Hbar0);

    Hbar1.zero();
    mrdsrg_obj_.H1_T1_C1(C1,T1,-1.0,Hbar1);
    mrdsrg_obj_.H1_T2_C1(C1,T2,-1.0,Hbar1);
    mrdsrg_obj_.H2_T1_C1(C2,T1,-1.0,Hbar1);
    mrdsrg_obj_.H2_T2_C1(C2,T2,-1.0,Hbar1);

    Hbar2.zero();
    mrdsrg_obj_.H1_T2_C2(C1,T2,-1.0,Hbar2);
    mrdsrg_obj_.H2_T1_C2(C2,T1,-1.0,Hbar2);
    mrdsrg_obj_.H2_T2_C2(C2,T2,-1.0,Hbar2);

    //     b) add Hermitian conjugate
    Hbar0 *= 2.0;
    C1["pq"] = Hbar1["pq"];
    C1["PQ"] = Hbar1["PQ"];
    Hbar1["pq"] += C1["qp"];
    Hbar1["PQ"] += C1["QP"];

    C2["pqrs"] = Hbar2["pqrs"];
    C2["pQrS"] = Hbar2["pQrS"];
    C2["PQRS"] = Hbar2["PQRS"];
    Hbar2["pqrs"] += C2["rspq"];
    Hbar2["pQrS"] += C2["rSpQ"];
    Hbar2["PQRS"] += C2["RSPQ"];

    // Step 4: set values for the rhs of the ODE
    dxdt[0] = Hbar0;

    nelement = 1;
    Hbar1.iterate([&](const std::vector<size_t>&, const std::vector<SpinType>&, double& value){
        dxdt[nelement] = value;
        ++nelement;
    });
    Hbar2.iterate([&](const std::vector<size_t>&, const std::vector<SpinType>&, double& value){
        dxdt[nelement] = value;
        ++nelement;
    });

    auto t_end = std::chrono::high_resolution_clock::now();
    auto t_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start).count();
    mrdsrg_obj_.srg_time_ += t_ms / 1000.0;
}

void MRSRG_Print::operator() (const odeint_state_type& x, const double t){
    double Ediff;
    size_t size = energies_.size();
    if(size == 0){
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
    outfile->Printf("\n    %5zu  %10.5f  %16.12f %10.3e  %10.3e %10.3e  %8.3f",
                    size, t, x[0], Ediff, Hbar1od, Hbar2od, mrdsrg_obj_.srg_time_);
    mrdsrg_obj_.srg_time_ = 0.0;
    mrdsrg_obj_.Hbar0_ = x[0];
}

double MRDSRG::compute_energy_lsrg2(){
    // print title
    outfile->Printf("\n\n  ==> Computing MR-LSRG(2) Energy <==\n");
    outfile->Printf("\n    Reference:");
    outfile->Printf("\n      J. Chem. Phys. 2016 (in preparation)\n");
    if(options_.get_str("THREEPDC") == "ZERO"){
        outfile->Printf("\n    Skip Lambda3 contributions in [O2, T2].");
    }

    double start_time = 0.0;
    double end_time = options_.get_double("DSRG_S");
    if(end_time > 1000.0){
        end_time = 1000.0;
        outfile->Printf("\n    Set max s to 1000.");
    }

    double initial_step = options_.get_double("SRG_DT");
    std::string srg_odeint = options_.get_str("SRG_ODEINT");
    outfile->Printf("\n    Max s:             %10.6f", end_time);
    outfile->Printf("\n    ODE algorithm:     %10s", srg_odeint.c_str());
    outfile->Printf("\n    Initial time step: %10.6f", initial_step);
    outfile->Printf("\n");

    std::string title;
    std::string indent(4, ' ');
    std::string dash(79, '-');
    title += indent + str(boost::format("%5c  %10c  %=27s  %=21s  %=8s\n")
                          % ' ' % ' ' % "Energy (a.u.)" % "Non-Diagonal Norm" % " ");
    title += indent + std::string (19, ' ') + std::string (27, '-') + "  "
            + std::string (21, '-') + "  " + std::string (8, ' ') + "\n";
    title += indent + str(boost::format("%5s  %=10s  %=16s %=10s  %=10s %=10s  %=8s\n")
                          % "Iter." % "s" % "Corr." % "Delta" % "Hbar1" % "Hbar2" % "Time (s)");
    title += indent + dash;
    outfile->Printf("\n%s", title.c_str());

    // initialize tensors
    Hbar1_ = BTF_->build(tensor_type_,"Hbar1",spin_cases({"gg"}));
    Hbar2_ = BTF_->build(tensor_type_,"Hbar2",spin_cases({"gggg"}));
    C1_ = BTF_->build(tensor_type_,"C1",spin_cases({"gg"}));
    C2_ = BTF_->build(tensor_type_,"C2",spin_cases({"gggg"}));
//    O1_ = BTF_->build(tensor_type_,"O1",spin_cases({"gg"}));
//    O2_ = BTF_->build(tensor_type_,"O2",spin_cases({"gggg"}));
    O1_ = BTF_->build(tensor_type_,"O1",diag_one_labels());
    O2_ = BTF_->build(tensor_type_,"O2",diag_two_labels());
    BlockedTensor::set_expert_mode(true);

    // set up ODE initial conditions
    odeint_state_type x;
    Hbar0_ = 0.0;
    x.push_back(Hbar0_);

    F_.iterate([&](const std::vector<size_t>&, const std::vector<SpinType>&, double& value){
        x.push_back(value);
    });
    V_.iterate([&](const std::vector<size_t>&, const std::vector<SpinType>&, double& value){
        x.push_back(value);
    });

    double absolute_error = options_.get_double("SRG_ODEINT_ABSERR");
    double relative_error = options_.get_double("SRG_ODEINT_RELERR");
    srg_time_ = 0.0;
    MRSRG_ODEInt mrsrg_flow_computer(*this);
    MRSRG_Print mrsrg_printer(*this);

    // start iterations
    if (srg_odeint == "FEHLBERG78"){
        integrate_adaptive(
                    make_controlled(absolute_error, relative_error,
                                    runge_kutta_fehlberg78<odeint_state_type>()),
                    mrsrg_flow_computer,
                    x,start_time,end_time,initial_step,
                    mrsrg_printer);
    } else if (srg_odeint == "CASHKARP"){
        integrate_adaptive(
                    make_controlled(absolute_error, relative_error,
                                    runge_kutta_cash_karp54<odeint_state_type>()),
                    mrsrg_flow_computer,
                    x,start_time,end_time,initial_step,
                    mrsrg_printer);
    } else if (srg_odeint == "DOPRI5"){
        integrate_adaptive(
                    make_controlled(absolute_error, relative_error,
                                    runge_kutta_dopri5<odeint_state_type>()),
                    mrsrg_flow_computer,
                    x,start_time,end_time,initial_step,
                    mrsrg_printer);
    }

    // print summary
    outfile->Printf("\n    %s", dash.c_str());
    outfile->Printf("\n\n  ==> MR-LSRG(2) Energy Summary <==\n");
    std::vector<std::pair<std::string,double>> energy;
    energy.push_back({"E0 (reference)", Eref_});
    energy.push_back({"MR-LSRG(2) correlation energy", Hbar0_});
    energy.push_back({"MR-LSRG(2) total energy", Eref_ + Hbar0_});
    for (auto& str_dim: energy){
        outfile->Printf("\n    %-30s = %23.15f", str_dim.first.c_str(), str_dim.second);
    }

    return Hbar0_;
}

void SRGPT2_ODEInt::operator() (const odeint_state_type& x,odeint_state_type& dxdt,const double t){
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
    size_t nelement = 0;
    for(const auto& block: mrdsrg_obj_.od_one_labels()){
        C1.block(block).iterate([&](const std::vector<size_t>&, double& value){
            value = x[nelement];
            ++nelement;
        });
    }
    for(const auto& block: mrdsrg_obj_.od_two_labels()){
        C2.block(block).iterate([&](const std::vector<size_t>&, double& value){
            value = x[nelement];
            ++nelement;
        });
    }

    // Step 2: compute first-order eta
}

double MRDSRG::compute_energy_srgpt2(){
    // print title
    outfile->Printf("\n\n  ==> Computing SRG-MRPT2 Energy <==\n");
    outfile->Printf("\n    Reference:");
    outfile->Printf("\n      J. Chem. Phys. 2016 (in preparation)\n");
    if(options_.get_str("THREEPDC") == "ZERO"){
        outfile->Printf("\n    Skip Lambda3 contributions in [O2, T2].");
    }
    Hbar1_ = BTF_->build(tensor_type_,"Hbar1",od_one_labels());
    Hbar2_ = BTF_->build(tensor_type_,"Hbar2",od_two_labels());

    return 0.0;
}

}}
