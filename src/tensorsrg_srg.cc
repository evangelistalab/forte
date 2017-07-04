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

#include "mini-boost/boost/numeric/odeint.hpp"

#include "tensorsrg.h"

using namespace psi;
using namespace boost::numeric::odeint;

namespace psi {
namespace forte {

/* The rhs of x' = f(x) */
void TensorSRG_ODEInterface::operator()(const odeint_state_type& x, odeint_state_type& dxdt,
                                        const double t) {
    throw std::runtime_error("TensorSRG_ODEInterface::operator() is not implemented yet!");

    /*
    // Step 1. Read the Hamiltonian from the vector x
    tensorsrg_obj_.Hbar0 = x[0];

    size_t k = 1;
    tensorsrg_obj_.Hbar1.iterate_over_elements(
                [&](std::vector<size_t>& m,std::vector<MOSetSpinType>&
    spin,double& value){value = x[k]; k++;});
    tensorsrg_obj_.Hbar2.iterate_over_elements(
                [&](std::vector<size_t>& m,std::vector<MOSetSpinType>&
    spin,double& value){value = x[k]; k++;});

    // Step 2. Compute the SRG Hamiltonian flow from Hbar and store it in O
    tensorsrg_obj_.compute_srg_step();

    // Step 3. Store the Hamiltonian flow in the rhs of the ODE
    tensorsrg_obj_.Hbar0 = x[0];

    k = 1;
    dxdt[0] = tensorsrg_obj_.C0;
    tensorsrg_obj_.C1.iterate_over_elements(
                [&](std::vector<size_t>& m,std::vector<MOSetSpinType>&
    spin,double& value){dxdt[k] = value; k++;});
    tensorsrg_obj_.C2.iterate_over_elements(
                [&](std::vector<size_t>& m,std::vector<MOSetSpinType>&
    spin,double& value){dxdt[k] = value; k++;});

    neval_ += 1;
    */
}

struct push_back_state_and_time_srg {
    std::vector<double>& m_states;
    std::vector<double>& m_times;

    push_back_state_and_time_srg(std::vector<double>& states, std::vector<double>& times)
        : m_states(states), m_times(times) {}

    void operator()(const odeint_state_type& x, double t) {
        m_states.push_back(x[0]);
        m_times.push_back(t);
        outfile->Printf("\n    @SRG%4d %24.15f %24.15f", int(m_states.size()), t, x[0]);

        //        outfile->Printf("\n %9d %20.12f
        //        %20.12f",int(m_states.size()),t,x[0]);
    }
};

double TensorSRG::compute_srg_energy() {
    throw std::runtime_error("TensorSRG::compute_srg_energy() is not implemented yet!");
    return 0.0;
    /*
    vector<double> e_vec,times;

    size_t x_size = 1 + Hbar1.size() + Hbar2.size();
    odeint_state_type x(x_size);

    // Initialize Hbar with the normal ordered Hamiltonian
    Hbar0 = E0_;
    Hbar1["pq"] = F["pq"];
    Hbar1["PQ"] = F["PQ"];
    Hbar2["pqrs"] = V["pqrs"];
    Hbar2["pQrS"] = V["pQrS"];
    Hbar2["PQRS"] = V["PQRS"];

    x[0] = Hbar0;
    size_t k = 1;
    Hbar1.iterate_over_elements(
                [&](std::vector<size_t>& m,std::vector<MOSetSpinType>&
    spin,double& value){ x[k] = value; k++;});
    Hbar2.iterate_over_elements(
                [&](std::vector<size_t>& m,std::vector<MOSetSpinType>&
    spin,double& value){ x[k] = value; k++;});

    TensorSRG_ODEInterface tensorsrg_flow_computer(*this);

    double start_time = 0.0;
    double end_time = options_.get_double("SRG_SMAX");

    double initial_step = options_.get_double("SRG_DT");
    string srg_odeint = options_.get_str("SRG_ODEINT");

    double absolute_error_tollerance = options_.get_double("SRG_ODEINT_ABSERR");
    double relative_error_tollerance = options_.get_double("SRG_ODEINT_RELERR");


    outfile->Printf("\n\n  SRG-SD Computation");
    outfile->Printf("\n  Max s:             %10.6f",end_time);
    outfile->Printf("\n  Initial time step: %10.6f",initial_step);
    outfile->Printf("\n
    --------------------------------------------------------------");
    outfile->Printf("\n         Cycle        s (a.u.)                 Energy
    (a.u.)");
    outfile->Printf("\n
    --------------------------------------------------------------");

    size_t steps = 0;
    if (srg_odeint == "FEHLBERG78"){
        outfile->Printf("\n  Integrating the SRG equations using the Fehlberg 78
    algorithm");
        integrate_adaptive(
                    make_controlled(absolute_error_tollerance,
                                    relative_error_tollerance,
                                    runge_kutta_fehlberg78<odeint_state_type>()),
                    tensorsrg_flow_computer,
                    x,start_time,end_time,initial_step,
                    push_back_state_and_time_srg( e_vec , times ));
    }else if (srg_odeint == "CASHKARP"){
        outfile->Printf("\n  Integrating the SRG equations using the Cash-Karp
    54 algorithm");
        integrate_adaptive(
                    make_controlled(absolute_error_tollerance,
                                    relative_error_tollerance,
                                    runge_kutta_cash_karp54<odeint_state_type>()),
                    tensorsrg_flow_computer,
                    x,start_time,end_time,initial_step,
                    push_back_state_and_time_srg( e_vec , times ));
    }else if (srg_odeint == "DOPRI5"){
        outfile->Printf("\n  Integrating the SRG equations using the
    Dormand-Prince 5 algorithm");
        integrate_adaptive(
                    make_controlled(absolute_error_tollerance,
                                    relative_error_tollerance,
                                    runge_kutta_dopri5<odeint_state_type>()),
                    tensorsrg_flow_computer,
                    x,start_time,end_time,initial_step,
                    push_back_state_and_time_srg( e_vec , times ));
    }
    double final_energy = e_vec.back();

    outfile->Printf("\n
    --------------------------------------------------------------");
    outfile->Printf("\n\n\n    SRG-SD correlation energy      =
    %25.15f",final_energy-E0_);
    outfile->Printf("\n  * SRG-SD total energy            =
    %25.15f\n",final_energy);


    // Set some environment variables
    Process::environment.globals["CURRENT ENERGY"] = final_energy;
    Process::environment.globals["SRG-SD ENERGY"] = final_energy;
    Process::environment.globals["SRG ENERGY"] = final_energy;
    return final_energy;
        */
}

////    size_t steps = integrate(sosrg_flow_computer,
////            x , 0.0 , 35.0 , 0.001 ,
////            push_back_state_and_time_srg( e_vec , times ) );

////    size_t steps = integrate_adaptive(make_dense_output( 1.0e-12 , 1.0e-12 ,
/// runge_kutta_dopri5< state_type >() ),
////                       sosrg_flow_computer,
////                       x,0.0,10.0,0.001,
////                       push_back_state_and_time_srg( e_vec , times ));

void TensorSRG::compute_srg_step() {
    throw std::runtime_error("TensorSRG::compute_srg_step() is not implemented yet!");

    /*
    // Step 1. Compute the generator (stored in eta)
    if (options_.get_str("SRG_ETA") == "WEGNER_BLOCK"){
        // a. copy the Hamiltonian
        eta1.zero();
        eta2.zero();

        // a. prepare the off-diagonal part
        g1["pq"] = Hbar1["pq"];
        g1["PQ"] = Hbar1["PQ"];
        g2["pqrs"] = Hbar2["pqrs"];
        g2["pQrS"] = Hbar2["pQrS"];
        g2["PQRS"] = Hbar2["PQRS"];

        // b. zero the diagonal blocks of Hbar
        g1.block("ov")->zero();
        g1.block("vo")->zero();
        g1.block("OV")->zero();
        g1.block("VO")->zero();
        g2.block("oovv")->zero();
        g2.block("vvoo")->zero();
        g2.block("oOvV")->zero();
        g2.block("vVoO")->zero();
        g2.block("OOVV")->zero();
        g2.block("VVOO")->zero();

        double S0 = 0.0;
        // c. compute the commutator [Hd,Hoff]
        full_commutator_A_B_C(-1.0,Hbar1,Hbar2,g1,g2,S0,eta1,eta2);
    }else if (options_.get_str("SRG_ETA") == "WEGNER_BLOCK2"){
//        S1.zero();
//        S2.zero();
//        R1["pq"] = Hbar1["pq"];
//        R1["PQ"] = Hbar1["PQ"];
//        R2["pqrs"] = Hbar2["pqrs"];
//        R2["pQrS"] = Hbar2["pQrS"];
//        R2["PQRS"] = Hbar2["PQRS"];

//        Hbar1.block("ov")->zero();
//        Hbar1.block("vo")->zero();
//        Hbar1.block("OV")->zero();
//        Hbar1.block("VO")->zero();
//        Hbar2.block("oovv")->zero();
//        Hbar2.block("vvoo")->zero();
//        Hbar2.block("oOvV")->zero();
//        Hbar2.block("vVoO")->zero();
//        Hbar2.block("OOVV")->zero();
//        Hbar2.block("VVOO")->zero();

//        double S0 = 0.0;
//        hermitian_commutator_A_B_C(1.0,Hbar1,Hbar2,R1,R2,S0,R1,R2);

//        // d. copy the off-diagonal part of Hbar back in place
//        Hbar1["ia"] = R1["ia"];
//        Hbar1["ai"] = R1["ia"];
//        Hbar1["IA"] = R1["IA"];
//        Hbar1["AI"] = R1["IA"];
//        Hbar2["ijab"] = R2["ijab"];
//        Hbar2["abij"] = R2["ijab"];
//        Hbar2["iJaB"] = R2["iJaB"];
//        Hbar2["aBiJ"] = R2["iJaB"];
//        Hbar2["IJAB"] = R2["IJAB"];
//        Hbar2["ABIJ"] = R2["IJAB"];
    }else if (options_.get_str("SRG_ETA") == "WHITE"){

        Tensor& Hbar1_oo = *Hbar1.block("oo");
        Tensor& Hbar1_ov = *Hbar1.block("ov");
        Tensor& Hbar1_vv = *Hbar1.block("vv");

        Tensor& Hbar1_OO = *Hbar1.block("OO");
        Tensor& Hbar1_OV = *Hbar1.block("OV");
        Tensor& Hbar1_VV = *Hbar1.block("VV");

        Tensor& Hbar2_oooo = *Hbar2.block("oooo");
        Tensor& Hbar2_oOoO = *Hbar2.block("oOoO");
        Tensor& Hbar2_OOOO = *Hbar2.block("OOOO");

        Tensor& Hbar2_oovv = *Hbar2.block("oovv");
        Tensor& Hbar2_oOvV = *Hbar2.block("oOvV");
        Tensor& Hbar2_OOVV = *Hbar2.block("OOVV");

        Tensor& Hbar2_vovo = *Hbar2.block("vovo");
        Tensor& Hbar2_vOvO = *Hbar2.block("vOvO");
        Tensor& Hbar2_oVoV = *Hbar2.block("oVoV");
        Tensor& Hbar2_VOVO = *Hbar2.block("VOVO");

        Tensor& Hbar2_vvvv = *Hbar2.block("vvvv");
        Tensor& Hbar2_vVvV = *Hbar2.block("vVvV");
        Tensor& Hbar2_VVVV = *Hbar2.block("VVVV");

        S1.fill_one_electron_spin([&](size_t i,MOSetSpinType si,size_t
a,MOSetSpinType sa){
            if (si  == Alpha){
                size_t ii = mos_to_aocc[i];
                size_t aa = mos_to_avir[a];
                return Hbar1_ov(ii,aa) / (Hbar1_vv(aa,aa) - Hbar1_oo(ii,ii) -
Hbar2_vovo(aa,ii,aa,ii));
            }else if (si  == Beta){
                size_t ii = mos_to_bocc[i];
                size_t aa = mos_to_bvir[a];
                return Hbar1_OV(ii,aa) / (Hbar1_VV(aa,aa) - Hbar1_OO(ii,ii) -
Hbar2_VOVO(aa,ii,aa,ii));
            }
            return 0.0;
        });

        S2.fill_two_electron_spin([&](size_t i,MOSetSpinType si,
                                      size_t j,MOSetSpinType sj,
                                      size_t a,MOSetSpinType sa,
                                      size_t b,MOSetSpinType sb){
            if ((si == Alpha) and (sj == Alpha)){
                size_t ii = mos_to_aocc[i];
                size_t jj = mos_to_aocc[j];
                size_t aa = mos_to_avir[a];
                size_t bb = mos_to_avir[b];

                double A = Hbar2_vvvv(aa,bb,aa,bb) + Hbar2_oooo(ii,jj,ii,jj)
                        - Hbar2_vovo(aa,ii,aa,ii) - Hbar2_vovo(bb,ii,bb,ii)
                        - Hbar2_vovo(aa,jj,aa,jj) - Hbar2_vovo(bb,jj,bb,jj);

                return Hbar2_oovv(ii,jj,aa,bb) / (Hbar1_vv(aa,aa) +
Hbar1_vv(bb,bb) - Hbar1_oo(ii,ii) - Hbar1_oo(jj,jj) + A);
            }else if ((si == Alpha) and (sj == Beta) ){
                size_t ii = mos_to_aocc[i];
                size_t jj = mos_to_bocc[j];
                size_t aa = mos_to_avir[a];
                size_t bb = mos_to_bvir[b];

                double A = Hbar2_vVvV(aa,bb,aa,bb) + Hbar2_oOoO(ii,jj,ii,jj)
                        - Hbar2_vovo(aa,ii,aa,ii) - Hbar2_oVoV(ii,bb,ii,bb)
                        - Hbar2_vOvO(aa,jj,aa,jj) - Hbar2_VOVO(bb,jj,bb,jj);

                return Hbar2_oOvV(ii,jj,aa,bb) / (Hbar1_vv(aa,aa) +
Hbar1_VV(bb,bb) - Hbar1_oo(ii,ii) - Hbar1_OO(jj,jj) + A);
            }else if ((si == Beta)  and (sj == Beta) ){
                size_t ii = mos_to_bocc[i];
                size_t jj = mos_to_bocc[j];
                size_t aa = mos_to_bvir[a];
                size_t bb = mos_to_bvir[b];

                double A = Hbar2_VVVV(aa,bb,aa,bb) + Hbar2_OOOO(ii,jj,ii,jj)
                        - Hbar2_VOVO(aa,ii,aa,ii) - Hbar2_VOVO(bb,ii,bb,ii)
                        - Hbar2_VOVO(aa,jj,aa,jj) - Hbar2_VOVO(bb,jj,bb,jj);

                return Hbar2_OOVV(ii,jj,aa,bb) / (Hbar1_VV(aa,aa) +
Hbar1_VV(bb,bb) - Hbar1_OO(ii,ii) - Hbar1_OO(jj,jj) + A);
            }
            return 0.0;
        });
    }else{
        outfile->Printf("\n\n  Please specify a valid option for the parameter
SRG_ETA\n");
        exit(1);
    }

    // Step 2. Compute the Hamiltonian flow as dH/ds = O = -[H,S]
    C0 = 0.0;
    C1.zero();
    C2.zero();

    if (options_.get_str("SRG_ETA") == "WEGNER_BLOCK"){
        full_commutator_A_B_C(-1.0,Hbar1,Hbar2,eta1,eta2,C0,C1,C2);
    }else if (options_.get_str("SRG_ETA") == "WEGNER_BLOCK2"){
//        modified_commutator_A_B_C(-1.0,Hbar1,Hbar2,S1,S2,C0,C1,C2);
    }else{
        commutator_A_B_C(-1.0,Hbar1,Hbar2,S1,S2,C0,C1,C2,1);
    }
    */
}
}
}
