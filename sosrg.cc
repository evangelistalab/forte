#include <cmath>

#include <boost/numeric/odeint.hpp>

#include <libpsio/psio.hpp>
#include <libmints/wavefunction.h>
#include <libmints/molecule.h>
#include <functional>

#include "sosrg.h"

using namespace std;
using namespace psi;
using namespace boost::numeric::odeint;

namespace psi{ namespace libadaptive{

SOSRG::SOSRG(Options &options, ExplorerIntegrals* ints, TwoIndex G1)
    : SOBase(options,ints,G1), srgop(SRGOpUnitary), srgcomm(SRCommutators)
{
    outfile->Printf("\n\n      --------------------------------------");
    outfile->Printf("\n          Similarity Renormalization Group");
    outfile->Printf("\n                Spin-orbital code");
    outfile->Printf("\n");
    outfile->Printf("\n                Version 0.1.0");
    outfile->Printf("\n");
    outfile->Printf("\n       written by Francesco A. Evangelista");
    outfile->Printf("\n      --------------------------------------\n");
    outfile->Flush();
    sosrg_startup();
    if(options_.get_str("SRG_MODE") == "SRG"){
        compute_similarity_renormalization_group();
    }else if(options_.get_str("SRG_MODE") == "CT"){
        compute_canonical_transformation_energy();
    }
}

SOSRG::~SOSRG()
{
}

/* The rhs of x' = f(x) */
void SOSRG_ODEInterface::operator() (const state_type &x , state_type &dxdt , const double t)
{
//    outfile->Printf("\n  Computing the Hamiltonian flow at time %f",t);

    // Step 1. Read the Hamiltonian from the vector x
    int nso_ = sosrg_obj_.nso_;
    sosrg_obj_.Hbar0_ = x[0];
    int i = 1;
    loop_p loop_q{
        sosrg_obj_.Hbar1_[p][q] = x[i];
        i += 1;
    }
    loop_p loop_q loop_r loop_s{
        sosrg_obj_.Hbar2_[p][q][r][s] = x[i];
        i += 1;
    }

    // Step 2. Compute the SRG Hamiltonian flow from Hbar and store it in S
    sosrg_obj_.compute_similarity_renormalization_group_step();


    // Step 3. Store the Hamiltonian flow in the rhs of the ODE
    dxdt[0] = sosrg_obj_.S0_;
    i = 1;
    loop_p loop_q{
        dxdt[i] = sosrg_obj_.S1_[p][q];
        i += 1;
    }
    loop_p loop_q loop_r loop_s{
        dxdt[i] = sosrg_obj_.S2_[p][q][r][s];
        i += 1;
    }
    neval_ += 1;
//    outfile->Printf("\n  %20.12f %20.12f %20.12f",t,sosrg_obj_.Hbar0_ ,sosrg_obj_.S0_);
}

struct push_back_state_and_time
{
    std::vector< double >& m_states;
    std::vector< double >& m_times;

    push_back_state_and_time( std::vector< double > &states , std::vector< double > &times )
    : m_states( states ) , m_times( times ) { }

    void operator()( const state_type &x , double t )
    {
        m_states.push_back( x[0] );
        m_times.push_back( t );
        outfile->Printf("\n %20.12f %20.12f",t,x[0]);
    }
};

void SOSRG::compute_similarity_renormalization_group()
{
    vector<double> e_vec,times;

//    const double gam = 0.15;

    state_type x(1 + nso_ * nso_ + nso_ * nso_ * nso_ * nso_);
    // Initialize Hbar with the normal ordered Hamiltonian
    Hbar0_ = E0_;
    x[0] = E0_;
    int i = 1;
    loop_p loop_q{
        x[i] = Hbar1_[p][q] = F_[p][q];
        i += 1;
    }
    loop_p loop_q loop_r loop_s{
        x[i] = Hbar2_[p][q][r][s] = V_[p][q][r][s];
        i += 1;
    }

    SOSRG_ODEInterface sosrg_flow_computer(*this);

//    size_t steps = integrate(sosrg_flow_computer,
//            x , 0.0 , 35.0 , 0.001 ,
//            push_back_state_and_time( e_vec , times ) );

//    size_t steps = integrate_adaptive(make_dense_output( 1.0e-12 , 1.0e-12 , runge_kutta_dopri5< state_type >() ),
//                       sosrg_flow_computer,
//                       x,0.0,10.0,0.001,
//                       push_back_state_and_time( e_vec , times ));

    double end_time = options_.get_double("SRG_SMAX");
    size_t steps = integrate_adaptive(
                        make_controlled( 1.0e-9 , 1.0e-9 , runge_kutta_fehlberg78<state_type>()),
//                        controlled_runge_kutta< runge_kutta_fehlberg78<state_type> >(),
                        sosrg_flow_computer,
                        x,0.0,end_time,0.001,
                        push_back_state_and_time( e_vec , times ));

    //    size_t steps = integrate_const( stepper , sosrg_flow_computer , x , 0.0 , 40.0 , 0.1 );


    /* output */
    for( size_t i=0; i<=steps; i++ )
    {
        outfile->Printf("\n %20.12f %20.12f",times[i],e_vec[i]);
    }
    outfile->Printf("\n\n  The SRG integration required %d evaluations",sosrg_flow_computer.neval());
}

void SOSRG::compute_similarity_renormalization_group_step()
{

//    outfile->Printf("\n  |Hbar1| = %20e |Hbar2| = %20e",norm(Hbar1_),norm(Hbar2_));


    // Step 1. Compute the generator (stored in eta)
    if (options_.get_str("SRG_ETA") == "WEGNER_BLOCK"){
        // a. copy the Hamiltonian
        add(1.0,Hbar1_,0.0,O1_);
        add(1.0,Hbar2_,0.0,O2_);

        // b. zero the off-diagonal blocks of G
        loop_p loop_q{
            if (Nv_[p] * No_[q] == 1.0){
                O1_[p][q] = 0.0;
                O1_[q][p] = 0.0;
            }
        }
        loop_p loop_q loop_r loop_s{
            if (Nv_[p] * Nv_[q] * No_[r] * No_[s] == 1.0){
                O2_[p][q][r][s] = 0.0;
                O2_[r][s][p][q] = 0.0;
            }
        }

        // c. compute the generator as eta = [G,H]
        double eta0 = 0.0;
        zero(eta1_);
        zero(eta2_);
        commutator_A_B_C(1.0,O1_,O2_,Hbar1_,Hbar2_,eta0,eta1_,eta2_);

    }else if (options_.get_str("SRG_ETA") == "WEGNER_DIAG"){
        // a. Copy the diagonal of Hbar1 to G
        zero(O1_);
        zero(O2_);
        loop_p{
            O1_[p][p] = Hbar1_[p][p];
        }

        // b. compute the generator as eta = [G,H]
        double eta0 = 0.0;
        zero(eta1_);
        zero(eta2_);
        commutator_A_B_C(1.0,O1_,O2_,Hbar1_,Hbar2_,eta0,eta1_,eta2_);

    }else if (options_.get_str("SRG_ETA") == "WHITE"){
        loop_p loop_q{
            if (Nv_[p] * No_[q] == 1.0){
                double e1 = Hbar1_[p][q] / (Hbar1_[p][p] - Hbar1_[q][q] - Hbar2_[p][q][p][q]);
                eta1_[p][q] = e1;
                eta1_[q][p] = -e1;
            }
        }
        loop_p loop_q loop_r loop_s{
            if (Nv_[p] * Nv_[q] * No_[r] * No_[s] == 1.0){
                double A = + Hbar2_[p][q][p][q] + Hbar2_[r][s][r][s]
                           - Hbar2_[p][r][p][r] - Hbar2_[q][s][q][s]
                           - Hbar2_[p][s][p][s] - Hbar2_[q][r][q][r];
                double e2 = Hbar2_[p][q][r][s] / (Hbar1_[p][p] + Hbar1_[q][q] - Hbar1_[r][r] - Hbar1_[s][s] + A);
                eta2_[p][q][r][s] = e2;
                eta2_[r][s][p][q] = - e2;
            }
        }
    }

    // Step 2. Compute the Hamiltonian flow as dH/ds = [eta,H] (stored in S)
    S0_ = 0.0;
    zero(S1_);
    zero(S2_);
    commutator_A_B_C(1.0,eta1_,eta2_,Hbar1_,Hbar2_,S0_,S1_,S2_);
}

void SOSRG::compute_canonical_transformation_energy()
{
    outfile->Printf("\n\n  ######################################");
    outfile->Printf("\n  ### Computing the CCSD BCH energy  ###");
    outfile->Printf("\n  ######################################");
    // Start the CCSD cycle
    double old_energy = 0.0;
    bool   converged  = false;
    int    cycle      = 0;
    compute_recursive_single_commutator();
    while(!converged){
        outfile->Printf("\n  Updating the S amplitudes...");
        outfile->Flush();
        update_S1();
        update_S2();
        outfile->Printf(" done.");
        outfile->Flush();

        outfile->Printf("\n  Compute recursive single commutator...");
        outfile->Flush();
        double energy = compute_recursive_single_commutator();
        outfile->Printf(" done.");
        outfile->Flush();


        outfile->Printf("\n  --------------------------------------------");
        outfile->Printf("\n  nExc           |S|                  |R|");
        outfile->Printf("\n  --------------------------------------------");
        outfile->Printf("\n    1     %15e      %15e",norm(S1_),norm(S1_));
        outfile->Printf("\n    2     %15e      %15e",norm(S2_),norm(S2_));
        outfile->Printf("\n  --------------------------------------------");
        double delta_energy = energy-old_energy;
        old_energy = energy;
        outfile->Printf("\n  @CC %4d %25.15f %25.15f",cycle,energy,delta_energy);

        if(fabs(delta_energy) < options_.get_double("E_CONVERGENCE")){
            converged = true;
        }

        if(cycle > options_.get_int("MAXITER")){
            outfile->Printf("\n\n\tThe calculation did not converge in %d cycles\n\tQuitting PSIMRCC\n",options_.get_int("MAXITER"));
            outfile->Flush();
//            exit(1);
            converged = true;
        }
        outfile->Flush();
        cycle++;

        outfile->Printf("\n  NEXT CYCLE");
        outfile->Flush();
    }
    outfile->Printf("\n\n      * CCSD-BCH total energy      = %25.15f",old_energy);
}

double SOSRG::compute_recursive_single_commutator()
{
    outfile->Printf("\n\n  Computing the BCH expansion using the");
    if (srgcomm == SRCommutators){
        outfile->Printf(" single-reference normal ordering formalism.");
    }
    outfile->Printf("\n  -----------------------------------------------------------------");
    outfile->Printf("\n  nComm          |C1|                 |C2|                  E" );
    outfile->Printf("\n  -----------------------------------------------------------------");

    // Initialize Hbar and O with the Hamiltonian
    loop_p loop_q{
        Hbar1_[p][q] = F_[p][q];
        O1_[p][q] = F_[p][q];
    }
    loop_p loop_q loop_r loop_s{
        Hbar2_[p][q][r][s] = V_[p][q][r][s];
        O2_[p][q][r][s] = V_[p][q][r][s];
    }
    double E0 = E0_;

    outfile->Printf("\n  %2d %20e %20e %20.12f",0,norm(Hbar1_),norm(Hbar2_),E0);
    for (int n = 1; n < 20; ++n) {
        double factor = 1.0 / static_cast<double>(n);

        zero(C1_);
        zero(C2_);
        commutator_A_B_C(factor,O1_,O2_,S1_,S2_,E0,C1_,C2_);
        add(1.0,C1_,1.0,Hbar1_);
        add(1.0,C2_,1.0,Hbar2_);
        add(1.0,C1_,0.0,O1_);
        add(1.0,C2_,0.0,O2_);

        double norm_O2 = norm(O1_);
        double norm_O1 = norm(O2_);
        outfile->Printf("\n  %2d %20e %20e %20.12f",n,norm_O1,norm_O2,E0);
        if (std::sqrt(norm_O2 * norm_O2 + norm_O1 * norm_O1) < 1.0e-12){
            break;
        }
    }
    outfile->Printf("\n  -----------------------------------------------------------------");
    outfile->Flush();
    return E0;
}


void SOSRG::sosrg_startup()
{
    // Compute the MP2 energy
    double emp2 = 0.0;
    loop_p loop_q loop_r loop_s {
        //double numerator = std::pow(V_[p][q][r][s],2.0) * G1_[p][p] * G1_[q][q] * E1_[r][r] * E1_[s][s];
        double numerator = std::pow(V_[p][q][r][s],2.0) * No_[p] * No_[q] * Nv_[r] * Nv_[s];
        double denominator = F_[p][p] + F_[q][q] - F_[r][r] - F_[s][s];
        if (denominator != 0.0)
            emp2 += 0.25 * numerator / denominator;
    }


    outfile->Printf("\n\n  emp2 = %20.12f",emp2);


    if (options_.get_str("SRG_OP") == "UNITARY"){
        srgop = SRGOpUnitary;
        outfile->Printf("\n\n  Using a unitary operator\n");
    }
    if (options_.get_str("SRG_OP") == "CC"){
        srgop = SRGOpCC;
        outfile->Printf("\n\n  Using an excitation operator\n");
    }

    allocate(Hbar1_);
    allocate(Hbar2_);
    allocate(eta1_);
    allocate(eta2_);
    allocate(O1_);
    allocate(O2_);
    allocate(S1_);
    allocate(S2_);
    allocate(C1_);
    allocate(C2_);
}

void SOSRG::sosrg_cleanup()
{
    release(Hbar1_);
    release(Hbar2_);
    release(eta1_);
    release(eta2_);
    release(O1_);
    release(O2_);
    release(S1_);
    release(S2_);
    release(C1_);
    release(C2_);
}

void SOSRG::update_S1()
{
    loop_p loop_q{
        if (F_[p][p] - F_[q][q] != 0.0){
            S1_[p][q] += - Nv_[p] * No_[q] * Hbar1_[p][q] / (F_[p][p] - F_[q][q]);
        }
    }
    if (srgop == SRGOpUnitary){
        loop_p loop_q{
            if (F_[p][p] - F_[q][q] != 0.0){
                S1_[p][q] += - No_[p] * Nv_[q] * Hbar1_[p][q] / (F_[p][p] - F_[q][q]);
            }
        }
    }
}

void SOSRG::update_S2()
{
    loop_p loop_q loop_r loop_s{
        if (F_[p][p] + F_[q][q] - F_[r][r] - F_[s][s] != 0.0){
            S2_[p][q][r][s] += - Nv_[p] * Nv_[q] * No_[r] * No_[s] * Hbar2_[p][q][r][s] / (F_[p][p] + F_[q][q] - F_[r][r] - F_[s][s]);
        }
    }

    if (srgop == SRGOpUnitary){
        loop_p loop_q loop_r loop_s{
            if (F_[p][p] + F_[q][q] - F_[r][r] - F_[s][s] != 0.0){
                S2_[p][q][r][s] += - No_[p] * No_[q] * Nv_[r] * Nv_[s] * Hbar2_[p][q][r][s] / (F_[p][p] + F_[q][q] - F_[r][r] - F_[s][s]);
            }
        }
    }
}

}} // EndNamespaces
