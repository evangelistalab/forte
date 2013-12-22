#include <cmath>

#include <libpsio/psio.hpp>
#include <libmints/wavefunction.h>
#include <libmints/molecule.h>

#include "sosrg.h"

using namespace std;
using namespace psi;

namespace psi{ namespace libadaptive{

SOSRG::SOSRG(Options &options, ExplorerIntegrals* ints, TwoIndex G1)
    : SOBase(options,ints,G1), srgop(SRGOpUnitary), srgcomm(SRCommutators)
{
    fprintf(outfile,"\n\n      --------------------------------------");
    fprintf(outfile,"\n          Similarity Renormalization Group");
    fprintf(outfile,"\n");
    fprintf(outfile,"\n                Version 0.1.0");
    fprintf(outfile,"\n");
    fprintf(outfile,"\n       written by Francesco A. Evangelista");
    fprintf(outfile,"\n      --------------------------------------\n");
    fflush(outfile);
    sosrg_startup(options);
    compute_canonical_transformation_energy(options);
}

SOSRG::~SOSRG()
{
}

void SOSRG::compute_canonical_transformation_energy(Options &options)
{
    fprintf(outfile,"\n\n  ######################################");
    fprintf(outfile,"\n  ### Computing the CCSD BCH energy  ###");
    fprintf(outfile,"\n  ######################################");
    // Start the CCSD cycle
    double old_energy = 0.0;
    bool   converged  = false;
    int    cycle      = 0;
    compute_recursive_single_commutator();
    while(!converged){
        fprintf(outfile,"\n  Updating the S amplitudes...");
        fflush(outfile);
        update_S1();
        update_S2();
        fprintf(outfile," done.");
        fflush(outfile);

        fprintf(outfile,"\n  Compute recursive single commutator...");
        fflush(outfile);
        double energy = compute_recursive_single_commutator();
        fprintf(outfile," done.");
        fflush(outfile);


        fprintf(outfile,"\n  --------------------------------------------");
        fprintf(outfile,"\n  nExc           |S|                  |R|");
        fprintf(outfile,"\n  --------------------------------------------");
        fprintf(outfile,"\n    1     %15e      %15e",norm(S1_),norm(S1_));
        fprintf(outfile,"\n    2     %15e      %15e",norm(S2_),norm(S2_));
        fprintf(outfile,"\n  --------------------------------------------");
        double delta_energy = energy-old_energy;
        old_energy = energy;
        fprintf(outfile,"\n  @CC %4d %25.15f %25.15f",cycle,energy,delta_energy);

        if(fabs(delta_energy) < options.get_double("E_CONVERGENCE")){
            converged = true;
        }

        if(cycle > options.get_int("MAXITER")){
            fprintf(outfile,"\n\n\tThe calculation did not converge in %d cycles\n\tQuitting PSIMRCC\n",options.get_int("MAX_ITERATIONS"));
            fflush(outfile);
            exit(1);
        }
        fflush(outfile);
        cycle++;

        fprintf(outfile,"\n  NEXT CYCLE");
        fflush(outfile);
    }
    fprintf(outfile,"\n\n      * CCSD-BCH total energy      = %25.15f",old_energy);
}

double SOSRG::compute_recursive_single_commutator()
{
    fprintf(outfile,"\n\n  Computing the BCH expansion using the");
    if (srgcomm == SRCommutators){
        fprintf(outfile," single-reference normal ordering formalism.");
    }
    fprintf(outfile,"\n  -----------------------------------------------------------------");
    fprintf(outfile,"\n  nComm          |C1|                 |C2|                  E" );
    fprintf(outfile,"\n  -----------------------------------------------------------------");

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

    fprintf(outfile,"\n  %2d %20e %20e %20.12f",0,norm(Hbar1_),norm(Hbar2_),E0);
    for (int n = 1; n < 20; ++n) {
        double factor = 1.0 / static_cast<double>(n);

        double E11 = 0.0;
        double E12 = 0.0;
        double E21 = 0.0;
        double E22 = 0.0;
        commutator_A1_B1_C0(O1_,S1_,+1.0,E11);
        commutator_A1_B2_C0(O1_,S2_,+1.0,E12);
        commutator_A1_B2_C0(S1_,O2_,-1.0,E21);
        commutator_A2_B2_C0(O2_,S2_,+1.0,E22);

        E0 += factor * (E11 + E12 + E21 + E22);

        zero(C1_);
        commutator_A1_B1_C1(O1_,S1_,+1.0,C1_);
        commutator_A1_B2_C1(O1_,S2_,+1.0,C1_);
        commutator_A1_B2_C1(S1_,O2_,-1.0,C1_);
        commutator_A2_B2_C1(O2_,S2_,+1.0,C1_);
        add(factor,C1_,1.0,Hbar1_);

        zero(C2_);
        commutator_A1_B2_C2(O1_,S2_,+1.0,C2_);
        commutator_A1_B2_C2(S1_,O2_,-1.0,C2_);
        commutator_A2_B2_C2(O2_,S2_,+1.0,C2_);
        add(factor,C2_,1.0,Hbar2_);

        add(factor,C1_,0.0,O1_);
        add(factor,C2_,0.0,O2_);

        double norm_O2 = norm(O1_);
        double norm_O1 = norm(O2_);
        fprintf(outfile,"\n  %2d %20e %20e %20.12f",n,norm_O1,norm_O2,E0);
        if (std::sqrt(norm_O2 * norm_O2 + norm_O1 * norm_O1) < 1.0e-12){
            break;
        }
    }
    fprintf(outfile,"\n  -----------------------------------------------------------------");
    fflush(outfile);
    return E0;
}


void SOSRG::sosrg_startup(Options& options)
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


    fprintf(outfile,"\n\n  emp2 = %20.12f",emp2);


    if (options.get_str("SOSRG_OP") == "UNITARY"){
        srgop = SRGOpUnitary;
        fprintf(outfile,"\n\n  Using a unitary operator\n");
    }
    if (options.get_str("SOSRG_OP") == "CC"){
        srgop = SRGOpCC;
        fprintf(outfile,"\n\n  Using an excitation operator\n");
    }

    allocate(Hbar1_);
    allocate(Hbar2_);
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
