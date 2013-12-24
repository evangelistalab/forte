#include <cmath>

#include <libpsio/psio.hpp>
#include <libmints/wavefunction.h>
#include <libmints/molecule.h>

#include "mosrg.h"

using namespace std;
using namespace psi;

namespace psi{ namespace libadaptive{

MOSRG::MOSRG(Options &options, ExplorerIntegrals* ints, TwoIndex G1aa, TwoIndex G1bb)
    : MOBase(options,ints,G1aa,G1bb), srgop(SRGOpUnitary), srgcomm(SRCommutators)
{
    fprintf(outfile,"\n\n      --------------------------------------");
    fprintf(outfile,"\n          Similarity Renormalization Group");
    fprintf(outfile,"\n");
    fprintf(outfile,"\n                Version 0.1.0");
    fprintf(outfile,"\n");
    fprintf(outfile,"\n       written by Francesco A. Evangelista");
    fprintf(outfile,"\n      --------------------------------------\n");
    fflush(outfile);
    mosrg_startup(options);
    compute_canonical_transformation_energy(options);
}

MOSRG::~MOSRG()
{
}

void MOSRG::compute_canonical_transformation_energy(Options &options)
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
            fprintf(outfile,"\n\n\tThe calculation did not converge in %d cycles\n\tQuitting PSIMRCC\n",options.get_int("MAXITER"));
            fflush(outfile);
//            exit(1);
            converged = true;
//            break;
        }
        fflush(outfile);
        cycle++;

        fprintf(outfile,"\n  NEXT CYCLE");
        fflush(outfile);
    }
    fprintf(outfile,"\n\n      * CCSD-BCH total energy      = %25.15f",old_energy);
}

double MOSRG::compute_recursive_single_commutator()
{
    fprintf(outfile,"\n\n  Computing the BCH expansion using the");
    if (srgcomm == SRCommutators){
        fprintf(outfile," single-reference normal ordering formalism.");
    }
    fprintf(outfile,"\n  -----------------------------------------------------------------");
    fprintf(outfile,"\n  nComm          |C1|                 |C2|                  E" );
    fprintf(outfile,"\n  -----------------------------------------------------------------");

    // Initialize Hbar and O with the Hamiltonian
    add(1.0,F_,0.0,Hbar1_);
    add(1.0,F_,0.0,O1_);
    add(1.0,V_,0.0,Hbar2_);
    add(1.0,V_,0.0,O2_);
    double E0 = E0_;

    fprintf(outfile,"\n  %2d %20e %20e %20.12f",0,norm(Hbar1_),norm(Hbar2_),E0);
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
        fprintf(outfile,"\n  %2d %20e %20e %20.12f",n,norm_O1,norm_O2,E0);
        if (std::sqrt(norm_O2 * norm_O2 + norm_O1 * norm_O1) < 1.0e-12){
            break;
        }
    }
    fprintf(outfile,"\n  -----------------------------------------------------------------");
    fflush(outfile);
    return E0;
}

/*

void MOSRG::compute_canonical_transformation_energy(Options &options)
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

double MOSRG::compute_recursive_single_commutator()
{
    fprintf(outfile,"\n\n  Computing the BCH expansion using the");
    if (srgcomm == SRCommutators){
        fprintf(outfile," single-reference normal ordering formalism.");
    }
    fprintf(outfile,"\n  -----------------------------------------------------------------");
    fprintf(outfile,"\n  nComm          |C1|                 |C2|                  E" );
    fprintf(outfile,"\n  -----------------------------------------------------------------");

    // Initialize Hbar and O with the Hamiltonian
    add(1.0,F_,0.0,Hbar1_);
    add(1.0,F_,0.0,O1_);
    add(1.0,V_,0.0,Hbar2_);
    add(1.0,V_,0.0,O2_);
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
*/

void MOSRG::mosrg_startup(Options& options)
{
    // Compute the MP2 energy
    double emp2 = 0.0;
    loop_mo_p loop_mo_q loop_mo_r loop_mo_s {
        double numerator_aaaa = std::pow(V_.aaaa[p][q][r][s],2.0) * No_.a[p] * No_.a[q] * Nv_.a[r] * Nv_.a[s];
        double denominator_aaaa = F_.aa[p][p] + F_.aa[q][q] - F_.aa[r][r] - F_.aa[s][s];
        if (denominator_aaaa != 0.0)
            emp2 += 0.25 * numerator_aaaa / denominator_aaaa;

        double numerator_bbbb = std::pow(V_.bbbb[p][q][r][s],2.0) * No_.b[p] * No_.b[q] * Nv_.b[r] * Nv_.b[s];
        double denominator_bbbb = F_.bb[p][p] + F_.bb[q][q] - F_.bb[r][r] - F_.bb[s][s];
        if (denominator_bbbb != 0.0)
            emp2 += 0.25 * numerator_bbbb / denominator_bbbb;

        double numerator_abab = std::pow(V_.abab[p][q][r][s],2.0) * No_.a[p] * No_.b[q] * Nv_.a[r] * Nv_.b[s];
        double denominator_abab = F_.aa[p][p] + F_.bb[q][q] - F_.aa[r][r] - F_.bb[s][s];
        if (denominator_abab != 0.0)
            emp2 += numerator_abab / denominator_abab;
    }


    fprintf(outfile,"\n\n  emp2 = %20.12f",emp2);


    if (options.get_str("SRG_OP") == "UNITARY"){
        srgop = SRGOpUnitary;
        fprintf(outfile,"\n\n  Using a unitary operator\n");
    }
    if (options.get_str("SRG_OP") == "CC"){
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

void MOSRG::mosrg_cleanup()
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

void MOSRG::update_S1()
{
    loop_mo_p loop_mo_q{
        if (F_.aa[p][p] - F_.aa[q][q] != 0.0){
            S1_.aa[p][q] += - Nv_.a[p] * No_.a[q] * Hbar1_.aa[p][q] / (F_.aa[p][p] - F_.aa[q][q]);
        }
        if (F_.bb[p][p] - F_.bb[q][q] != 0.0){
            S1_.bb[p][q] += - Nv_.b[p] * No_.b[q] * Hbar1_.bb[p][q] / (F_.bb[p][p] - F_.bb[q][q]);
        }
    }
    if (srgop == SRGOpUnitary){
        loop_mo_p loop_mo_q{
            if (F_.aa[p][p] - F_.aa[q][q] != 0.0){
                S1_.aa[p][q] += - No_.a[p] * Nv_.a[q] * Hbar1_.aa[p][q] / (F_.aa[p][p] - F_.aa[q][q]);
            }
            if (F_.bb[p][p] - F_.bb[q][q] != 0.0){
                S1_.bb[p][q] += - No_.b[p] * Nv_.b[q] * Hbar1_.bb[p][q] / (F_.bb[p][p] - F_.bb[q][q]);
            }
        }
    }
}

void MOSRG::update_S2()
{
    loop_mo_p loop_mo_q loop_mo_r loop_mo_s{
        if (F_.aa[p][p] + F_.aa[q][q] - F_.aa[r][r] - F_.aa[s][s] != 0.0){
            S2_.aaaa[p][q][r][s] += - Nv_.a[p] * Nv_.a[q] * No_.a[r] * No_.a[s] * Hbar2_.aaaa[p][q][r][s] / (F_.aa[p][p] + F_.aa[q][q] - F_.aa[r][r] - F_.aa[s][s]);
        }
        if (F_.aa[p][p] + F_.bb[q][q] - F_.aa[r][r] - F_.bb[s][s] != 0.0){
            S2_.abab[p][q][r][s] += - Nv_.a[p] * Nv_.b[q] * No_.a[r] * No_.b[s] * Hbar2_.abab[p][q][r][s] / (F_.aa[p][p] + F_.bb[q][q] - F_.aa[r][r] - F_.bb[s][s]);
        }
        if (F_.bb[p][p] + F_.bb[q][q] - F_.bb[r][r] - F_.bb[s][s] != 0.0){
            S2_.bbbb[p][q][r][s] += - Nv_.b[p] * Nv_.b[q] * No_.b[r] * No_.b[s] * Hbar2_.bbbb[p][q][r][s] / (F_.bb[p][p] + F_.bb[q][q] - F_.bb[r][r] - F_.bb[s][s]);
        }
    }

    if (srgop == SRGOpUnitary){
        loop_mo_p loop_mo_q loop_mo_r loop_mo_s{
            if (F_.aa[p][p] + F_.aa[q][q] - F_.aa[r][r] - F_.aa[s][s] != 0.0){
                S2_.aaaa[p][q][r][s] += - No_.a[p] * No_.a[q] * Nv_.a[r] * Nv_.a[s] * Hbar2_.aaaa[p][q][r][s] / (F_.aa[p][p] + F_.aa[q][q] - F_.aa[r][r] - F_.aa[s][s]);
            }
            if (F_.aa[p][p] + F_.bb[q][q] - F_.aa[r][r] - F_.bb[s][s] != 0.0){
                S2_.abab[p][q][r][s] += - No_.a[p] * No_.b[q] * Nv_.a[r] * Nv_.b[s] * Hbar2_.abab[p][q][r][s] / (F_.aa[p][p] + F_.bb[q][q] - F_.aa[r][r] - F_.bb[s][s]);
            }
            if (F_.bb[p][p] + F_.bb[q][q] - F_.bb[r][r] - F_.bb[s][s] != 0.0){
                S2_.bbbb[p][q][r][s] += - No_.b[p] * No_.b[q] * Nv_.b[r] * Nv_.b[s] * Hbar2_.bbbb[p][q][r][s] / (F_.bb[p][p] + F_.bb[q][q] - F_.bb[r][r] - F_.bb[s][s]);
            }
        }
    }
}

}} // EndNamespaces
