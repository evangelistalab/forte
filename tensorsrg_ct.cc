#include <cmath>

#include <boost/numeric/odeint.hpp>

#include "libdiis/diismanager.h"

#include "tensorsrg.h"

using namespace std;
using namespace psi;

namespace psi{ namespace libadaptive{

double TensorSRG::compute_ct_energy()
{
    compute_mp2_guess();

    // Start the CTSD cycle
    double old_energy = 0.0;
    bool   converged  = false;
    int    cycle      = 0;

    int max_diis_vectors = options_.get_int("DIIS_MAX_VECS");
    DIISManager diis_manager(max_diis_vectors, "L-CTSD DIIS vector", DIISManager::OldestAdded, DIISManager::InCore);

    diis_manager.set_error_vector_size(5,
                                       DIISEntry::Pointer,S1.block("ov")->nelements(),
                                       DIISEntry::Pointer,S1.block("OV")->nelements(),
                                       DIISEntry::Pointer,S2.block("oovv")->nelements(),
                                       DIISEntry::Pointer,S2.block("oOvV")->nelements(),
                                       DIISEntry::Pointer,S2.block("OOVV")->nelements());
    diis_manager.set_vector_size(5,
                                 DIISEntry::Pointer,S1.block("ov")->nelements(),
                                 DIISEntry::Pointer,S1.block("OV")->nelements(),
                                 DIISEntry::Pointer,S2.block("oovv")->nelements(),
                                 DIISEntry::Pointer,S2.block("oOvV")->nelements(),
                                 DIISEntry::Pointer,S2.block("OOVV")->nelements());


    fprintf(outfile,"\n  L-CTSD Computation");
    fprintf(outfile,"\n  --------------------------------------------------------------");
    fprintf(outfile,"\n         Cycle        Energy (a.u.)            Delta(E) (a.u.)");
    fprintf(outfile,"\n  --------------------------------------------------------------");
    compute_hbar();
    while(!converged){
        if (print_ > 1){
            fprintf(outfile,"\n  Updating the S amplitudes...");
            fflush(outfile);
        }

        update_S1();
        update_S2();

        if (print_ > 1){
            fprintf(outfile," done.");
            fflush(outfile);
        }

        diis_manager.add_entry(10,
                               Hbar1.block("ov")->t(),
                               Hbar1.block("OV")->t(),
                               Hbar2.block("oovv")->t(),
                               Hbar2.block("oOvV")->t(),
                               Hbar2.block("OOVV")->t(),
                               S1.block("ov")->t(),
                               S1.block("OV")->t(),
                               S2.block("oovv")->t(),
                               S2.block("oOvV")->t(),
                               S2.block("OOVV")->t());
        if (cycle > max_diis_vectors){
            if (cycle % max_diis_vectors == 2){
                fprintf(outfile," -> DIIS");
                diis_manager.extrapolate(5,
                                       S1.block("ov")->t(),
                                       S1.block("OV")->t(),
                                       S2.block("oovv")->t(),
                                       S2.block("oOvV")->t(),
                                       S2.block("OOVV")->t());
            }
        }

        if (print_ > 1){
            fprintf(outfile,"\n  Compute recursive single commutator...");
            fflush(outfile);
        }

        double energy = compute_hbar();

        if (print_ > 1){
            fprintf(outfile," done.");
            fflush(outfile);
        }

        double delta_energy = energy-old_energy;
        old_energy = energy;

        fprintf(outfile,"\n    @CT %4d %24.15f %24.15f",cycle,energy,delta_energy);

        if (print_ > 1){
            fprintf(outfile,"\n  --------------------------------------------");
            fprintf(outfile,"\n  nExc           |S|                  |R|");
            fprintf(outfile,"\n  --------------------------------------------");
            fprintf(outfile,"\n    1     %15e      %15e",S1.norm(),0.0);
            fprintf(outfile,"\n    2     %15e      %15e",S2.norm(),0.0);
            fprintf(outfile,"\n  --------------------------------------------");
        }

        if(fabs(delta_energy) < options_.get_double("E_CONVERGENCE")){
            converged = true;
        }

        if(cycle > options_.get_int("MAXITER")){
            fprintf(outfile,"\n\n\tThe calculation did not converge in %d cycles\n\tQuitting.\n",options_.get_int("MAXITER"));
            fflush(outfile);
            converged = true;
            old_energy = 0.0;
        }
        fflush(outfile);
        cycle++;
    }
    fprintf(outfile,"\n  --------------------------------------------------------------");
    fprintf(outfile,"\n\n\n    L-CTSD correlation energy      = %25.15f",old_energy-reference_energy());
    fprintf(outfile,"\n  * L-CTSD total energy            = %25.15f\n",old_energy);

    // Set some environment variables
    Process::environment.globals["CURRENT ENERGY"] = old_energy;
    Process::environment.globals["CTSD ENERGY"] = old_energy;
    Process::environment.globals["LCTSD ENERGY"] = old_energy;
    return old_energy;
}

double TensorSRG::compute_hbar()
{
    if (print_ > 1){
        fprintf(outfile,"\n\n  Computing the similarity-transformed Hamiltonian");
        fprintf(outfile,"\n  -----------------------------------------------------------------");
        fprintf(outfile,"\n  nComm           C0                 |C1|                  |C2|" );
        fprintf(outfile,"\n  -----------------------------------------------------------------");
    }

    // Initialize Hbar and O with the normal ordered Hamiltonian
    Hbar0 = reference_energy();
    Hbar1["pq"] = F["pq"];
    Hbar1["PQ"] = F["PQ"];
    Hbar2["pqrs"] = V["pqrs"];
    Hbar2["pQrS"] = V["pQrS"];
    Hbar2["PQRS"] = V["PQRS"];

    O1["pq"] = F["pq"];
    O1["PQ"] = F["PQ"];
    O2["pqrs"] = V["pqrs"];
    O2["pQrS"] = V["pQrS"];
    O2["PQRS"] = V["PQRS"];

    if (print_ > 1){
        fprintf(outfile,"\n  %2d %20.12f %20e %20e",0,Hbar0,Hbar1.norm(),Hbar2.norm());
    }

    int maxn = options_.get_int("SRG_RSC_NCOMM");
    double ct_threshold = options_.get_double("SRG_RSC_THRESHOLD");
    for (int n = 1; n <= maxn; ++n) {
        double factor = 1.0 / static_cast<double>(n);

        double C0 = 0;
        C1.zero();
        C2.zero();

        // Compute the commutator C = 1/n [O,S]
        commutator_A_B_C(factor,O1,O2,S1,S2,C0,C1,C2);

        // Hbar += C
        Hbar0 += C0;
        Hbar1["pq"] += C1["pq"];
        Hbar1["PQ"] += C1["PQ"];
        Hbar2["pqrs"] += C2["pqrs"];
        Hbar2["pQrS"] += C2["pQrS"];
        Hbar2["PQRS"] += C2["PQRS"];

        // O = C
        O1["pq"] = C1["pq"];
        O1["PQ"] = C1["PQ"];
        O2["pqrs"] = C2["pqrs"];
        O2["pQrS"] = C2["pQrS"];
        O2["PQRS"] = C2["PQRS"];

        // Check |C|
        double norm_C1 = C1.norm();
        double norm_C2 = C2.norm();

        if (print_ > 1){
            fprintf(outfile,"\n  %2d %20.12f %20e %20e",n,C0,norm_C1,norm_C2);
            fflush(outfile);
        }
        if (std::sqrt(norm_C2 * norm_C2 + norm_C1 * norm_C1) < ct_threshold){
            break;
        }
    }
    if (print_ > 1){
        fprintf(outfile,"\n  -----------------------------------------------------------------");
        fflush(outfile);
    }
    return Hbar0;
}

void TensorSRG::update_S1()
{
    S1["ia"] += Hbar1["ia"] / D1["ia"];
    S1["IA"] += Hbar1["IA"] / D1["IA"];
}

void TensorSRG::update_S2()
{
    S2["ijab"] += Hbar2["ijab"] / D2["ijab"];
    S2["iJaB"] += Hbar2["iJaB"] / D2["iJaB"];
    S2["IJAB"] += Hbar2["IJAB"] / D2["IJAB"];
}

}} // EndNamespaces
