#include "tensor_basic.h"
#include "tensor_labeled.h"
#include "tensor_product.h"
#include "tensorsrg.h"

#include <cmath>

#include <boost/numeric/odeint.hpp>

#include <libpsio/psio.hpp>
#include <libmints/wavefunction.h>
#include <libmints/molecule.h>
#include <libmints/vector.h>

#include "mosrg.h"

using namespace std;
using namespace psi;

namespace psi{ namespace libadaptive{

TensorSRG::TensorSRG(boost::shared_ptr<Wavefunction> wfn, Options &options, ExplorerIntegrals* ints)
    : MethodBase(wfn,options,ints)
{
    startup();
}

TensorSRG::~TensorSRG()
{
    cleanup();
}

void TensorSRG::startup()
{
    fprintf(outfile,"\n\n      --------------------------------------");
    fprintf(outfile,"\n          Similarity Renormalization Group");
    fprintf(outfile,"\n                tensor-based code");
    fprintf(outfile,"\n");
    fprintf(outfile,"\n                Version 0.1.0");
    fprintf(outfile,"\n");
    fprintf(outfile,"\n       written by Francesco A. Evangelista");
    fprintf(outfile,"\n      --------------------------------------\n");

    fprintf(outfile,"\n      Debug level = %d\n",debug_);
    fprintf(outfile,"\n      Print level = %d\n",print_);
    fflush(outfile);

    BlockedTensor::print_mo_spaces();

    S1.resize_spin_components("S1","ov");
    S2.resize_spin_components("S2","oovv");
    R1.resize_spin_components("R1","ov");
    R2.resize_spin_components("R2","oovv");
    Hbar1.resize_spin_components("Hbar1","ii");
    Hbar2.resize_spin_components("Hbar2","iiii");
    O1.resize_spin_components("O1","ii");
    O2.resize_spin_components("O2","iiii");
    C1.resize_spin_components("C1","ii");
    C2.resize_spin_components("C2","iiii");
    I_ioiv.resize_spin_components("C2","ioiv");
}

void TensorSRG::cleanup()
{
    print_timings();
}

double TensorSRG::compute_mp2_guess()
{
    S2["ijab"] = V["ijab"] / D2["ijab"];
    S2["iJaB"] = V["iJaB"] / D2["iJaB"];
    S2["IJAB"] = V["IJAB"] / D2["IJAB"];

    double Eaa = 0.25 * BlockedTensor::dot(S2["ijab"],V["ijab"]);
    double Eab = BlockedTensor::dot(S2["iJaB"],V["iJaB"]);
    double Ebb = 0.25 * BlockedTensor::dot(S2["IJAB"],V["IJAB"]);

    double mp2_correlation_energy = Eaa + Eab + Ebb;
    double ref_energy = reference_energy();
    fprintf(outfile,"\n\n    SCF energy                            = %20.15f",ref_energy);
    fprintf(outfile,"\n    MP2 correlation energy                = %20.15f",mp2_correlation_energy);
    fprintf(outfile,"\n  * MP2 total energy                      = %20.15f\n",ref_energy + mp2_correlation_energy);
    return ref_energy + mp2_correlation_energy;
}

double TensorSRG::compute_energy()
{
    if(options_.get_str("SRG_MODE") == "SRG"){
        compute_srg_energy();
    }else if(options_.get_str("SRG_MODE") == "CT"){
        return compute_ct_energy();
    }else if(options_.get_str("SRG_MODE") == "DSRG"){
//        compute_driven_srg_energy();
    }
    print_timings();
    return 0.0;
}

double TensorSRG::compute_ct_energy()
{
    compute_mp2_guess();

    // Start the CCSD cycle
    double old_energy = 0.0;
    bool   converged  = false;
    int    cycle      = 0;

    int max_diis_vectors = options_.get_int("DIIS_MAX_VECS");
    DIISManager diis_manager(max_diis_vectors, "L-CTSD DIIS vector", DIISManager::OldestAdded, DIISManager::InCore);
//    size_t nmo2 = nmo_ * nmo_;
//    size_t nmo4 = nmo_ * nmo_ * nmo_ * nmo_;
//    Vector diis_error("De",3 * nmo4 + 2 * nmo2);
//    Vector diis_var("Dv",3 * nmo4 + 2 * nmo2);
//    diis_manager.set_error_vector_size(1,DIISEntry::Vector,&diis_error);
//    diis_manager.set_vector_size(1,DIISEntry::Vector,&diis_var);

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
//                diis_manager.extrapolate(1,&diis_var);
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

        fprintf(outfile,"\n    @CC %4d %24.15f %24.15f",cycle,energy,delta_energy);

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
