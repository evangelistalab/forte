#include <cmath>

#include <boost/numeric/odeint.hpp>

#include "libdiis/diismanager.h"

#include "tensorsrg.h"

using namespace std;
using namespace psi;

namespace psi{ namespace libadaptive{

double TensorSRG::compute_ct_energy()
{
    bool do_dsrg = false;
    if (options_.get_double("DSRG_S") > 0.0){
        do_dsrg = true;
        compute_mp2_guess_driven_srg();
    }else{
        compute_mp2_guess();
    }

    // Start the CTSD cycle
    double dsrg_s = options_.get_double("DSRG_S");
    double old_energy = 0.0;
    bool   converged  = false;
    int    cycle      = 0;

    boost::shared_ptr<DIISManager> diis_manager;

    int max_diis_vectors = options_.get_int("DIIS_MAX_VECS");
    if (max_diis_vectors > 0){
        diis_manager = boost::shared_ptr<DIISManager>(new DIISManager(max_diis_vectors, "L-CTSD DIIS vector", DIISManager::OldestAdded, DIISManager::InCore));
        diis_manager->set_error_vector_size(5,
                                           DIISEntry::Pointer,S1.block("ov")->nelements(),
                                           DIISEntry::Pointer,S1.block("OV")->nelements(),
                                           DIISEntry::Pointer,S2.block("oovv")->nelements(),
                                           DIISEntry::Pointer,S2.block("oOvV")->nelements(),
                                           DIISEntry::Pointer,S2.block("OOVV")->nelements());
        diis_manager->set_vector_size(5,
                                     DIISEntry::Pointer,S1.block("ov")->nelements(),
                                     DIISEntry::Pointer,S1.block("OV")->nelements(),
                                     DIISEntry::Pointer,S2.block("oovv")->nelements(),
                                     DIISEntry::Pointer,S2.block("oOvV")->nelements(),
                                     DIISEntry::Pointer,S2.block("OOVV")->nelements());
    }

    if (dsrg_s == 0.0){
        fprintf(outfile,"\n  Linearized Canonical Transformation Theory with Singles and Doubles");
    }else{
        fprintf(outfile,"\n  Driven Similarity Renormalization Group with Singles and Doubles (s = %f a.u.)",dsrg_s);
    }
    fprintf(outfile,"\n  --------------------------------------------------------------------------------------------------");
    fprintf(outfile,"\n         Cycle     Energy (a.u.)     Delta(E)   |Hbar1|    |Hbar2|     |S1|    |S2|  max(S1) max(S2)");
    fprintf(outfile,"\n  --------------------------------------------------------------------------------------------------");

    compute_hbar();

    while(!converged){
        if (print_ > 1){
            fprintf(outfile,"\n  Updating the S amplitudes...");
            fflush(outfile);
        }

        if (do_dsrg){
            update_S1_dsrg();
            update_S2_dsrg();
        }else{
            update_S1();
            update_S2();
        }

        if (print_ > 1){
            fprintf(outfile,"\n  --------------------------------------------");
            fprintf(outfile,"\n  nExc           |S|                  |R|");
            fprintf(outfile,"\n  --------------------------------------------");
            fprintf(outfile,"\n    1     %15e      %15e",S1.norm(),0.0);
            fprintf(outfile,"\n    2     %15e      %15e",S2.norm(),0.0);
            fprintf(outfile,"\n  --------------------------------------------");

            auto max_S2aa = S2.block("oovv")->max_abs_element();
            auto max_S2ab = S2.block("oOvV")->max_abs_element();
            auto max_S2bb = S2.block("OOVV")->max_abs_element();
            fprintf(outfile,"\n  Largest S2 (aa): %20.12f  ",max_S2aa.first);
            for (size_t index: max_S2aa.second){
                fprintf(outfile," %zu",index);
            }
            fprintf(outfile,"\n  Largest S2 (ab): %20.12f  ",max_S2ab.first);
            for (size_t index: max_S2ab.second){
                fprintf(outfile," %zu",index);
            }
            fprintf(outfile,"\n  Largest S2 (bb): %20.12f  ",max_S2bb.first);
            for (size_t index: max_S2bb.second){
                fprintf(outfile," %zu",index);
            }
        }

        if (print_ > 1){
            fprintf(outfile," done.");
            fflush(outfile);
        }
        if(diis_manager){
            if (do_dsrg){
                diis_manager->add_entry(10,
                                        DS1.block("ov")->t(),
                                        DS1.block("OV")->t(),
                                        DS2.block("oovv")->t(),
                                        DS2.block("oOvV")->t(),
                                        DS2.block("OOVV")->t(),
                                        S1.block("ov")->t(),
                                        S1.block("OV")->t(),
                                        S2.block("oovv")->t(),
                                        S2.block("oOvV")->t(),
                                        S2.block("OOVV")->t());
            }else{
                diis_manager->add_entry(10,
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
            }
            if (cycle > max_diis_vectors){
                if (cycle % max_diis_vectors == 2){
                    fprintf(outfile," -> DIIS");
                    diis_manager->extrapolate(5,
                                             S1.block("ov")->t(),
                                             S1.block("OV")->t(),
                                             S2.block("oovv")->t(),
                                             S2.block("oOvV")->t(),
                                             S2.block("OOVV")->t());
                }
            }
        }
        if (print_ > 1){
            fprintf(outfile,"\n  Compute recursive single commutator...");
            fflush(outfile);
        }

        // Compute the new similarity-transformed Hamiltonian
        double energy = E0_ + compute_hbar();

        if (print_ > 1){
            fprintf(outfile," done.");
            fflush(outfile);
        }

        double delta_energy = energy-old_energy;
        old_energy = energy;


        auto max_S1a = S1.block("ov")->max_abs_element();
        auto max_S1b = S1.block("OV")->max_abs_element();
        auto max_S2aa = S2.block("oovv")->max_abs_element();
        auto max_S2ab = S2.block("oOvV")->max_abs_element();
        auto max_S2bb = S2.block("OOVV")->max_abs_element();
        std::vector<double> S1_vec = {max_S1b.first,max_S1b.first};
        std::vector<double> S2_vec = {max_S2aa.first,max_S2ab.first,max_S2bb.first};

        double max_S1 = *max_element(S1_vec.begin(),S1_vec.end());
        double max_S2 = *max_element(S2_vec.begin(),S2_vec.end());

        double norm_H1a = Hbar1.block("ov")->norm();
        double norm_H1b  = Hbar1.block("OV")->norm();
        double norm_H2aa  = Hbar2.block("oovv")->norm();
        double norm_H2ab = Hbar2.block("oOvV")->norm();
        double norm_H2bb = Hbar2.block("OOVV")->norm();

        double norm_Hbar1_ex = std::sqrt(norm_H1a * norm_H1a + norm_H1b * norm_H1b);
        double norm_Hbar2_ex = std::sqrt(0.25 * norm_H2aa * norm_H2aa + norm_H2ab * norm_H2ab + 0.25 * norm_H2bb * norm_H2bb);

        double norm_S1a  = S1.block("ov")->norm();
        double norm_S1b  = S1.block("OV")->norm();
        double norm_S2aa = S2.block("oovv")->norm();
        double norm_S2ab = S2.block("oOvV")->norm();
        double norm_S2bb = S2.block("OOVV")->norm();

        double norm_S1 = norm_S1a + norm_S1b;
        double norm_S2 = 0.25 * (norm_S2aa + 4.0 * norm_S2ab + norm_S2bb);

        fprintf(outfile,"\n    @CT %4d %20.12f %11.3e %10.3e %10.3e %7.4f %7.4f %7.4f %7.4f",cycle,energy,delta_energy,norm_Hbar1_ex,norm_Hbar2_ex,max_S1,max_S2,norm_S1,norm_S2);

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
    fprintf(outfile,"\n  --------------------------------------------------------------------------------------------------");

    if (dsrg_s == 0.0){
        fprintf(outfile,"\n\n\n    L-CTSD correlation energy      = %25.15f",old_energy-E0_);
        fprintf(outfile,"\n  * L-CTSD total energy            = %25.15f\n",old_energy);
    }else{
        fprintf(outfile,"\n\n\n    DSRG-SD correlation energy      = %25.15f",old_energy-E0_);
        fprintf(outfile,"\n  * DSRG-SD total energy            = %25.15f\n",old_energy);
    }
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
    Hbar0 = 0.0;
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

void TensorSRG::update_S1_dsrg()
{
    double srg_s = options_.get_double("DSRG_S");

    Tensor& Fa_oo = *F.block("oo");
    Tensor& Fa_vv = *F.block("vv");
    Tensor& Fb_OO = *F.block("OO");
    Tensor& Fb_VV = *F.block("VV");
    Tensor& Ha_ov = *Hbar1.block("ov");
    Tensor& Hb_OV = *Hbar1.block("OV");
    Tensor& S1_ov = *S1.block("ov");
    Tensor& S1_OV = *S1.block("OV");

    R1.zero();

    R1.fill_one_electron_spin([&](size_t p,MOSetSpinType sp,size_t q,MOSetSpinType sq){
        if (sp  == Alpha){
            size_t pp = mos_to_aocc[p];
            size_t qq = mos_to_avir[q];
            double denominator = Fa_oo(pp,pp) - Fa_vv(qq,qq);
            double exp_factor = one_minus_exp_div_x(srg_s,denominator,dsrg_power_);
            return (Ha_ov(pp,qq) + S1_ov(pp,qq) * denominator) * exp_factor;
        }else if (sp  == Beta){
            size_t pp = mos_to_bocc[p];
            size_t qq = mos_to_bvir[q];
            double denominator = Fb_OO(pp,pp) - Fb_VV(qq,qq);
            double exp_factor = one_minus_exp_div_x(srg_s,denominator,dsrg_power_);
            return (Hb_OV(pp,qq) + S1_OV(pp,qq) * denominator) * exp_factor;
        }
        return 0.0;
    });

    // Compute the change in amplitudes
    DS1["ia"] = S1["ia"];
    DS1["ia"] -= R1["ia"];
    DS1["IA"] = S1["IA"];
    DS1["IA"] -= R1["IA"];

    S1["ia"] = R1["ia"];
    S1["IA"] = R1["IA"];
}

void TensorSRG::update_S2_dsrg()
{
    double srg_s = options_.get_double("DSRG_S");

    Tensor& Fa_oo = *F.block("oo");
    Tensor& Fa_vv = *F.block("vv");
    Tensor& Fb_OO = *F.block("OO");
    Tensor& Fb_VV = *F.block("VV");
    Tensor& S2_oovv = *S2.block("oovv");
    Tensor& S2_oOvV = *S2.block("oOvV");
    Tensor& S2_OOVV = *S2.block("OOVV");
    Tensor& H_oovv = *Hbar2.block("oovv");
    Tensor& H_oOvV = *Hbar2.block("oOvV");
    Tensor& H_OOVV = *Hbar2.block("OOVV");

    R2.zero();

    R2.fill_two_electron_spin([&](size_t p,MOSetSpinType sp,
                              size_t q,MOSetSpinType sq,
                              size_t r,MOSetSpinType sr,
                              size_t s,MOSetSpinType ss){
        if ((sp == Alpha) and (sq == Alpha)){
            size_t pp = mos_to_aocc[p];
            size_t qq = mos_to_aocc[q];
            size_t rr = mos_to_avir[r];
            size_t ss = mos_to_avir[s];
            double denominator = Fa_oo(pp,pp) + Fa_oo(qq,qq) - Fa_vv(rr,rr) - Fa_vv(ss,ss);
            double exp_factor = one_minus_exp_div_x(srg_s,denominator,dsrg_power_);
            return (H_oovv(pp,qq,rr,ss) + S2_oovv(pp,qq,rr,ss) * denominator ) * exp_factor;
        }else if ((sp == Alpha) and (sq == Beta) ){
            size_t pp = mos_to_aocc[p];
            size_t qq = mos_to_bocc[q];
            size_t rr = mos_to_avir[r];
            size_t ss = mos_to_bvir[s];
            double denominator = Fa_oo(pp,pp) + Fb_OO(qq,qq) - Fa_vv(rr,rr) - Fb_VV(ss,ss);
            double exp_factor = one_minus_exp_div_x(srg_s,denominator,dsrg_power_);
            return (H_oOvV(pp,qq,rr,ss) + S2_oOvV(pp,qq,rr,ss) * denominator ) * exp_factor;
        }else if ((sp == Beta)  and (sq == Beta) ){
            size_t pp = mos_to_bocc[p];
            size_t qq = mos_to_bocc[q];
            size_t rr = mos_to_bvir[r];
            size_t ss = mos_to_bvir[s];
            double denominator = Fb_OO(pp,pp) + Fb_OO(qq,qq) - Fb_VV(rr,rr) - Fb_VV(ss,ss);
            double exp_factor = one_minus_exp_div_x(srg_s,denominator,dsrg_power_);
            return (H_OOVV(pp,qq,rr,ss) + S2_OOVV(pp,qq,rr,ss) * denominator ) * exp_factor;
        }
        return 0.0;
    });

    // Compute the change in amplitudes
    DS2["ijab"]  = S2["ijab"];
    DS2["iJaB"]  = S2["iJaB"];
    DS2["IJAB"]  = S2["IJAB"];
    DS2["ijab"] -= R2["ijab"];
    DS2["iJaB"] -= R2["iJaB"];
    DS2["IJAB"] -= R2["IJAB"];

    S2["ijab"] = R2["ijab"];
    S2["iJaB"] = R2["iJaB"];
    S2["IJAB"] = R2["IJAB"];
}

}} // EndNamespaces
