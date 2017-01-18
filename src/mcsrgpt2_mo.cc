/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2017 by its authors (see LICENSE, AUTHORS).
 *
 * The copyrights for code used from other parties are included in
 * the corresponding files.
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see http://www.gnu.org/licenses/.
 *
 * @END LICENSE
 */

#include "psi4/libqt/qt.h"
#include <string>
#include <numeric>
#include <utility>
#include <algorithm>
#include <ctype.h>
#include "mini-boost/boost/algorithm/string/predicate.hpp"
#include <fstream>
#include <iostream>
#include "mini-boost/boost/format.hpp"
#include "mcsrgpt2_mo.h"

#define Delta(i,j) ((i==j) ? 1 : 0)

using namespace std;

namespace psi{ namespace forte{

MCSRGPT2_MO::MCSRGPT2_MO(SharedWavefunction ref_wfn, Options &options,
                         std::shared_ptr<ForteIntegrals> ints, std::shared_ptr<MOSpaceInfo> mo_space_info)
    : FCI_MO(ref_wfn, options, ints, mo_space_info)
{
    compute_energy();
    reference();

    print_method_banner({"Driven Similarity Renormalization Group", "Second-Order Perturbative Analysis", "Chenyang Li"});

    startup(options);
    if(options.get_str("CORR_LEVEL") == "SRG_PT2"){
        Process::environment.globals["CURRENT ENERGY"] = compute_energy_srg();
    } else {
        Process::environment.globals["CURRENT ENERGY"] = compute_energy_dsrg();
    }
}

MCSRGPT2_MO::~MCSRGPT2_MO()
{
    cleanup();
}

void MCSRGPT2_MO::cleanup(){
//    delete integral_;
}

void MCSRGPT2_MO::startup(Options &options){

    // Source Operator
    source_ = options.get_str("SOURCE");
    if(sourcemap.find(source_) == sourcemap.end()){
        outfile->Printf("\n  Source operator %s is not available.", source_.c_str());
        outfile->Printf("\n  Only these source operators are available: ");
        for (const auto& keys: sourcemap){
            std::string key = keys.first;
            outfile->Printf("%s ", key.c_str());
        }
        outfile->Printf("\n");
        throw PSIEXCEPTION("Source operator is not available.");
    }

    // Print Delta
    if(print_ > 1){
        PrintDelta();
        test_D1_RE();
        test_D2_RE();
    }

    // DSRG Parameters
    s_ = options.get_double("DSRG_S");
    if(s_ < 0){
        throw PSIEXCEPTION("DSRG_S cannot be negative numbers.");
    }
    taylor_threshold_ = options.get_int("TAYLOR_THRESHOLD");
    if(taylor_threshold_ <= 0){
        throw PSIEXCEPTION("TAYLOR_THRESHOLD must be an integer greater than 0.");
    }
    expo_delta_ = options.get_double("DELTA_EXPONENT");
    if(expo_delta_ <= 1.0){
        throw PSIEXCEPTION("DELTA_EXPONENT must be greater than 1.0.");
    }
    double e_conv = -log10(options.get_double("E_CONVERGENCE"));
    taylor_order_ = floor((e_conv / taylor_threshold_ + 1.0) / expo_delta_) + 1;

    // Print Original Orbital Indices
    outfile->Printf("\n\n  ==> Correlated Subspace Indices <==\n");
    print_idx("CORE", idx_c_);
    print_idx("ACTIVE", idx_a_);
    print_idx("HOLE", idx_h_);
    print_idx("VIRTUAL", idx_v_);
    print_idx("PARTICLE", idx_p_);
    outfile->Printf("\n");
    outfile->Flush();

    // Compute Reference Energy
    outfile->Printf("\n  Computing reference energy using density cumulant ...");
    outfile->Flush();
    compute_ref();
    outfile->Printf("\t\t\tDone.");
    outfile->Flush();

    // 2-Particle Density Cumulant
    string twopdc = options.get_str("TWOPDC");
    if(twopdc == "ZERO"){
        L2aa_ = d4(na_, d3(na_, d2(na_, d1(na_))));
        L2ab_ = d4(na_, d3(na_, d2(na_, d1(na_))));
        L2bb_ = d4(na_, d3(na_, d2(na_, d1(na_))));
    }

    if(options.get_str("CORR_LEVEL") == "SRG_PT2"){
        // compute [1 - exp(-s * x^2)] / x^2
        srg_source_ = std::make_shared<LABS_SOURCE>(s_,taylor_threshold_);

        // no need to form amplitudes nor effective integrals
        // do need to reform the Fock matrix
        Fa_srg_ = d2(ncmo_, d1(ncmo_));
        Fb_srg_ = d2(ncmo_, d1(ncmo_));
        Form_Fock_SRG();

//        // zero all acitve two-electron integrals
//        for(size_t u = 0; u < na_; ++u){
//            size_t nu = idx_a_[u];
//            for(size_t v = 0; v < na_; ++v){
//                size_t nv = idx_a_[v];
//                for(size_t x = 0; x < na_; ++x){
//                    size_t nx = idx_a_[x];
//                    for(size_t y = 0; y < na_; ++y){
//                        size_t ny = idx_a_[y];

//                        integral_->set_tei(nu,nv,nx,ny,0.0,true,true);
//                        integral_->set_tei(nu,nv,nx,ny,0.0,true,false);
//                        integral_->set_tei(nu,nv,nx,ny,0.0,false,false);
//                    }
//                }
//            }
//        }
    } else {
        // Form T Amplitudes
        T2aa_ = d4(nh_, d3(nh_, d2(npt_, d1(npt_))));
        T2ab_ = d4(nh_, d3(nh_, d2(npt_, d1(npt_))));
        T2bb_ = d4(nh_, d3(nh_, d2(npt_, d1(npt_))));
        T1a_ = d2(nh_, d1(npt_));
        T1b_ = d2(nh_, d1(npt_));

        string t_algorithm = options.get_str("T_ALGORITHM");
        t1_amp_ = options.get_str("T1_AMP");
        bool t1_zero = t1_amp_ == "ZERO";
        outfile->Printf("\n");
        outfile->Printf("\n  Computing MR-DSRG-PT2 T amplitudes ...");
        outfile->Flush();
        if(boost::starts_with(t_algorithm, "DSRG")){
            outfile->Printf("\n  Form T amplitudes using %s formalism.", t_algorithm.c_str());
            Form_T2_DSRG(T2aa_,T2ab_,T2bb_,t_algorithm);
            if(!t1_zero){
                Form_T1_DSRG(T1a_,T1b_);
            }else{outfile->Printf("\n  Zero T1 amplitudes.");}
        }else if(t_algorithm == "SELEC"){
            outfile->Printf("\n  Form T amplitudes using DSRG_SELEC formalism. (c->a, c->v, a->v)");
            Form_T2_SELEC(T2aa_,T2ab_,T2bb_);
            if(!t1_zero){
                Form_T1_DSRG(T1a_,T1b_);
            }else{outfile->Printf("\n  Zero T1 amplitudes.");}
        }else if(t_algorithm == "ISA"){
            outfile->Printf("\n  Form T amplitudes using intruder state avoidance (ISA) formalism.");
            double b = options.get_double("ISA_B");
            Form_T2_ISA(T2aa_,T2ab_,T2bb_,b);
            if(!t1_zero){
                Form_T1_ISA(T1a_,T1b_,b);
            }else{outfile->Printf("\n  Zero T1 amplitudes.");}
        }
        outfile->Printf("\n  Done.");
        outfile->Flush();

        // Check T Amplitudes
        T2Naa_ = 0.0, T2Nab_ = 0.0, T2Nbb_ = 0.0;
        T2Maxaa_ = 0.0, T2Maxab_ = 0.0, T2Maxbb_ = 0.0;
        Check_T2("AA",T2aa_,T2Naa_,T2Maxaa_,options);
        Check_T2("AB",T2ab_,T2Nab_,T2Maxab_,options);
        Check_T2("BB",T2bb_,T2Nbb_,T2Maxbb_,options);

        T1Na_ = 0.0, T1Nb_ = 0.0;
        T1Maxa_ = 0.0, T1Maxb_ = 0.0;
        Check_T1("A",T1a_,T1Na_,T1Maxa_,options);
        Check_T1("B",T1b_,T1Nb_,T1Maxb_,options);

        bool dsrgpt = options.get_bool("DSRGPT");

        // Effective Fock Matrix
        Fa_dsrg_ = d2(ncmo_, d1(ncmo_));
        Fb_dsrg_ = d2(ncmo_, d1(ncmo_));
        outfile->Printf("\n");
        outfile->Printf("\n  Computing the MR-DSRG-PT2 effective Fock matrix ...");
        outfile->Flush();
        Form_Fock_DSRG(Fa_dsrg_,Fb_dsrg_,dsrgpt);
        outfile->Printf("\t\t\tDone.");
        outfile->Flush();

        // Effective Two Electron Integrals
        outfile->Printf("\n  Computing the MR-DSRG-PT2 effective two-electron integrals ...");
        outfile->Flush();
        Form_APTEI_DSRG(dsrgpt);
        outfile->Printf("\tDone.");
        outfile->Flush();
    }
}

double MCSRGPT2_MO::ElementRH(const string &source, const double &D, const double &V){
    if(fabs(V) < 1.0e-12) return 0.0;
    switch (sourcemap[source]) {
    case AMP:{
        double RD = D / V;
        return V * exp(-s_ * pow(fabs(RD), expo_delta_));
    }
    case EMP2:{
        double RD = D / (V * V);
        return V * exp(-s_ * pow(fabs(RD), expo_delta_));
    }
    case LAMP:{
        double RD = D / V;
//        outfile->Printf("\n  D = %20.15f, V = %20.15f, RD = %20.15f, EXP = %20.15f", D, V, RD, V * exp(-s_ * fabs(RD)));
        return V * exp(-s_ * fabs(RD));
    }
    case LEMP2:{
        double RD = D / (V * V);
        return V * exp(-s_ * fabs(RD));
    }
    default:{
        return V * exp(-s_ * pow(fabs(D), expo_delta_));
    }
    }
}

void MCSRGPT2_MO::Form_Fock_SRG(){
    timer_on("Fock_SRG");

//    for(size_t i = 0; i < nh_; ++i){
//        size_t ni = idx_h_[i];
//        for(size_t a = 0; a < npt_; ++a){
//            size_t na = idx_p_[a];

//            Fa_srg_[ni][na] = Fa_[ni][na];
//            Fa_srg_[na][ni] = Fa_[na][ni];
//            Fb_srg_[ni][na] = Fb_[ni][na];
//            Fb_srg_[na][ni] = Fb_[na][ni];

//            double va = 0.0, vb = 0.0;
//            for(size_t u = 0; u < na_; ++u){
//                size_t nu = idx_a_[u];
//                for(size_t v = 0; v < na_; ++v){
//                    size_t nv = idx_a_[v];

//                    va += Da_[nu][nv] * integral_->aptei_aa(ni,nv,na,nu);
//                    va += Db_[nu][nv] * integral_->aptei_ab(ni,nv,na,nu);
//                    vb += Da_[nu][nv] * integral_->aptei_ab(nv,ni,nu,na);
//                    vb += Db_[nu][nv] * integral_->aptei_bb(ni,nv,na,nu);
//                }
//            }

//            Fa_srg_[ni][na] -= va;
//            Fa_srg_[na][ni] -= va;
//            Fb_srg_[ni][na] -= vb;
//            Fb_srg_[na][ni] -= vb;
//        }
//    }

    for(size_t p = 0; p < ncmo_; ++p){
        for(size_t q = 0; q < ncmo_; ++q){
            double va = integral_->oei_a(p,q);
            double vb = integral_->oei_b(p,q);

            for(size_t m = 0; m < nc_; ++m){
                size_t nm = idx_c_[m];

                va += integral_->aptei_aa(p,nm,q,nm);
                va += integral_->aptei_ab(p,nm,q,nm);
                vb += integral_->aptei_bb(p,nm,q,nm);
                vb += integral_->aptei_ab(nm,p,nm,q);
            }

            Fa_srg_[p][q] = va;
            Fb_srg_[p][q] = vb;
        }
    }

    for(size_t u = 0; u < na_; ++u){
        size_t nu = idx_a_[u];
        for(size_t v = 0; v < na_; ++v){
            size_t nv = idx_a_[v];
            Fa_srg_[nu][nv] = 0.0;
            Fb_srg_[nu][nv] = 0.0;
        }
    }

    timer_off("Fock_SRG");
}

void MCSRGPT2_MO::Form_Fock_DSRG(d2 &A, d2 &B, const bool &dsrgpt){
    timer_on("Fock_DSRG");
    for(size_t p=0; p<ncmo_; ++p){
        for(size_t q=0; q<ncmo_; ++q){
            A[p][q] = Fa_[p][q];
            B[p][q] = Fb_[p][q];
        }
    }
    if(dsrgpt){
        for(size_t i=0; i<nh_; ++i){
            size_t ni = idx_h_[i];
            for(size_t a=0; a<npt_; ++a){
                size_t na = idx_p_[a];
                double value_a = 0.0, value_b = 0.0;
                for(size_t u=0; u<na_; ++u){
                    size_t nu = idx_a_[u];
                    for(size_t x=0; x<na_; ++x){
                        size_t nx = idx_a_[x];
                        value_a += (Fa_[nx][nx] - Fa_[nu][nu]) * T2aa_[i][u][a][x] * Da_[nx][nu];
                        value_a += (Fb_[nx][nx] - Fb_[nu][nu]) * T2ab_[i][u][a][x] * Db_[nx][nu];
                        value_b += (Fa_[nx][nx] - Fa_[nu][nu]) * T2ab_[u][i][x][a] * Da_[nx][nu];
                        value_b += (Fb_[nx][nx] - Fb_[nu][nu]) * T2bb_[i][u][a][x] * Db_[nx][nu];
                    }
                }
                value_a += Fa_[ni][na];
                value_b += Fb_[ni][na];

                double Da = Fa_[ni][ni] - Fa_[na][na];
                double Db = Fb_[ni][ni] - Fb_[na][na];

                A[ni][na] += ElementRH(source_, Da, value_a);
                A[na][ni] += ElementRH(source_, Da, value_a);
                B[ni][na] += ElementRH(source_, Db, value_b);
                B[na][ni] += ElementRH(source_, Db, value_b);

            }
        }
    }
    timer_off("Fock_DSRG");
}

void MCSRGPT2_MO::Form_APTEI_DSRG(const bool &dsrgpt){
    timer_on("APTEI_DSRG");

    if(dsrgpt){
        for(size_t i=0; i<nh_; ++i){
            size_t ni = idx_h_[i];
            for(size_t j=0; j<nh_; ++j){
                size_t nj = idx_h_[j];
                for(size_t a=0; a<npt_; ++a){
                    size_t na = idx_p_[a];
                    for(size_t b=0; b<npt_; ++b){
                        size_t nb = idx_p_[b];
                        double Daa = Fa_[ni][ni] + Fa_[nj][nj] - Fa_[na][na] - Fa_[nb][nb];
                        double Dab = Fa_[ni][ni] + Fb_[nj][nj] - Fa_[na][na] - Fb_[nb][nb];
                        double Dbb = Fb_[ni][ni] + Fb_[nj][nj] - Fb_[na][na] - Fb_[nb][nb];

                        double Vaa = integral_->aptei_aa(ni,nj,na,nb);
                        double Vab = integral_->aptei_ab(ni,nj,na,nb);
                        double Vbb = integral_->aptei_bb(ni,nj,na,nb);

                        Vaa += ElementRH(source_, Daa, Vaa);
                        Vab += ElementRH(source_, Dab, Vab);
                        Vbb += ElementRH(source_, Dbb, Vbb);

                        integral_->set_tei(ni,nj,na,nb,Vaa,true,true);
                        integral_->set_tei(na,nb,ni,nj,Vaa,true,true);

                        integral_->set_tei(ni,nj,na,nb,Vab,true,false);
                        integral_->set_tei(na,nb,ni,nj,Vab,true,false);

                        integral_->set_tei(ni,nj,na,nb,Vbb,false,false);
                        integral_->set_tei(na,nb,ni,nj,Vbb,false,false);

                    }
                }
            }
        }
    }
    timer_off("APTEI_DSRG");
}

void MCSRGPT2_MO::compute_ref(){
    timer_on("Compute Ref");
    Eref_ = 0.0;
    for(size_t p=0; p<nh_; ++p){
        size_t np = idx_h_[p];
        for(size_t q=0; q<nh_; ++q){
            size_t nq = idx_h_[q];
            Eref_ += (integral_->oei_a(nq,np) + Fa_[nq][np]) * Da_[np][nq];
            Eref_ += (integral_->oei_b(nq,np) + Fb_[nq][np]) * Db_[np][nq];
        }
    }
    Eref_ *= 0.5;
    for(size_t p=0; p<na_; ++p){
        size_t np = idx_a_[p];
        for(size_t q=0; q<na_; ++q){
            size_t nq = idx_a_[q];
            for(size_t r=0; r<na_; ++r){
                size_t nr = idx_a_[r];
                for(size_t s=0; s<na_; ++s){
                    size_t ns = idx_a_[s];
                    Eref_ += 0.25 * integral_->aptei_aa(np,nq,nr,ns) * L2aa_[p][q][r][s];
                    Eref_ += 0.25 * integral_->aptei_bb(np,nq,nr,ns) * L2bb_[p][q][r][s];
                    Eref_ += integral_->aptei_ab(np,nq,nr,ns) * L2ab_[p][q][r][s];
                }
            }
        }
    }
    Eref_ += e_nuc_ + integral_->frozen_core_energy();
//    outfile->Printf("\n    E0 (cumulant) %15c = %22.15f", ' ', Eref_);
    timer_off("Compute Ref");
}

double MCSRGPT2_MO::ElementT(const string &source, const double &D, const double &V){
    if(fabs(V) < 1.0e-12) return 0.0;
    switch (sourcemap[source]) {
    case AMP:{
        double RD = D / V;
        double Z = pow(s_, 1 / expo_delta_) * RD;
        if(fabs(Z) < pow(0.1, taylor_threshold_)){
            return Taylor_Exp(Z, taylor_order_, expo_delta_) * sqrt(s_);
        }else{
            return (1 - exp(-1.0 * s_ * pow(fabs(RD), expo_delta_))) * V / D;
        }
    }
    case EMP2:{
        double RD = D / V;
        double Z = pow(s_, 1 / expo_delta_) * RD / V;
        if(fabs(Z) < pow(0.1, taylor_threshold_)){
            return Taylor_Exp(Z, taylor_order_, expo_delta_) * sqrt(s_) / V;
        }else{
            return (1 - exp(-1.0 * s_ * pow(fabs(RD / V), expo_delta_))) * V / D;
        }
    }
    case LAMP:{
        double RD = D / V;
        double Z = s_ * RD;
        if(fabs(Z) < pow(0.1, taylor_threshold_)){
            return Taylor_Exp_Linear(Z, 2 * taylor_order_) * s_;
        }else{
            return (1 - exp(-1.0 * s_ * fabs(RD))) * V / D;
        }
    }
    case LEMP2:{
        double RD = D / V;
        double Z = s_ * RD / V;
        if(fabs(Z) < pow(0.1, taylor_threshold_)){
            return Taylor_Exp_Linear(Z, 2 * taylor_order_) * s_ / V;
        }else{
            return (1 - exp(-1.0 * s_ * fabs(RD / V))) * V / D;
        }
    }
    default:{
        double Z = pow(s_, 1 / expo_delta_) * D;
        if(fabs(Z) < pow(0.1, taylor_threshold_)){
            return Taylor_Exp(Z, taylor_order_, expo_delta_) * pow(s_, 1 / expo_delta_) * V;
        }else{
            return (1 - exp(-1.0 * s_ * pow(fabs(D), expo_delta_))) * V / D;
        }
    }
    }
}

void MCSRGPT2_MO::Form_T2_DSRG(d4 &AA, d4 &AB, d4 &BB, string &T_ALGOR){
    timer_on("Form T2");
    for(size_t i=0; i<nh_; ++i){
        size_t ni = idx_h_[i];
        for(size_t j=0; j<nh_; ++j){
            size_t nj = idx_h_[j];
            for(size_t a=0; a<npt_; ++a){
                size_t na = idx_p_[a];
                for(size_t b=0; b<npt_; ++b){
                    size_t nb = idx_p_[b];

                    double Daa = Fa_[ni][ni] + Fa_[nj][nj] - Fa_[na][na] - Fa_[nb][nb];
                    double Dab = Fa_[ni][ni] + Fb_[nj][nj] - Fa_[na][na] - Fb_[nb][nb];
                    double Dbb = Fb_[ni][ni] + Fb_[nj][nj] - Fb_[na][na] - Fb_[nb][nb];

                    double Vaa = integral_->aptei_aa(na,nb,ni,nj);
                    double Vab = integral_->aptei_ab(na,nb,ni,nj);
                    double Vbb = integral_->aptei_bb(na,nb,ni,nj);

                    AA[i][j][a][b] = ElementT(source_, Daa, Vaa);
                    AB[i][j][a][b] = ElementT(source_, Dab, Vab);
                    BB[i][j][a][b] = ElementT(source_, Dbb, Vbb);

                }
            }
        }
    }

    // Zero Internal Excitations
    for(size_t i=0; i<na_; ++i){
        for(size_t j=0; j<na_; ++j){
            for(size_t k=0; k<na_; ++k){
                for(size_t l=0; l<na_; ++l){
                    AA[i][j][k][l] = 0.0;
                    AB[i][j][k][l] = 0.0;
                    BB[i][j][k][l] = 0.0;
                }
            }
        }
    }

    // Zero Semi-Internal Excitations
    if(T_ALGOR == "DSRG_NOSEMI"){
        outfile->Printf("\n  Exclude excitations of (active, active -> active, virtual) and (core, active -> active, active).");
        for(size_t x=0; x<na_; ++x){
            for(size_t y=0; y<na_; ++y){
                for(size_t z=0; z<na_; ++z){
                    for(size_t nm=0; nm<nc_; ++nm){
                        size_t m = nm + na_;
                        AA[m][z][y][x] = 0.0;
                        AA[z][m][y][x] = 0.0;
                        AB[m][z][y][x] = 0.0;
                        AB[z][m][y][x] = 0.0;
                        BB[m][z][y][x] = 0.0;
                        BB[z][m][y][x] = 0.0;
                    }
                    for(size_t ne=0; ne<nv_; ++ne){
                        size_t e = ne + na_;
                        AA[x][y][z][e] = 0.0;
                        AA[x][y][e][z] = 0.0;
                        AB[x][y][z][e] = 0.0;
                        AB[x][y][e][z] = 0.0;
                        BB[x][y][z][e] = 0.0;
                        BB[x][y][e][z] = 0.0;
                    }
                }
            }
        }
    }
    timer_off("Form T2");
}

void MCSRGPT2_MO::Form_T1_DSRG(d2 &A, d2 &B){
    timer_on("Form T1");
    for(size_t i=0; i<nh_; ++i){
        size_t ni = idx_h_[i];
        for(size_t a=0; a<npt_; ++a){
            size_t na = idx_p_[a];

            double Da = Fa_[ni][ni] - Fa_[na][na];
            double Db = Fb_[ni][ni] - Fb_[na][na];

            double RFa = Fa_[ni][na];
            double RFb = Fb_[ni][na];

            for(size_t u=0; u<na_; ++u){
                size_t nu = idx_a_[u];
                for(size_t x=0; x<na_; ++x){
                    size_t nx = idx_a_[x];

                    double A_Da = Fa_[nx][nx] - Fa_[nu][nu];
                    double A_Db = Fb_[nx][nx] - Fb_[nu][nu];

                    if(t1_amp_ == "SRG"){
                        double Vaa = integral_->aptei_aa(na,nx,ni,nu);
                        double Vbb = integral_->aptei_bb(na,nx,ni,nu);
                        double Vab_aa = integral_->aptei_ab(nx,na,nu,ni);
                        double Vab_ab = integral_->aptei_ab(na,nx,ni,nu);

                        double factor = 0.0;
                        factor = 1.0 - exp(s_ * (2 * Da - A_Da) * A_Da);
                        RFa -= Vaa * Da_[nx][nu] * factor;

                        factor = 1.0 - exp(s_ * (2 * Da - A_Db) * A_Db);
                        RFa -= Vab_ab * Db_[nx][nu] * factor;

                        factor = 1.0 - exp(s_ * (2 * Db - A_Da) * A_Da);
                        RFb -= Vab_aa * Da_[nx][nu] * factor;

                        factor = 1.0 - exp(s_ * (2 * Db - A_Db) * A_Db);
                        RFb -= Vbb * Db_[nx][nu] * factor;
                    }else{
                        RFa += A_Da * T2aa_[i][u][a][x] * Da_[nx][nu];
                        RFa += A_Db * T2ab_[i][u][a][x] * Db_[nx][nu];
                        RFb += A_Da * T2ab_[u][i][x][a] * Da_[nx][nu];
                        RFb += A_Db * T2bb_[i][u][a][x] * Db_[nx][nu];
                    }
                }
            }

            A[i][a] = ElementT(source_, Da, RFa);
            B[i][a] = ElementT(source_, Db, RFb);

        }
    }

    // Zero Internal Excitations
    for(size_t i=0; i<na_; ++i){
        for(size_t j=0; j<na_; ++j){
            A[i][j] = 0.0;
            B[i][j] = 0.0;
        }
    }
    timer_off("Form T1");
}

void MCSRGPT2_MO::Form_T2_SELEC(d4 &AA, d4 &AB, d4 &BB){
    timer_on("Form T2");
    for(size_t i=0; i<nc_; ++i){
        size_t ni = idx_c_[i];
        for(size_t j=0; j<nc_; ++j){
            size_t nj = idx_c_[j];
            for(size_t a=0; a<na_; ++a){
                size_t na = idx_a_[a];
                for(size_t b=0; b<na_; ++b){
                    size_t nb = idx_a_[b];

                    double Daa = Fa_[ni][ni] + Fa_[nj][nj] - Fa_[na][na] - Fa_[nb][nb];
                    double Dab = Fa_[ni][ni] + Fb_[nj][nj] - Fa_[na][na] - Fb_[nb][nb];
                    double Dbb = Fb_[ni][ni] + Fb_[nj][nj] - Fb_[na][na] - Fb_[nb][nb];

                    double Vaa = integral_->aptei_aa(na,nb,ni,nj);
                    double Vab = integral_->aptei_ab(na,nb,ni,nj);
                    double Vbb = integral_->aptei_bb(na,nb,ni,nj);

                    i += na_;
                    j += na_;
                    AA[i][j][a][b] = ElementT("STANDARD", Daa, Vaa);
                    AB[i][j][a][b] = ElementT("STANDARD", Dab, Vab);
                    BB[i][j][a][b] = ElementT("STANDARD", Dbb, Vbb);
                    i -= na_;
                    j -= na_;
                }
            }
        }
    }
    for(size_t i=0; i<nc_; ++i){
        size_t ni = idx_c_[i];
        for(size_t j=0; j<nc_; ++j){
            size_t nj = idx_c_[j];
            for(size_t a=0; a<nv_; ++a){
                size_t na = idx_v_[a];
                for(size_t b=0; b<nv_; ++b){
                    size_t nb = idx_v_[b];

                    double Daa = Fa_[ni][ni] + Fa_[nj][nj] - Fa_[na][na] - Fa_[nb][nb];
                    double Dab = Fa_[ni][ni] + Fb_[nj][nj] - Fa_[na][na] - Fb_[nb][nb];
                    double Dbb = Fb_[ni][ni] + Fb_[nj][nj] - Fb_[na][na] - Fb_[nb][nb];

                    double Vaa = integral_->aptei_aa(na,nb,ni,nj);
                    double Vab = integral_->aptei_ab(na,nb,ni,nj);
                    double Vbb = integral_->aptei_bb(na,nb,ni,nj);


                    i += na_;
                    j += na_;
                    a += na_;
                    b += na_;
                    AA[i][j][a][b] = ElementT("STANDARD", Daa, Vaa);
                    AB[i][j][a][b] = ElementT("STANDARD", Dab, Vab);
                    BB[i][j][a][b] = ElementT("STANDARD", Dbb, Vbb);
                    i -= na_;
                    j -= na_;
                    a -= na_;
                    b -= na_;
                }
            }
        }
    }
    for(size_t i=0; i<na_; ++i){
        size_t ni = idx_a_[i];
        for(size_t j=0; j<na_; ++j){
            size_t nj = idx_a_[j];
            for(size_t a=0; a<nv_; ++a){
                size_t na = idx_v_[a];
                for(size_t b=0; b<nv_; ++b){
                    size_t nb = idx_v_[b];

                    double Daa = Fa_[ni][ni] + Fa_[nj][nj] - Fa_[na][na] - Fa_[nb][nb];
                    double Dab = Fa_[ni][ni] + Fb_[nj][nj] - Fa_[na][na] - Fb_[nb][nb];
                    double Dbb = Fb_[ni][ni] + Fb_[nj][nj] - Fb_[na][na] - Fb_[nb][nb];

                    double Vaa = integral_->aptei_aa(na,nb,ni,nj);
                    double Vab = integral_->aptei_ab(na,nb,ni,nj);
                    double Vbb = integral_->aptei_bb(na,nb,ni,nj);

                    a += na_;
                    b += na_;
                    AA[i][j][a][b] = ElementT("STANDARD", Daa, Vaa);
                    AB[i][j][a][b] = ElementT("STANDARD", Dab, Vab);
                    BB[i][j][a][b] = ElementT("STANDARD", Dbb, Vbb);
                    a -= na_;
                    b -= na_;
                }
            }
        }
    }
    timer_off("Form T2");
}

void MCSRGPT2_MO::Form_T2_ISA(d4 &AA, d4 &AB, d4 &BB, const double &b_const){
    timer_on("Form T2");
    for(size_t i=0; i<nh_; ++i){
        size_t ni = idx_h_[i];
        for(size_t j=0; j<nh_; ++j){
            size_t nj = idx_h_[j];
            for(size_t a=0; a<npt_; ++a){
                size_t na = idx_p_[a];
                for(size_t b=0; b<npt_; ++b){
                    size_t nb = idx_p_[b];

                    double Daa = Fa_[ni][ni] + Fa_[nj][nj] - Fa_[na][na] - Fa_[nb][nb];
                    double Dab = Fa_[ni][ni] + Fb_[nj][nj] - Fa_[na][na] - Fb_[nb][nb];
                    double Dbb = Fb_[ni][ni] + Fb_[nj][nj] - Fb_[na][na] - Fb_[nb][nb];

                    double scalar_aa = integral_->aptei_aa(na,nb,ni,nj);
                    double scalar_ab = integral_->aptei_ab(na,nb,ni,nj);
                    double scalar_bb = integral_->aptei_bb(na,nb,ni,nj);

                    AA[i][j][a][b] = scalar_aa / (Daa + b_const / Daa);
                    AB[i][j][a][b] = scalar_ab / (Dab + b_const / Dab);
                    BB[i][j][a][b] = scalar_bb / (Dbb + b_const / Dbb);
                }
            }
        }
    }

    // Zero Internal Excitations
    for(size_t i=0; i<na_; ++i){
        for(size_t j=0; j<na_; ++j){
            for(size_t k=0; k<na_; ++k){
                for(size_t l=0; l<na_; ++l){
                    AA[i][j][k][l] = 0.0;
                    AB[i][j][k][l] = 0.0;
                    BB[i][j][k][l] = 0.0;
                }
            }
        }
    }
    timer_off("Form T2");
}

void MCSRGPT2_MO::Form_T1_ISA(d2 &A, d2 &B, const double &b_const){
    timer_on("Form T1");
    for(size_t i=0; i<nh_; ++i){
        size_t ni = idx_h_[i];
        for(size_t a=0; a<npt_; ++a){
            size_t na = idx_p_[a];

            double scalar_a = Fa_[ni][na];
            double scalar_b = Fb_[ni][na];

            for(size_t u=0; u<na_; ++u){
                size_t nu = idx_a_[u];
                for(size_t x=0; x<na_; ++x){
                    size_t nx = idx_a_[x];

                    scalar_a += (Fa_[nx][nx] - Fa_[nu][nu]) * T2aa_[i][u][a][x] * Da_[nx][nu];
                    scalar_a += (Fb_[nx][nx] - Fb_[nu][nu]) * T2ab_[i][u][a][x] * Db_[nx][nu];
                    scalar_b += (Fa_[nx][nx] - Fa_[nu][nu]) * T2ab_[u][i][x][a] * Da_[nx][nu];
                    scalar_b += (Fb_[nx][nx] - Fb_[nu][nu]) * T2bb_[i][u][a][x] * Db_[nx][nu];
                }
            }

            double delta_a = Fa_[ni][ni] - Fa_[na][na];
            double delta_b = Fb_[ni][ni] - Fb_[na][na];

            A[i][a] = scalar_a / (delta_a + b_const / delta_a);
            B[i][a] = scalar_b / (delta_b + b_const / delta_b);
        }
    }

    // Zero Internal Excitations
    for(size_t i=0; i<na_; ++i){
        for(size_t j=0; j<na_; ++j){
            A[i][j] = 0.0;
            B[i][j] = 0.0;
        }
    }
    timer_off("Form T1");
}

inline bool ReverseSortT2(const tuple<double, size_t, size_t, size_t, size_t> &lhs, const tuple<double, size_t, size_t, size_t, size_t> &rhs){
    return fabs(get<0>(rhs)) < fabs(get<0>(lhs));
}

void MCSRGPT2_MO::Check_T2(const string &x, const d4 &M, double &Norm, double &MaxT, Options &options){
    timer_on("Check T2");
    int ntamp = options.get_int("NTAMP");
    double intruder = options.get_double("INTRUDER_TAMP");
    vector<tuple<double, size_t, size_t, size_t, size_t>> Max;
    vector<tuple<double, size_t, size_t, size_t, size_t>> Large(ntamp, make_tuple(0.0, 0, 0, 0, 0));
    double value = 0.0;
    size_t count = 0;
    for(size_t i=0; i<nh_; ++i){
        size_t ni = idx_h_[i];
        for(size_t j=0; j<nh_; ++j){
            size_t nj = idx_h_[j];
            for(size_t a=0; a<npt_; ++a){
                size_t na = idx_p_[a];
                for(size_t b=0; b<npt_; ++b){
                    size_t nb = idx_p_[b];
                    double m = M[i][j][a][b];
                    value += pow(m, 2.0);
                    if(fabs(m) > fabs(get<0>(Large[ntamp-1]))){
                        Large[ntamp-1] = make_tuple(m,ni,nj,na,nb);
                    }
                    sort(Large.begin(), Large.end(), ReverseSortT2);
                    if(fabs(m) > intruder)
                        Max.push_back(make_tuple(m,ni,nj,na,nb));
                    sort(Max.begin(), Max.end(), ReverseSortT2);
                    if(fabs(m) > options.get_double("E_CONVERGENCE")) ++count;
                }
            }
        }
    }
    Norm = sqrt(value);
    MaxT = get<0>(Large[0]);

    // Print
    outfile->Printf("\n");
    outfile->Printf("\n  ==> Largest T2 amplitudes for spin case %s: <==", x.c_str());
    if(x == "AA") outfile->Printf("\n");
    if(x == "AB") outfile->Printf("\n   %3c %3c %3c %3c %9c  %3c %3c %3c %3c %9c  %3c %3c %3c %3c %9c", ' ', '_', ' ', '_', ' ', ' ', '_', ' ', '_', ' ', ' ', '_', ' ', '_', ' ');
    if(x == "BB") outfile->Printf("\n   %3c %3c %3c %3c %9c  %3c %3c %3c %3c %9c  %3c %3c %3c %3c %9c", '_', '_', '_', '_', ' ', '_', '_', '_', '_', ' ', '_', '_', '_', '_', ' ');
    outfile->Printf("\n   %3c %3c %3c %3c %9c  %3c %3c %3c %3c %9c  %3c %3c %3c %3c %9c", 'i', 'j', 'a', 'b', ' ', 'i', 'j', 'a', 'b', ' ', 'i', 'j', 'a', 'b', ' ');
    outfile->Printf("\n  --------------------------------------------------------------------------------");
    for(size_t n = 0; n != ntamp; ++n){
        if(n%3 == 0) outfile->Printf("\n  ");
        outfile->Printf("[%3zu %3zu %3zu %3zu] %8.5f ", get<1>(Large[n]), get<2>(Large[n]), get<3>(Large[n]), get<4>(Large[n]), get<0>(Large[n]));
    }
    outfile->Printf("\n  --------------------------------------------------------------------------------");
    outfile->Printf("\n  Norm of T2%s vector: (nonzero elements: %12zu) %25.15lf.", x.c_str(), count, Norm);
    outfile->Printf("\n  --------------------------------------------------------------------------------");
    outfile->Printf("\n");
    outfile->Printf("\n  ==> T2 intruder states analysis for spin case %s: <==", x.c_str());
    outfile->Printf("\n  -----------------------------------------------------------------------------------------");
    outfile->Printf("\n      Amplitude        Value   Numerator                   Denominator");
    outfile->Printf("\n  -----------------------------------------------------------------------------------------");
    for(size_t n = 0; n != Max.size(); ++n){
        size_t i = get<1>(Max[n]);
        size_t j = get<2>(Max[n]);
        size_t a = get<3>(Max[n]);
        size_t b = get<4>(Max[n]);
        double t2 = get<0>(Max[n]);
        double fi = (x != "BB") ? (Fa_[i][i]) : (Fb_[i][i]);
        double fj = (x == "AA") ? (Fa_[j][j]) : (Fb_[j][j]);
        double fa = (x != "BB") ? (Fa_[a][a]) : (Fb_[a][a]);
        double fb = (x == "AA") ? (Fa_[b][b]) : (Fb_[b][b]);
        double down = fi + fj - fa - fb;
        double up = t2 * down;
        outfile->Printf("\n  [%3zu %3zu %3zu %3zu] = %7.4f = %7.4f / (%7.4f + %7.4f - %7.4f - %7.4f = %7.4f)", i, j, a, b, t2, up, fi, fj, fa, fb, down);
    }
    outfile->Printf("\n  -----------------------------------------------------------------------------------------");
    timer_off("Check T2");
}

inline bool ReverseSortT1(const tuple<double, size_t, size_t> &lhs, const tuple<double, size_t, size_t> &rhs){
    return fabs(get<0>(rhs)) < fabs(get<0>(lhs));
}

void MCSRGPT2_MO::Check_T1(const string &x, const d2 &M, double &Norm, double &MaxT, Options &options){
    timer_on("Check T1");
    int ntamp = options.get_int("NTAMP");
    double intruder = options.get_double("INTRUDER_TAMP");
    vector<tuple<double, size_t, size_t>> Max;
    vector<tuple<double, size_t, size_t>> Large(ntamp, make_tuple(0.0, 0, 0));
    double value = 0.0;
    size_t count = 0;
    for(size_t i=0; i<nh_; ++i){
        size_t ni = idx_h_[i];
        for(size_t a=0; a<npt_; ++a){
            size_t na = idx_p_[a];
            double m = M[i][a];
            value += pow(m, 2.0);
            if(fabs(m) > fabs(get<0>(Large[ntamp-1]))){
                Large[ntamp-1] = make_tuple(m,ni,na);
            }
            sort(Large.begin(), Large.end(), ReverseSortT1);
            if(fabs(m) > intruder)
                Max.push_back(make_tuple(m,ni,na));
            sort(Max.begin(), Max.end(), ReverseSortT1);
            if(fabs(m) > options.get_double("E_CONVERGENCE")) ++count;
        }
    }
    Norm = sqrt(value);
    MaxT = get<0>(Large[0]);

    // Print
    outfile->Printf("\n");
    outfile->Printf("\n  ==> Largest T1 amplitudes for spin case %s: <==", x.c_str());
    if(x == "A") outfile->Printf("\n");
    if(x == "B") outfile->Printf("\n   %3c %3c %3c %3c %9c  %3c %3c %3c %3c %9c  %3c %3c %3c %3c %9c", '_', ' ', '_', ' ', ' ', '_', ' ', '_', ' ', ' ', '_', ' ', '_', ' ', ' ');
    outfile->Printf("\n   %3c %3c %3c %3c %9c  %3c %3c %3c %3c %9c  %3c %3c %3c %3c %9c", 'i', ' ', 'a', ' ', ' ', 'i', ' ', 'a', ' ', ' ', 'i', ' ', 'a', ' ', ' ');
    outfile->Printf("\n  --------------------------------------------------------------------------------");
    for(size_t n = 0; n != ntamp; ++n){
        if(n%3 == 0) outfile->Printf("\n  ");
        outfile->Printf("[%3zu %3c %3zu %3c] %8.5f ", get<1>(Large[n]), ' ', get<2>(Large[n]), ' ', get<0>(Large[n]));
    }
    outfile->Printf("\n  --------------------------------------------------------------------------------");
    outfile->Printf("\n  Norm of T1%s vector: (nonzero elements: %12zu) %26.15lf.", x.c_str(), count, Norm);
    outfile->Printf("\n  --------------------------------------------------------------------------------");
    outfile->Printf("\n");
    outfile->Printf("\n  ==> T1 intruder states analysis for spin case %s: <==", x.c_str());
    outfile->Printf("\n  ---------------------------------------------------------------------");
    outfile->Printf("\n      Amplitude        Value   Numerator          Denominator");
    outfile->Printf("\n  ---------------------------------------------------------------------");
    for(size_t n = 0; n != Max.size(); ++n){
        size_t i = get<1>(Max[n]);
        size_t a = get<2>(Max[n]);
        double t2 = get<0>(Max[n]);
        double fi = (x == "A") ? (Fa_[i][i]) : (Fb_[i][i]);
        double fa = (x == "A") ? (Fa_[a][a]) : (Fb_[a][a]);
        double down = fi - fa;
        double up = t2 * down;
        outfile->Printf("\n  [%3zu %3c %3zu %3c] = %7.4f = %7.4f / (%7.4f - %7.4f = %7.4f)", i, ' ', a, ' ', t2, up, fi, fa, down);
    }
    outfile->Printf("\n  ---------------------------------------------------------------------");
    timer_off("Check T1");
}

void MCSRGPT2_MO::test_D1_RE(){
    double small_threshold = 0.1;
    std::vector<std::pair<std::vector<size_t>, double>> smallD1;

    for(size_t i = 0; i < nh_; ++i){
        size_t ni = idx_h_[i];
        for(size_t a = 0; a < npt_; ++a){
            size_t na = idx_p_[a];

            // must belong to the same irrep
            if((sym_ncmo_[ni] ^ sym_ncmo_[na]) != 0) continue;

            // cannot be all active
            if(std::find(idx_a_.begin(), idx_a_.end(), ni) != idx_a_.end() &&
               std::find(idx_a_.begin(), idx_a_.end(), na) != idx_a_.end()) {
                continue;
            }else{
                double Da = Fa_[ni][ni] - Fa_[na][na];

                for(size_t k = 0; k < nh_; ++k){
                    size_t nk = idx_h_[k];
                    Da -= integral_->aptei_aa(ni, na, na, nk) * Da_[nk][ni];
                }

                for(size_t v = 0; v < na_; ++v){
                    size_t nv = idx_a_[v];
                    Da += integral_->aptei_aa(ni, nv, na, ni) * Da_[na][nv];
                }

                if(std::fabs(Da) < small_threshold){
                    smallD1.push_back(std::make_pair(std::vector<size_t> {ni,na}, Da));
                }
            }
        }
    }

//    // core-virtual block
//    for(size_t n = 0; n < nc_; ++n){
//        size_t nn = idx_c_[n];
//        for(size_t f = 0; f < nv_; ++f){
//            size_t nf = idx_v_[f];
//            if((sym_ncmo_[nn] ^ sym_ncmo_[nf]) != 0) continue;

//            double Da = Fa_[nn][nn] - Fa_[nf][nf];
//            Da -= integral_->aptei_aa(nn, nf, nf, nn);

//            if(std::fabs(Da) < small_threshold){
//                smallD1.push_back(std::make_pair(std::vector<size_t> {nn,nf}, Da));
//            }
//        }
//    }

//    // core-active block
//    for(size_t n = 0; n < nc_; ++n){
//        size_t nn = idx_c_[n];
//        for(size_t v = 0; v < na_; ++v){
//            size_t nv = idx_a_[v];
//            if((sym_ncmo_[nn] ^ sym_ncmo_[nv]) != 0) continue;

//            double Da = Fa_[nn][nn] - Fa_[nv][nv];
//            Da -= integral_->aptei_aa(nn, nv, nv, nn);
//            for(size_t y = 0; y < na_; ++y){
//                size_t ny = idx_a_[y];
//                Da += Da_[nv][ny] * integral_->aptei_aa(nn, ny, nv, nn);
//            }

//            if(std::fabs(Da) < small_threshold){
//                smallD1.push_back(std::make_pair(std::vector<size_t> {nn,nv}, Da));
//            }
//        }
//    }

//    // active-virtual block
//    for(size_t x = 0; x < na_; ++x){
//        size_t nx = idx_a_[x];
//        for(size_t f = 0; f < nv_; ++f){
//            size_t nf = idx_v_[f];
//            if((sym_ncmo_[nx] ^ sym_ncmo_[nf]) != 0) continue;

//            double Da = Fa_[nx][nx] - Fa_[nf][nf];
//            for(size_t u = 0; u < na_; ++u){
//                size_t nu = idx_a_[u];
//                Da -= Da_[nu][nx] * integral_->aptei_aa(nx, nf, nf, nu);
//            }

//            if(std::fabs(Da) < small_threshold){
//                smallD1.push_back(std::make_pair(std::vector<size_t> {nx,nf}, Da));
//            }
//        }
//    }

    // print
    print_h2("Small Denominators for T1 with RE Partitioning");
    if(smallD1.size() == 0){
        outfile->Printf("\n    NULL.");
    }else{
        std::string indent(4, ' ');
        std::string dash(47, '-');
        std::string title = indent
                + str(boost::format("%=9s    %=15s    %=15s\n") % "Indices" % "Denominator"
                      % "Original Denom.") + indent + dash;
        outfile->Printf("\n%s", title.c_str());
        for(const auto& pair: smallD1){
            size_t i = pair.first[0];
            size_t j = pair.first[1];
            double D = pair.second;
            double Dold = Fa_[i][i] - Fa_[j][j];
            outfile->Printf("\n    %4zu %4zu    %15.12f    %15.12f", i, j, D, Dold);
        }
        outfile->Printf("\n    %s", dash.c_str());
    }
}

void MCSRGPT2_MO::test_D2_RE(){
    double small_threshold = 0.5;
    std::vector<std::pair<std::vector<size_t>, double>> smallD2aa;
    std::vector<std::pair<std::vector<size_t>, double>> smallD2ab;

    for(size_t i = 0; i < nh_; ++i){
        size_t ni = idx_h_[i];
        for(size_t j = i; j < nh_; ++j){
            size_t nj = idx_h_[j];
            for(size_t a = 0; a < npt_; ++a){
                size_t na = idx_p_[a];
                for(size_t b = a; b < npt_; ++b){
                    size_t nb = idx_p_[b];

                    // product must be all symmetric
                    if((sym_ncmo_[ni] ^ sym_ncmo_[nj] ^ sym_ncmo_[na] ^ sym_ncmo_[nb]) != 0) continue;

                    // cannot be all active
                    if(std::find(idx_a_.begin(), idx_a_.end(), ni) != idx_a_.end() &&
                       std::find(idx_a_.begin(), idx_a_.end(), nj) != idx_a_.end() &&
                       std::find(idx_a_.begin(), idx_a_.end(), na) != idx_a_.end() &&
                       std::find(idx_a_.begin(), idx_a_.end(), nb) != idx_a_.end()) {
                        continue;
                    }else{
                        double Daa = Fa_[ni][ni] + Fa_[nj][nj] - Fa_[na][na] - Fa_[nb][nb];
                        double Dab = Fa_[ni][ni] + Fb_[nj][nj] - Fa_[na][na] - Fb_[nb][nb];

                        for(size_t c = 0; c < npt_; ++c){
                            size_t nc = idx_p_[c];
                            for(size_t d = 0; d < npt_; ++d){
                                size_t nd = idx_p_[d];
                                Daa -= 0.5 * integral_->aptei_aa(nc, nd, na, nb) * (1.0 - Da_[na][nc]) * (1.0 - Da_[nb][nd]);
                                Dab -= integral_->aptei_ab(nc, nd, na, nb) * (1.0 - Da_[na][nc]) * (1.0 - Db_[nb][nd]);
                            }
                        }

                        for(size_t v = 0; v < na_; ++v){
                            size_t nv = idx_a_[v];
                            for(size_t y = 0; y < na_; ++y){
                                size_t ny = idx_a_[y];
                                Daa += 0.5 * integral_->aptei_aa(nv, ny, na, nb) * Da_[na][nv] * Da_[nb][ny];
                                Dab += integral_->aptei_ab(nv, ny, na, nb) * Da_[na][nv] * Db_[nb][ny];
                            }
                        }

                        for(size_t k = 0; k < nh_; ++k){
                            size_t nk = idx_h_[k];
                            for(size_t l = 0; l < nh_; ++l){
                                size_t nl = idx_h_[l];
                                Daa -= 0.5 * integral_->aptei_aa(ni, nj, nk, nl) * Da_[nk][ni] * Da_[nl][nj];
                                Dab -= integral_->aptei_ab(ni, nj, nk, nl) * Da_[nk][ni] * Db_[nl][nj];
                            }
                        }

                        for(size_t u = 0; u < na_; ++u){
                            size_t nu = idx_a_[u];
                            for(size_t x = 0; x < na_; ++x){
                                size_t nx = idx_a_[x];
                                Daa += 0.5 * integral_->aptei_aa(ni, nj, nu, nx) * (1.0 - Da_[nu][ni]) * (1.0 - Da_[nx][nj]);
                                Dab += integral_->aptei_ab(ni, nj, nu, nx) * (1.0 - Da_[nu][ni]) * (1.0 - Db_[nx][nj]);
                            }
                        }

                        for(size_t k = 0; k < nh_; ++k){
                            size_t nk = idx_h_[k];

                            Daa -= integral_->aptei_aa(na, ni, nk, na) * Da_[nk][ni];
                            Daa -= integral_->aptei_aa(nb, ni, nk, nb) * Da_[nk][ni];
                            Daa -= integral_->aptei_aa(na, nj, nk, na) * Da_[nk][nj];
                            Daa -= integral_->aptei_aa(nb, nj, nk, nb) * Da_[nk][nj];

                            Dab -= integral_->aptei_aa(na, ni, nk, na) * Da_[nk][ni];
                            Dab += integral_->aptei_ab(ni, nb, nk, nb) * Da_[nk][ni];
                            Dab += integral_->aptei_ab(na, nj, na, nk) * Db_[nk][nj];
                            Dab -= integral_->aptei_bb(nb, nj, nk, nb) * Da_[nk][nj];
                        }

                        for(size_t y = 0; y < na_; ++y){
                            size_t ny = idx_a_[y];

                            Daa += integral_->aptei_aa(ny, ni, ni, na) * Da_[na][ny];
                            Daa += integral_->aptei_aa(ny, ni, ni, nb) * Da_[nb][ny];
                            Daa += integral_->aptei_aa(ny, nj, nj, na) * Da_[na][ny];
                            Daa += integral_->aptei_aa(ny, nj, nj, nb) * Da_[nb][ny];

                            Dab -= integral_->aptei_aa(ny, ni, ni, na) * Da_[na][ny];
                            Dab += integral_->aptei_ab(ni, ny, ni, nb) * Db_[nb][ny];
                            Dab += integral_->aptei_ab(ny, nj, na, nj) * Da_[na][ny];
                            Dab -= integral_->aptei_bb(ny, nj, nj, nb) * Db_[nb][ny];
                        }

                        if(std::fabs(Daa) < small_threshold && ni != nj && na != nb){
                            smallD2aa.push_back(std::make_pair(std::vector<size_t> {ni,nj,na,nb}, Daa));
                        }
                        if(std::fabs(Dab) < small_threshold){
                            smallD2ab.push_back(std::make_pair(std::vector<size_t> {ni,nj,na,nb}, Dab));
                        }
                    }
                }
            }
        }
    }

    // print
    print_h2("Small Denominators for T2aa with RE Partitioning");
    if(smallD2aa.size() == 0){
        outfile->Printf("\n    NULL.");
    }else{
        std::string indent(4, ' ');
        std::string dash(57, '-');
        std::string title = indent +
                str(boost::format("%=19s    %=15s    %=15s\n") % "Indices" % "Denominator"
                    % "Original Denom.") + indent + dash;
        outfile->Printf("\n%s", title.c_str());
        for(const auto& pair: smallD2aa){
            size_t i = pair.first[0];
            size_t j = pair.first[1];
            size_t k = pair.first[2];
            size_t l = pair.first[3];
            double D = pair.second;
            double Dold = Fa_[i][i] + Fa_[j][j] - Fa_[k][k] - Fa_[l][l];
            outfile->Printf("\n    %4zu %4zu %4zu %4zu    %15.12f    %15.12f",
                            i, j, k, l, D, Dold);
        }
        outfile->Printf("\n    %s", dash.c_str());
    }

    print_h2("Small Denominators for T2ab with RE Partitioning");
    if(smallD2ab.size() == 0){
        outfile->Printf("\n    NULL.");
    }else{
        std::string indent(4, ' ');
        std::string dash(57, '-');
        std::string title = indent +
                str(boost::format("%=19s    %=15s    %=15s\n") % "Indices" % "Denominator"
                    % "Original Denom.") + indent + dash;
        outfile->Printf("\n%s", title.c_str());
        for(const auto& pair: smallD2ab){
            size_t i = pair.first[0];
            size_t j = pair.first[1];
            size_t k = pair.first[2];
            size_t l = pair.first[3];
            double D = pair.second;
            double Dold = Fa_[i][i] + Fb_[j][j] - Fa_[k][k] - Fb_[l][l];
            outfile->Printf("\n    %4zu %4zu %4zu %4zu    %15.12f    %15.12f",
                            i, j, k, l, D, Dold);
        }
        outfile->Printf("\n    %s", dash.c_str());
    }
}

void MCSRGPT2_MO::test_D2_Dyall(){
    double small_threshold = 0.1;
    std::vector<std::pair<std::vector<size_t>, double>> smallD2aa;
    std::vector<std::pair<std::vector<size_t>, double>> smallD2ab;

    // core-core-active-active block
    for(size_t m = 0; m < nc_; ++m){
        size_t nm = idx_c_[m];
        for(size_t n = 0; n < nc_; ++n){
            size_t nn = idx_c_[n];
            for(size_t u = 0; u < na_; ++u){
                size_t nu = idx_a_[u];
                for(size_t v = 0; v < na_; ++v){
                    size_t nv = idx_a_[v];
                    if((sym_ncmo_[nm] ^ sym_ncmo_[nn] ^ sym_ncmo_[nu] ^ sym_ncmo_[nv]) != 0) continue;

                    double Daa = Fa_[nm][nm] + Fa_[nn][nn] - Fa_[nu][nu] - Fa_[nv][nv];
                    double Dab = Fa_[nm][nm] + Fb_[nn][nn] - Fa_[nu][nu] - Fb_[nv][nv];

                    Daa -= 0.5 * integral_->aptei_aa(nu, nv, nu, nv);
                    Dab -= 0.5 * integral_->aptei_ab(nu, nv, nu, nv);

                    for(size_t x = 0; x < na_; ++x){
                        size_t nx = idx_a_[x];
                        Daa += Da_[nu][nx] * integral_->aptei_aa(nx, nv, nu, nv);
                        Dab += Da_[nu][nx] * integral_->aptei_ab(nx, nv, nu, nv);
                    }

                    if(std::fabs(Daa) < small_threshold){
                        smallD2aa.push_back(std::make_pair(std::vector<size_t> {nm,nn,nu,nv}, Daa));
                    }
                    if(std::fabs(Dab) < small_threshold){
                        smallD2ab.push_back(std::make_pair(std::vector<size_t> {nm,nn,nu,nv}, Dab));
                    }
                }
            }
        }
    }

    // active-active-virtual-virtual block
    for(size_t x = 0; x < na_; ++x){
        size_t nx = idx_a_[x];
        for(size_t y = 0; y < na_; ++y){
            size_t ny = idx_a_[y];
            for(size_t e = 0; e < nv_; ++e){
                size_t ne = idx_v_[e];
                for(size_t f = 0; f < nv_; ++f){
                    size_t nf = idx_v_[f];
                    if((sym_ncmo_[nx] ^ sym_ncmo_[ny] ^ sym_ncmo_[ne] ^ sym_ncmo_[nf]) != 0) continue;

                    double Daa = Fa_[nx][nx] + Fa_[ny][ny] - Fa_[ne][ne] - Fa_[nf][nf];
                    double Dab = Fa_[nx][nx] + Fb_[ny][ny] - Fa_[ne][ne] - Fb_[nf][nf];

                    Daa += 0.5 * integral_->aptei_aa(nx, ny, nx, ny);
                    Dab += 0.5 * integral_->aptei_ab(nx, ny, nx, ny);

                    for(size_t u = 0; u < na_; ++u){
                        size_t nu = idx_a_[u];
                        Daa -= Da_[nu][nx] * integral_->aptei_aa(nx, ny, nu, ny);
                        Dab -= Da_[nu][nx] * integral_->aptei_ab(nx, ny, nu, ny);
                    }

                    if(std::fabs(Daa) < small_threshold){
                        smallD2aa.push_back(std::make_pair(std::vector<size_t> {nx,ny,ne,nf}, Daa));
                    }
                    if(std::fabs(Dab) < small_threshold){
                        smallD2ab.push_back(std::make_pair(std::vector<size_t> {nx,ny,ne,nf}, Dab));
                    }
                }
            }
        }
    }

    // active-core-active-virtual block
    for(size_t m = 0; m < nc_; ++m){
        size_t nm = idx_c_[m];
        for(size_t y = 0; y < na_; ++y){
            size_t ny = idx_a_[y];
            for(size_t e = 0; e < nv_; ++e){
                size_t ne = idx_v_[e];
                for(size_t v = 0; v < na_; ++v){
                    size_t nv = idx_a_[v];
                    if((sym_ncmo_[nm] ^ sym_ncmo_[ny] ^ sym_ncmo_[ne] ^ sym_ncmo_[nv]) != 0) continue;

                    double Daa = Fa_[nm][nm] + Fa_[ny][ny] - Fa_[ne][ne] - Fa_[nv][nv];
                    double D1 = Fa_[nm][nm] + Fb_[ny][ny] - Fa_[ne][ne] - Fb_[nv][nv];
                    double D2 = Fa_[nm][nm] + Fb_[ny][ny] - Fb_[ne][ne] - Fa_[nv][nv];
                    double D3 = Fb_[nm][nm] + Fa_[ny][ny] - Fa_[ne][ne] - Fb_[nv][nv];

                    for(size_t u = 0; u < na_; ++u){
                        size_t nu = idx_a_[u];
                        Daa += Da_[nu][ny] * integral_->aptei_aa(ny, nv, nu, nv);
                        Daa -= Da_[nv][nu] * integral_->aptei_aa(nu, ny, nv, ny);

                        D1 += Db_[nu][ny] * integral_->aptei_bb(ny, nv, nu, nv);
                        D1 -= Db_[nv][nu] * integral_->aptei_bb(nu, ny, nv, ny);

                        D2 += Db_[nu][ny] * integral_->aptei_ab(nv, ny, nv, nu);
                        D2 -= Da_[nv][nu] * integral_->aptei_ab(nu, ny, nv, ny);

                        D3 += Da_[nu][ny] * integral_->aptei_ab(ny, nv, nu, nv);
                        D3 -= Db_[nv][nu] * integral_->aptei_ab(ny, nu, ny, nv);
                    }

                    if(std::fabs(Daa) < small_threshold){
                        smallD2aa.push_back(std::make_pair(std::vector<size_t> {nm,ny,ne,nv}, Daa));
                    }
                    if(std::fabs(D1) < small_threshold){
                        smallD2ab.push_back(std::make_pair(std::vector<size_t> {nm,ny,ne,nv}, D1));
                    }
                    if(std::fabs(D2) < small_threshold){
                        smallD2ab.push_back(std::make_pair(std::vector<size_t> {nm,ny,nv,ne}, D2));
                    }
                    if(std::fabs(D3) < small_threshold){
                        smallD2ab.push_back(std::make_pair(std::vector<size_t> {ny,nm,ne,nv}, D3));
                    }
                }
            }
        }
    }

    // active-active-active-virtual block
    for(size_t x = 0; x < na_; ++x){
        size_t nx = idx_a_[x];
        for(size_t y = 0; y < na_; ++y){
            size_t ny = idx_a_[y];
            for(size_t z = 0; z < na_; ++z){
                size_t nz = idx_a_[z];
                for(size_t e = 0; e < nv_; ++e){
                    size_t ne = idx_v_[e];
                    if((sym_ncmo_[nx] ^ sym_ncmo_[ny] ^ sym_ncmo_[nz] ^ sym_ncmo_[ne]) != 0) continue;

                    double Daa = Fa_[nx][nx] + Fa_[ny][ny] - Fa_[nz][nz] - Fa_[ne][ne];
                    double Dab = Fa_[nx][nx] + Fb_[ny][ny] - Fa_[nz][nz] - Fb_[ne][ne];

                    Daa += 0.5 * integral_->aptei_aa(nx, ny, nx, ny);
                    Dab += 0.5 * integral_->aptei_ab(nx, ny, nx, ny);

                    for(size_t u = 0; u < na_; ++u){
                        size_t nu = idx_a_[u];

                        Daa -= Da_[nu][nx] * integral_->aptei_aa(nx, ny, nu, ny);
                        Daa += Da_[nu][nx] * integral_->aptei_aa(nx, nz, nu, nz);
                        Daa -= Da_[nz][nu] * integral_->aptei_aa(nx, nu, nx, nz);
                        Daa += Da_[nu][ny] * integral_->aptei_aa(ny, nz, nu, nz);
                        Daa -= Da_[nz][nu] * integral_->aptei_aa(ny, nu, ny, nz);

                        Dab -= Da_[nu][nx] * integral_->aptei_ab(nx, ny, nu, ny);
                        Dab += Da_[nu][nx] * integral_->aptei_aa(nx, nz, nu, nz);
                        Dab -= Da_[nz][nu] * integral_->aptei_aa(nx, nu, nx, nz);
                        Dab += Db_[nu][ny] * integral_->aptei_ab(nz, ny, nz, nu);
                        Dab -= Da_[nz][nu] * integral_->aptei_ab(nu, ny, nz, ny);
                    }

                    if(std::fabs(Daa) < small_threshold){
                        smallD2aa.push_back(std::make_pair(std::vector<size_t> {nx,ny,nz,ne}, Daa));
                    }
                    if(std::fabs(Dab) < small_threshold){
                        smallD2ab.push_back(std::make_pair(std::vector<size_t> {nx,ny,nz,ne}, Dab));
                    }
                }
            }
        }
    }

    // core-active-active-active block
    for(size_t m = 0; m < nc_; ++m){
        size_t nm = idx_c_[m];
        for(size_t w = 0; w < na_; ++w){
            size_t nw = idx_a_[w];
            for(size_t u = 0; u < na_; ++u){
                size_t nu = idx_a_[u];
                for(size_t v = 0; v < na_; ++v){
                    size_t nv = idx_a_[v];
                    if((sym_ncmo_[nm] ^ sym_ncmo_[nw] ^ sym_ncmo_[nu] ^ sym_ncmo_[nv]) != 0) continue;

                    double Daa = Fa_[nm][nm] + Fa_[nw][nw] - Fa_[nu][nu] - Fa_[nv][nv];
                    double Dab = Fa_[nm][nm] + Fb_[nw][nw] - Fa_[nu][nu] - Fb_[nv][nv];

                    Daa -= 0.5 * integral_->aptei_aa(nu, nv, nu, nv);
                    Dab -= 0.5 * integral_->aptei_ab(nu, nv, nu, nv);

                    for(size_t x = 0; x < na_; ++x){
                        size_t nx = idx_a_[x];

                        Daa += Da_[nu][nx] * integral_->aptei_aa(nx, nv, nu, nv);
                        Daa += Da_[nx][nw] * integral_->aptei_aa(nu, nw, nu, nx);
                        Daa -= Da_[nu][nx] * integral_->aptei_aa(nx, nw, nu, nw);
                        Daa += Da_[nx][nw] * integral_->aptei_aa(nw, nv, nx, nv);
                        Daa -= Da_[nv][nx] * integral_->aptei_aa(nx, nw, nv, nw);

                        Dab += Da_[nu][nx] * integral_->aptei_ab(nx, nv, nu, nv);
                        Dab += Db_[nx][nw] * integral_->aptei_ab(nu, nw, nu, nx);
                        Dab -= Da_[nu][nx] * integral_->aptei_ab(nx, nw, nu, nw);
                        Dab += Db_[nx][nw] * integral_->aptei_bb(nw, nv, nx, nv);
                        Dab -= Db_[nv][nx] * integral_->aptei_bb(nx, nw, nv, nw);
                    }

                    if(std::fabs(Daa) < small_threshold){
                        smallD2aa.push_back(std::make_pair(std::vector<size_t> {nm,nw,nu,nv}, Daa));
                    }
                    if(std::fabs(Dab) < small_threshold){
                        smallD2ab.push_back(std::make_pair(std::vector<size_t> {nm,nw,nu,nv}, Dab));
                    }
                }
            }
        }
    }

    // print
    print_h2("Small Denominators for T2aa with Dyall Partitioning");
    if(smallD2aa.size() == 0){
        outfile->Printf("\n    NULL.");
    }else{
        std::string indent(4, ' ');
        std::string dash(57, '-');
        std::string title = indent +
                str(boost::format("%=19s    %=15s    %=15s\n") % "Indices" % "Denominator"
                    % "Original Denom.") + indent + dash;
        outfile->Printf("\n%s", title.c_str());
        for(const auto& pair: smallD2aa){
            size_t i = pair.first[0];
            size_t j = pair.first[1];
            size_t k = pair.first[2];
            size_t l = pair.first[3];
            double D = pair.second;
            double Dold = Fa_[i][i] + Fa_[j][j] - Fa_[k][k] - Fa_[l][l];
            outfile->Printf("\n    %4zu %4zu %4zu %4zu    %15.12f    %15.12f",
                            i, j, k, l, D, Dold);
        }
        outfile->Printf("\n    %s", dash.c_str());
    }

    print_h2("Small Denominators for T2ab with Dyall Partitioning");
    if(smallD2ab.size() == 0){
        outfile->Printf("\n    NULL.");
    }else{
        std::string indent(4, ' ');
        std::string dash(57, '-');
        std::string title = indent +
                str(boost::format("%=19s    %=15s    %=15s\n") % "Indices" % "Denominator"
                    % "Original Denom.") + indent + dash;
        outfile->Printf("\n%s", title.c_str());
        for(const auto& pair: smallD2ab){
            size_t i = pair.first[0];
            size_t j = pair.first[1];
            size_t k = pair.first[2];
            size_t l = pair.first[3];
            double D = pair.second;
            double Dold = Fa_[i][i] + Fb_[j][j] - Fa_[k][k] - Fb_[l][l];
            outfile->Printf("\n    %4zu %4zu %4zu %4zu    %15.12f    %15.12f",
                            i, j, k, l, D, Dold);
        }
        outfile->Printf("\n    %s", dash.c_str());
    }
}

void MCSRGPT2_MO::PrintDelta(){
    ofstream out_delta;
    out_delta.open("Delta_ijab");
    for(size_t i=0; i<nh_; ++i){
        size_t ni = idx_h_[i];
        for(size_t j=i; j<nh_; ++j){
            size_t nj = idx_h_[j];
            for(size_t a=0; a<npt_; ++a){
                size_t na = idx_p_[a];
                for(size_t b=a; b<npt_; ++b){
                    size_t nb = idx_p_[b];

                    if(ni == nj && ni == na && ni == nb) continue;
                    if((sym_ncmo_[ni] ^ sym_ncmo_[nj] ^ sym_ncmo_[na] ^ sym_ncmo_[nb]) != 0) continue;
                    if(std::find(idx_a_.begin(), idx_a_.end(), ni) != idx_a_.end() &&
                       std::find(idx_a_.begin(), idx_a_.end(), nj) != idx_a_.end() &&
                       std::find(idx_a_.begin(), idx_a_.end(), na) != idx_a_.end() &&
                       std::find(idx_a_.begin(), idx_a_.end(), nb) != idx_a_.end())
                    {continue;}else{
                        double Daa = Fa_[ni][ni] + Fa_[nj][nj] - Fa_[na][na] - Fa_[nb][nb];
//                        double Dab = Fa_[ni][ni] + Fb_[nj][nj] - Fa_[na][na] - Fb_[nb][nb];
//                        double Dbb = Fb_[ni][ni] + Fb_[nj][nj] - Fb_[na][na] - Fb_[nb][nb];

                        out_delta << boost::format("%3d %3d %3d %3d %20.15f\n") % ni % nj % na % nb % Daa;
//                        out_delta << boost::format("%3d %3d %3d %3d %20.15f\n") % ni % nj % na % nb % Dab;
//                        out_delta << boost::format("%3d %3d %3d %3d %20.15f\n") % ni % nj % na % nb % Dbb;
                    }
                }
            }
        }
    }
    out_delta.close();
    out_delta.clear();
    out_delta.open("Delta_ia");
    for(size_t i=0; i<nh_; ++i){
        size_t ni = idx_h_[i];
        for(size_t a=0; a<npt_; ++a){
            size_t na = idx_p_[a];
            if(ni == na) continue;
            if((sym_ncmo_[ni] ^ sym_ncmo_[na]) != 0) continue;
            if(std::find(idx_a_.begin(), idx_a_.end(), ni) != idx_a_.end() &&
               std::find(idx_a_.begin(), idx_a_.end(), na) != idx_a_.end())
            {continue;}else{
                double delta_a = Fa_[ni][ni] - Fa_[na][na];
//                double delta_b = Fb_[ni][ni] - Fb_[na][na];
                out_delta << boost::format("%3d %3d %20.15f\n") % ni % na % delta_a;
//                out_delta << boost::format("%3d %3d %20.15f\n") % ni % na % delta_b;
            }
        }
    }
    out_delta.close();
}

double MCSRGPT2_MO::compute_energy_dsrg(){
    timer_on("E_MCDSRGPT2");
    double T1max = T1Maxa_, T2max = T2Maxaa_;
    if(fabs(T1max) < fabs(T1Maxb_)) T1max = T1Maxb_;
    if(fabs(T2max) < fabs(T2Maxab_)) T2max = T2Maxab_;
    if(fabs(T2max) < fabs(T2Maxbb_)) T2max = T2Maxbb_;
    double T1norm = sqrt(pow(T1Na_,2) + pow(T1Nb_,2));
    double T2norm = sqrt(pow(T2Naa_,2) + 4 * pow(T2Nab_,2) + pow(T2Nbb_,2));

    double E2 = 0.0;
    double E5_1 = 0.0;
    double E5_2 = 0.0;
    double E6_1 = 0.0;
    double E6_2 = 0.0;
    double E7 = 0.0;
    double E8_1 = 0.0;
    double E8_2 = 0.0;
    double E8_3 = 0.0;
    double E10_1 = 0.0;
    double E10_2 = 0.0;

    outfile->Printf("\n");
    outfile->Printf("\n  Computing energy of [F, T1] ...");
    outfile->Flush();
    E_FT1(E2);
    outfile->Printf("\t\t\t\t\tDone.");
    outfile->Flush();

    outfile->Printf("\n  Computing energy of [V, T1] and [F, T2] ...");
    outfile->Flush();
    E_VT1_FT2(E6_1,E6_2,E5_1,E5_2);
    outfile->Printf("\t\t\t\tDone.");
    outfile->Flush();

    outfile->Printf("\n  Computing energy of [V, T2] C_2^4 ...");
    outfile->Flush();
    E_VT2_2(E7);
    outfile->Printf("\t\t\t\t\tDone.");
    outfile->Flush();

    outfile->Printf("\n  Computing energy of [V, T2] C_2^2 * C_4 ...");
    outfile->Flush();
    E_VT2_4PP(E8_1);
    E_VT2_4HH(E8_2);
    E_VT2_4PH(E8_3);
    outfile->Printf("\t\t\t\tDone.");
    outfile->Flush();

    if(options_.get_str("THREEPDC") != "ZERO"){
        outfile->Printf("\n  Computing energy of [V, T2] C_2 * C_6 ...");
        outfile->Flush();
        E_VT2_6(E10_1,E10_2);
        outfile->Printf("\t\t\t\tDone.");
        outfile->Flush();
    }else{
        E10_1 = 0.0;
        E10_2 = 0.0;
    }

    double E5 = E5_1 + E5_2;
    double E6 = E6_1 + E6_2;
    double E8 = E8_1 + E8_2 + E8_3;
    double E10 = E10_1 + E10_2;
    double EVT2 = E7 + E8 + E10;

    Ecorr_ = E2 + E5 + E6 + EVT2;
    Etotal_ = Eref_ + Ecorr_;

    // Print
    outfile->Printf("\n  ");
    outfile->Printf("\n  ==> MC-DSRG-PT2 Energy Summary <==");
    outfile->Printf("\n  ");
    outfile->Printf("\n    E0 (cumulant) %15c = %22.15f", ' ', Eref_);
    outfile->Printf("\n    E([F, T1]) %18c = %22.15lf", ' ', E2);
    outfile->Printf("\n    E([V, T1]) %18c = %22.15lf", ' ', E5);
    outfile->Printf("\n    E([V, T1]: V) %15c = %22.15lf", ' ', E5_1);
    outfile->Printf("\n    E([V, T1]: C) %15c = %22.15lf", ' ', E5_2);
    outfile->Printf("\n    E([F, T2]) %18c = %22.15lf", ' ', E6);
    outfile->Printf("\n    E([F, T2]: V) %15c = %22.15lf", ' ', E6_1);
    outfile->Printf("\n    E([F, T2]: C) %15c = %22.15lf", ' ', E6_2);
    outfile->Printf("\n    E([V, T2] C_2^4) %12c = %22.15lf", ' ', E7);
    outfile->Printf("\n    E([V, T2] C_2^2 * C_4) %6c = %22.15lf", ' ', E8);
    outfile->Printf("\n    E([V, T2] C_2^2 * C_4: PP) %2c = %22.15lf", ' ', E8_1);
    outfile->Printf("\n    E([V, T2] C_2^2 * C_4: HH) %2c = %22.15lf", ' ', E8_2);
    outfile->Printf("\n    E([V, T2] C_2^2 * C_4: PH) %2c = %22.15lf", ' ', E8_3);
    outfile->Printf("\n    E([V, T2] C_2 * C_6) %8c = %22.15lf", ' ', E10);
    outfile->Printf("\n    E([V, T2] C_2 * C_6: H) %5c = %22.15lf", ' ', E10_1);
    outfile->Printf("\n    E([V, T2] C_2 * C_6: P) %5c = %22.15lf", ' ', E10_2);
    outfile->Printf("\n    E([V, T2]) %18c = %22.15lf", ' ', EVT2);
    outfile->Printf("\n    E(SRGPT2) %19c = %22.15lf", ' ', Ecorr_);
    outfile->Printf("\n  * E(Total) %20c = %22.15lf", ' ', Etotal_);
    outfile->Printf("\n    max(T1) %21c = %22.15lf", ' ', T1max);
    outfile->Printf("\n    max(T2) %21c = %22.15lf", ' ', T2max);
    outfile->Printf("\n    ||T1|| %22c = %22.15lf", ' ', T1norm);
    outfile->Printf("\n    ||T2|| %22c = %22.15lf", ' ', T2norm);
    outfile->Printf("\n    ");
    timer_off("E_MCDSRGPT2");
    return Etotal_;
}

void MCSRGPT2_MO::E_FT1(double &E){
    timer_on("[F, T1]");
    E = 0.0;
    for(size_t i=0; i<nh_; ++i){
        size_t ni = idx_h_[i];
        for(size_t j=0; j<nh_; ++j){
            size_t nj = idx_h_[j];
            for(size_t a=0; a<npt_; ++a){
                size_t na = idx_p_[a];
                for(size_t b=0; b<npt_; ++b){
                    size_t nb = idx_p_[b];
                    E += Fa_dsrg_[nb][nj] * T1a_[i][a] * Da_[nj][ni] * (Delta(na,nb) - Da_[na][nb]);
                    E += Fb_dsrg_[nb][nj] * T1b_[i][a] * Db_[nj][ni] * (Delta(na,nb) - Db_[na][nb]);
                }
            }
        }
    }
    timer_off("[F, T1]");
}

void MCSRGPT2_MO::E_VT1_FT2(double &EF1, double &EF2, double &EV1, double &EV2){
    timer_on("[F, T2] & [V, T1]");
    EF1 = 0.0;
    EF2 = 0.0;
    EV1 = 0.0;
    EV2 = 0.0;
    for(size_t u=0; u<na_; ++u){
        size_t nu = idx_a_[u];
        for(size_t v=0; v<na_; ++v){
            size_t nv = idx_a_[v];
            for(size_t x=0; x<na_; ++x){
                size_t nx = idx_a_[x];
                for(size_t y=0; y<na_; ++y){
                    size_t ny = idx_a_[y];

                    for(size_t e=0; e<nv_; ++e){
                        size_t ne = idx_v_[e];
                        size_t te = e + na_;
                        EV1 += integral_->aptei_aa(nx,ny,ne,nv) * T1a_[u][te] * L2aa_[x][y][u][v];
                        EV1 += integral_->aptei_bb(nx,ny,ne,nv) * T1b_[u][te] * L2bb_[x][y][u][v];
                        EV1 += 2 * integral_->aptei_ab(nx,ny,ne,nv) * T1a_[u][te] * L2ab_[x][y][u][v];
                        EV1 += 2 * integral_->aptei_ab(ny,nx,nv,ne) * T1b_[u][te] * L2ab_[y][x][v][u];

                        EF1 += Fa_dsrg_[ne][nx] * T2aa_[u][v][te][y] * L2aa_[x][y][u][v];
                        EF1 += Fb_dsrg_[ne][nx] * T2bb_[u][v][te][y] * L2bb_[x][y][u][v];
                        EF1 += 2 * Fa_dsrg_[ne][nx] * T2ab_[u][v][te][y] * L2ab_[x][y][u][v];
                        EF1 += 2 * Fb_dsrg_[ne][nx] * T2ab_[v][u][y][te] * L2ab_[y][x][v][u];
                    }

                    for(size_t m=0; m<nc_; ++m){
                        size_t nm = idx_c_[m];
                        size_t tm = m + na_;
                        EV2 -= integral_->aptei_aa(nm,ny,nu,nv) * T1a_[tm][x] * L2aa_[x][y][u][v];
                        EV2 -= integral_->aptei_bb(nm,ny,nu,nv) * T1b_[tm][x] * L2bb_[x][y][u][v];
                        EV2 -= 2 * integral_->aptei_ab(nm,ny,nu,nv) * T1a_[tm][x] * L2ab_[x][y][u][v];
                        EV2 -= 2 * integral_->aptei_ab(ny,nm,nv,nu) * T1b_[tm][x] * L2ab_[y][x][v][u];

                        EF2 -= Fa_dsrg_[nv][nm] * T2aa_[u][tm][x][y] * L2aa_[x][y][u][v];
                        EF2 -= Fb_dsrg_[nv][nm] * T2bb_[u][tm][x][y] * L2bb_[x][y][u][v];
                        EF2 -= 2 * Fa_dsrg_[nv][nm] * T2ab_[tm][u][y][x] * L2ab_[y][x][v][u];
                        EF2 -= 2 * Fb_dsrg_[nv][nm] * T2ab_[u][tm][x][y] * L2ab_[x][y][u][v];
                    }
                }
            }
        }
    }
    EV1 *= 0.5;
    EV2 *= 0.5;
    EF1 *= 0.5;
    EF2 *= 0.5;
    timer_off("[F, T2] & [V, T1]");
}

void MCSRGPT2_MO::E_VT2_2(double &E){
    timer_on("[V, T2] C_2^4");
    E = 0.0;
    d4 C1aa (npt_, d3(npt_, d2(nh_, d1(nh_))));
    d4 C1ab (npt_, d3(npt_, d2(nh_, d1(nh_))));
    d4 C1bb (npt_, d3(npt_, d2(nh_, d1(nh_))));
    for(size_t k=0; k<nh_; ++k){
        size_t nk = idx_h_[k];
        for(size_t l=0; l<nh_; ++l){
            size_t nl = idx_h_[l];
            for(size_t a=0; a<npt_; ++a){
                size_t na = idx_p_[a];
                for(size_t d=0; d<npt_; ++d){
                    size_t nd = idx_p_[d];
                    for(size_t c=0; c<npt_; ++c){
                        size_t nc = idx_p_[c];
                        C1aa[a][d][k][l] += integral_->aptei_aa(nk,nl,nc,nd) * (Delta(na,nc) - Da_[na][nc]);
                        C1ab[a][d][k][l] += integral_->aptei_ab(nk,nl,nc,nd) * (Delta(na,nc) - Da_[na][nc]);
                        C1bb[a][d][k][l] += integral_->aptei_bb(nk,nl,nc,nd) * (Delta(na,nc) - Db_[na][nc]);
                    }
                }
            }
        }
    }
    d4 C2aa (npt_, d3(npt_, d2(nh_, d1(nh_))));
    d4 C2ab (npt_, d3(npt_, d2(nh_, d1(nh_))));
    d4 C2bb (npt_, d3(npt_, d2(nh_, d1(nh_))));
    for(size_t k=0; k<nh_; ++k){
        for(size_t l=0; l<nh_; ++l){
            for(size_t a=0; a<npt_; ++a){
                for(size_t b=0; b<npt_; ++b){
                    size_t nb = idx_p_[b];
                    for(size_t d=0; d<npt_; ++d){
                        size_t nd = idx_p_[d];
                        C2aa[a][b][k][l] += C1aa[a][d][k][l] * (Delta(nb,nd) - Da_[nb][nd]);
                        C2ab[a][b][k][l] += C1ab[a][d][k][l] * (Delta(nb,nd) - Db_[nb][nd]);
                        C2bb[a][b][k][l] += C1bb[a][d][k][l] * (Delta(nb,nd) - Db_[nb][nd]);
                    }
                }
            }
        }
    }
    C1aa = d4(npt_, d3(npt_, d2(nh_, d1(nh_))));
    C1ab = d4(npt_, d3(npt_, d2(nh_, d1(nh_))));
    C1bb = d4(npt_, d3(npt_, d2(nh_, d1(nh_))));
    for(size_t b=0; b<npt_; ++b){
        for(size_t l=0; l<nh_; ++l){
            for(size_t a=0; a<npt_; ++a){
                for(size_t i=0; i<nh_; ++i){
                    size_t ni = idx_h_[i];
                    for(size_t k=0; k<nh_; ++k){
                        size_t nk = idx_h_[k];
                        C1aa[a][b][i][l] += C2aa[a][b][k][l] * Da_[nk][ni];
                        C1ab[a][b][i][l] += C2ab[a][b][k][l] * Da_[nk][ni];
                        C1bb[a][b][i][l] += C2bb[a][b][k][l] * Db_[nk][ni];
                    }
                }
            }
        }
    }
    C2aa = d4(npt_, d3(npt_, d2(nh_, d1(nh_))));
    C2ab = d4(npt_, d3(npt_, d2(nh_, d1(nh_))));
    C2bb = d4(npt_, d3(npt_, d2(nh_, d1(nh_))));
    for(size_t a=0; a<npt_; ++a){
        for(size_t b=0; b<npt_; ++b){
            for(size_t i=0; i<nh_; ++i){
                for(size_t j=0; j<nh_; ++j){
                    size_t nj = idx_h_[j];
                    for(size_t l=0; l<nh_; ++l){
                        size_t nl = idx_h_[l];
                        C2aa[a][b][i][j] += C1aa[a][b][i][l] * Da_[nl][nj];
                        C2ab[a][b][i][j] += C1ab[a][b][i][l] * Db_[nl][nj];
                        C2bb[a][b][i][j] += C1bb[a][b][i][l] * Db_[nl][nj];
                    }
                }
            }
        }
    }
    for(size_t i=0; i<nh_; ++i){
        for(size_t j=0; j<nh_; ++j){
            for(size_t a=0; a<npt_; ++a){
                for(size_t b=0; b<npt_; ++b){
                    E += T2aa_[i][j][a][b] * C2aa[a][b][i][j];
                    E += 4 * T2ab_[i][j][a][b] * C2ab[a][b][i][j];
                    E += T2bb_[i][j][a][b] * C2bb[a][b][i][j];
                }
            }
        }
    }
    E *= 0.25;
    timer_off("[V, T2] C_2^4");
}

void MCSRGPT2_MO::E_VT2_4PP(double &E){
    timer_on("[V, T2] C_4 * C_2^2: PP");
    E = 0.0;
    d4 C1aa (npt_, d3(npt_, d2(na_, d1(na_))));
    d4 C1ab (npt_, d3(npt_, d2(na_, d1(na_))));
    d4 C1bb (npt_, d3(npt_, d2(na_, d1(na_))));
    for(size_t x=0; x<na_; ++x){
        size_t nx = idx_a_[x];
        for(size_t y=0; y<na_; ++y){
            size_t ny = idx_a_[y];
            for(size_t d=0; d<npt_; ++d){
                size_t nd = idx_p_[d];
                for(size_t a=0; a<npt_; ++a){
                    size_t na = idx_p_[a];
                    for(size_t c=0; c<npt_; ++c){
                        size_t nc = idx_p_[c];
                        C1aa[a][d][x][y] += integral_->aptei_aa(nx,ny,nc,nd) * (Delta(na,nc) - Da_[na][nc]);
                        C1ab[a][d][x][y] += integral_->aptei_ab(nx,ny,nc,nd) * (Delta(na,nc) - Da_[na][nc]);
                        C1bb[a][d][x][y] += integral_->aptei_bb(nx,ny,nc,nd) * (Delta(na,nc) - Db_[na][nc]);
                    }
                }
            }
        }
    }
    d4 C2aa (npt_, d3(npt_, d2(na_, d1(na_))));
    d4 C2ab (npt_, d3(npt_, d2(na_, d1(na_))));
    d4 C2bb (npt_, d3(npt_, d2(na_, d1(na_))));
    for(size_t x=0; x<na_; ++x){
        for(size_t y=0; y<na_; ++y){
            for(size_t a=0; a<npt_; ++a){
                for(size_t b=0; b<npt_; ++b){
                    size_t nb = idx_p_[b];
                    for(size_t d=0; d<npt_; ++d){
                        size_t nd = idx_p_[d];
                        C2aa[a][b][x][y] += C1aa[a][d][x][y] * (Delta(nb,nd) - Da_[nb][nd]);
                        C2ab[a][b][x][y] += C1ab[a][d][x][y] * (Delta(nb,nd) - Db_[nb][nd]);
                        C2bb[a][b][x][y] += C1bb[a][d][x][y] * (Delta(nb,nd) - Db_[nb][nd]);
                    }
                }
            }
        }
    }
    C1aa = d4(npt_, d3(npt_, d2(na_, d1(na_))));
    C1ab = d4(npt_, d3(npt_, d2(na_, d1(na_))));
    C1bb = d4(npt_, d3(npt_, d2(na_, d1(na_))));
    for(size_t u=0; u<na_; ++u){
        for(size_t v=0; v<na_; ++v){
            for(size_t a=0; a<npt_; ++a){
                for(size_t b=0; b<npt_; ++b){
                    for(size_t x=0; x<na_; ++x){
                        for(size_t y=0; y<na_; ++y){
                            C1aa[a][b][u][v] += C2aa[a][b][x][y] * L2aa_[x][y][u][v];
                            C1bb[a][b][u][v] += C2bb[a][b][x][y] * L2bb_[x][y][u][v];
                            C1ab[a][b][u][v] += C2ab[a][b][x][y] * L2ab_[x][y][u][v];
                        }
                    }
                }
            }
        }
    }
    for(size_t u=0; u<na_; ++u){
        for(size_t v=0; v<na_; ++v){
            for(size_t a=0; a<npt_; ++a){
                for(size_t b=0; b<npt_; ++b){
                    E += C1aa[a][b][u][v] * T2aa_[u][v][a][b];
                    E += C1bb[a][b][u][v] * T2bb_[u][v][a][b];
                    E += 8 * C1ab[a][b][u][v] * T2ab_[u][v][a][b];
                }
            }
        }
    }
    E *= 0.125;
    timer_off("[V, T2] C_4 * C_2^2: PP");
}

void MCSRGPT2_MO::E_VT2_4HH(double &E){
    timer_on("[V, T2] C_4 * C_2^2: HH");
    E = 0.0;
    d4 C1aa (na_, d3(na_, d2(nh_, d1(nh_))));
    d4 C1ab (na_, d3(na_, d2(nh_, d1(nh_))));
    d4 C1bb (na_, d3(na_, d2(nh_, d1(nh_))));
    for(size_t u=0; u<na_; ++u){
        size_t nu = idx_a_[u];
        for(size_t v=0; v<na_; ++v){
            size_t nv = idx_a_[v];
            for(size_t l=0; l<nh_; ++l){
                size_t nl = idx_h_[l];
                for(size_t k=0; k<nh_; ++k){
                    size_t nk = idx_h_[k];
                    for(size_t i=0; i<nh_; ++i){
                        size_t ni = idx_h_[i];
                        C1aa[u][v][i][l] += integral_->aptei_aa(nk,nl,nu,nv) * Da_[nk][ni];
                        C1ab[u][v][i][l] += integral_->aptei_ab(nk,nl,nu,nv) * Da_[nk][ni];
                        C1bb[u][v][i][l] += integral_->aptei_bb(nk,nl,nu,nv) * Db_[nk][ni];
                    }
                }
            }
        }
    }
    d4 C2aa (na_, d3(na_, d2(nh_, d1(nh_))));
    d4 C2ab (na_, d3(na_, d2(nh_, d1(nh_))));
    d4 C2bb (na_, d3(na_, d2(nh_, d1(nh_))));
    for(size_t u=0; u<na_; ++u){
        for(size_t v=0; v<na_; ++v){
            for(size_t i=0; i<nh_; ++i){
                for(size_t l=0; l<nh_; ++l){
                    size_t nl = idx_h_[l];
                    for(size_t j=0; j<nh_; ++j){
                        size_t nj = idx_h_[j];
                        C2aa[u][v][i][j] += C1aa[u][v][i][l] * Da_[nl][nj];
                        C2ab[u][v][i][j] += C1ab[u][v][i][l] * Db_[nl][nj];
                        C2bb[u][v][i][j] += C1bb[u][v][i][l] * Db_[nl][nj];
                    }
                }
            }
        }
    }
    C1aa = d4(na_, d3(na_, d2(nh_, d1(nh_))));
    C1ab = d4(na_, d3(na_, d2(nh_, d1(nh_))));
    C1bb = d4(na_, d3(na_, d2(nh_, d1(nh_))));
    for(size_t x=0; x<na_; ++x){
        for(size_t y=0; y<na_; ++y){
            for(size_t i=0; i<nh_; ++i){
                for(size_t j=0; j<nh_; ++j){
                    for(size_t u=0; u<na_; ++u){
                        for(size_t v=0; v<na_; ++v){
                            C1aa[x][y][i][j] += C2aa[u][v][i][j] * L2aa_[x][y][u][v];
                            C1ab[x][y][i][j] += C2ab[u][v][i][j] * L2ab_[x][y][u][v];
                            C1bb[x][y][i][j] += C2bb[u][v][i][j] * L2bb_[x][y][u][v];
                        }
                    }
                }
            }
        }
    }
    for(size_t x=0; x<na_; ++x){
        for(size_t y=0; y<na_; ++y){
            for(size_t i=0; i<nh_; ++i){
                for(size_t j=0; j<nh_; ++j){
                    E += C1aa[x][y][i][j] * T2aa_[i][j][x][y];
                    E += 8 * C1ab[x][y][i][j] * T2ab_[i][j][x][y];
                    E += C1bb[x][y][i][j] * T2bb_[i][j][x][y];
                }
            }
        }
    }
    E *= 0.125;
    timer_off("[V, T2] C_4 * C_2^2: HH");
}

void MCSRGPT2_MO::E_VT2_4PH(double &E){
    timer_on("[V, T2] C_4 * C_2^2: PH");
    E = 0.0;
    d4 C11 (na_, d3(npt_, d2(nh_, d1(na_))));
    d4 C12 (na_, d3(npt_, d2(nh_, d1(na_))));
    d4 C13 (na_, d3(npt_, d2(nh_, d1(na_))));
    d4 C14 (na_, d3(npt_, d2(nh_, d1(na_))));
    d4 C19 (na_, d3(npt_, d2(nh_, d1(na_))));
    d4 C110 (na_, d3(npt_, d2(nh_, d1(na_))));
    for(size_t x=0; x<na_; ++x){
        size_t nx = idx_a_[x];
        for(size_t v=0; v<na_; ++v){
            size_t nv = idx_a_[v];
            for(size_t j=0; j<nh_; ++j){
                size_t nj = idx_h_[j];
                for(size_t a=0; a<npt_; ++a){
                    size_t na = idx_p_[a];
                    for(size_t b=0; b<npt_; ++b){
                        size_t nb = idx_p_[b];
                        C11[v][a][j][x] += integral_->aptei_aa(nj,nx,nv,nb) * (Delta(na,nb) - Da_[na][nb]);
                        C12[v][a][j][x] -= integral_->aptei_ab(nx,nj,nv,nb) * (Delta(na,nb) - Db_[na][nb]);
                        C13[v][a][j][x] += integral_->aptei_bb(nj,nx,nv,nb) * (Delta(na,nb) - Db_[na][nb]);
                        C14[v][a][j][x] -= integral_->aptei_ab(nj,nx,nb,nv) * (Delta(na,nb) - Da_[na][nb]);
                        C19[v][a][j][x] += integral_->aptei_ab(nx,nj,nb,nv) * (Delta(na,nb) - Da_[na][nb]);
                        C110[v][a][j][x] += integral_->aptei_ab(nj,nx,nv,nb) * (Delta(na,nb) - Db_[na][nb]);
                    }
                }
            }
        }
    }
    d4 C21 (na_, d3(npt_, d2(nh_, d1(na_))));
    d4 C22 (na_, d3(npt_, d2(nh_, d1(na_))));
    d4 C23 (na_, d3(npt_, d2(nh_, d1(na_))));
    d4 C24 (na_, d3(npt_, d2(nh_, d1(na_))));
    d4 C29 (na_, d3(npt_, d2(nh_, d1(na_))));
    d4 C210 (na_, d3(npt_, d2(nh_, d1(na_))));
    for(size_t x=0; x<na_; ++x){
        for(size_t v=0; v<na_; ++v){
            for(size_t a=0; a<npt_; ++a){
                for(size_t i=0; i<nh_; ++i){
                    size_t ni = idx_h_[i];
                    for(size_t j=0; j<nh_; ++j){
                        size_t nj = idx_h_[j];
                        C21[v][a][i][x] += C11[v][a][j][x] * Da_[nj][ni];
                        C22[v][a][i][x] += C12[v][a][j][x] * Db_[nj][ni];
                        C23[v][a][i][x] += C13[v][a][j][x] * Db_[nj][ni];
                        C24[v][a][i][x] += C14[v][a][j][x] * Da_[nj][ni];
                        C29[v][a][i][x] += C19[v][a][j][x] * Db_[nj][ni];
                        C210[v][a][i][x] += C110[v][a][j][x] * Da_[nj][ni];
                    }
                }
            }
        }
    }
    d4 C31 (na_, d3(na_, d2(na_, d1(na_))));
    d4 C32 (na_, d3(na_, d2(na_, d1(na_))));
    d4 C33 (na_, d3(na_, d2(na_, d1(na_))));
    d4 C34 (na_, d3(na_, d2(na_, d1(na_))));
    d4 C35 (na_, d3(na_, d2(na_, d1(na_))));
    d4 C36 (na_, d3(na_, d2(na_, d1(na_))));
    d4 C37 (na_, d3(na_, d2(na_, d1(na_))));
    d4 C38 (na_, d3(na_, d2(na_, d1(na_))));
    d4 C39 (na_, d3(na_, d2(na_, d1(na_))));
    d4 C310 (na_, d3(na_, d2(na_, d1(na_))));
    for(size_t u=0; u<na_; ++u){
        for(size_t y=0; y<na_; ++y){
            for(size_t x=0; x<na_; ++x){
                for(size_t v=0; v<na_; ++v){
                    for(size_t i=0; i<nh_; ++i){
                        for(size_t a=0; a<npt_; ++a){
                            C31[u][v][x][y] += C21[v][a][i][x] * T2aa_[i][u][a][y];
                            C32[u][v][x][y] += C22[v][a][i][x] * T2ab_[u][i][y][a];
                            C33[u][v][x][y] += C23[v][a][i][x] * T2bb_[i][u][a][y];
                            C34[u][v][x][y] += C24[v][a][i][x] * T2ab_[i][u][a][y];
                            C35[u][v][x][y] += C21[v][a][i][x] * T2ab_[i][u][a][y];
                            C36[u][v][x][y] += C22[v][a][i][x] * T2bb_[i][u][a][y];
                            C37[u][v][x][y] += C24[v][a][i][x] * T2aa_[i][u][a][y];
                            C38[u][v][x][y] += C23[v][a][i][x] * T2ab_[u][i][y][a];
                            C39[u][v][x][y] -= C29[v][a][i][x] * T2ab_[u][i][a][y];
                            C310[u][v][x][y] -= C210[v][a][i][x] * T2ab_[i][u][y][a];
                        }
                    }
                }
            }
        }
    }
    for(size_t u=0; u<na_; ++u){
        for(size_t v=0; v<na_; ++v){
            for(size_t x=0; x<na_; ++x){
                for(size_t y=0; y<na_; ++y){
                    E += C31[u][v][x][y] * L2aa_[x][y][u][v];
                    E += C32[u][v][x][y] * L2aa_[x][y][u][v];
                    E += C33[u][v][x][y] * L2bb_[x][y][u][v];
                    E += C34[u][v][x][y] * L2bb_[x][y][u][v];
                    E -= C35[u][v][x][y] * L2ab_[x][y][v][u];
                    E -= C36[u][v][x][y] * L2ab_[x][y][v][u];
                    E -= C37[u][v][x][y] * L2ab_[y][x][u][v];
                    E -= C38[u][v][x][y] * L2ab_[y][x][u][v];
                    E += C39[u][v][x][y] * L2ab_[x][y][u][v];
                    E += C310[u][v][x][y] * L2ab_[y][x][v][u];
                }
            }
        }
    }
    timer_off("[V, T2] C_4 * C_2^2: PH");
}

void MCSRGPT2_MO::E_VT2_6(double &E1, double &E2){
    timer_on("[V, T2] C_6 * C_2");
    E1 = 0.0;
    E2 = 0.0;
    for(size_t u=0; u<na_; ++u){
        size_t nu = idx_a_[u];
        for(size_t v=0; v<na_; ++v){
            size_t nv = idx_a_[v];
            for(size_t w=0; w<na_; ++w){
                size_t nw = idx_a_[w];
                for(size_t x=0; x<na_; ++x){
                    size_t nx = idx_a_[x];
                    for(size_t y=0; y<na_; ++y){
                        size_t ny = idx_a_[y];
                        for(size_t z=0; z<na_; ++z){
                            size_t nz = idx_a_[z];
                            for(size_t i=0; i<nh_; ++i){
                                size_t ni = idx_h_[i];
                                // L3aaa & L3bbb
                                E1 += integral_->aptei_aa(ni,nz,nu,nv) * T2aa_[i][w][x][y] * L3aaa_[x][y][z][u][v][w];
                                E1 += integral_->aptei_bb(ni,nz,nu,nv) * T2bb_[i][w][x][y] * L3bbb_[x][y][z][u][v][w];
                                // L3aab
                                E1 -= 2 * L3aab_[x][z][y][u][v][w] * T2ab_[i][w][x][y] * integral_->aptei_aa(ni,nz,nu,nv);
                                // L3aba & L3baa
                                E1 -= 2 * L3aab_[x][y][z][u][w][v] * T2aa_[i][w][x][y] * integral_->aptei_ab(ni,nz,nu,nv);
                                E1 += 4 * L3aab_[x][z][y][u][w][v] * T2ab_[w][i][x][y] * integral_->aptei_ab(nz,ni,nu,nv);
                                // L3abb & L3bab
                                E1 += 4 * L3abb_[x][y][z][u][v][w] * T2ab_[i][w][x][y] * integral_->aptei_ab(ni,nz,nu,nv);
                                E1 -= 2 * L3abb_[z][x][y][u][v][w] * T2bb_[i][w][x][y] * integral_->aptei_ab(nz,ni,nu,nv);
                                // L3bba
                                E1 -= 2 * L3abb_[x][y][z][w][u][v] * T2ab_[w][i][x][y] * integral_->aptei_bb(ni,nz,nu,nv);
                            }
                            for(size_t a=0; a<npt_; ++a){
                                size_t na = idx_p_[a];
                                // L3aaa & L3bbb
                                E2 += integral_->aptei_aa(nx,ny,nw,na) * T2aa_[u][v][a][z] * L3aaa_[x][y][z][u][v][w];
                                E2 += integral_->aptei_bb(nx,ny,nw,na) * T2bb_[u][v][a][z] * L3bbb_[x][y][z][u][v][w];
                                // L3aab
                                E2 += 2 * L3aab_[x][z][y][u][v][w] * T2aa_[u][v][a][z] * integral_->aptei_ab(nx,ny,na,nw);
                                // L3aba & L3baa
                                E2 -= 4 * L3aab_[x][z][y][u][w][v] * T2ab_[u][v][z][a] * integral_->aptei_ab(nx,ny,nw,na);
                                E2 -= 2 * L3aab_[x][y][z][u][w][v] * T2ab_[u][v][a][z] * integral_->aptei_aa(nx,ny,nw,na);
                                // L3abb & L3bab
                                E2 -= 4 * L3abb_[x][y][z][u][v][w] * T2ab_[u][v][a][z] * integral_->aptei_ab(nx,ny,na,nw);
                                E2 -= 2 * L3abb_[z][x][y][u][v][w] * T2ab_[u][v][z][a] * integral_->aptei_bb(nx,ny,nw,na);
                                // L3bba
                                E2 += 2 * L3abb_[x][y][z][w][u][v] * T2bb_[u][v][a][z] * integral_->aptei_ab(nx,ny,nw,na);
                            }
                        }
                    }
                }
            }
        }
    }
    E1 *= 0.25;
    E2 *= 0.25;
    timer_off("[V, T2] C_6 * C_2");
}

double MCSRGPT2_MO::ESRG_11(){
    double E = 0.0;

    for(size_t i = 0; i < nh_; ++i){
        size_t ni = idx_h_[i];
        for(size_t a = 0; a < npt_; ++a){
            size_t na = idx_p_[a];

            // i, a cannot all be active
            if(i < na_ && a < na_) continue;

            for(size_t j = 0; j < nh_; ++j){
                size_t nj = idx_h_[j];
                for(size_t b = 0; b < npt_; ++b){
                    size_t nb = idx_p_[b];

                    double va = 0.0, vb = 0.0;

                    double d1 = Fa_[ni][ni] - Fa_[na][na];
                    double d2 = Fa_[nb][nb] - Fa_[nj][nj];
                    va += Fa_srg_[nb][nj] * Fa_srg_[ni][na]
                            * d1 * srg_source_->compute_renormalized_denominator(d1 * d1 + d2 * d2);

                    d1 = Fb_[ni][ni] - Fb_[na][na];
                    d2 = Fb_[nb][nb] - Fb_[nj][nj];
                    va += Fb_srg_[nb][nj] * Fb_srg_[ni][na]
                            * d1 * srg_source_->compute_renormalized_denominator(d1 * d1 + d2 * d2);

                    for(size_t u = 0; u < na_; ++u){
                        size_t nu = idx_a_[u];
                        for(size_t v = 0; v < na_; ++v){
                            size_t nv = idx_a_[v];

                            d1 = Fa_[ni][ni] + Fa_[nv][nv] - Fa_[na][na] - Fa_[nu][nu];
                            d2 = Fa_[nb][nb] - Fa_[nj][nj];
                            va += Fa_srg_[nb][nj] * integral_->aptei_aa(ni,nv,na,nu) * Da_[nu][nv]
                                    * d1 * srg_source_->compute_renormalized_denominator(d1 * d1 + d2 * d2);
                            d1 = Fa_[ni][ni] + Fb_[nv][nv] - Fa_[na][na] - Fb_[nu][nu];
                            va += Fa_srg_[nb][nj] * integral_->aptei_ab(ni,nv,na,nu) * Db_[nu][nv]
                                    * d1 * srg_source_->compute_renormalized_denominator(d1 * d1 + d2 * d2);

                            d1 = Fa_[nv][nv] + Fb_[ni][ni] - Fa_[nu][nu] - Fb_[na][na];
                            d2 = Fb_[nb][nb] - Fb_[nj][nj];
                            vb += Fb_srg_[nb][nj] * integral_->aptei_ab(nv,ni,nu,na) * Da_[nu][nv]
                                    * d1 * srg_source_->compute_renormalized_denominator(d1 * d1 + d2 * d2);
                            d1 = Fb_[nv][nv] + Fb_[ni][ni] - Fb_[nu][nu] - Fb_[na][na];
                            vb += Fb_srg_[nb][nj] * integral_->aptei_bb(nv,ni,nu,na) * Db_[nu][nv]
                                    * d1 * srg_source_->compute_renormalized_denominator(d1 * d1 + d2 * d2);


                            d1 = Fa_[ni][ni] - Fa_[na][na];
                            d2 = Fa_[nb][nb] + Fa_[nv][nv] - Fa_[nj][nj] - Fa_[nu][nu];
                            va += Fa_srg_[ni][na] * integral_->aptei_aa(nb,nv,nj,nu) * Da_[nu][nv]
                                    * d1 * srg_source_->compute_renormalized_denominator(d1 * d1 + d2 * d2);
                            d2 = Fa_[nb][nb] + Fb_[nv][nv] - Fa_[nj][nj] - Fb_[nu][nu];
                            va += Fa_srg_[ni][na] * integral_->aptei_ab(nb,nv,nj,nu) * Db_[nu][nv]
                                    * d1 * srg_source_->compute_renormalized_denominator(d1 * d1 + d2 * d2);

                            d1 = Fb_[ni][ni] - Fb_[na][na];
                            d2 = Fa_[nv][nv] + Fb_[nb][nb] - Fa_[nu][nu] - Fb_[nj][nj];
                            vb += Fb_srg_[ni][na] * integral_->aptei_ab(nv,nb,nu,nj) * Da_[nu][nv]
                                    * d1 * srg_source_->compute_renormalized_denominator(d1 * d1 + d2 * d2);
                            d2 = Fb_[nv][nv] + Fb_[nb][nb] - Fb_[nu][nu] - Fb_[nj][nj];
                            vb += Fb_srg_[ni][na] * integral_->aptei_bb(nv,nb,nu,nj) * Db_[nu][nv]
                                    * d1 * srg_source_->compute_renormalized_denominator(d1 * d1 + d2 * d2);

                            for(size_t x = 0; x < na_; ++x){
                                size_t nx = idx_a_[x];
                                for(size_t y = 0; y < na_; ++y){
                                    size_t ny = idx_a_[y];

                                    d1 = Fa_[ni][ni] + Fa_[nv][nv] - Fa_[na][na] - Fa_[nu][nu];
                                    d2 = Fa_[nb][nb] + Fa_[ny][ny] - Fa_[nj][nj] - Fa_[nx][nx];
                                    va += integral_->aptei_aa(nb,ny,nj,nx) * integral_->aptei_aa(ni,nv,na,nu)
                                            * Da_[nx][ny] * Da_[nu][nv]
                                            * d1 * srg_source_->compute_renormalized_denominator(d1 * d1 + d2 * d2);

                                    d1 = Fa_[ni][ni] + Fb_[nv][nv] - Fa_[na][na] - Fb_[nu][nu];
                                    va += integral_->aptei_aa(nb,ny,nj,nx) * integral_->aptei_ab(ni,nv,na,nu)
                                            * Da_[nx][ny] * Db_[nu][nv]
                                            * d1 * srg_source_->compute_renormalized_denominator(d1 * d1 + d2 * d2);

                                    d1 = Fa_[ni][ni] + Fa_[nv][nv] - Fa_[na][na] - Fa_[nu][nu];
                                    d2 = Fa_[nb][nb] + Fb_[ny][ny] - Fa_[nj][nj] - Fb_[nx][nx];
                                    va += integral_->aptei_ab(nb,ny,nj,nx) * integral_->aptei_aa(ni,nv,na,nu)
                                            * Db_[nx][ny] * Da_[nu][nv]
                                            * d1 * srg_source_->compute_renormalized_denominator(d1 * d1 + d2 * d2);

                                    d1 = Fa_[ni][ni] + Fb_[nv][nv] - Fa_[na][na] - Fb_[nu][nu];
                                    va += integral_->aptei_ab(nb,ny,nj,nx) * integral_->aptei_ab(ni,nv,na,nu)
                                            * Db_[nx][ny] * Db_[nu][nv]
                                            * d1 * srg_source_->compute_renormalized_denominator(d1 * d1 + d2 * d2);

                                    d1 = Fb_[ni][ni] + Fb_[nv][nv] - Fb_[na][na] - Fb_[nu][nu];
                                    d2 = Fb_[nb][nb] + Fb_[ny][ny] - Fb_[nj][nj] - Fb_[nx][nx];
                                    vb += integral_->aptei_bb(nb,ny,nj,nx) * integral_->aptei_bb(ni,nv,na,nu)
                                            * Db_[nx][ny] * Db_[nu][nv]
                                            * d1 * srg_source_->compute_renormalized_denominator(d1 * d1 + d2 * d2);

                                    d1 = Fa_[nv][nv] + Fb_[ni][ni] - Fb_[nu][nu] - Fb_[na][na];
                                    vb += integral_->aptei_bb(nb,ny,nj,nx) * integral_->aptei_ab(nv,ni,nu,na)
                                            * Db_[nx][ny] * Da_[nu][nv]
                                            * d1 * srg_source_->compute_renormalized_denominator(d1 * d1 + d2 * d2);

                                    d1 = Fb_[ni][ni] + Fb_[nv][nv] - Fb_[na][na] - Fb_[nu][nu];
                                    d2 = Fa_[ny][ny] + Fb_[nb][nb] - Fa_[nx][nx] - Fb_[nj][nj];
                                    vb += integral_->aptei_ab(ny,nb,nx,nj) * integral_->aptei_bb(ni,nv,na,nu)
                                            * Da_[nx][ny] * Db_[nu][nv]
                                            * d1 * srg_source_->compute_renormalized_denominator(d1 * d1 + d2 * d2);

                                    d1 = Fa_[nv][nv] + Fb_[ni][ni] - Fb_[nu][nu] - Fb_[na][na];
                                    vb += integral_->aptei_ab(ny,nb,nx,nj) * integral_->aptei_ab(nv,ni,nu,na)
                                            * Da_[nx][ny] * Da_[nu][nv]
                                            * d1 * srg_source_->compute_renormalized_denominator(d1 * d1 + d2 * d2);
                                }
                            }
                        }
                    }

                    E += 2.0 * va * Da_[nj][ni] * (Delta(na,nb) - Da_[na][nb]);
                    E += 2.0 * vb * Db_[nj][ni] * (Delta(na,nb) - Db_[na][nb]);
                }
            }
        }
    }

    return E;
}

double MCSRGPT2_MO::ESRG_12(){
    double E = 0.0;

    for(size_t u = 0; u < na_; ++u){
        size_t nu = idx_a_[u];
        for(size_t v = 0; v < na_; ++v){
            size_t nv = idx_a_[v];
            for(size_t x = 0; x < na_; ++x){
                size_t nx = idx_a_[x];
                for(size_t y = 0; y < na_; ++y){
                    size_t ny = idx_a_[y];

                    double vaa = 0.0, vab = 0.0, vbb = 0.0;

                    // virtual
                    for(size_t e = 0; e < nv_; ++e){
                        size_t ne = idx_v_[e];

                        double d1 = Fa_[nu][nu] - Fa_[ne][ne];
                        double d2 = Fa_[ne][ne] + Fa_[nv][nv] - Fa_[nx][nx] - Fa_[ny][ny];
                        vaa += integral_->aptei_aa(ne,nv,nx,ny) * Fa_srg_[nu][ne]
                                * d1 * srg_source_->compute_renormalized_denominator(d1 * d1 + d2 * d2);

                        d1 = Fb_[nu][nu] - Fb_[ne][ne];
                        d2 = Fb_[ne][ne] + Fb_[nv][nv] - Fb_[nx][nx] - Fb_[ny][ny];
                        vbb += integral_->aptei_bb(ne,nv,nx,ny) * Fb_srg_[nu][ne]
                                * d1 * srg_source_->compute_renormalized_denominator(d1 * d1 + d2 * d2);

                        d1 = Fa_[nu][nu] - Fa_[ne][ne];
                        d2 = Fa_[ne][ne] + Fb_[nv][nv] - Fa_[nx][nx] - Fb_[ny][ny];
                        vab += integral_->aptei_ab(ne,nv,nx,ny) * Fa_srg_[nu][ne]
                                * d1 * srg_source_->compute_renormalized_denominator(d1 * d1 + d2 * d2);

                        d1 = Fb_[nv][nv] - Fb_[ne][ne];
                        d2 = Fa_[nu][nu] + Fb_[ne][ne] - Fa_[nx][nx] - Fb_[ny][ny];
                        vab += integral_->aptei_ab(nu,ne,nx,ny) * Fb_srg_[nv][ne]
                                * d1 * srg_source_->compute_renormalized_denominator(d1 * d1 + d2 * d2);

                        for(size_t w = 0; w < na_; ++w){
                            size_t nw = idx_a_[w];
                            for(size_t z = 0; z < na_; ++z){
                                size_t nz = idx_a_[z];

                                d1 = Fa_[nu][nu] + Fa_[nw][nw] - Fa_[ne][ne] - Fa_[nz][nz];
                                d2 = Fa_[ne][ne] + Fa_[nv][nv] - Fa_[nx][nx] - Fa_[ny][ny];
                                vaa += integral_->aptei_aa(ne,nv,nx,ny) * integral_->aptei_aa(nu,nw,ne,nz) * Da_[nz][nw]
                                        * d1 * srg_source_->compute_renormalized_denominator(d1 * d1 + d2 * d2);
                                d1 = Fa_[nu][nu] + Fb_[nw][nw] - Fa_[ne][ne] - Fb_[nz][nz];
                                vaa += integral_->aptei_aa(ne,nv,nx,ny) * integral_->aptei_ab(nu,nw,ne,nz) * Db_[nz][nw]
                                        * d1 * srg_source_->compute_renormalized_denominator(d1 * d1 + d2 * d2);

                                d1 = Fb_[nu][nu] + Fb_[nw][nw] - Fb_[ne][ne] - Fb_[nz][nz];
                                d2 = Fb_[ne][ne] + Fb_[nv][nv] - Fb_[nx][nx] - Fb_[ny][ny];
                                vbb += integral_->aptei_bb(ne,nv,nx,ny) * integral_->aptei_bb(nu,nw,ne,nz) * Db_[nz][nw]
                                        * d1 * srg_source_->compute_renormalized_denominator(d1 * d1 + d2 * d2);
                                d1 = Fa_[nw][nw] + Fb_[nu][nu] - Fa_[nz][nz] - Fb_[ne][ne];
                                vbb += integral_->aptei_bb(ne,nv,nx,ny) * integral_->aptei_ab(nw,nu,nz,ne) * Da_[nz][nw]
                                        * d1 * srg_source_->compute_renormalized_denominator(d1 * d1 + d2 * d2);

                                d1 = Fa_[nu][nu] + Fa_[nw][nw] - Fa_[ne][ne] - Fa_[nz][nz];
                                d2 = Fa_[ne][ne] + Fb_[nv][nv] - Fa_[nx][nx] - Fb_[ny][ny];
                                vab += integral_->aptei_ab(ne,nv,nx,ny) * integral_->aptei_aa(nu,nw,ne,nz) * Da_[nz][nw]
                                        * d1 * srg_source_->compute_renormalized_denominator(d1 * d1 + d2 * d2);
                                d1 = Fa_[nu][nu] + Fb_[nw][nw] - Fa_[ne][ne] - Fb_[nz][nz];
                                vab += integral_->aptei_ab(ne,nv,nx,ny) * integral_->aptei_ab(nu,nw,ne,nz) * Db_[nz][nw]
                                        * d1 * srg_source_->compute_renormalized_denominator(d1 * d1 + d2 * d2);

                                d1 = Fa_[nw][nw] + Fb_[nv][nv] - Fa_[nz][nz] - Fb_[ne][ne];
                                d2 = Fa_[nu][nu] + Fb_[ne][ne] - Fa_[nx][nx] - Fb_[ny][ny];
                                vab += integral_->aptei_ab(nu,ne,nx,ny) * integral_->aptei_ab(nw,nv,nz,ne) * Da_[nz][nw]
                                        * d1 * srg_source_->compute_renormalized_denominator(d1 * d1 + d2 * d2);
                                d1 = Fb_[nw][nw] + Fb_[nv][nv] - Fb_[nz][nz] - Fb_[ne][ne];
                                vab += integral_->aptei_ab(nu,ne,nx,ny) * integral_->aptei_bb(nw,nv,nz,ne) * Db_[nz][nw]
                                        * d1 * srg_source_->compute_renormalized_denominator(d1 * d1 + d2 * d2);
                            }
                        }
                    }

                    // core
                    for(size_t m = 0; m < nc_; ++m){
                        size_t nm = idx_c_[m];

                        double d1 = Fa_[nm][nm] - Fa_[nx][nx];
                        double d2 = Fa_[nu][nu] + Fa_[nv][nv] - Fa_[nm][nm] - Fa_[ny][ny];
                        vaa -= integral_->aptei_aa(nu,nv,nm,ny) * Fa_srg_[nm][nx]
                                * d1 * srg_source_->compute_renormalized_denominator(d1 * d1 + d2 * d2);

                        d1 = Fb_[nm][nm] - Fb_[nx][nx];
                        d2 = Fb_[nu][nu] + Fb_[nv][nv] - Fb_[nm][nm] - Fb_[ny][ny];
                        vbb -= integral_->aptei_bb(nu,nv,nm,ny) * Fb_srg_[nm][nx]
                                * d1 * srg_source_->compute_renormalized_denominator(d1 * d1 + d2 * d2);

                        d1 = Fa_[nm][nm] - Fa_[nx][nx];
                        d2 = Fa_[nu][nu] + Fb_[nv][nv] - Fa_[nm][nm] - Fb_[ny][ny];
                        vab -= integral_->aptei_ab(nu,nv,nm,ny) * Fa_srg_[nm][nx]
                                * d1 * srg_source_->compute_renormalized_denominator(d1 * d1 + d2 * d2);

                        d1 = Fb_[nm][nm] - Fb_[ny][ny];
                        d2 = Fa_[nu][nu] + Fb_[nv][nv] - Fa_[nx][nx] - Fb_[nm][nm];
                        vab -= integral_->aptei_ab(nu,nv,nx,nm) * Fa_srg_[nm][ny]
                                * d1 * srg_source_->compute_renormalized_denominator(d1 * d1 + d2 * d2);

                        for(size_t w = 0; w < na_; ++w){
                            size_t nw = idx_a_[w];
                            for(size_t z = 0; z < na_; ++z){
                                size_t nz = idx_a_[z];

                                d1 = Fa_[nm][nm] + Fa_[nw][nw] - Fa_[nx][nx] - Fa_[nz][nz];
                                d2 = Fa_[nu][nu] + Fa_[nv][nv] - Fa_[nm][nm] - Fa_[ny][ny];
                                vaa -= integral_->aptei_aa(nu,nv,nm,ny) * integral_->aptei_aa(nm,nw,nx,nz) * Da_[nw][nz]
                                        * d1 * srg_source_->compute_renormalized_denominator(d1 * d1 + d2 * d2);
                                d1 = Fa_[nm][nm] + Fb_[nw][nw] - Fa_[nx][nx] - Fb_[nz][nz];
                                vaa -= integral_->aptei_aa(nu,nv,nm,ny) * integral_->aptei_ab(nm,nw,nx,nz) * Db_[nw][nz]
                                        * d1 * srg_source_->compute_renormalized_denominator(d1 * d1 + d2 * d2);

                                d1 = Fb_[nm][nm] + Fb_[nw][nw] - Fb_[nx][nx] - Fb_[nz][nz];
                                d2 = Fb_[nu][nu] + Fb_[nv][nv] - Fb_[nm][nm] - Fb_[ny][ny];
                                vbb -= integral_->aptei_bb(nu,nv,nm,ny) * integral_->aptei_bb(nm,nw,nx,nz) * Db_[nw][nz]
                                        * d1 * srg_source_->compute_renormalized_denominator(d1 * d1 + d2 * d2);
                                d1 = Fa_[nw][nw] + Fb_[nm][nm] - Fa_[nz][nz] - Fb_[nx][nx];
                                vbb -= integral_->aptei_bb(nu,nv,nm,ny) * integral_->aptei_ab(nw,nm,nz,nx) * Da_[nw][nz]
                                        * d1 * srg_source_->compute_renormalized_denominator(d1 * d1 + d2 * d2);

                                d1 = Fa_[nm][nm] + Fa_[nw][nw] - Fa_[nx][nx] - Fa_[nz][nz];
                                d2 = Fa_[nu][nu] + Fb_[nv][nv] - Fa_[nm][nm] - Fb_[ny][ny];
                                vab -= integral_->aptei_ab(nu,nv,nm,ny) * integral_->aptei_aa(nm,nw,nx,nz) * Da_[nw][nz]
                                        * d1 * srg_source_->compute_renormalized_denominator(d1 * d1 + d2 * d2);
                                d1 = Fa_[nm][nm] + Fb_[nw][nw] - Fa_[nx][nx] - Fb_[nz][nz];
                                vab -= integral_->aptei_ab(nu,nv,nm,ny) * integral_->aptei_ab(nm,nw,nx,nz) * Db_[nw][nz]
                                        * d1 * srg_source_->compute_renormalized_denominator(d1 * d1 + d2 * d2);

                                d1 = Fa_[nw][nw] + Fb_[nm][nm] - Fa_[nz][nz] - Fb_[ny][ny];
                                d2 = Fa_[nu][nu] + Fb_[nv][nv] - Fa_[nx][nx] - Fb_[nm][nm];
                                vab -= integral_->aptei_ab(nu,nv,nx,nm) * integral_->aptei_ab(nw,nm,nz,ny) * Da_[nw][nz]
                                        * d1 * srg_source_->compute_renormalized_denominator(d1 * d1 + d2 * d2);
                                d1 = Fb_[nw][nw] + Fb_[nm][nm] - Fb_[nz][nz] - Fb_[ny][ny];
                                vab -= integral_->aptei_ab(nu,nv,nx,nm) * integral_->aptei_bb(nw,nm,nz,ny) * Db_[nw][nz]
                                        * d1 * srg_source_->compute_renormalized_denominator(d1 * d1 + d2 * d2);
                            }
                        }
                    }

                    E += vaa * L2aa_[x][y][u][v];
                    E += 2.0 * vab * L2ab_[x][y][u][v];
                    E += vbb * L2bb_[x][y][u][v];
                }
            }
        }
    }

    return E;
}

double MCSRGPT2_MO::ESRG_21(){
    double E = 0.0;

    for(size_t u = 0; u < na_; ++u){
        size_t nu = idx_a_[u];
        for(size_t v = 0; v < na_; ++v){
            size_t nv = idx_a_[v];
            for(size_t x = 0; x < na_; ++x){
                size_t nx = idx_a_[x];
                for(size_t y = 0; y < na_; ++y){
                    size_t ny = idx_a_[y];

                    double vaa = 0.0, vab = 0.0, vbb = 0.0;

                    // virtual
                    for(size_t e = 0; e < nv_; ++e){
                        size_t ne = idx_v_[e];

                        double d1 = Fa_[nu][nu] + Fa_[nv][nv] - Fa_[ne][ne] - Fa_[ny][ny];
                        double d2 = Fa_[ne][ne] - Fa_[nx][nx];
                        vaa += Fa_srg_[ne][nx] * integral_->aptei_aa(nu,nv,ne,ny)
                                * d1 * srg_source_->compute_renormalized_denominator(d1 * d1 + d2 * d2);

                        d1 = Fb_[nu][nu] + Fb_[nv][nv] - Fb_[ne][ne] - Fb_[ny][ny];
                        d2 = Fb_[ne][ne] - Fb_[nx][nx];
                        vbb += Fb_srg_[ne][nx] * integral_->aptei_bb(nu,nv,ne,ny)
                                * d1 * srg_source_->compute_renormalized_denominator(d1 * d1 + d2 * d2);

                        d1 = Fa_[nu][nu] + Fb_[nv][nv] - Fa_[ne][ne] - Fb_[ny][ny];
                        d2 = Fa_[ne][ne] - Fa_[nx][nx];
                        vab += Fa_srg_[ne][nx] * integral_->aptei_ab(nu,nv,ne,ny)
                                * d1 * srg_source_->compute_renormalized_denominator(d1 * d1 + d2 * d2);

                        d1 = Fa_[nu][nu] + Fb_[nv][nv] - Fa_[nx][nx] - Fb_[ne][ne];
                        d2 = Fb_[ne][ne] - Fb_[ny][ny];
                        vab += Fb_srg_[ne][ny] * integral_->aptei_ab(nu,nv,nx,ne)
                                * d1 * srg_source_->compute_renormalized_denominator(d1 * d1 + d2 * d2);

                        for(size_t w = 0; w < na_; ++w){
                            size_t nw = idx_a_[w];
                            for(size_t z = 0; z < na_; ++z){
                                size_t nz = idx_a_[z];

                                d1 = Fa_[nu][nu] + Fa_[nv][nv] - Fa_[ne][ne] - Fa_[ny][ny];
                                d2 = Fa_[ne][ne] + Fa_[nw][nw] - Fa_[nx][nx] - Fa_[nz][nz];
                                vaa += integral_->aptei_aa(nu,nv,ne,ny) * integral_->aptei_aa(ne,nw,nx,nz) * Da_[nz][nw]
                                        * d1 * srg_source_->compute_renormalized_denominator(d1 * d1 + d2 * d2);
                                d2 = Fa_[ne][ne] + Fb_[nw][nw] - Fa_[nx][nx] - Fb_[nz][nz];
                                vaa += integral_->aptei_aa(nu,nv,ne,ny) * integral_->aptei_ab(ne,nw,nx,nz) * Db_[nz][nw]
                                        * d1 * srg_source_->compute_renormalized_denominator(d1 * d1 + d2 * d2);

                                d1 = Fb_[nu][nu] + Fb_[nv][nv] - Fb_[ne][ne] - Fb_[ny][ny];
                                d2 = Fb_[ne][ne] + Fb_[nw][nw] - Fb_[nx][nx] - Fb_[nz][nz];
                                vbb += integral_->aptei_bb(nu,nv,ne,ny) * integral_->aptei_bb(ne,nw,nx,nz) * Db_[nz][nw]
                                        * d1 * srg_source_->compute_renormalized_denominator(d1 * d1 + d2 * d2);
                                d2 = Fa_[nw][nw] + Fb_[ne][ne] - Fa_[nz][nz] - Fb_[nx][nx];
                                vbb += integral_->aptei_bb(nu,nv,ne,ny) * integral_->aptei_ab(nw,ne,nz,nx) * Da_[nz][nw]
                                        * d1 * srg_source_->compute_renormalized_denominator(d1 * d1 + d2 * d2);

                                d1 = Fa_[nu][nu] + Fb_[nv][nv] - Fa_[ne][ne] - Fb_[ny][ny];
                                d2 = Fa_[ne][ne] + Fa_[nw][nw] - Fa_[nx][nx] - Fa_[nz][nz];
                                vab += integral_->aptei_ab(nu,nv,ne,ny) * integral_->aptei_aa(ne,nw,nx,nz) * Da_[nz][nw]
                                        * d1 * srg_source_->compute_renormalized_denominator(d1 * d1 + d2 * d2);
                                d2 = Fa_[ne][ne] + Fb_[nw][nw] - Fa_[nx][nx] - Fb_[nz][nz];
                                vab += integral_->aptei_ab(nu,nv,ne,ny) * integral_->aptei_ab(ne,nw,nx,nz) * Db_[nz][nw]
                                        * d1 * srg_source_->compute_renormalized_denominator(d1 * d1 + d2 * d2);

                                d1 = Fa_[nu][nu] + Fb_[nv][nv] - Fa_[nx][nx] - Fb_[ne][ne];
                                d2 = Fa_[nw][nw] + Fb_[ne][ne] - Fa_[nz][nz] - Fb_[ny][ny];
                                vab += integral_->aptei_ab(nu,nv,nx,ne) * integral_->aptei_ab(nw,ne,nz,ny) * Da_[nz][nw]
                                        * d1 * srg_source_->compute_renormalized_denominator(d1 * d1 + d2 * d2);
                                d2 = Fb_[nw][nw] + Fb_[ne][ne] - Fb_[nz][nz] - Fb_[ny][ny];
                                vab += integral_->aptei_ab(nu,nv,nx,ne) * integral_->aptei_bb(nw,ne,nz,ny) * Db_[nz][nw]
                                        * d1 * srg_source_->compute_renormalized_denominator(d1 * d1 + d2 * d2);
                            }
                        }
                    }

                    // core
                    for(size_t m = 0; m < nc_; ++m){
                        size_t nm = idx_c_[m];

                        double d1 = Fa_[nu][nu] + Fa_[nm][nm] - Fa_[nx][nx] - Fa_[ny][ny];
                        double d2 = Fa_[nv][nv] - Fa_[nm][nm];
                        vaa -= Fa_srg_[nv][nm] * integral_->aptei_aa(nu,nm,nx,ny)
                                * d1 * srg_source_->compute_renormalized_denominator(d1 * d1 + d2 * d2);

                        d1 = Fb_[nu][nu] + Fb_[nm][nm] - Fb_[nx][nx] - Fb_[ny][ny];
                        d2 = Fb_[nv][nv] - Fb_[nm][nm];
                        vbb -= Fb_srg_[nv][nm] * integral_->aptei_bb(nu,nm,nx,ny)
                                * d1 * srg_source_->compute_renormalized_denominator(d1 * d1 + d2 * d2);

                        d1 = Fa_[nu][nu] + Fb_[nm][nm] - Fa_[nx][nx] - Fb_[ny][ny];
                        d2 = Fb_[nv][nv] - Fb_[nm][nm];
                        vab -= Fb_srg_[nv][nm] * integral_->aptei_ab(nu,nm,nx,ny)
                                * d1 * srg_source_->compute_renormalized_denominator(d1 * d1 + d2 * d2);

                        d1 = Fa_[nm][nm] + Fb_[nv][nv] - Fa_[nx][nx] - Fb_[ny][ny];
                        d2 = Fa_[nu][nu] - Fa_[nm][nm];
                        vab -= Fa_srg_[nu][nm] * integral_->aptei_ab(nm,nv,nx,ny)
                                * d1 * srg_source_->compute_renormalized_denominator(d1 * d1 + d2 * d2);

                        for(size_t w = 0; w < na_; ++w){
                            size_t nw = idx_a_[w];
                            for(size_t z = 0; z < na_; ++z){
                                size_t nz = idx_a_[z];

                                d1 = Fa_[nu][nu] + Fa_[nm][nm] - Fa_[nx][nx] - Fa_[ny][ny];
                                d2 = Fa_[nv][nv] + Fa_[nw][nw] - Fa_[nm][nm] - Fa_[nz][nz];
                                vaa -= integral_->aptei_aa(nu,nm,nx,ny) * integral_->aptei_aa(nv,nw,nm,nz) * Da_[nw][nz]
                                        * d1 * srg_source_->compute_renormalized_denominator(d1 * d1 + d2 * d2);
                                d2 = Fa_[nv][nv] + Fb_[nw][nw] - Fa_[nm][nm] - Fb_[nz][nz];
                                vaa -= integral_->aptei_aa(nu,nm,nx,ny) * integral_->aptei_ab(nv,nw,nm,nz) * Db_[nw][nz]
                                        * d1 * srg_source_->compute_renormalized_denominator(d1 * d1 + d2 * d2);

                                d1 = Fb_[nu][nu] + Fb_[nm][nm] - Fb_[nx][nx] - Fb_[ny][ny];
                                d2 = Fb_[nv][nv] + Fb_[nw][nw] - Fb_[nm][nm] - Fb_[nz][nz];
                                vbb -= integral_->aptei_bb(nu,nm,nx,ny) * integral_->aptei_bb(nv,nw,nm,nz) * Db_[nw][nz]
                                        * d1 * srg_source_->compute_renormalized_denominator(d1 * d1 + d2 * d2);
                                d2 = Fa_[nw][nw] + Fb_[nv][nv] - Fa_[nz][nz] - Fb_[nm][nm];
                                vbb -= integral_->aptei_bb(nu,nm,nx,ny) * integral_->aptei_ab(nw,nv,nz,nm) * Da_[nw][nz]
                                        * d1 * srg_source_->compute_renormalized_denominator(d1 * d1 + d2 * d2);

                                d1 = Fa_[nu][nu] + Fb_[nm][nm] - Fa_[nx][nx] - Fb_[ny][ny];
                                d2 = Fa_[nw][nw] + Fb_[nv][nv] - Fa_[nz][nz] - Fb_[nm][nm];
                                vab -= integral_->aptei_ab(nu,nm,nx,ny) * integral_->aptei_ab(nw,nv,nz,nm) * Da_[nw][nz]
                                        * d1 * srg_source_->compute_renormalized_denominator(d1 * d1 + d2 * d2);
                                d2 = Fb_[nw][nw] + Fb_[nv][nv] - Fb_[nz][nz] - Fb_[nm][nm];
                                vab -= integral_->aptei_ab(nu,nm,nx,ny) * integral_->aptei_bb(nw,nv,nz,nm) * Db_[nw][nz]
                                        * d1 * srg_source_->compute_renormalized_denominator(d1 * d1 + d2 * d2);

                                d1 = Fa_[nm][nm] + Fb_[nv][nv] - Fa_[nx][nx] - Fb_[ny][ny];
                                d2 = Fa_[nu][nu] + Fa_[nw][nw] - Fa_[nm][nm] - Fa_[nz][nz];
                                vab -= integral_->aptei_ab(nm,nv,nx,ny) * integral_->aptei_aa(nu,nw,nm,nz) * Da_[nw][nz]
                                        * d1 * srg_source_->compute_renormalized_denominator(d1 * d1 + d2 * d2);
                                d2 = Fa_[nu][nu] + Fb_[nw][nw] - Fa_[nm][nm] - Fb_[nz][nz];
                                vab -= integral_->aptei_ab(nm,nv,nx,ny) * integral_->aptei_ab(nu,nw,nm,nz) * Db_[nw][nz]
                                        * d1 * srg_source_->compute_renormalized_denominator(d1 * d1 + d2 * d2);
                            }
                        }
                    }

                    E += vaa * L2aa_[x][y][u][v];
                    E += 2.0 * vab * L2ab_[x][y][u][v];
                    E += vbb * L2bb_[x][y][u][v];
                }
            }
        }
    }

    return E;
}

double MCSRGPT2_MO::ESRG_22_2(){
    double E = 0.0;

    for(size_t i = 0; i < nh_; ++i){
        size_t ni = idx_h_[i];
        for(size_t j = 0; j < nh_; ++j){
            size_t nj = idx_h_[j];
            for(size_t a = 0; a < npt_; ++a){
                size_t na = idx_p_[a];
                for(size_t b = 0; b < npt_; ++b){
                    size_t nb = idx_p_[b];

                    // i, j, a, b cannot all be active
                    if(i < na_ && j < na_ && a < na_ && b < na_) continue;

                    for(size_t k = 0; k < nh_; ++k){
                        size_t nk = idx_h_[k];
                        for(size_t l = 0; l < nh_; ++l){
                            size_t nl = idx_h_[l];
                            for(size_t c = 0; c < npt_; ++c){
                                size_t nc = idx_p_[c];
                                for(size_t d = 0; d < npt_; ++d){
                                    size_t nd = idx_p_[d];

                                    double d1 = Fa_[ni][ni] + Fa_[nj][nj] - Fa_[na][na] - Fa_[nb][nb];
                                    double d2 = Fa_[nc][nc] + Fa_[nd][nd] - Fa_[nk][nk] - Fa_[nl][nl];
                                    E += 0.5 * integral_->aptei_aa(nc,nd,nk,nl) * integral_->aptei_aa(na,nb,ni,nj)
                                            * Da_[nk][ni] * Da_[nl][nj] * (Delta(na,nc) - Da_[na][nc]) * (Delta(nb,nd) - Da_[nb][nd])
                                            * d1 * srg_source_->compute_renormalized_denominator(d1 * d1 + d2 * d2);

                                    d1 = Fa_[ni][ni] + Fb_[nj][nj] - Fa_[na][na] - Fb_[nb][nb];
                                    d2 = Fa_[nc][nc] + Fb_[nd][nd] - Fa_[nk][nk] - Fb_[nl][nl];
                                    E += 2.0 * integral_->aptei_ab(nc,nd,nk,nl) * integral_->aptei_ab(na,nb,ni,nj)
                                            * Da_[nk][ni] * Db_[nl][nj] * (Delta(na,nc) - Da_[na][nc]) * (Delta(nb,nd) - Db_[nb][nd])
                                            * d1 * srg_source_->compute_renormalized_denominator(d1 * d1 + d2 * d2);

                                    d1 = Fb_[ni][ni] + Fb_[nj][nj] - Fb_[na][na] - Fb_[nb][nb];
                                    d2 = Fb_[nc][nc] + Fb_[nd][nd] - Fb_[nk][nk] - Fb_[nl][nl];
                                    E += 0.5 * integral_->aptei_bb(nc,nd,nk,nl) * integral_->aptei_bb(na,nb,ni,nj)
                                            * Db_[nk][ni] * Db_[nl][nj] * (Delta(na,nc) - Db_[na][nc]) * (Delta(nb,nd) - Db_[nb][nd])
                                            * d1 * srg_source_->compute_renormalized_denominator(d1 * d1 + d2 * d2);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    return E;
}

double MCSRGPT2_MO::ESRG_22_4(){
    double E = 0.0;

    for(size_t u = 0; u < na_; ++u){
        size_t nu = idx_a_[u];
        for(size_t v = 0; v < na_; ++v){
            size_t nv = idx_a_[v];
            for(size_t x = 0; x < na_; ++x){
                size_t nx = idx_a_[x];
                for(size_t y = 0; y < na_; ++y){
                    size_t ny = idx_a_[y];

                    double vaa = 0.0, vab = 0.0, vbb = 0.0;

                    // hole-hole
                    for(size_t i = 0; i < nh_; ++i){
                        size_t ni = idx_h_[i];
                        for(size_t j = 0; j < nh_; ++j){
                            size_t nj = idx_h_[j];

                            // i, j cannot all be active
                            if(i < na_ && j < na_) continue;

                            for(size_t k = 0; k < nh_; ++k){
                                size_t nk = idx_h_[k];
                                for(size_t l = 0; l < nh_; ++l){
                                    size_t nl = idx_h_[l];

                                    double d1 = Fa_[ni][ni] + Fa_[nj][nj] - Fa_[nx][nx] - Fa_[ny][ny];
                                    double d2 = Fa_[nu][nu] + Fa_[nv][nv] - Fa_[nk][nk] - Fa_[nl][nl];
                                    vaa += 0.25 * integral_->aptei_aa(nu,nv,nk,nl) * integral_->aptei_aa(ni,nj,nx,ny)
                                            * Da_[nk][ni] * Da_[nl][nj] * d1
                                            * srg_source_->compute_renormalized_denominator(d1 * d1 + d2 * d2);

                                    d1 = Fb_[ni][ni] + Fb_[nj][nj] - Fb_[nx][nx] - Fb_[ny][ny];
                                    d2 = Fb_[nu][nu] + Fb_[nv][nv] - Fb_[nk][nk] - Fb_[nl][nl];
                                    vbb += 0.25 * integral_->aptei_bb(nu,nv,nk,nl) * integral_->aptei_bb(ni,nj,nx,ny)
                                            * Db_[nk][ni] * Db_[nl][nj] * d1
                                            * srg_source_->compute_renormalized_denominator(d1 * d1 + d2 * d2);

                                    d1 = Fa_[ni][ni] + Fb_[nj][nj] - Fa_[nx][nx] - Fb_[ny][ny];
                                    d2 = Fa_[nu][nu] + Fb_[nv][nv] - Fa_[nk][nk] - Fb_[nl][nl];
                                    vab += 2.0 * integral_->aptei_ab(nu,nv,nk,nl) * integral_->aptei_ab(ni,nj,nx,ny)
                                            * Da_[nk][ni] * Db_[nl][nj] * d1
                                            * srg_source_->compute_renormalized_denominator(d1 * d1 + d2 * d2);
                                }
                            }
                        }
                    }

                    // particle-particle
                    for(size_t a = 0; a < npt_; ++a){
                        size_t na = idx_p_[a];
                        for(size_t b = 0; b < npt_; ++b){
                            size_t nb = idx_p_[b];

                            // a, b cannot all be active
                            if(a < na_ && b < na_) continue;

                            for(size_t c = 0; c < npt_; ++c){
                                size_t nc = idx_p_[c];
                                for(size_t d = 0; d < npt_; ++d){
                                    size_t nd = idx_p_[d];

                                    double d1 = Fa_[nu][nu] + Fa_[nv][nv] - Fa_[na][na] - Fa_[nb][nb];
                                    double d2 = Fa_[nc][nc] + Fa_[nd][nd] - Fa_[nx][nx] - Fa_[ny][ny];
                                    vaa += 0.25 * integral_->aptei_aa(nc,nd,nx,ny) * integral_->aptei_aa(nu,nv,na,nb)
                                            * (Delta(na,nc) - Da_[na][nc]) * (Delta(nb,nd) - Da_[nb][nd]) * d1
                                            * srg_source_->compute_renormalized_denominator(d1 * d1 + d2 * d2);

                                    d1 = Fb_[nu][nu] + Fb_[nv][nv] - Fb_[na][na] - Fb_[nb][nb];
                                    d2 = Fb_[nc][nc] + Fb_[nd][nd] - Fb_[nx][nx] - Fb_[ny][ny];
                                    vbb += 0.25 * integral_->aptei_bb(nc,nd,nx,ny) * integral_->aptei_bb(nu,nv,na,nb)
                                            * (Delta(na,nc) - Db_[na][nc]) * (Delta(nb,nd) - Db_[nb][nd]) * d1
                                            * srg_source_->compute_renormalized_denominator(d1 * d1 + d2 * d2);

                                    d1 = Fa_[nu][nu] + Fb_[nv][nv] - Fa_[na][na] - Fb_[nb][nb];
                                    d2 = Fa_[nc][nc] + Fb_[nd][nd] - Fa_[nx][nx] - Fb_[ny][ny];
                                    vab += 2.0 * integral_->aptei_ab(nc,nd,nx,ny) * integral_->aptei_ab(nu,nv,na,nb)
                                            * (Delta(na,nc) - Da_[na][nc]) * (Delta(nb,nd) - Db_[nb][nd]) * d1
                                            * srg_source_->compute_renormalized_denominator(d1 * d1 + d2 * d2);
                                }
                            }
                        }
                    }

                    // particle-hole
                    for(size_t i = 0; i < nh_; ++i){
                        size_t ni = idx_h_[i];
                        for(size_t a = 0; a < npt_; ++a){
                            size_t na = idx_p_[a];

                            // i, a cannot all be active
                            if(i < na_ && a < na_) continue;

                            for(size_t j = 0; j < nh_; ++j){
                                size_t nj = idx_h_[j];
                                for(size_t b = 0; b < npt_; ++b){
                                    size_t nb = idx_p_[b];

                                    double d1 = Fa_[ni][ni] + Fa_[nv][nv] - Fa_[na][na] - Fa_[ny][ny];
                                    double d2 = Fa_[nb][nb] + Fa_[nu][nu] - Fa_[nj][nj] - Fa_[nx][nx];
                                    vaa += 2.0 * integral_->aptei_aa(nb,nu,nj,nx) * integral_->aptei_aa(ni,nv,na,ny)
                                            * Da_[nj][ni] * (Delta(na,nb) - Da_[na][nb]) * d1
                                            * srg_source_->compute_renormalized_denominator(d1 * d1 + d2 * d2);

                                    d1 = Fa_[nv][nv] + Fb_[ni][ni] - Fa_[ny][ny] - Fb_[na][na];
                                    d2 = Fa_[nu][nu] + Fb_[nb][nb] - Fa_[nx][nx] - Fb_[nj][nj];
                                    vaa += 2.0 * integral_->aptei_ab(nu,nb,nx,nj) * integral_->aptei_ab(nv,ni,ny,na)
                                            * Db_[nj][ni] * (Delta(na,nb) - Db_[na][nb]) * d1
                                            * srg_source_->compute_renormalized_denominator(d1 * d1 + d2 * d2);


                                    d1 = Fb_[ni][ni] + Fb_[nv][nv] - Fb_[na][na] - Fb_[ny][ny];
                                    d2 = Fb_[nb][nb] + Fb_[nu][nu] - Fb_[nj][nj] - Fb_[nx][nx];
                                    vbb += 2.0 * integral_->aptei_bb(nb,nu,nj,nx) * integral_->aptei_bb(ni,nv,na,ny)
                                            * Db_[nj][ni] * (Delta(na,nb) - Db_[na][nb]) * d1
                                            * srg_source_->compute_renormalized_denominator(d1 * d1 + d2 * d2);

                                    d1 = Fa_[ni][ni] + Fb_[nv][nv] - Fa_[na][na] - Fb_[ny][ny];
                                    d2 = Fa_[nb][nb] + Fb_[nu][nu] - Fa_[nj][nj] - Fb_[nx][nx];
                                    vbb += 2.0 * integral_->aptei_ab(nb,nu,nj,nx) * integral_->aptei_ab(ni,nv,na,ny)
                                            * Da_[nj][ni] * (Delta(na,nb) - Da_[na][nb]) * d1
                                            * srg_source_->compute_renormalized_denominator(d1 * d1 + d2 * d2);


                                    d1 = Fa_[nu][nu] + Fb_[ni][ni] - Fa_[na][na] - Fb_[ny][ny];
                                    d2 = Fa_[nb][nb] + Fb_[nv][nv] - Fa_[nx][nx] - Fb_[nj][nj];
                                    vab -= 2.0 * integral_->aptei_ab(nb,nv,nx,nj) * integral_->aptei_ab(nu,ni,na,ny)
                                            * Db_[nj][ni] * (Delta(na,nb) - Da_[na][nb]) * d1
                                            * srg_source_->compute_renormalized_denominator(d1 * d1 + d2 * d2);

                                    d1 = Fa_[ni][ni] + Fb_[nv][nv] - Fa_[na][na] - Fb_[ny][ny];
                                    d2 = Fa_[nu][nu] + Fa_[nb][nb] - Fa_[nx][nx] - Fa_[nj][nj];
                                    vab += 2.0 * integral_->aptei_aa(nu,nb,nx,nj) * integral_->aptei_ab(ni,nv,na,ny)
                                            * Da_[nj][ni] * (Delta(na,nb) - Da_[na][nb]) * d1
                                            * srg_source_->compute_renormalized_denominator(d1 * d1 + d2 * d2);

                                    d1 = Fb_[ni][ni] + Fb_[nv][nv] - Fb_[na][na] - Fb_[ny][ny];
                                    d2 = Fa_[nu][nu] + Fb_[nb][nb] - Fa_[nx][nx] - Fb_[nj][nj];
                                    vab += 2.0 * integral_->aptei_ab(nu,nb,nx,nj) * integral_->aptei_bb(ni,nv,na,ny)
                                            * Db_[nj][ni] * (Delta(na,nb) - Db_[na][nb]) * d1
                                            * srg_source_->compute_renormalized_denominator(d1 * d1 + d2 * d2);

                                    d1 = Fa_[ni][ni] + Fb_[nv][nv] - Fa_[nx][nx] - Fb_[na][na];
                                    d2 = Fa_[nu][nu] + Fb_[nb][nb] - Fa_[nj][nj] - Fb_[ny][ny];
                                    vab -= 2.0 * integral_->aptei_ab(nu,nb,nj,ny) * integral_->aptei_ab(ni,nv,nx,na)
                                            * Da_[nj][ni] * (Delta(na,nb) - Db_[na][nb]) * d1
                                            * srg_source_->compute_renormalized_denominator(d1 * d1 + d2 * d2);

                                    d1 = Fa_[ni][ni] + Fa_[nu][nu] - Fa_[na][na] - Fa_[nx][nx];
                                    d2 = Fa_[nb][nb] + Fb_[nv][nv] - Fa_[nj][nj] - Fb_[ny][ny];
                                    vab += 2.0 * integral_->aptei_ab(nb,nv,nj,ny) * integral_->aptei_aa(ni,nu,na,nx)
                                            * Da_[nj][ni] * (Delta(na,nb) - Da_[na][nb]) * d1
                                            * srg_source_->compute_renormalized_denominator(d1 * d1 + d2 * d2);

                                    d1 = Fa_[nu][nu] + Fb_[ni][ni] - Fa_[nx][nx] - Fb_[na][na];
                                    d2 = Fb_[nb][nb] + Fb_[nv][nv] - Fb_[nj][nj] - Fb_[ny][ny];
                                    vab += 2.0 * integral_->aptei_bb(nb,nv,nj,ny) * integral_->aptei_ab(nu,ni,nx,na)
                                            * Db_[nj][ni] * (Delta(na,nb) - Db_[na][nb]) * d1
                                            * srg_source_->compute_renormalized_denominator(d1 * d1 + d2 * d2);
                                }
                            }
                        }
                    }

                    E += vaa * L2aa_[x][y][u][v];
                    E += vab * L2ab_[x][y][u][v];
                    E += vbb * L2bb_[x][y][u][v];
                }
            }
        }
    }

    return E;
}

double MCSRGPT2_MO::ESRG_22_6(){
    double E = 0.0;

    for(size_t u = 0; u < na_; ++u){
        size_t nu = idx_a_[u];
        for(size_t v = 0; v < na_; ++v){
            size_t nv = idx_a_[v];
            for(size_t w = 0; w < na_; ++w){
                size_t nw = idx_a_[w];
                for(size_t x = 0; x < na_; ++x){
                    size_t nx = idx_a_[x];
                    for(size_t y = 0; y < na_; ++y){
                        size_t ny = idx_a_[y];
                        for(size_t z = 0; z < na_; ++z){
                            size_t nz = idx_a_[z];

                            double vaaa = 0.0, vaab = 0.0, vabb = 0.0, vbbb = 0.0;

                            // core
                            for(size_t m = 0; m < nc_; ++m){
                                size_t nm = idx_c_[m];

                                double d1 = Fa_[nm][nm] + Fa_[nw][nw] - Fa_[nx][nx] - Fa_[ny][ny];
                                double d2 = Fa_[nu][nu] + Fa_[nv][nv] - Fa_[nm][nm] - Fa_[nz][nz];
                                vaaa += 0.5 * integral_->aptei_aa(nu,nv,nm,nz) * integral_->aptei_aa(nm,nw,nx,ny)
                                        * d1 * srg_source_->compute_renormalized_denominator(d1 * d1 + d2 * d2);

                                d1 = Fb_[nm][nm] + Fb_[nw][nw] - Fb_[nx][nx] - Fb_[ny][ny];
                                d2 = Fb_[nu][nu] + Fb_[nv][nv] - Fb_[nm][nm] - Fb_[nz][nz];
                                vbbb += 0.5 * integral_->aptei_bb(nu,nv,nm,nz) * integral_->aptei_bb(nm,nw,nx,ny)
                                        * d1 * srg_source_->compute_renormalized_denominator(d1 * d1 + d2 * d2);

                                d1 = Fb_[nm][nm] + Fa_[nv][nv] - Fa_[nx][nx] - Fb_[nz][nz];
                                d2 = Fa_[nu][nu] + Fb_[nw][nw] - Fb_[nm][nm] - Fa_[ny][ny];
                                vaab += 2.0 * integral_->aptei_ab(nu,nw,ny,nm) * integral_->aptei_ab(nv,nm,nx,nz)
                                        * d1 * srg_source_->compute_renormalized_denominator(d1 * d1 + d2 * d2);

                                d1 = Fa_[nm][nm] + Fb_[nw][nw] - Fa_[nx][nx] - Fb_[nz][nz];
                                d2 = Fa_[nu][nu] + Fa_[nv][nv] - Fa_[nm][nm] - Fa_[ny][ny];
                                vaab -= integral_->aptei_aa(nu,nv,nm,ny) * integral_->aptei_ab(nm,nw,nx,nz)
                                        * d1 * srg_source_->compute_renormalized_denominator(d1 * d1 + d2 * d2);

                                d1 = Fa_[nm][nm] + Fa_[nv][nv] - Fa_[nx][nx] - Fa_[ny][ny];
                                d2 = Fa_[nu][nu] + Fb_[nw][nw] - Fa_[nm][nm] - Fb_[nz][nz];
                                vaab -= integral_->aptei_ab(nu,nw,nm,nz) * integral_->aptei_aa(nm,nv,nx,ny)
                                        * d1 * srg_source_->compute_renormalized_denominator(d1 * d1 + d2 * d2);

                                d1 = Fa_[nm][nm] + Fb_[nw][nw] - Fa_[nx][nx] - Fb_[ny][ny];
                                d2 = Fa_[nu][nu] + Fb_[nv][nv] - Fa_[nm][nm] - Fb_[nz][nz];
                                vabb += 2.0 * integral_->aptei_ab(nu,nv,nm,nz) * integral_->aptei_ab(nm,nw,nx,ny)
                                        * d1 * srg_source_->compute_renormalized_denominator(d1 * d1 + d2 * d2);

                                d1 = Fb_[nm][nm] + Fa_[nu][nu] - Fa_[nx][nx] - Fb_[ny][ny];
                                d2 = Fb_[nw][nw] + Fb_[nv][nv] - Fb_[nm][nm] - Fb_[nz][nz];
                                vabb -= integral_->aptei_bb(nv,nw,nm,nz) * integral_->aptei_ab(nu,nm,nx,ny)
                                        * d1 * srg_source_->compute_renormalized_denominator(d1 * d1 + d2 * d2);

                                d1 = Fb_[nm][nm] + Fb_[nw][nw] - Fb_[nz][nz] - Fb_[ny][ny];
                                d2 = Fa_[nu][nu] + Fb_[nv][nv] - Fb_[nm][nm] - Fa_[nx][nx];
                                vabb -= integral_->aptei_ab(nu,nv,nx,nm) * integral_->aptei_bb(nm,nw,ny,nz)
                                        * d1 * srg_source_->compute_renormalized_denominator(d1 * d1 + d2 * d2);
                            }

                            // virtual
                            for(size_t e = 0; e < nv_; ++e){
                                size_t ne = idx_v_[e];

                                double d1 = Fa_[nu][nu] + Fa_[nv][nv] - Fa_[ne][ne] - Fa_[nz][nz];
                                double d2 = Fa_[nw][nw] + Fa_[ne][ne] - Fa_[nx][nx] - Fa_[ny][ny];
                                vaaa += 0.5 * integral_->aptei_aa(nw,ne,nx,ny) * integral_->aptei_aa(nu,nv,ne,nz)
                                        * d1 * srg_source_->compute_renormalized_denominator(d1 * d1 + d2 * d2);

                                d1 = Fb_[nu][nu] + Fb_[nv][nv] - Fb_[ne][ne] - Fb_[nz][nz];
                                d2 = Fb_[nw][nw] + Fb_[ne][ne] - Fb_[nx][nx] - Fb_[ny][ny];
                                vbbb += 0.5 * integral_->aptei_bb(nw,ne,nx,ny) * integral_->aptei_bb(nu,nv,ne,nz)
                                        * d1 * srg_source_->compute_renormalized_denominator(d1 * d1 + d2 * d2);

                                d1 = Fa_[nu][nu] + Fb_[nw][nw] - Fa_[ny][ny] - Fb_[ne][ne];
                                d2 = Fa_[nv][nv] + Fb_[ne][ne] - Fa_[nx][nx] - Fb_[nz][nz];
                                vaab -= 2.0 * integral_->aptei_ab(nv,ne,nx,nz) * integral_->aptei_ab(nu,nw,ny,ne)
                                        * d1 * srg_source_->compute_renormalized_denominator(d1 * d1 + d2 * d2);

                                d1 = Fa_[nu][nu] + Fa_[nv][nv] - Fa_[ne][ne] - Fa_[ny][ny];
                                d2 = Fa_[ne][ne] + Fb_[nw][nw] - Fa_[nx][nx] - Fb_[nz][nz];
                                vaab += integral_->aptei_ab(ne,nw,nx,nz) * integral_->aptei_aa(nu,nv,ne,ny)
                                        * d1 * srg_source_->compute_renormalized_denominator(d1 * d1 + d2 * d2);

                                d1 = Fa_[nu][nu] + Fb_[nw][nw] - Fa_[ne][ne] - Fb_[nz][nz];
                                d2 = Fa_[nv][nv] + Fa_[ne][ne] - Fa_[nx][nx] - Fa_[ny][ny];
                                vaab -= integral_->aptei_aa(nv,ne,nx,ny) * integral_->aptei_ab(nu,nw,ne,nz)
                                        * d1 * srg_source_->compute_renormalized_denominator(d1 * d1 + d2 * d2);

                                d1 = Fa_[nu][nu] + Fb_[nv][nv] - Fa_[ne][ne] - Fb_[nz][nz];
                                d2 = Fa_[ne][ne] + Fb_[nw][nw] - Fa_[nx][nx] - Fb_[ny][ny];
                                vabb -= 2.0 * integral_->aptei_ab(ne,nw,nx,ny) * integral_->aptei_ab(nu,nv,ne,nz)
                                        * d1 * srg_source_->compute_renormalized_denominator(d1 * d1 + d2 * d2);

                                d1 = Fa_[nu][nu] + Fb_[nv][nv] - Fa_[nx][nx] - Fb_[ne][ne];
                                d2 = Fb_[nw][nw] + Fb_[ne][ne] - Fb_[ny][ny] - Fb_[nz][nz];
                                vabb -= integral_->aptei_bb(nw,ne,ny,nz) * integral_->aptei_ab(nu,nv,nx,ne)
                                        * d1 * srg_source_->compute_renormalized_denominator(d1 * d1 + d2 * d2);

                                d1 = Fb_[nv][nv] + Fb_[nw][nw] - Fb_[ne][ne] - Fb_[nz][nz];
                                d2 = Fa_[nu][nu] + Fb_[ne][ne] - Fa_[nx][nx] - Fb_[ny][ny];
                                vabb += integral_->aptei_ab(nu,ne,nx,ny) * integral_->aptei_bb(nv,nw,ne,nz)
                                        * d1 * srg_source_->compute_renormalized_denominator(d1 * d1 + d2 * d2);
                            }

                            E += vaaa * L3aaa_[x][y][z][u][v][w];
                            E += vaab * L3aab_[x][y][z][u][v][w];
                            E += vabb * L3abb_[x][y][z][u][v][w];
                            E += vbbb * L3bbb_[x][y][z][u][v][w];
                        }
                    }
                }
            }
        }
    }

    return E;
}

double MCSRGPT2_MO::compute_energy_srg(){
    outfile->Printf("\n");
    outfile->Printf("\n  Computing energy of [eta1, H1] ...");
    double Esrg_11 = ESRG_11();
    outfile->Printf("\t\t\t\t\tDone.");

    outfile->Printf("\n  Computing energy of [eta1, H2] ...");
    double Esrg_12 = ESRG_12();
    outfile->Printf("\t\t\t\t\tDone.");

    outfile->Printf("\n  Computing energy of [eta2, H1] ...");
    double Esrg_21 = ESRG_21();
    outfile->Printf("\t\t\t\t\tDone.");

    outfile->Printf("\n  Computing energy of [eta2, H2] C2 ...");
    double Esrg_22_2 = ESRG_22_2();
    outfile->Printf("\t\t\t\t\tDone.");

    outfile->Printf("\n  Computing energy of [eta2, H2] C4 ...");
    double Esrg_22_4 = ESRG_22_4();
    outfile->Printf("\t\t\t\t\tDone.");

    outfile->Printf("\n  Computing energy of [eta2, H2] C6 ...");
    double Esrg_22_6 = ESRG_22_6();
    outfile->Printf("\t\t\t\t\tDone.");

    double Esrg_22 = Esrg_22_2 + Esrg_22_4 + Esrg_22_6;
    double Ecorr = Esrg_11 + Esrg_12 + Esrg_21 + Esrg_22;
    double Etotal = Ecorr + Eref_;

    std::vector<std::pair<std::string,double>> energy;
    energy.push_back({"E0 (reference)", Eref_});
    energy.push_back({"[eta_1, H_1]", Esrg_11});
    energy.push_back({"[eta_1, H_2]", Esrg_12});
    energy.push_back({"[eta_2, H_1]", Esrg_21});
    energy.push_back({"[eta_2, H_2] C2", Esrg_22_2});
    energy.push_back({"[eta_2, H_2] C4", Esrg_22_4});
    energy.push_back({"[eta_2, H_2] C6", Esrg_22_6});
    energy.push_back({"[eta_2, H_2]", Esrg_22});
    energy.push_back({"SRG-MRPT2 correlation energy", Ecorr});
    energy.push_back({"SRG-MRPT2 total energy", Etotal});

    print_h2("SRG-MRPT2 energy summary");
    for (auto& str_dim : energy){
        outfile->Printf("\n    %-30s = %22.15f",str_dim.first.c_str(),str_dim.second);
    }

    return Etotal;
}

}}
