/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2017 by its authors (see COPYING, COPYING.LESSER,
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

#include <iomanip>
#include <sstream>
#include <complex>
#include <stdlib.h>

#include "psi4/libpsi4util/process.h"
#include "psi4/libmints/molecule.h"
#include "psi4/libmints/pointgrp.h"
#include "psi4/libpsio/psio.hpp"

#include "../helpers/printing.h"
#include "tdaci.h"

using namespace psi;

/* Complex datatype */
struct _fcomplex { float re, im; };
typedef struct _fcomplex fcomplex;

/* CHEEV prototype */
extern "C" {
extern void cheev( char* jobz, char* uplo, int* n, fcomplex* a, int* lda,
                float* w, fcomplex* work, int* lwork, float* rwork, int* info );
}

namespace psi {
namespace forte {

//#ifdef _OPENMP
//#include <omp.h>
//#else
//#define omp_get_max_threads() 1
//#define omp_get_thread_num() 0
//#define omp_get_num_threads() 1
//#endif

void set_TDACI_options(ForteOptions& foptions) {
    foptions.add_int("TDACI_HOLE", 0, "");
    foptions.add_str("TDACI_PROPOGATOR", "EXACT", "Type of propogator");
    foptions.add_int("TDACI_NSTEP", 20, "Number of steps");
    foptions.add_double("TDACI_TIMESTEP", 1.0, "Timestep (as)");
    foptions.add_double("TDACI_ETA", 1e-12, "Path filtering threshold");
    foptions.add_double("TDACI_CN_CONVERGENCE", 1e-12, "Convergence threshold for CN iterations");
    foptions.add_bool("TDACI_PRINT_WFN", true, "Print coefficients to files");
    foptions.add_array("TDACI_OCC_ORB","" );
    foptions.add_int("TDACI_KRYLOV_DIM", 5, "Dimension of Krylov subspace for Lanczos method");
//    foptions.add_int("TDACI_TAYLOR_ORDER", 1, "Maximum order of taylor expansion used");
}

TDACI::TDACI(SharedWavefunction ref_wfn, Options& options,
                       std::shared_ptr<ForteIntegrals> ints,
                       std::shared_ptr<MOSpaceInfo> mo_space_info)
    : Wavefunction(options), ints_(ints), mo_space_info_(mo_space_info) {
    // Copy the wavefunction information
    shallow_copy(ref_wfn);
    wfn_ = ref_wfn;
}

TDACI::~TDACI() {}

double TDACI::compute_energy() {

    double en = 0.0;
    int nact = mo_space_info_->size("ACTIVE");
    int hole = options_.get_int("TDACI_HOLE");

    // 1. Grab an ACI wavefunction
    auto aci = std::make_shared<AdaptiveCI>(wfn_, options_, ints_, mo_space_info_);
    aci->set_quiet(true);
    aci->compute_energy();
    DeterminantHashVec aci_dets = aci->get_wavefunction();
    SharedMatrix aci_coeffs = aci->get_evecs();
    outfile->Printf("\n  ACI wavefunction built");
    
    // 2. Generate the n-1 Determinants (not just core)
  //  DeterminantHashVec ann_dets;
    for( int i = 0; i < nact; ++i ){
        annihilate_wfn(aci_dets, ann_dets,i);  
    }
    size_t nann = ann_dets.size();
    outfile->Printf("\n  size of ann dets: %zu", ann_dets.size());

    // 3. Build the full n-1 Hamiltonian
    std::vector<std::string> det_str( nann);
    std::shared_ptr<FCIIntegrals> fci_ints = aci->get_aci_ints();
    SharedMatrix full_aH = std::make_shared<Matrix>("aH",nann, nann);
    for( size_t I = 0; I < nann; ++I ){
        Determinant detI = ann_dets.get_det(I);
        det_str[I] = detI.str(nact).c_str();
        for( size_t J = I; J < nann; ++J ){
            Determinant detJ = ann_dets.get_det(J);
            double value = fci_ints->slater_rules(detI,detJ);
            full_aH->set(I,J, value);
            full_aH->set(J,I, value);
        }
    }
    save_vector(det_str, "determinants.txt");
    
    // 4. Prepare initial state by removing an electron from aci wfn 
    //DeterminantHashVec core_dets;
    SharedVector core_coeffs = std::make_shared<Vector>("init", nann);
    core_coeffs->zero();
    
    const det_hashvec& dets = aci_dets.wfn_hash();
    size_t ndet = dets.size();
    size_t ncore = 0;
    for ( size_t I = 0; I < ndet; ++I ){
        auto& detI = dets[I];
        if( detI.get_alfa_bit(hole) == true ){
            Determinant adet(detI);
            adet.set_alfa_bit(hole, false);
            size_t idx = ann_dets.get_idx(adet);
            core_coeffs->set(idx, core_coeffs->get(idx) + aci_coeffs->get(aci_dets.get_idx(detI),0) ); 
            core_dets_.add(adet);
        }  
    }
    outfile->Printf("\n  Size of initial state: %zu", core_dets_.size());    
    
    // 5. Renormalize wave function
    outfile->Printf("\n  Renormalizing wave function");
    double norm = core_coeffs->norm();
    norm = 1.0/ norm;
    core_coeffs->scale(norm);

    // 5. Propogate
    if( options_.get_str("TDACI_PROPOGATOR") == "EXACT" ){
        propogate_exact( core_coeffs, full_aH );
    } else if ( options_.get_str("TDACI_PROPOGATOR") == "CN"){
        propogate_cn( core_coeffs, full_aH );
    } else if ( options_.get_str("TDACI_PROPOGATOR") == "QCN" ){
        propogate_QCN( core_coeffs, full_aH );
    } else if ( options_.get_str("TDACI_PROPOGATOR") == "LINEAR" ){
        propogate_taylor1( core_coeffs, full_aH);
    } else if ( options_.get_str("TDACI_PROPOGATOR") == "QUADRATIC" ){
        propogate_taylor2( core_coeffs, full_aH);
    } else if (options_.get_str("TDACI_PROPOGATOR") == "RK4" ){
        propogate_RK4( core_coeffs, full_aH);
    } else if (options_.get_str("TDACI_PROPOGATOR") == "LANCZOS" ){
        propogate_lanczos( core_coeffs, full_aH);
    } else if (options_.get_str("TDACI_PROPOGATOR") == "ALL" ){
        propogate_exact( core_coeffs, full_aH );
        propogate_cn( core_coeffs, full_aH );
        propogate_taylor1( core_coeffs, full_aH);
        propogate_taylor2( core_coeffs, full_aH);
        propogate_RK4( core_coeffs, full_aH);
        propogate_QCN( core_coeffs, full_aH);
        propogate_lanczos( core_coeffs, full_aH);
    }
    

    

    return en;
}

void TDACI::propogate_exact(SharedVector C0, SharedMatrix H) {



    std::vector<int> orbs(options_["TDACI_OCC_ORB"].size());
    //std::vector<int> orbs(6);
    //for( int h = 0; h < options_["TDACI_OCC_ORB"].size(); ++h ){
    for( int h = 0; h < options_["TDACI_OCC_ORB"].size(); ++h ){
        int orb = options_["TDACI_OCC_ORB"][h].to_integer();
        orbs[h] = orb;
    }

    Timer t1;
    size_t ndet = C0->dim();

    // Diagonalize the full Hamiltonian
    SharedMatrix evecs = std::make_shared<Matrix>("evecs",ndet,ndet);
    SharedVector evals = std::make_shared<Vector>("evals",ndet);

    outfile->Printf("\n  Diagonalizing Hamiltonian");
    H->diagonalize(evecs, evals);

    int nstep = options_.get_int("TDACI_NSTEP");
    double dt = options_.get_double("TDACI_TIMESTEP");

    SharedVector ct_r = std::make_shared<Vector>("ct_R",ndet);
    SharedVector ct_i = std::make_shared<Vector>("ct_I",ndet);
    ct_r->zero();
    ct_i->zero();

    // Convert to a.u. from as
    double conv = 1.0/24.18884326505;
    double time = dt;

    std::vector<std::vector<double>> occupations(orbs.size(), std::vector<double>(nstep));
    SharedVector int1 = std::make_shared<Vector>("int1", ndet);
    SharedVector int1r = std::make_shared<Vector>("int2r", ndet);
    SharedVector int1i = std::make_shared<Vector>("int2i", ndet);
    // First multiply the evecs by the initial vector
    int1->gemv(true, 1.0, &(*evecs), &(*C0), 0.0);

    for( int n = 0; n < nstep; ++n ){
        for( int I = 0; I < ndet; ++I ){
            int1r->set(I, int1->get(I) * std::cos( -1.0 * evals->get(I) * time*conv ) ); 
            int1i->set(I, int1->get(I) * std::sin( -1.0 * evals->get(I) * time*conv ) ); 
        }
        ct_r->gemv(false, 1.0, &(*evecs), &(*int1r), 0.0);
        ct_i->gemv(false, 1.0, &(*evecs), &(*int1i), 0.0);
        

        std::vector<double> occ = compute_occupation(ct_r, ct_i, orbs);
        for( int i = 0; i < orbs.size(); ++i ){
            //occupations[i].push_back(occ[i]);
            occupations[i][n] = occ[i];
        }

        if( options_.get_bool("TDACI_PRINT_WFN")){
            if( std::abs( time - round(time) ) <= 1e-8){
                std::stringstream ss;
                ss << std::fixed << std::setprecision(3) << time;
              //  save_vector(mag,"exact_" + ss.str()+ ".txt");
                save_vector(ct_r, "exact_" + ss.str() +"_r.txt");
                save_vector(ct_i, "exact_" + ss.str() +"_i.txt");
            }
        }
        int1r->zero();
        int1i->zero();

        time += dt;
    }
    for( int i = 0; i < orbs.size(); ++i ){
        save_vector(occupations[i], "occupations_" + std::to_string(orbs[i]) + ".txt");
    }
    outfile->Printf("\n Time spent propogating (exact): %1.6f s", t1.get()); 
}

void TDACI::propogate_cn( SharedVector C0, SharedMatrix H ){

    outfile->Printf("\n  Propogating with Crank-Nicholson algorithm");

    Timer total;
    size_t ndet = C0->dim();

    int nstep = options_.get_int("TDACI_NSTEP");
    double dt = options_.get_double("TDACI_TIMESTEP");
    double conv = 1.0/24.18884326505;
    dt *= conv;
    double time = dt;


    // Copy initial state into iteratively updated vectors
    SharedVector ct_r = std::make_shared<Vector>("ct_R",ndet);
    SharedVector ct_i = std::make_shared<Vector>("ct_I",ndet);

    ct_r->copy(C0->clone());
    ct_i->zero();

    // Get the zeroed diagonal H
    SharedMatrix H0 = std::make_shared<Matrix>("H0",ndet,ndet);
    H0->copy(H->clone());
    H0->zero_diagonal();

    for( int n = 1; n <= nstep; ++n ){
//        outfile->Printf("\n  Propogating at t = %1.6f", time/conv);        
        // Form b vector
        Timer b;
        SharedVector b_r = std::make_shared<Vector>("br",ndet);
        SharedVector b_i = std::make_shared<Vector>("bi",ndet);

        //b_r->copy(ct_r->clone());
        //b_i->copy(ct_i->clone());

        b_r->gemv(false, 0.5*dt, &(*H), &(*ct_i), 0.0);
        b_i->gemv(false, -1.0*0.5*dt, &(*H), &(*ct_r), 0.0);

        b_r->add(ct_r);
        b_i->add(ct_i);
    
//        outfile->Printf("\n  Form b: %1.4f s", b.get());

        // Form D^(-1) matrix
        Timer dinv;
        SharedMatrix Dinv_r = std::make_shared<Matrix>("Dr", ndet,ndet); 
        SharedMatrix Dinv_i = std::make_shared<Matrix>("Di", ndet,ndet); 
        Dinv_r->zero();
        Dinv_i->zero();

        for( size_t I = 0; I < ndet; ++I ){
            double HII = H->get(I,I);
            Dinv_r->set(I,I, 1.0/(1.0 + HII*HII*0.25*dt*dt) );
            Dinv_i->set(I,I, (HII*dt*-0.5)/(1.0 + HII*HII*0.25*dt*dt) );
        }
//        outfile->Printf("\n  Form Dinv: %1.4f s", dinv.get());

        // Transform b my D^(-1)
        Timer db;
        // real part
        SharedVector Dbr = std::make_shared<Vector>("Dbr",ndet); 
        Dbr->gemv(false, 1.0, &(*Dinv_r), &(*b_r), 0.0); 
        Dbr->gemv(false, -1.0, &(*Dinv_i), &(*b_i), 1.0); 
        // Imag part
        SharedVector Dbi = std::make_shared<Vector>("Dbi",ndet); 
        Dbi->gemv(false, 1.0, &(*Dinv_r), &(*b_i), 0.0); 
        Dbi->gemv(false, 1.0, &(*Dinv_i), &(*b_r), 1.0); 
//        outfile->Printf("\n  Form DB: %1.4f s", db.get());

        // Converge C(t+dt)
        bool converged = false;
        SharedVector ct_r_new = std::make_shared<Vector>("ct_R",ndet);
        SharedVector ct_i_new = std::make_shared<Vector>("ct_I",ndet);

        while( !converged ){
            SharedVector r_new = std::make_shared<Vector>("ct_R",ndet);
            r_new->zero();
            r_new->gemv(false,1.0, &(*Dinv_r), &(*ct_i), 0.0);
            r_new->gemv(false,1.0, &(*Dinv_i), &(*ct_r), 1.0);
            r_new->scale(0.5*dt); 
            
            ct_r_new->copy(Dbr->clone());
            ct_r_new->gemv(false,1.0,&(*H0),&(*r_new),1.0); 
            
            SharedVector i_new = std::make_shared<Vector>("ct_R",ndet);
            i_new->zero();
            i_new->gemv(false,1.0, &(*Dinv_r), &(*ct_r), 0.0);
            i_new->gemv(false,1.0, &(*Dinv_i), &(*ct_i), 1.0);
            i_new->scale(-0.5*dt); 
            
            ct_i_new->copy(Dbi->clone());
            ct_i_new->gemv(false,1.0,&(*H0),&(*i_new),1.0); 

            // Test convergence
            SharedVector err = std::make_shared<Vector>("err",ndet);
            double norm = 0.0;  
            for( size_t I = 0; I < ndet; ++I ){
                double rn = ct_r_new->get(I);
                double in = ct_i_new->get(I);
                norm += (rn*rn + in*in);
            }
            ct_r_new->scale( 1.0/sqrt(norm));
            ct_i_new->scale( 1.0/sqrt(norm));
            for( size_t I = 0; I < ndet; ++I ){
                double rn = ct_r_new->get(I);
                double in = ct_i_new->get(I);
                double ro = ct_r->get(I);
                double io = ct_i->get(I);
                err->set(I, (rn*rn + in*in) - (ro*ro + io*io));                
            }

 //           outfile->Printf("\n  %1.9f", err->norm());
            if( err->norm() <= options_.get_double("TDACI_CN_CONVERGENCE") ){
                converged = true;
            }
            
            ct_r->copy( ct_r_new->clone() );
            ct_i->copy( ct_i_new->clone() );
        }

        std::stringstream ss;
        ss << std::fixed << std::setprecision(3) << time/conv;
        double norm = 0.0;
        SharedVector mag = std::make_shared<Vector>("mag",ndet);
        for( int I = 0; I < ndet; ++I ){
            double re = ct_r->get(I);
            double im = ct_i->get(I);
            mag->set(I, (re*re) + (im*im));
            norm += (re*re) + (im*im);
        }        
        norm = std::sqrt(norm);
//        outfile->Printf("\n norm: %1.6f", norm);
        ct_r->scale(1.0/norm);
        ct_i->scale(1.0/norm);

//        outfile->Printf("\n  norm(t=%1.3f) = %1.5f", time/conv, norm);
        
        double intp;
        if( options_.get_bool("TDACI_PRINT_WFN")){
            if( std::fabs( (time/conv) - round(time/conv)) <= 1e-8 ){
                outfile->Printf("\n Saving wavefunction for t = %1.3f as", time/conv);
              //  save_vector(mag,"CN_" + ss.str()+ ".txt");
                save_vector(ct_r,"CN_" + ss.str()+ "_r.txt");
                save_vector(ct_i,"CN_" + ss.str()+ "_i.txt");
            }
        }

        time += dt;
    }
    outfile->Printf("\n  Time spent propogating (CN): %1.6f", total.get());
}

void TDACI::propogate_taylor1(SharedVector C0, SharedMatrix H  ) {
    outfile->Printf("\n  Propogating with linear Taylor algorithm");
    
    Timer t1;
    // The screening criterion
    double eta = options_.get_double("TDACI_ETA");        
    double d_tau = options_.get_double("TDACI_TIMESTEP")*0.0413413745758;        
    double tau = 0.0;
    int nstep = options_.get_int("TDACI_NSTEP");
    
    size_t ndet = C0->dim();
    //The imaginary part
    SharedVector C0_r = std::make_shared<Vector>("C0r", ndet);
    SharedVector C0_i = std::make_shared<Vector>("C0i", ndet);
    SharedVector Ct_r = std::make_shared<Vector>("Ctr", ndet);
    SharedVector Ct_i = std::make_shared<Vector>("Cti", ndet);

    C0_r->copy(C0->clone());
    C0_i->zero();
    
    auto active_sym = mo_space_info_->symmetry("ACTIVE");
   // WFNOperator op(active_sym, fci_ints);
   // op.build_strings(ann_dets);
   // op.op_s_lists(ann_dets);
   // op.tp_s_lists(ann_dets);
   // std::vector<std::pair<std::vector<size_t>, std::vector<double>>> H_sparse = op.build_H_sparse(ann_dets);
    
    int print_val = 9;
    int print_interval = 10;

    for( int N = 0; N < nstep; ++N ){
        std::vector<size_t> counter(ndet, 0);
        tau += d_tau;
//        outfile->Printf("\n  Propogating at t = %1.6f", tau/0.0413413745758);        

        SharedVector sigma_r = std::make_shared<Vector>("Sr", ndet);        
        SharedVector sigma_i = std::make_shared<Vector>("Si", ndet);        
    
        Ct_r->zero();
        Ct_i->zero();
        sigma_r->zero();
        sigma_i->zero();

        // Compute first order correction
        sigma_r->gemv(false, 1.0, &(*H), &(*C0_r), 0.0);
        sigma_i->gemv(false, 1.0, &(*H), &(*C0_i), 0.0);

        sigma_r->scale(d_tau);
        sigma_i->scale(d_tau);

        Ct_r->add(C0_r);
        Ct_i->add(C0_i);

        Ct_r->add(sigma_i);
        Ct_i->subtract(sigma_r);

         // Adaptively screen the wavefunction
        // if( eta > 0.0 ){
        //     std::vector<std::pair<double, size_t>> sumsq(ndet);
        //     double norm = 0.0;
        //     for( int I = 0; I < ndet; ++I ){
        //         double re = C0_r->get(I);
        //         double im = C0_i->get(I);
        //         sumsq[I] = std::make_pair(re*re + im*im, I);
        //         norm += std::sqrt( re*re + im*im );
        //     } 
        //     norm = 1.0/ std::sqrt(norm);
        //     std::sort(sumsq.begin(), sumsq.end()); 
        //     double sum = 0.0;
        //     for( int I = 0; I < ndet; ++I ){
        //         double& val = sumsq[I].first;
        //         size_t idx = sumsq[I].second;
        //         if( sum + val <= eta*norm ){
        //             sum += val;
        //             C0_r->set(idx,0.0);
        //             C0_i->set(idx,0.0);
        //         }else{
        //             counter[idx]++;
        //         }
        //     }
        // }
        // Renormalize C_tau
        double norm = 0.0;
        for( int I = 0; I < ndet; ++I ){
            double re = Ct_r->get(I);
            double im = Ct_i->get(I);
            norm += re*re + im*im;
        } 
        norm = 1.0/ std::sqrt(norm);
        Ct_r->scale(norm);        
        Ct_i->scale(norm);        

        // print the wavefunction
        if( options_.get_bool("TDACI_PRINT_WFN")){
            if( std::abs( (tau/0.0413413745758) - round(tau/0.0413413745758)) <= 1e-8){ 

                outfile->Printf("\n Saving wavefunction for t = %1.3f as", tau/0.0413413745758);
                std::stringstream ss;
                ss << std::fixed << std::setprecision(3) << tau/0.0413413745758;
                save_vector(Ct_r,"taylor_" + ss.str()+ "_r.txt");
                save_vector(Ct_i,"taylor_" + ss.str()+ "_i.txt");
               
            }
        }
        C0_r->copy(Ct_r->clone());
        C0_i->copy(Ct_i->clone());
        print_val += print_interval;
    }
    outfile->Printf("\n  Time spent propogating (linear): %1.6f", t1.get());
}

void TDACI::propogate_taylor2(SharedVector C0, SharedMatrix H  ) {
    

    outfile->Printf("\n  Propogating with quadratic Taylor algorithm");
    Timer t2;
    // The screening criterion
    double eta = options_.get_double("TDACI_ETA");        
    double d_tau = options_.get_double("TDACI_TIMESTEP")*0.0413413745758;        
    double tau = 0.0;
    int nstep = options_.get_int("TDACI_NSTEP");
    
    size_t ndet = C0->dim();
    //The imaginary part
    SharedVector C0_r = std::make_shared<Vector>("C0r", ndet);
    SharedVector C0_i = std::make_shared<Vector>("C0i", ndet);
    SharedVector Ct_r = std::make_shared<Vector>("Ctr", ndet);
    SharedVector Ct_i = std::make_shared<Vector>("Cti", ndet);

    C0_r->copy(C0->clone());
    C0_i->zero();
    
    auto active_sym = mo_space_info_->symmetry("ACTIVE");
   // WFNOperator op(active_sym, fci_ints);
   // op.build_strings(ann_dets);
   // op.op_s_lists(ann_dets);
   // op.tp_s_lists(ann_dets);
   // std::vector<std::pair<std::vector<size_t>, std::vector<double>>> H_sparse = op.build_H_sparse(ann_dets);
    

    for( int N = 0; N < nstep; ++N ){
        std::vector<size_t> counter(ndet, 0);
        tau += d_tau;

//        outfile->Printf("\n  Propogating at t = %1.6f", tau/0.0413413745758);        
        SharedVector sigma_r = std::make_shared<Vector>("Sr", ndet);        
        SharedVector sigma_i = std::make_shared<Vector>("Si", ndet);        
    
        Ct_r->zero();
        Ct_i->zero();
        sigma_r->zero();
        sigma_i->zero();

        // Compute first order correction
        sigma_r->gemv(false, 1.0, &(*H), &(*C0_r), 0.0);
        sigma_i->gemv(false, 1.0, &(*H), &(*C0_i), 0.0);

        sigma_r->scale(d_tau);
        sigma_i->scale(d_tau);

        Ct_r->add(C0_r);
        Ct_i->add(C0_i);

        Ct_r->add(sigma_i);
        Ct_i->subtract(sigma_r);
        // Quadratic correction
        SharedVector sigmaq_r = std::make_shared<Vector>("Sr", ndet);        
        SharedVector sigmaq_i = std::make_shared<Vector>("Si", ndet);        
        sigma_r->scale(1.0/d_tau);
        sigma_i->scale(1.0/d_tau);
        sigmaq_r->gemv(false, 1.0, &(*H), &(*sigma_r), 0.0);
        sigmaq_i->gemv(false, 1.0, &(*H), &(*sigma_i), 0.0);

        sigmaq_r->scale( d_tau * d_tau * 0.5);
        sigmaq_i->scale( d_tau * d_tau * 0.5);

        Ct_r->subtract(sigmaq_r);
        Ct_i->subtract(sigmaq_i);

        //for( int I = 0; I < ndet; ++I ){
        //    //auto& row_indices = H_sparse[I].first; 
        //    //auto& row_values  = H_sparse[I].second;
        //    //size_t row_dim = row_values.size();
        //    double re = 0.0;
        //    double im = 0.0;
        //    for( int J = 0; J < ndet; ++J ){
        //        //re += row_values[J]* sigma_r[J]; 
        //        //im += row_values[J]* sigma_i[J]; 
        //        re += full_aH->get(I,J)* sigma_r[J]; 
        //        im += full_aH->get(I,J)* sigma_i[J]; 
        //    }
        //    re *= d_tau * d_tau * 0.25;
        //    im *= d_tau * d_tau * 0.25;
        //    
        //    C_tau[I].first -= re;
        //    C_tau[I].second -= im;
        //    
        //}


        // Renormalize C_tau
        double norm = 0.0;
        for( int I = 0; I < ndet; ++I ){
            double re = Ct_r->get(I);
            double im = Ct_i->get(I);
            norm += re*re + im*im;
        } 
        norm = 1.0/ std::sqrt(norm);
        Ct_r->scale(norm);        
        Ct_i->scale(norm);        


        // print the wavefunction
        if( options_.get_bool("TDACI_PRINT_WFN")){
            if( std::abs( (tau/0.0413413745758) - round(tau/0.0413413745758)) <= 1e-8){ 
                outfile->Printf("\n Saving wavefunction for t = %1.3f as", tau/0.0413413745758);
                std::stringstream ss;
                ss << std::fixed << std::setprecision(3) << tau/0.0413413745758;
                save_vector(Ct_r,"taylor2_" + ss.str()+ "_r.txt");
                save_vector(Ct_i,"taylor2_" + ss.str()+ "_i.txt");
            }
        }
        C0_r->copy(Ct_r->clone());
        C0_i->copy(Ct_i->clone());
    } 
    outfile->Printf("\n  Time spent propogating (quadratic): %1.6f", t2.get());
}

void TDACI::propogate_RK4(SharedVector C0, SharedMatrix H  ) {

    outfile->Printf("\n  Propogating with 4th order Runge-Kutta algorithm");

    Timer total;
    size_t ndet = C0->dim();

    int nstep = options_.get_int("TDACI_NSTEP");
    double dt = options_.get_double("TDACI_TIMESTEP");
    double conv = 1.0/24.18884326505;
    dt *= conv;
    double time = dt;


    // Copy initial state into iteratively updated vectors
    SharedVector ct_r = std::make_shared<Vector>("ct_R",ndet);
    SharedVector ct_i = std::make_shared<Vector>("ct_I",ndet);

    ct_r->copy(C0->clone());
    ct_i->zero();

    for( int n = 1; n <= nstep; ++n ){

        // k1
        SharedVector k1r = std::make_shared<Vector>("k1r", ndet);
        SharedVector k1i = std::make_shared<Vector>("k1i", ndet);

        k1r->gemv(false, 1.0, &(*H), &(*ct_i), 0.0);
        k1i->gemv(false, -1.0, &(*H), &(*ct_r), 0.0);
        
        // k2
        SharedVector intr = std::make_shared<Vector>("intr", ndet);
        SharedVector inti = std::make_shared<Vector>("inti", ndet);

        intr->copy(ct_r->clone());
        inti->copy(ct_i->clone());

        k1r->scale(0.5*dt);
        k1i->scale(0.5*dt);
        //k1r->scale(0.5);
        //k1i->scale(0.5);
        intr->add(k1r);
        inti->add(k1i);
        k1r->scale(2.0/dt);
        k1i->scale(2.0/dt);
        //k1r->scale(2.0);
        //k1i->scale(2.0);

        SharedVector k2r = std::make_shared<Vector>("k2r", ndet);
        SharedVector k2i = std::make_shared<Vector>("k2i", ndet);

        k2r->gemv(false, 1.0, &(*H), &(*inti), 0.0);
        k2i->gemv(false, -1.0, &(*H), &(*intr), 0.0);
        
        // k3
        intr->copy(ct_r->clone());
        inti->copy(ct_i->clone());

        k2r->scale(0.5*dt);
        k2i->scale(0.5*dt);
        //k2r->scale(0.5);
        //k2i->scale(0.5);
        intr->add(k2r); 
        inti->add(k2i);
        k2r->scale(2.0 * 1.0/dt);
        k2i->scale(2.0 * 1.0/dt);
        //k2r->scale(2.0);
        //k2i->scale(2.0);

        SharedVector k3r = std::make_shared<Vector>("k3r", ndet);
        SharedVector k3i = std::make_shared<Vector>("k3i", ndet);

        k3r->gemv(false, 1.0, &(*H), &(*inti), 0.0);
        k3i->gemv(false, -1.0, &(*H), &(*intr), 0.0);

        // k4
        intr->copy(ct_r->clone());
        inti->copy(ct_i->clone());
        
        k3r->scale(dt);
        k3i->scale(dt);
        intr->add(k3r); 
        inti->add(k3i);
        k3r->scale(1.0/dt);
        k3i->scale(1.0/dt);

        SharedVector k4r = std::make_shared<Vector>("k4r", ndet);
        SharedVector k4i = std::make_shared<Vector>("k4i", ndet);

        k4r->gemv(false, 1.0, &(*H), &(*inti), 0.0);
        k4i->gemv(false, -1.0, &(*H), &(*intr), 0.0);
    
        // Compile all intermediates

        k1r->scale(dt/6.0);
        k2r->scale(dt/3.0);
        k3r->scale(dt/3.0);
        k4r->scale(dt/6.0);
        k1r->add(k2r);
        k1r->add(k3r);
        k1r->add(k4r);
        ct_r->add(k1r);
        
        k1i->scale(dt/6.0);
        k2i->scale(dt/3.0);
        k3i->scale(dt/3.0);
        k4i->scale(dt/6.0);
        k1i->add(k2i);
        k1i->add(k3i);
        k1i->add(k4i);
        ct_i->add(k1i);

        std::stringstream ss;
        ss << std::fixed << std::setprecision(3) << time/conv;
        double norm = 0.0;
        for( int I = 0; I < ndet; ++I ){
            double re = ct_r->get(I);
            double im = ct_i->get(I);
            norm += (re*re) + (im*im);
        }        
        norm = std::sqrt(norm);
        ct_r->scale(1.0/norm);
        ct_i->scale(1.0/norm);

        if( options_.get_bool("TDACI_PRINT_WFN")){
            if( std::fabs( (time/conv) - round(time/conv)) <= 1e-8 ){
                outfile->Printf("\n Saving wavefunction for t = %1.3f as", time/conv);
                save_vector(ct_r,"RK4_" + ss.str()+ "_r.txt");
                save_vector(ct_i,"RK4_" + ss.str()+ "_i.txt");
            }
        }

        time += dt;
    }
    outfile->Printf("\n  Time spent propogating (RK4): %1.6f", total.get());

}

void TDACI::propogate_QCN(SharedVector C0, SharedMatrix H  ) {

    Timer total;
    size_t ndet = C0->dim();

    int nstep = options_.get_int("TDACI_NSTEP");
    double dt = options_.get_double("TDACI_TIMESTEP");
    double conv = 1.0/24.18884326505;
    dt *= conv;
    double time = dt;

    // Copy initial state into iteratively updated vectors
    SharedVector ct_r = std::make_shared<Vector>("ct_R",ndet);
    SharedVector ct_i = std::make_shared<Vector>("ct_I",ndet);

    ct_r->copy(C0->clone());
    ct_i->zero();

    for( int n = 1; n <= nstep; ++n ){

        SharedVector b_r = std::make_shared<Vector>("br",ndet);
        SharedVector b_i = std::make_shared<Vector>("bi",ndet);

        // Quadratic propogator for b

        b_r->copy(ct_r->clone());
        b_i->copy(ct_i->clone());

        SharedVector sigma_r = std::make_shared<Vector>("sr",ndet);
        SharedVector sigma_i = std::make_shared<Vector>("si",ndet);
        sigma_r->gemv(false, 0.5*dt, &(*H), &(*ct_r), 0.0);
        sigma_i->gemv(false, 0.5*dt, &(*H), &(*ct_i), 0.0);

        b_r->add(sigma_i);
        b_i->subtract(sigma_r);
        // Quadratic correction
        b_r->gemv(false, -0.5*dt, &(*H), &(*sigma_r), 1.0);
        b_i->gemv(false, -0.5*dt, &(*H), &(*sigma_i), 1.0);
    
        bool converged = false;
        SharedVector ct_r_new = std::make_shared<Vector>("ct_R",ndet);
        SharedVector ct_i_new = std::make_shared<Vector>("ct_I",ndet);


        while( !converged ){
            ct_r_new->copy(b_r->clone());
            ct_i_new->copy(b_i->clone());
             
            SharedVector tmp_r = std::make_shared<Vector>("t_r",ndet);
            SharedVector tmp_i = std::make_shared<Vector>("t_i",ndet);
            
            tmp_r->gemv(false, 0.5*dt, &(*H), &(*ct_r), 0.0);
            tmp_i->gemv(false, 0.5*dt, &(*H), &(*ct_i), 0.0);

            ct_r_new->add(tmp_i);
            ct_i_new->subtract(tmp_r);

            ct_r_new->gemv(false, 0.5*dt, &(*H), &(*tmp_r), 1.0);
            ct_i_new->gemv(false, 0.5*dt, &(*H), &(*tmp_i), 1.0);
            
            // Test convergence
            SharedVector err = std::make_shared<Vector>("err",ndet);
            double norm = 0.0;  
            for( size_t I = 0; I < ndet; ++I ){
                double rn = ct_r_new->get(I);
                double in = ct_i_new->get(I);
                norm += (rn*rn + in*in);
            }
            ct_r_new->scale( 1.0/sqrt(norm));
            ct_i_new->scale( 1.0/sqrt(norm));
            for( size_t I = 0; I < ndet; ++I ){
                double rn = ct_r_new->get(I);
                double in = ct_i_new->get(I);
                double ro = ct_r->get(I);
                double io = ct_i->get(I);
                err->set(I, (rn*rn + in*in) - (ro*ro + io*io));                
            }

 //           outfile->Printf("\n  %1.9f", err->norm());
            if( err->norm() <= options_.get_double("TDACI_CN_CONVERGENCE") ){
                converged = true;
            }
            
            ct_r->copy( ct_r_new->clone() );
            ct_i->copy( ct_i_new->clone() );
        }
        if( options_.get_bool("TDACI_PRINT_WFN")){
            if( std::fabs( (time/conv) - round(time/conv)) <= 1e-8 ){
                std::stringstream ss;
                ss << std::fixed << std::setprecision(3) << time/conv;
                outfile->Printf("\n Saving wavefunction for t = %1.3f as", time/conv);
                save_vector(ct_r,"QCN_" + ss.str()+ "_r.txt");
                save_vector(ct_i,"QCN_" + ss.str()+ "_i.txt");
            }
        }

        time += dt;
    }
} 

void TDACI::propogate_lanczos(SharedVector C0, SharedMatrix H  ) {



    outfile->Printf("\n  Propogating with Arnoldi-Lanzcos algorithm");
    Timer total;
    size_t ndet = C0->dim();

    int nstep = options_.get_int("TDACI_NSTEP");
    double dt = options_.get_double("TDACI_TIMESTEP");
    double conv = 1.0/24.18884326505;
    dt *= conv;
    double time = dt;

    // Copy initial state into iteratively updated vectors
    SharedVector ct_r = std::make_shared<Vector>("ct_R",ndet);
    SharedVector ct_i = std::make_shared<Vector>("ct_I",ndet);


    ct_r->copy(C0->clone());
    ct_i->zero();

    int krylov_dim = options_.get_int("TDACI_KRYLOV_DIM");

    for( int N = 0; N < nstep; ++N ){
    
        // 1. Form the Krylov subspace vectors and subspace hamiltonian simultaneously
        std::vector<std::pair<SharedVector,SharedVector>> Kn(krylov_dim);
        //SharedMatrix Hs = std::make_shared<Matrix>("Hs", krylov_order,krylov_order);
        //Hs->zero();
        fcomplex *Hs = new fcomplex[krylov_dim*krylov_dim];
        //std::vector<double> Hs(4*krylov_dim*krylov_dim, 0.0); 
        // 1a. Generate non-orthonormalized basis
        Kn[0] = std::make_pair(ct_r, ct_i);
        for( int k = 0 ; k < krylov_dim; ++k ){
        
            // Need to get last diagonal


            SharedVector wk_r = std::make_shared<Vector>("r",ndet); 
            SharedVector wk_i = std::make_shared<Vector>("i",ndet); 
            wk_r->zero();
            wk_i->zero();
    
            SharedVector qk_r = std::make_shared<Vector>("r",ndet); 
            SharedVector qk_i = std::make_shared<Vector>("i",ndet); 
            qk_r->zero();
            qk_i->zero();

            qk_r->add(Kn[k].first);
            qk_i->add(Kn[k].second);

            wk_r->gemv(false, 1.0, &(*H), &(*qk_r), 0.0);
            wk_i->gemv(false, 1.0, &(*H), &(*qk_i), 0.0);
            // Modified Gram-Schmidt
            for( int i = 0; i <= k; ++i ){
                SharedVector qi_r = std::make_shared<Vector>("r",ndet); 
                SharedVector qi_i = std::make_shared<Vector>("i",ndet); 
                qi_r->zero();
                qi_i->zero();
                qi_r->add(Kn[i].first);
                qi_i->add(Kn[i].second);

                float hik_r = wk_r->vector_dot(qi_r) + wk_i->dot(&(*qi_i)); 
                float hik_i = wk_i->vector_dot(qi_r) - wk_r->dot(&(*qi_i)); 
            
            //    outfile->Printf("\n r: %1.5f, i:%1.5f", hik_r, hik_i); 
                Hs[krylov_dim*i + k] = {0.0,0.0 };
                Hs[krylov_dim*k + i] = {hik_r, hik_i};
               // Hs[2*krylov_dim*k + 2*i] = hik_r;
               // Hs[2*krylov_dim*k + 2*i + 1] = hik_i;
               // Hs[2*krylov_dim*i + 2*k] = hik_r;
               // Hs[2*krylov_dim*i + 2*k + 1] = hik_i;

                qi_r->scale(hik_r);
                qi_i->scale(hik_i);
                wk_r->subtract( qi_r );
                wk_i->subtract( qi_i );
            }
                
            float norm = 0.0;
            for( int I = 0; I < ndet; ++I ){
                double re = wk_r->get(I);
                double im = wk_i->get(I);
                norm += re*re + im*im; 
            }
            norm = sqrt(norm);
            //outfile->Printf("\n  norm(%d) = %1.5f", k, norm);
            wk_r->scale(1.0/norm);
            wk_i->scale(1.0/norm);

            if( k < (krylov_dim-1)){
                Hs[krylov_dim*(k+1) + k] = {norm,0.0};
                //Hs[2*krylov_dim*(k+1) + 2*k] = norm;
                //Hs[2*krylov_dim*(k+1) + 2*k + 1] = 0.0;
               // Hs[2*krylov_dim*k + 2*(k+1)] = norm;
               // Hs[2*krylov_dim*k + 2*(k+1) + 1] = 0.0;
                Kn[k+1] = std::make_pair(wk_r,wk_i);
            }
        }

        //test, print Hs
        outfile->Printf("\n");
        for( int i = 0; i < krylov_dim; ++i ){
            for( int j = 0; j < krylov_dim; ++j ){
                auto vec1 = Kn[i].first;
                auto vec2 = Kn[j].first;
                outfile->Printf("%5.3f\t", vec1->dot(&(*vec2)));
            }
            outfile->Printf("\n");
        }
        
        for( int i = 0; i < krylov_dim; ++i){
            outfile->Printf("\n");
            for( int j = 0; j < krylov_dim; ++j){
                outfile->Printf("%5.2f+%5.2fi\t", Hs[krylov_dim*i + j].re, Hs[krylov_dim*i + j].im);
            }
        }
        // Diagonalize matrix in Krylov subspace
        int n = krylov_dim, lda = krylov_dim, info, lwork;
        fcomplex wkopt;
        fcomplex* work;
        /* Local arrays */
        /* rwork dimension should be at least max(1,3*n-2) */
        float w[n], rwork[3*n-2];
        lwork = 2*n-1;
        cheev( "V", "L", &n, Hs, &lda, w, work, &lwork, rwork, &info );

        delete[] Hs;

    }    

}

//void TDACI::propogate_verlet(std::vector<std::pair<double,double>>& C0, std::vector<std::pair<double,double>>& C_tau, std::shared_ptr<FCIIntegrals> fci_ints, DeterminantHashVec& ann_dets  ) {
//    
//    // The screening criterion
//    double eta = options_.get_double("TDACI_ETA");        
//    double d_tau = options_.get_double("TDACI_TIMESTEP")*0.0413413745758;        
//    double tau = 0.0;
//    int nstep = options_.get_int("TDACI_NSTEP");
//    
//    size_t ndet = ann_dets.size();
//    C_tau.resize(ndet);
//    // Save initial wavefunction
//    
//    outfile->Printf("\n Saving wavefunction for t = 0.0 as");
//    std::vector<double> zr(ndet);
//    std::vector<double> zi(ndet);
//    for( int I = 0; I < ndet; ++I ){
//        zr[I] = C0[I].first;
//        zi[I] = C0[I].second;
//    } 
//    save_vector(zr,"tau_0.0_r.txt");
//    save_vector(zi,"tau_0.0_i.txt");
//    
//    //read hamiltonian from disk
//    outfile->Printf("\n  Loading Hamiltonian");
//    SharedMatrix full_aH = std::make_shared<Matrix>("aH",ndet,ndet);
//
//    std::ifstream file("hamiltonian.txt", std::ios::in);
//    if( !file ){
//        outfile->Printf("\n  Could not open file");
//        outfile->Printf("\n  Building Hamiltonian from scratch");
//        for( size_t I = 0; I < ndet; ++I ){
//            Determinant detI = ann_dets.get_det(I);
//            for( size_t J = I; J < ndet; ++J ){
//                Determinant detJ = ann_dets.get_det(J);
//                double value = fci_ints->slater_rules(detI,detJ);
//                full_aH->set(I,J, value);
//                full_aH->set(J,I, value);
//            }
//        }
//    } else {
//        for( size_t I = 0; I < ndet; ++I ){
//            for( size_t J = 0; J < ndet; ++J ){
//                double num = 0.0;
//                file >> num;
//                full_aH->set(I,J,num); 
//            }
//        }
//    }
//    outfile->Printf("  ...done");
//
//
//}

void TDACI::save_matrix( SharedMatrix mat, std::string name) {
    
    size_t dim = mat->nrow();
    std::ofstream file;
    file.open(name, std::ofstream::out | std::ofstream::trunc);
    for( size_t I = 0; I < dim; ++I ){
        for( size_t J = 0; J < dim; ++J ){
            file << std::setw(12) << std::setprecision(11) << mat->get(I,J) << " " ;
        }
        file << "\n";
    }
}
void TDACI::save_vector( SharedVector vec, std::string name) {
    
    size_t dim = vec->dim();
    std::ofstream file;
    file.open(name, std::ofstream::out | std::ofstream::trunc);
    for( size_t I = 0; I < dim; ++I ){
        file << std::setw(12) << std::setprecision(11) << vec->get(I) << "\n" ;
    }
}
void TDACI::save_vector( std::vector<std::string>& vec, std::string name) {
    
    size_t dim = vec.size();
    std::ofstream file;
    file.open(name, std::ofstream::out | std::ofstream::trunc);
    for( size_t I = 0; I < dim; ++I ){
        file << vec[I] << "\n" ;
    }
}

void TDACI::save_vector( std::vector<size_t>& vec, std::string name) {
    
    size_t dim = vec.size();
    std::ofstream file;
    file.open(name, std::ofstream::out | std::ofstream::trunc);
    for( size_t I = 0; I < dim; ++I ){
        file << vec[I] << "\n" ;
    }
}
void TDACI::save_vector( std::vector<double>& vec, std::string name) {
    
    size_t dim = vec.size();
    std::ofstream file;
    file.open(name, std::ofstream::out | std::ofstream::trunc);
    for( size_t I = 0; I < dim; ++I ){
        file << std::setw(12) << std::setprecision(11) << vec[I] << "\n" ;
    }
}
void TDACI::annihilate_wfn( DeterminantHashVec& olddets,
                            DeterminantHashVec& anndets, int frz_orb ) {


    int ncmo = mo_space_info_->size("ACTIVE");
    // Loop through determinants, annihilate frz_orb(alpha)
    const det_hashvec& dets = olddets.wfn_hash();
    size_t ndet = dets.size();
    for( size_t I = 0; I < ndet; ++I ){
        auto& detI = dets[I]; 
        if( detI.get_alfa_bit(frz_orb) == true ){
            Determinant new_det(detI);
            new_det.set_alfa_bit(frz_orb, false);
            if( !anndets.has_det(new_det) ){
//                outfile->Printf("\n %s", new_det.str(ncmo ).c_str());
                anndets.add(new_det);
            }
        }
    }
}

std::vector<double> TDACI::compute_occupation( SharedVector Cr, SharedVector Ci, std::vector<int>& orbs ) {

    int nact = Cr->dim();
    std::vector<double> occ_vec(orbs.size(), 0.0); 
    
    for(int i = 0; i < orbs.size(); ++i  ){ 
        double occ = 0.0;
        int orb = orbs[i];
        for( int I = 0; I < nact; ++I) {
            
            const Determinant& detI = ann_dets.get_det(I);
            if( detI.get_alfa_bit(orb) == true ){
                size_t idx = ann_dets.get_idx(detI);
                double re = Cr->get(idx);
                double im = Ci->get(idx);
                occ += re*re + im*im; 
            }
        }      
        occ_vec[i] = occ;
    }

    return occ_vec;
}


}}
