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

#include "psi4/libpsi4util/process.h"
#include "psi4/libmints/molecule.h"
#include "psi4/libmints/pointgrp.h"
#include "psi4/libpsio/psio.hpp"

#include "../helpers/printing.h"
#include "tdaci.h"

using namespace psi;

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
    foptions.add_str("TDACI_PROPOGATOR", "EXACT", "Type of propogator");
    foptions.add_int("TDACI_NSTEP", 20, "Number of steps");
    foptions.add_double("TDACI_TIMESTEP", 1.0, "Timestep (as)");
    foptions.add_double("TDACI_ETA", 1e-12, "Path filtering threshold");
    foptions.add_int("TDACI_TAYLOR_ORDER", 1, "Maximum order of taylor expansion used");
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


    // 1. Grab an ACI wavefunction
    auto aci = std::make_shared<AdaptiveCI>(wfn_, options_, ints_, mo_space_info_);
    aci->set_quiet(true);
    aci->compute_energy();
    DeterminantHashVec aci_dets = aci->get_wavefunction();
    SharedMatrix aci_coeffs = aci->get_evecs();
    outfile->Printf("\n  ACI wavefunction built");
    
    // 2. Now, build the full Hamiltonian in the n-1 space (not just core)
    DeterminantHashVec ann_dets;
    for( int i = 0; i < nact; ++i ){
        annihilate_wfn(aci_dets, ann_dets,i);  
    }
    size_t nann = ann_dets.size();
    outfile->Printf("\n  size of ann dets: %zu", ann_dets.size());

    std::vector<std::string> det_vec(nann);
    // Save occupations to file
    const det_hashvec& annhash = ann_dets.wfn_hash();
    for ( size_t I = 0; I < nann; ++I ){
        auto& detI = annhash[I];
        det_vec[I] = detI.str(nact);
    }
    save_vector(det_vec, "ann_dets.txt");
    

    std::ifstream file("c_init.txt", std::ios::in);
    if( !file ){

        // 3. Remove a core electron from aci wfn, compile coeffss
        DeterminantHashVec core_dets;
        std::vector<double> core_coeffs(nann, 0.0); 

        const det_hashvec& dets = aci_dets.wfn_hash();
        size_t ndet = dets.size();
        size_t ncore = 0;
        for ( size_t I = 0; I < ndet; ++I ){
            auto& detI = dets[I];
            if( detI.get_alfa_bit(0) == true ){
                Determinant adet(detI);
                adet.set_alfa_bit(0, false);
                size_t idx = ann_dets.get_idx(adet);
                core_coeffs[idx] +=  aci_coeffs->get(aci_dets.get_idx(detI),0); 
                core_dets.add(adet);
  //              outfile->Printf("\n %s  %s",detI.str(ncmo).c_str(),  adet.str(ncmo).c_str());
            }  
            if( detI.get_beta_bit(0) == true ){
                Determinant adet(detI);
                adet.set_beta_bit(0, false);
                size_t idx = ann_dets.get_idx(adet);
                core_coeffs[idx] +=  aci_coeffs->get(aci_dets.get_idx(detI),0); 
                core_dets.add(adet);
  //              outfile->Printf("\n %s  %s",detI.str(ncmo).c_str(),  adet.str(ncmo).c_str());
            }  
        }
        outfile->Printf("\n  size of core dets: %zu", core_dets.size());    
        
        // 2. Renormalize wave function
        outfile->Printf("\n  Renormalizing wave function");
        double norm = 0.0;//core_coeffs->norm();
        for( auto& val : core_coeffs ){
            norm += val * val;
        }
        norm = 1.0/ std::sqrt(norm);
        for( auto& val : core_coeffs ){
            val *= norm;
        }

        // 3. Build full n-1 hamiltonian
        if( options_.get_str("TDACI_PROPOGATOR") == "EXACT" ){
            std::shared_ptr<FCIIntegrals> fci_ints = aci->get_aci_ints();
            SharedMatrix full_aH = std::make_shared<Matrix>("aH",nann, nann);
            for( size_t I = 0; I < nann; ++I ){
                Determinant detI = ann_dets.get_det(I);
                for( size_t J = I; J < nann; ++J ){
                    Determinant detJ = ann_dets.get_det(J);
                    double value = fci_ints->slater_rules(detI,detJ);
                    full_aH->set(I,J, value);
                    full_aH->set(J,I, value);
                }
            }
            // Print full Hamiltonian
            outfile->Printf("\n Writing Hamiltonian");
            save_matrix( full_aH, "hamiltonian.txt");
            outfile->Printf("  ...done");

            outfile->Printf("\n Diagonalizing H");
            SharedMatrix full_evecs = std::make_shared<Matrix>("evec", nann,nann);
            SharedVector full_evals = std::make_shared<Vector>("evals", nann);
            full_aH->diagonalize(full_evecs, full_evals);
            outfile->Printf("  ...done");

            outfile->Printf("\n Writing c_init");
            save_vector(core_coeffs,"c_init.txt");
            outfile->Printf("  ...done");

            outfile->Printf("\n Writing full evecs");
            save_matrix(full_evecs, "full_evecs.txt");
            outfile->Printf("  ...done");

            outfile->Printf("\n Writing full evals");
            save_vector(full_evals, "full_evals.txt");
            outfile->Printf("  ...done");
        }
    } else {
        if (options_.get_str("TDACI_PROPOGATOR") == "TAYLOR" ){
            SharedVector C0 = std::make_shared<Vector>("C0r", ann_dets.size());
            std::shared_ptr<FCIIntegrals> fci_ints = aci->get_aci_ints();
            //std::vector<std::pair<double,double>> C_new;
            //std::vector<double> C0(nann);
            for( size_t I = 0; I < nann; ++I ){
                double num = 0.0;
                file >> num;
                C0->set(I,num);
            }
            
            propogate_taylor(C0, fci_ints, ann_dets );

        }
//        if (options_.get_str("TDACI_PROPOGATOR") == "VERLET" ){
//            propogate_verlet(C0, C_new, fci_ints, ann_dets);
//        }
    
    }
    return en;
}

void TDACI::propogate_taylor(SharedVector C0_r, std::shared_ptr<FCIIntegrals> fci_ints, DeterminantHashVec& ann_dets  ) {
    

    // The screening criterion
    double eta = options_.get_double("TDACI_ETA");        
    double d_tau = options_.get_double("TDACI_TIMESTEP")*0.0413413745758;        
    double tau = 0.0;
    int nstep = options_.get_int("TDACI_NSTEP");
    
    size_t ndet = ann_dets.size();
    //The imaginary part
    SharedVector C0_i = std::make_shared<Vector>("C0i", ndet);
    SharedVector Ct_r = std::make_shared<Vector>("Ctr", ndet);
    SharedVector Ct_i = std::make_shared<Vector>("Cti", ndet);
    C0_i->zero();
    
    outfile->Printf("\n Saving wavefunction for t = 0.0 as");
    save_vector(C0_r,"tau_0.00_r.txt");
    save_vector(C0_i,"tau_0.00_i.txt");
    auto active_sym = mo_space_info_->symmetry("ACTIVE");
   // WFNOperator op(active_sym, fci_ints);
   // op.build_strings(ann_dets);
   // op.op_s_lists(ann_dets);
   // op.tp_s_lists(ann_dets);
   // std::vector<std::pair<std::vector<size_t>, std::vector<double>>> H_sparse = op.build_H_sparse(ann_dets);
    
    //read hamiltonian from disk
    outfile->Printf("\n  Loading Hamiltonian");
    SharedMatrix full_aH = std::make_shared<Matrix>("aH",ndet,ndet);

    std::ifstream file("hamiltonian.txt", std::ios::in);
    if( !file ){
        outfile->Printf("\n  Could not open file");
        outfile->Printf("\n  Building Hamiltonian from scratch");
        for( size_t I = 0; I < ndet; ++I ){
            Determinant detI = ann_dets.get_det(I);
            for( size_t J = I; J < ndet; ++J ){
                Determinant detJ = ann_dets.get_det(J);
                double value = fci_ints->slater_rules(detI,detJ);
                full_aH->set(I,J, value);
                full_aH->set(J,I, value);
            }
        }
    } else {
        for( size_t I = 0; I < ndet; ++I ){
            for( size_t J = 0; J < ndet; ++J ){
                double num = 0.0;
                file >> num;
                full_aH->set(I,J,num); 
            }
        }
    }
    outfile->Printf("  ...done");

    int print_val = 9;
    int print_interval = 10;
    int order = options_.get_int("TDACI_TAYLOR_ORDER");

    outfile->Printf("\n  Propogating with tau = %1.2f", d_tau);
    outfile->Printf("\n Truncation Taylor expansion at order %d", order);
    for( int N = 0; N < nstep; ++N ){
        std::vector<size_t> counter(ndet, 0);
        tau += d_tau;

        SharedVector sigma_r = std::make_shared<Vector>("Sr", ndet);        
        SharedVector sigma_i = std::make_shared<Vector>("Si", ndet);        
    
        Ct_r->zero();
        Ct_i->zero();
        sigma_r->zero();
        sigma_i->zero();

        // Compute first order correction
        if( order >= 1 ){

            sigma_r->gemv(false, 1.0, &(*full_aH), &(*C0_r), 0.0);
            sigma_i->gemv(false, 1.0, &(*full_aH), &(*C0_i), 0.0);

            sigma_r->scale(d_tau);
            sigma_i->scale(d_tau);

            Ct_r->add(C0_r);
            Ct_i->add(C0_i);

            Ct_r->add(sigma_i);
            Ct_i->subtract(sigma_r);

            // Adaptively screen the wavefunction
            if( eta > 0.0 ){
                std::vector<std::pair<double, size_t>> sumsq(ndet);
                double norm = 0.0;
                for( int I = 0; I < ndet; ++I ){
                    double re = C0_r->get(I);
                    double im = C0_i->get(I);
                    sumsq[I] = std::make_pair(re*re + im*im, I);
                    norm += std::sqrt( re*re + im*im );
                } 
                norm = 1.0/ std::sqrt(norm);
                std::sort(sumsq.begin(), sumsq.end()); 
                double sum = 0.0;
                for( int I = 0; I < ndet; ++I ){
                    double& val = sumsq[I].first;
                    size_t idx = sumsq[I].second;
                    if( sum + val <= eta*norm ){
                        sum += val;
                        C0_r->set(idx,0.0);
                        C0_i->set(idx,0.0);
                    }else{
                        counter[idx]++;
                    }
                }
            }
        }
        // Quadratic correction
        if( order >= 2 ){
            
            sigma_r->gemv(false, 1.0, &(*full_aH), &(*sigma_r), 0.0);
            sigma_i->gemv(false, 1.0, &(*full_aH), &(*sigma_i), 0.0);

            sigma_r->scale( d_tau * d_tau * 0.25);
            sigma_i->scale( d_tau * d_tau * 0.25);

            Ct_r->subtract(sigma_r);
            Ct_i->subtract(sigma_i);

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

        }

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

        C0_r->copy(Ct_r->clone());
        C0_i->copy(Ct_i->clone());

        // print the wavefunction
        if( N == print_val ){ 
            outfile->Printf("\n Saving wavefunction for t = %1.3f as", tau/0.0413413745758);
           // std::vector<double> real(ndet);
           // std::vector<double> imag(ndet);
           // for( int I = 0; I < ndet; ++I ){
           //     real[I] = C_tau[I].first;
           //     imag[I] = C_tau[I].second;
           //     
           // } 
            //save_vector(sumsq,"tau_"+ std::to_string(tau) + ".txt");
            std::stringstream ss;
            ss << std::fixed << std::setprecision(2) << tau/0.0413413745758;
            save_vector(Ct_r,"tau_" + ss.str()+ "_r.txt");
            save_vector(Ct_i,"tau_" + ss.str()+ "_i.txt");
            save_vector( counter, "det_"+ss.str()+"_counter.txt");
            print_val += print_interval;
        }
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
        if( detI.get_beta_bit(frz_orb) == true ){
            Determinant new_det(detI);
            new_det.set_beta_bit(frz_orb, false);
            if( !anndets.has_det(new_det) ){
//                outfile->Printf("\n %s", new_det.str(ncmo ).c_str());
                anndets.add(new_det);
            }
        }   
    }
}

//void TDACI::renormalize_wfn( SharedVector vec ) {
//    
//    // compute the norm
//    double norm = 0.0;
//    for( auto& I : vec ){
//        norm += I*I;
//    }
//    outfile->Printf("\n  norm = %1.5f", norm);
//    norm = 1.0 / std::sqrt(norm);
//    for( auto& I : vec ){
//        I *= norm;
//    } 
//        
//    //check
//    //double nnorm = 0.0;
//    //for( auto& I : vec ){
//    //    nnorm += I*I;
//    //}
//    //outfile->Printf("\n new  norm = %1.5f", nnorm);
//
//} 

}}
