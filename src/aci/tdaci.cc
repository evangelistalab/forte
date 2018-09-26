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
    int ncmo = mo_space_info_->size("ACTIVE");

    // 1. Grab an ACI wavefunction
    auto aci = std::make_shared<AdaptiveCI>(wfn_, options_, ints_, mo_space_info_);
    aci->set_quiet(true);
    aci->compute_energy();
    DeterminantHashVec aci_dets = aci->get_wavefunction();
    SharedMatrix aci_coeffs = aci->get_evecs();
    outfile->Printf("\n  ACI wavefunction built");
    
    // 2. Now, build the full Hamiltonian in the n-1 space (not just core)
    DeterminantHashVec ann_dets;
    for( int i = 0; i < ncmo; ++i ){
        annihilate_wfn(aci_dets, ann_dets,i);  
    }

    size_t nann = ann_dets.size();
    outfile->Printf("\n  size of ann dets: %zu", ann_dets.size());

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
//            outfile->Printf("\n %s  %s",detI.str(ncmo).c_str(),  adet.str(ncmo).c_str());
        }  
        if( detI.get_beta_bit(0) == true ){
            Determinant adet(detI);
            adet.set_beta_bit(0, false);
            size_t idx = ann_dets.get_idx(adet);
            core_coeffs[idx] +=  aci_coeffs->get(aci_dets.get_idx(detI),0); 
            core_dets.add(adet);
//            outfile->Printf("\n %s  %s",detI.str(ncmo).c_str(),  adet.str(ncmo).c_str());
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
        save_matrix( full_aH, "hamiltonian.txt");
        SharedMatrix full_evecs = std::make_shared<Matrix>("evec", nann,nann);
        SharedVector full_evals = std::make_shared<Vector>("evals", nann);
        full_aH->diagonalize(full_evecs, full_evals);
        save_vector(core_coeffs,"c_init.txt");
        save_matrix(full_evecs, "full_evecs.txt");
        save_vector(full_evals, "full_evals.txt");
    } else if (options_.get_str("TDACI_PROPOGATOR") == "TAYLOR" ){
        std::shared_ptr<FCIIntegrals> fci_ints = aci->get_aci_ints();
        std::vector<std::pair<double,double>> C_new;
        std::vector<std::pair<double,double>> C0(nann);
        for( size_t I = 0; I < nann; ++I ){
            C0[I] = std::make_pair(core_coeffs[I], 0.0 );
        }
        
        propogate_taylor(C0, C_new, fci_ints, ann_dets );

    }

    return en;
}

void TDACI::propogate_taylor(std::vector<std::pair<double,double>>& C0, std::vector<std::pair<double,double>>& C_tau, std::shared_ptr<FCIIntegrals> fci_ints, DeterminantHashVec& ann_dets  ) {
    

    // The screening criterion
    double eta = options_.get_double("TDACI_ETA");        
    double d_tau = options_.get_double("TDACI_TIMESTEP")*0.0413413745758;        
    double tau = 0.0;

    int nstep = options_.get_int("TDACI_NSTEP");
    
    // 1. Copy initial wfn into new one
    size_t ndet = ann_dets.size();
    C_tau.resize(ndet);
 //   for( int I = 0; I < ndet; ++I ){ 
 //       C_tau[I] = C0[I];
 //   }


    // Save initial wavefunction
    
    outfile->Printf("\n Saving wavefunction for t = 0.0 as");
    std::vector<double> sum_sq(ndet);
    for( int I = 0; I < ndet; ++I ){
        double re = C0[I].first;
        double im = C0[I].second;
        sum_sq[I] = re*re + im*im;
    } 
    //save_vector(sumsq,"tau_"+ std::to_string(tau) + ".txt");
    save_vector(sum_sq,"tau_0.0.txt");

    auto active_sym = mo_space_info_->symmetry("ACTIVE");
    WFNOperator op(active_sym, fci_ints);
    op.build_strings(ann_dets);
    op.op_s_lists(ann_dets);
    op.tp_s_lists(ann_dets);
    std::vector<std::pair<std::vector<size_t>, std::vector<double>>> H_sparse = op.build_H_sparse(ann_dets);

    int print_val = 1;

    outfile->Printf("\n  Propogating with tau = %1.2f", d_tau);
    for( int N = 1; N <= nstep; ++N ){
        tau += d_tau;
        for( size_t I = 0; I < ndet; ++I) {
            auto& C0_I = C0[I];
            auto& row_indices = H_sparse[I].first; 
            auto& row_values  = H_sparse[I].second;
            size_t row_dim = row_values.size();
            double re = 0.0;
            double im = 0.0;
            for( int J = 0; J < row_dim; ++J ){
                size_t idx = row_indices[J];
                double HIJC_r = C0[idx].first * row_values[J];
                double HIJC_i = C0[idx].second * row_values[J];
                re = C0_I.first + d_tau * HIJC_i; 
                im = C0_I.second - d_tau * HIJC_r; 
            }        
            C_tau[I] = std::make_pair(re,im);
        } 
        C0 = C_tau;   

        // Adaptively screen the wavefunction
        if( eta > 0.0 ){
            std::vector<std::pair<double, size_t>> sumsq(ndet);
            double norm = 0.0;
            for( int I = 0; I < ndet; ++I ){
                double re = C0[I].first;
                double im = C0[I].second;
                sumsq[I] = std::make_pair(re*re + im*im, I);
                norm += std::sqrt( re*re + im*im );
            } 
            norm = 1.0/ std::sqrt(norm);
            std::sort(sumsq.begin(), sumsq.end()); 
            double sum = 0.0;
            for( int I = 0; I < ndet; ++I ){
                double& val = sumsq[I].first;
                if( sum + val <= eta*norm ){
                    sum += val;
                    size_t idx = sumsq[I].second;
                    C0[idx] = std::make_pair(0.0,0.0);
                }else{
                    break;
                }
            }
        }
        // print the wavefunction
        //if( N % 100 == 0 ){ 
        if( N == (print_val) ){ 
            outfile->Printf("\n Saving wavefunction for t = %1.3f as", tau/0.0413413745758);
            std::vector<double> sum_sq(ndet);
            for( int I = 0; I < ndet; ++I ){
                double re = C0[I].first;
                double im = C0[I].second;
                sum_sq[I] = re*re + im*im;
            } 
            //save_vector(sumsq,"tau_"+ std::to_string(tau) + ".txt");
            std::stringstream ss;
            ss << std::setprecision(3) << tau/0.0413413745758;
            save_vector(sum_sq,"tau_" + ss.str()+ ".txt");
            print_val *= 10;
        }
    } 
}


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
