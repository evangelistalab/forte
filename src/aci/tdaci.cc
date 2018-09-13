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
    
    // 2. Now, build the full Hamiltonian in the n-1 space
    DeterminantHashVec ann_dets;
    for( int i = 0; i < ncmo; ++i ){
        annihilate_wfn(aci_dets, ann_dets,i);  
    }

    size_t nann = ann_dets.size();
    outfile->Printf("\n  size of ann dets: %zu", ann_dets.size());

    // 3. Remove a core electron from aci wfn, compile coeffss
    outfile->Printf("\n  Applying annihilation to orbital 0");
    DeterminantHashVec core_dets;
    //std::vector<double> core_coeffs( ann_dets.size(), 0.0);
    SharedVector core_coeffs = std::make_shared<Vector>("core", nann); 
    core_coeffs->zero();

    const det_hashvec& dets = aci_dets.wfn_hash();
    size_t ndet = dets.size();
    size_t ncore = 0;
    for ( size_t I = 0; I < ndet; ++I ){
        auto& detI = dets[I];
        if( detI.get_alfa_bit(0) == true ){
            Determinant adet(detI);
            adet.set_alfa_bit(0, false);
            size_t idx = ann_dets.get_idx(adet);
            core_coeffs->set(idx, core_coeffs->get(idx) + aci_coeffs->get(aci_dets.get_idx(detI),0)); 
            core_dets.add(adet);
//            outfile->Printf("\n %s  %s",detI.str(ncmo).c_str(),  adet.str(ncmo).c_str());
        }  
        if( detI.get_beta_bit(0) == true ){
            Determinant adet(detI);
            adet.set_beta_bit(0, false);
            size_t idx = ann_dets.get_idx(adet);
            core_coeffs->set(idx, core_coeffs->get(idx) + aci_coeffs->get(aci_dets.get_idx(detI),0)); 
            core_dets.add(adet);
//            outfile->Printf("\n %s  %s",detI.str(ncmo).c_str(),  adet.str(ncmo).c_str());
        }  
    }
    outfile->Printf("\n  size of core dets: %zu", core_dets.size());    
    
    // 2. Renormalize wave function
    outfile->Printf("\n  Renormalizing wave function");
//    renormalize_wfn( core_coeffs );
    double norm = core_coeffs->norm();
    norm = 1.0/norm;
    core_coeffs->scale(norm);

    // 3. Build full n-1 hamiltonian
    std::shared_ptr<FCIIntegrals> fci_ints_ = aci->get_aci_ints();

    SharedMatrix full_aH = std::make_shared<Matrix>("aH",nann, nann);
    for( size_t I = 0; I < nann; ++I ){
        Determinant detI = ann_dets.get_det(I);
        for( size_t J = I; J < nann; ++J ){
            Determinant detJ = ann_dets.get_det(J);
            double value = fci_ints_->slater_rules(detI,detJ);
            full_aH->set(I,J, value);
            full_aH->set(J,I, value);
        }
    }

    // Print full Hamiltonian
    save_matrix( full_aH, "hamiltonian.txt");

    SharedMatrix full_evecs = std::make_shared<Matrix>("evec", nann,nann);
    SharedVector full_evals = std::make_shared<Vector>("evals", nann);

    full_aH->diagonalize(full_evecs, full_evals);


//    full_evecs->print();
//    core_coeffs->print();

    save_vector(core_coeffs,"c_init.txt");
    save_matrix(full_evecs, "full_evecs.txt");
    save_vector(full_evals, "full_evals.txt");

    return en;
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
