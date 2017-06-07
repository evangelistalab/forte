/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2017 by its authors (see COPYING, COPYING.LESSER, AUTHORS).
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

#include "psi4/libmints/molecule.h"
#include "psi4/libmints/wavefunction.h"
#include "psi4/liboptions/liboptions.h"

#include "ci_reference.h"

#include <algorithm>

namespace psi {
namespace forte {

CI_Reference::CI_Reference( std::shared_ptr<Wavefunction> wfn, Options& options, 
                            std::shared_ptr<MOSpaceInfo> mo_space_info, STLBitsetDeterminant det, 
                            int multiplicity, double ms )
                        : wfn_(wfn), mo_space_info_(mo_space_info)
{
    multiplicity_ = multiplicity;
    ms_ = ms;

    root_sym_ = options.get_int("ROOT_SYM");

    nirrep_ = wfn_->nirrep();
    Dimension doccpi = wfn_->doccpi();
    Dimension soccpi = wfn_->soccpi();
    size_t ninact = mo_space_info_->size("INACTIVE_DOCC");
    mo_symmetry_ = mo_space_info_->symmetry("ACTIVE");
    nactpi_ = mo_space_info_->get_dimension("ACTIVE");

    frzcpi_ = mo_space_info_->get_dimension("INACTIVE_DOCC");

    // First determine number of alpha and beta electrons
    // Assume ms = 0.5 * ( Na - Nb ) 
    int nel = 0;
    for (int h = 0; h < nirrep_; ++h) {    
        nel += 2 * doccpi[h] + soccpi[h];
    }
   
    nel -= 2 * ninact;
     
    nalpha_ = 0.5 * (nel + 2 * ms_ ); 
    nbeta_ = nel - nalpha_;
        
    outfile->Printf("\n  Number of active orbitals: %d", STLBitsetDeterminant::nmo_);
    outfile->Printf("\n  Number of active alpha electrons: %d", nalpha_);
    outfile->Printf("\n  Number of active beta electrons: %d", nbeta_);
}

CI_Reference::~CI_Reference() {}   

void CI_Reference::build_reference( std::vector<STLBitsetDeterminant>& ref_space )
{
    ref_space.clear();
    int nact = mo_space_info_->size("ACTIVE");

    // Get the active mos
    auto active_mos = sym_labeled_orbitals("RHF");
    int homo = 0;
    
    for( int i = 0; i < active_mos.size(); ++i ){
        outfile->Printf("\n  %d  %d  %1.6f", std::get<2>(active_mos[i]), std::get<1>(active_mos[i]), std::get<0>(active_mos[i]));
        if( std::get<0>(active_mos[i]) > 0.0 ){
            homo = i;
            break;
        } 
    }

    // Pick the active subspace around the HOMO/LUMO gap
    std::vector<int> active_subspace;

    // Find a better way to pick the size
    int subspace_size = 6;
    if( subspace_size > nact ){
        subspace_size = nact;
    }


    int start_mo = homo - static_cast<int>(subspace_size/2.0);
    
    // Check against small calculations
    if( start_mo < 0 ){
        start_mo = 0;
    }

    outfile->Printf("\n start mo: %d", start_mo);
    for( int i = start_mo, max_i = start_mo + subspace_size; i < max_i; ++i ){
        active_subspace.push_back( std::get<2>(active_mos[i]) ); 
    }

    // Get number of electrons in subspace
    int nalpha_sub = nalpha_ - start_mo; 
    int nbeta_sub = nbeta_ - start_mo; 
    outfile->Printf("\n number alpha sub: %d", nalpha_sub); 
    outfile->Printf("\n number beta sub: %d", nbeta_sub); 

    // Fill a subspace determinant with initial permutation
    std::vector<bool> tmp_det_a(subspace_size,false);
    std::vector<bool> tmp_det_b(subspace_size,false);
    for( int i = 0; i < nalpha_sub; ++i ){
        tmp_det_a[i] = true; 
    } 
    for( int i = 0; i < nbeta_sub; ++i ){
        tmp_det_b[i] = true; 
    } 

    // Set the frozen part of a determinat
    STLBitsetDeterminant core_det;
    for( int i = 0; i < start_mo; ++i ){
        core_det.set_alfa_bit( std::get<2>(active_mos[i]), true );
        core_det.set_beta_bit( std::get<2>(active_mos[i]), true );
    }

    // Make sure we start with the first permutation
    std::sort( begin(tmp_det_a), end(tmp_det_a));
    std::sort( begin(tmp_det_b), end(tmp_det_b));

    // Generate all permutations, add the correct ones
    do {
        do {
            // Build determinant
            STLBitsetDeterminant det(core_det);     
            int sym = 0;
            for( int p = 0; p < subspace_size; ++p ){
                det.set_alfa_bit( active_subspace[p], tmp_det_a[p]);
                det.set_beta_bit( active_subspace[p], tmp_det_b[p]);
                if ( tmp_det_a[p] ){
                    sym ^= mo_symmetry_[active_subspace[p]];
                }
                if (tmp_det_b[p]) {
                    sym ^= mo_symmetry_[active_subspace[p]];
                }
            }
            // Check symmetry
            if( sym == root_sym_ ){
                ref_space.push_back(det);               
            }            

        } while( std::next_permutation( tmp_det_b.begin(), tmp_det_b.begin() + subspace_size  ) );
    } while( std::next_permutation( tmp_det_a.begin(), tmp_det_a.begin() + subspace_size ) );

    // Diagonalize the reference space Hamiltonian
    outfile->Printf("\n The reference space contains %zu determinants", ref_space.size());
    //SparseCISolver solver;
    //solver.set_spin_project_full(true);
    //solver.diagonalize_hamiltonian( ref

}

std::vector<std::tuple<double, int, int>> CI_Reference::sym_labeled_orbitals(std::string type) 
{
    int nact = mo_space_info_->size("ACTIVE");

    std::vector<std::tuple<double, int, int>> labeled_orb;

    std::shared_ptr<Vector> epsilon_a = wfn_->epsilon_a();

    if (type == "RHF" or type == "ROHF" or type == "ALFA") {

        // Create a vector of orbital energy and index pairs
        std::vector<std::pair<double, int>> orb_e;
        int cumidx = 0;
        for (int h = 0; h < nirrep_; ++h) {
            for (int a = 0; a < nactpi_[h]; ++a) {
                orb_e.push_back(
                    std::make_pair(epsilon_a->get(h, frzcpi_[h] + a), a + cumidx));
            }
            cumidx += nactpi_[h];
        }

        // Create a vector that stores the orbital energy, symmetry, and idx
        for (size_t a = 0; a < nact; ++a) {
            labeled_orb.push_back(
                std::make_tuple(orb_e[a].first, mo_symmetry_[a], orb_e[a].second));
        }
        // Order by energy, low to high
        std::sort(labeled_orb.begin(), labeled_orb.end());
    }
    if (type == "BETA") {
        // Create a vector of orbital energies and index pairs
        std::shared_ptr<Vector> epsilon_b = wfn_->epsilon_b();
        std::vector<std::pair<double, int>> orb_e;
        int cumidx = 0;
        for (int h = 0; h < nirrep_; ++h) {
            for (size_t a = 0, max = nactpi_[h]; a < max; ++a) {
                orb_e.push_back(
                    std::make_pair(epsilon_b->get(h, frzcpi_[h] + a), a + cumidx));
            }
            cumidx += nactpi_[h];
        }

        // Create a vector that stores the orbital energy, sym, and idx
        for (size_t a = 0; a < nact; ++a) {
            labeled_orb.push_back(
                std::make_tuple(orb_e[a].first, mo_symmetry_[a], orb_e[a].second));
        }
        std::sort(labeled_orb.begin(), labeled_orb.end());
    }
    return labeled_orb;
}


}} // End Namespaces
