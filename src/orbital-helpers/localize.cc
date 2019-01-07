/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2019 by its authors (see COPYING, COPYING.LESSER, AUTHORS).
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

#include "psi4/libpsio/psio.h"
#include "psi4/libpsio/psio.hpp"
#include "psi4/libmints/local.h"
#include "psi4/libqt/qt.h"
#include "psi4/libmints/matrix.h"
#include "psi4/libmints/vector.h"

#include "base_classes/reference.h"

#include "localize.h"

using namespace psi;

namespace forte {

LOCALIZE::LOCALIZE(StateInfo state, std::shared_ptr<SCFInfo> scf_info,
                   std::shared_ptr<ForteOptions> options, std::shared_ptr<ForteIntegrals> ints,
                   std::shared_ptr<MOSpaceInfo> mo_space_info)
    : OrbitalTransform(scf_info, options, ints, mo_space_info), scf_info_(scf_info),
      options_(options), ints_(ints) {
    nfrz_ = mo_space_info->size("FROZEN_DOCC");
    nrst_ = mo_space_info->size("RESTRICTED_DOCC");
    namo_ = mo_space_info->size("ACTIVE");

    if (ints_->nirrep() > 1) {
        throw psi::PSIEXCEPTION("\n\n ERROR: Localizer only implemented for C1 symmetry!");
    }

    
    orbital_spaces_ = options->get_int_vec("LOCALIZE_SPACE");
    outfile->Printf("\n\n");
}

void LOCALIZE::set_orbital_space(std::vector<int>& orbital_spaces){
    orbital_spaces_ = orbital_spaces;
}

void LOCALIZE::localize() {

    if( orbital_spaces_.size() == 0 ){
        outfile->Printf("\n  Error: Orbital space for localization is not set!");
        exit(1);
    } else if ( (orbital_spaces_.size() % 2 ) != 0 ) {
        outfile->Printf("\n  Error: Orbital space for localization not properly set!");
        exit(1);
    } 

    // Get references to C matrices
    psi::SharedMatrix Ca = ints_->Ca();
    psi::SharedMatrix Cb = ints_->Cb();

    size_t nmo = Ca->rowdim();
   
    U_ = std::make_shared<psi::Matrix>("U", nmo, nmo);
    U_->identity();

    // loop through each space
    for( size_t f_idx = 0, max = orbital_spaces_.size(); f_idx < max - 1; f_idx += 2 ){
    
        // indices are INCLUSIVE
        size_t first = orbital_spaces_[f_idx]; 
        size_t last  = orbital_spaces_[f_idx+1]; 

        //print
        outfile->Printf("\n  Localizing orbitals:");
        for( size_t orb = first; orb <= last; ++orb){
            outfile->Printf(" %d", orb);
        }
        
        if( last < first ){
            outfile->Printf("\n  Error: Orbital space for localization not properly set!");
            exit(1);
        } 

        // number of orbitals to localize
        size_t orb_dim = last - first + 1;

        // Build C matrix to localize
        psi::SharedMatrix Ca_loc = std::make_shared<psi::Matrix>("Caact", nmo, orb_dim);

        for( size_t i = 0; i < orb_dim; ++i ){
            psi::SharedVector col = Ca->get_column(0, first + i);
            Ca_loc->set_column(0, i, col); 
        }
    
        // localize
        std::shared_ptr<psi::BasisSet> primary = ints_->wfn()->basisset();
        std::shared_ptr<psi::Localizer> loc_a = psi::Localizer::build(local_method_, primary, Ca_loc);
        loc_a->localize();
        
        // Grab the transformation and localized matrices
        psi::SharedMatrix Laocc = loc_a->L();

        //Set Ca, Cb, and U
        for( size_t i = 0; i < orb_dim; ++i ){
            psi::SharedVector C_col = Laocc->get_column(0, i);
            
            Ca->set_column(0, i+first, C_col);
            Cb->set_column(0, i+first, C_col);
        }
    }

    ints_->update_orbitals(Ca, Cb);
}

psi::SharedMatrix LOCALIZE::get_Ua() { return Ua_; }
psi::SharedMatrix LOCALIZE::get_Ub() { return Ub_; }

} // namespace forte

