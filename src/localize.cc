#include <libpsio/psio.h>
#include <libpsio/psio.hpp>
#include <libmints/molecule.h>

#include <liboptions/liboptions.h>
#include <libmints/local.h> 
#include <libmints/matrix.h>
#include <libmints/vector.h>

#include "localize.h"

namespace psi{ namespace forte{


LOCALIZE::LOCALIZE(boost::shared_ptr<Wavefunction> wfn, Options &options, std::shared_ptr<ForteIntegrals>  ints, std::shared_ptr<MOSpaceInfo> mo_space_info) : wfn_(wfn),ints_(ints)
{
    nfrz_ = mo_space_info->size("FROZEN_DOCC");
    nrst_ = mo_space_info->size("RESTRICTED_DOCC");
    namo_ = mo_space_info->size("ACTIVE");

    int nel = 0;    
    int natom = Process::environment.molecule()->natom();
    for(int i=0; i < natom;i++){
        nel += static_cast<int>(Process::environment.molecule()->Z(i));
    }
    nel -= options.get_int("CHARGE");

    // The wavefunction multiplicity
    multiplicity_ = options.get_int("MULTIPLICITY");
    outfile->Printf("\n MULT: %d",multiplicity_);
    int ms = multiplicity_ - 1;

    // The number of active electrons
    int nactel = nel - 2*nfrz_ - 2*nrst_;

    naocc_ = ((nactel - (nactel % 2)) / 2 ) + (nactel % 2);
    navir_ = namo_ - naocc_;

    abs_act_ = mo_space_info->get_absolute_mo("ACTIVE");

    local_type_ = options.get_str("LOCALIZE_TYPE");

    if( local_type_ == "BOYS" or local_type_ == "SPLIT_BOYS" ){
        local_method_ = "BOYS";
    }
    if( local_type_ == "PM" or local_type_ == "SPLIT_PM" ){
        local_method_ = "PIPEK_MEZEY";
    }

}

void LOCALIZE::localize_orbitals()
{
    SharedMatrix Ca = wfn_->Ca();
    SharedMatrix Cb = wfn_->Cb();

    Dimension nsopi = wfn_->nsopi();
    int nirrep = wfn_->nirrep();
    int off = 0;
    if( multiplicity_ == 3 ){
        naocc_ -= 1;
        navir_ -= 1;
        off = 2;
    }

    SharedMatrix Caocc( new Matrix("Caocc", nsopi[0], naocc_ )); 
    SharedMatrix Cavir( new Matrix("Cavir", nsopi[0], navir_ )); 

    for(int h = 0; h < nirrep; h++){
        for(int mu = 0; mu < nsopi[h]; mu++){
            for(int i = 0; i < naocc_; i++){
                Caocc->set(h, mu, i, Ca->get(h, mu, abs_act_[i]));
            }
            for(int i = 0; i < navir_; ++i){
                Cavir->set(h, mu, i, Ca->get(h, mu, abs_act_[i+naocc_+off]));
            } 
        }
    }

    boost::shared_ptr<BasisSet> primary = wfn_->basisset();

    boost::shared_ptr<Localizer> loc_a = Localizer::build( local_type_, primary, Caocc); 
    loc_a->localize();
    
    SharedMatrix Laocc = loc_a->L();

    boost::shared_ptr<Localizer> loc_v = Localizer::build( local_type_, primary, Cavir); 
    loc_v->localize();
    
    SharedMatrix Lvir = loc_v->L();

    for( int h = 0; h < nirrep; ++h){
        for( int i = 0; i < naocc_; ++i){
            SharedVector vec = Laocc->get_column(h, i);
            Ca->set_column(h, i+nfrz_+nrst_, vec );
            Cb->set_column(h, i+nfrz_+nrst_, vec );
        } 
        for( int i = 0 ; i < navir_; ++i){
            SharedVector vec = Lvir->get_column(h, i);
            Ca->set_column(h, i+nfrz_+nrst_+naocc_ + off, vec );
            Cb->set_column(h, i+nfrz_+nrst_+naocc_ + off, vec );
        } 
    }
   
    ints_->retransform_integrals();
}

LOCALIZE::~LOCALIZE()
{
}

}} // End Namespaces
