#include <libpsio/psio.h>
#include <libpsio/psio.hpp>

#include <liboptions/liboptions.h>
#include <libmints/local.h> 
#include <libmints/matrix.h>
#include <libmints/vector.h>

#include "localize.h"

namespace psi{ namespace forte{


LOCALIZE::LOCALIZE(boost::shared_ptr<Wavefunction> wfn, Options &options, std::shared_ptr<ForteIntegrals>  ints, std::shared_ptr<MOSpaceInfo> mo_space_info)
{
    size_t nfrz = mo_space_info->size("FROZEN_DOCC");
    size_t ncmo = mo_space_info->size("CORRELATED");
    int nalpha = wfn->nalpha();
    size_t nirrep = mo_space_info->nirrep();
    
    SharedMatrix Ca = wfn->Ca();
    SharedMatrix Cb = wfn->Cb();


    SharedMatrix Caocc = wfn->Ca_subset("MO", "ACTIVE_OCC");
    SharedMatrix Cavir = wfn->Ca_subset("MO", "ACTIVE_VIR");

    Caocc->print();

    boost::shared_ptr<BasisSet> primary = wfn->basisset();

    local_type_ = options.get_str("LOCALIZE_TYPE");
//    update_cmat(primary, Ca_new, Cb_new);

    boost::shared_ptr<Localizer> loc_a = Localizer::build( local_type_, primary, Caocc); 
    loc_a->localize();
    
    SharedMatrix Laocc = loc_a->L();

    boost::shared_ptr<Localizer> loc_v = Localizer::build( local_type_, primary, Cavir); 
    loc_v->localize();
    
    SharedMatrix Lvir = loc_v->L();

    for( int h = 0; h < nirrep; ++h){
        for( int i = 0; i < nalpha; ++i){
            SharedVector vec = Laocc->get_column(h, i + nfrz);
            Ca->set_column(h, i+nfrz, vec );
            Cb->set_column(h, i+nfrz, vec );
        } 
        for( int i = nalpha; i < ncmo; ++i){
            SharedVector vec = Lvir->get_column(h, i-nalpha);
//            Ca->set_column(h, i+nfrz, vec );
//            Cb->set_column(h, i+nfrz, vec );
        } 
    }
    Ca->print();
   
    ints->retransform_integrals();
}

LOCALIZE::~LOCALIZE()
{
}

//void LOCALIZE::update_cmat( boost::shared_ptr<BasisSet> primary,  SharedMatrix Ca, SharedMatrix Cb)
//{
//
//    if( local_type_ == "BOYS" ){
//        BoysLocalizer blr_a(primary, Ca);
//        BoysLocalizer blr_b(primary, Cb);
//        blr_a.localize();
//        blr_b.localize();
//
//        Ca->copy(blr_a.L());
//        Cb->copy(blr_b.L());
//
//    }else if( local_type_ == "PM" ){
//        Ca->print();
//        
//        PMLocalizer pmr_a(primary, Ca);
//        PMLocalizer pmr_b(primary, Cb);
//        pmr_a.localize();
//        pmr_b.localize();
//        
//        SharedMatrix Ua = pmr_a.L();
//        SharedMatrix La = pmr_a.L();
//        SharedMatrix Lb = pmr_b.L();
//       
//       
//        Ca->copy(La);
//        Cb->copy(pmr_b.L());
//        Ca->print(); 
//    }//else if( local_type_ == "SPLIT_BOYS" ){
//   // }else if( local_type_ == "SPLIT_PM" ){
//   // }else{
//   // }
//
//}

}} // End Namespaces
