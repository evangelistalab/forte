/*
 *  fci_davidson_liu.cc
 *  Capriccio
 *
 *  Created by Francesco Evangelista on 3/21/09.
 *  Copyright 2009 __MyCompanyName__. All rights reserved.
 *
 */

#include <cmath>
#include <numeric>

#include <boost/timer.hpp>

#include <libciomr/libciomr.h>
#include <liboptions/liboptions.h>
#include <libmoinfo/libmoinfo.h>
#include <libpsio/psio.h>
#include <libpsio/psio.hpp>

#include "integrals.h"
#include "fci_solver.h"
#include "string_lists.h"
#include "wavefunction.h"
#include "helpers.h"

#include <psi4-dec.h>

using namespace std;
using namespace psi;
using namespace boost;

extern double h1_aa_timer;
extern double h1_bb_timer;
extern double h2_aaaa_timer;
extern double h2_aabb_timer;
extern double h2_bbbb_timer;
extern double oo_list_timer;
extern double vo_list_timer;
extern double vovo_list_timer;
extern double vvoo_list_timer;

namespace psi{ namespace libadaptive{

FCI::FCI(boost::shared_ptr<Wavefunction> wfn, Options &options, ExplorerIntegrals* ints)
    : Wavefunction(options,_default_psio_lib_),
      options_(options),
      ints_(ints)
{
    // Copy the wavefunction information
    copy(wfn);

    startup();
}

void FCI::startup()
{  
    print_method_banner({"String-based Full Configuration Interaction","written by Francesco A. Evangelista"});
}

double FCI::compute_energy()
{
    size_t na = doccpi_.sum() + soccpi_.sum();
    size_t nb = doccpi_.sum();

    Dimension frzcpi = ints_->frzcpi();
    Dimension ncmopi = ints_->ncmopi();

    std::vector<size_t> core_mo, active_mo;

    for (size_t h = 0, p = 0; h < ints_->nirrep(); ++h){
        for (size_t f = 0; f < frzcpi[h]; ++f){
            core_mo.push_back(p);
            p += 1;
        }
        for (size_t c = 0; c < ncmopi[h]; ++c){
            active_mo.push_back(p);
            p += 1;
        }
        p += nmopi_[h] - frzcpi[h] - ncmopi[h];
    }

    FCISolver fcisolver(core_mo,active_mo,na,nb,options_.get_int("ROOT_SYM"),ints_);
    return fcisolver.compute_energy();
}


FCISolver::FCISolver(std::vector<size_t> core_mo,std::vector<size_t> active_mo,size_t na, size_t nb,size_t symmetry,ExplorerIntegrals* ints)
    :core_mo_(core_mo), active_mo_(active_mo), ints_(ints), symmetry_(symmetry), na_(na), nb_(nb)
{
    startup();
}

void FCISolver::startup()
{
    nirrep_ = ints_->nirrep();
    Dimension ncmopi = ints_->ncmopi();

    // We define a Pitzer pair to be the pair of number (h,mo_h), where h is the irrep of an orbital
    // and mo_h is the orbital index in irrep h
    std::vector<std::pair<size_t,size_t>> mo_to_pitzer_pair;
    for (size_t h = 0, p = 0; h < nirrep_; ++h){
        for (size_t ph = 0; ph < ncmopi[h]; ++p, ++ph){
            mo_to_pitzer_pair.push_back(std::make_pair(h,ph));
        }
    }

    // Create a Dimension object used to build the string lists
    Dimension fcimopi(ncmopi.n());
    for (size_t i : active_mo_){
        size_t h = mo_to_pitzer_pair[i].first;
        fcimopi[h] += 1;
    }

    // Create the string lists
    lists_ = boost::shared_ptr<StringLists>(new StringLists(twoSubstituitionVVOO,fcimopi,core_mo_,active_mo_,na_,nb_));
}

/*
 * See Appendix A in J. Comput. Chem. 2001 vol. 22 (13) pp. 1574-1589
*/
double FCISolver::compute_energy()
{
    boost::timer t;

//    // Setup the lists required by the FCI algorithm

    FCIWfn::allocate_temp_space(lists_,ints_,symmetry_);
    int cycle = 0;
    int root = 0;
    double old_energy = 0.0;

    int max_guess_vectors = 30;
//    double** G;
//    double** a;
//    double*  rho;
//    allocate2(double,G,max_guess_vectors,max_guess_vectors);
//    allocate2(double,a,max_guess_vectors,max_guess_vectors);
//    allocate1(double,rho,max_guess_vectors);

//    // Are we selecting the CI determinants?
//    bool select_by_e = (options_.get_str("SELECT") == "LAMBDA") ? true : false;
//    double select_threshold = options_.get_double("SELECT_THRESHOLD");
//    if(select_by_e){
//        outfile->Printf("\n\n  Applying the energy selection criterium to the wave function");
//        outfile->Printf("\n  Lambda = %f Eh\n",select_threshold);
//    }

//    vector<boost::shared_ptr<FCIWfn> > b;
//    vector<boost::shared_ptr<FCIWfn> > sigma;
//    FCIWfn r(lists,ints);  // the residual
//    FCIWfn d(lists,ints);  // the delta


    FCIWfn Hdiag(lists_,ints_,symmetry_);
    Hdiag.form_H_diagonal();
    FCIWfn C(lists_,ints_,symmetry_);
    FCIWfn HC(lists_,ints_,symmetry_);
    C.initial_guess(Hdiag);

    C.print();
    C.Hamiltonian(HC,twoSubstituitionVVOO);
    HC.print();
    double energy = C.dot(HC);
    HC.normalize();

    outfile->Printf("\n  FCI energy = %20.12f",energy);


//    // Create a model space object
//    DetModelSpace model_space(lists,ints);
//    double energy = model_space.diagonalize(true);
//    outfile->Printf("\n\n  Reference energy = %20.12f",energy);

//    model_space.print_eigenvector();

//    double min_energy_det = Hdiag.min_element();

//    // Add b_i
//    b.push_back(boost::shared_ptr<FCIWfn>(new FCIWfn(lists,ints)));
//    b[0]->set_to(model_space);
//    b[0]->read();

//    // Compute sigma_i = H b_i
//    sigma.push_back(boost::shared_ptr<FCIWfn>(new FCIWfn(lists,ints)));
//    b[0]->Hamiltonian(*sigma[0]);
//    if (select_by_e){
//        sigma[0]->select_by_energy(ints,min_energy_det,select_threshold);
//    }

//    G[0][0] = sigma[0]->dot(*b[0]);

//    int L = 1;

//    sq_rsp(L,L,G,rho,1,a,1.0e-14);

//    energy = rho[root];
//    double delta_energy = energy - old_energy;

//    outfile->Printf("\n\n--------------------------------------------------");
//    outfile->Printf("\n  iter             E                  Delta(E)");
//    outfile->Printf("\n--------------------------------------------------");
//    outfile->Printf("\n  %4d %20.12f %20.12f",cycle
//                    ,energy + moinfo->get_nuclear_energy()
//                    ,delta_energy + moinfo->get_nuclear_energy());
//    outfile->Flush();

//    double convergence = std::pow(0.1, static_cast<double>(options_.get_int("CONVERGENCE")));
//    while ((std::abs(delta_energy) > convergence) && (cycle < max_guess_vectors - 1)) {
//        old_energy = energy;

//        // Form the residual
//        r.zero();
//        for(int i = 0; i < L; ++i)
//            r.plus_equal(a[i][root],*sigma[i]);
//        for(int i = 0; i < L; ++i)
//            r.plus_equal(-a[i][root] * rho[root],*b[i] );

//        // Update
//        d.davidson_update(rho[root],Hdiag,r);

//        // Orthnormalize delta with respect to all b's and add it to the list
//        d.normalize();
//        b.push_back(boost::shared_ptr<FCIWfn>(new FCIWfn(lists,ints)));
//        b[L]->set_to(d);
//        for(int i = 0; i < L; ++i)
//            b[L]->plus_equal(-b[L]->dot(*b[i]),*b[i]);
//        b[L]->normalize();

//        sigma.push_back(boost::shared_ptr<FCIWfn>(new FCIWfn(lists,ints)));
//        b[L]->Hamiltonian(*sigma[L]);
//        if (select_by_e){
//            sigma[L]->select_by_energy(ints,min_energy_det,select_threshold);
//        }

//        for(int i = 0; i < L; ++i)
//            G[L][i] = G[i][L] = sigma[L]->dot(*b[i]);
//        G[L][L] = sigma[L]->dot(*b[L]);

//        L++;

//        sq_rsp(L,L,G,rho,1,a,1.0e-14);
//        energy = rho[0];

//        delta_energy = energy - old_energy;

//        cycle++;

//        outfile->Printf("\n  %4d %20.12f %20.12f",cycle,energy + moinfo->get_nuclear_energy(),delta_energy);
//        outfile->Flush();
//    }

//    outfile->Printf("\n--------------------------------------------------");

//    outfile->Printf("\n\n  * FCI total energy  =  %20.15f\n",energy + moinfo->get_nuclear_energy());

//    if(options_.get_bool("SAVE_WFN")){
//        // From trial vector x
//        r.zero();
//        for(int i = 0; i < L; ++i)
//            r.plus_equal(a[i][0],*b[i]);

//        // Normalize the wfn
//        r.normalize();

//        r.save();
//    }

////    if (options_.get_int("DECOMPOSE_RANK") > 0){
////        // From trial vector x
////        r.zero();
////        for(int i = 0; i < L; ++i)
////            r.plus_equal(a[i][0],*b[i]);

////        // Normalize the wfn
////        d.zero();
////        r.normalize();

////        size_t rank = 1;
////        for (int n = 0; n < 13; n++){
////            Hdiag.set_to(r);
////            Hdiag.decompose(rank /*options_.get_int("DECOMPOSE_RANK")*/);
////            Hdiag.Hamiltonian(d);
////            double decomposed_energy = d.dot(Hdiag);
////            outfile->Printf("\n  decomposed_energy %10zu     = %26.16f\n",rank,decomposed_energy+ moinfo->get_nuclear_energy());
////            outfile->Flush();
////            rank *= 2;
////        }
////    }

//    outfile->Printf("\n  timing for H1_aa     = %10.3f s",h1_aa_timer);
//    outfile->Printf("\n  timing for H1_bb     = %10.3f s",h1_bb_timer);
//    outfile->Printf("\n  timing for H2_aaaa   = %10.3f s",h2_aaaa_timer);
//    outfile->Printf("\n  timing for H2_aabb   = %10.3f s",h2_aabb_timer);
//    outfile->Printf("\n  timing for H2_bbbb   = %10.3f s",h2_bbbb_timer);
//    outfile->Printf("\n  timing for fci       = %10.3f s",t.elapsed());
//    outfile->Printf("\n\n");
//    outfile->Flush();


//    release2(G);
//    release2(a);
//    release1(rho);
//    FCIWfn::release_temp_space();
    return energy;
}

}}
