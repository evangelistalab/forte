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
#include <libmints/molecule.h>
#include <libpsio/psio.h>
#include <libpsio/psio.hpp>

#include "integrals.h"
#include "iterative_solvers.h"
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

class MOSpaceInfo;

FCI::FCI(boost::shared_ptr<Wavefunction> wfn, Options &options, ExplorerIntegrals* ints, std::shared_ptr<MOSpaceInfo> mo_space_info)
    : Wavefunction(options,_default_psio_lib_),
      options_(options), ints_(ints), mo_space_info_(mo_space_info)
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
    double nuclear_repulsion_energy = molecule_->nuclear_repulsion_energy();

    Dimension active_dim = mo_space_info_->get_dimension("ACTIVE");
    size_t nfdocc = mo_space_info_->size("FROZEN_DOCC");
    std::vector<size_t> rdocc = mo_space_info_->get_corr_abs_mo("RESTRICTED_DOCC");
    std::vector<size_t> active = mo_space_info_->get_corr_abs_mo("ACTIVE");

    int charge       = Process::environment.molecule()->molecular_charge();
    int multiplicity = Process::environment.molecule()->multiplicity();
    int nel = 0;

    // If the charge has changed, recompute the number of electrons
    // Or if you cannot find the number of electrons
    if((nel == 0) or options_["CHARGE"].has_changed()){
        charge = options_.get_int("CHARGE");
        nel = 0;
        int natom = Process::environment.molecule()->natom();
        for(int i=0; i < natom;i++){
            nel += static_cast<int>(Process::environment.molecule()->Z(i));
        }
        nel -= charge;
    }

    if(options_["MULTIPLICITY"].has_changed()){
        multiplicity = options_.get_int("MULTIPLICITY");
    }

    if( ((nel + 1 - multiplicity) % 2) != 0)
        throw PSIEXCEPTION("\n\n  FCI: Wrong multiplicity.\n\n");
    nel -= 2 * nfdocc - rdocc.size();


    size_t na = (nel + multiplicity - 1) / 2;
    size_t nb =  nel - na;

    outfile->Printf("\n  Multiplicity: %d",multiplicity);

//    size_t na = doccpi_.sum() + soccpi_.sum() - nfdocc - rdocc.size();
//    size_t nb = doccpi_.sum() - nfdocc - rdocc.size();

    FCISolver fcisolver(active_dim,rdocc,active,na,nb,options_.get_int("ROOT_SYM"),ints_);

    double fci_energy = fcisolver.compute_energy() + nuclear_repulsion_energy;

    Process::environment.globals["CURRENT ENERGY"] = fci_energy;
    Process::environment.globals["FCI ENERGY"] = fci_energy;

    return fci_energy;
}


FCISolver::FCISolver(Dimension active_dim,std::vector<size_t> core_mo,std::vector<size_t> active_mo,size_t na, size_t nb,size_t symmetry,ExplorerIntegrals* ints)
    : active_dim_(active_dim), core_mo_(core_mo), active_mo_(active_mo), ints_(ints), nirrep_(active_dim.n()), symmetry_(symmetry), na_(na), nb_(nb), nroot_(0)
{
    startup();
}

void FCISolver::startup()
{
    // Create the string lists
    lists_ = boost::shared_ptr<StringLists>(new StringLists(twoSubstituitionVVOO,active_dim_,core_mo_,active_mo_,na_,nb_));

    size_t ndfci = 0;
    for (int h = 0; h < nirrep_; ++h){
        size_t nastr = lists_->alfa_graph()->strpi(h);
        size_t nbstr = lists_->beta_graph()->strpi(h ^ symmetry_);
        ndfci += nastr * nbstr;
    }
    outfile->Printf("\n  Number of determinants    = %zu",ndfci);

}

/*
 * See Appendix A in J. Comput. Chem. 2001 vol. 22 (13) pp. 1574-1589
*/
double FCISolver::compute_energy()
{
    boost::timer t;

    double nuclear_repulsion_energy = Process::environment.molecule()->nuclear_repulsion_energy();

    FCIWfn::allocate_temp_space(lists_,ints_,symmetry_);

    nroot_ = 1;

    FCIWfn Hdiag(lists_,ints_,symmetry_);
    FCIWfn C(lists_,ints_,symmetry_);
    FCIWfn HC(lists_,ints_,symmetry_);

    size_t fci_size = Hdiag.size();
    Hdiag.form_H_diagonal();

    SharedVector b(new Vector("b",fci_size));
    SharedVector sigma(new Vector("sigma",fci_size));

    DavidsonLiuSolver dls(fci_size,nroot_);
    dls.set_print_level(0);
    Hdiag.copy_to(sigma);
    dls.startup(sigma);

    bool converged = false;
    double energy = 0.0;

    for (int cycle = 0; cycle < 30; ++cycle){
        bool add_sigma = true;
        for (int r = 0; r < nroot_ * 10; ++r){ // TODO : fix this loop
            dls.get_b(b);
            C.copy(b);
            C.Hamiltonian(HC,twoSubstituitionVVOO);
            HC.copy_to(sigma);
            add_sigma = dls.add_sigma(sigma);
            if (not add_sigma) break;
        }
        converged = dls.update();
        energy = dls.eigenvalues()->get(0) + nuclear_repulsion_energy;
        outfile->Printf("\n %3d  %20.12f",cycle,energy);
        if (converged) break;
    }


    if (converged){
        dls.get_results();
        C.copy(dls.eigenvector(0));
        C.compute_rdms();
    }
//    C.initial_guess(Hdiag,1);
//    for (int cycle = 0; cycle < 1000; ++cycle){
//        C.Hamiltonian(HC,twoSubstituitionVVOO);
//        energy = C.dot(HC) + nuclear_repulsion_energy;
//        outfile->Printf("\n %3d  %20.12f",cycle,energy);
//        C.copy(HC);
//        C.normalize();
//    }

    return energy;
}

}}
