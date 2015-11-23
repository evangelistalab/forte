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

#include <boost/format.hpp>
#include <boost/timer.hpp>

#include <libciomr/libciomr.h>
#include <liboptions/liboptions.h>
#include <libmoinfo/libmoinfo.h>
#include <libmints/molecule.h>
#include <libpsio/psio.h>
#include <libpsio/psio.hpp>
#include "libmints/matrix.h"
#include "libmints/vector.h"

#include "stl_bitset_determinant.h"
#include "integrals.h"
#include "iterative_solvers.h"
#include "fci_solver.h"
#include "string_lists.h"
#include "helpers.h"
#include "reference.h"

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

int fci_debug_level = 4;

namespace psi{ namespace forte{

class MOSpaceInfo;

FCI::FCI(boost::shared_ptr<Wavefunction> wfn, Options &options,
         std::shared_ptr<ForteIntegrals>  ints, std::shared_ptr<MOSpaceInfo> mo_space_info)
    : Wavefunction(options,_default_psio_lib_),
      options_(options), ints_(ints), mo_space_info_(mo_space_info)
{
    // Copy the wavefunction information
    copy(wfn);
    print_ = options_.get_int("PRINT");

    startup();
}

FCI::~FCI()
{
    if (fcisolver_ != nullptr) delete fcisolver_;
}

void FCI::set_max_rdm_level(int value)
{
    max_rdm_level_ = value;
}
void FCI::set_fci_iterations(int value)
{
    fci_iterations_ = value;
}
void FCI::print_no(bool value)
{
    print_no_ = value;
}

void FCI::startup()
{  
    if(print_)
        print_method_banner({"String-based Full Configuration Interaction","by Francesco A. Evangelista"});

    max_rdm_level_ = options_.get_int("FCI_MAX_RDM");
    fci_iterations_ = options_.get_int("FCI_ITERATIONS");
    print_no_       = options_.get_bool("PRINT_NO");
}

double FCI::compute_energy()
{
    Dimension active_dim = mo_space_info_->get_dimension("ACTIVE");
    size_t nfdocc = mo_space_info_->size("FROZEN_DOCC");
    std::vector<size_t> rdocc = mo_space_info_->get_corr_abs_mo("RESTRICTED_DOCC");
    std::vector<size_t> active = mo_space_info_->get_corr_abs_mo("ACTIVE");

    int charge = Process::environment.molecule()->molecular_charge();
    if(options_["CHARGE"].has_changed()){
        charge = options_.get_int("CHARGE");
    }

    int nel = 0;
    int natom = Process::environment.molecule()->natom();
    for(int i=0; i < natom;i++){
        nel += static_cast<int>(Process::environment.molecule()->Z(i));
    }
    // If the charge has changed, recompute the number of electrons
    // Or if you cannot find the number of electrons
    nel -= charge;

    int multiplicity = Process::environment.molecule()->multiplicity();
    if(options_["MULTIPLICITY"].has_changed()){
        multiplicity = options_.get_int("MULTIPLICITY");
    }

    int ms = multiplicity - 1;
    if(options_.get_str("JOB_TYPE") == "DSRG-MRPT2" or
            options_.get_str("JOB_TYPE") == "THREE-DSRG-MRPT2")
    {
        ms   = 0;
    }
    if(options_["MS"].has_changed()){
        ms = options_.get_int("MS");
    }

    if(ms < 0){
        outfile->Printf("\n  Ms must be no less than 0.");
        outfile->Printf("\n  Ms = %2d, MULTIPLICITY = %2d", ms, multiplicity);
        outfile->Printf("\n  Check (specify) Ms value (component of multiplicity)! \n");
        throw PSIEXCEPTION("Ms must be no less than 0. Check output for details.");
    }

    if (print_){
        outfile->Printf("\n  Number of electrons: %d",nel);
        outfile->Printf("\n  Charge: %d",charge);
        outfile->Printf("\n  Multiplicity: %d",multiplicity);
        outfile->Printf("\n  Davidson subspace max dim: %d",options_.get_int("DAVIDSON_SUBSPACE_PER_ROOT"));
        outfile->Printf("\n  Davidson subspace min dim: %d",options_.get_int("DAVIDSON_COLLAPSE_PER_ROOT"));
        if (ms % 2 == 0){
            outfile->Printf("\n  M_s: %d",ms / 2);
        }else{
            outfile->Printf("\n  M_s: %d/2",ms);
        }
    }

    if( ((nel - ms) % 2) != 0)
        throw PSIEXCEPTION("\n\n  FCI: Wrong value of M_s.\n\n");

    // Adjust the number of for frozen and restricted doubly occupied
    size_t nactel = nel - 2 * nfdocc - 2 * rdocc.size();

    size_t na = (nactel + ms) / 2;
    size_t nb =  nactel - na;

    fcisolver_ = new FCISolver(active_dim,rdocc,active,na,nb,multiplicity,options_.get_int("ROOT_SYM"),ints_, mo_space_info_,
                               options_.get_int("NTRIAL_PER_ROOT"),print_, options_);
    // tweak some options
    fcisolver_->set_max_rdm_level(max_rdm_level_);
    fcisolver_->set_nroot(options_.get_int("NROOT"));
    fcisolver_->set_root(options_.get_int("ROOT"));
    fcisolver_->test_rdms(options_.get_bool("TEST_RDMS"));
    fcisolver_->set_fci_iterations(options_.get_int("FCI_ITERATIONS"));
    fcisolver_->set_collapse_per_root(options_.get_int("DAVIDSON_COLLAPSE_PER_ROOT"));
    fcisolver_->set_subspace_per_root(options_.get_int("DAVIDSON_SUBSPACE_PER_ROOT"));
    fcisolver_->print_no(print_no_);

    double fci_energy = fcisolver_->compute_energy();

    Process::environment.globals["CURRENT ENERGY"] = fci_energy;
    Process::environment.globals["FCI ENERGY"] = fci_energy;

    return fci_energy;
}

Reference FCI::reference()
{
    fcisolver_->set_max_rdm_level(3);
    return fcisolver_->reference();
}


FCISolver::FCISolver(Dimension active_dim, std::vector<size_t> core_mo,
                     std::vector<size_t> active_mo,
                     size_t na, size_t nb, size_t multiplicity, size_t symmetry,
                     std::shared_ptr<ForteIntegrals> ints, std::shared_ptr<MOSpaceInfo> mo_space_info,
                     size_t ntrial_per_root, int print, Options& options)
    : active_dim_(active_dim), core_mo_(core_mo), active_mo_(active_mo),
      ints_(ints), nirrep_(active_dim.n()), symmetry_(symmetry),
      na_(na), nb_(nb), multiplicity_(multiplicity), nroot_(0),
      ntrial_per_root_(ntrial_per_root), mo_space_info_(mo_space_info),
      print_(print),
      options_(options)
{
    nroot_ = options_.get_int("NROOT");
    startup();
}

FCISolver::FCISolver(Dimension active_dim, std::vector<size_t> core_mo,
                     std::vector<size_t> active_mo,
                     size_t na, size_t nb, size_t multiplicity, size_t symmetry,
                     std::shared_ptr<ForteIntegrals> ints, std::shared_ptr<MOSpaceInfo> mo_space_info, Options& options)
    : active_dim_(active_dim), core_mo_(core_mo), active_mo_(active_mo),
      ints_(ints), nirrep_(active_dim.n()), symmetry_(symmetry),
      na_(na), nb_(nb), multiplicity_(multiplicity), nroot_(0),
      mo_space_info_(mo_space_info), options_(options)
{
    ntrial_per_root_ = options_.get_int("NTRIAL_PER_ROOT");
    print_ = options_.get_int("PRINT");
    startup();
}

void FCISolver::set_max_rdm_level(int value)
{
    max_rdm_level_ = value;
}

void FCISolver::set_nroot(int value)
{
    nroot_ = value;
}

void FCISolver::set_root(int value)
{
    root_ = value;
}

void FCISolver::set_fci_iterations(int value)
{
    fci_iterations_ = value;
}

void FCISolver::set_collapse_per_root(int value)
{
    collapse_per_root_ = value;
}

void FCISolver::set_subspace_per_root(int value)
{
    subspace_per_root_ = value;
}

void FCISolver::startup()
{
    // Create the string lists
    lists_ = std::shared_ptr<StringLists>(new StringLists(twoSubstituitionVVOO,active_dim_,core_mo_,active_mo_,na_,nb_,print_));

    size_t ndfci = 0;
    for (int h = 0; h < nirrep_; ++h){
        size_t nastr = lists_->alfa_graph()->strpi(h);
        size_t nbstr = lists_->beta_graph()->strpi(h ^ symmetry_);
        ndfci += nastr * nbstr;
    }
    if(print_){
        // Print a summary of options
        std::vector<std::pair<std::string,int>> calculation_info{
            {"Number of determinants",ndfci},
            {"Symmetry",symmetry_},
            {"Multiplicity",multiplicity_},
            {"Number of roots",nroot_},
            {"Target root",root_},
            {"Trial vectors per root",ntrial_per_root_}};

        // Print some information
        outfile->Printf("\n\n  ==> FCI Solver <==\n\n");
        for (auto& str_dim : calculation_info){
            outfile->Printf("    %-39s %10d\n",str_dim.first.c_str(),str_dim.second);
        }
        outfile->Flush();
    }
}

/*
 * See Appendix A in J. Comput. Chem. 2001 vol. 22 (13) pp. 1574-1589
*/
double FCISolver::compute_energy()
{
    boost::timer t;

    double nuclear_repulsion_energy = Process::environment.molecule()->nuclear_repulsion_energy();

    std::shared_ptr<FCIIntegrals> fci_ints = std::make_shared<FCIIntegrals>(lists_,ints_);
    DynamicBitsetDeterminant::set_ints(fci_ints);

    FCIWfn::allocate_temp_space(lists_,print_);

    FCIWfn Hdiag(lists_,symmetry_);
    C_ = std::make_shared<FCIWfn>(lists_,symmetry_);
    FCIWfn HC(lists_,symmetry_);

    size_t fci_size = Hdiag.size();
    Hdiag.form_H_diagonal(fci_ints);

    SharedVector b(new Vector("b",fci_size));
    SharedVector sigma(new Vector("sigma",fci_size));

    Hdiag.copy_to(sigma);

    DavidsonLiuSolver dls(fci_size,nroot_);
    dls.set_e_convergence(options_.get_double("E_CONVERGENCE"));
    dls.set_print_level(print_);
    dls.set_collapse_per_root(collapse_per_root_);
    dls.set_subspace_per_root(subspace_per_root_);
    dls.startup(sigma);

    size_t guess_size = dls.collapse_size();
    auto guess = initial_guess(Hdiag,guess_size,multiplicity_,fci_ints);

    std::vector<int> guess_list;
    for (size_t g = 0; g < guess.size(); ++g){
        if (guess[g].first == multiplicity_) guess_list.push_back(g);
    }

    // number of guess to be used
    size_t nguess = std::min(guess_list.size(),guess_size);

    if (nguess == 0){
        throw PSIEXCEPTION("\n\n  Found zero FCI guesses with the requested multiplicity.\n\n");
    }

    for (size_t n = 0; n < nguess; ++n){
        HC.set(guess[guess_list[n]].second);
        HC.copy_to(sigma);
        dls.add_guess(sigma);
    }

    // Prepare a list of bad roots to project out and pass them to the solver
    std::vector<std::vector<std::pair<size_t,double>>> bad_roots;
    int gr = 0;
    for (auto& g : guess){
        if (g.first != multiplicity_){
            outfile->Printf("\n  Projecting out root %d",gr);
            HC.set(g.second);
            HC.copy_to(sigma);
            std::vector<std::pair<size_t,double>> bad_root;
            for (size_t I = 0; I < fci_size; ++I){
                if (std::fabs(sigma->get(I)) > 1.0e-12){
                    bad_root.push_back(std::make_pair(I,sigma->get(I)));
                }
            }
            bad_roots.push_back(bad_root);
        }
        gr += 1;
    }
    dls.set_project_out(bad_roots);

    SolverStatus converged = SolverStatus::NotConverged;

    if(print_){
        outfile->Printf("\n  ==> Diagonalizing Hamiltonian <==\n");
        outfile->Printf("\n  ----------------------------------------");
        outfile->Printf("\n    Iter.      Avg. Energy       Delta_E");
        outfile->Printf("\n  ----------------------------------------");
    }

    double old_avg_energy = 0.0;
    int real_cycle = 1;
    for (int cycle = 0; cycle < fci_iterations_; ++cycle){
        bool add_sigma = true;
        do{
            dls.get_b(b);
            C_->copy(b);
            C_->Hamiltonian(HC,fci_ints,twoSubstituitionVVOO);
            HC.copy_to(sigma);
            add_sigma = dls.add_sigma(sigma);
        } while (add_sigma);

        converged = dls.update();

        if (converged != SolverStatus::Collapse){
            double avg_energy = 0.0;
            for (int r = 0; r < nroot_; ++r){
                avg_energy += dls.eigenvalues()->get(r) + nuclear_repulsion_energy;
            }
            avg_energy /= static_cast<double>(nroot_);
            if (print_){
                outfile->Printf("\n    %3d  %20.12f  %+.3e",real_cycle,avg_energy,avg_energy - old_avg_energy);
            }
            old_avg_energy = avg_energy;
            real_cycle++;
        }

        if (converged == SolverStatus::Converged) break;
    }

    if (print_){
        outfile->Printf("\n  ----------------------------------------");
        if (converged == SolverStatus::Converged){
            outfile->Printf("\n  The Davidson-Liu algorithm converged in %d iterations.", real_cycle);
        }
    }

    if (converged == SolverStatus::NotConverged){
        outfile->Printf("\n  FCI did not converge!");
        exit(1);
    }

    // Print determinants
    if (print_){
        for (int r = 0; r < nroot_; ++r){
            outfile->Printf("\n\n  ==> Root No. %d <==\n",r);

            C_->copy(dls.eigenvector(r));
            std::vector<std::tuple<double,double,size_t,size_t,size_t>>
                    dets_config = C_->max_abs_elements(guess_size * ntrial_per_root_);
            Dimension nactvpi = mo_space_info_->get_dimension("ACTIVE");

            for(auto& det_config: dets_config){
                double ci_abs, ci;
                size_t h, add_Ia, add_Ib;
                std::tie(ci_abs, ci, h, add_Ia, add_Ib) = det_config;

                if(ci_abs < 0.1) continue;

                boost::dynamic_bitset<> Ia_v = lists_->alfa_str(h,add_Ia);
                boost::dynamic_bitset<> Ib_v = lists_->beta_str(h ^ symmetry_,add_Ib);

                outfile->Printf("\n    ");
                size_t offset = 0;
                for(int h = 0; h < nirrep_; ++h){
                    for(int k = 0; k < nactvpi[h]; ++k){
                        size_t i = k + offset;
                        bool a = Ia_v[i];
                        bool b = Ib_v[i];
                        if(a == b){
                            outfile->Printf("%d", a ? 2 : 0);
                        }else{
                            outfile->Printf("%c", a ? 'a' : 'b');
                        }
                    }
                    if(nactvpi[h] != 0)
                        outfile->Printf(" ");
                    offset += nactvpi[h];
                }
                outfile->Printf("%15.8f", ci);
            }

            double root_energy = dls.eigenvalues()->get(r) + nuclear_repulsion_energy;
            outfile->Printf("\n\n    Total Energy: %25.15f",root_energy);
        }
    }


    // Compute the RDMs
    if (converged == SolverStatus::Converged){
        C_->copy(dls.eigenvector(root_));
        if (print_) outfile->Printf("\n\n  ==> RDMs for Root No. %d <==",root_);
        C_->compute_rdms(max_rdm_level_);

        if(print_ > 1){C_->energy_from_rdms(fci_ints);}

        // Optionally, test the RDMs
        if (test_rdms_) C_->rdm_test();

        // Print the NO if energy converged
        if(print_no_) {C_->print_natural_orbitals();}
    }
    else
    {
        outfile->Printf("\n CI did not converge.");
        throw PSIEXCEPTION("CI did not converge.  Try setting FCI_ITERATIONS higher");
    }

    energy_ = dls.eigenvalues()->get(root_) + nuclear_repulsion_energy;

    return energy_;
}

std::vector<std::pair<int,std::vector<std::tuple<size_t,size_t,size_t,double>>>>
FCISolver::initial_guess(FCIWfn& diag, size_t n, size_t multiplicity,
                         std::shared_ptr<FCIIntegrals> fci_ints)
{
    boost::timer t;

    double nuclear_repulsion_energy = Process::environment.molecule()->nuclear_repulsion_energy();
    double scalar_energy = fci_ints->scalar_energy();

    size_t ntrial = n * ntrial_per_root_;

    // Get the list of most important determinants
    std::vector<std::tuple<double,size_t,size_t,size_t>> dets = diag.min_elements(ntrial);

    size_t num_dets = dets.size();

    std::vector<STLBitsetDeterminant> bsdets;

    // Build the full determinants
    size_t nact = active_mo_.size();
    // The corrleated MO, not the actual number of molecule orbitals
    size_t nmo =  mo_space_info_->size("ACTIVE");

    for (auto det : dets){
        double e;
        size_t h, add_Ia, add_Ib;
        std::tie(e,h,add_Ia,add_Ib) = det;
        boost::dynamic_bitset<> Ia_v = lists_->alfa_str(h,add_Ia);
        boost::dynamic_bitset<> Ib_v = lists_->beta_str(h ^ symmetry_,add_Ib);

        std::vector<bool> Ia(nmo,false);
        std::vector<bool> Ib(nmo,false);

        for (size_t i = 0; i < nact; ++i){
            if (Ia_v[i]) Ia[i] = true;
            if (Ib_v[i]) Ib[i] = true;
        }
        STLBitsetDeterminant bsdet(Ia,Ib);
        bsdets.push_back(bsdet);
    }

    // Make sure that the spin space is complete
    STLBitsetDeterminant::enforce_spin_completeness(bsdets);
    if (bsdets.size() > num_dets){
        bool* Ia = new bool[nact];
        bool* Ib = new bool[nact];
        size_t nnew_dets = bsdets.size() - num_dets;
        outfile->Printf("\n  Initial guess space is incomplete.\n  Adding %d determinant(s).",nnew_dets);
        for (size_t i = 0; i < nnew_dets; ++i){
            // Find the address of a determinant
            size_t h, add_Ia, add_Ib;
            for (size_t j = 0; j < nact; ++j){
                Ia[j] = bsdets[num_dets + i].get_alfa_bit(j);
                Ib[j] = bsdets[num_dets + i].get_beta_bit(j);
            }
            h = lists_->alfa_graph()->sym(Ia);
            add_Ia = lists_->alfa_graph()->rel_add(Ia);
            add_Ib = lists_->beta_graph()->rel_add(Ib);
            std::tuple<double,size_t,size_t,size_t> d(0.0,h,add_Ia,add_Ib);
            dets.push_back(d);
        }
        delete[] Ia;
        delete[] Ib;
    }
    num_dets = dets.size();

    Matrix H("H",num_dets,num_dets);
    Matrix evecs("Evecs",num_dets,num_dets);
    Vector evals("Evals",num_dets);

    for (size_t I = 0; I < num_dets; ++I){
        for (size_t J = I; J < num_dets; ++J){
            double HIJ = bsdets[I].slater_rules(bsdets[J]);
            if (I == J) HIJ += scalar_energy;
            H.set(I,J,HIJ);
            H.set(J,I,HIJ);
        }
    }
    H.diagonalize(evecs, evals);


    std::vector<std::pair<int,std::vector<std::tuple<size_t,size_t,size_t,double>>>> guess;

    std::vector<string> s2_labels({"singlet","doublet","triplet","quartet","quintet","sextet","septet","octet","nonet","decaet","11-et","12-et"});
    std::vector<string> table;

    for (size_t r = 0; r < num_dets; ++r){
        double energy = evals.get(r) + nuclear_repulsion_energy;
        double norm = 0.0;
        double S2 = 0.0;
        for (size_t I = 0; I < num_dets; ++I){
            for (size_t J = 0; J < num_dets; ++J){
                const double S2IJ = bsdets[I].spin2(bsdets[J]);
                S2 += evecs.get(I,r) * evecs.get(J,r) * S2IJ;
            }
            norm += std::pow(evecs.get(I,r),2.0);
        }
        S2 /= norm;
        double S = std::fabs(0.5 * (std::sqrt(1.0 + 4.0 * S2) - 1.0));
        int SS = std::round(S * 2.0);
        int state_multp = SS + 1;
        std::string state_label = s2_labels[SS];
        table.push_back(boost::str(boost::format("    %3d  %20.12f  %.3f  %s") % r % energy % std::fabs(S2) % state_label.c_str()));
        // Save states of the desired multiplicity
        std::vector<std::tuple<size_t,size_t,size_t,double>> solution;
        for (size_t I = 0; I < num_dets; ++I){
            auto det = dets[I];
            double e;
            size_t h, add_Ia, add_Ib;
            std::tie(e,h,add_Ia,add_Ib) = det;
            solution.push_back(std::make_tuple(h,add_Ia,add_Ib,evecs.get(I,r)));
        }
        guess.push_back(std::make_pair(state_multp,solution));
    }
    if (print_){
        print_h2("FCI Initial Guess");
        outfile->Printf("\n  ---------------------------------------------");
        outfile->Printf("\n    Root            Energy     <S^2>   Spin");
        outfile->Printf("\n  ---------------------------------------------");
        outfile->Printf("\n%s",to_string(table,"\n").c_str());
        outfile->Printf("\n  ---------------------------------------------");
        outfile->Printf("\n  Timing for initial guess  = %10.3f s\n",t.elapsed());
        outfile->Flush();
    }

    return guess;
}

Reference FCISolver::reference()
{
    size_t nact = active_dim_.sum();
    size_t nact2 = nact * nact;
    size_t nact3 = nact2 * nact;
    size_t nact4 = nact3 * nact;
    size_t nact5 = nact4 * nact;

    // One-particle density matrices in the active space
    std::vector<double>& opdm_a = C_->opdm_a();
    std::vector<double>& opdm_b = C_->opdm_b();
    ambit::Tensor L1a = ambit::Tensor::build(ambit::kCore,"L1a",{nact,nact});
    ambit::Tensor L1b = ambit::Tensor::build(ambit::kCore,"L1b",{nact,nact});
    if (na_ >= 1){
        L1a.iterate([&](const::vector<size_t>& i,double& value){
            value = opdm_a[i[0] * nact + i[1]]; });
    }
    if (nb_ >= 1){
        L1b.iterate([&](const::vector<size_t>& i,double& value){
            value = opdm_b[i[0] * nact + i[1]]; });
    }

    // Two-particle density matrices in the active space
    ambit::Tensor L2aa = ambit::Tensor::build(ambit::kCore,"L2aa",{nact,nact,nact,nact});
    ambit::Tensor L2ab = ambit::Tensor::build(ambit::kCore,"L2ab",{nact,nact,nact,nact});
    ambit::Tensor L2bb = ambit::Tensor::build(ambit::kCore,"L2bb",{nact,nact,nact,nact});
    ambit::Tensor g2aa = ambit::Tensor::build(ambit::kCore,"L2aa",{nact,nact,nact,nact});
    ambit::Tensor g2ab = ambit::Tensor::build(ambit::kCore,"L2ab",{nact,nact,nact,nact});
    ambit::Tensor g2bb = ambit::Tensor::build(ambit::kCore,"L2bb",{nact,nact,nact,nact});

    if (na_ >= 2){
        std::vector<double>& tpdm_aa = C_->tpdm_aa();
        L2aa.iterate([&](const::vector<size_t>& i,double& value){
            value = tpdm_aa[i[0] * nact3 + i[1] * nact2 + i[2] * nact + i[3]]; });
    }
    if ((na_ >= 1) and (nb_ >= 1)){
        std::vector<double>& tpdm_ab = C_->tpdm_ab();
        L2ab.iterate([&](const::vector<size_t>& i,double& value){
            value = tpdm_ab[i[0] * nact3 + i[1] * nact2 + i[2] * nact + i[3]]; });
    }
    if (nb_ >= 2){
        std::vector<double>& tpdm_bb = C_->tpdm_bb();
        L2bb.iterate([&](const::vector<size_t>& i,double& value){
            value = tpdm_bb[i[0] * nact3 + i[1] * nact2 + i[2] * nact + i[3]]; });
    }
    g2aa.copy(L2aa);
    g2ab.copy(L2ab);
    g2bb.copy(L2bb);

    // Convert the 2-RDMs to 2-RCMs
    L2aa("pqrs") -= L1a("pr") * L1a("qs");
    L2aa("pqrs") += L1a("ps") * L1a("qr");

    L2ab("pqrs") -= L1a("pr") * L1b("qs");

    L2bb("pqrs") -= L1b("pr") * L1b("qs");
    L2bb("pqrs") += L1b("ps") * L1b("qr");

    // Three-particle density matrices in the active space
    ambit::Tensor L3aaa = ambit::Tensor::build(ambit::kCore,"L3aaa",{nact,nact,nact,nact,nact,nact});
    ambit::Tensor L3aab = ambit::Tensor::build(ambit::kCore,"L3aab",{nact,nact,nact,nact,nact,nact});
    ambit::Tensor L3abb = ambit::Tensor::build(ambit::kCore,"L3abb",{nact,nact,nact,nact,nact,nact});
    ambit::Tensor L3bbb = ambit::Tensor::build(ambit::kCore,"L3bbb",{nact,nact,nact,nact,nact,nact});
    if (na_ >= 3){
        std::vector<double>& tpdm_aaa = C_->tpdm_aaa();
        L3aaa.iterate([&](const::vector<size_t>& i,double& value){
            value = tpdm_aaa[i[0] * nact5 + i[1] * nact4 + i[2] * nact3 + i[3] * nact2 + i[4] * nact + i[5]]; });
    }
    if ((na_ >= 2) and (nb_ >= 1)){
        std::vector<double>& tpdm_aab = C_->tpdm_aab();
        L3aab.iterate([&](const::vector<size_t>& i,double& value){
            value = tpdm_aab[i[0] * nact5 + i[1] * nact4 + i[2] * nact3 + i[3] * nact2 + i[4] * nact + i[5]]; });
    }
    if ((na_ >= 1) and (nb_ >= 2)){
        std::vector<double>& tpdm_abb = C_->tpdm_abb();
        L3abb.iterate([&](const::vector<size_t>& i,double& value){
            value = tpdm_abb[i[0] * nact5 + i[1] * nact4 + i[2] * nact3 + i[3] * nact2 + i[4] * nact + i[5]]; });
    }
    if (nb_ >= 3){
        std::vector<double>& tpdm_bbb = C_->tpdm_bbb();
        L3bbb.iterate([&](const::vector<size_t>& i,double& value){
            value = tpdm_bbb[i[0] * nact5 + i[1] * nact4 + i[2] * nact3 + i[3] * nact2 + i[4] * nact + i[5]]; });
    }

    // Convert the 3-RDMs to 3-RCMs
    L3aaa("pqrstu") -= L1a("ps") * L2aa("qrtu");
    L3aaa("pqrstu") += L1a("pt") * L2aa("qrsu");
    L3aaa("pqrstu") += L1a("pu") * L2aa("qrts");

    L3aaa("pqrstu") -= L1a("qt") * L2aa("prsu");
    L3aaa("pqrstu") += L1a("qs") * L2aa("prtu");
    L3aaa("pqrstu") += L1a("qu") * L2aa("prst");

    L3aaa("pqrstu") -= L1a("ru") * L2aa("pqst");
    L3aaa("pqrstu") += L1a("rs") * L2aa("pqut");
    L3aaa("pqrstu") += L1a("rt") * L2aa("pqsu");

    L3aaa("pqrstu") -= L1a("ps") * L1a("qt") * L1a("ru");
    L3aaa("pqrstu") -= L1a("pt") * L1a("qu") * L1a("rs");
    L3aaa("pqrstu") -= L1a("pu") * L1a("qs") * L1a("rt");

    L3aaa("pqrstu") += L1a("ps") * L1a("qu") * L1a("rt");
    L3aaa("pqrstu") += L1a("pu") * L1a("qt") * L1a("rs");
    L3aaa("pqrstu") += L1a("pt") * L1a("qs") * L1a("ru");


    L3aab("pqRstU") -= L1a("ps") * L2ab("qRtU");
    L3aab("pqRstU") += L1a("pt") * L2ab("qRsU");

    L3aab("pqRstU") -= L1a("qt") * L2ab("pRsU");
    L3aab("pqRstU") += L1a("qs") * L2ab("pRtU");

    L3aab("pqRstU") -= L1b("RU") * L2aa("pqst");

    L3aab("pqRstU") -= L1a("ps") * L1a("qt") * L1b("RU");
    L3aab("pqRstU") += L1a("pt") * L1a("qs") * L1b("RU");


    L3abb("pQRsTU") -= L1a("ps") * L2bb("QRTU");

    L3abb("pQRsTU") -= L1b("QT") * L2ab("pRsU");
    L3abb("pQRsTU") += L1b("QU") * L2ab("pRsT");

    L3abb("pQRsTU") -= L1b("RU") * L2ab("pQsT");
    L3abb("pQRsTU") += L1b("RT") * L2ab("pQsU");

    L3abb("pQRsTU") -= L1a("ps") * L1b("QT") * L1b("RU");
    L3abb("pQRsTU") += L1a("ps") * L1b("QU") * L1b("RT");


    L3bbb("pqrstu") -= L1b("ps") * L2bb("qrtu");
    L3bbb("pqrstu") += L1b("pt") * L2bb("qrsu");
    L3bbb("pqrstu") += L1b("pu") * L2bb("qrts");

    L3bbb("pqrstu") -= L1b("qt") * L2bb("prsu");
    L3bbb("pqrstu") += L1b("qs") * L2bb("prtu");
    L3bbb("pqrstu") += L1b("qu") * L2bb("prst");

    L3bbb("pqrstu") -= L1b("ru") * L2bb("pqst");
    L3bbb("pqrstu") += L1b("rs") * L2bb("pqut");
    L3bbb("pqrstu") += L1b("rt") * L2bb("pqsu");

    L3bbb("pqrstu") -= L1b("ps") * L1b("qt") * L1b("ru");
    L3bbb("pqrstu") -= L1b("pt") * L1b("qu") * L1b("rs");
    L3bbb("pqrstu") -= L1b("pu") * L1b("qs") * L1b("rt");

    L3bbb("pqrstu") += L1b("ps") * L1b("qu") * L1b("rt");
    L3bbb("pqrstu") += L1b("pu") * L1b("qt") * L1b("rs");
    L3bbb("pqrstu") += L1b("pt") * L1b("qs") * L1b("ru");

    if (print_ > 1)
        for (auto L1 : {L1a,L1b}){
            outfile->Printf("\n\n** %s **",L1.name().c_str());
            L1.iterate([&](const::vector<size_t>& i,double& value){
                if (std::fabs(value) > 1.0e-15)
                    outfile->Printf("\n  Lambda [%3lu][%3lu] = %18.15lf", i[0], i[1], value);
            });

        }

    if (print_ > 2)
        for (auto L2 : {L2aa,L2ab,L2bb}){
            outfile->Printf("\n\n** %s **",L2.name().c_str());
            L2.iterate([&](const::vector<size_t>& i,double& value){
                if (std::fabs(value) > 1.0e-15)
                    outfile->Printf("\n  Lambda [%3lu][%3lu][%3lu][%3lu] = %18.15lf", i[0], i[1], i[2], i[3], value);
            });

        }

    if (print_ > 3)
        for (auto L3 : {L3aaa,L3aab,L3abb,L3bbb}){
            outfile->Printf("\n\n** %s **",L3.name().c_str());
            L3.iterate([&](const::vector<size_t>& i,double& value){
                if (std::fabs(value) > 1.0e-15)
                    outfile->Printf("\n  Lambda [%3lu][%3lu][%3lu][%3lu][%3lu][%3lu] = %18.15lf", i[0], i[1], i[2], i[3], i[4], i[5], value);
            });
        }

    Reference fci_ref(energy_,L1a,L1b,L2aa,L2ab,L2bb,L3aaa,L3aab,L3abb,L3bbb);
    fci_ref.set_g2aa(g2aa);
    fci_ref.set_g2ab(g2ab);
    fci_ref.set_g2bb(g2bb);
    return fci_ref;
}

}}
