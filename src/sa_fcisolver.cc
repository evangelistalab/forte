#include "sa_fcisolver.h"
#include <libmints/mints.h>

namespace psi{ namespace forte{

SA_FCISolver::SA_FCISolver(Options& options, boost::shared_ptr<Wavefunction> wfn)
    : options_(options), wfn_(wfn)
{
    read_options();
}

void SA_FCISolver::read_options()
{
   int number_of_states_per_irrep_mult = options_["SA_STATES"].size();
   std::vector<std::tuple<int, int, int> >parsed_option;
   bool print = options_.get_bool("CASSCF_DEBUG_PRINTING");
   int number_of_states = 0;
   for(int i = 0; i < number_of_states_per_irrep_mult; i++)
   {
       int symmetry = options_["SA_STATES"][i][0].to_integer();
       int multiplicity = options_["SA_STATES"][i][1].to_integer();
       int states_per_sym_mult = options_["SA_STATES"][i][2].to_integer();
       std::tuple<int, int, int> tuple_state = std::make_tuple(symmetry, multiplicity, states_per_sym_mult);
       number_of_states += states_per_sym_mult;
       parsed_option.push_back(tuple_state);
       if(symmetry > wfn_->nirrep() or symmetry < 0)
       {
           outfile->Printf("Please check the symmetry of your molecule and the symmetry you specified in SA_STATES");
           outfile->Printf("\n The number of irreps is %d and you specified %d symmetry", wfn_->nirrep(), symmetry);
           throw PSIEXCEPTION("You either asked for irrep greater than number of irreps or requested a negative irrep");

       }
       if(multiplicity < 0)
       {
           throw PSIEXCEPTION("You asked for a negative multiplicity");
       }
   }
   parsed_options_ = parsed_option;
   outfile->Printf("\n There are %d separate states", number_of_states_per_irrep_mult);
   outfile->Printf("\n There are %d total states to be averaged", number_of_states);
   number_of_states_ = number_of_states;
   if(print)
   {
       for(auto& state : parsed_option)
       {
           outfile->Printf("\n %d %d %d", std::get<0>(state), std::get<1>(state),std::get<2>(state));
       }
   }

}
double SA_FCISolver::compute_energy()
{
    double E_sa_casscf = 0.0;
    Reference sa_cas;
    std::vector<double> casscf_energies;
    std::vector<Reference> sa_cas_ref;
    size_t na = mo_space_info_->size("ACTIVE");


    for(const auto& cas_solutions : parsed_options_)
    {
        int symmetry = std::get<0>(cas_solutions);
        int multiplicity = std::get<1>(cas_solutions);
        int nroot        = std::get<2>(cas_solutions);
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


        // Default: lowest spin solution
        int ms = (multiplicity + 1) % 2;

        if(options_["MS"].has_changed()){
            ms = options_.get_int("MS");
        }

        if(ms < 0){
            outfile->Printf("\n  Ms must be no less than 0.");
            outfile->Printf("\n  Ms = %2d, MULTIPLICITY = %2d", ms, multiplicity);
            outfile->Printf("\n  Check (specify) Ms value (component of multiplicity)! \n");
            throw PSIEXCEPTION("Ms must be no less than 0. Check output for details.");
        }

        if (options_.get_int("PRINT")){
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
        FCISolver fcisolver(active_dim,rdocc,active,na,nb,multiplicity,symmetry,ints_, mo_space_info_,
                        options_.get_int("NTRIAL_PER_ROOT"),options_.get_int("PRINT"), options_);
        fcisolver.set_max_rdm_level(2);
        fcisolver.test_rdms(options_.get_bool("TEST_RDMS"));
        fcisolver.set_fci_iterations(options_.get_int("FCI_ITERATIONS"));
        fcisolver.set_collapse_per_root(options_.get_int("DAVIDSON_COLLAPSE_PER_ROOT"));
        fcisolver.set_subspace_per_root(options_.get_int("DAVIDSON_SUBSPACE_PER_ROOT"));
        fcisolver.print_no(false);
        fcisolver.use_user_integrals_and_restricted_docc(true);
        if(fci_ints_ == nullptr)
        {
            outfile->Printf("\n\n You need to set fci_ints");
            throw PSIEXCEPTION("Set FCI INTS");
        }
        else{
            fcisolver.set_integral_pointer(fci_ints_);
        }
        fcisolver.set_nroot(nroot);
        for(int root_number = 0; root_number < nroot; root_number++)
        {
            fcisolver.set_root(root_number);
            double E_casscf = fcisolver.compute_energy();
            casscf_energies.push_back(E_casscf);
            sa_cas_ref.push_back(fcisolver.reference());
        }
    }
    if(options_.get_str("SA_WEIGHTS") == "EQUAL")
    {
        for(auto& casscf_energy : casscf_energies)
        {
            E_sa_casscf += casscf_energy;
        }
        E_sa_casscf /= casscf_energies.size();
        sa_cas.set_Eref(E_sa_casscf);
        ambit::Tensor L1a_sa = ambit::Tensor::build(ambit::CoreTensor, "L1a_sa",   {na, na});
        ambit::Tensor L1b_sa = ambit::Tensor::build(ambit::CoreTensor, "L1b_sa",   {na, na});
        ambit::Tensor L2aa_sa = ambit::Tensor::build(ambit::CoreTensor, "L2aa_sa", {na, na, na, na});
        ambit::Tensor L2ab_sa = ambit::Tensor::build(ambit::CoreTensor, "L2ab_sa", {na, na, na, na});
        ambit::Tensor L2bb_sa = ambit::Tensor::build(ambit::CoreTensor, "L2bb_sa", {na, na, na, na});
        for(auto& casscf_ref : sa_cas_ref)
        {
            L1a_sa("u, v") += casscf_ref.L1a()("u, v");
            L1b_sa("u, v") += casscf_ref.L1b()("u, v");
            L2aa_sa("u, v, x, y") += casscf_ref.g2aa()("u, v, x, y");
            L2ab_sa("u, v, x, y") += casscf_ref.g2ab()("u, v, x, y");
            L2bb_sa("u, v, x, y") += casscf_ref.g2bb()("u, v, x, y");
        }
        //sa_cas.set_g2ab(L2ab);
        //sa_cas.set_g2bb(L2bb);

        /// You need to first get SA RDM and use those to convert to sa-cumulants.
        L1a_sa.scale(1.0 / number_of_states_);
        L1b_sa.scale(1.0 / number_of_states_);
        L2aa_sa.scale(1.0 / number_of_states_);
        L2ab_sa.scale(1.0 / number_of_states_);
        L2bb_sa.scale(1.0 / number_of_states_);

        L2aa_sa("pqrs") -= L1a_sa("pr") * L1a_sa("qs");
        L2aa_sa("pqrs") += L1a_sa("ps") * L1a_sa("qr");

        L2ab_sa("pqrs") -= L1a_sa("pr") * L1b_sa("qs");

        L2bb_sa("pqrs") -= L1b_sa("pr") * L1b_sa("qs");
        L2bb_sa("pqrs") += L1b_sa("ps") * L1b_sa("qr");



        sa_cas.set_L1a(L1a_sa);
        sa_cas.set_L1b(L1b_sa);
        sa_cas.set_L2aa(L2aa_sa);
        sa_cas.set_L2ab(L2ab_sa);
        sa_cas.set_L2bb(L2bb_sa);

    }
    sa_ref_ = sa_cas;


    return E_sa_casscf;
}

}}
