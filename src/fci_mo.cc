#include <cmath>
#include <numeric>
#include <algorithm>
#include <iomanip>
#include <sstream>
#include "mini-boost/boost/algorithm/string/predicate.hpp"
#include "fci_vector.h"
#include "fci_mo.h"

using namespace std;

namespace psi{ namespace forte{

FCI_MO::FCI_MO(SharedWavefunction ref_wfn, Options& options,
               std::shared_ptr<ForteIntegrals> ints, std::shared_ptr<MOSpaceInfo> mo_space_info)
    : Wavefunction(options), integral_(ints), mo_space_info_(mo_space_info)
{
    shallow_copy(ref_wfn);
    reference_wavefunction_ = ref_wfn;
    print_method_banner({"Complete Active Space Configuration Interaction","Chenyang Li"});
    startup();
}

FCI_MO::~FCI_MO()
{
    cleanup();
}

void FCI_MO::cleanup(){
}

void FCI_MO::startup(){

    // read options
    read_options();

    // setup integrals
    fci_ints_ = std::make_shared<FCIIntegrals>(integral_, mo_space_info_->get_corr_abs_mo("ACTIVE"),
                                               mo_space_info_->get_corr_abs_mo("RESTRICTED_DOCC"));
    ambit::Tensor tei_active_aa = integral_->aptei_aa_block(idx_a_, idx_a_, idx_a_, idx_a_);
    ambit::Tensor tei_active_ab = integral_->aptei_ab_block(idx_a_, idx_a_, idx_a_, idx_a_);
    ambit::Tensor tei_active_bb = integral_->aptei_bb_block(idx_a_, idx_a_, idx_a_, idx_a_);
    fci_ints_->set_active_integrals(tei_active_aa, tei_active_ab, tei_active_bb);
    fci_ints_->compute_restricted_one_body_operator();
    STLBitsetDeterminant::set_ints(fci_ints_);
    DynamicBitsetDeterminant::set_ints(fci_ints_);

    // compute so quadrupole for orbital extents
    compute_SOquadrupole();

    // compute orbital extents
    compute_orbital_extents();
}

void FCI_MO::read_options(){

    // test reference type
    ref_type_ = options_.get_str("REFERENCE");
    if(ref_type_ == "UHF" || ref_type_ == "UKS" || ref_type_ == "CUHF"){
        outfile->Printf("\n  Unrestricted reference is detected.");
        outfile->Printf("\n  We suggest using unrestricted natural orbitals.");
    }

    // active space type
    active_space_type_ = options_.get_str("ACTIVE_SPACE_TYPE");

    // IP / EA
    ipea_ = options_.get_str("IPEA");

    // set orbitals
    semi_ = options_.get_bool("SEMI_CANONICAL");

    // print level
    print_ = options_.get_int("PRINT");

    // energy convergence
    econv_ = options_.get_double("E_CONVERGENCE");
    dconv_ = options_.get_double("D_CONVERGENCE");

    // nuclear repulsion
    boost::shared_ptr<Molecule> molecule = Process::environment.molecule();
    e_nuc_ = molecule->nuclear_repulsion_energy();

    // number of Irrep
    nirrep_ = this->nirrep();

    // obtain MOs
    nmo_ = this->nmo();
    nmopi_ = this->nmopi();
    ncmo_ = mo_space_info_->size("CORRELATED");
    ncmopi_ = mo_space_info_->get_dimension("CORRELATED");

    // obtain frozen orbitals
    frzcpi_ = mo_space_info_->get_dimension("FROZEN_DOCC");
    frzvpi_ = mo_space_info_->get_dimension("FROZEN_UOCC");
    nfrzc_ = mo_space_info_->size("FROZEN_DOCC");
    nfrzv_ = mo_space_info_->size("FROZEN_UOCC");

    // obtain active orbitals
    if(options_["ACTIVE"].size() == 0){
        outfile->Printf("\n  Please specify the ACTIVE occupations.");
        outfile->Printf("\n  Single-reference computations should set ACTIVE to zeros.");
        outfile->Printf("\n  For example, ACTIVE [0,0,0,0] depending on the symmetry. \n");
        throw PSIEXCEPTION("Please specify the ACTIVE occupations. Check output for details.");
    }
    active_ = mo_space_info_->get_dimension("ACTIVE");
    na_ = active_.sum();

    // obitan inactive orbitals
    core_ = mo_space_info_->get_dimension("RESTRICTED_DOCC");
    virtual_ = mo_space_info_->get_dimension("RESTRICTED_UOCC");
    nc_ = core_.sum();
    nv_ = virtual_.sum();

    // compute number of electrons
    int natom = molecule->natom();
    size_t nelec = 0;
    for(int i=0; i<natom; ++i){
        nelec += molecule->fZ(i);
    }
    int charge = molecule->molecular_charge();
    if(options_["CHARGE"].has_changed()){
        charge = options_.get_int("CHARGE");
    }
    nelec -= charge;
    multi_ = molecule->multiplicity();
    if(options_["MULTIPLICITY"].has_changed()){
        multi_ = options_.get_int("MULTIPLICITY");
    }
    if(multi_ < 1){
        outfile->Printf("\n  MULTIPLICITY must be no less than 1.");
        outfile->Printf("\n  MULTIPLICITY = %2d", multi_);
        outfile->Printf("\n  Check (specify) Multiplicity! \n");
        throw PSIEXCEPTION("MULTIPLICITY must be no less than 1. Check output for details.");
    }
    ms_ = options_.get_int("MS");
    if(ms_ < 0){
        outfile->Printf("\n  Ms must be no less than 0.");
        outfile->Printf("\n  Ms = %2d, MULTIPLICITY = %2d", ms_, multi_);
        outfile->Printf("\n  Check (specify) Ms value (component of multiplicity)! \n");
        throw PSIEXCEPTION("Ms must be no less than 0. Check output for details.");
    }
    nalfa_ = (nelec + ms_) / 2;
    nbeta_ = (nelec - ms_) / 2;
    if(nalfa_ < 0 || nbeta_ < 0 || (nalfa_ + nbeta_) != nelec){
        outfile->Printf("\n  Number of alpha electrons or beta electrons is negative.");
        outfile->Printf("\n  Nalpha = %5ld, Nbeta = %5ld", nalfa_, nbeta_);
        outfile->Printf("\n  Charge = %3d, Multiplicity = %3d, Ms = %.1f", charge, multi_, ms_);
        outfile->Printf("\n  Check the Charge, Multiplicity, and Ms! \n");
        outfile->Printf("\n  Note that Ms is 2 * Sz \n");
        throw PSIEXCEPTION("Negative number of alpha electrons or beta electrons. Check output for details.");
    }
    if(nalfa_ - nc_ - nfrzc_ > na_){
        outfile->Printf("\n  Not enough active orbitals to arrange electrons!");
        outfile->Printf("\n  Number of orbitals: active = %5zu, core = %5zu", na_, nc_);
        outfile->Printf("\n  Number of alpha electrons: Nalpha = %5ld", nalfa_);
        outfile->Printf("\n  Check core and active orbitals! \n");
        throw PSIEXCEPTION("Not enough active orbitals to arrange electrons! Check output for details.");
    }

    // obtain root symmetry
    root_sym_ = options_.get_int("ROOT_SYM");

    // obtain number of roots and roots of interest
    nroot_ = options_.get_int("NROOT");
    root_ = options_.get_int("ROOT");
    if(root_ >= nroot_){
        outfile->Printf("\n  NROOT = %3d, ROOT = %3d", nroot_, root_);
        outfile->Printf("\n  ROOT must be smaller than NROOT.");
        throw PSIEXCEPTION("ROOT must be smaller than NROOT.");
    }

    // setup symmetry index of active orbitals
    for(int h = 0; h < nirrep_; ++h){
        for(size_t i = 0; i < size_t(active_[h]); ++i){
            sym_active_.push_back(h);
        }
    }

    // setup symmetry index of correlated orbitals
    for(int h = 0; h < nirrep_; ++h){
        for(size_t i = 0; i < size_t(ncmopi_[h]); ++i){
            sym_ncmo_.push_back(h);
        }
    }

    // obtain absolute indices of core, active and virtual
    idx_c_ = mo_space_info_->get_corr_abs_mo("RESTRICTED_DOCC");
    idx_a_ = mo_space_info_->get_corr_abs_mo("ACTIVE");
    idx_v_ = mo_space_info_->get_corr_abs_mo("RESTRICTED_UOCC");

    // setup hole and particle indices (Active must start first for old mcsrgpt2 code)
    nh_ = nc_ + na_;
    npt_ = na_ + nv_;
    idx_h_ = vector<size_t> (idx_a_);
    idx_h_.insert(idx_h_.end(), idx_c_.begin(), idx_c_.end());
    idx_p_ = vector<size_t> (idx_a_);
    idx_p_.insert(idx_p_.end(), idx_v_.begin(), idx_v_.end());

    // print input summary
    std::vector<std::pair<std::string,size_t>> info;
    info.push_back({"number of atoms", natom});
    info.push_back({"number of electrons", nelec});
    info.push_back({"molecular charge", charge});
    info.push_back({"number of alpha electrons", nalfa_});
    info.push_back({"number of beta electrons", nbeta_});
    info.push_back({"multiplicity", multi_});
    info.push_back({"ms (2 * Sz)", ms_});
    info.push_back({"number of molecular orbitals", nmo_});

    if(print_ > 0){print_h2("Input Summary");}
    if(print_ > 0){
        for (auto& str_dim: info){
            outfile->Printf("\n    %-30s = %5zu",str_dim.first.c_str(),str_dim.second);
        }
    }

    // print orbital spaces
    if(print_ > 0){
        print_h2("Orbital Spaces");
        print_irrep("TOTAL MO", nmopi_);
        print_irrep("FROZEN CORE", frzcpi_);
        print_irrep("FROZEN VIRTUAL", frzvpi_);
        print_irrep("CORRELATED MO", ncmopi_);
        print_irrep("CORE", core_);
        print_irrep("ACTIVE", active_);
        print_irrep("VIRTUAL", virtual_);
    }

    // print orbital indices
    if(print_ > 0){
        print_h2("Correlated Subspace Indices");
        print_idx("CORE", idx_c_);
        print_idx("ACTIVE", idx_a_);
        print_idx("HOLE", idx_h_);
        print_idx("VIRTUAL", idx_v_);
        print_idx("PARTICLE", idx_p_);
        outfile->Printf("\n");
        outfile->Flush();
    }

    // state averaging
    if(options_["AVG_STATES"].has_changed()){

        CharacterTable ct = Process::environment.molecule()->point_group()->char_table();
        std::vector<std::string> irrep_symbol;
        for(int h = 0; h < nirrep_; ++h){
            irrep_symbol.push_back(std::string(ct.gamma(h).symbol()));
        }

        size_t nstates = 0;
        int nentry = options_["AVG_STATES"].size();

        // figure out total number of states
        std::vector<int> nstatespim;
        std::vector<int> irreps;
        std::vector<int> multis;
        for(int i = 0; i < nentry; ++i){
            if(options_["AVG_STATES"][i].size() != 3){
                outfile->Printf("\n  Error: invalid input of AVG_STATES. Each entry should take an array of three numbers.");
                throw PSIEXCEPTION("Invalid input of AVG_STATES");
            }

            // irrep
            int irrep = options_["AVG_STATES"][i][0].to_integer();
            if(irrep >= nirrep_ || irrep < 0){
                outfile->Printf("\n  Error: invalid irrep in AVG_STATES. Please check the input irrep (start from 0) not to exceed %d",
                                nirrep_ - 1);
                throw PSIEXCEPTION("Invalid irrep in AVG_STATES");
            }
            irreps.push_back(irrep);

            // multiplicity
            int multi = options_["AVG_STATES"][i][1].to_integer();
            if(multi < 1){
                outfile->Printf("\n  Error: invalid multiplicity in AVG_STATES.");
                throw PSIEXCEPTION("Invaid multiplicity in AVG_STATES");
            }
            multis.push_back(multi);

            // number of states of irrep and multiplicity
            int nstates_this = options_["AVG_STATES"][i][2].to_integer();
            if(nstates_this < 1){
                outfile->Printf("\n  Error: invalid nstates in AVG_STATES. nstates of a certain irrep and multiplicity should greater than 0.");
                throw PSIEXCEPTION("Invalid nstates in AVG_STATES.");
            }
            nstatespim.push_back(nstates_this);
            nstates += nstates_this;
        }

        // test input weights
        std::vector<std::vector<double>> weights;
        if(options_["AVG_WEIGHTS"].has_changed()){
            if(options_["AVG_WEIGHTS"].size() != nentry){
                outfile->Printf("\n  Error: mismatched number of entries in AVG_STATES (%d) and AVG_WEIGHTS (%d).",
                                nentry, options_["AVG_WEIGHTS"].size());
                throw PSIEXCEPTION("Mismatched number of entries in AVG_STATES and AVG_WEIGHTS.");
            }

            double wsum = 0.0;
            for(int i = 0; i < nentry; ++i){
                int nw = options_["AVG_WEIGHTS"][i].size();
                if(nw != nstatespim[i]){
                    outfile->Printf("\n  Error: mismatched number of weights in entry %d of AVG_WEIGHTS. Asked for %d states but only %d weights.",
                                    i, nstatespim[i], nw);
                    throw PSIEXCEPTION("Mismatched number of weights in AVG_WEIGHTS.");
                }

                std::vector<double> weight;
                for(int n = 0; n < nw; ++n){
                    double w = options_["AVG_WEIGHTS"][i][n].to_double();
                    if(w < 0.0){
                        outfile->Printf("\n  Error: negative weights in AVG_WEIGHTS.");
                        throw PSIEXCEPTION("Negative weights in AVG_WEIGHTS.");
                    }
                    weight.push_back(w);
                    wsum += w;
                }
                weights.push_back(weight);
            }
            if(fabs(wsum - 1.0) > 1.0e-10){
                outfile->Printf("\n  Error: AVG_WEIGHTS entries do not add up to 1.0. Sum = %.10f", wsum);
                throw PSIEXCEPTION("AVG_WEIGHTS entries do not add up to 1.0.");
            }

        } else {
            // use equal weights
            double w = 1.0 / nstates;
            for(int i = 0; i < nentry; ++i){
                std::vector<double> weight (nstatespim[i], w);
                weights.push_back(weight);
            }
        }

        // form option parser
        for(int i = 0; i < nentry; ++i){
            std::tuple<int, int, int, std::vector<double>> avg_info = std::make_tuple(irreps[i], multis[i], nstatespim[i], weights[i]);
            sa_info_.push_back(avg_info);
        }

        // printing summary
        print_h2("State Averaging Summary");
        int lweight = *std::max_element(nstatespim.begin(), nstatespim.end());
        if(lweight == 1){
            lweight = 7;
        }else{
            lweight *= 6;
            lweight -= 1;
        }
        int ltotal = 6 + 2 + 6 + 2 + 7 + 2 + lweight;
        std::string blank (lweight - 7, ' ');
        std::string dash (ltotal, '-');
        outfile->Printf("\n    Irrep.  Multi.  Nstates  %sWeights", blank.c_str());
        outfile->Printf("\n    %s", dash.c_str());
        for(int i = 0; i < nentry; ++i){
            std::string w_str;
            for(double w: weights[i]){
                std::stringstream ss;
                ss << std::fixed << std::setprecision(3) << w;
                w_str += ss.str() + " ";
            }
            w_str.pop_back(); // delete the last space character

            std::stringstream ss;
            ss << std::setw(4) << std::right << irrep_symbol[irreps[i]] << "    "
               << std::setw(4) << std::right << multis[i] << "    "
               << std::setw(5) << std::right << nstatespim[i] << "    "
               << std::setw(lweight) << w_str;
            outfile->Printf("\n    %s", ss.str().c_str());
        }
        outfile->Printf("\n    %s", dash.c_str());
        outfile->Printf("\n    Total number of states: %d", nstates);
        outfile->Printf("\n    %s\n", dash.c_str());
    }
}

double FCI_MO::compute_energy(){

    // allocate density
    Da_ = d2(ncmo_, d1(ncmo_));
    Db_ = d2(ncmo_, d1(ncmo_));
    L1a = ambit::Tensor::build(ambit::CoreTensor,"L1a",{na_, na_});
    L1b = ambit::Tensor::build(ambit::CoreTensor,"L1b",{na_, na_});

    // allocate Fock matrix
    Fa_ = d2(ncmo_, d1(ncmo_));
    Fb_ = d2(ncmo_, d1(ncmo_));

    // form determinants
    form_p_space();

    // diagonalize the CASCI Hamiltonian
    diag_algorithm_ = options_.get_str("DIAG_ALGORITHM");
    Diagonalize_H(determinant_, eigen_);
    if(print_ > 2 && !quiet_){
        for(pair<SharedVector, double> x: eigen_){
            outfile->Printf("\n\n  Spin selected CI vectors\n");
            (x.first)->print();
            outfile->Printf("  Energy  =  %20.15lf\n", x.second);
        }
    }

    // store CI vectors in eigen_
    if(nroot_ > eigen_.size()){
        outfile->Printf("\n  Too many roots of interest!");
        if(eigen_.size() > 1)
            outfile->Printf("\n  There are only %3d roots that satisfy the condition!", eigen_.size());
        else
            outfile->Printf("\n  There is only %3d root that satisfy the condition!", eigen_.size());
        outfile->Printf("\n  Check root_sym, multi, etc.");
        outfile->Printf("\n  If unrestricted orbitals are used, spin contamination may be severe (> 5%%).");
        throw PSIEXCEPTION("Too many roots of interest.");
    }
    print_CI(nroot_, options_.get_double("PRINT_CI_VECTOR"), eigen_, determinant_);

    // prepare ci_rdms for one density
    int dim = (eigen_[0].first)->dim();
    SharedMatrix evecs (new Matrix("evecs",dim,dim));
    for(int i = 0; i < eigen_.size(); ++i){
        evecs->set_column(0,i,(eigen_[i]).first);
    }
    CI_RDMS ci_rdms (options_,fci_ints_,determinant_,evecs,root_,root_);

    // form density
    FormDensity(ci_rdms, Da_, Db_);
    if(print_ > 1){
        print_d2("Da", Da_);
        print_d2("Db", Db_);
    }

    // Fock Matrix
    size_t count = 0;
    Form_Fock(Fa_,Fb_);
    Check_Fock(Fa_,Fb_,dconv_,count);
    if(print_ > 1){
        print_d2("Fa", Fa_);
        print_d2("Fb", Fb_);
    }

    // Orbitals. If use Kevin's CASSCF, this part is ignored.
    if(!default_orbitals_){
        if(semi_ && count != 0){
            // Semi-canonicalize orbitals
            outfile->Printf("\n  Use semi-canonical orbitals.\n");
            semi_canonicalize();

            // Form and Diagonalize the CASCI Hamiltonian
            Diagonalize_H(determinant_, eigen_);
            if(print_ > 2){
                for(pair<SharedVector, double> x: eigen_){
                    outfile->Printf("\n\n  Spin selected CI vectors\n");
                    (x.first)->print();
                    outfile->Printf("  Energy  =  %20.15lf\n", x.second);
                }
            }

            // Store CI Vectors in eigen_
            print_CI(nroot_, options_.get_double("PRINT_CI_VECTOR"), eigen_, determinant_);

            // prepare ci_rdms for one density
            int dim = (eigen_[0].first)->dim();
            SharedMatrix evecs (new Matrix("evecs",dim,dim));
            for(int i = 0; i < eigen_.size(); ++i){
                evecs->set_column(0,i,(eigen_[i]).first);
            }
            CI_RDMS ci_rdms (options_,fci_ints_,determinant_,evecs,root_,root_);

            // Form Density
            FormDensity(ci_rdms, Da_, Db_);
            if(print_ > 1){
                print_d2("Da", Da_);
                print_d2("Db", Db_);
            }

            // Fock Matrix
            count = 0;
            Form_Fock(Fa_,Fb_);
            Check_Fock(Fa_,Fb_,dconv_,count);
            if(print_ > 1){
                print_d2("Fa", Fa_);
                print_d2("Fb", Fb_);
            }
        }else{
//            nat_orbs();
        }
    }

    Eref_ = eigen_[root_].second;
    Process::environment.globals["CURRENT ENERGY"] = Eref_;
    return Eref_;
}

void FCI_MO::form_p_space(){
    // clean previous determinants
    determinant_.clear();

    // form determinants
    if(active_space_type_ == "COMPLETE" || active_space_type_ == "DOCI"){
        form_det();
    }else if(active_space_type_ == "CIS"){
        form_det_cis();
    }else if(active_space_type_ == "CISD"){
        form_det_cisd();
    }
}

void FCI_MO::form_det(){

    // Number of alpha and beta electrons in active
    int na_a = nalfa_ - nc_ - nfrzc_;
    int nb_a = nbeta_ - nc_ - nfrzc_;

    // Alpha and Beta Strings
    Timer tstrings;
    std::string str = "Forming alpha and beta strings";
    if(!quiet_){outfile->Printf("\n  %-35s ...", str.c_str());}
    vector<vector<vector<bool>>> a_string = Form_String(na_a);
    vector<vector<vector<bool>>> b_string = Form_String(nb_a);
    if(!quiet_){outfile->Printf("  Done. Timing %15.6f s", tstrings.get());}

    // Form Determinant
    Timer tdet;
    str = "Forming determinants";
    if(!quiet_){outfile->Printf("\n  %-35s ...", str.c_str());}
    if(options_.get_str("ACTIVE_SPACE_TYPE") == "DOCI"){
        if(root_sym_ != 0 || multi_ != 1){
            outfile->Printf("\n  State must be totally symmetric for DOCI.");
            throw PSIEXCEPTION("State must be totally symmetric for DOCI.");
        } else {
            for(int i = 0; i != nirrep_; ++i){
                size_t sa = a_string[i].size();
                for(size_t alfa = 0; alfa < sa; ++alfa){
                    determinant_.push_back(STLBitsetDeterminant(a_string[i][alfa], a_string[i][alfa]));
                }
            }
        }
    } else {
        for(int i = 0; i != nirrep_; ++i){
            int j = i ^ root_sym_;
            size_t sa = a_string[i].size();
            size_t sb = b_string[j].size();
            for(size_t alfa = 0; alfa < sa; ++alfa){
                for(size_t beta = 0; beta < sb; ++beta){
                    determinant_.push_back(STLBitsetDeterminant(a_string[i][alfa], b_string[j][beta]));
                }
            }
        }
    }
    if(!quiet_){outfile->Printf("  Done. Timing %15.6f s", tdet.get());}

    // printing
    std::vector<std::pair<std::string,size_t>> info;
    info.push_back({"number of alpha active electrons", na_a});
    info.push_back({"number of beta active electrons", nb_a});
    info.push_back({"root symmetry (zero based)", root_sym_});
    info.push_back({"number of determinants", determinant_.size()});

    print_h2("Determinants Summary");
    for (auto& str_dim: info){
        outfile->Printf("\n    %-35s = %5zu",str_dim.first.c_str(),str_dim.second);
    }
    if(print_ > 2) print_det(determinant_);
    if(!quiet_){outfile->Printf("\n");}
    outfile->Flush();

    if(determinant_.size() == 0){
        outfile->Printf("\n  There is no determinant matching the conditions!");
        outfile->Printf("\n  Check the wavefunction symmetry and multiplicity.");
        throw PSIEXCEPTION("No determinant matching the conditions!");
    }
}

vector<vector<vector<bool>>> FCI_MO::Form_String(const int& active_elec, const bool& print){

    timer_on("FORM String");
    vector<vector<vector<bool>>> String(nirrep_,vector<vector<bool>>());

    // initalize the string (only active)
    int symmetry = 0;
    bool *I_init = new bool[na_];
    for(size_t i = 0; i < na_; ++i) I_init[i] = 0;
    for(size_t i = na_ - active_elec; i < na_; ++i)  I_init[i] = 1;

    do{
        // permute the active
        vector<bool> string_a;
        int sym = symmetry;
        for(size_t i=0; i<na_; ++i){
            string_a.push_back(I_init[i]);
            if(I_init[i] == 1){
                sym ^= sym_active_[i];
            }
        }
        String[sym].push_back(string_a);
    }while(next_permutation(I_init, I_init + na_));

    if(print == true && !quiet_){
        print_h2("Possible String");
        for(size_t i = 0; i != String.size(); ++i){
            outfile->Printf("\n  symmetry = %lu \n", i);
            for(size_t j = 0; j != String[i].size(); ++j){
                outfile->Printf("    ");
                for(bool b: String[i][j]){
                    outfile->Printf("%d ", b);
                }
                outfile->Printf("\n");
            }
        }
    }

    delete[] I_init;
    timer_off("FORM String");
    return String;
}

void FCI_MO::form_det_cis(){
    // add close-shell ref
    vector<bool> string_ref = Form_String_Ref();
    if(root_sym_ == 0){
        determinant_.push_back(STLBitsetDeterminant(string_ref, string_ref));
    }

    // singles string
    // have to do full singles string because it sets up the ao_ and av_
    vector<vector<vector<bool>>> string_singles = Form_String_Singles(string_ref);
    if(ipea_ == "IP"){
        string_singles = Form_String_IP(string_ref);
    }
    if(ipea_ == "EA"){
        string_singles = Form_String_EA(string_ref);
    }

    // symmetry of ref (just active)
    int symmetry = 0;
    for(int i = 0; i < na_; ++i){
        if(string_ref[i]){
            symmetry ^= sym_active_[i];
        }
    }

    // singles
    Timer tdet;
    string str = "Forming determinants";
    if(!quiet_) {outfile->Printf("\n  %-35s ...", str.c_str());}

    int i = symmetry ^ root_sym_;
    size_t single_size = string_singles[i].size();
    for(size_t x = 0; x < single_size; ++x){
        determinant_.push_back(STLBitsetDeterminant(string_singles[i][x], string_ref));
        determinant_.push_back(STLBitsetDeterminant(string_ref, string_singles[i][x]));
    }

    if(!quiet_){outfile->Printf("  Done. Timing %15.6f s", tdet.get());}

    // Number of alpha and beta electrons in active
    int na_a = nalfa_ - nc_ - nfrzc_;
    int nb_a = nbeta_ - nc_ - nfrzc_;

    // printing
    std::vector<std::pair<std::string,size_t>> info;
    info.push_back({"number of alpha active electrons", na_a});
    info.push_back({"number of beta active electrons", nb_a});
    info.push_back({"root symmetry (zero based)", root_sym_});
    info.push_back({"number of determinants", determinant_.size()});

    if(!quiet_)
    {
        print_h2("Determinants Summary");
        for (auto& str_dim: info){
            outfile->Printf("\n    %-35s = %5zu",str_dim.first.c_str(),str_dim.second);
        }
        print_det(determinant_);
        outfile->Printf("\n");
        outfile->Flush();
    }

    if(determinant_.size() == 0){
        outfile->Printf("\n  There is no determinant matching the conditions!");
        outfile->Printf("\n  Check the wavefunction symmetry and multiplicity.");
        throw PSIEXCEPTION("No determinant matching the conditions!");
    }
}

void FCI_MO::form_det_cisd(){
    // add close-shell ref
    vector<bool> string_ref = Form_String_Ref();
    cisd_ex_no_hf_ = options_.get_bool("CISD_EX_NO_HF");
    if(root_sym_ == 0){
        determinant_.push_back(STLBitsetDeterminant(string_ref, string_ref));

        if(root_ != 0 && cisd_ex_no_hf_){
            determinant_.pop_back();
        }
    }

    // singles string
    vector<vector<vector<bool>>> string_singles = Form_String_Singles(string_ref);
    vector<vector<vector<bool>>> string_singles_ipea;
    if(ipea_ == "IP"){
        string_singles_ipea = Form_String_IP(string_ref);
    }
    if(ipea_ == "EA"){
        string_singles_ipea = Form_String_EA(string_ref);
    }

    // doubles string
    vector<vector<vector<bool>>> string_doubles = Form_String_Doubles(string_ref);

    // symmetry of ref (just active)
    int symmetry = 0;
    for(int i = 0; i < na_; ++i){
        if(string_ref[i]){
            symmetry ^= sym_active_[i];
        }
    }

    if(root_sym_ != 0 || root_ != 0 || !cisd_ex_no_hf_){
        if(cisd_ex_no_hf_ && root_sym_ == 0){
            root_ -= 1;
            nroot_ -= 1;
        }

        // singles
        Timer tdet;
        string str = "Forming determinants";
        if(!quiet_) {outfile->Printf("\n  %-35s ...", str.c_str());}

        int i = symmetry ^ root_sym_;
        if(ipea_ == "NONE"){
            size_t single_size = string_singles[i].size();
            for(size_t x = 0; x < single_size; ++x){
                determinant_.push_back(STLBitsetDeterminant(string_singles[i][x], string_ref));
                determinant_.push_back(STLBitsetDeterminant(string_ref, string_singles[i][x]));
            }
        } else {
            size_t single_size = string_singles_ipea[i].size();
            for(size_t x = 0; x < single_size; ++x){
                determinant_.push_back(STLBitsetDeterminant(string_singles_ipea[i][x], string_ref));
                determinant_.push_back(STLBitsetDeterminant(string_ref, string_singles_ipea[i][x]));
            }
        }

        // doubles
        size_t double_size = string_doubles[i].size();
        for(size_t x = 0; x < double_size; ++x){
            determinant_.push_back(STLBitsetDeterminant(string_doubles[i][x], string_ref));
            determinant_.push_back(STLBitsetDeterminant(string_ref, string_doubles[i][x]));
        }

        for(int h = 0; h < nirrep_; ++h){
            size_t single_size_a = string_singles[h].size();
            for(size_t x = 0; x < single_size_a; ++x){
                int sym = h ^ root_sym_;

                size_t single_size_b = string_singles[sym].size();
                for(size_t y = 0; y < single_size_b; ++y){
                    determinant_.push_back(STLBitsetDeterminant(string_singles[h][x], string_singles[sym][y]));
                }

                if(ipea_ != "NONE"){
                    size_t single_ipea_size_b = string_singles_ipea[sym].size();
                    for(size_t y = 0; y < single_ipea_size_b; ++y){
                        determinant_.push_back(STLBitsetDeterminant(string_singles[h][x], string_singles_ipea[sym][y]));
                        determinant_.push_back(STLBitsetDeterminant(string_singles_ipea[sym][y], string_singles[h][x]));
                    }
                }
            }
        }

        if(!quiet_){outfile->Printf("  Done. Timing %15.6f s", tdet.get());}
    }

    // Number of alpha and beta electrons in active
    int na_a = nalfa_ - nc_ - nfrzc_;
    int nb_a = nbeta_ - nc_ - nfrzc_;

    // printing
    std::vector<std::pair<std::string,size_t>> info;
    info.push_back({"number of alpha active electrons", na_a});
    info.push_back({"number of beta active electrons", nb_a});
    info.push_back({"root symmetry (zero based)", root_sym_});
    info.push_back({"number of determinants", determinant_.size()});

    if(!quiet_)
    {
        print_h2("Determinants Summary");
        for (auto& str_dim: info){
            outfile->Printf("\n    %-35s = %5zu",str_dim.first.c_str(),str_dim.second);
        }
        print_det(determinant_);
        outfile->Printf("\n");
        outfile->Flush();
    }

    if(determinant_.size() == 0){
        outfile->Printf("\n  There is no determinant matching the conditions!");
        outfile->Printf("\n  Check the wavefunction symmetry and multiplicity.");
        throw PSIEXCEPTION("No determinant matching the conditions!");
    }
}

vector<bool> FCI_MO::Form_String_Ref(const bool &print){
    timer_on("FORM String Ref");

    vector<bool> String;
    Dimension doccpi(this->doccpi());
    if(ipea_ == "EA"){
        doccpi[0] += 1;
    }
    for(int h = 0; h < nirrep_; ++h){
        int act_docc = doccpi[h] - frzcpi_[h] - core_[h];
        int act = active_[h];
        for(int i = 0; i < act; ++i){
            String.push_back(i < act_docc);
        }
    }
    active_o_ = doccpi - frzcpi_ - core_;
    active_v_ = active_ - active_o_;

    ao_.clear();
    av_.clear();
    for(int h = 0; h < nirrep_; ++h){
        int h_local = h;
        size_t offset = 0;
        while(--h_local >= 0){
            offset += active_[h_local];
        }

        for(size_t i = 0; i < active_[h]; ++i){
            if(i < active_o_[h]){
                ao_.push_back(i + offset);
            } else {
                av_.push_back(i + offset);
            }
        }
    }

    if(print){
        print_h2("Reference String");
        outfile->Printf("    ");
        for(bool b: String){
            outfile->Printf("%d ", b);
        }
    }

    timer_off("FORM String Ref");
    return String;
}

vector<vector<vector<bool>>> FCI_MO::Form_String_Singles(const vector<bool> &ref_string, const bool &print){
    timer_on("FORM String Singles");
    vector<vector<vector<bool>>> String(nirrep_,vector<vector<bool>>());

    // occupied and unoccupied indices, symmetry (active)
    int symmetry = 0;
    vector<int> uocc, occ;
    for(size_t i = 0; i < na_; ++i){
        if(ipea_ != "NONE" &&
                std::find(diffused_orbs_.begin(), diffused_orbs_.end(), i) != diffused_orbs_.end()){
            continue;
        }

        if(ref_string[i]){
            occ.push_back(i);
            symmetry ^= sym_active_[i];
        }else{
            uocc.push_back(i);
        }
    }

    // singles
    for(const int& a: uocc){
        vector<bool> string_local(ref_string);
        string_local[a] = true;
        int sym = symmetry ^ sym_active_[a];
        for(const int& i: occ){
            string_local[i] = false;
            sym ^= sym_active_[i];
            String[sym].push_back(string_local);
            // need to reset
            string_local[i] = true;
            sym ^= sym_active_[i];
        }
    }

    if(print){
        print_h2("Singles String");
        for(size_t i = 0; i != String.size(); ++i){
            if(String[i].size() != 0){
                outfile->Printf("\n  symmetry = %lu \n", i);
            }
            for(size_t j = 0; j != String[i].size(); ++j){
                outfile->Printf("    ");
                for(bool b: String[i][j]){
                    outfile->Printf("%d ", b);
                }
                outfile->Printf("\n");
            }
        }
    }

    timer_off("FORM String Singles");
    return String;
}

vector<vector<vector<bool>>> FCI_MO::Form_String_IP(const vector<bool> &ref_string, const bool &print){
    timer_on("FORM String Singles IP");
    vector<vector<vector<bool>>> String(nirrep_,vector<vector<bool>>());

    // occupied and unoccupied indices, symmetry (active)
    int symmetry = 0;
    vector<int> occ;
    for(int i = 0; i < na_; ++i){
        if(ref_string[i]){
            occ.push_back(i);
            symmetry ^= sym_active_[i];
        }
    }

    // singles
    for(const int& i: occ){
        vector<bool> string_local(ref_string);
        string_local[idx_diffused_] = true;

        string_local[i] = false;
        int sym = symmetry ^ sym_active_[i];
        String[sym].push_back(string_local);
    }

    if(print){
        print_h2("Singles String IP");
        for(size_t i = 0; i != String.size(); ++i){
            if(String[i].size() != 0){
                outfile->Printf("\n  symmetry = %lu \n", i);
            }
            for(size_t j = 0; j != String[i].size(); ++j){
                outfile->Printf("    ");
                for(bool b: String[i][j]){
                    outfile->Printf("%d ", b);
                }
                outfile->Printf("\n");
            }
        }
    }

    timer_off("FORM String Singles IP");
    return String;
}

vector<vector<vector<bool>>> FCI_MO::Form_String_EA(const vector<bool> &ref_string, const bool &print){
    timer_on("FORM String Singles EA");
    vector<vector<vector<bool>>> String(nirrep_,vector<vector<bool>>());

    // occupied and unoccupied indices, symmetry (active)
    int symmetry = 0;
    vector<int> uocc;
    for(int i = 0; i < na_; ++i){
        if(!ref_string[i]){
            uocc.push_back(i);
        } else {
            symmetry ^= sym_active_[i];
        }
    }

    // singles
    for(const int& a: uocc){
        vector<bool> string_local(ref_string);
        string_local[a] = true;
        int sym = symmetry ^ sym_active_[a];

        string_local[idx_diffused_] = false;
        String[sym].push_back(string_local);
    }

    if(print){
        print_h2("Singles String EA");
        for(size_t i = 0; i != String.size(); ++i){
            if(String[i].size() != 0){
                outfile->Printf("\n  symmetry = %lu \n", i);
            }
            for(size_t j = 0; j != String[i].size(); ++j){
                outfile->Printf("    ");
                for(bool b: String[i][j]){
                    outfile->Printf("%d ", b);
                }
                outfile->Printf("\n");
            }
        }
    }

    timer_off("FORM String Singles EA");
    return String;
}

vector<vector<vector<bool>>> FCI_MO::Form_String_Doubles(const vector<bool> &ref_string, const bool &print){
    timer_on("FORM String Doubles");
    vector<vector<vector<bool>>> String(nirrep_,vector<vector<bool>>());

    // occupied and unoccupied indices, symmetry (active)
    int symmetry = 0;
    vector<int> uocc, occ;
    for(int i = 0; i < na_; ++i){
        if(ipea_ != "NONE" && i != idx_diffused_ &&
                std::find(diffused_orbs_.begin(), diffused_orbs_.end(), i) != diffused_orbs_.end()){
            continue;
        }

        if(ref_string[i]){
            occ.push_back(i);
            symmetry ^= sym_active_[i];
        }else{
            uocc.push_back(i);
        }
    }

    // doubles
    for(const int& a: uocc){
        vector<bool> string_a(ref_string);
        string_a[a] = true;
        int sym_a = symmetry ^ sym_active_[a];

        for(const int& b: uocc){
            if(b > a){
                vector<bool> string_b(string_a);
                string_b[b] = true;
                int sym_b = sym_a ^ sym_active_[b];

                for(const int& i: occ){
                    vector<bool> string_i(string_b);
                    string_i[i] = false;
                    int sym_i = sym_b ^ sym_active_[i];

                    for(const int& j: occ){
                        if(j > i){
                            vector<bool> string_j(string_i);
                            string_j[j] = false;
                            int sym_j = sym_i ^ sym_active_[j];
                            String[sym_j].push_back(string_j);
                        }
                    }
                }
            }
        }
    }

    if(print){
        print_h2("Doubles String");
        for(size_t i = 0; i != String.size(); ++i){
            if(String[i].size() != 0){
                outfile->Printf("\n  symmetry = %lu \n", i);
            }
            for(size_t j = 0; j != String[i].size(); ++j){
                outfile->Printf("    ");
                for(bool b: String[i][j]){
                    outfile->Printf("%d ", b);
                }
                outfile->Printf("\n");
            }
        }
    }

    timer_off("FORM String Doubles");
    return String;
}

void FCI_MO::semi_canonicalize(){
    SharedMatrix Ua (new Matrix("Unitary A", nmopi_, nmopi_));
    SharedMatrix Ub (new Matrix("Unitary B", nmopi_, nmopi_));
    BD_Fock(Fa_,Fb_,Ua,Ub,"Fock");
    SharedMatrix Ca = this->Ca();
    SharedMatrix Cb = this->Cb();
    SharedMatrix Ca_new(Ca->clone());
    SharedMatrix Cb_new(Cb->clone());
    Ca_new->gemm(false,false,1.0,Ca,Ua,0.0);
    Cb_new->gemm(false,false,1.0,Cb,Ub,0.0);

    // overlap of original and semicanonical orbitals
    SharedMatrix MOoverlap = Matrix::triplet(Ca,this->S(),Ca_new,true,false,false);
    MOoverlap->set_name("MO overlap");

    // copy semicanonical orbital to wavefunction
    Ca->copy(Ca_new);
    Cb->copy(Cb_new);

    // test active orbital ordering
    for(int h = 0; h < nirrep_; ++h){
        int actv_start = frzcpi_[h] + core_[h];
        int actv_end   = actv_start + active_[h];

        std::map<int, int> indexmap;
        std::vector<int> idx_0;
        for(int i = actv_start; i < actv_end; ++i){
            int ii = 0; // corresponding index in semicanonical basis
            double smax = 0.0;

            for(int j = actv_start; j < actv_end; ++j){
                double s = MOoverlap->get(h,i,j);
                if(fabs(s) > smax){
                    smax = fabs(s);
                    ii = j;
                }
            }

            if(ii != i){
                indexmap[i] = ii;
                idx_0.push_back(i);
                outfile->Printf("\n  i = %3d, ii = %3d, smax = %.15f", i, ii, smax);
            }
        }

        // find orbitals to swap if the loop is closed
        std::vector<int> idx_swap;
        for(const int& x: idx_0){
            // if index x is already in the to-be-swapped index, then continue
            if(std::find(idx_swap.begin(), idx_swap.end(), x) != idx_swap.end()){
                continue;
            }

            std::vector<int> temp;
            int local = x;

            while(indexmap.find(indexmap[local]) != indexmap.end()){
                if(std::find(temp.begin(), temp.end(), local) == temp.end()){
                    temp.push_back(local);
                } else {
                    // a loop found
                    break;
                }

                local = indexmap[local];
            }

            // start from the point that has the value of "local" and copy to idx_swap
            int pos = std::find(temp.begin(), temp.end(), local) - temp.begin();
            for(int i = pos; i < temp.size(); ++i){
                if(std::find(idx_swap.begin(), idx_swap.end(), temp[i]) == idx_swap.end()){
                    idx_swap.push_back(temp[i]);
                }
            }
        }

        // remove the swapped orbitals from the vector of orginal orbitals
        idx_0.erase(std::remove_if(idx_0.begin(), idx_0.end(),
                                   [&](int i) {return std::find(idx_swap.begin(), idx_swap.end(), i) != idx_swap.end();}),
                    idx_0.end());

        // swap orbitals
        for(const int& x: idx_swap){
            int h_local = h;
            size_t ni = x - frzcpi_[h];
            size_t nj = indexmap[x] - frzcpi_[h];
            while((--h_local) >= 0){
                ni += ncmopi_[h_local];
                nj += ncmopi_[h_local];
            }
            outfile->Printf("\n  Orbital ordering changed due to semicanonicalization. Swapped orbital %3zu back to %3zu.", nj, ni);
            Ca->set_column(h, x, Ca_new->get_column(h, indexmap[x]));
            Cb->set_column(h, x, Cb_new->get_column(h, indexmap[x]));
        }

        // throw warnings when inconsistency is detected
        for(const int& x: idx_0){
            int h_local = h;
            size_t ni = x - frzcpi_[h];
            size_t nj = indexmap[x] - frzcpi_[h];
            while((--h_local) >= 0){
                ni += ncmopi_[h_local];
                nj += ncmopi_[h_local];
            }
            outfile->Printf("\n  Orbital %3zu may have changed to semicanonical orbital %3zu. Please interpret orbitals with caution.", ni, nj);
        }
    }

//    // test integral transformation
//    ambit::Tensor U = ambit::Tensor::build(ambit::CoreTensor,"Orbital rotation",{ncmo_,ncmo_});
//    ambit::Tensor H = ambit::Tensor::build(ambit::CoreTensor,"OEI",{ncmo_,ncmo_});
//    ambit::Tensor F = ambit::Tensor::build(ambit::CoreTensor,"Fock",{ncmo_,ncmo_});
//    ambit::Tensor Da = ambit::Tensor::build(ambit::CoreTensor,"Da",{ncmo_,ncmo_});
//    ambit::Tensor Db = ambit::Tensor::build(ambit::CoreTensor,"Db",{ncmo_,ncmo_});
//    ambit::Tensor Vaa = ambit::Tensor::build(ambit::CoreTensor,"Vaa",{ncmo_,ncmo_,ncmo_,ncmo_});
//    ambit::Tensor Vab = ambit::Tensor::build(ambit::CoreTensor,"Vab",{ncmo_,ncmo_,ncmo_,ncmo_});
//    ambit::Tensor H_trans = ambit::Tensor::build(ambit::CoreTensor,"OEI transformed by integral class",{ncmo_,ncmo_});
//    ambit::Tensor Vaa_trans = ambit::Tensor::build(ambit::CoreTensor,"Vaa transformed by integral class",{ncmo_,ncmo_,ncmo_,ncmo_});
//    ambit::Tensor Vab_trans = ambit::Tensor::build(ambit::CoreTensor,"Vab transformed by integral class",{ncmo_,ncmo_,ncmo_,ncmo_});

//    Vaa.iterate([&](const std::vector<size_t>& i,double& value){
//        value = integral_->aptei_aa(i[0],i[1],i[2],i[3]);
//    });
//    Vab.iterate([&](const std::vector<size_t>& i,double& value){
//        value = integral_->aptei_ab(i[0],i[1],i[2],i[3]);
//    });
//    H.iterate([&](const std::vector<size_t>& i,double& value){
//        value = integral_->oei_a(i[0],i[1]);
//    });
////    H.print();

//    Da.iterate([&](const std::vector<size_t>& i,double& value){
//        value = Da_[i[0]][i[1]];
//    });
//    Db.iterate([&](const std::vector<size_t>& i,double& value){
//        value = Db_[i[0]][i[1]];
//    });


//    for(int h = 0; h < nirrep_; ++h){
//        int h_local = h;
//        size_t offset = 0;
//        while(--h_local >= 0){
//            offset += ncmopi_[h_local];
//        }

//        for(size_t p = 0; p < ncmopi_[h]; ++p){
//            size_t np = p + offset;
//            for(size_t q = 0; q < ncmopi_[h]; ++q){
//                size_t nq = q + offset;

//                size_t idx = np * ncmo_ + nq;
//                U.data()[idx] = Ua->get(h,p+frzcpi_[h],q+frzcpi_[h]);
//            }
//        }
//    }
//    U.print();

//    ambit::Tensor tempH = ambit::Tensor::build(ambit::CoreTensor,"H temp",{ncmo_,ncmo_});
//    ambit::Tensor tempDa = ambit::Tensor::build(ambit::CoreTensor,"Da temp",{ncmo_,ncmo_});
//    ambit::Tensor tempVaa = ambit::Tensor::build(ambit::CoreTensor,"Vaa temp",{ncmo_,ncmo_,ncmo_,ncmo_});
//    ambit::Tensor tempVab = ambit::Tensor::build(ambit::CoreTensor,"Vab temp",{ncmo_,ncmo_,ncmo_,ncmo_});
//    tempVaa("cdkl") = U("ac") * U("bd") * Vaa("abij") * U("jl") * U("ik");
//    tempVab("cdkl") = U("ac") * U("bd") * Vab("abij") * U("jl") * U("ik");
//    tempH("rs") = U("pr") * H("pq") * U("qs");
//    tempDa("rs") = U("pr") * Da("pq") * U("qs");
////    tempDa.print();
////    tempVaa.print();

//    F("pq")  = tempH("pq");
//    F("pq") += U("xp") * Vaa("xrys") * U("yq") * Da("sr");
//    F("pq") += U("xp") * Vab("xrys") * U("yq") * Db("sr");
////    F.print();

    outfile->Printf("\n\n");
    integral_->retransform_integrals();
    ambit::Tensor tei_active_aa = integral_->aptei_aa_block(idx_a_, idx_a_, idx_a_, idx_a_);
    ambit::Tensor tei_active_ab = integral_->aptei_ab_block(idx_a_, idx_a_, idx_a_, idx_a_);
    ambit::Tensor tei_active_bb = integral_->aptei_bb_block(idx_a_, idx_a_, idx_a_, idx_a_);
    fci_ints_->set_active_integrals(tei_active_aa, tei_active_ab, tei_active_bb);
    fci_ints_->compute_restricted_one_body_operator();

//    Vaa_trans.iterate([&](const std::vector<size_t>& i,double& value){
//        value = integral_->aptei_aa(i[0],i[1],i[2],i[3]);
//    });
//    Vab_trans.iterate([&](const std::vector<size_t>& i,double& value){
//        value = integral_->aptei_ab(i[0],i[1],i[2],i[3]);
//    });
//    H_trans.iterate([&](const std::vector<size_t>& i,double& value){
//        value = integral_->oei_a(i[0],i[1]);
//    });
////    Vab_trans.print();

//    size_t ncmo2 = ncmo_ * ncmo_;
//    size_t ncmo3 = ncmo_ * ncmo2;
//    for(size_t p = 0; p < ncmo_; ++p){
//        for(size_t q = 0; q < ncmo_; ++q){

//            double v1 = tempH.data()[p * ncmo_ + q];
//            double v2 = H_trans.data()[p * ncmo_ + q];
////            double diff = fabs(v1) - fabs(v2);
//            double diff = v1 - v2;
//            if(fabs(diff) > 1.0e-12){
//                outfile->Printf("\n  diff of H[%3zu][%3zu] = %20.15f", p, q, diff);
//            }

//            for(size_t r = 0; r < ncmo_; ++r){
//                for(size_t s = 0; s < ncmo_; ++s){
//                    double v3 = tempVaa.data()[p * ncmo3 + q * ncmo2 + r * ncmo_ + s];
//                    double v4 = Vaa_trans.data()[p * ncmo3 + q * ncmo2 + r * ncmo_ + s];
//                    double diff = fabs(v3) - fabs(v4);
//                    if(fabs(diff) > 1.0e-12){
//                        outfile->Printf("\n  diff of V[%3zu][%3zu][%3zu][%3zu] = %20.15f", p, q, r, s, diff);
//                    }
//                }
//            }
//        }
//    }
}

void FCI_MO::Diagonalize_H(const vecdet &det, vector<pair<SharedVector, double>> &eigen){
    timer_on("Diagonalize H");
    Timer tdiagH;
    std::string str = "Diagonalizing Hamiltonian";
    if(!quiet_){outfile->Printf("\n  %-35s ...", str.c_str());}
    size_t det_size = det.size();
    eigen.clear();

    STLBitsetDeterminant::set_ints(fci_ints_);
    std::vector<STLBitsetDeterminant> P_space;
    for(size_t x = 0; x != det_size; ++x){
        std::vector<bool> alfa_bits = det[x].get_alfa_bits_vector_bool();
        std::vector<bool> beta_bits = det[x].get_beta_bits_vector_bool();
        STLBitsetDeterminant bs_det(alfa_bits,beta_bits);
        P_space.push_back(bs_det);
    }

    SparseCISolver sparse_solver;
    int nroot = det_size < 5 * nroot_ ? det_size : 5 * nroot_;
    SharedMatrix vec_tmp;
    SharedVector val_tmp;
    DiagonalizationMethod diag_method = DLSolver;
    if(diag_algorithm_ == "FULL"){
        diag_method = Full;
    }
    if(ipea_ != "NONE"){
//        sparse_solver.set_print_details(true);
        sparse_solver.set_force_diag_method(true);
    }
    sparse_solver.diagonalize_hamiltonian(P_space,val_tmp,vec_tmp,nroot,multi_,diag_method);

    // add doubly occupied energy
    double vdocc = fci_ints_->scalar_energy();
    for(int i = 0; i != nroot; ++i){
        double value = val_tmp->get(i);
        val_tmp->set(i, value + vdocc);
    }

    // check spin
    int count = 0;
    if(!quiet_){outfile->Printf("\n\n  Reference type: %s", ref_type_.c_str());}
    double threshold = 0.1;
    if(ref_type_ == "UHF" || ref_type_ == "UKS" || ref_type_ == "CUHF"){
        threshold = 0.20 * multi_;    // 20% off from the multiplicity of the spin eigen state
    }
    if(!quiet_){outfile->Printf("\n  Threshold for spin check: %.2f", threshold);}

    for(int i = 0; i != nroot; ++i){
        double S2 = 0.0;
        for (int I = 0; I < det_size; ++I){
            for (int J = 0; J < det_size; ++J){
                double S2IJ = P_space[I].spin2(P_space[J]);
                S2 += S2IJ * vec_tmp->get(I,i) * vec_tmp->get(J,i);
            }
        }
        double S = std::fabs(0.5 * (std::sqrt(1.0 + 4.0 * S2) - 1.0));
        double multi_real = 2.0 * S + 1;

        if(std::fabs(multi_ - multi_real) > threshold){
            if(!quiet_){outfile->Printf("\n\n  Ask for S^2 = %.4f, this S^2 = %.4f, continue searching...", 0.25 * (multi_ * multi_ - 1.0), S2);}
            continue;
        }
        else{
            std::vector<string> s2_labels({"singlet","doublet","triplet","quartet","quintet","sextet","septet","octet","nonet","decaet"});
            std::string state_label = s2_labels[std::round(S * 2.0)];
            if(!quiet_){outfile->Printf("\n\n  Spin State: S^2 = %5.3f, S = %5.3f, %s (from %zu determinants)",S2,S,state_label.c_str(),det_size);}
            ++count;
            eigen.push_back(make_pair(vec_tmp->get_column(0,i), val_tmp->get(i) + e_nuc_));
        }
        if(count == nroot_) break;
    }
    if(!quiet_){outfile->Printf("  Done. Timing %15.6f s", tdiagH.get());}
    timer_off("Diagonalize H");
}

inline bool ReverseAbsSort(const tuple<double, int> &lhs, const tuple<double, int> &rhs){
    return abs(get<0>(rhs)) < abs(get<0>(lhs));
}

void FCI_MO::print_CI(const int &nroot, const double &CI_threshold, const vector<pair<SharedVector, double>> &eigen, const vecdet &det){
    timer_on("Print CI Vectors");
    if(!quiet_){
        outfile->Printf("\n\n  * * * * * * * * * * * * * * * * *");
        outfile->Printf("\n  *  CI Vectors & Configurations  *");
        outfile->Printf("\n  * * * * * * * * * * * * * * * * *");
        outfile->Printf("\n");
        outfile->Flush();
    }

    dominant_dets_.clear();
    for(int i = 0; i != nroot; ++i){
        vector<tuple<double, int>> ci_selec; // tuple<coeff, index>

        // choose CI coefficients greater than CI_threshold
        for(size_t j = 0; j < det.size(); ++j){
            double value = (eigen[i].first)->get(j);
            if(std::fabs(value) > CI_threshold)
                ci_selec.push_back(make_tuple(value, j));
        }
        sort(ci_selec.begin(), ci_selec.end(), ReverseAbsSort);
        dominant_dets_.push_back(det[get<1>(ci_selec[0])]);

        if(!quiet_){outfile->Printf("\n  ==> Root No. %d <==\n", i);}
        for(size_t j = 0; j < ci_selec.size(); ++j){
            if(!quiet_){outfile->Printf("\n    ");}
            double ci = get<0>(ci_selec[j]);
            size_t index = get<1>(ci_selec[j]);
            size_t ncmopi = 0;
            for(int h = 0; h < nirrep_; ++h){
                for(size_t k = 0; k < active_[h]; ++k){
                    size_t x = k + ncmopi;
                    bool a = det[index].get_alfa_bit(x);
                    bool b = det[index].get_beta_bit(x);
                    if(a == b){
                        if(!quiet_){outfile->Printf("%d", a==1 ? 2 : 0);}
                    } else {
                        if(!quiet_){outfile->Printf("%c", a==1 ? 'a' : 'b');}
                    }
                }
                if(active_[h] != 0)
                    if(!quiet_){outfile->Printf(" ");}
                ncmopi += active_[h];
            }
            if(!quiet_){outfile->Printf(" %20.10f", ci);}
        }
        if(!quiet_){outfile->Printf("\n\n    Total Energy:   %.15lf\n\n", eigen[i].second);}
        outfile->Flush();
    }

    timer_off("Print CI Vectors");
}

void FCI_MO::FormDensity(CI_RDMS &ci_rdms, d2 &A, d2 &B){
    timer_on("FORM Density");
    Timer tdensity;
    std::string str = "Forming one-particle density";
    if(!quiet_){outfile->Printf("\n  %-35s ...", str.c_str());}

    for(size_t p = 0; p < nc_; ++p){
        size_t np = idx_c_[p];
            A[np][np] = 1.0;
            B[np][np] = 1.0;
    }

    size_t dim = na_ * na_;
    vector<double> opdm_a (dim, 0.0);
    vector<double> opdm_b (dim, 0.0);

    ci_rdms.compute_1rdm(opdm_a,opdm_b);

    for(size_t p = 0; p < na_; ++p){
        size_t np = idx_a_[p];
        for(size_t q = p; q < na_; ++q){
            size_t nq = idx_a_[q];

            if((sym_active_[p] ^ sym_active_[q]) != 0) continue;

            size_t index = p * na_ + q;
            A[np][nq] = opdm_a[index];
            B[np][nq] = opdm_b[index];

            A[nq][np] = A[np][nq];
            B[nq][np] = B[np][nq];
        }
    }

    fill_density();
    if(!quiet_){outfile->Printf("  Done. Timing %15.6f s", tdensity.get());}
    timer_off("FORM Density");
}

double FCI_MO::OneOP(const STLBitsetDeterminant &J, STLBitsetDeterminant &Jnew, const size_t &p, const bool &sp, const size_t &q, const bool &sq){
    timer_on("1PO");
    vector<vector<bool>> tmp;
    tmp.push_back(J.get_alfa_bits_vector_bool());
    tmp.push_back(J.get_beta_bits_vector_bool());

    double sign = 1.0;

    if(tmp[sq][q]){
        sign *= CheckSign(tmp[sq],q);
        tmp[sq][q] = 0;
    }else{timer_off("1PO"); return 0.0;}

    if(!tmp[sp][p]){
        sign *= CheckSign(tmp[sp],p);
        tmp[sp][p] = 1;
        Jnew = STLBitsetDeterminant (tmp[0],tmp[1]);
        timer_off("1PO");
        return sign;
    }else{timer_off("1PO"); return 0.0;}
}

void FCI_MO::print_d2(const string &str, const d2 &OnePD){
    timer_on("PRINT Density");
    SharedMatrix M (new Matrix(str.c_str(), OnePD.size(), OnePD[0].size()));
    for(size_t i = 0; i != OnePD.size(); ++i){
        for(size_t j = 0; j != OnePD[i].size(); ++j){
            M->pointer()[i][j] = OnePD[i][j];
        }
    }
    M->print();
    timer_off("PRINT Density");
}

void FCI_MO::FormCumulant2(CI_RDMS &ci_rdms, d4 &AA, d4 &AB, d4 &BB){
    timer_on("FORM 2-Cumulant");
    Timer tL2;
    std::string str = "Forming Lambda2";
    if(!quiet_){outfile->Printf("\n  %-35s ...", str.c_str());}

    size_t dim = na_ * na_ * na_ * na_;
    vector<double> tpdm_aa (dim, 0.0);
    vector<double> tpdm_ab (dim, 0.0);
    vector<double> tpdm_bb (dim, 0.0);

    ci_rdms.compute_2rdm(tpdm_aa,tpdm_ab,tpdm_bb);

    FormCumulant2AA(tpdm_aa, tpdm_bb, AA, BB);
    FormCumulant2AB(tpdm_ab, AB);
    fill_cumulant2();

    outfile->Printf("  Done. Timing %15.6f s", tL2.get());
    timer_off("FORM 2-Cumulant");
}

void FCI_MO::FormCumulant2AA(const vector<double> &tpdm_aa, const vector<double> &tpdm_bb, d4 &AA, d4 &BB){
    size_t dim2 = na_ * na_;
    size_t dim3 = na_ * dim2;

    for(size_t p = 0; p < na_; ++p){
        size_t np = idx_a_[p];
        for(size_t q = p + 1; q < na_; ++q){
            size_t nq = idx_a_[q];
            for(size_t r = 0; r < na_; ++r){
                size_t nr = idx_a_[r];
                for(size_t s = r + 1; s < na_; ++s){
                    size_t ns = idx_a_[s];

                    if((sym_active_[p] ^ sym_active_[q] ^ sym_active_[r] ^ sym_active_[s]) != 0) continue;

                    size_t index = p * dim3 + q * dim2 + r * na_ + s;

                    AA[p][q][r][s] += tpdm_aa[index];
                    BB[p][q][r][s] += tpdm_bb[index];

                    AA[p][q][r][s] -= Da_[np][nr] * Da_[nq][ns];
                    AA[p][q][r][s] += Da_[np][ns] * Da_[nq][nr];
                    AA[p][q][s][r] -= AA[p][q][r][s];
                    AA[q][p][r][s] -= AA[p][q][r][s];
                    AA[q][p][s][r] += AA[p][q][r][s];

                    BB[p][q][r][s] -= Db_[np][nr] * Db_[nq][ns];
                    BB[p][q][r][s] += Db_[np][ns] * Db_[nq][nr];
                    BB[p][q][s][r] -= BB[p][q][r][s];
                    BB[q][p][r][s] -= BB[p][q][r][s];
                    BB[q][p][s][r] += BB[p][q][r][s];
                }
            }
        }
    }
}

void FCI_MO::FormCumulant2AB(const vector<double> &tpdm_ab, d4 &AB){
    size_t dim2 = na_ * na_;
    size_t dim3 = na_ * dim2;

    for(size_t p = 0; p < na_; ++p){
        size_t np = idx_a_[p];
        for(size_t q = 0; q < na_; ++q){
            size_t nq = idx_a_[q];
            for(size_t r = 0; r < na_; ++r){
                size_t nr = idx_a_[r];
                for(size_t s = 0; s < na_; ++s){
                    size_t ns = idx_a_[s];

                    if((sym_active_[p] ^ sym_active_[q] ^ sym_active_[r] ^ sym_active_[s]) != 0) continue;

                    size_t index = p * dim3 + q * dim2 + r * na_ + s;
                    AB[p][q][r][s] += tpdm_ab[index];

                    AB[p][q][r][s] -= Da_[np][nr] * Db_[nq][ns];
                }
            }
        }
    }
}

void FCI_MO::print2PDC(const string &str, const d4 &TwoPDC, const int &PRINT){
    timer_on("PRINT 2-Cumulant");
    outfile->Printf("\n  ** %s **", str.c_str());
    size_t count = 0;
    for(size_t i = 0; i != TwoPDC.size(); ++i){
        for(size_t j = 0; j != TwoPDC[i].size(); ++j){
            for(size_t k = 0; k != TwoPDC[i][j].size(); ++k){
                for(size_t l = 0; l != TwoPDC[i][j][k].size(); ++l){
                    if(fabs(TwoPDC[i][j][k][l]) > 1.0e-15){
                        ++count;
                        if(PRINT > 2)
                            outfile->Printf("\n  Lambda [%3lu][%3lu][%3lu][%3lu] = %18.15lf", i, j, k, l, TwoPDC[i][j][k][l]);
                    }
                }
            }
        }
    }
    outfile->Printf("\n");
    outfile->Printf("\n  Number of Nonzero Elements: %zu", count);
    outfile->Printf("\n");
    timer_off("PRINT 2-Cumulant");
}

double FCI_MO::TwoOP(const STLBitsetDeterminant &J, STLBitsetDeterminant &Jnew, const size_t &p, const bool &sp, const size_t &q, const bool &sq, const size_t &r, const bool &sr, const size_t &s, const bool &ss){
    timer_on("2PO");
    vector<vector<bool>> tmp;
    tmp.push_back(J.get_alfa_bits_vector_bool());
    tmp.push_back(J.get_beta_bits_vector_bool());

    double sign = 1.0;

    if(tmp[sr][r]){
        sign *= CheckSign(tmp[sr],r);
        tmp[sr][r] = 0;
    }else{timer_off("2PO"); return 0.0;}

    if(tmp[ss][s]){
        sign *= CheckSign(tmp[ss],s);
        tmp[ss][s] = 0;
    }else{timer_off("2PO"); return 0.0;}

    if(!tmp[sq][q]){
        sign *= CheckSign(tmp[sq],q);
        tmp[sq][q] = 1;
    }else{timer_off("2PO"); return 0.0;}

    if(!tmp[sp][p]){
        sign *= CheckSign(tmp[sp],p);
        tmp[sp][p] = 1;
        Jnew = STLBitsetDeterminant (tmp[0],tmp[1]);
        timer_off("2PO");
        return sign;
    }else{timer_off("2PO"); return 0.0;}
}

void FCI_MO::FormCumulant3(CI_RDMS &ci_rdms, d6 &AAA, d6 &AAB, d6 &ABB, d6 &BBB, string &DC){
    timer_on("FORM 3-Cumulant");
    Timer tL3;
    std::string str = "Forming Lambda3";
    outfile->Printf("\n  %-35s ...", str.c_str());

    size_t dim = na_ * na_ * na_ * na_ * na_ * na_;
    vector<double> tpdm_aaa (dim, 0.0);
    vector<double> tpdm_aab (dim, 0.0);
    vector<double> tpdm_abb (dim, 0.0);
    vector<double> tpdm_bbb (dim, 0.0);

    ci_rdms.compute_3rdm(tpdm_aaa,tpdm_aab,tpdm_abb,tpdm_bbb);

    FormCumulant3AAA(tpdm_aaa, tpdm_bbb, AAA, BBB, DC);
    FormCumulant3AAB(tpdm_aab, tpdm_abb, AAB, ABB, DC);
    fill_cumulant3();

    outfile->Printf("  Done. Timing %15.6f s", tL3.get());
    timer_off("FORM 3-Cumulant");
}

void FCI_MO::FormCumulant3AAA(const vector<double> &tpdm_aaa, const vector<double> &tpdm_bbb, d6 &AAA, d6 &BBB, string &DC){
    size_t dim2 = na_ * na_;
    size_t dim3 = na_ * dim2;
    size_t dim4 = na_ * dim3;
    size_t dim5 = na_ * dim4;

    for(size_t p = 0; p != na_; ++p){
        size_t np = idx_a_[p];
        for(size_t q = p + 1; q != na_; ++q){
            size_t nq = idx_a_[q];
            for(size_t r = q + 1; r != na_; ++r){
                size_t nr = idx_a_[r];
                for(size_t s = 0; s != na_; ++s){
                    size_t ns = idx_a_[s];
                    for(size_t t = s + 1; t != na_; ++t){
                        size_t nt = idx_a_[t];
                        for(size_t u = t + 1; u != na_; ++u){
                            size_t nu = idx_a_[u];

                            if((sym_active_[p] ^ sym_active_[q] ^ sym_active_[r] ^ sym_active_[s] ^ sym_active_[t] ^ sym_active_[u]) != 0) continue;

                            if(DC == "MK"){
                                size_t index = p * dim5 + q * dim4 + r * dim3 + s * dim2 + t * na_ + u;

                                AAA[p][q][r][s][t][u] += tpdm_aaa[index];
                                BBB[p][q][r][s][t][u] += tpdm_bbb[index];
                            }

                            AAA[p][q][r][s][t][u] -= P3DDD(Da_,np,nq,nr,ns,nt,nu);
                            AAA[p][q][r][s][t][u] -= P3DC(Da_,L2aa_,p,q,r,s,t,u);

                            BBB[p][q][r][s][t][u] -= P3DDD(Db_,np,nq,nr,ns,nt,nu);
                            BBB[p][q][r][s][t][u] -= P3DC(Db_,L2bb_,p,q,r,s,t,u);

                            size_t cop[] = {p, q, r};
                            size_t aop[] = {s, t, u};
                            int P1 = 1;
                            do{
                                int P2 = 1;
                                do{
                                    double sign = pow(-1.0, int(P1 / 2) + int(P2 / 2));
                                    AAA[cop[0]][cop[1]][cop[2]][aop[0]][aop[1]][aop[2]] = sign * AAA[p][q][r][s][t][u];
                                    BBB[cop[0]][cop[1]][cop[2]][aop[0]][aop[1]][aop[2]] = sign * BBB[p][q][r][s][t][u];
                                    ++P2;
                                }while(std::next_permutation(aop, aop + 3));
                                ++P1;
                            }while(std::next_permutation(cop, cop + 3));

                        }
                    }
                }
            }
        }
    }
}

void FCI_MO::FormCumulant3AAB(const vector<double> &tpdm_aab, const vector<double> &tpdm_abb, d6 &AAB, d6 &ABB, string &DC){
    size_t dim2 = na_ * na_;
    size_t dim3 = na_ * dim2;
    size_t dim4 = na_ * dim3;
    size_t dim5 = na_ * dim4;

    for(size_t p = 0; p != na_; ++p){
        size_t np = idx_a_[p];
        for(size_t q = p + 1; q != na_; ++q){
            size_t nq = idx_a_[q];
            for(size_t r = 0; r != na_; ++r){
                size_t nr = idx_a_[r];
                for(size_t s = 0; s != na_; ++s){
                    size_t ns = idx_a_[s];
                    for(size_t t = s + 1; t != na_; ++t){
                        size_t nt = idx_a_[t];
                        for(size_t u = 0; u != na_; ++u){
                            size_t nu = idx_a_[u];

                            if((sym_active_[p] ^ sym_active_[q] ^ sym_active_[r] ^ sym_active_[s] ^ sym_active_[t] ^ sym_active_[u]) != 0) continue;

                            if(DC == "MK"){
                                size_t index = p * dim5 + q * dim4 + r * dim3 + s * dim2 + t * na_ + u;
                                AAB[p][q][r][s][t][u] += tpdm_aab[index];

                                index = r * dim5 + p * dim4 + q * dim3 + u * dim2 + s * na_ + t;
                                ABB[r][p][q][u][s][t] += tpdm_abb[index];
                            }

                            AAB[p][q][r][s][t][u] -= (Da_[np][ns] * Da_[nq][nt] * Db_[nr][nu] - Da_[nq][ns] * Da_[np][nt] * Db_[nr][nu]);
                            AAB[p][q][r][s][t][u] -= (Da_[np][ns] * L2ab_[q][r][t][u] - Da_[np][nt] * L2ab_[q][r][s][u]);
                            AAB[p][q][r][s][t][u] -= (Da_[nq][nt] * L2ab_[p][r][s][u] - Da_[nq][ns] * L2ab_[p][r][t][u]);
                            AAB[p][q][r][s][t][u] -= (Db_[nr][nu] * L2aa_[p][q][s][t]);
                            AAB[q][p][r][s][t][u] -= AAB[p][q][r][s][t][u];
                            AAB[p][q][r][t][s][u] -= AAB[p][q][r][s][t][u];
                            AAB[q][p][r][t][s][u] += AAB[p][q][r][s][t][u];

                            ABB[r][p][q][u][s][t] -= (Db_[np][ns] * Db_[nq][nt] * Da_[nr][nu] - Db_[nq][ns] * Db_[np][nt] * Da_[nr][nu]);
                            ABB[r][p][q][u][s][t] -= (Db_[np][ns] * L2ab_[r][q][u][t] - Db_[np][nt] * L2ab_[r][q][u][s]);
                            ABB[r][p][q][u][s][t] -= (Db_[nq][nt] * L2ab_[r][p][u][s] - Db_[nq][ns] * L2ab_[r][p][u][t]);
                            ABB[r][p][q][u][s][t] -= (Da_[nr][nu] * L2bb_[p][q][s][t]);
                            ABB[r][q][p][u][s][t] -= ABB[r][p][q][u][s][t];
                            ABB[r][p][q][u][t][s] -= ABB[r][p][q][u][s][t];
                            ABB[r][q][p][u][t][s] += ABB[r][p][q][u][s][t];
                        }
                    }
                }
            }
        }
    }
}

void FCI_MO::FormCumulant3_DIAG(const vecdet &dets, const int &root, d6 &AAA, d6 &AAB, d6 &ABB, d6 &BBB){
    timer_on("FORM 3-Cumulant");
    Timer tL3;
    std::string str = "Forming Lambda3";
    outfile->Printf("\n  %-35s ...", str.c_str());
    for(size_t p=0; p<na_; ++p){
        size_t np = idx_a_[p];
        for(size_t q=0; q<na_; ++q){
            size_t nq = idx_a_[q];
            for(size_t r=0; r<na_; ++r){
                size_t nr = idx_a_[r];

                size_t size = dets.size();
                for(size_t ket = 0; ket != size; ++ket){
                    STLBitsetDeterminant Jaaa(vector<bool> (2*ncmo_)), Jaab(vector<bool> (2*ncmo_)), Jabb(vector<bool> (2*ncmo_)), Jbbb(vector<bool> (2*ncmo_));
                    double aaa = 1.0, aab = 1.0, abb = 1.0, bbb = 1.0, vket = (eigen_[root].first)->get(ket);;
                    aaa *= ThreeOP(dets[ket],Jaaa,p,0,q,0,r,0,p,0,q,0,r,0) * vket;
                    aab *= ThreeOP(dets[ket],Jaab,p,0,q,0,r,1,p,0,q,0,r,1) * vket;
                    abb *= ThreeOP(dets[ket],Jabb,p,0,q,1,r,1,p,0,q,1,r,1) * vket;
                    bbb *= ThreeOP(dets[ket],Jbbb,p,1,q,1,r,1,p,1,q,1,r,1) * vket;

                    for(size_t bra = 0; bra != size; ++bra){
                        double vbra = (eigen_[root].first)->get(bra);
                        AAA[p][q][r][p][q][r] += aaa * (dets[bra] == Jaaa) * vbra;
                        AAB[p][q][r][p][q][r] += aab * (dets[bra] == Jaab) * vbra;
                        ABB[p][q][r][p][q][r] += abb * (dets[bra] == Jabb) * vbra;
                        BBB[p][q][r][p][q][r] += bbb * (dets[bra] == Jbbb) * vbra;
                    }
                }


                AAA[p][q][r][p][q][r] -= P3DDD(Da_,np,nq,nr,np,nq,nr);
                AAA[p][q][r][p][q][r] -= P3DC(Da_,L2aa_,p,q,r,p,q,r);
                AAA[p][q][r][p][r][q] -= AAA[p][q][r][p][q][r];
                AAA[p][q][r][q][p][r] -= AAA[p][q][r][p][q][r];
                AAA[p][q][r][q][r][p]  = AAA[p][q][r][p][q][r];
                AAA[p][q][r][r][p][q]  = AAA[p][q][r][p][q][r];
                AAA[p][q][r][r][q][p] -= AAA[p][q][r][p][q][r];

                AAB[p][q][r][p][q][r] -= (Da_[np][np] * Da_[nq][nq] * Db_[nr][nr] - Da_[nq][np] * Da_[np][nq] * Db_[nr][nr]);
                AAB[p][q][r][p][q][r] -= (Da_[np][np] * L2ab_[q][r][q][r] - Da_[np][nq] * L2ab_[q][r][p][r]);
                AAB[p][q][r][p][q][r] -= (Da_[nq][nq] * L2ab_[p][r][p][r] - Da_[nq][np] * L2ab_[p][r][q][r]);
                AAB[p][q][r][p][q][r] -= (Db_[nr][nr] * L2aa_[p][q][p][q]);
                AAB[p][q][r][q][p][r] -= AAB[p][q][r][p][q][r];

                ABB[p][q][r][p][q][r] -= (Da_[np][np] * Db_[nq][nq] * Db_[nr][nr] - Da_[np][np] * Db_[nr][nq] * Db_[nq][nr]);
                ABB[p][q][r][p][q][r] -= (Db_[nq][nq] * L2ab_[p][r][p][r] - Db_[nq][nr] * L2ab_[p][r][p][q]);
                ABB[p][q][r][p][q][r] -= (Db_[nr][nr] * L2ab_[p][q][p][q] - Db_[nr][nq] * L2ab_[p][q][p][r]);
                ABB[p][q][r][p][q][r] -= (Da_[np][np] * L2bb_[q][r][q][r]);
                ABB[p][r][q][p][q][r] -= ABB[p][q][r][p][q][r];

                BBB[p][q][r][p][q][r] -= P3DDD(Db_,np,nq,nr,np,nq,nr);
                BBB[p][q][r][p][q][r] -= P3DC(Db_,L2bb_,p,q,r,p,q,r);
                BBB[p][q][r][p][r][q] -= BBB[p][q][r][p][q][r];
                BBB[p][q][r][q][p][r] -= BBB[p][q][r][p][q][r];
                BBB[p][q][r][q][r][p]  = BBB[p][q][r][p][q][r];
                BBB[p][q][r][r][p][q]  = BBB[p][q][r][p][q][r];
                BBB[p][q][r][r][q][p] -= BBB[p][q][r][p][q][r];
            }
        }
    }
    fill_cumulant3();
    outfile->Printf("  Done. Timing %15.6f s", tL3.get());
    timer_off("FORM 3-Cumulant");
}

void FCI_MO::print3PDC(const string &str, const d6 &ThreePDC, const int &PRINT){
    timer_on("PRINT 3-Cumulant");
    outfile->Printf("\n  ** %s **", str.c_str());
    size_t count = 0;
    for(size_t i = 0; i != ThreePDC.size(); ++i){
        for(size_t j =0; j != ThreePDC[i].size(); ++j){
            for(size_t k = 0; k != ThreePDC[i][j].size(); ++k){
                for(size_t l = 0; l != ThreePDC[i][j][k].size(); ++l){
                    for(size_t m = 0; m != ThreePDC[i][j][k][l].size(); ++m){
                        for(size_t n = 0; n != ThreePDC[i][j][k][l][m].size(); ++n){
                            if(fabs(ThreePDC[i][j][k][l][m][n]) > 1.0e-15){
                                ++count;
                                if(PRINT > 3)
                                    outfile->Printf("\n  Lambda [%3lu][%3lu][%3lu][%3lu][%3lu][%3lu] = %18.15lf", i, j, k, l, m, n, ThreePDC[i][j][k][l][m][n]);
                            }
                        }
                    }
                }
            }
        }
    }
    outfile->Printf("\n");
    outfile->Printf("\n  Number of Nonzero Elements: %zu", count);
    outfile->Printf("\n");
    timer_off("PRINT 3-Cumulant");
}

double FCI_MO::ThreeOP(const STLBitsetDeterminant &J, STLBitsetDeterminant &Jnew, const size_t &p, const bool &sp, const size_t &q, const bool &sq, const size_t &r, const bool &sr, const size_t &s, const bool &ss, const size_t &t, const bool &st, const size_t &u, const bool &su){
    timer_on("3PO");
    vector<vector<bool>> tmp;
    tmp.push_back(J.get_alfa_bits_vector_bool());
    tmp.push_back(J.get_beta_bits_vector_bool());

    double sign = 1.0;

    if(tmp[ss][s]){
        sign *= CheckSign(tmp[ss],s);
        tmp[ss][s] = 0;
    }else{timer_off("3PO"); return 0.0;}

    if(tmp[st][t]){
        sign *= CheckSign(tmp[st],t);
        tmp[st][t] = 0;
    }else{timer_off("3PO"); return 0.0;}

    if(tmp[su][u]){
        sign *= CheckSign(tmp[su],u);
        tmp[su][u] = 0;
    }else{timer_off("3PO"); return 0.0;}

    if(!tmp[sr][r]){
        sign *= CheckSign(tmp[sr],r);
        tmp[sr][r] = 1;
    }else{timer_off("3PO"); return 0.0;}

    if(!tmp[sq][q]){
        sign *= CheckSign(tmp[sq],q);
        tmp[sq][q] = 1;
    }else{timer_off("3PO"); return 0.0;}

    if(!tmp[sp][p]){
        sign *= CheckSign(tmp[sp],p);
        tmp[sp][p] = 1;
        Jnew = STLBitsetDeterminant (tmp[0],tmp[1]);
        timer_off("3PO");
        return sign;
    }else{timer_off("3PO"); return 0.0;}
}

void FCI_MO::Form_Fock(d2 &A, d2 &B){
    timer_on("Form Fock");
    Timer tfock;
    std::string str = "Forming generalized Fock matrix";
    if(!quiet_){outfile->Printf("\n  %-35s ...", str.c_str());}

    SharedMatrix DaM(new Matrix("DaM", ncmo_, ncmo_));
    SharedMatrix DbM(new Matrix("DbM", ncmo_, ncmo_));
    for (size_t m = 0; m < nc_; m++) {
        size_t nm = idx_c_[m];
        for( size_t n = 0; n < nc_; n++){
            size_t nn = idx_c_[n];
            DaM->set(nm,nn,Da_[nm][nn]);
            DbM->set(nm,nn,Db_[nm][nn]);
        }
    }
    for (size_t u = 0; u < na_; u++){
        size_t nu = idx_a_[u];
        for(size_t v = 0; v < na_; v++){
            size_t nv = idx_a_[v];
            DaM->set(nu,nv, Da_[nu][nv]);
            DbM->set(nu,nv, Db_[nu][nv]);
        }
    }
    integral_->make_fock_matrix(DaM, DbM);
    if(!quiet_){outfile->Printf("  Done. Timing %15.6f s", tfock.get());}

    for(size_t p = 0; p < ncmo_; ++p){
        for(size_t q = 0; q < ncmo_; ++q){
            A[p][q] = integral_->get_fock_a(p,q);
            B[p][q] = integral_->get_fock_b(p,q);
        }
    }

//    ambit::Tensor Fa = ambit::Tensor::build(ambit::CoreTensor,"Fa",{ncmo_,ncmo_});
//    ambit::Tensor Fb = ambit::Tensor::build(ambit::CoreTensor,"Fb",{ncmo_,ncmo_});

//    Fa.iterate([&](const std::vector<size_t>& i,double& value){
//        value = integral_->oei_a(i[0],i[1]);
//        for(const size_t& c: idx_c_){
//            value += integral_->aptei_aa(i[0],c,i[1],c);
//            value += integral_->aptei_ab(i[0],c,i[1],c);
//        }
//    });
//    Fb.iterate([&](const std::vector<size_t>& i,double& value){
//        value = integral_->oei_b(i[0],i[1]);
//        for(const size_t& c: idx_c_){
//            value += integral_->aptei_bb(i[0],c,i[1],c);
//            value += integral_->aptei_ab(c,i[0],c,i[1]);
//        }
//    });

//    std::vector<size_t> idx_corr (ncmo_);
//    std::iota(idx_corr.begin(), idx_corr.end(), 0);
//    ambit::Tensor V = integral_->aptei_aa_block(idx_corr,idx_a_,idx_corr,idx_a_);
//    Fa("pq") += V("puqv") * L1a("vu");

//    V = integral_->aptei_ab_block(idx_corr,idx_a_,idx_corr,idx_a_);
//    Fa("pq") += V("puqv") * L1b("vu");

//    V = integral_->aptei_ab_block(idx_a_,idx_corr,idx_a_,idx_corr);
//    Fb("pq") += V("upvq") * L1a("vu");

//    V = integral_->aptei_bb_block(idx_corr,idx_a_,idx_corr,idx_a_);
//    Fb("pq") += V("puqv") * L1b("vu");

//    for(size_t p = 0; p < ncmo_; ++p){
//        for(size_t q = 0; q < ncmo_; ++q){
//            A[p][q] = Fa.data()[p * ncmo_ + q];
//            B[p][q] = Fb.data()[p * ncmo_ + q];
//        }
//    }
    timer_off("Form Fock");
}

void FCI_MO::Check_Fock(const d2 &A, const d2 &B, const double &E, size_t &count){
    timer_on("Check Fock");
    Timer tfock;
    std::string str = "Checking Fock matrices (Fa, Fb)";
    if(!quiet_){outfile->Printf("\n  %-35s ...", str.c_str());}
    if(!quiet_){outfile->Printf("\n  Nonzero criteria: > %.2E", E);}
    Check_FockBlock(A, B, E, count, nc_, idx_c_, "CORE");
    if(options_.get_str("ACTIVE_SPACE_TYPE") == "COMPLETE" || options_.get_str("ACTIVE_SPACE_TYPE") == "DOCI"){
        Check_FockBlock(A, B, E, count, na_, idx_a_, "ACTIVE");
    }else{
        vector<size_t> idx_a_o, idx_a_v;
        for(int i = 0; i < ao_.size(); ++i){
            idx_a_o.push_back(idx_a_[ao_[i]]);
        }
        for(int i = 0; i < av_.size(); ++i){
            idx_a_v.push_back(idx_a_[av_[i]]);
        }
        Check_FockBlock(A, B, E, count, ao_.size(), idx_a_o, "ACT_O");
        Check_FockBlock(A, B, E, count, av_.size(), idx_a_v, "ACT_V");
    }
    Check_FockBlock(A, B, E, count, nv_, idx_v_, "VIRTUAL");
    str = "Done checking Fock matrices.";
    if(!quiet_){
        outfile->Printf("\n  %-47s", str.c_str());
        outfile->Printf("Timing %15.6f s", tfock.get());
        outfile->Printf("\n");
        outfile->Flush();
    }
    timer_off("Check Fock");
}

void FCI_MO::Check_FockBlock(const d2 &A, const d2 &B, const double &E, size_t &count, const size_t &dim, const vector<size_t> &idx, const string &str){
    double maxa = 0.0, maxb = 0.0;
    size_t a = 0, b = 0;
    for(size_t p=0; p<dim; ++p){
        size_t np = idx[p];
        for(size_t q=0; q<dim; ++q){
            size_t nq = idx[q];
            if(np != nq){
                if(fabs(A[np][nq]) > E){
                    ++a;
                    maxa = (fabs(A[np][nq]) > maxa) ? fabs(A[np][nq]) : maxa;
                }
                if(fabs(B[np][nq]) > E){
                    ++b;
                    maxb = (fabs(B[np][nq]) > maxb) ? fabs(B[np][nq]) : maxb;
                }
            }
        }
    }
    count += a+b;
    if(!quiet_)
    {
        if(a == 0){
            outfile->Printf("\n  Fa_%-7s block is diagonal.", str.c_str());
        }else{
            outfile->Printf("\n  Warning: Fa_%-7s NOT diagonal!", str.c_str());
            outfile->Printf("\n  Nonzero off-diagonal: %5zu. Largest value: %18.15lf", a, maxa);
        }
        if(b == 0){
            outfile->Printf("\n  Fb_%-7s block is diagonal.", str.c_str());
        }else{
            outfile->Printf("\n  Warning: Fb_%-7s NOT diagonal!", str.c_str());
            outfile->Printf("\n  Nonzero off-diagonal: %5zu. Largest value: %18.15lf", b, maxb);
        }
        outfile->Flush();
    }
}

void FCI_MO::BD_Fock(const d2 &Fa, const d2 &Fb, SharedMatrix &Ua, SharedMatrix &Ub, const string& name){
    timer_on("Block Diagonal 2D Matrix");
    Timer tbdfock;
    std::string str = "Diagonalizing " + name;
    outfile->Printf("\n  %-35s ...", str.c_str());

    // separate Fock to core, active, virtual blocks
    SharedMatrix Fc_a(new Matrix("Fock core alpha",core_,core_));
    SharedMatrix Fc_b(new Matrix("Fock core beta",core_,core_));
    SharedMatrix Fv_a(new Matrix("Fock virtual alpha",virtual_,virtual_));
    SharedMatrix Fv_b(new Matrix("Fock virtual beta",virtual_,virtual_));
    // core and virtual
    for (size_t h = 0, offset = 0; h < nirrep_; ++h){
        h = static_cast<int> (h);
        for (size_t i = 0; i < core_[h]; ++i){
            for (size_t j = 0; j < core_[h]; ++j){
                Fc_a->set(h,i,j,Fa[offset + i][offset + j]);
                Fc_b->set(h,i,j,Fb[offset + i][offset + j]);
            }
        }
        offset += core_[h] + active_[h];

        for (size_t a = 0; a < virtual_[h]; ++a){
            for (size_t b = 0; b < virtual_[h]; ++b){
                Fv_a->set(h,a,b,Fa[offset + a][offset + b]);
                Fv_b->set(h,a,b,Fb[offset + a][offset + b]);
            }
        }
        offset += virtual_[h];
    }
    // active
    SharedMatrix Fa_a, Fa_b, Fao_a, Fao_b, Fav_a, Fav_b;
    std::string active_type = options_.get_str("ACTIVE_SPACE_TYPE");
    if(active_type == "COMPLETE" || active_type == "DOCI"){
        Fa_a = SharedMatrix(new Matrix("Fock active alpha",active_,active_));
        Fa_b = SharedMatrix(new Matrix("Fock active beta",active_,active_));
        for (size_t h = 0, offset = 0; h < nirrep_; ++h){
            h = static_cast<int> (h);
            offset += core_[h];
            for (int u = 0; u < active_[h]; ++u){
                for (int v = 0; v < active_[h]; ++v){
                    Fa_a->set(h,u,v,Fa[offset + u][offset + v]);
                    Fa_b->set(h,u,v,Fb[offset + u][offset + v]);
                }
            }
            offset += active_[h] + virtual_[h];
        }
    }else{
        Fao_a = SharedMatrix(new Matrix("Fock active occupied alpha",active_o_,active_o_));
        Fao_b = SharedMatrix(new Matrix("Fock active occupied beta",active_o_,active_o_));
        Fav_a = SharedMatrix(new Matrix("Fock active virtual alpha",active_v_,active_v_));
        Fav_b = SharedMatrix(new Matrix("Fock active virtual beta",active_v_,active_v_));
        for (size_t h = 0, offset = 0; h < nirrep_; ++h){
            h = static_cast<int> (h);
            offset += core_[h];
            // active occupied
            for (int u = 0; u < active_o_[h]; ++u){
                for (int v = 0; v < active_o_[h]; ++v){
                    Fao_a->set(h,u,v,Fa[offset + u][offset + v]);
                    Fao_b->set(h,u,v,Fb[offset + u][offset + v]);
                }
            }
            // active virtual
            for (int u = active_o_[h]; u < active_[h]; ++u){
                int nu = u - active_o_[h];
                for (int v = active_o_[h]; v < active_[h]; ++v){
                    int nv = v - active_o_[h];
                    Fav_a->set(h,nu,nv,Fa[offset + u][offset + v]);
                    Fav_b->set(h,nu,nv,Fb[offset + u][offset + v]);
                }
            }
            offset += active_[h] + virtual_[h];
        }
    }

    // diagonalize Fock blocks
    std::vector<SharedMatrix> blocks;
    std::vector<SharedMatrix> evecs;
    std::vector<SharedVector> evals;
    if(active_type == "COMPLETE" || active_type == "DOCI"){
        blocks = {Fc_a,Fc_b,Fv_a,Fv_b,Fa_a,Fa_b};
    }else{
        blocks = {Fc_a,Fc_b,Fv_a,Fv_b,Fao_a,Fao_b,Fav_a,Fav_b};
    }
    for(auto F: blocks){
        std::string name = "U for " + F->name();
        SharedMatrix U(new Matrix(name,F->rowspi(),F->colspi()));
        SharedVector lambda(new Vector("lambda",F->rowspi()));
        F->diagonalize(U,lambda);
        evecs.push_back(U);
        evals.push_back(lambda);
//        U->eivprint(lambda);
//        SharedMatrix X = Matrix::triplet(U,F,U,true,false,false);
//        X->print();
    }

    // fill in the unitary rotation
    for (int h = 0; h < nirrep_; ++h){
        size_t offset = 0;

        // frozen core
        for(size_t i = 0; i < frzcpi_[h]; ++i){
            Ua->set(h,i,i,1.0);
            Ub->set(h,i,i,1.0);
        }
        offset += frzcpi_[h];

        // core
        for (size_t i = 0; i < core_[h]; ++i){
            for (size_t j = 0; j < core_[h]; ++j){
                Ua->set(h,offset + i,offset + j,evecs[0]->get(h,i,j));
                Ub->set(h,offset + i,offset + j,evecs[1]->get(h,i,j));
            }
        }
        offset += core_[h];

        // active
        if(active_type == "COMPLETE" || active_type == "DOCI"){
            for (int u = 0; u < active_[h]; ++u){
                for (int v = 0; v < active_[h]; ++v){
                    Ua->set(h,offset + u, offset + v,evecs[4]->get(h,u,v));
                    Ub->set(h,offset + u, offset + v,evecs[5]->get(h,u,v));
                }
            }
        }else{
            for (int u = 0; u < active_o_[h]; ++u){
                for (int v = 0; v < active_o_[h]; ++v){
                    Ua->set(h,offset + u, offset + v,evecs[4]->get(h,u,v));
                    Ub->set(h,offset + u, offset + v,evecs[5]->get(h,u,v));
                }
            }
            for (int u = active_o_[h]; u < active_[h]; ++u){
                int nu = u - active_o_[h];
                for (int v = active_o_[h]; v < active_[h]; ++v){
                    int nv = v - active_o_[h];
                    Ua->set(h,offset + u, offset + v,evecs[6]->get(h,nu,nv));
                    Ub->set(h,offset + u, offset + v,evecs[7]->get(h,nu,nv));
                }
            }
        }
        offset += active_[h];

        // virtual
        for (size_t a = 0; a < virtual_[h]; ++a){
            for (size_t b = 0; b < virtual_[h]; ++b){
                Ua->set(h,offset + a,offset + b,evecs[2]->get(h,a,b));
                Ub->set(h,offset + a,offset + b,evecs[3]->get(h,a,b));
            }
        }
        offset += virtual_[h];

        // frozen virtual
        for(size_t i = 0; i < frzvpi_[h]; ++i){
            size_t j = i + offset;
            Ua->set(h,j,j,1.0);
            Ub->set(h,j,j,1.0);
        }
        offset += frzvpi_[h];
    }

    outfile->Printf("  Done. Timing %15.6f s\n", tbdfock.get());
    timer_off("Block Diagonal 2D Matrix");
}


//void FCI_MO::nat_orbs(){
//    outfile->Printf("\n  Use natural orbitals.");

//    bool natural = CheckDensity();
//    if(!natural){
//        boost::shared_ptr<Wavefunction> wfn = Process::environment.wavefunction();
//        SharedMatrix Ua (new Matrix("Unitary A", nmopi_, nmopi_));
//        SharedMatrix Ub (new Matrix("Unitary B", nmopi_, nmopi_));
//        BD_2D_Matrix(Da_,Db_,Ua,Ub,"density","C");
//        BD_2D_Matrix(Da_,Db_,Ua,Ub,"density","A");
//        BD_2D_Matrix(Fa_,Fb_,Ua,Ub,"Fock","V");
//        SharedMatrix Ca = wfn->Ca();
//        SharedMatrix Cb = wfn->Cb();
//        SharedMatrix Ca_new(Ca->clone());
//        SharedMatrix Cb_new(Cb->clone());
//        Ca_new->gemm(false,false,1.0,Ca,Ua,0.0);
//        Cb_new->gemm(false,false,1.0,Cb,Ub,0.0);
//        Ca->copy(Ca_new);
//        Cb->copy(Cb_new);

//        integral_->retransform_integrals();
//        fci_ints_ = std::make_shared<FCIIntegrals>(integral_, mo_space_info_->get_corr_abs_mo("ACTIVE"), mo_space_info_->get_corr_abs_mo("RESTRICTED_DOCC"));
//        auto active_mo = mo_space_info_->get_corr_abs_mo("ACTIVE");
//        ambit::Tensor tei_active_aa = integral_->aptei_aa_block(active_mo, active_mo, active_mo, active_mo);
//        ambit::Tensor tei_active_ab = integral_->aptei_ab_block(active_mo, active_mo, active_mo, active_mo);
//        ambit::Tensor tei_active_bb = integral_->aptei_bb_block(active_mo, active_mo, active_mo, active_mo);
//        fci_ints_->set_active_integrals(tei_active_aa, tei_active_ab, tei_active_bb);
//        fci_ints_->compute_restricted_one_body_operator();

//        // Form and Diagonalize the CASCI Hamiltonian
//        Diagonalize_H(determinant_, eigen_);
//        if(print_ > 2){
//            for(pair<SharedVector, double> x: eigen_){
//                outfile->Printf("\n\n  Spin selected CI vectors\n");
//                (x.first)->print();
//                outfile->Printf("  Energy  =  %20.15lf\n", x.second);
//            }
//        }

//        // Store CI Vectors in eigen_
//        Store_CI(nroot_, options_.get_double("PRINT_CI_VECTOR"), eigen_, determinant_);

//        // Form Density
//        Da_ = d2(ncmo_, d1(ncmo_));
//        Db_ = d2(ncmo_, d1(ncmo_));
//        L1a = ambit::Tensor::build(ambit::CoreTensor,"L1a", {na_, na_});
//        L1b = ambit::Tensor::build(ambit::CoreTensor,"L1b", {na_, na_});
//        FormDensity(determinant_, root_, Da_, Db_);
//        CheckDensity();
//        if(print_ > 1){
//            print_d2("Da", Da_);
//            print_d2("Db", Db_);
//        }

//        // Fock Matrix
//        Fa_ = d2(ncmo_, d1(ncmo_));
//        Fb_ = d2(ncmo_, d1(ncmo_));
//        Form_Fock(Fa_,Fb_);
//        if(print_ > 1){
//            print_d2("Fa", Fa_);
//            print_d2("Fb", Fb_);
//        }
//    }
//}

bool FCI_MO::CheckDensity(){
    // check blocks
    auto checkblocks = [&](const size_t& dim, const vector<size_t>& idx) -> vector<double> {
        double maxa = 0.0, maxb = 0.0;
        for(size_t p = 0; p < dim; ++p){
            size_t np = idx[p];
            for(size_t q = 0; q < dim; ++q){
                size_t nq = idx[q];
                if(np != nq){
                    if(fabs(Da_[np][nq]) > maxa) maxa = Da_[np][nq];
                    if(fabs(Db_[np][nq]) > maxb) maxb = Db_[np][nq];
                }
            }
        }
        return {maxa, maxb};
    };

    outfile->Printf("\n    Checking if orbitals are natural orbitals ...");
    vector<double> maxes, temp;
    temp = checkblocks(nc_, idx_c_);
    maxes.insert(maxes.end(),temp.begin(),temp.end());
    temp = checkblocks(na_, idx_a_);
    maxes.insert(maxes.end(),temp.begin(),temp.end());
    temp = checkblocks(nv_, idx_v_);
    maxes.insert(maxes.end(),temp.begin(),temp.end());

    double maxes_sum = 0.0;
    for(auto it = maxes.begin(); it != maxes.end(); ++it){
        maxes_sum += *it;
    }

    bool natural = false;
    if(fabs(maxes_sum) > 10.0 * dconv_){
        std::string sep(3 + 16 * 3, '-');
        outfile->Printf("\n    Warning! Orbitals are not natural orbitals!");
        outfile->Printf("\n    Max off-diagonal values of core, active, virtual blocks of the density matrix");
        outfile->Printf("\n       %15s %15s %15s", "core", "active", "virtual");
        outfile->Printf("\n    %s", sep.c_str());
        outfile->Printf("\n    Da %15.10f %15.10f %15.10f", maxes[0], maxes[2], maxes[4]);
        outfile->Printf("\n    Db %15.10f %15.10f %15.10f", maxes[1], maxes[3], maxes[5]);
        outfile->Printf("\n    %s\n", sep.c_str());
    }else{
        outfile->Printf("     OK.");
        natural = true;
    }
    return natural;
}

void FCI_MO::compute_SOquadrupole(){
    so_Qpole_.clear();
    for(const auto& name: {"XX","XY","XZ","YY","YZ","ZZ"}){
        so_Qpole_.push_back(SharedMatrix(new Matrix(name,this->nsopi(),this->nsopi()) ));
    }

    boost::shared_ptr<BasisSet> basisset = this->basisset();
    boost::shared_ptr<IntegralFactory> ints = boost::shared_ptr<IntegralFactory>(
                new IntegralFactory(basisset,basisset,basisset,basisset));
    boost::shared_ptr<OneBodySOInt> soqOBI(ints->so_quadrupole());

    soqOBI->compute(so_Qpole_);
    soqOBI.reset();
}

void FCI_MO::compute_orbital_extents(){
    // absolute indices of active orbitals before frozen core
    vector<size_t> abs_mo_a = mo_space_info_->get_absolute_mo("ACTIVE");

    // orbital coefficients
    SharedMatrix Ca = this->Ca();

    // initialize vector saving current orbital extents
    orb_extents_ = vector<d2> (nirrep_, d2());

    // compute active orbital extents
    size_t offset = 0;
    for(int h = 0; h < nirrep_; ++h){
        size_t na = active_[h];
        size_t nsopi = this->nsopi()[h];
        d2 orb_extents_pi (na, d1());

        for(size_t u = 0; u < na; ++u){
            size_t u_abs = abs_mo_a[u + offset];

            double sumx = 0.0, sumy = 0.0, sumz = 0.0;
            for(size_t k = 0; k < nsopi; ++k){
                for(size_t l = 0; l < nsopi; ++l){
                    double tmp = Ca->get(h,k,u_abs) * Ca->get(h,l,u_abs);
                    sumx += so_Qpole_[0]->get(h,k,l) * tmp;
                    sumy += so_Qpole_[3]->get(h,k,l) * tmp;
                    sumz += so_Qpole_[5]->get(h,k,l) * tmp;
                }
            }
            orb_extents_pi[u] = {fabs(sumx), fabs(sumy), fabs(sumz)};
        }

        orb_extents_[h] = orb_extents_pi;
        offset += na;
    }

    // find the diffused orbital index (active zero based)
    if(ipea_ != "NONE"){
        size_t wrong = 999999999;
        idx_diffused_ = wrong;

        diffused_orbs_.clear();
        size_t offset = 0;
        for(int h = 0; h < nirrep_; ++h){
            for(size_t i = 0; i < active_[h]; ++i){
                double orbext = orb_extents_[h][i][0]
                        + orb_extents_[h][i][1] + orb_extents_[h][i][2];

                if(orbext > 1.0e6){
                    diffused_orbs_.push_back(i + offset);

                    if(h == 0){
                        idx_diffused_ = i; // totally symmetric diffused orbital
                    }
                }
            }
            offset += active_[h];
        }

        if(idx_diffused_ == wrong){
            outfile->Printf("\n  Totally symmetric diffused orbital is not found.");
            outfile->Printf("\n  Make sure a diffused s function is added to the basis.");
            throw PSIEXCEPTION("Totally symmetric diffused orbital is not found.");
        }
    }
}

void FCI_MO::fill_density(){
    L1a.iterate([&](const::vector<size_t>& i,double& value){
        size_t np = idx_a_[i[0]];
        size_t nq = idx_a_[i[1]];
        value = Da_[np][nq];
    });
    L1b.iterate([&](const::vector<size_t>& i,double& value){
        size_t np = idx_a_[i[0]];
        size_t nq = idx_a_[i[1]];
        value = Db_[np][nq];
    });
}

void FCI_MO::fill_cumulant2(){
    L2aa.iterate([&](const::vector<size_t>& i,double& value){
        value = L2aa_[i[0]][i[1]][i[2]][i[3]];
    });
    L2ab.iterate([&](const::vector<size_t>& i,double& value){
        value = L2ab_[i[0]][i[1]][i[2]][i[3]];
    });
    L2bb.iterate([&](const::vector<size_t>& i,double& value){
        value = L2bb_[i[0]][i[1]][i[2]][i[3]];
    });
}

void FCI_MO::fill_cumulant3(){
    L3aaa.iterate([&](const::vector<size_t>& i,double& value){
        value = L3aaa_[i[0]][i[1]][i[2]][i[3]][i[4]][i[5]];
    });
    L3aab.iterate([&](const::vector<size_t>& i,double& value){
        value = L3aab_[i[0]][i[1]][i[2]][i[3]][i[4]][i[5]];
    });
    L3abb.iterate([&](const::vector<size_t>& i,double& value){
        value = L3abb_[i[0]][i[1]][i[2]][i[3]][i[4]][i[5]];
    });
    L3bbb.iterate([&](const::vector<size_t>& i,double& value){
        value = L3bbb_[i[0]][i[1]][i[2]][i[3]][i[4]][i[5]];
    });
}

void FCI_MO::fill_density(vector<double> &opdm_a, vector<double> &opdm_b){
    Da_ = d2(ncmo_, d1(ncmo_));
    Db_ = d2(ncmo_, d1(ncmo_));
    L1a = ambit::Tensor::build(ambit::CoreTensor,"L1a",{na_, na_});
    L1b = ambit::Tensor::build(ambit::CoreTensor,"L1b",{na_, na_});

    // fill in L1a and L1b
    L1a.data() = opdm_a;
    L1b.data() = opdm_b;

    // fill in Da_ and Db_
    for(size_t p = 0; p < nc_; ++p){
        size_t np = idx_c_[p];
            Da_[np][np] = 1.0;
            Db_[np][np] = 1.0;
    }

    for(size_t p = 0; p < na_; ++p){
        size_t np = idx_a_[p];
        for(size_t q = p; q < na_; ++q){
            size_t nq = idx_a_[q];

            if((sym_active_[p] ^ sym_active_[q]) != 0) continue;

            size_t index = p * na_ + q;
            Da_[np][nq] = opdm_a[index];
            Db_[np][nq] = opdm_b[index];

            Da_[nq][np] = Da_[np][nq];
            Db_[nq][np] = Db_[np][nq];
        }
    }
}

void FCI_MO::compute_ref(){
    timer_on("Compute Ref");

    // prepare ci_rdms
    int dim = (eigen_[0].first)->dim();
    SharedMatrix evecs (new Matrix("evecs",dim,dim));
    for(int i = 0; i < eigen_.size(); ++i){
        evecs->set_column(0,i,(eigen_[i]).first);
    }
    CI_RDMS ci_rdms (options_,fci_ints_,determinant_,evecs, root_, root_);

    // 2-PDC
    L2aa_ = d4(na_, d3(na_, d2(na_, d1(na_))));
    L2ab_ = d4(na_, d3(na_, d2(na_, d1(na_))));
    L2bb_ = d4(na_, d3(na_, d2(na_, d1(na_))));
    L2aa = ambit::Tensor::build(ambit::CoreTensor,"L2aa",{na_, na_, na_, na_});
    L2ab = ambit::Tensor::build(ambit::CoreTensor,"L2ab",{na_, na_, na_, na_});
    L2bb = ambit::Tensor::build(ambit::CoreTensor,"L2bb",{na_, na_, na_, na_});

    FormCumulant2(ci_rdms, L2aa_, L2ab_, L2bb_);
    if(print_ > 2){
        print2PDC("L2aa", L2aa_, print_);
        print2PDC("L2ab", L2ab_, print_);
        print2PDC("L2bb", L2bb_, print_);
    }

    // 3-PDC
    string threepdc = options_.get_str("THREEPDC");
    string t_algorithm = options_.get_str("T_ALGORITHM");
    if(threepdc != "ZERO"){
        L3aaa_ = d6(na_, d5(na_, d4(na_, d3(na_, d2(na_, d1(na_))))));
        L3aab_ = d6(na_, d5(na_, d4(na_, d3(na_, d2(na_, d1(na_))))));
        L3abb_ = d6(na_, d5(na_, d4(na_, d3(na_, d2(na_, d1(na_))))));
        L3bbb_ = d6(na_, d5(na_, d4(na_, d3(na_, d2(na_, d1(na_))))));
        L3aaa = ambit::Tensor::build(ambit::CoreTensor,"L3aaa",{na_, na_, na_, na_, na_, na_});
        L3aab = ambit::Tensor::build(ambit::CoreTensor,"L3aab",{na_, na_, na_, na_, na_, na_});
        L3abb = ambit::Tensor::build(ambit::CoreTensor,"L3abb",{na_, na_, na_, na_, na_, na_});
        L3bbb = ambit::Tensor::build(ambit::CoreTensor,"L3bbb",{na_, na_, na_, na_, na_, na_});

        if(boost::starts_with(threepdc, "MK") && t_algorithm != "DSRG_NOSEMI"){
            FormCumulant3(ci_rdms, L3aaa_, L3aab_, L3abb_, L3bbb_, threepdc);
        }
        else if (threepdc == "DIAG"){
            FormCumulant3_DIAG(determinant_, root_, L3aaa_, L3aab_, L3abb_, L3bbb_);
        }
        if(print_ > 3){
            print3PDC("L3aaa", L3aaa_, print_);
            print3PDC("L3aab", L3aab_, print_);
            print3PDC("L3abb", L3abb_, print_);
            print3PDC("L3bbb", L3bbb_, print_);
        }
    }

    timer_off("Compute Ref");
}

Reference FCI_MO::reference()
{
    if(options_["AVG_WEIGHTS"].has_changed()){
        compute_sa_ref();
    } else {
        compute_ref();
    }

    Reference ref;
    ref.set_Eref(Eref_);
    ref.set_L1a(L1a);
    ref.set_L1b(L1b);
    ref.set_L2aa(L2aa);
    ref.set_L2ab(L2ab);
    ref.set_L2bb(L2bb);
    if(options_.get_str("THREEPDC") != "ZERO"){
        ref.set_L3aaa(L3aaa);
        ref.set_L3aab(L3aab);
        ref.set_L3abb(L3abb);
        ref.set_L3bbb(L3bbb);
    }
    return ref;
}

void FCI_MO::set_orbs(SharedMatrix Ca, SharedMatrix Cb){
    SharedMatrix Ca_wfn = this->Ca();
    SharedMatrix Cb_wfn = this->Cb();
    Ca_wfn->copy(Ca);
    Cb_wfn->copy(Cb);
    integral_->retransform_integrals();
    ambit::Tensor tei_active_aa = integral_->aptei_aa_block(idx_a_, idx_a_, idx_a_, idx_a_);
    ambit::Tensor tei_active_ab = integral_->aptei_ab_block(idx_a_, idx_a_, idx_a_, idx_a_);
    ambit::Tensor tei_active_bb = integral_->aptei_bb_block(idx_a_, idx_a_, idx_a_, idx_a_);
    fci_ints_->set_active_integrals(tei_active_aa, tei_active_ab, tei_active_bb);
    fci_ints_->compute_restricted_one_body_operator();
}

double FCI_MO::compute_sa_energy(){
    // averaged energy and density
    double Ecas_sa = 0.0;
    size_t nelement = na_ * na_;
    vector<double> sa_opdm_a (nelement, 0.0);
    vector<double> sa_opdm_b (nelement, 0.0);
    eigens_.clear();

    // allocate Fock matrix
    Fa_ = d2(ncmo_, d1(ncmo_));
    Fb_ = d2(ncmo_, d1(ncmo_));

    // loop over all averaged states
    int nstates = 0;
    for(const auto& info: sa_info_){
        // get current symmetry, multiplicity, nroots, weights
        int irrep, multi, nroots;
        std::vector<double> weights;
        std::tie (irrep, multi, nroots, weights) = info;
        nstates += nroots;

        root_sym_ = irrep;
        multi_ = multi;
        nroot_ = nroots;
        root_ = nroot_ - 1; // not necessary

        // form determinants
        form_p_space();

        // diagonalize the CASCI Hamiltonian
        diag_algorithm_ = options_.get_str("DIAG_ALGORITHM");
        eigen_.clear();
        Diagonalize_H(determinant_, eigen_);
        eigens_.push_back(eigen_);

        // store CI vectors in eigen_
        if(nroot_ > eigen_.size()){
            outfile->Printf("\n  Too many roots of interest!");
            if(eigen_.size() > 1){
                outfile->Printf("\n  There are only %3d roots that satisfy the condition!", eigen_.size());
            } else {
                outfile->Printf("\n  There is only %3d root that satisfy the condition!", eigen_.size());
            }
            outfile->Printf("\n  Check root_sym, multi, and number of determinants.");
            throw PSIEXCEPTION("Too many roots of interest.");
        }
        print_CI(nroot_, options_.get_double("PRINT_CI_VECTOR"), eigen_, determinant_);

        // compute one density using ci_rdms
        int dim = (eigen_[0].first)->dim();
        SharedMatrix evecs (new Matrix("evecs",dim,dim));
        for(int i = 0; i < eigen_.size(); ++i){
            evecs->set_column(0,i,(eigen_[i]).first);
        }

        for(int i = 0; i < nroots; ++i){
            double weight = weights[i];
            Ecas_sa += weight * eigen_[i].second;

            CI_RDMS ci_rdms (options_,fci_ints_,determinant_,evecs,i,i);
            ci_rdms.set_symmetry(irrep);
            vector<double> opdm_a (nelement, 0.0);
            vector<double> opdm_b (nelement, 0.0);
            ci_rdms.compute_1rdm(opdm_a,opdm_b);

            std::for_each(opdm_a.begin(), opdm_a.end(), [&](double& v) {v *= weight;});
            std::for_each(opdm_b.begin(), opdm_b.end(), [&](double& v) {v *= weight;});
            std::transform (sa_opdm_a.begin(), sa_opdm_a.end(), opdm_a.begin(), sa_opdm_a.begin(), std::plus<double>());
            std::transform (sa_opdm_b.begin(), sa_opdm_b.end(), opdm_b.begin(), sa_opdm_b.begin(), std::plus<double>());
        }

    } // end looping over all averaged states
    eigen_.clear(); // make sure other code use eigens_ for state average
    outfile->Printf("\n  Total Energy (averaged over %d states): %20.15f\n", nstates, Ecas_sa);

    // fill in Da_, Db_, L1a, L1b
    fill_density(sa_opdm_a,sa_opdm_b);
    if(print_ > 1){
        print_d2("Da", Da_);
        print_d2("Db", Db_);
    }

    // form Fock matrix
    size_t count = 0;
    Form_Fock(Fa_,Fb_);
    Check_Fock(Fa_,Fb_,dconv_,count);
    if(print_ > 1){
        print_d2("Fa", Fa_);
        print_d2("Fb", Fb_);
    }

    // Orbitals. If use Kevin's CASSCF, this part is ignored.
    if(!default_orbitals_){
        if(semi_ && count != 0){
            // Semi-canonicalize orbitals
            outfile->Printf("\n  Use semi-canonical orbitals.\n");
            semi_canonicalize();

            // clear previous stuff
            Ecas_sa = 0.0;
            sa_opdm_a = vector<double> (nelement, 0.0);
            sa_opdm_b = vector<double> (nelement, 0.0);
            eigens_.clear();

            // loop over all averaging states
            for(const auto& info: sa_info_){
                // set current symmetry, multiplicity, nroots
                int irrep, multi, nroots;
                std::vector<double> weights;
                std::tie (irrep, multi, nroots, weights) = info;

                root_sym_ = irrep;
                multi_ = multi;
                nroot_ = nroots;
                root_ = nroot_ - 1; // not necessary

                // form determinants
                form_p_space();

                // diagonalize the CASCI Hamiltonian
                eigen_.clear();
                Diagonalize_H(determinant_, eigen_);
                eigens_.push_back(eigen_);

                // print CI vectors in eigen_
                print_CI(nroot_, options_.get_double("PRINT_CI_VECTOR"), eigen_, determinant_);

                // compute one density using ci_rdms
                int dim = (eigen_[0].first)->dim();
                SharedMatrix evecs (new Matrix("evecs",dim,dim));
                for(int i = 0; i < eigen_.size(); ++i){
                    evecs->set_column(0,i,(eigen_[i]).first);
                }

                for(int i = 0; i < nroots; ++i){
                    double weight = weights[i];
                    Ecas_sa += weight * eigen_[i].second;

                    CI_RDMS ci_rdms (options_,fci_ints_,determinant_,evecs,i,i);
                    ci_rdms.set_symmetry(irrep);
                    vector<double> opdm_a (nelement, 0.0);
                    vector<double> opdm_b (nelement, 0.0);
                    ci_rdms.compute_1rdm(opdm_a,opdm_b);
                    for(double& x: opdm_a){
                        outfile->Printf("\n  %20.15f",x);
                    }

                    std::for_each(opdm_a.begin(), opdm_a.end(), [&](double& v) {v *= weight;});
                    std::for_each(opdm_b.begin(), opdm_b.end(), [&](double& v) {v *= weight;});
                    std::transform (sa_opdm_a.begin(), sa_opdm_a.end(), opdm_a.begin(), sa_opdm_a.begin(), std::plus<double>());
                    std::transform (sa_opdm_b.begin(), sa_opdm_b.end(), opdm_b.begin(), sa_opdm_b.begin(), std::plus<double>());
                }

            } // end looping over all averaged states
            eigen_.clear(); // make sure other code use eigens_ for state average
            outfile->Printf("\n    Total Energy (averaged over %d states): %20.15f\n", nstates, Ecas_sa);

            // fill in Da_, Db_, L1a, L1b
            fill_density(sa_opdm_a,sa_opdm_b);
            if(print_ > 1){
                print_d2("Da", Da_);
                print_d2("Db", Db_);
            }

            // form Fock matrix
            count = 0;
            Form_Fock(Fa_,Fb_);
            Check_Fock(Fa_,Fb_,dconv_,count);
            if(print_ > 1){
                print_d2("Fa", Fa_);
                print_d2("Fb", Fb_);
            }

        }else{
//            nat_orbs();
        }
    }

    Eref_ = Ecas_sa;
    Process::environment.globals["CURRENT ENERGY"] = Eref_;
    return Eref_;
}

void FCI_MO::compute_sa_ref(){
    timer_on("Compute SA Ref");

    // prepare averaged 2- and 3-densities
    size_t nelement2 = na_ * na_ * na_ * na_;
    vector<double> sa_tpdm_aa (nelement2, 0.0);
    vector<double> sa_tpdm_ab (nelement2, 0.0);
    vector<double> sa_tpdm_bb (nelement2, 0.0);

    size_t nelement3 = na_ * na_ * nelement2;
    bool no_3pdc = (options_.get_str("THREEPDC") == "ZERO");
    vector<double> sa_tpdm_aaa, sa_tpdm_aab, sa_tpdm_abb, sa_tpdm_bbb;
    if(!no_3pdc){
        sa_tpdm_aaa = vector<double> (nelement3, 0.0);
        sa_tpdm_aab = vector<double> (nelement3, 0.0);
        sa_tpdm_abb = vector<double> (nelement3, 0.0);
        sa_tpdm_bbb = vector<double> (nelement3, 0.0);
    }

    // loop over all averaged states
    int nentry = sa_info_.size();
    for(int n = 0; n < nentry; ++n){
        // get current nroots and weights
        int nroots, irrep;
        std::vector<double> weights;
        std::tie (irrep, std::ignore, nroots, weights) = sa_info_[n];

        // compute 2rdms and 3rdms
        int dim = (eigens_[n][0].first)->dim();
        SharedMatrix evecs (new Matrix("evecs",dim,dim));
        for(int i = 0; i < eigens_[n].size(); ++i){
            evecs->set_column(0,i,(eigens_[n][i]).first);
        }

        for(int i = 0; i < nroots; ++i){
            double weight = weights[i];
            CI_RDMS ci_rdms (options_,fci_ints_,determinant_,evecs,i,i);
            ci_rdms.set_symmetry(irrep);

            vector<double> tpdm_aa (nelement2, 0.0);
            vector<double> tpdm_ab (nelement2, 0.0);
            vector<double> tpdm_bb (nelement2, 0.0);
            ci_rdms.compute_2rdm(tpdm_aa,tpdm_ab,tpdm_bb);

            std::for_each(tpdm_aa.begin(), tpdm_aa.end(), [&](double& v) {v *= weight;});
            std::for_each(tpdm_ab.begin(), tpdm_ab.end(), [&](double& v) {v *= weight;});
            std::for_each(tpdm_bb.begin(), tpdm_bb.end(), [&](double& v) {v *= weight;});
            std::transform (sa_tpdm_aa.begin(), sa_tpdm_aa.end(), tpdm_aa.begin(), sa_tpdm_aa.begin(), std::plus<double>());
            std::transform (sa_tpdm_ab.begin(), sa_tpdm_ab.end(), tpdm_ab.begin(), sa_tpdm_ab.begin(), std::plus<double>());
            std::transform (sa_tpdm_bb.begin(), sa_tpdm_bb.end(), tpdm_bb.begin(), sa_tpdm_bb.begin(), std::plus<double>());

            if(!no_3pdc){
                vector<double> tpdm_aaa (nelement3, 0.0);
                vector<double> tpdm_aab (nelement3, 0.0);
                vector<double> tpdm_abb (nelement3, 0.0);
                vector<double> tpdm_bbb (nelement3, 0.0);
                ci_rdms.compute_3rdm(tpdm_aaa,tpdm_aab,tpdm_abb,tpdm_bbb);

                std::for_each(tpdm_aaa.begin(), tpdm_aaa.end(), [&](double& v) {v *= weight;});
                std::for_each(tpdm_aab.begin(), tpdm_aab.end(), [&](double& v) {v *= weight;});
                std::for_each(tpdm_abb.begin(), tpdm_abb.end(), [&](double& v) {v *= weight;});
                std::for_each(tpdm_bbb.begin(), tpdm_bbb.end(), [&](double& v) {v *= weight;});
                std::transform (sa_tpdm_aaa.begin(), sa_tpdm_aaa.end(), tpdm_aaa.begin(), sa_tpdm_aaa.begin(), std::plus<double>());
                std::transform (sa_tpdm_aab.begin(), sa_tpdm_aab.end(), tpdm_aab.begin(), sa_tpdm_aab.begin(), std::plus<double>());
                std::transform (sa_tpdm_abb.begin(), sa_tpdm_abb.end(), tpdm_abb.begin(), sa_tpdm_abb.begin(), std::plus<double>());
                std::transform (sa_tpdm_bbb.begin(), sa_tpdm_bbb.end(), tpdm_bbb.begin(), sa_tpdm_bbb.begin(), std::plus<double>());
            }
        }

    } // end looping over all averaged states

    // compute 2-cumulants and 3-cumulants
    compute_cumulant2(sa_tpdm_aa, sa_tpdm_ab, sa_tpdm_bb);

    if(!no_3pdc){
        compute_cumulant3(sa_tpdm_aaa,sa_tpdm_aab,sa_tpdm_abb,sa_tpdm_bbb);
    }

    timer_off("Compute SA Ref");
}

void FCI_MO::compute_cumulant2(vector<double>& tpdm_aa, vector<double>& tpdm_ab, vector<double>& tpdm_bb){
    L2aa = ambit::Tensor::build(ambit::CoreTensor,"L2aa",{na_, na_, na_, na_});
    L2ab = ambit::Tensor::build(ambit::CoreTensor,"L2ab",{na_, na_, na_, na_});
    L2bb = ambit::Tensor::build(ambit::CoreTensor,"L2bb",{na_, na_, na_, na_});

    // copy incoming 2rdms to 2cumulants
    L2aa.data() = tpdm_aa;
    L2ab.data() = tpdm_ab;
    L2bb.data() = tpdm_bb;

    // add wedge product of 1cumulants (1rdms)
    L2aa("pqrs") -= L1a("pr") * L1a("qs");
    L2aa("pqrs") += L1a("ps") * L1a("qr");

    L2bb("pqrs") -= L1b("pr") * L1b("qs");
    L2bb("pqrs") += L1b("ps") * L1b("qr");

    L2ab("pqrs") -= L1a("pr") * L1b("qs");
}

void FCI_MO::compute_cumulant3(vector<double>& tpdm_aaa, vector<double>& tpdm_aab, vector<double>& tpdm_abb, vector<double>& tpdm_bbb){
    L3aaa = ambit::Tensor::build(ambit::CoreTensor,"L3aaa",{na_, na_, na_, na_, na_, na_});
    L3aab = ambit::Tensor::build(ambit::CoreTensor,"L3aab",{na_, na_, na_, na_, na_, na_});
    L3abb = ambit::Tensor::build(ambit::CoreTensor,"L3abb",{na_, na_, na_, na_, na_, na_});
    L3bbb = ambit::Tensor::build(ambit::CoreTensor,"L3bbb",{na_, na_, na_, na_, na_, na_});

    // copy incoming 3rdms to 3cumulants
    L3aaa.data() = tpdm_aaa;
    L3aab.data() = tpdm_aab;
    L3abb.data() = tpdm_abb;
    L3bbb.data() = tpdm_bbb;

    // add wedge product of 1cumulants and 2cumulants
    // - step 1: aaa
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

    // - step 2: aab
    L3aab("pqRstU") -= L1a("ps") * L2ab("qRtU");
    L3aab("pqRstU") += L1a("pt") * L2ab("qRsU");

    L3aab("pqRstU") -= L1a("qt") * L2ab("pRsU");
    L3aab("pqRstU") += L1a("qs") * L2ab("pRtU");

    L3aab("pqRstU") -= L1b("RU") * L2aa("pqst");

    L3aab("pqRstU") -= L1a("ps") * L1a("qt") * L1b("RU");
    L3aab("pqRstU") += L1a("pt") * L1a("qs") * L1b("RU");

    // - step 3: abb
    L3abb("pQRsTU") -= L1a("ps") * L2bb("QRTU");

    L3abb("pQRsTU") -= L1b("QT") * L2ab("pRsU");
    L3abb("pQRsTU") += L1b("QU") * L2ab("pRsT");

    L3abb("pQRsTU") -= L1b("RU") * L2ab("pQsT");
    L3abb("pQRsTU") += L1b("RT") * L2ab("pQsU");

    L3abb("pQRsTU") -= L1a("ps") * L1b("QT") * L1b("RU");
    L3abb("pQRsTU") += L1a("ps") * L1b("QU") * L1b("RT");

    // - step 4: bbb
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
}

}}
