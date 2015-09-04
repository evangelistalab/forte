#include <cmath>
#include <libmints/wavefunction.h>
#include <libmints/molecule.h>
#include <libmints/matrix.h>
#include <vector>
#include <tuple>
#include <numeric>
#include <algorithm>
#include <boost/algorithm/string/predicate.hpp>
#include "fci_mo.h"

using namespace std;

namespace psi{ namespace forte{

FCI_MO::FCI_MO(Options &options, ForteIntegrals *ints) : integral_(ints)
{

    // Basic Preparation: Form Determinants
    startup(options);

    // Form and Diagonalize the CASCI Hamiltonian
    diag_algorithm_ = options.get_str("DIAG_ALGORITHM");
    Diagonalize_H(determinant_, eigen_);
    if(print_ > 2){
        for(pair<SharedVector, double> x: eigen_){
            outfile->Printf("\n\n  Spin selected CI vectors\n");
            (x.first)->print();
            outfile->Printf("  Energy  =  %20.15lf\n", x.second);
        }
    }

    // Store CI Vectors in eigen_
    print_CI_threshold = options.get_double("PRINT_CI_VECTOR");
    if(nroot_ > eigen_.size()){
        outfile->Printf("\n  Too many roots of interest!");
        if(eigen_.size() > 1)
            outfile->Printf("\n  There are only %3d roots that satisfy the condition!", eigen_.size());
        else
            outfile->Printf("\n  There are only %3d root that satisfy the condition!", eigen_.size());
        outfile->Printf("\n  Check root_sym, multi, etc.");
        outfile->Printf("\n  If unrestricted orbitals are used, spin contamination may be severe (> 5%%).");
        throw PSIEXCEPTION("Too many roots of interest.");
    }
    Store_CI(nroot_, print_CI_threshold, eigen_, determinant_);

    // Form Density
    FormDensity(determinant_, root_, Da_, Db_);
    if(print_ > 1){
        print_d2("Da", Da_);
        print_d2("Db", Db_);
    }

    // Fock Matrix
    size_t count = 0;
    Form_Fock(Fa_,Fb_);
    Check_Fock(Fa_,Fb_,econv_,count);
    if(print_ > 1){
        print_d2("Fa", Fa_);
        print_d2("Fb", Fb_);
    }

    // Semi-Canonicalize Orbitals
    if(count != 0 && options.get_bool("SEMI_CANONICAL")){
        semi_canonicalize();
    }

    // Form 2-PDC
    FormCumulant2(determinant_, root_, L2aa_, L2ab_, L2bb_);
    if(print_ > 2){
        print2PDC("L2aa", L2aa_, print_);
        print2PDC("L2ab", L2ab_, print_);
        print2PDC("L2bb", L2bb_, print_);
    }

    // Form 3-PDC
    string threepdc = options.get_str("THREEPDC");
    string t_algorithm = options.get_str("T_ALGORITHM");
    if(boost::starts_with(threepdc, "MK") && t_algorithm != "DSRG_NOSEMI"){
        FormCumulant3(determinant_, root_, L3aaa_, L3aab_, L3abb_, L3bbb_, threepdc);
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

    // Reference Energy
    compute_ref();
}

FCI_MO::~FCI_MO()
{
    cleanup();
}

void FCI_MO::cleanup(){
//    delete integral_;
}

void FCI_MO::startup(Options &options){

    // Print Title
    print_title();

    // Read from options
    read_info(options);

    // Form determinants
    form_det();

    // Density
    Da_ = d2(ncmo_, d1(ncmo_));
    Db_ = d2(ncmo_, d1(ncmo_));
    L1a = ambit::Tensor::build(ambit::kCore,"L1a", {na_, na_});
    L1b = ambit::Tensor::build(ambit::kCore,"L1b", {na_, na_});

    // Fock
    Fa_ = d2(ncmo_, d1(ncmo_));
    Fb_ = d2(ncmo_, d1(ncmo_));

    // Cumulants
    L2aa_ = d4(na_, d3(na_, d2(na_, d1(na_))));
    L2ab_ = d4(na_, d3(na_, d2(na_, d1(na_))));
    L2bb_ = d4(na_, d3(na_, d2(na_, d1(na_))));
    L2aa = ambit::Tensor::build(ambit::kCore,"L2aa",{na_, na_, na_, na_});
    L2ab = ambit::Tensor::build(ambit::kCore,"L2ab",{na_, na_, na_, na_});
    L2bb = ambit::Tensor::build(ambit::kCore,"L2bb",{na_, na_, na_, na_});

    L3aaa_ = d6(na_, d5(na_, d4(na_, d3(na_, d2(na_, d1(na_))))));
    L3aab_ = d6(na_, d5(na_, d4(na_, d3(na_, d2(na_, d1(na_))))));
    L3abb_ = d6(na_, d5(na_, d4(na_, d3(na_, d2(na_, d1(na_))))));
    L3bbb_ = d6(na_, d5(na_, d4(na_, d3(na_, d2(na_, d1(na_))))));
    L3aaa = ambit::Tensor::build(ambit::kCore,"L3aaa", {na_, na_, na_, na_, na_, na_});
    L3aab = ambit::Tensor::build(ambit::kCore,"L3aab", {na_, na_, na_, na_, na_, na_});
    L3abb = ambit::Tensor::build(ambit::kCore,"L3abb", {na_, na_, na_, na_, na_, na_});
    L3bbb = ambit::Tensor::build(ambit::kCore,"L3bbb", {na_, na_, na_, na_, na_, na_});
}

void FCI_MO::print_title(){
    outfile->Printf("\n");
    outfile->Printf("\n  ***************************************************");
    outfile->Printf("\n  * Complete Active Space Configuration Interaction *");
    outfile->Printf("\n  *                 by Chenyang Li                  *");
    outfile->Printf("\n  ***************************************************");
    outfile->Printf("\n");
    outfile->Flush();
}

void FCI_MO::read_info(Options &options){

    boost::shared_ptr<Wavefunction> wfn = Process::environment.wavefunction();
    boost::shared_ptr<Molecule> molecule = Process::environment.molecule();

    // Reference type
    ref_type_ = options.get_str("REFERENCE");
    if(ref_type_ == "UHF" || ref_type_ == "UKS" || ref_type_ == "CUHF"){
        outfile->Printf("\n  Unrestricted reference is detected.");
        if(!options.get_bool("UNO")){
            outfile->Printf("\n  Warning! Warning! Warning! Warning!");
            outfile->Printf("\n  We suggest using unrestricted natural orbitals.");
            outfile->Printf("\n  Otherwise, semi-canonicalization will fail for beta spin.");
            outfile->Printf("\n  Unrestricted natural orbitals can be computed by setting \"UNO\" option to \"true\".");
        }else{
            outfile->Printf("\n  Unrestricted natural orbitals are employed. Good Choice!");
        }
    }

    // Print Level
    print_ = options.get_int("PRINT");

    // Energy convergence
    econv_ = options.get_double("E_CONVERGENCE");

    // Nuclear Repulsion
    e_nuc_ = molecule->nuclear_repulsion_energy();

    // Number of Irrep
    nirrep_ = wfn->nirrep();

    // MOs
    nmo_ = wfn->nmo();
    nmopi_ = wfn->nmopi();
    ncmo_ = integral_->ncmo();
    ncmopi_ = integral_->ncmopi();

    // Frozen Orbitals
    frzcpi_ = integral_->frzcpi();
    frzvpi_ = integral_->frzvpi();
    nfrzc_ = frzcpi_.sum();
    nfrzv_ = frzvpi_.sum();

    // Core and Active
    if(options["ACTIVE"].size() == 0){
        outfile->Printf("\n  Please specify the ACTIVE occupations.");
        outfile->Printf("\n  Single-reference computations should set ACTIVE to zeros.");
        outfile->Printf("\n  For example, ACTIVE [0,0,0,0] depending on the symmetry. \n");
        throw PSIEXCEPTION("Please specify the ACTIVE occupations. Check output for details.");
    }
    core_ = Dimension (nirrep_, "Core MOs");
    active_ = Dimension (nirrep_, "Active MOs");
    virtual_ = Dimension (nirrep_, "Virtual MOs");
    if (options["RESTRICTED_DOCC"].size() == nirrep_ && options["ACTIVE"].size() == nirrep_){
        for (int h=0; h<nirrep_; ++h){
            core_[h] = options["RESTRICTED_DOCC"][h].to_integer();
            active_[h] = options["ACTIVE"][h].to_integer();
            virtual_[h] = ncmopi_[h] - core_[h] - active_[h];
        }
    }else{
        outfile->Printf("\n  The size of RESTRICTED_DOCC or ACTIVE occupation does not match the number of Irrep.");
        outfile->Printf("\n  Number of irreps: %2d", nirrep_);
        outfile->Printf("\n  Size of RESTRICTED_DOCC: %2d", options["RESTRICTED_DOCC"].size());
        outfile->Printf("\n  Size of ACTIVE: %2d", options["ACTIVE"].size());
        outfile->Printf("\n  Check RESTRICTED_DOCC and ACTIVE! \n");
        throw PSIEXCEPTION("Wrong RESTRICTED_DOCC or ACTIVE. Check output for details.");
    }
    nc_ = core_.sum();
    na_ = active_.sum();
    nv_ = virtual_.sum();

    // Number of Electrons and Orbitals
    int natom = molecule->natom();
    size_t nelec = 0;
    for(int i=0; i<natom; ++i){
        nelec += molecule->fZ(i);
    }
    int charge = molecule->molecular_charge();
    if(options["CHARGE"].has_changed()){
        charge = options.get_int("CHARGE");
    }
    nelec -= charge;
    multi_ = molecule->multiplicity();
    if(options["MULTI"].has_changed()){
        multi_ = options.get_int("MULTI");
    }
    if(multi_ < 1){
        outfile->Printf("\n  MULTI must be no less than 1.");
        outfile->Printf("\n  MULTI = %2d", multi_);
        outfile->Printf("\n  Check (specify) Multiplicity! \n");
        throw PSIEXCEPTION("MULTI must be no less than 1. Check output for details.");
    }
    ms_ = options.get_int("MS");
    if(ms_ < 0){
        outfile->Printf("\n  Ms must be no less than 0.");
        outfile->Printf("\n  Ms = %2d, MULTI = %2d", ms_, multi_);
        outfile->Printf("\n  Check (specify) Ms value (component of multiplicity)! \n");
        throw PSIEXCEPTION("Ms must be no less than 0. Check output for details.");
    }
    nalfa_ = (nelec + ms_) / 2;
    nbeta_ = (nelec - ms_) / 2;
    if(nalfa_ < 0 || nbeta_ < 0 || (nalfa_ + nbeta_) != nelec ){
        outfile->Printf("\n  Number of alpha electrons or beta electrons is negative.");
        outfile->Printf("\n  Nalpha = %5ld, Nbeta = %5ld", nalfa_, nbeta_);
        outfile->Printf("\n  Charge = %3d, Multi = %3d, Ms = %.1f", charge, multi_, ms_ / 2.0);
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

    // Root Symmetry
    root_sym_ = options.get_int("ROOT_SYM");

    // Number of roots and root of interest
    nroot_ = options.get_int("NROOT");
    root_ = options.get_int("ROOT");
    if(root_ >= nroot_){
        outfile->Printf("\n  NROOT = %3d, ROOT = %3d", nroot_, root_);
        outfile->Printf("\n  ROOT must be smaller than NROOT.");
        throw PSIEXCEPTION("ROOT must be smaller than NROOT.");
    }

    // Symmetry Index of Active Orbitals
    for(int h=0; h<nirrep_; ++h){
        for(size_t i=0; i<active_[h]; ++i){
            sym_active_.push_back(h);
        }
    }

    // Symmetry Index of Correlated Orbitals
    for(int h=0; h<nirrep_; ++h){
        for(size_t i=0; i<ncmopi_[h]; ++i){
            sym_ncmo_.push_back(h);
        }
    }

    // Index of Core, Active and Virtual
    int ncmopi = 0;
    for(int h=0; h<nirrep_; ++h){
        size_t c = core_[h];
        size_t ca = core_[h] + active_[h];
        for(size_t i=0; i<ncmopi_[h]; ++i){
            size_t idx = i + ncmopi;
            if(i < c)
                idx_c_.push_back(idx);
            if(i >= c && i < ca)
                idx_a_.push_back(idx);
            if(i >= ca)
                idx_v_.push_back(idx);
        }
        ncmopi += ncmopi_[h];
    }

    // Hole and Particle Index
    nh_ = nc_ + na_;
    npt_ = na_ + nv_;
    idx_h_ = vector<size_t> (idx_a_);
    idx_h_.insert(idx_h_.end(), idx_c_.begin(), idx_c_.end());
    idx_p_ = vector<size_t> (idx_a_);
    idx_p_.insert(idx_p_.end(), idx_v_.begin(), idx_v_.end());

    // Print
    options.print();
    std::vector<std::pair<std::string,size_t>> info;
    info.push_back({"number of atoms", natom});
    info.push_back({"number of electrons", nelec});
    info.push_back({"molecular charge", charge});
    info.push_back({"number of alpha electrons", nalfa_});
    info.push_back({"number of beta electrons", nbeta_});
    info.push_back({"multiplicity", multi_});
    info.push_back({"number of molecular orbitals", nmo_});

    outfile->Printf("\n  ==> Input Summary <==\n");
    for (auto& str_dim: info){
        outfile->Printf("\n    %-30s = %5zu",str_dim.first.c_str(),str_dim.second);
    }

    outfile->Printf("\n\n  ==> Orbital Spaces <==\n");
    print_irrep("TOTAL MO", nmopi_);
    print_irrep("FROZEN CORE", frzcpi_);
    print_irrep("FROZEN VIRTUAL", frzvpi_);
    print_irrep("CORRELATED MO", ncmopi_);
    print_irrep("CORE", core_);
    print_irrep("ACTIVE", active_);
    print_irrep("VIRTUAL", virtual_);

    outfile->Printf("\n\n  ==> Correlated Subspace Indices <==\n");
    print_idx("CORE", idx_c_);
    print_idx("ACTIVE", idx_a_);
    print_idx("HOLE", idx_h_);
    print_idx("VIRTUAL", idx_v_);
    print_idx("PARTICLE", idx_p_);
    outfile->Printf("\n");
    outfile->Flush();

}

void FCI_MO::form_det(){

    // Number of alpha and beta electrons in active
    int na_a = nalfa_ - nc_ - nfrzc_;
    int nb_a = nbeta_ - nc_ - nfrzc_;

    // Alpha and Beta Strings
    Timer tstrings;
    std::string str = "Forming alpha and beta strings";
    outfile->Printf("\n  %-35s ...", str.c_str());
    vector<vector<vector<bool>>> a_string = Form_String(na_a,0);
    vector<vector<vector<bool>>> b_string = Form_String(nb_a,0);
    outfile->Printf("  Done. Timing %15.6f s", tstrings.get());

    // Form Determinant
    Timer tdet;
    str = "Forming determinants";
    outfile->Printf("\n  %-35s ...", str.c_str());
    StringDeterminant::set_ints(integral_);
    for(int i = 0; i != nirrep_; ++i){
        int j = i ^ root_sym_;
        size_t sa = a_string[i].size();
        size_t sb = b_string[j].size();
        for(size_t alfa = 0; alfa < sa; ++alfa){
            for(size_t beta = 0; beta < sb; ++beta){
                determinant_.push_back(StringDeterminant(a_string[i][alfa], b_string[j][beta]));
            }
        }
    }
    outfile->Printf("  Done. Timing %15.6f s", tdet.get());

    // Print
    std::vector<std::pair<std::string,size_t>> info;
    info.push_back({"number of alpha active electrons", na_a});
    info.push_back({"number of beta active electrons", nb_a});
    info.push_back({"root symmetry (zero based)", root_sym_});
    info.push_back({"number of determinants", determinant_.size()});

    outfile->Printf("\n\n  ==> Determinants Summary <==\n");
    for (auto& str_dim: info){
        outfile->Printf("\n    %-35s = %5zu",str_dim.first.c_str(),str_dim.second);
    }
    if(print_ > 2)  print_det(determinant_);
    outfile->Printf("\n");
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

    // Symmetry of core
    int symmetry = 0;
    for(int h=0; h<nirrep_; ++h){
        for(int i=0; i<core_[h]; ++i){
            symmetry ^= h;
        }
    }

    // Initalize the String (only active)
    bool *I_init = new bool[na_];
    for(size_t i=0; i<na_; ++i) I_init[i] = 0;
    for(size_t i=na_-active_elec; i<na_; ++i)  I_init[i] = 1;

    do{
        // Permutation the Active
        vector<bool> string_a;
        int sym = symmetry;
        for(size_t i=0; i<na_; ++i){
            string_a.push_back(I_init[i]);
            if(I_init[i] == 1){
                sym ^= sym_active_[i];
            }
        }

        // Form the String with dimension nmo_
        vector<bool> str(ncmo_, 0);
        size_t shift_a = 0;
        size_t shift_c = 0;
        size_t shift_sa = 0;
        for(int h=0; h<nirrep_; ++h){
            for(size_t c=0; c<core_[h]; ++c){
                str[c+shift_c] = 1;
            }
            shift_a = shift_c + core_[h];
            for(size_t a=0; a<active_[h]; ++a){
                str[a+shift_a] = string_a[a+shift_sa];
            }
            shift_c += ncmopi_[h];
            shift_sa += active_[h];
        }
        String[sym].push_back(str);
    }while(next_permutation(I_init, I_init+na_));

    if(print == true){
        outfile->Printf("\n\n  Possible String \n");
        for(size_t i=0; i != String.size(); ++i){
            outfile->Printf("\n  symmetry = %lu \n", i);
            for(size_t j=0; j != String[i].size(); ++j){
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

void FCI_MO::semi_canonicalize(){
    boost::shared_ptr<Wavefunction> wfn = Process::environment.wavefunction();

    outfile->Printf("\n  Use semi-canonical orbitals.\n");
    SharedMatrix Ua (new Matrix("Unitary A", nmopi_, nmopi_));
    SharedMatrix Ub (new Matrix("Unitary B", nmopi_, nmopi_));
    BD_Fock(Fa_,Fb_,Ua,Ub);
    SharedMatrix Ca = wfn->Ca();
    SharedMatrix Cb = wfn->Cb();
    SharedMatrix Ca_new(Ca->clone());
    SharedMatrix Cb_new(Cb->clone());
    Ca_new->gemm(false,false,1.0,Ca,Ua,0.0);
    Cb_new->gemm(false,false,1.0,Cb,Ub,0.0);
    Ca->copy(Ca_new);
    Cb->copy(Cb_new);

    integral_->retransform_integrals();

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
    Store_CI(nroot_, print_CI_threshold, eigen_, determinant_);

    // Form Density
    Da_ = d2(ncmo_, d1(ncmo_));
    Db_ = d2(ncmo_, d1(ncmo_));
    L1a = ambit::Tensor::build(ambit::kCore,"L1a", {na_, na_});
    L1b = ambit::Tensor::build(ambit::kCore,"L1b", {na_, na_});
    FormDensity(determinant_, root_, Da_, Db_);
    if(print_ > 1){
        print_d2("Da", Da_);
        print_d2("Db", Db_);
    }

    // Fock Matrix
    size_t count = 0;
    Fa_ = d2(ncmo_, d1(ncmo_));
    Fb_ = d2(ncmo_, d1(ncmo_));
    Form_Fock(Fa_,Fb_);
    Check_Fock(Fa_,Fb_,econv_,count);
    if(print_ > 1){
        print_d2("Fa", Fa_);
        print_d2("Fb", Fb_);
    }
}

void FCI_MO::Diagonalize_H(const vecdet &det, vector<pair<SharedVector, double>> &eigen){
    timer_on("Diagonalize H");
    Timer tdiagH;
    std::string str = "Diagonalizing Hamiltonian";
    outfile->Printf("\n  %-35s ...", str.c_str());
    size_t det_size = det.size();
    eigen.clear();

    BitsetDeterminant::set_ints(integral_);
    std::vector<BitsetDeterminant> P_space;
    for(size_t x = 0; x != det_size; ++x){
        std::vector<bool> alfa_bits = det[x].get_alfa_bits_vector_bool();
        std::vector<bool> beta_bits = det[x].get_beta_bits_vector_bool();
        BitsetDeterminant bs_det(alfa_bits,beta_bits);
        P_space.push_back(bs_det);
    }
    SparseCISolver sparse_solver;
    int nroot = det_size < 25 ? det_size : 25;
    SharedMatrix vec_tmp;
    SharedVector val_tmp;
    sparse_solver.diagonalize_hamiltonian(P_space,val_tmp,vec_tmp,nroot,DavidsonLiuList);

    // Check spin
    int count = 0;
    outfile->Printf("\n\n  Reference type: %s", ref_type_.c_str());
    double threshold = 1.0e-4;
    if(ref_type_ == "UHF" || ref_type_ == "UKS" || ref_type_ == "CUHF"){
        threshold = 0.10 * multi_;    // 10% off from the multiplicity of the spin eigen state
    }
    outfile->Printf("\n  Threshold for spin check: %.4f", threshold);

    for (int i = 0; i != nroot; ++i){
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
            outfile->Printf("\n\n  Ask for S^2 = %.4f, this S^2 = %.4f, continue searching...", 0.25 * (multi_ * multi_ - 1.0), S2);
            continue;
        }
        else{
            std::vector<string> s2_labels({"singlet","doublet","triplet","quartet","quintet","sextet","septet","octet","nonet","decaet"});
            std::string state_label = s2_labels[std::round(S * 2.0)];
            outfile->Printf("\n\n  Spin State: S^2 = %5.3f, S = %5.3f, %s (from %zu determinants)",S2,S,state_label.c_str(),det_size);
            ++count;
            eigen.push_back(make_pair(vec_tmp->get_column(0,i), val_tmp->get(i) + e_nuc_));
        }
        if(count == nroot_) break;
    }
    outfile->Printf("  Done. Timing %15.6f s", tdiagH.get());
    timer_off("Diagonalize H");
}

inline bool ReverseAbsSort(const tuple<double, int> &lhs, const tuple<double, int> &rhs){
    return abs(get<0>(rhs)) < abs(get<0>(lhs));
}

void FCI_MO::Store_CI(const int &nroot, const double &CI_threshold, const vector<pair<SharedVector, double>> &eigen, const vecdet &det){

    timer_on("STORE CI Vectors");
    outfile->Printf("\n\n  * * * * * * * * * * * * * * * * *");
    outfile->Printf("\n  *  CI Vectors & Configurations  *");
    outfile->Printf("\n  * * * * * * * * * * * * * * * * *");
    outfile->Printf("\n");
    outfile->Flush();

    for(int i = 0; i != nroot; ++i){
        vector<tuple<double, int>> ci_selec;

        for(size_t j = 0; j < det.size(); ++j){
            double value = (eigen[i].first)->get(j);
            if(std::fabs(value) > CI_threshold)
                ci_selec.push_back(make_tuple(value, j));
        }
        sort(ci_selec.begin(), ci_selec.end(), ReverseAbsSort);

        outfile->Printf("\n  ==> Root No. %d <==\n", i+1);
        for(size_t j = 0; j < ci_selec.size(); ++j){
            outfile->Printf("\n    ");
            double ci = get<0>(ci_selec[j]);
            size_t index = get<1>(ci_selec[j]);
            size_t ncmopi = 0;
            for(int h = 0; h < nirrep_; ++h){
                for(size_t k = 0; k < active_[h]; ++k){
                    size_t x = core_[h] + k + ncmopi;
                    bool a = det[index].get_alfa_bit(x);
                    bool b = det[index].get_beta_bit(x);
                    if(a == b)
                        outfile->Printf("%d", a==1 ? 2 : 0);
                    else
                        outfile->Printf("%c", a==1 ? 'a' : 'b');
                }
                if(active_[h] != 0)
                    outfile->Printf(" ");
                ncmopi += ncmopi_[h];
            }
            outfile->Printf(" %20.8f", ci);
        }
        outfile->Printf("\n\n    Total Energy:   %.15lf\n\n", eigen[i].second);
        outfile->Flush();
    }

    timer_off("STORE CI Vectors");
}

void FCI_MO::FormDensity(const vecdet &dets, const int &root, d2 &A, d2 &B){
    timer_on("FORM Density");
    Timer tdensity;
    std::string str = "Forming one-particle density";
    outfile->Printf("\n  %-35s ...", str.c_str());

    for(size_t p = 0; p < nc_; ++p){
        size_t np = idx_c_[p];
            A[np][np] = 1.0;
            B[np][np] = 1.0;
    }

    for(size_t p = 0; p < na_; ++p){
        size_t np = idx_a_[p];
        for(size_t q = p; q < na_; ++q){
            size_t nq = idx_a_[q];

            if((sym_active_[p] ^ sym_active_[q]) != 0) continue;

            size_t size = dets.size();
            for(size_t ket = 0; ket != size; ++ket){
                StringDeterminant Ja(vector<bool> (2*ncmo_)), Jb(vector<bool> (2*ncmo_));
                double a = 1.0, b = 1.0, vket = (eigen_[root].first)->get(ket);
                if(std::fabs(vket) < econv_)
                    continue;
                a *= OneOP(dets[ket],Ja,np,0,nq,0) * vket;
                b *= OneOP(dets[ket],Jb,np,1,nq,1) * vket;
                for(size_t bra = 0; bra != size; ++bra){
                    double vbra = (eigen_[root].first)->get(bra);
                    if(std::fabs(vbra) < econv_)
                        continue;
                    A[np][nq] += a * (dets[bra] == Ja) * vbra;
                    B[np][nq] += b * (dets[bra] == Jb) * vbra;
                }
            }
            A[nq][np] = A[np][nq];
            B[nq][np] = B[np][nq];
        }

    }
    fill_density();
    outfile->Printf("  Done. Timing %15.6f s", tdensity.get());
    timer_off("FORM Density");
}

double FCI_MO::OneOP(const StringDeterminant &J, StringDeterminant &Jnew, const size_t &p, const bool &sp, const size_t &q, const bool &sq){
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
        Jnew = StringDeterminant (tmp[0],tmp[1],0);
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

void FCI_MO::FormCumulant2(const vecdet &dets, const int &root, d4 &AA, d4 &AB, d4 &BB){
    timer_on("FORM 2-Cumulant");
    Timer tL2;
    std::string str = "Forming Lambda2";
    outfile->Printf("\n  %-35s ...", str.c_str());
    FormCumulant2AA(dets, root, AA, BB);
    FormCumulant2AB(dets, root, AB);
    fill_cumulant2();
    outfile->Printf("  Done. Timing %15.6f s", tL2.get());
    timer_off("FORM 2-Cumulant");
}

void FCI_MO::FormCumulant2AA(const vecdet &dets, const int &root, d4 &AA, d4 &BB){
    for(size_t p = 0; p < na_; ++p){
        size_t np = idx_a_[p];
        for(size_t q = p + 1; q < na_; ++q){
            size_t nq = idx_a_[q];
            for(size_t r = 0; r < na_; ++r){
                size_t nr = idx_a_[r];
                for(size_t s = r + 1; s < na_; ++s){
                    size_t ns = idx_a_[s];

                    if((sym_active_[p] ^ sym_active_[q] ^ sym_active_[r] ^ sym_active_[s]) != 0) continue;

                    size_t size = dets.size();
                    for(size_t ket = 0; ket != size; ++ket){
                        StringDeterminant Jaa(vector<bool> (2*ncmo_)), Jbb(vector<bool> (2*ncmo_));
                        double aa = 1.0, bb = 1.0, vket = (eigen_[root].first)->get(ket);
                        if(std::fabs(vket) < econv_) continue;
                        aa *= TwoOP(dets[ket],Jaa,np,0,nq,0,nr,0,ns,0) * vket;
                        bb *= TwoOP(dets[ket],Jbb,np,1,nq,1,nr,1,ns,1) * vket;

                        for(size_t bra = 0; bra != size; ++bra){
                            double vbra = (eigen_[root].first)->get(bra);
                            if(std::fabs(vbra) < econv_) continue;
                            AA[p][q][r][s] += aa * (dets[bra] == Jaa) * vbra;
                            BB[p][q][r][s] += bb * (dets[bra] == Jbb) * vbra;
                        }
                    }

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

void FCI_MO::FormCumulant2AB(const vecdet &dets, const int &root, d4 &AB){
    for(size_t p = 0; p < na_; ++p){
        size_t np = idx_a_[p];
        for(size_t q = 0; q < na_; ++q){
            size_t nq = idx_a_[q];
            for(size_t r = 0; r < na_; ++r){
                size_t nr = idx_a_[r];
                for(size_t s = 0; s < na_; ++s){
                    size_t ns = idx_a_[s];

                    if((sym_active_[p] ^ sym_active_[q] ^ sym_active_[r] ^ sym_active_[s]) != 0) continue;

                    size_t size = dets.size();
                    for(size_t ket = 0; ket != size; ++ket){
                        StringDeterminant Jab(vector<bool> (2*ncmo_));
                        double ab = 1.0, vket = (eigen_[root].first)->get(ket);
                        if(std::fabs(vket) < econv_) continue;
                        ab *= TwoOP(dets[ket],Jab,np,0,nq,1,nr,0,ns,1) * vket;

                        for(size_t bra = 0; bra != size; ++bra){
                            double vbra = (eigen_[root].first)->get(bra);
                            if(std::fabs(vbra) < econv_) continue;
                            AB[p][q][r][s] += ab * (dets[bra] == Jab) * vbra;
                        }
                    }
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

double FCI_MO::TwoOP(const StringDeterminant &J, StringDeterminant &Jnew, const size_t &p, const bool &sp, const size_t &q, const bool &sq, const size_t &r, const bool &sr, const size_t &s, const bool &ss){
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
        Jnew = StringDeterminant (tmp[0],tmp[1],0);
        timer_off("2PO");
        return sign;
    }else{timer_off("2PO"); return 0.0;}
}

void FCI_MO::FormCumulant3(const vecdet &dets, const int &root, d6 &AAA, d6 &AAB, d6 &ABB, d6 &BBB, string &DC){
    timer_on("FORM 3-Cumulant");
    Timer tL3;
    std::string str = "Forming Lambda3";
    outfile->Printf("\n  %-35s ...", str.c_str());
    FormCumulant3AAA(dets, root, AAA, BBB, DC);
    FormCumulant3AAB(dets, root, AAB, ABB, DC);
    fill_cumulant3();
    outfile->Printf("  Done. Timing %15.6f s", tL3.get());
    timer_off("FORM 3-Cumulant");
}

void FCI_MO::FormCumulant3AAA(const vecdet &dets, const int &root, d6 &AAA, d6 &BBB, string &DC){
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
                                size_t size = dets.size();
                                for(size_t ket = 0; ket != size; ++ket){
                                    StringDeterminant Jaaa(vector<bool> (2*ncmo_)), Jbbb(vector<bool> (2*ncmo_));
                                    double aaa = 1.0, bbb = 1.0, vket = (eigen_[root].first)->get(ket);
                                    if(std::fabs(vket) < econv_) continue;
                                    aaa *= ThreeOP(dets[ket],Jaaa,np,0,nq,0,nr,0,ns,0,nt,0,nu,0) * vket;
                                    bbb *= ThreeOP(dets[ket],Jbbb,np,1,nq,1,nr,1,ns,1,nt,1,nu,1) * vket;

                                    for(size_t bra = 0; bra != size; ++bra){
                                        double vbra = (eigen_[root].first)->get(bra);
                                        if(std::fabs(vbra) < econv_) continue;
                                        AAA[p][q][r][s][t][u] += aaa * (dets[bra] == Jaaa) * vbra;
                                        BBB[p][q][r][s][t][u] += bbb * (dets[bra] == Jbbb) * vbra;
                                    }
                                }
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

void FCI_MO::FormCumulant3AAB(const vecdet &dets, const int &root, d6 &AAB, d6 &ABB, string &DC){
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
                                size_t size = dets.size();
                                for(size_t ket = 0; ket != size; ++ket){
                                    StringDeterminant Jaab(vector<bool> (2*ncmo_)), Jabb(vector<bool> (2*ncmo_));
                                    double aab = 1.0, abb = 1.0, vket = (eigen_[root].first)->get(ket);
                                    if(std::fabs(vket) < econv_) continue;
                                    aab *= ThreeOP(dets[ket],Jaab,np,0,nq,0,nr,1,ns,0,nt,0,nu,1) * vket;
                                    abb *= ThreeOP(dets[ket],Jabb,nr,0,np,1,nq,1,nu,0,ns,1,nt,1) * vket;

                                    for(size_t bra = 0; bra != size; ++bra){
                                        double vbra = (eigen_[root].first)->get(bra);
                                        if(std::fabs(vbra) < econv_) continue;
                                        AAB[p][q][r][s][t][u] += aab * (dets[bra] == Jaab) * vbra;
                                        ABB[r][p][q][u][s][t] += abb * (dets[bra] == Jabb) * vbra;
                                    }
                                }
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
                    StringDeterminant Jaaa(vector<bool> (2*ncmo_)), Jaab(vector<bool> (2*ncmo_)), Jabb(vector<bool> (2*ncmo_)), Jbbb(vector<bool> (2*ncmo_));
                    double aaa = 1.0, aab = 1.0, abb = 1.0, bbb = 1.0, vket = (eigen_[root].first)->get(ket);;
                    aaa *= ThreeOP(dets[ket],Jaaa,np,0,nq,0,nr,0,np,0,nq,0,nr,0) * vket;
                    aab *= ThreeOP(dets[ket],Jaab,np,0,nq,0,nr,1,np,0,nq,0,nr,1) * vket;
                    abb *= ThreeOP(dets[ket],Jabb,np,0,nq,1,nr,1,np,0,nq,1,nr,1) * vket;
                    bbb *= ThreeOP(dets[ket],Jbbb,np,1,nq,1,nr,1,np,1,nq,1,nr,1) * vket;

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

double FCI_MO::ThreeOP(const StringDeterminant &J, StringDeterminant &Jnew, const size_t &p, const bool &sp, const size_t &q, const bool &sq, const size_t &r, const bool &sr, const size_t &s, const bool &ss, const size_t &t, const bool &st, const size_t &u, const bool &su){
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
        Jnew = StringDeterminant (tmp[0],tmp[1],0);
        timer_off("3PO");
        return sign;
    }else{timer_off("3PO"); return 0.0;}
}

void FCI_MO::Form_Fock(d2 &A, d2 &B){
    timer_on("Form Fock");
    boost::shared_ptr<Matrix> DaM(new Matrix("DaM", ncmo_, ncmo_));
    boost::shared_ptr<Matrix> DbM(new Matrix("DbM", ncmo_, ncmo_));
    //for(size_t p=0; p<ncmo_; ++p){
    //    for(size_t q=0; q<ncmo_; ++q){
    //        double vaa = 0.0, vab = 0.0, vba = 0.0, vbb = 0.0;
    //        for(size_t r=0; r<na_; ++r){
    //            size_t nr = idx_a_[r];
    //            for(size_t s=0; s<na_; ++s){
    //                size_t ns = idx_a_[s];
    //                vaa += integral_->aptei_aa(q,nr,p,ns) * Da_[nr][ns];
    //                vab += integral_->aptei_ab(q,nr,p,ns) * Db_[nr][ns];
    //                vba += integral_->aptei_ab(nr,q,ns,p) * Da_[nr][ns];
    //                vbb += integral_->aptei_bb(q,nr,p,ns) * Db_[nr][ns];
    //            }
    //        }
    //        for(size_t r=0; r<nc_; ++r){
    //            size_t nr = idx_c_[r];
    //            vaa += integral_->aptei_aa(q,nr,p,nr);
    //            vab += integral_->aptei_ab(q,nr,p,nr);
    //            vba += integral_->aptei_ab(nr,q,nr,p);
    //            vbb += integral_->aptei_bb(q,nr,p,nr);
    //        }
    //        A[p][q] = integral_->oei_a(p,q) + vaa + vab;
    //        B[p][q] = integral_->oei_b(p,q) + vba + vbb;
    //    }
    //}
    for (size_t m = 0; m < nc_; m++) {
        for( size_t n = 0; n < nc_; n++){
            size_t nm = idx_c_[m];
            size_t nn = idx_c_[n];
            DaM->set(nm,nn,Da_[nm][nn]);
            DbM->set(nm,nn,Db_[nm][nn]);
        }
    }
    for (size_t u = 0; u < na_; u++){
        for(size_t v = 0; v < na_; v++){
            size_t nu = idx_a_[u];
            size_t nv = idx_a_[v];
            DaM->set(nu,nv, Da_[nu][nv]);
            DbM->set(nu,nv, Db_[nu][nv]);
        }
    }
    Timer tfock;
    std::string str = "Forming generalized Fock matrix";
    outfile->Printf("\n  %-35s ...", str.c_str());
    integral_->make_fock_matrix(DaM, DbM);
    outfile->Printf("  Done. Timing %15.6f s", tfock.get());

    for(size_t p=0; p<ncmo_; ++p){
        for(size_t q=0; q<ncmo_; ++q){
            A[p][q] = integral_->get_fock_a(p,q);
            B[p][q] = integral_->get_fock_b(p,q);
        }
    }
    timer_off("Form Fock");
}

void FCI_MO::Check_Fock(const d2 &A, const d2 &B, const double &E, size_t &count){
    timer_on("Check Fock");
    Timer tfock;
    std::string str = "Checking Fock matrices (Fa, Fb)";
    outfile->Printf("\n  %-35s ...", str.c_str());
    outfile->Printf("\n  Nonzero criteria: > %.2E", E);
    Check_FockBlock(A, B, E, count, nc_, idx_c_, "CORE");
    Check_FockBlock(A, B, E, count, na_, idx_a_, "ACTIVE");
    Check_FockBlock(A, B, E, count, nv_, idx_v_, "VIRTUAL");
    str = "Done checking Fock matrices.";
    outfile->Printf("\n  %-47s", str.c_str());
    outfile->Printf("Timing %15.6f s", tfock.get());
    outfile->Printf("\n");
    outfile->Flush();
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

void FCI_MO::BD_Fock(const d2 &Fa, const d2 &Fb, SharedMatrix &Ua, SharedMatrix &Ub){
    timer_on("Block Diagonal Fock");
    Timer tbdfock;
    std::string str = "Block diagonalizing Fock matrices";
    outfile->Printf("\n  %-35s ...", str.c_str());
    size_t nc = 0, na = 0, nv = 0;
    for(int h=0; h<nirrep_; ++h){

        // No rotations for frozen orbitals
        for(size_t i=0; i<frzcpi_[h]; ++i){
            Ua->set(h,i,i,1.0);
            Ub->set(h,i,i,1.0);
        }
        for(size_t i=0; i<frzvpi_[h]; ++i){
            size_t shift = frzcpi_[h] + ncmopi_[h];
            Ua->set(h,i+shift,i+shift,1.0);
            Ub->set(h,i+shift,i+shift,1.0);
        }

        // Core
        SharedMatrix CoreA (new Matrix("Core A", core_[h], core_[h]));
        SharedMatrix CoreB (new Matrix("Core B", core_[h], core_[h]));
        SharedMatrix EvecCA (new Matrix("Evec Core A", core_[h], core_[h]));
        SharedMatrix EvecCB (new Matrix("Evec Core B", core_[h], core_[h]));
        SharedVector EvalCA (new Vector("Eval Core A", core_[h]));
        SharedVector EvalCB (new Vector("Eval Core B", core_[h]));
        for(size_t i=0; i<core_[h]; ++i){
            for(size_t j=0; j<core_[h]; ++j){
                double fa = Fa[idx_c_[i+nc]][idx_c_[j+nc]];
                double fb = Fb[idx_c_[i+nc]][idx_c_[j+nc]];
                CoreA->set(i,j,fa);
                CoreB->set(i,j,fb);
            }
        }
        // direct diagonalization (unlikely to have millions of orbitals)
        CoreA->diagonalize(EvecCA,EvalCA);
        CoreB->diagonalize(EvecCB,EvalCB);
//        EvecCA->eivprint(EvalCA);
//        EvecCB->eivprint(EvalCB);

        for(size_t i=0; i<core_[h]; ++i){
            for(size_t j=0; j<core_[h]; ++j){
                double ua = EvecCA->pointer()[i][j];
                double ub = EvecCB->pointer()[i][j];
                size_t shift = frzcpi_[h];
                Ua->set(h,i+shift,j+shift,ua);
                Ub->set(h,i+shift,j+shift,ub);
            }
        }

        // Active
        SharedMatrix ActiveA (new Matrix("Active A", active_[h], active_[h]));
        SharedMatrix ActiveB (new Matrix("Active B", active_[h], active_[h]));
        SharedMatrix EvecAA (new Matrix("Evec Active A", active_[h], active_[h]));
        SharedMatrix EvecAB (new Matrix("Evec Active B", active_[h], active_[h]));
        SharedVector EvalAA (new Vector("Eval Active A", active_[h]));
        SharedVector EvalAB (new Vector("Eval Active B", active_[h]));
        for(size_t i=0; i<active_[h]; ++i){
            for(size_t j=0; j<active_[h]; ++j){
                double fa = Fa[idx_a_[i+na]][idx_a_[j+na]];
                double fb = Fb[idx_a_[i+na]][idx_a_[j+na]];
                ActiveA->set(i,j,fa);
                ActiveB->set(i,j,fb);
            }
        }
        // direct diagonalization (unlikely to have millions of orbitals)
        ActiveA->diagonalize(EvecAA,EvalAA);
        ActiveB->diagonalize(EvecAB,EvalAB);
//        EvecAA->eivprint(EvalAA);
//        EvecAB->eivprint(EvalAB);

        for(size_t i=0; i<active_[h]; ++i){
            for(size_t j=0; j<active_[h]; ++j){
                double ua = EvecAA->get(i,j);
                double ub = EvecAB->get(i,j);
                size_t shift = frzcpi_[h] + core_[h];
                Ua->set(h,i+shift,j+shift,ua);
                Ub->set(h,i+shift,j+shift,ub);
            }
        }

        // Virtual
        size_t nvh = ncmopi_[h] - core_[h] - active_[h];
        SharedMatrix VirA (new Matrix("Virtual A", nvh, nvh));
        SharedMatrix VirB (new Matrix("Virtual B", nvh, nvh));
        SharedMatrix EvecVA (new Matrix("Evec Virtual A", nvh, nvh));
        SharedMatrix EvecVB (new Matrix("Evec Virtual B", nvh, nvh));
        SharedVector EvalVA (new Vector("Eval Virtual A", nvh));
        SharedVector EvalVB (new Vector("Eval Virtual B", nvh));
        for(size_t i=0; i<nvh; ++i){
            for(size_t j=0; j<nvh; ++j){
                double fa = Fa[idx_v_[i+nv]][idx_v_[j+nv]];
                double fb = Fb[idx_v_[i+nv]][idx_v_[j+nv]];
                VirA->set(i,j,fa);
                VirB->set(i,j,fb);
            }
        }
        // direct diagonalization (unlikely to have millions of orbitals)
        VirA->diagonalize(EvecVA,EvalVA);
        VirB->diagonalize(EvecVB,EvalVB);
//        EvecVA->eivprint(EvalVA);
//        EvecVB->eivprint(EvalVB);

        for(size_t i=0; i<nvh; ++i){
            for(size_t j=0; j<nvh; ++j){
                double ua = EvecVA->get(i,j);
                double ub = EvecVB->get(i,j);
                size_t shift = frzcpi_[h] + core_[h] + active_[h];
                Ua->set(h,i+shift,j+shift,ua);
                Ub->set(h,i+shift,j+shift,ub);
            }
        }

        nc += core_[h];
        na += active_[h];
        nv += ncmopi_[h] - core_[h] - active_[h];
    }
    outfile->Printf("  Done. Timing %15.6f s\n", tbdfock.get());
    timer_off("Block Diagonal Fock");
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

void FCI_MO::compute_ref(){
    timer_on("Compute Ref");
    Eref_ = 0.0;
    for(size_t p=0; p<nh_; ++p){
        size_t np = idx_h_[p];
        for(size_t q=0; q<nh_; ++q){
            size_t nq = idx_h_[q];
            Eref_ += (integral_->oei_a(nq,np) + Fa_[nq][np]) * Da_[np][nq];
            Eref_ += (integral_->oei_b(nq,np) + Fb_[nq][np]) * Db_[np][nq];
        }
    }
    Eref_ *= 0.5;
    for(size_t p=0; p<na_; ++p){
        size_t np = idx_a_[p];
        for(size_t q=0; q<na_; ++q){
            size_t nq = idx_a_[q];
            for(size_t r=0; r<na_; ++r){
                size_t nr = idx_a_[r];
                for(size_t s=0; s<na_; ++s){
                    size_t ns = idx_a_[s];
                    Eref_ += 0.25 * integral_->aptei_aa(np,nq,nr,ns) * L2aa_[p][q][r][s];
                    Eref_ += 0.25 * integral_->aptei_bb(np,nq,nr,ns) * L2bb_[p][q][r][s];
                    Eref_ += integral_->aptei_ab(np,nq,nr,ns) * L2ab_[p][q][r][s];
                }
            }
        }
    }
//    outfile->Printf("\n  Energy = %.15f",Eref_);
    Eref_ += e_nuc_ + integral_->frozen_core_energy();
    timer_off("Compute Ref");
}

Reference FCI_MO::reference()
{
    Reference ref(Eref_,L1a,L1b,L2aa,L2ab,L2bb,L3aaa,L3aab,L3abb,L3bbb);
    return ref;
}

}}
