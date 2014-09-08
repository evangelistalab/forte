#include <cmath>
#include <libmints/wavefunction.h>
#include <libmints/molecule.h>
#include <libmints/matrix.h>
#include <vector>
#include <tuple>
#include <numeric>
#include <algorithm>
#include "fci_mo.h"

using namespace std;

namespace psi{ namespace main{

FCI_MO::FCI_MO(Options &options){
    fprintf(outfile,"\n");
    fprintf(outfile,"\n  ***************************************************");
    fprintf(outfile,"\n  *                                                 *");
    fprintf(outfile,"\n  * Complete Active Space Configuration Interaction *");
    fprintf(outfile,"\n  *                                                 *");
    fprintf(outfile,"\n  *                 by Chenyang Li                  *");
    fprintf(outfile,"\n  *                                                 *");
    fprintf(outfile,"\n  ***************************************************");
    fprintf(outfile,"\n");

    // Basic Preparation: Form Determinants
    boost::shared_ptr<Wavefunction> wfn = Process::environment.wavefunction();
//    SharedMatrix Ca = wfn->Ca();
//    SharedMatrix Cb = wfn->Cb();
//    Cb->copy(Ca);
    integral_ = new libadaptive::ExplorerIntegrals(options,false);
    startup(options);
    if(determinant_.size() == 0){
        fprintf(outfile, "\n  There is no determinant matching the conditions!");
        exit(1);
    }

    // Form and Diagonalize the CASCI Hamiltonian
    Diagonalize_H(determinant_, Evecs_, Evals_);
    Evecs_->print();
    Evals_->print();

    // Store CI Vectors in CI_vec_(vector<vector<double>>)
    int NState = options.get_int("NROOT");
    double Print_CI_Vector = options.get_double("PRINT_CI_VECTOR");
    if(NState > Evecs_->coldim()){
        fprintf(outfile, "\n  Too many states of interest! There are only %3d states that satisfy the condition!", Evecs_->coldim());
        exit(1);
    }
    Store_CI(NState, Print_CI_Vector, Evecs_, Evals_, determinant_);

    int ground_state = 0;

    // Form Density
    Da_ = d2(nmo_, d1(nmo_));
    Db_ = d2(nmo_, d1(nmo_));
    FormDensity(determinant_, CI_vec_, ground_state, Da_, Db_);
    print_d2("Da", Da_);
    print_d2("Db", Db_);

    // Fock Matrix
    int e_conv = options.get_int("E_CONV");
    size_t count = 0;
    Fa_ = d2(nmo_, d1(nmo_));
    Fb_ = d2(nmo_, d1(nmo_));
    Form_Fock(Fa_,Fb_);
    Check_Fock(Fa_,Fb_,e_conv-1,count);
    print_d2("Fa", Fa_);
    print_d2("Fb", Fb_);

    // Semi-Canonical Orbitals
    if(count != 0){
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

        delete integral_;
        integral_ = new libadaptive::ExplorerIntegrals(options,false);
        libadaptive::StringDeterminant::set_ints(integral_);

        // Form and Diagonalize the CASCI Hamiltonian
        Diagonalize_H(determinant_, Evecs_, Evals_);
        Evecs_->print();
        Evals_->print();

        // Store CI Vectors in CI_vec_(vector<vector<double>>)
        Store_CI(NState, Print_CI_Vector, Evecs_, Evals_, determinant_);

        // Form Density
        Da_ = d2(nmo_, d1(nmo_));
        Db_ = d2(nmo_, d1(nmo_));
        FormDensity(determinant_, CI_vec_, ground_state, Da_, Db_);
//        print_d2("Da", Da_);
//        print_d2("Db", Db_);

        // Fock Matrix
        count = 0;
        Fa_ = d2(nmo_, d1(nmo_));
        Fb_ = d2(nmo_, d1(nmo_));
        Form_Fock(Fa_,Fb_);
        Check_Fock(Fa_,Fb_,e_conv-1,count);
        print_d2("Fa", Fa_);
        print_d2("Fb", Fb_);
    }

    // Form 2-PDC
    L2aa_ = d4(na_, d3(na_, d2(na_, d1(na_))));
    L2ab_ = d4(na_, d3(na_, d2(na_, d1(na_))));
    L2bb_ = d4(na_, d3(na_, d2(na_, d1(na_))));
    FormCumulant2_A(determinant_, CI_vec_, ground_state, L2aa_, L2ab_, L2bb_);
    bool Print = 1;
    print2PDC("L2aa", L2aa_, Print);
    print2PDC("L2ab", L2ab_, Print);
    print2PDC("L2bb", L2bb_, Print);

    // Form 3-PDC
    L3aaa_ = d6(na_, d5(na_, d4(na_, d3(na_, d2(na_, d1(na_))))));
    L3aab_ = d6(na_, d5(na_, d4(na_, d3(na_, d2(na_, d1(na_))))));
    L3abb_ = d6(na_, d5(na_, d4(na_, d3(na_, d2(na_, d1(na_))))));
    L3bbb_ = d6(na_, d5(na_, d4(na_, d3(na_, d2(na_, d1(na_))))));
    FormCumulant3_A(determinant_, CI_vec_, ground_state, L3aaa_, L3aab_, L3abb_, L3bbb_);
    print3PDC("L3aaa", L3aaa_, Print);
    print3PDC("L3aab", L3aab_, Print);
    print3PDC("L3abb", L3abb_, Print);
    print3PDC("L3bbb", L3bbb_, Print);
}

FCI_MO::~FCI_MO()
{
    cleanup();
}

void FCI_MO::cleanup(){
//    delete integral_;
}

void FCI_MO::startup(Options &options){

    // Nuclear Repulsion
    boost::shared_ptr<Molecule> molecule = Process::environment.molecule();
    e_nuc_ = molecule->nuclear_repulsion_energy();

    // Number of Irrep
    boost::shared_ptr<Wavefunction> wfn = Process::environment.wavefunction();
    nirrep_  = wfn->nirrep();
    nmopi_ = wfn->nmopi();

    // Core and Active
    core_ = Dimension (nirrep_, "Core MOs");
    active_ = Dimension (nirrep_, "Active MOs");
    nc_ = 0;
    na_ = 0;
    for (int h=0; h<nirrep_; ++h){
        core_[h] = options["RESTRICTED_DOCC"][h].to_integer();
        active_[h] = options["ACTIVE"][h].to_integer();
        nc_ += core_[h];
        na_ += active_[h];
    }

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
    multi_ = molecule->multiplicity();
    if(options["MULTI"].has_changed()){
        multi_ = options.get_int("MULTI");
    }
    if(multi_ < 1){
        fprintf(outfile, "\n  MULTI must be no less than 1. Check Multiplicity!");
    }
    nalfa_ = (nelec - charge + multi_ -1) / 2;
    nbeta_ = (nelec - charge - multi_ + 1) / 2;
    if(nalfa_ < 0 || nbeta_ < 0){
        fprintf(outfile, "\n  Check the Charge and Multiplicity! \n");
        exit(1);
    }
    if(nalfa_ - nc_ > na_){
        fprintf(outfile, "\n  Not enough active orbitals to arrange electrons!");
        fprintf(outfile, "\n  Check core and active orbitals! \n");
        exit(1);
    }
    nmo_ = wfn->nmo();
    nv_ = nmo_ - nc_ - na_;

    state_sym_ = options.get_int("ROOT_SYM");  // Electronic State
    int na_a = nalfa_ - nc_;             // No. of a electrons in active
    int nb_a = nbeta_ - nc_;             // No. of b electrons in active

    // Symmetry Index of Active Orbitals
    for(int h=0; h<nirrep_; ++h){
        for(size_t i=0; i<active_[h]; ++i){
            sym_active_.push_back(h);
        }
    }

    // Index of Core, Active and Virtual
    int nmopi = 0;
    for(int h=0; h<nirrep_; ++h){
        size_t c = core_[h];
        size_t ca = core_[h] + active_[h];
        for(size_t i=0; i<nmopi_[h]; ++i){
            size_t idx = i + nmopi;
            if(i < c)
                idx_c_.push_back(idx);
            if(i >= c && i < ca)
                idx_a_.push_back(idx);
            if(i >= ca)
                idx_v_.push_back(idx);
        }
        nmopi += nmopi_[h];
    }

    // Hole and Particle Index
    nh_ = nc_ + na_;
    npt_ = na_ + nv_;
    idx_h_ = vector<size_t> (idx_a_);
    idx_h_.insert(idx_h_.end(), idx_c_.begin(), idx_c_.end());
    idx_p_ = vector<size_t> (idx_a_);
    idx_p_.insert(idx_p_.end(), idx_v_.begin(), idx_v_.end());

    // Alpha and Beta Strings
    vector<vector<vector<bool>>> a_string = Form_String(na_a,0);
    vector<vector<vector<bool>>> b_string = Form_String(nb_a,0);

    // Form Determinant
    libadaptive::StringDeterminant::set_ints(integral_);
    for(vector<vector<vector<bool>>>::size_type i = 0; i != a_string.size(); ++i){
        int j = i ^ state_sym_;
        for(vector<vector<bool>>::iterator it_a = a_string[i].begin(); it_a != a_string[i].end(); ++it_a){
            for(vector<vector<bool>>::iterator it_b = b_string[j].begin(); it_b != b_string[j].end(); ++it_b){
                determinant_.push_back(libadaptive::StringDeterminant(*it_a, *it_b));
            }
        }
    }

    // Print
    options.print();

    fprintf(outfile, "\n  Number of Atoms   = %10d,   Number of Electrons  = %10zu", natom, nelec);
    fprintf(outfile, "\n  Molecular Charge  = %10d,   Multiplicity         = %10d", charge, multi_);
    fprintf(outfile, "\n  Number of alpha   = %10zu,   Number of beta       = %10zu", nalfa_, nbeta_);
    fprintf(outfile, "\n  Number of MO      = %10zu,   Number of Virtual    = %10zu", nmo_, nv_);
    fprintf(outfile, "\n");

    fprintf(outfile, "\n  Irrep:");
    print_irrep("Basis", nmopi_);
    print_irrep("Core", core_);
    print_irrep("Active", active_);
    fprintf(outfile, "\n");

    fprintf(outfile, "\n  Subspace Indices:");
    print_idx("Core", idx_c_);
    print_idx("Active", idx_a_);
    print_idx("Virtual", idx_v_);
    print_idx("Hole", idx_h_);
    print_idx("Particle", idx_p_);
    fprintf(outfile, "\n");

    fprintf(outfile, "\n  State Symmetry: %d", state_sym_);
    fprintf(outfile, "\n  Number of active electrons: %5d (a = %d, b = %d)", na_a+nb_a, na_a, nb_a);
    fprintf(outfile, "\n  Number of determinants:     %5lu", determinant_.size());
    print_det(determinant_);
    fprintf(outfile, "\n");

}

vector<vector<vector<bool>>> FCI_MO::Form_String(const int& active_elec, const bool& print){

    timer_on("FORM String");
    vector<vector<vector<bool>>> String(nirrep_,vector<vector<bool>>());

    // Initalize the String
    bool *I_init = new bool[nmo_];
    for(size_t i = 0; i< nmo_; ++i) I_init[i] = 0;
    for(size_t i = 0; i < nc_; ++i) I_init[i] = 1;
    for(size_t i = nc_ + na_-active_elec; i< nc_ + na_; ++i)  I_init[i] = 1;

    do{
        vector <bool> det;
        int symmetry = 0;
        for(size_t i=0; i< nmo_; ++i){
            det.push_back(I_init[i]);
            if(I_init[i] == 1){
                symmetry ^= sym_active_[i];
            }
        }
        String[symmetry].push_back(det);
    }while(next_permutation(I_init + nc_, I_init + nc_ + na_));

    if(print == true){
        fprintf(outfile, "\n\n  Possible String \n");
        for(vector<vector<vector<bool>>>::size_type i=0; i != String.size(); ++i){
            fprintf(outfile, "\n  symmetry = %lu \n", i);
            for(vector<vector<bool>>::const_iterator iter = String[i].begin(); iter != String[i].end(); ++iter){
                fprintf(outfile, "    ");
                for(bool b: *iter){
                    fprintf(outfile, "%d ", b);
                }
                fprintf(outfile, "\n");
            }
        }
    }

    delete[] I_init;
    timer_off("FORM String");
    return String;
}

void FCI_MO::Diagonalize_H(const vecdet &det, SharedMatrix &vec, SharedVector &val){
    timer_on("Diagonalize H");
    SharedMatrix H (new Matrix("CASCI Hamiltonian", det.size(), det.size()));
    SharedMatrix S2 (new Matrix("S^2", det.size(), det.size()));
    SharedMatrix Us (new Matrix("Unitary S^2", det.size(), det.size()));
    SharedVector Vs (new Vector("Eigen values S^2", det.size()));

    SharedMatrix HU (new Matrix("Unitary CASCI Hamiltonian", det.size(), det.size()));
    SharedVector HV (new Vector("Eigen values CASCI Hamiltonian", det.size()));

    for(vector<libadaptive::StringDeterminant>::size_type bra = 0; bra < det.size(); ++bra){
        for(vector<libadaptive::StringDeterminant>::size_type ket = 0; ket < det.size(); ++ket){
            double Hvalue = det[bra].slater_rules(det[ket]);
            Hvalue += (bra == ket ? e_nuc_ : 0.0);
            H->set(bra,ket,Hvalue);
            double Svalue = det[bra].spin2(det[ket]);
            S2->set(bra,ket,Svalue);
        }
    }
    S2->diagonalize(Us, Vs);
    H->diagonalize(HU, HV);
    H->print();
    HV->print();
    S2->print();
    Us->print();
    Vs->print();

    // spin selection according to multiplicity
    size_t n = 0, start = 0;
    double s = (pow(multi_,2.0) - 1) / 4;
    for(size_t i=0; i != det.size(); ++i){
        if(fabs(Vs->get(i) - s) < 1.0e-10){
            if(n == 0) start = i;
            ++n;
        }
    }
    SharedMatrix H_ss (new Matrix("Spin selected CASCI Hamiltonian", n, n));
    SharedMatrix Us_ss (new Matrix("Spin selected unitary S^2", det.size(), n));
    for(size_t i=0; i != det.size(); ++i){
        for(size_t j=0; j < n; ++j){
            double value = Us->get(i,j+start);
            Us_ss->set(i,j,value);
        }
    }
    H_ss = Matrix::triplet(Us_ss,H,Us_ss,true,false,false);
    H_ss->print();

    SharedMatrix vec_ss (new Matrix("Spin selected CI vectors", n, n));
    val = SharedVector (new Vector("Spin selected Hamiltonian eigen values", n));
    H_ss->diagonalize(vec_ss, val);
    vec = Matrix::doublet(Us_ss,vec_ss,false,false);

    timer_off("Diagonalize H");
}

inline bool ReverseAbsSort(const tuple<double, int> &lhs, const tuple<double, int> &rhs){
    return abs(get<0>(rhs)) < abs(get<0>(lhs));
}

void FCI_MO::Store_CI(const int &Nstate, const double &CI_threshold, const SharedMatrix &Evecs, const SharedVector &Evals, const vecdet &det){

    timer_on("STORE CI Vectors");
    fprintf(outfile, "\n  * * * * * * * * * * * * * * * * *");
    fprintf(outfile, "\n  *  CI Vectors & Configurations  *");
    fprintf(outfile, "\n  * * * * * * * * * * * * * * * * *");
    fprintf(outfile, "\n");

    CI_vec_ = vector<vector<double>> (Nstate, vector<double>());
    vector<tuple<double, int>> vec_tuple;

    for(int i=0; i<Nstate; ++i){
        for(size_t j=0; j<det.size(); ++j){
            double value = Evecs->get(j,i);
            CI_vec_[i].push_back(value);
            if(fabs(value) > CI_threshold)
                vec_tuple.push_back(make_tuple(value, j));
        }
        sort(vec_tuple.begin(), vec_tuple.end(), ReverseAbsSort);

        fprintf(outfile, "\n  ==> State No. %d <==", i+1);
        fprintf(outfile, "\n");
        for(int j=0; j<vec_tuple.size(); ++j){
            fprintf(outfile, "\n    ");
            double ci = get<0>(vec_tuple[j]);
            size_t index = get<1>(vec_tuple[j]);
            size_t nmopi = 0;
            for(int h=0; h<nirrep_; ++h){
                for(size_t k=0; k<active_[h]; ++k){
                    size_t x = core_[h] + k + nmopi;
                    bool a = det[index].get_alfa_bit(x);
                    bool b = det[index].get_beta_bit(x);
                    if(a == b)
                        fprintf(outfile, "%d", a==1 ? 2 : 0);
                    else
                        fprintf(outfile, "%c", a==1 ? 'a' : 'b');
                }
                if(active_[h] != 0)
                    fprintf(outfile, " ");
                nmopi += nmopi_[h];
            }
            fprintf(outfile, " %20.8f", ci);
        }
        fprintf(outfile, "\n");
        fprintf(outfile, "\n    Total Energy:   %.15lf", Evals->get(i));
        fprintf(outfile, "\n\n");
        vec_tuple.clear();
    }
    timer_off("STORE CI Vectors");
}

void FCI_MO::FormDensity(const vector<libadaptive::StringDeterminant> &dets, const vector<vector<double>> &CI_vector, const int &state, d2 &Da, d2 &Db){

    timer_on("FORM Density");
    vector<libadaptive::StringDeterminant>::size_type size = dets.size();

    for(vector<libadaptive::StringDeterminant>::size_type bra = 0; bra != size; ++bra){
        for(vector<libadaptive::StringDeterminant>::size_type ket = 0; ket != size; ++ket){

            double value = CI_vector[state][bra] * CI_vector[state][ket];

            libadaptive::StringDeterminant I = dets[bra];
            libadaptive::StringDeterminant J = dets[ket];
            bool *Ia = I.get_alfa_bits();
            bool *Ib = I.get_beta_bits();
            bool *Ja = J.get_alfa_bits();
            bool *Jb = J.get_beta_bits();

            int nadiff = 0;
            int nbdiff = 0;
            // Count how many differences in mos are there and the number of alpha/beta electrons
            for(vector<size_t>::size_type i = 0; i != idx_a_.size(); ++i){
                size_t n = idx_a_[i];
                if (Ia[n] != Ja[n]) ++nadiff;
                if (Ib[n] != Jb[n]) ++nbdiff;
            }
            nadiff /= 2;
            nbdiff /= 2;
            if((nadiff > 1) || (nbdiff > 1))
                continue;

            // Case 1: PhiI = PhiJ
            if ((nadiff == 0) and (nbdiff == 0)) {
                size_t nmopi = 0;
                for(int h=0; h<nirrep_; ++h){
                    for(size_t i=0; i<(core_[h]+active_[h]); ++i){
                        size_t n = i + nmopi;
                        if (Ia[n])  Da[n][n] += value;
                        if (Ib[n])  Db[n][n] += value;
                    }
                    nmopi += nmopi_[h];
                }
            }

            // Case 2: PhiI = a+ i PhiJ
            size_t i = 0, a = 0;
            if ((nadiff == 1) and (nbdiff == 0)) {
                for(vector<size_t>::size_type m = 0; m != idx_a_.size(); ++m){
                    size_t n = idx_a_[m];
                    if ((Ia[n] != Ja[n]) and Ja[n]) i = n;
                    if ((Ia[n] != Ja[n]) and Ia[n]) a = n;
                }
                Da[i][a] += value * CheckSign(Ia,a) * CheckSign(Ja,i);
            }
            if ((nadiff == 0) and (nbdiff == 1)) {
                for(vector<size_t>::size_type m = 0; m != idx_a_.size(); ++m){
                    size_t n = idx_a_[m];
                    if ((Ib[n] != Jb[n]) and Jb[n]) i = n;
                    if ((Ib[n] != Jb[n]) and Ib[n]) a = n;
                }
                Db[i][a] += value * CheckSign(Ib,a) * CheckSign(Jb,i);
            }
        }
    }
    timer_off("FORM Density");
}

void FCI_MO::print_d2(const string &str, const d2 &OnePD){
    timer_on("PRINT Density");
//    size_t count = 0;
    SharedMatrix M (new Matrix(str.c_str(), OnePD.size(), OnePD[0].size()));
    for(vector<vector<double>>::size_type i = 0; i != OnePD.size(); ++i){
        for(vector<double>::size_type j = 0; j != OnePD[i].size(); ++j){
            M->pointer()[i][j] = OnePD[i][j];
//            if(fabs(OnePD[i][j]) > 1.0e-15) ++count;
        }
    }
    M->print();
//    fprintf(outfile, "  Number of Nonzero Elements: %zu", count);
//    fprintf(outfile, "\n\n");
    timer_off("PRINT Density");
}

void FCI_MO::FormCumulant2_A(const vecdet &dets, const vector<vector<double>> &CI_vector, const int &state, d4 &AA, d4 &AB, d4 &BB){
    timer_on("FORM 2-Cumulant");
    for(size_t p=0; p<na_; ++p){
        size_t np = idx_a_[p];
        for(size_t q=0; q<na_; ++q){
            size_t nq = idx_a_[q];
            for(size_t r=0; r<na_; ++r){
                size_t nr = idx_a_[r];
                for(size_t s=0; s<na_; ++s){
                    size_t ns = idx_a_[s];

                    if((sym_active_[p] ^ sym_active_[q] ^ sym_active_[r] ^ sym_active_[s]) != 0) continue;

                    vecdet::size_type size = dets.size();
                    for(vecdet::size_type ket = 0; ket != size; ++ket){
                        libadaptive::StringDeterminant Jaa(vector<bool> (2*nmo_)), Jab(vector<bool> (2*nmo_)), Jbb(vector<bool> (2*nmo_));
                        double aa = 1.0, ab = 1.0, bb = 1.0;
                        aa *= TwoOP(dets[ket],Jaa,np,0,nq,0,nr,0,ns,0) * CI_vector[state][ket];
                        ab *= TwoOP(dets[ket],Jab,np,0,nq,1,nr,0,ns,1) * CI_vector[state][ket];
                        bb *= TwoOP(dets[ket],Jbb,np,1,nq,1,nr,1,ns,1) * CI_vector[state][ket];

                        for(vecdet::size_type bra = 0; bra != size; ++bra){
                            AA[p][q][r][s] += aa * (dets[bra] == Jaa) * CI_vector[state][bra];
                            AB[p][q][r][s] += ab * (dets[bra] == Jab) * CI_vector[state][bra];
                            BB[p][q][r][s] += bb * (dets[bra] == Jbb) * CI_vector[state][bra];
                        }
                    }

                    AA[p][q][r][s] -= Da_[np][nr] * Da_[nq][ns];
                    AA[p][q][r][s] += Da_[np][ns] * Da_[nq][nr];

                    AB[p][q][r][s] -= Da_[np][nr] * Db_[nq][ns];

                    BB[p][q][r][s] -= Db_[np][nr] * Db_[nq][ns];
                    BB[p][q][r][s] += Db_[np][ns] * Db_[nq][nr];
                }
            }
        }
    }
    timer_off("FORM 2-Cumulant");
}

void FCI_MO::print2PDC(const string &str, const d4 &TwoPDC, const bool &PRINT){
    timer_on("PRINT 2-Cumulant");
    fprintf(outfile, "\n  ** %s **", str.c_str());
    size_t count = 0;
    for(vector<vector<vector<vector<double>>>>::size_type i = 0; i != TwoPDC.size(); ++i){
        for(vector<vector<vector<double>>>::size_type j = 0; j != TwoPDC[i].size(); ++j){
            for(vector<vector<double>>::size_type k = 0; k != TwoPDC[i][j].size(); ++k){
                for(vector<double>::size_type l = 0; l != TwoPDC[i][j][k].size(); ++l){
                    if(fabs(TwoPDC[i][j][k][l]) > 1.0e-15){
                        ++count;
                        if(PRINT)
                            fprintf(outfile, "\n  Lambda [%3lu][%3lu][%3lu][%3lu] = %18.15lf", i, j, k, l, TwoPDC[i][j][k][l]);
                    }
                }
            }
        }
    }
    fprintf(outfile, "\n");
    fprintf(outfile, "\n  Number of Nonzero Elements: %zu", count);
    fprintf(outfile, "\n");
    timer_off("PRINT 2-Cumulant");
}

double FCI_MO::TwoOperator(const libadaptive::StringDeterminant &I, const libadaptive::StringDeterminant &J, const size_t &p, const bool &sp, const size_t &q, const bool &sq, const size_t &r, const bool &sr, const size_t &s, const bool &ss){
    timer_on("2PO");
    vector<bool> Ia = I.get_alfa_bits_vector_bool();
    vector<bool> Ib = I.get_beta_bits_vector_bool();
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
        timer_off("2PO");
        return sign * (tmp[0] == Ia) * (tmp[1] == Ib);
    }else{timer_off("2PO"); return 0.0;}
}

double FCI_MO::TwoOP(const libadaptive::StringDeterminant &J, libadaptive::StringDeterminant &Jnew, const size_t &p, const bool &sp, const size_t &q, const bool &sq, const size_t &r, const bool &sr, const size_t &s, const bool &ss){
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
        Jnew = libadaptive::StringDeterminant (tmp[0],tmp[1],0);
        timer_off("2PO");
        return sign;
    }else{timer_off("2PO"); return 0.0;}
}

void FCI_MO::FormCumulant3_A(const vecdet &dets, const vector<vector<double> > &CI_vector, const int &state, d6 &AAA, d6 &AAB, d6 &ABB, d6 &BBB){
    timer_on("FORM 3-Cumulant");
    for(size_t p=0; p<na_; ++p){
        size_t np = idx_a_[p];
        for(size_t q=0; q<na_; ++q){
            size_t nq = idx_a_[q];
            for(size_t r=0; r<na_; ++r){
                size_t nr = idx_a_[r];
                for(size_t s=0; s<na_; ++s){
                    size_t ns = idx_a_[s];
                    for(size_t t=0; t<na_; ++t){
                        size_t nt = idx_a_[t];
                        for(size_t u=0; u<na_; ++u){
                            size_t nu = idx_a_[u];

                            if((sym_active_[p] ^ sym_active_[q] ^ sym_active_[r] ^ sym_active_[s] ^ sym_active_[t] ^ sym_active_[u]) != 0) continue;

                            vecdet::size_type size = dets.size();
                            for(vecdet::size_type ket = 0; ket != size; ++ket){
                                libadaptive::StringDeterminant Jaaa(vector<bool> (2*nmo_)), Jaab(vector<bool> (2*nmo_)), Jabb(vector<bool> (2*nmo_)), Jbbb(vector<bool> (2*nmo_));
                                double aaa = 1.0, aab = 1.0, abb = 1.0, bbb = 1.0;
                                aaa *= ThreeOP(dets[ket],Jaaa,np,0,nq,0,nr,0,ns,0,nt,0,nu,0) * CI_vector[state][ket];
                                aab *= ThreeOP(dets[ket],Jaab,np,0,nq,0,nr,1,ns,0,nt,0,nu,1) * CI_vector[state][ket];
                                abb *= ThreeOP(dets[ket],Jabb,np,0,nq,1,nr,1,ns,0,nt,1,nu,1) * CI_vector[state][ket];
                                bbb *= ThreeOP(dets[ket],Jbbb,np,1,nq,1,nr,1,ns,1,nt,1,nu,1) * CI_vector[state][ket];

                                for(vecdet::size_type bra = 0; bra != size; ++bra){
                                    AAA[p][q][r][s][t][u] += aaa * (dets[bra] == Jaaa) * CI_vector[state][bra];
                                    AAB[p][q][r][s][t][u] += aab * (dets[bra] == Jaab) * CI_vector[state][bra];
                                    ABB[p][q][r][s][t][u] += abb * (dets[bra] == Jabb) * CI_vector[state][bra];
                                    BBB[p][q][r][s][t][u] += bbb * (dets[bra] == Jbbb) * CI_vector[state][bra];
                                }
                            }

                            AAA[p][q][r][s][t][u] -= P3DDD(Da_,np,nq,nr,ns,nt,nu);
                            AAA[p][q][r][s][t][u] -= P3DC(Da_,L2aa_,p,q,r,s,t,u);

                            AAB[p][q][r][s][t][u] -= (Da_[np][ns] * Da_[nq][nt] * Db_[nr][nu] - Da_[nq][ns] * Da_[np][nt] * Db_[nr][nu]);
                            AAB[p][q][r][s][t][u] -= (Da_[np][ns] * L2ab_[q][r][t][u] - Da_[np][nt] * L2ab_[q][r][s][u]);
                            AAB[p][q][r][s][t][u] -= (Da_[nq][nt] * L2ab_[p][r][s][u] - Da_[nq][ns] * L2ab_[p][r][t][u]);
                            AAB[p][q][r][s][t][u] -= (Db_[nr][nu] * L2aa_[p][q][s][t]);

                            ABB[p][q][r][s][t][u] -= (Da_[np][ns] * Db_[nq][nt] * Db_[nr][nu] - Da_[np][ns] * Db_[nr][nt] * Db_[nq][nu]);
                            ABB[p][q][r][s][t][u] -= (Db_[nq][nt] * L2ab_[p][r][s][u] - Db_[nq][nu] * L2ab_[p][r][s][t]);
                            ABB[p][q][r][s][t][u] -= (Db_[nr][nu] * L2ab_[p][q][s][t] - Db_[nr][nt] * L2ab_[p][q][s][u]);
                            ABB[p][q][r][s][t][u] -= (Da_[np][ns] * L2bb_[q][r][t][u]);

                            BBB[p][q][r][s][t][u] -= P3DDD(Db_,np,nq,nr,ns,nt,nu);
                            BBB[p][q][r][s][t][u] -= P3DC(Db_,L2bb_,p,q,r,s,t,u);
                        }
                    }
                }
            }
        }
    }
    timer_off("FORM 3-Cumulant");
}

void FCI_MO::FormCumulant3_B(const vecdet &dets, const vector<vector<double> > &CI_vector, const int &state, d6 &AAA, d6 &AAB, d6 &ABB, d6 &BBB){
    timer_on("FORM 3-Cumulant");
    vector<vector<vector<size_t>>> vec_tuple (nirrep_, vector<vector<size_t>>());
    for(size_t p=0; p<na_; ++p){
        for(size_t q=0; q<na_; ++q){
            for(size_t r=0; r<na_; ++r){
                int sym = sym_active_[p] ^ sym_active_[q] ^ sym_active_[r];
                vector<size_t> vec;
                vec.push_back(p);
                vec.push_back(q);
                vec.push_back(r);
                vec_tuple[sym].push_back(vec);
            }
        }
    }
    for(int h=0; h<nirrep_; ++h){
        for(vector<tuple<size_t,size_t,size_t>>::size_type aop = 0; aop != vec_tuple[h].size(); ++aop){
            size_t s = vec_tuple[h][aop][0];
            size_t t = vec_tuple[h][aop][1];
            size_t u = vec_tuple[h][aop][2];
            size_t ns = idx_a_[s];
            size_t nt = idx_a_[t];
            size_t nu = idx_a_[u];

            for(vector<tuple<size_t,size_t,size_t>>::size_type cop = 0; cop != vec_tuple[h].size(); ++cop){
                size_t p = vec_tuple[h][cop][0];
                size_t q = vec_tuple[h][cop][1];
                size_t r = vec_tuple[h][cop][2];
                size_t np = idx_a_[p];
                size_t nq = idx_a_[q];
                size_t nr = idx_a_[r];

                vecdet::size_type size = dets.size();
                for(vecdet::size_type ket = 0; ket != size; ++ket){
                    libadaptive::StringDeterminant Jaaa(vector<bool> (2*nmo_)), Jaab(vector<bool> (2*nmo_)), Jabb(vector<bool> (2*nmo_)), Jbbb(vector<bool> (2*nmo_));
                    double aaa = 1.0, aab = 1.0, abb = 1.0, bbb = 1.0;
                    aaa *= ThreeOP(dets[ket],Jaaa,np,0,nq,0,nr,0,ns,0,nt,0,nu,0) * CI_vector[state][ket];
                    aab *= ThreeOP(dets[ket],Jaab,np,0,nq,0,nr,1,ns,0,nt,0,nu,1) * CI_vector[state][ket];
                    abb *= ThreeOP(dets[ket],Jabb,np,0,nq,1,nr,1,ns,0,nt,1,nu,1) * CI_vector[state][ket];
                    bbb *= ThreeOP(dets[ket],Jbbb,np,1,nq,1,nr,1,ns,1,nt,1,nu,1) * CI_vector[state][ket];

                    for(vecdet::size_type bra = 0; bra != size; ++bra){
                        AAA[p][q][r][s][t][u] += aaa * (dets[bra] == Jaaa) * CI_vector[state][bra];
                        AAB[p][q][r][s][t][u] += aab * (dets[bra] == Jaab) * CI_vector[state][bra];
                        ABB[p][q][r][s][t][u] += abb * (dets[bra] == Jabb) * CI_vector[state][bra];
                        BBB[p][q][r][s][t][u] += bbb * (dets[bra] == Jbbb) * CI_vector[state][bra];
                    }
                }

                AAA[p][q][r][s][t][u] -= P3DDD(Da_,np,nq,nr,ns,nt,nu);
                AAA[p][q][r][s][t][u] -= P3DC(Da_,L2aa_,p,q,r,s,t,u);

                AAB[p][q][r][s][t][u] -= (Da_[np][ns] * Da_[nq][nt] * Db_[nr][nu] - Da_[nq][ns] * Da_[np][nt] * Db_[nr][nu]);
                AAB[p][q][r][s][t][u] -= (Da_[np][ns] * L2ab_[q][r][t][u] - Da_[np][nt] * L2ab_[q][r][s][u]);
                AAB[p][q][r][s][t][u] -= (Da_[nq][nt] * L2ab_[p][r][s][u] - Da_[nq][ns] * L2ab_[p][r][t][u]);
                AAB[p][q][r][s][t][u] -= (Db_[nr][nu] * L2aa_[p][q][s][t]);

                ABB[p][q][r][s][t][u] -= (Da_[np][ns] * Db_[nq][nt] * Db_[nr][nu] - Da_[np][ns] * Db_[nr][nt] * Db_[nq][nu]);
                ABB[p][q][r][s][t][u] -= (Db_[nq][nt] * L2ab_[p][r][s][u] - Db_[nq][nu] * L2ab_[p][r][s][t]);
                ABB[p][q][r][s][t][u] -= (Db_[nr][nu] * L2ab_[p][q][s][t] - Db_[nr][nt] * L2ab_[p][q][s][u]);
                ABB[p][q][r][s][t][u] -= (Da_[np][ns] * L2bb_[q][r][t][u]);

                BBB[p][q][r][s][t][u] -= P3DDD(Db_,np,nq,nr,ns,nt,nu);
                BBB[p][q][r][s][t][u] -= P3DC(Db_,L2bb_,p,q,r,s,t,u);
            }
        }
    }
    timer_off("FORM 3-Cumulant");
}

void FCI_MO::print3PDC(const string &str, const d6 &ThreePDC, const bool &PRINT){
    timer_on("PRINT 3-Cumulant");
    fprintf(outfile, "\n  ** %s **", str.c_str());
    size_t count = 0;
    for(vector<vector<vector<vector<vector<vector<double>>>>>>::size_type i = 0; i != ThreePDC.size(); ++i){
        for(vector<vector<vector<vector<vector<double>>>>>::size_type j =0; j != ThreePDC[i].size(); ++j){
            for(vector<vector<vector<vector<double>>>>::size_type k = 0; k != ThreePDC[i][j].size(); ++k){
                for(vector<vector<vector<double>>>::size_type l = 0; l != ThreePDC[i][j][k].size(); ++l){
                    for(vector<vector<double>>::size_type m = 0; m != ThreePDC[i][j][k][l].size(); ++m){
                        for(vector<double>::size_type n = 0; n != ThreePDC[i][j][k][l][m].size(); ++n){
                            if(fabs(ThreePDC[i][j][k][l][m][n]) > 1.0e-15){
                                ++count;
                                if(PRINT)
                                    fprintf(outfile, "\n  Lambda [%3lu][%3lu][%3lu][%3lu][%3lu][%3lu] = %18.15lf", i, j, k, l, m, n, ThreePDC[i][j][k][l][m][n]);
                            }
                        }
                    }
                }
            }
        }
    }
    fprintf(outfile, "\n");
    fprintf(outfile, "\n  Number of Nonzero Elements: %zu", count);
    fprintf(outfile, "\n");
    timer_off("PRINT 3-Cumulant");
}

double FCI_MO::ThreeOperator(const libadaptive::StringDeterminant &I, const libadaptive::StringDeterminant &J, const size_t &p, const bool &sp, const size_t &q, const bool &sq, const size_t &r, const bool &sr, const size_t &s, const bool &ss, const size_t &t, const bool &st, const size_t &u, const bool &su){
    timer_on("3PO");
    vector<bool> Ia = I.get_alfa_bits_vector_bool();
    vector<bool> Ib = I.get_beta_bits_vector_bool();
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
        timer_off("3PO");
        return sign * (tmp[0] == Ia) * (tmp[1] == Ib);
    }else{timer_off("3PO"); return 0.0;}
}

double FCI_MO::ThreeOP(const libadaptive::StringDeterminant &J, libadaptive::StringDeterminant &Jnew, const size_t &p, const bool &sp, const size_t &q, const bool &sq, const size_t &r, const bool &sr, const size_t &s, const bool &ss, const size_t &t, const bool &st, const size_t &u, const bool &su){
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
        Jnew = libadaptive::StringDeterminant (tmp[0],tmp[1],0);
        timer_off("3PO");
        return sign;
    }else{timer_off("3PO"); return 0.0;}
}

void FCI_MO::Form_Fock(d2 &A, d2 &B){
    timer_on("Form Fock");
    for(size_t p=0; p<nmo_; ++p){
        for(size_t q=0; q<nmo_; ++q){
            double vaa = 0.0, vab = 0.0, vba = 0.0, vbb = 0.0;
            for(size_t r=0; r<nh_; ++r){
                size_t nr = idx_h_[r];
                for(size_t s=0; s<nh_; ++s){
                    size_t ns = idx_h_[s];
                    vaa += integral_->aptei_aa(q,nr,p,ns) * Da_[nr][ns];
                    vab += integral_->aptei_ab(q,nr,p,ns) * Db_[nr][ns];
                    vba += integral_->aptei_ab(nr,q,ns,p) * Da_[nr][ns];
                    vbb += integral_->aptei_bb(q,nr,p,ns) * Db_[nr][ns];
                }
            }
            A[p][q] = integral_->oei_a(p,q) + vaa + vab;
            B[p][q] = integral_->oei_b(p,q) + vba + vbb;
        }
    }
    timer_off("Form Fock");
}

void FCI_MO::Check_Fock(const d2 &A, const d2 &B, const int &E, size_t &count){
    timer_on("Check Fock");
    fprintf(outfile, "\n  Checking Fock matrices (Fa, Fb) ... ");
    fprintf(outfile, "\n  Nonzero criteria: > 1.0E-%d", E);
    Check_FockBlock(A, B, E, count, nc_, idx_c_, "core");
    Check_FockBlock(A, B, E, count, na_, idx_a_, "active");
    Check_FockBlock(A, B, E, count, nv_, idx_v_, "virtual");
    fprintf(outfile, "\n  Done checking Fock matrices.");
    fprintf(outfile, "\n");
    timer_off("Check Fock");
}

void FCI_MO::Check_FockBlock(const d2 &A, const d2 &B, const int &E, size_t &count, const size_t &dim, const vector<size_t> &idx, const string &str){
    double maxa = 0.0, maxb = 0.0;
    size_t a = 0, b = 0;
    for(size_t p=0; p<dim; ++p){
        size_t np = idx[p];
        for(size_t q=0; q<dim; ++q){
            size_t nq = idx[q];
            if(np != nq){
                if(fabs(A[np][nq]) > pow(0.1,E)){
                    ++a;
                    maxa = (fabs(A[np][nq]) > maxa) ? fabs(A[np][nq]) : maxa;
                }
                if(fabs(B[np][nq]) > pow(0.1,E)){
                    ++b;
                    maxb = (fabs(B[np][nq]) > maxb) ? fabs(B[np][nq]) : maxb;
                }
            }
        }
    }
    count += a+b;
    if(a == 0){
        fprintf(outfile, "\n  Fa_%-7s block is diagonal.", str.c_str());
    }else{
        fprintf(outfile, "\n  Warning: Fa_%-7s NOT diagonal!", str.c_str());
        fprintf(outfile, "\n  Nonzero off-diagonal: %5zu. Largest value: %18.15lf", a, maxa);
    }
    if(b == 0){
        fprintf(outfile, "\n  Fb_%-7s block is diagonal.", str.c_str());
    }else{
        fprintf(outfile, "\n  Warning: Fb_%-7s NOT diagonal!", str.c_str());
        fprintf(outfile, "\n  Nonzero off-diagonal: %5zu. Largest value: %18.15lf", b, maxb);
    }
}

void FCI_MO::BD_Fock(const d2 &Fa, const d2 &Fb, SharedMatrix &Ua, SharedMatrix &Ub){
    timer_on("Block Diagonal Fock");
    size_t nc = 0, na = 0, nv = 0;
    for(int h=0; h<nirrep_; ++h){
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
        CoreA->diagonalize(EvecCA,EvalCA);
        CoreB->diagonalize(EvecCB,EvalCB);

//        EvecCA->eivprint(EvalCA);
//        EvecCB->eivprint(EvalCB);

        for(size_t i=0; i<core_[h]; ++i){
            for(size_t j=0; j<core_[h]; ++j){
                double ua = EvecCA->pointer()[i][j];
                double ub = EvecCB->pointer()[i][j];
                Ua->set(h,i,j,ua);
                Ub->set(h,i,j,ub);
            }
        }

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
        ActiveA->diagonalize(EvecAA,EvalAA);
        ActiveB->diagonalize(EvecAB,EvalAB);

//        EvecAA->eivprint(EvalAA);
//        EvecAB->eivprint(EvalAB);

        for(size_t i=0; i<active_[h]; ++i){
            for(size_t j=0; j<active_[h]; ++j){
                double ua = EvecAA->get(i,j);
                double ub = EvecAB->get(i,j);
                size_t shift = core_[h];
                Ua->set(h,i+shift,j+shift,ua);
                Ub->set(h,i+shift,j+shift,ub);
            }
        }

        size_t nvh = nmopi_[h] - core_[h] - active_[h];
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
        VirA->diagonalize(EvecVA,EvalVA);
        VirB->diagonalize(EvecVB,EvalVB);

//        EvecVA->eivprint(EvalVA);
//        EvecVB->eivprint(EvalVB);

        for(size_t i=0; i<nvh; ++i){
            for(size_t j=0; j<nvh; ++j){
                double ua = EvecVA->get(i,j);
                double ub = EvecVB->get(i,j);
                size_t shift = core_[h] + active_[h];
                Ua->set(h,i+shift,j+shift,ua);
                Ub->set(h,i+shift,j+shift,ub);
            }
        }

        nc += core_[h];
        na += active_[h];
        nv += nmopi_[h] - core_[h] - active_[h];
    }
    timer_off("Block Diagonal Fock");
}

void FCI_MO::BD_Fock_B(const d2 &Fa, const d2 &Fb, SharedMatrix &Ua, SharedMatrix &Ub){
    timer_on("Block Diagonal Fock");
    size_t nc = 0, na = 0, nv = 0;
    SharedMatrix Fa_cp (new Matrix("Fa Copy", nmopi_, nmopi_));
    SharedMatrix Fb_cp (new Matrix("Fb Copy", nmopi_, nmopi_));
    SharedMatrix Facp (new Matrix("Fb Copy", nmo_, nmo_));
    SharedMatrix Fbcp (new Matrix("Fb Copy", nmo_, nmo_));
    for(size_t i=0; i<nmo_; ++i){
        for(size_t j=0; j<nmo_; ++j){
            Facp->set(i,j,Fa_[i][j]);
            Fbcp->set(i,j,Fb_[i][j]);
        }
    }
    Facp->print();
//    Fbcp->print();
    for(int h=0; h<nirrep_; ++h){
        size_t c = core_[h];
        size_t a = active_[h];
        size_t v = nmopi_[h] - c - a;
        for(size_t i=0; i<c; ++i){
            for(size_t j=0; j<c; ++j){
                double fa = Fa[idx_c_[i+nc]][idx_c_[j+nc]];
                double fb = Fb[idx_c_[i+nc]][idx_c_[j+nc]];
                Fa_cp->set(h,i,j,fa);
                Fb_cp->set(h,i,j,fb);
            }
        }
        for(size_t i=0; i<a; ++i){
            for(size_t j=0; j<a; ++j){
                double fa = Fa[idx_a_[i+na]][idx_a_[j+na]];
                double fb = Fb[idx_a_[i+na]][idx_a_[j+na]];
                size_t shift = c;
                Fa_cp->set(h,i+shift,j+shift,fa);
                Fb_cp->set(h,i+shift,j+shift,fb);
            }
        }
        for(size_t i=0; i<v; ++i){
            for(size_t j=0; j<v; ++j){
                double fa = Fa[idx_v_[i+nv]][idx_v_[j+nv]];
                double fb = Fb[idx_v_[i+nv]][idx_v_[j+nv]];
                size_t shift = a + c;
                Fa_cp->set(h,i+shift,j+shift,fa);
                Fb_cp->set(h,i+shift,j+shift,fb);
            }
        }
        nc += c;
        na += a;
        nv += v;
    }
    SharedVector EvalsA (new Vector("Eigen values A", nmopi_));
    SharedVector EvalsB (new Vector("Eigen values B", nmopi_));
    Fa_cp->diagonalize(Ua,EvalsA);
    Fb_cp->diagonalize(Ub,EvalsB);
    Ua->eivprint(EvalsA);
    Fa_cp->print();
    SharedMatrix Fa_diag (new Matrix("Fa Diagonal", nmopi_, nmopi_));
    SharedMatrix temp (new Matrix("Fa temp", nmopi_, nmopi_));
    temp->gemm(true,false,1.0,Ua,Fa_cp,0.0);
    Fa_diag->gemm(false,false,1.0,temp,Ua,0.0);
    Fa_diag->print();
    timer_off("Block Diagonal Fock");
}

void FCI_MO::TRANS_C(const SharedMatrix &C, const SharedMatrix &U, SharedMatrix &Cnew){
    timer_on("Transform C");
    for(int h=0; h<nirrep_; ++h){
        for(size_t i=0; i<nmopi_[h]; ++i){
            for(size_t j=0; j<nmopi_[h]; ++j){
                Cnew->set(h,i,j,0.0);
                double value = 0.0;
                for(size_t k=0; k<nmopi_[h]; ++k){
                    value += C->get(h,i,k) * U->get(h,k,j);
                }
                Cnew->set(h,i,j,value);
                fprintf(outfile, "\n  Ca_new[%zu][%zu] = %.15lf", i, j, value);
                fprintf(outfile, "\n  Ca_new[%zu][%zu] = %.15lf", i, j, Cnew->get(h,i,j));
            }
        }
    }
    timer_off("Transform C");
}

void FCI_MO::COPY(const SharedMatrix &Cnew, SharedMatrix &C){
    for(int h=0; h<nirrep_; ++h){
        for(size_t i=0; i<nmopi_[h]; ++i){
            for(size_t j=0; j<nmopi_[h]; ++j){
                double value = Cnew->get(h,i,j);
                C->set(h,i,j,value);
                fprintf(outfile, "\n  Ca[%zu][%zu] = %.15lf", i,j, value);
            }
        }
    }
}

}}
