#ifndef FCI_MO_H
#define FCI_MO_H

#include <libqt/qt.h>
#include <liboptions/liboptions.h>
#include <libmints/vector.h>
#include <libmints/matrix.h>
#include <vector>
#include <tuple>
#include <string>
#include "integrals.h"
#include "string_determinant.h"

using namespace std;

typedef vector<double> d1;
typedef vector<d1> d2;
typedef vector<d2> d3;
typedef vector<d3> d4;
typedef vector<d4> d5;
typedef vector<d5> d6;
typedef vector<psi::libadaptive::StringDeterminant> vecdet;

namespace psi{ namespace main{
class FCI_MO
{
public:
    FCI_MO(Options &options, libadaptive::ExplorerIntegrals *ints);
    ~FCI_MO();

protected:
    void startup(Options &options);

    void cleanup();

    libadaptive::ExplorerIntegrals *integral_;

    int print_;

    // Nucear Repulsion Energy
    double e_nuc_;

    // Multiplicity
    int multi_;
    int ms_;

    // Number of Irrep
    int nirrep_;

    // Total MOs
    size_t nmo_;
    Dimension nmopi_;

    // Correlating MOs
    size_t ncmo_;
    Dimension ncmopi_;

    // Frozen Orbitals
    Dimension frzcpi_;
    Dimension frzvpi_;
    size_t nfrzc_;
    size_t nfrzv_;

    // Core Orbitals
    Dimension core_;
    size_t nc_;
    vector<size_t> idx_c_;

    // Active Orbitals
    Dimension active_;
    size_t na_;
    vector<size_t> idx_a_;

    // Number of Nonfrozen Virtuals
    size_t nv_;
    Dimension virtual_;
    vector<size_t> idx_v_;

    // Hole
    size_t nh_;
    vector<size_t> idx_h_;

    // Particle;
    size_t npt_;
    vector<size_t> idx_p_;

    // Number of Alpha Electrons
    long int nalfa_;

    // Number of Beta Electrons
    long int nbeta_;

    // Symmetry of the wavefunction
    int root_sym_;

    // Symmetry of Active Orbitals
    vector<int> sym_active_;

    // Print Size of a Array with Irrep
    void print_irrep(const string &str, const Dimension &array){
        outfile->Printf("\n  %-15s", str.c_str());
        for(int h=0; h<nirrep_; ++h){
            outfile->Printf("%3d", array[h]);
        }
    }

    // Print Indices
    void print_idx(const string &str, const vector<size_t> &vec){
        outfile->Printf("\n  %-15s", str.c_str());
        size_t c = 0;
        for(size_t x: vec){
            outfile->Printf("%3zu ", x);
            ++c;
            if(c % 20 == 0) outfile->Printf("\n  %-10c", ' ');
        }
    }

    // Form a String
    vector<vector<vector<bool>>> Form_String(const int &active_elec, const bool &print);

    // Proper Determinant
    vecdet determinant_;

    // Print Determinant
    void print_det(const vecdet &dets){
        outfile->Printf("\n  Determinants: |alpha|beta>");
        for(libadaptive::StringDeterminant x: dets){
            x.print();
        }
    }

    // Eigen Vectors and Values of CASCI Hamiltonian
    SharedMatrix Evecs_;
    SharedVector Evals_;

    // Diagonalize the CASCI Hamiltonian
    void Diagonalize_H(const vecdet &det, SharedMatrix &vec, SharedVector &val);

    // CI Vectors
    vector<vector<double>> CI_vec_;

    // Store and Print the CI Vectors and Configurations
    void Store_CI(const int &nroot, const double &CI_threshold, const SharedMatrix &Evecs, const SharedVector &Evals, const vecdet &det);

    // Density Matrix
    d2 Da_;
    d2 Db_;

    // 2-Body Density Cumulant
    d4 L2aa_;
    d4 L2ab_;
    d4 L2bb_;

    // 3-Body Density Cumulant
    d6 L3aaa_;
    d6 L3aab_;
    d6 L3abb_;
    d6 L3bbb_;

    // Print Functions
    void print_d2(const string &str, const d2 &OnePD);
    void print2PDC(const string &str, const d4 &TwoPDC, const int &PRINT);
    void print3PDC(const string &str, const d6 &ThreePDC, const int &PRINT);

    // Form Density Matrix
    void FormDensity(const vecdet &determinants, const vector<vector<double>> &CI_vector, const int &root, d2 &Da, d2 &Db);

    // Form 2-Particle Density Cumulant  A: Straightforward; B: Efficient
    void FormCumulant2_A(const vecdet &determinants, const vector<vector<double>> &CI_vector, const int &root, d4 &AA, d4 &AB, d4 &BB);
    void FormCumulant2_B(const vecdet &determinants, const vector<vector<double>> &CI_vector, const int &root, d4 &AA, d4 &AB, d4 &BB);

    // Form 3-Particle Density Cumulant  A: Straightforward; B: Efficient
    void FormCumulant3_A(const vecdet &determinants, const vector<vector<double>> &CI_vector, const int &root, d6 &AAA, d6 &AAB, d6 &ABB, d6 &BBB, string &DC);
    void FormCumulant3_B(const vecdet &determinants, const vector<vector<double>> &CI_vector, const int &root, d6 &AAA, d6 &AAB, d6 &ABB, d6 &BBB);

    // N-Particle Operator
    double TwoOperator(const libadaptive::StringDeterminant &I, const libadaptive::StringDeterminant &J, const size_t &p, const bool &sp, const size_t &q, const bool &sq, const size_t &r, const bool &sr, const size_t &s, const bool &ss);
    double TwoOP(const libadaptive::StringDeterminant &J, libadaptive::StringDeterminant &Jnew, const size_t &p, const bool &sp, const size_t &q, const bool &sq, const size_t &r, const bool &sr, const size_t &s, const bool &ss);
    double ThreeOperator(const libadaptive::StringDeterminant &I, const libadaptive::StringDeterminant &J, const size_t &p, const bool &sp, const size_t &q, const bool &sq, const size_t &r, const bool &sr, const size_t &s, const bool &ss, const size_t &t, const bool &st, const size_t &u, const bool &su);
    double ThreeOP(const libadaptive::StringDeterminant &J, libadaptive::StringDeterminant &Jnew, const size_t &p, const bool &sp, const size_t &q, const bool &sq, const size_t &r, const bool &sr, const size_t &s, const bool &ss, const size_t &t, const bool &st, const size_t &u, const bool &su);

    // Fock Matrix
    d2 Fa_;
    d2 Fb_;
    void Form_Fock(d2 &A, d2 &B);
    void Check_Fock(const d2 &A, const d2 &B, const int &E, size_t &count);
    void Check_FockBlock(const d2 &A, const d2 &B, const int &E, size_t &count, const size_t &dim, const vector<size_t> &idx, const string &str);
    void BD_Fock(const d2 &Fa, const d2 &Fb, SharedMatrix &Ua, SharedMatrix &Ub);
    void TRANS_C(const SharedMatrix &C, const SharedMatrix &U, SharedMatrix &Cnew);
    void COPY(const SharedMatrix &Cnew, SharedMatrix &C);

    // Check Sign (inline functons)
    double CheckSign(const vector<bool>& I, const int &n){
        timer_on("Check Sign");
        size_t count = 0;
        for(vector<bool>::const_iterator iter = I.begin(); iter != I.begin()+n; ++iter){
            if(*iter) ++count;
        }
        timer_off("Check Sign");
        return pow(-1.0,count%2);
    }
    double CheckSign(bool *I, const int &n){
        timer_on("Check Sign");
        size_t count = 0;
        for(size_t i=0; i<n; ++i){
            if(I[i])  ++count;
        }
        timer_off("Check Sign");
        return pow(-1.0,count%2);
    }

//    // Creation and Annihilate Operator
//    double AOP(vector<bool> &Ja, vector<bool> &Jb, const size_t &idx, const bool &spin){
//        double sign = 0.0;
//        if(spin == 0 && Ja[idx]) {sign = CheckSign(Ja,idx); Ja[idx] = 0;}
//        if(spin == 1 && Jb[idx]) {sign = CheckSign(Jb,idx); Jb[idx] = 0;}
//        return sign;
//    }
//    double COP(vector<bool> &Ja, vector<bool> &Jb, const size_t &idx, const bool &spin){
//        double sign = 0.0;
//        if(spin == 0 && !Ja[idx]) {sign *= CheckSign(Ja,idx); Ja[idx] = 1;}
//        if(spin == 1 && !Jb[idx]) {sign *= CheckSign(Jb,idx); Jb[idx] = 1;}
//        return sign;
//    }

    // Permutations for 3-PDC
    double P3DDD(const d2 &Density, const size_t &p, const size_t &q, const size_t &r, const size_t &s, const size_t &t, const size_t &u){
        double E = 0.0;
        int index[] = {0,1,2};
        size_t cop[] = {p,q,r};
        int count1 = 1;
        do{
            int count2 = count1 / 2;
            E += pow(-1.0,count2) * Density[cop[index[0]]][s] * Density[cop[index[1]]][t] * Density[cop[index[2]]][u];
            ++count1;
        }while(std::next_permutation(index,index+3));
        return E;
    }
    double P3DC(const d2 &Density, const d4 &Cumulant, const size_t &p, const size_t &q, const size_t &r, const size_t &s, const size_t &t, const size_t &u){
        double E = 0.0;
        int idc[] = {0,1,2};    // creation index of cop[]
        int ida[] = {0,1,2};    // annihilation index of aop[]
        size_t cop[] = {p,q,r}; // abs. creation index
        size_t aop[] = {s,t,u}; // abs. annihilation index
        int a = 1;              // a and b decide the sign
        do{
            if(a%2 == 0){
                ++a;
                continue;
            }
            int count1 = a / 2;
            int b = 1;
            do{
                if(b%2 == 0){
                    ++b;
                    continue;
                }
                int count2 = b / 2;
                size_t Didx1 = idx_a_[cop[idc[0]]];      // first index (creation) of denisty
                size_t Didx2 = idx_a_[aop[ida[0]]];      // second index (annihilation) of density
                double value = Density[Didx1][Didx2];
                value *= Cumulant[cop[idc[1]]][cop[idc[2]]][aop[ida[1]]][aop[ida[2]]];
                E += pow(-1.0,(count1+count2)) * value;
                ++b;
            }while(std::next_permutation(ida,ida+3));
            ++a;
        }while(std::next_permutation(idc,idc+3));
        return E;
    }

};
}}

#endif // FCI_MO_H
