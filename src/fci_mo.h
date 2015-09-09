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
#include "bitset_determinant.h"
#include "sparse_ci_solver.h"
#include "ambit/tensor.h"
#include "reference.h"
#include "helpers.h"

using namespace std;


typedef vector<double> d1;
typedef vector<d1> d2;
typedef vector<d2> d3;
typedef vector<d3> d4;
typedef vector<d4> d5;
typedef vector<d5> d6;
typedef vector<psi::forte::StringDeterminant> vecdet;

namespace psi{ namespace forte{
class FCI_MO
{
public:
    FCI_MO(Options &options, ForteIntegrals* ints, std::shared_ptr<MOSpaceInfo> mo_space_info);
    ~FCI_MO();

    Reference reference();

protected:
    // Basic Preparation
    void print_title();
    void startup(Options &options);
    void read_info(Options &options);
    void cleanup();

    // Integrals
    ForteIntegrals* integral_;
    std::string int_type_;
	std::shared_ptr<FCIIntegrals> fci_ints_;
    // Reference Type
    std::string ref_type_;

    // MO space info
    std::shared_ptr<MOSpaceInfo> mo_space_info_;

    // Print Levels
    int print_;

    // Nucear Repulsion Energy
    double e_nuc_;

    // Multiplicity
    int multi_;
    int ms_;

    // Symmetry
    int nirrep_;             // number of irrep
    int root_sym_;           // root
    vector<int> sym_active_; // active MOs
    vector<int> sym_ncmo_;   // correlated MOs

    // Molecular Orbitals
    size_t nmo_;             // total MOs
    Dimension nmopi_;
    size_t ncmo_;            // correlated MOs
    Dimension ncmopi_;
    Dimension frzcpi_;       // frozen core
    Dimension frzvpi_;       // frozen virtual
    size_t nfrzc_;
    size_t nfrzv_;
    Dimension core_;         // core MOs
    size_t nc_;
    vector<size_t> idx_c_;
    Dimension active_;       // active MOs
    size_t na_;
    vector<size_t> idx_a_;
    size_t nv_;              // virtual MOs
    Dimension virtual_;
    vector<size_t> idx_v_;
    size_t nh_;              // hole MOs
    vector<size_t> idx_h_;
    size_t npt_;             // particle MOs
    vector<size_t> idx_p_;

    // Number of Alpha and Beta Electrons
    long int nalfa_;
    long int nbeta_;

    // Determinants
    void form_det();
    vecdet determinant_;
    vector<vector<vector<bool>>> Form_String(const int &active_elec, const bool &print);

    // Choice of Roots
    int nroot_;  // number of roots
    int root_;   // which root in nroot

    // Diagonalize the CASCI Hamiltonian
    vector<pair<SharedVector,double>> eigen_;
    std::string diag_algorithm_;
    void Diagonalize_H(const vecdet &det, vector<pair<SharedVector,double>> &eigen);

    // Store and Print the CI Vectors and Configurations
    double print_CI_threshold;
    void Store_CI(const int &nroot, const double &CI_threshold, const vector<pair<SharedVector,double>> &eigen, const vecdet &det);

    // semi-canonicalize
    void semi_canonicalize();

    // Density Matrix
    d2 Da_;
    d2 Db_;
    ambit::Tensor L1a;    // only in active
    ambit::Tensor L1b;    // only in active
    void fill_density();

    // 2-Body Density Cumulant
    d4 L2aa_;
    d4 L2ab_;
    d4 L2bb_;
    ambit::Tensor L2aa;
    ambit::Tensor L2ab;
    ambit::Tensor L2bb;
    void fill_cumulant2();

    // 3-Body Density Cumulant
    d6 L3aaa_;
    d6 L3aab_;
    d6 L3abb_;
    d6 L3bbb_;
    ambit::Tensor L3aaa;
    ambit::Tensor L3aab;
    ambit::Tensor L3abb;
    ambit::Tensor L3bbb;
    void fill_cumulant3();

    // Print Functions
    void print_d2(const string &str, const d2 &OnePD);
    void print2PDC(const string &str, const d4 &TwoPDC, const int &PRINT);
    void print3PDC(const string &str, const d6 &ThreePDC, const int &PRINT);

    // Form Density Matrix
    void FormDensity(const vecdet &determinants, const int &root, d2 &A, d2 &B);

    // Form 2-Particle Density Cumulant
    void FormCumulant2(const vecdet &determinants, const int &root, d4 &AA, d4 &AB, d4 &BB);
    void FormCumulant2AA(const vecdet &determinants, const int &root, d4 &AA, d4 &BB);
    void FormCumulant2AB(const vecdet &determinants, const int &root, d4 &AB);

    // Form 3-Particle Density Cumulant
    void FormCumulant3(const vecdet &determinants, const int &root, d6 &AAA, d6 &AAB, d6 &ABB, d6 &BBB, string &DC);
    void FormCumulant3_DIAG(const vecdet &determinants, const int &root, d6 &AAA, d6 &AAB, d6 &ABB, d6 &BBB);
    void FormCumulant3AAA(const vecdet &determinants, const int &root, d6 &AAA, d6 &BBB, string &DC);
    void FormCumulant3AAB(const vecdet &determinants, const int &root, d6 &AAB, d6 &ABB, string &DC);

    // N-Particle Operator
    double OneOP(const StringDeterminant &J, StringDeterminant &Jnew, const size_t &p, const bool &sp, const size_t &q, const bool &sq);
    double TwoOP(const StringDeterminant &J, StringDeterminant &Jnew, const size_t &p, const bool &sp, const size_t &q, const bool &sq, const size_t &r, const bool &sr, const size_t &s, const bool &ss);
    double ThreeOP(const StringDeterminant &J, StringDeterminant &Jnew, const size_t &p, const bool &sp, const size_t &q, const bool &sq, const size_t &r, const bool &sr, const size_t &s, const bool &ss, const size_t &t, const bool &st, const size_t &u, const bool &su);

    // Fock Matrix
    d2 Fa_;
    d2 Fb_;
    void Form_Fock(d2 &A, d2 &B);
    void Check_Fock(const d2 &A, const d2 &B, const double &E, size_t &count);
    void Check_FockBlock(const d2 &A, const d2 &B, const double &E, size_t &count, const size_t &dim, const vector<size_t> &idx, const string &str);
    void BD_Fock(const d2 &Fa, const d2 &Fb, SharedMatrix &Ua, SharedMatrix &Ub);

    // Reference Energy
    double econv_;
    double Eref_;
    void compute_ref();

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

    // Print Size of a Array with Irrep
    void print_irrep(const string &str, const Dimension &array){
        outfile->Printf("\n    %-30s", str.c_str());
        outfile->Printf("[");
        for(int h=0; h<nirrep_; ++h){
            outfile->Printf(" %4d ", array[h]);
        }
        outfile->Printf("]");
    }

    // Print Indices
    void print_idx(const string &str, const vector<size_t> &vec){
        outfile->Printf("\n    %-30s", str.c_str());
        size_t c = 0;
        for(size_t x: vec){
            outfile->Printf("%4zu ", x);
            ++c;
            if(c % 15 == 0) outfile->Printf("\n  %-32c", ' ');
        }
    }

    // Print Determinants
    void print_det(const vecdet &dets){
        outfile->Printf("\n\n  ==> Determinants |alpha|beta> <==\n");
        for(StringDeterminant x: dets){
            outfile->Printf("  ");
            x.print();
        }
        outfile->Printf("\n");
    }

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
