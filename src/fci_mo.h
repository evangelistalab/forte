#ifndef _fci_mo_h_
#define _fci_mo_h_

#include <libqt/qt.h>
#include <libpsio/psio.hpp>
#include <libpsio/psio.h>
#include <liboptions/liboptions.h>
#include <libmints/vector.h>
#include <libmints/matrix.h>
#include <libmints/wavefunction.h>
#include <libmints/molecule.h>
#include <vector>
#include <tuple>
#include <string>

#include "integrals.h"
#include "dynamic_bitset_determinant.h"
#include "stl_bitset_determinant.h"
#include "sparse_ci_solver.h"
#include "ambit/tensor.h"
#include "reference.h"
#include "helpers.h"
#include "ci_rdms.h"

using namespace std;

using d1 = vector<double>;
using d2 = vector<d1>;
using d3 = vector<d2>;
using d4 = vector<d3>;
using d5 = vector<d4>;
using d6 = vector<d5>;
using vecdet = vector<psi::forte::STLBitsetDeterminant>;

namespace psi{ namespace forte{
class FCI_MO : public Wavefunction
{
public:

    /**
     * @brief FCI_MO Constructor
     * @param wfn The main wavefunction object
     * @param options PSI4 and FORTE options
     * @param ints ForteInegrals
     * @param mo_space_info MOSpaceInfo
     */
    FCI_MO(boost::shared_ptr<Wavefunction> wfn, Options &options, std::shared_ptr<ForteIntegrals> ints, std::shared_ptr<MOSpaceInfo> mo_space_info);

    /// Destructor
    ~FCI_MO();

    /// Compute CASCI energy
    double compute_energy();

    /// Returns the reference object
    Reference reference();

    /// Set symmetry of the root
    void set_root_sym(const int& root_sym) {root_sym_ = root_sym;}

    /// Set number of roots
    void set_nroots(const int& nroot) {nroot_ = nroot;}

    /// Set which root is preferred
    void set_root(const int& root) {root_ = root;}

    /// Use whatever orbitals passed to this code
    void use_default_orbitals(const bool& default_orbitals) {default_orbitals_ = default_orbitals;}

    /// Set to use semicanonical
    void set_semicanonical(const bool& semi) {semi_ = semi;}
    /// Quiet mode (no printing, for use with CASSCF)
    void set_quite_mode(const bool& quiet) {quiet_ = quiet;}

protected:
    /// Basic Preparation
    void startup();
    void read_options();
    void cleanup();

    /// The wavefunction pointer
    boost::shared_ptr<Wavefunction> wfn_;

    /// Integrals
    std::shared_ptr<ForteIntegrals>  integral_;
    std::string int_type_;
	std::shared_ptr<FCIIntegrals> fci_ints_;

    /// Reference Type
    std::string ref_type_;

    /// MO space info
    std::shared_ptr<MOSpaceInfo> mo_space_info_;

    /// Print Levels
    int print_;
    /// Quiet mode (Do not print anything in FCI)
    bool quiet_ = false;

    /// Nucear Repulsion Energy
    double e_nuc_;

    /// Convergence
    double econv_;
    double dconv_;

    /// Multiplicity
    int multi_;
    int ms_;

    /// Symmetry
    int nirrep_;             // number of irrep
    int root_sym_;           // root
    vector<int> sym_active_; // active MOs
    vector<int> sym_ncmo_;   // correlated MOs

    /// Molecular Orbitals
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
    Dimension active_o_;  // active occupied for incomplete active space
    vector<size_t> ao_;
    Dimension active_v_;  // active virtual for incomplete active space
    vector<size_t> av_;

    /// Number of Alpha and Beta Electrons
    long int nalfa_;
    long int nbeta_;

    /// Determinants
    void form_det();
    void form_det_cis();
    void form_det_cisd();
    vecdet determinant_;

    /// Orbital Strings
    vector<vector<vector<bool>>> Form_String(const int &active_elec, const bool &print = false);
    vector<bool> Form_String_Ref(const bool &print = false);
    vector<vector<vector<bool>>> Form_String_Singles(const vector<bool> &ref_string, const bool &print = false);
    vector<vector<vector<bool>>> Form_String_Doubles(const vector<bool> &ref_string, const bool &print = false);

    /// Choice of Roots
    int nroot_;  // number of roots
    int root_;   // which root in nroot
    Dimension nrootspi_; // number of roots per irrep

    /// Diagonalize the CASCI Hamiltonian
    vector<pair<SharedVector,double>> eigen_;
    ///The algorithm for diagonalization
    std::string diag_algorithm_;

    void Diagonalize_H(const vecdet &det, vector<pair<SharedVector,double>> &eigen);

    /// Store and Print the CI Vectors and Configurations
    void Store_CI(const int &nroot, const double &CI_threshold, const vector<pair<SharedVector,double>> &eigen, const vecdet &det);

    /// Use whatever orbitals passed to this code
    bool default_orbitals_ = false;
    /// Semi-canonicalize orbitals
    bool semi_;
    void semi_canonicalize(const size_t &count);
    /// Use natural orbitals
    void nat_orbs();

    /// Density Matrix
    d2 Da_;
    d2 Db_;
    ambit::Tensor L1a;    // only in active
    ambit::Tensor L1b;    // only in active
    void fill_density();

    /// 2-Body Density Cumulant
    d4 L2aa_;
    d4 L2ab_;
    d4 L2bb_;
    ambit::Tensor L2aa;
    ambit::Tensor L2ab;
    ambit::Tensor L2bb;
    void fill_cumulant2();

    /// 3-Body Density Cumulant
    d6 L3aaa_;
    d6 L3aab_;
    d6 L3abb_;
    d6 L3bbb_;
    ambit::Tensor L3aaa;
    ambit::Tensor L3aab;
    ambit::Tensor L3abb;
    ambit::Tensor L3bbb;
    void fill_cumulant3();

    /// Print Functions
    void print_d2(const string &str, const d2 &OnePD);
    void print2PDC(const string &str, const d4 &TwoPDC, const int &PRINT);
    void print3PDC(const string &str, const d6 &ThreePDC, const int &PRINT);

    /// Form Density Matrix
    void FormDensity(const vecdet &determinants, const int &root, d2 &A, d2 &B);
    /// Check Density Matrix
    bool CheckDensity();

    /// Form 2-Particle Density Cumulant
    void FormCumulant2(CI_RDMS &ci_rdms, const int &root, d4 &AA, d4 &AB, d4 &BB);
    void FormCumulant2AA(const vector<double> &tpdm_aa, const vector<double> &tpdm_bb, d4 &AA, d4 &BB);
    void FormCumulant2AB(const vector<double> &tpdm_ab, d4 &AB);
//    void FormCumulant2(const vecdet &determinants, const int &root, d4 &AA, d4 &AB, d4 &BB);
//    void FormCumulant2AA(const vecdet &determinants, const int &root, d4 &AA, d4 &BB);
//    void FormCumulant2AB(const vecdet &determinants, const int &root, d4 &AB);

    /// Form 3-Particle Density Cumulant
    void FormCumulant3(CI_RDMS &ci_rdms, const int &root, d6 &AAA, d6 &AAB, d6 &ABB, d6 &BBB, string &DC);
    void FormCumulant3AAA(const vector<double> &tpdm_aaa, const vector<double> &tpdm_bbb, d6 &AAA, d6 &BBB, string &DC);
    void FormCumulant3AAB(const vector<double> &tpdm_aab, const vector<double> &tpdm_abb, d6 &AAB, d6 &ABB, string &DC);
    void FormCumulant3_DIAG(const vecdet &determinants, const int &root, d6 &AAA, d6 &AAB, d6 &ABB, d6 &BBB);
//    void FormCumulant3(const vecdet &determinants, const int &root, d6 &AAA, d6 &AAB, d6 &ABB, d6 &BBB, string &DC);
//    void FormCumulant3AAA(const vecdet &determinants, const int &root, d6 &AAA, d6 &BBB, string &DC);
//    void FormCumulant3AAB(const vecdet &determinants, const int &root, d6 &AAB, d6 &ABB, string &DC);

    /// N-Particle Operator
    double OneOP(const STLBitsetDeterminant &J, STLBitsetDeterminant &Jnew, const size_t &p, const bool &sp, const size_t &q, const bool &sq);
    double TwoOP(const STLBitsetDeterminant &J, STLBitsetDeterminant &Jnew, const size_t &p, const bool &sp, const size_t &q, const bool &sq, const size_t &r, const bool &sr, const size_t &s, const bool &ss);
    double ThreeOP(const STLBitsetDeterminant &J, STLBitsetDeterminant &Jnew, const size_t &p, const bool &sp, const size_t &q, const bool &sq, const size_t &r, const bool &sr, const size_t &s, const bool &ss, const size_t &t, const bool &st, const size_t &u, const bool &su);

    /// Fock Matrix
    d2 Fa_;
    d2 Fb_;
    void Form_Fock(d2 &A, d2 &B);
    void Check_Fock(const d2 &A, const d2 &B, const double &E, size_t &count);
    void Check_FockBlock(const d2 &A, const d2 &B, const double &E, size_t &count, const size_t &dim, const vector<size_t> &idx, const string &str);
    void BD_Fock(const d2 &Fa, const d2 &Fb, SharedMatrix &Ua, SharedMatrix &Ub, const string &name);

    /// Reference Energy
    double Eref_;
    void compute_ref();

    /// Check Sign (inline functons)
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
        for(int i=0; i<n; ++i){
            if(I[i])  ++count;
        }
        timer_off("Check Sign");
        return pow(-1.0,count%2);
    }

    /// Print Size of a Array with Irrep
    void print_irrep(const string &str, const Dimension &array){
        outfile->Printf("\n    %-30s", str.c_str());
        outfile->Printf("[");
        for(int h=0; h<nirrep_; ++h){
            outfile->Printf(" %4d ", array[h]);
        }
        outfile->Printf("]");
    }

    /// Print Indices
    void print_idx(const string &str, const vector<size_t> &vec){
        outfile->Printf("\n    %-30s", str.c_str());
        size_t c = 0;
        for(size_t x: vec){
            outfile->Printf("%4zu ", x);
            ++c;
            if(c % 15 == 0) outfile->Printf("\n  %-32c", ' ');
        }
    }

    /// Print Determinants
    void print_det(const vecdet &dets){
        outfile->Printf("\n\n  ==> Determinants |alpha|beta> <==\n");
        for(STLBitsetDeterminant x: dets){
            outfile->Printf("  ");
            x.print();
        }
        outfile->Printf("\n");
    }

    /// Permutations for 3-PDC
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

#endif // _fci_mo_h_
