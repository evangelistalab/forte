#ifndef _mcsrgpt2_mo_h_
#define _mcsrgpt2_mo_h_

#include <boost/assign.hpp>
#include <liboptions/liboptions.h>
#include <libmints/vector.h>
#include <libmints/matrix.h>
#include <vector>
#include <cmath>
#include "fci_mo.h"
#include "integrals.h"

using namespace std;

using d1 = vector<double>;
using d2 = vector<d1>;
using d3 = vector<d2>;
using d4 = vector<d3>;
using d5 = vector<d4>;
using d6 = vector<d5>;

namespace psi{ namespace forte{

class MCSRGPT2_MO : public FCI_MO
{
public:
    /**
     * @brief The Constructor for the pilot DSRG-MRPT2 code
     * @param options The main options object
     * @param ints A pointer to an allocated integral object
     * @param mo_space_info The MOSpaceInfo object
     */
    MCSRGPT2_MO(boost::shared_ptr<Wavefunction> wfn, Options &options,
                std::shared_ptr<ForteIntegrals> ints, std::shared_ptr<MOSpaceInfo> mo_space_info);

    /// Destructor
    ~MCSRGPT2_MO();

protected:

    /// Source Operators
    enum sourceop{STANDARD, AMP, EMP2, LAMP, LEMP2};
    std::map<std::string, sourceop> sourcemap = boost::assign::map_list_of("STANDARD", STANDARD)
            ("AMP", AMP)("EMP2", EMP2)("LAMP", LAMP)("LEMP2", LEMP2);

    /// Basis preparation
    void startup(Options &options);

    void cleanup();

    /// DSRG s Parameter
    double s_;

    /// Source Operator
    string source_;

    /// Exponent of Delta
    double expo_delta_;

    /// Taylor Expansion Threshold
    int taylor_threshold_;
    int taylor_order_;

    /// Reference Energy
    void compute_ref();

    /// T1 Amplitude
    d2 T1a_;
    d2 T1b_;
    double T1Na_;    // Norm T1a
    double T1Nb_;    // Norm T1b
    double T1Maxa_;  // Max T1a
    double T1Maxb_;  // Max T1b

    /// T2 Amplitude
    d4 T2aa_;
    d4 T2ab_;
    d4 T2bb_;
    double T2Naa_;   // Norm T2aa
    double T2Nab_;   // Norm T2ab
    double T2Nbb_;   // Norm T2bb
    double T2Maxaa_; // Max T2aa
    double T2Maxab_; // Max T2ab
    double T2Maxbb_; // Max T2bb

    /// Form T Amplitudes
    void Form_T2_DSRG(d4 &AA, d4 &AB, d4 &BB, string &T_ALGOR);
    void Form_T1_DSRG(d2 &A, d2 &B);
    void Form_T2_ISA(d4 &AA, d4 &AB, d4 &BB, const double &b_const);
    void Form_T1_ISA(d2 &A, d2 &B, const double &b_const);
    void Form_T2_SELEC(d4 &AA, d4 &AB, d4 &BB);

    /// Check T Amplitudes
    void Check_T1(const string &x, const d2 &M, double &Norm, double &MaxT, Options &options);
    void Check_T2(const string &x, const d4 &M, double &Norm, double &MaxT, Options &options);

    /// Effective Fock Matrix
    d2 Fa_dsrg_;
    d2 Fb_dsrg_;
    void Form_Fock_DSRG(d2 &A, d2 &B, const bool &dsrgpt);

    /// Effective Two Electron Integral
    d4 vaa_dsrg_;
    d4 vab_dsrg_;
    d4 vbb_dsrg_;
    void Form_APTEI_DSRG(const bool &dsrgpt);

    /// Print Delta
    void PrintDelta();

    /// Computes the DSRG-MRPT2 energy
    double compute_energy_dsrg();

    /// Energy Components
    void E_FT1(double &E);
    void E_VT1_FT2(double &EF1, double &EF2, double &EV1, double &EV2);
    void E_VT2_2(double &E);
    void E_VT2_4PP(double &E);
    void E_VT2_4HH(double &E);
    void E_VT2_4PH(double &E);
    void E_VT2_6(double &E1, double &E2);

    double Eref_;
    double Ecorr_;
    double Etotal_;

    /// Timings
    void Print_Timing();
    Timer dsrg_timer;
    double T2_timing;
    double T1_timing;
    double FT1_timing;
    double FT2_timing;
    double VT1_timing;
    double VT2C2_timing;
    double VT2C4_timing;
    double VT2C6_timing;

    /// Compute an addition element of renorm. H according to source operator
    double ElementRH(const string& source, const double& D, const double& V);

    /// Compute an element of T according to source operator
    double ElementT(const string& source, const double& D, const double& V);

    /// Taylor Expansion of [1 - exp(-|Z|^g)] / Z = Z^{g-1} \sum_{n=1} \frac{1}{n!} (-1)^{n+1} Z^{(n-1)g})
    double Taylor_Exp(const double& Z, const int& n, const double& g){
        bool Znegative = Z < 0.0 ? 1 : 0;
        double Zcopy = Znegative ? -Z : Z;

        double value = 1, tmp = 1;
        for(int x=0; x<(n-1); ++x){
            tmp *= -1.0 * pow(Zcopy, g) / (x+2);
            value += tmp;
        }
        value *= pow(Zcopy, g - 1.0);
        return Znegative ? -value : value;
    }

    /// Taylor Expansion of [1 - exp(-|Z|)] / Z
    double Taylor_Exp_Linear(const double& Z, const int& n){
        bool Zabs = Z > 0.0 ? 1 : 0;
        if(n > 0){
            double value = 1, tmp = 1;
            for(int x=0; x<(n-1); ++x){
                tmp *= pow(-1.0, Zabs) * Z / (x+2);
                value += tmp;
            }
            return value * pow(-1.0, Zabs + 1);
        }else{return 0.0;}
    }
};
}}

#endif // _mcsrgpt2_mo_h_
