#ifndef MCSRGPT2_MO_H
#define MCSRGPT2_MO_H

#include <liboptions/liboptions.h>
#include <libmints/vector.h>
#include <libmints/matrix.h>
#include <vector>
#include <cmath>
#include "fci_mo.h"
#include "integrals.h"

using namespace std;

typedef vector<double> d1;
typedef vector<d1> d2;
typedef vector<d2> d3;
typedef vector<d3> d4;
typedef vector<d4> d5;
typedef vector<d5> d6;

namespace psi{ namespace main{

class MCSRGPT2_MO : public FCI_MO
{
public:
    MCSRGPT2_MO(Options &options, libadaptive::ExplorerIntegrals *ints);
    ~MCSRGPT2_MO();

protected:

    void startup(Options &options);

    void cleanup();

    // DSRG s Parameter
    double s_;

    // Taylor Expansion Threshold
    int taylor_threshold_;
    int taylor_order_;

    // Reference Energy
    void compute_ref();

    // T1 Amplitude
    d2 T1a_;
    d2 T1b_;
    double T1Na_;    // Norm T1a
    double T1Nb_;    // Norm T1b
    double T1Maxa_;  // Max T1a
    double T1Maxb_;  // Max T1b

    // T2 Amplitude
    d4 T2aa_;
    d4 T2ab_;
    d4 T2bb_;
    double T2Naa_;   // Norm T2aa
    double T2Nab_;   // Norm T2ab
    double T2Nbb_;   // Norm T2bb
    double T2Maxaa_; // Max T2aa
    double T2Maxab_; // Max T2ab
    double T2Maxbb_; // Max T2bb

    // Form T Amplitudes
    void Form_T2_DSRG(d4 &AA, d4 &AB, d4 &BB, string &T_ALGOR);
    void Form_T1_DSRG(d2 &A, d2 &B);
    void Form_T2_ISA(d4 &AA, d4 &AB, d4 &BB, const double &b_const);
    void Form_T1_ISA(d2 &A, d2 &B, const double &b_const);
    void Form_T2_SELEC(d4 &AA, d4 &AB, d4 &BB);

    // Check T Amplitudes
    void Check_T1(const string &x, const d2 &M, double &Norm, double &MaxT, Options &options);
    void Check_T2(const string &x, const d4 &M, double &Norm, double &MaxT, Options &options);

    // Effective Fock Matrix
    d2 Fa_dsrg_;
    d2 Fb_dsrg_;
    void Form_Fock_DSRG(d2 &A, d2 &B, const bool &dsrgpt);

    // Effective Two Electron Integral
    d4 vaa_dsrg_;
    d4 vab_dsrg_;
    d4 vbb_dsrg_;
    void Form_APTEI_DSRG(d4 &AA, d4 &AB, d4 &BB, const bool &dsrgpt);

    // Computes the MC-SRGPT2 energy
    double compute_energy();

    // Energy Components
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

    // Taylor Expansion of [1 - exp(-s * D^2)] / D = sqrt(s) * (\sum_{n=1} \frac{1}{n!} (-1)^{n+1} Z^{2n-1})
    double Taylor_Exp(const double &Z, const int &n){
        if(n > 0){
            double value = Z, tmp = Z;
            for(int x=0; x<(n-1); ++x){
                tmp *= pow(Z,2.0) / (x+2);
                value += tmp;
            }
            return value;
        }else{return 0.0;}
    }

};
}}

#endif // MCSRGPT2_MO_H
