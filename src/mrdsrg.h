#ifndef _mrdsrg_h_
#define _mrdsrg_h_

#include <cmath>
#include <boost/assign.hpp>

#include <liboptions/liboptions.h>
#include <libmints/wavefunction.h>
#include <libpsio/psio.hpp>
#include <libpsio/psio.h>
#include <ambit/blocked_tensor.h>

#include "integrals.h"
#include "reference.h"
#include "blockedtensorfactory.h"

using namespace ambit;
namespace psi{ namespace forte{

class MRDSRG : public Wavefunction
{
protected:

    // => Class initialization and termination <= //

    /// Called in the constructor
    void startup();
    /// Called in the destructor
    void cleanup();

    /// Print levels
    int print_;

    /// The wavefunction pointer
    boost::shared_ptr<Wavefunction> wfn_;

    /// The reference object
    Reference reference_;

    /// The molecular integrals required by MethodBase
    std::shared_ptr<ForteIntegrals>  ints_;

    /// MO space info
    std::shared_ptr<MOSpaceInfo> mo_space_info_;

    /// List of alpha core MOs
    std::vector<size_t> acore_mos;
    /// List of alpha active MOs
    std::vector<size_t> aactv_mos;
    /// List of alpha virtual MOs
    std::vector<size_t> avirt_mos;
    /// List of beta core MOs
    std::vector<size_t> bcore_mos;
    /// List of beta active MOs
    std::vector<size_t> bactv_mos;
    /// List of beta virtual MOs
    std::vector<size_t> bvirt_mos;

    /// Alpha core label
    std::string acore_label;
    /// Alpha active label
    std::string aactv_label;
    /// Alpha virtual label
    std::string avirt_label;
    /// Beta core label
    std::string bcore_label;
    /// Beta active label
    std::string bactv_label;
    /// Beta virtual label
    std::string bvirt_label;

    /// Map from space label to list of MOs
    std::map<char, std::vector<size_t>> label_to_spacemo;

    /// Fill up integrals
    void build_ints();
    /// Fill up density matrix and density cumulants
    void build_density();
    /// Build Fock matrix and diagonal Fock matrix elements
    void build_fock();


    // => DSRG related <= //

    /// Correlation level
    enum class CORR_LV {LDSRG2, LDSRG2_P3, PT2, PT3, QDSRG2, QDSRG2_P3};
    std::map<std::string, CORR_LV> corrlevelmap =
            boost::assign::map_list_of("LDSRG2", CORR_LV::LDSRG2)("LDSRG2_P3", CORR_LV::LDSRG2_P3)
            ("PT2", CORR_LV::PT2)("PT3", CORR_LV::PT3)
            ("QDSRG2", CORR_LV::QDSRG2)("QDSRG2_P3", CORR_LV::QDSRG2_P3);

    /// The flow parameter
    double s_;
    /// Source operator
    std::string source_;

    /// Smaller than which we will do Taylor expansion of f(z) = (1-exp(-z^2))/z
    double taylor_threshold_;
    /// Order of the Taylor expansion of f(z) = (1-exp(-z^2))/z
    int taylor_order_;

    /// Kevin's Tensor Wrapper
    std::shared_ptr<BlockedTensorFactory> BTF;
    /// Tensor type for AMBIT
    TensorType tensor_type_;

    /// One-electron integral
    ambit::BlockedTensor H;
    /// Two-electron integral
    ambit::BlockedTensor V;
    /// Generalized Fock matrix
    ambit::BlockedTensor F;
    /// One-particle density matrix
    ambit::BlockedTensor Gamma1;
    /// One-hole density matrix
    ambit::BlockedTensor Eta1;
    /// Two-body denisty cumulant
    ambit::BlockedTensor Lambda2;
    /// Three-body density cumulant
    ambit::BlockedTensor Lambda3;
    /// Single excitation amplitude
    ambit::BlockedTensor T1;
    /// Double excitation amplitude
    ambit::BlockedTensor T2;
    /// Difference of consecutive singles
    ambit::BlockedTensor DT1;
    /// Difference of consecutive doubles
    ambit::BlockedTensor DT2;

    /// Diagonal elements of Fock matrices
    std::vector<double> Fa;
    std::vector<double> Fb;

    /// Renormalize denominator
    double renormalized_denominator(double D);
    double renormalized_denominator_labs(double D);
//    double renormalized_denominator_amp(double V,double D);
//    double renormalized_denominator_emp2(double V,double D);
//    double renormalized_denominator_lamp(double V,double D);
//    double renormalized_denominator_lemp2(double V,double D);

    /// Algorithm for computing amplitudes
    std::string T_algor_;
    /// Analyze T1 and T2 amplitudes
    void analyze_amplitudes(std::string name, BlockedTensor &T1, BlockedTensor &T2);

    /// RMS of T2
    double T2rms;
    /// Norm of T2
    double T2norm;
    double t2aa_norm;
    double t2ab_norm;
    double t2bb_norm;
    /// Signed max of T2
    double T2max;
    /// Initial guess of T2
    void guess_t2(BlockedTensor& V, BlockedTensor& T2);
    /// Update T2 in every iteration
    void update_t2();
    /// Check T2 and store the largest amplitudes
    void check_t2(BlockedTensor &T2);

    /// RMS of T1
    double T1rms;
    /// Norm of T1
    double T1norm;
    double t1a_norm;
    double t1b_norm;
    /// Signed max of T1
    double T1max;
    /// Initial guess of T1
    void guess_t1(BlockedTensor& F, BlockedTensor& T2, BlockedTensor& T1);
    /// Update T1 in every iteration
    void update_t1();
    /// Check T1 and store the largest amplitudes
    void check_t1(BlockedTensor& T1);

    /// Number of amplitudes will be printed in amplitude summary
    int ntamp_;
    /// Threshold for amplitudes considered as intruders
    double intruder_tamp_;
    /// List of large amplitudes
    std::vector<std::pair<std::vector<size_t>, double>> lt1a;
    std::vector<std::pair<std::vector<size_t>, double>> lt1b;
    std::vector<std::pair<std::vector<size_t>, double>> lt2aa;
    std::vector<std::pair<std::vector<size_t>, double>> lt2ab;
    std::vector<std::pair<std::vector<size_t>, double>> lt2bb;

    /// Compute DSRG-transformed Hamiltonian Hbar
    void compute_hbar();
    /// Zero-body Hbar
    double Hbar0;
    /// One-body Hbar
    ambit::BlockedTensor Hbar1;
    /// Two-body Hbar
    ambit::BlockedTensor Hbar2;
    /// Temporary one-body Hamiltonian
    ambit::BlockedTensor O1;
    ambit::BlockedTensor C1;
    /// Temporary two-body Hamiltonian
    ambit::BlockedTensor O2;
    ambit::BlockedTensor C2;

    /// Norm of off-diagonal Hbar2
    double Hbar2od_norm(const std::vector<std::string>& blocks);
    /// Norm of off-diagonal Hbar1
    double Hbar1od_norm(const std::vector<std::string>& blocks);

    /// Compute zero-body term of commutator [H1, T1]
    void H1_T1_C0(BlockedTensor& H1, BlockedTensor& T1, const double& alpha, double& C0);
    /// Compute zero-body term of commutator [H1, T2]
    void H1_T2_C0(BlockedTensor& H1, BlockedTensor& T2, const double& alpha, double& C0);
    /// Compute zero-body term of commutator [H2, T1]
    void H2_T1_C0(BlockedTensor& H2, BlockedTensor& T1, const double& alpha, double& C0);
    /// Compute zero-body term of commutator [H2, T2]
    void H2_T2_C0(BlockedTensor& H2, BlockedTensor& T2, const double& alpha, double& C0);

    /// Compute one-body term of commutator [H1, T1]
    void H1_T1_C1(BlockedTensor& H1, BlockedTensor& T1, const double& alpha, BlockedTensor& C1);
    /// Compute one-body term of commutator [H1, T2]
    void H1_T2_C1(BlockedTensor& H1, BlockedTensor& T2, const double& alpha, BlockedTensor& C1);
    /// Compute one-body term of commutator [H2, T1]
    void H2_T1_C1(BlockedTensor& H2, BlockedTensor& T1, const double& alpha, BlockedTensor& C1);
    /// Compute one-body term of commutator [H2, T2]
    void H2_T2_C1(BlockedTensor& H2, BlockedTensor& T2, const double& alpha, BlockedTensor& C1);

    /// Compute two-body term of commutator [H2, T1]
    void H2_T1_C2(BlockedTensor& H2, BlockedTensor& T1, const double& alpha, BlockedTensor& C2);
    /// Compute two-body term of commutator [H1, T2]
    void H1_T2_C2(BlockedTensor& H1, BlockedTensor& T2, const double& alpha, BlockedTensor& C2);
    /// Compute two-body term of commutator [H2, T2]
    void H2_T2_C2(BlockedTensor& H2, BlockedTensor& T2, const double& alpha, BlockedTensor& C2);

    /// Compute MR-LDSRG(2)
    double compute_energy_ldsrg2();

    /// Zeroth-order Hamiltonian
    ambit::BlockedTensor H0th;
    /// Compute DSRG-MRPT2 energy
    double compute_energy_pt2();
    /// Compute DSRG-MRPT3 energy
    double compute_energy_pt3();
    /// Check if orbitals are semi-canonicalized
    void check_semicanonical();


    // => Reference relaxation <= //

    /// Local One-electron integral for Resetting Integrals
    ambit::BlockedTensor H_local;
    /// Local Two-electron integral for Resetting Integrals
    ambit::BlockedTensor V_local;
    /// Transfer Integrals for FCI
    void transfer_integrals();
    /// Reset Integrals to Bare Hamiltonian
    void reset_ints(BlockedTensor& H, BlockedTensor& V);
    /// Semicanonicalize orbitals
    void semi_canonicalizer();


    // => Useful printings <= //

    /// Print a summary of the options
    void print_options();
    /// Print amplitudes summary
    void print_amp_summary(const std::string& name,
                           const std::vector<std::pair<std::vector<size_t>, double>>& list,
                           const double &norm, const size_t& number_nonzero);
    /// Print intruder analysis
    void print_intruder(const std::string& name,
                        const std::vector<std::pair<std::vector<size_t>, double>>& list);
    /// Print commutator timings
    void print_comm_time();


    // => Useful Inline functions <= //

    /// Return exp(-s * D^2)
    double renormalized_exp(double D) {return std::exp(-s_ * std::pow(D, 2.0));}
    /// Taylor Expansion of [1 - exp(- Z^2)] / Z
    double Taylor_Exp(const double& Z, const int& n){
        if(n > 0){
            double value = Z, tmp = Z;
            for(int x = 0; x < n - 1; ++x){
                tmp *= -1.0 * std::pow(Z, 2.0) / (x + 2);
                value += tmp;
            }
            return value;
        }else{return 0.0;}
    }

    /// Return exp(-s * |D|)
    double renormalized_exp_linear(double D) {return std::exp(-s_ * std::fabs(D));}
    /// Taylor Expansion of [1 - exp(-|Z|)] / Z
    double Taylor_Exp_Linear(const double& Z, const int& n){
        double Zabs = std::fabs(Z);
        if(n > 0){
            double value = 1.0, tmp = 1.0;
            for(int x = 0; x < n - 1; ++x){
                tmp *= -1.0 * Zabs / (x + 2);
                value += tmp;
            }
            if(Z >= 0.0){
                return value;
            }else{
                return -value;
            }
        }else{return 0.0;}
    }

    /// Non-Negative Integer Exponential
    size_t natPow(size_t x, size_t p){
      size_t i = 1;
      for (size_t j = 1; j <= p; j++)  i *= x;
      return i;
    }

public:

    // => Constructor <= //
    MRDSRG(Reference reference,boost::shared_ptr<Wavefunction> wfn,Options &options,std::shared_ptr<ForteIntegrals>  ints,std::shared_ptr<MOSpaceInfo> mo_space_info);

    // => Destructor <= //
    ~MRDSRG();

    /// The energy of the reference
    double Eref;

    /// The frozen-core energy
    double frozen_core_energy;

    /// Compute the corr_level energy with fixed reference
    double compute_energy();

    /// Compute the corr_level energy with relaxed reference
    double compute_energy_relaxed();
};

}}
#endif // _mrdsrg_h_
