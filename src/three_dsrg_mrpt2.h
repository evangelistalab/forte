#ifndef _three_dsrg_mrpt2_h_
#define _three_dsrg_mrpt2_h_

#include <fstream>

#include "liboptions/liboptions.h"
#include "libmints/wavefunction.h"

#include "integrals.h"
#include <ambit/blocked_tensor.h>
#include "reference.h"
#include <string>
#include <vector>
//#include <libthce/thce.h>
#include "blockedtensorfactory.h"

namespace psi{

namespace forte{

/**
 * @brief The MethodBase class
 * This class provides basic functions to write electronic structure
 * pilot codes using the Tensor classes
 */
class THREE_DSRG_MRPT2 : public Wavefunction
{
protected:

    // => Class data <= //

    /// The reference object
    Reference reference_;

    /// The molecular integrals required by MethodBase
    std::shared_ptr<ForteIntegrals>  ints_;
    /// The type of SCF reference
    std::string ref_type_;
    /// The number of corrleated MO
    size_t ncmo_;
    /// The number of auxiliary/cholesky basis functions
    size_t nthree_;

    /// The number of correlated orbitals per irrep (excluding frozen core and virtuals)
    Dimension ncmopi_;
    /// The number of restricted doubly occupied orbitals per irrep (core)
    Dimension rdoccpi_;
    /// The number of active orbitals per irrep (active)
    Dimension actvpi_;
    /// The number of restricted unoccupied orbitals per irrep (virtual)
    Dimension ruoccpi_;

    /// List of alpha core MOs
    std::vector<size_t> acore_mos_;
    size_t core_;
    /// List of alpha active MOs
    std::vector<size_t> aactv_mos_;
    size_t active_;
    /// List of alpha virtual MOs
    std::vector<size_t> avirt_mos_;
    size_t virtual_;

    /// List of beta core MOs
    std::vector<size_t> bcore_mos_;
    /// List of beta active MOs
    std::vector<size_t> bactv_mos_;
    /// List of beta virtual MOs
    std::vector<size_t> bvirt_mos_;

    /// List of eigenvalues for fock alpha
    std::vector<double> Fa_;
    /// List of eigenvalues for fock beta
    std::vector<double> Fb_;

    /// Map from all the MOs to the alpha core
    std::map<size_t,size_t> mos_to_acore_;
    /// Map from all the MOs to the alpha active
    std::map<size_t,size_t> mos_to_aactv_;
    /// Map from all the MOs to the alpha virtual
    std::map<size_t,size_t> mos_to_avirt_;

    /// Map from all the MOs to the beta core
    std::map<size_t,size_t> mos_to_bcore_;
    /// Map from all the MOs to the beta active
    std::map<size_t,size_t> mos_to_bactv_;
    /// Map from all the MOs to the beta virtual
    std::map<size_t,size_t> mos_to_bvirt_;

    /// The flow parameter
    double s_;

    /// Threshold for the Taylor expansion of f(z) = (1-exp(-z^2))/z
    double taylor_threshold_;
    /// Order of the Taylor expansion of f(z) = (1-exp(-z^2))/z
    int taylor_order_;

    // => Tensors <= //
    ambit::TensorType tensor_type_;
    ambit::BlockedTensor H_;
    ambit::BlockedTensor F_;
    ambit::BlockedTensor F_no_renorm_;
    ambit::BlockedTensor Gamma1_;
    ambit::BlockedTensor Eta1_;
    ambit::BlockedTensor Lambda2_;
    ambit::BlockedTensor Delta1_;
    ambit::BlockedTensor RDelta1_;
    ambit::BlockedTensor T1_;
    ambit::BlockedTensor RExp1_;  // < one-particle exponential for renormalized Fock matrix
    //These three are defined as member variables, but if integrals use DiskDF, these are not to be computed for the entire code
    ambit::BlockedTensor T2_;
    ambit::BlockedTensor V_;
    ambit::BlockedTensor ThreeIntegral_;
    ambit::BlockedTensor H0_;
    ambit::BlockedTensor Hbar1_;
    ambit::BlockedTensor Hbar2_;
    ambit::BlockedTensor O1_;
    ambit::BlockedTensor O2_;


    /// A vector of strings that avoids creating ccvv indices
    std::vector<std::string> no_hhpp_;

    // => Class initialization and termination <= //

    /// Called in the constructor
    void startup();
    ///Compute frozen natural orbitals
    /// Called in the destructor
    void cleanup();
    void print_summary();

    double renormalized_denominator(double D);


    /// compute the minimal amount of T2 for each term
    /// The spaces correspond to all the blocks you want to use
    ambit::BlockedTensor compute_T2_minimal(const std::vector<std::string> & spaces);
    /// compute ASTEI from DF/CD integrals
    /// function will take the spaces for V and use that to create the blocks for B
    ambit::BlockedTensor compute_B_minimal(const std::vector<std::string>& Vspaces);

    /// Computes the t1 amplitudes for three different cases of spin (alpha all, beta all, and alpha beta)
    void compute_t1();
    /// If DF or Cholesky, this function is not used
    void compute_t2();
    void check_t1();
    double T1norm_;
    double T1max_;

    //Compute V and maybe renormalize
    ambit::BlockedTensor compute_V_minimal(const std::vector<std::string> &, bool renormalize = true);
    /// Renormalize Fock matrix and two-electron integral
    void renormalize_F();
    void renormalize_V();
    double renormalized_exp(double D) {return std::exp(-s_ * std::pow(D, 2.0));}

    /// Compute DSRG-PT2 correlation energy - Group of functions to calculate individual pieces of energy
    double compute_ref();
    double E_FT1();
    double E_VT1();
    double E_FT2();
    double E_VT2_2();
    ///Compute hhva and acvv terms
    double E_VT2_2_one_active();
    ///Different algorithms for handling ccvv term
    /// Core -> builds everything in core.  Probably fastest
    double E_VT2_2_core();
    /// ambit -> Uses ambit library to perform contractions
    double E_VT2_2_ambit();
    ///fly_open-> Code Kevin wrote at first with open mp threading
    double E_VT2_2_fly_openmp();
    double E_VT2_2_batch_core();
    double E_VT2_2_batch_virtual();
    double E_VT2_4PP();
    double E_VT2_4HH();
    double E_VT2_4PH();
    double E_VT2_6();

    double Hbar0_ = 0.0;
    double scalar_energy_fci_ = 0.0;
    ///
    /// Compute zero-body term of commutator [H1, T1]
    void H1_T1_C0(ambit::BlockedTensor& H1, ambit::BlockedTensor& T1, const double& alpha, double& C0);
    /// Compute zero-body term of commutator [H1, T2]
    void H1_T2_C0(ambit::BlockedTensor& H1, ambit::BlockedTensor& T2, const double& alpha, double& C0);
    /// Compute zero-body term of commutator [H2, T1]
    void H2_T1_C0(ambit::BlockedTensor& H2, ambit::BlockedTensor& T1, const double& alpha, double& C0);
    /// Compute zero-body term of commutator [H2, T2]
    void H2_T2_C0(ambit::BlockedTensor& H2, ambit::BlockedTensor& T2, const double& alpha, double& C0);

    /// Compute one-body term of commutator [H1, T1]
    void H1_T1_C1(ambit::BlockedTensor& H1, ambit::BlockedTensor& T1, const double& alpha, ambit::BlockedTensor& C1);
    /// Compute one-body term of commutator [H1, T2]
    void H1_T2_C1(ambit::BlockedTensor& H1, ambit::BlockedTensor& T2, const double& alpha, ambit::BlockedTensor& C1);
    /// Compute one-body term of commutator [H2, T1]
    void H2_T1_C1(ambit::BlockedTensor& H2, ambit::BlockedTensor& T1, const double& alpha, ambit::BlockedTensor& C1);
    /// Compute one-body term of commutator [H2, T2]
    void H2_T2_C1(ambit::BlockedTensor& H2, ambit::BlockedTensor& T2, const double& alpha, ambit::BlockedTensor& C1);

    /// Compute two-body term of commutator [H2, T1]
    void H2_T1_C2(ambit::BlockedTensor& H2, ambit::BlockedTensor& T1, const double& alpha, ambit::BlockedTensor& C2);
    /// Compute two-body term of commutator [H1, T2]
    void H1_T2_C2(ambit::BlockedTensor& H1, ambit::BlockedTensor& T2, const double& alpha, ambit::BlockedTensor& C2);
    /// Compute two-body term of commutator [H2, T2]
    void H2_T2_C2(ambit::BlockedTensor& H2, ambit::BlockedTensor& T2, const double& alpha, ambit::BlockedTensor& C2);
    double time_H1_T1_C0 = 0.0;
    double time_H1_T2_C0 = 0.0;
    double time_H2_T1_C0 = 0.0;
    double time_H2_T2_C0 = 0.0;
    double time_H1_T1_C1 = 0.0;
    double time_H1_T2_C1 = 0.0;
    double time_H2_T1_C1 = 0.0;
    double time_H2_T2_C1 = 0.0;
    double time_H1_T2_C2 = 0.0;
    double time_H2_T1_C2 = 0.0;
    double time_H2_T2_C2 = 0.0;

    void de_normal_order();

    double relaxed_energy();

    std::vector<std::vector<double> > compute_restricted_docc_operator_dsrg();

    // Print levels
    int print_;
    // Print detailed timings
    bool detail_time_ = false;

    // Taylor Expansion of [1 - exp(-s * D^2)] / D = sqrt(s) * (\sum_{n=1} \frac{1}{n!} (-1)^{n+1} Z^{2n-1})
    double Taylor_Exp(const double& Z, const int& n){
        if(n > 0){
            double value = Z, tmp = Z;
            for(int x=0; x<(n-1); ++x){
                tmp *= std::pow(Z, 2.0) / (x+2);
                value += tmp;
            }
            return value;
        }else{return 0.0;}
    }

    ///This function will remove the indices that do not have at least one active index
    ///This function generates all possible MO spaces and spin components
    /// Param:  std::string is the lables - "cav"
    /// Will take a string like cav and generate all possible combinations of this
    /// for a four character string
    boost::shared_ptr<BlockedTensorFactory> BTF_;

    //The MOSpace object
    std::shared_ptr<MOSpaceInfo> mo_space_info_;
    IntegralType integral_type_;


public:

    // => Constructors <= //

    THREE_DSRG_MRPT2(Reference reference, SharedWavefunction ref_wfn, Options &options, std::shared_ptr<ForteIntegrals>  ints, std::shared_ptr<MOSpaceInfo>
    mo_space_info);

    ~THREE_DSRG_MRPT2();

    /// Compute the DSRG-MRPT2 energy
    double compute_energy();
    /// Allow the reference to relax
    void relax_reference_once();

    /// The energy of the reference
    double Eref_;

    /// The frozen-core energy
    double frozen_core_energy_;
private:
	//maximum number of threads
	int num_threads_;
	/// Do we have OpenMP?
	static bool have_omp_;
    /// Do we have MPI (actually use GA)
    static bool have_mpi_;
};

}} // End Namespaces

#endif // _three_dsrg_mrpt2_h_
