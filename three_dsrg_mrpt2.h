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
#include "blockedtensorfactory.h"

namespace psi{

namespace libadaptive{

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
    ExplorerIntegrals* ints_;
    /// The type of SCF reference
    std::string ref_type_;

    /// The number of correlated orbitals per irrep (excluding frozen core and virtuals)
    Dimension ncmopi_;
    /// The number of restricted doubly occupied orbitals per irrep (core)
    Dimension rdoccpi_;
    /// The number of active orbitals per irrep (active)
    Dimension actvpi_;
    /// The number of restricted unoccupied orbitals per irrep (virtual)
    Dimension ruoccpi_;

    /// List of alpha core MOs
    std::vector<size_t> acore_mos;
    size_t core_;
    /// List of alpha active MOs
    std::vector<size_t> aactv_mos;
    size_t active_;
    /// List of alpha virtual MOs
    std::vector<size_t> avirt_mos;
    size_t virtual_;

    /// List of beta core MOs
    std::vector<size_t> bcore_mos;
    /// List of beta active MOs
    std::vector<size_t> bactv_mos;
    /// List of beta virtual MOs
    std::vector<size_t> bvirt_mos;

    /// List of eigenvalues for fock alpha
    std::vector<double> Fa;
    /// List of eigenvalues for fock beta
    std::vector<double> Fb;

    /// Map from all the MOs to the alpha core
    std::map<size_t,size_t> mos_to_acore;
    /// Map from all the MOs to the alpha active
    std::map<size_t,size_t> mos_to_aactv;
    /// Map from all the MOs to the alpha virtual
    std::map<size_t,size_t> mos_to_avirt;

    /// Map from all the MOs to the beta core
    std::map<size_t,size_t> mos_to_bcore;
    /// Map from all the MOs to the beta active
    std::map<size_t,size_t> mos_to_bactv;
    /// Map from all the MOs to the beta virtual
    std::map<size_t,size_t> mos_to_bvirt;

    /// The flow parameter
    double s_;

    /// Threshold for the Taylor expansion of f(z) = (1-exp(-z^2))/z
    double taylor_threshold_;
    /// Order of the Taylor expansion of f(z) = (1-exp(-z^2))/z
    int taylor_order_;

    // => Tensors <= //
    TensorType tensor_type_;
    ambit::BlockedTensor H;
    ambit::BlockedTensor F;
    ambit::BlockedTensor V;
    ambit::BlockedTensor Gamma1;
    ambit::BlockedTensor Eta1;
    ambit::BlockedTensor Lambda2;
    ambit::BlockedTensor Lambda3;
    ambit::BlockedTensor Delta1;
    ambit::BlockedTensor RDelta1;
    ambit::BlockedTensor T1;
    ambit::BlockedTensor T2;
    ambit::BlockedTensor RExp1;  // < one-particle exponential for renormalized Fock matrix
    ambit::BlockedTensor ThreeIntegral;

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

    void compute_t1();

    /// Computes the t2 amplitudes for three different cases of spin (alpha all, beta all, and alpha beta)
    void compute_t2();
    void check_t2();
    double T2norm;
    double T2max;

    /// Computes the t1 amplitudes for three different cases of spin (alpha all, beta all, and alpha beta)
    void check_t1();
    double T1norm;
    double T1max;

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
    ///Different algorithms for handling ccvv term
    /// Core -> builds everything in core.  Probably fastest
    double E_VT2_2_core();
    /// ambit -> Uses ambit library to perform contractions
    double E_VT2_2_ambit();
    ///fly_open-> Code Kevin wrote at first with open mp threading
    double E_VT2_2_fly_openmp();
    double E_VT2_4PP();
    double E_VT2_4HH();
    double E_VT2_4PH();
    double E_VT2_6();

    // Print levels
    int print_;

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
    std::vector<std::string> spin_cases_avoid(const std::vector<std::string>& in_str_vec);
    ///This function generates all possible MO spaces and spin components
    /// Param:  std::string is the lables - "cav"
    /// Will take a string like cav and generate all possible combinations of this
    /// for a four character string
    std::vector<std::string> generate_all_indices(const std::string, std::string);
    boost::shared_ptr<BlockedTensorFactory> BTF;


public:

    // => Constructors <= //

    THREE_DSRG_MRPT2(Reference reference,boost::shared_ptr<Wavefunction> wfn, Options &options, ExplorerIntegrals* ints);

    ~THREE_DSRG_MRPT2();

    /// Compute the DSRG-MRPT2 energy
    double compute_energy();

    /// The energy of the reference
    double Eref;

    /// The frozen-core energy
    double frozen_core_energy;
private:
	//maximum number of threads
	int num_threads_;
	/// Do we have OpenMP?
	static bool have_omp_;
};

}} // End Namespaces

#endif // _three_dsrg_mrpt2_h_
