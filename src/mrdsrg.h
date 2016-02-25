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
#include "dsrg_source.h"
#include "dsrg_time.h"

using namespace ambit;
namespace psi{ namespace forte{

class MRDSRG : public Wavefunction
{
public:
    /**
     * MRDSRG Constructor
     * @param ref_wfn The reference wavefunction object
     * @param options The main options object
     * @param ints A pointer to an allocated integral object
     * @param mo_space_info The MOSpaceInfo object
     */
    MRDSRG(Reference reference, ref_wfn, Options& options,
           std::shared_ptr<ForteIntegrals> ints, std::shared_ptr<MOSpaceInfo> mo_space_info);

    /// Destructor
    ~MRDSRG();

    /// Compute the corr_level energy with fixed reference
    double compute_energy();

    /// Compute the corr_level energy with relaxed reference
    double compute_energy_relaxed();

protected:

    // => Class initialization and termination <= //

    /// Start-up function called in the constructor
    void startup();
    /// Clean-up function called in the destructor
    void cleanup();

    /// Read options
    void read_options();

    /// Print levels
    int print_;

    /// The reference object
    Reference reference_;

    /// The energy of the reference
    double Eref_;

    /// The frozen-core energy
    double frozen_core_energy_;

    /// The molecular integrals required by MethodBase
    std::shared_ptr<ForteIntegrals>  ints_;

    /// Are orbitals semi-canonicalized?
    bool semi_canonical_;

    /// MO space info
    std::shared_ptr<MOSpaceInfo> mo_space_info_;

    /// List of alpha core MOs
    std::vector<size_t> acore_mos_;
    /// List of alpha active MOs
    std::vector<size_t> aactv_mos_;
    /// List of alpha virtual MOs
    std::vector<size_t> avirt_mos_;
    /// List of beta core MOs
    std::vector<size_t> bcore_mos_;
    /// List of beta active MOs
    std::vector<size_t> bactv_mos_;
    /// List of beta virtual MOs
    std::vector<size_t> bvirt_mos_;

    /// Alpha core label
    std::string acore_label_;
    /// Alpha active label
    std::string aactv_label_;
    /// Alpha virtual label
    std::string avirt_label_;
    /// Beta core label
    std::string bcore_label_;
    /// Beta active label
    std::string bactv_label_;
    /// Beta virtual label
    std::string bvirt_label_;

    /// Map from space label to list of MOs
    std::map<char, std::vector<size_t>> label_to_spacemo_;

    /// Fill up integrals
    void build_ints();
    /// Fill up density matrix and density cumulants
    void build_density();
    /// Build Fock matrix and diagonal Fock matrix elements
    void build_fock(BlockedTensor& H, BlockedTensor& V);


    // => DSRG related <= //

    /// Correlation level
    enum class CORR_LV {LDSRG2, LDSRG2_P3, PT2, PT3, QDSRG2, QDSRG2_P3, CEPA0};
    std::map<std::string, CORR_LV> corrlevelmap =
            boost::assign::map_list_of("LDSRG2", CORR_LV::LDSRG2)("LDSRG2_P3", CORR_LV::LDSRG2_P3)
            ("PT2", CORR_LV::PT2)("PT3", CORR_LV::PT3)
            ("QDSRG2", CORR_LV::QDSRG2)("QDSRG2_P3", CORR_LV::QDSRG2_P3)
            ("CEPA0", CORR_LV::CEPA0);

    /// The flow parameter
    double s_;

    /// Source operator
    std::string source_;
    /// The dsrg source operator
    std::shared_ptr<DSRG_SOURCE> dsrg_source_;

    /// Smaller than which we will do Taylor expansion for renormalization
    double taylor_threshold_;

    /// Timings for computing the commutators
    DSRG_TIME dsrg_time_;

    /// Kevin's Tensor Wrapper
    std::shared_ptr<BlockedTensorFactory> BTF_;
    /// Tensor type for AMBIT
    TensorType tensor_type_;

    /// One-electron integral
    ambit::BlockedTensor H_;
    /// Two-electron integral
    ambit::BlockedTensor V_;
    /// Generalized Fock matrix
    ambit::BlockedTensor F_;
    /// One-particle density matrix
    ambit::BlockedTensor Gamma1_;
    /// One-hole density matrix
    ambit::BlockedTensor Eta1_;
    /// Two-body denisty cumulant
    ambit::BlockedTensor Lambda2_;
    /// Three-body density cumulant
    ambit::BlockedTensor Lambda3_;
    /// Single excitation amplitude
    ambit::BlockedTensor T1_;
    /// Double excitation amplitude
    ambit::BlockedTensor T2_;
    /// Difference of consecutive singles
    ambit::BlockedTensor DT1_;
    /// Difference of consecutive doubles
    ambit::BlockedTensor DT2_;
    /// Unitary matrix to block diagonal Fock
    ambit::BlockedTensor U_;

    /// Diagonal elements of Fock matrices
    std::vector<double> Fa_;
    std::vector<double> Fb_;

    /// Automatic adjust the flow parameter
    enum class SMART_S {DSRG_S, MIN_DELTA1, MAX_DELTA1, DAVG_MIN_DELTA1, DAVG_MAX_DELTA1};
    std::map<std::string, SMART_S> smartsmap =
            boost::assign::map_list_of("DSRG_S", SMART_S::DSRG_S)
            ("MIN_DELTA1", SMART_S::MIN_DELTA1)("DAVG_MIN_DELTA1", SMART_S::DAVG_MIN_DELTA1)
            ("MAX_DELTA1", SMART_S::MAX_DELTA1)("DAVG_MAX_DELTA1", SMART_S::DAVG_MAX_DELTA1);
    /// Automatic adjusting the flow parameter
    double make_s_smart();
    /// Algorithm to compute energy threshold according to Delta1
    double smart_s_min_delta1();
    double smart_s_max_delta1();
    double smart_s_davg_min_delta1();
    double smart_s_davg_max_delta1();

    /// Algorithm for computing amplitudes
    std::string T_algor_;
    /// Initial guess of T amplitudes
    void guess_t(BlockedTensor& V, BlockedTensor& T2, BlockedTensor& F, BlockedTensor& T1);
    /// Update T amplitude in every iteration
    void update_t();
    /// Analyze T1 and T2 amplitudes
    void analyze_amplitudes(std::string name, BlockedTensor& T1, BlockedTensor& T2);

    /// RMS of T2
    double T2rms_;
    /// Norm of T2
    double T2norm_;
    double t2aa_norm_;
    double t2ab_norm_;
    double t2bb_norm_;
    /// Signed max of T2
    double T2max_;
    /// Check T2 and store the largest amplitudes
    void check_t2(BlockedTensor& T2);
    /// Initial guess of T2
    void guess_t2_std(BlockedTensor& V, BlockedTensor& T2);
    void guess_t2_noccvv(BlockedTensor& V, BlockedTensor& T2);
    /// Update T2 in every iteration
    void update_t2_std();
    void update_t2_noccvv();

    /// RMS of T1
    double T1rms_;
    /// Norm of T1
    double T1norm_;
    double t1a_norm_;
    double t1b_norm_;
    /// Signed max of T1
    double T1max_;
    /// Check T1 and store the largest amplitudes
    void check_t1(BlockedTensor& T1);
    /// Initial guess of T1
    void guess_t1_std(BlockedTensor& F, BlockedTensor& T2, BlockedTensor& T1);
    void guess_t1_nocv(BlockedTensor& F, BlockedTensor& T2, BlockedTensor& T1);
    /// Update T1 in every iteration
    void update_t1_std();
    void update_t1_nocv();

    /// Write T2 to files MRDSRG_T2_XX.dat, XX = AA, AB, BB
    void write_t2_file(BlockedTensor& T2, const std::string& spin);
    /// Read T2 from files MRDSRG_T2_XX.dat, XX = AA, AB, BB
    void read_t2_file(BlockedTensor& T2, const std::string& spin);
    /// Write T1 to files MRDSRG_T1_X.dat, X = A, B
    void write_t1_file(BlockedTensor& T1, const std::string& spin);
    /// Read T1 from files MRDSRG_T1_X.dat, X = A, B
    void read_t1_file(BlockedTensor& T1, const std::string& spin);

    /// Number of amplitudes will be printed in amplitude summary
    int ntamp_;
    /// Threshold for amplitudes considered as intruders
    double intruder_tamp_;
    /// List of large amplitudes
    std::vector<std::pair<std::vector<size_t>, double>> lt1a_;
    std::vector<std::pair<std::vector<size_t>, double>> lt1b_;
    std::vector<std::pair<std::vector<size_t>, double>> lt2aa_;
    std::vector<std::pair<std::vector<size_t>, double>> lt2ab_;
    std::vector<std::pair<std::vector<size_t>, double>> lt2bb_;

    /// Compute DSRG-transformed Hamiltonian Hbar
    void compute_hbar();
    /// Zero-body Hbar
    double Hbar0_;
    /// One-body Hbar
    ambit::BlockedTensor Hbar1_;
    /// Two-body Hbar
    ambit::BlockedTensor Hbar2_;
    /// Temporary one-body Hamiltonian
    ambit::BlockedTensor O1_;
    ambit::BlockedTensor C1_;
    /// Temporary two-body Hamiltonian
    ambit::BlockedTensor O2_;
    ambit::BlockedTensor C2_;

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

    /// Compute diagonal blocks labels
    std::vector<std::string> diag_labels();
    /// Compute off-diagonal blocks labels of a one-body operator
    std::vector<std::string> od_one_labels();
    /// Compute off-diagonal blocks labels of a two-body operator
    std::vector<std::string> od_two_labels();
    /// Copy T1 and T2 to a big vector for DIIS
    std::vector<double> copy_amp_diis(BlockedTensor& T1, const std::vector<std::string>& label1,
                                      BlockedTensor& T2, const std::vector<std::string>& label2);
    /// Compute number of elements of the big vector for DIIS
    size_t vector_size_diis(BlockedTensor& T1, const std::vector<std::string>& label1,
                            BlockedTensor& T2, const std::vector<std::string>& label2);
    /// Copy extrapolated values back to T1 and T2
    void return_amp_diis(BlockedTensor& T1, const std::vector<std::string>& label1,
                         BlockedTensor& T2, const std::vector<std::string>& label2,
                         const std::vector<double>& data);

    /// Compute MR-DSRG-CEPA0
    double compute_energy_cepa0();
    /// Compute MR-LDSRG(2)
    double compute_energy_ldsrg2();

    /// Zeroth-order Hamiltonian
    ambit::BlockedTensor H0th_;
    /// Compute DSRG-MRPT2 energy
    double compute_energy_pt2();
    /// Compute DSRG-MRPT3 energy
    double compute_energy_pt3();
    /// Check if orbitals are semi-canonicalized
    bool check_semicanonical();


    // => Reference relaxation <= //

    /// Transfer integrals for FCI
    void transfer_integrals();
    /// Reset integrals to bare Hamiltonian
    void reset_ints(BlockedTensor& H, BlockedTensor& V);
    /// Diagonalize the diagonal blocks of the Fock matrix
    std::vector<std::vector<double>> diagonalize_Fock_diagblocks(BlockedTensor& U);
    /// Separate an 2D ambit::Tensor according to its irrep
    ambit::Tensor separate_tensor(ambit::Tensor& tens, const Dimension& irrep, const int& h);
    /// Combine a separated 2D ambit::Tensor
    void combine_tensor(ambit::Tensor& tens, ambit::Tensor& tens_h, const Dimension& irrep, const int& h);


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
    /// Check the max and norm of density
    void check_density(BlockedTensor& D, const std::string& name);
    /// Print the summary of 2- and 3-body density cumulant
    void print_cumulant_summary();
};

}}
#endif // _mrdsrg_h_
