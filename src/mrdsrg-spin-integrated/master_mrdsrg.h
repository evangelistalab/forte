#ifndef _master_mrdsrg_h_
#define _master_mrdsrg_h_

#include <cmath>
#include <memory>
#include <tuple>

#include "ambit/tensor.h"
#include "ambit/blocked_tensor.h"

#include "../dynamic_correlation_solver.h"
#include "../integrals/integrals.h"
#include "../fci/fci_integrals.h"
#include "../reference.h"
#include "../helpers.h"
#include "../blockedtensorfactory.h"
#include "../mrdsrg-helper/dsrg_source.h"
#include "../mrdsrg-helper/dsrg_time.h"

using namespace ambit;
namespace psi {
namespace forte {
class MASTER_DSRG : public DynamicCorrelationSolver {
  public:
    /**
     * Constructor
     * @param ref_wfn The reference wavefunction object
     * @param options The main options object
     * @param ints A pointer to an allocated integral object
     * @param mo_space_info The MOSpaceInfo object
     */
    MASTER_DSRG(Reference reference, SharedWavefunction ref_wfn, Options& options,
                std::shared_ptr<ForteIntegrals> ints, std::shared_ptr<MOSpaceInfo> mo_space_info);

    /// Destructor
    virtual ~MASTER_DSRG();

    /// Compute energy
    virtual double compute_energy() = 0;

    /// Compute DSRG transformed Hamiltonian
    virtual std::shared_ptr<FCIIntegrals> compute_Heff_actv();

    /// Compute second-order effective Hamiltonian couplings (child class overrides)
    /// <M|H + HA(N)|N> = Heff1 * TrD1 + Heff2 * TrD2 + Heff3 * TrD3 if CAS
    virtual void compute_Heff_2nd_coupling(double& H0, ambit::Tensor& H1a, ambit::Tensor& H1b,
                                           ambit::Tensor& H2aa, ambit::Tensor& H2ab,
                                           ambit::Tensor& H2bb, ambit::Tensor& H3aaa,
                                           ambit::Tensor& H3aab, ambit::Tensor& H3abb,
                                           ambit::Tensor& H3bbb) {
        throw PSIEXCEPTION("Child class should override this function");
    }

    /// Compute DSRG dressed density
    //    virtual void compute_density() = 0;

    /// Compute DSRG transformed dipole integrals
    //    virtual void compute_dm_eff(std::vector<double>& M0, std::vector<BlockedTensor>& M1,
    //                                std::vector<BlockedTensor>& M2) = 0;

    /// Return the Hbar of a given order
    std::vector<ambit::Tensor> Hbar(int n);

    /// Set unitary matrix (in active space) from original to semicanonical
    void set_Uactv(ambit::Tensor& Ua, ambit::Tensor& Ub) {
        Uactv_ = BTF_->build(tensor_type_, "Uactv", spin_cases({"aa"}));
        Uactv_.block("aa")("pq") = Ua("pq");
        Uactv_.block("AA")("pq") = Ub("pq");
    }

    /// Set active active occupied MOs (relative to active)
    void set_actv_occ(std::vector<size_t> actv_occ) {
        actv_occ_mos_ = std::vector<size_t>(actv_occ);
    }

    /// Set active active unoccupied MOs (relative to active)
    void set_actv_uocc(std::vector<size_t> actv_uocc) {
        actv_uocc_mos_ = std::vector<size_t>(actv_uocc);
    }

  protected:
    /// Startup function called in constructor
    void startup();

    /// Warnings <description, changes in this run, how to get rid of it>
    std::vector<std::tuple<const char*, const char*, const char*>> warnings_;

    // ==> settings from options <==

    /// Read options
    void read_options();

    /// Printing level
    int print_;

    /// The flow parameter
    double s_;
    /// Source operator
    std::string source_;
    /// The dsrg source operator
    std::shared_ptr<DSRG_SOURCE> dsrg_source_;
    /// Threshold for the Taylor expansion of f(z) = (1-exp(-z^2))/z
    double taylor_threshold_;

    /// The integral integral
    std::string ints_type_;
    /// If ERI density fitted or Cholesky decomposed
    bool eri_df_;

    /// Multi-state computation if true
    bool multi_state_;
    /// Multi-state algorithm
    std::string multi_state_algorithm_;

    /// Number of amplitudes will be printed in amplitude summary
    int ntamp_;
    /// Threshold for amplitudes considered as intruders
    double intruder_tamp_;

    /// Relaxation type
    std::string relax_ref_;

    /// Timings for computing the commutators
    DSRG_TIME dsrg_time_;

    /// Active orbital rotation from semicanonicalizor (set from outside)
    ambit::BlockedTensor Uactv_;
    /// Rotate 2-body DSRG transformed integrals from semicanonical back to original
    void rotate_ints_semi_to_origin(const std::string& name, BlockedTensor& H1, BlockedTensor& H2);
    /// Rotate 3-body DSRG transformed integrals from semicanonical back to original
    void rotate_ints_semi_to_origin(const std::string& name, BlockedTensor& H1, BlockedTensor& H2,
                                    BlockedTensor& H3);

    // ==> some common energies for all DSRG levels <==

    /// The energy of the reference
    double Eref_;
    /// The nuclear repulsion energy
    double Enuc_;
    /// The frozen core energy
    double Efrzc_;

    /// Initial check on reference energy
    void check_init_reference_energy();

    /// Compute reference (MK vacuum) energy from ForteIntegral and Fock_
    double compute_reference_energy_from_ints(std::shared_ptr<ForteIntegrals> ints);

    /// Compute reference (MK vacuum) energy
    double compute_reference_energy(BlockedTensor H, BlockedTensor F, BlockedTensor V);
    double compute_reference_energy_df(BlockedTensor H, BlockedTensor F, BlockedTensor B);

    // ==> MO space info <==

    /// Read MO space info
    void read_MOSpaceInfo();

    /// List of core MOs
    std::vector<size_t> core_mos_;
    /// List of active MOs
    std::vector<size_t> actv_mos_;
    /// List of virtual MOs
    std::vector<size_t> virt_mos_;

    /// List of active active occupied MOs (relative to active)
    std::vector<size_t> actv_occ_mos_;
    /// List of active active unoccupied MOs (relative to active)
    std::vector<size_t> actv_uocc_mos_;

    /// List of auxiliary MOs when DF/CD
    std::vector<size_t> aux_mos_;

    // ==> Ambit tensor settings <==

    /// Set Ambit tensor labels
    void set_ambit_MOSpace();

    /// Kevin's Tensor Wrapper
    std::shared_ptr<BlockedTensorFactory> BTF_;
    /// Tensor type for Ambit
    ambit::TensorType tensor_type_;

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

    /// Auxillary basis label
    std::string aux_label_;

    /// Map from space label to list of MOs
    std::map<char, std::vector<size_t>> label_to_spacemo_;

    /// Compute diagonal blocks labels of a one-body operator
    std::vector<std::string> diag_one_labels();
    /// Compute diagonal blocks labels of a two-body operator
    std::vector<std::string> diag_two_labels();
    /// Compute retaining excitation blocks labels of a two-body operator
    std::vector<std::string> re_two_labels();
    /// Compute off-diagonal blocks labels of a one-body operator
    std::vector<std::string> od_one_labels();
    std::vector<std::string> od_one_labels_hp();
    std::vector<std::string> od_one_labels_ph();
    /// Compute off-diagonal blocks labels of a two-body operator
    std::vector<std::string> od_two_labels();
    std::vector<std::string> od_two_labels_hhpp();
    std::vector<std::string> od_two_labels_pphh();

    // ==> fill in densities from Reference <==
    /** Lambda3 is no longer stored !!! */

    /// Initialize density cumulants
    void init_density();
    /// Fill in density cumulants from the Reference
    void fill_density();

    /// One-particle density matrix
    ambit::BlockedTensor Gamma1_;
    /// One-hole density matrix
    ambit::BlockedTensor Eta1_;
    /// Two-body denisty cumulant
    ambit::BlockedTensor Lambda2_;

    // ==> Fock matrix related <==

    /// Fock matrix
    ambit::BlockedTensor Fock_;
    /// Diagonal elements of Fock matrix with alpha spin
    std::vector<double> Fdiag_a_;
    /// Diagonal elements of Fock matrix with beta spin
    std::vector<double> Fdiag_b_;

    /// Initialize Fock matrix
    void init_fock();
    /// Build Fock matrix from ForteIntegrals
    void build_fock_from_ints(std::shared_ptr<ForteIntegrals> ints, BlockedTensor& F);
    /// Fill in diagonal elements of Fock matrix to Fdiag
    void fill_Fdiag(BlockedTensor& F, std::vector<double>& Fa, std::vector<double>& Fb);
    /// Check orbitals if semicanonical
    bool check_semi_orbs();

    // ==> integrals <==

    /// Scalar of the DSRG transformed Hamiltonian
    double Hbar0_;
    /// DSRG transformed 1-body Hamiltonian (active only in DSRG-PT, but full in MRDSRG)
    ambit::BlockedTensor Hbar1_;
    /// DSRG transformed 2-body Hamiltonian (active only in DSRG-PT, but full in MRDSRG)
    ambit::BlockedTensor Hbar2_;
    /// DSRG transformed 3-body Hamiltonian (active only in DSRG-PT, but full in MRDSRG)
    ambit::BlockedTensor Hbar3_;

    /**
     * De-normal-order a 2-body DSRG transformed integrals
     * This will change H0 and H1 !!!
     */
    void deGNO_ints(const std::string& name, double& H0, BlockedTensor& H1, BlockedTensor& H2);

    /**
     * De-normal-order a 3-body DSRG transformed integrals
     * This will change H0, H1, and H2 !!!
     */
    void deGNO_ints(const std::string& name, double& H0, BlockedTensor& H1, BlockedTensor& H2,
                    BlockedTensor& H3);

    /**
     * De-normal-order the T1 and T2 amplitudes and return the effective T1
     * T1eff = T1 - T2["ivau"] * D1["uv"]
     *
     * This assumes no internal amplitudes !!!
     */
    ambit::BlockedTensor deGNO_Tamp(BlockedTensor& T1, BlockedTensor& T2, BlockedTensor& D1);

    // ==> dipole moment <==

    /// Compute dipole or not
    bool do_dm_;
    /// Dipole moment directions
    std::vector<std::string> dm_dirs_{"X", "Y", "Z"};
    /// Setup dipole integrals and DSRG transformed integrals
    void init_dm_ints();

    /// Nuclear dipole moments
    std::vector<double> dm_nuc_;
    /// Frozen-core contributions to permament dipole
    std::vector<double> dm_frzc_;
    /// Electronic dipole moment of the reference
    std::vector<double> dm_ref_;

    /// MO bare dipole integrals of size ncmo by ncmo
    std::vector<ambit::BlockedTensor> dm_;

    /// Fill in bare MO dipole integrals
    void fill_MOdm(std::vector<SharedMatrix>& dm_a, std::vector<SharedMatrix>& dm_b);
    /// Compute dipole moment of the reference
    void compute_dm_ref();
    /// Compute dipole for a certain direction or not
    std::vector<bool> do_dm_dirs_;

    /// DSRG transformed dipole scalar
    std::vector<double> Mbar0_;
    /// DSRG transformed 1-body dipole integrals (active only)
    std::vector<BlockedTensor> Mbar1_;
    /// DSRG transformed 2-body dipole integrals (active only)
    std::vector<BlockedTensor> Mbar2_;
    /// DSRG transformed 3-body dipole integrals (active only)
    std::vector<BlockedTensor> Mbar3_;

    // ==> commutators <==
    /**
      * H1, C1, G1: a rank 2 tensor of all MOs in general
      * H2, C2, G2: a rank 4 tensor of all MOs in general
      * C3: a rank 6 tensor of all MOs in general
      * T1: a rank 2 tensor of hole-particle
      * T2: a rank 4 tensor of hole-hole-particle-particle
      * V: antisymmetrized 2-electron integrals
      * B: 3-index integrals from DF/CD
      */

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

    //    /// Compute one-body actv-actv term of commutator [H1, T1]
    //    void H1_T1_C1aa(BlockedTensor& H1, BlockedTensor& T1, const double& alpha, BlockedTensor&
    //    C1);
    //    /// Compute one-body actv-actv term of commutator [H1, T2]
    //    void H1_T2_C1aa(BlockedTensor& H1, BlockedTensor& T2, const double& alpha, BlockedTensor&
    //    C1);
    //    /// Compute one-body actv-actv term of commutator [H2, T1]
    //    void H2_T1_C1aa(BlockedTensor& H2, BlockedTensor& T1, const double& alpha, BlockedTensor&
    //    C1);
    //    /// Compute one-body actv-actv term of commutator [H2, T2]
    //    void H2_T2_C1aa(BlockedTensor& H2, BlockedTensor& T2, const double& alpha, BlockedTensor&
    //    C1);

    //    /// Compute one-body hole-particle term of commutator [H1, T1]
    //    void H1_T1_C1hp(BlockedTensor& H1, BlockedTensor& T1, const double& alpha, BlockedTensor&
    //    C1);
    //    /// Compute one-body hole-particle term of commutator [H1, T2]
    //    void H1_T2_C1hp(BlockedTensor& H1, BlockedTensor& T2, const double& alpha, BlockedTensor&
    //    C1);
    //    /// Compute one-body hole-particle term of commutator [H2, T1]
    //    void H2_T1_C1hp(BlockedTensor& H2, BlockedTensor& T1, const double& alpha, BlockedTensor&
    //    C1);
    //    /// Compute one-body hole-particle term of commutator [H2, T2]
    //    void H2_T2_C1hp(BlockedTensor& H2, BlockedTensor& T2, const double& alpha, BlockedTensor&
    //    C1);

    //    /// Compute one-body particle-hole term of commutator [H1, T1]
    //    void H1_T1_C1ph(BlockedTensor& H1, BlockedTensor& T1, const double& alpha, BlockedTensor&
    //    C1);
    //    /// Compute one-body particle-hole term of commutator [H1, T2]
    //    void H1_T2_C1ph(BlockedTensor& H1, BlockedTensor& T2, const double& alpha, BlockedTensor&
    //    C1);
    //    /// Compute one-body particle-hole term of commutator [H2, T1]
    //    void H2_T1_C1ph(BlockedTensor& H2, BlockedTensor& T1, const double& alpha, BlockedTensor&
    //    C1);
    //    /// Compute one-body particle-hole term of commutator [H2, T2]
    //    void H2_T2_C1ph(BlockedTensor& H2, BlockedTensor& T2, const double& alpha, BlockedTensor&
    //    C1);

    /// Compute two-body term of commutator [H2, T1]
    void H2_T1_C2(BlockedTensor& H2, BlockedTensor& T1, const double& alpha, BlockedTensor& C2);
    /// Compute two-body term of commutator [H1, T2]
    void H1_T2_C2(BlockedTensor& H1, BlockedTensor& T2, const double& alpha, BlockedTensor& C2);
    /// Compute two-body term of commutator [H2, T2]
    void H2_T2_C2(BlockedTensor& H2, BlockedTensor& T2, const double& alpha, BlockedTensor& C2);

    /// Compute three-body term of commutator [H2, T2]
    void H2_T2_C3(BlockedTensor& H2, BlockedTensor& T2, const double& alpha, BlockedTensor& C3,
                  const bool& active_only = true);

    /// Compute one- and two-body off-diagonal term of commutator [[H1, A2]2d, T1+2]od
    void H1_A2_T_Cod(BlockedTensor& H1, BlockedTensor& A2, BlockedTensor& T1, BlockedTensor& T2,
                     const double& alpha, BlockedTensor& C1, BlockedTensor& C2);

    /// Compute one-body term of commutator [V, T1], V is constructed from B (DF/CD)
    void V_T1_C1_DF(BlockedTensor& B, BlockedTensor& T1, const double& alpha, BlockedTensor& C1);
    /// Compute one-body term of commutator [V, T2], V is constructed from B (DF/CD)
    void V_T2_C1_DF(BlockedTensor& B, BlockedTensor& T2, const double& alpha, BlockedTensor& C1);
    /// Compute two-body term of commutator [V, T1], V is constructed from B (DF/CD)
    void V_T1_C2_DF(BlockedTensor& B, BlockedTensor& T1, const double& alpha, BlockedTensor& C2);
    /// Compute two-body term of commutator [V, T2], V is constructed from B (DF/CD)
    void V_T2_C2_DF(BlockedTensor& B, BlockedTensor& T2, const double& alpha, BlockedTensor& C2);

    /// Compute two-body term of commutator [V, T2], V is constructed from B (DF/CD) in batches
    void V_T2_C2_DF_BATCH(BlockedTensor& B, BlockedTensor& T2, const double& alpha,
                          BlockedTensor& C2);

    /// Compute zero-body term of commutator [H1, G1]
    void H1_G1_C0(BlockedTensor& H1, BlockedTensor& G1, const double& alpha, double& C0);
    /// Compute zero-body term of commutator [H1, G2]
    void H1_G2_C0(BlockedTensor& H1, BlockedTensor& G2, const double& alpha, double& C0);
    /// Compute zero-body term of commutator [H2, G2]
    void H2_G2_C0(BlockedTensor& H2, BlockedTensor& G2, const double& alpha, double& C0);

    /// Compute one-body term of commutator [H1, G1]
    void H1_G1_C1(BlockedTensor& H1, BlockedTensor& G1, const double& alpha, BlockedTensor& C1);
    /// Compute one-body term of commutator [H1, G2]
    void H1_G2_C1(BlockedTensor& H1, BlockedTensor& G2, const double& alpha, BlockedTensor& C1);
    /// Compute one-body term of commutator [H2, G2]
    void H2_G2_C1(BlockedTensor& H2, BlockedTensor& G2, const double& alpha, BlockedTensor& C1);

    /// Compute two-body term of commutator [H1, G2]
    void H1_G2_C2(BlockedTensor& H1, BlockedTensor& G2, const double& alpha, BlockedTensor& C2);
    /// Compute two-body term of commutator [H2, G2]
    void H2_G2_C2(BlockedTensor& H2, BlockedTensor& G2, const double& alpha, BlockedTensor& C2);
};
}
}
#endif // MASTER_MRDSRG_H
