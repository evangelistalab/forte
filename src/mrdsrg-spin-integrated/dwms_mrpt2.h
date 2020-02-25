#ifndef _dwms_mrpt2_h_
#define _dwms_mrpt2_h_



#include "mrdsrg-spin-integrated/master_mrdsrg.h"
#include "sci/fci_mo.h"

namespace forte {

class ForteOptions;
class SCFInfo;

class DWMS_DSRGPT2 {
  public:
    /**
     * @brief DWMS_DSRGPT2 Constructor
     * @param ref_wfn The reference wavefunction object
     * @param options PSI4 and FORTE options
     * @param ints ForteInegrals
     * @param mo_space_info MOSpaceInfo
     */
    DWMS_DSRGPT2(std::shared_ptr<SCFInfo> scf_info, std::shared_ptr<ForteOptions> options,
                 std::shared_ptr<ForteIntegrals> ints, std::shared_ptr<MOSpaceInfo> mo_space_info);

    /// Destructor
    ~DWMS_DSRGPT2();

    /// compute energy and return the ground state energy
    double compute_energy();

  private:
    /// The molecular integrals
    std::shared_ptr<ForteIntegrals> ints_;

    /// The MO space info
    std::shared_ptr<MOSpaceInfo> mo_space_info_;

    /// The SCF info
    std::shared_ptr<SCFInfo> scf_info_;

    /// The ForteOptions
    std::shared_ptr<ForteOptions> foptions_;

    /// preparation
    void startup();
    void read_options();
    void test_options();
    void print_options();

    /// gaussian cutoff for density reweighting
    double zeta_;

    /// DWMS algorithm
    std::string algorithm_;

    /// use what energies to determine the weights
    /// and what CI vectors to do DWMS-DSRG
    std::string dwms_ref_;

    /// DWMS correlation level
    std::string dwms_corrlv_;

    /// Consider X(αβ) = A(β) - A(α) in SA algorithm
    bool do_delta_amp_;

    /// Iteratively update the reference CI coefficients
    bool dwms_iterate_;
    /// Max number of iteration for the update of the reference CI
    int dwms_maxiter_;
    /// Current number of iteration for the update of the reference CI
    int dwms_niter_ = 1;
    /// DWMS energy convergence
    double dwms_e_convergence_;

    /// form Hbar3 for DSRG-MRPT2
    bool do_hbar3_;
    /// max body of Hbar computed
    int max_hbar_level_;

    /// transform integrals to semicanonical basis (only PT2 when DF)
    bool do_semi_;

    /// save a copy of original orbitals
    psi::SharedMatrix Ca_copy_;
    psi::SharedMatrix Cb_copy_;

    /// transform integrals to original basis
    void transform_ints0();

    /// nuclear repulsion energy
    double Enuc_;

    /// precompute energy -- CASCI or SA-DSRG-PT2/3
    std::shared_ptr<FCI_MO> precompute_energy();

    /// perform DSRG-PT2/3 computation and return the dressed integrals within active space
    std::shared_ptr<ActiveSpaceIntegrals> compute_dsrg_pt(std::shared_ptr<MASTER_DSRG>& dsrg_pt,
                                                  RDMs& reference, std::string level = "PT2");

    /// perform a macro DSRG-PT2/3 computation
    std::shared_ptr<ActiveSpaceIntegrals> compute_macro_dsrg_pt(std::shared_ptr<MASTER_DSRG>& dsrg_pt,
                                                        std::shared_ptr<FCI_MO> fci_mo, int entry,
                                                        int root);

    /// compute DWSA energies
    void compute_dwsa_energy(std::shared_ptr<FCI_MO>& fci_mo);
    void compute_dwsa_energy_iterate(std::shared_ptr<FCI_MO>& fci_mo);

    /// update eigen vectors and values for a given entry
    /// old_eigen - original states
    /// new_vals - new energy
    /// new_vecs - linear combinations of original states
    std::vector<std::pair<psi::SharedVector, double>>
    compute_new_eigen(const std::vector<std::pair<psi::SharedVector, double>>& old_eigen,
                      psi::SharedVector new_vals, psi::SharedMatrix new_vecs);

    /// compute MS or XMS energies
    void compute_dwms_energy(std::shared_ptr<FCI_MO>& fci_mo);

    /// rotate 2nd-order effective Hamiltonian from semicanonical to original
    void rotate_H1(ambit::Tensor& H1a, ambit::Tensor& H1b);
    void rotate_H2(ambit::Tensor& H2aa, ambit::Tensor& H2ab, ambit::Tensor& H2bb);
    void rotate_H3(ambit::Tensor& H3aaa, ambit::Tensor& H3aab, ambit::Tensor& H3abb,
                   ambit::Tensor& H3bbb);

    /// contract H with transition densities
    double contract_Heff_1TrDM(ambit::Tensor& H1a, ambit::Tensor& H1b, RDMs& TrD,
                               bool transpose);
    double contract_Heff_2TrDM(ambit::Tensor& H2aa, ambit::Tensor& H2ab, ambit::Tensor& H2bb,
                               RDMs& TrD, bool transpose);
    double contract_Heff_3TrDM(ambit::Tensor& H3aaa, ambit::Tensor& H3aab, ambit::Tensor& H3abb,
                               ambit::Tensor& H3bbb, RDMs& TrD, bool transpose);

    /// compute DWMS energies by diagonalizing separate Hamiltonians
    void compute_dwms_energy_separated_H(std::shared_ptr<FCI_MO>& fci_mo);

    /// initial guesses if separate diagonalizations and require orthogonalized final CI vectors
    std::vector<std::vector<psi::SharedVector>> initial_guesses_;

    /// compute DWMS weights and return a new sa_info
    std::vector<std::tuple<int, int, int, std::vector<double>>>
    compute_dwms_weights(const std::vector<std::tuple<int, int, int, std::vector<double>>>& sa_info,
                         int entry, int root, const std::vector<std::vector<double>>& energy);

    /// max level computed for reduced density in DSRG
    int max_rdm_level_;

    /// if using factorized integrals
    bool eri_df_;

    /// a shared_ptr of ActiveSpaceIntegrals (mostly used in CI_RDMS)
    std::shared_ptr<ActiveSpaceIntegrals> fci_ints_;

    /// energy of original CASCI
    std::vector<std::vector<double>> Eref_0_;
    /// energy of SA-DSRG-PT2/3 (if computed)
    std::vector<std::vector<double>> Ept_0_;
    /// energy of DWMS-DSRG-PT2/3
    std::vector<std::vector<double>> Ept_;

    /// irrep symbols
    std::vector<std::string> irrep_symbol_;

    /// multiplicity symbols
    std::vector<std::string> multi_symbol_;

    /// unitary matrices (in active space) from original to semicanonical
    ambit::Tensor Ua_;
    ambit::Tensor Ub_;

    /// print implementation note
    void print_impl_note();
    /// print implementation note on separated H scheme
    void print_impl_note_sH();
    /// print implementation note on MS or XMS
    void print_impl_note_ms();
    /// print implementation note on SA or XSA
    void print_impl_note_sa();

    /// print title
    void print_title(const std::string& title);

    /// print sa_info
    void print_sa_info(const std::string& name,
                       const std::vector<std::tuple<int, int, int, std::vector<double>>>& sa_info);

    /// print overlap matrix between DWMS roots
    void print_overlap(const std::vector<psi::SharedVector>& evecs, const std::string& Sname);

    /// print energy list summary
    void
    print_energy_list(std::string name, const std::vector<std::vector<double>>& energy,
                      const std::vector<std::tuple<int, int, int, std::vector<double>>>& sa_info,
                      bool pass_process = false);
};
} // namespace forte

#endif // DWMS_MRPT2_H
