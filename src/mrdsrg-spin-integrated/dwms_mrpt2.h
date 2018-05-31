#ifndef _dwms_mrpt2_h_
#define _dwms_mrpt2_h_

#include "psi4/liboptions/liboptions.h"

#include "../integrals/integrals.h"
#include "../helpers.h"
#include "../fci_mo.h"
#include "../sparse_ci/determinant.h"

namespace psi {
namespace forte {

void set_DWMS_options(ForteOptions& foptions);

class DWMS_DSRGPT2 : public Wavefunction {
  public:
    /**
     * @brief DWMS_DSRGPT2 Constructor
     * @param ref_wfn The reference wavefunction object
     * @param options PSI4 and FORTE options
     * @param ints ForteInegrals
     * @param mo_space_info MOSpaceInfo
     */
    DWMS_DSRGPT2(SharedWavefunction ref_wfn, Options& options, std::shared_ptr<ForteIntegrals> ints,
                 std::shared_ptr<MOSpaceInfo> mo_space_info);

    /// Destructor
    ~DWMS_DSRGPT2();

    /// compute energy and return the ground state energy
    double compute_energy();

  private:
    /// The molecular integrals
    std::shared_ptr<ForteIntegrals> ints_;

    /// The MO space info
    std::shared_ptr<MOSpaceInfo> mo_space_info_;

    /// preparation
    void startup();

    /// gaussian cutoff for density reweighting
    double zeta_;

    /// DWMS algorithm
    std::string algorithm_;

    /// use what energies to determine the weights
    /// and what CI vectors to do DWMS-DSRG
    std::string dwms_ref_;

    /// DWMS correlation level
    std::string dwms_corrlv_;

    /// form Hbar3 for DSRG-MRPT2
    bool do_hbar3_;

    //    /// use what energies to determine the weights
    //    std::string dwms_e_;

    //    /// use what CI vectors to perform multi-state computation
    //    std::string dwms_ci_;

    /// precompute energy -- CASCI or SA-DSRG-PT2
    std::shared_ptr<FCI_MO> precompute_energy_old();

    /// precompute energy -- CASCI or SA-DSRG-PT2
    std::shared_ptr<FCI_MO> precompute_energy();

    /// perform DSRG-PT2 computation and return the dressed integrals within active space
    std::shared_ptr<FCIIntegrals> compute_dsrg_pt(std::shared_ptr<MASTER_DSRG>& dsrg_pt,
                                                  Reference& reference, std::string level = "PT2");

    /// compute DWMS energies by diagonalizing separate Hamiltonians
    double compute_dwms_energy_separated_H();

    /// compute MS or XMS energies
    double compute_dwms_energy();

    /// compute DWSA energies
    double compute_dwsa_energy();

    /// compute Reference
    Reference compute_Reference(CI_RDMS& ci_rdms, bool do_cumulant);

    /// compute state-averaged Reference
    Reference
    compute_Reference_SA(const std::vector<det_vec>& p_spaces,
                         const std::vector<SharedMatrix>& civecs,
                         const std::vector<std::tuple<int, int, int, std::vector<double>>>& info);

    /// compute Fock matrix within the active space
    void compute_Fock_actv(const det_vec& p_space, SharedMatrix civecs, ambit::Tensor Fa,
                           ambit::Tensor Fb);

    /// rotate CI vectors according to XMS
    SharedMatrix xms_rotate_civecs(const det_vec& p_space, SharedMatrix civecs, ambit::Tensor Fa,
                                   ambit::Tensor Fb);

    /// initial guesses if DWMS-1 or DWMS-AVG1
    std::vector<std::vector<SharedVector>> initial_guesses_;

    /// compute DWMS weights and return a new sa_info
    std::vector<std::tuple<int, int, int, std::vector<double>>>
    compute_dwms_weights(const std::vector<std::tuple<int, int, int, std::vector<double>>>& sa_info,
                         int entry, int root, const std::vector<std::vector<double>>& energy);

    /// max level computed for reduced density in DSRG
    int max_rdm_level_;

    /// active space type vcis or vcisd if true
    bool actv_vci_;

    /// if using factorized integrals
    bool eri_df_;

    /// a shared_ptr of FCIIntegrals (mostly used in CI_RDMS)
    std::shared_ptr<FCIIntegrals> fci_ints_;

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

    /// print implementaion note on separated H scheme
    void print_note();

    /// print implementation note on MS or XMS
    void print_note_ms();

    /// print implementation note on SA or XSA
    void print_note_sa();

    /// print current job title
    void print_current_title(int multi, int irrep, int root);

    /// print sa_info
    void print_sa_info(const std::string& name,
                       const std::vector<std::tuple<int, int, int, std::vector<double>>>& sa_info);

    /// print overlap matrix between DWMS roots
    void print_overlap(const std::vector<SharedVector>& evecs, const std::string& Sname);

    /// print energy list summary
    void
    print_energy_list(std::string name, const std::vector<std::vector<double>>& energy,
                      const std::vector<std::tuple<int, int, int, std::vector<double>>>& sa_info,
                      bool pass_process = false);
};
}
}
#endif // DWMS_MRPT2_H
