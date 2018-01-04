#ifndef _dwms_mrpt2_h_
#define _dwms_mrpt2_h_

#include "psi4/liboptions/liboptions.h"

#include "../integrals/integrals.h"
#include "../helpers.h"
#include "../fci_mo.h"

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

    /// precompute energy -- SA-CASCI or SA-DSRG-PT2
    std::shared_ptr<FCI_MO> precompute_energy();

    /// perform DSRG-PT2 computation and return the dressed integrals within active space
    std::shared_ptr<FCIIntegrals> compute_dsrg_pt2(std::shared_ptr<FCI_MO> fci_mo,
                                                   Reference& reference);

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

    /// energy of original SA-CASCI
    std::vector<std::vector<double>> Eref_0_;
    /// energy of SA-DSRG-PT2 (if computed)
    std::vector<std::vector<double>> Ept2_0_;
    /// energy of DWMS-DSRG-PT2
    std::vector<std::vector<double>> Ept2_;

    /// irrep symbols
    std::vector<std::string> irrep_symbol_;

    /// multiplicity symbols
    std::vector<std::string> multi_symbol_;

    /// unitary matrices (in active space) from original to semicanonical
    ambit::Tensor Ua_;
    ambit::Tensor Ub_;

    /// print implementaion note
    void print_note();

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
