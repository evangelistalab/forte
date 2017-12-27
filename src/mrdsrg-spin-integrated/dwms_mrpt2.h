#ifndef _dwms_mrpt2_h_
#define _dwms_mrpt2_h_

#include "psi4/liboptions/liboptions.h"

#include "../integrals/integrals.h"
#include "../helpers.h"

namespace psi {
namespace forte {

void set_DWMS_options(ForteOptions& foptions);

void compute_dwms_mrpt2_energy(SharedWavefunction ref_wfn, Options& options,
                               std::shared_ptr<ForteIntegrals> ints,
                               std::shared_ptr<MOSpaceInfo> mo_space_info);

//class DWMS_DSRGPT2 : public Wavefunction {
//  public:
//    /**
//     * @brief DWMS_DSRGPT2 Constructor
//     * @param ref_wfn The reference wavefunction object
//     * @param options PSI4 and FORTE options
//     * @param ints ForteInegrals
//     * @param mo_space_info MOSpaceInfo
//     */
//    DWMS_DSRGPT2(SharedWavefunction ref_wfn, Options& options, std::shared_ptr<ForteIntegrals> ints,
//                 std::shared_ptr<MOSpaceInfo> mo_space_info);

//    /// Destructor
//    ~DWMS_DSRGPT2();

//    /// compute energy and return the ground state energy
//    double compute_energy();

//  private:
//    /// preparation
//    void startup();

//    /// gaussian cutoff for density reweighting
//    double zeta_;

//    /// DWMS algorithm
//    std::string algorithm_;

//    /// compute DWMS weights and return a new sa_info
//    std::vector<std::tuple<int, int, int, std::vector<double>>>
//    compute_dwms_weights(const std::vector<std::tuple<int, int, int, std::vector<double>>>& sa_info,
//                         int entry, int root, const std::vector<std::vector<double>>& energy);

//    /// total number of roots
//    int total_nroots_;

//    /// energy of original SA-CASCI
//    std::vector<std::vector<double>> Eref_0_;
//    /// energy of SA-DSRG-PT2 (if computed)
//    std::vector<std::vector<double>> Ept2_0_;
//    /// energy of ensemble MK vacuum
//    std::vector<std::vector<double>> Eref_;
//    /// energy of DWMS-DSRG-PT2
//    std::vector<std::vector<double>> Ept2_;

//    /// irrep symbols
//    std::vector<std::string> irrep_symbol_;

//    /// multiplicity symbols
//    std::vector<std::string> multi_symbol_;

//    /// unitary matrices (in active space) from original to semicanonical
//    ambit::Tensor Ua_;
//    ambit::Tensor Ub_;

//    /// max level of reduced density matrices for DSRG computations
//    int max_rdm_level_;

//    /// fci_mo for diagonalization
//    std::shared_ptr<FCI_MO> fci_mo_;

//    /// recompute new reference energy
//    /// TODO: figure out this
//    double recompute_Eref();

//    /// print implementaion note
//    void print_note();

//    /// print overlap matrix between DWMS roots
//    void print_overlap(const std::vector<SharedVector>& evecs);

//    /// print energy list summary
//    void print_energy_list(std::string name, const std::vector<std::vector<double>>& energy,
//                           bool pass_process = false);
//};
}
}
#endif // DWMS_MRPT2_H
