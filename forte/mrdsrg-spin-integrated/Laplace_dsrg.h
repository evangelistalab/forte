#ifndef _laplace_dsrg_h_
#define _laplace_dsrg_h_

#include "master_mrdsrg.h"

using namespace ambit;

namespace forte {

class LaplaceDSRG {
  public:
    LaplaceDSRG(std::shared_ptr<ForteOptions> options, std::shared_ptr<ForteIntegrals> ints,
                std::shared_ptr<MOSpaceInfo> mo_space_info, psi::SharedVector epsilon_rdocc,
                psi::SharedVector epsilon_virtual, psi::SharedVector epsilon_active,
                psi::SharedMatrix Gamma1, psi::SharedMatrix Eta1);

    ~LaplaceDSRG();

    void prepare_cholesky_coeff(std::shared_ptr<AtomicOrbitalHelper> ao_helper_ptr,
                                const std::string& algorithm);

    void Load_int(double& theta_NB, const std::string& algorithm);

    double compute_ccvv();
    double compute_cavv();
    double compute_ccav();
    void print_header();
    size_t vir_start() { return vir_start_; }

  protected:
    std::shared_ptr<ForteOptions> foptions_;
    std::shared_ptr<ForteIntegrals> ints_;
    std::shared_ptr<MOSpaceInfo> mo_space_info_;
    psi::SharedVector eps_rdocc_;
    psi::SharedVector eps_virtual_;
    psi::SharedVector eps_active_;
    psi::SharedMatrix Cwfn_;
    psi::SharedMatrix C_pq_;
    psi::SharedMatrix S_;
    SparseMap i_p_;

    std::shared_ptr<psi::BasisSet> primary_;
    std::shared_ptr<psi::BasisSet> auxiliary_;

    size_t nmo_;
    size_t nthree_;
    size_t nfrozen_;
    size_t ncore_;
    size_t nactive_;
    size_t nvirtual_;

    /// Thresholds
    double Omega_;
    double laplace_threshold_;
    double vir_tol_;
    size_t vir_start_;
    size_t weights_;

    /// CCVV
    double theta_NB_;
    double theta_NB_IAP_;
    double theta_ij_;
    double theta_schwarz_;
    double theta_ij_sqrt_;

    /// CAVV
    double theta_NB_cavv_;
    double theta_XNB_cavv_;
    double theta_NB_IAP_cavv_;
    double theta_NB_XAP_cavv_;
    double theta_ij_cavv_;
    double theta_schwarz_cavv_;
    double theta_ij_sqrt_cavv_;

    std::vector<std::vector<psi::SharedMatrix>> P_xbar_u_cavv_;

    /// CCAV
    double theta_NB_ccav_;
    double theta_NB_IAP_ccav_;
    double theta_ij_ccav_;
    double theta_schwarz_ccav_;
    double theta_ij_sqrt_ccav_;
    std::vector<std::vector<psi::SharedMatrix>> P_xbar_u_ccav_;

    /// Energy
    double E_J_;
    double E_K_;

    /// Common use
    std::vector<psi::SharedMatrix> P_iu_;
    std::vector<psi::SharedMatrix> T_ibar_i_list_;
    SparseMap ao_list_per_q_;

    /// CCVV
    std::vector<psi::SharedMatrix> Occupied_cholesky_;
    std::vector<psi::SharedMatrix> Virtual_cholesky_;
    psi::SharedMatrix Cholesky_Occ_;

    std::vector<psi::SharedMatrix> Occupied_cholesky_abs_;
    std::vector<psi::SharedMatrix> Virtual_cholesky_abs_;
    psi::SharedMatrix Cholesky_Occ_abs_;

    psi::SharedMatrix N_pu_;

    /// CAVV
    std::vector<psi::SharedMatrix> Active_cholesky_;
    std::vector<psi::SharedMatrix> Active_cholesky_abs_;
    psi::SharedMatrix Gamma1_mat_;
    std::vector<SparseMap> xbar_u_p_;

    /// CCAV
    psi::SharedMatrix Eta1_mat_;
};
} // namespace forte

#endif
