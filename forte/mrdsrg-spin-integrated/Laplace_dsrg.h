#ifndef _laplace_dsrg_h_
#define _laplace_dsrg_h_

#include "master_mrdsrg.h"

using namespace ambit;

namespace forte {

class LaplaceDSRG {
  public:
    LaplaceDSRG(std::shared_ptr<ForteOptions> options,
                         std::shared_ptr<ForteIntegrals> ints,
                         std::shared_ptr<MOSpaceInfo> mo_space_info,
                         psi::SharedVector epsilon_rdocc, psi::SharedVector epsilon_virtual,
                         psi::SharedVector epsilon_active, psi::SharedMatrix Gamma1,
                         psi::SharedMatrix Eta1);

    ~LaplaceDSRG();

    double compute_ccvv();
    double compute_cavv();
    double compute_ccav();
    void print_header();
    int vir_start() { return vir_start_; }
    int vir_start_cavv() { return vir_start_cavv_; }
    int vir_start_ccav() { return vir_start_ccav_; }

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

    int nmo_;
    int nthree_;
    int nfrozen_;
    int ncore_;
    int nactive_;
    int nvirtual_;

    /// Thresholds
    double Omega_;
    double laplace_threshold_;
    double vir_tol_;
    /// CCVV
    double theta_NB_;
    double theta_NB_IAP_;
    double theta_ij_;
    double theta_schwarz_;
    double theta_ij_sqrt_;
    int vir_start_;

    /// CAVV
    double theta_NB_cavv_;
    double theta_NB_IAP_cavv_;
    double theta_ij_cavv_;
    double theta_schwarz_cavv_;
    double theta_ij_sqrt_cavv_;
    int vir_start_cavv_;
  

    /// CCAV
    double theta_NB_ccav_;
    double theta_NB_IAP_ccav_;
    double theta_ij_ccav_;
    double theta_schwarz_ccav_;
    double theta_ij_sqrt_ccav_;
    int vir_start_ccav_;

    /// Energy
    double E_J_;
    double E_K_;

    std::vector<psi::SharedMatrix> P_iu;


    /// CCVV
    std::vector<psi::SharedMatrix> Occupied_cholesky_;
    std::vector<psi::SharedMatrix> Virtual_cholesky_;
    psi::SharedMatrix Cholesky_Occ_;

    std::vector<psi::SharedMatrix> Occupied_cholesky_abs_;
    std::vector<psi::SharedMatrix> Virtual_cholesky_abs_;
    psi::SharedMatrix Cholesky_Occ_abs_;


    /// CAVV
    std::vector<psi::SharedMatrix> Occupied_cholesky_cavv_;
    std::vector<psi::SharedMatrix> Occupied_cholesky_cavv_abs_;
    std::vector<psi::SharedMatrix> Active_cholesky_cavv_;
    std::vector<psi::SharedMatrix> Active_cholesky_cavv_abs_;
    std::vector<psi::SharedMatrix> Virtual_cholesky_cavv_;
    std::vector<psi::SharedMatrix> Virtual_cholesky_cavv_abs_;
    psi::SharedMatrix Gamma1_mat_;

    /// CCAV
    std::vector<psi::SharedMatrix> Occupied_cholesky_ccav_;
    std::vector<psi::SharedMatrix> Occupied_cholesky_ccav_abs_;
    std::vector<psi::SharedMatrix> Active_cholesky_ccav_;
    std::vector<psi::SharedMatrix> Active_cholesky_ccav_abs_;
    std::vector<psi::SharedMatrix> Virtual_cholesky_ccav_;
    std::vector<psi::SharedMatrix> Virtual_cholesky_ccav_abs_;
    psi::SharedMatrix Eta1_mat_;
};
} // namespace forte

#endif
