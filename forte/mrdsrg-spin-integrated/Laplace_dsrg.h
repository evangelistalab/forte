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
    void clear_maps(const std::string& algorithm);
    void fill_maps(const std::string& algorithm, const int& nweight, const int& nocc,
                   const int& nact, const int& nvir);
    void compute_coulomb(const std::string& algorithm, const int& nocc, const int& nact,
                         const int& nvir);
    void compute_exchange(const std::string& algorithm, const int& nocc, const int& nact,
                          const int& nvir);
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

    /// CCAV
    double theta_NB_ccav_;
    double theta_NB_IAP_ccav_;
    double theta_ij_ccav_;
    double theta_schwarz_ccav_;
    double theta_ij_sqrt_ccav_;

    /// Energy
    double E_J_;
    double E_K_;

    /// SparseMaps related
    SparseMap ao_list_per_q_;
    SparseMap i_bar_p_up_; /// Construct [ibar]_p
    SparseMap a_bar_p_up_; /// Construct [abar]_p
    SparseMap ibar_p_;     /// Construct {ibar}_p
    SparseMap xbar_p_;     /// Construct {xbar}_p
    SparseMap ibar_x_p_;   /// Construct {ibar_x}_p
    SparseMap P_ibar_;     /// Construct {P}_ibar.  Obtain this by "inversion" of {ibar}_p
    SparseMap P_xbar_;     /// Construct {P}_xbar
    SparseMap P_ibar_x_;   /// Construct {p}_ibar_x
    SparseMap abar_ibar_;  /// Construct {a_bar}_ibar
    SparseMap abar_xbar_;  /// Construct {a_bar}_xbar
    SparseMap xbar_ibar_;  /// Construct {x_bar}_ibar

    psi::SharedMatrix N_pu_;
    std::vector<psi::SharedMatrix> P_iu_;
    std::vector<psi::SharedMatrix> T_ibar_i_list_;
    std::vector<SparseMap> xbar_u_p_;
    std::vector<std::vector<psi::SharedMatrix>> P_xbar_u_cavv_;
    std::vector<std::vector<psi::SharedMatrix>> P_xbar_u_ccav_;
    std::vector<psi::SharedMatrix> P_ibar_u_;             /// Construct (P|ibar u)
    std::vector<psi::SharedMatrix> P_ibar_abar_;          /// Construct (P|ibar abar)
    std::vector<psi::SharedMatrix> P_xbar_abar_;          /// Construct (P|xbar abar)
    std::vector<psi::SharedMatrix> P_ibar_xbar_;          /// Construct (P|ibar xbar)
    std::vector<psi::SharedMatrix> i_bar_a_bar_P_;        /// Construct (ibar abar|P)
    std::vector<psi::SharedMatrix> x_bar_a_bar_P_;        /// Construct (xbar abar|P)
    std::vector<psi::SharedMatrix> i_bar_x_bar_P_;        /// Construct (ibar xbar|P)
    std::vector<psi::SharedMatrix> i_bar_a_bar_P_sliced_; /// Constrcut Sliced (ibar abar|P);
    std::vector<psi::SharedMatrix> x_bar_a_bar_P_sliced_; /// Construct Sliced (xbar abar|P)
    std::vector<psi::SharedMatrix> i_bar_x_bar_P_sliced_; /// Construct Sliced (ibar xbar|P)
    psi::SharedMatrix Z_pq_;                              /// Construct Z_pq
    psi::SharedMatrix ZA_pq_;                             /// Construct ZA_pq
    std::vector<psi::SharedMatrix> B_ia_Q_;               /// Construct B_ia_Q
    std::vector<psi::SharedMatrix> B_xa_Q_;               /// Construct B_xa_Q
    std::vector<psi::SharedMatrix> B_ix_Q_;               /// Construct B_ix_Q
    psi::SharedMatrix Q_ia_;                              /// Construc Q_ia square;
    psi::SharedMatrix Q_xa_;
    psi::SharedMatrix Q_ix_;
    std::vector<int> vir_intersection_per_ij_;
    std::vector<int> aux_intersection_per_ij_;
    std::vector<int> aux_in_B_i_;
    std::vector<int> aux_in_B_j_;
    std::vector<int> aux_in_B_ix_;
    std::vector<int> vir_intersection_per_ix_;
    std::vector<int> aux_intersection_per_ix_;
    std::vector<int> aux_intersection_per_ijx_;
    std::vector<int> aux_intersection_per_ixj_;

    /// CCVV Cholesky
    std::vector<psi::SharedMatrix> Occupied_cholesky_;
    std::vector<psi::SharedMatrix> Virtual_cholesky_;
    psi::SharedMatrix Cholesky_Occ_;

    std::vector<psi::SharedMatrix> Occupied_cholesky_abs_;
    std::vector<psi::SharedMatrix> Virtual_cholesky_abs_;
    psi::SharedMatrix Cholesky_Occ_abs_;

    /// CAVV Cholesky
    std::vector<psi::SharedMatrix> Active_cholesky_;
    std::vector<psi::SharedMatrix> Active_cholesky_abs_;
    psi::SharedMatrix Gamma1_mat_;

    /// CCAV Cholesky
    psi::SharedMatrix Eta1_mat_;
};
} // namespace forte

#endif
