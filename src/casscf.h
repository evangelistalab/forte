#ifndef CASSCF_H
#define CASSCF_H

#include <liboptions/liboptions.h>
#include <libmints/wavefunction.h>
#include <libthce/lreri.h>


#include "integrals.h"
#include "ambit/blocked_tensor.h"
#include "reference.h"
#include "helpers.h"
#include "blockedtensorfactory.h"

namespace psi{ namespace forte{

class CASSCF
{
public:
    /**
     * @brief CASSCF::CASSCF
     * @param options -> Options object
     * @param ints    -> The integral object.  I may not use this as I need the AO based integrals
     * @param mo_space_info -> The MOSpaceInfo object for getting active space information
     * This class will implement the AO based CASSCF by Hohenstein and Martinez.
     * Ref is .  Hohenstein J.Chem.Phys, 142, 224103.
     * This reference has a nice algorithmic flowchart.  Look it up
     *
     */
    CASSCF(Options &options,
           std::shared_ptr<ForteIntegrals> ints, std::shared_ptr<MOSpaceInfo> mo_space_info);
    /// Compute CASSCF given a 1RDM and 2RDM
    void compute_casscf();
    /// Use daniels code to compute Orbital optimization
    //void compute_casscf_soscf();
    ///Return the Converged CMatrix
    SharedMatrix Call(){return Call_;}
    ///Return the final gamma1
    ambit::Tensor gamma1(){return gamma1_;}
    ///Return the final gamma2;
    ambit::Tensor gamma2(){return gamma2_;}
    double E_casscf(){return E_casscf_;}
private:
    /// The active one RDM in the MO basis
    SharedMatrix gamma1M_;
    ambit::Tensor gamma1_;

    /// The active two RDM (may need to be symmetrized)
    ambit::Tensor gamma2_;
    SharedMatrix gamma2M_;
    /// The reference object generated from Francesco's Full CI
    Reference cas_ref_;
    /// The energy computed in FCI with updates from CASSCF and CI
    double E_casscf_;
    /// The OPtions object
    Options options_;
    /// The ForteIntegrals pointer
    boost::shared_ptr<Wavefunction> wfn_;
    std::shared_ptr<ForteIntegrals> ints_;
    /// The mo_space_info
    std::shared_ptr<MOSpaceInfo> mo_space_info_;
    /// The MO Coefficient matrix in Pfitzer ordering in whatever symmetry
    /// this matrix is ao by nmo
    SharedMatrix Call_;
    /// C matrix in the SO basis
    SharedMatrix Ca_sym_;

    /// The dimension for number of molecular orbitals
    Dimension nmopi_;
    /// The number of correlated molecular orbitals (Restricted Core + Active + Restricted_UOCC + Frozen_Virt
    size_t nmo_;
    /// The number of active orbitals
    size_t na_;
    /// The number of irreps
    size_t nirrep_;
    /// The number of SO (AO for C matrices)
    Dimension nsopi_;
    /// the number of restricted_docc
    size_t nrdocc_;
    /// The number of frozen_docc
    size_t nfrozen_;
    /// The number of virtual orbitals
    size_t nvir_;

    /// These member variables are all summarized in Algorithm 1
    /// Equation 9

    /// The Fock matrix due to Frozen core orbitals
    SharedMatrix F_froze_;
    /// The core Fock Matrix
    SharedMatrix F_core_;
    /// The F_act_ -> ie the fock matrix of nmo by nmo generated using the all active portion of the OPM
    /// Equation 10
    SharedMatrix F_act_;
    /// Intermediate in forming orbital gradient matrix
    SharedMatrix Y_;
    /// Z intermediate
    SharedMatrix Z_;
    /// The Orbital gradient
    SharedMatrix g_;
    /// The diagonal Hessian
    SharedMatrix d_;
    /// The type of integrals
    IntegralType int_type_;
    /// Do a Complete SOSCF using Daniels' code
    bool do_soscf_;



    /// private functions

    /// Perform a CAS-CI with the updated MO coefficients
    void cas_ci();
    /// This function will implement steps 4 and 9 of algorithm
    void form_fock_core();
    ///Implement step 9 of algoritm
    void form_fock_active();
    /// Assemble the orbital gradient (10-15)
    void orbital_gradient();
    /// Assemble the diagonal Hessian (20-22)
    void diagonal_hessian();
    /// check the cas_ci energy with spin-free RDM
    double cas_check(Reference cas);
    /// Make C_matrix symmetry aware from SO C
    boost::shared_ptr<Matrix> make_c_sym_aware();

    void startup();

    /// DEBUG PRINTING
    bool casscf_debug_print_;
    /// Compute core Hamiltonian in SO basis
    boost::shared_ptr<Matrix> compute_so_hamiltonian();
    /// Set the dferi object
    boost::shared_ptr<DFERI> set_df_object();
    /// Get the Frozen Orbs in SO basis
    std::map<std::string, boost::shared_ptr<Matrix> > orbital_subset_helper();
    /// set frozen_core_orbitals
    boost::shared_ptr<Matrix> set_frozen_core_orbitals();


};

}}

#endif // CASSCF_H
