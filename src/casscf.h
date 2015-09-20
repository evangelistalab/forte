#ifndef CASSCF_H
#define CASSCF_H

#include <liboptions/liboptions.h>
#include <libmints/wavefunction.h>

#include "integrals.h"
#include "ambit/blocked_tensor.h"
#include "reference.h"
#include "helpers.h"
#include "blockedtensorfactory.h"

namespace psi{ namespace forte{

/*
 * For the first implementation of this method, I will just try and get it to work with DF.
 *
 *
 */

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
    void compute_casscf();
private:
    /// The active one RDM in the MO basis
    ambit::Tensor gamma1_;
    /// The active two RDM (may need to be symmetrized)
    ambit::Tensor gamma2_;
    /// The reference object generated from Francesco's Full CI
    Reference cas_ref_;
    /// The OPtions object
    Options options_;
    /// The ForteIntegrals pointer
    std::shared_ptr<ForteIntegrals> ints_;
    /// The mo_space_info
    std::shared_ptr<MOSpaceInfo> mo_space_info_;
    /// The MO Coefficient matrix in Pfitzer ordering in whatever symmetry
    SharedMatrix Call_;
    /// The MO Coefficient in the in the Pfitzer ordering for C1
    SharedMatrix C_sym_aware_;

    /// The dimension for number of molecular orbitals
    Dimension nmopi_;
    /// The number of molecular orbitals (Frozen Core + Restricted Core + Active + Restricted_UOCC + Frozen_Virt
    size_t nmo_;
    /// The number of active orbitals
    size_t na_;
    /// The number of irreps
    ///size_t nirrep_;

    /// These member variables are all summarized in Algorithm 1
    /// The core Fock Matrix
    /// Equation 9

    ambit::Tensor F_core_;
    /// The F_act_ -> ie the fock matrix of nmo by nmo generated using the all active portion of the OPM
    /// Equation 10
    ambit::Tensor F_act_;
    /// Intermediate in forming orbital gradient matrix
    ambit::Tensor Y_;
    /// Z intermediate
    ambit::Tensor Z_;
    /// The Orbital gradient
    ambit::Tensor g_;
    /// The diagonal Hessian
    ambit::Tensor d_;



    /// private functions

    /// Perform a CAS-CI with the updated MO coefficients
    void cas_ci();
    /// This function will implement steps 4 and 9 of algorithm
    void form_fock_core();
    ///Implement step 9 of algoritm
    void form_fock_active();
    /// Assemble the orbital gradient (10-15)
    void orbital_gradient();

    void startup();


};

}}

#endif // CASSCF_H
