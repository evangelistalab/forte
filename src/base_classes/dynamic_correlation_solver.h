#ifndef _dynamic_correlation_solver_h_
#define _dynamic_correlation_solver_h_

#include <memory>

#include "psi4/liboptions/liboptions.h"

#include "integrals/integrals.h"
#include "integrals/active_space_integrals.h"
#include "base_classes/rdms.h"
#include "base_classes/coupling_coefficients.h"

namespace forte {

class SCFInfo;
class ForteOptions;

class DynamicCorrelationSolver {
  public:
    /**
     * Constructor
     * @param ref_wfn The reference wavefunction object
     * @param options The main options object
     * @param ints A pointer to an allocated integral object
     * @param mo_space_info The MOSpaceInfo object
     */
    DynamicCorrelationSolver(RDMs rdms, std::shared_ptr<SCFInfo> scf_info,
                             std::shared_ptr<ForteOptions> options,
                             std::shared_ptr<ForteIntegrals> ints,
                             std::shared_ptr<MOSpaceInfo> mo_space_info);

    /// Compute energy
    virtual double compute_energy() = 0;

    /// Compute dressed Hamiltonian
    virtual std::shared_ptr<ActiveSpaceIntegrals> compute_Heff_actv() = 0;

    /// Destructor
    virtual ~DynamicCorrelationSolver() = default;

    /// Set coupling coefficients
    /// TODO: remove this when implemented more efficient way of computing CI response
    virtual void set_coupling_coefficients(const CouplingCoefficients& cc) { coupling_coefficients_ = cc; }

    /// Set CI coefficients
    /// TODO: remove this when implemented more efficient way of computing CI response
    virtual void set_ci_vectors(const std::vector<ambit::Tensor>& ci_vectors) {
        ci_vectors_ = ci_vectors;
    }

  protected:
    /// The molecular integrals
    std::shared_ptr<ForteIntegrals> ints_;

    /// The MO space info
    std::shared_ptr<MOSpaceInfo> mo_space_info_;

    /// The RDMs and cumulants of the reference wave function
    RDMs rdms_;

    /// The SCFInfo
    std::shared_ptr<SCFInfo> scf_info_;

    /// The ForteOptions
    std::shared_ptr<ForteOptions> foptions_;

    /// The coupling coefficients
    CouplingCoefficients coupling_coefficients_;

    /// The CI coefficients
    std::vector<ambit::Tensor> ci_vectors_;
};

std::shared_ptr<DynamicCorrelationSolver>
make_dynamic_correlation_solver(const std::string& type, std::shared_ptr<ForteOptions> options,
                                std::shared_ptr<ForteIntegrals> ints,
                                std::shared_ptr<MOSpaceInfo> mo_space_info);

} // namespace forte

#endif // DYNAMIC_CORRELATION_SOLVER_H
