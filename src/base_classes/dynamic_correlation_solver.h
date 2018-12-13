#ifndef _dynamic_correlation_solver_h_
#define _dynamic_correlation_solver_h_

#include <memory>

#include "psi4/liboptions/liboptions.h"
#include "psi4/libmints/wavefunction.h"

#include "integrals/integrals.h"
#include "fci/fci_integrals.h"
#include "reference.h"

namespace psi {
namespace forte {
class DynamicCorrelationSolver : public Wavefunction {
  public:
    /**
     * Constructor
     * @param ref_wfn The reference wavefunction object
     * @param options The main options object
     * @param ints A pointer to an allocated integral object
     * @param mo_space_info The MOSpaceInfo object
     */
    DynamicCorrelationSolver(Reference reference, SharedWavefunction ref_wfn, Options& options,
                             std::shared_ptr<ForteIntegrals> ints,
                             std::shared_ptr<MOSpaceInfo> mo_space_info);

    /// Compute energy
    virtual double compute_energy() = 0;

    /// Compute dressed Hamiltonian
    virtual std::shared_ptr<FCIIntegrals> compute_Heff_actv() = 0;

    /// Destructor
    virtual ~DynamicCorrelationSolver() = default;

  protected:
    /// The molecular integrals
    std::shared_ptr<ForteIntegrals> ints_;

    /// The MO space info
    std::shared_ptr<MOSpaceInfo> mo_space_info_;

    /// The reference object (cumulants)
    Reference reference_;
};
}
}
#endif // DYNAMIC_CORRELATION_SOLVER_H
