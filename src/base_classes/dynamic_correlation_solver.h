#ifndef _dynamic_correlation_solver_h_
#define _dynamic_correlation_solver_h_

#include <memory>



#include "integrals/integrals.h"
#include "integrals/active_space_integrals.h"
#include "base_classes/rdms.h"
#include "base_classes/forte_options.h"

namespace forte {

class SCFInfo;

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

    /// Common settings
    void startup();

    /// Nuclear repulsion energy
    double Enuc_;

    /// Frozen core energy
    double Efrzc_;

    /// Printing level
    int print_;

    /// The integral type
    std::string ints_type_;
    /// If ERI density fitted or Cholesky decomposed
    bool eri_df_;

    // ==> DIIS control <==

    /// Cycle number to start DIIS
    int diis_start_;
    /// Minimum number of DIIS vectors
    int diis_min_vec_;
    /// Maximum number of DIIS vectors
    int diis_max_vec_;
    /// Frequency of extrapolating the current DIIS vectors
    int diis_freq_;
};

std::shared_ptr<DynamicCorrelationSolver>
make_dynamic_correlation_solver(const std::string& type, std::shared_ptr<ForteOptions> options,
                                std::shared_ptr<ForteIntegrals> ints,
                                std::shared_ptr<MOSpaceInfo> mo_space_info);

} // namespace forte

#endif // DYNAMIC_CORRELATION_SOLVER_H
