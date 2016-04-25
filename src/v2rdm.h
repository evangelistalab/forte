#ifndef _v2rdm_h_
#define _v2rdm_h_

#include <map>
#include <libiwl/iwl.h>
#include <libpsio/psio.hpp>
#include "integrals.h"
#include "helpers.h"
#include "reference.h"

#define PSIF_V2RDM_D2AA       270
#define PSIF_V2RDM_D2AB       271
#define PSIF_V2RDM_D2BB       272

using namespace ambit;
namespace psi{ namespace forte{

class V2RDM : public Wavefunction
{
public:
    /**
     * V2RDM Constructor
     * @param ref_wfn The reference wavefunction object
     * @param options The main options object
     * @param ints A pointer to an allocated integral object
     * @param mo_space_info The MOSpaceInfo object
     */
    V2RDM(SharedWavefunction ref_wfn, Options& options,
          std::shared_ptr<ForteIntegrals> ints, std::shared_ptr<MOSpaceInfo> mo_space_info);

    /// Destructor
    ~V2RDM();

    /// Returns the reference object of forte
    Reference reference();

protected:
    /// Start-up function called in the constructor
    void startup();

    /// The molecular integrals
    std::shared_ptr<ForteIntegrals>  ints_;

    /// The frozen-core energy
    double frozen_core_energy_;

    /// MO space info
    std::shared_ptr<MOSpaceInfo> mo_space_info_;

    /// Number of irrep
    int nirrep_;
    /// MO per irrep;
    Dimension nmopi_;
    /// Frozen docc per irrep
    Dimension fdoccpi_;
    /// Restricted docc per irrep
    Dimension rdoccpi_;
    /// Active per irrep
    Dimension active_;
    /// Map active absolute index to relative index
    std::map<size_t, size_t> abs_to_rel_;

    /// List of core MOs
    std::vector<size_t> core_mos_;
    /// List of active MOs
    std::vector<size_t> actv_mos_;

    /// Read two particle density
    void read_2pdm();
    /// Build one particle density
    void build_opdm();
    /// Read three particle density
    void read_3pdm();

    /// Compute the reference energy
    double compute_ref_energy();

    /// One particle density matrix
    ambit::Tensor D1a_;    // only in active
    ambit::Tensor D1b_;    // only in active

    /// Two particle density matrix
    ambit::Tensor D2aa_;
    ambit::Tensor D2ab_;
    ambit::Tensor D2bb_;

    /// Three particle density matrix
    ambit::Tensor D3aaa_;
    ambit::Tensor D3aab_;
    ambit::Tensor D3abb_;
    ambit::Tensor D3bbb_;
};

}}
#endif // V2RDM_H
