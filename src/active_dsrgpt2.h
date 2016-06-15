#ifndef _active_dsrgpt2_h_
#define _active_dsrgpt2_h_

#include <libqt/qt.h>
#include <libpsio/psio.hpp>
#include <libpsio/psio.h>
#include <liboptions/liboptions.h>
#include <libmints/vector.h>
#include <libmints/matrix.h>
#include <libmints/wavefunction.h>
#include <libmints/molecule.h>
#include <vector>
#include <string>

#include "integrals.h"
#include "reference.h"
#include "helpers.h"
#include "fci_mo.h"
#include "stl_bitset_determinant.h"
#include "dsrg_mrpt2.h"
#include "dsrg_mrpt3.h"
#include "three_dsrg_mrpt2.h"

namespace psi{ namespace forte{
class ACTIVE_DSRGPT2 : public Wavefunction
{
public:
    /**
     * @brief ACTIVE_DSRGPT2 Constructor
     * @param ref_wfn The reference wavefunction object
     * @param options PSI4 and FORTE options
     * @param ints ForteInegrals
     * @param mo_space_info MOSpaceInfo
     */
    ACTIVE_DSRGPT2(SharedWavefunction ref_wfn, Options &options,
                   std::shared_ptr<ForteIntegrals> ints,
                   std::shared_ptr<MOSpaceInfo> mo_space_info);

    /// Destructor
    ~ACTIVE_DSRGPT2();

    /// Compute energy
    double compute_energy();

private:
    /// Basic Preparation
    void startup();

    /// Integrals
    std::shared_ptr<ForteIntegrals> ints_;

    /// MO space info
    std::shared_ptr<MOSpaceInfo> mo_space_info_;

    /// Name of the code
    std::string code_name_;

    /// Total number of roots
    int total_nroots_;

    /// Number of roots per irrep
    std::vector<int> nrootpi_;

    /// Irrep symbol
    std::vector<std::string> irrep_symbol_;

    /// Reference energies
    std::vector<std::vector<double>> ref_energies_;

    /// DSRGPT2 energies
    std::vector<std::vector<double>> pt_energies_;

    /// Singles (T1) percentage
    std::vector<std::vector<std::pair<int,double>>> t1_percentage_;

    /// Dominant determinants
    std::vector<std::vector<STLBitsetDeterminant>> dominant_dets_;

    /// Compute the excitaion type based on ref_det
    std::string compute_ex_type(const STLBitsetDeterminant& det1, const STLBitsetDeterminant& ref_det);

    /// Print summary
    void print_summary();

    /// Orbital extents of current state
    std::vector<double> current_orb_extents_;

    /// Orbital extents (nirrep BY nrootpi BY <r^2>)
    std::vector<std::vector<std::vector<double>>> orb_extents_;

    /// Flatten the structure of orbital extents in fci_mo and return a vector of <r^2>
    std::vector<double> flatten_fci_orbextents(const std::vector<std::vector<std::vector<double>>>& fci_orb_extents);
};
}}

#endif // ACTIVE_DSRGPT2_H
