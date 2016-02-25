#ifndef DMRG_H
#define DMRG_H

#include <liboptions/liboptions.h>
#include <libmints/wavefunction.h>
#include <libfock/jk.h>
#include "reference.h"
#include "integrals.h"
#include "helpers.h"

#include "chemps2/Irreps.h"
#include "chemps2/Problem.h"
#include "chemps2/CASSCF.h"
#include "chemps2/Initialize.h"
#include "chemps2/EdmistonRuedenberg.h"

namespace psi { namespace  forte {


class DMRGSCF : public Wavefunction
{
public:
    DMRGSCF(SharedWavefunction ref_wfn, Options& options, std::shared_ptr<MOSpaceInfo> mo_space_info, std::shared_ptr<ForteIntegrals> ints);
    double compute_energy();

    Reference reference()
    {
        return dmrg_ref_;
    }

private:
    Reference dmrg_ref_;
    std::shared_ptr<MOSpaceInfo> mo_space_info_;
    std::shared_ptr<ForteIntegrals> ints_;
    void set_up_ints();
    void compute_reference(double* one_rdm, double* two_rdm, double* three_rdm, CheMPS2::DMRGSCFindices * iHandler);
    ///Ported over codes from DMRGSCF plugin
    void startup();

    ///Form the active fock matrix
    void buildJK(SharedMatrix MO_RDM, SharedMatrix MO_JK, SharedMatrix Cmat, boost::shared_ptr<JK> myJK);
    ///Form Inactive fock matrix
    void buildQmatOCC( CheMPS2::DMRGSCFmatrix * theQmatOCC,
                       CheMPS2::DMRGSCFindices * iHandler,
                       SharedMatrix MO_RDM, SharedMatrix MO_JK, SharedMatrix Cmat,
                       boost::shared_ptr<JK> myJK);

    ///Form active fock matrix
    void buildQmatACT( CheMPS2::DMRGSCFmatrix * theQmatACT, CheMPS2::DMRGSCFindices * iHandler, double * DMRG1DM,
                      SharedMatrix MO_RDM, SharedMatrix MO_JK,
                      SharedMatrix Cmat,
                      boost::shared_ptr<JK> myJK);

    void buildHamDMRG( boost::shared_ptr<IntegralTransform> ints, boost::shared_ptr<MOSpace> Aorbs_ptr,
                  CheMPS2::DMRGSCFmatrix * theQmatOCC, CheMPS2::DMRGSCFindices * iHandler,
                  CheMPS2::Hamiltonian * HamDMRG, boost::shared_ptr<PSIO> psio);
    void buildHamDMRGForte(
                  CheMPS2::DMRGSCFmatrix * theQmatOCC, CheMPS2::DMRGSCFindices * iHandler,
                  CheMPS2::Hamiltonian * HamDMRG, std::shared_ptr<ForteIntegrals> ints);


    void fillRotatedTEI_coulomb( boost::shared_ptr<IntegralTransform> ints, boost::shared_ptr<MOSpace> OAorbs_ptr,
                            CheMPS2::DMRGSCFmatrix * theTmatrix, CheMPS2::DMRGSCFintegrals * theRotatedTEI,
                            CheMPS2::DMRGSCFindices * iHandler, boost::shared_ptr<PSIO> psio
                            );
    void fillRotatedTEI_exchange( boost::shared_ptr<IntegralTransform> ints, boost::shared_ptr<MOSpace> OAorbs_ptr,
                                  boost::shared_ptr<MOSpace> Vorbs_ptr, CheMPS2::DMRGSCFintegrals * theRotatedTEI,
                                  CheMPS2::DMRGSCFindices * iHandler, boost::shared_ptr<PSIO> psio );
    void copyUNITARYtoPSIMX( CheMPS2::DMRGSCFunitary * unitary, CheMPS2::DMRGSCFindices * iHandler, SharedMatrix target );
    void update_WFNco( CheMPS2::DMRGSCFmatrix * Coeff_orig, CheMPS2::DMRGSCFindices * iHandler,
                       CheMPS2::DMRGSCFunitary * unitary,
                       SharedMatrix work1, SharedMatrix work2 );


    ///Makes sure that CHEMPS2 and PSI4 have same symmetry
    int chemps2_groupnumber(const string SymmLabel);
    ///Copies PSI4Matrices to CHEMPS2 matrices and vice versa
    void copyPSIMXtoCHEMPS2MX( SharedMatrix source, CheMPS2::DMRGSCFindices * iHandler, CheMPS2::DMRGSCFmatrix * target );
    void copyCHEMPS2MXtoPSIMX( CheMPS2::DMRGSCFmatrix * source, CheMPS2::DMRGSCFindices * iHandler, SharedMatrix target );

};

}}
#endif // DMRG_H
