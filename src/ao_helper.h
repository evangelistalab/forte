#ifndef _atomic_orbital_h_
#define _atomic_orbital_h_
#include <lib3index/denominator.h>
#include <libmints/mints.h>
#include <psi4-dec.h>
namespace psi { namespace forte {

class AtomicOrbitalHelper
{
protected:
    SharedMatrix AO_Screen_;
    SharedMatrix TransAO_Screen_;

    //LaplaceDenominator Laplace_;
    SharedMatrix Occupied_Laplace_;
    SharedMatrix Virtual_Laplace_;
    double laplace_tolerance_ = 1e-10;

    SharedMatrix CMO_;
    SharedVector eps_rdocc_;
    SharedVector eps_virtual_;

    SharedMatrix POcc_;
    SharedMatrix PVir_;
    void Compute_Psuedo_Density();

    int weights_;
    int nbf_;
    int nrdocc_;
    int nvir_;
    /// How many orbitals does it take to go from occupied to virtual (ie should be active)
    int shift_;
public:
    SharedMatrix AO_Screen(){return AO_Screen_;}
    SharedMatrix TransAO_Screen(){return TransAO_Screen_;}
    SharedMatrix Occupied_Laplace(){return Occupied_Laplace_;}
    SharedMatrix Virtual_Laplace(){return Virtual_Laplace_;}
    SharedMatrix POcc(){return POcc_;}
    SharedMatrix PVir(){return PVir_;}
    int          Weights(){return weights_;}

    AtomicOrbitalHelper(SharedMatrix CMO, SharedVector eps_occ, SharedVector eps_vir, double laplace_tolerance);
    AtomicOrbitalHelper(SharedMatrix CMO, SharedVector eps_occ, SharedVector eps_vir, double laplace_tolerance, int shift);
    /// Compute (mu nu | mu nu)^{(1/2)}
    void Compute_AO_Screen(boost::shared_ptr<BasisSet>& primary);
    void Estimate_TransAO_Screen(boost::shared_ptr<BasisSet>& primary, boost::shared_ptr<BasisSet>& auxiliary);

    ~AtomicOrbitalHelper();

};

}}

#endif
