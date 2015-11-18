#include "alternativescasscf.h"
#include <libpsio/psio.h>
#include <libmints/wavefunction.h>
#include <libmints/matrix.h>
#include <libmints/vector.h>
#include <vector>
#include <liboptions/liboptions.h>
#include <libciomr/libciomr.h>
#include <liboptions/liboptions.h>
#include <libmoinfo/libmoinfo.h>
#include <libmints/molecule.h>
#include <libpsio/psio.h>
#include <libpsio/psio.hpp>
#include "helpers.h"

namespace psi { namespace forte {

FiniteTemperatureHF::FiniteTemperatureHF(boost::shared_ptr<Wavefunction> wfn, Options& options, std::shared_ptr<MOSpaceInfo> mo_space)
    : Wavefunction(options, _default_psio_lib_),
      wfn_(wfn),
      mo_space_info_(mo_space),
      options_(options)
{
    copy(wfn);
    startup();
}

void FiniteTemperatureHF::startup()
{
    sMat_ = wfn_->S();
    hMat_ = wfn_->H();
    nmo_  = mo_space_info_->size("ALL");
    print_method_banner({"Finite Temperature Hartree-Fock","Kevin Hannon"});
}
double FiniteTemperatureHF::compute_energy()
{

}
boost::shared_ptr<Matrix> FiniteTemperatureHF::frac_occupation(boost::shared_ptr<Matrix> C, int &iteration, bool &t_done)
{
    double T = 0.0;
    T = options_.get_int("TEMPERATURE");
    int increment = options_.get_int("TEMPERATURE_INCREMENT");
    T /= 3.157746E5;

    std::vector<double> ni(nbf_);
    ef_ = bisection(ni, T);

    double sim = 0.0;
    for(auto& ft: ni){
        sim += ft;
    }



}


}}
