#include "scf_info.h"

namespace forte {

SCFInfo::SCFInfo(psi::SharedWavefunction wfn) : doccpi_(wfn->doccpi()), 
                 soccpi_(wfn->soccpi()), energy_(wfn->reference_energy()), 
                 epsilon_a_(wfn->epsilon_a()), epsilon_b_(wfn->epsilon_b())
{

}


psi::Dimension SCFInfo::doccpi() { return doccpi_; }

psi::Dimension SCFInfo::soccpi() { return soccpi_; }

double SCFInfo::reference_energy() { return energy_; }

std::shared_ptr<psi::Vector> SCFInfo::epsilon_a() { return epsilon_a_; }

std::shared_ptr<psi::Vector> SCFInfo::epsilon_b() { return epsilon_b_; }

}
