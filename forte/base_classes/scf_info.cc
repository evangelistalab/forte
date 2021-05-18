#include "scf_info.h"

namespace forte {

SCFInfo::SCFInfo(psi::SharedWavefunction wfn)
    : doccpi_(wfn->doccpi()), soccpi_(wfn->soccpi()), energy_(wfn->energy()),
      epsilon_a_(wfn->epsilon_a()), epsilon_b_(wfn->epsilon_b()) {}

SCFInfo::SCFInfo(const psi::Dimension& doccpi, const psi::Dimension& soccpi,
                 double reference_energy, std::shared_ptr<psi::Vector> epsilon_a,
                 std::shared_ptr<psi::Vector> epsilon_b)
    : doccpi_(doccpi), soccpi_(soccpi), energy_(reference_energy), epsilon_a_(epsilon_a),
      epsilon_b_(epsilon_b) {}

psi::Dimension SCFInfo::doccpi() { return doccpi_; }

psi::Dimension SCFInfo::soccpi() { return soccpi_; }

double SCFInfo::reference_energy() { return energy_; }

std::shared_ptr<psi::Vector> SCFInfo::epsilon_a() { return epsilon_a_; }

std::shared_ptr<psi::Vector> SCFInfo::epsilon_b() { return epsilon_b_; }

} // namespace forte
