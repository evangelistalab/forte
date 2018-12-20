#include "psi4/libpsi4util/process.h"
#include "psi4/libmints/wavefunction.h"
#include "psi4/libmints/molecule.h"

#include "state_info.h"

namespace forte {

StateInfo::StateInfo(int na, int nb, int multiplicity, int twice_ms, int irrep)
    : na_(na), nb_(nb), multiplicity_(multiplicity), twice_ms_(twice_ms), irrep_(irrep) {}

StateInfo::StateInfo(psi::SharedWavefunction wfn) {
    int charge = psi::Process::environment.molecule()->molecular_charge();
    if (wfn->options()["CHARGE"].has_changed()) {
        charge = wfn->options().get_int("CHARGE");
    }

    int nel = 0;
    int natom = psi::Process::environment.molecule()->natom();
    for (int i = 0; i < natom; i++) {
        nel += static_cast<int>(psi::Process::environment.molecule()->Z(i));
    }
    // If the charge has changed, recompute the number of electrons
    // Or if you cannot find the number of electrons
    nel -= charge;

    multiplicity_ = psi::Process::environment.molecule()->multiplicity();
    if (wfn->options()["MULTIPLICITY"].has_changed()) {
        multiplicity_ = wfn->options().get_int("MULTIPLICITY");
    }

    // If the user did not specify ms determine the value from the input or
    // take the lowest value consistent with the value of "MULTIPLICITY"
    if (wfn->options()["MS"].has_changed()) {
        twice_ms_ = std::round(2.0 * wfn->options().get_double("MS"));
    } else {
        // Default: lowest spin solution
        twice_ms_ = (multiplicity_ + 1) % 2;
    }

    if (((nel - twice_ms_) % 2) != 0)
        throw psi::PSIEXCEPTION("\n\n  FCI: Wrong value of M_s.\n\n");

    na_ = (nel + twice_ms_) / 2;
    nb_ = nel - na_;

    name_ = wfn->molecule()->name();
}

int StateInfo::na() const { return na_; }

int StateInfo::nb() const { return nb_; }

int StateInfo::multiplicity() const { return multiplicity_; }

int StateInfo::twice_ms() const { return twice_ms_; }

int StateInfo::irrep() const { return irrep_; }

std::string StateInfo::name() const { return name_; }

} // namespace forte
