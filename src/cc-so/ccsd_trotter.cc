#include "psi4/libpsi4util/process.h"
#include "psi4/libmints/molecule.h"

#include "base_classes/mo_space_info.h"
#include "base_classes/scf_info.h"
#include "helpers/helpers.h"
#include "helpers/printing.h"
#include "helpers/timer.h"
#include "cc.h"

using namespace psi;

namespace forte {

void CC_SO::compute_ccsd_trotter(BlockedTensor& H1, BlockedTensor& H2, BlockedTensor& T1,
                                 BlockedTensor& T2, double& C0, BlockedTensor& C1,
                                 BlockedTensor& C2) {
    // scale amplitudes
    T1.scale(1.0 / trotter_level_);
    T2.scale(1.0 / trotter_level_);

    // zero output
    C0 = 0.0;
    C1.zero();
    C2.zero();

    // prepare intermediates
    double X0 = 0.0;
    auto X1 = ambit::BlockedTensor::build(ambit::CoreTensor, "X1", {"gg"});
    auto X2 = ambit::BlockedTensor::build(ambit::CoreTensor, "X2", {"gggg"});
    auto Y1 = ambit::BlockedTensor::build(ambit::CoreTensor, "Y1", {"gg"});
    auto Y2 = ambit::BlockedTensor::build(ambit::CoreTensor, "Y2", {"gggg"});

    Y1["pq"] = H1["pq"];
    Y2["pqrs"] = H2["pqrs"];

    // transform Hamiltonian
    for (int i = 1; i < trotter_level_ + 1; ++i) {
        compute_ccsd_hamiltonian(Y1, Y2, T1, T2, X0, X1, X2);

        double Z0 = X0;

        if (trotter_sym_) {
            Y1["pq"] = 0.5 * X1["pq"];
            Y1["pq"] += 0.5 * X1["qp"];
            Y2["pqrs"] = 0.5 * X2["pqrs"];
            Y2["pqrs"] += 0.5 * X2["rspq"];
        } else {
            Y1["pq"] = X1["qp"];
            Y2["pqrs"] = X2["rspq"];
        }

        compute_ccsd_hamiltonian(Y1, Y2, T1, T2, X0, X1, X2);

        C0 += Z0 + X0;
        if (trotter_sym_) {
            Y1["pq"] = 0.5 * X1["pq"];
            Y1["pq"] += 0.5 * X1["qp"];
            Y2["pqrs"] = 0.5 * X2["pqrs"];
            Y2["pqrs"] += 0.5 * X2["rspq"];
        } else {
            Y1["pq"] = X1["qp"];
            Y2["pqrs"] = X2["rspq"];
        }

        outfile->Printf("\n    Trotter iter. %2d corr. energy: %20.12f = %20.12f + %20.12f", i, Z0 + X0, Z0, X0);
    }

    // symmetrize Hamiltonian
    C1["pq"] = 0.5 * Y1["pq"];
    C1["pq"] += 0.5 * Y1["qp"];
    C2["pqrs"] = 0.5 * Y2["pqrs"];
    C2["rspq"] += 0.5 * Y2["rspq"];

    // unscale amplitudes
    T1.scale(trotter_level_);
    T2.scale(trotter_level_);
}

} // namespace forte
