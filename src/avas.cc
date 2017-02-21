#include "psi4/psi4-dec.h"

#include "avas.h"
#include "psi4/libmints/vector.h"

namespace psi {
namespace forte {

void make_avas(SharedWavefunction ref_wfn, Options& options, SharedMatrix Ps) {
    if (Ps) {
        outfile->Printf("\n  Generating AVAS orbitals\n");

        // Allocate a matrix for the occupied block
        int nocc = ref_wfn->nalpha();
        int nmo = ref_wfn->nmo();
        int nvir = nmo - nocc;
        outfile->Printf("\n  Number of occupied MOs: %6d", nocc);
        outfile->Printf("\n  Number of virtual MOs:  %6d", nvir);
        outfile->Printf("\n");

        auto Socc = std::make_shared<Matrix>("S occupied block", nocc, nocc);
        auto Svir = std::make_shared<Matrix>("S virtual block", nvir, nvir);

        SharedMatrix CPsC = Ps->clone();
        CPsC->transform(ref_wfn->Ca());

        // Grab the occupied block and diagonalize it
        for (int i = 0; i < nocc; i++) {
            for (int j = 0; j < nocc; j++) {
                double value = CPsC->get(i, j);
                Socc->set(i, j, value);
            }
        }
//        Socc->print();

        auto Uocc = std::make_shared<Matrix>("U occupied block", nocc, nocc);
        auto sigmaocc = std::make_shared<Vector>("sigma occupied block", nocc);

        Socc->diagonalize(Uocc, sigmaocc, descending);

//        Uocc->print();
//        sigmaocc->print();

        // Grab the virtual block and diagonalize it
        for (int a = 0; a < nvir; a++) {
            for (int b = 0; b < nvir; b++) {
                double value = CPsC->get(nocc + a, nocc + b);
                Svir->set(a, b, value);
            }
        }
//        Svir->print();

        auto Uvir = std::make_shared<Matrix>("U virtual block", nvir, nvir);
        auto sigmavir = std::make_shared<Vector>("sigma virtual block", nvir);

        Svir->diagonalize(Uvir, sigmavir, descending);

//        Uvir->print();
//        sigmavir->print();


        outfile->Printf("\n  Orbital overlap with ao subspace:\n");
        outfile->Printf("    ========================\n");
        outfile->Printf("    Irrep   MO   <phi|P|phi>\n");
        outfile->Printf("    ------------------------\n");
            for (int i = 0; i < nocc; i++) {
                outfile->Printf("      %1d   %4d    %.6f\n", 2, i + 1,
                                sigmaocc->get(i));
            }
            for (int i = 0; i < nvir; i++) {
                outfile->Printf("      %1d   %4d    %.6f\n", 0, nocc + i + 1,
                                sigmavir->get(i));
            }
        outfile->Printf("    ========================\n");

        // Form the full matrix U
        auto U = std::make_shared<Matrix>("U", nmo, nmo);

        for (int i = 0; i < nocc; i++) {
            for (int j = 0; j < nocc; j++) {
                double value = Uocc->get(i, j);
                U->set(i, j, value);
            }
        }
        for (int a = 0; a < nvir; a++) {
            for (int b = 0; b < nvir; b++) {
                double value = Uvir->get(a, b);
                U->set(a + nocc, b + nocc, value);
            }
        }

        auto Ca_tilde = Matrix::doublet(ref_wfn->Ca(),U);

        ref_wfn->Ca()->copy(Ca_tilde);
    }
}
}
}
