/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2017 by its authors (see COPYING, COPYING.LESSER, AUTHORS).
 *
 * The copyrights for code used from other parties are included in
 * the corresponding files.
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program.  If not, see http://www.gnu.org/licenses/.
 *
 * @END LICENSE
 */

#include "psi4/libpsi4util/process.h"
#include "psi4/libmints/molecule.h"
#include "psi4/libmints/wavefunction.h"
#include "psi4/libmints/mintshelper.h"
#include "psi4/liboptions/liboptions.h"
#include "psi4/libfock/jk.h"
#include "math.h"
#include "ownscf.h"
#include "forte_options.h"

namespace psi {
namespace forte {

// using namespace ambit;

OwnSCF::OwnSCF(SharedWavefunction ref_wfn, Options& options, std::shared_ptr<ForteIntegrals> ints,
               std::shared_ptr<MOSpaceInfo> mo_space_info)
    : Wavefunction(options), ints_(ints), mo_space_info_(mo_space_info) {

    shallow_copy(ref_wfn);
    ref_wfn_ = ref_wfn;
    print_method_banner({"Independent Hartree-Fock Calculation", "Nan He"});

    startup();
}

void set_SCF_options(ForteOptions& foptions) {
    foptions.add_int("MAXCYC", 50, "Max cycle of HF iteration");
    foptions.add_str("INTEGRAL_METHOD", "JK", "Method of integral calculation");
    foptions.add_double("E_CONV", 1e-10, "The energy convergence criteria");
    foptions.add_double("D_CONV", 1e-10, "The density convergence criteria");
}

void OwnSCF::startup() {
    // Read Options
    E_conv_ = options_.get_double("E_CONV");
    D_conv_ = options_.get_double("D_CONV");
    Maxcyc_ = options_.get_int("MAXCYC");
}


SharedMatrix AO_TEI_scf(SharedWavefunction wfn) {
    MintsHelper mints(wfn);
    SharedMatrix TEI = mints.ao_eri();
    return TEI;
}

SharedMatrix build_Cocc_scf(SharedMatrix C, int nirrep, Dimension nmopi, Dimension noccpi) {
	SharedMatrix C_occ(new Matrix("C_occ", nirrep, nmopi, nmopi)); // make it ready for
																   // changes!!here
	for (int h = 0; h < nirrep; ++h) {
		for (int u = 0; u < nmopi[h]; ++u) {
			for (int i = 0; i < noccpi[h]; ++i) {
				C_occ->set(h, u, i, C->get(h, u, i));
			}
		}
	}
	return C_occ;
}

double eri_index_scf(SharedMatrix TEI, int u, int v, int l, int s, Dimension nmopi) {
    return TEI->get(u * nmopi[0] + v, l * nmopi[0] + s); // Currently works only for C1 !!!
}

void build_D_scf(SharedMatrix C, Dimension noccpi, SharedMatrix D) {
    int nirrep = C->nirrep();
    Dimension nmopi = C->colspi();
    for (int h = 0; h < nirrep; ++h) {
        for (int u = 0; u < nmopi[h]; ++u) {
            for (int v = 0; v < nmopi[h]; ++v) {
                double tmp = 0.0;
                for (int m = 0; m < noccpi[h]; ++m) {
                    tmp += C->get(h, u, m) * C->get(h, v, m);
                }
                D->set(h, u, v, tmp);
            }
        }
    }
}


double calculate_HF_energy(SharedMatrix hpF, SharedMatrix D) {
    // hpF = h_core + Fock_current
    int nirrep = D->nirrep();
    Dimension nmopi = D->colspi();
    double E = 0.0;
    for (int h = 0; h < nirrep; ++h) {
        for (int u = 0; u < nmopi[h]; ++u) {
            for (int v = 0; v < nmopi[h]; ++v) {
                E += D->get(h, u, v) * hpF->get(h, u, v);
            }
        }
    }
    return E;
}

void build_G(SharedWavefunction wfn, SharedMatrix C, SharedMatrix G, Options& options) {
    int nirrep = wfn->nirrep();
    Dimension nmopi = wfn->nmopi();
    Dimension noccpi = wfn->doccpi();

    if (options.get_str("INTEGRAL_METHOD") == "JK") {
        std::shared_ptr<JK> JK_occ;
        JK_occ = JK::build_JK(wfn->basisset(), wfn->get_basisset("DF_BASIS_SCF"), options);
        JK_occ->set_memory(Process::environment.get_memory() * 0.8);

        // JK_core->set_cutoff(options_.get_double("INTEGRAL_SCREENING"));
        // JK_occ->set_cutoff(options_.get_double("INTEGRAL_SCREENING"));
        JK_occ->initialize();
        JK_occ->set_do_J(true);
        // JK_core->set_allow_desymmetrization(true);
        JK_occ->set_do_K(true);

        SharedMatrix C_occ = build_Cocc_scf(C, nirrep, nmopi, noccpi);
        // Known problem: when building system G, here we should use system nmopi and system noccpi
        // !!!

        std::vector<std::shared_ptr<Matrix>>& Cl = JK_occ->C_left();
        std::vector<std::shared_ptr<Matrix>>& Cr = JK_occ->C_right();
        Cl.clear();
        Cr.clear();
        Cl.push_back(C_occ);
        Cr.push_back(C_occ);

        JK_occ->compute();

        G->copy(JK_occ->J()[0]);
        // SharedMatrix G = JK_occ->J()[0];
        SharedMatrix K = JK_occ->K()[0];

        JK_occ->finalize();

        G->scale(2.0);
        G->subtract(K); // G = (2J-K)D

        // G->transform(C);
    }
    if (options.get_str("INTEGRAL_METHOD") == "MINTS") {
        SharedMatrix TEI = AO_TEI_scf(wfn);
        SharedMatrix D(new Matrix("Density", nirrep, nmopi, nmopi));
        build_D_scf(C, noccpi, D);
        // D->print();
        G->zero();
        double tmp = 0.0;
        double jkfactor = 0.0;
        for (int h = 0; h < nirrep; ++h) { // This part doesn't work for symmetry higher than C1!!
            for (int u = 0; u < nmopi[h]; ++u) {
                for (int v = 0; v < nmopi[h]; ++v) {
                    tmp = 0.0;
                    for (int l = 0; l < nmopi[h]; ++l) {
                        for (int s = 0; s < nmopi[h]; ++s) {
                            jkfactor = 2 * eri_index_scf(TEI, u, v, l, s, nmopi) -
                                       eri_index_scf(TEI, u, l, v, s, nmopi);
                            tmp += D->get(h, l, s) * jkfactor;
                        }
                    }
                    G->set(h, u, v, tmp);
                }
            }
        }
    }
}

double do_HF(SharedWavefunction wfn, SharedMatrix hcore, int nirrep, Dimension nmopi,
             Dimension noccpi, int Maxcyc, double E_conv, Options& options) {

    // Hartree-Fock calculation with given core hamiltonian hcore

    SharedMatrix Fock(new Matrix("Fock", nirrep, nmopi, nmopi));          // AO Fock
    SharedMatrix S = wfn->S();                                            // overlap matrix
    SharedMatrix L(new Matrix("L", nirrep, nmopi, nmopi));                // S eigenvectors
    SharedVector lm(new Vector("lambda", nirrep, nmopi));                 // S eigenvalues
    SharedVector lminvhalf(new Vector("lambda inv half", nirrep, nmopi)); // S eigenvalues -1/2
    SharedMatrix LM(new Matrix("LM", nirrep, nmopi, nmopi));

    // HF calculation for environment
    outfile->Printf("\n \n Hartree-Fock SCF calculation for given h_core in embedding \n");

    Fock->copy(hcore);

    // Construct -1/2 overlap matrix
    S->diagonalize(L, lm);

    for (int h = 0; h < nirrep; ++h) {
        for (int i = 0; i < nmopi[h]; ++i) {
            double tmp = 1.0 / lm->get(h, i);
            lminvhalf->set(h, i, sqrt(tmp));
        }
    }
    LM->set_diagonal(lminvhalf);

    SharedMatrix S_half = Matrix::triplet(L, LM, L, false, false, true);

    int count = 0;
    double eps = 0.0;
    double E_scf = 0.0;
    double Etmp = 0.0;
    SharedMatrix C_star(new Matrix("C*", nirrep, nmopi, nmopi));
    SharedVector epis(new Vector("Eigens", nirrep, nmopi));
    SharedMatrix Fock_uv(new Matrix("Fock uv", nirrep, nmopi, nmopi));
    SharedMatrix C_iter(new Matrix("C_iterate", nirrep, nmopi, nmopi));
    SharedMatrix hpF(new Matrix("hcore uv + F uv", nirrep, nmopi, nmopi));
    SharedMatrix Temp(new Matrix("Temperate Mat", nirrep, nmopi, nmopi));
    SharedMatrix D_iter(new Matrix("Density Matrix", nirrep, nmopi, nmopi));
    SharedMatrix G(new Matrix("G_uv", nirrep, nmopi, nmopi));

    while (count < Maxcyc) { // SCF loop
        // rotate F
        Fock_uv->copy(Fock);

        Temp->zero();
        Temp = Matrix::triplet(S_half, Fock, S_half, true, false, false);
        Fock->copy(Temp); // F' = S-1/2T F S-1/2

        // outfile->Printf("\n ****** Checkpoint: Fock_iter after rotation iter %d ******", count);
        // Fock->print();

        C_star->zero();
        epis->zero();
        Fock->diagonalize(C_star, epis);

        if (count == 0) {
            E_scf = epis->get(0);
        }

        // rotate C
        C_iter->zero();
        C_iter = Matrix::doublet(S_half, C_star, false, false);

        // calculate energy here
        build_D_scf(C_iter, noccpi, D_iter);
        Etmp = E_scf;

        hpF->zero();
        hpF->add(hcore);
        hpF->add(Fock_uv);
        E_scf = calculate_HF_energy(hpF, D_iter);
        outfile->Printf("\n SCF iterations %d, E_scf = %8.8f", count, E_scf);
        eps = fabs(E_scf - Etmp);
        if (eps < E_conv) {
            outfile->Printf("\n SCF converged after %d iterations.", count);
            E_scf += wfn->molecule()->nuclear_repulsion_energy(wfn->get_dipole_field_strength());
            outfile->Printf("\n E_SCF_electron = %8.8f \n", E_scf);
            break;
        }

        // update Fock
        build_G(wfn, C_iter, G, options);

        Fock->zero();

        Fock->add(hcore);
        Fock->add(G);
        ++count;
    }
    // Write to wfn, modify this for open-shell in the future
    wfn->Fa()->copy(Fock_uv);
    wfn->Da()->copy(D_iter);
    // Return SCF energy
    return E_scf;
}

double OwnSCF::compute_energy() {
    // run HF calculation
    double E = do_HF(ref_wfn_, ref_wfn_->H(), nirrep_, nmopi_, doccpi_, Maxcyc_, E_conv_, options_);

    return E;
}

OwnSCF::~OwnSCF() {}
} // namespace forte
} // namespace psi
