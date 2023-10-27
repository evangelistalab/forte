/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2023 by its authors (see COPYING, COPYING.LESSER, AUTHORS).
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

#ifdef HAVE_CHEMPS2

#include "psi4/libpsi4util/process.h"
#include "psi4/libdpd/dpd.h"
#include "psi4/libiwl/iwl.hpp"
#include "psi4/libmints/typedefs.h"
#include "psi4/libtrans/integraltransform.h"
#include "psi4/psifiles.h"
// Header above this comment contains typedef std::shared_ptr<psi::Matrix>
// std::shared_ptr<psi::Matrix>;
#include "psi4/libciomr/libciomr.h"
#include "psi4/libfock/jk.h"
#include "psi4/libmints/writer_file_prefix.h"
#include "psi4/libpsio/psio.hpp"
#include "psi4/psi4-dec.h"
// Header above allows to obtain "filename.moleculename" with
// psi::get_writer_file_prefix()

#include <fstream>
#include <iostream>
#include <stdlib.h>
#include <cmath>

#include <sys/types.h>
#include <sys/stat.h>

#include "chemps2/CASSCF.h"
#include "chemps2/EdmistonRuedenberg.h"
#include "chemps2/Initialize.h"
#include "chemps2/Irreps.h"
#include "chemps2/Problem.h"

#include "ambit/blocked_tensor.h"
#include "dmrgscf.h"
#include "base_classes/mo_space_info.h"
#include "helpers/printing.h"
#include "integrals/integrals.h"
#include "base_classes/forte_options.h"

// This allows us to be lazy in getting the spaces in DPD calls
#define ID(x) ints->DPD_ID(x)

using namespace psi;

namespace forte {

DMRGSCF::DMRGSCF(StateInfo state, std::shared_ptr<SCFInfo> scf_info,
                 std::shared_ptr<ForteOptions> options, std::shared_ptr<ForteIntegrals> ints,
                 std::shared_ptr<MOSpaceInfo> mo_space_info)
    : state_(state), scf_info_(scf_info), options_(options), ints_(ints),
      mo_space_info_(mo_space_info) {
    print_method_banner({"Density Matrix Renormalization Group SCF", "Sebastian Wouters"});
    dmrg_iterations_ = options_->get_int("DMRGSCF_MAX_ITER");
}

int DMRGSCF::chemps2_groupnumber(const string SymmLabel) {

    int SyGroup = 0;
    bool stopFindGN = false;
    const int magic_number_max_groups_chemps2 = 8;
    do {
        if (SymmLabel.compare(CheMPS2::Irreps::getGroupName(SyGroup)) == 0) {
            stopFindGN = true;
        } else {
            SyGroup++;
        }
    } while ((!stopFindGN) && (SyGroup < magic_number_max_groups_chemps2));

    outfile->Printf("\n  Psi4 symmetry group was found to be <%s>.", SymmLabel.c_str());
    if (SyGroup >= magic_number_max_groups_chemps2) {
        outfile->Printf(
            "\n  CheMPS2 did not recognize this symmetry group name. CheMPS2 only knows:");
        for (int cnt = 0; cnt < magic_number_max_groups_chemps2; cnt++) {
            outfile->Printf("\n     <%s>", (CheMPS2::Irreps::getGroupName(cnt)).c_str());
        }
        throw psi::PSIEXCEPTION("CheMPS2 did not recognize the symmetry group name!");
    }
    return SyGroup;
}

void DMRGSCF::buildTmatrix(CheMPS2::DMRGSCFmatrix* theTmatrix, CheMPS2::DMRGSCFindices* iHandler,
                           std::shared_ptr<psi::PSIO> psio, std::shared_ptr<psi::Matrix> Cmat) {

    const int nirrep = ints_->nirrep();
    const int nmo = mo_space_info_->size("ALL");
    const int nTriMo = nmo * (nmo + 1) / 2;
    int* mopi = mo_space_info_->dimension("ALL");
    int* sopi = scf_info_->nsopi();
    const int nso = scf_info_->nso();
    const int nTriSo = nso * (nso + 1) / 2;
    double* work1 = new double[nTriSo];
    double* work2 = new double[nTriSo];
    IWL::read_one(psio.get(), PSIF_OEI, PSIF_SO_T, work1, nTriSo, 0, 0, "outfile");
    IWL::read_one(psio.get(), PSIF_OEI, PSIF_SO_V, work2, nTriSo, 0, 0, "outfile");
    for (int n = 0; n < nTriSo; n++) {
        work1[n] += work2[n];
    }
    delete[] work2;

    std::shared_ptr<psi::Matrix> soOei;
    soOei = std::make_shared<psi::Matrix>("SO OEI", nirrep, sopi, sopi);
    std::shared_ptr<psi::Matrix> half;
    half = std::make_shared<psi::Matrix>("Half", nirrep, mopi, sopi);
    std::shared_ptr<psi::Matrix> moOei;
    moOei = std::make_shared<psi::Matrix>("MO OEI", nirrep, mopi, mopi);

    soOei->set(work1);
    half->gemm(true, false, 1.0, Cmat, soOei, 0.0);
    moOei->gemm(false, false, 1.0, half, Cmat, 0.0);
    delete[] work1;

    copyPSIMXtoCHEMPS2MX(moOei, iHandler, theTmatrix);
}

void DMRGSCF::buildJK(std::shared_ptr<psi::Matrix> MO_RDM, std::shared_ptr<psi::Matrix> MO_JK,
                      std::shared_ptr<psi::Matrix> Cmat, std::shared_ptr<psi::JK> myJK) {

    const int nso = scf_info_->nso();
    int* nsopi = scf_info_->nsopi();
    const int nmo = mo_space_info_->size("ALL");
    int* nmopi = mo_space_info_->dimension("ALL");
    const int nirrep = ints_->nirrep();

    // nso can be different from nmo
    std::shared_ptr<psi::Matrix> SO_RDM;
    SO_RDM = std::make_shared<psi::Matrix>("SO RDM", nirrep, nsopi, nsopi);
    std::shared_ptr<psi::Matrix> Identity;
    Identity = std::make_shared<psi::Matrix>("Identity", nirrep, nsopi, nsopi);
    std::shared_ptr<psi::Matrix> SO_JK;
    SO_JK = std::make_shared<psi::Matrix>("SO JK", nirrep, nsopi, nsopi);
    std::shared_ptr<psi::Matrix> work;
    work = std::make_shared<psi::Matrix>("work", nirrep, nsopi, nmopi);

    work->gemm(false, false, 1.0, Cmat, MO_RDM, 0.0);
    SO_RDM->gemm(false, true, 1.0, work, Cmat, 0.0);

    std::vector<std::shared_ptr<psi::Matrix>>& CL = myJK->C_left();
    CL.clear();
    CL.push_back(SO_RDM);

    std::vector<std::shared_ptr<psi::Matrix>>& CR = myJK->C_right();
    CR.clear();
    Identity->identity();
    CR.push_back(Identity);

    myJK->set_do_J(true);
    myJK->set_do_K(true);
    myJK->set_do_wK(false);
    myJK->compute();

    SO_JK->copy(myJK->K()[0]);
    SO_JK->scale(-0.5);
    SO_JK->add(myJK->J()[0]);

    work->gemm(false, false, 1.0, SO_JK, Cmat, 0.0);
    MO_JK->gemm(true, false, 1.0, Cmat, work, 0.0);
}

void DMRGSCF::copyPSIMXtoCHEMPS2MX(std::shared_ptr<psi::Matrix> source,
                                   CheMPS2::DMRGSCFindices* iHandler,
                                   CheMPS2::DMRGSCFmatrix* target) {

    for (int irrep = 0; irrep < iHandler->getNirreps(); irrep++) {
        for (int orb1 = 0; orb1 < iHandler->getNORB(irrep); orb1++) {
            for (int orb2 = 0; orb2 < iHandler->getNORB(irrep); orb2++) {
                target->set(irrep, orb1, orb2, source->get(irrep, orb1, orb2));
            }
        }
    }
}

void DMRGSCF::copyCHEMPS2MXtoPSIMX(CheMPS2::DMRGSCFmatrix* source,
                                   CheMPS2::DMRGSCFindices* iHandler,
                                   std::shared_ptr<psi::Matrix> target) {

    for (int irrep = 0; irrep < iHandler->getNirreps(); irrep++) {
        for (int orb1 = 0; orb1 < iHandler->getNORB(irrep); orb1++) {
            for (int orb2 = 0; orb2 < iHandler->getNORB(irrep); orb2++) {
                target->set(irrep, orb1, orb2, source->get(irrep, orb1, orb2));
            }
        }
    }
}

void DMRGSCF::buildQmatOCC(CheMPS2::DMRGSCFmatrix* theQmatOCC, CheMPS2::DMRGSCFindices* iHandler,
                           std::shared_ptr<psi::Matrix> MO_RDM, std::shared_ptr<psi::Matrix> MO_JK,
                           std::shared_ptr<psi::Matrix> Cmat, std::shared_ptr<psi::JK> myJK) {

    MO_RDM->zero();
    for (int irrep = 0; irrep < iHandler->getNirreps(); irrep++) {
        for (int orb = 0; orb < iHandler->getNOCC(irrep); orb++) {
            MO_RDM->set(irrep, orb, orb, 2.0);
        }
    }
    buildJK(MO_RDM, MO_JK, Cmat, myJK);
    copyPSIMXtoCHEMPS2MX(MO_JK, iHandler, theQmatOCC);
}

void DMRGSCF::buildQmatACT(CheMPS2::DMRGSCFmatrix* theQmatACT, CheMPS2::DMRGSCFindices* iHandler,
                           double* DMRG1DM, std::shared_ptr<psi::Matrix> MO_RDM,
                           std::shared_ptr<psi::Matrix> MO_JK, std::shared_ptr<psi::Matrix> Cmat,
                           std::shared_ptr<psi::JK> myJK) {

    MO_RDM->zero();
    const int nOrbDMRG = iHandler->getDMRGcumulative(iHandler->getNirreps());
    for (int irrep = 0; irrep < iHandler->getNirreps(); irrep++) {
        const int NOCC = iHandler->getNOCC(irrep);
        const int shift = iHandler->getDMRGcumulative(irrep);
        for (int orb1 = 0; orb1 < iHandler->getNDMRG(irrep); orb1++) {
            for (int orb2 = orb1; orb2 < iHandler->getNDMRG(irrep); orb2++) {
                const double value = DMRG1DM[shift + orb1 + nOrbDMRG * (shift + orb2)];
                MO_RDM->set(irrep, NOCC + orb1, NOCC + orb2, value);
                MO_RDM->set(irrep, NOCC + orb2, NOCC + orb1, value);
            }
        }
    }
    buildJK(MO_RDM, MO_JK, Cmat, myJK);
    copyPSIMXtoCHEMPS2MX(MO_JK, iHandler, theQmatACT);
}

void DMRGSCF::buildHamDMRG(std::shared_ptr<psi::IntegralTransform> ints,
                           std::shared_ptr<psi::MOSpace> Aorbs_ptr,
                           CheMPS2::DMRGSCFmatrix* theTmatrix, CheMPS2::DMRGSCFmatrix* theQmatOCC,
                           CheMPS2::DMRGSCFindices* iHandler, CheMPS2::Hamiltonian* HamDMRG,
                           std::shared_ptr<psi::PSIO> psio) {

    ints->update_orbitals();
    // Since we don't regenerate the SO ints, we don't call sort_so_tei, and the
    // OEI are not updated !!!!!
    ints->transform_tei(Aorbs_ptr, Aorbs_ptr, Aorbs_ptr, Aorbs_ptr);
    dpd_set_default(ints->get_dpd_id());
    const int nirrep = ints_->nirrep();

    // Econstant and one-electron integrals
    {
        double Econstant = ints_->nuclear_repulsion_energy();
        for (int h = 0; h < iHandler->getNirreps(); h++) {
            const int NOCC = iHandler->getNOCC(h);
            for (int froz = 0; froz < NOCC; froz++) {
                Econstant += 2 * theTmatrix->get(h, froz, froz) + theQmatOCC->get(h, froz, froz);
            }
            const int shift = iHandler->getDMRGcumulative(h);
            for (int orb1 = 0; orb1 < iHandler->getNDMRG(h); orb1++) {
                for (int orb2 = orb1; orb2 < iHandler->getNDMRG(h); orb2++) {
                    HamDMRG->setTmat(shift + orb1, shift + orb2,
                                     theTmatrix->get(h, NOCC + orb1, NOCC + orb2) +
                                         theQmatOCC->get(h, NOCC + orb1, NOCC + orb2));
                }
            }
        }
        HamDMRG->setEconst(Econstant);
    }

    // Two-electron integrals
    psi::dpdbuf4 K;
    psio->open(PSIF_LIBTRANS_DPD, PSIO_OPEN_OLD);
    psi::global_dpd_->buf4_init(&K, PSIF_LIBTRANS_DPD, 0, ID("[S,S]"), ID("[S,S]"), ID("[S>=S]+"),
                                ID("[S>=S]+"), 0, "MO Ints (SS|SS)");
    for (int h = 0; h < nirrep; ++h) {
        psi::global_dpd_->buf4_mat_irrep_init(&K, h);
        psi::global_dpd_->buf4_mat_irrep_rd(&K, h);
        for (int pq = 0; pq < K.params->rowtot[h]; ++pq) {
            const int p = K.params->roworb[h][pq][0];
            const int q = K.params->roworb[h][pq][1];
            for (int rs = 0; rs < K.params->coltot[h]; ++rs) {
                const int r = K.params->colorb[h][rs][0];
                const int s = K.params->colorb[h][rs][1];
                HamDMRG->setVmat(p, r, q, s, K.matrix[h][pq][rs]);
            }
        }
        psi::global_dpd_->buf4_mat_irrep_close(&K, h);
    }
    psi::global_dpd_->buf4_close(&K);
    psio->close(PSIF_LIBTRANS_DPD, PSIO_OPEN_OLD);
}
void DMRGSCF::buildHamDMRGForte(CheMPS2::DMRGSCFmatrix* theQmatOCC,
                                CheMPS2::DMRGSCFindices* iHandler, CheMPS2::Hamiltonian* HamDMRG,
                                std::shared_ptr<ForteIntegrals> ints) {
    /// Retransform all the integrals for now (TODO:  CASSCF-like integral
    /// transformation)
    size_t na = mo_space_info_->size("ACTIVE");
    std::vector<size_t> active_orbs = mo_space_info_->corr_absolute_mo("ACTIVE");
    ambit::Tensor VMat = ints_->aptei_ab_block(active_orbs, active_orbs, active_orbs, active_orbs);
    for (size_t u = 0; u < na; u++) {
        for (size_t v = 0; v < na; v++) {
            for (size_t x = 0; x < na; x++) {
                for (size_t y = 0; y < na; y++) {
                    size_t offset = u * na * na * na + v * na * na + x * na + y;
                    HamDMRG->setVmat(u, x, v, y, VMat.data()[offset]);
                }
            }
        }
    }
}

void DMRGSCF::fillRotatedTEI_coulomb(std::shared_ptr<psi::IntegralTransform> ints,
                                     std::shared_ptr<psi::MOSpace> OAorbs_ptr,
                                     CheMPS2::DMRGSCFmatrix* theTmatrix,
                                     CheMPS2::DMRGSCFintegrals* theRotatedTEI,
                                     CheMPS2::DMRGSCFindices* iHandler,
                                     std::shared_ptr<psi::PSIO> psio) {

    ints->update_orbitals();
    // Since we don't regenerate the SO ints, we don't call sort_so_tei, and the
    // OEI are not updated !!!!!
    ints->transform_tei(OAorbs_ptr, OAorbs_ptr, psi::MOSpace::all, psi::MOSpace::all);
    dpd_set_default(ints->get_dpd_id());
    const int nirrep = ints_->nirrep();

    // One-electron integrals
    {
        const int nmo = mo_space_info_->size("ALL");
        const int nTriMo = nmo * (nmo + 1) / 2;
        const int nso = scf_info_->nso();
        const int nTriSo = nso * (nso + 1) / 2;
        int* mopi = mo_space_info_->dimension("ALL");
        int* sopi = scf_info_->nsopi();
        double* work1 = new double[nTriSo];
        double* work2 = new double[nTriSo];
        IWL::read_one(psio.get(), PSIF_OEI, PSIF_SO_T, work1, nTriSo, 0, 0, "outfile");
        IWL::read_one(psio.get(), PSIF_OEI, PSIF_SO_V, work2, nTriSo, 0, 0, "outfile");
        for (int n = 0; n < nTriSo; n++) {
            work1[n] += work2[n];
        }
        delete[] work2;

        std::shared_ptr<psi::Matrix> soOei;
        soOei = std::make_shared<psi::Matrix>("SO OEI", nirrep, sopi, sopi);
        std::shared_ptr<psi::Matrix> half;
        half = std::make_shared<psi::Matrix>("Half", nirrep, mopi, sopi);
        std::shared_ptr<psi::Matrix> moOei;
        moOei = std::make_shared<psi::Matrix>("MO OEI", nirrep, mopi, mopi);

        soOei->set(work1);
        half->gemm(true, false, 1.0, ints_->Ca(), soOei, 0.0);
        moOei->gemm(false, false, 1.0, half, ints_->Ca(), 0.0);
        delete[] work1;

        copyPSIMXtoCHEMPS2MX(moOei, iHandler, theTmatrix);
    }

    // Two-electron integrals
    dpdbuf4 K;
    psio->open(PSIF_LIBTRANS_DPD, PSIO_OPEN_OLD);
    // To only process the permutationally unique integrals, change the
    // ID("[A,A]") to ID("[A>=A]+")
    // psi::global_dpd_->buf4_init(&K, PSIF_LIBTRANS_DPD, 0, ID("[A,A]"),
    // ID("[A,A]"), ID("[A>=A]+"), ID("[A>=A]+"), 0, "MO Ints (AA|AA)");
    // int buf4_init(dpdbuf4 *Buf, int inputfile, int irrep, int pqnum, int
    // rsnum, int file_pqnum, int file_rsnum, int anti, const char *label);
    psi::global_dpd_->buf4_init(&K, PSIF_LIBTRANS_DPD, 0, ID("[Q,Q]"), ID("[A,A]"), ID("[Q>=Q]+"),
                                ID("[A>=A]+"), 0, "MO Ints (QQ|AA)");
    for (int h = 0; h < nirrep; ++h) {
        psi::global_dpd_->buf4_mat_irrep_init(&K, h);
        psi::global_dpd_->buf4_mat_irrep_rd(&K, h);
        for (int pq = 0; pq < K.params->rowtot[h]; ++pq) {
            const int p = K.params->roworb[h][pq][0];
            const int q = K.params->roworb[h][pq][1];
            const int psym = K.params->psym[p];
            const int qsym = K.params->qsym[q];
            const int prel = p - K.params->poff[psym];
            const int qrel = q - K.params->qoff[qsym];
            for (int rs = 0; rs < K.params->coltot[h]; ++rs) {
                const int r = K.params->colorb[h][rs][0];
                const int s = K.params->colorb[h][rs][1];
                const int rsym = K.params->rsym[r];
                const int ssym = K.params->ssym[s];
                const int rrel = r - K.params->roff[rsym];
                const int srel = s - K.params->soff[ssym];
                theRotatedTEI->set_coulomb(psym, qsym, rsym, ssym, prel, qrel, rrel, srel,
                                           K.matrix[h][pq][rs]);
            }
        }
        psi::global_dpd_->buf4_mat_irrep_close(&K, h);
    }
    psi::global_dpd_->buf4_close(&K);
    psio->close(PSIF_LIBTRANS_DPD, PSIO_OPEN_OLD);
}

void DMRGSCF::fillRotatedTEI_exchange(std::shared_ptr<psi::IntegralTransform> ints,
                                      std::shared_ptr<psi::MOSpace> OAorbs_ptr,
                                      std::shared_ptr<psi::MOSpace> Vorbs_ptr,
                                      CheMPS2::DMRGSCFintegrals* theRotatedTEI,
                                      CheMPS2::DMRGSCFindices* iHandler,
                                      std::shared_ptr<psi::PSIO> psio) {

    ints->update_orbitals();
    ints->transform_tei(Vorbs_ptr, OAorbs_ptr, Vorbs_ptr, OAorbs_ptr);
    dpd_set_default(ints->get_dpd_id());

    // Two-electron integrals
    dpdbuf4 K;
    psio->open(PSIF_LIBTRANS_DPD, PSIO_OPEN_OLD);
    // To only process the permutationally unique integrals, change the
    // ID("[A,A]") to ID("[A>=A]+")
    // psi::global_dpd_->buf4_init(&K, PSIF_LIBTRANS_DPD, 0, ID("[A,A]"),
    // ID("[A,A]"), ID("[A>=A]+"), ID("[A>=A]+"), 0, "MO Ints (AA|AA)");
    // int buf4_init(dpdbuf4 *Buf, int inputfile, int irrep, int pqnum, int
    // rsnum, int file_pqnum, int file_rsnum, int anti, const char *label);
    psi::global_dpd_->buf4_init(&K, PSIF_LIBTRANS_DPD, 0, ID("[T,Q]"), ID("[T,Q]"), ID("[T,Q]"),
                                ID("[T,Q]"), 0, "MO Ints (TQ|TQ)");
    for (int h = 0; h < iHandler->getNirreps(); ++h) {
        psi::global_dpd_->buf4_mat_irrep_init(&K, h);
        psi::global_dpd_->buf4_mat_irrep_rd(&K, h);
        for (int pq = 0; pq < K.params->rowtot[h]; ++pq) {
            const int p = K.params->roworb[h][pq][0];
            const int q = K.params->roworb[h][pq][1];
            const int psym = K.params->psym[p];
            const int qsym = K.params->qsym[q];
            const int prel =
                p - K.params->poff[psym] + iHandler->getNOCC(psym) + iHandler->getNDMRG(psym);
            const int qrel = q - K.params->qoff[qsym];
            for (int rs = 0; rs < K.params->coltot[h]; ++rs) {
                const int r = K.params->colorb[h][rs][0];
                const int s = K.params->colorb[h][rs][1];
                const int rsym = K.params->rsym[r];
                const int ssym = K.params->ssym[s];
                const int rrel =
                    r - K.params->roff[rsym] + iHandler->getNOCC(rsym) + iHandler->getNDMRG(rsym);
                const int srel = s - K.params->soff[ssym];
                theRotatedTEI->set_exchange(qsym, ssym, psym, rsym, qrel, srel, prel, rrel,
                                            K.matrix[h][pq][rs]);
            }
        }
        psi::global_dpd_->buf4_mat_irrep_close(&K, h);
    }
    psi::global_dpd_->buf4_close(&K);
    psio->close(PSIF_LIBTRANS_DPD, PSIO_OPEN_OLD);
}

void DMRGSCF::copyUNITARYtoPSIMX(CheMPS2::DMRGSCFunitary* unitary,
                                 CheMPS2::DMRGSCFindices* iHandler,
                                 std::shared_ptr<psi::Matrix> target) {

    for (int irrep = 0; irrep < iHandler->getNirreps(); irrep++) {
        for (int orb1 = 0; orb1 < iHandler->getNORB(irrep); orb1++) {
            for (int orb2 = 0; orb2 < iHandler->getNORB(irrep); orb2++) {
                target->set(irrep, orb1, orb2,
                            unitary->getBlock(irrep)[orb1 + iHandler->getNORB(irrep) * orb2]);
            }
        }
    }
}

void DMRGSCF::update_WFNco(std::shared_ptr<psi::Matrix> Coeff_orig,
                           CheMPS2::DMRGSCFindices* iHandler, CheMPS2::DMRGSCFunitary* unitary,
                           std::shared_ptr<psi::Matrix> work1, std::shared_ptr<psi::Matrix> work2) {

    // copyCHEMPS2MXtoPSIMX( Coeff_orig, iHandler, work1 );
    copyUNITARYtoPSIMX(unitary, iHandler, work2);
    ints_->Ca()->gemm(false, true, 1.0, Coeff_orig, work2, 0.0);
    ints_->Cb()->copy(ints_->Ca());
}

double DMRGSCF::compute_energy() {
    /* This plugin is able to perform a DMRGSCF calculation in a molecular
     * orbital active space. */

    /*******************************
     *   Environment information   *
     *******************************/
    std::shared_ptr<psi::PSIO> psio(_default_psio_lib_); // Grab the global (default)
                                                         // psi::PSIO object, for file I/O

    /*************************
     *   Fetch the options   *
     *************************/

    const int wfn_irrep = options_->get_int("DMRG_WFN_IRREP");
    const int wfn_multp = options_->get_int("MULTIPLICITY");
    int* dmrg_states = options_->get_int_list("DMRG_STATES").data();
    const int ndmrg_states = options_->get_int_list("DMRG_STATES").size();
    double* dmrg_econv = options_->get_double_list("DMRG_ECONV").data();
    const int ndmrg_econv = options_->get_double_list("DMRG_ECONV").size();
    int* dmrg_maxsweeps = options_->get_int_list("DMRG_MAXSWEEPS").data();
    const int ndmrg_maxsweeps = options_->get_int_list("DMRG_MAXSWEEPS").size();
    double* dmrg_noiseprefactors = options_->get_double_list("DMRG_NOISEPREFACTORS").data();
    double* dmrg_davidson_tol = options_->get_double_list("DMRG_DAVIDSON_RTOL").data();
    const int ndmrg_davidson_tol = options_->get_double_list("DMRG_DAVIDSON_RTOL").size();
    const int ndmrg_noiseprefactors = options_->get_double_list("DMRG_NOISEPREFACTORS").size();
    const bool dmrg_print_corr = options_->get_bool("DMRG_PRINT_CORR");
    const bool mps_chkpt = options_->get_bool("DMRG_CHKPT");
    // int * frozen_docc                 =
    // options_->get_int_array("FROZEN_DOCC");
    // int * active                      = options_->get_int_array("ACTIVE");
    /// Sebastian optimizes the frozen_docc
    int* frozen_docc = options_->get_int_list("DMRG_FROZEN_DOCC").data();
    psi::Dimension active = mo_space_info_->dimension("ACTIVE");
    const double dmrgscf_convergence = options_->get_double("D_CONVERGENCE");
    const bool dmrgscf_store_unit = options_->get_bool("DMRG_STORE_UNIT");
    const bool dmrgscf_do_diis = options_->get_bool("DMRG_DO_DIIS");
    const double dmrgscf_diis_branch = options_->get_double("DMRG_DIIS_BRANCH");
    const bool dmrgscf_store_diis = options_->get_bool("DMRG_STORE_DIIS");
    const int dmrgscf_which_root = options_->get_int("DMRG_WHICH_ROOT");
    const bool dmrgscf_state_avg = options_->get_bool("DMRG_AVG_STATES");
    const string dmrgscf_active_space = options_->get_str("DMRG_ACTIVE_SPACE");
    const bool dmrgscf_loc_random = options_->get_bool("DMRG_LOC_RANDOM");
    const int dmrgscf_num_vec_diis = CheMPS2::DMRGSCF_numDIISvecs;
    const std::string unitaryname =
        psi::get_writer_file_prefix(ints_->wfn()->name()) + ".unitary.h5";
    const std::string diisname = psi::get_writer_file_prefix(ints_->wfn()->name()) + ".DIIS.h5";
    bool three_pdm = false;
    if (options_->get_str("JOB_TYPE") == "DSRG-MRPT2" or
        options_->get_str("JOB_TYPE") == "THREE-DSRG-MRPT2") {
        if (options_->get_str("THREEPDC") != "ZERO")
            three_pdm = true;
    }

    /****************************************
     *   Check if the input is consistent   *
     ****************************************/
    // const int SyGroup = chemps2_groupnumber(state_.irrep());
    const int SyGroup = state_.irrep();
    const int nmo = mo_space_info_->size("ALL");
    const int nirrep = ints_->nirrep();
    int* orbspi = mo_space_info_->dimension("ALL");
    int* docc = scf_info_->doccpi();
    int* socc = scf_info_->soccpi();
    if (wfn_irrep < 0) {
        throw psi::PSIEXCEPTION("Option WFN_IRREP (integer) may not be smaller than zero!");
    }
    if (wfn_multp < 1) {
        throw psi::PSIEXCEPTION("Option WFN_MULTP (integer) should be larger or "
                                "equal to one: WFN_MULTP = (2S+1) >= 1 !");
    }
    if (ndmrg_states == 0) {
        throw psi::PSIEXCEPTION("Option DMRG_STATES (integer array) should be set!");
    }
    if (ndmrg_econv == 0) {
        throw psi::PSIEXCEPTION("Option DMRG_ECONV (double array) should be set!");
    }
    if (ndmrg_maxsweeps == 0) {
        throw psi::PSIEXCEPTION("Option DMRG_MAXSWEEPS (integer array) should be set!");
    }
    if (ndmrg_noiseprefactors == 0) {
        throw psi::PSIEXCEPTION("Option DMRG_NOISEPREFACTORS (double array) should be set!");
    }
    if (ndmrg_states != ndmrg_econv) {
        throw psi::PSIEXCEPTION("Options DMRG_STATES (integer array) and DMRG_ECONV "
                                "(double array) should contain the same number of "
                                "elements!");
    }
    if (ndmrg_states != ndmrg_maxsweeps) {
        throw psi::PSIEXCEPTION("Options DMRG_STATES (integer array) and "
                                "DMRG_MAXSWEEPS (integer array) should contain the "
                                "same number of elements!");
    }
    if (ndmrg_states != ndmrg_noiseprefactors) {
        throw psi::PSIEXCEPTION("Options DMRG_STATES (integer array) and "
                                "DMRG_NOISEPREFACTORS (double array) should contain "
                                "the same number of elements!");
    }
    for (int cnt = 0; cnt < ndmrg_states; cnt++) {
        if (dmrg_states[cnt] < 2) {
            throw psi::PSIEXCEPTION("Entries in DMRG_STATES (integer array) should "
                                    "be larger than 1!");
        }
    }
    if (dmrgscf_convergence <= 0.0) {
        throw psi::PSIEXCEPTION("Option D_CONVERGENCE (double) must be larger than zero!");
    }
    if (dmrgscf_diis_branch <= 0.0) {
        throw psi::PSIEXCEPTION("Option DMRG_DIIS_BRANCH (double) must be larger than zero!");
    }
    if (dmrg_iterations_ < 1) {
        throw psi::PSIEXCEPTION("Option DMRG_MAX_ITER (integer) must be larger than zero!");
    }
    if (dmrgscf_which_root < 1) {
        throw psi::PSIEXCEPTION("Option DMRG_WHICH_ROOT (integer) must be larger than zero!");
    }

    /*******************************************
     *   Create a CheMPS2::ConvergenceScheme   *
     *******************************************/

    CheMPS2::Initialize::Init();
    CheMPS2::ConvergenceScheme* OptScheme = new CheMPS2::ConvergenceScheme(ndmrg_states);
    for (int cnt = 0; cnt < ndmrg_states; cnt++) {
        if (ndmrg_davidson_tol != ndmrg_states)
            OptScheme->setInstruction(cnt, dmrg_states[cnt], dmrg_econv[cnt], dmrg_maxsweeps[cnt],
                                      dmrg_noiseprefactors[cnt]);
        else {
            OptScheme->set_instruction(cnt, dmrg_states[cnt], dmrg_econv[cnt], dmrg_maxsweeps[cnt],
                                       dmrg_noiseprefactors[cnt], dmrg_davidson_tol[cnt]);
        }
    }

    /******************************************************************************
     *   Print orbital information; check consistency of frozen_docc and active
     **
     ******************************************************************************/

    int* nvirtual = new int[nirrep];
    bool virtualsOK = true;
    for (int cnt = 0; cnt < nirrep; cnt++) {
        nvirtual[cnt] = orbspi[cnt] - frozen_docc[cnt] - active[cnt];
        if (nvirtual[cnt] < 0) {
            virtualsOK = false;
        }
    }
    outfile->Printf("\n  wfn_irrep   = %d", wfn_irrep);
    outfile->Printf("\n  wfn_multp   = %d", wfn_multp);
    outfile->Printf("\n  numOrbitals = [%d", orbspi[0]);
    for (int cnt = 1; cnt < nirrep; cnt++) {
        outfile->Printf(", %d", orbspi[cnt]);
    }
    outfile->Printf("\n  R(O)HF DOCC = [%d", docc[0]);
    for (int cnt = 1; cnt < nirrep; cnt++) {
        outfile->Printf(", %d", docc[cnt]);
    }
    outfile->Printf("]");
    outfile->Printf("\n  R(O)HF SOCC = [%d", socc[0]);
    for (int cnt = 1; cnt < nirrep; cnt++) {
        outfile->Printf(", %d", socc[cnt]);
    }
    outfile->Printf("]");
    outfile->Printf("\n  frozen_docc = [%d", frozen_docc[0]);
    for (int cnt = 1; cnt < nirrep; cnt++) {
        outfile->Printf(", %d", frozen_docc[cnt]);
    }
    outfile->Printf("]");
    outfile->Printf("\n  active      = [%d", active[0]);
    for (int cnt = 1; cnt < nirrep; cnt++) {
        outfile->Printf(", %d", active[cnt]);
    }
    outfile->Printf("]");
    outfile->Printf("\n  virtual     = [%d", nvirtual[0]);
    for (int cnt = 1; cnt < nirrep; cnt++) {
        outfile->Printf(", %d", nvirtual[cnt]);
    }
    outfile->Printf("]");
    if (!virtualsOK) {
        throw psi::PSIEXCEPTION("For at least one irrep: frozen_docc[ irrep ] + "
                                "active[ irrep ] > numOrbitals[ irrep ]!");
    }
    outfile->Printf("\n  DMRGSCF computation run with %d iterations", dmrg_iterations_);

    /**********************************************
     *   Create another bit of DMRGSCF preamble   *
     **********************************************/
    CheMPS2::DMRGSCFindices* iHandler =
        new CheMPS2::DMRGSCFindices(nmo, SyGroup, frozen_docc, active, nvirtual);
    CheMPS2::DMRGSCFunitary* unitary = new CheMPS2::DMRGSCFunitary(iHandler);
    CheMPS2::DIIS* theDIIS = NULL;
    CheMPS2::DMRGSCFintegrals* theRotatedTEI = new CheMPS2::DMRGSCFintegrals(iHandler);
    size_t nOrbDMRG = mo_space_info_->size("ACTIVE");
    double* DMRG1DM = new double[nOrbDMRG * nOrbDMRG];
    double* DMRG2DM = new double[nOrbDMRG * nOrbDMRG * nOrbDMRG * nOrbDMRG];
    double* DMRG3DM;
    if (three_pdm) {
        DMRG3DM = new double[nOrbDMRG * nOrbDMRG * nOrbDMRG * nOrbDMRG * nOrbDMRG * nOrbDMRG];
    }
    CheMPS2::DMRGSCFmatrix* theFmatrix = new CheMPS2::DMRGSCFmatrix(iHandler);
    theFmatrix->clear();
    CheMPS2::DMRGSCFmatrix* theQmatOCC = new CheMPS2::DMRGSCFmatrix(iHandler);
    theQmatOCC->clear();
    CheMPS2::DMRGSCFmatrix* theQmatACT = new CheMPS2::DMRGSCFmatrix(iHandler);
    theQmatACT->clear();
    CheMPS2::DMRGSCFmatrix* theTmatrix = new CheMPS2::DMRGSCFmatrix(iHandler);
    theTmatrix->clear();
    CheMPS2::DMRGSCFwtilde* wmattilde = new CheMPS2::DMRGSCFwtilde(iHandler);
    delete[] nvirtual;

    /***************************************************
     *   Create the active space Hamiltonian storage   *
     ***************************************************/

    int nElectrons = 0;
    for (int cnt = 0; cnt < nirrep; cnt++) {
        nElectrons += 2 * docc[cnt] + socc[cnt];
    }
    outfile->Printf("\n  nElectrons  = %d", nElectrons);

    // Number of electrons in the active space
    int nDMRGelectrons = nElectrons;
    for (int cnt = 0; cnt < nirrep; cnt++) {
        nDMRGelectrons -= 2 * frozen_docc[cnt];
    }
    outfile->Printf("\n  nEl. active = %d", nDMRGelectrons);

    // Create the CheMPS2::Hamiltonian --> fill later
    int* orbitalIrreps = new int[nOrbDMRG];
    int counterFillOrbitalIrreps = 0;
    for (int h = 0; h < nirrep; h++) {
        for (int cnt = 0; cnt < active[h];
             cnt++) { // Only the active space is treated with DMRG-SCF!
            orbitalIrreps[counterFillOrbitalIrreps] = h;
            counterFillOrbitalIrreps++;
        }
    }
    CheMPS2::Hamiltonian* HamDMRG = new CheMPS2::Hamiltonian(nOrbDMRG, SyGroup, orbitalIrreps);
    delete[] orbitalIrreps;

    /* Create the CheMPS2::Problem
       You can fill Ham later, as Problem only keeps a pointer to the
       Hamiltonian object.
       Since only doubly occupied frozen orbitals are allowed, wfn_multp and
       wfn_irrep do not change. */
    CheMPS2::Problem* Prob =
        new CheMPS2::Problem(HamDMRG, wfn_multp - 1, nDMRGelectrons, wfn_irrep);
    if (!(Prob->checkConsistency())) {
        throw psi::PSIEXCEPTION("CheMPS2::Problem : No Hilbert state vector "
                                "compatible with all symmetry sectors!");
    }
    Prob->SetupReorderD2h(); // Does nothing if group not d2h

    /**************************************
     *   Input is parsed and consistent   *
     *   Start with DMRGSCF               *
     **************************************/

    std::shared_ptr<psi::Matrix> work1;
    work1 = std::make_shared<psi::Matrix>("work1", nirrep, orbspi, orbspi);
    std::shared_ptr<psi::Matrix> work2;
    work2 = std::make_shared<psi::Matrix>("work2", nirrep, orbspi, orbspi);
    std::shared_ptr<psi::JK> myJK =
        std::shared_ptr<psi::JK>(new DiskJK(ints_->basisset(), ints_->wfn()->options()));

    myJK->set_cutoff(0.0);
    myJK->initialize();
    auto Coeff_orig = std::make_shared<psi::Matrix>(ints_->Ca());
    // copyPSIMXtoCHEMPS2MX(this->Ca(), iHandler, );

    std::vector<int> OAorbs; // Occupied + active
    std::vector<int> Aorbs;  // Only active
    std::vector<int> Vorbs;  // Virtual
    std::vector<int> empty;
    for (int h = 0; h < iHandler->getNirreps(); h++) {
        for (int orb = 0; orb < iHandler->getNOCC(h) + iHandler->getNDMRG(h); orb++) {
            OAorbs.push_back(iHandler->getOrigNOCCstart(h) + orb);
        }
        for (int orb = 0; orb < iHandler->getNDMRG(h); orb++) {
            Aorbs.push_back(iHandler->getOrigNDMRGstart(h) + orb);
        }
        for (int orb = 0; orb < iHandler->getNVIRT(h); orb++) {
            Vorbs.push_back(iHandler->getOrigNVIRTstart(h) + orb);
        }
    }
    std::shared_ptr<psi::MOSpace> OAorbs_ptr;
    OAorbs_ptr = std::shared_ptr<psi::MOSpace>(new psi::MOSpace('Q', OAorbs, empty));
    std::shared_ptr<psi::MOSpace> Aorbs_ptr;
    Aorbs_ptr = std::shared_ptr<psi::MOSpace>(new psi::MOSpace('S', Aorbs, empty));
    std::shared_ptr<psi::MOSpace> Vorbs_ptr;
    Vorbs_ptr = std::shared_ptr<psi::MOSpace>(new psi::MOSpace('T', Vorbs, empty));
    std::vector<std::shared_ptr<psi::MOSpace>> spaces;
    spaces.push_back(OAorbs_ptr);
    spaces.push_back(Aorbs_ptr);
    spaces.push_back(Vorbs_ptr);
    spaces.push_back(psi::MOSpace::all);
    // CheMPS2 requires RHF or ROHF orbitals.
    std::shared_ptr<IntegralTransform> ints;
    ints = std::shared_ptr<IntegralTransform>(new IntegralTransform(
        ints_->wfn(), spaces, psi::IntegralTransform::TransformationType::Restricted));
    ints->set_keep_iwl_so_ints(true);
    ints->set_keep_dpd_so_ints(true);
    // ints->set_print(6);

    outfile->Printf("\n  ###########################################################");
    outfile->Printf("\n  ###                                                     ###");
    outfile->Printf("\n  ###                       DMRG-SCF                      ###");
    outfile->Printf("\n  ###                                                     ###");
    outfile->Printf("\n  ###            CheMPS2 by Sebastian Wouters             ###");
    outfile->Printf("\n  ###        https://github.com/SebWouters/CheMPS2        ###");
    outfile->Printf("\n  ###   Comput. Phys. Commun. 185 (6), 1501-1514 (2014)   ###");
    outfile->Printf("\n  ###                                                     ###");
    outfile->Printf("\n  ###########################################################");
    outfile->Printf("\n  Number of variables in the x-matrix = %d", unitary->getNumVariablesX());

    // Convergence variables
    double gradNorm = 1.0;
    double updateNorm = 1.0;
    double* theupdate = new double[unitary->getNumVariablesX()];
    for (int cnt = 0; cnt < unitary->getNumVariablesX(); cnt++) {
        theupdate[cnt] = 0.0;
    }
    double* theDIISparameterVector = NULL;
    double Energy = 1e8;

    int theDIISvectorParamSize = 0;
    int maxlinsize = 0;
    for (int irrep = 0; irrep < nirrep; irrep++) {
        const int linsize_irrep = iHandler->getNORB(irrep);
        theDIISvectorParamSize += linsize_irrep * (linsize_irrep - 1) / 2;
        if (linsize_irrep > maxlinsize) {
            maxlinsize = linsize_irrep;
        }
    }

    const int nOrbDMRG_pow4 = nOrbDMRG * nOrbDMRG * nOrbDMRG * nOrbDMRG;
    const int unitary_worksize = 4 * maxlinsize * maxlinsize;
    const int sizeWorkMem = (nOrbDMRG_pow4 > unitary_worksize) ? nOrbDMRG_pow4 : unitary_worksize;
    double* mem1 = new double[sizeWorkMem];
    double* mem2 = new double[sizeWorkMem];

    CheMPS2::EdmistonRuedenberg* theLocalizer = NULL;
    if (dmrgscf_active_space.compare("LOC") == 0) {
        theLocalizer =
            new CheMPS2::EdmistonRuedenberg(HamDMRG->getVmat(), iHandler->getGroupNumber());
    }

    // Load unitary from disk
    if (dmrgscf_store_unit) {
        struct stat stFileInfo;
        int intStat = stat(unitaryname.c_str(), &stFileInfo);
        if (intStat == 0) {
            unitary->loadU(unitaryname);
        }
    }

    // Load DIIS from disk
    if ((dmrgscf_do_diis) && (dmrgscf_store_diis)) {
        struct stat stFileInfo;
        int intStat = stat(diisname.c_str(), &stFileInfo);
        if (intStat == 0) {
            if (theDIIS == NULL) {
                theDIIS = new CheMPS2::DIIS(theDIISvectorParamSize, unitary->getNumVariablesX(),
                                            dmrgscf_num_vec_diis);
                theDIISparameterVector = new double[theDIISvectorParamSize];
            }
            theDIIS->loadDIIS(diisname);
        }
    }

    int nIterations = 0;

    /*   Actual DMRGSCF loops   */
    while ((gradNorm > dmrgscf_convergence) && (nIterations < dmrg_iterations_)) {

        nIterations++;

        // Update the unitary transformation
        if (unitary->getNumVariablesX() > 0) {

            std::ofstream capturing;
            std::streambuf* cout_buffer;
            std::string chemps2filename = "output.chemps2";
            outfile->Printf(
                "\n  CheMPS2 output is temporarily written to the file %s and will be copied here.",
                chemps2filename.c_str());
            capturing.open(chemps2filename.c_str(), std::ios::trunc); // truncate
            cout_buffer = std::cout.rdbuf(capturing.rdbuf());

            unitary->updateUnitary(mem1, mem2, theupdate, true, true); // multiply = compact = true
            if ((dmrgscf_do_diis) && (updateNorm <= dmrgscf_diis_branch)) {
                if (dmrgscf_active_space.compare("NO") == 0) {
                    std::cout << "DIIS has started. Active space not rotated to NOs "
                                 "anymore!"
                              << std::endl;
                }
                if (dmrgscf_active_space.compare("LOC") == 0) {
                    std::cout << "DIIS has started. Active space not rotated to "
                                 "localized orbitals anymore!"
                              << std::endl;
                }
                if (theDIIS == NULL) {
                    theDIIS = new CheMPS2::DIIS(theDIISvectorParamSize, unitary->getNumVariablesX(),
                                                dmrgscf_num_vec_diis);
                    theDIISparameterVector = new double[theDIISvectorParamSize];
                    unitary->makeSureAllBlocksDetOne(mem1, mem2);
                }
                unitary->getLog(theDIISparameterVector, mem1, mem2);
                theDIIS->appendNew(theupdate, theDIISparameterVector);
                theDIIS->calculateParam(theDIISparameterVector);
                unitary->updateUnitary(mem1, mem2, theDIISparameterVector, false,
                                       false); // multiply = compact = false
            }

            std::cout.rdbuf(cout_buffer);
            capturing.close();
            std::ifstream copying;
            copying.open(chemps2filename, std::ios::in); // read only
            if (copying.is_open()) {
                std::string line;
                while (getline(copying, line)) {
                    outfile->Printf("\n  %s", line.c_str());
                }
                copying.close();
            }
            system(("rm " + chemps2filename).c_str());
        }
        if ((dmrgscf_store_unit) && (gradNorm != 1.0)) {
            unitary->saveU(unitaryname);
        }
        if ((dmrgscf_store_diis) && (updateNorm != 1.0) && (theDIIS != NULL)) {
            theDIIS->saveDIIS(diisname);
        }

        // Fill HamDMRG
        update_WFNco(Coeff_orig, iHandler, unitary, work1, work2);
        buildTmatrix(theTmatrix, iHandler, psio, ints_->Ca());
        buildQmatOCC(theQmatOCC, iHandler, work1, work2, ints_->Ca(), myJK);
        buildHamDMRG(ints, Aorbs_ptr, theTmatrix, theQmatOCC, iHandler, HamDMRG, psio);

        // Localize the active space and reorder the orbitals within each irrep
        // based on the exchange matrix
        if ((dmrgscf_active_space.compare("LOC") == 0) &&
            (theDIIS == NULL)) { // When the DIIS has started: stop

            std::ofstream capturing;
            std::streambuf* cout_buffer;
            std::string chemps2filename = "output.chemps2";
            outfile->Printf(
                "\n  CheMPS2 output is temporarily written to the file %s and will be copied here.",
                chemps2filename.c_str());
            capturing.open(chemps2filename.c_str(), std::ios::trunc); // truncate
            cout_buffer = std::cout.rdbuf(capturing.rdbuf());

            theLocalizer->Optimize(mem1, mem2, dmrgscf_loc_random);
            theLocalizer->FiedlerExchange(maxlinsize, mem1, mem2);
            CheMPS2::CASSCF::fillLocalizedOrbitalRotations(theLocalizer->getUnitary(), iHandler,
                                                           mem1);
            unitary->rotateActiveSpaceVectors(mem1, mem2);

            std::cout.rdbuf(cout_buffer);
            capturing.close();
            std::ifstream copying;
            copying.open(chemps2filename, std::ios::in); // read only
            if (copying.is_open()) {
                std::string line;
                while (getline(copying, line)) {
                    outfile->Printf("\n  %s", line.c_str());
                }
                copying.close();
            }
            system(("rm " + chemps2filename).c_str());

            update_WFNco(Coeff_orig, iHandler, unitary, work1, work2);
            buildTmatrix(theTmatrix, iHandler, psio, ints_->Ca());
            buildQmatOCC(theQmatOCC, iHandler, work1, work2, ints_->Ca(), myJK);
            buildHamDMRG(ints, Aorbs_ptr, theTmatrix, theQmatOCC, iHandler, HamDMRG, psio);
            outfile->Printf("\n  Rotated the active space to localized orbitals, sorted according "
                            "to the exchange matrix.");
        }

        // Do the DMRG sweeps, and calculate the 2DM
        {
            std::ofstream capturing;
            std::streambuf* cout_buffer;
            std::string chemps2filename = "output.chemps2";
            outfile->Printf(
                "\n  CheMPS2 output is temporarily written to the file and will be copied here.",
                chemps2filename.c_str());
            capturing.open(chemps2filename.c_str(), std::ios::trunc); // truncate
            cout_buffer = std::cout.rdbuf(capturing.rdbuf());

            for (int cnt = 0; cnt < nOrbDMRG_pow4; cnt++) {
                DMRG2DM[cnt] = 0.0;
            } // Clear the 2-RDM (to allow for state-averaged calculations)
            const string psi4TMPpath = psi::PSIOManager::shared_object()->get_default_path();
            CheMPS2::DMRG* theDMRG = new CheMPS2::DMRG(Prob, OptScheme, mps_chkpt, psi4TMPpath);
            for (int state = 0; state < dmrgscf_which_root; state++) {
                if (state > 0) {
                    theDMRG->newExcitation(std::fabs(Energy));
                }
                Energy = theDMRG->Solve();
                if (dmrgscf_state_avg) { // When SA-DMRGSCF: 2DM += current 2DM
                    // theDMRG->calc2DMandCorrelations();
                    theDMRG->calc_rdms_and_correlations(three_pdm);
                    CheMPS2::CASSCF::copy2DMover(theDMRG->get2DM(), nOrbDMRG, DMRG2DM);
                }
                if ((state == 0) && (dmrgscf_which_root > 1)) {
                    theDMRG->activateExcitations(dmrgscf_which_root - 1);
                }
            }
            if (!(dmrgscf_state_avg)) { // When SS-DMRGSCF: 2DM += last 2DM
                // theDMRG->calc2DMandCorrelations();
                theDMRG->calc_rdms_and_correlations(three_pdm);
                CheMPS2::CASSCF::copy2DMover(theDMRG->get2DM(), nOrbDMRG, DMRG2DM);
            }
            if (dmrg_print_corr) {
                theDMRG->getCorrelations()->Print();
            }
            if (CheMPS2::DMRG_storeRenormOptrOnDisk) {
                theDMRG->deleteStoredOperators();
            }
            if ((dmrgscf_state_avg) && (dmrgscf_which_root > 1)) {
                const double averagingfactor = 1.0 / dmrgscf_which_root;
                for (int cnt = 0; cnt < nOrbDMRG_pow4; cnt++) {
                    DMRG2DM[cnt] *= averagingfactor;
                }
            }
            CheMPS2::CASSCF::setDMRG1DM(nDMRGelectrons, nOrbDMRG, DMRG1DM, DMRG2DM);
            // CheMPS2::CASSCF::calcNOON( iHandler, mem1, mem2, DMRG1DM );
            if (three_pdm) {
                theDMRG->get3DM()->fill_ham_index(1.0, false, DMRG3DM, 0, nOrbDMRG);
            }
            delete theDMRG;

            std::cout.rdbuf(cout_buffer);
            capturing.close();
            std::ifstream copying;
            copying.open(chemps2filename, std::ios::in); // read only
            if (copying.is_open()) {
                std::string line;
                while (getline(copying, line)) {
                    outfile->Printf("\n  %s", line.c_str());
                }
                copying.close();
            }
            system(("rm " + chemps2filename).c_str());
        }

        bool wfn_co_updated = false;
        if ((dmrgscf_active_space.compare("NO") == 0) &&
            (theDIIS == NULL)) { // When the DIIS has started: stop
            CheMPS2::CASSCF::copy_active(DMRG1DM, theFmatrix, iHandler, true);
            CheMPS2::CASSCF::block_diagonalize('A', theFmatrix, unitary, mem1, mem2, iHandler, true,
                                               DMRG2DM, nullptr, nullptr);
            CheMPS2::CASSCF::setDMRG1DM(nDMRGelectrons, nOrbDMRG, DMRG1DM, DMRG2DM);
            update_WFNco(Coeff_orig, iHandler, unitary, work1, work2);
            wfn_co_updated = true;
            buildTmatrix(theTmatrix, iHandler, psio, ints_->Ca());
            buildQmatOCC(theQmatOCC, iHandler, work1, work2, ints_->Ca(), myJK);
            outfile->Printf(
                "\n  Rotated the active space to natural orbitals, sorted according to the NOON.");
        }

        if (dmrg_iterations_ == nIterations) {
            if (dmrgscf_store_unit) {
                unitary->saveU(unitaryname);
            }
            break;
        }

        if (!wfn_co_updated) {
            update_WFNco(Coeff_orig, iHandler, unitary, work1, work2);
        }
        buildQmatACT(theQmatACT, iHandler, DMRG1DM, work1, work2, ints_->Ca(), myJK);
        fillRotatedTEI_coulomb(ints, OAorbs_ptr, theTmatrix, theRotatedTEI, iHandler,
                               psio); // Also fills the T-matrix
        fillRotatedTEI_exchange(ints, OAorbs_ptr, Vorbs_ptr, theRotatedTEI, iHandler, psio);

        {
            std::ofstream capturing;
            std::streambuf* cout_buffer;
            std::string chemps2filename = "output.chemps2";
            outfile->Printf(
                "\n  CheMPS2 output is temporarily written to the file %s and will be copied here.",
                chemps2filename.c_str());
            capturing.open(chemps2filename.c_str(), std::ios::trunc); // truncate
            cout_buffer = std::cout.rdbuf(capturing.rdbuf());

            CheMPS2::CASSCF::buildFmat(theFmatrix, theTmatrix, theQmatOCC, theQmatACT, iHandler,
                                       theRotatedTEI, DMRG2DM, DMRG1DM);
            CheMPS2::CASSCF::buildWtilde(wmattilde, theTmatrix, theQmatOCC, theQmatACT, iHandler,
                                         theRotatedTEI, DMRG2DM, DMRG1DM);
            CheMPS2::CASSCF::augmentedHessianNR(theFmatrix, wmattilde, iHandler, unitary, theupdate,
                                                &updateNorm, &gradNorm);

            std::cout.rdbuf(cout_buffer);
            capturing.close();
            std::ifstream copying;
            copying.open(chemps2filename, std::ios::in); // read only
            if (copying.is_open()) {
                std::string line;
                while (getline(copying, line)) {
                    outfile->Printf("\n  %s", line.c_str());
                }
                copying.close();
            }
            system(("rm " + chemps2filename).c_str());
        }
    }
    compute_reference(DMRG1DM, DMRG2DM, DMRG3DM, iHandler);
    for (int i = 0; i < nOrbDMRG; i++)
        for (int j = 0; j < nOrbDMRG; j++)

            delete[] mem1;
    delete[] mem2;
    delete[] theupdate;
    if (theDIISparameterVector != NULL) {
        delete[] theDIISparameterVector;
    }
    if (theLocalizer != NULL) {
        delete theLocalizer;
    }
    if (theDIIS != NULL) {
        delete theDIIS;
    }

    delete wmattilde;
    delete theTmatrix;
    delete theQmatOCC;
    delete theQmatACT;
    delete theFmatrix;
    delete[] DMRG1DM;
    delete[] DMRG2DM;
    if (three_pdm) {
        delete[] DMRG3DM;
    }
    delete theRotatedTEI;
    delete unitary;
    delete iHandler;

    delete OptScheme;
    delete Prob;
    delete HamDMRG;

    outfile->Printf("The DMRG-SCF energy = %3.10f \n", Energy);
    psi::Process::environment.globals["CURRENT ENERGY"] = Energy;
    psi::Process::environment.globals["DMRGSCF ENERGY"] = Energy;
    dmrg_ref_.set_Eref(Energy);
    return Energy;
}
void DMRGSCF::compute_reference(double* one_rdm, double* two_rdm, double* three_rdm,
                                CheMPS2::DMRGSCFindices* iHandler) {
    // if(options_->get_int("MULTIPLICITY") != 1 &&
    // options_->get_int("DMRG_WFN_MULTP") != 1)
    //{
    //    outfile->Printf("\n\n Spinadapted formalism requires spin-averaged
    //    quantitities");
    //    throw psi::PSIEXCEPTION("You need to spin averaged things");
    //}
    RDMs dmrg_ref;
    size_t na = mo_space_info_->size("ACTIVE");
    ambit::Tensor gamma1_a = ambit::Tensor::build(ambit::CoreTensor, "gamma1_a", {na, na});
    ambit::Tensor gamma2_dmrg =
        ambit::Tensor::build(ambit::CoreTensor, "Gamma2_DMRG", {na, na, na, na});
    ambit::Tensor gamma2_aa =
        ambit::Tensor::build(ambit::CoreTensor, "gamma2_aa", {na, na, na, na});
    ambit::Tensor gamma2_ab =
        ambit::Tensor::build(ambit::CoreTensor, "gamma2_ab", {na, na, na, na});

    const int nOrbDMRG = iHandler->getDMRGcumulative(iHandler->getNirreps());
    std::vector<double>& gamma1_data = gamma1_a.data();
    for (int irrep = 0; irrep < iHandler->getNirreps(); irrep++) {
        const int shift = iHandler->getDMRGcumulative(irrep);
        for (int orb1 = 0; orb1 < iHandler->getNDMRG(irrep); orb1++) {
            for (int orb2 = orb1; orb2 < iHandler->getNDMRG(irrep); orb2++) {
                const double value = one_rdm[shift + orb1 + nOrbDMRG * (shift + orb2)];
                gamma1_data[shift + orb1 + nOrbDMRG * (shift + orb2)] = 0.5 * value;
                gamma1_data[shift + orb2 + nOrbDMRG * (shift + orb1)] = 0.5 * value;
            }
        }
    }
    /// Gamma_a = 1_RDM / 2
    /// Gamma_b = 1_RDM / 2
    dmrg_ref.set_L1a(gamma1_a);
    dmrg_ref.set_L1b(gamma1_a);
    /// Form 2_rdms
    {
        gamma2_dmrg.iterate([&](const std::vector<size_t>& i, double& value) {
            value = two_rdm[i[0] * na * na * na + i[1] * na * na + i[2] * na + i[3]];
        });
        /// gamma2_aa = 1 / 6 * (Gamma2(pqrs) - Gamma2(pqsr))
        gamma2_aa.copy(gamma2_dmrg);
        gamma2_aa("p, q, r, s") = gamma2_dmrg("p, q, r, s") - gamma2_dmrg("p, q, s, r");
        gamma2_aa.scale(1.0 / 6.0);

        gamma2_ab("p, q, r, s") = (2.0 * gamma2_dmrg("p, q, r, s") + gamma2_dmrg("p, q, s, r"));
        gamma2_ab.scale(1.0 / 6.0);
        dmrg_ref.set_g2aa(gamma2_aa);
        dmrg_ref.set_g2bb(gamma2_aa);
        dmrg_ref.set_g2ab(gamma2_ab);
        ambit::Tensor cumulant2_aa =
            ambit::Tensor::build(ambit::CoreTensor, "Cumulant2_aa", {na, na, na, na});
        ambit::Tensor cumulant2_ab =
            ambit::Tensor::build(ambit::CoreTensor, "Cumulant2_ab", {na, na, na, na});
        cumulant2_aa.copy(gamma2_aa);
        cumulant2_aa("pqrs") -= gamma1_a("pr") * gamma1_a("qs");
        cumulant2_aa("pqrs") += gamma1_a("ps") * gamma1_a("qr");

        cumulant2_ab.copy(gamma2_ab);
        cumulant2_ab("pqrs") -= gamma1_a("pr") * gamma1_a("qs");
        dmrg_ref.set_L2aa(cumulant2_aa);
        dmrg_ref.set_L2ab(cumulant2_ab);
        dmrg_ref.set_L2bb(cumulant2_aa);
    }
    if ((options_->get_str("THREEPDC") != "ZERO") &&
        (options_->get_str("JOB_TYPE") == "DSRG-MRPT2" or
         options_->get_str("JOB_TYPE") == "THREE-DSRG-MRPT2")) {
        ambit::Tensor gamma3_dmrg =
            ambit::Tensor::build(ambit::CoreTensor, "Gamma3_DMRG", {na, na, na, na, na, na});
        ambit::Tensor gamma3_aaa =
            ambit::Tensor::build(ambit::CoreTensor, "Gamma3_aaa", {na, na, na, na, na, na});
        ambit::Tensor gamma3_aab =
            ambit::Tensor::build(ambit::CoreTensor, "Gamma3_aab", {na, na, na, na, na, na});
        ambit::Tensor gamma3_abb =
            ambit::Tensor::build(ambit::CoreTensor, "Gamma2_abb", {na, na, na, na, na, na});
        gamma3_dmrg.iterate([&](const std::vector<size_t>& i, double& value) {
            value = three_rdm[i[0] * na * na * na * na * na + i[1] * na * na * na * na +
                              i[2] * na * na * na + i[3] * na * na + i[4] * na + i[5]];
        });
        gamma3_aaa("p, q, r, s, t, u") = gamma3_dmrg("p, q, r, s, t, u") +
                                         gamma3_dmrg("p, q, r, t, u, s") +
                                         gamma3_dmrg("p, q, r, u, s, t");
        gamma3_aaa.scale(1.0 / 12.0);
        gamma3_aab("p, q, r, s, t, u") =
            (gamma3_dmrg("p, q, r, s, t, u") - gamma3_dmrg("p, q, r, t, u, s") -
             gamma3_dmrg("p, q, r, u, s, t") - 2.0 * gamma3_dmrg("p, q, r, t, s, u"));
        gamma3_aab.scale(1.0 / 12.0);
        // gamma3_abb("p, q, r, s, t, u") = (gamma3_dmrg("p, q, r, s, t, u") -
        // gamma3_dmrg("p, q, r, t, u, s") - gamma3_dmrg("p, q, r, u, s, t") -
        // 2.0 * gamma3_dmrg("p, q, r, t, s, u"));
        // gamma3_abb.scale(1.0/12.0);

        ambit::Tensor L1a = dmrg_ref.g1a();
        ambit::Tensor L1b = dmrg_ref.g1b();
        ambit::Tensor L2aa = dmrg_ref.L2aa();
        ambit::Tensor L2ab = dmrg_ref.L2ab();
        // ambit::Tensor L2bb = dmrg_ref.L2bb();
        // Convert the 3-RDMs to 3-RCMs
        gamma3_aaa("pqrstu") -= L1a("ps") * L2aa("qrtu");
        gamma3_aaa("pqrstu") += L1a("pt") * L2aa("qrsu");
        gamma3_aaa("pqrstu") += L1a("pu") * L2aa("qrts");

        gamma3_aaa("pqrstu") -= L1a("qt") * L2aa("prsu");
        gamma3_aaa("pqrstu") += L1a("qs") * L2aa("prtu");
        gamma3_aaa("pqrstu") += L1a("qu") * L2aa("prst");

        gamma3_aaa("pqrstu") -= L1a("ru") * L2aa("pqst");
        gamma3_aaa("pqrstu") += L1a("rs") * L2aa("pqut");
        gamma3_aaa("pqrstu") += L1a("rt") * L2aa("pqsu");

        gamma3_aaa("pqrstu") -= L1a("ps") * L1a("qt") * L1a("ru");
        gamma3_aaa("pqrstu") -= L1a("pt") * L1a("qu") * L1a("rs");
        gamma3_aaa("pqrstu") -= L1a("pu") * L1a("qs") * L1a("rt");

        gamma3_aaa("pqrstu") += L1a("ps") * L1a("qu") * L1a("rt");
        gamma3_aaa("pqrstu") += L1a("pu") * L1a("qt") * L1a("rs");
        gamma3_aaa("pqrstu") += L1a("pt") * L1a("qs") * L1a("ru");

        gamma3_aab("pqRstU") -= L1a("ps") * L2ab("qRtU");
        gamma3_aab("pqRstU") += L1a("pt") * L2ab("qRsU");

        gamma3_aab("pqRstU") -= L1a("qt") * L2ab("pRsU");
        gamma3_aab("pqRstU") += L1a("qs") * L2ab("pRtU");

        gamma3_aab("pqRstU") -= L1b("RU") * L2aa("pqst");

        gamma3_aab("pqRstU") -= L1a("ps") * L1a("qt") * L1b("RU");
        gamma3_aab("pqRstU") += L1a("pt") * L1a("qs") * L1b("RU");

        // gamma3_abb("pQRsTU") -= L1a("ps") * L2aa("QRTU");

        // gamma3_abb("pQRsTU") -= L1b("QT") * L2ab("pRsU");
        // gamma3_abb("pQRsTU") += L1b("QU") * L2ab("pRsT");

        // gamma3_abb("pQRsTU") -= L1b("RU") * L2ab("pQsT");
        // gamma3_abb("pQRsTU") += L1b("RT") * L2ab("pQsU");

        // gamma3_abb("pQRsTU") -= L1a("ps") * L1b("QT") * L1b("RU");
        // gamma3_abb("pQRsTU") += L1a("ps") * L1b("QU") * L1b("RT");
        gamma3_abb("p, q, r, s, t, u") = gamma3_aab("q,r,p,t,u,s");

        dmrg_ref.set_L3aaa(gamma3_aaa);
        dmrg_ref.set_L3aab(gamma3_aab);
        dmrg_ref.set_L3abb(gamma3_abb);
        dmrg_ref.set_L3bbb(gamma3_aaa);
    }
    dmrg_ref_ = dmrg_ref;
}
} // namespace forte

#endif // #ifdef HAVE_CHEMPS2
