/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2021 by its authors (see COPYING, COPYING.LESSER, AUTHORS).
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

#include <algorithm>
#include <ctype.h>
#include <fstream>
#include <iostream>
#include <numeric>
#include <string>
#include <utility>

#include "psi4/libpsi4util/PsiOutStream.h"

#include "boost/algorithm/string/predicate.hpp"
#include "boost/format.hpp"

#include "psi4/libqt/qt.h"

#include "mcsrgpt2_mo.h"
#include "sci/fci_mo.h"
#include "orbital-helpers/semi_canonicalize.h"
#include "helpers/printing.h"

#define Delta(i, j) ((i == j) ? 1 : 0)

using namespace psi;

namespace forte {

MCSRGPT2_MO::MCSRGPT2_MO(RDMs rdms, std::shared_ptr<ForteOptions> options,
                         std::shared_ptr<ForteIntegrals> ints,
                         std::shared_ptr<MOSpaceInfo> mo_space_info)
    : integral_(ints), reference_(rdms), mo_space_info_(mo_space_info), options_(options) {

    print_method_banner({"(Driven) Similarity Renormalization Group",
                         "Second-Order Perturbative Analysis", "Chenyang Li"});

    // prepare mo space indices
    prepare_mo_space();

    // fill in non-tensor based cumulants
    int max_rdm_level = (options->get_str("THREEPDC") == "ZERO") ? 2 : 3;
    fill_naive_cumulants(rdms, max_rdm_level);

    // build Fock matrix
    Fa_ = d2(ncmo_, d1(ncmo_));
    Fb_ = d2(ncmo_, d1(ncmo_));
    Form_Fock(Fa_, Fb_);

    startup();
}

MCSRGPT2_MO::~MCSRGPT2_MO() { cleanup(); }

void MCSRGPT2_MO::cleanup() {
    //    delete integral_;
}

double MCSRGPT2_MO::compute_energy() {
    if (options_->get_str("CORR_LEVEL") == "SRG_PT2") {
        return compute_energy_srg();
    } else {
        Form_AMP_DSRG();
        return compute_energy_dsrg();
    }
}

void MCSRGPT2_MO::prepare_mo_space() {
    // MO space
    ncmo_ = mo_space_info_->size("CORRELATED");
    nactv_ = mo_space_info_->size("ACTIVE");
    ncore_ = mo_space_info_->size("RESTRICTED_DOCC");
    nvirt_ = mo_space_info_->size("RESTRICTED_UOCC");

    // obtain absolute indices of core, active and virtual
    core_mos_ = mo_space_info_->corr_absolute_mo("RESTRICTED_DOCC");
    actv_mos_ = mo_space_info_->corr_absolute_mo("ACTIVE");
    virt_mos_ = mo_space_info_->corr_absolute_mo("RESTRICTED_UOCC");

    // setup hole and particle indices (Active must start first)
    nhole_ = ncore_ + nactv_;
    npart_ = nactv_ + nvirt_;
    hole_mos_ = std::vector<size_t>(actv_mos_);
    hole_mos_.insert(hole_mos_.end(), core_mos_.begin(), core_mos_.end());
    part_mos_ = std::vector<size_t>(actv_mos_);
    part_mos_.insert(part_mos_.end(), virt_mos_.begin(), virt_mos_.end());

    // setup symmetry index of active/correlated orbitals
    auto actv_dim = mo_space_info_->dimension("ACTIVE");
    auto ncmopi = mo_space_info_->dimension("CORRELATED");
    for (int h = 0, nirrep = mo_space_info_->nirrep(); h < nirrep; ++h) {
        for (size_t i = 0; i < size_t(actv_dim[h]); ++i) {
            sym_actv_.push_back(h);
        }
        for (size_t i = 0; i < size_t(ncmopi[h]); ++i) {
            sym_ncmo_.push_back(h);
        }
    }
}

void MCSRGPT2_MO::startup() {
    // Source Operator
    source_ = options_->get_str("SOURCE");
    if (sourcemap.find(source_) == sourcemap.end()) {
        outfile->Printf("\n  Source operator %s is not available.", source_.c_str());
        outfile->Printf("\n  Only these source operators are available: ");
        for (const auto& keys : sourcemap) {
            std::string key = keys.first;
            outfile->Printf("%s ", key.c_str());
        }
        outfile->Printf("\n");
        throw psi::PSIEXCEPTION("Source operator is not available.");
    }

    // Print Delta
    print_ = options_->get_int("PRINT");
    if (print_ > 1) {
        PrintDelta();
        test_D1_RE();
        test_D2_RE();
    }

    // DSRG Parameters
    s_ = options_->get_double("DSRG_S");
    if (s_ < 0) {
        throw psi::PSIEXCEPTION("DSRG_S cannot be negative numbers.");
    }
    taylor_threshold_ = options_->get_int("TAYLOR_THRESHOLD");
    if (taylor_threshold_ <= 0) {
        throw psi::PSIEXCEPTION("TAYLOR_THRESHOLD must be an integer greater than 0.");
    }
    expo_delta_ = options_->get_double("DSRG_POWER");
    if (expo_delta_ <= 1.0) {
        throw psi::PSIEXCEPTION("DELTA_EXPONENT must be greater than 1.0.");
    }
    double e_conv = -log10(options_->get_double("E_CONVERGENCE"));
    taylor_order_ = floor((e_conv / taylor_threshold_ + 1.0) / expo_delta_) + 1;

    // Print Original Orbital Indices
    print_h2("Correlated Subspace Indices");
    auto print_idx = [&](const std::string& str, const std::vector<size_t>& vec) {
        outfile->Printf("\n    %-30s", str.c_str());
        size_t c = 0;
        for (size_t x : vec) {
            outfile->Printf("%4zu ", x);
            ++c;
            if (c % 15 == 0)
                outfile->Printf("\n  %-32c", ' ');
        }
    };
    print_idx("CORE", core_mos_);
    print_idx("ACTIVE", actv_mos_);
    print_idx("HOLE", hole_mos_);
    print_idx("VIRTUAL", virt_mos_);
    print_idx("PARTICLE", part_mos_);
    outfile->Printf("\n");

    // Compute RDMs Energy
    outfile->Printf("\n  Computing reference energy using density cumulant ...");
    compute_ref();
    outfile->Printf("\t\t\tDone.");

    // 2-Particle Density Cumulant
    std::string twopdc = options_->get_str("TWOPDC");
    if (twopdc == "ZERO") {
        L2aa_ = d4(nactv_, d3(nactv_, d2(nactv_, d1(nactv_))));
        L2ab_ = d4(nactv_, d3(nactv_, d2(nactv_, d1(nactv_))));
        L2bb_ = d4(nactv_, d3(nactv_, d2(nactv_, d1(nactv_))));
    }
}

double MCSRGPT2_MO::ElementRH(const std::string& source, const double& D, const double& V) {
    if (std::fabs(V) < 1.0e-12)
        return 0.0;
    switch (sourcemap[source]) {
    case AMP: {
        double RD = D / V;
        return V * exp(-s_ * pow(std::fabs(RD), expo_delta_));
    }
    case EMP2: {
        double RD = D / (V * V);
        return V * exp(-s_ * pow(std::fabs(RD), expo_delta_));
    }
    case LAMP: {
        double RD = D / V;
        //        outfile->Printf("\n  D = %20.15f, V = %20.15f, RD = %20.15f,
        //        EXP = %20.15f", D, V, RD, V * exp(-s_ * std::fabs(RD)));
        return V * exp(-s_ * std::fabs(RD));
    }
    case LEMP2: {
        double RD = D / (V * V);
        return V * exp(-s_ * std::fabs(RD));
    }
    default: {
        return V * exp(-s_ * pow(std::fabs(D), expo_delta_));
    }
    }
}

void MCSRGPT2_MO::Form_Fock_SRG() {
    timer_on("Fock_SRG");

    //    for(size_t i = 0; i < nh_; ++i){
    //        size_t ni = idx_h_[i];
    //        for(size_t a = 0; a < npt_; ++a){
    //            size_t na = idx_p_[a];

    //            Fa_srg_[ni][na] = Fa_[ni][na];
    //            Fa_srg_[na][ni] = Fa_[na][ni];
    //            Fb_srg_[ni][na] = Fb_[ni][na];
    //            Fb_srg_[na][ni] = Fb_[na][ni];

    //            double va = 0.0, vb = 0.0;
    //            for(size_t u = 0; u < na_; ++u){
    //                size_t nu = idx_a_[u];
    //                for(size_t v = 0; v < na_; ++v){
    //                    size_t nv = idx_a_[v];

    //                    va += Da_[nu][nv] * integral_->aptei_aa(ni,nv,na,nu);
    //                    va += Db_[nu][nv] * integral_->aptei_ab(ni,nv,na,nu);
    //                    vb += Da_[nu][nv] * integral_->aptei_ab(nv,ni,nu,na);
    //                    vb += Db_[nu][nv] * integral_->aptei_bb(ni,nv,na,nu);
    //                }
    //            }

    //            Fa_srg_[ni][na] -= va;
    //            Fa_srg_[na][ni] -= va;
    //            Fb_srg_[ni][na] -= vb;
    //            Fb_srg_[na][ni] -= vb;
    //        }
    //    }

    for (size_t p = 0; p < ncmo_; ++p) {
        for (size_t q = 0; q < ncmo_; ++q) {
            double va = integral_->oei_a(p, q);
            double vb = integral_->oei_b(p, q);

            for (size_t m = 0; m < ncore_; ++m) {
                size_t nm = core_mos_[m];

                va += integral_->aptei_aa(p, nm, q, nm);
                va += integral_->aptei_ab(p, nm, q, nm);
                vb += integral_->aptei_bb(p, nm, q, nm);
                vb += integral_->aptei_ab(nm, p, nm, q);
            }

            Fa_srg_[p][q] = va;
            Fb_srg_[p][q] = vb;
        }
    }

    for (size_t u = 0; u < nactv_; ++u) {
        size_t nu = actv_mos_[u];
        for (size_t v = 0; v < nactv_; ++v) {
            size_t nv = actv_mos_[v];
            Fa_srg_[nu][nv] = 0.0;
            Fb_srg_[nu][nv] = 0.0;
        }
    }

    timer_off("Fock_SRG");
}

void MCSRGPT2_MO::Form_Fock_DSRG(d2& A, d2& B, const bool& dsrgpt) {
    timer_on("Fock_DSRG");
    for (size_t p = 0; p < ncmo_; ++p) {
        for (size_t q = 0; q < ncmo_; ++q) {
            A[p][q] = Fa_[p][q];
            B[p][q] = Fb_[p][q];
        }
    }
    if (dsrgpt) {
        for (size_t i = 0; i < nhole_; ++i) {
            size_t ni = hole_mos_[i];
            for (size_t a = 0; a < npart_; ++a) {
                size_t na = part_mos_[a];
                double value_a = 0.0, value_b = 0.0;
                for (size_t u = 0; u < nactv_; ++u) {
                    size_t nu = actv_mos_[u];
                    for (size_t x = 0; x < nactv_; ++x) {
                        size_t nx = actv_mos_[x];
                        value_a += (Fa_[nx][nx] - Fa_[nu][nu]) * T2aa_[i][u][a][x] * Da_[nx][nu];
                        value_a += (Fb_[nx][nx] - Fb_[nu][nu]) * T2ab_[i][u][a][x] * Db_[nx][nu];
                        value_b += (Fa_[nx][nx] - Fa_[nu][nu]) * T2ab_[u][i][x][a] * Da_[nx][nu];
                        value_b += (Fb_[nx][nx] - Fb_[nu][nu]) * T2bb_[i][u][a][x] * Db_[nx][nu];
                    }
                }
                value_a += Fa_[ni][na];
                value_b += Fb_[ni][na];

                double Da = Fa_[ni][ni] - Fa_[na][na];
                double Db = Fb_[ni][ni] - Fb_[na][na];

                A[ni][na] += ElementRH(source_, Da, value_a);
                A[na][ni] += ElementRH(source_, Da, value_a);
                B[ni][na] += ElementRH(source_, Db, value_b);
                B[na][ni] += ElementRH(source_, Db, value_b);
            }
        }
    }
    timer_off("Fock_DSRG");
}

void MCSRGPT2_MO::Form_APTEI_DSRG(const bool& dsrgpt) {
    timer_on("APTEI_DSRG");

    if (dsrgpt) {
        for (size_t i = 0; i < nhole_; ++i) {
            size_t ni = hole_mos_[i];
            for (size_t j = 0; j < nhole_; ++j) {
                size_t nj = hole_mos_[j];
                for (size_t a = 0; a < npart_; ++a) {
                    size_t na = part_mos_[a];
                    for (size_t b = 0; b < npart_; ++b) {
                        size_t nb = part_mos_[b];
                        double Daa = Fa_[ni][ni] + Fa_[nj][nj] - Fa_[na][na] - Fa_[nb][nb];
                        double Dab = Fa_[ni][ni] + Fb_[nj][nj] - Fa_[na][na] - Fb_[nb][nb];
                        double Dbb = Fb_[ni][ni] + Fb_[nj][nj] - Fb_[na][na] - Fb_[nb][nb];

                        double Vaa = integral_->aptei_aa(ni, nj, na, nb);
                        double Vab = integral_->aptei_ab(ni, nj, na, nb);
                        double Vbb = integral_->aptei_bb(ni, nj, na, nb);

                        Vaa += ElementRH(source_, Daa, Vaa);
                        Vab += ElementRH(source_, Dab, Vab);
                        Vbb += ElementRH(source_, Dbb, Vbb);

                        integral_->set_tei(ni, nj, na, nb, Vaa, true, true);
                        integral_->set_tei(na, nb, ni, nj, Vaa, true, true);

                        integral_->set_tei(ni, nj, na, nb, Vab, true, false);
                        integral_->set_tei(na, nb, ni, nj, Vab, true, false);

                        integral_->set_tei(ni, nj, na, nb, Vbb, false, false);
                        integral_->set_tei(na, nb, ni, nj, Vbb, false, false);
                    }
                }
            }
        }
    }
    timer_off("APTEI_DSRG");
}

void MCSRGPT2_MO::compute_ref() {
    timer_on("Compute Ref");
    Eref_ = 0.0;
    for (size_t p = 0; p < nhole_; ++p) {
        size_t np = hole_mos_[p];
        for (size_t q = 0; q < nhole_; ++q) {
            size_t nq = hole_mos_[q];
            Eref_ += (integral_->oei_a(nq, np) + Fa_[nq][np]) * Da_[np][nq];
            Eref_ += (integral_->oei_b(nq, np) + Fb_[nq][np]) * Db_[np][nq];
        }
    }
    Eref_ *= 0.5;
    for (size_t p = 0; p < nactv_; ++p) {
        size_t np = actv_mos_[p];
        for (size_t q = 0; q < nactv_; ++q) {
            size_t nq = actv_mos_[q];
            for (size_t r = 0; r < nactv_; ++r) {
                size_t nr = actv_mos_[r];
                for (size_t s = 0; s < nactv_; ++s) {
                    size_t ns = actv_mos_[s];
                    Eref_ += 0.25 * integral_->aptei_aa(np, nq, nr, ns) * L2aa_[p][q][r][s];
                    Eref_ += 0.25 * integral_->aptei_bb(np, nq, nr, ns) * L2bb_[p][q][r][s];
                    Eref_ += integral_->aptei_ab(np, nq, nr, ns) * L2ab_[p][q][r][s];
                }
            }
        }
    }
    Eref_ += integral_->nuclear_repulsion_energy() + integral_->frozen_core_energy();
    //    outfile->Printf("\n    E0 (cumulant) %15c = %22.15f", ' ', Eref_);
    timer_off("Compute Ref");
}

double MCSRGPT2_MO::ElementT(const std::string& source, const double& D, const double& V) {
    if (std::fabs(V) < 1.0e-12)
        return 0.0;
    switch (sourcemap[source]) {
    case AMP: {
        double RD = D / V;
        double Z = pow(s_, 1 / expo_delta_) * RD;
        if (std::fabs(Z) < pow(0.1, taylor_threshold_)) {
            return Taylor_Exp(Z, taylor_order_, expo_delta_) * sqrt(s_);
        } else {
            return (1 - exp(-1.0 * s_ * pow(std::fabs(RD), expo_delta_))) * V / D;
        }
    }
    case EMP2: {
        double RD = D / V;
        double Z = pow(s_, 1 / expo_delta_) * RD / V;
        if (std::fabs(Z) < pow(0.1, taylor_threshold_)) {
            return Taylor_Exp(Z, taylor_order_, expo_delta_) * sqrt(s_) / V;
        } else {
            return (1 - exp(-1.0 * s_ * pow(std::fabs(RD / V), expo_delta_))) * V / D;
        }
    }
    case LAMP: {
        double RD = D / V;
        double Z = s_ * RD;
        if (std::fabs(Z) < pow(0.1, taylor_threshold_)) {
            return Taylor_Exp_Linear(Z, 2 * taylor_order_) * s_;
        } else {
            return (1 - exp(-1.0 * s_ * std::fabs(RD))) * V / D;
        }
    }
    case LEMP2: {
        double RD = D / V;
        double Z = s_ * RD / V;
        if (std::fabs(Z) < pow(0.1, taylor_threshold_)) {
            return Taylor_Exp_Linear(Z, 2 * taylor_order_) * s_ / V;
        } else {
            return (1 - exp(-1.0 * s_ * std::fabs(RD / V))) * V / D;
        }
    }
    default: {
        double Z = pow(s_, 1 / expo_delta_) * D;
        if (std::fabs(Z) < pow(0.1, taylor_threshold_)) {
            return Taylor_Exp(Z, taylor_order_, expo_delta_) * pow(s_, 1 / expo_delta_) * V;
        } else {
            return (1 - exp(-1.0 * s_ * pow(std::fabs(D), expo_delta_))) * V / D;
        }
    }
    }
}

void MCSRGPT2_MO::Form_T2_DSRG(d4& AA, d4& AB, d4& BB, std::string& T_ALGOR) {
    timer_on("Form T2");
    for (size_t i = 0; i < nhole_; ++i) {
        size_t ni = hole_mos_[i];
        for (size_t j = 0; j < nhole_; ++j) {
            size_t nj = hole_mos_[j];
            for (size_t a = 0; a < npart_; ++a) {
                size_t na = part_mos_[a];
                for (size_t b = 0; b < npart_; ++b) {
                    size_t nb = part_mos_[b];

                    double Daa = Fa_[ni][ni] + Fa_[nj][nj] - Fa_[na][na] - Fa_[nb][nb];
                    double Dab = Fa_[ni][ni] + Fb_[nj][nj] - Fa_[na][na] - Fb_[nb][nb];
                    double Dbb = Fb_[ni][ni] + Fb_[nj][nj] - Fb_[na][na] - Fb_[nb][nb];

                    double Vaa = integral_->aptei_aa(na, nb, ni, nj);
                    double Vab = integral_->aptei_ab(na, nb, ni, nj);
                    double Vbb = integral_->aptei_bb(na, nb, ni, nj);

                    AA[i][j][a][b] = ElementT(source_, Daa, Vaa);
                    AB[i][j][a][b] = ElementT(source_, Dab, Vab);
                    BB[i][j][a][b] = ElementT(source_, Dbb, Vbb);
                }
            }
        }
    }

    // Zero Internal Excitations
    for (size_t i = 0; i < nactv_; ++i) {
        for (size_t j = 0; j < nactv_; ++j) {
            for (size_t k = 0; k < nactv_; ++k) {
                for (size_t l = 0; l < nactv_; ++l) {
                    AA[i][j][k][l] = 0.0;
                    AB[i][j][k][l] = 0.0;
                    BB[i][j][k][l] = 0.0;
                }
            }
        }
    }

    // Zero Semi-Internal Excitations
    if (T_ALGOR == "DSRG_NOSEMI") {
        outfile->Printf("\n  Exclude excitations of (active, active -> active, "
                        "virtual) and (core, active -> active, active).");
        for (size_t x = 0; x < nactv_; ++x) {
            for (size_t y = 0; y < nactv_; ++y) {
                for (size_t z = 0; z < nactv_; ++z) {
                    for (size_t nm = 0; nm < ncore_; ++nm) {
                        size_t m = nm + nactv_;
                        AA[m][z][y][x] = 0.0;
                        AA[z][m][y][x] = 0.0;
                        AB[m][z][y][x] = 0.0;
                        AB[z][m][y][x] = 0.0;
                        BB[m][z][y][x] = 0.0;
                        BB[z][m][y][x] = 0.0;
                    }
                    for (size_t ne = 0; ne < nvirt_; ++ne) {
                        size_t e = ne + nactv_;
                        AA[x][y][z][e] = 0.0;
                        AA[x][y][e][z] = 0.0;
                        AB[x][y][z][e] = 0.0;
                        AB[x][y][e][z] = 0.0;
                        BB[x][y][z][e] = 0.0;
                        BB[x][y][e][z] = 0.0;
                    }
                }
            }
        }
    }
    timer_off("Form T2");
}

void MCSRGPT2_MO::Form_T1_DSRG(d2& A, d2& B) {
    timer_on("Form T1");
    for (size_t i = 0; i < nhole_; ++i) {
        size_t ni = hole_mos_[i];
        for (size_t a = 0; a < npart_; ++a) {
            size_t na = part_mos_[a];

            double Da = Fa_[ni][ni] - Fa_[na][na];
            double Db = Fb_[ni][ni] - Fb_[na][na];

            double RFa = Fa_[ni][na];
            double RFb = Fb_[ni][na];

            for (size_t u = 0; u < nactv_; ++u) {
                size_t nu = actv_mos_[u];
                for (size_t x = 0; x < nactv_; ++x) {
                    size_t nx = actv_mos_[x];

                    double A_Da = Fa_[nx][nx] - Fa_[nu][nu];
                    double A_Db = Fb_[nx][nx] - Fb_[nu][nu];

                    if (t1_amp_ == "SRG") {
                        double Vaa = integral_->aptei_aa(na, nx, ni, nu);
                        double Vbb = integral_->aptei_bb(na, nx, ni, nu);
                        double Vab_aa = integral_->aptei_ab(nx, na, nu, ni);
                        double Vab_ab = integral_->aptei_ab(na, nx, ni, nu);

                        double factor = 0.0;
                        factor = 1.0 - exp(s_ * (2 * Da - A_Da) * A_Da);
                        RFa -= Vaa * Da_[nx][nu] * factor;

                        factor = 1.0 - exp(s_ * (2 * Da - A_Db) * A_Db);
                        RFa -= Vab_ab * Db_[nx][nu] * factor;

                        factor = 1.0 - exp(s_ * (2 * Db - A_Da) * A_Da);
                        RFb -= Vab_aa * Da_[nx][nu] * factor;

                        factor = 1.0 - exp(s_ * (2 * Db - A_Db) * A_Db);
                        RFb -= Vbb * Db_[nx][nu] * factor;
                    } else {
                        RFa += A_Da * T2aa_[i][u][a][x] * Da_[nx][nu];
                        RFa += A_Db * T2ab_[i][u][a][x] * Db_[nx][nu];
                        RFb += A_Da * T2ab_[u][i][x][a] * Da_[nx][nu];
                        RFb += A_Db * T2bb_[i][u][a][x] * Db_[nx][nu];
                    }
                }
            }

            A[i][a] = ElementT(source_, Da, RFa);
            B[i][a] = ElementT(source_, Db, RFb);
        }
    }

    // Zero Internal Excitations
    for (size_t i = 0; i < nactv_; ++i) {
        for (size_t j = 0; j < nactv_; ++j) {
            A[i][j] = 0.0;
            B[i][j] = 0.0;
        }
    }
    timer_off("Form T1");
}

void MCSRGPT2_MO::Form_T2_SELEC(d4& AA, d4& AB, d4& BB) {
    timer_on("Form T2");
    for (size_t i = 0; i < ncore_; ++i) {
        size_t ni = core_mos_[i];
        for (size_t j = 0; j < ncore_; ++j) {
            size_t nj = core_mos_[j];
            for (size_t a = 0; a < nactv_; ++a) {
                size_t na = actv_mos_[a];
                for (size_t b = 0; b < nactv_; ++b) {
                    size_t nb = actv_mos_[b];

                    double Daa = Fa_[ni][ni] + Fa_[nj][nj] - Fa_[na][na] - Fa_[nb][nb];
                    double Dab = Fa_[ni][ni] + Fb_[nj][nj] - Fa_[na][na] - Fb_[nb][nb];
                    double Dbb = Fb_[ni][ni] + Fb_[nj][nj] - Fb_[na][na] - Fb_[nb][nb];

                    double Vaa = integral_->aptei_aa(na, nb, ni, nj);
                    double Vab = integral_->aptei_ab(na, nb, ni, nj);
                    double Vbb = integral_->aptei_bb(na, nb, ni, nj);

                    i += nactv_;
                    j += nactv_;
                    AA[i][j][a][b] = ElementT("STANDARD", Daa, Vaa);
                    AB[i][j][a][b] = ElementT("STANDARD", Dab, Vab);
                    BB[i][j][a][b] = ElementT("STANDARD", Dbb, Vbb);
                    i -= nactv_;
                    j -= nactv_;
                }
            }
        }
    }
    for (size_t i = 0; i < ncore_; ++i) {
        size_t ni = core_mos_[i];
        for (size_t j = 0; j < ncore_; ++j) {
            size_t nj = core_mos_[j];
            for (size_t a = 0; a < nvirt_; ++a) {
                size_t na = virt_mos_[a];
                for (size_t b = 0; b < nvirt_; ++b) {
                    size_t nb = virt_mos_[b];

                    double Daa = Fa_[ni][ni] + Fa_[nj][nj] - Fa_[na][na] - Fa_[nb][nb];
                    double Dab = Fa_[ni][ni] + Fb_[nj][nj] - Fa_[na][na] - Fb_[nb][nb];
                    double Dbb = Fb_[ni][ni] + Fb_[nj][nj] - Fb_[na][na] - Fb_[nb][nb];

                    double Vaa = integral_->aptei_aa(na, nb, ni, nj);
                    double Vab = integral_->aptei_ab(na, nb, ni, nj);
                    double Vbb = integral_->aptei_bb(na, nb, ni, nj);

                    i += nactv_;
                    j += nactv_;
                    a += nactv_;
                    b += nactv_;
                    AA[i][j][a][b] = ElementT("STANDARD", Daa, Vaa);
                    AB[i][j][a][b] = ElementT("STANDARD", Dab, Vab);
                    BB[i][j][a][b] = ElementT("STANDARD", Dbb, Vbb);
                    i -= nactv_;
                    j -= nactv_;
                    a -= nactv_;
                    b -= nactv_;
                }
            }
        }
    }
    for (size_t i = 0; i < nactv_; ++i) {
        size_t ni = actv_mos_[i];
        for (size_t j = 0; j < nactv_; ++j) {
            size_t nj = actv_mos_[j];
            for (size_t a = 0; a < nvirt_; ++a) {
                size_t na = virt_mos_[a];
                for (size_t b = 0; b < nvirt_; ++b) {
                    size_t nb = virt_mos_[b];

                    double Daa = Fa_[ni][ni] + Fa_[nj][nj] - Fa_[na][na] - Fa_[nb][nb];
                    double Dab = Fa_[ni][ni] + Fb_[nj][nj] - Fa_[na][na] - Fb_[nb][nb];
                    double Dbb = Fb_[ni][ni] + Fb_[nj][nj] - Fb_[na][na] - Fb_[nb][nb];

                    double Vaa = integral_->aptei_aa(na, nb, ni, nj);
                    double Vab = integral_->aptei_ab(na, nb, ni, nj);
                    double Vbb = integral_->aptei_bb(na, nb, ni, nj);

                    a += nactv_;
                    b += nactv_;
                    AA[i][j][a][b] = ElementT("STANDARD", Daa, Vaa);
                    AB[i][j][a][b] = ElementT("STANDARD", Dab, Vab);
                    BB[i][j][a][b] = ElementT("STANDARD", Dbb, Vbb);
                    a -= nactv_;
                    b -= nactv_;
                }
            }
        }
    }
    timer_off("Form T2");
}

void MCSRGPT2_MO::Form_T2_ISA(d4& AA, d4& AB, d4& BB, const double& b_const) {
    timer_on("Form T2");
    for (size_t i = 0; i < nhole_; ++i) {
        size_t ni = hole_mos_[i];
        for (size_t j = 0; j < nhole_; ++j) {
            size_t nj = hole_mos_[j];
            for (size_t a = 0; a < npart_; ++a) {
                size_t na = part_mos_[a];
                for (size_t b = 0; b < npart_; ++b) {
                    size_t nb = part_mos_[b];

                    double Daa = Fa_[ni][ni] + Fa_[nj][nj] - Fa_[na][na] - Fa_[nb][nb];
                    double Dab = Fa_[ni][ni] + Fb_[nj][nj] - Fa_[na][na] - Fb_[nb][nb];
                    double Dbb = Fb_[ni][ni] + Fb_[nj][nj] - Fb_[na][na] - Fb_[nb][nb];

                    double scalar_aa = integral_->aptei_aa(na, nb, ni, nj);
                    double scalar_ab = integral_->aptei_ab(na, nb, ni, nj);
                    double scalar_bb = integral_->aptei_bb(na, nb, ni, nj);

                    AA[i][j][a][b] = scalar_aa / (Daa + b_const / Daa);
                    AB[i][j][a][b] = scalar_ab / (Dab + b_const / Dab);
                    BB[i][j][a][b] = scalar_bb / (Dbb + b_const / Dbb);
                }
            }
        }
    }

    // Zero Internal Excitations
    for (size_t i = 0; i < nactv_; ++i) {
        for (size_t j = 0; j < nactv_; ++j) {
            for (size_t k = 0; k < nactv_; ++k) {
                for (size_t l = 0; l < nactv_; ++l) {
                    AA[i][j][k][l] = 0.0;
                    AB[i][j][k][l] = 0.0;
                    BB[i][j][k][l] = 0.0;
                }
            }
        }
    }
    timer_off("Form T2");
}

void MCSRGPT2_MO::Form_T1_ISA(d2& A, d2& B, const double& b_const) {
    timer_on("Form T1");
    for (size_t i = 0; i < nhole_; ++i) {
        size_t ni = hole_mos_[i];
        for (size_t a = 0; a < npart_; ++a) {
            size_t na = part_mos_[a];

            double scalar_a = Fa_[ni][na];
            double scalar_b = Fb_[ni][na];

            for (size_t u = 0; u < nactv_; ++u) {
                size_t nu = actv_mos_[u];
                for (size_t x = 0; x < nactv_; ++x) {
                    size_t nx = actv_mos_[x];

                    scalar_a += (Fa_[nx][nx] - Fa_[nu][nu]) * T2aa_[i][u][a][x] * Da_[nx][nu];
                    scalar_a += (Fb_[nx][nx] - Fb_[nu][nu]) * T2ab_[i][u][a][x] * Db_[nx][nu];
                    scalar_b += (Fa_[nx][nx] - Fa_[nu][nu]) * T2ab_[u][i][x][a] * Da_[nx][nu];
                    scalar_b += (Fb_[nx][nx] - Fb_[nu][nu]) * T2bb_[i][u][a][x] * Db_[nx][nu];
                }
            }

            double delta_a = Fa_[ni][ni] - Fa_[na][na];
            double delta_b = Fb_[ni][ni] - Fb_[na][na];

            A[i][a] = scalar_a / (delta_a + b_const / delta_a);
            B[i][a] = scalar_b / (delta_b + b_const / delta_b);
        }
    }

    // Zero Internal Excitations
    for (size_t i = 0; i < nactv_; ++i) {
        for (size_t j = 0; j < nactv_; ++j) {
            A[i][j] = 0.0;
            B[i][j] = 0.0;
        }
    }
    timer_off("Form T1");
}

inline bool ReverseSortT2(const std::tuple<double, size_t, size_t, size_t, size_t>& lhs,
                          const std::tuple<double, size_t, size_t, size_t, size_t>& rhs) {
    return std::fabs(std::get<0>(rhs)) < std::fabs(std::get<0>(lhs));
}

void MCSRGPT2_MO::Check_T2(const std::string& x, const d4& M, double& Norm, double& MaxT,
                           std::shared_ptr<ForteOptions> options) {
    timer_on("Check T2");
    size_t ntamp = options->get_int("NTAMP");
    double intruder = options->get_double("INTRUDER_TAMP");
    std::vector<std::tuple<double, size_t, size_t, size_t, size_t>> Max;
    std::vector<std::tuple<double, size_t, size_t, size_t, size_t>> Large(
        ntamp, std::make_tuple(0.0, 0, 0, 0, 0));
    double value = 0.0;
    size_t count = 0;
    for (size_t i = 0; i < nhole_; ++i) {
        size_t ni = hole_mos_[i];
        for (size_t j = 0; j < nhole_; ++j) {
            size_t nj = hole_mos_[j];
            for (size_t a = 0; a < npart_; ++a) {
                size_t na = part_mos_[a];
                for (size_t b = 0; b < npart_; ++b) {
                    size_t nb = part_mos_[b];
                    double m = M[i][j][a][b];
                    value += pow(m, 2.0);
                    if (std::fabs(m) > std::fabs(std::get<0>(Large[ntamp - 1]))) {
                        Large[ntamp - 1] = std::make_tuple(m, ni, nj, na, nb);
                    }
                    sort(Large.begin(), Large.end(), ReverseSortT2);
                    if (std::fabs(m) > intruder)
                        Max.push_back(std::make_tuple(m, ni, nj, na, nb));
                    sort(Max.begin(), Max.end(), ReverseSortT2);
                    if (std::fabs(m) > options->get_double("E_CONVERGENCE"))
                        ++count;
                }
            }
        }
    }
    Norm = sqrt(value);
    MaxT = std::get<0>(Large[0]);

    // Print
    outfile->Printf("\n");
    outfile->Printf("\n  ==> Largest T2 amplitudes for spin case %s: <==", x.c_str());
    if (x == "AA")
        outfile->Printf("\n");
    if (x == "AB")
        outfile->Printf("\n   %3c %3c %3c %3c %9c  %3c %3c %3c %3c %9c  %3c "
                        "%3c %3c %3c %9c",
                        ' ', '_', ' ', '_', ' ', ' ', '_', ' ', '_', ' ', ' ', '_', ' ', '_', ' ');
    if (x == "BB")
        outfile->Printf("\n   %3c %3c %3c %3c %9c  %3c %3c %3c %3c %9c  %3c "
                        "%3c %3c %3c %9c",
                        '_', '_', '_', '_', ' ', '_', '_', '_', '_', ' ', '_', '_', '_', '_', ' ');
    outfile->Printf("\n   %3c %3c %3c %3c %9c  %3c %3c %3c %3c %9c  %3c %3c %3c %3c %9c", 'i', 'j',
                    'a', 'b', ' ', 'i', 'j', 'a', 'b', ' ', 'i', 'j', 'a', 'b', ' ');
    outfile->Printf("\n  "
                    "----------------------------------------------------------"
                    "----------------------");
    for (size_t n = 0; n != ntamp; ++n) {
        if (n % 3 == 0)
            outfile->Printf("\n  ");
        outfile->Printf("[%3zu %3zu %3zu %3zu] %8.5f ", std::get<1>(Large[n]),
                        std::get<2>(Large[n]), std::get<3>(Large[n]), std::get<4>(Large[n]),
                        std::get<0>(Large[n]));
    }
    outfile->Printf("\n  "
                    "----------------------------------------------------------"
                    "----------------------");
    outfile->Printf("\n  Norm of T2%s vector: (nonzero elements: %12zu) %25.15lf.", x.c_str(),
                    count, Norm);
    outfile->Printf("\n  "
                    "----------------------------------------------------------"
                    "----------------------");
    outfile->Printf("\n");
    outfile->Printf("\n  ==> T2 intruder states analysis for spin case %s: <==", x.c_str());
    outfile->Printf("\n  "
                    "----------------------------------------------------------"
                    "-------------------------------");
    outfile->Printf("\n      Amplitude        Value   Numerator                "
                    "   Denominator");
    outfile->Printf("\n  "
                    "----------------------------------------------------------"
                    "-------------------------------");
    for (size_t n = 0; n != Max.size(); ++n) {
        size_t i = std::get<1>(Max[n]);
        size_t j = std::get<2>(Max[n]);
        size_t a = std::get<3>(Max[n]);
        size_t b = std::get<4>(Max[n]);
        double t2 = std::get<0>(Max[n]);
        double fi = (x != "BB") ? (Fa_[i][i]) : (Fb_[i][i]);
        double fj = (x == "AA") ? (Fa_[j][j]) : (Fb_[j][j]);
        double fa = (x != "BB") ? (Fa_[a][a]) : (Fb_[a][a]);
        double fb = (x == "AA") ? (Fa_[b][b]) : (Fb_[b][b]);
        double down = fi + fj - fa - fb;
        double up = t2 * down;
        outfile->Printf("\n  [%3zu %3zu %3zu %3zu] = %7.4f = %7.4f / (%7.4f + "
                        "%7.4f - %7.4f - %7.4f = %7.4f)",
                        i, j, a, b, t2, up, fi, fj, fa, fb, down);
    }
    outfile->Printf("\n  "
                    "----------------------------------------------------------"
                    "-------------------------------");
    timer_off("Check T2");
}

inline bool ReverseSortT1(const std::tuple<double, size_t, size_t>& lhs,
                          const std::tuple<double, size_t, size_t>& rhs) {
    return std::fabs(std::get<0>(rhs)) < std::fabs(std::get<0>(lhs));
}

void MCSRGPT2_MO::Check_T1(const std::string& x, const d2& M, double& Norm, double& MaxT,
                           std::shared_ptr<ForteOptions> options) {
    timer_on("Check T1");
    size_t ntamp = options->get_int("NTAMP");
    double intruder = options->get_double("INTRUDER_TAMP");
    std::vector<std::tuple<double, size_t, size_t>> Max;
    std::vector<std::tuple<double, size_t, size_t>> Large(ntamp, std::make_tuple(0.0, 0, 0));
    double value = 0.0;
    size_t count = 0;
    for (size_t i = 0; i < nhole_; ++i) {
        size_t ni = hole_mos_[i];
        for (size_t a = 0; a < npart_; ++a) {
            size_t na = part_mos_[a];
            double m = M[i][a];
            value += pow(m, 2.0);
            if (std::fabs(m) > std::fabs(std::get<0>(Large[ntamp - 1]))) {
                Large[ntamp - 1] = std::make_tuple(m, ni, na);
            }
            sort(Large.begin(), Large.end(), ReverseSortT1);
            if (std::fabs(m) > intruder)
                Max.push_back(std::make_tuple(m, ni, na));
            sort(Max.begin(), Max.end(), ReverseSortT1);
            if (std::fabs(m) > options->get_double("E_CONVERGENCE"))
                ++count;
        }
    }
    Norm = sqrt(value);
    MaxT = std::get<0>(Large[0]);

    // Print
    outfile->Printf("\n");
    outfile->Printf("\n  ==> Largest T1 amplitudes for spin case %s: <==", x.c_str());
    if (x == "A")
        outfile->Printf("\n");
    if (x == "B")
        outfile->Printf("\n   %3c %3c %3c %3c %9c  %3c %3c %3c %3c %9c  %3c "
                        "%3c %3c %3c %9c",
                        '_', ' ', '_', ' ', ' ', '_', ' ', '_', ' ', ' ', '_', ' ', '_', ' ', ' ');
    outfile->Printf("\n   %3c %3c %3c %3c %9c  %3c %3c %3c %3c %9c  %3c %3c %3c %3c %9c", 'i', ' ',
                    'a', ' ', ' ', 'i', ' ', 'a', ' ', ' ', 'i', ' ', 'a', ' ', ' ');
    outfile->Printf("\n  "
                    "----------------------------------------------------------"
                    "----------------------");
    for (size_t n = 0; n != ntamp; ++n) {
        if (n % 3 == 0)
            outfile->Printf("\n  ");
        outfile->Printf("[%3zu %3c %3zu %3c] %8.5f ", std::get<1>(Large[n]), ' ',
                        std::get<2>(Large[n]), ' ', std::get<0>(Large[n]));
    }
    outfile->Printf("\n  "
                    "----------------------------------------------------------"
                    "----------------------");
    outfile->Printf("\n  Norm of T1%s vector: (nonzero elements: %12zu) %26.15lf.", x.c_str(),
                    count, Norm);
    outfile->Printf("\n  "
                    "----------------------------------------------------------"
                    "----------------------");
    outfile->Printf("\n");
    outfile->Printf("\n  ==> T1 intruder states analysis for spin case %s: <==", x.c_str());
    outfile->Printf("\n  "
                    "----------------------------------------------------------"
                    "-----------");
    outfile->Printf("\n      Amplitude        Value   Numerator          Denominator");
    outfile->Printf("\n  "
                    "----------------------------------------------------------"
                    "-----------");
    for (size_t n = 0; n != Max.size(); ++n) {
        size_t i = std::get<1>(Max[n]);
        size_t a = std::get<2>(Max[n]);
        double t2 = std::get<0>(Max[n]);
        double fi = (x == "A") ? (Fa_[i][i]) : (Fb_[i][i]);
        double fa = (x == "A") ? (Fa_[a][a]) : (Fb_[a][a]);
        double down = fi - fa;
        double up = t2 * down;
        outfile->Printf("\n  [%3zu %3c %3zu %3c] = %7.4f = %7.4f / (%7.4f - %7.4f = %7.4f)", i, ' ',
                        a, ' ', t2, up, fi, fa, down);
    }
    outfile->Printf("\n  "
                    "----------------------------------------------------------"
                    "-----------");
    timer_off("Check T1");
}

void MCSRGPT2_MO::test_D1_RE() {
    double small_threshold = 0.1;
    std::vector<std::pair<std::vector<size_t>, double>> smallD1;

    for (size_t i = 0; i < nhole_; ++i) {
        size_t ni = hole_mos_[i];
        for (size_t a = 0; a < npart_; ++a) {
            size_t na = part_mos_[a];

            // must belong to the same irrep
            if ((sym_ncmo_[ni] ^ sym_ncmo_[na]) != 0)
                continue;

            // cannot be all active
            if (std::find(actv_mos_.begin(), actv_mos_.end(), ni) != actv_mos_.end() &&
                std::find(actv_mos_.begin(), actv_mos_.end(), na) != actv_mos_.end()) {
                continue;
            } else {
                double Da = Fa_[ni][ni] - Fa_[na][na];

                for (size_t k = 0; k < nhole_; ++k) {
                    size_t nk = hole_mos_[k];
                    Da -= integral_->aptei_aa(ni, na, na, nk) * Da_[nk][ni];
                }

                for (size_t v = 0; v < nactv_; ++v) {
                    size_t nv = actv_mos_[v];
                    Da += integral_->aptei_aa(ni, nv, na, ni) * Da_[na][nv];
                }

                if (std::fabs(Da) < small_threshold) {
                    smallD1.push_back(std::make_pair(std::vector<size_t>{ni, na}, Da));
                }
            }
        }
    }

    //    // core-virtual block
    //    for(size_t n = 0; n < nc_; ++n){
    //        size_t nn = idx_c_[n];
    //        for(size_t f = 0; f < nv_; ++f){
    //            size_t nf = idx_v_[f];
    //            if((sym_ncmo_[nn] ^ sym_ncmo_[nf]) != 0) continue;

    //            double Da = Fa_[nn][nn] - Fa_[nf][nf];
    //            Da -= integral_->aptei_aa(nn, nf, nf, nn);

    //            if(std::fabs(Da) < small_threshold){
    //                smallD1.push_back(std::make_pair(std::vector<size_t>
    //                {nn,nf}, Da));
    //            }
    //        }
    //    }

    //    // core-active block
    //    for(size_t n = 0; n < nc_; ++n){
    //        size_t nn = idx_c_[n];
    //        for(size_t v = 0; v < na_; ++v){
    //            size_t nv = idx_a_[v];
    //            if((sym_ncmo_[nn] ^ sym_ncmo_[nv]) != 0) continue;

    //            double Da = Fa_[nn][nn] - Fa_[nv][nv];
    //            Da -= integral_->aptei_aa(nn, nv, nv, nn);
    //            for(size_t y = 0; y < na_; ++y){
    //                size_t ny = idx_a_[y];
    //                Da += Da_[nv][ny] * integral_->aptei_aa(nn, ny, nv, nn);
    //            }

    //            if(std::fabs(Da) < small_threshold){
    //                smallD1.push_back(std::make_pair(std::vector<size_t>
    //                {nn,nv}, Da));
    //            }
    //        }
    //    }

    //    // active-virtual block
    //    for(size_t x = 0; x < na_; ++x){
    //        size_t nx = idx_a_[x];
    //        for(size_t f = 0; f < nv_; ++f){
    //            size_t nf = idx_v_[f];
    //            if((sym_ncmo_[nx] ^ sym_ncmo_[nf]) != 0) continue;

    //            double Da = Fa_[nx][nx] - Fa_[nf][nf];
    //            for(size_t u = 0; u < na_; ++u){
    //                size_t nu = idx_a_[u];
    //                Da -= Da_[nu][nx] * integral_->aptei_aa(nx, nf, nf, nu);
    //            }

    //            if(std::fabs(Da) < small_threshold){
    //                smallD1.push_back(std::make_pair(std::vector<size_t>
    //                {nx,nf}, Da));
    //            }
    //        }
    //    }

    // print
    print_h2("Small Denominators for T1 with RE Partitioning");
    if (smallD1.size() == 0) {
        outfile->Printf("\n    NULL.");
    } else {
        std::string indent(4, ' ');
        std::string dash(47, '-');
        std::string title = indent +
                            str(boost::format("%=9s    %=15s    %=15s\n") % "Indices" %
                                "Denominator" % "Original Denom.") +
                            indent + dash;
        outfile->Printf("\n%s", title.c_str());
        for (const auto& pair : smallD1) {
            size_t i = pair.first[0];
            size_t j = pair.first[1];
            double D = pair.second;
            double Dold = Fa_[i][i] - Fa_[j][j];
            outfile->Printf("\n    %4zu %4zu    %15.12f    %15.12f", i, j, D, Dold);
        }
        outfile->Printf("\n    %s", dash.c_str());
    }
}

void MCSRGPT2_MO::test_D2_RE() {
    double small_threshold = 0.5;
    std::vector<std::pair<std::vector<size_t>, double>> smallD2aa;
    std::vector<std::pair<std::vector<size_t>, double>> smallD2ab;

    for (size_t i = 0; i < nhole_; ++i) {
        size_t ni = hole_mos_[i];
        for (size_t j = i; j < nhole_; ++j) {
            size_t nj = hole_mos_[j];
            for (size_t a = 0; a < npart_; ++a) {
                size_t na = part_mos_[a];
                for (size_t b = a; b < npart_; ++b) {
                    size_t nb = part_mos_[b];

                    // product must be all symmetric
                    if ((sym_ncmo_[ni] ^ sym_ncmo_[nj] ^ sym_ncmo_[na] ^ sym_ncmo_[nb]) != 0)
                        continue;

                    // cannot be all active
                    if (std::find(actv_mos_.begin(), actv_mos_.end(), ni) != actv_mos_.end() &&
                        std::find(actv_mos_.begin(), actv_mos_.end(), nj) != actv_mos_.end() &&
                        std::find(actv_mos_.begin(), actv_mos_.end(), na) != actv_mos_.end() &&
                        std::find(actv_mos_.begin(), actv_mos_.end(), nb) != actv_mos_.end()) {
                        continue;
                    } else {
                        double Daa = Fa_[ni][ni] + Fa_[nj][nj] - Fa_[na][na] - Fa_[nb][nb];
                        double Dab = Fa_[ni][ni] + Fb_[nj][nj] - Fa_[na][na] - Fb_[nb][nb];

                        for (size_t c = 0; c < npart_; ++c) {
                            size_t nc = part_mos_[c];
                            for (size_t d = 0; d < npart_; ++d) {
                                size_t nd = part_mos_[d];
                                Daa -= 0.5 * integral_->aptei_aa(nc, nd, na, nb) *
                                       (1.0 - Da_[na][nc]) * (1.0 - Da_[nb][nd]);
                                Dab -= integral_->aptei_ab(nc, nd, na, nb) * (1.0 - Da_[na][nc]) *
                                       (1.0 - Db_[nb][nd]);
                            }
                        }

                        for (size_t v = 0; v < nactv_; ++v) {
                            size_t nv = actv_mos_[v];
                            for (size_t y = 0; y < nactv_; ++y) {
                                size_t ny = actv_mos_[y];
                                Daa += 0.5 * integral_->aptei_aa(nv, ny, na, nb) * Da_[na][nv] *
                                       Da_[nb][ny];
                                Dab +=
                                    integral_->aptei_ab(nv, ny, na, nb) * Da_[na][nv] * Db_[nb][ny];
                            }
                        }

                        for (size_t k = 0; k < nhole_; ++k) {
                            size_t nk = hole_mos_[k];
                            for (size_t l = 0; l < nhole_; ++l) {
                                size_t nl = hole_mos_[l];
                                Daa -= 0.5 * integral_->aptei_aa(ni, nj, nk, nl) * Da_[nk][ni] *
                                       Da_[nl][nj];
                                Dab -=
                                    integral_->aptei_ab(ni, nj, nk, nl) * Da_[nk][ni] * Db_[nl][nj];
                            }
                        }

                        for (size_t u = 0; u < nactv_; ++u) {
                            size_t nu = actv_mos_[u];
                            for (size_t x = 0; x < nactv_; ++x) {
                                size_t nx = actv_mos_[x];
                                Daa += 0.5 * integral_->aptei_aa(ni, nj, nu, nx) *
                                       (1.0 - Da_[nu][ni]) * (1.0 - Da_[nx][nj]);
                                Dab += integral_->aptei_ab(ni, nj, nu, nx) * (1.0 - Da_[nu][ni]) *
                                       (1.0 - Db_[nx][nj]);
                            }
                        }

                        for (size_t k = 0; k < nhole_; ++k) {
                            size_t nk = hole_mos_[k];

                            Daa -= integral_->aptei_aa(na, ni, nk, na) * Da_[nk][ni];
                            Daa -= integral_->aptei_aa(nb, ni, nk, nb) * Da_[nk][ni];
                            Daa -= integral_->aptei_aa(na, nj, nk, na) * Da_[nk][nj];
                            Daa -= integral_->aptei_aa(nb, nj, nk, nb) * Da_[nk][nj];

                            Dab -= integral_->aptei_aa(na, ni, nk, na) * Da_[nk][ni];
                            Dab += integral_->aptei_ab(ni, nb, nk, nb) * Da_[nk][ni];
                            Dab += integral_->aptei_ab(na, nj, na, nk) * Db_[nk][nj];
                            Dab -= integral_->aptei_bb(nb, nj, nk, nb) * Da_[nk][nj];
                        }

                        for (size_t y = 0; y < nactv_; ++y) {
                            size_t ny = actv_mos_[y];

                            Daa += integral_->aptei_aa(ny, ni, ni, na) * Da_[na][ny];
                            Daa += integral_->aptei_aa(ny, ni, ni, nb) * Da_[nb][ny];
                            Daa += integral_->aptei_aa(ny, nj, nj, na) * Da_[na][ny];
                            Daa += integral_->aptei_aa(ny, nj, nj, nb) * Da_[nb][ny];

                            Dab -= integral_->aptei_aa(ny, ni, ni, na) * Da_[na][ny];
                            Dab += integral_->aptei_ab(ni, ny, ni, nb) * Db_[nb][ny];
                            Dab += integral_->aptei_ab(ny, nj, na, nj) * Da_[na][ny];
                            Dab -= integral_->aptei_bb(ny, nj, nj, nb) * Db_[nb][ny];
                        }

                        if (std::fabs(Daa) < small_threshold && ni != nj && na != nb) {
                            smallD2aa.push_back(
                                std::make_pair(std::vector<size_t>{ni, nj, na, nb}, Daa));
                        }
                        if (std::fabs(Dab) < small_threshold) {
                            smallD2ab.push_back(
                                std::make_pair(std::vector<size_t>{ni, nj, na, nb}, Dab));
                        }
                    }
                }
            }
        }
    }

    // print
    print_h2("Small Denominators for T2aa with RE Partitioning");
    if (smallD2aa.size() == 0) {
        outfile->Printf("\n    NULL.");
    } else {
        std::string indent(4, ' ');
        std::string dash(57, '-');
        std::string title = indent +
                            str(boost::format("%=19s    %=15s    %=15s\n") % "Indices" %
                                "Denominator" % "Original Denom.") +
                            indent + dash;
        outfile->Printf("\n%s", title.c_str());
        for (const auto& pair : smallD2aa) {
            size_t i = pair.first[0];
            size_t j = pair.first[1];
            size_t k = pair.first[2];
            size_t l = pair.first[3];
            double D = pair.second;
            double Dold = Fa_[i][i] + Fa_[j][j] - Fa_[k][k] - Fa_[l][l];
            outfile->Printf("\n    %4zu %4zu %4zu %4zu    %15.12f    %15.12f", i, j, k, l, D, Dold);
        }
        outfile->Printf("\n    %s", dash.c_str());
    }

    print_h2("Small Denominators for T2ab with RE Partitioning");
    if (smallD2ab.size() == 0) {
        outfile->Printf("\n    NULL.");
    } else {
        std::string indent(4, ' ');
        std::string dash(57, '-');
        std::string title = indent +
                            str(boost::format("%=19s    %=15s    %=15s\n") % "Indices" %
                                "Denominator" % "Original Denom.") +
                            indent + dash;
        outfile->Printf("\n%s", title.c_str());
        for (const auto& pair : smallD2ab) {
            size_t i = pair.first[0];
            size_t j = pair.first[1];
            size_t k = pair.first[2];
            size_t l = pair.first[3];
            double D = pair.second;
            double Dold = Fa_[i][i] + Fb_[j][j] - Fa_[k][k] - Fb_[l][l];
            outfile->Printf("\n    %4zu %4zu %4zu %4zu    %15.12f    %15.12f", i, j, k, l, D, Dold);
        }
        outfile->Printf("\n    %s", dash.c_str());
    }
}

void MCSRGPT2_MO::test_D2_Dyall() {
    double small_threshold = 0.1;
    std::vector<std::pair<std::vector<size_t>, double>> smallD2aa;
    std::vector<std::pair<std::vector<size_t>, double>> smallD2ab;

    // core-core-active-active block
    for (size_t m = 0; m < ncore_; ++m) {
        size_t nm = core_mos_[m];
        for (size_t n = 0; n < ncore_; ++n) {
            size_t nn = core_mos_[n];
            for (size_t u = 0; u < nactv_; ++u) {
                size_t nu = actv_mos_[u];
                for (size_t v = 0; v < nactv_; ++v) {
                    size_t nv = actv_mos_[v];
                    if ((sym_ncmo_[nm] ^ sym_ncmo_[nn] ^ sym_ncmo_[nu] ^ sym_ncmo_[nv]) != 0)
                        continue;

                    double Daa = Fa_[nm][nm] + Fa_[nn][nn] - Fa_[nu][nu] - Fa_[nv][nv];
                    double Dab = Fa_[nm][nm] + Fb_[nn][nn] - Fa_[nu][nu] - Fb_[nv][nv];

                    Daa -= 0.5 * integral_->aptei_aa(nu, nv, nu, nv);
                    Dab -= 0.5 * integral_->aptei_ab(nu, nv, nu, nv);

                    for (size_t x = 0; x < nactv_; ++x) {
                        size_t nx = actv_mos_[x];
                        Daa += Da_[nu][nx] * integral_->aptei_aa(nx, nv, nu, nv);
                        Dab += Da_[nu][nx] * integral_->aptei_ab(nx, nv, nu, nv);
                    }

                    if (std::fabs(Daa) < small_threshold) {
                        smallD2aa.push_back(
                            std::make_pair(std::vector<size_t>{nm, nn, nu, nv}, Daa));
                    }
                    if (std::fabs(Dab) < small_threshold) {
                        smallD2ab.push_back(
                            std::make_pair(std::vector<size_t>{nm, nn, nu, nv}, Dab));
                    }
                }
            }
        }
    }

    // active-active-virtual-virtual block
    for (size_t x = 0; x < nactv_; ++x) {
        size_t nx = actv_mos_[x];
        for (size_t y = 0; y < nactv_; ++y) {
            size_t ny = actv_mos_[y];
            for (size_t e = 0; e < nvirt_; ++e) {
                size_t ne = virt_mos_[e];
                for (size_t f = 0; f < nvirt_; ++f) {
                    size_t nf = virt_mos_[f];
                    if ((sym_ncmo_[nx] ^ sym_ncmo_[ny] ^ sym_ncmo_[ne] ^ sym_ncmo_[nf]) != 0)
                        continue;

                    double Daa = Fa_[nx][nx] + Fa_[ny][ny] - Fa_[ne][ne] - Fa_[nf][nf];
                    double Dab = Fa_[nx][nx] + Fb_[ny][ny] - Fa_[ne][ne] - Fb_[nf][nf];

                    Daa += 0.5 * integral_->aptei_aa(nx, ny, nx, ny);
                    Dab += 0.5 * integral_->aptei_ab(nx, ny, nx, ny);

                    for (size_t u = 0; u < nactv_; ++u) {
                        size_t nu = actv_mos_[u];
                        Daa -= Da_[nu][nx] * integral_->aptei_aa(nx, ny, nu, ny);
                        Dab -= Da_[nu][nx] * integral_->aptei_ab(nx, ny, nu, ny);
                    }

                    if (std::fabs(Daa) < small_threshold) {
                        smallD2aa.push_back(
                            std::make_pair(std::vector<size_t>{nx, ny, ne, nf}, Daa));
                    }
                    if (std::fabs(Dab) < small_threshold) {
                        smallD2ab.push_back(
                            std::make_pair(std::vector<size_t>{nx, ny, ne, nf}, Dab));
                    }
                }
            }
        }
    }

    // active-core-active-virtual block
    for (size_t m = 0; m < ncore_; ++m) {
        size_t nm = core_mos_[m];
        for (size_t y = 0; y < nactv_; ++y) {
            size_t ny = actv_mos_[y];
            for (size_t e = 0; e < nvirt_; ++e) {
                size_t ne = virt_mos_[e];
                for (size_t v = 0; v < nactv_; ++v) {
                    size_t nv = actv_mos_[v];
                    if ((sym_ncmo_[nm] ^ sym_ncmo_[ny] ^ sym_ncmo_[ne] ^ sym_ncmo_[nv]) != 0)
                        continue;

                    double Daa = Fa_[nm][nm] + Fa_[ny][ny] - Fa_[ne][ne] - Fa_[nv][nv];
                    double D1 = Fa_[nm][nm] + Fb_[ny][ny] - Fa_[ne][ne] - Fb_[nv][nv];
                    double D2 = Fa_[nm][nm] + Fb_[ny][ny] - Fb_[ne][ne] - Fa_[nv][nv];
                    double D3 = Fb_[nm][nm] + Fa_[ny][ny] - Fa_[ne][ne] - Fb_[nv][nv];

                    for (size_t u = 0; u < nactv_; ++u) {
                        size_t nu = actv_mos_[u];
                        Daa += Da_[nu][ny] * integral_->aptei_aa(ny, nv, nu, nv);
                        Daa -= Da_[nv][nu] * integral_->aptei_aa(nu, ny, nv, ny);

                        D1 += Db_[nu][ny] * integral_->aptei_bb(ny, nv, nu, nv);
                        D1 -= Db_[nv][nu] * integral_->aptei_bb(nu, ny, nv, ny);

                        D2 += Db_[nu][ny] * integral_->aptei_ab(nv, ny, nv, nu);
                        D2 -= Da_[nv][nu] * integral_->aptei_ab(nu, ny, nv, ny);

                        D3 += Da_[nu][ny] * integral_->aptei_ab(ny, nv, nu, nv);
                        D3 -= Db_[nv][nu] * integral_->aptei_ab(ny, nu, ny, nv);
                    }

                    if (std::fabs(Daa) < small_threshold) {
                        smallD2aa.push_back(
                            std::make_pair(std::vector<size_t>{nm, ny, ne, nv}, Daa));
                    }
                    if (std::fabs(D1) < small_threshold) {
                        smallD2ab.push_back(
                            std::make_pair(std::vector<size_t>{nm, ny, ne, nv}, D1));
                    }
                    if (std::fabs(D2) < small_threshold) {
                        smallD2ab.push_back(
                            std::make_pair(std::vector<size_t>{nm, ny, nv, ne}, D2));
                    }
                    if (std::fabs(D3) < small_threshold) {
                        smallD2ab.push_back(
                            std::make_pair(std::vector<size_t>{ny, nm, ne, nv}, D3));
                    }
                }
            }
        }
    }

    // active-active-active-virtual block
    for (size_t x = 0; x < nactv_; ++x) {
        size_t nx = actv_mos_[x];
        for (size_t y = 0; y < nactv_; ++y) {
            size_t ny = actv_mos_[y];
            for (size_t z = 0; z < nactv_; ++z) {
                size_t nz = actv_mos_[z];
                for (size_t e = 0; e < nvirt_; ++e) {
                    size_t ne = virt_mos_[e];
                    if ((sym_ncmo_[nx] ^ sym_ncmo_[ny] ^ sym_ncmo_[nz] ^ sym_ncmo_[ne]) != 0)
                        continue;

                    double Daa = Fa_[nx][nx] + Fa_[ny][ny] - Fa_[nz][nz] - Fa_[ne][ne];
                    double Dab = Fa_[nx][nx] + Fb_[ny][ny] - Fa_[nz][nz] - Fb_[ne][ne];

                    Daa += 0.5 * integral_->aptei_aa(nx, ny, nx, ny);
                    Dab += 0.5 * integral_->aptei_ab(nx, ny, nx, ny);

                    for (size_t u = 0; u < nactv_; ++u) {
                        size_t nu = actv_mos_[u];

                        Daa -= Da_[nu][nx] * integral_->aptei_aa(nx, ny, nu, ny);
                        Daa += Da_[nu][nx] * integral_->aptei_aa(nx, nz, nu, nz);
                        Daa -= Da_[nz][nu] * integral_->aptei_aa(nx, nu, nx, nz);
                        Daa += Da_[nu][ny] * integral_->aptei_aa(ny, nz, nu, nz);
                        Daa -= Da_[nz][nu] * integral_->aptei_aa(ny, nu, ny, nz);

                        Dab -= Da_[nu][nx] * integral_->aptei_ab(nx, ny, nu, ny);
                        Dab += Da_[nu][nx] * integral_->aptei_aa(nx, nz, nu, nz);
                        Dab -= Da_[nz][nu] * integral_->aptei_aa(nx, nu, nx, nz);
                        Dab += Db_[nu][ny] * integral_->aptei_ab(nz, ny, nz, nu);
                        Dab -= Da_[nz][nu] * integral_->aptei_ab(nu, ny, nz, ny);
                    }

                    if (std::fabs(Daa) < small_threshold) {
                        smallD2aa.push_back(
                            std::make_pair(std::vector<size_t>{nx, ny, nz, ne}, Daa));
                    }
                    if (std::fabs(Dab) < small_threshold) {
                        smallD2ab.push_back(
                            std::make_pair(std::vector<size_t>{nx, ny, nz, ne}, Dab));
                    }
                }
            }
        }
    }

    // core-active-active-active block
    for (size_t m = 0; m < ncore_; ++m) {
        size_t nm = core_mos_[m];
        for (size_t w = 0; w < nactv_; ++w) {
            size_t nw = actv_mos_[w];
            for (size_t u = 0; u < nactv_; ++u) {
                size_t nu = actv_mos_[u];
                for (size_t v = 0; v < nactv_; ++v) {
                    size_t nv = actv_mos_[v];
                    if ((sym_ncmo_[nm] ^ sym_ncmo_[nw] ^ sym_ncmo_[nu] ^ sym_ncmo_[nv]) != 0)
                        continue;

                    double Daa = Fa_[nm][nm] + Fa_[nw][nw] - Fa_[nu][nu] - Fa_[nv][nv];
                    double Dab = Fa_[nm][nm] + Fb_[nw][nw] - Fa_[nu][nu] - Fb_[nv][nv];

                    Daa -= 0.5 * integral_->aptei_aa(nu, nv, nu, nv);
                    Dab -= 0.5 * integral_->aptei_ab(nu, nv, nu, nv);

                    for (size_t x = 0; x < nactv_; ++x) {
                        size_t nx = actv_mos_[x];

                        Daa += Da_[nu][nx] * integral_->aptei_aa(nx, nv, nu, nv);
                        Daa += Da_[nx][nw] * integral_->aptei_aa(nu, nw, nu, nx);
                        Daa -= Da_[nu][nx] * integral_->aptei_aa(nx, nw, nu, nw);
                        Daa += Da_[nx][nw] * integral_->aptei_aa(nw, nv, nx, nv);
                        Daa -= Da_[nv][nx] * integral_->aptei_aa(nx, nw, nv, nw);

                        Dab += Da_[nu][nx] * integral_->aptei_ab(nx, nv, nu, nv);
                        Dab += Db_[nx][nw] * integral_->aptei_ab(nu, nw, nu, nx);
                        Dab -= Da_[nu][nx] * integral_->aptei_ab(nx, nw, nu, nw);
                        Dab += Db_[nx][nw] * integral_->aptei_bb(nw, nv, nx, nv);
                        Dab -= Db_[nv][nx] * integral_->aptei_bb(nx, nw, nv, nw);
                    }

                    if (std::fabs(Daa) < small_threshold) {
                        smallD2aa.push_back(
                            std::make_pair(std::vector<size_t>{nm, nw, nu, nv}, Daa));
                    }
                    if (std::fabs(Dab) < small_threshold) {
                        smallD2ab.push_back(
                            std::make_pair(std::vector<size_t>{nm, nw, nu, nv}, Dab));
                    }
                }
            }
        }
    }

    // print
    print_h2("Small Denominators for T2aa with Dyall Partitioning");
    if (smallD2aa.size() == 0) {
        outfile->Printf("\n    NULL.");
    } else {
        std::string indent(4, ' ');
        std::string dash(57, '-');
        std::string title = indent +
                            str(boost::format("%=19s    %=15s    %=15s\n") % "Indices" %
                                "Denominator" % "Original Denom.") +
                            indent + dash;
        outfile->Printf("\n%s", title.c_str());
        for (const auto& pair : smallD2aa) {
            size_t i = pair.first[0];
            size_t j = pair.first[1];
            size_t k = pair.first[2];
            size_t l = pair.first[3];
            double D = pair.second;
            double Dold = Fa_[i][i] + Fa_[j][j] - Fa_[k][k] - Fa_[l][l];
            outfile->Printf("\n    %4zu %4zu %4zu %4zu    %15.12f    %15.12f", i, j, k, l, D, Dold);
        }
        outfile->Printf("\n    %s", dash.c_str());
    }

    print_h2("Small Denominators for T2ab with Dyall Partitioning");
    if (smallD2ab.size() == 0) {
        outfile->Printf("\n    NULL.");
    } else {
        std::string indent(4, ' ');
        std::string dash(57, '-');
        std::string title = indent +
                            str(boost::format("%=19s    %=15s    %=15s\n") % "Indices" %
                                "Denominator" % "Original Denom.") +
                            indent + dash;
        outfile->Printf("\n%s", title.c_str());
        for (const auto& pair : smallD2ab) {
            size_t i = pair.first[0];
            size_t j = pair.first[1];
            size_t k = pair.first[2];
            size_t l = pair.first[3];
            double D = pair.second;
            double Dold = Fa_[i][i] + Fb_[j][j] - Fa_[k][k] - Fb_[l][l];
            outfile->Printf("\n    %4zu %4zu %4zu %4zu    %15.12f    %15.12f", i, j, k, l, D, Dold);
        }
        outfile->Printf("\n    %s", dash.c_str());
    }
}

void MCSRGPT2_MO::PrintDelta() {
    std::ofstream out_delta;
    out_delta.open("Delta_ijab");
    for (size_t i = 0; i < nhole_; ++i) {
        size_t ni = hole_mos_[i];
        for (size_t j = i; j < nhole_; ++j) {
            size_t nj = hole_mos_[j];
            for (size_t a = 0; a < npart_; ++a) {
                size_t na = part_mos_[a];
                for (size_t b = a; b < npart_; ++b) {
                    size_t nb = part_mos_[b];

                    if (ni == nj && ni == na && ni == nb)
                        continue;
                    if ((sym_ncmo_[ni] ^ sym_ncmo_[nj] ^ sym_ncmo_[na] ^ sym_ncmo_[nb]) != 0)
                        continue;
                    if (std::find(actv_mos_.begin(), actv_mos_.end(), ni) != actv_mos_.end() &&
                        std::find(actv_mos_.begin(), actv_mos_.end(), nj) != actv_mos_.end() &&
                        std::find(actv_mos_.begin(), actv_mos_.end(), na) != actv_mos_.end() &&
                        std::find(actv_mos_.begin(), actv_mos_.end(), nb) != actv_mos_.end()) {
                        continue;
                    } else {
                        double Daa = Fa_[ni][ni] + Fa_[nj][nj] - Fa_[na][na] - Fa_[nb][nb];
                        //                        double Dab = Fa_[ni][ni] +
                        //                        Fb_[nj][nj] - Fa_[na][na] -
                        //                        Fb_[nb][nb];
                        //                        double Dbb = Fb_[ni][ni] +
                        //                        Fb_[nj][nj] - Fb_[na][na] -
                        //                        Fb_[nb][nb];

                        out_delta << boost::format("%3d %3d %3d %3d %20.15f\n") % ni % nj % na %
                                         nb % Daa;
                        //                        out_delta <<
                        //                        boost::format("%3d %3d %3d %3d
                        //                        %20.15f\n") % ni % nj % na %
                        //                        nb % Dab;
                        //                        out_delta <<
                        //                        boost::format("%3d %3d %3d %3d
                        //                        %20.15f\n") % ni % nj % na %
                        //                        nb % Dbb;
                    }
                }
            }
        }
    }
    out_delta.close();
    out_delta.clear();
    out_delta.open("Delta_ia");
    for (size_t i = 0; i < nhole_; ++i) {
        size_t ni = hole_mos_[i];
        for (size_t a = 0; a < npart_; ++a) {
            size_t na = part_mos_[a];
            if (ni == na)
                continue;
            if ((sym_ncmo_[ni] ^ sym_ncmo_[na]) != 0)
                continue;
            if (std::find(actv_mos_.begin(), actv_mos_.end(), ni) != actv_mos_.end() &&
                std::find(actv_mos_.begin(), actv_mos_.end(), na) != actv_mos_.end()) {
                continue;
            } else {
                double delta_a = Fa_[ni][ni] - Fa_[na][na];
                //                double delta_b = Fb_[ni][ni] - Fb_[na][na];
                out_delta << boost::format("%3d %3d %20.15f\n") % ni % na % delta_a;
                //                out_delta << boost::format("%3d %3d
                //                %20.15f\n") % ni % na % delta_b;
            }
        }
    }
    out_delta.close();
}

void MCSRGPT2_MO::Form_AMP_DSRG() {
    // Form T Amplitudes
    T2aa_ = d4(nhole_, d3(nhole_, d2(npart_, d1(npart_))));
    T2ab_ = d4(nhole_, d3(nhole_, d2(npart_, d1(npart_))));
    T2bb_ = d4(nhole_, d3(nhole_, d2(npart_, d1(npart_))));
    T1a_ = d2(nhole_, d1(npart_));
    T1b_ = d2(nhole_, d1(npart_));

    std::string t_algorithm = options_->get_str("T_ALGORITHM");
    t1_amp_ = options_->get_str("T1_AMP");
    bool t1_zero = t1_amp_ == "ZERO";
    outfile->Printf("\n");
    outfile->Printf("\n  Computing MR-DSRG-PT2 T amplitudes ...");

    if (boost::starts_with(t_algorithm, "DSRG")) {
        outfile->Printf("\n  Form T amplitudes using %s formalism.", t_algorithm.c_str());
        Form_T2_DSRG(T2aa_, T2ab_, T2bb_, t_algorithm);
        if (!t1_zero) {
            Form_T1_DSRG(T1a_, T1b_);
        } else {
            outfile->Printf("\n  Zero T1 amplitudes.");
        }
    } else if (t_algorithm == "SELEC") {
        outfile->Printf("\n  Form T amplitudes using DSRG_SELEC formalism. "
                        "(c->a, c->v, a->v)");
        Form_T2_SELEC(T2aa_, T2ab_, T2bb_);
        if (!t1_zero) {
            Form_T1_DSRG(T1a_, T1b_);
        } else {
            outfile->Printf("\n  Zero T1 amplitudes.");
        }
    } else if (t_algorithm == "ISA") {
        outfile->Printf("\n  Form T amplitudes using intruder state "
                        "avoidance (ISA) formalism.");
        double b = options_->get_double("ISA_B");
        Form_T2_ISA(T2aa_, T2ab_, T2bb_, b);
        if (!t1_zero) {
            Form_T1_ISA(T1a_, T1b_, b);
        } else {
            outfile->Printf("\n  Zero T1 amplitudes.");
        }
    }
    outfile->Printf("\n  Done.");

    // Check T Amplitudes
    T2Naa_ = 0.0, T2Nab_ = 0.0, T2Nbb_ = 0.0;
    T2Maxaa_ = 0.0, T2Maxab_ = 0.0, T2Maxbb_ = 0.0;
    Check_T2("AA", T2aa_, T2Naa_, T2Maxaa_, options_);
    Check_T2("AB", T2ab_, T2Nab_, T2Maxab_, options_);
    Check_T2("BB", T2bb_, T2Nbb_, T2Maxbb_, options_);

    T1Na_ = 0.0, T1Nb_ = 0.0;
    T1Maxa_ = 0.0, T1Maxb_ = 0.0;
    Check_T1("A", T1a_, T1Na_, T1Maxa_, options_);
    Check_T1("B", T1b_, T1Nb_, T1Maxb_, options_);

    bool dsrgpt = options_->get_bool("DSRGPT");

    // Effective Fock Matrix
    Fa_dsrg_ = d2(ncmo_, d1(ncmo_));
    Fb_dsrg_ = d2(ncmo_, d1(ncmo_));
    outfile->Printf("\n");
    outfile->Printf("\n  Computing the MR-DSRG-PT2 effective Fock matrix ...");

    Form_Fock_DSRG(Fa_dsrg_, Fb_dsrg_, dsrgpt);
    outfile->Printf("\t\t\tDone.");

    // Effective Two Electron Integrals
    outfile->Printf("\n  Computing the MR-DSRG-PT2 effective two-electron "
                    "integrals ...");

    Form_APTEI_DSRG(dsrgpt);
    outfile->Printf("\tDone.");
}

double MCSRGPT2_MO::compute_energy_dsrg() {
    timer_on("E_MCDSRGPT2");
    double T1max = T1Maxa_, T2max = T2Maxaa_;
    if (std::fabs(T1max) < std::fabs(T1Maxb_))
        T1max = T1Maxb_;
    if (std::fabs(T2max) < std::fabs(T2Maxab_))
        T2max = T2Maxab_;
    if (std::fabs(T2max) < std::fabs(T2Maxbb_))
        T2max = T2Maxbb_;
    double T1norm = sqrt(pow(T1Na_, 2) + pow(T1Nb_, 2));
    double T2norm = sqrt(pow(T2Naa_, 2) + 4 * pow(T2Nab_, 2) + pow(T2Nbb_, 2));

    double E2 = 0.0;
    double E5_1 = 0.0;
    double E5_2 = 0.0;
    double E6_1 = 0.0;
    double E6_2 = 0.0;
    double E7 = 0.0;
    double E8_1 = 0.0;
    double E8_2 = 0.0;
    double E8_3 = 0.0;
    double E10_1 = 0.0;
    double E10_2 = 0.0;

    outfile->Printf("\n");
    outfile->Printf("\n  Computing energy of [F, T1] ...");

    E_FT1(E2);
    outfile->Printf("\t\t\t\t\tDone.");

    outfile->Printf("\n  Computing energy of [V, T1] and [F, T2] ...");

    E_VT1_FT2(E6_1, E6_2, E5_1, E5_2);
    outfile->Printf("\t\t\t\tDone.");

    outfile->Printf("\n  Computing energy of [V, T2] C_2^4 ...");

    E_VT2_2(E7);
    outfile->Printf("\t\t\t\t\tDone.");

    if (options_->get_str("TWOPDC") != "ZERO") {
        outfile->Printf("\n  Computing energy of [V, T2] C_2^2 * C_4 ...");

        E_VT2_4PP(E8_1);
        E_VT2_4HH(E8_2);
        E_VT2_4PH(E8_3);
        outfile->Printf("\t\t\t\tDone.");
    }

    if (options_->get_str("THREEPDC") != "ZERO") {
        outfile->Printf("\n  Computing energy of [V, T2] C_2 * C_6 ...");

        E_VT2_6(E10_1, E10_2);
        outfile->Printf("\t\t\t\tDone.");
    }

    double E5 = E5_1 + E5_2;
    double E6 = E6_1 + E6_2;
    double E8 = E8_1 + E8_2 + E8_3;
    double E10 = E10_1 + E10_2;
    double EVT2 = E7 + E8 + E10;

    Ecorr_ = E2 + E5 + E6 + EVT2;
    Etotal_ = Eref_ + Ecorr_;

    // Print
    outfile->Printf("\n  ");
    outfile->Printf("\n  ==> MC-DSRG-PT2 Energy Summary <==");
    outfile->Printf("\n  ");
    outfile->Printf("\n    E0 (cumulant) %15c = %22.15f", ' ', Eref_);
    outfile->Printf("\n    E([F, T1]) %18c = %22.15lf", ' ', E2);
    outfile->Printf("\n    E([V, T1]) %18c = %22.15lf", ' ', E5);
    outfile->Printf("\n    E([V, T1]: V) %15c = %22.15lf", ' ', E5_1);
    outfile->Printf("\n    E([V, T1]: C) %15c = %22.15lf", ' ', E5_2);
    outfile->Printf("\n    E([F, T2]) %18c = %22.15lf", ' ', E6);
    outfile->Printf("\n    E([F, T2]: V) %15c = %22.15lf", ' ', E6_1);
    outfile->Printf("\n    E([F, T2]: C) %15c = %22.15lf", ' ', E6_2);
    outfile->Printf("\n    E([V, T2] C_2^4) %12c = %22.15lf", ' ', E7);
    outfile->Printf("\n    E([V, T2] C_2^2 * C_4) %6c = %22.15lf", ' ', E8);
    outfile->Printf("\n    E([V, T2] C_2^2 * C_4: PP) %2c = %22.15lf", ' ', E8_1);
    outfile->Printf("\n    E([V, T2] C_2^2 * C_4: HH) %2c = %22.15lf", ' ', E8_2);
    outfile->Printf("\n    E([V, T2] C_2^2 * C_4: PH) %2c = %22.15lf", ' ', E8_3);
    outfile->Printf("\n    E([V, T2] C_2 * C_6) %8c = %22.15lf", ' ', E10);
    outfile->Printf("\n    E([V, T2] C_2 * C_6: H) %5c = %22.15lf", ' ', E10_1);
    outfile->Printf("\n    E([V, T2] C_2 * C_6: P) %5c = %22.15lf", ' ', E10_2);
    outfile->Printf("\n    E([V, T2]) %18c = %22.15lf", ' ', EVT2);
    outfile->Printf("\n    E(SRGPT2) %19c = %22.15lf", ' ', Ecorr_);
    outfile->Printf("\n  * E(Total) %20c = %22.15lf", ' ', Etotal_);
    outfile->Printf("\n    max(T1) %21c = %22.15lf", ' ', T1max);
    outfile->Printf("\n    max(T2) %21c = %22.15lf", ' ', T2max);
    outfile->Printf("\n    ||T1|| %22c = %22.15lf", ' ', T1norm);
    outfile->Printf("\n    ||T2|| %22c = %22.15lf", ' ', T2norm);
    outfile->Printf("\n    ");
    timer_off("E_MCDSRGPT2");
    return Etotal_;
}

void MCSRGPT2_MO::E_FT1(double& E) {
    timer_on("[F, T1]");
    E = 0.0;
    for (size_t i = 0; i < nhole_; ++i) {
        size_t ni = hole_mos_[i];
        for (size_t j = 0; j < nhole_; ++j) {
            size_t nj = hole_mos_[j];
            for (size_t a = 0; a < npart_; ++a) {
                size_t na = part_mos_[a];
                for (size_t b = 0; b < npart_; ++b) {
                    size_t nb = part_mos_[b];
                    E +=
                        Fa_dsrg_[nb][nj] * T1a_[i][a] * Da_[nj][ni] * (Delta(na, nb) - Da_[na][nb]);
                    E +=
                        Fb_dsrg_[nb][nj] * T1b_[i][a] * Db_[nj][ni] * (Delta(na, nb) - Db_[na][nb]);
                }
            }
        }
    }
    timer_off("[F, T1]");
}

void MCSRGPT2_MO::E_VT1_FT2(double& EF1, double& EF2, double& EV1, double& EV2) {
    timer_on("[F, T2] & [V, T1]");
    EF1 = 0.0;
    EF2 = 0.0;
    EV1 = 0.0;
    EV2 = 0.0;
    for (size_t u = 0; u < nactv_; ++u) {
        size_t nu = actv_mos_[u];
        for (size_t v = 0; v < nactv_; ++v) {
            size_t nv = actv_mos_[v];
            for (size_t x = 0; x < nactv_; ++x) {
                size_t nx = actv_mos_[x];
                for (size_t y = 0; y < nactv_; ++y) {
                    size_t ny = actv_mos_[y];

                    for (size_t e = 0; e < nvirt_; ++e) {
                        size_t ne = virt_mos_[e];
                        size_t te = e + nactv_;
                        EV1 +=
                            integral_->aptei_aa(nx, ny, ne, nv) * T1a_[u][te] * L2aa_[x][y][u][v];
                        EV1 +=
                            integral_->aptei_bb(nx, ny, ne, nv) * T1b_[u][te] * L2bb_[x][y][u][v];
                        EV1 += 2 * integral_->aptei_ab(nx, ny, ne, nv) * T1a_[u][te] *
                               L2ab_[x][y][u][v];
                        EV1 += 2 * integral_->aptei_ab(ny, nx, nv, ne) * T1b_[u][te] *
                               L2ab_[y][x][v][u];

                        EF1 += Fa_dsrg_[ne][nx] * T2aa_[u][v][te][y] * L2aa_[x][y][u][v];
                        EF1 += Fb_dsrg_[ne][nx] * T2bb_[u][v][te][y] * L2bb_[x][y][u][v];
                        EF1 += 2 * Fa_dsrg_[ne][nx] * T2ab_[u][v][te][y] * L2ab_[x][y][u][v];
                        EF1 += 2 * Fb_dsrg_[ne][nx] * T2ab_[v][u][y][te] * L2ab_[y][x][v][u];
                    }

                    for (size_t m = 0; m < ncore_; ++m) {
                        size_t nm = core_mos_[m];
                        size_t tm = m + nactv_;
                        EV2 -=
                            integral_->aptei_aa(nm, ny, nu, nv) * T1a_[tm][x] * L2aa_[x][y][u][v];
                        EV2 -=
                            integral_->aptei_bb(nm, ny, nu, nv) * T1b_[tm][x] * L2bb_[x][y][u][v];
                        EV2 -= 2 * integral_->aptei_ab(nm, ny, nu, nv) * T1a_[tm][x] *
                               L2ab_[x][y][u][v];
                        EV2 -= 2 * integral_->aptei_ab(ny, nm, nv, nu) * T1b_[tm][x] *
                               L2ab_[y][x][v][u];

                        EF2 -= Fa_dsrg_[nv][nm] * T2aa_[u][tm][x][y] * L2aa_[x][y][u][v];
                        EF2 -= Fb_dsrg_[nv][nm] * T2bb_[u][tm][x][y] * L2bb_[x][y][u][v];
                        EF2 -= 2 * Fa_dsrg_[nv][nm] * T2ab_[tm][u][y][x] * L2ab_[y][x][v][u];
                        EF2 -= 2 * Fb_dsrg_[nv][nm] * T2ab_[u][tm][x][y] * L2ab_[x][y][u][v];
                    }
                }
            }
        }
    }
    EV1 *= 0.5;
    EV2 *= 0.5;
    EF1 *= 0.5;
    EF2 *= 0.5;
    timer_off("[F, T2] & [V, T1]");
}

void MCSRGPT2_MO::E_VT2_2(double& E) {
    timer_on("[V, T2] C_2^4");
    E = 0.0;
    d4 C1aa(npart_, d3(npart_, d2(nhole_, d1(nhole_))));
    d4 C1ab(npart_, d3(npart_, d2(nhole_, d1(nhole_))));
    d4 C1bb(npart_, d3(npart_, d2(nhole_, d1(nhole_))));
    for (size_t k = 0; k < nhole_; ++k) {
        size_t nk = hole_mos_[k];
        for (size_t l = 0; l < nhole_; ++l) {
            size_t nl = hole_mos_[l];
            for (size_t a = 0; a < npart_; ++a) {
                size_t na = part_mos_[a];
                for (size_t d = 0; d < npart_; ++d) {
                    size_t nd = part_mos_[d];
                    for (size_t c = 0; c < npart_; ++c) {
                        size_t nc = part_mos_[c];
                        C1aa[a][d][k][l] +=
                            integral_->aptei_aa(nk, nl, nc, nd) * (Delta(na, nc) - Da_[na][nc]);
                        C1ab[a][d][k][l] +=
                            integral_->aptei_ab(nk, nl, nc, nd) * (Delta(na, nc) - Da_[na][nc]);
                        C1bb[a][d][k][l] +=
                            integral_->aptei_bb(nk, nl, nc, nd) * (Delta(na, nc) - Db_[na][nc]);
                    }
                }
            }
        }
    }
    d4 C2aa(npart_, d3(npart_, d2(nhole_, d1(nhole_))));
    d4 C2ab(npart_, d3(npart_, d2(nhole_, d1(nhole_))));
    d4 C2bb(npart_, d3(npart_, d2(nhole_, d1(nhole_))));
    for (size_t k = 0; k < nhole_; ++k) {
        for (size_t l = 0; l < nhole_; ++l) {
            for (size_t a = 0; a < npart_; ++a) {
                for (size_t b = 0; b < npart_; ++b) {
                    size_t nb = part_mos_[b];
                    for (size_t d = 0; d < npart_; ++d) {
                        size_t nd = part_mos_[d];
                        C2aa[a][b][k][l] += C1aa[a][d][k][l] * (Delta(nb, nd) - Da_[nb][nd]);
                        C2ab[a][b][k][l] += C1ab[a][d][k][l] * (Delta(nb, nd) - Db_[nb][nd]);
                        C2bb[a][b][k][l] += C1bb[a][d][k][l] * (Delta(nb, nd) - Db_[nb][nd]);
                    }
                }
            }
        }
    }
    C1aa = d4(npart_, d3(npart_, d2(nhole_, d1(nhole_))));
    C1ab = d4(npart_, d3(npart_, d2(nhole_, d1(nhole_))));
    C1bb = d4(npart_, d3(npart_, d2(nhole_, d1(nhole_))));
    for (size_t b = 0; b < npart_; ++b) {
        for (size_t l = 0; l < nhole_; ++l) {
            for (size_t a = 0; a < npart_; ++a) {
                for (size_t i = 0; i < nhole_; ++i) {
                    size_t ni = hole_mos_[i];
                    for (size_t k = 0; k < nhole_; ++k) {
                        size_t nk = hole_mos_[k];
                        C1aa[a][b][i][l] += C2aa[a][b][k][l] * Da_[nk][ni];
                        C1ab[a][b][i][l] += C2ab[a][b][k][l] * Da_[nk][ni];
                        C1bb[a][b][i][l] += C2bb[a][b][k][l] * Db_[nk][ni];
                    }
                }
            }
        }
    }
    C2aa = d4(npart_, d3(npart_, d2(nhole_, d1(nhole_))));
    C2ab = d4(npart_, d3(npart_, d2(nhole_, d1(nhole_))));
    C2bb = d4(npart_, d3(npart_, d2(nhole_, d1(nhole_))));
    for (size_t a = 0; a < npart_; ++a) {
        for (size_t b = 0; b < npart_; ++b) {
            for (size_t i = 0; i < nhole_; ++i) {
                for (size_t j = 0; j < nhole_; ++j) {
                    size_t nj = hole_mos_[j];
                    for (size_t l = 0; l < nhole_; ++l) {
                        size_t nl = hole_mos_[l];
                        C2aa[a][b][i][j] += C1aa[a][b][i][l] * Da_[nl][nj];
                        C2ab[a][b][i][j] += C1ab[a][b][i][l] * Db_[nl][nj];
                        C2bb[a][b][i][j] += C1bb[a][b][i][l] * Db_[nl][nj];
                    }
                }
            }
        }
    }
    for (size_t i = 0; i < nhole_; ++i) {
        for (size_t j = 0; j < nhole_; ++j) {
            for (size_t a = 0; a < npart_; ++a) {
                for (size_t b = 0; b < npart_; ++b) {
                    E += T2aa_[i][j][a][b] * C2aa[a][b][i][j];
                    E += 4 * T2ab_[i][j][a][b] * C2ab[a][b][i][j];
                    E += T2bb_[i][j][a][b] * C2bb[a][b][i][j];
                }
            }
        }
    }
    E *= 0.25;
    timer_off("[V, T2] C_2^4");
}

void MCSRGPT2_MO::E_VT2_4PP(double& E) {
    timer_on("[V, T2] C_4 * C_2^2: PP");
    E = 0.0;
    d4 C1aa(npart_, d3(npart_, d2(nactv_, d1(nactv_))));
    d4 C1ab(npart_, d3(npart_, d2(nactv_, d1(nactv_))));
    d4 C1bb(npart_, d3(npart_, d2(nactv_, d1(nactv_))));
    for (size_t x = 0; x < nactv_; ++x) {
        size_t nx = actv_mos_[x];
        for (size_t y = 0; y < nactv_; ++y) {
            size_t ny = actv_mos_[y];
            for (size_t d = 0; d < npart_; ++d) {
                size_t nd = part_mos_[d];
                for (size_t a = 0; a < npart_; ++a) {
                    size_t na = part_mos_[a];
                    for (size_t c = 0; c < npart_; ++c) {
                        size_t nc = part_mos_[c];
                        C1aa[a][d][x][y] +=
                            integral_->aptei_aa(nx, ny, nc, nd) * (Delta(na, nc) - Da_[na][nc]);
                        C1ab[a][d][x][y] +=
                            integral_->aptei_ab(nx, ny, nc, nd) * (Delta(na, nc) - Da_[na][nc]);
                        C1bb[a][d][x][y] +=
                            integral_->aptei_bb(nx, ny, nc, nd) * (Delta(na, nc) - Db_[na][nc]);
                    }
                }
            }
        }
    }
    d4 C2aa(npart_, d3(npart_, d2(nactv_, d1(nactv_))));
    d4 C2ab(npart_, d3(npart_, d2(nactv_, d1(nactv_))));
    d4 C2bb(npart_, d3(npart_, d2(nactv_, d1(nactv_))));
    for (size_t x = 0; x < nactv_; ++x) {
        for (size_t y = 0; y < nactv_; ++y) {
            for (size_t a = 0; a < npart_; ++a) {
                for (size_t b = 0; b < npart_; ++b) {
                    size_t nb = part_mos_[b];
                    for (size_t d = 0; d < npart_; ++d) {
                        size_t nd = part_mos_[d];
                        C2aa[a][b][x][y] += C1aa[a][d][x][y] * (Delta(nb, nd) - Da_[nb][nd]);
                        C2ab[a][b][x][y] += C1ab[a][d][x][y] * (Delta(nb, nd) - Db_[nb][nd]);
                        C2bb[a][b][x][y] += C1bb[a][d][x][y] * (Delta(nb, nd) - Db_[nb][nd]);
                    }
                }
            }
        }
    }
    C1aa = d4(npart_, d3(npart_, d2(nactv_, d1(nactv_))));
    C1ab = d4(npart_, d3(npart_, d2(nactv_, d1(nactv_))));
    C1bb = d4(npart_, d3(npart_, d2(nactv_, d1(nactv_))));
    for (size_t u = 0; u < nactv_; ++u) {
        for (size_t v = 0; v < nactv_; ++v) {
            for (size_t a = 0; a < npart_; ++a) {
                for (size_t b = 0; b < npart_; ++b) {
                    for (size_t x = 0; x < nactv_; ++x) {
                        for (size_t y = 0; y < nactv_; ++y) {
                            C1aa[a][b][u][v] += C2aa[a][b][x][y] * L2aa_[x][y][u][v];
                            C1bb[a][b][u][v] += C2bb[a][b][x][y] * L2bb_[x][y][u][v];
                            C1ab[a][b][u][v] += C2ab[a][b][x][y] * L2ab_[x][y][u][v];
                        }
                    }
                }
            }
        }
    }
    for (size_t u = 0; u < nactv_; ++u) {
        for (size_t v = 0; v < nactv_; ++v) {
            for (size_t a = 0; a < npart_; ++a) {
                for (size_t b = 0; b < npart_; ++b) {
                    E += C1aa[a][b][u][v] * T2aa_[u][v][a][b];
                    E += C1bb[a][b][u][v] * T2bb_[u][v][a][b];
                    E += 8 * C1ab[a][b][u][v] * T2ab_[u][v][a][b];
                }
            }
        }
    }
    E *= 0.125;
    timer_off("[V, T2] C_4 * C_2^2: PP");
}

void MCSRGPT2_MO::E_VT2_4HH(double& E) {
    timer_on("[V, T2] C_4 * C_2^2: HH");
    E = 0.0;
    d4 C1aa(nactv_, d3(nactv_, d2(nhole_, d1(nhole_))));
    d4 C1ab(nactv_, d3(nactv_, d2(nhole_, d1(nhole_))));
    d4 C1bb(nactv_, d3(nactv_, d2(nhole_, d1(nhole_))));
    for (size_t u = 0; u < nactv_; ++u) {
        size_t nu = actv_mos_[u];
        for (size_t v = 0; v < nactv_; ++v) {
            size_t nv = actv_mos_[v];
            for (size_t l = 0; l < nhole_; ++l) {
                size_t nl = hole_mos_[l];
                for (size_t k = 0; k < nhole_; ++k) {
                    size_t nk = hole_mos_[k];
                    for (size_t i = 0; i < nhole_; ++i) {
                        size_t ni = hole_mos_[i];
                        C1aa[u][v][i][l] += integral_->aptei_aa(nk, nl, nu, nv) * Da_[nk][ni];
                        C1ab[u][v][i][l] += integral_->aptei_ab(nk, nl, nu, nv) * Da_[nk][ni];
                        C1bb[u][v][i][l] += integral_->aptei_bb(nk, nl, nu, nv) * Db_[nk][ni];
                    }
                }
            }
        }
    }
    d4 C2aa(nactv_, d3(nactv_, d2(nhole_, d1(nhole_))));
    d4 C2ab(nactv_, d3(nactv_, d2(nhole_, d1(nhole_))));
    d4 C2bb(nactv_, d3(nactv_, d2(nhole_, d1(nhole_))));
    for (size_t u = 0; u < nactv_; ++u) {
        for (size_t v = 0; v < nactv_; ++v) {
            for (size_t i = 0; i < nhole_; ++i) {
                for (size_t l = 0; l < nhole_; ++l) {
                    size_t nl = hole_mos_[l];
                    for (size_t j = 0; j < nhole_; ++j) {
                        size_t nj = hole_mos_[j];
                        C2aa[u][v][i][j] += C1aa[u][v][i][l] * Da_[nl][nj];
                        C2ab[u][v][i][j] += C1ab[u][v][i][l] * Db_[nl][nj];
                        C2bb[u][v][i][j] += C1bb[u][v][i][l] * Db_[nl][nj];
                    }
                }
            }
        }
    }
    C1aa = d4(nactv_, d3(nactv_, d2(nhole_, d1(nhole_))));
    C1ab = d4(nactv_, d3(nactv_, d2(nhole_, d1(nhole_))));
    C1bb = d4(nactv_, d3(nactv_, d2(nhole_, d1(nhole_))));
    for (size_t x = 0; x < nactv_; ++x) {
        for (size_t y = 0; y < nactv_; ++y) {
            for (size_t i = 0; i < nhole_; ++i) {
                for (size_t j = 0; j < nhole_; ++j) {
                    for (size_t u = 0; u < nactv_; ++u) {
                        for (size_t v = 0; v < nactv_; ++v) {
                            C1aa[x][y][i][j] += C2aa[u][v][i][j] * L2aa_[x][y][u][v];
                            C1ab[x][y][i][j] += C2ab[u][v][i][j] * L2ab_[x][y][u][v];
                            C1bb[x][y][i][j] += C2bb[u][v][i][j] * L2bb_[x][y][u][v];
                        }
                    }
                }
            }
        }
    }
    for (size_t x = 0; x < nactv_; ++x) {
        for (size_t y = 0; y < nactv_; ++y) {
            for (size_t i = 0; i < nhole_; ++i) {
                for (size_t j = 0; j < nhole_; ++j) {
                    E += C1aa[x][y][i][j] * T2aa_[i][j][x][y];
                    E += 8 * C1ab[x][y][i][j] * T2ab_[i][j][x][y];
                    E += C1bb[x][y][i][j] * T2bb_[i][j][x][y];
                }
            }
        }
    }
    E *= 0.125;
    timer_off("[V, T2] C_4 * C_2^2: HH");
}

void MCSRGPT2_MO::E_VT2_4PH(double& E) {
    timer_on("[V, T2] C_4 * C_2^2: PH");
    E = 0.0;
    d4 C11(nactv_, d3(npart_, d2(nhole_, d1(nactv_))));
    d4 C12(nactv_, d3(npart_, d2(nhole_, d1(nactv_))));
    d4 C13(nactv_, d3(npart_, d2(nhole_, d1(nactv_))));
    d4 C14(nactv_, d3(npart_, d2(nhole_, d1(nactv_))));
    d4 C19(nactv_, d3(npart_, d2(nhole_, d1(nactv_))));
    d4 C110(nactv_, d3(npart_, d2(nhole_, d1(nactv_))));
    for (size_t x = 0; x < nactv_; ++x) {
        size_t nx = actv_mos_[x];
        for (size_t v = 0; v < nactv_; ++v) {
            size_t nv = actv_mos_[v];
            for (size_t j = 0; j < nhole_; ++j) {
                size_t nj = hole_mos_[j];
                for (size_t a = 0; a < npart_; ++a) {
                    size_t na = part_mos_[a];
                    for (size_t b = 0; b < npart_; ++b) {
                        size_t nb = part_mos_[b];
                        C11[v][a][j][x] +=
                            integral_->aptei_aa(nj, nx, nv, nb) * (Delta(na, nb) - Da_[na][nb]);
                        C12[v][a][j][x] -=
                            integral_->aptei_ab(nx, nj, nv, nb) * (Delta(na, nb) - Db_[na][nb]);
                        C13[v][a][j][x] +=
                            integral_->aptei_bb(nj, nx, nv, nb) * (Delta(na, nb) - Db_[na][nb]);
                        C14[v][a][j][x] -=
                            integral_->aptei_ab(nj, nx, nb, nv) * (Delta(na, nb) - Da_[na][nb]);
                        C19[v][a][j][x] +=
                            integral_->aptei_ab(nx, nj, nb, nv) * (Delta(na, nb) - Da_[na][nb]);
                        C110[v][a][j][x] +=
                            integral_->aptei_ab(nj, nx, nv, nb) * (Delta(na, nb) - Db_[na][nb]);
                    }
                }
            }
        }
    }
    d4 C21(nactv_, d3(npart_, d2(nhole_, d1(nactv_))));
    d4 C22(nactv_, d3(npart_, d2(nhole_, d1(nactv_))));
    d4 C23(nactv_, d3(npart_, d2(nhole_, d1(nactv_))));
    d4 C24(nactv_, d3(npart_, d2(nhole_, d1(nactv_))));
    d4 C29(nactv_, d3(npart_, d2(nhole_, d1(nactv_))));
    d4 C210(nactv_, d3(npart_, d2(nhole_, d1(nactv_))));
    for (size_t x = 0; x < nactv_; ++x) {
        for (size_t v = 0; v < nactv_; ++v) {
            for (size_t a = 0; a < npart_; ++a) {
                for (size_t i = 0; i < nhole_; ++i) {
                    size_t ni = hole_mos_[i];
                    for (size_t j = 0; j < nhole_; ++j) {
                        size_t nj = hole_mos_[j];
                        C21[v][a][i][x] += C11[v][a][j][x] * Da_[nj][ni];
                        C22[v][a][i][x] += C12[v][a][j][x] * Db_[nj][ni];
                        C23[v][a][i][x] += C13[v][a][j][x] * Db_[nj][ni];
                        C24[v][a][i][x] += C14[v][a][j][x] * Da_[nj][ni];
                        C29[v][a][i][x] += C19[v][a][j][x] * Db_[nj][ni];
                        C210[v][a][i][x] += C110[v][a][j][x] * Da_[nj][ni];
                    }
                }
            }
        }
    }
    d4 C31(nactv_, d3(nactv_, d2(nactv_, d1(nactv_))));
    d4 C32(nactv_, d3(nactv_, d2(nactv_, d1(nactv_))));
    d4 C33(nactv_, d3(nactv_, d2(nactv_, d1(nactv_))));
    d4 C34(nactv_, d3(nactv_, d2(nactv_, d1(nactv_))));
    d4 C35(nactv_, d3(nactv_, d2(nactv_, d1(nactv_))));
    d4 C36(nactv_, d3(nactv_, d2(nactv_, d1(nactv_))));
    d4 C37(nactv_, d3(nactv_, d2(nactv_, d1(nactv_))));
    d4 C38(nactv_, d3(nactv_, d2(nactv_, d1(nactv_))));
    d4 C39(nactv_, d3(nactv_, d2(nactv_, d1(nactv_))));
    d4 C310(nactv_, d3(nactv_, d2(nactv_, d1(nactv_))));
    for (size_t u = 0; u < nactv_; ++u) {
        for (size_t y = 0; y < nactv_; ++y) {
            for (size_t x = 0; x < nactv_; ++x) {
                for (size_t v = 0; v < nactv_; ++v) {
                    for (size_t i = 0; i < nhole_; ++i) {
                        for (size_t a = 0; a < npart_; ++a) {
                            C31[u][v][x][y] += C21[v][a][i][x] * T2aa_[i][u][a][y];
                            C32[u][v][x][y] += C22[v][a][i][x] * T2ab_[u][i][y][a];
                            C33[u][v][x][y] += C23[v][a][i][x] * T2bb_[i][u][a][y];
                            C34[u][v][x][y] += C24[v][a][i][x] * T2ab_[i][u][a][y];
                            C35[u][v][x][y] += C21[v][a][i][x] * T2ab_[i][u][a][y];
                            C36[u][v][x][y] += C22[v][a][i][x] * T2bb_[i][u][a][y];
                            C37[u][v][x][y] += C24[v][a][i][x] * T2aa_[i][u][a][y];
                            C38[u][v][x][y] += C23[v][a][i][x] * T2ab_[u][i][y][a];
                            C39[u][v][x][y] -= C29[v][a][i][x] * T2ab_[u][i][a][y];
                            C310[u][v][x][y] -= C210[v][a][i][x] * T2ab_[i][u][y][a];
                        }
                    }
                }
            }
        }
    }
    for (size_t u = 0; u < nactv_; ++u) {
        for (size_t v = 0; v < nactv_; ++v) {
            for (size_t x = 0; x < nactv_; ++x) {
                for (size_t y = 0; y < nactv_; ++y) {
                    E += C31[u][v][x][y] * L2aa_[x][y][u][v];
                    E += C32[u][v][x][y] * L2aa_[x][y][u][v];
                    E += C33[u][v][x][y] * L2bb_[x][y][u][v];
                    E += C34[u][v][x][y] * L2bb_[x][y][u][v];
                    E -= C35[u][v][x][y] * L2ab_[x][y][v][u];
                    E -= C36[u][v][x][y] * L2ab_[x][y][v][u];
                    E -= C37[u][v][x][y] * L2ab_[y][x][u][v];
                    E -= C38[u][v][x][y] * L2ab_[y][x][u][v];
                    E += C39[u][v][x][y] * L2ab_[x][y][u][v];
                    E += C310[u][v][x][y] * L2ab_[y][x][v][u];
                }
            }
        }
    }
    timer_off("[V, T2] C_4 * C_2^2: PH");
}

void MCSRGPT2_MO::E_VT2_6(double& E1, double& E2) {
    timer_on("[V, T2] C_6 * C_2");
    E1 = 0.0;
    E2 = 0.0;
    for (size_t u = 0; u < nactv_; ++u) {
        size_t nu = actv_mos_[u];
        for (size_t v = 0; v < nactv_; ++v) {
            size_t nv = actv_mos_[v];
            for (size_t w = 0; w < nactv_; ++w) {
                size_t nw = actv_mos_[w];
                for (size_t x = 0; x < nactv_; ++x) {
                    size_t nx = actv_mos_[x];
                    for (size_t y = 0; y < nactv_; ++y) {
                        size_t ny = actv_mos_[y];
                        for (size_t z = 0; z < nactv_; ++z) {
                            size_t nz = actv_mos_[z];
                            for (size_t i = 0; i < nhole_; ++i) {
                                size_t ni = hole_mos_[i];
                                // L3aaa & L3bbb
                                E1 += integral_->aptei_aa(ni, nz, nu, nv) * T2aa_[i][w][x][y] *
                                      L3aaa_[x][y][z][u][v][w];
                                E1 += integral_->aptei_bb(ni, nz, nu, nv) * T2bb_[i][w][x][y] *
                                      L3bbb_[x][y][z][u][v][w];
                                // L3aab
                                E1 -= 2 * L3aab_[x][z][y][u][v][w] * T2ab_[i][w][x][y] *
                                      integral_->aptei_aa(ni, nz, nu, nv);
                                // L3aba & L3baa
                                E1 -= 2 * L3aab_[x][y][z][u][w][v] * T2aa_[i][w][x][y] *
                                      integral_->aptei_ab(ni, nz, nu, nv);
                                E1 += 4 * L3aab_[x][z][y][u][w][v] * T2ab_[w][i][x][y] *
                                      integral_->aptei_ab(nz, ni, nu, nv);
                                // L3abb & L3bab
                                E1 += 4 * L3abb_[x][y][z][u][v][w] * T2ab_[i][w][x][y] *
                                      integral_->aptei_ab(ni, nz, nu, nv);
                                E1 -= 2 * L3abb_[z][x][y][u][v][w] * T2bb_[i][w][x][y] *
                                      integral_->aptei_ab(nz, ni, nu, nv);
                                // L3bba
                                E1 -= 2 * L3abb_[x][y][z][w][u][v] * T2ab_[w][i][x][y] *
                                      integral_->aptei_bb(ni, nz, nu, nv);
                            }
                            for (size_t a = 0; a < npart_; ++a) {
                                size_t na = part_mos_[a];
                                // L3aaa & L3bbb
                                E2 += integral_->aptei_aa(nx, ny, nw, na) * T2aa_[u][v][a][z] *
                                      L3aaa_[x][y][z][u][v][w];
                                E2 += integral_->aptei_bb(nx, ny, nw, na) * T2bb_[u][v][a][z] *
                                      L3bbb_[x][y][z][u][v][w];
                                // L3aab
                                E2 += 2 * L3aab_[x][z][y][u][v][w] * T2aa_[u][v][a][z] *
                                      integral_->aptei_ab(nx, ny, na, nw);
                                // L3aba & L3baa
                                E2 -= 4 * L3aab_[x][z][y][u][w][v] * T2ab_[u][v][z][a] *
                                      integral_->aptei_ab(nx, ny, nw, na);
                                E2 -= 2 * L3aab_[x][y][z][u][w][v] * T2ab_[u][v][a][z] *
                                      integral_->aptei_aa(nx, ny, nw, na);
                                // L3abb & L3bab
                                E2 -= 4 * L3abb_[x][y][z][u][v][w] * T2ab_[u][v][a][z] *
                                      integral_->aptei_ab(nx, ny, na, nw);
                                E2 -= 2 * L3abb_[z][x][y][u][v][w] * T2ab_[u][v][z][a] *
                                      integral_->aptei_bb(nx, ny, nw, na);
                                // L3bba
                                E2 += 2 * L3abb_[x][y][z][w][u][v] * T2bb_[u][v][a][z] *
                                      integral_->aptei_ab(nx, ny, nw, na);
                            }
                        }
                    }
                }
            }
        }
    }
    E1 *= 0.25;
    E2 *= 0.25;
    timer_off("[V, T2] C_6 * C_2");
}

double MCSRGPT2_MO::ESRG_11() {
    double E = 0.0;

    for (size_t i = 0; i < nhole_; ++i) {
        size_t ni = hole_mos_[i];
        for (size_t a = 0; a < npart_; ++a) {
            size_t na = part_mos_[a];

            // i, a cannot all be active
            if (i < nactv_ && a < nactv_)
                continue;

            for (size_t j = 0; j < nhole_; ++j) {
                size_t nj = hole_mos_[j];
                for (size_t b = 0; b < npart_; ++b) {
                    size_t nb = part_mos_[b];

                    double va = 0.0, vb = 0.0;

                    double d1 = Fa_[ni][ni] - Fa_[na][na];
                    double d2 = Fa_[nb][nb] - Fa_[nj][nj];
                    va += Fa_srg_[nb][nj] * Fa_srg_[ni][na] * d1 *
                          srg_source_->compute_renormalized_denominator(d1 * d1 + d2 * d2);

                    d1 = Fb_[ni][ni] - Fb_[na][na];
                    d2 = Fb_[nb][nb] - Fb_[nj][nj];
                    va += Fb_srg_[nb][nj] * Fb_srg_[ni][na] * d1 *
                          srg_source_->compute_renormalized_denominator(d1 * d1 + d2 * d2);

                    for (size_t u = 0; u < nactv_; ++u) {
                        size_t nu = actv_mos_[u];
                        for (size_t v = 0; v < nactv_; ++v) {
                            size_t nv = actv_mos_[v];

                            d1 = Fa_[ni][ni] + Fa_[nv][nv] - Fa_[na][na] - Fa_[nu][nu];
                            d2 = Fa_[nb][nb] - Fa_[nj][nj];
                            va += Fa_srg_[nb][nj] * integral_->aptei_aa(ni, nv, na, nu) *
                                  Da_[nu][nv] * d1 *
                                  srg_source_->compute_renormalized_denominator(d1 * d1 + d2 * d2);
                            d1 = Fa_[ni][ni] + Fb_[nv][nv] - Fa_[na][na] - Fb_[nu][nu];
                            va += Fa_srg_[nb][nj] * integral_->aptei_ab(ni, nv, na, nu) *
                                  Db_[nu][nv] * d1 *
                                  srg_source_->compute_renormalized_denominator(d1 * d1 + d2 * d2);

                            d1 = Fa_[nv][nv] + Fb_[ni][ni] - Fa_[nu][nu] - Fb_[na][na];
                            d2 = Fb_[nb][nb] - Fb_[nj][nj];
                            vb += Fb_srg_[nb][nj] * integral_->aptei_ab(nv, ni, nu, na) *
                                  Da_[nu][nv] * d1 *
                                  srg_source_->compute_renormalized_denominator(d1 * d1 + d2 * d2);
                            d1 = Fb_[nv][nv] + Fb_[ni][ni] - Fb_[nu][nu] - Fb_[na][na];
                            vb += Fb_srg_[nb][nj] * integral_->aptei_bb(nv, ni, nu, na) *
                                  Db_[nu][nv] * d1 *
                                  srg_source_->compute_renormalized_denominator(d1 * d1 + d2 * d2);

                            d1 = Fa_[ni][ni] - Fa_[na][na];
                            d2 = Fa_[nb][nb] + Fa_[nv][nv] - Fa_[nj][nj] - Fa_[nu][nu];
                            va += Fa_srg_[ni][na] * integral_->aptei_aa(nb, nv, nj, nu) *
                                  Da_[nu][nv] * d1 *
                                  srg_source_->compute_renormalized_denominator(d1 * d1 + d2 * d2);
                            d2 = Fa_[nb][nb] + Fb_[nv][nv] - Fa_[nj][nj] - Fb_[nu][nu];
                            va += Fa_srg_[ni][na] * integral_->aptei_ab(nb, nv, nj, nu) *
                                  Db_[nu][nv] * d1 *
                                  srg_source_->compute_renormalized_denominator(d1 * d1 + d2 * d2);

                            d1 = Fb_[ni][ni] - Fb_[na][na];
                            d2 = Fa_[nv][nv] + Fb_[nb][nb] - Fa_[nu][nu] - Fb_[nj][nj];
                            vb += Fb_srg_[ni][na] * integral_->aptei_ab(nv, nb, nu, nj) *
                                  Da_[nu][nv] * d1 *
                                  srg_source_->compute_renormalized_denominator(d1 * d1 + d2 * d2);
                            d2 = Fb_[nv][nv] + Fb_[nb][nb] - Fb_[nu][nu] - Fb_[nj][nj];
                            vb += Fb_srg_[ni][na] * integral_->aptei_bb(nv, nb, nu, nj) *
                                  Db_[nu][nv] * d1 *
                                  srg_source_->compute_renormalized_denominator(d1 * d1 + d2 * d2);

                            for (size_t x = 0; x < nactv_; ++x) {
                                size_t nx = actv_mos_[x];
                                for (size_t y = 0; y < nactv_; ++y) {
                                    size_t ny = actv_mos_[y];

                                    d1 = Fa_[ni][ni] + Fa_[nv][nv] - Fa_[na][na] - Fa_[nu][nu];
                                    d2 = Fa_[nb][nb] + Fa_[ny][ny] - Fa_[nj][nj] - Fa_[nx][nx];
                                    va += integral_->aptei_aa(nb, ny, nj, nx) *
                                          integral_->aptei_aa(ni, nv, na, nu) * Da_[nx][ny] *
                                          Da_[nu][nv] * d1 *
                                          srg_source_->compute_renormalized_denominator(d1 * d1 +
                                                                                        d2 * d2);

                                    d1 = Fa_[ni][ni] + Fb_[nv][nv] - Fa_[na][na] - Fb_[nu][nu];
                                    va += integral_->aptei_aa(nb, ny, nj, nx) *
                                          integral_->aptei_ab(ni, nv, na, nu) * Da_[nx][ny] *
                                          Db_[nu][nv] * d1 *
                                          srg_source_->compute_renormalized_denominator(d1 * d1 +
                                                                                        d2 * d2);

                                    d1 = Fa_[ni][ni] + Fa_[nv][nv] - Fa_[na][na] - Fa_[nu][nu];
                                    d2 = Fa_[nb][nb] + Fb_[ny][ny] - Fa_[nj][nj] - Fb_[nx][nx];
                                    va += integral_->aptei_ab(nb, ny, nj, nx) *
                                          integral_->aptei_aa(ni, nv, na, nu) * Db_[nx][ny] *
                                          Da_[nu][nv] * d1 *
                                          srg_source_->compute_renormalized_denominator(d1 * d1 +
                                                                                        d2 * d2);

                                    d1 = Fa_[ni][ni] + Fb_[nv][nv] - Fa_[na][na] - Fb_[nu][nu];
                                    va += integral_->aptei_ab(nb, ny, nj, nx) *
                                          integral_->aptei_ab(ni, nv, na, nu) * Db_[nx][ny] *
                                          Db_[nu][nv] * d1 *
                                          srg_source_->compute_renormalized_denominator(d1 * d1 +
                                                                                        d2 * d2);

                                    d1 = Fb_[ni][ni] + Fb_[nv][nv] - Fb_[na][na] - Fb_[nu][nu];
                                    d2 = Fb_[nb][nb] + Fb_[ny][ny] - Fb_[nj][nj] - Fb_[nx][nx];
                                    vb += integral_->aptei_bb(nb, ny, nj, nx) *
                                          integral_->aptei_bb(ni, nv, na, nu) * Db_[nx][ny] *
                                          Db_[nu][nv] * d1 *
                                          srg_source_->compute_renormalized_denominator(d1 * d1 +
                                                                                        d2 * d2);

                                    d1 = Fa_[nv][nv] + Fb_[ni][ni] - Fb_[nu][nu] - Fb_[na][na];
                                    vb += integral_->aptei_bb(nb, ny, nj, nx) *
                                          integral_->aptei_ab(nv, ni, nu, na) * Db_[nx][ny] *
                                          Da_[nu][nv] * d1 *
                                          srg_source_->compute_renormalized_denominator(d1 * d1 +
                                                                                        d2 * d2);

                                    d1 = Fb_[ni][ni] + Fb_[nv][nv] - Fb_[na][na] - Fb_[nu][nu];
                                    d2 = Fa_[ny][ny] + Fb_[nb][nb] - Fa_[nx][nx] - Fb_[nj][nj];
                                    vb += integral_->aptei_ab(ny, nb, nx, nj) *
                                          integral_->aptei_bb(ni, nv, na, nu) * Da_[nx][ny] *
                                          Db_[nu][nv] * d1 *
                                          srg_source_->compute_renormalized_denominator(d1 * d1 +
                                                                                        d2 * d2);

                                    d1 = Fa_[nv][nv] + Fb_[ni][ni] - Fb_[nu][nu] - Fb_[na][na];
                                    vb += integral_->aptei_ab(ny, nb, nx, nj) *
                                          integral_->aptei_ab(nv, ni, nu, na) * Da_[nx][ny] *
                                          Da_[nu][nv] * d1 *
                                          srg_source_->compute_renormalized_denominator(d1 * d1 +
                                                                                        d2 * d2);
                                }
                            }
                        }
                    }

                    E += 2.0 * va * Da_[nj][ni] * (Delta(na, nb) - Da_[na][nb]);
                    E += 2.0 * vb * Db_[nj][ni] * (Delta(na, nb) - Db_[na][nb]);
                }
            }
        }
    }

    return E;
}

double MCSRGPT2_MO::ESRG_12() {
    double E = 0.0;

    for (size_t u = 0; u < nactv_; ++u) {
        size_t nu = actv_mos_[u];
        for (size_t v = 0; v < nactv_; ++v) {
            size_t nv = actv_mos_[v];
            for (size_t x = 0; x < nactv_; ++x) {
                size_t nx = actv_mos_[x];
                for (size_t y = 0; y < nactv_; ++y) {
                    size_t ny = actv_mos_[y];

                    double vaa = 0.0, vab = 0.0, vbb = 0.0;

                    // virtual
                    for (size_t e = 0; e < nvirt_; ++e) {
                        size_t ne = virt_mos_[e];

                        double d1 = Fa_[nu][nu] - Fa_[ne][ne];
                        double d2 = Fa_[ne][ne] + Fa_[nv][nv] - Fa_[nx][nx] - Fa_[ny][ny];
                        vaa += integral_->aptei_aa(ne, nv, nx, ny) * Fa_srg_[nu][ne] * d1 *
                               srg_source_->compute_renormalized_denominator(d1 * d1 + d2 * d2);

                        d1 = Fb_[nu][nu] - Fb_[ne][ne];
                        d2 = Fb_[ne][ne] + Fb_[nv][nv] - Fb_[nx][nx] - Fb_[ny][ny];
                        vbb += integral_->aptei_bb(ne, nv, nx, ny) * Fb_srg_[nu][ne] * d1 *
                               srg_source_->compute_renormalized_denominator(d1 * d1 + d2 * d2);

                        d1 = Fa_[nu][nu] - Fa_[ne][ne];
                        d2 = Fa_[ne][ne] + Fb_[nv][nv] - Fa_[nx][nx] - Fb_[ny][ny];
                        vab += integral_->aptei_ab(ne, nv, nx, ny) * Fa_srg_[nu][ne] * d1 *
                               srg_source_->compute_renormalized_denominator(d1 * d1 + d2 * d2);

                        d1 = Fb_[nv][nv] - Fb_[ne][ne];
                        d2 = Fa_[nu][nu] + Fb_[ne][ne] - Fa_[nx][nx] - Fb_[ny][ny];
                        vab += integral_->aptei_ab(nu, ne, nx, ny) * Fb_srg_[nv][ne] * d1 *
                               srg_source_->compute_renormalized_denominator(d1 * d1 + d2 * d2);

                        for (size_t w = 0; w < nactv_; ++w) {
                            size_t nw = actv_mos_[w];
                            for (size_t z = 0; z < nactv_; ++z) {
                                size_t nz = actv_mos_[z];

                                d1 = Fa_[nu][nu] + Fa_[nw][nw] - Fa_[ne][ne] - Fa_[nz][nz];
                                d2 = Fa_[ne][ne] + Fa_[nv][nv] - Fa_[nx][nx] - Fa_[ny][ny];
                                vaa += integral_->aptei_aa(ne, nv, nx, ny) *
                                       integral_->aptei_aa(nu, nw, ne, nz) * Da_[nz][nw] * d1 *
                                       srg_source_->compute_renormalized_denominator(d1 * d1 +
                                                                                     d2 * d2);
                                d1 = Fa_[nu][nu] + Fb_[nw][nw] - Fa_[ne][ne] - Fb_[nz][nz];
                                vaa += integral_->aptei_aa(ne, nv, nx, ny) *
                                       integral_->aptei_ab(nu, nw, ne, nz) * Db_[nz][nw] * d1 *
                                       srg_source_->compute_renormalized_denominator(d1 * d1 +
                                                                                     d2 * d2);

                                d1 = Fb_[nu][nu] + Fb_[nw][nw] - Fb_[ne][ne] - Fb_[nz][nz];
                                d2 = Fb_[ne][ne] + Fb_[nv][nv] - Fb_[nx][nx] - Fb_[ny][ny];
                                vbb += integral_->aptei_bb(ne, nv, nx, ny) *
                                       integral_->aptei_bb(nu, nw, ne, nz) * Db_[nz][nw] * d1 *
                                       srg_source_->compute_renormalized_denominator(d1 * d1 +
                                                                                     d2 * d2);
                                d1 = Fa_[nw][nw] + Fb_[nu][nu] - Fa_[nz][nz] - Fb_[ne][ne];
                                vbb += integral_->aptei_bb(ne, nv, nx, ny) *
                                       integral_->aptei_ab(nw, nu, nz, ne) * Da_[nz][nw] * d1 *
                                       srg_source_->compute_renormalized_denominator(d1 * d1 +
                                                                                     d2 * d2);

                                d1 = Fa_[nu][nu] + Fa_[nw][nw] - Fa_[ne][ne] - Fa_[nz][nz];
                                d2 = Fa_[ne][ne] + Fb_[nv][nv] - Fa_[nx][nx] - Fb_[ny][ny];
                                vab += integral_->aptei_ab(ne, nv, nx, ny) *
                                       integral_->aptei_aa(nu, nw, ne, nz) * Da_[nz][nw] * d1 *
                                       srg_source_->compute_renormalized_denominator(d1 * d1 +
                                                                                     d2 * d2);
                                d1 = Fa_[nu][nu] + Fb_[nw][nw] - Fa_[ne][ne] - Fb_[nz][nz];
                                vab += integral_->aptei_ab(ne, nv, nx, ny) *
                                       integral_->aptei_ab(nu, nw, ne, nz) * Db_[nz][nw] * d1 *
                                       srg_source_->compute_renormalized_denominator(d1 * d1 +
                                                                                     d2 * d2);

                                d1 = Fa_[nw][nw] + Fb_[nv][nv] - Fa_[nz][nz] - Fb_[ne][ne];
                                d2 = Fa_[nu][nu] + Fb_[ne][ne] - Fa_[nx][nx] - Fb_[ny][ny];
                                vab += integral_->aptei_ab(nu, ne, nx, ny) *
                                       integral_->aptei_ab(nw, nv, nz, ne) * Da_[nz][nw] * d1 *
                                       srg_source_->compute_renormalized_denominator(d1 * d1 +
                                                                                     d2 * d2);
                                d1 = Fb_[nw][nw] + Fb_[nv][nv] - Fb_[nz][nz] - Fb_[ne][ne];
                                vab += integral_->aptei_ab(nu, ne, nx, ny) *
                                       integral_->aptei_bb(nw, nv, nz, ne) * Db_[nz][nw] * d1 *
                                       srg_source_->compute_renormalized_denominator(d1 * d1 +
                                                                                     d2 * d2);
                            }
                        }
                    }

                    // core
                    for (size_t m = 0; m < ncore_; ++m) {
                        size_t nm = core_mos_[m];

                        double d1 = Fa_[nm][nm] - Fa_[nx][nx];
                        double d2 = Fa_[nu][nu] + Fa_[nv][nv] - Fa_[nm][nm] - Fa_[ny][ny];
                        vaa -= integral_->aptei_aa(nu, nv, nm, ny) * Fa_srg_[nm][nx] * d1 *
                               srg_source_->compute_renormalized_denominator(d1 * d1 + d2 * d2);

                        d1 = Fb_[nm][nm] - Fb_[nx][nx];
                        d2 = Fb_[nu][nu] + Fb_[nv][nv] - Fb_[nm][nm] - Fb_[ny][ny];
                        vbb -= integral_->aptei_bb(nu, nv, nm, ny) * Fb_srg_[nm][nx] * d1 *
                               srg_source_->compute_renormalized_denominator(d1 * d1 + d2 * d2);

                        d1 = Fa_[nm][nm] - Fa_[nx][nx];
                        d2 = Fa_[nu][nu] + Fb_[nv][nv] - Fa_[nm][nm] - Fb_[ny][ny];
                        vab -= integral_->aptei_ab(nu, nv, nm, ny) * Fa_srg_[nm][nx] * d1 *
                               srg_source_->compute_renormalized_denominator(d1 * d1 + d2 * d2);

                        d1 = Fb_[nm][nm] - Fb_[ny][ny];
                        d2 = Fa_[nu][nu] + Fb_[nv][nv] - Fa_[nx][nx] - Fb_[nm][nm];
                        vab -= integral_->aptei_ab(nu, nv, nx, nm) * Fa_srg_[nm][ny] * d1 *
                               srg_source_->compute_renormalized_denominator(d1 * d1 + d2 * d2);

                        for (size_t w = 0; w < nactv_; ++w) {
                            size_t nw = actv_mos_[w];
                            for (size_t z = 0; z < nactv_; ++z) {
                                size_t nz = actv_mos_[z];

                                d1 = Fa_[nm][nm] + Fa_[nw][nw] - Fa_[nx][nx] - Fa_[nz][nz];
                                d2 = Fa_[nu][nu] + Fa_[nv][nv] - Fa_[nm][nm] - Fa_[ny][ny];
                                vaa -= integral_->aptei_aa(nu, nv, nm, ny) *
                                       integral_->aptei_aa(nm, nw, nx, nz) * Da_[nw][nz] * d1 *
                                       srg_source_->compute_renormalized_denominator(d1 * d1 +
                                                                                     d2 * d2);
                                d1 = Fa_[nm][nm] + Fb_[nw][nw] - Fa_[nx][nx] - Fb_[nz][nz];
                                vaa -= integral_->aptei_aa(nu, nv, nm, ny) *
                                       integral_->aptei_ab(nm, nw, nx, nz) * Db_[nw][nz] * d1 *
                                       srg_source_->compute_renormalized_denominator(d1 * d1 +
                                                                                     d2 * d2);

                                d1 = Fb_[nm][nm] + Fb_[nw][nw] - Fb_[nx][nx] - Fb_[nz][nz];
                                d2 = Fb_[nu][nu] + Fb_[nv][nv] - Fb_[nm][nm] - Fb_[ny][ny];
                                vbb -= integral_->aptei_bb(nu, nv, nm, ny) *
                                       integral_->aptei_bb(nm, nw, nx, nz) * Db_[nw][nz] * d1 *
                                       srg_source_->compute_renormalized_denominator(d1 * d1 +
                                                                                     d2 * d2);
                                d1 = Fa_[nw][nw] + Fb_[nm][nm] - Fa_[nz][nz] - Fb_[nx][nx];
                                vbb -= integral_->aptei_bb(nu, nv, nm, ny) *
                                       integral_->aptei_ab(nw, nm, nz, nx) * Da_[nw][nz] * d1 *
                                       srg_source_->compute_renormalized_denominator(d1 * d1 +
                                                                                     d2 * d2);

                                d1 = Fa_[nm][nm] + Fa_[nw][nw] - Fa_[nx][nx] - Fa_[nz][nz];
                                d2 = Fa_[nu][nu] + Fb_[nv][nv] - Fa_[nm][nm] - Fb_[ny][ny];
                                vab -= integral_->aptei_ab(nu, nv, nm, ny) *
                                       integral_->aptei_aa(nm, nw, nx, nz) * Da_[nw][nz] * d1 *
                                       srg_source_->compute_renormalized_denominator(d1 * d1 +
                                                                                     d2 * d2);
                                d1 = Fa_[nm][nm] + Fb_[nw][nw] - Fa_[nx][nx] - Fb_[nz][nz];
                                vab -= integral_->aptei_ab(nu, nv, nm, ny) *
                                       integral_->aptei_ab(nm, nw, nx, nz) * Db_[nw][nz] * d1 *
                                       srg_source_->compute_renormalized_denominator(d1 * d1 +
                                                                                     d2 * d2);

                                d1 = Fa_[nw][nw] + Fb_[nm][nm] - Fa_[nz][nz] - Fb_[ny][ny];
                                d2 = Fa_[nu][nu] + Fb_[nv][nv] - Fa_[nx][nx] - Fb_[nm][nm];
                                vab -= integral_->aptei_ab(nu, nv, nx, nm) *
                                       integral_->aptei_ab(nw, nm, nz, ny) * Da_[nw][nz] * d1 *
                                       srg_source_->compute_renormalized_denominator(d1 * d1 +
                                                                                     d2 * d2);
                                d1 = Fb_[nw][nw] + Fb_[nm][nm] - Fb_[nz][nz] - Fb_[ny][ny];
                                vab -= integral_->aptei_ab(nu, nv, nx, nm) *
                                       integral_->aptei_bb(nw, nm, nz, ny) * Db_[nw][nz] * d1 *
                                       srg_source_->compute_renormalized_denominator(d1 * d1 +
                                                                                     d2 * d2);
                            }
                        }
                    }

                    E += vaa * L2aa_[x][y][u][v];
                    E += 2.0 * vab * L2ab_[x][y][u][v];
                    E += vbb * L2bb_[x][y][u][v];
                }
            }
        }
    }

    return E;
}

double MCSRGPT2_MO::ESRG_21() {
    double E = 0.0;

    for (size_t u = 0; u < nactv_; ++u) {
        size_t nu = actv_mos_[u];
        for (size_t v = 0; v < nactv_; ++v) {
            size_t nv = actv_mos_[v];
            for (size_t x = 0; x < nactv_; ++x) {
                size_t nx = actv_mos_[x];
                for (size_t y = 0; y < nactv_; ++y) {
                    size_t ny = actv_mos_[y];

                    double vaa = 0.0, vab = 0.0, vbb = 0.0;

                    // virtual
                    for (size_t e = 0; e < nvirt_; ++e) {
                        size_t ne = virt_mos_[e];

                        double d1 = Fa_[nu][nu] + Fa_[nv][nv] - Fa_[ne][ne] - Fa_[ny][ny];
                        double d2 = Fa_[ne][ne] - Fa_[nx][nx];
                        vaa += Fa_srg_[ne][nx] * integral_->aptei_aa(nu, nv, ne, ny) * d1 *
                               srg_source_->compute_renormalized_denominator(d1 * d1 + d2 * d2);

                        d1 = Fb_[nu][nu] + Fb_[nv][nv] - Fb_[ne][ne] - Fb_[ny][ny];
                        d2 = Fb_[ne][ne] - Fb_[nx][nx];
                        vbb += Fb_srg_[ne][nx] * integral_->aptei_bb(nu, nv, ne, ny) * d1 *
                               srg_source_->compute_renormalized_denominator(d1 * d1 + d2 * d2);

                        d1 = Fa_[nu][nu] + Fb_[nv][nv] - Fa_[ne][ne] - Fb_[ny][ny];
                        d2 = Fa_[ne][ne] - Fa_[nx][nx];
                        vab += Fa_srg_[ne][nx] * integral_->aptei_ab(nu, nv, ne, ny) * d1 *
                               srg_source_->compute_renormalized_denominator(d1 * d1 + d2 * d2);

                        d1 = Fa_[nu][nu] + Fb_[nv][nv] - Fa_[nx][nx] - Fb_[ne][ne];
                        d2 = Fb_[ne][ne] - Fb_[ny][ny];
                        vab += Fb_srg_[ne][ny] * integral_->aptei_ab(nu, nv, nx, ne) * d1 *
                               srg_source_->compute_renormalized_denominator(d1 * d1 + d2 * d2);

                        for (size_t w = 0; w < nactv_; ++w) {
                            size_t nw = actv_mos_[w];
                            for (size_t z = 0; z < nactv_; ++z) {
                                size_t nz = actv_mos_[z];

                                d1 = Fa_[nu][nu] + Fa_[nv][nv] - Fa_[ne][ne] - Fa_[ny][ny];
                                d2 = Fa_[ne][ne] + Fa_[nw][nw] - Fa_[nx][nx] - Fa_[nz][nz];
                                vaa += integral_->aptei_aa(nu, nv, ne, ny) *
                                       integral_->aptei_aa(ne, nw, nx, nz) * Da_[nz][nw] * d1 *
                                       srg_source_->compute_renormalized_denominator(d1 * d1 +
                                                                                     d2 * d2);
                                d2 = Fa_[ne][ne] + Fb_[nw][nw] - Fa_[nx][nx] - Fb_[nz][nz];
                                vaa += integral_->aptei_aa(nu, nv, ne, ny) *
                                       integral_->aptei_ab(ne, nw, nx, nz) * Db_[nz][nw] * d1 *
                                       srg_source_->compute_renormalized_denominator(d1 * d1 +
                                                                                     d2 * d2);

                                d1 = Fb_[nu][nu] + Fb_[nv][nv] - Fb_[ne][ne] - Fb_[ny][ny];
                                d2 = Fb_[ne][ne] + Fb_[nw][nw] - Fb_[nx][nx] - Fb_[nz][nz];
                                vbb += integral_->aptei_bb(nu, nv, ne, ny) *
                                       integral_->aptei_bb(ne, nw, nx, nz) * Db_[nz][nw] * d1 *
                                       srg_source_->compute_renormalized_denominator(d1 * d1 +
                                                                                     d2 * d2);
                                d2 = Fa_[nw][nw] + Fb_[ne][ne] - Fa_[nz][nz] - Fb_[nx][nx];
                                vbb += integral_->aptei_bb(nu, nv, ne, ny) *
                                       integral_->aptei_ab(nw, ne, nz, nx) * Da_[nz][nw] * d1 *
                                       srg_source_->compute_renormalized_denominator(d1 * d1 +
                                                                                     d2 * d2);

                                d1 = Fa_[nu][nu] + Fb_[nv][nv] - Fa_[ne][ne] - Fb_[ny][ny];
                                d2 = Fa_[ne][ne] + Fa_[nw][nw] - Fa_[nx][nx] - Fa_[nz][nz];
                                vab += integral_->aptei_ab(nu, nv, ne, ny) *
                                       integral_->aptei_aa(ne, nw, nx, nz) * Da_[nz][nw] * d1 *
                                       srg_source_->compute_renormalized_denominator(d1 * d1 +
                                                                                     d2 * d2);
                                d2 = Fa_[ne][ne] + Fb_[nw][nw] - Fa_[nx][nx] - Fb_[nz][nz];
                                vab += integral_->aptei_ab(nu, nv, ne, ny) *
                                       integral_->aptei_ab(ne, nw, nx, nz) * Db_[nz][nw] * d1 *
                                       srg_source_->compute_renormalized_denominator(d1 * d1 +
                                                                                     d2 * d2);

                                d1 = Fa_[nu][nu] + Fb_[nv][nv] - Fa_[nx][nx] - Fb_[ne][ne];
                                d2 = Fa_[nw][nw] + Fb_[ne][ne] - Fa_[nz][nz] - Fb_[ny][ny];
                                vab += integral_->aptei_ab(nu, nv, nx, ne) *
                                       integral_->aptei_ab(nw, ne, nz, ny) * Da_[nz][nw] * d1 *
                                       srg_source_->compute_renormalized_denominator(d1 * d1 +
                                                                                     d2 * d2);
                                d2 = Fb_[nw][nw] + Fb_[ne][ne] - Fb_[nz][nz] - Fb_[ny][ny];
                                vab += integral_->aptei_ab(nu, nv, nx, ne) *
                                       integral_->aptei_bb(nw, ne, nz, ny) * Db_[nz][nw] * d1 *
                                       srg_source_->compute_renormalized_denominator(d1 * d1 +
                                                                                     d2 * d2);
                            }
                        }
                    }

                    // core
                    for (size_t m = 0; m < ncore_; ++m) {
                        size_t nm = core_mos_[m];

                        double d1 = Fa_[nu][nu] + Fa_[nm][nm] - Fa_[nx][nx] - Fa_[ny][ny];
                        double d2 = Fa_[nv][nv] - Fa_[nm][nm];
                        vaa -= Fa_srg_[nv][nm] * integral_->aptei_aa(nu, nm, nx, ny) * d1 *
                               srg_source_->compute_renormalized_denominator(d1 * d1 + d2 * d2);

                        d1 = Fb_[nu][nu] + Fb_[nm][nm] - Fb_[nx][nx] - Fb_[ny][ny];
                        d2 = Fb_[nv][nv] - Fb_[nm][nm];
                        vbb -= Fb_srg_[nv][nm] * integral_->aptei_bb(nu, nm, nx, ny) * d1 *
                               srg_source_->compute_renormalized_denominator(d1 * d1 + d2 * d2);

                        d1 = Fa_[nu][nu] + Fb_[nm][nm] - Fa_[nx][nx] - Fb_[ny][ny];
                        d2 = Fb_[nv][nv] - Fb_[nm][nm];
                        vab -= Fb_srg_[nv][nm] * integral_->aptei_ab(nu, nm, nx, ny) * d1 *
                               srg_source_->compute_renormalized_denominator(d1 * d1 + d2 * d2);

                        d1 = Fa_[nm][nm] + Fb_[nv][nv] - Fa_[nx][nx] - Fb_[ny][ny];
                        d2 = Fa_[nu][nu] - Fa_[nm][nm];
                        vab -= Fa_srg_[nu][nm] * integral_->aptei_ab(nm, nv, nx, ny) * d1 *
                               srg_source_->compute_renormalized_denominator(d1 * d1 + d2 * d2);

                        for (size_t w = 0; w < nactv_; ++w) {
                            size_t nw = actv_mos_[w];
                            for (size_t z = 0; z < nactv_; ++z) {
                                size_t nz = actv_mos_[z];

                                d1 = Fa_[nu][nu] + Fa_[nm][nm] - Fa_[nx][nx] - Fa_[ny][ny];
                                d2 = Fa_[nv][nv] + Fa_[nw][nw] - Fa_[nm][nm] - Fa_[nz][nz];
                                vaa -= integral_->aptei_aa(nu, nm, nx, ny) *
                                       integral_->aptei_aa(nv, nw, nm, nz) * Da_[nw][nz] * d1 *
                                       srg_source_->compute_renormalized_denominator(d1 * d1 +
                                                                                     d2 * d2);
                                d2 = Fa_[nv][nv] + Fb_[nw][nw] - Fa_[nm][nm] - Fb_[nz][nz];
                                vaa -= integral_->aptei_aa(nu, nm, nx, ny) *
                                       integral_->aptei_ab(nv, nw, nm, nz) * Db_[nw][nz] * d1 *
                                       srg_source_->compute_renormalized_denominator(d1 * d1 +
                                                                                     d2 * d2);

                                d1 = Fb_[nu][nu] + Fb_[nm][nm] - Fb_[nx][nx] - Fb_[ny][ny];
                                d2 = Fb_[nv][nv] + Fb_[nw][nw] - Fb_[nm][nm] - Fb_[nz][nz];
                                vbb -= integral_->aptei_bb(nu, nm, nx, ny) *
                                       integral_->aptei_bb(nv, nw, nm, nz) * Db_[nw][nz] * d1 *
                                       srg_source_->compute_renormalized_denominator(d1 * d1 +
                                                                                     d2 * d2);
                                d2 = Fa_[nw][nw] + Fb_[nv][nv] - Fa_[nz][nz] - Fb_[nm][nm];
                                vbb -= integral_->aptei_bb(nu, nm, nx, ny) *
                                       integral_->aptei_ab(nw, nv, nz, nm) * Da_[nw][nz] * d1 *
                                       srg_source_->compute_renormalized_denominator(d1 * d1 +
                                                                                     d2 * d2);

                                d1 = Fa_[nu][nu] + Fb_[nm][nm] - Fa_[nx][nx] - Fb_[ny][ny];
                                d2 = Fa_[nw][nw] + Fb_[nv][nv] - Fa_[nz][nz] - Fb_[nm][nm];
                                vab -= integral_->aptei_ab(nu, nm, nx, ny) *
                                       integral_->aptei_ab(nw, nv, nz, nm) * Da_[nw][nz] * d1 *
                                       srg_source_->compute_renormalized_denominator(d1 * d1 +
                                                                                     d2 * d2);
                                d2 = Fb_[nw][nw] + Fb_[nv][nv] - Fb_[nz][nz] - Fb_[nm][nm];
                                vab -= integral_->aptei_ab(nu, nm, nx, ny) *
                                       integral_->aptei_bb(nw, nv, nz, nm) * Db_[nw][nz] * d1 *
                                       srg_source_->compute_renormalized_denominator(d1 * d1 +
                                                                                     d2 * d2);

                                d1 = Fa_[nm][nm] + Fb_[nv][nv] - Fa_[nx][nx] - Fb_[ny][ny];
                                d2 = Fa_[nu][nu] + Fa_[nw][nw] - Fa_[nm][nm] - Fa_[nz][nz];
                                vab -= integral_->aptei_ab(nm, nv, nx, ny) *
                                       integral_->aptei_aa(nu, nw, nm, nz) * Da_[nw][nz] * d1 *
                                       srg_source_->compute_renormalized_denominator(d1 * d1 +
                                                                                     d2 * d2);
                                d2 = Fa_[nu][nu] + Fb_[nw][nw] - Fa_[nm][nm] - Fb_[nz][nz];
                                vab -= integral_->aptei_ab(nm, nv, nx, ny) *
                                       integral_->aptei_ab(nu, nw, nm, nz) * Db_[nw][nz] * d1 *
                                       srg_source_->compute_renormalized_denominator(d1 * d1 +
                                                                                     d2 * d2);
                            }
                        }
                    }

                    E += vaa * L2aa_[x][y][u][v];
                    E += 2.0 * vab * L2ab_[x][y][u][v];
                    E += vbb * L2bb_[x][y][u][v];
                }
            }
        }
    }

    return E;
}

double MCSRGPT2_MO::ESRG_22_2() {
    double E = 0.0;

    for (size_t i = 0; i < nhole_; ++i) {
        size_t ni = hole_mos_[i];
        for (size_t j = 0; j < nhole_; ++j) {
            size_t nj = hole_mos_[j];
            for (size_t a = 0; a < npart_; ++a) {
                size_t na = part_mos_[a];
                for (size_t b = 0; b < npart_; ++b) {
                    size_t nb = part_mos_[b];

                    // i, j, a, b cannot all be active
                    if (i < nactv_ && j < nactv_ && a < nactv_ && b < nactv_)
                        continue;

                    for (size_t k = 0; k < nhole_; ++k) {
                        size_t nk = hole_mos_[k];
                        for (size_t l = 0; l < nhole_; ++l) {
                            size_t nl = hole_mos_[l];
                            for (size_t c = 0; c < npart_; ++c) {
                                size_t nc = part_mos_[c];
                                for (size_t d = 0; d < npart_; ++d) {
                                    size_t nd = part_mos_[d];

                                    double d1 =
                                        Fa_[ni][ni] + Fa_[nj][nj] - Fa_[na][na] - Fa_[nb][nb];
                                    double d2 =
                                        Fa_[nc][nc] + Fa_[nd][nd] - Fa_[nk][nk] - Fa_[nl][nl];
                                    E += 0.5 * integral_->aptei_aa(nc, nd, nk, nl) *
                                         integral_->aptei_aa(na, nb, ni, nj) * Da_[nk][ni] *
                                         Da_[nl][nj] * (Delta(na, nc) - Da_[na][nc]) *
                                         (Delta(nb, nd) - Da_[nb][nd]) * d1 *
                                         srg_source_->compute_renormalized_denominator(d1 * d1 +
                                                                                       d2 * d2);

                                    d1 = Fa_[ni][ni] + Fb_[nj][nj] - Fa_[na][na] - Fb_[nb][nb];
                                    d2 = Fa_[nc][nc] + Fb_[nd][nd] - Fa_[nk][nk] - Fb_[nl][nl];
                                    E += 2.0 * integral_->aptei_ab(nc, nd, nk, nl) *
                                         integral_->aptei_ab(na, nb, ni, nj) * Da_[nk][ni] *
                                         Db_[nl][nj] * (Delta(na, nc) - Da_[na][nc]) *
                                         (Delta(nb, nd) - Db_[nb][nd]) * d1 *
                                         srg_source_->compute_renormalized_denominator(d1 * d1 +
                                                                                       d2 * d2);

                                    d1 = Fb_[ni][ni] + Fb_[nj][nj] - Fb_[na][na] - Fb_[nb][nb];
                                    d2 = Fb_[nc][nc] + Fb_[nd][nd] - Fb_[nk][nk] - Fb_[nl][nl];
                                    E += 0.5 * integral_->aptei_bb(nc, nd, nk, nl) *
                                         integral_->aptei_bb(na, nb, ni, nj) * Db_[nk][ni] *
                                         Db_[nl][nj] * (Delta(na, nc) - Db_[na][nc]) *
                                         (Delta(nb, nd) - Db_[nb][nd]) * d1 *
                                         srg_source_->compute_renormalized_denominator(d1 * d1 +
                                                                                       d2 * d2);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    return E;
}

double MCSRGPT2_MO::ESRG_22_4() {
    double E = 0.0;

    for (size_t u = 0; u < nactv_; ++u) {
        size_t nu = actv_mos_[u];
        for (size_t v = 0; v < nactv_; ++v) {
            size_t nv = actv_mos_[v];
            for (size_t x = 0; x < nactv_; ++x) {
                size_t nx = actv_mos_[x];
                for (size_t y = 0; y < nactv_; ++y) {
                    size_t ny = actv_mos_[y];

                    double vaa = 0.0, vab = 0.0, vbb = 0.0;

                    // hole-hole
                    for (size_t i = 0; i < nhole_; ++i) {
                        size_t ni = hole_mos_[i];
                        for (size_t j = 0; j < nhole_; ++j) {
                            size_t nj = hole_mos_[j];

                            // i, j cannot all be active
                            if (i < nactv_ && j < nactv_)
                                continue;

                            for (size_t k = 0; k < nhole_; ++k) {
                                size_t nk = hole_mos_[k];
                                for (size_t l = 0; l < nhole_; ++l) {
                                    size_t nl = hole_mos_[l];

                                    double d1 =
                                        Fa_[ni][ni] + Fa_[nj][nj] - Fa_[nx][nx] - Fa_[ny][ny];
                                    double d2 =
                                        Fa_[nu][nu] + Fa_[nv][nv] - Fa_[nk][nk] - Fa_[nl][nl];
                                    vaa += 0.25 * integral_->aptei_aa(nu, nv, nk, nl) *
                                           integral_->aptei_aa(ni, nj, nx, ny) * Da_[nk][ni] *
                                           Da_[nl][nj] * d1 *
                                           srg_source_->compute_renormalized_denominator(d1 * d1 +
                                                                                         d2 * d2);

                                    d1 = Fb_[ni][ni] + Fb_[nj][nj] - Fb_[nx][nx] - Fb_[ny][ny];
                                    d2 = Fb_[nu][nu] + Fb_[nv][nv] - Fb_[nk][nk] - Fb_[nl][nl];
                                    vbb += 0.25 * integral_->aptei_bb(nu, nv, nk, nl) *
                                           integral_->aptei_bb(ni, nj, nx, ny) * Db_[nk][ni] *
                                           Db_[nl][nj] * d1 *
                                           srg_source_->compute_renormalized_denominator(d1 * d1 +
                                                                                         d2 * d2);

                                    d1 = Fa_[ni][ni] + Fb_[nj][nj] - Fa_[nx][nx] - Fb_[ny][ny];
                                    d2 = Fa_[nu][nu] + Fb_[nv][nv] - Fa_[nk][nk] - Fb_[nl][nl];
                                    vab += 2.0 * integral_->aptei_ab(nu, nv, nk, nl) *
                                           integral_->aptei_ab(ni, nj, nx, ny) * Da_[nk][ni] *
                                           Db_[nl][nj] * d1 *
                                           srg_source_->compute_renormalized_denominator(d1 * d1 +
                                                                                         d2 * d2);
                                }
                            }
                        }
                    }

                    // particle-particle
                    for (size_t a = 0; a < npart_; ++a) {
                        size_t na = part_mos_[a];
                        for (size_t b = 0; b < npart_; ++b) {
                            size_t nb = part_mos_[b];

                            // a, b cannot all be active
                            if (a < nactv_ && b < nactv_)
                                continue;

                            for (size_t c = 0; c < npart_; ++c) {
                                size_t nc = part_mos_[c];
                                for (size_t d = 0; d < npart_; ++d) {
                                    size_t nd = part_mos_[d];

                                    double d1 =
                                        Fa_[nu][nu] + Fa_[nv][nv] - Fa_[na][na] - Fa_[nb][nb];
                                    double d2 =
                                        Fa_[nc][nc] + Fa_[nd][nd] - Fa_[nx][nx] - Fa_[ny][ny];
                                    vaa += 0.25 * integral_->aptei_aa(nc, nd, nx, ny) *
                                           integral_->aptei_aa(nu, nv, na, nb) *
                                           (Delta(na, nc) - Da_[na][nc]) *
                                           (Delta(nb, nd) - Da_[nb][nd]) * d1 *
                                           srg_source_->compute_renormalized_denominator(d1 * d1 +
                                                                                         d2 * d2);

                                    d1 = Fb_[nu][nu] + Fb_[nv][nv] - Fb_[na][na] - Fb_[nb][nb];
                                    d2 = Fb_[nc][nc] + Fb_[nd][nd] - Fb_[nx][nx] - Fb_[ny][ny];
                                    vbb += 0.25 * integral_->aptei_bb(nc, nd, nx, ny) *
                                           integral_->aptei_bb(nu, nv, na, nb) *
                                           (Delta(na, nc) - Db_[na][nc]) *
                                           (Delta(nb, nd) - Db_[nb][nd]) * d1 *
                                           srg_source_->compute_renormalized_denominator(d1 * d1 +
                                                                                         d2 * d2);

                                    d1 = Fa_[nu][nu] + Fb_[nv][nv] - Fa_[na][na] - Fb_[nb][nb];
                                    d2 = Fa_[nc][nc] + Fb_[nd][nd] - Fa_[nx][nx] - Fb_[ny][ny];
                                    vab += 2.0 * integral_->aptei_ab(nc, nd, nx, ny) *
                                           integral_->aptei_ab(nu, nv, na, nb) *
                                           (Delta(na, nc) - Da_[na][nc]) *
                                           (Delta(nb, nd) - Db_[nb][nd]) * d1 *
                                           srg_source_->compute_renormalized_denominator(d1 * d1 +
                                                                                         d2 * d2);
                                }
                            }
                        }
                    }

                    // particle-hole
                    for (size_t i = 0; i < nhole_; ++i) {
                        size_t ni = hole_mos_[i];
                        for (size_t a = 0; a < npart_; ++a) {
                            size_t na = part_mos_[a];

                            // i, a cannot all be active
                            if (i < nactv_ && a < nactv_)
                                continue;

                            for (size_t j = 0; j < nhole_; ++j) {
                                size_t nj = hole_mos_[j];
                                for (size_t b = 0; b < npart_; ++b) {
                                    size_t nb = part_mos_[b];

                                    double d1 =
                                        Fa_[ni][ni] + Fa_[nv][nv] - Fa_[na][na] - Fa_[ny][ny];
                                    double d2 =
                                        Fa_[nb][nb] + Fa_[nu][nu] - Fa_[nj][nj] - Fa_[nx][nx];
                                    vaa += 2.0 * integral_->aptei_aa(nb, nu, nj, nx) *
                                           integral_->aptei_aa(ni, nv, na, ny) * Da_[nj][ni] *
                                           (Delta(na, nb) - Da_[na][nb]) * d1 *
                                           srg_source_->compute_renormalized_denominator(d1 * d1 +
                                                                                         d2 * d2);

                                    d1 = Fa_[nv][nv] + Fb_[ni][ni] - Fa_[ny][ny] - Fb_[na][na];
                                    d2 = Fa_[nu][nu] + Fb_[nb][nb] - Fa_[nx][nx] - Fb_[nj][nj];
                                    vaa += 2.0 * integral_->aptei_ab(nu, nb, nx, nj) *
                                           integral_->aptei_ab(nv, ni, ny, na) * Db_[nj][ni] *
                                           (Delta(na, nb) - Db_[na][nb]) * d1 *
                                           srg_source_->compute_renormalized_denominator(d1 * d1 +
                                                                                         d2 * d2);

                                    d1 = Fb_[ni][ni] + Fb_[nv][nv] - Fb_[na][na] - Fb_[ny][ny];
                                    d2 = Fb_[nb][nb] + Fb_[nu][nu] - Fb_[nj][nj] - Fb_[nx][nx];
                                    vbb += 2.0 * integral_->aptei_bb(nb, nu, nj, nx) *
                                           integral_->aptei_bb(ni, nv, na, ny) * Db_[nj][ni] *
                                           (Delta(na, nb) - Db_[na][nb]) * d1 *
                                           srg_source_->compute_renormalized_denominator(d1 * d1 +
                                                                                         d2 * d2);

                                    d1 = Fa_[ni][ni] + Fb_[nv][nv] - Fa_[na][na] - Fb_[ny][ny];
                                    d2 = Fa_[nb][nb] + Fb_[nu][nu] - Fa_[nj][nj] - Fb_[nx][nx];
                                    vbb += 2.0 * integral_->aptei_ab(nb, nu, nj, nx) *
                                           integral_->aptei_ab(ni, nv, na, ny) * Da_[nj][ni] *
                                           (Delta(na, nb) - Da_[na][nb]) * d1 *
                                           srg_source_->compute_renormalized_denominator(d1 * d1 +
                                                                                         d2 * d2);

                                    d1 = Fa_[nu][nu] + Fb_[ni][ni] - Fa_[na][na] - Fb_[ny][ny];
                                    d2 = Fa_[nb][nb] + Fb_[nv][nv] - Fa_[nx][nx] - Fb_[nj][nj];
                                    vab -= 2.0 * integral_->aptei_ab(nb, nv, nx, nj) *
                                           integral_->aptei_ab(nu, ni, na, ny) * Db_[nj][ni] *
                                           (Delta(na, nb) - Da_[na][nb]) * d1 *
                                           srg_source_->compute_renormalized_denominator(d1 * d1 +
                                                                                         d2 * d2);

                                    d1 = Fa_[ni][ni] + Fb_[nv][nv] - Fa_[na][na] - Fb_[ny][ny];
                                    d2 = Fa_[nu][nu] + Fa_[nb][nb] - Fa_[nx][nx] - Fa_[nj][nj];
                                    vab += 2.0 * integral_->aptei_aa(nu, nb, nx, nj) *
                                           integral_->aptei_ab(ni, nv, na, ny) * Da_[nj][ni] *
                                           (Delta(na, nb) - Da_[na][nb]) * d1 *
                                           srg_source_->compute_renormalized_denominator(d1 * d1 +
                                                                                         d2 * d2);

                                    d1 = Fb_[ni][ni] + Fb_[nv][nv] - Fb_[na][na] - Fb_[ny][ny];
                                    d2 = Fa_[nu][nu] + Fb_[nb][nb] - Fa_[nx][nx] - Fb_[nj][nj];
                                    vab += 2.0 * integral_->aptei_ab(nu, nb, nx, nj) *
                                           integral_->aptei_bb(ni, nv, na, ny) * Db_[nj][ni] *
                                           (Delta(na, nb) - Db_[na][nb]) * d1 *
                                           srg_source_->compute_renormalized_denominator(d1 * d1 +
                                                                                         d2 * d2);

                                    d1 = Fa_[ni][ni] + Fb_[nv][nv] - Fa_[nx][nx] - Fb_[na][na];
                                    d2 = Fa_[nu][nu] + Fb_[nb][nb] - Fa_[nj][nj] - Fb_[ny][ny];
                                    vab -= 2.0 * integral_->aptei_ab(nu, nb, nj, ny) *
                                           integral_->aptei_ab(ni, nv, nx, na) * Da_[nj][ni] *
                                           (Delta(na, nb) - Db_[na][nb]) * d1 *
                                           srg_source_->compute_renormalized_denominator(d1 * d1 +
                                                                                         d2 * d2);

                                    d1 = Fa_[ni][ni] + Fa_[nu][nu] - Fa_[na][na] - Fa_[nx][nx];
                                    d2 = Fa_[nb][nb] + Fb_[nv][nv] - Fa_[nj][nj] - Fb_[ny][ny];
                                    vab += 2.0 * integral_->aptei_ab(nb, nv, nj, ny) *
                                           integral_->aptei_aa(ni, nu, na, nx) * Da_[nj][ni] *
                                           (Delta(na, nb) - Da_[na][nb]) * d1 *
                                           srg_source_->compute_renormalized_denominator(d1 * d1 +
                                                                                         d2 * d2);

                                    d1 = Fa_[nu][nu] + Fb_[ni][ni] - Fa_[nx][nx] - Fb_[na][na];
                                    d2 = Fb_[nb][nb] + Fb_[nv][nv] - Fb_[nj][nj] - Fb_[ny][ny];
                                    vab += 2.0 * integral_->aptei_bb(nb, nv, nj, ny) *
                                           integral_->aptei_ab(nu, ni, nx, na) * Db_[nj][ni] *
                                           (Delta(na, nb) - Db_[na][nb]) * d1 *
                                           srg_source_->compute_renormalized_denominator(d1 * d1 +
                                                                                         d2 * d2);
                                }
                            }
                        }
                    }

                    E += vaa * L2aa_[x][y][u][v];
                    E += vab * L2ab_[x][y][u][v];
                    E += vbb * L2bb_[x][y][u][v];
                }
            }
        }
    }

    return E;
}

double MCSRGPT2_MO::ESRG_22_6() {
    double E = 0.0;

    for (size_t u = 0; u < nactv_; ++u) {
        size_t nu = actv_mos_[u];
        for (size_t v = 0; v < nactv_; ++v) {
            size_t nv = actv_mos_[v];
            for (size_t w = 0; w < nactv_; ++w) {
                size_t nw = actv_mos_[w];
                for (size_t x = 0; x < nactv_; ++x) {
                    size_t nx = actv_mos_[x];
                    for (size_t y = 0; y < nactv_; ++y) {
                        size_t ny = actv_mos_[y];
                        for (size_t z = 0; z < nactv_; ++z) {
                            size_t nz = actv_mos_[z];

                            double vaaa = 0.0, vaab = 0.0, vabb = 0.0, vbbb = 0.0;

                            // core
                            for (size_t m = 0; m < ncore_; ++m) {
                                size_t nm = core_mos_[m];

                                double d1 = Fa_[nm][nm] + Fa_[nw][nw] - Fa_[nx][nx] - Fa_[ny][ny];
                                double d2 = Fa_[nu][nu] + Fa_[nv][nv] - Fa_[nm][nm] - Fa_[nz][nz];
                                vaaa += 0.5 * integral_->aptei_aa(nu, nv, nm, nz) *
                                        integral_->aptei_aa(nm, nw, nx, ny) * d1 *
                                        srg_source_->compute_renormalized_denominator(d1 * d1 +
                                                                                      d2 * d2);

                                d1 = Fb_[nm][nm] + Fb_[nw][nw] - Fb_[nx][nx] - Fb_[ny][ny];
                                d2 = Fb_[nu][nu] + Fb_[nv][nv] - Fb_[nm][nm] - Fb_[nz][nz];
                                vbbb += 0.5 * integral_->aptei_bb(nu, nv, nm, nz) *
                                        integral_->aptei_bb(nm, nw, nx, ny) * d1 *
                                        srg_source_->compute_renormalized_denominator(d1 * d1 +
                                                                                      d2 * d2);

                                d1 = Fb_[nm][nm] + Fa_[nv][nv] - Fa_[nx][nx] - Fb_[nz][nz];
                                d2 = Fa_[nu][nu] + Fb_[nw][nw] - Fb_[nm][nm] - Fa_[ny][ny];
                                vaab += 2.0 * integral_->aptei_ab(nu, nw, ny, nm) *
                                        integral_->aptei_ab(nv, nm, nx, nz) * d1 *
                                        srg_source_->compute_renormalized_denominator(d1 * d1 +
                                                                                      d2 * d2);

                                d1 = Fa_[nm][nm] + Fb_[nw][nw] - Fa_[nx][nx] - Fb_[nz][nz];
                                d2 = Fa_[nu][nu] + Fa_[nv][nv] - Fa_[nm][nm] - Fa_[ny][ny];
                                vaab -= integral_->aptei_aa(nu, nv, nm, ny) *
                                        integral_->aptei_ab(nm, nw, nx, nz) * d1 *
                                        srg_source_->compute_renormalized_denominator(d1 * d1 +
                                                                                      d2 * d2);

                                d1 = Fa_[nm][nm] + Fa_[nv][nv] - Fa_[nx][nx] - Fa_[ny][ny];
                                d2 = Fa_[nu][nu] + Fb_[nw][nw] - Fa_[nm][nm] - Fb_[nz][nz];
                                vaab -= integral_->aptei_ab(nu, nw, nm, nz) *
                                        integral_->aptei_aa(nm, nv, nx, ny) * d1 *
                                        srg_source_->compute_renormalized_denominator(d1 * d1 +
                                                                                      d2 * d2);

                                d1 = Fa_[nm][nm] + Fb_[nw][nw] - Fa_[nx][nx] - Fb_[ny][ny];
                                d2 = Fa_[nu][nu] + Fb_[nv][nv] - Fa_[nm][nm] - Fb_[nz][nz];
                                vabb += 2.0 * integral_->aptei_ab(nu, nv, nm, nz) *
                                        integral_->aptei_ab(nm, nw, nx, ny) * d1 *
                                        srg_source_->compute_renormalized_denominator(d1 * d1 +
                                                                                      d2 * d2);

                                d1 = Fb_[nm][nm] + Fa_[nu][nu] - Fa_[nx][nx] - Fb_[ny][ny];
                                d2 = Fb_[nw][nw] + Fb_[nv][nv] - Fb_[nm][nm] - Fb_[nz][nz];
                                vabb -= integral_->aptei_bb(nv, nw, nm, nz) *
                                        integral_->aptei_ab(nu, nm, nx, ny) * d1 *
                                        srg_source_->compute_renormalized_denominator(d1 * d1 +
                                                                                      d2 * d2);

                                d1 = Fb_[nm][nm] + Fb_[nw][nw] - Fb_[nz][nz] - Fb_[ny][ny];
                                d2 = Fa_[nu][nu] + Fb_[nv][nv] - Fb_[nm][nm] - Fa_[nx][nx];
                                vabb -= integral_->aptei_ab(nu, nv, nx, nm) *
                                        integral_->aptei_bb(nm, nw, ny, nz) * d1 *
                                        srg_source_->compute_renormalized_denominator(d1 * d1 +
                                                                                      d2 * d2);
                            }

                            // virtual
                            for (size_t e = 0; e < nvirt_; ++e) {
                                size_t ne = virt_mos_[e];

                                double d1 = Fa_[nu][nu] + Fa_[nv][nv] - Fa_[ne][ne] - Fa_[nz][nz];
                                double d2 = Fa_[nw][nw] + Fa_[ne][ne] - Fa_[nx][nx] - Fa_[ny][ny];
                                vaaa += 0.5 * integral_->aptei_aa(nw, ne, nx, ny) *
                                        integral_->aptei_aa(nu, nv, ne, nz) * d1 *
                                        srg_source_->compute_renormalized_denominator(d1 * d1 +
                                                                                      d2 * d2);

                                d1 = Fb_[nu][nu] + Fb_[nv][nv] - Fb_[ne][ne] - Fb_[nz][nz];
                                d2 = Fb_[nw][nw] + Fb_[ne][ne] - Fb_[nx][nx] - Fb_[ny][ny];
                                vbbb += 0.5 * integral_->aptei_bb(nw, ne, nx, ny) *
                                        integral_->aptei_bb(nu, nv, ne, nz) * d1 *
                                        srg_source_->compute_renormalized_denominator(d1 * d1 +
                                                                                      d2 * d2);

                                d1 = Fa_[nu][nu] + Fb_[nw][nw] - Fa_[ny][ny] - Fb_[ne][ne];
                                d2 = Fa_[nv][nv] + Fb_[ne][ne] - Fa_[nx][nx] - Fb_[nz][nz];
                                vaab -= 2.0 * integral_->aptei_ab(nv, ne, nx, nz) *
                                        integral_->aptei_ab(nu, nw, ny, ne) * d1 *
                                        srg_source_->compute_renormalized_denominator(d1 * d1 +
                                                                                      d2 * d2);

                                d1 = Fa_[nu][nu] + Fa_[nv][nv] - Fa_[ne][ne] - Fa_[ny][ny];
                                d2 = Fa_[ne][ne] + Fb_[nw][nw] - Fa_[nx][nx] - Fb_[nz][nz];
                                vaab += integral_->aptei_ab(ne, nw, nx, nz) *
                                        integral_->aptei_aa(nu, nv, ne, ny) * d1 *
                                        srg_source_->compute_renormalized_denominator(d1 * d1 +
                                                                                      d2 * d2);

                                d1 = Fa_[nu][nu] + Fb_[nw][nw] - Fa_[ne][ne] - Fb_[nz][nz];
                                d2 = Fa_[nv][nv] + Fa_[ne][ne] - Fa_[nx][nx] - Fa_[ny][ny];
                                vaab -= integral_->aptei_aa(nv, ne, nx, ny) *
                                        integral_->aptei_ab(nu, nw, ne, nz) * d1 *
                                        srg_source_->compute_renormalized_denominator(d1 * d1 +
                                                                                      d2 * d2);

                                d1 = Fa_[nu][nu] + Fb_[nv][nv] - Fa_[ne][ne] - Fb_[nz][nz];
                                d2 = Fa_[ne][ne] + Fb_[nw][nw] - Fa_[nx][nx] - Fb_[ny][ny];
                                vabb -= 2.0 * integral_->aptei_ab(ne, nw, nx, ny) *
                                        integral_->aptei_ab(nu, nv, ne, nz) * d1 *
                                        srg_source_->compute_renormalized_denominator(d1 * d1 +
                                                                                      d2 * d2);

                                d1 = Fa_[nu][nu] + Fb_[nv][nv] - Fa_[nx][nx] - Fb_[ne][ne];
                                d2 = Fb_[nw][nw] + Fb_[ne][ne] - Fb_[ny][ny] - Fb_[nz][nz];
                                vabb -= integral_->aptei_bb(nw, ne, ny, nz) *
                                        integral_->aptei_ab(nu, nv, nx, ne) * d1 *
                                        srg_source_->compute_renormalized_denominator(d1 * d1 +
                                                                                      d2 * d2);

                                d1 = Fb_[nv][nv] + Fb_[nw][nw] - Fb_[ne][ne] - Fb_[nz][nz];
                                d2 = Fa_[nu][nu] + Fb_[ne][ne] - Fa_[nx][nx] - Fb_[ny][ny];
                                vabb += integral_->aptei_ab(nu, ne, nx, ny) *
                                        integral_->aptei_bb(nv, nw, ne, nz) * d1 *
                                        srg_source_->compute_renormalized_denominator(d1 * d1 +
                                                                                      d2 * d2);
                            }

                            E += vaaa * L3aaa_[x][y][z][u][v][w];
                            E += vaab * L3aab_[x][y][z][u][v][w];
                            E += vabb * L3abb_[x][y][z][u][v][w];
                            E += vbbb * L3bbb_[x][y][z][u][v][w];
                        }
                    }
                }
            }
        }
    }

    return E;
}

double MCSRGPT2_MO::compute_energy_srg() {
    // compute [1 - exp(-s * x^2)] / x^2
    srg_source_ = std::make_shared<LABS_SOURCE>(s_, taylor_threshold_);

    // no need to form amplitudes nor effective integrals
    // do need to reform the Fock matrix
    Fa_srg_ = d2(ncmo_, d1(ncmo_));
    Fb_srg_ = d2(ncmo_, d1(ncmo_));
    Form_Fock_SRG();

    //        // zero all acitve two-electron integrals
    //        for(size_t u = 0; u < na_; ++u){
    //            size_t nu = idx_a_[u];
    //            for(size_t v = 0; v < na_; ++v){
    //                size_t nv = idx_a_[v];
    //                for(size_t x = 0; x < na_; ++x){
    //                    size_t nx = idx_a_[x];
    //                    for(size_t y = 0; y < na_; ++y){
    //                        size_t ny = idx_a_[y];

    //                        integral_->set_tei(nu,nv,nx,ny,0.0,true,true);
    //                        integral_->set_tei(nu,nv,nx,ny,0.0,true,false);
    //                        integral_->set_tei(nu,nv,nx,ny,0.0,false,false);
    //                    }
    //                }
    //            }
    //        }

    outfile->Printf("\n");
    outfile->Printf("\n  Computing energy of [eta1, H1] ...");
    double Esrg_11 = ESRG_11();
    outfile->Printf("\t\t\t\t\tDone.");

    outfile->Printf("\n  Computing energy of [eta1, H2] ...");
    double Esrg_12 = ESRG_12();
    outfile->Printf("\t\t\t\t\tDone.");

    outfile->Printf("\n  Computing energy of [eta2, H1] ...");
    double Esrg_21 = ESRG_21();
    outfile->Printf("\t\t\t\t\tDone.");

    outfile->Printf("\n  Computing energy of [eta2, H2] C2 ...");
    double Esrg_22_2 = ESRG_22_2();
    outfile->Printf("\t\t\t\t\tDone.");

    outfile->Printf("\n  Computing energy of [eta2, H2] C4 ...");
    double Esrg_22_4 = ESRG_22_4();
    outfile->Printf("\t\t\t\t\tDone.");

    outfile->Printf("\n  Computing energy of [eta2, H2] C6 ...");
    double Esrg_22_6 = ESRG_22_6();
    outfile->Printf("\t\t\t\t\tDone.");

    double Esrg_22 = Esrg_22_2 + Esrg_22_4 + Esrg_22_6;
    double Ecorr = Esrg_11 + Esrg_12 + Esrg_21 + Esrg_22;
    double Etotal = Ecorr + Eref_;

    std::vector<std::pair<std::string, double>> energy;
    energy.push_back({"E0 (reference)", Eref_});
    energy.push_back({"[eta_1, H_1]", Esrg_11});
    energy.push_back({"[eta_1, H_2]", Esrg_12});
    energy.push_back({"[eta_2, H_1]", Esrg_21});
    energy.push_back({"[eta_2, H_2] C2", Esrg_22_2});
    energy.push_back({"[eta_2, H_2] C4", Esrg_22_4});
    energy.push_back({"[eta_2, H_2] C6", Esrg_22_6});
    energy.push_back({"[eta_2, H_2]", Esrg_22});
    energy.push_back({"SRG-MRPT2 correlation energy", Ecorr});
    energy.push_back({"SRG-MRPT2 total energy", Etotal});

    print_h2("SRG-MRPT2 energy summary");
    for (auto& str_dim : energy) {
        outfile->Printf("\n    %-30s = %22.15f", str_dim.first.c_str(), str_dim.second);
    }

    return Etotal;
}

void MCSRGPT2_MO::print_Fock(const std::string& spin, const d2& Fock) {
    std::string name = "Fock " + spin;
    outfile->Printf("  ==> %s <==\n\n", name.c_str());
    double econv = options_->get_double("E_CONVERGENCE");

    // print Fock block
    auto print_Fock_block = [&](const std::string& name1, const std::string& name2,
                                const std::vector<size_t>& idx1, const std::vector<size_t>& idx2) {
        size_t dim1 = idx1.size();
        size_t dim2 = idx2.size();
        std::string bname = name1 + "-" + name2;

        psi::Matrix F(bname, dim1, dim2);
        for (size_t i = 0; i < dim1; ++i) {
            size_t ni = idx1[i];
            for (size_t j = 0; j < dim2; ++j) {
                size_t nj = idx2[j];
                F.set(i, j, Fock[ni][nj]);
            }
        }

        F.print();

        if (dim1 != dim2) {
            std::string bnamer = name2 + "-" + name1;
            psi::Matrix Fr(bnamer, dim2, dim1);
            for (size_t i = 0; i < dim2; ++i) {
                size_t ni = idx2[i];
                for (size_t j = 0; j < dim1; ++j) {
                    size_t nj = idx1[j];
                    Fr.set(i, j, Fock[ni][nj]);
                }
            }

            psi::SharedMatrix FT = Fr.transpose();
            for (size_t i = 0; i < dim1; ++i) {
                for (size_t j = 0; j < dim2; ++j) {
                    double diff = FT->get(i, j) - F.get(i, j);
                    FT->set(i, j, diff);
                }
            }
            if (FT->rms() > 100.0 * econv) {
                outfile->Printf("  Warning: %s not symmetric for %s and %s blocks\n", name.c_str(),
                                bname.c_str(), bnamer.c_str());
                Fr.print();
            }
        }
    };

    // diagonal blocks
    print_Fock_block("C", "C", core_mos_, core_mos_);
    print_Fock_block("V", "V", virt_mos_, virt_mos_);
    print_Fock_block("A", "A", actv_mos_, actv_mos_);

    // off-diagonal blocks
    print_Fock_block("C", "A", core_mos_, actv_mos_);
    print_Fock_block("C", "V", core_mos_, virt_mos_);
    print_Fock_block("A", "V", actv_mos_, virt_mos_);
}

void MCSRGPT2_MO::Form_Fock(d2& A, d2& B) {
    timer_on("Form Fock");
    compute_Fock_ints();

    for (size_t p = 0; p < ncmo_; ++p) {
        for (size_t q = 0; q < ncmo_; ++q) {
            A[p][q] = integral_->get_fock_a(p, q);
            B[p][q] = integral_->get_fock_b(p, q);
        }
    }
    timer_off("Form Fock");

    if (print_ > 1) {
        print_Fock("Alpha", A);
        print_Fock("Beta", B);
    }
}

void MCSRGPT2_MO::compute_Fock_ints() {
    local_timer tfock;
    outfile->Printf("\n  %-35s ...", "Forming generalized Fock matrix");

    integral_->make_fock_matrix(reference_.g1a(), reference_.g1b());

    outfile->Printf("  Done. Timing %15.6f s", tfock.get());
}

void MCSRGPT2_MO::fill_naive_cumulants(RDMs ref, const int level) {
    // fill in 1-cumulant (same as 1-RDM) to D1a_, D1b_
    ambit::Tensor L1a = ref.g1a();
    ambit::Tensor L1b = ref.g1b();
    fill_one_cumulant(L1a, L1b);
    if (print_ > 1) {
        print_density("Alpha", Da_);
        print_density("Beta", Db_);
    }

    // fill in 2-cumulant to L2aa_, L2ab_, L2bb_
    if (level >= 2) {
        ambit::Tensor L2aa = ref.L2aa();
        ambit::Tensor L2ab = ref.L2ab();
        ambit::Tensor L2bb = ref.L2bb();
        fill_two_cumulant(L2aa, L2ab, L2bb);
        if (print_ > 2) {
            print2PDC("L2aa", L2aa_, print_);
            print2PDC("L2ab", L2ab_, print_);
            print2PDC("L2bb", L2bb_, print_);
        }
    }

    // fill in 3-cumulant to L3aaa_, L3aab_, L3abb_, L3bbb_
    if (level >= 3) {
        ambit::Tensor L3aaa = ref.L3aaa();
        ambit::Tensor L3aab = ref.L3aab();
        ambit::Tensor L3abb = ref.L3abb();
        ambit::Tensor L3bbb = ref.L3bbb();
        fill_three_cumulant(L3aaa, L3aab, L3abb, L3bbb);
        if (print_ > 3) {
            print3PDC("L3aaa", L3aaa_, print_);
            print3PDC("L3aab", L3aab_, print_);
            print3PDC("L3abb", L3abb_, print_);
            print3PDC("L3bbb", L3bbb_, print_);
        }
    }
}

void MCSRGPT2_MO::fill_one_cumulant(ambit::Tensor& L1a, ambit::Tensor& L1b) {
    Da_ = d2(ncmo_, d1(ncmo_));
    Db_ = d2(ncmo_, d1(ncmo_));

    for (size_t p = 0; p < ncore_; ++p) {
        size_t np = core_mos_[p];
        Da_[np][np] = 1.0;
        Db_[np][np] = 1.0;
    }

    std::vector<double>& opdc_a = L1a.data();
    std::vector<double>& opdc_b = L1b.data();

    // TODO: try omp here
    for (size_t p = 0; p < nactv_; ++p) {
        size_t np = actv_mos_[p];
        for (size_t q = p; q < nactv_; ++q) {
            size_t nq = actv_mos_[q];

            if ((sym_actv_[p] ^ sym_actv_[q]) != 0)
                continue;

            size_t index = p * nactv_ + q;
            Da_[np][nq] = opdc_a[index];
            Db_[np][nq] = opdc_b[index];

            Da_[nq][np] = Da_[np][nq];
            Db_[nq][np] = Db_[np][nq];
        }
    }
}

void MCSRGPT2_MO::fill_two_cumulant(ambit::Tensor& L2aa, ambit::Tensor& L2ab, ambit::Tensor& L2bb) {
    L2aa_ = d4(nactv_, d3(nactv_, d2(nactv_, d1(nactv_))));
    L2ab_ = d4(nactv_, d3(nactv_, d2(nactv_, d1(nactv_))));
    L2bb_ = d4(nactv_, d3(nactv_, d2(nactv_, d1(nactv_))));

    std::vector<double>& tpdc_aa = L2aa.data();
    std::vector<double>& tpdc_ab = L2ab.data();
    std::vector<double>& tpdc_bb = L2bb.data();

    size_t dim2 = nactv_ * nactv_;
    size_t dim3 = nactv_ * dim2;

    // TODO: try omp here
    for (size_t p = 0; p < nactv_; ++p) {
        for (size_t q = 0; q < nactv_; ++q) {
            for (size_t r = 0; r < nactv_; ++r) {
                for (size_t s = 0; s < nactv_; ++s) {

                    if ((sym_actv_[p] ^ sym_actv_[q] ^ sym_actv_[r] ^ sym_actv_[s]) != 0)
                        continue;

                    size_t index = p * dim3 + q * dim2 + r * nactv_ + s;

                    L2aa_[p][q][r][s] = tpdc_aa[index];
                    L2ab_[p][q][r][s] = tpdc_ab[index];
                    L2bb_[p][q][r][s] = tpdc_bb[index];
                }
            }
        }
    }
}

void MCSRGPT2_MO::fill_three_cumulant(ambit::Tensor& L3aaa, ambit::Tensor& L3aab,
                                      ambit::Tensor& L3abb, ambit::Tensor& L3bbb) {
    L3aaa_ = d6(nactv_, d5(nactv_, d4(nactv_, d3(nactv_, d2(nactv_, d1(nactv_))))));
    L3aab_ = d6(nactv_, d5(nactv_, d4(nactv_, d3(nactv_, d2(nactv_, d1(nactv_))))));
    L3abb_ = d6(nactv_, d5(nactv_, d4(nactv_, d3(nactv_, d2(nactv_, d1(nactv_))))));
    L3bbb_ = d6(nactv_, d5(nactv_, d4(nactv_, d3(nactv_, d2(nactv_, d1(nactv_))))));

    size_t dim2 = nactv_ * nactv_;
    size_t dim3 = nactv_ * dim2;
    size_t dim4 = nactv_ * dim3;
    size_t dim5 = nactv_ * dim4;

    auto fill = [&](d6& L3, ambit::Tensor& L3t) {
        std::vector<double>& data = L3t.data();

        // TODO: try omp here
        for (size_t p = 0; p != nactv_; ++p) {
            for (size_t q = 0; q != nactv_; ++q) {
                for (size_t r = 0; r != nactv_; ++r) {
                    for (size_t s = 0; s != nactv_; ++s) {
                        for (size_t t = 0; t != nactv_; ++t) {
                            for (size_t u = 0; u != nactv_; ++u) {

                                if ((sym_actv_[p] ^ sym_actv_[q] ^ sym_actv_[r] ^ sym_actv_[s] ^
                                     sym_actv_[t] ^ sym_actv_[u]) != 0)
                                    continue;

                                size_t index =
                                    p * dim5 + q * dim4 + r * dim3 + s * dim2 + t * nactv_ + u;

                                L3[p][q][r][s][t][u] = data[index];
                            }
                        }
                    }
                }
            }
        }
    };

    fill(L3aaa_, L3aaa);
    fill(L3aab_, L3aab);
    fill(L3abb_, L3abb);
    fill(L3bbb_, L3bbb);
}

void MCSRGPT2_MO::print_density(const std::string& spin, const d2& density) {
    std::string name = "Density " + spin;
    outfile->Printf("  ==> %s <==\n\n", name.c_str());

    psi::SharedMatrix dens(new psi::Matrix("A-A", nactv_, nactv_));
    for (size_t u = 0; u < nactv_; ++u) {
        size_t nu = actv_mos_[u];
        for (size_t v = 0; v < nactv_; ++v) {
            size_t nv = actv_mos_[v];
            dens->set(u, v, density[nu][nv]);
        }
    }

    dens->print();
}

void MCSRGPT2_MO::print2PDC(const std::string& str, const d4& TwoPDC, const int& PRINT) {
    timer_on("PRINT 2-Cumulant");
    outfile->Printf("\n  ** %s **", str.c_str());
    size_t count = 0;
    size_t size = TwoPDC.size();
    for (size_t i = 0; i != size; ++i) {
        for (size_t j = 0; j != size; ++j) {
            for (size_t k = 0; k != size; ++k) {
                for (size_t l = 0; l != size; ++l) {
                    if (std::fabs(TwoPDC[i][j][k][l]) > 1.0e-15) {
                        ++count;
                        if (PRINT > 2)
                            outfile->Printf("\n  Lambda "
                                            "[%3lu][%3lu][%3lu][%3lu] = "
                                            "%18.15lf",
                                            i, j, k, l, TwoPDC[i][j][k][l]);
                    }
                }
            }
        }
    }
    outfile->Printf("\n");
    outfile->Printf("\n  Number of Nonzero Elements: %zu", count);
    outfile->Printf("\n");
    timer_off("PRINT 2-Cumulant");
}

void MCSRGPT2_MO::print3PDC(const std::string& str, const d6& ThreePDC, const int& PRINT) {
    timer_on("PRINT 3-Cumulant");
    outfile->Printf("\n  ** %s **", str.c_str());
    size_t count = 0;
    size_t size = ThreePDC.size();
    for (size_t i = 0; i != size; ++i) {
        for (size_t j = 0; j != size; ++j) {
            for (size_t k = 0; k != size; ++k) {
                for (size_t l = 0; l != size; ++l) {
                    for (size_t m = 0; m != size; ++m) {
                        for (size_t n = 0; n != size; ++n) {
                            if (std::fabs(ThreePDC[i][j][k][l][m][n]) > 1.0e-15) {
                                ++count;
                                if (PRINT > 3)
                                    outfile->Printf("\n  Lambda "
                                                    "[%3lu][%3lu][%3lu][%3lu][%"
                                                    "3lu][%3lu] = %18.15lf",
                                                    i, j, k, l, m, n, ThreePDC[i][j][k][l][m][n]);
                            }
                        }
                    }
                }
            }
        }
    }
    outfile->Printf("\n");
    outfile->Printf("\n  Number of Nonzero Elements: %zu", count);
    outfile->Printf("\n");
    timer_off("PRINT 3-Cumulant");
}

} // namespace forte
