/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2024 by its authors (see COPYING, COPYING.LESSER,
 * AUTHORS).
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

#include "psi4/psi4-dec.h"
#include "psi4/libpsi4util/PsiOutStream.h"
#include "psi4/libpsi4util/process.h"

#include "base_classes/forte_options.h"
#include "helpers/printing.h"
#include "helpers/helpers.h"
#include "post_process/post_process.h"

using namespace psi;

namespace forte {

PostProcess::PostProcess(const std::string method, std::shared_ptr<RDMs> rdms, std::shared_ptr<ForteOptions> options,
                  std::shared_ptr<MOSpaceInfo> mo_space_info,
                  std::shared_ptr<ForteIntegrals> ints,
                  std::shared_ptr<ActiveSpaceIntegrals> as_ints)
    : method_(method), rdms_(rdms), options_(options), mo_space_info_(mo_space_info), ints_(ints), as_ints_(as_ints) {

    nactpi_ = mo_space_info_->dimension("ACTIVE");
    nirrep_ = nactpi_.n();
    nact_ = nactpi_.sum();

    print_h1("Wavefunction Post-processing");

}

void PostProcess::process() {

	if (method_ == "SPIN_CORRELATION"){
		spin_analysis();
	}else if (method_ == "UNPAIRED_DENSITY"){
        unpaired_density();
	}

}


std::tuple<std::shared_ptr<psi::Matrix>, std::shared_ptr<psi::Matrix>, std::shared_ptr<psi::Vector>,std::shared_ptr<psi::Vector>> PostProcess::compute_active_nos() {

    print_h2("Natural Orbitals");

    auto g1a = rdms_->g1a();
    auto g1b = rdms_->g1b();

    std::shared_ptr<psi::Matrix> Ua, Ub;

    psi::Dimension nmopi = mo_space_info_->dimension("ALL");
    psi::Dimension fdocc = mo_space_info_->dimension("FROZEN_DOCC");
    psi::Dimension rdocc = mo_space_info_->dimension("RESTRICTED_DOCC");

    auto opdm_a = std::make_shared<psi::Matrix>("OPDM_A", nirrep_, nactpi_, nactpi_);
    auto opdm_b = std::make_shared<psi::Matrix>("OPDM_B", nirrep_, nactpi_, nactpi_);

    int offset = 0;
    for (size_t h = 0; h < nirrep_; h++) {
        for (int u = 0; u < nactpi_[h]; u++) {
            for (int v = 0; v < nactpi_[h]; v++) {
                opdm_a->set(h, u, v, g1a.data()[(u + offset) * nact_ + v + offset]);
                opdm_b->set(h, u, v, g1b.data()[(u + offset) * nact_ + v + offset]);
            }
        }
        offset += nactpi_[h];
    }

    auto OCC_A = std::make_shared<Vector>("ALPHA OCCUPATION", nactpi_);
    auto OCC_B = std::make_shared<Vector>("BETA OCCUPATION", nactpi_);
    auto NO_A = std::make_shared<psi::Matrix>(nirrep_, nactpi_, nactpi_);
    auto NO_B = std::make_shared<psi::Matrix>(nirrep_, nactpi_, nactpi_);

    opdm_a->diagonalize(NO_A, OCC_A, descending);
    opdm_b->diagonalize(NO_B, OCC_B, descending);

    // Build full transformation matrices from e-vecs
    Ua = std::make_shared<psi::Matrix>("Ua", nmopi, nmopi);
    Ub = std::make_shared<psi::Matrix>("Ub", nmopi, nmopi);

    Ua->identity();
    Ub->identity();

    for (size_t h = 0; h < nirrep_; ++h) {
        size_t irrep_offset = 0;

        // Frozen core and Restricted docc are unchanged
        irrep_offset += fdocc[h] + rdocc[h];
        // Only change the active block
        for (int p = 0; p < nactpi_[h]; ++p) {
            for (int q = 0; q < nactpi_[h]; ++q) {
                Ua->set(h, p + irrep_offset, q + irrep_offset, NO_A->get(h, p, q));
                Ub->set(h, p + irrep_offset, q + irrep_offset, NO_B->get(h, p, q));
            }
        }
    }
    return std::make_tuple(Ua, Ub, OCC_A,OCC_B);
}

void PostProcess::spin_analysis() {
    size_t nact = static_cast<unsigned long>(nact_);
    size_t nact2 = nact * nact;
    size_t nact3 = nact * nact2;

    auto UA = std::make_shared<psi::Matrix>(nact, nact);
    auto UB = std::make_shared<psi::Matrix>(nact, nact);

    if (options_->get_str("SPIN_BASIS") == "NO") {
        outfile->Printf("\n  Computing spin correlation in NO basis \n");

        auto no_tuple = compute_active_nos();
        std::tie(UA, UB,std::ignore,std::ignore) = no_tuple; 

        int nmo = mo_space_info_->size("ALL");

        auto Ua_full = std::make_shared<psi::Matrix>(nmo, nmo);
        auto Ub_full = std::make_shared<psi::Matrix>(nmo, nmo);

        Ua_full->identity();
        Ub_full->identity();

        auto actpi = mo_space_info_->absolute_mo("ACTIVE");
        for (size_t h = 0; h < nirrep_; ++h) {
            // skip frozen/restricted docc
            int nact_i = nactpi_[h];
            for (int i = 0; i < nact_i; ++i) {
                for (int j = 0; j < nact_i; ++j) {
                    Ua_full->set(actpi[i], actpi[j], UA->get(i, j));
                    Ub_full->set(actpi[i], actpi[j], UB->get(i, j));
                }
            }
        }

        auto CA = as_ints_->ints()->Ca()->clone();
        auto CB = as_ints_->ints()->Cb()->clone();

        auto Ca_new = psi::linalg::doublet(CA, Ua_full, false, false);
        auto Cb_new = psi::linalg::doublet(CB, Ub_full, false, false);

        as_ints_->ints()->update_orbitals(Ca_new, Cb_new, false);

    } else if (options_->get_str("SPIN_BASIS") == "LOCAL") {
        outfile->Printf("\n  Computing spin correlation in local basis \n");

        auto loc = std::make_shared<Localize>(options_, as_ints_->ints(), mo_space_info_);

        std::vector<size_t> actmo = mo_space_info_->absolute_mo("ACTIVE");
        std::vector<int> loc_mo(2);
        loc_mo[0] = static_cast<int>(actmo[0]);
        loc_mo[1] = static_cast<int>(actmo.back());
        loc->set_orbital_space(loc_mo);
        loc->compute_transformation();
        UA = loc->get_Ua()->clone();
        UB = loc->get_Ub()->clone();

    } else if (options_->get_str("SPIN_BASIS") == "CANONICAL") {
        outfile->Printf("\n  Computing spin correlation in reference basis \n");
        UA->identity();
        UB->identity();
    }
    ambit::Tensor Ua = ambit::Tensor::build(ambit::CoreTensor, "U", {nact, nact});
    ambit::Tensor Ub = ambit::Tensor::build(ambit::CoreTensor, "U", {nact, nact});
    Ua.iterate([&](const std::vector<size_t>& i, double& value) { value = UA->get(i[0], i[1]); });
    Ub.iterate([&](const std::vector<size_t>& i, double& value) { value = UB->get(i[0], i[1]); });

    // rotate 1- and 2-RDMs to desired basis
    auto L1aT = ambit::Tensor::build(ambit::CoreTensor, "Transformed L1a", {nact, nact});
    auto L1bT = ambit::Tensor::build(ambit::CoreTensor, "Transformed L1b", {nact, nact});

    auto ordm_a = rdms_->g1a();
    auto ordm_b = rdms_->g1b();
    L1aT("pq") = Ua("ap") * ordm_a("ab") * Ua("bq");
    L1bT("pq") = Ub("ap") * ordm_b("ab") * Ub("bq");

    std::vector<size_t> dim4{nact, nact, nact, nact};
    auto L2aaT = ambit::Tensor::build(ambit::CoreTensor, "Transformed L2aa", dim4);
    auto L2abT = ambit::Tensor::build(ambit::CoreTensor, "Transformed L2ab", dim4);
    auto L2bbT = ambit::Tensor::build(ambit::CoreTensor, "Transformed L2bb", dim4);

    auto trdm_aa = rdms_->g2aa();
    auto trdm_ab = rdms_->g2ab();
    auto trdm_bb = rdms_->g2bb();
    L2aaT("pqrs") = Ua("ap") * Ua("bq") * trdm_aa("abcd") * Ua("cr") * Ua("ds");
    L2abT("pqrs") = Ua("ap") * Ub("bq") * trdm_ab("abcd") * Ua("cr") * Ub("ds");
    L2bbT("pqrs") = Ub("ap") * Ub("bq") * trdm_bb("abcd") * Ub("cr") * Ub("ds");

    // Now form the spin correlation
    auto spin_corr = std::make_shared<psi::Matrix>("Spin Correlation", nact, nact);
    auto spin_fluct = std::make_shared<psi::Matrix>("Spin Fluctuation", nact, nact);
    auto spin_z = std::make_shared<psi::Matrix>("Spin-z Correlation", nact, nact);

    const auto& l1a = L1aT.data();
    const auto& l1b = L1bT.data();
    const auto& l2aa = L2aaT.data();
    const auto& l2ab = L2abT.data();
    const auto& l2bb = L2bbT.data();
    for (size_t i = 0; i < nact; ++i) {
        for (size_t j = 0; j < nact; ++j) {
            double value = (l2aa[i * nact3 + j * nact2 + i * nact + j] +
                            l2bb[i * nact3 + j * nact2 + i * nact + j] -
                            l2ab[i * nact3 + j * nact2 + i * nact + j] -
                            l2ab[j * nact3 + i * nact2 + j * nact + i]);
            if (i == j) {
                value += (l1a[nact * i + j] + l1b[nact * i + j]);
            }

            value +=
                (l1a[nact * i + i] - l1b[nact * i + i]) * (l1a[nact * j + j] - l1b[nact * j + j]);

            spin_z->set(i, j, value);
        }
    }

    for (size_t i = 0; i < nact; ++i) {
        for (size_t j = 0; j < nact; ++j) {
            double value = 0.0;
            if (i == j) {
                value += 0.75 * (l1a[nact * i + j] + l1b[nact * i + j]);
            }
            value -= 0.5 * (l2ab[i * nact3 + j * nact2 + j * nact + i] +
                            l2ab[j * nact3 + i * nact2 + i * nact + j]);

            value += 0.25 * (l2aa[i * nact3 + j * nact2 + i * nact + j] +
                             l2bb[i * nact3 + j * nact2 + i * nact + j] -
                             l2ab[i * nact3 + j * nact2 + i * nact + j] -
                             l2ab[j * nact3 + i * nact2 + j * nact + i]);

            spin_corr->set(i, j, value);
            value -=
                0.25 *
                (l1a[i * nact + i] * l1a[j * nact + j] + l1b[i * nact + i] * l1b[j * nact + j] -
                 l1a[i * nact + i] * l1b[j * nact + j] - l1b[i * nact + i] * l1a[j * nact + j]);
            spin_fluct->set(i, j, value);
        }
    }

    outfile->Printf("\n");
    // spin_corr->print();
    spin_fluct->print();
    spin_z->print();
    auto spin_evecs = std::make_shared<psi::Matrix>(nact, nact);
    auto spin_evals = std::make_shared<psi::Vector>(nact);
    auto spin_evecs2 = std::make_shared<psi::Matrix>(nact, nact);
    auto spin_evals2 = std::make_shared<psi::Vector>(nact);

    //    spin_corr->diagonalize(spin_evecs, spin_evals);
    //    spin_evals->print();

    if (options_->get_bool("SPIN_MAT_TO_FILE")) {
        std::ofstream file;
        file.open("spin_mat.txt", std::ofstream::out | std::ofstream::trunc);
        for (size_t i = 0; i < nact; ++i) {
            for (size_t j = 0; j < nact; ++j) {
                file << std::setw(12) << std::setprecision(6) << spin_corr->get(i, j) << " ";
            }
            file << "\n";
        }
        file.close();
        std::ofstream file2;
        file.open("spin_fluct.txt", std::ofstream::out | std::ofstream::trunc);
        for (size_t i = 0; i < nact; ++i) {
            for (size_t j = 0; j < nact; ++j) {
                file << std::setw(12) << std::setprecision(6) << spin_fluct->get(i, j) << " ";
            }
            file2 << "\n";
        }
        file2.close();
    }
    /*
        // Build spin-correlation densities
        auto Ca = reference_wavefunction_->Ca();
        psi::Dimension nactpi = mo_space_info_->get_dimension("ACTIVE");
        std::vector<size_t> actpi = mo_space_info_->get_absolute_mo("ACTIVE");
        auto Ca_copy = Ca->clone();
        for( int i = 0; i < nact; ++i ){
            auto vec = std::make_shared<Vector>(nmo_);
            vec->zero();
            for( int j = 0; j < nact; ++j ){
                auto col = Ca_copy->get_column(0,actpi[j]);
                double spin = spin_corr->get(j,i);
                for( int k = 0; k < nmo_; ++k ){
                    double val = col->get(k) * col->get(k);
                    col->set(k, val);
                }
                col->scale(spin);
                vec->add(col);
           }
            Ca->set_column(0,actpi[i], vec);
        }
    */
    if (options_->get_bool("SPIN_TEST")) {
        // make a test
        double value = 0.0;
        for (size_t i = 0; i < nact; ++i) {
            for (size_t j = i; j < nact; ++j) {
                value += spin_fluct->get(i, j);
            }
        }

        psi::Process::environment.globals["SPIN CORRELATION TEST"] = value;
    }
}

void PostProcess::unpaired_density() {

    outfile->Printf("\n Computing unpaired electron numbers/density in NO basis.");
    size_t nact = static_cast<unsigned long>(nact_);

    auto UA = std::make_shared<psi::Matrix>(nact, nact);
    auto UB = std::make_shared<psi::Matrix>(nact, nact);
    auto OCC_A = std::make_shared<psi::Vector>(nact);
    auto OCC_B = std::make_shared<psi::Vector>(nact);

    // Get natural orbitals
    auto no_tuple = compute_active_nos();
    std::tie(UA, UB,OCC_A,OCC_B) = no_tuple; 

    std::vector<double> nus(nact);

    outfile->Printf("\n    N.O.      occ        nu_i  ");
    outfile->Printf("\n  --------  ---------  --------");

    // the total unpaired electrons (alpha only)
    double nu_total = 0.0;
    for (size_t i = 0; i < nact; ++i){
        double ni = OCC_A->get(i);// + OCC_B->get(i);
        //double nu_i = (ni*ni)*(1.0-ni)*(1.0-ni);
        double nu_i = (ni)*(1.0-ni);
        nus[i] = nu_i;
        nu_total += nu_i;
        outfile->Printf("\n    %2d  \t %7.5f   \t %7.5f ", i, ni,nu_i);
    }

    outfile->Printf("\n  total: %5.3f", nu_total);

    // Plotting of unpaired densities should be handled externally
    // Here, I'll put the unpaired numbers in a file and
    // transform the orbitals, so the plots can be made
    std::ofstream ofile;
    ofile.open("unpaired_electrons_alpha.txt", std::ofstream::out | std::ofstream::trunc);
    for (size_t i = 0; i < nact; ++i) {
        ofile << std::setprecision(12) << nus[i]  << "\n";
    }
    ofile.close();

    ints_->rotate_orbitals(UA,UB);
}


void perform_post_processing(const std::string method, std::shared_ptr<RDMs> rdms, std::shared_ptr<ForteOptions> options,
                           std::shared_ptr<MOSpaceInfo> mo_space_info,
                           std::shared_ptr<ForteIntegrals> ints,
                           std::shared_ptr<ActiveSpaceIntegrals> as_ints) {
    PostProcess proc(method, rdms, options, mo_space_info,ints, as_ints);
    proc.process();
}

} // namespace forte
