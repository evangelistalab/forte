/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2020 by its authors (see COPYING, COPYING.LESSER, AUTHORS).
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

#include <cmath>
#include <fstream>

#include "psi4/libdpd/dpd.h"
#include "psi4/libmints/mintshelper.h"
#include "psi4/libmints/matrix.h"
#include "psi4/libpsio/psio.hpp"
#include "psi4/psi4-dec.h"
#include "psi4/psifiles.h"

#include "base_classes/mo_space_info.h"
#include "helpers/blockedtensorfactory.h"
#include "helpers/string_algorithms.h"

#include "custom_integrals.h"

#define IOFFINDEX(i) (i * (i + 1) / 2)
#define PAIRINDEX(i, j) ((i > j) ? (IOFFINDEX(i) + (j)) : (IOFFINDEX(j) + (i)))
#define four(i, j, k, l) PAIRINDEX(PAIRINDEX(i, j), PAIRINDEX(k, l))

using namespace psi;

namespace forte {

/**
 * @brief CustomIntegrals::CustomIntegrals
 * @param options - psi options class
 * @param restricted - type of integral transformation
 * @param resort_frozen_core -
 */
CustomIntegrals::CustomIntegrals(std::shared_ptr<ForteOptions> options,
                                 std::shared_ptr<psi::Wavefunction> ref_wfn,
                                 std::shared_ptr<MOSpaceInfo> mo_space_info,
                                 IntegralSpinRestriction restricted)
    : ForteIntegrals(options, ref_wfn, mo_space_info, restricted) {
    integral_type_ = Custom;
    print_info();
    outfile->Printf("\n  Using Custom integrals\n\n");

    gather_integrals();

    freeze_core_orbitals();
}

double CustomIntegrals::aptei_aa(size_t p, size_t q, size_t r, size_t s) {
    return aphys_tei_aa[aptei_index(p, q, r, s)];
}

double CustomIntegrals::aptei_ab(size_t p, size_t q, size_t r, size_t s) {
    return aphys_tei_ab[aptei_index(p, q, r, s)];
}

double CustomIntegrals::aptei_bb(size_t p, size_t q, size_t r, size_t s) {
    return aphys_tei_bb[aptei_index(p, q, r, s)];
}

ambit::Tensor CustomIntegrals::aptei_aa_block(const std::vector<size_t>& p,
                                              const std::vector<size_t>& q,
                                              const std::vector<size_t>& r,
                                              const std::vector<size_t>& s) {
    ambit::Tensor ReturnTensor =
        ambit::Tensor::build(tensor_type_, "Return", {p.size(), q.size(), r.size(), s.size()});
    ReturnTensor.iterate([&](const std::vector<size_t>& i, double& value) {
        value = aptei_aa(p[i[0]], q[i[1]], r[i[2]], s[i[3]]);
    });
    return ReturnTensor;
}

ambit::Tensor CustomIntegrals::aptei_ab_block(const std::vector<size_t>& p,
                                              const std::vector<size_t>& q,
                                              const std::vector<size_t>& r,
                                              const std::vector<size_t>& s) {
    ambit::Tensor ReturnTensor =
        ambit::Tensor::build(tensor_type_, "Return", {p.size(), q.size(), r.size(), s.size()});
    ReturnTensor.iterate([&](const std::vector<size_t>& i, double& value) {
        value = aptei_ab(p[i[0]], q[i[1]], r[i[2]], s[i[3]]);
    });
    return ReturnTensor;
}

ambit::Tensor CustomIntegrals::aptei_bb_block(const std::vector<size_t>& p,
                                              const std::vector<size_t>& q,
                                              const std::vector<size_t>& r,
                                              const std::vector<size_t>& s) {
    ambit::Tensor ReturnTensor =
        ambit::Tensor::build(tensor_type_, "Return", {p.size(), q.size(), r.size(), s.size()});
    ReturnTensor.iterate([&](const std::vector<size_t>& i, double& value) {
        value = aptei_bb(p[i[0]], q[i[1]], r[i[2]], s[i[3]]);
    });
    return ReturnTensor;
}

ambit::Tensor CustomIntegrals::three_integral_block(const std::vector<size_t>&,
                                                    const std::vector<size_t>&,
                                                    const std::vector<size_t>&) {
    outfile->Printf("\n Oh no!, you tried to grab a ThreeIntegral but this "
                    "is not there!!");
    throw psi::PSIEXCEPTION("INT_TYPE=DF/CHOLESKY to use ThreeIntegral");
}

ambit::Tensor CustomIntegrals::three_integral_block_two_index(const std::vector<size_t>&, size_t,
                                                              const std::vector<size_t>&) {
    outfile->Printf("\n Oh no! this isn't here");
    throw psi::PSIEXCEPTION("INT_TYPE=DISKDF");
}

double** CustomIntegrals::three_integral_pointer() {
    outfile->Printf("\n Doh! There is no Three_integral here.  Use DF/CD");
    throw psi::PSIEXCEPTION("INT_TYPE=DF/CHOLESKY to use ThreeIntegral!");
}

void CustomIntegrals::set_tei(size_t p, size_t q, size_t r, size_t s, double value, bool alpha1,
                              bool alpha2) {
    size_t index = aptei_index(p, q, r, s);
    if (alpha1 == true and alpha2 == true)
        aphys_tei_aa[index] = value;
    if (alpha1 == true and alpha2 == false)
        aphys_tei_ab[index] = value;
    if (alpha1 == false and alpha2 == false)
        aphys_tei_bb[index] = value;
}

void CustomIntegrals::gather_integrals() {
    // Read the integrals from a file
    std::string filename("INTDUMP");
    std::ifstream file(filename);

    if (not file.is_open()) {
    }
    std::string str((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());

    std::vector<std::string> lines = split_string(str, "\n");

    std::string open_tag("&FCI");
    std::string close_tag("&END");

    int nelec = 0;
    int norb = 0;
    int ms2 = 0;
    std::vector<int> orbsym;
    std::vector<double> two_electron_integrals_chemist;

    bool parsing_section = false;
    for (const auto& line : lines) {
        outfile->Printf("\n%s", line.c_str());
        if (line.find(close_tag) != std::string::npos) {
            parsing_section = false;
            // now we know how many orbitals are there and we can allocate memory for the one- and
            // two-electron integrals
            custom_integrals_allocate(norb, orbsym);
            two_electron_integrals_chemist.assign(num_tei_, 0.0);
        } else if (line.find(open_tag) != std::string::npos) {
            parsing_section = true;
        } else {
            if (parsing_section) {
                std::vector<std::string> split_line = split_string(line, "=");
                if (split_line[0] == "NORB") {
                    split_line[1].pop_back();
                    norb = stoi(split_line[1]);
                }
                if (split_line[0] == "NELEC") {
                    split_line[1].pop_back();
                    nelec = stoi(split_line[1]);
                }
                if (split_line[0] == "MS2") {
                    split_line[1].pop_back();
                    ms2 = stoi(split_line[1]);
                }
                if (split_line[0] == "ORBSYM") {
                    split_line[1].pop_back();
                    std::vector<std::string> vals = split_string(split_line[1], ",");
                    for (const auto& val : vals) {
                        orbsym.push_back(stoi(val));
                    }
                }
            } else {
                if (line.size() > 10) {
                    std::vector<std::string> split_line = split_string(line, " ");
                    double integral = stoi(split_line[0]);
                    int p = stoi(split_line[1]);
                    int q = stoi(split_line[2]);
                    int r = stoi(split_line[3]);
                    int s = stoi(split_line[4]);

                    if (q == 0) {
                        // orbital energies, skip them
                    } else if ((r == 0) and (s == 0)) {
                        // one-electron integrals
                        full_one_electron_integrals_a_[(p - 1) * aptei_idx_ + q - 1] = integral;
                        full_one_electron_integrals_b_[(p - 1) * aptei_idx_ + q - 1] = integral;
                        full_one_electron_integrals_a_[(q - 1) * aptei_idx_ + p - 1] = integral;
                        full_one_electron_integrals_b_[(q - 1) * aptei_idx_ + p - 1] = integral;
                        one_electron_integrals_a_[(p - 1) * aptei_idx_ + q - 1] = integral;
                        one_electron_integrals_b_[(p - 1) * aptei_idx_ + q - 1] = integral;
                        one_electron_integrals_a_[(q - 1) * aptei_idx_ + p - 1] = integral;
                        one_electron_integrals_b_[(q - 1) * aptei_idx_ + p - 1] = integral;
                    } else {
                        // two-electron integrals
                        two_electron_integrals_chemist[four(p, q, r, s)] = integral;
                    }
                }
            }
        }
    }

    // Store the integrals
    for (size_t p = 0; p < nmo_; ++p) {
        for (size_t q = 0; q < nmo_; ++q) {
            for (size_t r = 0; r < nmo_; ++r) {
                for (size_t s = 0; s < nmo_; ++s) {
                    // <pq||rs> = <pq|rs> - <pq|sr> = (pr|qs) - (ps|qr)
                    double direct = two_electron_integrals_chemist[INDEX4(p, r, q, s)];
                    double exchange = two_electron_integrals_chemist[INDEX4(p, s, q, r)];
                    size_t index = aptei_index(p, q, r, s);
                    aphys_tei_aa[index] = direct - exchange;
                    aphys_tei_ab[index] = direct;
                    aphys_tei_bb[index] = direct - exchange;
                }
            }
        }
    }

    //    std::string s(std::istreambuf_iterator<char>(file >> std::skipws),
    //                   std::istreambuf_iterator<char>());

    //    std::copy(std::istream_iterator<std::string>(file),
    //              std::istream_iterator<std::string>(),
    //              std::back_inserter(lines));

    //    outfile->Printf("%s",s.c_str());

    //    for (size_t p = 0; p < nmo_; ++p) {
    //        for (size_t q = 0; q < nmo_; ++q) {
    //            one_electron_integrals_a_[p * nmo_ + q] = 0.0;
    //            one_electron_integrals_b_[p * nmo_ + q] = 0.0;
    //        }
    //    }

    //    for (size_t pqrs = 0; pqrs < num_aptei_; ++pqrs)
    //        aphys_tei_aa[pqrs] = 0.0;
    //    for (size_t pqrs = 0; pqrs < num_aptei_; ++pqrs)
    //        aphys_tei_ab[pqrs] = 0.0;
    //    for (size_t pqrs = 0; pqrs < num_aptei_; ++pqrs)
    //        aphys_tei_bb[pqrs] = 0.0;

    //    // Store the integrals
    //    for (size_t p = 0; p < nmo_; ++p) {
    //        for (size_t q = 0; q < nmo_; ++q) {
    //            for (size_t r = 0; r < nmo_; ++r) {
    //                for (size_t s = 0; s < nmo_; ++s) {
    //                    // <pq||rs> = <pq|rs> - <pq|sr> = (pr|qs) - (ps|qr)
    //                    double direct = 0.0;
    //                    double exchange = 0.0;
    //                    size_t index = aptei_index(p, q, r, s);
    //                    aphys_tei_aa[index] = direct - exchange;
    //                    aphys_tei_ab[index] = direct;
    //                    aphys_tei_bb[index] = direct - exchange;
    //                }
    //            }
    //        }
    //    }
}

void CustomIntegrals::custom_integrals_allocate(int norb, const std::vector<int>& orbsym) {
    auto result = std::max_element(orbsym.begin(), orbsym.end());
    nirrep_ = *result; // set the number of irreps
    nso_ = norb;
    nmo_ = norb;
    std::vector<int> nmopi(nirrep_, 0);
    for (int sym : orbsym) {
        nmopi[sym - 1] += 1;
    }
    nsopi_ = nmopi;
    nmopi_ = nmopi;

    frzcpi_ = mo_space_info_->dimension("FROZEN_DOCC");
    frzvpi_ = mo_space_info_->dimension("FROZEN_UOCC");
    ncmopi_ = mo_space_info_->dimension("CORRELATED");

    ncmo_ = ncmopi_.sum();

    // Create an array that maps the CMOs to the MOs (cmotomo_).
    for (int h = 0, q = 0; h < nirrep_; ++h) {
        q += frzcpi_[h]; // skip the frozen core
        for (int r = 0; r < ncmopi_[h]; ++r) {
            cmotomo_.push_back(q);
            q++;
        }
        q += frzvpi_[h]; // skip the frozen virtual
    }

    // Indexing
    // This is important!  Set the indexing to work using the number of
    // molecular integrals
    aptei_idx_ = nmo_;
    num_tei_ = INDEX4(nmo_ - 1, nmo_ - 1, nmo_ - 1, nmo_ - 1) + 1;
    num_aptei_ = nmo_ * nmo_ * nmo_ * nmo_;
    //    num_threads_ = omp_get_max_threads();
    print_ = options_->get_int("PRINT");
    /// If MO_ROTATE is set in option, call rotate_mos.
    /// Wasn't really sure where to put this function, but since, integrals is
    /// always called, this seems like a good spot.
    auto rotate_mos_list = options_->get_int_vec("ROTATE_MOS");
    if (rotate_mos_list.size() > 0) {
        outfile->Printf("\n  The option ROTATE_MOS is not supported with custom integrals\n");
        exit(1);
    }
    // full one-electron integrals
    full_one_electron_integrals_a_.assign(nmo_ * nmo_, 0.0);
    full_one_electron_integrals_b_.assign(nmo_ * nmo_, 0.0);

    // these will hold only the correlated part
    one_electron_integrals_a_.assign(ncmo_ * ncmo_, 0.0);
    one_electron_integrals_b_.assign(ncmo_ * ncmo_, 0.0);
    fock_matrix_a_.assign(ncmo_ * ncmo_, 0.0);
    fock_matrix_b_.assign(ncmo_ * ncmo_, 0.0);

    // Allocate the memory required to store the two-electron integrals
    aphys_tei_aa.resize(num_aptei_);
    aphys_tei_ab.resize(num_aptei_);
    aphys_tei_bb.resize(num_aptei_);
}

void CustomIntegrals::resort_integrals_after_freezing() {
    if (print_ > 0) {
        outfile->Printf("\n  Resorting integrals after freezing core.");
    }
    // Resort the four-index integrals
    resort_four(aphys_tei_aa, cmotomo_);
    resort_four(aphys_tei_ab, cmotomo_);
    resort_four(aphys_tei_bb, cmotomo_);
}

void CustomIntegrals::resort_four(std::vector<double>& tei, std::vector<size_t>& map) {
    // Store the integrals in a temporary array
    size_t num_aptei_corr = ncmo_ * ncmo_ * ncmo_ * ncmo_;
    std::vector<double> temp_ints(num_aptei_corr, 0.0);
    for (size_t p = 0; p < ncmo_; ++p) {
        for (size_t q = 0; q < ncmo_; ++q) {
            for (size_t r = 0; r < ncmo_; ++r) {
                for (size_t s = 0; s < ncmo_; ++s) {
                    size_t pqrs_cmo = ncmo_ * ncmo_ * ncmo_ * p + ncmo_ * ncmo_ * q + ncmo_ * r + s;
                    size_t pqrs_mo =
                        nmo_ * nmo_ * nmo_ * map[p] + nmo_ * nmo_ * map[q] + nmo_ * map[r] + map[s];
                    temp_ints[pqrs_cmo] = tei[pqrs_mo];
                }
            }
        }
    }
    temp_ints.swap(tei);
}

void CustomIntegrals::make_fock_matrix(std::shared_ptr<psi::Matrix> gamma_a,
                                       std::shared_ptr<psi::Matrix> gamma_b) {
    for (size_t p = 0; p < ncmo_; ++p) {
        for (size_t q = 0; q < ncmo_; ++q) {
            fock_matrix_a_[p * ncmo_ + q] = oei_a(p, q);
            fock_matrix_b_[p * ncmo_ + q] = oei_b(p, q);
        }
    }
    double zero = 1e-12;
    /// TODO: Either use ambit or use structure of gamma.
    for (size_t r = 0; r < ncmo_; ++r) {
        for (size_t s = 0; s < ncmo_; ++s) {
            double gamma_a_rs = gamma_a->get(r, s);
            if (std::fabs(gamma_a_rs) > zero) {
                for (size_t p = 0; p < ncmo_; ++p) {
                    for (size_t q = 0; q < ncmo_; ++q) {
                        fock_matrix_a_[p * ncmo_ + q] += aptei_aa(p, r, q, s) * gamma_a_rs;
                        fock_matrix_b_[p * ncmo_ + q] += aptei_ab(r, p, s, q) * gamma_a_rs;
                    }
                }
            }
        }
    }
    for (size_t r = 0; r < ncmo_; ++r) {
        for (size_t s = 0; s < ncmo_; ++s) {
            double gamma_b_rs = gamma_b->get(r, s);
            if (std::fabs(gamma_b_rs) > zero) {
                for (size_t p = 0; p < ncmo_; ++p) {
                    for (size_t q = 0; q < ncmo_; ++q) {
                        fock_matrix_a_[p * ncmo_ + q] += aptei_ab(p, r, q, s) * gamma_b_rs;
                        fock_matrix_b_[p * ncmo_ + q] += aptei_bb(p, r, q, s) * gamma_b_rs;
                    }
                }
            }
        }
    }
}
} // namespace forte
