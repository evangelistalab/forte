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

#include <set>

#include "psi4/psi4-dec.h"
#include "psi4/libpsi4util/PsiOutStream.h"
#include "psi4/libmints/vector.h"

#include "helpers/timer.h"

#include "sparse_ci/determinant_hashvector.h"
#include "ci_spin_adaptation.h"

namespace forte {

/// @brief A flag to enable/disable debug messages
constexpr bool DEBUG_SPIN_ADAPTATION = false;

#if DEBUG_SPIN_ADAPTATION
template <typename... Args> void debug(const std::string& format, Args... args) {
    std::string new_format = "[DEBUG] " + format;
    psi::outfile->Printf(new_format.c_str(), args...);
}
#else
template <typename... Args> void debug(const std::string& format, Args... args) {}
#endif

auto generate_spin_couplings(int N, int twoS) -> std::vector<String>;

double ClebschGordan(double twoS, double twoM, int dtwoS, double dtwoM) {
    if (dtwoS == 1)
        return std::sqrt(0.5 * (twoS + dtwoM * twoM) / twoS);
    if (dtwoS == -1)
        return -dtwoM * std::sqrt(0.5 * (twoS + 2. - dtwoM * twoM) / (twoS + 2.));
    return 0.0;
}

double overlap(int N, const String& spin_coupling, const String& det_occ) {
    double overlap = 1.0;

    // compute the overlap between the determinant and the CSF
    int pi = 0;
    int qi = 0;
    for (int i = 0; i < N; i++) {
        const int dpi = spin_coupling[i];
        const int dqi = det_occ[i];
        int dtwoS = 1 - 2 * dpi;
        int dtwoM = 1 - 2 * dqi;
        pi += dtwoS;
        qi += dtwoM;
        if (std::abs(qi) > pi)
            return 0.0;
        overlap *= ClebschGordan(pi, qi, dtwoS, dtwoM);
    }
    return overlap;
}

SpinAdapter::SpinAdapter(int na, int nb, int twoS, int twoMs, int norb)
    : twoS_(twoS), twoMs_(twoMs), norb_(norb) {}

size_t SpinAdapter::ncsf() const { return ncsf_; }

size_t SpinAdapter::ndet() const { return ndet_; }
void SpinAdapter::det_C_to_csf_C(const std::vector<double>& det_C, std::vector<double>& csf_C) {

    // zero the vector csf_C
    std::fill(csf_C.begin(), csf_C.end(), 0.0);

    // // loop over all the elements of csf_to_det_coeff_ and add the contribution to csf_C
    for (size_t i = 0; i < ncsf_; i++) {
        const auto& start = csf_to_det_bounds_[i];
        const auto& end = csf_to_det_bounds_[i + 1];
        for (size_t j = start; j < end; j++) {
            const auto& [det_idx, coeff] = csf_to_det_coeff_[j];
            csf_C[i] += coeff * det_C[det_idx];
        }
    }
}

void SpinAdapter::csf_C_to_det_C(const std::vector<double>& csf_C, std::vector<double>& det_C) {

    // zero the vector det_C
    std::fill(det_C.begin(), det_C.end(), 0.0);

    // loop over all the elements of csf_to_det_coeff_ and add the contribution to det_C
    for (size_t i = 0; i < ncsf_; i++) {
        const auto& start = csf_to_det_bounds_[i];
        const auto& end = csf_to_det_bounds_[i + 1];
        for (size_t j = start; j < end; j++) {
            const auto& [det_idx, coeff] = csf_to_det_coeff_[j];
            det_C[det_idx] += coeff * csf_C[i];
        }
    }
}

void SpinAdapter::det_C_to_csf_C(std::shared_ptr<psi::Vector>& det_C,
                                 std::shared_ptr<psi::Vector>& csf_C) {

    // zero the vector csf_C
    csf_C->zero();

    // // loop over all the elements of csf_to_det_coeff_ and add the contribution to csf_C
    for (size_t i = 0; i < ncsf_; i++) {
        const auto& start = csf_to_det_bounds_[i];
        const auto& end = csf_to_det_bounds_[i + 1];
        for (size_t j = start; j < end; j++) {
            const auto& [det_idx, coeff] = csf_to_det_coeff_[j];
            csf_C->add(i, coeff * det_C->get(det_idx));
        }
    }
}

void SpinAdapter::csf_C_to_det_C(std::shared_ptr<psi::Vector>& csf_C,
                                 std::shared_ptr<psi::Vector>& det_C) {

    // zero the vector det_C
    det_C->zero();

    // loop over all the elements of csf_to_det_coeff_ and add the contribution to det_C
    for (size_t i = 0; i < ncsf_; i++) {
        const auto& start = csf_to_det_bounds_[i];
        const auto& end = csf_to_det_bounds_[i + 1];
        for (size_t j = start; j < end; j++) {
            const auto& [det_idx, coeff] = csf_to_det_coeff_[j];
            det_C->add(det_idx, coeff * csf_C->get(i));
        }
    }
}

void SpinAdapter::prepare_couplings(const std::vector<Determinant>& dets) {
    psi::outfile->Printf("\n\n  ==> Spin Adapter <==\n\n");
    // debug("    Determinants:\n");
    ndet_ = 0;
    for (const auto& d : dets) {
        // debug("    %6zu %s\n", ndet_, str(d, norb_).c_str());
        ndet_++;
    }

    DeterminantHashVec det_hash(dets);

    // find all the configurations
    local_timer t1;
    std::set<Configuration> confs;
    std::vector<size_t> count_N(2 * norb_ + 1, 0);
    for (const auto& d : dets) {
        confs.insert(Configuration(d));
    }

    // loop over all the configurations and count the number of determinants with the same N
    for (const auto& conf : confs) {
        // psi::outfile->Printf("    %s -> %d\n", str(conf).c_str(), conf.count_socc());
        if (conf.count_socc() >= twoS_) {
            count_N[conf.count_socc()]++;
        }
    }

    // here we compute the number of couplings and the number of CSFs
    size_t predicted_ncouplng = 0;
    size_t predicted_ncsf = 0;
    for (size_t N = 0; N < count_N.size(); N++) {
        if (count_N[N] > 0) {
            // debug("    N = %zu: %zu\n", N, count_N[N]);
            size_t ncoupling_N = 0;
            size_t ncsf_N = 0;
            const auto spin_couplings = make_spin_couplings(N, twoS_);
            const auto determinant_occ = make_determinant_occupations(N, twoMs_);
            for (const auto& spin_coupling : spin_couplings) {
                for (const auto& det_occ : determinant_occ) {
                    auto o = overlap(N, spin_coupling, det_occ);
                    if (std::fabs(o) > 0.0) {
                        ncoupling_N++;
                    }
                }
                ncsf_N++;
            }
            predicted_ncouplng += ncoupling_N * count_N[N];
            predicted_ncsf += ncsf_N * count_N[N];
        }
    }

    csf_to_det_coeff_.resize(predicted_ncouplng);
    csf_to_det_bounds_.resize(predicted_ncsf + 1);

    confs_ = std::vector<Configuration>(confs.begin(), confs.end());
    psi::outfile->Printf("    Timings: find configurations:        %10.4f\n", t1.get());
    // debug("    Configurations:\n");
    ncsf_ = 0;
    ncoupling_ = 0;
    local_timer t2;
    for (const auto& conf : confs_) {
        if (conf.count_socc() >= twoS_) {
            conf_to_csfs(conf, twoS_, twoMs_, det_hash);
        }
    }

    psi::outfile->Printf("    Timings: find CSFs :                 %10.4f\n", t2.get());
    psi::outfile->Printf("\n    Number of CSFs (predicted):          %10zu\n", predicted_ncsf);
    psi::outfile->Printf("    Number of couplings (predicted):     %10zu\n", predicted_ncouplng);
}

void SpinAdapter::conf_to_csfs(const Configuration& conf, int twoS, int twoMs,
                               DeterminantHashVec& det_hash) {
    // number of unpaired electrons
    const auto N = conf.count_socc();
    String docc = conf.get_docc_str();
    std::vector<int> socc_vec(norb_);
    conf.get_socc_vec(norb_, socc_vec);

    // debug("    conf: %s\n", str(conf, norb_).c_str());
    // debug("    N: %d\n", N);

    // make the spin couplings for the CSFs
    const auto spin_couplings = make_spin_couplings(N, twoS);
    const auto determinant_occ = make_determinant_occupations(N, twoMs);

    csf_to_det_bounds_[0] = 0;
    Determinant det;
    // loop over the spin couplings
    for (const auto& spin_coupling : spin_couplings) {
        // debug("    spin_coupling: %s\n", str(spin_coupling, N).c_str());
        // loop over the determinants
        for (const auto& det_occ : determinant_occ) {
            auto o = overlap(N, spin_coupling, det_occ);
            if (std::fabs(o) > 0.0) {
                det.set_str(docc, docc);
                // keep track of the sign of the singly occupied orbitals
                for (int i = N - 1; i >= 0; i--) {
                    if (det_occ.get_bit(i)) {
                        o *= det.create_beta_bit(socc_vec[i]);
                    } else {
                        o *= det.create_alfa_bit(socc_vec[i]);
                    }
                }
                csf_to_det_coeff_[ncoupling_].first = det_hash.get_idx(det);
                csf_to_det_coeff_[ncoupling_].second = o;
                ncoupling_ += 1;
                // debug("      CSF(%3zu) <- %+e %s (%zu)\n", ncsf_, o, str(det, norb_).c_str(),
                //       det_hash.get_idx(det));
            }
            // debug("      determinant: %s = %s: %e\n", str(det_occ, N).c_str(),
            //       str(det, norb_).c_str(), o);
        }
        ncsf_ += 1;
        csf_to_det_bounds_[ncsf_] = ncoupling_;
    }
}

auto SpinAdapter::make_spin_couplings(int N, int twoS) -> std::vector<String> {
    if (N == 0)
        return std::vector<String>(1, String());
    std::vector<String> couplings;
    auto nup = (N + twoS) / 2;
    String coupling;
    // false = 0 = up, true = 1 = down
    // The coupling should always start with up
    for (int i = 0; i < nup; i++)
        coupling[i] = false;
    for (int i = nup; i < N; i++)
        coupling[i] = true;
    /// Generate all permutations of the path
    do {
        bool valid = true;
        for (int i = 0, p = 0; i < N; i++) {
            p += 1 - 2 * coupling[i];
            if (p < 0)
                valid = false;
        }
        if (valid)
            couplings.push_back(coupling);
    } while (std::next_permutation(coupling.begin() + 1, coupling.begin() + N));
    return couplings;
}

void backtrack_spin_couplings(int nu, int nd, int sum2S, String& coupling,
                              std::vector<String>& couplings, int depth) {
    // debug("backtrack_spin_couplings: %d %d %d %s\n", nu, nd, sum2S, str(coupling,
    // depth).c_str());
    if (nu == 0 and nd == 0) {
        couplings.push_back(coupling);
    }
    if (nu > 0) {
        coupling[depth] = true;
        backtrack_spin_couplings(nu - 1, nd, sum2S + 1, coupling, couplings, depth + 1);
    }
    if (nd > 0 and sum2S > 0) {
        coupling[depth] = false;
        backtrack_spin_couplings(nu, nd - 1, sum2S - 1, coupling, couplings, depth + 1);
    }
}

auto generate_spin_couplings(int N, int twoS) -> std::vector<String> {
    std::vector<String> couplings;
    if (N == 0)
        return couplings;
    int nup = (N + twoS) / 2;
    int ndown = (N - twoS) / 2;
    // debug("nup: %d, ndown: %d\n", nup, ndown);
    String coupling;
    int sum2S = 1;
    backtrack_spin_couplings(nup - 1, ndown, sum2S, coupling, couplings, 1);
    return couplings;
}

// def enumerate_paths(nu, nd):
//     path = []
//     cum2S = 0
//     first = True
//     yield from backtrack(nu, nd, path, cum2S, first)

// def backtrack(nu, nd, path, cum2S, first):
//     if nu == 0 and nd == 0:
//         yield path
//     if nu > 0:
//         path.append(cum2S + 1)
//         yield from backtrack(nu-1, nd, path, cum2S + 1, False)
//         path.pop()
//     if nd > 0 and cum2S > 0 and first == False:
//         path.append(cum2S - 1)
//         yield from backtrack(nu, nd-1, path, cum2S - 1, False)
//         path.pop()

// def generate_paths(N,twoS):
//     paths = []
//     nup = (N + twoS) // 2
//     ndown = (N - twoS) // 2
//     print(f'{nup=} {ndown=}')
//     for path in enumerate_paths(nup,ndown):
//         print(path)
//         paths.append(list(path))
//     return paths

auto SpinAdapter::make_determinant_occupations(int N, int twoMs) -> std::vector<String> {
    std::vector<String> det_occs;
    if (N == 0)
        return std::vector<String>(1, String());
    auto nup = (N + twoMs) / 2;
    String det_occ;
    // true = 1 = up, false = 0 = down
    // The det_occ should always start with up
    for (int i = 0; i < nup; i++)
        det_occ[i] = false;
    for (int i = nup; i < N; i++)
        det_occ[i] = true;
    /// Generate all permutations of the path
    do {
        det_occs.push_back(det_occ);
    } while (std::next_permutation(det_occ.begin(), det_occ.begin() + N));
    return det_occs;
}

} // namespace forte
