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

#include "ci_spin_adaptation.h"
#include <set>
// #include <numeric>

#include "psi4/psi4-dec.h"
#include "psi4/libpsi4util/PsiOutStream.h"
// #include "psi4/libmints/vector.h"

// #include "forte-def.h"
// #include "ci_reference.h"
// #include "base_classes/forte_options.h"
// #include "helpers/helpers.h"
// #include "helpers/printing.h"
// #include "helpers/timer.h"

// #include <algorithm>

namespace forte {

auto generate_spin_couplings(int N, int twoS) -> std::vector<String>;

SpinAdapter::SpinAdapter(int na, int nb, int twoS, int twoMs, int norb)
    : na_(na), nb_(nb), twoS_(twoS), twoMs_(twoMs), norb_(norb) {}

void SpinAdapter::csf_C_to_det_C(const std::vector<double>& csf_C, std::vector<double>& det_C) {}

void SpinAdapter::det_C_to_csf_C(const std::vector<double>& det_C, std::vector<double>& csf_C) {}

void SpinAdapter::prepare_couplings(const std::vector<Determinant>& dets) {
#if DEBUG_SPIN_ADAPTATION
    psi::outfile->Printf("\n\n  ==> Spin Adapter <==");
    psi::outfile->Printf("\n\n    Determinants:\n");
    for (const auto& d : dets) {
        psi::outfile->Printf("    %s\n", str(d, norb_).c_str());
    }
#endif

    // find all the configurations
    std::set<Configuration> confs;
    for (const auto& d : dets) {
        confs.insert(Configuration(d));
    }
    confs_ = std::vector<Configuration>(confs.begin(), confs.end());

#if DEBUG_SPIN_ADAPTATION
    psi::outfile->Printf("\n    Configurations:\n");
#endif
    for (size_t i = 0, maxi = confs_.size(); i < maxi; i++) {
        const auto& c = confs_[i];
        auto D = conf_to_csfs(c, twoS_, twoMs_);
    }

    double ClebschGordan(double twoS, double twoM, int dtwoS, double dtwoM) {
        if (dtwoS == 1)
            return std::sqrt(0.5 * (twoS + dtwoM * twoM) / twoS);
        if (dtwoS == -1)
            return -dtwoM * std::sqrt(0.5 * (twoS + 2. - dtwoM * twoM) / (twoS + 2.));
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
            // psi::outfile->Printf("CG(%2d,%2d,%2d,%2d) -> %e (%e)\n", dtwoS, dtwoM, pi, qi,
            //                      ClebschGordan(pi, qi, dtwoS, dtwoM), overlap);
            overlap *= ClebschGordan(pi, qi, dtwoS, dtwoM);
        }
        return overlap;
    }

    auto SpinAdapter::conf_to_csfs(const Configuration& conf, size_t i, int twoS, int twoMs)
        ->std::vector<std::vector<double>> {
        std::vector<std::vector<double>> result;

        // number of unpaired electrons
        const auto N = conf.count_socc();
        String docc = conf.get_docc_str();
        std::vector<int> socc_vec(norb_);
        conf.get_socc_vec(norb_, socc_vec);

#if DEBUG_SPIN_ADAPTATION
        psi::outfile->Printf("\n    conf: %s\n", str(conf, norb_).c_str());
        psi::outfile->Printf("    N: %d\n", N);
#endif

        // make the spin couplings for the CSFs
        const auto spin_couplings = generate_spin_couplings(N, twoS);
        const auto determinant_occ = make_determinant_occupations(N, twoS);

        // loop over the spin couplings
        for (const auto& spin_coupling : spin_couplings) {
#if DEBUG_SPIN_ADAPTATION
            psi::outfile->Printf("    spin_coupling: %s\n", str(spin_coupling, N).c_str());
#endif
            // loop over the determinants
            for (const auto& det_occ : determinant_occ) {
                auto o = overlap(N, spin_coupling, det_occ);
                Determinant det(docc, docc);
                for (int i = N - 1; i >= 0; i--) {
                    if (not det_occ[i]) {
                        det.create_alfa_bit(socc_vec[i]);
                    } else {
                        det.create_beta_bit(socc_vec[i]);
                    }
                }
#if DEBUG_SPIN_ADAPTATION
                psi::outfile->Printf("      determinant: %s = %s: %e\n", str(det_occ, N).c_str(),
                                     str(det, norb_).c_str(), o);
#endif
            }
        }

        return result;
    }

    auto SpinAdapter::make_spin_couplings(int N, int twoS)->std::vector<String> {
        std::vector<String> couplings;
        if (N == 0)
            return couplings;
        auto nup = (N + twoS) / 2;
        auto ndown = (N - twoS) / 2;
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
        psi::outfile->Printf("backtrack_spin_couplings: %d %d %d %s\n", nu, nd, sum2S,
                             str(coupling, depth).c_str());
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

    auto generate_spin_couplings(int N, int twoS)->std::vector<String> {
        std::vector<String> couplings;
        if (N == 0)
            return couplings;
        int nup = (N + twoS) / 2;
        int ndown = (N - twoS) / 2;
        psi::outfile->Printf("nup: %d, ndown: %d\n", nup, ndown);
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

    auto SpinAdapter::make_determinant_occupations(int N, int twoS)->std::vector<String> {
        std::vector<String> det_occs;
        if (N == 0)
            return det_occs;
        auto nup = (N + twoS) / 2;
        auto ndown = (N - twoS) / 2;
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
