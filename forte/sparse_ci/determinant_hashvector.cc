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
#include <algorithm>

#include "psi4/libmints/matrix.h"

#include "determinant_hashvector.h"
#include <numeric>
#include <cmath>

namespace forte {

DeterminantHashVec::DeterminantHashVec(std::vector<Determinant>& dets) {
    // The dimension of the wavefunction
    wfn_ = det_hashvec(dets);
}

DeterminantHashVec::DeterminantHashVec(const std::vector<Determinant>& dets) {
    // The dimension of the wavefunction
    wfn_ = det_hashvec(dets);
}

DeterminantHashVec::DeterminantHashVec(Determinant& det) { wfn_.add(det); }

DeterminantHashVec::DeterminantHashVec() {}

DeterminantHashVec::DeterminantHashVec(const det_hashvec& wfn) : wfn_(wfn) {}

DeterminantHashVec::DeterminantHashVec(det_hashvec&& wfn) { wfn_.swap(wfn); }

const det_hashvec& DeterminantHashVec::wfn_hash() const { return wfn_; }

det_hashvec& DeterminantHashVec::wfn_hash() { return wfn_; }

void DeterminantHashVec::clear() { wfn_.clear(); }

std::vector<Determinant> DeterminantHashVec::determinants() const { return wfn_.toVector(); }

std::vector<std::pair<Determinant, size_t>> DeterminantHashVec::determinant_index_pairs() const {
    return wfn_.toKeyIndex();
}

size_t DeterminantHashVec::size() const { return wfn_.size(); }

size_t DeterminantHashVec::add(const Determinant& det) { return wfn_.add(det); }

const Determinant& DeterminantHashVec::get_det(const size_t value) const {
    // Iterate through map to find the right one
    // Possibly a faster way to do this?
    return wfn_[value];
}

size_t DeterminantHashVec::get_idx(const Determinant& det) const { return wfn_.find(det); }

void DeterminantHashVec::make_spin_complete(int nmo) {
    std::vector<size_t> closed(nmo, 0);
    std::vector<size_t> open(nmo, 0);
    std::vector<size_t> open_bits(nmo, 0);
    DeterminantHashVec new_dets;

    for (det_hashvec::iterator it = wfn_.begin(), endit = wfn_.end(); it != endit; ++it) {
        const Determinant& det = *it;
        for (int i = 0; i < nmo; ++i) {
            closed[i] = open[i] = 0;
            open_bits[i] = false;
        }
        int naopen = 0;
        int nbopen = 0;
        int nclosed = 0;
        for (int i = 0; i < nmo; ++i) {
            if (det.get_alfa_bit(i) and (not det.get_beta_bit(i))) {
                open[naopen + nbopen] = i;
                naopen += 1;
            } else if ((not det.get_alfa_bit(i)) and det.get_beta_bit(i)) {
                open[naopen + nbopen] = i;
                nbopen += 1;
            } else if (det.get_alfa_bit(i) and det.get_beta_bit(i)) {
                closed[nclosed] = i;
                nclosed += 1;
            }
        }

        if (naopen + nbopen == 0)
            continue;

        // Generate the strings 1111100000
        //                      {nao}{nbo}
        for (int i = 0; i < nbopen; ++i)
            open_bits[i] = false; // 0
        for (int i = nbopen; i < naopen + nbopen; ++i)
            open_bits[i] = true; // 1
        do {
            // Determinant new_det(nmo); <- xsize
            Determinant new_det;
            for (int c = 0; c < nclosed; ++c) {
                new_det.set_alfa_bit(closed[c], true);
                new_det.set_beta_bit(closed[c], true);
            }
            for (int o = 0; o < naopen + nbopen; ++o) {
                if (open_bits[o]) { //? not
                    new_det.set_alfa_bit(open[o], true);
                } else {
                    new_det.set_beta_bit(open[o], true);
                }
            }
            if (!(this->has_det(new_det)) and !(new_dets.has_det(new_det))) {
                new_dets.add(new_det);
            }
        } while (std::next_permutation(open_bits.begin(), open_bits.begin() + naopen + nbopen));
    }
    this->merge(new_dets);
}

bool DeterminantHashVec::has_det(const Determinant& det) const {
    return wfn_.find(det) != det_hashvec::npos;
}

double DeterminantHashVec::overlap(std::vector<double>& det1_evecs, DeterminantHashVec& det2,
                                   std::shared_ptr<psi::Matrix> det2_evecs, int root) {

    double overlap = 0.0;

    for (size_t i = 0, wfn_size = wfn_.size(); i < wfn_size; ++i) {
        if (det2.has_det(wfn_[i])) {
            size_t idx = det2.get_idx(wfn_[i]);
            overlap += det1_evecs[i] * det2_evecs->get(idx, root);
        }
    }
    //    for (det_hashvec::iterator it = wfn_.begin(), endit = wfn_.end(); it != endit; ++it) {
    //        if (det2.has_det(*it)) {
    //            size_t idx = det2.get_idx(*it);
    //            overlap += det1_evecs[wfn_.find(*it)] * det2_evecs->get(idx, root);
    //        }
    //    }
    overlap = std::abs(overlap);
    return overlap;
}

double DeterminantHashVec::overlap(std::shared_ptr<psi::Matrix> det1_evecs, int root1,
                                   DeterminantHashVec& det2,
                                   std::shared_ptr<psi::Matrix> det2_evecs, int root2) {
    double overlap = 0.0;
    for (size_t i = 0, wfn_size = wfn_.size(); i < wfn_size; ++i) {
        if (det2.has_det(wfn_[i])) {
            size_t idx = det2.get_idx(wfn_[i]);
            overlap += det1_evecs->get(i, root1) * det2_evecs->get(idx, root2);
        }
    }
    //    for (det_hashvec::iterator it = wfn_.begin(), endit = wfn_.end(); it != endit; ++it) {
    //        if (det2.has_det(*it)) {
    //            size_t idx = det2.get_idx(*it);
    //            overlap += det1_evecs->get(wfn_.find(*it), root1) * det2_evecs->get(idx, root2);
    //        }
    //    }
    return overlap;
}

void DeterminantHashVec::subspace(DeterminantHashVec& dets, std::shared_ptr<psi::Matrix> evecs,
                                  std::vector<double>& new_evecs, size_t dim, int root) {
    // Clear current wfn
    this->clear();
    new_evecs.assign(dim, 0.0);

    std::vector<std::pair<double, size_t>> det_weights;
    // for( size_t I = 0, maxI = dets.size(); I < maxI; ++I ){
    const det_hashvec& map = dets.wfn_hash();

    for (size_t i = 0, map_size = map.size(); i < map_size; ++i) {
        det_weights.push_back(std::make_pair(std::abs(evecs->get(i, root)), i));
    }
    //    for (det_hashvec::iterator it = map.begin(), endit = map.end(); it != endit; ++it) {
    //        det_weights.push_back(std::make_pair(std::abs(evecs->get(map.find(*it), root)),
    //        map.find(*it)));
    //        //      outfile->Printf("\n %1.6f  %zu  %s", evecs->get(it->second,
    //        //      root), it->second, *it.str().c_str());
    //    }
    std::sort(det_weights.rbegin(), det_weights.rend());

    // Build this wfn with most important subset
    for (size_t I = 0; I < dim; ++I) {
        const Determinant& detI = dets.get_det(det_weights[I].second);
        this->add(detI);
        new_evecs[I] = evecs->get(det_weights[I].second, root);
        // outfile->Printf("\n %1.6f  %s", new_evecs[I],
        // this->get_det(det_weights[I].second).str().c_str());
        //        outfile->Printf("\n %1.6f  %s", new_evecs[I],
        //        detI.str().c_str());
    }
}

void DeterminantHashVec::merge(DeterminantHashVec& dets) {
    for (const Determinant& d : dets.wfn_) {
        wfn_.add(d);
    }
}

void DeterminantHashVec::copy(DeterminantHashVec& dets) {
    this->clear();
    wfn_ = dets.wfn_;
}

void DeterminantHashVec::swap(DeterminantHashVec& dets) { wfn_.swap(dets.wfn_); }

void DeterminantHashVec::swap(det_hashvec& dets) { wfn_.swap(dets); }
} // namespace forte
