/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2017 by its authors (see LICENSE, AUTHORS).
 *
 * The copyrights for code used from other parties are included in
 * the corresponding files.
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see http://www.gnu.org/licenses/.
 *
 * @END LICENSE
 */

#include "determinant_map.h"
#include <numeric>

namespace psi {
namespace forte {

bool descending_pair(const std::pair<double, size_t> p1,
                     const std::pair<double, size_t> p2) {
    return p1 > p2;
}

DeterminantMap::DeterminantMap(std::vector<STLBitsetDeterminant>& dets) {
    // The dimension of the wavefunction
    wfn_size_ = dets.size();

    // Take the determinants and coefficients and build the hash
    for (size_t I = 0; I < wfn_size_; ++I) {
        wfn_[dets[I]] = I;
    }
}

DeterminantMap::DeterminantMap(const std::vector<STLBitsetDeterminant>& dets) {
    // The dimension of the wavefunction
    wfn_size_ = dets.size();

    // Take the determinants and coefficients and build the hash
    for (size_t I = 0; I < wfn_size_; ++I) {
        wfn_[dets[I]] = I;
    }
}

DeterminantMap::DeterminantMap(STLBitsetDeterminant& det) {
    wfn_[det] = 0;
    wfn_size_ = wfn_.size();
}

DeterminantMap::DeterminantMap() { wfn_size_ = wfn_.size(); }

DeterminantMap::DeterminantMap(detmap& wfn) : wfn_(wfn) {
    wfn_size_ = wfn.size();
}

const detmap& DeterminantMap::wfn_hash() const { return wfn_; }

detmap& DeterminantMap::wfn_hash() { return wfn_; }

void DeterminantMap::clear() {
    wfn_.clear();
    wfn_size_ = wfn_.size();
}

std::vector<STLBitsetDeterminant> DeterminantMap::determinants() const {
    std::vector<STLBitsetDeterminant> space(wfn_size_);

    for (detmap::const_iterator it = wfn_.begin(); it != wfn_.end(); ++it) {
        space[it->second] = it->first;
    }
    return space;
}

size_t DeterminantMap::size() const { return wfn_.size(); }

void DeterminantMap::add(const STLBitsetDeterminant& det) {
    wfn_[det] = wfn_size_;
    wfn_size_ = wfn_.size();
}

STLBitsetDeterminant DeterminantMap::get_det(const size_t value) const {
    // Iterate through map to find the right one
    // Possibly a faster way to do this?
    STLBitsetDeterminant det;

    for (detmap::const_iterator it = wfn_.begin(), endit = wfn_.end();
         it != endit; ++it) {
        if (it->second == value) {
            det = it->first;
            break;
        }
    }
    return det;
}

size_t DeterminantMap::get_idx(const STLBitsetDeterminant& det) const {
    size_t idx = wfn_.at(det);

    return idx;
}

void DeterminantMap::make_spin_complete() {
    int nmo = this->get_det(0).nmo_;
    size_t ndet_added = 0;
    std::vector<size_t> closed(nmo, 0);
    std::vector<size_t> open(nmo, 0);
    std::vector<size_t> open_bits(nmo, 0);
    DeterminantMap new_dets;

    for (det_hash<size_t>::iterator it = wfn_.begin(), endit = wfn_.end();
         it != endit; ++it) {
        const STLBitsetDeterminant& det = it->first;
        //        outfile->Printf("\n  Original determinant: %s",
        //        det.str().c_str());
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
            STLBitsetDeterminant new_det;
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
            if ((wfn_.count(new_det) == 0) and !(new_dets.has_det(new_det))) {
                new_dets.add(new_det);
                //                outfile->Printf("\n  added determinant:
                //                %s", new_det.str().c_str());
                ndet_added++;
            }
        } while (std::next_permutation(open_bits.begin(),
                                       open_bits.begin() + naopen + nbopen));
    }
    // if( ndet_added > 0 ){
    //    outfile->Printf("\n\n  Determinant space is spin incomplete!");
    //    outfile->Printf("\n  %zu more determinants were needed.", ndet_added);
    //}else{
    //    outfile->Printf("\n\n  Determinant space is spin complete.");
    //}
    this->merge(new_dets);
}

bool DeterminantMap::has_det(const STLBitsetDeterminant& det) const {
    bool has = false;

    if (wfn_.count(det) != 0) {
        has = true;
    }
    return has;
}

double DeterminantMap::overlap(std::vector<double>& det1_evecs,
                               DeterminantMap& det2, SharedMatrix det2_evecs,
                               int root) {

    double overlap = 0.0;

    for (detmap::iterator it = wfn_.begin(), endit = wfn_.end(); it != endit;
         ++it) {
        if (det2.has_det(it->first)) {
            size_t idx = det2.get_idx(it->first);
            overlap += det1_evecs[it->second] * det2_evecs->get(idx, root);
        }
    }
    overlap = std::abs(overlap);
    return overlap;
}

double DeterminantMap::overlap(SharedMatrix det1_evecs, int root1,
                               DeterminantMap& det2, SharedMatrix det2_evecs,
                               int root2) {
    double overlap = 0.0;
    for (detmap::iterator it = wfn_.begin(), endit = wfn_.end(); it != endit;
         ++it) {
        if (det2.has_det(it->first)) {
            size_t idx = det2.get_idx(it->first);
            overlap += det1_evecs->get(it->second, root1) *
                       det2_evecs->get(idx, root2);
        }
    }
    return overlap;
}

void DeterminantMap::subspace(DeterminantMap& dets, SharedMatrix evecs,
                              std::vector<double>& new_evecs, int dim,
                              int root) {
    // Clear current wfn
    this->clear();
    new_evecs.assign(dim, 0.0);

    std::vector<std::pair<double, size_t>> det_weights;
    // for( size_t I = 0, maxI = dets.size(); I < maxI; ++I ){
    detmap map = dets.wfn_hash();
    for (detmap::iterator it = map.begin(), endit = map.end(); it != endit;
         ++it) {
        det_weights.push_back(
            std::make_pair(std::abs(evecs->get(it->second, root)), it->second));
        //      outfile->Printf("\n %1.6f  %zu  %s", evecs->get(it->second,
        //      root), it->second, it->first.str().c_str());
    }
    std::sort(det_weights.begin(), det_weights.end(), descending_pair);

    // Build this wfn with most important subset
    for (size_t I = 0; I < dim; ++I) {
        const STLBitsetDeterminant& detI = dets.get_det(det_weights[I].second);
        this->add(detI);
        new_evecs[I] = evecs->get(det_weights[I].second, root);
        // outfile->Printf("\n %1.6f  %s", new_evecs[I],
        // this->get_det(det_weights[I].second).str().c_str());
        //        outfile->Printf("\n %1.6f  %s", new_evecs[I],
        //        detI.str().c_str());
    }
}

void DeterminantMap::merge(DeterminantMap& dets) {
    det_hash<size_t> detmap = dets.wfn_hash();
    for (det_hash<size_t>::iterator it = detmap.begin(), endit = detmap.end();
         it != endit; ++it) {
        if (!(this->has_det(it->first))) {
            this->add(it->first);
        }
    }
}

void DeterminantMap::copy( DeterminantMap& dets ){
    this->clear();
    wfn_ = dets.wfn_;
    wfn_size_ = dets.size();
}

}
}
