#include "aci.h"


namespace psi {
namespace forte {

bool pair_comp(const std::pair<double, Determinant> E1, const std::pair<double, Determinant> E2) {
    return E1.first < E2.first;
}

void AdaptiveCI::get_excited_determinants_sr(SharedMatrix evecs, 
                                             DeterminantHashVec& P_space,
                                             det_hash<double>& V_hash )
{
Timer build;
    size_t max_P = P_space.size();
    const det_hashvec& P_dets = P_space.wfn_hash();
    int nroot = 1;

// Loop over reference determinants
#pragma omp parallel
    {
        int num_thread = omp_get_num_threads();
        int tid = omp_get_thread_num();
        size_t bin_size = max_P / num_thread;
        bin_size += (tid < (max_P % num_thread)) ? 1 : 0;
        size_t start_idx =
            (tid < (max_P % num_thread))
               ? tid * bin_size
                : (max_P % num_thread) * (bin_size + 1) + (tid - (max_P % num_thread)) * bin_size;
        size_t end_idx = start_idx + bin_size;

        det_hash<double> V_hash_t;
        for (size_t P = start_idx; P < end_idx; ++P) {
            const Determinant& det(P_dets[P]);
            double Cp = evecs->get(P,0);

            std::vector<int> aocc = det.get_alfa_occ(nact_); // TODO check size
            std::vector<int> bocc = det.get_beta_occ(nact_); // TODO check size
            std::vector<int> avir = det.get_alfa_vir(nact_); // TODO check size
            std::vector<int> bvir = det.get_beta_vir(nact_); // TODO check size

            int noalpha = aocc.size();
            int nobeta = bocc.size();
            int nvalpha = avir.size();
            int nvbeta = bvir.size();
            Determinant new_det(det);
            // Generate alpha excitations
            for (int i = 0; i < noalpha; ++i) {
                int ii = aocc[i];
                for (int a = 0; a < nvalpha; ++a) {
                    int aa = avir[a];
                    if ((mo_symmetry_[ii] ^ mo_symmetry_[aa]) == 0) {
                        double HIJ = fci_ints_->slater_rules_single_alpha(det, ii, aa) * Cp;
                        if ( std::abs(HIJ) >= screen_thresh_) {
                            new_det = det;
                            new_det.set_alfa_bit(ii, false);
                            new_det.set_alfa_bit(aa, true);
                            if( V_hash_t.count(new_det) == 0){
                                V_hash_t[new_det] = HIJ; 
                            } else {
                                V_hash_t[new_det] += HIJ;
                            }
                        }
                    }
                }
            }
            // Generate beta excitations
            for (int i = 0; i < nobeta; ++i) {
                int ii = bocc[i];
                for (int a = 0; a < nvbeta; ++a) {
                    int aa = bvir[a];
                    if ((mo_symmetry_[ii] ^ mo_symmetry_[aa]) == 0) {
                        double HIJ = fci_ints_->slater_rules_single_beta(det, ii, aa) * Cp;
                        if ( std::abs(HIJ) >= screen_thresh_) {
                            new_det = det;
                            new_det.set_beta_bit(ii, false);
                            new_det.set_beta_bit(aa, true);
                            if( V_hash_t.count(new_det) == 0){
                                V_hash_t[new_det] = HIJ; 
                            } else {
                                V_hash_t[new_det] += HIJ;
                            }
                        }
                    }
                }
            }
            // Generate aa excitations
            for (int i = 0; i < noalpha; ++i) {
                int ii = aocc[i];
                for (int j = i + 1; j < noalpha; ++j) {
                    int jj = aocc[j];
                    for (int a = 0; a < nvalpha; ++a) {
                        int aa = avir[a];
                        for (int b = a + 1; b < nvalpha; ++b) {
                            int bb = avir[b];
                            if ((mo_symmetry_[ii] ^ mo_symmetry_[jj] ^ mo_symmetry_[aa] ^ mo_symmetry_[bb]) == 0) {
                                double HIJ = fci_ints_->tei_aa(ii, jj, aa, bb) * Cp;
                                if ( std::abs(HIJ) >= screen_thresh_) {
                                    new_det = det;
                                    HIJ *= new_det.double_excitation_aa(ii, jj, aa, bb);
                                    if( V_hash_t.count(new_det) == 0){
                                        V_hash_t[new_det] = HIJ; 
                                    } else {
                                        V_hash_t[new_det] += HIJ;
                                    }
                                }
                            }
                        }
                    }
                }
            }
            // Generate ab excitations
            for (int i = 0; i < noalpha; ++i) {
                int ii = aocc[i];
                for (int j = 0; j < nobeta; ++j) {
                    int jj = bocc[j];
                    for (int a = 0; a < nvalpha; ++a) {
                        int aa = avir[a];
                        for (int b = 0; b < nvbeta; ++b) {
                            int bb = bvir[b];
                            if ((mo_symmetry_[ii] ^ mo_symmetry_[jj] ^ mo_symmetry_[aa] ^ mo_symmetry_[bb]) == 0){ 
                                double HIJ = fci_ints_->tei_ab(ii, jj, aa, bb) * Cp;
                                if ( std::abs(HIJ) >= screen_thresh_) {
                                    new_det = det;
                                    HIJ *= new_det.double_excitation_ab(ii, jj, aa, bb);
                                    if( V_hash_t.count(new_det) == 0){
                                        V_hash_t[new_det] = HIJ; 
                                    } else {
                                        V_hash_t[new_det] += HIJ;
                                    }
                                }
                            }
                        }
                    }
                }
            }

            // Generate bb excitations
            for (int i = 0; i < nobeta; ++i) {
                int ii = bocc[i];
                for (int j = i + 1; j < nobeta; ++j) {
                    int jj = bocc[j];
                    for (int a = 0; a < nvbeta; ++a) {
                        int aa = bvir[a];
                        for (int b = a + 1; b < nvbeta; ++b) {
                            int bb = bvir[b];
                            if ((mo_symmetry_[ii] ^ mo_symmetry_[jj] ^ mo_symmetry_[aa] ^ mo_symmetry_[bb]) == 0) {
                                double HIJ = fci_ints_->tei_bb(ii, jj, aa, bb) * Cp;
                                if ( std::abs(HIJ) >= screen_thresh_) {
                                    new_det = det;
                                    HIJ *= new_det.double_excitation_bb(ii, jj, aa, bb);
                                    if( V_hash_t.count(new_det) == 0){
                                        V_hash_t[new_det] = HIJ; 
                                    } else {
                                        V_hash_t[new_det] += HIJ;
                                    }

                                }
                            }
                        }
                    }
                }
            }
        }
if ( tid == 0 ) outfile->Printf("\n  Time spent forming F space: %20.6f", build.get());
Timer merge_t;
size_t n_new = 0;
size_t n_repeat = 0;
#pragma omp critical
        {
            for( auto& pair : V_hash_t ){
                const Determinant& det = pair.first;
                if (V_hash.count(det) != 0) {
                    V_hash[det] += pair.second;
                    n_repeat++;
                } else {
                    V_hash[det] = pair.second;
                    n_new++;
                }
            }
        outfile->Printf("\n td[%d] = %zu new entries", tid, n_new);
        outfile->Printf("\n td[%d] = %zu repeat entries", tid, n_repeat);
        }
if ( tid == 0 ) outfile->Printf("\n  Time spent merging thread F spaces: %20.6f", merge_t.get());
    } // Close threads

   // for( auto& pair : V_hash ){
   //     outfile->Printf("\n  %9.8f  %s", pair.second, pair.first.str(nact_).c_str());
   // }

}


void AdaptiveCI::get_excited_determinants2(int nroot, SharedMatrix evecs,
                                           DeterminantHashVec& P_space,
                                           det_hash<std::vector<double>>& V_hash) {
    const size_t n_dets = P_space.size();

    int nmo = fci_ints_->nmo();
    double max_mem = options_.get_double("PT2_MAX_MEM");

    size_t guess_size = n_dets * nmo * nmo;
    double nbyte = (1073741824 * max_mem) / (sizeof(double));

    int nbin = static_cast<int>(std::ceil(guess_size / (nbyte)));

#pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int ntds = omp_get_num_threads();

        if ((ntds > nbin)) {
            nbin = ntds;
        }

        if (tid == 0) {
            outfile->Printf("\n  Number of bins for exitation space:  %d", nbin);
            outfile->Printf("\n  Number of threads: %d", ntds);
        }
        size_t bin_size = n_dets / ntds;
        bin_size += (tid < (n_dets % ntds)) ? 1 : 0;
        size_t start_idx = (tid < (n_dets % ntds)) ? tid * bin_size
                                                   : (n_dets % ntds) * (bin_size + 1) +
                                                         (tid - (n_dets % ntds)) * bin_size;
        size_t end_idx = start_idx + bin_size;
        for (int bin = 0; bin < nbin; ++bin) {

            det_hash<std::vector<double>> A_I;
            // std::vector<std::pair<Determinant, std::vector<double>>> A_I;

            const det_hashvec& dets = P_space.wfn_hash();
            for (size_t I = start_idx; I < end_idx; ++I) {
                double c_norm = evecs->get_row(0, I)->norm();
                const Determinant& det = dets[I];
                std::vector<int> aocc = det.get_alfa_occ(nmo);
                std::vector<int> bocc = det.get_beta_occ(nmo);
                std::vector<int> avir = det.get_alfa_vir(nmo);
                std::vector<int> bvir = det.get_beta_vir(nmo);

                int noalpha = aocc.size();
                int nobeta = bocc.size();
                int nvalpha = avir.size();
                int nvbeta = bvir.size();
                Determinant new_det(det);

                // Generate alpha excitations
                for (int i = 0; i < noalpha; ++i) {
                    int ii = aocc[i];
                    for (int a = 0; a < nvalpha; ++a) {
                        int aa = avir[a];
                        if ((mo_symmetry_[ii] ^ mo_symmetry_[aa]) == 0) {
                            new_det = det;
                            new_det.set_alfa_bit(ii, false);
                            new_det.set_alfa_bit(aa, true);
                            if (P_space.has_det(new_det))
                                continue;

                            // Check if the determinant goes in this bin
                            size_t hash_val = Determinant::Hash()(new_det);
                            if ((hash_val % nbin) == bin) {
                                double HIJ = fci_ints_->slater_rules_single_alpha(new_det, ii, aa);
                                if ((std::fabs(HIJ) * c_norm >= screen_thresh_)) {
                                    std::vector<double> coupling(nroot_);
                                    for (int n = 0; n < nroot_; ++n) {
                                        coupling[n] = HIJ * evecs->get(I, n);
                                        if (A_I.find(new_det) != A_I.end()) {
                                            coupling[n] += A_I[new_det][n];
                                        }
                                        A_I[new_det] = coupling;
                                    }
                                }
                            }
                        }
                    }
                }
                // Generate beta excitations
                for (int i = 0; i < nobeta; ++i) {
                    int ii = bocc[i];
                    for (int a = 0; a < nvbeta; ++a) {
                        int aa = bvir[a];
                        if ((mo_symmetry_[ii] ^ mo_symmetry_[aa]) == 0) {
                            new_det = det;
                            new_det.set_beta_bit(ii, false);
                            new_det.set_beta_bit(aa, true);
                            if (P_space.has_det(new_det))
                                continue;

                            // Check if the determinant goes in this bin
                            size_t hash_val = Determinant::Hash()(new_det);
                            if ((hash_val % nbin) == bin) {
                                double HIJ = fci_ints_->slater_rules_single_beta(new_det, ii, aa);
                                if ((std::fabs(HIJ) * c_norm >= screen_thresh_)) {
                                    std::vector<double> coupling(nroot_);
                                    for (int n = 0; n < nroot_; ++n) {
                                        coupling[n] = HIJ * evecs->get(I, n);
                                        if (A_I.find(new_det) != A_I.end()) {
                                            coupling[n] += A_I[new_det][n];
                                        }
                                        A_I[new_det] = coupling;
                                    }
                                }
                            }
                        }
                    }
                }
                // Generate aa excitations
                for (int i = 0; i < noalpha; ++i) {
                    int ii = aocc[i];
                    for (int j = i + 1; j < noalpha; ++j) {
                        int jj = aocc[j];
                        for (int a = 0; a < nvalpha; ++a) {
                            int aa = avir[a];
                            for (int b = a + 1; b < nvalpha; ++b) {
                                int bb = avir[b];
                                if ((mo_symmetry_[ii] ^ mo_symmetry_[jj] ^ mo_symmetry_[aa] ^
                                     mo_symmetry_[bb]) == 0) {
                                    new_det = det;
                                    double sign = new_det.double_excitation_aa(ii, jj, aa, bb);
                                    if (P_space.has_det(new_det))
                                        continue;

                                    // Check if the determinant goes in this bin
                                    size_t hash_val = Determinant::Hash()(new_det);
                                    if ((hash_val % nbin) == bin) {
                                        double HIJ = fci_ints_->tei_aa(ii, jj, aa, bb);
                                        if ((std::fabs(HIJ) * c_norm >= screen_thresh_)) {
                                            HIJ *= sign;
                                            std::vector<double> coupling(nroot_);
                                            for (int n = 0; n < nroot_; ++n) {
                                                coupling[n] = HIJ * evecs->get(I, n);
                                                if (A_I.find(new_det) != A_I.end()) {
                                                    coupling[n] += A_I[new_det][n];
                                                }
                                                A_I[new_det] = coupling;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                // Generate bb excitations
                for (int i = 0; i < nobeta; ++i) {
                    int ii = bocc[i];
                    for (int j = i + 1; j < nobeta; ++j) {
                        int jj = bocc[j];
                        for (int a = 0; a < nvbeta; ++a) {
                            int aa = bvir[a];
                            for (int b = a + 1; b < nvbeta; ++b) {
                                int bb = bvir[b];
                                if ((mo_symmetry_[ii] ^ mo_symmetry_[jj] ^ mo_symmetry_[aa] ^
                                     mo_symmetry_[bb]) == 0) {
                                    new_det = det;
                                    double sign = new_det.double_excitation_bb(ii, jj, aa, bb);
                                    if (P_space.has_det(new_det))
                                        continue;

                                    // Check if the determinant goes in this bin
                                    size_t hash_val = Determinant::Hash()(new_det);
                                    if ((hash_val % nbin) == bin) {
                                        double HIJ = fci_ints_->tei_bb(ii, jj, aa, bb);
                                        if ((std::fabs(HIJ) * c_norm >= screen_thresh_)) {
                                            HIJ *= sign;
                                            std::vector<double> coupling(nroot_);
                                            for (int n = 0; n < nroot_; ++n) {
                                                coupling[n] = HIJ * evecs->get(I, n);
                                                if (A_I.find(new_det) != A_I.end()) {
                                                    coupling[n] += A_I[new_det][n];
                                                }
                                                A_I[new_det] = coupling;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                // Generate ab excitations
                for (int i = 0; i < noalpha; ++i) {
                    int ii = aocc[i];
                    for (int j = 0; j < nobeta; ++j) {
                        int jj = bocc[j];
                        for (int a = 0; a < nvalpha; ++a) {
                            int aa = avir[a];
                            for (int b = 0; b < nvbeta; ++b) {
                                int bb = bvir[b];
                                if ((mo_symmetry_[ii] ^ mo_symmetry_[jj] ^ mo_symmetry_[aa] ^
                                     mo_symmetry_[bb]) == 0) {
                                    new_det = det;
                                    double sign = new_det.double_excitation_ab(ii, jj, aa, bb);
                                    if (P_space.has_det(new_det))
                                        continue;

                                    // Check if the determinant goes in this bin
                                    size_t hash_val = Determinant::Hash()(new_det);
                                    if ((hash_val % nbin) == bin) {
                                        double HIJ = fci_ints_->tei_ab(ii, jj, aa, bb);
                                        if ((std::fabs(HIJ) * c_norm >= screen_thresh_)) {
                                            HIJ *= sign;
                                            std::vector<double> coupling(nroot_);
                                            for (int n = 0; n < nroot_; ++n) {
                                                coupling[n] = HIJ * evecs->get(I, n);
                                                if (A_I.find(new_det) != A_I.end()) {
                                                    coupling[n] += A_I[new_det][n];
                                                }
                                                A_I[new_det] = coupling;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }

#pragma omp critical
            {
                for (auto& dpair : A_I) {
                    const std::vector<double>& coupling = dpair.second;
                    const Determinant& det = dpair.first;
                    if (V_hash.count(det) != 0) {
                        for (int n = 0; n < nroot; ++n) {
                            V_hash[det][n] += coupling[n];
                        }
                    } else {
                        V_hash[det] = coupling;
                    }
                }
            }
            // outfile->Printf("\n TD, bin, size of hash: %d %d %zu", tid, bin, A_I.size());
        }
    }
}

void AdaptiveCI::get_excited_determinants(int nroot, SharedMatrix evecs,
                                          DeterminantHashVec& P_space,
                                          det_hash<std::vector<double>>& V_hash) {
    size_t max_P = P_space.size();
    const det_hashvec& P_dets = P_space.wfn_hash();

// Loop over reference determinants
#pragma omp parallel
    {
        int num_thread = omp_get_num_threads();
        int tid = omp_get_thread_num();
        size_t bin_size = max_P / num_thread;
        bin_size += (tid < (max_P % num_thread)) ? 1 : 0;
        size_t start_idx =
            (tid < (max_P % num_thread))
                ? tid * bin_size
                : (max_P % num_thread) * (bin_size + 1) + (tid - (max_P % num_thread)) * bin_size;
        size_t end_idx = start_idx + bin_size;

        if (omp_get_thread_num() == 0 and !quiet_mode_) {
            outfile->Printf("\n  Using %d threads.", num_thread);
        }
        // This will store the excited determinant info for each thread
        std::vector<std::pair<Determinant, std::vector<double>>>
            thread_ex_dets; //( noalpha * nvalpha  );

        for (size_t P = start_idx; P < end_idx; ++P) {
            const Determinant& det(P_dets[P]);
            double evecs_P_row_norm = evecs->get_row(0, P)->norm();

            std::vector<int> aocc = det.get_alfa_occ(nact_);
            std::vector<int> bocc = det.get_beta_occ(nact_);
            std::vector<int> avir = det.get_alfa_vir(nact_);
            std::vector<int> bvir = det.get_beta_vir(nact_);

            int noalpha = aocc.size();
            int nobeta = bocc.size();
            int nvalpha = avir.size();
            int nvbeta = bvir.size();
            Determinant new_det(det);

            // Generate alpha excitations
            for (int i = 0; i < noalpha; ++i) {
                int ii = aocc[i];
                for (int a = 0; a < nvalpha; ++a) {
                    int aa = avir[a];
                    if ((mo_symmetry_[ii] ^ mo_symmetry_[aa]) == 0) {
                        double HIJ = fci_ints_->slater_rules_single_alpha(det, ii, aa);
                        if ((std::fabs(HIJ) * evecs_P_row_norm >= screen_thresh_)) {
                            //      if( std::abs(HIJ * evecs->get(0, P)) > screen_thresh_ ){
                            new_det = det;
                            new_det.set_alfa_bit(ii, false);
                            new_det.set_alfa_bit(aa, true);
                            if (!(P_space.has_det(new_det))) {
                                std::vector<double> coupling(nroot, 0.0);
                                for (int n = 0; n < nroot; ++n) {
                                    coupling[n] += HIJ * evecs->get(P, n);
                                }
                                // thread_ex_dets[i * noalpha + a] =
                                // std::make_pair(new_det,coupling);
                                thread_ex_dets.push_back(std::make_pair(new_det, coupling));
                            }
                        }
                    }
                }
            }

            // Generate beta excitations
            for (int i = 0; i < nobeta; ++i) {
                int ii = bocc[i];
                for (int a = 0; a < nvbeta; ++a) {
                    int aa = bvir[a];
                    if ((mo_symmetry_[ii] ^ mo_symmetry_[aa]) == 0) {
                        double HIJ = fci_ints_->slater_rules_single_beta(det, ii, aa);
                        if ((std::fabs(HIJ) * evecs_P_row_norm >= screen_thresh_)) {
                            // if( std::abs(HIJ * evecs->get(0, P)) > screen_thresh_ ){
                            new_det = det;
                            new_det.set_beta_bit(ii, false);
                            new_det.set_beta_bit(aa, true);
                            if (!(P_space.has_det(new_det))) {
                                std::vector<double> coupling(nroot, 0.0);
                                for (int n = 0; n < nroot; ++n) {
                                    coupling[n] += HIJ * evecs->get(P, n);
                                }
                                // thread_ex_dets[i * nobeta + a] =
                                // std::make_pair(new_det,coupling);
                                thread_ex_dets.push_back(std::make_pair(new_det, coupling));
                            }
                        }
                    }
                }
            }

            // Generate aa excitations
            for (int i = 0; i < noalpha; ++i) {
                int ii = aocc[i];
                for (int j = i + 1; j < noalpha; ++j) {
                    int jj = aocc[j];
                    for (int a = 0; a < nvalpha; ++a) {
                        int aa = avir[a];
                        for (int b = a + 1; b < nvalpha; ++b) {
                            int bb = avir[b];
                            if ((mo_symmetry_[ii] ^ mo_symmetry_[jj] ^ mo_symmetry_[aa] ^
                                 mo_symmetry_[bb]) == 0) {
                                double HIJ = fci_ints_->tei_aa(ii, jj, aa, bb);
                                if ((std::fabs(HIJ) * evecs_P_row_norm >= screen_thresh_)) {
                                    new_det = det;
                                    HIJ *= new_det.double_excitation_aa(ii, jj, aa, bb);
                                    // if( std::abs(HIJ * evecs->get(0, P)) > screen_thresh_ ){

                                    if (!(P_space.has_det(new_det))) {
                                        std::vector<double> coupling(nroot, 0.0);
                                        for (int n = 0; n < nroot; ++n) {
                                            coupling[n] += HIJ * evecs->get(P, n);
                                        }
                                        // thread_ex_dets[i *
                                        // noalpha*noalpha*nvalpha +
                                        // j*nvalpha*noalpha +  a*nvalpha + b ]
                                        // = std::make_pair(new_det,coupling);
                                        thread_ex_dets.push_back(std::make_pair(new_det, coupling));
                                    }
                                }
                            }
                        }
                    }
                }
            }

            // Generate ab excitations
            for (int i = 0; i < noalpha; ++i) {
                int ii = aocc[i];
                for (int j = 0; j < nobeta; ++j) {
                    int jj = bocc[j];
                    for (int a = 0; a < nvalpha; ++a) {
                        int aa = avir[a];
                        for (int b = 0; b < nvbeta; ++b) {
                            int bb = bvir[b];
                            if ((mo_symmetry_[ii] ^ mo_symmetry_[jj] ^ mo_symmetry_[aa] ^
                                 mo_symmetry_[bb]) == 0) {
                                double HIJ = fci_ints_->tei_ab(ii, jj, aa, bb);
                                if ((std::fabs(HIJ) * evecs_P_row_norm >= screen_thresh_)) {
                                    new_det = det;
                                    HIJ *= new_det.double_excitation_ab(ii, jj, aa, bb);
                                    // if( std::abs(HIJ * evecs->get(0, P)) > screen_thresh_ ){

                                    if (!(P_space.has_det(new_det))) {
                                        std::vector<double> coupling(nroot, 0.0);
                                        for (int n = 0; n < nroot; ++n) {
                                            coupling[n] += HIJ * evecs->get(P, n);
                                        }
                                        // thread_ex_dets[i * nobeta * nvalpha
                                        // *nvbeta + j * bvalpha * nvbeta + a *
                                        // nvalpha]
                                        thread_ex_dets.push_back(std::make_pair(new_det, coupling));
                                    }
                                }
                            }
                        }
                    }
                }
            }

            // Generate bb excitations
            for (int i = 0; i < nobeta; ++i) {
                int ii = bocc[i];
                for (int j = i + 1; j < nobeta; ++j) {
                    int jj = bocc[j];
                    for (int a = 0; a < nvbeta; ++a) {
                        int aa = bvir[a];
                        for (int b = a + 1; b < nvbeta; ++b) {
                            int bb = bvir[b];
                            if ((mo_symmetry_[ii] ^
                                 (mo_symmetry_[jj] ^ (mo_symmetry_[aa] ^ mo_symmetry_[bb]))) == 0) {
                                double HIJ = fci_ints_->tei_bb(ii, jj, aa, bb);
                                if ((std::fabs(HIJ) * evecs_P_row_norm >= screen_thresh_)) {
                                    // if( std::abs(HIJ * evecs->get(0, P)) >= screen_thresh_ ){
                                    new_det = det;
                                    HIJ *= new_det.double_excitation_bb(ii, jj, aa, bb);

                                    if (!(P_space.has_det(new_det))) {
                                        std::vector<double> coupling(nroot, 0.0);
                                        for (int n = 0; n < nroot; ++n) {
                                            coupling[n] += HIJ * evecs->get(P, n);
                                        }
                                        thread_ex_dets.push_back(std::make_pair(new_det, coupling));
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

#pragma omp critical
        {
            for (size_t I = 0, maxI = thread_ex_dets.size(); I < maxI; ++I) {
                std::vector<double>& coupling = thread_ex_dets[I].second;
                Determinant& det = thread_ex_dets[I].first;
                if (V_hash.count(det) != 0) {
                    for (int n = 0; n < nroot; ++n) {
                        V_hash[det][n] += coupling[n];
                    }
                } else {
                    V_hash[det] = coupling;
                }
            }
        }
    } // Close threads
}

void AdaptiveCI::get_core_excited_determinants(SharedMatrix evecs, DeterminantHashVec& P_space,
                                               det_hash<std::vector<double>>& V_hash) {
    size_t max_P = P_space.size();
    const det_hashvec& P_dets = P_space.wfn_hash();
    int nroot = 1;

// Loop over reference determinants
#pragma omp parallel
    {
        int num_thread = omp_get_num_threads();
        int tid = omp_get_thread_num();
        size_t bin_size = max_P / num_thread;
        bin_size += (tid < (max_P % num_thread)) ? 1 : 0;
        size_t start_idx =
            (tid < (max_P % num_thread))
                ? tid * bin_size
                : (max_P % num_thread) * (bin_size + 1) + (tid - (max_P % num_thread)) * bin_size;
        size_t end_idx = start_idx + bin_size;

        if (omp_get_thread_num() == 0 and !quiet_mode_) {
            outfile->Printf("\n  Using %d threads.", num_thread);
        }
        // This will store the excited determinant info for each thread
        std::vector<std::pair<Determinant, std::vector<double>>>
            thread_ex_dets; //( noalpha * nvalpha  );

        for (size_t P = start_idx; P < end_idx; ++P) {
            const Determinant& det(P_dets[P]);
            double evecs_P_row_norm = evecs->get_row(0, P)->norm();

            std::vector<int> aocc = det.get_alfa_occ(nact_); // TODO check size
            std::vector<int> bocc = det.get_beta_occ(nact_); // TODO check size
            std::vector<int> avir = det.get_alfa_vir(nact_); // TODO check size
            std::vector<int> bvir = det.get_beta_vir(nact_); // TODO check size

            int noalpha = aocc.size();
            int nobeta = bocc.size();
            int nvalpha = avir.size();
            int nvbeta = bvir.size();
            Determinant new_det(det);

            // Generate alpha excitations
            for (int i = 0; i < noalpha; ++i) {
                int ii = aocc[i];
                for (int a = 0; a < nvalpha; ++a) {
                    int aa = avir[a];
                    if (((mo_symmetry_[ii] ^ mo_symmetry_[aa]) == 0) and
                        ((aa != hole_) or (!det.get_beta_bit(aa)))) {
                        double HIJ = fci_ints_->slater_rules_single_alpha(det, ii, aa);
                        if ((std::fabs(HIJ) * evecs_P_row_norm >= screen_thresh_)) {
                            //      if( std::abs(HIJ * evecs->get(0, P)) > screen_thresh_ ){
                            new_det = det;
                            new_det.set_alfa_bit(ii, false);
                            new_det.set_alfa_bit(aa, true);
                            if (!(P_space.has_det(new_det))) {
                                std::vector<double> coupling(nroot, 0.0);
                                for (int n = 0; n < nroot; ++n) {
                                    coupling[n] += HIJ * evecs->get(P, n);
                                }
                                // thread_ex_dets[i * noalpha + a] =
                                // std::make_pair(new_det,coupling);
                                thread_ex_dets.push_back(std::make_pair(new_det, coupling));
                            }
                        }
                    }
                }
            }
            // Generate beta excitations
            for (int i = 0; i < nobeta; ++i) {
                int ii = bocc[i];
                for (int a = 0; a < nvbeta; ++a) {
                    int aa = bvir[a];
                    if (((mo_symmetry_[ii] ^ mo_symmetry_[aa]) == 0) and
                        ((aa != hole_) or (!det.get_alfa_bit(aa)))) {
                        double HIJ = fci_ints_->slater_rules_single_beta(det, ii, aa);
                        if ((std::fabs(HIJ) * evecs_P_row_norm >= screen_thresh_)) {
                            // if( std::abs(HIJ * evecs->get(0, P)) > screen_thresh_ ){
                            new_det = det;
                            new_det.set_beta_bit(ii, false);
                            new_det.set_beta_bit(aa, true);
                            if (!(P_space.has_det(new_det))) {
                                std::vector<double> coupling(nroot, 0.0);
                                for (int n = 0; n < nroot; ++n) {
                                    coupling[n] += HIJ * evecs->get(P, n);
                                }
                                // thread_ex_dets[i * nobeta + a] =
                                // std::make_pair(new_det,coupling);
                                thread_ex_dets.push_back(std::make_pair(new_det, coupling));
                            }
                        }
                    }
                }
            }

            // Generate aa excitations
            for (int i = 0; i < noalpha; ++i) {
                int ii = aocc[i];
                for (int j = i + 1; j < noalpha; ++j) {
                    int jj = aocc[j];
                    for (int a = 0; a < nvalpha; ++a) {
                        int aa = avir[a];
                        for (int b = a + 1; b < nvalpha; ++b) {
                            int bb = avir[b];
                            if (((mo_symmetry_[ii] ^ mo_symmetry_[jj] ^ mo_symmetry_[aa] ^
                                  mo_symmetry_[bb]) == 0) and
                                (aa != hole_ and bb != hole_)) {
                                double HIJ = fci_ints_->tei_aa(ii, jj, aa, bb);
                                if ((std::fabs(HIJ) * evecs_P_row_norm >= screen_thresh_)) {
                                    new_det = det;
                                    HIJ *= new_det.double_excitation_aa(ii, jj, aa, bb);
                                    // if( std::abs(HIJ * evecs->get(0, P)) > screen_thresh_ ){

                                    if (!(P_space.has_det(new_det))) {
                                        std::vector<double> coupling(nroot, 0.0);
                                        for (int n = 0; n < nroot; ++n) {
                                            coupling[n] += HIJ * evecs->get(P, n);
                                        }
                                        // thread_ex_dets[i *
                                        // noalpha*noalpha*nvalpha +
                                        // j*nvalpha*noalpha +  a*nvalpha + b ]
                                        // = std::make_pair(new_det,coupling);
                                        thread_ex_dets.push_back(std::make_pair(new_det, coupling));
                                    }
                                }
                            }
                        }
                    }
                }
            }

            // Generate ab excitations
            for (int i = 0; i < noalpha; ++i) {
                int ii = aocc[i];
                for (int j = 0; j < nobeta; ++j) {
                    int jj = bocc[j];
                    for (int a = 0; a < nvalpha; ++a) {
                        int aa = avir[a];
                        for (int b = 0; b < nvbeta; ++b) {
                            int bb = bvir[b];
                            if (((mo_symmetry_[ii] ^ mo_symmetry_[jj] ^ mo_symmetry_[aa] ^
                                  mo_symmetry_[bb]) == 0) and
                                (aa != hole_ and bb != hole_)) {
                                double HIJ = fci_ints_->tei_ab(ii, jj, aa, bb);
                                if ((std::fabs(HIJ) * evecs_P_row_norm >= screen_thresh_)) {
                                    new_det = det;
                                    HIJ *= new_det.double_excitation_ab(ii, jj, aa, bb);
                                    // if( std::abs(HIJ * evecs->get(0, P)) > screen_thresh_ ){

                                    if (!(P_space.has_det(new_det))) {
                                        std::vector<double> coupling(nroot, 0.0);
                                        for (int n = 0; n < nroot; ++n) {
                                            coupling[n] += HIJ * evecs->get(P, n);
                                        }
                                        // thread_ex_dets[i * nobeta * nvalpha
                                        // *nvbeta + j * bvalpha * nvbeta + a *
                                        // nvalpha]
                                        thread_ex_dets.push_back(std::make_pair(new_det, coupling));
                                    }
                                }
                            }
                        }
                    }
                }
            }

            // Generate bb excitations
            for (int i = 0; i < nobeta; ++i) {
                int ii = bocc[i];
                for (int j = i + 1; j < nobeta; ++j) {
                    int jj = bocc[j];
                    for (int a = 0; a < nvbeta; ++a) {
                        int aa = bvir[a];
                        for (int b = a + 1; b < nvbeta; ++b) {
                            int bb = bvir[b];
                            if (((mo_symmetry_[ii] ^
                                  (mo_symmetry_[jj] ^ (mo_symmetry_[aa] ^ mo_symmetry_[bb]))) ==
                                 0) and
                                (aa != hole_ and bb != hole_)) {
                                double HIJ = fci_ints_->tei_bb(ii, jj, aa, bb);
                                if ((std::fabs(HIJ) * evecs_P_row_norm >= screen_thresh_)) {
                                    // if( std::abs(HIJ * evecs->get(0, P)) >= screen_thresh_ ){
                                    new_det = det;
                                    HIJ *= new_det.double_excitation_bb(ii, jj, aa, bb);

                                    if (!(P_space.has_det(new_det))) {
                                        std::vector<double> coupling(nroot, 0.0);
                                        for (int n = 0; n < nroot; ++n) {
                                            coupling[n] += HIJ * evecs->get(P, n);
                                        }
                                        thread_ex_dets.push_back(std::make_pair(new_det, coupling));
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
#pragma omp critical
        {
            for (size_t I = 0, maxI = thread_ex_dets.size(); I < maxI; ++I) {
                std::vector<double>& coupling = thread_ex_dets[I].second;
                Determinant& det = thread_ex_dets[I].first;
                if (V_hash.count(det) != 0) {
                    for (int n = 0; n < nroot; ++n) {
                        V_hash[det][n] += coupling[n];
                    }
                } else {
                    V_hash[det] = coupling;
                }
            }
        }
    } // Close threads
}

double AdaptiveCI::get_excited_determinants_batch( SharedMatrix evecs, SharedVector evals,
                                                 DeterminantHashVec& P_space,
                            std::vector<std::pair<double,Determinant>>& F_space ) {
    const size_t n_dets = P_space.size();

    int nmo = fci_ints_->nmo();
    double max_mem = options_.get_double("ACI_MAX_MEM");
    double aci_scale = options_.get_double("ACI_SCALE_SIGMA");

    size_t guess_size = n_dets * nmo * nmo;
    double guess_mem = guess_size * 400.0e-7; //Est of map size in MB
    int nruns = static_cast<int>(std::ceil(guess_mem/max_mem));

    double total_excluded = 0.0;

    #pragma omp parallel
    {
        int thread_id  = omp_get_thread_num();
        int n_threads = omp_get_max_threads();
        int nbin = nruns*n_threads;    
        if( thread_id == 0 ){
            outfile->Printf("\n  Setting nbin to %d based on estimated memory (%6.3f MB)", nbin, guess_mem);
        }

        if (n_threads >= nbin) {
            nbin = n_threads;
            if( thread_id == 0 ){
                outfile->Printf("\n  WARNING: Batching not required, use another algorithm");
            }
        }
    
        if (options_["ACI_NBATCH"].has_changed()) {
            nbin = options_.get_int("ACI_NBATCH");
            if( thread_id == 0 ){
                outfile->Printf("\n  Overwriting nbin to %d based on user input", nbin);
            }
        } 
        std::vector<std::pair<double,Determinant>> F_td;
    
        int bin_size = nbin / n_threads;
        bin_size += (thread_id < (nbin % n_threads)) ? 1 : 0;
        int start_idx =
            (thread_id < (nbin % n_threads))
               ? thread_id * bin_size
                : (nbin % n_threads) * (bin_size + 1) + (thread_id - (nbin % n_threads)) * bin_size;
        int end_idx = start_idx + bin_size;
    
        if (thread_id == 0) {
            outfile->Printf("\n  Number of bins for exitation space:  %d", nbin);
            outfile->Printf("\n  Number of threads: %d", n_threads);
        }
    
        // Loop over bins
        for (int bin = start_idx; bin < end_idx; ++bin) {

            Timer sp;
            // 1. Build the full bin-subset
            det_hash<double> A_b = get_bin_F_space(bin,nbin,evecs,P_space);
            outfile->Printf("\n F subspace (bin %d) takes %1.6f s",bin, sp.get()); 

            // 2. Put the dets/vals in a sortable list (F_tmp)
            std::vector<std::pair<double,Determinant>> F_tmp; 
            for( auto& det_p : A_b ){
                auto& det = det_p.first;
                double& V = det_p.second;
        
                double delta = fci_ints_->energy(det) - evals->get(0); 
        
                F_tmp.push_back( std::make_pair( std::fabs(0.5*(delta - sqrt(delta*delta + V*V*4.0))), det ));
            }
            A_b.clear();
            outfile->Printf("\n  Size of F subspace: %zu dets (bin %d)", F_tmp.size(), bin);
        
            // 3. Sort the list
            std::sort( F_tmp.begin(), F_tmp.end(), pair_comp);
        
            // 4. Screen subspaces
            double b_sigma = sigma_ * (aci_scale/nbin);
//            outfile->Printf("\n  Using sigma = %1.5f", b_sigma);
            double excluded = 0.0;
            size_t ndet = 0; 
            for( size_t I = 0, max_I = F_tmp.size(); I < max_I; ++I ){
                double en = F_tmp[I].first;
                Determinant& det = F_tmp[I].second;
                if( excluded + en < b_sigma ){
                    excluded += en;
                } else {
                    F_td.push_back(std::make_pair(en, det));
                    ndet++;
                }
            }
            outfile->Printf("\n  Added %zu dets from bin %d, (td %d)", ndet, bin, thread_id); 
            total_excluded += excluded;

        } // End iteration over bins


        //6. Merge thread spaces
        #pragma omp critical
        {
            for( auto& pair : F_td ){
                F_space.push_back(pair);
            }
        }
    } // close threads

    outfile->Printf("\n  Screened out %1.10f Eh of correlation", total_excluded);
    return total_excluded;
}



det_hash<double> AdaptiveCI::get_bin_F_space( int bin, int nbin, SharedMatrix evecs,
                                                 DeterminantHashVec& P_space) {

    const size_t n_dets = P_space.size();
    const det_hashvec& dets = P_space.wfn_hash();
    int nmo = fci_ints_->nmo();
    std::vector<int> act_mo = mo_space_info_->get_dimension("ACTIVE").blocks();
    det_hash<double> A_b;

    // Loop over P space determinants
    for (size_t I = 0; I < n_dets; ++I) {
        double c_I = evecs->get(I,0);
        const Determinant& det = dets[I];
        std::vector<std::vector<int>> noalpha = det.get_asym_occ(nmo,act_mo);
        std::vector<std::vector<int>> nobeta  = det.get_bsym_occ(nmo,act_mo);
        std::vector<std::vector<int>> nvalpha = det.get_asym_vir(nmo,act_mo);
        std::vector<std::vector<int>> nvbeta  = det.get_bsym_vir(nmo,act_mo);

        std::vector<int> aocc = det.get_alfa_occ(nmo);
        std::vector<int> bocc = det.get_beta_occ(nmo);
        std::vector<int> avir = det.get_alfa_vir(nmo);
        std::vector<int> bvir = det.get_beta_vir(nmo);


        Determinant new_det(det);
        // Generate alpha excitations
        for( int h = 0; h < nirrep_; ++h){
            for (auto& ii : noalpha[h]) {
                for (auto& aa : nvalpha[h]) {
                    new_det = det;
                    new_det.set_alfa_bit(ii, false);
                    new_det.set_alfa_bit(aa, true);
                    size_t hash_val = Determinant::Hash()(new_det);
                    if ( (hash_val % nbin) == bin) {
                        double HIJ = fci_ints_->slater_rules_single_alpha(det, ii, aa) * c_I;
                        if ((std::fabs(HIJ) >= screen_thresh_)) {
                            if (A_b.count(new_det) == 0) {
                                A_b[new_det] = HIJ;
                            } else {
                                A_b[new_det] += HIJ;
                            }
                        }
                    }
                }
            }
            // Generate beta excitations
            for (auto& ii : nobeta[h]) {
                for (auto& aa : nvbeta[h]) {
                    // Check if the determinant goes in this bin
                    new_det = det;
                    new_det.set_beta_bit(ii, false);
                    new_det.set_beta_bit(aa, true);
                    size_t hash_val = Determinant::Hash()(new_det);
                    if ((hash_val % nbin) == bin) {
                        double HIJ = fci_ints_->slater_rules_single_beta(det, ii, aa) * c_I;
                        if ((std::fabs(HIJ) >= screen_thresh_)) {
                            if (A_b.count(new_det) == 0) {
                                A_b[new_det] = HIJ;
                            } else {
                                A_b[new_det] += HIJ;
                            }
                        }
                    }
                }
            }
        }
        for ( int p = 0; p < nirrep_; ++p){
            for ( int q = p; q < nirrep_; ++q){
                for ( int r = 0; r < nirrep_; ++r){
                    int sp = p^q^r;
                    if( sp < r) continue;

                    // Generate aa excitations
                    for (int i = 0, max_i = noalpha[p].size(); i < max_i; ++i) {
                        int ii = noalpha[p][i];
                        for (int j = (p == q ? i + 1 : 0), maxj = noalpha[q].size(); j < maxj; ++j) {
                            int jj = noalpha[q][j];
                            for (int a = 0, max_a = nvalpha[r].size(); a < max_a; ++a) {
                                int aa = nvalpha[r][a];
                                for (int b = (r == sp ? a + 1 : 0), maxb = nvalpha[sp].size(); b < maxb; ++b) {
                                    int bb = nvalpha[sp][b];

                                    // Check if the determinant goes in this bin
                                    new_det = det;
                                    double sign = new_det.double_excitation_aa(ii, jj, aa, bb);
                                    size_t hash_val = Determinant::Hash()(new_det);
                                    if ((hash_val % nbin) == bin) {
                                        double HIJ = fci_ints_->tei_aa(ii, jj, aa, bb) * sign * c_I;
                                        if ((std::fabs(HIJ) >= screen_thresh_)) {
                                            if (A_b.count(new_det) == 0) {
                                                A_b[new_det] = HIJ;
                                            } else {
                                                A_b[new_det] +=  HIJ;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                    // Generate bb excitations
                    for (int i = 0, max_i = nobeta[p].size(); i < max_i; ++i) {
                        int ii = nobeta[p][i];
                        for (int j = (p == q ? i + 1 : 0), maxj = nobeta[q].size(); j < maxj; ++j) {
                            int jj = nobeta[q][j];
                            for (int a = 0, max_a = nvbeta[r].size(); a < max_a; ++a) {
                                int aa = nvbeta[r][a];
                                for (int b = (r == sp ? a + 1 : 0), maxb = nvbeta[sp].size(); b < maxb; ++b) {
                                    int bb = nvbeta[sp][b];
                                    // Check if the determinant goes in this bin
                                    new_det = det;
                                    double sign = new_det.double_excitation_bb(ii, jj, aa, bb);
                                    size_t hash_val = Determinant::Hash()(new_det);
                                    if ((hash_val % nbin) == bin) {
                                        double HIJ = fci_ints_->tei_bb(ii, jj, aa, bb) * sign * c_I;
                                        if ((std::fabs(HIJ) >= screen_thresh_)) {
                                            if (A_b.count(new_det) == 0) {
                                                A_b[new_det] = HIJ;
                                            } else {
                                                A_b[new_det] +=  HIJ;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        for ( int p = 0; p < nirrep_; ++p){
            for ( int q = 0; q < nirrep_; ++q){
                for ( int r = 0; r < nirrep_; ++r){
                    int sp = p^q^r;
                    // Generate ab excitations
                    for (auto& ii : noalpha[p]) {
                        for (auto& jj : nobeta[q]) {
                            for (auto& aa : nvalpha[r]) {
                                for (auto& bb : nvbeta[sp]) {
                                    new_det = det;
                                    double sign = new_det.double_excitation_ab(ii, jj, aa, bb);
                                    size_t hash_val = Determinant::Hash()(new_det);
                                    if ((hash_val % nbin) == bin) {
                                        double HIJ = fci_ints_->tei_ab(ii, jj, aa, bb) * sign * c_I;
                                        if ((std::fabs(HIJ) >= screen_thresh_)) {
                                            if (A_b.count(new_det) == 0) {
                                                A_b[new_det] = HIJ;
                                            } else {
                                                A_b[new_det] += HIJ;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }// end loop over reference

    // Remove duplicates
    for( det_hashvec::iterator it = dets.begin(), endit = dets.end(); it != endit; ++it) {
        A_b[*it] = 0.0;
    }  

    return A_b;
}
}}
