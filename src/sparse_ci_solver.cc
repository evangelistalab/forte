#include <cmath>

#include <boost/timer.hpp>

#include <libmints/matrix.h>
#include <libmints/vector.h>
#include <libciomr/libciomr.h>
#include <libqt/qt.h>

#include "forte-def.h"
#include "iterative_solvers.h"
#include "sparse_ci_solver.h"
#include "fci_vector.h"

namespace psi{ namespace forte{

void SigmaVectorFull::compute_sigma(Matrix& sigma, Matrix &b, int nroot){
    sigma.gemm(false,true,1.0,H_,b,0.0);
}

void SigmaVectorFull::get_diagonal(Vector& diag){
    double** h = H_->pointer();
    for (size_t I = 0; I < size_; ++I){
        diag.set(I,h[I][I]);
    }
}

void SigmaVectorSparse::compute_sigma(Matrix& sigma, Matrix &b, int nroot){
    double** sigma_p = sigma.pointer();
    double** b_p = b.pointer();
    for (size_t J = 0; J < size_; ++J){
        for (int r = 0; r < nroot; ++r){
            sigma_p[J][r] = 0.0;
        }
        std::vector<double>& H_row = H_[J].second;
        std::vector<int>& index_row = H_[J].first;
        size_t maxc = index_row.size();
        for (size_t c = 0; c < maxc; ++c){
            int K = index_row[c];
            double HJK = H_row[c];
            for (int r = 0; r < nroot; ++r){
                sigma_p[J][r] +=  HJK * b_p[r][K];
            }
        }
    }
}

void SigmaVectorSparse::get_diagonal(Vector& diag){
    for (size_t I = 0; I < size_; ++I){
        diag.set(I,H_[I].second[0]);
    }
}

SigmaVectorList::SigmaVectorList(const std::vector<STLBitsetDeterminant>& space)
    : SigmaVector(space.size()), space_(space)
{
	using det_hash = std::unordered_map<STLBitsetDeterminant,size_t,STLBitsetDeterminant::Hash>;
    using bstmap_it = det_hash::iterator;

    size_t max_I = space.size();

    size_t na_ann = 0;
    size_t nb_ann = 0;
    size_t naa_ann = 0;
    size_t nab_ann = 0;
    size_t nbb_ann = 0;
   // std::map<STLBitsetDeterminant,size_t> map_a_ann;
   // std::map<STLBitsetDeterminant,size_t> map_b_ann;

   // std::map<STLBitsetDeterminant,size_t> map_aa_ann;
   // std::map<STLBitsetDeterminant,size_t> map_ab_ann;
   // std::map<STLBitsetDeterminant,size_t> map_bb_ann;
	det_hash map_a_ann;
	det_hash map_b_ann;
	
	det_hash map_aa_ann;
	det_hash map_ab_ann;
	det_hash map_bb_ann;

    a_ann_list.resize(max_I);
    b_ann_list.resize(max_I);
    aa_ann_list.resize(max_I);
    ab_ann_list.resize(max_I);
    bb_ann_list.resize(max_I);

    outfile->Printf("\n  Generating determinants with N-1 electrons.\n");
    for (size_t I = 0; I < max_I; ++I){
        STLBitsetDeterminant detI = space[I];

        double EI = detI.energy();
        diag_.push_back(EI);

        std::vector<int> aocc = detI.get_alfa_occ();
        std::vector<int> bocc = detI.get_beta_occ();

        int noalpha = aocc.size();
        int nobeta  = bocc.size();

        std::vector<std::pair<size_t,short>> a_ann(noalpha);
        std::vector<std::pair<size_t,short>> b_ann(nobeta);

        // Generate alpha annihilation
        for (int i = 0; i < noalpha; ++i){
            int ii = aocc[i];
            STLBitsetDeterminant detJ(detI);
            detJ.set_alfa_bit(ii,false);

            double sign = detI.slater_sign_alpha(ii);

            bstmap_it it = map_a_ann.find(detJ);
            size_t detJ_add;
            // detJ is not in the map, add it
            if (it == map_a_ann.end()){
                detJ_add = na_ann;
                map_a_ann[detJ] = na_ann;
                na_ann++;
            }else{
                detJ_add = it->second;
            }
            a_ann[i] = std::make_pair(detJ_add,(sign > 0.5) ? (ii + 1) : (-ii-1));
        }
        a_ann_list[I] = a_ann;
        // Generate beta annihilation
        for (int i = 0; i < nobeta; ++i){
            int ii = bocc[i];
            STLBitsetDeterminant detJ(detI);
            detJ.set_beta_bit(ii,false);

            double sign = detI.slater_sign_beta(ii);

            bstmap_it it = map_b_ann.find(detJ);
            size_t detJ_add;
            // detJ is not in the map, add it
            if (it == map_b_ann.end()){
                detJ_add = nb_ann;
                map_b_ann[detJ] = nb_ann;
                nb_ann++;
            }else{
                detJ_add = it->second;
            }
            b_ann[i] = std::make_pair(detJ_add,(sign > 0.5) ? (ii + 1) : (-ii-1));
        }
        b_ann_list[I] = b_ann;
    }

    a_cre_list.resize(map_a_ann.size());
    b_cre_list.resize(map_b_ann.size());


    size_t num_tuples_sigles = 0;
    for (size_t I = 0; I < max_I; ++I){
        const std::vector<std::pair<size_t,short>>& a_ann = a_ann_list[I];
        for (const std::pair<size_t,short>& J_sign : a_ann){
            size_t J = J_sign.first;
            short sign = J_sign.second;
            a_cre_list[J].push_back(std::make_pair(I,sign));
            num_tuples_sigles++;
        }
        const std::vector<std::pair<size_t,short>>& b_ann = b_ann_list[I];
        for (const std::pair<size_t,short>& J_sign : b_ann){
            size_t J = J_sign.first;
            short sign = J_sign.second;
            b_cre_list[J].push_back(std::make_pair(I,sign));
            num_tuples_sigles++;
        }
    }
    size_t mem_tuple_singles = num_tuples_sigles * (sizeof(size_t) + sizeof(short));

    //    outfile->Printf("\n  Size of lists:");
    //    outfile->Printf("\n  |I> ->  a_p |I>: %zu",a_ann_list.size());
    //    outfile->Printf("\n  |I> ->  a_p |I>: %zu",b_ann_list.size());
    //    outfile->Printf("\n  |A> -> a+_q |A>: %zu",a_cre_list.size());
    //    outfile->Printf("\n  |A> -> a+_q |A>: %zu",b_cre_list.size());


    outfile->Printf("\n  Generating determinants with N-2 electrons.\n");
    for (size_t I = 0; I < max_I; ++I){
        STLBitsetDeterminant detI = space[I];

        std::vector<int> aocc = detI.get_alfa_occ();
        std::vector<int> bocc = detI.get_beta_occ();

        size_t noalpha = aocc.size();
        size_t nobeta  = bocc.size();

        std::vector<std::tuple<size_t,short,short>> aa_ann(noalpha * (noalpha - 1) / 2);
        std::vector<std::tuple<size_t,short,short>> ab_ann(noalpha * nobeta);
        std::vector<std::tuple<size_t,short,short>> bb_ann(nobeta * (nobeta - 1) / 2);

        // Generate alpha-alpha annihilation
        for (size_t i = 0, ij = 0; i < noalpha; ++i){
            for (size_t j = i + 1; j < noalpha; ++j, ++ij){
                int ii = aocc[i];
                int jj = aocc[j];
                STLBitsetDeterminant detJ(detI);
                detJ.set_alfa_bit(ii,false);
                detJ.set_alfa_bit(jj,false);

                double sign = detI.slater_sign_alpha(ii) * detI.slater_sign_alpha(jj);

                bstmap_it it = map_aa_ann.find(detJ);
                size_t detJ_add;
                // detJ is not in the map, add it
                if (it == map_aa_ann.end()){
                    detJ_add = naa_ann;
                    map_aa_ann[detJ] = naa_ann;
                    naa_ann++;
                }else{
                    detJ_add = it->second;
                }
                aa_ann[ij] = std::make_tuple(detJ_add,(sign > 0.5) ? (ii + 1) : (-ii-1),jj);
            }
        }
        aa_ann_list[I] = aa_ann;
        // Generate beta-beta annihilation
        for (size_t i = 0, ij = 0; i < nobeta; ++i){
            for (size_t j = i + 1; j < nobeta; ++j, ++ij){
                int ii = bocc[i];
                int jj = bocc[j];
                STLBitsetDeterminant detJ(detI);
                detJ.set_beta_bit(ii,false);
                detJ.set_beta_bit(jj,false);

                double sign = detI.slater_sign_beta(ii) * detI.slater_sign_beta(jj);;

                bstmap_it it = map_bb_ann.find(detJ);
                size_t detJ_add;
                // detJ is not in the map, add it
                if (it == map_bb_ann.end()){
                    detJ_add = nbb_ann;
                    map_bb_ann[detJ] = nbb_ann;
                    nbb_ann++;
                }else{
                    detJ_add = it->second;
                }
                bb_ann[ij] = std::make_tuple(detJ_add,(sign > 0.5) ? (ii + 1) : (-ii-1),jj);
            }
        }
        bb_ann_list[I] = bb_ann;
        for (size_t i = 0, ij = 0; i < noalpha; ++i){
            for (size_t j = 0; j < nobeta; ++j, ++ij){
                int ii = aocc[i];
                int jj = bocc[j];
                STLBitsetDeterminant detJ(detI);
                detJ.set_alfa_bit(ii,false);
                detJ.set_beta_bit(jj,false);

                double sign = detI.slater_sign_alpha(ii) * detI.slater_sign_beta(jj);

                bstmap_it it = map_ab_ann.find(detJ);
                size_t detJ_add;
                // detJ is not in the map, add it
                if (it == map_ab_ann.end()){
                    detJ_add = nab_ann;
                    map_ab_ann[detJ] = nab_ann;
                    nab_ann++;
                }else{
                    detJ_add = it->second;
                }
                ab_ann[ij] = std::make_tuple(detJ_add,(sign > 0.5) ? (ii + 1) : (-ii-1),jj);
            }
        }
        ab_ann_list[I] = ab_ann;
    }

    aa_cre_list.resize(map_aa_ann.size());
    ab_cre_list.resize(map_ab_ann.size());
    bb_cre_list.resize(map_bb_ann.size());


    size_t num_tuples_doubles = 0;
    for (size_t I = 0; I < max_I; ++I){
        const std::vector<std::tuple<size_t,short,short>>& aa_ann = aa_ann_list[I];
        for (const std::tuple<size_t,short,short>& J_sign : aa_ann){
            size_t J = std::get<0>(J_sign);
            short i = std::get<1>(J_sign);
            short j = std::get<2>(J_sign);
            aa_cre_list[J].push_back(std::make_tuple(I,i,j));
            num_tuples_doubles++;
        }
        const std::vector<std::tuple<size_t,short,short>>& bb_ann = bb_ann_list[I];
        for (const std::tuple<size_t,short,short>& J_sign : bb_ann){
            size_t J = std::get<0>(J_sign);
            short i = std::get<1>(J_sign);
            short j = std::get<2>(J_sign);
            bb_cre_list[J].push_back(std::make_tuple(I,i,j));
            num_tuples_doubles++;
        }
        const std::vector<std::tuple<size_t,short,short>>& ab_ann = ab_ann_list[I];
        for (const std::tuple<size_t,short,short>& J_sign : ab_ann){
            size_t J = std::get<0>(J_sign);
            short i = std::get<1>(J_sign);
            short j = std::get<2>(J_sign);
            ab_cre_list[J].push_back(std::make_tuple(I,i,j));
            num_tuples_doubles++;
        }
    }

    size_t mem_tuple_doubles = num_tuples_doubles * (sizeof(size_t) + 2 * sizeof(short));

    //    outfile->Printf("\n  Size of lists:");
    //    outfile->Printf("\n  |I> ->  a_p |I>: %zu",aa_ann_list.size());
    //    outfile->Printf("\n  |I> ->  a_p |I>: %zu",ab_ann_list.size());
    //    outfile->Printf("\n  |I> ->  a_p |I>: %zu",bb_ann_list.size());
    //    outfile->Printf("\n  |A> -> a+_q |A>: %zu",aa_cre_list.size());
    //    outfile->Printf("\n  |A> -> a+_q |A>: %zu",ab_cre_list.size());
    //    outfile->Printf("\n  |A> -> a+_q |A>: %zu",bb_cre_list.size());
    outfile->Printf("\n  Memory for singles: %f MB",double(mem_tuple_singles) / (1024. * 1024.) ); // Convert to MB
    outfile->Printf("\n  Memory for doubles: %f MB",double(mem_tuple_doubles) / (1024. * 1024.) );
}

void SigmaVectorList::compute_sigma(Matrix& sigma, Matrix& b, int nroot)
{
    sigma.zero();
    double** sigma_p = sigma.pointer();
    double** b_p = b.pointer();
    for (size_t J = 0; J < size_; ++J){
        // reference
        for (int a = 0; a < nroot; ++a){
            sigma_p[J][a] += diag_[J] * b_p[a][J];
        }

        // aa singles
        for (auto& aJ_mo_sign : a_ann_list[J]){
            const size_t aJ_add = aJ_mo_sign.first;
            const size_t p = std::abs(aJ_mo_sign.second) - 1;
            for (auto& aaJ_mo_sign : a_cre_list[aJ_add]){
                const size_t q = std::abs(aaJ_mo_sign.second) - 1;
                if (p != q){
                    const double HIJ = space_[aaJ_mo_sign.first].slater_rules(space_[J]);
                    const size_t I = aaJ_mo_sign.first;
                    for (int a = 0; a < nroot; ++a){
                        sigma_p[I][a] += HIJ * b_p[a][J];
                    }
                }
            }
        }

        // bb singles
        for (auto& bJ_mo_sign : b_ann_list[J]){
            const size_t bJ_add = bJ_mo_sign.first;
            const size_t p = std::abs(bJ_mo_sign.second) - 1;
            for (auto& bbJ_mo_sign : b_cre_list[bJ_add]){
                const size_t q = std::abs(bbJ_mo_sign.second) - 1;
                if (p != q){
                    const double HIJ = space_[bbJ_mo_sign.first].slater_rules(space_[J]);
                    const size_t I = bbJ_mo_sign.first;
                    for (int a = 0; a < nroot; ++a){
                        sigma_p[I][a] += HIJ * b_p[a][J];
                    }
                }
            }
        }

        // aaaa doubles
        for (auto& aaJ_mo_sign : aa_ann_list[J]){
            const size_t aaJ_add = std::get<0>(aaJ_mo_sign);
            const double sign_pq = std::get<1>(aaJ_mo_sign) > 0.0 ? 1.0 : -1.0;
            const size_t p = std::abs(std::get<1>(aaJ_mo_sign)) - 1;
            const size_t q = std::get<2>(aaJ_mo_sign);
            for (auto& aaaaJ_mo_sign : aa_cre_list[aaJ_add]){
                const size_t r = std::abs(std::get<1>(aaaaJ_mo_sign)) - 1;
                const size_t s = std::get<2>(aaaaJ_mo_sign);
                if ((p != r) and (q != s) and (p != s) and (q != r)){
                    const size_t aaaaJ_add = std::get<0>(aaaaJ_mo_sign);
                    const double sign_rs = std::get<1>(aaaaJ_mo_sign) > 0.0 ? 1.0 : -1.0;
                    const size_t I = aaaaJ_add;
                    const double HIJ = sign_pq * sign_rs * STLBitsetDeterminant::fci_ints_->tei_aa(p,q,r,s);
                    for (int a = 0; a < nroot; ++a){
                        sigma_p[I][a] += HIJ * b_p[a][J];
                    }
                }
            }
        }

        // aabb singles
        for (auto& abJ_mo_sign : ab_ann_list[J]){
            const size_t abJ_add = std::get<0>(abJ_mo_sign);
            const double sign_pq = std::get<1>(abJ_mo_sign) > 0.0 ? 1.0 : -1.0;
            const size_t p = std::abs(std::get<1>(abJ_mo_sign)) - 1;
            const size_t q = std::get<2>(abJ_mo_sign);
            for (auto& ababJ_mo_sign : ab_cre_list[abJ_add]){
                const size_t r = std::abs(std::get<1>(ababJ_mo_sign)) - 1;
                const size_t s = std::get<2>(ababJ_mo_sign);
                if ((p != r) and (q != s)){
                    const size_t ababJ_add = std::get<0>(ababJ_mo_sign);
                    const double sign_rs = std::get<1>(ababJ_mo_sign) > 0.0 ? 1.0 : -1.0;
                    const size_t I = ababJ_add;
                    const double HIJ = sign_pq * sign_rs * STLBitsetDeterminant::fci_ints_->tei_ab(p,q,r,s);
                    for (int a = 0; a < nroot; ++a){
                        sigma_p[I][a] += HIJ * b_p[a][J];
                    }
                }
            }
        }

        // bbbb singles
        for (auto& bbJ_mo_sign : bb_ann_list[J]){
            const size_t bbJ_add = std::get<0>(bbJ_mo_sign);
            const double sign_pq = std::get<1>(bbJ_mo_sign) > 0.0 ? 1.0 : -1.0;
            const size_t p = std::abs(std::get<1>(bbJ_mo_sign)) - 1;
            const size_t q = std::get<2>(bbJ_mo_sign);
            for (auto& bbbbJ_mo_sign : bb_cre_list[bbJ_add]){
                const size_t r = std::abs(std::get<1>(bbbbJ_mo_sign)) - 1;
                const size_t s = std::get<2>(bbbbJ_mo_sign);
                if ((p != r) and (q != s) and (p != s) and (q != r)){
                    const size_t bbbbJ_add = std::get<0>(bbbbJ_mo_sign);
                    const double sign_rs = std::get<1>(bbbbJ_mo_sign) > 0.0 ? 1.0 : -1.0;
                    const size_t I = bbbbJ_add;
                    const double HIJ = sign_pq * sign_rs * STLBitsetDeterminant::fci_ints_->tei_bb(p,q,r,s);
                    for (int a = 0; a < nroot; ++a){
                        sigma_p[I][a] += HIJ * b_p[a][J];
                    }
                }
            }
        }
    }
}

void SigmaVectorList::compute_sigma(SharedVector sigma, SharedVector b)
{
    sigma->zero();
    double* sigma_p = sigma->pointer();
    double* b_p = b->pointer();
    for (size_t J = 0; J < size_; ++J){
        // reference
        sigma_p[J] += diag_[J] * b_p[J];

        // aa singles
        for (auto& aJ_mo_sign : a_ann_list[J]){
            const size_t aJ_add = aJ_mo_sign.first;
            const size_t p = std::abs(aJ_mo_sign.second) - 1;
            for (auto& aaJ_mo_sign : a_cre_list[aJ_add]){
                const size_t q = std::abs(aaJ_mo_sign.second) - 1;
                if (p != q){
                    const double HIJ = space_[aaJ_mo_sign.first].slater_rules(space_[J]);
                    const size_t I = aaJ_mo_sign.first;
                    sigma_p[I] += HIJ * b_p[J];
                }
            }
        }

        // bb singles
        for (auto& bJ_mo_sign : b_ann_list[J]){
            const size_t bJ_add = bJ_mo_sign.first;
            const size_t p = std::abs(bJ_mo_sign.second) - 1;
            for (auto& bbJ_mo_sign : b_cre_list[bJ_add]){
                const size_t q = std::abs(bbJ_mo_sign.second) - 1;
                if (p != q){
                    const double HIJ = space_[bbJ_mo_sign.first].slater_rules(space_[J]);
                    const size_t I = bbJ_mo_sign.first;
                    sigma_p[I] += HIJ * b_p[J];
                }
            }
        }

        // aaaa doubles
        for (auto& aaJ_mo_sign : aa_ann_list[J]){
            const size_t aaJ_add = std::get<0>(aaJ_mo_sign);
            const double sign_pq = std::get<1>(aaJ_mo_sign) > 0.0 ? 1.0 : -1.0;
            const size_t p = std::abs(std::get<1>(aaJ_mo_sign)) - 1;
            const size_t q = std::get<2>(aaJ_mo_sign);
            for (auto& aaaaJ_mo_sign : aa_cre_list[aaJ_add]){
                const size_t r = std::abs(std::get<1>(aaaaJ_mo_sign)) - 1;
                const size_t s = std::get<2>(aaaaJ_mo_sign);
                if ((p != r) and (q != s) and (p != s) and (q != r)){
                    const size_t aaaaJ_add = std::get<0>(aaaaJ_mo_sign);
                    const double sign_rs = std::get<1>(aaaaJ_mo_sign) > 0.0 ? 1.0 : -1.0;
                    const size_t I = aaaaJ_add;
                    const double HIJ = sign_pq * sign_rs * STLBitsetDeterminant::fci_ints_->tei_aa(p,q,r,s);
                    sigma_p[I] += HIJ * b_p[J];
                }
            }
        }

        // aabb singles
        for (auto& abJ_mo_sign : ab_ann_list[J]){
            const size_t abJ_add = std::get<0>(abJ_mo_sign);
            const double sign_pq = std::get<1>(abJ_mo_sign) > 0.0 ? 1.0 : -1.0;
            const size_t p = std::abs(std::get<1>(abJ_mo_sign)) - 1;
            const size_t q = std::get<2>(abJ_mo_sign);
            for (auto& ababJ_mo_sign : ab_cre_list[abJ_add]){
                const size_t r = std::abs(std::get<1>(ababJ_mo_sign)) - 1;
                const size_t s = std::get<2>(ababJ_mo_sign);
                if ((p != r) and (q != s)){
                    const size_t ababJ_add = std::get<0>(ababJ_mo_sign);
                    const double sign_rs = std::get<1>(ababJ_mo_sign) > 0.0 ? 1.0 : -1.0;
                    const size_t I = ababJ_add;
                    const double HIJ = sign_pq * sign_rs * STLBitsetDeterminant::fci_ints_->tei_ab(p,q,r,s);
                    sigma_p[I] += HIJ * b_p[J];
                }
            }
        }

        // bbbb singles
        for (auto& bbJ_mo_sign : bb_ann_list[J]){
            const size_t bbJ_add = std::get<0>(bbJ_mo_sign);
            const double sign_pq = std::get<1>(bbJ_mo_sign) > 0.0 ? 1.0 : -1.0;
            const size_t p = std::abs(std::get<1>(bbJ_mo_sign)) - 1;
            const size_t q = std::get<2>(bbJ_mo_sign);
            for (auto& bbbbJ_mo_sign : bb_cre_list[bbJ_add]){
                const size_t r = std::abs(std::get<1>(bbbbJ_mo_sign)) - 1;
                const size_t s = std::get<2>(bbbbJ_mo_sign);
                if ((p != r) and (q != s) and (p != s) and (q != r)){
                    const size_t bbbbJ_add = std::get<0>(bbbbJ_mo_sign);
                    const double sign_rs = std::get<1>(bbbbJ_mo_sign) > 0.0 ? 1.0 : -1.0;
                    const size_t I = bbbbJ_add;
                    const double HIJ = sign_pq * sign_rs * STLBitsetDeterminant::fci_ints_->tei_bb(p,q,r,s);
                    sigma_p[I] += HIJ * b_p[J];
                }
            }
        }
    }
}

void SigmaVectorList::get_hamiltonian(Matrix& H)
{
    double** h_p = H.pointer();
    for (size_t J = 0; J < size_; ++J){
        // reference
        h_p[J][J] = diag_[J];

        // aa singles
        for (auto& aJ_mo_sign : a_ann_list[J]){
            const size_t aJ_add = aJ_mo_sign.first;
            const size_t p = std::abs(aJ_mo_sign.second) - 1;
            for (auto& aaJ_mo_sign : a_cre_list[aJ_add]){
                const size_t q = std::abs(aaJ_mo_sign.second) - 1;
                if (p != q){
                    const double HIJ = space_[aaJ_mo_sign.first].slater_rules(space_[J]);
                    const size_t I = aaJ_mo_sign.first;
                    h_p[I][J] = HIJ;
                }
            }
        }

        // bb singles
        for (auto& bJ_mo_sign : b_ann_list[J]){
            const size_t bJ_add = bJ_mo_sign.first;
            const size_t p = std::abs(bJ_mo_sign.second) - 1;
            for (auto& bbJ_mo_sign : b_cre_list[bJ_add]){
                const size_t q = std::abs(bbJ_mo_sign.second) - 1;
                if (p != q){
                    const double HIJ = space_[bbJ_mo_sign.first].slater_rules(space_[J]);
                    const size_t I = bbJ_mo_sign.first;
                    h_p[I][J] = HIJ;
                }
            }
        }

        // aaaa doubles
        for (auto& aaJ_mo_sign : aa_ann_list[J]){
            const size_t aaJ_add = std::get<0>(aaJ_mo_sign);
            const double sign_pq = std::get<1>(aaJ_mo_sign) > 0.0 ? 1.0 : -1.0;
            const size_t p = std::abs(std::get<1>(aaJ_mo_sign)) - 1;
            const size_t q = std::get<2>(aaJ_mo_sign);
            for (auto& aaaaJ_mo_sign : aa_cre_list[aaJ_add]){
                const size_t r = std::abs(std::get<1>(aaaaJ_mo_sign)) - 1;
                const size_t s = std::get<2>(aaaaJ_mo_sign);
                if ((p != r) and (q != s) and (p != s) and (q != r)){
                    const size_t aaaaJ_add = std::get<0>(aaaaJ_mo_sign);
                    const double sign_rs = std::get<1>(aaaaJ_mo_sign) > 0.0 ? 1.0 : -1.0;
                    const size_t I = aaaaJ_add;
                    const double HIJ1 = sign_pq * sign_rs * STLBitsetDeterminant::fci_ints_->tei_aa(p,q,r,s);
                    h_p[I][J] = HIJ1;
                }
            }
        }

        // aabb singles
        for (auto& abJ_mo_sign : ab_ann_list[J]){
            const size_t abJ_add = std::get<0>(abJ_mo_sign);
            const double sign_pq = std::get<1>(abJ_mo_sign) > 0.0 ? 1.0 : -1.0;
            const size_t p = std::abs(std::get<1>(abJ_mo_sign)) - 1;
            const size_t q = std::get<2>(abJ_mo_sign);
            for (auto& ababJ_mo_sign : ab_cre_list[abJ_add]){
                const size_t r = std::abs(std::get<1>(ababJ_mo_sign)) - 1;
                const size_t s = std::get<2>(ababJ_mo_sign);
                if ((p != r) and (q != s)){
                    const size_t ababJ_add = std::get<0>(ababJ_mo_sign);
                    const double sign_rs = std::get<1>(ababJ_mo_sign) > 0.0 ? 1.0 : -1.0;
                    const size_t I = ababJ_add;
                    const double HIJ1 = sign_pq * sign_rs * STLBitsetDeterminant::fci_ints_->tei_ab(p,q,r,s);
                    h_p[I][J] = HIJ1;
                }
            }
        }

        // bbbb singles
        for (auto& bbJ_mo_sign : bb_ann_list[J]){
            const size_t bbJ_add = std::get<0>(bbJ_mo_sign);
            const double sign_pq = std::get<1>(bbJ_mo_sign) > 0.0 ? 1.0 : -1.0;
            const size_t p = std::abs(std::get<1>(bbJ_mo_sign)) - 1;
            const size_t q = std::get<2>(bbJ_mo_sign);
            for (auto& bbbbJ_mo_sign : bb_cre_list[bbJ_add]){
                const size_t r = std::abs(std::get<1>(bbbbJ_mo_sign)) - 1;
                const size_t s = std::get<2>(bbbbJ_mo_sign);
                if ((p != r) and (q != s) and (p != s) and (q != r)){
                    const size_t bbbbJ_add = std::get<0>(bbbbJ_mo_sign);
                    const double sign_rs = std::get<1>(bbbbJ_mo_sign) > 0.0 ? 1.0 : -1.0;
                    const size_t I = bbbbJ_add;
                    const double HIJ1 = sign_pq * sign_rs * STLBitsetDeterminant::fci_ints_->tei_bb(p,q,r,s);
                    h_p[I][J] = HIJ1;
                }
            }
        }
    }
}

std::vector<std::pair<std::vector<int>,std::vector<double>>> SigmaVectorList::get_sparse_hamiltonian()
{
    boost::timer t_h_build2;

    size_t num_nonzero = 0;

    std::vector<std::pair<std::vector<int>,std::vector<double>>> H_sparse(size_);

    for (size_t J = 0; J < size_; ++J){
        std::vector<double> H_row;
        std::vector<int> index_row;

        // the diagonal term
        H_row.push_back(diag_[J]);
        index_row.push_back(J);

        // aa singles
        for (auto& aJ_mo_sign : a_ann_list[J]){
            const size_t aJ_add = aJ_mo_sign.first;
            const size_t p = std::abs(aJ_mo_sign.second) - 1;
            for (auto& aaJ_mo_sign : a_cre_list[aJ_add]){
                const size_t q = std::abs(aaJ_mo_sign.second) - 1;
                if (p != q){
                    const double HIJ = space_[aaJ_mo_sign.first].slater_rules(space_[J]);
                    const size_t I = aaJ_mo_sign.first;
                    if (std::fabs(HIJ) >= 1.0e-12){
                        H_row.push_back(HIJ);
                        index_row.push_back(I);
                        num_nonzero += 1;
                    }
                }
            }
        }

        // bb singles
        for (auto& bJ_mo_sign : b_ann_list[J]){
            const size_t bJ_add = bJ_mo_sign.first;
            const size_t p = std::abs(bJ_mo_sign.second) - 1;
            for (auto& bbJ_mo_sign : b_cre_list[bJ_add]){
                const size_t q = std::abs(bbJ_mo_sign.second) - 1;
                if (p != q){
                    const double HIJ = space_[bbJ_mo_sign.first].slater_rules(space_[J]);
                    const size_t I = bbJ_mo_sign.first;
                    if (std::fabs(HIJ) >= 1.0e-12){
                        H_row.push_back(HIJ);
                        index_row.push_back(I);
                        num_nonzero += 1;
                    }
                }
            }
        }

        // aaaa doubles
        for (auto& aaJ_mo_sign : aa_ann_list[J]){
            const size_t aaJ_add = std::get<0>(aaJ_mo_sign);
            const double sign_pq = std::get<1>(aaJ_mo_sign) > 0.0 ? 1.0 : -1.0;
            const size_t p = std::abs(std::get<1>(aaJ_mo_sign)) - 1;
            const size_t q = std::get<2>(aaJ_mo_sign);
            for (auto& aaaaJ_mo_sign : aa_cre_list[aaJ_add]){
                const size_t r = std::abs(std::get<1>(aaaaJ_mo_sign)) - 1;
                const size_t s = std::get<2>(aaaaJ_mo_sign);
                if ((p != r) and (q != s) and (p != s) and (q != r)){
                    const size_t aaaaJ_add = std::get<0>(aaaaJ_mo_sign);
                    const double sign_rs = std::get<1>(aaaaJ_mo_sign) > 0.0 ? 1.0 : -1.0;
                    const size_t I = aaaaJ_add;
                    const double HIJ = sign_pq * sign_rs * STLBitsetDeterminant::fci_ints_->tei_aa(p,q,r,s);
                    if (std::fabs(HIJ) >= 1.0e-12){
                        H_row.push_back(HIJ);
                        index_row.push_back(I);
                        num_nonzero += 1;
                    }
                }
            }
        }

        // aabb singles
        for (auto& abJ_mo_sign : ab_ann_list[J]){
            const size_t abJ_add = std::get<0>(abJ_mo_sign);
            const double sign_pq = std::get<1>(abJ_mo_sign) > 0.0 ? 1.0 : -1.0;
            const size_t p = std::abs(std::get<1>(abJ_mo_sign)) - 1;
            const size_t q = std::get<2>(abJ_mo_sign);
            for (auto& ababJ_mo_sign : ab_cre_list[abJ_add]){
                const size_t r = std::abs(std::get<1>(ababJ_mo_sign)) - 1;
                const size_t s = std::get<2>(ababJ_mo_sign);
                if ((p != r) and (q != s)){
                    const size_t ababJ_add = std::get<0>(ababJ_mo_sign);
                    const double sign_rs = std::get<1>(ababJ_mo_sign) > 0.0 ? 1.0 : -1.0;
                    const size_t I = ababJ_add;
                    const double HIJ = sign_pq * sign_rs * STLBitsetDeterminant::fci_ints_->tei_ab(p,q,r,s);
                    if (std::fabs(HIJ) >= 1.0e-12){
                        H_row.push_back(HIJ);
                        index_row.push_back(I);
                        num_nonzero += 1;
                    }
                }
            }
        }

        // bbbb singles
        for (auto& bbJ_mo_sign : bb_ann_list[J]){
            const size_t bbJ_add = std::get<0>(bbJ_mo_sign);
            const double sign_pq = std::get<1>(bbJ_mo_sign) > 0.0 ? 1.0 : -1.0;
            const size_t p = std::abs(std::get<1>(bbJ_mo_sign)) - 1;
            const size_t q = std::get<2>(bbJ_mo_sign);
            for (auto& bbbbJ_mo_sign : bb_cre_list[bbJ_add]){
                const size_t r = std::abs(std::get<1>(bbbbJ_mo_sign)) - 1;
                const size_t s = std::get<2>(bbbbJ_mo_sign);
                if ((p != r) and (q != s) and (p != s) and (q != r)){
                    const size_t bbbbJ_add = std::get<0>(bbbbJ_mo_sign);
                    const double sign_rs = std::get<1>(bbbbJ_mo_sign) > 0.0 ? 1.0 : -1.0;
                    const size_t I = bbbbJ_add;
                    const double HIJ = sign_pq * sign_rs * STLBitsetDeterminant::fci_ints_->tei_bb(p,q,r,s);
                    if (std::fabs(HIJ) >= 1.0e-12){
                        H_row.push_back(HIJ);
                        index_row.push_back(I);
                        num_nonzero += 1;
                    }
                }
            }
        }
        H_sparse[J] = std::make_pair(index_row,H_row);
    }
    outfile->Printf("\n  The sparse Hamiltonian matrix contains %zu nonzero elements out of %zu (%f)",num_nonzero,size_ * size_,double(num_nonzero)/double(size_ * size_));
    outfile->Printf("\n  %s: %f s","Time spent building H",t_h_build2.elapsed());
    outfile->Flush();
    return H_sparse;
}

void SigmaVectorList::get_diagonal(Vector& diag)
{
    for (size_t I = 0; I < diag_.size(); ++I){
        diag.set(I,diag_[I]);
    }
}

void SparseCISolver::set_spin_project(bool value)
{
    spin_project_ = value;
}

void SparseCISolver::set_e_convergence(double value)
{
    e_convergence_ = value;
}

void SparseCISolver::set_maxiter_davidson(int value)
{
    maxiter_davidson_ = value;
}

void SparseCISolver::diagonalize_hamiltonian(const std::vector<STLBitsetDeterminant>& space,SharedVector& evals,SharedMatrix& evecs,int nroot,int multiplicity,DiagonalizationMethod diag_method)
{
    if (space.size() <= 5){
        diagonalize_full(space,evals,evecs,nroot,multiplicity);
    }else{
        if (diag_method == Full){
            diagonalize_full(space,evals,evecs,nroot,multiplicity);
        }else if (diag_method == DavidsonLiuDense){
            diagonalize_davidson_liu_dense(space,evals,evecs,nroot,multiplicity);
        }else if (diag_method == DavidsonLiuSparse){
            diagonalize_davidson_liu_sparse(space,evals,evecs,nroot,multiplicity);
        }else if (diag_method == DavidsonLiuList){
            diagonalize_davidson_liu_list(space,evals,evecs,nroot,multiplicity);
        }else if (diag_method == DLSolver){
            diagonalize_davidson_liu_solver(space,evals,evecs,nroot,multiplicity);
        }
    }
}

void SparseCISolver::diagonalize_full(const std::vector<STLBitsetDeterminant>& space,SharedVector& evals,SharedMatrix& evecs,int nroot,int multiplicity)
{
    // Find all the eigenvalues and eigenvectors of the Hamiltonian
    SharedMatrix H = build_full_hamiltonian(space);

    size_t dim_space = space.size();
    evecs.reset(new Matrix("U",dim_space,dim_space));
    evals.reset(new Vector("e",dim_space));

    // Diagonalize H
    boost::timer t_diag;
    H->diagonalize(evecs,evals);
    outfile->Printf("\n  %s: %f s","Time spent diagonalizing H using Full",t_diag.elapsed());
}

void SparseCISolver::diagonalize_davidson_liu_dense(const std::vector<STLBitsetDeterminant>& space,SharedVector& evals,SharedMatrix& evecs,int nroot,int multiplicity)
{
    outfile->Printf("\n  Using <diagonalize_davidson_liu_dense>");
    outfile->Flush();
    // Find all the eigenvalues and eigenvectors of the Hamiltonian
    SharedMatrix H = build_full_hamiltonian(space);

    size_t dim_space = space.size();
    evecs.reset(new Matrix("U",dim_space,nroot));
    evals.reset(new Vector("e",nroot));

    // Diagonalize H
    SigmaVectorFull svf (H);
    SigmaVector* sigma_vector = &svf;
    auto guess = initial_guess(space,nroot,multiplicity);
    davidson_liu_guess(guess,sigma_vector,evals,evecs,nroot,multiplicity);
}

void SparseCISolver::diagonalize_davidson_liu_sparse(const std::vector<STLBitsetDeterminant>& space, SharedVector& evals, SharedMatrix& evecs, int nroot, int multiplicity)
{
    outfile->Printf("\n\n  Davidson-liu sparse algorithm");
    outfile->Flush();
    // Find all the eigenvalues and eigenvectors of the Hamiltonian
    std::vector<std::pair<std::vector<int>,std::vector<double>>> H = parallel_ ? build_sparse_hamiltonian_parallel(space) : build_sparse_hamiltonian(space);

    size_t dim_space = space.size();
    evecs.reset(new Matrix("U",dim_space,nroot));
    evals.reset(new Vector("e",nroot));

    // Diagonalize H
    SigmaVectorSparse svs (H);
    SigmaVector* sigma_vector = &svs;
    auto guess = initial_guess(space,nroot,multiplicity);
    davidson_liu_guess(guess,sigma_vector,evals,evecs,nroot,multiplicity);
}

void SparseCISolver::diagonalize_davidson_liu_list(const std::vector<STLBitsetDeterminant>& space, SharedVector& evals, SharedMatrix& evecs, int nroot, int multiplicity)
{
    outfile->Printf("\n\n  Davidson-liu list algorithm");
    outfile->Flush();

    size_t dim_space = space.size();
    evecs.reset(new Matrix("U",dim_space,nroot));
    evals.reset(new Vector("e",nroot));

    // Diagonalize H
    SigmaVectorList svl (space);
    SigmaVector* sigma_vector = &svl;
    auto guess = initial_guess(space,nroot,multiplicity);
    davidson_liu_guess(guess,sigma_vector,evals,evecs,nroot,multiplicity);
}

void SparseCISolver::diagonalize_davidson_liu_solver(const std::vector<STLBitsetDeterminant>& space, SharedVector& evals, SharedMatrix& evecs, int nroot, int multiplicity)
{
    outfile->Printf("\n\n  Davidson-liu solver algorithm");
    outfile->Flush();

    size_t dim_space = space.size();
    evecs.reset(new Matrix("U",dim_space,nroot));
    evals.reset(new Vector("e",nroot));

    // Diagonalize H
    SigmaVectorList svl (space);
    SigmaVector* sigma_vector = &svl;
    davidson_liu_solver(space,sigma_vector,evals,evecs,nroot,multiplicity);
}

SharedMatrix SparseCISolver::build_full_hamiltonian(const std::vector<STLBitsetDeterminant> &space)
{
    // Build the H matrix
    size_t dim_space = space.size();
    SharedMatrix H(new Matrix("H",dim_space,dim_space));
    //If you are using DiskDF, Kevin found that openmp does not like this! 
    int threads = 0;
    if(STLBitsetDeterminant::fci_ints_->get_integral_type()==DiskDF)
    {
       threads = 1;
    }
    else
    {
       threads = omp_get_max_threads();
    }
    #pragma omp parallel for schedule(dynamic) num_threads(threads)
    for (size_t I = 0; I < dim_space; ++I){
        const STLBitsetDeterminant& detI = space[I];
        for (size_t J = I; J < dim_space; ++J){
            const STLBitsetDeterminant& detJ = space[J];
            double HIJ = detI.slater_rules(detJ);
            H->set(I,J,HIJ);
            H->set(J,I,HIJ);
        }
    }
    return H;
}

std::vector<std::pair<std::vector<int>,std::vector<double>>> SparseCISolver::build_sparse_hamiltonian(const std::vector<STLBitsetDeterminant> &space)
{
    boost::timer t_h_build2;
    std::vector<std::pair<std::vector<int>,std::vector<double>>> H_sparse;
    size_t dim_space = space.size();

    size_t num_nonzero = 0;
    // Form the Hamiltonian matrix
    for (size_t I = 0; I < dim_space; ++I){
        std::vector<double> H_row;
        std::vector<int> index_row;
        const STLBitsetDeterminant& detI = space[I];
        double HII = detI.slater_rules(detI);
        H_row.push_back(HII);
        index_row.push_back(I);
        for (size_t J = 0; J < dim_space; ++J){
            if (I != J){
                const STLBitsetDeterminant& detJ = space[J];
                double HIJ = detI.slater_rules(detJ);
                if (std::fabs(HIJ) >= 1.0e-12){
                    H_row.push_back(HIJ);
                    index_row.push_back(J);
                    num_nonzero += 1;
                }
            }
        }
        H_sparse.push_back(make_pair(index_row,H_row));
    }
    outfile->Printf("\n  The sparse Hamiltonian matrix contains %zu nonzero elements out of %zu (%f)",num_nonzero,dim_space * dim_space,double(num_nonzero)/double(dim_space * dim_space));
    outfile->Printf("\n  %s: %f s","Time spent building H",t_h_build2.elapsed());
    outfile->Flush();
    return H_sparse;
}


std::vector<std::pair<std::vector<int>,std::vector<double>>> SparseCISolver::build_sparse_hamiltonian_parallel(const std::vector<STLBitsetDeterminant> &space)
{
    boost::timer t_h_build2;
    // Allocate as many elements as we need
    size_t dim_space = space.size();
    std::vector<std::pair<std::vector<int>,std::vector<double>>> H_sparse(dim_space);

    size_t num_nonzero = 0;

    outfile->Printf("\n  Building H using OpenMP-take2");
    outfile->Flush();

    // Form the Hamiltonian matrix

#pragma omp parallel for schedule(dynamic)
    for (size_t I = 0; I < dim_space; ++I){
        std::vector<double> H_row;
        std::vector<int> index_row;
        const STLBitsetDeterminant& detI = space[I];
        double HII = detI.slater_rules(detI);
        H_row.push_back(HII);
        index_row.push_back(I);
        for (size_t J = 0; J < dim_space; ++J){
            if (I != J){
                const STLBitsetDeterminant detJ = space[J];
                double HIJ = detI.slater_rules(detJ);
                if (std::fabs(HIJ) >= 1.0e-12){
                    H_row.push_back(HIJ);
                    index_row.push_back(J);
                }
            }
        }

#pragma omp critical(save_h_row)
        {
            H_sparse[I] = make_pair(index_row,H_row);
            num_nonzero += index_row.size();
        }
    }
    outfile->Printf("\n  The sparse Hamiltonian matrix contains %zu nonzero elements out of %zu (%f)",num_nonzero,dim_space * dim_space,double(num_nonzero)/double(dim_space * dim_space));
    outfile->Printf("\n  %s: %f s","Time spent building H (openmp)",t_h_build2.elapsed());
    outfile->Flush();
    return H_sparse;
}

std::vector<std::pair<double,std::vector<std::pair<size_t,double>>>> SparseCISolver::initial_guess(const std::vector<STLBitsetDeterminant>& space, int nroot, int multiplicity)
{
    size_t ndets = space.size();
    size_t nguess = std::min(static_cast<size_t>(nroot) * 20,ndets);
    std::vector<std::pair<double,std::vector<std::pair<size_t,double>>>> guess(nguess);

    // Find the ntrial lowest diagonals
    std::vector<std::pair<STLBitsetDeterminant,size_t>> guess_dets_pos;
    std::vector<std::pair<double,size_t>> smallest(ndets);
    for(size_t I = 0; I < ndets; ++I){
        smallest[I] = std::make_pair(space[I].energy(),I);
    }
    std::sort(smallest.begin(),smallest.end());

    std::vector<STLBitsetDeterminant> guess_det;
    for(size_t i = 0; i < nguess; i++) {
        size_t I = smallest[i].second;
        guess_dets_pos.push_back(std::make_pair(space[I],I));  // store a det and its position
        guess_det.push_back(space[I]);
    }

    if (spin_project_){
        STLBitsetDeterminant::enforce_spin_completeness(guess_det);
        if (guess_det.size() > nguess){
            size_t nnew_dets = guess_det.size() - nguess;
            outfile->Printf("\n  Initial guess space is incomplete.\n  Trying to add %d determinant(s).",nnew_dets);
            int nfound = 0;
            for (size_t i = 0; i < nnew_dets; ++i){
                for (size_t j = nguess; j < ndets; ++j){
                    size_t J = smallest[j].second;
                    if (space[J] == guess_det[nguess + i]){
                        guess_dets_pos.push_back(std::make_pair(space[J],J));  // store a det and its position
                        nfound++;
                        break;
                    }
                }
            }
            outfile->Printf("  %d determinant(s) added.",nfound);
        }
        nguess = guess_dets_pos.size();
    }

    // Form the S^2 operator matrix and diagonalize it
    Matrix S2("S^2",nguess,nguess);
    for(size_t I = 0; I < nguess; I++) {
        for(size_t J = I; J < nguess; J++) {
            const STLBitsetDeterminant& detI = guess_dets_pos[I].first;
            const STLBitsetDeterminant& detJ = guess_dets_pos[J].first;
            double S2IJ = detI.spin2(detJ);
            S2.set(I,J,S2IJ);
            S2.set(J,I,S2IJ);
        }
    }
    Matrix S2evecs("S^2",nguess,nguess);
    Vector S2evals("S^2",nguess);
    S2.diagonalize(S2evecs,S2evals);

    // Form the Hamiltonian
    Matrix H("H",nguess,nguess);
    for(size_t I = 0; I < nguess; I++) {
        for(size_t J = I; J < nguess; J++) {
            const STLBitsetDeterminant& detI = guess_dets_pos[I].first;
            const STLBitsetDeterminant& detJ = guess_dets_pos[J].first;
            double HIJ = detI.slater_rules(detJ);
            H.set(I,J,HIJ);
            H.set(J,I,HIJ);
        }
    }

    // Project H onto the spin-adapted subspace
    H.transform(S2evecs);

    // Find groups of solutions with same spin
    double Stollerance = 1.0e-6;
    std::map<int,std::vector<int>> mult_list;
    for (size_t i = 0; i < nguess; ++i){
        double mult = std::sqrt(1.0 + 4.0 * S2evals.get(i)); // 2S + 1 = Sqrt(1 + 4 S (S + 1))
        int mult_int = std::round(mult);
        double error = mult - static_cast<double>(mult_int);
        if (std::fabs(error) < Stollerance){
            mult_list[mult_int].push_back(i);
        }else{
            outfile->Printf("\n  Found a guess vector with spin not close to integer value (%f)",mult);
        }
    }
    if (mult_list[multiplicity].size() < static_cast<size_t>(nroot)){
        size_t nfound = mult_list[multiplicity].size();
        outfile->Printf("\n  Error: %d guess vectors with 2S+1 = %d but only %d were found!",nguess,multiplicity,nfound);
        exit(1);
    }

    std::vector<int> mult_vals;
    for (auto kv : mult_list){
        mult_vals.push_back(kv.first);
    }
    std::sort(mult_vals.begin(),mult_vals.end());

    for (int m : mult_vals){
        std::vector<int>& mult_list_s = mult_list[m];
        int nspin_states = mult_list_s.size();
        outfile->Printf("\n  Initial guess found %d solutions with 2S+1 = %d %c",nspin_states,m,m == multiplicity ? '*' : ' ');
        // Extract the spin manifold
        Matrix HS2("HS2",nspin_states,nspin_states);
        Vector HS2evals("HS2",nspin_states);
        Matrix HS2evecs("HS2",nspin_states,nspin_states);
        for(int I = 0; I < nspin_states; I++) {
            for(int J = 0; J < nspin_states; J++) {
                HS2.set(I,J,H.get(mult_list_s[I],mult_list_s[J]));
            }
        }
        HS2.diagonalize(HS2evecs,HS2evals);

        // Project the spin-adapted solution onto the full manifold
        for (int r = 0; r < nspin_states; ++r){
            std::vector<std::pair<size_t,double>> det_C;
            for (size_t I = 0; I < nguess; I++) {
                double CIr = 0.0;
                for (int J = 0; J < nspin_states; ++J){
                    CIr += S2evecs.get(I,mult_list_s[J]) * HS2evecs(J,r);
                }
                det_C.push_back(std::make_pair(guess_dets_pos[I].second,CIr));
            }
            guess.push_back(std::make_pair(m,det_C));
        }
    }

    return guess;

//    // Check the spin
//    for (int r = 0; r < nguess; ++r){
//        double s2 = 0.0;
//        double e = 0.0;
//        for (size_t i = 0; i < nguess; i++) {
//            for (size_t j = 0; j < nguess; j++) {
//                size_t I = guess_dets_pos[i].second;
//                size_t J = guess_dets_pos[j].second;
//                double CI = evecs->get(I,r);
//                double CJ = evecs->get(J,r);
//                s2 += space[I].spin2(space[J]) * CI * CJ;
//                e += space[I].slater_rules(space[J]) * CI * CJ;
//            }
//        }
//        outfile->Printf("\n  Guess Root %d: <E> = %f, <S^2> = %f",r,e,s2);
//    }
 }

bool SparseCISolver::davidson_liu_solver(const std::vector<STLBitsetDeterminant>& space,
                                         SigmaVector* sigma_vector,
                                         SharedVector Eigenvalues,
                                         SharedMatrix Eigenvectors,
                                         int nroot,
                                         int multiplicity)
{
    print_details_ = true;
    size_t fci_size = sigma_vector->size();
    DavidsonLiuSolver dls(fci_size,nroot);
    dls.set_e_convergence(e_convergence_);
    dls.set_print_level(0);

    // allocate vectors
    SharedVector b(new Vector("b",fci_size));
    SharedVector sigma(new Vector("sigma",fci_size));

    // get and pass diagonal
    sigma_vector->get_diagonal(*sigma);
    dls.startup(sigma);

    size_t guess_size = dls.collapse_size();
    outfile->Printf("\n  number of guess vectors: %d",guess_size);

    auto guess = initial_guess(space,nroot,multiplicity);

    std::vector<int> guess_list;
    for (size_t g = 0; g < guess.size(); ++g){
        if (guess[g].first == multiplicity) guess_list.push_back(g);
    }

    // number of guess to be used
    size_t nguess = std::min(guess_list.size(),guess_size);

    if (nguess == 0){
        throw PSIEXCEPTION("\n\n  Found zero FCI guesses with the requested multiplicity.\n\n");
    }

    for (size_t n = 0; n < nguess; ++n){
        b->zero();
        for (auto& guess_vec_info : guess[guess_list[n]].second){
            b->set(guess_vec_info.first,guess_vec_info.second);
        }
        outfile->Printf("\n  Adding guess %d (multiplicity = %f)",n,guess[guess_list[n]].first);
        dls.add_guess(b);
    }

    // Prepare a list of bad roots to project out and pass them to the solver
    std::vector<std::vector<std::pair<size_t,double>>> bad_roots;
    for (auto& g : guess){
        if (g.first != multiplicity) bad_roots.push_back(g.second);
    }
    dls.set_project_out(bad_roots);

    SolverStatus converged = SolverStatus::NotConverged;

    if(print_details_){
        outfile->Printf("\n\n  ==> Diagonalizing Hamiltonian <==\n");
        outfile->Printf("\n  ----------------------------------------");
        outfile->Printf("\n    Iter.      Avg. Energy       Delta_E");
        outfile->Printf("\n  ----------------------------------------");
    }

    double old_avg_energy = 0.0;
    int real_cycle = 1;
    for (int cycle = 0; cycle < maxiter_davidson_; ++cycle){
        bool add_sigma = true;
        do{
            dls.get_b(b);
            sigma_vector->compute_sigma(sigma,b);
            add_sigma = dls.add_sigma(sigma);
        } while (add_sigma);

        converged = dls.update();

        if (converged != SolverStatus::Collapse){
            double avg_energy = 0.0;
            for (int r = 0; r < nroot; ++r) avg_energy += dls.eigenvalues()->get(r);
            avg_energy /= static_cast<double>(nroot);
            if (print_details_){
                outfile->Printf("\n    %3d  %20.12f  %+.3e",real_cycle,avg_energy,avg_energy - old_avg_energy);
            }
            old_avg_energy = avg_energy;
            real_cycle++;
        }

        if (converged == SolverStatus::Converged) break;
    }

    if (print_details_){
        outfile->Printf("\n  ----------------------------------------");
        if (converged == SolverStatus::Converged){
            outfile->Printf("\n  The Davidson-Liu algorithm converged in %d iterations.", real_cycle);
        }
    }

    if (converged == SolverStatus::NotConverged){
        outfile->Printf("\n  FCI did not converge!");
        exit(1);
    }

//    dls.get_results();
    SharedVector evals = dls.eigenvalues();
    SharedMatrix evecs = dls.eigenvectors();
    for (int r = 0; r < nroot; ++r){
        Eigenvalues->set(r,evals->get(r));
        for (size_t I = 0; I < fci_size; ++I){
            Eigenvectors->set(I,r,evecs->get(r,I));
        }
    }
    return true;
}

bool SparseCISolver::davidson_liu_guess(std::vector<std::pair<double,std::vector<std::pair<size_t,double>>>> guess,
                                        SigmaVector* sigma_vector,
                                        SharedVector Eigenvalues,
                                        SharedMatrix Eigenvectors,
                                        int nroot,
                                        int multiplicity)
{
    // Start a timer
    boost::timer t_davidson;

    int maxiter = 100;
    bool print = print_details_;

    // Use unit vectors as initial guesses
    size_t N = sigma_vector->size();
    // Number of roots
    int M = nroot;

    if (N == 0) throw std::runtime_error("SparseCISolver::davidson_liu called with space of dimension zero.");
    size_t collapse_size = ncollapse_per_root_ * M;
    size_t subspace_size = nsubspace_per_root_ * M;

    // Maximum number of vectors stored
    int maxdim = subspace_size;

    double e_convergence = 1.0e-12;

    if (print_details_){
        outfile->Printf("\n  Size of the Hamiltonian: %d x %d",N,N);
    }

    // current set of guess vectors stored by row
    Matrix b("b",maxdim,N);
    b.zero();

    // guess vectors formed from old vectors, stored by row
    Matrix bnew("bnew",maxdim,N);

    // residual eigenvectors, stored by row
    Matrix f("f",maxdim,N);

    // sigma vectors, stored by column
    Matrix sigma("sigma",N, maxdim);

    // Davidson mini-Hamitonian
    Matrix G("G",maxdim, maxdim);
    // A metric matrix
    Matrix S("S",maxdim, maxdim);
    // Eigenvectors of the Davidson mini-Hamitonian
    Matrix alpha("alpha",maxdim, maxdim);
    // Eigenvalues of the Davidson mini-Hamitonian
    Vector lambda("lambda",maxdim);
    // Old eigenvalues of the Davidson mini-Hamitonian
    Vector lambda_old("lambda",maxdim);
    // Diagonal elements of the Hamiltonian
    Vector Hdiag("Hdiag",N);

    sigma_vector->get_diagonal(Hdiag);

    size_t initial_size = M;

    // Find the initial_size lowest diagonals

    std::vector<int> guess_mult;
    for (auto& g : guess){
        guess_mult.push_back(g.first);
    }
    auto it = std::find(guess_mult.begin(),guess_mult.end(),multiplicity);
    size_t guess_first = std::distance(guess_mult.begin(), it);
    for(int i = 0; i < M; i++) { // loop over roots
        for (auto& guess_vec_info : guess[i + guess_first].second){
            b.set(i,guess_vec_info.first,guess_vec_info.second);
        }
    }

    int L = initial_size;
    int iter = 0;
    int converged = 0;
    while((converged < M) and (iter < maxiter)) {
        double* lambda_p = lambda.pointer();
        double* Adiag_p = Hdiag.pointer();
        double** b_p = b.pointer();
        double** f_p = f.pointer();
        double** alpha_p = alpha.pointer();
        double** sigma_p = sigma.pointer();

        bool skip_check = false;

        // Step #2: Build and Diagonalize the Subspace Hamiltonian
        sigma_vector->compute_sigma(sigma,b,L);

        G.zero();
        G.gemm(false,false,1.0,b,sigma,0.0);

        // diagonalize mini-matrix
        G.diagonalize(alpha,lambda);

        // Overlap matrix
        S.gemm(false,true,1.0,b,b,0.0);

        // Check for orthogonality
        bool printed_S = false;
        for (int i = 0; i < L; ++i){
            double diag = S.get(i,i);
            double zero = false;
            double one = false;
            if (std::fabs(diag - 1.0) < 1.e-6){
                one = true;
            }
            if (std::fabs(diag) < 1.e-6){
                zero = true;
            }
            if ((not zero) and (not one)){
                if (not printed_S) {
                    outfile->Printf("\n  WARNING: Vector %d is not normalized or zero");
                    printed_S = true;
                }
            }
            double offdiag = 0.0;
            for (int j = i + 1; j < L; ++j){
                offdiag += std::fabs(S.get(i,j));
            }
            if (offdiag > 1.0e-6){
                if (not printed_S) {
                    outfile->Printf("\n  WARNING: The vectors are not orthogonal");
                    printed_S = true;
                }

            }
        }

        // If L is close to maxdim, collapse to one guess per root */
        if(maxdim - L < M) {
            if(print) {
                outfile->Printf("Subspace too large: maxdim = %d, L = %d\n", maxdim, L);
                outfile->Printf("Collapsing eigenvectors.\n");
            }
            bnew.zero();
            double** bnew_p = bnew.pointer();
            for(size_t i = 0; i < collapse_size; i++) {
                for(int j = 0; j < L; j++) {
                    for(size_t k = 0; k < N; k++) {
                        bnew_p[i][k] += alpha_p[j][i] * b_p[j][k];
                    }
                }
            }

            // normalize new vectors
            for(size_t i = 0; i < collapse_size; i++){
                double norm = 0.0;
                for(size_t k = 0; k < N; k++){
                    norm += bnew_p[i][k] * bnew_p[i][k];
                }
                norm = std::sqrt(norm);
                for(size_t k = 0; k < N; k++){
                    bnew_p[i][k] = bnew_p[i][k] / norm;
                }
            }

            // Copy them into place
            b.zero();
            L = 0;
            for(size_t k = 0; k < collapse_size; k++){
                if(schmidt_add(b_p,k, N, bnew_p[k])) {
                    L++;  // <- Increase L if we add one more basis vector
                }
            }

            skip_check = true;

            // Rebuild and Diagonalize the Subspace Hamiltonian
            sigma_vector->compute_sigma(sigma,b,L);

            G.zero();
            G.gemm(false,false,1.0,b,sigma,0.0);

            // diagonalize mini-matrix
            G.diagonalize(alpha,lambda);
        }

        // Step #3: Build the Correction Vectors
        // form preconditioned residue vectors
        f.zero();
        for(int k = 0; k < M; k++){  // loop over roots
            for(size_t I = 0; I < N; I++) {  // loop over elements
                for(int i = 0; i < L; i++) {
                    f_p[k][I] += alpha_p[i][k] * (sigma_p[I][i] - lambda_p[k] * b_p[i][I]);
                }
                double denom = lambda_p[k] - Adiag_p[I];
                if(fabs(denom) > 1e-6){
                    f_p[k][I] /= denom;
                }
                else{
                    f_p[k][I] = 0.0;
                }
            }
        }

        // Step #4: Orthonormalize the Correction Vectors
        /* normalize each residual */
        for(int k = 0; k < M; k++) {
            double norm = 0.0;
            for(size_t I = 0; I < N; I++) {
                norm += f_p[k][I] * f_p[k][I];
            }
            norm = std::sqrt(norm);
            for(size_t I = 0; I < N; I++) {
                f_p[k][I] /= norm;
            }
        }

        if(spin_project_){
            for(int i = 0; i < M; i++){
                int n = 0;
                for (auto& g : guess){
                    int g_mult = g.first;
                    double overlap = 0.0;
                    for (auto& guess_vec_info : g.second){
                        size_t I = guess_vec_info.first;
                        double CI = guess_vec_info.second;
                        overlap += f_p[i][I] * CI;
                    }
                    n++;
                    if (g_mult != multiplicity){
                        for (auto& guess_vec_info : g.second){
                            size_t I = guess_vec_info.first;
                            double CI = guess_vec_info.second;
                            f_p[i][I] -= overlap * CI;
                        }
                    }
                }
            }
        }

        // schmidt orthogonalize the f[k] against the set of b[i] and add new vectors
        for(int k = 0; k < M; k++){
            if (L < subspace_size){
                if(schmidt_add(b_p, L, N, f_p[k])) {
                    L++;  // <- Increase L if we add one more basis vector
                }
            }
        }

        // check convergence on all roots
        if(!skip_check) {
            converged = 0;
            if(print) {
                outfile->Printf("Root      Eigenvalue       Delta  Converged?\n");
                outfile->Printf("---- -------------------- ------- ----------\n");
            }
            for(int k = 0; k < M; k++) {
                double diff = std::fabs(lambda.get(k) - lambda_old.get(k));
                bool this_converged = false;
                if(diff < e_convergence) {
                    this_converged = true;
                    converged++;
                }
                lambda_old.set(k,lambda.get(k));
                if(print) {
                    outfile->Printf("%3d  %20.14f %4.3e    %1s\n", k, lambda.get(k), diff,
                                    this_converged ? "Y" : "N");
                }
            }
        }

        outfile->Flush();

        iter++;
    }

    /* generate final eigenvalues and eigenvectors */
    //if(converged == M) {
    double** alpha_p = alpha.pointer();
    double** b_p = b.pointer();
    double* eps = Eigenvalues->pointer();
    double** v = Eigenvectors->pointer();

    for(int i = 0; i < M; i++) {
        eps[i] = lambda.get(i);
        for(size_t I = 0; I < N; I++){
            v[I][i] = 0.0;
        }
        for(int j = 0; j < L; j++) {
            for(size_t I=0; I < N; I++) {
                v[I][i] += alpha_p[j][i] * b_p[j][I];
            }
        }
        // Normalize v
        double norm = 0.0;
        for(size_t I = 0; I < N; I++) {
            norm += v[I][i] * v[I][i];
        }
        norm = std::sqrt(norm);
        for(size_t I = 0; I < N; I++) {
            v[I][i] /= norm;
        }
    }
    outfile->Printf("\n  The Davidson-Liu algorithm converged in %d iterations.", iter);
    outfile->Printf("\n  %s: %f s","Time spent diagonalizing H",t_davidson.elapsed());
    return true;
}


bool SparseCISolver::davidson_liu(SigmaVector* sigma_vector, SharedVector Eigenvalues, SharedMatrix Eigenvectors, int nroot_s)
{
    // Start a timer
    boost::timer t_davidson;

    int maxiter = 100;
    bool print = print_details_;

    // Use unit vectors as initial guesses
    size_t N = sigma_vector->size();
    // Number of roots
    int M = nroot_s;

    if (N == 0) throw std::runtime_error("SparseCISolver::davidson_liu called with space of dimension zero.");
    size_t collapse_size = 2 * M;
    size_t subspace_size = 4 * M;

    // Maximum number of vectors stored
    int maxdim = subspace_size;

    double e_convergence = 1.0e-12;

    if (print_details_){
        outfile->Printf("\n  Size of the Hamiltonian: %d x %d",N,N);
    }

    // current set of guess vectors stored by row
    Matrix b("b",maxdim,N);
    b.zero();

    // guess vectors formed from old vectors, stored by row
    Matrix bnew("bnew",maxdim,N);

    // residual eigenvectors, stored by row
    Matrix f("f",maxdim,N);

    // sigma vectors, stored by column
    Matrix sigma("sigma",N, maxdim);

    // Davidson mini-Hamitonian
    Matrix G("G",maxdim, maxdim);
    // A metric matrix
    Matrix S("S",maxdim, maxdim);
    // Eigenvectors of the Davidson mini-Hamitonian
    Matrix alpha("alpha",maxdim, maxdim);
    // Eigenvalues of the Davidson mini-Hamitonian
    Vector lambda("lambda",maxdim);
    // Old eigenvalues of the Davidson mini-Hamitonian
    Vector lambda_old("lambda",maxdim);
    // Diagonal elements of the Hamiltonian
    Vector Hdiag("Hdiag",N);

    sigma_vector->get_diagonal(Hdiag);

    size_t initial_size = collapse_size;

    // Find the initial_size lowest diagonals
    {
        std::vector<std::pair<double,size_t>> smallest(N);
        for(size_t j = 0; j < N; ++j){
            smallest[j] = std::make_pair(Hdiag.get(j),j);
        }
        std::sort(smallest.begin(),smallest.end());
        for(int i = 0; i < M; i++) {
            b.set(i,smallest[i].second,1.0);
        }
    }

    int L = initial_size;
    int iter = 0;
    int converged = 0;
    while((converged < M) and (iter < maxiter)) {
        double* lambda_p = lambda.pointer();
        double* Adiag_p = Hdiag.pointer();
        double** b_p = b.pointer();
        double** f_p = f.pointer();
        double** alpha_p = alpha.pointer();
        double** sigma_p = sigma.pointer();

        bool skip_check = false;

        // Step #2: Build and Diagonalize the Subspace Hamiltonian
        sigma_vector->compute_sigma(sigma,b,L);

        G.zero();
        G.gemm(false,false,1.0,b,sigma,0.0);

        // diagonalize mini-matrix
        G.diagonalize(alpha,lambda);

        // Overlap matrix
        S.gemm(false,true,1.0,b,b,0.0);

        // Check for orthogonality
        bool printed_S = false;
        for (int i = 0; i < L; ++i){
            double diag = S.get(i,i);
            double zero = false;
            double one = false;
            if (std::fabs(diag - 1.0) < 1.e-6){
                one = true;
            }
            if (std::fabs(diag) < 1.e-6){
                zero = true;
            }
            if ((not zero) and (not one)){
                if (not printed_S) {
                    outfile->Printf("\n  WARNING: Vector %d is not normalized or zero");
                    printed_S = true;
                }
            }
            double offdiag = 0.0;
            for (int j = i + 1; j < L; ++j){
                offdiag += std::fabs(S.get(i,j));
            }
            if (offdiag > 1.0e-6){
                if (not printed_S) {
                    outfile->Printf("\n  WARNING: The vectors are not orthogonal");
                    printed_S = true;
                }

            }
        }

        // If L is close to maxdim, collapse to one guess per root */
        if(maxdim - L < M) {
            if(print) {
                outfile->Printf("Subspace too large: maxdim = %d, L = %d\n", maxdim, L);
                outfile->Printf("Collapsing eigenvectors.\n");
            }
            bnew.zero();
            double** bnew_p = bnew.pointer();
            for(int i = 0; i < collapse_size; i++) {
                for(int j = 0; j < L; j++) {
                    for(int k = 0; k < N; k++) {
                        bnew_p[i][k] += alpha_p[j][i] * b_p[j][k];
                    }
                }
            }

            // normalize new vectors
            for(size_t i = 0; i < collapse_size; i++){
                double norm = 0.0;
                for(int k = 0; k < N; k++){
                    norm += bnew_p[i][k] * bnew_p[i][k];
                }
                norm = std::sqrt(norm);
                for(int k = 0; k < N; k++){
                    bnew_p[i][k] = bnew_p[i][k] / norm;
                }
            }

            // Copy them into place
            b.zero();
            L = 0;
            for(size_t k = 0; k < collapse_size; k++){
                if(schmidt_add(b_p,k, N, bnew_p[k])) {
                    L++;  // <- Increase L if we add one more basis vector
                }
            }

            skip_check = true;

            // Rebuild and Diagonalize the Subspace Hamiltonian
            sigma_vector->compute_sigma(sigma,b,L);

            G.zero();
            G.gemm(false,false,1.0,b,sigma,0.0);

            // diagonalize mini-matrix
            G.diagonalize(alpha,lambda);
        }

        // Step #3: Build the Correction Vectors
        // form preconditioned residue vectors
        f.zero();
        for(int k = 0; k < M; k++){  // loop over roots
            for(int I = 0; I < N; I++) {  // loop over elements
                for(int i = 0; i < L; i++) {
                    f_p[k][I] += alpha_p[i][k] * (sigma_p[I][i] - lambda_p[k] * b_p[i][I]);
                }
                double denom = lambda_p[k] - Adiag_p[I];
                if(fabs(denom) > 1e-6){
                    f_p[k][I] /= denom;
                }
                else{
                    f_p[k][I] = 0.0;
                }
            }
        }

        // Step #4: Orthonormalize the Correction Vectors
        /* normalize each residual */
        for(int k = 0; k < M; k++) {
            double norm = 0.0;
            for(int I = 0; I < N; I++) {
                norm += f_p[k][I] * f_p[k][I];
            }
            norm = std::sqrt(norm);
            for(int I = 0; I < N; I++) {
                f_p[k][I] /= norm;
            }
        }

        // schmidt orthogonalize the f[k] against the set of b[i] and add new vectors
        for(int k = 0; k < M; k++){
            if (L < subspace_size){
                if(schmidt_add(b_p, L, N, f_p[k])) {
                    L++;  // <- Increase L if we add one more basis vector
                }
            }
        }

        // check convergence on all roots
        if(!skip_check) {
            converged = 0;
            if(print) {
                outfile->Printf("Root      Eigenvalue       Delta  Converged?\n");
                outfile->Printf("---- -------------------- ------- ----------\n");
            }
            for(int k = 0; k < M; k++) {
                double diff = std::fabs(lambda.get(k) - lambda_old.get(k));
                bool this_converged = false;
                if(diff < e_convergence) {
                    this_converged = true;
                    converged++;
                }
                lambda_old.set(k,lambda.get(k));
                if(print) {
                    outfile->Printf("%3d  %20.14f %4.3e    %1s\n", k, lambda.get(k), diff,
                                    this_converged ? "Y" : "N");
                }
            }
        }

        outfile->Flush();

        iter++;
    }

    /* generate final eigenvalues and eigenvectors */
    //if(converged == M) {
    double** alpha_p = alpha.pointer();
    double** b_p = b.pointer();
    double* eps = Eigenvalues->pointer();
    double** v = Eigenvectors->pointer();

    for(int i = 0; i < M; i++) {
        eps[i] = lambda.get(i);
        for(int I = 0; I < N; I++){
            v[I][i] = 0.0;
        }
        for(int j = 0; j < L; j++) {
            for(int I=0; I < N; I++) {
                v[I][i] += alpha_p[j][i] * b_p[j][I];
            }
        }
        // Normalize v
        double norm = 0.0;
        for(int I = 0; I < N; I++) {
            norm += v[I][i] * v[I][i];
        }
        norm = std::sqrt(norm);
        for(int I = 0; I < N; I++) {
            v[I][i] /= norm;
        }
    }
    outfile->Printf("\n  The Davidson-Liu algorithm converged in %d iterations.", iter);
    outfile->Printf("\n  %s: %f s","Time spent diagonalizing H",t_davidson.elapsed());
    return true;
}

void SparseCISolver::compute_H_expectation_val(const std::vector<STLBitsetDeterminant> space, SharedVector& evals, const SharedMatrix evecs,int nroot, DiagonalizationMethod diag_method)
{
	// Build the Hamiltonian
	bool Hmat = true;
	SharedMatrix Hm;
	std::vector<std::pair<std::vector<int>,std::vector<double>>> Hs;
	size_t space_size = space.size();

	if( space.size() <= 100 ){
		Hm = build_full_hamiltonian(space);
	}else{
		if( diag_method == Full ){
			Hm = build_full_hamiltonian(space);
		}else if( diag_method == DavidsonLiuDense ){
			Hm = build_full_hamiltonian(space);
		}else{
			Hs = parallel_ ? build_sparse_hamiltonian_parallel(space) : build_sparse_hamiltonian(space);
			Hmat = false;
		}
	}

	// Compute expectation value
	evals.reset(new Vector("evals", nroot));
	if(Hmat){
		outfile->Printf("\n  Using full algorithm");
        for(int n = 0; n < nroot; ++n){
			for( size_t I = 0; I < space_size; ++I ){
				for( size_t J = 0; J < space_size; ++J){
					evals->add(n, evecs->get(I,n) *  Hm->get(I,J) * evecs->get(J,n) );
				}
			}
		}
	}else{
        for(int n = 0; n < nroot; ++n){
			for(size_t I = 0; I < space_size; ++I){
				std::vector<double> H_val = Hs[I].second;
				std::vector<int> H_idx = Hs[I].first;
				for(size_t J = 0, maxJ = H_val.size(); J < maxJ; ++J){
					evals->add(n, evecs->get(I,n) * H_val[J] * evecs->get(H_idx[J],n) );
				}
			}
		}
	}
	
}

}}
