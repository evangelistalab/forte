/*
 *  string_oo_list.cpp
 *  Capriccio
 *
 *  Created by Francesco Evangelista on 3/13/09.
 *  Copyright 2009 __MyCompanyName__. All rights reserved.
 *
 */

#include "string_lists.h"

using namespace boost;

namespace psi{ namespace libadaptive{

/**
 * Returns the list of alfa strings connected by a^{+}_p a^{+}_q a_q a_p
 * @param pq_sym symmetry of the pq pair
 * @param pq     relative PAIRINDEX of the pq pair
 * @param h      symmetry of the I strings in the list
 */
std::vector<StringSubstitution>& StringLists::get_alfa_oo_list(int pq_sym,size_t pq, int h)
{
    boost::tuple<int,size_t,int> pq_pair(pq_sym,pq,h);
    return alfa_oo_list[pq_pair];
}

/**
 * Returns the list of beta strings connected by a^{+}_p a^{+}_q a_q a_p
 * @param pq_sym symmetry of the pq pair
 * @param pq     relative PAIRINDEX of the pq pair
 * @param h      symmetry of the I strings in the list
 */
std::vector<StringSubstitution>& StringLists::get_beta_oo_list(int pq_sym,size_t pq, int h)
{
    boost::tuple<int,size_t,int> pq_pair(pq_sym,pq,h);
    return beta_oo_list[pq_pair];
}

/**
 * Generate the list of strings connected by a^{+}_p a^{+}_q a_q a_p
 * @param graph graph for numbering the strings generated
 * @param list  the OO list
 */
void StringLists::make_oo_list(GraphPtr graph,OOList& list)
{
    // Loop over irreps of the pair pq
    for(int pq_sym = 0; pq_sym < nirrep_; ++pq_sym){
        size_t max_pq = pairpi_[pq_sym];
        for(size_t pq = 0; pq < max_pq; ++pq){
            make_oo(graph,list,pq_sym,pq);
        }
    }
}

/**
 * Generate all the pairs of strings I,J connected by
 * Op = a^{+}_p a^{+}_q a_q a_p,  that is: J = ± Op I.
 * @param pq_sym symmetry of the pq pair
 * @param pq     relative PAIRINDEX of the pq pair
 */
void StringLists::make_oo(GraphPtr graph,OOList& list,int pq_sym,size_t pq)
{
    int k = graph->nones() - 2;
    if(k >= 0){
        int p = nn_list[pq_sym][pq].first;
        int q = nn_list[pq_sym][pq].second;

        int n = graph->nbits() - 2;
        bool* b = new bool[n];
        bool* I = new bool[ncmo_];
        bool* J = new bool[ncmo_];

        for(int h = 0; h < nirrep_; ++h){
            // Create the key to the map
            boost::tuple<int,size_t,int> pq_pair(pq_sym,pq,h);

            // Generate the strings 1111100000
            //                      { k }{n-k}
            for(int i = 0; i < n - k; ++i) b[i] = false; // 0
            for(int i = n - k; i < n; ++i) b[i] = true;  // 1
            do{
                int k = 0;
                for (int i = 0; i < q ; ++i){
                    J[i] = I[i] = b[k];
                    k++;
                }
                for (int i = q + 1; i < p ; ++i){
                    J[i] = I[i] = b[k];
                    k++;
                }
                for (int i = p + 1; i < ncmo_ ; ++i){
                    J[i] = I[i] = b[k];
                    k++;
                }
                I[p] = true;
                I[q] = true;
                J[p] = true;
                J[q] = true;
                // Add the sting only of irrep(I) is h
                if(graph->sym(I) == h)
                    list[pq_pair].push_back(StringSubstitution(1,graph->rel_add(I),graph->rel_add(J)));
            } while (std::next_permutation(b,b+n));
        }  // End loop over h

        delete[] J;
        delete[] I;
        delete[] b;
    }
}

}}
