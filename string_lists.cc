#include <boost/timer.hpp>

#include <psi4-dec.h>

#include "string_lists.h"

namespace psi{ namespace libadaptive{

StringLists::StringLists(RequiredLists required_lists,Dimension cmopi,std::vector<size_t> core_mo,std::vector<size_t> cmo_to_mo, size_t na, size_t nb)
    : required_lists_(required_lists), cmopi_(cmopi), na_(na), nb_(nb), fomo_to_mo_(core_mo), cmo_to_mo_(cmo_to_mo)
{
    startup();
}

void StringLists::startup()
{
    nirrep_ = cmopi_.n();
    ncmo_ = cmopi_.sum();

    cmopi_offset_.push_back(0);
    for(int h = 1; h < nirrep_; ++h){
        cmopi_offset_.push_back(cmopi_offset_[h-1] + cmopi_[h-1]);
    }

    std::vector<int> cmopi_int;

    for (int h = 0; h < nirrep_; ++h){
        cmopi_int.push_back(cmopi_[h]);
    }

    // Allocate the alfa and beta graphs
    alfa_graph_ = boost::shared_ptr<BinaryGraph>(new BinaryGraph(ncmo_,na_,cmopi_int));
    beta_graph_ = boost::shared_ptr<BinaryGraph>(new BinaryGraph(ncmo_,nb_,cmopi_int));
    pair_graph_ = boost::shared_ptr<BinaryGraph>(new BinaryGraph(ncmo_,2,cmopi_int));
    if (na_ >= 3){
        alfa_graph_3h_ = boost::shared_ptr<BinaryGraph>(new BinaryGraph(ncmo_,na_ - 3,cmopi_int));
    }
    if (nb_ >= 3){
        beta_graph_3h_ = boost::shared_ptr<BinaryGraph>(new BinaryGraph(ncmo_,nb_ - 3,cmopi_int));
    }


    nas_ = 0;
    nbs_ = 0;
    for(int h = 0; h < nirrep_; ++h){
        nas_ += alfa_graph_->strpi(h);
        nbs_ += beta_graph_->strpi(h);
    }

    outfile->Printf("\n  ==> String Lists <==\n");
    outfile->Printf("\n  Number of alpha electrons     = %zu",na_);
    outfile->Printf("\n  Number of beta electrons      = %zu",nb_);
    outfile->Printf("\n  Number of alpha strings       = %zu",alfa_graph_->nstr());
    outfile->Printf("\n  Number of beta strings        = %zu",nbs_);
    if (na_ >= 3){
        outfile->Printf("\n  Number of alpha strings (N-3) = %zu",alfa_graph_3h_->nstr());
    }
    if (nb_ >= 3){
        outfile->Printf("\n  Number of beta strings (N-3)  = %zu",beta_graph_3h_->nstr());
    }

    outfile->Flush();

    // Timers
    double vo_list_timer = 0.0;
    double nn_list_timer = 0.0;
    double oo_list_timer = 0.0;
    double kh_list_timer = 0.0;
    double vovo_list_timer = 0.0;
    double vvoo_list_timer = 0.0;

    {
        boost::timer t;
        make_pair_list(pair_graph_,nn_list);
        nn_list_timer += t.elapsed();
    }

    {
        boost::timer t;
        make_vo_list(alfa_graph_,alfa_vo_list);
        make_vo_list(beta_graph_,beta_vo_list);
        vo_list_timer += t.elapsed();
    }

    {
        boost::timer t;
        make_oo_list(alfa_graph_,alfa_oo_list);
        make_oo_list(beta_graph_,beta_oo_list);
        oo_list_timer += t.elapsed();
    }

    {
        boost::timer t;
        make_kh_list(alfa_graph_,alfa_kh_list);
        make_kh_list(beta_graph_,beta_kh_list);
        kh_list_timer += t.elapsed();
    }

    {
        boost::timer t;
        make_3h_list(alfa_graph_,alfa_graph_3h_,alfa_3h_list);
        make_3h_list(beta_graph_,beta_graph_3h_,beta_3h_list);
        kh_list_timer += t.elapsed();
    }

    if(required_lists_ == twoSubstituitionVVOO){
        boost::timer t;
        make_vvoo_list(alfa_graph_,alfa_vvoo_list);
        make_vvoo_list(beta_graph_,beta_vvoo_list);
        vvoo_list_timer += t.elapsed();
    }else if (required_lists_ == twoSubstituitionVOVO){
        boost::timer t;
        make_vovo_list(alfa_graph_,alfa_vovo_list);
        make_vovo_list(beta_graph_,beta_vovo_list);
        vovo_list_timer += t.elapsed();
    }

    double total_time = nn_list_timer + vo_list_timer + oo_list_timer + vvoo_list_timer + vovo_list_timer;
    outfile->Printf("\n  Timing for NN strings     = %10.3f s",nn_list_timer);
    outfile->Printf("\n  Timing for VO strings     = %10.3f s",vo_list_timer);
    outfile->Printf("\n  Timing for OO strings     = %10.3f s",oo_list_timer);
    outfile->Printf("\n  Timing for Knowles-Handy  = %10.3f s",kh_list_timer);
    outfile->Printf("\n  Timing for VVOO strings   = %10.3f s",vvoo_list_timer);
    outfile->Printf("\n  Timing for VOVO strings   = %10.3f s",vovo_list_timer);
    outfile->Printf("\n  Total timing              = %10.3f s",total_time);
    outfile->Flush();
}


/**
 * Generate all the pairs p > q with pq in pq_sym
 * these are stored as pair<int,int> in pair_list[pq_sym][pairpi]
 */
void StringLists::make_pair_list(GraphPtr graph,NNList& list)
{
    // Loop over irreps of the pair pq
    for(int pq_sym = 0; pq_sym < nirrep_; ++pq_sym){
        list.push_back(std::vector<std::pair<int,int> >(0));
        // Loop over irreps of p
        for(int p_sym = 0; p_sym < nirrep_; ++p_sym){
            int q_sym = pq_sym ^ p_sym;
            for(int p_rel = 0; p_rel < cmopi_[p_sym]; ++p_rel){
                for(int q_rel = 0; q_rel < cmopi_[q_sym]; ++q_rel){
                    int p_abs = p_rel + cmopi_offset_[p_sym];
                    int q_abs = q_rel + cmopi_offset_[q_sym];
                    if(p_abs > q_abs)
                        list[pq_sym].push_back(std::make_pair(p_abs,q_abs));
                }
            }
        }
        pairpi_.push_back(list[pq_sym].size());
    }
    //  int h = 0;
    //  foreach(PairList& list_irrep,list){
    //    outfile->Printf("\n Irrep %d",h);
    //    foreach(Pair& pq, list_irrep){
    //
    //      outfile->Printf("\n Pair [%2d,%2d]",pq.first,pq.second);
    //    }
    //    h++;
    //  }
    //  outfile->Flush();
}

short StringLists::string_sign(const bool* I,size_t n)
{
    short sign = 1;
    for(int i = 0; i < n; ++i){  // This runs up to the operator before n
        if (I[i]) sign *= -1;
    }
    return(sign);
}

void StringLists::print_string(bool* I,size_t n)
{
    //outfile->Printf();
    for (size_t i = 0; i < n; ++i){
        outfile->Printf("%1d",I[i]);
    }
}

///*
//#ifdef PRINT_COMBINATIONS
//    cout << "\n p = " << p << " q = " << q;
//    cout << "\n b = ";
//    for ( int i = 0 ; i < n ; ++i){
//      cout << b[i] ? 1 : 0;
//    }
//    cout << "\n I = ";
//    for ( int i = 0 ; i < ncmos; ++i){
//      cout << I[i] ? 1 : 0;
//    }
//    cout << "  Add(I) = " << graph->address(I);
//    cout << "\n J = ";
//    for ( int i = 0 ; i < ncmos; ++i){
//      cout << J[i] ? 1 : 0;
//    }
//    cout << "  Add(J) = " <<  graph->address(J);
//    cout << "\n sign = " << sign << endl
//#endif
//#include "Memory/Memory.h"



//  int n = graph->nbits() - 2 - (p==q ? 0 : 1);
//  int k = graph->nones() - 2;
//  bool* b = new bool[n];
//  bool* I = new bool[ncmos];
//  bool* J = new bool[ncmos];

//  for(int h = 0; h < nirrep_; ++h){
//    // Create the key to the map
//    boost::tuple<size_t,size_t,size_t,size_t,int> pqrs_pair(p,q,r,s,h);

//    for(int i = 0; i < ncmos; ++i) I[i] = J[i] = false; // 0

//    I[q] = true;
//    I[s] = true;

//    J[s] = false;
//    J[r] = true;
//    J[q] = false;
//    J[p] = true;

//    // Generate the strings 1111100000
//    //                      { k }{n-k}
//    for(int i = 0; i < n - k; ++i) b[i] = false; // 0
//    for(int i = n - k; i < n; ++i) b[i] = true;  // 1
//    do{
//      int k = 0;
//      short sign = 1;
//      for (int i = 0; i < min(p,q) ; ++i){
//        J[i] = I[i] = b[k];
//        k++;
//      }
//      for (int i = min(p,q) + 1; i < max(p,q) ; ++i){
//        J[i] = I[i] = b[k];
//        if(b[k])
//          sign *= -1;
//        k++;
//      }
//      for (int i = max(p,q) + 1; i < ncmos ; ++i){
//        J[i] = I[i] = b[k];
//        k++;
//      }
//      I[p] = 0;
//      I[q] = 1;
//      J[q] = 0;
//      J[p] = 1;

//      // Add the sting only of irrep(I) is h
//      if(graph->sym(I) == h)
//        list[pq_pair].push_back(StringSubstitution(sign,graph->rel_add(I),graph->rel_add(J)));
//  } while (std::next_permutation(b,b+n));

//  }  // End loop over h

//  delete[] J;
//  delete[] I;
//  delete[] b;
//*/

}}
