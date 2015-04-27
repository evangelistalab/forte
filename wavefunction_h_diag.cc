#include <boost/timer.hpp>

//#include <libmoinfo/libmoinfo.h>
//#include <libqt/qt.h>

#include "wavefunction.h"

//using namespace std;
//using namespace psi;
//using namespace boost;


//#include <psi4-dec.h>

namespace psi{ namespace libadaptive{

void FCIWfn::form_H_diagonal()
{
    boost::timer t;

    int wfn_sym = symmetry_;
    int n  = ncmo_;
    int ka = alfa_graph_->nones();
    int kb = beta_graph_->nones();

    bool* Ia = new bool[n];
    bool* Ib = new bool[n];

    // Generate the alfa string 1111000000
    //                          {ka}{n-ka}
    for(int i = 0; i < n - ka; ++i) Ia[i] = false; // 0
    for(int i = n - ka; i < n; ++i) Ia[i] = true;  // 1
    // Loop over all alfa strings
    do{
        // Compute irrep
        int alfa_sym = alfa_graph_->sym(Ia);
        int beta_sym = alfa_sym ^ wfn_sym;

        double** C_ha = C_[alfa_sym]->pointer();

        // Generate the beta string 1111000000
        //                          {kb}{n-kb}
        for(int i = 0; i < n - kb; ++i) Ib[i] = false; // 0
        for(int i = n - kb; i < n; ++i) Ib[i] = true;  // 1
        // Loop over all beta strings
        do{
            // Check if the product of strings gives the right irrep
            if(beta_graph_->sym(Ib) == beta_sym){
                size_t addIa = alfa_graph_->rel_add(Ia);
                size_t addIb = beta_graph_->rel_add(Ib);
                C_ha[addIa][addIb] = determinant_energy(Ia,Ib,n);
                //        outfile->Printf("\n |[%1d][%3d][%3d]> energy = %20.12f",alfa_sym,static_cast<int> (addIa),
                //                                                                         static_cast<int> (addIb),coefficients[alfa_sym][addIa][addIb]);
            }
        } while (std::next_permutation(Ib,Ib+n));

    } while (std::next_permutation(Ia,Ia+n));

    hdiag_timer += t.elapsed();
    outfile->Printf("\n  Timing for Hdiag          = %10.3f s\n",hdiag_timer);
    outfile->Flush();
}

double FCIWfn::determinant_energy(bool*& Ia,bool*& Ib,int n)
{
    double energy(ints_->frozen_core_energy());
    for(int p = 0; p < n; ++p){
        if(Ia[p]) energy += oei_aa(p,p);
        if(Ib[p]) energy += oei_bb(p,p);
        for(int q = 0; q < n; ++q){
            if(Ia[p] && Ia[q])
                energy += 0.5 * tei_aaaa(p,q,p,q);
            if(Ib[p] && Ib[q])
                energy += 0.5 * tei_bbbb(p,q,p,q);
            if(Ia[p] && Ib[q])
                energy += tei_aabb(p,q,p,q);
        }
    }
    return(energy);
}

}}
