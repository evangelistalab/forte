#include <boost/timer.hpp>

#include <numeric>
#include <vector>

#include "libmints/matrix.h"
#include "libmints/vector.h"

#include "bitset_determinant.h"
#include "wavefunction.h"

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

void FCIWfn::initial_guess(FCIWfn& diag,size_t num_dets)
{
    // Find the lowest energy determinants
    size_t tot_det = std::accumulate(detpi_.begin(),detpi_.end(),0);
    num_dets = std::min(num_dets,tot_det);

    std::vector<std::tuple<double,size_t,size_t,size_t,std::vector<bool>,std::vector<bool>>> dets(num_dets);

    for (auto& d : dets){
        std::get<0>(d) = 1.0e10;
    }

    double emax = 1.0e100;

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
                double e = determinant_energy(Ia,Ib,n);

                if (e < emax){
                    // Find where to inser this determinant
                    dets.pop_back();
                    auto it = std::find_if(dets.begin(),dets.end(),[&e](const std::tuple<double,size_t,size_t,size_t,std::vector<bool>,std::vector<bool>>& t){return e < std::get<0>(t);});

                    std::vector<bool> Ia_b(Ia,Ia + n);
                    std::vector<bool> Ib_b(Ib,Ib + n);
                    dets.insert(it,std::make_tuple(e,alfa_sym,addIa,addIb,Ia_b,Ib_b));
                    emax = std::get<0>(dets.back());
                }
            }
        } while (std::next_permutation(Ib,Ib+n));

    } while (std::next_permutation(Ia,Ia+n));


    int nroots = 4;
    Matrix H("H",num_dets,num_dets);
    Matrix evecs("Evecs",num_dets,num_dets);
    Vector evals("Evals",num_dets);

    for (size_t I = 0; I < num_dets; ++I){
        BitsetDeterminant detI(std::get<4>(dets[I]),std::get<5>(dets[I]));
        for (size_t J = I; J < num_dets; ++J){
            BitsetDeterminant detJ(std::get<4>(dets[J]),std::get<5>(dets[J]));
            double HIJ = detI.slater_rules(detJ);
            H.set(I,J,HIJ);
            H.set(J,I,HIJ);
        }
    }

    H.diagonalize(evecs,evals);

    for (size_t I = 0; I < num_dets; ++I){
        size_t alfa_sym = std::get<1>(dets[I]);
        size_t Ia = std::get<2>(dets[I]);
        size_t Ib = std::get<3>(dets[I]);
        C_[alfa_sym]->set(Ia,Ib,evecs.get(I,0));
    }

    hdiag_timer += t.elapsed();
    outfile->Printf("\n  Timing for initial guess  = %10.3f s\n",hdiag_timer);
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
