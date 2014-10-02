
#include "adaptive-ci.h"

#include <cmath>

#include <boost/timer.hpp>
#include <boost/format.hpp>

#include "adaptive-ci.h"
#include "cartographer.h"
#include "string_determinant.h"

using namespace std;
using namespace psi;

namespace psi{ namespace libadaptive{

///**
// * Find all the Slater determinants with an energy lower than determinant_threshold_
// */
//void Explorer::genetic_search(psi::Options& options)
//{
//    outfile->Printf("\n\n  Searching for the lowest energy Slater determinants using a genetic algorithm\n");

//    int nfrzc = frzcpi_.sum();
//    int nfrzv = frzvpi_.sum();
//    int naocc = nalpha_ - nfrzc;
//    int nbocc = nbeta_ - nfrzc;
//    int navir = nmo_ - naocc - nfrzc - nfrzv;
//    int nbvir = nmo_ - nbocc - nfrzc - nfrzv;

//    // copy the reference determinant
//    StringDeterminant det(reference_determinant_);

//    int npopulation = 100;



//    outfile->Printf("\n\n  The new reference determinant is:");
//    reference_determinant_.print();
//    outfile->Printf("\n  and its energy: %.12f Eh",min_energy_);

//    outfile->Printf("\n\n  The determinants visited fall in the range [%f,%f]",min_energy_,max_energy_);

//    outfile->Printf("\n\n  Number of full ci determinants    = %llu",num_total_dets);
//    outfile->Printf("\n\n  Number of determinants visited    = %ld (%e)",num_dets_visited,double(num_dets_visited) / double(num_total_dets));
//    outfile->Printf("\n  Number of determinants accepted   = %ld (%e)",num_dets_accepted,double(num_dets_accepted) / double(num_total_dets));
//    outfile->Printf("\n  Number of permutations visited    = %ld",num_permutations);
//    outfile->Printf("\n  Time spent on generating strings  = %f s",time_string);
//    outfile->Printf("\n  Time spent on generating dets     = %f s",time_dets);
//    outfile->Printf("\n  Precompute algorithm time elapsed = %f s",t.elapsed());
//    fflush(outfile);
//}

}} // EndNamespaces



