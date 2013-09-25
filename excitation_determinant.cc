#include <libmoinfo/libmoinfo.h>

#include "excitation_determinant.h"

using namespace std;
using namespace psi;

#include <psi4-dec.h>

namespace psi{ namespace libadaptive{

ExcitationDeterminant::ExcitationDeterminant()
    : naex_(0), nbex_(0)
{
}

ExcitationDeterminant::~ExcitationDeterminant()
{
}

ExcitationDeterminant::ExcitationDeterminant(const ExcitationDeterminant& det)
    : naex_(det.naex_), nbex_(det.nbex_), alpha_ops_(det.alpha_ops_), beta_ops_(det.beta_ops_)
{
}

ExcitationDeterminant& ExcitationDeterminant::operator=(const ExcitationDeterminant& rhs) {
    naex_ = rhs.naex_;
    nbex_ = rhs.nbex_;
    alpha_ops_  = rhs.alpha_ops_;
    beta_ops_  = rhs.beta_ops_;
    return *this;
}

/**
 * Print the determinant
 */
void ExcitationDeterminant::print()
{
    fprintf(outfile,"\n  {");
    for(int p = 0; p < nbex_; ++p){
        fprintf(outfile," %d",bann(p));
    }
    fprintf(outfile,"->");
    for(int p = 0; p < nbex_; ++p){
        fprintf(outfile," %d",bcre(p));
    }
    fprintf(outfile,"}{");
    for(int p = 0; p < naex_; ++p){
        fprintf(outfile," %d",aann(p));
    }
    fprintf(outfile,"->");
    for(int p = 0; p < naex_; ++p){
        fprintf(outfile," %d",acre(p));
    }
    fprintf(outfile,"}");
    fflush(outfile);
}

void ExcitationDeterminant::to_pitzer(const std::vector<int>& qt_to_pitzer)
{
    for(int p = 0; p < naex_; ++p){
        alpha_ops_[2 * p] = qt_to_pitzer[aann(p)];
        alpha_ops_[2 * p + 1] = qt_to_pitzer[acre(p)];
    }
    for(int p = 0; p < nbex_; ++p){
        beta_ops_[2 * p] = qt_to_pitzer[bann(p)];
        beta_ops_[2 * p + 1] = qt_to_pitzer[bcre(p)];
    }
}

}} // End Namespaces

