#include "psi4/psi4-dec.h"

#include "bitset_csf.h"
#include "fci_vector.h"

using namespace std;
using namespace psi;

namespace psi{ namespace forte{

BitsetCSF::BitsetCSF()
{
}

BitsetCSF::BitsetCSF(const std::vector<bool>& occupation_a,
                     const std::vector<bool>& occupation_b,
                     short s, short ms, short index)
    : docc_(occupation_a.size()), socc_(occupation_a.size()),
      s_(s), ms_(ms), index_(index)
{
    int nmo = occupation_a.size();
    for(int p = 0; p < nmo; ++p){
        if (occupation_a[p] and occupation_b[p]){
            docc_[p] = true;
        }else if (occupation_a[p] xor occupation_b[p]){
            socc_[p] = true;
        }
    }
}

void BitsetCSF::print() const
{
    outfile->Printf("\n  |");
    size_t nmo = docc_.size();
    for(size_t p = 0; p < nmo; ++p){
        if (docc_[p]){
            outfile->Printf("2");
        }else if (socc_[p]){
            outfile->Printf("1");
        }else{
            outfile->Printf("0");
        }
    }
    outfile->Printf("> (s = %s, ms = %d, idx = %d)",s_,ms_,index_);
    outfile->Flush();
}

std::string BitsetCSF::str() const
{
    size_t nmo = docc_.size();
    std::string s('0',nmo + 2);
    s[0] = '|';
    s[nmo + 1] = '>';
    for(size_t p = 0; p < nmo; ++p){
        if (docc_[p]){
            s[p + 1] = '2';
        }else if (socc_[p]){
            s[p + 1] = '1';
        }
    }
    return s;
}

}} // end namespace
