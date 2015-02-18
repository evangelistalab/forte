#include "reference.h"

namespace psi{ namespace libadaptive{

Reference::Reference()
{
}

Reference::Reference(double Eref, SharedTensor L1a, SharedTensor L1b,
                     SharedTensor L2aa, SharedTensor L2ab, SharedTensor L2bb)
    : Eref_(Eref), L1a_(L1a), L1b_(L1b), L2aa_(L2aa), L2ab_(L2ab), L2bb_(L2bb)
{
}

Reference::Reference(double Eref, SharedTensor L1a, SharedTensor L1b,
                     SharedTensor L2aa, SharedTensor L2ab, SharedTensor L2bb,
                     SharedTensor L3aaa, SharedTensor L3aab, SharedTensor L3abb, SharedTensor L3bbb)
    : Eref_(Eref), L1a_(L1a), L1b_(L1b), L2aa_(L2aa), L2ab_(L2ab), L2bb_(L2bb), L3aaa_(L3aaa), L3aab_(L3aab), L3abb_(L3abb), L3bbb_(L3bbb)
{
}

Reference::~Reference()
{
}

}}
