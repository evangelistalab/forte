#include "reference.h"

namespace psi{ namespace forte{

Reference::Reference()
{
}

Reference::Reference(double Eref, ambit::Tensor L1a, ambit::Tensor L1b,
                     ambit::Tensor L2aa, ambit::Tensor L2ab, ambit::Tensor L2bb)
    : Eref_(Eref), L1a_(L1a), L1b_(L1b), L2aa_(L2aa), L2ab_(L2ab), L2bb_(L2bb)
{
}

Reference::Reference(double Eref, ambit::Tensor L1a, ambit::Tensor L1b,
                     ambit::Tensor L2aa, ambit::Tensor L2ab, ambit::Tensor L2bb,
                     ambit::Tensor L3aaa, ambit::Tensor L3aab, ambit::Tensor L3abb, ambit::Tensor L3bbb)
    : Eref_(Eref), L1a_(L1a), L1b_(L1b), L2aa_(L2aa), L2ab_(L2ab), L2bb_(L2bb), L3aaa_(L3aaa), L3aab_(L3aab), L3abb_(L3abb), L3bbb_(L3bbb)
{
}
//This constructor will read the DMRG RDM from a file and read the energy
//Reference::Reference(std::string)
//{
//}
//
Reference::~Reference()
{
}

}}
