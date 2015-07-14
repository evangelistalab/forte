#ifndef UNO_H
#define UNO_H

#include <libmints/matrix.h>
#include <libmints/vector.h>
#include <libmints/wavefunction.h>
#include <liboptions/liboptions.h>
#include "integrals.h"

namespace psi{
namespace libadaptive{

class UNO
{
public:
    // => Constructor <= //
    UNO(Options &options);
    //  => Destructor <= //
    ~UNO();

private:
};

}}
#endif // UNO_H
