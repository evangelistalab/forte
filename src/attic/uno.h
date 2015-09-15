#ifndef UNO_H
#define UNO_H

#include <liboptions/liboptions.h>
#include <libmints/matrix.h>
#include <libmints/vector.h>
#include <libmints/wavefunction.h>
#include <libmints/writer.h>
#include <libmints/writer_file_prefix.h>
#include "integrals.h"

namespace psi{
namespace forte{

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
