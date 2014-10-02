#include <cmath>

#include <libpsio/psio.hpp>
#include <libmints/wavefunction.h>
#include <libmints/molecule.h>

#include "fcimc.h"

using namespace std;
using namespace psi;

namespace psi{ namespace libadaptive{

FCIMC::FCIMC(Options &options, ExplorerIntegrals* ints)
{
    outfile->Printf("\n\n      --------------------------------------");
    outfile->Printf("\n          Full Configuration Interaction");
    outfile->Printf("\n             Quantum Monte Carlo");
    outfile->Printf("\n");
    outfile->Printf("\n                Version 0.1.0");
    outfile->Printf("\n");
    outfile->Printf("\n       written by Francesco A. Evangelista");
    outfile->Printf("\n      --------------------------------------\n");

    int nwalkers = 10;
    std::vector<StringDeterminant> dets;
}

FCIMC::~FCIMC()
{
}

}} // EndNamespaces
