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
    fprintf(outfile,"\n\n      --------------------------------------");
    fprintf(outfile,"\n          Full Configuration Interaction");
    fprintf(outfile,"\n             Quantum Monte Carlo");
    fprintf(outfile,"\n");
    fprintf(outfile,"\n                Version 0.1.0");
    fprintf(outfile,"\n");
    fprintf(outfile,"\n       written by Francesco A. Evangelista");
    fprintf(outfile,"\n      --------------------------------------\n");

    int nwalkers = 10;
    std::vector<StringDeterminant> dets;
}

FCIMC::~FCIMC()
{
}

}} // EndNamespaces
