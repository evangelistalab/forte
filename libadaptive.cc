#include <libplugin/plugin.h>
#include "psi4-dec.h"
#include <libdpd/dpd.h>
#include "psifiles.h"
#include <libpsio/psio.hpp>
#include <libtrans/integraltransform.h>
#include <libmints/wavefunction.h>

#include "explorer.h"

// This allows us to be lazy in getting the spaces in DPD calls
#define ID(x) ints.DPD_ID(x)

INIT_PLUGIN

namespace psi{ namespace libadaptive{

extern "C" int
read_options(std::string name, Options &options)
{
    if (name == "LIBADAPTIVE" || options.read_globals()) {
        /*- The amount of information printed
            to the output file -*/
        options.add_int("PRINT", 1);

        // Options for the Explorer class
        /*- The symmetry of the electronic state.  If a value is provided
            it overrides the multiplicity of the SCF solution. -*/
        options.add_int("SYMMETRY",0);
        /*- The multiplicity of the electronic state.  If a value is provided
            it overrides the multiplicity of the SCF solution. -*/
        options.add_int("MULTIPLICITY",0);
        /*- The charge of the molecule.  If a value is provided
            it overrides the multiplicity of the SCF solution. -*/
        options.add_int("CHARGE",0);
        /*- The energy threshold for the determinant energy in Hartree -*/
        options.add_double("DET_THRESHOLD",1.0);
        /*- The energy threshold for the MP denominators energy in Hartree -*/
        options.add_double("DEN_THRESHOLD",0.0);
        /*- Write an output file? -*/
        options.add_bool("WRITE_FILE",true);
        /*- Write the determinant occupation? -*/
        options.add_bool("WRITE_OCCUPATION",true);
        /*- Write the determinant energy? -*/
        options.add_bool("WRITE_DET_ENERGY",true);
        /*- Write the denominator energy? -*/
        options.add_bool("WRITE_DEN_ENERGY",false);
        /*- Write the excitation level? -*/
        options.add_bool("WRITE_EXC_LEVEL",false);
        /*- The energy buffer for building the Hamiltonian matrix in Hartree -*/
        options.add_double("H_BUFFER",0.0);
    }
    return true;
}


extern "C" PsiReturnType
libadaptive(Options &options)
{
    Explorer explorer(options);
    return Success;
}

}} // End Namespaces
