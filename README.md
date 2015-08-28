# forte
Adaptive quantum chemistry methods

Installation directions for forte:

Generating a Makefile is done via autogeneration of Makefile through PSI4.  
"psi4 --new-plugin testplugin"
Take the Makefile and move the Makefile to where the forte src is.

Forte requires the ambit library ( https://github.com/jturney/ambit )

In order to compile the code, you have to add these two lines to the Makefile: 

PSIPLUGIN = -L$(OBJDIR)/lib -lplugin -LWHERE_AMBIT_IS_LOCATED/obj/src -lambit
INCLUDES += -I/WHERE_AMBIT_IS_LOCATED/ambit/include/ambit



