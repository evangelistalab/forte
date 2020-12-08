import sys
from .pymodule import *

from .register_forte_options import *

def reset_forte_options():
    """
    Create a new ForteOptions object, set default values, push to Psi4, and return.
    """
    # Create a freash ForteOptions object
    forte_options = forte.ForteOptions()

    # Register options defined in Forte in the forte_options object
    register_forte_options(forte_options)

    # If we are running psi4, push the options defined in forte_options to psi
    if 'psi4' in sys.modules:
        psi_options = psi4.core.get_options()
        psi_options.set_current_module('FORTE')
        # TODO: need Psi4 functionality to clean up all Forte module options
        # something like psi_options.clean_options_module('FORTE')
        # because the function below only calls `add` functions of Psi4 Options,
        # which doesn't update option values if the option key is already in the map.
        forte_options.push_options_to_psi4(psi_options)

    return forte_options
