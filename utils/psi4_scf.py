import psi4

def psi4_scf(geom, basis, reference, functional = 'hf', options = {}):
 
    # build the molecule object
    mol = psi4.geometry(geom)

    # add basis/reference/scf_type to options passed by the user
    options['basis'] = basis
    options['reference'] = reference
    options['scf_type'] = 'pk'

    psi4.set_options(options)

    # pipe output to the file output.dat
    psi4.core.set_output_file('output.dat', True)

    # run scf and return the energy and a wavefunction object (will work only if pass return_wfn=True)
    E_scf, wfn = psi4.energy(functional, return_wfn=True)

    return (E_scf, wfn)
