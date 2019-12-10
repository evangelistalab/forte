import psi4

def psi4_scf(geom, basis, reference, functional = 'hf'):
 
    # build the molecule object
    mol = psi4.geometry(geom)

    # set basis/options
    psi4.set_options({'basis': basis, 'reference' : reference, 'scf_type': 'pk'})

    # pipe output to the file output.dat
    psi4.core.set_output_file('output.dat', False)

    # run scf and return the energy and a wavefunction object (will work only if pass return_wfn=True)
    E_scf, wfn = psi4.energy(functional, return_wfn=True)

    return (E_scf, wfn)
