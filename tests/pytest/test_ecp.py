import psi4
import pytest


def is_psi4_new_enough(version_feature_introduced):
    import psi4
    from pkg_resources import parse_version
    return parse_version(psi4.__version__) >= parse_version(version_feature_introduced)


using_psi4_1p1 = pytest.mark.skipif(is_psi4_new_enough("1.2a1.dev190") is False,
                                    reason="Psi4 1.1 not include ECPs. Update to developement head")


@using_psi4_1p1
def test_fci_ecp_1():
    """Water-Argon complex with ECP present; RHF energy from FCI."""

    nucenergy =   23.253113522963400
    refenergy =  -96.673557940220277

    arwater = psi4.geometry("""
        Ar  0.000000000000     0.000000000000     3.000000000000
        O   0.000000000000     0.000000000000    -0.071143036192
        H   0.000000000000    -0.758215806856     0.564545805801
        H   0.000000000000     0.758215806856     0.564545805801
    """)

    psi4.set_options({
        'scf_type':       'pk',
        'basis':          'lanl2dz',
        'df_scf_guess':   False,
        'd_convergence':  10,
    })

    psi4.set_module_options("FORTE", {
      'job_type': 'fci',
      'restricted_docc': [5,0,2,2],
      'active':          [0,0,0,0],
    })

    e = psi4.energy('forte')
    assert psi4.compare_values(nucenergy, arwater.nuclear_repulsion_energy(), 10, "Nuclear repulsion energy")
    assert psi4.compare_values(refenergy, e, 10, "FCI energy with ECP")


@using_psi4_1p1
def test_fci_ecp_2():
    """Water-Argon complex with ECP present; CASCI(6,6)."""

    nucenergy =   23.253113522963400
    refenergy =  -96.68319147222

    arwater = psi4.geometry("""
        Ar  0.000000000000     0.000000000000     3.000000000000
        O   0.000000000000     0.000000000000    -0.071143036192
        H   0.000000000000    -0.758215806856     0.564545805801
        H   0.000000000000     0.758215806856     0.564545805801
    """)

    psi4.set_options({
        'scf_type':      'pk',
        'basis':         'lanl2dz',
        'df_scf_guess':  False,
        'd_convergence': 10,
    })

    psi4.set_module_options("FORTE", {
      'job_type': 'fci',
      'restricted_docc': [4,0,1,1],
      'active':          [2,0,2,2],
    })

    e = psi4.energy('forte')
    assert psi4.compare_values(nucenergy, arwater.nuclear_repulsion_energy(), 10, "Nuclear repulsion energy")
    assert psi4.compare_values(refenergy, e, 10, "FCI energy with ECP")

