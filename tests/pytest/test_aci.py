import psi4

def test_psi4_basic():
    """tu1-h2o-energy"""
    #! Sample HF/cc-pVDZ H2O computation

    h2o = psi4.geometry("""
      O
      H 1 0.96
      H 1 0.96 2 104.5
    """)

    psi4.set_options({'basis': "cc-pVDZ"})
    psi4.energy('scf')

    assert psi4.compare_values(-76.0266327341067125, psi4.variable('SCF TOTAL ENERGY'), 6, 'SCF energy')


# LAB 25 May 2017 segfaulting after PQ-space in both psithon and psiapi forms
#def test_aci_1():
#    """Basic ACI calculation with energy threshold selection"""
#
#    import forte
#
#    refscf = -14.839846512738 #TEST
#    refaci = -14.888681221669 #TEST
#    refacipt2 = -14.890314474716 #TEST
#
#    li2 = psi4.geometry("""
#       Li
#       Li 1 2.0000
#    """)
#
#    psi4.set_options({
#      'basis': 'DZ',
#      'e_convergence': 10,
#      'd_convergence': 10,
#      'r_convergence': 10,
#      'guess': 'gwh',
#    })
#
#    psi4.set_module_options("SCF", {
#        'scf_type': 'pk',
#        'reference': 'rhf',
#        'docc': [2,0,0,0,0,1,0,0],
#    })
#
#    psi4.set_module_options("FORTE", {
#        'job_type': 'aci',
#        'multiplicity': 1,
#        'ACI_SELECT_TYPE': 'energy',
#        'sigma': 0.00001000,
#        'gamma': 100.0,
#        'aci_nroot': 1,
#        'charge': 0,
#        'aci_perturb_select': True,
#        'aci_enforce_spin_complete': False,
#    })
#
#    psi4.energy('scf')
#    assert psi4.compare_values(refscf, psi4.variable("CURRENT ENERGY"),9, "SCF energy")
#
#    psi4.energy('forte')
#    assert psi4.compare_values(refaci, psi4.variable("ACI ENERGY"),9, "ACI energy")
#    assert psi4.compare_values(refacipt2, psi4.variable("ACI+PT2 ENERGY"),8, "ACI+PT2 energy")


def test_aci_10():
    """Perform aci on benzyne"""

    refscf    = -229.20378006852584
    refaci    = -229.359450812283
    refacipt2 = -229.360444943286

    mbenzyne = psi4.geometry("""
      0 1
       C   0.0000000000  -2.5451795941   0.0000000000
       C   0.0000000000   2.5451795941   0.0000000000
       C  -2.2828001669  -1.3508352528   0.0000000000
       C   2.2828001669  -1.3508352528   0.0000000000
       C   2.2828001669   1.3508352528   0.0000000000
       C  -2.2828001669   1.3508352528   0.0000000000
       H  -4.0782187459  -2.3208602146   0.0000000000
       H   4.0782187459  -2.3208602146   0.0000000000
       H   4.0782187459   2.3208602146   0.0000000000
       H  -4.0782187459   2.3208602146   0.0000000000

      units bohr
    """)

    psi4.set_options({
       'basis': 'DZ',
       'df_basis_mp2': 'cc-pvdz-ri',
       'reference': 'uhf',
       'scf_type': 'pk',
       'd_convergence': 10,
       'e_convergence': 12,
       'guess': 'gwh',
    })

    psi4.set_module_options("FORTE", {
      'root_sym': 0,
      'frozen_docc':     [2,1,0,0,0,0,2,1],
      'restricted_docc': [3,2,0,0,0,0,2,3],
      'active':          [1,0,1,2,1,2,1,0],
      'multiplicity': 1,
      'aci_nroot': 1,
      'job_type': 'aci',
      'sigma': 0.001,
      'aci_select_type': 'aimed_energy',
      'aci_spin_projection': 1,
      'aci_enforce_spin_complete': True,
      'aci_add_aimed_degenerate': False,
      'aci_project_out_spin_contaminants': False,
      'diag_algorithm': 'full',
      'aci_quiet_mode': True,
    })

    scf = psi4.energy('scf')
    assert psi4.compare_values(refscf, scf,10,"SCF Energy")

    psi4.energy('forte')
    assert psi4.compare_values(refaci, psi4.variable("ACI ENERGY"),10,"ACI energy")
    assert psi4.compare_values(refacipt2, psi4.variable("ACI+PT2 ENERGY"),8,"ACI+PT2 energy")

