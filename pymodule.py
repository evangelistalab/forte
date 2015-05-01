import psi4
import re
import os
import inputparser
import math
import warnings
from driver import *
from wrappers import *
from molutil import *
import p4util
from p4xcpt import *


def run_libadaptive(name, **kwargs):
    r"""Function encoding sequence of PSI module and plugin calls so that
    libadaptive can be called via :py:func:`~driver.energy`. For post-scf plugins.

    >>> energy('libadaptive')

    """
    lowername = name.lower()
    kwargs = p4util.kwargs_lower(kwargs)

    # Your plugin's psi4 run sequence goes here
    #psi4.set_local_option('LIBADAPTIVE', 'PRINT', 1)
    #scf_helper(name, **kwargs)
    returnvalue = psi4.plugin('libadaptive.so')
    psi4.set_variable('CURRENT ENERGY', returnvalue)
    return returnvalue

def run_adaptive_ci(name, **kwargs):
    r"""Function encoding sequence of PSI module and plugin calls so that
    libadaptive can be called via :py:func:`~driver.energy`. For post-scf plugins.

    >>> energy('aci')

    """
    lowername = name.lower()
    kwargs = p4util.kwargs_lower(kwargs)

    # Your plugin's psi4 run sequence goes here
    scf_helper(name, **kwargs)
    psi4.set_local_option('LIBADAPTIVE', 'JOB_TYPE', 'ACI_SPARSE')
    returnvalue = psi4.plugin('libadaptive.so')
    psi4.set_variable('CURRENT ENERGY', returnvalue)
    return returnvalue

def run_ex_aci(name, **kwargs):
    r"""Function encoding sequence of PSI module and plugin calls so that
    libadaptive can be called via :py:func:`~driver.energy`. For post-scf plugins.

    >>> energy('ex-aci')

    """
    lowername = name.lower()
    kwargs = p4util.kwargs_lower(kwargs)

    # Your plugin's psi4 run sequence goes here
    scf_helper(name, **kwargs)
    psi4.set_local_option('LIBADAPTIVE', 'JOB_TYPE', 'EX-ACI')
    returnvalue = psi4.plugin('libadaptive.so')
    psi4.set_variable('CURRENT ENERGY', returnvalue)
    return returnvalue

# Adaptive Path-Integral CI
def run_adaptive_pici(name, **kwargs):
    r"""Function encoding sequence of PSI module and plugin calls so that
    libadaptive can be called via :py:func:`~driver.energy`. For post-scf plugins.

    >>> energy('apici')

    """
    lowername = name.lower()
    kwargs = p4util.kwargs_lower(kwargs)

    # Your plugin's psi4 run sequence goes here
    scf_helper(name, **kwargs)
    psi4.set_local_option('LIBADAPTIVE', 'JOB_TYPE', 'APICI')
    returnvalue = psi4.plugin('libadaptive.so')
    psi4.set_variable('CURRENT ENERGY', returnvalue)
    return returnvalue

# (Fast) Adaptive Path-Integral CI
def run_fast_adaptive_pici(name, **kwargs):
    r"""Function encoding sequence of PSI module and plugin calls so that
    libadaptive can be called via :py:func:`~driver.energy`. For post-scf plugins.

    >>> energy('apici')

    """
    lowername = name.lower()
    kwargs = p4util.kwargs_lower(kwargs)

    # Your plugin's psi4 run sequence goes here
    scf_helper(name, **kwargs)
    psi4.set_local_option('LIBADAPTIVE', 'JOB_TYPE', 'FAPICI')
    returnvalue = psi4.plugin('libadaptive.so')
    psi4.set_variable('CURRENT ENERGY', returnvalue)
    return returnvalue

# Full CI Quantum Monte-Carlo
def run_fciqmc(name, **kwargs):
    r"""Function encoding sequence of PSI module and plugin calls so that
    libadaptive can be called via :py:func:`~driver.energy`. For post-scf plugins.

    >>> energy('fciqmc')

    """
    lowername = name.lower()
    kwargs = p4util.kwargs_lower(kwargs)

    # Your plugin's psi4 run sequence goes here
    scf_helper(name, **kwargs)
    psi4.set_local_option('LIBADAPTIVE', 'JOB_TYPE', 'FCIQMC')
    returnvalue = psi4.plugin('libadaptive.so')
    psi4.set_variable('CURRENT ENERGY', returnvalue)
    return returnvalue


# Adaptive Path-Integral CI
def run_fno_apifci(name, **kwargs):
    r"""Function encoding sequence of PSI module and plugin calls so that
    libadaptive can be called via :py:func:`~driver.energy`. For post-scf plugins.

    >>> energy('apici')

    """
    lowername = name.lower()
    kwargs = p4util.kwargs_lower(kwargs)

    # Your plugin's psi4 run sequence goes here
    #scf_helper(name, **kwargs)
    run_fnocc("fno-mp3", **kwargs)
    mints = psi4.MintsHelper()
    mints.integrals() 
    psi4.set_local_option('LIBADAPTIVE', 'JOB_TYPE', 'APICI')
    returnvalue = psi4.plugin('libadaptive.so')
    psi4.set_variable('CURRENT ENERGY', returnvalue)
    return returnvalue

def run_ct(name, **kwargs):
    r"""Function encoding sequence of PSI module and plugin calls so that
    libadaptive can be called via :py:func:`~driver.energy`. For post-scf plugins.

    >>> energy('ct')

    """
    lowername = name.lower()
    kwargs = p4util.kwargs_lower(kwargs)

    # Your plugin's psi4 run sequence goes here
    scf_helper(name, **kwargs)
    psi4.set_local_option('LIBADAPTIVE', 'JOB_TYPE', 'TENSORSRG')
    psi4.set_local_option('LIBADAPTIVE','SRG_MODE','CT')
    psi4.plugin('libadaptive.so')
    returnvalue = psi4.get_variable('CURRENT ENERGY')
    return returnvalue

def run_ct_ci(name, **kwargs):
    r"""Function encoding sequence of PSI module and plugin calls so that
    libadaptive can be called via :py:func:`~driver.energy`. For post-scf plugins.

    >>> energy('ct')

    """
    lowername = name.lower()
    kwargs = p4util.kwargs_lower(kwargs)

    # Your plugin's psi4 run sequence goes here
    scf_helper(name, **kwargs)
    psi4.set_local_option('LIBADAPTIVE', 'JOB_TYPE', 'TENSORSRG-CI')
    psi4.set_local_option('LIBADAPTIVE','SRG_MODE','CT')
    psi4.plugin('libadaptive.so')
    returnvalue = psi4.get_variable('CURRENT ENERGY')
    return returnvalue

def run_sr_dsrg_aci(name, **kwargs):
    r"""Function encoding sequence of PSI module and plugin calls so that
    libadaptive can be called via :py:func:`~driver.energy`. For post-scf plugins.

    >>> energy('sr-dsrg-aci')

    """
    lowername = name.lower()
    kwargs = p4util.kwargs_lower(kwargs)

    # Your plugin's psi4 run sequence goes here
    scf_helper(name, **kwargs)
    psi4.set_local_option('LIBADAPTIVE','JOB_TYPE','SR-DSRG-ACI')
    psi4.set_local_option('LIBADAPTIVE','SRG_MODE','CT')
    psi4.plugin('libadaptive.so')
    returnvalue = psi4.get_variable('CURRENT ENERGY')
    return returnvalue

def run_sr_dsrg_apici(name, **kwargs):
    r"""Function encoding sequence of PSI module and plugin calls so that
    libadaptive can be called via :py:func:`~driver.energy`. For post-scf plugins.

    >>> energy('sr-dsrg-apici')

    """
    lowername = name.lower()
    kwargs = p4util.kwargs_lower(kwargs)

    # Your plugin's psi4 run sequence goes here
    scf_helper(name, **kwargs)
    psi4.set_local_option('LIBADAPTIVE','JOB_TYPE','SR-DSRG-APICI')
    psi4.set_local_option('LIBADAPTIVE','SRG_MODE','CT')
    psi4.plugin('libadaptive.so')
    returnvalue = psi4.get_variable('CURRENT ENERGY')
    return returnvalue

def run_srg(name, **kwargs):
    r"""Function encoding sequence of PSI module and plugin calls so that
    libadaptive can be called via :py:func:`~driver.energy`. For post-scf plugins.

    >>> energy('srg')

    """
    lowername = name.lower()
    kwargs = p4util.kwargs_lower(kwargs)

    # Your plugin's psi4 run sequence goes here
    scf_helper(name, **kwargs)
    psi4.set_local_option('LIBADAPTIVE', 'JOB_TYPE','TENSORSRG')
    psi4.set_local_option('LIBADAPTIVE','SRG_MODE','SRG')
    psi4.plugin('libadaptive.so')
    returnvalue = psi4.get_variable('CURRENT ENERGY')
    return returnvalue

def run_sr_lctsd(name, **kwargs):
    r"""Function encoding sequence of PSI module and plugin calls so that
    libadaptive can be called via :py:func:`~driver.energy`. For post-scf plugins.

    >>> energy('sr-lctsd')

    """
    lowername = name.lower()
    kwargs = p4util.kwargs_lower(kwargs)

    # Your plugin's psi4 run sequence goes here
    scf_helper(name, **kwargs)
    psi4.set_local_option('LIBADAPTIVE', 'JOB_TYPE', 'SRG')
    psi4.set_local_option('LIBADAPTIVE','SRG_MODE','CT')
    psi4.plugin('libadaptive.so')
    returnvalue = psi4.get_variable('CURRENT ENERGY')
    return returnvalue

def run_sr_srgsd(name, **kwargs):
    r"""Function encoding sequence of PSI module and plugin calls so that
    libadaptive can be called via :py:func:`~driver.energy`. For post-scf plugins.

    >>> energy('sr-srgsd')

    """
    lowername = name.lower()
    kwargs = p4util.kwargs_lower(kwargs)

    # Your plugin's psi4 run sequence goes here
    scf_helper(name, **kwargs)
    psi4.set_local_option('LIBADAPTIVE', 'JOB_TYPE','SRG')
    psi4.set_local_option('LIBADAPTIVE','SRG_MODE','SRG')
    psi4.plugin('libadaptive.so')
    returnvalue = psi4.get_variable('CURRENT ENERGY')
    return returnvalue

def run_sr_dsrgsd(name, **kwargs):
    r"""Function encoding sequence of PSI module and plugin calls so that
    libadaptive can be called via :py:func:`~driver.energy`. For post-scf plugins.

    >>> energy('sr-dsrgsd')

    """
    lowername = name.lower()
    kwargs = p4util.kwargs_lower(kwargs)

    # Your plugin's psi4 run sequence goes here
    scf_helper(name, **kwargs)
    psi4.set_local_option('LIBADAPTIVE', 'JOB_TYPE','SRG')
    psi4.set_local_option('LIBADAPTIVE','SRG_MODE','DSRG')
    psi4.plugin('libadaptive.so')
    returnvalue = psi4.get_variable('CURRENT ENERGY')
    return returnvalue

def run_mrdsrgpt2(name, **kwargs):
    r"""Function encoding sequence of PSI module and plugin calls so that
    libadaptive can be called via :py:func:`~driver.energy`. For post-scf plugins.

    >>> energy('mr-dsrg-pt2')

    """
    lowername = name.lower()
    kwargs = p4util.kwargs_lower(kwargs)

    # Your plugin's psi4 run sequence goes here
    psi4.set_local_option('LIBADAPTIVE', 'JOB_TYPE','MR-DSRG-PT2')
    psi4.plugin('libadaptive.so')
    returnvalue = psi4.get_variable('CURRENT ENERGY')
    return returnvalue

def run_dsrg_mrpt2(name, **kwargs):
    r"""Function encoding sequence of PSI module and plugin calls so that
    libadaptive can be called via :py:func:`~driver.energy`. For post-scf plugins.

    >>> energy('dsrg-mrpt2')

    """
    lowername = name.lower()
    kwargs = p4util.kwargs_lower(kwargs)

    # Your plugin's psi4 run sequence goes here
    psi4.set_local_option('LIBADAPTIVE', 'JOB_TYPE','DSRG-MRPT2')
    psi4.plugin('libadaptive.so')
    returnvalue = psi4.get_variable('CURRENT ENERGY')
    return returnvalue

def run_three_dsrg_mrpt2(name, **kwargs):
    r"""Function encoding sequence of PSI module and plugin calls so that
    libadaptive can be called via :py:func:`~driver.energy`. For post-scf plugins.

    >>> energy('three_dsrg-mrpt2')

    """
    lowername = name.lower()
    kwargs = p4util.kwargs_lower(kwargs)

    # Your plugin's psi4 run sequence goes here
    psi4.set_local_option('LIBADAPTIVE', 'JOB_TYPE','THREE_DSRG-MRPT2')
    psi4.plugin('libadaptive.so')
    returnvalue = psi4.get_variable('CURRENT ENERGY')
    return returnvalue

# Integration with driver routines
procedures['energy']['libadaptive'] = run_libadaptive
procedures['energy']['aci'] = run_adaptive_ci
procedures['energy']['ex-aci'] = run_ex_aci
procedures['energy']['apici'] = run_adaptive_pici
procedures['energy']['fapici'] = run_fast_adaptive_pici
procedures['energy']['fciqmc'] = run_fciqmc
procedures['energy']['fno-apifci'] = run_fno_apifci
procedures['energy']['ct'] = run_ct
procedures['energy']['srg'] = run_srg
procedures['energy']['ct-ci'] = run_ct_ci
procedures['energy']['sr-lctsd'] = run_sr_lctsd
procedures['energy']['sr-srgsd'] = run_sr_srgsd
procedures['energy']['sr-dsrgsd'] = run_sr_dsrgsd
procedures['energy']['sr-dsrg-aci'] = run_sr_dsrg_aci
procedures['energy']['mr-dsrg-pt2'] = run_mrdsrgpt2
procedures['energy']['dsrg-mrpt2'] = run_dsrg_mrpt2
procedures['energy']['three-dsrg-mrpt2'] = run_three_dsrg_mrpt2
