#!/usr/bin/env python

import os
import subprocess

psi4command = "/Users/francesco/Source/psi4-github-compile-c++11-debug/bin/psi4"

print "Running test using psi4 executable found in:\n%s" % psi4command

adaptive_ci_tests = ["casci-1","casci-2","casci-3","casci-4","lambda+sd-ci-1","lambda+sd-ci-2"]
ct_tests = ["ct-1","ct-2","ct-3","ct-4","ct-5","ct-6"]
srg_tests = ["srg-1","srg-2"]
dsrg_tests = ["dsrg-1","dsrg-2","mr-dsrg-pt2-1"]

tests = adaptive_ci_tests + ct_tests + srg_tests + dsrg_tests
maindir = os.getcwd()
for d in tests:
    print "\nRunning test %s" % d
    os.chdir(d)
    subprocess.call([psi4command])
    os.chdir(maindir)
