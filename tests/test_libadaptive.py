#!/usr/bin/env python

import sys
import os
import subprocess

psi4command = ""

if len(sys.argv) == 1:
    cmd = ["which","psi4"]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    res = p.stdout.readlines()
    psi4command = res[0][:-1]
elif len(sys.argv) == 2:
    psi4command = sys.argv[1]

print "Running test using psi4 executable found in:\n%s" % psi4command

fci_tests = ["fci-1","fci-2","fci-3","fci-4","fci-rdms-1","fci-rdms-2"]

lambda_ci_tests = ["casci-1","casci-2","casci-3","casci-4",
                     "casci-5-fc","casci-6-fc","casci-7-fc","casci-8-fc",
                     "lambda+sd-ci-1","lambda+sd-ci-2"]

adaptive_ci_tests = ["adaptive-ci-1","adaptive-ci-2","adaptive-ci-3",
                     "adaptive-ci-4","adaptive-ci-5","adaptive-ci-6",
                     "adaptive-ci-7","adaptive-ci-8", "ex-aci-1", "ex-aci-2", "ex-aci-3","ex-aci-4"]

apifci_tests = ["apifci-1","apifci-2","apifci-3"]

ct_tests = ["ct-1","ct-2","ct-3","ct-4","ct-5","ct-6","ct-7-fc"]
srg_tests = [] #["srg-1","srg-2"]
dsrg_tests = ["dsrg-1","dsrg-2"]
dsrg_mrpt2_tests = ["mr-dsrg-pt2-1","dsrg-mrpt2-1","dsrg-mrpt2-2","dsrg-mrpt2-3","dsrg-mrpt2-4",
                    "cd-dsrg-mrpt2-1","cd-dsrg-mrpt2-2","cd-dsrg-mrpt2-3","cd-dsrg-mrpt2-4", "cd-dsrg-mrpt2-5",
                    "dsrg-mrpt2-mp2-no", "df-dsrg-mrpt2-1", "df-dsrg-mrpt2-2", "df-dsrg-mrpt2-3", "df-dsrg-mrpt2-4", "df-dsrg-mrpt2-5",
                    "diskdf-dsrg-mrpt2-1", "diskdf-dsrg-mrpt2-2", "diskdf-dsrg-mrpt2-3", "diskdf-dsrg-mrpt2-4", "diskdf-dsrg-mrpt2-5"]

tests =  fci_tests + dsrg_mrpt2_tests + adaptive_ci_tests + apifci_tests + ct_tests + srg_tests + dsrg_tests
maindir = os.getcwd()
for d in tests:
    print "\nRunning test %s" % d
    os.chdir(d)
    subprocess.call([psi4command])
    os.chdir(maindir)
