#!/usr/bin/env python

import sys
import os
import subprocess
import re
import datetime

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'

timing_re = re.compile(r"Your calculation took (\d+.\d+) seconds")

psi4command = ""


print "Running forte tests using the psi4 executable found in:\n  %s\n" % psi4command

fci_tests = ["fci-1","fci-2","fci-3","fci-4","fci-5","fci-6","fci-rdms-1","fci-rdms-2","fci-ex-1"]

lambda_ci_tests = ["casci-1","casci-2","casci-3","casci-4",
                     "casci-5-fc","casci-6-fc","casci-7-fc","casci-8-fc",
                     "lambda+sd-ci-1","lambda+sd-ci-2"]

adaptive_ci_tests = ["aci-1","aci-2","aci-3",
                     "aci-4","aci-5","aci-6",
                     "aci-7","aci-8","aci-9",
                     "aci-10","aci-11","aci-12"]

apifci_tests = ["apifci-1","apifci-2","apifci-3","apifci-4","apifci-5", "apifci-6", "apifci-7"]
fciqmc_tests = ["fciqmc"]
ct_tests = ["ct-1","ct-2","ct-3","ct-4","ct-5","ct-6","ct-7-fc"]
dsrg_tests = ["dsrg-1","dsrg-2"]
dsrg_mrpt2_tests = ["mr-dsrg-pt2-1","dsrg-mrpt2-1","dsrg-mrpt2-2","dsrg-mrpt2-3","dsrg-mrpt2-4", "dsrg-mrpt2-5",
                    "cd-dsrg-mrpt2-1","cd-dsrg-mrpt2-2","cd-dsrg-mrpt2-3","cd-dsrg-mrpt2-4", "cd-dsrg-mrpt2-5",
                    "df-dsrg-mrpt2-1", "df-dsrg-mrpt2-2", "df-dsrg-mrpt2-3", "df-dsrg-mrpt2-4", "df-dsrg-mrpt2-5",
                    "df-dsrg-mrpt2-threading1", "df-dsrg-mrpt2-threading2", "df-dsrg-mrpt2-threading4",
                    "diskdf-dsrg-mrpt2-1", "diskdf-dsrg-mrpt2-2", "diskdf-dsrg-mrpt2-3", "diskdf-dsrg-mrpt2-4", "diskdf-dsrg-mrpt2-5"]
casscf_tests = ["casscf", "casscf-2", "casscf-4", "casscf-5", "casscf-6"]

#tests =  fci_tests + casscf_tests + dsrg_mrpt2_tests + adaptive_ci_tests + apifci_tests + fciqmc_tests + ct_tests + dsrg_tests
tests =  fci_tests + casscf_tests + dsrg_mrpt2_tests + adaptive_ci_tests + apifci_tests 
maindir = os.getcwd()
if len(sys.argv) == 1:
    cmd = ["which","psi4"]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    res = p.stdout.readlines()
    if len(res) == 0:
        print "Could not detect your PSI4 executable.  Please specify its location."
        exit(1)
    psi4command = res[0][:-1]
elif len(sys.argv) == 2:
    psi4command = sys.argv[1]
#elif len(sys.argv) == 3:
#    tests = sys.argv[2]

print "Running forte tests using the psi4 executable found in:\n  %s\n" % psi4command


test_results = {}
for d in tests:
    print "Running test %s" % d.upper()

    os.chdir(d)
    successful = True
    # Run psi
    try:
        out = subprocess.check_output([psi4command])
    except:
        # something went wrong
        successful = False
        test_results[d] = "DOES NOT MATCH"

    if successful:
        # Check if FORTE ended successfully
        timing = open("output.dat").read()
        m = timing_re.search(timing)
        if m:
            test_results[d] = "PASSED"
        else:
            test_results[d] = "FAILED"
        print out
    os.chdir(maindir)


summary = []
failed = []
nomatch = []
for d in tests:
    if test_results[d] == "PASSED":
        msg = bcolors.OKGREEN + "PASSED" + bcolors.ENDC
    elif test_results[d] == "FAILED":
        msg = bcolors.FAIL + "FAILED" + bcolors.ENDC
        failed.append(d)
    elif test_results[d] == "DOES NOT MATCH":
        msg = bcolors.FAIL + "DOES NOT MATCH" + bcolors.ENDC
        nomatch.append(d)

    filler = "." * (81 - len(d + msg))
    summary.append("        %s%s%s" % (d.upper(),filler,msg))

print "Summary:"
print " " * 8 + "-" * 72
print "\n".join(summary)
print " " * 8 + "-" * 72

nfailed = len(failed)
nnomatch = len(nomatch)
if nnomatch + nfailed == 0:
    print "Tests: All passed\n"
else:
    print "Tests: %d passed, %d failed, %d did not match\n" % (len(tests) -  nnomatch - nfailed,nfailed,nnomatch)
    # Get the current date and time
    dt = datetime.datetime.now()
    now = dt.strftime("%Y-%m-%d-%H:%M")
    if nfailed > 0:
        failed_log = open("failed_tests","w+")
        failed_log.write("# %s\n" % now)
        failed_log.write("\n".join(failed))
    if nnomatch > 0:
        nomatch_log = open("nomatch_tests","w+")
        nomatch_log.write("# %s\n" % now)
        nomatch_log.write("\n".join(nomatch))
