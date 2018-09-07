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
timing_re = re.compile(r"Psi4 exiting successfully. Buy a developer a beer!")

psi4command = ""


print("Running forte tests using the psi4 executable found in:\n  %s\n" % psi4command)

adaptive_ci_tests = ["aci-1"]
tests = adaptive_ci_tests

maindir = os.getcwd()
if len(sys.argv) == 1:
    cmd = ["which","psi4"]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    res = p.stdout.readlines()
    if len(res) == 0:
        print("Could not detect your PSI4 executable.  Please specify its location.")
        exit(1)
    psi4command = res[0][:-1]
elif len(sys.argv) == 2:
    psi4command = sys.argv[1]
#elif len(sys.argv) == 3:
#    tests = sys.argv[2]

print("Running forte tests using the psi4 executable found in:\n  %s\n" % psi4command)


test_results = {}
for d in tests:
    print("Running test %s" % d.upper())

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
        print(out)
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

print("Summary:")
print(" " * 8 + "-" * 72)
print("\n".join(summary))
print(" " * 8 + "-" * 72)

test_result_log = open("test_results","w+")
test_result_log.write("\n".join(summary))

nfailed = len(failed)
nnomatch = len(nomatch)
if nnomatch + nfailed == 0:
    print("Tests: All passed\n")
else:
    print("Tests: %d passed, %d failed, %d did not match\n" % (len(tests) -  nnomatch - nfailed,nfailed,nnomatch))
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
