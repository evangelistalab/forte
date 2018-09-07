#!/usr/bin/env python

import sys
import os
import subprocess
import re
import datetime
import time


class bcolors:
    HEADER = ''
    OKBLUE = ''
    OKGREEN = ''
    WARNING = ''
    FAIL = ''
    ENDC = ''

timing_re = re.compile(r"Your calculation took (\d+.\d+) seconds")
timing_re = re.compile(r"Psi4 exiting successfully. Buy a developer a beer!")

psi4command = ""


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
test_time = {}
for d in tests:
    print("Running test %s" % d.upper())

    start = time.time()
    os.chdir(d)
    successful = True
    # Run psi
    try:
        out = subprocess.check_output([psi4command])
    except:
        # something went wrong
        successful = False
        test_results[d] = "FAILED"

    if successful:
        # Check if FORTE ended successfully
        timing = open("output.dat").read()
        m = timing_re.search(timing)
        if m:
            test_results[d] = "PASSED"
        else:
            test_results[d] = "FAILED"
        print(out.decode("utf-8"))
    os.chdir(maindir)
    end = time.time()
    test_time[d] = end - start

total_time = 0.0
summary = []
failed = []
for d in tests:
    if test_results[d] == "PASSED":
        msg = bcolors.OKGREEN + "PASSED" + bcolors.ENDC
    elif test_results[d] == "FAILED":
        msg = bcolors.FAIL + "FAILED" + bcolors.ENDC
        failed.append(d)
    duration = test_time[d]
    total_time += duration
    filler = "." * max(0,67 - len(d + msg))
    summary.append("    %s%s%s  %5.1f" % (d.upper(),filler,msg,duration))

print("Summary:")
print(" " * 4 + "=" * 76)
print("    TEST" + ' ' * 57 + 'RESULT TIME (s)')
print(" " * 4 + "-" * 76)
print("\n".join(summary))
print(" " * 4 + "=" * 76)

test_result_log = open("test_results","w+")
test_result_log.write("\n".join(summary))

nfailed = len(failed)
if nfailed == 0:
    print("Tests: All passed\n")
else:
    print("Tests: %d passed and %d failed\n" % (len(tests) -  nfailed,nfailed))
    # Get the current date and time
    dt = datetime.datetime.now()
    now = dt.strftime("%Y-%m-%d-%H:%M")
   
    print("The following tests failed:")
    for failed_test in failed:
        print("  %s" % failed_test)
    
print("\nTotal time: %6.1f s\n" % total_time)
if nfailed > 0:
    failed_log = open("failed_tests","w")
    failed_log.write("# %s\n" % now)
    failed_log.write("\n".join(failed))
    failed_log.close()
    exit(1)
