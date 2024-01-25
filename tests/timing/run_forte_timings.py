#!/usr/bin/env python

import sys
import os
import subprocess
import datetime
import re
import string

# Define tests here
fci_tests = ["fci-1", "fci-6", "fci-2_2", "fci-4_4", "fci-6_6", "fci-8_8", "fci-10_10", "fci-12_12", "fci-14_14"]
gasci_tests = [
    "gasci-1",
    "gasci-6",
    "gasci-2_2",
    "gasci-4_4",
    "gasci-6_6",
    "gasci-8_8",
    "gasci-10_10",
    "gasci-12_12",
    "gasci-14_14",
]
tests = fci_tests + gasci_tests


class bcolors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"


timing_re = re.compile(r"Psi4 wall time for execution: (\d+:\d+:\d+.\d+)")
error_re = re.compile(r"TestComparisonError")

psi4command = ""

if len(sys.argv) == 1:
    cmd = ["which", "psi4"]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    res = p.stdout.readlines()
    if len(res) == 0:
        print("Could not detect your PSI4 executable.  Please specify its location.")
        exit(1)
    psi4command = res[0][:-1]
elif len(sys.argv) == 2:
    psi4command = sys.argv[1]

print("Running test using psi4 executable found in:\n%s" % psi4command)

maindir = os.getcwd()

# Run and collect timings from output files
print("\nRun and collect timings from output files:")
timing_info = []
for d in tests:
    os.chdir(d)
    print("Running test %-s" % (d.upper(),))
    subprocess.call([psi4command])
    timing = open("output.dat").read()
    m = timing_re.search(timing)
    m2 = error_re.search(timing)
    message = ""
    if m2:
        if m:
            timing_info.append((d, m.groups()[0]))
            message = bcolors.FAIL + "FAILED" + bcolors.ENDC + " took " + timing_info[-1][1]
        else:
            message = bcolors.FAIL + "FAILED" + bcolors.ENDC
    else:
        if m:
            timing_info.append((d, m.groups()[0]))
            message = bcolors.OKGREEN + "PASSED" + bcolors.ENDC + " took " + timing_info[-1][1]
        else:
            message = bcolors.FAIL + "FAILED" + bcolors.ENDC + " took " + timing_info[-1][1]
    filler = " " * (66 - len(d))
    print("        %-s%s%s" % (d.upper(), filler, message))
    os.chdir(maindir)

# Get the current date and time
dt = datetime.datetime.now()
now = dt.strftime("%Y-%m-%d-%H:%M")
output = open("timings-%s.txt" % now, "w+")

# Collect all the timings
print("\nTimings:")
for nt in timing_info:
    name = nt[0]
    timing = nt[1]
    filler = " " * (64 - len(name))
    str = "%-s%s%s" % (name.upper(), filler, timing)
    output.write(str + "\n")
    print("        " + str)
