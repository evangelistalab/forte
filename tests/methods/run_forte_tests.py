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

fci_tests = ["fci-1","fci-2","fci-3","fci-4","fci-5","fci-6","fci-7","fci-rdms-1","fci-rdms-2","fci-one-electron","fci-ex-1"]

lambda_ci_tests = ["casci-1","casci-2","casci-3","casci-4",
                     "casci-5-fc","casci-6-fc","casci-7-fc","casci-8-fc",
                     "lambda+sd-ci-1","lambda+sd-ci-2"]

adaptive_ci_tests = ["aci-1","aci-2","aci-3",
                     "aci-4","aci-5","aci-6",
                     "aci-7","aci-8","aci-9",
                     "aci-10","aci-11","aci-12",
                     "aci-13","aci-14","aci-15","aci-16","aci-17","aci_scf-1","cis-aci-1",
                     "aci-mrcisd-1","aci-mrcisd-2"]

pci_tests = ["pci-1","pci-2","pci-3","pci-4","pci-5", "pci-6", "pci-7", "pci-8","pci-9"]
pci_hashvec_tests = ["pci_hashvec-1","pci_hashvec-2","pci_hashvec-3","pci_hashvec-4","pci_hashvec-5", "pci_hashvec-6"]
fciqmc_tests = ["fciqmc"]
ct_tests = ["ct-1","ct-2","ct-3","ct-4","ct-5","ct-6","ct-7-fc"]
dsrg_tests = ["dsrg-1","dsrg-2"]
mrdsrg_tests = ["mrdsrg-pt2-1","mrdsrg-pt2-2","mrdsrg-pt2-3","mrdsrg-pt2-4",
                "mrdsrg-ldsrg2-qc-1","mrdsrg-ldsrg2-qc-2",
                "mrdsrg-srgpt2-1","mrdsrg-srgpt2-2"]
mrdsrg_df_tests = ["mrdsrg-ldsrg2-df-seq-1","mrdsrg-ldsrg2-df-seq-2","mrdsrg-ldsrg2-df-seq-3",
                   "mrdsrg-ldsrg2-df-seq-nivo-1","mrdsrg-ldsrg2-df-seq-nivo-2","mrdsrg-ldsrg2-df-seq-nivo-3"]
dsrg_mrpt3_tests = ["dsrg-mrpt3-1","dsrg-mrpt3-2","dsrg-mrpt3-3","dsrg-mrpt3-4","dsrg-mrpt3-5",
                    "dsrg-mrpt3-6-sa","dsrg-mrpt3-7-CO","dsrg-mrpt3-8-sa-C2H4"]
dsrg_mrpt2_tests = ["mr-dsrg-pt2-1","mr-dsrg-pt2-2","mr-dsrg-pt2-3","mr-dsrg-pt2-4",
                    "dsrg-mrpt2-1","dsrg-mrpt2-2","dsrg-mrpt2-3","dsrg-mrpt2-4","dsrg-mrpt2-5",
                    "dsrg-mrpt2-6","dsrg-mrpt2-7-casscf-natorbs","dsrg-mrpt2-8-sa",
                    "dsrg-mrpt2-9-xms","dsrg-mrpt2-10-CO","dsrg-mrpt2-11-sa-C2H4",
                    "dsrg-mrpt2-12-localized-actv",
                    "cd-dsrg-mrpt2-1","cd-dsrg-mrpt2-2","cd-dsrg-mrpt2-3","cd-dsrg-mrpt2-4",
                    "cd-dsrg-mrpt2-5","cd-dsrg-mrpt2-6","cd-dsrg-mrpt2-7-sa",
                    "df-dsrg-mrpt2-1","df-dsrg-mrpt2-2","df-dsrg-mrpt2-3","df-dsrg-mrpt2-4",
                    "df-dsrg-mrpt2-5","df-dsrg-mrpt2-6","df-dsrg-mrpt2-7-localized-actv",
                    "df-dsrg-mrpt2-threading1","df-dsrg-mrpt2-threading2","df-dsrg-mrpt2-threading4",
                    "diskdf-dsrg-mrpt2-threading1","diskdf-dsrg-mrpt2-threading4",
                    "diskdf-dsrg-mrpt2-1","diskdf-dsrg-mrpt2-2","diskdf-dsrg-mrpt2-3",
                    "diskdf-dsrg-mrpt2-4", "diskdf-dsrg-mrpt2-5",
                    "aci-dsrg-mrpt2-1","aci-dsrg-mrpt2-2","df-aci-dsrg-mrpt2-1","df-aci-dsrg-mrpt2-2"]
active_dsrgpt2_tests = ["actv-dsrg-1-C2H4-cis", "actv-dsrg-2-C2H4-cisd", "actv-dsrg-3-C4H6-cisd",
                        "actv-dsrg-4-C4H6-triplet", "actv-dsrg-5-actv-independence",
                        "actv-dsrg-size-intensive", "actv-dsrg-ipea-1", "actv-dsrg-ipea-2"]
dwms_dsrgpt_tests = ["dwms-dsrgpt-1-sa", "dwms-dsrgpt-5-ms", "dwms-dsrgpt-8-separated"]

casscf_tests = ["casscf", "casscf-2","casscf-3", "casscf-4", "casscf-5", "casscf-6", "casscf-7",
                "df-casscf-1"]
dmrg_tests = ["dmrgscf-1", "df-dmrgscf-1", "cd-dmrgscf-1", "dmrg-dsrg-mrpt2-1", "dmrg-dsrg-mrpt2-2"]

ambit_tests = ["ambit-1"]

#tests =  fci_tests + casscf_tests + dsrg_mrpt2_tests + adaptive_ci_tests + pci_tests + fciqmc_tests + ct_tests + dsrg_tests
tests = ambit_tests
tests += fci_tests + casscf_tests + dsrg_mrpt2_tests + dsrg_mrpt3_tests + mrdsrg_tests + mrdsrg_df_tests + adaptive_ci_tests + pci_tests + pci_hashvec_tests
tests += active_dsrgpt2_tests + dwms_dsrgpt2_tests

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
