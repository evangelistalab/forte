# Forte method tests

To run the test suite that checks the implementation of all Forte methods use the python script `run_forte_tests.py` found in this directory.

```
> python run_forte_tests.py

Running forte tests using the psi4 executable found in:
  /Users/fevange/Source/psi4/objdir-Debug-llvm/stage/bin/psi4

Test group ACI
    Running test ACI-1
	ACI energy........................................................PASSED
	ACI+PT2 energy....................................................PASSED
...
```    

## Test file
The list of all tests is found in the file `tests.yaml`. This file looks like this:
```
aci:
  short:
   - aci-1
   ...
   - aci_scf-1
  long:
   - aci-6
   - cis-aci-1
  unused:
   - aci-mrcisd-1
   - aci-mrcisd-2
actv-dsrg:
  short:
   ...
```
The first level in this list indicates the test group (aci, actv-dsrg). At the second level, the tests are divided into different types according to duration and use (short, long, unused).

## Selecting different test types/groups
By default `run_forte_tests.py` runs only the tests in the group **short**. To run all tests call
```
run_forte_tests.py --type all
```
It is also possible to run tests in a certan group by passing the option `--group`. For example, to run only the full CI tests call
```
run_forte_tests.py --group fci
```

## Executing only the test the previously failed
After running the tests, the test script will write out a list of all the test that failed in the file `failed_tests.yaml`. To run only the tests contained in this file call `run_forte_tests.py` with the option `--failed`.

## Test script options
There are other optional arguments that can be used to modify the behavior of the test script. To list them just run `python run_forte_tests.py --help`. Here is the output of this command:
```
> python run_forte_tests.py --help

usage: run_forte_tests.py [-h] [--psi4_exec PSI4_EXEC] [--file FILE]
                          [--failed] [--bw] [--failed_dump]
                          [--type {all,short}] [--group GROUP]

Run Forte tests.

optional arguments:
  -h, --help            show this help message and exit
  --psi4_exec PSI4_EXEC
                        the location of the psi4 executable
  --file FILE           the yaml file containing the list of tests (default:
                        tests.yaml)
  --failed              run only failed tests (listed in the file
                        failed_tests)
  --bw                  print the summary in black and white? (default: color)
  --failed_dump         dump the output of the failed tests to stdout?
  --type {all,short}    which type of test to run? (default: short)
  --group GROUP         which group of tests to run? (default: all)
```
