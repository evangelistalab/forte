#!/usr/bin/env python

import argparse
import datetime
import os
import shutil
import subprocess
import sys
import re
import time
import yaml

MAINDIR = os.getcwd()

TIMING_RE = re.compile(r'Psi4 exiting successfully. Buy a developer a beer!')

TEST_LEVELS = {
    'short': ['short'],
    'medium': ['medium'],
    'long': ['long'],
    'standard': ['short', 'medium'],
    'all': ['short', 'medium', 'long']
}


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'


def run_job(jobdir, psi4command, test_results, test_time):
    """Run a test in jobdir using the psi4command"""
    start = time.time()
    os.chdir(jobdir)
    successful = True
    # Run Psi4
    try:
        out = subprocess.check_output([psi4command, "-n2"])
    except:
        # something went wrong
        successful = False
        test_results[jobdir] = 'FAILED'

    # check if Forte ended successfully
    if successful:
        timing = open('output.dat').read()
        m = TIMING_RE.search(timing)
        if m:
            test_results[jobdir] = 'PASSED'
        else:
            test_results[jobdir] = 'FAILED'
            successful = False
        print(out.decode('utf-8'))
    os.chdir(MAINDIR)
    end = time.time()
    test_time[jobdir] = end - start
    return successful


def prepare_summary(jobdir, test_results, test_time, summary, color):
    """Append the result of a computation to a summary"""
    if test_results[jobdir] == 'PASSED':
        if color:
            msg = bcolors.OKGREEN + 'PASSED' + bcolors.ENDC
        else:
            msg = 'PASSED'
    elif test_results[jobdir] == 'FAILED':
        if color:
            msg = bcolors.FAIL + 'FAILED' + bcolors.ENDC
        else:
            msg = 'FAILED'
    duration = test_time[jobdir]
    if color:
        filler = '.' * max(0, 76 - len(jobdir + msg))
    else:
        filler = '.' * max(0, 67 - len(jobdir + msg))
    summary.append('    {0}{1}{2}{3:7.1f}'.format(jobdir.upper(), filler, msg,
                                                  duration))
    return duration


def setup_argument_parser():
    """Setup an ArgumentParser object to deal with user input."""
    parser = argparse.ArgumentParser(description='Run Forte tests.')
    parser.add_argument('--psi4_exec',
                        help='the location of the psi4 executable')
    parser.add_argument('--file',
                        help='the yaml file containing the list of tests (default: tests.yaml)',
                        default='tests.yaml')
    parser.add_argument('--failed',
                        help='run only failed tests (listed in the file failed_tests)',
                        action='store_true')
    parser.add_argument('--bw',
                        help='print the summary in black and white? (default: color)',
                        action='store_true')
    parser.add_argument('--failed_dump',
                        help='dump the output of the failed tests to stdout?',
                        action='store_true')
    parser.add_argument('--type',
                        help='which type of test to run? (default: standard)',
                        choices={'short', 'medium','long','standard', 'all'},
                        default='standard')
    parser.add_argument('--group',
                        help='which group of tests to run? (default: None)',
                        default=None)
    return parser.parse_args()


def find_psi4(args):
    """Find the psi4 executable or use value provided by the user."""
    psi4command = None
    # if not provided try to detect psi4
    if args.psi4_exec == None:
        psi4command = shutil.which('psi4')
    else:
        psi4command = args.psi4_exec

    if psi4command == None:
        print(
            'Could not detect your PSI4 executable.  Please specify its location.'
        )
        exit(1)

    return psi4command


def main():
    psi4command = ''
    total_time = 0.0
    summary = []
    test_results = {}
    test_time = {}
    failed_tests = {}

    args = setup_argument_parser()
    psi4command = find_psi4(args)

    print('Running forte tests using the psi4 executable found in:\n  %s\n' %
          psi4command)

    # default is to run tests listed in tests.yaml
    test_dict_file = args.file

    # optionally, run only tests that previously failed
    if args.failed:
        print('Running only failed tests')
        test_dict_file = 'failed_tests.yaml'
    # read the yaml file
    with open(test_dict_file, 'rt') as infile:
        test_dict = yaml.load(infile, Loader=yaml.FullLoader)

    tested_groups = test_dict.keys()
    if args.group != None:
        tested_groups = [args.group]

    ntests = 0
    nfailed = 0
    # loop over group tests
    for test_group, test_levels in test_dict.items():
        if test_group in tested_groups:
            print('Test group {}'.format(test_group.upper()))
            group_failed_tests = {}  # test that failed in this group
            for test_level, tests in test_levels.items():
                local_failed_tests = []
                if test_level in TEST_LEVELS[args.type]:
                    for test in tests:
                        print('    Running test {}'.format(test.upper()))
                        successful = run_job(test, psi4command, test_results,
                                             test_time)
                        if not successful:
                            local_failed_tests.append(test)
                            nfailed += 1
                        total_time += prepare_summary(test, test_results,
                                                      test_time, summary,
                                                      not args.bw)
                    ntests += len(tests)
                    if len(local_failed_tests) > 0:
                        group_failed_tests[test_level] = local_failed_tests
            if len(group_failed_tests) > 0:
                failed_tests[test_group] = group_failed_tests

    # print a summary of the tests
    summary_str = 'Summary:\n'
    summary_str += ' ' * 4 + '=' * 76 + '\n'
    summary_str += '    TEST' + ' ' * 57 + 'RESULT TIME (s)\n'
    summary_str += ' ' * 4 + '-' * 76 + '\n'
    summary_str += '\n'.join(summary) + '\n'
    summary_str += ' ' * 4 + '=' * 76

    print(summary_str)
    print('\nTotal time: %6.1f s\n' % total_time)

    import datetime
    now = datetime.datetime.now()
    file_name = 'test_results_%s.txt' % now.strftime("%Y-%m-%d-%H%M")

    with open(file_name, 'w') as outfile:
        outfile.write(summary_str)
        outfile.write('\nTotal time: %6.1f s\n' % total_time)

    # save the list of failed tests
    with open('failed_tests.yaml', 'w') as outfile:
        yaml.dump(failed_tests, outfile, default_flow_style=False)

    if nfailed == 0:
        print('Tests: All passed ({} tests)\n'.format(ntests))
    else:
        print('Tests: {} passed and {} failed\n'.format(
            ntests - nfailed, nfailed))
        # Get the current date and time
        dt = datetime.datetime.now()
        now = dt.strftime('%Y-%m-%d-%H:%M')

        print('The following tests failed:')
        for test_group, test_levels in failed_tests.items():
            print('Test group {}'.format(test_group.upper()))
            for test_level, tests in test_levels.items():
                for test in tests:
                    print('    {}'.format(test.upper()))

        if args.failed_dump:
            for test_group, test_levels in failed_tests.items():
                for test_level, tests in test_levels.items():
                    for test in tests:
                        print('\n\n==> %s TEST OUTPUT <==\n' % test.upper())
                        subprocess.call('cat %s/output.dat' % test, shell=True)
                        print('\n')
        exit(1)


if __name__ == '__main__':
    main()
