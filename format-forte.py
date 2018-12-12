#!/usr/bin/env python

import argparse
import glob

import sys
import re
import subprocess
import os
import datetime

from os import listdir, environ
from os.path import isfile, join

help_text = """A script to format and commit Forte."""

parser = argparse.ArgumentParser(description=help_text)
parser.add_argument('files', nargs='*', help='Files to format', type=str)
parser.add_argument('-a', action="store_true", help='Format all *.cc and *.h files')

args = parser.parse_args()

file_list = []

# a) Default. Format only modified files (according to git status)
git_status = subprocess.check_output(['git','status','--porcelain'])
for line in git_status.split('\n'):
    split_line = line.split()
    if len(split_line) == 2:
        if 'M' in split_line[0]:
            file_list.append(split_line[1])

# b) -a option. Format all files
if args.a:
    ccfiles = glob.glob('src/*.cc')
    hfiles = glob.glob('src/*.h')
    file_list.extend(ccfiles)
    file_list.extend(hfiles)

# c) Format a list of files provided by the user
if len(args.files) > 0:
    file_list = args.files

print "Formatting the following files:\n" + "\n".join(file_list)

command = ['clang-format','-i','-style=file']
command.extend(file_list)
subprocess.call(command)
