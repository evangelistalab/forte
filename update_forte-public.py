#!/usr/bin/env python
# vim:ft=python

import os
import os.path
import sys
import string
import re
import subprocess
import shutil
import datetime
import time

private_source_dir = '.'
public_source_dir = '../forte-public'

copy_dirs = ['.','src']
copy_suffixes = ['.cc','.h']

public_tag  = '//[forte-public]'
private_tag = '//[forte-private]'

def process_file(dir,file):
    # Open the file
    lines = open(os.path.join(private_source_dir,dir,file)).readlines()
    public_file = []
    public = False
    for line in lines:
        if public_tag in line:
            public = True
        elif private_tag in line:
            public = False
        elif public:
            public_file.append(line)
            
    if len(public_file) > 0:
        print "Adding file %s" % os.path.join(public_source_dir,dir,file)
        new_file = open(os.path.join(public_source_dir,dir,file),'w+')
        new_file.write("".join(public_file))

def copy_files():
    for dir in copy_dirs:
        dest_dir = os.path.join(public_source_dir,dir)
        if not os.path.isdir(dest_dir):
            os.mkdir(dest_dir)
        dir_path = private_source_dir + '/' + dir
        for file in os.listdir(dir_path):
            file_suffix = os.path.splitext(file)[1]
            if file_suffix in copy_suffixes:
                process_file(dir,file)

def remove_old_content():
    if os.path.isdir(public_source_dir):
        shutil.rmtree(public_source_dir)
    os.mkdir(public_source_dir)

def main(argv):
    remove_old_content()
    copy_files()

if __name__ == '__main__':
    main(sys.argv)
