import os
import sys

if len(sys.argv) == 2:
    n = 0
    new_file = ''
    str = ''
    for l in open(sys.argv[1],'r').readlines():
        if (len(l.strip()) > 0):
            str += l.rstrip()
            print("str[-1]",str[-1])
            if (str[-1] == ';'):
                numspaces = len(str) - len(str.lstrip(' '))
                space = ' ' * numspaces
                new_file += '{}timer t{};\n'.format(space,n)
                new_file += str
                new_file += '\n{}t{}.stop();\n\n'.format(space,n)
                n += 1
                str = ''
    print(new_file)
