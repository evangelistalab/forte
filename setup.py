from setuptools import find_packages

import os
import re
import sys
import sysconfig
import platform
import subprocess

from distutils.version import LooseVersion
from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext
from shutil import copyfile, copymode

class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=''):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)

class CMakeBuild(build_ext):

    build_ext.user_options = build_ext.user_options + [
        ('ambitpath', None, 'the path to ambit'),
        ('max_det_orb', 64, 'the maximum number of orbitals used by the Determinant class'),
        ]

    def initialize_options(self):
        self.ambitpath = None
        self.max_det_orb = 64
        return build_ext.initialize_options(self)

    def run(self):
        try:
            out = subprocess.check_output(['cmake', '--version'])
        except OSError:
            raise RuntimeError(
                "CMake must be installed to build the following extensions: " +
                ", ".join(e.name for e in self.extensions))

        if platform.system() == "Windows":
            cmake_version = LooseVersion(re.search(r'version\s*([\d.]+)',
                                         out.decode()).group(1))
            if cmake_version < '3.1.0':
                raise RuntimeError("CMake >= 3.1.0 is required on Windows")

        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        import subprocess

        cfg = 'Debug' if self.debug else 'Release'

        print(f'\n  Forte compilation options')
        print(f'\n    BUILD_TYPE = {cfg}')
        print(f'    AMBITPATH = {self.ambitpath}')
        print(f'    MAX_DET_ORB = {self.max_det_orb}\n')

        if 'AMBITPATH' not in os.environ or self.ambitpath == None or self.ambitpath == 'None' or self.ambitpath == '':
            msg = """
    Please specifiy the ambit path. This can be done in two ways:
    1) Set the environmental variable AMBITPATH to the ambit install directory.
    2) Modify the setup.cfg file to include the lines:                
        >[CMakeBuild]
        >ambitpath=<path to ambit install dir>
"""
            raise RuntimeError(msg)
        ambitpath = os.environ['AMBITPATH']

        # grab the cmake configuration from psi4
        process = subprocess.Popen(['psi4', '--plugin-compile'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = process.communicate()
        cmake_args = out.decode("utf-8").split()[1:]

        # append cmake arguments
        cmake_args += [f'-Dambit_DIR={ambitpath}/share/cmake/ambit']
        cmake_args += [f'-DCMAKE_BUILD_TYPE={cfg}']
        cmake_args += [f'-DMAX_DET_ORB={self.max_det_orb}']
        
        cmake_args += [f'-DENABLE_ForteTests=TRUE']

        # define build arguments
        build_args = ['-j2']

        # call cmake and build
        subprocess.check_call(['cmake'] + cmake_args)
        subprocess.check_call(['cmake', '--build', '.'] + build_args)

        print() # Add empty line for nicer output

setup(
    name='forte',
    version='0.2.0',
    author='Forte team',
    description='A hybrid Python/C++ quantum chemistry ipackage for strongly correlated electrons.',
    long_description='Forte is an open-source plugin to Psi4 that implements a variety of quantum chemistry methods for strongly correlated electrons.',
    packages=['forte'],
    # tell setuptools that all packages will be under the '.' directory
    package_dir={'':'.'},
    # add an extension module named 'forte' to the package
    ext_modules=[CMakeExtension('.')],
    # add custom build_ext command
    cmdclass=dict(build_ext=CMakeBuild),
    test_suite='tests',
    zip_safe=False,
)
