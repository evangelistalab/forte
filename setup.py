from setuptools import find_packages

import os
import psutil
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
        ('enable_codecov', False, 'enable code coverage'),
        ('cmake_config_options', '', 'cmake configuration'),
        ('cmake_build_options', '', 'cmake build options')
        ]

    def initialize_options(self):
        self.ambitpath = None
        self.max_det_orb = 64
        self.enable_codecov = 'OFF'
        self.cmake_config_options = ''
        ncore_phys = psutil.cpu_count(logical=False)
        ncore_virt = psutil.cpu_count(logical=True)
        nprocs = ncore_phys + 1 if ncore_phys != ncore_virt else ncore_phys
        self.cmake_build_options = f'-j{nprocs}'
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
        print(f'    MAX_DET_ORB = {self.max_det_orb}')
        print(f'    ENABLE_CODECOV = {str(self.enable_codecov).upper()}')
        print(f'    CMAKE_CONFIG_OPTIONS = {self.cmake_config_options}')
        print(f'    CMAKE_BUILD_OPTIONS = {self.cmake_build_options}\n')

        if 'AMBITPATH' in os.environ:
            self.ambitpath = os.environ['AMBITPATH']

        if self.ambitpath in [None, 'None', '']:
            msg = """
    Please specifiy a correct ambit path. This can be done in two ways:
    1) Set the environmental variable AMBITPATH to the ambit install directory.
    2) Modify the setup.cfg file to include the lines:
        >[CMakeBuild]
        >ambitpath=<path to ambit install dir>
"""
            raise RuntimeError(msg)

        # grab the cmake configuration from psi4
        process = subprocess.Popen(['psi4', '--plugin-compile'],
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = process.communicate()
        cmake_args = out.decode("utf-8").split()[1:]

        # append cmake arguments
        cmake_args += [f'-Dambit_DIR={self.ambitpath}/share/cmake/ambit']
        cmake_args += [f'-DCMAKE_BUILD_TYPE={cfg}']
        cmake_args += [f'-DMAX_DET_ORB={self.max_det_orb}']
        cmake_args += [f'-DENABLE_CODECOV={str(self.enable_codecov).upper()}']
        cmake_args += [f'-DENABLE_ForteTests=TRUE']
        cmake_args += self.cmake_config_options.split()

        # define build arguments
        build_args = self.cmake_build_options.split()

        # call cmake and build
        subprocess.check_call(['cmake'] + cmake_args)
        subprocess.check_call(['cmake', '--build', '.'] + build_args)

        print() # Add empty line for nicer output

setup(
    name='forte',
    version='0.2.0',
    author='Forte team',
    description='A hybrid Python/C++ quantum chemistry package for strongly correlated electrons.',
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
