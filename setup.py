import os
import re
import platform
import subprocess

from distutils.version import LooseVersion
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext


class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=''):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):

    build_ext.user_options = build_ext.user_options + [
        # Notes: the first option is the option string
        #        the second option is an abbreviated form of an option, which we avoid with None
        ('ambitpath', None, 'the path to ambit'),
        ('max-det-orb', None, 'the maximum number of orbitals used by the Determinant class'),
        ('enable-codecov', None, 'enable code coverage'),
        ('cmake-config-options', None, 'cmake configuration'),
        ('cmake-build-options', None, 'cmake build options'),
        ('nprocs', None, 'number of threads used to compile Forte')
    ]

    def initialize_options(self):
        self.ambitpath = None
        self.max_det_orb = 64
        self.enable_codecov = 'OFF'
        self.cmake_config_options = ''
        self.cmake_build_options = ''
        self.nprocs = None
        return build_ext.initialize_options(self)

    def run(self):
        try:
            out = subprocess.check_output(['cmake', '--version'])
        except OSError:
            raise RuntimeError(
                "CMake must be installed to build the following extensions: " +
                ", ".join(e.name for e in self.extensions)
            )

        if platform.system() == "Windows":
            cmake_version = LooseVersion(re.search(r'version\s*([\d.]+)', out.decode()).group(1))
            if cmake_version < '3.1.0':
                raise RuntimeError("CMake >= 3.1.0 is required on Windows")

        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        import subprocess

        cfg = 'Debug' if self.debug else 'Release'

        # if nprocs is not specified we use os.cpu_count() to find the CPU count
        if self.nprocs in [None, 'None', '']:
            # None is returned if the number of CPUs is undetermined
            cpu_count = os.cpu_count()
            # use the maximum number of processors
            self.nprocs = 1 if cpu_count is None else cpu_count

        print('\n  Forte compilation options')
        print(f'\n    BUILD_TYPE = {cfg}')
        print(f'    AMBITPATH = {self.ambitpath}')
        print(f'    MAX_DET_ORB = {self.max_det_orb}')
        print(f'    ENABLE_CODECOV = {str(self.enable_codecov).upper()}')
        print(f'    CMAKE_CONFIG_OPTIONS = {self.cmake_config_options}')
        print(f'    CMAKE_BUILD_OPTIONS = {self.cmake_build_options}')
        print(f'    NPROCS = {self.nprocs}\n')

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
        process = subprocess.Popen(['psi4', '--plugin-compile'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
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
        build_args.append(f'-j{self.nprocs}')

        # call cmake and build
        subprocess.check_call(['cmake'] + cmake_args)
        subprocess.check_call(['cmake', '--build', '.'] + build_args)

        print()  # Add empty line for nicer output


setup(
    name='forte',
    version='0.2.3',
    author='Forte developers',
    description='A hybrid Python/C++ quantum chemistry package for strongly correlated electrons.',
    long_description=
    'Forte is an open-source plugin to Psi4 that implements a variety of quantum chemistry methods for strongly correlated electrons.',
    packages=['forte'],
    # tell setuptools that all packages will be under the '.' directory
    package_dir={'': '.'},
    # add an extension module named 'forte' to the package
    ext_modules=[CMakeExtension('forte')],
    # add custom build_ext command
    cmdclass=dict(build_ext=CMakeBuild),
    test_suite='tests',
    zip_safe=False
)
