#
# @BEGIN LICENSE
#
# forte_inversion by Psi4 Developer, a plugin to:
#
# Psi4: an open-source quantum chemistry software package
#
# Copyright (c) 2007-2016 The Psi4 Developers.
#
# The copyrights for code used from other parties are included in
# the corresponding files.
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
#
# @END LICENSE
#

cmake_minimum_required(VERSION 3.0 FATAL_ERROR)

# These were the compilers used to compile Psi4. They may or may not be
# required to successfully run your plugin.
set(CMAKE_C_COMPILER /usr/bin/clang)
set(CMAKE_CXX_COMPILER /usr/bin/clang++)
set(CMAKE_Fortran_COMPILER )

project(forte_inversion CXX)

find_package(psi4 1.0 REQUIRED)

link_directories(../ambit-bin-debug/lib)

include_directories(../ambit-bin-debug/include/)
include_directories(src/mini-boost)


add_psi4_plugin(forte
src/aci.cc src/dsrg_wick.cc src/orbitaloptimizer.cc
src/aci_string.cc src/dynamic_bitset_determinant.cc src/own_integrals.cc
src/active_dsrgpt2.cc src/effective_integrals.cc src/parallel_ccvv_algorithms.cc
src/ao_helper.cc src/fci_mo.cc src/paralleldfmo.cc
src/aosubspace.cc src/fci_solver.cc src/pci.cc
src/bitset_csf.cc src/fci_vector.cc src/reference.cc
src/bitset_performance.cc src/fci_vector_h_diag.cc src/sa_fcisolver.cc
src/blockedtensorfactory.cc src/fci_vector_hamiltonian.cc src/so-mrdsrg.cc
src/casscf.cc src/fci_vector_rdm.cc src/sparse_ci_solver.cc
src/cholesky_integrals.cc src/fcimc.cc src/sparse_ci_wfn.cc
src/ci_rdms.cc src/finite_temperature.cc src/sq.cc
src/conventional_integrals.cc src/helpers.cc src/static_linked_hash.cc
src/davidson.cc src/integrals.cc src/stl_bitset_determinant.cc
src/determinant_map.cc src/iterative_solvers.cc src/stl_bitset_string.cc
src/df_integrals.cc src/localize.cc src/string_hole_list.cc
src/direct_ci.cc src/main.cc src/string_lists.cc
src/diskdf_integrals.cc src/mcsrgpt2_mo.cc src/string_oo_list.cc
src/distribute_df_integrals.cc src/methodbase.cc src/string_vo_list.cc
src/dmrgscf.cc src/mp2_nos.cc src/string_vvoo_list.cc
src/dmrgsolver.cc src/mrdsrg.cc src/tensorsrg.cc
src/dsrg_mrpt.cc src/mrdsrg_amplitude.cc src/tensorsrg_antihermitian_commutators.cc
src/dsrg_mrpt2.cc src/mrdsrg_commutator.cc src/tensorsrg_ct.cc
src/dsrg_mrpt2_ms.cc src/mrdsrg_nonpt.cc src/tensorsrg_full_commutator.cc
src/dsrg_mrpt3.cc src/mrdsrg_pt.cc src/tensorsrg_hermitian_commutator.cc
src/dsrg_mrpt_2nd.cc src/mrdsrg_smart_s.cc src/tensorsrg_save.cc
src/dsrg_mrpt_amp.cc src/mrdsrg_so.cc src/tensorsrg_srg.cc
src/dsrg_mrpt_comm.cc src/mrdsrg_srg.cc src/three_dsrg_mrpt2.cc
src/dsrg_source.cc src/operator.cc src/v2rdm.cc
src/dsrg_time.cc src/options.cc)

target_link_libraries(forte ambit) 
