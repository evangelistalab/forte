#
# @BEGIN LICENSE
#
# forte_template by Psi4 Developer, a plugin to:
#
# Psi4: an open-source quantum chemistry software package
#
# Copyright (c) 2007-2017 The Psi4 Developers.
#
# The copyrights for code used from other parties are included in
# the corresponding files.
#
# This file is part of Psi4.
#
# Psi4 is free software; you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, version 3.
#
# Psi4 is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License along
# with Psi4; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
#
# @END LICENSE
#

cmake_minimum_required(VERSION 3.1 FATAL_ERROR)

project(forte_template CXX)

find_package(psi4 1.0 REQUIRED)

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

link_directories(/Users/jeffschriber/src/ambit/install/lib)
include_directories(/Users/jeffschriber/src/ambit/install/include)
include_directories(src/mini-boost)

add_psi4_plugin(forte
src/active_dsrgpt2.cc
src/ao_helper.cc
src/avas.cc
src/bitset_performance.cc
src/blockedtensorfactory.cc
src/casscf.cc
src/cc.cc
src/ci_rdms.cc
src/davidson.cc
src/determinant_map.cc
src/dmrgscf.cc
src/dmrgsolver.cc
src/dynamic_bitset_determinant.cc
src/fci_mo.cc
src/fcimc.cc
src/finite_temperature.cc
src/forte_old_methods.cc
src/forte_old_options.cc
src/forte_options.cc
src/helpers.cc
src/iterative_solvers.cc
src/main.cc
src/mcsrgpt2_mo.cc
src/methodbase.cc
src/mp2_nos.cc
src/mrci.cc
src/mrpt2.cc
src/operator.cc
src/options.cc
src/orbitaloptimizer.cc
src/reference.cc
src/sa_fcisolver.cc
src/semi_canonicalize.cc
src/sparse_ci_solver.cc
src/sq.cc
src/stl_bitset_determinant.cc
src/stl_bitset_string.cc
src/tensorsrg.cc
src/tensorsrg_antihermitian_commutators.cc
src/tensorsrg_ct.cc
src/tensorsrg_full_commutator.cc
src/tensorsrg_hermitian_commutator.cc
src/tensorsrg_save.cc
src/tensorsrg_srg.cc
src/v2rdm.cc
src/aci/aci.cc
src/aci/aci_string.cc
src/aci/direct_ci.cc
src/aosubspace/aosubspace.cc
src/ci-no/ci-no.cc
src/fci/fci.cc
src/fci/fci_integrals.cc
src/fci/fci_solver.cc
src/fci/fci_vector.cc
src/fci/fci_vector_h_diag.cc
src/fci/fci_vector_hamiltonian.cc
src/fci/fci_vector_rdm.cc
src/fci/string_hole_list.cc
src/fci/string_lists.cc
src/fci/string_oo_list.cc
src/fci/string_vo_list.cc
src/fci/string_vvoo_list.cc
src/integrals/cholesky_integrals.cc
src/integrals/conventional_integrals.cc
src/integrals/df_integrals.cc
src/integrals/diskdf_integrals.cc
src/integrals/distribute_df_integrals.cc
src/integrals/effective_integrals.cc
src/integrals/integrals.cc
src/integrals/own_integrals.cc
src/integrals/parallel_ccvv_algorithms.cc
src/integrals/paralleldfmo.cc
src/layers/nn.cc
src/layers/nn_optimizer.cc
src/mrdsrg-helper/dsrg_source.cc
src/mrdsrg-helper/dsrg_time.cc
src/mrdsrg-so/mrdsrg_so.cc
src/mrdsrg-so/so-mrdsrg.cc
src/mrdsrg-spin-adapted/dsrg_mrpt.cc
src/mrdsrg-spin-adapted/dsrg_mrpt_2nd.cc
src/mrdsrg-spin-adapted/dsrg_mrpt_amp.cc
src/mrdsrg-spin-adapted/dsrg_mrpt_comm.cc
src/mrdsrg-spin-free/dsrg_mrpt2.cc
src/mrdsrg-spin-free/dsrg_mrpt2_ms.cc
src/mrdsrg-spin-free/dsrg_mrpt3.cc
src/mrdsrg-spin-free/mrdsrg.cc
src/mrdsrg-spin-free/mrdsrg_amplitude.cc
src/mrdsrg-spin-free/mrdsrg_commutator.cc
src/mrdsrg-spin-free/mrdsrg_nonpt.cc
src/mrdsrg-spin-free/mrdsrg_pt.cc
src/mrdsrg-spin-free/mrdsrg_smart_s.cc
src/mrdsrg-spin-free/mrdsrg_srg.cc
src/mrdsrg-spin-free/three_dsrg_mrpt2.cc
src/orbital-helper/localize.cc
src/orbital-helper/unpaired_density.cc
src/pci/ewci.cc
src/pci/pci.cc
src/pci/pci_hashvec.cc
src/pci/pci_simple.cc
)
    target_link_libraries(forte PRIVATE ambit)
if (MPI_FOUND)
    target_link_libraries(forte PRIVATE ${MPI_LIBRARIES})
endif()
