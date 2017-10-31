cmake_policy(SET CMP0048 NEW)  # project_VERSION* variables populated from project(... VERSION x.x.x) string
project(forte
        VERSION 0.1
        LANGUAGES CXX C)
set(forte_AUTHORS      "Francesco A. Evangelista and Group")
set(forte_DESCRIPTION  "Quantum Chemistry Methods for Strongly Correlated Electrons plugin to Psi4")
set(forte_URL          "https://github.com/evangelistalab/forte")
set(forte_LICENSE      "GPL-3.0+")

cmake_minimum_required(VERSION 3.1 FATAL_ERROR)
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

include(psi4OptionsTools)

option_with_print(ENABLE_CheMPS2 "Enable CheMPS2 solver." OFF)
option_with_print(ENABLE_OPENMP "Enable threadsafe linking to OpenMP parallelized programs." ON)
option_with_print(ENABLE_MPI "Enable MPI parallelization" OFF)
option_with_print(ENABLE_GA "Enable Global Arrays" OFF)
#option_with_print(ENABLE_GENERIC "Enable mostly static linking in shared library" OFF)

include(autocmake_omp)
include(autocmake_mpi)  # MPI option A
#include(custom_static_library)


#find_package(psi4 1.1 REQUIRED COMPONENTS ambit chemps2)
find_package(psi4 1.0 REQUIRED)

find_package(TargetLAPACK REQUIRED)
find_package(ambit REQUIRED)
if(ENABLE_CheMPS2)
    find_package(CheMPS2 1.8.3 CONFIG REQUIRED)
endif()

# Including custom cmake rules
include(cmake/clang-cxx-dev-tools.cmake)
include(cmake/git-version.cmake)

# List of CC files
add_psi4_plugin(forte
src/active_dsrgpt2.cc
src/active_space_solver.cc
src/ao_helper.cc
src/avas.cc
src/bitset_performance.cc
src/blockedtensorfactory.cc
src/casscf.cc
src/cc.cc
src/ci_rdms.cc
src/ci_reference.cc
src/davidson.cc
src/determinant_map.cc
src/determinant_hashvector.cc
src/dmrgscf.cc
src/dmrgsolver.cc
src/dynamic_bitset_determinant.cc
src/dynamic_correlation.cc
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
src/sparse_ci/sparse_ci_solver.cc
src/sparse_ci/sigma_vector.cc
src/sparse_ci/stl_bitset_determinant.cc
src/sparse_ci/stl_bitset_string.cc
src/sq.cc
src/tensorsrg.cc
src/tensorsrg_antihermitian_commutators.cc
src/tensorsrg_ct.cc
src/tensorsrg_full_commutator.cc
src/tensorsrg_hermitian_commutator.cc
src/tensorsrg_save.cc
src/tensorsrg_srg.cc
src/v2rdm.cc
src/aci/aci.cc
src/aosubspace/aosubspace.cc
src/ci-no/ci-no.cc
src/fci/binary_graph.cc
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
src/integrals/integrals.cc
src/integrals/own_integrals.cc
src/integrals/parallel_ccvv_algorithms.cc
src/integrals/paralleldfmo.cc
src/mrdsrg-helper/dsrg_source.cc
src/mrdsrg-helper/dsrg_time.cc
src/mrdsrg-helper/run_dsrg.cc
src/mrdsrg-so/mrdsrg_so.cc
src/mrdsrg-so/so-mrdsrg.cc
src/mrdsrg-spin-adapted/dsrg_mrpt.cc
src/mrdsrg-spin-adapted/dsrg_mrpt_2nd.cc
src/mrdsrg-spin-adapted/dsrg_mrpt_amp.cc
src/mrdsrg-spin-adapted/dsrg_mrpt_comm.cc
src/mrdsrg-spin-integrated/master_mrdsrg.cc
src/mrdsrg-spin-integrated/dsrg_mrpt2.cc
src/mrdsrg-spin-integrated/dsrg_mrpt2_ms.cc
src/mrdsrg-spin-integrated/dsrg_mrpt3.cc
src/mrdsrg-spin-integrated/mrdsrg.cc
src/mrdsrg-spin-integrated/mrdsrg_amplitude.cc
src/mrdsrg-spin-integrated/mrdsrg_commutator.cc
src/mrdsrg-spin-integrated/mrdsrg_nonpt.cc
src/mrdsrg-spin-integrated/mrdsrg_pt.cc
src/mrdsrg-spin-integrated/mrdsrg_smart_s.cc
src/mrdsrg-spin-integrated/mrdsrg_srg.cc
src/mrdsrg-spin-integrated/three_dsrg_mrpt2.cc
src/orbital-helper/localize.cc
src/orbital-helper/unpaired_density.cc
src/orbital-helper/iao_builder.cc
src/orbital-helper/es-nos.cc
src/pci/pci.cc
src/pci/pci_hashvec.cc
src/pci/pci_simple.cc
src/pci/ewci.cc
src/libthce/thce.cc
src/libthce/lreri.cc
)

target_link_libraries(forte PRIVATE ambit::ambit)
if(TARGET CheMPS2::chemps2)
    target_link_libraries(forte PRIVATE CheMPS2::chemps2)
endif()

target_include_directories(forte PRIVATE src/mini-boost)

if(ENABLE_MPI)
    target_link_libraries(forte PRIVATE ${MPI_CXX_LIBRARIES})  # MPI option A
    #target_link_libraries(forte PRIVATE MPI::MPI_CXX)  # MPI option B
    #target_add_definitions(forte PRIVATE HAVE_MPI)  # MPI option B
endif()

if(ENABLE_GA)
    target_link_libraries(forte PRIVATE GlobalArrays::ga)
endif()
