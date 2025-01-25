if (POLICY CMP0025)
  cmake_policy(SET CMP0025 NEW)
endif ()

cmake_policy(SET CMP0048 NEW)  # project_VERSION* variables populated from project(... VERSION x.x.x) string

project(forte
        VERSION 0.4.3
        LANGUAGES CXX C)
set(forte_AUTHORS      "Francesco A. Evangelista and Group")
set(forte_DESCRIPTION  "Quantum Chemistry Methods for Strongly Correlated Electrons plugin to Psi4")
set(forte_URL          "https://github.com/evangelistalab/forte")
set(forte_LICENSE      "GPL-3.0+")
set(TargetOpenMP_FIND_COMPONENTS "CXX")

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

include(psi4OptionsTools)

option_with_print(ENABLE_CheMPS2 "Enable CheMPS2 solver" OFF)
option_with_print(ENABLE_block2 "Enable block2 solver" OFF)
option_with_print(ENABLE_OPENMP "Enable threadsafe linking to OpenMP parallelized programs" ON)
option_with_print(ENABLE_MPI "Enable MPI parallelization" OFF)
option_with_print(ENABLE_GA "Enable Global Arrays" OFF)
option_with_print(MAX_DET_ORB "Set the maximum number of orbitals in a determinant" OFF)
option_with_print(ENABLE_CODECOV "Enable compilation with code coverage flags" OFF)
option_with_print(ENABLE_UNTESTED_CODE "Enable code not covered by code coverage" OFF)
option_with_default(FORTE_INSTALL_PYMODDIR "Location within CMAKE_INSTALL_PREFIX to which the Python module is installed. Empty string queries Python interpreter. Don't start with '/'" prefix)
# for trial builds, it's worth using `set(CMAKE_INTERPROCEDURAL_OPTIMIZATION OFF)` or passing it as a `-D`. Saves 75% of build time to not run LTO (added by pybind11) at the link stage.

include(autocmake_omp)  # no longer useful, probably need to copy psi4/external/common/lapack to cmake
include(autocmake_mpi)  # MPI option A

find_package(psi4 1.4 REQUIRED)
find_package(pybind11 CONFIG REQUIRED)
find_package(Python COMPONENTS Interpreter Development REQUIRED)
find_package(TargetLAPACK REQUIRED)
find_package(ambit 0.6 REQUIRED)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(DMRG_SOLVER "")
if(ENABLE_CheMPS2)
    find_package(CheMPS2 1.8.3 CONFIG REQUIRED)
    add_definitions(-DHAVE_CHEMPS2)
    set(HAVE_CHEMPS2 true)
endif()

if(ENABLE_block2)
    find_package(block2 2.0 CONFIG REQUIRED)
    add_definitions(-DHAVE_BLOCK2)
    set(HAVE_BLOCK2 true)
endif()

set(Boost_USE_STATIC_LIBS OFF)
set(Boost_USE_MULTITHREADED ON)
set(Boost_USE_STATIC_RUNTIME OFF)
find_package(Boost REQUIRED)

if(Boost_FOUND)
    include_directories(${Boost_INCLUDE_DIRS})
endif()

# Including custom cmake rules
include(cmake/clang-cxx-dev-tools.cmake)
include(cmake/git-version.cmake)

add_compile_options(-Wall -Wextra -pedantic)

if(ENABLE_CODECOV)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --coverage")
endif()

if(ENABLE_UNTESTED_CODE)
    add_definitions(-DENABLE_UNTESTED_CODE)
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-c++1z-extensions") # avoid warnings for C++17

# List of CC files
pybind11_add_module(_forte
api/active_space_integrals_api.cc
api/active_space_method_api.cc
api/active_space_solver_api.cc
api/ci_rdms_api.cc
api/configuration_api.cc
api/cube_file_api.cc
api/determinant_api.cc
api/dl_solver_api.cc
api/forte_python_module.cc
api/genci_api.cc
api/integrals_api.cc
api/mcscf_api.cc
api/mo_space_info_api.cc
api/ndarray_api.cc
api/rdms_api.cc
api/options_api.cc
api/orbital_api.cc
api/rdms_api.cc
api/scf_info_api.cc
api/sparse_ci_solver_api.cc
api/sparse_exp_api.cc
api/sparse_fact_exp_api.cc
api/sparse_operator_api.cc
api/sparse_operator_list_api.cc
api/sparse_operator_sim_trans_api.cc
api/sparse_state_api.cc
api/sq_operator_string_api.cc
api/state_info_api.cc
api/string_api.cc
api/symmetry_api.cc
base_classes/active_space_method.cc
base_classes/active_space_solver.cc
base_classes/dynamic_correlation_solver.cc
base_classes/forte_options.cc
base_classes/mo_space_info.cc
base_classes/orbital_transform.cc
base_classes/orbitals.cc
base_classes/rdms.cc
base_classes/scf_info.cc
base_classes/state_info.cc
ci_ex_states/excited_state_solver.cc
ci_rdm/ci_rdms.cc
ci_rdm/ci_rdms_dynamic.cc
ci_rdm/ci_rdms_sf.cc
dmrg/block2_dmrg_solver.cc
dmrg/dmrgsolver.cc
external/external_active_space_method.cc
fci/fci_solver.cc
fci/fci_solver_initial_guess.cc
fci/fci_solver_rdms.cc
fci/fci_solver_transition_rdms.cc
fci/fci_string_address.cc
fci/fci_string_address.cc
fci/fci_string_hole_list.cc
fci/fci_string_lists.cc
fci/fci_string_oo_list.cc
fci/fci_string_vo_list.cc
fci/fci_string_vvoo_list.cc
fci/fci_vector.cc
fci/fci_vector_h_diag.cc
fci/fci_vector_hamiltonian.cc
fci/fci_vector_rdm.cc
fci/fci_vector_spin2.cc
fci/fci_vector_transition_rdm.cc
forte.cc
genci/ci_occupation.cc
genci/genci_solver.cc
genci/genci_solver_initial_guess.cc
genci/genci_solver_rdms.cc
genci/genci_solver_transition_rdms.cc
genci/genci_string_address.cc
genci/genci_string_address.cc
genci/genci_string_lists.cc
genci/genci_vector.cc
genci/genci_vector_h_diag.cc
genci/genci_vector_hamiltonian.cc
genci/genci_vector_rdm.cc
genci/genci_vector_spin2.cc
genci/genci_vector_transition_rdm.cc
genci/string_lists_makers.cc
gradient_tpdm/backtransform_tpdm.cc
gradient_tpdm/integraltransform_tpdm_restricted.cc
gradient_tpdm/integraltransform_tpdm_unrestricted.cc
helpers/blockedtensorfactory.cc
helpers/combinatorial.cc
helpers/cube_file.cc
helpers/davidson_liu_solver.cc
helpers/determinant_helpers.cc
helpers/disk_io.cc
helpers/helpers.cc
helpers/lbfgs/lbfgs.cc
helpers/lbfgs/lbfgs_param.cc
helpers/lbfgs/rosenbrock.cc
helpers/printing.cc
helpers/spinorbital_helpers.cc
helpers/string_algorithms.cc
helpers/symmetry.cc
helpers/threading.cc
integrals/active_space_integrals.cc
integrals/cholesky_integrals.cc
integrals/conventional_integrals.cc
integrals/custom_integrals.cc
integrals/df_integrals.cc
integrals/diskdf_integrals.cc
integrals/distribute_df_integrals.cc
integrals/integrals.cc
integrals/make_integrals.cc
integrals/one_body_integrals.cc
integrals/parallel_ccvv_algorithms.cc
integrals/paralleldfmo.cc
integrals/psi4_integrals.cc
mcscf/cpscf.cc
mcscf/mcscf_2step.cc
mcscf/mcscf_orb_grad.cc
mcscf/mcscf_orb_grad_deriv.cc
mrdsrg-helper/dsrg_mem.cc
mrdsrg-helper/dsrg_source.cc
mrdsrg-helper/dsrg_time.cc
mrdsrg-helper/dsrg_transformed.cc
mrdsrg-helper/run_dsrg.cc
mrdsrg-so/mrdsrg_so.cc
mrdsrg-spin-adapted/dsrg_mrpt.cc
mrdsrg-spin-adapted/dsrg_mrpt_2nd.cc
mrdsrg-spin-adapted/dsrg_mrpt_amp.cc
mrdsrg-spin-adapted/dsrg_mrpt_comm.cc
mrdsrg-spin-adapted/sa_dsrgpt.cc
mrdsrg-spin-adapted/sa_ldsrg2.cc
mrdsrg-spin-adapted/sa_mrdsrg.cc
mrdsrg-spin-adapted/sa_mrdsrg_amps.cc
mrdsrg-spin-adapted/sa_mrdsrg_diis.cc
mrdsrg-spin-adapted/sa_mrpt2.cc
mrdsrg-spin-adapted/sa_mrpt2_oeprop.cc
mrdsrg-spin-adapted/sa_mrpt3.cc
mrdsrg-spin-adapted/sadsrg.cc
mrdsrg-spin-adapted/sadsrg_amps_analysis.cc
mrdsrg-spin-adapted/sadsrg_block_labels.cc
mrdsrg-spin-adapted/sadsrg_comm.cc
mrdsrg-spin-integrated/dsrg_mrpt2.cc
mrdsrg-spin-integrated/dsrg_mrpt2_grad/dsrg_mrpt2_deriv_multipliers.cc
mrdsrg-spin-integrated/dsrg_mrpt2_grad/dsrg_mrpt2_deriv_setb.cc
mrdsrg-spin-integrated/dsrg_mrpt2_grad/dsrg_mrpt2_deriv_solve_zvector.cc
mrdsrg-spin-integrated/dsrg_mrpt2_grad/dsrg_mrpt2_deriv_tensors.cc
mrdsrg-spin-integrated/dsrg_mrpt2_grad/dsrg_mrpt2_deriv_write_rdms.cc
mrdsrg-spin-integrated/dsrg_mrpt2_grad/dsrg_mrpt2_gradient.cc
mrdsrg-spin-integrated/dsrg_mrpt2_ms.cc
mrdsrg-spin-integrated/dsrg_mrpt3.cc
mrdsrg-spin-integrated/master_mrdsrg.cc
mrdsrg-spin-integrated/mrdsrg.cc
mrdsrg-spin-integrated/mrdsrg_amplitude.cc
mrdsrg-spin-integrated/mrdsrg_commutator.cc
mrdsrg-spin-integrated/mrdsrg_diis.cc
mrdsrg-spin-integrated/mrdsrg_nonpt.cc
mrdsrg-spin-integrated/mrdsrg_pt.cc
mrdsrg-spin-integrated/mrdsrg_smart_s.cc
mrdsrg-spin-integrated/mrdsrg_srg.cc
mrdsrg-spin-integrated/three_dsrg_mrpt2.cc
orbital-helpers/ao_helper.cc
orbital-helpers/aosubspace.cc
orbital-helpers/ci-no/ci-no.cc
orbital-helpers/ci-no/mrci-no.cc
orbital-helpers/fragment_projector.cc
orbital-helpers/iao_builder.cc
orbital-helpers/localize.cc
orbital-helpers/mp2_nos.cc
orbital-helpers/mrpt2_nos.cc
orbital-helpers/orbital_embedding.cc
orbital-helpers/orbitaloptimizer.cc
orbital-helpers/pao_builder.cc
orbital-helpers/semi_canonicalize.cc
orbital-helpers/unpaired_density.cc
pci/pci.cc
pci/pci_sigma.cc
post_process/spin_corr.cc
sci/aci.cc
sci/aci_build_F.cc
sci/asci.cc
sci/detci.cc
sci/gasaci_build_F.cc
sci/mrpt2.cc
sci/sci.cc
sci/tdci.cc
sparse_ci/ci_reference.cc
sparse_ci/ci_spin_adaptation.cc
sparse_ci/determinant_functions.cc
sparse_ci/determinant_hashvector.cc
sparse_ci/determinant_substitution_lists.cc
sparse_ci/sigma_vector.cc
sparse_ci/sigma_vector_dynamic.cc
sparse_ci/sigma_vector_full.cc
sparse_ci/sigma_vector_sparse_list.cc
sparse_ci/sorted_string_list.cc
sparse_ci/sparse_ci_solver.cc
sparse_ci/sparse_exp.cc
sparse_ci/sparse_fact_exp.cc
sparse_ci/sparse_initial_guess.cc
sparse_ci/sparse_operator.cc
sparse_ci/sparse_operator_sim_trans.cc
sparse_ci/sparse_operator_hamiltonian.cc
sparse_ci/sparse_state.cc
sparse_ci/sparse_state_functions.cc
sparse_ci/sq_operator_string.cc
sparse_ci/sq_operator_string_ops.cc
v2rdm/v2rdm.cc
)

target_include_directories(
  _forte
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_BINARY_DIR}
  )

target_link_libraries(_forte PRIVATE tgt::MathOpenMP)
target_link_libraries(_forte PRIVATE psi4::core)
target_link_libraries(_forte PRIVATE ambit::ambit)

if(TARGET CheMPS2::chemps2)
    target_link_libraries(_forte PRIVATE CheMPS2::chemps2)
    add_definitions(-DHAVE_CHEMPS2)
endif()

if(TARGET block2::block2)
    target_link_libraries(_forte PRIVATE block2::block2)
    add_definitions(-DHAVE_BLOCK2)
    add_definitions(-D_USE_GLOBAL_VARIABLE -D_EXPLICIT_TEMPLATE -D_LARGE_BOND -D_USE_CORE -D_USE_SU2SZ -D_USE_DMRG)
endif()

if(ENABLE_MPI)
    target_link_libraries(_forte PRIVATE ${MPI_CXX_LIBRARIES})  # MPI option A
    #target_link_libraries(forte PRIVATE MPI::MPI_CXX)  # MPI option B
    #target_add_definitions(forte PRIVATE HAVE_MPI)  # MPI option B
endif()

if(ENABLE_GA)
    target_link_libraries(_forte PRIVATE GlobalArrays::ga)
endif()


if (FORTE_INSTALL_PYMODDIR STREQUAL "prefix")

    # Note that this block is *Linux-style* install to `CMAKE_INSTALL_PREFIX` not *Python-style* install to `Python_EXECUTABLE`'s site-packages.
    execute_process(
      COMMAND ${Python_EXECUTABLE} -c
        "import os, sys, sysconfig as s; spdir = s.get_path('platlib'); print(spdir.replace(os.path.commonpath([sys.prefix, spdir]), '').lstrip(os.path.sep));"
      OUTPUT_VARIABLE FORTE_INSTALL_PYMODDIR
      OUTPUT_STRIP_TRAILING_WHITESPACE
      )
endif()
message(STATUS "Showing option CMAKE_INSTALL_PREFIX/FORTE_INSTALL_PYMODDIR: ${CMAKE_INSTALL_PREFIX}/${FORTE_INSTALL_PYMODDIR}")

install(
  FILES
    $<TARGET_FILE:_forte>
    ../pytest.ini
  COMPONENT Forte_Python
  DESTINATION ${FORTE_INSTALL_PYMODDIR}/forte
  )

install(
  DIRECTORY
    "${CMAKE_CURRENT_SOURCE_DIR}/"
  COMPONENT Forte_Python
  DESTINATION ${FORTE_INSTALL_PYMODDIR}/forte
  FILES_MATCHING
    PATTERN "*.py"
    PATTERN "__pycache__" EXCLUDE
    PATTERN "*pyc" EXCLUDE
    # installs harmless empty dirs for c++ unless excluded below
    PATTERN "api" EXCLUDE
    PATTERN "attic" EXCLUDE
    PATTERN "base_classes" EXCLUDE
    PATTERN "casscf" EXCLUDE
    PATTERN "ci_ex_states" EXCLUDE
    PATTERN "ci_rdm" EXCLUDE
    PATTERN "cmake" EXCLUDE
    PATTERN "dmrg" EXCLUDE
    PATTERN "external" EXCLUDE
    PATTERN "fci" EXCLUDE
    PATTERN "genci" EXCLUDE
    PATTERN "gradient_tpdm" EXCLUDE
    PATTERN "helpers" EXCLUDE
    PATTERN "integrals" EXCLUDE
    PATTERN "lib" EXCLUDE
    PATTERN "mrdsrg-helper" EXCLUDE
    PATTERN "mrdsrg-so" EXCLUDE
    PATTERN "mrdsrg-spin-adapted" EXCLUDE
    PATTERN "mrdsrg-spin-integrated" EXCLUDE
    PATTERN "orbital-helpers" EXCLUDE
    PATTERN "pci" EXCLUDE
    PATTERN "post_process" EXCLUDE
    PATTERN "sci" EXCLUDE
    PATTERN "sparse_ci" EXCLUDE
    PATTERN "v2rdm" EXCLUDE
  )

install(
  FILES
    ../conftest.py
  COMPONENT Forte_Python
  DESTINATION ${FORTE_INSTALL_PYMODDIR}/forte/tests
  )

install(
  DIRECTORY
    "${CMAKE_CURRENT_SOURCE_DIR}/../tests/pytest/"
    "${CMAKE_CURRENT_SOURCE_DIR}/../tests/pytest-methods/"
  DESTINATION ${FORTE_INSTALL_PYMODDIR}/forte/tests
  FILES_MATCHING
    PATTERN "*.py"
    PATTERN "*.xyz"
    PATTERN "INTDUMP*"
    PATTERN "__pycache__" EXCLUDE
    PATTERN "*pyc" EXCLUDE
  )
