# Downloaded from
#   https://github.com/coderefinery/autocmake/blob/master/modules/mpi.cmake
# * moved option up

#.rst:
#
# Enables MPI support.
#
# Variables used::
#
#   ENABLE_MPI
#   MPI_FOUND
#
# Variables modified (provided the corresponding language is enabled)::
#
#   CMAKE_Fortran_FLAGS
#   CMAKE_C_FLAGS
#   CMAKE_CXX_FLAGS
#
# autocmake.yml configuration::
#
#   docopt: "--mpi Enable MPI parallelization [default: False]."
#   define: "'-DENABLE_MPI={0}'.format(arguments['--mpi'])"
#
#option(ENABLE_MPI "Enable MPI parallelization" OFF)

# on Cray configure with -D MPI_FOUND=1
if(ENABLE_MPI AND NOT MPI_FOUND)
    find_package(MPI)
    if(MPI_FOUND)
        if(DEFINED CMAKE_Fortran_COMPILER_ID)
            set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} ${MPI_COMPILE_FLAGS}")
        endif()
        if(DEFINED CMAKE_C_COMPILER_ID)
            set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${MPI_COMPILE_FLAGS}")
        endif()
        if(DEFINED CMAKE_CXX_COMPILER_ID)
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${MPI_COMPILE_FLAGS}")
        endif()
        include_directories(${MPI_INCLUDE_PATH})

        add_definitions(-DHAVE_MPI)
    else()
        message(FATAL_ERROR "-- You asked for MPI, but CMake could not find any MPI installation, check $PATH")
    endif()
endif()
