#.rst:
# FindGlobalArrays
# ----------------
#
# Find the native GlobalArrays includes and libraries.
#
# Imported Targets
# ^^^^^^^^^^^^^^^^
#
# If GA is found, this module defines the following :prop_tgt:`IMPORTED`
# targets::
#
#  GlobalArrays::ga      - The main GA library.
#  GA::armrrays::ga ci   - The ARMCI support library used by GA.
#
# Result Variables
# ^^^^^^^^^^^^^^^^
#
# This module will set the following variables in your project::
#
#  GlobalArrays_FOUND          - True if GA found on the local system
#  GlobalArrays_INCLUDE_DIRS   - Location of GA header files.
#  GlobalArrays_LIBRARIES      - The GA libraries.
#
# Hints
# ^^^^^
#
# Set ``GA_ROOT_DIR`` to a directory that contains a GA installation.
#
# This script expects to find libraries at ``$GA_ROOT_DIR/lib`` and the GA
# headers at ``$GA_ROOT_DIR/include/ga/src``.
#
# Cache Variables
# ^^^^^^^^^^^^^^^
#
# This module may set the following variables depending on platform and type
# of GA installation discovered.  These variables may optionally be set to
# help this module find the correct files::
#
#  ARMCI_LIBRARY       - Location of the ARMCI library.
#  GA_LIBRARY          - Location of the GA library.
#

include(${CMAKE_CURRENT_LIST_DIR}/FindPackageHandleStandardArgs.cmake)
include (GNUInstallDirs)

#=============================================================================
# If the user has provided ``GA_ROOT_DIR``, use it!  Choose items found
# at this location over system locations.
if( EXISTS "$ENV{GA_ROOT_DIR}" )
  file( TO_CMAKE_PATH "$ENV{GA_ROOT_DIR}" GA_ROOT_DIR )
  set( GA_ROOT_DIR "${GA_ROOT_DIR}" CACHE PATH "Prefix for GlobalArrays installation." )
endif()

#=============================================================================
# Set GA_INCLUDE_DIRS and GA_LIBRARIES. Try
# to find the libraries at $GA_ROOT_DIR (if provided) or in standard system
# locations.  These find_library and find_path calls will prefer custom
# locations over standard locations (HINTS).  If the requested file is not found
# at the HINTS location, standard system locations will be still be searched
# (/usr/lib64 (Redhat), lib/i386-linux-gnu (Debian)).

find_path( GA_INCLUDE_DIR
  NAMES global/src/ga.h
  HINTS ${GA_ROOT_DIR}
  PATH_SUFFIXES include ${CMAKE_INSTALL_INCLUDEDIR})
find_library( GA_LIBRARY
  NAMES ga
  HINTS ${GA_ROOT_DIR}
  PATH_SUFFIXES lib lib64 ${CMAKE_INSTALL_LIBDIR})
find_library( ARMCI_LIBRARY
  NAMES armci
  HINTS ${GA_ROOT_DIR}
  PATH_SUFFIXES lib lib64 ${CMAKE_INSTALL_LIBDIR})
set( GlobalArrays_INCLUDE_DIRS ${GA_INCLUDE_DIR} )
set( GlobalArrays_LIBRARIES ${GA_LIBRARY} ${ARMCI_LIBRARY} )

#=============================================================================
# handle the QUIETLY and REQUIRED arguments and set GA_FOUND to TRUE if all
# listed variables are TRUE
find_package_handle_standard_args( GlobalArrays
  FOUND_VAR
    GSAL_FOUND
  REQUIRED_VARS
    GlobalArrays_INCLUDE_DIR
    GlobalArrays_LIBRARY
    ARMCI_LIBRARY
    )

mark_as_advanced( GA_ROOT_DIR GA_LIBRARY GA_INCLUDE_DIR
  ARMCI_LIBRARY )

#=============================================================================

if( GlobalArrays_FOUND AND NOT TARGET GlobalArrays::ga )
    add_library( GlobalArrays::ga    UNKNOWN IMPORTED )
    add_library( GlobalArrays::armci UNKNOWN IMPORTED )
    set_target_properties( GlobalArrays::armci PROPERTIES
      IMPORTED_LOCATION                 "${ARMCI_LIBRARY}"
      INTERFACE_INCLUDE_DIRECTORIES     "${GA_INCLUDE_DIR}"
      IMPORTED_LINK_INTERFACE_LANGUAGES "CXX" )
    set_target_properties( GlobalArrays::ga PROPERTIES
      IMPORTED_LOCATION                 "${GA_LIBRARY}"
      INTERFACE_INCLUDE_DIRECTORIES     "${GA_INCLUDE_DIR}"
      IMPORTED_LINK_INTERFACE_LANGUAGES "C"
      INTERFACE_COMPILE_DEFINITIONS     "HAVE_GA"
      INTERFACE_LINK_LIBRARIES          GlobalArrays::armci )
endif()

