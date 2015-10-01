#ifndef _forte_def_h_
#define _forte_def_h_

// Define OPENMP functions in case they are not available
#ifdef _OPENMP
    #include <omp.h>
#else
    #define omp_get_max_threads() 1
    #define omp_get_thread_num()  0
#endif

#endif // _forte_def_h_
