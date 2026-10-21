!> Minimal omp_lib replacement for OpenMP runtimes that ship no Fortran module
!> (e.g. the conda-forge LLVM Flang 23 with libomp). Only the routines GVEC uses.
!> Compiled only if CMake finds no usable omp_lib module (GVEC_OMP_LIB_SHIM).
MODULE omp_lib
  USE, INTRINSIC :: ISO_C_BINDING, ONLY: C_INT, C_DOUBLE
  IMPLICIT NONE
  PUBLIC
  INTERFACE
    FUNCTION omp_get_wtime() BIND(C, NAME="omp_get_wtime")
      IMPORT :: C_DOUBLE
      REAL(C_DOUBLE) :: omp_get_wtime
    END FUNCTION omp_get_wtime
    FUNCTION omp_get_max_threads() BIND(C, NAME="omp_get_max_threads")
      IMPORT :: C_INT
      INTEGER(C_INT) :: omp_get_max_threads
    END FUNCTION omp_get_max_threads
  END INTERFACE
END MODULE omp_lib
