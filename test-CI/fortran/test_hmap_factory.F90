! Copyright (c) 2026 GVEC Contributors, Max Planck Institute for Plasma Physics
! License: MIT
PROGRAM test_hmap_factory
  USE MODgvec_Globals, ONLY: wp, MPIroot, reset_subregion
  USE MODgvec_c_hmap, ONLY: c_hmap, c_hmap_auxvar
  USE MODgvec_hmap_RZ, ONLY: t_hmap_RZ, t_hmap_RZ_auxvar
  USE MODgvec_hmap_cyl, ONLY: t_hmap_cyl
  USE MODgvec_hmap, ONLY: hmap_new, hmap_new_auxvar
  IMPLICIT NONE
  CLASS(c_hmap), ALLOCATABLE :: hmap, copy
  CLASS(c_hmap_auxvar), ALLOCATABLE :: aux(:)
  REAL(wp), PARAMETER :: zeta(2) = [0.0_wp, 0.5_wp]
  INTEGER :: i

  MPIroot = .TRUE.
  CALL reset_subregion()
  DO i = 1, 3
    ! Exercise an unallocated destination, reuse, and a change of dynamic type.
    IF(i == 3) THEN
      DEALLOCATE(hmap)
      ALLOCATE(t_hmap_cyl :: hmap)
    END IF
    CALL hmap_new(hmap, 1)
    SELECT TYPE(hmap)
    TYPE IS(t_hmap_RZ)
      IF(.NOT.hmap%initialized) ERROR STOP 'RZ constructor state was lost'
    CLASS DEFAULT
      ERROR STOP 'hmap_new lost the constructor dynamic type'
    END SELECT
    IF(hmap%which_hmap /= 1) ERROR STOP 'incorrect mapping number'
  END DO

  CALL hmap_new(copy, 1, hmap_in=hmap)
  SELECT TYPE(copy)
  TYPE IS(t_hmap_RZ)
    IF(.NOT.copy%initialized) ERROR STOP 'RZ copy state was lost'
  CLASS DEFAULT
    ERROR STOP 'hmap_new lost the copied dynamic type'
  END SELECT
  CALL hmap_new_auxvar(copy, zeta, aux, .TRUE.)
  SELECT TYPE(aux)
  TYPE IS(t_hmap_RZ_auxvar)
    IF(SIZE(aux) /= SIZE(zeta)) ERROR STOP 'incorrect auxiliary shape'
    IF(ANY(aux%zeta /= zeta)) ERROR STOP 'incorrect auxiliary coordinates'
    IF(.NOT.ALL(aux%do_2nd_der)) ERROR STOP 'incorrect derivative request'
  CLASS DEFAULT
    ERROR STOP 'incorrect auxiliary dynamic type'
  END SELECT
  IF(ANY(ABS(copy%eval([2.0_wp, 1.0_wp, 0.0_wp]) - &
             [2.0_wp, 0.0_wp, 1.0_wp]) > 10*EPSILON(1.0_wp))) &
    ERROR STOP 'incorrect cylindrical-to-Cartesian mapping'
  DEALLOCATE(aux, copy, hmap)
END PROGRAM test_hmap_factory
