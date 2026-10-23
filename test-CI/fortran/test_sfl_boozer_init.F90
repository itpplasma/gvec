! Copyright (c) 2026 GVEC Contributors, Max Planck Institute for Plasma Physics
! License: MIT
PROGRAM test_sfl_boozer_init
  USE, INTRINSIC :: ieee_arithmetic, ONLY: ieee_is_finite
  USE MODgvec_Globals, ONLY: wp, MPIroot, reset_subregion
  USE MODgvec_hmap_frenet, ONLY: t_hmap_frenet, t_hmap_frenet_auxvar
  USE MODgvec_SFL_Boozer, ONLY: t_sfl_boozer, sfl_boozer_init
  IMPLICIT NONE
  TYPE(t_hmap_frenet), TARGET :: hmap
  TYPE(t_sfl_boozer), ALLOCATABLE :: sf
  REAL(wp), PARAMETER :: radius = 3.0_wp, tol = 100*EPSILON(1.0_wp)
  CHARACTER(LEN=8) :: sin_cos = '_sin_'
  REAL(wp) :: zeta, radial(3), tangent(3), expected(3), actual(3)
  INTEGER :: i, cycle

  MPIroot = .TRUE.
  CALL reset_subregion()
  ! A circular guiding curve has an analytic Frenet frame and curvature 1/R.
  hmap = t_hmap_frenet(1, 0, -1, [radius], [0.0_wp], [0.0_wp], [0.0_wp], .FALSE.)
  DO cycle = 1, 2
    ALLOCATE(sf)
    CALL sfl_boozer_init(sf, [1, 1], [3, 5], 1, sin_cos, hmap, 2, &
                         [0.25_wp, 0.75_wp], [0.4_wp, 0.6_wp], [1.0_wp, 2.0_wp])
    IF(.NOT.sf%initialized) ERROR STOP 'Boozer initialization failed'
    IF(.NOT.ASSOCIATED(sf%hmap, hmap)) ERROR STOP 'Boozer map association was lost'
    IF(sf%nrho /= 2) ERROR STOP 'incorrect radial shape'
    IF(ANY(sf%rho_pos /= [0.25_wp, 0.75_wp])) ERROR STOP 'radial positions were lost'
    IF(ANY(sf%iota /= [0.4_wp, 0.6_wp])) ERROR STOP 'iota was lost'
    IF(ANY(sf%phiPrime /= [1.0_wp, 2.0_wp])) ERROR STOP 'flux derivative was lost'
    IF(.NOT.ALLOCATED(sf%hmap_xv)) ERROR STOP 'Boozer auxiliary data missing'
    IF(SIZE(sf%hmap_xv) /= sf%nu_fbase%mn_IP) ERROR STOP 'incorrect auxiliary shape'
    SELECT TYPE(xv => sf%hmap_xv)
    TYPE IS(t_hmap_frenet_auxvar)
      DO i = 1, SIZE(xv)
        zeta = sf%nu_fbase%x_IP(2, i)
        radial = [COS(zeta), SIN(zeta), 0.0_wp]
        tangent = [-SIN(zeta), COS(zeta), 0.0_wp]
        IF(.NOT.xv(i)%do_2nd_der) ERROR STOP 'second derivative request was lost'
        IF(ABS(xv(i)%zeta - zeta) > tol) ERROR STOP 'auxiliary angle was lost'
        IF(.NOT.ALL(ieee_is_finite(xv(i)%X0))) ERROR STOP 'nonfinite guiding curve'
        IF(MAXVAL(ABS(xv(i)%X0 - radius*radial)) > tol) ERROR STOP 'guiding curve was lost'
        IF(MAXVAL(ABS(xv(i)%T - tangent)) > tol) ERROR STOP 'tangent was lost'
        IF(MAXVAL(ABS(xv(i)%N + radial)) > tol) ERROR STOP 'normal was lost'
        IF(MAXVAL(ABS(xv(i)%B - [0.0_wp, 0.0_wp, 1.0_wp])) > tol) ERROR STOP 'binormal was lost'
        IF(ABS(xv(i)%kappa - 1.0_wp/radius) > tol) ERROR STOP 'curvature was lost'
        expected = (radius - 0.2_wp)*radial + [0.0_wp, 0.0_wp, -0.1_wp]
        actual = sf%hmap%eval_aux(0.2_wp, -0.1_wp, xv(i))
        IF(.NOT.ALL(ieee_is_finite(actual))) ERROR STOP 'nonfinite auxiliary map'
        IF(MAXVAL(ABS(actual - expected)) > tol) ERROR STOP 'incorrect auxiliary map'
      END DO
    CLASS DEFAULT
      ERROR STOP 'Boozer auxiliary dynamic type was lost'
    END SELECT
    DEALLOCATE(sf)
  END DO
END PROGRAM test_sfl_boozer_init
