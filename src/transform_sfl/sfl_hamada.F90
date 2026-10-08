!===================================================================================================================================
! Copyright (c) 2026 GVEC Contributors, Max Planck Institute for Plasma Physics
! License: MIT
!===================================================================================================================================
#include "defines.FPP"
!===================================================================================================================================
!>
!!# Module **SFL hamada**
!!
!! Angular potential nu of straight-field line HAMADA coordinates on one flux surface.
!!
!! theta^H=theta+lambda+iota*nu(theta,zeta)
!! zeta^H =zeta                +nu(theta,zeta)
!!
!! Hamada coordinates have a Jacobian that is constant on each flux surface, J_H=<J>=1/(4pi^2) int J dthet dzeta.
!! The angular Jacobian of the map is linear in nu,
!!   det(d(theta^H,zeta^H)/d(theta,zeta)) = (1+dlambda/dtheta)(1+dnu/dzeta) + (iota-dlambda/dzeta)*dnu/dtheta ,
!! and in PEST angles (theta^P=theta+lambda,zeta), J_H=<J> becomes the magnetic differential equation
!!   (iota d/dtheta^P + d/dzeta) nu = J/(<J>(1+dlambda/dtheta)) - 1 =: f ,
!! which is diagonal in a Fourier series exp(i(m theta^P - n zeta)):  nu_mn = f_mn / (i(iota*m-n)).
!! Since dtheta^P = (1+dlambda/dtheta) dtheta, the PEST Fourier coefficients of f are integrals on the logical grid:
!!   f_mn ~ int ( J/<J> - (1+dlambda/dtheta) ) exp(-i(m(theta+lambda) - n zeta)) dtheta dzeta .
!! Finally, nu is evaluated at the logical integration points and projected onto the logical fourier basis.
!!
!! The source f_mn has to vanish on rational surfaces iota=n/m (current closure condition), otherwise nu_mn is singular.
!! Modes with |iota*m-n| below a threshold are set to zero and reported.
!===================================================================================================================================
MODULE MODgvec_SFL_Hamada
! MODULES
USE MODgvec_Globals, ONLY:wp,abort,TWOPI,UNIT_stdOut
#if MPI
USE MODgvec_Globals, ONLY:MPIroot
#endif
USE MODgvec_fbase   ,ONLY: t_fbase
IMPLICIT NONE
PRIVATE

PUBLIC :: get_nu_hamada
!===================================================================================================================================

CONTAINS

!===================================================================================================================================
!> compute the Fourier modes of the Hamada angular potential nu on one flux surface, from the values of the Jacobian and of
!> lambda at the integration points of nu_fbase.
!!
!===================================================================================================================================
SUBROUTINE get_nu_hamada(nu_fbase,iota,LA_IP,dLAdthet_IP,Jac_IP,nu)
  ! MODULES
  IMPLICIT NONE
  ! INPUT VARIABLES -------------------------!
  TYPE(t_fbase),INTENT(IN) :: nu_fbase                   !! fourier basis of nu, defines the integration points
  REAL(wp)     ,INTENT(IN) :: iota                       !! rotational transform on the flux surface
  REAL(wp)     ,INTENT(IN) :: LA_IP(nu_fbase%mn_IP)      !! lambda at the integration points
  REAL(wp)     ,INTENT(IN) :: dLAdthet_IP(nu_fbase%mn_IP) !! dlambda/dtheta at the integration points
  REAL(wp)     ,INTENT(IN) :: Jac_IP(nu_fbase%mn_IP)     !! Jacobian of the logical coordinates at the integration points
  ! OUTPUT VARIABLES -------------------------!
  REAL(wp)     ,INTENT(OUT):: nu(nu_fbase%modes)         !! Fourier modes of nu in the logical angles
  ! LOCAL VARIABLES --------------------------!
  INTEGER                  :: mn_IP,m_max,n_max,nfp,i_mn,m,n,n_res
  REAL(wp)                 :: Jac_avg,dthet_dzeta,denom,tol_res,f_res
  REAL(wp)                 :: g_IP(nu_fbase%mn_IP),nu_IP(nu_fbase%mn_IP)
  COMPLEX(wp),ALLOCATABLE  :: exp_t(:,:),exp_z(:,:),f_mn(:,:),nu_mn(:,:)
  ! CODE --------------------------------------------------------------------------------------------------------------------------!
  mn_IP       = nu_fbase%mn_IP
  m_max       = nu_fbase%mn_max(1)
  n_max       = nu_fbase%mn_max(2)
  nfp         = nu_fbase%nfp
  dthet_dzeta = nu_fbase%d_thet*nu_fbase%d_zeta

  Jac_avg = SUM(Jac_IP)/REAL(mn_IP,wp)
  IF(ABS(Jac_avg).LT.TINY(1.0_wp)) CALL abort(__STAMP__, &
       'get_nu_hamada: average of the Jacobian is zero')
  ! source multiplied by the angular Jacobian of the PEST angles, f*(1+dlambda/dtheta)
  g_IP = Jac_IP/Jac_avg - (1.0_wp + dLAdthet_IP)

  ALLOCATE(exp_t(0:m_max,mn_IP),exp_z(-n_max:n_max,mn_IP))
  ALLOCATE(f_mn(0:m_max,-n_max:n_max),nu_mn(0:m_max,-n_max:n_max))

  !$OMP PARALLEL DO SCHEDULE(STATIC) DEFAULT(NONE) PRIVATE(i_mn,m,n) &
  !$OMP   SHARED(mn_IP,m_max,n_max,nfp,nu_fbase,LA_IP,exp_t,exp_z)
  DO i_mn=1,mn_IP
    ! exp(i m theta^P) and exp(-i n nfp zeta) at the integration point
    exp_t(0,i_mn) = (1.0_wp,0.0_wp)
    exp_z(0,i_mn) = (1.0_wp,0.0_wp)
    IF(m_max.GT.0) exp_t(1,i_mn) = EXP(CMPLX(0.0_wp, nu_fbase%x_IP(1,i_mn)+LA_IP(i_mn),wp))
    IF(n_max.GT.0) exp_z(1,i_mn) = EXP(CMPLX(0.0_wp,-REAL(nfp,wp)*nu_fbase%x_IP(2,i_mn),wp))
    DO m=2,m_max
      exp_t(m,i_mn) = exp_t(m-1,i_mn)*exp_t(1,i_mn)
    END DO
    DO n=2,n_max
      exp_z(n,i_mn) = exp_z(n-1,i_mn)*exp_z(1,i_mn)
    END DO
    DO n=1,n_max
      exp_z(-n,i_mn) = CONJG(exp_z(n,i_mn))
    END DO
  END DO
  !$OMP END PARALLEL DO

  ! f_mn = 1/(2pi^2) int g exp(-i(m theta^P - n nfp zeta)) dtheta dzeta,
  ! such that f = Re( sum_mn f_mn exp(i(m theta^P-n nfp zeta)) ), over the half plane of modes m>0 or (m=0,n>0)
  !$OMP PARALLEL DO SCHEDULE(STATIC) DEFAULT(NONE) PRIVATE(m,n) &
  !$OMP   SHARED(m_max,n_max,g_IP,exp_t,exp_z,f_mn,dthet_dzeta)
  DO m=0,m_max
    DO n=-n_max,n_max
      f_mn(m,n) = SUM(g_IP(:)*CONJG(exp_t(m,:)*exp_z(n,:)))*(dthet_dzeta*2.0_wp/(TWOPI*TWOPI))
    END DO
  END DO
  !$OMP END PARALLEL DO

  ! solve (iota d/dtheta^P + d/dzeta) nu = f mode by mode
  tol_res = 1.0e-10_wp
  n_res = 0
  f_res = 0.0_wp
  nu_mn = (0.0_wp,0.0_wp)
  DO m=0,m_max
    DO n=-n_max,n_max
      IF((m.EQ.0).AND.(n.LE.0)) CYCLE
      denom = iota*REAL(m,wp) - REAL(n*nfp,wp)
      IF(ABS(denom).LT.tol_res*MAX(1.0_wp,REAL(m,wp))) THEN
        n_res = n_res+1
        f_res = MAX(f_res,ABS(f_mn(m,n)))
        CYCLE
      END IF
      nu_mn(m,n) = f_mn(m,n)/CMPLX(0.0_wp,denom,wp)
    END DO
  END DO
  IF(n_res.GT.0) THEN
    SWRITE(UNIT_stdOut,'(6X,A,I4,A,E11.4,A,E11.4)') 'WARNING get_nu_hamada: ',n_res, &
          ' resonant mode(s) set to zero, iota=',iota,', max |f_mn|=',f_res
  END IF

  ! evaluate nu at the logical integration points (theta^P=theta+lambda) and project onto the logical basis
  !$OMP PARALLEL DO SCHEDULE(STATIC) DEFAULT(NONE) PRIVATE(i_mn,m) &
  !$OMP   SHARED(mn_IP,m_max,nu_mn,exp_t,exp_z,nu_IP)
  DO i_mn=1,mn_IP
    nu_IP(i_mn) = 0.0_wp
    DO m=0,m_max
      nu_IP(i_mn) = nu_IP(i_mn) + REAL(exp_t(m,i_mn)*SUM(nu_mn(m,:)*exp_z(:,i_mn)),wp)
    END DO
  END DO
  !$OMP END PARALLEL DO

  CALL nu_fbase%projectIPtoDOF(.FALSE.,dthet_dzeta,0,nu_IP,nu)
  nu = nu*nu_fbase%snorm_base
  IF(nu_fbase%mn_zero_mode.GT.0) nu(nu_fbase%mn_zero_mode) = 0.0_wp

  DEALLOCATE(exp_t,exp_z,f_mn,nu_mn)
END SUBROUTINE get_nu_hamada

END MODULE MODgvec_SFL_Hamada
