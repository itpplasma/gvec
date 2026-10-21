! Copyright (c) 2026 GVEC Contributors, Max Planck Institute for Plasma Physics
! License: MIT
! Measure f90wrap handles in C integers using GVEC's Fortran compiler.
program sizeof_fortran_t
    use, intrinsic :: iso_c_binding, only: c_int
    implicit none
    type pointer_type_t
        type(pointer_type_t), pointer :: p => null()
    end type
    type pointer_class_t
        class(pointer_class_t), pointer :: p => null()
    end type
    type(pointer_type_t) :: typed
    type(pointer_class_t) :: polymorphic

    print '(i0)', max(size(transfer(typed, [0_c_int])), &
                      size(transfer(polymorphic, [0_c_int])))
end program sizeof_fortran_t
