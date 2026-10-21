The trial-map determinant and its JVP/VJP are generated with FortSym. The
including module supplies GVEC's `wp`, so the leaf follows its numerical kind.
The generator checks the affine-map determinant and orientation derivative
identities symbolically. No symbolic library is required to build GVEC.

Regenerate with an already built FortSym (tested at
7e8ef7abdd90c49bb5f75e2c119e7d181656bcc6):

```sh
gfortran -I"$FORTSYM_BUILD/include" gen/gen_trial_jacobian.f90 \
  -L"$FORTSYM_BUILD/lib" -Wl,-rpath,"$FORTSYM_BUILD/lib" -lfortsym \
  -o /tmp/gen_trial_jacobian
/tmp/gen_trial_jacobian src/functionals/mhd3d/trial_jacobian.inc
```
