# Exact nested regression candidates

[V] `input.landreman` and `input.issan` come from the integer-family
[KIN6D](https://github.com/itpplasma/kin6d) exporter at
0dc35d0058f42ce91963dcc94dc8abfcd0637917 (KIN6D PR 24). They have eps=0.5,
delta=1/64 and tau=0 or 0.4. Regenerate with `nested_oracle_export` and the
matching `examples/nested-oracles/*-integer.nml`, using mmax=nmax=12,
angles=64, quadrature=8 and turns=1. Use `output/input.nested_equilibrium`.
The material angle must be untwisted as in that revision before prescribing
iota=2. Angles are converted by GVEC's existing theta/zeta adapters.

[V] Coefficients smaller than 1e-11 were omitted from these text fixtures.
The tau=0 fixture also omits RBS and ZBC: all were below 1e-13. This lets
GVEC select its symmetric basis for Landreman and all sine/cosine modes for
Issan, rather than selecting asymmetry merely because a zero array exists.
Both fixtures retain their matched toroidal flux and linear pressure profile
in normalized toroidal flux. No profile is fitted to a numerical result.

[V] `field-reference.txt` contains eight Cartesian point evaluations from
KIN6D b4927ff1a3ffc6df3bc4b3532bd92a9e24f887a5, from rows 17, 513, 1513 and
3071 (zero based) of each integer-family `nested_oracle_export.nc`.
It checks the independent complex-step-capable Python oracle used by this
test against the FortSym-generated Fortran leaves. The latter's eight exact
divergence/tension lemmas also pass at the current retained definition.

[V] The native gfortran-12.2 mini-ladder at eps=0.5, delta=1/64 and
`minimize_tol=1e-6` folds at the following iterations. All six aborts come
from `src/functionals/mhd3d/mhd3d_minimize.F90:496`, when the predicted map
has a nonpositive Jacobian after iteration 100.

| Fourier maximum, radial elements | Landreman, tau=0 | Issan, tau=0.4 |
|---|---:|---:|
| 4, 4 | 11319 | 6847 |
| 6, 6 | 8439 | 6587 |
| 8, 8 | 8271 | 6565 |

[H] The folding's mathematical/numerical root cause is unresolved. An
independent continuum-energy probe of three admissible displacement modes
found stationary first variations and positive curvature in that subspace;
it does not establish stability or explain the discrete fold. Neither the
paper nor the reference implementation is declared wrong on this evidence.

[H] The proposed six-mode solved-case tolerances are provisional physical
accuracy targets, not established regression gates. Keep the PR draft until
the cause is resolved and a converged mini-ladder justifies the gates. A
native small force alone is insufficient: the tests also compare field,
surface position, rotational transform and strong force balance to the oracle.

[V] `surface_distance` intersects a ray from the analytic axis with the exact
surface in the candidate's geometric-phi section. Its maximum is a sampled
upper bound on nearest-surface distance; it is not a continuous enclosure.
Field and force errors are equal-weight sample RMS values, not volume norms.
The sheared families are not admitted here: matching their pressure and iota
in toroidal-flux coordinates is separate work from their boundary export.
