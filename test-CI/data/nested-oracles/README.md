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

[V] The original cold-start solves at modes/elements 4/6/8 abort on a
nonpositive predicted Jacobian. The minimizer now rejects that trial, keeps
the accepted map, reduces the step and resets acceleration, including after
iteration 100. The focused regression forces this rejection after iteration
101 and checks subsequent progress, positive sampled Jacobians and a fixed
boundary. This retry also existed in the fork's older perf-descent-solver
branch; its unrelated staging/performance changes are not required here.

[V] Retrying prevents the immediate abort but does not solve the cold-start
problem. Default and exact-axis starts still stall above the requested
1e-6 force tolerance. Disabling Lambda initialization, plain gradient
steps, strict energy decrease, cubic splines and increased quadrature did
not establish cold-start convergence. The default initializer is in
`src/functionals/mhd3d/mhd3d.F90:1035`; Jacobian acceptance in
`mhd3d_evalfunc.F90:338` samples the integration grid. An independent finer
grid detects negative Jacobians near rho=0.925 in accepted stalled states.

[V] Initializing the whole interior from the analytic chart gives the
following sampled field errors after satisfying the native 1e-6 criterion
with gfortran 12.2. These are initialization/representation checks, not
cold-start solver regression evidence.

| Modes / radial elements | Landreman B relative RMS | Issan B relative RMS |
|---|---:|---:|
| 6 / 6 | 3.79e-5 | 1.48e-5 |
| 8 / 8 | 1.93e-6 | 7.72e-7 |
| 10 / 10 | 1.52e-7 | 3.59e-8 |
| 12 / 12 | 1.75e-8 | 4.54e-9 |

[V] Continuum curvature probes of both early and late four-mode departure
directions are positive at two quadratures. A broader 54-function Ritz
probe is also positive in its tested subspace. None proves full stability.
[H] A discrete admissibility/representation defect remains a candidate;
the four-mode truncation is insufficient for this diagnostic ladder.

[H] The six-mode solved-case thresholds remain provisional physical accuracy
targets. No converged cold-start ladder justifies solver regression gates;
keep the PR draft. An exact-interior start must not replace the cold-start
regression merely to pass CI. Boundary continuation from an eps=0 cold solve
converges at its first stage but stalls at eps=0.1; a successful continuation
recipe has not been established.

[V] `surface_distance` intersects a ray from the analytic axis with the exact
surface in the candidate's geometric-phi section. Its maximum is a sampled
upper bound on nearest-surface distance; it is not a continuous enclosure.
Field and force errors are equal-weight sample RMS values, not volume norms.
The sheared families are not admitted here: matching their pressure and iota
in toroidal-flux coordinates is separate work from their boundary export.
