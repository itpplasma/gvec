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

[V] Trial acceptance now also evaluates a separate grid with twice as many
Gauss points per radial element and twice as many points in each angle. It
uses the existing basis and mapping evaluators, preallocated work arrays,
and an OpenMP reduction. A FortSym-generated `wp` determinant leaf includes
JVP/VJP forms; its affine-map identity is checked symbolically by the generator.
The original integration-grid check remains in place. Neither sampling grid
certifies continuous positivity, and the singular axis is not sampled.
`test_offgrid_jacobian.py` constructs an analytic quintic map with positive
Jacobians at every original integration point but a negative off-grid well;
no trial of this map is accepted even at an effectively zero step.

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

[V] The fast CI regression uses exact-interior starts at modes/elements 6 and
8 for both tau values. `nested_initialization.py` projects the analytic chart
onto the native Fourier/quintic spline basis and checks its flux labels
against the independent field oracle before solving. No stored numerical
solution is loaded. All four cases satisfy native force <=1e-6; B and |B|
RMS gates are three times the corresponding B RMS values above. This margin
covers compiler/stopping-point variation at the solver tolerance; it does not
require agreement in digits below that tolerance. Strong-force RMS gates also use three times the ladder values: 1.2994e-3
and 1.3218e-4 (Landreman, 6/8), 4.5840e-4 and 1.4798e-4 (Issan, 6/8).
Surface and iota diagnostics retain the existing physical-accuracy gates.

[V] These tests validate discretisation and the minimiser fixed point. Cold
starts at eps=0.5 remain a separate solver-robustness problem, not a fast-CI
admission criterion. The earlier boundary continuation from an eps=0 cold
solve stalled at eps=0.1. Keep the fork PR draft.

[V] `surface_distance` intersects a ray from the analytic axis with the exact
surface in the candidate's geometric-phi section. Its maximum is a sampled
upper bound on nearest-surface distance; it is not a continuous enclosure.
Field and force errors are equal-weight sample RMS values, not volume norms.
The sheared families are not admitted here: matching their pressure and iota
in toroidal-flux coordinates is separate work from their boundary export.
