# Copyright (c) 2026 GVEC Contributors, Max Planck Institute for Plasma Physics
# License: MIT
import numpy as np
import xarray as xr
from collections.abc import Mapping
from scipy.constants import mu_0
from pandas import read_csv
from tqdm import tqdm
import time
import logging
import shutil
from datetime import datetime
from typing import Literal
from pathlib import Path

# GVEC
import gvec
from gvec.coils import CoilSet, Coil
from gvec.core.state import State
from gvec.core.minimize import MinimizationState
from .abstract import (
    AbstractRun,
    DEFAULT_MINIMIZE_TOL,
    DEFAULT_TOTALITER,
    DIAGNOSTICS_NPOINTS,
)
from gvec.util import write_parameter_file_ini, flatten_parameters, chdir
from gvec.core.compute import ev2ft
from gvec import fourier
from gvec.lib import _state


DEFAULT_FORCE_RATIO = 0.01
DEFAULT_FORCE_TOL_START = 1e-3
DEFAULT_ALPHA = 10.0
DEFAULT_BLEND_DEPTH = 0.6
DEFAULT_GTOL = 1e-8
DEFAULT_XTOL = 1e-5
DEFAULT_DIGITS_BIEST = 8
BIEST_MIN_NZETA = 6  # BIEST asserts Nt >= PATCH_DIM (= 6) toroidal points per field period
DEFAULT_FB_TOLERANCE = 1e-8
FORCE_RATIO_MINIMUM_FORCE = 1e-12
DEFAULT_FTOL = 1e-5
MOVING_AVERAGE_WINDOW = 10

logger = logging.getLogger("gvec.run.free_boundary")


class FreeBoundaryRun(AbstractRun):
    def __init__(
        self,
        params: Mapping,
        coil_set: CoilSet | Coil,
        state: Path | State | None = None,
        redirect_gvec_stdout: bool = True,
        quiet: bool = False,
        parameter_format: Literal["toml", "yaml"] = "toml",
        save_fb_updates: bool = False,
    ):
        params["FreeBoundary"] = True
        super().__init__(params, state, redirect_gvec_stdout, quiet, parameter_format)
        self.coil_set = coil_set
        self.save_fb_updates = save_fb_updates
        self.iters_used = 0
        self._resolved_force_ratio = 1.0

        self.m_max, self.n_max = self.state.get_mn_max()
        self._reset_convergence_statistics()

        if self._has_initial_stage:
            self.stages[0]["free_boundary"] = {"updates": 0}

    def _reset_convergence_statistics(self):
        """Forget the step/gradient norms and moving averages; each stage starts its own statistics."""
        self.target_mavg_last = np.nan
        self.target_mavg = np.nan

        self.xnorm = np.nan
        self.gnorm = np.nan
        self.fdelta = np.nan

        self.xnorm_mavg = np.nan
        self.gnorm_mavg = np.nan

    def _allocate_diagnostics(self, track_boundary_fields: bool):
        """Setup the diagnostics dataset."""
        diagnostics = self.state.evaluate("xyz", "theta", "zeta", rho=1.0).isel(rad=0)
        fb_updates = max(self._state_parameters["free_boundary"]["updates"], 1)
        diagnostics["updates"] = ("updates", np.arange(fb_updates))
        diagnostics["force_balance"] = (
            ("tor", "pol", "updates"),
            np.full(shape=(self.state.z_n, self.state.t_n, fb_updates), fill_value=np.nan),
        )
        diagnostics.force_balance.attrs = dict(
            long_name="free-boundary force balance on the LCFS.", symbol=r"$F_{b}}$"
        )
        if track_boundary_fields:
            for key, long_name, symbol in [
                ("B_jump", "jump of the magnetic field on the LCFS", r"$B_{\text{jump}}$"),
                ("B_int", "internal magnetic field on the LCFS", r"$B_{\text{int}}$"),
                ("B_ext", "external magnetic field on the LCFS", r"$B_{\text{ext}}$"),
                ("K", "surface current on the LCFS", r"$K$"),
            ]:
                diagnostics[key] = (
                    ("xyz", "tor", "pol", "updates"),
                    np.full(
                        shape=(3, self.state.z_n, self.state.t_n, fb_updates), fill_value=np.nan
                    ),
                )
                diagnostics[key].attrs = dict(long_name=long_name, symbol=symbol)

        for key in [
            "free_boundary_target",
            "free_boundary_target_mavg",
            "fdelta",
            "xnorm",
            "xnorm_mavg",
            "gnorm",
            "gnorm_mavg",
            "Fnorm_X1_b_IP",
            "Fnorm_X2_b_IP",
            "Fnorm_X1_b_modes",
            "Fnorm_X2_b_modes",
            "Fnorm_X1_b",
            "Fnorm_X2_b",
            "spectral_width",
            "dt",
            "resolved_force_ratio",
        ]:
            diagnostics[key] = (
                "updates",
                np.full(shape=fb_updates, fill_value=np.nan),
            )

        m, n, F = self.state._get_boundary_forces()
        diagnostics["m_modes"] = ("mode", m)
        diagnostics["n_modes"] = ("mode", n)
        diagnostics["boundary_forces"] = (
            ("modes", "updates"),
            np.full(shape=(len(m), fb_updates), fill_value=np.nan),
        )

        if self.state.I_tor:
            diagnostics["iota"] = (
                ("rho_iota", "updates"),
                np.full(shape=(DIAGNOSTICS_NPOINTS, fb_updates), fill_value=np.nan),
            )
            diagnostics["rho_iota"] = np.linspace(0, 1, DIAGNOSTICS_NPOINTS)

        if self.diagnostics_stage is None:
            self.diagnostics_stage = xr.DataTree()
        self.track_boundary_fields = track_boundary_fields
        self.diagnostics_stage[f"stage_{self.nth_stage}"] = diagnostics

    def _update_force_ratio(self, diag_aux, update):
        force_balance_ds = diag_aux.dataset.isel(updates=update)
        force_balance_ds["N_FP"] = self.state.nfp
        fourier_force_balance = ev2ft(force_balance_ds[["N_FP", "force_balance"]])
        cut_fb = fourier_force_balance.where(
            (fourier_force_balance.m <= self.m_max), drop=True
        ).where(np.abs(fourier_force_balance.n) <= self.n_max, drop=True)
        full_force = max(
            (fourier_force_balance.force_balance_mns**2).sum().item()
            + (fourier_force_balance.force_balance_mnc**2).sum().item(),
            FORCE_RATIO_MINIMUM_FORCE,
        )
        cut_force = (cut_fb.force_balance_mns**2).sum().item() + (
            cut_fb.force_balance_mnc**2
        ).sum().item()
        self._resolved_force_ratio = cut_force / full_force

    def _update_diagnostics(
        self,
        B_ext_2: np.ndarray,
        B_coils: np.ndarray,
        B_plas_ext: np.ndarray,
        update: int,
        dt: float,
    ):
        """Helper function to fill the diagnostics dataset

        Parameters
        ----------
        B_ext_2 : np.ndarray
            Squared external magnetic field to evaluate the free-boundary force balance
        B_coils : np.ndarray
            Coil field at the boundary of shape (3,tor,pol)
        B_plas_ext : np.ndarray
            Plasma contributionn to the external field at the boundary (BIEST) of shape (3,tor,pol)
        update : int
            Determines which entry in the allocated dataset is filled.
        """

        free_boundary_target, force_balance, ev_gvec = self.state._free_boundary_target(B_ext_2)
        B_ext = B_coils + B_plas_ext
        B_jump = ev_gvec.B - B_ext

        diag_aux = self.diagnostics_stage[f"stage_{self.nth_stage}"]
        diag_aux.force_balance[:, :, update] = force_balance
        diag_aux.free_boundary_target[update] = free_boundary_target
        Fnorm = self.state.get_Fnorm_boundary()
        if self.track_boundary_fields:
            # mu_0*K = n x [[B]]
            diag_aux.K[:, :, :, update] = np.cross(ev_gvec.normal, B_jump, axis=0) / mu_0
            diag_aux.B_ext[:, :, :, update] = B_ext
            diag_aux.B_jump[:, :, :, update] = B_jump.data
            diag_aux.B_int[:, :, :, update] = ev_gvec.B.data

        diag_aux.Fnorm_X1_b_IP[update] = Fnorm[0]
        diag_aux.Fnorm_X1_b_modes[update] = Fnorm[1]
        diag_aux.Fnorm_X2_b_IP[update] = Fnorm[2]
        diag_aux.Fnorm_X2_b_modes[update] = Fnorm[3]
        diag_aux.Fnorm_X1_b[update] = Fnorm[4]
        diag_aux.Fnorm_X2_b[update] = Fnorm[5]

        diag_aux.spectral_width[update] = _state.get_spectral_width()

        diag_aux.dt[update] = dt
        self.free_boundary_target = free_boundary_target.item()

        self._update_force_ratio(diag_aux, update)
        diag_aux.resolved_force_ratio[update] = self._resolved_force_ratio

        diag_aux.boundary_forces[:, update] = self.state._get_boundary_forces()[2]

        diag_aux.xnorm[update] = self.xnorm
        diag_aux.gnorm[update] = self.gnorm

        # moving averages
        diag_aux.free_boundary_target_mavg[update] = np.mean(
            diag_aux.free_boundary_target[
                update - min(MOVING_AVERAGE_WINDOW, update) : update + 1
            ]
        )
        diag_aux.fdelta[update] = (
            np.std(
                diag_aux.free_boundary_target_mavg[
                    update - min(MOVING_AVERAGE_WINDOW, update) : update + 1
                ]
            )
            if update != 0
            else np.nan
        )

        diag_aux.xnorm_mavg[update] = np.mean(
            diag_aux.xnorm[update - min(MOVING_AVERAGE_WINDOW, update) : update + 1].dropna(
                dim="updates"
            )
        )
        diag_aux.gnorm_mavg[update] = np.mean(
            diag_aux.gnorm[update - min(MOVING_AVERAGE_WINDOW, update) : update + 1].dropna(
                dim="updates"
            )
        )

        self.fdelta = diag_aux.fdelta[update].item()
        self.xnorm_mavg = diag_aux.xnorm_mavg[update].item()
        self.gnorm_mavg = diag_aux.gnorm_mavg[update].item()

        if np.isnan(self.target_mavg_last):
            self.target_mavg_last = diag_aux.free_boundary_target_mavg[update]
        else:
            self.target_mavg_last = diag_aux.free_boundary_target_mavg[update - 1]

        self.target_mavg = diag_aux.free_boundary_target_mavg[update]

        self.diagnostics_stage[f"stage_{self.nth_stage}"] = diag_aux

    def _run_stage(self):
        """Perform the free-boundary run.

        algorithm
        ---------
        1. start with a fixed-boundary minimization from the initial guess
        2. calculate the external field using BIEST and the coils
        3. update the boundary forces and push the boundary via boundary perturbations
        4. minimize the fixed-boundary equilibrium with the new boundary
        5. repeat from 2. until convergence  or until the budget is used up
        """

        # start with a fixed boundary equilibrium
        # the norms of the previous stage must not feed this stage's moving averages and stop criteria
        self._reset_convergence_statistics()

        # how many free boundary updates,
        # NOTE: required parameter
        fb_updates = self._state_parameters["free_boundary"]["updates"]

        # target termination criterion
        free_boundary_tolerance = self._state_parameters["free_boundary"].get(
            "tolerance", DEFAULT_FB_TOLERANCE
        )

        # ftol termination criterion
        ftol = self._state_parameters["free_boundary"].get("ftol", DEFAULT_FTOL)

        minimizer_type = self._state_parameters["free_boundary"].get("minimizer_type", "GD")

        step_size_heuristic = False
        match minimizer_type:
            case "GD" | "mGD" | "lBFGS":
                step_size_heuristic = True
            case "BFGS" | "dBFGS" | "SR1" | "aGD" | "ADAM":
                step_size_heuristic = False
            case _:
                raise ValueError(f"Unknown free-boundary minimizer_type: {minimizer_type}")

        # If True, saves diagnostics after each free-b update
        save_diagnostics = self._state_parameters["free_boundary"].get(
            "save_diagnostics", False
        )

        # resolved force termination criterion
        force_ratio_limit = self._state_parameters["free_boundary"].get(
            "force_ratio_limit", DEFAULT_FORCE_RATIO
        )

        # If True, adaptively decrease the force tolerance according to the free-b target
        ramp_force_tol = self._state_parameters["free_boundary"].get("ramp_force_tol", True)

        track_boundary_fields = self._state_parameters["free_boundary"].get(
            "track_boundary_fields", False
        )

        # starting tolerance for the ramping
        ramp_force_tol_start = self._state_parameters["free_boundary"].get(
            "ramp_force_tol_start", DEFAULT_FORCE_TOL_START
        )

        # ramping factor for force tolerance: force_tolerance = free_boundary_target / alpha
        alpha = self._state_parameters["free_boundary"].get("alpha", DEFAULT_ALPHA)

        # how far the boundary is blend into an equilibrium after a free-boundary push
        blend_depth = self._state_parameters["free_boundary"].get(
            "blend_depth", DEFAULT_BLEND_DEPTH
        )

        # gradient termination criterion
        gtol = self._state_parameters["free_boundary"].get("gtol", DEFAULT_GTOL)

        # DOF change termination criterion
        # NOTE: fourier coefficients are weighted according to expected spectral decay
        xtol = self._state_parameters["free_boundary"].get("xtol", DEFAULT_XTOL)

        # how accurate BIEST solves for the exterior field
        digits_BIEST = self._state_parameters["free_boundary"].get(
            "digits_BIEST", DEFAULT_DIGITS_BIEST
        )

        # option to limit the number of updates per free-boundary update
        # TODO: Can this  be replaced by maxIter?
        max_iter_per_update = self._state_parameters["free_boundary"].get(
            "max_iter_per_update", DEFAULT_TOTALITER
        )

        # targeted minimize_tol
        minimize_tol_min = self._state_parameters.get("minimize_tol", DEFAULT_MINIMIZE_TOL)

        # minimum stepsize
        min_dt = self._state_parameters["free_boundary"].get("min_dt", 1e-4)

        # Flag to restart the free boundary minimizer a step-size reduction.
        reset_minimizer = False

        if ramp_force_tol:
            minimize_tol = ramp_force_tol_start
            current_lowest_min_tol = ramp_force_tol_start
        else:
            minimize_tol = minimize_tol_min

            # find previous state
        if self.state.statefile:
            logger.debug(f"Restart from statefile {self.state.statefile}")

        # prepare the run directory
        self.rundir = self.project_dir / Path(f"stage-{self.nth_stage:1d}")
        if self.rundir.exists():
            logger.debug(f"Removing existing run directory {self.rundir}")
            shutil.rmtree(self.rundir)
        self.rundir.mkdir()
        logger.debug(f"Created run directory {self.rundir}")

        self.state = MinimizationState.new(
            self._state_parameters,
            rundir=self.rundir,
            statefile=self.state.statefile,
            redirect_stdout=self.redirect_gvec_stdout,
            stdout_path=self.rundir / "stdout.txt",
            header=[
                "GVEC parameter file",
                f"Auto-generated with `pygvec run` (stage {self.nth_stage})",
                f"Created at {datetime.now().isoformat()}",
                f"pyGVEC v{gvec.__version__}",
            ],
        )
        with chdir(self.rundir):
            # write parameterfile & run GVEC
            self.m_max, self.n_max = self.state.get_mn_max()

            # initial stepsize for the free-b minimization
            dt = self.state.dt_free
            max_dt = self.state.max_dt_free

            if self.save_intermediate:
                self.state.save(
                    f"!Auto-generated with `pygvec run` (stage {self.nth_stage})\n"
                    f"!Created at {datetime.now().isoformat()}\n"
                    f"!pyGVEC v{gvec.__version__}\n",
                )

            self.state.solve(
                minimize_tol=minimize_tol,
                maxIter=self._state_parameters["maxIter"],
                save_intermediate=self.save_intermediate,
                save_header=f"!Auto-generated with `pygvec run` (stage {self.nth_stage})\n",
            )
            self._allocate_diagnostics(track_boundary_fields)
            self.iters_used = self.state.get_iterations()

            # NOTE: always use self-consistent I_tor values
            self.I_tor_boundary = (
                self.state.evaluate("I_tor", rho=1.0).squeeze()["I_tor"].item()
            )

            # Call to BIEST
            (B2, B_plas_ext, B_coils, B_ext_2) = _get_B2_fp(
                state=self.state,
                coil_set=self.coil_set,
                ntheta=self.state.t_n,
                nzeta=self.state.z_n,
                I_tor=self.I_tor_boundary,
                diagnostic_output=True,
                digits=digits_BIEST,
            )

            self._update_diagnostics(B_ext_2, B_coils, B_plas_ext, 0, dt)

            fb_update = 0
            method = self._state_parameters["free_boundary"].get("method", "descent")
            if method not in ("descent", "newton"):
                raise ValueError(f"Unknown free-boundary method: {method}")
            try:  # allow clean abort with KeybordInterrupt
                if method == "newton" and fb_updates > 1:
                    fb_update = self._run_newton(
                        fb_updates=fb_updates,
                        blend_depth=blend_depth,
                        minimize_tol=minimize_tol_min,
                        maxIter=self._state_parameters["maxIter"],
                        digits_BIEST=digits_BIEST,
                        free_boundary_tolerance=free_boundary_tolerance,
                    )
                for fb_update in range(1, fb_updates if method == "descent" else 0):
                    logger.info(
                        f"============= free-boundary update {fb_update} / {fb_updates} ============="
                    )
                    # abort if budget is used up
                    if self.iters_used >= self.totaliter:
                        logger.warning(
                            "Maximum number of iterations reached! aborting free-boundary iterations."
                        )
                        break

                    # ensure that the maximum number of iterations is kept
                    max_iter_per_update_aux = min(
                        self.totaliter - self.iters_used, max_iter_per_update
                    )

                    if ramp_force_tol:
                        # minimize_tol is ahead of free boundary target
                        minimize_tol = np.max(
                            [
                                np.min(
                                    [self.free_boundary_target / alpha, ramp_force_tol_start]
                                ),
                                minimize_tol_min,
                            ]
                        )
                        # this is to ensure that minimize tol never increases
                        if minimize_tol < current_lowest_min_tol:
                            current_lowest_min_tol = minimize_tol
                        minimize_tol = np.min([minimize_tol, current_lowest_min_tol])
                        logger.info(f"Setting gvec minimize_tol: {minimize_tol:.2e}")

                    fb_target_reached = self.free_boundary_target <= free_boundary_tolerance
                    # ftol is relative to the moving-average target, and ftol/xtol act only on a full
                    # moving-average window (before that, fdelta and xnorm_mavg describe 1-2 updates)
                    window_full = fb_update > MOVING_AVERAGE_WINDOW
                    ftol_reached = window_full and self.fdelta <= ftol * abs(self.target_mavg)
                    xtol_reached = window_full and self.xnorm_mavg <= xtol
                    gtol_reached = self.gnorm_mavg <= gtol
                    force_ratio_limit_reached = self._resolved_force_ratio <= force_ratio_limit

                    if fb_target_reached:
                        logger.info(
                            "Reached free boundary target tolerance! Terminating free boundary updates."
                        )
                        break
                    elif ftol_reached:
                        logger.info("Reached ftol! Terminating free boundary updates.")
                        break
                    elif xtol_reached:
                        logger.info("Reached xtol! Terminating free boundary updates.")
                        break
                    elif gtol_reached:
                        logger.info("Reached gtol! Terminating free boundary updates.")
                        break
                    elif force_ratio_limit_reached:
                        logger.info(
                            "Force ratio below specified limit! Terminating free boundary updates."
                        )
                        break
                    else:
                        logger.info(f"Δf: {self.fdelta:.2e}, ftol: {ftol:.2e}.")
                        logger.info(
                            f"xnorm: {self.xnorm:.2e}, m.avg. xnorm: {self.xnorm_mavg:.2e}, xtol: {xtol:.2e}."
                        )
                        logger.info(
                            f"gnorm: {self.gnorm:.2e}, m.avg. gnorm: {self.gnorm_mavg:.2e}, gtol: {gtol:.2e}."
                        )
                        logger.info(
                            f"force ratio {self._resolved_force_ratio:.2e}, limit {force_ratio_limit:.2e}"
                        )
                        t_fb_0 = time.time()

                        # Step-size dt is reduced if the moving average of the target increases until the minimum step-size is reached,
                        # otherwise it is increased. However, some algorithms such as BFGS also perform step-size estimations, which overwrite
                        # the provided step-size, if the estimated step-size is SMALLER.
                        if step_size_heuristic:
                            dt, reset_minimizer = _descent_step_size(
                                dt, self.target_mavg_last, self.target_mavg, min_dt, max_dt
                            )
                            if reset_minimizer:
                                logger.warning(
                                    "Reached minimum step-size! Resetting free boundary minimizer."
                                )

                        logger.info(f"Free boundary dt: {dt:.2e}")

                        # pushes boundary and does a fixed-b run
                        self.state._free_boundary_update(
                            maxIter=min(
                                max_iter_per_update_aux, self._state_parameters["maxIter"]
                            ),
                            B2=B2,
                            dt=dt,
                            reset_minimizer=reset_minimizer,
                            blend_depth=blend_depth,
                            minimize_tol=minimize_tol,
                            save_header=f"!Auto-generated with `pygvec run` (stage {self.nth_stage})\n"
                            + f"!free-boundary update {fb_update}\n",
                            save_intermediate=self.save_intermediate,
                        )
                        dt = self.state.dt_free

                        if self.save_fb_updates:
                            self.state.save(
                                f"!Auto-generated with `pygvec run` (stage {self.nth_stage})\n"
                                f"!free-boundary update {fb_update}\n"
                                f"!Created at {datetime.now().isoformat()}\n"
                                f"!pyGVEC v{gvec.__version__}\n"
                            )

                        # abort criteria
                        self.xnorm = self.state.get_xnorm() / dt
                        self.gnorm = self.state.get_gnorm()
                        # NOTE: always use self consistent I_tor
                        self.I_tor_boundary = (
                            self.state.evaluate("I_tor", rho=1.0).squeeze()["I_tor"].item()
                        )

                        # Call to BIEST
                        (B2, B_plas_ext, B_coils, B_ext_2) = _get_B2_fp(
                            state=self.state,
                            coil_set=self.coil_set,
                            ntheta=self.state.t_n,
                            nzeta=self.state.z_n,
                            I_tor=self.I_tor_boundary,
                            diagnostic_output=True,
                            digits=digits_BIEST,
                        )

                        self._update_diagnostics(B_ext_2, B_coils, B_plas_ext, fb_update, dt)

                        t_fb_1 = time.time()
                        logger.info(f"Time for free boundary update: {t_fb_1 - t_fb_0:.4}s")
                        reset_minimizer = False
                        self.iters_used = self.state.get_iterations()
                        if self.state.I_tor is None:
                            self.I_tor_boundary = self.state.evaluate(
                                "I_tor", rho=1.0
                            ).I_tor.item()
                        else:
                            self.diagnostics_stage[f"stage_{self.nth_stage}"].iota[
                                :, fb_update
                            ] = self.state.evaluate_profile(
                                "iota",
                                self.diagnostics_stage[f"stage_{self.nth_stage}"]["rho_iota"],
                            )

                    logger.info(
                        f"free-boundary - target: {self.free_boundary_target:.3e}, tolerance: {free_boundary_tolerance}, iters: {self.iters_used} / {self.totaliter} "
                    )
                    if save_diagnostics:
                        try:
                            self.diagnostics_stage[f"stage_{self.nth_stage}"].to_netcdf(
                                "diagnostics_aux.nc"
                            )
                        except PermissionError:
                            logger.warning(
                                "Permission error when attemting to write diagnostics to netcdf! Maybe the data-set is opened somewhere else?"
                            )

                # ensure that the final result is in fixed-boundary equilibrium with specified tolerance
                logger.info("=================================================================")
                logger.info("FINAL free-boundary optimization criteria")
                logger.info(
                    f"free-boundary - target: {self.free_boundary_target:.3e}, tolerance: {free_boundary_tolerance}, iters: {self.iters_used} / {self.totaliter} "
                )
                logger.info(f"Δf: {self.fdelta:.2e}, ftol: {ftol}.")
                logger.info(f"xnorm: {self.xnorm:.2e}, xtol: {xtol:.2e}.")
                logger.info(f"gnorm: {self.gnorm:.2e}, gtol: {gtol:.2e}.")
                logger.info(
                    f"resolved force-balance ratio: {self._resolved_force_ratio:.2e}, specified limit: {force_ratio_limit:.2e}."
                )
                logger.info("=================================================================")
                if self._state_parameters["maxIter"] > 0:
                    logger.info(
                        "Minimizing fixed-boundary equilibrium for the final boundary ..."
                    )
                    self.state.solve(
                        minimize_tol=minimize_tol_min,
                        maxIter=self._state_parameters["maxIter"],
                        save_header=f"!Auto-generated with `pygvec run` (stage {self.nth_stage})\n",
                    )

            except KeyboardInterrupt:
                logger.warning("Aborting free-boundary due to keyboard interrupt!")
            if (
                self._resolved_force_ratio < force_ratio_limit
                and self.free_boundary_target > free_boundary_tolerance
            ):
                fb_aux = (
                    self.diagnostics_stage[f"stage_{self.nth_stage}"]
                    .force_balance.isel(updates=fb_update - 1)
                    .transpose("pol", "tor")
                )
                M_proposed, N_proposed = minimal_modes_force_balance(
                    fb_aux,
                    zeta0=fb_aux.zeta[0].item(),
                    theta0=fb_aux.theta[0].item(),
                    tolerance=free_boundary_tolerance,
                )
                logger.info(
                    "Not all modes in the free-boundary force  balance can be resolved with the provided angular resolution.\n"
                    + "Try reducing minimze_tol or adapt the angular resolution.\n"
                    + "Proposed angular resolution increase for the solution variables X1 / X2 / LA:\n"
                    + f"    m_max - current {self.m_max} -> proposed {M_proposed}\n"
                    + f"    n_max - current {self.n_max} -> proposed {N_proposed}"
                )

            if self.free_boundary_target > free_boundary_tolerance and self.nth_stage > 0:
                logger.warning(
                    f"Free-boundary target not reached! target: {self.free_boundary_target:.3e}, specified tol.: {free_boundary_tolerance:.3e}"
                )

            self.state.save(
                f"!Auto-generated with `pygvec run` (stage {self.nth_stage})\n"
                f"!free-boundary update {fb_update}\n"
                f"!Created at {datetime.now().isoformat()}\n"
                f"!pyGVEC v{gvec.__version__}\n"
            )

        iterations = self.state.get_iterations()
        iteration_offset = self.GVEC_iter_used
        self.GVEC_iter_used += iterations

        # postprocessing
        statefiles_found = sorted(self.rundir.glob("*State*.dat"))
        if len(statefiles_found) == 0:
            raise FileNotFoundError(f"Could not find any State files in rundir {self.rundir}")
        self.state.statefile = Path(statefiles_found[-1]).absolute()

        logfiles = sorted(self.rundir.glob("logMinimizer_*"))
        if logfiles:
            logfile = logfiles[-1]
            log_df = read_csv(logfile, sep=",", header=0)

            diag_minimizer = xr.Dataset(
                dict(
                    total_iteration=iteration_offset
                    + np.array(log_df["#iterations"], dtype=int),
                    force_X1=("total_iteration", np.array(log_df["normF_X1"])),
                    force_X2=("total_iteration", np.array(log_df["normF_X2"])),
                    force_LA=("total_iteration", np.array(log_df["normF_LA"])),
                    W_MHD=("total_iteration", np.array(log_df["W_MHD3D"])),
                )
            )

            if self.diagnostics_minimizer is None:
                diag_minimizer.force_X1.attrs = dict(
                    long_name="absolute MHD force on X1", symbol=r"|F_{X^1}|"
                )
                diag_minimizer.force_X2.attrs = dict(
                    long_name="absolute MHD force on X2", symbol=r"|F_{X^2}|"
                )
                diag_minimizer.force_LA.attrs = dict(
                    long_name="absolute MHD force on lambda", symbol=r"|F_{\lambda}|"
                )
                diag_minimizer.W_MHD.attrs = dict(
                    long_name="total MHD energy", symbol=r"W_{MHD}"
                )
                self.diagnostics_minimizer = diag_minimizer
            else:
                self.diagnostics_minimizer = xr.concat(
                    [self.diagnostics_minimizer, diag_minimizer], dim="total_iteration"
                )

            forces = self.diagnostics_minimizer.isel(total_iteration=-1)[
                ["force_X1", "force_X2", "force_LA"]
            ]

            self.max_force = max([forces[force].item() for force in forces])
        else:
            self.max_force = np.inf

    # ============= NEWTON BOUNDARY UPDATE ===========================
    def _fb_residual(self, digits_BIEST: int):
        """Boundary force of the current (interior-converged) state with B_ext from the coils and BIEST
        evaluated on the current boundary. Returns (residual, (B_ext_2, B_coils, B_plas_ext))."""
        r, fields, self.I_tor_boundary = _boundary_residual(self.state, self.coil_set, digits_BIEST)
        return r, fields

    def _fb_move_and_solve(self, pert, blend_depth, minimize_tol, maxIter):
        return _move_and_solve(self.state, pert, blend_depth, minimize_tol, maxIter)

    def _fd_jacobian_parallel(self, perts, blend_depth, minimize_tol, maxIter, digits_BIEST, workers):
        """Forward differences of the boundary force along the columns of `perts` (n_dofs x k), computed by a pool
        of worker processes (one GVEC instance each), see _FDWorkerPool. Each worker restarts from a statefile of the
        current (checkpointed) state and differences against its own base residual. The pool is started at the first
        call and kept for the Newton iteration (closed by _run_newton).
        Returns the (n_dofs x k) matrix of residual differences (not divided by the step).
        """
        self.state._restore_dofs()
        self.state.save("!Newton free-boundary base state\n")
        if getattr(self, "_fd_pool", None) is None:
            self._fd_pool = _FDWorkerPool(
                workers, self._state_parameters, self.coil_set, blend_depth, minimize_tol, maxIter, digits_BIEST,
                Path.cwd() / "newton_workers",
            )
        return self._fd_pool.jacobian(Path(self.state.statefile).absolute(), perts)

    def _run_newton(self, *args, **kwargs) -> int:
        """Newton iteration for the free boundary, see _run_newton_loop; closes the FD worker pool at the end."""
        self._fd_pool = None
        try:
            return self._run_newton_loop(*args, **kwargs)
        finally:
            if self._fd_pool is not None:
                self._fd_pool.close()
                self._fd_pool = None

    def _run_newton_loop(
        self,
        fb_updates: int,
        blend_depth: float,
        minimize_tol: float,
        maxIter: int,
        digits_BIEST: int,
        free_boundary_tolerance: float,
    ) -> int:
        """Newton iteration for the free boundary.

        Unknowns: the normal displacement of the boundary, xi(theta, zeta) = sum_k y_k phi_k with the Fourier
        functions phi_k of the X1 boundary basis; the boundary coefficients change by dx = P y, P[:, k] being the
        boundary-basis projection of phi_k times the unit normal of the boundary curve in the zeta = const plane
        (`newton_space = "normal"`, default). Equations: g(y) = P^T F_b = 0, the virtual work of the boundary force
        F_b (pressure balance [[B^2/2 + mu_0 p]] in the weak form; the spectral-condensation term of the descent
        force is left out, since its normal part is not zero and biases the shape by up to ~0.4 mm) along the
        normal displacements. F_b is that of the interior equilibrium with the
        displaced boundary and B_ext = coils + BIEST evaluated on the displaced boundary. With
        `newton_space = "full"`, the unknowns are all boundary coefficients (P = I); this includes the nearly
        singular tangential re-parametrisations, whose stiffness is only the spectral-condensation weight.

        The Jacobian dg/dy is built by forward differences with step `newton_fd_step` [m] (one interior solve and
        one BIEST call per unknown; `newton_workers` > 1 runs them in parallel processes), updated by Broyden's
        rule after accepted steps and rebuilt when a step is rejected. Steps are Levenberg-Marquardt regularised,
        limited to `newton_max_step` [m] and backtracked on |g|. Unlike gradient descent on the boundary force,
        the rate does not depend on the ratio of stiff (toroidal-field, area) and soft (poloidal-field, position
        and shape) boundary modes, and modes that are unstable with fixed coil currents (vertical position) are
        converged as well.

        Stops when the largest normal-mode amplitude of the step is below `newton_xtol` [m], when a step smaller
        than `newton_noise_step` [m] does not decrease |g| (noise floor), when |g| fell below `newton_rtol` times its
        initial value or below `newton_gtol`, when the free-boundary target is below the tolerance, or when the
        update budget is used up. A small step or a failed line search is a step-size criterion only: a step in the
        stiff modes can be micrometres while |g| is still far above what the problem reaches. With `newton_gtol` > 0
        (absolute, in the units of |g|), a small step therefore does not stop the update while |g| > `newton_gtol`
        unless it was computed from a freshly built finite-difference Jacobian and did not reduce |g| by more than a
        factor `newton_rebuild_ratio` (then the run stops with a warning that it stalled above `newton_gtol`). A
        rejected step always rebuilds a Broyden Jacobian before the noise-floor stop is considered.
        Returns the last update index."""
        fbp = self._state_parameters["free_boundary"]
        h = fbp.get("newton_fd_step", 1e-4)
        xtol = fbp.get("newton_xtol", 1e-6)
        rtol = fbp.get("newton_rtol", 1e-8)
        gtol = fbp.get("newton_gtol", 0.0)
        max_step = fbp.get("newton_max_step", 0.05)
        lm = fbp.get("newton_lm", 1e-12)
        max_backtracks = fbp.get("newton_max_backtracks", 6)
        save_jacobian = fbp.get("newton_save_jacobian", False)
        workers = int(fbp.get("newton_workers", 1))
        space = fbp.get("newton_space", "normal")
        noise_step = fbp.get("newton_noise_step", 1e-5)
        rebuild_ratio = fbp.get("newton_rebuild_ratio", 0.5)

        self.newton_history = []
        # the finite differences need a converged base state
        self.state.reset_descent()
        self.state.solve(logfinaliter=False, maxIter=maxIter, minimize_tol=minimize_tol)
        self.state._checkpoint_dofs()
        r, fields = self._fb_residual(digits_BIEST)

        def basis():
            if space == "normal":
                return _normal_displacement_basis(self.state)
            if space == "full":
                return np.eye(r.size), np.ones(r.size)
            raise ValueError(f"Unknown newton_space: {space}")

        P, ynorm = basis()  # ynorm: max over the boundary of |xi| per unit y_k
        g = P.T @ r
        g_init = np.linalg.norm(g)
        k = P.shape[1]
        J = None
        self._newton_last_fd_jacobian = None
        # optional start Jacobian (a newton_jacobian_*.npz of a run with the same resolution and newton_space),
        # e.g. the RMP-off Jacobian for an RMP amplitude scan
        if (jac_file := fbp.get("newton_initial_jacobian")) is not None:
            J = np.load(jac_file)["J"]
            if J.shape != (k, k):
                raise ValueError(f"newton_initial_jacobian has shape {J.shape}, expected {(k, k)}")
            self._newton_last_fd_jacobian = J
            logger.info(f"Newton: initial Jacobian from {jac_file}")
        jac_fresh = False
        n_jac = 0
        fb_update = 0

        def fd_jacobian():
            nonlocal n_jac
            t0 = time.time()
            perts = h * P
            if workers > 1:
                dR = self._fd_jacobian_parallel(perts, blend_depth, minimize_tol, maxIter, digits_BIEST, workers)
            else:
                dR = np.zeros((r.size, k))
                for j in range(k):
                    if not self._fb_move_and_solve(perts[:, j], blend_depth, minimize_tol, maxIter):
                        raise RuntimeError(f"Newton FD Jacobian: invalid state for unknown {j}")
                    dR[:, j] = self._fb_residual(digits_BIEST)[0] - r
                self.state._restore_dofs()
            Jfd = P.T @ dR / h
            self._newton_last_fd_jacobian = Jfd
            n_jac += 1
            logger.info(f"Newton: FD Jacobian ({k} columns, {workers} workers) in {time.time() - t0:.1f}s")
            if save_jacobian:
                m, nn, _ = self.state._get_boundary_forces()
                np.savez(
                    f"newton_jacobian_{n_jac:03d}.npz",
                    J=Jfd, g=g, r=r, P=P, x=self.state._get_boundary_dofs(), m=m, n=nn, h=h,
                )
            return Jfd

        def lm_step(J, g, lam):
            JtJ = J.T @ J
            D = np.diag(np.diag(JtJ)).copy()
            D[D == 0] = 1.0
            return np.linalg.solve(JtJ + lam * D, -J.T @ g)

        for fb_update in range(1, fb_updates):
            logger.info(f"============= Newton free-boundary update {fb_update} / {fb_updates} =============")
            if J is None:
                J = fd_jacobian()
                jac_fresh = True
            lam = lm
            accepted = False
            for attempt in range(3):
                dy = lm_step(J, g, lam)
                dmax = np.max(np.abs(dy) * ynorm)
                if dmax > max_step:
                    dy *= max_step / dmax
                alpha = 1.0
                for _ in range(max_backtracks):
                    sy = alpha * dy
                    if self._fb_move_and_solve(P @ sy, blend_depth, minimize_tol, maxIter):
                        r_new, fields_new = self._fb_residual(digits_BIEST)
                        g_new = P.T @ r_new
                        if np.linalg.norm(g_new) < (1.0 - 1e-4 * alpha) * np.linalg.norm(g):
                            accepted = True
                            break
                    alpha *= 0.5
                if accepted:
                    break
                if np.max(np.abs(dy) * ynorm) < noise_step and jac_fresh:
                    logger.info(
                        "Newton step below newton_noise_step rejected with a fresh FD Jacobian: |g| is at its noise "
                        "floor (inner tolerance, BIEST accuracy); stopping."
                    )
                    break
                if not jac_fresh:
                    logger.info("Newton step rejected with a Broyden Jacobian: rebuilding the FD Jacobian.")
                    self.state._restore_dofs()
                    J = fd_jacobian()
                    jac_fresh = True
                else:
                    lam = max(100.0 * lam, 1e-8)
                    logger.info(f"Newton step rejected: increasing the Levenberg-Marquardt parameter to {lam:.1e}.")
            if not accepted:
                self.state._restore_dofs()
                if np.max(np.abs(dy) * ynorm) >= noise_step:
                    logger.warning("Newton free boundary: no decrease of |g| along the step; stopping.")
                elif gtol > 0.0 and np.linalg.norm(g) > gtol:
                    logger.warning(
                        f"Newton free boundary: stalled above newton_gtol, |g| {np.linalg.norm(g):.3e} > {gtol:.3e}."
                    )
                fb_update -= 1
                break
            # Broyden update and new base point; the basis follows the boundary. A Broyden Jacobian that
            # reduces |g| by less than newton_rebuild_ratio per step is replaced by a new FD Jacobian.
            slow = np.linalg.norm(g_new) > rebuild_ratio * np.linalg.norm(g)
            step_fresh = jac_fresh
            J = J + np.outer((g_new - g) - J @ sy, sy) / (sy @ sy)
            if slow and not jac_fresh:
                logger.info("Newton: slow decrease of |g| with a Broyden Jacobian; rebuilding the FD Jacobian.")
                J = None
            jac_fresh = False
            r, fields = r_new, fields_new
            self.state._checkpoint_dofs()
            P, ynorm = basis()
            g = P.T @ r
            self.xnorm = float(np.max(np.abs(sy) * ynorm))
            self.gnorm = float(np.linalg.norm(g))
            self._update_diagnostics(*fields, fb_update, alpha)
            self.iters_used = self.state.get_iterations()
            self.newton_history.append(
                dict(
                    update=fb_update,
                    residual=self.gnorm,
                    residual_rel=self.gnorm / g_init,
                    step_max=self.xnorm,
                    alpha=alpha,
                    target=self.free_boundary_target,
                    jacobians=n_jac,
                )
            )
            logger.info(
                f"Newton: |g| {self.gnorm:.3e} (rel {self.gnorm / g_init:.2e}), max normal-mode step {self.xnorm:.2e} m, "
                f"alpha {alpha:.3f}, target {self.free_boundary_target:.3e}, FD Jacobians {n_jac}"
            )
            if self.gnorm < rtol * g_init:
                logger.info("Newton free boundary: |g| below newton_rtol.")
                break
            if gtol > 0.0 and self.gnorm < gtol:
                logger.info("Newton free boundary: |g| below newton_gtol.")
                break
            if self.xnorm < xtol:
                if gtol <= 0.0 or self.gnorm < gtol:
                    logger.info("Newton free boundary: boundary step below newton_xtol.")
                    break
                if step_fresh and slow:
                    logger.warning(
                        f"Newton free boundary: step below newton_xtol from a fresh FD Jacobian without a decrease "
                        f"of |g| ({self.gnorm:.3e} > newton_gtol {gtol:.3e}): stalled; stopping."
                    )
                    break
                if not step_fresh:
                    logger.info("Newton: step below newton_xtol but |g| above newton_gtol; rebuilding the FD Jacobian.")
                    J = None
            if self.free_boundary_target <= free_boundary_tolerance:
                logger.info("Newton free boundary: free-boundary target below tolerance.")
                break
        # final (Broyden-updated) Jacobian, e.g. to be saved with np.savez(path, J=...) and passed as
        # newton_initial_jacobian to a related run; the run directory is removed after the stage
        self.newton_jacobian = J if J is not None else self._newton_last_fd_jacobian
        return fb_update

    def plot_diagnostics_minimization(self):
        import matplotlib.pyplot as plt
        from gvec.plotting.plots1d import _add_rationals_to_iota_plot

        fig_fb, axs_fb = plt.subplots(5, 1, figsize=(10, 10), layout="constrained", sharex=True)
        quantities = [
            "free_boundary_target",
            "free_boundary_target_mavg",
            "Fnorm_X1_b",
            "Fnorm_X2_b",
            "resolved_force_ratio",
            "xnorm",
            "xnorm_mavg",
            "gnorm",
            "gnorm_mavg",
        ]
        if self.state.I_tor:
            fig_iota, ax_iota = plt.subplots()
            quantities.append("iota")
        diagnostics = self.diagnostics_stage

        updates_offset = 0
        for ith_stage, stage in enumerate(diagnostics):
            ds = xr.Dataset(diagnostics[stage])[quantities].dropna(dim="updates")
            # skip the setup stage with zero free boundary updates.
            if len(ds.updates) == 0:
                continue

            axs_fb[0].semilogy(
                ds.updates + updates_offset,
                ds.free_boundary_target,
                color=f"C{ith_stage}",
                label="value " + stage.replace("_", " "),
            )
            axs_fb[0].semilogy(
                ds.updates + updates_offset,
                ds.free_boundary_target_mavg,
                color=f"C{ith_stage}",
                linestyle="-.",
                label="mov. avg. " + stage.replace("_", " "),
            )

            axs_fb[1].semilogy(
                ds.updates + updates_offset,
                ds.Fnorm_X1_b,
                color=f"C{ith_stage}",
                label=r"$X^1$ " + stage.replace("_", " "),
            )
            axs_fb[1].semilogy(
                ds.updates + updates_offset,
                ds.Fnorm_X2_b,
                "--",
                color=f"C{ith_stage}",
                label=r"$X^2$ " + stage.replace("_", " "),
            )
            axs_fb[2].semilogy(
                ds.updates + updates_offset,
                ds.resolved_force_ratio,
                color=f"C{ith_stage}",
                label=stage.replace("_", " "),
            )

            axs_fb[3].semilogy(
                ds.updates + updates_offset,
                ds.xnorm,
                color=f"C{ith_stage}",
                label="value " + stage.replace("_", " "),
            )
            axs_fb[3].semilogy(
                ds.updates + updates_offset,
                ds.xnorm_mavg,
                color=f"C{ith_stage}",
                linestyle="-.",
                label="mov. avg. " + stage.replace("_", " "),
            )

            axs_fb[4].semilogy(
                ds.updates + updates_offset,
                ds.gnorm,
                color=f"C{ith_stage}",
                label="value " + stage.replace("_", " "),
            )
            axs_fb[4].semilogy(
                ds.updates + updates_offset,
                ds.gnorm_mavg,
                color=f"C{ith_stage}",
                linestyle="-.",
                label="mov. avg. " + stage.replace("_", " "),
            )

            updates_offset += ds.updates.max()
            if self.state.I_tor:
                max_iter = ds.updates.max()
                for update in ds.updates:
                    ds_aux = ds.sel(updates=update)
                    ax_iota.plot(
                        ds_aux.iota.rho_iota,
                        ds_aux.iota,
                        "-.",
                        color=f"C{ith_stage}",
                        alpha=min(0.01 + (update / (5 * max_iter)).item(), 1),
                    )
                ax_iota.plot(
                    ds_aux.iota.rho_iota,
                    ds_aux.iota,
                    color=f"C{ith_stage}",
                    label=r"final $\iota$ " + stage.replace("_", " "),
                )
        if self.state.I_tor:
            ax_iota.set(xlabel=r"$\rho$", ylabel=r"$\iota$")
            _add_rationals_to_iota_plot(ax_iota, nfp=self.state.nfp)
            ax_iota.legend()
        axs_fb[0].set(ylabel="free boundary target")
        axs_fb[1].set(ylabel=r"$\left\|\hat{\mathbf{F}}_\text{b}\right\|^\text{max}$")
        axs_fb[2].set(ylabel="resolved force ratio")
        axs_fb[3].set(ylabel="xnorm")
        axs_fb[4].set(ylabel="gnorm", xlabel="free boundary update")

        for ax in axs_fb:
            ax.grid("on")
            ax.legend(ncol=2)
        out = ((fig_fb, axs_fb),)
        if self.state.I_tor:
            out += ((fig_iota, ax_iota),)
        return out

    def plot_diagnostics_fixed_boundary(self):
        import matplotlib.pyplot as plt

        dm = self.diagnostics_minimizer
        fig, axs = plt.subplots(2, 1, tight_layout=True, sharex=True)
        axs[0].plot(dm.total_iteration, dm.W_MHD)
        axs[1].semilogy(dm.total_iteration, dm.force_X1, label=r"$X^1$")
        axs[1].semilogy(dm.total_iteration, dm.force_X2, label=r"$X^2$")
        axs[1].semilogy(dm.total_iteration, dm.force_LA, label=r"$\lambda$")
        axs[0].set(ylabel=r"$W_\text{MHD}$")
        axs[1].set(
            xlabel="fixed boundary iteration",
            ylabel=r"$\left\|\hat{\mathbf{F}}_\text{i}\right\|^\text{max}$",
        )
        axs[1].legend()
        return (fig, axs)


def _descent_step_size(dt, target_mavg_last, target_mavg, min_dt, max_dt):
    """Step-size heuristic of the descent update: halve dt (down to min_dt) when the moving-average target
    increased, else grow it by 10 % (up to max_dt). Returns (dt, reset_minimizer); the free-boundary minimizer is
    reset (its memory, e.g. the mGD momentum or the lBFGS pairs, is wiped) when dt is held at min_dt."""
    if target_mavg_last < target_mavg or dt < min_dt:
        dt = max(0.5 * dt, min_dt)
    else:
        dt = min(1.1 * dt, max_dt)
    return dt, dt <= min_dt


# ============= NEWTON HELPERS ===========================
def _boundary_residual(state: MinimizationState, coil_set, digits_BIEST: int):
    """Boundary force of `state` with B_ext (coils + BIEST) on its current boundary.
    Returns (residual, (B_ext_2, B_coils, B_plas_ext), I_tor at the boundary)."""
    I_tor = state.evaluate("I_tor", rho=1.0).squeeze()["I_tor"].item()
    (B2, B_plas_ext, B_coils, B_ext_2) = _get_B2_fp(
        state=state,
        coil_set=coil_set,
        ntheta=state.t_n,
        nzeta=state.z_n,
        I_tor=I_tor,
        diagnostic_output=True,
        digits=digits_BIEST,
    )
    state.set_Bext_squared(B2)
    return state._eval_boundary_forces(), (B_ext_2, B_coils, B_plas_ext), I_tor


def _move_and_solve(state: MinimizationState, pert, blend_depth, minimize_tol, maxIter):
    """From the checkpointed state, add the boundary perturbation `pert` and re-solve the interior.
    Returns False (state restored) if the perturbed state is not valid (detJ <= 0)."""
    state._restore_dofs()
    if not state._apply_boundary_perturbation(pert, blend_depth):
        state._restore_dofs()
        return False
    state.reset_descent()
    state.solve(logfinaliter=False, maxIter=maxIter, minimize_tol=minimize_tol)
    return True


def _allowed_cpus():
    """CPUs available to this process: the cgroup (v2) cpuset if readable, else the parent's or own affinity."""
    import os

    def parse(text):
        cpus = []
        for part in text.strip().split(","):
            if "-" in part:
                a, b = part.split("-")
                cpus += list(range(int(a), int(b) + 1))
            elif part:
                cpus.append(int(part))
        return cpus

    try:
        rel = Path("/proc/self/cgroup").read_text().splitlines()[0].split(":", 2)[2]
        cpus = parse((Path("/sys/fs/cgroup" + rel) / "cpuset.cpus.effective").read_text())
        if cpus:
            return sorted(cpus)
    except (OSError, IndexError, ValueError):
        pass
    for pid in (os.getppid(), 0):
        try:
            return sorted(os.sched_getaffinity(pid))
        except OSError:
            continue
    return [0]


class _FDWorkerPool:
    """Persistent worker processes for the Newton FD Jacobian, scheduled dynamically (each worker takes the next
    column as soon as it is done with the previous one).

    `workers` processes run single-threaded, each pinned to its own CPU of the CPUs available to this process
    (cgroup), not of this thread's affinity: with OMP_PROC_BIND the master thread is bound to one core, and the
    workers would inherit that mask. The main process is idle during the FD Jacobian: if more CPUs are available
    than `workers`, one more worker runs on up to OMP_NUM_THREADS (of the main process) of the remaining CPUs, with as
    many OpenMP threads, so that the pool uses the CPUs of the workers and of the main process. Plain subprocesses (no multiprocessing pool): no shared semaphores and no re-import of the driver
    script. Commands and results are pickled frames on dedicated pipes; the GVEC output goes to the worker
    directories.
    """

    def __init__(self, workers, params, coil_set, blend_depth, minimize_tol, maxIter, digits, root):
        import os
        import pickle
        import subprocess
        import sys

        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        task = self.root / "task.pkl"
        task.write_bytes(pickle.dumps((params, coil_set, blend_depth, minimize_tol, maxIter, digits)))
        cpus = _allowed_cpus()
        sets = [[cpus[i % len(cpus)]] for i in range(workers)]
        main_threads = max(1, int(os.environ.get("OMP_NUM_THREADS", "1").split(",")[0] or 1))
        if len(cpus) > workers:
            sets.append(cpus[workers : workers + main_threads])
        self.procs, self.inp, self.out = [], [], []
        for i, cpuset in enumerate(sets):
            wdir = self.root / f"worker_{i:03d}"
            wdir.mkdir(parents=True, exist_ok=True)
            env = dict(os.environ, OMP_NUM_THREADS=str(len(cpuset)), OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
            env["OMP_PROC_BIND"] = "false"
            env.pop("OMP_PLACES", None)
            r_cmd, w_cmd = os.pipe()
            r_res, w_res = os.pipe()
            cmd = [sys.executable, "-c", "from gvec.core.run.free_boundary import _fd_worker_main; _fd_worker_main()",
                   str(task), str(wdir), str(r_cmd), str(w_res)]
            with open(wdir / "worker.log", "w") as log:
                self.procs.append(
                    subprocess.Popen(
                        cmd, env=env, stdout=log, stderr=subprocess.STDOUT, pass_fds=(r_cmd, w_res),
                        preexec_fn=lambda cpuset=cpuset: os.sched_setaffinity(0, cpuset),
                    )
                )
            os.close(r_cmd)
            os.close(w_res)
            self.inp.append(os.fdopen(w_cmd, "wb"))
            self.out.append(os.fdopen(r_res, "rb"))

    def _send(self, i, msg):
        import pickle

        pickle.dump(msg, self.inp[i])
        self.inp[i].flush()

    def jacobian(self, statefile, perts):
        """Residual differences (n_dofs x k) for the columns of `perts` from the base state in `statefile`."""
        import pickle
        import selectors

        k = perts.shape[1]
        dR = np.zeros((perts.shape[0], k))
        todo = list(range(k))
        sel = selectors.DefaultSelector()
        busy = 0
        stats = []  # (worker, base seconds, column seconds, column iterations)
        t0 = time.time()
        for i in range(len(self.procs)):
            self._send(i, ("base", str(statefile)))
            if todo:
                j = todo.pop(0)
                self._send(i, ("col", j, perts[:, j]))
                sel.register(self.out[i], selectors.EVENT_READ, i)
                busy += 1
        try:
            while busy:
                for key, _ in sel.select():
                    i = key.data
                    try:
                        msg = pickle.load(self.out[i])
                    except EOFError:
                        msg = ("error", f"worker {i} exited (code {self.procs[i].wait()})")
                    if msg[0] == "error":
                        raise RuntimeError(f"Newton FD worker {i} failed: {msg[1]}; see {self.root}/worker_*/worker.log")
                    _, j, col, info = msg
                    dR[:, j] = col
                    stats.append((i, *info))
                    if todo:
                        j = todo.pop(0)
                        self._send(i, ("col", j, perts[:, j]))
                    else:
                        sel.unregister(self.out[i])
                        busy -= 1
        finally:
            sel.close()
        tb = [st[1] for st in stats if st[1] is not None]
        tc = np.array([st[2] for st in stats])
        it = np.array([st[3] for st in stats])
        logger.info(
            f"FD worker pool: {k} columns on {len(self.procs)} workers in {time.time() - t0:.1f}s; base restart max "
            f"{max(tb, default=0.0):.1f}s; column min/mean/max {tc.min():.1f}/{tc.mean():.1f}/{tc.max():.1f}s, "
            f"iterations min/mean/max {it.min()}/{it.mean():.0f}/{it.max()}"
        )
        return dR

    def close(self):
        for i, p in enumerate(self.procs):
            try:
                if p.poll() is None:
                    self._send(i, ("stop",))
                self.inp[i].close()
            except OSError:
                pass
        for p in self.procs:
            try:
                p.wait(timeout=60)
            except Exception:
                p.kill()
        for f in self.out:
            f.close()
        self.procs, self.inp, self.out = [], [], []


def _fd_worker_main():
    """Entry point of a Newton FD worker process: argv = task.pkl, worker directory, command fd, result fd.
    Commands: ("base", statefile) restarts from the statefile, re-solves (to minimize_tol) and evaluates the base
    residual; ("col", j, pert) returns ("col", j, residual(base + pert) - base residual, (base restart seconds or
    None, column seconds, column iterations)); ("stop",) exits."""
    import os
    import pickle
    import sys
    import traceback

    params, coil_set, blend_depth, minimize_tol, maxIter, digits = pickle.loads(Path(sys.argv[1]).read_bytes())
    wdir = Path(sys.argv[2])
    inp = os.fdopen(int(sys.argv[3]), "rb")
    out = os.fdopen(int(sys.argv[4]), "wb")
    state, r0 = None, None
    nbase = 0
    try:
        while True:
            try:
                msg = pickle.load(inp)
            except EOFError:
                break
            if msg[0] == "stop":
                break
            if msg[0] == "base":
                nbase += 1  # binding the new state releases the previous one (with_binding)
                rundir = wdir / f"base_{nbase:03d}"
                rundir.mkdir(parents=True, exist_ok=True)
                state = MinimizationState.new(
                    params, rundir=rundir, statefile=msg[1], redirect_stdout=True, stdout_path=rundir / "stdout.txt"
                )
                t_base = time.time()
                with chdir(rundir):
                    state.reset_descent()
                    state.solve(logfinaliter=False, maxIter=maxIter, minimize_tol=minimize_tol)
                    state._checkpoint_dofs()
                    r0 = _boundary_residual(state, coil_set, digits)[0]
                t_base = time.time() - t_base
            elif msg[0] == "col":
                _, j, pert = msg
                t_col, it0 = time.time(), state.get_iterations()
                with chdir(state.rundir):
                    if not _move_and_solve(state, pert, blend_depth, minimize_tol, maxIter):
                        raise RuntimeError(f"Newton FD Jacobian: invalid state for unknown {j}")
                    col = _boundary_residual(state, coil_set, digits)[0] - r0
                info = (t_base, time.time() - t_col, state.get_iterations() - it0)
                t_base = None  # reported with the first column after a base restart
                pickle.dump(("col", j, col, info), out)
                out.flush()
    except Exception:
        traceback.print_exc()
        try:
            pickle.dump(("error", traceback.format_exc().splitlines()[-1]), out)
            out.flush()
        except OSError:
            pass
        sys.exit(1)


def _boundary_mode_list(state: MinimizationState):
    """[(var, sc, m, n)] for the boundary dofs in the order of _get_boundary_forces (sine block first);
    n is the toroidal mode number per field period (the basis function is sin/cos(m theta - n nfp zeta))."""
    m, n, _ = state._get_boundary_forces()
    n = n // state.nfp  # the Fortran mode table holds n * nfp
    n1 = state._get_X1_n_modes()
    out = []
    for var, sl in (("X1", slice(0, n1)), ("X2", slice(n1, m.size))):
        mm, nn = m[sl], n[sl]
        zero = np.nonzero((mm == 0) & (nn == 0))[0]
        cos_start = zero[0] if zero.size else mm.size
        if "_cos_" == state.parameters.get(f"{var}_sin_cos", "_sincos_"):
            cos_start = 0
        for i in range(mm.size):
            out.append((var, "s" if i < cos_start else "c", int(mm[i]), int(nn[i])))
    return out


def _normal_displacement_basis(state: MinimizationState, oversample: int = 4):
    """Matrix P (n_dofs x k): column k holds the boundary coefficients of the in-plane normal displacement
    xi = phi_k (unit normal of the boundary curve in each zeta = const plane), phi_k running over the Fourier
    functions of the X1 boundary basis. Also returns max |phi_k| (= 1). Projection by trapezoidal quadrature on a
    uniform grid that resolves the products exactly up to `oversample` times the maximum mode numbers."""
    modes = _boundary_mode_list(state)
    nfp = state.nfp
    mmax = max(md[2] for md in modes)
    nmax = max(abs(md[3]) for md in modes)
    nt = oversample * (mmax + 1) + 1
    nz = 1 if nmax == 0 else oversample * (nmax + 1) + 1
    theta = np.arange(nt) * 2 * np.pi / nt
    zeta = np.arange(nz) * 2 * np.pi / nfp / nz
    ev = state.evaluate("X1", "X2", "dX1_dt", "dX2_dt", rho=[1.0], theta=theta, zeta=zeta)
    ev = ev.isel(rad=0)
    Rt = ev.dX1_dt.transpose("pol", "tor").values
    Zt = ev.dX2_dt.transpose("pol", "tor").values
    norm = np.hypot(Rt, Zt)
    nR, nZ = Zt / norm, -Rt / norm
    TH, ZE = np.meshgrid(theta, zeta, indexing="ij")

    def fun(sc, m, n):
        arg = m * TH - n * nfp * ZE
        return np.sin(arg) if sc == "s" else np.cos(arg)

    phis = [fun(sc, m, n) for (var, sc, m, n) in modes]
    x1 = [i for i, md in enumerate(modes) if md[0] == "X1"]
    P = np.zeros((len(modes), len(x1)))
    for col, k in enumerate(x1):
        fR, fZ = phis[k] * nR, phis[k] * nZ
        for i, md in enumerate(modes):
            f = fR if md[0] == "X1" else fZ
            P[i, col] = np.sum(f * phis[i]) / np.sum(phis[i] ** 2)
    return P, np.ones(len(x1))


# ============= BIEST CALL ===========================
def _get_B2_fp(
    state: State,
    coil_set: CoilSet | Coil,
    ntheta: int,
    nzeta: int,
    I_tor: float,
    diagnostic_output: bool,
    digits: int,
):
    """Setup BIEST with points corresponding to one field period and return the flattened squared exterior field for free boundary gvec.

    Parameters
    ----------
    state : State
        GVEC state object. This is used to extract the boundary points
    coil_set : CoilSet | Coil
        Free-boundary coil set. The coil magnetic field is evaluated at the boundary-integration points.
    ntheta : int
        Number of theta integration points.
    nzeta : int
        Number of zeta integration points.
    I_tor : float
        Toroidal current at the boundary.
    diagnostic_output : bool, optional
        Debug-flag to enable the return of the intermediate fields, by default False.
        If True B2_flat, B_plas_ext, B_coils, B2 for one field period are returned

    Returns
    -------
    np.ndarray(B2_ext)
        Squared exterior field at the GVEC integration points flattened in GVEC order.
    """

    try:
        from fbtools.exterior_vacuum_field import get_B_ext_plasma
    except ImportError as e:
        raise ImportError(
            "Could not import fbtools. Was free-boundary gvec (`gvec[free_boundary]`) installed?"
        ) from e

    # BIEST needs at least BIEST_MIN_NZETA toroidal points per field period (axisymmetric states have a
    # single GVEC integration point). Evaluate on an odd multiple k of the GVEC points, so that the GVEC
    # (cell-centered) points are a subset of the BIEST points, and sample the result back.
    k_zeta = 1
    while k_zeta * nzeta < BIEST_MIN_NZETA:
        k_zeta += 2
    nzeta_biest = k_zeta * nzeta
    zeta_idx = k_zeta * np.arange(nzeta) + (k_zeta - 1) // 2

    # cell-centered points; the spacing must not be read from the points themselves
    theta = (np.arange(ntheta) + 0.5) * (2 * np.pi / ntheta)
    zeta = (np.arange(nzeta_biest) + 0.5) * (2 * np.pi / state.nfp / nzeta_biest)

    # get the boundary for the full device (=all field periods)
    boundary = state.evaluate("pos", rho=1.0, zeta=zeta, theta=theta)
    boundary = boundary.sel(rho=1).transpose("xyz", "tor", "pol")

    # coil-field at the boundary
    ev_coils = coil_set.eval_B(boundary.pos)
    B_coils = ev_coils.B.transpose("xyz", "tor", "pol").values
    if state.clockwise_zeta:
        boundary_biest = np.flip(boundary.pos.values, axis=1)
        B_coils_biest = np.flip(B_coils.copy(), axis=1)
    else:
        boundary_biest = boundary.pos.values
        B_coils_biest = B_coils.copy()

    # BIEST imposes the circulation I_biest along its poloidal index (theta) when the tangent of its toroidal index
    # crossed with e_theta points out of the plasma, and -I_biest otherwise. Its toroidal index runs along +phi (the
    # field periods are completed by rotations about +z). GVEC's (rho, theta, zeta) is right-handed, so
    # e_theta x e_zeta points outwards: with clockwise zeta the toroidal tangent is -e_zeta and the orientation is
    # outward, with counter-clockwise zeta it is e_zeta and inward, which reverses the sign of the current. GVEC's
    # I_tor is the circulation of B along theta in both cases.
    I_biest = mu_0 * I_tor if state.clockwise_zeta else -mu_0 * I_tor

    # main BIEST calculation
    time_0 = time.time()
    B_plas_ext = get_B_ext_plasma(
        boundary_biest,
        state.nfp,
        B_coils_biest,
        I_biest,
        digits=digits,
    )
    time_1 = time.time()
    logger.info(
        f"time for BIEST: {time_1 - time_0:.3}s for {ntheta * nzeta_biest} points"
    )
    if state.clockwise_zeta:
        B_plas_ext = np.flip(B_plas_ext, axis=1)

    # back to the GVEC integration points
    B_plas_ext = B_plas_ext[:, zeta_idx, :]
    B_coils = B_coils[:, zeta_idx, :]

    # Bext^2
    B_external = B_plas_ext + B_coils

    B2 = np.sum(B_external**2, axis=0)

    B2_flat = np.asfortranarray(B2.flatten(order="C"))
    if diagnostic_output:
        return (
            B2_flat,
            B_plas_ext,
            B_coils,
            B2,
        )
    else:
        return B2_flat


# ========= RESOLVED BOUNDARY FORCES ========================
def minimal_modes_force_balance(X, tolerance=1e-8, **kwargs_fft2d):
    """
    Find the minimal maximum mode numbers (M, N) such that the error is below the tolerance.
    First dimension of X to be theta,
    second dimension is assumed to be zeta.
    """
    Xcos, Xsin = fourier.fft2d(X, **kwargs_fft2d)

    M, N = Xcos.shape[0] - 1, Xcos.shape[1] // 2

    m, n = fourier.fft2d_modes(M, N, grid=True)
    Mrange, Nrange = np.arange(M + 1), np.arange(0, N + 1)
    error = np.full((M + 1, N + 1), np.inf)
    norm = np.sqrt(np.sum(Xcos**2 + Xsin**2))
    for Mnew in Mrange:
        for Nnew in Nrange:
            # sum magnitudes of all modes above the cutoff
            mask = (m > Mnew) | (n > Nnew) | (n < -Nnew)
            err = Xcos[mask] ** 2 + Xsin[mask] ** 2
            error[Mnew, Nnew] = np.sqrt(np.sum(err)) / norm

    # select candidates with error below the tolerance
    mcan, ncan = np.meshgrid(Mrange, Nrange, indexing="ij")
    mask = error < tolerance
    # restrict candidates to those with minimum DoFs
    dofs = ncan + 1 + mcan * (2 * ncan + 1)
    mask &= dofs == dofs[mask].min()
    # select candidate with minimum error
    mask &= error == error[mask].min()

    return mcan[mask].item(), ncan[mask].item()
