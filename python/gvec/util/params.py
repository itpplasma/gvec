# Copyright (c) 2025 GVEC Contributors, Max Planck Institute for Plasma Physics
# License: MIT
import re
import shutil
from collections.abc import Iterable, Mapping, MutableMapping
from numbers import Integral, Real
from copy import deepcopy
from pathlib import Path
from typing import Literal
import warnings

import numpy as np
from numpy.typing import ArrayLike
from numpy.polynomial import Polynomial

# NOTE: this exception handling is relevenat for the CI pipeline, since some util methods are used without installing gvec and therefore scipy
try:
    from scipy.interpolate import BSpline, make_interp_spline
except ImportError:
    BSpline = None

from .util import logger

PARAMETER_SCALE_DEFAULT = 1.0
PARAMETER_BC_VALS_DEFAULT = (0, 0)
PARAMETER_BC_TYPE_DEFAULT = "not_a_knot"
PROFILE_INTERPOLATION_RHO2_MIN = 1e-4


class CaseInsensitiveDict(MutableMapping):
    # Adapted from requests.structures.CaseInsensitiveDict
    # See: https://github.com/psf/requests/blob/main/src/requests/structures.py
    # Original license: Apache License 2.0
    r"""A dictionary-like Mutable Mapping where string keys are case-insensitive.

    Implements all methods and operations of
    ``MutableMapping`` as well as dict's ``copy``. Also
    provides ``lower_items`` and ``lower_keys``.

    Keys that are not strings will be stored as-is.
    The structure remembers the case of the last used key, and
    ``iter(instance)``, ``keys()``, ``items()``, ``iterkeys()``, and ``iteritems()``
    will contain case-sensitive keys. However, querying and contains
    testing is case insensitive.

    Raises
    ------
    ValueError
        If the constructor, ``.update``, or equality comparison
        operations are given keys that have equal ``.lower()`` representations.
        This is to avoid ambiguity in lookups and ensure consistent behavior.

    Examples
    --------

    >>> cid = CaseInsensitiveDict()
    >>> cid['param'] = 'value'
    >>> cid['Param'] == 'value'
    True
    """

    def __init__(self, data=(), /, **kwargs):
        self._data = {}
        self.update(data, **kwargs)

    @classmethod
    def recursive(cls, data):
        """Recursively convert a mapping to a CaseInsensitiveDict. Also converts iterables to lists."""
        if isinstance(data, Mapping):
            return cls({k: cls.recursive(v) for k, v in data.items()})
        elif isinstance(data, Iterable) and not isinstance(data, str):
            return [cls.recursive(v) for v in data]
        else:
            return data

    @staticmethod
    def _idx(key):
        return key.lower() if isinstance(key, str) else key

    def __setitem__(self, key, value):
        # Use the lowercased key for lookups, but remember the last key alongside the value.
        self._data[self._idx(key)] = (key, value)

    def __getitem__(self, key):
        return self._data[self._idx(key)][1]

    def __delitem__(self, key):
        del self._data[self._idx(key)]

    def update(self, data=(), /, **kwargs):
        updates = {}
        updates.update(data, **kwargs)
        idxs = {self._idx(key) for key in updates}
        if len(idxs) != len(updates):
            raise ValueError("Duplicate keys passed to CaseInsensitiveDict.update")
        for key, value in updates.items():
            self[key] = value

    def __iter__(self):
        return (key for key, value in self._data.values())

    def lower_keys(self):
        return (idx for idx in self._data.keys())

    def lower_items(self):
        return ((idx, value) for idx, (key, value) in self._data.items())

    def __len__(self):
        return len(self._data)

    def __eq__(self, other):
        if isinstance(other, Mapping):
            other = CaseInsensitiveDict(other)
        else:
            return NotImplemented
        # Compare insensitively
        return dict(self.lower_items()) == dict(other.lower_items())

    def serialize(self):
        """Recursively serialize this object, converting Mappings to dicts and Iterables to lists.

        Deprecated: Use the standalone ``serialize`` function instead.
        """
        warnings.warn(
            "CaseInsensitiveDict.serialize has been replaced with a standalone serialize",
            DeprecationWarning,
        )

        return serialize(self)

    def __repr__(self):
        return f"{self.__class__.__name__}{dict(self.items())}"

    def copy(self):
        """Return a deep copy."""
        return deepcopy(self)

    def __or__(self, other):
        """Union/Merge operator 'a | b' (without modifying self)."""
        if not isinstance(other, Mapping):
            return NotImplemented
        result = CaseInsensitiveDict(self)
        result.update(other)
        return result

    def __ior__(self, other):
        """In-place union/merge operator 'a |= b' (modifies self)."""
        if not isinstance(other, Mapping):
            return NotImplemented
        self.update(other)
        return self


def serialize(value):
    """Recursively serialize a value.

    Will try to convert to (dict, list, str, int, float, bool)."""

    def mapping(v):
        return {key: serialize(value) for key, value in v.items()}

    def array(v):
        return v.tolist()

    def iterable(v):
        return [serialize(value) for value in v]

    def scalar(v):
        return v.item()

    # first try known types
    match value:
        case Mapping():
            return mapping(value)
        case np.ndarray():
            return array(value)
        case str() | bool():
            return value
        case Iterable():
            return iterable(value)
        case np.number():
            return scalar(value)
        case Integral():
            return int(value)
        case Real():
            return float(value)

    # try duck typing
    for converter in [mapping, array, iterable, scalar, float, str]:
        try:
            return converter(value)
        except (TypeError, AttributeError):
            continue
    raise TypeError(f"Cannot serialize value of type {type(value)}: {value}")


def adapt_parameter_file(source: str | Path, target: str | Path, **kwargs):
    """
    Copy the `source` file to the `target` file and replace the parameters according to ``kwargs``.

    Args:
        source (str or Path): The path to the source parameter file.
        target (str or Path): The path to the target parameter file.
        **kwargs: Keyword arguments representing the parameters to be replaced.
                  if the value of the key is "!", the line with the keyword is uncommented, if possible

    Raises:
        AssertionError: If the number of occurrences for any parameter is not exactly 1.

    Notes:
        - If no parameters are provided in ``kwargs``, the function simply copies the ``source`` file to the ``target`` file.
        - The function replaces the parameters in the format ``key = value``, where value is either a sequence of characters containing
          no whitespace or a single pair of parentheses with any content. The value from ``kwargs`` is inserted using the standard python
          string conversion. There may be a comment, starting with ``!``, after the value.
        - If a parameter already exists in the ``source`` file, its value is replaced with the corresponding value from ``kwargs``.
        - If a parameter does not exist in the ``source`` file, it is added to the ``target`` file.
        - If the value of the key starts with "!", the line with the keyword is just uncommented.  (i.e. ``"!key=2.5" -> "key=2.5"``)
          If no line with the keyword is found, the key is added with the value, excluding the leading ``"!"``  (i.e. value is ``"!0.5" -> "key=0.5"`` is added)

    Example:
        ``adapt_parameter_file('/path/to/source.ini', '/path/to/target.ini', param1=1.2, param2="(1, 2, 3)")``
    """
    if not len(kwargs.keys()):
        shutil.copy2(source, target)
        return

    for key, value in kwargs.items():
        if isinstance(value, Mapping) or isinstance(value, str):
            pass
        elif isinstance(value, bool):
            kwargs[key] = "T" if value else "F"
        elif isinstance(value, Iterable):
            kwargs[key] = f"(/{', '.join(map(str, value))}/)"
        else:
            kwargs[key] = str(value)
    kwargs = {key.lower(): value for key, value in kwargs.items()}

    # initialize occurrences counters for all parameters to be set
    occurrences = {}
    for key in kwargs:
        if isinstance(kwargs[key], Mapping):
            for m, n in kwargs[key]:
                occurrences[key, m, n] = 0
        else:
            occurrences[key] = 0

    with open(source, "r") as source_file:
        source_file = source_file.readlines()
    with open(target, "w") as target_file:
        for line in source_file:
            if m := re.match(
                r"\s*([^!=\s\(]+)\s*\(\s*([-\d]+);\s*([-\d]+)\)\s*=\s*([-+\d\.Ee]+)",
                line,
            ):
                key, *mn, value = m.groups()
                if key.lower() in kwargs:
                    if (int(mn[0]), int(mn[1])) in kwargs[key.lower()]:
                        line = f"{key}({mn[0]};{mn[1]}) = {kwargs[key.lower()][(int(mn[0]), int(mn[1]))]}\n"
                        occurrences[key.lower(), int(mn[0]), int(mn[1])] += 1
            elif m := re.match(
                r"([\s!]*)("
                + "|".join(
                    [
                        key.lower()
                        for key, value in kwargs.items()
                        if not isinstance(value, Mapping)
                    ]
                )
                + r")(\s*=\s*)(\([^\)]*\)|[^!\s]*)(.*)",
                line,
                re.IGNORECASE,
            ):
                prefix, key, sep, value, suffix = m.groups()
                if "!" in prefix:  # found commented keyword
                    if str(kwargs[key.lower()])[0] == "!":  # only uncomment keyword
                        line = f"{key}{sep}{value}{suffix}\n"
                        occurrences[key.lower()] += 1
                else:  # found uncommented keywords
                    if str(kwargs[key.lower()])[0] != "!":  # use new keyword
                        line = f"{prefix}{key}{sep}{kwargs[key.lower()]}{suffix}\n"
                        occurrences[key.lower()] += 1
                    else:  # use the existing keyword,value pair with a comment
                        line = (
                            f"{prefix}{key}{sep}{value} !!WAS ALREADY UNCOMMENTED!! {suffix}\n"
                        )
                        occurrences[key.lower()] += 1
            target_file.write(line)
        # add key,value pair if not existing in parameterfile.
        for key, o in occurrences.items():
            if o == 0:
                if isinstance(key, tuple):
                    key, m, n = key
                    if str(kwargs[key][m, n]) != "!":
                        target_file.write(f"\n{key}({m};{n}) = {kwargs[key][m, n]}")
                        occurrences[key, m, n] += 1
                else:
                    if str(kwargs[key]) == "!":
                        continue  # ignore 'uncomment' value if key is not found
                    elif str(kwargs[key])[0] == "!":
                        # use default value '!default' if key is not found
                        target_file.write(f"\n{key} = {kwargs[key][1:]}")
                    else:
                        # add parameter at the end if key is not found
                        target_file.write(f"\n{key} = {kwargs[key]}")
                    occurrences[key] += 1
    assert all([o == 1 for o in occurrences.values()]), (
        f"bad number of occurrences in adapt_parameter_file: {occurrences}"
    )


def write_parameter_file_ini(
    parameters: Mapping, path: str | Path = "parameter.ini", header: str = ""
):
    """
    Write the parameters to the specified parameter file in GVEC-ini format.

    Args:
        parameters: A mapping containing the parameters to be written to the parameter file.
        path: The path to the parameter file.
    """
    parameters = parameters.copy()
    for key, value in parameters.items():
        if isinstance(value, Mapping) or isinstance(value, str):
            pass
        elif isinstance(value, bool):
            parameters[key] = "T" if value else "F"
        elif isinstance(value, Iterable):
            parameters[key] = f"(/{', '.join(map(str, value))}/)"
        else:
            parameters[key] = str(value)

    with open(path, "w") as file:
        file.write(header)
        for key, value in parameters.items():
            if isinstance(value, Mapping):
                for (m, n), val in value.items():
                    file.write(f"{key}({m};{n}) = {val}\n")
            else:
                file.write(f"{key} = {value}\n")


def read_parameter_file_ini(path: str | Path) -> CaseInsensitiveDict:
    """
    Read the parameters from the specified parameter file in GVEC-ini format.

    Args:
        path (str | Path): The path to the parameter file.

    Returns:
        CaseInsensitiveDict: A mapping (with case insensitive keys) containing the parameters from the parameter file.

    Example:
    >>> read_parameter_file_ini('/path/to/parameter.ini')
    {'param1': 1.2, 'param2': (1, 2, 3), 'param3': {(-1, 0): 0.5, (0, 0): 1.0}}
    """
    INT = r"[-+]?\d+"
    FLOAT = r"[-+]?\d*\.?\d*(?:[eE][-+]?\d+)?"
    STR = r"\S+"
    KEY = r"\w+"

    def convert(value: str):
        if "," in value:
            return tuple(convert(v) for v in value.split(","))
        if re.fullmatch(INT, value):
            return int(value)
        if re.fullmatch(FLOAT, value):
            return float(value)
        if value.upper() == "T":
            return True
        if value.upper() == "F":
            return False
        if re.fullmatch(STR, value):
            return value
        raise ValueError(f"Cannot parse value '{value}' in parameter file {path}")

    # follow the implementation in src/globals/readintools.f90:FillStrings
    parameters = CaseInsensitiveDict()
    with open(path, "r") as file:
        # read lines and preprocess them
        lines = []
        for line in file:
            # remove comments `!` and `#`
            line = re.split(r"[!#]", line)[0]
            # remove array brackets `(/` and `/)`
            line = re.sub(r"\(\/", "", line)
            line = re.sub(r"\/\)", "", line)
            # remove whitespace
            line = re.sub(r"\s+", "", line).strip()
            # skip empty lines
            if len(line) == 0:
                continue
            # combine lines that end with a `&`
            if lines and lines[-1].endswith("&"):
                lines[-1] = lines[-1][:-1] + line
            else:
                lines.append(line)

        # parse the lines
        for line in lines:
            # match parameter in the form `key(m;n) = value` with m,n integers
            if ln := re.fullmatch(rf"({KEY})\(({INT});({INT})\)=(.+)", line):
                key, m, n, value = ln.groups()
                m, n = int(m), int(n)

                if key in parameters and not isinstance(parameters[key], MutableMapping):
                    raise TypeError(
                        f"Trying to set indices for parameter '{key}' in {path}, but it is already set to a non-mapping value: {parameters[key]}"
                    )
                if key not in parameters:
                    parameters[key] = {}
                if (m, n) in parameters[key]:
                    raise IndexError(
                        f"Duplicate indices ({m}, {n}) for parameter '{key}' in {path}"
                    )
                parameters[key][m, n] = convert(value)
            # match parameter in the form `key = value`
            elif "=" in line:
                key, value = line.split("=", 1)
                if key in parameters:
                    raise IndexError(f"Duplicate parameter '{key}' in {path}")
                if not re.fullmatch(KEY, key):
                    raise ValueError(f"Invalid key '{key}' in parameter file {path}")
                parameters[key] = convert(value)
    return parameters


VMEC_PROFILE_SAMPLES = 401


def vmec_profile_grid() -> np.ndarray:
    """Normalised flux samples for converted VMEC profiles, clustered at both ends
    where VMEC profiles steepen; 401 values keep a parameter-file line below
    GVEC's 16384 characters."""
    return 0.5 * (
        1 - np.cos(np.pi * np.arange(VMEC_PROFILE_SAMPLES) / (VMEC_PROFILE_SAMPLES - 1))
    )


def _vmec_aux(nml: Mapping, prefix: str) -> tuple[np.ndarray, np.ndarray]:
    s = np.atleast_1d(np.asarray(nml.get(f"{prefix}_aux_s", []), dtype=float))
    f = np.atleast_1d(np.asarray(nml.get(f"{prefix}_aux_f", []), dtype=float))
    f = np.pad(f, (0, max(0, len(s) - len(f))))
    # VMEC ends the knot list at the first non-increasing abscissa (zero padding)
    n = len(s)
    for i in range(1, len(s)):
        if s[i] <= s[i - 1]:
            n = i
            break
    return s[:n], f[:n]


def _vmec_clamped_spline(s: np.ndarray, f: np.ndarray):
    from scipy.interpolate import CubicSpline

    # end slopes from the quadratic through the outer three points, as in LIBSTELL spline_cubic
    c = ((f[2] - f[0]) / (s[2] - s[0]) - (f[1] - f[0]) / (s[1] - s[0])) / (s[2] - s[1])
    yp1 = (f[1] - f[0]) / (s[1] - s[0]) - c * (s[1] - s[0])
    c = ((f[-3] - f[-1]) / (s[-3] - s[-1]) - (f[-2] - f[-1]) / (s[-2] - s[-1])) / (
        s[-3] - s[-2]
    )
    ypn = (f[-2] - f[-1]) / (s[-2] - s[-1]) - c * (s[-2] - s[-1])
    return CubicSpline(s, f, bc_type=((1, yp1), (1, ypn)))


def _vmec_sum_atan(c: np.ndarray, x: np.ndarray) -> np.ndarray:
    c = np.pad(c, (0, max(0, 21 - len(c))))
    x = np.minimum(x, 1.0)
    y = np.full_like(x, c[0])
    # ATAN2 takes each term to its limit at x = 1 for any exponent of (1 - x), as in PARVMEC
    for k in (1, 5, 9, 13, 17):
        y += 2 / np.pi * c[k] * np.arctan2(c[k + 1] * x ** c[k + 2], (1 - x) ** c[k + 3])
    return y


def _gauss_legendre_integral(fprime, x: np.ndarray) -> np.ndarray:
    nodes, weights = np.polynomial.legendre.leggauss(10)
    nodes, weights = 0.5 * (nodes + 1), 0.5 * weights
    return np.array([xi * np.sum(weights * fprime(xi * nodes)) for xi in x])


def vmec_profile(nml: Mapping, kind: Literal["pmass", "piota", "pcurr"]) -> np.ndarray:
    """Evaluate a VMEC input profile on ``s = vmec_profile_grid()``.

    Follows LIBSTELL ``profile_functions.f``: ``pmass`` without ``mu0*pres_scale``,
    ``piota`` as iota, and ``pcurr`` as the enclosed current I(s), not yet scaled to CURTOR.
    """
    from scipy.interpolate import Akima1DInterpolator

    prefix = {"pmass": "am", "piota": "ai", "pcurr": "ac"}[kind]
    ptype = str(nml.get(f"{kind}_type", "power_series")).strip().lower()
    c = np.atleast_1d(np.asarray(nml.get(prefix, [0.0]), dtype=float))
    c = np.nan_to_num(c)
    s = vmec_profile_grid()
    x = np.minimum(np.abs(s * nml.get("bloat", 1.0)), 1.0) if kind != "piota" else s
    aux_s, aux_f = _vmec_aux(nml, prefix)
    current_density = kind == "pcurr" and ptype in ("power_series", "two_power", "gauss_trunc")

    def line_segments(xv):
        return np.interp(xv, aux_s, aux_f) + np.where(
            xv > aux_s[-1],
            (xv - aux_s[-1]) * (aux_f[-1] - aux_f[-2]) / (aux_s[-1] - aux_s[-2]),
            0.0,
        )

    match ptype:
        case "power_series":
            if current_density:
                return x * Polynomial(c / np.arange(1, len(c) + 1))(x)
            return Polynomial(c)(x)
        case "power_series_i":
            return x * Polynomial(c)(x)
        case "two_power":

            def two_power(xv):
                return c[0] * (1 - xv ** c[1]) ** c[2]

            return _gauss_legendre_integral(two_power, x) if current_density else two_power(x)
        case "gauss_trunc":
            if current_density:
                return _gauss_legendre_integral(
                    lambda xv: c[0]
                    * (np.exp(-((xv / c[1]) ** 2)) - np.exp(-((1 / c[1]) ** 2))),
                    x,
                )
            return (
                c[0]
                / (1 - np.exp(-((1 / c[1]) ** 2)))
                * (np.exp(-((x / c[1]) ** 2)) - np.exp(-((1 / c[1]) ** 2)))
            )
        case "two_lorentz":

            def lorentz(width, power1, power2, xv):
                return 1 / (1 + (xv / width**2) ** power1) ** power2

            return c[0] * (
                c[1]
                * (lorentz(c[2], c[3], c[4], x) - lorentz(c[2], c[3], c[4], 1.0))
                / (1 - lorentz(c[2], c[3], c[4], 1.0))
                + (1 - c[1])
                * (lorentz(c[5], c[6], c[7], x) - lorentz(c[5], c[6], c[7], 1.0))
                / (1 - lorentz(c[5], c[6], c[7], 1.0))
            )
        case "sum_atan":
            return _vmec_sum_atan(c, x)
        case "line_segment" | "line_segment_i":
            return line_segments(x)
        case "line_segment_ip":
            fine = np.union1d(x, aux_s[(aux_s > 0) & (aux_s < 1)])
            prime = line_segments(fine)
            integral = np.concatenate(
                [[0.0], np.cumsum(0.5 * (prime[1:] + prime[:-1]) * np.diff(fine))]
            )
            return np.interp(x, fine, integral)
        case "cubic_spline" | "cubic_spline_i":
            return _vmec_clamped_spline(aux_s, aux_f)(x)
        case "cubic_spline_ip":
            return _vmec_clamped_spline(aux_s, aux_f).antiderivative()(x)
        case "akima_spline" | "akima_spline_i":
            return Akima1DInterpolator(aux_s, aux_f)(x)
        case "akima_spline_ip":
            return Akima1DInterpolator(aux_s, aux_f).antiderivative()(x)
        case _:
            raise ValueError(
                f"VMEC {kind} profile of type {ptype} is not supported for conversion"
            )


def parameters_from_vmec(nml: Mapping, name: str) -> CaseInsensitiveDict:
    def rectangular_boundary_values(value) -> np.ndarray:
        """Convert f90nml's sparse, possibly ragged boundary array to 2-D."""
        if isinstance(value, np.ndarray):
            rows = value.tolist()
        else:
            rows = value

        if not isinstance(rows, (list, tuple)):
            return np.asarray(rows, dtype=float).reshape(1, 1)
        if not rows:
            return np.empty((0, 0), dtype=float)
        if not all(isinstance(row, (list, tuple, np.ndarray)) for row in rows):
            return np.asarray(rows, dtype=float).reshape(-1, 1)

        width = max(len(row) for row in rows)
        rectangular = np.full((len(rows), width), np.nan)
        for row_index, row in enumerate(rows):
            for column_index, entry in enumerate(row):
                if entry is not None:
                    rectangular[row_index, column_index] = entry
        return rectangular

    def as_list(value) -> list:
        if isinstance(value, list):
            return value
        else:
            return [value]

    try:
        nml = nml.todict()  # f90nml.Namelist -> dict | fills '_start_index' attribute
    except AttributeError:
        pass

    stellsym = not nml.get("lasym", False)  # stellarator symmetry
    params = CaseInsensitiveDict(
        ProjectName=name,
        which_hmap=1,
        minimize_tol=1e-7,
        totalIter=10000,
        logIter=100,
        nfp=nml.get("nfp", 1),
        PhiEdge=nml.get("phiedge", 1.0),
        X1X2_deg=5,
        LA_deg=5,
        sgrid=dict(
            grid_type=0,
            nElems=5,
        ),
    )

    mgrid_file = str(nml.get("mgrid_file", "none")).strip()
    if nml.get("lfreeb", False) and mgrid_file.lower() not in ("", "none"):
        params["free_boundary"] = dict(
            coil_set=mgrid_file,
            extcur=as_list(nml.get("extcur", [])),
            updates=200,
            tolerance=1e-3,
            minimizer_type="dBFGS",
        )

    # --- profiles --- #
    def sampled(values: np.ndarray) -> dict:
        return {
            "type": "interpolation",
            "rho2": vmec_profile_grid(),
            "vals": values,
        }

    def power_series(kind: str) -> bool:
        return str(nml.get(f"{kind}_type", "power_series")).strip().lower() == "power_series"

    if "am" in nml or "am_aux_f" in nml:
        spres_ped = abs(nml.get("spres_ped", 1.0))
        if power_series("pmass") and nml.get("bloat", 1.0) == 1.0 and spres_ped >= 1.0:
            params["pres"] = {
                "type": "polynomial",
                "coefs": as_list(nml["am"]),
                "scale": nml.get("pres_scale", 1.0),
            }
        else:
            pres = nml.get("pres_scale", 1.0) * vmec_profile(nml, "pmass")
            s_pres = vmec_profile_grid()
            # VMEC holds the pressure at its pedestal value outside SPRES_PED (profil1d)
            pres = np.where(s_pres > spres_ped, np.interp(spres_ped, s_pres, pres), pres)
            params["pres"] = sampled(pres)
    else:
        logger.warning("No pressure profile defined.")

    if "ai" in nml or "ai_aux_f" in nml:
        if power_series("piota") and not nml.get("lrfp", False):
            params["iota"] = {
                "type": "polynomial",
                "coefs": as_list(nml["ai"]),
            }
        else:
            iota = vmec_profile(nml, "piota")
            params["iota"] = sampled(1.0 / iota if nml.get("lrfp", False) else iota)

    if nml.get("ncurr", 0) == 1:  # ncurr = 0: flux conservation | ncurr = 1: current constraint
        curtor = nml.get("curtor", 0.0)
        if curtor == 0.0:
            params["I_tor"] = {"type": "polynomial", "coefs": [0.0], "scale": 1.0}
        elif power_series("pcurr") and nml.get("bloat", 1.0) == 1.0:
            coefs = [0] + [
                p / (i + 1) for i, p in enumerate(as_list(nml.get("ac", [0.0])))
            ]  # I'(s) -> I(s)
            params["I_tor"] = {"type": "polynomial", "coefs": coefs}
            if sum(coefs) == 0.0:
                params["I_tor"]["scale"] = 1.0
            else:
                params["I_tor"]["scale"] = curtor / sum(coefs)
        else:
            current = vmec_profile(nml, "pcurr")
            # VMEC drops the current when the profile vanishes at the edge
            scale = (
                curtor / current[-1]
                if abs(current[-1]) > abs(np.finfo(float).eps * curtor)
                else 0.0
            )
            params["I_tor"] = sampled(scale * current)
        params["picard_current"] = "auto"
    if "ai" not in nml and nml.get("ncurr", 0) == 0:
        logger.warning("No iota or current profile defined.")

    # --- boundary --- #
    M, N = nml.get("mpol", 2) - 1, nml.get("ntor", 0)
    for vmec_key, gvec_key in [
        ("rbc", "X1_b_cos"),
        ("rbs", "X1_b_sin"),
        ("zbc", "X2_b_cos"),
        ("zbs", "X2_b_sin"),
    ]:
        if vmec_key not in nml:
            continue
        # f90nml represents sparse VMEC assignments as ragged lists (for
        # example ``RBC(0, 0)``, ``RBC(0, 1)``, ``RBC(1, 1)``).  Converting
        # those lists directly with ``np.array(..., dtype=float)`` raises a
        # ValueError before the converter can preserve the supplied modes.
        values = rectangular_boundary_values(nml[vmec_key])
        m0, n0 = 0, -N
        if "_start_index" in nml and vmec_key in nml["_start_index"]:
            n0, m0 = nml["_start_index"][vmec_key]
            if "mpol" not in nml:
                M = max(M, values.shape[0] - 1 + m0)  # m in [0 ... M]
                logger.debug(f"Setting M={M} from shape of '{vmec_key}' and start index {m0}")
            if "ntor" not in nml:
                N = max(N, -n0, values.shape[1] - 1 + n0)  # n in [-N ... N]
                logger.debug(f"Setting N={N} from shape of '{vmec_key}' and start index {n0}")
        else:
            raise ValueError(
                f"VMEC namelist array '{vmec_key}' has shape {values.shape} that does not match the expected shape {(M + 1, 2 * N + 1)=} and no '_start_index' is given in the namelist."
            )
        params[gvec_key] = {}
        for m in range(m0, m0 + values.shape[0]):
            for n in range(n0, n0 + values.shape[1]):
                if m == 0 and n < 0:
                    continue
                v = values[m - m0, n - n0]
                if not np.isnan(v) and v != 0.0:
                    params[gvec_key][m, n] = values[m - m0, n - n0]

    if "rbs" in nml or "zbc" in nml:
        if stellsym:
            logger.warning(
                "VMEC namelist contains 'RBS' or 'ZBC' but is supposed to be stellarator symmetric. Assuming asymmetry."
            )
        params["X1_sin_cos"] = "_sincos_"
        params["X2_sin_cos"] = "_sincos_"
        params["LA_sin_cos"] = "_sincos_"
    else:
        if not stellsym:
            logger.warning(
                "VMEC namelist does not contain 'RBS' or 'ZBC' but is supposed to be non-stellarator symmetric. Assuming symmetry."
            )
        params["X1_sin_cos"] = "_cos_"
        params["X2_sin_cos"] = "_sin_"
        params["LA_sin_cos"] = "_sin_"

    # --- axis --- #
    for vmec_key, gvec_key in [
        ("raxis_cc", "X1_a_cos"),
        ("raxis_cs", "X1_a_sin"),
        ("zaxis_cc", "X2_a_cos"),
        ("zaxis_cs", "X2_a_sin"),
    ]:
        if vmec_key not in nml:
            continue
        values = np.array(nml[vmec_key], dtype=float)
        if values.ndim == 0:
            continue
        params[gvec_key] = {(0, n): v for n, v in enumerate(values)}
    if not any(k in params for k in ["X1_a_cos", "X1_a_sin", "X2_a_cos", "X2_a_sin"]):
        params["init_average_axis"] = True

    # --- other --- #
    params.update(
        X1_mn_max=(M, N),
        X2_mn_max=(M, N),
        LA_mn_max=(M, N),
    )
    return params


def stack_parameters(parameters: Mapping) -> CaseInsensitiveDict:
    """Stack parameters into a hierarchical dictionary"""
    output = CaseInsensitiveDict()
    for key, value in parameters.items():
        if "_" not in key:
            output[key] = value
            continue
        group, name = key.split("_", 1)
        if group in ["iota", "pres", "sgrid"]:
            if group not in output:
                output[group] = CaseInsensitiveDict()
            output[group][name] = value
        else:
            output[key] = value
    return output


def flatten_parameters(parameters: Mapping) -> CaseInsensitiveDict:
    """Flatten parameters from a hierarchical dictionary"""
    output = CaseInsensitiveDict()
    for key, value in parameters.items():
        if key.lower() in ["stages", "i_tor", "picard_current", "totaliter"]:
            continue  # not supported by fortran-GVEC
        elif isinstance(value, Mapping) and not re.match(
            r"(x1|x2|la)(pert:?)?_[a|b]_(sin|cos)", key.lower()
        ):
            for subkey, subvalue in value.items():
                if subkey in ["save_diagnostics", "track_boundary_fields"]:
                    continue  # free boundary parameters not supported by fortran-GVEC
                elif subkey == "coil_set":
                    if hasattr(subvalue, "eval_B"):
                        subvalue = "python class"
                    else:
                        subvalue = str(subkey)

                output[f"{key}_{subkey}"] = subvalue
        else:
            output[key] = value
    return output


def stringify_mn_parameters(parameters: Mapping) -> CaseInsensitiveDict:
    """Convert boundary indices (mode numbers) from tuples of ints to strings."""
    output = CaseInsensitiveDict()
    for key, value in parameters.items():
        if re.match(r"(x1|x2|la)(pert:?)?_[a|b]_(sin|cos)", key.lower()):
            output[key] = {}
            for (m, n), val in value.items():
                output[key][f"({m}, {n:2d})"] = val
        elif key.lower() == "stages":
            output[key] = [stringify_mn_parameters(stage) for stage in value]
        elif isinstance(value, Mapping):
            output[key] = stringify_mn_parameters(value)
        else:
            output[key] = value
    return output


def unstringify_mn_parameters(parameters: Mapping) -> CaseInsensitiveDict:
    """Convert boundary indices (mode numbers) from strings to tuples of ints."""
    output = CaseInsensitiveDict()
    for key, value in parameters.items():
        if re.match(r"(x1|x2|la)(pert:?)?_[a|b]_(sin|cos)", key.lower()):
            output[key] = CaseInsensitiveDict()
            for mn, val in value.items():
                m, n = map(int, mn.strip("()").split(","))
                output[key][(m, n)] = val
        elif key.lower() == "stages":
            output[key] = [unstringify_mn_parameters(stage) for stage in value]
        elif isinstance(value, Mapping):
            output[key] = unstringify_mn_parameters(value)
        else:
            output[key] = value
    return output


def read_parameters(
    path: Path | str, format: Literal["ini", "yaml", "toml"] | None = None
) -> CaseInsensitiveDict:
    import tomlkit
    import yaml

    path = Path(path)
    # auto-detect format
    if format is None:
        format = path.suffix[1:]

    if format == "ini":
        inputs = read_parameter_file_ini(path)
        inputs = stack_parameters(inputs)
    elif format == "yaml":
        with open(path, "r") as file:
            inputs = yaml.safe_load(file)
        inputs = unstringify_mn_parameters(inputs)
    elif format == "toml":
        with open(path, "r") as file:
            inputs = tomlkit.parse(file.read()).unwrap()
        inputs = unstringify_mn_parameters(inputs)
    else:
        raise ValueError(f"Unknown parameter file format {format}")
    return inputs


def write_parameters(
    parameters: Mapping,
    path: Path | str = "parameter.ini",
    format: Literal["ini", "yaml", "toml"] | None = None,
    header: list[str] | str | None = None,
):
    import yaml
    from gvec._version import __version__

    if header is None:
        header = [
            "GVEC parameter file",
            f"pyGVEC v{__version__}",
        ]

    path = Path(path)
    # auto-detect format
    if format is None:
        format = path.suffix[1:]
    if isinstance(header, str):
        header = [header]

    if format == "ini":
        outputs = flatten_parameters(parameters)
        header = "".join([f"!{line}\n" for line in header])
        write_parameter_file_ini(outputs, path)
    elif format == "yaml":
        outputs = stringify_mn_parameters(parameters)
        with open(path, "w") as file:
            for line in header:
                file.write(f"# {line}\n")
            # ToDo: specify style/flow?
            yaml.safe_dump(serialize(outputs), file, sort_keys=False)
    elif format == "toml":
        with open(path, "w") as file:
            doc = format_toml(parameters)
            file.write(doc.as_string())
    else:
        raise ValueError(f"Unknown parameter file format {format}")


def format_toml(parameters: CaseInsensitiveDict, header: list[str] = []):
    """Format GVEC parameters as a TOML document.

    Parameters
    ----------
    parameters : CaseInsensitiveDict
        Parameter values to serialize. Values are expected to be in their
        native, unstringified form.

    Returns
    -------
    tomlkit.items.TOMLDocument
        TOML document containing the GVEC parameter layout. Write the result
        with ``doc.as_string()``.
    """
    from tomlkit import document, comment, nl, array, table, aot, key as toml_key
    from tomlkit.exceptions import ConvertError

    def add_params(doc, keys, parameters):
        if any(key in parameters for key in keys):
            for key in keys:
                if key in parameters:
                    value = parameters[key]
                    if isinstance(value, CaseInsensitiveDict):
                        value = serialize(value)
                    try:
                        doc[key] = value
                    except ConvertError as e:
                        raise TypeError(f"Could not convert '{key}': '{value}' for TOML") from e
            doc.add(nl())

    def add_section(doc, section_name):
        doc.add(nl())
        doc.add(comment(f"--- {section_name} --- #"))
        doc.add(nl())

    parameters: dict = serialize(parameters)
    parameters: CaseInsensitiveDict = stringify_mn_parameters(parameters)

    doc = document()
    if header:
        for line in header:
            doc.add(comment(line))
        doc.add(nl())

    MAIN_PARAMS = ["ProjectName", "nfp", "which_hmap", "PhiEdge"]
    for key in MAIN_PARAMS:
        if key in parameters:
            doc[key] = parameters[key]
    if "picard_current" in parameters and isinstance(parameters["picard_current"], str):
        doc["picard_current"] = parameters["picard_current"]
    doc.add(nl())

    FOURIER_PARAMS = [f"{Q}_{key}" for Q in ["X1", "X2", "LA"] for key in ["mn_max", "sin_cos"]]
    FOURIER_PARAMS += ["fac_nyq", "mn_nyq"]
    add_params(doc, FOURIER_PARAMS, parameters)

    RADIAL_PARAMS = ["X1X2_deg", "LA_deg", "degGP"]
    add_params(doc, RADIAL_PARAMS, parameters)

    MINIMIZER_PARAMS = [
        "minimize_tol",
        "totalIter",
        "maxIter",
        "outputIter",
        "logIter",
        "MinimizerType",
        "PrecondType",
        "start_dt",
    ]
    add_params(doc, MINIMIZER_PARAMS, parameters)

    EXTRA_PARAMS = [
        "sgrid",
        "iota",
        "pres",
        "I_tor",
        "picard_current",
        "stages",
        "free_boundary",
    ]
    DOF_PARAMS = [
        f"{Q}{pert}_{x}_{fun}"
        for Q in ["X1", "X2", "LA"]
        for x in "ab"
        for fun in ["cos", "sin"]
        for pert in ["", "pert"]
    ]
    HANDLED_PARAMS = (
        MAIN_PARAMS
        + FOURIER_PARAMS
        + RADIAL_PARAMS
        + MINIMIZER_PARAMS
        + EXTRA_PARAMS
        + DOF_PARAMS
    )
    OTHER_PARAMS = [
        key for key in parameters if key.lower() not in [p.lower() for p in HANDLED_PARAMS]
    ]
    add_params(doc, OTHER_PARAMS, parameters)

    if "sgrid" in parameters:
        doc["sgrid"] = serialize(parameters["sgrid"])

    if "picard_current" in parameters and isinstance(parameters["picard_current"], Mapping):
        doc["picard_current"] = serialize(parameters["picard_current"])

    add_section(doc, "Profiles")
    for key in ["pres", "I_tor", "iota"]:
        if key in parameters:
            doc[key] = table()
            for subkey, subvalue in serialize(parameters[key]).items():
                if isinstance(subvalue, list):
                    if len(subvalue) <= 8:
                        doc[key][subkey] = subvalue
                    else:
                        doc[key][subkey] = array()
                        for i in range(len(subvalue) // 5 + 1):
                            items = subvalue[i * 5 : (i + 1) * 5]
                            if len(items):
                                doc[key][subkey].add_line(*items)
                        doc[key][subkey].add_line(indent="")
                else:
                    doc[key][subkey] = subvalue

    if "free_boundary" in parameters and isinstance(parameters["free_boundary"], Mapping):
        add_section(doc, "Free Boundary")
        doc["free_boundary"] = parameters["free_boundary"].serialize()

    if "stages" in parameters:
        add_section(doc, "Stages")
        doc["stages"] = aot()
        for stage in parameters["stages"]:
            flat = table()
            for key, value in serialize(stage).items():
                if isinstance(value, Mapping):
                    for subkey, subvalue in value.items():
                        flat.append(toml_key([key, subkey]), subvalue)
                else:
                    flat.append(toml_key(key), value)
            doc["stages"].append(flat)

    add_section(doc, "Boundary and Axis degrees of freedom")
    for key in DOF_PARAMS:
        if key in parameters:
            doc[key] = parameters[key]

    return doc


def bspl2gvec(
    name: Literal["iota", "pres"],
    bspl: BSpline = None,
    knots: ArrayLike = None,
    coefs: ArrayLike = None,
    params: dict = {},
) -> dict:
    """Translates a scipy B-spline object or B-spline coefficients and knots for either a iota or pressure profile into a dictionary entries
    that can be handed to `adapt_parameter_file`.

    Args:
        name (str): profile identifyer, has to be either `iota` or `pres`.
        bspl (scipy.interpolate.BSpline): scipy BSpline object. If this is not provided `knots` and `coefs` are expected.
        knots (ArrayLike): Knots for the B-splines. Note that repeated edge knots according to the degree are expected.
        coefs (ArrayLike): Coefficients for the B-splines.
        params (dict, optional): Dictionary of gvec input parameters that will be adapted. Defaults to {}.

    Raises:
        ValueError: If `name` is neither `iota` nor `pres`.
        TypeError: If neither `bspl` nor `knots` and `coefs` is provided.

    Returns:
        dict: Dictionary of gvec input parameters
    """
    if name not in ["iota", "pres"]:
        raise ValueError(
            "Specified profile is not known!"
            + "`which_profile` has to be either `iota` or `pres`."
        )
    if (bspl is None) and (knots is None or coefs is None):
        raise TypeError(
            "`bspl` and at least one of `knots` or `coefs` are None."
            + "Please provide either `bspl` or `knots` and `coefs`"
        )

    if bspl is not None:
        params[f"{name}_coefs"] = bspl.c
        params[f"{name}_knots"] = bspl.t
    else:
        params[f"{name}_coefs"] = coefs
        params[f"{name}_knots"] = knots
    params[f"{name}_type"] = "bspline"

    return params


def profile_from_parameters(params: CaseInsensitiveDict) -> BSpline | Polynomial:
    match params.get("type", "polynomial"):
        case "polynomial":
            coefs = np.array(params["coefs"])
            coefs *= params.get("scale", PARAMETER_SCALE_DEFAULT)
            return np.polynomial.Polynomial(coefs)

        case "bspline":
            coefs = np.array(params["coefs"], dtype=float)
            coefs *= params.get("scale", PARAMETER_SCALE_DEFAULT)
            knots = np.array(params["knots"], dtype=float)
            deg = np.sum(knots == knots[0]) - 1
            if not np.isclose(knots[0], 0.0) or not np.isclose(knots[-1], 1.0):
                raise ValueError(
                    f"Knots for B-spline profile must start at 0.0 and end at 1.0, but got {knots[0]} and {knots[-1]}"
                )
            return BSpline(knots, coefs, deg)

        case "interpolation":
            k = 3
            vals = np.array(params["vals"], dtype=float)
            vals *= params.get("scale", PARAMETER_SCALE_DEFAULT)
            rho2 = np.array(params["rho2"], dtype=float)
            if min(np.sqrt(rho2)) > PROFILE_INTERPOLATION_RHO2_MIN:
                rho2 = np.append([0.0], rho2)
                vals = np.append([0.0], vals)
            BC_vals = params.get("BC_vals", PARAMETER_BC_VALS_DEFAULT)
            BC_type = [None, None]
            knots = rho2.copy()
            for i, key in enumerate(["axis", "edge"]):
                match params.get(f"BC_type_{key}", PARAMETER_BC_TYPE_DEFAULT):
                    case "not_a_knot":
                        pass
                    case "1st_deriv":
                        BC_type[i] = [(1, BC_vals[i])]
                    case "2nd_deriv":
                        BC_type[i] = [(2, BC_vals[i])]
                    case _:
                        raise ValueError(f"Unknown BC_type_{key}: {params[f'BC_type_{key}']}")
            if BC_type[0] is None:
                knots = np.concatenate([knots[:1], knots[2:]])
            if BC_type[1] is None:
                knots = np.concatenate([knots[:-2], knots[-1:]])
            knots = np.concatenate([np.zeros(k), knots, np.ones(k)])
            return make_interp_spline(rho2, vals, k=k, t=knots, bc_type=BC_type)
        case _:
            raise ValueError(f"Unknown profile type: {params['type']}")


def profile_to_parameters(profile: BSpline | Polynomial) -> CaseInsensitiveDict:
    if isinstance(profile, np.polynomial.Polynomial):
        return CaseInsensitiveDict(
            type="polynomial",
            coefs=profile.coef,
        )
    elif isinstance(profile, BSpline):
        return CaseInsensitiveDict(
            type="bspline",
            coefs=profile.c,
            knots=profile.t,
        )
    else:
        raise TypeError(f"Unknown profile type: {type(profile)}")


def get_boundary_from_statefile(filename: str | Path):
    """
    A simple but not optimal way of extracting the boundary fourier coefficients
    from a state-file for a stellarator symmetric case.

    Parameters
    ----------
    filename : str | Path
        Path to the state-file (of format .dat)

    Returns
    -------
    CaseInsensitiveDict
        CID with "X1_b_cos", "X2_b_sin" and "nfp"
    """
    data_dict = CaseInsensitiveDict({"X1_b_cos": {}, "X2_b_sin": {}})
    with open(filename) as f:
        nfp_flag = False
        X1_flag = False
        X2_flag = False
        for line in f:
            if "## LA: m,n,LA(1:nbase,iMode) #" in line:
                return data_dict

            if nfp_flag:
                data_dict["nfp"] = int(line.split(",")[0])
                nfp_flag = False

            if "nfp" in line:
                nfp_flag = True

            if X2_flag:
                line_list = line.split(",")
                m = int(line_list[0])
                n = int(line_list[1]) // data_dict["nfp"]
                data_dict["X2_b_sin"][(m, n)] = float(line_list[-1])
                X1_flag = False

            if "## X2: m,n,X2(1:nbase,iMode)" in line:
                X2_flag = True

            if X1_flag and not X2_flag:
                line_list = line.split(",")
                m = int(line_list[0])
                n = int(line_list[1]) // data_dict["nfp"]
                data_dict["X1_b_cos"][(m, n)] = float(line_list[-1])

            if "## X1: m,n,X1(1:nbase,iMode)" in line:
                X1_flag = True
    return data_dict
