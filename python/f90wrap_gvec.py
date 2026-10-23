# Copyright (c) 2026 GVEC Contributors, Max Planck Institute for Plasma Physics
# License: MIT
"""Run f90wrap with the pointer ABI of GVEC's selected Fortran compiler."""

import re
import subprocess
import sys
from pathlib import Path

from f90wrap.scripts import main as f90wrap


if __name__ == "__main__":
    handle_size = int(subprocess.check_output([sys.argv.pop(1)], text=True))
    # f90wrap has no CLI option for this. Its bundled probe measures the
    # compiler used to build f90wrap, which can differ from GVEC's compiler.
    f90wrap.sizeof_fortran_t = lambda: handle_size

    class PythonWrapperGenerator(f90wrap.pywrap.PythonWrapperGenerator):
        def write(self, *args):
            # Array accessors also need the measured size. Keep this local to
            # GVEC's generated module; other packages may use another ABI.
            super().write(
                *(
                    arg.replace("f90wrap.runtime.sizeof_fortran_t", str(handle_size))
                    for arg in args
                )
            )

    f90wrap.pywrap.PythonWrapperGenerator = PythonWrapperGenerator
    result = f90wrap.main()
    if result:
        sys.exit(result)

    # Initialize objects with polymorphic allocatable components in place.
    constructors = {
        "init_sfl_potential": "call init_sfl_potential_in_place(sfl_boozer={obj}, {args})",
        "t_rprofile_bspl": "call bsplprofile_init(sf={obj}, {args})",
        "t_rprofile_poly": "call polyprofile_init(sf={obj}, {args})",
        "t_newton_root1d_wrap_min1d": "call newton_root1d_wrap_min1d_init(sf={obj}, {args})",
        "t_fbase": "call {obj}%init({args})",
    }
    pattern = re.compile(r"(\w+%p%obj) = (\w+)\(([^)]*)\)")
    found = set()
    for wrapper in sorted(Path(".").glob("f90wrap_*.f90")):
        text = wrapper.read_text()

        def initialize_in_place(match):
            constructor = match.group(2)
            if constructor not in constructors:
                return match.group(0)
            found.add(constructor)
            return constructors[constructor].format(obj=match.group(1), args=match.group(3))

        changed = pattern.sub(initialize_in_place, text)
        if changed != text:
            wrapper.write_text(changed)
    missing = set(constructors) - found
    if missing:
        sys.exit(f"in-place constructors not found in the f90wrap output: {missing}")
