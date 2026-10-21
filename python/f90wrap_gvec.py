# Copyright (c) 2026 GVEC Contributors, Max Planck Institute for Plasma Physics
# License: MIT
"""Run f90wrap with the pointer ABI of GVEC's selected Fortran compiler."""

import subprocess
import sys

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
    sys.exit(f90wrap.main())
