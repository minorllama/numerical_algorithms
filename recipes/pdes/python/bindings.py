"""ctypes interface to pde-recipes. Inputs and outputs are NumPy arrays."""
import ctypes as C
from operator import index
import os
from pathlib import Path
import sys

import numpy as np

N = C.c_size_t
F = C.c_double
I = C.c_int32
INPUT = np.ctypeslib.ndpointer(dtype=np.float64, flags=("C_CONTIGUOUS", "ALIGNED"))
OUTPUT = np.ctypeslib.ndpointer(
    dtype=np.float64, flags=("C_CONTIGUOUS", "ALIGNED", "WRITEABLE")
)
ERRORS = {
    1: "invalid argument",
    2: "nonfinite data or result",
    3: "singular pivot",
    4: "CFL condition violated",
    5: "Poisson RHS has nonzero mean",
    6: "backend failure",
    127: "Rust panic",
}


def library_path():
    names = {
        "win32": "pde_recipes.dll",
        "darwin": "libpde_recipes.dylib",
    }
    name = names.get(sys.platform, "libpde_recipes.so")
    default = Path(__file__).resolve().parents[1] / "target" / "release" / name
    return Path(os.environ.get("PDE_RECIPES_LIB", default)).resolve()


def setup(path=None):
    lib = C.CDLL(str(path or library_path()))
    signatures = {
        "pde_burgers": [N, F, F, F, N, I],
        "pde_black_scholes": [N, F, F, F, F, F, N],
        "pde_vlasov": [N, N, F, F, F, F, N],
        "pde_poisson": [N, N, F, F, I, I],
        "pde_elliptic_cr": [N, N, F, F, INPUT, INPUT],
        "pde_block_cr": [N, N, F, F, INPUT, INPUT],
        "pde_adi": [N, N, F, F, N],
    }
    for name, signature in signatures.items():
        fn = getattr(lib, name)
        fn.argtypes = [INPUT, OUTPUT, *signature]
        fn.restype = I
    return lib


def array(x, ndim):
    x = np.require(x, dtype=np.float64, requirements=["C", "A"])
    if x.ndim != ndim:
        raise ValueError(f"expected {ndim} dimensions, got {x.ndim}")
    return x


def step_count(steps):
    n = index(steps)
    if n <= 0 or n > N(-1).value:
        raise ValueError("steps must be a positive size_t")
    return n


def invoke(lib, name, x, args, out):
    if out is None:
        out = np.empty_like(x)
    if not isinstance(out, np.ndarray) or out.shape != x.shape:
        raise ValueError("output must be a NumPy array with the input shape")
    code = getattr(lib, name)(x, out, *args)
    if code:
        raise ValueError(f"{name}: {ERRORS.get(code, code)}")
    return out


class Solvers:
    def __init__(self, path=None):
        self.lib = setup(path)

    def burgers(self, u, dx, nu, dt, steps, method=0, out=None):
        x = array(u, 1)
        return invoke(self.lib, "pde_burgers", x, (x.size, dx, nu, dt, step_count(steps), method), out)

    def black_scholes(self, payoff, smax, strike, r, sigma, dt, steps, out=None):
        x = array(payoff, 1)
        args = (x.size, smax, strike, r, sigma, dt, step_count(steps))
        return invoke(self.lib, "pde_black_scholes", x, args, out)

    def vlasov(self, f, dx, dv, vmin, dt, steps, out=None):
        x = array(f, 2)
        nv, nx = x.shape
        return invoke(self.lib, "pde_vlasov", x, (nx, nv, dx, dv, vmin, dt, step_count(steps)), out)

    def poisson(self, rhs, dx, dy, bc=(0, 0), out=None):
        x = array(rhs, 2)
        ny, nx = x.shape
        return invoke(self.lib, "pde_poisson", x, (nx, ny, dx, dy, *bc), out)

    def elliptic_cr(self, rhs, dx, dy, drift, reaction, out=None, block=False):
        x, b, c = array(rhs, 2), array(drift, 1), array(reaction, 1)
        ny, nx = x.shape
        if b.shape != (ny,) or c.shape != (ny,):
            raise ValueError("drift and reaction must have ny entries")
        name = "pde_block_cr" if block else "pde_elliptic_cr"
        return invoke(self.lib, name, x, (nx, ny, dx, dy, b, c), out)

    def adi(self, u, rx, ry, steps, out=None):
        x = array(u, 2)
        ny, nx = x.shape
        return invoke(self.lib, "pde_adi", x, (nx, ny, rx, ry, step_count(steps)), out)
