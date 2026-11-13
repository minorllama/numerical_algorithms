"""Call the Rust C ABI and compare against independent SciPy solvers."""
import sys

import numpy as np
from scipy import fft, sparse
from scipy.integrate import solve_ivp
from scipy.sparse.linalg import expm_multiply, spsolve
from scipy.special import ndtr

from bindings import Solvers


def evolve(rhs, y, t, method="DOP853", **kwargs):
    solution = solve_ivp(rhs, (0.0, t), y.ravel(), method=method,
                         t_eval=[t], rtol=2e-10, atol=2e-12, **kwargs)
    if not solution.success:
        raise RuntimeError(solution.message)
    return solution.y[:, -1].reshape(y.shape)


def relerr(x, y):
    return np.linalg.norm(x - y) / np.linalg.norm(y)


def report(label, error, threshold):
    print(f"{label:48s} {error:.6e}   (limit {threshold:g})")
    assert np.isfinite(error) and error < threshold, label


def wavenumber(n, h):
    return 2 * np.pi * fft.fftfreq(n, d=h)


def burgers_rhs(k, nu):
    cutoff = np.abs(fft.fftfreq(k.size) * k.size) < k.size / 3
    def rhs(_, u):
        uh = fft.fft(u)
        filtered = fft.ifft(uh * cutoff).real
        hflux = 0.5 * cutoff * fft.fft(filtered**2)
        return fft.ifft(-1j * k * hflux - nu * k**2 * uh).real
    return rhs


def burgers_case(lib, n, method=0):
    dx, nu, time = 2*np.pi/n, 0.08, 0.4
    x = dx * np.arange(n)
    u = 0.35 + np.sin(x) + 0.2 * np.cos(2*x)
    steps = 2*n
    target = np.empty_like(u)
    got = lib.burgers(u, dx, nu, time/steps, steps, method=method, out=target)
    assert got is target
    ref = evolve(burgers_rhs(wavenumber(n, dx), nu), u, time)
    report("Burgers mean drift", abs(got.mean()-u.mean()), 2e-13)
    return relerr(got, ref)


def test_burgers(lib):
    low = burgers_case(lib, 48)
    high = burgers_case(lib, 96)
    report("Burgers LW vs SciPy pseudospectral IVP", high, 1.5e-3)
    report("Burgers fine/coarse error ratio", high/low, 0.35)
    report("Burgers MUSCL vs SciPy pseudospectral IVP", burgers_case(lib, 96, 1), 5e-3)
    report("Burgers Godunov vs SciPy pseudospectral IVP", burgers_case(lib, 96, 2), 3e-2)


def bs_exact(s, k, r, sigma, t):
    q = np.zeros_like(s)
    w = sigma * np.sqrt(t)
    a = (np.log(s[1:]/k) + (r+0.5*sigma*sigma)*t)/w
    q[1:] = s[1:]*ndtr(a)-k*np.exp(-r*t)*ndtr(a-w)
    return q


def bs_reference(payoff, smax, k, r, sigma, t):
    j = np.arange(1, payoff.size-1)
    diffusion = 0.5*(sigma*j)**2
    drift = 0.5*r*j
    lo, hi = diffusion-drift, diffusion+drift
    a = sparse.diags([lo[1:], -2*diffusion-r, hi[:-1]], [-1,0,1], format="csc")
    def rhs(tau, y):
        z = a @ y
        z[-1] += hi[-1]*(smax-k*np.exp(-r*tau))
        return z
    inner = evolve(rhs, payoff[1:-1], t, method="BDF", jac=a)
    return np.r_[0., inner, smax-k*np.exp(-r*t)]


def test_black_scholes(lib):
    n, smax, k, r, sigma, t, steps = 481, 400., 100., 0.04, 0.24, 0.6, 240
    s = np.linspace(0, smax, n)
    payoff = np.maximum(s-k, 0.)
    got = lib.black_scholes(payoff, smax, k, r, sigma, t/steps, steps)
    ref = bs_reference(payoff, smax, k, r, sigma, t)
    mask = (s >= 60) & (s <= 160)
    report("Black–Scholes vs SciPy BDF, max price error",
           np.max(np.abs(got-ref)), 2e-4)
    report("Black–Scholes vs exact, max price error near K",
           np.max(np.abs(got[mask]-bs_exact(s,k,r,sigma,t)[mask])), 3e-3)


def vlasov_rhs(v, dx, dv, nx):
    kx = wavenumber(nx, dx)
    kv = wavenumber(v.size, dv)
    inv = np.zeros(nx, dtype=np.complex128)
    np.divide(1., 1j*kx, out=inv, where=kx!=0)
    if nx % 2 == 0:
        inv[nx//2] = 0.
    def rhs(_, y):
        f = y.reshape(v.size, nx)
        e = fft.ifft(inv * fft.fft(dv*f.sum(axis=0))).real
        fx = fft.ifft(1j*kx*fft.fft(f,axis=1),axis=1).real
        fv = fft.ifft(1j*kv[:,None]*fft.fft(f,axis=0),axis=0).real
        return (-v[:,None]*fx-e*fv).ravel()
    return rhs


def vlasov_case(lib, n):
    nx, nv, vmin, t, steps = n, 2*n, -6., 0.65, 2*n
    dx, dv = 2*np.pi/nx, 12./nv
    x, v = dx*np.arange(nx), vmin+dv*np.arange(nv)
    maxwell = np.exp(-0.5*v*v)/np.sqrt(2*np.pi)
    f = maxwell[:,None] * (1+0.12*np.cos(x)+0.04*np.sin(2*x))
    f += 0.025*(maxwell*v)[:,None]*np.cos(2*x)
    got = lib.vlasov(f, dx, dv, vmin, t/steps, steps)
    ref = evolve(vlasov_rhs(v, dx, dv, nx), f, t)
    report("Vlasov relative mass drift", abs(got.sum()/f.sum()-1), 1e-12)
    return relerr(got,ref)


def test_vlasov(lib):
    low = vlasov_case(lib, 24)
    high = vlasov_case(lib, 48)
    report("Vlasov–Poisson vs SciPy pseudospectral IVP", high, 1.5e-3)
    report("Vlasov fine/coarse error ratio", high/low, 0.38)


def lap1(n, h, bc):
    a = sparse.diags([np.ones(n-1), -2*np.ones(n), np.ones(n-1)], [-1,0,1], format="lil")
    if bc == 0:
        a[0,-1] = a[-1,0] = 1
    if bc == 2:
        a[0,0] = a[-1,-1] = -1
    return a.tocsr()/h**2


def lap2(nx, ny, dx, dy, bc):
    return sparse.kronsum(lap1(nx,dx,bc[0]), lap1(ny,dy,bc[1]), format="csr")


def poisson_case(lib, bc):
    nx, ny, dx, dy = 13, 10, 0.22, 0.34
    rhs = np.random.default_rng(238).normal(size=(ny,nx))
    a = lap2(nx,ny,dx,dy,bc)
    if 1 not in bc:
        rhs -= rhs.mean()
        ref = np.r_[0.,spsolve(a[1:,1:],rhs.ravel()[1:])]
        ref -= ref.mean()
    else:
        ref = spsolve(a,rhs.ravel())
    got = lib.poisson(rhs,dx,dy,bc)
    report(f"Poisson {bc} vs SciPy sparse solve", relerr(got.ravel(),ref), 1e-12)


def test_poisson(lib):
    for bx in range(3):
        for by in range(3):
            poisson_case(lib,(bx,by))
    try:
        lib.poisson(np.ones((6,5)),1.,1.)
    except ValueError as err:
        assert "nonzero mean" in str(err)
    else:
        raise AssertionError("incompatible Poisson RHS was accepted")


def test_cyclic_reduction(lib):
    nx, ny, dx, dy = 15, 12, 0.2, 0.31
    b = np.linspace(-0.2,0.3,ny)
    c = -0.5-np.linspace(0,1,ny)**2
    rhs = np.random.default_rng(348).normal(size=(ny,nx))
    a = lap2(nx,ny,dx,dy,(1,1))
    d = sparse.diags([-np.ones(ny-1), np.ones(ny-1)], [-1,1])/(2*dy)
    y = sparse.diags(b) @ d + sparse.diags(c)
    a += sparse.kron(y, sparse.eye(nx), format="csr")
    ref = spsolve(a,rhs.ravel())
    got = lib.elliptic_cr(rhs,dx,dy,b,c)
    report("Modal cyclic reduction vs SciPy sparse solve", relerr(got.ravel(),ref), 1e-12)
    block = lib.elliptic_cr(rhs,dx,dy,b,c,block=True)
    report("Block cyclic reduction vs SciPy sparse solve", relerr(block.ravel(),ref), 1e-12)


def test_adi(lib):
    nx, ny, dt, steps = 32, 24, 0.01, 24
    dx, dy = 2*np.pi/nx, 2*np.pi/ny
    x, y = np.arange(nx)*dx, np.arange(ny)*dy
    u = np.cos(x)[None,:] * np.sin(2*y)[:,None] + 0.2*np.sin(3*x)
    a = sparse.kronsum(0.3*lap1(nx,dx,0),0.8*lap1(ny,dy,0))
    ref = expm_multiply(a*(dt*steps), u.ravel())
    got = lib.adi(u,0.3*dt/dx**2,0.8*dt/dy**2,steps)
    report("ADI vs SciPy matrix exponential", relerr(got.ravel(),ref), 1e-4)


def test_abi(lib):
    u = np.linspace(0.,1.,8)
    out = np.full_like(u, -11.)
    try:
        lib.burgers(u,0.1,0.01,2.,1,out=out)
    except ValueError as err:
        assert "CFL" in str(err) and (out == -11).all()
    else:
        raise AssertionError("unstable timestep was accepted")
    expect = lib.burgers(u,0.1,0.01,0.01,2)
    got = lib.burgers(u,0.1,0.01,0.01,2,out=u)
    assert got is u and np.array_equal(got,expect)


def main():
    lib = Solvers()
    for pde_test in (test_burgers,test_black_scholes,test_vlasov,test_poisson,
                 test_cyclic_reduction,test_adi,test_abi):
        pde_test(lib)
    


if __name__ == "__main__":
    if "-h" in sys.argv:
        print("PDE_RECIPES_LIB=$soPath/libpde_recipes.so python python/verify.py")
    else:
        main()
