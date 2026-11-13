#ifndef PDE_RECIPES_H
#define PDE_RECIPES_H
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

/* Native doubles, C-contiguous arrays. In and out may alias. The result is
 * written only on success. Keep all buffers valid throughout the call.
 * Return codes: 0 success, 1 invalid argument, 2 nonfinite data or result,
 * 3 singular pivot, 4 CFL violation, 5 incompatible Poisson source, 6 backend failure, 127 panic.
 */
int32_t pde_burgers(const double *in, double *out, size_t n, double dx,
                   double nu, double dt, size_t steps, int32_t method);
int32_t pde_black_scholes(const double *in, double *out, size_t n, double s_max,
                         double strike, double rate, double volatility,
                         double dt, size_t steps);
/* Phase space layout: f[iv][ix], ix varies fastest. */
int32_t pde_vlasov(const double *in, double *out, size_t nx, size_t nv,
                  double dx, double dv, double v_min, double dt, size_t steps);
/* Grid layout: u[iy][ix]. Boundary codes: periodic=0, Dirichlet=1, Neumann=2. */
int32_t pde_poisson(const double *in, double *out, size_t nx, size_t ny,
                   double dx, double dy, int32_t bc_x, int32_t bc_y);
/* drift and reaction are vectors of length ny. */
int32_t pde_elliptic_cr(const double *in, double *out, size_t nx, size_t ny,
                       double dx, double dy, const double *drift, const double *reaction);
int32_t pde_block_cr(const double *in, double *out, size_t nx, size_t ny,
                     double dx, double dy, const double *drift, const double *reaction);
int32_t pde_adi(const double *in, double *out, size_t nx, size_t ny,
               double rx, double ry, size_t steps);

#ifdef __cplusplus
}
#endif
#endif
