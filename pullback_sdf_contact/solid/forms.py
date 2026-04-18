import ufl

from .materials import sigma_linear


def solid_forms(u, v, cfg, dx):
    sigma = sigma_linear(u, cfg.E, cfg.nu)
    body_force = ufl.as_vector((0.0, 0.0, -cfg.body_force_z))
    R = ufl.inner(sigma, ufl.sym(ufl.grad(v))) * dx - ufl.dot(body_force, v) * dx
    K = ufl.derivative(R, u)
    return R, K
