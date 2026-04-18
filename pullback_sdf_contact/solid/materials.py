import ufl


def sigma_linear(u, E, nu):
    d = len(u)
    I = ufl.Identity(d)
    eps = ufl.sym(ufl.grad(u))
    mu = E / (2.0 * (1.0 + nu))
    lmbda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    return 2.0 * mu * eps + lmbda * ufl.tr(eps) * I
