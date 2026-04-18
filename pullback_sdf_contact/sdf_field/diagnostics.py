import ufl

from metric.tensors import kinematics


def eikonal_residual_form(u, phi, dx_band):
    _, _, A = kinematics(u)
    g = ufl.grad(phi)
    q = ufl.dot(g, A * g) - 1.0
    return q * q * dx_band
