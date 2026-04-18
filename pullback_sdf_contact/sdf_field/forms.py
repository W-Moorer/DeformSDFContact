import ufl

from metric.tensors import kinematics


def sdf_forms(u, phi, eta, phi0, beta, dx_band, dx_reg=None):
    if dx_reg is None:
        dx_reg = dx_band

    _, _, A = kinematics(u)
    g = ufl.grad(phi)
    q = ufl.dot(g, A * g) - 1.0
    R = q * ufl.dot(ufl.grad(eta), A * g) * dx_band + beta * (phi - phi0) * eta * dx_reg
    K_phi_phi = ufl.derivative(R, phi)
    K_phi_u = ufl.derivative(R, u)
    return R, K_phi_phi, K_phi_u
