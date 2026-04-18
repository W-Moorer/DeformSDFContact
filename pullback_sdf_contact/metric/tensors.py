import ufl


def kinematics(u):
    d = len(u)
    I = ufl.Identity(d)
    F = I + ufl.grad(u)
    C = F.T * F
    A = ufl.inv(C)
    return F, C, A
