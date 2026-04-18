from dolfinx import fem


def create_displacement_space(domain, degree: int = 1):
    return fem.VectorFunctionSpace(domain, ("Lagrange", degree))
