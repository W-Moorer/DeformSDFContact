from dolfinx import fem


def create_sdf_space(domain, degree: int = 2):
    return fem.FunctionSpace(domain, ("Lagrange", degree))
