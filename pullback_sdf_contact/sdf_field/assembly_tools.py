from dolfinx import fem
import ufl

from .forms import sdf_forms


def _compile_form(expr):
    if hasattr(fem, "form") and callable(fem.form):
        return fem.form(expr)
    return fem.Form(expr)


def assemble_sdf_objects(u, phi, phi0, beta, dx_band, dx_reg=None):
    """
    Return compiled DOLFINx forms for the SDF residual and tangents.
    """
    eta = ufl.TestFunction(phi.function_space)
    R_phi_expr, K_phi_phi_expr, K_phi_u_expr = sdf_forms(
        u, phi, eta, phi0, beta, dx_band, dx_reg=dx_reg
    )
    return (
        _compile_form(R_phi_expr),
        _compile_form(K_phi_phi_expr),
        _compile_form(K_phi_u_expr),
    )
