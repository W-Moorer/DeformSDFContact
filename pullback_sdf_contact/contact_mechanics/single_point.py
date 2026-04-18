import numpy as np

from .laws import normal_law_penalty


def contact_residual_single_point(g_n, G_u, penalty):
    """
    Returns:
      R_uc   shape = (ndof_u,)
      lam    scalar
      kn     scalar
    """
    lam, kn = normal_law_penalty(g_n, penalty)
    R_uc = lam * np.asarray(G_u, dtype=np.float64)
    return R_uc, lam, kn


def contact_tangent_uphi_single_point(g_n, G_u, G_a, penalty):
    """
    Returns:
      K_uphi_c   shape = (ndof_u, ndof_phi)
      lam
      kn
    """
    lam, kn = normal_law_penalty(g_n, penalty)
    G_u = np.asarray(G_u, dtype=np.float64)
    G_a = np.asarray(G_a, dtype=np.float64)
    K_uphi_c = -kn * np.outer(G_u, G_a)
    return K_uphi_c, lam, kn


def contact_tangent_uu_single_point(g_n, G_u, penalty):
    """
    Returns:
      K_uu_c   shape = (ndof_u, ndof_u)
      lam
      kn
    """
    lam, kn = normal_law_penalty(g_n, penalty)
    G_u = np.asarray(G_u, dtype=np.float64)
    K_uu_c = -kn * np.outer(G_u, G_u)
    return K_uu_c, lam, kn
