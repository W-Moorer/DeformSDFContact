def normal_law_penalty(g_n, eps):
    if g_n < 0.0:
        lam = eps * (-g_n)
        kn = eps
    else:
        lam = 0.0
        kn = 0.0
    return lam, kn
