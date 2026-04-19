import csv
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "pullback_sdf_contact"
FIG_DIR = Path(__file__).resolve().parent / "figures"


def read_csv_rows(path):
    with open(path, "r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_summary_by_mode(path):
    rows = read_csv_rows(path)
    return {row["mode"]: row for row in rows}


def read_history_points(path, x_key="load_value", y_key="reaction_norm"):
    rows = read_csv_rows(path)
    accepted_rows = [row for row in rows if row.get("accepted", "True") == "True"]
    x = np.array([float(row[x_key]) for row in accepted_rows], dtype=float)
    y = np.array([float(row[y_key]) for row in accepted_rows], dtype=float)
    return x, y


def read_vector_field_h5(path):
    with h5py.File(path, "r") as handle:
        geometry = handle["Mesh/mesh/geometry"][...]
        func_group = next(iter(handle["Function"].values()))
        values = func_group["0"][...]
    return geometry, values


def build_structured_displacement_grid(path):
    geometry, values = read_vector_field_h5(path)
    x_unique = np.unique(geometry[:, 0])
    y_unique = np.unique(geometry[:, 1])
    z_unique = np.unique(geometry[:, 2])
    grid = np.zeros((x_unique.size, y_unique.size, z_unique.size, values.shape[1]), dtype=float)

    for coord, vec in zip(geometry, values):
        ix = int(np.where(np.isclose(x_unique, coord[0]))[0][0])
        iy = int(np.where(np.isclose(y_unique, coord[1]))[0][0])
        iz = int(np.where(np.isclose(z_unique, coord[2]))[0][0])
        grid[ix, iy, iz, :] = vec

    return x_unique, y_unique, z_unique, grid


def build_structured_scalar_grid(path):
    geometry, values = read_vector_field_h5(path)
    values = np.asarray(values).reshape(-1)
    x_unique = np.unique(geometry[:, 0])
    y_unique = np.unique(geometry[:, 1])
    z_unique = np.unique(geometry[:, 2])
    grid = np.zeros((x_unique.size, y_unique.size, z_unique.size), dtype=float)

    for coord, value in zip(geometry, values):
        ix = int(np.where(np.isclose(x_unique, coord[0]))[0][0])
        iy = int(np.where(np.isclose(y_unique, coord[1]))[0][0])
        iz = int(np.where(np.isclose(z_unique, coord[2]))[0][0])
        grid[ix, iy, iz] = float(value)

    return x_unique, y_unique, z_unique, grid


def configure_style():
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "legend.fontsize": 9,
            "figure.titlesize": 12,
        }
    )


def ensure_dirs():
    FIG_DIR.mkdir(parents=True, exist_ok=True)


def save_figure(fig, path):
    fig.savefig(path, bbox_inches="tight", pad_inches=0.04)


def make_iterations_figure():
    staggered = read_summary_by_mode(DATA_DIR / "staggered_baseline_summary.csv")
    monolithic = read_summary_by_mode(DATA_DIR / "monolithic_baseline_summary.csv")

    modes = ["baseline", "aggressive"]
    labels = ["Staggered", "Mono-dense", "Mono-PETSc/LU"]
    data = np.array(
        [
            [
                float(staggered[m]["total_outer_iterations"]),
                9.0 if m == "baseline" else 21.0,
                float(monolithic[m]["total_newton_iterations"]),
            ]
            for m in modes
        ]
    )

    x = np.arange(len(modes))
    width = 0.22
    fig, ax = plt.subplots(figsize=(6.8, 3.8))
    for i, label in enumerate(labels):
        ax.bar(x + (i - 1) * width, data[:, i], width=width, label=label)
    ax.set_xticks(x)
    ax.set_xticklabels(["Baseline", "Aggressive"])
    ax.set_ylabel("Total nonlinear iterations")
    ax.set_title("Iteration counts: staggered vs monolithic")
    ax.legend(frameon=True)
    fig.tight_layout()
    save_figure(fig, FIG_DIR / "stage_summary_staggered_vs_monolithic_iterations.pdf")
    plt.close(fig)


def make_benchmark_schematic():
    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    ax.set_aspect("equal")
    ax.axis("off")

    # Block front face
    front = np.array([[0.0, 0.0], [2.8, 0.0], [2.8, 2.8], [0.0, 2.8], [0.0, 0.0]])
    # Back shift for pseudo-3D box
    shift = np.array([0.75, 0.55])
    back = front + shift

    ax.fill(front[:, 0], front[:, 1], color="#d9e6f2", zorder=2, ec="#2b4c7e", lw=1.4)
    ax.fill(back[:, 0], back[:, 1], color="#eef4fa", zorder=1, ec="#2b4c7e", lw=1.2)

    # Side/top edges
    for idx in range(4):
        ax.plot([front[idx, 0], back[idx, 0]], [front[idx, 1], back[idx, 1]], color="#2b4c7e", lw=1.2)
    ax.plot(front[:, 0], front[:, 1], color="#2b4c7e", lw=1.4)
    ax.plot(back[:, 0], back[:, 1], color="#2b4c7e", lw=1.2)

    # Top contact surface highlight
    top_face = np.array([
        [0.0, 2.8],
        [2.8, 2.8],
        [3.55, 3.35],
        [0.75, 3.35],
        [0.0, 2.8],
    ])
    ax.fill(top_face[:, 0], top_face[:, 1], color="#9ecae1", alpha=0.85, ec="#2b4c7e", lw=1.2, zorder=3)

    # Rigid plane indenter
    plane = np.array([
        [-0.2, 3.9],
        [3.3, 3.9],
        [4.05, 4.45],
        [0.55, 4.45],
        [-0.2, 3.9],
    ])
    ax.fill(plane[:, 0], plane[:, 1], color="#bdbdbd", ec="#4d4d4d", lw=1.4, zorder=4)

    # Downward load arrows
    for xpos in [0.7, 1.7, 2.7]:
        ax.annotate(
            "",
            xy=(xpos + 0.4, 3.45),
            xytext=(xpos + 0.4, 4.2),
            arrowprops=dict(arrowstyle="-|>", lw=1.6, color="#d62728"),
            zorder=5,
        )

    # Fixed bottom
    ax.plot([-0.1, 2.95], [0.0, 0.0], color="#333333", lw=2.0, zorder=5)
    for xpos in np.linspace(0.0, 2.8, 12):
        ax.plot([xpos - 0.12, xpos], [-0.22, 0.0], color="#333333", lw=1.0, zorder=5)

    # Labels
    ax.text(1.45, 1.35, "Elastic block", ha="center", va="center", fontsize=11, color="#1f3552")
    ax.text(2.1, 3.07, "Slave contact surface", ha="center", va="bottom", fontsize=10, color="#1f3552")
    ax.text(1.95, 4.57, "Rigid flat indenter", ha="center", va="bottom", fontsize=10, color="#444444")
    ax.text(3.52, 3.85, "Load increment", ha="left", va="center", fontsize=10, color="#d62728")
    ax.text(1.4, -0.34, "Bottom face fixed", ha="center", va="top", fontsize=10, color="#333333")

    # Axes hint
    origin = np.array([3.65, 0.45])
    ax.annotate("", xy=origin + [0.6, 0.0], xytext=origin, arrowprops=dict(arrowstyle="-|>", lw=1.2, color="black"))
    ax.annotate("", xy=origin + [0.0, 0.6], xytext=origin, arrowprops=dict(arrowstyle="-|>", lw=1.2, color="black"))
    ax.annotate("", xy=origin + [0.35, 0.27], xytext=origin, arrowprops=dict(arrowstyle="-|>", lw=1.2, color="black"))
    ax.text(origin[0] + 0.68, origin[1] - 0.02, "x", fontsize=9)
    ax.text(origin[0] - 0.06, origin[1] + 0.68, "z", fontsize=9)
    ax.text(origin[0] + 0.4, origin[1] + 0.32, "y", fontsize=9)

    ax.set_xlim(-0.4, 4.5)
    ax.set_ylim(-0.5, 4.9)
    fig.tight_layout()
    save_figure(fig, FIG_DIR / "stage_summary_benchmark_schematic.pdf")
    plt.close(fig)


def make_response_figure(y_key, ylabel, filename, title):
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.6), sharey=False)
    cases = [
        ("baseline", axes[0], "Baseline"),
        ("aggressive", axes[1], "Aggressive"),
    ]
    for mode, ax, subtitle in cases:
        sx, sy = read_history_points(
            DATA_DIR / f"contact_history_{mode}_consistent_linearized.csv",
            y_key=y_key,
        )
        dx, dy = read_history_points(
            DATA_DIR / f"monolithic_history_{mode}_dense.csv",
            y_key=y_key,
        )
        px, py = read_history_points(
            DATA_DIR / f"monolithic_history_{mode}_3x3x3_petsc_block_lu.csv",
            y_key=y_key,
        )
        ax.plot(sx, sy, marker="o", linewidth=1.8, label="Staggered")
        ax.plot(dx, dy, marker="s", linewidth=1.8, label="Mono-dense")
        ax.plot(px, py, marker="^", linewidth=1.8, label="Mono-PETSc/LU")
        ax.set_xlabel("Load")
        ax.set_ylabel(ylabel)
        ax.set_title(subtitle)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=True)
    fig.suptitle(title, y=1.03)
    fig.tight_layout()
    fig.subplots_adjust(top=0.74)
    save_figure(fig, FIG_DIR / filename)
    plt.close(fig)


def make_linear_iterations_figure():
    small = {
        "LU": 9.0,
        "GMRES+ILU": 9.0,
        "GMRES+FS(mult)+ILU": 9.0,
    }
    larger = {
        "LU": 9.0,
        "GMRES+ILU": 9.0,
        "GMRES+FS(mult)+ILU": 19.0,
    }
    labels = list(small.keys())
    x = np.arange(len(labels))
    width = 0.34

    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    ax.bar(x - width / 2, [small[k] for k in labels], width=width, label="3x3x3")
    ax.bar(x + width / 2, [larger[k] for k in labels], width=width, label="4x4x4")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Total linear iterations")
    ax.set_title("Linear iterations under different PCs")
    ax.legend(frameon=True)
    fig.tight_layout()
    save_figure(fig, FIG_DIR / "stage_summary_linear_iterations_block_pc.pdf")
    plt.close(fig)


def _top_surface_grid(path):
    geometry, values = read_vector_field_h5(path)
    top = np.isclose(geometry[:, 2], geometry[:, 2].max())
    coords = geometry[top]
    disp = values[top]
    x_unique = np.unique(coords[:, 0])
    y_unique = np.unique(coords[:, 1])
    grid = np.full((y_unique.size, x_unique.size), np.nan, dtype=float)

    for coord, vec in zip(coords, disp):
        ix = int(np.where(np.isclose(x_unique, coord[0]))[0][0])
        iy = int(np.where(np.isclose(y_unique, coord[1]))[0][0])
        grid[iy, ix] = -float(vec[2])

    return x_unique, y_unique, grid, coords, disp


def _diagonal_profile(coords, disp):
    diagonal = np.isclose(coords[:, 0], coords[:, 1])
    diag_coords = coords[diagonal]
    diag_disp = disp[diagonal]
    order = np.argsort(diag_coords[:, 0])
    x = diag_coords[order, 0]
    uz = -diag_disp[order, 2]
    return x, uz


def make_field_comparison_figure():
    mono_baseline = DATA_DIR / "output_u_monolithic_baseline_3x3x3_petsc_block_lu.h5"
    mono_aggressive = DATA_DIR / "output_u_monolithic_aggressive_3x3x3_petsc_block_lu.h5"
    stag_baseline = DATA_DIR / "output_u_contact_loadpath_baseline_consistent_linearized.h5"
    stag_aggressive = DATA_DIR / "output_u_contact_loadpath_aggressive_consistent_linearized.h5"

    xb, yb, grid_b, coords_b, disp_b = _top_surface_grid(mono_baseline)
    xa, ya, grid_a, coords_a, disp_a = _top_surface_grid(mono_aggressive)
    _, _, _, coords_sb, disp_sb = _top_surface_grid(stag_baseline)
    _, _, _, coords_sa, disp_sa = _top_surface_grid(stag_aggressive)
    line_b_x, line_b_stag = _diagonal_profile(coords_sb, disp_sb)
    _, line_b_mono = _diagonal_profile(coords_b, disp_b)
    line_a_x, line_a_stag = _diagonal_profile(coords_sa, disp_sa)
    _, line_a_mono = _diagonal_profile(coords_a, disp_a)

    vmin = min(np.nanmin(grid_b), np.nanmin(grid_a))
    vmax = max(np.nanmax(grid_b), np.nanmax(grid_a))

    fig, axes = plt.subplots(2, 2, figsize=(8.6, 6.8))

    extent_b = [xb.min(), xb.max(), yb.min(), yb.max()]
    extent_a = [xa.min(), xa.max(), ya.min(), ya.max()]
    m0 = axes[0, 0].imshow(
        grid_b,
        origin="lower",
        extent=extent_b,
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
        aspect="equal",
    )
    axes[0, 0].set_title("Baseline: top-surface compression")
    axes[0, 0].set_xlabel("x")
    axes[0, 0].set_ylabel("y")

    m1 = axes[0, 1].imshow(
        grid_a,
        origin="lower",
        extent=extent_a,
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
        aspect="equal",
    )
    axes[0, 1].set_title("Aggressive: top-surface compression")
    axes[0, 1].set_xlabel("x")
    axes[0, 1].set_ylabel("y")

    axes[1, 0].plot(line_b_x, line_b_stag, marker="o", linewidth=1.8, label="Staggered")
    axes[1, 0].plot(line_b_x, line_b_mono, marker="s", linewidth=1.8, label="Monolithic")
    axes[1, 0].set_title("Baseline: diagonal top profile")
    axes[1, 0].set_xlabel("Top-surface diagonal coordinate")
    axes[1, 0].set_ylabel(r"Compression magnitude $-u_z$")
    axes[1, 0].legend(frameon=True)

    axes[1, 1].plot(line_a_x, line_a_stag, marker="o", linewidth=1.8, label="Staggered")
    axes[1, 1].plot(line_a_x, line_a_mono, marker="s", linewidth=1.8, label="Monolithic")
    axes[1, 1].set_title("Aggressive: diagonal top profile")
    axes[1, 1].set_xlabel("Top-surface diagonal coordinate")
    axes[1, 1].set_ylabel(r"Compression magnitude $-u_z$")
    axes[1, 1].legend(frameon=True)

    cbar = fig.colorbar(m1, ax=axes[0, :], shrink=0.92, pad=0.06)
    cbar.set_label(r"Compression magnitude $-u_z$")
    fig.suptitle("Representative displacement fields and profiles", y=0.98)
    fig.subplots_adjust(left=0.08, right=0.94, bottom=0.08, top=0.90, wspace=0.28, hspace=0.34)
    save_figure(fig, FIG_DIR / "stage_summary_field_profiles.pdf")
    plt.close(fig)


def make_top_profile_mode_comparison():
    stag_baseline = DATA_DIR / "output_u_contact_loadpath_baseline_consistent_linearized.h5"
    stag_aggressive = DATA_DIR / "output_u_contact_loadpath_aggressive_consistent_linearized.h5"
    mono_baseline = DATA_DIR / "output_u_monolithic_baseline_3x3x3_petsc_block_lu.h5"
    mono_aggressive = DATA_DIR / "output_u_monolithic_aggressive_3x3x3_petsc_block_lu.h5"

    _, _, _, coords_sb, disp_sb = _top_surface_grid(stag_baseline)
    _, _, _, coords_sa, disp_sa = _top_surface_grid(stag_aggressive)
    _, _, _, coords_mb, disp_mb = _top_surface_grid(mono_baseline)
    _, _, _, coords_ma, disp_ma = _top_surface_grid(mono_aggressive)

    x_b, uz_sb = _diagonal_profile(coords_sb, disp_sb)
    _, uz_mb = _diagonal_profile(coords_mb, disp_mb)
    x_a, uz_sa = _diagonal_profile(coords_sa, disp_sa)
    _, uz_ma = _diagonal_profile(coords_ma, disp_ma)

    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.6), sharey=True)
    axes[0].plot(x_b, uz_sb, marker="o", linewidth=1.8, label="Staggered")
    axes[0].plot(x_b, uz_mb, marker="s", linewidth=1.8, label="Monolithic")
    axes[0].set_title("Baseline")
    axes[0].set_xlabel("Top diagonal coordinate")
    axes[0].set_ylabel(r"Compression magnitude $-u_z$")
    axes[0].legend(frameon=True)

    axes[1].plot(x_a, uz_sa, marker="o", linewidth=1.8, label="Staggered")
    axes[1].plot(x_a, uz_ma, marker="s", linewidth=1.8, label="Monolithic")
    axes[1].set_title("Aggressive")
    axes[1].set_xlabel("Top diagonal coordinate")
    axes[1].set_ylabel(r"Compression magnitude $-u_z$")
    axes[1].legend(frameon=True)

    fig.suptitle("Top-surface diagonal profiles: baseline vs aggressive", y=1.02)
    fig.tight_layout()
    save_figure(fig, FIG_DIR / "stage_summary_top_profile_baseline_vs_aggressive.pdf")
    plt.close(fig)


def _dominant_projected_direction(tensor):
    vals, vecs = np.linalg.eigh(tensor)
    idx = int(np.argmax(np.abs(vals)))
    vec = vecs[:, idx]
    proj = np.array([vec[0], vec[2]], dtype=float)
    norm = np.linalg.norm(proj)
    if norm < 1e-14:
        return np.array([0.0, 0.0], dtype=float)
    proj = proj / norm
    if proj[1] < 0.0:
        proj = -proj
    return proj


def make_stress_strain_glyph_figure():
    path = DATA_DIR / "output_u_monolithic_aggressive_3x3x3_petsc_block_lu.h5"
    x, y, z, u = build_structured_displacement_grid(path)
    E = 40.0
    nu = 0.3
    mu = E / (2.0 * (1.0 + nu))
    lmbda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))

    dux_dx = np.gradient(u[..., 0], x, axis=0, edge_order=2)
    dux_dy = np.gradient(u[..., 0], y, axis=1, edge_order=2)
    dux_dz = np.gradient(u[..., 0], z, axis=2, edge_order=2)
    duy_dx = np.gradient(u[..., 1], x, axis=0, edge_order=2)
    duy_dy = np.gradient(u[..., 1], y, axis=1, edge_order=2)
    duy_dz = np.gradient(u[..., 1], z, axis=2, edge_order=2)
    duz_dx = np.gradient(u[..., 2], x, axis=0, edge_order=2)
    duz_dy = np.gradient(u[..., 2], y, axis=1, edge_order=2)
    duz_dz = np.gradient(u[..., 2], z, axis=2, edge_order=2)

    exx = dux_dx
    eyy = duy_dy
    ezz = duz_dz
    exy = 0.5 * (dux_dy + duy_dx)
    exz = 0.5 * (dux_dz + duz_dx)
    eyz = 0.5 * (duy_dz + duz_dy)
    tr = exx + eyy + ezz

    sigma_xx = 2.0 * mu * exx + lmbda * tr
    sigma_yy = 2.0 * mu * eyy + lmbda * tr
    sigma_zz = 2.0 * mu * ezz + lmbda * tr
    sigma_xy = 2.0 * mu * exy
    sigma_xz = 2.0 * mu * exz
    sigma_yz = 2.0 * mu * eyz

    slice_idx = 0  # front face y = 0
    X, Z = np.meshgrid(x, z, indexing="ij")
    strain_slice = ezz[:, slice_idx, :]
    stress_slice = sigma_zz[:, slice_idx, :]

    dirs_strain = np.zeros((x.size, z.size, 2), dtype=float)
    dirs_stress = np.zeros((x.size, z.size, 2), dtype=float)
    for i in range(x.size):
        for k in range(z.size):
            eps_tensor = np.array(
                [
                    [exx[i, slice_idx, k], exy[i, slice_idx, k], exz[i, slice_idx, k]],
                    [exy[i, slice_idx, k], eyy[i, slice_idx, k], eyz[i, slice_idx, k]],
                    [exz[i, slice_idx, k], eyz[i, slice_idx, k], ezz[i, slice_idx, k]],
                ]
            )
            sig_tensor = np.array(
                [
                    [sigma_xx[i, slice_idx, k], sigma_xy[i, slice_idx, k], sigma_xz[i, slice_idx, k]],
                    [sigma_xy[i, slice_idx, k], sigma_yy[i, slice_idx, k], sigma_yz[i, slice_idx, k]],
                    [sigma_xz[i, slice_idx, k], sigma_yz[i, slice_idx, k], sigma_zz[i, slice_idx, k]],
                ]
            )
            dirs_strain[i, k, :] = _dominant_projected_direction(eps_tensor)
            dirs_stress[i, k, :] = _dominant_projected_direction(sig_tensor)

    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.8), sharex=True, sharey=True)

    levels_stress = np.linspace(stress_slice.min(), stress_slice.max(), 10)
    levels_strain = np.linspace(strain_slice.min(), strain_slice.max(), 10)

    cf0 = axes[0].contourf(X, Z, stress_slice, levels=levels_stress, cmap="coolwarm")
    axes[0].quiver(
        X, Z, dirs_stress[..., 0], dirs_stress[..., 1],
        color="black", angles="xy", scale_units="xy", scale=8.0, width=0.004
    )
    axes[0].set_title(r"Stress: $\sigma_{zz}$ + dominant direction")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("z")

    cf1 = axes[1].contourf(X, Z, strain_slice, levels=levels_strain, cmap="viridis")
    axes[1].quiver(
        X, Z, dirs_strain[..., 0], dirs_strain[..., 1],
        color="white", angles="xy", scale_units="xy", scale=8.0, width=0.004
    )
    axes[1].set_title(r"Strain: $\varepsilon_{zz}$ + dominant direction")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("z")

    for ax in axes:
        ax.set_aspect("equal")

    cbar0 = fig.colorbar(cf0, ax=axes[0], shrink=0.88, pad=0.03)
    cbar0.set_label(r"$\sigma_{zz}$")
    cbar1 = fig.colorbar(cf1, ax=axes[1], shrink=0.88, pad=0.03)
    cbar1.set_label(r"$\varepsilon_{zz}$")
    fig.suptitle("Representative stress/strain view on the front face", y=1.02)
    fig.subplots_adjust(left=0.08, right=0.95, bottom=0.12, top=0.82, wspace=0.28)
    save_figure(fig, FIG_DIR / "stage_summary_stress_strain_glyphs.pdf")
    plt.close(fig)


def _compute_linear_stress_fields_from_h5(path, E, nu):
    x, y, z, u = build_structured_displacement_grid(path)
    mu = E / (2.0 * (1.0 + nu))
    lmbda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))

    dux_dx = np.gradient(u[..., 0], x, axis=0, edge_order=2)
    dux_dy = np.gradient(u[..., 0], y, axis=1, edge_order=2)
    dux_dz = np.gradient(u[..., 0], z, axis=2, edge_order=2)
    duy_dx = np.gradient(u[..., 1], x, axis=0, edge_order=2)
    duy_dy = np.gradient(u[..., 1], y, axis=1, edge_order=2)
    duy_dz = np.gradient(u[..., 1], z, axis=2, edge_order=2)
    duz_dx = np.gradient(u[..., 2], x, axis=0, edge_order=2)
    duz_dy = np.gradient(u[..., 2], y, axis=1, edge_order=2)
    duz_dz = np.gradient(u[..., 2], z, axis=2, edge_order=2)

    exx = dux_dx
    eyy = duy_dy
    ezz = duz_dz
    exy = 0.5 * (dux_dy + duy_dx)
    exz = 0.5 * (dux_dz + duz_dx)
    eyz = 0.5 * (duy_dz + duz_dy)
    tr = exx + eyy + ezz

    sigma_xx = 2.0 * mu * exx + lmbda * tr
    sigma_yy = 2.0 * mu * eyy + lmbda * tr
    sigma_zz = 2.0 * mu * ezz + lmbda * tr
    sigma_xy = 2.0 * mu * exy
    sigma_xz = 2.0 * mu * exz
    sigma_yz = 2.0 * mu * eyz

    von_mises = np.sqrt(
        0.5
        * (
            (sigma_xx - sigma_yy) ** 2
            + (sigma_yy - sigma_zz) ** 2
            + (sigma_zz - sigma_xx) ** 2
            + 6.0 * (sigma_xy**2 + sigma_xz**2 + sigma_yz**2)
        )
    )
    return x, y, z, von_mises, sigma_zz


def make_deformed_von_mises_figure():
    E = 40.0
    nu = 0.3
    cases = [
        ("Baseline", DATA_DIR / "output_u_monolithic_baseline_3x3x3_petsc_block_lu.h5"),
        ("Aggressive", DATA_DIR / "output_u_monolithic_aggressive_3x3x3_petsc_block_lu.h5"),
    ]

    panels = []
    vmin = np.inf
    vmax = -np.inf
    for title, path in cases:
        x, y, z, u = build_structured_displacement_grid(path)
        _, _, _, von_mises, _ = _compute_linear_stress_fields_from_h5(path, E, nu)
        iy = 0
        X, Z = np.meshgrid(x, z, indexing="ij")
        Xd = X + u[:, iy, :, 0]
        Zd = Z + u[:, iy, :, 2]
        vm = von_mises[:, iy, :]
        panels.append((title, X, Z, Xd, Zd, vm))
        vmin = min(vmin, float(vm.min()))
        vmax = max(vmax, float(vm.max()))

    levels = np.linspace(vmin, vmax, 12)
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.8), sharex=False, sharey=False)
    for ax, (title, X, Z, Xd, Zd, vm) in zip(axes, panels):
        cf = ax.contourf(Xd, Zd, vm, levels=levels, cmap="magma")
        ax.plot([X.min(), X.max(), X.max(), X.min(), X.min()],
                [Z.min(), Z.min(), Z.max(), Z.max(), Z.min()],
                linestyle="--", color="#6b7280", linewidth=1.1, label="Undeformed outline")
        for i in range(Xd.shape[0]):
            ax.plot(Xd[i, :], Zd[i, :], color="white", linewidth=0.45, alpha=0.55)
        for k in range(Xd.shape[1]):
            ax.plot(Xd[:, k], Zd[:, k], color="white", linewidth=0.45, alpha=0.55)
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("z")
        ax.set_aspect("equal")
        ax.legend(frameon=True, loc="lower left")

    cbar = fig.colorbar(cf, ax=axes, shrink=0.92, pad=0.03)
    cbar.set_label("von Mises stress")
    fig.suptitle("Deformed front-face contour with von Mises stress overlay", y=1.02)
    fig.subplots_adjust(left=0.07, right=0.93, bottom=0.13, top=0.82, wspace=0.24)
    save_figure(fig, FIG_DIR / "stage_summary_deformed_von_mises.pdf")
    plt.close(fig)


def make_phi_transport_consistency_figure():
    snapshots = [("040", 0.04), ("080", 0.08), ("120", 0.12)]
    phi_profiles = []
    top_profiles = []

    for tag, load in snapshots:
        phi_path = DATA_DIR / f"output_phi_monolithic_aggressive_snapshot_load{tag}.h5"
        u_path = DATA_DIR / f"output_u_monolithic_aggressive_snapshot_load{tag}.h5"

        x_phi, y_phi, z_phi, phi = build_structured_scalar_grid(phi_path)
        x_u, y_u, z_u, u = build_structured_displacement_grid(u_path)

        ix_mid = x_phi.size // 2
        iy_front = 0
        phi_profiles.append((load, z_phi.copy(), phi[ix_mid, iy_front, :].copy()))

        top_x = x_u.copy()
        top_z = z_u[-1] + u[:, iy_front, -1, 2]
        top_profiles.append((load, top_x, top_z))

    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.7))
    for load, z_ref, phi_line in phi_profiles:
        axes[0].plot(z_ref, phi_line, marker="o", linewidth=1.8, label=f"Load = {load:.2f}")
    axes[0].set_title(r"Pull-back $\phi$ profile in the reference frame")
    axes[0].set_xlabel("Reference vertical coordinate $Z$")
    axes[0].set_ylabel(r"$\widehat{\phi}(X)$")
    axes[0].legend(frameon=True)

    for load, x_top, z_top in top_profiles:
        axes[1].plot(x_top, z_top, marker="s", linewidth=1.8, label=f"Load = {load:.2f}")
    axes[1].set_title("Transported top surface in the current frame")
    axes[1].set_xlabel("Front-face coordinate x")
    axes[1].set_ylabel(r"Current top position $z = 1 + u_z$")
    axes[1].legend(frameon=True)

    fig.suptitle("Why the pull-back SDF can stay fixed while the current interface moves", y=1.03)
    fig.tight_layout()
    fig.subplots_adjust(top=0.80, wspace=0.28)
    save_figure(fig, FIG_DIR / "stage_summary_phi_transport_consistency.pdf")
    plt.close(fig)


def make_stress_snapshots_figure():
    E = 40.0
    nu = 0.3
    loads = [("040", 0.04), ("080", 0.08), ("120", 0.12)]
    slices = []
    vmin = np.inf
    vmax = -np.inf
    for tag, load in loads:
        path = DATA_DIR / f"output_u_monolithic_aggressive_snapshot_load{tag}.h5"
        x, y, z, von_mises, _ = _compute_linear_stress_fields_from_h5(path, E, nu)
        slice_idx = 0
        vm_slice = von_mises[:, slice_idx, :]
        slices.append((load, x, z, vm_slice))
        vmin = min(vmin, float(vm_slice.min()))
        vmax = max(vmax, float(vm_slice.max()))

    fig, axes = plt.subplots(1, 3, figsize=(9.4, 3.3), sharex=True, sharey=True)
    for ax, (load, x, z, vm_slice) in zip(axes, slices):
        X, Z = np.meshgrid(x, z, indexing="ij")
        cf = ax.contourf(X, Z, vm_slice, levels=np.linspace(vmin, vmax, 12), cmap="magma")
        ax.set_title(f"Load = {load:.2f}")
        ax.set_xlabel("x")
        ax.set_ylabel("z")
        ax.set_aspect("equal")
    cbar = fig.colorbar(cf, ax=axes, shrink=0.92, pad=0.03)
    cbar.set_label("von Mises stress")
    fig.suptitle("Stress snapshots along the aggressive load path", y=1.03)
    fig.subplots_adjust(left=0.06, right=0.93, bottom=0.16, top=0.80, wspace=0.22)
    save_figure(fig, FIG_DIR / "stage_summary_stress_snapshots.pdf")
    plt.close(fig)


def main():
    ensure_dirs()
    configure_style()
    make_benchmark_schematic()
    make_iterations_figure()
    make_response_figure(
        "reaction_norm",
        "Reaction norm",
        "stage_summary_reaction_load_curve.pdf",
        "Load-response curves",
    )
    make_response_figure(
        "max_penetration",
        "Maximum penetration",
        "stage_summary_penetration_load_curve.pdf",
        "Load-penetration curves",
    )
    make_linear_iterations_figure()
    make_field_comparison_figure()
    make_top_profile_mode_comparison()
    make_deformed_von_mises_figure()
    make_phi_transport_consistency_figure()
    make_stress_strain_glyph_figure()
    make_stress_snapshots_figure()


if __name__ == "__main__":
    main()
