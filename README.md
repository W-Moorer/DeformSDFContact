# DeformSDFContact

Minimal PETSc-first research scaffold for flexible contact and finite-element experiments on the current WSL2 Ubuntu 22.04 machine.

This repository is intentionally **not** in the business-implementation phase yet.
It currently contains:

- a clean-environment launcher for WSL2 + PETSc/DOLFINx
- a small smoke-test suite that validates the PETSc-first stack
- a placeholder `src/` package for future implementation

## Current Environment

Validated on the current machine with:

- WSL2 Ubuntu 22.04.5
- Python 3.10.12
- OpenMPI 4.1.2
- PETSc 3.15.5
- petsc4py 3.15.1
- DOLFINx 0.3.0

These are primarily **system packages** provided by Ubuntu Jammy. They are not managed by this repository.

## Why The Clean Env Launcher Exists

In the default WSLg shell environment, `mpi4py`, `petsc4py`, and `dolfinx` can hang during import/initialization because of inherited graphics-related environment variables.

Use the wrapper below for every Python, pytest, and MPI command related to this project:

```bash
./scripts/run_clean_env.sh <command> [args...]
```

The wrapper currently:

- unsets `DISPLAY`
- unsets `WAYLAND_DISPLAY`
- unsets `XDG_RUNTIME_DIR`
- unsets `LD_PRELOAD`
- clears inherited `OMPI_MCA_*`
- sets `HWLOC_COMPONENTS=-gl`
- preserves `PATH`
- enables `mpirun` under the current root-based WSL session

## Repository Layout

```text
.
|-- docs/
|   `-- design/
|-- paper/
|-- pullback_sdf_contact/
|-- scripts/
|   `-- run_clean_env.sh
|-- src/
|   `-- deformsdfcontact/
|-- tests/
|   |-- smoke/
|   `-- unit/
|-- pyproject.toml
`-- README.md
```

`pullback_sdf_contact/` contains existing project work and experiments.

`src/deformsdfcontact/` is only a placeholder package right now. No contact, pull-back SDF, predictor, or monolithic solver implementation has been added there yet.

## Running Smoke Tests

Run the full smoke-test set with:

```bash
./scripts/run_clean_env.sh pytest tests/smoke
```

Run the extracted kinematics unit tests with:

```bash
./scripts/run_clean_env.sh pytest tests/unit/test_kinematics.py
```

Run the extracted materials unit tests with:

```bash
./scripts/run_clean_env.sh pytest tests/unit/test_materials.py
```

Run tests individually with:

```bash
./scripts/run_clean_env.sh python3 tests/smoke/test_petsc4py_import.py
./scripts/run_clean_env.sh python3 tests/smoke/test_petsc_snes_custom.py
./scripts/run_clean_env.sh python3 tests/smoke/test_dolfinx_poisson.py
./scripts/run_clean_env.sh python3 tests/smoke/test_dolfinx_block_nest.py
```

Current smoke-test goals:

- `test_petsc4py_import.py`: verify `mpi4py`/`petsc4py` import and PETSc runtime visibility
- `test_petsc_snes_custom.py`: verify direct custom residual/Jacobian control through `petsc4py.PETSc.SNES`
- `test_dolfinx_poisson.py`: verify minimal DOLFINx 0.3.0 Poisson assembly and PETSc solve
- `test_dolfinx_block_nest.py`: verify DOLFINx 0.3.0 block/nest assembly lands on PETSc matrices

## Current DOLFINx 0.3.0 Limits

The current environment is suitable for smoke tests and early scaffold work, but it should be treated as a **transition environment**, not the long-term research target.

Relevant current limitations:

- no `dolfinx.fem.petsc` Python module in the installed version
- no Python-level `dolfinx.geometry.determine_point_ownership`
- no Python-level `dolfinx.geometry.compute_colliding_cells`
- older PETSc/DOLFINx APIs mean more manual glue for blocked nonlinear workflows

This matters for future work on:

- block Jacobians and custom preconditioners
- custom contact-face assembly
- query-point ownership and collision filtering
- pull-back SDF and custom tangent logic

## Before Business Development

Before starting formal business implementation, the following still needs a deliberate go/no-go decision:

1. Keep using the current `apt`-based DOLFINx 0.3.0 stack for early implementation, or
2. Create a **separate isolated upgraded environment** for the long-term solver work

Current recommendation from the scouting phase:

- keep the current system environment for smoke tests and early repository setup
- plan an isolated upgraded environment before serious block/contact implementation begins

Until that decision is made, this repository should stay focused on:

- environment reproducibility
- smoke tests
- project structure
- documentation of technical constraints
