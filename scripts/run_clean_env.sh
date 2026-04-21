#!/usr/bin/env bash
set -euo pipefail

show_help() {
  cat <<'EOF'
Usage:
  scripts/run_clean_env.sh <command> [args...]

Examples:
  scripts/run_clean_env.sh python3 -c "from mpi4py import MPI; print(MPI.COMM_WORLD.Get_size())"
  scripts/run_clean_env.sh pytest -q
  scripts/run_clean_env.sh mpirun -n 2 python3 script.py

This wrapper clears WSLg-related variables that can hang mpi4py/petsc4py/dolfinx
in the current Ubuntu 22.04 environment, sets HWLOC_COMPONENTS=-gl, preserves PATH,
and then execs the requested command in the current working directory.
EOF
}

if [[ "${1-}" == "-h" || "${1-}" == "--help" ]]; then
  show_help
  exit 0
fi

if [[ $# -eq 0 ]]; then
  show_help
  exit 1
fi

# WSLg-related variables can trigger hangs during MPI/PETSc startup.
unset DISPLAY
unset WAYLAND_DISPLAY
unset XDG_RUNTIME_DIR

# Keep the launcher reproducible by dropping inherited preload and Open MPI tweaks.
unset LD_PRELOAD
while IFS='=' read -r name _; do
  unset "$name"
done < <(env | grep '^OMPI_MCA_' || true)

export HWLOC_COMPONENTS=-gl
export PATH="${PATH:-/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin}"

if [[ "$(id -u)" -eq 0 ]]; then
  export OMPI_ALLOW_RUN_AS_ROOT=1
  export OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
fi

if [[ "$1" == "pytest" ]] && ! command -v pytest >/dev/null 2>&1; then
  exec python3 -m pytest "${@:2}"
fi

exec "$@"
