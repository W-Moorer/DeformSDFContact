import importlib
import os
import platform

os.environ.setdefault("HWLOC_COMPONENTS", "-gl,-opencl,-cuda")


def check_module(name):
    try:
        module = importlib.import_module(name)
    except Exception as exc:
        print(f"{name}: MISSING ({exc.__class__.__name__}: {exc})", flush=True)
        return None

    print(f"{name}: AVAILABLE", flush=True)
    return module


def main():
    print(f"Python: {platform.python_version()}", flush=True)

    check_module("mpi4py")
    check_module("ufl")
    petsc4py_mod = check_module("petsc4py")
    dolfinx_mod = check_module("dolfinx")
    check_module("basix")

    if dolfinx_mod is not None:
        print(f"dolfinx version: {dolfinx_mod.__version__}", flush=True)

    if petsc4py_mod is not None:
        try:
            from petsc4py import PETSc

            print(f"PETSc version: {PETSc.Sys.getVersion()}", flush=True)
        except Exception as exc:
            print(f"PETSc version: UNAVAILABLE ({exc.__class__.__name__}: {exc})", flush=True)


if __name__ == "__main__":
    main()
