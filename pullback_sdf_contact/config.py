from dataclasses import dataclass


@dataclass
class MeshConfig:
    Lx: float = 1.0
    Ly: float = 1.0
    Lz: float = 1.0
    nx: int = 8
    ny: int = 8
    nz: int = 8
    band_zmin: float = 0.7


@dataclass
class SolidConfig:
    model: str = "linear_elastic"
    E: float = 1e6
    nu: float = 0.3
    body_force_z: float = 0.0


@dataclass
class SDFConfig:
    degree: int = 2
    beta: float = 1e-6


@dataclass
class SolverConfig:
    mode: str = "staggered"
    max_it: int = 5
    verbose: bool = True
