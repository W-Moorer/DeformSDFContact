from dataclasses import dataclass

import numpy as np


@dataclass
class SlaveQuadraturePoint:
    facet_id: int
    X_ref: np.ndarray
    weight: float


def build_slave_quadrature(*args, **kwargs):
    """
    占位函数。
    后续在从面参考界面上生成积分点。
    """
    return []
