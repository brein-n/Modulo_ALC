import numpy as np
from dataclasses import dataclass


@dataclass
class PuntoImagen:
    u: float
    v: float


@dataclass
class PuntoReal:
    U: float
    V: float
    W: float


def construir_Q(puntos_imagen: list[PuntoImagen],puntos_reales: list[PuntoReal]) -> np.ndarray:

    if len(puntos_imagen) != 12 or len(puntos_reales) != 12:
        raise ValueError("Ambas listas deben contener exactamente 12 puntos")

    Q = np.zeros((24, 12))

    for i in range(12):
        ui = puntos_imagen[i].u
        vi = puntos_imagen[i].v

        Ui = puntos_reales[i].U
        Vi = puntos_reales[i].V
        Wi = puntos_reales[i].W

        Q[2*i] = [
        Ui, Vi, Wi, 1,
        0, 0, 0, 0,
        -ui*Ui, -ui*Vi, -ui*Wi, -ui
        ]

        Q[2*i + 1] = [
        0, 0, 0, 0,
        Ui, Vi, Wi, 1,
        -vi*Ui, -vi*Vi, -vi*Wi, -vi
        ]

    return Q
