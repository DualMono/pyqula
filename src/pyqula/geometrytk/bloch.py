import numpy as np
from numba import jit
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ..hamiltonians import Hamiltonian


def bloch_phase(h: "Hamiltonian", d, k):
    """
    Return the Bloch phase for this d vector
    """
    if h.dimensionality == 0:
        return 1.0
    elif h.dimensionality == 1:
        if isinstance(k, (float, int)):
            kp = k
        else:
            kp = k[0]  # extract the first component
        dt = np.array(d)[0]
        kt = np.array([kp])[0]
        return np.exp(1j * dt * kt * np.pi * 2.)
    elif h.dimensionality == 2:
        dt = np.array(d)[0:2]
        kt = np.array(k)[0:2]
        return np.exp(1j * dt.dot(kt) * np.pi * 2.)
    elif h.dimensionality == 3:
        dt = np.array(d)[0:3]
        kt = np.array(k)[0:3]
        return np.exp(1j * dt.dot(kt) * np.pi * 2.)
    else:
        raise ValueError(f"Dimension should be in {{0, 1, 2, 3}} but was {h.dimensionality}")


@jit(nopython=True, fastmath=True)
def bloch_phase_2d(d: np.ndarray, k: np.ndarray) -> np.complex128:
    """
    Return the Bloch phase for this d vector. Numba compiled 2D version
    """
    dot_prod = 0.0
    for i in range(2):
        dot_prod += d[i] * k[i]
    return np.exp(1j * dot_prod * np.pi * 2)
