import numpy as np
from numba import jit


def bloch_phase(self,d,k):
    """
    Return the Bloch phase for this d vector
    """
    if self.dimensionality == 0: return 1.0
    elif self.dimensionality == 1:
      try: kp = k[0] # extract the first component
      except: kp = k # ups, assume that it is a float
      dt = np.array(d)[0]
      kt = np.array([kp])[0]
      return np.exp(1j*dt*kt*np.pi*2.)
    elif self.dimensionality == 2:
      dt = np.array(d)[0:2]
      kt = np.array(k)[0:2]
      return np.exp(1j*dt.dot(kt)*np.pi*2.)
    elif self.dimensionality == 3:
      dt = np.array(d)[0:3]
      kt = np.array(k)[0:3]
      return np.exp(1j*dt.dot(kt)*np.pi*2.)
    else: raise


@jit(nopython=True, fastmath=True)
def bloch_phase_2d(d: np.ndarray, k: np.ndarray) -> np.complex128:
    """
    Return the Bloch phase for this d vector. Numba compiled 2D version
    """
    dot_prod = 0.0
    for i in range(2):
        dot_prod += d[i] * k[i]
    return np.exp(1j * dot_prod * np.pi * 2)
